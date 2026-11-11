"""Point mutants of one protein, as fields in the frame of the mutated residue (plan item PP1).

A series of point mutants is the one protein series in which the alignment problem of 3D-QSAR
does not arise: every variant keeps the backbone of the wild type, so all of them are already in
one frame.  Two things follow for a field model of a mutational effect (a change of folding or
binding free energy):

1. **The sample is a change, and the change is local.**  A mutant differs from the wild type in
   one side chain.  The field of the whole protein is dominated by the shared backbone, so the
   kernel between two whole mutants is nearly one whatever was changed.  The sample has to be
   the *difference* of the two fields (:func:`cmfqsar.localenv.difference_field`), which is the
   two side chains with opposite signs, or the local environment of the site.
2. **A new position needs a frame of its own.**  In the frame of the protein the difference
   fields of two mutants at different positions do not overlap, so a model would predict a
   held-out position from nothing.  The backbone of every residue gives a frame by rule: the
   origin at the alpha carbon, one axis along the bond to the nitrogen, the plane through the
   carbonyl carbon.  In this frame the side chains of all residues are superimposed on their
   backbone, the change at one position is comparable with the change at another, and the
   co-field is a map around a generic residue: where added bulk, charge or a hydrogen bond
   raises or lowers the stability.

What one mutant costs.  The side chain is placed by PDBFixer from its template, its hydrogens
are added, and it is relaxed with the Amber ff14SB force field through OpenMM with every other
atom held fixed, so that the mutant and the wild type differ in exactly the atoms of the
mutated residue.  The partial charges of the force field are the electrostatic field
(``charge_model="file"``), which keeps the charges of a protein physical where a Gasteiger
scheme on a thousand atoms would not be.  OpenMM and PDBFixer are optional dependencies, like
ORCA and xtb, and are imported only when a structure is built.

The RDKit molecule of a protein is read from the PDB text with proximity bonding, which
reproduces the bond list of the OpenMM topology exactly, and the bond orders of the standard
residues come from RDKit's templates.  Two repairs are needed on top of that: the template
puts the double bond of a histidine ring on the delta nitrogen whatever its protonation, and
the formal charges of lysine, arginine, carboxylates and termini have to be set from the
valence, because a PDB file carries none.
"""
from __future__ import annotations

import dataclasses
import io
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

from .fields import MoleculeFields
from .localenv import difference_field, environment

#: atom names held fixed when a neighbour's side chain is allowed to move: the backbone and the
#: hydrogens that belong to it
BACKBONE = frozenset(("N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3", "HA", "HA2", "HA3"))

#: atom names left out of the field of a side chain.  The alpha hydrogens are *not* among them:
#: they are what glycine presents where another residue has a side chain, so counting them keeps
#: every residue's side-chain field non-empty and makes a substitution to or from glycine an
#: ordinary sample.  The alpha hydrogen of the wild type and that of the mutant sit in nearly the
#: same place and so nearly cancel in a difference field, which leaves the change itself.
MAIN_CHAIN = frozenset(("N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3"))

AA3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
       "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
       "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
AA1 = {v: k for k, v in AA3.items()}
AA1.update({"HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "ASH": "D", "GLH": "E", "LYN": "K"})
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: residue volume in cubic angstrom (Zamyatnin, Prog. Biophys. Mol. Biol. 24 (1972) 107)
VOLUME = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
          "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
          "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
#: hydropathy index (Kyte & Doolittle, J. Mol. Biol. 157 (1982) 105)
HYDROPATHY = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
              "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
              "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
#: formal charge of the side chain at pH 7
CHARGE = {"R": 1.0, "K": 1.0, "D": -1.0, "E": -1.0}

_MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class Mutation:
    """One substitution: the wild-type residue, its number in the file, the new residue."""
    wild: str
    position: int
    new: str

    @classmethod
    def parse(cls, label: str) -> "Mutation":
        m = _MUTATION.match(label.strip())
        if not m or m.group(1) not in AA3 or m.group(3) not in AA3:
            raise ValueError(f"{label!r} is not a substitution written as V1Q")
        return cls(m.group(1), int(m.group(2)), m.group(3))

    @property
    def label(self) -> str:
        return f"{self.wild}{self.position}{self.new}"

    @property
    def fixer_name(self) -> str:
        """The form PDBFixer takes: VAL-1-GLN."""
        return f"{AA3[self.wild]}-{self.position}-{AA3[self.new]}"


@dataclass
class Structure:
    """A protein with hydrogens and force-field charges, as PDB text plus arrays."""
    name: str
    pdb_block: str
    charges: np.ndarray
    energy_before: float = float("nan")     # kJ/mol, before the side chain was relaxed
    energy_after: float = float("nan")
    #: how many starting turns of the side chain ended with two atoms on top of each other
    collapsed: int = 0

    def save(self, path: str) -> str:
        np.savez_compressed(path, name=self.name, pdb_block=self.pdb_block,
                            charges=np.asarray(self.charges, float),
                            energy=np.array([self.energy_before, self.energy_after,
                                             float(self.collapsed)]))
        return path

    @classmethod
    def load(cls, path: str) -> "Structure":
        with np.load(path, allow_pickle=False) as z:
            e = z["energy"]
            return cls(str(z["name"]), str(z["pdb_block"]), np.asarray(z["charges"], float),
                       float(e[0]), float(e[1]), int(e[2]) if len(e) > 2 else 0)


# --------------------------------------------------------------------------- #
# Building structures: PDBFixer and OpenMM
# --------------------------------------------------------------------------- #

def _openmm():
    try:
        from openmm import app, unit
        from openmm import openmm as mm
        from pdbfixer import PDBFixer
    except ImportError as exc:                       # pragma: no cover - depends on the machine
        raise ImportError("building a mutant needs OpenMM and PDBFixer: "
                          "python -m pip install openmm pdbfixer") from exc
    return app, unit, mm, PDBFixer


_FORCEFIELD = {}


def _forcefield(files: Sequence[str] = ("amber14-all.xml",)):
    app = _openmm()[0]
    key = tuple(files)
    if key not in _FORCEFIELD:
        _FORCEFIELD[key] = app.ForceField(*files)
    return _FORCEFIELD[key]


def _charges_of(system, unit, mm) -> np.ndarray:
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    if not nb:
        raise ValueError("the force field gave no non-bonded force, so there are no charges")
    return np.array([nb[0].getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                     for i in range(system.getNumParticles())], float)


def _pdb_text(app, topology, positions) -> str:
    buf = io.StringIO()
    app.PDBFile.writeFile(topology, positions, buf, keepIds=True)
    return buf.getvalue()


def first_chain(pdb_path: str) -> str:
    """The identifier of the first chain of a PDB file."""
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                return line[21]
    raise ValueError(f"{pdb_path} has no atoms")


def prepare_wild_type(pdb_in: str, pdb_out: str, ph: float = 7.0) -> Structure:
    """Complete the heavy atoms of a structure, add the hydrogens it lacks, remove everything
    that is not protein, and write it.  Every mutant is then built from this file, so that the
    hydrogens the wild type already has are kept and the mutant differs from it only in the
    mutated residue."""
    app, unit, mm, PDBFixer = _openmm()
    f = PDBFixer(filename=pdb_in)
    f.findMissingResidues()
    f.findMissingAtoms()
    f.addMissingAtoms()
    f.removeHeterogens(False)
    f.addMissingHydrogens(ph)
    text = _pdb_text(app, f.topology, f.positions)
    with open(pdb_out, "w") as fh:
        fh.write(text)
    system = _forcefield().createSystem(f.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    return Structure(os.path.splitext(os.path.basename(pdb_out))[0], text, _charges_of(system, unit, mm))


def _free_atoms(top, positions, unit, chain: str, position: int, shell: float) -> List[int]:
    """The atoms allowed to move: the whole mutated residue, and the side chains of the residues
    with any atom within `shell` angstrom of it.

    The mutated residue needs its backbone free because PDBFixer builds the new side chain from
    a template and places it roughly: with the backbone held, the first side-chain atom is
    sometimes left at an angle no force field would allow, and the minimiser cannot repair it.
    The neighbours need their side chains free because a larger residue does not fit into the
    space the smaller one left, and a protein answers such a substitution by moving the side
    chains around it.  Every backbone outside the mutated residue stays fixed, so the fold does
    not drift and the frame of the site is the frame of the wild type.
    """
    xyz = np.array(positions.value_in_unit(unit.angstrom))
    own = [a.index for a in top.atoms()
           if a.residue.chain.id == chain and a.residue.id == str(position)]
    if not own:
        raise ValueError(f"residue {position} of chain {chain} has no atoms")
    free = list(own)
    if shell > 0:
        from scipy.spatial import cKDTree
        near: set = set()
        for group in cKDTree(xyz).query_ball_point(xyz[own], float(shell)):
            near.update(int(i) for i in group)
        for res in top.residues():
            if res.chain.id == chain and res.id == str(position):
                continue
            atoms = list(res.atoms())
            if any(a.index in near for a in atoms):
                free += [a.index for a in atoms if a.name not in BACKBONE]
    return free


def _rotated_positions(xyz: np.ndarray, atoms: Dict[str, int], side: Sequence[int],
                       angle: float) -> np.ndarray:
    """Turn the side chain about the CA-CB bond by `angle` radians (the first torsion).

    Every side-chain atom except CB itself moves, which is what turning that torsion does.
    A residue with no CB (glycine, alanine with only hydrogens beyond it) is left alone.
    """
    if "CA" not in atoms or "CB" not in atoms or not len(side):
        return xyz
    ca, cb = xyz[atoms["CA"]], xyz[atoms["CB"]]
    axis = cb - ca
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return xyz
    axis = axis / norm
    out = xyz.copy()
    moving = [i for i in side if i != atoms["CB"]]
    if not moving:
        return xyz
    v = out[moving] - cb
    c, s = np.cos(angle), np.sin(angle)
    out[moving] = cb + v * c + np.cross(axis, v) * s + np.outer(v @ axis, axis) * (1.0 - c)
    return out


def build_mutant(wt_pdb: str, mutation: Mutation, chain: Optional[str] = None, ph: float = 7.0,
                 steps: int = 1000, tolerance: float = 1.0, platform: str = "CPU",
                 shell: float = 6.0, rotamers: int = 6, threads: int = 1) -> Structure:
    """One point mutant of the prepared wild type, relaxed in place.

    PDBFixer replaces the residue, builds the new side chain from its template and adds its
    hydrogens.  That placement is rough and sometimes leaves the new side chain inside a
    neighbour, which a local minimisation cannot always undo, so the side chain is started from
    `rotamers` evenly spaced turns of its first torsion and the lowest-energy result is kept.
    Each start is relaxed with the force field: the mutated residue and the side chains of its
    neighbours within `shell` angstrom may move (:func:`_free_atoms`), every other atom is
    fixed, for at most `steps` iterations or until the largest force falls below `tolerance`
    kJ/mol/nm.  ``shell=0`` frees the mutated residue alone, ``rotamers<=1`` keeps PDBFixer's
    placement.
    """
    app, unit, mm, PDBFixer = _openmm()
    chain = chain or first_chain(wt_pdb)
    f = PDBFixer(filename=wt_pdb)
    found = [r for r in f.topology.residues() if r.chain.id == chain and r.id == str(mutation.position)]
    if not found:
        raise ValueError(f"residue {mutation.position} of chain {chain} is not in {wt_pdb}")
    if AA1.get(found[0].name) != mutation.wild:
        raise ValueError(f"residue {mutation.position} of {wt_pdb} is {found[0].name}, "
                         f"not {AA3[mutation.wild]} as {mutation.label} says")
    f.applyMutations([mutation.fixer_name], chain)
    f.findMissingResidues()
    f.findMissingAtoms()
    f.addMissingAtoms()
    f.addMissingHydrogens(ph)
    top, pos = f.topology, f.positions
    system = _forcefield().createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None)
    free = set(_free_atoms(top, pos, unit, chain, mutation.position, shell))
    for atom in top.atoms():
        if atom.index not in free:
            system.setParticleMass(atom.index, 0.0)
    own = {a.name: a.index for a in top.atoms()
           if a.residue.chain.id == chain and a.residue.id == str(mutation.position)}
    side = [i for n, i in own.items() if n not in BACKBONE]
    # one thread per context: a build of several thousand mutants runs as several processes
    # side by side, and without this each of them takes a thread for every core, which makes
    # every one of them about ten times slower
    properties = {"Threads": str(int(threads))} if platform == "CPU" and threads else None
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform),
                     properties) if properties else \
        mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform))
    start = np.array(pos.value_in_unit(unit.angstrom))
    turns = [0.0] if int(rotamers) <= 1 else list(np.linspace(0.0, 2.0 * np.pi, int(rotamers),
                                                              endpoint=False))
    e0, best, collapsed = float("nan"), None, 0
    for k, angle in enumerate(turns):
        xyz = start if angle == 0.0 else _rotated_positions(start, own, side, float(angle))
        ctx.setPositions(xyz * 0.1)              # angstrom -> nanometre
        energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if k == 0:
            e0 = energy
        mm.LocalEnergyMinimizer.minimize(ctx, tolerance, int(steps))
        state = ctx.getState(getEnergy=True, getPositions=True)
        e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        got = np.array(state.getPositions().value_in_unit(unit.angstrom))
        if closest_approach(got) < MIN_SEPARATION:
            collapsed += 1                        # see the note in `closest_approach`
            continue
        if best is None or e1 < best[0]:
            best = (e1, state.getPositions())
    del ctx
    if best is None:
        raise ValueError(f"{mutation.label}: every one of the {len(turns)} starting turns of the "
                         f"side chain ended with two atoms on top of each other, so there is no "
                         f"structure to keep")
    text = _pdb_text(app, top, best[1])
    return Structure(mutation.label, text, _charges_of(system, unit, mm), e0, best[0], collapsed)


#: the shortest distance two atoms may have, in angstrom.  A bond from carbon to hydrogen is
#: 1.09, and the shortest bond in a protein is about 0.96 (the hydroxyl of a serine), so nothing
#: below this is a structure.
MIN_SEPARATION = 0.7


def closest_approach(coords: np.ndarray) -> float:
    """The distance between the two closest atoms, in angstrom.

    This is the check that says whether a minimised structure is a structure at all.  In the
    Amber force fields the hydrogen atoms bound to nitrogen and oxygen carry no repulsion
    radius, so a hydrogen and an acceptor of the opposite charge attract each other with
    nothing to stop them: the pair falls together, the Coulomb energy runs away towards minus
    infinity, and a rule that keeps the lowest energy keeps exactly that result.  A structure
    with two atoms closer than :data:`MIN_SEPARATION` is thrown away for this reason.
    """
    from scipy.spatial import cKDTree
    xyz = np.asarray(coords, float)
    if len(xyz) < 2:
        return float("inf")
    d, _i = cKDTree(xyz).query(xyz, k=2)
    return float(d[:, 1].min())


def side_chain_geometry(pdb_block: str, position: int) -> Dict[str, float]:
    """The bond length CA-CB and the angle N-CA-CB of one residue, which say whether the side
    chain was placed properly: about 1.53 angstrom and 110 degrees in a sound structure."""
    at: Dict[str, np.ndarray] = {}
    for line in pdb_block.splitlines():
        if line.startswith("ATOM") and line[22:26].strip() == str(position):
            at[line[12:16].strip()] = np.array([float(line[30 + 8 * k:38 + 8 * k]) for k in range(3)])
    if not {"N", "CA", "CB"} <= set(at):
        return {}
    v1, v2 = at["N"] - at["CA"], at["CB"] - at["CA"]
    cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return {"ca_cb": float(np.linalg.norm(v2)),
            "n_ca_cb": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))}


# --------------------------------------------------------------------------- #
# The RDKit molecule of a protein
# --------------------------------------------------------------------------- #

def _fix_histidines(mol: Chem.RWMol) -> None:
    """Put the ring double bond on the unprotonated nitrogen of every histidine."""
    rings: Dict[Tuple[str, int, str], Dict[str, int]] = {}
    for a in mol.GetAtoms():
        info = a.GetPDBResidueInfo()
        if info is None or info.GetResidueName().strip() not in ("HIS", "HID", "HIE", "HIP"):
            continue
        key = (info.GetChainId(), info.GetResidueNumber(), info.GetInsertionCode())
        rings.setdefault(key, {})[info.GetName().strip()] = a.GetIdx()
    for atoms in rings.values():
        if not all(k in atoms for k in ("ND1", "CE1", "NE2")):
            continue
        nd1, ce1, ne2 = atoms["ND1"], atoms["CE1"], atoms["NE2"]
        h_on = {i: any(n.GetAtomicNum() == 1 for n in mol.GetAtomWithIdx(i).GetNeighbors())
                for i in (nd1, ne2)}
        b1, b2 = mol.GetBondBetweenAtoms(nd1, ce1), mol.GetBondBetweenAtoms(ce1, ne2)
        if b1 is None or b2 is None:
            continue
        if h_on[nd1] and not h_on[ne2]:            # HID: the double bond goes to NE2
            b1.SetBondType(Chem.BondType.SINGLE)
            b2.SetBondType(Chem.BondType.DOUBLE)
        else:                                      # HIE and HIP: on ND1
            b1.SetBondType(Chem.BondType.DOUBLE)
            b2.SetBondType(Chem.BondType.SINGLE)


def protein_from_pdb(pdb_block: str, charges: Optional[Sequence[float]] = None) -> Chem.Mol:
    """An RDKit molecule of a protein with explicit hydrogens, formal charges from the valence
    and, when given, the partial charges in the per-atom property ``partial_charge`` that
    ``charge_model="file"`` reads."""
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False, proximityBonding=True)
    if mol is None:
        raise ValueError("RDKit could not read the PDB text")
    rw = Chem.RWMol(mol)
    _fix_histidines(rw)
    for a in rw.GetAtoms():
        a.SetNoImplicit(True)
        a.SetFormalCharge(0)
        v = sum(b.GetBondTypeAsDouble() for b in a.GetBonds())
        if a.GetSymbol() == "N" and v == 4:
            a.SetFormalCharge(1)
        elif a.GetSymbol() == "O" and v == 1:
            a.SetFormalCharge(-1)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    if charges is not None:
        q = np.asarray(charges, float).reshape(-1)
        if len(q) != mol.GetNumAtoms():
            raise ValueError(f"{len(q)} charges for {mol.GetNumAtoms()} atoms")
        for a, c in zip(mol.GetAtoms(), q):
            a.SetDoubleProp("partial_charge", float(c))
    return mol


def protein_of(structure: Structure) -> Chem.Mol:
    return protein_from_pdb(structure.pdb_block, structure.charges)


def residue_atoms(mol: Chem.Mol, position: int, chain: Optional[str] = None) -> Dict[str, int]:
    """Atom name -> index for one residue, by its number in the file."""
    out: Dict[str, int] = {}
    for a in mol.GetAtoms():
        info = a.GetPDBResidueInfo()
        if info is None or info.GetResidueNumber() != int(position):
            continue
        if chain is not None and info.GetChainId().strip() != chain.strip():
            continue
        out[info.GetName().strip()] = a.GetIdx()
    if not out:
        raise ValueError(f"no residue {position} in the molecule")
    return out


def side_chain_atoms(mol: Chem.Mol, position: int, chain: Optional[str] = None) -> np.ndarray:
    """The atoms of one residue beyond its main chain, the alpha hydrogens among them
    (:data:`MAIN_CHAIN` says which names are left out and why)."""
    atoms = residue_atoms(mol, position, chain)
    return np.array(sorted(i for n, i in atoms.items() if n not in MAIN_CHAIN), int)


def residue_name(mol: Chem.Mol, position: int) -> str:
    atoms = residue_atoms(mol, position)
    return mol.GetAtomWithIdx(next(iter(atoms.values()))).GetPDBResidueInfo().GetResidueName().strip()


# --------------------------------------------------------------------------- #
# The frame of a residue
# --------------------------------------------------------------------------- #

def backbone_frame(coords: np.ndarray, atoms: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of the frame of one residue: the origin at CA, the first axis along
    CA->N, the second in the plane of N, CA and C towards C, the third their cross product.
    Coordinates are taken into the frame as ``(x - origin) @ R.T``."""
    ca, n, c = (np.asarray(coords[atoms[k]], float) for k in ("CA", "N", "C"))
    e1 = n - ca
    e1 /= np.linalg.norm(e1)
    v = c - ca
    e3 = np.cross(e1, v)
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    return ca, np.stack([e1, e2, e3])


def in_residue_frame(fields: MoleculeFields, mol: Chem.Mol, position: int, rho: float = 0.0,
                     keep: Optional[Sequence[int]] = None, name: str = "",
                     chain: Optional[str] = None) -> MoleculeFields:
    """The fields of a protein seen from one residue: translated to its alpha carbon, rotated
    into its backbone frame and, with `rho` > 0, under the Gaussian envelope of
    :func:`cmfqsar.localenv.environment`.  `keep` restricts the atoms, as there."""
    atoms = residue_atoms(mol, position, chain)
    origin, rot = backbone_frame(fields.coords, atoms)
    env = environment(fields, atoms["CA"], rho=float(rho), keep=keep, name=name)
    coords = np.asarray(env.coords, float) @ rot.T
    return dataclasses.replace(env, coords=coords)


def side_chain_field(fields: MoleculeFields, mol: Chem.Mol, position: int, rho: float = 0.0,
                     name: str = "", chain: Optional[str] = None) -> MoleculeFields:
    """The side chain of one residue alone, in the frame of its backbone.  With `rho` > 0 its
    atoms carry the envelope factors of the environment they are part of, so that a context
    and a side chain under the same `rho` add up to the environment of the site."""
    keep = side_chain_atoms(mol, position, chain)
    if len(keep) == 0:
        raise ValueError(f"residue {position} has no atom beyond its main chain, so it has no "
                         f"side-chain field; a complete structure always has at least one")
    out = in_residue_frame(fields, mol, position, rho, keep=keep, name=name, chain=chain)
    return _without_centre(out)


def _without_centre(env: MoleculeFields) -> MoleculeFields:
    """`environment` puts the centre atom first whether or not it was asked for; drop it when
    it is not one of the kept atoms (its weight row is kept at zero to keep the field valid)."""
    w = np.asarray(env.weights, float).copy()
    w[0] = 0.0
    return dataclasses.replace(env, weights=w)


def site_context(fields: MoleculeFields, mol: Chem.Mol, position: int, rho: float,
                 name: str = "", chain: Optional[str] = None) -> MoleculeFields:
    """The surroundings of one residue without its own side chain: the environment of the
    alpha carbon under the envelope `rho`, with the side-chain atoms of the residue left out.
    It is the same for every mutant at the position, which is what makes it cheap."""
    whole = in_residue_frame(fields, mol, position, rho, name=name, chain=chain)
    own = set(int(i) for i in side_chain_atoms(mol, position, chain))
    keep = np.array([k for k, src in enumerate(whole.member_atom_map) if int(src) not in own], int)
    return _subset(whole, keep)


def _subset(m: MoleculeFields, keep: np.ndarray) -> MoleculeFields:
    back = -np.ones(m.n_atoms, int)
    back[keep] = np.arange(len(keep))
    bonds = np.asarray(m.bonds, int).reshape(-1, 2)
    if len(bonds):
        ok = (back[bonds[:, 0]] >= 0) & (back[bonds[:, 1]] >= 0)
        bonds = np.stack([back[bonds[ok, 0]], back[bonds[ok, 1]]], axis=1)
    charges = (np.asarray(m.charges, float)[keep] if m.charges is not None
               and len(m.charges) == m.n_atoms else np.zeros(len(keep)))
    return dataclasses.replace(
        m, coords=np.asarray(m.coords, float)[keep], weights=np.asarray(m.weights, float)[keep],
        radii=np.asarray(m.radii, float)[keep], symbols=[m.symbols[k] for k in keep],
        atom_types=[m.atom_types[k] for k in keep], bonds=bonds, charges=charges,
        member_atom_map=np.asarray(m.member_atom_map, int)[keep])


def site_difference(mutant: MoleculeFields, mutant_mol: Chem.Mol, wild: MoleculeFields,
                    wild_mol: Chem.Mol, position: int, rho: float = 0.0, name: str = "",
                    chain: Optional[str] = None) -> MoleculeFields:
    """The change at the site: the side chain of the mutant minus the side chain of the wild
    type, both in the frame of the residue's backbone.  Everything else cancels exactly when
    the mutant was built by :func:`build_mutant`; :func:`frame_shift` checks that."""
    a = side_chain_field(mutant, mutant_mol, position, rho, chain=chain)
    b = side_chain_field(wild, wild_mol, position, rho, chain=chain)
    return difference_field(a, b, name=name)


def frame_shift(mutant_mol: Chem.Mol, wild_mol: Chem.Mol, position: int,
                chain: Optional[str] = None, backbone_only: bool = True) -> float:
    """The largest displacement, in angstrom, of an atom outside the mutated residue between
    the wild type and the mutant, matched by residue number and atom name.

    With `backbone_only`, which is the default, only backbone atoms are compared, and the
    answer is zero for a structure built by :func:`build_mutant`: every backbone outside the
    mutated residue is held fixed, which is what keeps the fold and the frame of the site the
    same in all variants.  Without it, the side chains of the neighbours count as well, and
    the answer says how far they moved to make room, which the build allows on purpose.

    Not a number comes back when the two structures do not hold the same atoms outside the
    mutated residue, since then there is nothing to compare.
    """
    def table(mol):
        out = {}
        xyz = mol.GetConformer().GetPositions()
        for a in mol.GetAtoms():
            info = a.GetPDBResidueInfo()
            if info.GetResidueNumber() == int(position):
                continue
            name = info.GetName().strip()
            if backbone_only and name not in BACKBONE:
                continue
            out[(info.GetChainId(), info.GetResidueNumber(), name)] = xyz[a.GetIdx()]
        return out
    ta, tb = table(mutant_mol), table(wild_mol)
    shared = set(ta) & set(tb)
    if not shared or len(shared) != len(ta) or len(shared) != len(tb):
        return float("nan")
    return float(max(np.linalg.norm(ta[k] - tb[k]) for k in shared))


# --------------------------------------------------------------------------- #
# References that need no structure
# --------------------------------------------------------------------------- #

def substitution_features(mutations: Sequence[Mutation]) -> Tuple[np.ndarray, List[str]]:
    """Columns a linear regression on the substitution alone can use: the change of residue
    volume, of hydropathy and of formal charge, then the one-hot codes of the wild-type and of
    the new residue (20 + 20 columns)."""
    rows, names = [], []
    names = ["d_volume", "d_hydropathy", "d_charge"] + [f"wild_{a}" for a in AMINO_ACIDS] \
        + [f"new_{a}" for a in AMINO_ACIDS]
    for m in mutations:
        row = [VOLUME[m.new] - VOLUME[m.wild], HYDROPATHY[m.new] - HYDROPATHY[m.wild],
               CHARGE.get(m.new, 0.0) - CHARGE.get(m.wild, 0.0)]
        row += [1.0 if a == m.wild else 0.0 for a in AMINO_ACIDS]
        row += [1.0 if a == m.new else 0.0 for a in AMINO_ACIDS]
        rows.append(row)
    return np.array(rows, float), names
