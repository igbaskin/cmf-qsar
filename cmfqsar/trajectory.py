"""
A molecular-dynamics trajectory as a conformational ensemble (plan item D7).

The ensemble machinery of this program takes a molecule carrying several conformations and a
weight for each, and builds the mean field over them (the mean-map kernel of
`cmfqsar/fields.py`).  Where those conformations come from is of no concern to the kernel.  So far
they have come from the conformer generator, which samples the molecule on its own, in no
surroundings.  A trajectory of the ligand inside its pocket samples it where it actually sits,
with the protein holding it, and this module turns such a trajectory into exactly the object the
ensemble machinery already takes: one RDKit molecule, a few dozen conformations, one weight each.

The chain is five steps, and each is a decision worth stating.

1. **Read.** ``mdtraj`` or ``MDAnalysis``, whichever is installed, both optional dependencies in
   the way ORCA and xtb are external programs.  XTC, DCD, NetCDF, TRR and multi-model PDB all
   work, because the readers handle them; this module does not parse any format itself.
2. **Strip.** Water, the common ion residues and anything else the caller names are dropped.  A
   trajectory is mostly solvent, and solvent is not part of the ligand's field.
3. **Superimpose.** When the topology holds a receptor, every frame is superimposed on the
   **backbone of the binding site** -- the residues near the ligand in the first frame -- so that
   what is left in the ligand's coordinates is its motion *relative to the pocket*.  That is the
   quantity a field model of a series needs, because it is what differs between two ligands in
   the same site.  Superimposing on the ligand's own atoms instead would remove exactly that and
   leave only its internal flexibility; it is what happens when there is no receptor, and the
   result says which was used.
4. **Subsample.** The kernel of an ensemble of k conformations costs k^2 atom-pair integrals, so
   a hundred thousand frames is not an option and the thinning has to happen here rather than be
   left to the caller.  Frames are clustered by heavy-atom distance with a leader algorithm: the
   first frame starts a cluster, every later frame joins the first cluster whose representative
   is within the cutoff, and starts a new one otherwise.  If that gives more clusters than asked
   for, the cutoff is raised and the pass repeated.  The representative of a cluster is the frame
   that opened it.
5. **Weight.** The weight of a conformation is its **cluster population** divided by the number
   of frames, and nothing else.  It is never a Boltzmann factor of the frame energy.  An
   equilibrated trajectory has already sampled the Boltzmann distribution -- that is what running
   it was for -- so weighting the frames by exp(-E/kT) a second time would square the
   distribution and sharpen the ensemble onto its deepest basin.  The weights are written to the
   molecule property ``conformer_weights``, which is what ``conformer_weighting="given"`` reads.

The force-field charges of the topology, where the format carries them (PSF, PRMTOP, a TPR), are
written as the per-atom property ``q_mm``, so that ``prop:q_mm`` is a field like any other and a
model can use the charges the simulation itself was run with.

**This module is the reader, and the question it was built for is not settled by it.** Plan item
D8 is the measurement: short simulations on a set whose ligands have crystal poses, and the three
comparisons that say whether the trajectory mean field beats the single crystal pose.
``benchmarks/run_trajectory.py`` measures what can be measured without those simulations -- that
the two readers agree, what the thinning costs, and how fast the mean field settles as the number
of representatives grows -- and says plainly that it answers the reader's own questions and not
D8's.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

#: residue names dropped as solvent
SOLVENT_RESIDUES = frozenset({"HOH", "WAT", "SOL", "TIP", "TIP3", "TIP4", "TP3", "T3P", "T4P",
                              "DOD", "D2O", "H2O", "SPC", "SPCE"})
#: residue names dropped as counter-ions
ION_RESIDUES = frozenset({"NA", "NA+", "SOD", "CL", "CL-", "CLA", "K", "K+", "POT", "LI", "CS",
                          "RB", "BR", "IOD", "I", "F", "MG", "ZN", "CA", "CAL", "MN", "FE",
                          "CU", "NI", "CO", "SR", "BA"})
#: residue names taken as protein, for the binding-site backbone
AMINO_ACIDS = frozenset({"ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "CYM", "GLN", "GLU", "GLH",
                         "GLY", "HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "ILE", "LEU",
                         "LYS", "LYN", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
                         "ASH", "ACE", "NME", "NMA"})
#: atom names in a protein file whose first two letters are an element symbol but mean
#: something else: CA is an alpha carbon and not calcium, HG a gamma hydrogen and not mercury
AMBIGUOUS_ATOM_NAMES = frozenset({"CA", "CB", "CD", "CE", "CG", "CH", "CZ", "ND", "NE", "NH",
                                  "NZ", "OD", "OE", "OG", "OH", "SD", "SG", "HD", "HE", "HG",
                                  "HH", "HZ"})
#: the atoms a backbone superposition uses
BACKBONE_ATOMS = ("N", "CA", "C")
#: per-atom property holding the force field's partial charge
CHARGE_PROP = "q_mm"

DEFAULT_MAX_FRAMES = 30
DEFAULT_CUTOFF = 1.0            # angstrom, the leader-clustering radius
DEFAULT_SITE_RADIUS = 8.0       # angstrom, what counts as the binding site


class TrajectoryError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------------------

@dataclass
class Trajectory:
    """One trajectory as plain arrays, whichever reader produced it. Coordinates in angstrom."""
    coords: np.ndarray                     # (n_frames, n_atoms, 3)
    elements: List[str]
    atom_names: List[str]
    residues: List[str]                    # residue name per atom
    resids: np.ndarray                     # residue index per atom
    bonds: List[Tuple[int, int]] = field(default_factory=list)
    charges: Optional[np.ndarray] = None   # per atom, when the topology carries them
    box: Optional[np.ndarray] = None       # (n_frames, 3, 3) cell vectors in angstrom
    engine: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[1])

    def subset(self, idx: Sequence[int]) -> "Trajectory":
        idx = np.asarray(idx, int)
        keep = {int(i): k for k, i in enumerate(idx)}
        bonds = [(keep[i], keep[j]) for i, j in self.bonds if i in keep and j in keep]
        return Trajectory(coords=self.coords[:, idx, :].copy(),
                          elements=[self.elements[i] for i in idx],
                          atom_names=[self.atom_names[i] for i in idx],
                          residues=[self.residues[i] for i in idx],
                          resids=self.resids[idx].copy(), bonds=bonds,
                          charges=None if self.charges is None else self.charges[idx].copy(),
                          box=None if self.box is None else self.box.copy(),
                          engine=self.engine)


def available_engines() -> List[str]:
    """Which of the two readers can be imported here."""
    out = []
    for name in ("mdtraj", "MDAnalysis"):
        try:
            __import__(name)
            out.append("mdtraj" if name == "mdtraj" else "mdanalysis")
        except Exception:                    # noqa: BLE001 - an optional dependency
            pass
    return out


def _load_mdtraj(topology: str, trajectory: Optional[str], stride: int) -> Trajectory:
    import mdtraj

    t = (mdtraj.load(trajectory, top=topology, stride=max(1, int(stride)))
         if trajectory else mdtraj.load(topology, stride=max(1, int(stride))))
    top = t.topology
    elements, names, residues, resids = [], [], [], []
    for a in top.atoms:
        sym = getattr(a.element, "symbol", "") or ""
        elements.append(sym if sym and sym != "VS" else _element_of(str(a.name)))
        names.append(str(a.name))
        residues.append(str(a.residue.name).upper())
        resids.append(int(a.residue.index))
    bonds = [(int(b[0].index), int(b[1].index)) for b in top.bonds]
    box = None
    if getattr(t, "unitcell_vectors", None) is not None:
        box = np.asarray(t.unitcell_vectors, float) * 10.0          # nm to angstrom
    return Trajectory(coords=np.asarray(t.xyz, float) * 10.0,       # nm to angstrom
                      elements=elements, atom_names=names, residues=residues,
                      resids=np.asarray(resids, int), bonds=bonds, charges=None,
                      box=box, engine="mdtraj")


def _load_mdanalysis(topology: str, trajectory: Optional[str], stride: int) -> Trajectory:
    import MDAnalysis

    u = MDAnalysis.Universe(topology, trajectory) if trajectory else MDAnalysis.Universe(topology)
    atoms = u.atoms
    elements = []
    for a in atoms:
        elements.append(_element_of(str(a.name), str(getattr(a, "type", "") or ""),
                                    str(getattr(a, "element", "") or "")))
    names = [str(a.name) for a in atoms]
    residues = [str(a.resname).upper() for a in atoms]
    resids = np.asarray([int(a.resindex) for a in atoms], int)
    try:
        bonds = [(int(b.atoms[0].index), int(b.atoms[1].index)) for b in u.bonds]
    except Exception:                        # noqa: BLE001 - a topology without bonds
        bonds = []
    charges = None
    try:
        q = np.asarray(atoms.charges, float)
        if q.shape == (len(atoms),) and np.isfinite(q).all():
            charges = q
    except Exception:                        # noqa: BLE001 - a topology without charges
        charges = None
    frames, boxes = [], []
    for ts in u.trajectory[::max(1, int(stride))]:
        frames.append(np.asarray(ts.positions, float).copy())
        try:
            boxes.append(np.asarray(ts.triclinic_dimensions, float).copy())
        except Exception:                    # noqa: BLE001 - a trajectory without a cell
            boxes.append(None)
    if not frames:
        raise TrajectoryError("the trajectory holds no frames")
    box = np.stack(boxes) if boxes and all(b is not None for b in boxes) else None
    return Trajectory(coords=np.stack(frames), elements=elements, atom_names=names,
                      residues=residues, resids=resids, bonds=bonds, charges=charges,
                      box=box, engine="mdanalysis")


def load(topology: str, trajectory: Optional[str] = None, stride: int = 1,
         engine: str = "auto") -> Trajectory:
    """Read a topology and (optionally) a trajectory into plain arrays.

    ``engine`` is ``mdtraj``, ``mdanalysis`` or ``auto``.  Under ``auto`` MDAnalysis is preferred
    when the topology is one of the formats that carry force-field charges, because those charges
    are the point of reading them; otherwise mdtraj is, being the faster reader.
    """
    for path in (topology, trajectory):
        if path and not os.path.isfile(path):
            raise TrajectoryError(f"no such file: {path}")
    have = available_engines()
    if not have:
        raise TrajectoryError("reading a trajectory needs mdtraj or MDAnalysis; neither is "
                              "installed. Both are optional: pip install mdtraj, or "
                              "pip install MDAnalysis")
    want = str(engine).lower()
    if want == "auto":
        charged = os.path.splitext(topology)[1].lower() in (".psf", ".prmtop", ".parm7", ".top",
                                                            ".tpr", ".pdbqt", ".mol2")
        order = ["mdanalysis", "mdtraj"] if charged else ["mdtraj", "mdanalysis"]
    elif want in ("mdtraj", "mdanalysis"):
        order = [want]
    else:
        raise TrajectoryError(f"engine must be mdtraj, mdanalysis or auto, not {engine!r}")
    errors = []
    for name in order:
        if name not in have:
            errors.append(f"{name} is not installed")
            continue
        try:
            return (_load_mdtraj if name == "mdtraj" else _load_mdanalysis)(
                topology, trajectory, stride)
        except Exception as exc:             # noqa: BLE001 - try the other reader
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise TrajectoryError("; ".join(errors))


# ----------------------------------------------------------------------------------------------
# selecting
# ----------------------------------------------------------------------------------------------

def strip_solvent(traj: Trajectory, extra: Sequence[str] = ()) -> Tuple[Trajectory, int]:
    """Drop water, counter-ions and any residue named in `extra`; returns (trajectory, dropped)."""
    drop = SOLVENT_RESIDUES | ION_RESIDUES | {str(r).upper() for r in extra}
    keep = [i for i, r in enumerate(traj.residues) if r not in drop]
    if not keep:
        raise TrajectoryError("stripping the solvent left no atoms")
    return traj.subset(keep), traj.n_atoms - len(keep)


def ligand_indices(traj: Trajectory, resname: Optional[str] = None) -> np.ndarray:
    """The atoms of the ligand.

    With `resname` given, the atoms of that residue.  Without it, the largest residue that is
    neither protein nor nucleic nor solvent, which is what a ligand is in almost every prepared
    system; if there is none, every atom, which is the case of a trajectory of the ligand alone.
    """
    if resname:
        idx = [i for i, r in enumerate(traj.residues) if r == str(resname).upper()]
        if not idx:
            have = sorted(set(traj.residues))
            raise TrajectoryError(f"no residue named {resname!r}; the file has {have[:12]}")
        return np.asarray(idx, int)
    counts: Dict[int, int] = {}
    for i, r in enumerate(traj.residues):
        if r in AMINO_ACIDS or r in SOLVENT_RESIDUES or r in ION_RESIDUES:
            continue
        counts[int(traj.resids[i])] = counts.get(int(traj.resids[i]), 0) + 1
    if not counts:
        return np.arange(traj.n_atoms)
    best = max(counts, key=lambda k: counts[k])
    return np.asarray([i for i in range(traj.n_atoms) if int(traj.resids[i]) == best], int)


def site_backbone(traj: Trajectory, ligand: Sequence[int], radius: float = DEFAULT_SITE_RADIUS,
                  frame: int = 0) -> np.ndarray:
    """Backbone atoms of the residues with an atom within `radius` of the ligand in `frame`.

    Empty when the topology holds no protein, which is how the caller learns that there is no
    receptor to superimpose on.
    """
    protein = np.asarray([i for i, r in enumerate(traj.residues) if r in AMINO_ACIDS], int)
    if protein.size == 0:
        return protein
    lig = np.asarray(ligand, int)
    xyz = traj.coords[int(frame)]
    d = np.linalg.norm(xyz[protein][:, None, :] - xyz[lig][None, :, :], axis=2)
    near = protein[d.min(axis=1) <= float(radius)]
    resids = set(int(traj.resids[i]) for i in near)
    return np.asarray([i for i in protein
                       if int(traj.resids[i]) in resids
                       and traj.atom_names[i].upper() in BACKBONE_ATOMS], int)


# ----------------------------------------------------------------------------------------------
# superimposing and thinning
# ----------------------------------------------------------------------------------------------

def make_whole(coords: np.ndarray, group: Sequence[int],
               box: Optional[np.ndarray]) -> np.ndarray:
    """Undo the wrapping inside `group`: every atom moved to the image nearest the first one.

    A trajectory is usually written with every atom wrapped into the cell, which splits any
    molecule that straddles a face: half of it appears a box length away from the other half.
    For a molecule small enough that no two of its atoms are further apart than half the cell --
    a ligand, always -- taking each atom's image nearest the first atom puts it back together.
    """
    coords = np.asarray(coords, float)
    if box is None:
        return coords
    group = np.asarray(group, int)
    out = coords.copy()
    for k in range(coords.shape[0]):
        cell = np.asarray(box[k], float)
        inv = np.linalg.inv(cell)
        anchor = coords[k][group[0]]
        d = coords[k][group] - anchor
        shift = np.round(d @ inv) @ cell
        out[k][group] = coords[k][group] - shift
    return out


def reimage(coords: np.ndarray, group: Sequence[int], anchor: Sequence[int],
            box: Optional[np.ndarray]) -> np.ndarray:
    """Move `group`, as one piece, to the periodic image nearest `anchor`.

    The ligand of a simulation wanders across the cell and is written wrapped, so in some frames
    it appears a box length from the protein it is bound to. Nothing physical has happened: the
    image beside the protein is the same molecule. Without this step those frames look like a
    ligand sitting fifty angstroms away, which is how a periodic-boundary artefact turns into a
    false conclusion about a simulation.
    """
    coords = np.asarray(coords, float)
    if box is None:
        return coords
    group = np.asarray(group, int)
    anchor = np.asarray(anchor, int)
    out = coords.copy()
    for k in range(coords.shape[0]):
        cell = np.asarray(box[k], float)
        inv = np.linalg.inv(cell)
        d = out[k][group].mean(axis=0) - out[k][anchor].mean(axis=0)
        out[k][group] -= np.round(d @ inv) @ cell
    return out


def _kabsch(before: np.ndarray, after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The rigid transform that takes `before` onto `after` (both (N, 3), same order)."""
    cb, ca = before.mean(axis=0), after.mean(axis=0)
    H = (before - cb).T @ (after - ca)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, ca - R @ cb


def superimpose(coords: np.ndarray, fit_atoms: Sequence[int], reference: int = 0,
                target: Optional[np.ndarray] = None) -> np.ndarray:
    """Every frame moved so that `fit_atoms` lie on the target.

    The target is frame `reference` of this trajectory by default, and `target` replaces it with
    positions from somewhere else -- the one structure a whole series was built from, so that
    several trajectories come back in one frame rather than each in its own.

    The whole system is moved, not only the fitted atoms, so what the ligand's coordinates then
    show is its motion relative to whatever was fitted.
    """
    coords = np.asarray(coords, float)
    fit = np.asarray(fit_atoms, int)
    if fit.size < 3:
        raise TrajectoryError("a superposition needs at least three atoms")
    if target is None:
        target = coords[int(reference)][fit]
    else:
        target = np.asarray(target, float)
        if target.shape != (fit.size, 3):
            raise TrajectoryError(f"the target has {target.shape} positions for {fit.size} "
                                  f"fitted atoms")
    out = np.empty_like(coords)
    for k in range(coords.shape[0]):
        R, t = _kabsch(coords[k][fit], target)
        out[k] = coords[k] @ R.T + t
    return out


def rmsd_to(coords: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Root-mean-square distance of every frame from `frame`, where they stand, with no fit."""
    d = np.asarray(coords, float) - np.asarray(frame, float)[None, :, :]
    return np.sqrt((d ** 2).sum(axis=2).mean(axis=1))


def leader_cluster(coords: np.ndarray, cutoff: float = DEFAULT_CUTOFF,
                   max_frames: int = DEFAULT_MAX_FRAMES, grow: float = 1.5,
                   max_passes: int = 40) -> Tuple[np.ndarray, np.ndarray, float]:
    """Thin the frames; returns (representatives, population of each, the radius that was used).

    The leader algorithm in one pass over the frames in their own order: a frame joins the first
    representative within `cutoff` and becomes a representative itself otherwise.  The order is
    the trajectory's, so the result is deterministic and does not depend on a random seed.  When
    a pass produces more representatives than `max_frames`, the cutoff is multiplied by `grow`
    and the pass repeated, so the number asked for is an upper bound that is always met.  The
    radius returned is the one the successful pass used, and every frame is within it of its own
    representative.
    """
    coords = np.asarray(coords, float)
    n = coords.shape[0]
    if n == 0:
        raise TrajectoryError("no frames to cluster")
    cut = float(cutoff)
    for _ in range(int(max_passes)):
        reps: List[int] = []
        counts: List[int] = []
        for k in range(n):
            placed = False
            if reps:
                d = rmsd_to(coords[reps], coords[k])
                j = int(np.argmin(d))
                if d[j] <= cut:
                    counts[j] += 1
                    placed = True
            if not placed:
                reps.append(k)
                counts.append(1)
                if max_frames and len(reps) > int(max_frames):
                    break
        if not max_frames or len(reps) <= int(max_frames):
            return np.asarray(reps, int), np.asarray(counts, int), cut
        cut *= float(grow)
    # the cutoff grew forty times and still did not fit: keep the first max_frames leaders
    return (np.asarray(reps[:int(max_frames)], int),
            np.asarray(counts[:int(max_frames)], int), cut)


# ----------------------------------------------------------------------------------------------
# the molecule
# ----------------------------------------------------------------------------------------------

def _from_topology(traj: Trajectory) -> Chem.Mol:
    """A molecule built from the topology's bonds, with the bond orders worked out afterwards."""
    if not traj.bonds:
        raise TrajectoryError("the topology carries no bonds, so the ligand cannot be built from "
                              "it; give a template molecule with template=")
    rw = Chem.RWMol()
    for sym in traj.elements:
        rw.AddAtom(Chem.Atom(_atomic_number(sym)))
    for i, j in traj.bonds:
        if rw.GetBondBetweenAtoms(int(i), int(j)) is None:
            rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(traj.coords[0]):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    try:
        from rdkit.Chem import rdDetermineBonds
        rdDetermineBonds.DetermineBondOrders(mol, charge=0)
    except Exception as exc:                 # noqa: BLE001 - reported, not fatal
        raise TrajectoryError(f"the bond orders could not be worked out from the topology "
                              f"({type(exc).__name__}: {exc}); give a template molecule with "
                              f"template=")
    Chem.SanitizeMol(mol)
    return mol


def _atomic_number(symbol: str) -> int:
    """The atomic number of an element symbol, with a message that names a symbol it cannot read."""
    s = str(symbol).strip()
    try:
        return int(Chem.GetPeriodicTable().GetAtomicNumber(s.capitalize()))
    except Exception:                        # noqa: BLE001 - RDKit raises on an unknown symbol
        raise TrajectoryError(f"{symbol!r} is not an element symbol; the topology may carry "
                              f"force-field atom types rather than elements")


def _element_of(name: str, atom_type: str = "", element: str = "") -> str:
    """The element of an atom, from whichever of the three the topology filled in.

    A topology does not have to carry elements. PDBQT carries AutoDock types (``OA``, ``HD``), a
    force field carries its own (``CT``, ``HA``), and a PDB may carry only the atom name. Each is
    tried in turn: the leading letters as a two-letter symbol first, then as a one-letter one.
    The names of :data:`AMBIGUOUS_ATOM_NAMES` are the exception, because ``CA`` in a protein file
    is an alpha carbon and not calcium, and ``HG`` is a gamma hydrogen and not mercury.
    """
    table = Chem.GetPeriodicTable()

    def known(sym: str) -> bool:
        try:
            return bool(sym) and int(table.GetAtomicNumber(sym)) > 0
        except Exception:                    # noqa: BLE001 - RDKit raises on an unknown symbol
            return False

    for raw, from_name in ((element, False), (atom_type, False), (name, True)):
        letters = "".join(ch for ch in str(raw or "") if ch.isalpha()).upper()
        if not letters:
            continue
        if not (from_name and letters[:2] in AMBIGUOUS_ATOM_NAMES):
            two = letters[:2].capitalize()
            if len(letters) >= 2 and known(two):
                return two
        if known(letters[0]):
            return letters[0]
    raise TrajectoryError(f"the element of atom {name!r} (type {atom_type!r}) could not be "
                          f"worked out from the topology")


def _template_order(traj: Trajectory, template: Chem.Mol) -> np.ndarray:
    """How the template's atoms map onto the trajectory's, or an error saying why they do not.

    The easy case, and the usual one, is that the two list the same elements in the same order,
    which is what happens when the simulation was set up from the molecule the data set holds.
    Otherwise the trajectory's bonds are used to build a graph and the template is matched onto
    it as a substructure.
    """
    want = [_atomic_number(s) for s in traj.elements]
    have = [a.GetAtomicNum() for a in template.GetAtoms()]
    if len(want) != len(have):
        raise TrajectoryError(f"the template has {len(have)} atoms and the selection {len(want)}; "
                              f"hydrogens are a common cause -- a template needs the same ones "
                              f"the trajectory has")
    if want == have:
        return np.arange(len(want))
    if not traj.bonds:
        raise TrajectoryError("the template lists its atoms in another order and the topology "
                             "has no bonds to match them by")
    rw = Chem.RWMol()
    for z in want:
        rw.AddAtom(Chem.Atom(int(z)))
    for i, j in traj.bonds:
        if rw.GetBondBetweenAtoms(int(i), int(j)) is None:
            rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    graph = rw.GetMol()
    flat = Chem.Mol(template)
    for b in flat.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in flat.GetAtoms():
        a.SetIsAromatic(False)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
    match = graph.GetSubstructMatch(flat, useChirality=False)
    if not match or len(match) != len(want):
        raise TrajectoryError("the template could not be matched onto the trajectory's atoms")
    return np.asarray(match, int)


def to_molecule(traj: Trajectory, frames: Sequence[int], weights: Sequence[float],
                template: Optional[Chem.Mol] = None, name: str = "",
                charge_prop: str = CHARGE_PROP) -> Chem.Mol:
    """One molecule carrying the chosen frames as conformations, with their weights.

    `frames` are indices into the trajectory and `weights` the matching populations, normalised
    here so that they sum to one.  The weights go to the molecule property ``conformer_weights``,
    which ``conformer_weighting="given"`` reads.
    """
    frames = [int(k) for k in frames]
    w = np.asarray(weights, float)
    if len(w) != len(frames):
        raise TrajectoryError("one weight per frame is needed")
    if w.min() < 0 or w.sum() <= 0:
        raise TrajectoryError("the weights must be non-negative and not all zero")
    w = w / w.sum()

    if template is not None:
        order = _template_order(traj, template)
        mol = Chem.Mol(template)
    else:
        order = np.arange(traj.n_atoms)
        mol = _from_topology(traj)
    mol.RemoveAllConformers()
    for k in frames:
        conf = Chem.Conformer(mol.GetNumAtoms())
        xyz = traj.coords[k]
        for a, src in enumerate(order):
            x, y, z = xyz[int(src)]
            conf.SetAtomPosition(a, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)
    mol.SetProp("conformer_weights", ",".join(f"{v:.6g}" for v in w))
    mol.SetProp("n_frames_clustered", str(int(traj.n_frames)))
    if name:
        mol.SetProp("_Name", name)
    if traj.charges is not None:
        q = np.asarray(traj.charges, float)[order]
        for a, v in zip(mol.GetAtoms(), q):
            a.SetDoubleProp(charge_prop, float(v))
        Chem.CreateAtomDoublePropertyList(mol, charge_prop)
    return mol


# ----------------------------------------------------------------------------------------------
# the whole chain
# ----------------------------------------------------------------------------------------------

@dataclass
class Ensemble:
    """What `read_trajectory` returns."""
    mol: Chem.Mol                     # the ligand, one conformation per representative frame
    weights: np.ndarray               # cluster populations, summing to one
    populations: np.ndarray           # the counts themselves
    frames: np.ndarray                # which frames the representatives are
    n_frames: int                     # frames read, after the stride
    stripped: int                     # atoms dropped as solvent or ions
    fitted_on: str                    # "binding-site backbone" or "the ligand itself"
    fit_atoms: int
    cutoff: float                     # the clustering radius actually used
    engine: str

    def describe(self) -> str:
        return (f"{self.n_frames} frames, {self.stripped} solvent and ion atoms dropped, "
                f"superimposed on {self.fitted_on} ({self.fit_atoms} atoms), thinned to "
                f"{len(self.frames)} conformations at {self.cutoff:.2f} A "
                f"(largest cluster {self.populations.max() / max(self.n_frames, 1):.0%} of the "
                f"frames), read by {self.engine}")


def _reference_positions(reference: str, traj: "Trajectory",
                         fit: Sequence[int]) -> np.ndarray:
    """The positions the fitted atoms have in `reference`, checked atom by atom.

    The trajectory lists the protein first, because that is the order a system is built in, so a
    protein atom's index in the trajectory is its index in the reference as well.  The elements
    are compared rather than trusted: a mismatch means the two do not describe the same protein
    and a superposition on it would be meaningless.
    """
    ref = load(reference)
    fit = np.asarray(fit, int)
    if int(fit.max()) >= ref.n_atoms:
        raise TrajectoryError(f"{os.path.basename(reference)} has {ref.n_atoms} atoms and the "
                              f"superposition needs atom {int(fit.max())} of the trajectory; the "
                              f"reference must be the structure the system was built from")
    mine = [traj.elements[i] for i in fit]
    theirs = [ref.elements[i] for i in fit]
    if mine != theirs:
        bad = next(k for k in range(len(mine)) if mine[k] != theirs[k])
        raise TrajectoryError(f"{os.path.basename(reference)} does not list the same atoms as "
                              f"the trajectory: atom {int(fit[bad])} is {theirs[bad]} there and "
                              f"{mine[bad]} here")
    return ref.coords[0][fit]


def read_trajectory(topology: str, trajectory: Optional[str] = None, *,
                    ligand: Optional[str] = None, template: Optional[Chem.Mol] = None,
                    max_frames: int = DEFAULT_MAX_FRAMES, cutoff: float = DEFAULT_CUTOFF,
                    stride: int = 1, site_radius: float = DEFAULT_SITE_RADIUS,
                    extra_solvent: Sequence[str] = (), engine: str = "auto",
                    name: str = "", heavy_only: bool = True,
                    reference: Optional[str] = None) -> Ensemble:
    """A trajectory as a weighted conformational ensemble of its ligand (plan item D7).

    `topology` may be a PDB, PSF, PRMTOP or anything the reader understands; `trajectory` is the
    frames (XTC, DCD, NetCDF, TRR), or None when the topology file holds them itself, as a
    multi-model PDB does.  `ligand` names the residue; without it the largest residue that is
    neither protein nor solvent is taken.  `template` is the ligand as the data set holds it and
    is the reliable way to get the bond orders right; without one they are worked out from the
    topology's bonds, which can fail and then says so.

    `heavy_only` measures the distances between frames on the heavy atoms alone, which is what a
    conformational difference means; the hydrogens still travel with the frames that are kept.

    `reference` is the structure every frame is superimposed on, and it matters whenever several
    trajectories of one series are read.  Without it each trajectory is superimposed on its own
    first frame, which is that run's equilibrated protein: two runs equilibrate to protein
    conformations that differ by a few tenths of an angstrom, so their ligands end up in frames
    that differ by the same rotation and the series loses the common alignment a field kernel
    depends on.  Pass the one structure every run was built from -- the prepared receptor -- and
    every ligand comes back in that structure's frame.  The file must list the protein atoms
    first and in the same order as the trajectory does, which is the case when the systems were
    built from it; the elements are checked and a mismatch is refused.
    """
    traj = load(topology, trajectory, stride=stride, engine=engine)
    n_read = traj.n_frames
    traj, dropped = strip_solvent(traj, extra_solvent)
    lig = ligand_indices(traj, ligand)
    if lig.size == 0:
        raise TrajectoryError("the ligand selection is empty")

    if traj.box is not None:
        # the ligand first made whole, then brought to the image beside the protein; both are
        # undone wrapping and neither moves the molecule in any physical sense
        protein = np.asarray([i for i, r in enumerate(traj.residues) if r in AMINO_ACIDS], int)
        coords = make_whole(traj.coords, lig, traj.box)
        if protein.size >= 3:
            coords = reimage(coords, lig, protein, traj.box)
        traj = Trajectory(coords=coords, elements=traj.elements, atom_names=traj.atom_names,
                          residues=traj.residues, resids=traj.resids, bonds=traj.bonds,
                          charges=traj.charges, box=traj.box, engine=traj.engine)

    fit = site_backbone(traj, lig, radius=site_radius)
    if fit.size >= 3:
        fitted_on = "the binding-site backbone"
    elif reference is not None:
        # a common reference is a frame of the receptor, so there has to be a receptor to fit
        # on. No backbone within the radius means the ligand is not in a site -- either the
        # topology holds no protein, or the ligand left during the simulation -- and fitting on
        # the ligand instead would put this one molecule in a frame of its own.
        raise TrajectoryError(
            f"no protein backbone within {site_radius:g} A of the ligand in the first frame, "
            f"so it is not in a site and cannot be put in the frame of "
            f"{os.path.basename(reference)}; a ligand that left its site is not a bound pose")
    else:
        # no receptor: the ligand is superimposed on its own first pose, which leaves only its
        # internal flexibility and not its motion in a site
        fit = lig
        fitted_on = "the ligand itself"
    target = None
    if reference is not None:
        target = _reference_positions(reference, traj, fit)
    coords = superimpose(traj.coords, fit, target=target)
    if target is not None:
        fitted_on += " of the common reference"
    traj = Trajectory(coords=coords, elements=traj.elements, atom_names=traj.atom_names,
                      residues=traj.residues, resids=traj.resids, bonds=traj.bonds,
                      charges=traj.charges, box=traj.box, engine=traj.engine)

    measure = lig
    if heavy_only:
        heavy = np.asarray([i for i in lig if traj.elements[i].upper() != "H"], int)
        if heavy.size >= 3:
            measure = heavy
    reps, counts, used = leader_cluster(coords[:, measure, :], cutoff=cutoff,
                                        max_frames=max_frames)

    lig_traj = traj.subset(lig)
    mol = to_molecule(lig_traj, reps, counts.astype(float), template=template, name=name)
    return Ensemble(mol=mol, weights=counts / counts.sum(), populations=counts,
                    frames=reps, n_frames=n_read, stripped=dropped, fitted_on=fitted_on,
                    fit_atoms=int(fit.size), cutoff=used, engine=traj.engine)
