"""
Bulk electrolyte simulations, for plan item BT5.

:mod:`cmfqsar.shells` cuts solvation shells out of a bulk trajectory. This module produces the
trajectories it cuts them from, so that the question of item BT5 can be asked: does an ensemble
of shells sampled from a liquid model an electrolyte better than the single optimised complex of
item BT4?

:mod:`cmfqsar.md` does the same service for item D8, a ligand inside a protein. An electrolyte is
a different system and needs a different recipe, so it gets its own module rather than another
mode of that one.

What one run is, and why each choice was made.

**A box of a named formulation.** The input is a salt, the solvents with their weight fractions
and the molality, which is what a conductivity table reports. Weight fractions and molality fix
the number of molecules exactly, with no density needed: the mass of the solvent follows from the
molecule counts and the molar masses, and the salt count follows from the mass. A table that gives
volume fractions or molarity would need a density and so an assumption; this module takes the
first kind and says so, and the realised molality is reported next to the one asked for, because
rounding the ion count to a whole number changes it by a per cent or two.

**Packed with packmol, closed by the barostat.** The starting box is built at a slightly low
density, which packmol can fill without overlaps, and the pressure coupling then closes it to the
liquid's own density. The density that comes out is reported: it is the first check that the force
field describes the liquid at all, and a box ten per cent off is a result about the force field
rather than about the electrolyte.

**GAFF with graph-network charges.** The solvents and the anion are typed with GAFF-2.11, as item
D8's ligands are. Their partial charges come from the OpenFF graph network, which reproduces
AM1-BCC charges in under a second per species, so no semi-empirical calculation is needed for
each of them. The cation is the Amber ion parameter set, because GAFF has no lithium. Three
consequences are worth stating: boron is outside GAFF, so the borate salts of a conductivity
table cannot be run here and are refused by name; a fixed-charge force field has no polarisation,
which is known to overbind a small cation to a carbonate; and the shells this produces are
therefore a model of the liquid, not a measurement of it.

**The topology is built here and not read back.** packmol writes coordinates without bond records,
and a force field matched to a bondless topology silently fails to find a template or, worse,
finds the wrong one. The topology is therefore assembled from the same molecule templates the box
was packed from, in the same order, and the packed file supplies coordinates only. The atom count
is checked against it.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

AVOGADRO = 6.02214076e23

#: solvents this module can parameterise: a SMILES and a density in g/cm3 at 298 K.
#: The density is used only to start the box at a plausible size; the barostat decides the rest.
SOLVENTS: Dict[str, Tuple[str, float]] = {
    "PC": ("CC1COC(=O)O1", 1.205),          # propylene carbonate
    "EC": ("C1COC(=O)O1", 1.321),           # ethylene carbonate
    "DMC": ("COC(=O)OC", 1.069),            # dimethyl carbonate
    "DEC": ("CCOC(=O)OCC", 0.975),          # diethyl carbonate
    "EMC": ("CCOC(=O)OC", 1.007),           # ethyl methyl carbonate
    "EA": ("CCOC(C)=O", 0.902),             # ethyl acetate
    "MA": ("COC(C)=O", 0.932),              # methyl acetate
    "DME": ("COCCOC", 0.867),               # 1,2-dimethoxyethane
    "DOL": ("C1COCO1", 1.060),              # 1,3-dioxolane
    "THF": ("C1CCOC1", 0.889),              # tetrahydrofuran
    "AN": ("CC#N", 0.786),                  # acetonitrile
}

#: salts this module can parameterise: the cation element and the anion's SMILES.
#: A borate is absent on purpose -- GAFF has no boron, so LiBF4 and LiBOB cannot be run here.
SALTS: Dict[str, Tuple[str, str]] = {
    "LiPF6": ("Li", "F[P-](F)(F)(F)(F)F"),
    "LiClO4": ("Li", "[O-][Cl](=O)(=O)=O"),
    "LiTFSI": ("Li", "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F"),
    "LiCF3SO3": ("Li", "[O-]S(=O)(=O)C(F)(F)F"),
    "NaPF6": ("Na", "F[P-](F)(F)(F)(F)F"),
}

#: the ion's residue name in the topology, which is also the Amber ion template's name
ION_RESIDUE = {"Li": "LI", "Na": "NA", "K": "K"}


class ElectrolyteError(RuntimeError):
    """A formulation cannot be built, or the simulation stack is not here."""


# ----------------------------------------------------------------------------------------------
# what is installed
# ----------------------------------------------------------------------------------------------

def tools_on_path() -> List[str]:
    """Put the running interpreter's own directory on PATH, and say what that added.

    GAFF atom types come from AmberTools' ``antechamber``, which the template generator calls as
    a program. In a conda environment that program sits beside the interpreter, and running the
    interpreter by its full path does not put it on PATH -- which is the usual reason a machine
    that has every package still cannot type a molecule. The fix is small enough to make here so
    that the failure cannot happen, and :func:`require` reports what is still missing.
    """
    here = os.path.dirname(os.path.abspath(sys.executable))
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if here and here not in parts:
        os.environ["PATH"] = here + os.pathsep + os.environ.get("PATH", "")
    return [t for t in ("antechamber", "packmol") if shutil.which(t) is not None]


def available() -> Dict[str, bool]:
    """Which parts of the stack can be used here."""
    found = tools_on_path()
    out = {"packmol": "packmol" in found, "antechamber": "antechamber" in found}
    for name, mod in (("openmm", "openmm"), ("openmmforcefields", "openmmforcefields"),
                      ("openff", "openff.toolkit"), ("nagl", "openff.nagl")):
        try:
            __import__(mod)
            out[name] = True
        except Exception:                    # noqa: BLE001 - an optional dependency
            out[name] = False
    return out


def require() -> None:
    missing = [k for k, v in available().items() if not v]
    if missing:
        raise ElectrolyteError(
            f"a bulk simulation needs {', '.join(missing)}, which is not usable here. The stack "
            f"is conda-forge only: micromamba create -n md -c conda-forge openmm packmol "
            f"ambertools openmmforcefields openff-toolkit openff-nagl rdkit")


# ----------------------------------------------------------------------------------------------
# a formulation
# ----------------------------------------------------------------------------------------------

@dataclass
class Formulation:
    """One electrolyte as a conductivity table reports it.

    `solvents` maps a name in :data:`SOLVENTS` to its weight fraction of the solvent, and
    `molality` is moles of salt per kilogram of solvent.
    """

    salt: str
    solvents: Dict[str, float]
    molality: float
    temperature: float = 298.15
    name: str = ""

    def __post_init__(self) -> None:
        if self.salt not in SALTS:
            raise ElectrolyteError(
                f"this module cannot parameterise {self.salt!r}; it knows "
                f"{', '.join(sorted(SALTS))}. A borate salt is absent because GAFF has no boron.")
        unknown = [s for s in self.solvents if s not in SOLVENTS]
        if unknown:
            raise ElectrolyteError(f"unknown solvent(s) {', '.join(sorted(unknown))}; this module "
                                   f"knows {', '.join(sorted(SOLVENTS))}")
        total = float(sum(self.solvents.values()))
        if total <= 0:
            raise ElectrolyteError("the solvent weight fractions have to add up to more than zero")
        self.solvents = {k: float(v) / total for k, v in self.solvents.items() if v > 0}
        if not (self.molality > 0):
            raise ElectrolyteError("the molality has to be above zero")
        if not self.name:
            self.name = self.label()

    def label(self) -> str:
        """A short name that says what the formulation is, usable as a file name."""
        parts = "_".join(f"{s}{round(100 * f):02d}" for s, f in sorted(self.solvents.items()))
        return f"{self.salt}_{parts}_m{self.molality:.3f}".replace(".", "p")

    @property
    def cation(self) -> str:
        return SALTS[self.salt][0]

    @property
    def anion_smiles(self) -> str:
        return SALTS[self.salt][1]

    def molar_masses(self) -> Dict[str, float]:
        out = {s: molar_mass(SOLVENTS[s][0]) for s in self.solvents}
        out["ANION"] = molar_mass(self.anion_smiles)
        out["ION"] = molar_mass(f"[{self.cation}+]")
        return out

    def counts(self, n_solvent: int = 250, min_ions: int = 4) -> Dict[str, int]:
        """How many molecules of each species the box holds, and how many ion pairs.

        The solvent counts follow from the weight fractions and the molar masses, the ion count
        from the molality and the solvent mass. Both are whole numbers, so the realised molality
        differs a little from the one asked for; :meth:`realised_molality` reports it.
        """
        M = self.molar_masses()
        share = {s: self.solvents[s] / M[s] for s in self.solvents}
        scale = float(sum(share.values()))
        raw = {s: n_solvent * share[s] / scale for s in share}
        counts = {s: max(1, int(round(v))) for s, v in raw.items()}
        # put the rounding error on the most abundant species, so the total is what was asked
        drift = n_solvent - sum(counts.values())
        if drift != 0:
            big = max(counts, key=lambda s: counts[s])
            counts[big] = max(1, counts[big] + drift)
        mass = sum(counts[s] * M[s] for s in counts)          # g per box, times Avogadro
        n_ion = max(int(min_ions), int(round(self.molality * mass / 1000.0)))
        counts["ION"] = n_ion
        counts["ANION"] = n_ion
        return counts

    def realised_molality(self, counts: Dict[str, int]) -> float:
        """The molality the whole-number counts actually give."""
        M = self.molar_masses()
        mass = sum(counts[s] * M[s] for s in self.solvents)
        return float(counts["ION"] * 1000.0 / mass)

    def mass(self, counts: Dict[str, int]) -> float:
        """The mass of the box in grams, times Avogadro's number."""
        M = self.molar_masses()
        return float(sum(counts[s] * M[s] for s in counts))

    def start_density(self) -> float:
        """The solvent mixture's density by ideal mixing, g/cm3."""
        inv = sum(self.solvents[s] / SOLVENTS[s][1] for s in self.solvents)
        return float(1.0 / inv)

    def box_side(self, counts: Dict[str, int], loosen: float = 1.10) -> float:
        """The side of the starting cube in angstrom, a little larger than the liquid's."""
        volume = self.mass(counts) / AVOGADRO / self.start_density()      # cm3
        return float((volume * 1e24) ** (1.0 / 3.0) * loosen)


def molar_mass(smiles: str) -> float:
    """The molar mass of a SMILES, hydrogens included, computed rather than transcribed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ElectrolyteError(f"{smiles!r} is not a SMILES this program can read")
    return float(Descriptors.MolWt(Chem.AddHs(mol)))


# ----------------------------------------------------------------------------------------------
# the molecule templates
# ----------------------------------------------------------------------------------------------

def template(smiles: str, residue: str, seed: int = 0) -> Chem.Mol:
    """One species as an RDKit molecule with a conformation and PDB residue information.

    The same object serves three purposes: packmol packs copies of it, the topology is assembled
    from its bonds, and :mod:`cmfqsar.shells` uses it to give a cut-out shell its bond orders.
    That is why the atom order must never be changed between them.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ElectrolyteError(f"{smiles!r} is not a SMILES this program can read")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=int(seed)) != 0:
        raise ElectrolyteError(f"no conformation could be generated for {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:                        # noqa: BLE001 - MMFF has no parameters for PF6
        pass
    Chem.SanitizeMol(mol)
    name = residue[:3].upper().ljust(3)
    for k, atom in enumerate(mol.GetAtoms()):
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(name)
        info.SetResidueNumber(1)
        info.SetIsHeteroAtom(True)
        info.SetName(f"{atom.GetSymbol()}{k}"[:4].ljust(4))
        atom.SetMonomerInfo(info)
    mol.SetProp("_Name", residue)
    return mol


def templates(form: Formulation, seed: int = 0) -> Dict[str, Chem.Mol]:
    """A template per species of the formulation, keyed by its residue name.

    The cation has no template: it is one atom and the force field types it as an ion.
    """
    out: Dict[str, Chem.Mol] = {}
    for i, s in enumerate(sorted(form.solvents)):
        out[s] = template(SOLVENTS[s][0], s, seed=seed + i)
    out["ANI"] = template(form.anion_smiles, "ANI", seed=seed + 50)
    return out


def ion_pdb(path: str, ion: str) -> None:
    """A one-atom PDB of the bare cation, for packmol to place."""
    res = ION_RESIDUE.get(ion, ion.upper())
    name = res.ljust(4)[:4]
    with open(path, "w", encoding="ascii") as fh:
        fh.write(f"HETATM    1 {name} {res:<3s} A   1       0.000   0.000   0.000  "
                 f"1.00  0.00          {ion.upper():>2s}\n")
        fh.write("END\n")


# ----------------------------------------------------------------------------------------------
# packing
# ----------------------------------------------------------------------------------------------

def species_order(form: Formulation, counts: Dict[str, int]) -> List[Tuple[str, int]]:
    """The species and their counts in the order the box is packed and the topology is built.

    Fixing this order in one place is what lets the packed coordinates and the assembled
    topology be matched atom for atom.
    """
    out = [(s, int(counts[s])) for s in sorted(form.solvents)]
    out.append(("ANI", int(counts["ANION"])))
    out.append((ION_RESIDUE.get(form.cation, form.cation.upper()), int(counts["ION"])))
    return out


def pack(form: Formulation, counts: Dict[str, int], workdir: str, *, seed: int = 1,
         tolerance: float = 2.2, loosen: float = 1.10) -> Tuple[str, float]:
    """Write the species' PDB files, run packmol, and return (packed PDB, box side)."""
    if shutil.which("packmol") is None:
        raise ElectrolyteError("packmol is not on PATH, so no box can be packed")
    os.makedirs(workdir, exist_ok=True)
    tmpl = templates(form)
    for name, mol in tmpl.items():
        Chem.MolToPDBFile(mol, os.path.join(workdir, f"{name}.pdb"))
    ion_res = ION_RESIDUE.get(form.cation, form.cation.upper())
    ion_pdb(os.path.join(workdir, f"{ion_res}.pdb"), form.cation)

    side = form.box_side(counts, loosen=loosen)
    out_pdb = os.path.join(workdir, "box.pdb")
    lines = [f"tolerance {tolerance:.2f}", "filetype pdb", f"output {out_pdb}",
             f"seed {int(seed)}"]
    for name, n in species_order(form, counts):
        lines += [f"structure {os.path.join(workdir, name + '.pdb')}",
                  f"  number {int(n)}",
                  f"  inside cube 1.0 1.0 1.0 {side - 2.0:.3f}",
                  "end structure"]
    inp = os.path.join(workdir, "pack.inp")
    with open(inp, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(inp, "r", encoding="ascii") as fh:
        res = subprocess.run(["packmol"], stdin=fh, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(out_pdb):
        tail = (res.stdout or "")[-1200:] + (res.stderr or "")[-1200:]
        raise ElectrolyteError(f"packmol failed for {form.name}: {tail}")
    return out_pdb, side


# ----------------------------------------------------------------------------------------------
# the simulation
# ----------------------------------------------------------------------------------------------

@dataclass
class Settings:
    """What one run does. The defaults are the ones item BT5's measurement used."""
    temperature: float = 298.15
    pressure: float = 1.0
    timestep_fs: float = 2.0
    cutoff_nm: float = 1.0
    n_solvent: int = 250
    min_ions: int = 4
    equilibration_ps: float = 150.0
    production_ps: float = 1000.0
    frame_ps: float = 5.0
    seed: int = 1
    platform: str = "CUDA"
    charge_model: str = "openff-gnn-am1bcc-0.1.0-rc.3.pt"
    small_molecule_forcefield: str = "gaff-2.11"
    ion_forcefield: str = "amber14/tip3p.xml"
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def production_steps(self) -> int:
        return max(1, int(round(self.production_ps * 1e3 / self.timestep_fs)))

    @property
    def equilibration_steps(self) -> int:
        return max(1, int(round(self.equilibration_ps * 1e3 / self.timestep_fs)))

    @property
    def frame_steps(self) -> int:
        return max(1, int(round(self.frame_ps * 1e3 / self.timestep_fs)))


def build(form: Formulation, counts: Dict[str, int], packed_pdb: str, side: float,
          settings: Settings):
    """(topology, system, positions) for one packed box."""
    require()
    import openmm
    from openmm import app, unit
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SystemGenerator

    tmpl = templates(form)
    offs = []
    for name in sorted(tmpl):
        off = Molecule.from_rdkit(tmpl[name], allow_undefined_stereo=True)
        off.assign_partial_charges(settings.charge_model)
        offs.append(off)

    top = app.Topology()
    chain = top.addChain()
    ion_res = ION_RESIDUE.get(form.cation, form.cation.upper())
    for name, n in species_order(form, counts):
        for _ in range(int(n)):
            res = top.addResidue(name, chain)
            if name == ion_res:
                top.addAtom(ion_res, app.Element.getBySymbol(form.cation), res)
                continue
            mol = tmpl[name]
            atoms = [top.addAtom(f"{a.GetSymbol()}{k}",
                                 app.Element.getBySymbol(a.GetSymbol()), res)
                     for k, a in enumerate(mol.GetAtoms())]
            for b in mol.GetBonds():
                top.addBond(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])
    top.setPeriodicBoxVectors(np.eye(3) * side / 10.0 * unit.nanometer)

    packed = app.PDBFile(packed_pdb)
    if packed.topology.getNumAtoms() != top.getNumAtoms():
        raise ElectrolyteError(f"the packed box has {packed.topology.getNumAtoms()} atoms and "
                               f"the topology built here {top.getNumAtoms()}")
    forcefield_kwargs = {"constraints": app.HBonds, "rigidWater": True, "removeCMMotion": True}
    periodic = {"nonbondedMethod": app.PME,
                "nonbondedCutoff": float(settings.cutoff_nm) * unit.nanometer}
    gen = SystemGenerator(forcefields=[settings.ion_forcefield],
                          small_molecule_forcefield=settings.small_molecule_forcefield,
                          molecules=offs, forcefield_kwargs=forcefield_kwargs,
                          periodic_forcefield_kwargs=periodic)
    system = gen.create_system(top, molecules=offs)
    net = net_charge(system)
    if abs(net) > 1e-3:
        raise ElectrolyteError(f"the box carries a net charge of {net:.3f}, so the same number "
                               f"of cations and anions did not go in")
    system.addForce(openmm.MonteCarloBarostat(float(settings.pressure) * unit.bar,
                                              float(settings.temperature) * unit.kelvin))
    return top, system, packed.positions


def net_charge(system) -> float:
    """The sum of the partial charges of a built system, which has to be zero."""
    import openmm
    from openmm import unit

    nb = next((f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
    if nb is None:
        return 0.0
    return float(sum(nb.getParticleParameters(p)[0].value_in_unit(unit.elementary_charge)
                     for p in range(system.getNumParticles())))


def run(form: Formulation, workdir: str, settings: Optional[Settings] = None,
        progress=None) -> Dict[str, object]:
    """Pack, equilibrate and run one formulation; returns what was done and where it is.

    The result holds the paths of the topology PDB and the DCD of production frames, the
    realised molality, the density the barostat settled at, and the speed, so that a later
    reading of the trajectory can report the cost of producing it.
    """
    require()
    import openmm
    from openmm import app, unit

    settings = settings or Settings()
    os.makedirs(workdir, exist_ok=True)
    done = os.path.join(workdir, "run.json")
    if os.path.exists(done):
        with open(done, encoding="utf-8") as fh:
            return json.load(fh)

    counts = form.counts(settings.n_solvent, settings.min_ions)
    packed_pdb, side = pack(form, counts, workdir, seed=settings.seed)
    top, system, positions = build(form, counts, packed_pdb, side, settings)

    integrator = openmm.LangevinMiddleIntegrator(
        float(settings.temperature) * unit.kelvin, 1.0 / unit.picosecond,
        float(settings.timestep_fs) * unit.femtosecond)
    integrator.setRandomNumberSeed(int(settings.seed))
    try:
        platform = openmm.Platform.getPlatformByName(settings.platform)
    except Exception:                        # noqa: BLE001 - fall back rather than fail
        platform = None
    sim = (app.Simulation(top, system, integrator, platform) if platform is not None
           else app.Simulation(top, system, integrator))
    sim.context.setPositions(positions)
    sim.minimizeEnergy(maxIterations=5000)
    sim.context.setVelocitiesToTemperature(float(settings.temperature) * unit.kelvin,
                                           int(settings.seed))
    if progress:
        progress(f"{form.name}: equilibrating {settings.equilibration_ps:.0f} ps")
    sim.step(settings.equilibration_steps)

    topology_pdb = os.path.join(workdir, "topology.pdb")
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(topology_pdb, "w", encoding="ascii") as fh:
        app.PDBFile.writeFile(top, state.getPositions(), fh)
    dcd = os.path.join(workdir, "production.dcd")
    sim.reporters.append(app.DCDReporter(dcd, settings.frame_steps))
    if progress:
        progress(f"{form.name}: running {settings.production_ps:.0f} ps")
    t0 = time.time()
    sim.step(settings.production_steps)
    seconds = time.time() - t0

    state = sim.context.getState()
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    volume = float(abs(np.linalg.det(box)))
    density = form.mass(counts) / AVOGADRO / (volume * 1e-24)
    out = {
        "name": form.name, "salt": form.salt, "solvents": form.solvents,
        "molality": float(form.molality),
        "realised_molality": form.realised_molality(counts),
        "temperature": float(settings.temperature), "counts": counts,
        "atoms": int(system.getNumParticles()),
        "start_side": float(side), "final_side": float(volume ** (1.0 / 3.0)),
        "density": float(density), "start_density": form.start_density(),
        "production_ps": float(settings.production_ps),
        "frames": int(settings.production_steps // settings.frame_steps),
        "seconds": float(seconds),
        "ns_per_day": float(settings.production_ps * 1e-3 / (seconds / 86400.0)),
        "topology": topology_pdb, "trajectory": dcd,
        "charge_model": settings.charge_model,
        "forcefield": settings.small_molecule_forcefield,
        "ion_forcefield": settings.ion_forcefield,
    }
    with open(done, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    return out


def shell_templates(form: Formulation) -> Dict[str, Chem.Mol]:
    """The templates :mod:`cmfqsar.shells` needs to turn a cut-out shell into a molecule.

    The same objects the box was packed from, keyed by the residue name the topology carries.
    """
    return dict(templates(form))
