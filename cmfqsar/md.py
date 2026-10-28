"""
Short ligand-in-pocket simulations, for plan item D8.

`cmfqsar/trajectory.py` (item D7) reads a simulation as a conformational ensemble. This module
produces the simulations that reader was built for, so that the question of item D8 can be
asked: does the mean field of a ligand's trajectory in its pocket model a series better than the
single crystal pose it started from?

What one run is, and why each choice was made.

**The whole protein, not the trimmed pocket.** ``examples/pockets/`` holds binding sites cut to
the residues near the ligand. Those files are right for display and for the complementarity
score, and wrong for a simulation: every cut leaves a broken peptide bond that a preparation
tool turns into a charged terminus, so a pocket of twenty residues would carry twenty charges
that the protein does not have. The source entry named in the file's ``REMARK 400`` line is
downloaded instead and simulated whole.

**The ligands keep their frame.** The free-energy collections give their ligands in the
coordinates of the reference entry, so the protein and the ligands are already in one frame and
nothing is docked or superimposed here. That is what makes the comparison of item D8 fair: the
crystal pose and the trajectory describe the same molecule in the same place.

**GAFF through OpenMM**, with AM1-BCC charges, which is the item's own choice; the protein is
Amber ff14SB and the water TIP3P. Hydrogen mass repartitioning allows a 4 fs step.

**The backbone is restrained lightly** (``restraint_k``, on the alpha carbons) through the
equilibration and released for the production run unless ``restrain_production`` is set. A few
nanoseconds is not long enough for a protein to need holding, but it is long enough for a
loosely packed crystal structure to drift, and the field model cares about the ligand's position
relative to the site rather than about the protein's own motion.

**Nothing is optimised away.** Every run writes its topology, its trajectory and a record of the
settings, and a run already finished is not repeated, so a set can be built over several
sessions.

OpenMM and the OpenFF/AmberTools stack are not dependencies of this program. They are imported
when a simulation is actually run, and :func:`available` says whether they are there.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

RCSB_URL = "https://files.rcsb.org/download/{code}.pdb"
#: kcal/mol/A^2 on the alpha carbons while the system settles
DEFAULT_RESTRAINT = 5.0


class MDError(RuntimeError):
    pass


@dataclass
class Settings:
    """Everything that decides what a run is; written beside the trajectory."""
    nanoseconds: float = 2.0
    equilibration_ps: float = 200.0
    timestep_fs: float = 4.0            # 4 fs is safe with hydrogen mass repartitioning
    temperature: float = 300.0          # kelvin
    pressure: float = 1.0               # bar; 0 runs NVT
    padding: float = 10.0               # angstrom of water around the complex
    ionic_strength: float = 0.15        # molar
    frame_ps: float = 10.0              # how often a frame is written
    restraint_k: float = DEFAULT_RESTRAINT
    restrain_production: bool = False
    small_molecule_forcefield: str = "gaff-2.11"
    protein_forcefield: str = "amber14-all.xml"
    water_forcefield: str = "amber14/tip3pfb.xml"
    platform: str = "CUDA"
    seed: int = 0

    @property
    def production_steps(self) -> int:
        return int(round(self.nanoseconds * 1e6 / self.timestep_fs))

    @property
    def equilibration_steps(self) -> int:
        return int(round(self.equilibration_ps * 1e3 / self.timestep_fs))

    @property
    def frame_steps(self) -> int:
        return max(1, int(round(self.frame_ps * 1e3 / self.timestep_fs)))


def available() -> Dict[str, bool]:
    """Which parts of the stack can be imported here."""
    out = {}
    for name, mod in (("openmm", "openmm"), ("pdbfixer", "pdbfixer"),
                      ("openmmforcefields", "openmmforcefields"),
                      ("openff", "openff.toolkit")):
        try:
            __import__(mod)
            out[name] = True
        except Exception:                    # noqa: BLE001 - an optional dependency
            out[name] = False
    return out


def charge_toolkit() -> Optional[str]:
    """The registered toolkit that can assign AM1-BCC charges, or None.

    AmberTools registers itself only when ``antechamber`` is on PATH, which running the
    environment's interpreter by its full path does not do; this is the usual reason a run that
    has every package still cannot charge a ligand.
    """
    try:
        from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
    except Exception:                        # noqa: BLE001 - an optional dependency
        return None
    for tk in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits:
        name = getattr(tk, "toolkit_name", "")
        if "AmberTools" in name or "OpenEye" in name:
            return name
    return None


def require(charges: bool = True) -> None:
    missing = [k for k, v in available().items() if not v]
    if missing:
        raise MDError(f"a simulation needs {', '.join(missing)}, which is not installed here. "
                      f"The stack is conda-forge only: "
                      f"micromamba create -n md -c conda-forge openmm ambertools "
                      f"openmmforcefields openff-toolkit pdbfixer rdkit")
    if charges and charge_toolkit() is None:
        raise MDError("no toolkit here can assign AM1-BCC charges. AmberTools is usually the "
                      "one, and it registers only when antechamber is on PATH, which running "
                      "the environment's interpreter by its full path does not do. Put the "
                      "environment's bin directory on PATH first, for example "
                      "'export PATH=$MAMBA_ROOT_PREFIX/envs/md/bin:$PATH', or activate it")


# ----------------------------------------------------------------------------------------------
# the receptor
# ----------------------------------------------------------------------------------------------

def entry_of(pocket_pdb: str) -> Optional[str]:
    """The source entry a pocket file of ``examples/pockets/`` names in its REMARK 400 lines."""
    with open(pocket_pdb, errors="replace") as fh:
        for line in fh:
            if line.startswith("REMARK 400") and "Source entry" in line:
                parts = line.split("Source entry", 1)[1].split()
                if parts:
                    return parts[0].strip().upper()
            if line.startswith(("ATOM", "HETATM")):
                break
    return None


def fetch_entry(code: str, directory: str) -> str:
    """The PDB entry, downloaded from RCSB once and kept. Entries there are public domain."""
    code = str(code).strip().upper()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{code}.pdb")
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return path
    url = RCSB_URL.format(code=code)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
    except Exception as exc:                 # noqa: BLE001 - reported with the address tried
        raise MDError(f"{code} could not be downloaded from {url}: {exc}")
    if len(data) < 1000:
        raise MDError(f"{code} came back as {len(data)} bytes from {url}")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def chains_of(pdb_path: str) -> List[str]:
    """The chain identifier of every atom record, in file order."""
    out = []
    with open(pdb_path, errors="replace") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                out.append(line[21])
    return out


def chains_near(pdb_path: str, ligands: Sequence[Chem.Mol], radius: float = 5.0,
                min_atoms: int = 20) -> List[str]:
    """The chains of an entry whose atoms line the site the ligands occupy.

    A crystal entry often holds several copies of the protein -- 4HW3, the receptor of the mcl1
    set, holds twelve -- and simulating all of them costs ten times what the question needs. The
    ligands of a free-energy collection are given in the entry's own frame, so the chain they sit
    in can be found by distance. A chain is kept when at least `min_atoms` of its atoms are
    within `radius` of some ligand atom, which keeps both chains of a site formed by two.
    """
    points = []
    for m in ligands:
        if m is None or m.GetNumConformers() == 0:
            continue
        points.append(np.asarray(m.GetConformer().GetPositions(), float))
    if not points:
        raise MDError("the ligands have no coordinates, so the chain cannot be found by distance")
    lig = np.vstack(points)

    coords: Dict[str, List[Tuple[float, float, float]]] = {}
    with open(pdb_path, errors="replace") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain = line[21]
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            coords.setdefault(chain, []).append(xyz)
    out = []
    for chain, xyz in coords.items():
        a = np.asarray(xyz, float)
        # the ligand set is small and a chain is a few thousand atoms, so this is one block
        near = int((np.linalg.norm(a[:, None, :] - lig[None, :, :], axis=2)
                    .min(axis=1) <= float(radius)).sum())
        if near >= int(min_atoms):
            out.append((near, chain))
    if not out:
        raise MDError(f"no chain of {os.path.basename(pdb_path)} has {min_atoms} atoms within "
                      f"{radius:g} A of a ligand; are the ligands in the entry's frame?")
    out.sort(reverse=True)
    return [c for _n, c in out]


def prepare_receptor(pdb_path: str, out_path: str, ph: float = 7.0,
                     keep_chains: Sequence[str] = ()) -> str:
    """The entry made ready to simulate: one copy, no water, missing atoms built, hydrogens on.

    Heteroatoms are dropped, the co-crystallised ligand included: the ligand that is simulated
    is the one from the data set, placed in the same frame, and keeping the crystal ligand as
    well would put two molecules in one site.
    """
    require(charges=False)
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=pdb_path)
    if keep_chains:
        want = {str(c).strip() for c in keep_chains}
        drop = [i for i, ch in enumerate(fixer.topology.chains()) if ch.id.strip() not in want]
        if drop:
            fixer.removeChains(drop)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    # a missing loop in the middle of a chain is built; a missing tail is not, because a tail
    # built from nothing would be a guess that moves through the simulation
    keys = list(fixer.missingResidues)
    chains = list(fixer.topology.chains())
    for key in keys:
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(float(ph))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    return out_path


# ----------------------------------------------------------------------------------------------
# the system
# ----------------------------------------------------------------------------------------------

def _openff_molecule(mol: Chem.Mol):
    """The ligand as an OpenFF molecule, hydrogens explicit and the pose kept."""
    from openff.toolkit.topology import Molecule

    work = Chem.Mol(mol)
    if any(a.GetTotalNumHs() > 0 for a in work.GetAtoms()):
        work = Chem.AddHs(work, addCoords=True)
    if work.GetNumConformers() == 0:
        raise MDError("the ligand has no coordinates, so it has no pose to simulate")
    off = Molecule.from_rdkit(work, allow_undefined_stereo=True)
    return off, work


def build(receptor_pdb: str, ligand: Chem.Mol, settings: Settings):
    """(simulation-ready topology, system, positions) for one complex in water."""
    require()
    import openmm
    from openmm import app, unit
    from openmmforcefields.generators import SystemGenerator

    off_ligand, rd_ligand = _openff_molecule(ligand)
    forcefield_kwargs = {"constraints": app.HBonds, "rigidWater": True,
                         "removeCMMotion": True, "hydrogenMass": 3.0 * unit.amu}
    periodic = {"nonbondedMethod": app.PME,
                "nonbondedCutoff": 1.0 * unit.nanometer}
    generator = SystemGenerator(
        forcefields=[settings.protein_forcefield, settings.water_forcefield],
        small_molecule_forcefield=settings.small_molecule_forcefield,
        molecules=[off_ligand], forcefield_kwargs=forcefield_kwargs,
        periodic_forcefield_kwargs=periodic)

    protein = app.PDBFile(receptor_pdb)
    modeller = app.Modeller(protein.topology, protein.positions)
    lig_top = off_ligand.to_topology().to_openmm()
    lig_pos = off_ligand.conformers[0].to_openmm()
    modeller.add(lig_top, lig_pos)
    modeller.addSolvent(generator.forcefield, model="tip3p",
                        padding=float(settings.padding) * unit.angstrom,
                        ionicStrength=float(settings.ionic_strength) * unit.molar,
                        neutralize=True)
    system = generator.create_system(modeller.topology, molecules=[off_ligand])
    if settings.pressure > 0:
        system.addForce(openmm.MonteCarloBarostat(float(settings.pressure) * unit.bar,
                                                  float(settings.temperature) * unit.kelvin))
    return modeller.topology, system, modeller.positions, rd_ligand


def add_restraints(system, topology, positions, k: float):
    """Harmonic restraints on the alpha carbons; returns the index of the force that was added."""
    import openmm
    from openmm import unit

    force = openmm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", float(k) * unit.kilocalories_per_mole / unit.angstrom ** 2)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    n = 0
    for atom in topology.atoms():
        if atom.name == "CA" and atom.residue.name not in ("HOH", "WAT", "NA", "CL"):
            p = positions[atom.index]
            force.addParticle(atom.index, [p.x, p.y, p.z])
            n += 1
    if n == 0:
        return None, 0
    return system.addForce(force), n


# ----------------------------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------------------------

@dataclass
class RunOutcome:
    name: str
    topology: str = ""
    trajectory: str = ""
    frames: int = 0
    seconds: float = 0.0
    ns_per_day: float = 0.0
    atoms: int = 0
    error: str = ""
    reused: bool = False


def simulate(ligand: Chem.Mol, name: str, receptor_pdb: str, outdir: str,
             settings: Optional[Settings] = None, log=print) -> RunOutcome:
    """One ligand: build, minimise, equilibrate with restraints, run, write the trajectory."""
    settings = settings or Settings()
    t0 = time.time()
    out = RunOutcome(name=name)
    top_path = os.path.join(outdir, f"{name}.pdb")
    dcd_path = os.path.join(outdir, f"{name}.dcd")
    done_path = os.path.join(outdir, f"{name}.json")
    try:
        if os.path.isfile(done_path) and os.path.isfile(dcd_path):
            with open(done_path) as fh:
                rec = json.load(fh)
            if rec.get("settings") == asdict(settings):
                out.topology, out.trajectory = top_path, dcd_path
                out.frames = int(rec.get("frames", 0))
                out.seconds = float(rec.get("seconds", 0.0))
                out.atoms = int(rec.get("atoms", 0))
                out.ns_per_day = float(rec.get("ns_per_day", 0.0))
                out.reused = True
                return out

        require()
        import openmm
        from openmm import app, unit

        os.makedirs(outdir, exist_ok=True)
        topology, system, positions, _rd = build(receptor_pdb, ligand, settings)
        out.atoms = topology.getNumAtoms()
        _idx, n_restrained = add_restraints(system, topology, positions, settings.restraint_k)

        integrator = openmm.LangevinMiddleIntegrator(
            float(settings.temperature) * unit.kelvin, 1.0 / unit.picosecond,
            float(settings.timestep_fs) * unit.femtosecond)
        integrator.setRandomNumberSeed(int(settings.seed) + abs(hash(name)) % 100000)
        try:
            platform = openmm.Platform.getPlatformByName(settings.platform)
        except Exception:                    # noqa: BLE001 - fall back and say so
            log(f"  {name}: platform {settings.platform} is not available; using the fastest one")
            platform = None
        sim = (app.Simulation(topology, system, integrator, platform) if platform
               else app.Simulation(topology, system, integrator))
        sim.context.setPositions(positions)

        log(f"  {name}: {out.atoms} atoms, {n_restrained} alpha carbons restrained; minimising")
        sim.minimizeEnergy()
        sim.context.setVelocitiesToTemperature(float(settings.temperature) * unit.kelvin,
                                               int(settings.seed) + 1)
        sim.step(settings.equilibration_steps)
        if not settings.restrain_production and n_restrained:
            sim.context.setParameter("k", 0.0)
        with open(top_path, "w") as fh:
            app.PDBFile.writeFile(topology, sim.context.getState(getPositions=True).getPositions(),
                                  fh, keepIds=True)
        sim.reporters.append(app.DCDReporter(dcd_path, settings.frame_steps))
        t_run = time.time()
        sim.step(settings.production_steps)
        wall = time.time() - t_run
        out.frames = settings.production_steps // settings.frame_steps
        out.topology, out.trajectory = top_path, dcd_path
        out.seconds = time.time() - t0
        out.ns_per_day = settings.nanoseconds * 86400.0 / max(wall, 1e-9)
        with open(done_path, "w") as fh:
            json.dump({"name": name, "frames": out.frames, "atoms": out.atoms,
                       "seconds": out.seconds, "ns_per_day": out.ns_per_day,
                       "settings": asdict(settings)}, fh, indent=1)
        log(f"  {name}: {out.frames} frames in {wall / 60:.1f} min "
            f"({out.ns_per_day:.0f} ns/day)")
    except Exception as exc:                 # noqa: BLE001 - a failed ligand is not fatal
        out.error = f"{type(exc).__name__}: {exc}"
        out.seconds = time.time() - t0
        log(f"  {name}: FAILED -- {out.error}")
    return out


def simulate_set(mols: Sequence[Chem.Mol], names: Sequence[str], receptor_pdb: str, outdir: str,
                 settings: Optional[Settings] = None, log=print) -> List[RunOutcome]:
    """Every ligand of a set, in order; one that fails is reported and the rest go on."""
    settings = settings or Settings()
    out = []
    for i, (m, n) in enumerate(zip(mols, names)):
        log(f"[{i + 1}/{len(mols)}] {n}")
        out.append(simulate(m, n, receptor_pdb, outdir, settings=settings, log=log))
    return out
