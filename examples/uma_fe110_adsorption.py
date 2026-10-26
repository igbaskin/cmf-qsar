"""Adsorption energies and adsorbed poses of corrosion inhibitors on Fe(110) with the UMA potential.

The screening tier of the corrosion-inhibitor project: every molecule of a scaffold-node CSV
(examples/corrosion/*.csv) is placed on a bcc Fe(110) slab in a few starting poses, each pose is
relaxed with Meta's UMA machine-learned interatomic potential (the OC20 task head, trained on
DFT adsorption calculations of molecules on metal surfaces), and the lowest-energy pose gives

    E_ads = E(slab + molecule) - E(slab) - E(molecule alone in a 20 A box),

all three with the same model and task head.  The result is an SDF of the adsorbed molecules
alone (surface normal along z, surface plane at z = 0, the cell centre at x = y = 0) with the
energies as SD properties, which the CMF application reads as an aligned, labelled data set:

    load_dataset("examples/corrosion_fe110/157_benzimidazole_uma.sdf", activity_prop="E_ads_kJmol")

The slab: bcc Fe, a = 2.78 A (the minimum of UMA's oc20 head for bcc Fe; --lattice changes
it), four layers, 6x6 orthogonal surface cell (16.7 x 11.8 A; grown automatically when a
molecule's footprint plus --margin would not fit), 15 A of vacuum, nominally spin-polarised
(initial moments 2.2 mu_B, which UMA's oc20 head does not read but VASP will, when the same
structures go to the calibration tier).  All slab atoms are fixed by default: for the oc20 head
the ideal Fe(110) surface is a saddle point, and any symmetry-breaking perturbation (an
adsorbate, a 0.05 A shake) slides the free layers into a spurious reconstruction with Fe-Fe
contacts of 2.1-2.3 A that lowers the energy by ~0.25 eV per free atom and dominates the
adsorption energy; a rigid slab keeps the three energies consistent (--fixed-layers N frees
the top `layers - N` layers if a future model does not show this problem).

The molecules: the neutral form (charges are removed where a proton can be added or taken
away; permanent charges stay and are recorded), embedded with ETKDG + MMFF.  Starting poses:
"flat" (the molecular plane parallel to the surface, lowest heavy atom 2.2 A above it),
"upright" (the lone pair of the most basic nitrogen pointing down, N 2.0 A above the surface;
the most nucleophilic S or O when there is no nitrogen) and "random" (a random rotation).
Each pose is first docked (40 steps with the heavy atoms held at their height, so that the
molecule settles into the chemisorption region instead of drifting off a surface that, for a
potential without dispersion, has no physisorption well), then relaxed freely.  A pose whose
bonds broke is reported but never chosen while an intact one exists; a pose that ended
unbound (all heavy atoms above 4 A, E_ads ~ 0) is a valid result and is marked `bound` = 0.
`gas_intact` = 0 flags a molecule whose free-molecule reference changed its bonding during
the relaxation: its E_ads is then an artefact.

Run in the WSL environment ~/uma-env (see examples/corrosion_fe110/README.md):

    python examples/uma_fe110_adsorption.py examples/corrosion/157_c1ccc2_nH_cnc2c1.csv \
        -o examples/corrosion_fe110/157_benzimidazole_uma.sdf [--limit 3] [--select Benzimidazole]
        [--cell 6x6] [--poses flat,upright,random] [--fmax 0.05] [--steps 300] [--device cuda]

Every molecule's record (energies of all poses, the final geometries) goes to <out>_work/, so a
stopped run resumes where it was (--fresh recomputes), and the SDF and the CSV summary next to
it are rewritten after every molecule.  The UMA weights need a licence on Hugging Face: accept the
licence of facebook/UMA and run `hf auth login` once in the environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.build import bcc110
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE, LBFGS
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

EV_TO_KJMOL = 96.485332
FE_MAGMOM = 2.2
POSE_LABELS = ("flat", "upright", "random")
OPTIMIZERS = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}


# ----------------------------------------------------------------------------------------- slab

def build_fe110_slab(nx: int = 6, ny: int = 6, layers: int = 4, a: float = 2.78, vacuum: float = 15.0,
                     fixed_layers: Optional[int] = None, magmom: float = FE_MAGMOM) -> Atoms:
    """A bcc Fe(110) slab: `layers` layers in an orthogonal nx x ny surface cell (ny even), `vacuum`
    A of empty space between periodic images along z, the bottom `fixed_layers` layers fixed (all
    of them by default: see the module docstring).  ASE tags the layers 1 (top) .. `layers` (bottom)."""
    if fixed_layers is None:
        fixed_layers = layers
    if ny % 2:
        raise ValueError("the orthogonal bcc(110) cell needs an even ny")
    slab = bcc110("Fe", size=(nx, ny, layers), a=a, vacuum=vacuum / 2.0, orthogonal=True)
    slab.pbc = (True, True, True)
    slab.info.pop("adsorbate_info", None)        # ASE's site table: unhashable, and extxyz cannot write it
    tags = slab.get_tags()
    slab.set_constraint(FixAtoms(mask=tags > layers - fixed_layers))
    slab.set_initial_magnetic_moments([magmom] * len(slab))
    slab.info.update(nx=nx, ny=ny, layers=layers, a=a, vacuum=vacuum, fixed_layers=fixed_layers)
    return slab


def top_layer_z(atoms: Atoms) -> float:
    """Mean height of the top Fe layer (tag 1)."""
    tags = atoms.get_tags()
    z = atoms.positions[(tags == 1) & (atoms.numbers == 26), 2]
    return float(z.mean())


def cell_for_footprint(extent: Tuple[float, float], a: float = 2.83, nx: int = 6, ny: int = 6,
                       margin: float = 6.0) -> Tuple[int, int]:
    """The smallest nx x ny >= the defaults whose cell holds a footprint of `extent` (long, short
    side, A) with `margin` A between periodic images: nx steps the x side by a, ny by a/sqrt(2)
    (the orthogonal bcc(110) cell of ASE)."""
    long_side, short_side = max(extent), min(extent)
    while nx * a < long_side + margin:
        nx += 1
    while ny * a / math.sqrt(2.0) < short_side + margin:
        ny += 2                                   # the orthogonal bcc(110) cell needs an even ny
    return nx, ny


# ------------------------------------------------------------------------------------ molecules

def neutralize(mol: Chem.Mol) -> Chem.Mol:
    """The neutral form: protons added or removed where the charge is a protonation state
    (ammonium, carboxylate, ...); permanent charges (quaternary N, nitro) stay."""
    try:
        return rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        return mol


def embed_smiles(smiles: str, seed: int = 0) -> Chem.Mol:
    """RDKit molecule with hydrogens and one ETKDG + MMFF conformer, in the neutral form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    mol = neutralize(mol)
    m = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(m, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(m, params) != 0:
            raise RuntimeError(f"3D embedding failed for {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    except Exception:
        pass
    Chem.SanitizeMol(m)        # MMFF set-up re-perceives aromaticity in place; restore RDKit's model
    return m


def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 6 and any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetAtomicNum() in (8, 16)
        for b in atom.GetBonds())


def anchor_atom(mol: Chem.Mol) -> Optional[int]:
    """The atom that points at the surface in the upright pose: the most basic nitrogen, by a
    simple rule (aliphatic amine > pyridine-type aromatic N > imine > aniline > pyrrole-type
    NH; amide, nitro and quaternary N never), else a thiol/thioether S, else a hydroxyl/ether O."""
    best, best_score = None, 0.0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        score = 0.0
        if z == 7:
            if atom.GetFormalCharge() != 0:
                continue
            heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            if any(_is_carbonyl_carbon(n) for n in heavy):
                continue                                                    # amide / sulfonamide-like
            if any(n.GetAtomicNum() in (7, 8) and not n.GetIsAromatic() for n in heavy):
                continue                                                    # nitro, N-oxide, hydrazide
            if atom.GetIsAromatic():
                score = 3.0 if atom.GetTotalNumHs() == 0 and atom.GetDegree() == 2 else 0.5
            elif any(n.GetIsAromatic() for n in heavy):
                score = 1.5                                                 # aniline-type
            elif any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
                score = 2.0                                                 # imine
            else:
                score = 4.0                                                 # aliphatic amine
        elif z == 16:
            if atom.GetDegree() <= 2 and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()):
                score = 0.4                                                 # thiol / thioether
            elif atom.GetDegree() == 1:
                score = 0.4                                                 # thione S
        elif z == 8:
            if atom.GetDegree() <= 2 and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()):
                score = 0.2
        if score > best_score:
            best, best_score = atom.GetIdx(), score
    return best


def mol_coords(mol: Chem.Mol) -> np.ndarray:
    return np.array(mol.GetConformer().GetPositions(), dtype=float)


def molecule_atoms(mol: Chem.Mol, box: float = 20.0) -> Atoms:
    """The molecule alone, centred in a cubic periodic box (the gas-phase reference)."""
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    atoms = Atoms(symbols, positions=mol_coords(mol), cell=[box, box, box], pbc=True)
    atoms.center()
    return atoms


# ----------------------------------------------------------------------------------------- poses

def principal_axes(coords: np.ndarray) -> np.ndarray:
    """Rows: the principal axes of the point cloud, by decreasing extent, as a right-handed frame."""
    centred = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axes = np.eye(3)
    axes[:len(vt)] = vt
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    return axes


def _rotation_to_z(direction: np.ndarray, sign: float = -1.0) -> np.ndarray:
    """Rotation matrix taking `direction` to sign * z."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    target = np.array([0.0, 0.0, sign])
    v = np.cross(d, target)
    c = float(np.dot(d, target))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _tilt_anchor_lowest(coords: np.ndarray, anchor: int, clearance: float = 0.0) -> np.ndarray:
    """The smallest tilt about the anchor atom that leaves every other atom at least `clearance`
    A above it (an azole with its lone pair straight down has the neighbouring C-H lower than
    the N; the tilt lets the N, not the H, meet the surface first).  A grid over tilt axes in
    the surface plane and tilt angles; the first angle that clears is used.  Where no tilt up to
    90 degrees clears (an amine with protruding alkyl H), the lone pair stays straight down."""
    out = coords - coords[anchor]
    others = np.delete(np.arange(len(out)), anchor)
    if len(others) == 0 or out[others, 2].min() >= clearance - 1e-9:
        return coords
    for angle in np.radians(np.arange(1.0, 91.0, 1.0)):
        for psi in np.radians(np.arange(0.0, 360.0, 10.0)):
            rot = _axis_rotation([math.cos(psi), math.sin(psi), 0.0], angle)
            cand = out @ rot.T
            if cand[others, 2].min() >= clearance - 1e-9:
                return cand + coords[anchor]
    return coords


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation matrix about a unit axis."""
    k = np.asarray(axis, dtype=float)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * kx @ kx


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation matrix (from a random unit quaternion)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def lone_pair_direction(mol: Chem.Mol, coords: np.ndarray, anchor: int) -> np.ndarray:
    """Where the anchor atom's lone pair points: away from the mean of its bonded neighbours
    (the centroid direction when it has none)."""
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(anchor).GetNeighbors()]
    if not nbrs:
        return coords[anchor] - coords.mean(axis=0)
    return coords[anchor] - coords[nbrs].mean(axis=0)


def orient(coords: np.ndarray, label: str, anchor: Optional[int], rng: np.random.Generator,
           mol: Optional[Chem.Mol] = None) -> np.ndarray:
    """The molecule rotated about its centroid into the starting orientation of a pose:
    flat - the plane of the two longest principal axes is the xy plane, the longest axis along x;
    upright - the anchor atom's lone pair points down (-z; the atom itself when `mol` is not
    given), the molecule turned about z so that its longest extent lies along x;
    random - a random rotation."""
    centred = coords - coords.mean(axis=0)
    if label == "flat":
        return centred @ principal_axes(centred).T
    if label == "upright":
        if anchor is None:
            raise ValueError("no anchor atom for the upright pose")
        direction = lone_pair_direction(mol, centred, anchor) if mol is not None else centred[anchor]
        rot = _rotation_to_z(direction, sign=-1.0)
        out = _tilt_anchor_lowest(centred @ rot.T, anchor)
        xy = out[:, :2] - out[:, :2].mean(axis=0)
        if len(out) > 1 and np.abs(xy).max() > 1e-6:
            _, _, vt = np.linalg.svd(xy, full_matrices=False)
            cos_t, sin_t = vt[0]
            r2 = np.array([[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
            out = out @ r2.T
        return out - out[anchor] * [1.0, 1.0, 0.0]        # the anchor over the centre of the cell
    if label == "random":
        return centred @ random_rotation(rng).T
    raise ValueError(f"unknown pose {label!r}")


def footprint(oriented: np.ndarray) -> Tuple[float, float]:
    """(x extent, y extent) of an oriented molecule, A."""
    span = oriented.max(axis=0) - oriented.min(axis=0)
    return float(span[0]), float(span[1])


def place_on_slab(slab: Atoms, mol: Chem.Mol, oriented: np.ndarray, height: float) -> Atoms:
    """Slab + molecule: the oriented molecule with its xy origin over the centre of the cell, its
    lowest heavy atom `height` A above the top layer (hydrogens may start lower; the docking
    stage lets them move).  The molecule's atoms follow the slab's, in RDKit order."""
    cell = slab.cell.lengths()
    z_top = top_layer_z(slab)
    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    pos = oriented.copy()
    pos[:, 0] += cell[0] / 2.0
    pos[:, 1] += cell[1] / 2.0
    pos[:, 2] += z_top + height - (pos[heavy, 2].min() if heavy.any() else pos[:, 2].min())
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    system = slab.copy()
    system.extend(Atoms(symbols, positions=pos))
    system.set_constraint(slab.constraints)
    magmoms = np.zeros(len(system))
    magmoms[:len(slab)] = slab.get_initial_magnetic_moments()
    system.set_initial_magnetic_moments(magmoms)
    system.info = dict(slab.info)
    return system


# ------------------------------------------------------------------------------------ relaxation

def relax(atoms: Atoms, calc, fmax: float = 0.05, steps: int = 300, optimizer: str = "bfgs",
          logfile=None, trajectory=None) -> Tuple[float, int, bool]:
    """Relax in place; (energy, number of steps, converged)."""
    atoms.calc = calc
    opt = OPTIMIZERS[optimizer](atoms, logfile=logfile, trajectory=trajectory)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return float(atoms.get_potential_energy()), int(opt.nsteps), converged


def dock_and_relax(system: Atoms, n_slab: int, mol: Chem.Mol, calc, fmax: float = 0.05, steps: int = 300,
                   optimizer: str = "bfgs", dock_steps: int = 40, dock_fmax: float = 0.1,
                   logfile=None) -> Tuple[float, int, bool]:
    """Two stages: first the heavy atoms of the molecule keep their height (they move in the
    surface plane only; hydrogens and everything else are free) for up to `dock_steps` steps,
    so the molecule settles into the chemisorption region instead of drifting off a surface
    that, for a potential without dispersion, has no physisorption well; then everything the
    slab constraint allows is free.  Returns the energy and the total steps of both stages and
    whether the free stage converged."""
    from ase.constraints import FixedPlane
    base = list(system.constraints)
    heavy = [n_slab + a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    n_dock = 0
    if dock_steps > 0 and heavy:
        system.set_constraint(base + [FixedPlane(heavy, [0.0, 0.0, 1.0])])
        _, n_dock, _ = relax(system, calc, dock_fmax, dock_steps, optimizer, logfile=logfile)
        system.set_constraint(base)
    e, n, conv = relax(system, calc, fmax, steps, optimizer, logfile=logfile)
    return e, n + n_dock, conv


def bonds_intact(mol: Chem.Mol, start: np.ndarray, final: np.ndarray, stretch: float = 1.3) -> bool:
    """No bond of the RDKit graph longer than `stretch` times its starting length."""
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d0 = np.linalg.norm(start[i] - start[j])
        d1 = np.linalg.norm(final[i] - final[j])
        if d1 > stretch * max(d0, 0.5):
            return False
    return True


def adsorbed_molecule(mol: Chem.Mol, system: Atoms, n_slab: int) -> Chem.Mol:
    """The molecule's atoms of a relaxed slab+molecule system as an RDKit molecule in the surface
    frame: z = 0 at the (relaxed) top Fe layer, x = y = 0 at the cell centre."""
    cell = system.cell.lengths()
    pos = system.positions[n_slab:].copy()
    pos[:, 0] -= cell[0] / 2.0
    pos[:, 1] -= cell[1] / 2.0
    pos[:, 2] -= top_layer_z(system)
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i, xyz in enumerate(pos):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    return out


# ------------------------------------------------------------------------------- calculators

def uma_calculator(model: str = "uma-s-1p2p1", task: str = "oc20", device: str = "cuda",
                   checkpoint: Optional[str] = None):
    """The fairchem ASE calculator for a UMA model and task head (imports fairchem lazily).
    `checkpoint` is a downloaded .pt file, used instead of the Hugging Face download of `model`."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    if checkpoint:
        predictor = pretrained_mlip.load_predict_unit(checkpoint, device=device)
    else:
        predictor = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predictor, task_name=task)


def gpu_memory_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 2 ** 20
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------------- the run

def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if not r.get("smiles"):
            raise ValueError(f"{path}: every row needs a 'smiles' column")
    return rows


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "mol"


def clean_slab_energy(nx: int, ny: int, calc, work: str, args) -> Tuple[Atoms, float, int]:
    """The relaxed clean slab for a cell size and its energy, cached in `work` as JSON + xyz."""
    tag = f"slab_{nx}x{ny}x{args.layers}_fixed{args.fixed_layers}"
    rec_path = os.path.join(work, tag + ".json")
    xyz_path = os.path.join(work, tag + ".xyz")
    if os.path.exists(rec_path) and os.path.exists(xyz_path) and not args.fresh:
        from ase.io import read
        rec = json.load(open(rec_path))
        slab = read(xyz_path)
        slab.set_constraint(FixAtoms(mask=slab.get_tags() > args.layers - args.fixed_layers))
        slab.set_initial_magnetic_moments([FE_MAGMOM] * len(slab))
        slab.info.update(nx=nx, ny=ny, layers=args.layers, a=args.lattice, vacuum=args.vacuum)
        print(f"[slab] {tag}: E = {rec['energy']:.4f} eV (cached)", flush=True)
        return slab, rec["energy"], rec["n_steps"]
    slab = build_fe110_slab(nx, ny, args.layers, args.lattice, args.vacuum, args.fixed_layers)
    t0 = time.time()
    e, n, conv = relax(slab, calc, args.fmax, args.steps, args.optimizer,
                       logfile=os.path.join(work, tag + ".log"))
    print(f"[slab] {tag}: {len(slab)} Fe, E = {e:.4f} eV, {n} steps, converged={conv}, "
          f"{time.time() - t0:.0f} s", flush=True)
    from ase.io import write
    write(xyz_path, slab)
    json.dump({"energy": e, "n_steps": n, "converged": conv, "n_atoms": len(slab),
               "cell": slab.cell.lengths().tolist()}, open(rec_path, "w"), indent=1)
    return slab, e, n


def process_molecule(row: Dict[str, str], index: int, calc, work: str, args,
                     slab_cache: Dict[Tuple[int, int], Tuple[Atoms, float]]) -> Dict:
    """One molecule: gas-phase reference, the poses, the choice; returns the record."""
    name = row.get("name") or f"mol_{index + 1}"
    t_start = time.time()
    mol = embed_smiles(row["smiles"], seed=args.seed + index)
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    anchor = anchor_atom(mol)
    coords0 = mol_coords(mol)
    rng = np.random.default_rng(args.seed + index)

    gas = molecule_atoms(mol, box=args.box)
    e_mol, n_mol, conv_mol = relax(gas, calc, args.fmax, args.steps, args.optimizer,
                                   logfile=os.path.join(work, safe_name(name) + "_gas.log"))
    gas_coords = gas.positions - gas.positions.mean(axis=0)
    gas_intact = bonds_intact(mol, coords0, gas.positions)
    if not gas_intact:
        print(f"  [{name}] warning: gas-phase relaxation changed the bonding; E_mol and E_ads are unreliable",
              flush=True)

    # the cell: the flat footprint (plus margin) must fit between the periodic images
    flat = orient(gas_coords, "flat", anchor, rng, mol)
    nx, ny = (args.nx, args.ny) if not args.auto_cell else cell_for_footprint(
        footprint(flat), args.lattice, args.nx, args.ny, args.margin)
    if (nx, ny) not in slab_cache:
        slab, e_slab, _ = clean_slab_energy(nx, ny, calc, work, args)
        slab_cache[(nx, ny)] = (slab, e_slab)
    slab, e_slab = slab_cache[(nx, ny)]

    poses = []
    for k, label in enumerate(args.poses):
        if label == "upright" and anchor is None:
            print(f"  [{name}] no anchor atom: the upright pose becomes a random one", flush=True)
            label = "random"
        try:
            oriented = orient(gas_coords, label, anchor, rng, mol)
        except ValueError as exc:
            print(f"  [{name}] {label}: {exc}", flush=True)
            continue
        height = args.height_upright if label == "upright" else args.height
        system = place_on_slab(slab, mol, oriented, height)
        start = system.positions[len(slab):].copy()
        pose_tag = f"{safe_name(name)}_{k}_{label}"
        t0 = time.time()
        e, n, conv = dock_and_relax(system, len(slab), mol, calc, args.fmax, args.steps, args.optimizer,
                                    args.dock_steps, args.dock_fmax,
                                    logfile=os.path.join(work, pose_tag + ".log"))
        final = system.positions[len(slab):]
        z_top = top_layer_z(system)
        heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
        height_final = float(final[heavy, 2].min() - z_top) if heavy.any() else float(final[:, 2].min() - z_top)
        intact = bonds_intact(mol, start, final)
        bound = height_final < args.desorbed
        e_ads = e - e_slab - e_mol
        from ase.io import write
        write(os.path.join(work, pose_tag + ".xyz"), system)
        poses.append({"label": label, "energy": e, "e_ads_eV": e_ads, "n_steps": n, "converged": conv,
                      "intact": intact, "bound": bound, "height": height_final, "elapsed_s": time.time() - t0,
                      "molblock": Chem.MolToMolBlock(adsorbed_molecule(mol, system, len(slab)))})
        print(f"  [{name}] {label:8s} E_ads = {e_ads:8.3f} eV  steps={n:3d} conv={conv!s:5s} "
              f"intact={intact!s:5s} bound={bound!s:5s} h={height_final:.2f} A  {time.time() - t0:.0f} s", flush=True)

    if not poses:
        raise RuntimeError(f"{name}: no pose could be relaxed")
    good = [p for p in poses if p["intact"]] or poses
    best = min(good, key=lambda p: p["energy"])
    rec = {"index": index, "name": name, "smiles": row["smiles"], "charge": charge,
           "anchor_atom": anchor, "cell": [nx, ny, args.layers], "e_slab_eV": e_slab,
           "e_mol_eV": e_mol, "n_steps_gas": n_mol, "gas_intact": gas_intact, "poses": poses, "best": best["label"],
           "best_index": poses.index(best),
           "e_ads_eV": best["e_ads_eV"], "e_ads_kJmol": best["e_ads_eV"] * EV_TO_KJMOL,
           "elapsed_s": time.time() - t_start, "gpu_mb": gpu_memory_mb(), "row": row}
    return rec


def write_outputs(records: Sequence[Dict], out_sdf: str, out_csv: str, columns: Sequence[str],
                  args) -> None:
    """The SDF of the chosen poses and the CSV summary, in input order."""
    recs = sorted(records, key=lambda r: r["index"])
    writer = Chem.SDWriter(out_sdf)
    for r in recs:
        best = r["poses"][r["best_index"]]
        m = Chem.MolFromMolBlock(best["molblock"], removeHs=False)
        m.SetProp("_Name", r["name"])
        m.SetProp("E_ads_eV", f"{r['e_ads_eV']:.4f}")
        m.SetProp("E_ads_kJmol", f"{r['e_ads_kJmol']:.2f}")
        m.SetProp("pose_label", r["best"])
        m.SetProp("n_steps", str(best["n_steps"]))
        for c in columns:
            if c != "name" and r["row"].get(c, "") != "":
                m.SetProp(c, r["row"][c])
        m.SetProp("charge", str(r["charge"]))
        m.SetProp("E_slab_eV", f"{r['e_slab_eV']:.4f}")
        m.SetProp("E_mol_eV", f"{r['e_mol_eV']:.4f}")
        m.SetProp("poses_eV", ";".join(_pose_text(p) for p in r["poses"]))
        m.SetProp("bound", str(int(best.get("bound", True))))
        m.SetProp("gas_intact", str(int(r.get("gas_intact", True))))
        m.SetProp("cell", "x".join(str(v) for v in r["cell"]))
        m.SetProp("model", f"{args.model}/{args.task}")
        writer.write(m)
    writer.close()
    extra = [c for c in columns if c not in ("name", "smiles")]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "smiles", "E_ads_eV", "E_ads_kJmol", "pose_label", "n_steps", "intact", "bound",
                    "gas_intact", "height_A", "E_slab_eV", "E_mol_eV", "cell", "poses_eV", "time_s"] + extra)
        for r in recs:
            best = r["poses"][r["best_index"]]
            w.writerow([r["name"], r["smiles"], f"{r['e_ads_eV']:.4f}", f"{r['e_ads_kJmol']:.2f}", r["best"],
                        best["n_steps"], int(best["intact"]), int(best.get("bound", True)),
                        int(r.get("gas_intact", True)), f"{best['height']:.2f}",
                        f"{r['e_slab_eV']:.4f}", f"{r['e_mol_eV']:.4f}", "x".join(str(v) for v in r["cell"]),
                        ";".join(_pose_text(p) for p in r["poses"]),
                        f"{r['elapsed_s']:.0f}"] + [r["row"].get(c, "") for c in extra])


def _pose_text(p: Dict) -> str:
    """'label:E_ads' with '!' for broken bonds and '~' for a pose that ended unbound."""
    return f"{p['label']}:{p['e_ads_eV']:.3f}" + ("" if p["intact"] else "!") + ("" if p.get("bound", True) else "~")


def run(rows: List[Dict[str, str]], out_sdf: str, calc, args, columns: Optional[Sequence[str]] = None,
        log: Callable[[str], None] = print) -> List[Dict]:
    """The whole run: the records of every molecule, the SDF and CSV rewritten after each one."""
    out_csv = os.path.splitext(out_sdf)[0] + ".csv"
    work = os.path.splitext(out_sdf)[0] + "_work"
    os.makedirs(work, exist_ok=True)
    columns = list(columns if columns is not None else rows[0].keys())
    slab_cache: Dict[Tuple[int, int], Tuple[Atoms, float]] = {}
    slab, e_slab, _ = clean_slab_energy(args.nx, args.ny, calc, work, args)     # the default cell first
    slab_cache[(args.nx, args.ny)] = (slab, e_slab)
    records: List[Dict] = []
    t_run = time.time()
    for index, row in enumerate(rows):
        name = row.get("name") or f"mol_{index + 1}"
        rec_path = os.path.join(work, safe_name(name) + ".json")
        if os.path.exists(rec_path) and not args.fresh:
            rec = json.load(open(rec_path))
            if rec.get("smiles") == row["smiles"]:
                rec["index"], rec["row"] = index, row
                records.append(rec)
                log(f"[{index + 1}/{len(rows)}] {name}: E_ads = {rec['e_ads_eV']:.3f} eV ({rec['best']}, cached)")
                write_outputs(records, out_sdf, out_csv, columns, args)
                continue
        log(f"[{index + 1}/{len(rows)}] {name}  {row['smiles']}")
        try:
            rec = process_molecule(row, index, calc, work, args, slab_cache)
        except Exception as exc:              # one failed molecule must not stop the node
            log(f"  [{name}] FAILED: {exc!r}")
            continue
        json.dump(rec, open(rec_path, "w"), indent=1)
        records.append(rec)
        log(f"  [{name}] E_ads = {rec['e_ads_eV']:.3f} eV = {rec['e_ads_kJmol']:.1f} kJ/mol ({rec['best']}), "
            f"{rec['elapsed_s']:.0f} s, GPU peak {rec['gpu_mb'] or 0:.0f} MB, "
            f"{(time.time() - t_run) / 60:.1f} min elapsed")
        write_outputs(records, out_sdf, out_csv, columns, args)
    return records


def parse_cell(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[x,]\s*(\d+)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError("--cell takes NxM, e.g. 6x6")
    return int(m.group(1)), int(m.group(2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="input CSV with a 'smiles' column (and 'name'; other columns are kept)")
    p.add_argument("-o", "--out", required=True, help="output SDF; the CSV summary and the work dir go next to it")
    p.add_argument("--cell", type=parse_cell, default=(6, 6), help="surface cell NxM (default 6x6)")
    p.add_argument("--no-auto-cell", dest="auto_cell", action="store_false",
                   help="never grow the cell for a large molecule")
    p.add_argument("--margin", type=float, default=6.0, help="A between periodic images of the molecule (6)")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--fixed-layers", type=int, default=None,
                   help="slab layers held fixed, counted from the bottom (default: all; see the docstring)")
    p.add_argument("--lattice", type=float, default=2.78,
                   help="bcc Fe lattice constant, A (2.78 = the oc20 head's minimum; PBE 2.83)")
    p.add_argument("--vacuum", type=float, default=15.0)
    p.add_argument("--poses", default="flat,upright,random",
                   help="comma-separated pose labels among flat, upright, random (repeats allowed)")
    p.add_argument("--height", type=float, default=2.2,
                   help="starting height of the lowest heavy atom in the flat/random poses, A (2.2)")
    p.add_argument("--height-upright", type=float, default=2.0, help="starting height of the anchor atom, A")
    p.add_argument("--dock-steps", type=int, default=40,
                   help="steps of the docking stage, heavy atoms held at their height (0 = none)")
    p.add_argument("--dock-fmax", type=float, default=0.1)
    p.add_argument("--desorbed", type=float, default=4.0,
                   help="a pose whose heavy atoms all end above this height (A) is unbound")
    p.add_argument("--box", type=float, default=20.0, help="the gas-phase box, A")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="bfgs")
    p.add_argument("--model", default="uma-s-1p2p1", help="UMA checkpoint (uma-s-1p2p1, uma-s-1p1, uma-m-1p1)")
    p.add_argument("--checkpoint", default=None,
                   help="a downloaded UMA .pt file, instead of fetching --model from Hugging Face")
    p.add_argument("--task", default="oc20")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="only the first N molecules")
    p.add_argument("--select", default=None, help="comma-separated names (case-insensitive) to run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fresh", action="store_true", help="recompute molecules that have a record")
    return p


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    """Parsed options -> the fields `run` reads (nx, ny, the pose list)."""
    args.nx, args.ny = args.cell
    if args.fixed_layers is None:
        args.fixed_layers = args.layers
    if isinstance(args.poses, str):
        args.poses = [s.strip().lower() for s in args.poses.split(",") if s.strip()]
    bad = [s for s in args.poses if s not in POSE_LABELS]
    if bad:
        raise ValueError(f"unknown pose labels {bad}; choose among {POSE_LABELS}")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = prepare_args(build_parser().parse_args(argv))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    rows = read_rows(args.csv)
    columns = list(rows[0].keys())
    if args.select:
        wanted = {s.strip().lower() for s in args.select.split(",")}
        rows = [r for r in rows if (r.get("name") or "").lower() in wanted]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("no molecules selected", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"{len(rows)} molecules from {args.csv}; model {args.model}/{args.task} on {args.device}; "
          f"cell {args.nx}x{args.ny}x{args.layers}, a = {args.lattice} A, poses {args.poses}, "
          f"fmax {args.fmax}, steps {args.steps}, {args.optimizer}", flush=True)
    t0 = time.time()
    calc = uma_calculator(args.model, args.task, args.device, args.checkpoint)
    print(f"model loaded in {time.time() - t0:.0f} s", flush=True)
    records = run(rows, args.out, calc, args, columns)
    print(f"done: {len(records)}/{len(rows)} molecules in {(time.time() - t0) / 60:.1f} min -> {args.out}", flush=True)
    return 0 if len(records) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
