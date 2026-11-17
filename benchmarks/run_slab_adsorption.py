#!/usr/bin/env python
"""
Adsorption energies and adsorbed poses of molecules on any slab, with the UMA potential (plan
item BT8).

`examples/uma_fe110_adsorption.py` places molecules on bcc Fe(110) by three pose rules, docks
them, relaxes them with Meta's UMA machine-learned potential and writes the lowest-energy pose
with

    E_ads = E(slab + molecule) - E(slab) - E(molecule alone in a 20 A box).

Everything in that script except the slab is general.  This script runs the same pose rules,
the same docking stage, the same choice of pose and the same output on any slab, so the
adsorption benchmark of plan item EC6 can be made on battery surfaces and the molecules of an
electrolyte can be put on lithium.  It imports the Fe(110) script and replaces only its three
slab-specific functions -- where the top layer is, how large the surface cell must be, and the
clean slab with its energy -- so the pose protocol exists once.

The surfaces
------------
``--surface`` names a recipe or reads a file:

  fe110          bcc Fe(110), orthogonal cell of a x a/sqrt(2) per repeat, a = 2.78 A by
                 default (the minimum of UMA's oc20 head for bcc Fe), 6 x 6 x 4: the slab of the
                 Fe(110) script, built by its own function
  li100          bcc Li(100), square cell of a x a per repeat, 5 x 5 x 4; a = the minimum of the
                 bulk with the same model and task head (``--lattice auto``, the default)
  graphite0001   the basal plane of AB-stacked graphite, orthogonal cell of a x a sqrt(3) per
                 repeat with four atoms in each layer, a = 2.46 A and 3.35 A between layers
                 (the measured lattice; the model has no dispersion, so its own minimum of the
                 interlayer distance means nothing), 7 x 4 x 3
  PATH           a slab from a file ASE reads (extended XYZ, CIF, POSCAR): the surface normal
                 along z, vacuum above the top layer; the cell of the file is the repeat unit

The reference: the molecule far above the slab
---------------------------------------------
UMA's oc20 head was trained on molecules on surfaces, and its energy does not add over two parts
that do not interact: a slab with a molecule 20 A above it has an energy that differs from the
slab plus the same molecule alone in a box by -0.10 to -0.33 eV for the solvents on Li(100) and
graphite (+0.03 to +0.15 eV for three benzimidazoles on Fe(110)), varying with the molecule and
the surface.  So the reference of the Fe(110) script, the molecule alone, carries that offset
into every adsorption energy.  This script therefore also writes

    E_ads(far) = E(slab + molecule) - E(slab + the same molecule, relaxed alone, 20 A above),

the two energies in one cell, made 40 A taller for both, which is free of the offset.  On a
surface the molecule is chemisorbed on (benzimidazole on iron, 1 eV and more) the offset is a
small correction; on a surface it barely touches, such as graphite for a model without
dispersion, it is the whole of E_ads.  Both energies are in ``<out>_far.csv`` and the table.

The layers are found from the heights of the atoms (atoms within 0.6 A in z form one layer),
numbered from the top, and the slab is rigid unless ``--fixed-layers`` frees the upper ones:
on the Fe(110) slab the oc20 head turns the ideal surface into a saddle point, and a rigid slab
keeps the three energies of E_ads consistent.

Usage (in the WSL environment of the Fe(110) script, ``~/uma-env``)
------------------------------------------------------------------
    python benchmarks/run_slab_adsorption.py examples/slab_adsorption/solvents.csv \\
        --surface li100 -o benchmarks/results/slab_adsorption/li100.sdf
    python benchmarks/run_slab_adsorption.py MOLECULES.csv --surface my_facet.extxyz -o OUT.sdf

``--results DIR`` copies the SDF and the CSV there when the run ends, so the work folder (one
geometry and one log per pose) can stay in the cache.  ``--report`` rebuilds
``benchmarks/results/slab_adsorption/table.md`` from the stored CSV files.

Every other option is the Fe(110) script's (``--poses``, ``--height``, ``--dock-steps``,
``--fmax``, ``--steps``, ``--model``, ``--task``, ...); ``--cell`` is in repeats of the surface
unit and is grown for a molecule whose footprint does not fit, as there.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results", "slab_adsorption")
#: the stored Fe(110) run the recipe is checked against
FE110_STORED = os.path.join(ROOT, "examples", "corrosion_fe110", "157_benzimidazole_uma.csv")
sys.path.insert(0, os.path.join(ROOT, "examples"))

import uma_fe110_adsorption as fe  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402

#: atoms whose heights differ by less than this belong to one layer, in angstrom
LAYER_TOLERANCE = 0.6
#: the measured lattice of graphite: in-plane constant and the distance between layers, A
GRAPHITE_A = 2.46
GRAPHITE_SPACING = 3.35


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
def layer_tags(z: np.ndarray, tolerance: float = LAYER_TOLERANCE) -> np.ndarray:
    """Tag 1 for the top layer, 2 for the one below, ...: heights grouped from the top."""
    z = np.asarray(z, float)
    order = np.argsort(-z)
    tags = np.zeros(len(z), int)
    layer, top = 0, None
    for i in order:
        if top is None or top - z[i] > tolerance:
            layer += 1
            top = z[i]
        tags[i] = layer
    return tags


def top_layer_z(atoms: Atoms) -> float:
    """The mean height of the slab's top layer (tag 1), whatever its elements.

    The Fe(110) script averages the tag-1 iron atoms; a molecule's atoms carry tag 0, so the
    slab's top layer is what is averaged here too.
    """
    tags = atoms.get_tags()
    z = atoms.positions[tags == 1, 2]
    if not len(z):
        raise ValueError("the slab has no atom tagged as its top layer")
    return float(z.mean())


def finish_slab(slab: Atoms, fixed_layers: Optional[int], vacuum: float) -> Atoms:
    """Tags by layer, the fixed layers counted from the bottom, the vacuum along z."""
    slab.center(vacuum=vacuum / 2.0, axis=2)
    slab.pbc = (True, True, True)
    tags = layer_tags(slab.positions[:, 2])
    slab.set_tags(tags)
    n_layers = int(tags.max())
    fixed = n_layers if fixed_layers is None else min(int(fixed_layers), n_layers)
    slab.set_constraint(FixAtoms(mask=tags > n_layers - fixed))
    slab.info.pop("adsorbate_info", None)
    slab.info.update(layers=n_layers, fixed_layers=fixed, vacuum=vacuum)
    return slab


# --------------------------------------------------------------------------- #
# The surfaces
# --------------------------------------------------------------------------- #
@dataclass
class Surface:
    """A slab recipe: how to build nx x ny repeats, and the size of one repeat in x and y."""

    name: str
    build: Callable[[int, int, int, float, float, Optional[int]], Atoms]
    unit: Callable[[float], Tuple[float, float]]
    cell: Tuple[int, int]
    layers: int
    lattice: Optional[float]           # None: the minimum of the bulk with the model
    ny_step: int = 1
    bulk: Optional[Callable[[float], Atoms]] = None
    describe: str = ""


def _fe110(nx, ny, layers, a, vacuum, fixed):
    slab = fe.build_fe110_slab(nx, ny, layers, a, vacuum, fixed)
    return finish_slab(slab, fixed, vacuum)


def _li100(nx, ny, layers, a, vacuum, fixed):
    from ase.build import bcc100
    slab = bcc100("Li", size=(nx, ny, layers), a=a, vacuum=vacuum / 2.0, orthogonal=True)
    return finish_slab(slab, fixed, vacuum)


def _li_bulk(a):
    from ase.build import bulk
    return bulk("Li", "bcc", a=a, cubic=True)


def _graphite(nx, ny, layers, a, vacuum, fixed):
    """AB-stacked graphite, orthogonal cell a x a sqrt(3), four atoms per layer and repeat."""
    b = a * math.sqrt(3.0)
    d = a / math.sqrt(3.0)                     # the C-C bond
    layer_a = np.array([[0.0, 0.0], [a / 2, b / 2], [0.0, d], [a / 2, b / 2 + d]])
    layer_b = layer_a + [0.0, d]               # B over A: shifted by one bond along y
    pos, sym = [], []
    for k in range(layers):
        base = layer_a if k % 2 == 0 else layer_b
        for i in range(nx):
            for j in range(ny):
                for x, y in base:
                    pos.append([x + i * a, (y + j * b) % (ny * b), -k * GRAPHITE_SPACING])
                    sym.append("C")
    slab = Atoms(sym, positions=pos, cell=[nx * a, ny * b, 1.0])
    return finish_slab(slab, fixed, vacuum)


SURFACES = {
    "fe110": Surface("fe110", _fe110, lambda a: (a, a / math.sqrt(2.0)), (6, 6), 4, 2.78,
                     ny_step=2, describe="bcc Fe(110)"),
    "li100": Surface("li100", _li100, lambda a: (a, a), (5, 5), 4, None,
                     bulk=_li_bulk, describe="bcc Li(100)"),
    "graphite0001": Surface("graphite0001", _graphite, lambda a: (a, a * math.sqrt(3.0)),
                            (7, 4), 3, GRAPHITE_A, describe="graphite (0001), AB stacking"),
}


def surface_from_file(path: str) -> Surface:
    """A recipe that repeats the slab of a file: its cell is one repeat."""
    from ase.io import read
    unit_slab = read(path)
    a_x, a_y = (float(v) for v in unit_slab.cell.lengths()[:2])

    def build(nx, ny, layers, a, vacuum, fixed):
        slab = unit_slab.repeat((nx, ny, 1))
        slab.set_constraint()
        return finish_slab(slab, fixed, vacuum)

    n_layers = int(layer_tags(unit_slab.positions[:, 2]).max())
    return Surface(os.path.basename(path), build, lambda a: (a_x, a_y), (1, 1), n_layers, 1.0,
                   describe=f"the slab of {os.path.basename(path)}")


def bulk_minimum(surface: Surface, calc, work: str, log=print) -> float:
    """The lattice constant at the minimum of the bulk energy with the model, cached."""
    path = os.path.join(work, f"bulk_{surface.name}.json")
    if os.path.exists(path):
        rec = json.load(open(path))
        log(f"[bulk] {surface.name}: a = {rec['a']:.4f} A (cached)")
        return float(rec["a"])
    from ase.eos import EquationOfState
    guess = surface.bulk(3.5).cell.lengths()[0]
    scan = np.linspace(0.92, 1.08, 9) * guess
    vols, energies = [], []
    for a in scan:
        b = surface.bulk(a)
        b.calc = calc
        vols.append(b.get_volume())
        energies.append(b.get_potential_energy())
    # re-centre the scan on the minimum it found, then fit
    a0 = float(scan[int(np.argmin(energies))])
    scan = np.linspace(0.96, 1.04, 9) * a0
    vols, energies = [], []
    for a in scan:
        b = surface.bulk(a)
        b.calc = calc
        vols.append(b.get_volume())
        energies.append(b.get_potential_energy())
    v0, e0, _ = EquationOfState(vols, energies).fit()
    a = float(scan[0] * (v0 / vols[0]) ** (1.0 / 3.0))
    json.dump({"a": a, "scan": [float(x) for x in scan], "energies": [float(e) for e in energies]},
              open(path, "w"), indent=1)
    log(f"[bulk] {surface.name}: a = {a:.4f} A at the minimum of the model")
    return a


# --------------------------------------------------------------------------- #
# The three functions of the Fe(110) script that know the slab
# --------------------------------------------------------------------------- #
def install(surface: Surface, args) -> None:
    """Make the Fe(110) script's pose code work on `surface`.

    The pose code calls `top_layer_z`, `cell_for_footprint` and `clean_slab_energy` by name, so
    replacing the three names in its module is all it takes; everything else -- the poses, the
    docking, the choice, the records and the output -- runs as written there.
    """
    ux, uy = surface.unit(args.lattice)

    def cell_for_footprint(extent, a=None, nx=None, ny=None, margin=6.0):
        nx = args.nx if nx is None else nx
        ny = args.ny if ny is None else ny
        # the pose rules turn the long axis of the molecule along x, as the Fe(110) script
        # assumes: the x side grows for the long side and the y side for the short one
        long_side, short_side = max(extent), min(extent)
        while nx * ux < long_side + margin:
            nx += 1
        while ny * uy < short_side + margin:
            ny += surface.ny_step
        return nx, ny

    def clean_slab_energy(nx, ny, calc, work, a):
        tag = f"slab_{surface.name}_{nx}x{ny}x{args.layers}_fixed{args.fixed_layers}"
        rec_path = os.path.join(work, tag + ".json")
        slab = surface.build(nx, ny, args.layers, args.lattice, args.vacuum, args.fixed_layers)
        if os.path.exists(rec_path) and not args.fresh:
            rec = json.load(open(rec_path))
            print(f"[slab] {tag}: E = {rec['energy']:.4f} eV (cached)", flush=True)
            return slab, rec["energy"], rec["n_steps"]
        t0 = time.time()
        e, n, conv = fe.relax(slab, calc, args.fmax, args.steps, args.optimizer,
                              logfile=os.path.join(work, tag + ".log"))
        print(f"[slab] {tag}: {len(slab)} atoms, E = {e:.4f} eV, {n} steps, converged={conv}, "
              f"{time.time() - t0:.0f} s", flush=True)
        from ase.io import write
        write(os.path.join(work, tag + ".xyz"), slab)
        json.dump({"energy": e, "n_steps": n, "converged": conv, "n_atoms": len(slab),
                   "cell": slab.cell.lengths().tolist(), "surface": surface.name,
                   "lattice": args.lattice}, open(rec_path, "w"), indent=1)
        return slab, e, n

    fe.top_layer_z = top_layer_z
    fe.cell_for_footprint = cell_for_footprint
    fe.clean_slab_energy = clean_slab_energy


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def build_parser():
    p = fe.build_parser()
    p.description = __doc__.split("\n\n")[0]
    p.add_argument("--results", default=None, metavar="DIR",
                   help="copy the SDF and the CSV to this folder when the run ends")
    p.add_argument("--surface", default="fe110",
                   help="a recipe (" + ", ".join(SURFACES) + ") or a slab file; default fe110")
    for action in p._actions:
        if action.dest == "lattice":
            action.default = None
            action.type = str
            action.help = ("lattice constant of the recipe, A, or 'auto' for the minimum of the "
                           "bulk with the model; default: the recipe's own (auto for li100)")
        elif action.dest == "cell":
            action.default = None
            action.help = "surface cell NxM in repeats of the surface unit (default: the recipe's)"
        elif action.dest == "layers":
            action.default = None
    return p


def resolve(args, calc_factory, work: str, log=print) -> Surface:
    """The surface and the defaults it gives the options that were not set."""
    surface = SURFACES.get(args.surface) or (
        surface_from_file(args.surface) if os.path.exists(args.surface) else None)
    if surface is None:
        raise SystemExit(f"--surface {args.surface!r} is neither a recipe "
                         f"({', '.join(SURFACES)}) nor a file")
    if args.cell is None:
        args.cell = surface.cell
    if args.layers is None:
        args.layers = surface.layers
    lattice = args.lattice if args.lattice is not None else (
        "auto" if surface.lattice is None else surface.lattice)
    if str(lattice).lower() == "auto":
        if surface.bulk is None:
            raise SystemExit(f"--lattice auto needs a bulk for {surface.name}; give a number")
        args.lattice = bulk_minimum(surface, calc_factory(), work, log)
    else:
        args.lattice = float(lattice)
    return surface


# --------------------------------------------------------------------------- #
# The reference far above the slab
# --------------------------------------------------------------------------- #
#: how far above the top layer the reference molecule is put, and how much taller the cell is
FAR_HEIGHT = 20.0
FAR_EXTRA_Z = 40.0


def far_reference(rows, records, calc, work: str, args, out_csv: str, log=print) -> List[dict]:
    """E_ads against the same molecule far above the slab, for every record; cached per molecule.

    The molecule is relaxed alone as the Fe(110) script relaxes it (same embedding seed, same
    settings), put with its lowest heavy atom FAR_HEIGHT above the top layer over the middle of
    the cell, and the cell of both systems is made FAR_EXTRA_Z taller, so the two energies are
    single points in one cell and the molecule does not meet the periodic image of the slab.
    """
    from ase.io import read
    out = []
    by_name = {r["name"]: r for r in records}
    for index, row in enumerate(rows):
        name = row.get("name") or f"mol_{index + 1}"
        rec = by_name.get(name)
        if rec is None:
            continue
        cache = os.path.join(work, fe.safe_name(name) + "_far.json")
        if os.path.exists(cache) and not args.fresh:
            got = json.load(open(cache))
            if got.get("smiles") == row["smiles"]:
                out.append(got)
                continue
        best = rec["poses"][rec["best_index"]]
        path = os.path.join(work, f"{fe.safe_name(name)}_{rec['best_index']}_{best['label']}.xyz")
        system = read(path)
        mol = fe.embed_smiles(row["smiles"], seed=args.seed + index)
        gas = fe.molecule_atoms(mol, box=args.box)
        e_gas, _, _ = fe.relax(gas, calc, args.fmax, args.steps, args.optimizer)
        n_slab = len(system) - len(gas)
        cell = system.cell.array.copy()
        cell[2, 2] += FAR_EXTRA_Z
        system.set_constraint()
        system.set_cell(cell, scale_atoms=False)
        slab = system[:n_slab]
        z_top = float(slab.positions[slab.get_tags() == 1, 2].mean()) if (
            slab.get_tags() == 1).any() else float(slab.positions[:, 2].max())
        pos = gas.positions - gas.positions.mean(axis=0)
        heavy = np.array([s != "H" for s in gas.get_chemical_symbols()])
        pos[:, 0] += cell[0, 0] / 2.0
        pos[:, 1] += cell[1, 1] / 2.0
        pos[:, 2] += z_top + FAR_HEIGHT - pos[heavy if heavy.any() else slice(None), 2].min()
        far = slab.copy()
        far.extend(Atoms(gas.get_chemical_symbols(), positions=pos))
        energies = {}
        for key, atoms in (("pose", system), ("far", far), ("slab", slab)):
            atoms.calc = calc
            energies[key] = float(atoms.get_potential_energy())
        got = {"name": name, "smiles": row["smiles"], "E_ads_eV": rec["e_ads_eV"],
               "E_ads_far_eV": energies["pose"] - energies["far"],
               "offset_eV": energies["far"] - energies["slab"] - e_gas,
               "gas_energy_difference_eV": e_gas - rec["e_mol_eV"]}
        json.dump(got, open(cache, "w"), indent=1)
        log(f"  [{name}] E_ads against the molecule far above: {got['E_ads_far_eV']:+.3f} eV "
            f"(against the molecule alone {rec['e_ads_eV']:+.3f}; offset of the model "
            f"{got['offset_eV']:+.3f})")
        out.append(got)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "smiles", "E_ads_eV", "E_ads_far_eV", "offset_eV",
                    "gas_energy_difference_eV"])
        for g in out:
            w.writerow([g["name"], g["smiles"], f"{g['E_ads_eV']:.4f}", f"{g['E_ads_far_eV']:.4f}",
                        f"{g['offset_eV']:.4f}", f"{g['gas_energy_difference_eV']:.4f}"])
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def report(results_dir: str = RESULTS) -> str:
    L = ["# Molecules on any slab: the generator of the adsorption benchmarks (plan item BT8)\n",
         "Two adsorption energies are given. *Against the molecule alone* is the Fe(110) "
         "script's: E(slab + molecule) - E(slab) - E(molecule relaxed alone in a box). *Against "
         "the molecule far above* is E(slab + molecule) - E(slab + the same relaxed molecule "
         "20 A above), both in a cell 40 A taller, so the two systems have the same atoms in "
         "the same cell. The *offset of the model* is E(slab + molecule far above) - E(slab) - "
         "E(molecule alone): zero for a model whose energy adds over parts that do not "
         "interact. UMA's oc20 head, trained on molecules on surfaces, gives -0.10 to -0.33 eV "
         "for the solvents and +0.03 to +0.15 eV for the benzimidazoles, so the first "
         "energy carries it and the second does not. The pose of every molecule is chosen "
         "against the first, which does not change the choice, since the poses of one "
         "molecule on one surface share it.\n"]
    check = os.path.join(results_dir, "fe110_check.csv")
    if os.path.exists(check) and os.path.exists(FE110_STORED):
        old = {r["name"]: r for r in _rows(FE110_STORED)}
        L.append("## The Fe(110) recipe against the Fe(110) script\n")
        L.append("The first molecules of `examples/corrosion/157_c1ccc2_nH_cnc2c1.csv`, run by "
                 "this "
                 "script with `--surface fe110`, against the energies the Fe(110) script stored "
                 "in `examples/corrosion_fe110/157_benzimidazole_uma.csv`.\n")
        far = _far(os.path.join(results_dir, "fe110_check_far.csv"))
        L.append("| Molecule | Stored E_ads, eV | Now, eV | Difference, meV | Pose, stored and now "
                 "| Against the molecule far above, eV | Offset of the model, eV |")
        L.append("|---|---:|---:|---:|---|---:|---:|")
        for r in _rows(check):
            o = old.get(r["name"])
            if o is None:
                continue
            d = 1000 * (float(r["E_ads_eV"]) - float(o["E_ads_eV"]))
            f = far.get(r["name"], {})
            L.append(f"| {r['name']} | {float(o['E_ads_eV']):+.4f} | {float(r['E_ads_eV']):+.4f} | "
                     f"{d:+.1f} | {o['pose_label']}, {r['pose_label']} | "
                     f"{_num(f.get('E_ads_far_eV'))} | {_num(f.get('offset_eV'))} |")
        L.append("")
    for name, title in (("li100", "Li(100)"), ("graphite0001", "graphite (0001)")):
        path = os.path.join(results_dir, f"{name}.csv")
        if not os.path.exists(path):
            continue
        meta_path = os.path.join(results_dir, f"{name}_run.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        L.append(f"## Electrolyte solvents on {title}\n")
        minutes = sum(float(r.get("time_s") or 0) for r in _rows(path)) / 60
        if meta:
            L.append(f"Lattice constant {meta['lattice']:.4f} A, cell "
                     f"{'x'.join(map(str, meta['cell']))}, model {meta['model']}, "
                     f"{minutes:.0f} minutes of relaxations (the sum over the molecules). "
                     "E_ads < 0 is binding. A pose whose heavy atoms "
                     "all end more than 4 A above the surface is unbound; 'intact' says whether "
                     "every bond of the molecule survived the relaxation.\n")
        far = _far(os.path.join(results_dir, f"{name}_far.csv"))
        L.append("| Solvent | E_ads against the molecule far above, eV | kJ/mol | E_ads against "
                 "the molecule alone, eV | Offset of the model, eV | Pose | Height of the lowest "
                 "heavy atom, A | Bound | Intact | All poses against the molecule alone, eV |")
        L.append("|---|---:|---:|---:|---:|---|---:|---|---|---|")

        def key(r):
            f = far.get(r["name"])
            return float(f["E_ads_far_eV"]) if f else float(r["E_ads_eV"])
        for r in sorted(_rows(path), key=key):
            f = far.get(r["name"], {})
            e_far = f.get("E_ads_far_eV")
            kj = "" if e_far is None else f"{96.485332 * float(e_far):+.1f}"
            L.append(f"| {r['name']} | {_num(e_far)} | {kj} | {float(r['E_ads_eV']):+.3f} | "
                     f"{_num(f.get('offset_eV'))} | {r['pose_label']} | {float(r['height_A']):.2f} "
                     f"| {'yes' if r['bound'] == '1' else 'no'} | "
                     f"{'yes' if r['intact'] == '1' else 'no'} | {r['poses_eV']} |")
        L.append("")
    return "\n".join(L)


def _far(path: str) -> dict:
    return {r["name"]: r for r in _rows(path)} if os.path.exists(path) else {}


def _num(v) -> str:
    return "-" if v in (None, "") else f"{float(v):+.3f}"


def write_table(results_dir: str = RESULTS) -> str:
    path = os.path.join(results_dir, "table.md")
    text = report(results_dir)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--report":
        print(write_table(RESULTS))
        return 0
    args = build_parser().parse_args(argv)
    rows = fe.read_rows(args.csv)
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
    work = os.path.splitext(args.out)[0] + "_work"
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    holder: List = []

    def calc_factory():
        if not holder:
            holder.append(fe.uma_calculator(args.model, args.task, args.device, args.checkpoint))
            print(f"model loaded in {time.time() - t0:.0f} s", flush=True)
        return holder[0]

    surface = resolve(args, calc_factory, work)
    args = fe.prepare_args(args)
    install(surface, args)
    print(f"{len(rows)} molecules from {args.csv}; {surface.describe}, a = {args.lattice:.4f} A, "
          f"cell {args.nx}x{args.ny}x{args.layers}; model {args.model}/{args.task} on "
          f"{args.device}; poses {args.poses}, fmax {args.fmax}, steps {args.steps}, "
          f"{args.optimizer}", flush=True)
    records = fe.run(rows, args.out, calc_factory(), args, columns)
    print(f"done: {len(records)}/{len(rows)} molecules in {(time.time() - t0) / 60:.1f} min -> "
          f"{args.out}", flush=True)
    far_csv = os.path.splitext(args.out)[0] + "_far.csv"
    far_reference(rows, records, calc_factory(), work, args, far_csv)
    if args.results:
        os.makedirs(args.results, exist_ok=True)
        for path in (args.out, os.path.splitext(args.out)[0] + ".csv", far_csv):
            shutil.copy(path, os.path.join(args.results, os.path.basename(path)))
        json.dump({"argv": list(argv) if argv is not None else sys.argv[1:],
                   "surface": surface.name, "lattice": args.lattice,
                   "cell": [args.nx, args.ny, args.layers], "model": f"{args.model}/{args.task}",
                   "written": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(args.results, os.path.splitext(os.path.basename(args.out))[0]
                                    + "_run.json"), "w"), indent=1)
    return 0 if len(records) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
