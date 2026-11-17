"""
Substituted cobalt and iron porphyrins: do fields predict HER and ORR descriptors?

The series is computed by ``examples/make_porphyrin_examples.py`` (see its text for the method
and its limits). Each member is a porphyrin, an aza-porphyrin or the phthalocyanine with Co or
Fe in the ring. The targets are free energies of adsorption of H, OH, O and OOH on the metal and
the oxygen-reduction limiting potential computed from the last three.

The sample is the bare complex. The target belongs to the complex together with its adsorbate,
but a catalyst is described before it meets the adsorbate, so the adsorbate is not part of the
description.

What is compared
----------------
All models use the same folds and the same ridge search inside each training part (as in
``run_metal.py``). Each metal is a separate series, because the metals differ much more than the
substituents do.

``composition``
    Element counts, size and the first coordination sphere. No geometry of the ligand.
``parts``
    For each substituent, how many are on meso carbons and how many on beta carbons, plus the
    number of nitrogen atoms in the meso positions and a flag for the phthalocyanine. This is the
    reference that a one-hot description of the parts gives. It cannot say anything about a
    substituent that is not in the training part.
``hammett``
    The sum of the Hammett constants (sigma-para, Hansch, Leo and Taft, Chem. Rev. 91 (1991)
    165) of the substituents on meso carbons, the same sum for beta carbons, the number of meso
    nitrogens and the phthalocyanine flag. This is the standard chemical reference.
``invariant``
    The kernel of the autocorrelation of the fields. It needs no frame.
``overlap-core``
    The overlap kernel of the five fields with the frame of the macrocycle core: the 24 atoms of
    the four pyrrole units and the four meso atoms are matched to the unsubstituted complex,
    in the one of the 8 equivalent numberings that puts the substituents first (see
    :func:`core_frame`).
``overlap-rule``
    The same kernel with the frame of the principal axes of the whole complex.
``overlap-raw``
    The same kernel with the coordinates as the file has them, only moved to put the metal at the
    origin. It is the control that shows what a frame is worth.

Folds
-----
``random-0``, ``random-1``, ``random-2``
    Five folds from three different seeds. A gain that appears in one seed only is a property of
    the folds, not of the model.
``leave-one-out``
    Every complex is predicted from all the others.
``substituent-out``
    All complexes that carry a given substituent are held out together, so that the test part
    contains substituents that the training part has never seen. A reference that counts parts
    has nothing to say about a new substituent. This design asks whether a model can use the
    chemistry of a group it has not met.

Usage
-----
    python benchmarks/run_porphyrin.py                # all stages
    python benchmarks/run_porphyrin.py --stage data   # only the data description
    python benchmarks/run_porphyrin.py --report       # only rebuild results/porphyrin/table.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import Chem, RDLogger                                            # noqa: E402

import run_metal as RM                                                      # noqa: E402

from cmfqsar.complexes import (_kabsch, _set_positions, build_complex,      # noqa: E402
                               principal_metal_frame)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "porphyrin")
EXAMPLES = os.path.join(ROOT, "examples", "porphyrin")
TABLE = os.path.join(EXAMPLES, "porphyrins.csv")
STRUCTURES = os.path.join(EXAMPLES, "structures.extxyz")

METALS = ("Co", "Fe")
TARGETS = ("dG_H", "dG_OH", "dG_O", "dG_OOH", "U_L_ORR")
MODELS = ("composition", "parts", "hammett", "invariant", "overlap-core", "overlap-rule",
          "overlap-raw")
MESO = (5, 10, 15, 20)

#: Hammett sigma-para constants, Hansch C., Leo A., Taft R. W., Chem. Rev. 91 (1991) 165-195
SIGMA_PARA = {"F": 0.06, "Cl": 0.23, "Br": 0.23, "CH3": -0.17, "CF3": 0.54, "CN": 0.66,
              "NO2": 0.78, "OCH3": -0.27, "NH2": -0.66, "OH": -0.37, "CHO": 0.42, "CCH": 0.23,
              "Ph": -0.01, "NMe2": -0.83, "CO2Me": 0.45}

PAIRS = {
    "overlap-core vs parts": ("parts", "overlap-core"),
    "overlap-core vs hammett": ("hammett", "overlap-core"),
    "overlap-core vs invariant": ("invariant", "overlap-core"),
    "overlap-core vs overlap-raw": ("overlap-raw", "overlap-core"),
    "overlap-core vs overlap-rule": ("overlap-rule", "overlap-core"),
    "invariant vs hammett": ("hammett", "invariant"),
    "hammett vs parts": ("parts", "hammett"),
    "hammett vs composition": ("composition", "hammett"),
}

COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Cl": 1.02, "Br": 1.20}


# ------------------------------------------------------------------------------- the data

def read_structures(path: str = STRUCTURES) -> Dict[str, dict]:
    """The bare complexes of the example file: {name: {symbols, xyz, multiplicity}}."""
    out: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        n = int(lines[i])
        head = lines[i + 1]
        body = lines[i + 2:i + 2 + n]
        i += 2 + n
        name = re.search(r'name="([^"]+)"', head).group(1)
        ads = re.search(r'adsorbate="([^"]+)"', head).group(1)
        if ads != "none":
            continue
        mult = int(re.search(r"multiplicity=(\d+)", head).group(1))
        sym = [b.split()[0] for b in body]
        xyz = np.array([[float(v) for v in b.split()[1:4]] for b in body])
        out[name] = {"symbols": sym, "xyz": xyz, "multiplicity": mult}
    return out


def read_table(path: str = TABLE) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_substituents(text: str) -> List[Tuple[int, str]]:
    return [(int(a), b) for a, b in (t.split(":") for t in text.split(";") if t)]


# ------------------------------------------------------------------------------- the frame

def _bonds(sym: Sequence[str], xyz: np.ndarray, skip: int) -> List[set]:
    """Neighbour sets by distance (1.25 times the sum of covalent radii), metal left out."""
    n = len(sym)
    adj = [set() for _ in range(n)]
    for i in range(n):
        if i == skip:
            continue
        for j in range(i + 1, n):
            if j == skip:
                continue
            if np.linalg.norm(xyz[i] - xyz[j]) < 1.25 * (COVALENT[sym[i]] + COVALENT[sym[j]]):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _fragment_size(adj: List[set], start: int, blocked: set) -> int:
    seen, stack = {start}, [start]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in seen and b not in blocked:
                seen.add(b)
                stack.append(b)
    return len(seen)


def pyrrole_units(sym: Sequence[str], xyz: np.ndarray, metal: int):
    """The four pyrrole units of a metallo-porphyrin, as dicts of atom indices.

    A unit is the inner nitrogen N, the two alpha atoms next to it, the two beta atoms and, for
    each alpha atom, the meso atom outside the five-membered ring. Raises ValueError if the
    complex is not a porphyrin-like macrocycle.
    """
    adj = _bonds(sym, xyz, metal)
    d = np.linalg.norm(xyz - xyz[metal], axis=1)
    inner = [i for i in np.argsort(d) if sym[i] == "N"][:4]
    units = []
    for n in inner:
        found = None
        for a1 in adj[n]:
            for x in adj[a1] - {n}:
                for y in adj[x] - {a1, n}:
                    for a2 in adj[y]:
                        if a2 != x and a2 != n and n in adj[a2] and a2 != a1:
                            found = (a1, x, y, a2)
        if found is None:
            raise ValueError("no five-membered ring through an inner nitrogen")
        a1, b1, b2, a2 = found
        meso1 = [m for m in adj[a1] if m not in (n, b1)]
        meso2 = [m for m in adj[a2] if m not in (n, b2)]
        if len(meso1) != 1 or len(meso2) != 1:
            raise ValueError("an alpha atom has no single meso neighbour")
        units.append({"N": int(n), "a": (int(a1), int(a2)), "b": (int(b1), int(b2)),
                      "meso": (int(meso1[0]), int(meso2[0]))})
    return units, adj


def numberings(units) -> List[List[int]]:
    """The 8 equivalent orderings of the 24 core atoms (4 starting units, 2 directions).

    Each ordering lists, for the four units in turn: N, alpha, beta, beta, alpha, meso, where
    the meso atom is the one that follows the unit in the walk around the ring.
    """
    meso_to_unit: Dict[int, List[Tuple[int, int]]] = {}
    for k, u in enumerate(units):
        for side in (0, 1):
            meso_to_unit.setdefault(u["meso"][side], []).append((k, side))
    out = []
    for start, first in product(range(4), (0, 1)):
        order, k, enter = [], start, first
        for _ in range(4):
            u = units[k]
            exit_side = 1 - enter
            order += [u["N"], u["a"][enter], u["b"][enter], u["b"][exit_side],
                      u["a"][exit_side], u["meso"][exit_side]]
            nxt = [t for t in meso_to_unit[u["meso"][exit_side]] if t[0] != k]
            if len(nxt) != 1:
                raise ValueError("the meso atoms do not close a ring of four units")
            k, enter = nxt[0]
        out.append(order)
    return out


def substituent_signature(order: Sequence[int], sym: Sequence[str], adj: List[set]) -> tuple:
    """For each beta and meso position of a numbering: (element, size of what is attached).

    The size is the number of atoms in the group outside the core, 0 for hydrogen. Two
    numberings are compared by this tuple, so the one that is chosen puts the heavier groups at
    the first positions.
    """
    core = set(order)
    sig = []
    for pos, atom in enumerate(order):
        if pos % 6 in (2, 3, 5):
            out = [b for b in adj[atom] if b not in core and sym[b] != "H"]
            size = sum(_fragment_size(adj, b, core) for b in out)
            sig.append((sym[atom] == "N", len(out), size))
        else:
            sig.append((False, 0, 0))
    return tuple(sig)


def canonical_core(sym, xyz, metal: int):
    units, adj = pyrrole_units(sym, xyz, metal)
    best = None
    for order in numberings(units):
        sig = substituent_signature(order, sym, adj)
        if best is None or sig > best[0]:
            best = (sig, order)
    return best[1]


def core_frame(series: Dict[str, dict]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Coordinates of every complex in the frame of the macrocycle core of the parent.

    The metal goes to the origin. The 24 core atoms, in the numbering of
    :func:`canonical_core`, are matched to those of the unsubstituted complex of the same metal
    with the best rotation (never a reflection). Returns the new coordinates and the root mean
    square distance of the match in angstrom.
    """
    new, rmsd = {}, {}
    ref: Dict[str, np.ndarray] = {}
    for name, rec in series.items():
        metal_sym = name.split("-")[0]
        if name.endswith("-porphine"):
            sym = rec["symbols"]
            m = sym.index(metal_sym)
            order = canonical_core(sym, rec["xyz"], m)
            ref[metal_sym] = rec["xyz"][order] - rec["xyz"][m]
    for name, rec in series.items():
        metal_sym = name.split("-")[0]
        sym = rec["symbols"]
        m = sym.index(metal_sym)
        order = canonical_core(sym, rec["xyz"], m)
        c = rec["xyz"][m]
        a = rec["xyz"][order] - c
        R = _kabsch(a, ref[metal_sym])
        new[name] = (rec["xyz"] - c) @ R
        rmsd[name] = float(np.sqrt(np.mean(np.sum((a @ R - ref[metal_sym]) ** 2, axis=1))))
    return new, rmsd


def to_molecule(name: str, sym: Sequence[str], xyz: np.ndarray) -> Optional[Chem.Mol]:
    mol, report = build_complex(sym, xyz, total_charge=0, name=name)
    if mol is None:
        raise ValueError(f"{name}: {report.fault}")
    mol.SetProp("_Name", name)
    return mol


# ------------------------------------------------------------------------------- the models

def descriptors(rows: List[dict], model: str, mols=None) -> np.ndarray:
    """The columns of the reference models for the complexes in `rows`."""
    subs = [parse_substituents(r["substituents"]) for r in rows]
    aza = [len([t for t in r["aza"].split(";") if t]) for r in rows]
    pc = [1.0 if r["porphyrin"] == "phthalocyanine" else 0.0 for r in rows]
    if model == "hammett":
        out = []
        for s, z, p in zip(subs, aza, pc):
            meso = sum(SIGMA_PARA[x] for pos, x in s if pos in MESO)
            beta = sum(SIGMA_PARA[x] for pos, x in s if pos not in MESO)
            out.append([meso, beta, float(z), p])
        return np.asarray(out, float)
    if model == "parts":
        names = sorted(SIGMA_PARA)
        out = []
        for s, z, p in zip(subs, aza, pc):
            row = [float(sum(1 for pos, x in s if x == n and pos in MESO)) for n in names]
            row += [float(sum(1 for pos, x in s if x == n and pos not in MESO)) for n in names]
            out.append(row + [float(z), p])
        return np.asarray(out, float)
    if model == "composition":
        elements = RM.element_columns(mols)
        return RM.composition_matrix(mols, elements)
    raise ValueError(model)


def prepare(model: str, rows: List[dict], mols: dict, names: List[str]):
    if model in ("hammett", "parts"):
        return ("descriptors", descriptors(rows, model))
    if model == "composition":
        return ("descriptors", descriptors(rows, model, mols["core"]))
    if model == "invariant":
        return RM.invariant_kernels(mols["core"], names)
    if model == "overlap-core":
        return RM.overlap_kernels(mols["core"], names)
    if model == "overlap-rule":
        return RM.overlap_kernels(mols["rule"], names)
    if model == "overlap-raw":
        return RM.overlap_kernels(mols["raw"], names)
    raise ValueError(model)


def group_folds(groups: Sequence[str], k: int = 5, seed: int = 0):
    """Folds that keep each group whole: the first group of a complex decides its fold."""
    uniq = sorted(set(groups))
    order = np.arange(len(uniq))
    np.random.RandomState(seed).shuffle(order)
    where = {}
    for fold, ids in enumerate(np.array_split(order, min(k, len(uniq)))):
        for i in ids:
            where[uniq[i]] = fold
    g = np.array([where[s] for s in groups])
    return [(np.where(g != f)[0], np.where(g == f)[0]) for f in range(g.max() + 1)
            if np.any(g == f) and np.any(g != f)]


def substituent_groups(rows: List[dict]) -> List[str]:
    """One label per complex: its substituent (the most common one if there are several), the
    aza label for an aza-porphyrin without substituents, the phthalocyanine, or the parent."""
    out = []
    for r in rows:
        s = parse_substituents(r["substituents"])
        if r["porphyrin"] == "phthalocyanine":
            out.append("phthalocyanine")
        elif s:
            out.append(sorted(x for _, x in s)[0])
        elif r["aza"]:
            out.append("aza")
        else:
            out.append("parent")
    return out


# ------------------------------------------------------------------------------- the stages

def load_series(metal: str, target: str, drop: Sequence[str] = ()):
    """Rows, names and target values of the complexes of one metal that have the target and a
    structure. `drop` names porphyrins to leave out."""
    rows = [r for r in read_table() if r["metal"] == metal and r[target] != ""
            and r["porphyrin"] not in drop]
    series = read_structures()
    rows = [r for r in rows if r["name"] in series]
    return rows, np.array([float(r[target]) for r in rows])


def build_molecules(rows: List[dict]) -> Tuple[dict, List[str], Dict[str, float]]:
    series = read_structures()
    sub = {r["name"]: series[r["name"]] for r in rows}
    framed, rmsd = core_frame(read_structures())
    names = [r["name"] for r in rows]
    mols = {"core": [], "raw": [], "rule": []}
    for n in names:
        rec = sub[n]
        m = rec["symbols"].index(n.split("-")[0])
        mols["core"].append(to_molecule(n, rec["symbols"], framed[n]))
        mols["raw"].append(to_molecule(n, rec["symbols"], rec["xyz"] - rec["xyz"][m]))
        mols["rule"].append(principal_metal_frame(to_molecule(n, rec["symbols"], rec["xyz"]))[0])
    return mols, names, {n: rmsd[n] for n in names}


def data_stage() -> Dict[str, object]:
    out: Dict[str, object] = {}
    table = read_table()
    series = read_structures()
    framed, rmsd = core_frame(series)
    out["frame_rmsd_mean"] = float(np.mean(list(rmsd.values())))
    out["frame_rmsd_max"] = float(np.max(list(rmsd.values())))
    out["frame_rmsd_worst"] = max(rmsd, key=rmsd.get)
    for metal in METALS:
        rec: Dict[str, object] = {}
        mine = [r for r in table if r["metal"] == metal]
        rec["complexes"] = len(mine)
        for t in TARGETS:
            v = np.array([float(r[t]) for r in mine if r[t] != ""])
            rec[t] = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()),
                      "min": float(v.min()), "max": float(v.max())}
        spins = {}
        for a in ("bare", "H", "OH", "O", "OOH"):
            for r in mine:
                m = r[f"mult_{a}"]
                if m != "":
                    spins.setdefault(a, {}).setdefault(m, 0)
                    spins[a][m] += 1
        rec["spin_states"] = spins
        out[metal] = rec
    return out


def stage_models(metal: str, target: str, models: Sequence[str],
                 drop: Sequence[str] = ()) -> Dict[str, object]:
    rows, y = load_series(metal, target, drop)
    if len(rows) < 8:
        return {"n": len(rows), "skipped": "too few complexes"}
    mols, names, _ = build_molecules(rows)
    designs = {f"random-{s}": RM.random_folds(len(y), 5, s) for s in range(3)}
    designs["leave-one-out"] = RM.leave_one_out(len(y))
    designs["substituent-out"] = group_folds(substituent_groups(rows), 5, 0)
    print(f"  === {metal} {target}: n = {len(y)}, sd {y.std():.3f}", flush=True)
    prepared = {}
    for model in models:
        t0 = time.time()
        prepared[model] = prepare(model, rows, mols, names)
        print(f"    prepared {model} ({time.time() - t0:.0f} s)", flush=True)
    out = {"n": len(y), "y_sd": float(y.std()), "models": list(models), "designs": {}}
    for dname, design in designs.items():
        rec = {"baseline_mae": RM._baseline(y, design), "n_folds": len(design), "models": {},
               "pairs": {}}
        print(f"  [{dname}] {len(design)} folds, baseline MAE {rec['baseline_mae']:.3f} eV",
              flush=True)
        for model in models:
            rec["models"][model] = RM.nested(prepared[model], y, design, f"{dname}/{model}", "eV")
        for label, (a, b) in PAIRS.items():
            rec["pairs"][label] = RM.paired_interval(rec["models"][a]["abs_errors"],
                                                     rec["models"][b]["abs_errors"])
        out["designs"][dname] = rec
    out["names"] = names
    return out


VARIANTS = {"all": (), "no-phthalocyanine": ("phthalocyanine",)}


def models_stage(only_metal: Optional[str] = None, only_target: Optional[str] = None,
                 variant: str = "all"):
    out: Dict[str, object] = {}
    for metal in METALS:
        if only_metal and metal != only_metal:
            continue
        for target in TARGETS:
            if only_target and target != only_target:
                continue
            out[f"{metal}/{target}"] = stage_models(metal, target, MODELS, VARIANTS[variant])
    return out


# ------------------------------------------------------------------------------- the table

def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def write_table(path: Optional[str] = None) -> str:
    path = path or os.path.join(RESULTS, "table.md")
    lines = ["# Substituted Co and Fe porphyrins: models of the HER and ORR descriptors", "",
             "Mean absolute error (MAE, eV; volts for U_L_ORR) with the ridge and the field "
             "width chosen inside each training part. `baseline` predicts the training mean. "
             "Lower is better.", ""]
    dfile = os.path.join(RESULTS, "data.json")
    if os.path.isfile(dfile):
        d = json.load(open(dfile, encoding="utf-8"))
        lines += ["## The data", "",
                  "| metal | complexes | target | n | mean | sd | min | max |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"]
        for metal in METALS:
            rec = d[metal]
            for t in TARGETS:
                s = rec[t]
                lines.append(f"| {metal} | {rec['complexes']} | {t} | {s['n']} | "
                             f"{_fmt(s['mean'], 2)} | {_fmt(s['sd'], 2)} | {_fmt(s['min'], 2)} | "
                             f"{_fmt(s['max'], 2)} |")
        lines += ["", f"Frame of the core: root mean square distance to the parent "
                      f"{_fmt(d['frame_rmsd_mean'], 3)} A on average, at most "
                      f"{_fmt(d['frame_rmsd_max'], 3)} A ({d['frame_rmsd_worst']}).", ""]
    for variant in VARIANTS:
        mfile = os.path.join(RESULTS, "models.json" if variant == "all"
                             else f"models_{variant}.json")
        if not os.path.isfile(mfile):
            continue
        m = json.load(open(mfile, encoding="utf-8"))
        lines += [f"# Models, variant: {variant}", ""]
        for key, rec in m.items():
            if "designs" not in rec:
                continue
            lines += [f"## {key}  (n = {rec['n']}, sd of the target {_fmt(rec['y_sd'], 3)})", "",
                      "| design | baseline | " + " | ".join(rec["models"]) + " |",
                      "|---|---:|" + "---:|" * len(rec["models"])]
            for dname, dr in rec["designs"].items():
                cells = [_fmt(dr["models"][mo]["mae_pooled"]) for mo in rec["models"]]
                best = min(range(len(cells)), key=lambda i: float(cells[i]))
                cells = [f"**{c}**" if i == best else c for i, c in enumerate(cells)]
                lines.append(f"| {dname} | {_fmt(dr['baseline_mae'])} | " + " | ".join(cells)
                             + " |")
            lines += ["", "Paired differences of the absolute error (first model minus second; "
                          "a positive value means the second model is better), with the 95 % "
                          "interval:", "",
                      "| comparison | " + " | ".join(rec["designs"]) + " |",
                      "|---|" + "---:|" * len(rec["designs"])]
            for label in PAIRS:
                cells = []
                for dname in rec["designs"]:
                    p = rec["designs"][dname]["pairs"][label]
                    star = "*" if p["excludes_zero"] else ""
                    cells.append(f"{p['mean']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]{star}")
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            lines.append("")
        lines += ["`*` marks an interval that does not contain zero.", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")
    return path


STAGES = ("data", "models")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", choices=STAGES + ("all",), default="all")
    ap.add_argument("--metal", choices=METALS, default=None, help="models stage: one metal")
    ap.add_argument("--target", choices=TARGETS, default=None, help="models stage: one target")
    ap.add_argument("--variant", choices=tuple(VARIANTS), default=None,
                    help="models stage: one variant (default: every variant)")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        write_table()
        return
    for stage in (STAGES if args.stage == "all" else (args.stage,)):
        print(f"[{stage}]", flush=True)
        t0 = time.time()
        if stage == "data":
            jobs = [("data", data_stage())]
        else:
            jobs = [("models" if v == "all" else f"models_{v}",
                     models_stage(args.metal, args.target, v))
                    for v in ([args.variant] if args.variant else VARIANTS)]
        for fname, res in jobs:
            with open(os.path.join(RESULTS, f"{fname}.json"), "w", encoding="utf-8",
                      newline="\n") as fh:
                json.dump(res, fh, indent=1)
        print(f"  {stage} done in {time.time() - t0:.0f} s", flush=True)
    write_table()


if __name__ == "__main__":
    main()
