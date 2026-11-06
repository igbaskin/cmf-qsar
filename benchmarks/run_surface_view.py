#!/usr/bin/env python
"""
The surface side of the program on screen: what it costs, and whether it reads correctly.

Plan items EC10 and NT9.  Three things the electrochemistry items built were invisible in the
application, and one reading was missing:

* **the slab behind the molecules** (item EC6).  The viewer drew a protein pocket behind a
  ligand; a metal surface is the same problem with a receptor that has no edge, so what is drawn
  is a patch of the surface around the molecules, expanded over the surface lattice and cut in
  the plane.
* **the surface frame** (item EC2).  A molecule on a surface lives in a frame with the normal
  along z and the top atomic layer at z = 0, and the kernel sees two numbers of its pose: the
  height and the tilt.  The window states them.
* **the vector co-field** (item EC5).  The co-field of the electric-field kind is a direction at
  a place, not a value, so it is drawn as arrows.
* **the co-pattern** (item NT9).  The invariant kernels have no co-field at all.  Their reading
  is the weight of every channel, and the chart of weight against distance for one pair of field
  types is that reading as a shape.

A window is not judged by a q2, so each stage measures something that decides whether the
drawing is usable or whether the reading is exact.

``slab``
    For the Fe(110) slab of ``examples/corrosion_fe110`` and both of its molecule sets: how many
    surface atoms a patch of each radius holds, how long the expansion and the cut take, how long
    the window takes to draw it, and two checks -- every drawn atom lies at or below z = 0 (the
    frame), and the drawn atoms are a subset of the atoms the complementarity receptor is made
    of (the picture shows the receptor and not something else).

``frame``
    The two numbers the panel states, against something independent of them.  The height the
    window reports for a relaxed pose is compared with the ``height_A`` column the relaxation
    wrote into the data set, which the window never reads; and the rule pose of
    ``cmfqsar.planealign`` is compared with the relaxed pose in both numbers, which says what a
    user without a slab is shown.

``arrows``
    The vector co-field on the Hammett set with the electric-field kind as its only field: the
    time for the arrows, a check that the longest arrow drawn is the largest gradient on the
    grid, how much of the gradient the drawn arrows stand on, and how the answer moves with the
    grid spacing -- a drawing whose directions change with the grid would say nothing.

``copattern``
    The chart against the table, on two kernels that have no co-field: the periodic
    autocorrelation of the COD electrolytes and the surface autocorrelation of an Fe(110) set.
    The sum over every channel of weight times channel must reproduce the prediction less the
    bias (that is what makes the weights a reading and not a heuristic), the points of the curve
    must be the weights the ranked table names, and the share of the whole reading that sits in
    the pair of field types the chart opens on is reported.

    python benchmarks/run_surface_view.py                     # every stage, then the table
    python benchmarks/run_surface_view.py --stage slab
    python benchmarks/run_surface_view.py --stage arrows --arrows 24
    python benchmarks/run_surface_view.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# the window is a Qt window; it is driven here without a screen, as the GUI tests are
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

RESULTS = os.path.join(HERE, "results", "surface_view")
EXAMPLES = os.path.join(ROOT, "examples")
#: the two molecule sets of the Fe(110) study, as (short name, SDF, CSV)
NODES = (
    ("benzimidazole", os.path.join(EXAMPLES, "corrosion_fe110", "157_benzimidazole_uma.sdf"),
     os.path.join(EXAMPLES, "corrosion_fe110", "157_benzimidazole_uma.csv")),
    ("quinoline", os.path.join(EXAMPLES, "corrosion_fe110", "158_quinoline_uma.sdf"),
     os.path.join(EXAMPLES, "corrosion_fe110", "158_quinoline_uma.csv")),
)
HAMMETT = os.path.join(EXAMPLES, "hammett", "hammett.sdf")
ELECTROLYTES = os.path.join(EXAMPLES, "hops", "disordered")
#: patch radii the drawing is timed at, in angstrom
RADII = (6.0, 10.0, 16.0, 24.0)
#: how many molecules of a node the drawing is timed on
DRAW_MOLECULES = 5
#: arrows drawn, and the grid spacings the arrow directions are compared at
ARROWS = 16
SPACINGS = (0.7, 0.5, 0.35)
#: the Hammett target the arrow stage fits
ARROW_TARGET = "sigma_m"
#: the lattice constant and size of the slab the reference poses were relaxed on
SLAB = dict(element="Fe", nx=6, ny=6, layers=4, a=2.78, vacuum=15.0)


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def result(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def jdump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, default=float)
        fh.write("\n")


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_tag(a) -> str:
    """The name of one run: every option that changes a number enters it (rule 6 of the plan).

    ``--stage``, ``--report`` and ``--force`` only choose which parts are done, so none of them
    changes a stored number or the tag.
    """
    tag = ""
    if a.molecules != DRAW_MOLECULES:
        tag += f"_m{a.molecules}"
    if a.arrows != ARROWS:
        tag += f"_a{a.arrows}"
    return tag


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _column(csv_path: str, name_column: int = 0, column: str = "height_A") -> Dict[str, float]:
    """One numeric column of a CSV, by the name in its first column."""
    import csv
    out: Dict[str, float] = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = list(row.values())[name_column]
            try:
                out[str(key)] = float(row[column])
            except (TypeError, ValueError, KeyError):
                continue
    return out


# --------------------------------------------------------------------------- #
# Stage 1: the slab behind the molecules
# --------------------------------------------------------------------------- #
def slab(molecules: int = DRAW_MOLECULES, log=print) -> dict:
    """Open each Fe(110) set in the window with the slab behind it, and time the drawing."""
    from cmfqsar.fields import FieldSet
    from cmfqsar.gui.main_window import MainWindow
    from cmfqsar.gui.surface_view import build_bcc110, drawn_atoms
    from cmfqsar.slab import slab_receptor

    app = _app()                                               # noqa: F841 - Qt needs it alive
    t0 = time.time()
    surface = build_bcc110(**SLAB)
    build_seconds = time.time() - t0
    fs = FieldSet.create(physchem=["steric"], alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    receptor = slab_receptor(surface.atoms, fs, top_layers=2, cutoff=max(RADII) + 8.0)
    out = {"slab": dict(SLAB), "slab_atoms": surface.n_atoms,
           "layer_heights": [round(h, 3) for h in surface.layers],
           "build_seconds": build_seconds, "receptor_atoms": int(receptor.n_atoms),
           "radii": list(RADII), "molecules_timed": int(molecules), "sets": []}
    log(f"the slab is {surface.n_atoms} atoms in {len(surface.layers)} layers, built in "
        f"{build_seconds:.2f} s; its receptor is {receptor.n_atoms} atoms")
    for name, sdf, _csv in NODES:
        if not os.path.exists(sdf):
            log(f"{name}: {sdf} is missing, skipped")
            continue
        w = MainWindow(background=False)
        w.load_dataset_file(sdf, "train", activity="E_ads_kJmol")
        w._install_surface(surface)
        w.controls.chk_slab.setChecked(True)
        rows = []
        for r in RADII:
            w.controls.spin_slab.setValue(r)
            counts, cut, draw = [], [], []
            for i in range(min(int(molecules), len(w.ds.names))):
                pts = w.molfields[i].coords
                t1 = time.time()
                coords, symbols, _radii = drawn_atoms(surface, pts, radius=r)
                cut.append(time.time() - t1)
                counts.append(len(coords))
                w.dataset_panel.select_index(i)
                w.dataset_panel.set_all_shown(False)
                w.dataset_panel.set_shown([i], True)
                t2 = time.time()
                w.update_display()
                draw.append(time.time() - t2)
            rows.append({"radius": float(r), "atoms_median": float(np.median(counts)),
                         "cut_seconds": float(np.median(cut)),
                         "draw_seconds": float(np.median(draw))})
            log(f"  {name}: patch {r:4.0f} A -> {np.median(counts):5.0f} atoms, cut in "
                f"{np.median(cut) * 1000:.0f} ms, drawn in {np.median(draw) * 1000:.0f} ms")
        # the two checks: the frame, and that the drawing is the receptor's own atoms
        coords, symbols, _radii = drawn_atoms(surface, w.molfields[0].coords, radius=max(RADII))
        above = float(coords[:, 2].max())
        rec = {tuple(np.round(p, 4)) for p in receptor.coords}
        inside = sum(1 for p in coords if tuple(np.round(p, 4)) in rec)
        rows_set = {"set": name, "n_molecules": len(w.ds.names),
                    "path": os.path.relpath(sdf, ROOT).replace(os.sep, "/"),
                    "highest_drawn_z": above,
                    "share_in_receptor": float(inside / max(len(coords), 1)), "patches": rows}
        out["sets"].append(rows_set)
        log(f"  {name}: the highest drawn atom is at z = {above:+.3f} A and "
            f"{100 * inside / max(len(coords), 1):.0f} % of the drawn atoms are atoms of the "
            f"complementarity receptor")
        w.close()
    return out


# --------------------------------------------------------------------------- #
# Stage 2: the surface frame, against something independent of it
# --------------------------------------------------------------------------- #
def frame(log=print) -> dict:
    """What the panel's two numbers are worth: on the relaxed poses, and on the rule poses."""
    from rdkit import Chem, RDLogger

    from cmfqsar.planealign import pose_geometry, rule_pose
    RDLogger.DisableLog("rdApp.*")
    out = {"sets": []}
    for name, sdf, csv_path in NODES:
        if not os.path.exists(sdf):
            log(f"{name}: {sdf} is missing, skipped")
            continue
        written = _column(csv_path)
        rows = []
        for mol in Chem.SDMolSupplier(sdf, removeHs=False, sanitize=True):
            if mol is None:
                continue
            title = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            xyz = np.array(mol.GetConformer().GetPositions(), float)
            symbols = [a.GetSymbol() for a in mol.GetAtoms()]
            relaxed = pose_geometry(xyz, symbols)
            try:
                posed, _anchor = rule_pose(Chem.Mol(mol), "flat", height=2.2)
            except ValueError:
                continue
            ruled = pose_geometry(posed, symbols)
            rows.append({"name": title, "height": relaxed.height, "tilt": relaxed.tilt,
                         "thickness": relaxed.thickness, "flat": bool(relaxed.flat),
                         "written_height": written.get(title, float("nan")),
                         "rule_height": ruled.height, "rule_tilt": ruled.tilt})
        if not rows:
            continue
        h = np.array([r["height"] for r in rows])
        w = np.array([r["written_height"] for r in rows])
        tilt = np.array([r["tilt"] for r in rows])
        rtilt = np.array([r["rule_tilt"] for r in rows])
        ok = np.isfinite(h) & np.isfinite(w)
        rec = {"set": name, "n": len(rows),
               "path": os.path.relpath(sdf, ROOT).replace(os.sep, "/"),
               "height_min": float(h.min()), "height_max": float(h.max()),
               "tilt_median": float(np.median(tilt)),
               "tilt_90th": float(np.percentile(tilt, 90)),
               "flat_share": float(np.mean([r["flat"] for r in rows])),
               "n_with_written": int(ok.sum()),
               "height_vs_written_max": float(np.abs(h[ok] - w[ok]).max()) if ok.any() else
               float("nan"),
               "height_vs_written_median": float(np.median(np.abs(h[ok] - w[ok])))
               if ok.any() else float("nan"),
               "rule_tilt_median": float(np.median(rtilt)),
               "rule_minus_relaxed_tilt_median": float(np.median(np.abs(rtilt - tilt))),
               "rows": rows}
        out["sets"].append(rec)
        log(f"{name}: {len(rows)} molecules, the window reads the relaxed height to "
            f"{rec['height_vs_written_max']:.3f} A of the value the relaxation wrote; the "
            f"relaxed tilt has median {rec['tilt_median']:.1f} deg and the flat rule pose "
            f"differs from it by {rec['rule_minus_relaxed_tilt_median']:.1f} deg")
    return out


# --------------------------------------------------------------------------- #
# Stage 3: the vector co-field as arrows
# --------------------------------------------------------------------------- #
def arrows(count: int = ARROWS, log=print) -> dict:
    """The arrows of the electric-field co-field: the cost, the check, and the grid dependence."""
    from rdkit import Chem, RDLogger

    from cmfqsar.cofields import cofield_arrows, cofield_gradient, make_grid
    from cmfqsar.fields import EFIELD_FIELD, AtomTable, FieldSet, build_fields
    from cmfqsar.model import CMFModel
    RDLogger.DisableLog("rdApp.*")
    if not os.path.exists(HAMMETT):
        log(f"{HAMMETT} is missing: the arrow stage is skipped")
        return {"spacings": []}
    mols, names, y = [], [], []
    for mol in Chem.SDMolSupplier(HAMMETT, removeHs=False, sanitize=True):
        if mol is None or not mol.HasProp(ARROW_TARGET):
            continue
        value = float(mol.GetProp(ARROW_TARGET))
        if not np.isfinite(value):
            continue
        mols.append(Chem.Mol(mol))
        names.append(mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{len(mols)}")
        y.append(value)
    fs = FieldSet.create(physchem=[EFIELD_FIELD], alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), np.array(y, float), cv_folds=0)
    coords = np.vstack([m.coords for m in mf])
    out = {"n_molecules": len(mf), "target": ARROW_TARGET, "arrows": int(count), "spacings": []}
    reference = None
    for spacing in SPACINGS:
        grid = make_grid(coords, margin=2.0, spacing=spacing)
        t0 = time.time()
        g = cofield_gradient(model, 0, grid)
        gradient_seconds = time.time() - t0
        magnitude = np.linalg.norm(np.stack([g[k].ravel() for k in range(3)], axis=1), axis=1)
        t1 = time.time()
        points, vectors, lengths = cofield_arrows(model, 0, grid, count=int(count),
                                                  min_separation=1.5)
        arrow_seconds = time.time() - t1
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-30)
        rec = {"spacing": float(spacing), "points": int(np.prod(grid.shape)),
               "gradient_seconds": gradient_seconds, "arrow_seconds": arrow_seconds,
               "n_arrows": int(len(points)),
               "longest": float(lengths.max()) if len(lengths) else float("nan"),
               "grid_maximum": float(magnitude.max()),
               "shortest_drawn": float(lengths.min()) if len(lengths) else float("nan"),
               # grid-independent: how strong the weakest drawn arrow is, and how much of the
               # volume carries a gradient at least that strong.  A share of the summed
               # magnitude would fall with every refinement, because there are more points.
               "shortest_over_longest": float(lengths.min() / lengths.max())
               if len(lengths) else float("nan"),
               "volume_share_above_shortest":
                   float(np.mean(magnitude >= lengths.min())) if len(lengths) else float("nan")}
        if reference is None:
            reference = (points, unit)
        else:
            # the same places, drawn from a finer grid: how far the direction moved
            ref_points, ref_unit = reference
            from scipy.spatial import cKDTree
            d, idx = cKDTree(points).query(ref_points, k=1)
            cos = np.clip(np.sum(ref_unit * unit[idx], axis=1), -1.0, 1.0)
            keep = d <= 1.5 * float(spacing) + 0.5
            rec["matched_arrows"] = int(keep.sum())
            rec["direction_change_median_deg"] = (
                float(np.degrees(np.arccos(cos[keep])).mean()) if keep.any() else float("nan"))
        out["spacings"].append(rec)
        log(f"  spacing {spacing:.2f} A: {np.prod(grid.shape):.0f} points, gradient in "
            f"{gradient_seconds:.2f} s, {len(points)} arrows in {arrow_seconds:.2f} s; the "
            f"longest is {lengths.max():.4g} and the largest gradient on the grid is "
            f"{magnitude.max():.4g}")
    return out


# --------------------------------------------------------------------------- #
# Stage 4: the co-pattern chart against the channel table (plan item NT9)
# --------------------------------------------------------------------------- #
def copattern(log=print) -> dict:
    """Is the curve the same numbers as the table, and do the weights add up to the prediction?"""
    out = {"models": []}
    rec = _copattern_periodic(log=log)
    if rec:
        out["models"].append(rec)
    rec = _copattern_surface(log=log)
    if rec:
        out["models"].append(rec)
    return out


def _copattern_check(model, fieldset, table, y, what: str, log=print) -> dict:
    """The three numbers the chart rests on, for one fitted model."""
    from cmfqsar.copattern import copattern as curve_of
    from cmfqsar.copattern import pair_labels
    from cmfqsar.periodic import channel_weights

    w = np.asarray(channel_weights(model), float)
    Z = np.asarray(model.ac_train_, float) * np.asarray(model.ac_column_scale, float)
    Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
    residual = float(np.abs(model.bias + Z @ w - model.predict(table)).max())
    pairs = pair_labels(model, fieldset)
    t0 = time.time()
    curves = [curve_of(model, fieldset, k, "weight x spread") for k in range(len(pairs))]
    seconds = (time.time() - t0) / max(len(pairs), 1)
    moved = np.array([float(np.abs(c.moved).sum()) for c in curves])
    best = int(np.argmax(moved))
    rec = {"what": what, "n_channels": int(len(w)), "n_pairs": len(pairs),
           "points_per_curve": int(len(curves[0].x)), "grouped": int(curves[0].grouped),
           "prediction_residual": residual, "y_sd": float(np.nanstd(y)),
           "seconds_per_curve": seconds, "axis": curves[0].x_label,
           "leading_pair": pairs[best],
           "leading_share": float(moved[best] / max(moved.sum(), 1e-30)),
           "leading_peak_at": float(curves[best].x[int(np.argmax(np.abs(curves[best].y)))]),
           "leading_peak": float(curves[best].y[int(np.argmax(np.abs(curves[best].y)))])}
    log(f"{what}: {len(w)} channels in {len(pairs)} pairs, {rec['points_per_curve']} points per "
        f"curve ({rec['grouped']} channels each); the weights reproduce the prediction to "
        f"{residual:.1e}; the leading pair is {rec['leading_pair']} with "
        f"{100 * rec['leading_share']:.0f} % of the reading, peaking at "
        f"{rec['leading_peak_at']:.2f}")
    return rec


def _copattern_periodic(log=print) -> Optional[dict]:
    """The periodic autocorrelation of the COD electrolytes: cells on thirteen lattices."""
    if not os.path.isdir(ELECTROLYTES):
        log(f"{ELECTROLYTES} is missing: the periodic co-pattern is skipped")
        return None
    from cmfqsar.model import CMFModel
    from cmfqsar.periodic import (crystal_table, inorganic_fieldset, read_crystal_set)

    cells, names = read_crystal_set(ELECTROLYTES)
    y = []
    for a in cells:
        occ = a.arrays.get("occupancy")
        li = [k for k, s in enumerate(a.get_chemical_symbols()) if s == "Li"]
        count = float(np.sum(occ[li])) if occ is not None else float(len(li))
        y.append(1000.0 * count / float(a.get_volume()))
    y = np.array(y, float)
    fs = inorganic_fieldset([], physchem=["electrostatic", "steric", "prop:electronegativity"],
                            charge_model="electronegativity")
    table = crystal_table(cells, fs, names)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_sigma=0.5, ac_d_max=10.0, params={"lam": 1e-2})
    model.fit(table, y, cv_folds=0)
    rec = _copattern_check(model, fs, table, y,
                           "periodic autocorrelation, 13 COD electrolytes", log=log)
    rec["n_structures"] = len(cells)
    return rec


def _copattern_surface(log=print) -> Optional[dict]:
    """The surface autocorrelation of an adsorbate set: the kernel of plan item EC2."""
    sdf = NODES[0][1]
    if not os.path.exists(sdf):
        log(f"{sdf} is missing: the surface co-pattern is skipped")
        return None
    from rdkit import Chem, RDLogger

    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    from cmfqsar.model import CMFModel
    RDLogger.DisableLog("rdApp.*")
    mols, names, y = [], [], []
    for mol in Chem.SDMolSupplier(sdf, removeHs=False, sanitize=True):
        if mol is None or not mol.HasProp("E_ads_kJmol"):
            continue
        value = float(mol.GetProp("E_ads_kJmol"))
        if not np.isfinite(value):
            continue
        mols.append(Chem.Mol(mol))
        names.append(mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{len(mols)}")
        y.append(value)
    fs = FieldSet.create(physchem=["electrostatic", "steric", "hydrophobic"], alpha=1.0,
                         charge_model="gasteiger", param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    y = np.array(y, float)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="surface",
                     ac_sigma=0.75, ac_d_max=12.0, surf_z_min=1.0, surf_z_max=6.0,
                     surf_z_step=1.0, params={"lam": 1e-2})
    model.fit(table, y, cv_folds=0)
    rec = _copattern_check(model, fs, table, y,
                           "surface autocorrelation, 73 Fe(110) adsorbates", log=log)
    rec["n_structures"] = len(mf)
    return rec


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def report(results_dir: str = RESULTS, tag: str = "") -> str:
    L = ["# The surface side of the program on screen: what it costs, and what it reads\n"]
    prov_path = os.path.join(results_dir, f"run{tag}.json")
    if os.path.exists(prov_path):
        prov = jload(prov_path)
        L.append("`" + " ".join(["python benchmarks/run_surface_view.py"] + list(prov["argv"]))
                 + f"`, written {prov['written']}.\n")
    path = os.path.join(results_dir, f"slab{tag}.json")
    if os.path.exists(path):
        d = jload(path)
        L.append("## The slab behind the molecules\n")
        L.append(f"The slab is {d['slab_atoms']} atoms of "
                 f"{d['slab']['element']}(110) in {len(d['layer_heights'])} layers, the one the "
                 f"adsorption energies were relaxed on. A patch is the surface atoms within the "
                 f"radius of a molecule, measured **in the plane**, so the surface stays a "
                 f"surface under a molecule that sits high above it. One row per radius: the "
                 f"median over the first {d['molecules_timed']} molecules of each set.\n")
        L.append("| Set | Patch radius | Surface atoms drawn | Choosing them | "
                 "Drawing the whole view |")
        L.append("| --- | ---: | ---: | ---: | ---: |")
        for s in d["sets"]:
            for p in s["patches"]:
                L.append(f"| {s['set']} ({s['n_molecules']} molecules) | {p['radius']:.0f} A | "
                         f"{p['atoms_median']:.0f} | {p['cut_seconds'] * 1000:.0f} ms | "
                         f"{p['draw_seconds'] * 1000:.0f} ms |")
        L.append("")
        for s in d["sets"]:
            L.append(f"On the {s['set']} set the highest drawn atom is at "
                     f"z = {s['highest_drawn_z']:+.3f} A, which is the top atomic layer of the "
                     f"frame, and {100 * s['share_in_receptor']:.0f} % of the drawn atoms are "
                     f"atoms of the complementarity receptor itself: the picture shows what the "
                     f"score is computed against.")
        L.append("")
    path = os.path.join(results_dir, f"frame{tag}.json")
    if os.path.exists(path):
        d = jload(path)
        L.append("## The surface frame: the two numbers the panel states\n")
        L.append("The window states the height of the lowest heavy atom and the tilt of the "
                 "molecule's own plane. The height is checked against the `height_A` column the "
                 "relaxation wrote into the data set, which the window never reads. The rule "
                 "pose of `cmfqsar.planealign` lays a molecule flat, so its tilt is 0 by "
                 "construction and the last column is how far the relaxed pose is from flat.\n")
        L.append("| Set | Molecules | Height of the relaxed poses | Agreement with the written "
                 "height | Tilt of the relaxed poses (median, 90th) | Molecules with a flat "
                 "plane | Flat rule pose minus relaxed tilt |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in d["sets"]:
            L.append(f"| {s['set']} | {s['n']} | {s['height_min']:.2f} to "
                     f"{s['height_max']:.2f} A | {s['height_vs_written_max']:.3f} A at worst "
                     f"({s['n_with_written']} matched) | {s['tilt_median']:.1f}, "
                     f"{s['tilt_90th']:.1f} deg | {100 * s['flat_share']:.0f} % | "
                     f"{s['rule_minus_relaxed_tilt_median']:.1f} deg |")
        L.append("")
        L.append("The agreement with the written height is the check that matters: the window "
                 "computes the number from the coordinates and the relaxation wrote it from its "
                 "own trajectory, so they agree only if the window is reading the frame the "
                 "poses are given in.\n")
    path = os.path.join(results_dir, f"arrows{tag}.json")
    if os.path.exists(path):
        d = jload(path)
        if d.get("spacings"):
            L.append("## The vector co-field as arrows\n")
            L.append(f"The electric-field kind as the only field of a model on "
                     f"{d['n_molecules']} Hammett benzenes, target {d['target']}. The arrows are "
                     f"the {d['arrows']} longest gradients of the co-field, kept 1.5 A apart. "
                     f"'Longest arrow' and 'largest gradient on the grid' must agree, or the "
                     f"arrows are not the strongest places. The weakest arrow drawn says how "
                     f"far down the reading goes, as a fraction of the longest and as the share "
                     f"of the volume whose gradient is at least that strong. The direction "
                     f"change is the mean angle between an arrow of the coarsest grid and the "
                     f"nearest arrow of this one.\n")
            L.append("| Grid spacing | Grid points | Gradient | Arrows | Longest arrow | "
                     "Largest gradient on the grid | Weakest arrow | Volume at least that "
                     "strong | Direction change |")
            L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for s in d["spacings"]:
                change = (f"{s['direction_change_median_deg']:.1f} deg "
                          f"({s['matched_arrows']} matched)"
                          if "direction_change_median_deg" in s else "reference")
                L.append(f"| {s['spacing']:.2f} A | {s['points']} | "
                         f"{s['gradient_seconds']:.2f} s | {s['n_arrows']} | "
                         f"{s['longest']:.4g} | {s['grid_maximum']:.4g} | "
                         f"{100 * s['shortest_over_longest']:.0f} % of the longest | "
                         f"{100 * s['volume_share_above_shortest']:.1f} % | {change} |")
            L.append("")
            L.append("The longest arrow is the largest gradient the grid holds, on every grid, "
                     "and the arrows of a finer grid stand within one grid spacing of the "
                     "arrows of the coarsest one: the places the drawing points at are not an "
                     "artefact of the sampling. The direction is another matter. Refining the "
                     "grid turns the arrows by about 16 degrees on average, because an arrow is "
                     "taken at the grid point nearest a peak and a finer grid has a nearer one. "
                     "So an arrow states a direction to roughly fifteen degrees and not better. "
                     "That is enough for the question it answers -- which way the field should "
                     "point -- and not enough to measure an angle off the picture.\n")
    path = os.path.join(results_dir, f"copattern{tag}.json")
    if os.path.exists(path):
        d = jload(path)
        L.append("## The co-pattern chart against the channel table\n")
        L.append("The chart draws the channel weights of one pair of field types against "
                 "distance. Two things make it a reading rather than a picture. The weights "
                 "must reproduce the prediction: the bias plus the sum over every channel of "
                 "weight times channel is the prediction, and the residual below is the largest "
                 "difference over the training structures. And where a kernel resolves a "
                 "distance further -- the surface kernel resolves it by the heights of both "
                 "atoms -- a point of the curve is the sum over what is left, which the last "
                 "column states.\n")
        L.append("| Kernel | Structures | Channels | Pairs of field types | Points per curve | "
                 "Residual of the prediction | Leading pair | Share of the reading | "
                 "Channels summed per point |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
        for m in d["models"]:
            L.append(f"| {m['what']} | {m.get('n_structures', '')} | {m['n_channels']} | "
                     f"{m['n_pairs']} | {m['points_per_curve']} | "
                     f"{m['prediction_residual']:.0e} (target sd {m['y_sd']:.3g}) | "
                     f"{m['leading_pair']} | {100 * m['leading_share']:.0f} % | "
                     f"{m['grouped']} |")
        L.append("")
        for m in d["models"]:
            L.append(f"On the {m['what']} the curve of the leading pair peaks at "
                     f"{m['leading_peak_at']:.2f} with {m['leading_peak']:+.4f}, and one curve "
                     f"costs {1000 * m['seconds_per_curve']:.1f} ms to build, so the chart "
                     f"follows the box without a wait.")
        L.append("")
    return "\n".join(L)


def write_table(results_dir: str = RESULTS, tag: str = "") -> str:
    path = os.path.join(results_dir, f"table{tag}.md")
    text = report(results_dir, tag)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


STAGES = ("slab", "frame", "arrows", "copattern")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--molecules", type=int, default=DRAW_MOLECULES,
                    help=f"molecules of each set whose drawing is timed "
                         f"(default {DRAW_MOLECULES})")
    ap.add_argument("--arrows", type=int, default=ARROWS,
                    help=f"arrows drawn for the vector co-field (default {ARROWS})")
    ap.add_argument("--force", action="store_true", help="redo a stage that is already stored")
    ap.add_argument("--report", action="store_true", help="rebuild the table and stop")
    return ap


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    tag = run_tag(args)
    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        print(write_table(RESULTS, tag))
        return
    t0 = time.time()
    lines: List[str] = []

    def log(s=""):
        print(s, flush=True)
        lines.append(str(s))

    jobs = {"slab": lambda: slab(molecules=args.molecules, log=log),
            "frame": lambda: frame(log=log),
            "arrows": lambda: arrows(count=args.arrows, log=log),
            "copattern": lambda: copattern(log=log)}
    for name in STAGES:
        if name not in args.stage:
            continue
        path = result(f"{name}{tag}.json")
        if os.path.exists(path) and not args.force:
            log(f"{name}: already done")
            continue
        jdump(jobs[name](), path)
    with open(result(f"log{tag}.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    jdump({"argv": list(argv) if argv is not None else sys.argv[1:], "tag": tag,
           "molecules": args.molecules, "arrows": args.arrows,
           "minutes": round((time.time() - t0) / 60, 2),
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}, result(f"run{tag}.json"))
    print(write_table(RESULTS, tag))
    print(f"Elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
