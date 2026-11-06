#!/usr/bin/env python
"""
What the application costs on a periodic structure, and whether its reading adds up (item M13).

The periodic side of the program became visible in the application: a cell with its periodic
images, the co-field inside the cell, and one sphere per site coloured by what that site adds to
the prediction.  A window is not judged by a q2, so this script measures the two things that
decide whether it is usable:

**What one cell costs to show.**  For every example set in the repository: the atoms of a cell,
the radius the fitted kernel reaches to, how many image atoms lie inside that radius, and the
time to expand them, to build the spheres and lines the view is made of, and to draw the frame.
An image expansion that takes a second per cell would make the window unusable, and a cutoff
that reaches further than the drawing shows would make it misleading; both are numbers, so both
are measured.

**Whether the reading is exact.**  The colouring of the sites is a decomposition of the
prediction, not a heat map: the numbers behind the colours must sum to the prediction less the
bias.  That holds for the overlap kernel over the atoms of the cell (decomposition rule 3) and
for the autocorrelation over the sites (plan item M5), and the residual of the sum is reported
for both.  The co-field inside the cell is measured in the same stage: the grid it is evaluated
on, the time it takes, and -- for the overlap kernel, whose co-field is a sum over the periodic
images out to the kernel's cutoff -- how much the images beyond that cutoff would change it
inside the cell, which is what the truncation of the image sum costs.

The sets
--------
* ``examples/hops/disordered`` -- thirteen solid electrolytes from the COD, cells of 4 to 256
  atoms on thirteen *different* lattices, partly occupied sites among them.  This is the case
  for the periodic autocorrelation, which needs no common frame.
* ``examples/sac/substrates.extxyz`` -- the single-atom catalysts of plan item EC8: 139 relaxed
  8 x 8 sheets of 127 atoms, all on **one** lattice, periodic in two directions with vacuum
  along the third.  This is the case for the periodic overlap kernel and the only set in the
  repository with a co-field inside the cell, and the measured property used here is the height
  of the metal above the sheet, which the drawing can be checked against: the metal is the site
  the model should put its largest contribution on.
* ``examples/hops/*.extxyz`` -- the three DFT-relaxed solid-electrolyte supercells of item BT7,
  112 to 128 atoms, as the large-cell case for the drawing.

Usage
-----
    python benchmarks/run_cell_view.py                    # every stage, then the table
    python benchmarks/run_cell_view.py --stage draw
    python benchmarks/run_cell_view.py --stage read --limit 20
    python benchmarks/run_cell_view.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# the window is a Qt window; it is driven here without a screen, as the GUI tests are
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

RESULTS = os.path.join(HERE, "results", "cell_view")
EXAMPLES = os.path.join(ROOT, "examples")
#: the sets drawn, as (short name, path, periodic directions, what it is)
SETS = (
    ("disordered", os.path.join(EXAMPLES, "hops", "disordered"), None,
     "thirteen COD solid electrolytes, thirteen different lattices"),
    ("hosts", os.path.join(EXAMPLES, "hops"), None,
     "three DFT-relaxed solid-electrolyte supercells"),
    ("sheets", os.path.join(EXAMPLES, "sac", "substrates.extxyz"), "ab",
     "139 single-atom sheets on one lattice, periodic in two directions"),
)
#: how many cells of a set the drawing stage times (the cost is per cell, not per set)
DRAW_CELLS = 5
#: cells of the one-lattice set fitted in the reading stage
LIMIT = 20
#: spacing of the co-field grid inside the cell, in angstrom
SPACING = 0.5
#: the property of the single-atom set used as the target: the height of the metal above the
#: sheet, which is a geometric quantity the per-site reading can be checked against
TARGET = "height"


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

    ``--stage`` and ``--report`` only choose which parts are done, and ``--force`` only decides
    whether a finished stage is done again, so neither changes a stored number or the tag.
    """
    tag = ""
    if a.cells != DRAW_CELLS:
        tag += f"_cells{a.cells}"
    if a.limit != LIMIT:
        tag += f"_n{a.limit}"
    if a.spacing != SPACING:
        tag += f"_sp{a.spacing:g}"
    return tag


# --------------------------------------------------------------------------- #
# Stage 1: what one cell costs to show
# --------------------------------------------------------------------------- #
def draw(cells: int = DRAW_CELLS, log=print) -> dict:
    """Open every example set in the window and time the drawing of its cells."""
    from PySide6.QtWidgets import QApplication
    from cmfqsar.gui.crystal_window import CrystalWindow
    from cmfqsar.periodic import cell_image_indices, overlap_cutoff, inorganic_fieldset

    app = QApplication.instance() or QApplication([])          # noqa: F841 - Qt needs it alive
    out = {"cells_timed": int(cells), "sets": []}
    #: the radius the overlap kernel of the standard inorganic field set reaches to; it is the
    #: radius the window offers for the images once a model is fitted
    cutoff = overlap_cutoff([s for s in inorganic_fieldset(["Li", "O"]).specs if s.enabled])
    for name, path, pbc, what in SETS:
        if not os.path.exists(path):
            log(f"{name}: {path} is missing, skipped")
            continue
        w = CrystalWindow(background=False)
        if pbc:
            w.cmb_pbc.setCurrentIndex({"abc": 1, "ab": 2, "a": 3}[pbc])
        t0 = time.time()
        w.load(path)
        read_seconds = time.time() - t0
        n = len(w.names)
        w.spin_images.setValue(round(cutoff, 1))
        rows = []
        for i in range(min(int(cells), n)):
            crystal = w._crystal(i)
            t1 = time.time()
            coords, _index = cell_image_indices(crystal, cutoff)
            expand = time.time() - t1
            w.table.setCurrentCell(i, 0)
            t2 = time.time()
            w.update_display()
            drawn = time.time() - t2
            rows.append({"name": w.names[i], "atoms": int(crystal.n_atoms),
                         "images": int(len(coords) - crystal.n_atoms),
                         "expand_seconds": expand, "draw_seconds": drawn,
                         "periodic": int(crystal.n_periodic),
                         "volume": float(crystal.volume)})
            log(f"  {w.names[i]:22s} {crystal.n_atoms:4d} atoms, "
                f"{len(coords) - crystal.n_atoms:6d} images within {cutoff:.1f} A, "
                f"expanded in {expand * 1000:.0f} ms, drawn in {drawn * 1000:.0f} ms")
        atoms = [len(a) for a in w.atoms]
        out["sets"].append({
            "set": name, "what": what, "path": os.path.relpath(path, ROOT).replace(os.sep, "/"),
            "n_structures": n, "read_seconds": read_seconds,
            "atoms_min": int(min(atoms)), "atoms_max": int(max(atoms)),
            "pbc": pbc or "from the files", "cutoff": float(cutoff), "cells": rows})
        log(f"{name}: {n} structures read in {read_seconds:.1f} s, "
            f"{min(atoms)} to {max(atoms)} atoms per cell")
        w.close()
    return out


# --------------------------------------------------------------------------- #
# Stage 2: is the reading exact, and what does it cost
# --------------------------------------------------------------------------- #
def read(limit: int = LIMIT, spacing: float = SPACING, log=print) -> dict:
    """Fit both periodic kernels in the window and check the readings it draws.

    The autocorrelation is fitted on the thirteen COD electrolytes, which stand on thirteen
    lattices and are exactly what it is for; the overlap kernel on the single-atom sheets, which
    stand on one.  For each, the per-site split is summed and compared with the prediction, and
    for the overlap kernel the co-field inside the cell is evaluated and the images its sum
    leaves out are measured.
    """
    from PySide6.QtWidgets import QApplication
    from cmfqsar.gui.crystal_window import CrystalWindow, site_contributions_of
    from cmfqsar.cofields import evaluate_gaussians
    from cmfqsar.periodic import crystal_table

    app = QApplication.instance() or QApplication([])          # noqa: F841
    out = {"limit": int(limit), "spacing": float(spacing), "models": []}

    # --- the autocorrelation, on cells that share no lattice
    w = CrystalWindow(background=False)
    w.load(os.path.join(EXAMPLES, "hops", "disordered"))
    # a target that is a property of the cell and not a label: the number of lithium sites per
    # cubic nanometre, which every one of these conductors has and which the fields can see
    y = []
    for a in w.atoms:
        occ = a.arrays.get("occupancy")
        li = [k for k, s in enumerate(a.get_chemical_symbols()) if s == "Li"]
        count = float(np.sum(occ[li])) if occ is not None else float(len(li))
        y.append(1000.0 * count / float(a.get_volume()))
    w.y = np.array(y, float)
    w.target_name = "Li sites per cubic nanometre"
    w.populate()
    out["models"].append(_fit_and_read(w, kernel=0, spacing=spacing, log=log,
                                       what="periodic autocorrelation on 13 COD electrolytes"))
    w.close()

    # --- the overlap kernel, on one lattice, with a co-field
    path = os.path.join(EXAMPLES, "sac", "substrates.extxyz")
    if not os.path.exists(path):
        log("the single-atom sheets are missing: the overlap part is skipped")
        return out
    w = CrystalWindow(background=False)
    w.cmb_pbc.setCurrentIndex(2)                               # ab: a layer
    w.load(path)
    w.atoms, w.names = w.atoms[:int(limit)], w.names[:int(limit)]
    w.y = np.zeros(0)
    w.import_targets(os.path.join(EXAMPLES, "sac", "sac_hoer.csv"), column=TARGET)
    w.chk_physchem["electrostatic"].setChecked(False)          # a sheet of one element and a metal
    rec = _fit_and_read(w, kernel=1, spacing=spacing, log=log,
                        what=f"periodic overlap kernel on {len(w.names)} single-atom sheets")

    # the co-field of a periodic model must be periodic: one lattice vector away, the same value
    model = w.model
    crystal = w._crystal(0, model.fieldset)
    # the co-field inside the cell is a sum over the images of the training cells out to the
    # kernel's cutoff; doubling that cutoff adds the images the sum left out, and what they
    # change inside the cell is what the truncation costs.  Points inside the cell itself
    # (fractional coordinates), at the heights of the atoms
    rng = np.random.RandomState(0)
    A = np.asarray(crystal.cell, float)
    frac = rng.rand(200, 3)
    z = np.asarray(crystal.coords, float)[:, 2]
    frac[:, 2] = (z.min() + rng.rand(200) * (z.max() - z.min())) / A[2, 2]
    pts = frac @ A
    r0 = w._kernel_cutoff()
    centres, coeffs, widths = model.cofield_coefficients(0)
    here = evaluate_gaussians(pts, centres, coeffs, widths)
    model.pbc_overlap_r_max = 2.0 * r0
    centres, coeffs, widths = model.cofield_coefficients(0)
    full = evaluate_gaussians(pts, centres, coeffs, widths)
    model.pbc_overlap_r_max = None
    scale = float(np.abs(full).max()) or 1.0
    rec["truncation"] = float(np.max(np.abs(here - full)) / scale)
    rec["truncation_cutoff"] = float(r0)
    log(f"  the images beyond {r0:.1f} A change the co-field inside the cell by "
        f"{rec['truncation']:.1e} of its peak")

    # the metal is the site the model should put its largest contribution on
    values, _how = site_contributions_of(model, crystal)
    biggest = int(np.argmax(np.abs(values)))
    rec["largest_site"] = crystal.symbols[biggest]
    rec["largest_is_the_metal"] = bool(crystal.symbols[biggest] not in ("C", "N"))
    log(f"  the largest per-site contribution of {crystal.name} is site {biggest + 1} "
        f"({crystal.symbols[biggest]}): {values[biggest]:+.3f}")
    out["models"].append(rec)
    w.close()
    return out


def _fit_and_read(w, kernel: int, spacing: float, log=print, what: str = "") -> dict:
    """Fit one model in the window, then time and check the readings it offers."""
    from cmfqsar import periodic
    from cmfqsar.gui.crystal_window import site_contributions_of
    from cmfqsar.periodic import crystal_table

    w.cmb_kernel.setCurrentIndex(kernel)
    w.cmb_charges.setCurrentText("electronegativity")
    w.spin_min_structures.setValue(3)
    w.spin_folds.setValue(3)
    w.ed_alphas.setText("2" if kernel else "1")
    w.ed_lams.setText("1e-3,1e-2,0.1")
    t0 = time.time()
    w.build_model()
    fit_seconds = time.time() - t0
    model = w.model
    if model is None:
        raise SystemExit("the model was not fitted; see the log above")
    log(f"{what}: fitted in {fit_seconds:.1f} s, {model.stats.q2:.3f} q2 "
        f"({w.spin_folds.value()} folds)")
    rec = {"what": what, "kernel": "overlap" if kernel else "autocorrelation",
           "n_cells": int(len(w.names)), "fields": [s.label for s in model.fieldset.specs
                                                    if s.enabled],
           "fit_seconds": fit_seconds, "q2": float(model.stats.q2),
           "rmse_cv": float(model.stats.rmse_cv), "target": w.target_name,
           "cutoff": float(w._kernel_cutoff())}

    # the per-site split: the time it takes, and the residual of what it claims
    residuals, seconds = [], []
    for i in range(min(5, len(w.names))):
        crystal = w._crystal(i, model.fieldset)
        t1 = time.time()
        values, how = site_contributions_of(model, crystal)
        seconds.append(time.time() - t1)
        predicted = float(model.predict(crystal_table([w.atoms[i]], model.fieldset,
                                                      [w.names[i]]))[0])
        residuals.append(abs(float(np.sum(values)) + float(model.bias) - predicted))
    rec["split"] = how
    rec["split_seconds"] = float(np.mean(seconds))
    rec["split_residual"] = float(np.max(residuals))
    rec["lattice_spread"] = float(periodic.lattice_spread(model.train).get("length_spread", 0.0))
    rec["y_sd"] = float(np.nanstd(w.y))
    log(f"  the per-site split: {np.mean(seconds) * 1000:.0f} ms per cell, the sum differs from "
        f"the prediction by at most {np.max(residuals):.1e}")

    if kernel:
        w.table.setCurrentCell(0, 0)
        w.spin_spacing.setValue(spacing)
        t2 = time.time()
        w.compute_cofield()
        rec["cofield_seconds"] = time.time() - t2
        _i, _name, label, volume, grid = w._volume
        rec["cofield_field"] = label
        rec["cofield_points"] = int(np.prod(grid.shape))
        rec["cofield_shape"] = [int(v) for v in grid.shape]
        rec["cofield_spacing"] = float(grid.spacing)
        rec["cofield_peak"] = float(np.abs(volume).max())
        log(f"  the {label} co-field inside the cell: {rec['cofield_points']} points at "
            f"{grid.spacing:.2f} A in {rec['cofield_seconds']:.1f} s")
    return rec


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def report(results_dir: str = RESULTS, tag: str = "") -> str:
    """Rebuild the table from the stored result files."""
    L = ["# A periodic structure in the application: what it costs to show, and what it reads\n"]
    prov_path = os.path.join(results_dir, f"run{tag}.json")
    if os.path.exists(prov_path):
        prov = jload(prov_path)
        L.append("`" + " ".join(["python benchmarks/run_cell_view.py"] + list(prov["argv"]))
                 + f"`, written {prov['written']}.\n")
    draw_path = os.path.join(results_dir, f"draw{tag}.json")
    if os.path.exists(draw_path):
        d = jload(draw_path)
        L.append("## What one cell costs to show\n")
        L.append("One row per example set: the cells it holds, and the median over the first "
                 f"{d['cells_timed']} cells of the time to expand the periodic images, and of "
                 "the time the window takes to draw the cell, its atoms and those images. The "
                 "radius is the one the periodic overlap kernel of the standard inorganic field "
                 "set reaches to, which is the radius the window offers once a model is "
                 "fitted.\n")
        L.append("| Set | Structures | Atoms per cell | Periodic in | Images within the cutoff | "
                 "Expanding them | Drawing the view | Reading the files |")
        L.append("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
        for s in d["sets"]:
            cells = s["cells"]
            if not cells:
                continue
            images = int(np.median([c["images"] for c in cells]))
            expand = float(np.median([c["expand_seconds"] for c in cells]))
            drawn = float(np.median([c["draw_seconds"] for c in cells]))
            per = {3: "3 directions", 2: "2 directions", 1: "1 direction",
                   0: "nothing"}[cells[0]["periodic"]]
            L.append(f"| {s['set']}: {s['what']} | {s['n_structures']} | "
                     f"{s['atoms_min']} to {s['atoms_max']} | {per} | {images} "
                     f"(r = {s['cutoff']:.1f} A) | {expand * 1000:.0f} ms | {drawn * 1000:.0f} ms "
                     f"| {s['read_seconds']:.1f} s |")
        L.append("")
    read_path = os.path.join(results_dir, f"read{tag}.json")
    if os.path.exists(read_path):
        r = jload(read_path)
        L.append("## Is the reading exact, and what does it cost\n")
        L.append("The colouring of the sites is a decomposition of the prediction, so its "
                 "numbers must sum to the prediction less the bias; the residual below is the "
                 "largest difference over five cells. The co-field exists for the overlap "
                 "kernel only, and being the field of a periodic model it must repeat from cell "
                 "to cell: the periodicity is the largest difference between its value at a "
                 "point and at the same point one lattice vector away, as a fraction of its "
                 "peak.\n")
        L.append("| Model | Cells | Spread of the lattice | q2 (3 folds) | Per-site split | "
                 "Residual of the sum | Co-field inside the cell | Left out by the cutoff |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
        for m in r["models"]:
            co = (f"{m['cofield_points']} points at {m['cofield_spacing']:.2f} A, "
                  f"{m['cofield_seconds']:.1f} s" if "cofield_points" in m
                  else "none: the kernel is a distribution of distances")
            per = (f"{m['truncation']:.0e} beyond {m['truncation_cutoff']:.1f} A"
                   if "truncation" in m else "-")
            L.append(f"| {m['what']} | {m['n_cells']} | {100 * m['lattice_spread']:.1f} % | "
                     f"{m['q2']:.3f} | {m['split_seconds'] * 1000:.0f} ms per cell | "
                     f"{m['split_residual']:.0e} (target sd {m['y_sd']:.2f}) | {co} | {per} |")
        L.append("")
        L.append("The residual of the autocorrelation split is rounding. The residual of the "
                 "overlap split is not: the periodic overlap integral runs over the cell of one "
                 "structure and the images of the other, so its two orders agree only on "
                 "exactly one lattice, and the split is one of the two orders while the kernel "
                 "is their mean. Relaxed cells stand on one lattice only to the spread shown, "
                 "and the residual is what that costs; the window reports it beside the "
                 "colouring whenever it exceeds a tenth of a per cent of the prediction.\n")
        for m in r["models"]:
            if "largest_site" in m:
                L.append(f"The target of the one-lattice set is the height of the metal above "
                         f"the sheet, and the largest per-site contribution of the first cell is "
                         f"its **{m['largest_site']}** site, which is the metal itself: the "
                         f"colouring points at the atom the property belongs to.\n")
    return "\n".join(L)


def write_table(results_dir: str = RESULTS, tag: str = "") -> str:
    path = os.path.join(results_dir, f"table{tag}.md")
    text = report(results_dir, tag)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", nargs="*", default=["draw", "read"], choices=["draw", "read"])
    ap.add_argument("--cells", type=int, default=DRAW_CELLS,
                    help=f"cells of each set whose drawing is timed (default {DRAW_CELLS})")
    ap.add_argument("--limit", type=int, default=LIMIT,
                    help=f"cells of the one-lattice set that are fitted (default {LIMIT})")
    ap.add_argument("--spacing", type=float, default=SPACING,
                    help=f"spacing of the co-field grid inside the cell (default {SPACING} A)")
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

    if "draw" in args.stage:
        path = result(f"draw{tag}.json")
        if os.path.exists(path) and not args.force:
            log("draw: already done")
        else:
            jdump(draw(cells=args.cells, log=log), path)
    if "read" in args.stage:
        path = result(f"read{tag}.json")
        if os.path.exists(path) and not args.force:
            log("read: already done")
        else:
            jdump(read(limit=args.limit, spacing=args.spacing, log=log), path)
    with open(result(f"log{tag}.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    jdump({"argv": list(argv) if argv is not None else sys.argv[1:], "tag": tag,
           "cells": args.cells, "limit": args.limit, "spacing": args.spacing,
           "minutes": round((time.time() - t0) / 60, 2),
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}, result(f"run{tag}.json"))
    print(write_table(RESULTS, tag))
    print(f"Elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
