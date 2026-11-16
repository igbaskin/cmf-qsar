#!/usr/bin/env python
"""
Grid fields from DFT against the fitted atom-centred fields (plan item M9).

`cmfqsar/gridfields.py` takes the DFT output itself as the fields: the electron density, the
electron localisation function (ELF) and the electrostatic potential, as values on a grid. This
script asks the question the plan sets for that module. Rule 3 keeps the kernel in closed form;
the item may break it only if a grid kernel predicts better than the program's atom-centred kernel
by more than the interval.

The two kernels of the item
---------------------------
* **A shared frame** -- the overlap integral of two grids on the same points
  (`grid_overlap_kernel`), against the program's closed-form overlap kernel of the atom-centred
  fields on the same frame.
* **No frame** -- the radial auto- and cross-correlations of the grids (`radial_correlations`),
  against the field autocorrelation of the atom-centred fields, laid out in the same blocks and
  given the same kernel (`cmfqsar.invariant.autocorrelation_kernel`).

A third kernel on the same structures separates two explanations of any difference: the **fitted
grid**, each grid replaced by its best fit with Gaussians of three widths on every atom
(`fit_atom_gaussians`), given to the same two kernels. If the grid beats the atom-centred fields
and the fitted grid does as well, what helped is the DFT quantity, not its shape on the grid.

The data
--------
*Crystals* -- the 1,500 ABO3 perovskite cells of item M4 (`matbench_perovskites`, formation energy
of the cell, Castelli et al. 2012), each given one static PBE calculation in VASP
(`benchmarks/grid_dft_wsl.py`, same settings for every cell, no relaxation): the valence density
from CHGCAR, the ELF from ELFCAR and the electrostatic potential from LOCPOT. The grids are put on
24 x 24 x 24 points of each cell. The atom-centred kernels are those of M4 with its four fields
(electrostatic, steric, electronegativity, inverse square root of the mass): the periodic
autocorrelation of the cells as given, and the periodic overlap on one lattice.

*Molecules* -- the four corrosion-inhibitor series that have ORCA single points
(`run_corrosion_qm.py`, 20 to 29 molecules each, aligned on their common scaffold): the density and
the electrostatic potential of the neutral molecule, written by `orca_plot` on 61 x 61 x 61 points.
The density is used without its cores (`cmfqsar.cdft.valence_window`); the potential is used where
the density is below 0.002 electrons per cubic bohr, outside the molecule, because inside it the
potential of the nuclei dominates. The atom-centred kernels are the five ordinary fields.

The gate
--------
Rule 3 rests on the conceptual-DFT study, which found that an exact grid kernel does no better
than the same field *fitted* with atom-centred Gaussians, whose kernel is closed-form. The gate
repeats that comparison on the DFT fields of this item: **grid minus fitted grid**, for each frame,
as the difference of the cross-validated q² (R² for the crystals) with a 95 % paired bootstrap
interval over the structures. The grid kernel earns its exception from rule 3 only where the
interval lies wholly above zero.

A second question is reported beside it and does not decide the gate: **grid minus the
program's own fields** (the four fields of M4, the five ordinary fields of a molecule). It asks
whether the DFT output knows more than the empirical fields; where it does and the fitted grid
does as well, that knowledge can enter a closed-form kernel, and rule 3 need not be broken.

This reading of the gate was settled after a trial of the pipeline on the first 144 cells, in
which the grid beat the empirical fields clearly and its own fit narrowly; both comparisons are
reported in full whatever they show. Five folds repeated three times; the setting (window width
or attenuation) and the ridge are chosen by the evidence on the training block of each fold.
The crystals decide the item; the molecules are reported series by series, because 20 to 29
molecules give intervals of about 0.1 to 0.2.

The reference that needs no grid (added 16 November 2026)
--------------------------------------------------------
The DFT run that gives the grids of a cell also gives its total energy, and the target is a
formation energy. Stage ``energies`` reads that energy from each OUTCAR; stage ``reference``
compares the grid kernel with a ridge regression on the energy and the element counts, on the
same folds. The grid input is only worth its cost if it beats this regression.

Usage
-----
    python benchmarks/run_gridfields.py --stage cells        # writes the cells for VASP
    (inside WSL) python3 benchmarks/grid_dft_wsl.py CELLS WORKDIR   # about an hour, 4 workers
    python benchmarks/run_gridfields.py --stage collect --vasp-dir DIR
    python benchmarks/run_gridfields.py --stage crystals
    python benchmarks/run_gridfields.py --stage energies --vasp-dir DIR  # the energy of each run
    python benchmarks/run_gridfields.py --stage reference                # grid against that energy
    python benchmarks/run_gridfields.py --stage cubes --qm-results DIR   # orca_plot, 15 minutes
    python benchmarks/run_gridfields.py --stage molecules
    python benchmarks/run_gridfields.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar import gridfields as G                                         # noqa: E402
from cmfqsar.cdft import BOHR, PLOT_DENSITY, PLOT_POTENTIAL, valence_window  # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel, channel_scale          # noqa: E402
from cmfqsar.model import r2_score, rmse                                     # noqa: E402

import run_resolution as RS                                                  # noqa: E402
import run_wavelet as RW                                                     # noqa: E402

CACHE = os.path.join(HERE, "cache", "gridfields")
RESULTS = os.path.join(HERE, "results", "gridfields")
FOLDS, REPEATS, SEED = 5, 3, 0
#: the ridges searched.  Wider than the lists of run_wavelet.py and run_topomer.py, which stop
#: at 1e-3: on these kernels that edge was chosen in every fold, so it limited the result
LAMS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)

# --- crystals ---------------------------------------------------------------
CELL_SHAPE = (24, 24, 24)
CRYSTAL_FIELDS = ("density", "elf", "potential")
CRYSTAL_FILES = {"density": "CHGCAR.gz", "elf": "ELFCAR.gz", "potential": "LOCPOT.gz"}
#: the potential's mean is set by convention (VASP sets it to zero), so it is always centred
CENTRE = {"density": False, "elf": False, "potential": True}
#: the window widths of the radial channels, as M4 uses for its autocorrelation, and its reach
CRYSTAL_SIGMAS = (0.35, 0.5, 0.75)
CRYSTAL_R_MAX = 10.0
CRYSTAL_REACH = CRYSTAL_R_MAX + 4.0 * max(CRYSTAL_SIGMAS)
#: widths of the atom-centred fit, in inverse square angstrom
FIT_WIDTHS = (8.0, 2.0, 0.5)
#: a richer fit, for the check that the gate does not rest on a weak fit (stage widefit)
WIDE_WIDTHS = (32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5)
WIDE = "grid fitted with seven widths"
#: attenuations of M4's periodic overlap kernel on one lattice
#: M4 searched 1, 2 and 4; 4 was chosen in every fold here, so the list goes on to 16
OVERLAP_ALPHAS = (1.0, 2.0, 4.0, 8.0, 16.0)

# --- molecules --------------------------------------------------------------
SETS = ("7_c1nncs1", "12_c1ccoc1", "13_c1c_nH_nn1", "161_c1ccc2scnc2c1")
INTERVALS = 60
MOL_SPACING = 0.3
#: the density below which the potential is kept (0.002 e/bohr^3), in e/A^3
RHO_SURFACE = 0.002 / BOHR ** 3
MOL_SIGMAS = (0.5, 0.75, 1.0)
MOL_R_MAX = 12.0
MOL_REACH = MOL_R_MAX + 4.0 * max(MOL_SIGMAS)
MOL_FIELDS = ("density", "potential")
#: the attenuations of the five fields: those of run_topomer.py, whose largest (5) was chosen in
#: most folds of two series, and two more beyond it
MOL_ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# --------------------------------------------------------------------------- #
# Cross-validation shared by both parts
# --------------------------------------------------------------------------- #

def select_ridge(K: np.ndarray, y: np.ndarray, tr: np.ndarray) -> float:
    """The ridge of :data:`LAMS` with the highest log evidence on the training block."""
    return select_width(K[None], y, tr)[1]


def select_width(stack: np.ndarray, y: np.ndarray, tr: np.ndarray):
    """The kernel of the stack and the ridge with the highest log evidence on the training block."""
    from cmfqsar.evidence import log_evidence

    best = None
    for w in range(stack.shape[0]):
        Ktr = stack[w][np.ix_(tr, tr)]
        for lam in LAMS:
            e = log_evidence(Ktr, y[tr], float(lam), fit_bias=True)
            if np.isfinite(e) and (best is None or e > best[0]):
                best = (e, w, float(lam))
    return best[1], best[2]


def cross_validate(models: Dict[str, tuple], y: np.ndarray, labels: Dict[str, Sequence[str]]):
    """Every model on the same repeated folds; a model is ("one", stack) or ("sum", K).

    "one" chooses one kernel of the stack and the ridge by the evidence on the training block;
    "sum" chooses the ridge only. Returns the mean q² over the repeats, its spread, the chosen
    settings and the mean held-out prediction of every structure (for the paired bootstrap).
    """
    n = len(y)
    preds = {m: np.zeros((REPEATS, n)) for m in models}
    chosen: Dict[str, List[str]] = {m: [] for m in models}
    for rep in range(REPEATS):
        for te in RW.folds(n, FOLDS, SEED + rep):
            tr = np.setdiff1d(np.arange(n), te)
            for m, (how, data) in models.items():
                if how == "one":
                    w, lam = select_width(data, y, tr)
                    K = data[w]
                    chosen[m].append(f"{labels[m][w]}, ridge {lam:g}")
                else:
                    K = data
                    lam = select_ridge(K, y, tr)
                    chosen[m].append(f"ridge {lam:g}")
                preds[m][rep, te] = RS.predict(K, y, tr, te, lam)
    out = {}
    for m in models:
        r2s = [float(r2_score(y, preds[m][r])) for r in range(REPEATS)]
        settings = {s: chosen[m].count(s) for s in sorted(set(chosen[m]))}
        out[m] = {"q2_mean": float(np.mean(r2s)), "q2_sd": float(np.std(r2s)),
                  "rmse_mean": float(np.mean([rmse(y, preds[m][r]) for r in range(REPEATS)])),
                  "chosen": settings, "_pred": preds[m].mean(axis=0)}
    return out


def compare(res: Dict[str, dict], y: np.ndarray, pairs: Sequence[tuple]) -> List[dict]:
    """q² of a minus q² of b with the paired bootstrap interval, for every (a, b) pair."""
    rows = []
    for a, b in pairs:
        g = RS.paired(y, res[a]["_pred"], res[b]["_pred"])
        rows.append({"model": a, "against": b, **g, "above_zero": bool(g["lo"] > 0),
                     "below_zero": bool(g["hi"] < 0)})
    return rows


def _passes(row: dict) -> bool:
    """The gate: the interval of the difference wholly above zero."""
    return bool(row["lo"] > 0)


def strip(res: Dict[str, dict]) -> Dict[str, dict]:
    return {m: {k: v for k, v in r.items() if not k.startswith("_")} for m, r in res.items()}


def cosine(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(K), 1e-300))
    return K / np.outer(d, d)


def joint_cosine(Ks: Sequence[np.ndarray]) -> np.ndarray:
    """The mean of cosine-normalised kernels: every field with unit diagonal, equal weights."""
    return np.mean([cosine(K) for K in Ks], axis=0)


def radial_kernel(F: np.ndarray, n_bins: int, n_fields: int) -> np.ndarray:
    """The kernel of `cmfqsar.invariant` on radial channels: blocks scaled, joint cosine."""
    blocks = G.radial_block_sizes(G.RadialSpec(step=1.0, r_max=n_bins - 1.0), n_fields)
    sc = channel_scale(F, n_bins, blocks=blocks)
    return autocorrelation_kernel(F, scale=sc, n_bins=n_bins, blocks=blocks)


# --------------------------------------------------------------------------- #
# Crystals: the cells, the VASP grids, the kernels
# --------------------------------------------------------------------------- #

def m4_set():
    """The cells, names, targets and reference cell of item M4's subset."""
    import run_periodic_overlap as P

    rec = jload(os.path.join(HERE, "results", "periodic_overlap", "prepared.json"))
    from cmfqsar import periodic
    every, y_all, ids = periodic.structures_from_matbench(P.fetch(), target=P.TARGET)
    rows = np.asarray(rec["rows"], int)
    atoms = [every[i] for i in rows]
    names = [f"{a.get_chemical_formula()}_{ids[i]}" for a, i in zip(atoms, rows)]
    return atoms, names, np.asarray(y_all, float)[rows], rec


def stage_cells(path: Optional[str] = None) -> str:
    """Write the cells of M4's subset as extended XYZ, one name per frame, for the VASP runner."""
    from ase.io import write

    atoms, names, _y, _rec = m4_set()
    out = []
    for a, n in zip(atoms, names):
        b = a.copy()
        b.info = {"name": n}
        out.append(b)
    path = path or os.path.join(CACHE, "cells.extxyz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write(path, out, format="extxyz")
    log(f"{len(out)} cells written to {path}; run benchmarks/grid_dft_wsl.py on it inside WSL")
    return path


def _collect_one(args):
    """Read, resample, fit and bin the three grids of one cell; write one npz file."""
    name, run_dir, out = args
    if os.path.isfile(out):
        return name, "cached"
    done = os.path.join(run_dir, "done.json")
    if not os.path.isfile(done) or not jload(done).get("ok"):
        return name, "no converged run"
    grids = []
    for f in CRYSTAL_FIELDS:
        g = G.read_vasp_grid(os.path.join(run_dir, CRYSTAL_FILES[f]), kind=f)
        grids.append(G.resample_fractional(g, CELL_SHAPE))
    fitted, errors = [], []
    for f, g in zip(CRYSTAL_FIELDS, grids):
        fg, _c, err = G.fit_atom_gaussians(g, FIT_WIDTHS, centre=CENTRE[f])
        if CENTRE[f]:
            fg = fg.with_values(fg.values + g.values.mean())
        fitted.append(fg)
        errors.append(err)
    centre = [CENTRE[f] for f in CRYSTAL_FIELDS]
    h_grid = G.radial_histogram(grids, CRYSTAL_REACH, centre=centre)
    h_fit = G.radial_histogram(fitted, CRYSTAL_REACH, centre=centre)
    np.savez_compressed(out, grid=np.stack([g.values for g in grids]).astype(np.float32),
                        fitted=np.stack([g.values for g in fitted]).astype(np.float32),
                        hist_grid=h_grid.hist, hist_fitted=h_fit.hist, volume=grids[0].volume,
                        fit_error=np.asarray(errors), cell=grids[0].cell,
                        electrons=grids[0].integral())
    return name, "ok"


def stage_collect(vasp_dir: str, jobs: int = 6) -> dict:
    """Every cell's grids from the VASP runs on the shared shape, with fits and histograms."""
    _atoms, names, _y, _rec = m4_set()
    os.makedirs(os.path.join(CACHE, "crystals"), exist_ok=True)
    tasks = [(n, os.path.join(vasp_dir, n), os.path.join(CACHE, "crystals", safe(n) + ".npz"))
             for n in names]
    status: Dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, (name, st) in enumerate(ex.map(_collect_one, tasks, chunksize=4)):
            status[st] = status.get(st, 0) + 1
            if i % 100 == 0:
                log(f"collect {i + 1}/{len(tasks)} ({time.time() - t0:.0f} s): {status}")
    log(f"collect done: {status}")
    return status


def _widefit_one(args):
    """The three grids of one cell fitted with :data:`WIDE_WIDTHS`; one npz file."""
    name, atoms, src, out = args
    if os.path.isfile(out):
        return name, "cached", None
    if not os.path.isfile(src):
        return name, "no grids", None
    z = np.load(src)
    pos = atoms.get_scaled_positions(wrap=True) @ np.asarray(z["cell"])
    fitted, errors, grids = [], [], []
    for f, q in enumerate(CRYSTAL_FIELDS):
        g = G.GridField(z["grid"][f].astype(float), z["cell"], periodic=True,
                        symbols=atoms.get_chemical_symbols(), positions=pos, kind=q)
        fg, _c, err = G.fit_atom_gaussians(g, WIDE_WIDTHS, centre=CENTRE[q])
        if CENTRE[q]:
            fg = fg.with_values(fg.values + g.values.mean())
        grids.append(g)
        fitted.append(fg)
        errors.append(err)
    h = G.radial_histogram(fitted, CRYSTAL_REACH, centre=[CENTRE[q] for q in CRYSTAL_FIELDS])
    np.savez_compressed(out, fitted=np.stack([g.values for g in fitted]).astype(np.float32),
                        hist_fitted=h.hist, fit_error=np.asarray(errors))
    return name, "ok", errors


def stage_widefit(jobs: int = 6) -> dict:
    """Every cell's grids fitted again with seven widths instead of three (a check of the gate)."""
    atoms, names, _y, _rec = m4_set()
    os.makedirs(os.path.join(CACHE, "crystals_wide"), exist_ok=True)
    tasks = [(n, a, os.path.join(CACHE, "crystals", safe(n) + ".npz"),
              os.path.join(CACHE, "crystals_wide", safe(n) + ".npz"))
             for a, n in zip(atoms, names)]
    status: Dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, (name, st, _e) in enumerate(ex.map(_widefit_one, tasks, chunksize=4)):
            status[st] = status.get(st, 0) + 1
            if i % 250 == 0:
                log(f"widefit {i + 1}/{len(tasks)} ({time.time() - t0:.0f} s): {status}")
    log(f"widefit done: {status}")
    return status


def crystal_grids(names: Sequence[str]):
    """The collected arrays of the cells that have them, and the indices of those cells."""
    keep, data = [], []
    for i, n in enumerate(names):
        path = os.path.join(CACHE, "crystals", safe(n) + ".npz")
        if os.path.isfile(path):
            keep.append(i)
            data.append(dict(np.load(path)))
    return np.asarray(keep, int), data


def crystal_kernels(force: bool = False):
    """Every kernel of the crystal part on the cells with grids, cached in one npz file."""
    import run_periodic_overlap as P

    path = os.path.join(CACHE, "crystal_kernels.npz")
    atoms, names, y, rec = m4_set()
    if os.path.isfile(path) and not force:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}, y[z["keep"]], [names[i] for i in z["keep"]]
    keep, data = crystal_grids(names)
    log(f"{len(keep)} of {len(names)} cells have grids")
    wide = [os.path.join(CACHE, "crystals_wide", safe(names[i]) + ".npz") for i in keep]
    if all(os.path.isfile(w) for w in wide):
        for d, w in zip(data, wide):
            z = np.load(w)
            d["wide"], d["hist_wide"] = z["fitted"], z["hist_fitted"]
            d["wide_error"] = z["fit_error"]
    out = {"keep": keep}
    n_f = len(CRYSTAL_FIELDS)
    for tag in ("grid", "fitted") + (("wide",) if "wide" in data[0] else ()):
        stack = []
        for s in CRYSTAL_SIGMAS:
            spec = G.RadialSpec(r_max=CRYSTAL_R_MAX, step=s, sigma=s)
            F = np.stack([G.radial_profiles(G.RadialHistogram(
                d["hist_" + tag], _pairs(n_f), CRYSTAL_REACH, float(d["volume"])), spec)
                for d in data])
            stack.append(radial_kernel(F, spec.n_bins, n_f))
        out[f"{tag}_radial"] = np.stack(stack)
        X = [np.stack([d[tag][f].ravel().astype(float) - (d[tag][f].mean() if CENTRE[q] else 0)
                       for d in data]) for f, q in enumerate(CRYSTAL_FIELDS)]
        Ks = [x @ x.T for x in X]
        for f, q in enumerate(CRYSTAL_FIELDS):
            out[f"{tag}_overlap_{q}"] = cosine(Ks[f])
        out[f"{tag}_overlap"] = joint_cosine(Ks)
        log(f"  {tag}: radial and overlap kernels")
    sub_atoms = [atoms[i] for i in keep]
    sub_names = [names[i] for i in keep]
    from cmfqsar import periodic
    fs = periodic.inorganic_fieldset(elements=[], physchem=P.FIELDS, charge_model=rec["charges"],
                                     alpha=1.0)
    given = periodic.crystal_table(sub_atoms, fs, sub_names)
    cell = np.asarray(rec["reference_cell"], float)
    one = periodic.crystal_table([P.reframed(a, cell) for a in sub_atoms], fs, sub_names)
    out["fields_radial"] = np.stack([P.joint_autocorrelation(given, fs, s) for s in P.AC_SIGMAS])
    log("  fields: periodic autocorrelation")
    stack = []
    for a in OVERLAP_ALPHAS:
        t0 = time.time()
        stack.append(P.joint_overlap(one, fs, a, True))
        log(f"  fields: periodic overlap on one lattice, alpha {a:g} ({time.time() - t0:.0f} s)")
    out["fields_overlap"] = np.stack(stack)
    out["composition"] = composition_kernel(sub_atoms)
    out["fit_error"] = np.stack([d["fit_error"] for d in data])
    if "wide" in data[0]:
        out["wide_error"] = np.stack([d["wide_error"] for d in data])
    out["electrons"] = np.asarray([float(d["electrons"]) for d in data])
    np.savez(path, **out)
    return out, y[keep], sub_names


def composition_kernel(atoms) -> np.ndarray:
    """The reference that knows only which elements a cell holds: cosine of the element counts.

    Two cells with the same A and B elements get 1, cells that share one of them about 0.8 (the
    three oxygens are shared by almost all), so the kernel is the composition and nothing else.
    It is the one-number reference for anything a DFT grid could know about the elements.
    """
    elements = sorted({s for a in atoms for s in a.get_chemical_symbols()})
    col = {e: i for i, e in enumerate(elements)}
    X = np.zeros((len(atoms), len(elements)))
    for i, a in enumerate(atoms):
        for s in a.get_chemical_symbols():
            X[i, col[s]] += 1.0
    return cosine(X @ X.T)


def _pairs(n: int):
    from cmfqsar.invariant import field_pairs
    return field_pairs(n)


def stage_crystals(force: bool = False) -> dict:
    """The crystal measurement: grid against atom-centred kernels, both frames."""
    K, y, names = crystal_kernels(force=force)
    sig = [f"sigma {s:g}" for s in CRYSTAL_SIGMAS]
    alp = [f"alpha {a:g}" for a in OVERLAP_ALPHAS]
    import run_periodic_overlap as P
    acs = [f"sigma {s:g}" for s in P.AC_SIGMAS]
    models = {"grid radial": ("one", K["grid_radial"]),
              "fitted grid radial": ("one", K["fitted_radial"]),
              "fields radial": ("one", K["fields_radial"]),
              "grid overlap": ("sum", K["grid_overlap"]),
              "fitted grid overlap": ("sum", K["fitted_overlap"]),
              "fields overlap": ("one", K["fields_overlap"]),
              "composition": ("sum", K["composition"]),
              "grid density overlap": ("sum", K["grid_overlap_density"]),
              "grid elf overlap": ("sum", K["grid_overlap_elf"]),
              "grid potential overlap": ("sum", K["grid_overlap_potential"]),
              "fields and grid radial": ("one", 0.5 * (K["fields_radial"] + K["grid_radial"])),
              "fields and grid overlap": ("one", 0.5 * (K["fields_overlap"]
                                                        + K["grid_overlap"][None]))}
    labels = {"grid radial": sig, "fitted grid radial": sig, "fields radial": acs,
              "fields overlap": alp, "fields and grid radial": sig,
              "fields and grid overlap": alp}
    pairs = [("grid radial", "fields radial"),
             ("grid overlap", "fields overlap"),
             ("grid radial", "fitted grid radial"),
             ("grid overlap", "fitted grid overlap"),
             ("fields and grid radial", "fields radial"),
             ("fields and grid overlap", "fields overlap"),
             ("grid overlap", "composition")]
    if "wide_overlap" in K:
        models[WIDE + " radial"] = ("one", K["wide_radial"])
        models[WIDE + " overlap"] = ("sum", K["wide_overlap"])
        labels[WIDE + " radial"] = sig
        pairs += [("grid radial", WIDE + " radial"), ("grid overlap", WIDE + " overlap")]
    t0 = time.time()
    res = cross_validate(models, y, labels)
    log(f"crystals: cross-validation of {len(models)} models on {len(y)} cells, "
        f"{time.time() - t0:.0f} s")
    rows = compare(res, y, pairs)
    off = ~np.eye(len(y), dtype=bool)
    corr = {f"{a} / {b}": float(np.corrcoef(K[a][1][off] if K[a].ndim == 3 else K[a][off],
                                            K[b][1][off] if K[b].ndim == 3 else K[b][off])[0, 1])
            for a, b in (("grid_radial", "fields_radial"), ("grid_radial", "fitted_radial"),
                         ("grid_overlap", "fields_overlap"), ("grid_overlap", "fitted_overlap"))}
    fit = K["fit_error"]
    payload = {"cells": int(len(y)), "target": "formation energy of the cell, eV",
               "y_mean": float(np.mean(y)), "y_sd": float(np.std(y)),
               "models": strip(res), "comparisons": rows, "kernel_correlations": corr,
               "fit_error_mean": {f: float(np.mean(fit[:, i])) for i, f in
                                  enumerate(CRYSTAL_FIELDS)},
               "electrons_mean": float(np.mean(K["electrons"])),
               "gate": {"radial": _passes(rows[2]), "overlap": _passes(rows[3])},
               "folds": FOLDS, "repeats": REPEATS, "cell_shape": list(CELL_SHAPE),
               "sigmas": list(CRYSTAL_SIGMAS), "r_max": CRYSTAL_R_MAX,
               "fit_widths": list(FIT_WIDTHS)}
    if "wide_error" in K:
        payload["wide_widths"] = list(WIDE_WIDTHS)
        payload["wide_error_mean"] = {f: float(np.mean(K["wide_error"][:, i]))
                                      for i, f in enumerate(CRYSTAL_FIELDS)}
    jdump(payload, os.path.join(RESULTS, "crystals.json"))
    for m, r in res.items():
        log(f"  {m:24s} R2 {r['q2_mean']:+.3f}  {r['chosen']}")
    for r in rows:
        log(f"  {r['model']} - {r['against']}: {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    return payload


# --------------------------------------------------------------------------- #
# Crystals: the energy of the same DFT run as the reference
# --------------------------------------------------------------------------- #

#: the energies of the static runs, one row per cell, written by stage `energies`
ENERGIES = os.path.join(RESULTS, "run_energies.csv")


def outcar_energy(path: str) -> Optional[float]:
    """The total energy of a VASP run, in eV, or None: the last value of energy(sigma->0) in its
    OUTCAR (plain or gzip), the energy extrapolated to zero smearing."""
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    value = None
    try:
        with opener(path, "rt", errors="ignore") as fh:
            for line in fh:
                if "energy(sigma->0)" in line:
                    value = float(line.split("energy(sigma->0)")[1].split("=")[1].split()[0])
    except (OSError, EOFError, ValueError, IndexError):
        return None
    return value


def stage_energies(vasp_dir: str) -> dict:
    """The total energy of every converged static run: the number the grid's run also gives."""
    _atoms, names, _y, _rec = m4_set()
    rows, status = [], {"ok": 0, "no converged run": 0, "no energy": 0}
    for n in names:
        run = os.path.join(vasp_dir, n)
        done = os.path.join(run, "done.json")
        if not os.path.isfile(done) or not jload(done).get("ok"):
            status["no converged run"] += 1
            continue
        e = outcar_energy(os.path.join(run, "OUTCAR.gz"))
        if e is None:
            status["no energy"] += 1
            continue
        rows.append((safe(n), e))
        status["ok"] += 1
    os.makedirs(RESULTS, exist_ok=True)
    with open(ENERGIES, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("cell,energy_eV\n")
        for n, e in rows:
            fh.write(f"{n},{e:.8f}\n")
    log(f"energies: {status}")
    return status


def run_energy_columns(names: Sequence[str], atoms) -> tuple:
    """(the energy of each cell's run, its element counts, the element names), in that order."""
    energy = {}
    with open(ENERGIES, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            n, e = line.strip().split(",")
            energy[n] = float(e)
    E = np.asarray([energy[safe(n)] for n in names], float)
    elements = sorted({s for a in atoms for s in a.get_chemical_symbols()})
    col = {e: i for i, e in enumerate(elements)}
    C = np.zeros((len(atoms), len(elements)))
    for i, a in enumerate(atoms):
        for s in a.get_chemical_symbols():
            C[i, col[s]] += 1.0
    return E, C, elements


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """The kernel of a ridge regression on the standardised columns of X."""
    X = np.asarray(X, float)
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return Z @ Z.T


def stage_reference() -> dict:
    """The grid against a linear regression on the energy of its own DFT run and the elements.

    The grid of a cell costs one DFT calculation, and that calculation also gives a total energy.
    The target is a formation energy, which is a total energy minus the energies of the elements.
    So a linear regression on the run's energy and the element counts is the reference the grid
    has to beat: it needs nothing that the grid does not already need.
    """
    K, y, names = crystal_kernels()
    atoms, all_names, _y, _rec = m4_set()
    by_name = dict(zip(all_names, atoms))
    sub = [by_name[n] for n in names]
    E, C, elements = run_energy_columns(names, sub)
    lin = linear_kernel(np.c_[E, C])
    both = 0.5 * (K["grid_overlap"] / np.mean(np.diag(K["grid_overlap"]))
                  + lin / np.mean(np.diag(lin)))
    models = {"grid overlap": ("sum", K["grid_overlap"]),
              "energy of the run": ("sum", linear_kernel(E[:, None])),
              "energy of the run and element counts": ("sum", lin),
              "grid overlap and energy of the run": ("sum", both)}
    t0 = time.time()
    res = cross_validate(models, y, {})
    rows = compare(res, y, [("energy of the run and element counts", "grid overlap"),
                            ("grid overlap and energy of the run",
                             "energy of the run and element counts")])
    A = np.c_[np.ones(len(y)), E, C]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    payload = {"cells": int(len(y)), "elements": len(elements), "models": strip(res),
               "comparisons": rows, "slope_on_energy": float(coef[1]),
               "seconds": round(time.time() - t0, 1), "folds": FOLDS, "repeats": REPEATS}
    jdump(payload, os.path.join(RESULTS, "reference.json"))
    for m, r in res.items():
        log(f"  {m:40s} R2 {r['q2_mean']:+.3f}  {r['chosen']}")
    for r in rows:
        log(f"  {r['model']} - {r['against']}: {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    return payload


# --------------------------------------------------------------------------- #
# Molecules: orca_plot grids of the corrosion series
# --------------------------------------------------------------------------- #

def series(set_name: str, qm_results: Optional[str] = None):
    """The molecules of one series with a finished quantum job: (mols, names, y, job dirs)."""
    from rdkit import Chem

    import run_cdft_kernels as CK
    import run_corrosion_qm as Q

    qm_results = qm_results or Q.RESULTS
    mols, names, y, _core, _ref, _rmsd = Q.prepare(os.path.join(Q.C.DATA, set_name + ".csv"))
    sdf = os.path.join(Q.RESULTS, set_name, f"{set_name}_qm.sdf")
    annotated = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(sdf, removeHs=False)}
    workdir = os.path.join(qm_results, set_name, "orca")
    dirs = CK.job_dirs(workdir) if os.path.isdir(workdir) else {}
    keep = [i for i, n in enumerate(names) if n in annotated]
    return ([annotated[names[i]] for i in keep], [names[i] for i in keep], y[keep],
            {names[i]: dirs.get(names[i]) for i in keep})


def _cube_one(args):
    """Density and potential cubes of one molecule, kept as a compressed npz; cubes deleted."""
    from cmfqsar.cdft import run_orca_plot

    name, job, out, coords = args
    if os.path.isfile(out):
        return name, "cached"
    if job is None:
        return name, "no quantum job"
    gbw = os.path.join(job, "sp_n.gbw")
    grids, made = {}, []
    for kind, plot in (("density", PLOT_DENSITY), ("potential", PLOT_POTENTIAL)):
        cube = run_orca_plot(gbw, kind=plot, intervals=INTERVALS)
        made.append(cube)
        grids[kind] = G.read_cube(cube, kind=kind)
    d = grids["density"]
    dev = float(np.max(np.abs(d.positions - coords)))
    np.savez_compressed(out, density=d.values.astype(np.float32),
                        potential=grids["potential"].values.astype(np.float32),
                        cell=d.cell, origin=d.origin, positions=d.positions,
                        symbols=np.asarray(d.symbols), max_deviation=dev)
    for c in made:
        os.remove(c)
    return name, "ok"


def stage_cubes(qm_results: Optional[str] = None, jobs: int = 6, sets=SETS) -> dict:
    status: Dict[str, int] = {}
    for s in sets:
        mols, names, _y, dirs = series(s, qm_results)
        os.makedirs(os.path.join(CACHE, "molecules", s), exist_ok=True)
        tasks = [(n, dirs[n], os.path.join(CACHE, "molecules", s, safe(n) + ".npz"),
                  m.GetConformer().GetPositions()) for m, n in zip(mols, names)]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            for name, st in ex.map(_cube_one, tasks):
                status[st] = status.get(st, 0) + 1
        log(f"cubes {s}: {len(tasks)} molecules, {time.time() - t0:.0f} s, {status}")
    return status


def molecule_grids(set_name: str, names: Sequence[str]) -> List[Dict[str, G.GridField]]:
    """The windowed density and potential of every molecule of a series, on its own grid."""
    out = []
    for n in names:
        z = np.load(os.path.join(CACHE, "molecules", set_name, safe(n) + ".npz"))
        if float(z["max_deviation"]) > 0.01:
            raise ValueError(f"{n}: the cube's atoms are {float(z['max_deviation']):.3f} A from "
                             "the aligned structure; the grid is not in the series' frame")
        rho = G.GridField(z["density"].astype(float), z["cell"], z["origin"], False,
                          [str(s) for s in z["symbols"]], z["positions"], "density", n)
        pot = rho.with_values(z["potential"].astype(float), kind="potential")
        dens = rho.with_values(rho.values * valence_window(rho.values))
        surf = 1.0 / (1.0 + (rho.values / RHO_SURFACE) ** 2)
        out.append({"density": dens, "potential": pot.with_values(pot.values * surf)})
    return out


def fitted_molecule(g: Dict[str, G.GridField]) -> Dict[str, G.GridField]:
    return {k: G.fit_atom_gaussians(v, FIT_WIDTHS)[0] for k, v in g.items()}


def molecule_kernels(set_name: str, mols, names) -> Dict[str, object]:
    """Grid, fitted-grid and atom-centred kernels of one series."""
    import run_cdft_kernels as CK
    import run_topomer as T
    from cmfqsar.fields import FieldSet, build_fields
    from cmfqsar.invariant import AutocorrelationSpec, block_sizes, feature_matrix

    grids = molecule_grids(set_name, names)
    fits = [fitted_molecule(g) for g in grids]
    errors = {k: float(np.mean([np.linalg.norm(g[k].values - f[k].values)
                                / max(np.linalg.norm(g[k].values), 1e-300)
                                for g, f in zip(grids, fits)])) for k in MOL_FIELDS}
    origin, shape = G.common_box([g["density"] for g in grids], spacing=MOL_SPACING)
    K: Dict[str, object] = {"fit_error": errors, "box": [int(v) for v in shape]}
    for tag, src in (("grid", grids), ("fitted", fits)):
        per = []
        for k in MOL_FIELDS:
            boxed = [G.on_common_box(g[k], origin, shape, MOL_SPACING) for g in src]
            per.append(G.grid_overlap_kernel(boxed, normalize=False))
            K[f"{tag}_overlap_{k}"] = cosine(per[-1])
        K[f"{tag}_overlap"] = joint_cosine(per)
        hists = [G.radial_histogram([g[k] for k in MOL_FIELDS], MOL_REACH) for g in src]
        stack = []
        for s in MOL_SIGMAS:
            spec = G.RadialSpec(r_max=MOL_R_MAX, step=s, sigma=s)
            F = np.stack([G.radial_profiles(h, spec) for h in hists])
            stack.append(radial_kernel(F, spec.n_bins, len(MOL_FIELDS)))
        K[f"{tag}_radial"] = np.stack(stack)
    base = CK.base_kernels(mols, names, alphas=MOL_ALPHAS)
    K["fields_overlap"] = np.stack([base[a] for a in MOL_ALPHAS])
    K["alphas"] = list(MOL_ALPHAS)
    fs = FieldSet.create(physchem=T.FIELDS, alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    stack = []
    for s in MOL_SIGMAS:
        spec = AutocorrelationSpec(d_max=MOL_R_MAX, step=s, sigma=s)
        F = feature_matrix(mf, spec, fieldset=fs)
        blocks = block_sizes(spec, len(fs.specs))
        sc = channel_scale(F, spec.n_bins, blocks=blocks)
        stack.append(autocorrelation_kernel(F, scale=sc, n_bins=spec.n_bins, blocks=blocks))
    K["fields_radial"] = np.stack(stack)
    return K


def stage_molecules(sets=SETS) -> dict:
    out = {}
    for s in sets:
        t0 = time.time()
        mols, names, y, _dirs = series(s)
        K = molecule_kernels(s, mols, names)
        sig = [f"sigma {v:g}" for v in MOL_SIGMAS]
        alp = [f"alpha {a:g}" for a in K["alphas"]]
        models = {"grid overlap": ("sum", K["grid_overlap"]),
                  "fitted grid overlap": ("sum", K["fitted_overlap"]),
                  "fields overlap": ("one", K["fields_overlap"]),
                  "grid density overlap": ("sum", K["grid_overlap_density"]),
                  "grid potential overlap": ("sum", K["grid_overlap_potential"]),
                  "grid radial": ("one", K["grid_radial"]),
                  "fitted grid radial": ("one", K["fitted_radial"]),
                  "fields radial": ("one", K["fields_radial"])}
        labels = {"fields overlap": alp, "grid radial": sig, "fitted grid radial": sig,
                  "fields radial": sig}
        res = cross_validate(models, y, labels)
        rows = compare(res, y, [("grid overlap", "fields overlap"),
                                ("grid radial", "fields radial"),
                                ("grid overlap", "fitted grid overlap"),
                                ("grid radial", "fitted grid radial")])
        off = ~np.eye(len(y), dtype=bool)
        corr = {"grid / fitted overlap": float(np.corrcoef(K["grid_overlap"][off],
                                                           K["fitted_overlap"][off])[0, 1]),
                "grid / fields overlap": float(np.corrcoef(K["grid_overlap"][off],
                                                           K["fields_overlap"][1][off])[0, 1])}
        payload = {"set": s, "molecules": int(len(y)), "models": strip(res), "comparisons": rows,
                   "fit_error_mean": K["fit_error"], "box": K["box"],
                   "kernel_correlations": corr, "seconds": round(time.time() - t0, 1)}
        jdump(payload, os.path.join(RESULTS, f"molecules_{s}.json"))
        log(f"molecules {s}: {len(y)} molecules, {payload['seconds']:.0f} s")
        for m, r in res.items():
            log(f"  {m:24s} q2 {r['q2_mean']:+.3f}  {r['chosen']}")
        for r in rows:
            log(f"  {r['model']} - {r['against']}: {r['mean']:+.3f} "
                f"[{r['lo']:+.3f}, {r['hi']:+.3f}]")
        out[s] = payload
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def _iv(r: dict) -> str:
    return f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def _role(row: dict) -> str:
    """What a comparison decides: the gate of rule 3, or the second question."""
    if row["against"].startswith("fitted"):
        return "the gate (grid against its own atom-centred fit)"
    if row["against"].startswith(WIDE):
        return "check of the gate (a richer atom-centred fit)"
    if row["model"].startswith("fields and"):
        return "whether the grid adds to the fields"
    if row["against"] == "composition":
        return "whether the grid knows more than the elements"
    return "second question (DFT grid against the empirical fields)"


def report(path: Optional[str] = None) -> str:
    lines = ["# Grid fields from DFT against the fitted atom-centred fields (plan item M9)", ""]
    cpath = os.path.join(RESULTS, "crystals.json")
    if os.path.isfile(cpath):
        c = jload(cpath)
        m = c["models"]
        lines += ["## Crystals", "",
                  f"{c['cells']} ABO3 perovskite cells of item M4, formation energy of the cell "
                  f"(mean {c['y_mean']:+.3f} eV, spread {c['y_sd']:.3f} eV). Grids from one "
                  "static PBE calculation per cell in VASP, on "
                  f"{' x '.join(map(str, c['cell_shape']))} points of the cell. Five folds "
                  "repeated three times; the setting and the ridge chosen by the evidence on "
                  "the training block.", "",
                  "| Kernel | Frame | R², mean of 3 repeats | RMSE, eV | "
                  "Settings chosen (folds) |",
                  "|---|---|---:|---:|---|"]
        frame = {"radial": "none (radial)", "overlap": "one lattice"}
        for name in ("grid radial", "fitted grid radial", "fields radial",
                     "fields and grid radial", "grid overlap", "fitted grid overlap",
                     "fields overlap", "fields and grid overlap", "composition",
                     "grid density overlap", "grid elf overlap",
                     "grid potential overlap", WIDE + " radial", WIDE + " overlap"):
            if name not in m:
                continue
            r = m[name]
            fr = ("none (element counts)" if name == "composition"
                  else frame["radial" if "radial" in name else "overlap"])
            ch = ", ".join(f"{k} ({v})" for k, v in r["chosen"].items())
            lines.append(f"| {name} | {fr} | {r['q2_mean']:+.3f} | {r['rmse_mean']:.4f} | {ch} |")
        lines += ["", "| Difference of R² | What it decides | Mean [95 % interval] | "
                  "Interval above zero |", "|---|---|---|---|"]
        for r in c["comparisons"]:
            lines.append(f"| {r['model']} minus {r['against']} | {_role(r)} | {_iv(r)} | "
                         f"{'yes' if r['above_zero'] else 'no'} |")
        fe = c["fit_error_mean"]
        lines += ["", "Relative error of the atom-centred fit, mean over the cells: "
                  + ", ".join(f"{k} {v:.1%}" for k, v in fe.items())
                  + (", with seven widths " + ", ".join(
                      f"{k} {v:.1%}" for k, v in c["wide_error_mean"].items())
                     if "wide_error_mean" in c else "")
                  + f". Mean electron count of the density grids: {c['electrons_mean']:.2f}.",
                  "", "Correlation of the off-diagonal kernel values: "
                  + "; ".join(f"{k} {v:+.3f}" for k, v in c["kernel_correlations"].items())
                  + ".", ""]
    rpath = os.path.join(RESULTS, "reference.json")
    if os.path.isfile(rpath):
        r = jload(rpath)
        lines += ["### The energy of the same run as the reference", "",
                  "The grid of a cell costs one DFT calculation, and that calculation also gives "
                  "a total energy. The reference is a ridge regression on that energy and the "
                  f"counts of the {r['elements']} elements, on the same folds; it needs nothing "
                  "the grid does not already need.", "",
                  "| Model | R², mean of 3 repeats | RMSE, eV | Ridge chosen (folds) |",
                  "|---|---:|---:|---|"]
        for name, mr in r["models"].items():
            ch = ", ".join(f"{k} ({v})" for k, v in mr["chosen"].items())
            lines.append(f"| {name} | {mr['q2_mean']:+.3f} | {mr['rmse_mean']:.4f} | {ch} |")
        lines += ["", "| Difference of R² | Mean [95 % interval] |", "|---|---|"]
        for cmp_row in r["comparisons"]:
            lines.append(f"| {cmp_row['model']} minus {cmp_row['against']} | {_iv(cmp_row)} |")
        lines += ["", f"Slope of the target on the energy of the run, in one least-squares fit "
                  f"with the element counts: {r['slope_on_energy']:.3f}.", ""]
    rows = []
    for s in SETS:
        p = os.path.join(RESULTS, f"molecules_{s}.json")
        if os.path.isfile(p):
            rows.append(jload(p))
    if rows:
        lines += ["## Molecules", "",
                  "The four corrosion-inhibitor series with ORCA single points. Grids of the "
                  f"neutral molecule from orca_plot on {INTERVALS + 1}³ points; the density "
                  "without its cores, the potential outside the molecule. q² from five folds "
                  "repeated three times.", "",
                  "| Series | Molecules | Grid overlap | Fitted grid overlap | Fields overlap | "
                  "Grid radial | Fitted grid radial | Fields radial |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            m = r["models"]
            lines.append(f"| {r['set']} | {r['molecules']} | " + " | ".join(
                f"{m[k]['q2_mean']:+.3f}" for k in ("grid overlap", "fitted grid overlap",
                                                    "fields overlap", "grid radial",
                                                    "fitted grid radial", "fields radial"))
                + " |")
        lines += ["", "| Series | Grid overlap minus fields overlap | Grid radial minus fields "
                  "radial | Grid overlap minus fitted grid overlap | Grid radial minus fitted "
                  "grid radial |", "|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['set']} | " + " | ".join(_iv(c) for c in r["comparisons"]) + " |")
        lines.append("")
    text = "\n".join(lines)
    path = path or os.path.join(RESULTS, "table.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("cells", "collect", "widefit", "crystals", "energies",
                                        "reference", "cubes", "molecules"),
                    action="append", help="stage(s) to run; the report always follows")
    ap.add_argument("--vasp-dir", help="the work directory of grid_dft_wsl.py (stages collect "
                    "and energies)")
    ap.add_argument("--qm-results", help="results directory of run_corrosion_qm.py that holds "
                    "the ORCA jobs (stage cubes); default: this checkout's")
    ap.add_argument("--jobs", type=int, default=6, help="parallel workers")
    ap.add_argument("--force", action="store_true", help="recompute the cached kernels")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    for st in args.stage or []:
        if st == "cells":
            stage_cells()
        elif st == "collect":
            if not args.vasp_dir:
                ap.error("--stage collect needs --vasp-dir")
            stage_collect(args.vasp_dir, args.jobs)
        elif st == "widefit":
            stage_widefit(args.jobs)
        elif st == "crystals":
            stage_crystals(force=args.force)
        elif st == "energies":
            if not args.vasp_dir:
                ap.error("--stage energies needs --vasp-dir")
            stage_energies(args.vasp_dir)
        elif st == "reference":
            stage_reference()
        elif st == "cubes":
            stage_cubes(args.qm_results, args.jobs)
        elif st == "molecules":
            stage_molecules()
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
