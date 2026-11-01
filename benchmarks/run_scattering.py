#!/usr/bin/env python
"""
A measured scattering curve as the sample: the total, untyped descriptor (plan item M12).

A diffraction experiment on a powder, a glass or a nanoparticle gives one curve and no
coordinates. `cmfqsar/scattering.py` shows that curve is not a different object from the
descriptors of this program but the *untyped* one: the reduced pair distribution function G(r)
is the periodic autocorrelation of a single field whose weight is the atom's scattering length.
This script asks the two questions that follow.

**What does summing away the types cost?** The typed descriptor has one channel block per pair of
field types; a measurement cannot separate them, because every atom scatters into the same
detector. Stage `fit` puts the total curve against the typed descriptor on the same structures,
the same folds and the same ridge grid. If the total keeps most of the accuracy, a measurement
is worth having; if it does not, the item stops there and says so.

**Would a real curve do?** A computed curve is sharper and cleaner than any measurement. Stage
`measured` puts computed curves through what an instrument does to them -- a finite range in Q,
counting noise, an error in the number density, a limited range in r -- and measures the
accuracy twice over: *matched*, where training and test curves are degraded alike, which is what
a laboratory with a set of measurements would have; and *transfer*, where the model is fitted on
computed curves and asked about degraded ones, which is what someone with a structure database
and one measurement in hand would do.

No experimental data is used anywhere here, and none is implied: there is no public set of
measured G(r) curves with measured properties to fit. What the degradation stage reports is how
much quality a measurement would have to have, not that a measurement worked.

The data and the protocol are the ones of `run_reciprocal.py` (plan item M6) -- the same three
Matbench tasks, the same subset, the same five published folds -- so the numbers here and there
can be read side by side.

Usage
-----
    python benchmarks/run_scattering.py                      # every stage, then the table
    python benchmarks/run_scattering.py --limit 400           # a quicker run
    python benchmarks/run_scattering.py --stage concept      # the figure that explains it
    python benchmarks/run_scattering.py --stage identity
    python benchmarks/run_scattering.py --stage fit
    python benchmarks/run_scattering.py --stage measured
    python benchmarks/run_scattering.py --stage reader
    python benchmarks/run_scattering.py --report
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar import scattering  # noqa: E402
import run_reciprocal as M6  # noqa: E402

FEATURES = os.path.join(HERE, "cache", "scattering")
RESULTS = os.path.join(HERE, "results", "scattering")

TASKS = M6.TASKS
LIMIT = 600
#: the grid the curves are reported on.  20 angstrom is the range a laboratory pair distribution
#: measurement reaches comfortably; 0.05 angstrom is finer than any instrument resolves, which is
#: what makes the degradation stage necessary.
CURVE = scattering.CurveSpec(r_min=1.0, r_max=20.0, r_step=0.05, sigma=0.05,
                             q_min=0.5, q_max=25.0, q_step=0.05)
#: the shorter range, for the part of the comparison that matches the real-space cutoffs of M6
SHORT = scattering.CurveSpec(r_min=1.0, r_max=10.0, r_step=0.05, sigma=0.05,
                             q_min=0.5, q_max=25.0, q_step=0.05)
CURVES: Dict[str, scattering.CurveSpec] = {"10 A": SHORT, "20 A": CURVE}
SOURCES = ("xray", "neutron")

#: what a measurement does, as the degradation stage varies it.  Each row is one setting; the
#: baseline keeps everything and is what the computed curve is.
DEGRADATIONS: Tuple[Tuple[str, dict], ...] = (
    ("as computed", {}),
    ("Q_max 25 1/A", {"q_max": 25.0}),
    ("Q_max 20 1/A", {"q_max": 20.0}),
    ("Q_max 15 1/A", {"q_max": 15.0}),
    ("Q_max 10 1/A", {"q_max": 10.0}),
    ("Q_max 20, noise 2 %", {"q_max": 20.0, "noise": 0.02}),
    ("Q_max 20, noise 10 %", {"q_max": 20.0, "noise": 0.10}),
    ("Q_max 20, density off 2 %", {"q_max": 20.0, "density_error": 0.02}),
    ("Q_max 20, density off 10 %", {"q_max": 20.0, "density_error": 0.10}),
    ("Q_max 20, r above 2 A", {"q_max": 20.0, "r_min": 2.0}),
    ("Q_max 20, r below 10 A", {"q_max": 20.0, "r_cut": 10.0}),
    ("a laboratory measurement", {"q_max": 20.0, "noise": 0.05, "density_error": 0.05,
                                  "r_min": 1.5, "r_cut": 15.0}),
)
#: structures whose identity residual is computed
IDENTITY_SAMPLE = 30


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def result(name: str) -> str:
    return os.path.join(RESULTS, name)


jdump, jload = M6.jdump, M6.jload


# --------------------------------------------------------------------------- #
# The curves of a set
# --------------------------------------------------------------------------- #
def usable(table, source: str) -> np.ndarray:
    """Which structures this source can describe: every element has to have a weight."""
    ok = np.zeros(len(table.molecules), bool)
    for k, m in enumerate(table.molecules):
        try:
            scattering.scattering_lengths(m.symbols, source)
            ok[k] = True
        except ValueError:
            ok[k] = False
    return ok


def curves_of(task: str, table, limit: int, source: str, name: str,
              spec: scattering.CurveSpec) -> np.ndarray:
    """G(r) of every structure of a task, computed once and kept in benchmarks/cache."""
    path = os.path.join(FEATURES, f"{task}_{limit}_{source}_{name.replace(' ', '')}.npz")
    if os.path.isfile(path):
        with np.load(path) as z:
            return z["F"]
    os.makedirs(FEATURES, exist_ok=True)
    t0 = time.time()
    ok = usable(table, source)
    F = np.full((len(table.molecules), len(spec.r)), np.nan)
    rows = np.flatnonzero(ok)
    F[rows] = scattering.computed_curves([table.molecules[int(i)] for i in rows], spec, source)
    np.savez_compressed(path, F=F)
    log(f"  {os.path.basename(path)}: {len(rows)} of {len(ok)} structures, "
        f"{time.time() - t0:.0f} s")
    return F


def kernel_of(F: np.ndarray) -> np.ndarray:
    return scattering.curve_kernel(F)


def degraded(F: np.ndarray, spec: scattering.CurveSpec, density: np.ndarray,
             how: dict, seed: int = 0) -> np.ndarray:
    """A whole set of curves put through one setting of :data:`DEGRADATIONS`."""
    r = spec.r
    out = np.full_like(F, np.nan)
    kw = dict(how)
    r_min = kw.pop("r_min", None)
    r_cut = kw.pop("r_cut", None)
    for k in range(F.shape[0]):
        if not np.all(np.isfinite(F[k])):
            continue
        row = scattering.measured_like(spec, r, F[k], density=float(density[k]),
                                       seed=seed + k, **kw)
        if r_min is not None:
            row = np.where(r >= r_min, row, 0.0)
        if r_cut is not None:
            row = np.where(r <= r_cut, row, 0.0)
        out[k] = row
    return out


# --------------------------------------------------------------------------- #
# The identity the item rests on
# --------------------------------------------------------------------------- #
def identity(tasks: Sequence[str] = tuple(TASKS), limit: int = LIMIT,
             force: bool = False) -> dict:
    """How far the descriptor route is from the definition of G(r), on real structures."""
    path = result("identity.json")
    if os.path.isfile(path) and not force:
        log(f"identity: {path} is there")
        return jload(path)
    out = {"sample": IDENTITY_SAMPLE, "tasks": {}}
    for task in tasks:
        table, _, _, _ = M6.load_set(task, limit)
        rng = np.random.RandomState(0)
        pick = rng.choice(len(table.molecules), min(IDENTITY_SAMPLE, len(table.molecules)),
                          replace=False)
        rec = {}
        for sigma in (0.1, 0.05, 0.02):
            spec = scattering.CurveSpec(r_min=1.0, r_max=10.0, r_step=sigma, sigma=sigma)
            worst = [scattering.pair_distribution_residual(table.molecules[int(i)], spec)["worst"]
                     for i in pick]
            rec[f"{sigma:g}"] = {"median": float(np.median(worst)), "worst": float(np.max(worst))}
            log(f"{task}, channel {sigma:g} A: median {np.median(worst):.1e}, "
                f"worst {np.max(worst):.1e}")
        out["tasks"][task] = rec
    jdump(out, path)
    return out


# --------------------------------------------------------------------------- #
# Stage A: what summing away the types costs
# --------------------------------------------------------------------------- #
def fit(tasks: Sequence[str] = tuple(TASKS), limit: int = LIMIT, force: bool = False) -> dict:
    path = result("fit.json")
    out = jload(path) if os.path.isfile(path) and not force else {}
    if out.get("limit") != int(limit):
        out = {"limit": int(limit)}
    for task in tasks:
        log(f"{task}: {TASKS[task]['label']} ({TASKS[task]['unit']})")
        table, y, fs, _ = M6.load_set(task, limit)
        rec = out.setdefault(task, {"descriptors": {}})
        rec["n"] = int(len(y))
        Ks: Dict[str, np.ndarray] = {}
        settings: Dict[str, List[dict]] = {}
        # the typed descriptor, at the cutoffs item M6 measured
        for name, (r_max, sigmas) in M6.REAL.items():
            for sigma in sigmas:
                F, blocks = M6.real_features(task, table, limit, r_max, sigma)
                Ks[f"typed {name} sigma {sigma:g}"] = M6.kernel_of(F, blocks)
        settings["the typed descriptor"] = [
            {"keys": [f"typed {name} sigma {s:g}"], "t": 0.0}
            for name, (_, sigmas) in M6.REAL.items() for s in sigmas]
        # the total curve.  A source that has no weight for one of the elements of a structure
        # cannot describe it, so that source is measured on the structures it covers, and the
        # typed descriptor is measured on the same ones: a comparison between two sets of
        # structures would say nothing about the descriptors.
        rows: Dict[str, np.ndarray] = {}
        for source in SOURCES:
            keys = []
            covered = np.ones(len(y), bool)
            for name, spec in CURVES.items():
                F = curves_of(task, table, limit, source, name, spec)
                covered &= np.all(np.isfinite(F), axis=1)
                Ks[f"total {source} {name}"] = kernel_of(np.nan_to_num(F))
                keys.append({"keys": [f"total {source} {name}"], "t": 0.0})
            rec[f"coverage_{source}"] = float(np.mean(covered))
            rows[source] = np.flatnonzero(covered)
            if covered.all():
                settings[f"the total curve, {source}"] = keys
            else:
                log(f"  {source}: {int(covered.sum())} of {len(y)} structures have a weight for "
                    f"every element; it and its baseline are measured on those")
                settings[f"the total curve, {source} (covered subset)"] = keys
                settings[f"the typed descriptor ({source} subset)"] = settings[
                    "the typed descriptor"]
        for name, sett in settings.items():
            if name in rec["descriptors"] and not force:
                log(f"  {name}: there")
                continue
            log(f"  {name}")
            take = None
            for source in SOURCES:
                if name.endswith(f"{source} subset)") or name.endswith(
                        f"{source} (covered subset)"):
                    take = rows[source]
            rec["descriptors"][name] = evaluate_on(Ks, sett, y, take)
            jdump(out, path)
        # what the typing is worth, with an interval
        rec["cost_of_the_total"] = {}
        for source in SOURCES:
            full = rec["coverage_" + source] >= 1.0
            total = rec["descriptors"].get(
                f"the total curve, {source}" if full
                else f"the total curve, {source} (covered subset)")
            typed = rec["descriptors"].get(
                "the typed descriptor" if full else f"the typed descriptor ({source} subset)")
            if typed is None or total is None:
                continue
            d, lo, hi = M6.bootstrap_difference(y, np.asarray(total["predictions"]),
                                                np.asarray(typed["predictions"]))
            rec["cost_of_the_total"][source] = {"d_r2": d, "low": lo, "high": hi,
                                                "on": "all" if full else "the covered subset"}
            log(f"  the total curve ({source}) against the typed descriptor: "
                f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        jdump(out, path)
        del Ks
    return out


def evaluate_on(Ks: Dict[str, np.ndarray], settings: List[dict], y: np.ndarray,
                rows=None) -> dict:
    """:func:`run_reciprocal.evaluate` on a subset of the structures, folds included.

    The predictions come back on the full length with "not a number" where a structure was left
    out, so that two descriptors measured on different subsets can still be compared row by row.
    """
    if rows is None:
        return M6.evaluate(Ks, settings, y)
    rows = np.asarray(rows, int)
    sub = {k: K[np.ix_(rows, rows)] for k, K in Ks.items()}
    rec = M6.evaluate(sub, settings, y[rows])
    full = np.full(len(y), np.nan)
    full[rows] = np.asarray(rec["predictions"], float)
    rec["predictions"] = full.tolist()
    rec["n"] = int(len(rows))
    return rec


# --------------------------------------------------------------------------- #
# Stage B: what a measurement would have to be
# --------------------------------------------------------------------------- #
def measured(tasks: Sequence[str] = tuple(TASKS), limit: int = LIMIT, source: str = "xray",
             force: bool = False) -> dict:
    path = result("measured.json")
    out = jload(path) if os.path.isfile(path) and not force else {}
    if out.get("limit") != int(limit) or out.get("source") != source:
        out = {"limit": int(limit), "source": source,
               "settings": [name for name, _ in DEGRADATIONS]}
    for task in tasks:
        log(f"{task}: what a measurement would have to be")
        table, y, _, _ = M6.load_set(task, limit)
        density = np.array([m.n_atoms / m.volume for m in table.molecules], float)
        clean = curves_of(task, table, limit, source, "20 A", CURVE)
        if not np.all(np.isfinite(clean)):
            log(f"  {source} cannot describe every structure of {task}; skipped")
            continue
        rec = out.setdefault(task, {"matched": {}, "transfer": {}})
        for name, how in DEGRADATIONS:
            if name in rec["matched"] and not force:
                continue
            rough = degraded(clean, CURVE, density, how)
            log(f"  {name}")
            rec["matched"][name] = M6.evaluate({"one": kernel_of(rough)},
                                               [{"keys": ["one"], "t": 0.0}], y)["r2"]
            # transfer: the model is fitted on the computed curves and asked about these
            rec["transfer"][name] = transfer_r2(clean, rough, y)
            rec["change"] = rec.get("change", {})
            rec["change"][name] = float(np.mean([
                np.corrcoef(clean[k], rough[k])[0, 1] for k in range(len(y))]))
            log(f"    matched R2 {rec['matched'][name]:+.3f}, "
                f"transfer R2 {rec['transfer'][name]:+.3f}, "
                f"curve correlation {rec['change'][name]:+.4f}")
            jdump(out, path)
        jdump(out, path)
    return out


def transfer_r2(clean: np.ndarray, rough: np.ndarray, y: np.ndarray) -> float:
    """Fit on the computed curves of the training fold, predict the degraded test curves.

    The ridge is chosen inside the training fold on the computed curves alone, because someone
    with a structure database and a measurement in hand has nothing else to choose it on.
    """
    from cmfqsar.model import _Learner, cv_predict, r2_score, rmse

    pred = np.full(len(y), np.nan)
    for tr, te in M6.folds(len(y)):
        K_tr = scattering.curve_kernel(clean[tr])
        best = None
        for lam in M6.LAMS:
            p = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=M6.INNER, seed=0)
            err = rmse(y[tr], p)
            if best is None or err < best[0]:
                best = (err, lam)
        learner = _Learner("krr", {"lam": best[1]}, True).fit(K_tr, y[tr])
        pred[te] = learner.predict(scattering.curve_kernel(clean[tr], rough[te]))
    return float(r2_score(y, pred))


# --------------------------------------------------------------------------- #
# Stage C: a file written and read back
# --------------------------------------------------------------------------- #
def reader(task: str = "log_kvrh", limit: int = LIMIT, force: bool = False) -> dict:
    """Write curves the way a measurement program writes them, read them back, and match."""
    path = result("reader.json")
    if os.path.isfile(path) and not force:
        log(f"reader: {path} is there")
        return jload(path)
    table, y, _, _ = M6.load_set(task, limit)
    cells = table.molecules[:25]
    folder = os.path.join(FEATURES, "written")
    os.makedirs(folder, exist_ok=True)
    computed = scattering.computed_curves(cells, CURVE, "xray")
    density = np.array([m.n_atoms / m.volume for m in cells], float)
    written = []
    for k, c in enumerate(cells):
        rough = scattering.measured_like(CURVE, CURVE.r, computed[k], q_max=20.0, noise=0.05,
                                         density_error=0.05, density=float(density[k]), seed=k)
        f = os.path.join(folder, f"{k:03d}.gr")
        with open(f, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# a curve written the way a measurement program writes one\n")
            fh.write(f"# qmax = 20.0\n# rho0 = {density[k]:.6f}\n#L r G(r)\n")
            for x, yv in zip(CURVE.r, rough):
                fh.write(f"{x:.4f} {yv:.6f}\n")
        written.append(f)
    read = [scattering.read_curve(f) for f in written]
    M = scattering.curve_matrix(read, CURVE.r)
    K = scattering.curve_kernel(computed, M)
    match = int(np.sum(np.argmax(K, axis=1) == np.arange(len(cells))))
    out = {"task": task, "n": len(cells), "folder": folder,
           "header_read": read[0]["header"],
           "matched_its_own_structure": match,
           "mean_kernel_with_its_own": float(np.mean(np.diag(K))),
           "mean_kernel_with_the_others": float(
               (K.sum() - np.trace(K)) / (K.size - len(cells)))}
    log(f"reader: {match} of {len(cells)} curves match the structure they were made from; "
        f"kernel with its own {out['mean_kernel_with_its_own']:.3f}, with the others "
        f"{out['mean_kernel_with_the_others']:.3f}")
    jdump(out, path)
    figure(task, computed, M, K)
    return out


def figure(task: str, computed: np.ndarray, measured_rows: np.ndarray, K: np.ndarray) -> None:
    """A computed curve, the same curve as a measurement would report it, and the match."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    r = CURVE.r
    ax.plot(r, computed[0], lw=1.3, color="#2f6fad", label="computed from the structure")
    ax.plot(r, measured_rows[0], lw=1.1, color="#c2562a",
            label="written out, read back as a measurement")
    ax.set_xlabel("r (A)")
    ax.set_ylabel("G(r)")
    ax.set_title(f"A curve and the same curve measured: {task}")
    ax.legend(fontsize=8, frameon=False)
    im = bx.imshow(K, cmap="RdBu_r", vmin=-1, vmax=1)
    bx.set_xlabel("computed curve")
    bx.set_ylabel("curve read from a file")
    bx.set_title("The kernel between the two sets")
    fig.colorbar(im, ax=bx, fraction=0.046)
    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    fig.savefig(result("reader.png"), dpi=150)
    plt.close(fig)
    log(f"figure: {result('reader.png')}")


# --------------------------------------------------------------------------- #
# The concept, drawn on one structure
# --------------------------------------------------------------------------- #
def concept(force: bool = False) -> dict:
    """One structure described four ways: typed, total, measured, and read back from a file."""
    path = result("concept.json")
    if os.path.isfile(path) and not force:
        log(f"concept: {path} is there")
        rec = jload(path)
        concept_figure(rec)
        return rec
    from ase import Atoms

    from cmfqsar import periodic

    fs = periodic.inorganic_fieldset(["Ba", "Ti", "O"], physchem=["prop:electronegativity"],
                                     charge_model="electronegativity", alpha=1.0)
    # a cubic perovskite, the shape of the Matbench set this item is measured on: A at the
    # corner, B at the centre, the three oxygens on the faces
    a = 4.00
    atoms = Atoms("BaTiO3", cell=np.eye(3) * a, pbc=True,
                  scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)])
    cell = periodic.crystal_fields(atoms, fs, "BaTiO3")
    # the typed descriptor: one block per pair of field types
    ps = periodic.PeriodicSpec(r_max=CURVE.r_max, step=0.05, sigma=0.05, taper=0.0,
                               include_self=False, background=True)
    typed = periodic.periodic_autocorrelation(cell, ps).reshape(-1, ps.n_bins)
    labels = [periodic.periodic_channel_labels(fs, ps)[k * ps.n_bins].split(" @ ")[0]
              for k in range(typed.shape[0])]
    # the total: the one block a detector can measure
    r, g = scattering.pair_distribution(cell, CURVE, "xray")
    density = cell.n_atoms / cell.volume
    rough = scattering.measured_like(CURVE, r, g, q_max=20.0, noise=0.05, density_error=0.05,
                                     density=density, seed=0)
    out = {"name": cell.name, "n_atoms": int(cell.n_atoms), "r": r.tolist(),
           "channels": ps.centres.tolist(), "typed": typed.tolist(), "labels": labels,
           "total": g.tolist(), "measured": rough.tolist(),
           "n_blocks": int(typed.shape[0]), "density": float(density),
           "kernel_total_measured": float(scattering.curve_kernel(
               g[None, :], rough[None, :])[0, 0])}
    jdump(out, path)
    log(f"concept: {out['n_blocks']} typed blocks collapse to one curve; the measured curve "
        f"matches the computed one at {out['kernel_total_measured']:.4f}")
    concept_figure(out)
    return out


def concept_figure(rec: dict) -> None:
    """Three panels: the typed blocks, the one curve they sum to, and the measurement of it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(14.5, 4.2))
    d = np.asarray(rec["channels"], float)
    T = np.asarray(rec["typed"], float)
    order = np.argsort(-np.abs(T).max(axis=1))[:6]
    for k in order:
        ax.plot(d, T[k], lw=1.0, label=rec["labels"][int(k)])
    ax.set_xlim(1.0, 10.0)
    ax.set_xlabel("distance r (A)")
    ax.set_ylabel("channel of that pair of field types")
    ax.set_title(f"(a) What the program computes:\n{rec['n_blocks']} typed blocks")
    ax.legend(fontsize=6.5, frameon=False, ncol=1)

    r = np.asarray(rec["r"], float)
    g = np.asarray(rec["total"], float)
    bx.plot(r, g, lw=1.3, color="#2f6fad")
    bx.axhline(0.0, color="0.7", lw=0.8)
    bx.set_xlim(1.0, 10.0)
    bx.set_xlabel("distance r (A)")
    bx.set_ylabel("G(r)")
    bx.set_title("(b) What a detector can measure:\none curve, the types summed away")

    rough = np.asarray(rec["measured"], float)
    cx.plot(r, g, lw=1.6, color="#2f6fad", alpha=0.45, label="computed")
    cx.plot(r, rough, lw=1.0, color="#c2562a",
            label="as an instrument would report it")
    cx.axhline(0.0, color="0.7", lw=0.8)
    cx.set_xlim(1.0, 10.0)
    cx.set_xlabel("distance r (A)")
    cx.set_ylabel("G(r)")
    cx.set_title("(c) What an instrument does to it:\nQ to 20 1/A, 5 % noise, 5 % density error")
    cx.legend(fontsize=8, frameon=False, loc="lower left")
    cx.text(0.02, 0.97, f"kernel between the two: {rec['kernel_total_measured']:.4f}",
            transform=cx.transAxes, ha="left", va="top", fontsize=8.5,
            bbox=dict(fc="white", ec="0.8", lw=0.6))
    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    fig.savefig(result("concept.png"), dpi=150)
    plt.close(fig)
    log(f"figure: {result('concept.png')}")


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def report() -> str:
    lines = ["# A measured scattering curve as the sample (plan item M12)", "",
             "No experimental data is used here. The curves are computed from structures and "
             "then put through what an instrument does to them, so what follows says how good a "
             "measurement would have to be, not that a measurement worked.", ""]
    fit_path, meas_path = result("fit.json"), result("measured.json")
    idn_path, rd_path = result("identity.json"), result("reader.json")
    if os.path.isfile(idn_path):
        rec = jload(idn_path)
        lines += ["## The curve is the untyped descriptor", "",
                  f"G(r) computed as the one-field periodic autocorrelation, against G(r) summed "
                  f"over the neighbour list from its definition, on {rec['sample']} structures of "
                  f"each task: the largest relative difference over the curve, median and worst "
                  f"over the structures. The identity is exact in the limit of a narrow channel, "
                  f"so the error falls with the channel width.", ""]
        widths = sorted({w for r in rec["tasks"].values() for w in r}, key=float, reverse=True)
        lines += ["| The set | " + " | ".join(f"channel {w} A" for w in widths) + " |",
                  "|---" + "|---:" * len(widths) + "|"]
        for t, r in rec["tasks"].items():
            cells = [f"{r[w]['median']:.1e} (worst {r[w]['worst']:.1e})" for w in widths]
            lines.append(f"| {TASKS[t]['label']} | " + " | ".join(cells) + " |")
        lines.append("")
    if os.path.isfile(fit_path):
        rec = jload(fit_path)
        tasks = [t for t in TASKS if t in rec]
        lines += ["## What summing away the types costs", "",
                  f"Nested cross-validation over the five published Matbench folds of "
                  f"{rec.get('limit', LIMIT)} structures of each task, the protocol of "
                  f"`run_reciprocal.py`. The typed descriptor is the periodic autocorrelation "
                  f"with one block per pair of field types; the total curve is G(r), which is "
                  f"the one block an experiment can measure.", ""]
        head = "| Descriptor | " + " | ".join(f"{TASKS[t]['label']}: R2 | MAE" for t in tasks) + " |"
        lines += [head, "|---" + "|---:" * (2 * len(tasks)) + "|"]
        names = ["the typed descriptor"]
        for s in SOURCES:
            for suffix in ("", " (covered subset)"):
                key = f"the total curve, {s}{suffix}"
                if any(key in rec[t]["descriptors"] for t in tasks):
                    names.append(key)
            key = f"the typed descriptor ({s} subset)"
            if any(key in rec[t]["descriptors"] for t in tasks):
                names.append(key)
        for name in names:
            cells = []
            for t in tasks:
                d = rec[t]["descriptors"].get(name)
                cells += ["n/a", "n/a"] if d is None else [f"{d['r2']:+.3f}", f"{d['mae']:.4f}"]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        lines += ["**The difference, with a 95 % interval from 2,000 bootstrap resamples of the "
                  "structures.** A negative number is what the typing is worth.", ""]
        lines += ["| The curve | " + " | ".join(TASKS[t]["label"] for t in tasks) + " |",
                  "|---" + "|---:" * len(tasks) + "|"]
        for source in SOURCES:
            cells = []
            for t in tasks:
                g = rec[t].get("cost_of_the_total", {}).get(source)
                cells.append("n/a" if g is None
                             else f"{g['d_r2']:+.3f} [{g['low']:+.3f}, {g['high']:+.3f}]")
            lines.append(f"| {source} | " + " | ".join(cells) + " |")
        lines.append("")
        cover = [(t, rec[t].get(f"coverage_{s}")) for t in tasks for s in SOURCES]
        if any(c is not None and c < 1.0 for _, c in cover):
            lines += ["Where a source covers fewer than all the structures, it is because the "
                      "module tabulates no scattering length for one of their elements and "
                      "refuses to guess one.", ""]
    if os.path.isfile(meas_path):
        rec = jload(meas_path)
        tasks = [t for t in TASKS if t in rec]
        lines += ["## What a measurement would have to be", "",
                  f"The same curves ({rec.get('source', 'xray')}) put through what an instrument "
                  f"does to them. **Matched** fits and tests on curves degraded alike, which is "
                  f"what a laboratory with a set of measurements has. **Transfer** fits on the "
                  f"computed curves and predicts the degraded ones, which is what someone with a "
                  f"structure database and a measurement in hand does. The last column is how "
                  f"much the curve itself changed, as its correlation with the computed one.", ""]
        head = ("| What the measurement does | "
                + " | ".join(f"{TASKS[t]['label']}: matched | transfer" for t in tasks)
                + " | Curve |")
        lines += [head, "|---" + "|---:" * (2 * len(tasks) + 1) + "|"]
        for name in rec.get("settings", [n for n, _ in DEGRADATIONS]):
            cells, corr = [], []
            for t in tasks:
                m = rec[t]["matched"].get(name)
                x = rec[t]["transfer"].get(name)
                cells += ["n/a", "n/a"] if m is None else [f"{m:+.3f}", f"{x:+.3f}"]
                if rec[t].get("change", {}).get(name) is not None:
                    corr.append(rec[t]["change"][name])
            tail = f" | {np.mean(corr):+.3f} |" if corr else " | n/a |"
            lines.append(f"| {name} | " + " | ".join(cells) + tail)
        lines.append("")
    if os.path.isfile(rd_path):
        rec = jload(rd_path)
        lines += ["## A file written and read back", "",
                  f"{rec['n']} curves of the {TASKS[rec['task']]['label']} set written in the "
                  f"two-column form the common programs write, with a header, then read back "
                  f"and matched against the structures they came from. "
                  f"{rec['matched_its_own_structure']} of {rec['n']} match their own structure "
                  f"best; the kernel with its own structure is "
                  f"{rec['mean_kernel_with_its_own']:.3f} on average and with the others "
                  f"{rec['mean_kernel_with_the_others']:.3f}. The header read back: "
                  f"{rec['header_read']}.", "",
                  "![A curve and the same curve measured](reader.png)", ""]
    if os.path.isfile(result("concept.json")):
        rec = jload(result("concept.json"))
        lines += ["## What a detector sums away", "",
                  f"One cell of {rec['name']} described three ways: the "
                  f"{rec['n_blocks']} typed channel blocks the program computes, the one curve "
                  f"a detector can measure, and that curve as an instrument would report it "
                  f"(Q to 20 1/A, 5 % counting noise, the number density wrong by 5 %). The "
                  f"kernel between the computed curve and the measured one is "
                  f"{rec['kernel_total_measured']:.4f}.", "",
                  "![What a detector sums away](concept.png)", ""]
    text = "\n".join(lines) + "\n"
    os.makedirs(RESULTS, exist_ok=True)
    with open(result("table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(f"table: {result('table.md')}")
    return text


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage",
                   choices=["concept", "identity", "fit", "measured", "reader"],
                   default=None)
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    p.add_argument("--limit", type=int, default=LIMIT)
    p.add_argument("--source", choices=list(SOURCES), default="xray")
    p.add_argument("--force", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args(argv)
    if a.report:
        print(report())
        return 0
    if a.stage in (None, "concept"):
        concept(a.force)
    if a.stage in (None, "identity"):
        identity(a.tasks, a.limit, a.force)
    if a.stage in (None, "fit"):
        fit(a.tasks, a.limit, a.force)
    if a.stage in (None, "measured"):
        measured(a.tasks, a.limit, a.source, a.force)
    if a.stage in (None, "reader"):
        reader("log_kvrh", a.limit, a.force)
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
