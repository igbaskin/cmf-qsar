#!/usr/bin/env python
"""
A site of a periodic cell as the sample, and a cell-level model read site by site (plan item M5).

The periodic autocorrelation of a cell is a sum over ordered pairs of its atoms, so it splits
exactly into one row per site (:mod:`cmfqsar.sites`).  That gives two things, and this script
measures both.

**A site as the sample.**  The row of site *l* is the typed radial profile of the cell seen from
that site: which field type stands how far from it, with the periodic images summed in and
nothing cut out of the solid.  It is a local descriptor that needs no alignment.  Two sets are
measured with it:

  single-atom catalysts   the 139 M@C(4-x)Nx sites of `examples/sac` (the data of the EC8 pilot,
                          `benchmarks/run_sac.py`), one measured site per cell, against the four
                          binding energies of H, O, OH and OOH.  The question EC8 left open is
                          exactly this one: does the site resolution of the periodic kernel do
                          what cutting a cluster out of the sheet does?
  lithium sites           the bond-valence site energy of every Li site of the three solid
                          electrolytes of `examples/hops` (plan item BT7): many sites per cell,
                          so the folds hold out whole compounds, and a fold is a compound the
                          model has never seen.

The references under rule 1 are recomputed here on the same folds and the same cells rather than
quoted: counts of the elements in distance shells around the site (the simplest thing that could
work), the periodic autocorrelation of the **whole cell** (the reference of items M1 to M14), and,
on the catalysts, the two site descriptors of the EC8 pilot -- the rotation-invariant
autocorrelation of the cut environment (item LP2) and the overlap kernel of the same environment.
The pilot's own numbers are not comparable with these, because five bare sheets have been
recomputed since it ran and its targets were 98 catalysts where these are 103.

**A cell-level model read site by site.**  For a model on the whole-cell descriptor the split
works in the other direction: the cell descriptor is the mean of its site rows and the linear
layer is linear, so the prediction minus the bias is the sum of one number per site
(:func:`cmfqsar.sites.site_contributions`), exactly.  On the catalysts that gives the local
co-pattern the autocorrelation kernel could not give: of the 127 sites of the sheet, which ones
carry the binding energy.  The measurement is the share of the total weight that falls on the
metal and on its first coordination shell, against the share those five sites would get if the
weight were spread evenly.

Usage
-----
    python benchmarks/run_sites.py                 # everything
    python benchmarks/run_sites.py --sac           # the catalysts alone
    python benchmarks/run_sites.py --lithium       # the Li site energies alone
    python benchmarks/run_sites.py --contributions # the site contributions and the figure
    python benchmarks/run_sites.py --report        # the table from the stored results
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar import sites as S  # noqa: E402
from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.fields import AtomTable  # noqa: E402
from cmfqsar.localenv import choose_lam, cross_validate_lams  # noqa: E402
from cmfqsar.model import CMFModel, r2_score  # noqa: E402
from cmfqsar.periodic import (PeriodicSpec, crystal_fields, inorganic_fieldset,  # noqa: E402
                              periodic_channel_labels, periodic_feature_matrix)

RESULTS = os.path.join(HERE, "results", "sites")
HOPS = os.path.join(ROOT, "examples", "hops")

#: the cutoffs of the site profile that the inner cross-validation chooses between, in angstrom
CUTOFFS = (4.0, 6.0, 8.0)
#: the ridge values, the same for every method
LAMS = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
SEED = 0
#: how far the shells of the count reference reach, in angstrom
SHELLS = (2.0, 3.0, 4.5, 6.0)
STEP = 0.5
SIGMA = 0.5


def spec_for(r_max: float) -> PeriodicSpec:
    return PeriodicSpec(r_max=float(r_max), step=STEP, sigma=SIGMA)


def linear_kernel(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    """The cosine-normalised linear kernel of a feature matrix, column-scaled as the
    autocorrelation kernel scales its channels."""
    A = np.asarray(X, float)
    sd = np.maximum(A.std(axis=0), 1e-12)
    A = A / sd
    K = A @ A.T
    if not normalize:
        return K
    d = np.sqrt(np.maximum(np.diag(K), 1e-300))
    return K / np.outer(d, d)


# =========================================================================== #
# The single-atom catalysts: one measured site per cell
# =========================================================================== #
def sac_kernels(data, crystals_by_kind: Dict[str, list], targets: AtomTargets,
                log=print) -> Dict[str, Dict[str, np.ndarray]]:
    """Every method as a family of candidate kernels over the catalysts."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    site_fam: Dict[str, np.ndarray] = {}
    for kind, crystals in crystals_by_kind.items():
        for r_max in CUTOFFS:
            t0 = time.time()
            X = S.site_features(crystals, targets, spec_for(r_max))
            site_fam[f"{kind} r={r_max:g}"] = S.site_kernel(X)
            log(f"    site profile, {kind}, r {r_max:g} A: {X.shape[1]} channels, "
                f"{time.time() - t0:.1f} s")
    out["site profile (M5)"] = site_fam
    cell_fam: Dict[str, np.ndarray] = {}
    from cmfqsar.invariant import autocorrelation_kernel
    for kind, crystals in crystals_by_kind.items():
        for r_max in CUTOFFS:
            t0 = time.time()
            F = periodic_feature_matrix(crystals, spec_for(r_max),
                                        n_fields=crystals[0].weights.shape[1])
            cell_fam[f"{kind} r={r_max:g}"] = autocorrelation_kernel(F)
            log(f"    whole cell, {kind}, r {r_max:g} A: {F.shape[1]} channels, "
                f"{time.time() - t0:.1f} s")
    out["whole cell autocorrelation"] = cell_fam
    counts: Dict[str, np.ndarray] = {}
    crystals = crystals_by_kind["classic"]
    elements = S.elements_of(crystals)
    for depth in (2, len(SHELLS)):
        t0 = time.time()
        X = S.shell_count_features(crystals, targets, elements, SHELLS[:depth])
        counts[f"{depth} shells"] = linear_kernel(X)
        log(f"    counts in {depth} shells: {X.shape[1]} columns, {time.time() - t0:.1f} s")
    out["counts around the site"] = counts
    return out


def sac_run(args, log) -> dict:
    import run_sac as R
    data = R.load_set(log)
    log(f"  {len(data)} catalysts, {len(set(data.metals))} metals, {len(set(data.motifs))} motifs")
    for note in data.notes[:3]:
        log("  note: " + note)
    sites = [R._metal_index(a) for a in data.structures]
    kinds = ("classic", "extended", "indicator")
    crystals_by_kind = {}
    for kind in kinds:
        fs = R.fieldset_for(kind, data.metals)
        crystals_by_kind[kind] = R.crystals_for(data, fs)
    n_sites = [c.n_atoms for c in crystals_by_kind["classic"]]
    # the index of the measured sites: one site of every cell, and the value is set per target
    index = S.one_site_targets(sites, np.zeros(len(data)), n_sites, name="binding")
    log("  building the kernels")
    families = sac_kernels(data, crystals_by_kind, index, log)
    # the two site descriptors of the EC8 pilot, on the same cells and the same folds
    t0 = time.time()
    overlap, invariant, _ = R.site_kernels(data, "extended", log=log)
    families["site overlap, cut cluster (EC8)"] = overlap
    families["site invariant, cut cluster (LP2)"] = invariant
    log(f"    the cut-cluster kernels of the pilot: {time.time() - t0:.0f} s")
    out = {"n": len(data), "cutoffs": list(CUTOFFS), "lams": list(LAMS), "shells": list(SHELLS),
           "notes": data.notes, "models": [], "pairs": [],
           "targets": {t: int(np.sum(np.isfinite(data.dE[a])))
                       for t, a in zip(R.TARGETS, R.ADSORBATES)}}
    for design in ("random", "metal-out", "motif-out"):
        for adsorbate in R.ADSORBATES:
            y = data.dE[adsorbate]
            target = f"dE_{adsorbate}"
            results = {}
            for name, fam in families.items():
                res = R.cross_validate(name, fam, y, design, target, data, seed=SEED, lams=LAMS)
                results[name] = res
                log("  " + res.summary())
                out["models"].append(res.entry())
            base = results["site profile (M5)"]
            for other in ("whole cell autocorrelation", "counts around the site",
                          "site overlap, cut cluster (EC8)",
                          "site invariant, cut cluster (LP2)"):
                d, lo, hi = R.bootstrap_difference(y, base.predicted, results[other].predicted,
                                                   runs=args.bootstrap, seed=SEED)
                out["pairs"].append({"design": design, "target": target, "against": other,
                                     "delta": d, "low": lo, "high": hi})
                log(f"    site profile against {other}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return out


# =========================================================================== #
# The lithium sites: many measured sites per cell
# =========================================================================== #
def lithium_targets(log=print) -> Tuple[list, AtomTargets, List[str]]:
    """The bond-valence site energy of every Li site of the three electrolytes.

    `examples/hops/hops_bvse.csv` holds one row per hop with the site energy of both ends, so a
    site appears in as many rows as it has hops; the value is the same in each and is taken once.
    """
    from ase.io import read
    names = ["phosphate", "borophosphate", "borate"]
    structures = [read(os.path.join(HOPS, f"{n}.extxyz")) for n in names]
    per = [np.full(len(a), np.nan) for a in structures]
    with io.open(os.path.join(HOPS, "hops_bvse.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            k = names.index(row["structure"])
            for which in ("a", "b"):
                site = int(row[which])
                value = float(row[f"site_{which}"])
                per[k][site] = value
    for k, n in enumerate(names):
        v = per[k]
        log(f"  {n}: {int(np.sum(np.isfinite(v)))} Li sites of {len(v)} atoms, site energy "
            f"{np.nanmin(v):+.3f} to {np.nanmax(v):+.3f} eV")
    return structures, AtomTargets.from_arrays(per, name="bond-valence site energy"), names


def lithium_run(args, log) -> dict:
    structures, targets, names = lithium_targets(log)
    fs = inorganic_fieldset(["Li", "B", "P", "O"], charge_model="file")
    crystals = [crystal_fields(a, fs, n) for a, n in zip(structures, names)]
    folds = len(names)                      # one compound out, which is what the index gives
    out = {"compounds": names, "n_sites": len(targets), "folds": folds, "models": [],
           "spread": float(np.std(targets.y))}
    best = {}
    for r_max in CUTOFFS:
        X = S.site_features(crystals, targets, spec_for(r_max))
        K = S.site_kernel(X)
        res = choose_lam(K, targets, LAMS, folds=folds, seed=SEED,
                         name=f"site profile (M5), r={r_max:g} A")
        log("  " + res.summary())
        out["models"].append({"name": res.name, "q2": res.q2, "rmse": res.error, "mae": res.mae,
                              "lam": res.lam, "n": res.n_samples})
        if not best or res.q2 > best["q2"]:
            best = {"q2": res.q2, "name": res.name, "res": res}
    elements = S.elements_of(crystals)
    Xc = S.shell_count_features(crystals, targets, elements, SHELLS)
    res = choose_lam(linear_kernel(Xc), targets, LAMS, folds=folds, seed=SEED,
                     name="counts around the site")
    log("  " + res.summary())
    out["models"].append({"name": res.name, "q2": res.q2, "rmse": res.error, "mae": res.mae,
                          "lam": res.lam, "n": res.n_samples})
    # the baseline a grouped fold cannot avoid: the mean site energy of the compounds in training
    y = np.asarray(targets.y, float)
    pred = np.full(len(y), np.nan)
    for te in targets.folds(folds, SEED):
        tr = np.setdiff1d(np.arange(len(y)), te)
        pred[te] = float(np.mean(y[tr]))
    out["models"].append({"name": "the mean of the other compounds", "q2": r2_score(y, pred),
                          "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
                          "mae": float(np.mean(np.abs(y - pred))), "lam": float("nan"),
                          "n": len(y)})
    log(f"  the mean of the other compounds: q2 {out['models'][-1]['q2']:+.3f}")
    # where the error sits: one number per compound, for the best cutoff
    res = best["res"]
    per_compound = {}
    for k, n in enumerate(names):
        rows = targets.of_molecule(k)
        ok = rows[np.isfinite(res.predicted[rows])]
        yk, pk = y[ok], res.predicted[ok]
        # the offset of a compound the model has never seen is what it cannot know; the ranking
        # inside the compound is what it can, so both are reported
        centred = r2_score(yk - yk.mean(), pk - pk.mean()) if np.std(yk) > 0 else float("nan")
        if len(ok) > 2 and np.std(yk) > 0:
            from scipy.stats import spearmanr
            rho = float(spearmanr(yk, pk)[0])
        else:
            rho = float("nan")
        per_compound[n] = {"n": int(len(ok)), "mae": float(np.mean(np.abs(yk - pk))),
                           "spread": float(np.std(yk)), "offset": float(np.mean(pk - yk)),
                           "q2_centred": float(centred), "spearman": rho}
        log(f"    {n}: {len(ok)} sites, mean absolute error {per_compound[n]['mae']:.3f} eV "
            f"against a spread of {per_compound[n]['spread']:.3f} eV; the offset of the whole "
            f"compound is {per_compound[n]['offset']:+.3f} eV, and inside the compound the "
            f"ranking has a Spearman correlation of {rho:+.2f} (q2 about its own mean "
            f"{centred:+.3f})")
    out["per_compound"] = per_compound
    out["best"] = best["name"]
    return out


# =========================================================================== #
# A whole-cell model read site by site
# =========================================================================== #
def contributions_run(args, log) -> dict:
    import run_sac as R
    data = R.load_set(lambda s: None)
    fs = R.fieldset_for("extended", data.metals)
    crystals = R.crystals_for(data, fs)
    sites = [R._metal_index(a) for a in data.structures]
    y = data.dE["OH"]
    ok = np.where(np.isfinite(y))[0]
    table = AtomTable.from_molecules([crystals[i] for i in ok])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_sigma=SIGMA, ac_d_max=6.0, params={"lam": 0.01})
    model.fit(table, y[ok], cv_folds=5)
    log(f"  the whole-cell model of dE_OH: q2 {model.stats.q2:+.3f}, "
        f"root mean square error {model.stats.rmse_cv:.3f} eV on {len(ok)} catalysts")
    out = {"n": int(len(ok)), "q2": float(model.stats.q2), "rmse": float(model.stats.rmse_cv),
           "shares": [], "cells": {}}
    shares = []
    for i in ok:
        c, m = crystals[i], sites[i]
        v = S.site_contributions(model, c)
        total = float(np.sum(np.abs(v)))
        d = c.to_atoms().get_distances(m, range(c.n_atoms), mic=True)
        first = np.where((d > 0) & (d < 2.4))[0]
        here = float(abs(v[m]) + np.sum(np.abs(v[first]))) / max(total, 1e-30)
        shares.append({"name": data.names[i], "metal": float(abs(v[m]) / max(total, 1e-30)),
                       "site_and_shell": here, "n_shell": int(len(first)),
                       "even": float((1 + len(first)) / c.n_atoms)})
    out["shares"] = shares
    sh = np.array([s["site_and_shell"] for s in shares])
    ev = np.array([s["even"] for s in shares])
    me = np.array([s["metal"] for s in shares])
    log(f"  the metal and its first shell carry {100 * sh.mean():.1f} % of the weight "
        f"(the metal alone {100 * me.mean():.1f} %), where an even spread over the "
        f"{crystals[0].n_atoms} sites would give {100 * ev.mean():.1f} %")
    out["summary"] = {"site_and_shell": float(sh.mean()), "metal": float(me.mean()),
                      "even": float(ev.mean()), "ratio": float(sh.mean() / max(ev.mean(), 1e-30))}
    # the weight by distance from the metal, averaged over the catalysts
    edges = np.array([0.0, 1.0, 2.4, 3.5, 5.0, 7.0, 1e3])
    band = np.zeros(len(edges) - 1)
    held = np.zeros(len(edges) - 1)          # how many sites each band holds, on average
    for i in ok:
        c, m = crystals[i], sites[i]
        v = np.abs(S.site_contributions(model, c))
        d = c.to_atoms().get_distances(m, range(c.n_atoms), mic=True)
        total = max(float(v.sum()), 1e-30)
        for b in range(len(edges) - 1):
            here = (d >= edges[b]) & (d < edges[b + 1])
            band[b] += float(v[here].sum()) / total
            held[b] += float(here.sum()) / c.n_atoms
    band /= len(ok)
    held /= len(ok)
    # the weight a band carries against the weight an even spread would put in the same band:
    # a band far from the metal holds many more sites, so the shares alone would mislead
    enrich = band / np.maximum(held, 1e-30)
    out["by_distance"] = {"edges": edges[:-1].tolist(), "share": band.tolist(),
                          "share_of_sites": held.tolist(), "enrichment": enrich.tolist()}
    for b in range(len(band)):
        top = edges[b + 1] if b + 1 < len(edges) - 1 else float("inf")
        log(f"    {edges[b]:.1f} to {top:.1f} A from the metal: {100 * band[b]:.1f} % of the "
            f"weight on {100 * held[b]:.1f} % of the sites, {enrich[b]:.1f} times an even spread")
    # two cells to show, and the channels of the metal site
    labels = periodic_channel_labels(fs, model._ac_spec())
    for pick in ("Fe@N4", "Pt@C4"):
        if pick not in data.names:
            continue
        i = data.names.index(pick)
        c, m = crystals[i], sites[i]
        v = S.site_contributions(model, c)
        out["cells"][pick] = {
            "metal": data.metals[i], "motif": data.motifs[i], "site": int(m),
            "contribution": v.tolist(),
            "symbols": list(c.symbols),
            "coords": np.asarray(c.coords, float).tolist(),
            "top_sites": S.site_contribution_table(model, c, n=6),
            "top_channels": S.site_top_channels(model, c, m, labels, n=6)}
        log(f"  {pick}: the six sites that move the prediction most: "
            + ", ".join(f"{lab} {val:+.3f}" for lab, val in out["cells"][pick]["top_sites"]))
    figure(out, log)
    return out


def figure(res: dict, log=print) -> None:
    """The contributions drawn on the sheet, and the weight by distance from the metal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = list(res.get("cells", {}).items())
    if not cells:
        return
    fig, axes = plt.subplots(1, len(cells) + 1, figsize=(5.2 * (len(cells) + 1), 4.8))
    for ax, (name, cell) in zip(axes, cells):
        xyz = np.asarray(cell["coords"], float)
        v = np.asarray(cell["contribution"], float)
        symbols = list(cell["symbols"])
        top = float(np.max(np.abs(v)))
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=v, cmap="coolwarm", vmin=-top, vmax=top,
                        s=95, edgecolors="k", linewidths=0.3)
        m = int(cell["site"])
        ax.scatter([xyz[m, 0]], [xyz[m, 1]], s=280, facecolors="none", edgecolors="k",
                   linewidths=1.6)
        d = np.linalg.norm(xyz - xyz[m], axis=1)
        for k in np.where((d > 0) & (d < 2.4))[0]:
            ax.annotate(symbols[k], (xyz[k, 0], xyz[k, 1]), textcoords="offset points",
                        xytext=(6, 5), fontsize=9)
        ax.annotate(symbols[m], (xyz[m, 0], xyz[m, 1]), textcoords="offset points",
                    xytext=(10, 9), fontsize=12, fontweight="bold")
        ax.set_title(f"{name}: what each site adds")
        ax.set_xlabel("x, angstrom")
        ax.set_ylabel("y, angstrom")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="contribution to dE(OH), eV")
    ax = axes[-1]
    edges = list(res["by_distance"]["edges"]) + [None]
    enrich = np.array(res["by_distance"]["enrichment"])
    ax.bar(range(len(enrich)), enrich, color="#4878a8")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
    ax.annotate("an even spread over the sites", (len(enrich) - 0.4, 1.05), ha="right",
                fontsize=9)
    ax.set_xticks(range(len(enrich)))
    ax.set_xticklabels([f"{edges[b]:.1f}\nto {edges[b + 1]:.1f}" if edges[b + 1] is not None
                        else f"beyond\n{edges[b]:.1f}" for b in range(len(enrich))], fontsize=9)
    ax.set_xlabel("distance from the metal, angstrom")
    ax.set_ylabel("weight per site, times an even spread")
    ax.set_title("Where the prediction comes from")
    fig.suptitle("A whole-cell model read site by site (plan item M5)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(RESULTS, "contributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"  figure written to {path}")


# =========================================================================== #
# The report
# =========================================================================== #
def report(log=print) -> str:
    lines = ["# Site-resolved models of a periodic cell (plan item M5)", ""]
    sac = _load("sac.json")
    if sac:
        lines += ["## The single-atom catalysts: one measured site per cell", "",
                  f"{sac['n']} catalysts; measured targets "
                  + ", ".join(f"{k} {v}" for k, v in sac["targets"].items()) + ".",
                  "Every method chose its settings in an inner cross-validation of the same kind "
                  "as the outer folds, over the cutoffs "
                  + ", ".join(f"{c:g}" for c in sac["cutoffs"]) + " A and the ridge values.", "",
                  "| Method | Folds | " + " | ".join(sorted(sac["targets"]))
                  + " | Mean of the four |",
                  "| --- | --- | " + " | ".join("---:" for _ in sac["targets"]) + " | ---: |"]
        names, designs = [], []
        for m in sac["models"]:
            if m["name"] not in names:
                names.append(m["name"])
            if m["design"] not in designs:
                designs.append(m["design"])
        by = {(m["name"], m["design"], m["target"]): m for m in sac["models"]}
        for name in names:
            for design in designs:
                row, got = [name, design], []
                for target in sorted(sac["targets"]):
                    m = by.get((name, design, target))
                    row.append(f"{m['q2']:+.3f}" if m else "")
                    if m:
                        got.append(m["q2"])
                row.append(f"{sum(got) / len(got):+.3f}" if got else "")
                lines.append("| " + " | ".join(row) + " |")
        lines += ["", "q2 out of fold.  The paired difference of the site profile against every "
                      "other method, resampling the catalysts:", "",
                  "| Against | Folds | Target | Difference in q2 | 95 % interval |",
                  "| --- | --- | --- | ---: | --- |"]
        for p in sac["pairs"]:
            lines.append(f"| {p['against']} | {p['design']} | {p['target']} | {p['delta']:+.3f} "
                         f"| {p['low']:+.3f} to {p['high']:+.3f} |")
        lines.append("")
    li = _load("lithium.json")
    if li:
        lines += ["## The lithium sites: many measured sites per cell", "",
                  f"{li['n_sites']} Li sites in {len(li['compounds'])} compounds "
                  f"({', '.join(li['compounds'])}), {li['folds']} folds holding out whole "
                  f"compounds; the site energies have a spread of {li['spread']:.3f} eV.", "",
                  "| Method | q2 | Root mean square error, eV | Mean absolute error, eV |",
                  "| --- | ---: | ---: | ---: |"]
        for m in li["models"]:
            lines.append(f"| {m['name']} | {m['q2']:+.3f} | {m['rmse']:.3f} | {m['mae']:.3f} |")
        lines += ["", "By compound, for " + li["best"] + ".  The offset is how far the whole "
                      "compound is predicted from where it sits; the last two columns leave that "
                      "offset out and ask only whether the sites inside the compound are ordered "
                      "correctly.", "",
                  "| Compound | Sites | Mean absolute error, eV | Spread, eV | Offset, eV "
                  "| q2 about its own mean | Spearman correlation |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, v in li["per_compound"].items():
            lines.append(f"| {name} | {v['n']} | {v['mae']:.3f} | {v['spread']:.3f} "
                         f"| {v['offset']:+.3f} | {v['q2_centred']:+.3f} | {v['spearman']:+.2f} |")
        lines.append("")
    co = _load("contributions.json")
    if co:
        s = co["summary"]
        lines += ["## A whole-cell model read site by site", "",
                  f"A model of the OH binding energy on the whole-cell autocorrelation of "
                  f"{co['n']} catalysts (q2 {co['q2']:+.3f} in five folds, "
                  f"{co['rmse']:.3f} eV).  Its prediction splits exactly over the sites of the "
                  f"cell.  The metal and its first coordination shell carry "
                  f"{100 * s['site_and_shell']:.1f} % of the total weight (the metal alone "
                  f"{100 * s['metal']:.1f} %), where an even spread over every site of the sheet "
                  f"would give {100 * s['even']:.1f} %: a factor of {s['ratio']:.1f}.", "",
                  "| Distance from the metal, angstrom | Share of the weight, per cent "
                  "| Share of the sites, per cent | Times an even spread |",
                  "| --- | ---: | ---: | ---: |"]
        edges = co["by_distance"]["edges"]
        for b, share in enumerate(co["by_distance"]["share"]):
            top = edges[b + 1] if b + 1 < len(edges) else None
            band = f"{edges[b]:.1f} to {top:.1f}" if top else f"beyond {edges[b]:.1f}"
            lines.append(f"| {band} | {100 * share:.1f} "
                         f"| {100 * co['by_distance']['share_of_sites'][b]:.1f} "
                         f"| {co['by_distance']['enrichment'][b]:.1f} |")
        lines.append("")
    text = "\n".join(lines)
    with io.open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    log(text)
    return text


def _load(name: str) -> Optional[dict]:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    return json.load(io.open(path, encoding="utf-8"))


def _save(name: str, data: dict) -> None:
    with io.open(os.path.join(RESULTS, name), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sac", action="store_true", help="the single-atom catalysts")
    ap.add_argument("--lithium", action="store_true", help="the Li site energies")
    ap.add_argument("--contributions", action="store_true",
                    help="a whole-cell model read site by site")
    ap.add_argument("--report", action="store_true", help="the table from the stored results")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    log_fh = io.open(os.path.join(RESULTS, "run.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        log_fh.write(str(s) + "\n")
        log_fh.flush()

    if args.report:
        report(log)
        return
    everything = not (args.sac or args.lithium or args.contributions)
    t0 = time.time()
    log(f"=== {time.strftime('%Y-%m-%d %H:%M')} site-resolved models (plan item M5)")
    if args.lithium or everything:
        log("The lithium sites of the three electrolytes")
        _save("lithium.json", lithium_run(args, log))
    if args.contributions or everything:
        log("A whole-cell model read site by site")
        _save("contributions.json", contributions_run(args, log))
    if args.sac or everything:
        log("The single-atom catalysts")
        _save("sac.json", sac_run(args, log))
    report(log)
    log(f"=== {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
