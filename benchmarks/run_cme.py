#!/usr/bin/env python
"""
The expected field of a molecule with a given activity (conditional mean embedding), measured.

:mod:`cmfqsar.cme` estimates the field expected at an activity as a weighted sum of training
fields, with the width and ridge chosen by the out-of-fold loss in the field space. It draws a
map in the same basis as the co-field. This script asks what that map is good for, on the eight
Sutherland sets with the joint kernel of five fields at alpha = 0.3.

Three maps are compared throughout, all on the same grid (0.8 A, margin 2 A):

* **co-field**: the regression map of kernel ridge regression, ridge chosen by the evidence;
* **pattern**: the covariance of the field with the activity, the slope of the straight-line
  conditional mean embedding (a forward model, or pattern, in the terms of Haufe et al. 2014);
* **contrast**: the expected field at the 90th percentile of the activities minus the one at
  the 10th, from the Gaussian conditional mean embedding.

Stage ``real``
    The measured activities: the out-of-fold share of the field variance that the activity
    explains, with a straight line and with the Gaussian kernel on the activity, and how alike the
    three maps of the whole set are (correlation over the grid points the molecules occupy).

Stage ``halves``
    Each set is cut into two random halves 20 times, every map is computed on each half, and the
    two halves' maps are correlated. A map that comes back from the other half of the data is a
    map of the series and not of its noise.

Stage ``planted``
    Synthetic activities with a known answer, on the real molecules. The steric field at one grid
    point r0 (five points per set, among the 5 % of points where the steric field varies most) is
    standardised to s. *Linear*: y = s + noise, half of the variance from s; where does each map put
    its peak? *Curved*: y = g(s) + noise, 80 % of the variance from g, for three shapes g: an
    optimum -(s - 1)^2, a plateau min(s, 0) and a threshold max(s, 0). What field at r0 does each
    method expect at the 90th and at the 10th percentile of the activity? The answer to compare
    with is the conditional mean of s at that activity, computed by Monte Carlo over 4,000 draws
    of the noise on the same molecules.

Stage ``rank``
    Five-fold cross-validation, three seeds: do molecules close to the expected field of an
    active molecule tend to be active? Scores: the distance to the expected field at the 90th
    percentile of the training activities (Gaussian and straight line), the distance to the mean
    field of the top tenth of the training set, and the prediction of kernel ridge regression.

Usage
-----
    python benchmarks/run_cme.py                   # every stage
    python benchmarks/run_cme.py --stage planted
    python benchmarks/run_cme.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import RDLogger  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from cmfqsar.cme import Standardiser, cme_weights, distance_to, fit_cme, pattern  # noqa: E402
from cmfqsar.cofields import make_grid, per_molecule_fields  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, krr_fit  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "cme")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 0.3
LAMS_KRR = (1e-3, 1e-2, 0.1, 1.0, 10.0)
SPACING = 0.8
MARGIN = 2.0
OCCUPIED = 0.01          # a grid point counts when the mean |field| there exceeds this share of its max
HI, LO = 90, 10          # the percentiles of the activity the contrast map compares
HALVES = 20
POINTS = 5               # planted points per set
REPLICATES = 10          # noise draws per planted point
MC_DRAWS = 4000
STAGES = ("real", "halves", "planted", "rank")
SHAPES = {"optimum": lambda s: -(s - 1.0) ** 2,
          "plateau": lambda s: np.minimum(s, 0.0),
          "threshold": lambda s: np.maximum(s, 0.0)}


# --------------------------------------------------------------------------- the data

def load(name):
    """Molecules, activities, per-field kernels, the joint kernel and the grid fields of a set."""
    ds = load_dataset(os.path.join(DATA, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    y = y[keep]
    fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA, charge_model="file")
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = model.joint_kernel(model.prepare_training_kernels(table))
    grid = make_grid(table.coords, margin=MARGIN, spacing=SPACING)
    pts = grid.points()
    X = {}
    for f, spec in enumerate(fs.specs):
        Xf = per_molecule_fields(model, f, pts, basis=model.training_field_basis(f))
        occ = np.abs(Xf).mean(axis=0)
        X[spec.name] = Xf[:, occ > OCCUPIED * occ.max()]
    steric = per_molecule_fields(model, FIELDS.index("steric"), pts,
                                 basis=model.training_field_basis(FIELDS.index("steric")))
    occ = np.abs(steric).mean(axis=0)
    keep_pts = occ > OCCUPIED * occ.max()
    return {"y": y, "K": K, "X": X, "steric": steric[:, keep_pts], "steric_pts": pts[keep_pts],
            "names": names}


def krr_lam(K, y):
    return max(LAMS_KRR, key=lambda lam: log_evidence(K, y, lam, fit_bias=True))


def maps(K, y, X, hi=None, lo=None, cme=None):
    """The three maps of one field matrix X (n, P): co-field, pattern and contrast."""
    lam = krr_lam(K, y)
    a, _b = krr_fit(K, y, lam, True)
    cme = cme or fit_cme(K, y, "gauss", seeds=(0,))
    hi = np.percentile(y, HI) if hi is None else hi
    lo = np.percentile(y, LO) if lo is None else lo
    B = cme.weights([hi, lo])
    return {"co-field": a @ X, "pattern": pattern(y, X), "contrast": (B[:, 0] - B[:, 1]) @ X}


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


KINDS = ("co-field", "pattern", "contrast")


# --------------------------------------------------------------------------- stages

def stage_real(name, d):
    y, K = d["y"], d["K"]
    out = {"n": len(y)}
    for ky in ("linear", "gauss"):
        f = fit_cme(K, y, ky)
        out[ky] = {"explained": f.explained, "gamma": f.gamma, "lam": f.lam, "edge": f.at_edge}
    cme = fit_cme(K, y, "gauss", seeds=(0,))
    sims = {"co-field vs pattern": [], "pattern vs contrast": [], "co-field vs contrast": []}
    for fname, X in d["X"].items():
        m = maps(K, y, X, cme=cme)
        sims["co-field vs pattern"].append(corr(m["co-field"], m["pattern"]))
        sims["pattern vs contrast"].append(corr(m["pattern"], m["contrast"]))
        sims["co-field vs contrast"].append(corr(m["co-field"], m["contrast"]))
    out["similarity"] = {k: float(np.mean(v)) for k, v in sims.items()}
    out["krr_lam"] = krr_lam(K, y)
    print(f"  real: explained {100 * out['linear']['explained']:.1f} % (line), "
          f"{100 * out['gauss']['explained']:.1f} % (Gaussian, gamma {out['gauss']['gamma']:g}); "
          + ", ".join(f"{k} {v:+.2f}" for k, v in out["similarity"].items()), flush=True)
    return out


def stage_halves(name, d):
    y, K = d["y"], d["K"]
    n = len(y)
    hi, lo = np.percentile(y, HI), np.percentile(y, LO)
    rng = np.random.RandomState(0)
    rep = {k: [] for k in KINDS}
    for t in range(HALVES):
        perm = rng.permutation(n)
        A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
        cA = fit_cme(K[np.ix_(A, A)], y[A], "gauss", seeds=(0,))
        cB = fit_cme(K[np.ix_(B, B)], y[B], "gauss", seeds=(0,))
        per = {k: [] for k in KINDS}
        for X in d["X"].values():
            mA = maps(K[np.ix_(A, A)], y[A], X[A], hi, lo, cA)
            mB = maps(K[np.ix_(B, B)], y[B], X[B], hi, lo, cB)
            for k in KINDS:
                per[k].append(corr(mA[k], mB[k]))
        for k in KINDS:
            rep[k].append(float(np.mean(per[k])))
    out = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in rep.items()}
    print("  halves: " + ", ".join(f"{k} {v['mean']:+.3f}" for k, v in out.items()), flush=True)
    return out


def planted_points(d, rng):
    S, P = d["steric"], d["steric_pts"]
    var = S.var(axis=0)
    cand = np.where(var >= np.percentile(var, 95))[0]
    rng.shuffle(cand)
    chosen = []
    for c in cand:
        if all(np.linalg.norm(P[c] - P[o]) >= 3.0 for o in chosen):
            chosen.append(int(c))
        if len(chosen) == POINTS:
            break
    return chosen


def stage_planted(name, d):
    K, S, P = d["K"], d["steric"], d["steric_pts"]
    n = len(d["y"])
    rng = np.random.RandomState(1)
    lin = {k: {"distance": [], "at_r0": []} for k in KINDS}
    curved = {shape: {end: {k: [] for k in ("gauss", "linear", "tenth")}
                      for end in ("top", "bottom")} for shape in SHAPES}
    noise_c = np.sqrt(0.25)                         # 80 % of the variance from the signal
    for c in planted_points(d, rng):
        s = Standardiser.fit(S[:, c])(S[:, c])
        for shape, g in SHAPES.items():
            sig = g(s)
            sig = (sig - sig.mean()) / sig.std()
            # the Monte Carlo answer: E[s | y near y*] over many draws of the noise
            draws = sig[None, :] + noise_c * rng.randn(MC_DRAWS, n)
            ends = {"top": float(np.percentile(draws, HI)),
                    "bottom": float(np.percentile(draws, LO))}
            sb = np.broadcast_to(s, draws.shape)
            truth = {e: float(sb[np.abs(draws - v) < 0.05].mean()) for e, v in ends.items()}
            for t in range(REPLICATES):
                y = sig + noise_c * rng.randn(n)
                for ky in ("gauss", "linear"):
                    w = fit_cme(K, y, ky, seeds=(0,)).weights([ends["top"], ends["bottom"]])
                    curved[shape]["top"][ky].append(float(w[:, 0] @ s) - truth["top"])
                    curved[shape]["bottom"][ky].append(float(w[:, 1] @ s) - truth["bottom"])
                curved[shape]["top"]["tenth"].append(
                    float(s[y >= np.percentile(y, HI)].mean()) - truth["top"])
                curved[shape]["bottom"]["tenth"].append(
                    float(s[y <= np.percentile(y, LO)].mean()) - truth["bottom"])
        for t in range(REPLICATES):
            # linear: half of the variance from s
            y = s + rng.randn(n)
            m = maps(K, y, S, cme=fit_cme(K, y, "gauss", seeds=(0,)))
            for k in KINDS:
                v = np.abs(m[k])
                lin[k]["distance"].append(float(np.linalg.norm(P[int(np.argmax(v))] - P[c])))
                lin[k]["at_r0"].append(float(v[c] / v.max()))
    out = {"linear": {k: {"median distance": float(np.median(v["distance"])),
                          "within 1.5 A": float(np.mean(np.asarray(v["distance"]) <= 1.5)),
                          "at r0": float(np.mean(v["at_r0"]))} for k, v in lin.items()},
           "curved": {shape: {end: {k: {"bias": float(np.mean(v)),
                                        "rmse": float(np.sqrt(np.mean(np.square(v))))}
                                    for k, v in e.items()} for end, e in ends_.items()}
                      for shape, ends_ in curved.items()}}
    print("  planted line: " + ", ".join(
        f"{k} {v['median distance']:.1f} A ({100 * v['within 1.5 A']:.0f} %)"
        for k, v in out["linear"].items()), flush=True)
    for shape, e in out["curved"].items():
        print(f"  planted {shape}: " + "; ".join(
            f"{end} " + ", ".join(f"{k} {v['rmse']:.2f}" for k, v in r.items())
            for end, r in e.items()), flush=True)
    return out


def stage_rank(name, d):
    y, K = d["y"], d["K"]
    n = len(y)
    kself = np.diag(K)
    res = {k: {"spearman": [], "top recall": []} for k in
           ("regression", "expected field, Gaussian", "expected field, line", "mean of top tenth")}
    for seed in (0, 1, 2):
        order = np.random.RandomState(seed).permutation(n)
        score = {k: np.zeros(n) for k in res}
        for te in [order[k::5] for k in range(5)]:
            tr = np.setdiff1d(np.arange(n), te)
            Ktr, Kte = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
            a, b = krr_fit(Ktr, y[tr], krr_lam(Ktr, y[tr]), True)
            score["regression"][te] = Kte @ a + b
            q = np.percentile(y[tr], HI)
            for ky, key in (("gauss", "expected field, Gaussian"), ("linear", "expected field, line")):
                w = fit_cme(Ktr, y[tr], ky, seeds=(0,)).weights([q])[:, 0]
                score[key][te] = -distance_to(w, Ktr, Kte, kself[te])
            top = (y[tr] >= q).astype(float)
            score["mean of top tenth"][te] = -distance_to(top / top.sum(), Ktr, Kte, kself[te])
        best = y >= np.percentile(y, HI)
        for k, s in score.items():
            res[k]["spearman"].append(float(spearmanr(s, y).correlation))
            picked = s >= np.percentile(s, HI)
            res[k]["top recall"].append(float((best & picked).sum() / best.sum()))
    out = {k: {m: float(np.mean(v)) for m, v in r.items()} for k, r in res.items()}
    print("  rank: " + ", ".join(f"{k} {v['spearman']:+.2f}/{100 * v['top recall']:.0f} %"
                                 for k, v in out.items()), flush=True)
    return out


RUN = {"real": stage_real, "halves": stage_halves, "planted": stage_planted, "rank": stage_rank}


def run(stages, sets=SETS, force=False):
    os.makedirs(RESULTS, exist_ok=True)
    for name in sets:
        path = os.path.join(RESULTS, f"{name}.json")
        rec = json.load(open(path)) if os.path.exists(path) else {"set": name}
        todo = [s for s in stages if force or s not in rec]
        if not todo:
            print(f"{name}: done")
            continue
        t0 = time.time()
        d = load(name)
        print(f"=== {name}: {len(d['y'])} molecules, data in {time.time() - t0:.0f} s", flush=True)
        for s in todo:
            t = time.time()
            rec[s] = RUN[s](name, d)
            rec.setdefault("seconds", {})[s] = round(time.time() - t, 1)
            json.dump(rec, open(path, "w"), indent=1)


# --------------------------------------------------------------------------- report

def _load(results=RESULTS):
    out = []
    for name in SETS:
        p = os.path.join(results, f"{name}.json")
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out


def report(results=RESULTS):
    data = _load(results)
    if not data:
        return "no results"
    L = ["# The expected field at an activity (conditional mean embedding)", "",
         f"Eight Sutherland sets, the joint kernel of five fields at alpha = {ALPHA}, maps on a "
         f"{SPACING} A grid. Co-field: the map of kernel ridge regression. Pattern: the covariance "
         f"of the field with the activity. Contrast: the expected field at the {HI}th percentile "
         f"of the activity minus the one at the {LO}th (Gaussian kernel on the activity).", ""]
    have = [d for d in data if "real" in d]
    if have:
        L += ["## How much of the field variance the activity explains, and how alike the maps are",
              "",
              "| set | n | explained, straight line | explained, Gaussian kernel | gamma | "
              "co-field vs pattern | pattern vs contrast | co-field vs contrast |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for d in have:
            r = d["real"]
            s = r["similarity"]
            L.append(f"| {d['set']} | {r['n']} | {100 * r['linear']['explained']:.1f} % | "
                     f"{100 * r['gauss']['explained']:.1f} % | {r['gauss']['gamma']:g}"
                     f"{' (edge)' if r['gauss']['edge'] else ''} | "
                     f"{s['co-field vs pattern']:+.2f} | {s['pattern vs contrast']:+.2f} | "
                     f"{s['co-field vs contrast']:+.2f} |")
        L += ["", "The correlations are over the grid points the molecules occupy, averaged over "
              "the five fields.", ""]
    have = [d for d in data if "halves" in d]
    if have:
        L += ["## Does a map come back from the other half of the data?", "",
              f"Correlation between the maps of two random halves, mean over {HALVES} cuts "
              f"(standard deviation over the cuts).", "",
              "| set | co-field | pattern | contrast |", "|---|---:|---:|---:|"]
        for d in have:
            h = d["halves"]
            L.append(f"| {d['set']} | " + " | ".join(
                f"{h[k]['mean']:+.3f} ({h[k]['sd']:.3f})" for k in KINDS) + " |")
        L.append("| **mean** | " + " | ".join(
            f"**{np.mean([d['halves'][k]['mean'] for d in have]):+.3f}**" for k in KINDS) + " |")
        L.append("")
    have = [d for d in data if "planted" in d]
    if have:
        L += ["## A planted signal at one point of the steric field", "",
              f"{POINTS} points per set, {REPLICATES} noise draws per point.", "",
              "Linear signal, y = s + noise: where is the peak of the steric map?", "",
              "| map | median distance of the peak from the planted point | peak within 1.5 A | "
              "map at the planted point / its peak |", "|---|---:|---:|---:|"]
        for k in KINDS:
            v = [d["planted"]["linear"][k] for d in have]
            L.append(f"| {k} | {np.mean([x['median distance'] for x in v]):.1f} A | "
                     f"{100 * np.mean([x['within 1.5 A'] for x in v]):.0f} % | "
                     f"{np.mean([x['at r0'] for x in v]):.2f} |")
        L += ["", "A curved signal, y = g(s) + noise: the field at the planted point that each "
              f"method expects at the {HI}th (top) and the {LO}th (bottom) percentile of the "
              "activity, minus the Monte Carlo answer, in standard deviations of s. Root mean "
              "square error (mean error).", "",
              "| shape g(s) | end | expected field, Gaussian kernel | expected field, straight "
              "line | mean of the tenth of the molecules at that end |",
              "|---|---|---:|---:|---:|"]
        label = {"optimum": "optimum, -(s - 1)^2", "plateau": "plateau, min(s, 0)",
                 "threshold": "threshold, max(s, 0)"}
        for shape in SHAPES:
            for end in ("top", "bottom"):
                cells = []
                for k in ("gauss", "linear", "tenth"):
                    v = [d["planted"]["curved"][shape][end][k] for d in have]
                    cells.append(f"{np.sqrt(np.mean([x['rmse'] ** 2 for x in v])):.2f} "
                                 f"({np.mean([x['bias'] for x in v]):+.2f})")
                L.append(f"| {label[shape]} | {end} | " + " | ".join(cells) + " |")
        L.append("")
    have = [d for d in data if "rank" in d]
    if have:
        keys = list(have[0]["rank"].keys())
        L += ["## Do molecules close to the expected field of an active tend to be active?", "",
              "Five-fold cross-validation, three seeds. Spearman correlation of the score with "
              f"the activity / the share of the top tenth found in the top tenth of the scores.", "",
              "| set | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
        for d in have:
            L.append(f"| {d['set']} | " + " | ".join(
                f"{d['rank'][k]['spearman']:+.2f} / {100 * d['rank'][k]['top recall']:.0f} %"
                for k in keys) + " |")
        L.append("| **mean** | " + " | ".join(
            f"**{np.mean([d['rank'][k]['spearman'] for d in have]):+.2f} / "
            f"{100 * np.mean([d['rank'][k]['top recall'] for d in have]):.0f} %**"
            for k in keys) + " |")
        L.append("")
    text = "\n".join(L)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=STAGES, nargs="*", default=list(STAGES))
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        run(a.stage, a.sets, a.force)
    print(report())


if __name__ == "__main__":
    main()
