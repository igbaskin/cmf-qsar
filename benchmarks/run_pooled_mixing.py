#!/usr/bin/env python
"""
Do the other series know a series' field weights better than the series itself?

The joint kernel is a sum over fields, `K = sum_f h_f c_f K_f`, and `benchmarks/run_mixing.py`
found that fitting the weights on the series at hand -- by alignment or by the evidence --
predicts no better than equal weights. A series of thirty molecules cannot say much about five
weights. This benchmark asks whether a *collection* of series can: `cmfqsar.mixing.pool_priors`
learns, from the posteriors of many series, the log-shares a typical series needs and how much
series differ from each other (a two-level, or hierarchical, prior), and a series then gets its
weights from the evidence under that prior instead of under the fixed one centred at equal
shares.

The measurement is leave-one-series-out at the level of the collection. For every set of the
Mittal collection the prior is learned from the other 37 sets, never from the set itself. Then,
on random 70/30 splits of the set, five choices of weights are made on the training part and
scored on the molecules held out:

* ``uniform``: equal weights;
* ``align``: kernel alignment on this series;
* ``evidence``: the evidence optimum on this series under the fixed prior;
* ``population``: the typical weights of the other series, nothing fitted on this one;
* ``pooled``: the evidence optimum on this series under the prior learned from the others.

``population`` and ``pooled`` are the two ends of what pooling can do: the first ignores the
series, the second lets the series move away from the others as far as its data justify. The
attenuation is chosen on the training part by the evidence with equal weights, the same for
every choice, so only the weights differ.

Two stages, both cached under ``benchmarks/results/pooled_mixing/``: the posterior of every
set on all its molecules (``posteriors/<set>.npz``), then the evaluation (``<set>.json``).

Usage
-----
    python benchmarks/run_pooled_mixing.py                  # every Mittal set
    python benchmarks/run_pooled_mixing.py --sets ACE BZR   # posteriors of all, evaluation of two
    python benchmarks/run_pooled_mixing.py --jobs 8
    python benchmarks/run_pooled_mixing.py --report
"""
from __future__ import annotations

import argparse
import glob
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

import kernel_cache as KC  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.mixing import PRIOR_SD, evidence_mixing, pool_priors, sample_mixing  # noqa: E402
from cmfqsar.model import CMFModel, align_mixing, krr_fit, r2_score  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "pooled_mixing")
DATA = os.path.join(ROOT, "examples", "mittal")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
DRAWS = 300
BURN = 50
REPEATS = 10
#: below this many molecules a set counts as small, the split `run_pharmfields.py` uses
SIZE_SPLIT = 40
WEIGHTS = ("uniform", "align", "evidence", "population", "pooled")


def all_sets(data=DATA):
    return [os.path.splitext(os.path.basename(p))[0]
            for p in sorted(glob.glob(os.path.join(data, "*.sdf")))]


def load(name, data=DATA):
    ds = load_dataset(os.path.join(data, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    return mols, names, y[keep]


def kernels_for(mols, names):
    """alpha -> (per-field training kernels (F, n, n), their scale), through the kernel cache."""
    out = {}
    for a in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=float(a), charge_model="gasteiger")
        m = CMFModel(fieldset=fs, method="krr", normalize=True)
        table = AtomTable.from_molecules([build_fields(mm, fs, nn)
                                          for mm, nn in zip(mols, names)])
        K = np.asarray(KC.training_kernels(m, table), float)
        out[float(a)] = (K, np.asarray(m.scale, float))
    return out


def labels_of():
    return [s.label for s in FieldSet.create(physchem=FIELDS).specs]


def _combine(Kf, h, scale):
    return np.tensordot(np.asarray(h, float) * np.asarray(scale, float), Kf, axes=(0, 0))


def _best_lam(K, y):
    best = None
    for lam in LAMS:
        e = log_evidence(K, y, float(lam), fit_bias=True)
        if np.isfinite(e) and (best is None or e > best[0]):
            best = (e, float(lam))
    return best[1] if best else 1.0


def _best_alpha(kernels, y, idx):
    """The attenuation the evidence prefers with equal weights, on the molecules `idx`."""
    best = None
    for a, (Kf, scale) in kernels.items():
        K = _combine(Kf[:, idx][:, :, idx], np.ones(Kf.shape[0]), scale)
        e = log_evidence(K, y[idx], _best_lam(K, y[idx]), fit_bias=True)
        if np.isfinite(e) and (best is None or e > best[0]):
            best = (e, a)
    return best[1] if best else float(ALPHAS[2])


def _score(Kf, h, scale, y, tr, te):
    K = _combine(Kf, h, scale)
    lam = _best_lam(K[np.ix_(tr, tr)], y[tr])
    a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
    return float(r2_score(y[te], K[np.ix_(te, tr)] @ a + b))


# --- stage 1: the posterior of every set on all its molecules ------------------------------

def posterior_path(name, results=RESULTS):
    return os.path.join(results, "posteriors", f"{name}.npz")


def run_posterior(name, force=False, data=DATA, results=RESULTS):
    path = posterior_path(name, results)
    if os.path.exists(path) and not force:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    mols, names, y = load(name, data)
    kernels = kernels_for(mols, names)
    idx = np.arange(len(y))
    alpha = _best_alpha(kernels, y, idx)
    Kf, scale = kernels[alpha]
    post = sample_mixing(Kf, y, labels_of(), scale, draws=DRAWS, burn=BURN, seed=0)
    np.savez(path, samples=post.samples, lam=post.lam, alpha=alpha, n=len(y),
             map_shares=post.map_shares, seconds=time.time() - t0)
    print(f"posterior {name}: {len(y)} molecules, alpha {alpha:g}, mean shares "
          + " ".join(f"{v:.2f}" for v in post.mean) + f", {time.time() - t0:.0f} s", flush=True)
    return path


def load_posteriors(names, results=RESULTS):
    out = {}
    for name in names:
        with np.load(posterior_path(name, results)) as z:
            out[name] = {"samples": np.array(z["samples"]), "alpha": float(z["alpha"]),
                         "n": int(z["n"])}
    return out


# --- stage 2: every set against the prior its neighbours give it ---------------------------

def run_set(name, posts, repeats=REPEATS, force=False, data=DATA, results=RESULTS):
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    labels = labels_of()
    others = [v["samples"] for k, v in posts.items() if k != name]
    prior = pool_priors(others, labels)
    prior_is = pool_priors(others, labels, method="importance")
    mols, names, y = load(name, data)
    n = len(y)
    kernels = kernels_for(mols, names)
    print(f"=== {name}: {n} molecules; prior from {len(others)} sets: tau {prior.sd:.2f}, "
          "typical shares " + " ".join(f"{v:.2f}" for v in prior.shares), flush=True)

    rows = []
    for rep in range(repeats):
        rng = np.random.RandomState(rep)
        order = rng.permutation(n)
        n_te = max(5, int(round(0.3 * n)))
        te, tr = np.sort(order[:n_te]), np.sort(order[n_te:])
        alpha = _best_alpha(kernels, y, tr)
        Kf, scale = kernels[alpha]
        Ktr = Kf[:, tr][:, :, tr]
        hs = {"uniform": np.ones(len(labels)),
              "align": align_mixing(Ktr, y[tr], scale),
              "evidence": evidence_mixing(Ktr, y[tr], scale),
              "population": prior.h(),
              "pooled": evidence_mixing(Ktr, y[tr], scale, prior_mean=prior.mean,
                                        prior_sd=prior.sd)}
        r2 = {k: _score(Kf, h, scale, y, tr, te) for k, h in hs.items()}
        rows.append({"repeat": rep, "alpha": alpha, "r2": r2,
                     "h": {k: [float(v) for v in h] for k, h in hs.items()}})
        print(f"  repeat {rep}: alpha {alpha:g}; R2pred " +
              "  ".join(f"{k} {r2[k]:+.3f}" for k in WEIGHTS), flush=True)

    def _prior(p):
        return {"method": p.method, "shares": {l: float(v) for l, v in zip(labels, p.shares)},
                "mean": [float(v) for v in p.mean], "sd": float(p.sd), "note": p.note,
                "n_series": int(p.n_series), "log_marginal": float(p.log_marginal)}

    out = {"set": name, "n": int(n), "repeats": int(repeats), "fields": labels,
           "draws": DRAWS, "prior": _prior(prior), "prior_importance": _prior(prior_is),
           "rows": rows, "seconds": round(time.time() - t0, 1)}
    os.makedirs(results, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    return out


# --- the report -------------------------------------------------------------------------------

def _paired(diff, rng, n_boot=5000):
    bt = [float(np.mean(rng.choice(diff, len(diff), replace=True))) for _ in range(n_boot)]
    return float(np.mean(diff)), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))


def report(results=RESULTS):
    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No pooled-mixing results yet; run `python benchmarks/run_pooled_mixing.py`."
    data = sorted([json.load(open(f)) for f in files], key=lambda r: -r["n"])
    labels = data[0]["fields"]
    per = {w: np.array([float(np.mean([r["r2"][w] for r in d["rows"]])) for d in data])
           for w in WEIGHTS}
    ns = np.array([d["n"] for d in data])
    small = ns < SIZE_SPLIT
    rng = np.random.default_rng(0)

    lines = ["# The field weights of a series, learned from the other series", "",
             f"{len(data)} Mittal sets, {data[0]['repeats']} random 70/30 splits each. For every "
             f"set a two-level prior over the field weights is learned from the other "
             f"{len(data) - 1} sets (`cmfqsar.mixing.pool_priors`, Gaussian form, "
             f"{data[0]['draws']} posterior draws per set), never from the set itself. The "
             "attenuation is chosen on the training part by the evidence with equal weights; "
             "the weights are then set five ways on the training part and scored on the "
             "molecules held out (R²pred, mean over the splits).", "",
             "| set | n | " + " | ".join(WEIGHTS) + " | prior τ |",
             "|---|---:|" + "---:|" * (len(WEIGHTS) + 1)]
    for d, *_ in zip(data):
        cells = [f"{float(np.mean([r['r2'][w] for r in d['rows']])):+.3f}" for w in WEIGHTS]
        lines.append(f"| {d['set']} | {d['n']} | " + " | ".join(cells)
                     + f" | {d['prior']['sd']:.2f} |")
    for title, mask in (("mean, all sets", np.ones(len(data), bool)),
                        (f"mean, n ≥ {SIZE_SPLIT}", ~small), (f"mean, n < {SIZE_SPLIT}", small)):
        if mask.any():
            lines.append(f"| **{title}** | {int(mask.sum())} | "
                         + " | ".join(f"**{float(per[w][mask].mean()):+.3f}**" for w in WEIGHTS)
                         + " |  |")

    lines += ["", "## Paired differences", "",
              "Mean difference in R²pred over the sets, with a 95 % bootstrap interval over sets:",
              "", "| comparison | all sets | n ≥ 40 | n < 40 |", "|---|---:|---:|---:|"]
    pairs = (("pooled − uniform", "pooled", "uniform"),
             ("pooled − evidence", "pooled", "evidence"),
             ("population − uniform", "population", "uniform"),
             ("pooled − population", "pooled", "population"),
             ("evidence − uniform", "evidence", "uniform"))
    summary = {}
    for title, a, b in pairs:
        cells = []
        for mask in (np.ones(len(data), bool), ~small, small):
            if mask.sum() < 2:
                cells.append("")
                continue
            m, lo, hi = _paired(per[a][mask] - per[b][mask], rng)
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            if mask.all():
                summary[title] = (m, lo, hi)
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    wins = int(np.sum(per["pooled"] > per["evidence"]))
    wins_u = int(np.sum(per["pooled"] > per["uniform"]))
    lines += ["", f"The pooled weights beat the series' own evidence weights on {wins} of "
              f"{len(data)} sets and equal weights on {wins_u} of {len(data)}."]

    taus = np.array([d["prior"]["sd"] for d in data])
    taus_is = np.array([d["prior_importance"]["sd"] for d in data])
    lines += ["", "## What the collection says a series needs", "",
              f"The prior is learned {len(data)} times, each time without one set. The spread of "
              f"the log-shares between series is τ = {np.median(taus):.2f} (range "
              f"{taus.min():.2f} to {taus.max():.2f}); the fixed prior has {PRIOR_SD}. The "
              f"importance-reweighting estimate, which makes no Gaussian assumption, gives "
              f"τ = {np.median(taus_is):.2f} (range {taus_is.min():.2f} to {taus_is.max():.2f}).",
              "", "| field | typical share (Gaussian) | range over the leave-one-out priors | "
              "typical share (importance) |", "|---|---:|---:|---:|"]
    for label in labels:
        sh = np.array([d["prior"]["shares"][label] for d in data])
        sh2 = np.array([d["prior_importance"]["shares"][label] for d in data])
        lines.append(f"| {label} | {sh.mean():.3f} | [{sh.min():.3f}, {sh.max():.3f}] | "
                     f"{sh2.mean():.3f} |")
    lines.append(f"\nEqual shares would be {1.0 / len(labels):.3f}.")
    notes = sorted({d["prior"]["note"] for d in data if d["prior"]["note"]})
    if notes:
        lines += ["", "Notes from the pooling: " + "; ".join(notes) + "."]

    m, lo, hi = summary["pooled − uniform"]
    me, loe, hie = summary["pooled − evidence"]
    mp, lop, hip = summary["population − uniform"]
    lines += ["", "## Reading", ""]
    if lo > 0:
        lines.append(f"Pooling gains: against equal weights the pooled weights are {m:+.3f} in "
                     f"R²pred and the interval [{lo:+.3f}, {hi:+.3f}] excludes zero. "
                     "Other series do know something about a series' weights that the series "
                     "itself cannot see.")
    elif hi < 0:
        lines.append(f"Pooling costs: against equal weights the pooled weights are {m:+.3f} in "
                     f"R²pred and the interval [{lo:+.3f}, {hi:+.3f}] excludes zero. What a "
                     "typical series needs is not what this series needs.")
    else:
        lines.append(f"Pooling neither gains nor costs on average: against equal weights the "
                     f"pooled weights are {m:+.3f} in R²pred with an interval of "
                     f"[{lo:+.3f}, {hi:+.3f}] that contains zero.")
    if loe > 0:
        lines.append(f"Against the series' own evidence weights the pooled weights gain {me:+.3f} "
                     f"[{loe:+.3f}, {hie:+.3f}]: the prior from the other series removes the "
                     "damage that fitting five weights on one series does.")
    else:
        lines.append(f"Against the series' own evidence weights the pooled weights are {me:+.3f} "
                     f"[{loe:+.3f}, {hie:+.3f}].")
    lines.append(f"The population weights alone, with nothing fitted on the series, are "
                 f"{mp:+.3f} [{lop:+.3f}, {hip:+.3f}] against equal weights.")
    if small.sum() >= 2:
        ms, los, his = _paired(per["pooled"][small] - per["uniform"][small], rng)
        lines.append(f"On the {int(small.sum())} sets below {SIZE_SPLIT} molecules, where a prior "
                     f"should matter most, pooled against uniform is {ms:+.3f} "
                     f"[{los:+.3f}, {his:+.3f}].")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                    # noqa: BLE001 - a stream that cannot be reconfigured
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=None, help="sets to evaluate (default: all)")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--jobs", type=int, default=1, help="sets in parallel (processes)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    names = all_sets()
    todo = a.sets if a.sets else names
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        with ProcessPoolExecutor(a.jobs) as ex:
            list(ex.map(run_posterior, names, [a.force] * len(names)))
        posts = load_posteriors(names)
        with ProcessPoolExecutor(a.jobs) as ex:
            list(ex.map(run_set, todo, [posts] * len(todo), [a.repeats] * len(todo),
                        [a.force] * len(todo)))
    else:
        for name in names:
            run_posterior(name, force=a.force)
        posts = load_posteriors(names)
        for name in todo:
            run_set(name, posts, repeats=a.repeats, force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
