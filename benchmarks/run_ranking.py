#!/usr/bin/env python
"""
Is a ranking SVM worth having, or does kernel ridge rank just as well?

Two questions, and they have different answers.

**On clean data**, where the activities are one assay's numbers and mean what they say, a
regression model is already a ranking model: its predictions are a score, and the order of that
score is all a ranking model ever produces.  Fitting the order instead of the values discards
information -- how far apart two molecules are, not merely which is higher -- so the expectation
is that ranking does not win, and the first table checks it.

**On pooled data** the opposite holds.  Potencies collected from several papers carry an
offset per source that no descriptor explains, and a regression model spends its capacity
fitting those offsets.  A ranking model told which molecules share a source compares only within
one, and the offsets cancel exactly.  The second table makes that concrete the only way a
benchmark can: it takes the same sets, splits each into synthetic sources **by
scaffold**, shifts each source by an offset, and asks both models to recover the *true* order
they never saw. The sources have to be chemistry and not a matter of chance: an offset assigned at
random is unpredictable from structure, so a regression model averages it away and loses
nothing. An offset that travels with a scaffold is exactly what a model will learn as if it
were an effect of the scaffold, and that is the failure worth measuring.

The pooled columns are therefore scored **within a source**: the share of same-source pairs put
ordered correctly. A model that only ever compared molecules inside a source cannot be asked
where one source sits relative to another -- it was never shown -- and the question a chemist
has is in any case which compound of *this* series to make next.

Both tables are Spearman's rho on held-out molecules, over the eight Sutherland sets with ten
random 70/30 splits each, every model choosing its own hyper-parameters on its own criterion
inside the training part (RMSEcv for ridge, rho for ranking).

Usage
-----
    python benchmarks/run_ranking.py
    python benchmarks/run_ranking.py --sets ACE DHFR --repeats 5
    python benchmarks/run_ranking.py --report
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

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, krr_fit  # noqa: E402
from cmfqsar.chemotypes import chemotype_labels  # noqa: E402
from cmfqsar.ranking import concordant_pairs, fit_rank, rank_scores  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "ranking")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0)
CS = (0.1, 1.0, 10.0)
#: the solver works on the kernel between pairs, so this is the parameter that decides the cost
MAX_PAIRS = 1000
N_SOURCES = 3            # synthetic papers, for the pooled experiment
OFFSET = 1.5             # the spread of their offsets, in units of the activity's own sd
FOLDS = 5                # inner folds; the pairwise fit is the expensive part here


def kernels(mols, names, alphas=ALPHAS):
    out = []
    for alpha in alphas:
        fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="file")
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
        out.append((float(alpha), model.joint_kernel(model.prepare_training_kernels(table))))
    return out


def _ridge(Ks, y, tr, te, seed):
    """Kernel ridge, alpha and lambda chosen on RMSEcv inside the training part."""
    best = None
    for alpha, K in Ks:
        Ktr = K[np.ix_(tr, tr)]
        for lam in LAMS:
            p = cv_predict(Ktr, y[tr], "krr", {"lam": float(lam)}, folds=FOLDS, seed=seed)
            err = float(np.sqrt(np.mean((y[tr] - p) ** 2)))
            if best is None or err < best[0]:
                best = (err, K, float(lam))
    _err, K, lam = best
    a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
    return K[np.ix_(te, tr)] @ a + b


def _rank(Ks, y, tr, te, seed, groups=None, gap=0.0):
    """Ranking SVM, alpha and C chosen on rho inside the training part."""
    g_tr = None if groups is None else np.asarray(groups)[tr]
    best = None
    for alpha, K in Ks:
        Ktr = K[np.ix_(tr, tr)]
        for C in CS:
            p = cv_predict(Ktr, y[tr], "ranksvm",
                           {"C": float(C), "gap": gap, "max_pairs": MAX_PAIRS, "seed": seed},
                           folds=FOLDS, seed=seed, fit_bias=False, groups=g_tr)
            rho = rank_scores(y[tr], p, gap=gap, groups=g_tr)["spearman"]
            if np.isfinite(rho) and (best is None or rho > best[0]):
                best = (rho, K, float(C))
    if best is None:
        return np.full(len(te), np.nan)
    _rho, K, C = best
    fit = fit_rank(K[np.ix_(tr, tr)], y[tr], C=C, gap=gap, groups=g_tr, seed=seed,
                   max_pairs=MAX_PAIRS)
    return fit.score(K[np.ix_(te, tr)])


def run_set(name, repeats=10, force=False, data=DATA, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(os.path.join(data, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    y = y[keep]
    n = len(y)
    sd = float(np.std(y))
    print(f"=== {name}: {n} molecules", flush=True)
    Ks = kernels(mols, names)
    # the synthetic sources: scaffold families, collapsed to N_SOURCES by size, so that an
    # offset travels with the chemistry exactly as it does in a pooled literature set
    labels, _scaffolds = chemotype_labels(mols, min_size=3)
    src = np.minimum(np.asarray(labels, int), N_SOURCES - 1)
    print(f"    sources by scaffold: " + ", ".join(f"{int((src == k).sum())}" for k in
                                                   range(N_SOURCES)) + " molecules", flush=True)

    rows = []
    for rep in range(repeats):
        rng = np.random.RandomState(rep)
        order = rng.permutation(n)
        n_te = max(5, int(round(0.3 * n)))
        te, tr = np.sort(order[:n_te]), np.sort(order[n_te:])

        # 1. clean activities
        rows.append({"repeat": rep, "data": "clean", "model": "krr",
                     **rank_scores(y[te], _ridge(Ks, y, tr, te, rep))})
        rows.append({"repeat": rep, "data": "clean", "model": "ranksvm",
                     **rank_scores(y[te], _rank(Ks, y, tr, te, rep))})

        # 2. the same molecules reported by sources that each studied one scaffold family
        shift = rng.normal(scale=OFFSET * sd, size=N_SOURCES)
        y_pooled = y + shift[src]
        for model, score in (("krr", _ridge(Ks, y_pooled, tr, te, rep)),
                             ("ranksvm", _rank(Ks, y_pooled, tr, te, rep)),
                             ("ranksvm-groups", _rank(Ks, y_pooled, tr, te, rep, groups=src))):
            rows.append({"repeat": rep, "data": "pooled", "model": model,
                         **rank_scores(y[te], score),
                         "within": concordant_pairs(y[te], score, groups=src[te])})
        print(f"  repeat {rep} done", flush=True)

    out = {"set": name, "n": int(n), "activity_sd": sd, "rows": rows,
           "repeats": int(repeats), "n_sources": N_SOURCES, "offset_sd": OFFSET,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"  {name}: {out['seconds']:.0f} s")
    return out


def report(results=RESULTS):
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No ranking results yet; run `python benchmarks/run_ranking.py`."
    data = [json.load(open(f)) for f in files]
    d0 = data[0]

    def mean(set_rows, data_kind, model, key="spearman"):  # noqa: D401
        v = [r[key] for r in set_rows if r["data"] == data_kind and r["model"] == model
             and np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")

    lines = ["# Does fitting the order beat fitting the values?", "",
             f"{len(data)} Sutherland sets, {d0['repeats']} random 70/30 splits each; "
             "Spearman's rho of the held-out molecules. The pooled rows give the same molecules "
             f"to {d0['n_sources']} synthetic sources and shift each by an offset of "
             f"{d0['offset_sd']} standard deviations, then ask for the order of the *true* "
             "activity, which no model saw.", "",
             "| set | n | clean: KRR | clean: rank | pooled: KRR | pooled: rank | pooled: rank by source |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    cols = [("clean", "krr", "spearman"), ("clean", "ranksvm", "spearman"),
            ("pooled", "krr", "within"), ("pooled", "ranksvm", "within"),
            ("pooled", "ranksvm-groups", "within")]
    tot = {c: [] for c in cols}
    for d in data:
        cells = []
        for c in cols:
            v = mean(d["rows"], *c)
            tot[c].append(v)
            cells.append(f"{v:+.3f}" if np.isfinite(v) else "--")
        lines.append(f"| {d['set']} | {d['n']} | " + " | ".join(cells) + " |")
    lines.append("| **mean** |  | " + " | ".join(f"**{np.nanmean(tot[c]):+.3f}**" for c in cols)
                 + " |")
    lines += ["", "## The pooled sets, ordered across sources", "",
              "The same models and splits, scored by Spearman's rho over *all* held-out pairs. "
              "A model that only ever compared molecules inside a source cannot say "
              "where one source lies relative to another, so this column shows that "
              "limitation.", "",
              "| set | KRR | rank | rank by source |", "|---|---:|---:|---:|"]
    gcols = [("pooled", "krr", "spearman"), ("pooled", "ranksvm", "spearman"),
             ("pooled", "ranksvm-groups", "spearman")]
    gtot = {c: [] for c in gcols}
    for d in data:
        cells = []
        for c in gcols:
            v = mean(d["rows"], *c)
            gtot[c].append(v)
            cells.append(f"{v:+.3f}" if np.isfinite(v) else "--")
        lines.append(f"| {d['set']} | " + " | ".join(cells) + " |")
    lines.append("| **mean** | " + " | ".join(f"**{np.nanmean(gtot[c]):+.3f}**" for c in gcols)
                 + " |")
    lines += ["", "The clean columns are Spearman's rho over the held-out molecules; the "
              "pooled columns are the share of **same-source** held-out pairs ordered "
              "correctly, so 0.5 is no better than chance. The two halves are not comparable with each "
              "other."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    for name in a.sets:
        run_set(name, repeats=a.repeats, force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
