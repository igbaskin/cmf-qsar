#!/usr/bin/env python
"""
Does a posterior over the field weights say anything a point estimate does not?

The joint kernel is a sum over fields, `K = sum_f h_f c_f K_f`, and `cmfqsar/mixing.py` puts a
prior on the weights and integrates, which gives each field's **share** of the kernel with an
interval on it and a Bayes factor against *this field does nothing*. Two questions follow, and
they are not the same question.

**Does weighing the fields by the evidence predict better?** Uniform weights, kernel alignment
(per field and jointly) and the evidence optimum are each fitted on the training part of a split
and scored on the molecules held out. Item D6 found that giving the evidence more freedom over
the shape of the kernel improved its own number and not prediction; this asks whether the same
happens here.

**Is the posterior right about which fields matter?** That is what a posterior over the weights
is *for*, and it can be checked: take the Bayes factor the model assigns each field, then remove
that field outright, refit and see what the held-out molecules say. If the factor is a guide, the
fields it calls indispensable are the ones whose removal costs prediction.

Usage
-----
    python benchmarks/run_mixing.py
    python benchmarks/run_mixing.py --sets BZR GPB --repeats 3
    python benchmarks/run_mixing.py --report
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

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.mixing import drop_one, evidence_mixing, sample_mixing  # noqa: E402
from cmfqsar.model import CMFModel, align_mixing, krr_fit, r2_score  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "mixing")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
DRAWS = 150
BURN = 50
WEIGHTS = ("uniform", "align", "alignf", "evidence")


def _combine(Kf, h, scale):
    return np.tensordot(np.asarray(h, float) * np.asarray(scale, float), Kf, axes=(0, 0))


def _best_lam(K, y):
    best = None
    for lam in LAMS:
        e = log_evidence(K, y, float(lam), fit_bias=True)
        if np.isfinite(e) and (best is None or e > best[0]):
            best = (e, float(lam))
    return best[1] if best else 1.0


def _score(Kf, h, scale, y, tr, te):
    K = _combine(Kf, h, scale)
    lam = _best_lam(K[np.ix_(tr, tr)], y[tr])
    a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
    return float(r2_score(y[te], K[np.ix_(te, tr)] @ a + b))


def run_set(name, repeats=5, force=False, data=DATA, results=RESULTS):
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
    print(f"=== {name}: {n} molecules", flush=True)

    kernels = {}
    for a in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=float(a), charge_model="file")
        m = CMFModel(fieldset=fs, method="krr", normalize=True)
        table = AtomTable.from_molecules([build_fields(mm, fs, nn)
                                          for mm, nn in zip(mols, names)])
        kernels[float(a)] = (np.asarray(m.prepare_training_kernels(table)),
                             np.asarray(m.scale, float))
    labels = [s.label for s in FieldSet.create(physchem=FIELDS).specs]

    rows = []
    for rep in range(repeats):
        rng = np.random.RandomState(rep)
        order = rng.permutation(n)
        n_te = max(5, int(round(0.3 * n)))
        te, tr = np.sort(order[:n_te]), np.sort(order[n_te:])

        best = None                      # the attenuation, by the evidence with equal weights
        for a, (Kf, scale) in kernels.items():
            K = _combine(Kf[:, tr][:, :, tr], np.ones(len(labels)), scale)
            e = log_evidence(K, y[tr], _best_lam(K, y[tr]), fit_bias=True)
            if np.isfinite(e) and (best is None or e > best[0]):
                best = (e, a)
        alpha = best[1]
        Kf, scale = kernels[alpha]
        Ktr = Kf[:, tr][:, :, tr]

        hs = {"uniform": np.ones(len(labels)),
              "align": align_mixing(Ktr, y[tr], scale),
              "alignf": align_mixing(Ktr, y[tr], scale, mode="alignf"),
              "evidence": evidence_mixing(Ktr, y[tr], scale)}
        r2 = {k: _score(Kf, h, scale, y, tr, te) for k, h in hs.items()}

        post = sample_mixing(Ktr, y[tr], labels, scale, draws=DRAWS, burn=BURN, seed=rep)
        iv = post.interval(0.9)
        bf = drop_one(Ktr, y[tr], labels, scale)

        # what the held-out molecules say when a field is taken away outright
        without = {}
        for f, label in enumerate(labels):
            on = np.ones(len(labels))
            on[f] = 0.0
            without[label] = _score(Kf, on, scale, y, tr, te) - r2["uniform"]

        rows.append({"repeat": rep, "alpha": alpha, "r2": r2,
                     "shares": {l: float(v) for l, v in zip(labels, post.mean)},
                     "lo": {l: float(v) for l, v in zip(labels, iv[:, 0])},
                     "hi": {l: float(v) for l, v in zip(labels, iv[:, 1])},
                     "bayes_factor": bf, "without": without,
                     "h": {k: [float(v) for v in h] for k, h in hs.items()}})
        print(f"  repeat {rep}: alpha {alpha:g}; R2pred " +
              "  ".join(f"{k} {r2[k]:+.3f}" for k in WEIGHTS), flush=True)

    out = {"set": name, "n": int(n), "repeats": int(repeats), "fields": labels,
           "draws": DRAWS, "rows": rows, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def report(results=RESULTS):
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No mixing results yet; run `python benchmarks/run_mixing.py`."
    data = [json.load(open(f)) for f in files]
    labels = data[0]["fields"]
    lines = ["# A posterior over the field weights", "",
             f"{len(data)} Sutherland sets, {data[0]['repeats']} random 70/30 splits each. The "
             "attenuation is chosen by the evidence with equal weights, the weights are then set "
             "four ways on the training part, and every choice is scored on the molecules held "
             f"out. The posterior is {data[0]['draws']} slice-sampling draws.", "",
             "## Does weighing the fields predict better?", "",
             "| set | n | " + " | ".join(WEIGHTS) + " |",
             "|---|---:|" + "---:|" * len(WEIGHTS)]
    per = {w: [] for w in WEIGHTS}
    for d in sorted(data, key=lambda r: -r["n"]):
        cells = []
        for w in WEIGHTS:
            v = float(np.mean([r["r2"][w] for r in d["rows"]]))
            per[w].append(v)
            cells.append(f"{v:+.3f}")
        lines.append(f"| {d['set']} | {d['n']} | " + " | ".join(cells) + " |")
    lines.append("| **mean** |  | " + " | ".join(f"**{float(np.mean(per[w])):+.3f}**"
                                                 for w in WEIGHTS) + " |")
    rng = np.random.default_rng(0)
    diff = np.asarray(per["evidence"], float) - np.asarray(per["uniform"], float)
    bt = [float(np.mean(rng.choice(diff, len(diff), replace=True))) for _ in range(5000)]
    lines += ["",
              f"Against equal weights, the evidence optimum is {float(diff.mean()):+.3f} in "
              f"R²pred with a 95 % interval of [{np.percentile(bt, 2.5):+.3f}, "
              f"{np.percentile(bt, 97.5):+.3f}]"
              + (", which contains zero." if np.percentile(bt, 2.5) <= 0 <= np.percentile(bt, 97.5)
                 else ".")
              + " This is item D6 again: the evidence takes whatever freedom it is given over the "
              "shape of the kernel, improves its own number, and leaves prediction where it was.",
              "", "## Is the posterior right about which fields matter?", "",
              "Each field's Bayes factor against *this field does nothing*, beside what actually "
              "happens to the held-out molecules when it is removed:", "",
              "| field | mean share | 90 % interval | Bayes factor | ΔR²pred without it |",
              "|---|---:|---:|---:|---:|"]
    bfs, drops = [], []
    for label in labels:
        sh = float(np.mean([r["shares"][label] for d in data for r in d["rows"]]))
        lo = float(np.mean([r["lo"][label] for d in data for r in d["rows"]]))
        hi = float(np.mean([r["hi"][label] for d in data for r in d["rows"]]))
        bf = float(np.mean([r["bayes_factor"][label] for d in data for r in d["rows"]]))
        wo = float(np.mean([r["without"][label] for d in data for r in d["rows"]]))
        bfs += [r["bayes_factor"][label] for d in data for r in d["rows"]]
        drops += [r["without"][label] for d in data for r in d["rows"]]
        lines.append(f"| {label} | {sh:.3f} | [{lo:.3f}, {hi:.3f}] | {bf:+.1f} | {wo:+.3f} |")
    bfs = np.asarray(bfs, float)
    drops = np.asarray(drops, float)
    ok = np.isfinite(bfs) & np.isfinite(drops)
    rho = float(spearmanr(bfs[ok], -drops[ok]).statistic)
    lines += ["",
              f"Over all {int(ok.sum())} (set, split, field) cases the Bayes factor and the cost "
              f"of removing the field correlate at Spearman ρ **{rho:+.3f}** -- the sign is "
              "set so that positive means *the evidence was right*: a field it calls "
              "indispensable is one whose removal costs the held-out molecules.", "",
              ("That is the finding: the posterior cannot improve a prediction, and it can tell "
               "you which fields the prediction depends on."
               if rho > 0.3 else
               "That is weaker than the interpretation claim deserves. The Bayes factor orders "
               "the fields by what the training activities say, and the held-out molecules do "
               "not fully agree; read it as what this training set can see rather than as a "
               "property of the chemistry."), "",
              "A note on the intervals. They are wide, and that is the answer rather than a "
              "failure of the sampler: on a congeneric series the field kernels are strongly "
              "correlated, so many mixtures explain the activities about equally well, and a "
              "point estimate that hides this -- any of the three point estimates above -- is "
              "claiming a precision the data do not support."]
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
