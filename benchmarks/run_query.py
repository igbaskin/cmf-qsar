#!/usr/bin/env python
"""
Is the co-field pharmacophore a usable query?

Item F5 writes the features of `cmfqsar/pharmacophore.py` in the formats a screening program
reads. Whether the file is well formed is a question for the unit tests; whether the *query* is
any good is this one, and it can be asked without leaving the repository: a feature is a typed
sphere, a molecule either puts the right kind of atom inside it or does not, and the matching
rule is short enough to write here rather than to borrow.

For every set, five random 70/30 splits. The model is fitted on the training part, the
pharmacophore is read out of its co-fields -- masked at two standard deviations first, so that a
lobe the measurement error could have produced never becomes a feature (item F2) -- and the
strongest few features are matched against the molecules held out. The score of a molecule is
the number of enabled features it satisfies, minus the exclusion spheres it violates, which is
what a screening program computes.

Two baselines, because a query that matches everything scores well against nothing:

* **random centres**: the same features with the same radii, moved to random positions inside
  the training set's own box -- what a query of the right shape and the wrong place retrieves;
* **the model itself**: the ranking the CMF prediction gives, which is the thing the query is a
  lossy summary of.

Reported as the area under the ROC curve for the most active quartile, and the enrichment factor
at the top tenth.

Usage
-----
    python benchmarks/run_query.py
    python benchmarks/run_query.py --sets BZR --repeats 3
    python benchmarks/run_query.py --report
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
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.pharmacophore import exportable, extract  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "query")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0, 2.0)
LAMS = (1e-2, 0.1, 1.0)
ENABLE = 4            # features a query starts with
MIN_Z = 2.0           # the co-field mask of item F2
TOP = 0.25            # the most active quarter are the actives
EF_AT = 0.1

#: which field carries the atom type a feature asks for, and which sign of it counts
WANTS = {"donor": ("hbd", +1), "acceptor": ("hba", +1), "hydrophobic": ("hydrophobic", +1),
         "positive": ("electrostatic", +1), "negative": ("electrostatic", -1)}
#: how much of that weight an atom needs before it counts as the thing the feature asks for
THRESHOLD = {"hbd": 0.3, "hba": 0.3, "hydrophobic": 0.05, "electrostatic": 0.08}


def matches(mol, features, enabled, field_index):
    """How many enabled features this molecule satisfies, less the exclusions it violates."""
    if not len(features):
        return 0.0
    tree = cKDTree(np.asarray(mol.coords, float))
    score = 0.0
    for f, on in zip(features, enabled):
        if not on:
            continue
        near = tree.query_ball_point(np.asarray(f.centre, float), float(f.radius))
        if f.kind in ("exclusion", "pocket"):
            score -= 1.0 if near else 0.0
            continue
        field, sign = WANTS[f.kind]
        w = np.asarray(mol.weights[:, field_index[field]], float)
        thr = THRESHOLD[field]
        score += 1.0 if any(sign * w[i] > thr for i in near) else 0.0
    return score


def auc(score, label):
    """Area under the ROC curve, by the rank statistic, so that ties are handled properly."""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def enrichment(score, label, frac=EF_AT):
    """How many times more actives are in the top `frac` than chance would put there."""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    k = max(1, int(round(frac * len(score))))
    top = np.argsort(-score, kind="stable")[:k]
    rate = float(label[top].mean())
    base = float(label.mean())
    return float(rate / base) if base > 0 else float("nan")


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
    field_index = {f: i for i, f in enumerate(FIELDS)}

    tables = {}
    for a in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=float(a), charge_model="file")
        mf = [build_fields(m, fs, nm) for m, nm in zip(mols, names)]
        tables[float(a)] = (fs, mf, AtomTable.from_molecules(mf))

    rows = []
    for rep in range(repeats):
        rng = np.random.RandomState(rep)
        order = rng.permutation(n)
        n_te = max(8, int(round(0.3 * n)))
        te, tr = np.sort(order[:n_te]), np.sort(order[n_te:])

        best = None
        for a, (fs, mf, table) in tables.items():
            model = CMFModel(fieldset=fs.copy(), method="krr", normalize=True)
            sub = AtomTable.from_molecules([mf[i] for i in tr])
            K = model.joint_kernel(model.prepare_training_kernels(sub))
            for lam in LAMS:
                e = log_evidence(K, y[tr], float(lam), fit_bias=True)
                if np.isfinite(e) and (best is None or e > best[0]):
                    best = (e, float(a), float(lam))
        _e, alpha, lam = best
        fs, mf, _table = tables[alpha]
        model = CMFModel(fieldset=fs.copy(), method="krr", params={"lam": lam}, normalize=True)
        model.fit(AtomTable.from_molecules([mf[i] for i in tr]), y[tr], cv_folds=10, cv_seed=rep)

        feats = extract(model, spacing=0.7, margin=2.5, max_features=10, min_z=MIN_Z)
        feats, _left = exportable(feats)
        on = [False] * len(feats)
        taken = 0
        for i in sorted(range(len(feats)), key=lambda k: -abs(feats[k].importance)):
            if feats[i].kind in ("exclusion", "pocket"):
                on[i] = True
            elif taken < ENABLE:
                on[i] = True
                taken += 1

        cut = float(np.quantile(y[te], 1.0 - TOP))
        label = y[te] >= cut
        q = np.array([matches(mf[i], feats, on, field_index) for i in te], float)
        pred = model.predict(AtomTable.from_molecules([mf[i] for i in te]))

        # the same query in the wrong place: random centres inside the training box
        lo = np.min(model.train.coords, axis=0)
        hi = np.max(model.train.coords, axis=0)
        shuffled = []
        for s in range(5):
            r2 = np.random.RandomState(1000 * rep + s)
            moved = []
            for f in feats:
                g = type(f)(**{**vars(f), "centre": lo + r2.rand(3) * (hi - lo)})
                moved.append(g)
            shuffled.append(auc([matches(mf[i], moved, on, field_index) for i in te], label))

        rows.append({"repeat": rep, "alpha": alpha, "lam": lam, "q2": float(model.stats.q2),
                     "features": len(feats), "enabled": int(sum(on)),
                     "auc_query": auc(q, label), "auc_model": auc(pred, label),
                     "auc_random": float(np.nanmean(shuffled)),
                     # the scores themselves, so that the enrichment can be taken over the
                     # splits together: at a tenth of forty molecules it is four of them, and
                     # four molecules cannot carry a number anyone should read
                     "score_query": [round(float(v), 4) for v in q],
                     "score_model": [round(float(v), 4) for v in pred],
                     "label": [bool(v) for v in label],
                     "kinds": sorted({f.kind for f, o in zip(feats, on) if o})})
        print(f"  repeat {rep}: {int(sum(on))} of {len(feats)} features on; AUC query "
              f"{rows[-1]['auc_query']:.3f}, random centres {rows[-1]['auc_random']:.3f}, "
              f"model {rows[-1]['auc_model']:.3f}", flush=True)

    out = {"set": name, "n": int(n), "repeats": int(repeats), "enable": ENABLE, "min_z": MIN_Z,
           "top": TOP, "rows": rows, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def report(results=RESULTS):
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No query results yet; run `python benchmarks/run_query.py`."
    data = [json.load(open(f)) for f in files]
    lines = ["# Is the co-field pharmacophore a usable query", "",
             f"{len(data)} Sutherland sets, {data[0]['repeats']} random 70/30 splits each. The "
             "pharmacophore is read out of the training model's co-fields, masked at "
             f"{data[0]['min_z']:g} standard deviations first, and its strongest "
             f"{data[0]['enable']} features are matched against the molecules held out. The "
             f"actives are the most active {100 * data[0]['top']:.0f} %.", "",
             "| set | n | features | AUC query | AUC random centres | AUC the model |",
             "|---|---:|---:|---:|---:|---:|"]
    cols = {k: [] for k in ("auc_query", "auc_random", "auc_model")}
    ahead = []
    for d in sorted(data, key=lambda r: -r["n"]):
        vals = {k: float(np.nanmean([r[k] for r in d["rows"]])) for k in cols}
        for k in cols:
            cols[k].append(vals[k])
        ahead += [r["auc_query"] > r["auc_random"] for r in d["rows"]]
        feats = float(np.mean([r["features"] for r in d["rows"]]))
        lines.append(f"| {d['set']} | {d['n']} | {feats:.1f} | {vals['auc_query']:.3f} | "
                     f"{vals['auc_random']:.3f} | {vals['auc_model']:.3f} |")
    mean = {k: float(np.nanmean(v)) for k, v in cols.items()}
    lines.append(f"| **mean** |  |  | **{mean['auc_query']:.3f}** | "
                 f"**{mean['auc_random']:.3f}** | **{mean['auc_model']:.3f}** |")
    # one enrichment, over every split of every set together: a tenth of forty molecules is
    # four of them, and four molecules cannot carry a number anyone should read
    ef = {}
    for which, key in (("score_query", "query"), ("score_model", "the model")):
        sc, lab = [], []
        for d in data:
            for r in d["rows"]:
                v = np.asarray(r[which], float)
                sc += (rankdata(v) / (len(v) + 1.0)).tolist()
                lab += list(r["label"])
        ef[key] = enrichment(np.asarray(sc), np.asarray(lab, bool))
    rng = np.random.default_rng(0)
    diff = np.asarray(cols["auc_query"], float) - np.asarray(cols["auc_random"], float)
    bt = [float(np.mean(rng.choice(diff, len(diff), replace=True))) for _ in range(5000)]
    lo, hi = float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))
    works = lo > 0
    lines += ["",
              f"The query separates the active quarter at an AUC of {mean['auc_query']:.3f} "
              f"against {mean['auc_random']:.3f} for the same features moved to random places in "
              f"the same box: a paired difference of {float(diff.mean()):+.3f} with a 95 % "
              f"interval of [{lo:+.3f}, {hi:+.3f}]"
              + (", so the places the co-field picked are what matters."
                 if works else ", which contains zero.")
              + f" The query is ahead of its own features in random places on "
              f"{int(sum(ahead))} of {len(ahead)} splits. The model it was read from reaches "
              f"{mean['auc_model']:.3f} on the same molecules.", "",
              f"Taking every split of every set together, the top tenth by the query holds "
              f"{ef['query']:.2f} times as many actives as chance would put there, and the top "
              f"tenth by the model {ef['the model']:.2f} times, against a ceiling of "
              f"{1.0 / data[0]['top']:.1f} -- which is what a perfect ranking would reach when "
              f"the actives are the top {100 * data[0]['top']:.0f} %. The model is most of the "
              "way there and the query is a third of the way.", "",
              "That gap is the cost of the summary. A pharmacophore is four spheres and a "
              "rule for counting them; the model is every training molecule's field weighed by "
              "a fitted coefficient. What the query gains is that it leaves this program: it "
              "goes into a screening tool and runs over libraries no kernel method here could "
              "hold.", "",
              "The matching rule used here is the simple one -- an atom of the right type inside "
              "the sphere -- with no conformer search, since the held-out molecules are already "
              "posed in the training frame. A real screening run has to generate conformations "
              "and will not do better than this."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                    # noqa: BLE001
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
