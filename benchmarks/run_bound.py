#!/usr/bin/env python
"""
Does ranking by a confidence bound beat ranking by the prediction?

A regression model used to prioritise a library has always been read here as a point estimate:
compute the predicted activity, sort, take the top. The objection is that a screening library is
by construction dominated by molecules unlike the training set, that a kernel method returns the
training mean presented as a prediction for exactly those, and that a point estimate cannot
tell that answer from a real one. The remedy is to rank by a lower bound instead,

    score = prediction - beta * posterior standard deviation

which promotes a molecule only when the model is both optimistic and sure. `CMFModel.rank_score`
computes it; this measures whether it helps, on the two regimes that differ in how much the
uncertainty varies across the library:

    random     a random split of one series, where every held-out molecule has training
               neighbours and the spread of the posterior is narrow
    chemotype  one Bemis-Murcko chemotype held out at a time, where a held-out molecule has no
               scaffold-mate in the training set and the spread is wide

The second is the regime the objection is about, and the one where a bound can act. The
metrics are the early-recognition ones, since what a prioritisation is judged by is whether the
actives are at the top: a molecule counts as active when its activity is above the median of the
training part, so the labels never depend on anything held out.

Usage
-----
    python benchmarks/run_bound.py
    python benchmarks/run_bound.py --sets THR,COX2 --betas 0,0.5,1,2
    python benchmarks/run_bound.py --report
    python benchmarks/run_bound.py --reference      # the reference of the tutorial section

The stage --reference ranks with the distance of a molecule to its nearest training molecule in
place of the posterior standard deviation, rescaled to the same mean. It is the one-number
version of the uncertainty: if it does as well, the standard deviation adds nothing beyond how
far a molecule lies from the training set. The random split is repeated with five seeds there,
and the differences get bootstrap intervals over the sets.
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

from cmfqsar.chemotypes import chemotype_labels  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, screening_scores  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "bound")
SUTHERLAND = os.path.join(ROOT, "examples", "sutherland")

SETS = ("COX2", "DHFR", "BZR", "AChE", "THR", "ACE", "GPB", "THER")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
ALPHA = 0.5
LAM = 0.1


def tags_first(betas):
    """The tag of the first beta, used to record the enrichment ceiling only once per fold."""
    return f"{betas[0]:g}"


def _fit_and_rank(mf, y, train, test, fs, betas):
    """(beta -> ranking score over `test`), plus the spread of the posterior there."""
    tr = AtomTable.from_molecules([mf[i] for i in train])
    te = AtomTable.from_molecules([mf[i] for i in test])
    model = CMFModel(fieldset=fs, method="gp", normalize=True, params={"lam": LAM})
    model.fit(tr, y[train], cv_folds=0)
    out = {}
    for b in betas:
        score, yp, sd = model.rank_score(te, beta=float(b))
        out[f"{b:g}"] = np.asarray(score, float)
    return out, np.asarray(sd, float), np.asarray(yp, float)


def run(sets=SETS, betas=BETAS, out_dir=RESULTS):
    os.makedirs(out_dir, exist_ok=True)
    for name in sets:
        path = os.path.join(SUTHERLAND, f"{name}.sdf")
        if not os.path.exists(path):
            print(f"  {name}: no such file, skipped")
            continue
        t0 = time.time()
        ds = load_dataset(path)
        y = np.asarray(ds.activity, float)
        keep = np.flatnonzero(np.isfinite(y))
        mols = [ds.mols[i] for i in keep]
        names = [ds.names[i] for i in keep]
        y = y[keep]
        fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA, charge_model="file")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        labels, _ = chemotype_labels(mols, min_size=3)
        n = len(y)

        record = {"set": name, "n": int(n), "betas": [float(b) for b in betas],
                  "alpha": ALPHA, "lam": LAM, "regimes": {}}

        # the two regimes, as lists of (train, test) index arrays
        rng = np.random.default_rng(0)
        order = rng.permutation(n)
        folds = [order[i::5] for i in range(5)]
        splits = {"random": [(np.setdiff1d(np.arange(n), f), f) for f in folds],
                  "chemotype": [(np.flatnonzero(labels != g), np.flatnonzero(labels == g))
                                for g in sorted(set(labels.tolist()))]}

        for regime, parts in splits.items():
            acc = {f"{b:g}": {"auc": [], "ef5": [], "ef10": [], "bedroc": []} for b in betas}
            sd_spread, ceiling = [], []
            for train, test in parts:
                if len(test) < 4 or len(train) < 8:
                    continue
                med = float(np.median(y[train]))
                lab = np.where(y[test] > med, 1.0, -1.0)
                if len(set(lab.tolist())) < 2:
                    continue
                try:
                    scores, sd, yp = _fit_and_rank(mf, y, train, test, fs, betas)
                except Exception as exc:                    # noqa: BLE001
                    print(f"    {regime}: a fold failed: {exc}")
                    continue
                sd_spread.append(float(np.std(sd) / max(np.mean(sd), 1e-12)))
                for tag, sc in scores.items():
                    m = screening_scores(lab, sc, fractions=(0.05, 0.10))
                    if tag == tags_first(betas):
                        ceiling.append((m.get("max_ef_5", np.nan), m.get("max_ef_10", np.nan)))
                    acc[tag]["auc"].append(m.get("auc", np.nan))
                    acc[tag]["ef5"].append(m.get("ef_5", np.nan))
                    acc[tag]["ef10"].append(m.get("ef_10", np.nan))
                    acc[tag]["bedroc"].append(m.get("bedroc_20", np.nan))
            record["regimes"][regime] = {
                "relative_sd_spread": float(np.nanmean(sd_spread)) if sd_spread else float("nan"),
                "n_folds": len(sd_spread),
                "max_ef5": float(np.nanmean([c[0] for c in ceiling])) if ceiling else float("nan"),
                "max_ef10": float(np.nanmean([c[1] for c in ceiling])) if ceiling else float("nan"),
                "scores": {tag: {k: (float(np.nanmean(v)) if len(v) else float("nan"))
                                 for k, v in d.items()} for tag, d in acc.items()}}
            print(f"  {name} {regime}: {len(sd_spread)} folds, relative spread of the posterior "
                  f"{record['regimes'][regime]['relative_sd_spread']:.3f}", flush=True)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
            json.dump(record, fh, indent=1)
        print(f"  {name} written, {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------- #
# The stage --reference: the distance to the nearest training molecule as the uncertainty
# --------------------------------------------------------------------------- #
REFERENCE_JSON = "reference.json"
REFERENCE_BETAS = (1.0, 2.0)
REFERENCE_SEEDS = 5                  # the random five-fold split, repeated with five seeds
REFERENCE_BOOT = 4000
REFERENCE_SEED = 20261115


def field_kernels_of(mf, fs):
    """(a model with the settings of `_fit_and_rank`, the per-field kernels of the whole set).

    A cosine-normalised field kernel of two molecules does not depend on which molecules are in
    the training part. The scale of a field does: it makes the mean diagonal of the field over
    the training part one, and a field that is zero on some molecules (the donor field on a
    molecule without a donor) has a mean diagonal below one. So the per-field kernels are
    computed once and the scale again for every fold (`joint_for`)."""
    model = CMFModel(fieldset=fs, method="gp", normalize=True, params={"lam": LAM})
    return model, model.prepare_training_kernels(AtomTable.from_molecules(mf))


def joint_for(model, F, train):
    """The joint kernel of the whole set with the field scale of the training part `train`."""
    model.scale = model._compute_scale(F[:, train][:, :, train])
    return model.joint_kernel(F)


def gp_on_blocks(K, y, train, test):
    """(prediction, posterior standard deviation, distance to the nearest training molecule)
    over `test`, from the blocks of the joint kernel `K` of this fold (`joint_for`). The fit and
    the variance are those of `CMFModel(method="gp")`; the distance is the feature-space distance
    to the closest training molecule, sqrt(K(x,x) + K(t,t) - 2 K(x,t)) at its smallest."""
    from cmfqsar.model import _Learner

    learner = _Learner("gp", {"lam": LAM}).fit(K[np.ix_(train, train)], y[train])
    Kt = K[np.ix_(test, train)]
    yp = Kt @ learner.alpha_ + learner.b_
    var = learner.predict_variance(Kt, np.diag(K)[test])
    sd = np.sqrt(np.maximum(np.asarray(var, float), 0.0))
    d2 = np.diag(K)[test][:, None] + np.diag(K)[train][None, :] - 2.0 * Kt
    nearest = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return yp, sd, nearest


def reference_scores(yp, sd, nearest, betas=REFERENCE_BETAS):
    """The scores to rank by: the prediction, the bound with the posterior standard deviation,
    and the bound with the distance to the nearest training molecule. The distance is rescaled to
    the mean of the standard deviation over the same held-out part, which uses no activity, so
    that one beta means the same amount of caution for both."""
    scale = float(np.mean(sd)) / max(float(np.mean(nearest)), 1e-12)
    out = {"prediction": yp}
    for b in betas:
        out[f"standard deviation, beta {b:g}"] = yp - b * sd
        out[f"nearest distance, beta {b:g}"] = yp - b * scale * nearest
    return out


def _rank_corr(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(ra) > 2 else float("nan")


def reference_set(name, seeds=REFERENCE_SEEDS, check=True):
    """Every fold of one set: BEDROC(20) and AUC of every score, and the rank correlation
    between the standard deviation and the distance on the held-out part."""
    ds = load_dataset(os.path.join(SUTHERLAND, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.flatnonzero(np.isfinite(y))
    mols = [ds.mols[i] for i in keep]
    names = [ds.names[i] for i in keep]
    y = y[keep]
    fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA, charge_model="file")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    labels, _ = chemotype_labels(mols, min_size=3)
    n = len(y)
    model, F = field_kernels_of(mf, fs)
    splits = {"random": [], "chemotype": [(np.flatnonzero(labels != g), np.flatnonzero(labels == g))
                                           for g in sorted(set(labels.tolist()))]}
    for seed in range(int(seeds)):
        order = np.random.default_rng(seed).permutation(n)       # seed 0 is the stored split
        splits["random"] += [(np.setdiff1d(np.arange(n), order[i::5]), order[i::5])
                             for i in range(5)]
    out = {"set": name, "n": int(n), "seeds": int(seeds), "regimes": {}}
    checked = 0
    for regime, parts in splits.items():
        folds = []
        done_here = False
        for train, test in parts:
            if len(test) < 4 or len(train) < 8:
                continue
            med = float(np.median(y[train]))
            lab = np.where(y[test] > med, 1.0, -1.0)
            if len(set(lab.tolist())) < 2:
                continue
            yp, sd, nearest = gp_on_blocks(joint_for(model, F, train), y, train, test)
            if check and not done_here:
                # control: the blocks give what a model fitted on this fold gives
                _scores, sd_m, yp_m = _fit_and_rank(mf, y, train, test, fs, (0.0,))
                if not (np.allclose(yp, yp_m, atol=1e-8) and np.allclose(sd, sd_m, atol=1e-8)):
                    raise ValueError(f"{name}: the kernel blocks do not reproduce the model")
                done_here = True
                checked += 1
            row = {"n_test": int(len(test)), "sd_vs_distance": _rank_corr(sd, nearest),
                   "relative_sd_spread": float(np.std(sd) / max(np.mean(sd), 1e-12))}
            for key, sc in reference_scores(yp, sd, nearest).items():
                m = screening_scores(lab, sc, fractions=(0.05, 0.10))
                row[key] = {"bedroc": float(m.get("bedroc_20", np.nan)),
                            "auc": float(m.get("auc", np.nan))}
            folds.append(row)
        out["regimes"][regime] = folds
    out["control_folds_checked"] = int(checked)
    return out


def _set_means(rec, regime, key, metric):
    v = [f[key][metric] for f in rec["regimes"][regime]]
    return float(np.nanmean(v)) if v else float("nan")


def paired_over_sets(a, b, seed=REFERENCE_SEED, boot=REFERENCE_BOOT):
    """The mean of a - b over the sets, with a 95 % bootstrap interval over the sets."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, len(d), size=(boot, len(d)))].mean(axis=1)
    return {"mean": float(d.mean()), "low": float(np.percentile(draws, 2.5)),
            "high": float(np.percentile(draws, 97.5)), "ahead": int(np.sum(d > 0)),
            "n_sets": int(len(d))}


REFERENCE_PAIRS = (("standard deviation, beta 2", "prediction"),
                   ("nearest distance, beta 2", "prediction"),
                   ("standard deviation, beta 2", "nearest distance, beta 2"),
                   ("standard deviation, beta 1", "prediction"),
                   ("nearest distance, beta 1", "prediction"),
                   ("standard deviation, beta 1", "nearest distance, beta 1"))


def reference_stage(sets=SETS, seeds=REFERENCE_SEEDS, out_dir=RESULTS):
    """The stage --reference over the sets, with the summaries; writes reference.json."""
    recs = {}
    for name in sets:
        t0 = time.time()
        recs[name] = reference_set(name, seeds)
        print(f"  {name}: {sum(len(v) for v in recs[name]['regimes'].values())} folds, "
              f"{time.time() - t0:.0f} s", flush=True)
    keys = list(reference_scores(np.zeros(2), np.ones(2), np.ones(2)))
    summary = {}
    for regime in ("random", "chemotype"):
        names = [s for s in recs if recs[s]["regimes"][regime]]
        per_set = {k: {m: [_set_means(recs[s], regime, k, m) for s in names]
                       for m in ("bedroc", "auc")} for k in keys}
        summary[regime] = {
            "sets": names,
            "folds": int(sum(len(recs[s]["regimes"][regime]) for s in names)),
            "mean": {k: {m: float(np.mean(v)) for m, v in per_set[k].items()} for k in keys},
            "sd_vs_distance": float(np.mean([np.nanmean([f["sd_vs_distance"]
                                                          for f in recs[s]["regimes"][regime]])
                                             for s in names])),
            "pairs": {f"{a} minus {b}": {m: paired_over_sets(per_set[a][m], per_set[b][m])
                                         for m in ("bedroc", "auc")}
                      for a, b in REFERENCE_PAIRS}}
    out = {"seeds": int(seeds), "betas": list(REFERENCE_BETAS), "alpha": ALPHA, "lam": LAM,
           "summary": summary, "sets": recs}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, REFERENCE_JSON), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def reference_section(out_dir=RESULTS):
    """The section of table.md for the stage --reference, or nothing before it has run."""
    path = os.path.join(out_dir, REFERENCE_JSON)
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    lines = ["## The reference: the distance to the nearest training molecule (stage --reference)",
             "",
             "The bound subtracts beta posterior standard deviations from the prediction. The "
             "reference subtracts, instead, the feature-space distance of the molecule to its "
             "nearest training molecule, rescaled to the mean standard deviation of the same "
             "held-out part (no activity is used for that). The random split is repeated with "
             f"{d['seeds']} seeds (seed 0 is the split above). Every number is a mean over the "
             "folds of a set and then over the sets; an interval is a 95 % bootstrap interval "
             "over the sets.", ""]
    for regime, s in d["summary"].items():
        lines += [f"### {regime} folds", "",
                  f"{s['folds']} folds over {len(s['sets'])} sets. Rank correlation between the "
                  f"standard deviation and the distance on a held-out part: "
                  f"{s['sd_vs_distance']:+.3f}.", "",
                  "| Score to rank by | BEDROC(20) | AUC |", "|---|---:|---:|"]
        for k, v in s["mean"].items():
            lines.append(f"| {k} | {v['bedroc']:.3f} | {v['auc']:.3f} |")
        lines += ["", "| First minus second | BEDROC(20) | sets where the first is ahead | AUC |",
                  "|---|---:|---:|---:|"]
        for k, v in s["pairs"].items():
            b, a = v["bedroc"], v["auc"]
            lines.append(f"| {k} | {b['mean']:+.3f} [{b['low']:+.3f}, {b['high']:+.3f}] | "
                         f"{b['ahead']} of {b['n_sets']} | {a['mean']:+.3f} [{a['low']:+.3f}, "
                         f"{a['high']:+.3f}] |")
        lines.append("")
    return lines


def _load(out_dir=RESULTS):
    out = {}
    for name in SETS:
        p = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(p):
            out[name] = json.load(open(p))
    return out


def report(path=None, out_dir=RESULTS):
    """results/bound/table.md."""
    data = _load(out_dir)
    if not data:
        print("no results yet; run the benchmark first")
        return
    sets = [s for s in SETS if s in data]
    betas = data[sets[0]]["betas"]
    tags = [f"{b:g}" for b in betas]
    lines = ["# Ranking by a confidence bound against ranking by the prediction", "",
             "Early recognition of the actives of a held-out part, a molecule counting as active",
             "when its activity is above the median of the training part. beta is how many",
             "posterior standard deviations are subtracted from the prediction before ranking;",
             "beta = 0 is the point estimate, which is what ranking by a regression model has",
             "always meant. Averaged over the folds of each set, then over the sets.", ""]
    for regime in ("random", "chemotype"):
        spread = [data[s]["regimes"][regime]["relative_sd_spread"] for s in sets
                  if regime in data[s]["regimes"]]
        lines += [f"## {regime} folds", "",
                  f"Relative spread of the posterior standard deviation across a held-out part: "
                  f"{np.nanmean(spread):.3f}. A bound can only act where this is large.", "",
                  f"The actives are about half of a held-out part here, so the enrichment "
                  f"factors saturate early: their ceilings are "
                  f"{np.nanmean([data[s]['regimes'][regime].get('max_ef5', np.nan) for s in sets]):.2f} "
                  f"at 5 per cent and "
                  f"{np.nanmean([data[s]['regimes'][regime].get('max_ef10', np.nan) for s in sets]):.2f} "
                  f"at 10. BEDROC is the metric that is not yet saturated.", "",
                  "| beta | AUC | EF 5% | EF 10% | BEDROC(20) |", "|---|---:|---:|---:|---:|"]
        for tag in tags:
            row = []
            for key in ("auc", "ef5", "ef10", "bedroc"):
                vals = [data[s]["regimes"][regime]["scores"][tag][key] for s in sets
                        if regime in data[s]["regimes"]]
                row.append(float(np.nanmean(vals)))
            lines.append(f"| {tag} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
        lines.append("")
    lines += reference_section(out_dir)
    lines += ["## Sets", "", "| set | molecules |", "|---|---:|"]
    for s in sets:
        lines.append(f"| {s} | {data[s]['n']} |")
    text = "\n".join(lines) + "\n"
    path = path or os.path.join(out_dir, "table.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--betas", default=",".join(f"{b:g}" for b in BETAS))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reference", action="store_true",
                    help="only the stage that ranks with the distance to the nearest training "
                         "molecule as the uncertainty, the reference of the tutorial section")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if args.reference:
        reference_stage(sets=tuple(s for s in args.sets.split(",") if s))
        report()
        return
    run(sets=tuple(s for s in args.sets.split(",") if s),
        betas=tuple(float(b) for b in args.betas.split(",") if b))
    report()


if __name__ == "__main__":
    main()
