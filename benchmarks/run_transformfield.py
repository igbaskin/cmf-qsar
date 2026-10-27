#!/usr/bin/env python
"""
Does a transformation carry its effect from one series to another? (plan item MC6)

A matched molecular pair is usually a row in a table: a transformation written as two group
names, and the activity change it produced.  In a series whose molecules are in one frame it is
also a **difference of two fields**: a local object sitting where the two groups differ, with its
geometry kept.  This asks whether that object transfers -- whether a change measured in other
series predicts the change in a series the model has never seen.

Plan item D5 is the warning and the reference.  Increments fitted on one series did not transfer.
An increment is a number attached to an atom type; the field difference is the same
transformation with its shape and charge kept, so it can tell a methyl that points into a pocket
from one that points at the solvent.

How it is measured
------------------
* **The data** are the only sets in the repository with a crystal pose for every ligand: the ten
  Merck KGaA FEP series of Schindler et al. and the eight Schrodinger JACS series.  Both are
  congeneric series prepared in the binding site, so the two members of a pair are already in one
  frame.  The duplicate ``eg5_alternativeloop`` is left out, since it is the same ligands.
* **The pairs** come from the matched-pair rule of plan item MC1 (RDKit's `rdMMPA`).
* **The sample is the pair**, its descriptor the posed difference field of
  `cmfqsar/transformfield.py`, and its target the measured change.  The kernel is the ordinary
  overlap kernel, which is bilinear, so the ridge is fitted **without an intercept** and the
  model is exactly antisymmetric: reverse a pair and it predicts the opposite change.
* **The folds hold out a whole series**, and the ridge is chosen inside the training series by
  the same rule, so the held-out series is never seen in any way.
* **The reference of rule 1** is the transformation as a number: the mean measured change of the
  same transformation in the training series, which is what D5 fitted and transferred.  Where
  the held-out transformation was never seen, that reference can only predict no change, and the
  share of pairs where that happens is reported.
* **The gate** is that the held-out prediction beats the mean change of its own series.  q2 is
  therefore taken about each series' own mean, which is the strongest form of that reference.

Usage
-----
    python benchmarks/run_transformfield.py
    python benchmarks/run_transformfield.py --fields electrostatic,steric --report
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.bioisostere import matched_pairs  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.localenv import LOCAL_SPEC, local_features  # noqa: E402
from cmfqsar.model import krr_fit  # noqa: E402
from cmfqsar.transformfield import (average_field, increment_predict,  # noqa: E402
                                    increment_table, pair_field, shared)

RESULTS = os.path.join(HERE, "results", "transformfield")
FIELDS = ["electrostatic", "steric", "hydrophobic"]
ALPHA = 1.0
#: the grid runs high on purpose: at a large ridge the model predicts no change at all, so a
#: value chosen at the top is the model saying it has nothing, rather than a grid cut too short
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4)
#: the same ligands as eg5 with another protein loop, so its pairs would be counted twice
SKIP = {"eg5_alternativeloop"}


def _say(text):
    print(text, flush=True)


def paths():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "examples", "schindler", "*.sdf"))):
        name = os.path.basename(p)[:-4]
        if name not in SKIP:
            out.append((name, p))
    for p in sorted(glob.glob(os.path.join(ROOT, "examples", "ross", "jacs_set", "*.sdf"))):
        out.append((os.path.basename(p)[:-4], p))
    return out


def collect(fieldset, max_pairs=0, seed=0):
    """Every matched pair of every series, as a posed difference field."""
    items, counts = [], {}
    for name, path in paths():
        ds = load_dataset(path)
        y = np.asarray(ds.activity, float)
        if not np.isfinite(y).any():
            continue
        mols = [Chem.RemoveHs(m) for m in ds.mols]
        fields = [build_fields(m, fieldset, n) for m, n in zip(mols, ds.names)]
        pairs = matched_pairs(mols, y, ds.names)
        if max_pairs and len(pairs) > max_pairs:
            keep = np.random.RandomState(seed).choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in sorted(keep)]
        got = [t for t in (pair_field(mols, fields, p, name) for p in pairs) if t is not None]
        counts[name] = dict(molecules=len(mols), pairs=len(pairs), posed=len(got))
        items.extend(got)
        _say(f"  {name:24s} {len(mols):3d} molecules, {len(got):4d} pairs")
    return items, counts


def kernel(items, fieldset):
    """One kernel over every pair, each field block scaled to unit mean diagonal."""
    specs = [s for s in fieldset.specs if s.enabled]
    P = AtomTable.from_molecules([t.field for t in items])
    K = np.asarray(kernel_matrices(P, P, specs, symmetric=True), float)
    h = fieldset.mixing_vector()
    out = np.zeros(K.shape[1:])
    for f in range(len(specs)):
        d = float(np.mean(np.abs(np.diag(K[f]))))
        out += h[f] * K[f] / (d if d > 1e-300 else 1.0)
    return out


def invariant_kernel(items):
    """The same pairs described without a frame, as the autocorrelation of the difference field.

    This is the control for the frame: if the framed field fails and this fails in the same way,
    the frame convention is not what went wrong.
    """
    X = local_features([t.field for t in items], spec=LOCAL_SPEC)
    K = np.asarray(autocorrelation_kernel(X, n_bins=LOCAL_SPEC.n_bins, normalize=False), float)
    d = float(np.mean(np.abs(np.diag(K))))
    return K / (d if d > 1e-300 else 1.0)


def choose_lam(K, y, series, train, names, cap=1500, seed=0):
    """The ridge chosen inside the training series, by the same leave-one-series-out rule.

    The search runs on at most `cap` of the training pairs, taken evenly from the series, because
    the choice does not need every pair and a full search would cost more than the measurement.
    """
    rng = np.random.RandomState(int(seed))
    rows = []
    per = max(1, int(cap // max(len(names), 1)))
    for s in names:
        got = np.array([i for i in train if series[i] == s], int)
        if len(got) > per:
            got = got[np.sort(rng.choice(len(got), per, replace=False))]
        rows.append(got)
    small = np.concatenate(rows)
    best = None
    for lam in LAMS:
        got, want, lab = [], [], []
        for s in names:
            itest = np.array([i for i in small if series[i] == s], int)
            itrain = np.array([i for i in small if series[i] != s], int)
            if not len(itest) or not len(itrain):
                continue
            got.append(_fit_predict(K, y, itrain, itest, lam))
            want.append(y[itest])
            lab.extend([s] * len(itest))
        if not got:
            continue
        q = q2_within(np.concatenate(want), np.concatenate(got), lab)
        if best is None or q > best[0]:
            best = (q, lam)
    return float(best[1]) if best else float(LAMS[0])


def q2_within(y, pred, series):
    """q2 with each series compared against its own mean change, which is the item's reference."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = 0.0
    for s in set(series):
        m = np.array([x == s for x in series])
        ss_tot += float(np.sum((y[m] - y[m].mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _fit_predict(K, y, train, test, lam):
    a, b = krr_fit(K[np.ix_(train, train)], y[train], float(lam), fit_bias=False)
    return K[np.ix_(test, train)] @ a + b


def run(fields=FIELDS, alpha=ALPHA, max_pairs=0, seed=0, results=RESULTS):
    t0 = time.time()
    fieldset = FieldSet.create(physchem=list(fields), alpha=float(alpha))
    _say("Pairs by series:")
    items, counts = collect(fieldset, max_pairs=max_pairs, seed=seed)
    series = [t.series for t in items]
    names = sorted(set(series))
    y = np.asarray([t.delta for t in items], float)
    _say(f"{len(items)} pairs in {len(names)} series, {time.time() - t0:.0f} s")

    many = shared(items, least=2)
    _say(f"{len(set(t.key for t in items))} transformations, {len(many)} of them in more than "
         f"one series")

    K = kernel(items, fieldset)
    _say(f"kernel {K.shape[0]}x{K.shape[0]}, {time.time() - t0:.0f} s")

    Kfree = invariant_kernel(items)
    _say(f"the frame-free kernel too, {time.time() - t0:.0f} s")

    idx = {s: np.array([i for i, v in enumerate(series) if v == s], int) for s in names}
    pred_field = np.zeros(len(y))
    pred_free = np.zeros(len(y))
    pred_incr = np.zeros(len(y))
    seen_incr = np.zeros(len(y), bool)
    chosen = {}
    for s in names:
        test = idx[s]
        train = np.array([i for i in range(len(y)) if series[i] != s], int)
        rest = [n for n in names if n != s]
        lam = choose_lam(K, y, series, train, rest, seed=seed)
        chosen[s] = lam
        pred_field[test] = _fit_predict(K, y, train, test, lam)
        lam_free = choose_lam(Kfree, y, series, train, rest, seed=seed)
        pred_free[test] = _fit_predict(Kfree, y, train, test, lam_free)
        table = increment_table([items[i] for i in train])
        pred_incr[test], seen_incr[test] = increment_predict(table, [items[i] for i in test])
        _say(f"  {s:24s} ridge {lam:g}, {time.time() - t0:.0f} s")
    _say(f"folds done, {time.time() - t0:.0f} s")

    # the control: the same descriptor, trained inside each series, so that nothing has to
    # transfer. It says whether a difference field describes a change at all. The folds hold out
    # whole molecules, not pairs: two pairs that share a molecule share half of their field and
    # part of their measurement, so splitting pairs at random would let the answer in.
    pred_in = np.zeros(len(y))
    done_in = np.zeros(len(y), bool)
    rng = np.random.RandomState(int(seed))
    for s in names:
        rows_s = idx[s]
        mols_here = sorted({n for i in rows_s for n in items[i].names})
        fold_of = {m: int(rng.randint(0, 5)) for m in mols_here}
        best = None
        for lam in LAMS:                      # the ridge is searched here too, on these folds
            got = np.zeros(len(rows_s))
            used = np.zeros(len(rows_s), bool)
            for f in range(5):
                out_f = [k for k, i in enumerate(rows_s)
                         if f in (fold_of[items[i].names[0]], fold_of[items[i].names[1]])]
                in_f = [k for k, i in enumerate(rows_s)
                        if f not in (fold_of[items[i].names[0]], fold_of[items[i].names[1]])]
                if len(out_f) and len(in_f) >= 5:
                    got[out_f] = _fit_predict(K, y, rows_s[in_f], rows_s[out_f], lam)
                    used[out_f] = True
            if not used.any():
                continue
            q = q2_within(y[rows_s][used], got[used], [s] * int(used.sum()))
            if best is None or q > best[0]:
                best = (q, lam, got, used)
        if best is not None:
            pred_in[rows_s] = best[2]
            done_in[rows_s] = best[3]
    _say(f"  inside one series q2 {q2_within(y[done_in], pred_in[done_in], [series[i] for i in np.where(done_in)[0]]):+.3f} on {done_in.mean():.0%} of the pairs, {time.time() - t0:.0f} s")

    rows = [dict(name="no change at all", q2=q2_within(y, np.zeros(len(y)), series)),
            dict(name="the transformation as a field, inside one series (no transfer)",
                 q2=q2_within(y[done_in], pred_in[done_in],
                              [series[i] for i in np.where(done_in)[0]]),
                 share=float(done_in.mean())),
            dict(name="the transformation as a number (plan item D5)",
                 q2=q2_within(y, pred_incr, series)),
            dict(name="the transformation as a field, without a frame",
                 q2=q2_within(y, pred_free, series)),
            dict(name="the transformation as a field", q2=q2_within(y, pred_field, series))]
    for r in rows:
        _say(f"  {r['name']:46s} q2 {r['q2']:+.3f}")
    _say(f"  the transformation was seen in training for {seen_incr.mean():.1%} of the pairs")

    boot = _bootstrap(y, pred_field, pred_incr, series, names, seed)
    _say(f"  the field against the number {boot['difference'][0]:+.3f} "
         f"[{boot['difference'][1]:+.3f}, {boot['difference'][2]:+.3f}]")
    _say(f"  the field against no change  {boot['field'][0]:+.3f} "
         f"[{boot['field'][1]:+.3f}, {boot['field'][2]:+.3f}]")

    per_series = []
    for s in names:
        m = idx[s]
        per_series.append(dict(
            series=s, pairs=int(len(m)), lam=float(chosen[s]),
            q2_field=float(q2_within(y[m], pred_field[m], [s] * len(m))),
            q2_number=float(q2_within(y[m], pred_incr[m], [s] * len(m))),
            q2_free=float(q2_within(y[m], pred_free[m], [s] * len(m))),
            q2_inside=float(q2_within(y[m][done_in[m]], pred_in[m][done_in[m]],
                                      [s] * int(done_in[m].sum()))),
            seen=float(seen_incr[m].mean()),
            spread=float(np.std(y[m]))))

    out = dict(fields=list(fields), alpha=float(alpha), pairs=len(y), series=len(names),
               transformations=len(set(t.key for t in items)), shared=len(many),
               seen=float(seen_incr.mean()), rows=rows, boot=boot, per_series=per_series,
               counts=counts, seconds=time.time() - t0,
               most_shared=[_one_transformation(items, k, v) for k, v in
                            sorted(many.items(), key=lambda kv: -len(kv[1]))[:12]])
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def _one_transformation(items, key, in_series):
    """What one transformation did, series by series: the question of the item without a model."""
    per = {}
    for t in items:
        if t.key == key:
            per.setdefault(t.series, []).append(t.delta)
        elif t.key == (key[1], key[0]):
            per.setdefault(t.series, []).append(-t.delta)
    means = {s: float(np.mean(v)) for s, v in per.items()}
    v = np.asarray(list(means.values()), float)
    return dict(frm=key[0], to=key[1], series=sorted(means), pairs=sum(len(x) for x in
                per.values()), mean=float(v.mean()), sd=float(v.std()),
                lowest=float(v.min()), highest=float(v.max()))


def _bootstrap(y, pred_field, pred_incr, series, names, seed=0, runs=2000):
    """Intervals by resampling whole series, which is what the folds are."""
    rng = np.random.RandomState(int(seed))
    idx = {s: np.array([i for i, v in enumerate(series) if v == s], int) for s in names}
    field, diff = [], []
    for _ in range(int(runs)):
        pick = [names[i] for i in rng.randint(0, len(names), len(names))]
        rows = np.concatenate([idx[s] for s in pick])
        lab = [s for s in pick for _ in idx[s]]
        field.append(q2_within(y[rows], pred_field[rows], lab))
        diff.append(field[-1] - q2_within(y[rows], pred_incr[rows], lab))
    def interval(v, point):
        v = np.asarray([x for x in v if np.isfinite(x)], float)
        if not len(v):
            return [point, float("nan"), float("nan")]
        return [point, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    point_f = q2_within(y, pred_field, series)
    point_d = point_f - q2_within(y, pred_incr, series)
    return dict(field=interval(field, point_f), difference=interval(diff, point_d))


def report(results=RESULTS):
    path = os.path.join(results, "summary.json")
    if not os.path.exists(path):
        raise SystemExit("nothing to report: run the benchmark first")
    with io.open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    out = ["# Does a transformation carry its effect from one series to another? "
           "(plan item MC6)", "",
           "A matched molecular pair in a fixed pose is a difference of two fields: a local "
           "object with its geometry kept, rather than a number attached to an atom type. This "
           "asks whether that object transfers to a series the model has never seen. Plan item "
           "D5 is the reference and the warning: increments fitted on one series did not "
           "transfer.", "",
           f"{d['pairs']} matched pairs from {d['series']} congeneric series with a crystal pose "
           f"for every ligand (the Merck KGaA FEP sets and the Schrodinger JACS sets). "
           f"{d['transformations']} distinct transformations, {d['shared']} of them in more than "
           f"one series. The fields are {', '.join(d['fields'])} at an attenuation of "
           f"{d['alpha']:g}; the folds hold out a whole series and the ridge is chosen inside "
           "the training series the same way; the ridge has no intercept, so the model predicts "
           "exactly the opposite change for the reverse of a pair.", "",
           "| what predicts the change | q2 against each series' own mean |", "| --- | ---: |"]
    for r in d["rows"]:
        out.append(f"| {r['name']} | {r['q2']:+.3f} |")
    b = d["boot"]
    out += ["",
            f"The field against the number: {b['difference'][0]:+.3f} "
            f"[{b['difference'][1]:+.3f}, {b['difference'][2]:+.3f}], resampling whole series. "
            f"The field against predicting no change: {b['field'][0]:+.3f} "
            f"[{b['field'][1]:+.3f}, {b['field'][2]:+.3f}].", "",
            f"The transformation of a held-out pair had been seen in another series for "
            f"{d['seen']:.1%} of the pairs, so the reference of D5 can say nothing at all about "
            f"the rest and predicts no change for them.", "",
            "| series | pairs | spread of the change | q2, field | q2, no frame | q2, number "
            "| q2 inside the series | transformation seen |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in d["per_series"]:
        out.append(f"| {s['series']} | {s['pairs']} | {s['spread']:.2f} | {s['q2_field']:+.3f} "
                   f"| {s.get('q2_free', float('nan')):+.3f} | {s['q2_number']:+.3f} "
                   f"| {s.get('q2_inside', float('nan')):+.3f} | {s['seen']:.0%} |")
    if d.get("most_shared"):
        out += ["", "## The same transformation in several series", "",
                "Before any model: what did one transformation actually do, series by series? "
                "The mean change is taken inside each series first, so a series with many pairs "
                "does not outweigh the rest.", "",
                "| transformation | series | pairs | mean change | lowest | highest |",
                "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for t in d["most_shared"]:
            out.append(f"| {t['frm']} -> {t['to']} | {len(t['series'])} | {t['pairs']} | "
                       f"{t['mean']:+.2f} | {t['lowest']:+.2f} | {t['highest']:+.2f} |")
        turns = sum(1 for t in d["most_shared"] if t["lowest"] < 0 < t["highest"])
        out += ["",
                f"{turns} of these {len(d['most_shared'])} transformations raise the activity in "
                f"one series and lower it in another. That is the item's question answered by "
                f"the data before any model is fitted: a transformation of this kind does not "
                f"have one effect to carry."]
    out += [""] + _verdict(d)
    return "\n".join(out)


def _verdict(d):
    b = d["boot"]
    field = next(r for r in d["rows"] if r["name"] == "the transformation as a field")
    number = next(r for r in d["rows"] if "as a number" in r["name"])
    passes = np.isfinite(b["field"][1]) and b["field"][1] > 0
    better = np.isfinite(b["difference"][1]) and b["difference"][1] > 0
    out = ["## The verdict", ""]
    if passes:
        out += [f"**A transformation does carry some of its effect across series.** The field "
                f"model reaches q2 {field['q2']:+.3f} against each series' own mean change, and "
                f"the interval over series is {b['field'][1]:+.3f} to {b['field'][2]:+.3f}, "
                f"which excludes zero. That is the gate of the item."]
    else:
        worse = np.isfinite(b["field"][2]) and b["field"][2] < 0
        out += [f"**A transformation does not carry its effect across series.** The field model "
                f"reaches q2 {field['q2']:+.3f} against each series' own mean change, with an "
                f"interval of {b['field'][1]:+.3f} to {b['field'][2]:+.3f} over the series"
                + (", which lies entirely below zero: the model is worse than saying that every "
                   "pair changes the activity by the average of its series."
                   if worse else ", which contains zero.")
                + " The gate of the item is not passed, and the result stands beside plan item "
                  "D5 rather than replacing it: what a group does depends on the pocket it does "
                  "it in, and neither a number nor a field carries that from one protein to "
                  "another."]
        free = next((r for r in d["rows"] if "without a frame" in r["name"]), None)
        if free is not None:
            out += ["",
                    f"**It is not the frame.** The same pairs described without one, as the "
                    f"rotation-invariant autocorrelation of the difference field, reach q2 "
                    f"{free['q2']:+.3f}. A frame convention that put the groups in the wrong "
                    f"place would show up as a difference between those two rows."]
    out += ["",
            f"Against the reference of D5 -- the same transformation as a number, averaged over "
            f"the series that have it -- the field is "
            + ("ahead" if better else "not ahead")
            + f": q2 {field['q2']:+.3f} against {number['q2']:+.3f}, a difference of "
              f"{b['difference'][0]:+.3f} [{b['difference'][1]:+.3f}, {b['difference'][2]:+.3f}]."
              f" The number has a second limit that the field does not: it can say nothing about "
              f"a transformation it has never seen, which is {100 - 100 * d['seen']:.0f} per "
              f"cent of the pairs here, while a field is read for what it looks like and not for "
              f"its name.", ""]
    inside = next((r for r in d["rows"] if "inside one series" in r["name"]), None)
    if inside is not None:
        out += [
            f"**Is the descriptor itself the problem?** Trained inside each series in five "
            f"folds, where nothing has to transfer, the same difference fields reach q2 "
            f"{inside['q2']:+.3f}. Those folds hold out whole molecules, not pairs, since two "
            f"pairs that share a molecule share half of their field and part of their "
            f"measurement; the row covers the "
            f"{100 * inside.get('share', 1.0):.0f} per cent of the pairs that such a split can "
            f"test. Its ridge is chosen on the same folds it is scored on, so it is a best "
            f"case. "
            + ("That is a working model of a change, and it stops working as soon as it is "
               "asked to cross from one protein to another."
               if inside["q2"] > 0.1 else
               "That is barely better than the series mean either, so these pairs are hard to "
               "predict even without a transfer, and the measurement above cannot separate a "
               "failure to transfer from a descriptor that says little."), ""]
    out += [
            "**The narrow activity ranges are the known risk of these sets**, and the table "
            "above shows it: the spread of the change inside a series is often well under one "
            "log unit, so there is little for any model to explain. A series whose changes are "
            "all the same size cannot separate a good prediction from a mediocre one.", ""]
    return out


def write_table(results=RESULTS) -> str:
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", default=",".join(FIELDS))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--max-pairs", type=int, default=0, metavar="N",
                    help="at most this many pairs per series")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    run(fields=[s.strip() for s in a.fields.split(",") if s.strip()], alpha=a.alpha,
        max_pairs=a.max_pairs, seed=a.seed)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
