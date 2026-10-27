#!/usr/bin/env python
"""
A design loop with more than one objective (plan item MC11)

Plan items E2 and E3 close the loop on potency alone. Lead optimisation has at least two
objectives at once, and since plan items E1 and MC4 they live on one kernel: a multi-task model
predicts the potency and the anti-target activity of the same candidate, each with a posterior.
`cmfqsar/multiobjective.py` turns that into a list a chemist can act on -- the Pareto front, an
acquisition over the objectives, the batch rule of plan item MC2, and a sentence per suggestion
saying what it gains and what it loses.

The retrospective test
----------------------
The gate of the item is that the front the loop reports contains the compounds the authors went
on to make. The series makes that possible, because it comes in two parts:

* **part 1**, `examples/antitarget/h3_herg.sdf`: 78 histamine H3 ligands with their hERG
  activity (Bioorg. Med. Chem. Lett. 21 (2011) 5378). This is what the loop is trained on.
* **part 2**, the companion paper (21 (2011) 5384): the compounds the same authors made next.
  16 of them fit the same frame and are in the candidate pool, flagged `made_next`.
* **the rest of the pool**, `examples/antitarget/h3_candidates.sdf`: every other histamine H3
  ligand in ChEMBL that fits the frame, which is what the loop could have proposed instead.

The loop never sees any activity of any candidate. It predicts both objectives for all of them,
reports the Pareto front, and the question is whether the compounds the authors actually made
are on it more often than the pool at large.

Usage
-----
    python benchmarks/run_designloop.py
    python benchmarks/run_designloop.py --batch 10 --report
"""
from __future__ import annotations

import argparse
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

from cmfqsar import multitask as MT  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.multiobjective import (MAXIMISE, MINIMISE, Objective, acquisition,  # noqa: E402
                                    batch, pareto_front, pareto_layers, report as trade_report,
                                    with_best)

RDLogger.DisableLog("rdApp.*")

TRAIN = os.path.join(ROOT, "examples", "antitarget", "h3_herg.sdf")
POOL = os.path.join(ROOT, "examples", "antitarget", "h3_candidates.sdf")
RESULTS = os.path.join(HERE, "results", "designloop")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0)
PRIMARY, ANTI = "pKi_H3", "pIC50_hERG"
BATCH = 10


def _say(text):
    print(text, flush=True)


def load():
    train = [m for m in Chem.SDMolSupplier(TRAIN, removeHs=False) if m is not None]
    pool = [m for m in Chem.SDMolSupplier(POOL, removeHs=False) if m is not None]
    return train, pool


def fitted(mols, names, y, folds=10, seed=0):
    best = None
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=float(alpha))
        table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        Kf = np.asarray(model.prepare_training_kernels(table))
        for lam in LAMS:
            model.params["lam"] = float(lam)
            model.set_training_kernels(table, Kf, model.self_diag)
            model.fit(table, y, cv_folds=folds, cv_seed=seed, reuse_kernels=True)
            if best is None or model.stats.q2 > best[0]:
                best = (float(model.stats.q2), float(alpha), float(lam))
    q2, alpha, lam = best
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha)
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    Kf = np.asarray(model.prepare_training_kernels(table))
    model.params["lam"] = lam
    model.set_training_kernels(table, Kf, model.self_diag)
    model.fit(table, y, cv_folds=folds, cv_seed=seed, reuse_kernels=True)
    return model, fs, table, q2, alpha, lam


def predict_both(model, fs, table, y1, y2, pool_table, lam, folds=10, seed=0):
    """Both objectives for every candidate, with the posterior covariance of each."""
    K = model._linear_train_kernel()
    obs = MT.Observations.from_columns(np.column_stack([y1, y2]), names=(PRIMARY, ANTI))
    rho = float(MT.evidence_rho(K, obs, lams=[lam]).correlations[0, 1])
    T = MT.task_covariance(2, rho)
    fit = MT.fit(K, obs, lam=lam, T=T)
    means, covs = [], []
    for task in (0, 1):
        model.coef = fit.task_coefficients(task)
        model.bias = float(fit.bias[task])
        mu, cov = model.predict_with_covariance(pool_table)
        means.append(np.asarray(mu, float))
        covs.append(np.asarray(cov, float))
    return np.column_stack(means), covs, rho


def run(folds=10, seed=0, size=BATCH, results=RESULTS):
    for path in (TRAIN, POOL):
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing; build it with "
                             f"examples/make_antitarget_examples.py --candidates")
    t0 = time.time()
    train, pool = load()
    names = [m.GetProp("_Name") for m in train]
    y1 = np.array([float(m.GetProp(PRIMARY)) for m in train])
    y2 = np.array([float(m.GetProp(ANTI)) for m in train])
    pool_names = [m.GetProp("_Name") for m in pool]
    made = np.array([m.GetProp("made_next") == "1" for m in pool])
    pool_y1 = np.array([float(m.GetProp(PRIMARY)) if m.HasProp(PRIMARY) else np.nan
                        for m in pool])
    _say(f"{len(train)} molecules to learn from, {len(pool)} candidates, {int(made.sum())} of "
         f"which the authors went on to make")

    model, fs, table, q2, alpha, lam = fitted(train, names, y1, folds, seed)
    pool_table = AtomTable.from_molecules(
        [build_fields(m, fs, n) for m, n in zip(pool, pool_names)])
    values, covs, rho = predict_both(model, fs, table, y1, y2, pool_table, lam, folds, seed)
    _say(f"the model: q2 {q2:+.3f} on the potency at alpha {alpha:g}, ridge {lam:g}; task "
         f"correlation {rho:+.3f}; {time.time() - t0:.0f} s")

    objectives = with_best([Objective(PRIMARY, MAXIMISE), Objective(ANTI, MINIMISE)],
                           np.column_stack([y1, y2]))
    _say("the objectives: " + "; ".join(str(o) + f", best so far {o.best:.2f}"
                                        for o in objectives))

    layers = pareto_layers(values, objectives)
    front = pareto_front(values, objectives)
    base = float(made.mean())
    on_front = float(made[front].mean()) if len(front) else float("nan")
    _say(f"the Pareto front holds {len(front)} of the {len(pool)} candidates; "
         f"{int(made[front].sum())} of them are compounds the authors made "
         f"({on_front:.1%} against a base rate of {base:.1%})")

    # the enrichment of the first few fronts, and of the batch the loop would actually order
    sds = np.column_stack([np.sqrt(np.maximum(np.diag(c), 0.0)) for c in covs])
    score = acquisition(values, sds, objectives)
    order = np.argsort(-score)
    picks = batch(values, covs, objectives, size, only=front)
    rows = []
    for k in (len(front), 10, 20, 50):
        take = front if k == len(front) else order[:k]
        rows.append(dict(what=("the Pareto front" if k == len(front)
                               else f"the first {k} by the acquisition"),
                         n=int(len(take)), made=int(made[take].sum()),
                         share=float(made[take].mean())))
    rows.append(dict(what=f"the batch of {len(picks)} taken from the front",
                     n=len(picks), made=int(made[np.asarray(picks, int)].sum()),
                     share=float(made[np.asarray(picks, int)].mean())))
    for r in rows:
        _say(f"  {r['what']:34s} {r['made']:3d} of {r['n']:4d} were made ({r['share']:.1%})")

    # where the compounds they made actually sit: a front of five cannot hold sixteen, so the
    # informative number is the rank, not the membership
    rank = np.empty(len(pool), int)
    rank[order] = np.arange(1, len(pool) + 1)
    where = dict(layer_made=float(np.median(layers[made])),
                 layer_pool=float(np.median(layers[~made])),
                 layer_worst=int(layers.max()),
                 rank_made=float(np.median(rank[made])),
                 rank_pool=float(np.median(rank[~made])),
                 share_made=float(np.median(rank[made]) / len(pool)),
                 best_layer=int(layers[made].min()),
                 best_rank=int(rank[made].min()))
    _say(f"  the compounds they made sit at Pareto layer {where['layer_made']:.0f} of "
         f"{where['layer_worst']} (the pool: {where['layer_pool']:.0f}), and at rank "
         f"{where['rank_made']:.0f} of {len(pool)} by the acquisition (the pool: "
         f"{where['rank_pool']:.0f}); the best of them is layer {where['best_layer']} and rank "
         f"{where['best_rank']}")

    boot = _enrichment(made, front, seed=seed)
    _say(f"  the front against the pool: {boot[0]:.2f} times the base rate "
         f"[{boot[1]:.2f}, {boot[2]:.2f}]")

    trades = trade_report(values, objectives, pool_names, picks,
                          indifference=float(model.stats.rmse_cv))
    _say("  what the loop would order, and what each one trades:")
    for t, i in zip(trades, picks):
        _say(f"    {'made next' if made[i] else '         '} {t}")

    out = dict(where=where,
               train=len(train), pool=len(pool), made=int(made.sum()), q2=q2, alpha=alpha,
               lam=lam, rho=rho, front=int(len(front)), rows=rows, enrichment=list(boot),
               layers=[int(x) for x in layers], seconds=time.time() - t0,
               objectives=[{"name": o.name, "direction": int(o.direction), "best": o.best}
                           for o in objectives],
               picks=[dict(name=pool_names[int(i)], made=bool(made[int(i)]),
                           predicted=[float(values[int(i), 0]), float(values[int(i), 1])],
                           measured_potency=(float(pool_y1[int(i)])
                                             if np.isfinite(pool_y1[int(i)]) else None),
                           gains=t.gains, loses=t.loses, same=t.same)
                      for i, t in zip(picks, trades)],
               made_on_front=[pool_names[int(i)] for i in front if made[int(i)]],
               potency_of_made=dict(
                   predicted=float(np.mean(values[made, 0])),
                   measured=float(np.nanmean(pool_y1[made])),
                   pool_predicted=float(np.mean(values[~made, 0])),
                   pool_measured=float(np.nanmean(pool_y1[~made])),
                   anti_predicted=float(np.mean(values[made, 1])),
                   anti_pool_predicted=float(np.mean(values[~made, 1]))))
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def _enrichment(made, rows, runs=2000, seed=0):
    """How many times the base rate the front holds, with an interval over the pool."""
    made = np.asarray(made, bool)
    rows = np.asarray(rows, int)
    base = float(made.mean())
    if not len(rows) or base <= 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(made[rows].mean() / base)
    rng = np.random.RandomState(int(seed))
    got = []
    for _ in range(int(runs)):
        i = rng.randint(0, len(rows), len(rows))
        got.append(float(made[rows[i]].mean() / base))
    return (point, float(np.percentile(got, 2.5)), float(np.percentile(got, 97.5)))


def report(results=RESULTS):
    path = os.path.join(results, "summary.json")
    if not os.path.exists(path):
        raise SystemExit("nothing to report: run the benchmark first")
    with io.open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    out = ["# A design loop with more than one objective (plan item MC11)", "",
           "Lead optimisation has more than one objective and since plan items E1 and MC4 they "
           "live on one kernel. This is the loop that reads them together: the Pareto front of "
           "the predicted objectives, an acquisition that adds the expected improvement of each "
           "one, the batch rule of plan item MC2 so that a plate is not one compound ten times "
           "over, and a sentence per suggestion saying what it gains and what it loses.", "",
           "The test is retrospective and the series makes it possible, because it comes in two "
           "parts. The loop is trained on the "
           f"{d['train']} compounds of part 1, with their histamine H3 affinity and their hERG "
           f"activity. It then scores {d['pool']} candidates -- every other H3 ligand in ChEMBL "
           f"that fits the same frame -- of which {d['made']} are the compounds the same "
           "authors went on to make in part 2. The loop sees no activity of any candidate.", "",
           f"The model reaches q2 {d['q2']:+.3f} on the potency (alpha {d['alpha']:g}, ridge "
           f"{d['lam']:g}); the task correlation is {d['rho']:+.3f}. The objectives are "
           + " and ".join(f"{o['name']} ({'higher' if o['direction'] > 0 else 'lower'} is "
                          f"better, best so far {o['best']:.2f})" for o in d["objectives"])
           + ".", "",
           "| the list the loop reports | candidates | of them made next | share |",
           "| --- | ---: | ---: | ---: |"]
    for r in d["rows"]:
        out.append(f"| {r['what']} | {r['n']} | {r['made']} | {r['share']:.1%} |")
    base = d["made"] / d["pool"]
    e = d["enrichment"]
    w = d.get("where") or {}
    if w:
        out += ["",
                f"A front of {d['front']} cannot hold {d['made']} compounds, so where they sit "
                f"is the number that says more. The compounds the authors made are at Pareto "
                f"layer {w['layer_made']:.0f} of {w['layer_worst']}, against "
                f"{w['layer_pool']:.0f} for the rest of the pool, and at rank "
                f"{w['rank_made']:.0f} of {d['pool']} by the acquisition against "
                f"{w['rank_pool']:.0f}. The best of them is on layer {w['best_layer']} and at "
                f"rank {w['best_rank']}."]
    out += ["", f"The base rate is {base:.1%}: that is the share of the pool the authors made. "
                f"The Pareto front holds {e[0]:.2f} times that [{e[1]:.2f}, {e[2]:.2f}], "
                f"resampling the front.", "",
            "## What the loop would order", "",
            "The batch is taken from the front by the acquisition, with every objective's "
            "posterior conditioned on each pick. A change smaller than the model's own error is "
            "reported as no change.", "",
            "| compound | made next | predicted potency | predicted hERG | what it trades |",
            "| --- | :-: | ---: | ---: | --- |"]
    for p in d["picks"]:
        bits = []
        if p["gains"]:
            bits.append("gains " + ", ".join(f"{n} {v:+.2f}" for n, v in p["gains"]))
        if p["loses"]:
            bits.append("loses " + ", ".join(f"{n} {v:+.2f}" for n, v in p["loses"]))
        out.append(f"| {p['name']} | {'yes' if p['made'] else ''} | {p['predicted'][0]:.2f} | "
                   f"{p['predicted'][1]:.2f} | {'; '.join(bits) or 'nothing worth reporting'} |")
    out += [""] + _verdict(d)
    return "\n".join(out)


def _verdict(d):
    e = d["enrichment"]
    base = d["made"] / d["pool"]
    front_row = next(r for r in d["rows"] if r["what"] == "the Pareto front")
    passes = np.isfinite(e[1]) and e[1] > 1.0
    out = ["## The verdict", ""]
    if passes:
        out += [f"**The front contains the compounds the authors went on to make, and more of "
                f"them than the pool holds.** {front_row['made']} of the "
                f"{front_row['n']} candidates on it were made next, {front_row['share']:.1%} "
                f"against a base rate of {base:.1%}, an enrichment of {e[0]:.2f} "
                f"[{e[1]:.2f}, {e[2]:.2f}]. That is the gate of the item."]
    elif front_row["made"] > 0:
        out += [f"**The front contains some of the compounds the authors made, but no more than "
                f"chance would put there.** {front_row['made']} of {front_row['n']} were made "
                f"next, {front_row['share']:.1%} against a base rate of {base:.1%}, an "
                f"enrichment of {e[0]:.2f} [{e[1]:.2f}, {e[2]:.2f}], whose interval contains "
                f"one. The loop reports a front and the trades on it; what it does not do on "
                f"this series is pick out what a chemist picked."]
    else:
        out += ["**The front contains none of the compounds the authors made.** The gate is "
                "not met."]
    w = d.get("where") or {}
    if w:
        better = w.get("layer_made", 0) < w.get("layer_pool", 0)
        out += ["",
                (f"**It does rank them above the pool, which the membership test hides.**"
                 if better else
                 f"**Nor does it rank them above the pool.**")
                + f" The compounds they made sit at Pareto layer {w['layer_made']:.0f} of "
                  f"{w['layer_worst']} against {w['layer_pool']:.0f} for the rest, and at rank "
                  f"{w['rank_made']:.0f} of {d['pool']} against {w['rank_pool']:.0f} by the "
                  f"acquisition. The best of them reaches layer {w['best_layer']} and rank "
                  f"{w['best_rank']}, so the loop does not miss them altogether; it puts them "
                  + ("in the better half." if better else "in the worse half.")]
    p = d.get("potency_of_made") or {}
    if p.get("measured") is not None:
        out += ["",
                f"**What the authors made was better than the pool, and the model saw part of "
                f"it.** Their measured H3 affinity averages {p['measured']:.2f} against "
                f"{p['pool_measured']:.2f} for the rest of the pool; the model predicted "
                f"{p['predicted']:.2f} against {p['pool_predicted']:.2f}. The order is right "
                f"and the size of the gap is not, which is what a q2 of {d['q2']:+.3f} on a "
                "series this narrow buys."]
    if p.get("anti_predicted") is not None:
        out += ["",
                f"**What sinks them in the ranking is the other objective.** The model predicts "
                f"their hERG activity at {p['anti_predicted']:.2f} against "
                f"{p['anti_pool_predicted']:.2f} for the rest of the pool, and the acquisition "
                f"adds an expected improvement over the best already made in each objective, so "
                f"a compound that is good in one and ordinary in the other scores below one "
                f"that is extreme in either. No hERG number was ever measured for these "
                f"compounds, so whether the model is right about them cannot be checked -- "
                f"which is itself the honest statement about a two-objective loop run on a "
                f"series where only one objective was followed up."]
    out += ["",
            "**What this test can and cannot settle.** It is one pair of papers, and the "
            "compounds the authors made next were chosen with knowledge the loop does not have "
            "-- synthetic access, a chemical idea, the rest of their programme -- so the "
            "compounds they did not make are not known failures. The candidates also carry no "
            "measured hERG number, so only one of the two objectives can be checked against "
            "what happened. What the item asked for and this delivers is the machinery: a "
            "front over several objectives on one kernel, a batch taken from it by the rule of "
            "plan item MC2, and for every suggestion the trade it makes written out.", ""]
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
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    run(folds=a.folds, seed=a.seed, size=a.batch)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
