#!/usr/bin/env python
"""
Would the design loop have suggested the substituent that was actually best?

`cmfqsar/design.py` enumerates replacements at one position, poses each by the topomer rule and
ranks them by what the model expects. The only honest test of that is retrospective and has to
hold out the answer: **leave one substituent out**. Every molecule carrying a given R-group at
the varied position is removed from the training set, the model is fitted on what is left, and
the whole library of R-groups -- including the one just removed -- is enumerated and ranked. The
rank the held-out group receives is then compared with the rank its measured activity deserves.

Two questions, kept apart:

* **Does the ranking agree with the measurements?** Spearman's rho between the predicted score of
  every held-out group and the activity of the molecule that carries it.
* **Would following it have helped?** The activity of the top-ranked suggestion, against the mean
  over the library (what picking at random gives) and against the best in the library (what an
  oracle gives) -- and how often the top suggestion lands in the best fifth.

Only sets whose scaffold comes off cleanly are usable: a core fused to the rest of the molecule
leaves something that is not a substituent, and the loop has nothing to vary. The ones that fail
that test are reported rather than dropped, since how far this reaches on public data is part of
the answer.

Usage
-----
    python benchmarks/run_design.py
    python benchmarks/run_design.py --sets GPB --min-cuttable 0.8
    python benchmarks/run_design.py --report
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

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.design import design, fragment_smiles  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.topomer import (TopomerSpec, attachment_label, choose_core, cuttable_fraction,  # noqa: E402
                             series_labels, substituents, topomer_fields)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "design")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.3, 0.5, 1.0, 2.0)
LAMS = (1e-2, 0.1, 1.0)
MIN_CUTTABLE = 0.75       # of the series, or the core is not an R-group core
MIN_GROUPS = 8            # distinct groups at the position, or there is nothing to rank
TOP = 0.2                 # the best fifth of the library


def _fit(mols, names, y, core, labels, spec):
    """A topomer model with alpha and lambda chosen by the evidence."""
    best = None
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=float(alpha))
        mf, _log = topomer_fields(mols, names, fs, core=core, labels=labels, spec=spec)
        keep = [i for i, f in enumerate(mf) if f is not None]
        if len(keep) < 5:
            continue
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        table = AtomTable.from_molecules([mf[i] for i in keep])
        K = model.joint_kernel(model.prepare_training_kernels(table))
        for lam in LAMS:
            ev = log_evidence(K, np.asarray(y)[keep], float(lam), fit_bias=True)
            if np.isfinite(ev) and (best is None or ev > best[0]):
                best = (ev, float(alpha), float(lam), fs, table, keep)
    if best is None:
        return None
    _ev, alpha, lam, fs, table, keep = best
    model = CMFModel(fieldset=fs, method="krr", params={"lam": lam}, normalize=True)
    model.fit(table, np.asarray(y)[keep], cv_folds=10, cv_seed=0)
    return model, alpha, lam, keep


def run_set(name, force=False, data=DATA, results=RESULTS, min_cuttable=MIN_CUTTABLE,
            acquire="greedy"):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(os.path.join(data, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y = y[ok]
    core, how = choose_core(mols, mode="framework")
    if core is None:
        out = {"set": name, "usable": False, "why": "no scaffold found"}
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"=== {name}: no scaffold")
        return out
    share = cuttable_fraction(mols, core)
    labels = series_labels(mols, core)
    spec = TopomerSpec()

    # the position that varies most, and what the series puts there
    at_site = {}
    for i, mol in enumerate(mols):
        for frag in substituents(mol, core):
            # by the canonical name, so that the same group written with explicit hydrogens
            # in the file and without them in a library is one group
            at_site.setdefault(attachment_label(frag), {}).setdefault(
                fragment_smiles(frag), []).append(i)
    label = max(at_site, key=lambda k: len(at_site[k])) if at_site else None
    groups = at_site.get(label, {})
    if share < min_cuttable or label is None or len(groups) < MIN_GROUPS:
        why = (f"the core comes off cleanly in only {share:.0%} of the series"
               if share < min_cuttable else
               f"only {len(groups)} distinct groups at the busiest position")
        out = {"set": name, "usable": False, "why": why, "cuttable": float(share),
               "core": Chem.MolToSmiles(core), "groups": len(groups)}
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"=== {name}: not usable -- {why}")
        return out

    print(f"=== {name}: core {Chem.MolToSmiles(core)} ({how}), cut cleanly in {share:.0%}, "
          f"position {label} with {len(groups)} groups", flush=True)
    library = sorted(groups)
    # the activity a group is credited with: the mean of the molecules carrying it
    truth = {g: float(np.mean(y[idx])) for g, idx in groups.items()}

    # the parent is chosen once and kept: it is the template every candidate is built on, its
    # group at the varied position is replaced in all of them, and a parent that changed between
    # folds would put the held-out groups on different baselines
    whole = _fit(mols, names, y, core, labels, spec)
    if whole is None:
        out = {"set": name, "usable": False, "why": "nothing could be posed"}
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"=== {name}: nothing could be posed")
        return out
    poseable = [i for i in whole[3]]
    parent = poseable[int(np.argmax(y[poseable]))]
    print(f"  parent: {names[parent]} ({y[parent]:.2f}), kept for every fold", flush=True)

    rows = []
    for held in library:
        drop = set(groups[held])
        tr = [i for i in range(len(mols)) if i not in drop]
        if len(tr) < 15:
            continue
        fit = _fit([mols[i] for i in tr], [names[i] for i in tr], y[tr], core, labels, spec)
        if fit is None:
            continue
        model, alpha, lam, keep = fit
        p = parent
        try:
            res = design(model, mols[p], core, label, fragments=library, labels=labels,
                         acquire=acquire, parent_name=names[p])
        except Exception as exc:                 # noqa: BLE001 - one parent, not the set
            print(f"  {held[:28]:28s} skipped ({exc})", flush=True)
            continue
        order = [c.smiles for c in res.candidates]
        scores = {c.smiles: c.mean for c in res.candidates}
        if held not in scores:
            continue
        rows.append({"held": held, "rank": order.index(held) + 1, "n_library": len(order),
                     "predicted": scores[held], "measured": truth[held],
                     "top": order[0], "top_measured": truth.get(order[0], float("nan")),
                     "q2": float(model.stats.q2), "alpha": alpha, "lam": lam,
                     "parent": names[p], "n_train": len(keep)})
        print(f"  {held[:30]:30s} measured {truth[held]:6.2f}   ranked "
              f"{rows[-1]['rank']:3d}/{len(order)}   top suggestion {order[0][:24]:24s} "
              f"({rows[-1]['top_measured']:.2f})", flush=True)

    out = {"set": name, "usable": True, "n": len(mols), "core": Chem.MolToSmiles(core),
           "cuttable": float(share), "label": int(label), "library": len(library),
           "acquire": acquire, "truth": truth, "rows": rows,
           "mean_activity": float(np.mean(list(truth.values()))),
           "best_activity": float(np.max(list(truth.values()))),
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def report(results=RESULTS):
    import glob
    from scipy.stats import spearmanr

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No design results yet; run `python benchmarks/run_design.py`."
    data = [json.load(open(f)) for f in files]
    usable = [d for d in data if d.get("usable") and d.get("rows")]
    skipped = [d for d in data if not d.get("usable")]
    lines = ["# Would the design loop have suggested the right substituent", "",
             "Leave one substituent out: every molecule carrying a given R-group at the varied "
             "position is removed, the topomer model is refitted on what is left, and the whole "
             "library of groups is enumerated and ranked. The group's measured activity is the "
             "mean over the molecules that carry it.", "",
             "| set | groups | position | q² of the models | ρ of ranking with measurement | "
             "top suggestion | library mean | library best |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    rhos, lifts, hits = [], [], []
    for d in sorted(usable, key=lambda r: -len(r["rows"])):
        rows = d["rows"]
        pred = [r["predicted"] for r in rows]
        meas = [r["measured"] for r in rows]
        rho = float(spearmanr(pred, meas).statistic) if len(rows) > 3 else float("nan")
        top = float(np.mean([r["top_measured"] for r in rows
                             if np.isfinite(r["top_measured"])]))
        q2 = float(np.mean([r["q2"] for r in rows]))
        rhos.append(rho)
        lifts.append(top - d["mean_activity"])
        cut = np.quantile(list(d["truth"].values()), 1.0 - TOP)
        hits += [bool(r["top_measured"] >= cut) for r in rows if np.isfinite(r["top_measured"])]
        lines.append(f"| {d['set']} | {len(rows)} of {d['library']} | {d['label']} | {q2:+.3f} | "
                     f"{rho:+.3f} | {top:.2f} | {d['mean_activity']:.2f} | "
                     f"{d['best_activity']:.2f} |")
    if rhos:
        lines.append(f"| **mean** |  |  |  | **{float(np.nanmean(rhos)):+.3f}** |  |  |  |")
    lines += ["",
              f"Over {sum(len(d['rows']) for d in usable)} held-out groups on "
              f"{len(usable)} sets, the ranking agrees with the measurements at a mean Spearman "
              f"ρ of **{float(np.nanmean(rhos)):+.3f}**, and the group the loop puts first is "
              f"worth **{float(np.mean(lifts)):+.2f}** log units more than the library average "
              f"-- it lands in the best {100 * TOP:.0f} % of the library "
              f"**{100 * float(np.mean(hits)):.0f} %** of the time.", ""]
    if skipped:
        lines += ["Sets the loop cannot reach, and why:", "",
                  "| set | why |", "|---|---|"]
        for d in skipped:
            lines.append(f"| {d['set']} | {d.get('why', 'unusable')} |")
        lines += ["",
                  "That is half the answer to how far this goes. A design loop over substituents "
                  "needs a series with a scaffold that comes off at single acyclic bonds, and "
                  "most public QSAR sets are not that: they are collections of several "
                  "chemotypes with a common framework found by force. On the sets that are "
                  "congeneric, the loop works; on the others there is nothing to vary and the "
                  "honest answer is to say so rather than to rank whatever the core leaves "
                  "behind."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--min-cuttable", type=float, default=MIN_CUTTABLE)
    ap.add_argument("--acquire", default="greedy",
                    help="what to rank by; greedy (the prediction) is the right default here, "
                         "since the question is which group is best and not which to try next")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    for name in a.sets:
        run_set(name, force=a.force, min_cuttable=a.min_cuttable, acquire=a.acquire)
    print()
    print(report())


if __name__ == "__main__":
    main()
