#!/usr/bin/env python
"""
Does a masked co-field recognise a scaffold the model has never seen? (plan item MC5)

Item E3 measured what scaffold hopping lacks: the topomer frame is defined against a scaffold
that is held fixed, so replacing the scaffold leaves no frame, and comparing two whole molecules
answers mostly with how similar their skeletons are. MC5 proposes comparing them only where the
model rewards field -- multiply the co-field by a soft mask on the atoms that contribute most,
and score a candidate by its overlap with that (`cmfqsar.maskedmatch`).

The question is whether the mask helps. A model is fitted on **one chemotype** of a series, and
the molecules of every *other* chemotype that are at least as active have to be found. Nothing of
those scaffolds is in the training set, which is what a hop is.

Two backgrounds, because the first one decides nothing
------------------------------------------------------
The gate MC5 states is the hops against the posed presumed inactives of `examples/decoys/`.
Measured, that task is at its ceiling before the mask is applied: plain field similarity to the
best-matching training molecule reaches an AUC of 0.98 to 1.00 on all four sets, with the
property-matched decoys as well as the plain ones. Those molecules come from other targets and
differ grossly from the actives, which is the artefact the manual names in *One-class
classification and virtual screening*. The task is reported here because the item asks for it,
and it cannot distinguish two measures that both separate the two groups completely.

The second background is the one that can. The negatives are the **weakly active molecules of the
same other chemotypes**: real molecules, measured against the same target, in the same frame, of
the same kind of size and chemistry as the hops. Nothing separates them from the hops but the
activity, so a ranking has to be about the activity. This is the harder question and the one a
chemist actually asks: among compounds of a scaffold I have never measured, which are the active
ones?

The protocol, per set
---------------------
1. Chemotypes from `cmfqsar.chemotypes.chemotype_labels` (Bemis-Murcko scaffolds, groups of three
   or more). The training scaffold is the **largest** real group -- the series one has most data
   on -- and the merged bucket of singletons is never chosen.
2. The model is fitted on that group alone, the attenuation and the ridge chosen by
   cross-validation inside it. Its q2 is reported in every table, because a model that does not
   predict has no co-field worth masking and its ranking means nothing.
3. **Hops** are the molecules outside the training group whose activity is at or above that
   group's median. **Weak** are the molecules outside it in the lowest third of the set's
   activities. **Decoys** are the 100 property-matched molecules of
   `examples/decoys/<set>_decoys_matched.sdf`.
4. Four rankings on each background: the masked cosine of MC5, the masked overlap itself, the
   unmasked field similarity to the best-matching training molecule (what the program had
   before), and the model's own prediction.

Everything is in the frame the files carry; nothing here poses anything.

Usage
-----
    python benchmarks/run_maskedmatch.py
    python benchmarks/run_maskedmatch.py --sets AChE --mask-atoms 5 10 20
    python benchmarks/run_maskedmatch.py --report
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

from cmfqsar.chemotypes import chemotype_labels  # noqa: E402
from cmfqsar.cofields import top_atom_mask  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.decoys import DecoyAligner  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.maskedmatch import MaskedScorer  # noqa: E402
from cmfqsar.model import CMFModel, grid_search, screening_scores  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "maskedmatch")
DATA = os.path.join(ROOT, "examples")
#: the sets that have both a chemotype structure and a posed background set
SETS = ("AChE", "THR", "ACE", "GPB")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0)
MASK_ATOMS = (5, 10, 20)
WEAK_QUANTILE = 1.0 / 3.0            # "weakly active" is the lowest third of the set
TASKS = ("hops against the weak", "hops against the decoys")
SCORES = ("masked cosine", "masked overlap", "field similarity", "prediction")


def activity_property(path: str) -> str:
    mol = next(m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None)
    for cand in ("pIC50", "pKi", "pEC50", "activity"):
        if mol.HasProp(cand):
            return cand
    raise SystemExit(f"{path}: no activity property this script knows")


def training_group(labels: np.ndarray, scaffolds) -> int:
    """The largest real chemotype; the merged bucket of singletons is not a scaffold."""
    sizes = [(int((labels == g).sum()), g) for g in range(int(labels.max()) + 1)
             if "small groups" not in scaffolds[g]]
    if not sizes:
        raise SystemExit("no chemotype of three molecules or more")
    return max(sizes)[1]


def _metrics(truth, score):
    return {k: float(v) for k, v in screening_scores(truth, score).items()}


def run_set(name, mask_atoms=MASK_ATOMS, force=False, data=DATA, results=RESULTS):
    out = os.path.join(results, f"{name}.json")
    if os.path.exists(out) and not force:
        print(f"    {name}: already there ({out})")
        return json.load(open(out, encoding="utf-8"))
    t0 = time.time()
    path = os.path.join(data, "sutherland", f"{name}.sdf")
    prop = activity_property(path)
    ds = load_dataset(path, activity_prop=prop)
    y = np.asarray(ds.activity, float)
    labels, scaffolds = chemotype_labels(list(ds.mols), min_size=3)
    g = training_group(labels, scaffolds)
    train = np.where(labels == g)[0]
    threshold = float(np.median(y[train]))
    weak_cut = float(np.quantile(y, WEAK_QUANTILE))
    hops = np.array([i for i in range(len(y)) if labels[i] != g and y[i] >= threshold], dtype=int)
    weak = np.array([i for i in range(len(y)) if labels[i] != g and y[i] <= weak_cut], dtype=int)
    print(f"=== {name}: {len(ds)} molecules, '{prop}'; training chemotype {g} "
          f"(n = {len(train)}, {scaffolds[g][:40]})")
    print(f"    {len(hops)} hops at or above {threshold:.2f}, {len(weak)} weak at or below "
          f"{weak_cut:.2f}")

    fs = FieldSet.create(physchem=FIELDS, alpha=float(ALPHAS[1]), charge_model="file")
    mf_train = [build_fields(ds.mols[i], fs, ds.names[i]) for i in train]
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    table = AtomTable.from_molecules(mf_train)
    grid_search(table, y[train], model, alphas=list(ALPHAS), lams=list(LAMS))
    model.fit(table, y[train], cv_folds=min(10, len(train)), reuse_kernels=True)
    print(f"    the model on that scaffold alone: alpha {model.fieldset.specs[0].alpha:g}, "
          f"lambda {model.params.get('lam'):g}, q2 {model.stats.q2:+.3f}")

    # the three groups of molecules, built once
    def fields_of(idx):
        return [build_fields(ds.mols[i], fs, ds.names[i]) for i in idx]

    mf_hops, mf_weak = fields_of(hops), fields_of(weak)
    mf_dec, dec_names = [], []
    dpath = os.path.join(data, "decoys", f"{name}_decoys_matched.sdf")
    for k, m in enumerate(Chem.SDMolSupplier(dpath, removeHs=False)):
        if m is None:
            continue
        n = m.GetProp("_Name") if m.HasProp("_Name") else f"decoy_{k + 1}"
        mf_dec.append(build_fields(m, fs, n))
        dec_names.append(n)
    print(f"    {len(mf_dec)} property-matched decoys")

    aligner = DecoyAligner(table, fs)

    def unmasked(mols):
        sim = np.array([float(aligner.similarities(m, m.coords)[1].max()) for m in mols])
        pred = np.asarray(model.predict(AtomTable.from_molecules(mols)), float)
        return sim, pred

    sim_h, pred_h = unmasked(mf_hops)
    sim_w, pred_w = unmasked(mf_weak)
    sim_d, pred_d = unmasked(mf_dec)

    record = {"set": name, "activity": prop, "n": int(len(ds)), "chemotype": int(g),
              "scaffold": scaffolds[g], "n_train": int(len(train)), "threshold": threshold,
              "weak_cut": weak_cut, "n_hops": int(len(hops)), "n_weak": int(len(weak)),
              "n_decoys": int(len(mf_dec)), "alpha": float(model.fieldset.specs[0].alpha),
              "lam": float(model.params.get("lam")), "q2": float(model.stats.q2),
              "tasks": {}}
    for task, (sim_n, pred_n, n_neg) in (
            ("hops against the weak", (sim_w, pred_w, len(mf_weak))),
            ("hops against the decoys", (sim_d, pred_d, len(mf_dec)))):
        if not n_neg or not len(mf_hops):
            continue
        truth = np.r_[np.ones(len(mf_hops)), -np.ones(n_neg)]
        record["tasks"][task] = {
            "n_positive": int(len(mf_hops)), "n_negative": int(n_neg),
            "field similarity": _metrics(truth, np.r_[sim_h, sim_n]),
            "prediction": _metrics(truth, np.r_[pred_h, pred_n]),
            "masks": {}}

    for k in mask_atoms:
        mask = top_atom_mask(model, mf_train, k=int(k))
        scorer = MaskedScorer(model, mask)
        sc_h = [scorer.score(m) for m in mf_hops]
        sc_w = [scorer.score(m) for m in mf_weak]
        sc_d = [scorer.score(m) for m in mf_dec]
        for task, sc_n in (("hops against the weak", sc_w),
                           ("hops against the decoys", sc_d)):
            if task not in record["tasks"]:
                continue
            truth = np.r_[np.ones(len(sc_h)), -np.ones(len(sc_n))]
            cos = np.array([s.cosine for s in sc_h + sc_n])
            ins = np.array([s.inside for s in sc_h + sc_n])
            record["tasks"][task]["masks"][str(k)] = {
                "mask_atoms": int(k), "mask_width": float(np.mean(mask.widths)),
                "masked cosine": _metrics(truth, cos),
                "masked overlap": _metrics(truth, ins),
                "inside_positive": float(np.mean([s.inside for s in sc_h])),
                "inside_negative": float(np.mean([s.inside for s in sc_n])),
                "outside_positive": float(np.mean([s.outside for s in sc_h])),
                "outside_negative": float(np.mean([s.outside for s in sc_n]))}
        a = record["tasks"].get("hops against the weak", {}).get("masks", {}).get(str(k))
        if a:
            print(f"    mask of {k:2d} atoms, hops against the weak: masked cosine AUC "
                  f"{a['masked cosine']['auc']:.3f}, masked overlap "
                  f"{a['masked overlap']['auc']:.3f}, field similarity "
                  f"{record['tasks']['hops against the weak']['field similarity']['auc']:.3f}, "
                  f"prediction "
                  f"{record['tasks']['hops against the weak']['prediction']['auc']:.3f}")
    record["seconds"] = time.time() - t0
    os.makedirs(results, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=1)
        fh.write("\n")
    print(f"    written to {out}  ({record['seconds']:.0f} s)")
    return record


def _auc(row, task, score, k):
    t = row["tasks"].get(task)
    if t is None:
        return None
    if score in ("field similarity", "prediction"):
        return t[score]["auc"]
    m = t["masks"].get(str(k))
    return None if m is None else m[score]["auc"]


def report(results=RESULTS, out=None):
    rows = [json.load(open(os.path.join(results, f"{n}.json"), encoding="utf-8"))
            for n in SETS if os.path.exists(os.path.join(results, f"{n}.json"))]
    if not rows:
        raise SystemExit("no results yet: run python benchmarks/run_maskedmatch.py")
    sizes = sorted({int(k) for r in rows for t in r["tasks"].values() for k in t["masks"]})
    main = sizes[len(sizes) // 2] if sizes else 10
    lines = ["# Does a masked co-field recognise a scaffold the model has never seen?", "",
             "A model is fitted on one chemotype of a series. The molecules of every other "
             "chemotype that are at least as active have to be ranked above a background. "
             "Nothing of those scaffolds is in the training set. AUC of the ranking; q2 is the "
             "model's own cross-validation on the one chemotype it was fitted on.", "",
             f"Mask of {main} atoms; every size is in the last table.", ""]
    for task in TASKS:
        lines += [f"## {task.capitalize()}", ""]
        if task == "hops against the decoys":
            lines += ["The background is the 100 property-matched presumed inactives. This is "
                      "the gate item MC5 states, and it decides nothing: the unmasked field "
                      "similarity is already at its ceiling, because molecules from other "
                      "targets differ from the actives in size and shape and not only in "
                      "activity.", ""]
        else:
            lines += ["The background is the weakly active molecules of the same other "
                      "chemotypes -- real molecules of the same target in the same frame, so "
                      "nothing separates them from the hops but the activity.", ""]
        lines += ["| set | training chemotype | n train | q2 | positives | negatives | "
                  "masked cosine | masked overlap | field similarity | prediction |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        got = []
        for r in rows:
            t = r["tasks"].get(task)
            if t is None:
                continue
            vals = [_auc(r, task, s, main) for s in SCORES]
            got.append(vals)
            cells = " | ".join("-" if v is None else f"{v:.3f}" for v in vals)
            lines.append(f"| {r['set']} | `{r['scaffold'][:32]}` | {r['n_train']} | "
                         f"{r['q2']:+.3f} | {t['n_positive']} | {t['n_negative']} | {cells} |")
        if got:
            arr = np.array([[np.nan if v is None else v for v in row] for row in got], float)
            mean = " | ".join(f"{v:.3f}" for v in np.nanmean(arr, axis=0))
            lines.append(f"| **mean** | | | | | | {mean} |")
            gain = arr[:, 0] - arr[:, 2]
            lines += ["", f"The masked cosine is {np.nanmean(gain):+.3f} AUC against the "
                          f"unmasked field similarity, ahead on "
                          f"{int(np.sum(gain > 0))} of {len(gain)} sets.", ""]

    lines += ["## How much of the prediction comes from inside the mask", "",
              "The second score the item asks for: a candidate whose promise is all outside the "
              "region the model rewards is one the model is extrapolating about. Mean over the "
              "molecules of each kind, in the units of the activity, on the harder task.", "",
              "| set | mask atoms | inside, hops | inside, weak | outside, hops | outside, weak |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        t = r["tasks"].get("hops against the weak")
        if t is None:
            continue
        for k in sorted(t["masks"], key=lambda s: int(s)):
            m = t["masks"][k]
            lines.append(f"| {r['set']} | {k} | {m['inside_positive']:+.3f} | "
                         f"{m['inside_negative']:+.3f} | {m['outside_positive']:+.3f} | "
                         f"{m['outside_negative']:+.3f} |")

    lines += ["", "## Every mask size, on the harder task", "",
              "| set | mask atoms | masked cosine | masked overlap | field similarity | "
              "prediction |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        t = r["tasks"].get("hops against the weak")
        if t is None:
            continue
        for k in sorted(t["masks"], key=lambda s: int(s)):
            m = t["masks"][k]
            lines.append(f"| {r['set']} | {k} | {m['masked cosine']['auc']:.3f} | "
                         f"{m['masked overlap']['auc']:.3f} | "
                         f"{t['field similarity']['auc']:.3f} | {t['prediction']['auc']:.3f} |")
    lines.append("")
    text = "\n".join(lines) + "\n"
    out = out or os.path.join(results, "table.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text)
    print(f"written to {out}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--mask-atoms", nargs="+", type=int, default=list(MASK_ATOMS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return 0
    for name in args.sets:
        run_set(name, mask_atoms=args.mask_atoms, force=args.force)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
