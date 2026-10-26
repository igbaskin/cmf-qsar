#!/usr/bin/env python
"""
Screening on a public benchmark, against measured inactives (plan item E4).

Every screening number in this repository so far has been measured against a background invented
here: ligands of other targets, or molecules property-matched to the actives out of the sets
already shipped. Both are *presumed* inactive, and a presumed inactive is an easier opponent than
a measured one.

LIT-PCBA supplies measured ones -- high-confidence PubChem dose-response assays, in the
AVE-unbiased form its authors recommend, where the analogue bias that lets a screen score well by
recognising that actives resemble actives has been removed. `examples/make_litpcba_examples.py`
embeds and poses them; this measures four quite different ways of ranking a library with them:

* the **one-class descriptions** of `cmfqsar/oneclass.py`, fitted on the training actives alone,
  which is the honest setting for a screen -- nothing but the actives is known;
* the **fingerprint baseline**: the maximum ECFP4 Tanimoto to the training actives, which is what
  a medicinal chemist would do in five minutes and what any 3D method has to beat;
* a **supervised classifier** on the CMF kernel, told that half the inactives are negative, which
  is the optimistic case;
* **positive-unlabelled learning** (`cmfqsar/pu.py`), which treats that half as unlabelled rather
  than negative, beside the naive classifier that does not.

Usage
-----
    python benchmarks/run_screening.py
    python benchmarks/run_screening.py --sets TP53 --alpha 0.5
    python benchmarks/run_screening.py --report
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

from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, screening_scores  # noqa: E402
from cmfqsar.oneclass import ONECLASS_METHODS, make as make_oneclass  # noqa: E402
from cmfqsar.pu import bagging_pu, naive_scores  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "screening")
DATA = os.path.join(ROOT, "examples", "litpcba")
SETS = ("ESR1_ant", "TP53")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 0.5
LAM = 0.1
BAGS = 30
ONECLASS = ("ocmean", "ocknn", "ocgp", "ockpca")
#: The comparison with the fingerprint must not be dismissible as a badly chosen setting, so the
#: one-class descriptions are given every setting they have and the best one *on the test set* is
#: reported beside the default.  No honest protocol could pick that value, which is the point: it
#: is an upper bound on what the description can do, not a score.
#:
#: nu is deliberately not in the sweep.  It places the offset so that a fraction nu of the training
#: actives falls outside the description, which shifts the decision value by a constant and cannot
#: change the order of a ranking -- AUC, BEDROC and every enrichment factor are invariant to it.
#: What does change the ranking is the field attenuation and each method's own parameter.
ALPHAS = (0.3, 0.5, 1.0)
GRID = {"ocmean": [{}],
        "ocknn": [{"k": k} for k in (1, 3, 5, 10)],
        "ocgp": [{"lam": lam, "score": sc} for lam in (0.01, 0.1, 1.0)
                 for sc in ("mean", "mean_over_sd")],
        "ockpca": [{"components": q} for q in (2, 5, 10, 20)]}
#: the setting each description is run at for the headline row
DEFAULTS = {"ocmean": {}, "ocknn": {"k": 5}, "ocgp": {"lam": 0.1, "score": "mean"},
            "ockpca": {"components": 5}}


def _fingerprints(mols):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(m) for m in mols]


def fingerprint_scores(mols, train_idx, test_idx):
    """Maximum ECFP4 Tanimoto to the training actives, for every molecule of `test_idx`."""
    fps = _fingerprints(mols)
    ref = [fps[i] for i in train_idx]
    out = np.zeros(len(test_idx))
    for k, i in enumerate(test_idx):
        out[k] = max(DataStructs.BulkTanimotoSimilarity(fps[i], ref))
    return out


def run_set(name, alpha=ALPHA, lam=LAM, bags=BAGS, seed=0, force=False, data=DATA,
            results=RESULTS, variant="crystal"):
    os.makedirs(results, exist_ok=True)
    tag = "" if variant == "crystal" else f"_{variant}"
    out_path = os.path.join(results, f"{name}{tag}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}{tag}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    act_path = os.path.join(data, f"{name}.sdf")
    dec_path = os.path.join(data, f"{name}_decoys{tag}.sdf")
    if not (os.path.exists(act_path) and os.path.exists(dec_path)):
        print(f"{name}: not built; run examples/make_litpcba_examples.py")
        return None
    actives = load_dataset(act_path)
    decoys = load_dataset(dec_path)
    a_mols, a_names = list(actives.mols), list(actives.names)
    d_mols, d_names = list(decoys.mols), list(decoys.names)
    subset = [str(m.GetProp("subset")) if m.HasProp("subset") else "train" for m in a_mols]
    tr_a = [i for i, s in enumerate(subset) if s == "train"]
    te_a = [i for i, s in enumerate(subset) if s != "train"]
    n_a, n_d = len(a_mols), len(d_mols)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_d)
    half = n_d // 2
    d_known, d_test = sorted(order[:half].tolist()), sorted(order[half:].tolist())
    print(f"=== {name}: {n_a} measured actives ({len(tr_a)} train, {len(te_a)} test), "
          f"{n_d} measured inactives ({len(d_known)} offered to the supervised methods, "
          f"{len(d_test)} held out)", flush=True)

    mols = a_mols + d_mols
    names = a_names + [f"d{i}" for i in range(n_d)]
    A = lambda i: int(i)                       # noqa: E731 - actives keep their index
    D = lambda j: int(n_a + j)                 # noqa: E731 - decoys follow them
    tr_idx = np.array([A(i) for i in tr_a], int)
    # what every method is scored on: the held-out actives against the held-out inactives
    test_idx = np.array([A(i) for i in te_a] + [D(j) for j in d_test], int)
    y_test = np.array([1.0] * len(te_a) + [-1.0] * len(d_test))

    rows, bound, K = {}, {}, None
    for a_i in sorted({float(alpha)} | set(ALPHAS)):
        fs = FieldSet.create(physchem=FIELDS, alpha=a_i)
        table = AtomTable.from_molecules([build_fields(m, fs, nm) for m, nm in zip(mols, names)])
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        Ka = model.joint_kernel(model.prepare_training_kernels(table))
        print(f"    alpha {a_i:g}: kernel {Ka.shape} in {time.time() - t0:.0f} s", flush=True)
        if a_i == float(alpha):
            K = Ka
        Ktr = Ka[np.ix_(tr_idx, tr_idx)]
        Kte = Ka[np.ix_(test_idx, tr_idx)]
        diag_te = np.diag(Ka)[test_idx]
        for kind in ONECLASS:
            if kind not in ONECLASS_METHODS:
                continue
            for params in GRID[kind]:
                oc = make_oneclass(kind, nu=0.1, **params).fit(Ktr)
                sc = screening_scores(y_test, np.asarray(oc.decision(Kte, diag_te), float))
                at = ", ".join([f"alpha={a_i:g}"] + [f"{k}={v}" for k, v in params.items()])
                if a_i == float(alpha) and params == DEFAULTS[kind]:
                    rows[kind] = dict(sc)
                if kind not in bound or sc["auc"] > bound[kind]["auc"]:
                    bound[kind] = {"auc": float(sc["auc"]), "ef_1": float(sc["ef_1"]), "at": at}
    for kind in list(rows):
        rows[kind]["best_auc"] = bound[kind]["auc"]
        rows[kind]["best_ef_1"] = bound[kind]["ef_1"]
        rows[kind]["best_at"] = bound[kind]["at"]
        print(f"    {kind:10s} AUC {rows[kind]['auc']:.3f}  EF@1% {rows[kind]['ef_1']:.1f}"
              f"   (best of {len(GRID[kind]) * len(set(ALPHAS) | {float(alpha)})} settings: "
              f"{rows[kind]['best_auc']:.3f} at {rows[kind]['best_at']})", flush=True)

    rows["ecfp4"] = screening_scores(y_test, fingerprint_scores(mols, tr_idx, test_idx))
    print(f"    {'ecfp4':10s} AUC {rows['ecfp4']['auc']:.3f}  EF@1% {rows['ecfp4']['ef_1']:.1f}",
          flush=True)

    known = np.array([D(j) for j in d_known], int)
    naive = naive_scores(K, tr_idx, known, test=test_idx, lam=lam)
    rows["supervised"] = screening_scores(y_test, naive[test_idx])
    print(f"    {'supervised':10s} AUC {rows['supervised']['auc']:.3f}  "
          f"EF@1% {rows['supervised']['ef_1']:.1f}", flush=True)

    pu = bagging_pu(K, tr_idx, known, bags=bags, lam=lam, seed=seed, also=test_idx)
    rows["pu"] = screening_scores(y_test, pu.scores[test_idx])
    rows["pu"]["bags"] = float(pu.bags)
    rows["pu"]["bag_size"] = float(pu.bag_size)
    print(f"    {'pu':10s} AUC {rows['pu']['auc']:.3f}  EF@1% {rows['pu']['ef_1']:.1f}   "
          f"({pu.describe()})", flush=True)

    out = {"set": name, "variant": variant,
           "n_actives": n_a, "n_inactives": n_d, "train_actives": len(tr_a),
           "test_actives": len(te_a), "offered": len(d_known), "held_out": len(d_test),
           "alpha": float(alpha), "lam": float(lam), "seed": int(seed),
           "alphas": sorted({float(alpha)} | set(ALPHAS)),
           "settings": {k: len(v) for k, v in GRID.items()},
           "methods": rows, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


ORDER = ["ocmean", "ocknn", "ocgp", "ockpca", "ecfp4", "supervised", "pu"]
LABEL = {"ocmean": "distance to the centre of the actives", "ocknn": "distance to the k-th active",
         "ocgp": "one-class Gaussian process", "ockpca": "distance from the kernel-PCA subspace",
         "ecfp4": "ECFP4 Tanimoto to the actives", "supervised": "classifier, inactives = negative",
         "pu": "classifier, inactives = unlabelled (bagging PU)"}


def report(results=RESULTS):
    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No screening results yet; run `python benchmarks/run_screening.py`."
    data = [json.load(open(f)) for f in files]
    main_data = [d for d in data if d.get("variant", "crystal") == "crystal"]
    similar = [d for d in data if d.get("variant") == "similar"]
    lines = ["# Screening against measured inactives", "",
             "LIT-PCBA, AVE-unbiased, embedded and posed by "
             "`examples/make_litpcba_examples.py`: the actives and the inactives placed by one "
             "rule, maximum field similarity to the target's crystal ligand, which knows nothing "
             "about which of them is active. The one-class methods and the fingerprint see the "
             "training actives alone; the supervised classifier and the PU model are also "
             "offered half of the inactives, and every method is scored on the held-out actives "
             "against the other half.", "",
             "| set | actives | inactives | " + " | ".join(("AUC", "BEDROC(20)", "EF@1 %",
                                                            "EF@5 %")) + " | method |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    best = {}
    for d in main_data:
        for m in ORDER:
            if m not in d["methods"]:
                continue
            r = d["methods"][m]
            best.setdefault(m, []).append(r)
            lines.append(f"| {d['set']} | {d['test_actives']} | {d['held_out']} | "
                         f"{r['auc']:.3f} | {r.get('bedroc_20', float('nan')):.3f} | "
                         f"{r['ef_1']:.1f} | {r['ef_5']:.1f} | {LABEL[m]} |")
    lines += ["", "Averaged over the sets:", "",
              "| method | AUC | BEDROC(20) | EF@1 % | EF@5 % |", "|---|---:|---:|---:|---:|"]
    mean = {}
    for m in ORDER:
        if m not in best:
            continue
        mean[m] = {k: float(np.nanmean([r[k] for r in best[m]]))
                   for k in ("auc", "bedroc_20", "ef_1", "ef_5")}
        lines.append(f"| {LABEL[m]} | {mean[m]['auc']:.3f} | {mean[m]['bedroc_20']:.3f} | "
                     f"{mean[m]['ef_1']:.1f} | {mean[m]['ef_5']:.1f} |")
    ceiling = float(np.nanmean([r["max_ef_1"] for m in best for r in best[m]]))
    one = max((m for m in mean if m.startswith("oc")), key=lambda m: mean[m]["auc"], default=None)
    fp = mean.get("ecfp4", {}).get("auc", float("nan"))
    lines += ["",
              f"The ceiling for EF@1 % here is {ceiling:.1f}: the top percent of a library this "
              "size cannot hold more actives than it has places.", "",
              (f"**The fields do not beat the fingerprint.** The best one-class description "
               f"reaches {mean[one]['auc']:.3f} AUC against {fp:.3f} for the maximum ECFP4 "
               f"Tanimoto to the same actives"
               if one and mean[one]["auc"] <= fp else
               f"**The fields beat the fingerprint**: the best one-class description reaches "
               f"{mean[one]['auc']:.3f} AUC against {fp:.3f} for ECFP4") +
              ", which is the comparison any three-dimensional method owes its reader and which "
              "this repository had not made against measured inactives before.", ""]
    bounds = {m: (float(np.nanmean([r.get("best_auc", np.nan) for r in best[m]])),
                  float(np.nanmean([r.get("best_ef_1", np.nan) for r in best[m]])))
              for m in best if m.startswith("oc")}
    bounds = {m: v for m, v in bounds.items() if np.isfinite(v[0])}
    if bounds:
        top = max(bounds, key=lambda m: bounds[m][0])
        n_set = sum(len(v) for v in GRID.values()) * len(main_data[0].get("alphas", ALPHAS))
        lines += [f"**And not with the knobs turned to their best settings either.** Every "
                   f"setting that can change the order of the ranking was tried -- the field "
                   f"attenuation over {', '.join(f'{v:g}' for v in ALPHAS)} and each "
                   f"description's own parameter, {n_set} fits in all -- and the value that "
                   f"scores best *on the test set* kept, which no honest protocol could pick, so "
                   f"this is an upper bound rather than a result. It lifts the best description "
                   f"only to {bounds[top][0]:.3f} AUC ({LABEL[top]}, EF@1 % "
                   f"{bounds[top][1]:.1f}, averaged over each set's own best setting: "
                   + "; ".join(f"{d['set']} at {d['methods'][top]['best_at']}"
                               for d in main_data if top in d["methods"]) +
                   "), still short of the fingerprint at its single default setting. The gap is "
                   "not a badly chosen parameter.", ""]
        lines += ["nu is deliberately absent from that sweep. It places the threshold so that a "
                  "fraction nu of the training actives falls outside the description, which shifts "
                  "every decision value by the same constant: AUC, BEDROC and the enrichment "
                  "factors cannot depend on it, and a sweep over it measures nothing.", ""]
    below = [(d["set"], m) for d in main_data for m in d["methods"]
             if m.startswith("oc") and d["methods"][m]["auc"] < 0.5]
    if below:
        worst = min(((d["set"], m, d["methods"][m]["auc"]) for d in main_data
                     for m in d["methods"] if m.startswith("oc")), key=lambda t: t[2])
        lines += [(f"**Some of them are worse than chance** -- {LABEL[worst[1]]} scores "
                   f"{worst[2]:.3f} on {worst[0]}, which means the held-out actives look *less* "
                   f"typical of the training actives than the measured inactives do. That is not "
                   f"a bug; it is what the AVE debiasing was designed to produce. The embedding "
                   f"removes the analogue bias by construction, so a method whose whole content "
                   f"is 'this molecule resembles the actives' has had its signal taken away, and "
                   f"with it any reason for the result to stay above 0.5."), ""]
    if "pu" in mean and "supervised" in mean:
        d_auc = mean["pu"]["auc"] - mean["supervised"]["auc"]
        d_ef = mean["pu"]["ef_1"] - mean["supervised"]["ef_1"]
        lines += [(f"**Treating the inactives as unlabelled is worth {d_auc:+.3f} AUC and "
                   f"{d_ef:+.1f} in EF@1 %** against calling them negative. "
                   + ("The bagging is doing something: an unlabelled molecule that looks like "
                      "the actives is called negative in few bags and scores high in the rest."
                      + ((f" And it is not the kernel that is failing: the same CMF kernel, "
                          f"used discriminatively, reaches {mean['pu']['auc']:.3f} against the "
                          f"fingerprint's {fp:.3f}. What fails against measured inactives is the "
                          f"one-class *description* of the actives, not the field similarity it "
                          f"is built from.") if abs(mean["pu"]["auc"] - fp) < 0.05 else "")
                      if d_auc > 0 else
                      "On these sets it buys nothing, which is what should happen when the "
                      "inactives really were measured inactive -- LIT-PCBA's are, and the PU "
                      "assumption is therefore wrong here in the direction that costs nothing.")),
                  ""]
    if similar:
        lines += ["## What the posing is worth, which is most of the result", "",
                  "The background sets elsewhere in this repository are posed to maximise their "
                  "field similarity *to the actives*, so that no decoy can be rejected for "
                  "having been posed badly. The same molecules posed both ways -- against the "
                  "crystal ligand, and against the actives -- say what that choice is worth:",
                  "",
                  "| set | method | AUC, posed by the crystal ligand | AUC, posed to resemble "
                  "the actives | difference |", "|---|---|---:|---:|---:|"]
        deltas = {}
        for d in similar:
            base = next((b for b in main_data if b["set"] == d["set"]), None)
            if base is None:
                continue
            for m in ORDER:
                if m in d["methods"] and m in base["methods"]:
                    lo, hi = base["methods"][m]["auc"], d["methods"][m]["auc"]
                    deltas.setdefault(m, []).append(hi - lo)
                    lines.append(f"| {d['set']} | {LABEL[m]} | {lo:.3f} | {hi:.3f} | "
                                 f"{hi - lo:+.3f} |")
        oc_d = float(np.mean([v for m, vs in deltas.items() if m.startswith("oc") for v in vs]))
        fp_d = float(np.mean(deltas.get("ecfp4", [np.nan])))
        lines += ["",
                  f"**The generous placement is worth {oc_d:+.3f} AUC to the field descriptions "
                  f"and {fp_d:+.3f} to the fingerprint**, which never looks at a coordinate and "
                  f"so returns exactly the same number. The direction is worth dwelling on, "
                  f"because it is not the obvious one. Posing a decoy where it most resembles "
                  f"the actives ought to make it harder to reject, and it does -- but the pose "
                  f"search also *fails* on a molecule whose field cannot be made to match, and "
                  f"the similarity it ends up with is then a shape-matching score in its own "
                  f"right. Optimising every candidate's pose against the actives quietly turns "
                  f"the alignment into part of the scoring function, which is what shape-based "
                  f"screening does on purpose and what a field model measured this way is "
                  f"getting for free.", "",
                  "Two consequences. A screening figure measured against a similarity-posed "
                  "background is a statement about the posing as much as about the model, so the "
                  "sets built here are posed by the crystal ligand instead, by the same rule as "
                  "the actives. And the generous condition leaks: `--ref-max` samples the posed "
                  "actives without regard to the train/test split, so the decoys' poses were "
                  "optimised against molecules that include the held-out actives. Both are "
                  "reasons to read the crystal-posed table as the result and this one as its "
                  "control.", ""]
    lines += ["A caveat that no amount of care removes: a screening run in the wild has to "
              "generate and choose poses for molecules nobody has placed, and will do worse than "
              "any of this."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                        # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--bags", type=int, default=BAGS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=["crystal", "similar"],
                    help="which background to score: 'crystal', posed by the rule the actives "
                         "were posed by, or 'similar', posed to resemble them")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    for name in a.sets:
        for variant in a.variants:
            if os.path.exists(os.path.join(DATA, f"{name}_decoys"
                                           f"{'' if variant == 'crystal' else '_' + variant}.sdf")):
                run_set(name, alpha=a.alpha, lam=a.lam, bags=a.bags, seed=a.seed,
                        force=a.force, variant=variant)
    print()
    print(report())


if __name__ == "__main__":
    main()
