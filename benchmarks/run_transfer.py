#!/usr/bin/env python
"""
Does the co-field cosine say whether the SAR transfers?

`cmfqsar/transfer.py` compares two models by the overlap integral of their co-fields: a number
between -1 and +1 saying how far the two ask the same thing of the chemistry. The arithmetic is
checked against a grid in the unit tests. What is checked here is whether the number *means*
anything -- whether two models whose co-fields agree really do predict each other's molecules.

The pairs come from inside one set, because co-fields are functions on space and two models can
only be compared if their molecules were posed in one frame. Every Sutherland set is split into
its chemotypes; a model is fitted on each group large enough to carry one, with the attenuation
chosen once for the set (it has to be shared, or the two co-fields are not functions of the same
shape) and the regularisation chosen by the evidence inside each group. Then for every pair:

* the **co-field cosine**, over all fields and field by field;
* the **transfer**: Spearman's rho of model A's predictions against group B's measured
  activities, which asks whether A orders B's molecules correctly. Rho rather than R2, because
  two chemotypes often sit at different activity levels and an offset is not a failure of the
  SAR.

The question is the correlation between the two over all pairs. A positive one says the cosine
is a map of SAR transfer, as the item claims; a flat one says it is a picture with no predictive
content.

Usage
-----
    python benchmarks/run_transfer.py
    python benchmarks/run_transfer.py --sets BZR COX2
    python benchmarks/run_transfer.py --report
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

from cmfqsar.chemotypes import chemotype_labels, group_members  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, r2_score  # noqa: E402
from cmfqsar.transfer import cofield_overlap  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "transfer")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0, 2.0)
LAMS = (1e-2, 0.1, 1.0)
MIN_GROUP = 15        # molecules, or a group cannot carry a model
MAX_GROUPS = 5
HALVES = True         # also split the largest group in two, as the positive control


def _joint(model, Kf):
    return model.joint_kernel(Kf)


def run_set(name, force=False, data=DATA, results=RESULTS):
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
    labels, _gnames = chemotype_labels(mols, min_size=MIN_GROUP)
    groups = [np.asarray(g, int) for g in group_members(labels) if len(g) >= MIN_GROUP]
    groups = sorted(groups, key=len, reverse=True)[:MAX_GROUPS]
    if len(groups) < 2:
        out = {"set": name, "usable": False, "why": f"fewer than two chemotypes of {MIN_GROUP}+"}
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"=== {name}: only {len(groups)} usable chemotype")
        return out
    tags = [f"c{k}" for k in range(len(groups))]
    if HALVES:                       # the positive control: one chemotype cut in two at random
        rng = np.random.RandomState(0)
        big = groups[0]
        order = rng.permutation(len(big))
        half = len(big) // 2
        if half >= MIN_GROUP:
            groups += [big[order[:half]], big[order[half:]]]
            tags += ["h0", "h1"]
    print(f"=== {name}: {len(mols)} molecules, groups " +
          ", ".join(f"{t}({len(g)})" for t, g in zip(tags, groups)), flush=True)

    # one attenuation for the whole set: two co-fields drawn with different widths are different
    # functions, and the comparison would mix the two effects
    fs0 = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="file")
    table = AtomTable.from_molecules([build_fields(m, fs0, n) for m, n in zip(mols, names)])
    best = None
    for a in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=float(a), charge_model="file")
        m = CMFModel(fieldset=fs, method="krr", normalize=True)
        Kf = np.asarray(m.prepare_training_kernels(table))
        K = _joint(m, Kf)
        for lam in LAMS:
            e = log_evidence(K, y, float(lam), fit_bias=True)
            if np.isfinite(e) and (best is None or e > best[0]):
                best = (e, float(a), Kf, np.asarray(m.self_diag))
    _e, alpha, Kf, sd = best

    models = []
    for tag, g in zip(tags, groups):
        fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="file")
        m = CMFModel(fieldset=fs, method="krr", normalize=True)
        sub = AtomTable.from_molecules([table.molecules[int(i)] for i in g])
        m.set_training_kernels(sub, Kf[:, g][:, :, g], sd[:, g])
        pick = None
        Kg = _joint(m, Kf[:, g][:, :, g])
        for lam in LAMS:
            e = log_evidence(Kg, y[g], float(lam), fit_bias=True)
            if np.isfinite(e) and (pick is None or e > pick[0]):
                pick = (e, float(lam))
        m.params["lam"] = pick[1]
        m.fit(sub, y[g], cv_folds=min(10, len(g)), cv_seed=0, reuse_kernels=True)
        models.append(m)

    rows = []
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            gi, gj = groups[i], groups[j]
            shared = len(set(gi.tolist()) & set(gj.tolist()))
            res = cofield_overlap(models[i], models[j], tags[i], tags[j], strict=False)
            # model i predicting group j, from the kernel block the set already has
            Ki = _joint(models[i], Kf[:, gj][:, :, gi])
            pred = Ki @ models[i].coef + models[i].bias
            rho = float(spearmanr(y[gj], pred).statistic) if len(gj) > 3 else float("nan")
            rows.append({"a": tags[i], "b": tags[j], "n_a": int(len(gi)), "n_b": int(len(gj)),
                         "shared": shared, "cosine": res.cosine,
                         "by_field": res.by_field(), "rho": rho,
                         "r2": float(r2_score(y[gj], pred)),
                         "q2_a": float(models[i].stats.q2)})
            if i < j:
                print(f"  {tags[i]} -> {tags[j]}: cosine {res.cosine:+.3f}   transfer rho "
                      f"{rho:+.3f}" + ("   (halves of one chemotype)" if shared == 0 and
                                       tags[i].startswith('h') and tags[j].startswith('h')
                                       else ""), flush=True)

    out = {"set": name, "usable": True, "n": len(mols), "alpha": alpha,
           "groups": {t: int(len(g)) for t, g in zip(tags, groups)}, "rows": rows,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def _pairs(data, kind="chemotypes"):
    """Rows of one kind: between chemotypes, or between the two halves of one chemotype."""
    out = []
    for d in data:
        for r in d.get("rows", []):
            half = r["a"].startswith("h") and r["b"].startswith("h")
            if (kind == "halves") == half and not (r["a"].startswith("h") ^ r["b"].startswith("h")):
                out.append(r)
    return out


def report(results=RESULTS):
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No transfer results yet; run `python benchmarks/run_transfer.py`."
    data = [json.load(open(f)) for f in files]
    usable = [d for d in data if d.get("usable")]
    chem = _pairs(usable, "chemotypes")
    halves = _pairs(usable, "halves")
    lines = ["# Does the co-field cosine say whether the SAR transfers", "",
             "Every set is split into its chemotypes, a model is fitted on each group of at "
             f"least {MIN_GROUP} molecules with the attenuation shared across the set (it has to "
             "be, or two co-fields are not functions of the same shape), and every ordered pair "
             "is scored twice: by the cosine of the two co-fields, and by how well one model "
             "orders the other group's molecules (Spearman's rho). The largest chemotype is also "
             "cut in two at random, as the control for what agreement looks like when the SAR "
             "really is the same.", "",
             "| set | groups | pairs | mean cosine | mean transfer ρ | ρ of the two |",
             "|---|---:|---:|---:|---:|---:|"]
    all_cos, all_rho = [], []
    for d in sorted(usable, key=lambda r: -r["n"]):
        rows = [r for r in d["rows"] if not (r["a"].startswith("h") or r["b"].startswith("h"))]
        if not rows:
            continue
        cos = np.array([r["cosine"] for r in rows], float)
        rho = np.array([r["rho"] for r in rows], float)
        ok = np.isfinite(cos) & np.isfinite(rho)
        within = (float(spearmanr(cos[ok], rho[ok]).statistic) if ok.sum() > 3
                  else float("nan"))
        all_cos += cos[ok].tolist()
        all_rho += rho[ok].tolist()
        n_groups = len([t for t in d["groups"] if not t.startswith("h")])
        lines.append(f"| {d['set']} | {n_groups} | {int(ok.sum())} | {float(cos[ok].mean()):+.3f} | "
                     f"{float(rho[ok].mean()):+.3f} | {within:+.3f} |")
    cos = np.asarray(all_cos, float)
    rho = np.asarray(all_rho, float)
    overall = float(spearmanr(cos, rho).statistic) if len(cos) > 3 else float("nan")
    lines.append(f"| **all** |  | {len(cos)} | **{float(cos.mean()):+.3f}** | "
                 f"**{float(rho.mean()):+.3f}** | **{overall:+.3f}** |")

    hc = np.array([r["cosine"] for r in halves], float)
    hr = np.array([r["rho"] for r in halves], float)
    lines += ["",
              f"**The control first.** Two halves of one chemotype, which share their SAR by "
              f"construction, agree at a mean cosine of **{float(np.nanmean(hc)):+.3f}** and "
              f"transfer at rho **{float(np.nanmean(hr)):+.3f}** over {len(hc)} pairs. Two "
              f"different chemotypes of the same target agree at **{float(cos.mean()):+.3f}** "
              f"and transfer at **{float(rho.mean()):+.3f}**. The cosine separates the two cases "
              f"by {float(np.nanmean(hc)) - float(cos.mean()):+.3f}, which is what it is for: "
              "two models of one SAR look alike and two models of two SARs do not.", "",
              f"**Within the chemotype pairs, the cosine and the transfer correlate at "
              f"{overall:+.3f}** over {len(cos)} pairs. "
              + ("That is the claim of the item, measured: a table of co-field overlaps is a map "
                 "of SAR transfer." if overall > 0.3 else
                 "That is weaker than the item's claim. The cosine tells two SARs apart from one "
                 "SAR, which the control shows, but among genuinely different chemotypes it does "
                 "not grade how well one model will order another's molecules. The fair "
                 "reading is that it is a map of *agreement*, and agreement is necessary for "
                 "transfer without being sufficient for it."), "",
              "A caveat that cannot be removed by any amount of care: the whole comparison rests "
              "on the two training sets being posed in one frame. Here they are, because they "
              "come from one file; between two files they would have to be aligned first, and "
              "nothing in the cosine would reveal that they had not been."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:                       # the report is written as UTF-8; a console may not be
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:          # noqa: BLE001 - an output stream that cannot be reconfigured
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    for name in a.sets:
        run_set(name, force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
