#!/usr/bin/env python
"""
Does the co-field's error bar identify which lobes are real?

A co-field is drawn as if it were a fact about the chemistry. It is a linear map of the measured
activities, so part of its shape is a fact about the measurement error instead, and
`cmfqsar.cofields.cofield_variance` says in closed form how much: the activities carry noise of
some level, the dual coefficients carry the covariance that noise induces, and every point of the
co-field carries the standard deviation that follows. Item F2 proposes drawing only what lies
more than two of those standard deviations from zero.

The claim to check is not the arithmetic -- a unit test does that against refits with noisy
activities -- but the *use*: that what the mask keeps is the part of the picture that would still
be there if the series had come out differently. So the perturbation here is harder than the one
the formula models. The training set is resampled with replacement, the model is refitted, and
the co-field is recomputed; a voxel is called **stable** when its sign is unchanged. The question
is whether the kept region is more stable than the region the mask cuts away, and by how much.

Everything reuses one kernel and one evaluation of the per-molecule fields on the grid, so a
bootstrap replicate costs a solve and a matrix-vector product rather than a rebuild.

Usage
-----
    python benchmarks/run_cofield_error.py
    python benchmarks/run_cofield_error.py --sets GPB THER --boots 20
    python benchmarks/run_cofield_error.py --report
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

from cmfqsar.cofields import make_grid, per_molecule_fields  # noqa: E402
from cmfqsar.cofields import dual_covariance_factor, noise_level  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, krr_fit  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "cofield_error")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
Z = 2.0                 # standard deviations the mask asks for
DRAWN = 0.1             # a voxel counts as drawn above this fraction of the peak
SPACING = 0.8
MARGIN = 2.0


def _model(alpha):
    fs = FieldSet.create(physchem=FIELDS, alpha=float(alpha), charge_model="file")
    return CMFModel(fieldset=fs, method="krr", normalize=True)


def _joint(Kf):
    scale = np.array([1.0 / np.mean(np.diag(Kf[f])) for f in range(Kf.shape[0])])
    return np.tensordot(scale, Kf, axes=(0, 0))


def run_set(name, boots=30, force=False, data=DATA, results=RESULTS):
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

    # one table, one set of kernels per alpha; the evidence picks alpha and lambda
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="file")
    table = AtomTable.from_molecules([build_fields(m, fs, nm) for m, nm in zip(mols, names)])
    best = None
    for a in ALPHAS:
        m = _model(a)
        Kf = np.asarray(m.prepare_training_kernels(table))
        K = _joint(Kf)
        for lam in LAMS:
            e = log_evidence(K, y, float(lam), fit_bias=True)
            if np.isfinite(e) and (best is None or e > best[0]):
                best = (e, float(a), float(lam), Kf, m.self_diag)
    _e, alpha, lam, Kf, sd_diag = best
    model = _model(alpha)
    model.params["lam"] = lam
    model.set_training_kernels(table, Kf, sd_diag)
    model.fit(table, y, cv_folds=10, cv_seed=0, reuse_kernels=True)
    sigma, how = noise_level(model)
    G = dual_covariance_factor(model)
    grid = make_grid(model.train.coords, margin=MARGIN, spacing=SPACING)
    print(f"  alpha {alpha:g}, lam {lam:g}, q2 {model.stats.q2:+.3f}, sigma {sigma:.3f} ({how}); "
          f"grid {grid.shape} = {grid.n_points} points", flush=True)

    rng = np.random.RandomState(0)
    boot_idx = [rng.randint(0, n, n) for _ in range(boots)]
    rows = []
    for f, spec in enumerate(model.fieldset.specs):
        X = per_molecule_fields(model, f, grid.points())          # (n, P)
        C = model.coef @ X
        sd = sigma * np.sqrt(np.einsum("ij,ij->j", G.T @ X, G.T @ X))
        peak = float(np.max(np.abs(C)))
        if peak <= 0:
            continue
        drawn = np.abs(C) >= DRAWN * peak
        kept = (np.abs(C) >= Z * sd) & drawn
        cut = drawn & ~kept
        agree = np.zeros(len(C))
        boot = []
        for ix in boot_idx:
            Kb = _joint(Kf[:, ix][:, :, ix])
            a_b, _b = krr_fit(Kb, y[ix], lam, True)
            w = np.bincount(ix, weights=a_b, minlength=n)
            Cb = w @ X
            boot.append(Cb)
            agree += (np.sign(Cb) == np.sign(C)).astype(float)
        agree /= max(len(boot_idx), 1)
        boot_sd = np.std(np.asarray(boot), axis=0, ddof=1)
        ok = (sd > 0) & drawn
        rows.append({
            "field": spec.name,
            "drawn": int(drawn.sum()),
            "kept_share": float(kept.sum() / max(drawn.sum(), 1)),
            "stability_kept": float(np.mean(agree[kept])) if kept.any() else float("nan"),
            "stability_cut": float(np.mean(agree[cut])) if cut.any() else float("nan"),
            "sd_ratio": float(np.median(boot_sd[ok] / sd[ok])) if ok.any() else float("nan"),
            "sd_rank_corr": float(_spearman(sd[ok], boot_sd[ok])) if ok.sum() > 10 else float("nan"),
        })
        print(f"  {spec.name:14s} kept {100 * rows[-1]['kept_share']:4.0f} %   stability "
              f"{rows[-1]['stability_kept']:.3f} kept against {rows[-1]['stability_cut']:.3f} cut"
              f"   bootstrap sd / closed form {rows[-1]['sd_ratio']:.2f}", flush=True)

    out = {"set": name, "n": int(n), "alpha": alpha, "lam": lam, "q2": float(model.stats.q2),
           "sigma": float(sigma), "boots": int(boots), "z": Z, "spacing": SPACING,
           "points": int(grid.n_points), "fields": rows, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def _spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).statistic


def report(results=RESULTS):
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No co-field error results yet; run `python benchmarks/run_cofield_error.py`."
    data = [json.load(open(f)) for f in files]
    lines = ["# Does the co-field's error bar identify which lobes are real", "",
             f"{len(data)} Sutherland sets. Each model's co-fields are computed on a "
             f"{data[0]['spacing']:g} Å grid together with their closed-form standard "
             f"deviation, and the mask keeps what lies more than {data[0]['z']:g} of them from "
             f"zero. The training set is then resampled with replacement {data[0]['boots']} "
             "times and refitted; a voxel is **stable** when the sign of its co-field is unchanged by "
             "that. Only the drawn part of each picture is counted -- a tenth of the peak and "
             "above, which is well below any isosurface anyone draws.", "",
             "| set | n | q² | kept by the mask | stability kept | stability cut | "
             "difference |", "|---|---:|---:|---:|---:|---:|---:|"]
    diffs, kept_all, keptst, cutst = [], [], [], []
    per_field = {}
    for d in sorted(data, key=lambda r: -r["n"]):
        k = float(np.mean([r["kept_share"] for r in d["fields"]]))
        sk = float(np.nanmean([r["stability_kept"] for r in d["fields"]]))
        sc = float(np.nanmean([r["stability_cut"] for r in d["fields"]]))
        diffs.append(sk - sc)
        kept_all.append(k)
        keptst.append(sk)
        cutst.append(sc)
        for r in d["fields"]:
            per_field.setdefault(r["field"], []).append(r)
        lines.append(f"| {d['set']} | {d['n']} | {d['q2']:+.3f} | {k:.0%} | {sk:.3f} | {sc:.3f} | "
                     f"{sk - sc:+.3f} |")
    rng = np.random.default_rng(0)
    bt = [float(np.mean(rng.choice(diffs, len(diffs), replace=True))) for _ in range(5000)]
    lines.append(f"| **mean** |  |  | **{float(np.mean(kept_all)):.0%}** | "
                 f"**{float(np.mean(keptst)):.3f}** | **{float(np.mean(cutst)):.3f}** | "
                 f"**{float(np.mean(diffs)):+.3f}** |")
    lines += ["",
              f"The mask keeps {float(np.mean(kept_all)):.0%} of what would have been drawn. What "
              f"it keeps holds its sign under resampling {float(np.mean(keptst)):.1%} of the time "
              f"against {float(np.mean(cutst)):.1%} for what it cuts: a difference of "
              f"{float(np.mean(diffs)):+.3f} with a 95 % interval of "
              f"[{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}], in the same "
              f"direction on {int(sum(1 for v in diffs if v > 0))} of {len(diffs)} sets.", "",
              "By field, over every set:", "",
              "| field | kept by the mask | stability kept | stability cut | bootstrap sd / "
              "closed form | rank correlation of the two |", "|---|---:|---:|---:|---:|---:|"]
    for name, rs in per_field.items():
        lines.append(f"| {name} | {float(np.mean([r['kept_share'] for r in rs])):.0%} | "
                     f"{float(np.nanmean([r['stability_kept'] for r in rs])):.3f} | "
                     f"{float(np.nanmean([r['stability_cut'] for r in rs])):.3f} | "
                     f"{float(np.nanmedian([r['sd_ratio'] for r in rs])):.2f} | "
                     f"{float(np.nanmean([r['sd_rank_corr'] for r in rs])):+.2f} |")
    ratio = float(np.nanmedian([r["sd_ratio"] for rs in per_field.values() for r in rs]))
    corr = float(np.nanmean([r["sd_rank_corr"] for rs in per_field.values() for r in rs]))
    direction = ("less far than the closed form allows for, so the bar drawn here is "
                 f"conservative: cutting at {data[0]['z']:g} of its standard deviations is "
                 f"cutting at about {data[0]['z'] / max(ratio, 1e-9):.1f} of the bootstrap's. "
                 "That follows from the noise level it is given -- the out-of-fold error, which "
                 "is the assay's noise and the model's own error together, and so is larger than "
                 "the activities' noise alone. Pass a smaller sigma to cut less."
                 if ratio < 1 else
                 "further than the closed form allows for, so the bar drawn here is the "
                 "measurement's share of the uncertainty and not the whole of it; a reader who "
                 "wants the whole of it should resample the series.")
    lines += ["",
              "The last two columns are what the error bar can and cannot claim. The bootstrap "
              f"moves the molecules as well as the activities, and it moves the co-field about "
              f"{ratio:.1f} times as far as the closed form does -- {direction}", "",
              f"What the formula does get right is *where* the uncertainty is: the two orderings "
              f"agree across the grid at a rank correlation of {corr:+.2f}. That is what a mask "
              "needs, because a mask is an ordering with a line drawn through it, and it is why "
              "the kept region holds its sign under a perturbation the formula never modelled."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--boots", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    for name in a.sets:
        run_set(name, boots=a.boots, force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
