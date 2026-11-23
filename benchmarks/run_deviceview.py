#!/usr/bin/env python
"""
The molecule vector of a trained device as a kernel view (plan item ND14b)

Plan item ND7 stacked a kernel from the bond graph under the five field kernels, and asked
whether it adds anything the rotation-invariant kernel of item C5 does not already give. This
benchmark asks the same of the molecule vector of a trained neural device: the *collectors*
(sums over the atoms, bonds and two-bond paths of the outputs of small networks) of the QSPR
Neural Device, trained on the training molecules of the series. A Gaussian kernel on this vector
is stacked under the field kernels (``cmfqsar/deviceview.py``) and gets a weight from the
evidence of item E6 like any field.

The devices are those of plan item ND1: the files in ``results/featuriser/device/`` hold the
collectors of every molecule for every split, for the device with message passing (``mp``) and
without it (``oneshot``). Nothing is trained again here. The splits are those of ND1: 70 %
training, 15 % validation, 15 % test; ten random splits of each of the eight Sutherland series
and five of COMPAS-1D. Every score is on the test molecules, which the device never saw.

What is compared (every row is a ridge regression on one joint kernel)
----------------------------------------------------------------------
* *fields*: the five field kernels (nine for COMPAS-1D) with the weights of the evidence;
* *fields + invariant*: the same with the rotation-invariant kernel as a view, which is the
  alignment-free description the program already has;
* *fields + device*: the same with the Gaussian kernel on the collectors of the device with
  message passing, weights by the evidence on the training molecules (plan item E6);
* *fields + invariant + device*: all of them;
* *fields + device, no message passing*: as *fields + device* with the device of ND1 that reads
  each atom, bond and path alone;
* *... weight on validation molecules*: the same two views, but the share of the view in the
  joint kernel and the width of its Gaussian are chosen on the validation molecules. The
  device was trained on the training molecules, so the collectors of those are fitted to their
  activities and the evidence on them can overstate the view (``cmfqsar/deviceview.py`` says
  why the validation molecules are the better judge);
* *device alone*: the Gaussian kernel on the collectors alone, width and ridge on the validation
  molecules, and *the device itself*: the prediction the device stored.

The kernel-correlation check of plan item EC5 comes first and is reported on its own: the
correlation of the kernel on the collectors with the kernel of the five fields and with the
kernel of the five fields plus the invariant kernel, over all pairs of molecules and over the
pairs of test molecules.

Sets and choices fixed before any run
-------------------------------------
* The width of the Gaussian is 0.5, 1 or 2 times the median distance of two training molecules,
  chosen on the validation molecules for the view alone. The ridge of the models with weights by
  the evidence is the best of 0.001 to 10 by the evidence on the training molecules (as in ND7).
* On the eight Sutherland sets the attenuation of the fields is the best of five by the evidence
  on the training molecules of the split. On COMPAS-1D it is 0.1, as in ND7, and the molecules
  are the 2,000 of ND7 (a random subset, seed 0); the training, validation and test molecules of
  a split are those of the 8,678 that fall in this subset.
* **The gate** (as for ND7, and as for ND14a on COMPAS-1D and one Sutherland series): *fields +
  device* must have a smaller absolute error than *fields + invariant* on COMPAS-1D and on DHFR,
  with a 95 % bootstrap interval over the test molecules that excludes zero on both. A gain
  that the invariant kernel already gives is not a result.

Usage
-----
    python benchmarks/run_deviceview.py --stage sutherland
    python benchmarks/run_deviceview.py --stage compas --seeds 0     # one split per process
    python benchmarks/run_deviceview.py --report
"""
from __future__ import annotations

import argparse
import copy
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

from rdkit import RDLogger  # noqa: E402

from cmfqsar import deviceview as DV  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, autocorrelation_kernel, feature_matrix  # noqa: E402
from cmfqsar.mixing import evidence_mixing, joint_kernel, with_views  # noqa: E402
from run_errorpred import DATA, FIELDS, LAMS, SETS, _choose, _kernels  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "deviceview")
DEVICES = os.path.join(HERE, "results", "featuriser", "device")
COMPAS_SUBSET = 2000
COMPAS_GROUPS = "ABC"
COMPAS_ALPHA = 0.1
SEEDS = {"sutherland": tuple(range(10)), "compas-1D": tuple(range(5))}

MODELS = ("fields", "fields + invariant", "fields + device", "fields + invariant + device",
          "fields + device, no message passing", "fields + device, weight on validation",
          "fields + device, no message passing, weight on validation", "device alone",
          "the device itself")
#: the models that need an evidence fit with the device in the stack are left out on COMPAS-1D
#: when asked (a fit takes 15 to 30 minutes there)
CORRELATIONS = ("device against fields", "device against fields + invariant",
                "invariant against fields", "device against device without message passing")


def _say(s: str) -> None:
    print(s, flush=True)


def _krr_abs(K, y, tr, rows, lam):
    """Ridge regression with a ridge in the units of the kernel (as the evidence is computed)."""
    Ktt = K[np.ix_(tr, tr)]
    return DV.ridge_predict(K, y, tr, rows, float(lam) / float(np.mean(np.diag(Ktt))))


def _entry(y, rows, pred):
    y = np.asarray(y, float)[rows]
    pred = np.asarray(pred, float)
    err = pred - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"abs_errors": np.abs(err).tolist(), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1.0 - np.sum(err ** 2) / ss) if ss > 0 else float("nan")}


def _fit_joint(Kf, field_names, views, y, tr):
    """The stack with the views, the weights by the evidence on the training molecules, the ridge
    by the evidence, and the joint kernel over all molecules."""
    sub = {k: v[np.ix_(tr, tr)] for k, v in views.items()}
    stack_tr, names, scale = with_views(Kf[:, tr][:, :, tr], sub, names=field_names)
    h = evidence_mixing(stack_tr, y[tr], scale=scale)
    stack, _n, _s = with_views(Kf, views, names=field_names, scale=scale)
    K = joint_kernel(stack, h, scale)
    Ktt = K[np.ix_(tr, tr)]
    lam = max(LAMS, key=lambda v: log_evidence(Ktt, y[tr], float(v), fit_bias=True))
    shares = np.asarray(h, float) * np.asarray(scale, float)
    shares = shares / max(float(shares.sum()), 1e-300)
    return K, float(lam), dict(zip(names, (float(s) for s in shares)))


def one_split(Kf, field_names, K_inv, y, mp, os_, with_oneshot_evidence=True, log=_say):
    """Every model on one split. `mp` and `os_` are the stored collectors of the two devices;
    their rows are the molecules of `Kf` (all rows of the set or of the subset)."""
    tr, va, te = mp["train"], mp["valid"], mp["test"]
    out = {"models": {}, "kernel_correlation": {}, "shares": {}}
    t0 = time.time()
    width, ridge_w, v_alone = DV.choose_width(mp["collectors"], y, tr, va)
    K_mp = DV.gaussian_view(mp["collectors"], tr, width)
    width_os, _r, _v = DV.choose_width(os_["collectors"], y, tr, va)
    K_os = DV.gaussian_view(os_["collectors"], tr, width_os)
    out["width"] = {"mp": width, "oneshot": width_os}
    out["valid_rmse_alone"] = v_alone

    fits = {}
    fits["fields"] = _fit_joint(Kf, field_names, {}, y, tr)
    fits["fields + invariant"] = _fit_joint(Kf, field_names, {"invariant": K_inv}, y, tr)
    fits["fields + device"] = _fit_joint(Kf, field_names, {"device": K_mp}, y, tr)
    fits["fields + invariant + device"] = _fit_joint(
        Kf, field_names, {"invariant": K_inv, "device": K_mp}, y, tr)
    if with_oneshot_evidence:
        fits["fields + device, no message passing"] = _fit_joint(
            Kf, field_names, {"device": K_os}, y, tr)
    for name, (K, lam, shares) in fits.items():
        out["models"][name] = _entry(y, te, _krr_abs(K, y, tr, te, lam))
        out["models"][name]["ridge"] = lam
        out["shares"][name] = shares

    # the kernel-correlation check of plan item EC5, before any comparison of accuracy
    K_f, K_fi = fits["fields"][0], fits["fields + invariant"][0]
    for label, rows in (("all", None), ("test", te)):
        out["kernel_correlation"][label] = {
            CORRELATIONS[0]: DV.kernel_correlation(K_mp, K_f, rows),
            CORRELATIONS[1]: DV.kernel_correlation(K_mp, K_fi, rows),
            CORRELATIONS[2]: DV.kernel_correlation(K_inv, K_f, rows),
            CORRELATIONS[3]: DV.kernel_correlation(K_mp, K_os, rows)}

    # the share of the view chosen on the validation molecules
    for label, K_view in (("fields + device, weight on validation", K_mp),
                          ("fields + device, no message passing, weight on validation", K_os)):
        share, lam, _e = DV.choose_weight(K_f, K_view, y, tr, va)
        K = (1.0 - share) * DV.cosine_normalise(K_f) + share * DV.cosine_normalise(K_view)
        out["models"][label] = _entry(y, te, DV.ridge_predict(K, y, tr, te, lam))
        out["models"][label].update({"share": share, "ridge": lam})
    out["models"]["device alone"] = _entry(y, te, DV.ridge_predict(K_mp, y, tr, te, ridge_w))
    if "pred" in mp:
        out["models"]["the device itself"] = _entry(y, te, mp["pred"][te])
    out["seconds"] = time.time() - t0
    log("    " + ", ".join(f"{k} {v['r2']:+.3f}" for k, v in out["models"].items()
                           if k in ("fields", "fields + invariant", "fields + device",
                                    "fields + device, weight on validation", "the device itself")))
    return out


def _collectors(name, seed):
    mp = DV.read_collectors(os.path.join(DEVICES, f"{name}_mp_{seed}.npz"))
    os_ = DV.read_collectors(os.path.join(DEVICES, f"{name}_oneshot_{seed}.npz"))
    for k in ("train", "valid", "test"):
        if not np.array_equal(mp[k], os_[k]):
            raise SystemExit(f"{name} seed {seed}: the two devices have different splits")
    return mp, os_


def run_sutherland(name, seeds, force=False, log=_say):
    os.makedirs(RESULTS, exist_ok=True)
    todo = [s for s in seeds if force or not os.path.exists(
        os.path.join(RESULTS, f"{name}_{s}.json"))]
    if not todo:
        log(f"{name}: already done")
        return
    ds = load_dataset(os.path.join(DATA, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    y = y[keep]
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="file")
    table = AtomTable.from_molecules([build_fields(m, fs, nm) for m, nm in zip(mols, names)])
    cache = _kernels(table)
    K_inv = autocorrelation_kernel(feature_matrix(table.molecules, AutocorrelationSpec(),
                                                  fieldset=fs))
    log(f"=== {name}: {len(y)} molecules")
    for seed in todo:
        mp, os_ = _collectors(name, seed)
        if list(mp["names"]) != names:
            raise SystemExit(f"{name}: the molecules of the device file differ from the set")
        alpha, _lam = _choose(cache, y[mp["train"]], mp["train"])
        Kf, _sd = cache[alpha]
        log(f"  seed {seed}: attenuation {alpha:g}")
        out = one_split(Kf, FIELDS, K_inv, y, mp, os_, log=log)
        out.update({"set": name, "seed": seed, "n": int(len(y)), "alpha": alpha,
                    "test": [int(i) for i in mp["test"]]})
        with open(os.path.join(RESULTS, f"{name}_{seed}.json"), "w") as fh:
            json.dump(out, fh)


def run_compas(seeds, subset=COMPAS_SUBSET, force=False, with_oneshot_evidence=False,
               log=_say):
    import run_compas as C
    from cmfqsar.model import CMFModel

    os.makedirs(RESULTS, exist_ok=True)
    todo = [s for s in seeds if force or not os.path.exists(
        os.path.join(RESULTS, f"compas-1D_{s}.json"))]
    if not todo:
        log("COMPAS-1D: already done")
        return
    ds, meta = C.load_prepared("1D")
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(ds.names), int(subset), replace=False))
    sub = copy.copy(ds)
    sub.mols = [ds.mols[i] for i in idx]
    sub.names = [ds.names[i] for i in idx]
    sub.activity = np.asarray(ds.activity, float)[idx]
    y_all = np.asarray(ds.activity, float)
    y = y_all[idx]
    fs = C.make_fieldset(COMPAS_GROUPS, meta, {g: COMPAS_ALPHA for g in COMPAS_GROUPS})
    table = C.build_table(sub, fs)
    Kf = np.asarray(CMFModel(fieldset=fs, method="krr", normalize=True)
                    .prepare_training_kernels(table), float)
    field_names = [s.name for s in fs.specs if s.enabled]
    K_inv = autocorrelation_kernel(feature_matrix(table.molecules, AutocorrelationSpec(),
                                                  fieldset=fs))
    position = {int(g): j for j, g in enumerate(idx)}
    log(f"=== COMPAS-1D: {len(idx)} of {len(ds.names)} molecules, {len(field_names)} fields")
    for seed in todo:
        mp, os_ = _collectors("compas-1D", seed)
        if list(mp["names"]) != list(ds.names):
            raise SystemExit("compas-1D: the molecules of the device file differ from the set")

        def restrict(d):
            r = dict(d)
            r["collectors"] = d["collectors"][idx]
            for k in ("train", "valid", "test"):
                r[k] = np.array([position[int(i)] for i in d[k] if int(i) in position], int)
            r["y"] = d["y"][idx]
            if "pred" in d:
                r["pred"] = d["pred"][idx]
            return r

        # the width of the view and its Gaussian use the training molecules of the subset
        # only, but the collectors are those of the device trained on all 6,074 of them
        mp_s, os_s = restrict(mp), restrict(os_)
        log(f"  seed {seed}: {len(mp_s['train'])} training, {len(mp_s['valid'])} validation, "
            f"{len(mp_s['test'])} test molecules")
        out = one_split(Kf, field_names, K_inv, y, mp_s, os_s,
                        with_oneshot_evidence=with_oneshot_evidence, log=log)
        out.update({"set": "compas-1D", "seed": seed, "n": int(len(idx)), "alpha": COMPAS_ALPHA,
                    "test": [int(idx[i]) for i in mp_s["test"]], "fields": field_names})
        with open(os.path.join(RESULTS, f"compas-1D_{seed}.json"), "w") as fh:
            json.dump(out, fh)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def _runs(name):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, f"{name}_*.json")))
            if os.path.basename(f).rsplit("_", 1)[0] == name]


def paired(runs, a, b, draws=2000, seed=0):
    """The difference of the absolute errors of two models over the test molecules of every
    seed: a molecule that is in several test parts counts once, with its mean difference."""
    diff = {}
    for r in runs:
        if a not in r["models"] or b not in r["models"]:
            continue
        da = np.asarray(r["models"][a]["abs_errors"], float)
        db = np.asarray(r["models"][b]["abs_errors"], float)
        for i, v in zip(r["test"], da - db):
            diff.setdefault(int(i), []).append(float(v))
    if not diff:
        return None
    d = np.array([np.mean(v) for v in diff.values()])
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(draws)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(d.mean()), "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "molecules": int(len(d))}


def _fmt(p, nd=3):
    return "-" if p is None else f"{p['mean']:+.{nd}f} [{p['low']:+.{nd}f}, {p['high']:+.{nd}f}]"


def _mean(runs, model, key="r2"):
    v = [r["models"][model][key] for r in runs if model in r["models"]]
    return float(np.mean(v)) if v else float("nan")


def write_table():
    sets = [s for s in SETS if _runs(s)]
    compas = _runs("compas-1D")
    groups = [(s, _runs(s)) for s in sets] + ([("COMPAS-1D", compas)] if compas else [])
    L = ["# The molecule vector of a trained device as a kernel view (plan item ND14b)", "",
         "A Gaussian kernel on the collectors of the QSPR Neural Device (the sums over its atoms, "
         "bonds and two-bond paths), trained on the training molecules of the series, is stacked "
         "under the field kernels and gets a weight by the evidence (plan item E6). The devices "
         "and the splits are those of plan item ND1; nothing is trained again. The sets are the "
         "eight Sutherland series (ten random splits each) and 2,000 molecules of COMPAS-1D "
         "(five splits, as in ND7). Every score is on the test molecules. *invariant* is the "
         "rotation-invariant autocorrelation kernel of plan item C5.", ""]
    if not groups:
        return "\n".join(L + ["No results yet; run `python benchmarks/run_deviceview.py`."])
    # the kernel-correlation check
    L += ["## The kernel-correlation check (plan item EC5)", "",
          "The correlation of the kernel on the collectors with the kernel the model already has, "
          "over all pairs of two different molecules after both kernels are normalised to a "
          "diagonal of one. Mean over the splits. A value near one means that the view adds no "
          "new description. *all* uses every molecule of the set, *test* only the test molecules "
          "(the collectors of the training molecules are fitted to their activities).", "",
          "| Set | Splits | Device against the five fields, all | Device against the five "
          "fields, test | Device against fields + invariant, all | Device against fields + "
          "invariant, test | Invariant against the five fields, all | Device against the device "
          "without message passing, all |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, runs in groups:
        def m(label, key):
            return float(np.mean([r["kernel_correlation"][label][key] for r in runs]))
        L.append(f"| {name} | {len(runs)} | {m('all', CORRELATIONS[0]):.3f} | "
                 f"{m('test', CORRELATIONS[0]):.3f} | {m('all', CORRELATIONS[1]):.3f} | "
                 f"{m('test', CORRELATIONS[1]):.3f} | {m('all', CORRELATIONS[2]):.3f} | "
                 f"{m('all', CORRELATIONS[3]):.3f} |")
    # the models
    L += ["", "## Test R² of every model", "",
          "Mean over the splits. The weights of the models in the first block are those of the "
          "evidence on the training molecules. In the second block the share of the view in the "
          "joint kernel and the width of its Gaussian are chosen on the validation molecules.", "",
          "| Model | " + " | ".join(n for n, _r in groups) + " | Mean over the Sutherland series |",
          "|---|" + "---:|" * (len(groups) + 1)]
    suth = [i for i, (n, _r) in enumerate(groups) if n != "COMPAS-1D"]
    for model in MODELS:
        vals = [_mean(r, model) for _n, r in groups]
        if all(np.isnan(v) for v in vals):
            continue
        ms = np.nanmean([vals[i] for i in suth]) if suth else float("nan")
        L.append(f"| {model} | " + " | ".join("-" if np.isnan(v) else f"{v:+.3f}" for v in vals)
                 + f" | {ms:+.3f} |")
    # the paired differences
    pairs = [("fields + device", "fields + invariant"), ("fields + device", "fields"),
             ("fields + invariant + device", "fields + invariant"),
             ("fields + device, weight on validation", "fields + invariant"),
             ("fields + device, weight on validation", "fields"),
             ("fields + device, no message passing", "fields + invariant"),
             ("fields + device, no message passing, weight on validation", "fields + invariant"),
             ("fields + invariant", "fields"), ("device alone", "the device itself"),
             ("fields + device, weight on validation", "the device itself")]
    L += ["", "## Paired differences", "",
          "Difference of the absolute error on the same test molecules, in units of the activity "
          "(for COMPAS-1D in eV), with a 95 % bootstrap interval. A molecule in the test part of "
          "several splits counts once. A negative value means that the first model has the "
          "smaller error.", "",
          "| Number | Comparison | " + " | ".join(n for n, _r in groups)
          + " | Sets better | Sets worse |", "|---:|---|" + "---:|" * (len(groups) + 2)]
    table = {}
    for k, (a, b) in enumerate(pairs, 1):
        cells, better, worse = [], 0, 0
        for n, runs in groups:
            p = paired(runs, a, b)
            table[(a, b, n)] = p
            cells.append(_fmt(p))
            if p and p["high"] < 0:
                better += 1
            if p and p["low"] > 0:
                worse += 1
        L.append(f"| {k} | {a} against {b} | " + " | ".join(cells) + f" | {better} | {worse} |")
    # the shares
    L += ["", "## The share of the device in the joint kernel", "",
          "The share is the weight times the scale of the term, divided by the sum over the terms "
          "(mean over the splits). The last column is the share chosen on the validation "
          "molecules.", "", "| Set | Share by the evidence in fields + device | Share by the "
          "evidence in fields + invariant + device | Share of the invariant kernel in fields + "
          "invariant | Share chosen on the validation molecules |", "|---|---:|---:|---:|---:|"]
    for n, runs in groups:
        a = np.mean([r["shares"]["fields + device"]["device"] for r in runs])
        b = np.mean([r["shares"]["fields + invariant + device"]["device"] for r in runs])
        c = np.mean([r["shares"]["fields + invariant"]["invariant"] for r in runs])
        d = np.mean([r["models"]["fields + device, weight on validation"]["share"] for r in runs])
        L.append(f"| {n} | {a:.0%} | {b:.0%} | {c:.0%} | {d:.0%} |")
    # the gate
    L += ["", "## The gate", "",
          "*fields + device* must have a smaller absolute error than *fields + invariant* on "
          "COMPAS-1D and on DHFR, with an interval that excludes zero on both.", "",
          "| Set | Difference of the absolute error | Gate |", "|---|---:|---|"]
    verdict = []
    for n in ("COMPAS-1D", "DHFR"):
        p = table.get(("fields + device", "fields + invariant", n))
        ok = bool(p and p["high"] < 0)
        verdict.append(ok if p else None)
        L.append(f"| {n} | {_fmt(p, 4)} | " + ("not run" if p is None
                                                 else "met" if ok else "not met") + " |")
    L += ["", "The gate is " + ("met." if all(v for v in verdict) else
                              "not met." if None not in verdict else "not yet decided.")]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", action="append", default=[], choices=["sutherland", "compas"])
    ap.add_argument("--sets", default=None, help="comma-separated Sutherland sets")
    ap.add_argument("--seeds", default=None, help="comma-separated")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--oneshot-evidence", action="store_true",
                    help="on COMPAS-1D also fit the evidence mixing with the device without "
                         "message passing (about 20 minutes more)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    for stage in args.stage:
        if stage == "sutherland":
            seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
                     else SEEDS["sutherland"])
            for name in (args.sets.split(",") if args.sets else SETS):
                run_sutherland(name, seeds, args.force)
        else:
            seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS["compas-1D"]
            run_compas(seeds, force=args.force, with_oneshot_evidence=args.oneshot_evidence)
    if args.report or not args.stage:
        text = write_table()
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
