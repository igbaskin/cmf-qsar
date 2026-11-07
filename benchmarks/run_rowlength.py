#!/usr/bin/env python
"""
What the row length is worth, where the invariant kernel divides it out (plan item M15).

:func:`cmfqsar.invariant.autocorrelation_kernel` rescales the channels and then divides every
feature row by its own length, so a kernel entry is the cosine of the angle between two rows and
the length of a row never reaches the model.  Every kernel built on that function inherits this:
the local environment kernel of a Li hop, the count kernel, the site profiles of plan item M5 and
the whole-cell autocorrelation of items M1 to M14.

The length is one number per sample.  For a whole cell it is close to how much field the cell
carries, which is how large and how dense it is; for a local environment it is how crowded that
environment is.  Item BT7's external check measured what losing it costs on one task: a forest
given the raw rows reached q2 +0.844 on the published hop barriers where the same forest on the
rows the kernel is built from reached +0.758.  This script asks the same question of the kernel
itself, on the series that already exist.

**Four arms, on the same samples, the same folds and the same ridge grid.**

    the length divided out          the kernel as the program builds it today: the reference
    the length kept                 the same kernel without the division, divided instead by its
                                    own mean diagonal, which is one constant for the whole matrix
                                    and takes no row's length away
    the length as an extra channel  (1 - w) * the reference + w * the kernel on the length alone,
                                    with w chosen inside the training part
    the length alone                the kernel on that one number, which says how much of the
                                    target the length by itself carries

The third arm is the form plan item M15 asks for: the kernel as it stands plus the length as one
more channel.  It has one parameter more than the reference, so it is chosen inside the training
part and never on the fold it is scored on.  The other arms are given the same nested treatment,
so the comparison is between descriptions and not between search budgets.

**The series.**  Each is fitted the way its own benchmark fits it, so that a number here can be
read beside the number there.

    lithium     the bond-valence site energy of every Li site of the three solid electrolytes of
                `examples/hops`, site profiles (item M5), one compound held out at a time
    sac         the H, O, OH and OOH binding energies of the single-atom catalysts of
                `examples/sac`, site profiles, under the three designs of item EC8
    hops        the barriers of the prepared LiTraj sample of item BT7, on three descriptions at
                once: counts about the midpoint, the two end sites, and the hop environment
    defects     the formation energies of the 2,000-cell MoS2 and WSe2 defect series of item M14,
                whole-cell autocorrelation
    molecules   the binding constants of the beta-cyclodextrin guests, whole-molecule
                autocorrelation.  This is the case the division was written for: the guests differ
                in size, and the length is close to the size

Usage
-----
    python benchmarks/run_rowlength.py --lithium          # about a minute
    python benchmarks/run_rowlength.py --sac              # a few minutes
    python benchmarks/run_rowlength.py --molecules        # about ten minutes
    python benchmarks/run_rowlength.py --hops             # about twenty minutes
    python benchmarks/run_rowlength.py --defects          # about an hour
    python benchmarks/run_rowlength.py --report           # the table from the stored results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

RESULTS = os.path.join(HERE, "results", "rowlength")

#: the reference arm: what the program does today
ARM_OUT = "the length divided out"
#: the same kernel with the division left out
ARM_KEPT = "the length kept"
#: the reference mixed with a kernel on the length alone, the weight chosen in the training part
ARM_PLUS = "the length as an extra channel"
#: the kernel on the length alone
ARM_ONLY = "the length alone"
ARMS = (ARM_OUT, ARM_KEPT, ARM_PLUS, ARM_ONLY)

#: the mixing weights the third arm chooses between, inside the training part
MIX_WEIGHTS = (0.1, 0.25, 0.5, 0.75)
#: the ridge grid, wide enough for every series here
LAMS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
SEED = 0
#: bootstrap draws for the interval of rule 4
BOOTSTRAP = 1000


def log(msg: str) -> None:
    print(msg, flush=True)


def result(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def jdump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_default)
        fh.write("\n")


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------- the four arms


def arm_families(settings: Dict[str, Tuple[np.ndarray, int, Optional[Sequence[int]]]],
                 weights: Sequence[float] = MIX_WEIGHTS,
                 ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """The four arms, each a family of kernels over the same descriptor settings.

    `settings` maps the name of a descriptor setting -- a cutoff, a width -- to its feature
    matrix, the number of bins of one channel block and the block widths, which is what the
    channel scale needs.  Every arm is offered the same settings, so a setting that suits one arm
    better cannot decide the comparison: the choice is made inside the training part either way.

    The channel scale is taken from all the rows, as it is in the benchmarks these series come
    from.  That is a property of the stored protocol and not of this measurement, and it is the
    same for all four arms.
    """
    from cmfqsar.invariant import (autocorrelation_kernel, channel_scale, length_kernel,
                                   row_lengths)
    fams: Dict[str, Dict[str, np.ndarray]] = {a: {} for a in ARMS}
    lengths: Dict[str, np.ndarray] = {}
    for name, (X, n_bins, blocks) in settings.items():
        X = np.asarray(X, float)
        s = channel_scale(X, n_bins, blocks=blocks)
        cosine = autocorrelation_kernel(X, scale=s, n_bins=n_bins, blocks=blocks, normalize=True)
        kept = autocorrelation_kernel(X, scale=s, n_bins=n_bins, blocks=blocks, normalize=False)
        kept = kept / max(float(np.mean(np.diag(kept))), 1e-300)
        ell = row_lengths(X, scale=s)
        only = length_kernel(ell)
        fams[ARM_OUT][name] = cosine
        fams[ARM_KEPT][name] = kept
        fams[ARM_ONLY][name] = only
        for w in weights:
            fams[ARM_PLUS][f"{name}, weight {w:g}"] = (1.0 - w) * cosine + w * only
        lengths[name] = ell
    return fams, lengths


def length_summary(ell: np.ndarray, y: np.ndarray) -> dict:
    """What the row length looks like, and how far it goes with the target on its own.

    The two correlations say whether there is anything in the length to find before any model is
    fitted: Pearson for a straight line, Spearman for any rising or falling relation.
    """
    from scipy.stats import spearmanr
    ell = np.asarray(ell, float)
    y = np.asarray(y, float)
    ok = np.isfinite(ell) & np.isfinite(y)
    out = {"mean": float(np.mean(ell[ok])), "sd": float(np.std(ell[ok])),
           "min": float(np.min(ell[ok])), "max": float(np.max(ell[ok])),
           "spread_over_mean": float(np.std(ell[ok]) / max(abs(np.mean(ell[ok])), 1e-300)),
           "n": int(ok.sum())}
    if ok.sum() > 2 and np.std(ell[ok]) > 0 and np.std(y[ok]) > 0:
        out["pearson"] = float(np.corrcoef(ell[ok], y[ok])[0, 1])
        out["spearman"] = float(spearmanr(ell[ok], y[ok])[0])
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
    return out


def nested_on_splits(kernels: Dict[str, np.ndarray], y: np.ndarray, splits,
                     lams: Sequence[float] = LAMS, inner: int = 5, seed: int = SEED) -> dict:
    """Out-of-fold predictions with the setting and the ridge chosen inside each training part.

    This is for a series whose samples are plain rows with no grouping, which is the molecular
    set here.  The series that hold out whole structures or whole compounds use the routine of
    their own benchmark instead, so that their folds are the folds their stored table was
    measured on.
    """
    from cmfqsar.model import cv_predict, krr_fit, r2_score, rmse
    y = np.asarray(y, float)
    pred = np.full(len(y), np.nan)
    chosen = []
    for tr, te in splits:
        best = None
        for key, K in kernels.items():
            Ktr = K[np.ix_(tr, tr)]
            for lam in lams:
                p = cv_predict(Ktr, y[tr], "krr", {"lam": lam}, folds=inner, seed=seed)
                q = r2_score(y[tr], p)
                if best is None or q > best[0]:
                    best = (q, key, lam)
        _, key, lam = best
        chosen.append({"setting": key, "lam": float(lam)})
        K = kernels[key]
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], float(lam))
        pred[te] = K[np.ix_(te, tr)] @ a + b
    ok = np.isfinite(pred)
    return {"predicted": pred, "q2": float(r2_score(y[ok], pred[ok])),
            "rmse": float(rmse(y[ok], pred[ok])),
            "mae": float(np.mean(np.abs(y[ok] - pred[ok]))),
            "n": int(ok.sum()), "chosen": chosen}


def paired_q2_difference(y: np.ndarray, a: np.ndarray, b: np.ndarray, runs: int = BOOTSTRAP,
                         seed: int = SEED, groups: Optional[np.ndarray] = None):
    """The difference in q2 of two sets of predictions, with its interval (rule 4).

    The draw is over whole groups when `groups` is given -- a structure, a compound -- because two
    samples of one structure are not independent; over samples otherwise.
    """
    from cmfqsar.model import r2_score
    y = np.asarray(y, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    idx = np.where(ok)[0]
    rng = np.random.RandomState(seed)
    diffs = []
    if groups is None:
        for _ in range(runs):
            i = idx[rng.randint(0, len(idx), len(idx))]
            if np.var(y[i]) <= 0:
                continue
            diffs.append(r2_score(y[i], a[i]) - r2_score(y[i], b[i]))
    else:
        groups = np.asarray(groups)
        keys = np.unique(groups[idx])
        by = {k: idx[groups[idx] == k] for k in keys}
        for _ in range(runs):
            draw = keys[rng.randint(0, len(keys), len(keys))]
            i = np.concatenate([by[k] for k in draw])
            if np.var(y[i]) <= 0:
                continue
            diffs.append(r2_score(y[i], a[i]) - r2_score(y[i], b[i]))
    d = float(r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
    if not diffs:
        return d, float("nan"), float("nan")
    return d, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _row(series: str, descriptor: str, target: str, arm: str, q2: float, error: float,
         mae: float, n: int, chosen: Optional[list] = None, extra: Optional[dict] = None) -> dict:
    out = {"series": series, "descriptor": descriptor, "target": target, "arm": arm,
           "q2": float(q2), "rmse": float(error), "mae": float(mae), "n": int(n)}
    if chosen is not None:
        out["chosen"] = chosen
    if extra:
        out.update(extra)
    return out


# ----------------------------------------------------------------------------- lithium sites (M5)


def lithium_run(args) -> dict:
    """The site profiles of the three solid electrolytes, one compound held out at a time."""
    import run_sites as S5
    from cmfqsar import sites as S
    from cmfqsar.localenv import paired_interval
    from cmfqsar.periodic import crystal_fields, inorganic_fieldset
    from run_litraj import nested_result

    structures, targets, names = S5.lithium_targets(log)
    fs = inorganic_fieldset(["Li", "B", "P", "O"], charge_model="file")
    crystals = [crystal_fields(a, fs, n) for a, n in zip(structures, names)]
    folds = len(names)                      # one compound out, as item M5 measures it
    settings = {}
    for r_max in S5.CUTOFFS:
        t0 = time.time()
        spec = S5.spec_for(r_max)
        X = S.site_features(crystals, targets, spec)
        settings[f"r={r_max:g} A"] = (X, spec.n_bins, None)
        log(f"  site profile r {r_max:g} A: {X.shape[0]} sites, {X.shape[1]} channels, "
            f"{time.time() - t0:.1f} s")
    fams, lengths = arm_families(settings)
    y = np.asarray(targets.y, float)
    out = {"series": "lithium", "descriptor": "site profile (M5)", "compounds": names,
           "n_samples": len(targets), "folds": folds, "cutoffs": list(S5.CUTOFFS),
           "weights": list(MIX_WEIGHTS), "spread": float(np.std(y)), "rows": [], "pairs": [],
           "lengths": {k: length_summary(v, y) for k, v in lengths.items()}}
    for k, s in out["lengths"].items():
        log(f"  the row length at {k}: mean {s['mean']:.3g}, spread {s['sd']:.3g} "
            f"({100 * s['spread_over_mean']:.0f} per cent of the mean); with the target "
            f"Pearson {s['pearson']:+.3f}, Spearman {s['spearman']:+.3f}")
    results = {}
    for arm in ARMS:
        res, picked = nested_result(fams[arm], targets, folds, args.seed,
                                    f"site profile, {arm}", lams=LAMS)
        results[arm] = res
        out["rows"].append(_row("lithium", "site profile (M5)", "site energy", arm, res.q2,
                                res.error, res.mae, res.n_samples, picked))
        log(f"  {arm}: q2 {res.q2:+.3f}, error {res.error:.3f} eV "
            f"(chose {sorted({c['setting'] for c in picked})})")
    for arm in ARMS[1:]:
        d, lo, hi = paired_interval(results[arm], results[ARM_OUT], targets, runs=args.bootstrap,
                                    seed=args.seed)
        out["pairs"].append({"a": arm, "b": ARM_OUT, "diff": d, "low": lo, "high": hi})
        log(f"  {arm} against the reference: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return out


# ------------------------------------------------------------------- single-atom catalysts (EC8)


def sac_run(args) -> dict:
    """The site profiles of the single-atom catalysts, four targets and three designs."""
    import run_sac as R
    import run_sites as S5
    from cmfqsar import sites as S

    data = R.load_set(log)
    sites = [R._metal_index(a) for a in data.structures]
    fs = R.fieldset_for("classic", data.metals)
    crystals = R.crystals_for(data, fs)
    index = S.one_site_targets(sites, np.zeros(len(data)), [c.n_atoms for c in crystals],
                               name="binding")
    settings = {}
    for r_max in S5.CUTOFFS:
        t0 = time.time()
        spec = S5.spec_for(r_max)
        X = S.site_features(crystals, index, spec)
        settings[f"r={r_max:g} A"] = (X, spec.n_bins, None)
        log(f"  site profile r {r_max:g} A: {X.shape[1]} channels, {time.time() - t0:.1f} s")
    fams, lengths = arm_families(settings)
    out = {"series": "sac", "descriptor": "site profile (M5)", "n": len(data),
           "cutoffs": list(S5.CUTOFFS), "weights": list(MIX_WEIGHTS), "rows": [], "pairs": [],
           "lengths": {}}
    for adsorbate, target in zip(R.ADSORBATES, R.TARGETS):
        out["lengths"][target] = {k: length_summary(v, data.dE[adsorbate])
                                  for k, v in lengths.items()}
    for k, s in out["lengths"][R.TARGETS[0]].items():
        log(f"  the row length at {k}: mean {s['mean']:.3g}, spread {s['sd']:.3g} "
            f"({100 * s['spread_over_mean']:.0f} per cent of the mean)")
    for design in ("random", "metal-out", "motif-out"):
        for adsorbate, target in zip(R.ADSORBATES, R.TARGETS):
            y = data.dE[adsorbate]
            results = {}
            for arm in ARMS:
                res = R.cross_validate(f"site profile, {arm}", fams[arm], y, design, target,
                                       data, seed=args.seed, lams=LAMS)
                results[arm] = res
                n_tested = int(np.sum(np.isfinite(res.predicted) & np.isfinite(y)))
                out["rows"].append(_row("sac", "site profile (M5)", target, arm, res.q2,
                                        res.rmse, res.mae, n_tested,
                                        extra={"design": design,
                                               "chosen": [list(c) for c in res.chosen]}))
                log(f"  {design} {target} {arm}: q2 {res.q2:+.3f}")
            for arm in ARMS[1:]:
                d, lo, hi = R.bootstrap_difference(y, results[arm].predicted,
                                                   results[ARM_OUT].predicted,
                                                   runs=args.bootstrap, seed=args.seed)
                out["pairs"].append({"design": design, "target": target, "a": arm, "b": ARM_OUT,
                                     "diff": d, "low": lo, "high": hi})
                log(f"    {arm} against the reference: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return out


# --------------------------------------------------------------------------- the hop sample (BT7)


def hops_run(args) -> dict:
    """Three descriptions of the same hops: counts, the two end sites, the hop environment."""
    from cmfqsar.hops import hop_count_features, hop_types
    from cmfqsar.localenv import LOCAL_SPEC, local_features, paired_interval
    from run_hops import fieldset_for, site_pair_features
    from run_litraj import PREPARED, load_prepared, nested_result

    hs, doc = load_prepared(args.prepared or PREPARED, args.max_structures)
    fs = fieldset_for(hs, "file")
    crystals = hs.crystals(fs)
    t = hs.targets()
    y = np.asarray(t.y, float)
    log(f"  {len(t)} hops in {t.n_measured_molecules} structures, barriers mean {y.mean():.3f}, "
        f"standard deviation {y.std(ddof=1):.3f} eV")
    t0 = time.time()
    envs = hs.environments(crystals, rows=t.atom, rho=args.rho, r_max=args.r_max)
    log(f"  the hop environments at rho {args.rho:g}, r {args.r_max:g} A: "
        f"{time.time() - t0:.0f} s")
    types = hop_types(envs, min_atoms=20)
    descriptors = {
        "counts about the midpoint": {"counts": (hop_count_features(envs, types), 1, None)},
        "the two end sites (M5)": {f"r={args.site_r:g} A":
                                   (site_pair_features(hs, crystals, t.atom, args.site_r), 1,
                                    None)},
        "the hop environment (BT7)": {f"rho={args.rho:g}":
                                      (local_features(envs), LOCAL_SPEC.n_bins, None)},
    }
    out = {"series": "hops", "n_samples": len(t), "n_structures": len(hs.structures),
           "folds": args.folds, "weights": list(MIX_WEIGHTS), "rho": args.rho,
           "r_max": args.r_max, "site_r": args.site_r, "y_sd": float(y.std(ddof=1)),
           "rows": [], "pairs": [], "lengths": {}}
    for descriptor, settings in descriptors.items():
        fams, lengths = arm_families(settings)
        out["lengths"][descriptor] = {k: length_summary(v, y) for k, v in lengths.items()}
        for k, s in out["lengths"][descriptor].items():
            log(f"  {descriptor}, the row length at {k}: spread "
                f"{100 * s['spread_over_mean']:.0f} per cent of the mean; with the barrier "
                f"Pearson {s['pearson']:+.3f}, Spearman {s['spearman']:+.3f}")
        results = {}
        for arm in ARMS:
            res, picked = nested_result(fams[arm], t, args.folds, args.seed,
                                        f"{descriptor}, {arm}", lams=LAMS)
            results[arm] = res
            out["rows"].append(_row("hops", descriptor, "barrier", arm, res.q2, res.error,
                                    res.mae, res.n_samples, picked))
            log(f"  {descriptor}, {arm}: q2 {res.q2:+.3f}, error {res.error:.3f} eV, "
                f"mean absolute error {res.mae:.3f} eV")
        for arm in ARMS[1:]:
            d, lo, hi = paired_interval(results[arm], results[ARM_OUT], t, runs=args.bootstrap,
                                        seed=args.seed)
            out["pairs"].append({"descriptor": descriptor, "a": arm, "b": ARM_OUT, "diff": d,
                                 "low": lo, "high": hi})
            log(f"    {arm} against the reference: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return out


# ------------------------------------------------------------------------ the defect series (M14)


def defects_run(args) -> dict:
    """The whole-cell autocorrelation of the 2DMD defect series, on the folds of item M14."""
    import run_2dmd as D
    from cmfqsar import periodic

    out = {"series": "defects", "descriptor": "whole-cell autocorrelation", "limit": args.limit,
           "sigmas": list(D.AC_SIGMAS), "r_max": D.AC_R_MAX, "weights": list(MIX_WEIGHTS),
           "rows": [], "pairs": [], "lengths": {}, "materials": list(args.materials),
           "lams": list(D.LAMS), "inner": D.INNER, "outer": D.OUTER}
    for material in args.materials:
        prep = D.prepare(material)
        series = D.Series(prep, D.pick(prep, args.limit))
        y = np.asarray(series.y, float)
        out["n_samples"] = out.get("n_samples", 0) + len(y)
        table = series.whole()
        settings = {}
        for sigma in D.AC_SIGMAS:
            t0 = time.time()
            spec = periodic.PeriodicSpec(r_max=D.AC_R_MAX, step=sigma, sigma=sigma,
                                         taper=D.AC_TAPER, background=True)
            F = periodic.periodic_feature_matrix(table.molecules, spec,
                                                 n_fields=table.weights.shape[1])
            blocks = periodic.periodic_block_sizes(spec, table.weights.shape[1])
            settings[f"sigma {sigma:g}"] = (F, spec.n_bins, blocks)
            log(f"  {material} sigma {sigma:g}: {F.shape[0]} cells, {F.shape[1]} channels, "
                f"{time.time() - t0:.0f} s")
        fams, lengths = arm_families(settings)
        del settings                        # the feature matrices are no longer needed
        out["lengths"][material] = {k: length_summary(v, y) for k, v in lengths.items()}
        for k, s in out["lengths"][material].items():
            log(f"  {material}, the row length at {k}: spread "
                f"{100 * s['spread_over_mean']:.1f} per cent of the mean; with the energy "
                f"Pearson {s['pearson']:+.3f}, Spearman {s['spearman']:+.3f}")
        splits = D.folds(len(y))
        results = {}
        for arm in ARMS:
            t0 = time.time()
            res = D.evaluate(fams[arm], y, series.n_defects, series.classes, splits)
            results[arm] = res
            out["rows"].append(_row("defects", f"whole-cell autocorrelation, {material}",
                                    "formation energy", arm, res["r2"], res["rmse"], res["mae"],
                                    res["n_tested"],
                                    extra={"material": material,
                                           "mae_site": res["mae_site"],
                                           "mae_site_unseen": res["mae_site_unseen"],
                                           "settings": sorted({f["setting"]
                                                               for f in res["folds"]})}))
            log(f"  {material} {arm}: R2 {res['r2']:+.4f}, {1000 * res['mae_site']:.1f} meV per "
                f"site ({time.time() - t0:.0f} s)")
        for arm in ARMS[1:]:
            d, lo, hi = paired_q2_difference(y, np.asarray(results[arm]["predictions"], float),
                                             np.asarray(results[ARM_OUT]["predictions"], float),
                                             runs=args.bootstrap, seed=args.seed)
            out["pairs"].append({"material": material, "a": arm, "b": ARM_OUT, "diff": d,
                                 "low": lo, "high": hi})
            log(f"    {arm} against the reference: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        del fams, results
    return out


# ------------------------------------------------------------- a molecular set: the guests of beta-CD


def molecules_run(args) -> dict:
    """The beta-cyclodextrin guests: the case the division was written for.

    The guests differ in size by a factor of several, so the length of a row is close to how
    large the guest is. Rule 1's own reference here is the invariant kernel of
    `benchmarks/run_cyclodextrin.py`, which is the first arm.
    """
    from sklearn.model_selection import KFold

    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import FieldSet, build_fields
    from cmfqsar.invariant import AutocorrelationSpec, feature_matrix
    import run_cyclodextrin as C

    # The family here is the width of the autocorrelation, not the field width alpha that
    # `benchmarks/run_cyclodextrin.py` searches over for this kernel: the autocorrelation is a
    # sum over pairs of atom weights, and the atom weights do not depend on alpha, so all six of
    # that script's alphas give one and the same invariant kernel. The middle setting below is
    # the default the stored table was measured at.
    AC_SETTINGS = (0.5, 0.75, 1.0)

    path = os.path.join(ROOT, "examples", "cyclodextrin", "beta_cyclodextrin.csv")
    ds = load_dataset(path)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y = y[ok]
    if args.limit and len(mols) > args.limit:
        keep = np.linspace(0, len(mols) - 1, args.limit).astype(int)
        mols = [mols[i] for i in keep]
        names = [names[i] for i in keep]
        y = y[keep]
    heavy = np.array([m.GetNumHeavyAtoms() for m in mols], float)
    log(f"  {len(mols)} guests, {heavy.min():.0f} to {heavy.max():.0f} heavy atoms "
        f"(median {np.median(heavy):.0f})")
    fs = FieldSet.create(physchem=C.FIELDS, alpha=0.5, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    settings = {}
    for width in AC_SETTINGS:
        t0 = time.time()
        spec = AutocorrelationSpec(step=width, sigma=width)
        X = feature_matrix(mf, spec)
        settings[f"width {width:g} A"] = (X, spec.n_bins, None)
        log(f"  width {width:g} A: {X.shape[1]} channels, {time.time() - t0:.0f} s")
    fams, lengths = arm_families(settings)
    out = {"series": "molecules", "descriptor": "whole-molecule autocorrelation",
           "file": "examples/cyclodextrin/beta_cyclodextrin.csv", "n": len(mols),
           "widths": list(AC_SETTINGS), "alpha": 0.5, "weights": list(MIX_WEIGHTS),
           "folds": args.folds,
           "heavy_atoms": {"min": float(heavy.min()), "max": float(heavy.max()),
                           "median": float(np.median(heavy))},
           "rows": [], "pairs": [], "lengths": {k: length_summary(v, y)
                                                for k, v in lengths.items()}}
    # how far the length is the size of the guest, which is what makes this series the case the
    # division was written for
    ell = lengths[f"width {AC_SETTINGS[1]:g} A"]
    out["length_against_heavy_atoms"] = float(np.corrcoef(ell, heavy)[0, 1])
    log(f"  the row length against the heavy-atom count: "
        f"{out['length_against_heavy_atoms']:+.3f}")
    for k, s in out["lengths"].items():
        log(f"  the row length at {k}: spread {100 * s['spread_over_mean']:.0f} per cent of the "
            f"mean; with the constant Pearson {s['pearson']:+.3f}, Spearman {s['spearman']:+.3f}")
    splits = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(y))
    results = {}
    for arm in ARMS:
        res = nested_on_splits(fams[arm], y, splits, lams=LAMS, seed=args.seed)
        results[arm] = res
        out["rows"].append(_row("molecules", "whole-molecule autocorrelation", "log K", arm,
                                res["q2"], res["rmse"], res["mae"], res["n"], res["chosen"]))
        log(f"  {arm}: q2 {res['q2']:+.3f}, error {res['rmse']:.3f}")
    for arm in ARMS[1:]:
        d, lo, hi = paired_q2_difference(y, results[arm]["predicted"],
                                         results[ARM_OUT]["predicted"], runs=args.bootstrap,
                                         seed=args.seed)
        out["pairs"].append({"a": arm, "b": ARM_OUT, "diff": d, "low": lo, "high": hi})
        log(f"    {arm} against the reference: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return out


# ----------------------------------------------------------------------------- the table

STAGES = ("lithium", "sac", "hops", "defects", "molecules")


def _fmt(x, digits=3, sign=True) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def report(out: str = "") -> str:
    """The table of every stored stage."""
    lines = ["# What the row length is worth (plan item M15)", "",
             "The invariant kernel divides every feature row by its own length. These are the "
             "four ways of treating that one number, each with the setting and the ridge chosen "
             "inside the training part, on the samples and folds of the benchmark each series "
             "comes from. The difference is in q2 against the first arm, which is what the "
             "program does today, with the bootstrap interval of rule 4.", ""]
    docs = {}
    for stage in STAGES:
        path = result(f"{stage}.json")
        if os.path.isfile(path):
            docs[stage] = jload(path)
    if docs:
        lines += _verdict_table(docs)
    found = False
    for stage in STAGES:
        path = result(f"{stage}.json")
        if not os.path.isfile(path):
            continue
        found = True
        doc = jload(path)
        lines.append(f"## {stage}")
        lines.append("")
        head = doc.get("descriptor", "")
        n = doc.get("n") or doc.get("n_samples")
        lines.append(f"{head}, {n} samples." if head else f"{n} samples.")
        lines.append("")
        pairs = {}
        for p in doc.get("pairs", []):
            where = p.get("descriptor") or p.get("material") or p.get("design", "")
            pairs[(where, p.get("target", ""), p["a"])] = p
        # the defect series is fitted to four decimals by its own benchmark, and its error is
        # read per defect site rather than per cell, so both are carried here
        digits = 4 if stage == "defects" else 3
        per_site = any("mae_site" in r for r in doc.get("rows", []))
        head_row = "| description | target | arm | q2 | error "
        head_row += "| meV per site " if per_site else ""
        lines.append(head_row + "| difference against the first arm |")
        lines.append("|---|---|---|---|" + ("---|" if per_site else "") + "---|---|")
        for r in doc.get("rows", []):
            group = r.get("design") or r.get("material") or r["descriptor"]
            target = r.get("target", "")
            if r["arm"] == ARM_OUT:
                diff = "the reference"
            else:
                p = (pairs.get((group, target, r["arm"]))
                     or pairs.get((group, "", r["arm"]))
                     or pairs.get((r["descriptor"], target, r["arm"]))
                     or pairs.get(("", target, r["arm"]))
                     or pairs.get(("", "", r["arm"])))
                if p is None:
                    diff = "--"
                elif np.isfinite(p["low"]):
                    diff = (f"{_fmt(p['diff'], digits)} [{_fmt(p['low'], digits)}, "
                            f"{_fmt(p['high'], digits)}]")
                else:
                    # fewer than four groups to draw from, so rule 4 can give no interval
                    diff = f"{_fmt(p['diff'], digits)}, no interval"
            where = r["descriptor"]
            if r.get("design"):
                where = f"{where}, {r['design']}"
            site = (f"| {1000 * r['mae_site']:.1f} " if "mae_site" in r
                    else ("| -- " if per_site else ""))
            lines.append(f"| {where} | {target} | {r['arm']} | {_fmt(r['q2'], digits)} | "
                         f"{_fmt(r['rmse'], 3, sign=False)} {site}| {diff} |")
        lines.append("")
        lines += _length_table(doc)
    if not found:
        lines.append("Nothing is stored yet: run the stages of `benchmarks/run_rowlength.py`.")
        lines.append("")
    text = "\n".join(lines) + "\n"
    path = out or result("table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def _flat_lengths(doc: dict) -> Dict[str, dict]:
    """The length summaries of a stage, however deep its own grouping goes."""
    flat = {}
    for key, value in (doc.get("lengths") or {}).items():
        if isinstance(value, dict) and "mean" in value:
            flat[key] = value
        elif isinstance(value, dict):
            for inner, v in value.items():
                flat[f"{key}, {inner}"] = v
    return flat


def _verdict_table(docs: Dict[str, dict]) -> List[str]:
    """One row per series: how far the length goes with the target, and what keeping it did.

    The last two columns are the point of the measurement. A comparison counts as better or worse
    only when the interval of rule 4 is on one side of zero; everything else is inside the
    interval and is no result.
    """
    lines = ["## What it says", "",
             "| series | the length against the target: Pearson | comparisons | the length kept:"
             " better / worse / inside the interval | the length as an extra channel: better / "
             "worse / inside | the largest gain from keeping it |", "|---|---|---|---|---|---|"]
    for stage, doc in docs.items():
        cors = [s["pearson"] for s in _flat_lengths(doc).values() if np.isfinite(s["pearson"])]
        span = (f"{min(cors):+.2f} to {max(cors):+.2f}" if len(cors) > 1
                else (f"{cors[0]:+.2f}" if cors else "--"))
        counts = {}
        for arm in (ARM_KEPT, ARM_PLUS):
            ps = [p for p in doc.get("pairs", []) if p["a"] == arm]
            known = [p for p in ps if np.isfinite(p["low"])]
            up = sum(1 for p in known if p["low"] > 0)
            down = sum(1 for p in known if p["high"] < 0)
            counts[arm] = (up, down, len(known) - up - down, len(ps) - len(known))
        best = max((p for p in doc.get("pairs", []) if p["a"] == ARM_KEPT
                    and np.isfinite(p["low"])), key=lambda p: p["diff"], default=None)
        gain = (f"{_fmt(best['diff'])} [{_fmt(best['low'])}, {_fmt(best['high'])}]" if best
                else "no interval")
        cells = []
        for arm in (ARM_KEPT, ARM_PLUS):
            up, down, inside, unknown = counts[arm]
            cell = f"{up} / {down} / {inside}"
            if unknown:
                cell += f" ({unknown} with no interval)"
            cells.append(cell)
        n = len([p for p in doc.get("pairs", []) if p["a"] == ARM_KEPT])
        lines.append(f"| {stage} | {span} | {n} | {cells[0]} | {cells[1]} | {gain} |")
    lines.append("")
    # where q2 is already near one, the gain in q2 cannot show much and the error does
    for stage, doc in docs.items():
        site = {}
        for r in doc.get("rows", []):
            if "mae_site" in r:
                site.setdefault(r.get("material", stage), {})[r["arm"]] = r["mae_site"]
        if not site:
            continue
        refs = [v[ARM_OUT] for v in site.values() if ARM_OUT in v]
        best = max((r["q2"] for r in doc["rows"] if r["arm"] == ARM_OUT), default=float("nan"))
        parts = [f"from {1000 * v[ARM_OUT]:.1f} to {1000 * v[ARM_KEPT]:.1f} meV on {k}"
                 for k, v in site.items() if ARM_OUT in v and ARM_KEPT in v]
        if parts and refs:
            lines.append(f"On the {stage} series q2 is already {best:+.4f}, so it cannot show "
                         f"much. The error per defect site is what moves: " +
                         ", ".join(parts) + ", when the length is kept.")
            lines.append("")
    return lines


def _length_table(doc: dict) -> List[str]:
    """What the length itself looks like on this series."""
    lengths = doc.get("lengths") or {}
    if not lengths:
        return []
    lines = ["| setting | spread of the length, per cent of its mean | with the target: Pearson "
             "| Spearman |", "|---|---|---|---|"]
    flat = {}
    for key, value in lengths.items():
        if isinstance(value, dict) and "mean" in value:
            flat[key] = value
        elif isinstance(value, dict):
            for inner, v in value.items():
                flat[f"{key}, {inner}"] = v
    for key, s in flat.items():
        lines.append(f"| {key} | {100 * s['spread_over_mean']:.0f} | {_fmt(s['pearson'])} | "
                     f"{_fmt(s['spearman'])} |")
    lines.append("")
    return lines


def write_provenance(args, stage: str, seconds: float) -> None:
    """One record per stage of what produced it (rule 6, plan item G6)."""
    doc = {"stage": stage, "argv": sys.argv[1:], "seed": args.seed, "lams": list(LAMS),
           "weights": list(MIX_WEIGHTS), "bootstrap": args.bootstrap,
           "written": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round(seconds / 60.0, 1)}
    jdump(doc, result(f"run_{stage}.json"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for stage in STAGES:
        p.add_argument(f"--{stage}", action="store_true", help=f"the {stage} series alone")
    p.add_argument("--report", action="store_true", help="rebuild the table from stored results")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    p.add_argument("--folds", type=int, default=10, help="outer folds where the series has no "
                                                         "grouping of its own")
    p.add_argument("--limit", type=int, default=2000, help="cells per material in the defect "
                                                           "series, guests in the molecular one")
    p.add_argument("--materials", nargs="+", default=["MoS2", "WSe2"])
    p.add_argument("--prepared", default="", help="the prepared hop sample")
    p.add_argument("--max-structures", type=int, default=0)
    p.add_argument("--rho", type=float, default=0.25, help="envelope width of the hop environment")
    p.add_argument("--r-max", type=float, default=6.0)
    p.add_argument("--site-r", type=float, default=6.0)
    args = p.parse_args()
    wanted = [s for s in STAGES if getattr(args, s)]
    if args.report and not wanted:
        print(report())
        return 0
    if not wanted:
        wanted = list(STAGES)
    for stage in wanted:
        log(f"=== {stage} ===")
        t0 = time.time()
        doc = globals()[f"{stage}_run"](args)
        jdump(doc, result(f"{stage}.json"))
        write_provenance(args, stage, time.time() - t0)
        log(f"  {stage}: {(time.time() - t0) / 60:.1f} minutes")
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
