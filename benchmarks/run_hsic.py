#!/usr/bin/env python
"""
Does the activity depend on the fields beyond a reference?  The conditional HSIC test, measured.

:func:`cmfqsar.hsic.hsic_test` asks whether the activity depends on the field kernel, and with
strata of a reference z whether it depends on the fields *beyond* z, by permuting the activities
within the strata. For a count the strata are its values and the null is exact. This script checks
that claim, measures what binning a continuous reference costs, and compares the test with the
usual answer: a cross-validated comparison of a model on z with a model on z and the fields.

Stage ``calibration`` (the eight Sutherland sets, the joint kernel at alpha = 0.3)
    Synthetic activities on the real molecules. Under the null, y = g(z) + noise with
    g(u) = u + u^2 / 2 of the standardised reference, half of the variance from g: the activity
    depends on the reference only, and the fields see the reference. Under the alternative a field
    direction is added that the reference does not explain (the first principal component of the
    kernel centred within the strata), carrying 5 or 15 % of the variance. References: the number
    of heavy atoms (exact strata) and Crippen logP cut into 5 or 10 quantile bins. Rules compared:
    the plain test; the conditional test; for logP, the conditional test with a straight line in
    logP removed within each bin after every permutation (the repair of the binned test); the
    plain test on the residual of a straight-line fit of y on z (the naive repair); and the model
    comparison, called positive when the paired 95 %
    bootstrap interval of the gain in squared error excludes zero (on the six sets of at most
    150 molecules, because its cost grows with the cube of n).

Stage ``real``
    The measured activities of the eight sets, with both references: the p-values and the
    alignment of the three tests, beside the cross-validated gain of adding the fields to a model
    on the reference.

Stage ``ether``
    The 1,048 ethers of item EC7 with the binding energy of Li+, Na+ and Mg2+. The fields are known
    to beat a model on the number of oxygens that touch the ion (``benchmarks/results/ether``), so
    the test given that count should reject; it is also given the oxygen count of the formula.
    A null calibration on these molecules with a synthetic activity of the contact count.

Usage
-----
    python benchmarks/run_hsic.py                   # every stage
    python benchmarks/run_hsic.py --stage real
    python benchmarks/run_hsic.py --report
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
from rdkit.Chem import Crippen  # noqa: E402

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.hsic import centre_within, hsic_test, strata_of  # noqa: E402
from cmfqsar.model import cv_predict  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "hsic")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
ALPHA = 0.3
LAMS = (1e-2, 0.1, 1.0)
GAMMAS = (0.1, 1.0)
LEVEL = 0.05
SHARES = (0.0, 0.05, 0.15)       # the share of the variance the added field direction carries
TEST_REPLICATES = 400
CV_REPLICATES = 40
#: the model comparison grows with the cube of n; on COX2 and DHFR it would take hours
CV_MAX_N = 150
PERMUTATIONS = 199
REFERENCES = {"heavy atoms": ("heavy", None), "logP, 5 bins": ("logp", 5),
              "logP, 10 bins": ("logp", 10)}


# --------------------------------------------------------------------------- pieces

def g(u):
    return u + 0.5 * u * u


def standard(v):
    v = np.asarray(v, float)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def residual_on_line(y, z):
    A = np.c_[np.ones(len(z)), z]
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def field_direction(K, strata):
    """A direction of the field that the strata do not explain: the leading eigenvector of the
    kernel centred within the strata, standardised."""
    vals, vecs = np.linalg.eigh(centre_within(K, strata))
    return standard(vecs[:, -1])


def synthetic(rng, gz, s, share):
    """y = g(z) + b s + noise, with g(z) and the noise of equal variance and b s carrying `share`."""
    b = np.sqrt(2.0 * share / (1.0 - share)) if share > 0 else 0.0
    return gz + b * s + rng.randn(len(gz))


def gauss_kernel(z, gamma):
    zs = standard(z)
    return np.exp(-gamma * (zs[:, None] - zs[None, :]) ** 2)


def cv_errors(K, y, seeds):
    """Out-of-fold squared errors at the ridge that is best on average over the fold seeds."""
    best = None
    for lam in LAMS:
        e = np.mean([(y - cv_predict(K, y, "krr", {"lam": lam}, folds=10, seed=s)) ** 2
                     for s in seeds], axis=0)
        if best is None or e.mean() < best.mean():
            best = e
    return best


def model_comparison(K_field, z, y, seeds=(0, 1, 2), draws=1000, seed=0):
    """The gain in squared error of the model on z and the fields over the model on z alone,
    with a paired bootstrap interval over the molecules. Each model takes its best width."""
    Kf = K_field / np.mean(np.diag(K_field))
    ref = min((cv_errors(gauss_kernel(z, gm), y, seeds) for gm in GAMMAS), key=np.mean)
    joint = min((cv_errors(gauss_kernel(z, gm) + Kf, y, seeds) for gm in GAMMAS), key=np.mean)
    d = ref - joint
    rng = np.random.RandomState(seed)
    boot = d[rng.randint(0, len(d), (draws, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    var = float(np.var(y))
    return {"gain_q2": float(d.mean() / var), "low": float(lo / var), "high": float(hi / var),
            "positive": bool(lo > 0)}


def reference_values(kind, mols):
    if kind == "heavy":
        return np.array([m.GetNumHeavyAtoms() for m in mols], float)
    if kind == "logp":
        return np.array([Crippen.MolLogP(m) for m in mols], float)
    raise ValueError(kind)


def load_set(name):
    from run_conformal import kernels
    ds = load_dataset(os.path.join(DATA, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    return mols, y[keep], dict(kernels(mols, names, (ALPHA,)))[ALPHA]


# --------------------------------------------------------------------------- stage: calibration

def calibration_set(name):
    mols, _y, K = load_set(name)
    out = {"set": name, "n": len(mols), "references": {}}
    for ref, (kind, bins) in REFERENCES.items():
        z = reference_values(kind, mols)
        s, exact = strata_of(z, bins=bins)
        gz = standard(g(standard(z)))
        direction = field_direction(K, s)
        rec = {"exact": exact, "strata": int(s.max() + 1), "shares": {}}
        for share in SHARES:
            rng = np.random.RandomState(int(1000 * share) + 17)
            counts = {"plain": 0, "conditional": 0, "within-bin line": 0, "residual": 0}
            cv_pos, cv_done = 0, 0
            for t in range(TEST_REPLICATES):
                y = synthetic(rng, gz, direction, share)
                counts["plain"] += hsic_test(K, y, permutations=PERMUTATIONS,
                                             seed=t).p_value <= LEVEL
                counts["conditional"] += hsic_test(K, y, strata=s, exact=exact,
                                                   permutations=PERMUTATIONS,
                                                   seed=t).p_value <= LEVEL
                counts["residual"] += hsic_test(K, residual_on_line(y, z),
                                                permutations=PERMUTATIONS,
                                                seed=t).p_value <= LEVEL
                if not exact:
                    counts["within-bin line"] += hsic_test(
                        K, y, strata=s, exact=exact, permutations=PERMUTATIONS, seed=t,
                        covariate=z).p_value <= LEVEL
                if t < CV_REPLICATES and kind == "heavy" and len(mols) <= CV_MAX_N:
                    cv_pos += model_comparison(K, z, y, seed=t)["positive"]
                    cv_done += 1
            if exact:
                counts.pop("within-bin line")
            rec["shares"][str(share)] = {
                **{k: v / TEST_REPLICATES for k, v in counts.items()},
                "model comparison": (cv_pos / cv_done) if cv_done else None,
                "replicates": TEST_REPLICATES, "cv_replicates": cv_done}
        out["references"][ref] = rec
        print(f"  {name}, {ref}: " + "; ".join(
            f"share {sh}: " + ", ".join(f"{k} {v:.2f}" for k, v in r.items()
                                       if isinstance(v, float))
            for sh, r in rec["shares"].items()), flush=True)
    return out


# --------------------------------------------------------------------------- stage: real

def real_set(name):
    mols, y, K = load_set(name)
    out = {"set": name, "n": len(mols), "activity_sd": float(np.std(y)),
           "plain": _res(hsic_test(K, y, permutations=999)), "references": {}}
    for ref, (kind, bins) in REFERENCES.items():
        z = reference_values(kind, mols)
        s, exact = strata_of(z, bins=bins)
        out["references"][ref] = {
            "conditional": _res(hsic_test(K, y, strata=s, exact=exact, permutations=999)),
            "within-bin line": (None if exact else
                                _res(hsic_test(K, y, strata=s, exact=exact, permutations=999,
                                               covariate=z))),
            "residual": _res(hsic_test(K, residual_on_line(y, z), permutations=999)),
            "model comparison": model_comparison(K, z, y, seeds=(0, 1, 2, 3, 4)),
            "reference_alignment": float(_alignment(K, z)),
        }
    print(f"  {name}: plain p {out['plain']['p_value']:.3g}; " + "; ".join(
        f"{r}: p {v['conditional']['p_value']:.3g}, gain {v['model comparison']['gain_q2']:+.3f}"
        for r, v in out["references"].items()), flush=True)
    return out


def _res(r):
    return {"p_value": r.p_value, "alignment": r.alignment, "z": r.z, "strata": r.strata,
            "informative": r.informative, "exact": r.exact}


def _alignment(K, z):
    """How much of the field the reference explains: the centred alignment of K with a Gaussian
    kernel of z."""
    L = gauss_kernel(z, 1.0)
    Kc, Lc = centre_within(K), centre_within(L)
    return np.sum(Kc * Lc) / np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))


# --------------------------------------------------------------------------- stage: ether

def ether_stage():
    import run_ether as RE
    mols, names, rows = RE.load()
    with open(os.path.join(HERE, "results", "ether", "contacts.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    if list(stored["names"]) != list(names):
        raise SystemExit("the stored contact counts are not in the order of the ether file")
    contacts = np.array([c[0] for c in stored["contacts_relaxed"]], float)
    o_count = np.array([float(r["o_count"]) for r in rows])
    t0 = time.time()
    K_all, _ = RE.kernel_of(RE.field_features(mols, names, RE.FIELDS, (), 0.5),
                            np.arange(len(mols)))
    print(f"  ether kernel {K_all.shape} in {time.time() - t0:.0f} s", flush=True)
    out = {"n": len(mols), "ions": {}, "calibration": {}}
    for ion in RE.ION_ORDER:
        _probe, prop, label = RE.IONS[ion]
        y_all = RE.target(rows, prop)
        ok = np.flatnonzero(np.isfinite(y_all))
        K = K_all[np.ix_(ok, ok)]
        y = y_all[ok]
        rec = {"label": label, "n": int(len(ok)),
               "plain": _res(hsic_test(K, y, permutations=999))}
        for ref, z in (("contacts at the relaxed ion", contacts[ok]),
                       ("oxygens in the formula", o_count[ok])):
            s, exact = strata_of(z)
            rec[ref] = _res(hsic_test(K, y, strata=s, exact=exact, permutations=999))
        des = stored["ions"][ion]["designs"]["random"]
        rec["stored"] = {k: v for k, v in des["paired"].items()
                         if k.startswith("contacts at the relaxed ion, Gaussian minus fields")}
        out["ions"][ion] = rec
        print(f"  {label}: plain p {rec['plain']['p_value']:.3g}, given contacts p "
              f"{rec['contacts at the relaxed ion']['p_value']:.3g} (alignment "
              f"{rec['contacts at the relaxed ion']['alignment']:.3f})", flush=True)
    # a null on these molecules: an activity of the contact count alone
    ok = np.flatnonzero(np.isfinite(RE.target(rows, RE.IONS["li"][1])))
    K = K_all[np.ix_(ok, ok)]
    z = contacts[ok]
    s, _ = strata_of(z)
    gz = standard(g(standard(z)))
    rng = np.random.RandomState(5)
    hits = {"plain": 0, "conditional": 0, "residual": 0}
    reps = 100
    for t in range(reps):
        y = gz + rng.randn(len(z))
        hits["plain"] += hsic_test(K, y, permutations=PERMUTATIONS, seed=t).p_value <= LEVEL
        hits["conditional"] += hsic_test(K, y, strata=s, permutations=PERMUTATIONS,
                                         seed=t).p_value <= LEVEL
        hits["residual"] += hsic_test(K, residual_on_line(y, z), permutations=PERMUTATIONS,
                                      seed=t).p_value <= LEVEL
    out["calibration"] = {k: v / reps for k, v in hits.items()}
    out["calibration"]["replicates"] = reps
    print(f"  ether null: {out['calibration']}", flush=True)
    return out


# --------------------------------------------------------------------------- the report

def _binom_upper(k, n):
    """The upper end of the exact 95 % interval of a binomial share (Clopper-Pearson)."""
    from scipy.stats import beta
    return float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0


def report(results=RESULTS):
    L = ["# The conditional HSIC test: does the activity depend on the fields beyond a reference?",
         ""]
    path = os.path.join(results, "calibration.json")
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))["sets"]
        L += ["## Calibration and power on synthetic activities", "",
              f"The {len(data)} Sutherland sets, the joint field kernel at alpha = {ALPHA}. "
              "y = g(z) + noise with half of the variance from g, plus a field direction the "
              "reference does not explain carrying the share given. At share 0 the activity "
              "depends on the reference only, so every rejection is a false positive and a "
              f"calibrated test rejects {LEVEL:.0%} of the time. Rejections at the 5 % level, "
              f"pooled over the sets ({TEST_REPLICATES} replicates per set; the model comparison "
              f"{CV_REPLICATES} per set, for the heavy-atom count and the {sum(d['n'] <= CV_MAX_N for d in data)} "
              f"sets of at most {CV_MAX_N} molecules only).", "",
              "| reference | strata | share of the field direction | plain test | "
              "conditional test | conditional test, line removed within bins | "
              "plain test on the straight-line residual | model comparison |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for ref in REFERENCES:
            exact = data[0]["references"][ref]["exact"]
            for share in SHARES:
                cells = []
                for key in ("plain", "conditional", "within-bin line", "residual",
                            "model comparison"):
                    v = [d["references"][ref]["shares"][str(share)].get(key) for d in data]
                    v = [x for x in v if x is not None]
                    if not v:
                        cells.append("--")
                        continue
                    m = float(np.mean(v))
                    flag = ""
                    if share == 0.0:
                        n_rep = TEST_REPLICATES if key != "model comparison" else CV_REPLICATES
                        k = int(round(m * n_rep * len(v)))
                        if _binom_upper(k, n_rep * len(v)) < LEVEL:
                            flag = ""
                        elif m > LEVEL + 2 * np.sqrt(LEVEL * (1 - LEVEL) / (n_rep * len(v))):
                            flag = " **too many**"
                    cells.append(f"{m:.1%}{flag}")
                L.append(f"| {ref} | {'values' if exact else 'bins'} | {share:.0%} | "
                         + " | ".join(cells) + " |")
        L += ["", "**too many** marks a false-positive rate more than two binomial standard "
              "errors above 5 %.", ""]

    path = os.path.join(results, "real.json")
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))["sets"]
        L += ["## The measured activities", "",
              "p-values from 999 permutations. The model comparison is the gain in q2 of adding "
              "the fields to a Gaussian-kernel model on the reference, out-of-fold over five fold "
              "seeds, with its paired 95 % bootstrap interval over the molecules. *Reference "
              "alignment* is how much of the field the reference itself explains.", "",
              "| set | n | p, fields | reference | reference alignment | p, beyond the reference "
              "| alignment beyond it | p, beyond it with the line removed within bins | "
              "p, straight-line residual | gain in q2 of adding the fields "
              "|", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        agree = {r: [0, 0] for r in REFERENCES}
        for d in data:
            for ref in REFERENCES:
                r = d["references"][ref]
                mc = r["model comparison"]
                used = r["within-bin line"] or r["conditional"]
                test_pos = used["p_value"] <= LEVEL
                wb = r["within-bin line"]
                wb_cell = "--" if wb is None else f"{wb['p_value']:.3g}"
                agree[ref][0] += test_pos == mc["positive"]
                agree[ref][1] += 1
                L.append(f"| {d['set']} | {d['n']} | {d['plain']['p_value']:.3g} | {ref} | "
                         f"{r['reference_alignment']:.2f} | {r['conditional']['p_value']:.3g} | "
                         f"{r['conditional']['alignment']:.3f} | {wb_cell} | "
                         f"{r['residual']['p_value']:.3g} | "
                         f"{mc['gain_q2']:+.3f} [{mc['low']:+.3f}, {mc['high']:+.3f}] |")
        L += ["", "Agreement of the conditional test at 5 % (for logP, the version with the line "
              "removed within bins) with the model comparison (positive "
              "when its interval excludes zero): " + ", ".join(
                  f"{ref} {a} of {n}" for ref, (a, n) in agree.items()) + ".", ""]

    path = os.path.join(results, "ether.json")
    if os.path.exists(path):
        d = json.load(open(path, encoding="utf-8"))
        L += ["## The ethers of item EC7", "",
              "The alignment-free field kernel at alpha = 0.5 of the 1,050 ethers in the "
              "conformation of the lithium complex. The stored comparison is the gain of the "
              "field model over the Gaussian model on the contacts at the relaxed ion, mean "
              "absolute error in eV, random folds (`benchmarks/results/ether/contacts.json`).", "",
              "| ion | n | p, fields | p, beyond the contacts | alignment beyond the contacts | "
              "p, beyond the formula's oxygens | stored: fields better than the contacts by |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for ion, r in d["ions"].items():
            st = next(iter(r["stored"].values()), None)
            stc = (f"{st['mean']:+.3f} eV [{st['low']:+.3f}, {st['high']:+.3f}]" if st else "--")
            c = r["contacts at the relaxed ion"]
            o = r["oxygens in the formula"]
            L.append(f"| {r['label']} | {r['n']} | {r['plain']['p_value']:.3g} | "
                     f"{c['p_value']:.3g} | {c['alignment']:.3f} | {o['p_value']:.3g} | {stc} |")
        cal = d["calibration"]
        L += ["", f"A null on these molecules (the lithium set, an activity of the contact count "
              f"alone plus noise, {cal['replicates']} replicates): the plain test rejects "
              f"{cal['plain']:.0%}, the conditional test {cal['conditional']:.0%}, the plain test "
              f"on the straight-line residual {cal['residual']:.0%}.", ""]
    text = "\n".join(L) + "\n"
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["calibration", "real", "ether", "all"], default="all")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args()
    if a.report:
        print(report(a.results))
        return
    os.makedirs(a.results, exist_ok=True)

    def save(name, payload):
        with open(os.path.join(a.results, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)

    for stage, fn in (("calibration", calibration_set), ("real", real_set)):
        if a.stage in (stage, "all"):
            t0 = time.time()
            save(stage, {"sets": [fn(s) for s in a.sets], "seconds": round(time.time() - t0, 1)})
    if a.stage in ("ether", "all"):
        t0 = time.time()
        out = ether_stage()
        out["seconds"] = round(time.time() - t0, 1)
        save("ether", out)
    print(report(a.results))


if __name__ == "__main__":
    main()
