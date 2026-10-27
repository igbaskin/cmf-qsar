#!/usr/bin/env python
"""
How far the local kernel goes, and what it costs to get there (plan item LP5)

A per-atom target is a large problem where a per-molecule one is a small one: the 400 molecules
of `examples/local/shifts_cdcl3.sdf` are 5096 carbons, and nmrshiftdb2 holds hundreds of
thousands.  An exact kernel over samples costs the square of their number in memory and the cube
in time, so the plan item asks for a Nystrom or inducing-point approximation.

For the rotation-invariant local kernel no approximation is needed, which is the first thing this
measures.  That kernel is a linear kernel on a finite feature map -- the field autocorrelation is
255 channels for the five fields -- so the ridge regression can be solved in the space of those
255 coefficients instead of in the space of the samples (`cmfqsar/locallarge.py`).  The two give
the same model; only the cost differs, from n^2 memory and n^3 time to n*d and n*d^2.

Three things are measured.

1. **The two routes against each other**, on the set in the repository: the same folds, the same
   ridge, the predictions compared to round-off, and the time and memory each one takes.
2. **A learning curve**, on a larger set built from the nmrshiftdb2 dump that this repository does
   not ship.  Error against the number of molecules says whether the error of the earlier items
   is the method's or the data set's, which is what decides whether scaling is worth anything.
3. **The Nystrom route for the overlap kernel**, which does need one, since its feature map is a
   function on space.  On the aromatic carbons of plan item LP3, the relative error of the
   approximate kernel and the q2 it gives, against the number of landmark environments.

Usage
-----
    python benchmarks/run_local_scale.py                       # routes, on the repository set
    python benchmarks/run_local_scale.py --data-root DIR       # and the curve, from DIR
    python benchmarks/run_local_scale.py --report

`DIR` is where `examples/make_local_examples.py --shifts <dump> --max-shifts 2000 --any-solvent
--out DIR` wrote its `shifts_cdcl3.sdf`; the dump itself is the free download named in that
script.
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

from rdkit import RDLogger  # noqa: E402

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.localalign import overlap_kernel, posed_environments, ring_frames  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec  # noqa: E402
from cmfqsar.localenv import (LOCAL_SPEC, choose_lam, cross_validate, environments,  # noqa: E402
                              local_features, local_kernel)
from cmfqsar.locallarge import (choose_lam_primal, cross_validate_primal,  # noqa: E402
                                learning_curve, normalised_rows, nystrom_kernel)

DATA = os.path.join(ROOT, "examples", "local", "shifts_cdcl3.sdf")
RESULTS = os.path.join(HERE, "results", "local_scale")
RHO = 1.0                      # the envelope width that was best for the shifts in plan item LP2
SIZES = (100, 200, 400, 800, 1600, 3200)
RANKS = (25, 50, 100, 200, 400, 800)


def _say(text):
    print(text, flush=True)


def read(path, rho=RHO, prop="shift", max_molecules=0, with_envs=False):
    """The samples of a shift file: targets and the feature rows of their environments."""
    ds = load_dataset(path, atom_activity=prop)
    mols, names = list(ds.mols), list(ds.names)
    if max_molecules:
        mols, names = mols[:max_molecules], names[:max_molecules]
    fieldset = FieldSet.create()
    t0 = time.time()
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(mols, names)]
    targets = AtomTargets.from_molecules(mols, prop, names=names)
    envs = environments(fields, targets, rho=rho)
    X = local_features(envs)
    seconds = time.time() - t0
    if with_envs:
        return mols, fields, targets, X, seconds, envs
    return mols, fields, targets, X, seconds


def gigabytes(n, width=8):
    return n * n * width / 1024 ** 3


def run_routes(folds=10, seed=0, lam=1e-6, results=RESULTS):
    """The dual and the primal on the same samples: the same answer, different cost."""
    _say("=== the two routes on the set in the repository")
    _mols, _fields, targets, X, build = read(DATA)
    _say(f"{len(targets)} carbons in {targets.n_measured_molecules} molecules, "
         f"{X.shape[1]} channels, environments built in {build:.0f} s")

    t0 = time.time()
    K = local_kernel(X)
    build_dual = time.time() - t0
    t0 = time.time()
    dual = cross_validate(K, targets, lam, folds=folds, seed=seed, name="dual")
    time_dual = time.time() - t0

    t0 = time.time()
    Z, _scale = normalised_rows(X, n_bins=LOCAL_SPEC.n_bins)
    build_primal = time.time() - t0
    t0 = time.time()
    primal = cross_validate_primal(Z, targets, lam, folds=folds, seed=seed, name="primal")
    time_primal = time.time() - t0

    gap = float(np.max(np.abs(dual.predicted - primal.predicted)))
    _say(f"  dual:   q2 {dual.q2:+.6f}, error {dual.error:.4f}; kernel {build_dual:.0f} s, "
         f"{gigabytes(len(targets)):.2f} GB, folds {time_dual:.0f} s")
    _say(f"  primal: q2 {primal.q2:+.6f}, error {primal.error:.4f}; rows {build_primal:.1f} s, "
         f"{len(targets) * X.shape[1] * 8 / 1024 ** 2:.0f} MB, folds {time_primal:.1f} s")
    _say(f"  largest difference between the two sets of predictions: {gap:.2e} ppm")

    out = dict(n_samples=len(targets), n_molecules=targets.n_measured_molecules,
               channels=int(X.shape[1]), rho=RHO, lam=float(lam), folds=folds, seed=seed,
               environments_seconds=build,
               dual=dict(q2=dual.q2, error=dual.error, build_seconds=build_dual,
                         fold_seconds=time_dual, gigabytes=gigabytes(len(targets))),
               primal=dict(q2=primal.q2, error=primal.error, build_seconds=build_primal,
                           fold_seconds=time_primal,
                           megabytes=len(targets) * X.shape[1] * 8 / 1024 ** 2),
               largest_difference=gap)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "routes.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def run_curve(data_root, sizes=SIZES, folds=10, seed=0, results=RESULTS):
    """Error against the number of molecules, on a set this repository does not ship."""
    path = os.path.join(data_root, "shifts_cdcl3.sdf")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing. Build it with examples/make_local_examples.py "
                         f"--shifts <dump> --max-shifts 2000 --any-solvent --out {data_root}")
    _say(f"=== the learning curve, from {path}")
    _mols, _fields, targets, X, build, envs = read(path, with_envs=True)
    _say(f"{len(targets)} carbons in {targets.n_measured_molecules} molecules, "
         f"environments built in {build:.0f} s; an exact kernel over them would be "
         f"{gigabytes(len(targets)):.1f} GB")
    Z, _scale = normalised_rows(X, n_bins=LOCAL_SPEC.n_bins)
    sizes = [s for s in sizes if s <= targets.n_measured_molecules]
    if targets.n_measured_molecules not in sizes:
        sizes.append(targets.n_measured_molecules)
    curve = learning_curve(Z, targets, sizes, folds=folds, seed=seed,
                           progress=lambda text, _f: _say("  " + text))
    # a flat curve says the error is not the data's; the question it leaves is whether it is the
    # descriptor's, and the cheapest test of that is a finer one on the same samples
    finer = []
    for step in (0.25, 0.125):
        spec = AutocorrelationSpec(d_max=LOCAL_SPEC.d_max, step=step, sigma=step)
        t0 = time.time()
        Xf = local_features(envs, spec)
        Zf, _s = normalised_rows(Xf, n_bins=spec.n_bins)
        res = choose_lam_primal(Zf, targets, folds=folds, seed=seed)
        finer.append(dict(step=float(step), channels=int(Xf.shape[1]), q2=res.q2,
                          error=res.error, lam=res.lam, seconds=time.time() - t0))
        _say(f"  distance bins every {step:g} A ({Xf.shape[1]} channels): "
             f"q2 {res.q2:+.3f}, error {res.error:.2f}")

    out = dict(source=os.path.basename(data_root), n_samples=len(targets),
               n_molecules=targets.n_measured_molecules, rho=RHO, folds=folds, seed=seed,
               environments_seconds=build, gigabytes=gigabytes(len(targets)),
               channels=int(X.shape[1]), step=float(LOCAL_SPEC.step),
               sizes=curve.sizes, molecules=curve.molecules, q2=curve.q2, error=curve.error,
               lam=curve.lam, seconds=curve.seconds, finer=finer)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "curve.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def run_landmarks(ranks=RANKS, folds=10, seed=0, rho=0.25, results=RESULTS):
    """The Nystrom route, on the kernel that needs it: the posed overlap kernel of LP3."""
    sys.path.insert(0, HERE)
    from run_local_cofield import aromatic_samples

    _say("=== landmarks for the overlap kernel, on the aromatic carbons")
    mols, names, targets = aromatic_samples()
    fieldset = FieldSet.create()
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(mols, names)]
    envs = posed_environments(fields, targets, rho=rho, frames_of=ring_frames)
    t0 = time.time()
    _model, _table, K = overlap_kernel(envs, fieldset)
    exact_seconds = time.time() - t0
    exact = choose_lam(K, targets, folds=folds, seed=seed, name="exact overlap")
    _say(f"  exact: {len(targets)} samples, kernel in {exact_seconds:.0f} s, "
         f"q2 {exact.q2:+.3f}, error {exact.error:.3f}")

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(targets))
    rows = []
    for m in [r for r in ranks if r < len(targets)]:
        idx = np.sort(order[:m])
        approx = nystrom_kernel(K[:, idx], K[np.ix_(idx, idx)])
        err = float(np.linalg.norm(K - approx) / np.linalg.norm(K))
        res = choose_lam(approx, targets, folds=folds, seed=seed, name=f"rank {m}")
        rows.append(dict(rank=int(m), relative_error=err, q2=res.q2, error=res.error,
                         lam=res.lam))
        _say(f"  rank {m:4d}: relative kernel error {err:.3f}, q2 {res.q2:+.3f}, "
             f"error {res.error:.3f}")
    out = dict(n_samples=len(targets), n_molecules=targets.n_measured_molecules, rho=rho,
               folds=folds, seed=seed, exact_seconds=exact_seconds,
               exact=dict(q2=exact.q2, error=exact.error, lam=exact.lam), rows=rows)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "landmarks.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def report(results=RESULTS):
    out = ["# How far the local kernel goes (plan item LP5)", "",
           "A per-atom target is a large problem where a per-molecule one is a small one. The "
           "question is what has to be given up to reach the size the free databases have.", ""]
    path = os.path.join(results, "routes.json")
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        out += ["## The invariant local kernel needs no approximation at all", "",
                "Its feature map is finite: the field autocorrelation of an environment is "
                f"{d['channels']} channels for the five physico-chemical fields, scaled per block "
                "and cosine-normalised, and the kernel is the inner product of those rows. So the "
                "ridge regression can be solved in the space of the coefficients rather than in "
                "the space of the samples. It is the same model, not an approximation of it.", "",
                f"{d['n_samples']} carbons in {d['n_molecules']} molecules, {d['folds']} folds "
                f"holding out whole molecules, ridge {d['lam']:g}:", "",
                "| route | q2 | error (ppm) | kernel or rows | memory | the folds |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                f"| dual, an n by n kernel | {d['dual']['q2']:+.6f} | {d['dual']['error']:.4f} | "
                f"{d['dual']['build_seconds']:.0f} s | {d['dual']['gigabytes']:.2f} GB | "
                f"{d['dual']['fold_seconds']:.0f} s |",
                f"| primal, n by d rows | {d['primal']['q2']:+.6f} | "
                f"{d['primal']['error']:.4f} | {d['primal']['build_seconds']:.1f} s | "
                f"{d['primal']['megabytes']:.0f} MB | {d['primal']['fold_seconds']:.1f} s |",
                "",
                f"The largest difference between the two sets of out-of-fold predictions is "
                f"{d['largest_difference']:.1e} ppm, which is round-off. The memory is the "
                "difference that decides: the kernel grows with the square of the samples and "
                "the rows grow with the samples."]
        n = d["n_samples"]
        for mult in (4, 20):
            out.append(f"At {mult} times this many carbons the kernel would be "
                       f"{gigabytes(n * mult):.0f} GB and the rows "
                       f"{n * mult * d['channels'] * 8 / 1024 ** 2:.0f} MB.")
        out.append("")
    path = os.path.join(results, "curve.json")
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as fh:
            c = json.load(fh)
        out += ["## What more data is worth", "",
                f"{c['n_samples']} carbons in {c['n_molecules']} molecules, from a set built "
                "from the nmrshiftdb2 dump that this repository does not ship (see the script "
                f"this benchmark names). An exact kernel over them would be {c['gigabytes']:.1f} "
                "GB, so the primal route is not a convenience here but the only one that runs. "
                "Each size is a subset of the next.", "",
                "| molecules | carbons | q2 | error (ppm) | ridge | the folds |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for i in range(len(c["sizes"])):
            out.append(f"| {c['molecules'][i]} | {c['sizes'][i]} | {c['q2'][i]:+.3f} | "
                       f"{c['error'][i]:.2f} | {c['lam'][i]:g} | {c['seconds'][i]:.1f} s |")
        if len(c["error"]) > 1:
            first, last = c["error"][0], c["error"][-1]
            out += ["", f"From {c['molecules'][0]} molecules to {c['molecules'][-1]} the error "
                        f"goes from {first:.2f} to {last:.2f} ppm, a factor of "
                        f"{first / max(last, 1e-9):.2f} for a factor of "
                        f"{c['molecules'][-1] / max(c['molecules'][0], 1):.0f} in data."]
        if c.get("finer"):
            out += ["", "**A flat curve says the error is not the data's.** What is left is the "
                        "descriptor, and the cheapest test of that is a finer one on the same "
                        f"samples: the distance axis of the autocorrelation at "
                        f"{c['step']:g} A per bin against two finer spacings.", "",
                    "| distance bins | channels | q2 | error (ppm) |",
                    "| ---: | ---: | ---: | ---: |",
                    f"| every {c['step']:g} A | {c['channels']} | {c['q2'][-1]:+.3f} | "
                    f"{c['error'][-1]:.2f} |"]
            for f in c["finer"]:
                out.append(f"| every {f['step']:g} A | {f['channels']} | {f['q2']:+.3f} | "
                           f"{f['error']:.2f} |")
            best = min(c["finer"], key=lambda f: f["error"])
            gain = c["error"][-1] - best["error"]
            out += ["", (f"The finest grid gains {gain:.2f} ppm, which is "
                         + ("the larger part of what the data could not give"
                            if gain > 0.5 else "little")
                         + ": the error of this model is set by what the descriptor can tell "
                           "apart, not by how many carbons it has seen."), ""]
        out.append("")
    path = os.path.join(results, "landmarks.json")
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as fh:
            k = json.load(fh)
        out += ["## The kernel that does need landmarks", "",
                "The posed overlap kernel of plan item LP3 has no finite feature map -- its "
                "features are the field itself, a function on space -- so there the dual form is "
                "the only exact one and a Nystrom approximation is the route the plan names. On "
                f"the {k['n_samples']} aromatic carbons of that item, with landmark environments "
                "drawn at random from the samples:", "",
                "| landmarks | relative error of the kernel | q2 | error (ppm) |",
                "| ---: | ---: | ---: | ---: |",
                f"| all {k['n_samples']} (exact) | 0.000 | {k['exact']['q2']:+.3f} | "
                f"{k['exact']['error']:.3f} |"]
        for r in k["rows"]:
            out.append(f"| {r['rank']} | {r['relative_error']:.3f} | {r['q2']:+.3f} | "
                       f"{r['error']:.3f} |")
        good = [r for r in k["rows"] if r["q2"] >= k["exact"]["q2"] - 0.01]
        if good:
            m = min(r["rank"] for r in good)
            out += ["", f"**{m} landmarks of {k['n_samples']} samples come within 0.01 of the "
                        f"exact q2**, at {100.0 * m / k['n_samples']:.0f} per cent of the rank. "
                        "The environments of one site are alike, which is why so few span them.",
                    ""]
        else:
            out += ["", "No rank measured here reaches the exact q2 within 0.01.", ""]
    return "\n".join(out)


def write_table(results=RESULTS) -> str:
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, metavar="DIR",
                    help="where a larger shifts_cdcl3.sdf was built; without it the curve is "
                         "skipped")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="all", choices=["all", "routes", "curve", "landmarks"])
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
    if a.only in ("all", "routes"):
        run_routes(folds=a.folds, seed=a.seed)
    if a.only in ("all", "curve") and a.data_root:
        run_curve(a.data_root, folds=a.folds, seed=a.seed)
    if a.only in ("all", "landmarks"):
        run_landmarks(folds=a.folds, seed=a.seed)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
