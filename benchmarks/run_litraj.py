#!/usr/bin/env python
"""
The hop models of plan item BT7 on the published nebBVSE122k benchmark: the external check of BT7.

``benchmarks/prepare_litraj.py`` draws a sample of the nebBVSE122k set of Dembitskiy et al.,
npj Computational Materials 11, 131 (2025), which holds 122,421 bond-valence barriers of single
Li hops in 11,290 structures of the Materials Project.  This script fits on that sample the same
models ``benchmarks/run_hops.py`` fits on the stored public set of item BT7, with the same
protocol: 10 folds that hold out whole structures, the same ridge grid, the environment within
6 A of the hop midpoint, and a paired bootstrap that resamples structures, because the hops of
one cell are not independent of each other.

What the comparison can and cannot say:

* **The protocol matches.**  The published split of the collection holds out whole structures as
  well -- no structure of its test part appears in its training part -- so both sides measure
  prediction for a cell that was never seen, and the numbers are comparable in kind.
* **The label range differs, and q2 follows it.**  The barriers of the collection spread 2.4
  times as widely as those of the public set of item BT7 (standard deviation 1.97 eV against
  0.81 eV), because it covers every chemistry that holds lithium rather than ternary lithium
  oxides alone.  The same absolute error therefore gives a much higher q2 on it, so the table
  reports the error in eV beside q2 and prints the standard deviation of the sample.
* **The training set differs in size.**  The published models are fitted on 96,849 hops of 8,945
  structures; a kernel model holds one number per pair of samples, so this one is fitted on a
  sample about forty times smaller.  The learning curve at the end of the table is what says how
  much of the difference that explains.

Usage
-----
    python benchmarks/run_litraj.py                  # the run that writes the stored result
    python benchmarks/run_litraj.py --report         # rebuild the table from the stored result
    python benchmarks/run_litraj.py --max-structures 40    # a quick look
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar.hops import (Hop, HopModel, HopSet, bottleneck,  # noqa: E402
                          cross_validate_hop_model, framed_environments, hop_count_features,
                          hop_types)
from cmfqsar.localenv import (LAMS, choose_lam, count_kernel, local_features,  # noqa: E402
                              local_kernel, paired_interval)
from cmfqsar.sites import site_kernel  # noqa: E402

from run_hops import _entry, fieldset_for, site_pair_features  # noqa: E402

RESULTS = os.path.join(HERE, "results", "litraj")
PREPARED = os.path.join(RESULTS, "prepared.json")
#: the envelope width the public set of item BT7 does best at; a list measures a width series
RHOS = (0.25,)
R_MAX = 6.0
FOLDS = 10
BOOTSTRAP = 1000
#: the cutoff of the site profile the site-resolved reference is taken at, in angstrom; item BT7
#: compares at the radius the hop environments themselves use
SITE_R = 6.0
#: how many structures the learning curve is drawn at
CURVE = (25, 50, 100, 200)
#: Table 2 of the paper: what its models reach on the test part of the collection, as
#: (model, mean absolute error in eV, root mean squared error in eV, R2)
PUBLISHED = (("M3GNet (graph network, dummy atom at the hop midpoint)", 0.21, 0.44, 0.95),
             ("XGBRF (gradient-boosted forest, Voronoi features of the hop)", 0.26, 0.49, 0.94),
             ("RF (random forest, the same features)", 0.26, 0.50, 0.94),
             ("Nequip (graph network)", 0.25, 0.52, 0.93),
             ("KNN (nearest neighbours, the same features)", 0.27, 0.58, 0.92),
             ("Allegro (graph network)", 0.38, 0.65, 0.89))
#: hops and structures of the training part of the collection, and of the test part those
#: numbers are measured on
PUBLISHED_TRAIN = (96849, 8945)
PUBLISHED_TEST = (13167, 1172)


# ------------------------------------------------------------------- one run and what it wrote


def run_tag(a) -> str:
    """The name of one run of this script, used for every file that run writes.

    Every option that changes a stored number enters the tag -- which prepared sample is read,
    how much of it, the envelope widths, the radius of the environment, the cutoff of the
    site-resolved reference, the folds, the seed and the number of bootstrap draws -- and the
    options that only choose which parts of the run are done leave it alone.  Pointing
    ``--prepared`` at the stored sample gives the empty tag, which is why ``public.json`` has no
    suffix.
    """
    tag = ""
    if os.path.abspath(a.prepared) != os.path.abspath(PREPARED):
        stem = os.path.splitext(os.path.basename(a.prepared))[0]
        tag += "_" + stem
    if a.max_structures:
        tag += f"_n{a.max_structures}"
    if list(a.rho) != list(RHOS):
        tag += "_rho" + "-".join(f"{r:g}" for r in a.rho)
    if a.r_max != R_MAX:
        tag += f"_r{a.r_max:g}"
    if a.site_r != SITE_R:
        tag += f"_siter{a.site_r:g}"
    if a.folds != FOLDS:
        tag += f"_folds{a.folds}"
    if a.seed != 0:
        tag += f"_seed{a.seed}"
    if a.bootstrap != BOOTSTRAP:
        tag += f"_boot{a.bootstrap}"
    return tag


def run_provenance(results_dir: str = RESULTS, tag: str = "") -> Optional[dict]:
    """What the run with this tag was asked to do, or ``None`` if the run did not record it."""
    path = os.path.join(results_dir, f"run{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_provenance(args, tag: str, argv: Optional[List[str]] = None, **extra) -> str:
    """The command of this run and every option that changes a number, next to its results."""
    out = {"argv": list(argv) if argv is not None else sys.argv[1:], "tag": tag,
           "prepared": os.path.relpath(args.prepared, ROOT).replace(os.sep, "/"),
           "max_structures": args.max_structures, "rho": [float(r) for r in args.rho],
           "r_max": args.r_max, "site_r": args.site_r, "folds": args.folds,
           "seed": args.seed, "bootstrap": args.bootstrap,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.update(extra)
    path = os.path.join(RESULTS, f"run{tag}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    return path


# ----------------------------------------------------------------------------- the data


def load_prepared(path: str = PREPARED, max_structures: int = 0) -> tuple:
    """The prepared sample as a HopSet, with its oxidation states as a per-atom array.

    Returns the set and the document it was read from, whose header carries what the whole
    collection looks like.
    """
    from ase import Atoms
    doc = json.load(open(path))
    rows = doc["structures"]
    if max_structures:
        rows = rows[:max_structures]
    structures, names, hops = [], [], []
    for k, row in enumerate(rows):
        a = Atoms(symbols=row["symbols"], positions=np.asarray(row["positions"], float),
                  cell=np.asarray(row["cell"], float), pbc=True)
        a.set_array("oxi_states", np.array([float(row["oxidation"][s]) for s in row["symbols"]]))
        structures.append(a)
        names.append(f"{row['id']} {row['formula']}")
        cell = np.asarray(row["cell"], float)
        pos = np.asarray(row["positions"], float)
        for h in row["hops"]:
            off = np.asarray(h["offset"], int)
            shift = off @ cell
            hop = Hop(k, int(h["a"]), int(h["b"]), tuple(int(x) for x in off),
                      start=pos[h["a"]].copy(), vector=pos[h["b"]] + shift - pos[h["a"]],
                      shift=shift, y=float(h["y"]))
            hop.name = f"{row['id']}: {h['edge_id']}"
            hops.append(hop)
    return HopSet(structures, names, hops, "Li"), doc


# ----------------------------------------------------------------------------- the run


def run(args, log) -> dict:
    t0 = time.time()
    hs, doc = load_prepared(args.prepared, args.max_structures)
    fs = fieldset_for(hs, "file")
    crystals = hs.crystals(fs)
    t = hs.targets()
    y = np.asarray(t.y, float)
    log(hs.describe())
    log(f"  the whole collection: {doc['n_index_hops']} hops of "
        f"{doc['n_index_structures']} structures and {doc['n_index_chemsys']} chemical systems, "
        f"barriers mean {doc['index_y_mean']:.3f}, standard deviation {doc['index_y_sd']:.3f} eV")
    log(f"  this sample: {len(t)} hops in {t.n_measured_molecules} structures, barriers mean "
        f"{y.mean():.3f}, standard deviation {y.std(ddof=1):.3f} eV; "
        f"fields: {', '.join(fs.names)}")
    out = {"n_structures": len(hs.structures), "n_samples": len(t), "fields": fs.names,
           "r_max": args.r_max, "folds": args.folds, "site_r": args.site_r,
           "y_mean": float(y.mean()), "y_sd": float(y.std(ddof=1)),
           "lengths": [float(np.min([hs.hops[r].length for r in t.atom])),
                       float(np.max([hs.hops[r].length for r in t.atom]))],
           "collection": {k: doc[k] for k in ("source", "dataset", "reference", "licence",
                                              "n_index_hops", "n_index_structures",
                                              "n_index_chemsys", "index_y_mean", "index_y_sd",
                                              "drawn", "max_per_structure", "dropped_hops")},
           "models": [], "pairs": []}
    # the references: counts of atoms in shells about the midpoint, and the two numbers a chemist
    # starts from.  The bottleneck is read off the environments, which are needed anyway.
    envs0 = hs.environments(crystals, rows=t.atom, rho=args.rho[0], r_max=args.r_max)
    for r, e in zip(t.atom, envs0):
        h = hs.hops[int(r)]
        h.bottleneck, h.bottleneck_at, h.bottleneck_atoms = bottleneck(e, h.vector)
    types = hop_types(envs0, min_atoms=20)
    ref = choose_lam(count_kernel(hop_count_features(envs0, types)), t, folds=args.folds,
                     seed=args.seed, name="counts about the midpoint")
    log("  " + ref.summary())
    X = np.array([[hs.hops[r].length, hs.hops[r].bottleneck] for r in t.atom])
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    geo = choose_lam(X @ X.T, t, LAMS, folds=args.folds, seed=args.seed,
                     name="hop length and bottleneck")
    log("  " + geo.summary())
    sp = site_pair_features(hs, crystals, t.atom, args.site_r)
    sites = choose_lam(site_kernel(sp), t, folds=args.folds, seed=args.seed,
                       name=f"the two end sites (M5), cutoff {args.site_r:g} A")
    log("  " + sites.summary())
    out["models"] += [_entry(ref, extra={"key": "counts", "types": types}),
                      _entry(geo, extra={"key": "geometry"}),
                      _entry(sites, extra={"key": "sites", "site_r_max": float(args.site_r),
                                           "channels": int(sp.shape[1])})]
    results = {"counts": ref, "geometry": geo, "sites": sites}
    # the two hop kernels of item BT7
    best = None
    for rho in args.rho:
        envs = envs0 if rho == args.rho[0] else hs.environments(crystals, rows=t.atom, rho=rho,
                                                               r_max=args.r_max)
        inv = choose_lam(local_kernel(local_features(envs)), t, folds=args.folds, seed=args.seed,
                         name=f"invariant local, rho {rho:g}")
        log("  " + inv.summary())
        fr = framed_environments(hs, crystals, rows=t.atom, rho=rho, r_max=args.r_max)
        t1 = time.time()
        model = HopModel.build(fr, fs, n_jobs=args.jobs)
        cyl = cross_validate_hop_model(model, t, folds=args.folds, seed=args.seed,
                                       name=f"rotation-averaged overlap, rho {rho:g}")
        atoms = float(np.mean([e.n_atoms for e in envs]))
        log(f"  {cyl.summary()}  (kernel {time.time() - t1:.0f} s, "
            f"{atoms:.0f} atoms per environment)")
        out["models"] += [_entry(inv, rho, {"key": f"invariant {rho}", "atoms": atoms}),
                          _entry(cyl, rho, {"key": f"overlap {rho}", "atoms": atoms})]
        results[f"invariant {rho}"] = inv
        results[f"overlap {rho}"] = cyl
        for name, r in ((f"invariant {rho}", inv), (f"overlap {rho}", cyl)):
            for ref_name in ("counts", "geometry", "sites"):
                d, lo, hi = paired_interval(r, results[ref_name], t, runs=args.bootstrap,
                                            seed=args.seed)
                out["pairs"].append({"a": name, "b": ref_name, "diff": d, "low": lo, "high": hi})
        d, lo, hi = paired_interval(cyl, inv, t, runs=args.bootstrap, seed=args.seed)
        out["pairs"].append({"a": f"overlap {rho}", "b": f"invariant {rho}", "diff": d,
                             "low": lo, "high": hi})
        log(f"    overlap against invariant at the same width {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if best is None or cyl.q2 > best[0].q2:
            best = (cyl, rho, model)
    cyl, rho, model = best
    out["best_rho"] = rho
    # the learning curve of the best model, on the kernel already in hand: how much of the
    # distance to the published models is the size of the training set
    curve = []
    which = np.unique(t.molecule)
    rng = np.random.RandomState(args.seed)
    for n_struct in CURVE:
        if n_struct >= len(which):
            break
        q2s, errs = [], []
        for rep in range(3):
            drawn = set(rng.choice(which, n_struct, replace=False).tolist())
            rows = np.array([i for i in range(len(t)) if int(t.molecule[i]) in drawn])
            sub = t.subset(rows)
            r = choose_lam(model.K[np.ix_(rows, rows)], sub, LAMS,
                           folds=min(args.folds, n_struct), seed=args.seed + rep, name="subset")
            q2s.append(r.q2)
            errs.append(r.error)
        curve.append({"structures": n_struct, "q2": float(np.mean(q2s)),
                      "spread": float(np.std(q2s)), "rmse": float(np.mean(errs))})
        log(f"    learning curve: {n_struct} structures, q2 {np.mean(q2s):+.3f} "
            f"+- {np.std(q2s):.3f}, error {np.mean(errs):.3f} eV")
    curve.append({"structures": int(len(which)), "q2": cyl.q2, "spread": 0.0,
                  "rmse": cyl.error})
    out["learning_curve"] = curve
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------- the table


def write_table(results_dir: str = RESULTS, tag: str = "") -> str:
    """The table of one run, from the result files that carry its tag; returns the text.

    ``run<tag>.json`` holds the command that produced those files; the table prints it, so that a
    reader can repeat the run and so that a stored result with no command is visible as such.
    """
    pub = json.load(open(os.path.join(results_dir, f"public{tag}.json")))
    prov = run_provenance(results_dir, tag)
    col = pub["collection"]
    pairs = {(p["a"], p["b"]): p for p in pub["pairs"]}
    lines = []
    w = lines.append
    w("# The hop models on the published nebBVSE122k benchmark (the external check of item BT7)")
    w("")
    if prov:
        w("`" + " ".join(["python benchmarks/run_litraj.py"] + list(prov["argv"])) + "`, reading "
          f"`{prov['prepared']}` (written by `python benchmarks/prepare_litraj.py --fetch`), "
          f"seed {prov['seed']}, {prov['bootstrap']} bootstrap draws.")
    else:
        w("The command that produced these result files was not recorded.")
    w("")
    w(f"The collection is {col['dataset']} of {col['reference']}, {col['licence']} licence, "
      f"{col['source']}: {col['n_index_hops']} bond-valence barriers of single Li hops in "
      f"{col['n_index_structures']} structures of the Materials Project over "
      f"{col['n_index_chemsys']} chemical systems. Its label is the barrier of one hop on a rigid "
      f"framework, as in item BT7, but relaxed with the nudged elastic band method instead of "
      f"taken as the minimax path on a grid.")
    w("")
    w(f"A kernel model holds one number per pair of samples, so this run reads a sample: "
      f"{col['drawn']} structures drawn at random with at most {col['max_per_structure']} of "
      f"their hops each, which leaves {pub['n_samples']} hops in {pub['n_structures']} "
      f"structures. Hop lengths {pub['lengths'][0]:.2f} to {pub['lengths'][1]:.2f} A. "
      f"{pub['folds']} folds holding out whole structures; environments within "
      f"{pub['r_max']:g} A of the midpoint. Fields: {', '.join(pub['fields'])}.")
    w("")
    w(f"**The barriers of this sample spread more widely than those of the public set of item "
      f"BT7**: standard deviation {pub['y_sd']:.3f} eV against 0.807 eV, mean "
      f"{pub['y_mean']:.3f} eV, because the collection covers every chemistry that holds lithium "
      f"rather than ternary lithium oxides alone. q2 is measured against the spread of the "
      f"sample, so the same error in eV gives a higher q2 here: **the error in eV is the column "
      f"to compare with item BT7**. Against the published models of the next section both columns "
      f"hold, because this sample and the whole collection spread almost alike "
      f"({pub['y_sd']:.2f} eV against {col['index_y_sd']:.2f} eV).")
    w("")
    w("| descriptor | envelope width rho | q2 | error (eV) | mean absolute error (eV) | ridge "
      "| against the counts | against length and bottleneck | against the two end sites |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in pub["models"]:
        key = m.get("key", "")
        cells = []
        for ref in ("counts", "geometry", "sites"):
            p = pairs.get((key, ref))
            cells.append("--" if p is None else
                         f"{p['diff']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]")
        rho = f"{m['rho']:g}" if m.get("rho") is not None else "--"
        w(f"| {m['name']} | {rho} | {m['q2']:+.3f} | {m['rmse']:.3f} | {m['mae']:.3f} | "
          f"{m['lam']:g} | " + " | ".join(cells) + " |")
    w("")
    best = f"overlap {pub['best_rho']}"
    p = pairs.get((best, f"invariant {pub['best_rho']}"))
    if p:
        w(f"Overlap in the hop frame against the invariant kernel at the same width: "
          f"{p['diff']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}].")
        w("")
    curve = pub.get("learning_curve") or []
    if curve:
        w("Learning curve of the rotation-averaged overlap model, three random draws of the "
          "training structures at each size. Each draw has a label spread of its own, so the "
          "error in eV is not comparable from row to row and q2 is the column to read:")
        w("")
        w("| training structures | q2 | error (eV) |")
        w("| ---: | ---: | ---: |")
        for c in curve:
            w(f"| {c['structures']} | {c['q2']:+.3f} | {c['rmse']:.3f} |")
        w("")
    w("## What the published models reach on the whole collection")
    w("")
    w(f"Table 2 of the paper, on its test part ({PUBLISHED_TEST[0]} hops of "
      f"{PUBLISHED_TEST[1]} structures, none of them in the training part), fitted on "
      f"{PUBLISHED_TRAIN[0]} hops of {PUBLISHED_TRAIN[1]} structures:")
    w("")
    w("| model | mean absolute error (eV) | error (eV) | R2 |")
    w("| --- | ---: | ---: | ---: |")
    for name, mae, rmse_, r2 in PUBLISHED:
        w(f"| {name} | {mae:.2f} | {rmse_:.2f} | {r2:.2f} |")
    w("")
    w("Those models are fitted on about forty times as many hops as the sample above, which is "
      "the one difference the protocol does not remove, so the learning curve is what says how "
      "much of the distance the size of the training set explains.")
    w("")
    if len(curve) >= 2:
        a, b = curve[-2], curve[-1]
        steps = float(np.log2(PUBLISHED_TRAIN[1] / b["structures"]))
        per = (b["q2"] - a["q2"]) / float(np.log2(b["structures"] / a["structures"]))
        w(f"Between {a['structures']} and {b['structures']} training structures the overlap model "
          f"gains {b['q2'] - a['q2']:+.3f} in q2, which is {per:+.3f} for every doubling of the "
          f"training set. The training part of the collection holds {PUBLISHED_TRAIN[1]} "
          f"structures, {steps:.1f} doublings beyond this sample, so that rate carries the model "
          f"to about q2 {b['q2'] + per * steps:+.2f}, short of the {PUBLISHED[0][3]:+.2f} the "
          f"graph network reaches. A learning curve is not a law and may bend either way, but at "
          f"the rate measured here the size of the training set does not account for the "
          f"distance.")
        w("")
    text = "\n".join(lines)
    with open(os.path.join(results_dir, f"table{tag}.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(text)
    return text


# ----------------------------------------------------------------------------- the command


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prepared", default=PREPARED, help="the prepared sample to read")
    ap.add_argument("--max-structures", type=int, default=0,
                    help="read only the first N structures of the sample (a quick look)")
    ap.add_argument("--rho", nargs="*", type=float, default=list(RHOS),
                    help="envelope widths to measure (default "
                         + ", ".join(f"{r:g}" for r in RHOS) + ")")
    ap.add_argument("--r-max", type=float, default=R_MAX,
                    help=f"radius of the environment about the midpoint, A (default {R_MAX:g})")
    ap.add_argument("--site-r", type=float, default=SITE_R,
                    help=f"cutoff of the site profile of the site-resolved reference, A "
                         f"(default {SITE_R:g})")
    ap.add_argument("--folds", type=int, default=FOLDS,
                    help=f"folds holding out whole structures (default {FOLDS})")
    ap.add_argument("--seed", type=int, default=0, help="the fold split and the draws (default 0)")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP,
                    help=f"paired bootstrap draws (default {BOOTSTRAP})")
    ap.add_argument("--jobs", type=int, default=8, help="processes for the overlap kernel")
    ap.add_argument("--report", action="store_true",
                    help="rebuild the table from the stored result and stop")
    return ap


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    tag = run_tag(args)

    if args.report:
        print(write_table(RESULTS, tag))
        return

    if not os.path.exists(args.prepared):
        raise SystemExit(f"{args.prepared} is missing: run "
                         f"benchmarks/prepare_litraj.py --fetch first")
    log_path = os.path.join(RESULTS, f"log{tag}.txt")
    log_file = open(log_path, "w", encoding="utf-8", newline="\n")

    def log(text: str):
        print(text, flush=True)
        log_file.write(text + "\n")
        log_file.flush()

    t0 = time.time()
    out = run(args, log)
    with open(os.path.join(RESULTS, f"public{tag}.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    minutes = (time.time() - t0) / 60
    log(f"done in {minutes:.1f} minutes")
    log_file.close()
    write_provenance(args, tag, argv, minutes=round(minutes, 1))
    print(write_table(RESULTS, tag))


if __name__ == "__main__":
    main()
