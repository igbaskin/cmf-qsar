#!/usr/bin/env python
"""
A migration path as the sample: does the field around a Li hop predict its barrier? (BT7)

Two measurements, both with folds that hold out whole structures.

1. **The public set.**  Ternary compounds of lithium and oxygen (Li, O and one more element) of the Materials Project within 0.05 eV/atom of
   the convex hull, every symmetry-distinct Li hop between 1.5 and 4.5 A, labelled with the
   bond-valence site-energy barrier of the softBV model (``benchmarks/prepare_hops.py`` writes
   ``results/hops/prepared_mp.json``).  The bond-valence barrier is a known function of the
   framework around the path, so this asks whether the kernel can learn such a function across
   structure types, and which descriptor it needs: the count of atoms in shells about the
   midpoint (the reference of rule 1), two numbers a chemist would start from (the hop length
   and the free space at the bottleneck), the two end sites of the hop described as plan item M5
   describes a single site of a cell (the reference the gate of BT7 asks for), the
   rotation-invariant local kernel of plan item LP2, or the overlap kernel in the frame of the
   hop averaged over the rotation about it, which is the one that has a co-field.  The co-field
   is read along the hop and away from its axis, and the out-of-fold barriers are run through the
   percolation step to see whether the predicted network gives the percolation thresholds the
   labelled one does.

2. **The three hosts of the Li interface study** (``examples/hops``): Li3PO4, Li3BP2O8 and
   LiBO2 as DFT-relaxed supercells with Bader charges.  The model trained on the public set is
   applied to their distinct hops, and the two hops with a DFT elastic-band barrier are listed
   next to the bond-valence and the predicted value.  Three structures are too few for a q2 of
   their own; what is measured is the transfer.

3. **The disordered conductors** (``examples/hops/disordered``, plan item M2): thirteen
   structures of the Crystallography Open Database from the families a solid-electrolyte study
   is about -- garnets, perovskites, LGPS, NASICON, LISICON, thio-phosphates, lithium nitride --
   ten of them with partly occupied or shared sites, read with their occupancies.  Their hops
   are labelled with the bond-valence barrier of the ordered framework, the public model is
   applied to them as it stands, a model is cross-validated within the set with folds that hold
   out whole structures, and the percolation thresholds of the labelled and the predicted
   networks are listed per structure.

Usage
-----
    python benchmarks/run_hops.py                     # all three parts
    python benchmarks/run_hops.py --fine-profile      # the co-field on a fine grid, for the manual
    python benchmarks/run_hops.py --max-structures 60 # a quick look, under its own file names
    python benchmarks/run_hops.py --interfaces        # the model read from outside (item MS7)
    python benchmarks/run_hops.py --report            # the table from the stored results

4. **The model read from outside** (``--interfaces``, plan item MS7).  The public model is
   rebuilt at the stored width and read twice: by the route this script uses, and through
   ``cmfqsar.interfaces.HopBarriers``, which is handed a structure and builds the environments
   itself.  What is compared is the barriers of the hosts' paths, the two barriers the stored
   table lists against DFT, and the percolation thresholds a conductivity estimate starts from.
   It writes ``interfaces<tag>.json``.

Every option that changes a number enters the tag that names the files a run writes, so that two
runs whose numbers differ never write the same file: ``public<tag>.json``, ``hosts<tag>.json``,
``disordered<tag>.json``, ``log<tag>.txt``, ``table<tag>.md`` and ``run<tag>.json``, which holds
the command that produced them.  ``--fine-profile`` writes ``profile_fine<tag>.json`` and
``run_fine_profile<tag>.json``.  The stored measurement has the empty tag.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar.dataio import relative_path  # noqa: E402
from cmfqsar.hops import (Hop, HopModel, HopSet, cross_validate_hop_model,  # noqa: E402
                          framed_environments, hop_count_features, hop_types, read_hop_labels,
                          saddle_from_barrier, spread_labels, structure_percolation)
from cmfqsar.invariant import field_pairs  # noqa: E402
from cmfqsar.localenv import (LAMS, LocalResult, choose_lam, count_kernel,  # noqa: E402
                              cross_validate_lams, local_features, local_kernel,
                              paired_interval)
from cmfqsar.model import r2_score, rmse  # noqa: E402
from cmfqsar.periodic import PeriodicSpec, elements_in, inorganic_fieldset  # noqa: E402
from cmfqsar.sites import site_autocorrelation, site_kernel, site_width  # noqa: E402

RESULTS = os.path.join(HERE, "results", "hops")
PREPARED = os.path.join(RESULTS, "prepared_mp.json")
EXAMPLES = os.path.join(ROOT, "examples", "hops")
HOSTS = ("phosphate", "borophosphate", "borate")
DISORDERED = os.path.join(EXAMPLES, "disordered")
#: the COD entries of the disordered set, with a short name and what is disordered in each
COD = (("1545083", "LLZO:Ta cubic (Li6.5La3Zr1.5Ta0.5O12)", "Li sites 0.38/0.62, Zr/Ta shared"),
       ("1552152", "LLZO:Ta cubic (Li6.2La3Zr1.2Ta0.8O12)", "Li sites 0.28/0.96, Zr/Ta shared"),
       ("1545085", "LLZO:Ta tetragonal", "ordered"),
       ("1525568", "LLTO rhombohedral (La0.5Li0.5TiO3)", "Li sites 0.16"),
       ("1000437", "LLTO tetragonal (La0.56Li0.33TiO3)", "La/Li shared sites"),
       ("7215190", "LGPS (Li10GeP2S12)", "Li sites 0.47 to 0.89, Ge/P shared"),
       ("7222155", "LiTi2(PO4)3 NASICON", "ordered"),
       ("7222156", "Li3.18Ti2(PO4)3 NASICON", "Li sites 0.34/0.72"),
       ("4002499", "Li2.92N", "Li site 0.96"),
       ("1524557", "LISICON (Li13.7Zn1.2Ge4O16)", "Li sites 0.16 to 0.98, Li/Zn shared"),
       ("9011044", "Li3PO4 (Pmnb)", "ordered"),
       ("1570310", "beta-Li3PS4", "Li sites 0.3/0.7"),
       ("1529523", "Li3.16Zn0.24GeO4", "Li sites 0.06 to 0.92, Li/Zn shared"))
RHOS = (0.05, 0.1, 0.25)
R_MAX = 6.0
#: distinct hops kept per structure in the sample; the rest stay in the network
MAX_PER_STRUCTURE = 6
FOLDS = 10
BOOTSTRAP = 1000
PROFILE_Z = (-2.0, -1.0, 0.0, 1.0, 2.0)
PROFILE_R = (0.0, 1.0, 2.0, 3.0, 4.0)
PHYSCHEM = ("electrostatic", "steric", "prop:electronegativity")
#: cutoffs of the site profile, in angstrom, at which the site-resolved reference of a hop is
#: taken (plan item M5 measured its site model at these three).  The table compares the hop
#: kernels against the one at ``R_MAX``, the radius their own environments use.
SITE_R = (4.0, 6.0, 8.0)
#: the step and the width of the Gaussian of that profile, as item M5 set them
SITE_STEP = 0.5
SITE_SIGMA = 0.5


# ------------------------------------------------------------------- one run and what it wrote


def run_tag(a) -> str:
    """The name of one run of this script, used for every file that run writes.

    Two rules. Two runs whose numbers differ must never write the same file, so every option that
    changes a number enters the tag: which prepared set is read, how much of it, the envelope
    widths, the radius of the environment, the cutoffs of the site-resolved reference, the folds,
    the seed and the number of bootstrap draws.
    And two ways of naming one input must give one tag, so ``--prepared`` pointing at the stored
    set gives the empty tag; that is why ``public.json`` has no suffix. The options that only
    choose which parts of the run are done (``--no-hosts``, ``--reuse-public``, ``--fine-profile``)
    and the number of processes (``--jobs``) leave the tag alone, because they change no stored
    number; ``--fine-profile`` writes files of its own name instead.
    """
    tag = ""
    if os.path.abspath(a.prepared) != os.path.abspath(PREPARED):
        stem = os.path.splitext(os.path.basename(a.prepared))[0]
        tag += "_" + (stem[len("prepared_"):] if stem.startswith("prepared_") else stem)
    if a.max_structures:
        tag += f"_n{a.max_structures}"
    if a.max_per_structure != MAX_PER_STRUCTURE:
        tag += f"_per{a.max_per_structure}"
    if list(a.rho) != list(RHOS):
        tag += "_rho" + "-".join(f"{r:g}" for r in a.rho)
    if a.r_max != R_MAX:
        tag += f"_r{a.r_max:g}"
    if a.folds != FOLDS:
        tag += f"_folds{a.folds}"
    if a.seed != 0:
        tag += f"_seed{a.seed}"
    if a.bootstrap != BOOTSTRAP:
        tag += f"_boot{a.bootstrap}"
    if list(a.site_r) != list(SITE_R):
        tag += "_siter" + "-".join(f"{r:g}" for r in a.site_r)
    return tag


def run_provenance(results_dir: str = RESULTS, tag: str = "", name: str = "run") -> Optional[dict]:
    """What the run with this tag was asked to do, or ``None`` if the run did not record it."""
    path = os.path.join(results_dir, f"{name}{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_provenance(args, tag: str, argv: Optional[List[str]] = None, name: str = "run",
                     **extra) -> str:
    """The command of this run and every option that changes a number, next to its results."""
    out = {"argv": list(argv) if argv is not None else sys.argv[1:], "tag": tag,
           "prepared": relative_path(args.prepared, ROOT).replace(os.sep, "/"),
           "max_structures": args.max_structures, "max_per_structure": args.max_per_structure,
           "rho": [float(r) for r in args.rho], "r_max": args.r_max, "folds": args.folds,
           "site_r": [float(r) for r in args.site_r],
           "seed": args.seed, "bootstrap": args.bootstrap,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.update(extra)
    path = os.path.join(RESULTS, f"{name}{tag}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    return path


# ----------------------------------------------------------------------------- the data


def load_prepared(path: str = PREPARED, max_structures: int = 0, max_per_structure: int = 0,
                  seed: int = 0) -> HopSet:
    """The prepared public set as a HopSet, with oxidation states as a per-atom array."""
    from ase import Atoms
    d = json.load(open(path))
    rows = d["structures"]
    if max_structures:
        rows = rows[:max_structures]
    rng = np.random.RandomState(seed)
    structures, names, hops = [], [], []
    for k, row in enumerate(rows):
        a = Atoms(symbols=row["symbols"], positions=np.asarray(row["positions"], float),
                  cell=np.asarray(row["cell"], float), pbc=True)
        a.set_array("oxi_states", np.array([float(row["oxidation"][s]) for s in row["symbols"]]))
        a.info["hull"] = row["hull"]
        a.info["percolation"] = row.get("percolation", {})
        structures.append(a)
        names.append(f"{row['id']} {row['formula']}")
        base = len(hops)
        cell = np.asarray(row["cell"], float)
        pos = np.asarray(row["positions"], float)
        reps = [i for i, h in enumerate(row["hops"]) if h["representative"] == i]
        if max_per_structure and len(reps) > max_per_structure:
            keep = set(rng.choice(reps, max_per_structure, replace=False).tolist())
        else:
            keep = set(reps)
        for i, h in enumerate(row["hops"]):
            off = np.asarray(h["offset"], int)
            shift = off @ cell
            vec = pos[h["b"]] + shift - pos[h["a"]]
            hop = Hop(k, int(h["a"]), int(h["b"]), tuple(int(x) for x in off),
                      start=pos[h["a"]].copy(), vector=vec, shift=shift,
                      y=float(h["y"]) if h["y"] is not None else float("nan"),
                      y_reverse=float(h["y_reverse"]) if h["y_reverse"] is not None else float("nan"),
                      site_a=float(h["site_a"]) if h["site_a"] is not None else float("nan"),
                      site_b=float(h["site_b"]) if h["site_b"] is not None else float("nan"),
                      saddle=float(h["saddle"]) if h["saddle"] is not None else float("nan"),
                      bottleneck=float(h["bottleneck"]), bottleneck_at=0.5,
                      bottleneck_atoms=list(h["bottleneck_atoms"]),
                      representative=base + int(h["representative"]),
                      multiplicity=int(h["multiplicity"]))
            hop.name = hop.label(names)
            if h["representative"] == i and i not in keep:
                hop.y = float("nan")                      # left out of the sample, kept in the net
            hops.append(hop)
    return HopSet(structures, names, hops, "Li")


def load_hosts() -> HopSet:
    """The three hosts of the Li interface study, with their bond-valence labels."""
    from cmfqsar.periodic import read_structures
    structures = [read_structures(os.path.join(EXAMPLES, f"{h}.extxyz"))[0] for h in HOSTS]
    hs = HopSet.build(structures, list(HOSTS), "Li")
    labels = os.path.join(EXAMPLES, "hops_bvse.csv")
    if os.path.exists(labels):
        read_hop_labels(labels, hs)
        spread_labels(hs)
    return hs


def fieldset_for(hopset: HopSet, charge_model: str):
    elements = elements_in(hopset.structures, min_structures=max(1, len(hopset.structures) // 20))
    return inorganic_fieldset(elements, physchem=PHYSCHEM, charge_model=charge_model,
                              other=len(elements) < len(elements_in(hopset.structures, 1)))


# ----------------------------------------------------------------------------- the models


def _entry(r: LocalResult, rho=None, extra=None) -> dict:
    out = {"name": r.name, "q2": r.q2, "rmse": r.error, "mae": r.mae, "lam": r.lam,
           "n": r.n_samples, "structures": r.n_molecules, "folds": r.folds}
    if rho is not None:
        out["rho"] = rho
    if extra:
        out.update(extra)
    return out


def geometry_baseline(hopset: HopSet, targets, folds: int, seed: int) -> LocalResult:
    """Ridge on the hop length and the bottleneck distance: two numbers a chemist starts from."""
    X = np.array([[hopset.hops[r].length, hopset.hops[r].bottleneck] for r in targets.atom])
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    return choose_lam(X @ X.T, targets, LAMS, folds=folds, seed=seed,
                      name="hop length and bottleneck")


def site_pair_features(hs: HopSet, crystals, rows, r_max: float) -> np.ndarray:
    """The site-resolved reference descriptor of a hop: the profiles of its two end sites.

    Plan item M5 describes one **site** of a periodic cell by the row of the cell's periodic
    autocorrelation that belongs to that site (:func:`cmfqsar.sites.site_autocorrelation`): the
    typed radial profile of the cell as seen from the site, built from distances alone and
    needing no alignment.  A hop has two end sites, so the obvious site-resolved descriptor of a
    hop is the two rows of its ends put together.  They are combined as their sum and the
    absolute value of their difference, which does not depend on which end is called first: the
    sum is what the two ends share, the absolute difference is how far apart they are, and
    neither changes when the hop is reversed.  That matters because the barrier of a hop and of
    its reverse are different numbers measured on the same pair of sites.

    What the descriptor cannot see is the hop itself.  It carries no direction, no length and no
    bottleneck, so two hops that join the same pair of sites through different images of the cell
    get the same row.  That is what a site-resolved model of a path is, and it is what the gate
    of plan item BT7 asks to be compared against.
    """
    spec = PeriodicSpec(r_max=float(r_max), step=SITE_STEP, sigma=SITE_SIGMA)
    n_fields = crystals[0].weights.shape[1]
    pairs = field_pairs(n_fields)
    width = site_width(spec, n_fields)
    rows = [int(i) for i in rows]
    out = np.zeros((len(rows), 2 * width))
    where = {}
    for n, i in enumerate(rows):
        where.setdefault(hs.hops[i].structure, []).append(n)
    for k in sorted(where):
        profiles = site_autocorrelation(crystals[k], spec, pairs)   # one split per cell
        for n in where[k]:
            h = hs.hops[rows[n]]
            a, b = profiles[h.a], profiles[h.b]
            out[n, :width] = a + b
            out[n, width:] = np.abs(a - b)
    return out


def site_pair_ceiling(hs: HopSet, targets) -> dict:
    """The highest q2 a descriptor of the two end sites alone can reach on this sample.

    Such a descriptor gives one row to every hop that joins the same unordered pair of sites of
    one cell, so a hop and its reverse, and two hops that run through different images of the
    cell, cannot be told apart.  The best any model on it can do for a group of hops with one
    row is to predict their mean barrier, and ``1 - SS(within group) / SS(total)`` is that limit.
    It says whether a site-resolved model that does badly does so because its description cannot
    see the hop at all, or for some other reason.
    """
    y = np.asarray(targets.y, float)
    groups: Dict[tuple, List[int]] = {}
    for n, i in enumerate(targets.atom):
        h = hs.hops[int(i)]
        groups.setdefault((h.structure, min(h.a, h.b), max(h.a, h.b)), []).append(n)
    within = sum(float(np.sum((y[v] - y[v].mean()) ** 2)) for v in groups.values())
    total = float(np.sum((y - y.mean()) ** 2))
    return {"pairs": len(groups),
            "shared": sum(len(v) for v in groups.values() if len(v) > 1),
            "largest": max(len(v) for v in groups.values()),
            "q2": 1.0 - within / total if total else float("nan"),
            "rmse": float(np.sqrt(within / max(len(y), 1)))}


def run_public(args, log) -> dict:
    t0 = time.time()
    hs = load_prepared(args.prepared, args.max_structures, args.max_per_structure, args.seed)
    fs = fieldset_for(hs, "file")
    crystals = hs.crystals(fs)
    reps = [i for i, h in enumerate(hs.hops) if h.representative == i and np.isfinite(h.y)]
    t = hs.targets(reps)
    log(hs.describe())
    log(f"  sample: {len(t)} distinct hops with a barrier in {t.n_measured_molecules} structures; "
        f"fields: {', '.join(fs.names)}")
    log("  " + t.describe().replace("atoms", "hops").replace("molecules", "structures"))
    out = {"n_structures": len(hs.structures), "n_hops": len(hs.hops), "n_samples": len(t),
           "fields": fs.names, "r_max": args.r_max, "folds": args.folds, "models": [],
           "pairs": [], "hull": 0.05}
    # the references
    envs0 = hs.environments(crystals, rows=t.atom, rho=RHOS[1], r_max=args.r_max)
    types = hop_types(envs0, min_atoms=20)
    ref = choose_lam(count_kernel(hop_count_features(envs0, types)), t, folds=args.folds,
                     seed=args.seed, name="counts about the midpoint")
    log("  " + ref.summary())
    geo = geometry_baseline(hs, t, args.folds, args.seed)
    log("  " + geo.summary())
    out["models"] += [_entry(ref, extra={"key": "counts", "types": types}),
                      _entry(geo, extra={"key": "geometry"})]
    results = {"counts": ref, "geometry": geo}
    # the site-resolved reference the gate of BT7 asks for: the two end sites of the hop,
    # described as item M5 describes a site, at each cutoff of the site profile
    site_keys = []
    ceiling = site_pair_ceiling(hs, t)
    out["site_pair_ceiling"] = ceiling
    log(f"  the two end sites name {ceiling['pairs']} distinct pairs for {len(t)} hops, "
        f"{ceiling['shared']} of which share their pair with another hop (largest group "
        f"{ceiling['largest']}), so no descriptor of the two end sites alone can pass "
        f"q2 {ceiling['q2']:+.3f} (error {ceiling['rmse']:.3f} eV)")
    out["site_r"] = [float(r) for r in args.site_r]
    out["site_ref"] = float(args.site_r[len(args.site_r) // 2])
    for r in args.site_r:
        if abs(float(r) - args.r_max) < 1e-9:
            out["site_ref"] = float(r)      # the cutoff the hop environments themselves use
    for r in args.site_r:
        t1 = time.time()
        X = site_pair_features(hs, crystals, t.atom, r)
        sites = choose_lam(site_kernel(X), t, folds=args.folds, seed=args.seed,
                           name=f"the two end sites (M5), cutoff {r:g} A")
        log(f"  {sites.summary()}  ({X.shape[1]} channels, {time.time() - t1:.0f} s)")
        key = f"sites {r:g}"
        site_keys.append(key)
        out["models"].append(_entry(sites, extra={"key": key, "site_r_max": float(r),
                                                  "channels": int(X.shape[1])}))
        results[key] = sites
        for ref_name in ("counts", "geometry"):
            d, lo, hi = paired_interval(sites, results[ref_name], t, runs=args.bootstrap,
                                        seed=args.seed)
            out["pairs"].append({"a": key, "b": ref_name, "diff": d, "low": lo, "high": hi})
    best = None
    for rho in args.rho:
        envs = envs0 if rho == RHOS[1] else hs.environments(crystals, rows=t.atom, rho=rho,
                                                           r_max=args.r_max)
        inv = choose_lam(local_kernel(local_features(envs)), t, folds=args.folds, seed=args.seed,
                         name=f"invariant local, rho {rho:g}")
        log("  " + inv.summary())
        fr = framed_environments(hs, crystals, rows=t.atom, rho=rho, r_max=args.r_max)
        t1 = time.time()
        model = HopModel.build(fr, fs, n_jobs=args.jobs)
        cyl = cross_validate_hop_model(model, t, folds=args.folds, seed=args.seed,
                                       name=f"rotation-averaged overlap, rho {rho:g}")
        log(f"  {cyl.summary()}  (kernel {time.time() - t1:.0f} s, "
            f"{np.mean([e.n_atoms for e in envs]):.0f} atoms per environment)")
        atoms = float(np.mean([e.n_atoms for e in envs]))
        out["models"] += [_entry(inv, rho, {"key": f"invariant {rho}", "atoms": atoms}),
                          _entry(cyl, rho, {"key": f"overlap {rho}", "atoms": atoms})]
        results[f"invariant {rho}"] = inv
        results[f"overlap {rho}"] = cyl
        for name, r in ((f"invariant {rho}", inv), (f"overlap {rho}", cyl)):
            for ref_name in ("counts", "geometry") + tuple(site_keys):
                d, lo, hi = paired_interval(r, results[ref_name], t, runs=args.bootstrap,
                                            seed=args.seed)
                out["pairs"].append({"a": name, "b": ref_name, "diff": d, "low": lo, "high": hi})
        d, lo, hi = paired_interval(cyl, inv, t, runs=args.bootstrap, seed=args.seed)
        out["pairs"].append({"a": f"overlap {rho}", "b": f"invariant {rho}", "diff": d,
                             "low": lo, "high": hi})
        log(f"    overlap against invariant at the same width {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if best is None or cyl.q2 > best[0].q2:
            best = (cyl, rho, model, fr)
    cyl, rho, model, fr = best
    out["best_rho"] = rho
    # the co-field along and around the hop, with the error the out-of-fold residuals allow
    model.fit(t.y, lam=cyl.lam)
    out["profile"] = {"z": list(PROFILE_Z), "r": list(PROFILE_R), "sigma": cyl.error, "fields": {}}
    for f, name in enumerate(fs.names):
        v, sd = model.profile(f, PROFILE_Z, PROFILE_R, sigma=cyl.error)
        out["profile"]["fields"][name] = {"value": v.tolist(), "sd": sd.tolist()}
    # the learning curve of the best model, on the kernel already in hand
    curve = []
    which = np.unique(t.molecule)
    rng = np.random.RandomState(args.seed)
    for n_struct in (25, 50, 100, 200, 400):
        if n_struct >= len(which):
            break
        q2s = []
        for rep in range(3):
            drawn = set(rng.choice(which, n_struct, replace=False).tolist())
            rows = np.array([i for i in range(len(t)) if int(t.molecule[i]) in drawn])
            sub = t.subset(rows)
            r = choose_lam(model.K[np.ix_(rows, rows)], sub, LAMS, folds=min(args.folds, n_struct),
                           seed=args.seed + rep, name="subset")
            q2s.append(r.q2)
        curve.append({"structures": n_struct, "q2": float(np.mean(q2s)),
                      "spread": float(np.std(q2s))})
        log(f"    learning curve: {n_struct} structures, q2 {np.mean(q2s):+.3f} +- {np.std(q2s):.3f}")
    curve.append({"structures": int(len(which)), "q2": cyl.q2, "spread": 0.0})
    out["learning_curve"] = curve
    # percolation from the out-of-fold barriers against the labelled network
    predicted = {int(a): float(p) for a, p in zip(t.atom, cyl.predicted)}
    for i, h in enumerate(hs.hops):
        if i not in predicted and h.representative in predicted:
            predicted[i] = predicted[h.representative]
    perc = []
    for k in range(len(hs.structures)):
        rows = hs.of_structure(k)
        if not all(i in predicted for i in rows):
            continue
        if not all(np.isfinite(hs.hops[i].saddle) for i in rows):
            continue                                   # a structure not fully labelled
        ref_p = structure_percolation(hs, k)
        pred_p = structure_percolation(hs, k, saddle_from_barrier(hs, predicted))
        perc.append({"structure": hs.names[k], "labelled": ref_p, "predicted": pred_p,
                     "grid": hs.structures[k].info.get("percolation", {})})
    out["percolation"] = perc
    for dim in ("E_1D", "E_2D", "E_3D"):
        a = np.array([p["labelled"][dim] for p in perc])
        b = np.array([p["predicted"][dim] for p in perc])
        ok = np.isfinite(a) & np.isfinite(b)
        agree = int(np.sum(np.isfinite(a) == np.isfinite(b)))
        out[f"percolation_{dim}"] = {
            "n": int(ok.sum()), "finite_agreement": agree / max(len(a), 1),
            "r": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan"),
            "mae": float(np.mean(np.abs(a[ok] - b[ok]))) if ok.sum() else float("nan")}
        log(f"    percolation {dim}: {ok.sum()} structures with a finite threshold in both, "
            f"r {out[f'percolation_{dim}']['r']:+.3f}, mean absolute difference "
            f"{out[f'percolation_{dim}']['mae']:.3f} eV; finite or not agrees for "
            f"{agree}/{len(a)}")
    out["seconds"] = time.time() - t0
    return out, (hs, fs, model, cyl, rho)


def rebuild_public_model(args, log):
    """The public model at the stored best width, without the rest of part 1.

    Parts 2 and 3 need the fitted public model; with ``--reuse-public`` the stored
    ``public.json`` supplies the width and the ridge and only that one kernel is computed.
    """
    pub = json.load(open(os.path.join(RESULTS, f"public{run_tag(args)}.json")))
    rho = pub["best_rho"]
    best = next(m for m in pub["models"] if m["name"].startswith("rotation") and m["rho"] == rho)
    hs = load_prepared(args.prepared, args.max_structures, args.max_per_structure, args.seed)
    fs = fieldset_for(hs, "file")
    crystals = hs.crystals(fs)
    reps = [i for i, h in enumerate(hs.hops) if h.representative == i and np.isfinite(h.y)]
    t = hs.targets(reps)
    log(f"rebuilding the public model at rho {rho} on {len(t)} hops (ridge {best['lam']:g})")
    fr = framed_environments(hs, crystals, rows=t.atom, rho=rho, r_max=args.r_max)
    model = HopModel.build(fr, fs, n_jobs=args.jobs).fit(t.y, lam=best["lam"])
    cyl = LocalResult(name=best["name"], predicted=np.zeros(len(t)), q2=best["q2"],
                      error=best["rmse"], mae=best["mae"], lam=best["lam"], n_samples=len(t),
                      n_molecules=t.n_measured_molecules, folds=pub["folds"])
    return hs, fs, model, cyl, rho


#: the (z, r) grid of the fine co-field profile the figures of the manual are drawn from
FINE_Z = tuple(float(z) for z in np.round(np.arange(-3.0, 3.01, 0.25), 2))
FINE_R = tuple(float(r) for r in np.round(np.arange(0.0, 4.01, 0.2), 2))


def write_fine_profile(trained, log, path: Optional[str] = None) -> str:
    """The co-field of the public model on a fine (z, r) grid, with its standard deviation,
    for every field, written as ``profile_fine.json``; docs/make_figures.py draws it."""
    hs, fs, model, cyl, rho = trained
    out = {"rho": rho, "lam": cyl.lam, "q2": cyl.q2, "sigma": cyl.error, "n_samples": len(model.coef),
           "z": list(FINE_Z), "r": list(FINE_R), "fields": {}}
    for f, name in enumerate(fs.names):
        v, sd = model.profile(f, FINE_Z, FINE_R, sigma=cyl.error)
        out["fields"][name] = {"value": v.tolist(), "sd": sd.tolist()}
    path = path or os.path.join(RESULTS, "profile_fine.json")
    json.dump(out, open(path, "w"), indent=0)
    log(f"fine co-field profile of {len(fs.names)} fields on {len(FINE_Z)} x {len(FINE_R)} "
        f"points written to {path}")
    return path


def run_hosts(args, log, trained) -> dict:
    hs = load_hosts()
    public, fs_public, model, cyl, rho = trained
    log(hs.describe())
    fs = inorganic_fieldset([e for e in fs_public.names if e.startswith("cif:")
                             and e != "cif:other"] and
                            [s.name[4:] for s in fs_public.specs if s.is_cif
                             and s.name != "cif:other"],
                            physchem=PHYSCHEM, charge_model="oxidation",
                            other=any(s.name == "cif:other" for s in fs_public.specs))
    crystals = hs.crystals(fs)
    reps = hs.annotate(crystals)
    rows = [r for r in reps if np.isfinite(hs.hops[r].y)]
    out = {"n_hops": len(hs.hops), "n_distinct": len(reps), "n_labelled": len(rows)}
    if not rows:
        log("  no bond-valence labels for the hosts (examples/hops/hops_bvse.csv missing)")
        return out
    fr = framed_environments(hs, crystals, rows=rows, rho=rho, r_max=args.r_max)
    pred = model.predict(fr)
    y = np.array([hs.hops[r].y for r in rows])
    out["transfer"] = {"q2": r2_score(y, pred), "rmse": rmse(y, pred),
                       "r": float(np.corrcoef(y, pred)[0, 1]), "rho": rho}
    log(f"  transfer of the public model to the hosts' {len(rows)} distinct hops: "
        f"q2 {out['transfer']['q2']:+.3f}, error {out['transfer']['rmse']:.3f} eV, "
        f"r {out['transfer']['r']:+.3f}")
    per_host = {}
    for k, name in enumerate(hs.names):
        sel = [i for i, r in enumerate(rows) if hs.hops[r].structure == k]
        if len(sel) > 2:
            per_host[name] = {"n": len(sel), "r": float(np.corrcoef(y[sel], pred[sel])[0, 1]),
                              "mae": float(np.mean(np.abs(y[sel] - pred[sel])))}
    out["per_host"] = per_host
    # within the hosts: leave one host out, with the same kernel
    t = hs.targets(rows)
    own = HopModel.build(fr, fs)
    loo = cross_validate_hop_model(own, t, folds=3, seed=args.seed,
                                   name="rotation-averaged overlap, leave one host out")
    envs = hs.environments(crystals, rows=rows, rho=rho, r_max=args.r_max)
    ref = choose_lam(count_kernel(hop_count_features(envs, hop_types(envs, 5))), t, folds=3,
                     seed=args.seed, name="counts about the midpoint")
    log("  " + loo.summary())
    log("  " + ref.summary())
    out["leave_one_host_out"] = [_entry(loo, rho), _entry(ref)]
    # the two DFT hops
    dft = os.path.join(EXAMPLES, "dft_barriers.csv")
    listing = []
    if os.path.exists(dft):
        import csv
        for row in csv.DictReader(open(dft, newline="")):
            k = hs.names.index(row["structure"])
            off = (int(row["offset_x"]), int(row["offset_y"]), int(row["offset_z"]))
            i = next(i for i, h in enumerate(hs.hops) if h.structure == k and h.a == int(row["a"])
                     and h.b == int(row["b"]) and h.offset == off)
            h = hs.hops[i]
            e = framed_environments(hs, crystals, rows=[i], rho=rho, r_max=args.r_max)
            p = float(model.predict(e)[0])
            listing.append({"structure": row["structure"], "hop": f"{h.a} -> {h.b}",
                            "length": h.length, "dft": float(row["barrier"]),
                            "dft_reverse": float(row["barrier_reverse"]), "bvse": h.y,
                            "bvse_reverse": h.y_reverse, "predicted": p,
                            "bottleneck": h.bottleneck, "bottleneck_atoms": h.bottleneck_atoms})
            log(f"  {row['structure']:14s} {h.a:3d} -> {h.b:3d} ({h.length:.2f} A): DFT "
                f"{float(row['barrier']):.3f}, bond-valence {h.y:.3f}, predicted {p:.3f} eV")
    out["dft_hops"] = listing
    return out


def load_disordered() -> HopSet:
    """The disordered conductors with their bond-valence labels, as a HopSet."""
    from cmfqsar.periodic import read_structures
    structures, names = [], []
    for cid, name, _what in COD:
        path = os.path.join(DISORDERED, f"cod_{cid}.cif")
        if os.path.exists(path):
            structures.append(read_structures(path)[0])
            names.append(f"cod_{cid}")
    hs = HopSet.build(structures, names, "Li")
    labels = os.path.join(DISORDERED, "hops_bvse.csv")
    if os.path.exists(labels):
        read_hop_labels(labels, hs)
        spread_labels(hs)
    return hs


def run_disordered(args, log, trained) -> dict:
    hs = load_disordered()
    if not hs.structures:
        log("  no disordered set (examples/hops/disordered is empty)")
        return {}
    public, fs_public, model, cyl, rho = trained
    log(hs.describe())
    fs = inorganic_fieldset([s.name[4:] for s in fs_public.specs if s.is_cif
                             and s.name != "cif:other"],
                            physchem=PHYSCHEM, charge_model="oxidation",
                            other=any(s.name == "cif:other" for s in fs_public.specs))
    crystals = hs.crystals(fs)
    reps = hs.annotate(crystals)
    rows = [r for r in reps if np.isfinite(hs.hops[r].y)]
    out = {"n_structures": len(hs.structures), "n_hops": len(hs.hops), "n_distinct": len(reps),
           "n_labelled": len(rows), "structures": []}
    if not rows:
        log("  no bond-valence labels (examples/hops/disordered/hops_bvse.csv missing)")
        return out
    fr = framed_environments(hs, crystals, rows=rows, rho=rho, r_max=args.r_max)
    pred = model.predict(fr)
    y = np.array([hs.hops[r].y for r in rows])
    out["transfer"] = {"q2": r2_score(y, pred), "rmse": rmse(y, pred),
                       "r": float(np.corrcoef(y, pred)[0, 1]), "rho": rho}
    log(f"  transfer of the public model to {len(rows)} distinct hops of {len(hs.structures)} "
        f"structures: q2 {out['transfer']['q2']:+.3f}, error {out['transfer']['rmse']:.3f} eV, "
        f"r {out['transfer']['r']:+.3f}")
    t = hs.targets(rows)
    own = HopModel.build(fr, fs, n_jobs=args.jobs)
    loo = cross_validate_hop_model(own, t, folds=len(hs.structures), seed=args.seed,
                                   name="rotation-averaged overlap, leave one structure out")
    envs = hs.environments(crystals, rows=rows, rho=rho, r_max=args.r_max)
    ref = choose_lam(count_kernel(hop_count_features(envs, hop_types(envs, 5))), t,
                     folds=len(hs.structures), seed=args.seed, name="counts about the midpoint")
    log("  " + loo.summary())
    log("  " + ref.summary())
    d, lo, hi = paired_interval(loo, ref, t, runs=args.bootstrap, seed=args.seed)
    out["leave_one_out"] = [_entry(loo, rho), _entry(ref)]
    out["leave_one_out_pair"] = {"diff": d, "low": lo, "high": hi}
    log(f"    against the counts {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    predicted_public = {int(r): float(p) for r, p in zip(rows, pred)}
    predicted_loo = {int(a): float(p) for a, p in zip(t.atom, loo.predicted)}
    for pr in (predicted_public, predicted_loo):
        for i, h in enumerate(hs.hops):
            if i not in pr and h.representative in pr:
                pr[i] = pr[h.representative]
    what = {cid: (name, w) for cid, name, w in COD}
    for k, label in enumerate(hs.names):
        cid = label[4:]
        sel = [i for i, r in enumerate(rows) if hs.hops[r].structure == k]
        srows = hs.of_structure(k)
        full = all(np.isfinite(hs.hops[i].saddle) for i in srows)
        entry = {"cod": cid, "name": what[cid][0], "disorder": what[cid][1],
                 "n_hops": len(srows), "n_distinct": len([r for r in reps
                                                          if hs.hops[r].structure == k]),
                 "n_labelled": len(sel),
                 "r_public": float(np.corrcoef(y[sel], pred[sel])[0, 1]) if len(sel) > 2 else float("nan"),
                 "mae_public": float(np.mean(np.abs(y[sel] - pred[sel]))) if sel else float("nan"),
                 "mean_barrier": float(np.mean(y[sel])) if sel else float("nan"),
                 "percolation": {}}
        if full:
            entry["percolation"] = {
                "labelled": structure_percolation(hs, k),
                "public": structure_percolation(hs, k, saddle_from_barrier(hs, predicted_public)),
                "within": structure_percolation(hs, k, saddle_from_barrier(hs, predicted_loo))}
        out["structures"].append(entry)
        p = entry["percolation"].get("labelled", {})
        log(f"  {entry['name']:42s} {entry['n_hops']:4d} hops {entry['n_distinct']:3d} distinct, "
            f"r {entry['r_public']:+.2f}, mean barrier {entry['mean_barrier']:.2f} eV"
            + (f", E_1D/2D/3D {p['E_1D']:.2f}/{p['E_2D']:.2f}/{p['E_3D']:.2f}" if p else ""))
    return out


# ----------------------------------------------------------------------------- the table


def _fmt(x, nd=3, sign=True):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


# =========================================================================== #
# The model read through the interfaces of plan item MS7
# =========================================================================== #
def interfaces_run(args, log) -> dict:
    """Does the stored measurement come out the same when the model is read from outside?

    This is the gate of plan item MS7.  The public model is rebuilt at the stored width and ridge
    and then read twice: by the route this script uses -- environments built here, passed to
    ``HopModel.predict`` -- and through :class:`cmfqsar.interfaces.HopBarriers`, which is given a
    structure and does the rest itself.  Three things are compared: the barriers of the hosts'
    paths, the two barriers the stored table lists against DFT, and the percolation thresholds a
    conductivity estimate starts from.
    """
    from cmfqsar.interfaces import HopBarriers

    pub = json.load(open(os.path.join(RESULTS, f"public{run_tag(args)}.json")))
    rho = pub["best_rho"]
    best = next(m for m in pub["models"] if m["name"].startswith("rotation") and m["rho"] == rho)
    out = {"rho": rho, "lam": best["lam"], "r_max": args.r_max, "stored_q2": best["q2"],
           "n_train": best["n"]}
    t0 = time.time()
    hs_pub = load_prepared(args.prepared, args.max_structures, args.max_per_structure, args.seed)
    fs_pub = fieldset_for(hs_pub, "file")
    crystals = hs_pub.crystals(fs_pub)
    reps = [i for i, h in enumerate(hs_pub.hops) if h.representative == i and np.isfinite(h.y)]
    t = hs_pub.targets(reps)
    log(f"  rebuilding the public model at rho {rho:g} on {len(t)} hops "
        f"(ridge {best['lam']:g}); the kernel is the slow part")
    fr = framed_environments(hs_pub, crystals, rows=t.atom, rho=rho, r_max=args.r_max)
    model = HopModel.build(fr, fs_pub, n_jobs=args.jobs).fit(t.y, lam=best["lam"])
    out["seconds_model"] = float(time.time() - t0)
    log(f"  the public model is ready in {out['seconds_model'] / 60:.1f} min")

    # --- the hosts, which the public model has never seen
    hs = load_hosts()
    fs = inorganic_fieldset([s.name[4:] for s in fs_pub.specs
                             if s.is_cif and s.name != "cif:other"],
                            physchem=PHYSCHEM, charge_model="oxidation",
                            other=any(s.name == "cif:other" for s in fs_pub.specs))
    host_crystals = hs.crystals(fs)
    host_reps = hs.annotate(host_crystals)
    rows = [r for r in host_reps if np.isfinite(hs.hops[r].y)]
    log(f"  the three hosts: {len(hs.hops)} paths, {len(host_reps)} distinct, {len(rows)} "
        f"with a bond-valence barrier")
    # route A: the route this script uses
    t1 = time.time()
    direct = {int(r): float(v) for r, v in
              zip(rows, model.predict(framed_environments(hs, host_crystals, rows=rows,
                                                          rho=rho, r_max=args.r_max)))}
    seconds_direct = time.time() - t1
    # route B: the same model read through the interface, one structure at a time
    hb = HopBarriers(model, fs, rho=rho, r_max=args.r_max, mobile=hs.mobile)
    t1 = time.time()
    through: Dict[int, float] = {}
    tables = {}
    for k, name in enumerate(hs.names):
        # the network is handed over as it is, and the interface reports one of its structures
        table = hb.table(hopset=hs, structure=k, name=name)
        tables[name] = table
        for row in table.rows:
            if int(row["path"]) in direct:
                through[int(row["path"])] = float(row["barrier"])
    seconds_through = time.time() - t1
    shared = sorted(set(direct) & set(through))
    gaps = [abs(direct[i] - through[i]) for i in shared]
    out["hosts"] = {
        "n_paths": int(len(hs.hops)), "n_distinct": int(len(host_reps)),
        "n_compared": int(len(shared)),
        "largest_difference": float(max(gaps)) if gaps else float("nan"),
        "seconds_direct": float(seconds_direct), "seconds_through": float(seconds_through)}
    log(f"    {len(shared)} of the hosts' distinct paths were predicted by both routes: the "
        f"largest difference is {out['hosts']['largest_difference']:.1e} eV "
        f"({seconds_direct:.0f} s this script's way, {seconds_through:.0f} s through the "
        f"interface)")

    # --- the two barriers the stored table lists against DFT
    stored = json.load(open(os.path.join(RESULTS, f"hosts{run_tag(args)}.json")))
    listing = []
    for row in stored.get("dft_hops", []):
        a, b = (int(x) for x in row["hop"].split("->"))
        table = tables.get(row["structure"])
        if table is None:
            continue
        k = hs.names.index(row["structure"])
        # the stored row names the path by its two sites and its length; the offset is not in it,
        # and two paths can share the sites with different images
        same = [i for i in hs.of_structure(k)
                if hs.hops[i].a == a and hs.hops[i].b == b
                and abs(hs.hops[i].length - float(row["length"])) < 1e-6]
        if not same or int(same[0]) not in table.barriers:
            listing.append({"structure": row["structure"], "hop": row["hop"],
                            "stored": row["predicted"], "through": None,
                            "difference": None, "representative": None,
                            "note": "this path is not in the network the interface was given"})
            continue
        i = int(same[0])
        # the stored number is this path's own prediction, so the interface is asked for this
        # path and not for the one that stands for it
        value = float(hb.table(hopset=hs, structure=k, paths=[i]).rows[0]["barrier"])
        rep = int(hs.hops[i].representative)
        grouped = float(table.barriers[i]) if i in table.barriers else float("nan")
        listing.append({"structure": row["structure"], "hop": row["hop"],
                        "stored": float(row["predicted"]), "through": value,
                        "difference": value - float(row["predicted"]),
                        "representative": None if rep == i else rep,
                        "through_its_group": None if rep == i else grouped, "note": ""})
        log(f"    {row['structure']:14s} {row['hop']:10s} stored {row['predicted']:.4f}, "
            f"through the interface {value:.4f} eV"
            + ("" if rep == i
               else f"; the path that stands for it gives {grouped:.4f}, which is what the "
                    f"network a simulation reads carries"))
    out["dft_hops"] = listing

    # --- the percolation thresholds: what a conductivity estimate starts from
    perc = []
    for k, name in enumerate(hs.names):
        rows_k = hs.of_structure(k)
        by_route = {int(i): direct.get(int(hs.hops[i].representative), float("nan"))
                    for i in rows_k}
        if not all(np.isfinite(v) for v in by_route.values()):
            log(f"    {name}: not every path has a bond-valence barrier, so this script's "
                f"route has no network to percolate; skipped")
            continue
        ref = structure_percolation(hs, k, saddle_from_barrier(hs, by_route))
        mine = tables[name].percolation()
        perc.append({"structure": name, "direct": ref, "through": mine,
                     "difference": {d: (mine[d] - ref[d]) if np.isfinite(mine[d])
                                    and np.isfinite(ref[d]) else None
                                    for d in ("E_1D", "E_2D", "E_3D")}})
        log(f"    {name:14s} " + "  ".join(
            f"{d}: {ref[d]:.3f} / {mine[d]:.3f}" for d in ("E_1D", "E_2D", "E_3D")))
    out["percolation"] = perc
    return out


def table_interfaces(res: dict) -> List[str]:
    """The section of the table that reads the gate of plan item MS7."""
    if not res:
        return []
    h = res["hosts"]
    lines = ["## The model read through the interfaces (plan item MS7)", "",
             f"The public model, rebuilt at the stored width (rho {res['rho']:g}) and ridge "
             f"({res['lam']:g}) on {res['n_train']} paths, is read twice: by the route this "
             f"script uses -- environments built here and handed to the model -- and through "
             f"`cmfqsar.interfaces.HopBarriers`, which is given a structure and builds the "
             f"environments itself. The structures are the three hosts of the lithium interface "
             f"study, which the public model has never seen.", "",
             "**The two routes give the same barriers.** The interface is a way in, not another "
             "model:", "",
             "| Paths of the hosts | Distinct by symmetry | Compared by both routes | Largest "
             "difference |",
             "| ---: | ---: | ---: | ---: |",
             f"| {h['n_paths']} | {h['n_distinct']} | {h['n_compared']} | "
             f"{h['largest_difference']:.1e} eV |", ""]
    if res.get("dft_hops"):
        lines += ["**The stored numbers come back.** The two paths the benchmark lists against a "
                  "DFT elastic-band barrier, as the stored table has them and as the interface "
                  "gives them:", "",
                  "| Host | Path | Stored | Through the interface | Difference |",
                  "| --- | --- | ---: | ---: | ---: |"]
        spread = []
        for row in res["dft_hops"]:
            if row["through"] is None:
                lines.append(f"| {row['structure']} | {row['hop']} | {row['stored']:.4f} | -- | "
                             f"{row['note']} |")
            else:
                lines.append(f"| {row['structure']} | {row['hop']} | {row['stored']:.4f} | "
                             f"{row['through']:.4f} | {row['difference']:+.1e} |")
                if row.get("through_its_group") is not None:
                    spread.append(abs(float(row["through_its_group"]) - float(row["through"])))
        lines += ["", "In eV."]
        if spread:
            lines += ["", f"Both of these paths are equivalent by symmetry to another one of "
                      f"their cell, so the network a simulation reads carries their group's "
                      f"barrier rather than their own; the two differ by up to "
                      f"{max(spread):.2f} eV, which is the error of the symmetry grouping and "
                      f"not of the interface. A caller who wants one particular jump asks for "
                      f"that path by name, which is what the column above reports."]
        lines += [""]
    if res.get("percolation"):
        lines += ["**And so do the percolation thresholds**, which is what a conductivity "
                  "estimate starts from: the lowest barrier at which the paths form a connected "
                  "network in one, two and three dimensions, by this script's route and through "
                  "the interface.", "",
                  "| Host | E_1D | E_2D | E_3D |",
                  "| --- | --- | --- | --- |"]
        for row in res["percolation"]:
            cells = []
            for d in ("E_1D", "E_2D", "E_3D"):
                a, b = row["direct"][d], row["through"][d]
                cells.append(f"{a:.3f} / {b:.3f}" if np.isfinite(a) and np.isfinite(b)
                             else ("none / none" if not np.isfinite(a) and not np.isfinite(b)
                                   else f"{a:.3f} / {b:.3f}"))
            lines.append(f"| {row['structure']} | " + " | ".join(cells) + " |")
        lines += ["", "In eV above the lowest site, by this script's route / through the "
                  "interface.", ""]
    lines += [f"The interface took {h['seconds_through']:.0f} s on these paths against "
              f"{h['seconds_direct']:.0f} s for the route inside this script. Both pay for the "
              f"same two things, the environments and the kernel against the training set; the "
              f"difference is that the interface works one structure at a time, so it evaluates "
              f"the kernel in three blocks instead of one.", ""]
    return lines


def write_table(results_dir: str = RESULTS, tag: str = "") -> str:
    """The table of one run, from the result files that carry its tag.

    ``run<tag>.json`` holds the command that produced those files; the table prints it, so that a
    reader can repeat the run and so that a stored result with no command is visible as such.
    """
    pub = json.load(open(os.path.join(results_dir, f"public{tag}.json")))
    hosts_path = os.path.join(results_dir, f"hosts{tag}.json")
    hosts = json.load(open(hosts_path)) if os.path.exists(hosts_path) else None
    L = []
    L.append("# A migration path as the sample: the barrier of a Li hop from the field around it\n")
    prov = run_provenance(results_dir, tag)
    if prov:
        L.append("`" + " ".join(["python benchmarks/run_hops.py"] + list(prov["argv"])) + "`, "
                 f"reading `{prov['prepared']}` (written by `python benchmarks/prepare_hops.py`), "
                 f"seed {prov['seed']}, {prov['bootstrap']} bootstrap draws.\n")
    else:
        L.append("The command that produced these result files was not recorded.\n")
    L.append(f"Public set: {pub['n_structures']} ternary compounds of lithium and oxygen (Li, O and one more element) of the Materials Project "
             f"within {pub['hull']} eV/atom of the hull, {pub['n_hops']} Li hops of 1.5 to 4.5 A, "
             f"{pub['n_samples']} distinct by symmetry and labelled with the bond-valence "
             f"site-energy barrier; {pub['folds']} folds holding out whole structures; "
             f"environments within {pub['r_max']} A of the midpoint. Fields: "
             + ", ".join(pub["fields"]) + ".\n")
    site_ref = pub.get("site_ref")
    if pub.get("site_r"):
        L.append("The site-resolved reference describes each of the two end sites of a hop by the "
                 "row of the periodic autocorrelation of its cell that belongs to that site (plan "
                 "item M5) and combines the two rows as their sum and the absolute value of their "
                 "difference, so that the descriptor does not depend on which end is called "
                 "first. Cutoffs of the site profile: "
                 + ", ".join(f"{r:g}" for r in pub["site_r"])
                 + f" A. The column *against the two end sites* compares with the one at "
                   f"{site_ref:g} A, the radius the hop environments themselves use.\n")
    ceiling = pub.get("site_pair_ceiling")
    if ceiling:
        L.append(f"The two end sites name {ceiling['pairs']} distinct pairs for "
                 f"{pub['n_samples']} hops, and {ceiling['shared']} hops share their pair with "
                 f"another hop (largest group {ceiling['largest']}): a hop and its reverse, and "
                 f"two hops through different images of the cell, get one descriptor. Predicting "
                 f"the mean barrier of every group is therefore the limit of any descriptor of "
                 f"the two end sites alone, at q2 {_fmt(ceiling['q2'])} "
                 f"(error {ceiling['rmse']:.3f} eV).\n")
    L.append("| descriptor | envelope width rho | q2 | error (eV) | mean absolute error (eV) | "
             "ridge | against the counts | against length and bottleneck | "
             "against the two end sites |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    pairs = {(p["a"], p["b"]): p for p in pub["pairs"]}

    def against(name, ref):
        p = pairs.get((name, ref))
        return "--" if p is None else f"{_fmt(p['diff'])} [{_fmt(p['low'])}, {_fmt(p['high'])}]"

    for m in pub["models"]:
        rho = m.get("rho")
        key = m.get("key") or (("invariant " if m["name"].startswith("invariant")
                                else "overlap ") + f"{rho}" if rho is not None else None)
        site_col = against(key, f"sites {site_ref:g}") if key and site_ref is not None else "--"
        L.append(f"| {m['name']} | {rho if rho is not None else '--'} | {_fmt(m['q2'])} | "
                 f"{_fmt(m['rmse'], sign=False)} | {_fmt(m['mae'], sign=False)} | {m['lam']:g} | "
                 f"{against(key, 'counts') if key else '--'} | "
                 f"{against(key, 'geometry') if key else '--'} | {site_col} |")
    L.append("")
    same = [p for p in pub["pairs"] if p["a"].startswith("overlap") and p["b"].startswith("invariant")]
    if same:
        L.append("Overlap in the hop frame against the invariant kernel at the same width: "
                 + "; ".join(f"rho {p['a'].split()[1]}: {_fmt(p['diff'])} [{_fmt(p['low'])}, "
                             f"{_fmt(p['high'])}]" for p in same) + ".\n")
    for r in pub.get("site_r", []):
        row = [q for q in pub["pairs"]
               if q["a"].startswith("overlap") and q["b"] == f"sites {r:g}"]
        if row:
            L.append(f"Overlap in the hop frame against the two end sites at cutoff {r:g} A: "
                     + "; ".join(f"rho {q['a'].split()[1]}: {_fmt(q['diff'])} "
                                 f"[{_fmt(q['low'])}, {_fmt(q['high'])}]" for q in row) + ".\n")
    if pub.get("learning_curve"):
        L.append("Learning curve of the best overlap model (q2 against the number of training "
                 "structures, three random draws): "
                 + "; ".join(f"{c['structures']}: {_fmt(c['q2'])}" for c in pub["learning_curve"])
                 + ".\n")
    prof = pub.get("profile")
    if prof:
        L.append(f"## The co-field along the hop (z, A from the midpoint) and away from its axis "
                 f"(r, A), best model at rho {pub['best_rho']}\n")
        L.append(f"Values in eV of barrier per unit of normalised field, with the standard "
                 f"deviation the out-of-fold error ({prof['sigma']:.3f} eV) allows in brackets.\n")
        for name, table in prof["fields"].items():
            v, sd = np.array(table["value"]), np.array(table["sd"])
            if np.abs(v).max() < 1e-9:
                continue
            L.append(f"**{name}**\n")
            L.append("| z \\ r | " + " | ".join(f"{r:.0f}" for r in prof["r"]) + " |")
            L.append("| --- | " + " | ".join("---:" for _ in prof["r"]) + " |")
            for z, row, srow in zip(prof["z"], v, sd):
                L.append(f"| {z:+.0f} | " + " | ".join(f"{x:+.2f} ({s:.2f})" for x, s in zip(row, srow))
                         + " |")
            L.append("")
        if os.path.exists(os.path.join(results_dir, f"profile_fine{tag}.json")):
            L.append(f"The same co-field on a finer grid is `profile_fine{tag}.json`, written by "
                     "`python benchmarks/run_hops.py --fine-profile`; the manual draws it as a "
                     "figure.\n")
    L.append("## From hops to conduction: percolation thresholds of the predicted network\n")
    L.append("| direction | structures with a finite threshold in both | correlation | "
             "mean absolute difference (eV) | finite or not agrees |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for dim in ("E_1D", "E_2D", "E_3D"):
        p = pub.get(f"percolation_{dim}", {})
        L.append(f"| {dim[2:]} | {p.get('n', 0)} | {_fmt(p.get('r'))} | "
                 f"{_fmt(p.get('mae'), sign=False)} | {p.get('finite_agreement', 0):.0%} |")
    L.append("")
    if hosts:
        L.append("## The three hosts of the Li interface study\n")
        tr = hosts.get("transfer")
        if tr:
            L.append(f"The public model applied to the {hosts['n_labelled']} distinct hops of "
                     f"Li3PO4, Li3BP2O8 and LiBO2 (bond-valence labels): q2 {_fmt(tr['q2'])}, "
                     f"error {tr['rmse']:.3f} eV, correlation {_fmt(tr['r'])}.")
            for name, p in hosts.get("per_host", {}).items():
                L.append(f"- {name}: {p['n']} hops, correlation {_fmt(p['r'])}, "
                         f"mean absolute error {p['mae']:.3f} eV")
            L.append("")
        for m in hosts.get("leave_one_host_out", []):
            L.append(f"- leave one host out, {m['name']}: q2 {_fmt(m['q2'])}, error "
                     f"{m['rmse']:.3f} eV")
        L.append("")
        if hosts.get("dft_hops"):
            L.append("| host | hop | length (A) | DFT barrier (eV) | DFT reverse | bond-valence | "
                     "bond-valence reverse | predicted | bottleneck (A) |")
            L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for h in hosts["dft_hops"]:
                L.append(f"| {h['structure']} | {h['hop']} | {h['length']:.2f} | {h['dft']:.3f} | "
                         f"{h['dft_reverse']:.3f} | {h['bvse']:.3f} | {h['bvse_reverse']:.3f} | "
                         f"{h['predicted']:.3f} | {h['bottleneck']:.2f} "
                         f"({' '.join(h['bottleneck_atoms'])}) |")
            L.append("")
            L.append("The DFT barriers are plain elastic-band values for a vacancy hop into a "
                     "relaxed vacancy, so they are lower bounds and include the relaxation the "
                     "rigid bond-valence landscape does not; the columns are not expected to "
                     "agree in value, only in order.\n")
    dis_path = os.path.join(results_dir, f"disordered{tag}.json")
    dis = json.load(open(dis_path)) if os.path.exists(dis_path) else None
    if dis and dis.get("structures"):
        L.append("## The disordered conductors (plan item M2)\n")
        tr = dis.get("transfer", {})
        L.append(f"{dis['n_structures']} structures of the Crystallography Open Database, "
                 f"{dis['n_hops']} hops, {dis['n_distinct']} distinct, {dis['n_labelled']} with a "
                 f"bond-valence barrier. The public model applied as it stands: q2 {_fmt(tr.get('q2'))}, "
                 f"error {tr.get('rmse', float('nan')):.3f} eV, correlation {_fmt(tr.get('r'))}.")
        for m in dis.get("leave_one_out", []):
            L.append(f"- leave one structure out, {m['name']}: q2 {_fmt(m['q2'])}, error "
                     f"{m['rmse']:.3f} eV")
        pr = dis.get("leave_one_out_pair")
        if pr:
            L.append(f"- the frame kernel against the counts, leave one structure out: "
                     f"{_fmt(pr['diff'])} [{_fmt(pr['low'])}, {_fmt(pr['high'])}]")
        L.append("")
        L.append("| structure | what is disordered | hops | distinct | mean barrier (eV) | "
                 "correlation with the public model | E_1D / E_2D / E_3D labelled (eV) | "
                 "public model | within the set |")
        L.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |")

        def trio(d):
            if not d:
                return "--"
            return " / ".join("inf" if not np.isfinite(d[k]) else f"{d[k]:.2f}"
                              for k in ("E_1D", "E_2D", "E_3D"))

        for e in dis["structures"]:
            p = e.get("percolation", {})
            L.append(f"| {e['name']} (COD {e['cod']}) | {e['disorder']} | {e['n_hops']} | "
                     f"{e['n_distinct']} | {e['mean_barrier']:.2f} | {_fmt(e['r_public'], 2)} | "
                     f"{trio(p.get('labelled'))} | {trio(p.get('public'))} | {trio(p.get('within'))} |")
        L.append("")
        L.append("The bond-valence landscape of a disordered structure is that of its ordered "
                 "framework (the sites not held by Li, each with its majority species); the "
                 "fields see every site at its occupancy. A threshold of inf means the hops "
                 "between 1.5 and 4.5 A do not connect through the cell in that many directions.\n")
    iface_path = os.path.join(results_dir, f"interfaces{tag}.json")
    if os.path.exists(iface_path):
        L.append("")
        L += table_interfaces(json.load(open(iface_path)))
    text = "\n".join(L)
    with open(os.path.join(results_dir, f"table{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prepared", default=PREPARED)
    ap.add_argument("--max-structures", type=int, default=0)
    ap.add_argument("--max-per-structure", type=int, default=MAX_PER_STRUCTURE,
                    help=f"distinct hops kept per structure in the sample (default "
                         f"{MAX_PER_STRUCTURE}); the rest stay in the network for the "
                         f"percolation step")
    ap.add_argument("--jobs", type=int, default=8,
                    help="processes for the rotation-averaged kernel (default 8)")
    ap.add_argument("--rho", nargs="*", type=float, default=list(RHOS))
    ap.add_argument("--r-max", type=float, default=R_MAX)
    ap.add_argument("--site-r", nargs="*", type=float, default=list(SITE_R),
                    help="cutoffs in angstrom of the site profile the site-resolved reference is "
                         "taken at (default " + ", ".join(f"{r:g}" for r in SITE_R) + ")")
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    ap.add_argument("--interfaces", action="store_true",
                    help="the model read through the interfaces of plan item MS7, against the "
                         "numbers this script stores; nothing else is run")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-hosts", action="store_true")
    ap.add_argument("--reuse-public", action="store_true",
                    help="keep the stored results of the public set and recompute only the "
                         "best model, then run the hosts and the disordered conductors")
    ap.add_argument("--fine-profile", action="store_true",
                    help="recompute the best public model and write its co-field on a fine "
                         "(z, r) grid to results/hops/profile_fine<tag>.json, for the figures of "
                         "the manual; nothing else is run")
    return ap


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    tag = run_tag(args)
    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        print(write_table(RESULTS, tag))
        return
    t0 = time.time()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    if not os.path.exists(args.prepared):
        raise SystemExit(f"{args.prepared} is missing: run benchmarks/prepare_hops.py first")
    if args.fine_profile:
        path = os.path.join(RESULTS, f"profile_fine{tag}.json")
        write_fine_profile(rebuild_public_model(args, log), log, path)
        write_provenance(args, tag, argv, name="run_fine_profile",
                         minutes=round((time.time() - t0) / 60, 1))
        return
    if args.interfaces:
        # Only this stage's own two files are written, as --fine-profile does, so that no two
        # commands of the registry write the same file. The table takes the new section up when
        # it is next rebuilt (`--report`, or `run_all.py --tables`).
        log("The model read through the interfaces of plan item MS7")
        res = interfaces_run(args, log)
        json.dump(res, open(os.path.join(RESULTS, f"interfaces{tag}.json"), "w"), indent=1)
        write_provenance(args, tag, argv, name="run_interfaces",
                         minutes=round((time.time() - t0) / 60, 1))
        return
    if args.reuse_public:
        trained = rebuild_public_model(args, log)
    else:
        public, trained = run_public(args, log)
        json.dump(public, open(os.path.join(RESULTS, f"public{tag}.json"), "w"), indent=1)
    if not args.no_hosts:
        hosts = run_hosts(args, log, trained)
        json.dump(hosts, open(os.path.join(RESULTS, f"hosts{tag}.json"), "w"), indent=1)
        dis = run_disordered(args, log, trained)
        json.dump(dis, open(os.path.join(RESULTS, f"disordered{tag}.json"), "w"), indent=1)
    mode = "a" if args.reuse_public else "w"
    with open(os.path.join(RESULTS, f"log{tag}.txt"), mode, encoding="utf-8") as fh:
        fh.write(("\n\n" if mode == "a" else "") + "\n".join(lines))
    if args.reuse_public or args.no_hosts:
        # A part of the run rebuilds a part of the files, with the numbers the whole run gives.
        # The recorded command stays the one that rebuilds all of them.
        print(f"  run{tag}.json left as it is: this run rebuilt only a part of the results")
    else:
        write_provenance(args, tag, argv, minutes=round((time.time() - t0) / 60, 1))
    print(write_table(RESULTS, tag))


if __name__ == "__main__":
    main()
