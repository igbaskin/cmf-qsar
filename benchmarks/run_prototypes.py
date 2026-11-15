#!/usr/bin/env python
"""
Inverse design on prototypes (plan item M11).

Every measurement of this program so far asks how well a model predicts. A design loop asks
something else: given a budget of calculations, can the model find a better structure than the
ones it was trained on? `cmfqsar/prototypes.py` holds the three pieces -- a candidate generator
over a prototype, a test for which candidates are the same structure, and a loop that proposes,
computes and refits -- and this script measures them.

What is measured
----------------
**How large the design space is** (`--stage space`). The five-atom cubic ABO3 cell with the A and
B elements that `matbench_perovskites` uses: how many candidates the generator allows, how many
of them the data set holds, and therefore how much of the space has no computed formation energy.
The same stage reduces a defect prototype by its symmetry (one substituent on the sites of a
supercell) and checks the reduction against the descriptor channels, which is the test that
matters for a model: two cells with the same channels are one structure for it.

**Whether the loop finds a better structure than it started from** (`--stage loop`). This is the
gate of the item, stated on a published set so that the answer can be checked. A pool of
`matbench_perovskites` cells, a start set of 25 drawn at random, and a budget of 100 calculations;
the oracle reads the published formation energy and the loop may not see a value it has not asked
for. Six rules are run on the same start sets: picking at random is the baseline the others have
to beat. Two descriptors are run, the two-point autocorrelation and the three-body channels of
plan item M10, because a design loop is only as good as the model that ranks for it.

**The same question where the arrangement is the signal** (`--stage defects`). The 2DMD defect
series of plan item M14: one host lattice, 2,000 arrangements of one to three point defects, and
the design question a defect study asks -- which arrangement is the most stable. The kernels are
the ones `run_2dmd.py` and `run_threebody.py` already cached, so this stage costs minutes.

**How well the model orders what it has not computed** (`--stage ordering`). A rule that
exploits can only propose a good candidate if the model puts it near the top of its order. This
stage fits the same model on the same start sets and reports the rank correlation on the
candidates left out, together with the place the model gives the best candidate of the pool. It
is what explains where the loop succeeds and where it fails.

**A closed loop, with a calculator and not a table** (`--stage closed`). Candidates the generator
proposes on an fcc prototype, an ASE EMT calculator as the oracle, and a relaxation inside every
call. The whole pool is also computed exhaustively, which no real study can do, so the rank of
what the loop found is known exactly. Replacing EMT with a machine-learned potential or with VASP
changes the cost of a call and nothing else in the loop.

**The reference: a model that needs no field** (`--stage cheap`). The same loop, the same start
sets and the same rules, with a linear kernel on columns a chemist writes down without any
field: the names of the elements on the A, B and X sites of a perovskite, the counts of each
kind of defect, and the pair expansion of the defect tutorial. A design loop is judged against
this before the fields are credited with what it finds. For the defect pools the same stage runs
the difference-field kernel of the defect tutorial, which describes only the changed sites. The
stage is run separately and is not part of the default list of stages.

No new data is fetched: the Matbench files and the 2DMD archive are the ones those scripts use,
and the closed loop needs no data at all.

Usage
-----
    python benchmarks/run_prototypes.py                      # every stage, then the table
    python benchmarks/run_prototypes.py --stage space
    python benchmarks/run_prototypes.py --stage loop --pool 4000 --budget 100
    python benchmarks/run_prototypes.py --stage defects --materials MoS2
    python benchmarks/run_prototypes.py --stage ordering
    python benchmarks/run_prototypes.py --stage closed
    python benchmarks/run_prototypes.py --stage cheap      # the reference of the tutorial
    python benchmarks/run_prototypes.py --report
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
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_2dmd as D                                                     # noqa: E402
import run_reciprocal as M6                                              # noqa: E402

from cmfqsar import periodic, prototypes as P, threebody                 # noqa: E402

FEATURES = os.path.join(HERE, "cache", "prototypes")
RESULTS = os.path.join(HERE, "results", "prototypes")

#: the rules the loop is run with; "random" is the baseline the others have to beat
RULES: Tuple[str, ...] = ("random", "greedy", "ucb", "ei", "variance", "maxmin")
#: how many candidates the retrospective pool holds, how many calls the loop may make, and
#: how many start sets are drawn
POOL, BUDGET, SEEDS = 4000, 100, 10
#: how many cells are computed before the loop begins. Two sizes are run, because the model is
#: what ranks the candidates and at 25 cells it knows almost nothing about this pool
STARTS: Tuple[int, ...] = (25, 100)
#: how many standard deviations the confidence bound of "ucb" uses
BETA = 1.0
#: the ridges the loop chooses from, by the evidence, at every step
LAMS = (1e-4, 1e-3, 1e-2, 0.1, 1.0)
#: the descriptors of the retrospective pool: the two-point autocorrelation and the three-body
#: channels of plan item M10, at the settings their own measurements chose for this task
DESCRIPTORS: Tuple[str, ...] = ("two-point", "three-body")
AC_R_MAX, AC_SIGMA = 6.0, 0.5
TB_R_MAX, TB_STEP, TB_SIGMA = 4.0, 0.5, 0.35
#: how many elements per site the exhaustive duplicate test of the double perovskite uses
DOUBLE_SMALL = 4
#: the reaches the duplicate test is measured at: can a longer sum tell the arrangements apart
CHANNEL_REACH = (("two-point 6 A", periodic.PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5,
                                                         taper=M6.AC_TAPER, background=True)),
                 ("two-point 14 A", periodic.PeriodicSpec(r_max=14.0, step=0.5, sigma=0.5,
                                                          taper=M6.AC_TAPER, background=True)),
                 ("three-body 5 A", threebody.ThreeBodySpec(r_max=5.0, step=0.5, sigma=0.35)),
                 ("three-body 9 A", threebody.ThreeBodySpec(r_max=9.0, step=0.5, sigma=0.35)))
#: the defect series: the cells, and which cached kernel stands for each descriptor
DEFECT_LIMIT = 2000
DEFECT_KERNELS = {"two-point": ("autocorrelation", "sigma 0.5, linear"),
                  "three-body": ("threebody_r4_s0.75", "linear")}
#: the closed loop: the prototype, the elements, how many sites they take, and the budget.
#: The composition is fixed and only the arrangement varies. With three elements free to choose,
#: the composition carried almost all of the energy difference, and that is a search problem
#: random choice solves.
CLOSED_CELL = (2, 2, 2)
CLOSED_ELEMENTS = ("Au",)
CLOSED_COUNT = 8
#: how many placements are drawn before the symmetry reduction, and how many candidates the
#: pool keeps after it
CLOSED_SAMPLE = 3000
CLOSED_POOL, CLOSED_START, CLOSED_BUDGET, CLOSED_SEEDS = 400, 10, 30, 5


def substituents(elements: Sequence[str]) -> str:
    """How to name the elements a site class may take, inside a sentence."""
    return elements[0] if len(elements) == 1 else "one of " + ", ".join(elements)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def result(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=M6._default)


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cached(path: str, build):
    """A feature matrix, computed once and kept in benchmarks/cache/prototypes."""
    if os.path.isfile(path):
        with np.load(path) as z:
            return z["F"], [int(b) for b in z["blocks"]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    F, blocks = build()
    np.savez_compressed(path, F=F, blocks=np.asarray(blocks, int))
    log(f"  {os.path.basename(path)}: {F.shape[0]} x {F.shape[1]} in {time.time() - t0:.0f} s")
    return F, blocks


def spec_of(descriptor: str):
    """The spec of one descriptor of the retrospective pool."""
    if descriptor == "two-point":
        return periodic.PeriodicSpec(r_max=AC_R_MAX, step=AC_SIGMA, sigma=AC_SIGMA,
                                     taper=M6.AC_TAPER, background=True)
    if descriptor == "three-body":
        return threebody.ThreeBodySpec(r_max=TB_R_MAX, step=TB_STEP, sigma=TB_SIGMA)
    raise ValueError(f"unknown descriptor {descriptor!r}; one of {', '.join(DESCRIPTORS)}")


# --------------------------------------------------------------------------- #
# How large the design space is
# --------------------------------------------------------------------------- #
def cation_pairs(structures: Sequence) -> List[Tuple[str, str]]:
    """The two cations of every ABO3 cell, the one with six close anions first.

    After relaxation the sites are no longer at the ideal coordinates, so the B cation is found
    the way a crystallographer would: it is the one inside the oxygen octahedron.
    """
    out = []
    for atoms in structures:
        symbols = list(atoms.get_chemical_symbols())
        anions = [i for i, s in enumerate(symbols) if s == "O"]
        cations = [i for i, s in enumerate(symbols) if s != "O"]
        if len(cations) != 2 or not anions:
            continue
        near = []
        for c in cations:
            d = np.asarray([atoms.get_distance(c, a, mic=True) for a in anions], float)
            near.append(float(np.mean(np.sort(d)[:3])))
        b, a = (cations[0], cations[1]) if near[0] < near[1] else (cations[1], cations[0])
        out.append((symbols[a], symbols[b]))
    return out


def space_group(atoms, symprec: float = 1e-4) -> Tuple[int, tuple]:
    """(the space group number, the Wyckoff positions with their elements) of one cell."""
    import spglib

    cell = (np.asarray(atoms.get_cell()[:], float),
            np.asarray(atoms.get_scaled_positions(), float),
            np.asarray(atoms.get_atomic_numbers(), int))
    data = spglib.get_symmetry_dataset(cell, symprec=float(symprec))
    number = data.number if hasattr(data, "number") else data["number"]
    wyckoffs = data.wyckoffs if hasattr(data, "wyckoffs") else data["wyckoffs"]
    return int(number), tuple(sorted(zip(wyckoffs, atoms.get_atomic_numbers())))


def space(limit: int = 0, force: bool = False) -> dict:
    """How many candidates a prototype allows, and how many of them anyone has computed."""
    path = result("space.json")
    if os.path.isfile(path) and not force:
        return jload(path)
    out: dict = {}
    atoms, y, ids = periodic.structures_from_matbench(M6.fetch("perovskites"), target="e_form")
    if limit:
        atoms, y = atoms[:int(limit)], np.asarray(y, float)[:int(limit)]
    pairs = cation_pairs(atoms)
    a_sites = sorted({a for a, _b in pairs})
    b_sites = sorted({b for _a, b in pairs})
    proto = P.Prototype.by_symbol(P.cubic_perovskite(3.9), {"Sr": a_sites, "Ti": b_sites})
    held = {p for p in pairs}
    out["perovskites"] = {
        "cells": len(atoms), "distinct_pairs": len(held), "a_elements": len(a_sites),
        "b_elements": len(b_sites), "candidates": proto.n_candidates,
        "not_computed": proto.n_candidates - len(held),
        "fraction_computed": len(held) / proto.n_candidates,
        "a_list": a_sites, "b_list": b_sites}
    log(f"ABO3: {len(atoms)} cells hold {len(held)} distinct A/B pairs; the prototype with "
        f"{len(a_sites)} A and {len(b_sites)} B elements allows {proto.n_candidates}, so "
        f"{100 * len(held) / proto.n_candidates:.1f} % of the space is computed")

    # one more site class, and the space is no longer exhausted: the double perovskite A2BB'O6,
    # which is the same prototype doubled with its two B sites free of each other
    double = P.cubic_perovskite(3.9) * (2, 1, 1)
    b_one, b_two = [i for i, s in enumerate(double.get_chemical_symbols()) if s == "Ti"]
    a_cls = P.SiteClass("A", tuple(i for i, s in enumerate(double.get_chemical_symbols())
                                   if s == "Sr"), tuple(a_sites))
    proto2 = P.Prototype(double, (a_cls, P.SiteClass("B", (b_one,), tuple(b_sites)),
                                  P.SiteClass("B'", (b_two,), tuple(b_sites))),
                         name="A2BB'O6")
    small = P.Prototype(double, (P.SiteClass("A", a_cls.sites, tuple(a_sites[:DOUBLE_SMALL])),
                                 P.SiteClass("B", (b_one,), tuple(b_sites[:DOUBLE_SMALL])),
                                 P.SiteClass("B2", (b_two,), tuple(b_sites[:DOUBLE_SMALL]))),
                        name="A2BB'O6, a few elements")
    kept2, dropped2 = P.unique(small, P.candidates(small))
    out["double_perovskite"] = {
        "sites": len(double), "candidates": proto2.n_candidates, "computed": 0,
        "small_elements": DOUBLE_SMALL, "small_candidates": small.n_candidates,
        "small_distinct": len(kept2), "small_duplicates": dropped2}
    log(f"A2BB'O6: the same prototype doubled allows {proto2.n_candidates} candidates and the "
        f"repository has a computed formation energy for none of them; with {DOUBLE_SMALL} "
        f"elements per site the reduction is exhaustive: {small.n_candidates} candidates, "
        f"{len(kept2)} distinct")

    # a defect prototype: one substituent on the sites of a supercell, reduced by symmetry
    from ase.build import bulk

    host = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1)
    rows = []
    for count in (1, 2, 3):
        proto_d = P.Prototype(host, (P.SiteClass("Cu", tuple(range(len(host))), ("Au",),
                                                 count=count),))
        pool = P.candidates(proto_d)
        perms = P.site_permutations(proto_d)
        kept, dropped = P.unique(proto_d, pool, perms)
        rows.append({"count": count, "placements": len(pool), "operations": int(perms.shape[0]),
                     "distinct": len(kept), "duplicates": dropped})
        log(f"{count} substituted site(s) on {len(host)}: {len(pool)} placements, "
            f"{perms.shape[0]} operations, {len(kept)} distinct arrangements")
    out["defect_prototype"] = {"sites": len(host), "rows": rows}

    # does the descriptor agree with the symmetry reduction
    proto_d = P.Prototype(host, (P.SiteClass("Cu", tuple(range(len(host))), ("Au",), count=3),))
    kept, _ = P.unique(proto_d, P.candidates(proto_d))
    fs = periodic.inorganic_fieldset(elements=[], physchem=periodic.INORGANIC_FIELDS,
                                     charge_model="electronegativity", alpha=1.0)
    table = periodic.crystal_table([c.atoms for c in kept], fs, [c.label for c in kept])
    n_fields = table.weights.shape[1]
    checks = {}
    groups: Dict[str, List[List[int]]] = {}
    for name, spec in CHANNEL_REACH:
        keys = P.channel_keys(table.molecules, spec, n_fields)
        by_key: Dict[tuple, List[int]] = {}
        for i, key in enumerate(keys):
            by_key.setdefault(key, []).append(i)
        merged = [v for v in by_key.values() if len(v) > 1]
        checks[name] = {"distinct_by_symmetry": len(kept), "distinct_by_channels": len(by_key),
                        "merged": [list(g) for g in merged]}
        groups[name] = merged
        log(f"{name}: {len(by_key)} channel sets for {len(kept)} symmetry-distinct "
            f"arrangements; merged {[list(g) for g in merged]}")

    # what a merged pair costs: the structures are distinct, and their energies differ
    from ase.calculators.emt import EMT
    from ase.utils.structure_comparator import SymmetryEquivalenceCheck

    same = SymmetryEquivalenceCheck(to_primitive=True)
    pairs = []
    for group in sorted({tuple(g) for gs in groups.values() for g in gs}):
        energies = []
        for i in group:
            atoms = kept[i].atoms.copy()
            atoms.calc = EMT()
            energies.append(float(atoms.get_potential_energy()) / len(atoms))
        equivalent = all(same.compare(kept[group[0]].atoms.copy(), kept[j].atoms.copy())
                         for j in group[1:])
        types = [space_group(kept[i].atoms) for i in group]
        pairs.append({"members": [kept[i].label for i in group],
                      "equivalent_structures": bool(equivalent),
                      "space_groups": [n for n, _w in types],
                      "same_wyckoff": len({w for _n, w in types}) == 1,
                      "energies": energies,
                      "spread_mev": 1000.0 * (max(energies) - min(energies))})
        log(f"  {' and '.join(kept[i].label for i in group)}: the same structure? "
            f"{equivalent}; space groups {[n for n, _w in types]}; EMT energies differ by "
            f"{pairs[-1]['spread_mev']:.1f} meV per atom")
    out["channel_check"] = {"reaches": checks, "merged_pairs": pairs,
                            "arrangements": len(kept)}
    jdump(out, path)
    return out


# --------------------------------------------------------------------------- #
# The retrospective loop on a published set
# --------------------------------------------------------------------------- #
def pool_of(task: str, limit: int) -> Tuple[list, np.ndarray, object]:
    """(the candidates, the published target, the field table) of a Matbench pool."""
    table, y, _fs, _rows = M6.load_set(task, limit)
    pool = P.as_candidates(table.molecules, [m.name for m in table.molecules])
    return pool, np.asarray(y, float), table


def kernel_for(task: str, descriptor: str, table, limit: int) -> np.ndarray:
    """The cosine-normalised kernel of one descriptor over the pool."""
    n_fields = table.weights.shape[1]
    spec = spec_of(descriptor)
    if descriptor == "two-point":
        name = f"{task}_{limit}_ac_r{AC_R_MAX:g}_s{AC_SIGMA:g}.npz"
    else:
        name = f"{task}_{limit}_tb_r{TB_R_MAX:g}_s{TB_STEP:g}.npz"

    def build():
        F = P.feature_matrix(table.molecules, spec, n_fields,
                             **({"dtype": np.float32} if descriptor == "three-body" else {}))
        return F, P.block_sizes(spec, n_fields)

    F, blocks = cached(os.path.join(FEATURES, name), build)
    return P.pool_kernel(np.asarray(F, float), blocks)


def run_rules(pool: list, y: np.ndarray, K: np.ndarray, starts: Sequence[Sequence[int]],
              budget: int, maximise: bool, rules: Sequence[str] = RULES,
              label: str = "") -> dict:
    """Every rule on the same start sets, and what each one found.

    The loop is paired: a start set is drawn once and every rule begins from it, so the
    difference between two rules is not the luck of the start.
    """
    want = (np.max if maximise else np.min)
    better = (np.greater if maximise else np.less)
    cut = float(np.percentile(y, 99.0 if maximise else 1.0))   # the best one per cent of the pool
    best_overall = float(want(y))
    out: Dict[str, dict] = {}
    for rule in rules:
        runs = []
        for seed, start in enumerate(starts):
            oracle = P.LookupOracle(y)
            rec = P.design_loop(pool, oracle, start=start, budget=int(budget), kernel=K,
                                rule=rule, beta=BETA, lams=LAMS, maximise=maximise, seed=seed)
            found = float(rec["best"])
            rank = int(np.sum(better(y, found))) + 1        # 1 = the best cell of the pool
            runs.append({"seed": seed, "start_best": float(rec["start_best"]), "best": found,
                         "rank": rank, "beat_start": P.first_better(rec, rec["start_best"]),
                         "reached_top": P.first_better(rec, cut),
                         "calls": len(rec["steps"])})
        beat = [r for r in runs if r["beat_start"] is not None]
        top = [r for r in runs if r["reached_top"] is not None]
        out[rule] = {
            "runs": runs,
            "best_mean": float(np.mean([r["best"] for r in runs])),
            "best_best": float(want([r["best"] for r in runs])),
            "rank_median": float(np.median([r["rank"] for r in runs])),
            "beat_start_runs": len(beat), "of_runs": len(runs),
            "beat_start_median": (float(np.median([r["beat_start"] for r in beat]))
                                  if beat else None),
            "reached_top_runs": len(top),
            "reached_top_median": (float(np.median([r["reached_top"] for r in top]))
                                   if top else None)}
        log(f"  {label}{rule:9s} best {out[rule]['best_mean']:+.4f} (rank "
            f"{out[rule]['rank_median']:.0f} of {len(y)}), beat its start set in "
            f"{len(beat)}/{len(runs)} runs, reached the best 1 % in {len(top)}/{len(runs)}"
            + (f" after {out[rule]['reached_top_median']:.0f} calls" if top else ""))
    out["pool"] = {"n": len(y), "best": best_overall, "cut": cut,
                   "mean": float(np.mean(y)), "sd": float(np.std(y))}
    return out


def starts_for(n: int, size: int, seeds: int, seed0: int = M6.SPLIT_SEED) -> List[List[int]]:
    """One start set per seed, the same for every rule and every descriptor."""
    rng = np.random.default_rng(seed0)
    return [sorted(int(i) for i in rng.choice(n, size, replace=False)) for _ in range(seeds)]


def loop(task: str = "perovskites", limit: int = POOL, budget: int = BUDGET,
         seeds: int = SEEDS, descriptors: Sequence[str] = DESCRIPTORS,
         starts: Sequence[int] = STARTS, force: bool = False) -> dict:
    """The gate: does the loop find a better structure than the start set, within the budget."""
    path = result("loop.json")
    out = jload(path) if os.path.isfile(path) and not force else {}
    pool, y, table = pool_of(task, limit)
    log(f"=== {task}: {len(pool)} cells, formation energy {y.min():+.3f} to {y.max():+.3f} "
        f"eV/atom, {table.molecules[0].n_atoms} atoms per cell")
    for descriptor in descriptors:
        K = None
        for size in starts:
            key = f"{task} | {descriptor} | {limit} | start {size}"
            if key in out and not force:
                log(f"{key}: there")
                continue
            if K is None:
                K = kernel_for(task, descriptor, table, limit)
            log(f"--- {key}")
            t0 = time.time()
            rec = run_rules(pool, y, K, starts_for(len(pool), int(size), int(seeds)), budget,
                            maximise=False, label=f"{descriptor:11s}")
            rec.update({"task": task, "descriptor": descriptor, "limit": int(limit),
                        "budget": int(budget), "start": int(size), "seeds": int(seeds),
                        "seconds": time.time() - t0})
            out[key] = rec
            jdump(out, path)
        del K
    return out


# --------------------------------------------------------------------------- #
# The same question on the defect series
# --------------------------------------------------------------------------- #
def defect_kernel(material: str, descriptor: str, n: int) -> Optional[np.ndarray]:
    """One cached kernel of the defect series, cosine-normalised."""
    kind, setting = DEFECT_KERNELS[descriptor]
    path = D.kernel_path(material, n, kind, setting)
    if not os.path.isfile(path):
        log(f"  {os.path.basename(path)} is not cached; run run_2dmd.py and run_threebody.py "
            f"first")
        return None
    return D.cosine(np.load(path))


def defects(materials: Sequence[str] = tuple(D.MATERIALS), limit: int = DEFECT_LIMIT,
            budget: int = BUDGET, seeds: int = SEEDS,
            descriptors: Sequence[str] = DESCRIPTORS, starts: Sequence[int] = STARTS,
            force: bool = False) -> dict:
    """The design question of a defect study: which arrangement is the most stable."""
    path = result("defects.json")
    out = jload(path) if os.path.isfile(path) and not force else {}
    for material in materials:
        series = None
        for descriptor in descriptors:
            K = None
            for size in starts:
                key = f"{material} | {descriptor} | {limit} | start {size}"
                if key in out and not force:
                    log(f"{key}: there")
                    continue
                if series is None:
                    prep = D.prepare(material)
                    series = D.Series(prep, D.pick(prep, limit))
                    y = np.asarray(series.y, float)
                    log(f"=== {material}: {len(y)} cells, energy {y.min():+.3f} to "
                        f"{y.max():+.3f} eV")
                if K is None:
                    K = defect_kernel(material, descriptor, len(y))
                    if K is None:
                        break
                log(f"--- {key}")
                t0 = time.time()
                rec = run_rules(P.as_candidates([None] * len(y), series.names), y, K,
                                starts_for(len(y), int(size), int(seeds)), budget,
                                maximise=False, label=f"{descriptor:11s}")
                rec.update({"material": material, "descriptor": descriptor,
                            "limit": int(limit), "budget": int(budget), "start": int(size),
                            "seeds": int(seeds), "seconds": time.time() - t0})
                out[key] = rec
                jdump(out, path)
            del K
    return out


# --------------------------------------------------------------------------- #
# A closed loop, with a calculator and not a table
# --------------------------------------------------------------------------- #
def closed(pool_limit: int = CLOSED_POOL, budget: int = CLOSED_BUDGET,
           seeds: int = CLOSED_SEEDS, force: bool = False) -> dict:
    """Propose, compute and refit, with every energy computed inside the loop.

    The oracle is an ASE EMT calculator with a relaxation, so a call costs a tenth of a second
    instead of hours of VASP, and the whole pool can also be computed exhaustively. That is what
    makes the rank of what the loop found exact.
    """
    path = result("closed.json")
    if os.path.isfile(path) and not force:
        return jload(path)
    from ase.build import bulk
    from ase.calculators.emt import EMT

    host = bulk("Cu", "fcc", a=3.6, cubic=True) * CLOSED_CELL
    proto = P.Prototype(host, (P.SiteClass("Cu", tuple(range(len(host))), CLOSED_ELEMENTS,
                                           count=CLOSED_COUNT),), name="fcc Cu")
    log(f"=== the closed loop: {len(host)} sites, {CLOSED_COUNT} of them taking "
        f"{substituents(CLOSED_ELEMENTS)}, {proto.n_candidates} placements")
    pool = P.candidates(proto, limit=CLOSED_SAMPLE, seed=M6.SPLIT_SEED)
    pool, dropped = P.unique(proto, pool)
    log(f"a sample of {len(pool) + dropped} placements leaves {len(pool)} distinct candidates "
        f"({dropped} duplicates)")
    if len(pool) > int(pool_limit):
        rng = np.random.default_rng(M6.SPLIT_SEED)
        take = sorted(int(i) for i in rng.choice(len(pool), int(pool_limit), replace=False))
        pool = [pool[i] for i in take]
        log(f"a sample of {len(pool)} of them is used")
    pool = [P.Candidate(i, c.label, c.assignment, c.atoms) for i, c in enumerate(pool)]

    # the exhaustive reference, which a real study cannot have
    t0 = time.time()
    reference = P.CalculatorOracle(EMT, relax=True)
    truth = np.asarray([reference(c) for c in pool], float)
    log(f"every candidate relaxed with EMT in {time.time() - t0:.0f} s "
        f"({reference.seconds / max(reference.calls, 1):.3f} s per call); energy "
        f"{truth.min():+.4f} to {truth.max():+.4f} eV per atom")

    fs = periodic.inorganic_fieldset(elements=[], physchem=periodic.INORGANIC_FIELDS,
                                     charge_model="electronegativity", alpha=1.0)
    table = periodic.crystal_table([c.atoms for c in pool], fs, [c.label for c in pool])
    spec = periodic.PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5, taper=M6.AC_TAPER,
                                 background=True)
    n_fields = table.weights.shape[1]
    F = P.feature_matrix(table.molecules, spec, n_fields)
    K = P.pool_kernel(F, P.block_sizes(spec, n_fields))
    starts = starts_for(len(pool), CLOSED_START, int(seeds))

    out: dict = {"sites": len(host), "elements": list(CLOSED_ELEMENTS), "count": CLOSED_COUNT,
                 "placements": proto.n_candidates, "pool": len(pool), "budget": int(budget),
                 "start": CLOSED_START, "seeds": int(seeds),
                 "reference_seconds": float(reference.seconds),
                 "seconds_per_call": float(reference.seconds / max(reference.calls, 1)),
                 "best": float(truth.min()), "worst": float(truth.max()),
                 "spread": float(truth.max() - truth.min()), "rules": {}}
    for rule in RULES:
        runs = []
        for seed, start in enumerate(starts):
            oracle = P.CalculatorOracle(EMT, relax=True)
            rec = P.design_loop(pool, oracle, start=start, budget=int(budget), kernel=K,
                                rule=rule, beta=BETA, lams=LAMS, maximise=False, seed=seed)
            found = float(rec["best"])
            runs.append({"seed": seed, "best": found,
                         "rank": int(np.sum(truth < found)) + 1,
                         "start_best": float(rec["start_best"]),
                         "found_global": bool(abs(found - truth.min()) < 1e-9),
                         "beat_start": P.first_better(rec, rec["start_best"]),
                         "calls": int(oracle.calls)})
        got = [r for r in runs if r["found_global"]]
        out["rules"][rule] = {
            "runs": runs, "best_mean": float(np.mean([r["best"] for r in runs])),
            "rank_median": float(np.median([r["rank"] for r in runs])),
            "found_global_runs": len(got), "of_runs": len(runs)}
        log(f"  {rule:9s} best {out['rules'][rule]['best_mean']:+.4f} eV per atom (rank "
            f"{out['rules'][rule]['rank_median']:.0f} of {len(pool)}), found the best of the "
            f"pool in {len(got)}/{len(runs)} runs")
    jdump(out, path)
    return out


# --------------------------------------------------------------------------- #
# How well the model orders what it has not computed
# --------------------------------------------------------------------------- #
def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rank correlation: Pearson's correlation of the two sets of ranks."""
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    if len(ra) < 3:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ordering(seeds: int = SEEDS, starts: Sequence[int] = STARTS,
             descriptors: Sequence[str] = DESCRIPTORS, limit: int = POOL,
             defect_limit: int = DEFECT_LIMIT, force: bool = False) -> dict:
    """What a rule that exploits has to work with: the model's order of the untried candidates.

    The same start sets as the loop, the same kernels and the same choice of the ridge. On the
    candidates the start set leaves out this stage reports the rank correlation between
    prediction and truth, the fraction of the variance explained, and where the model puts the
    best candidate of the pool. The last number is the one a greedy rule acts on: if the model
    puts the best candidate at place 500, the loop will not reach it within 100 calls.
    """
    from cmfqsar.evidence import log_evidence

    path = result("ranking.json")
    out = jload(path) if os.path.isfile(path) and not force else {}
    pools: List[Tuple[str, int]] = [("perovskites", int(limit))]
    pools += [(m, int(defect_limit)) for m in D.MATERIALS]
    for name, n_limit in pools:
        y = table = None
        for descriptor in descriptors:
            K = None
            for size in starts:
                key = f"{name} | {descriptor} | {n_limit} | start {size}"
                if key in out and not force:
                    log(f"{key}: there")
                    continue
                if y is None:
                    if name == "perovskites":
                        _pool, y, table = pool_of(name, n_limit)
                    else:
                        prep = D.prepare(name)
                        series = D.Series(prep, D.pick(prep, n_limit))
                        y = np.asarray(series.y, float)
                    log(f"=== {name}: {len(y)} candidates")
                if K is None:
                    K = (kernel_for(name, descriptor, table, n_limit) if table is not None
                         else defect_kernel(name, descriptor, len(y)))
                    if K is None:
                        break
                rows = []
                best = int(np.argmin(y))
                for seed, start in enumerate(starts_for(len(y), int(size), int(seeds))):
                    h = np.asarray(sorted(set(int(i) for i in start)), int)
                    p = np.asarray([i for i in range(len(y)) if i not in set(h.tolist())], int)
                    full = np.zeros(len(y))
                    full[h] = y[h]
                    pick = None
                    for lam in LAMS:
                        e = log_evidence(K[np.ix_(h, h)], y[h], float(lam), fit_bias=True)
                        if np.isfinite(e) and (pick is None or e > pick[0]):
                            pick = (float(e), float(lam))
                    lam = pick[1] if pick else float(LAMS[-1])
                    mu, _sd = P._posterior(K, full, h, p, lam, True)
                    resid = float(np.sum((mu - y[p]) ** 2))
                    spread = float(np.sum((y[p] - np.mean(y[p])) ** 2))
                    place = (None if best in set(h.tolist())
                             else int(np.sum(mu < mu[list(p).index(best)])) + 1)
                    rows.append({"seed": seed, "lam": float(lam),
                                 "spearman": rank_correlation(mu, y[p]),
                                 "r2": float(1.0 - resid / spread) if spread > 0 else None,
                                 "place_of_best": place, "of": int(len(p))})
                places = [r["place_of_best"] for r in rows if r["place_of_best"] is not None]
                rec = {"pool": name, "descriptor": descriptor, "limit": int(n_limit),
                       "start": int(size), "seeds": int(seeds), "n": int(len(y)),
                       "spearman_median": float(np.median([r["spearman"] for r in rows])),
                       "r2_median": float(np.median([r["r2"] for r in rows
                                                     if r["r2"] is not None])),
                       "place_median": float(np.median(places)) if places else None,
                       "place_best": int(min(places)) if places else None,
                       "runs": rows}
                out[key] = rec
                jdump(out, path)
                log(f"  {descriptor:11s} start {size:3d}: rank correlation "
                    f"{rec['spearman_median']:+.3f}, R2 {rec['r2_median']:+.3f}, the best "
                    f"candidate is put at place "
                    + ("--" if rec["place_median"] is None
                       else f"{rec['place_median']:.0f} of {len(y) - size}"))
            del K
    return out


# --------------------------------------------------------------------------- #
# The cheap reference: the same loop with a model that needs no field
# --------------------------------------------------------------------------- #
#: the models of the stage `cheap`, per pool: a linear kernel on columns that a chemist would
#: write down without any field. Perovskites: which element sits on A, which on B, and how
#: many X sites each anion takes (the names of the elements). Defects: how many defects of
#: each kind the cell holds, and the pair expansion of the defect tutorial (pairs of kinds in
#: each shell of host-site distances up to 12 A).
CHEAP_MODELS = {"perovskites": ("element names",),
                "defects": ("defect counts", "pair expansion", "difference fields")}
#: the field model of the defect tutorial, run in the same stage for the defect pools: the
#: overlap of the difference fields (the changed sites only), averaged over the symmetry of the
#: host with a Gaussian layer, as run_2dmd.py caches it. The setting is fixed in the middle of
#: that benchmark's grid, so that no measured energy chooses it. The whole-cell kernels of the
#: stage `defects` see a few defects as a small change of a large host signal; this one sees
#: only the defects.
DIFFERENCE_FIELDS = "difference fields"
DIFFERENCE_SETTING = ("arrangement", "alpha 1, gamma 1")
CHEAP_RULES: Tuple[str, ...] = ("greedy", "ucb")
CHEAP_PAIR_CUTOFF = 12.0


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """The linear kernel of columns scaled to a root mean square of one over the pool, divided
    by its mean diagonal, so that the ridges of the loop mean the same as for the field
    kernels. No column is centred and no row is normalised: a count must stay a count."""
    X = np.asarray(X, float)
    rms = np.sqrt(np.mean(X ** 2, axis=0))
    X = X[:, rms > 0] / rms[rms > 0]
    K = X @ X.T
    return K / float(np.mean(np.diag(K)))


def element_name_columns(task: str, limit: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(the columns of the names of the elements, the target, the column names) for the cells
    of the Matbench pool, in the order of :func:`pool_of`.

    Every cell of `matbench_perovskites` is a five-atom ABX3 cell with its atoms in one order:
    the A site, the B site and the three X sites. X is O, N, F or S, and many cells mix two of
    them. The columns are one for each element on the A site, one for each element on the B
    site, and for each element the number of X sites it takes (0 to 3)."""
    atoms, y, _ids = periodic.structures_from_matbench(M6.fetch(task),
                                                        target=M6.TASKS[task]["target"])
    rng = np.random.RandomState(0)                      # the rows of M6.load_set
    rows = np.sort(rng.choice(len(atoms), min(int(limit), len(atoms)), replace=False))
    sites = []
    for i in rows:
        s = list(atoms[i].get_chemical_symbols())
        if len(s) != 5:
            raise ValueError(f"cell {i} of the pool has {len(s)} atoms and not 5")
        sites.append(s)
    names = sorted({"A " + s[0] for s in sites} | {"B " + s[1] for s in sites}
                   | {"X " + x for s in sites for x in s[2:]})
    col = {n: k for k, n in enumerate(names)}
    X = np.zeros((len(sites), len(names)))
    for i, s in enumerate(sites):
        X[i, col["A " + s[0]]] = 1.0
        X[i, col["B " + s[1]]] = 1.0
        for x in s[2:]:
            X[i, col["X " + x]] += 1.0
    return X, np.asarray(y, float)[rows], names


def defect_columns(material: str, limit: int, model: str) -> Tuple[np.ndarray, np.ndarray]:
    """(the columns of one cheap model, the target) for the cells of a defect pool, in the
    order of :func:`defects`."""
    prep = D.prepare(material)
    recs = D.pick(prep, limit)
    y = np.array([r[D.TARGET] for r in recs], float)
    X, keys, _shells = D.pair_features(prep, recs, CHEAP_PAIR_CUTOFF)
    if model == "defect counts":
        X = X[:, [k for k, name in enumerate(keys) if name.startswith("n ")]]
    return X, y


def ordering_rows(K: np.ndarray, y: np.ndarray, starts: Sequence[Sequence[int]]) -> dict:
    """What :func:`ordering` measures, for one kernel and one list of start sets."""
    from cmfqsar.evidence import log_evidence

    best = int(np.argmin(y))
    rows = []
    for seed, start in enumerate(starts):
        h = np.asarray(sorted(set(int(i) for i in start)), int)
        hs = set(h.tolist())
        p = np.asarray([i for i in range(len(y)) if i not in hs], int)
        full = np.zeros(len(y))
        full[h] = y[h]
        pick = None
        for lam in LAMS:
            e = log_evidence(K[np.ix_(h, h)], y[h], float(lam), fit_bias=True)
            if np.isfinite(e) and (pick is None or e > pick[0]):
                pick = (float(e), float(lam))
        lam = pick[1] if pick else float(LAMS[-1])
        mu, _sd = P._posterior(K, full, h, p, lam, True)
        resid = float(np.sum((mu - y[p]) ** 2))
        spread = float(np.sum((y[p] - np.mean(y[p])) ** 2))
        place = None if best in hs else int(np.sum(mu < mu[list(p).index(best)])) + 1
        rows.append({"seed": seed, "lam": float(lam), "spearman": rank_correlation(mu, y[p]),
                     "r2": float(1.0 - resid / spread) if spread > 0 else None,
                     "place_of_best": place, "of": int(len(p))})
    places = [r["place_of_best"] for r in rows if r["place_of_best"] is not None]
    return {"spearman_median": float(np.median([r["spearman"] for r in rows])),
            "r2_median": float(np.median([r["r2"] for r in rows if r["r2"] is not None])),
            "place_median": float(np.median(places)) if places else None,
            "runs": rows}


def cheap_stage(seeds: int = SEEDS, starts: Sequence[int] = STARTS, limit: int = POOL,
                defect_limit: int = DEFECT_LIMIT, budget: int = BUDGET,
                materials: Sequence[str] = tuple(D.MATERIALS), force: bool = False) -> dict:
    """The reference of the design tutorial: the loop of the stages `loop` and `defects`, with
    the same start sets and the same choice of the ridge, but with a model that needs no field.
    The rule `random` does not use a model, so its numbers are those of the stored stages."""
    path = result("cheap.json")
    out = jload(path) if os.path.isfile(path) and not force else {}
    pools = [("perovskites", "perovskites", int(limit))]
    pools += [(m, "defects", int(defect_limit)) for m in materials]
    for name, kind, n_limit in pools:
        for model in CHEAP_MODELS[kind]:
            todo = [s for s in starts
                    if force or f"{name} | {model} | {n_limit} | start {s}" not in out]
            if not todo:
                log(f"{name} | {model}: there")
                continue
            if kind == "perovskites":
                X, y, _names = element_name_columns(name, n_limit)
                K, n_cols = linear_kernel(X), int(X.shape[1])
            elif model == DIFFERENCE_FIELDS:
                _X, y = defect_columns(name, n_limit, "defect counts")
                path_k = D.kernel_path(name, len(y), *DIFFERENCE_SETTING)
                if not os.path.isfile(path_k):
                    log(f"  {os.path.basename(path_k)} is not cached; run run_2dmd.py first")
                    continue
                K = np.load(path_k)
                K, n_cols = K / float(np.mean(np.diag(K))), 0
            else:
                X, y = defect_columns(name, n_limit, model)
                K, n_cols = linear_kernel(X), int(X.shape[1])
            log(f"=== {name} | {model}: {len(y)} candidates, {n_cols} columns")
            for size in todo:
                key = f"{name} | {model} | {n_limit} | start {size}"
                t0 = time.time()
                st = starts_for(len(y), int(size), int(seeds))
                rec = run_rules(P.as_candidates([None] * len(y), [str(i) for i in range(len(y))]),
                                y, K, st, budget, maximise=False, rules=CHEAP_RULES,
                                label=f"{model:18s}")
                rec.update({"pool_name": name, "model": model, "columns": n_cols,
                            "limit": int(n_limit), "budget": int(budget), "start": int(size),
                            "seeds": int(seeds), "ordering": ordering_rows(K, y, st),
                            "seconds": time.time() - t0})
                o = rec["ordering"]
                log(f"  start {size}: rank correlation {o['spearman_median']:+.3f}, R2 "
                    f"{o['r2_median']:+.3f}, the best candidate at place "
                    + ("--" if o["place_median"] is None else f"{o['place_median']:.0f}"))
                out[key] = rec
                jdump(out, path)
    return out


#: the cheap model of the closed loop: the number of pairs of substituent atoms in each of the
#: first shells of neighbours of the fcc host, which is the classic description of an alloy
CHEAP_CLOSED_SHELLS = 3


def closed_pool():
    """(the candidates, the exhaustive EMT energies, the seconds per call) of the closed loop,
    built as :func:`closed` builds them."""
    from ase.build import bulk
    from ase.calculators.emt import EMT

    host = bulk("Cu", "fcc", a=3.6, cubic=True) * CLOSED_CELL
    proto = P.Prototype(host, (P.SiteClass("Cu", tuple(range(len(host))), CLOSED_ELEMENTS,
                                           count=CLOSED_COUNT),), name="fcc Cu")
    pool = P.candidates(proto, limit=CLOSED_SAMPLE, seed=M6.SPLIT_SEED)
    pool, _dropped = P.unique(proto, pool)
    if len(pool) > CLOSED_POOL:
        rng = np.random.default_rng(M6.SPLIT_SEED)
        take = sorted(int(i) for i in rng.choice(len(pool), CLOSED_POOL, replace=False))
        pool = [pool[i] for i in take]
    pool = [P.Candidate(i, c.label, c.assignment, c.atoms) for i, c in enumerate(pool)]
    reference = P.CalculatorOracle(EMT, relax=True)
    truth = np.asarray([reference(c) for c in pool], float)
    return pool, truth, float(reference.seconds / max(reference.calls, 1)), host


def neighbour_pair_columns(pool, host, shells: int = CHEAP_CLOSED_SHELLS) -> np.ndarray:
    """For every candidate, the number of pairs of substituent atoms at each of the first
    `shells` distances of neighbours of the host lattice (minimum image)."""
    d = host.get_all_distances(mic=True)
    levels = np.unique(np.round(d[d > 0.1], 3))[:shells]
    X = np.zeros((len(pool), len(levels)))
    host_symbols = host.get_chemical_symbols()
    for i, c in enumerate(pool):
        s = c.atoms.get_chemical_symbols()
        sub = [k for k in range(len(s)) if s[k] != host_symbols[k]]
        for a in range(len(sub)):
            for b in range(a + 1, len(sub)):
                dd = round(float(d[sub[a], sub[b]]), 3)
                hit = np.where(np.abs(levels - dd) < 1e-3)[0]
                if len(hit):
                    X[i, int(hit[0])] += 1.0
    return X


def cheap_closed(force: bool = False) -> dict:
    """The closed loop of :func:`closed` with the counts of neighbour pairs as the model.

    The pool and the exhaustive energies are rebuilt; the best and the worst energy must equal
    those of the stored closed loop, which is checked. Each call of the loop is a lookup in the
    exhaustive energies, which are the same numbers a calculator call would return."""
    path = result("cheap.json")
    out = jload(path) if os.path.isfile(path) and not force else {}
    key = "closed loop | neighbour pairs"
    if key in out and not force:
        log(f"{key}: there")
        return out
    t0 = time.time()
    pool, truth, per_call, host = closed_pool()
    stored = jload(result("closed.json")) if os.path.isfile(result("closed.json")) else None
    if stored is not None:
        if (abs(stored["best"] - truth.min()) > 1e-9 or abs(stored["worst"] - truth.max()) > 1e-9
                or stored["pool"] != len(pool)):
            raise ValueError("the rebuilt pool of the closed loop is not the stored one")
    X = neighbour_pair_columns(pool, host)
    K = linear_kernel(X)
    starts = starts_for(len(pool), CLOSED_START, CLOSED_SEEDS)
    budget = int(stored["budget"]) if stored is not None else CLOSED_BUDGET   # as it was run
    rec = run_rules(pool, truth, K, starts, budget, maximise=False, rules=CHEAP_RULES,
                    label="neighbour pairs   ")
    for rule in CHEAP_RULES:
        runs = rec[rule]["runs"]
        rec[rule]["found_global_runs"] = sum(1 for r in runs if r["rank"] == 1)
    rec.update({"pool_name": "closed loop", "model": "neighbour pairs",
                "columns": int(X.shape[1]), "limit": len(pool), "budget": budget,
                "start": CLOSED_START, "seeds": CLOSED_SEEDS, "seconds_per_call": per_call,
                "same_pool_as_closed": stored is not None,
                "ordering": ordering_rows(K, truth, starts), "seconds": time.time() - t0})
    out[key] = rec
    jdump(out, path)
    o = rec["ordering"]
    log(f"  closed loop: rank correlation {o['spearman_median']:+.3f}, the best candidate at "
        f"place " + ("--" if o["place_median"] is None else f"{o['place_median']:.0f}"))
    return out


def cheap_section() -> List[str]:
    """The section of the table for the stage `cheap`, or nothing before it has run."""
    path = result("cheap.json")
    if not os.path.isfile(path):
        return []
    rec = jload(path)
    loops = {}
    for f in ("loop.json", "defects.json"):
        if os.path.isfile(result(f)):
            loops.update(jload(result(f)))
    lines = ["## The reference: the same loop with a model that needs no field (stage cheap)", "",
             "The loop of the two sections above, with the same start sets, the same budget and "
             "the same choice of the ridge, but its model is a linear kernel on columns that "
             "need no field. **element names**: which element sits on the A site, which on the "
             "B site, and how many of the three X sites each anion takes. **defect counts**: "
             "how many defects of each kind the cell holds. **pair expansion**: those counts "
             "and the pairs of kinds in each shell of host-site distances up to "
             f"{CHEAP_PAIR_CUTOFF:g} A, the reference of the defect tutorial. **difference fields** is not a cheap model: it "
             "is the field kernel of the defect tutorial (the overlap of the fields of the "
             "changed sites, averaged over the symmetry of the host, with a Gaussian layer; "
             f"{DIFFERENCE_SETTING[1]}, fixed). The three-body kernel of the sections above "
             "describes the whole cell, where a few defects are a small change of a large host "
             "signal. The rows of the three-body kernel and of random choice are repeated from "
             "the sections above.", "",
             "| The set | Model | Start | Rule | best found | Rank | beat its start set | "
             "reached the best 1 % | calls to reach it | place of the best candidate | rank "
             "correlation |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    rk = jload(result("ranking.json")) if os.path.isfile(result("ranking.json")) else {}

    def row(name, model, size, rule, v, n, order):
        top = (f"{v['reached_top_median']:.0f}" if v.get("reached_top_median") is not None
               else "--")
        place = ("--" if not order or order.get("place_median") is None
                 else f"{order['place_median']:.0f}")
        rho = "" if not order else f"{order['spearman_median']:+.3f}"
        return (f"| {name} | {model} | {size} | {rule} | {v['best_mean']:+.4f} | "
                f"{v['rank_median']:.0f} of {n} | {v['beat_start_runs']}/{v['of_runs']} | "
                f"{v['reached_top_runs']}/{v['of_runs']} | {top} | {place} | {rho} |")

    names = []
    for r in rec.values():
        if r["pool_name"] not in names and r["pool_name"] != "closed loop":
            names.append(r["pool_name"])
    names.sort(key=lambda n: (n != "perovskites", n))
    model_order = [m for models in CHEAP_MODELS.values() for m in models]
    for name in names:
        for size in STARTS:
            mine = sorted((r for r in rec.values()
                           if r["pool_name"] == name and r["start"] == size),
                          key=lambda r: model_order.index(r["model"]))
            if not mine:
                continue
            n_limit = mine[0]["limit"]
            field = loops.get(f"{name} | three-body | {n_limit} | start {size}")
            if field:
                n = field["pool"]["n"]
                lines.append(row(name, "random choice", size, "random", field["random"], n, None))
                order = rk.get(f"{name} | three-body | {n_limit} | start {size}")
                for rule in CHEAP_RULES:
                    lines.append(row(name, "three-body fields", size, rule, field[rule], n,
                                     order))
            for r in mine:
                for rule in CHEAP_RULES:
                    lines.append(row(name, r["model"], size, rule, r[rule], r["pool"]["n"],
                                     r["ordering"]))
    cl = rec.get("closed loop | neighbour pairs")
    if cl:
        lines += ["", f"The closed loop of the section above, with the number of pairs of Au "
                      f"atoms in each of the first {CHEAP_CLOSED_SHELLS} shells of neighbours "
                      "as the model, on the same pool and start sets. The rows of the field "
                      "kernel and of random choice are repeated from that section.", "",
                  "| Model | Rule | best found (eV per atom) | Rank | found the best of the "
                  "pool | place of the best candidate | rank correlation |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        fields = jload(result("closed.json"))["rules"] if os.path.isfile(
            result("closed.json")) else {}
        n = cl["limit"]
        for model, rule, v, order in (
                [("random choice", "random", fields.get("random"), None)]
                + [("two-point fields", r, fields.get(r), None) for r in CHEAP_RULES]
                + [("neighbour pairs", r, cl[r], cl["ordering"]) for r in CHEAP_RULES]):
            if not v:
                continue
            place = ("" if not order else "--" if order.get("place_median") is None
                     else f"{order['place_median']:.0f}")
            rho = "" if not order else f"{order['spearman_median']:+.3f}"
            lines.append(f"| {model} | {rule} | {v['best_mean']:+.4f} | "
                         f"{v['rank_median']:.0f} of {n} | {v['found_global_runs']}/"
                         f"{v['of_runs']} | {place} | {rho} |")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def report() -> str:
    """The table of every stage that has results."""
    lines = ["# Inverse design on prototypes (plan item M11)", ""]
    sp = result("space.json")
    if os.path.isfile(sp):
        rec = jload(sp)
        p = rec["perovskites"]
        lines += ["## How large the design space is", "",
                  f"The five-atom cubic ABO3 prototype with the {p['a_elements']} A and "
                  f"{p['b_elements']} B elements that `matbench_perovskites` uses allows "
                  f"{p['candidates']:,} candidates. The set holds {p['cells']:,} cells with "
                  f"{p['distinct_pairs']:,} distinct A/B pairs, so "
                  f"{100 * p['fraction_computed']:.1f} % of the space has a computed formation "
                  f"energy and {p['not_computed']:,} candidates have none.", ""]
        dp = rec.get("double_perovskite")
        if dp:
            lines += [f"One more site class and the space is no longer exhausted. The same "
                      f"prototype doubled, with its two B sites free of each other (A2BB'O6), "
                      f"allows {dp['candidates']:,} candidates, and the repository holds a "
                      f"computed formation energy for none of them. With "
                      f"{dp['small_elements']} elements per site the duplicate test runs over "
                      f"the whole enumeration: {dp['small_candidates']} candidates leave "
                      f"{dp['small_distinct']} distinct structures, the duplicates being the "
                      f"pairs that differ only in which B site holds which element.", ""]
        d = rec["defect_prototype"]
        lines += [f"Placements of one substituent on the {d['sites']} sites of an fcc supercell, "
                  f"reduced by the symmetry of the prototype:", "",
                  "| Substituted sites | Placements | Distinct arrangements | Duplicates |",
                  "|---:|---:|---:|---:|"]
        for row in d["rows"]:
            lines.append(f"| {row['count']} | {row['placements']} | {row['distinct']} | "
                         f"{row['duplicates']} |")
        lines.append("")
        cc = rec["channel_check"]
        lines += [f"Of those {cc['arrangements']} arrangements with three substituted sites, the "
                  f"descriptors separate fewer:", "",
                  "| Descriptor | Channel sets | Arrangements |", "|---|---:|---:|"]
        for name, c in cc["reaches"].items():
            lines.append(f"| {name} | {c['distinct_by_channels']} | "
                         f"{c['distinct_by_symmetry']} |")
        lines.append("")
        for pair in cc["merged_pairs"]:
            numbers = " and ".join(str(n) for n in pair["space_groups"])
            how = (f"the same space group, {pair['space_groups'][0]}, with different "
                   f"Wyckoff positions" if len(set(pair["space_groups"])) == 1
                   else "space groups " + numbers)
            lines.append(f"`{'` and `'.join(pair['members'])}` share their channels at every "
                         f"reach above, and they are not the same structure: ASE's symmetry "
                         f"comparator separates them, and they have {how}. An EMT calculation "
                         f"on the unrelaxed cells puts their energies "
                         f"{pair['spread_mev']:.1f} meV per atom apart, which is an error no "
                         f"model on these channels can avoid.")
        lines.append("")
    for name, title, unit in (("loop.json", "The gate: a published pool of perovskites",
                               "eV per atom"),
                              ("defects.json", "The defect series: which arrangement is the "
                                               "most stable", "eV")):
        path = result(name)
        if not os.path.isfile(path):
            continue
        rec = jload(path)
        keys = [k for k in sorted(rec) if isinstance(rec[k], dict) and "pool" in rec[k]]
        if not keys:
            continue
        first = rec[keys[0]]
        lines += [f"## {title}", "",
                  f"A budget of {first['budget']} calculations after the start set, "
                  f"{first['seeds']} start sets, and the same start sets for every rule and "
                  f"every descriptor. The target is minimised. **Start** is how many cells were "
                  f"computed before the loop began; **Rank** is where the best cell the loop "
                  f"found sits in the whole pool, 1 being the best cell there is.", "",
                  "| The set | Descriptor | Start | Rule | best found | Rank | beat its start "
                  "set | reached the best 1 % | calls to reach it |",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|"]
        for key in keys:
            r = rec[key]
            where = r.get("task") or r.get("material")
            for rule in RULES:
                if rule not in r:
                    continue
                v = r[rule]
                calls = ("--" if v["reached_top_median"] is None
                         else f"{v['reached_top_median']:.0f}")
                lines.append(f"| {where} | {r['descriptor']} | {r['start']} | {rule} | "
                             f"{v['best_mean']:+.4f} | "
                             f"{v['rank_median']:.0f} of {r['pool']['n']} | "
                             f"{v['beat_start_runs']}/{v['of_runs']} | "
                             f"{v['reached_top_runs']}/{v['of_runs']} | {calls} |")
        lines += ["", f"The best cell of each pool, which no rule is told: "
                      + ", ".join(f"{(rec[k].get('task') or rec[k].get('material'))} "
                                  f"{rec[k]['descriptor']} {rec[k]['pool']['best']:+.4f}"
                                  for k in keys[:1])
                      + f" {unit}.", ""]
    rk = result("ranking.json")
    if os.path.isfile(rk):
        rec = jload(rk)
        lines += ["## How well the model orders what it has not computed", "",
                  "The same start sets as the loop, the same kernels, and the ridge chosen the "
                  "same way. **Rank correlation** and **R squared** are measured on the "
                  "candidates the start set leaves out. **Place of the best candidate** is where "
                  "the model puts the best candidate of the pool in its own order, 1 meaning "
                  "first: it is what a rule that exploits acts on, and a budget of "
                  f"{BUDGET} calls cannot reach a candidate the model puts much further back.",
                  "",
                  "| The set | Descriptor | Start | rank correlation | R squared | place of the "
                  "best candidate |",
                  "|---|---|---:|---:|---:|---:|"]
        for key in sorted(rec):
            r = rec[key]
            place = ("--" if r["place_median"] is None
                     else f"{r['place_median']:.0f} of {r['n'] - r['start']}")
            lines.append(f"| {r['pool']} | {r['descriptor']} | {r['start']} | "
                         f"{r['spearman_median']:+.3f} | {r['r2_median']:+.3f} | {place} |")
        lines += [""]
    cl = result("closed.json")
    if os.path.isfile(cl):
        rec = jload(cl)
        lines += ["## A closed loop, with a calculator and not a table", "",
                  f"{rec['count']} of the {rec['sites']} sites of an fcc copper cell take "
                  f"{substituents(rec['elements'])}: {rec['placements']:,} placements, "
                  f"{rec['pool']} after the symmetry reduction and the sample. Every candidate "
                  f"is relaxed by an ASE EMT calculator inside the loop, at "
                  f"{rec['seconds_per_call']:.3f} s per call, and the whole pool is also "
                  f"computed exhaustively ({rec['reference_seconds']:.0f} s) so that the rank is "
                  f"exact. The energies span {rec['spread']:.4f} eV per atom.", "",
                  f"A start set of {rec['start']} candidates, a budget of {rec['budget']} "
                  f"calls, {rec['seeds']} start sets.", "",
                  "| Rule | best found (eV per atom) | Rank | found the best of the pool |",
                  "|---|---:|---:|---:|"]
        for rule in RULES:
            v = rec["rules"].get(rule)
            if not v:
                continue
            lines.append(f"| {rule} | {v['best_mean']:+.4f} | {v['rank_median']:.0f} of "
                         f"{rec['pool']} | {v['found_global_runs']}/{v['of_runs']} |")
        lines += ["", f"The best candidate of the pool is {rec['best']:+.4f} eV per atom.", ""]
    lines += cheap_section()
    text = "\n".join(lines) + "\n"
    with open(result("table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["space", "loop", "defects", "closed", "ordering", "cheap"],
                   default=None,
                   help="one stage; `cheap` is the reference of the design tutorial, the same "
                        "loop with a model that needs no field")
    p.add_argument("--task", default="perovskites")
    p.add_argument("--pool", type=int, default=POOL)
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--seeds", type=int, default=SEEDS)
    p.add_argument("--descriptors", nargs="*", default=list(DESCRIPTORS))
    p.add_argument("--starts", nargs="*", type=int, default=list(STARTS))
    p.add_argument("--materials", nargs="*", default=list(D.MATERIALS))
    p.add_argument("--defect-limit", type=int, default=DEFECT_LIMIT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args(argv)
    if a.report:
        print(report())
        return 0
    stages = [a.stage] if a.stage else ["space", "loop", "defects", "closed", "ordering"]
    if "space" in stages:
        space(force=a.force)
    if "loop" in stages:
        loop(a.task, a.pool, a.budget, a.seeds, a.descriptors, a.starts, force=a.force)
    if "defects" in stages:
        defects(a.materials, a.defect_limit, a.budget, a.seeds, a.descriptors, a.starts,
                force=a.force)
    if "ordering" in stages:
        ordering(a.seeds, a.starts, a.descriptors, a.pool, a.defect_limit, force=a.force)
    if "closed" in stages:
        closed(budget=a.budget if a.stage == "closed" else CLOSED_BUDGET, force=a.force)
    if "cheap" in stages:
        cheap_stage(a.seeds, a.starts, a.pool, a.defect_limit, a.budget, a.materials,
                    force=a.force)
        cheap_closed(force=a.force)
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
