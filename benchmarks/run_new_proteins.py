#!/usr/bin/env python
"""
Does a description of a protein site carry to a protein the model has not seen?  (plan item PP2)

PP1 to PP4 measured the change of folding stability of point mutants on four domains of the
mega-scale set (Tsuboyama et al., Nature 2023).  Four domains cannot say whether a description
that wins inside them also wins on a new protein.  PP2 repeats the measurement on 24 domains:
the four of PP1 and twenty more chosen by the same rules (`examples/make_mutant_examples.py`,
:data:`NEW_DOMAINS`), with whole domains held out.

The samples
-----------
Every measured position of every domain, four substitutions drawn per position (two draws), the
target ddG in kcal/mol.  Every description is joined with the three substitution columns (the change of volume,
hydropathy and charge).

Stage 1, from the wild-type structures alone
--------------------------------------------
* **three columns kernel** -- the substitution alone;
* **one count** -- the number of atoms around the site (PP1);
* **accessible area** -- the solvent-accessible area of the residue and its side chain (PP3);
* **field of the surroundings** -- PP1's context kernel (envelope rho 0.1, cosine), joined with
  the column kernel at the weights 0.25 and 0.5;
* **accessible area and field** -- the field joined with the area kernel instead.

Stage 2, from built mutants
---------------------------
`--build` builds the drawn mutants (`cmfqsar.mutants.build_mutant`, as PP1), relaxes the wild
type at each of their positions (`relax_wild_type`, as PP4) and splits every energy
(`energy_terms`).  The five energy columns of PP4 are then joined with the count and with the
area, on the samples that have them, next to the stage 1 references on the same samples.

Folds: whole domains, eight folds of three domains, three repeats; every choice (member of a
family, ridge) is made in inner folds of the same kind.  The interval of a difference is over the
repeats and the folds, as in PP1 to PP4.  The gate of the plan: a description joined with the
columns beats the columns, and a count, with domains held out.

Usage
-----
    python benchmarks/run_new_proteins.py --stage sites       # stage 1, about 2 hours
    python benchmarks/run_new_proteins.py --build --workers 8  # build, then both stages
    python benchmarks/run_new_proteins.py --stage energy     # stage 2 only
    python benchmarks/run_new_proteins.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import make_mutant_examples as MX  # noqa: E402
import run_mutants as M  # noqa: E402
import run_strain as ST  # noqa: E402
from cmfqsar import mutants as MU  # noqa: E402

DOMAINS = tuple(MX.ALL_DOMAINS)
RESULTS = os.path.join(ROOT, "benchmarks", "results", "new_proteins")
PER_POSITION = 4
DRAWS = (0, 1)
DOMAIN_FOLDS = 8
HOLDOUTS = ("domain",)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def drawn(draws=DRAWS, domains=DOMAINS) -> List[Tuple[str, str]]:
    """(domain, label) of every substitution some draw takes."""
    out = set()
    for draw in draws:
        s = M.SiteSamples({}, domains, PER_POSITION, draw)
        out.update(zip(s.domain, s.label))
    return sorted(out)


def jobs(draws=DRAWS, domains=DOMAINS) -> List[Tuple[str, str, str]]:
    labels = drawn(draws, domains)
    positions = sorted({(d, MU.Mutation.parse(lab).position) for d, lab in labels})
    return [("wt", d, str(p)) for d, p in positions] + [("mut", d, lab) for d, lab in labels]


def build_share(shard: int, shards: int, draws=DRAWS) -> None:
    """Every `shards`-th job from `shard` on: build the mutant if it is not built, then write
    the split energy; or relax the wild type at a position and write its split energy."""
    os.environ.setdefault("OPENMM_CPU_THREADS", "1")
    mine = jobs(draws)[shard::shards]
    todo = [j for j in mine if not os.path.exists(ST._terms_path(*j))]
    log(f"share {shard}: {len(mine)} jobs, {len(todo)} to do")
    t0, failed = time.time(), []
    for k, (kind, d, label) in enumerate(todo, 1):
        os.makedirs(os.path.join(ST.STRAIN, d), exist_ok=True)
        wt_pdb = os.path.join(M.BUILT, d, "wild_type.pdb")
        try:
            if kind == "wt":
                position = int(label)
                st = MU.relax_wild_type(wt_pdb, position)
                st.save(os.path.join(ST.STRAIN, d, f"wt{position}.npz"))
            else:
                m = MU.Mutation.parse(label)
                position = m.position
                path = os.path.join(M.BUILT, d, label + ".npz")
                if os.path.exists(path):
                    st = MU.Structure.load(path)
                else:
                    st = MU.build_mutant(wt_pdb, m)
                    st.save(path)
            terms = MU.energy_terms(st, position)
            terms.update(name=st.name, position=position, energy_before=st.energy_before,
                         energy_after=st.energy_after)
            with open(ST._terms_path(kind, d, label), "w", encoding="utf-8", newline="\n") as fh:
                json.dump(terms, fh, indent=1)
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            failed.append(f"{kind} {d} {label}: {type(exc).__name__}: {exc}")
        if k % 100 == 0 or k == len(todo):
            log(f"  share {shard}: {k}/{len(todo)} after {time.time() - t0:.0f} s, "
                f"{len(failed)} failed")
    if failed:
        with open(os.path.join(ST.STRAIN, f"failed_pp2_{shard}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(failed) + "\n")


def build(workers: int = 8) -> None:
    """The wild types are prepared here, once, before the shares start."""
    for d in DOMAINS:
        dest = os.path.join(M.BUILT, d)
        os.makedirs(dest, exist_ok=True)
        wt_pdb = os.path.join(dest, "wild_type.pdb")
        if not os.path.exists(wt_pdb):
            MU.prepare_wild_type(os.path.join(M.EXAMPLES, d, f"{d}.pdb"), wt_pdb).save(
                os.path.join(dest, "wild_type.npz"))
    env = dict(os.environ, OPENMM_CPU_THREADS="1",
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    procs = [subprocess.Popen([sys.executable, "-P", os.path.abspath(__file__),
                               "--build-shard", str(k), "--shards", str(workers)], env=env)
             for k in range(workers)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"building: exit codes {codes}")


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
SITE_COMPARISONS = (("one count", "three columns kernel"),
                    ("accessible area", "three columns kernel"),
                    ("field of the surroundings", "three columns kernel"),
                    ("field of the surroundings", "one count"),
                    ("accessible area", "one count"),
                    ("accessible area and field", "accessible area"))
ENERGY_COMPARISONS = (("energy terms", "three columns kernel"),
                      ("one count and energy terms", "one count"),
                      ("accessible area and energy terms", "accessible area"),
                      ("one count and energy terms", "accessible area"))


def site_kernels(s, K_field: np.ndarray, X_area: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    X_three = MU.substitution_features(s.mutation)[0][:, :3]
    Kc3 = M.gaussian_columns(X_three)
    count = M.burial_count_rows(s)[:, :1]
    K_area = M.gaussian_columns(np.c_[X_area, X_three])
    return {
        "three columns kernel": {"columns": Kc3},
        "one count": {"count": M.gaussian_columns(np.c_[count, X_three])},
        "accessible area": {"area": K_area},
        "field of the surroundings": {f"weight {w}": M.mixed(Kc3, K_field, w) for w in M.MIXES},
        "accessible area and field": {f"weight {w}": M.mixed(K_area, K_field, w)
                                      for w in M.MIXES},
    }


def energy_kernels(s, X_area: np.ndarray, X_energy: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    X_three = MU.substitution_features(s.mutation)[0][:, :3]
    count = M.burial_count_rows(s)[:, :1]
    lg = ST.signed_log(X_energy)
    return {
        "three columns kernel": {"columns": M.gaussian_columns(X_three)},
        "one count": {"count": M.gaussian_columns(np.c_[count, X_three])},
        "accessible area": {"area": M.gaussian_columns(np.c_[X_area, X_three])},
        "energy terms": ST._gauss_pair(X_energy, X_three, "energy"),
        "one count and energy terms": {
            "as it is": M.gaussian_columns(np.c_[count, X_energy, X_three]),
            "log scale": M.gaussian_columns(np.c_[count, lg, X_three])},
        "accessible area and energy terms": {
            "as it is": M.gaussian_columns(np.c_[X_area, X_energy, X_three]),
            "log scale": M.gaussian_columns(np.c_[X_area, lg, X_three])},
    }


def measure(s, kernels, comparisons, seeds, holdouts) -> dict:
    out = {}
    for by in holdouts:
        tg = s.targets(by)
        n_groups = int(tg.molecule.max()) + 1
        folds = min(DOMAIN_FOLDS, n_groups) if by == "domain" else min(M.FOLDS, n_groups)
        rows, keep = {}, {}
        res, chosen = M.mean_reference(tg, seeds, folds)
        rows["mean"], keep["mean"] = M.row_of(res, chosen, tg), res
        for name, fam in kernels.items():
            t0 = time.time()
            res, chosen = M.nested(fam, tg, seeds, name, folds)
            rows[name], keep[name] = M.row_of(res, chosen, tg), res
            rows[name]["per_domain"] = per_domain(res, s)
            log(f"    {by:8s} {name:34s} q2 {rows[name]['q2']:+.3f}  "
                f"({time.time() - t0:.0f} s)")
        diffs = {f"{a} - {b}": M.compare(keep[a], keep[b], tg) for a, b in comparisons}
        out[by] = {"rows": rows, "differences": diffs, "n_groups": n_groups, "folds": folds}
    return out


def per_domain(results, s) -> Dict[str, float]:
    """The correlation of the held-out predictions with ddG inside each domain, averaged over
    the repeats: how well the model orders the substitutions of one new protein."""
    y = np.asarray(s.y, float)
    dom = np.array(s.domain)
    out = {}
    for d in sorted(set(s.domain)):
        idx = dom == d
        r = [np.corrcoef(res.predicted[idx], y[idx])[0, 1] for res in results]
        out[d] = round(float(np.mean(r)), 3)
    return out


KERNEL_CACHE = os.path.join(ROOT, "benchmarks", "cache", "megascale",
                            "new_proteins_position_kernel.npz")


def position_kernel(contexts, cache: str = KERNEL_CACHE):
    """PP1's context kernel over every position (about 45 minutes for 24 domains), kept in
    `cache` with the positions it covers, and recomputed when they change."""
    keys = sorted({(d, p) for (r, d, p) in contexts if r == float(M.SITE_RHO)})
    names = np.array([f"{d}:{p}" for d, p in keys])
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as z:
            if z["names"].shape == names.shape and (z["names"] == names).all():
                return keys, np.asarray(z["K"], float)
    keys2, K = M.position_kernel(contexts)
    assert list(keys2) == list(keys)
    np.savez_compressed(cache, names=names, K=K)
    return keys, K


def _subset_contexts_ok(s, contexts) -> np.ndarray:
    return np.array([(float(M.SITE_RHO), d, int(p)) in contexts
                     for d, p in zip(s.domain, s.position)])


def run(stages=("sites", "energy"), draws=DRAWS, repeats: int = M.REPEATS, holdouts=HOLDOUTS,
        results: str = RESULTS) -> dict:
    import run_site_fields as SF
    os.makedirs(results, exist_ok=True)
    path = os.path.join(results, "new_proteins.json")
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            out = json.load(fh)
    out.update({"domains": list(DOMAINS), "per_position": PER_POSITION, "repeats": repeats,
                "domain_folds": DOMAIN_FOLDS})
    t0 = time.time()
    contexts = M.site_contexts(DOMAINS)
    keys, Kp = position_kernel(contexts)
    area = SF.accessible_area(DOMAINS)
    log(f"{len(keys)} positions of {len(DOMAINS)} domains; contexts, position kernel and "
        f"area after {time.time() - t0:.0f} s")
    out["positions"] = len(keys)
    seeds = list(range(repeats))
    wt = mut = ref = None
    if "energy" in stages:
        wt, mut = ST.read_terms(DOMAINS)
        ref = ST.residue_references(wt, mut) if mut else None
    for stage in stages:
        part = out.setdefault(stage, {"draws": {}})
        t_stage = time.time()
        for draw in draws:
            s = M.SiteSamples(contexts, DOMAINS, PER_POSITION, draw)
            s = ST.subset(s, _subset_contexts_ok(s, contexts))
            if stage == "energy":
                have = np.array([(d, int(p)) in wt and (d, lab) in mut
                                 for d, p, lab in zip(s.domain, s.position, s.label)])
                missing = int((~have).sum())
                if have.sum() < 0.5 * len(have):
                    log(f"--- energy, draw {draw}: only {int(have.sum())} of {len(have)} "
                        "substitutions have energies; run --build first")
                    continue
                s = ST.subset(s, have)
            X_area = np.array([area[(d, int(p))] for d, p in zip(s.domain, s.position)], float)
            K_field = M.expand_position_kernel(keys, Kp, s)
            if stage == "sites":
                kernels, comparisons, missing = site_kernels(s, K_field, X_area), \
                    SITE_COMPARISONS, 0
            else:
                X_energy, _total = ST.energy_rows(s, wt, mut, ref)
                kernels, comparisons = energy_kernels(s, X_area, X_energy), ENERGY_COMPARISONS
            del K_field
            log(f"--- {stage}, draw {draw}: {len(s)} substitutions at "
                f"{len(set(zip(s.domain, s.position)))} positions ({missing} left out), "
                f"ddG sd {s.y.std():.2f} kcal/mol")
            part["draws"][str(draw)] = {
                "n": len(s), "missing": missing,
                "positions": len(set(zip(s.domain, s.position))),
                "sd_y": round(float(s.y.std()), 4),
                "holdout": measure(s, kernels, comparisons, seeds, holdouts)}
            del kernels
            _write(out, path)
        part["seconds"] = round(time.time() - t_stage, 1)
        part["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write(out, path)
    log(f"done after {time.time() - t0:.0f} s")
    return out


def _write(out: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _iv(d: Sequence[float]) -> str:
    txt = f"{d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f})"
    return f"**{txt}**" if d[1] > 0 or d[2] < 0 else txt


STAGE_TITLES = {"sites": "Stage 1: descriptions from the wild-type structures",
                "energy": "Stage 2: the energy of the built mutants (plan item PP4)"}


def report(results: str = RESULTS) -> str:
    path = os.path.join(results, "new_proteins.json")
    if not os.path.exists(path):
        return "# PP2: no result yet\n"
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    L = ["# Does a description of a site carry to a new protein? (plan item PP2)\n",
         f"`python benchmarks/run_new_proteins.py`. {len(d['domains'])} domains of the mega-scale "
         f"set ({', '.join(d['domains'])}), {d.get('positions', '?')} positions, "
         f"{d['per_position']} substitutions drawn per position. Whole domains are held out, "
         f"{d['domain_folds']} folds, {d['repeats']} repeats; every choice is made in inner "
         "folds of the same kind. Every description is joined with the change of volume, "
         "hydropathy and charge.\n"]
    for stage in ("sites", "energy"):
        if stage not in d or not d[stage].get("draws"):
            continue
        part = d[stage]
        L.append(f"## {STAGE_TITLES[stage]}\n")
        L.append(f"Written {part.get('written', '?')}, {part.get('seconds', 0) / 60:.0f} "
                 "minutes.\n")
        for draw, dd in part["draws"].items():
            L.append(f"### Draw {draw}: {dd['n']} substitutions at {dd['positions']} positions"
                     + (f" ({dd['missing']} without energies left out)" if dd['missing'] else "")
                     + f", sd of ddG {dd['sd_y']:.2f} kcal/mol\n")
            for h, hd in dd["holdout"].items():
                L.append(f"q2 with {h}s held out, and the median over the domains of the "
                         "correlation of the held-out predictions with ddG inside a domain "
                         "(with the lowest and the highest domain):\n")
                L.append("| Description | q2 | Median r inside a domain | Lowest and highest |")
                L.append("|---|---:|---:|---|")
                for name, row in hd["rows"].items():
                    pdm = row.get("per_domain")
                    if pdm:
                        v = np.array(list(pdm.values()))
                        L.append(f"| {name} | {row['q2']:+.3f} | {np.median(v):+.3f} | "
                                 f"{v.min():+.3f} to {v.max():+.3f} |")
                    else:
                        L.append(f"| {name} | {row['q2']:+.3f} | | |")
                L.append("")
                L.append("Differences in q2, with the 95 % interval over the repeats and the "
                         "folds; bold where it excludes zero.\n")
                L.append("| Difference | Value (interval) |")
                L.append("|---|---|")
                for key, v in hd["differences"].items():
                    L.append(f"| {key} | {_iv(v)} |")
                L.append("")
    return "\n".join(L)


def write_table(results: str = RESULTS) -> str:
    os.makedirs(results, exist_ok=True)
    path = os.path.join(results, "table.md")
    text = report(results)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--build", action="store_true",
                   help="build the drawn mutants and split every energy before measuring")
    p.add_argument("--workers", type=int, default=8,
                   help="how many processes the build runs side by side")
    p.add_argument("--build-shard", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--shards", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--stage", nargs="*", default=["sites", "energy"],
                   choices=["sites", "energy"], help="which stages to measure")
    p.add_argument("--repeats", type=int, default=M.REPEATS)
    p.add_argument("--draws", type=int, nargs="*", default=list(DRAWS))
    p.add_argument("--holdout", nargs="*", default=list(HOLDOUTS))
    p.add_argument("--report", action="store_true", help="only rewrite the table")
    args = p.parse_args(argv)
    if args.build_shard is not None:
        build_share(args.build_shard, args.shards)
        return
    if args.build:
        build(args.workers)
    if not args.report:
        run(tuple(args.stage), tuple(args.draws), args.repeats, tuple(args.holdout))
    print(write_table(RESULTS))


if __name__ == "__main__":
    main()
