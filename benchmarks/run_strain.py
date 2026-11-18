#!/usr/bin/env python
"""
The strain the mutation leaves: does a force-field energy add to the description of a site?
(plan item PP4)

Every point mutant of plan item PP1 was built by placing the new side chain and minimising it,
with the side chains around it free, in the Amber ff14SB force field.  The energy of that
structure is a physical quantity that no field of the site sees.  PP3 found that the best
description of a site is its solvent-accessible area, ahead of a count of the atoms around it
and of every field description.  PP4 asks whether the energy adds to that, or to the field of
the surroundings, which is the fair way to ask whether a field model competes with a force
field or completes it.

The energies
------------
The wild type of PP1 was completed but never minimised, so it has no energy to compare with.
Stage 1 relaxes the wild type at every position exactly as a mutant at that position was
relaxed (:func:`cmfqsar.mutants.relax_wild_type`) and splits the energy of every structure into
the terms of the residue alone, of the other atoms alone, and of the interaction between them
(:func:`cmfqsar.mutants.energy_terms`).  For each mutant, against the wild type at the same
position, five columns follow (kJ/mol):

* **interaction, van der Waals** and **interaction, electrostatic** -- the change of the
  non-bonded energy between the residue and the rest of the protein;
* **interaction, bonded** -- the change of the bonds, angles and torsions that join the residue
  to its neighbours in the chain;
* **strain of the surroundings** -- the change of the energy of all other atoms, which are the
  same atoms in both structures;
* **strain of the residue** -- the energy of the new residue's own terms minus the median of
  that residue type over every structure, minus the same for the wild-type residue.  The
  median removes the part that only says which residue it is.

**Total energy difference** is the plain difference of the two total energies, as the plan
first described the item.  It changes with the atoms of the residue, so it is mostly a label of
the substitution; it is reported to show that.

The measurement
---------------
The samples are those of PP1's first stage: the positions with built mutants in the four
domains (Tsuboyama et al., Nature 2023), four substitutions per position on two draws, target
ddG in kcal/mol.  Every description is joined with the three substitution columns (the change of
volume, hydropathy and charge).  Each column kernel is a Gaussian on standardised columns.  The
energies have long tails, so each energy description is offered both as it is and on the scale
sign(x) ln(1 + |x| / 4.184), and the inner folds choose.  References: the Gaussian kernel on one
count of atoms around the site (PP1) and on the accessible area (PP3); the field of the
surroundings (PP1's context kernel at rho 0.1) joined with the columns.  Hold-outs: nine
consecutive residues (segment), whole domains and positions; ten folds (four for domains), three
repeats; every choice is made in inner folds.  The readings: the energy against each reference,
and each reference with the energy added against the reference alone.  The evidence of
:func:`cmfqsar.mixing.drop_one` for the energy as one more block next to the accessible area is
reported as well; it is computed on all samples of a draw, so it is not a held-out number.

Usage
-----
    python benchmarks/run_strain.py --compute --workers 8   # stage 1, about 15 minutes
    python benchmarks/run_strain.py                          # stage 2, a few minutes
    python benchmarks/run_strain.py --report
"""
from __future__ import annotations

import argparse
import glob
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

import run_mutants as M  # noqa: E402
from cmfqsar import mutants as MU  # noqa: E402

STRAIN = os.path.join(ROOT, "benchmarks", "cache", "megascale", "strain")
RESULTS = os.path.join(ROOT, "benchmarks", "results", "strain")
HOLDOUTS = ("segment", "domain", "position")
DRAWS = (0, 1)
#: the scale of the logarithmic form of an energy: 1 kcal/mol in kJ/mol
LOG_SCALE = 4.184
ENERGY_COLUMNS = ("interaction, van der Waals", "interaction, electrostatic",
                  "interaction, bonded", "strain of the surroundings", "strain of the residue")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Stage 1: relaxed wild types and the split energies
# --------------------------------------------------------------------------- #
def jobs(domains=M.DOMAINS) -> List[Tuple[str, str, str]]:
    """(kind, domain, label) for every structure stage 1 handles: the wild type at every
    position that has a built mutant, and every built mutant."""
    out = []
    for d in domains:
        labels = sorted(os.path.basename(p)[:-4]
                        for p in glob.glob(os.path.join(M.BUILT, d, "*.npz"))
                        if os.path.basename(p) != "wild_type.npz")
        positions = sorted({MU.Mutation.parse(lab).position for lab in labels})
        out += [("wt", d, str(p)) for p in positions]
        out += [("mut", d, lab) for lab in labels]
    return out


def _terms_path(kind: str, d: str, label: str) -> str:
    return os.path.join(STRAIN, d, f"{'wt' if kind == 'wt' else ''}{label}.json")


def compute_share(shard: int = 0, shards: int = 1, domains=M.DOMAINS) -> None:
    """Every `shards`-th job from `shard` on; a job already done is skipped."""
    os.environ.setdefault("OPENMM_CPU_THREADS", "1")
    mine = jobs(domains)[shard::shards]
    todo = [j for j in mine if not os.path.exists(_terms_path(*j))]
    log(f"share {shard}: {len(mine)} jobs, {len(todo)} to do")
    t0, failed = time.time(), []
    for k, (kind, d, label) in enumerate(todo, 1):
        os.makedirs(os.path.join(STRAIN, d), exist_ok=True)
        try:
            if kind == "wt":
                position = int(label)
                st = MU.relax_wild_type(os.path.join(M.BUILT, d, "wild_type.pdb"), position)
                st.save(os.path.join(STRAIN, d, f"wt{position}.npz"))
            else:
                position = MU.Mutation.parse(label).position
                st = MU.Structure.load(os.path.join(M.BUILT, d, label + ".npz"))
            terms = MU.energy_terms(st, position)
            terms.update(name=st.name, position=position, energy_before=st.energy_before,
                         energy_after=st.energy_after)
            with open(_terms_path(kind, d, label), "w", encoding="utf-8", newline="\n") as fh:
                json.dump(terms, fh, indent=1)
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            failed.append(f"{kind} {d} {label}: {type(exc).__name__}: {exc}")
        if k % 100 == 0 or k == len(todo):
            log(f"  share {shard}: {k}/{len(todo)} after {time.time() - t0:.0f} s, "
                f"{len(failed)} failed")
    if failed:
        with open(os.path.join(STRAIN, f"failed_{shard}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(failed) + "\n")


def compute(workers: int = 8) -> None:
    """Stage 1 in `workers` separate processes (a pool on Windows loses its children when the
    parent is interrupted; separate processes keep what each has written)."""
    env = dict(os.environ, OPENMM_CPU_THREADS="1",
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    procs = [subprocess.Popen([sys.executable, "-P", os.path.abspath(__file__),
                               "--compute-shard", str(k), "--shards", str(workers)], env=env)
             for k in range(workers)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"stage 1: exit codes {codes}")


def read_terms(domains=M.DOMAINS) -> Tuple[Dict[Tuple[str, int], dict], Dict[Tuple[str, str], dict]]:
    """The stored terms: wild types by (domain, position), mutants by (domain, label)."""
    wt, mut = {}, {}
    for d in domains:
        for p in glob.glob(os.path.join(STRAIN, d, "*.json")):
            with open(p, encoding="utf-8") as fh:
                t = json.load(fh)
            base = os.path.basename(p)[:-5]
            if base.startswith("wt"):
                wt[(d, int(base[2:]))] = t
            else:
                mut[(d, base)] = t
    return wt, mut


def _part(t: dict, part: str, kinds=MU.ENERGY_KINDS) -> float:
    return float(sum(t[part][k] for k in kinds))


def residue_references(wt: dict, mut: dict) -> Dict[str, float]:
    """The median energy of the residue's own terms for each residue type over every
    structure, wild types and mutants alike."""
    by: Dict[str, List[float]] = {}
    for t in list(wt.values()) + list(mut.values()):
        by.setdefault(t["residue"], []).append(_part(t, "site"))
    return {r: float(np.median(v)) for r, v in by.items()}


def energy_rows(s, wt: dict, mut: dict, ref: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """The five energy columns and the total energy difference of every sample (kJ/mol)."""
    rows, total = [], []
    bonded = ("bond", "angle", "torsion")
    for d, p, lab in zip(s.domain, s.position, s.label):
        w, m = wt[(d, int(p))], mut[(d, lab)]
        rows.append([
            m["interaction"]["van der Waals"] - w["interaction"]["van der Waals"],
            m["interaction"]["electrostatic"] - w["interaction"]["electrostatic"],
            _part(m, "interaction", bonded) - _part(w, "interaction", bonded),
            _part(m, "surroundings") - _part(w, "surroundings"),
            (_part(m, "site") - ref[m["residue"]]) - (_part(w, "site") - ref[w["residue"]]),
        ])
        total.append(m["total"] - w["total"])
    return np.array(rows, float), np.array(total, float)[:, None]


def signed_log(X: np.ndarray, scale: float = LOG_SCALE) -> np.ndarray:
    """sign(x) ln(1 + |x| / scale): linear near zero, logarithmic in the tails."""
    X = np.asarray(X, float)
    return np.sign(X) * np.log1p(np.abs(X) / float(scale))


# --------------------------------------------------------------------------- #
# Stage 2: the measurement
# --------------------------------------------------------------------------- #
def subset(s, keep: np.ndarray):
    """The samples `keep` of a :class:`run_mutants.Samples`, in their order."""
    keep = np.asarray(keep, bool)
    for name in ("domain", "label", "position", "mutation", "new", "old"):
        setattr(s, name, [v for v, k in zip(getattr(s, name), keep) if k])
    s.y = np.asarray(s.y)[keep]
    return s


def _gauss_pair(X: np.ndarray, X_three: np.ndarray, name: str) -> Dict[str, np.ndarray]:
    """An energy description joined with the three columns, as it is and on the log scale."""
    return {f"{name}, as it is": M.gaussian_columns(np.c_[X, X_three]),
            f"{name}, log scale": M.gaussian_columns(np.c_[signed_log(X), X_three])}


COMPARISONS = (("energy terms", "one count"), ("energy terms", "accessible area"),
               ("energy terms", "total energy difference"),
               ("accessible area and energy terms", "accessible area"),
               ("one count and energy terms", "one count"),
               ("one count and energy terms", "accessible area"),
               ("field and energy terms", "field of the surroundings"),
               ("accessible area", "one count"),
               ("field of the surroundings", "one count"))


def measure(s, X_energy, X_total, X_area, seeds, holdouts) -> dict:
    X_three = MU.substitution_features(s.mutation)[0][:, :3]
    Kc3 = M.gaussian_columns(X_three)
    count = M.burial_count_rows(s)[:, :1]
    K_field = M.unique_context_kernel(s, M.SITE_RHO, M.CONTEXT_ALPHA)
    field_family = {f"field, weight {w}": M.mixed(Kc3, K_field, w) for w in M.MIXES}
    energy = _gauss_pair(X_energy, X_three, "energy")
    field_energy = {f"field, weight {w}, {k}": M.mixed(K, K_field, w)
                    for w in M.MIXES for k, K in energy.items()}
    kernels = {
        "three columns kernel": {"columns": Kc3},
        "one count": {"count": M.gaussian_columns(np.c_[count, X_three])},
        "accessible area": {"area": M.gaussian_columns(np.c_[X_area, X_three])},
        "field of the surroundings": field_family,
        "total energy difference": _gauss_pair(X_total, X_three, "total"),
        "energy terms": energy,
        "one count and energy terms": {
            "as it is": M.gaussian_columns(np.c_[count, X_energy, X_three]),
            "log scale": M.gaussian_columns(np.c_[count, signed_log(X_energy), X_three])},
        "accessible area and energy terms": {
            "as it is": M.gaussian_columns(np.c_[X_area, X_energy, X_three]),
            "log scale": M.gaussian_columns(np.c_[X_area, signed_log(X_energy), X_three])},
        "field and energy terms": field_energy,
    }
    out = {}
    for by in holdouts:
        tg = s.targets(by)
        n_groups = int(tg.molecule.max()) + 1
        folds = min(M.FOLDS, n_groups) if by != "domain" else min(len(set(s.domain)), 4)
        rows, keep = {}, {}
        res, chosen = M.mean_reference(tg, seeds, folds)
        rows["mean"], keep["mean"] = M.row_of(res, chosen, tg), res
        for name, fam in kernels.items():
            res, chosen = M.nested(fam, tg, seeds, name, folds)
            rows[name], keep[name] = M.row_of(res, chosen, tg), res
            log(f"    {by:8s} {name:34s} q2 {rows[name]['q2']:+.3f}")
        diffs = {f"{a} - {b}": M.compare(keep[a], keep[b], tg) for a, b in COMPARISONS}
        out[by] = {"rows": rows, "differences": diffs, "n_groups": n_groups, "folds": folds}
    return out


def evidence(s, X_energy, X_area) -> Dict[str, float]:
    """The log Bayes factor against 'this block does nothing' (:func:`cmfqsar.mixing.drop_one`)
    for the accessible area and for the energy terms, each a block of one sum."""
    from cmfqsar.mixing import drop_one, with_views
    X_three = MU.substitution_features(s.mutation)[0][:, :3]
    K_area = M.gaussian_columns(np.c_[X_area, X_three])
    K_energy = M.gaussian_columns(np.c_[signed_log(X_energy), X_three])
    stack, names, scale = with_views(K_area[None], {"energy terms": K_energy},
                                     names=["accessible area"])
    y = np.asarray(s.y, float)
    return {k: round(v, 2) for k, v in drop_one(stack, y, names, scale).items()}


def correlations(X_energy, X_total, y) -> Dict[str, Dict[str, float]]:
    from scipy.stats import spearmanr
    cols = dict(zip(ENERGY_COLUMNS, X_energy.T))
    cols["total energy difference"] = X_total[:, 0]
    return {k: {"pearson": round(float(np.corrcoef(v, y)[0, 1]), 3),
                "spearman": round(float(spearmanr(v, y)[0]), 3),
                "median": round(float(np.median(v)), 2),
                "spread": [round(float(q), 1) for q in np.percentile(v, [5, 95])]}
            for k, v in cols.items()}


def run(draws=DRAWS, repeats: int = M.REPEATS, holdouts=HOLDOUTS, results: str = RESULTS) -> dict:
    import run_site_fields as SF
    os.makedirs(results, exist_ok=True)
    t0 = time.time()
    wt, mut = read_terms()
    if not wt or not mut:
        raise SystemExit("no stored energies: run with --compute first")
    ref = residue_references(wt, mut)
    area = SF.accessible_area()
    log(f"{len(wt)} wild-type sites, {len(mut)} mutants, area after {time.time() - t0:.0f} s")
    out = {"wild_type_sites": len(wt), "mutants": len(mut), "repeats": repeats,
           "log_scale": LOG_SCALE, "residue_references": {k: round(v, 2) for k, v in ref.items()},
           "relaxation_of_the_wild_type": _relaxation(wt), "draws": {}}
    seeds = list(range(repeats))
    for draw in draws:
        s = M.Samples(M.DOMAINS, M.MAX_PER_POSITION, draw)
        have = np.array([(d, int(p)) in wt and (d, lab) in mut
                         for d, p, lab in zip(s.domain, s.position, s.label)])
        missing = int((~have).sum())
        s = subset(s, have)
        X_energy, X_total = energy_rows(s, wt, mut, ref)
        X_area = np.array([area[(d, int(p))] for d, p in zip(s.domain, s.position)], float)
        log(f"--- draw {draw}: {len(s)} substitutions ({missing} without energies), "
            f"ddG sd {s.y.std():.2f} kcal/mol")
        out["draws"][str(draw)] = {
            "n": len(s), "missing": missing, "positions": len(set(zip(s.domain, s.position))),
            "sd_y": round(float(s.y.std()), 4),
            "correlation_with_ddG": correlations(X_energy, X_total, s.y),
            "evidence": evidence(s, X_energy, X_area),
            "holdout": measure(s, X_energy, X_total, X_area, seeds, holdouts)}
        _write(out, results)
    out["seconds"] = round(time.time() - t0, 1)
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write(out, results)
    log(f"done after {out['seconds']:.0f} s")
    return out


def _relaxation(wt: dict) -> Dict[str, float]:
    """How much the relaxation lowered the wild-type energy at a site (kJ/mol), which is what a
    comparison with the unrelaxed wild type would have mistaken for the substitution."""
    drop = np.array([t["energy_before"] - t["energy_after"] for t in wt.values()], float)
    return {"sites": int(len(drop)), "median": round(float(np.median(drop)), 1),
            "5th percentile": round(float(np.percentile(drop, 5)), 1),
            "95th percentile": round(float(np.percentile(drop, 95)), 1)}


def _write(out: dict, results: str) -> None:
    with open(os.path.join(results, "strain.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
ROWS = ("mean", "three columns kernel", "one count", "accessible area",
        "field of the surroundings", "total energy difference", "energy terms",
        "one count and energy terms", "accessible area and energy terms",
        "field and energy terms")


def _iv(d: Sequence[float]) -> str:
    txt = f"{d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f})"
    return f"**{txt}**" if d[1] > 0 or d[2] < 0 else txt


def report(results: str = RESULTS) -> str:
    path = os.path.join(results, "strain.json")
    if not os.path.exists(path):
        return "# PP4: no result yet\n"
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    L = ["# The strain the mutation leaves (plan item PP4)\n",
         f"`python benchmarks/run_strain.py`, written {d.get('written', '?')}, "
         f"{d.get('seconds', 0) / 60:.0f} minutes, after stage 1 "
         "(`--compute`) had relaxed the wild type at every position and split every energy.\n",
         f"{d['wild_type_sites']} wild-type sites and {d['mutants']} mutants of four domains of "
         "the mega-scale set, Amber ff14SB in vacuum. Each mutant is compared with the wild "
         "type relaxed in the same way at the same position. Every description is joined with "
         "the change of volume, hydropathy and charge. q2 in cross-validation, the mean of "
         f"{d['repeats']} repeats; every choice is made in inner folds of the same kind.\n"]
    r = d.get("relaxation_of_the_wild_type", {})
    if r.get("sites"):
        L.append(f"Relaxing the wild type at a site lowered its energy by {r['median']:.0f} "
                 f"kJ/mol (median; {r['5th percentile']:.0f} to {r['95th percentile']:.0f} for "
                 "the 5th and 95th percentiles). A mutant compared with the unrelaxed wild type "
                 "would have counted that drop as part of the substitution.\n")
    for draw, part in d["draws"].items():
        L.append(f"## Draw {draw}: {part['n']} substitutions at {part['positions']} positions, "
                 f"sd of ddG {part['sd_y']:.2f} kcal/mol\n")
        L.append("Each energy column against ddG over the substitutions (kJ/mol; the spread is "
                 "the 5th and 95th percentiles):\n")
        L.append("| Energy column | Pearson r with ddG | Spearman rho with ddG | Median | "
                 "Spread |")
        L.append("|---|---:|---:|---:|---|")
        for k, v in part["correlation_with_ddG"].items():
            L.append(f"| {k} | {v['pearson']:+.3f} | {v['spearman']:+.3f} | {v['median']:+.1f} "
                     f"| {v['spread'][0]:+.1f} to {v['spread'][1]:+.1f} |")
        L.append("")
        ev = part["evidence"]
        L.append("Log Bayes factor against \"this block does nothing\", for the two blocks of "
                 "one sum, on all samples of the draw (not a held-out number): "
                 + ", ".join(f"{k} {v:+.2f}" for k, v in ev.items()) + ".\n")
        hs = list(part["holdout"])
        L.append("| Description | " + " | ".join(f"q2, {h}s held out" for h in hs) + " |")
        L.append("|---|" + "---:|" * len(hs))
        for name in ROWS:
            if all(name in part["holdout"][h]["rows"] for h in hs):
                L.append(f"| {name} | " + " | ".join(
                    f"{part['holdout'][h]['rows'][name]['q2']:+.3f}" for h in hs) + " |")
        L.append("")
        L.append("Differences in q2, with the 95 % interval over the repeats and the folds; "
                 "bold where it excludes zero.\n")
        L.append("| Difference | " + " | ".join(f"{h}s held out" for h in hs) + " |")
        L.append("|---|" + "---|" * len(hs))
        for key in part["holdout"][hs[0]]["differences"]:
            L.append(f"| {key} | " + " | ".join(
                _iv(part["holdout"][h]["differences"][key]) for h in hs) + " |")
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
    p.add_argument("--compute", action="store_true",
                   help="stage 1: relax the wild types and split every energy")
    p.add_argument("--workers", type=int, default=8,
                   help="how many processes stage 1 runs side by side")
    p.add_argument("--compute-shard", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--shards", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--repeats", type=int, default=M.REPEATS)
    p.add_argument("--draws", type=int, nargs="*", default=list(DRAWS))
    p.add_argument("--holdout", nargs="*", default=list(HOLDOUTS))
    p.add_argument("--report", action="store_true", help="only rewrite the table")
    args = p.parse_args(argv)
    if args.compute_shard is not None:
        compute_share(args.compute_shard, args.shards)
        return
    if args.compute:
        compute(args.workers)
    if not args.report:
        run(args.draws, args.repeats, tuple(args.holdout))
    print(write_table(RESULTS))


if __name__ == "__main__":
    main()
