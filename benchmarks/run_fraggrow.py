#!/usr/bin/env python
"""
Can a field score say what to grow a fragment into? (plan item MC7)

A fragment that binds fills part of a site and leaves the rest empty.  The question a fragment
campaign asks next is what to put there.  `cmfqsar/fraggrow.py` answers it with the fields alone:
the pocket's offer, with the sign convention of `cmfqsar/complementarity.py`, minus what the
fragment already places, is the **gap**; a candidate group is built from its SMILES, placed on
the growth vector by rule with the torsion scanned, and scored by how much of the gap its own
field fills, where the pocket leaves room for it.

The gate of plan item MC7 is whether the group that was actually added comes out among the first
ten of such an enumeration.  The plan calls this the item most likely to return nothing, because
a one-body field score has never been shown to rank linkers.

How it is measured
------------------
* **The cases** come from the fragment sets of the Schrodinger free-energy collection
  (`examples/ross/fragments/`), which are the only sets here with a pocket and with small and
  large ligands in one crystal frame.  A case is a pair of ligands of one set where the smaller
  is a substructure of the larger and the difference hangs on a single bond.
* **Nothing about the answer enters the geometry.** The growth vector is the direction the
  fragment's own bonds leave empty; every candidate, the true one included, is built from its
  SMILES with a distance-geometry embedding and a force field, never from a crystal structure,
  and is placed by the same rule with the same turn scan.
* **Three rankings on the same candidates**, which is the ladder of rule 1:
  1. **at random**, which is what the enumeration is worth with no method at all;
  2. **by size**, the heavy atoms of the group, with clashing placements last -- geometry and
     nothing else;
  3. **by the field score**, with clashing placements last.
  The second is the one that matters: a bigger group fills more of any gap, so a field score
  that only reproduces the size ordering has added nothing.

Usage
-----
    python benchmarks/run_fraggrow.py
    python benchmarks/run_fraggrow.py --library 120 --report
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.bioisostere import substituents  # noqa: E402
from cmfqsar.complementarity import pocket_fields  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.fraggrow import (addition, build_group, gap, growth_vector,  # noqa: E402
                              rank_of, try_group)
from cmfqsar.protein import read_structure  # noqa: E402

RESULTS = os.path.join(HERE, "results", "fraggrow")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 1.0
LIBRARY = 200
TURNS = 12


def _say(text):
    print(text, flush=True)


def _load(path):
    return [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]


def sets_with_pockets():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "examples", "ross", "fragments", "*.sdf"))):
        name = os.path.basename(path)[:-4]
        pocket = os.path.join(ROOT, "examples", "pockets", "ross", "fragments", name + ".pdb")
        if os.path.exists(pocket):
            out.append((name, path, pocket))
    return out


def cases_of(name, path):
    """Every pair of one set where the smaller ligand is the larger one with a group taken off."""
    mols = _load(path)
    out = []
    for i, f in enumerate(mols):
        fh = Chem.RemoveHs(f)
        heavy_of = [a.GetIdx() for a in f.GetAtoms() if a.GetAtomicNum() > 1]
        for j, g in enumerate(mols):
            if i == j:
                continue
            got = addition(fh, Chem.RemoveHs(g))
            if not got:
                continue
            smiles, at_heavy, added = got
            out.append(dict(set=name, fragment=i, grown=j, smiles=smiles,
                            atom=int(heavy_of[at_heavy]), added=len(added),
                            names=[f.GetProp("_Name") if f.HasProp("_Name") else str(i),
                                   g.GetProp("_Name") if g.HasProp("_Name") else str(j)]))
    return mols, out


def library(cases, size=LIBRARY, seed=1):
    """The groups to enumerate: common substituents of the free-energy ligands, and the answers.

    The answers have to be in the library or there would be nothing to find; they are built and
    placed exactly like every other candidate, so being in it is no advantage.
    """
    seen = Counter()
    for path in sorted(glob.glob(os.path.join(ROOT, "examples", "ross", "jacs_set", "*.sdf"))):
        mols = [Chem.RemoveHs(m) for m in _load(path)]
        for s in substituents(mols, max_substituent=12, min_core=0.5):
            seen[s.group] += 1
    want = [g for g, _n in seen.most_common(int(size))]
    for c in cases:
        if c["smiles"] not in want:
            want.append(c["smiles"])
    built, dropped = {}, 0
    for g in want:
        grp = build_group(g, seed=seed)
        if grp is None:
            dropped += 1
            continue
        built[g] = grp
    return built, dropped


def run(size=LIBRARY, turns=TURNS, alpha=ALPHA, seed=1, results=RESULTS, max_cases=0):
    t0 = time.time()
    fieldset = FieldSet.create(physchem=FIELDS, alpha=float(alpha))
    all_cases, mols_of = [], {}
    for name, path, _pocket in sets_with_pockets():
        mols, cases = cases_of(name, path)
        mols_of[name] = mols
        all_cases.extend(cases)
        _say(f"  {name:26s} {len(mols):3d} ligands, {len(cases):3d} growing cases")
    if max_cases:
        all_cases = all_cases[:max_cases]
    _say(f"{len(all_cases)} cases in {len(mols_of)} sets, {time.time() - t0:.0f} s")

    groups, dropped = library(all_cases, size=size, seed=seed)
    _say(f"a library of {len(groups)} groups ({dropped} could not be built), "
         f"{time.time() - t0:.0f} s")

    pockets = {}
    for name, _path, pocket in sets_with_pockets():
        pockets[name] = pocket_fields(read_structure(pocket), fieldset)

    rows = []
    for k, c in enumerate(all_cases):
        mols = mols_of[c["set"]]
        f = mols[c["fragment"]]
        v = growth_vector(f, c["atom"])
        if v is None:
            continue
        pf = pockets[c["set"]]
        g = gap(pf, build_fields(f, fieldset, name="fragment"), fieldset, v)
        if not len(g):
            continue
        found = []
        for smiles, grp in groups.items():
            cand = try_group(f, v, grp, pf, fieldset, g, turns=turns)
            if cand is not None:
                found.append(cand)
        if not found:
            continue
        by_field = sorted(found, key=lambda x: (x.clashes, -x.score))
        by_size = sorted(found, key=lambda x: (x.clashes, -x.heavy))
        rng = np.random.RandomState(seed + k)
        shuffled = [found[i] for i in rng.permutation(len(found))]
        by_chance = shuffled
        by_fit = sorted(shuffled, key=lambda x: x.clashes)      # stable: random inside a group
        rows.append(dict(
            set=c["set"], names=c["names"], smiles=c["smiles"], added=c["added"],
            candidates=len(found),
            free=int(sum(1 for x in found if x.clashes == 0)),
            points=len(g),
            rank_field=rank_of(by_field, c["smiles"]),
            rank_size=rank_of(by_size, c["smiles"]),
            rank_chance=rank_of(by_chance, c["smiles"]),
            rank_fit=rank_of(by_fit, c["smiles"]),
            clashes=int(next(x.clashes for x in found if x.smiles == c["smiles"])),
            score=float(next(x.score for x in found if x.smiles == c["smiles"]))))
        _say(f"  {c['names'][0]} -> {c['names'][1]:14s} adds {c['smiles']:22s} "
             f"rank {rows[-1]['rank_field']:4d} of {len(found)} "
             f"(size {rows[-1]['rank_size']}, chance {rows[-1]['rank_chance']}), "
             f"{time.time() - t0:.0f} s")

    out = dict(cases=len(rows), library=len(groups), turns=turns, alpha=alpha,
               fields=FIELDS, seconds=time.time() - t0, rows=rows,
               summary={k: _summary(rows, k) for k in ("field", "size", "chance", "fit")})
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def _summary(rows, which):
    r = np.asarray([x[f"rank_{which}"] for x in rows if x[f"rank_{which}"]], float)
    n = np.asarray([x["candidates"] for x in rows if x[f"rank_{which}"]], float)
    if not len(r):
        return dict(median=float("nan"), top10=float("nan"), top1=float("nan"), n=0)
    return dict(median=float(np.median(r)), top10=float(np.mean(r <= 10)),
                top1=float(np.mean(r <= 1)), mean_share=float(np.mean(r / n)), n=int(len(r)))


def report(results=RESULTS):
    path = os.path.join(results, "summary.json")
    if not os.path.exists(path):
        raise SystemExit("nothing to report: run the benchmark first")
    with io.open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    out = ["# Can a field score say what to grow a fragment into? (plan item MC7)", "",
           "A bound fragment leaves a gap: what the pocket would still reward, after what the "
           "fragment already places is taken off. A candidate group is built from its SMILES, "
           "placed on the growth vector by rule with the turn about the new bond scanned, and "
           "scored by how much of the gap its own field fills where the pocket leaves room. The "
           "gate of the item is whether the group that was actually added comes out among the "
           "first ten.", "",
           f"{d['cases']} growing cases from the fragment sets of the Schrodinger free-energy "
           f"collection, a library of {d['library']} groups, {d['turns']} turns of the new bond "
           f"tried for each. Nothing about the answer enters the geometry: the growth vector is "
           f"the direction the fragment's own bonds leave empty, and every candidate, the true "
           f"one included, is built from its SMILES and placed the same way.", "",
           "| ranking | median rank of the true group | among the first ten | first |",
           "| --- | ---: | ---: | ---: |"]
    for key, label in (("chance", "at random"),
                       ("fit", "by whether it fits, at random inside that"),
                       ("size", "by whether it fits, then by the size of the group"),
                       ("field", "by whether it fits, then by the field score")):
        s = d["summary"][key]
        out.append(f"| {label} | {s['median']:.0f} | {s['top10']:.0%} | {s['top1']:.0%} |")
    out += ["", "| set | fragment | grown | added | rank by field | by size | candidates | "
                "without a clash |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for r in d["rows"]:
        out.append(f"| {r['set']} | {r['names'][0]} | {r['names'][1]} | `{r['smiles']}` | "
                   f"{r['rank_field']} | {r['rank_size']} | {r['candidates']} | {r['free']} |")
    out += [""] + _verdict(d)
    return "\n".join(out)


def _verdict(d):
    field, size, chance = (d["summary"][k] for k in ("field", "size", "chance"))
    fit = d["summary"].get("fit", chance)
    passes = field["top10"] >= 0.5
    better = field["median"] < size["median"]
    out = ["## The verdict", ""]
    if passes:
        out += [f"**The gate is met on most of the cases.** The group that was really added "
                f"comes out among the first ten in {field['top10']:.0%} of them, at a median "
                f"rank of {field['median']:.0f} out of a library of {d['library']}."]
    else:
        out += [f"**The gate is not met.** The group that was really added comes out among the "
                f"first ten in {field['top10']:.0%} of the cases, at a median rank of "
                f"{field['median']:.0f} in a library of {d['library']}; ranking the same "
                f"candidates at random puts it {chance['median']:.0f}, and refusing the "
                f"placements that clash while shuffling the rest puts it "
                f"{fit['median']:.0f}. What the plan expected of this item is what happened, "
                f"and rule 5 applies: the gap field stays a picture of where there is room and "
                f"what the pocket would reward there, which is worth drawing, and not a way of "
                f"choosing a group."]
    out += ["",
            f"**All of what the ranking knows is whether the group fits.** Refusing the "
            f"placements that clash and shuffling the rest puts the true group at a median rank "
            f"of {fit['median']:.0f}, against {chance['median']:.0f} with no method at all. "
            + (f"Sorting those candidates by the field score moves it to "
               f"{field['median']:.0f}, which is **worse**, and the share among the first ten "
               f"falls from {fit['top10']:.0%} to {field['top10']:.0%}. The score is not "
               f"neutral: it prefers the largest group that still fits, and the additions in "
               f"these cases are small, so ordering by it pushes the right answer down."
               if field["median"] > fit["median"] else
               f"Sorting those candidates by the field score moves it to "
               f"{field['median']:.0f}, and the share among the first ten goes from "
               f"{fit['top10']:.0%} to {field['top10']:.0%}."),
            "",
            f"**Against the other reference**, the size of the group, the field score is "
            + ("ahead" if better else "not ahead")
            + f": a median rank of {field['median']:.0f} against {size['median']:.0f}. A bigger "
              "group fills more of any gap, so this comparison says whether the fields added "
              "anything to counting atoms. The additions in these cases are small -- a fluorine, "
              "a methyl, a chlorine -- so a ranking that prefers large groups is working against "
              "itself here.",
            "",
            "**What the measurement cannot settle.** The library holds "
            f"{d['library']} groups and the true one is always in it, which is the easy part of "
            "a real enumeration; the cases are few, because a public set with a pocket, a "
            "fragment and the compound it grew into is rare; and the placement rule gives every "
            "candidate one embedded conformation and a turn about the new bond, where a real "
            "growing tool would minimise the group in the site. A negative result here is "
            "evidence against the one-body field score, not against fragment growing.", ""]
    return out


def write_table(results=RESULTS) -> str:
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=int, default=LIBRARY)
    ap.add_argument("--turns", type=int, default=TURNS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-cases", type=int, default=0)
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
    run(size=a.library, turns=a.turns, alpha=a.alpha, seed=a.seed, max_cases=a.max_cases)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
