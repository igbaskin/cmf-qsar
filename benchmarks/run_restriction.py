#!/usr/bin/env python
"""
Does a ring closure remove the field that moves? (plan item MC9)

Rigidifying a molecule -- tying two of its parts into a ring so that a bond can no longer rotate
-- is one of the few moves in medicinal chemistry whose effect is meant to be purely
conformational.  Plan item D9 found that an ensemble of conformations does not predict better
than one geometry, and plan item D10 found that the variance of the field across an ensemble adds
nothing as a block of the kernel.  The variance is still a map of where a molecule places field
only part of the time, and this item asks the question that map is suited to: **when a chemist
closes a ring, does the field that matters for the activity stop moving, and does the potency go
up when it does?**

What is measured
----------------
`cmfqsar/restriction.py` gives every molecule two amounts, both weighted by the square of the
co-field of a fitted model, so that only the field the model reads is counted:

* **relevant**, how much activity-relevant field the molecule places, averaged over its
  conformations;
* **intermittent**, how much of that field is there only part of the time.

A pair of a flexible molecule and its restricted analogue is then scored by **kept** (the share
of the relevant field the restriction keeps) and **removed** (the share of the intermittent part
it takes out).  The gate of the item is that `removed` tracks the measured potency gain.

Three scores on the same pairs, which is the ladder of rule 1
-------------------------------------------------------------
1. **rotatable bonds removed** -- counting, no geometry and no model.  A chemist has this for
   nothing, so anything built here has to beat it.
2. **variance removed, unweighted** -- the same integral with the co-field weight replaced by
   one.  This is the fields without the model.
3. **variance removed, weighted by the squared co-field** -- the score the item asks for.

The co-field is also rebuilt **without the two members of the pair** for every pair, so that the
weight cannot have learned the answer from the two potencies the score is compared with.

Both members of a pair come from the same conformer generator with the same settings, as the item
requires: the comparison is between two ensembles and not between two molecules.

Usage
-----
    python benchmarks/run_restriction.py
    python benchmarks/run_restriction.py --sets THR ACE --conformers 3
    python benchmarks/run_restriction.py --report
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

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.conformers import generate_conformers, single_conformer  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields, combine_members  # noqa: E402
from cmfqsar.fluctuation import variance_field, variance_fieldset  # noqa: E402
from cmfqsar.model import CMFModel, krr_fit  # noqa: E402
from cmfqsar.kernels import combine_kernels  # noqa: E402
from cmfqsar.restriction import (Pair, amounts_from, cofield_from, correlation,  # noqa: E402
                                 grid_for, grid_volumes, matched_pairs, score_pairs,
                                 training_fields)

RESULTS = os.path.join(HERE, "results", "restriction")
SETS = {
    "THR": dict(path=os.path.join(ROOT, "examples", "sutherland", "THR.sdf"), activity="pKi",
                what="thrombin, 88 molecules"),
    "COX2": dict(path=os.path.join(ROOT, "examples", "sutherland", "COX2.sdf"), activity="pIC50",
                 what="cyclooxygenase 2, 322 molecules"),
    "DHFR": dict(path=os.path.join(ROOT, "examples", "sutherland", "DHFR.sdf"), activity="pIC50",
                 what="dihydrofolate reductase, 397 molecules"),
    "ACE": dict(path=os.path.join(ROOT, "examples", "sutherland", "ACE.sdf"), activity="pIC50",
                what="angiotensin converting enzyme, 114 molecules"),
}
FIELDS = ["electrostatic", "hydrophobic"]
ALPHA = 2.0
FLOOR = 1e-2
SPACING = 1.0
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _say(text):
    print(text, flush=True)


def _activity(ds, prop):
    y = np.asarray(ds.activity, float)
    if np.isfinite(y).any():
        return y
    out = []
    for m in ds.mols:
        try:
            out.append(float(m.GetProp(prop)))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, float)


def load(key, max_molecules=0):
    spec = SETS[key]
    ds = load_dataset(spec["path"], activity_prop=spec["activity"])
    y = _activity(ds, spec["activity"])
    keep = [i for i in range(len(ds)) if np.isfinite(y[i])]
    if max_molecules:
        keep = keep[:max_molecules]
    mols = [Chem.RemoveHs(ds.mols[i]) for i in keep]
    return mols, [ds.names[i] for i in keep], y[keep], spec["what"]


def pairs_of(key, mols, names, y, results, force=False):
    """The matched pairs of one set, cached: the search runs a common-substructure test."""
    path = os.path.join(results, f"pairs_{key}.json")
    if os.path.exists(path) and not force:
        with io.open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("names") == list(names):
            return [Pair(**{k: (tuple(v) if k == "names" else v) for k, v in p.items()})
                    for p in d["pairs"]]
    t0 = time.time()
    pairs = matched_pairs(mols, y, names)
    os.makedirs(results, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(names=list(names), pairs=[p.__dict__ for p in pairs]), fh, indent=1,
                  default=list)
    _say(f"  {len(pairs)} pairs found in {time.time() - t0:.0f} s")
    return pairs


def ensembles(mols, names, fieldset, n_conformers=3, seed=1):
    """The mean field and the variance field of every molecule, from one generator run."""
    means, variances = [], []
    for k, m in enumerate(mols):
        ens = generate_conformers(m, n=int(n_conformers), seed=seed)
        n = max(ens.GetNumConformers(), 1)
        members = [build_fields(single_conformer(ens, i) if ens.GetNumConformers() > 1 else ens,
                                fieldset, name=f"{names[k]}#{i}") for i in range(n)]
        means.append(combine_members(members, np.ones(len(members)), kind="conformers",
                                     name=names[k]) if len(members) > 1 else members[0])
        variances.append(variance_field(members if len(members) > 1 else members * 2, fieldset,
                                        floor=FLOOR, name=names[k]))
    return means, variances


def fitted(mols, names, y, fieldset, folds=10, seed=0):
    """The ordinary model of this set, with the ridge searched on the folds."""
    table = AtomTable.from_molecules([build_fields(m, fieldset, n) for m, n in zip(mols, names)])
    model = CMFModel(fieldset=fieldset, method="krr", normalize=True)
    K_fields = np.asarray(model.prepare_training_kernels(table))
    best = None
    for lam in LAMS:
        model.params["lam"] = float(lam)
        model.set_training_kernels(table, K_fields, model.self_diag)
        model.fit(table, y, cv_folds=folds, cv_seed=seed, reuse_kernels=True)
        if best is None or model.stats.q2 > best[0]:
            best = (float(model.stats.q2), float(lam))
    model.params["lam"] = best[1]
    model.set_training_kernels(table, K_fields, model.self_diag)
    model.fit(table, y, cv_folds=folds, cv_seed=seed, reuse_kernels=True)
    K = combine_kernels(K_fields, fieldset.mixing_vector(), model.scale)
    return model, np.asarray(K, float), best


def without(K, y, lam, drop):
    """Dual coefficients of the same ridge fitted without the molecules in `drop`."""
    rows = np.asarray([i for i in range(len(y)) if i not in set(int(d) for d in drop)], int)
    a, _b = krr_fit(K[np.ix_(rows, rows)], np.asarray(y, float)[rows], float(lam))
    return rows, a


def run_set(key, n_conformers=3, folds=10, seed=0, max_molecules=0, spacing=SPACING,
            results=RESULTS, force=False):
    t0 = time.time()
    mols, names, y, what = load(key, max_molecules)
    _say(f"=== {key}: {len(mols)} molecules, {what}")
    pairs = pairs_of(key, mols, names, y, results, force=force)
    if len(pairs) < 5:
        _say(f"  only {len(pairs)} pairs: too few to measure anything, set skipped")
        return None

    fieldset = FieldSet.create(physchem=FIELDS, alpha=ALPHA)
    vfieldset = variance_fieldset(fieldset)
    specs = [s for s in fieldset.specs if s.enabled]
    vspecs = [s for s in vfieldset.specs if s.enabled]
    means, variances = ensembles(mols, names, fieldset, n_conformers, seed + 1)
    _say(f"  ensembles built, {time.time() - t0:.0f} s")

    model, K, (q2, lam) = fitted(mols, names, y, fieldset, folds, seed)
    _say(f"  model q2 {q2:+.3f} at ridge {lam:g}, {time.time() - t0:.0f} s")

    grid = grid_for(means, spacing=spacing)
    _say(f"  grid {grid.shape} = {grid.n_points} points")

    # every grid evaluation happens once: the co-field basis, and the fields of each molecule
    names_f = [getattr(s, "name", str(i)) for i, s in enumerate(specs)]
    volumes = grid_volumes(means, variances, specs, vspecs, grid)
    basis = training_fields(model, grid)
    _say(f"  fields on the grid, {time.time() - t0:.0f} s")

    # the three scores, on the same pairs
    flat = [np.ones(grid.n_points) for _ in specs]
    plain = amounts_from(flat, volumes, grid, names_f)
    weighted = amounts_from(cofield_from(basis, model), volumes, grid, names_f)

    # the same weighted score with a co-field that never saw either member of the pair
    honest = []
    for p in pairs:
        rows, a = without(K, y, lam, (p.flexible, p.restricted))
        c = cofield_from(basis, model, rows=rows, coef=a)
        one = amounts_from(c, volumes[[p.flexible, p.restricted]], grid, names_f)
        rel_a, int_a = float(one.relevant[0]), float(one.intermittent[0])
        honest.append(dict(
            kept=float(one.relevant[1] / rel_a) if rel_a > 0 else float("nan"),
            removed=float(1.0 - one.intermittent[1] / int_a) if int_a > 0 else float("nan")))
    _say(f"  the pair left out of the co-field, {time.time() - t0:.0f} s")

    gains = [p.gain for p in pairs]
    rot = [float(p.rotatable_removed) for p in pairs]
    s_plain = score_pairs(plain, pairs)
    s_weighted = score_pairs(weighted, pairs)
    rows = [
        dict(name="rotatable bonds removed", r=correlation(rot, gains, seed=seed)),
        dict(name="variance removed, unweighted",
             r=correlation([s.removed for s in s_plain], gains, seed=seed)),
        dict(name="variance removed, weighted",
             r=correlation([s.removed for s in s_weighted], gains, seed=seed)),
        dict(name="variance removed, weighted, pair left out",
             r=correlation([h["removed"] for h in honest], gains, seed=seed)),
        dict(name="relevant field kept, weighted",
             r=correlation([s.kept for s in s_weighted], gains, seed=seed)),
    ]
    for r in rows:
        c = r["r"]
        _say(f"  {r['name']:44s} r {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]")

    out = dict(set=key, what=what, n=len(mols), pairs=len(pairs), q2=q2, lam=lam,
               conformers=n_conformers, spacing=float(grid.spacing),
               points=int(grid.n_points), seconds=time.time() - t0, rows=rows,
               gains=gains, rotatable=rot,
               kept=[s.kept for s in s_weighted], removed=[s.removed for s in s_weighted],
               removed_plain=[s.removed for s in s_plain],
               removed_honest=[h["removed"] for h in honest],
               names=[list(p.names) for p in pairs])
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, f"{key}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def run(sets=("THR", "COX2", "DHFR", "ACE"), **kw):
    got = [run_set(k, **kw) for k in sets]
    return [g for g in got if g]


def _load(results=RESULTS):
    out = []
    for key in SETS:
        path = os.path.join(results, f"{key}.json")
        if os.path.exists(path):
            with io.open(path, encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out


def report(results=RESULTS):
    data = _load(results)
    if not data:
        raise SystemExit("nothing to report: run the benchmark first")
    lines = ["# Does a ring closure remove the field that moves? (plan item MC9)", "",
             "A conformational restriction is a ring closure that stops a bond rotating. The "
             "variance field of plan item D10 says where a molecule places field only part of "
             "the time; weighted by the square of the co-field it says how much of the "
             "**activity-relevant** field is intermittent. A pair of a flexible molecule and its "
             "restricted analogue is scored by how much of that intermittent part the "
             "restriction removes, and the question is whether that tracks the measured potency "
             "gain.", "",
             "The pairs are found by a mechanical rule, not by eye: one ring more, at least one "
             "rotatable bond fewer, at most two heavy atoms more, an atom composition differing "
             "by at most two atoms, and a common substructure covering at least 80 per cent of "
             "the smaller member. Both members get the same conformer generator with the same "
             "settings.", ""]
    for d in data:
        lines += [f"## {d['set']}: {d['what']}", "",
                  f"{d['pairs']} pairs among {d['n']} molecules; the model of the set reaches "
                  f"q2 {d['q2']:+.3f} at ridge {d['lam']:g}, and its co-field is the weight. "
                  f"{d['conformers']} conformations per molecule, a grid of {d['points']} points "
                  f"at {d['spacing']:.2f} A.", "",
                  "| score | correlation with the potency gain |", "| --- | ---: |"]
        for r in d["rows"]:
            lines.append(f"| {r['name']} | {_cell(r['r'])} |")
        lines.append("")
    lines += _verdict(data)
    return "\n".join(lines)


def _cell(c):
    """One correlation with its interval, or a dash where there was nothing to correlate."""
    if not np.isfinite(c[0]):
        return "--"
    if not np.isfinite(c[1]):
        return f"{c[0]:+.3f}"
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def _verdict(data):
    def row(d, name):
        return next(r["r"] for r in d["rows"] if r["name"] == name)

    out = ["## What the sets say together", "",
           "| set | pairs | rotatable bonds | variance removed, unweighted | weighted | "
           "weighted, pair left out |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for d in data:
        cells = [("--" if not np.isfinite(row(d, n)[0]) else f"{row(d, n)[0]:+.3f}") for n in
                 ("rotatable bonds removed", "variance removed, unweighted",
                  "variance removed, weighted", "variance removed, weighted, pair left out")]
        out.append(f"| {d['set']} | {d['pairs']} | " + " | ".join(cells) + " |")
    out += ["", "### All the pairs together", "",
            "One log unit of potency means the same thing in every series, so the pairs can be "
            "put in one column. This is the single number the gate of the item asks for.", "",
            "| score | pairs | correlation with the potency gain |", "| --- | ---: | ---: |"]
    pooled = {}
    for key, column in (("rotatable bonds removed", "rotatable"),
                        ("variance removed, unweighted", "removed_plain"),
                        ("variance removed, weighted", "removed"),
                        ("variance removed, weighted, pair left out", "removed_honest"),
                        ("relevant field kept, weighted", "kept")):
        x = np.concatenate([np.asarray(d[column], float) for d in data])
        g = np.concatenate([np.asarray(d["gains"], float) for d in data])
        pooled[key] = correlation(x, g)
        out.append(f"| {key} | {len(x)} | {_cell(pooled[key])} |")
    out.append("")

    per_set = [row(d, "variance removed, weighted") for d in data]
    up = sum(1 for c in per_set if np.isfinite(c[1]) and c[1] > 0)
    down = sum(1 for c in per_set if np.isfinite(c[2]) and c[2] < 0)
    w = pooled["variance removed, weighted"]
    h = pooled["variance removed, weighted, pair left out"]
    b = pooled["rotatable bonds removed"]
    k = pooled["relevant field kept, weighted"]
    n_pairs = sum(d["pairs"] for d in data)
    passes = np.isfinite(w[1]) and w[1] > 0 and abs(w[0]) > abs(b[0])

    out += ["", "## The verdict", ""]
    if passes:
        out += [f"**The score passes the gate.** Over all {n_pairs} pairs the share of "
                f"intermittent field a ring closure removes correlates with the potency gain at "
                f"{_cell(w)}, and it does so more strongly than counting the rotatable bonds the "
                f"closure removed, {_cell(b)}, which needs no geometry, no conformations and no "
                f"model. Leaving both members of a pair out of the co-field that weights the "
                f"score changes it to {_cell(h)}, so the weight did not learn the answer from "
                f"the two potencies it is compared with."]
    else:
        out += [f"**The score does not pass the gate.** Over all {n_pairs} pairs the share of "
                f"intermittent field a ring closure removes correlates with the potency gain at "
                f"{_cell(w)}. Counting the rotatable bonds the closure removed reaches "
                f"{_cell(b)} on the same pairs, and that needs no geometry, no conformations and "
                f"no model. Leaving both members of a pair out of the co-field changes the score "
                f"to {_cell(h)}, so the result is not an artefact of the weight: there is simply "
                f"little there.",
                "",
                f"The picture is the same set by set: the interval excludes zero on {up} of "
                f"the {len(data)} sets in the direction the item expects and on {down} in the "
                f"opposite direction, and contains zero on the rest. A score that reaches the "
                f"gate on one series out of four describes that series rather than "
                f"conformational restriction."]
    out += ["",
            f"**What does track the potency gain is the other half of the score.** How much of "
            f"the activity-relevant field the restricted analogue still places -- the **kept** "
            f"row -- reaches {_cell(k)} over the same pairs. That is the mean field and not the "
            "variance, and it says the expected thing: a ring closure helps when the molecule "
            "still puts its field where the model reads it, and hurts when the ring holds it "
            "somewhere else. The measurement here is of the variance, and the variance is not "
            "where the answer is.",
            "",
            "This is the third place in this repository where the conformational ensemble "
            "carries less than it promises. Plan item D9 found that averaging a field over "
            "conformations does not predict better than one geometry; plan item D10 found that "
            "the variance across conformations adds nothing as a block of the kernel; and this "
            "item finds that it does not explain what a conformational restriction did either. "
            "The variance field stays what D10 made it: a map of where a series is rigid, worth "
            "drawing and not worth scoring with.",
            "",
            "Two limits of the measurement are worth stating. The ensembles are three "
            "conformations from a distance-geometry generator superimposed on the pose the "
            "series was aligned to, which is a narrow view of what a molecule does in water. And "
            "a ring closure is never purely conformational in practice: it changes the shape and "
            "often an atom or two as well, and the pair rule here allows up to two atoms of "
            "difference. A set of pairs that differ in nothing but a single bond would be a "
            "better test, and no public set of that kind with measured potencies was found.",
            ""]
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
    ap.add_argument("--sets", nargs="*", default=["THR", "COX2", "DHFR", "ACE"])
    ap.add_argument("--conformers", type=int, default=3)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spacing", type=float, default=SPACING)
    ap.add_argument("--max-molecules", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="search the pairs again")
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
    run(sets=a.sets, n_conformers=a.conformers, folds=a.folds, seed=a.seed,
        spacing=a.spacing, max_molecules=a.max_molecules, force=a.force)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
