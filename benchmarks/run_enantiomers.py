#!/usr/bin/env python
"""
Can the program tell two enantiomers apart?  Plan item MC3, measured on a real series.

Item C7 built the handedness channels of `cmfqsar/stereo.py` and could only test them on a
synthetic set, because none of the five sets of this repository holds an enantiomeric pair.
`examples/make_enantiomer_examples.py` builds the missing data from ChEMBL: pairs of mirror
images, each with its own potency, both potencies measured in the same assay.

Why the null of this measurement is exact
-----------------------------------------
The field autocorrelation is a two-point description: it counts pairs of atoms at a distance,
and a mirror image has every such pair in the same amounts.  The two members of a pair are built
as exact reflections of one conformation, so their autocorrelation rows are identical to the last
bit -- the script checks this and reports the largest difference it finds.  A model built on that
description alone therefore *has* to give the two members of a pair the same number.  It cannot
rank them, and its share of correctly ranked pairs is one half by construction, not by chance.
Any ranking above one half comes from the handedness channels and from nothing else.

The two questions
-----------------
1. *The difference inside a pair.*  Every potency has the mean of its own pair taken off it, so
   what is left is half the difference between the two enantiomers, with a plus sign on one
   member and a minus sign on the other.  The pairs of a fold are held out whole, so a held-out
   pair has neither of its members in the training part.  Reported: q2 on that half-difference,
   the share of held-out pairs whose better enantiomer is named correctly, and the same share
   among the pairs whose two potencies differ by at least 0.5 and at least 1.0 log units.
2. *The potency itself.*  For every target with enough pairs, the ordinary question: does adding
   the handedness channels to the two-point kernel improve the prediction of the potency, with
   whole pairs held out?  This is the five-set comparison of item C7 repeated on a set that has
   enantiomeric pairs in it.

Where the handedness is read
----------------------------
The channels are a sum over the centres of a molecule, so they can be built from a chosen part of
them (`cmfqsar.stereo.centre_channels`).  Three variants are compared: every centre, only the
atoms that RDKit calls stereocentres of the structure, and every centre except those.  If the
description works the way it claims to, the stereocentres alone should be enough and the rest
should carry little.

A cheap reference
-----------------
The channels are also compared with a rule that learns one bit for each target: whether the R or
the S member of a pair was the better one more often among the training pairs, read at one
stereocentre (`configuration_reference`).  It needs no field and no kernel.  It was added on
14 November 2026, because a share above one half says that something is read, and not that the
channels read more than a configuration label.

Usage
-----
    python benchmarks/run_enantiomers.py
    python benchmarks/run_enantiomers.py --reference     # only the rule, from the saved result
    python benchmarks/run_enantiomers.py --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.chemotypes import chemotype_labels  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (autocorrelation_kernel, block_sizes,  # noqa: E402
                               channel_scale, feature_matrix)
from cmfqsar.model import krr_fit, r2_score  # noqa: E402
from cmfqsar.stereo import StereoSpec, centre_channels  # noqa: E402

#: the same kernels, the same distance grids and the same regularisations as item C7's script,
#: imported rather than written again so that the two measurements are on identical terms
import run_chirality as C  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "enantiomers")
RESULTS = os.path.join(HERE, "results", "enantiomers")
FIELDS = C.FIELDS
#: Item C7's script offers three regularisations, 0.01 to 1.  On this set the choice fell on the
#: largest of the three in most folds, which is the edge of that grid and would leave the result
#: open to the objection that the model was never allowed enough regularisation.  Two more
#: values are offered above and one below.  The chosen value does not move -- 1 is a real
#: optimum here and not a constraint -- and the numbers are the same, measured; the wider grid
#: is kept so that the record shows the choice is not at an edge.
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
FOLDS = 10
INNER_FOLDS = 5
SEED = 0
#: a target needs this many pairs before the potency itself is modelled for it alone
MIN_PAIRS_PER_TARGET = 15
#: the differences in potency at which the ranking is also reported, in log units
MARGINS = (0.0, 0.5, 1.0)
#: which centres the handedness channels are built from
PARTS = ("every centre", "stereocentres only", "every other centre")
#: For the potency itself, the comparison this item is about: the two-point kernel against the
#: same kernel with the handedness channels of item C7 beside it.  C5's signed three-point
#: variants are not repeated here; they are measured in `run_chirality.py`, which is their own
#: item, and running them on every target would treble this benchmark for a question that is
#: already answered.
POTENCY_VARIANTS = [v for v in C.VARIANTS if v[0] in ("two-point", "stereo centres (C7)")]


# --------------------------------------------------------------------------- #
# The data
# --------------------------------------------------------------------------- #
def stereocentres(mol):
    """The atoms RDKit calls stereocentres of this structure, read from its coordinates."""
    probe = Chem.Mol(mol)
    try:
        Chem.AssignStereochemistryFrom3D(probe)
    except Exception:
        return set()
    return {a.GetIdx() for a in probe.GetAtoms() if a.HasProp("_CIPCode")}


def load(path):
    """One file: the molecules, their potencies, and which pair each belongs to."""
    ds = load_dataset(path, activity_prop="pAffinity")
    y = np.asarray(ds.activity, float)
    tag = [m.GetProp("pair") if m.HasProp("pair") else "" for m in ds.mols]
    member = [m.GetProp("member") if m.HasProp("member") else "" for m in ds.mols]
    keep = [i for i in range(len(y)) if np.isfinite(y[i]) and tag[i] and member[i]]
    # a pair is kept only when both of its members are there
    counts = defaultdict(int)
    for i in keep:
        counts[tag[i]] += 1
    keep = [i for i in keep if counts[tag[i]] == 2]
    target = os.path.splitext(os.path.basename(path))[0]
    out = {"mols": [ds.mols[i] for i in keep], "names": [ds.names[i] for i in keep],
           "y": y[keep], "pair": [tag[i] for i in keep], "member": [member[i] for i in keep],
           "target": target}
    # the same pair can be measured against two targets, so the name of the group that is held
    # out together carries the target as well; without it the two measurements of one pair would
    # be one group and two of the four molecules would have no activity
    out["group"] = [f"{target}:{t}" for t in out["pair"]]
    out["centres"] = [stereocentres(m) for m in out["mols"]]
    out["assay"] = [m.GetProp("assay") if m.HasProp("assay") else "" for m in out["mols"]]
    return out


def pair_index(d):
    """(first, second) row numbers of every pair, in a fixed order."""
    where = defaultdict(dict)
    for i, (tag, member) in enumerate(zip(d["group"], d["member"])):
        where[tag][member] = i
    out = []
    for tag in sorted(where):
        slot = where[tag]
        if "first" in slot and "second" in slot:
            out.append((slot["first"], slot["second"]))
    return out


def gather(paths):
    """Every file as one set, with the target each molecule came from."""
    sets = [load(p) for p in paths]
    empty = [d["target"] for d in sets if not len(d["y"])]
    if empty:
        # an unreadable or empty file is a fault of the build and not a set with no pairs;
        # skipping it quietly once hid a file that a stopped run had left half written
        raise SystemExit("these files hold no usable pair, rebuild them: " + ", ".join(empty))
    whole = {k: [v for d in sets for v in d[k]]
             for k in ("mols", "names", "pair", "group", "member", "centres", "assay")}
    whole["y"] = np.concatenate([d["y"] for d in sets]) if sets else np.zeros(0)
    whole["target"] = [d["target"] for d in sets for _ in d["y"]]
    return sets, whole


# --------------------------------------------------------------------------- #
# The descriptions
# --------------------------------------------------------------------------- #
def stereo_rows(mf, mols, centres, part, n_fields):
    """The handedness channels, built from the chosen centres of every molecule."""
    rows = []
    for m, mol, cips in zip(mf, mols, centres):
        total = np.zeros(n_fields * n_fields)
        for atom, contribution in centre_channels(m, StereoSpec(), n_fields=n_fields):
            if part == "stereocentres only" and atom not in cips:
                continue
            if part == "every other centre" and atom in cips:
                continue
            total += contribution
        rows.append(total)
    return np.vstack(rows) if rows else np.zeros((0, n_fields * n_fields))


def kernels_with(mf, n_fields, S):
    """The two-point kernel with a handedness block beside it, carrying half the squared norm.

    This is `run_chirality._stereo_kernels` with the handedness block given from outside instead
    of built inside, so that a block from part of the centres is treated exactly as the whole one
    is.  `test_enantiomers.py` checks that the two agree when the whole matrix is passed.
    """
    S = np.asarray(S, float)
    sd = np.std(S, axis=0) if len(S) else np.zeros(S.shape[1])
    keep = sd > 1e-12
    S = S[:, keep] / sd[keep] if keep.any() else np.zeros((len(mf), 0))
    out = []
    for spec in C._specs(0):
        X = feature_matrix(mf, spec, n_fields=n_fields)
        blocks = block_sizes(spec, n_fields)
        scale = channel_scale(X, spec.n_bins, blocks=blocks)
        Xs = X * scale
        if S.shape[1]:
            Xs = np.hstack([Xs, S * float(np.sqrt(Xs.shape[1] / S.shape[1]))])
        out.append(autocorrelation_kernel(Xs, scale=np.ones(Xs.shape[1]),
                                          n_bins=Xs.shape[1], blocks=[Xs.shape[1]]))
    return out


def blindness(mf, n_fields, pairs):
    """The largest difference between the two-point rows of the two members of a pair.

    It should be zero: the description cannot see handedness, which is what makes the null of
    this measurement exact rather than approximate.
    """
    worst = 0.0
    for spec in C._specs(0):
        X = feature_matrix(mf, spec, n_fields=n_fields)
        for i, j in pairs:
            d = float(np.max(np.abs(X[i] - X[j]))) if len(X) else 0.0
            worst = max(worst, d)
    return worst


# --------------------------------------------------------------------------- #
# Fitting, with whole pairs held out
# --------------------------------------------------------------------------- #
def scaffold_groups(mols):
    """The group that is held out together: every molecule of one scaffold.

    Holding out a pair alone is not enough.  A set of this size holds families of near copies --
    fifty beta-blockers around one scaffold -- and a model that has seen one of them in training
    can read the answer off its neighbour instead of off the configuration.

    The configuration is taken off the molecule before the scaffold is read, and that is not a
    detail.  When the stereocentre is a ring atom it survives into the Bemis-Murcko scaffold,
    the two members of a pair then have scaffolds that differ by one tag, and they are put in
    different groups: 260 of 3,834 pairs of this collection.  Such a pair is split between the
    training part and the test part, and the damage is worse than losing the pair.  The two
    members have identical two-point rows, so a model that has seen one of them predicts the
    other as its measured value -- which, after the mean of the pair is taken off, has the
    opposite sign.  Split pairs are therefore ranked the wrong way round on purpose, and the
    description that cannot see handedness at all scored 47.9 % instead of the exact 50 % it
    has to score.  A chemotype is a skeleton, handedness is not part of one, and `check_groups`
    now refuses a run in which any pair is split.
    """
    flat = []
    for m in mols:
        c = Chem.Mol(m)
        Chem.RemoveStereochemistry(c)
        flat.append(c)
    labels, _names = chemotype_labels(flat, min_size=1)
    return [f"c{int(k)}" for k in labels]


def check_groups(groups, pairs):
    """Refuse a fold rule that would split a pair between training and test."""
    split = [(i, j) for i, j in pairs if groups[i] != groups[j]]
    if split:
        raise SystemExit(f"{len(split)} of {len(pairs)} pairs would be split between the "
                         f"training part and the test part; the groups are wrong")
    return len(set(groups))


def group_folds(groups, folds=FOLDS, seed=SEED):
    """Folds of whole groups: a pair is never split between training and test."""
    names = sorted(set(groups))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    which = {names[int(k)]: int(n % folds) for n, k in enumerate(order)}
    label = np.array([which[g] for g in groups])
    return [(np.flatnonzero(label != f), np.flatnonzero(label == f))
            for f in range(min(folds, len(names)))]


def inner_score(K, y, groups, lam, folds=INNER_FOLDS, seed=SEED):
    """Cross-validated r2 inside a training part, again with whole groups held out."""
    pred = np.full(len(y), np.nan)
    for tr, te in group_folds(groups, folds=folds, seed=seed):
        if len(tr) < 4 or not len(te):
            continue
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
        pred[te] = K[np.ix_(te, tr)] @ a + b
    ok = np.isfinite(pred)
    return float(r2_score(y[ok], pred[ok])) if ok.sum() > 2 else -np.inf


def nested(kernels, y, groups, folds=FOLDS, seed=SEED):
    """Leave whole pairs out; the grid and the regularisation are chosen inside the training
    part of every fold and never on the fold being predicted."""
    y = np.asarray(y, float)
    groups = list(groups)
    pred = np.full(len(y), np.nan)
    chosen = []
    for tr, te in group_folds(groups, folds=folds, seed=seed):
        if len(tr) < 8 or not len(te):
            continue
        inner = [groups[i] for i in tr]
        best = None
        for g, K in enumerate(kernels):
            for lam in LAMS:
                q = inner_score(K[np.ix_(tr, tr)], y[tr], inner, float(lam), seed=seed)
                if best is None or q > best[0]:
                    best = (q, g, float(lam))
        _q, g, lam = best
        K = kernels[g]
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
        pred[te] = K[np.ix_(te, tr)] @ a + b
        chosen.append({"grid": g, "lam": lam,
                       "at_edge": bool(lam == min(LAMS) or lam == max(LAMS))})
    return pred, chosen


# --------------------------------------------------------------------------- #
# What the predictions are worth
# --------------------------------------------------------------------------- #
def wilson(k, n):
    """A 95 % interval for a share, by Wilson's formula (it behaves at 0 and at 1)."""
    if not n:
        return [float("nan"), float("nan")]
    z = 1.959964
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


#: a predicted difference below this, in log units, is not an answer but a tie
TIE = 1e-9


def ranking(pred, y, pairs, margins=MARGINS):
    """How often the better enantiomer of a held-out pair is named correctly.

    A pair whose two potencies are equal has no better enantiomer and is set aside.  A model
    that gives its two members the same number names no better enantiomer either; such a pair
    counts as half a success, which is what it is worth.
    """
    out = {}
    got_all, want_all = [], []
    for i, j in pairs:
        if not (np.isfinite(pred[i]) and np.isfinite(pred[j])):
            continue
        if abs(float(y[i] - y[j])) <= TIE:
            continue
        got_all.append(float(pred[i] - pred[j]))
        want_all.append(float(y[i] - y[j]))
    got = np.asarray(got_all)
    want = np.asarray(want_all)
    out["pairs with two equal potencies, set aside"] = int(len(pairs) - len(got))
    for margin in margins:
        use = (np.abs(want) >= margin) if len(want) else np.zeros(0, bool)
        n = int(use.sum())
        tie = np.abs(got[use]) <= TIE if n else np.zeros(0, bool)
        right = float(np.sum((np.sign(got[use]) == np.sign(want[use])) & ~tie)
                      + 0.5 * np.sum(tie)) if n else 0.0
        out[f"margin {margin:.1f}"] = {
            "pairs": n, "share": (right / n) if n else float("nan"),
            "interval": wilson(right, n) if n else [float("nan")] * 2}
    if len(got) > 2:
        out["correlation of the differences"] = float(np.corrcoef(got, want)[0, 1])
        out["mean absolute error of the difference"] = float(np.mean(np.abs(got - want)))
    return out


def centred(y, pairs):
    """Half the difference inside each pair: plus on the first member, minus on the second."""
    out = np.full(len(y), np.nan)
    for i, j in pairs:
        half = 0.5 * (y[i] - y[j])
        out[i], out[j] = half, -half
    return out


# --------------------------------------------------------------------------- #
# The runs
# --------------------------------------------------------------------------- #
def fields_of(d):
    fs = FieldSet.create(physchem=FIELDS, alpha=C.ALPHA, charge_model="gasteiger")
    mf = [build_fields(m, fs, n) for m, n in zip(d["mols"], d["names"])]
    return mf, len(fs.specs)


NONE = "no handedness channels"
#: The two fold rules, which ask two different questions.  Holding out a pair asks whether a new
#: pair of a series the model knows can be ranked.  Holding out a scaffold asks whether a pair of
#: a series the model has never seen can be ranked, which is much harder and is the regime the
#: alignment-free kernel exists for.  The first number is the larger one, and reporting it alone
#: would say that a family of near copies had been learnt, not that handedness had been.
GROUPINGS = {"pairs held out": "pair", "scaffolds held out": "scaffold"}


def differences(pred, y, pairs):
    """(predicted difference, measured difference) of every held-out pair, for pooling later."""
    out = []
    for i, j in pairs:
        if np.isfinite(pred[i]) and np.isfinite(pred[j]):
            out.append([float(pred[i] - pred[j]), float(y[i] - y[j])])
    return out


def difference_run(d, label, folds=FOLDS, seed=SEED, parts=PARTS):
    """Question 1: the difference inside a pair, for one target or for all of them at once."""
    pairs = pair_index(d)
    mf, n_fields = fields_of(d)
    y = centred(d["y"], pairs)
    use = np.isfinite(y)
    if not use.all():                      # a molecule outside every pair cannot be fitted
        raise SystemExit(f"{int((~use).sum())} molecules belong to no pair")
    scaffolds = scaffold_groups(d["mols"])
    check_groups(scaffolds, pairs)
    check_groups(list(d["group"]), pairs)
    rows = {"set": label, "pairs": len(pairs), "molecules": int(use.sum()),
            "targets": len(set(d["target"])), "chemotypes": len(set(scaffolds)),
            "largest difference between the two-point rows of a pair":
                blindness(mf, n_fields, pairs),
            "spread of the difference": float(np.std(2.0 * y[use])),
            "variants": {}}
    for part in list(parts) + [NONE]:
        S = (np.zeros((len(mf), 0)) if part == NONE
             else stereo_rows(mf, d["mols"], d["centres"], part, n_fields))
        ks = kernels_with(mf, n_fields, S)
        rows["variants"][part] = {"columns": int(S.shape[1])}
        for name, kind in GROUPINGS.items():
            groups = list(d["group"]) if kind == "pair" else scaffolds
            pred, chosen = nested(ks, y, groups, folds=folds, seed=seed)
            ok = np.isfinite(pred) & use
            got = {"q2": float(r2_score(y[ok], pred[ok])) if ok.sum() > 2 else float("nan"),
                   "ranking": ranking(pred, d["y"], pairs),
                   "at_edge": float(np.mean([c["at_edge"] for c in chosen])) if chosen else 0.0,
                   "chosen": sorted(c["lam"] for c in chosen),
                   "differences": differences(pred, d["y"], pairs)}
            rows["variants"][part][name] = got
            print(f"  {part:22s} {name:19s} q2 {got['q2']:+.3f}  ranked right "
                  f"{got['ranking']['margin 0.0']['share'] * 100:.1f} % of "
                  f"{got['ranking']['margin 0.0']['pairs']}", flush=True)
    return rows


def pooled_ranking(rows, part, name, margins=MARGINS):
    """Every held-out pair of every target, from the models fitted on each target alone."""
    got = [v for row in rows for v in row["variants"][part][name]["differences"]]
    if not got:
        return {}
    pred = np.array([g[0] for g in got])
    want = np.array([g[1] for g in got])
    pairs = [(2 * i, 2 * i + 1) for i in range(len(got))]
    flat = np.empty(2 * len(got))
    flat[0::2], flat[1::2] = pred, 0.0
    truth = np.empty(2 * len(got))
    truth[0::2], truth[1::2] = want, 0.0
    return ranking(flat, truth, pairs, margins=margins)


# --------------------------------------------------------------------------- #
# A cheap reference: which configuration a target prefers
# --------------------------------------------------------------------------- #
#: The model whose shares the rule is compared with: the headline of this item, the channels read
#: at the stereocentres, with whole scaffolds held out.
REFERENCE_MODEL = ("stereocentres only", "scaffolds held out")
#: The rule reads the configuration code at one stereocentre.  A pair of molecules with one
#: stereocentre has no other choice.  With several, the centre is chosen by its canonical rank,
#: and both ends of the ranking are reported, so that the result does not rest on one choice.
REFERENCE_PICKS = ("lowest", "highest")
SUBSETS = ("all pairs", "one stereocentre", "two or more stereocentres")


def reference_codes(mol, centres):
    """The configuration code, R or S, at the stereocentre of lowest and of highest canonical rank.

    The rank is taken on the molecule without its configuration, so a pair of mirror images
    gives the same rank to the same atom in both members, and the two members get opposite
    codes at it.  A code is ``None`` where RDKit gives none.
    """
    none = {pick: None for pick in REFERENCE_PICKS}
    if not centres:
        return none
    flat = Chem.Mol(mol)
    Chem.RemoveStereochemistry(flat)
    ranks = list(Chem.CanonicalRankAtoms(flat, breakTies=False))
    probe = Chem.Mol(mol)
    try:
        Chem.AssignStereochemistryFrom3D(probe)
    except Exception:
        return none
    out = {}
    for pick, choose in (("lowest", min), ("highest", max)):
        atom = probe.GetAtomWithIdx(choose(centres, key=lambda a: ranks[a]))
        out[pick] = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
    return out


def preferred_code(votes):
    """The code that was the better enantiomer more often in the training pairs; None for a tie."""
    if votes.get("R", 0) == votes.get("S", 0):
        return None
    return "R" if votes["R"] > votes["S"] else "S"


def rule_guess(code, other, votes):
    """1 if the rule names this member the better one, 0 if it names the other, one half if it cannot.

    It cannot when the training pairs are tied, when a code is missing, or when the two members do
    not have opposite codes at the chosen centre.
    """
    want = preferred_code(votes)
    if want is None or code not in ("R", "S") or other not in ("R", "S") or code == other:
        return 0.5
    return 1.0 if code == want else 0.0


def _share_of(guess, difference):
    """The score of one pair for a guess that the first member is the better one."""
    if guess == 0.5:
        return 0.5
    return guess if difference > 0 else 1.0 - guess


def _model_share(predicted, measured):
    return 0.5 if abs(predicted) <= TIE else float(np.sign(predicted) == np.sign(measured))


def _paired_difference(a, b, draws, seed):
    """The mean of a minus b over pairs, with a 95 % interval from a bootstrap over the pairs."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.RandomState(seed)
    boot = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(draws)]
    return {"mean": float(d.mean()), "low": float(np.percentile(boot, 2.5)),
            "high": float(np.percentile(boot, 97.5))}


def configuration_reference(sets, rows, folds=FOLDS, seed=SEED, draws=4000):
    """The ranking of the better enantiomer by a rule that reads one configuration code.

    The model learns what the channels teach it about the difference inside a pair.  The rule
    learns one bit for each target: whether the R or the S member was the better one more often
    among the training pairs, read at one stereocentre.  It is scored on the same held-out pairs
    as the model, with the same folds (whole scaffolds, ten folds), so the two shares are
    counted on the same pairs.  `rows` is ``difference`` of the saved result: the predictions of
    the model are read from it and not made again.
    """
    by_target = {r["set"]: r for r in rows}
    first, grouping = REFERENCE_MODEL
    records = []
    for d in sets:
        if d["target"] not in by_target:
            continue
        pairs = pair_index(d)
        model = by_target[d["target"]]["variants"][first][grouping]["differences"]
        if len(model) != len(pairs):
            raise SystemExit(f"{d['target']}: the saved result holds {len(model)} held-out pairs "
                             f"and the data hold {len(pairs)}; run the measurement again")
        codes = [reference_codes(m, c) for m, c in zip(d["mols"], d["centres"])]
        groups = scaffold_groups(d["mols"])
        check_groups(groups, pairs)
        y = d["y"]
        for tr, te in group_folds(groups, folds=folds, seed=seed):
            train, test = set(tr.tolist()), set(te.tolist())
            votes = {pick: {"R": 0, "S": 0} for pick in REFERENCE_PICKS}
            for i, j in pairs:
                if i not in train or abs(y[i] - y[j]) <= TIE:
                    continue
                better = i if y[i] > y[j] else j
                for pick in REFERENCE_PICKS:
                    if codes[better][pick] in votes[pick]:
                        votes[pick][codes[better][pick]] += 1
            for k, (i, j) in enumerate(pairs):
                if i not in test:
                    continue
                diff = float(y[i] - y[j])
                rec = {"target": d["target"], "difference": diff,
                       "centres": len(d["centres"][i]),
                       "model": _model_share(*model[k])}
                for pick in REFERENCE_PICKS:
                    rec[pick] = _share_of(rule_guess(codes[i][pick], codes[j][pick], votes[pick]), diff)
                records.append(rec)
    out = {"folds": folds, "seed": seed, "model": " / ".join(REFERENCE_MODEL), "pooled": {},
           "targets": []}
    names = ("model",) + REFERENCE_PICKS
    for subset in SUBSETS:
        chosen = [r for r in records if subset == "all pairs"
                  or (subset == "one stereocentre" and r["centres"] == 1)
                  or (subset == "two or more stereocentres" and r["centres"] >= 2)]
        out["pooled"][subset] = {}
        for margin in MARGINS:
            use = [r for r in chosen if abs(r["difference"]) > TIE and abs(r["difference"]) >= margin]
            if not use:
                continue
            got = {k: np.array([r[k] for r in use]) for k in names}
            cell = {"pairs": len(use)}
            for k in names:
                cell[k] = float(got[k].mean())
                cell[k + " interval"] = wilson(float(got[k].sum()), len(use))
            for k in REFERENCE_PICKS:
                cell["model minus " + k] = _paired_difference(got["model"], got[k], draws, seed)
            cell["agreement of the model and the rule (lowest)"] = float(
                np.mean((got["model"] > 0.5) == (got["lowest"] > 0.5)))
            out["pooled"][subset][f"margin {margin:.1f}"] = cell
    for target in sorted({r["target"] for r in records}):
        use = [r for r in records if r["target"] == target and abs(r["difference"]) > TIE]
        if use:
            out["targets"].append({"target": target, "pairs": len(use),
                                   **{k: float(np.mean([r[k] for r in use])) for k in names}})
    out["above one half"] = {k: int(sum(t[k] > 0.5 for t in out["targets"])) for k in names}
    return out


def reference_stage(results=RESULTS, data=DATA, folds=FOLDS, seed=SEED, sets=None):
    """Run the rule on the saved result and write ``reference.json`` beside it."""
    path = os.path.join(results, "enantiomers.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing; run the measurement first")
    with open(path, encoding="utf-8") as fh:
        stored = json.load(fh)
    if sets is None:
        paths = sorted(glob.glob(os.path.join(data, "*.sdf")))
        sets, _whole = gather(paths)
    out = configuration_reference(sets, stored["difference"], folds=folds, seed=seed)
    with open(os.path.join(results, "reference.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def potency_run(d, folds=FOLDS, seed=SEED):
    """Question 2: the potency itself, for one target, with whole pairs held out."""
    pairs = pair_index(d)
    if len(pairs) < MIN_PAIRS_PER_TARGET:
        return None
    mf, n_fields = fields_of(d)
    y = np.asarray(d["y"], float)
    groups = scaffold_groups(d["mols"])
    check_groups(groups, pairs)
    row = {"target": d["target"], "pairs": len(pairs), "molecules": int(len(y)),
           "chemotypes": len(set(groups)), "spread": float(np.std(y)), "q2": {}, "ranking": {}}
    for label, shells in POTENCY_VARIANTS:
        ks = C._kernels(mf, n_fields, shells)
        pred, _chosen = nested(ks, y, groups, folds=folds, seed=seed)
        ok = np.isfinite(pred)
        row["q2"][label] = float(r2_score(y[ok], pred[ok])) if ok.sum() > 2 else float("nan")
        row["ranking"][label] = ranking(pred, y, pairs)["margin 0.0"]["share"]
        print(f"    {label:28s} q2 {row['q2'][label]:+.3f}  "
              f"ranked right {row['ranking'][label] * 100:.1f} %", flush=True)
    return row


def run(results=RESULTS, data=DATA, folds=FOLDS, seed=SEED):
    os.makedirs(results, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(data, "*.sdf")))
    if not paths:
        raise SystemExit(f"no sets in {data}; run examples/make_enantiomer_examples.py first")
    t0 = time.time()
    sets, whole = gather(paths)
    print(f"{len(whole['y'])} molecules of {len(pair_index(whole))} pairs "
          f"over {len(sets)} targets", flush=True)
    everything = pair_index(whole)
    out = {"files": [os.path.basename(p) for p in paths], "folds": folds, "seed": seed}
    # One model over every target is not built, for two reasons.  It would ask a single
    # function which of two enantiomers each of these targets prefers, which is not a question
    # a model of the structure alone can answer; and its kernel would have one row per molecule
    # of the whole collection, which costs hours.  Every target is fitted on its own instead,
    # and their held-out pairs are counted together at the end.
    out["overall"] = {"pairs": len(everything), "molecules": int(len(whole["y"])),
                      "targets": len(sets),
                      "spread of the difference":
                          float(np.std(2.0 * centred(whole["y"], everything)))}
    out["difference"] = []
    for d in sets:
        if len(pair_index(d)) < MIN_PAIRS_PER_TARGET:
            continue
        print(f"the difference inside a pair, {d['target']} "
              f"({len(pair_index(d))} pairs):", flush=True)
        out["difference"].append(difference_run(d, d["target"], folds=folds, seed=seed))
    key = "largest difference between the two-point rows of a pair"
    out["overall"][key] = max([r[key] for r in out["difference"]] or [float("nan")])
    out["targets"] = []
    for d in sets:
        if len(pair_index(d)) < MIN_PAIRS_PER_TARGET:
            continue
        print(f"the potency of {d['target']} ({len(pair_index(d))} pairs):", flush=True)
        row = potency_run(d, folds=folds, seed=seed)
        if row:
            out["targets"].append(row)
    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(results, "enantiomers.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    reference_stage(results=results, data=data, folds=folds, seed=seed, sets=sets)
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def report(results=RESULTS):
    path = os.path.join(results, "enantiomers.json")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    overall = d["overall"]
    blind = overall["largest difference between the two-point rows of a pair"]
    rows = list(d.get("difference", []))
    first = next(iter(GROUPINGS))
    set_aside = sum(r["variants"][NONE][first]["ranking"]
                    ["pairs with two equal potencies, set aside"] for r in rows)
    L = ["# Can the program tell two enantiomers apart?", "",
         f"Plan item MC3, on {overall['pairs']} enantiomeric pairs "
         f"({overall['molecules']} molecules) of {overall['targets']} targets, built by "
         "`examples/make_enantiomer_examples.py` from ChEMBL. Both potencies of a pair come "
         "from one assay. The two members of a pair are exact reflections of one conformation, "
         "so the two-point description gives them identical rows: the largest difference found "
         f"between the rows of a pair is {blind:.1e}. "
         "A model without handedness channels therefore cannot rank a pair, and its share of "
         "correctly ranked pairs is one half by construction, not by chance.", "",
         "The mean of each pair is taken off its two potencies, so what is left is half the "
         "difference between the two enantiomers. The share below is how often the better "
         "enantiomer of a held-out pair is named correctly, with a 95 % interval in brackets; a "
         "share whose interval holds 50 % is no better than naming one of the two at random.",
         "", "**The two fold rules answer two different questions.** With pairs held out, the "
         "two members of a held-out pair are not in the training part but other pairs of the "
         "same scaffold may be, so the question is whether a new pair of a series the model "
         "already knows can be ranked. With scaffolds held out, every molecule of a scaffold "
         "leaves together, so the question is whether a pair of a series the model has never "
         "seen can be ranked. The second is the harder question and the one the alignment-free "
         "kernel exists for. The two give similar answers here, which says the signal is not "
         "mostly read off a near neighbour of the held-out pair.", "",
         "A pair whose two potencies are equal has no better enantiomer and is left out of the "
         f"share; there are {set_aside} such pairs. A model that gives the two members of a "
         "pair the "
         "same number names no better enantiomer either, and such a pair counts as half a "
         "success, which is what it is worth."]
    heads = list(rows[0]["variants"]) if rows else []
    for name in GROUPINGS:
        L += ["", f"## The difference inside a pair, with {name}", "",
              "Every target is fitted on its own. The last row counts the held-out pairs of all "
              "of them together.", "",
              "| Target | Pairs | Scaffolds | " + " | ".join(heads) + " |",
              "|---" * (len(heads) + 3) + "|"]
        for row in rows:
            cells = []
            for part in heads:
                r = row["variants"][part][name]["ranking"]["margin 0.0"]
                lo, hi = r["interval"]
                cells.append(f"{r['share'] * 100:.1f} % ({lo * 100:.0f} to {hi * 100:.0f})"
                             if r["pairs"] else "no pair")
            L.append(f"| {row['set'].replace('_', ' ')} | {row['pairs']} | "
                     f"{row['chemotypes']} | " + " | ".join(cells) + " |")
        if rows:
            cells, counted = [], 0
            for part in heads:
                r = pooled_ranking(rows, part, name).get("margin 0.0", {})
                lo, hi = r.get("interval", [float("nan")] * 2)
                counted = max(counted, int(r.get("pairs", 0)))
                cells.append(f"**{r['share'] * 100:.1f} %** ({lo * 100:.0f} to {hi * 100:.0f})"
                             if r.get("pairs") else "no pair")
            L.append(f"| **every target, each fitted on its own** | {counted} | | "
                     + " | ".join(cells) + " |")
        L += ["", "As q2 of the half-difference (at or below zero means the model predicts the "
                  "difference no better than predicting zero for every pair):", "",
              "| Target | " + " | ".join(heads) + " |", "|---" * (len(heads) + 1) + "|"]
        for row in rows:
            L.append(f"| {row['set'].replace('_', ' ')} | "
                     + " | ".join(f"{row['variants'][p][name]['q2']:+.3f}" for p in heads)
                     + " |")
    if rows:
        L += ["", "Among the pairs whose two potencies differ by at least one log unit, with "
                  "every target fitted on its own:", "",
              "| Fold rule | " + " | ".join(heads) + " |", "|---" * (len(heads) + 1) + "|"]
        for name in GROUPINGS:
            cells = []
            for part in heads:
                r = pooled_ranking(rows, part, name).get("margin 1.0", {})
                cells.append(f"{r['share'] * 100:.1f} % of {r['pairs']}"
                             if r.get("pairs") else "no pair")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L += ["", f"The spread of the difference between the two potencies of a pair is "
                  f"{overall['spread of the difference']:.2f} log units."]
    ref_path = os.path.join(results, "reference.json")
    if rows and os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as fh:
            ref = json.load(fh)
        L += ["", "## A cheap reference: which configuration the target prefers", "",
              "The channels are compared with a rule that learns one bit for each target. The "
              "training pairs say whether the R or the S member of a pair was the better one "
              "more often, and the rule names that member in a held-out pair. It reads the "
              "configuration code at one stereocentre. The two members of a pair have opposite "
              "codes at the same atom, and the atom is chosen by its canonical rank, which is "
              "the rank that RDKit gives an atom from its place in the graph of the molecule. "
              "A molecule with one stereocentre has no other choice. With several, the choice is "
              "arbitrary, so the centre of lowest rank and the centre of highest rank are both "
              "given. The folds are those of the model (whole scaffolds), and the shares are "
              "counted on the same held-out pairs.", "",
              "| Pairs | Pairs counted | Channels at the stereocentres | Rule, centre of lowest "
              "rank | Rule, centre of highest rank | Channels minus rule (lowest) | Channels "
              "minus rule (highest) |", "|---" * 7 + "|"]
        for subset, cells in ref["pooled"].items():
            for margin, c in cells.items():
                label = subset + ("" if margin == "margin 0.0" else
                                  f", differing by {margin.split()[1]} log units or more")
                m1, m2 = c["model minus lowest"], c["model minus highest"]
                L.append(f"| {label} | {c['pairs']} | {c['model'] * 100:.1f} % | "
                         f"{c['lowest'] * 100:.1f} % | {c['highest'] * 100:.1f} % | "
                         f"{m1['mean'] * 100:+.1f} points ({m1['low'] * 100:+.1f} to "
                         f"{m1['high'] * 100:+.1f}) | {m2['mean'] * 100:+.1f} points "
                         f"({m2['low'] * 100:+.1f} to {m2['high'] * 100:+.1f}) |")
        a = ref["above one half"]
        L += ["", f"Of the {len(ref['targets'])} targets, the channels name the better enantiomer "
                  f"in more than half of the pairs on {a['model']}, the rule with the centre of "
                  f"lowest rank on {a['lowest']} and the rule with the centre of highest rank "
                  f"on {a['highest']}. The first three targets of the channels:", "",
              "| Target | Pairs | Channels | Rule (lowest) | Rule (highest) |", "|---" * 5 + "|"]
        for t in sorted(ref["targets"], key=lambda t: -t["model"])[:3]:
            L.append(f"| {t['target'].replace('_', ' ')} | {t['pairs']} | {t['model'] * 100:.1f} % "
                     f"| {t['lowest'] * 100:.1f} % | {t['highest'] * 100:.1f} % |")
    if d.get("targets"):
        L += ["", "## The potency itself, one target at a time", "",
              "The ordinary question of item C7, on sets that do hold enantiomeric pairs: does "
              "the handedness block add anything to the two-point kernel? Whole pairs are held "
              "out, and every variant chooses its distance grid and its regularisation inside "
              "the training part of each fold.", ""]
        labels = [k for k in d["targets"][0]["q2"]]
        L += ["| Target | Pairs | " + " | ".join(labels) + " |",
              "|---" * (len(labels) + 2) + "|"]
        for row in d["targets"]:
            L.append(f"| {row['target'].replace('_', ' ')} | {row['pairs']} | "
                     + " | ".join(f"{row['q2'][k]:+.3f}" for k in labels) + " |")
    L.append("")
    text = "\n".join(L)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--report", action="store_true", help="write the table from the saved result")
    ap.add_argument("--reference", action="store_true",
                    help="run only the rule that reads one configuration code, from the saved result")
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.reference:
        reference_stage(folds=args.folds, seed=args.seed)
    elif not args.report:
        run(folds=args.folds, seed=args.seed)
    print(report())


if __name__ == "__main__":
    main()
