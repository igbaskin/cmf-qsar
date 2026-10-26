#!/usr/bin/env python
"""
Do the signed three-point channels add anything?  C5 of the development plan, measured.

The field autocorrelation of `cmfqsar/invariant.py` is alignment-free and geometry-aware, and it
is two-point: it counts pairs of atoms at a distance, which is a quantity a mirror image has in
exactly the same amounts.  The kernel is therefore blind to chirality, and blind as well to the
orientation of one field with respect to another.  The signed triple product of three
displacement vectors is not, and `chirality()` adds it as a second block of channels, resolved
over a few radial shells.

Whether that is worth the columns it costs is an empirical question, and the standard is set by the
descriptor it extends rather than by an aligned model: if the three-point channels add nothing to
the two-point ones, the item is closed.

The test is leave-one-chemotype-out on the five chemotype-split sets, which is the regime the
alignment-free kernel exists for -- a held-out molecule has no scaffold-mate in the training part,
so nothing can be recovered from a neighbour of the same series.

Every variant gets the same freedom and the same fairness: three regularisations by three distance
grids, chosen **inside the training part of every fold** by a 10-fold cross-validation and never
on the fold being predicted.  The number of radial shells is not searched over; each setting is a
column of its own, so that the signed channels are never given more freedom than the kernel they
are being compared with.

Usage
-----
    python benchmarks/run_chirality.py
    python benchmarks/run_chirality.py --sets COX2 DHFR
    python benchmarks/run_chirality.py --report
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

from cmfqsar.chemotypes import chemotype_labels  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               block_sizes, channel_scale, feature_matrix, field_pairs)
from cmfqsar.model import cv_predict, krr_fit, r2_score  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "chirality")
DATA = os.path.join(ROOT, "examples")
#: the five sets the README's chemotype table uses
SETS = {
    "COX2": "sutherland/COX2.sdf",
    "DHFR": "sutherland/DHFR.sdf",
    "BZR": "sutherland/BZR.sdf",
    "AChE": "sutherland/AChE.sdf",
    "mittal/ACHE": "mittal/ACHE.sdf",
}
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 0.5                     # the attenuation does not enter this descriptor; it builds fields
LAMS = (0.01, 0.1, 1.0)
#: the three distance grids, as (sigma, step, d_max) -- the freedom the aligned model gets from
#: its three attenuations
GRIDS = ((0.5, 0.5, 10.0), (0.75, 0.75, 12.0), (1.0, 1.0, 14.0))
#: the variants compared: the two-point kernel, and the same with the signed channels over 1, 2
#: and 4 radial shells
VARIANTS = (("two-point", 0), ("signed, 1 shell", 1), ("signed, 2 shells", 2),
            ("signed, 4 shells", 4), ("signed, 4 shells, balanced", -4),
            ("stereo centres (C7)", "stereo"))

#: a negative shell count in VARIANTS means: scale the signed block up until it carries half the
#: squared norm of a molecule's channel vector.  Left alone it carries 40 columns against the
#: pairs' 255, so a negative result would be a statement about that ratio and not about the
#: information in the channels


def _specs(shells):
    out = []
    for sigma, step, d_max in GRIDS:
        out.append(AutocorrelationSpec(sigma=sigma, step=step, d_max=d_max,
                                       chirality=bool(shells),
                                       chirality_shells=max(1, int(shells))))
    return out


def _stereo_kernels(mf, n_fields):
    """The two-point channels with C7's signed centres beside them, carrying half the norm.

    The pair blocks are left exactly as they are, so this is the two-point kernel plus a block,
    and the block is scaled to the same squared norm as the rest -- the same balance the signed
    three-point variant is given, so the two are compared on equal terms.
    """
    from cmfqsar.stereo import StereoSpec, stereo_matrix

    S, _info = stereo_matrix(mf, StereoSpec(), n_fields=n_fields)
    sd = np.std(S, axis=0)
    S = S / np.where(sd > 1e-12, sd, 1.0)
    keep = sd > 1e-12
    S = S[:, keep]
    out = []
    for spec in _specs(0):
        X = feature_matrix(mf, spec, n_fields=n_fields)
        blocks = block_sizes(spec, n_fields)
        scale = channel_scale(X, spec.n_bins, blocks=blocks)
        Xs = X * scale
        if S.shape[1]:
            Xs = np.hstack([Xs, S * float(np.sqrt(Xs.shape[1] / S.shape[1]))])
        out.append(autocorrelation_kernel(Xs, scale=np.ones(Xs.shape[1]),
                                          n_bins=Xs.shape[1], blocks=[Xs.shape[1]]))
    return out


def _kernels(mf, n_fields, shells):
    """One cosine-normalised kernel over every molecule, per distance grid."""
    if shells == "stereo":
        return _stereo_kernels(mf, n_fields)
    balance = shells < 0
    shells = abs(shells)
    out = []
    for spec in _specs(shells):
        X = feature_matrix(mf, spec, n_fields=n_fields)
        blocks = block_sizes(spec, n_fields)
        scale = channel_scale(X, spec.n_bins, blocks=blocks)
        if balance and shells:
            n_pair = len(field_pairs(n_fields)) * spec.n_bins
            n_chi = int(sum(blocks)) - n_pair
            if n_chi > 0:
                # every block already has unit root mean square, so a block's share of the
                # squared norm is its number of columns; this makes the two parts equal
                scale = np.asarray(scale, float).copy()
                scale[n_pair:] *= float(np.sqrt(n_pair / n_chi))
        out.append(autocorrelation_kernel(X, scale=scale, n_bins=spec.n_bins, blocks=blocks))
    return out


def _grouped_nested(kernels, y, labels, inner_folds=10, seed=0):
    """Leave one chemotype out, choosing the grid and lambda inside the training part."""
    pred = np.full(len(y), np.nan)
    for g in sorted(set(labels.tolist())):
        te = np.flatnonzero(labels == g)
        tr = np.flatnonzero(labels != g)
        if len(tr) < 8 or not len(te):
            continue
        best = None
        for K in kernels:
            Ktr = K[np.ix_(tr, tr)]
            for lam in LAMS:
                p = cv_predict(Ktr, y[tr], "krr", {"lam": float(lam)},
                               folds=min(inner_folds, len(tr)), seed=seed)
                q = r2_score(y[tr], p)
                if best is None or q > best[0]:
                    best = (q, K, float(lam))
        _q, K, lam = best
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
        pred[te] = K[np.ix_(te, tr)] @ a + b
    return pred


# --------------------------------------------------------------------------- #
# The experiment the five sets cannot do
# --------------------------------------------------------------------------- #
#: how much of the activity's variance is carried by handedness rather than by composition
MIXTURES = (0.0, 0.25, 0.5, 0.75, 1.0)

#: a chiral centre with four different substituents, varied over a small library
SCAFFOLDS = [
    "N[C@@H](C)C(=O)O", "N[C@@H](CC)C(=O)O", "N[C@@H](CO)C(=O)O", "N[C@@H](CS)C(=O)O",
    "N[C@@H](CCl)C(=O)O", "N[C@@H](CF)C(=O)O", "N[C@@H](CN)C(=O)O", "N[C@@H](CCC)C(=O)O",
    "O[C@@H](C)C(=O)N", "O[C@@H](CC)C(=O)N", "O[C@@H](CO)C(=O)N", "O[C@@H](CCl)C(=O)N",
    "S[C@@H](C)C(=O)O", "S[C@@H](CC)C(=O)O", "F[C@@H](C)C(=O)O", "F[C@@H](CC)C(=O)O",
    "Cl[C@@H](C)C(=O)N", "Cl[C@@H](CC)C(=O)N", "N[C@@H](C)C(=O)OC", "N[C@@H](CC)C(=O)OC",
]


def _mirrored(mol):
    """The exact enantiomer: the same conformation reflected, not a new embedding.

    Two separately generated conformations of R and S differ in their torsions as much as in
    their handedness, and the two-point descriptor sees that difference -- which would let it
    appear to solve a problem it is blind to.  Reflecting one conformation leaves the internal
    geometry alone, so the pair differs in handedness and in nothing else, and the two-point
    channels of the two are identical in every bit.
    """
    from rdkit import Chem

    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i, p in enumerate(np.array(conf.GetPositions()) * np.array([1.0, 1.0, -1.0])):
        conf.SetAtomPosition(i, [float(v) for v in p])
    return m


def _pairs(seed=0):
    """Each scaffold as one conformation and its exact mirror image."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mols, names, hand = [], [], []
    for i, smi in enumerate(SCAFFOLDS):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = seed + i
        if AllChem.EmbedMolecule(m, p) != 0:
            continue
        Chem.SanitizeMol(m)
        for mm in (m, _mirrored(m)):
            # the handedness has to be the CIP descriptor and not the @@ of the SMILES: the same
            # tag is R on one scaffold and S on the next, since the priorities depend on the
            # substituents, and a model asked to learn that label would be asked to learn noise
            probe = Chem.Mol(mm)
            Chem.AssignStereochemistryFrom3D(probe)
            codes = [a.GetProp("_CIPCode") for a in probe.GetAtoms() if a.HasProp("_CIPCode")]
            if len(codes) != 1:
                continue
            mols.append(mm)
            names.append(f"s{i}-{codes[0]}")
            hand.append(+1.0 if codes[0] == "R" else -1.0)
    return mols, names, np.asarray(hand)


def synthetic(force=False, results=RESULTS, seed=0):
    """Can the signed channels learn an activity that depends on handedness?

    The five real sets cannot answer it: none of them contains an enantiomeric pair, so the
    blindness of the two-point kernel costs it nothing there and the measurement is about
    something else.  Here the activity is built as

        y = (1 - w) * composition + w * handedness

    over pairs that differ in handedness alone.  At w = 0 the signed channels have nothing to
    find and must not appear to find it; at w = 1 the two-point kernel is being asked to tell
    apart two molecules whose every channel is identical, which it cannot do at any lambda, and
    the signed kernel should.
    """
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, "synthetic.json")
    if os.path.exists(out_path) and not force:
        print("synthetic: already done")
        return json.load(open(out_path))
    t0 = time.time()
    mols, names, hand = _pairs(seed=seed)
    fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA, charge_model="gasteiger")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    n_fields = len(fs.specs)
    # the achiral half of the activity: a property of the composition, identical within a pair
    comp = np.array([float(m.GetNumHeavyAtoms()) for m in mols])
    comp = (comp - comp.mean()) / max(comp.std(), 1e-9)
    print(f"=== synthetic: {len(mols)} molecules, {len(mols) // 2} enantiomeric pairs", flush=True)

    kernels = {label: _kernels(mf, n_fields, shells) for label, shells in VARIANTS}
    # the claim the whole block rests on, checked rather than assumed
    K2 = kernels["two-point"][1]
    same = float(np.max(np.abs(np.diag(K2, 1)[::2] - 1.0)))
    print(f"    two-point kernel between the members of a pair: 1 - {same:.2e}", flush=True)

    # why it comes out as it does: is "R-ness" a direction two molecules can share?
    direction = _shared_direction(mf, names, hand, n_fields)
    direction_c7 = _shared_direction(mf, names, hand, n_fields, stereo=True)
    for what, d in (("C5, signed three-point", direction), ("C7, stereo centres", direction_c7)):
        print(f"    {what:24s} same handedness {d['same']:+.3f}, opposite {d['opposite']:+.3f}, "
              f"difference {d['difference']:+.3f} against a spread of {d['sd']:.3f}", flush=True)

    rows = []
    for w in MIXTURES:
        y = (1.0 - w) * comp + w * hand
        y = (y - y.mean()) / max(y.std(), 1e-9)
        row = {"w": float(w), "q2": {}}
        for label, _shells in VARIANTS:
            pred = _random_nested(kernels[label], y, folds=10, seed=seed)
            ok = np.isfinite(pred)
            row["q2"][label] = float(r2_score(y[ok], pred[ok])) if ok.any() else float("nan")
        rows.append(row)
        print("    w = {:.2f}  ".format(w)
              + "  ".join(f"{k} {row['q2'][k]:+.3f}" for k, _ in VARIANTS), flush=True)
    out = {"n": len(mols), "pairs": len(mols) // 2, "identical_to": same, "rows": rows,
           "direction": direction, "direction_c7": direction_c7,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def _shared_direction(mf, names, hand, n_fields, shells=4, stereo=False):
    """Do two molecules of the same handedness point the same way in the signed channels?

    A model transfers what it has learnt from one molecule to another along a direction in
    feature space.  The signed channels of a molecule and its own mirror image are exact
    negatives, which is what the block is for; whether the channels of two *different* molecules
    of the same handedness have anything in common is a different question, and the one that
    decides whether a kernel method can carry handedness from a training molecule to a new one.
    """
    from cmfqsar.invariant import chirality

    if stereo:
        from cmfqsar.stereo import StereoSpec, stereo_matrix
        V = stereo_matrix(mf, StereoSpec(), n_fields=n_fields)[0]
    else:
        spec = AutocorrelationSpec(chirality=True, chirality_shells=shells)
        V = np.array([chirality(m, spec) for m in mf])
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    scaffold = [n.split("-")[0] for n in names]
    same, opposite = [], []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if scaffold[i] == scaffold[j]:
                continue                       # a molecule against its own mirror image
            (same if hand[i] == hand[j] else opposite).append(float(V[i] @ V[j]))
    if not same or not opposite:
        return {"same": float("nan"), "opposite": float("nan"), "difference": float("nan"),
                "sd": float("nan"), "n": 0}
    return {"same": float(np.mean(same)), "opposite": float(np.mean(opposite)),
            "difference": float(np.mean(same) - np.mean(opposite)),
            "sd": float(np.std(same + opposite)), "n": len(same) + len(opposite)}


def _random_nested(kernels, y, folds=10, seed=0):
    """Random folds, with the grid and lambda chosen inside the training part of each."""
    from cmfqsar.model import _kfold_indices

    pred = np.full(len(y), np.nan)
    for te in _kfold_indices(len(y), folds, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        best = None
        for K in kernels:
            Ktr = K[np.ix_(tr, tr)]
            for lam in LAMS:
                p = cv_predict(Ktr, y[tr], "krr", {"lam": float(lam)},
                               folds=min(folds, len(tr)), seed=seed)
                q = r2_score(y[tr], p)
                if best is None or q > best[0]:
                    best = (q, K, float(lam))
        _q, K, lam = best
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
        pred[te] = K[np.ix_(te, tr)] @ a + b
    return pred


def run_set(name, force=False, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, name.replace("/", "_") + ".json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(os.path.join(DATA, SETS[name]))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    y = y[keep]
    labels, _scaffolds = chemotype_labels(mols, min_size=3)
    labels = np.asarray(labels)
    n_groups = len(set(labels.tolist()))
    print(f"=== {name}: {len(y)} molecules, {n_groups} chemotypes", flush=True)

    fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA, charge_model="file")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    n_fields = len(fs.specs)

    row = {"set": name, "n": int(len(y)), "n_groups": int(n_groups), "q2": {}, "columns": {}}
    for label, shells in VARIANTS:
        ks = _kernels(mf, n_fields, shells)
        pred = _grouped_nested(ks, y, labels)
        ok = np.isfinite(pred)
        row["q2"][label] = float(r2_score(y[ok], pred[ok])) if ok.any() else float("nan")
        if shells == "stereo":
            from cmfqsar.stereo import StereoSpec, stereo_matrix
            row["columns"][label] = int(
                feature_matrix(mf[:1], _specs(0)[1], n_fields=n_fields).shape[1]
                + stereo_matrix(mf[:1], StereoSpec(), n_fields=n_fields)[0].shape[1])
        else:
            row["columns"][label] = int(ks[1].shape[0] and feature_matrix(
                mf[:1], _specs(abs(shells))[1], n_fields=n_fields).shape[1])
        print(f"  {label:17s} q2 {row['q2'][label]:+.3f}  ({row['columns'][label]} channels)",
              flush=True)
    row["seconds"] = round(time.time() - t0, 1)
    json.dump(row, open(out_path, "w"), indent=1)
    return row


def report(results=RESULTS):
    import glob

    files = sorted(f for f in glob.glob(os.path.join(results, "*.json"))
                   if os.path.basename(f) != "synthetic.json")
    if not files:
        return "No chirality results yet; run `python benchmarks/run_chirality.py`."
    data = [json.load(open(f)) for f in files]
    order = [name for name, _ in VARIANTS]
    lines = ["# Can a field kernel see handedness?", "",
             "Two constructions: C5's signed three-point channels, summed over every triple of "
             "atoms in the molecule, and C7's stereo centres, read at one atom at a time. The "
             "five real sets below cannot separate them, because none of them contains an "
             "enantiomeric pair; the synthetic experiment at the bottom is what does.", "",
             "Leave-one-chemotype-out on the five chemotype-split sets, the regime the "
             "alignment-free kernel exists for. Three regularisations by three distance grids for "
             "every variant, chosen inside the training part of each fold; the number of radial "
             "shells is a column rather than a search, so the signed channels never get more "
             "freedom than the two-point kernel they extend.", "",
             "| set | n | chemotypes | " + " | ".join(order) + " |",
             "|---|---:|---:|" + "---:|" * len(order)]
    for d in sorted(data, key=lambda r: -r["n"]):
        best = max(d["q2"].values())
        cells = []
        for k in order:
            v = d["q2"].get(k, float("nan"))
            cells.append(f"**{v:+.3f}**" if abs(v - best) < 1e-9 else f"{v:+.3f}")
        lines.append(f"| {d['set']} | {d['n']} | {d['n_groups']} | " + " | ".join(cells) + " |")
    means = {k: float(np.nanmean([d["q2"].get(k, np.nan) for d in data])) for k in order}
    top = max(means.values())
    lines.append("| **mean** |  |  | " + " | ".join(
        (f"**{means[k]:+.3f}**" if abs(means[k] - top) < 1e-9 else f"{means[k]:+.3f}")
        for k in order) + " |")
    cols = {k: int(np.mean([d["columns"].get(k, 0) for d in data])) for k in order}
    lines += ["", "Channels per molecule, averaged over the sets: "
              + ", ".join(f"{k} {cols[k]}" for k in order) + ".", "",
              "## Is the difference larger than the scatter?", "",
              "Paired against the two-point kernel, set by set, with a bootstrap interval over "
              "the five sets. Five is a small number of sets and the interval says so.", "",
              "| variant | mean difference | 95 % interval | better on |", "|---|---:|---|---:|"]
    base = np.array([d["q2"].get(order[0], np.nan) for d in data])
    rng = np.random.default_rng(0)
    for k in order[1:]:
        diff = np.array([d["q2"].get(k, np.nan) for d in data]) - base
        diff = diff[np.isfinite(diff)]
        if not len(diff):
            continue
        bt = [float(np.mean(rng.choice(diff, len(diff), replace=True))) for _ in range(5000)]
        lines.append(f"| `{k}` | {float(np.mean(diff)):+.3f} | "
                     f"[{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}] | "
                     f"{int((diff > 0).sum())}/{len(diff)} |")
    # the one set that carries the mean, named rather than left to be noticed
    best = order[-1]
    diff = {d["set"]: d["q2"].get(best, np.nan) - d["q2"].get(order[0], np.nan) for d in data}
    worst_first = sorted(diff.items(), key=lambda kv: -abs(kv[1]))
    if worst_first:
        name, value = worst_first[0]
        rest = [v for k, v in diff.items() if k != name and np.isfinite(v)]
        lines += ["", f"At {best.replace('`', '')} the mean difference is "
                  f"{float(np.nanmean(list(diff.values()))):+.3f}, of which **{name} supplies "
                  f"{value:+.3f}**; over the other four sets it is "
                  f"{float(np.mean(rest)):+.3f}."]
    syn_path = os.path.join(results, "synthetic.json")
    if os.path.exists(syn_path):
        syn = json.load(open(syn_path))
        order2 = [name for name, _ in VARIANTS]
        lines += ["", "## The experiment the five sets cannot do", "",
                  f"{syn['pairs']} pairs of molecules that differ in handedness and in nothing "
                  "else -- one conformation and its exact mirror image, so the two-point channels "
                  f"of a pair are identical (their kernel entry is 1 - "
                  f"{syn['identical_to']:.0e}). The activity is "
                  "`(1 - w) * composition + w * handedness`, so `w` is the share of it that no "
                  "two-point descriptor can reach. Random 10-fold folds, the grid and lambda "
                  "chosen inside each training part.", "",
                  "| w | " + " | ".join(order2) + " |",
                  "|---:|" + "---:|" * len(order2)]
        for r in syn["rows"]:
            best = max(r["q2"].values())
            cells = [(f"**{r['q2'][k]:+.3f}**" if abs(r["q2"][k] - best) < 1e-9
                      else f"{r['q2'][k]:+.3f}") for k in order2]
            lines.append(f"| {r['w']:.2f} | " + " | ".join(cells) + " |")
        for key, what in (("direction", "C5's signed three-point channels"),
                          ("direction_c7", "C7's stereo centres")):
            dd = syn.get(key) or {}
            if dd and np.isfinite(dd.get("difference", float("nan"))):
                lines += ["", f"**{what}:** same handedness {dd['same']:+.3f}, opposite "
                          f"{dd['opposite']:+.3f}, a difference of {dd['difference']:+.3f} "
                          f"against a spread of {dd['sd']:.3f} over {dd['n']} pairs."]
        d = syn.get("direction") or {}
        if d and np.isfinite(d.get("difference", float("nan"))):
            lines += ["", "**Why.** A kernel method carries what it learnt from one molecule to "
                      "another along a direction in feature space. The signed channels of a "
                      "molecule and its own mirror image are exact negatives -- that is what the "
                      "block is for -- but between *different* molecules they share nothing: the "
                      f"cosine of the signed block is {d['same']:+.3f} for two molecules of the "
                      f"same handedness and {d['opposite']:+.3f} for two of opposite handedness, "
                      f"a difference of {d['difference']:+.3f} against a spread of {d['sd']:.3f} "
                      f"over {d['n']} pairs. Handedness is not a direction in this space, so "
                      "there is nothing for a linear model to transfer -- which explains all of "
                      "C5's failure, and what C7 was built to fix by reading the configuration "
                      "at one centre instead of summing over the molecule."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--synthetic-only", action="store_true",
                    help="only the handedness experiment, not the five real sets")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    if not a.synthetic_only:
        for name in a.sets:
            run_set(name, force=a.force)
    synthetic(force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
