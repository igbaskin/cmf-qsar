#!/usr/bin/env python
"""
Do quantum fields predict a reduction potential better than the ordinary fields, and better than
the global quantum numbers alone? (plan item EC1)

The series is 69 substituted phenazines with a computed one-electron reduction potential
(``examples/phenazine/``, built by ``examples/make_phenazine_examples.py``). Every molecule keeps
the geometry its source gives, and all of them are superimposed on the phenazine core, so the
series has a frame and the overlap kernel applies.

Item EC1 added three things to the quantum stage: the charge and the spin state read from the
record, a continuum solvent, and the Mulliken spin populations of the two ionic states as per-atom
fields. The last of these is the one the item is about. A reduction potential is the energy of
adding one electron, and the spin population of the radical anion says **where that electron went**.
That is a field, one number per atom, and the question is whether it carries anything the ordinary
fields and the global descriptors do not.

The models, all on the same molecules, the same folds and the same grids:

  ea            the electron affinity alone, one number per molecule, no geometry
  descriptors   the nine global quantum numbers, no geometry
  physchem      the five fields of the published table (Gasteiger electrostatic, steric,
                hydrophobic, hba, hbd)
  hirshfeld     the same, with the Hirshfeld charge field in place of the Gasteiger one
  fukui         physchem plus f+, f- and the dual descriptor as per-atom fields
  spin          physchem plus the spin population of the radical anion
  quantum       physchem plus all of the above quantum fields
  spin-only     the spin field by itself, to see whether it stands alone

``ea`` is the control that makes every other number readable, and it is a strict one. The target
is a computed reduction potential and the electron affinity is nearly the same quantity reduced to
one number: Pearson r between them is +0.77 on this series. A field that does not beat ``ea`` has
not earned the ten hours of DFT that produced it.

**The circularity, stated plainly.** The target is computed, not measured, and the spin field comes
from the same kind of calculation as the target. So a high score for ``spin`` is not evidence that
the field would predict an experiment; it is evidence that the field carries what the anion
calculation knows. ``ea`` is what separates the two readings: if ``spin`` beats ``physchem`` but not
``ea``, then the field is only recovering the global number, more expensively.

Two fold designs, because they ask different questions:

  random                 five random folds: can the model interpolate within the series?
  one source table held  the three source tables held out one at a time (29 external test
                         compounds, 20 two-substituent, 20 three-substituent): can it reach a
                         group of compounds it has not seen?

Every model chooses its ridge and its field width inside each fold, never on the held-out rows.
The comparisons are paired bootstrap intervals over the per-molecule absolute errors, so a
difference that does not exclude zero is not a difference.

Usage
-----
    python benchmarks/run_phenazine.py                 # every stage
    python benchmarks/run_phenazine.py --stage gate
    python benchmarks/run_phenazine.py --table
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
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger                                            # noqa: E402

import kernel_cache as KC                                                   # noqa: E402

from cmfqsar.cofields import cofield_volume, make_grid                      # noqa: E402
from cmfqsar.dataio import relative_path                                    # noqa: E402
from cmfqsar.descriptors import read_values                                 # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields                # noqa: E402
from cmfqsar.model import CMFModel, _Learner, cv_predict, r2_score, rmse    # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "phenazine")
DATA = os.path.join(ROOT, "examples", "phenazine")
SDF = os.path.join(DATA, "phenazine_qm.sdf")

#: the protocol constants of the other benchmarks, so that the numbers are comparable
SEED = 0
OUTER, INNER = 5, 10
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
ALPHAS = (0.3, 0.5, 1.0, 2.0)
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")

#: the target: the one-electron reduction potential in volts
TARGET = "E0_V"
#: the property that says which source table a molecule came from
GROUP = "subset"

#: the global quantum numbers a DFT paper on this series would use. The electron affinity is the
#: first of them because it is nearly the target itself.
DESCRIPTORS = ("EA", "IP", "E_homo", "E_lumo", "gap", "hardness", "softness", "electrophilicity",
               "dipole")
#: the single number that is closest to the target
ONE_DESCRIPTOR = ("EA",)

#: per-atom property fields, by group
HIRSHFELD = ("q_hirshfeld",)
FUKUI = ("fukui_plus", "fukui_minus", "dual")
SPIN = ("spin_anion",)

#: model -> (physico-chemical fields, per-atom property fields, global descriptors)
MODELS: Dict[str, Tuple[Sequence[str], Sequence[str], Sequence[str]]] = {
    "ea": ((), (), ONE_DESCRIPTOR),
    "descriptors": ((), (), DESCRIPTORS),
    "physchem": (FIELDS, (), ()),
    "hirshfeld": (FIELDS[1:], HIRSHFELD, ()),
    "fukui": (FIELDS, FUKUI, ()),
    "spin": (FIELDS, SPIN, ()),
    "quantum": (FIELDS[1:], HIRSHFELD + FUKUI + SPIN, ()),
    "spin-only": ((), SPIN, ()),
}
MODEL_ORDER = ("ea", "descriptors", "physchem", "hirshfeld", "fukui", "spin", "quantum",
               "spin-only")


# ------------------------------------------------------------------------------- the data


def load() -> Tuple[List[Chem.Mol], List[str], np.ndarray, np.ndarray]:
    """The molecules with their quantum properties, the target and the source table of each."""
    if not os.path.exists(SDF):
        raise SystemExit(f"{SDF} is missing: run examples/make_phenazine_examples.py --quantum")
    mols, names, y, groups = [], [], [], []
    for m in Chem.SDMolSupplier(SDF, removeHs=False):
        if m is None or not m.HasProp(TARGET):
            continue
        mols.append(m)
        names.append(m.GetProp("_Name"))
        y.append(float(m.GetProp(TARGET)))
        groups.append(m.GetProp(GROUP) if m.HasProp(GROUP) else "all")
    return mols, names, np.asarray(y, float), np.asarray(groups, dtype=object)


#: the phenazine core itself, which every molecule of the series has
CORE = Chem.MolFromSmarts("c1ccc2nc3ccccc3nc2c1")


def core_nitrogens(mol: Chem.Mol) -> List[int]:
    """The two ring nitrogens of the phenazine core.

    They are taken from the core match and not from the ring membership: each core nitrogen sits
    in one ring only (the central one; the atoms two rings share are the four carbons), so a
    ring-count test would also return a nitrogen of a pyridine or a pyrazole substituent. They
    matter because the spin population of the radical anion sits on them, so a co-field peak is
    read by its distance from them.
    """
    match = mol.GetSubstructMatch(CORE)
    return [i for i in match if mol.GetAtomWithIdx(i).GetSymbol() == "N"]


def core_rmsd(mol: Chem.Mol, ref_xyz: np.ndarray) -> Optional[float]:
    """How far this molecule's core sits from the reference core, in angstrom.

    The minimum over every way the core can be matched. A phenazine core has symmetry, so
    ``GetSubstructMatch`` may map two aligned molecules' cores onto each other in a rotated or
    reflected order; comparing one arbitrary matching with another would report several angstrom
    for a pair that lies exactly on top of itself.
    """
    best = None
    xyz_all = mol.GetConformer().GetPositions()
    for match in mol.GetSubstructMatches(CORE, uniquify=False):
        if len(match) != len(ref_xyz):
            continue
        xyz = xyz_all[list(match)]
        val = float(np.sqrt(np.mean(np.sum((xyz - ref_xyz) ** 2, axis=1))))
        if best is None or val < best:
            best = val
    return best


def data_stage(mols, names, y, groups) -> Dict[str, object]:
    """What the series is: the target, the frame, and where the added electron sits."""
    out: Dict[str, object] = {"n": len(mols), "target": TARGET, "sdf": relative_path(SDF, ROOT)}
    out["target_range"] = [float(y.min()), float(y.max())]
    out["target_mean"] = float(y.mean())
    out["target_sd"] = float(y.std())
    out["mean_model_mae"] = float(np.mean(np.abs(y - y.mean())))
    out["groups"] = {g: int(np.sum(groups == g)) for g in sorted(set(groups.tolist()))}
    out["level"] = mols[0].GetProp("qm_level") if mols[0].HasProp("qm_level") else ""
    out["solvent"] = mols[0].GetProp("qm_solvent") if mols[0].HasProp("qm_solvent") else ""
    print(f"  {len(mols)} phenazines, {TARGET} from {y.min():+.3f} to {y.max():+.3f} V "
          f"(mean {y.mean():+.3f}, sd {y.std():.3f} V)", flush=True)
    print(f"  the mean of the training set alone would make an error of "
          f"{out['mean_model_mae']:.3f} V", flush=True)
    print(f"  source tables: " + ", ".join(f"{g} ({n})" for g, n in out["groups"].items()),
          flush=True)

    # The frame. Every molecule was superimposed on the core, so the core atoms of two molecules
    # should sit on top of each other; this is the number that says whether they do.
    ref = mols[int(np.argmin([m.GetNumAtoms() for m in mols]))]
    ref_match = ref.GetSubstructMatch(CORE)
    spreads = []
    if ref_match:
        ref_xyz = ref.GetConformer().GetPositions()[list(ref_match)]
        spreads = [v for v in (core_rmsd(m, ref_xyz) for m in mols) if v is not None]
    if spreads:
        out["core_rmsd"] = {"mean": float(np.mean(spreads)), "sd": float(np.std(spreads)),
                            "max": float(np.max(spreads)), "n": len(spreads)}
        print(f"  the core of {len(spreads)} molecules sits {np.mean(spreads):.2f} +/- "
              f"{np.std(spreads):.2f} A from the reference core (worst {np.max(spreads):.2f} A)",
              flush=True)

    # Where the added electron goes. A doublet has one unpaired electron, so the spin populations
    # of the radical anion sum to 1, and their share on the two core nitrogens says how much of
    # the electron the core holds rather than the substituents.
    shares, totals = [], []
    for m in mols:
        if not m.HasProp("atom.dprop.spin_anion"):
            continue
        s = np.array([a.GetDoubleProp("spin_anion") for a in m.GetAtoms()], float)
        totals.append(float(np.sum(s)))
        n_idx = core_nitrogens(m)
        if n_idx:
            shares.append(float(np.sum(s[n_idx]) / max(np.sum(np.abs(s)), 1e-12)))
    if totals:
        out["spin_total"] = {"mean": float(np.mean(totals)), "sd": float(np.std(totals))}
        out["core_nitrogen_share"] = {"mean": float(np.mean(shares)), "sd": float(np.std(shares))}
        print(f"  the spin populations of the radical anion sum to "
              f"{np.mean(totals):.4f} +/- {np.std(totals):.4f} (a doublet must give 1)", flush=True)
        print(f"  the two core nitrogens carry {np.mean(shares):.1%} +/- {np.std(shares):.1%} "
              f"of it", flush=True)

    # The controls, as plain correlations, before any model is fitted.
    out["correlation_with_target"] = {}
    for d in DESCRIPTORS:
        z = np.array([float(m.GetProp(d)) for m in mols if m.HasProp(d)], float)
        if len(z) == len(y):
            out["correlation_with_target"][d] = float(np.corrcoef(y, z)[0, 1])
    pairs = sorted(out["correlation_with_target"].items(), key=lambda kv: -abs(kv[1]))
    print("  correlation of the target with the global numbers: "
          + ", ".join(f"{k} {v:+.2f}" for k, v in pairs[:4]), flush=True)

    # How alike are two molecules of this series through one field alone? The kernel is a
    # cosine-normalised overlap, so a mean near 1 means the field is nearly the same shape in
    # every molecule and the kernel cannot tell two of them apart by it. This is the number that
    # explains the gate: a field has to vary across the series before it can predict anything.
    out["field_similarity"] = {}
    for label, physchem, props in (("electrostatic", ("electrostatic",), ()),
                                   ("steric", ("steric",), ()),
                                   ("q_hirshfeld", (), ("q_hirshfeld",)),
                                   ("fukui_plus", (), ("fukui_plus",)),
                                   ("fukui_minus", (), ("fukui_minus",)),
                                   ("dual", (), ("dual",)),
                                   ("spin_anion", (), ("spin_anion",))):
        fs = FieldSet.create(physchem=list(physchem), atom_properties=list(props), alpha=1.0,
                             charge_model="gasteiger", param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        cmf = CMFModel(fieldset=fs, method="krr", normalize=True)
        K = _unit_diagonal(KC.joint_training_kernel(cmf, AtomTable.from_molecules(mf)))
        off = K[np.triu_indices_from(K, 1)]
        out["field_similarity"][label] = {"mean": float(off.mean()), "sd": float(off.std()),
                                          "min": float(off.min())}
        print(f"  two molecules are {off.mean():.3f} +/- {off.std():.3f} alike through "
              f"{label} alone", flush=True)
    return out


# ------------------------------------------------------------------ the folds and the regression


def random_folds(n: int, k: int = OUTER, seed: int = SEED):
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    return [(np.setdiff1d(np.arange(n), g), g) for g in np.array_split(order, k)]


def grouped_folds(groups: np.ndarray):
    """One fold per source table, held out whole."""
    out = []
    for g in sorted(set(np.asarray(groups).tolist())):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        if len(te) and len(tr):
            out.append((tr, te))
    return out


def standardise(train: np.ndarray, other: Optional[np.ndarray] = None) -> np.ndarray:
    """`other` (or `train` itself) centred and scaled by the training columns."""
    mean, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    rows = train if other is None else other
    return (rows - mean) / sd


def _unit_diagonal(K: np.ndarray) -> np.ndarray:
    """The same kernel with a unit diagonal, so that two models are on one scale.

    The joint overlap kernel sums one normalised block per field, so its diagonal is the number
    of fields that carry weight -- five for `physchem` and nine for `quantum`. Without this, a
    model with more fields would be handed a larger kernel and a different effective ridge, and
    the comparison would be of two scales rather than of two descriptions.
    """
    d = np.sqrt(np.clip(np.diag(np.asarray(K, float)), 1e-12, None))
    return np.asarray(K, float) / np.outer(d, d)


def prepare(model: str, mols, names) -> object:
    """Everything a model needs, computed once for the whole series.

    A field model gives one full kernel per width; a descriptor model gives its raw matrix, which
    is standardised inside each fold on the training rows alone.
    """
    physchem, props, desc = MODELS[model]
    if desc:
        # tagged, rather than told apart by its shape: a descriptor matrix with as many columns
        # as molecules would otherwise be mistaken for a kernel
        return ("descriptors", np.vstack([read_values(m, list(desc)) for m in mols]))
    out = {}
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=list(physchem), atom_properties=list(props), alpha=alpha,
                             charge_model="gasteiger", param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        table = AtomTable.from_molecules(mf)
        cmf = CMFModel(fieldset=fs, method="krr", normalize=True)
        out[alpha] = _unit_diagonal(KC.joint_training_kernel(cmf, table))
    return out


def _kernel_at(entry, tr, te=None):
    """Training and test blocks of one prepared model."""
    if isinstance(entry, tuple) and entry[0] == "descriptors":
        Z = entry[1]
        A = standardise(Z[tr])
        K_tr = A @ A.T
        if te is None:
            return K_tr, None
        return K_tr, standardise(Z[tr], Z[te]) @ A.T
    K = np.asarray(entry, float)
    if te is None:
        return K[np.ix_(tr, tr)], None
    return K[np.ix_(tr, tr)], K[np.ix_(te, tr)]


def nested(entry, y: np.ndarray, design, label: str = "") -> Dict[str, object]:
    """Out-of-fold predictions, the ridge and the field width chosen inside each fold."""
    pred = np.full(len(y), np.nan)
    per_fold = []
    t0 = time.time()
    alphas = list(entry.keys()) if isinstance(entry, dict) else [0.0]
    for k, (tr, te) in enumerate(design):
        best = None
        for alpha in alphas:
            part = entry[alpha] if isinstance(entry, dict) else entry
            K_tr = _kernel_at(part, tr)[0]
            for lam in LAMS:
                p = cv_predict(K_tr, y[tr], "krr", {"lam": float(lam)}, folds=INNER, seed=SEED)
                err = rmse(y[tr], p)
                if best is None or err < best[0]:
                    best = (err, alpha, float(lam))
        _e, alpha, lam = best
        part = entry[alpha] if isinstance(entry, dict) else entry
        K_tr, K_te = _kernel_at(part, tr, te)
        learner = _Learner("krr", {"lam": lam}, True).fit(K_tr, y[tr])
        pred[te] = learner.predict(K_te)
        per_fold.append({"fold": k, "alpha": float(alpha), "lam": lam, "n_test": int(len(te)),
                         "mae": float(np.mean(np.abs(pred[te] - y[te]))),
                         "rmse": float(rmse(y[te], pred[te]))})
    maes = np.array([f["mae"] for f in per_fold])
    rec = {"n": int(len(y)), "mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=0)),
           "mae_pooled": float(np.mean(np.abs(pred - y))),
           "rmse_pooled": float(rmse(y, pred)), "r2_pooled": float(r2_score(y, pred)),
           "per_fold": per_fold, "abs_errors": [float(v) for v in np.abs(pred - y)],
           "elapsed_s": time.time() - t0}
    print(f"    {label:<14} MAE {rec['mae_mean']:.4f} +/- {rec['mae_sd']:.4f} V, "
          f"R2 {rec['r2_pooled']:+.3f}  ({rec['elapsed_s']:.0f} s)", flush=True)
    return rec


def paired_interval(a, b, seed: int = SEED, draws: int = 2000) -> Dict[str, object]:
    """The paired difference of two models' absolute errors, with a bootstrap interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(draws)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(d)), "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


#: the comparisons the item's gate is read from. "a vs b" is positive when b has the smaller
#: error, that is when b is the better model.
PAIRS = {
    "hirshfeld vs physchem": ("physchem", "hirshfeld"),
    "fukui vs physchem": ("physchem", "fukui"),
    "spin vs physchem": ("physchem", "spin"),
    "quantum vs physchem": ("physchem", "quantum"),
    "spin vs descriptors": ("descriptors", "spin"),
    "quantum vs descriptors": ("descriptors", "quantum"),
    "physchem vs descriptors": ("descriptors", "physchem"),
    "quantum vs ea": ("ea", "quantum"),
    "spin vs ea": ("ea", "spin"),
    "spin-only vs descriptors": ("descriptors", "spin-only"),
}


def gate_stage(mols, names, y, groups) -> Dict[str, object]:
    """Every model under both fold designs, then the paired comparisons."""
    designs = {"random": random_folds(len(y)),
               "one source table held out": grouped_folds(groups)}
    out: Dict[str, object] = {"target": TARGET, "models": list(MODEL_ORDER),
                              "outer": OUTER, "inner": INNER, "lams": list(LAMS),
                              "alphas": list(ALPHAS), "fields": list(FIELDS),
                              "descriptors": list(DESCRIPTORS), "scores": {}, "comparisons": {}}
    prepared = {}
    for model in MODEL_ORDER:
        t0 = time.time()
        prepared[model] = prepare(model, mols, names)
        print(f"  {model:<14} prepared in {time.time() - t0:.0f} s", flush=True)
    for split, design in designs.items():
        print(f"\n  --- {split} ({len(design)} folds)", flush=True)
        for model in MODEL_ORDER:
            out["scores"][f"{model}|{split}"] = nested(prepared[model], y, design, label=model)
        comp = {}
        for title, (worse, better) in PAIRS.items():
            a = out["scores"][f"{worse}|{split}"]["abs_errors"]
            b = out["scores"][f"{better}|{split}"]["abs_errors"]
            comp[title] = paired_interval(a, b)
            d = comp[title]
            mark = "yes" if d["excludes_zero"] and d["mean"] > 0 else "no"
            print(f"    {title:<26} {d['mean']:+.4f} V [{d['low']:+.4f}, {d['high']:+.4f}]  "
                  f"better: {mark}", flush=True)
        out["comparisons"][split] = comp
    return out


# ------------------------------------------------------------------------------- the co-field


def cofield_stage(mols, names, y) -> Dict[str, object]:
    """Where the co-field of the quantum model has its weight.

    The reading that makes a peak interpretable is its distance from the two core nitrogens, which
    is where the spin population of the radical anion sits. A peak on a nitrogen says the model
    uses the part of the electron the core holds; a peak out at the substituent positions says it
    uses what the substituents do to it.
    """
    alpha = ALPHAS[2]
    lam = 0.1
    physchem, props, _d = MODELS["quantum"]
    fs = FieldSet.create(physchem=list(physchem), atom_properties=list(props), alpha=alpha,
                         charge_model="gasteiger", param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": lam})
    model.fit(table, y, cv_folds=0)
    grid = make_grid(np.vstack([m.coords for m in mf]), margin=2.0, spacing=0.5)
    points = grid.points()

    # where the core nitrogens of this series sit, in the frame the series is aligned in
    nitro = []
    for m in mols:
        conf = m.GetConformer()
        for i in core_nitrogens(m):
            pos = conf.GetAtomPosition(i)
            nitro.append([pos.x, pos.y, pos.z])
    N = np.asarray(nitro, float)
    # the plane of the core, so that "above the ring" can be told from "in the ring"
    centre = N.mean(axis=0)
    out: Dict[str, object] = {"alpha": alpha, "lam": lam, "spacing": 0.5,
                              "n_core_nitrogens": int(len(N)),
                              "nitrogen_spread": float(np.mean(np.linalg.norm(N - centre,
                                                                              axis=1)))}
    print(f"    the {len(N)} core nitrogens sit "
          f"{out['nitrogen_spread']:.2f} A from their common centre", flush=True)
    rows = []
    for f, spec in enumerate(fs.specs):
        flat = np.asarray(cofield_volume(model, f, grid), float).ravel()
        k = int(np.argmax(np.abs(flat)))
        xyz = points[k]
        d_n = float(np.min(np.linalg.norm(N - xyz, axis=1)))
        d_c = float(np.linalg.norm(xyz - centre))
        rows.append({"field": spec.label, "peak": float(flat[k]),
                     "distance_from_nitrogen": d_n, "distance_from_centre": d_c,
                     "point": [float(v) for v in xyz]})
        print(f"    {spec.label:<16} largest magnitude {flat[k]:+.4f} at {d_n:.2f} A from the "
              f"nearest core nitrogen and {d_c:.2f} A from the core centre", flush=True)
    out["fields"] = rows
    return out


# ------------------------------------------------------------------------------- the table


def _read(base: str, name: str):
    path = os.path.join(base, f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def write_table(path: Optional[str] = None) -> str:
    """The table of every stage that has run, read back from its JSON."""
    base = path or RESULTS
    os.makedirs(base, exist_ok=True)
    data, gate, cofield = _read(base, "data"), _read(base, "gate"), _read(base, "cofield")
    lines = ["# Item EC1: quantum fields on a phenazine reduction-potential series", ""]
    if data:
        lines += [f"{data['n']} substituted phenazines with a computed one-electron reduction "
                  f"potential, from {data['target_range'][0]:+.3f} to "
                  f"{data['target_range'][1]:+.3f} V (sd {data['target_sd']:.3f} V). The mean of "
                  f"the training set alone makes an error of {data['mean_model_mae']:.3f} V.", ""]
        if data.get("level"):
            lines += [f"Quantum stage: `{data['level']}`"
                      + (f" with the continuum solvent `{data['solvent']}`"
                         if data.get("solvent") else "")
                      + ", at the geometry each source gives, never optimised.", ""]
        lines += ["| What | Value |", "|---|---|"]
        if "core_rmsd" in data:
            d = data["core_rmsd"]
            lines.append(f"| Core of one molecule against the reference core | "
                         f"{d['mean']:.2f} +/- {d['sd']:.2f} A |")
        if "spin_total" in data:
            d = data["spin_total"]
            lines.append(f"| Spin populations of the radical anion, summed | "
                         f"{d['mean']:.4f} +/- {d['sd']:.4f} (a doublet must give 1) |")
        if "core_nitrogen_share" in data:
            d = data["core_nitrogen_share"]
            lines.append(f"| Share of it on the two core nitrogens | "
                         f"{d['mean']:.1%} +/- {d['sd']:.1%} |")
        for d, r in sorted(data.get("correlation_with_target", {}).items(),
                           key=lambda kv: -abs(kv[1]))[:3]:
            lines.append(f"| Correlation of the target with `{d}` | {r:+.3f} |")
        lines.append("")
        if data.get("field_similarity"):
            lines += ["### How much does each field vary across the series?", "",
                      "The kernel is a cosine-normalised overlap, so this is what it sees. A mean "
                      "near 1 means the field is nearly the same shape in every molecule, and a "
                      "model cannot tell two of them apart by it.", "",
                      "| field alone | similarity of two molecules | smallest |",
                      "|---|---:|---:|"]
            for label, d in sorted(data["field_similarity"].items(),
                                   key=lambda kv: -kv[1]["mean"]):
                lines.append(f"| `{label}` | {d['mean']:.3f} +/- {d['sd']:.3f} | "
                             f"{d['min']:+.3f} |")
            lines.append("")
    if gate:
        splits = list(gate["comparisons"].keys())
        lines += ["## What each description predicts", "",
                  "Mean absolute error over the folds, in volts. Every model chooses its ridge "
                  "and its field width inside each fold.", "",
                  "| model | " + " | ".join(splits) + " |", "|---|" + "---:|" * len(splits)]
        for model in gate["models"]:
            cells = []
            for split in splits:
                s = gate["scores"].get(f"{model}|{split}")
                cells.append(f"{s['mae_mean']:.4f} +/- {s['mae_sd']:.4f}" if s else "-")
            lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
        lines += ["", "## The comparisons", "",
                  "The paired difference of the two models' per-molecule absolute errors, with a "
                  "95 % bootstrap interval. A positive mean means the second model is the better "
                  "one. An interval that holds zero is not a difference.", "",
                  "| comparison | " + " | ".join(splits) + " |", "|---|---|" * 1
                  + "---|" * (len(splits) - 1)]
        for title in PAIRS:
            cells = []
            for split in splits:
                d = gate["comparisons"][split].get(title)
                cells.append(f"{d['mean']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]"
                             if d else "-")
            lines.append(f"| {title} | " + " | ".join(cells) + " |")
        lines.append("")
    if cofield:
        lines += ["## The co-field of the quantum model", "",
                  "The point of largest magnitude of each field's co-field, and how far it is "
                  "from the nearest core nitrogen. The spin population of the radical anion sits "
                  "on those nitrogens, so that distance is what makes the position readable.", "",
                  "| field | peak | distance from a core nitrogen, A | from the core centre, A |",
                  "|---|---:|---:|---:|"]
        for r in cofield["fields"]:
            lines.append(f"| {r['field']} | {r['peak']:+.4f} | "
                         f"{r['distance_from_nitrogen']:.2f} | "
                         f"{r['distance_from_centre']:.2f} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(base, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"written {os.path.join(base, 'table.md')}")
    return text


STAGES = ("data", "gate", "cofield")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--table", action="store_true", help="rebuild the table and stop")
    ap.add_argument("--force", action="store_true", help="rerun a stage whose JSON exists")
    args = ap.parse_args(argv)
    os.makedirs(args.results, exist_ok=True)
    if args.table:
        write_table(args.results)
        return
    mols, names, y, groups = load()
    for stage in args.stage:
        out = os.path.join(args.results, f"{stage}.json")
        if os.path.exists(out) and not args.force:
            print(f"[skip] {out} exists")
            continue
        print(f"\n=== {stage}", flush=True)
        if stage == "data":
            res = data_stage(mols, names, y, groups)
        elif stage == "gate":
            res = gate_stage(mols, names, y, groups)
        else:
            res = cofield_stage(mols, names, y)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"  written {out}", flush=True)
    write_table(args.results)


if __name__ == "__main__":
    main()
