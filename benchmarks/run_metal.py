"""
Transition-metal complexes: the fields of a complex against what its composition already says.

Plan item EC4.  The item adds what a coordination complex needs -- van der Waals parameters for
the metals, a reader that writes the metal-ligand bonds as dative bonds, a charge model that does
not collapse when a metal is present, and the frame the metal and its first coordination sphere
give -- and then asks whether a field model of a complex predicts a redox potential.

Two series are measured, because they answer different halves of the question.

**Iron, 2216 complexes (`examples/complexes_fe/fe_redox.sdf`).**  Mononuclear iron(II) complexes
from the Cambridge Structural Database through the tmQM collection, with a computed Fe(III)/Fe(II)
reduction potential.  They share the metal and nothing else: 377 different first coordination
spheres and almost no two complexes with the same ligand set.  This series says whether the
machinery works at this size, and what a frame is worth when the members share nothing.

**Nickel, 25 complexes (`examples/complexes_ni/ni_bpp.sdf`).**  Substituted triaqua
nickel-bis(pyrazolyl)pyridine complexes with computed potentials of two couples, all with the
same N3O3 coordination sphere and the same ligand skeleton.  This is the congeneric series in a
shared frame that the item's gate names, and it comes with its own population analysis, which is
the "quantum descriptors alone" control.

The models
----------
``composition``  element counts, the number of atoms, the charge and the metal site: no geometry
                 at all.  This is the control that matters, because a redox potential of a metal
                 complex is largely decided by what is coordinated to the metal, and items BT1 to
                 BT3 already found a composition beating the fields on a battery voltage.
``site``         the first coordination sphere alone: how many donors, of which elements, at what
                 distances.  It says how much of the composition's score is the metal site.
``quantum``      (nickel only) the natural population analysis of the same calculation the target
                 comes from: the charge and the spin density on the metal, on the ligand, on the
                 substituents and on the water molecules.  A field that does not beat this has
                 added nothing to a handful of global numbers.
``invariant``    the rotation-invariant field autocorrelation, which needs no frame.
``overlap-raw``  the overlap kernel on the five fields in the frame the file happens to carry.
``overlap-rule`` the overlap kernel after every complex is put in the frame of its own metal and
                 principal axes.  Against ``overlap-raw`` it measures what the frame is worth.
``physchem``     (nickel only) the overlap kernel on the five fields in the frame the first
                 coordination spheres give, which is a real shared frame for that series.

Protocol
--------
Five outer folds, ten inner folds for the ridge and the field width, the same constants as the
other benchmarks of this repository, and a paired bootstrap interval on the per-sample absolute
errors.  Two fold designs are used for the iron series: five random folds, and five folds built
from the first coordination spheres so that no sphere in a test fold was seen in training.  The
nickel series is measured with five random folds and with leave-one-out, because 25 complexes are
too few for one design to be trusted.

    python benchmarks/run_metal.py --stage data
    python benchmarks/run_metal.py --stage iron
    python benchmarks/run_metal.py --stage nickel
    python benchmarks/run_metal.py --stage cofield
    python benchmarks/run_metal.py --report
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

from rdkit import Chem, RDLogger                                            # noqa: E402
from rdkit.Chem import AllChem                                              # noqa: E402

import kernel_cache as KC                                                   # noqa: E402

from cmfqsar.cofields import cofield_volume, make_grid                      # noqa: E402
from cmfqsar.complexes import (coordination, donor_indices,                 # noqa: E402
                               metal_index, principal_metal_frame)
from cmfqsar.descriptors import read_values                                 # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields                # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               feature_matrix)
from cmfqsar.model import CMFModel, _Learner, cv_predict, r2_score, rmse    # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "metal")
IRON_SDF = os.path.join(ROOT, "examples", "complexes_fe", "fe_redox.sdf")
NICKEL_SDF = os.path.join(ROOT, "examples", "complexes_ni", "ni_bpp.sdf")

SEED = 0
OUTER, INNER = 5, 10
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
ALPHAS = (0.3, 0.5, 1.0, 2.0)
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")

IRON_TARGET = "reduction_pot"
NICKEL_TARGETS = ("E_ox_AN", "E_red_AN")
#: the population analysis of the nickel series, which is the quantum control of its gate
NICKEL_QUANTUM = ("npa_charge_Ni", "npa_charge_BPP", "npa_charge_subst", "npa_charge_waters",
                  "npa_spin_Ni", "npa_spin_BPP", "npa_spin_waters", "s2_after")

IRON_MODELS = ("composition", "site", "invariant", "overlap-raw", "overlap-rule")
#: the models cheap enough to run on the whole iron series
IRON_MODELS_FULL = ("composition", "site", "invariant")
NICKEL_MODELS = ("quantum", "site", "invariant", "overlap-rule", "physchem")

#: how many complexes the overlap kernel is measured on.  The overlap kernel is a sum over every
#: pair of atoms of every pair of molecules, and its cost grows faster than the square of the
#: number of complexes: on this series one field width took 16 s at 800 complexes and 62 s at
#: 1200, so all 2216 cost minutes for each of the four widths and each of the two frames.  The
#: descriptor models and the frameless kernel are cheap, so they are measured on the whole series
#: as well, which shows whether the subset changed the answer.
IRON_SUBSET = 800

#: the comparisons each gate is read from.  "a vs b" is positive when b has the smaller error,
#: that is when b is the better model.
IRON_PAIRS = {
    "invariant vs composition": ("composition", "invariant"),
    "overlap-rule vs composition": ("composition", "overlap-rule"),
    "overlap-rule vs overlap-raw": ("overlap-raw", "overlap-rule"),
    "invariant vs overlap-rule": ("overlap-rule", "invariant"),
    "composition vs site": ("site", "composition"),
}
NICKEL_PAIRS = {
    "physchem vs quantum": ("quantum", "physchem"),
    "invariant vs quantum": ("quantum", "invariant"),
    "physchem vs site": ("site", "physchem"),
    "physchem vs invariant": ("invariant", "physchem"),
    "overlap-rule vs physchem": ("physchem", "overlap-rule"),
}


# ------------------------------------------------------------------------------- the data

def load(path: str, targets: Sequence[str]):
    """The complexes of one file, and one column per target."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing; build it with examples/make_metal_examples.py")
    mols, names = [], []
    for mol in Chem.SDMolSupplier(path, removeHs=False, sanitize=True):
        if mol is None or not all(mol.HasProp(t) for t in targets):
            continue
        mols.append(mol)
        names.append(mol.GetProp("_Name"))
    y = {t: np.array([float(m.GetProp(t)) for m in mols]) for t in targets}
    return mols, names, y


def element_columns(mols) -> List[str]:
    """Every element that appears anywhere in the series, in a fixed order."""
    seen = set()
    for m in mols:
        seen.update(a.GetSymbol() for a in m.GetAtoms())
    return sorted(seen)


def composition_matrix(mols, elements: Sequence[str], with_site: bool = True) -> np.ndarray:
    """Element counts, size, charge and, if asked, the metal site: no geometry of the ligands."""
    rows = []
    for m in mols:
        counts = {e: 0 for e in elements}
        for a in m.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
        row = [float(counts[e]) for e in elements]
        row += [float(m.GetNumAtoms()), float(Chem.GetFormalCharge(m))]
        if with_site:
            row += site_row(m, elements)
        rows.append(row)
    return np.asarray(rows, float)


def site_row(mol, elements: Sequence[str]) -> List[float]:
    """The first coordination sphere as numbers: how many donors, of which element, how far."""
    c = coordination(mol)
    row = [c.get("n_donors", 0.0), c.get("metal_charge", 0.0),
           c.get("d_min", 0.0), c.get("d_mean", 0.0), c.get("d_max", 0.0)]
    row += [c.get("donor_" + e, 0.0) for e in elements]
    return row


def site_matrix(mols, elements: Sequence[str]) -> np.ndarray:
    return np.asarray([site_row(m, elements) + [float(Chem.GetFormalCharge(m))] for m in mols],
                      float)


def framed(mols) -> Tuple[List[Chem.Mol], np.ndarray]:
    """Every complex in the frame of its own metal and principal axes, and the degeneracy gaps."""
    out, gaps = [], []
    for m in mols:
        mol, gap = principal_metal_frame(m)
        out.append(mol)
        gaps.append(gap)
    return out, np.asarray(gaps, float)


def would_lose_its_electrostatic_field(mol) -> bool:
    """Would the electrostatic field of this complex be zero without the repair of item EC4?

    The Gasteiger scheme has no parameters for a d-block metal, and one metal atom made it return
    "not a number" for every atom of the molecule.  This asks the scheme directly, so the number
    of complexes the repair rescues is measured and not asserted.
    """
    work = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(work)
    q = np.array([float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge")
                  else np.nan for a in work.GetAtoms()])
    return bool(not np.any(np.isfinite(q)))


def field_similarity(mols, names, fields=FIELDS) -> Dict[str, Dict[str, float]]:
    """How alike are two members of the series through one field alone?

    The overlap kernel is a cosine of two fields, so a mean near one means the field has nearly
    the same shape in every complex and cannot tell two of them apart.  This is the reading that
    said why the spin field of item EC1 could not work, and it is read here for the same reason.
    """
    out = {}
    for name in fields:
        fs = FieldSet.create(physchem=[name], alpha=1.0, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        cmf = CMFModel(fieldset=fs, method="krr", normalize=True)
        K = _unit_diagonal(KC.joint_training_kernel(cmf, AtomTable.from_molecules(mf)))
        off = K[np.triu_indices_from(K, 1)]
        out[name] = {"mean": float(off.mean()), "sd": float(off.std()), "min": float(off.min())}
    return out


def data_stage() -> Dict[str, object]:
    """What the two series hold, how good their frames are, and what the repair rescued."""
    out: Dict[str, object] = {}
    iron, iron_names, iron_y = load(IRON_SDF, [IRON_TARGET])
    y = iron_y[IRON_TARGET]
    dative = np.array([sum(1 for b in m.GetBonds()
                           if b.GetBondType() == Chem.BondType.DATIVE) for m in iron])
    spheres = [m.GetProp("donors") for m in iron]
    nd = np.array([len(donor_indices(m)) for m in iron])
    _f, gaps = framed(iron)
    lost = sum(would_lose_its_electrostatic_field(m) for m in iron[:200])
    out["iron"] = {
        "n": len(iron), "target": IRON_TARGET,
        "y_mean": float(y.mean()), "y_sd": float(y.std()),
        "y_min": float(y.min()), "y_max": float(y.max()),
        "atoms_mean": float(np.mean([m.GetNumAtoms() for m in iron])),
        "dative_mean": float(dative.mean()), "dative_min": int(dative.min()),
        "dative_max": int(dative.max()),
        "n_spheres": len(set(spheres)), "n_donors_median": float(np.median(nd)),
        "metal_charges": sorted({int(m.GetAtomWithIdx(metal_index(m)).GetFormalCharge())
                                 for m in iron}),
        "principal_gap_median": float(np.median(gaps)),
        "principal_gap_below_0.05": float(np.mean(gaps < 0.05)),
        "zero_electrostatic_without_repair": int(lost), "zero_checked": 200,
    }
    print(f"    iron:   {len(iron)} complexes, {out['iron']['dative_mean']:.1f} dative bonds "
          f"each, {out['iron']['n_spheres']} different first coordination spheres", flush=True)
    print(f"            target {y.mean():+.2f} +/- {y.std():.2f} V "
          f"({y.min():+.2f} to {y.max():+.2f})", flush=True)
    print(f"            without the charge repair of this item, {lost} of 200 complexes have an "
          f"electrostatic field that is identically zero", flush=True)
    print(f"            the rule frame is nearly degenerate (gap below 0.05) for "
          f"{100 * out['iron']['principal_gap_below_0.05']:.0f} % of them", flush=True)

    nickel, ni_names, ni_y = load(NICKEL_SDF, NICKEL_TARGETS)
    matched = np.array([float(m.GetProp("donor_rmsd")) for m in nickel
                        if m.HasProp("donor_rmsd")])
    rule, rule_gaps = framed(nickel)
    # how far does the rule frame put the donors from where the matched frame puts them?
    ref = nickel[0]
    ref_d = ref.GetConformer().GetPositions()[donor_indices(ref)]
    rule_rmsd = []
    for mol in rule:
        d = mol.GetConformer().GetPositions()[donor_indices(mol)]
        if len(d) == len(ref_d):
            rule_rmsd.append(float(np.sqrt(np.mean(np.sum((d - ref_d) ** 2, axis=1)))))
    out["nickel"] = {
        "n": len(nickel), "targets": list(NICKEL_TARGETS),
        "y": {t: {"mean": float(ni_y[t].mean()), "sd": float(ni_y[t].std()),
                  "min": float(ni_y[t].min()), "max": float(ni_y[t].max())}
              for t in NICKEL_TARGETS},
        "spheres": sorted({m.GetProp("donors") for m in nickel}),
        "donor_rmsd_matched_mean": float(matched.mean()), "donor_rmsd_matched_max":
            float(matched.max()),
        "donor_rmsd_rule_mean": float(np.mean(rule_rmsd)),
        "donor_rmsd_rule_max": float(np.max(rule_rmsd)),
        "principal_gap_median": float(np.median(rule_gaps)),
        "field_similarity": field_similarity(nickel, ni_names),
    }
    print(f"    nickel: {len(nickel)} complexes, all {out['nickel']['spheres'][0]}, donors "
          f"matched to {matched.mean():.2f} A (worst {matched.max():.2f})", flush=True)
    for t in NICKEL_TARGETS:
        s = out["nickel"]["y"][t]
        print(f"            {t}: {s['mean']:+.2f} +/- {s['sd']:.2f} V", flush=True)
    print(f"            the rule frame puts the same donors {np.mean(rule_rmsd):.2f} A from the "
          f"matched frame (worst {np.max(rule_rmsd):.2f}), so on this series the rule is "
          f"{'as good as' if np.mean(rule_rmsd) < 0.5 else 'not'} the match", flush=True)
    for name, s in out["nickel"]["field_similarity"].items():
        print(f"            two complexes are {s['mean']:.3f} +/- {s['sd']:.3f} alike through "
              f"the {name} field alone", flush=True)
    return out


# ------------------------------------------------------------------------------- the protocol

def random_folds(n: int, k: int = OUTER, seed: int = SEED):
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    return [(np.setdiff1d(np.arange(n), g), g) for g in np.array_split(order, k)]


def leave_one_out(n: int):
    return [(np.setdiff1d(np.arange(n), [i]), np.array([i])) for i in range(n)]


def sphere_folds(spheres: Sequence[str], k: int = OUTER, seed: int = SEED):
    """Folds built from the first coordination spheres, each sphere whole in one fold.

    A complex in a test fold then has a coordination sphere that training never saw, which is the
    leave-one-chemotype-out design the item's gate asks for: the easy way to predict a redox
    potential is to recognise the coordination, and this design takes that away.
    """
    uniq = sorted(set(spheres))
    order = np.arange(len(uniq))
    np.random.RandomState(seed).shuffle(order)
    where = {}
    for fold, ids in enumerate(np.array_split(order, k)):
        for i in ids:
            where[uniq[i]] = fold
    g = np.array([where[s] for s in spheres])
    return [(np.where(g != f)[0], np.where(g == f)[0]) for f in range(k)
            if np.any(g == f) and np.any(g != f)]


def standardise(train: np.ndarray, other: Optional[np.ndarray] = None) -> np.ndarray:
    mean, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    rows = train if other is None else other
    return (rows - mean) / sd


def _unit_diagonal(K: np.ndarray) -> np.ndarray:
    """The same kernel with a unit diagonal, so that two models are on one scale."""
    d = np.sqrt(np.clip(np.diag(np.asarray(K, float)), 1e-12, None))
    return np.asarray(K, float) / np.outer(d, d)


def overlap_kernels(mols, names) -> Dict[float, np.ndarray]:
    out = {}
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=list(FIELDS), alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        cmf = CMFModel(fieldset=fs, method="krr", normalize=True)
        out[alpha] = _unit_diagonal(KC.joint_training_kernel(cmf, AtomTable.from_molecules(mf)))
    return out


def invariant_kernels(mols, names) -> Dict[float, np.ndarray]:
    out = {}
    spec = AutocorrelationSpec()
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=list(FIELDS), alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        out[alpha] = autocorrelation_kernel(feature_matrix(mf, spec), n_bins=spec.n_bins)
    return out


def prepare(model: str, mols, names, elements) -> object:
    """Everything one model needs, computed once for the whole series."""
    if model == "composition":
        return ("descriptors", composition_matrix(mols, elements))
    if model == "site":
        return ("descriptors", site_matrix(mols, elements))
    if model == "quantum":
        return ("descriptors", np.vstack([read_values(m, list(NICKEL_QUANTUM)) for m in mols]))
    if model == "invariant":
        return invariant_kernels(mols, names)
    if model == "overlap-raw" or model == "physchem":
        return overlap_kernels(mols, names)
    if model == "overlap-rule":
        return overlap_kernels(framed(mols)[0], names)
    raise ValueError(f"unknown model {model!r}")


def _kernel_at(entry, tr, te=None):
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


def nested(entry, y: np.ndarray, design, label: str = "", unit: str = "V") -> Dict[str, object]:
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
                p = cv_predict(K_tr, y[tr], "krr", {"lam": float(lam)},
                               folds=min(INNER, len(tr)), seed=SEED)
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
    print(f"    {label:<22} MAE {rec['mae_mean']:.4f} +/- {rec['mae_sd']:.4f} {unit}, "
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


def _baseline(y: np.ndarray, design) -> float:
    """The error of predicting the training mean, which every model has to beat."""
    err = []
    for tr, te in design:
        err.append(np.mean(np.abs(y[te] - y[tr].mean())))
    return float(np.mean(err))


def measure(mols, names, y, designs, models, pairs, unit="V") -> Dict[str, object]:
    elements = element_columns(mols)
    prepared = {}
    for model in models:
        t0 = time.time()
        prepared[model] = prepare(model, mols, names, elements)
        print(f"    prepared {model} ({time.time() - t0:.0f} s)", flush=True)
    out: Dict[str, object] = {"models": list(models), "elements": elements, "designs": {}}
    for dname, design in designs.items():
        print(f"  [{dname}] {len(design)} folds, baseline MAE "
              f"{_baseline(y, design):.4f} {unit}", flush=True)
        rec = {"baseline_mae": _baseline(y, design), "n_folds": len(design), "models": {}}
        for model in models:
            rec["models"][model] = nested(prepared[model], y, design, f"{dname}/{model}", unit)
        rec["pairs"] = {}
        for label, (a, b) in pairs.items():
            rec["pairs"][label] = paired_interval(rec["models"][a]["abs_errors"],
                                                  rec["models"][b]["abs_errors"])
            p = rec["pairs"][label]
            print(f"    {label:<30} {p['mean']:+.4f} {unit} "
                  f"[{p['low']:+.4f}, {p['high']:+.4f}]"
                  f"{'  (excludes zero)' if p['excludes_zero'] else ''}", flush=True)
        out["designs"][dname] = rec
    return out


def iron_stage() -> Dict[str, object]:
    mols, names, ys = load(IRON_SDF, [IRON_TARGET])
    y = ys[IRON_TARGET]
    spheres = [m.GetProp("donors") for m in mols]
    out: Dict[str, object] = {"n": len(y), "target": IRON_TARGET, "subset_n": IRON_SUBSET}
    print(f"  === all {len(y)} complexes, the models that cost little", flush=True)
    designs = {"random": random_folds(len(y)), "unseen-sphere": sphere_folds(spheres)}
    pairs_full = {k: v for k, v in IRON_PAIRS.items()
                  if all(m in IRON_MODELS_FULL for m in v)}
    out["full"] = measure(mols, names, y, designs, IRON_MODELS_FULL, pairs_full)
    idx = np.sort(np.random.RandomState(SEED).choice(len(y), size=min(IRON_SUBSET, len(y)),
                                                     replace=False))
    sub = [mols[i] for i in idx]
    sub_names = [names[i] for i in idx]
    sub_spheres = [spheres[i] for i in idx]
    print(f"  === a random {len(idx)} of them, every model including the overlap kernel",
          flush=True)
    sub_designs = {"random": random_folds(len(idx)), "unseen-sphere": sphere_folds(sub_spheres)}
    out["subset"] = measure(sub, sub_names, y[idx], sub_designs, IRON_MODELS, IRON_PAIRS)
    out["subset"]["indices"] = [int(i) for i in idx]
    return out


def nickel_stage() -> Dict[str, object]:
    mols, names, ys = load(NICKEL_SDF, NICKEL_TARGETS)
    out: Dict[str, object] = {"n": len(mols), "targets": list(NICKEL_TARGETS)}
    for target in NICKEL_TARGETS:
        y = ys[target]
        print(f"  === {target}", flush=True)
        designs = {"random": random_folds(len(y)), "leave-one-out": leave_one_out(len(y))}
        out[target] = measure(mols, names, y, designs, NICKEL_MODELS, NICKEL_PAIRS)
    return out


# ------------------------------------------------------------------------------- the co-field

def cofield_stage() -> Dict[str, object]:
    """Where the co-field of the nickel series has its weight, measured from the metal.

    The reading that makes a peak interpretable here is its distance from the nickel atom and
    from the plane of the three nitrogen donors.  A peak at the metal says the model uses the
    metal site, which every member shares; a peak out beyond the ligand says it uses what the
    substituents do, which is what differs between the members.
    """
    mols, names, ys = load(NICKEL_SDF, NICKEL_TARGETS)
    alpha = ALPHAS[2]
    lam = 0.1
    out: Dict[str, object] = {"alpha": alpha, "lam": lam, "spacing": 0.5}
    fs = FieldSet.create(physchem=list(FIELDS), alpha=alpha, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    grid = make_grid(np.vstack([m.coords for m in mf]), margin=2.0, spacing=0.5)
    points = grid.points()
    # the series is aligned on the metal and its donors, so the metal is one point for all of them
    ref = mols[0]
    xyz0 = ref.GetConformer().GetPositions()
    metal = xyz0[metal_index(ref)]
    donors = donor_indices(ref)
    nitrogen = np.array([xyz0[i] for i in donors
                         if ref.GetAtomWithIdx(i).GetSymbol() == "N"], float)
    oxygen = np.array([xyz0[i] for i in donors
                       if ref.GetAtomWithIdx(i).GetSymbol() == "O"], float)
    out["metal_spread"] = float(np.mean([np.linalg.norm(
        m.GetConformer().GetPositions()[metal_index(m)] - metal) for m in mols]))
    print(f"    the metal of every complex sits {out['metal_spread']:.3f} A from the "
          f"reference metal", flush=True)
    for target in NICKEL_TARGETS:
        model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": lam})
        model.fit(table, ys[target], cv_folds=0)
        rows = []
        for f, spec in enumerate(fs.specs):
            flat = np.asarray(cofield_volume(model, f, grid), float).ravel()
            k = int(np.argmax(np.abs(flat)))
            p = points[k]
            rows.append({"field": spec.label, "peak": float(flat[k]),
                         "distance_from_metal": float(np.linalg.norm(p - metal)),
                         "distance_from_nearest_nitrogen":
                             float(np.min(np.linalg.norm(nitrogen - p, axis=1))),
                         "distance_from_nearest_oxygen":
                             float(np.min(np.linalg.norm(oxygen - p, axis=1))),
                         "point": [float(v) for v in p]})
            r = rows[-1]
            print(f"    {target} {spec.label:<16} largest magnitude {r['peak']:+.4f} at "
                  f"{r['distance_from_metal']:.2f} A from the metal, "
                  f"{r['distance_from_nearest_nitrogen']:.2f} A from the nearest nitrogen "
                  f"donor and {r['distance_from_nearest_oxygen']:.2f} A from the nearest "
                  f"water oxygen", flush=True)
        out[target] = rows
    return out


# ------------------------------------------------------------------------------- the table

def _read(base: str, name: str):
    path = os.path.join(base, f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def write_table(path: Optional[str] = None) -> str:
    """The table of every stage that has run, read back from its JSON.

    `path` is the directory the stages wrote into, as in the other benchmarks of this
    repository; the table is written beside them as ``table.md``.
    """
    base = path or RESULTS
    os.makedirs(base, exist_ok=True)
    data, iron = _read(base, "data"), _read(base, "iron")
    nickel, cofield = _read(base, "nickel"), _read(base, "cofield")
    out: List[str] = ["# Transition-metal complexes: the fields against the composition", "",
                      "Plan item EC4. Two series of coordination complexes with computed redox",
                      "potentials: 2216 iron(II) complexes from the structural database, which",
                      "share the metal and nothing else, and 25 substituted nickel complexes,",
                      "which share one ligand skeleton and one coordination sphere. Errors are",
                      "mean absolute errors in volts over five outer folds, with the ridge and",
                      "the field width chosen inside each fold.", ""]
    if data:
        d = data["iron"]
        out += ["## What the series hold", "",
                f"The iron series holds {d['n']} complexes with {d['dative_mean']:.1f} dative",
                f"bonds each and {d['n_spheres']} different first coordination spheres; its",
                f"target is {d['y_mean']:+.2f} +/- {d['y_sd']:.2f} V",
                f"({d['y_min']:+.2f} to {d['y_max']:+.2f}).",
                f"Without the charge repair of this item,",
                f"{d['zero_electrostatic_without_repair']} of {d['zero_checked']} complexes",
                "checked have an electrostatic field that is identically zero, because one metal",
                "atom makes the Gasteiger scheme return nothing for every atom of the molecule.",
                ""]
        n = data["nickel"]
        out += [f"The nickel series holds {n['n']} complexes, all with the coordination sphere",
                f"{n['spheres'][0]}. Matching each first coordination sphere onto the first",
                f"complex's puts the donors {n['donor_rmsd_matched_mean']:.2f} A apart on",
                f"average (worst {n['donor_rmsd_matched_max']:.2f} A), so the series has a real",
                "shared frame. The rule frame of the principal axes puts the same donors",
                f"{n['donor_rmsd_rule_mean']:.2f} A from the matched frame",
                f"(worst {n['donor_rmsd_rule_max']:.2f} A).", "",
                "| field | mean similarity of two complexes | standard deviation | smallest |",
                "|---|---|---|---|"]
        for name, s in n["field_similarity"].items():
            out.append(f"| {name} | {s['mean']:.3f} | {s['sd']:.3f} | {s['min']:.3f} |")
        out += ["", "A mean near one means the field has nearly the same shape in every complex",
                "and cannot separate two of them.", ""]
    for title, part in (("## Iron: every complex of the series", "full"),
                        ("## Iron: a random subset, with the overlap kernel", "subset")):
        if not iron or part not in iron:
            continue
        block = iron[part]
        out += [title, "",
                f"{block['designs']['random']['models'][block['models'][0]]['n']} complexes. "
                + ("The overlap kernel is measured on this random subset and not on the whole "
                   "series, because its cost grows faster than the square of the number of "
                   "complexes: one field width took 16 s at 800 complexes and 62 s at 1200."
                   if part == "subset" else
                   "Only the models that cost little are measured on the whole series; the "
                   "overlap kernel is measured on the subset below."),
                ""]
        for dname, rec in block["designs"].items():
            out += [f"### {dname} folds (training-mean error {rec['baseline_mae']:.3f} V)", "",
                    "| model | MAE (V) | standard deviation | R2 |", "|---|---|---|---|"]
            for model in block["models"]:
                m = rec["models"][model]
                out.append(f"| {model} | {m['mae_mean']:.4f} | {m['mae_sd']:.4f} | "
                           f"{m['r2_pooled']:+.3f} |")
            out += ["", "| comparison | difference (V) | 95 % interval |", "|---|---|---|"]
            for label, p in rec["pairs"].items():
                out.append(f"| {label} | {p['mean']:+.4f} | "
                           f"[{p['low']:+.4f}, {p['high']:+.4f}] |")
            out += [""]
    if nickel:
        out += ["## Nickel: 25 complexes in a shared frame", ""]
        for target in nickel["targets"]:
            block = nickel[target]
            for dname, rec in block["designs"].items():
                out += [f"### {target}, {dname} "
                        f"(training-mean error {rec['baseline_mae']:.3f} V)", "",
                        "| model | MAE (V) | standard deviation | R2 |", "|---|---|---|---|"]
                for model in block["models"]:
                    m = rec["models"][model]
                    out.append(f"| {model} | {m['mae_mean']:.4f} | {m['mae_sd']:.4f} | "
                               f"{m['r2_pooled']:+.3f} |")
                out += ["", "| comparison | difference (V) | 95 % interval |", "|---|---|---|"]
                for label in NICKEL_PAIRS:
                    p = rec["pairs"][label]
                    out.append(f"| {label} | {p['mean']:+.4f} | "
                               f"[{p['low']:+.4f}, {p['high']:+.4f}] |")
                out += [""]
    if cofield:
        out += ["## Where the co-field of the nickel series has its weight", "",
                "| target | field | peak | from the metal (A) | from the nearest N donor (A) "
                "| from the nearest water O (A) |", "|---|---|---|---|---|---|"]
        for target in NICKEL_TARGETS:
            for row in cofield.get(target, []):
                out.append(f"| {target} | {row['field']} | {row['peak']:+.4f} | "
                           f"{row['distance_from_metal']:.2f} | "
                           f"{row['distance_from_nearest_nitrogen']:.2f} | "
                           f"{row['distance_from_nearest_oxygen']:.2f} |")
        out += [""]
    text = "\n".join(out) + "\n"
    dest = os.path.join(base, "table.md")
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"written {dest}")
    return text


STAGES = ("data", "iron", "nickel", "cofield")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", choices=STAGES + ("all",), default="all")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        write_table()
        return
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"[{stage}]", flush=True)
        t0 = time.time()
        res = {"data": data_stage, "iron": iron_stage, "nickel": nickel_stage,
               "cofield": cofield_stage}[stage]()
        res["seconds"] = time.time() - t0
        with open(os.path.join(RESULTS, f"{stage}.json"), "w", encoding="utf-8",
                  newline="\n") as fh:
            json.dump(res, fh, indent=1)
        print(f"  {stage} done in {res['seconds']:.0f} s", flush=True)
    write_table()


if __name__ == "__main__":
    main()
