"""
Fields read at a few anchored points, against the field model that reads them everywhere (EC3).

A kernel compares two fields over all space.  Sometimes the question is about one place: what the
reacting group of a Hammett series feels where it sits, what a metal centre feels from the second
coordination sphere.  A series in a shared frame has such places, and the field and its gradient
there are ordinary numbers, so plan item EC3 puts them in as descriptors.  For the electrostatic
field the gradient is the oriented electric field of the electrostatic-catalysis literature.

Two sources of the numbers are measured, because they are different descriptions of the same
charges.  ``field`` reads the Gaussian field of the CMF field set, which is what the kernel is
built from but falls off much faster than 1/r.  ``coulomb`` reads the true point-charge potential
and its gradient, in e/angstrom and e/angstrom^2, which is the physical quantity -- and the one
the module's own measurement says the Gaussian form represents badly at long range.

The two series are the ones with a frame by rule:

* **The Hammett sets** (95 substituted benzenes, `examples/hammett/hammett.sdf`), ring centroid at
  the origin, ipso carbon on +x, ring in the z = 0 plane.  The anchored points are the ring
  positions themselves and, 1.5 angstrom beyond the meta and para carbons, the places where the
  reacting group of the sigma definition sits.
* **The nickel complexes of item EC4** (25 complexes, `examples/complexes_ni/ni_bpp.sdf`),
  aligned on the metal and its first coordination sphere, so the metal is at the origin for every
  member.  The anchored points are the metal itself and the two places where item EC5's vector
  co-field put its strongest arrows, which are the two substituent positions of the ligand.

The models
----------
``physchem``        the five ordinary fields: the field model this item is measured against
``electrostatic``   that field alone
``anchors-near``    the points at the reaction centre (Hammett: beyond the meta and para carbons;
                    nickel: the metal), value and the three gradient components
``anchors-far``     the points out where the substituents are
``anchors-all``     both sets of points
``coulomb-all``     the same points read as a true point-charge potential and field
``physchem+anchors-all``  the field kernel plus the anchored block, each scaled to a mean
                    diagonal of one, which is how a descriptor block enters a CMF model

    python benchmarks/run_anchors.py --stage check
    python benchmarks/run_anchors.py --stage hammett
    python benchmarks/run_anchors.py --stage nickel
    python benchmarks/run_anchors.py --report
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

import kernel_cache as KC                                                   # noqa: E402

from cmfqsar.descriptors import (anchor_names, anchored_matrix,             # noqa: E402
                                 field_at_points, field_gradient_at_points)
from cmfqsar.fields import AtomTable, FieldSet, build_fields                # noqa: E402
from cmfqsar.model import CMFModel, _Learner, cv_predict, r2_score, rmse    # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "anchors")
HAMMETT_SDF = os.path.join(ROOT, "examples", "hammett", "hammett.sdf")
NICKEL_SDF = os.path.join(ROOT, "examples", "complexes_ni", "ni_bpp.sdf")

SEED = 0
OUTER, INNER = 5, 10
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
ALPHAS = (0.3, 0.5, 1.0, 2.0)
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")
KINDS = ("value", "gx", "gy", "gz")

HAMMETT_TARGETS = ("sigma_m", "sigma_p", "pi")
NICKEL_TARGETS = ("E_ox_AN", "E_red_AN")

#: The anchored points of the Hammett frame.  The ring is a regular hexagon of radius 1.395
#: angstrom in the z = 0 plane with the ipso carbon on +x, so every position is exact rather than
#: read off a molecule.  The two "site" points sit 1.5 angstrom beyond their ring carbon along the
#: same radius, which is about where the carboxyl carbon of a substituted benzoic acid is: that is
#: the place the sigma constants are defined at.
_R = 1.395
_SITE = _R + 1.5
HAMMETT_NEAR = {
    "meta_site": (-0.5 * _SITE, 0.8660254 * _SITE, 0.0),
    "para_site": (-_SITE, 0.0, 0.0),
}
HAMMETT_FAR = {
    "ipso": (_R, 0.0, 0.0),
    "centroid": (0.0, 0.0, 0.0),
    "face": (0.0, 0.0, 1.7),
}

#: The anchored points of the nickel series.  The metal is at the origin because the series is
#: aligned on it; the two substituent points are where item EC5's strongest co-field arrows were,
#: which is the measurement that says where this series differs.
NICKEL_NEAR = {"metal": (0.0, 0.0, 0.0)}
NICKEL_FAR = {
    "substituent_left": (-6.076, 1.390, -0.230),
    "substituent_right": (6.424, 1.390, -0.230),
}

MODEL_ORDER = ("electrostatic", "physchem", "anchors-near", "anchors-far", "anchors-all",
               "coulomb-all", "physchem+anchors-all")

#: the comparisons the gate is read from.  "a vs b" is positive when b has the smaller error.
PAIRS = {
    "anchors-all vs physchem": ("physchem", "anchors-all"),
    "anchors-near vs physchem": ("physchem", "anchors-near"),
    "coulomb-all vs anchors-all": ("anchors-all", "coulomb-all"),
    "physchem+anchors-all vs physchem": ("physchem", "physchem+anchors-all"),
    "anchors-all vs electrostatic": ("electrostatic", "anchors-all"),
}


# ------------------------------------------------------------------------------- the data

def load(path: str, targets: Sequence[str], name_prop: Optional[str] = None):
    mols, names = [], []
    for k, mol in enumerate(Chem.SDMolSupplier(path, removeHs=False, sanitize=True)):
        if mol is None:
            continue
        mols.append(mol)
        if name_prop and mol.HasProp(name_prop):
            names.append(mol.GetProp(name_prop))
        else:
            names.append(mol.GetProp("_Name") if mol.HasProp("_Name")
                         and mol.GetProp("_Name").strip() else f"mol_{k + 1}")
    cols = {t: (np.array([float(m.GetProp(t)) if m.HasProp(t) else np.nan for m in mols]),
                np.array([m.HasProp(t) for m in mols])) for t in targets}
    return mols, names, cols


def points_of(which: str, near: Dict[str, Tuple[float, float, float]],
              far: Dict[str, Tuple[float, float, float]]):
    table = {"near": near, "far": far, "all": {**near, **far}}[which]
    return list(table), np.asarray([table[k] for k in table], float)


# ------------------------------------------------------------------------------- the check

def check_stage() -> Dict[str, object]:
    """Does an anchored reading vary across the series, and does it track the target?

    Item EC5 learned to ask this before fitting anything: a descriptor that is nearly the same in
    every member cannot separate two of them, however physical it is.  Here the spread of every
    column and its correlation with each target are reported, for both sources, so that the models
    below can be read against what the columns can possibly carry.
    """
    out: Dict[str, object] = {}
    for tag, path, name_prop, near, far, targets in (
            ("hammett", HAMMETT_SDF, "substituent", HAMMETT_NEAR, HAMMETT_FAR, HAMMETT_TARGETS),
            ("nickel", NICKEL_SDF, None, NICKEL_NEAR, NICKEL_FAR, NICKEL_TARGETS)):
        mols, names, cols = load(path, targets, name_prop)
        labels, pts = points_of("all", near, far)
        fs = FieldSet.create(physchem=list(FIELDS), alpha=1.0, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        rec: Dict[str, object] = {"n": len(mols), "labels": labels,
                                  "points": [[float(c) for c in p] for p in pts]}
        # the gradient of a field is the derivative of its value: checked here on real data, at
        # every anchored point of the series, so the benchmark carries the check too
        worst = 0.0
        step = 1e-5
        for f in range(len(fs.specs)):
            for p in pts:
                grad = field_gradient_at_points(p[None, :], mf[0], fs.specs[f], f)[0]
                for axis in range(3):
                    shift = np.zeros(3)
                    shift[axis] = step
                    hi = field_at_points((p + shift)[None, :], mf[0], fs.specs[f], f)[0]
                    lo = field_at_points((p - shift)[None, :], mf[0], fs.specs[f], f)[0]
                    worst = max(worst, abs(grad[axis] - (hi - lo) / (2 * step)))
        rec["worst_gradient_error"] = float(worst)
        print(f"    {tag}: the gradient matches a finite difference to {worst:.2e} at every "
              f"anchored point of every field", flush=True)
        for source in ("field", "coulomb"):
            field_names = ["coulomb"] if source == "coulomb" else list(FIELDS)
            Z = anchored_matrix(mf, fs, pts, KINDS, source=source)
            columns = anchor_names(field_names, labels, KINDS)
            spread = Z.std(axis=0)
            table = []
            for t in targets:
                y, have = cols[t]
                keep = np.where(have)[0]
                r = np.zeros(Z.shape[1])
                for k in range(Z.shape[1]):
                    col = Z[keep, k]
                    if col.std() > 1e-12:
                        r[k] = abs(np.corrcoef(col, y[keep])[0, 1])
                best = int(np.argmax(r))
                table.append({"target": t, "best_column": columns[best],
                              "best_abs_r": float(r[best]),
                              "mean_abs_r": float(r.mean()),
                              "columns_above_0.5": int((r > 0.5).sum())})
                print(f"    {tag}/{source}: {t} is best tracked by {columns[best]} "
                      f"(|r| {r[best]:.3f}); {int((r > 0.5).sum())} of {len(columns)} columns "
                      f"reach |r| 0.5", flush=True)
            rec[source] = {"n_columns": int(Z.shape[1]), "targets": table,
                           "dead_columns": int((spread <= 1e-12).sum()),
                           "spread_median": float(np.median(spread))}
        out[tag] = rec
    return out


# ------------------------------------------------------------------------------- the protocol

def random_folds(n: int, k: int = OUTER, seed: int = SEED):
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    return [(np.setdiff1d(np.arange(n), g), g) for g in np.array_split(order, k)]


def leave_one_out(n: int):
    return [(np.setdiff1d(np.arange(n), [i]), np.array([i])) for i in range(n)]


def standardise(train: np.ndarray, other: Optional[np.ndarray] = None) -> np.ndarray:
    mean, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    rows = train if other is None else other
    return (rows - mean) / sd


def _unit_diagonal(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(np.asarray(K, float)), 1e-12, None))
    return np.asarray(K, float) / np.outer(d, d)


def field_kernels(mols, names, fields) -> Dict[float, np.ndarray]:
    out = {}
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=list(fields), alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        cmf = CMFModel(fieldset=fs, method="krr", normalize=True)
        out[alpha] = _unit_diagonal(KC.joint_training_kernel(cmf, AtomTable.from_molecules(mf)))
    return out


def anchor_block(mols, names, which: str, near, far, source: str = "field") -> np.ndarray:
    fs = FieldSet.create(physchem=list(FIELDS), alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    _labels, pts = points_of(which, near, far)
    return anchored_matrix(mf, fs, pts, KINDS, source=source)


def prepare(model: str, mols, names, near, far) -> object:
    if model == "electrostatic":
        return field_kernels(mols, names, ("electrostatic",))
    if model == "physchem":
        return field_kernels(mols, names, FIELDS)
    if model.startswith("anchors-"):
        return ("descriptors", anchor_block(mols, names, model.split("-", 1)[1], near, far))
    if model.startswith("coulomb-"):
        return ("descriptors", anchor_block(mols, names, model.split("-", 1)[1], near, far,
                                            source="coulomb"))
    if model == "physchem+anchors-all":
        return ("combined", field_kernels(mols, names, FIELDS),
                anchor_block(mols, names, "all", near, far))
    raise ValueError(f"unknown model {model!r}")


def _descriptor_kernel(Z: np.ndarray, tr, te=None):
    """The linear kernel of a descriptor block, scaled to a mean training diagonal of one.

    Scaled and not cosine-normalised, which is what :mod:`cmfqsar.descriptors` explains: dividing
    by the two norms would send a molecule that sits at the training mean in every column to a
    zero row, that is out of the model altogether.
    """
    A = standardise(Z[tr])
    K_tr = A @ A.T
    scale = float(np.mean(np.diag(K_tr)))
    scale = scale if scale > 1e-12 else 1.0
    if te is None:
        return K_tr / scale, None
    return K_tr / scale, (standardise(Z[tr], Z[te]) @ A.T) / scale


def _kernel_at(entry, tr, te=None, alpha=None):
    if isinstance(entry, tuple) and entry[0] == "descriptors":
        return _descriptor_kernel(entry[1], tr, te)
    if isinstance(entry, tuple) and entry[0] == "combined":
        K = np.asarray(entry[1][alpha], float)
        K_tr, K_te = _descriptor_kernel(entry[2], tr, te)
        if te is None:
            return K[np.ix_(tr, tr)] + K_tr, None
        return K[np.ix_(tr, tr)] + K_tr, K[np.ix_(te, tr)] + K_te
    K = np.asarray(entry, float)
    if te is None:
        return K[np.ix_(tr, tr)], None
    return K[np.ix_(tr, tr)], K[np.ix_(te, tr)]


def _alphas_of(entry):
    if isinstance(entry, dict):
        return list(entry)
    if isinstance(entry, tuple) and entry[0] == "combined":
        return list(entry[1])
    return [None]


def _part(entry, alpha):
    if isinstance(entry, dict):
        return entry[alpha]
    return entry


def nested(entry, y: np.ndarray, design, label: str, unit: str) -> Dict[str, object]:
    pred = np.full(len(y), np.nan)
    per_fold = []
    t0 = time.time()
    for k, (tr, te) in enumerate(design):
        best = None
        for alpha in _alphas_of(entry):
            K_tr = _kernel_at(_part(entry, alpha), tr, alpha=alpha)[0]
            for lam in LAMS:
                p = cv_predict(K_tr, y[tr], "krr", {"lam": float(lam)},
                               folds=min(INNER, len(tr)), seed=SEED)
                err = rmse(y[tr], p)
                if best is None or err < best[0]:
                    best = (err, alpha, float(lam))
        _e, alpha, lam = best
        K_tr, K_te = _kernel_at(_part(entry, alpha), tr, te, alpha=alpha)
        learner = _Learner("krr", {"lam": lam}, True).fit(K_tr, y[tr])
        pred[te] = learner.predict(K_te)
        per_fold.append({"fold": k, "alpha": None if alpha is None else float(alpha),
                         "lam": lam, "n_test": int(len(te)),
                         "mae": float(np.mean(np.abs(pred[te] - y[te])))})
    maes = np.array([f["mae"] for f in per_fold])
    rec = {"n": int(len(y)), "mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=0)),
           "mae_pooled": float(np.mean(np.abs(pred - y))), "rmse_pooled": float(rmse(y, pred)),
           "r2_pooled": float(r2_score(y, pred)), "per_fold": per_fold,
           "abs_errors": [float(v) for v in np.abs(pred - y)], "elapsed_s": time.time() - t0}
    print(f"    {label:<42} MAE {rec['mae_mean']:.4f} +/- {rec['mae_sd']:.4f} {unit}, "
          f"R2 {rec['r2_pooled']:+.3f}  ({rec['elapsed_s']:.0f} s)", flush=True)
    return rec


def paired_interval(a, b, seed: int = SEED, draws: int = 2000) -> Dict[str, object]:
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(draws)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(d)), "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def _baseline(y: np.ndarray, design) -> float:
    return float(np.mean([np.mean(np.abs(y[te] - y[tr].mean())) for tr, te in design]))


def measure(mols, names, y, designs, near, far, unit: str) -> Dict[str, object]:
    prepared = {}
    for model in MODEL_ORDER:
        t0 = time.time()
        prepared[model] = prepare(model, mols, names, near, far)
        print(f"    prepared {model} ({time.time() - t0:.0f} s)", flush=True)
    out: Dict[str, object] = {"models": list(MODEL_ORDER), "designs": {}}
    for dname, design in designs.items():
        print(f"  [{dname}] {len(design)} folds, training-mean error "
              f"{_baseline(y, design):.4f} {unit}", flush=True)
        rec = {"baseline_mae": _baseline(y, design), "n_folds": len(design), "models": {}}
        for model in MODEL_ORDER:
            rec["models"][model] = nested(prepared[model], y, design, f"{dname}/{model}", unit)
        rec["pairs"] = {}
        for label, (a, b) in PAIRS.items():
            rec["pairs"][label] = paired_interval(rec["models"][a]["abs_errors"],
                                                  rec["models"][b]["abs_errors"])
            p = rec["pairs"][label]
            print(f"    {label:<42} {p['mean']:+.4f} {unit} "
                  f"[{p['low']:+.4f}, {p['high']:+.4f}]"
                  f"{'  (excludes zero)' if p['excludes_zero'] else ''}", flush=True)
        out["designs"][dname] = rec
    return out


def hammett_stage() -> Dict[str, object]:
    mols, names, cols = load(HAMMETT_SDF, HAMMETT_TARGETS, "substituent")
    out: Dict[str, object] = {"targets": list(HAMMETT_TARGETS), "n": len(mols), "unit": "units"}
    for target in HAMMETT_TARGETS:
        y, have = cols[target]
        keep = np.where(have)[0]
        sub = [mols[i] for i in keep]
        print(f"  === {target}: {len(keep)} substituents", flush=True)
        out[target] = measure(sub, [names[i] for i in keep], y[keep],
                              {"random": random_folds(len(keep))},
                              HAMMETT_NEAR, HAMMETT_FAR, "units")
    return out


def nickel_stage() -> Dict[str, object]:
    mols, names, cols = load(NICKEL_SDF, NICKEL_TARGETS)
    out: Dict[str, object] = {"targets": list(NICKEL_TARGETS), "n": len(mols), "unit": "V"}
    for target in NICKEL_TARGETS:
        y, have = cols[target]
        keep = np.where(have)[0]
        sub = [mols[i] for i in keep]
        print(f"  === {target}: {len(keep)} complexes", flush=True)
        out[target] = measure(sub, [names[i] for i in keep], y[keep],
                              {"random": random_folds(len(keep)),
                               "leave-one-out": leave_one_out(len(keep))},
                              NICKEL_NEAR, NICKEL_FAR, "V")
    return out


# ------------------------------------------------------------------------------- the table

def _read(base: str, name: str):
    path = os.path.join(base, f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def _block(out: List[str], block, unit: str, title: str) -> None:
    out += [title, ""]
    for dname, rec in block["designs"].items():
        out += [f"#### {dname} (training-mean error {rec['baseline_mae']:.4f} {unit})", "",
                f"| model | MAE ({unit}) | standard deviation | R2 |", "|---|---|---|---|"]
        for model in block["models"]:
            m = rec["models"][model]
            out.append(f"| {model} | {m['mae_mean']:.4f} | {m['mae_sd']:.4f} | "
                       f"{m['r2_pooled']:+.3f} |")
        out += ["", f"| comparison | difference ({unit}) | 95 % interval |", "|---|---|---|"]
        for label, p in rec["pairs"].items():
            out.append(f"| {label} | {p['mean']:+.4f} | [{p['low']:+.4f}, {p['high']:+.4f}] |")
        out += [""]


def write_table(path: Optional[str] = None) -> str:
    base = path or RESULTS
    os.makedirs(base, exist_ok=True)
    check, hammett, nickel = _read(base, "check"), _read(base, "hammett"), _read(base, "nickel")
    out: List[str] = ["# Fields read at a few anchored points", "",
                      "Plan item EC3. A kernel compares two fields everywhere; these descriptors",
                      "read the field and its gradient at a few places of a shared frame. For the",
                      "electrostatic field the gradient is the oriented electric field. Five",
                      "outer folds, ten inner folds for the ridge and the field width.", ""]
    if check:
        out += ["## What the columns can carry", "",
                "A descriptor that is nearly the same in every member cannot separate two of",
                "them, so the spread of each column and its correlation with each target are",
                "measured before any model is fitted. The gradient is also checked against a",
                "finite difference of the value at every anchored point.", ""]
        for tag in ("hammett", "nickel"):
            if tag not in check:
                continue
            rec = check[tag]
            out += [f"### {tag}: {rec['n']} members, points "
                    f"{', '.join('`' + lab + '`' for lab in rec['labels'])}", "",
                    f"The gradient matches a finite difference to "
                    f"{rec['worst_gradient_error']:.1e} at every point of every field.", "",
                    "| source | columns | target | best column | its \\|r\\| | columns with "
                    "\\|r\\| above 0.5 |", "|---|---|---|---|---|---|"]
            for source in ("field", "coulomb"):
                if source not in rec:
                    continue
                for row in rec[source]["targets"]:
                    out.append(f"| {source} | {rec[source]['n_columns']} | {row['target']} | "
                               f"`{row['best_column']}` | {row['best_abs_r']:.3f} | "
                               f"{row['columns_above_0.5']} |")
            out += [""]
    for title, block in (("## The Hammett substituent constants", hammett),
                         ("## The nickel complexes of item EC4", nickel)):
        if not block:
            continue
        out += [title, ""]
        for target in block["targets"]:
            if target in block:
                _block(out, block[target], block.get("unit", "units"), f"### {target}")
    text = "\n".join(out) + "\n"
    dest = os.path.join(base, "table.md")
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"written {dest}")
    return text


STAGES = ("check", "hammett", "nickel")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", choices=STAGES + ("all",), default="all")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        write_table()
        return
    for stage in (STAGES if args.stage == "all" else (args.stage,)):
        print(f"[{stage}]", flush=True)
        t0 = time.time()
        res = {"check": check_stage, "hammett": hammett_stage, "nickel": nickel_stage}[stage]()
        res["seconds"] = time.time() - t0
        with open(os.path.join(RESULTS, f"{stage}.json"), "w", encoding="utf-8",
                  newline="\n") as fh:
            json.dump(res, fh, indent=1)
        print(f"  {stage} done in {res['seconds']:.0f} s", flush=True)
    write_table()


if __name__ == "__main__":
    main()
