"""
The electric-field kernel against the scalar one: does the vector quantity pay? (plan item EC5)

Every field in this program is a scalar and its kernel is the overlap of two scalars.  What a
reaction centre actually feels is not the potential but its gradient, the electric field, and the
quantity that pairs two molecules by it is the integral of ``grad phi_A . grad phi_B`` over space.
For atom-centred Gaussians that integral is in closed form -- it is the negative Laplacian of the
ordinary overlap with respect to the separation of the two atoms -- so the new field type costs no
quadrature and keeps rule 3 of the plan.  It carries the same charges as the electrostatic field,
so a comparison of the two is a comparison of two kernels on one description.

The item is justified only where the vector kernel beats the scalar one by more than the interval.
That is measured here on the two series where an overlap kernel is the right tool, which means a
series with a shared frame:

* **The Hammett sets** (`examples/hammett/hammett.sdf`): 95 substituted benzenes in one frame by
  rule, with the substituent constants sigma_m, sigma_p and pi.  The earlier Hammett measurement
  found that sigma_m is carried by the electrostatic field and sigma_p needs the resonance part
  that fragment charges cannot see, so this is the series where a better electrostatic description
  has the most room to show itself.
* **The nickel complexes of item EC4** (`examples/complexes_ni/ni_bpp.sdf`): 25 complexes sharing
  one ligand skeleton and one N3O3 coordination sphere, with the computed potentials of two
  couples.  A redox potential of a metal complex is the case the through-space catalysis
  literature describes, where the field a substituent exerts at the metal is said to decide it.

The iron series of item EC4 is **not** measured here, for two reasons recorded rather than argued:
item EC4 found that on a series whose members share nothing but the metal an overlap kernel of any
kind is behind the kernel that needs no frame, so the comparison would be between two tools that
are both wrong for it; and the compiled kernel loop does not implement this field kind, so the
2216 complexes would be an hour of the NumPy path for each field width.

The models
----------
``electrostatic``   the scalar electrostatic field alone
``efield``          the electric-field kind alone: the same charges, the gradient kernel
``physchem``        the five ordinary fields
``physchem+efield`` the same five with the electric-field kind added
``efield-for-es``   the five fields with the electric-field kind *in place of* the electrostatic
                    one, which asks whether the vector kernel is a better description of the same
                    thing rather than an addition to it

    python benchmarks/run_efield.py --stage check
    python benchmarks/run_efield.py --stage hammett
    python benchmarks/run_efield.py --stage nickel
    python benchmarks/run_efield.py --report
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

from cmfqsar.cofields import cofield_arrows, make_grid                      # noqa: E402
from cmfqsar.fields import (EFIELD_FIELD, AtomTable, FieldSet,              # noqa: E402
                            build_fields)
from cmfqsar.kernels import efield_pair_factor, kernel_matrices             # noqa: E402
from cmfqsar.model import CMFModel, _Learner, cv_predict, r2_score, rmse    # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "efield")
HAMMETT_SDF = os.path.join(ROOT, "examples", "hammett", "hammett.sdf")
NICKEL_SDF = os.path.join(ROOT, "examples", "complexes_ni", "ni_bpp.sdf")

SEED = 0
OUTER, INNER = 5, 10
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
ALPHAS = (0.3, 0.5, 1.0, 2.0)
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")

HAMMETT_TARGETS = ("sigma_m", "sigma_p", "pi")
NICKEL_TARGETS = ("E_ox_AN", "E_red_AN")

#: the field list of each model
MODELS: Dict[str, Tuple[str, ...]] = {
    "electrostatic": ("electrostatic",),
    "efield": (EFIELD_FIELD,),
    "physchem": FIELDS,
    "physchem+efield": FIELDS + (EFIELD_FIELD,),
    "efield-for-es": tuple(f for f in FIELDS if f != "electrostatic") + (EFIELD_FIELD,),
}
MODEL_ORDER = ("electrostatic", "efield", "physchem", "physchem+efield", "efield-for-es")

#: the comparisons the gate is read from.  "a vs b" is positive when b has the smaller error.
PAIRS = {
    "efield vs electrostatic": ("electrostatic", "efield"),
    "physchem+efield vs physchem": ("physchem", "physchem+efield"),
    "efield-for-es vs physchem": ("physchem", "efield-for-es"),
}


# ------------------------------------------------------------------------------- the data

def load(path: str, targets: Sequence[str]):
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing")
    mols, names = [], []
    for k, mol in enumerate(Chem.SDMolSupplier(path, removeHs=False, sanitize=True)):
        if mol is None:
            continue
        mols.append(mol)
        names.append(mol.GetProp("_Name") if mol.HasProp("_Name")
                     and mol.GetProp("_Name").strip() else f"mol_{k + 1}")
    cols = {}
    for t in targets:
        have = np.array([m.HasProp(t) for m in mols])
        vals = np.array([float(m.GetProp(t)) if m.HasProp(t) else np.nan for m in mols])
        cols[t] = (vals, have)
    return mols, names, cols


# ------------------------------------------------------------------------------- the check

def check_stage() -> Dict[str, object]:
    """Is the new kernel the integral it claims, and is it a different description?

    Three readings.  The first compares the closed form against a grid integration on a real
    molecule, which is the same check the unit tests make and is repeated here so that the
    benchmark's own table carries it.  The second counts how many atom pairs of a real series
    contribute a *negative* term, since this is the only pair integral in the program that can.
    The third asks whether the two kernels are different at all: if the vector kernel orders the
    pairs of a series exactly as the scalar one does, no model built on it can differ.
    """
    out: Dict[str, object] = {}
    mols, names, _cols = load(HAMMETT_SDF, HAMMETT_TARGETS)
    names = [m.GetProp("substituent") if m.HasProp("substituent") else n
             for m, n in zip(mols, names)]
    alpha = 1.0
    fs = FieldSet.create(physchem=["electrostatic", EFIELD_FIELD], alpha=alpha,
                         charge_model="gasteiger", param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    K = kernel_matrices(table, table, fs.specs)

    # the share of atom pairs whose gradient integral is negative
    zero_at = 6.0 / alpha
    neg = tot = 0
    for i in range(min(20, len(mf))):
        for j in range(i, min(20, len(mf))):
            d2 = ((mf[i].coords[:, None, :] - mf[j].coords[None, :, :]) ** 2).sum(axis=2)
            neg += int((d2 > zero_at).sum())
            tot += int(d2.size)
    out["negative_pair_share"] = neg / max(tot, 1)
    out["sign_change_distance"] = float(np.sqrt(zero_at))
    print(f"    the pair integral turns negative beyond {np.sqrt(zero_at):.2f} A, which is "
          f"{100 * out['negative_pair_share']:.0f} % of the atom pairs of this series", flush=True)

    # A molecule whose electrostatic field is identically zero has a zero self-kernel, and the
    # cosine of it with anything is undefined.  Those molecules are named and left out of the
    # similarity reading rather than divided by: the Gasteiger scheme has no parameters for a few
    # elements, and one such atom makes it return nothing for the whole molecule.  Item EC4
    # repaired this for the d-block metals; selenium and the other elements it does not know are
    # still affected, which is a fault of the existing series and is recorded here rather than
    # changed, because changing it would move the published Hammett tables.
    diag = np.stack([np.diag(K[0]), np.diag(K[1])])
    good = np.where((diag[0] > 1e-12) & (diag[1] > 1e-12))[0]
    out["zero_field_molecules"] = [names[i] for i in range(len(names)) if i not in set(good)]
    out["n_zero_field"] = len(out["zero_field_molecules"])
    if out["n_zero_field"]:
        print(f"    {out['n_zero_field']} of {len(names)} molecules have an electrostatic field "
              f"that is identically zero: {', '.join(out['zero_field_molecules'])}", flush=True)
    Kg = K[:, good][:, :, good]
    iu = np.triu_indices_from(Kg[0], 1)
    a = Kg[0][iu] / np.sqrt(np.outer(np.diag(Kg[0]), np.diag(Kg[0])))[iu]
    b = Kg[1][iu] / np.sqrt(np.outer(np.diag(Kg[1]), np.diag(Kg[1])))[iu]
    out["similarity_electrostatic"] = {"mean": float(a.mean()), "sd": float(a.std())}
    out["similarity_efield"] = {"mean": float(b.mean()), "sd": float(b.std())}
    out["kernel_correlation"] = float(np.corrcoef(a, b)[0, 1])
    print(f"    two molecules are {a.mean():.3f} +/- {a.std():.3f} alike through the scalar "
          f"kernel and {b.mean():.3f} +/- {b.std():.3f} through the vector one", flush=True)
    print(f"    the two cosine kernels correlate {out['kernel_correlation']:+.3f} over the "
          f"{len(a)} pairs of the series", flush=True)

    # the closed form against quadrature, on one real molecule against another
    from cmfqsar.cofields import evaluate_gaussian_gradients
    ref = [m.coords - m.coords.mean(axis=0) for m in mf[:2]]
    x = np.linspace(-9.0, 9.0, 140)
    h = x[1] - x[0]
    grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    g = [evaluate_gaussian_gradients(grid, c, m.weights[:, 1], np.full(len(c), alpha))
         for c, m in zip(ref, mf[:2])]
    numeric = float(np.einsum("ij,ij->", g[0], g[1]) * h ** 3)
    moved = AtomTable.from_molecules([_with_coords(mf[0], ref[0]), _with_coords(mf[1], ref[1])])
    closed = float(kernel_matrices(moved, moved, fs.specs)[1, 0, 1])
    out["quadrature"] = {"closed_form": closed, "grid_integral": numeric,
                         "relative_difference": abs(closed - numeric) / max(abs(numeric), 1e-12),
                         "grid_points": int(len(grid)), "grid_spacing": float(h)}
    print(f"    closed form {closed:.6f} against a grid integral {numeric:.6f} on two real "
          f"molecules: {100 * out['quadrature']['relative_difference']:.3f} % apart", flush=True)
    out["pair_factor"] = {str(d): float(efield_pair_factor(alpha, np.array([d * d]))[0])
                          for d in (0.0, 1.0, 2.0, 3.0, 4.0)}
    return out


def _with_coords(mol_fields, coords):
    import copy
    out = copy.copy(mol_fields)
    out.coords = np.asarray(coords, float)
    return out


# ------------------------------------------------------------------------------- the protocol

def random_folds(n: int, k: int = OUTER, seed: int = SEED):
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    return [(np.setdiff1d(np.arange(n), g), g) for g in np.array_split(order, k)]


def leave_one_out(n: int):
    return [(np.setdiff1d(np.arange(n), [i]), np.array([i])) for i in range(n)]


def _unit_diagonal(K: np.ndarray) -> np.ndarray:
    """The same kernel with a unit diagonal, so that models with different numbers of fields are
    on one scale and the comparison is of descriptions and not of scales."""
    d = np.sqrt(np.clip(np.diag(np.asarray(K, float)), 1e-12, None))
    return np.asarray(K, float) / np.outer(d, d)


def prepare(model: str, mols, names) -> Dict[float, np.ndarray]:
    out = {}
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=list(MODELS[model]), alpha=alpha,
                             charge_model="gasteiger", param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        cmf = CMFModel(fieldset=fs, method="krr", normalize=True)
        out[alpha] = _unit_diagonal(KC.joint_training_kernel(cmf, AtomTable.from_molecules(mf)))
    return out


def nested(entry, y: np.ndarray, design, label: str, unit: str) -> Dict[str, object]:
    pred = np.full(len(y), np.nan)
    per_fold = []
    t0 = time.time()
    for k, (tr, te) in enumerate(design):
        best = None
        for alpha in entry:
            K = entry[alpha]
            for lam in LAMS:
                p = cv_predict(K[np.ix_(tr, tr)], y[tr], "krr", {"lam": float(lam)},
                               folds=min(INNER, len(tr)), seed=SEED)
                err = rmse(y[tr], p)
                if best is None or err < best[0]:
                    best = (err, alpha, float(lam))
        _e, alpha, lam = best
        K = entry[alpha]
        learner = _Learner("krr", {"lam": lam}, True).fit(K[np.ix_(tr, tr)], y[tr])
        pred[te] = learner.predict(K[np.ix_(te, tr)])
        per_fold.append({"fold": k, "alpha": float(alpha), "lam": lam, "n_test": int(len(te)),
                         "mae": float(np.mean(np.abs(pred[te] - y[te])))})
    maes = np.array([f["mae"] for f in per_fold])
    rec = {"n": int(len(y)), "mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=0)),
           "mae_pooled": float(np.mean(np.abs(pred - y))), "rmse_pooled": float(rmse(y, pred)),
           "r2_pooled": float(r2_score(y, pred)), "per_fold": per_fold,
           "abs_errors": [float(v) for v in np.abs(pred - y)], "elapsed_s": time.time() - t0}
    print(f"    {label:<34} MAE {rec['mae_mean']:.4f} +/- {rec['mae_sd']:.4f} {unit}, "
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


def measure(mols, names, y, designs, unit: str) -> Dict[str, object]:
    prepared = {}
    for model in MODEL_ORDER:
        t0 = time.time()
        prepared[model] = prepare(model, mols, names)
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
            print(f"    {label:<34} {p['mean']:+.4f} {unit} "
                  f"[{p['low']:+.4f}, {p['high']:+.4f}]"
                  f"{'  (excludes zero)' if p['excludes_zero'] else ''}", flush=True)
        out["designs"][dname] = rec
    return out


def hammett_stage() -> Dict[str, object]:
    mols, names, cols = load(HAMMETT_SDF, HAMMETT_TARGETS)
    out: Dict[str, object] = {"targets": list(HAMMETT_TARGETS), "n": len(mols)}
    for target in HAMMETT_TARGETS:
        vals, have = cols[target]
        keep = np.where(have)[0]
        sub = [mols[i] for i in keep]
        sub_names = [names[i] for i in keep]
        y = vals[keep]
        print(f"  === {target}: {len(y)} substituents", flush=True)
        out[target] = measure(sub, sub_names, y, {"random": random_folds(len(y))}, "units")
    return out


def nickel_stage() -> Dict[str, object]:
    mols, names, cols = load(NICKEL_SDF, NICKEL_TARGETS)
    out: Dict[str, object] = {"targets": list(NICKEL_TARGETS), "n": len(mols)}
    for target in NICKEL_TARGETS:
        vals, have = cols[target]
        keep = np.where(have)[0]
        sub = [mols[i] for i in keep]
        sub_names = [names[i] for i in keep]
        y = vals[keep]
        print(f"  === {target}: {len(y)} complexes", flush=True)
        designs = {"random": random_folds(len(y)), "leave-one-out": leave_one_out(len(y))}
        out[target] = measure(sub, sub_names, y, designs, "V")
    out["arrows"] = arrows_stage(mols, names)
    return out


def arrows_stage(mols, names) -> Dict[str, object]:
    """Where the vector co-field points, for the nickel oxidation potential.

    The co-field of a scalar field is read as a value at a place.  The co-field of this kind is
    read as an arrow: it says which way the model wants the field to change there.  The arrows
    are reported by their distance from the metal, because that is what makes a position in this
    series interpretable -- every member has the same metal and the same six donors, so a reading
    at the metal cannot separate two of them.
    """
    from cmfqsar.complexes import donor_indices, metal_index
    y = np.array([float(m.GetProp("E_ox_AN")) for m in mols])
    fs = FieldSet.create(physchem=[EFIELD_FIELD], alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), y, cv_folds=0)
    grid = make_grid(np.vstack([m.coords for m in mf]), margin=2.0, spacing=0.5)
    points, vectors, lengths = cofield_arrows(model, 0, grid, count=8, min_separation=2.0)
    ref = mols[0]
    xyz = ref.GetConformer().GetPositions()
    metal = xyz[metal_index(ref)]
    donors = xyz[donor_indices(ref)]
    rows = []
    for p, v, length in zip(points, vectors, lengths):
        radial = float(np.dot(v, (p - metal) / max(np.linalg.norm(p - metal), 1e-9)))
        rows.append({"distance_from_metal": float(np.linalg.norm(p - metal)),
                     "distance_from_nearest_donor":
                         float(np.min(np.linalg.norm(donors - p, axis=1))),
                     "length": float(length), "radial_component": radial,
                     "point": [float(c) for c in p]})
        print(f"    arrow of length {length:.4f} at {rows[-1]['distance_from_metal']:.2f} A from "
              f"the metal, {rows[-1]['distance_from_nearest_donor']:.2f} A from the nearest "
              f"donor, {radial:+.4f} of it pointing away from the metal", flush=True)
    return {"target": "E_ox_AN", "alpha": 1.0, "lam": 0.1, "arrows": rows}


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
    out: List[str] = ["# The electric-field kernel against the scalar one", "",
                      "Plan item EC5. The kernel of this field type is the integral of the two",
                      "molecules' field *gradients* against each other, in closed form, and it",
                      "carries the same charges as the electrostatic field. So every comparison",
                      "below is between two kernels on one description. Five outer folds, ten",
                      "inner folds for the ridge and the field width.", ""]
    if check:
        q = check["quadrature"]
        out += ["## Is it the integral it claims, and is it different?", "",
                f"On two molecules of the Hammett series the closed form gives "
                f"{q['closed_form']:.6f} and a grid integral of {q['grid_points']} points at "
                f"{q['grid_spacing']:.3f} angstrom gives {q['grid_integral']:.6f}, "
                f"{100 * q['relative_difference']:.3f} % apart.", "",
                f"The pair integral changes sign at {check['sign_change_distance']:.2f} angstrom "
                f"(at the field width alpha = 1), and "
                f"{100 * check['negative_pair_share']:.0f} % of the atom pairs of this series lie "
                "beyond it, so most pairs contribute a negative term. This is the only pair "
                "integral in the program that can, and the kernel matrix is still positive "
                "semidefinite because it is an inner product of two vector fields.", "",
                f"Two molecules of the series are "
                f"{check['similarity_electrostatic']['mean']:.3f} +/- "
                f"{check['similarity_electrostatic']['sd']:.3f} alike through the scalar kernel "
                f"and {check['similarity_efield']['mean']:.3f} +/- "
                f"{check['similarity_efield']['sd']:.3f} through the vector one; the two "
                f"cosine kernels correlate {check['kernel_correlation']:+.3f}. A correlation of "
                "one would mean the vector kernel orders the pairs exactly as the scalar one "
                "does, and no model built on it could differ.", ""]
    if hammett:
        out += ["## The Hammett substituent constants", ""]
        for target in hammett["targets"]:
            if target in hammett:
                _block(out, hammett[target], "units", f"### {target}")
    if nickel:
        out += ["## The nickel complexes of item EC4", ""]
        for target in nickel["targets"]:
            if target in nickel:
                _block(out, nickel[target], "V", f"### {target}")
        arrows = nickel.get("arrows")
        if arrows:
            out += ["### Where the vector co-field points", "",
                    "The co-field of this kind is an arrow field: at each place it says which way "
                    "the model wants the field to change. The strongest arrows of the "
                    f"{arrows['target']} model, at least 2 angstrom apart:", "",
                    "| length | from the metal (A) | from the nearest donor (A) | pointing away "
                    "from the metal |", "|---|---|---|---|"]
            for r in arrows["arrows"]:
                out.append(f"| {r['length']:.4f} | {r['distance_from_metal']:.2f} | "
                           f"{r['distance_from_nearest_donor']:.2f} | "
                           f"{r['radial_component']:+.4f} |")
            out += [""]
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
