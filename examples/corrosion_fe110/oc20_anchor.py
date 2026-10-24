"""The surface kernel on DFT adsorption energies: OC20 systems on one pure-metal surface.

Input: the .npz written by oc20_extract.py (DFT-relaxed adsorbate atoms in the surface frame,
z = 0 at the top surface atom, x and y centred on the adsorbate; the OC20 relaxed adsorption
energy as the target).  The adsorbates are OC20's 82 small C/H/O/N fragments, so the fields are
indicator fields of the elements (one Gaussian per atom of that element) and the steric field,
the same construction as the continuous indicator fields of atom types.

Models, all KRR with lambda (and alpha for the overlap kernel) chosen in an inner 10-fold
cross-validation, 5 outer folds, 3 repeats:

- overlap fields, poses as they are (centred in the plane, turned however the cell had them);
- the surface kernel (in-plane invariant, heights kept), on the same poses and on poses
  shuffled in the plane (must agree);
- the three-dimensional autocorrelation kernel, which throws the height away;
- element counts (ridge) and adsorbate identity (one-hot ridge): what composition and
  "which fragment" explain on their own.

The outer folds are random, and, in a second pass, grouped by adsorbate (leave whole
adsorbates out), which asks whether a kernel trained on some fragments predicts the energies
of fragments it has never seen on the same metal.

    python examples/corrosion_fe110/oc20_anchor.py examples/corrosion_fe110/oc20_Fe.npz [--repeats 3]

Writes <input stem>_anchor.csv next to the input and prints the table.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from cmfqsar.fields import AtomTable, FieldSet, FieldSpec, MoleculeFields  # noqa: E402
from cmfqsar.model import CMFModel, _kfold_indices, grid_search, r2_score, rmse  # noqa: E402

RADII = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52}
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O"}
ALPHAS = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
LAMS = [10.0 ** k for k in range(-4, 2)]


def load(path):
    d = np.load(path, allow_pickle=False)
    n = d["n_atoms"]
    starts = np.concatenate([[0], np.cumsum(n)])
    systems = []
    for i in range(len(n)):
        sl = slice(starts[i], starts[i + 1])
        systems.append({"sid": str(d["sid"][i]), "y": float(d["y"][i]), "ads": str(d["ads"][i]),
                        "numbers": d["numbers"][sl], "xyz": d["xyz"][sl]})
    return systems


def fieldset():
    specs = [FieldSpec(name=f"cif:{SYMBOL[z]}", kind="gauss", alpha=1.0) for z in (6, 1, 7, 8)]
    specs.append(FieldSpec(name="steric", kind="steric", alpha=1.0))
    return FieldSet(specs=specs)


def molecule(sys_, fs, shuffle_rng=None):
    z = sys_["numbers"]
    xyz = np.asarray(sys_["xyz"], float).copy()
    if shuffle_rng is not None:
        t = shuffle_rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
        xyz = xyz @ R.T + np.array([shuffle_rng.uniform(-4, 4), shuffle_rng.uniform(-4, 4), 0.0])
    W = np.zeros((len(z), len(fs.specs)))
    for k, spec in enumerate(fs.specs):
        if spec.name.startswith("cif:"):
            W[:, k] = (np.array([SYMBOL.get(int(v), "?") for v in z]) == spec.name[4:]).astype(float)
        else:
            W[:, k] = 1.0
    radii = np.array([RADII.get(int(v), 1.7) for v in z])
    return MoleculeFields(name=sys_["sid"], coords=xyz, weights=W, radii=radii,
                          symbols=[SYMBOL.get(int(v), "X") for v in z], atom_types=[SYMBOL.get(int(v), "X") for v in z],
                          bonds=np.zeros((0, 2), int), charges=np.zeros(len(z)))


def grouped_folds(groups, k, seed):
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    return [np.flatnonzero(np.isin(groups, part)) for part in np.array_split(uniq, k)]


def nested_kernel(train_table, y, model_template, alphas, folds_fn, repeats, seed):
    """External CV with an inner grid search; folds_fn(rep) gives the outer test folds."""
    n = len(y)
    preds = np.zeros((repeats, n))
    for rep in range(repeats):
        for te in folds_fn(rep):
            tr = np.setdiff1d(np.arange(n), te)
            model = model_template.fresh_copy()
            sub = train_table.subset(tr)
            grid_search(sub, y[tr], model, alphas, lams=LAMS, folds=10, seed=seed)
            model.fit(sub, y[tr], cv_folds=0, reuse_kernels=True)
            preds[rep, te] = model.predict(train_table.subset(te))
    q2 = [r2_score(y, p) for p in preds]
    return float(np.mean(q2)), float(np.std(q2)), float(np.mean([rmse(y, p) for p in preds]))


def ridge_cv(X, y, folds_fn, repeats, seed):
    from sklearn.linear_model import RidgeCV
    n = len(y)
    preds = np.zeros((repeats, n))
    for rep in range(repeats):
        for te in folds_fn(rep):
            tr = np.setdiff1d(np.arange(n), te)
            m = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]).fit(X[tr], y[tr])
            preds[rep, te] = m.predict(X[te])
    q2 = [r2_score(y, p) for p in preds]
    return float(np.mean(q2)), float(np.std(q2)), float(np.mean([rmse(y, p) for p in preds]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--min-atoms", type=int, default=0, help="keep adsorbates with at least this many atoms")
    ap.add_argument("--tag", default="", help="suffix of the output table")
    ap.add_argument("--surf-sigma", type=float, default=0.5, help="surface kernel: in-plane width and step, A")
    ap.add_argument("--surf-z", type=float, nargs=3, default=(0.5, 4.5, 0.5), metavar=("ZMIN", "ZMAX", "ZSTEP"))
    ap.add_argument("--only-surface", action="store_true", help="run the surface kernel rows only")
    a = ap.parse_args(argv)
    systems = load(a.npz)
    if a.min_atoms:
        systems = [s for s in systems if len(s["numbers"]) >= a.min_atoms]
    if a.max:
        systems = systems[:a.max]
    y = np.array([s["y"] for s in systems])
    ads = np.array([s["ads"] for s in systems])
    print(f"{len(systems)} systems, {len(set(ads))} adsorbates, E_ads {y.min():.2f}..{y.max():.2f} eV, sd {y.std():.2f}")
    fs = fieldset()
    mols = [molecule(s, fs) for s in systems]
    rng = np.random.default_rng(a.seed)
    shuffled = [molecule(s, fs, rng) for s in systems]
    table = AtomTable.from_molecules(mols)
    table_sh = AtomTable.from_molecules(shuffled)
    counts = np.array([[int((s["numbers"] == z).sum()) for z in (6, 1, 7, 8)] for s in systems], float)
    uniq_ads = sorted(set(ads))
    onehot = np.array([[float(x == u) for u in uniq_ads] for x in ads])
    rows = []
    for scheme in ("random", "leave-adsorbate-out"):
        if scheme == "random":
            folds_fn = lambda rep: _kfold_indices(len(y), a.outer, a.seed + rep)
        else:
            folds_fn = lambda rep: grouped_folds(ads, a.outer, a.seed + rep)
        def run(label, fn):
            t0 = time.time()
            q2, sd, rm = fn()
            rows.append({"folds": scheme, "model": label, "n": len(y), "q2": round(q2, 3), "q2_sd": round(sd, 3), "rmse_eV": round(rm, 3)})
            print(f"[{scheme:19s}] {label:46s} q2 = {q2:6.3f} +- {sd:.3f}  rmse {rm:.3f} eV  {time.time() - t0:.0f} s", flush=True)
        if not a.only_surface:
            run("element counts (ridge)", lambda: ridge_cv(counts, y, folds_fn, a.repeats, a.seed))
            if scheme == "random":
                run("adsorbate identity (one-hot ridge)", lambda: ridge_cv(onehot, y, folds_fn, a.repeats, a.seed))
            run("overlap fields, poses as written", lambda: nested_kernel(
                table, y, CMFModel(fieldset=fs, method="krr", fit_bias=True), ALPHAS, folds_fn, a.repeats, a.seed))
            run("3D autocorrelation (height thrown away)", lambda: nested_kernel(
                table, y, CMFModel(fieldset=fs, method="krr", fit_bias=True, kernel_source="autocorrelation"), [1.0], folds_fn, a.repeats, a.seed))
        z0, z1, dz = a.surf_z
        surf = CMFModel(fieldset=fs, method="krr", fit_bias=True, kernel_source="surface",
                        surf_z_min=z0, surf_z_max=z1, surf_z_step=dz, ac_sigma=a.surf_sigma, ac_d_max=6.0)
        label = f"surface kernel (sigma {a.surf_sigma:g}, z {z0:g}-{z1:g} step {dz:g})"
        run(label, lambda: nested_kernel(table, y, surf, [1.0], folds_fn, a.repeats, a.seed))
        if scheme == "random" and not a.only_surface:
            run("surface kernel, poses shuffled in the plane", lambda: nested_kernel(table_sh, y, surf, [1.0], folds_fn, a.repeats, a.seed))
    out = os.path.splitext(a.npz)[0] + "_anchor" + (f"_{a.tag}" if a.tag else "") + ".csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
