"""The co-fields of the E_ads model: where more of a field raises or lowers the adsorption energy.

The model of compare_alignments.py in the adsorbed-pose frame (registered in the surface plane),
fitted once on the 72 reliable molecules with alpha and lambda from a 10-fold cross-validation,
is linear in the molecular fields, so its coefficients are fields themselves (Baskin & Zhokhova
2013, Eq. 9).  h_f c_f C_f(x, y, z) is drawn for the five physicochemical fields on two planes
parallel to the surface, 2.0 A (the height of a chemisorbed ring) and 3.5 A above the top Fe
layer, with the reference benzimidazole's heavy atoms drawn in.  Blue: field density there
makes E_ads more negative (stronger adsorption); red: weaker.

    python examples/corrosion_fe110/cofields_eads.py [--planes 2.0 3.5] [--half-width 8]

Writes cofields_eads.png and cofields_eads.txt next to this script.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import compare_alignments as ca  # noqa: E402
from cmfqsar.cofields import evaluate_gaussians  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, grid_search, r2_score  # noqa: E402


def cofield_slice(model: CMFModel, f: int, z: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    centres, coeffs, widths = model.cofield_coefficients(f)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    v = evaluate_gaussians(pts, centres, coeffs, widths)
    v *= model.fieldset.specs[f].h * model.scale[f]
    return v.reshape(X.shape)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--planes", type=float, nargs="+", default=[2.0, 3.5])
    ap.add_argument("--half-width", type=float, default=8.0)
    ap.add_argument("--spacing", type=float, default=0.2)
    a = ap.parse_args(argv)

    fr = ca.frames()
    mols, names, _, e_ads, gas_ok = fr["pose+mcs"]
    keep = np.flatnonzero(np.isfinite(e_ads) & gas_ok)
    mols = [mols[i] for i in keep]
    names = [names[i] for i in keep]
    y = e_ads[keep]
    fs = FieldSet.create()
    mf = [build_fields(m, fs, name=n) for m, n in zip(mols, names)]
    train = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", fit_bias=True)
    res = grid_search(train, y, model, ca.ALPHAS, lams=ca.LAMS, folds=10, seed=0)
    model.fit(train, y, cv_folds=10, cv_seed=0, reuse_kernels=True)
    yp = model.predict(train)
    lines = [f"E_ads model in the adsorbed-pose frame: {len(y)} molecules, "
             f"alpha = {res.best_alpha:g}, lambda = {res.best_params.get('lam', float('nan')):g}, "
             f"q2 (10-fold) = {res.best_q2:.3f}, RMSEcv = {res.best_rmse:.1f} kJ/mol, fitted r2 = {r2_score(y, yp):.3f}"]
    # the reference molecule, flat on the surface
    ref = mols[0]
    conf = ref.GetConformer()
    heavy = [(i, conf.GetAtomPosition(i)) for i in range(ref.GetNumAtoms()) if ref.GetAtomWithIdx(i).GetAtomicNum() > 1]
    lines.append(f"reference {names[0]}: heavy atoms at z = {min(p.z for _, p in heavy):.2f}..{max(p.z for _, p in heavy):.2f} A")

    xs = np.arange(-a.half_width, a.half_width + 1e-9, a.spacing)
    ys = np.arange(-a.half_width, a.half_width + 1e-9, a.spacing)
    fields = [s.name for s in fs.specs]
    slices = {(f, z): cofield_slice(model, f, z, xs, ys) for f in range(len(fields)) for z in a.planes}
    for f, name in enumerate(fields):
        for z in a.planes:
            s = slices[(f, z)]
            lines.append(f"  {name:14s} z = {z:.1f} A: min {s.min():8.2f}  max {s.max():8.2f}  "
                         f"(kJ/mol per unit of field density integrated over the plane cell)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow, ncol = len(a.planes), len(fields)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow), squeeze=False)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in ref.GetBonds()
             if ref.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomicNum() > 1 and ref.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomicNum() > 1]
    pos = {i: (p.x, p.y) for i, p in heavy}
    for r, z in enumerate(a.planes):
        vmax = max(np.abs(slices[(f, z)]).max() for f in range(len(fields))) or 1.0
        for f, name in enumerate(fields):
            ax = axes[r][f]
            im = ax.imshow(slices[(f, z)], origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")   # negative (stronger adsorption) blue
            for i, j in bonds:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="black", lw=1.0)
            for i, (x0, y0) in pos.items():
                sym = ref.GetAtomWithIdx(i).GetSymbol()
                ax.scatter([x0], [y0], s=14 if sym == "C" else 30, color="black" if sym == "C" else "darkgreen", zorder=3)
            ax.set_title(f"{name}, z = {z:.1f} A", fontsize=10)
            ax.set_xlabel("x, A", fontsize=8)
            if f == 0:
                ax.set_ylabel("y, A", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=axes[r].tolist(), shrink=0.8, label="h c C(r), kJ/mol per field unit")
    fig.suptitle(f"Co-fields of the E_ads model over Fe(110) (blue: stronger adsorption); "
                 f"q2 = {res.best_q2:.2f}, {len(y)} molecules; {names[0]} drawn (N in green)", fontsize=10)
    out = os.path.join(HERE, "cofields_eads.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    with open(os.path.join(HERE, "cofields_eads.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
