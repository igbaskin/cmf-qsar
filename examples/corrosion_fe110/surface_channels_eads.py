"""What the surface-kernel model of E_ads weighs: the channels, in place of a co-field.

A model on the surface kernel has no field in a fixed frame to draw, but it is linear in its
channels: one per pair of field types at an in-plane distance rho and two heights (z, z').
With cosine normalisation the prediction is

    y(x) = w . phi~(x) + b,    phi~(x) = s * phi(x) / |s * phi(x)|,    w = sum_j a_j phi~(x_j),

so `w` is the model's weight per channel, and the mean contribution of a channel over the
training set, w_c * mean_j phi~_jc, says which channels drive the predictions.  Both are drawn
for every pair of field types as maps over the in-plane distance and the height, summed over
the second height; negative (blue) means that density in that channel makes E_ads more
negative, stronger adsorption.

The model is the one of surface_kernel_eads.py: the UMA poses as written, no alignment,
sigma 0.75, heights 1-6 A in steps of 1, lambda by 10-fold cross-validation.

    python examples/corrosion_fe110/surface_channels_eads.py

Writes surface_channels_eads.png and surface_channels_eads.txt next to this script.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import compare_alignments as ca  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import field_pairs  # noqa: E402
from cmfqsar.model import CMFModel, grid_search, r2_score  # noqa: E402


def main():
    fr = ca.frames()
    mols, names, _, e_ads, gas_ok = fr["pose"]
    keep = np.flatnonzero(np.isfinite(e_ads) & gas_ok)
    mols = [mols[i] for i in keep]; names = [names[i] for i in keep]; y = e_ads[keep]
    fs = FieldSet.create()
    train = AtomTable.from_molecules([build_fields(m, fs, name=n) for m, n in zip(mols, names)])
    model = CMFModel(fieldset=fs, method="krr", fit_bias=True, kernel_source="surface")
    res = grid_search(train, y, model, [1.0], lams=ca.LAMS, folds=10, seed=0)
    model.fit(train, y, cv_folds=0, reuse_kernels=True)
    yp = model.predict(train)
    spec = model._ac_spec()
    labels = [s.label for s in fs.specs]
    pairs = field_pairs(len(labels))
    n_rho, n_z = len(spec.centres), len(spec.heights)
    F = model.ac_train_ * model.ac_scale_                                  # scaled channels
    F_norm = F / np.linalg.norm(F, axis=1, keepdims=True)
    w = model.coef @ F_norm                                                 # (n_channels,)
    contrib = w * F_norm.mean(axis=0)                                       # mean contribution per channel
    W = w.reshape(len(pairs), n_rho, n_z, n_z)
    C = contrib.reshape(len(pairs), n_rho, n_z, n_z)
    lines = [f"surface-kernel model of E_ads: {len(y)} molecules, lambda = {res.best_params.get('lam'):g}, "
             f"q2 (10-fold) = {res.best_q2:.3f}, fitted r2 = {r2_score(y, yp):.3f}, "
             f"{len(w)} channels = {len(pairs)} pairs x {n_rho} distances x {n_z}^2 heights"]
    per_pair = np.abs(C).sum(axis=(1, 2, 3))
    order = np.argsort(-per_pair)
    lines.append("share of the summed |contribution| per field pair:")
    for i in order:
        f, g = pairs[i]
        name = labels[f] if f == g else f"{labels[f]} x {labels[g]}"
        lines.append(f"  {name:32s} {100 * per_pair[i] / per_pair.sum():5.1f} %   net {C[i].sum():+8.2f} kJ/mol")
    flat = [(contrib[c], c) for c in range(len(contrib))]
    flat.sort()
    def describe(c):
        i, r, zi, zj = np.unravel_index(c, (len(pairs), n_rho, n_z, n_z))
        f, g = pairs[i]
        name = labels[f] if f == g else f"{labels[f]} x {labels[g]}"
        return f"{name} at rho {spec.centres[r]:.2f} A, heights {spec.heights[zi]:.0f} / {spec.heights[zj]:.0f} A"
    lines.append("channels that most strengthen adsorption (mean contribution, kJ/mol):")
    lines += [f"  {v:+7.2f}  {describe(c)}" for v, c in flat[:10]]
    lines.append("channels that most weaken it:")
    lines += [f"  {v:+7.2f}  {describe(c)}" for v, c in flat[-10:][::-1]]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 5
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.3 * nrow), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.3)
    maps = C.sum(axis=3)                                                    # (pairs, rho, z): over z'
    vmax = np.abs(maps).max() or 1.0
    for k, (f, g) in enumerate(pairs):
        ax = axes[k // ncol][k % ncol]
        im = ax.imshow(maps[k].T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=(spec.centres[0] - spec.step / 2, spec.centres[-1] + spec.step / 2,
                               spec.heights[0] - spec.z_step / 2, spec.heights[-1] + spec.z_step / 2))
        ax.set_title(labels[f] if f == g else f"{labels[f]} x {labels[g]}", fontsize=9)
        ax.set_xlabel("in-plane distance rho, A", fontsize=8)
        if k % ncol == 0:
            ax.set_ylabel("height z, A", fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(len(pairs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="mean contribution, kJ/mol (blue: stronger adsorption)")
    fig.suptitle(f"Surface-kernel model of E_ads, mean channel contributions summed over the second height; "
                 f"q2 = {res.best_q2:.2f}, {len(y)} molecules, no alignment", fontsize=10)
    out = os.path.join(HERE, "surface_channels_eads.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    with open(os.path.join(HERE, "surface_channels_eads.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", out)


if __name__ == "__main__":
    main()
