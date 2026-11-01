"""How nonlinear is a model with a Gaussian kernel layer, and where: the local co-field measured.

``cmfqsar.nonlinearity`` gives a model with a nonlinear kernel layer a co-field per molecule (the
co-field of the linear model tangent to it there), the spread of those over the training set as a
map, and a closed-form non-linearity index per field type.  This script measures the three on the
Hammett set of ``benchmarks/run_hammett.py`` (one conformer per substituent, Gasteiger charges,
the five physicochemical fields), with three questions:

1. **Does the index order the targets?**  ``sigma_p * pi`` on the substituents that have both is a
   target that is quadratic in the fields if the two factors are linear in them, so its local
   co-field must change from molecule to molecule and its index must come out above those of the
   constants themselves.  (It does: 0.41 against 0.10 to 0.32.  The constants are not linear in
   the normalised field either -- the Gaussian layer gains 0.10 to 0.16 in q2 on them -- so their
   indices are not zero; the ordering is the check, not the absolute values.)
2. **Where is the spread?**  For an additive-looking target it should be confined; for the product
   it should cover the substituent.  The figure cuts the ring plane for sigma_p and the product.
3. **Do the maps survive refitting?**  The spread map is read as a map of where the model is not
   additive, so it has to look the same when the model is refitted on part of the set.  Five
   random four-fifths subsets are refitted at the chosen settings, and the pointwise correlation
   between their spread maps (and between their mean local co-fields) is reported.

For every target the linear kernel and the Gaussian layer are fitted with alpha, lambda and gamma
chosen by 10-fold cross-validation; q2 of both, the chosen gamma, the index per field and for the
joint kernel, and the stability correlations go to ``results/nonlinearity/``.  The figure cuts the
ring plane: the linear co-field, the mean local co-field and the spread of the electrostatic field
side by side for sigma_p and for the product.

Usage
-----
    python benchmarks/run_nonlinearity.py
    python benchmarks/run_nonlinearity.py --report      # rebuild table.md from the result files
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar.cofields import Grid, evaluate_gaussians  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, grid_search  # noqa: E402
from cmfqsar.nonlinearity import cofield_spread, nonlinearity_indices  # noqa: E402
from run_hammett import PHYSCHEM, SINGLE_SDF, draw_ring  # noqa: E402

OUT_DIR = os.path.join(HERE, "results", "nonlinearity")
TARGETS = ["sigma_m", "sigma_p", "pi", "sigma_p*pi"]
ALPHAS = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
LAMS = [10.0 ** k for k in range(-4, 2)]
GAMMAS = [0.25, 0.5, 1.0, 2.0, 4.0]
#: the plane cut through the ring, as run_hammett.py draws it
XS = np.arange(-4.5, 7.5, 0.15)
YS = np.arange(-5.0, 5.0, 0.15)


def load(path: str, target: str):
    """Molecules, names and the target; ``a*b`` is the product of two tabulated constants."""
    parts = target.split("*")
    ds = load_dataset(path, activity_prop=parts[0])
    y = np.asarray(ds.activity, float).copy()
    for p in parts[1:]:
        other = load_dataset(path, activity_prop=p)
        assert list(other.names) == list(ds.names)
        y = y * np.asarray(other.activity, float)
    keep = [i for i in range(len(ds)) if np.isfinite(y[i])]
    return [ds.mols[i] for i in keep], [ds.names[i] for i in keep], y[keep]


def fit(table: AtomTable, y: np.ndarray, layer: str, folds: int, seed: int = 0):
    fs = FieldSet.create(physchem=PHYSCHEM)
    model = CMFModel(fieldset=fs, method="krr", fit_bias=True, kernel_layer=layer)
    kw = {"gammas": GAMMAS} if layer != "linear" else {}
    res = grid_search(table, y, model, ALPHAS, lams=LAMS, folds=folds, seed=seed, **kw)
    model.fit(table, y, cv_folds=folds, cv_seed=seed, reuse_kernels=True)
    return model, res


def refit_like(model: CMFModel, table: AtomTable, y: np.ndarray) -> CMFModel:
    """The same settings on another table: no search, the chosen alpha, lambda and gamma."""
    m = CMFModel(fieldset=model.fieldset.copy(), method="krr", fit_bias=True,
                 kernel_layer=model.kernel_layer, params=dict(model.params),
                 normalize=model.normalize)
    m.fit(table, y, cv_folds=0)
    return m


def plane_grid(z: float) -> Grid:
    return Grid(origin=np.array([XS[0], YS[0], z]), spacing=float(XS[1] - XS[0]),
                shape=(len(XS), len(YS), 1))


def linear_slice(model: CMFModel, f: int, grid: Grid) -> np.ndarray:
    centres, coeffs, widths = model.cofield_coefficients(f)
    v = evaluate_gaussians(grid.points(), centres, coeffs, widths)
    return (v * model.fieldset.specs[f].h * model.scale[f]).reshape(grid.shape)[:, :, 0]


def stability(model: CMFModel, mf, y: np.ndarray, f: int, grid: Grid, repeats: int = 5,
              seed: int = 0) -> Tuple[float, float, float]:
    """Mean pairwise correlation of the spread maps, and of the mean maps, over refits on
    random four-fifths of the set; and the mean index of those refits."""
    rng = np.random.default_rng(seed)
    n = len(mf)
    sds, means, idx = [], [], []
    for _ in range(repeats):
        pick = np.sort(rng.choice(n, size=int(round(0.8 * n)), replace=False))
        m = refit_like(model, AtomTable.from_molecules([mf[i] for i in pick]), y[pick])
        sp = cofield_spread(m, f, grid, include_scale=True)
        sds.append(sp.sd.ravel())
        means.append(sp.mean.ravel())
        idx.append({i.field: i.spread for i in nonlinearity_indices(m)}["joint"])

    def mean_corr(maps):
        c = np.corrcoef(np.stack(maps))
        off = ~np.eye(len(maps), dtype=bool)
        return float(np.mean(c[off]))

    return mean_corr(sds), mean_corr(means), float(np.mean(idx))


def plot(panels: Dict[str, Dict[str, np.ndarray]], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = list(panels)
    kinds = ["linear co-field", "mean local co-field (gauss)", "spread of the local co-fields"]
    fig, axes = plt.subplots(len(targets), 3, figsize=(11.0, 3.1 * len(targets)), squeeze=False)
    ext = (XS[0], XS[-1], YS[0], YS[-1])
    for i, t in enumerate(targets):
        vmax = max(np.abs(panels[t][k]).max() for k in kinds[:2]) or 1.0
        for j, k in enumerate(kinds):
            ax = axes[i, j]
            img = panels[t][k].T
            if j < 2:
                im = ax.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax,
                               vmax=vmax, aspect="equal")
            else:
                im = ax.imshow(img, origin="lower", extent=ext, cmap="magma_r", vmin=0.0,
                               vmax=max(float(img.max()), 1e-12), aspect="equal")
            draw_ring(ax)
            ax.set_title(f"{t}: {k}", fontsize=9)
            ax.set_xlabel("x / A", fontsize=8)
            if j == 0:
                ax.set_ylabel("y / A", fontsize=8)
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle("Electrostatic field, ring plane: the co-field of the linear model, and the mean "
                 "and the spread of the molecule-dependent co-fields of the Gaussian layer",
                 fontsize=9)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def report(out_dir: str = OUT_DIR, path: Optional[str] = None) -> str:
    def rows(name):
        p = os.path.join(out_dir, name)
        return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []

    lines = ["# How nonlinear is the Gaussian layer, and where", "",
             "`benchmarks/run_nonlinearity.py` on the Hammett set (one conformer per substituent, "
             "Gasteiger charges, five fields). The non-linearity index of a field is the share of "
             "the squared norm of the molecule-dependent co-fields that their spread over the "
             "training molecules makes up: 0 for a linear model. `sigma_p*pi` is a target made "
             "non-additive on purpose. Stability is the mean correlation between the maps of "
             "five refits on random four-fifths of the set.", "",
             "| target | n | q² linear | q² gauss | γ | index joint | index electrostatic | "
             "index steric | index hydrophobic | mean cosine joint | spread map stability | "
             "mean map stability | index of the refits |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows("summary.csv"):
        lines.append(f"| {r['target']} | {r['n']} | {float(r['q2_linear']):.2f} | "
                     f"{float(r['q2_gauss']):.2f} | {float(r['gamma']):g} | "
                     f"{float(r['index_joint']):.3f} | {float(r['index_electrostatic']):.3f} | "
                     f"{float(r['index_steric']):.3f} | {float(r['index_hydrophobic']):.3f} | "
                     f"{float(r['cosine_joint']):.3f} | {float(r['stability_spread']):.2f} | "
                     f"{float(r['stability_mean']):.2f} | {float(r['index_refits']):.3f} |")
    lines += ["", "The figure `maps.png` cuts the ring plane: the linear co-field, the mean of the "
              "local co-fields and their spread, for the electrostatic field.", ""]
    text = "\n".join(lines)
    path = path or os.path.join(out_dir, "table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sdf", default=SINGLE_SDF)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5, help="refits for the stability check")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        report(a.out_dir)
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    summary: List[dict] = []
    panels: Dict[str, Dict[str, np.ndarray]] = {}
    grid = plane_grid(0.0)
    for target in [t.strip() for t in a.targets.split(",") if t.strip()]:
        mols, names, y = load(a.sdf, target)
        fs = FieldSet.create(physchem=PHYSCHEM)
        mf = [build_fields(m, fs, name=n) for m, n in zip(mols, names)]
        table = AtomTable.from_molecules(mf)
        print(f"\n== {target}: {len(y)} substituents")
        lin, rl = fit(table, y, "linear", a.folds)
        gau, rg = fit(table, y, "gauss", a.folds)
        print(f"  linear  alpha {rl.best_alpha:g} lambda {rl.best_params['lam']:g} "
              f"q2 {rl.best_q2:.3f}")
        print(f"  gauss   alpha {rg.best_alpha:g} lambda {rg.best_params['lam']:g} "
              f"gamma {rg.best_params['gamma']:g} q2 {rg.best_q2:.3f}")
        idx = {i.field: i for i in nonlinearity_indices(gau)}
        for i in idx.values():
            print(f"    {i}")
        f = gau.fieldset.names.index("electrostatic")
        sp = cofield_spread(gau, f, grid, include_scale=True)
        st_sd, st_mean, idx_refit = stability(gau, mf, y, f, grid, repeats=a.repeats)
        print(f"  stability over {a.repeats} refits: spread map r = {st_sd:.2f}, mean map "
              f"r = {st_mean:.2f}, joint index {idx_refit:.3f}")
        summary.append({"target": target, "n": len(y),
                        "alpha_linear": rl.best_alpha, "lambda_linear": rl.best_params["lam"],
                        "q2_linear": round(rl.best_q2, 4),
                        "alpha_gauss": rg.best_alpha, "lambda_gauss": rg.best_params["lam"],
                        "gamma": rg.best_params["gamma"], "q2_gauss": round(rg.best_q2, 4),
                        **{f"index_{k}": round(v.spread, 4) for k, v in idx.items()},
                        **{f"cosine_{k}": round(v.cosine, 4) for k, v in idx.items()},
                        "stability_spread": round(st_sd, 4), "stability_mean": round(st_mean, 4),
                        "index_refits": round(idx_refit, 4)})
        if target in ("sigma_p", "sigma_p*pi"):
            panels[target] = {"linear co-field": linear_slice(lin, f, grid),
                              "mean local co-field (gauss)": sp.mean[:, :, 0],
                              "spread of the local co-fields": sp.sd[:, :, 0]}
    with open(os.path.join(a.out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    if panels:
        plot(panels, os.path.join(a.out_dir, "maps.png"))
    report(a.out_dir)
    print(f"\nwritten to {os.path.relpath(a.out_dir, ROOT)} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
