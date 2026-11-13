#!/usr/bin/env python
"""
Draw the figures of the manual that are computed, into ``docs/figures/``.

A figure in the manual is either a picture a benchmark script already writes next to its results
(the Hammett co-fields, the co-field around a carbon, the channel map of the surface model), or
one of the figures this script draws from the example data and the stored benchmark results.
Like the tables, they are generated, so that a figure cannot stay in the manual after the code
or the results that made it have changed:

    python docs/make_figures.py              # draw every figure into docs/figures/
    python docs/make_figures.py search       # only the figures whose name starts so
    python docs/make_figures.py --check      # draw into a temporary folder and compare bytes;
                                             # exit 1 when a figure differs from the committed one

Line and dot plots are written as SVG, which the PDF carries as vector drawings; maps of fields
and co-fields are written as PNG, because a map is a raster. The computed figures need the
package and its example data; the ones read from the benchmark results need only the files
under ``benchmarks/results``. Everything runs in about two minutes.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
RESULTS = os.path.join(ROOT, "benchmarks", "results")
sys.path.insert(0, ROOT)

from cmfqsar.dataio import relative_path  # noqa: E402

WIDTH = 6.3          # inches: the text width of the manual's page
DPI = 150

_CACHE = {}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "svg.fonttype": "path",      # glyphs as outlines: no font is needed to place the file
        "svg.hashsalt": "cmfqsar",   # element ids that do not change between runs
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    })
    return matplotlib


def save(fig, path: str):
    """Write the figure without a time stamp, so that an unchanged figure gives the same bytes."""
    import matplotlib.pyplot as plt
    if path.endswith(".svg"):
        fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.02,
                    metadata={"Date": None, "Creator": None})
    else:
        fig.savefig(path, format="png", bbox_inches="tight", pad_inches=0.02,
                    metadata={"Software": None})
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Shared objects: the steroid model, the selectivity model, a plane through a series
# --------------------------------------------------------------------------- #

def steroids():
    """The steroid training set, its fields and the fitted model, built once."""
    if "steroids" in _CACHE:
        return _CACHE["steroids"]
    from cmfqsar import CMFModel
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    from cmfqsar.model import grid_search

    ds = load_dataset(os.path.join(ROOT, "examples", "steroids_cbg.sdf"))
    fs = FieldSet.create()
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    idx = list(ds.train_indices)
    train = AtomTable.from_molecules([mf[i] for i in idx])
    y = ds.activity[idx]
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    search = grid_search(train, y, model, alphas=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0],
                         lams=[1e-3, 1e-2, 0.1, 1.0, 10.0])
    model.fit(train, y, cv_folds=10)
    out = {"ds": ds, "fs": fs, "mf": mf, "train_idx": idx, "y": y, "model": model,
           "search": search}
    _CACHE["steroids"] = out
    return out


def plane_of(mols):
    """An orthonormal frame (origin, e1, e2, e3) through the heavy atoms of aligned molecules.

    e1 and e2 span the mean plane of the series and e3 is its normal, so a slice at the origin
    along e3 cuts through the skeleton every molecule shares.
    """
    pts = np.vstack([m.coords[[s != "H" for s in m.symbols]] for m in mols])
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
    e1, e2, e3 = vt
    # a reproducible orientation: the long axis points to +x, the normal to +z
    if e1[0] < 0:
        e1 = -e1
    if e2[1] < 0:
        e2 = -e2
    e3 = np.cross(e1, e2)
    return origin, e1, e2, e3


def plane_points(frame, xs, ys, offset=0.0):
    origin, e1, e2, e3 = frame
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = (origin + offset * e3 + X.ravel()[:, None] * e1 + Y.ravel()[:, None] * e2)
    return pts, X.shape


def project(frame, coords):
    origin, e1, e2, _ = frame
    d = coords - origin
    return np.column_stack([d @ e1, d @ e2])


def draw_skeleton(ax, mol, frame, colour="black", hydrogens=False):
    """The bonds between heavy atoms of one molecule, projected onto the plane; with
    `hydrogens`, the bonds to the hydrogen atoms as well, drawn thinner."""
    xy = project(frame, mol.coords)
    heavy = np.array([s != "H" for s in mol.symbols])
    for a, b in mol.bonds:
        if heavy[a] and heavy[b]:
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], color=colour, lw=1.0,
                    solid_capstyle="round", zorder=3)
        elif hydrogens:
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], color=colour, lw=0.5,
                    solid_capstyle="round", zorder=3)
    ax.scatter(xy[heavy, 0], xy[heavy, 1], s=7, color=colour, zorder=4)
    if hydrogens:
        ax.scatter(xy[~heavy, 0], xy[~heavy, 1], s=2.5, color=colour, zorder=4)


def field_slice(mol, spec, f, pts):
    from cmfqsar.cofields import _centres_and_widths, evaluate_gaussians
    centres, weights, widths = _centres_and_widths(mol, spec, f)
    return evaluate_gaussians(pts, centres, weights, widths)


def cofield_slice(model, f, pts):
    """h_f c_f C_f on the points: the change of the prediction per unit of field."""
    from cmfqsar.cofields import evaluate_gaussians
    centres, coeffs, widths = model.cofield_coefficients(f)
    v = evaluate_gaussians(pts, centres, coeffs, widths)
    scale = model.scale[f] if getattr(model, "scale", None) is not None else 1.0
    return v * model.fieldset.specs[f].h * scale


def signed_map(ax, Z, xs, ys, cmap="RdBu_r", vmax=None):
    vmax = vmax or (np.abs(Z).max() or 1.0)
    return ax.imshow(Z, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=cmap,
                     vmin=-vmax, vmax=vmax, aspect="equal", interpolation="bilinear")


def axes_in_angstrom(ax, xlabel=True, ylabel=True):
    if xlabel:
        ax.set_xlabel("x / Å")
    if ylabel:
        ax.set_ylabel("y / Å")
    ax.tick_params(length=2)


# --------------------------------------------------------------------------- #
# The figures
# --------------------------------------------------------------------------- #

def fig_fields(path):
    """Chapter 2: the electrostatic field of one steroid at two attenuations, and its steric field."""
    import matplotlib.pyplot as plt
    from cmfqsar.fields import FieldSet, build_fields
    s = steroids()
    frame = plane_of([s["mf"][i] for i in s["train_idx"]])
    name = "cortisol" if "cortisol" in s["ds"].names else s["ds"].names[s["train_idx"][0]]
    rdmol = s["ds"].mols[s["ds"].names.index(name)]
    panels = [("electrostatic", 0.3), ("electrostatic", 3.0), ("steric", 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.35))
    for ax, (fname, alpha) in zip(axes, panels):
        fs = FieldSet.create(alpha=alpha)
        mol = build_fields(rdmol, fs, name)
        if ax is axes[0]:
            xy = project(frame, mol.coords)
            lo, hi = xy.min(axis=0) - 3.0, xy.max(axis=0) + 3.0
            xs, ys = np.arange(lo[0], hi[0], 0.12), np.arange(lo[1], hi[1], 0.12)
            pts, shape = plane_points(frame, xs, ys)
        f = fs.names.index(fname)
        Z = field_slice(mol, fs.specs[f], f, pts).reshape(shape)
        if fname == "electrostatic":
            im = signed_map(ax, Z, xs, ys, vmax=np.percentile(np.abs(Z), 99.5))
        else:
            im = ax.imshow(Z, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="Oranges",
                           vmin=0, vmax=np.percentile(Z, 99.5), aspect="equal", interpolation="bilinear")
        draw_skeleton(ax, mol, frame)
        ax.set_title(f"{fs.specs[f].label}, α = {alpha:g}", fontsize=8)
        axes_in_angstrom(ax, ylabel=ax is axes[0])
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03).ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    save(fig, path)


def fig_overlap(path):
    """Chapter 2: the overlap of two Gaussians against their distance, for three attenuations."""
    import matplotlib.pyplot as plt
    d = np.linspace(0, 6, 300)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.5))
    for alpha, ls in ((0.3, "-"), (1.0, "--"), (3.0, ":")):
        ax.plot(d, np.exp(-alpha / 4 * d ** 2), ls, color="#1f4e79", lw=1.4, label=f"α = {alpha:g}")
    ax.set_xlabel("distance between the two atoms / Å")
    ax.set_ylabel("overlap, relative to distance 0")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, 6)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, path)


def fig_search(path):
    """Chapter 7: the cross-validated q² over the grid of attenuation and ridge, steroid set."""
    import matplotlib.pyplot as plt
    s = steroids()
    table = s["search"].table
    alphas = sorted({r["alpha"] for r in table})
    lams = sorted({r["lam"] for r in table})
    Q = np.full((len(lams), len(alphas)), np.nan)
    for r in table:
        Q[lams.index(r["lam"]), alphas.index(r["alpha"])] = r["q2"]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 2.9))
    vmin = max(np.nanmin(Q), -0.5)
    im = ax.imshow(Q, origin="lower", cmap="viridis", vmin=vmin, vmax=np.nanmax(Q), aspect="auto")
    ax.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    ax.set_yticks(range(len(lams)), [f"{l:g}" for l in lams])
    ax.set_xlabel("attenuation α")
    ax.set_ylabel("ridge λ")
    best = np.unravel_index(np.nanargmax(Q), Q.shape)
    for i in range(len(lams)):
        for j in range(len(alphas)):
            if np.isfinite(Q[i, j]):
                colour = "white" if Q[i, j] < vmin + 0.6 * (np.nanmax(Q) - vmin) else "black"
                weight = "bold" if (i, j) == best else "normal"
                ax.text(j, i, f"{Q[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color=colour, fontweight=weight)
    ax.set_title(f"q² of the steroid training set, {len(s['y'])} molecules")
    fig.colorbar(im, ax=ax, shrink=0.9, pad=0.03, label="q²").ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, path)


def cofield_model():
    """The steroid model whose co-field is drawn, and the frame it is drawn in, built once.

    It is fitted at alpha = 1 with the ridge the search found best there: a sharper attenuation
    scores a little higher on this set and gives a co-field of isolated dots.
    """
    if "cofield_model" in _CACHE:
        return _CACHE["cofield_model"]
    from cmfqsar import CMFModel
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    s = steroids()
    rows = [r for r in s["search"].table if r["alpha"] == 1.0]
    best = max(rows, key=lambda r: r["q2"])
    fs = FieldSet.create(alpha=1.0)
    mf = [build_fields(m, fs, n) for m, n in zip(s["ds"].mols, s["ds"].names)]
    train = AtomTable.from_molecules([mf[i] for i in s["train_idx"]])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": best["lam"]})
    model.fit(train, s["y"], cv_folds=10)
    frame = plane_of([mf[i] for i in s["train_idx"]])
    xy = np.vstack([project(frame, mf[i].coords) for i in s["train_idx"]])
    lo, hi = xy.min(axis=0) - 2.0, xy.max(axis=0) + 2.0
    xs, ys = np.arange(lo[0], hi[0], 0.15), np.arange(lo[1], hi[1], 0.15)
    out = {"fs": fs, "mf": mf, "model": model, "frame": frame, "xs": xs, "ys": ys,
           "lam": best["lam"], "q2": round(float(model.stats.q2), 3)}
    _CACHE["cofield_model"] = out
    print(f"  co-field model: alpha = 1, lambda = {out['lam']:g}, q2 = {out['q2']}")
    return out


def fig_cofield(path):
    """Chapter 8: the electrostatic and the steric co-field of the steroid model in the ring plane."""
    import matplotlib.pyplot as plt
    s, c = steroids(), cofield_model()
    fs, mf, model, frame = c["fs"], c["mf"], c["model"], c["frame"]
    ref = mf[s["train_idx"][int(np.argmax(s["y"]))]]      # the most active training molecule
    xs, ys = c["xs"], c["ys"]
    pts, shape = plane_points(frame, xs, ys)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    for ax, name in zip(axes, ("electrostatic", "steric")):
        f = fs.names.index(name)
        Z = cofield_slice(model, f, pts).reshape(shape)
        im = signed_map(ax, Z, xs, ys)
        draw_skeleton(ax, ref, frame)
        ax.set_title(f"{fs.specs[f].label} co-field")
        axes_in_angstrom(ax, ylabel=ax is axes[0])
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03).ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, path)
    print(f"  co-field drawn with {ref.name}")


def fig_overlapmap(path):
    """Chapter 8: a test molecule's field, the co-field, and their product, whose integral is the
    contribution of that field to the prediction."""
    import matplotlib.pyplot as plt
    from cmfqsar.cofields import make_grid, overlap_volume
    s, c = steroids(), cofield_model()
    fs, mf, model, frame = c["fs"], c["mf"], c["model"], c["frame"]
    f = fs.names.index("electrostatic")
    # the test molecule whose electrostatic field contributes most, in either direction
    test = list(s["ds"].test_indices)
    contrib = {i: model.atom_contributions(mf[i], per_field=True)[f].sum() for i in test}
    i = max(contrib, key=lambda k: abs(contrib[k]))
    mol = mf[i]
    # the integral over space of the overlap, on a grid, against the exact contribution
    grid = make_grid(mol.coords, margin=4.0, spacing=0.25)
    integral = float(overlap_volume(model, mol, f, grid).sum() * grid.spacing ** 3)
    print(f"  overlap map: {mol.name}, electrostatic contribution {contrib[i]:+.3f} exact, "
          f"{integral:+.3f} integrated on the grid")
    xs, ys = c["xs"], c["ys"]
    pts, shape = plane_points(frame, xs, ys)
    X = field_slice(mol, fs.specs[f], f, pts).reshape(shape) * model.field_scale_for(f, mol)
    C = cofield_slice(model, f, pts).reshape(shape)
    # read as the equation of Fig. 1 of the 2013 paper: co-field, times field, equals the
    # integrand whose integral over space is the contribution
    panels = [(C, "co-field $C_f$"), (X, "field $X_f$"), (C * X, "product $C_f\\,X_f$")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.1))
    bars = []
    for ax, (Z, title) in zip(axes, panels):
        im = signed_map(ax, Z, xs, ys)
        draw_skeleton(ax, mol, frame)
        ax.set_title(title, fontsize=8)
        axes_in_angstrom(ax, ylabel=ax is axes[0])
        bars.append(fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02))
        bars[-1].ax.tick_params(labelsize=6.5)
    fig.suptitle("contribution of field $f$ to the prediction $= h_f c_f \\int C_f(\\mathbf{r})\\,"
                 "X_f(\\mathbf{r})\\,\\mathrm{d}\\mathbf{r}$\n"
                 + f"$= {contrib[i]:+.2f}$ for the electrostatic field of {mol.name}",
                 fontsize=8.5, y=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    for k, symbol in ((0, "×"), (1, "=")):
        left, right = bars[k].ax.get_position(), axes[k + 1].get_position()
        fig.text((left.x1 + right.x0) / 2, (right.y0 + right.y1) / 2, symbol, ha="center",
                 va="center", fontsize=14)
    save(fig, path)


def fig_selectivity(path):
    """Chapter 8: the selectivity co-field of the AMPA / glycine-NMDA antagonists."""
    import matplotlib.pyplot as plt
    from cmfqsar import CMFModel
    from cmfqsar import multitask as MT
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import AtomTable, FieldSet, build_fields

    ds = load_dataset(os.path.join(ROOT, "examples", "selectivity", "ampa_glycine.sdf"),
                      activity_prop="pKi_AMPA")
    gly = np.array([float(m.GetProp("pKi_glycine")) for m in ds.mols])
    fs = FieldSet.create(alpha=0.5)
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = model.training_kernel(AtomTable.from_molecules(mf))
    obs = MT.Observations.from_columns(np.column_stack([ds.activity, gly]), names=("AMPA", "glycine"))
    rho, lam, _, _ = MT.search(K, obs, rhos=[0.0, 0.5, 0.9], lams=[0.01, 0.1, 1.0])
    fit = MT.fit(K, obs, rho=rho, lam=lam)
    model.coef = fit.selectivity_coefficients(0, 1)
    model.bias = float(fit.bias[0] - fit.bias[1])

    frame = plane_of(mf)
    sel = ds.activity - gly
    ref = mf[int(np.nanargmax(sel))]                      # the most AMPA-selective compound
    xy = np.vstack([project(frame, m.coords) for m in mf])
    lo, hi = xy.min(axis=0) - 2.0, xy.max(axis=0) + 2.0
    xs, ys = np.arange(lo[0], hi[0], 0.15), np.arange(lo[1], hi[1], 0.15)
    pts, shape = plane_points(frame, xs, ys)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    for ax, name in zip(axes, ("electrostatic", "steric")):
        f = fs.names.index(name)
        Z = cofield_slice(model, f, pts).reshape(shape)
        im = signed_map(ax, Z, xs, ys)
        draw_skeleton(ax, ref, frame)
        ax.set_title(f"{fs.specs[f].label}: AMPA minus glycine")
        axes_in_angstrom(ax, ylabel=ax is axes[0])
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03).ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, path)


def fig_errorpred(path):
    """Chapter 7: the risk-coverage efficiency of the three error estimates, set by set."""
    import matplotlib.pyplot as plt
    names = [("GP variance", "GP variance"), ("distance to training set", "distance to the training set"),
             ("domain dissimilarity", "domain dissimilarity")]
    sets, values = [], {label: [] for _, label in names}
    for f in sorted(glob.glob(os.path.join(RESULTS, "errorpred", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        sets.append(d["set"])
        for key, label in names:
            values[label].append(np.mean([r["scores"][key]["efficiency"] for r in d["rows"]]))
    fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 2.6))
    x = np.arange(len(sets))
    markers = ("o", "s", "^")
    for k, (label, v) in enumerate(values.items()):
        ax.plot(x + (k - 1) * 0.18, v, markers[k], ms=4.5, label=label, color=f"C{k}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, sets)
    ax.set_ylabel("risk–coverage efficiency")
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    save(fig, path)


def _width(rho):
    return 1.0 / np.sqrt(float(rho))


def fig_envelope(path):
    """Chapter 11: q² against the envelope width for the shift and the pKa models."""
    import matplotlib.pyplot as plt
    shifts = json.load(open(os.path.join(RESULTS, "local", "shifts.json"), encoding="utf-8"))
    pka = json.load(open(os.path.join(RESULTS, "local", "pka.json"), encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5))
    styles = {"local field": ("o-", "C0", "field under the envelope"),
              "local difference field": ("s--", "C3", "difference field, acid minus anion")}

    def curve(ax, data, kind, what):
        rows = sorted((r for r in data["rows"] if r["kind"] == kind), key=lambda r: r["rho"])
        if not rows:
            return
        marker, colour, label = styles[kind]
        ax.plot([_width(r["rho"]) for r in rows], [r[what] for r in rows], marker, color=colour,
                ms=4, lw=1.1, label=label)

    for ax, data, title in ((axes[0], shifts, "¹³C shifts, 400 molecules"),
                            (axes[1], pka, "pKa, 912 acids")):
        for kind in styles:
            curve(ax, data, kind, "q2")
        ax.axhline(data["reference"]["q2"], color="gray", lw=1.0, ls=":", label="atom-centred counts")
        ax.set_xscale("log")
        ax.set_xlabel("envelope width 1/√ρ / Å")
        ax.set_title(title)
        ax.grid(True, lw=0.4, alpha=0.5)
    axes[0].set_ylabel("q²")
    axes[0].set_ylim(0.84, 1.0)
    axes[1].set_ylim(-0.05, 0.65)
    # the third panel: the differences against the counts with their intervals, pKa
    ax = axes[2]
    for kind in styles:
        rows = sorted((r for r in pka["rows"] if r["kind"] == kind), key=lambda r: r["rho"])
        marker, colour, label = styles[kind]
        w = np.array([_width(r["rho"]) for r in rows])
        d = np.array([r["difference"] for r in rows])
        lo = d - np.array([r["low"] for r in rows])
        hi = np.array([r["high"] for r in rows]) - d
        ax.errorbar(w, d, yerr=[lo, hi], fmt=marker[0], color=colour, ms=4, lw=1.0, capsize=2,
                    label=label)
    ax.axhline(0, color="gray", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("envelope width 1/√ρ / Å")
    ax.set_ylabel("Δq² against the counts")
    ax.set_title("pKa, gain over the counts")
    ax.grid(True, lw=0.4, alpha=0.5)
    widths = sorted({_width(r["rho"]) for r in pka["rows"]} | {_width(r["rho"]) for r in shifts["rows"]})
    for ax in axes:
        ax.set_xticks(widths, [f"{w:.2g}" for w in widths], minor=False)
        ax.set_xticks([], minor=True)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, path)


def paper_table2():
    """Table 2 of the 2013 paper, read from the benchmark script without importing it."""
    src = open(os.path.join(ROOT, "benchmarks", "run_sutherland.py"), encoding="utf-8").read()
    m = re.search(r"^PAPER = (\{.*?^\})", src, flags=re.S | re.M)
    return ast.literal_eval(m.group(1))


def fig_sutherland(path):
    """Chapter 12: q² and R²pred on the eight Sutherland sets, the paper against this program."""
    import matplotlib.pyplot as plt
    paper = paper_table2()
    sets = list(paper)
    runs = {"paper protocol": "paper", "program defaults": "default"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    x = np.arange(len(sets))
    for ax, stat, col, title in ((axes[0], "q2", 0, "q², cross-validation"),
                                 (axes[1], "r2_pred", 2, "R²pred, the published test set")):
        ax.plot(x - 0.2, [paper[s][col] for s in sets], "o", ms=5, color="black",
                label="Table 2 of [2]")
        for k, (label, tag) in enumerate(runs.items()):
            vals = []
            for s in sets:
                f = os.path.join(RESULTS, f"{s}_{tag}.json")
                d = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
                vals.append(d.get(stat, np.nan))
            ax.plot(x + 0.2 * k, vals, ("s", "^")[k], ms=4.5, color=f"C{k}",
                    label=f"this program, {label}")
        ax.set_xticks(x, sets, rotation=45)
        ax.set_title(title)
        ax.grid(True, axis="y", lw=0.4, alpha=0.5)
        ax.axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel("score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, path)


def fig_scale(path):
    """Chapter 11: the learning curve of the local shift model, and what a finer descriptor gains."""
    import matplotlib.pyplot as plt
    d = json.load(open(os.path.join(RESULTS, "local_scale", "curve.json"), encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.plot(d["molecules"], d["error"], "o-", color="C0", ms=4, lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks(d["molecules"], [str(m) for m in d["molecules"]])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("molecules in the training set")
    ax.set_ylabel("cross-validated error / ppm")
    ax.set_title(f"more data, bins every {d['step']:g} Å")
    ax.grid(True, lw=0.4, alpha=0.5)
    ax = axes[1]
    steps = [d["step"]] + [f["step"] for f in d["finer"]]
    errors = [d["error"][-1]] + [f["error"] for f in d["finer"]]
    x = np.arange(len(steps))
    ax.plot(x, errors, "s-", color="C3", ms=4, lw=1.2)
    for xi, e in zip(x, errors):
        ax.annotate(f"{e:.2f}", (xi, e), textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=7)
    ax.set_xticks(x, [f"{s:g}" for s in steps])
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_xlabel("width of the distance bins / Å")
    ax.set_title(f"a finer descriptor, {d['n_molecules']} molecules")
    ax.grid(True, lw=0.4, alpha=0.5)
    axes[0].set_ylim(min(errors) - 0.6, max(d["error"]) + 0.5)
    fig.tight_layout()
    save(fig, path)


def sutherland_model(name, task, method, params, alpha=0.5, labels=None):
    """A model on one Sutherland set with the charges of the file, and the frame to draw it in."""
    from cmfqsar import CMFModel
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    ds = load_dataset(os.path.join(ROOT, "examples", "sutherland", f"{name}.sdf"))
    fs = FieldSet.create(alpha=alpha, charge_model="file")
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    train = AtomTable.from_molecules(mf)
    y = ds.activity if labels is None else labels(ds.activity)
    model = CMFModel(fieldset=fs, task=task, method=method, normalize=True, params=params)
    model.fit(train, y, cv_folds=5)
    frame = plane_of(mf)
    xy = np.vstack([project(frame, m.coords) for m in mf])
    lo, hi = xy.min(axis=0) - 2.0, xy.max(axis=0) + 2.0
    xs, ys = np.arange(lo[0], hi[0], 0.15), np.arange(lo[1], hi[1], 0.15)
    return {"ds": ds, "fs": fs, "mf": mf, "model": model, "frame": frame, "xs": xs, "ys": ys}


def two_cofields(c, ref, path, titles, suptitle):
    """The electrostatic and the steric co-field of a model in its plane, with one molecule drawn."""
    import matplotlib.pyplot as plt
    fs, model, frame, xs, ys = c["fs"], c["model"], c["frame"], c["xs"], c["ys"]
    pts, shape = plane_points(frame, xs, ys)
    # the height follows the shape of the series, so a long flat series leaves no empty band
    height = min(3.4, max(2.3, 0.5 * WIDTH * (ys[-1] - ys[0]) / (xs[-1] - xs[0]) + 1.0))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, height))
    for ax, name, title in zip(axes, ("electrostatic", "steric"), titles):
        f = fs.names.index(name)
        Z = cofield_slice(model, f, pts).reshape(shape)
        im = signed_map(ax, Z, xs, ys)
        draw_skeleton(ax, ref, frame)
        ax.set_title(title)
        axes_in_angstrom(ax, ylabel=ax is axes[0])
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03).ax.tick_params(labelsize=7)
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    save(fig, path)


def fig_oneclass(path):
    """Chapter 9: the co-field of a one-class SVM fitted on the thrombin inhibitors alone."""
    nu = 0.1
    c = sutherland_model("THR", "oneclass", "ocsvm", {"nu": nu})
    ds, mf, model = c["ds"], c["mf"], c["model"]
    ref = mf[int(np.argmax(ds.activity))]
    st = model.stats
    print(f"  one-class SVM on THR: {len(mf)} actives, nu = {nu}, {st.n_support} support vectors, "
          f"{st.inlier_fraction:.2f} of the training set inside; drawn with {ref.name}")
    two_cofields(c, ref, path, ("electrostatic co-field", "steric co-field"),
                 f"one-class SVM of the {len(mf)} thrombin inhibitors, ν = {nu:g}, "
                 f"{st.n_support} support vectors")


def fig_kfda(path):
    """Chapter 7: the co-field of the kernel Fisher discriminant on AChE, active above pIC50 6.84."""
    threshold, lam = 6.84, 0.01
    c = sutherland_model("AChE", "classification", "kfda", {"lam": lam},
                         labels=lambda a: np.where(a >= threshold, 1.0, -1.0))
    ds, mf, model = c["ds"], c["mf"], c["model"]
    ref = mf[int(np.argmax(ds.activity))]
    cv = model.stats.cv or {}
    auc = float(cv.get("auc", float("nan")))
    n_act = int(np.sum(ds.activity >= threshold))
    print(f"  kernel FDA on AChE: {n_act} of {len(mf)} active above {threshold}, lambda = {lam}, "
          f"cross-validated AUC {auc:.3f}; drawn with {ref.name}")
    two_cofields(c, ref, path, ("electrostatic co-field", "steric co-field"),
                 f"kernel Fisher discriminant of AChE: {n_act} actives against "
                 f"{len(mf) - n_act} inactives, λ = {lam:g}, cross-validated AUC {auc:.2f}")


def fig_parent(path):
    """Chapter 2: the parent of the worked example, N-phenylbenzamide, drawn by RDKit with the
    positions the series substitutes marked: 3 and 4 on the ring of the acid, 3' and 4' on the
    ring of the amine, as the molecule names of the file count them."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    sup = Chem.SDMolSupplier(os.path.join(ROOT, "examples", "benzamides_synthetic.sdf"))
    parent = next(m for m in sup if m.GetProp("_Name") == "parent")
    mol = Chem.RemoveHs(parent)
    rdDepictor.SetPreferCoordGen(False)
    rdDepictor.Compute2DCoords(mol)
    # the ipso carbon of each ring: bonded to the carbonyl carbon (unprimed positions) or to the
    # nitrogen (primed positions); the other positions are counted along the ring from it
    amide = mol.GetSubstructMatch(Chem.MolFromSmarts("O=C(N)"))
    carbonyl_c, nitrogen = amide[1], amide[2]
    dist = Chem.GetDistanceMatrix(mol)
    for anchor, prime in ((carbonyl_c, ""), (nitrogen, "'")):
        ipso = next(a.GetIdx() for a in mol.GetAtomWithIdx(anchor).GetNeighbors()
                    if a.GetIsAromatic())
        ring = next(r for r in mol.GetRingInfo().AtomRings() if ipso in r)
        for number, steps in (("3", 2), ("4", 3)):
            atom = next(i for i in ring if int(dist[ipso, i]) == steps)
            mol.GetAtomWithIdx(atom).SetProp("atomNote", number + prime)
    d = rdMolDraw2D.MolDraw2DSVG(320, 150)
    opts = d.drawOptions()
    opts.annotationFontScale = 0.9
    opts.clearBackground = False
    d.DrawMolecule(mol)
    d.FinishDrawing()
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(d.GetDrawingText())
    print(f"  parent: {Chem.MolToSmiles(mol)}, positions 3, 4 on the acid ring and 3', 4' on "
          f"the amine ring")


ELEMENT_COLOURS = {"Li": "#8e6fd8", "O": "#d62728", "P": "#ff7f0e", "B": "#e8a0b0"}
HOP_RHO, HOP_R_MAX = 0.25, 6.0          # the public hop model: envelope width 2 A, reach 6 A


def phosphate_hop():
    """The environment of the Li hop of Li3PO4 the interface study computed with DFT, site 34
    into site 0, in its hop frame: z along the hop, the midpoint at the origin."""
    if "phosphate_hop" in _CACHE:
        return _CACHE["phosphate_hop"]
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    import run_hops as rh
    from cmfqsar.hops import framed_environments, bottleneck
    hs = rh.load_hosts()
    fs = rh.fieldset_for(hs, "file")
    crystals = hs.crystals(fs)
    row = next(i for i, h in enumerate(hs.hops)
               if h.structure == 0 and h.a == 34 and h.b == 0 and tuple(h.offset) == (0, 0, 0))
    hop = hs.hops[row]
    env = framed_environments(hs, crystals, rows=[row], rho=HOP_RHO, r_max=HOP_R_MAX)[0]
    gap, where, ring = bottleneck(env, hop.vector)
    out = {"hs": hs, "fs": fs, "hop": hop, "env": env, "gap": gap, "ring": ring}
    _CACHE["phosphate_hop"] = out
    print(f"  Li3PO4 hop {hop.a} -> {hop.b}: length {hop.length:.2f} A, {env.n_atoms - 1} atoms "
          f"within {HOP_R_MAX:g} A, bottleneck {gap:.2f} A ({' '.join(ring)}), "
          f"bond-valence barrier {hop.y:.2f} eV")
    return out


def averaged_field(env, f, b, zs, rs):
    """The field `f` of a hop environment averaged about the hop axis, on the (z, r) grid:
    each Gaussian of width `b` becomes exp(-b/2 [(z - z_l)^2 + (r - rho_l)^2]) e^-x I_0(x) with
    x = b r rho_l, the same closed form the kernel of the hop model uses."""
    from scipy.special import i0e
    c = np.asarray(env.coords, float)
    rho_l, z_l = np.hypot(c[:, 0], c[:, 1]), c[:, 2]
    w = np.asarray(env.weights, float)[:, f]
    Z, R = np.meshgrid(zs, rs, indexing="ij")
    out = np.zeros(Z.shape)
    for k in np.nonzero(np.abs(w) > 1e-12)[0]:
        out += w[k] * np.exp(-0.5 * b * ((Z - z_l[k]) ** 2 + (R - rho_l[k]) ** 2)) * i0e(b * R * rho_l[k])
    return out


def fig_hop(path):
    """Chapter 11: a hop as the sample. Left, the environment of one hop of Li3PO4 cut from the
    cell under the envelope and seen in the hop frame; right, its electrostatic field averaged
    about the hop axis, which is the view the rotation-averaged kernel and the co-field take."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    p = phosphate_hop()
    env, hop, fs = p["env"], p["hop"], p["fs"]
    c = np.asarray(env.coords, float)[1:]            # atom 0 is the marker at the midpoint
    symbols = list(env.symbols)[1:]
    radii = np.asarray(env.radii, float)[1:]
    d = np.linalg.norm(c, axis=1)
    envelope = np.exp(-HOP_RHO * d * d)
    half = 0.5 * hop.length
    colours = [ELEMENT_COLOURS.get(s, "0.5") for s in symbols]
    # the ring of the bottleneck: the atoms nearest to the path at its narrowest point
    pts = np.linspace(-half, half, 41)
    dist = np.sqrt((c[:, 0] ** 2 + c[:, 1] ** 2)[None, :] + (pts[:, None] - c[:, 2][None, :]) ** 2)
    k = int(np.argmin(dist.min(axis=1)))
    ring = np.where(dist[k] <= dist[k].min() + 0.3)[0]

    fig = plt.figure(figsize=(WIDTH, 3.1))
    ax1 = fig.add_axes([0.06, 0.08, 0.42, 0.84])
    ax2 = fig.add_axes([0.55, 0.14, 0.37, 0.70])
    cax = fig.add_axes([0.935, 0.30, 0.012, 0.38])
    # left: the projection onto the x-z plane of the frame, depth-sorted, faded by the envelope
    order = np.argsort(c[:, 1])
    for i in order:
        ax1.add_patch(Circle((c[i, 2], c[i, 0]), 0.32 * radii[i], facecolor=colours[i],
                             edgecolor="black" if i in ring else "none",
                             linewidth=1.2 if i in ring else 0,
                             alpha=0.15 + 0.85 * envelope[i], zorder=2 + envelope[i]))
    for level in (0.5, 0.1, 0.01):
        rad = np.sqrt(-np.log(level) / HOP_RHO)
        ax1.add_patch(Circle((0, 0), rad, fill=False, edgecolor="0.4", linestyle=":", lw=0.8, zorder=1))
        ax1.text(-rad * 0.72, rad * 0.72, f"{level:g}", fontsize=6.5, color="0.35", ha="center",
                 va="center", zorder=5)
    for z in (-half, half):
        ax1.add_patch(Circle((z, 0), 0.55, fill=False, edgecolor=ELEMENT_COLOURS["Li"],
                             linestyle="--", lw=1.2, zorder=6))
    ax1.plot([-half, half], [0, 0], color="black", lw=1.0, ls="--", zorder=5)
    ax1.plot([0], [0], marker="x", color="black", ms=6, zorder=7)
    ax1.annotate("site a", (-half, 0), xytext=(-half, -1.1), fontsize=7, ha="center")
    ax1.annotate("site b", (half, 0), xytext=(half, -1.1), fontsize=7, ha="center")
    ax1.annotate("z", (3.0, 0), xytext=(5.6, 0.15), fontsize=8, ha="center", style="italic")
    ax1.set_xlim(-HOP_R_MAX - 0.4, HOP_R_MAX + 0.4)
    ax1.set_ylim(-HOP_R_MAX - 0.4, HOP_R_MAX + 0.4)
    ax1.set_aspect("equal")
    ax1.set_xlabel("z along the hop / Å")
    ax1.set_ylabel("x / Å")
    ax1.set_title(f"the environment of a hop in Li$_3$PO$_4$\n{env.n_atoms - 1} atoms within "
                  f"{HOP_R_MAX:g} Å, envelope width {1 / np.sqrt(HOP_RHO):g} Å", fontsize=8)
    ax1.tick_params(length=2)
    # right: the electrostatic field averaged about the axis, in (z, r)
    f = fs.names.index("electrostatic")
    zs, rs = np.arange(-3.5, 3.51, 0.1), np.arange(0.0, 4.01, 0.1)
    X = averaged_field(env, f, float(fs.specs[f].alpha), zs, rs)
    vmax = np.abs(X).max()
    im = ax2.imshow(X.T, origin="lower", extent=(zs[0], zs[-1], rs[0], rs[-1]), cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, aspect="equal", interpolation="bilinear")
    rho_l = np.hypot(c[:, 0], c[:, 1])
    inside = (np.abs(c[:, 2]) <= 3.5) & (rho_l <= 4.0)
    ax2.scatter(c[inside, 2], rho_l[inside], s=4 + 26 * envelope[inside],
                color=np.array(colours)[inside], edgecolor="black", linewidth=0.3, zorder=4)
    ax2.plot([-half, half], [0, 0], color="black", lw=1.0, ls="--", zorder=3)
    ax2.plot([-half, half], [0, 0], marker="o", mfc="white", mec=ELEMENT_COLOURS["Li"], ms=6,
             lw=0, zorder=5, clip_on=False)
    ax2.set_xlabel("z along the hop / Å")
    ax2.set_ylabel("r from the axis / Å")
    ax2.set_title("electrostatic field, averaged\nabout the hop axis", fontsize=8)
    ax2.tick_params(length=2)
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=6.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=ELEMENT_COLOURS[s], label=s, ms=6)
               for s in ("Li", "O", "P")]
    ax1.legend(handles=handles, loc="upper right", frameon=False, fontsize=7, handletextpad=0.2)
    save(fig, path)


def fig_hopcofield(path):
    """Chapter 11: the co-field of the public hop model as a profile in (z, r), for the
    electrostatic field, the steric field and the oxygen indicator, from
    benchmarks/results/hops/profile_fine.json."""
    import matplotlib.pyplot as plt
    d = json.load(open(os.path.join(RESULTS, "hops", "profile_fine.json"), encoding="utf-8"))
    zs, rs = np.asarray(d["z"], float), np.asarray(d["r"], float)
    panels = [("electrostatic", "electrostatic co-field"), ("steric", "steric co-field"),
              ("cif:O", "oxygen indicator co-field")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.05))
    for ax, (name, title) in zip(axes, panels):
        V = np.asarray(d["fields"][name]["value"], float)
        SD = np.asarray(d["fields"][name]["sd"], float)
        vmax = np.abs(V).max()
        im = ax.imshow(V.T, origin="lower", extent=(zs[0], zs[-1], rs[0], rs[-1]), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="equal", interpolation="bilinear")
        # inside the dashed line the co-field is more than two standard deviations from zero
        ax.contour(zs, rs, (np.abs(V) / np.maximum(SD, 1e-12)).T, levels=[2.0], colors="black",
                   linewidths=0.7, linestyles="--")
        ax.axvline(0, color="0.3", lw=0.5)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("z along the hop / Å")
        if ax is axes[0]:
            ax.set_ylabel("r from the axis / Å")
        ax.tick_params(length=2)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03).ax.tick_params(labelsize=6.5)
    fig.suptitle(f"co-field of the hop model fitted to {d['n_samples']} hops of the public set, "
                 f"envelope width {1 / np.sqrt(d['rho']):g} Å, q² = {d['q2']:.2f}" + chr(10)
                 + "inside the dashed line the value is more than two standard deviations from zero",
                 fontsize=8, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    print(f"  hop co-field: rho {d['rho']}, lambda {d['lam']}, q2 {d['q2']:.3f}, sigma {d['sigma']:.3f}")
    save(fig, path)


def fig_isosurface(path):
    """Chapter 8: the same co-field drawn both ways: as the two isosurfaces at plus and minus
    one level, which is how the application and the literature draw a field, and as the cut of
    one plane through it, which is how the maps of this manual draw it."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    from skimage import measure
    from cmfqsar.cofields import evaluate_gaussians
    s, c = steroids(), cofield_model()
    fs, mf, model, frame = c["fs"], c["mf"], c["model"], c["frame"]
    ref = mf[s["train_idx"][int(np.argmax(s["y"]))]]      # the most active training molecule
    f = fs.names.index("electrostatic")
    origin, e1, e2, e3 = frame
    step = 0.3
    xs = np.arange(c["xs"][0], c["xs"][-1] + 1e-9, step)
    ys = np.arange(c["ys"][0], c["ys"][-1] + 1e-9, step)
    zs = np.arange(-4.0, 4.0 + 1e-9, step)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = origin + X.ravel()[:, None] * e1 + Y.ravel()[:, None] * e2 + Z.ravel()[:, None] * e3
    centres, coeffs, widths = model.cofield_coefficients(f)
    vol = evaluate_gaussians(pts, centres, coeffs, widths).reshape(X.shape)
    vol *= float(fs.specs[f].h) * float(model.scale[f])
    vmax = float(np.abs(vol).max())
    fraction = 0.3                                          # the iso level of the application
    level = fraction * vmax
    # the molecule in the coordinates of the frame
    local = (np.asarray(ref.coords, float) - origin) @ np.stack([e1, e2, e3]).T
    heavy = np.array([sym != "H" for sym in ref.symbols])
    segments = [(local[a], local[b]) for a, b in ref.bonds if heavy[a] and heavy[b]]

    fig = plt.figure(figsize=(WIDTH, 3.0))
    ax3 = fig.add_axes([-0.06, -0.05, 0.54, 1.06], projection="3d")
    padded = np.pad(vol, 1, mode="constant", constant_values=0.0)
    for sign, colour in ((1, "#c0392b"), (-1, "#2e6db4")):
        verts, faces, _, _ = measure.marching_cubes(padded, level=sign * level, spacing=(step,) * 3)
        verts = verts + np.array([xs[0], ys[0], zs[0]]) - step
        poly = Poly3DCollection(verts[faces], facecolors=colour, alpha=0.6, shade=True)
        poly.set_edgecolor("none")
        ax3.add_collection3d(poly)
    ax3.add_collection3d(Line3DCollection(segments, colors="black", linewidths=1.1))
    ax3.scatter(local[heavy, 0], local[heavy, 1], local[heavy, 2], s=6, color="black",
                depthshade=False)
    plane = [[xs[0], ys[0], 0], [xs[-1], ys[0], 0], [xs[-1], ys[-1], 0], [xs[0], ys[-1], 0]]
    ax3.add_collection3d(Poly3DCollection([plane], facecolor="0.5", edgecolor="0.25", alpha=0.12,
                                          linewidths=0.8, linestyles="--"))
    ax3.set_xlim(xs[0], xs[-1])
    ax3.set_ylim(ys[0], ys[-1])
    ax3.set_zlim(zs[0], zs[-1])
    ax3.set_box_aspect((xs[-1] - xs[0], ys[-1] - ys[0], zs[-1] - zs[0]))
    ax3.view_init(elev=24, azim=-57)
    ax3.set_xticks(np.arange(-8, 9, 4))
    ax3.set_yticks(np.arange(-4, 5, 4))
    ax3.set_zticks(np.arange(-4, 5, 4))
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_tick_params(labelsize=6, pad=-2)
    ax3.set_xlabel("x / Å", fontsize=7, labelpad=-5)
    ax3.set_ylabel("y / Å", fontsize=7, labelpad=-5)
    ax3.set_zlabel("z / Å", fontsize=7, labelpad=-7)
    ax3.set_title(f"isosurfaces at ±{100 * fraction:.0f} % of the largest value\n"
                  "with the plane z = 0 marked", fontsize=8.5, y=0.97)
    # right: the plane cut, with the lines where the two isosurfaces meet the plane
    ax2 = fig.add_axes([0.645, 0.22, 0.29, 0.56])
    cax = fig.add_axes([0.945, 0.30, 0.012, 0.40])
    k0 = int(np.argmin(np.abs(zs)))
    cut = vol[:, :, k0].T
    im = signed_map(ax2, cut, xs, ys, vmax=vmax)
    ax2.contour(xs, ys, cut, levels=[-level, level], colors=["#2e6db4", "#c0392b"],
                linewidths=1.0, linestyles="--")
    draw_skeleton(ax2, ref, frame)
    axes_in_angstrom(ax2)
    ax2.set_title("the same co-field cut by the plane z = 0\ndashed: where the isosurfaces meet it",
                  fontsize=8)
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=6.5)
    print(f"  isosurface figure: electrostatic co-field of the steroid model on {vol.shape} points, "
          f"level ±{level:.3f} ({100 * fraction:.0f} % of {vmax:.3f}), drawn with {ref.name}")
    save(fig, path)


def rotation_matrix(axis, degrees):
    """The rotation by `degrees` about the unit vector `axis` (Rodrigues' formula)."""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    t = np.radians(degrees)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) * np.cos(t) + np.sin(t) * K + (1 - np.cos(t)) * np.outer(k, k)


def fig_autocorr(path):
    """Chapter 11: the autocorrelation channels of two steroids, and of one of them turned and
    moved, which leaves every channel where it was."""
    import copy
    import matplotlib.pyplot as plt
    from cmfqsar.invariant import AutocorrelationSpec, autocorrelation
    s = steroids()
    fs, mf, idx, y = s["fs"], s["mf"], s["train_idx"], s["y"]
    a, b = mf[idx[int(np.argmax(y))]], mf[idx[int(np.argmin(y))]]
    spec = AutocorrelationSpec()
    moved = copy.copy(a)
    moved.coords = np.asarray(a.coords, float) @ rotation_matrix([1, 2, 3], 117).T + [5.0, -3.0, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    for ax, name in zip(axes, ("electrostatic", "steric")):
        f = fs.names.index(name)
        for mol, colour in ((a, "C3"), (b, "C0")):
            ax.plot(spec.centres, autocorrelation(mol, spec, pairs=[(f, f)]), "o-", ms=3, lw=1.1,
                    color=colour, label=mol.name)
        ax.plot(spec.centres, autocorrelation(moved, spec, pairs=[(f, f)]), "--", lw=1.0,
                color="black", label=f"{a.name}, turned and moved")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel("distance d / Å")
        ax.set_title(f"{fs.specs[f].label} autocorrelation", fontsize=9)
        ax.tick_params(length=2)
    axes[0].set_ylabel("channel value $\\Phi_{ff}(d)$")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    print(f"  autocorrelation channels of {a.name} and {b.name}, step {spec.step} A, sigma {spec.sigma} A")
    save(fig, path)


def fig_periodic(path):
    """Chapter 11: the periodic autocorrelation of Li3PO4: the pair histogram, the uniform
    background and their difference, for the oxygen-oxygen and the lithium-oxygen pair."""
    import matplotlib.pyplot as plt
    from cmfqsar.periodic import PeriodicSpec, periodic_autocorrelation
    p = phosphate_hop()
    fs = p["fs"]
    crystal = p["hs"].crystals(fs)[0]
    fo, fli = fs.names.index("cif:O"), fs.names.index("cif:Li")
    raw, full = PeriodicSpec(background=False), PeriodicSpec(background=True)
    d = raw.centres
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    for ax, pair, title in zip(axes, ((fo, fo), (fli, fo)), ("oxygen–oxygen", "lithium–oxygen")):
        P = periodic_autocorrelation(crystal, raw, pairs=[pair])
        Phi = periodic_autocorrelation(crystal, full, pairs=[pair])
        ax.plot(d, P, "o-", ms=3, lw=1.1, color="0.4", label="pair histogram $P_{fg}(d)/N$")
        ax.plot(d, P - Phi, ":", lw=1.4, color="C1", label="uniform background")
        ax.plot(d, Phi, "o-", ms=3, lw=1.3, color="C0", label="difference $\\Phi_{fg}(d)$")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel("distance d / Å")
        ax.set_title(f"{title} pair in Li$_3$PO$_4$", fontsize=9)
        ax.tick_params(length=2)
    axes[0].set_ylabel("channel value per atom of the cell")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    print(f"  periodic autocorrelation of Li3PO4, {crystal.n_atoms} atoms, cutoff {raw.r_max} A, "
          f"step {raw.step} A")
    save(fig, path)


def fig_topomer(path):
    """Chapter 11: the topomer cut of two benzamides that share one substituent: the molecules
    with the core marked, and their substituents rebuilt in the topomer pose, where the shared
    group lands on the same atoms whichever molecule it came from."""
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from cmfqsar.dataio import load_dataset
    from cmfqsar.topomer import choose_core, substituents, topomer_pose
    ds = load_dataset(os.path.join(ROOT, "examples", "benzamides_synthetic.sdf"))
    core, how = choose_core(ds.mols)
    # the molecules carry explicit hydrogens, so a hydrogen on the core is a substituent too;
    # only the groups with a heavy atom are drawn
    table = [{Chem.MolToSmiles(f): f for f in substituents(m, core)
              if any(at.GetAtomicNum() > 1 for at in f.GetAtoms())} for m in ds.mols]

    def one_cut(frags):
        """True when every substituent hangs from one core atom, which the pose rule needs."""
        return all(sum(1 for at in f.GetAtoms() if at.GetAtomicNum() == 0) == 1
                   for f in frags.values())

    pick = None
    for i, fa in enumerate(table):
        for smi, frag in fa.items():
            heavy = sum(1 for at in frag.GetAtoms() if at.GetAtomicNum() > 1)
            if heavy < 3 or len(fa) < 2 or not one_cut(fa):
                continue
            for j, fb in enumerate(table):
                if j != i and smi in fb and set(fb) != set(fa) and len(fb) >= 2 and one_cut(fb):
                    pick = (i, j, smi)
                    break
            if pick:
                break
        if pick:
            break
    i, j, shared = pick
    print(f"  topomer: core by {how}; {ds.names[i]} and {ds.names[j]} share {shared}")
    palette = {}

    def colour_of(key):
        if key not in palette:
            palette[key] = f"C{len(palette)}"
        return palette[key]

    fig = plt.figure(figsize=(WIDTH, 4.1))
    top = [fig.add_axes([0.02, 0.56, 0.47, 0.40]), fig.add_axes([0.51, 0.56, 0.47, 0.40])]
    bottom = fig.add_axes([0.06, 0.07, 0.90, 0.42])
    for ax, k in zip(top, (i, j)):
        mol = Chem.RemoveHs(ds.mols[k])
        rdDepictor.Compute2DCoords(mol)
        xy = mol.GetConformer().GetPositions()[:, :2]
        match = mol.GetSubstructMatch(core)
        in_core = {a: n for n, a in enumerate(match)}
        # every atom outside the core belongs to the core atom its branch hangs from
        owner = {}
        for a0 in match:
            for nb in mol.GetAtomWithIdx(a0).GetNeighbors():
                if nb.GetIdx() in in_core:
                    continue
                stack = [nb.GetIdx()]
                while stack:
                    a = stack.pop()
                    if a in owner or a in in_core:
                        continue
                    owner[a] = in_core[a0]
                    stack += [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()]
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in in_core and b in in_core:
                style = dict(color="0.55", lw=2.2)
            elif a in in_core or b in in_core:
                style = dict(color=colour_of(owner.get(a, owner.get(b))), lw=1.4, ls="--")
            else:
                style = dict(color=colour_of(owner[a]), lw=1.6)
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], solid_capstyle="round", **style)
        for a in range(mol.GetNumAtoms()):
            sym = mol.GetAtomWithIdx(a).GetSymbol()
            if sym != "C":
                ax.text(xy[a, 0], xy[a, 1], sym, fontsize=7, ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.05", fc="white", ec="none"))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(ds.names[k], fontsize=9)
    # the substituents in their topomer poses, each attachment point in its own region
    keys = sorted({at.GetIsotope() for fr in list(table[i].values()) + list(table[j].values())
                   for at in fr.GetAtoms() if at.GetAtomicNum() == 0})
    slot = {key: 5.0 * n for n, key in enumerate(keys)}
    for k, ls, lw in ((i, "-", 1.6), (j, "--", 1.6)):
        for frag in table[k].values():
            key = next(at.GetIsotope() for at in frag.GetAtoms() if at.GetAtomicNum() == 0)
            posed = topomer_pose(frag)
            xyz = posed.GetConformer().GetPositions()
            # an oblique view: x across, and y and z both tilted into the vertical
            uv = np.column_stack([xyz[:, 0] + slot[key], 0.8 * xyz[:, 1] + 0.6 * xyz[:, 2]])
            heavy = [at.GetIdx() for at in posed.GetAtoms() if at.GetAtomicNum() > 1]
            dummy = next(at.GetIdx() for at in posed.GetAtoms() if at.GetAtomicNum() == 0)
            colour = colour_of(key)
            for bond in posed.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a in heavy and b in heavy:
                    bottom.plot(uv[[a, b], 0], uv[[a, b], 1], ls=ls, lw=lw, color=colour,
                                solid_capstyle="round")
                elif dummy in (a, b) and (a in heavy or b in heavy):
                    bottom.plot(uv[[a, b], 0], uv[[a, b], 1], ls=":", lw=1.0, color="0.5")
            bottom.plot([uv[dummy, 0]], [uv[dummy, 1]], marker="o", mfc="white", mec=colour,
                        ms=7, lw=0)
    for key in keys:
        bottom.text(slot[key], -2.9, f"core atom {key}", ha="center", fontsize=7.5,
                    color=colour_of(key))
    bottom.set_aspect("equal")
    bottom.set_xlim(min(slot.values()) - 3.0, max(slot.values()) + 3.0)
    bottom.set_ylim(-3.3, 2.8)
    bottom.axis("off")
    bottom.set_title(f"the substituents rebuilt by the rule: {ds.names[i]} solid, {ds.names[j]} "
                     "dashed; open circle: the open valence, along −x; an oblique view", fontsize=8.5)
    save(fig, path)


def fig_confidence(path):
    """Chapter 5: the posterior variance of the steroid model for one test molecule turned away
    from its pose about two axes, with the prediction beside it."""
    import matplotlib.pyplot as plt
    from cmfqsar.varalign import VarianceAligner
    s, c = steroids(), cofield_model()
    ds, mf, model, frame = s["ds"], c["mf"], c["model"], c["frame"]
    i = ds.names.index("19-norprogesterone")
    mol, observed = mf[i], float(ds.activity[i])
    va = VarianceAligner(model, max_rmsd=None)
    coords = np.asarray(mol.coords, float)
    heavy = np.array([sym != "H" for sym in mol.symbols])
    centre = coords[heavy].mean(axis=0)
    angles = np.arange(-180, 181, 5)
    origin, e1, e2, e3 = frame
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    for axis, label, colour in ((e3, "about the normal of the plane", "C0"),
                                (e1, "about the long axis", "C3")):
        turned = [(coords - centre) @ rotation_matrix(axis, a).T + centre for a in angles]
        var = [va.variance(mol, x) for x in turned]
        pred = [va.predict(mol, x) for x in turned]
        axes[0].plot(angles, var, lw=1.3, color=colour, label=label)
        axes[1].plot(angles, pred, lw=1.3, color=colour, label=label)
        print(f"  confidence: {mol.name} {label}: variance {min(var):.3f} at "
              f"{angles[int(np.argmin(var))]:+d} deg, {max(var):.3f} at most")
    axes[1].axhline(observed, color="black", lw=0.8, ls=":", label="observed")
    for ax in axes:
        ax.axvline(0, color="0.6", lw=0.6)
        ax.set_xlabel("rotation from the aligned pose / degrees")
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.tick_params(length=2)
    axes[0].set_ylabel("posterior variance")
    axes[1].set_ylabel("predicted pK")
    axes[0].set_title(f"{mol.name}: the model's confidence", fontsize=9)
    axes[1].set_title("and its prediction", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, path)


def fig_laplacian(path):
    """Chapter 9: the neighbour graph of the Laplacian-regularised ridge over the 31 steroids,
    in kernel-PCA coordinates: training molecules coloured by activity, test molecules as the
    unlabelled ones."""
    import matplotlib.pyplot as plt
    from cmfqsar import CMFModel
    from cmfqsar.fields import AtomTable
    from cmfqsar.gtm import KernelPCA
    from cmfqsar.spectral import knn_laplacian
    s, c = steroids(), cofield_model()
    ds, mf = s["ds"], c["mf"]
    neighbours = 5
    K = CMFModel(fieldset=c["fs"], method="krr", normalize=True).training_kernel(
        AtomTable.from_molecules(mf))
    L = knn_laplacian(K, neighbours=neighbours)
    A = np.abs(L) > 1e-12
    np.fill_diagonal(A, False)
    xy = KernelPCA(n_components=2).fit(K).transform()
    train = np.zeros(len(mf), bool)
    train[list(s["train_idx"])] = True
    fig, ax = plt.subplots(figsize=(WIDTH * 0.72, 3.3))
    for a in range(len(mf)):
        for b in range(a + 1, len(mf)):
            if A[a, b] or A[b, a]:
                ax.plot(xy[[a, b], 0], xy[[a, b], 1], color="0.75", lw=0.6, zorder=1)
    sc = ax.scatter(xy[train, 0], xy[train, 1], c=ds.activity[train], cmap="viridis", s=36,
                    edgecolor="black", linewidth=0.4, zorder=3, label="labelled (training set)")
    ax.scatter(xy[~train, 0], xy[~train, 1], facecolor="white", edgecolor="black", s=36,
               linewidth=0.8, zorder=3, label="unlabelled (test set)")
    ax.set_xlabel("first kernel principal component")
    ax.set_ylabel("second")
    ax.set_title(f"the neighbour graph of the 31 steroids, {neighbours} neighbours each", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.tick_params(length=2)
    fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.03, label="pK").ax.tick_params(labelsize=7)
    fig.tight_layout()
    print(f"  Laplacian graph: {int(np.sum(np.triu(A | A.T, 1)))} edges among {len(mf)} molecules")
    save(fig, path)


def fig_probe(path):
    """Chapter 7: three GRID-like probe fields of cortisol in the mean plane of the steroid
    series, beside the built-in hydrogen-bond acceptor field."""
    import matplotlib.pyplot as plt
    from cmfqsar.fields import FieldSet, build_fields
    s, c = steroids(), cofield_model()
    ds, frame, xs, ys = s["ds"], c["frame"], c["xs"], c["ys"]
    i = ds.names.index("cortisol")
    probes = ("water", "carbonyl", "amine")
    fs = FieldSet.create(alpha=1.0, probes=list(probes))
    mol = build_fields(ds.mols[i], fs, ds.names[i])
    # read 2 A above the mean plane: in the plane itself every probe sits at the bottom of its
    # well on top of the atoms, and the map is one flat blob
    height = 2.0
    pts, shape = plane_points(frame, xs, ys, offset=height)
    panels = [(f"probe:{p}", f"{p} probe") for p in probes] + [("hba", "H-bond acceptor field")]
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.75))
    for ax, (name, title) in zip(axes, panels):
        f = fs.names.index(name)
        Z = field_slice(mol, fs.specs[f], f, pts).reshape(shape)
        if fs.specs[f].is_probe:
            # a probe field is never positive, so the scale runs from its deepest value to zero
            im = ax.imshow(Z, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="Blues_r",
                           vmin=Z.min(), vmax=0.0, aspect="equal", interpolation="bilinear")
        else:
            im = signed_map(ax, Z, xs, ys)
        draw_skeleton(ax, mol, frame)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03).ax.tick_params(labelsize=6)
    fig.suptitle(f"cortisol, {height:g} Å above the mean plane: what a probe would gain at each "
                 "point (darker: more), and a built-in field", fontsize=8, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    print(f"  probe fields of {mol.name}: " + ", ".join(
        f"{p} {field_slice(mol, fs.specs[fs.names.index('probe:' + p)], fs.names.index('probe:' + p), pts).min():+.2f}"
        for p in probes))
    save(fig, path)


def fig_chemspace(path):
    """Chapter 8: the two projections of the steroid set: the GTM activity landscape with the
    molecules on it, and the kernel-PCA score plot the GTM is fitted on."""
    import matplotlib.pyplot as plt
    from cmfqsar.fields import AtomTable
    from cmfqsar.gtm import CMFMap
    s, c = steroids(), cofield_model()
    ds, mf, fs = s["ds"], c["mf"], c["fs"]
    train, test = list(s["train_idx"]), list(ds.test_indices)
    # ten components, a 15 by 15 grid and three wide basis functions a side: twenty components
    # on 21 molecules leave the GTM a singular system, and sharper responsibilities an empty map
    cmap = CMFMap(n_components=10, grid=15, rbf=3, rbf_width=1.5, n_iter=100, seed=0).fit(
        AtomTable.from_molecules([mf[i] for i in train]), fs)
    # every node that carries any responsibility gets a value; with 21 molecules the default
    # density floor would leave most of the map empty
    land = cmap.as_image(cmap.activity_landscape(s["y"], min_density=1e-9))
    coords_test, _ = cmap.project(AtomTable.from_molecules([mf[i] for i in test]))
    scores_train = cmap.scores()
    scores_test = cmap.kpca.transform(cmap._kernel(AtomTable.from_molecules([mf[i] for i in test]),
                                                   cmap.train_))[:, :2]
    ratio = cmap.kpca.explained_variance_ratio
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    ax = axes[0]
    im = ax.imshow(land.T, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis",
                   interpolation="bilinear")
    ax.scatter(cmap.coords_[:, 0], cmap.coords_[:, 1], c=s["y"], cmap="viridis", s=30,
               edgecolor="black", linewidth=0.5, vmin=land[np.isfinite(land)].min(),
               vmax=land[np.isfinite(land)].max(), label="training")
    ax.scatter(coords_test[:, 0], coords_test[:, 1], facecolor="white", edgecolor="black", s=30,
               linewidth=0.8, label="test, projected")
    ax.set_title("GTM activity landscape", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, label="pK").ax.tick_params(labelsize=7)
    ax = axes[1]
    ax.scatter(scores_train[:, 0], scores_train[:, 1], c=s["y"], cmap="viridis", s=30,
               edgecolor="black", linewidth=0.5)
    ax.scatter(scores_test[:, 0], scores_test[:, 1], facecolor="white", edgecolor="black", s=30,
               linewidth=0.8)
    ax.set_xlabel(f"component 1 ({100 * ratio[0]:.0f} % of the variance)")
    ax.set_ylabel(f"component 2 ({100 * ratio[1]:.0f} %)")
    ax.set_title("kernel PCA score plot", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(length=2)
    fig.tight_layout()
    print(f"  chemical space map: {len(train)} training and {len(test)} test steroids, "
          f"components explain {100 * ratio[0]:.1f} and {100 * ratio[1]:.1f} %")
    save(fig, path)


def fig_gradient(path):
    """Chapter 9: the gradient of one prediction: the derivative with respect to every atom
    position, drawn as arrows, and with respect to the atom's charge, drawn as colour."""
    import matplotlib.pyplot as plt
    from cmfqsar.gradients import prediction_gradient
    s, c = steroids(), cofield_model()
    ds, mf, model, frame, fs = s["ds"], c["mf"], c["model"], c["frame"], c["fs"]
    i = ds.names.index("19-norprogesterone")
    mol = mf[i]
    g = prediction_gradient(model, mol)
    heavy = np.array([sym != "H" for sym in mol.symbols])
    xy = project(frame, mol.coords)
    origin, e1, e2, e3 = frame
    d_xy = np.column_stack([g.coords @ e1, g.coords @ e2])
    f = fs.names.index("electrostatic")
    dq = g.weights[:, f]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.25))
    for ax in axes:
        draw_skeleton(ax, mol, frame, colour="0.55")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[0]
    size = np.linalg.norm(d_xy[heavy], axis=1)
    scale = 2.6 / size.max()
    ax.quiver(xy[heavy, 0], xy[heavy, 1], d_xy[heavy, 0] * scale, d_xy[heavy, 1] * scale,
              angles="xy", scale_units="xy", scale=1, width=0.006, color="C3", zorder=5)
    ax.set_title("move each atom this way to raise the prediction\n"
                 f"longest arrow: {size.max():.2f} pK units per Å", fontsize=8)
    ax = axes[1]
    vmax = np.abs(dq[heavy]).max()
    sc = ax.scatter(xy[heavy, 0], xy[heavy, 1], c=dq[heavy], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=90, edgecolor="black", linewidth=0.4, zorder=5)
    ax.set_title("derivative with respect to each atom's charge\npK units per unit charge",
                 fontsize=8)
    fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.03).ax.tick_params(labelsize=7)
    fig.suptitle(f"{mol.name}: prediction {g.value:.2f}, observed {ds.activity[i]:.2f}", fontsize=9,
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    k, _ = g.largest_coordinate_terms(1)[0]
    print(f"  gradient of {mol.name}: value {g.value:.3f}; largest position derivative "
          f"{size.max():.3f} per A on atom {k} ({mol.symbols[k]}); largest charge derivative "
          f"{dq[heavy].max():+.3f}/{dq[heavy].min():+.3f}")
    save(fig, path)


def hammett_vector_model():
    """The model of the section on the vector co-field: the 95 Hammett benzenes, the
    electric-field kind as the only field, the meta constant as the property.  The settings are
    those of the arrow stage of ``benchmarks/run_surface_view.py``."""
    if "hammett_vector" not in _CACHE:
        from rdkit import Chem, RDLogger
        from cmfqsar.fields import EFIELD_FIELD, AtomTable, FieldSet, build_fields
        from cmfqsar.model import CMFModel
        RDLogger.DisableLog("rdApp.*")
        mols, names, y = [], [], []
        source = os.path.join(ROOT, "examples", "hammett", "hammett.sdf")
        for mol in Chem.SDMolSupplier(source, removeHs=False, sanitize=True):
            if mol is None or not mol.HasProp("sigma_m"):
                continue
            value = float(mol.GetProp("sigma_m"))
            if not np.isfinite(value):
                continue
            mols.append(Chem.Mol(mol))
            names.append(mol.GetProp("_Name"))
            y.append(value)
        fs = FieldSet.create(physchem=[EFIELD_FIELD], alpha=1.0, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
        model.fit(AtomTable.from_molecules(mf), np.array(y, float), cv_folds=0)
        _CACHE["hammett_vector"] = dict(mf=mf, names=names, model=model)
    return _CACHE["hammett_vector"]


def fig_vectorcofield(path):
    """Chapter 4: a vector co-field.  Left, the whole of it in the plane of the ring; right, the
    sixteen strongest arrows, which is what the window draws."""
    import matplotlib.pyplot as plt
    from cmfqsar.cofields import (cofield_arrows, evaluate_gaussian_gradients,
                                  evaluate_gaussians, make_grid)
    h = hammett_vector_model()
    mf, names, model = h["mf"], h["names"], h["model"]
    mol = mf[names.index("NO2")]
    # the series is in one frame by rule: the ring in the plane z = 0, the ipso carbon on +x
    frame = (np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    centres, coeffs, widths = model.cofield_coefficients(0)
    xs, ys = np.linspace(-3.2, 6.2, 189), np.linspace(-3.6, 3.6, 145)
    pts, shape = plane_points(frame, xs, ys)
    potential = evaluate_gaussians(pts, centres, coeffs, widths).reshape(shape)
    qx, qy = np.arange(-3.0, 6.01, 0.6), np.arange(-3.3, 3.31, 0.6)
    qpts, qshape = plane_points(frame, qx, qy)
    grad = evaluate_gaussian_gradients(qpts, centres, coeffs, widths)

    fig = plt.figure(figsize=(WIDTH, 2.85))
    ax = fig.add_axes((0.075, 0.14, 0.50, 0.72))
    im = signed_map(ax, potential, xs, ys)
    im.set_alpha(0.55)
    size = np.linalg.norm(grad[:, :2], axis=1)
    ax.quiver(qpts[:, 0], qpts[:, 1], grad[:, 0], grad[:, 1], angles="xy", scale_units="xy",
              scale=size.max() / 1.1, width=0.0045, color="0.1", zorder=5)
    draw_skeleton(ax, mol, frame, colour="0.35", hydrogens=True)
    ax.set_title("the whole vector co-field in the plane of the ring\n"
                 "colour: the scalar whose gradient it is", fontsize=7.5)
    axes_in_angstrom(ax)

    # what the window draws: the strongest arrows in space, the longest one 2.5 A long
    coords = np.vstack([m.coords for m in mf])
    grid = make_grid(coords, margin=2.0, spacing=0.5)
    points, vectors, lengths = cofield_arrows(model, 0, grid, count=16, min_separation=1.5)
    shown = vectors * (2.5 / lengths.max())
    ax3 = fig.add_axes((0.57, 0.0, 0.44, 0.88), projection="3d")
    heavy = np.array([s != "H" for s in mol.symbols])
    for a, b in mol.bonds:
        both = heavy[a] and heavy[b]
        ax3.plot(*np.column_stack([mol.coords[a], mol.coords[b]]), color="0.3",
                 lw=1.6 if both else 0.7, solid_capstyle="round")
    ax3.scatter(*mol.coords[heavy].T, s=14, color="0.3", depthshade=False)
    ax3.quiver(points[:, 0], points[:, 1], points[:, 2], shown[:, 0], shown[:, 1], shown[:, 2],
               length=1.0, normalize=False, arrow_length_ratio=0.3, color="#c26a00", lw=1.2)
    ax3.set_xlim(-1.3, 5.3)
    ax3.set_ylim(-3.3, 3.3)
    ax3.set_zlim(-3.3, 3.3)
    ax3.set_box_aspect((1, 1, 1), zoom=1.75)
    ax3.view_init(elev=28, azim=-72)
    ax3.set_axis_off()
    fig.text(0.79, 0.93, "the 16 strongest arrows, as the window draws them\n"
             f"longest: {lengths.max():.2f} $\\sigma_m$ units per unit of field per Å",
             ha="center", va="center", fontsize=7.5)
    print(f"  vector co-field: {len(mf)} molecules; longest arrow {lengths.max():.4f}, shortest "
          f"drawn {lengths.min():.4f}; largest in-plane gradient on the 0.6 A grid {size.max():.4f}")
    save(fig, path)


def fig_copattern(path):
    """Chapter 4: the concept of the co-pattern.  A channel is one pair of field types at one
    distance; a molecule has a pattern over the channels, the model has a weight for each, and the
    prediction is the bias plus the sum of weight times channel."""
    import matplotlib.pyplot as plt
    from cmfqsar import CMFModel
    from cmfqsar.copattern import copattern, pair_labels
    from cmfqsar.fields import AtomTable
    from cmfqsar.invariant import field_pairs
    from cmfqsar.periodic import channel_weights
    s = steroids()
    fs, mf, idx, y = s["fs"], s["mf"], s["train_idx"], s["y"]
    train = AtomTable.from_molecules([mf[i] for i in idx])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation",
                     params={"lam": 1e-2})
    model.fit(train, y, cv_folds=0)
    w = np.asarray(channel_weights(model), float)
    Z = np.asarray(model.ac_train_, float) * np.asarray(model.ac_column_scale, float)
    Z = Z / np.sqrt(np.sum(Z * Z, axis=1, keepdims=True))          # the pattern of every molecule
    predicted = model.predict(train)
    residual = float(np.abs(model.bias + Z @ w - predicted).max())
    # the pair of field types the chart of the window opens on: the largest summed weight x spread
    pairs = pair_labels(model, fs)
    curves = [copattern(model, fs, k, "weight x spread") for k in range(len(pairs))]
    k = int(np.argmax([np.abs(c.moved).sum() for c in curves]))
    cp = curves[k]
    d, nb = cp.x, len(cp.x)
    block = slice(k * nb, (k + 1) * nb)
    f, g = field_pairs(len(fs.names))[k]
    hi, lo = int(np.argmax(y)), int(np.argmin(y))
    shown = ((hi, "C3"), (lo, "C0"))
    peak = int(np.argmax(np.abs(cp.weight)))
    step = float(d[1] - d[0])

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.5))
    # (a) one channel on one molecule: the atom pairs about that far apart
    ax = axes[0, 0]
    mol = mf[idx[hi]]
    frame = plane_of([mol])
    xy = project(frame, mol.coords)
    dist = np.linalg.norm(mol.coords[:, None, :] - mol.coords[None, :, :], axis=-1)
    W = np.asarray(mol.weights, float)
    term = (np.exp(-((dist - d[peak]) ** 2) / (2.0 * step ** 2))
            * (W[:, f][:, None] * W[:, g][None, :] + W[:, g][:, None] * W[:, f][None, :]))
    # drawn: the heavy-atom pairs within half a step of the channel's distance, largest first
    heavy = np.array([sym != "H" for sym in mol.symbols])
    near = (np.abs(dist - d[peak]) <= 0.5 * step) & heavy[:, None] & heavy[None, :]
    term = np.triu(np.where(near, term, 0.0), 1)
    order = [q for q in np.argsort(np.abs(term).ravel())[::-1][:14] if term.ravel()[q] != 0.0]
    draw_skeleton(ax, mol, frame, colour="0.55")
    for flat in order:
        a, b = divmod(int(flat), term.shape[0])
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], lw=1.0, zorder=5,
                color="#b2182b" if term[a, b] > 0 else "#2166ac")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"a channel: one pair of field types at one distance\n"
                 f"{cp.pair.replace(' x ', ' × ')}, {d[peak]:.2f} Å, on {mol.name}", fontsize=7.5)
    # (b) the pattern of a molecule: its channels against distance
    ax = axes[0, 1]
    for i, colour in shown:
        ax.plot(d, Z[i, block], "o-", ms=3, lw=1.1, color=colour, label=mf[idx[i]].name)
    ax.set_title("the pattern of a molecule: its channels", fontsize=7.5)
    ax.set_ylabel("channel value")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.38 * (high - low))
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    # (c) the co-pattern of the model: one weight per channel
    ax = axes[1, 0]
    ax.bar(d, cp.weight, width=0.55 * step, color="0.45")
    ax.set_title("the co-pattern of the model: a weight per channel", fontsize=7.5)
    ax.set_ylabel("channel weight")
    # (d) weight times channel, which sums to the prediction
    ax = axes[1, 1]
    for j, (i, colour) in enumerate(shown):
        part = w[block] * Z[i, block]
        ax.bar(d + (j - 0.5) * 0.3 * step, part, width=0.3 * step, color=colour,
               label=f"{mf[idx[i]].name}: this pair adds {part.sum():+.2f}")
    ax.set_title("weight × channel: one term of the prediction each", fontsize=7.5)
    ax.set_ylabel("contribution to the prediction")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.42 * (high - low))
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axhline(0, color="0.6", lw=0.5)
        ax.axvline(d[peak], color="0.6", lw=0.5, ls=":")
        ax.set_xlabel("distance between two atoms / Å")
        ax.tick_params(length=2)
    def total(i):
        rest = predicted[i] - model.bias
        return (f"{mf[idx[i]].name}: {model.bias:.2f} {'+' if rest >= 0 else '−'} {abs(rest):.2f} "
                f"= {predicted[i]:.2f} (observed {y[i]:.2f})")

    fig.suptitle("prediction = bias + sum over all channels of (channel weight × channel value)\n"
                 f"{total(hi)};  {total(lo)}", fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 1.03))
    print(f"  co-pattern: {len(w)} channels in {len(pairs)} pairs; the weights reproduce the "
          f"prediction to {residual:.1e}; shown pair {cp.pair}, peak at {d[peak]:.2f} A")
    save(fig, path)


def fig_cospectrum(path):
    """Chapter 11: the concept of the co-spectrum.  A spectrum is a field on a line, the model
    has a weight over the same line, and the prediction is the bias plus the sum of weight
    times intensity along the axis.  The model is the one of ``benchmarks/run_spectra_ir.py``:
    the infrared spectra of the benzoic acids against the Hammett constant."""
    import matplotlib.pyplot as plt
    from rdkit import Chem, RDLogger
    from cmfqsar import spectrum as SP
    from cmfqsar.model import krr_fit
    RDLogger.DisableLog("rdApp.*")
    with open(os.path.join(RESULTS, "spectra_ir", "ir.json"), encoding="utf-8") as fh:
        rec = json.load(fh)
    width, power = float(rec["cospectrum"]["width"]), float(rec["cospectrum"]["power"])
    lam, (lo, hi) = float(rec["cospectrum"]["lam"]), rec["range"]
    window = (1560.0, 1800.0)            # the carbonyl stretch and the ring stretch below it
    folder = os.path.join(ROOT, "examples", "spectra")
    names, y = [], []
    for mol in Chem.SDMolSupplier(os.path.join(folder, "benzoic_acids_co.sdf"), removeHs=False):
        names.append(mol.GetProp("_Name"))
        y.append(float(mol.GetProp("sigma")))
    y = np.array(y, float)
    with open(os.path.join(folder, "benzoic_acids_ir.json"), encoding="utf-8") as fh:
        rows = {r["name"]: r for r in json.load(fh)["acids"]}
    spectra = [SP.Spectrum(rows[n]["wavenumbers"], rows[n]["intensities"], "sticks",
                           "wavenumber", n) for n in names]
    grid = SP.Grid.for_width(lo, hi, width)
    F = SP.feature_matrix(spectra, grid, width, power)
    dual, bias = krr_fit(SP.spectrum_kernel(F), y, lam)
    w = SP.co_spectrum(F, dual)                                  # the co-spectrum
    Z = F / np.sqrt(np.einsum("ij,ij->i", F, F))[:, None]        # the normalised spectra
    predicted = bias + Z @ w
    x = grid.x
    top, bottom = int(np.argmax(y)), int(np.argmin(y))
    shown = ((top, "C3"), (bottom, "C0"))
    zoom = (x >= window[0]) & (x <= window[1])

    def signed(value):
        return f"{value:+.2f}".replace("-", "−")

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.5))
    # (a) a spectrum is a field on a line: sticks, each broadened to a Gaussian
    ax = axes[0, 0]
    for i, colour in shown:
        sp = spectra[i]
        pos, height = np.asarray(sp.x, float), np.asarray(sp.y, float)
        inside = (pos >= window[0]) & (pos <= window[1])
        scale = Z[i, zoom].max() / max(height[inside].max(), 1e-12)
        ax.vlines(pos[inside], 0.0, height[inside] * scale, color=colour, lw=1.6)
        ax.plot(x[zoom], Z[i, zoom], color=colour, lw=0.9, label=names[i])
        ax.fill_between(x[zoom], 0.0, Z[i, zoom], color=colour, alpha=0.15, lw=0)
    ax.set_title("a spectrum is a field on a line: every stick\n"
                 f"becomes a Gaussian of width {width:g} cm$^{{-1}}$", fontsize=7.5)
    ax.set_ylabel("normalised intensity")
    ax.set_ylim(0.0, 1.45 * ax.get_ylim()[1])
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    # (b) the co-spectrum: the weight of the model over the same line
    ax = axes[0, 1]
    ax.fill_between(x[zoom], 0.0, w[zoom], where=w[zoom] >= 0, color="0.3", lw=0)
    ax.fill_between(x[zoom], 0.0, w[zoom], where=w[zoom] <= 0, color="0.7", lw=0)
    ax.set_title("the co-spectrum: the weight of the model\nover the same line", fontsize=7.5)
    ax.set_ylabel("weight")
    # (c) weight times intensity, place by place
    ax = axes[1, 0]
    for i, colour in shown:
        part = w * Z[i]
        ax.plot(x[zoom], part[zoom], color=colour, lw=0.9,
                label=f"{names[i]}: this region adds {signed(part[zoom].sum())}")
        ax.fill_between(x[zoom], 0.0, part[zoom], color=colour, alpha=0.25, lw=0)
    ax.set_title("weight × intensity: what each place adds", fontsize=7.5)
    ax.set_ylabel("contribution to the prediction")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.45 * (high - low))
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlim(window[1], window[0])              # the convention of infrared spectra
    # (d) the running sum along the whole axis, which ends at the prediction
    ax = axes[1, 1]
    ax.axvspan(window[0], window[1], color="0.9", lw=0)
    for i, colour in shown:
        running = bias + np.cumsum((w * Z[i])[::-1])[::-1]        # summed from the high end
        ax.plot(x, running, color=colour, lw=1.0)
        ax.annotate(signed(predicted[i]), (x[0], running[0]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7, color=colour,
                    annotation_clip=False)
    ax.axhline(bias, color="0.5", lw=0.6, ls="--")
    ax.set_xlim(hi, lo)
    ax.set_title("the running sum along the whole axis starts\n"
                 "at the bias and ends at the prediction", fontsize=7.5)
    ax.set_ylabel("bias + sum so far")
    for ax in axes.ravel():
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel("wavenumber / cm$^{-1}$")
        ax.tick_params(length=2)

    def total(i):
        rest = predicted[i] - bias
        return (f"{names[i]}: {bias:.2f} {'+' if rest >= 0 else '−'} {abs(rest):.2f} = "
                f"{signed(predicted[i])} (tabulated {signed(y[i])})")

    fig.suptitle("prediction = bias + sum along the axis of (weight × normalised intensity)\n"
                 f"{total(top)};  {total(bottom)}", fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 1.03))
    print(f"  co-spectrum: {len(spectra)} spectra, {len(x)} grid points, width {width:g}, ridge "
          f"{lam:g}; bias {bias:.4f}; {names[top]} {predicted[top]:+.3f} ({y[top]:+.2f}), "
          f"{names[bottom]} {predicted[bottom]:+.3f} ({y[bottom]:+.2f})")
    save(fig, path)


def fig_featurespace(path):
    """Chapter 2: the feature space of the overlap kernel.  A molecule is a point of it, and the
    point is the molecule's field; the model is a direction in it, and the direction is the
    co-field; a prediction is the bias plus the projection of the point on the direction."""
    import matplotlib.pyplot as plt
    from cmfqsar.fields import AtomTable
    s, c = steroids(), cofield_model()
    fs, mf, model, frame, xs, ys = c["fs"], c["mf"], c["model"], c["frame"], c["xs"], c["ys"]
    idx, y = s["train_idx"], np.asarray(s["y"], float)
    train = AtomTable.from_molecules([mf[i] for i in idx])
    K = np.asarray(model.training_kernel(train), float)
    a, bias = np.asarray(model.coef, float), float(model.bias)
    along = K @ a                                   # the inner product of every molecule with w
    length = float(np.sqrt(a @ K @ a))              # the length of the weight vector w
    residual = float(np.abs(along + bias - model.predict(train)).max())
    # the plane through the feature space that holds the direction of w and, across it, the
    # direction in which the training molecules spread most
    n = len(y)
    J = np.eye(n) - np.ones((n, n)) / n
    t = along / length
    u = t - t.mean()
    across = J @ K @ J - np.outer(u, u)
    values, vectors = np.linalg.eigh(across)
    v = vectors[:, -1] * np.sqrt(max(values[-1], 0.0))
    if v[int(np.argmax(y))] < 0:
        v = -v                                       # a reproducible orientation
    k = int(np.argmax(y))
    mol = mf[idx[k]]
    f = fs.names.index("electrostatic")
    pts, shape = plane_points(frame, xs, ys)

    fig = plt.figure(figsize=(WIDTH, 3.5))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.35), left=0.02, right=0.98, bottom=0.13,
                            top=0.9, wspace=0.22, hspace=0.32)
    ax = fig.add_subplot(grid[0, 0])
    signed_map(ax, field_slice(mol, fs.specs[f], f, pts).reshape(shape), xs, ys)
    draw_skeleton(ax, mol, frame)
    ax.set_title(f"a point of the feature space:\nthe field of {mol.name}", fontsize=7.5)
    ax2 = fig.add_subplot(grid[1, 0])
    signed_map(ax2, cofield_slice(model, f, pts).reshape(shape), xs, ys)
    draw_skeleton(ax2, mol, frame)
    ax2.set_title("a direction in it: the co-field,\nthe weight vector of the model", fontsize=7.5)
    for m in (ax, ax2):
        m.set_xticks([])
        m.set_yticks([])

    ax = fig.add_subplot(grid[:, 1])
    sc = ax.scatter(t, v, c=y, cmap="viridis", s=26, edgecolor="black", linewidth=0.4, zorder=4)
    ax.scatter([t[k]], [v[k]], s=110, facecolor="none", edgecolor="C3", linewidth=1.1, zorder=5)
    low = v.min() - 0.18 * (v.max() - v.min())
    ax.plot([t[k], t[k]], [v[k], low], color="C3", lw=0.8, ls="--", zorder=3)
    ax.annotate("", xy=(t.max(), low), xytext=(t.min(), low),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    ax.text(0.5 * (t.min() + t.max()), low, "direction of the co-field", ha="center",
            va="bottom", fontsize=7)
    ax.set_xlabel("projection on the direction of the co-field")
    ax.set_ylabel("the direction of largest spread across it")
    ax.tick_params(length=2)
    top = ax.secondary_xaxis("top", functions=(lambda p: bias + length * p,
                                               lambda q: (q - bias) / length))
    top.set_xlabel("prediction = bias + length of the weight vector × projection", fontsize=7.5)
    top.tick_params(length=2, labelsize=7.5)
    bar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.03)
    bar.set_label("observed activity", fontsize=7.5)
    bar.ax.tick_params(labelsize=7)
    print(f"  feature space: {n} molecules; |w| = {length:.4f}, bias {bias:.4f}; the projection "
          f"reproduces the prediction to {residual:.1e}; share of the spread across w in the "
          f"drawn direction {values[-1] / max(values.sum(), 1e-30):.2f}")
    save(fig, path)


def microstructure_set():
    """The 400 synthetic two-phase cells of ``benchmarks/run_microstructure.py`` at contrast
    100, generated again with the benchmark's own functions and seed, so that the figure does
    not depend on the benchmark's cache.  About ten seconds."""
    if "microstructure" not in _CACHE:
        sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
        import run_microstructure as M
        dim, grid, n, contrast = 2, 51, 400, 100.0
        rng = np.random.default_rng(0)
        phases = np.zeros((n, grid, grid), np.uint8)
        params, tensors = np.zeros((n, 3)), np.zeros((n, dim, dim))
        for i in range(n):                                   # the loop of M.build_data
            fraction = rng.uniform(*M.FRACTION)
            length = np.exp(rng.uniform(np.log(M.LENGTH[0]), np.log(M.LENGTH[1])))
            aspect = np.exp(rng.uniform(np.log(M.ASPECT[0]), np.log(M.ASPECT[1])))
            rot = M._rotation(dim, rng)
            phases[i] = M.generate(dim, grid, fraction, length, aspect, rot, rng)
            params[i] = (fraction, length, aspect)
            tensors[i] = M.effective_conductivity(phases[i], contrast)
        _CACHE["microstructure"] = dict(M=M, phase=phases, params=params, tensor=tensors,
                                        tag=M.tag_of(dim, grid, n, contrast))
    return _CACHE["microstructure"]


def fig_phasefield(path):
    """Chapter 11: the phase field of a microstructure and the co-field of a model on it.  A
    two-phase cell becomes a field on the voxel grid, the field becomes its two-point statistics
    over the lag vector, and the model is a weight over the same lags."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    d = microstructure_set()
    M, phase = d["M"], d["phase"]
    y = d["tensor"][:, 0, 0].copy()                            # the conductivity along x
    with open(os.path.join(RESULTS, "microstructure", "map_" + d["tag"] + ".json"),
              encoding="utf-8") as fh:
        lam = float(json.load(fh)["labels"]["xx"]["lam"])
    # the linear model on the two-point statistics, as in M.coefficient_map
    S = M.autocorrelations(M.smoothed_fields(phase, 0.0))       # (n, 2, grid, grid)
    n, grid = S.shape[0], S.shape[-1]
    A = S.reshape(n, -1)
    K = A @ A.T
    scale = float(np.mean(np.diag(K)))
    learner = M._Learner("krr", {"lam": lam}, True).fit(K / scale, y)
    w = (learner.alpha_ @ A / scale).reshape(S.shape[1:])        # the weight over the lags
    bias = float(learner.b_)
    predicted = bias + A @ w.ravel()
    residual = float(np.abs(predicted - learner.predict(K / scale)).max())
    # two cells of about the same composition and very different conductivity
    fraction = d["params"][:, 0]
    middle = np.where((fraction > 0.45) & (fraction < 0.55))[0]
    shown = (int(middle[np.argmax(y[middle])]), int(middle[np.argmin(y[middle])]))
    smooth = M.smoothed_fields(phase[list(shown)], 2.0)[:, 1]    # phase 1, width 2 voxels
    half = grid // 2
    lag = (-half - 0.5, half + 0.5, -half - 0.5, half + 0.5)
    window = 15.5                                              # lags drawn, in voxels

    def centred(a):
        return np.fft.fftshift(a).T

    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 6.4))
    two_phase = ListedColormap(["#f1ede4", "#3b4252"])
    for row, i in enumerate(shown):
        ax = axes[row, 0]
        ax.imshow(phase[i].T, origin="lower", cmap=two_phase, interpolation="nearest")
        ax.set_ylabel(f"cell {'AB'[row]}\nconductivity along x: {y[i]:.1f}", fontsize=7.5)
        ax = axes[row, 1]
        ax.imshow(smooth[row].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        ax = axes[row, 2]
        im_s = ax.imshow(centred(S[i, 0]), origin="lower", extent=lag, cmap="magma",
                         vmin=0.0, vmax=float(S[list(shown), 0].max()))
    axes[0, 0].set_title("a two-phase cell\n(dark: phase 1, the conductor)", fontsize=7.5)
    axes[0, 1].set_title("its phase field: the indicator of\nphase 1, smoothed over 2 voxels",
                         fontsize=7.5)
    axes[0, 2].set_title("its two-point statistics:\nphase 1 found again at a lag", fontsize=7.5)
    lim = float(np.abs(centred(w[0])).max())
    ax = axes[2, 0]
    im_w = ax.imshow(centred(w[0]), origin="lower", extent=lag, cmap="RdBu_r", vmin=-lim,
                     vmax=lim)
    ax.set_title("the co-field of the model: a weight for\nevery lag (phase 1 with phase 1)",
                 fontsize=7.5)
    parts = [(w * S[i]).sum(axis=0) for i in shown]          # both phase pairs, lag by lag
    plim = float(max(np.abs(p).max() for p in parts))
    for col, (i, part) in enumerate(zip(shown, parts), start=1):
        ax = axes[2, col]
        im_p = ax.imshow(centred(part), origin="lower", extent=lag, cmap="RdBu_r", vmin=-plim,
                         vmax=plim)
        ax.set_title(f"weight × statistics of cell {'AB'[col - 1]}\n"
                     f"sum over all lags: {part.sum():+.1f}".replace("-", "−"), fontsize=7.5)
    for ax in list(axes[:2, 2]) + list(axes[2]):
        ax.set_xlim(-window, window)
        ax.set_ylim(-window, window)
        ax.set_xlabel("lag along x / voxels", fontsize=7.5)
        ax.tick_params(length=2, labelsize=7)
    axes[2, 0].set_ylabel("lag along y / voxels", fontsize=7.5)
    for ax in axes[:2, :2].ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    def total(row):
        i = shown[row]
        rest = predicted[i] - bias
        return (f"cell {'AB'[row]}: {bias:.1f} {'+' if rest >= 0 else '−'} {abs(rest):.1f} = "
                f"{predicted[i]:.1f} (computed {y[i]:.1f})")

    fig.suptitle("prediction = bias + sum over all lag vectors and both phase pairs of "
                 "(weight × two-point statistics)\n" + total(0) + ";  " + total(1),
                 fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 1.02))
    print(f"  phase field: {n} cells, ridge {lam:g}; the weights reproduce the prediction to "
          f"{residual:.1e}; cell A {shown[0]} ({y[shown[0]]:.2f}), cell B {shown[1]} "
          f"({y[shown[1]]:.2f})")
    save(fig, path)


def fig_multiscale(path):
    """Chapter 11: the multiscale model of a battery cell and the place of this program in it.
    Above, the scheme: a fine level labels structures, a model of this program answers with a
    value and a standard deviation, a conversion turns the answer into a parameter of a
    continuum cell model, and the cell model gives the capacity.  Below, what
    ``benchmarks/run_cellmodel.py`` measured: the interval of the capacity when each parameter
    is drawn from its deviation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
    with open(os.path.join(RESULTS, "cellmodel", "cellmodel.json"), encoding="utf-8") as fh:
        rec = json.load(fh)
    ch, per, shares = rec["channels"], rec["per_parameter"], rec["variance_shares"]
    keys = {"diffusivity": "Positive particle diffusivity [m2.s-1]",
            "voltage": "Positive electrode OCP [V]",
            "conductivity": "Electrolyte conductivity [S.m-1]",
            "bruggeman": "Positive electrode Bruggeman coefficient (electrolyte)",
            "exchange": "Positive electrode exchange-current density [A.m-2]"}
    # one row per channel: the scale, the fine level, the model of this program, the parameter
    rows = [
        ("diffusivity", "an ion hop\nin a crystal\n0.3 nm",
         f"migration barriers\nfrom DFT\n{ch[keys['diffusivity']]['training']['hops']:,} hops",
         "model on the field\naround the hop path\nbarrier ± deviation",
         "solid diffusion\ncoefficient"),
        ("voltage", "a crystal\nand its ions\n1 nm",
         f"formation energies\nfrom DFT\n{ch[keys['voltage']]['n']} insertion pairs",
         "kernel on the\ncomposition\nvoltage ± deviation",
         "shift of the open-\ncircuit voltage"),
        ("conductivity", "a liquid\nelectrolyte\nmolecules",
         f"measured\nconductivities\n{ch[keys['conductivity']]['n']} electrolytes",
         "kernel on the\ncomposition\nconductivity ± deviation",
         "factor on the\nconductivity"),
        ("bruggeman", "a porous\nelectrode\n1 to 10 µm",
         f"solved two-phase\nstructures\n{ch[keys['bruggeman']]['n_train']} cells",
         "kernel on the two-\npoint statistics\nconductivity ± deviation",
         "Bruggeman\nexponent"),
    ]
    blue, green, sand, grey = "#dbe7f3", "#dcefdc", "#f6ead2", "#e9e9e9"

    fig = plt.figure(figsize=(WIDTH, 6.3))
    ax = fig.add_axes((0.0, 0.40, 1.0, 0.60))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    def box(x, y, w, h, label, colour, size=6.3, style="solid", weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor=colour, edgecolor="0.35", linewidth=0.7,
                                    linestyle=style))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size,
                linespacing=1.25, fontweight=weight)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.8, color="0.25", shrinkA=0, shrinkB=0))

    x_scale, x_fine, x_fast, x_par, x_cell = 0.005, 0.135, 0.36, 0.60, 0.815
    w_scale, w_fine, w_fast, w_par, w_cell = 0.11, 0.185, 0.20, 0.17, 0.18
    top, height, gap = 0.80, 0.135, 0.028
    heads = ((x_scale, w_scale, "scale"),
             (x_fine, w_fine, "fine level:\nlabels for a few\nhundred structures"),
             (x_fast, w_fast, "fast level: a model of\nthis program, with a\nstandard deviation"),
             (x_par, w_par, "conversion to a\nparameter of the\ncell model"),
             (x_cell, w_cell, "coarse level:\nthe cell\n0.1 mm and more"))
    for x, w, label in heads:
        ax.text(x + w / 2, 0.985, label, ha="center", va="top", fontsize=6.6, fontweight="bold",
                linespacing=1.2)
    for k, (_key, scale, fine, fast, par) in enumerate(rows):
        y = top - (k + 1) * height - k * gap
        mid = y + height / 2
        ax.text(x_scale + w_scale / 2, mid, scale, ha="center", va="center", fontsize=6.3,
                style="italic", linespacing=1.25)
        box(x_fine, y, w_fine, height, fine, grey)
        box(x_fast, y, w_fast, height, fast, green)
        box(x_par, y, w_par, height, par, sand)
        arrow(x_fine + w_fine, mid, x_fast, mid)
        arrow(x_fast + w_fast, mid, x_par, mid)
        arrow(x_par + w_par, mid, x_cell, mid)
    # the parameter that no model of this program predicts
    y = top - 5 * height - 4 * gap + 0.03
    box(x_fine, y, x_par + w_par - x_fine, height - 0.045,
        "exchange-current density: not predicted, it enters with a stated spread", "white",
        style="dashed")
    arrow(x_par + w_par, y + (height - 0.045) / 2, x_cell, y + (height - 0.045) / 2)
    low = y
    box(x_cell, low, w_cell, top - low,
        "continuum cell model\n(Doyle-Fuller-Newman)\n\nequations of transport\nin space and time\n\n"
        "each parameter is\ndrawn from its\ndeviation; one run\nper draw", blue)
    xm = x_cell + w_cell / 2                         # the arrow down, as a line and a triangle
    ax.plot([xm, xm], [low - 0.008, low - 0.055], color="0.25", lw=0.9, solid_capstyle="butt",
            clip_on=False)                           # it reaches below the axes
    ax.add_patch(Polygon([(xm - 0.006, low - 0.052), (xm + 0.006, low - 0.052),
                          (xm, low - 0.075)], closed=True, facecolor="0.25", edgecolor="0.25",
                         linewidth=0.5, clip_on=False))
    ax.text(x_cell + w_cell / 2, low - 0.09, "capacity per cycle,\nwith an interval",
            ha="center", va="top", fontsize=6.6, fontweight="bold", linespacing=1.2)

    # what was measured: the interval of the capacity when one parameter is drawn at a time
    ax2 = fig.add_axes((0.30, 0.07, 0.45, 0.215))
    published = float(rec["published"]["final"])
    order = ["diffusivity", "voltage", "conductivity", "bruggeman", "exchange"]
    names = ["solid diffusion coefficient", "open-circuit voltage", "electrolyte conductivity",
             "Bruggeman exponent", "exchange-current density"]
    for k, key in enumerate(order):
        lo_c, hi_c = per[keys[key]]["interval_of_the_last_cycle"]
        holds = bool(per[keys[key]]["holds_the_published_curve"]["all_inside"])
        colour = "C0" if holds else "C3"
        ax2.plot([lo_c, hi_c], [k, k], color=colour, lw=4.5, solid_capstyle="butt")
        if hi_c - lo_c < 0.1:                         # an interval too narrow to see as a line
            ax2.plot([0.5 * (lo_c + hi_c)], [k], marker="|", ms=9, mew=2.2, color=colour)
        share = 100.0 * float(shares[keys[key]]["share"])
        ax2.text(1.03, k, f"{share:.1f} % of the variance; "
                 + ("holds" if holds else "misses") + " the published capacity",
                 transform=ax2.get_yaxis_transform(), va="center", fontsize=6.3)
    ax2.axvline(published, color="0.3", lw=0.7, ls="--")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(names, fontsize=6.6)
    ax2.set_ylim(len(order) - 0.5, -0.5)
    ax2.set_xlim(0.0, 5.8)
    ax2.set_xlabel(f"capacity after {rec['cycles']} cycles / A·h (dashed: with the published "
                   "parameters)", fontsize=6.6)
    ax2.tick_params(length=2, labelsize=6.6)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.set_title(f"measured: the 95 % interval of the capacity when one parameter is drawn\n"
                  f"{rec['draws']} times from its deviation and the others keep their published "
                  "values", fontsize=6.8, loc="left", x=-0.62, linespacing=1.25)
    print("  multiscale: " + "; ".join(
        f"{key} {per[keys[key]]['interval_of_the_last_cycle'][0]:.3f}-"
        f"{per[keys[key]]['interval_of_the_last_cycle'][1]:.3f}" for key in order))
    save(fig, path)


def fig_idea(path):
    """Chapter 1: the method in six steps, each computed from the steroid model of chapter 8.
    Molecules in one frame, their fields, the kernel of every pair, the fitted coefficients, the
    co-field, and the prediction of the molecules that were held out."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from cmfqsar.fields import AtomTable
    s, c = steroids(), cofield_model()
    ds, fs, mf, model, frame, xs, ys = s["ds"], c["fs"], c["mf"], c["model"], c["frame"], c["xs"], c["ys"]
    idx, y = s["train_idx"], np.asarray(s["y"], float)
    train = AtomTable.from_molecules([mf[i] for i in idx])
    K = np.asarray(model.training_kernel(train), float)
    a = np.asarray(model.coef, float)
    k = int(np.argmax(y))
    mol = mf[idx[k]]
    f = fs.names.index("electrostatic")
    pts, shape = plane_points(frame, xs, ys)
    test_idx = [i for i in range(len(mf)) if i not in set(idx)]
    y_test = np.asarray(ds.activity, float)[test_idx]
    p_test = model.predict(AtomTable.from_molecules([mf[i] for i in test_idx]))

    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.1))
    ax = axes[0, 0]
    for i in idx:
        draw_skeleton(ax, mf[i], frame, colour="0.72")
    draw_skeleton(ax, mol, frame, colour="C3")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect("equal")
    ax.set_title(f"1. the molecules in one frame\n({len(idx)} steroids; one in red)", fontsize=7.5)
    ax = axes[0, 1]
    signed_map(ax, field_slice(mol, fs.specs[f], f, pts).reshape(shape), xs, ys)
    draw_skeleton(ax, mol, frame)
    ax.set_title("2. a field: a Gaussian on\nevery atom of the red one", fontsize=7.5)
    ax = axes[0, 2]
    im = ax.imshow(K, cmap="viridis", interpolation="nearest")
    ax.set_title("3. the kernel: the overlap of\nthe fields of every pair", fontsize=7.5)
    ax.set_xlabel("molecule", fontsize=7)
    ax.set_ylabel("molecule", fontsize=7)
    ax.tick_params(length=2, labelsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03).ax.tick_params(labelsize=6.5)
    ax = axes[1, 0]
    ax.bar(np.arange(len(a)), a, color=["#b2182b" if v > 0 else "#2166ac" for v in a], width=0.8)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_title("4. the fit: one coefficient\nfor every training molecule", fontsize=7.5)
    ax.set_xlabel("molecule", fontsize=7)
    ax.set_ylabel("coefficient", fontsize=7)
    ax.tick_params(length=2, labelsize=6.5)
    ax = axes[1, 1]
    signed_map(ax, cofield_slice(model, f, pts).reshape(shape), xs, ys)
    draw_skeleton(ax, mol, frame)
    ax.set_title("5. the co-field: the fields\ntimes the coefficients, summed", fontsize=7.5)
    ax = axes[1, 2]
    lo, hi = min(y_test.min(), p_test.min()) - 0.3, max(y_test.max(), p_test.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.7)
    ax.scatter(y_test, p_test, s=16, color="C0", edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("observed activity", fontsize=7)
    ax.set_ylabel("predicted activity", fontsize=7)
    ax.tick_params(length=2, labelsize=6.5)
    ax.set_title("6. a prediction: the overlap of\na new field with the co-field", fontsize=7.5)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=3.0, h_pad=1.4)
    # arrows from each step to the next one in its row
    for row in (0, 1):
        for col in (0, 1):
            left = axes[row, col].get_position()
            right = axes[row, col + 1].get_position()
            ymid = 0.5 * (left.y0 + left.y1)
            fig.patches.append(FancyArrowPatch(
                (left.x1 + 0.008, ymid), (right.x0 - (0.07 if col == 1 else 0.012), ymid),
                transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color="0.3"))
    print(f"  idea: {len(idx)} training and {len(test_idx)} test molecules; drawn molecule {mol.name}")
    save(fig, path)


def fig_diagnose(path):
    """Chapter 6: the decisions of ``cmf diagnose`` as a scheme.  The order of the questions and
    the margin are those of ``cmfqsar.diagnose._diagnose`` and ``_recommend``."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon
    from cmfqsar.diagnose import MEANINGFUL
    sand, green, grey, blue = "#f6ead2", "#dcefdc", "#e9e9e9", "#dbe7f3"

    fig = plt.figure(figsize=(WIDTH, 5.9))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    def box(x, y, w, h, label, colour, size=6.6, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.012",
                                    facecolor=colour, edgecolor="0.35", linewidth=0.7))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size,
                linespacing=1.25, fontweight=weight)

    def arrow(x0, y0, x1, y1, label=None):
        ax.plot([x0, x1], [y0, y1], color="0.25", lw=0.9, solid_capstyle="butt")
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) > abs(dy):                       # a head pointing right
            head = [(x1, y1 - 0.007), (x1, y1 + 0.007), (x1 + 0.012, y1)]
        else:                                       # a head pointing down
            head = [(x1 - 0.0065, y1), (x1 + 0.0065, y1), (x1, y1 - 0.013)]
        ax.add_patch(Polygon(head, closed=True, facecolor="0.25", edgecolor="0.25", linewidth=0.5))
        if label:
            if abs(dx) > abs(dy):
                ax.text(0.5 * (x0 + x1), y1 + 0.008, label, ha="center", va="bottom", fontsize=6.3)
            else:
                ax.text(x1 + 0.01, 0.5 * (y0 + y1), label, ha="left", va="center", fontsize=6.3)

    # the five routes that are scored first
    x_q, w_q, x_out, w_out = 0.205, 0.40, 0.675, 0.32
    ax.text(x_q + w_q / 2, 0.985, "five routes are scored by cross-validation, each with its "
            "settings chosen inside the folds", ha="center", va="top", fontsize=6.6)
    routes = ("aligned\nfields", "topomer\nrule", "invariant\nkernel", "atom-type\ncounts",
              "descriptors\nalone")
    for k, label in enumerate(routes):
        box(0.115 + k * 0.125, 0.885, 0.115, 0.06, label, blue, size=6.3)
    arrow(x_q + w_q / 2, 0.885, x_q + w_q / 2, 0.862)

    margin = f"{MEANINGFUL:g}"
    steps = (
        ("Is there anything\nto model?",
         "fewer than 10 molecules, or no\nspread of the activity?",
         "no model: too little to support\nany conclusion", grey),
        ("",
         "does no route reach q² = 0.1?",
         "no model: the set does not\nsupport one as it is", grey),
        ("Does the geometry\ncontribute anything?",
         f"are the descriptors alone within\n{margin} in q² of the aligned model?",
         "descriptors", green),
        ("Does the frame\ncarry signal?",
         f"is a kernel that needs no frame\nwithin {margin} in q² of the aligned model?",
         "invariant kernel or counts,\nthe better of the two", green),
        ("Can a frame exist\nat all?",
         "are the molecules not superimposed?",
         "topomer rule; or, where that rule is\nclearly worse, the aligned model\n"
         "after the frame is curated", green),
        ("",
         "are there several chemotypes?",
         "aligned model, with the\nframe checked", green),
    )
    height, gap, top = 0.082, 0.038, 0.862
    for k, (question, test, outcome, colour) in enumerate(steps):
        y = top - (k + 1) * height - k * gap
        mid = y + height / 2
        box(x_q, y, w_q, height, test, sand)
        box(x_out, y + 0.006, w_out, height - 0.012, outcome, colour,
            weight="bold" if colour == green else "normal", size=6.3)
        arrow(x_q + w_q, mid, x_out - 0.014, mid, "yes")
        if question:
            ax.text(x_q - 0.02, mid, question, ha="right", va="center", fontsize=6.6,
                    style="italic", linespacing=1.25)
        arrow(x_q + w_q / 2, y, x_q + w_q / 2, y - gap + 0.013, "no")
    y = top - 6 * height - 6 * gap - 0.062
    box(x_q + 0.07, y, w_q - 0.14, 0.062, "aligned model", green, weight="bold", size=6.6)
    ax.text(0.5, 0.012, "A descriptor route that reaches q² = 0.95 is read as the activity stored "
            "under another name, and it is not recommended.\nA chosen attenuation at the edge of "
            "its grid is reported as a warning beside any of these answers.", ha="center",
            va="bottom", fontsize=6.3, linespacing=1.3, color="0.25")
    save(fig, path)


def fig_ensemble(path):
    """Chapter 10: a conformational ensemble as one field.  The conformations of one steroid, the
    field of one of them, the mean field of all of them, which is the field the ensemble mode
    uses, and the variance of the field across them, which is the flexibility map."""
    import matplotlib.pyplot as plt
    from cmfqsar.conformers import align_conformers, generate_conformers, single_conformer
    from cmfqsar.fields import build_fields
    s, c = steroids(), cofield_model()
    ds, fs, frame, xs, ys = s["ds"], c["fs"], c["frame"], c["xs"], c["ys"]
    i = s["train_idx"][int(np.argmax(s["y"]))]              # the most active training molecule
    mol = generate_conformers(ds.mols[i], n=8, seed=0)      # its own pose stays the first one
    rings = sorted({a for ring in mol.GetRingInfo().AtomRings() for a in ring})
    align_conformers(mol, atom_ids=rings)                   # on the rigid ring system
    members = [build_fields(single_conformer(mol, k), fs, ds.names[i])
               for k in range(mol.GetNumConformers())]
    f = fs.names.index("electrostatic")
    pts, shape = plane_points(frame, xs, ys)
    slices = np.array([field_slice(m, fs.specs[f], f, pts) for m in members])
    mean = slices.mean(axis=0)                              # equal weights
    variance = (slices ** 2).mean(axis=0) - mean ** 2
    # the mean drawn here is the field the ensemble mode builds
    fs_ens = fs.copy()
    fs_ens.conformers = "ensemble"
    library = field_slice(build_fields(mol, fs_ens, ds.names[i]), fs_ens.specs[f], f, pts)
    gap = float(np.abs(library - mean).max())
    vmax = float(np.abs(slices[0]).max())

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.9))
    ax = axes[0, 0]
    for k in range(len(members) - 1, -1, -1):
        draw_skeleton(ax, members[k], frame, colour="black" if k == 0 else f"C{k % 10}")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect("equal")
    ax.set_title(f"{len(members)} conformations of {ds.names[i]},\nsuperimposed on the ring system",
                 fontsize=7.5)
    ax = axes[0, 1]
    signed_map(ax, slices[0].reshape(shape), xs, ys, vmax=vmax)
    draw_skeleton(ax, members[0], frame)
    ax.set_title("the field of one conformation", fontsize=7.5)
    ax = axes[1, 0]
    signed_map(ax, mean.reshape(shape), xs, ys, vmax=vmax)
    draw_skeleton(ax, members[0], frame, colour="0.4")
    ax.set_title("the mean field of the ensemble:\nthe field the model is given", fontsize=7.5)
    ax = axes[1, 1]
    im = ax.imshow(variance.reshape(shape), origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                   cmap="Purples", vmin=0.0, aspect="equal", interpolation="bilinear")
    draw_skeleton(ax, members[0], frame, colour="0.4")
    ax.set_title("the variance of the field across the\nconformations: the flexibility map",
                 fontsize=7.5)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    print(f"  ensemble: {len(members)} conformations of {ds.names[i]}; the mean drawn and the field "
          f"of the ensemble mode differ by {gap:.1e}; largest variance {variance.max():.3f} against "
          f"largest field {vmax:.3f}")
    save(fig, path)


def fig_shells(path):
    """Chapter 11: a distribution of solvation shells as one field.  The representative shells of
    one electrolyte of ``examples/electrolyte``, each in the frame of its lithium ion: one shell,
    all of them, the field of the one, and the mean field of all, which is the sample the model
    is given.  The fields are drawn summed along the direction of view, because a shell has no
    plane that holds its atoms."""
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from cmfqsar.cofields import _centres_and_widths
    from cmfqsar.fields import FieldSet, build_fields
    from cmfqsar.shells import ensemble_from_molecules
    name = "LiPF6_EC30_EMC70_m1p110"
    source = os.path.join(ROOT, "examples", "electrolyte", "shells", name + ".sdf")
    mols = [m for m in Chem.SDMolSupplier(source, removeHs=False) if m is not None]
    weights = np.array([float(m.GetProp("shell_weight")) for m in mols])
    fs = FieldSet.create(physchem=["electrostatic", "steric", "hydrophobic", "hba", "hbd"],
                         alpha=0.5, charge_model="gasteiger")
    members = [build_fields(m, fs, f"shell {k + 1}") for k, m in enumerate(mols)]
    ensemble = ensemble_from_molecules(mols, fs, name=name)       # the sample of the benchmark
    first = int(np.argmax(weights))
    f = fs.names.index("electrostatic")
    half = 7.0
    xs = ys = np.linspace(-half, half, 141)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    def projected(mol):
        """The field of `mol` integrated along z: a Gaussian keeps its width in the plane and
        gains the factor sqrt(2 pi / b)."""
        centres, w, b = _centres_and_widths(mol, fs.specs[f], f)
        out = np.zeros_like(X)
        for (cx, cy, _cz), wk, bk in zip(centres, w, b):
            if wk != 0.0:
                out += wk * np.sqrt(2.0 * np.pi / bk) * np.exp(-0.5 * bk * ((X - cx) ** 2 + (Y - cy) ** 2))
        return out

    colours = {"O": "#c0392b", "F": "#2e8b57", "P": "#d98c1f", "Li": "#6a3d9a"}

    def draw(ax, mol, alpha=1.0, marks=True):
        xyz = np.asarray(mol.coords, float)
        heavy = np.array([s != "H" for s in mol.symbols])
        for a, b in mol.bonds:
            if heavy[a] and heavy[b]:
                ax.plot(xyz[[a, b], 0], xyz[[a, b], 1], color="0.35", lw=0.8, alpha=alpha,
                        solid_capstyle="round", zorder=2)
        if marks:
            for element, colour in colours.items():
                sel = np.array([s == element for s in mol.symbols])
                if element != "Li" and sel.any():
                    ax.scatter(xyz[sel, 0], xyz[sel, 1], s=9, color=colour, zorder=3, linewidth=0)

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.6))
    ax = axes[0, 0]
    draw(ax, members[first])
    ax.set_title(f"one shell around a lithium ion\n({mols[first].GetProp('shell_composition')}; "
                 f"ANI is the anion)", fontsize=7.5)
    ax = axes[0, 1]
    contacts = 0
    for k, mol in enumerate(members):
        draw(ax, mol, alpha=float(0.25 + 0.75 * weights[k] / weights.max()), marks=False)
        xyz = np.asarray(mol.coords, float)
        near = np.linalg.norm(xyz, axis=1) < 2.6
        for element in ("O", "F"):
            sel = near & np.array([s == element for s in mol.symbols])
            contacts += int(sel.sum())
            ax.scatter(xyz[sel, 0], xyz[sel, 1], s=11, color=colours[element], zorder=3,
                       linewidth=0, alpha=0.85)
    ax.set_title(f"the {len(members)} representative shells of this electrolyte,\n"
                 "with the atoms that touch the ion", fontsize=7.5)
    one = projected(members[first])
    mean = projected(ensemble)
    vmax = 0.5 * float(np.abs(one).max())
    for ax, Z, title in ((axes[1, 0], one, "the electrostatic field of the one shell"),
                         (axes[1, 1], mean, "the mean field of all shells:\nthe sample the model is given")):
        ax.imshow(Z, origin="lower", extent=(-half, half, -half, half), cmap="RdBu_r", vmin=-vmax,
                  vmax=vmax, interpolation="bilinear")
        ax.set_title(title, fontsize=7.5)
    for ax in axes.ravel():
        ax.scatter([0.0], [0.0], marker="*", s=70, color=colours["Li"], edgecolor="white",
                   linewidth=0.4, zorder=5)
        ax.add_patch(plt.Circle((0.0, 0.0), 2.0, fill=False, color="0.2", lw=0.6, ls="--", zorder=4))
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.tick_params(length=2, labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel("x / Å", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("y / Å", fontsize=7.5)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    print(f"  shells: {name}, {len(members)} members, weights {weights.min():.3f} to "
          f"{weights.max():.3f} (sum {weights.sum():.3f}); the mean field holds "
          f"{ensemble.n_atoms} atoms; {contacts} atoms touch the ion")
    save(fig, path)


def fig_acquire(path):
    """Chapter 9: which molecule to make next, and ten at once.  The ACE set of the Sutherland
    collection, as the benchmark of the section starts it: twenty molecules drawn at random
    that hold no hit are the molecules already made, and the rest are the candidates.  Left, the
    candidates by prediction and posterior standard deviation, with the line of equal score of
    each rule through the candidate that rule takes.  Right, a batch of ten on a map of the
    candidates: the ten best scores, and the ten of the covariance rule."""
    import matplotlib.pyplot as plt
    from cmfqsar import CMFModel
    from cmfqsar.active import acquisition_scores, batch_by_covariance
    from cmfqsar.dataio import load_dataset
    from cmfqsar.evidence import log_evidence
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    ds = load_dataset(os.path.join(ROOT, "examples", "sutherland", "ACE.sdf"))
    y = np.asarray(ds.activity, float)
    fs = FieldSet.create(alpha=0.5, charge_model="file")
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    rng = np.random.default_rng(0)
    hit = y >= np.quantile(y, 0.95)
    have = np.sort(rng.choice(np.where(~hit)[0], size=20, replace=False))
    pool = np.array([i for i in range(len(y)) if i not in set(have.tolist())])
    made = AtomTable.from_molecules([mf[i] for i in have])
    candidates = AtomTable.from_molecules([mf[i] for i in pool])
    probe = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = probe.joint_kernel(probe.prepare_training_kernels(made))
    lams = (1e-3, 1e-2, 0.1, 1.0, 10.0)
    lam = max(lams, key=lambda v: log_evidence(K, y[have], float(v), fit_bias=True))
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": lam})
    model.fit(made, y[have], cv_folds=0)
    mu, cov = model.predict_with_covariance(candidates)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    best = float(y[have].max())
    beta = 1.0
    rules = (("greedy", "the prediction", "C0", "s"),
             ("ucb", "prediction + deviation", "C1", "D"),
             ("ei", "expected improvement", "C3", "o"),
             ("variance", "the deviation", "C2", "^"))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.15))
    ax = axes[0]
    ax.scatter(mu, sd, s=9, color="0.6", linewidth=0, zorder=2)
    gm = np.linspace(mu.min() - 1.9, max(mu.max(), best) + 0.6, 240)
    gs = np.linspace(0.0, sd.max() * 1.12, 200)
    GM, GS = np.meshgrid(gm, gs)
    picks = {}
    for kind, label, colour, marker in rules:
        score = acquisition_scores(kind, mu, sd, best=best, beta=beta)
        j = int(np.argmax(score))
        picks[kind] = j
        surface = acquisition_scores(kind, GM.ravel(), GS.ravel(), best=best, beta=beta)
        ax.contour(GM, GS, surface.reshape(GM.shape), levels=[float(score[j])], colors=[colour],
                   linewidths=0.9, zorder=3)
        ax.scatter([mu[j]], [sd[j]], s=46, marker=marker, facecolor="none", edgecolor=colour,
                   linewidth=1.3, zorder=5, label=label)
    ax.axvline(best, color="0.35", lw=0.7, ls=":")
    ax.text(best + 0.06, 0.98 * gs[-1], "the best made so far", fontsize=6.3, va="top",
            ha="left", rotation=90, color="0.25")
    ax.set_xlim(gm[0], gm[-1])
    ax.set_ylim(0.0, gs[-1])
    ax.set_xlabel("predicted activity")
    ax.set_ylabel("posterior standard deviation")
    ax.set_title(f"{len(pool)} candidates, and the one each rule takes", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", title="the rule maximises",
              title_fontsize=6.5)
    ax.tick_params(length=2)

    # a batch of ten, on the two leading directions of the candidates in the kernel
    ax = axes[1]
    Kc = probe.joint_kernel(probe.prepare_training_kernels(candidates))
    n = len(pool)
    J = np.eye(n) - np.ones((n, n)) / n
    values, vectors = np.linalg.eigh(J @ Kc @ J)
    xy = vectors[:, [-1, -2]] * np.sqrt(np.maximum(values[[-1, -2]], 0.0))
    score = acquisition_scores("ucb", mu, sd, best=best, beta=beta)
    ranked = [int(i) for i in np.argsort(-score)[:10]]
    spread = batch_by_covariance("ucb", mu, cov, 10, best=best, beta=beta)
    ax.scatter(xy[:, 0], xy[:, 1], s=9, color="0.7", linewidth=0, zorder=2)
    ax.scatter(xy[ranked, 0], xy[ranked, 1], s=40, marker="s", facecolor="none", edgecolor="C1",
               linewidth=1.1, zorder=4, label="the ten best scores")
    ax.scatter(xy[spread, 0], xy[spread, 1], s=30, marker="x", color="C4", linewidth=1.3,
               zorder=5, label="ten by the covariance rule")
    ax.set_xlabel("first direction of the candidates")
    ax.set_ylabel("second direction")
    ax.set_title("a batch of ten, on a map of the candidates", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    ax.tick_params(length=2)
    fig.tight_layout(w_pad=1.6)

    def mean_cosine(rows):
        d = np.sqrt(np.diag(Kc))
        C = (Kc / np.outer(d, d))[np.ix_(rows, rows)]
        return float((C.sum() - len(rows)) / (len(rows) * (len(rows) - 1)))

    print(f"  acquire: ACE, {len(have)} made, {len(pool)} candidates, lambda {lam:g}; picks "
          + ", ".join(f"{k} {ds.names[pool[j]]}" for k, j in picks.items())
          + f"; mean cosine inside the batch: best scores {mean_cosine(ranked):.3f}, covariance "
          f"rule {mean_cosine(spread):.3f}; shared {len(set(ranked) & set(spread))}")
    save(fig, path)


def fig_design(path):
    """Chapter 9: the design loop.  Above, its steps as a scheme.  Below, the loop as
    ``cmf design`` runs it on the benzamide series with its default settings: the parent with
    the position that is varied, and the candidates that come back, in the order of their
    expected improvement."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from cmfqsar import CMFModel
    from cmfqsar.dataio import load_dataset
    from cmfqsar.design import design, fragment_smiles, read_fragments
    from cmfqsar.evidence import log_evidence
    from cmfqsar.fields import AtomTable, FieldSet
    from cmfqsar.topomer import (TopomerSpec, attachment_label, choose_core, series_labels,
                                 substituents, topomer_fields)
    ds = load_dataset(os.path.join(ROOT, "examples", "benzamides_synthetic.sdf"))
    y = np.asarray(ds.activity, float)
    mols, names = list(ds.mols), list(ds.names)
    core, _how = choose_core(mols, mode="framework")
    labels = series_labels(mols, core)
    at_site = {}
    for mol in mols:
        for frag in substituents(mol, core):
            at_site.setdefault(attachment_label(frag), set()).add(Chem.MolToSmiles(frag))
    label = max(at_site, key=lambda k: len(at_site[k]))          # the position that varies most
    spec = TopomerSpec(torsion=180.0, seed=0)
    best = None
    for alpha in (0.3, 0.5, 1.0, 2.0):                             # the grids of the command
        fs = FieldSet.create(alpha=alpha, charge_model="gasteiger", param_scheme="rdkit")
        mf, _log = topomer_fields(mols, names, fs, core=core, labels=labels, spec=spec)
        keep = [i for i, m in enumerate(mf) if m is not None]
        probe = CMFModel(fieldset=fs, method="krr", normalize=True)
        table = AtomTable.from_molecules([mf[i] for i in keep])
        K = probe.joint_kernel(probe.prepare_training_kernels(table))
        for lam in (1e-3, 1e-2, 0.1, 1.0):
            ev = log_evidence(K, y[keep], float(lam), fit_bias=True)
            if np.isfinite(ev) and (best is None or ev > best[0]):
                best = (ev, alpha, lam, fs, table, keep)
    _ev, alpha, lam, fs, table, keep = best
    model = CMFModel(fieldset=fs, method="krr", params={"lam": lam}, normalize=True)
    model.fit(table, y[keep], cv_folds=10, cv_seed=0)
    p = int(keep[int(np.argmax(y[keep]))])                         # the most active parent
    frags = sorted(at_site.get(label, [])) + read_fragments("common")
    seen = {fragment_smiles(f) for f in at_site.get(label, [])}
    res = design(model, mols[p], core, label, fragments=frags, labels=labels, acquire="ei",
                 spec=spec, parent_name=names[p], series=mols, series_names=names, series_y=y,
                 known=seen, with_fields=0)
    shown, taken = [], set()
    for cand in res.candidates:                                    # a fragment listed twice is one
        key = cand.smiles
        if key not in taken:
            taken.add(key)
            shown.append(cand)
        if len(shown) == 10:
            break

    sand, green, grey, blue = "#f6ead2", "#dcefdc", "#e9e9e9", "#dbe7f3"
    fig = plt.figure(figsize=(WIDTH, 4.5))
    ax = fig.add_axes((0.0, 0.76, 1.0, 0.24))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    steps = ((grey, "a parent of the\nseries and one of\nits positions"),
             (grey, f"a library of\nfragments\n({len(set(frags))} here)"),
             (sand, "every fragment is\nput there and posed\nby the topomer rule"),
             (green, "a topomer model of\nthe series predicts\neach, with a deviation"),
             (blue, "a rule turns the two\nnumbers into a score,\nand the list is ranked"))
    w, gap, x0, yb, h = 0.168, 0.036, 0.008, 0.16, 0.66
    for k, (colour, text_) in enumerate(steps):
        x = x0 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, yb), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                    facecolor=colour, edgecolor="0.35", linewidth=0.7))
        ax.text(x + w / 2, yb + h / 2, text_, ha="center", va="center", fontsize=6.6,
                linespacing=1.25)
        if k:
            xa, ym = x - gap, yb + h / 2
            ax.plot([xa + 0.004, x - 0.012], [ym, ym], color="0.25", lw=0.9)
            ax.add_patch(Polygon([(x - 0.013, ym - 0.045), (x - 0.013, ym + 0.045), (x - 0.002, ym)],
                                 closed=True, facecolor="0.25", edgecolor="0.25", linewidth=0.5))

    # the parent, drawn flat, with the group at the varied position marked
    ax = fig.add_axes((0.0, 0.02, 0.45, 0.63))
    flat = Chem.RemoveHs(mols[p])
    match = flat.GetSubstructMatch(core)
    rdDepictor.SetPreferCoordGen(False)
    rdDepictor.Compute2DCoords(flat)
    xy = np.array([[q.x, q.y] for q in (flat.GetConformer().GetAtomPosition(i)
                                         for i in range(flat.GetNumAtoms()))])
    Chem.Kekulize(flat, clearAromaticFlags=True)
    centre = xy.mean(axis=0)
    for bond in flat.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color="0.15", lw=1.0, solid_capstyle="round", zorder=2)
        if bond.GetBondTypeAsDouble() == 2.0:
            d = xy[b] - xy[a]
            normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            if np.dot(normal, centre - 0.5 * (xy[a] + xy[b])) < 0:
                normal = -normal
            ia, ib = xy[a] + 0.15 * d + 0.13 * normal, xy[b] - 0.15 * d + 0.13 * normal
            ax.plot([ia[0], ib[0]], [ia[1], ib[1]], color="0.15", lw=1.0, zorder=2)
    for atom in flat.GetAtoms():
        if atom.GetSymbol() != "C":
            i = atom.GetIdx()
            ax.text(xy[i, 0], xy[i, 1], atom.GetSymbol(), ha="center", va="center", fontsize=7.5,
                    zorder=4, bbox=dict(boxstyle="circle,pad=0.12", facecolor="white",
                                        edgecolor="none"))
    site = match[label]                    # the core atom that carries the group
    inside = set(match)
    group, stack = set(), [n.GetIdx() for n in flat.GetAtomWithIdx(site).GetNeighbors()
                           if n.GetIdx() not in inside]
    while stack:
        i = stack.pop()
        if i not in group:
            group.add(i)
            stack += [n.GetIdx() for n in flat.GetAtomWithIdx(i).GetNeighbors()
                      if n.GetIdx() not in inside and n.GetIdx() not in group]
    marked = sorted(group) if group else [site]
    gc = xy[marked].mean(axis=0)
    radius = max(0.75, float(np.linalg.norm(xy[marked] - gc, axis=1).max()) + 0.6)
    ax.add_patch(plt.Circle(gc, radius, facecolor="#f4c7c3", edgecolor="C3", lw=0.9, alpha=0.6,
                            zorder=1))
    ax.set_xlim(xy[:, 0].min() - 1.4, xy[:, 0].max() + 1.4)
    ax.set_ylim(xy[:, 1].min() - 1.6, xy[:, 1].max() + 1.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"the parent, {names[p]}:\nthe group in the circle is replaced", fontsize=7.5)

    # what comes back
    ax = fig.add_axes((0.61, 0.115, 0.375, 0.545))
    rows = np.arange(len(shown))
    for k, cand in enumerate(shown):
        colour = "C0" if cand.known else "C3"
        ax.errorbar([cand.mean], [k], xerr=[cand.sd], fmt="o", ms=4, color=colour, capsize=2,
                    lw=0.9, mfc=colour if cand.known else "white")
    ax.axvline(res.parent_mean, color="0.4", lw=0.7, ls="--")
    ax.set_yticks(rows)
    ax.set_yticklabels(["–" + (c.smiles.split("]", 1)[1] if c.smiles.startswith("[") else c.smiles)
                        for c in shown], fontsize=6.3, family="DejaVu Sans Mono")
    ax.set_ylim(len(shown) - 0.5, -0.5)
    ax.set_xlabel("predicted activity ± its standard deviation\n(dashed: the prediction of "
                  "the parent)", fontsize=6.8)
    ax.tick_params(length=2, labelsize=6.8)
    ax.set_title("the ten fragments ranked first by expected improvement\n(filled: a group the "
                 "series already uses there; open: a new one)", fontsize=7.2, x=0.38)
    print(f"  design: {len(keep)} molecules posed, alpha {alpha:g}, lambda {lam:g}, q2 "
          f"{model.stats.q2:+.3f}; parent {names[p]}, position {label}, {len(set(frags))} fragments, "
          f"{len(res.candidates)} candidates; first {shown[0].smiles} "
          f"({shown[0].mean:.2f} +- {shown[0].sd:.2f})")
    save(fig, path)


def projected_gaussians(centres, weights, widths, frame, xs, ys):
    """A sum of Gaussians integrated along the normal of `frame`, on the grid of its plane.  A
    Gaussian exp(-b r^2 / 2) keeps its width in the plane and gains the factor sqrt(2 pi / b).
    It is the way to draw a field whose centres do not lie in one plane."""
    origin, e1, e2, _e3 = frame
    d = np.asarray(centres, float) - np.asarray(origin, float)
    px, py = d @ e1, d @ e2
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    out = np.zeros_like(X)
    for x0, y0, w, b in zip(px, py, np.asarray(weights, float), np.asarray(widths, float)):
        if w != 0.0:
            out += w * np.sqrt(2.0 * np.pi / b) * np.exp(-0.5 * b * ((X - x0) ** 2 + (Y - y0) ** 2))
    return out


def fig_water(path):
    """Chapter 5: the water as a field.  The hydration sites of one ligand of the mcl1 set, the
    field they make, and the co-field of a model fitted on the water fields alone.  The sites
    are those of ``examples/md/mcl1_water.json``; the fields are drawn summed along the normal of
    the ligand's plane, because the sites do not lie in it."""
    import matplotlib.pyplot as plt
    from cmfqsar import CMFModel
    from cmfqsar import solvent as SV
    from cmfqsar.cofields import _centres_and_widths
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    with open(os.path.join(ROOT, "examples", "md", "mcl1_water.json"), encoding="utf-8") as fh:
        rec = json.load(fh)
    alpha = float(rec["alpha"])
    sites = {}
    for row in rec["ligands"]:
        arr = np.asarray(row["sites"], float)
        if arr.size:
            coeff = np.asarray((row.get("coefficients") or {}).get(f"{alpha:g}", arr[:, 3]), float)
            sites[row["name"]] = (arr[:, :3], coeff, arr[:, 4])
    ds = load_dataset(os.path.join(ROOT, "examples", "openff", "mcl1.sdf"), activity_prop="pAffinity")
    keep = [i for i, n in enumerate(ds.names) if n in sites and np.isfinite(ds.activity[i])]
    names = [ds.names[i] for i in keep]
    y = np.asarray(ds.activity, float)[keep]
    fs_w = SV.solvent_fieldset(alpha=alpha)
    water = [SV.water_field(sites[n][0], sites[n][1], name=n) for n in names]
    table = AtomTable.from_molecules(water)
    best = None
    for lam in (1e-3, 1e-2, 0.1, 1.0, 10.0):
        model = CMFModel(fieldset=fs_w, method="krr", normalize=True, params={"lam": lam})
        model.fit(table, y, cv_folds=10)
        if best is None or model.stats.q2 > best[0]:
            best = (float(model.stats.q2), lam, model)
    q2, lam, model = best
    k = int(np.argmax(y))                                         # the most active ligand
    fs = FieldSet.create(alpha=alpha, charge_model="gasteiger")
    ligand = build_fields(ds.mols[keep[k]], fs, names[k])
    frame = plane_of([ligand])
    centres, occupancy = sites[names[k]][0], sites[names[k]][2]
    sxy = project(frame, centres)
    lxy = project(frame, ligand.coords)
    lo = np.minimum(sxy.min(axis=0), lxy.min(axis=0)) - 3.0
    hi = np.maximum(sxy.max(axis=0), lxy.max(axis=0)) + 3.0
    xs, ys = np.arange(lo[0], hi[0], 0.15), np.arange(lo[1], hi[1], 0.15)
    extent = (xs[0], xs[-1], ys[0], ys[-1])

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.55))
    ax = axes[0]
    draw_skeleton(ax, ligand, frame, colour="0.2")
    ax.scatter(sxy[:, 0], sxy[:, 1], s=260 * occupancy ** 2, facecolor="#9ecae1",
               edgecolor="#2171b5", linewidth=0.8, alpha=0.85, zorder=5)
    ax.set_title(f"the {len(centres)} hydration sites of one\nligand, from its simulation",
                 fontsize=7.5)
    ax = axes[1]
    c0, w0, b0 = _centres_and_widths(water[k], fs_w.specs[0], 0)
    ax.imshow(projected_gaussians(c0, w0, b0, frame, xs, ys), origin="lower", extent=extent,
              cmap="Blues", vmin=0.0, interpolation="bilinear")
    draw_skeleton(ax, ligand, frame, colour="0.2")
    ax.set_title("the water field of that ligand:\na Gaussian on every site", fontsize=7.5)
    ax = axes[2]
    cc, cw, cb = model.cofield_coefficients(0)
    Z = projected_gaussians(cc, cw, cb, frame, xs, ys)
    signed_map(ax, Z, xs, ys)
    draw_skeleton(ax, ligand, frame, colour="0.2")
    ax.set_title(f"the co-field of the water fields of\nall {len(names)} ligands (water alone)",
                 fontsize=7.5)
    for ax in axes:
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=0.8)
    print(f"  water: {len(names)} ligands with sites, alpha {alpha:g}, lambda {lam:g}, q2 of the "
          f"water alone {q2:+.3f}; drawn {names[k]} with {len(centres)} sites, occupancy "
          f"{occupancy.min():.2f} to {occupancy.max():.2f}")
    save(fig, path)


def fig_mask(path):
    """Chapter 9: scoring only where the co-field has weight.  The co-field of the steroid model,
    the mask on the ten atoms that carry the prediction, and their product, the masked
    co-field, all in the mean plane of the series."""
    import matplotlib.pyplot as plt
    from cmfqsar.cofields import evaluate_gaussians, masked_cofield, top_atom_mask
    s, c = steroids(), cofield_model()
    fs, mf, model, frame, xs, ys = c["fs"], c["mf"], c["model"], c["frame"], c["xs"], c["ys"]
    idx = s["train_idx"]
    ref = mf[idx[int(np.argmax(s["y"]))]]
    f = fs.names.index("electrostatic")
    pts, shape = plane_points(frame, xs, ys)
    mask = top_atom_mask(model, [mf[i] for i in idx], k=10)
    mc, mw, mb = mask.expansion()
    region = evaluate_gaussians(pts, mc, mw, mb)
    cofield = cofield_slice(model, f, pts)
    pc, pw, pb = masked_cofield(model, f, mask)
    scale = model.scale[f] if getattr(model, "scale", None) is not None else 1.0
    masked = evaluate_gaussians(pts, pc, pw, pb) * model.fieldset.specs[f].h * scale
    gap = float(np.abs(masked - cofield * region).max())
    vmax = float(np.abs(cofield).max())

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.75))
    signed_map(axes[0], cofield.reshape(shape), xs, ys, vmax=vmax)
    axes[0].set_title("the electrostatic co-field", fontsize=7.5)
    axes[1].imshow(region.reshape(shape), origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                   cmap="Greys", vmin=0.0, vmax=1.0, aspect="equal", interpolation="bilinear")
    mxy = project(frame, mc)
    axes[1].scatter(mxy[:, 0], mxy[:, 1], s=14, facecolor="none", edgecolor="C1", linewidth=0.9,
                    zorder=6)
    axes[1].set_title("the mask: the ten atoms that\ncarry the prediction", fontsize=7.5)
    signed_map(axes[2], masked.reshape(shape), xs, ys, vmax=vmax)
    axes[2].set_title("the masked co-field:\ntheir product", fontsize=7.5)
    for ax in axes:
        draw_skeleton(ax, ref, frame, colour="0.25")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=0.8)
    print(f"  mask: {len(mc)} atoms; the masked co-field and the product of co-field and mask "
          f"differ by {gap:.1e} at most (largest co-field {vmax:.3f})")
    save(fig, path)


def fig_matchedpair(path):
    """Chapter 9: a matched pair as a field placed in space.  The pair of the hif2a set with the
    largest measured change, found and posed as ``benchmarks/run_transformfield.py`` does it:
    the two molecules in the frame of the series, and the difference field of the two groups
    in the frame of their attachment point."""
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from cmfqsar.bioisostere import matched_pairs
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import FieldSet, build_fields
    from cmfqsar.transformfield import pair_field
    ds = load_dataset(os.path.join(ROOT, "examples", "openff", "hif2a.sdf"))
    mols = [Chem.RemoveHs(m) for m in ds.mols]
    names = list(ds.names)
    y = np.asarray(ds.activity, float)
    fs = FieldSet.create(alpha=0.5, charge_model="gasteiger")
    fields = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    pairs = matched_pairs(mols, y, names)
    posed = [(p, t) for p, t in ((p, pair_field(mols, fields, p, "hif2a")) for p in pairs)
             if t is not None]
    pair, t = max(posed, key=lambda q: abs(q[0].delta))
    a, b = int(pair.a), int(pair.b)

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5))
    ax = axes[0]
    frame = plane_of([fields[a], fields[b]])
    draw_skeleton(ax, fields[a], frame, colour="C0")
    draw_skeleton(ax, fields[b], frame, colour="C3")
    for which, atoms, colour in ((a, pair.atoms_a, "C0"), (b, pair.atoms_b, "C3")):
        xy = project(frame, fields[which].coords[np.asarray(atoms, int)])
        ax.scatter(xy[:, 0], xy[:, 1], s=42, facecolor="none", edgecolor=colour, linewidth=1.1,
                   zorder=6)
    ax.set_aspect("equal")
    ax.set_title(f"the pair in the frame of the series\n{names[a]} (blue) and {names[b]} (red): "
                 f"{pair.delta:+.2f}", fontsize=7.5)
    own = (np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    xyz = np.asarray(t.field.coords, float)
    lo, hi = xyz[:, :2].min(axis=0) - 4.0, xyz[:, :2].max(axis=0) + 4.0
    lo, hi = np.minimum(lo, -3.0), np.maximum(hi, 3.0)
    xs, ys = np.arange(lo[0], hi[0], 0.1), np.arange(lo[1], hi[1], 0.1)
    pts, shape = plane_points(own, xs, ys)
    steric = fs.names.index("steric")
    side = np.sign(np.asarray(t.field.weights, float)[:, steric])
    for ax, name in ((axes[1], "steric"), (axes[2], "electrostatic")):
        f = fs.names.index(name)
        signed_map(ax, field_slice(t.field, fs.specs[f], f, pts).reshape(shape), xs, ys)
        for (px, py, _pz), sign, symbol in zip(xyz, side, t.field.symbols):
            ax.text(px, py + (0.55 if sign > 0 else -0.55), symbol, ha="center", va="center",
                    fontsize=7, color="#7f0000" if sign > 0 else "#08306b", fontweight="bold",
                    zorder=6)
            ax.scatter([px], [py], s=9, color="#7f0000" if sign > 0 else "#08306b", zorder=6,
                       linewidth=0)
        ax.scatter([0.0], [0.0], marker="s", s=16, color="black", zorder=6)
        ax.plot([0.0, float(xyz[:, 0].min())], [0.0, 0.0], color="black", lw=0.8, zorder=5)
        ax.set_title(f"the difference field, {name}:\nsecond group minus first", fontsize=7.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=0.8)
    print(f"  matched pair: {len(pairs)} pairs in hif2a, {len(posed)} posed; drawn {names[a]} -> "
          f"{names[b]}, {pair.groups[0]} -> {pair.groups[1]}, measured {pair.delta:+.2f}; "
          f"{t.field.n_atoms} atoms in the difference field")
    save(fig, path)


def fig_ionprobe(path):
    """Chapter 11: an ion as the probe.  Triglyme of ``examples/ether`` in the geometry of its
    lithium complex with the ion taken out: the electrostatic field and the lithium probe field
    in the plane through the place of the ion and the oxygens, with the place the relaxation
    gave the ion and the deepest point of the probe field."""
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from cmfqsar.cofields import field_volume, make_grid
    from cmfqsar.fields import FieldSet, build_fields
    source = os.path.join(ROOT, "examples", "ether", "geometries.sdf")
    mol = next(m for m in Chem.SDMolSupplier(source, removeHs=False)
               if m is not None and m.GetProp("smiles") == "COCCOCCOCCOC")
    ion = np.array([float(v) for v in mol.GetProp("ion_xyz").split()])
    fs_p = FieldSet.create(physchem=[], probes=["lithium"], alpha=1.0, charge_model="gasteiger",
                           param_scheme="rdkit")
    fs_e = FieldSet.create(physchem=["electrostatic"], alpha=1.0, charge_model="gasteiger",
                           param_scheme="rdkit")
    probe, charges = build_fields(mol, fs_p, "triglyme"), build_fields(mol, fs_e, "triglyme")
    # the deepest point of the probe field, searched as benchmarks/run_ether.py searches it
    grid = make_grid(probe.coords, margin=3.0, spacing=0.4)
    volume = field_volume(probe, fs_p.specs[0], 0, grid).ravel()
    deepest = grid.points()[int(np.argmin(volume))]
    # the plane: through the ion, along the two directions in which the oxygens spread most
    oxygens = probe.coords[[s == "O" for s in probe.symbols]]
    _u, _s, vt = np.linalg.svd(np.vstack([oxygens, ion[None, :]]) - ion, full_matrices=False)
    frame = (ion, vt[0], vt[1], np.cross(vt[0], vt[1]))
    xs = ys = np.arange(-6.0, 6.01, 0.1)
    pts, shape = plane_points(frame, xs, ys)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    for ax, mf, spec, title in (
            (axes[0], charges, fs_e.specs[0], "the electrostatic field"),
            (axes[1], probe, fs_p.specs[0], "the lithium probe field:\nthe energy of the ion at each place")):
        Z = field_slice(mf, spec, 0, pts).reshape(shape)
        signed_map(ax, Z, xs, ys)
        draw_skeleton(ax, mf, frame, colour="0.25")
        oxy = project(frame, oxygens)
        ax.scatter(oxy[:, 0], oxy[:, 1], s=22, color="#c0392b", zorder=6, linewidth=0)
        ax.scatter([0.0], [0.0], marker="*", s=120, color="#6a3d9a", edgecolor="white",
                   linewidth=0.5, zorder=8, label="the ion, as the relaxation placed it")
        k = np.unravel_index(int(np.argmin(Z)), Z.shape)
        ax.scatter([xs[k[1]]], [ys[k[0]]], marker="+", s=60, color="black", linewidth=1.3,
                   zorder=9, label="the lowest point of this field in the plane")
        ax.set_title(title, fontsize=7.5)
        axes_in_angstrom(ax, ylabel=ax is axes[0])
    fig.tight_layout(w_pad=1.0, rect=(0, 0.07, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.8, loc="lower center", ncol=2)
    print(f"  ion probe: {mol.GetProp('_Name')} {mol.GetProp('smiles')}; the deepest point of the "
          f"probe field in space lies {np.linalg.norm(deepest - ion):.2f} A from the ion, and "
          f"{np.abs((deepest - ion) @ frame[3]):.2f} A off the drawn plane")
    save(fig, path)


def fig_fukui(path):
    """Chapter 5: why a Fukui function is not a field of atom-centred Gaussians.  The quantum maps
    of the section are not in the repository, so the function drawn is a model with the same
    shape: the square of the Hueckel pi* orbital of acrolein, built from 2p Slater functions.  It
    is fitted with the two bases of ``cmfqsar.anisotropic``, isotropic Gaussians alone and with
    p functions, and all three are drawn in the plane through the chain that stands on the
    molecular plane."""
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from cmfqsar.anisotropic import design, fit_field
    from cmfqsar.probes import basis_widths
    mol = Chem.AddHs(Chem.MolFromSmiles("C=CC=O"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    xyz = np.asarray(mol.GetConformer().GetPositions(), float)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    chain = xyz[heavy]
    centre = chain.mean(axis=0)
    _u, _s, vt = np.linalg.svd(xyz - centre, full_matrices=False)
    e1, e2, normal = vt[0], vt[1], vt[2]                      # the molecule lies in (e1, e2)
    # the Hueckel pi orbitals of C=C-C=O, the oxygen with alpha + beta; the third is pi*
    H = np.array([[0.0, 1, 0, 0], [1, 0.0, 1, 0], [0, 1, 0.0, 1], [0, 0, 1, 1.0]])
    values, vectors = np.linalg.eigh(H)
    lumo = vectors[:, np.argsort(-values)[2]]
    zeta = np.array([{6: 1.625, 8: 2.275}[mol.GetAtomWithIdx(i).GetAtomicNum()] for i in heavy]) / 0.529177

    def model(points):
        psi = np.zeros(len(points))
        for c, z, r0 in zip(lumo, zeta, chain):
            d = points - r0
            psi += c * (d @ normal) * np.exp(-z * np.linalg.norm(d, axis=1))
        return psi ** 2

    span = np.arange(-4.0, 4.01, 0.2)
    G = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    cloud = centre + G[:, [0]] * e1 + G[:, [1]] * e2 + G[:, [2]] * normal
    target = model(cloud)
    widths = basis_widths(4)
    fit_s = fit_field(target, cloud, xyz, widths, p_type=False, name="s")
    fit_sp = fit_field(target, cloud, xyz, widths, p_type=True, name="s and p")
    off = float(np.sum(target[np.abs(G[:, 2]) > 0.5] ** 2) / np.sum(target ** 2))

    def fitted(points, fit, p_type):
        B = design(points, xyz, widths, p_type=p_type).reshape(len(points), -1)
        return B @ np.asarray(fit.coeffs, float).reshape(-1)

    # the cut: the plane that holds the C=C bond and stands on the molecular plane
    c1, c2 = chain[0], chain[1]
    along = (c2 - c1) / np.linalg.norm(c2 - c1)
    middle = 0.5 * (c1 + c2)
    frame = (middle, along, normal, np.cross(along, normal))
    xs, ys = np.arange(-2.6, 2.61, 0.05), np.arange(-2.2, 2.21, 0.05)
    pts, shape = plane_points(frame, xs, ys)
    maps = [model(pts), fitted(pts, fit_s, False), fitted(pts, fit_sp, True)]
    vmax = float(maps[0].max())
    titles = ("a model Fukui function:\nπ* of acrolein, squared",
              f"its best fit by Gaussians\non the atoms (error {100 * fit_s.rel_error:.0f} %)",
              f"the fit with p functions\nadded (error {100 * fit_sp.rel_error:.0f} %)")
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.95), gridspec_kw=dict(width_ratios=(1, 1, 1, 1.15)))
    bond = np.array([[(c1 - middle) @ along, 0.0], [(c2 - middle) @ along, 0.0]])
    for ax, Z, title in zip(axes[:3], maps, titles):
        signed_map(ax, Z.reshape(shape), xs, ys, vmax=vmax)
        ax.plot(bond[:, 0], bond[:, 1], color="black", lw=1.0, zorder=4)
        ax.scatter(bond[:, 0], bond[:, 1], s=12, color="black", zorder=5)
        ax.set_title(title, fontsize=6.8)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("across the plane", fontsize=6.8)
    # the same three functions on the line through the terminal carbon, across the plane
    z = np.arange(-2.2, 2.21, 0.02)
    line = c1 + z[:, None] * normal
    ax = axes[3]
    ax.plot(z, model(line), color="black", lw=1.2, label="the model")
    ax.plot(z, fitted(line, fit_s, False), color="C0", lw=1.0, label="Gaussians")
    ax.plot(z, fitted(line, fit_sp, True), color="C1", lw=1.0, ls="--", label="with p functions")
    ax.axvline(0.0, color="0.6", lw=0.5)
    ax.set_xlabel("height above the plane / Å", fontsize=6.8)
    ax.set_title("through the end carbon:\nthe node is in the plane", fontsize=6.8)
    ax.tick_params(length=2, labelsize=6.3)
    ax.set_yticks([])
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.55 * (high - low))
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", handlelength=1.6)
    fig.tight_layout(w_pad=0.6)
    print(f"  fukui model: {off:.2f} of the squared function lies more than 0.5 A off the plane; "
          f"relative error {fit_s.rel_error:.3f} with s, {fit_sp.rel_error:.3f} with s and p")
    save(fig, path)


def fig_complex(path):
    """Chapter 5: a coordination complex.  The first nickel complex of ``examples/complexes_ni``
    as the builder of ``cmfqsar.complexes`` reads it, in three pictures, and the whole series in
    the frame the metal gives."""
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from cmfqsar.complexes import align_on_first_sphere, donor_indices, metal_index
    source = os.path.join(ROOT, "examples", "complexes_ni", "ni_bpp.sdf")
    mols = [m for m in Chem.SDMolSupplier(source, removeHs=False, sanitize=False) if m is not None]
    aligned, _rmsd = align_on_first_sphere(mols, ref_index=0)
    mol = aligned[0]
    mi, donors = metal_index(mol), donor_indices(mol)
    xyz = np.asarray(mol.GetConformer().GetPositions(), float)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    heavy = np.array([s != "H" for s in symbols])
    # the view: the plane of the three nitrogen donors and the metal
    nitrogens = [i for i in donors if symbols[i] == "N"]
    _u, _s, vt = np.linalg.svd(xyz[nitrogens + [mi]] - xyz[mi], full_matrices=False)
    # an oblique view: what stands on that plane is moved sideways, so that all six donors show
    frame = (xyz[mi], vt[0] + 0.45 * vt[2], vt[1] + 0.25 * vt[2], vt[2])
    xy = project(frame, xyz)
    rest = Chem.RWMol(mol)
    rest.RemoveAtom(mi)
    pieces = Chem.GetMolFrags(rest.GetMol())                       # atom indices without the metal
    back = [i for i in range(mol.GetNumAtoms()) if i != mi]
    fragments = [[back[j] for j in piece] for piece in pieces]
    charges = [sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in frag) for frag in fragments]
    palette = ["C0", "C2", "C1", "C4", "C5", "C6"]
    element = {"N": "#1f4e9c", "O": "#c0392b", "Ni": "#6a3d9a"}

    def bonds(ax, colour_of, metal_bonds="line", orders=False):
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if not (heavy[a] and heavy[b]):
                continue
            to_metal = mi in (a, b)
            if to_metal and metal_bonds == "none":
                continue
            colour = "0.3" if to_metal else colour_of(a)
            if to_metal and metal_bonds == "arrow":
                donor = b if a == mi else a
                ax.annotate("", xy=xy[mi] + 0.22 * (xy[donor] - xy[mi]), xytext=xy[donor],
                            arrowprops=dict(arrowstyle="-|>", color="0.2", lw=0.9), zorder=3)
                continue
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=colour, lw=1.0, zorder=2,
                    ls="--" if to_metal else "-", solid_capstyle="round")
            if orders and bond.GetBondTypeAsDouble() > 1.4 and not to_metal:
                d = xy[b] - xy[a]
                n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
                ia, ib = xy[a] + 0.15 * d + 0.11 * n, xy[b] - 0.15 * d + 0.11 * n
                ax.plot([ia[0], ib[0]], [ia[1], ib[1]], color=colour, lw=0.8, zorder=2)

    def atoms(ax, show_metal=True):
        for i in np.where(heavy)[0]:
            if i == mi and not show_metal:
                continue
            if symbols[i] in element:
                ax.text(xy[i, 0], xy[i, 1], symbols[i], ha="center", va="center", fontsize=6.5,
                        color=element[symbols[i]], fontweight="bold", zorder=5,
                        bbox=dict(boxstyle="circle,pad=0.1", facecolor="white", edgecolor="none"))

    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.25))
    ax = axes[0]
    bonds(ax, lambda a: "0.3")
    atoms(ax)
    ax.set_title("1. the bonds are\nfound by distance", fontsize=7.2)
    ax = axes[1]
    owner = {i: k for k, frag in enumerate(fragments) for i in frag}
    bonds(ax, lambda a: palette[owner[a] % len(palette)], metal_bonds="none", orders=True)
    atoms(ax, show_metal=False)
    for k, frag in enumerate(fragments):
        members = [i for i in frag if heavy[i]]
        centre = xy[members].mean(axis=0)
        shift = (0.0, -1.0) if len(members) > 1 else (0.75, 0.0)
        ax.text(centre[0] + shift[0], centre[1] + shift[1],
                f"{charges[k]:+d}" if charges[k] else "0", ha="center", va="center",
                fontsize=6.5, color=palette[k % len(palette)])
    ax.set_title("2. metal out: every\nligand gets its charge", fontsize=7.2)
    ax = axes[2]
    bonds(ax, lambda a: "0.3", metal_bonds="arrow", orders=True)
    atoms(ax)
    ax.text(xy[mi, 0] - 0.85, xy[mi, 1] + 0.8, f"{mol.GetAtomWithIdx(mi).GetFormalCharge():+d}",
            ha="center", va="center", fontsize=6.5, color=element["Ni"])
    ax.set_title("3. metal back: a dative\nbond from every donor", fontsize=7.2)
    ax = axes[3]
    for other in aligned:
        if other is None:
            continue
        oxyz = np.asarray(other.GetConformer().GetPositions(), float)
        oxy = project(frame, oxyz)
        oheavy = np.array([a.GetAtomicNum() > 1 for a in other.GetAtoms()])
        omi = metal_index(other)
        for bond in other.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if oheavy[a] and oheavy[b] and omi not in (a, b):
                ax.plot(oxy[[a, b], 0], oxy[[a, b], 1], color="0.45", lw=0.4, alpha=0.5, zorder=2)
        od = donor_indices(other)
        for i in od:
            ax.scatter([oxy[i, 0]], [oxy[i, 1]], s=7, zorder=4, linewidth=0,
                       color=element.get(other.GetAtomWithIdx(i).GetSymbol(), "0.3"))
    ax.scatter([0.0], [0.0], s=26, color=element["Ni"], zorder=5)
    ax.set_title(f"the frame of the metal:\nall {len(mols)} complexes", fontsize=7.2)
    span = np.abs(xy[heavy]).max() + 1.2
    for ax in axes:
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=0.5)
    print(f"  complex: {mol.GetProp('_Name')}, {len(donors)} donors, {len(fragments)} ligands with "
          f"charges {charges}, metal {mol.GetAtomWithIdx(mi).GetFormalCharge():+d}; "
          f"{sum(1 for m in aligned if m is not None)} complexes in the frame")
    save(fig, path)


def fig_metabolism(path):
    """Chapter 11: the site-of-metabolism model read on two drug molecules: every heavy atom
    coloured by its out-of-fold score, the measured sites ringed."""
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    import run_metabolism as rm
    from cmfqsar.localenv import LOCAL_SPEC, count_features, environments, local_features, types_of
    from cmfqsar.locallarge import choose_lam_primal, normalised_rows
    mols, fields, targets = rm.read()
    types = types_of(fields, min_atoms=10)
    Z_count, _ = normalised_rows(count_features(fields, targets, types), scale=np.ones(len(types) * 0 + count_features(fields, targets, types).shape[1]))
    rho = 0.25
    Z_field, _ = normalised_rows(local_features(environments(fields, targets, rho=rho)),
                                 n_bins=LOCAL_SPEC.n_bins)
    Z = np.hstack([Z_count, Z_field]) / np.sqrt(2.0)
    res = choose_lam_primal(Z, targets, lams=(0.01,), folds=10, seed=0, name="count and field")
    pred, y = np.asarray(res.predicted, float), np.asarray(targets.y, float)
    molecule, atom = np.asarray(targets.molecule, int), np.asarray(targets.atom, int)
    # one molecule whose top-ranked atom is a site and one where neither of the top two is
    picks = {}
    for m in range(len(mols)):
        rows = np.where(molecule == m)[0]
        if len(rows) < 14 or len(rows) > 24 or y[rows].sum() == 0:
            continue
        order = rows[np.argsort(-pred[rows])]
        if "hit" not in picks and y[order[0]] > 0:
            picks["hit"] = m
        if "miss" not in picks and y[order[0]] == 0 and y[order[1]] == 0:
            picks["miss"] = m
        if len(picks) == 2:
            break
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    for ax, (what, m) in zip(axes, sorted(picks.items())):
        rows = np.where(molecule == m)[0]
        rd = Chem.Mol(mols[m])
        rdDepictor.Compute2DCoords(rd)
        xy = rd.GetConformer().GetPositions()[:, :2]
        heavy = [a.GetIdx() for a in rd.GetAtoms() if a.GetAtomicNum() > 1]
        for b in rd.GetBonds():
            p, q = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if p in heavy and q in heavy:
                ax.plot(xy[[p, q], 0], xy[[p, q], 1], color="0.55", lw=1.2, zorder=1)
        score = pred[rows]
        sc = ax.scatter(xy[atom[rows], 0], xy[atom[rows], 1], c=score, cmap="viridis", s=70,
                        edgecolor="black", linewidth=0.4, zorder=3)
        sites = rows[y[rows] > 0]
        ax.scatter(xy[atom[sites], 0], xy[atom[sites], 1], s=190, facecolor="none",
                   edgecolor="C3", linewidth=1.8, zorder=4)
        top = rows[np.argsort(-score)[:2]]
        for rank, r in enumerate(top, 1):
            ax.annotate(str(rank), xy[atom[r]], xytext=(7, 7), textcoords="offset points",
                        fontsize=8, fontweight="bold")
        for a in heavy:
            sym = rd.GetAtomWithIdx(a).GetSymbol()
            if sym != "C":
                ax.annotate(sym, xy[a], xytext=(-7, -9), textcoords="offset points", fontsize=6.5,
                            color="0.3")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{targets.names[m] if hasattr(targets, 'names') else rd.GetProp('_Name')}: "
                     + ("a site is ranked first" if what == "hit" else "the two top atoms are not sites"),
                     fontsize=8)
        fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02).ax.tick_params(labelsize=6.5)
    fig.suptitle("out-of-fold score of every heavy atom; red ring: a measured site of metabolism; "
                 "1, 2: the atoms ranked highest", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    hits = rm.per_molecule_hits(targets, pred, 2)
    print(f"  metabolism: count and field at rho {rho}, top-2 hit rate {hits.mean():.3f}; "
          f"drawn {picks}")
    save(fig, path)


def fig_pharmacophore(path):
    """Chapter 8: the pharmacophore read out of the steroid co-fields, drawn over the two
    co-field maps as circles of the features' tolerance radii."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from cmfqsar.pharmacophore import colour_of, extract
    s, c = steroids(), cofield_model()
    fs, mf, model, frame, xs, ys = c["fs"], c["mf"], c["model"], c["frame"], c["xs"], c["ys"]
    ref = mf[s["train_idx"][int(np.argmax(s["y"]))]]
    feats = extract(model, level=0.5, max_features=8)
    origin, e1, e2, e3 = frame
    pts, shape = plane_points(frame, xs, ys)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    seen = {}
    for ax, name in zip(axes, ("electrostatic", "steric")):
        f = fs.names.index(name)
        Z = cofield_slice(model, f, pts).reshape(shape)
        im = signed_map(ax, Z, xs, ys)
        draw_skeleton(ax, ref, frame, colour="0.3")
        for ft in feats:
            cx, cy = project(frame, ft.centre[None, :])[0]
            height = abs(float((ft.centre - origin) @ e3))
            alpha = max(0.3, 1.0 - height / 3.0)
            patch = Circle((cx, cy), ft.radius, fill=False, edgecolor=colour_of(ft), lw=1.6,
                           ls="--" if ft.kind == "exclusion" else "-", alpha=alpha, zorder=6)
            ax.add_patch(patch)
            seen.setdefault(ft.kind, colour_of(ft))
        ax.set_title(f"{fs.specs[f].label} co-field", fontsize=9)
        axes_in_angstrom(ax, ylabel=ax is axes[0])
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03).ax.tick_params(labelsize=7)
    names = {"donor_bad": "donor disfavoured", "acceptor_bad": "acceptor disfavoured",
             "hydrophobic_bad": "hydrophobe disfavoured", "exclusion": "exclusion volume"}
    handles = [plt.Line2D([], [], color=col, lw=1.6, ls="--" if kind == "exclusion" else "-",
                          label=names.get(kind, kind)) for kind, col in seen.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("the pharmacophore read out of the steroid co-fields, over two of them",
                 fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    print(f"  pharmacophore: {len(feats)} features: " + "; ".join(
        f"{ft.kind} ({ft.field}) r {ft.radius:.1f}" for ft in feats))
    save(fig, path)


def fig_pocket(path):
    """Chapter 12: the electrostatic field of the most active CDK8 ligand and of its pocket in
    the ligand's mean plane, and where their product says the two are complementary."""
    import matplotlib.pyplot as plt
    from cmfqsar.complementarity import complementarity, pocket_fields
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import FieldSet, build_fields
    from cmfqsar.protein import read_pdb
    ds = load_dataset(os.path.join(ROOT, "examples", "openff", "cdk8.sdf"))
    protein = read_pdb(os.path.join(ROOT, "examples", "pockets", "openff", "cdk8.pdb"))
    fs = FieldSet.create()
    pocket = pocket_fields(protein, fs)
    i = int(np.nanargmax(ds.activity))
    lig = build_fields(ds.mols[i], fs, ds.names[i])
    comp = complementarity(lig, pocket, fs)
    f = fs.names.index("electrostatic")
    frame = plane_of([lig])
    xy = project(frame, lig.coords)
    lo, hi = xy.min(axis=0) - 4.0, xy.max(axis=0) + 4.0
    xs, ys = np.arange(lo[0], hi[0], 0.15), np.arange(lo[1], hi[1], 0.15)
    pts, shape = plane_points(frame, xs, ys)
    X = field_slice(lig, fs.specs[f], f, pts).reshape(shape)
    P = field_slice(pocket, fs.specs[f], f, pts).reshape(shape)
    origin, e1, e2, e3 = frame
    near = np.abs((np.asarray(pocket.coords) - origin) @ e3) < 1.5
    pxy = project(frame, np.asarray(pocket.coords)[near])
    inside = (pxy[:, 0] > xs[0]) & (pxy[:, 0] < xs[-1]) & (pxy[:, 1] > ys[0]) & (pxy[:, 1] < ys[-1])
    pxy = pxy[inside]
    panels = [(X, "ligand field $X_f$"), (P, "pocket field"), (-X * P, "$-X_f \\times$ pocket: complementary where red")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.15))
    for ax, (Z, title) in zip(axes, panels):
        im = signed_map(ax, Z, xs, ys)
        ax.scatter(pxy[:, 0], pxy[:, 1], s=6, color="0.35", zorder=2)
        draw_skeleton(ax, lig, frame)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03).ax.tick_params(labelsize=6)
    fig.suptitle(f"CDK8, {lig.name} (pAffinity {ds.activity[i]:.1f}) in its pocket: the "
                 "electrostatic field of each side in the ligand's mean plane\n"
                 "grey: pocket atoms within 1.5 Å of the plane", fontsize=8, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    print(f"  pocket: {pocket.n_atoms} pocket atoms, ligand {lig.name}; {comp.describe()[:160]}")
    save(fig, path)


def steroid_kernels():
    """The joint training kernel of the steroid training set at every attenuation of the search
    grid, built once: [(alpha, K)]."""
    if "steroid_kernels" in _CACHE:
        return _CACHE["steroid_kernels"]
    from cmfqsar import CMFModel
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    s = steroids()
    alphas = sorted({r["alpha"] for r in s["search"].table})
    out = []
    for a in alphas:
        fs = FieldSet.create(alpha=a)
        table = AtomTable.from_molecules([build_fields(s["ds"].mols[i], fs, s["ds"].names[i])
                                          for i in s["train_idx"]])
        out.append((a, CMFModel(fieldset=fs, method="krr", normalize=True).training_kernel(table)))
    _CACHE["steroid_kernels"] = out
    return out


def fig_scramble(path):
    """Chapter 7: y-randomisation of the steroid model, with the hyper-parameters held fixed and
    with the search repeated inside every shuffle."""
    import matplotlib.pyplot as plt
    from cmfqsar.validation import scramble_kernel, scramble_selection
    s = steroids()
    y, model = s["y"], s["model"]
    kernels = steroid_kernels()
    alpha, lam = float(s["fs"].specs[0].alpha), float(model.params["lam"])
    lams = sorted({r["lam"] for r in s["search"].table})
    K_best = next(K for a, K in kernels if a == alpha)
    runs = 199
    fixed = scramble_kernel(K_best, y, "krr", {"lam": lam}, folds=10, runs=runs, seed=0)
    redone = scramble_selection([K for _, K in kernels], y, "krr",
                                param_grid=[{"lam": l} for l in lams], folds=10, runs=runs, seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharex=True)
    for ax, res, title in zip(axes, (fixed, redone),
                              (f"hyper-parameters fixed at α = {alpha:g}, λ = {lam:g}",
                               f"the search over {len(kernels)} × {len(lams)} grid points redone")):
        above = int(np.sum(res.scrambled >= res.observed))
        p = (above + 1) / (res.runs + 1)
        ax.hist(res.scrambled, bins=25, color="0.6", edgecolor="white")
        ax.axvline(res.observed, color="C3", lw=1.5)
        ax.text(res.observed - 0.03, ax.get_ylim()[1] * 0.95, f"real q² = {res.observed:.2f}",
                color="C3", ha="right", va="top", fontsize=7.5)
        ax.set_title(f"{title}\nshuffled: mean {res.scrambled.mean():+.2f}, largest "
                     f"{res.scrambled.max():+.2f}, p = {p:.3f}", fontsize=8)
        ax.set_xlabel("cross-validated q² of the shuffled activities")
        ax.tick_params(length=2)
        plain = title.replace("α", "alpha").replace("λ", "lambda").replace("×", "x")
        print(f"  y-randomisation ({plain}): observed {res.observed:.3f}, {res.runs} shuffles, "
              f"mean {res.scrambled.mean():+.3f}, sd {res.scrambled.std():.3f}, "
              f"largest {res.scrambled.max():+.3f}, p {p:.3f}")
    axes[0].set_ylabel(f"shuffles out of {runs}")
    fig.tight_layout()
    save(fig, path)


def fig_evidence(path):
    """Chapter 7: the two criteria over the same grid of the steroid training set, the
    cross-validated q² and the log marginal likelihood."""
    import matplotlib.pyplot as plt
    from cmfqsar.evidence import evidence_grid
    s = steroids()
    table = s["search"].table
    alphas = sorted({r["alpha"] for r in table})
    lams = sorted({r["lam"] for r in table})
    Q = np.full((len(lams), len(alphas)), np.nan)
    for r in table:
        Q[lams.index(r["lam"]), alphas.index(r["alpha"])] = r["q2"]
    a_best, l_best, surface = evidence_grid(steroid_kernels(), s["y"], lams)
    E = np.full_like(Q, np.nan)
    for (a, l), e in surface.items():
        E[lams.index(l), alphas.index(a)] = e
    E = E - np.nanmax(E)                                   # relative to the best point
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    for ax, Z, title, fmt, floor in zip(axes, (Q, E), ("cross-validated q²",
                                                        "log evidence, relative to its maximum"),
                                        ("{:+.2f}", "{:.0f}"), (-0.5, -30.0)):
        vmin = max(np.nanmin(Z), floor)
        im = ax.imshow(Z, origin="lower", cmap="viridis", vmin=vmin, vmax=np.nanmax(Z), aspect="auto")
        ax.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas], fontsize=6.5)
        ax.set_yticks(range(len(lams)), [f"{l:g}" for l in lams], fontsize=6.5)
        ax.set_xlabel("attenuation α")
        if ax is axes[0]:
            ax.set_ylabel("ridge λ")
        best = np.unravel_index(np.nanargmax(Z), Z.shape)
        for i in range(len(lams)):
            for j in range(len(alphas)):
                if np.isfinite(Z[i, j]):
                    colour = "white" if Z[i, j] < vmin + 0.6 * (np.nanmax(Z) - vmin) else "black"
                    label = fmt.format(max(Z[i, j], floor)).replace("-0", "0")
                    ax.text(j, i, label, ha="center", va="center", fontsize=5.5, color=colour,
                            fontweight="bold" if (i, j) == best else "normal")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03).ax.tick_params(labelsize=6.5)
    q_best = np.unravel_index(np.nanargmax(Q), Q.shape)
    print(f"  evidence: best by q2 alpha {alphas[q_best[1]]:g}, lambda {lams[q_best[0]]:g}; "
          f"best by evidence alpha {a_best:g}, lambda {l_best:g}; "
          f"q2 there {Q[lams.index(l_best), alphas.index(a_best)]:+.3f}")
    fig.tight_layout()
    save(fig, path)


def fig_scatter(path):
    """Chapter 7: observed against predicted for the steroid model, cross-validation and test set.

    The model is the one of the search figure, at the best point of the grid, so the numbers in
    the legend are the ones the Statistics tab shows after a search on that set.
    """
    import matplotlib.pyplot as plt
    from cmfqsar.fields import AtomTable
    s = steroids()
    ds, mf, model = s["ds"], s["mf"], s["model"]
    te = list(ds.test_indices)
    y_test = ds.activity[te]
    yp = model.evaluate_test(AtomTable.from_molecules([mf[i] for i in te]), y_test)
    st = model.stats
    ycv = np.asarray(st.extra["y_cv"], float)
    alpha, lam = s["fs"].specs[0].alpha, float(model.params["lam"])
    worst = int(np.argmax(np.abs(yp - y_test)))
    print(f"  scatter: alpha = {alpha:g}, lambda = {lam:g}, q2 = {st.q2:.3f}, R2pred = {st.r2_pred:.3f} "
          f"(test variance) / {st.r2_pred_train_mean:.3f} (training mean), RMSEpred = {st.rmse_pred:.3f}; "
          f"largest error {ds.names[te[worst]]} ({yp[worst] - y_test[worst]:+.2f})")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, WIDTH * 0.62))
    values = np.concatenate([s["y"], ycv, y_test, yp])
    lo, hi = values.min() - 0.3, values.max() + 0.3
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, zorder=1)
    ax.scatter(s["y"], ycv, s=24, marker="o", color="C0", edgecolor="white", linewidth=0.5,
               zorder=2, label=f"cross-validation, q² = {st.q2:.2f}")
    ax.scatter(y_test, yp, s=28, marker="s", color="C3", edgecolor="white", linewidth=0.5,
               zorder=3, label=f"test set, R²pred = {st.r2_pred:.2f} or {st.r2_pred_train_mean:.2f}")
    ax.annotate(ds.names[te[worst]], (y_test[worst], yp[worst]), xytext=(5, 6),
                textcoords="offset points", fontsize=7, color="C3")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("observed pK")
    ax.set_ylabel("predicted pK")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"steroid set: {len(s['y'])} training and {len(te)} test molecules\n"
                 f"kernel ridge regression, α = {alpha:g}, λ = {lam:g}")
    fig.tight_layout()
    save(fig, path)


def fig_roc(path):
    """Chapter 9: the ROC curves of the kernel Fisher discriminant on AChE, on the split of the file."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from cmfqsar import CMFModel
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    threshold, alpha, lam = 6.84, 0.5, 0.01
    ds = load_dataset(os.path.join(ROOT, "examples", "sutherland", "AChE.sdf"))
    labels = np.where(ds.activity >= threshold, 1.0, -1.0)
    fs = FieldSet.create(alpha=alpha, charge_model="file")
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    tr, te = list(ds.train_indices), list(ds.test_indices)
    model = CMFModel(fieldset=fs, task="classification", method="kfda", normalize=True,
                     params={"lam": lam})
    model.fit(AtomTable.from_molecules([mf[i] for i in tr]), labels[tr], cv_folds=5)
    model.evaluate_test(AtomTable.from_molecules([mf[i] for i in te]), labels[te])
    st = model.stats
    curves = [("fit", labels[tr], st.extra["y_fit"], st.fit),
              (f"{st.cv_folds}-fold cross-validation", labels[tr], st.extra["y_cv"], st.cv),
              ("test set", labels[te], st.extra["y_test_pred"], st.test)]
    n_act = int(np.sum(labels[tr] > 0)), int(np.sum(labels[te] > 0))
    print(f"  ROC on AChE: {n_act[0]} of {len(tr)} training and {n_act[1]} of {len(te)} test "
          f"molecules active above {threshold}; AUC "
          + ", ".join(f"{label} {scores['auc']:.3f}" for label, _, _, scores in curves))
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, WIDTH * 0.62))
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--", zorder=1)
    for k, (label, y, decision, scores) in enumerate(curves):
        fp, tp, _ = roc_curve((np.asarray(y) > 0).astype(int), np.asarray(decision, float))
        ax.plot(fp, tp, lw=1.4, color=f"C{k}", zorder=2 + k,
                label=f"{label}, AUC = {scores['auc']:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"AChE: {n_act[0]} of {len(tr)} training and {n_act[1]} of {len(te)} test "
                 f"molecules active\nkernel Fisher discriminant, α = {alpha:g}, λ = {lam:g}")
    fig.tight_layout()
    save(fig, path)


def rule_molecules():
    """The two test molecules of the steroid set that the co-field model predicts highest and
    lowest, with their predictions: the examples of the decomposition rules."""
    if "rule_molecules" in _CACHE:
        return _CACHE["rule_molecules"]
    from cmfqsar.fields import AtomTable
    s, c = steroids(), cofield_model()
    test = list(s["ds"].test_indices)
    yp = c["model"].predict(AtomTable.from_molecules([c["mf"][i] for i in test]))
    order = np.argsort(yp)
    out = {"high": test[order[-1]], "low": test[order[0]],
           "pred": {test[k]: float(yp[k]) for k in range(len(test))}}
    _CACHE["rule_molecules"] = out
    return out


def fig_rule1(path):
    """Chapter 8, rule 1: the co-field is a weighted sum of the training molecules' fields."""
    import matplotlib.pyplot as plt
    from cmfqsar.cofields import evaluate_gaussians
    s, c = steroids(), cofield_model()
    fs, mf, model, frame, xs, ys = c["fs"], c["mf"], c["model"], c["frame"], c["xs"], c["ys"]
    f = fs.names.index("electrostatic")
    pts, shape = plane_points(frame, xs, ys)
    centres, coeffs, widths, index = model.cofield_basis(f)
    coef = np.asarray(model.coef, float)
    unit = float(fs.specs[f].h) * float(model.scale[f])

    def term(mask):
        v = evaluate_gaussians(pts, centres[mask], coeffs[mask] * coef[index[mask]], widths[mask])
        return (v * unit).reshape(shape)

    # the weight of training molecule j in the co-field is a_j s_j, with s_j the normalisation
    # of its own field; the three largest in magnitude are drawn, the other terms are summed
    weights = coef / np.sqrt(np.asarray(model.self_diag, float)[f])
    top = list(np.argsort(-np.abs(weights))[:3])
    maps = [term(index == j) for j in top]
    total = term(np.ones(len(index), bool))
    vmax = max(np.abs(m).max() for m in maps + [total])
    names = [mf[s["train_idx"][j]].name for j in top]
    print("  rule 1: electrostatic co-field; largest weights "
          + ", ".join(f"{n} ({w:+.2f})" for n, w in zip(names, weights[top])))
    # the axes are placed by hand, in inches, so that the figure is exactly two rows of maps of
    # the shape of the series with their titles, and no empty band is left between the rows
    ratio = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    pw = 0.27 * WIDTH
    ph = pw * ratio
    title_h, gap, top_h, bottom_h = 0.38, 0.12, 0.28, 0.06
    H = top_h + 2 * (title_h + ph) + gap + bottom_h
    fig = plt.figure(figsize=(WIDTH, H))
    lefts = [0.03 * WIDTH + k * (pw + 0.065 * WIDTH) for k in range(3)]
    y_top = H - top_h - title_h - ph
    y_bottom = bottom_h
    boxes = [(lefts[k], y_top) for k in range(3)] + [(lefts[1], y_bottom)]
    axes = [fig.add_axes([x / WIDTH, y / H, pw / WIDTH, ph / H]) for x, y in boxes]
    for ax, Z, title in zip(axes, maps + [total],
                            [f"{n}\n$a_j s_j = {w:+.2f}$" for n, w in zip(names, weights[top])]
                            + ["electrostatic co-field $C_f$"]):
        im = signed_map(ax, Z, xs, ys, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for j, ax in zip(top, axes[:3]):
        draw_skeleton(ax, mf[s["train_idx"][j]], frame)
    for k in (0, 1):
        left, right = axes[k].get_position(), axes[k + 1].get_position()
        fig.text((left.x1 + right.x0) / 2, (left.y0 + left.y1) / 2, "+", ha="center",
                 va="center", fontsize=13)
    bottom = axes[3].get_position()
    fig.text(bottom.x0 - 0.03, (bottom.y0 + bottom.y1) / 2,
             f"+ the {len(coef) - 3} other\ntraining molecules\n$=$", ha="right", va="center",
             fontsize=8)
    cax = fig.add_axes([bottom.x1 + 0.02, bottom.y0, 0.015, bottom.y1 - bottom.y0])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=6.5)
    fig.suptitle("rule 1: $C_f(\\mathbf{r}) = \\sum_j a_j s_j X_{fj}(\\mathbf{r})$, "
                 f"the steroid model, each term scaled by $h_f c_f$", fontsize=9,
                 y=1 - 0.5 * top_h / H, va="center")
    save(fig, path)


def fig_rule2(path):
    """Chapter 8, rule 2: a prediction is the bias plus one contribution per training molecule."""
    import matplotlib.pyplot as plt
    s, c, r = steroids(), cofield_model(), rule_molecules()
    mf, model, ds = c["mf"], c["model"], s["ds"]
    i = r["high"]
    contrib = model.training_contributions(mf[i])
    bias, pred = float(model.bias), r["pred"][i]
    assert abs(bias + contrib.sum() - pred) < 1e-6
    order = np.argsort(contrib)
    labels = [f"{ds.names[s['train_idx'][j]]} ({s['y'][j]:.2f})" for j in order]
    print(f"  rule 2: {mf[i].name}, prediction {pred:.2f} = bias {bias:.2f} + "
          f"{contrib.sum():+.2f}; largest {labels[-1]} {contrib[order[-1]]:+.2f}")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.85, 0.165 * len(contrib) + 1.1))
    y = np.arange(len(contrib))
    ax.barh(y, contrib[order], color=np.where(contrib[order] >= 0, "C3", "C0"), height=0.7)
    ax.set_yticks(y, labels, fontsize=7)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("contribution $a_j K(X, X_j)$ to the predicted pK")
    ax.set_title(f"rule 2: {mf[i].name}, prediction {pred:.2f} = bias {bias:.2f} "
                 f"+ sum of the contributions {contrib.sum():+.2f}\n"
                 f"training molecules with their observed pK in brackets", fontsize=8.5)
    ax.grid(True, axis="x", lw=0.4, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, path)


def fig_rule3(path):
    """Chapter 8, rule 3: a prediction is a sum over the atoms of the molecule, drawn as the
    paper draws it, for the highest and the lowest predicted test molecule."""
    import matplotlib.pyplot as plt
    s, c, r = steroids(), cofield_model(), rule_molecules()
    mf, model, frame, ds = c["mf"], c["model"], c["frame"], s["ds"]
    picks = [r["high"], r["low"]]
    values = [model.atom_contributions(mf[i]) for i in picks]
    bias = float(model.bias)
    vmax = max(np.abs(v).max() for v in values)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.45))
    for ax, i, v in zip(axes, picks, values):
        mol = mf[i]
        xy = project(frame, mol.coords)
        draw_skeleton(ax, mol, frame, colour="0.55", hydrogens=True)
        ax.scatter(xy[:, 0], xy[:, 1], s=260 * np.abs(v) / vmax,
                   color=np.where(v >= 0, "C3", "C0"), alpha=0.75, linewidths=0, zorder=5)
        for k in np.argsort(-np.abs(v))[:3]:
            ax.annotate(f"{v[k]:+.2f}", xy[k], xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=6.5, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{mol.name}, {len(v)} atoms\nobserved {ds.activity[i]:.2f}, predicted "
                     f"{r['pred'][i]:.2f} = bias {bias:.2f} {v.sum():+.2f}", fontsize=8)
        print(f"  rule 3: {mol.name}, sum of the atom contributions {v.sum():+.3f}, "
              f"largest {v[np.argmax(np.abs(v))]:+.3f} on {mol.symbols[int(np.argmax(np.abs(v)))]}")
    fig.suptitle("rule 3: the prediction as a sum over the atoms; red raises the predicted pK, "
                 "blue lowers it, the area is the magnitude", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, path)


FIGURES = {
    "idea.png": fig_idea,
    "fields.png": fig_fields,
    "overlap.svg": fig_overlap,
    "search.svg": fig_search,
    "diagnose.svg": fig_diagnose,
    "featurespace.png": fig_featurespace,
    "parent.svg": fig_parent,
    "hop.png": fig_hop,
    "hopcofield.png": fig_hopcofield,
    "autocorr.svg": fig_autocorr,
    "periodic.svg": fig_periodic,
    "topomer.svg": fig_topomer,
    "confidence.svg": fig_confidence,
    "laplacian.svg": fig_laplacian,
    "probe.png": fig_probe,
    "chemspace.png": fig_chemspace,
    "gradient.png": fig_gradient,
    "ensemble.png": fig_ensemble,
    "vectorcofield.png": fig_vectorcofield,
    "copattern.svg": fig_copattern,
    "cospectrum.svg": fig_cospectrum,
    "phasefield.png": fig_phasefield,
    "multiscale.svg": fig_multiscale,
    "shells.png": fig_shells,
    "metabolism.svg": fig_metabolism,
    "pharmacophore.png": fig_pharmacophore,
    "pocket.png": fig_pocket,
    "scramble.svg": fig_scramble,
    "evidence.svg": fig_evidence,
    "scatter.svg": fig_scatter,
    "rule1.png": fig_rule1,
    "rule2.svg": fig_rule2,
    "rule3.svg": fig_rule3,
    "roc.svg": fig_roc,
    "acquire.svg": fig_acquire,
    "design.svg": fig_design,
    "water.png": fig_water,
    "mask.png": fig_mask,
    "matchedpair.png": fig_matchedpair,
    "ionprobe.png": fig_ionprobe,
    "fukui.png": fig_fukui,
    "complex.svg": fig_complex,
    "errorpred.svg": fig_errorpred,
    "cofield.png": fig_cofield,
    "isosurface.png": fig_isosurface,
    "overlapmap.png": fig_overlapmap,
    "oneclass.png": fig_oneclass,
    "kfda.png": fig_kfda,
    "selectivity.png": fig_selectivity,
    "envelope.svg": fig_envelope,
    "scale.svg": fig_scale,
    "sutherland.svg": fig_sutherland,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prefixes", nargs="*", help="draw only the figures whose name starts so")
    ap.add_argument("--check", action="store_true",
                    help="draw into a temporary folder and fail if a figure differs")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    names = [n for n in FIGURES if not args.prefixes or any(n.startswith(p) for p in args.prefixes)]
    if not names:
        print("no figure matches")
        return 1
    setup_matplotlib()
    out = tempfile.mkdtemp(prefix="cmf-figures-") if args.check else args.out
    os.makedirs(out, exist_ok=True)
    failed = []
    for name in names:
        path = os.path.join(out, name)
        FIGURES[name](path)
        if args.check:
            committed = os.path.join(args.out, name)
            same = os.path.exists(committed) and open(committed, "rb").read() == open(path, "rb").read()
            print(f"{'ok      ' if same else 'DIFFERS '}{name}")
            if not same:
                failed.append(name)
        else:
            print(f"written {relative_path(path, ROOT)}  ({os.path.getsize(path) // 1024} kB)")
    if failed:
        print(f"{len(failed)} figure(s) differ from docs/figures: run python docs/make_figures.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
