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
    "fields.png": fig_fields,
    "overlap.svg": fig_overlap,
    "search.svg": fig_search,
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
    "vectorcofield.png": fig_vectorcofield,
    "copattern.svg": fig_copattern,
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
