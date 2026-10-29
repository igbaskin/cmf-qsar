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
                label="the 2013 paper, Table 2")
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
    "scatter.svg": fig_scatter,
    "rule1.png": fig_rule1,
    "rule2.svg": fig_rule2,
    "rule3.svg": fig_rule3,
    "roc.svg": fig_roc,
    "errorpred.svg": fig_errorpred,
    "cofield.png": fig_cofield,
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
            print(f"written {os.path.relpath(path, ROOT)}  ({os.path.getsize(path) // 1024} kB)")
    if failed:
        print(f"{len(failed)} figure(s) differ from docs/figures: run python docs/make_figures.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
