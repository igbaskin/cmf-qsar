"""One HTML file that says what a model is, how well it works and what it says (plan item F4).

This is written for the person who will never start the application: a colleague, a reviewer, a
supervisor who wants to see the model that produced a number.  It is one self-contained file --
every picture is a base64 PNG inside it, so it survives being emailed, attached to a report or put
on a shared drive -- and it draws only on what the model file already carries, so a report costs no
refitting and cannot disagree with the model it describes.

Seven sections, in the order a sceptical reader asks for them:

1. **What this model is** -- fields, attenuation, learner, kernel, how the charges were made.
2. **How well it fits and predicts** -- the statistics with the observed-against-predicted plots
   beside them, which is the only fair way to show a q².
3. **Whether to believe that** -- the validation that was actually run: y-randomisation, the
   bootstrap interval, the applicability domain, the conformal level, the error predictors.  A
   check that was not run is reported as not run, never silently omitted.
4. **What the model rests on** -- the mean absolute contribution of each field over the training
   molecules, which is what distinguishes a model that rests on its geometry from one that rests
   on one field.
5. **The co-fields**, projected onto the three coordinate planes from fixed viewpoints: where the
   model needs charge, bulk, a donor.  A projection rather than an isosurface, because a report has
   no camera and a reader has no mouse.
6. **The extremes of the series** -- the most and least active training molecules, each with its
   exact field-level Shapley decomposition and its largest atom contributions.
7. **The pharmacophore** read out of the co-fields, and the provenance of the model file.

Everything degrades rather than fails.  A nonlinear kernel has no co-fields, no contributions and
no pharmacophore; the report says so in place of each and keeps the rest.  A model with no test set
reports no R²pred.  A classifier reports its own statistics.
"""
from __future__ import annotations

import base64
import html
import io
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

CSS = """
:root { --ink:#1b1b1b; --soft:#666; --rule:#dcdcdc; --accent:#2c5f8a; --bg:#ffffff;
        --panel:#f7f8fa; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e8e8e8; --soft:#a8a8a8; --rule:#3a3a3a; --accent:#7fb3dd; --bg:#1a1a1a;
          --panel:#242424; }
}
* { box-sizing: border-box; }
body { margin: 0 auto; padding: 32px 16px 64px; max-width: 54rem; background: var(--bg);
       color: var(--ink); font: 16px/1.6 Georgia, "Times New Roman", serif; }
h1 { font-size: 1.9rem; margin: 0 0 .2em; line-height: 1.2; }
h2 { font-size: 1.25rem; margin: 2.2em 0 .6em; padding-bottom: .25em;
     border-bottom: 1px solid var(--rule); font-weight: normal; color: var(--accent); }
h3 { font-size: 1.02rem; margin: 1.6em 0 .4em; font-weight: bold; }
p, li { margin: .5em 0; }
.lede { color: var(--soft); margin: 0 0 1.5em; }
table { border-collapse: collapse; width: 100%; margin: .8em 0; font-size: .92rem;
        font-family: "Helvetica Neue", Arial, sans-serif; }
th, td { text-align: left; padding: .35em .6em; border-bottom: 1px solid var(--rule);
         vertical-align: top; }
th { font-weight: 600; color: var(--soft); font-size: .84rem; text-transform: uppercase;
     letter-spacing: .03em; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
figure { margin: 1.2em 0; }
figure img { width: 100%; height: auto; display: block; background: var(--panel);
             border: 1px solid var(--rule); border-radius: 3px; }
figcaption { color: var(--soft); font-size: .86rem; margin-top: .4em; }
.note { background: var(--panel); border-left: 3px solid var(--rule); padding: .6em .9em;
        margin: 1em 0; color: var(--soft); font-size: .92rem; }
.missing { color: var(--soft); font-style: italic; }
code, pre { font-family: "SF Mono", Consolas, monospace; font-size: .86rem; }
pre { background: var(--panel); padding: .8em 1em; overflow-x: auto; border-radius: 3px; }
footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid var(--rule);
         color: var(--soft); font-size: .84rem; }
@media print { body { max-width: none; } h2 { page-break-after: avoid; }
               figure { page-break-inside: avoid; } }
"""


@dataclass
class ReportSpec:
    """What the report draws, and how finely."""

    spacing: float = 0.6          # grid spacing for the co-field projections, angstrom
    margin: float = 3.0           # margin around the training atoms, angstrom
    level: float = 0.5            # fraction of the peak that bounds a pharmacophore lobe
    max_features: int = 12
    top_atoms: int = 8            # atom contributions listed for each extreme molecule
    cofield_z: float = 2.0        # draw only what lies this far from zero; 0 draws everything
    cofields: bool = True
    pharmacophore: bool = True
    dpi: int = 110


# --------------------------------------------------------------------------- #
# Small HTML helpers
# --------------------------------------------------------------------------- #

def _e(x) -> str:
    return html.escape(str(x), quote=False)


def _num(v, fmt="{:.3f}") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _e(v)
    return '<span class="missing">&mdash;</span>' if not np.isfinite(f) else fmt.format(f)


def _rows(pairs: Sequence[Sequence], numeric: Sequence[int] = ()) -> str:
    out = []
    for row in pairs:
        cells = []
        for i, c in enumerate(row):
            cls = ' class="n"' if i in numeric else ""
            cells.append(f"<td{cls}>{c}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(out)


def _table(header: Sequence[str], rows: Sequence[Sequence], numeric: Sequence[int] = ()) -> str:
    numeric_class = ' class="n"'
    head = "".join(f"<th{numeric_class if i in numeric else ''}>{_e(h)}</th>"
                   for i, h in enumerate(header))
    return (f"<table><thead><tr>{head}</tr></thead><tbody>\n"
            f"{_rows(rows, numeric)}\n</tbody></table>")


def _missing(text: str) -> str:
    return f'<p class="missing">{_e(text)}</p>'


def _figure(png: bytes, caption: str) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    return (f'<figure><img alt="{_e(caption)}" src="data:image/png;base64,{b64}">'
            f"<figcaption>{caption}</figcaption></figure>")


def _plt():
    """matplotlib with no display, imported late so that importing this module stays fast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _png(fig, dpi: int) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    _plt().close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def scatter_png(panels: Sequence[Dict], spec: ReportSpec) -> bytes:
    """Observed against predicted, one panel per set of predictions, on a shared square scale."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.9), squeeze=False)
    for ax, p in zip(axes[0], panels):
        obs, pred = np.asarray(p["obs"], float), np.asarray(p["pred"], float)
        ok = np.isfinite(obs) & np.isfinite(pred)
        obs, pred = obs[ok], pred[ok]
        lo = float(min(obs.min(), pred.min())) if len(obs) else 0.0
        hi = float(max(obs.max(), pred.max())) if len(obs) else 1.0
        pad = 0.06 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#999", lw=1, zorder=1)
        ax.scatter(obs, pred, s=26, facecolor=p.get("colour", "#2c5f8a"), edgecolor="white",
                   linewidth=.6, zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.set_title(p["title"], fontsize=10)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return _png(fig, spec.dpi)


_PLANES = ((0, 1, 2, "x", "y", "along z"), (0, 2, 1, "x", "z", "along y"),
           (1, 2, 0, "y", "z", "along x"))


def cofield_png(model, f: int, grid, volume: np.ndarray, spec: ReportSpec) -> bytes:
    """One field's co-field from the three fixed viewpoints, with the training atoms over it.

    Each panel is the signed extremum along the viewing axis -- the strongest thing the reader
    would see looking down that axis, with its sign kept -- which is the projection that does not
    cancel a positive lobe against a negative one behind it.
    """
    plt = _plt()
    coords = np.asarray(model.train.coords, float)
    vmax = float(np.max(np.abs(volume))) or 1.0
    axes_lo, axes_hi = grid.origin, grid.origin + (np.asarray(volume.shape) - 1) * grid.spacing
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 3.6))
    for ax, (a, b, c, la, lb, how) in zip(axs, _PLANES):
        hi = volume.max(axis=c)
        lo = volume.min(axis=c)
        proj = np.where(np.abs(hi) >= np.abs(lo), hi, lo)
        # proj is indexed (a, b); imshow expects (row = b, column = a), hence the transpose
        extent = [axes_lo[a], axes_hi[a], axes_lo[b], axes_hi[b]]
        ax.imshow(proj.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  aspect="equal", interpolation="bilinear")
        if len(coords):
            ax.scatter(coords[:, a], coords[:, b], s=1.2, color="#303030", alpha=.25,
                       linewidths=0)
        ax.set_xlabel(f"{la} / Å", fontsize=9)
        ax.set_ylabel(f"{lb} / Å", fontsize=9)
        ax.set_title(f"{la}–{lb}, viewed {how}", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"co-field: {model.fieldset.specs[f].label}", fontsize=11, y=1.02)
    return _png(fig, spec.dpi)


def contributions_png(names: Sequence[str], values: Sequence[float], spec: ReportSpec) -> bytes:
    """The mean absolute contribution of each field, as a bar per field."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(names) + 1.1))
    y = np.arange(len(names))
    ax.barh(y, values, color="#2c5f8a", height=.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean |contribution| per training molecule", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _png(fig, spec.dpi)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def _section_what(model, extra: Dict) -> str:
    fs = model.fieldset
    specs = [s for s in fs.specs if s.enabled]
    alphas = sorted({round(float(s.alpha), 6) for s in specs})
    alpha_txt = (f"{alphas[0]:g}" if len(alphas) == 1 else
                 ", ".join(f"{s.name} {s.alpha:g}" for s in specs))
    # the parameter dictionary carries a default for every learner; name only the ones this
    # learner uses, or a kernel ridge model appears to have been given an epsilon
    used = {"krr": ("lam",), "gp": ("lam",), "kpls": ("components",),
            "svr": ("C", "epsilon"), "nusvr": ("C", "nu"),
            "ranksvm": ("C",), "svc": ("C",), "nusvc": ("nu",), "gpc": ("amplitude",),
            "kfda": ("lam",), "klr": ("lam",), "lapkrr": ("lam", "manifold", "neighbours"),
            "ocsvm": ("nu",)}.get(model.method, ("lam",))
    if model.kernel_layer == "gauss":
        used = used + ("gamma",)
    pars = ", ".join(f"{k} = {float(model.params[k]):g}" for k in used if k in model.params)
    source = {"fields": "overlap integrals of the fields (an aligned series)",
              "autocorrelation": "the rotation-invariant autocorrelation (no alignment)",
              "surface": "the in-plane invariant kernel (molecules on a surface)",
              "periodic": "the periodic autocorrelation (crystals)"}.get(
        model.kernel_source or "fields", model.kernel_source)
    rows = [
        ("Task", _e(model.task)),
        ("Learner", f"{_e(model.method)}" + (f" ({_e(pars)})" if pars else "")),
        ("Kernel", _e(source) + ("" if model.is_linear
                                 else f", with a {_e(model.kernel_layer)} layer on top")),
        ("Fields", _e(", ".join(s.label for s in specs)) or '<span class="missing">none</span>'),
        ("Attenuation α", _e(alpha_txt)),
        ("Charges", _e(fs.charge_model)),
        ("Conventions", ("cosine-normalised field kernels" if model.normalize
                         else "raw field kernels")
         + (", each scaled to unit mean diagonal" if model.scale_fields else "")
         + (", with a fitted bias" if model.fit_bias else ", no bias")),
        ("Training molecules", f"{model.stats.n_train or (0 if model.train is None else model.train.n_mol)}"),
    ]
    if extra.get("dataset"):
        rows.append(("Data set", _e(os.path.basename(str(extra["dataset"])))))
    if extra.get("activity"):
        rows.append(("Activity", _e(extra["activity"])))
    mix = {s.name: float(s.h) for s in specs if float(s.h) != 1.0}
    if mix:
        rows.append(("Mixing coefficients", _e(", ".join(f"{k} {v:g}" for k, v in mix.items()))))
    return "<h2>What this model is</h2>" + _table(["", ""], rows)


def _section_statistics(model, spec: ReportSpec) -> str:
    st = model.stats
    out = ["<h2>How well it fits and predicts</h2>"]
    if model.task == "regression" and not st.rank_cv:
        rows = [("Fit, training set", _num(st.r2), _num(st.rmse_fit), f"n = {st.n_train}"),
                (f"Cross-validation ({st.cv_folds or '?'}-fold)", _num(st.q2), _num(st.rmse_cv),
                 "the number to read first"),
                ("Test set", _num(st.r2_pred), _num(st.rmse_pred),
                 f"m = {st.n_test}" if st.n_test else
                 '<span class="missing">no test set</span>'),
                ("External (nested) CV", _num(st.q2_ex), _num(st.rmse_ex),
                 "selection included" if np.isfinite(st.q2_ex) else
                 '<span class="missing">not run</span>')]
        out.append(_table(["", "r² / q²", "RMSE", ""], rows, numeric=(1, 2)))
    elif st.rank_cv:
        rows = [("Cross-validation", _num(st.rank_cv.get("spearman")),
                 _num(st.rank_cv.get("kendall")), _num(st.rank_cv.get("pairs"))),
                ("Test set", _num(st.rank_test.get("spearman")),
                 _num(st.rank_test.get("kendall")), _num(st.rank_test.get("pairs")))]
        out.append("<p>A ranking model fits the order and not the values, so it has no "
                   "r².</p>")
        out.append(_table(["", "Spearman ρ", "Kendall τ", "pairs right"], rows,
                          numeric=(1, 2, 3)))
    else:
        rows = [(k, _num(v)) for k, v in sorted(st.extra.get("classification", {}).items())]
        out.append(_table(["statistic", "value"], rows, numeric=(1,)) if rows else
                   _missing("no classification statistics were stored with this model"))

    panels = []
    y = None if model.y is None else np.asarray(model.y, float)
    if y is not None and "y_cv" in st.extra:
        panels.append({"obs": y, "pred": np.asarray(st.extra["y_cv"], float),
                       "title": f"cross-validation, q² = {st.q2:.3f}", "colour": "#2c5f8a"})
    if "y_test_pred" in st.extra and "y_test" in st.extra:
        panels.append({"obs": np.asarray(st.extra["y_test"], float),
                       "pred": np.asarray(st.extra["y_test_pred"], float),
                       "title": f"test set, R²pred = {st.r2_pred:.3f}", "colour": "#a0522d"})
    if panels:
        out.append(_figure(scatter_png(panels, spec),
                           "Observed against predicted, with the line of identity. The spread "
                           "around it is the model's error; a q² is a summary of this "
                           "picture and not a substitute for it."))
    else:
        out.append(_missing("the model file carries no predictions to plot"))
    return "\n".join(out)


def _section_validation(model) -> str:
    x = model.stats.extra
    out = ["<h2>Whether to believe that</h2>"]
    sc = x.get("scramble")
    if sc:
        out.append(f"<h3>y-randomisation</h3><p>{sc['runs']} shuffles of the activities. The "
                   f"real model scores {_num(sc['observed'])} on "
                   f"{_e(sc.get('statistic', 'the statistic'))}; the shuffled ones average "
                   f"{_num(sc['mean'])} (sd {_num(sc['sd'])}), the best of them reaching "
                   f"{_num(sc.get('max'))}. That is {_num(sc.get('z'), '{:.1f}')} standard "
                   f"deviations out, p = {_num(sc.get('p'), '{:.4f}')}.</p>")
    else:
        out.append("<h3>y-randomisation</h3>" + _missing("not run for this model"))
    bs = x.get("bootstrap")
    if bs:
        rows = [(_e(k), _num(v.get("value")), _num(v.get("lo")), _num(v.get("hi")),
                 f"{v.get('runs', '')}") for k, v in bs.items()]
        out.append("<h3>Bootstrap intervals</h3>"
                   + _table(["statistic", "value", "lower", "upper", "runs"], rows,
                            numeric=(1, 2, 3, 4)))
    else:
        out.append("<h3>Bootstrap intervals</h3>" + _missing("not run for this model"))
    dom = x.get("domain")
    if dom:
        sim = np.asarray(dom.get("similarity", []), float)
        inside = int(np.sum(sim >= dom["threshold"])) if len(sim) else 0
        out.append(f"<h3>Applicability domain</h3><p>Similarity to the {dom['k']} nearest "
                   f"training molecules, in the model's own kernel, against a threshold of "
                   f"{_num(dom['threshold'])} taken from the training set's own spread: "
                   f"<b>{inside} of {len(sim)}</b> scored molecules fall inside.</p>")
        if len(sim):
            order = np.argsort(sim)[:8]
            rows = [(_e(dom["names"][i]), _num(sim[i]),
                     "inside" if sim[i] >= dom["threshold"] else "<b>outside</b>")
                    for i in order]
            out.append(_table(["the eight least similar molecules", "similarity", ""], rows,
                              numeric=(1,)))
    else:
        out.append("<h3>Applicability domain</h3>" + _missing("not computed for this model"))
    conf = getattr(model, "conformal_", None)
    if conf is not None:
        out.append(f"<h3>Conformal interval</h3><p>{_e(conf.describe())}</p>")
    ep = x.get("error_predictors")
    if ep:
        rows = [(_e(r["name"]), _num(r.get("spearman")), _num(r.get("efficiency")),
                 _num(r.get("rmse_trusted")), _num(r.get("rmse_doubted"))) for r in ep]
        out.append("<h3>What knows when a prediction is bad</h3>"
                   + _table(["estimate", "Spearman ρ", "efficiency", "RMSE trusted",
                             "RMSE doubted"], rows, numeric=(1, 2, 3, 4))
                   + '<p class="note">Efficiency is 1 for the ordering the errors themselves '
                     'give and 0 for no better than ignoring the estimate.</p>')
    return "\n".join(out)


def _section_fields(model, spec: ReportSpec) -> str:
    out = ["<h2>What the model rests on</h2>"]
    if not (model.is_linear and model.uses_fields) or model.train is None:
        return "\n".join(out + [_missing(
            "the contribution of a field is a property of the linear overlap kernel; this model "
            "uses another and its decomposition is unavailable (the Shapley values below take "
            "its place)")])
    mols = list(model.train.molecules)
    specs = [s for s in model.fieldset.specs if s.enabled]
    idx = [i for i, s in enumerate(model.fieldset.specs) if s.enabled]
    totals = np.zeros(len(model.fieldset.specs))
    for mol in mols:
        per = model.atom_contributions(mol, per_field=True)
        totals += np.abs(per).sum(axis=1)
    totals /= max(len(mols), 1)
    out.append(_figure(contributions_png([s.label for s in specs], [totals[i] for i in idx], spec),
                       "How much of the prediction each field moves, averaged over the training "
                       "molecules. A model that rests on one field looks like one bar."))
    share = totals[idx] / (totals[idx].sum() or 1.0)
    rows = [(_e(s.label), _num(totals[i], "{:.4f}"), f"{100 * sh:.1f} %")
            for s, i, sh in zip(specs, idx, share)]
    out.append(_table(["field", "mean |contribution|", "share"], rows, numeric=(1, 2)))
    return "\n".join(out)


def _section_cofields(model, spec: ReportSpec) -> str:
    out = ["<h2>The co-fields</h2>"]
    if not (model.is_linear and model.uses_fields) or model.train is None:
        return "\n".join(out + [_missing(
            "co-fields are defined for the linear overlap kernel; this model uses another")])
    from .cofields import cofield_volume, cofield_with_error, make_grid

    grid = make_grid(model.train.coords, margin=spec.margin, spacing=spec.spacing)
    out.append("<p>Where the model needs something and where it needs its absence: the co-field is "
               "the model's weights placed back into space, so a red lobe is a region where more of "
               "that field raises the predicted activity and a blue one where it lowers it. Each "
               "field is shown from three fixed viewpoints, as the strongest signed value along "
               "the viewing axis, with the training atoms as grey dots for scale.</p>")
    masked, probe = spec.cofield_z > 0, None
    if masked:
        try:
            probe = cofield_with_error(model, 0, grid, include_scale=True, z=spec.cofield_z)
            out.append(f'<p class="note">Only what lies more than {spec.cofield_z:g} standard '
                       f'deviations from zero is drawn. The error bar propagates a noise level '
                       f'of {probe.sigma:.3f} in the activities ({_e(probe.sigma_from)}) through '
                       f'the fit, with the molecules and the kernel held fixed; the rest of each '
                       f'lobe is shape that the measurement error alone could have produced, and '
                       f'is cut away. How much survived is given under each picture.</p>')
        except (ValueError, RuntimeError) as exc:
            masked = False
            out.append(_missing(f"drawn whole: this model has no error bar to cut them by ({exc})"))
    for f, s in enumerate(model.fieldset.specs):
        if not s.enabled:
            continue
        kept = ""
        if masked:
            ce = cofield_with_error(model, f, grid, sigma=probe.sigma, include_scale=True,
                                    z=spec.cofield_z)
            vol = ce.masked
            kept = f" {100 * ce.surviving:.0f} % of what would have been drawn survived."
        else:
            vol = cofield_volume(model, f, grid, include_scale=True)
        if not np.any(np.isfinite(vol)) or float(np.max(np.abs(vol))) <= 0:
            out.append(_missing(f"nothing in the {s.label} co-field stands above its error bar"))
            continue
        out.append(_figure(cofield_png(model, f, grid, vol, spec),
                           f"Co-field of the {_e(s.label)} field. Red: more of this field raises "
                           f"the prediction; blue: it lowers it.{kept}"))
    return "\n".join(out)


def _section_extremes(model, spec: ReportSpec) -> str:
    out = ["<h2>The extremes of the series</h2>"]
    if model.train is None or model.y is None or len(np.asarray(model.y, float)) < 2:
        return "\n".join(out + [_missing("the model file carries no training activities")])
    from .shapley import field_shapley

    y = np.asarray(model.y, float)
    mols = list(model.train.molecules)
    names = list(getattr(model.train, "names", [])) or [f"molecule {i + 1}" for i in range(len(y))]
    out.append("<p>The most and the least active training molecules, each decomposed twice: over "
               "the fields, by exact Shapley values, which is a fair split of the prediction "
               "between them; and over its own atoms, which says where in the molecule the "
               "prediction came from.</p>")
    for label, i in (("Most active", int(np.argmax(y))), ("Least active", int(np.argmin(y)))):
        mol = mols[i]
        head = (f"<h3>{label}: {_e(names[i])} &mdash; observed {y[i]:.3f}")
        try:
            res = field_shapley(model, mol)
            head += f", predicted {res.prediction:.3f}</h3>"
            rows = [(_e(lab), _num(v)) for lab, v, _se in res.table()]
            rows.append(("<i>baseline (no fields)</i>", _num(res.baseline)))
            body = _table(["field", "Shapley value"], rows, numeric=(1,))
        except (ValueError, RuntimeError) as exc:
            head += "</h3>"
            body = _missing(f"no field decomposition: {exc}")
        out.append(head + body)
        if model.is_linear and (model.uses_fields
                                or model.kernel_source == "autocorrelation"):
            try:
                c = np.asarray(model.atom_contributions(mol), float)
                order = np.argsort(-np.abs(c))[:spec.top_atoms]
                rows = [(f"{_e(mol.symbols[j])}{j + 1}", _num(c[j], "{:+.4f}")) for j in order]
                out.append(_table([f"largest atom contributions", "contribution"], rows,
                                  numeric=(1,)))
            except (ValueError, RuntimeError):
                pass
    return "\n".join(out)


def _section_pharmacophore(model, spec: ReportSpec) -> str:
    out = ["<h2>The pharmacophore</h2>"]
    if not (model.is_linear and model.uses_fields) or model.train is None:
        return "\n".join(out + [_missing(
            "a pharmacophore is read out of the co-fields, which this kernel does not have")])
    from .pharmacophore import extract

    try:
        feats = extract(model, level=spec.level, max_features=spec.max_features,
                        spacing=spec.spacing, margin=spec.margin, min_z=spec.cofield_z)
    except (ValueError, RuntimeError) as exc:
        return "\n".join(out + [_missing(f"not available: {exc}")])
    if not feats:
        return "\n".join(out + [_missing("no lobe of any co-field was large enough to become a "
                                         "feature at this level")])
    rows = [(_e(ft.kind), _e(ft.field), f"{ft.centre[0]:.2f}, {ft.centre[1]:.2f}, "
             f"{ft.centre[2]:.2f}", _num(ft.radius, "{:.2f}"), _num(ft.importance, "{:+.3f}"),
             _e(ft.meaning)) for ft in feats]
    cut = (f" Read from the co-fields after they were cut at {spec.cofield_z:g} standard "
           "deviations, so no feature here is one the noise in the activities could have "
           "produced." if spec.cofield_z > 0 else "")
    out.append("<p>The lobes of the co-fields, read as typed features with tolerance radii: what "
               "a molecule should put where to be predicted active, and what it should keep "
               "out." + cut + "</p>")
    out.append(_table(["feature", "field", "centre / Å", "radius / Å", "weight",
                       "meaning"], rows, numeric=(3, 4)))
    return "\n".join(out)


def _section_provenance(model, extra: Dict) -> str:
    p = dict(getattr(model, "provenance", None) or {})
    if not p:
        try:
            p = dict(model.describe_training(model))
        except Exception:                      # a model file that records nothing
            p = {}
    rows = []
    order = ["cmfqsar", "created", "python", "numpy", "rdkit", "n_train", "charge_model",
             "charge_source", "param_scheme", "conformers", "explicit_hydrogens"]
    for k in order:
        if k in p and p[k] is not None:
            v = p[k]
            rows.append((_e(k.replace("_", " ")),
                         _num(v, "{:.2f}") if isinstance(v, float) else _e(v)))
    box = (p.get("box_min"), p.get("box_max"))
    if all(b is not None for b in box):
        rows.append(("training box / Å",
                     _e(" to ".join("(" + ", ".join(f"{v:.1f}" for v in b) + ")" for b in box))))
    if extra.get("train_names"):
        rows.append(("training molecules named",
                     _e(f"{len(extra['train_names'])} recorded in the model file")))
    return ("<h2>Provenance</h2>"
            + (_table(["", ""], rows) if rows else _missing("the model file records none"))
            + '<p class="note">Every number above was read from the model file itself. Nothing '
              'in this report was refitted, so it cannot disagree with the model it describes.</p>')


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

def build_html(model, extra: Optional[Dict] = None, title: Optional[str] = None,
               spec: Optional[ReportSpec] = None, source: str = "",
               progress=None) -> str:
    """The whole report as one self-contained HTML document."""
    spec = spec or ReportSpec()
    extra = dict(extra or {})
    name = title or (os.path.basename(str(extra.get("dataset") or source)) or "CMF model")
    sections: List[str] = []
    steps = [("what the model is", lambda: _section_what(model, extra)),
             ("statistics", lambda: _section_statistics(model, spec)),
             ("validation", lambda: _section_validation(model)),
             ("field contributions", lambda: _section_fields(model, spec))]
    if spec.cofields:
        steps.append(("co-fields", lambda: _section_cofields(model, spec)))
    steps.append(("the extremes of the series", lambda: _section_extremes(model, spec)))
    if spec.pharmacophore:
        steps.append(("pharmacophore", lambda: _section_pharmacophore(model, spec)))
    steps.append(("provenance", lambda: _section_provenance(model, extra)))
    for i, (what, fn) in enumerate(steps):
        if progress:
            progress(f"Report: {what}", i / len(steps))
        sections.append(fn())
    st = model.stats
    lede = (f"{st.n_train} training molecules, "
            f"{len([s for s in model.fieldset.specs if s.enabled])} fields, "
            f"{_e(model.method)}")
    if np.isfinite(st.q2):
        lede += f", q² = {st.q2:.3f}"
    if np.isfinite(st.r2_pred):
        lede += f", R²pred = {st.r2_pred:.3f}"
    made = time.strftime("%d %B %Y, %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CMF model report &mdash; {_e(name)}</title>
<style>{CSS}</style></head>
<body>
<h1>CMF model report</h1>
<p class="lede">{_e(name)} &mdash; {lede}</p>
{chr(10).join(sections)}
<footer>Written by cmfqsar from {_e(os.path.basename(source) or "a model in memory")} on {made}.
A continuous molecular field model: the molecules are represented by atom-centred Gaussian fields
and compared by the overlap integrals of those fields, so every number here is a property of the
fields and of the alignment they were built in.</footer>
</body></html>
"""


def write_report(path: str, model, extra: Optional[Dict] = None, title: Optional[str] = None,
                 spec: Optional[ReportSpec] = None, source: str = "", progress=None) -> str:
    """Write the report to `path` and return the path."""
    text = build_html(model, extra=extra, title=title, spec=spec, source=source,
                      progress=progress)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path
