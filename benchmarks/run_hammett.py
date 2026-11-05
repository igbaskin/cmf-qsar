"""A field-based Hammett equation: CMF models of sigma_m, sigma_p and pi on substituted benzenes.

The data set (``examples/make_hammett_examples.py``) puts every X-C6H5 in one frame, ring centroid
at the origin and the ipso carbon on +x, so the overlap kernel needs no alignment step and a
substituent constant is modelled exactly as a binding affinity would be: KRR on the overlap of the
molecular fields, alpha and lambda by cross-validation.  Because the model is linear in the field,
the prediction is the integral of the field against the co-field, and the co-field of sigma_m and
that of sigma_p should differ in *where* they sit around the ring.  That is what this script
checks, in three steps:

1. **Which field carries the constant.**  Every target is fitted with all five physicochemical
   fields and with each field alone.  pi should be nearly exact from the hydrophobic field by
   itself (its weights are Crippen fragment increments, so this is close to a tautology);
   sigma_m should depend mainly on the electrostatic field; sigma_p needs the resonance part, which
   fragment charges cannot see, so it should be the hardest.
2. **The co-fields.**  Electrostatic and steric co-fields of sigma_m and sigma_p in the ring plane
   and above it, side by side, with the ring drawn in.
3. **Quantum charges, when given.**  ``--charges-prop q_hirshfeld`` replaces the electrostatic
   field by a field of the named per-atom property, which ``examples/orca_atom_properties.py``
   writes; charges computed on the whole substituted molecule carry the polarisation of the ring
   by the substituent, which is where the resonance information is.

Everything is 10-fold cross-validated on the training set (there is no held-out set: with 95
substituents the folds are the fair measurement) and the fitted r2 is printed beside q2 so that
overfitting shows.

Usage
-----
    python benchmarks/run_hammett.py                       # Gasteiger charges, symmetrised ensembles
    python benchmarks/run_hammett.py --single              # one conformer per molecule instead
    python benchmarks/run_hammett.py --sdf examples/hammett/hammett_qm_sym.sdf --charges-prop q_hirshfeld

Every file a run writes carries that run's tag, so two runs never write the same file:
``summary<tag>.csv``, ``run<tag>.json`` (the command that produced it),
``predictions_<target><tag>.csv``, ``cofields<tag>.png`` and ``cv_scatter<tag>.png`` in
``benchmarks/results/hammett/``.  The default run has the empty tag.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from cmfqsar.cofields import evaluate_gaussians  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, grid_search, r2_score, rmse  # noqa: E402

DEFAULT_SDF = os.path.join(ROOT, "examples", "hammett", "hammett_sym.sdf")
SINGLE_SDF = os.path.join(ROOT, "examples", "hammett", "hammett.sdf")
OUT_DIR = os.path.join(HERE, "results", "hammett")

PHYSCHEM = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
TARGETS = ["sigma_m", "sigma_p", "pi"]
ALPHAS = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
LAMS = [10.0 ** k for k in range(-4, 2)]

#: ring carbon positions in the frame, ipso first, going round; benzene C-C 1.39 A
RING_R = 1.39
RING_XY = np.array([[RING_R * np.cos(t), RING_R * np.sin(t)]
                    for t in np.deg2rad([0, 60, 120, 180, 240, 300])])
RING_LABELS = ["ipso", "ortho", "meta", "para", "meta", "ortho"]


def field_set(fields: Sequence[str], ensemble: bool, charges_prop: Optional[str] = None,
              weighting: str = "uniform", kt: float = 0.6) -> FieldSet:
    """The field set for one fit; a charges property replaces the electrostatic field.

    `weighting` is how the conformers of an ensemble are weighed: "uniform", or "boltzmann" from
    the ``conformer_energies`` property (kcal/mol) at temperature `kt`.
    """
    physchem = [f for f in fields if f != "electrostatic" or not charges_prop]
    props = [charges_prop] if (charges_prop and "electrostatic" in fields) else []
    return FieldSet.create(physchem=physchem, atom_properties=props,
                           conformers="ensemble" if ensemble else "single",
                           conformer_weighting=weighting, conformer_kt=kt)


def load(path: str, target: str, ensemble: bool):
    ds = load_dataset(path, activity_prop=target, ensemble_by="name" if ensemble else None)
    keep = [i for i in range(len(ds)) if np.isfinite(ds.activity[i])]
    return [ds.mols[i] for i in keep], [ds.names[i] for i in keep], np.asarray(ds.activity[keep], float)


def fit(mols, names, y, fs: FieldSet, alphas, lams, folds: int) -> Tuple[CMFModel, object]:
    mf = [build_fields(m, fs, name=n) for m, n in zip(mols, names)]
    train = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", fit_bias=True)
    res = grid_search(train, y, model, alphas, lams=lams, folds=folds, seed=0)
    model.fit(train, y, cv_folds=folds, cv_seed=0, reuse_kernels=True)
    return model, res


def field_label(fields: Sequence[str], charges_prop: Optional[str]) -> str:
    out = list(fields)
    if charges_prop and "electrostatic" in out:
        out[out.index("electrostatic")] = f"prop:{charges_prop}"
    return "+".join(out)


def cofield_slice(model: CMFModel, f: int, z: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """h_f c_f C_f(x, y, z) on a plane: the contribution per unit of field at every point."""
    centres, coeffs, widths = model.cofield_coefficients(f)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    v = evaluate_gaussians(pts, centres, coeffs, widths)
    v *= model.fieldset.specs[f].h * model.scale[f]
    return v.reshape(X.shape)


def draw_ring(ax):
    for k in range(6):
        a, b = RING_XY[k], RING_XY[(k + 1) % 6]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.2)
    ax.scatter(RING_XY[:, 0], RING_XY[:, 1], s=18, color="black", zorder=3)
    ax.annotate("X", (RING_R + 1.45, 0), ha="center", va="center", fontsize=9, fontweight="bold")
    for k, lab in enumerate(RING_LABELS):
        if lab in ("meta", "para") and RING_XY[k, 1] >= 0:
            d = RING_XY[k] / np.linalg.norm(RING_XY[k])
            p = RING_XY[k] + 0.55 * d
            ax.annotate(lab, p, ha="center", va="center", fontsize=8, color="dimgray")


def plot_cofields(models: Dict[str, CMFModel], fields: Sequence[str], path: str,
                  planes: Sequence[float] = (0.0, 1.7)) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(-4.5, 7.5, 0.15)
    ys = np.arange(-5.0, 5.0, 0.15)
    cols = [(f, z) for f in fields for z in planes]
    targets = list(models)
    fig, axes = plt.subplots(len(targets), len(cols), figsize=(3.6 * len(cols), 3.1 * len(targets)),
                             squeeze=False)
    for j, (fname, z) in enumerate(cols):
        slices = {}
        for t in targets:
            m = models[t]
            f = m.fieldset.names.index(fname)
            slices[t] = cofield_slice(m, f, z, xs, ys)
        vmax = max(np.abs(s).max() for s in slices.values()) or 1.0
        for i, t in enumerate(targets):
            ax = axes[i, j]
            im = ax.imshow(slices[t], origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            draw_ring(ax)
            ax.set_title(f"{t}: {m.fieldset.specs[f].label}, z = {z:g} A", fontsize=9)
            ax.set_xlabel("x / A", fontsize=8)
            if j == 0:
                ax.set_ylabel("y / A", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=axes[:, j].tolist(), shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle("Co-fields of the substituent constants (red: raises the constant per unit of field)",
                 fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(results: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.0), squeeze=False)
    for ax, (t, (y, ycv, names)) in zip(axes[0], results.items()):
        ax.scatter(y, ycv, s=16, color="tab:blue")
        lo, hi = min(y.min(), ycv.min()), max(y.max(), ycv.max())
        ax.plot([lo, hi], [lo, hi], color="gray", lw=0.8)
        err = np.abs(y - ycv)
        for k in np.argsort(-err)[:6]:
            ax.annotate(names[k], (y[k], ycv[k]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{t}: q2 = {r2_score(y, ycv):.2f}, RMSEcv = {rmse(y, ycv):.3f}", fontsize=9)
        ax.set_xlabel("tabulated", fontsize=8)
        ax.set_ylabel("cross-validated prediction", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_tag(a) -> str:
    """The name of one run of this script, used for every file that run writes.

    Two rules. Two runs whose numbers differ must never write the same file, so every option that
    changes a number enters the tag. And two ways of naming one input must give one tag, so
    ``--single`` and ``--sdf examples/hammett/hammett.sdf`` are both ``_single``: they are the same
    data set. The default data set has the empty tag, which is why `summary.csv` has no suffix.
    """
    tag = ""
    path = a.sdf or (SINGLE_SDF if a.single else DEFAULT_SDF)
    ref = os.path.abspath(path)
    if ref == os.path.abspath(SINGLE_SDF):
        tag += "_single"
    elif ref != os.path.abspath(DEFAULT_SDF):
        stem = os.path.splitext(os.path.basename(path))[0]
        tag += "_" + (stem[len("hammett_"):] if stem.startswith("hammett_") else stem)
    if a.charges_prop:
        tag += f"_{a.charges_prop}"
    if a.weighting != "uniform":
        tag += f"_{a.weighting}" + (f"_kt{a.kt:g}" if a.kt != 0.6 else "")
    # The protocol options. A run over one target, or over another grid, or with the single-field
    # fits left out, holds fewer or different rows; without these suffixes it would write over the
    # summary of the full run, and the table built from that file would then show less than it
    # claims to show. The exact grid is in the provenance file, so `_grid` need not carry it.
    if [t.strip() for t in a.targets.split(",") if t.strip()] != list(TARGETS):
        tag += "_" + "-".join(t.strip() for t in a.targets.split(",") if t.strip())
    if a.folds != 10:
        tag += f"_folds{a.folds}"
    if ([float(x) for x in a.alphas.split(",") if x.strip()] != list(ALPHAS)
            or [float(x) for x in a.lams.split(",") if x.strip()] != list(LAMS)):
        tag += "_grid"
    if a.no_ablation:
        tag += "_noablation"
    return tag


def run_provenance(out_dir: str, tag: str) -> Optional[dict]:
    """What the run with this tag was asked to do, or ``None`` if the run did not record it."""
    path = os.path.join(out_dir, f"run{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def report(out_dir: str = OUT_DIR, path: Optional[str] = None) -> str:
    """One table per summary file: every target and field combination with its q² and RMSEcv.

    The summary files are ``summary<tag>.csv``, one per run of this script, and ``run<tag>.json``
    holds the command that produced each one. The table prints that command above each table, so
    that a reader can repeat any block and so that a stored result with no command is visible as
    such.
    """
    import csv
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("summary") and f.endswith(".csv"))
    lines = ["# A field-based Hammett equation", "",
             "One table per run of `benchmarks/run_hammett.py`, with the command that produced it. "
             "The tag of the summary file names the run: no tag is `hammett_sym.sdf`, one conformer "
             "per substituent with its four symmetry images and Gasteiger charges; `_single` the "
             "same conformers without the images; `_ens` the conformer ensemble; `_q_hirshfeld` the "
             "Hirshfeld charges of the quantum stage; `_boltzmann` Boltzmann instead of uniform "
             "weights. q² and RMSEcv by 10-fold cross-validation, α and λ chosen on it.", ""]
    for f in files:
        tag = f[len("summary"):-4]
        prov = run_provenance(out_dir, tag)
        lines += [f"## `{tag}`" if tag else "## The default data set", ""]
        if prov:
            cmd = " ".join(["python benchmarks/run_hammett.py"] + list(prov["argv"]))
            if "dataset" in prov:
                weights = ("" if not prov.get("ensemble") else
                           f", {prov['weighting']} weights over the records of each substituent")
                lines += [f"Data set `{prov['dataset']}`, charges {prov['charges']}{weights}, "
                          f"{prov['folds']}-fold cross-validation.", ""]
            lines += [f"`{cmd}`", ""]
        else:
            lines += ["The command that produced this file was not recorded.", ""]
        lines += ["| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in csv.DictReader(open(os.path.join(out_dir, f), encoding="utf-8")):
            lines.append(f"| {r['target']} | {r['fields']} | {r['n']} | {float(r['alpha']):g} | "
                         f"{float(r['lambda']):g} | {float(r['q2']):.2f} | "
                         f"{float(r['rmse_cv']):.2f} | {float(r['r2_fit']):.2f} | "
                         f"{float(r['sd_y']):.2f} |")
        lines.append("")
    text = "\n".join(lines)
    path = path or os.path.join(out_dir, "table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sdf", default=None, help=f"data set (default {os.path.relpath(DEFAULT_SDF, ROOT)})")
    ap.add_argument("--single", action="store_true",
                    help="one conformer per molecule, no symmetry images (examples/hammett/hammett.sdf)")
    ap.add_argument("--charges-prop", default=None, metavar="NAME",
                    help="per-atom property to use in place of the Gasteiger electrostatic field")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    ap.add_argument("--lams", default=",".join(f"{l:g}" for l in LAMS))
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--no-ablation", action="store_true", help="skip the single-field fits")
    ap.add_argument("--weighting", default="uniform", choices=["uniform", "boltzmann"],
                    help="how the conformers of an ensemble are weighed (default uniform)")
    ap.add_argument("--kt", type=float, default=0.6, help="kT of the Boltzmann weights, kcal/mol")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--report", action="store_true",
                    help="only rebuild results/hammett/table.md from the summary files")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.report:
        report(a.out_dir)
        return 0

    ensemble = not a.single
    path = a.sdf or (SINGLE_SDF if a.single else DEFAULT_SDF)
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]
    lams = [float(x) for x in a.lams.split(",") if x.strip()]
    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    os.makedirs(a.out_dir, exist_ok=True)
    tag = run_tag(a)

    combos: List[List[str]] = [PHYSCHEM]
    if not a.no_ablation:
        combos += [[f] for f in PHYSCHEM] + [["electrostatic", "steric"]]

    rows = []
    full_models: Dict[str, CMFModel] = {}
    scatter: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
    t0 = time.time()
    print(f"data set {os.path.relpath(path, ROOT)}, "
          f"{'symmetrised ensembles' if ensemble else 'single conformers'}, "
          f"charges {'prop:' + a.charges_prop if a.charges_prop else 'Gasteiger'}, "
          f"{a.weighting + ' weights, ' if ensemble else ''}{a.folds}-fold CV")
    for target in targets:
        mols, names, y = load(path, target, ensemble)
        print(f"\n== {target}: {len(y)} substituents, range {y.min():.2f} .. {y.max():.2f}")
        if ensemble:
            from cmfqsar.conformers import ensemble_summary
            info = ensemble_summary(mols, field_set(PHYSCHEM, True, a.charges_prop, a.weighting, a.kt))
            print(f"  {info['conformations_mean']:.1f} records per substituent (max "
                  f"{info['conformations_max']:.0f}), {info['effective_mean']:.1f} effective under "
                  f"the weights, kernel cost x{info['kernel_cost_factor']:.0f}")
        print(f"  {'fields':38s} {'alpha':>6s} {'lambda':>8s} {'q2':>6s} {'RMSEcv':>7s} {'r2 fit':>7s}")
        for fields in combos:
            fs = field_set(fields, ensemble, a.charges_prop, a.weighting, a.kt)
            model, res = fit(mols, names, y, fs, alphas, lams, a.folds)
            lab = field_label(fields, a.charges_prop)
            st = model.stats
            print(f"  {lab:38s} {res.best_alpha:6.2f} {res.best_params['lam']:8.4f} "
                  f"{res.best_q2:6.3f} {res.best_rmse:7.3f} {st.r2:7.3f}")
            rows.append({"target": target, "fields": lab, "n": len(y), "alpha": res.best_alpha,
                         "lambda": res.best_params["lam"], "q2": round(res.best_q2, 4),
                         "rmse_cv": round(res.best_rmse, 4), "r2_fit": round(st.r2, 4),
                         "sd_y": round(float(np.std(y)), 4)})
            if fields is PHYSCHEM:
                full_models[target] = model
                ycv = res.y_cv if res.y_cv is not None else st.extra.get("y_cv")
                scatter[target] = (y, np.asarray(ycv, float), names)
                with open(os.path.join(a.out_dir, f"predictions_{target}{tag}.csv"), "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["substituent", target, "predicted_cv", "fitted"])
                    for n_, yy, yc, yf in zip(names, y, ycv, st.extra["y_fit"]):
                        w.writerow([n_, f"{yy:.3f}", f"{yc:.3f}", f"{yf:.3f}"])

    with open(os.path.join(a.out_dir, f"summary{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # The command next to its numbers: rule 6 asks that one command rebuild every table, and that
    # is only checkable if each stored file says which command wrote it.
    with open(os.path.join(a.out_dir, f"run{tag}.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"argv": list(argv) if argv is not None else sys.argv[1:],
                   "tag": tag,
                   "dataset": os.path.relpath(path, ROOT).replace(os.sep, "/"),
                   "ensemble": ensemble,
                   "charges": f"prop:{a.charges_prop}" if a.charges_prop else "gasteiger",
                   "weighting": a.weighting, "kt": a.kt,
                   "targets": targets, "folds": a.folds, "alphas": alphas, "lams": lams,
                   "ablation": not a.no_ablation, "rows": len(rows),
                   "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "seconds_fitting": round(time.time() - t0, 1)}, fh, indent=1)
        fh.write("\n")
    plot_scatter(scatter, os.path.join(a.out_dir, f"cv_scatter{tag}.png"))
    sig = {t: m for t, m in full_models.items() if t.startswith("sigma")}
    if sig:
        elec = f"prop:{a.charges_prop}" if a.charges_prop else "electrostatic"
        plot_cofields(sig, [elec, "steric"], os.path.join(a.out_dir, f"cofields{tag}.png"))
    print(f"\nwritten to {os.path.relpath(a.out_dir, ROOT)} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
