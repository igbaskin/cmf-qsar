#!/usr/bin/env python
"""
Can a low-rank kernel reproduce the exact CMF model?  The Nyström measurement of plan item B6.

The exact model holds one n x n matrix per field and factorises the n x n joint kernel.  That
stops at ten to twenty thousand molecules on a 32 GB machine.  The Nyström approximation replaces
every field kernel by

    K_f  ~  K_f(X, Z) K_f(Z, Z)^+ K_f(Z, X),

a matrix of rank m built from the kernels against m landmarks Z.  Two kinds of landmark are
measured here, both computed by the kernel code the program already has:

* **landmark molecules**: a random subset of the training molecules, so K(X, Z) is a block of
  the exact kernel and nothing new is evaluated;
* **Gaussians in space** ("pseudo-atoms"): a cubic grid of centres with the field's own width
  covering the molecules, each a one-atom molecule of unit weight.  The CMF kernel is the Gram
  matrix of a Gaussian kernel on R^3 contracted with the atom weights, so a basis of Gaussians
  in the common frame spans the fields of every molecule that fits in the frame, and the rank it
  needs depends on the frame and the width, not on the number of molecules.

For every Sutherland set the exact model is selected by the usual grid search (attenuation and
ridge by 10-fold cross-validation).  At that attenuation the Nyström kernels are built for a
range of ranks and the same cross-validation is run on them with the same folds and the same
ridge, so that the only thing that differs is the approximation.  Reported: the relative error
of the joint training kernel, the cross-validated q2 with the exact model's ridge and with a
ridge re-selected for the approximate kernel, and the time to build the kernels.

Two further stages ask what a user would see.  ``select`` runs the whole grid search on
Nyström kernels and reports whether it chooses the same attenuation and ridge as the exact one.
``width`` fixes the grid of centres and varies the attenuation, because a narrow Gaussian needs
more centres per angstrom than a broad one and this is where the approximation is expected to
fail.

Usage
-----
    python benchmarks/run_nystrom.py                 # every stage, then the table
    python benchmarks/run_nystrom.py --stage rank
    python benchmarks/run_nystrom.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import RDLogger  # noqa: E402

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, FieldSpec, build_fields  # noqa: E402
from cmfqsar.kernels import kernel_matrices, kernel_matrices_multi  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, grid_search, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "nystrom")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
LAMS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
FOLDS = 10
SEED = 0
#: spacing of the cubic grid of Gaussian centres, in angstroms
SPACINGS = (4.0, 3.0, 2.5, 2.0, 1.5, 1.25)
#: how many landmark molecules, and how many random draws of them
LANDMARKS = (10, 20, 40, 80, 160)
DRAWS = 3
#: van der Waals radius given to a Gaussian centre for the radius-scaled steric field
CENTRE_RADIUS = 1.5
#: eigenvalues of K(Z, Z) below this share of the largest are dropped from the pseudo-inverse
RCOND = 1e-10


# --------------------------------------------------------------------------------- the data


def table_of(name, alpha=0.5):
    ds = load_dataset(os.path.join(ROOT, "examples", "sutherland", f"{name}.sdf"))
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha)
    mols = list(ds.mols)
    y = np.asarray(ds.activity, float)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    return fs, AtomTable.from_molecules(mf), y


def specs_at(fs, alpha):
    return [FieldSpec(**{**vars(s), "alpha": alpha}) for s in fs.specs]


# ------------------------------------------------------------------------- the landmarks


def gaussian_centres(T: AtomTable, spacing: float, n_fields: int) -> AtomTable:
    """A cubic grid of one-atom molecules of unit weight covering the atoms of `T`.

    A centre further than one spacing from every atom carries nothing and is dropped.
    """
    lo = T.coords.min(axis=0) - 0.5 * spacing
    hi = T.coords.max(axis=0) + 0.5 * spacing
    axes = [np.arange(lo[k], hi[k] + 1e-9, spacing) for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    d, _ = cKDTree(T.coords).query(pts, k=1)
    pts = pts[d <= spacing]
    m = len(pts)
    return AtomTable(coords=pts, weights=np.ones((m, n_fields)),
                     radii=np.full(m, CENTRE_RADIUS), offsets=np.arange(m + 1),
                     names=[f"c{i}" for i in range(m)], molecules=[])


def nystrom(K_xz: np.ndarray, K_zz: np.ndarray) -> np.ndarray:
    """K(X, Z) K(Z, Z)^+ K(Z, X) for every field, with a thresholded pseudo-inverse."""
    F = K_xz.shape[0]
    out = np.empty((F, K_xz.shape[1], K_xz.shape[1]))
    for f in range(F):
        w, V = np.linalg.eigh(0.5 * (K_zz[f] + K_zz[f].T))
        keep = w > RCOND * max(float(w.max()), 1e-300)
        B = K_xz[f] @ V[:, keep]
        out[f] = (B / w[keep]) @ B.T
    return out


def approx_kernels(T: AtomTable, Z: AtomTable, specs) -> np.ndarray:
    """The Nyström field kernels of `T` from landmarks `Z`."""
    K_xz = kernel_matrices(T, Z, specs, symmetric=False)
    K_zz = kernel_matrices(Z, Z, specs, symmetric=True)
    return nystrom(K_xz, K_zz)


def approx_kernels_multi(T: AtomTable, Z: AtomTable, spec_sets):
    """The same for a grid of attenuations, sharing the distance blocks."""
    K_xz = kernel_matrices_multi(T, Z, spec_sets, symmetric=False)
    K_zz = kernel_matrices_multi(Z, Z, spec_sets, symmetric=True)
    return [nystrom(a, b) for a, b in zip(K_xz, K_zz)]


# ------------------------------------------------------------------------- the scoring


def joint_of(fs, T, K_fields):
    """The joint training kernel the learner sees, with the scale of its own diagonal."""
    m = CMFModel(fieldset=fs.copy(), method="krr")
    m.set_training_kernels(T, K_fields)
    return m.joint_kernel(K_fields, T.descriptors)


def q2_at(K, y, lam):
    ycv = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=FOLDS, seed=SEED)
    return float(r2_score(y, ycv)), float(rmse(y, ycv))


def best_lam(K, y):
    scores = [(q2_at(K, y, lam)[0], lam) for lam in LAMS]
    q2, lam = max(scores)
    return float(lam), float(q2)


def rel_error(K_exact, K_approx):
    return float(np.linalg.norm(K_exact - K_approx) / np.linalg.norm(K_exact))


# ---------------------------------------------------------------------------- the stages


def stage_rank(force=False):
    """Every set at its selected attenuation: the exact model against both kinds of landmark."""
    path = os.path.join(RESULTS, "rank.json")
    if os.path.exists(path) and not force:
        print("rank: already done")
        return json.load(open(path))
    out = {"alphas": list(ALPHAS), "lams": list(LAMS), "folds": FOLDS, "spacings": list(SPACINGS),
           "landmarks": list(LANDMARKS), "draws": DRAWS, "sets": []}
    for name in SETS:
        fs, T, y = table_of(name)
        n = T.n_mol
        model = CMFModel(fieldset=fs.copy(), method="krr")
        res = grid_search(T, y, model, alphas=ALPHAS, lams=LAMS, folds=FOLDS, seed=SEED)
        alpha, lam = float(res.best_alpha), float(res.best_params["lam"])
        specs = specs_at(fs, alpha)
        t0 = time.time()
        K_ex_fields = kernel_matrices(T, T, specs)
        t_exact = time.time() - t0
        K_ex = joint_of(fs, T, K_ex_fields)
        q2_ex, rmse_ex = q2_at(K_ex, y, lam)
        print(f"{name}: {n} molecules, alpha {alpha:g}, lambda {lam:g}, "
              f"q2cv {q2_ex:+.4f}, exact kernel {t_exact:.2f} s", flush=True)
        row = {"set": name, "n_mol": int(n), "atoms_per_molecule": round(T.coords.shape[0] / n, 1),
               "box": [round(float(v), 1) for v in (T.coords.max(0) - T.coords.min(0))],
               "alpha": alpha, "lam": lam, "q2_exact": q2_ex, "rmse_exact": rmse_ex,
               "seconds_exact": round(t_exact, 3), "space": [], "molecules": []}
        for s in SPACINGS:
            Z = gaussian_centres(T, s, len(fs.specs))
            t0 = time.time()
            K_ap_fields = approx_kernels(T, Z, specs)
            dt = time.time() - t0
            K_ap = joint_of(fs, T, K_ap_fields)
            q2, r = q2_at(K_ap, y, lam)
            lam_own, q2_own = best_lam(K_ap, y)
            err = rel_error(K_ex, K_ap)
            row["space"].append({"spacing": s, "m": int(Z.n_mol), "rel_error": err,
                                 "q2": q2, "rmse": r, "lam_own": lam_own, "q2_own": q2_own,
                                 "seconds": round(dt, 3)})
            print(f"   centres {s:5.2f} A  m {Z.n_mol:5d}  error {err:.1e}  q2 {q2:+.4f}  "
                  f"own ridge {lam_own:g} -> {q2_own:+.4f}  {dt:.2f} s", flush=True)
        for m in [k for k in LANDMARKS if k < n] + [n]:
            draws = []
            for d in range(DRAWS if m < n else 1):
                idx = np.sort(np.random.RandomState(SEED + d).choice(n, m, replace=False))
                K_xz = K_ex_fields[:, :, idx]
                K_zz = K_ex_fields[:, idx][:, :, idx]
                t0 = time.time()
                K_ap_fields = nystrom(K_xz, K_zz)
                dt = time.time() - t0
                K_ap = joint_of(fs, T, K_ap_fields)
                q2, r = q2_at(K_ap, y, lam)
                lam_own, q2_own = best_lam(K_ap, y)
                draws.append({"rel_error": rel_error(K_ex, K_ap), "q2": q2, "rmse": r,
                              "lam_own": lam_own, "q2_own": q2_own})
            # the cost of the kernels this would need, if they were not cut from the exact one
            row["molecules"].append({"m": int(m), "draws": draws,
                                     "seconds_kernels_estimate": round(t_exact * m / n, 3)})
            q2s = [d["q2"] for d in draws]
            print(f"   molecules m {m:5d}  error {np.mean([d['rel_error'] for d in draws]):.1e}  "
                  f"q2 {np.mean(q2s):+.4f} [{min(q2s):+.4f}, {max(q2s):+.4f}]", flush=True)
        out["sets"].append(row)
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(out, open(path, "w"), indent=1)
    return out


def stage_select(force=False, spacing=2.0):
    """The whole grid search on Nyström kernels: does it choose the same model?"""
    path = os.path.join(RESULTS, "select.json")
    if os.path.exists(path) and not force:
        print("select: already done")
        return json.load(open(path))
    out = {"alphas": list(ALPHAS), "lams": list(LAMS), "folds": FOLDS, "spacing": spacing,
           "sets": []}
    for name in SETS:
        fs, T, y = table_of(name)
        model = CMFModel(fieldset=fs.copy(), method="krr")
        t0 = time.time()
        ex = grid_search(T, y, model, alphas=ALPHAS, lams=LAMS, folds=FOLDS, seed=SEED)
        t_ex = time.time() - t0
        Z = gaussian_centres(T, spacing, len(fs.specs))
        t0 = time.time()
        Ks = approx_kernels_multi(T, Z, [specs_at(fs, a) for a in ALPHAS])
        pre = {a: (K, None) for a, K in zip(ALPHAS, Ks)}
        model2 = CMFModel(fieldset=fs.copy(), method="krr")
        ap = grid_search(T, y, model2, alphas=ALPHAS, lams=LAMS, folds=FOLDS, seed=SEED,
                         precomputed=pre)
        t_ap = time.time() - t0
        row = {"set": name, "n_mol": int(T.n_mol), "m": int(Z.n_mol),
               "exact": {"alpha": float(ex.best_alpha), "lam": float(ex.best_params["lam"]),
                         "q2": float(ex.best_q2), "seconds": round(t_ex, 2)},
               "nystrom": {"alpha": float(ap.best_alpha), "lam": float(ap.best_params["lam"]),
                           "q2": float(ap.best_q2), "seconds": round(t_ap, 2)}}
        out["sets"].append(row)
        print(f"{name}: exact alpha {ex.best_alpha:g} lam {ex.best_params['lam']:g} "
              f"q2 {ex.best_q2:+.4f} ({t_ex:.1f} s); Nystrom m {Z.n_mol} alpha {ap.best_alpha:g} "
              f"lam {ap.best_params['lam']:g} q2 {ap.best_q2:+.4f} ({t_ap:.1f} s)", flush=True)
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(out, open(path, "w"), indent=1)
    return out


def stage_width(force=False, sets=("COX2", "THR"), spacings=(2.0, 1.5)):
    """A fixed grid of centres against every attenuation: where the basis stops spanning."""
    path = os.path.join(RESULTS, "width.json")
    if os.path.exists(path) and not force:
        print("width: already done")
        return json.load(open(path))
    out = {"alphas": list(ALPHAS), "spacings": list(spacings), "sets": []}
    for name in sets:
        fs, T, y = table_of(name)
        sets_specs = [specs_at(fs, a) for a in ALPHAS]
        K_ex_all = kernel_matrices_multi(T, T, sets_specs)
        row = {"set": name, "n_mol": int(T.n_mol), "rows": []}
        Zs = {s: gaussian_centres(T, s, len(fs.specs)) for s in spacings}
        K_ap_all = {s: approx_kernels_multi(T, Zs[s], sets_specs) for s in spacings}
        for ia, alpha in enumerate(ALPHAS):
            K_ex = joint_of(fs, T, K_ex_all[ia])
            lam, q2_ex = best_lam(K_ex, y)
            r = {"alpha": alpha, "lam": lam, "q2_exact": q2_ex, "space": []}
            sigma = 1.0 / np.sqrt(alpha)
            for s in spacings:
                K_ap = joint_of(fs, T, K_ap_all[s][ia])
                q2, _ = q2_at(K_ap, y, lam)
                r["space"].append({"spacing": s, "m": int(Zs[s].n_mol),
                                   "rel_error": rel_error(K_ex, K_ap), "q2": q2})
            row["rows"].append(r)
            print(f"{name} alpha {alpha:g} (width {sigma:.2f} A): exact q2 {q2_ex:+.4f}; "
                  + "; ".join(f"{e['spacing']:g} A m {e['m']} err {e['rel_error']:.1e} "
                              f"q2 {e['q2']:+.4f}" for e in r["space"]), flush=True)
        out["sets"].append(row)
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(out, open(path, "w"), indent=1)
    return out


# ---------------------------------------------------------------------------- the report


def report(results=RESULTS):
    parts = {}
    for name in ("rank", "select", "width"):
        p = os.path.join(results, f"{name}.json")
        if os.path.exists(p):
            parts[name] = json.load(open(p))
    if not parts:
        return "No Nyström results yet; run `python benchmarks/run_nystrom.py`."
    lines = ["# Can a low-rank kernel reproduce the exact CMF model?", "",
             "Plan item B6. The exact model holds an n x n matrix per field; the Nyström "
             "approximation builds a matrix of rank m from the kernels against m landmarks. The "
             "landmarks are either a random subset of the training molecules or a cubic grid of "
             "Gaussian centres of the field's own width covering the molecules. Every q² below is "
             f"10-fold cross-validated on the same folds; `error` is the relative Frobenius error "
             "of the joint training kernel.", ""]

    if "rank" in parts:
        k = parts["rank"]
        lines += ["## At the selected attenuation, against the exact model", "",
                  "The attenuation and the ridge are the exact grid search's choice for the set. "
                  "`q²` uses that ridge on the approximate kernel; `q² (own ridge)` re-selects "
                  "the ridge on the approximate kernel, which is what a user of the approximation "
                  "would do. For landmark molecules the figures are the mean over "
                  f"{k['draws']} random draws, with the range in brackets.", ""]
        for s in k["sets"]:
            lines += [f"**{s['set']}** -- {s['n_mol']} molecules, {s['atoms_per_molecule']:g} "
                      f"atoms each, box {' x '.join(f'{v:g}' for v in s['box'])} A; "
                      f"alpha {s['alpha']:g}, lambda {s['lam']:g}, exact q²cv "
                      f"**{s['q2_exact']:+.4f}**, exact kernel {s['seconds_exact']:.2f} s", "",
                      "| landmarks | m | error | q² | q² (own ridge) | kernel time |",
                      "|---|---:|---:|---:|---:|---:|"]
            for e in s["space"]:
                lines.append(f"| Gaussians, {e['spacing']:g} A | {e['m']} | {e['rel_error']:.1e} "
                             f"| {e['q2']:+.4f} | {e['q2_own']:+.4f} (λ {e['lam_own']:g}) | "
                             f"{e['seconds']:.2f} s |")
            for e in s["molecules"]:
                d = e["draws"]
                q2s = [x["q2"] for x in d]
                own = [x["q2_own"] for x in d]
                rng = "" if len(d) == 1 else f" [{min(q2s):+.3f}, {max(q2s):+.3f}]"
                lines.append(f"| molecules | {e['m']} | "
                             f"{np.mean([x['rel_error'] for x in d]):.1e} | "
                             f"{np.mean(q2s):+.4f}{rng} | {np.mean(own):+.4f} | "
                             f"~{e['seconds_kernels_estimate']:.2f} s |")
            lines.append("")
        # the summary: the coarsest grid of centres within 0.01 of the exact q2, and the fewest
        # landmark molecules for which every random draw is within 0.02 of it
        lines += ["### Summary", "",
                  "The coarsest grid of Gaussian centres whose q² (own ridge) is within 0.01 of "
                  "the exact one, and the fewest landmark molecules for which every one of the "
                  f"{k['draws']} random draws is within 0.02 of it.", "",
                  "| set | n | alpha | exact q² | coarsest grid within 0.01 | m | q² | "
                  "fewest molecules, every draw within 0.02 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s in k["sets"]:
            ok = [e for e in s["space"] if abs(e["q2_own"] - s["q2_exact"]) <= 0.01]
            okm = [e for e in s["molecules"]
                   if max(abs(x["q2"] - s["q2_exact"]) for x in e["draws"]) <= 0.02]
            e = ok[0] if ok else None
            lines.append(f"| {s['set']} | {s['n_mol']} | {s['alpha']:g} | {s['q2_exact']:+.3f} | "
                         + (f"{e['spacing']:g} A | {e['m']} | {e['q2_own']:+.3f} | " if e
                            else "none | - | - | ")
                         + (f"{okm[0]['m']} of {s['n_mol']} |" if okm else "none |"))
        lines.append("")

    if "select" in parts:
        g = parts["select"]
        lines += [f"## The whole grid search on Nyström kernels, centres {g['spacing']:g} A apart",
                  "",
                  "The same search over attenuation and ridge, once on exact kernels and once on "
                  "the approximate ones. The question is whether a user of the approximation "
                  "would end up with the same model.", "",
                  "| set | n | m | exact: alpha, λ, q² | Nyström: alpha, λ, q² | same choice | "
                  "exact search | Nyström search |", "|---|---:|---:|---|---|---|---:|---:|"]
        for s in g["sets"]:
            e, a = s["exact"], s["nystrom"]
            same = (e["alpha"] == a["alpha"]) and (e["lam"] == a["lam"])
            lines.append(f"| {s['set']} | {s['n_mol']} | {s['m']} | {e['alpha']:g}, {e['lam']:g}, "
                         f"{e['q2']:+.4f} | {a['alpha']:g}, {a['lam']:g}, {a['q2']:+.4f} | "
                         f"{'yes' if same else 'no'} | {e['seconds']:.1f} s | {a['seconds']:.1f} s |")
        lines.append("")

    if "width" in parts:
        w = parts["width"]
        lines += ["## A fixed grid of centres against the attenuation", "",
                  "A Gaussian of attenuation alpha has width 1/sqrt(alpha): 3.2 A at 0.1, 1.4 A "
                  "at 0.5, 0.45 A at 5. A grid of centres spans the fields only while its spacing "
                  "is not much larger than that width, so the approximation is expected to fail "
                  "at strong attenuation whatever the number of molecules. The ridge is the "
                  "exact model's best at each attenuation.", ""]
        for s in w["sets"]:
            heads = " | ".join(f"{sp:g} A: m, error, q²" for sp in w["spacings"])
            lines += [f"**{s['set']}** -- {s['n_mol']} molecules", "",
                      f"| alpha | width (A) | exact q² | {heads} |",
                      "|---:|---:|---:|" + "---|" * len(w["spacings"])]
            for r in s["rows"]:
                cells = " | ".join(f"{e['m']}, {e['rel_error']:.0e}, {e['q2']:+.3f}"
                                   for e in r["space"])
                lines.append(f"| {r['alpha']:g} | {1 / np.sqrt(r['alpha']):.2f} | "
                             f"{r['q2_exact']:+.3f} | {cells} |")
            lines.append("")
    return "\n".join(lines)


def write_table(results=RESULTS) -> str:
    """Rebuild ``table.md`` from the result files and return its text."""
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                        # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="*", default=["rank", "select", "width"],
                    choices=["rank", "select", "width"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    if "rank" in a.stage:
        stage_rank(a.force)
    if "select" in a.stage:
        stage_select(a.force)
    if "width" in a.stage:
        stage_width(a.force)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
