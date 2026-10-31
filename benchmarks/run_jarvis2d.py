#!/usr/bin/env python
"""
Continuous molecular fields of two-dimensional materials: the JARVIS-DFT 2D monolayers.

1,103 monolayers of the JARVIS-DFT 2D set (Choudhary et al., Sci. Rep. 7 (2017) 5179; the
``dft_2d`` file of 12 December 2022, downloaded once from figshare into ``benchmarks/cache/jarvis2d``)
with their OptB88vdW band gap (eV), formation energy (eV/atom) and, where computed, exfoliation
energy (meV/atom).  Every structure is a layer in a cell with vacuum along the third lattice
vector, which is the case plan item M1 is about.

What is measured is the three changes of M1, one at a time, against the behaviour before it:

    crystal-bg   the structure flagged as a crystal, as the files come: the neighbour list
                 crosses the vacuum and the background is per unit *volume*, vacuum included,
                 so the descriptor depends on a number the data set chose (the width of the
                 vacuum).  This is what the program did with a layer before M1.
    nobg         flagged as a layer, background off: the earlier workaround.
    layer-bg     flagged as a layer, background per unit *area* with the pairs' height
                 differences in it (the first half of M1).
    heights      as layer-bg, with the channels resolved by the heights of both atoms above the
                 mean plane of the layer, the two faces averaged (the second half of M1).
    heights-oriented  the same with the third lattice vector taken as a physical "up", a
                 control: a free-standing layer has no up, so this should not help.

each on two field sets, ``physchem+mass`` (electrostatic, steric, electronegativity, inverse
square root of the mass) and ``all`` (those plus one indicator field per element in at least
``--min-structures`` structures, plus one for the rest); the height channels run on the
physico-chemical set only, because they multiply every pair block by the square of the number
of height channels.  ``-gauss`` adds a Gaussian layer on the cosine-normalised kernel.

Two numbers are reported for every variant and target: the model's error under 5-fold
cross-validation (``KFold(shuffle=True, random_state=0)``; the width of the channels and the
regularisation chosen by 10-fold cross-validation inside every training fold; MAE mean +/- sd
over the folds, RMSE and R2 on the pooled held-out predictions), and the *vacuum sensitivity* of
the descriptor: the median over the structures of the relative change of the channel vector
when the vacuum of the cell is doubled, which is zero for a descriptor that does not know the
vacuum exists.

Usage
-----
    python benchmarks/run_jarvis2d.py [--targets gap,formation,exfoliation] [--variants ...] [--force]
    python benchmarks/run_jarvis2d.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar.invariant import autocorrelation_kernel, channel_scale  # noqa: E402
from cmfqsar.model import _Learner, cv_predict, r2_score, rmse  # noqa: E402
from cmfqsar.periodic import (INORGANIC_FIELDS, PeriodicSpec, centre_layer,  # noqa: E402
                              crystal_fields, elements_in, inorganic_fieldset,
                              periodic_block_sizes, periodic_channel_labels,
                              periodic_feature_matrix)

URL = "https://ndownloader.figshare.com/files/38521268"
CACHE = os.path.join(HERE, "cache", "jarvis2d")
RESULTS = os.path.join(HERE, "results", "jarvis2d")
OUTER, INNER, SPLIT_SEED = 5, 10, 0
LAMS = (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
SIGMAS = (0.5, 0.75)
GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0)
R_MAX, TAPER = 10.0, 2.0
Z_MAX, Z_STEP = 3.0, 1.0

MASS = INORGANIC_FIELDS + ("prop:inverse_sqrt_mass",)
TARGETS = {
    "gap": ("optb88vdw_bandgap", "eV", "OptB88vdW band gap"),
    "formation": ("formation_energy_peratom", "eV/atom", "formation energy per atom"),
    "exfoliation": ("exfoliation_energy", "meV/atom", "exfoliation energy per atom"),
}
VARIANTS = {
    "crystal-bg": dict(physchem=MASS, elements=False, layer=False),
    "nobg": dict(physchem=MASS, elements=False, layer=True, background=False),
    "layer-bg": dict(physchem=MASS, elements=False, layer=True),
    "heights": dict(physchem=MASS, elements=False, layer=True, heights=True),
    "heights-oriented": dict(physchem=MASS, elements=False, layer=True, heights=True, oriented=True),
    "all-crystal-bg": dict(physchem=MASS, elements=True, layer=False),
    "all-nobg": dict(physchem=MASS, elements=True, layer=True, background=False),
    "all-layer-bg": dict(physchem=MASS, elements=True, layer=True),
    "layer-bg-gauss": dict(physchem=MASS, elements=False, layer=True, gauss=True),
    "heights-gauss": dict(physchem=MASS, elements=False, layer=True, heights=True, gauss=True),
    "all-layer-bg-gauss": dict(physchem=MASS, elements=True, layer=True, gauss=True),
    # the controls for the Gaussian layer: the same kernel on the crystal flags and without background
    "crystal-bg-gauss": dict(physchem=MASS, elements=False, layer=False, gauss=True),
    "nobg-gauss": dict(physchem=MASS, elements=False, layer=True, background=False, gauss=True),
    "all-crystal-bg-gauss": dict(physchem=MASS, elements=True, layer=False, gauss=True),
    "all-nobg-gauss": dict(physchem=MASS, elements=True, layer=True, background=False, gauss=True),
}


# --------------------------------------------------------------------------- data
def fetch():
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "d2-12-12-2022.zip")
    if not os.path.exists(path):
        print(f"Downloading {URL} ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (cmfqsar benchmark)"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as fh:
            fh.write(r.read())
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".json")][0]
        return json.loads(z.read(name))


def to_atoms(rec, layer: bool):
    """One JARVIS record as ASE ``Atoms``: a layer with the vacuum along c, gathered and
    centred, or the same cell flagged as a crystal."""
    from ase import Atoms
    a = rec["atoms"]
    cell = np.asarray(a["lattice_mat"], float)
    coords = np.asarray(a["coords"], float)
    if a.get("cartesian", False):
        atoms = Atoms(symbols=list(a["elements"]), positions=coords, cell=cell, pbc=True)
    else:
        atoms = Atoms(symbols=list(a["elements"]), scaled_positions=coords, cell=cell, pbc=True)
    atoms.set_pbc((True, True, False))
    atoms = centre_layer(atoms)
    if not layer:
        atoms.set_pbc(True)
    atoms.info["name"] = rec["jid"]
    return atoms


def with_vacuum_doubled(atoms):
    out = atoms.copy()
    cell = out.get_cell()[:].copy()
    cell[2] = 2.0 * cell[2]
    pos = out.get_positions()
    out.set_cell(cell, scale_atoms=False)
    out.set_positions(pos + 0.5 * (cell[2] - atoms.get_cell()[:][2])[None, :])
    return out


def load(args):
    recs = fetch()
    print(f"JARVIS-DFT 2D: {len(recs)} monolayers")
    thick = []
    for r in recs:
        a = to_atoms(r, True)
        z = a.get_positions() @ (np.cross(a.get_cell()[0], a.get_cell()[1])
                                 / np.linalg.norm(np.cross(a.get_cell()[0], a.get_cell()[1])))
        thick.append(float(z.max() - z.min()))
    thick = np.array(thick)
    print(f"  thickness of the layers: median {np.median(thick):.2f} A, 90 % below "
          f"{np.percentile(thick, 90):.2f} A, max {thick.max():.2f} A; "
          f"{int((thick > 2 * Z_MAX).sum())} thicker than the height grid (+/- {Z_MAX:g} A)")
    targets = {}
    for key, (field, unit, label) in TARGETS.items():
        y = np.array([float(r[field]) if r.get(field) not in (None, "na", "") else np.nan for r in recs])
        print(f"  {label}: {int(np.isfinite(y).sum())} values, median {np.nanmedian(y):.3g} {unit}")
        targets[key] = y
    return recs, targets, thick


# --------------------------------------------------------------------------- the model
def gauss_layer(K, gamma, scale):
    return np.exp(-gamma * (2.0 - 2.0 * K) / scale)


def median_d2(K_tr):
    d2 = 2.0 - 2.0 * K_tr[np.triu_indices_from(K_tr, k=1)]
    m = float(np.median(d2))
    return m if m > 1e-12 else 1.0


def spec_for(cfg, sigma):
    return PeriodicSpec(r_max=R_MAX, step=sigma, sigma=sigma, taper=cfg.get("taper", TAPER),
                        background=cfg.get("background", True),
                        z_max=Z_MAX if cfg.get("heights") else 0.0, z_step=Z_STEP,
                        oriented=bool(cfg.get("oriented", False)))


def features(recs, fs, cfg, sigma, doubled=False):
    atoms = [to_atoms(r, cfg["layer"]) for r in recs]
    if doubled:
        atoms = [with_vacuum_doubled(a) for a in atoms]
    crystals = [crystal_fields(a, fs, a.info["name"]) for a in atoms]
    spec = spec_for(cfg, sigma)
    return periodic_feature_matrix(crystals, spec, n_fields=fs.n_fields), spec


def vacuum_sensitivity(F, F2):
    num = np.linalg.norm(F - F2, axis=1)
    den = np.maximum(np.linalg.norm(F, axis=1), 1e-300)
    return float(np.median(num / den)), float(np.max(num / den))


def folds_for(n):
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=OUTER, shuffle=True, random_state=SPLIT_SEED).split(np.arange(n)))


def run_variant(tag, target, recs, y, elements, args, feats_cache):
    cfg = VARIANTS[tag]
    fs = inorganic_fieldset(elements if cfg["elements"] else [], physchem=cfg["physchem"])
    gauss = cfg.get("gauss", False)
    ok = np.isfinite(y)
    idx = np.flatnonzero(ok)
    y = y[ok]
    unit = TARGETS[target][1]
    print(f"\n=== {tag} on {target}: {fs.n_fields} fields, {len(y)} structures"
          + (", Gaussian layer" if gauss else ""))
    feats, specs, sens = {}, {}, {}
    for s in SIGMAS:
        key = (tag.replace("-gauss", ""), s)
        if key not in feats_cache:
            t0 = time.time()
            F, spec = features(recs, fs, cfg, s)
            F2, _ = features(recs, fs, cfg, s, doubled=True)
            feats_cache[key] = (F, spec, vacuum_sensitivity(F, F2))
            print(f"  features {F.shape} at sigma {s:g}: {time.time() - t0:.0f} s; vacuum sensitivity "
                  f"median {feats_cache[key][2][0]:.2e}, max {feats_cache[key][2][1]:.2e}")
        F, spec, sv = feats_cache[key]
        feats[s], specs[s], sens[s] = F[idx], spec, sv
    folds = folds_for(len(y))
    pred = np.full(len(y), np.nan)
    per_fold, chosen = [], []
    t0 = time.time()
    for k, (tr, te) in enumerate(folds):
        best = None
        for s, F in feats.items():
            blocks = periodic_block_sizes(specs[s], fs.n_fields)
            sc = channel_scale(F[tr], specs[s].n_bins, blocks=blocks)
            K_tr = autocorrelation_kernel(F[tr], scale=sc, n_bins=specs[s].n_bins, blocks=blocks)
            med = median_d2(K_tr) if gauss else None
            for g in (GAMMAS if gauss else (None,)):
                Kg = gauss_layer(K_tr, g, med) if gauss else K_tr
                for lam in LAMS:
                    p = cv_predict(Kg, y[tr], "krr", {"lam": lam}, folds=INNER, seed=0)
                    err = rmse(y[tr], p)
                    if best is None or err < best[0]:
                        best = (err, s, g, lam, sc, blocks, med)
        err, s, g, lam, sc, blocks, med = best
        F = feats[s]
        K_tr = autocorrelation_kernel(F[tr], scale=sc, n_bins=specs[s].n_bins, blocks=blocks)
        K_te = autocorrelation_kernel(F[tr], F[te], scale=sc, n_bins=specs[s].n_bins, blocks=blocks)
        if gauss:
            K_tr, K_te = gauss_layer(K_tr, g, med), gauss_layer(K_te, g, med)
        learner = _Learner("krr", {"lam": lam}, True).fit(K_tr, y[tr])
        pred[te] = learner.predict(K_te)
        mae = float(np.mean(np.abs(pred[te] - y[te])))
        per_fold.append({"fold": k, "sigma": s, "gamma": g, "lam": lam, "inner_rmse": float(err),
                         "mae": mae, "rmse": float(rmse(y[te], pred[te])), "n_test": int(len(te))})
        chosen.append((s, g, lam))
        print(f"  fold {k}: sigma {s:g}, lam {lam:g}" + (f", gamma {g:g}" if gauss else "")
              + f"  ->  MAE {mae:.4g} {unit}  ({time.time() - t0:.0f} s)")
    maes = np.array([f["mae"] for f in per_fold])
    dummy = float(np.mean([np.mean(np.abs(y[te] - np.mean(y[tr]))) for tr, te in folds]))
    out = {
        "variant": tag, "target": target, "unit": unit, "n": int(len(y)),
        "fields": [sp.label for sp in fs.specs], "n_fields": fs.n_fields, "gauss_layer": gauss,
        "layer_flags": cfg["layer"], "background": cfg.get("background", True),
        "heights": bool(cfg.get("heights")), "oriented": bool(cfg.get("oriented", False)),
        "z_max": Z_MAX if cfg.get("heights") else None, "z_step": Z_STEP if cfg.get("heights") else None,
        "r_max": R_MAX, "taper": TAPER, "sigmas": list(SIGMAS), "lams": list(LAMS),
        "n_channels": {f"{s:g}": int(feats[s].shape[1]) for s in SIGMAS},
        "vacuum_sensitivity": {f"{s:g}": {"median": sens[s][0], "max": sens[s][1]} for s in SIGMAS},
        "mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=0)), "mae_dummy": dummy,
        "rmse_pooled": float(rmse(y, pred)), "r2_pooled": float(r2_score(y, pred)),
        "per_fold": per_fold, "elapsed_s": time.time() - t0,
        "names": [recs[i]["jid"] for i in idx], "y": [float(v) for v in y],
        "y_pred": [float(v) for v in pred],
    }
    print(f"  MAE {maes.mean():.4g} +/- {maes.std():.2g} {unit} (mean predictor {dummy:.4g}); "
          f"pooled R2 {out['r2_pooled']:.3f}")
    if not gauss:
        s, _, lam = max(set(chosen), key=chosen.count)
        F, spec = feats[s], specs[s]
        blocks = periodic_block_sizes(spec, fs.n_fields)
        sc = channel_scale(F, spec.n_bins, blocks=blocks)
        K = autocorrelation_kernel(F, scale=sc, n_bins=spec.n_bins, blocks=blocks)
        learner = _Learner("krr", {"lam": lam}, True).fit(K, y)
        Z = F * sc
        Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
        w = Z.T @ learner.alpha_
        sd = Z.std(axis=0)
        labels = periodic_channel_labels(fs, spec)
        order = np.argsort(-np.abs(w) * sd)
        out["co_pattern"] = {"sigma": s, "lam": lam,
                             "top": [{"channel": labels[c], "weight": float(w[c]), "spread": float(sd[c]),
                                      "effect": float(w[c] * sd[c])} for c in order[:25]]}
        print("  co-pattern (weight x spread, the channels that move the prediction most):")
        for row in out["co_pattern"]["top"][:10]:
            print(f"    {row['effect']:+9.3f}  {row['channel']}")
    return out


# --------------------------------------------------------------------------- the report
def report(path):
    rows = []
    for target in TARGETS:
        for tag in VARIANTS:
            f = os.path.join(RESULTS, f"{target}_{tag}.json")
            if os.path.exists(f):
                rows.append(json.load(open(f)))
    lines = ["# Continuous molecular fields of two-dimensional materials: JARVIS-DFT 2D", "",
             "1,103 monolayers (JARVIS-DFT 2D, file of 12 December 2022), 5-fold cross-validation "
             "(shuffled, seed 0); the width of the channels and the regularisation chosen by 10-fold "
             "cross-validation inside every training fold. MAE mean +/- sd over the folds, RMSE and R2 "
             "on the pooled held-out predictions; 'mean' is the MAE of predicting the training mean. "
             "Vacuum sensitivity: median relative change of the channel vector when the vacuum of the "
             "cell is doubled (at the smaller channel width).", "",
             "| target | variant | flags | background | heights | fields | channels | vacuum sensitivity "
             "| MAE | mean | RMSE | R2 | time (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s0 = f"{SIGMAS[0]:g}"
        hz = (f"+/- {r['z_max']:g} A every {r['z_step']:g}" + (", oriented" if r["oriented"] else "")
              if r["heights"] else "no")
        lines.append(f"| {TARGETS[r['target']][2]} ({r['unit']}) | {r['variant']} | "
                     f"{'layer' if r['layer_flags'] else 'crystal'} | "
                     f"{('per area' if r['layer_flags'] else 'per volume') if r['background'] else 'kept'} | "
                     f"{hz} | {r['n_fields']} | {r['n_channels'][s0]} | "
                     f"{r['vacuum_sensitivity'][s0]['median']:.1e} | "
                     f"{r['mae_mean']:.3g} +/- {r['mae_sd']:.2g} | {r['mae_dummy']:.3g} | "
                     f"{r['rmse_pooled']:.3g} | {r['r2_pooled']:.3f} | {r['elapsed_s']:.0f} |")
    lines.append("")
    for r in rows:
        if "co_pattern" in r:
            lines += [f"## Co-pattern of `{r['variant']}` on the {TARGETS[r['target']][2]} "
                      f"(sigma {r['co_pattern']['sigma']:g} A, lambda {r['co_pattern']['lam']:g})", "",
                      "| effect | channel |", "|---|---|"]
            lines += [f"| {t['effect']:+.3f} | {t['channel']} |" for t in r["co_pattern"]["top"][:10]]
            lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Report written to {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="gap,formation,exfoliation")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--min-structures", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        report(os.path.join(RESULTS, "table.md"))
        return
    recs, targets, thick = load(args)
    elements = elements_in([to_atoms(r, True) for r in recs], args.min_structures)
    print(f"  indicator fields for {len(elements)} elements: {', '.join(elements)}")
    feats_cache = {}
    for target in [t for t in args.targets.split(",") if t]:
        y = targets[target]
        if np.isfinite(y).sum() < 100:
            print(f"\n{target}: fewer than 100 values, skipped")
            continue
        for tag in [v for v in args.variants.split(",") if v]:
            out_path = os.path.join(RESULTS, f"{target}_{tag}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"{target}/{tag}: done, skipped (use --force)")
                continue
            res = run_variant(tag, target, recs, y, elements, args, feats_cache)
            res["thickness_median"] = float(np.median(thick))
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=1)
    report(os.path.join(RESULTS, "table.md"))


if __name__ == "__main__":
    main()
