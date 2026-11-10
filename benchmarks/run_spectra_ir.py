#!/usr/bin/env python
"""
A spectrum as the sample where the spectrum should hold the answer: the infrared spectra of
the substituted benzoic acids against their Hammett constants (plan item SP1, second
measurement of the half that needs no new data).

The first measurement of the item (`run_spectra.py`) put a powder pattern against a phonon
frequency and found almost nothing, because a diffraction pattern does not show what decides
that property.  This script takes a pair in which the experiment should show it.  The sample is
the infrared spectrum of an acid: every normal mode of the ORCA frequency job of plan item SP2
as a stick, 189 acids (`examples/spectra/benzoic_acids_ir.json`).  The target is the Hammett
constant of the substituent at its position, which is a measured number: for most substituents
it is the measured shift of the pKa of this very acid against benzoic acid.  The spectrum is
computed and the target is measured, and the calculation is not told the target.

The questions:

1. Does the kernel of the spectra predict the constant, and how does it compare with a count
   of the atoms around the carboxyl group and with the structural field kernels?
2. Does the whole spectrum say more than its one band that a chemist would read, the carbonyl
   stretch?  The reference is a linear regression on the position of that band.
3. Does the spectrum add to the structural kernel when the two are joined?
4. Where on the axis does the model put its weight (the co-spectrum)?

A control with a known answer comes first: the length of the C=O bond of the optimised
geometry follows the carbonyl band (Badger's rule), so a kernel that can read the position of
a line has to predict it from the spectrum.

Protocol, as in parts 7 and 8 of `run_band.py`: ten repeats of ten folds, once with whole acids
held out and once with whole substituents held out (the series holds each substituent twice,
and a spectrum shows the bands of the substituent itself).  Every setting (the width of the
broadening, the power of the intensity, the width of an envelope, the share of a kernel in a
sum, the ridge) is chosen on five folds inside every training part.  A difference in q2
carries a 95 % bootstrap interval over the units that the folds hold out.

Usage
-----
    python benchmarks/run_spectra_ir.py            # the measurement, then the table
    python benchmarks/run_spectra_ir.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar import spectrum as SP  # noqa: E402
from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.localenv import (count_features, count_kernel, envelope_width,  # noqa: E402
                              environment, local_features, local_kernel, types_of)
from cmfqsar.model import krr_fit  # noqa: E402
import run_band as B  # noqa: E402

SPECTRA = os.path.join(ROOT, "examples", "spectra", "benzoic_acids_ir.json")
RESULTS = os.path.join(HERE, "results", "spectra_ir")

#: the part of the axis that is used, in cm^-1: the range of a mid-infrared instrument
RANGE = (400.0, 4000.0)
#: the widths of the broadening tried, in cm^-1
WIDTHS = (2.0, 5.0, 10.0, 20.0, 50.0)
#: the powers of the intensity tried: 1 keeps the intensity, 0.5 takes its square root
POWERS = (1.0, 0.5)
#: the factors of the Gaussian kernel on top of the spectrum kernel
GAMMAS = (0.25, 1.0, 4.0)
#: the region of the carbonyl stretch, in cm^-1
CARBONYL = (1600.0, 1850.0)
#: the shares of the spectrum kernel tried in a sum with a structural kernel
SHARES = (0.25, 0.5, 0.75)
#: the decays of the envelope of the structural kernels (widths of 10, 4.5, 2 and 1 A)
RHOS = B.CHARGE_RHOS
REPEATS = B.REPEATS
FOLDS = B.FOLDS
LAMS = B.WIDE_LAMS
SCRAMBLES = 50
#: what a region of the mid-infrared holds in a benzoic acid, for the table of the co-spectrum
REGIONS = ((3450.0, 3800.0, "O-H stretch"),
           (2800.0, 3200.0, "C-H stretch"),
           (1650.0, 1800.0, "C=O stretch"),
           (1550.0, 1650.0, "ring stretch"),
           (1000.0, 1550.0, "C-O stretch, O-H bend and ring modes"),
           (400.0, 1000.0, "out-of-plane bends and ring deformations"))


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# The data
# --------------------------------------------------------------------------- #
def read_spectra(path: str, names: Sequence[str]) -> List[SP.Spectrum]:
    """The infrared stick spectra of the acids, in the order of `names`."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run examples/make_band_examples.py --ir")
    with open(path, encoding="utf-8") as fh:
        rows = {r["name"]: r for r in json.load(fh)["acids"]}
    missing = [n for n in names if n not in rows]
    if missing:
        raise SystemExit(f"{path} has no spectrum for {missing[0]} ({len(missing)} acids)")
    return [SP.Spectrum(rows[n]["wavenumbers"], rows[n]["intensities"], "sticks",
                        "wavenumber", n) for n in names]


def with_target(targets: AtomTargets, y: np.ndarray, name: str) -> AtomTargets:
    """The same samples with another target: one value per acid, on its carbonyl carbon."""
    return AtomTargets(molecule=np.asarray(targets.molecule, int),
                       atom=np.asarray(targets.atom, int), y=np.asarray(y, float),
                       n_mol=targets.n_mol, name=name,
                       per_molecule=list(targets.per_molecule))


def setting_name(width: float, power: float) -> str:
    return f"{width:g} cm^-1, power {power:g}"


def setting_values(name: str) -> Tuple[float, float]:
    width, power = name.split(" cm^-1, power ")
    return float(width), float(power.split(",")[0])


def outside(spectrum: SP.Spectrum, lo: float, hi: float) -> SP.Spectrum:
    """The spectrum without the lines between two positions."""
    keep = (spectrum.x < lo) | (spectrum.x > hi)
    return SP.Spectrum(spectrum.x[keep], spectrum.y[keep], spectrum.kind, spectrum.axis,
                       spectrum.name)


def spectrum_kernels(spectra: Sequence[SP.Spectrum], lo: float, hi: float,
                     widths=WIDTHS, powers=POWERS) -> Dict[str, np.ndarray]:
    """The cosine kernel of the spectra between `lo` and `hi` for every width and power."""
    return {setting_name(w, p): SP.kernel(spectra, lo, hi, w, power=p)
            for w in widths for p in powers}


def layered(kernels: Dict[str, np.ndarray], gammas=GAMMAS) -> Dict[str, np.ndarray]:
    """A Gaussian kernel on top of every kernel: exp(-factor D^2 / m), with D^2 = 2 - 2 K the
    squared distance of two normalised spectra and m the median of D^2 over all pairs.  The
    median uses the spectra and not the target."""
    out = {}
    for key, K in kernels.items():
        d2 = 2.0 - 2.0 * K
        m = float(np.median(d2[np.triu_indices(len(K), 1)]))
        m = m if m > 1e-12 else 1.0
        for g in gammas:
            out[f"{key}, factor {g:g}"] = np.exp(-g * d2 / m)
    return out


def structure_kernels(fields, targets: AtomTargets, rhos=RHOS) -> Dict[str, np.ndarray]:
    """The invariant local kernel around the carbonyl carbon for every envelope width."""
    out = {}
    for rho in rhos:
        plain = [environment(fields[int(targets.molecule[r])], int(targets.atom[r]), rho)
                 for r in range(len(targets))]
        out[f"{envelope_width(rho):.2f} A"] = local_kernel(local_features(plain))
    return out


def joined(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
           shares=SHARES) -> Dict[str, np.ndarray]:
    """Every sum of a kernel of `a` and a kernel of `b`, the second with the share `s` and
    the first with ``1 - s``.  Both are cosine kernels, so the diagonal of the sum is one."""
    return {f"{ka} + {kb}, share {s:g}": (1.0 - s) * Ka + s * Kb
            for ka, Ka in a.items() for kb, Kb in b.items() for s in shares}


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def nested(kernels: Dict[str, np.ndarray], tg: AtomTargets, seeds, name: str):
    """Repeated cross-validation with the kernel and the ridge chosen inside every training
    part (:func:`run_band.nested_choice`); the results and how often each kernel was chosen."""
    results, chosen = [], {}
    for s in seeds:
        res, picks = B.nested_choice(kernels, tg, lams=LAMS, folds=FOLDS, seed=s, name=name)
        results.append(res)
        for key, _lam in picks:
            chosen[key] = chosen.get(key, 0) + 1
    return results, chosen


def regression(X: np.ndarray, tg: AtomTargets, seeds, name: str):
    return [B.regression_reference(X, tg, FOLDS, s, name) for s in seeds], {}


def row_of(results, chosen: Dict[str, int], tg: AtomTargets, table: dict) -> dict:
    y = np.asarray(tg.y, float)
    q2, sd, err = B.repeated_q2(results, y)
    twin = B.twin_difference(results, tg, table)
    return {"q2": q2, "sd": sd, "rmse": err, "chosen": chosen,
            "twin_q2": twin["q2"], "twin_error": twin["error"]}


def compare(a, b, tg: AtomTargets) -> List[float]:
    d, lo, hi = B.paired_repeats(a, b, tg)
    return [float(d), float(lo), float(hi)]


def most_chosen(chosen: Dict[str, int]) -> str:
    return max(chosen, key=chosen.get)


def target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """The correlation between the centred kernel and the products of the centred targets,
    over the pairs of samples: how far samples that a description finds alike have alike
    targets.  It needs no fit."""
    n = len(y)
    h = np.eye(n) - 1.0 / n
    t = np.asarray(y, float) - float(np.mean(y))
    upper = np.triu_indices(n, 1)
    return float(np.corrcoef((h @ K @ h)[upper], np.outer(t, t)[upper])[0, 1])


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
ROWS = (
    ("counts", "atom-centred counts around the carbonyl carbon"),
    ("band", "linear regression on the position of the carbonyl band"),
    ("band_position", "linear regression on the position of the carbonyl band, meta and para "
                      "separately (two equations; it is told the position)"),
    ("structure", "structure: the five fields, Gasteiger charges"),
    ("structure_q", "structure: the electrostatic field alone, quantum charges"),
    ("spectrum", "infrared spectrum, 400 to 4000 cm^-1"),
    ("spectrum_gauss", "infrared spectrum, Gaussian kernel on top"),
    ("carbonyl", "infrared spectrum, the region of the carbonyl stretch alone"),
    ("rest", "infrared spectrum without the region of the carbonyl stretch"),
    ("structure+spectrum", "structure (five fields) and infrared spectrum, joined"),
    ("structure_q+spectrum", "structure (quantum charges) and infrared spectrum, joined"),
)
LABEL = dict(ROWS)
COMPARE = (
    ("spectrum", "counts"), ("spectrum", "band"), ("spectrum", "band_position"),
    ("spectrum", "structure"), ("spectrum", "structure_q"),
    ("spectrum_gauss", "spectrum"), ("spectrum_gauss", "band"),
    ("carbonyl", "band"), ("rest", "spectrum"), ("rest", "counts"),
    ("structure+spectrum", "structure"), ("structure+spectrum", "spectrum"),
    ("structure_q+spectrum", "structure_q"),
)
CONTROL_ROWS = (
    ("band", "linear regression on the position of the carbonyl band"),
    ("spectrum", "infrared spectrum, 400 to 4000 cm^-1"),
    ("carbonyl", "infrared spectrum, the region of the carbonyl stretch alone"),
    ("rest", "infrared spectrum without the region of the carbonyl stretch"),
)


def run(repeats: int = REPEATS, results: str = RESULTS, scrambles: int = SCRAMBLES,
        data: str = B.DATA, structures: str = B.STRUCTURES, quantum: str = B.QUANTUM,
        spectra_path: str = SPECTRA) -> dict:
    t_all = time.time()
    os.makedirs(results, exist_ok=True)
    mols, names, fields5, targets, _fs5, table = B.read(data, structures=structures)
    spectra = read_spectra(spectra_path, names)
    spectra = [spectra[int(m)] for m in targets.molecule]
    band = np.asarray(targets.y, float)
    sigma = np.asarray(table["sigma"], float)
    para = np.array([p == "para" for p in table["position"]], float)
    optimised = list(load_dataset(data, atom_activity=B.TARGET).mols)
    length = B.carbonyl_lengths(optimised, targets)
    seeds = list(range(int(repeats)))
    n_lines = [int(np.sum((s.x >= RANGE[0]) & (s.x <= RANGE[1]))) for s in spectra]
    log(f"=== {len(spectra)} acids, {int(np.median(n_lines))} lines between {RANGE[0]:g} and "
        f"{RANGE[1]:g} cm^-1 at the median ({min(n_lines)} to {max(n_lines)}); the Hammett "
        f"constant lies between {sigma.min():+.2f} and {sigma.max():+.2f} (standard "
        f"deviation {sigma.std():.2f})")

    # ---- the kernels
    t0 = time.time()
    kernels = {"spectrum": spectrum_kernels(spectra, *RANGE),
               "carbonyl": spectrum_kernels(spectra, *CARBONYL),
               "rest": spectrum_kernels([outside(s, *CARBONYL) for s in spectra], *RANGE)}
    kernels["spectrum_gauss"] = layered(kernels["spectrum"])
    log(f"  kernels of the spectra, {time.time() - t0:.0f} s")
    t0 = time.time()
    qmols = B.with_quantum_charges(mols, names, quantum)
    fs_q = FieldSet.create(physchem=("electrostatic",), charge_model="file")
    fields_q = [build_fields(m, fs_q, name=n) for m, n in zip(qmols, names)]
    kernels["structure"] = structure_kernels(fields5, targets)
    kernels["structure_q"] = structure_kernels(fields_q, targets)
    kernels["counts"] = {"counts": count_kernel(count_features(
        fields5, targets, types_of(fields5, min_atoms=10)))}
    kernels["structure+spectrum"] = joined(kernels["structure"], kernels["spectrum"])
    kernels["structure_q+spectrum"] = joined(kernels["structure_q"], kernels["spectrum"])
    log(f"  structural kernels, {time.time() - t0:.0f} s")

    out = {"n": len(spectra), "repeats": int(repeats), "folds": FOLDS, "range": list(RANGE),
           "carbonyl": list(CARBONYL), "widths": list(WIDTHS), "powers": list(POWERS),
           "gammas": list(GAMMAS), "shares": list(SHARES),
           "lines": {"median": int(np.median(n_lines)), "min": min(n_lines),
                     "max": max(n_lines)},
           "sigma": {"min": float(sigma.min()), "max": float(sigma.max()),
                     "sd": float(sigma.std())},
           "r_band_sigma": float(np.corrcoef(band, sigma)[0, 1]),
           "r_band_length": float(np.corrcoef(band, length)[0, 1]),
           "length": {"min": float(length.min()), "max": float(length.max()),
                      "sd": float(length.std())},
           "n_substituents": len(set(table["substituent"])), "units": {}, "control": {}}

    # ---- the control: the length of the C=O bond from the spectrum
    tg_len = with_target(targets, length * 100.0, "C=O length, pm")
    for unit, tg in (("acids", tg_len),
                     ("substituents", B.substituent_targets(tg_len, table))):
        t0 = time.time()
        res = {"band": regression(band, tg, seeds, LABEL["band"])}
        for key in ("spectrum", "carbonyl", "rest"):
            res[key] = nested(kernels[key], tg, seeds, LABEL[key])
        out["control"][unit] = {k: row_of(r, c, tg, table) for k, (r, c) in res.items()}
        out["control"][unit]["spectrum_against_band"] = compare(res["spectrum"][0],
                                                                res["band"][0], tg)
        log(f"  control, whole {unit} held out ({time.time() - t0:.0f} s): "
            + "; ".join(f"{k} q2 {out['control'][unit][k]['q2']:+.3f}"
                        for k in ("band", "spectrum", "carbonyl", "rest")))

    # ---- the Hammett constant
    tg_sigma = with_target(targets, sigma, "Hammett constant")
    picked = {}
    for unit, tg in (("acids", tg_sigma),
                     ("substituents", B.substituent_targets(tg_sigma, table))):
        res = {"band": regression(band, tg, seeds, LABEL["band"]),
               "band_position": regression(np.c_[band, para, band * para], tg, seeds,
                                           LABEL["band_position"])}
        for key, _label in ROWS:
            if key in kernels:
                t0 = time.time()
                res[key] = nested(kernels[key], tg, seeds, LABEL[key])
                q2, sd, _e = B.repeated_q2(res[key][0], sigma)
                log(f"  whole {unit} held out, {LABEL[key]}: q2 {q2:+.3f} (sd {sd:.3f}), "
                    f"{len(kernels[key])} settings, {time.time() - t0:.0f} s")
        rows = {k: row_of(r, c, tg, table) for k, (r, c) in res.items()}
        diffs = [{"row": a, "reference": b, "difference": compare(res[a][0], res[b][0], tg)}
                 for a, b in COMPARE]
        out["units"][unit] = {"rows": rows, "differences": diffs}
        picked[unit] = {k: most_chosen(c) for k, (r, c) in res.items() if c}
        for k in ("band", "band_position"):
            log(f"  whole {unit} held out, {LABEL[k]}: q2 {rows[k]['q2']:+.3f}")

    # ---- how alike the kernels are, and how far each follows the target (no fit)
    key_s, key_k = picked["substituents"]["spectrum"], picked["substituents"]["structure"]
    key_q = picked["substituents"]["structure_q"]
    full = {"spectrum": kernels["spectrum"][key_s], "structure": kernels["structure"][key_k],
            "structure_q": kernels["structure_q"][key_q], "counts": kernels["counts"]["counts"],
            "carbonyl": kernels["carbonyl"][picked["substituents"]["carbonyl"]],
            "rest": kernels["rest"][picked["substituents"]["rest"]]}
    out["picked"] = picked
    out["alignment"] = {k: target_alignment(K, sigma) for k, K in full.items()}
    out["alignment_by_setting"] = {k: target_alignment(K, sigma)
                                   for k, K in kernels["spectrum"].items()}
    out["agreement"] = {k: B.kernel_agreement(full["spectrum"], K)
                        for k, K in full.items() if k != "spectrum"}

    # ---- the co-spectrum of the model on all acids
    out["cospectrum"] = cospectrum(spectra, sigma, key_s, tg_sigma, table, scrambles, results)
    out["seconds"] = round(time.time() - t_all, 1)
    with open(os.path.join(results, "ir.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def cospectrum_again(results: str = RESULTS, scrambles: int = SCRAMBLES,
                     data: str = B.DATA, structures: str = B.STRUCTURES,
                     spectra_path: str = SPECTRA) -> dict:
    """The co-spectrum of a finished run, computed again with the setting that the run chose,
    and written back into its result file.  Nothing else in the file is touched."""
    path = os.path.join(results, "ir.json")
    with open(path, encoding="utf-8") as fh:
        out = json.load(fh)
    _mols, names, _fields, targets, _fs, table = B.read(data, structures=structures)
    spectra = read_spectra(spectra_path, names)
    spectra = [spectra[int(m)] for m in targets.molecule]
    sigma = np.asarray(table["sigma"], float)
    out["cospectrum"] = cospectrum(spectra, sigma, out["picked"]["substituents"]["spectrum"],
                                   with_target(targets, sigma, "Hammett constant"), table,
                                   scrambles, results)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def region_of(x: float) -> str:
    for lo, hi, label in REGIONS:
        if lo <= x <= hi:
            return label
    return "between the named regions"


def cospectrum(spectra, y: np.ndarray, setting: str, tg: AtomTargets, table: dict,
               scrambles: int, results: str) -> dict:
    """The weight over the wavenumber of the model fitted on all spectra with the setting
    that the folds chose most often, and the same reading on targets in random order."""
    width, power = setting_values(setting)
    grid = SP.Grid.for_width(RANGE[0], RANGE[1], width)
    F = SP.feature_matrix(spectra, grid, width, power)
    K = SP.spectrum_kernel(F)
    lam = B.choose_lam(K, B.substituent_targets(tg, table), lams=LAMS, folds=FOLDS, seed=0,
                       name="spectrum").lam

    x = grid.x
    norm = np.sqrt(np.maximum(np.einsum("ij,ij->i", F, F), 1e-300))
    Z = F / norm[:, None]
    masks = [(x >= lo) & (x <= hi) for lo, hi, _label in REGIONS]

    def effect_of(target):
        a, _b = krr_fit(K, target, lam)
        w = SP.co_spectrum(F, a)
        return w, SP.co_spectrum_effect(F, w)

    folds = B.substituent_targets(tg, table).folds(FOLDS, 0)

    def held_out(target):
        """What every region adds to the prediction of an acid, from the model fitted
        without the substituent of that acid; one row per region."""
        added = np.zeros((len(masks), len(target)))
        for te in folds:
            tr = np.setdiff1d(np.arange(len(target)), te)
            a, _b = krr_fit(K[np.ix_(tr, tr)], target[tr], lam)
            w_fold = Z[tr].T @ a
            for k, cols in enumerate(masks):
                added[k, te] = Z[np.ix_(te, cols)] @ w_fold[cols]
        return added

    def correlations(added, target):
        return [float(np.corrcoef(row, target)[0, 1]) if np.std(row) > 0 else 0.0
                for row in added]

    w, effect = effect_of(y)
    rng = np.random.default_rng(0)
    null = np.array([np.abs(effect_of(rng.permutation(y))[1]).max()
                     for _ in range(int(scrambles))])
    seen = correlations(held_out(y), y)
    null_r = []
    for _ in range(int(scrambles)):
        shuffled = rng.permutation(y)
        null_r.append(correlations(held_out(shuffled), shuffled))
    null_r = np.array(null_r)
    regions = []
    for k, (lo, hi, label) in enumerate(REGIONS):
        regions.append({"from": lo, "to": hi, "what": label,
                        "sd": float((Z[:, masks[k]] @ w[masks[k]]).std()),
                        "r_held_out": seen[k],
                        "null_mean": float(null_r[:, k].mean()),
                        "null_max": float(null_r[:, k].max())})
    strongest = [dict(r, what=region_of(r["x"])) for r in
                 SP.strongest_regions(x, effect, n=10, separation=4.0 * width)]
    keep = max(1, len(x) // 1500)
    rec = {"setting": setting, "width": width, "power": power, "lam": float(lam),
           "largest_effect": float(np.abs(effect).max()),
           "null_largest_effect": [float(v) for v in null],
           "above_null": int(np.sum(np.abs(effect) > null.max())),
           "grid_points": int(len(x)),
           "regions": regions, "strongest": strongest,
           "x": [float(v) for v in x[::keep]], "effect": [float(v) for v in effect[::keep]],
           "mean_spectrum": [float(v) for v in Z.mean(axis=0)[::keep]]}
    lo, hi = CARBONYL
    zoom = (x >= lo) & (x <= hi)
    rec["zoom"] = {"x": [float(v) for v in x[zoom]], "effect": [float(v) for v in effect[zoom]],
                   "mean_spectrum": [float(v) for v in Z.mean(axis=0)[zoom]]}
    figure(rec, results)
    return rec


def figure(rec: dict, results: str = RESULTS) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    null = float(np.max(rec["null_largest_effect"])) if rec["null_largest_effect"] else 0.0
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0))
    x = np.array(rec["x"])
    axes[0].plot(x, rec["mean_spectrum"], color="0.25", lw=0.7)
    axes[0].set_ylabel("mean normalised\nintensity")
    axes[0].set_title("Infrared spectra of the benzoic acids against the Hammett constant:\n"
                      "the mean spectrum, the co-spectrum, and the region of the C=O stretch",
                      fontsize=10)
    axes[1].plot(x, rec["effect"], color="tab:red", lw=0.7)
    axes[1].set_ylabel("weight x spread\nof the intensity")
    axes[1].set_xlabel("wavenumber, cm$^{-1}$")
    zx = np.array(rec["zoom"]["x"])
    axes[2].plot(zx, rec["zoom"]["effect"], color="tab:red", lw=0.9)
    scale = max(np.abs(rec["zoom"]["effect"]).max(), 1e-12) / max(
        np.max(rec["zoom"]["mean_spectrum"]), 1e-12)
    axes[2].plot(zx, np.array(rec["zoom"]["mean_spectrum"]) * scale, color="0.6", lw=0.7,
                 ls=":")
    axes[2].set_ylabel("weight x spread,\nregion of the C=O stretch")
    axes[2].set_xlabel("wavenumber, cm$^{-1}$ (dotted: the mean spectrum, scaled)")
    for ax in axes[1:]:
        ax.axhline(0.0, color="0.6", lw=0.6)
        for sign in (-1.0, 1.0):
            ax.axhline(sign * null, color="0.5", lw=0.6, ls="--")
    for ax in axes[:2]:
        ax.set_xlim(RANGE[1], RANGE[0])            # the convention of infrared spectra
    axes[2].set_xlim(CARBONYL[1], CARBONYL[0])
    fig.tight_layout()
    fig.savefig(os.path.join(results, "cospectrum.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def _iv(d: Sequence[float]) -> str:
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"


def _side(d: Sequence[float]) -> str:
    return "not separated" if d[1] <= 0.0 <= d[2] else ("above" if d[1] > 0 else "below")


def _chosen(row: dict) -> str:
    c = row.get("chosen") or {}
    if not c:
        return ""
    total = sum(c.values())
    top = sorted(c.items(), key=lambda kv: -kv[1])[:2]
    return "; ".join(f"{k} ({100.0 * v / total:.0f} %)" for k, v in top)


def _twin(v) -> str:
    return "" if v is None else f"{v:+.2f}"


def report(results: str = RESULTS) -> str:
    """The table of the benchmark as text; it is also written to ``table.md``."""
    path = os.path.join(results, "ir.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    units = (("acids", "whole acids held out"),
             ("substituents", "whole substituents held out (both acids of a substituent "
                              "together)"))
    lines = ["# A spectrum as the sample where the spectrum should hold the answer: infrared "
             "spectra against the Hammett constant (plan item SP1)", "",
             f"The {d['n']} substituted benzoic acids of plan item SP2 "
             f"({d['n_substituents']} substituents, meta and para).  The sample is the "
             "infrared spectrum of the acid from its ORCA frequency job: one stick for every "
             "normal mode, at the scaled harmonic wavenumber, with the infrared intensity "
             f"({d['lines']['median']} sticks between {d['range'][0]:g} and "
             f"{d['range'][1]:g} cm^-1 at the median, {d['lines']['min']} to "
             f"{d['lines']['max']}).  The target is the Hammett constant of the substituent at "
             f"its position ({d['sigma']['min']:+.2f} to {d['sigma']['max']:+.2f}, standard "
             f"deviation {d['sigma']['sd']:.2f}), which is a measured number.  The spectrum is "
             "computed, and the calculation is not told the target.", "",
             "The kernel of two spectra is the overlap of the two after every stick is "
             "replaced by a Gaussian (`cmfqsar/spectrum.py`), divided by the norms of the "
             "two.  The width of the Gaussian (one of "
             + ", ".join(f"{w:g}" for w in d["widths"]) + " cm^-1), the power of the "
             "intensity (1, or 0.5 for the square root) and the ridge are chosen on 5 folds "
             "inside every training part.  The cross-validation has "
             f"{d['folds']} folds and is repeated {d['repeats']} times with other folds.  A "
             "difference in q2 carries its 95 % bootstrap interval over the units that the "
             "folds hold out.", "",
             "The series holds each substituent twice, meta and para.  A spectrum shows the "
             "bands of the substituent itself, so with whole acids held out a model can "
             "recognise the substituent from the other acid.  The numbers with whole "
             "substituents held out are the ones that the reading uses.", ""]
    # ---- the control
    lines += ["## Control: the length of the C=O bond from the spectrum", "",
              "The length of the C=O bond of the optimised geometry "
              f"({100 * d['length']['min']:.2f} to {100 * d['length']['max']:.2f} pm, "
              f"standard deviation {100 * d['length']['sd']:.3f} pm) follows the position of "
              f"the carbonyl band (r {d['r_band_length']:+.3f}; Badger's rule).  The position "
              "of one line of the spectrum therefore holds the answer, and a kernel that can "
              "read the position of a line has to find it.", "",
              "| description | q2, whole acids held out | q2, whole substituents held out | "
              "settings chosen most often, whole substituents held out |",
              "|---|---:|---:|---|"]
    for key, label in CONTROL_ROWS:
        a, s = d["control"]["acids"][key], d["control"]["substituents"][key]
        lines.append(f"| {label} | {a['q2']:+.3f} ({a['sd']:.3f}) | {s['q2']:+.3f} "
                     f"({s['sd']:.3f}) | {_chosen(s)} |")
    lines.append("")
    # ---- no fit
    al = d["alignment"]
    lines += ["## Before any fit: how far each kernel follows the target", "",
              "The correlation between the centred kernel and the products of the centred "
              "Hammett constants, over the pairs of acids, at the setting chosen most often "
              "with whole substituents held out.  The first measurement of this item found "
              "that this number, which needs no fit, says in advance what the fit finds.", "",
              "| description | setting | correlation with the target | correlation with the "
              "kernel of the whole spectrum |", "|---|---|---:|---:|"]
    names = {"counts": LABEL["counts"], "structure": LABEL["structure"],
             "structure_q": LABEL["structure_q"], "spectrum": LABEL["spectrum"],
             "carbonyl": LABEL["carbonyl"], "rest": LABEL["rest"]}
    for key, label in names.items():
        setting = d["picked"]["substituents"].get(key, "")
        other = "1" if key == "spectrum" else f"{d['agreement'][key]:+.3f}"
        lines.append(f"| {label} | {setting if key != 'counts' else ''} | {al[key]:+.3f} | "
                     f"{other} |")
    lines.append("")
    # ---- the errors
    lines += ["## The Hammett constant from each description", "",
              f"The band of the carbonyl stretch correlates with the Hammett constant with r "
              f"{d['r_band_sigma']:+.3f}.  The last column is the q2 of the predicted "
              "difference between the para acid and the meta acid of one substituent, which "
              "is the part of the constant that resonance decides.", ""]
    for unit, title in units:
        rows = d["units"][unit]["rows"]
        lines += [f"### The Hammett constant, {title}", "",
                  "| description | q2 (sd over the repeats) | RMSE | settings chosen most "
                  "often | q2 of the difference between the para and the meta acid |",
                  "|---|---:|---:|---|---:|"]
        for key, label in ROWS:
            r = rows[key]
            lines.append(f"| {label} | {r['q2']:+.3f} ({r['sd']:.3f}) | {r['rmse']:.3f} | "
                         f"{_chosen(r)} | {_twin(r['twin_q2'])} |")
        lines += ["", "| description | compared with | difference in q2 | reading |",
                  "|---|---|---:|---|"]
        for c in d["units"][unit]["differences"]:
            lines.append(f"| {LABEL[c['row']]} | {LABEL[c['reference']]} | "
                         f"{_iv(c['difference'])} | {_side(c['difference'])} |")
        lines.append("")
    # ---- the co-spectrum
    co = d["cospectrum"]
    null = np.array(co["null_largest_effect"])
    lines += ["## The co-spectrum", "",
              "The model on all acids with the setting chosen most often with whole "
              f"substituents held out ({co['setting']}; ridge {co['lam']:g}).  Its prediction "
              "is a constant plus the sum over the axis of a weight times the normalised "
              "intensity.  The effect of a grid point is its weight times the standard "
              "deviation of the intensity there over the acids (`cospectrum.png`).", "",
              f"The largest effect is {co['largest_effect']:.4f}.  In {len(null)} models "
              f"fitted to the constants in random order the largest effect is "
              f"{null.mean():.4f} at the mean and {null.max():.4f} at most.  "
              f"{co['above_null']} of the {co['grid_points']} grid points are above that "
              "largest value.", "",
              "The ten places with the largest effect:", "",
              "| wavenumber, cm^-1 | effect | region |", "|---:|---:|---|"]
    for r in co["strongest"]:
        lines.append(f"| {r['x']:.0f} | {r['value']:+.4f} | {r['what']} |")
    lines += ["", "What every region adds to a prediction is the sum of weight times "
              "intensity inside the region.  The first number column is its standard "
              "deviation over the acids, for the model on all acids.  The next is read out of "
              "fold: the model is fitted without the substituent of an acid, the sum of the "
              "region is computed for that acid, and the sums of all acids are correlated "
              "with the Hammett constant.  The last columns give the same correlation for "
              f"{len(null)} sets of constants in random order.", "",
              "| region | from, cm^-1 | to, cm^-1 | standard deviation of what the region "
              "adds | held-out correlation of what the region adds with the Hammett constant "
              "| the same for constants in random order, mean (largest) | above the largest |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for r in co["regions"]:
        lines.append(f"| {r['what']} | {r['from']:g} | {r['to']:g} | {r['sd']:.3f} | "
                     f"{r['r_held_out']:+.2f} | {r['null_mean']:+.2f} ({r['null_max']:+.2f}) | "
                     f"{'yes' if r['r_held_out'] > r['null_max'] else 'no'} |")
    lines.append("")
    lines += _verdict(d)
    text = "\n".join(lines)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def _verdict(d: dict) -> List[str]:
    """The reading of the numbers, written from the numbers."""
    acids, subs = d["units"]["acids"], d["units"]["substituents"]

    def q(unit, key):
        return unit["rows"][key]["q2"]

    def diff(unit, a, b):
        return next(c["difference"] for c in unit["differences"]
                    if c["row"] == a and c["reference"] == b)

    def said(unit, a, b):
        c = diff(unit, a, b)
        return f"{_iv(c)} ({_side(c)})"

    ctl = d["control"]["substituents"]
    out = ["## Reading", "",
           f"**The control.** With whole substituents held out the spectrum predicts the "
           f"length of the C=O bond with q2 {ctl['spectrum']['q2']:+.3f}, and the regression "
           f"on the position of the carbonyl band with {ctl['band']['q2']:+.3f} (the spectrum "
           f"against the regression: {_iv(ctl['spectrum_against_band'])}).  The region of the "
           f"carbonyl stretch alone gives {ctl['carbonyl']['q2']:+.3f}, and the spectrum "
           f"without that region {ctl['rest']['q2']:+.3f}.  So the kernel reads the position "
           "of a line, but less exactly than a regression on the position itself: the "
           "lengths differ by a tenth of a picometre, the band moves by a few cm^-1, and the "
           "kernel sees such a shift only as a small change of the overlap of two broadened "
           "lines.", "",
           f"**The Hammett constant from the spectrum.** With whole substituents held out the "
           f"spectrum reaches q2 {q(subs, 'spectrum'):+.3f}.  The atom-centred counts reach "
           f"{q(subs, 'counts'):+.3f}, the regression on the position of the carbonyl band "
           f"{q(subs, 'band'):+.3f}, the five fields {q(subs, 'structure'):+.3f} and the "
           f"electrostatic field with quantum charges {q(subs, 'structure_q'):+.3f}.  The "
           f"spectrum is {said(subs, 'spectrum', 'band')} against the regression on its one "
           f"band, {said(subs, 'spectrum', 'counts')} against the counts, "
           f"{said(subs, 'spectrum', 'structure')} against the five fields and "
           f"{said(subs, 'spectrum', 'structure_q')} against the field with quantum charges.  "
           f"With a Gaussian kernel on top it reaches {q(subs, 'spectrum_gauss'):+.3f} "
           f"({said(subs, 'spectrum_gauss', 'spectrum')} against the spectrum without it).",
           "",
           f"**Where the information is.** The region of the carbonyl stretch alone reaches "
           f"{q(subs, 'carbonyl'):+.3f} ({said(subs, 'carbonyl', 'band')} against the "
           f"regression on the position of the band).  The spectrum without that region "
           f"reaches {q(subs, 'rest'):+.3f} ({said(subs, 'rest', 'spectrum')} against the "
           "whole spectrum).", "",
           f"**Whole acids held out.** There the spectrum reaches {q(acids, 'spectrum'):+.3f} "
           f"and the spectrum without the carbonyl region {q(acids, 'rest'):+.3f}; the five "
           f"fields reach {q(acids, 'structure'):+.3f}.  The difference from the numbers "
           "above is what the model gains when it has seen the other acid of the same "
           "substituent.", "",
           f"**The spectrum beside the structure.** Joined with the five fields the q2 is "
           f"{q(subs, 'structure+spectrum'):+.3f} with whole substituents held out: "
           f"{said(subs, 'structure+spectrum', 'structure')} against the five fields alone "
           f"and {said(subs, 'structure+spectrum', 'spectrum')} against the spectrum alone.  "
           f"Joined with the field with quantum charges it is "
           f"{q(subs, 'structure_q+spectrum'):+.3f} "
           f"({said(subs, 'structure_q+spectrum', 'structure_q')} against that field alone).",
           ""]
    co = d["cospectrum"]
    null = max(co["null_largest_effect"])
    above = [f"{r['what']} ({r['r_held_out']:+.2f})" for r in co["regions"]
             if r["r_held_out"] > r["null_max"]]
    below = [r["what"] for r in co["regions"] if r["r_held_out"] <= r["null_max"]]
    first = co["strongest"][0]
    out += [f"**The co-spectrum.** The place with the largest effect is at "
            f"{first['x']:.0f} cm^-1 ({first['what']}), with the sign "
            f"{'plus' if first['value'] > 0 else 'minus'}: intensity at the upper side of the "
            "carbonyl band raises the predicted constant, as the band itself rises with the "
            "constant.  No single grid point is above the largest effect in models fitted "
            f"to the constants in random order ({co['largest_effect']:.3f} against "
            f"{null:.3f}), so the co-spectrum cannot be read point by point.  It can be read "
            "by region.  With the substituent of an acid held out, what a region adds to "
            "the prediction correlates with the Hammett constant above every model on random "
            "constants in these regions: " + (", ".join(above) if above else "none") + "."
            + (f"  It does not in these: {', '.join(below)}." if below else ""), ""]
    gate_structure = diff(subs, "structure+spectrum", "structure")[1] > 0
    gate_quantum = diff(subs, "structure_q+spectrum", "structure_q")[1] > 0
    reaches_counts = diff(subs, "spectrum", "counts")[2] >= 0
    out += ["**The gate of the item.** The item is a result when the structural kernel with "
            "the spectrum is above the structural kernel alone by more than the interval, or "
            "when the spectrum alone reaches the descriptor that needs no structure in three "
            "dimensions (here the atom-centred counts).  With whole substituents held out "
            f"the first condition is {'met' if gate_structure else 'not met'} for the five "
            f"fields and {'met' if gate_quantum else 'not met'} for the field with quantum "
            f"charges, and the second condition is {'met' if reaches_counts else 'not met'}.",
            "",
            "**What the result does and does not say.** The spectrum of this benchmark is "
            "computed with a quantum program on a geometry that the program optimised.  The "
            "five fields are built on a geometry of a force field with charges from the bond "
            "graph.  The gain of the spectrum over the five fields is therefore what a "
            "quantum calculation knows about the acid, and the field with charges from one "
            "fast quantum calculation holds the same.  What the result does say is that the "
            "kernel of `cmfqsar/spectrum.py` takes a property out of a spectrum that holds "
            "it: more than a regression on the one band that a chemist would read, and as "
            "much as the best structural description of this series.  A measured spectrum "
            "would be the experiment itself and not a calculation, and no measured spectrum "
            "is used here.", ""]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--cospectrum", action="store_true",
                    help="compute only the co-spectrum again, with the setting of the "
                         "finished run")
    ap.add_argument("--report", action="store_true", help="only write the table")
    a = ap.parse_args(argv)
    if a.cospectrum:
        cospectrum_again()
    elif not a.report:
        run(repeats=a.repeats)
    text = report()
    if not text:
        print("no results yet: run the script without --report first")
        return 1
    print(f"Report written to {os.path.join(RESULTS, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
