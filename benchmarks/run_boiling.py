#!/usr/bin/env python
"""
Measured normal boiling points with the rotation-invariant field kernel, against the references
a boiling point model has to beat. The design is that of ``benchmarks/run_logp.py`` and
``benchmarks/run_solubility.py``; the fitting, the selection and the kernels are imported from
the first.

The sets are made by ``benchmarks/prepare_boiling.py`` from the data of the OPERA models
(Mansouri et al., J. Cheminform. 10 (2018) 10; MIT licence): ``examples/boiling/physprop.sdf``
(the PHYSPROP boiling points curated for OPERA, with the paper's training and test parts) and
``examples/boiling/echemportal.sdf`` (boiling points from registration dossiers that OPERA 2.9
added, none of them in PHYSPROP). The target is the boiling point in degrees C. One ETKDG +
MMFF94 conformer per compound.

The question, fixed before the measurement
------------------------------------------
A boiling point is a property of the liquid. It grows with the size of the molecule (the
dispersion forces) and with its polar groups, above all the hydrogen-bond donors. Unlike a
melting point or a solubility, no crystal is involved, so a description of one molecule has all
the information except the shape of the liquid around it. Group contributions are the classic
method: Joback's equation gives the boiling point in K as 198.2 plus one increment for each of
41 groups. The main comparison is

    invariant kernel (as the program builds it) minus Joback's groups refitted in each fold,

with the difference of q2 and a 95 % paired bootstrap interval over the compounds, on each
test. The cosine normalisation of the invariant kernel removes the size of a molecule, which is
the first thing a boiling point depends on, so the kernel without that normalisation and the
kernel plus the group counts are measured too.

The models
----------
* **invariant, cosine** and **invariant, not normalised** -- as in the logP benchmark.
* **invariant + Joback groups** -- the cosine kernel plus h times the linear kernel on the group
  counts, the width taken from the cosine model in each fold.
* **linear regression on molecular weight** -- the one-number reference.
* **Joback refitted** -- ridge regression on the counts of the 41 Joback groups and the number of
  heavy atoms that no group matches, fitted in each fold.
* **field totals**, **eight counts**, **ECFP4**, **refitted Crippen increments** and
  **topological autocorrelation** -- as in the logP benchmark.
* **topological + Joback groups** -- the topological kernel plus h times the group kernel, with
  the maximum distance the topological model chose in each fold.

The **published Joback equation**, with its own increments and no fitting, is reported for the
compounds whose heavy atoms each fall into exactly one group. The groups are matched with the
SMARTS patterns of J. Biggs as the thermo library gives them (C. Bell, github.com/CalebBell/
thermo, MIT licence), with the boiling point increments of K. G. Joback and R. C. Reid, Chem.
Eng. Commun. 57 (1987) 233-243. thermo resolves an atom matched by two groups by keeping the
larger match; here such a compound counts as not covered.
* **invariant + topological** and **invariant + eight counts** (added 17 November 2026) -- the
  cosine kernel plus h times the topological kernel, or plus h times the Gaussian kernel on the
  eight counts. In each block the width, the largest distance and the count width are those the
  single models chose there by the evidence on the same training block; only h and the ridge
  are chosen here.

How the numbers are made
------------------------
As in the logP benchmark: kernel ridge regression with a bias; the setting and the ridge chosen
by the log evidence on each training block; five folds repeated three times on the training
part of PHYSPROP and on the eChemPortal set. Two external tests fit on the whole training part of
PHYSPROP: one predicts the test part of PHYSPROP, which is the test of the OPERA paper (R2 0.93,
RMSE 22.08 C there, with 13 descriptors); the other predicts the eChemPortal set.

Usage
-----
    python benchmarks/run_boiling.py               # everything; the features are cached
    python benchmarks/run_boiling.py --quick 300   # a smoke test on 300 compounds per part
    python benchmarks/run_boiling.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

from cmfqsar.descriptors import fingerprint_kernel  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, feature_matrix  # noqa: E402
from cmfqsar.model import r2_score, rmse  # noqa: E402

import run_logp as L  # noqa: E402
from run_logp import (COUNT_WIDTHS, D_MAX, FIELDS, HS, LAMS, REPEATS, SIGMAS,  # noqa: E402
                      TOPO_MAX, add_pairs, crippen_counts, eight_counts, gaussian_at,
                      gaussian_stack, invariant_kernels,
                      jdump, linear_block, log, selfcheck, standardise, topological_features,
                      topological_kernel)
import run_resolution as RS  # noqa: E402
import run_wavelet as RW  # noqa: E402

DATA = os.path.join(ROOT, "examples", "boiling")
RESULTS = os.path.join(HERE, "results", "boiling")
CACHE = os.path.join(HERE, "cache", "boiling")
#: the parts of the data: the two PHYSPROP parts of the OPERA paper and the eChemPortal set
PARTS = ("physprop_train", "physprop_test", "echemportal")
#: the parts with cross-validation
SETS = ("physprop_train", "echemportal")
#: the external tests: name -> (the part fitted, the part predicted)
EXTERNALS = {"opera_test": ("physprop_train", "physprop_test"),
             "to_echemportal": ("physprop_train", "echemportal")}
TESTS = SETS + tuple(EXTERNALS)
TITLES = {"physprop_train": "PHYSPROP training part, CV", "echemportal": "eChemPortal, CV",
          "opera_test": "PHYSPROP test part (the OPERA test)",
          "to_echemportal": "PHYSPROP training part to eChemPortal"}
#: the models in the order of the table; the json stores them with sorted keys
MODELS = ("invariant, cosine", "invariant, not normalised", "invariant + Joback groups",
          "linear regression on molecular weight", "Joback refitted", "field totals",
          "eight counts", "ECFP4", "refitted Crippen increments", "topological autocorrelation",
          "topological + Joback groups", "invariant + topological",
          "invariant + eight counts")

#: Joback's 41 groups: the name, the SMARTS of J. Biggs and the boiling point increment in K,
#: as the thermo library gives them (thermo/group_contribution/joback.py, MIT licence,
#: Copyright (C) 2017-2020 Caleb Bell)
JOBACK = [
    ("-CH3", "[CX4H3]", 23.58),
    ("-CH2-", "[!R;CX4H2]", 22.88),
    (">CH-", "[!R;CX4H]", 21.74),
    (">C<", "[!R;CX4H0]", 18.25),
    ("=CH2", "[CX3H2]", 18.18),
    ("=CH-", "[!R;CX3H1;!$([CX3H1](=O))]", 24.96),
    ("=C<", "[$([!R;#6X3H0]);!$([!R;#6X3H0]=[#8])]", 24.14),
    ("=C=", "[$([CX2H0](=*)=*)]", 26.15),
    ("#CH", "[$([CX2H1]#[!#7])]", 9.2),
    ("#C-", "[$([CX2H0]#[!#7])]", 27.38),
    ("-CH2- (ring)", "[R;CX4H2]", 27.15),
    (">CH- (ring)", "[R;CX4H]", 21.78),
    (">C< (ring)", "[R;CX4H0]", 21.32),
    ("=CH- (ring)", "[R;CX3H1,cX3H1]", 26.73),
    ("=C< (ring)", "[$([R;#6X3H0]);!$([R;#6X3H0]=[#8])]", 31.01),
    ("-F", "[F]", -0.03),
    ("-Cl", "[Cl]", 38.13),
    ("-Br", "[Br]", 66.86),
    ("-I", "[I]", 93.84),
    ("-OH (alcohol)", "[OX2H;!$([OX2H]-[#6]=[O]);!$([OX2H]-a)]", 92.88),
    ("-OH (phenol)", "[$([OX2H]-a)]", 76.34),
    ("-O- (nonring)", "[OX2H0;!R;!$([OX2H0]-[#6]=[#8])]", 22.42),
    ("-O- (ring)", "[#8X2H0;R;!$([#8X2H0]~[#6]=[#8])]", 31.22),
    (">C=O (nonring)", "[$([CX3H0](=[OX1]));!$([CX3](=[OX1])-[OX2]);!R]=O", 76.75),
    (">C=O (ring)", "[$([#6X3H0](=[OX1]));!$([#6X3](=[#8X1])~[#8X2]);R]=O", 94.97),
    ("O=CH- (aldehyde)", "[CH;D2;$(C-!@C)](=O)", 72.24),
    ("-COOH (acid)", "[OX2H]-[C]=O", 169.09),
    ("-COO- (ester)", "[#6X3H0;!$([#6X3H0](~O)(~O)(~O))](=[#8X1])[#8X2H0]", 81.1),
    ("=O (other than above)", "[OX1H0;!$([OX1H0]~[#6X3]);!$([OX1H0]~[#7X3]~[#8])]", -10.5),
    ("-NH2", "[NX3H2]", 73.23),
    (">NH (nonring)", "[NX3H1;!R]", 50.17),
    (">NH (ring)", "[#7X3H1;R]", 52.82),
    (">N- (nonring)", "[#7X3H0;!$([#7](~O)~O)]", 11.74),
    ("-N= (nonring)", "[#7X2H0;!R]", 74.6),
    ("-N= (ring)", "[#7X2H0;R]", 57.55),
    ("=NH", "[#7X2H1]", 83.08),
    ("-CN", "[#6X2]#[#7X1H0]", 125.66),
    ("-NO2", "[$([#7X3,#7X3+][!#8])](=[O])~[O-]", 152.54),
    ("-SH", "[SX2H]", 63.56),
    ("-S- (nonring)", "[#16X2H0;!R]", 68.78),
    ("-S- (ring)", "[#16X2H0;R]", 52.1),
]


class Model(L.Model):
    """The model of the logP benchmark, with the parts and the two external tests of this one."""

    def select(self, y: np.ndarray, blocks):
        n = len(y)
        cv = {name: np.zeros((REPEATS, n)) for name in SETS}
        external = {}
        chosen: Dict[str, Dict[str, int]] = {name: {} for name in TESTS}
        for k, (name, rep, _, te) in enumerate(blocks):
            ev = np.array([c["ev"][k] for c in self.cands])
            c, j = np.unravel_index(int(np.nanargmax(ev)), ev.shape)
            p = self.cands[c]["pred"][k][j]
            key = f"{self.labels[c]}, ridge {LAMS[j]:g}"
            chosen[name][key] = chosen[name].get(key, 0) + 1
            if name in EXTERNALS:
                external[name] = p
            else:
                cv[name][rep, te] = p
        return cv, external, chosen


def splits(idx: Dict[str, np.ndarray]):
    out = []
    for name in SETS:
        ii = idx[name]
        for rep in range(REPEATS):
            for te in RW.folds(len(ii), L.FOLDS, L.SEED + rep):
                tr = np.setdiff1d(np.arange(len(ii)), te)
                out.append((name, rep, ii[tr], ii[te]))
    for name, (fit, pred) in EXTERNALS.items():
        out.append((name, 0, idx[fit], idx[pred]))
    return out


def load(quick: int = 0):
    mols, y, tag = [], [], []
    for fname in ("physprop", "echemportal"):
        sup = Chem.SDMolSupplier(os.path.join(DATA, f"{fname}.sdf"), removeHs=False)
        block = [m for m in sup if m is not None]
        parts = ([f"physprop_{m.GetProp('split')}" for m in block] if fname == "physprop"
                 else ["echemportal"] * len(block))
        if quick:
            keep = []
            for part in sorted(set(parts)):
                ii = [i for i, p in enumerate(parts) if p == part]
                keep += [ii[j] for j in np.linspace(0, len(ii) - 1,
                                                    min(quick, len(ii))).astype(int)]
            keep = sorted(keep)
            block = [block[i] for i in keep]
            parts = [parts[i] for i in keep]
        mols += block
        y += [float(m.GetProp("bp")) for m in block]
        tag += parts
    tag = np.array(tag)
    idx = {name: np.where(tag == name)[0] for name in PARTS}
    return mols, np.array(y), idx


def features(mfs, fs, sigma: float, names: np.ndarray, use_cache: bool) -> np.ndarray:
    path = os.path.join(CACHE, f"features_{sigma:g}_{len(names)}.npz")
    if use_cache and os.path.isfile(path):
        z = np.load(path)
        if z["names"].shape == names.shape and np.all(z["names"] == names):
            return z["F"].astype(float)
    F = feature_matrix(mfs, AutocorrelationSpec(d_max=D_MAX, step=sigma, sigma=sigma),
                       fieldset=fs)
    if use_cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, F=F.astype(np.float32), names=names)
    return F


def joback(heavy):
    """Joback group counts, the heavy atoms in no group, and the published boiling point in C.

    Returns the count matrix (one column per group, the last one the atoms that no group
    matches) and the published estimate, which is NaN for a compound whose heavy atoms are not
    each matched by exactly one group.
    """
    pats = [Chem.MolFromSmarts(s) for _, s, _ in JOBACK]
    inc = np.array([t for _, _, t in JOBACK])
    C = np.zeros((len(heavy), len(JOBACK) + 1))
    published = np.full(len(heavy), np.nan)
    for i, m in enumerate(heavy):
        used: Counter = Counter()
        for j, p in enumerate(pats):
            hits = m.GetSubstructMatches(p)
            C[i, j] = len(hits)
            for h in hits:
                used.update(h)
        C[i, -1] = m.GetNumAtoms() - len(used)
        if C[i, -1] == 0 and max(used.values()) == 1:
            published[i] = 198.2 + float(C[i, :-1] @ inc) - 273.15
    return C, published


def joback_selfcheck() -> None:
    """Acetone, the example of thermo's documentation: two -CH3, one >C=O, Tb 322.11 K."""
    C, pub = joback([Chem.MolFromSmiles("CC(=O)C")])
    names = [g for g, _, _ in JOBACK]
    assert C[0, names.index("-CH3")] == 2 and C[0, names.index(">C=O (nonring)")] == 1
    assert C[0].sum() == 3 and abs(pub[0] + 273.15 - 322.11) < 1e-9, (C[0], pub[0])
    log("Joback self-check: acetone gives 322.11 K, as in thermo")


def run(quick: int = 0) -> Dict[str, object]:
    t0 = time.time()
    selfcheck()
    joback_selfcheck()
    mols, y, idx = load(quick)
    heavy = [Chem.RemoveHs(m) for m in mols]
    names = np.array([m.GetProp("_Name") for m in mols])
    blocks = splits(idx)
    log(f"{len(mols)} compounds: " + ", ".join(f"{k} {len(v)}" for k, v in idx.items()))
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mfs = [build_fields(m, fs, str(n)) for m, n in zip(mols, names)]
    totals = np.array([mf.weights.sum(axis=0) for mf in mfs])
    mw = np.array([Descriptors.MolWt(m) for m in heavy])
    Cj, published = joback(heavy)
    log(f"fields built ({time.time() - t0:.0f} s); Joback covers "
        f"{np.isfinite(published).sum()} of {len(mols)} compounds")

    Kmw = linear_block(standardise(mw[:, None]))
    Kjob = linear_block(Cj)
    models = {k: Model(k) for k in MODELS}
    models["linear regression on molecular weight"].add("linear", Kmw, y, blocks)
    models["Joback refitted"].add("linear", Kjob, y, blocks)
    C, crippen_types = crippen_counts(mols)
    models["refitted Crippen increments"].add("linear", linear_block(C), y, blocks)
    log(f"{len(crippen_types)} Crippen atom types ({time.time() - t0:.0f} s)")

    n_f = len(fs.specs)
    n_pairs = n_f * (n_f + 1) // 2
    top = max(TOPO_MAX)
    Ftop = topological_features(mols, mfs, top).reshape(len(mols), n_pairs, top + 1)

    def topo(dm):
        return topological_kernel(Ftop[:, :, :dm + 1].reshape(len(mols), -1), dm, n_f)

    for dm in TOPO_MAX:
        models["topological autocorrelation"].add(f"up to {dm} bonds", topo(dm), y, blocks)
    reach = [models["topological autocorrelation"].best_label(k) for k in range(len(blocks))]
    for dm in TOPO_MAX:
        only = {k for k, w in enumerate(reach) if w == f"up to {dm} bonds"}
        if not only:
            continue
        Kt = topo(dm)
        for h in HS:
            models["topological + Joback groups"].add(f"up to {dm} bonds, h {h:g}",
                                                      Kt + h * Kjob, y, blocks, only)
    log(f"topological models done ({time.time() - t0:.0f} s)")
    for w, K in zip(COUNT_WIDTHS, gaussian_stack(standardise(totals[:, totals.std(axis=0) > 0]))):
        models["field totals"].add(f"width {w:g}", K, y, blocks)
    for w, K in zip(COUNT_WIDTHS, gaussian_stack(eight_counts(heavy))):
        models["eight counts"].add(f"width {w:g}", K, y, blocks)
    models["ECFP4"].add("Tanimoto", fingerprint_kernel(heavy), y, blocks)
    log(f"references done ({time.time() - t0:.0f} s)")
    for s in SIGMAS:
        Kc, Ku = invariant_kernels(features(mfs, fs, s, names, use_cache=not quick), s, n_f)
        models["invariant, cosine"].add(f"sigma {s:g}", Kc, y, blocks)
        models["invariant, not normalised"].add(f"sigma {s:g}", Ku, y, blocks)
        del Kc, Ku
        log(f"sigma {s:g} done ({time.time() - t0:.0f} s)")
    width = [models["invariant, cosine"].best_label(k) for k in range(len(blocks))]
    for s in SIGMAS:
        only = {k for k, w in enumerate(width) if w == f"sigma {s:g}"}
        if not only:
            continue
        Kc, _ = invariant_kernels(features(mfs, fs, s, names, use_cache=not quick), s, n_f)
        for h in HS:
            models["invariant + Joback groups"].add(f"sigma {s:g}, h {h:g}", Kc + h * Kjob,
                                                    y, blocks, only)
        log(f"combined model at sigma {s:g}, {len(only)} blocks ({time.time() - t0:.0f} s)")
    cwidth = [models["eight counts"].best_label(k) for k in range(len(blocks))]
    Z8 = eight_counts(heavy)
    add_pairs(models, y, blocks, width, reach, cwidth,
              lambda s: invariant_kernels(features(mfs, fs, s, names, use_cache=not quick), s,
                                          n_f)[0],
              topo, lambda w: gaussian_at(Z8, w))
    log(f"pair models done ({time.time() - t0:.0f} s)")

    res, preds = {}, {}
    for name in MODELS:
        cv, ext, chosen = models[name].select(y, blocks)
        r = {}
        for set_name in SETS:
            ii = idx[set_name]
            yy = y[ii]
            q2 = [float(r2_score(yy, cv[set_name][rep, ii])) for rep in range(REPEATS)]
            r[set_name] = {"q2_mean": float(np.mean(q2)), "q2_sd": float(np.std(q2)),
                           "rmse_mean": float(np.mean([rmse(yy, cv[set_name][rep, ii])
                                                       for rep in range(REPEATS)])),
                           "chosen": chosen[set_name]}
            preds[(name, set_name)] = cv[set_name][:, ii].mean(axis=0)
        for test, (_, part) in EXTERNALS.items():
            yy = y[idx[part]]
            r[test] = {"r2": float(r2_score(yy, ext[test])), "rmse": float(rmse(yy, ext[test])),
                       "chosen": chosen[test]}
            preds[(name, test)] = ext[test]
        res[name] = r
        log(f"  {name:38s} " + "  ".join(
            f"{t} {r[t]['q2_mean'] if t in SETS else r[t]['r2']:+.3f} "
            f"({r[t]['rmse_mean'] if t in SETS else r[t]['rmse']:.1f})" for t in TESTS))

    pub = {}
    for part in PARTS:
        ii = idx[part]
        ok = ii[np.isfinite(published[ii])]
        pub[part] = {"covered": int(len(ok)), "of": int(len(ii)),
                     "r2": float(r2_score(y[ok], published[ok])) if len(ok) > 2 else None,
                     "rmse": float(rmse(y[ok], published[ok])) if len(ok) > 2 else None}
        log(f"  published Joback on {part}: {len(ok)} of {len(ii)} covered, "
            f"R2 {pub[part]['r2']}, RMSE {pub[part]['rmse']}")

    pairs = [("invariant, cosine", "Joback refitted"),
             ("invariant + Joback groups", "Joback refitted"),
             ("invariant, cosine", "linear regression on molecular weight"),
             ("invariant, not normalised", "invariant, cosine"),
             ("invariant, cosine", "field totals"),
             ("invariant, cosine", "eight counts"),
             ("invariant, cosine", "ECFP4"),
             ("invariant, cosine", "refitted Crippen increments"),
             ("invariant, cosine", "topological autocorrelation"),
             ("invariant + Joback groups", "topological + Joback groups"),
             ("topological autocorrelation", "refitted Crippen increments"),
             ("invariant + topological", "invariant, cosine"),
             ("invariant + eight counts", "invariant, cosine"),
             ("invariant + eight counts", "eight counts"),
             ("invariant + Joback groups", "invariant + eight counts")]
    comps = []
    for a, b in pairs:
        for test in TESTS:
            part = EXTERNALS[test][1] if test in EXTERNALS else test
            g = RS.paired(y[idx[part]], preds[(a, test)], preds[(b, test)])
            comps.append({"model": a, "against": b, "test": test, **g})
            log(f"  {a} - {b} [{test}]: {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]")
    payload = {"compounds": {k: int(len(v)) for k, v in idx.items()}, "quick": quick,
               "y": {k: {"mean": float(y[v].mean()), "sd": float(y[v].std()),
                         "min": float(y[v].min()), "max": float(y[v].max())}
                     for k, v in idx.items()},
               "published_joback": pub, "joback_groups": len(JOBACK),
               "models": res, "comparisons": comps, "sigmas": list(SIGMAS), "hs": list(HS),
               "lams": list(LAMS), "count_widths": list(COUNT_WIDTHS), "d_max": D_MAX,
               "folds": L.FOLDS, "repeats": REPEATS, "topo_max": list(TOPO_MAX),
               "crippen_types": len(crippen_types), "seconds": round(time.time() - t0, 1)}
    jdump(payload, os.path.join(RESULTS, "quick.json" if quick else "boiling.json"))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, "predictions_quick.npz" if quick else "predictions.npz"),
             names=names, y=y, **{f"{m}|{t}": p for (m, t), p in preds.items()})
    if not quick:
        report(payload)
    return payload


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "boiling.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    n = payload["compounds"]
    sd = payload["y"]
    pub = payload["published_joback"]
    lines = [f"Measured normal boiling points, C: the PHYSPROP set curated for OPERA, its "
             f"training part ({n['physprop_train']} compounds, spread "
             f"{sd['physprop_train']['sd']:.1f}) and test part ({n['physprop_test']}), and "
             f"boiling points from eChemPortal ({n['echemportal']} compounds, spread "
             f"{sd['echemportal']['sd']:.1f}). Five folds repeated three times on the training "
             f"part and on the eChemPortal set; the external tests fit on the whole training "
             f"part. The setting and the ridge are chosen by the evidence on the training "
             f"block. Each cell gives q² (cross-validation) or R² (external test), with the "
             f"RMSE in C in brackets.", "",
             "| Model | " + " | ".join(TITLES[t] for t in TESTS) + " |",
             "|---|" + "---:|" * len(TESTS)]
    for name in MODELS:
        r = payload["models"][name]
        cells = [f"{r[t]['q2_mean']:+.3f} ({r[t]['rmse_mean']:.1f})" if t in SETS
                 else f"{r[t]['r2']:+.3f} ({r[t]['rmse']:.1f})" for t in TESTS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "| Published Joback equation, no fitting | Compounds covered | R² | RMSE, C |",
              "|---|---:|---:|---:|"]
    for part in ("physprop_train", "physprop_test", "echemportal"):
        p = pub[part]
        r2 = "--" if p["r2"] is None else f"{p['r2']:+.3f}"
        rm = "--" if p["rmse"] is None else f"{p['rmse']:.1f}"
        lines.append(f"| {part} | {p['covered']} of {p['of']} | {r2} | {rm} |")
    lines += ["", "| Difference of q² or R² | Test | Mean [95 % interval] |", "|---|---|---|"]
    for c in payload["comparisons"]:
        lines.append(f"| {c['model']} minus {c['against']} | {TITLES[c['test']]} | "
                     f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] |")
    text = "\n".join(lines) + "\n"
    # the table is written here, so that run_all.py --tables and --check can rebuild it
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=0,
                    help="use this many compounds of each part and do not touch the cache")
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    run(a.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
