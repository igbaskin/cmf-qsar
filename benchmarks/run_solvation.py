#!/usr/bin/env python
"""
A cation with its first shell as the sample: plan item BT4.

`cmfqsar/solvation.py` builds a solvation complex, optimises it with GFN2-xTB and puts it in the
frame its own coordination polyhedron defines. This script measures the three things that decide
whether any of that is worth having, and it measures them against data that already exist.

The reference
-------------
``examples/ether/`` holds 1,050 Li(+)-ether complexes relaxed with density functional theory
(wB97X-D3/def2-TZVP) together with their binding energies, prepared for item EC7. Every stage
here uses them, so the builder and the frame are checked against a published relaxation rather
than against themselves.

What is measured
----------------
**1. The builder** (``--stage builder``). For each molecule: its free form optimised with xTB, the
complex the builder places and optimises from a generated conformer, and the published complex
re-optimised with xTB. That separates two questions that are easy to confuse. Does the placing
find the right arrangement -- the number of oxygens that reach the ion and their distances -- and
does the method reproduce the energy once the arrangement is right? The published geometry is the
answer to both.

**2. The gate** (``--stage gate``). Item BT4 asks whether a model on the complex beats the
five-field model on the free solvent, which is the test of whether the shell carries what the
solvent alone does not. Four models on the same molecules, the same folds and the same target
(the published binding energy):

    solvent, no frame     the ether alone, alignment-free kernel: the model of item EC7
    complex, no frame     the ion put back, alignment-free kernel
    complex, cation frame the overlap kernel in the frame the polyhedron fixes
    solvent, cation frame the ion removed again, the same frame kept

The last one is there to separate the two things the complex brings: the ion's own field, and the
frame that having one atom in common makes possible. Without it a gain could be credited to the
wrong one.

**3. The co-field** (``--stage gate`` as well). A frame is what a co-field needs. The item asks
that it name the donor oxygens, so the fitted co-field of the complex model is evaluated on a grid
and the question is asked plainly: how far from the ion, and how far from the nearest donor
oxygen, does its largest magnitude lie?

**4. The series the item names** (``--stage series``). Carbonates, ethers and fluorinated ethers
with one to four solvent molecules around Li(+), every complex built and optimised here, giving
the three targets the item asks for: the binding energy, the desolvation energy of one molecule
and the coordination number that survives the optimisation.

Usage
-----
    python benchmarks/run_solvation.py                       # every stage, then the table
    python benchmarks/run_solvation.py --stage builder --limit 150
    python benchmarks/run_solvation.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger                                            # noqa: E402
from rdkit.Geometry import Point3D                                          # noqa: E402

from cmfqsar.cofields import cofield_volume, make_grid                      # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields                # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               block_sizes, channel_scale, feature_matrix)
from cmfqsar.model import CMFModel, _Learner, cv_predict, r2_score, rmse    # noqa: E402
from cmfqsar import solvation as S                                          # noqa: E402

import kernel_cache as KC                                                   # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "solvation")
SDF = os.path.join(ROOT, "examples", "ether", "geometries.sdf")
WORK = os.path.join(HERE, "cache", "solvation")

SEED = 0
OUTER, INNER = 5, 10
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
ALPHAS = (0.3, 0.5, 1.0, 2.0)
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")
ION = "Li"
#: the first shell is counted inside this distance, as item EC7 counted it
SHELL = 2.6
#: starting conformers the builder tries per molecule, each optimised, the lowest kept
STARTS = 5

#: the series item BT4 names, as SMILES. Carbonates and ethers are the two families every
#: electrolyte paper uses; the fluorinated ethers are the family the recent ones add.
SERIES = {
    "carbonate": (("EC", "C1COC(=O)O1"), ("PC", "CC1COC(=O)O1"), ("DMC", "COC(=O)OC"),
                  ("DEC", "CCOC(=O)OCC"), ("EMC", "CCOC(=O)OC"), ("FEC", "FC1COC(=O)O1"),
                  ("VC", "C1=COC(=O)O1")),
    "ether": (("DME", "COCCOC"), ("DEE", "CCOCCOCC"), ("DOL", "C1COCO1"), ("THF", "C1CCOC1"),
              ("diglyme", "COCCOCCOC"), ("triglyme", "COCCOCCOCCOC"),
              ("tetraglyme", "COCCOCCOCCOCCOC")),
    "fluorinated ether": (("FDME", "FCOCCOCF"), ("TFEE", "FC(F)(F)COCC(F)(F)F"),
                          ("BTFE", "FC(F)(F)COCCOCC(F)(F)F"), ("TTE", "FC(F)(F)CCOCCOCC(F)(F)F"),
                          ("HFE", "FC(F)(F)C(F)(F)COC")),
}
SHELL_SIZES = (1, 2, 3, 4)


# --------------------------------------------------------------------------------- the data


def reference_complexes(limit=0):
    """The published complexes: the ether of the SDF with its lithium put back at `ion_xyz`.

    Returns (complexes, solvents, binding energies, names). The solvent is the same molecule
    without the ion, in the same geometry, which is what item EC7 modelled.
    """
    complexes, solvents, y, names = [], [], [], []
    for m in Chem.SDMolSupplier(SDF, removeHs=False, sanitize=True):
        if m is None:
            continue
        if limit and len(complexes) >= limit:
            break
        props = m.GetPropsAsDict()
        if not props.get("li_ebind"):
            continue
        xyz = [float(v) for v in str(props["ion_xyz"]).split()]
        rw = Chem.RWMol()
        atom = Chem.Atom(ION)
        atom.SetFormalCharge(1)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
        rw.InsertMol(m)
        mol = rw.GetMol()
        pos = np.asarray(m.GetConformer().GetPositions(), float)
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetAtomPosition(0, Point3D(*[float(v) for v in xyz]))
        for k, p in enumerate(pos):
            conf.SetAtomPosition(k + 1, Point3D(*[float(v) for v in p]))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        Chem.SanitizeMol(mol)
        name = m.GetProp("_Name")
        mol.SetProp("_Name", name)
        mol.SetProp("smiles", str(props.get("smiles", "")))
        complexes.append(mol)
        solvents.append(m)
        y.append(float(props["li_ebind"]))
        names.append(name)
    return complexes, solvents, np.asarray(y, float), names


def without_ion(mol):
    """The complex's solvent part, in the complex geometry: what item EC7 used."""
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(int(S.ion_index(mol, ION)))
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


# ----------------------------------------------------------------------- stage 1: the builder


def builder_stage(limit=150, engine=None):
    """The placed-and-optimised complex against the published relaxation."""
    engine = engine or S.Engine(workdir=WORK)
    complexes, solvents, y, names = reference_complexes(limit)
    print(f"  {len(complexes)} published complexes, binding energy {y.min():.2f} to "
          f"{y.max():.2f} eV", flush=True)
    e_ion = S.ion_energy(ION, engine)
    rows, failed = [], []
    t0 = time.time()
    for k, (cx, solvent, target, name) in enumerate(zip(complexes, solvents, y, names)):
        smiles = cx.GetProp("smiles")
        rec = {"name": name, "smiles": smiles, "dft_ebind": float(target)}
        donors_ref, d_ref = S.coordination(cx, 0, ION, SHELL)
        rec.update({"dft_donors": int(len(donors_ref)),
                    "dft_nearest": float(d_ref[0]) if len(d_ref) else float("nan"),
                    "dft_mean_distance": float(d_ref.mean()) if len(d_ref) else float("nan")})
        try:
            free = S._embed(Chem.MolFromSmiles(smiles), seed=k)
            _m, e_free = S.optimise(free, f"free_{name}", engine, index=3 * k + 1)
            # the published arrangement, this method's energy: the method alone
            _m2, e_ref = S.optimise(cx, f"refopt_{name}", engine, index=3 * k + 2)
            got_ref = S.coordination(_m2, 0, ION, SHELL)
            # the builder's own arrangement, from generated conformers: placing and method
            built, e_built, tried = S.best_wrapped_complex(
                Chem.MolFromSmiles(smiles), f"built_{name}", ion=ION,
                n_donors=int(len(donors_ref)) or 0, conformers=STARTS, engine=engine,
                index=k + 1)
            got_built = S.coordination(built.mol, 0, ION, SHELL)
            rec["starts_tried"] = len(tried)
            rec["start_spread_ev"] = (float(np.max(tried) - np.min(tried)) if len(tried) > 1
                                      else 0.0)
            rec.update({
                "xtb_ebind_from_dft_geometry": S.binding_energy(e_ref, e_ion, [e_free]),
                "xtb_ebind_from_the_builder": S.binding_energy(e_built, e_ion, [e_free]),
                "xtb_donors_from_dft_geometry": int(len(got_ref[0])),
                "xtb_donors_from_the_builder": int(len(got_built[0])),
                "xtb_nearest_from_dft_geometry": (float(got_ref[1][0]) if len(got_ref[1])
                                                  else float("nan")),
                "xtb_nearest_from_the_builder": (float(got_built[1][0]) if len(got_built[1])
                                                 else float("nan")),
                "xtb_mean_from_the_builder": (float(got_built[1].mean()) if len(got_built[1])
                                              else float("nan")),
            })
            rows.append(rec)
        except Exception as exc:                                        # noqa: BLE001
            failed.append({"name": name, "smiles": smiles, "error": f"{type(exc).__name__}: {exc}"})
        if (k + 1) % 25 == 0:
            print(f"    {k + 1}/{len(complexes)} ({time.time() - t0:.0f} s, "
                  f"{len(failed)} failed)", flush=True)

    def stat(key):
        v = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))], float)
        return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else float("nan"),
                "sd": float(v.std(ddof=0)) if len(v) else float("nan")}

    out = {"n": len(rows), "failed": failed, "level": engine.level(), "shell": SHELL,
           "ion_energy_ev": e_ion, "rows": rows, "elapsed_s": time.time() - t0}
    for key in ("starts_tried", "start_spread_ev",
                "dft_donors", "xtb_donors_from_dft_geometry", "xtb_donors_from_the_builder",
                "dft_nearest", "xtb_nearest_from_dft_geometry", "xtb_nearest_from_the_builder",
                "dft_mean_distance", "xtb_mean_from_the_builder",
                "dft_ebind", "xtb_ebind_from_dft_geometry", "xtb_ebind_from_the_builder"):
        out[key] = stat(key)
    for key in ("xtb_ebind_from_dft_geometry", "xtb_ebind_from_the_builder"):
        a = np.array([r["dft_ebind"] for r in rows if np.isfinite(r.get(key, np.nan))])
        b = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
        if len(a) > 2:
            out[f"{key}_vs_dft"] = {
                "r": float(np.corrcoef(a, b)[0, 1]),
                "mean_error": float(np.mean(b - a)),
                "mae": float(np.mean(np.abs(b - a))),
                "rmse": float(rmse(a, b)),
                "n": int(len(a))}
    print(f"  donors: DFT {out['dft_donors']['mean']:.2f} +/- {out['dft_donors']['sd']:.2f}, "
          f"xtb from that geometry {out['xtb_donors_from_dft_geometry']['mean']:.2f}, "
          f"from the builder {out['xtb_donors_from_the_builder']['mean']:.2f}")
    print(f"  nearest O: DFT {out['dft_nearest']['mean']:.3f} A, "
          f"xtb {out['xtb_nearest_from_dft_geometry']['mean']:.3f} A, "
          f"builder {out['xtb_nearest_from_the_builder']['mean']:.3f} A")
    for key in ("xtb_ebind_from_dft_geometry", "xtb_ebind_from_the_builder"):
        if f"{key}_vs_dft" in out:
            v = out[f"{key}_vs_dft"]
            print(f"  {key}: r {v['r']:+.3f}, mean error {v['mean_error']:+.3f} eV, "
                  f"MAE {v['mae']:.3f} eV on {v['n']}")
    return out


# -------------------------------------------------------------------------- stage 2: the gate


def folds(n, k=OUTER, seed=SEED):
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    return [(np.setdiff1d(np.arange(n), g), g) for g in np.array_split(order, k)]


def paired_interval(a, b, seed=SEED, draws=2000):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(draws)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(d)), "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def _invariant_kernels(mols, names):
    """One autocorrelation kernel per field width."""
    out = {}
    spec = AutocorrelationSpec()
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=list(FIELDS), alpha=alpha, charge_model="gasteiger")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        F = feature_matrix(mf, spec, fieldset=fs)
        out[alpha] = (F, spec, block_sizes(spec, fs.n_fields))
    return out


def _overlap_kernels(mols, names):
    """One overlap kernel per field width, in whatever frame the molecules are already in."""
    out = {}
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=list(FIELDS), alpha=alpha, charge_model="gasteiger")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        table = AtomTable.from_molecules(mf)
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        out[alpha] = KC.joint_training_kernel(model, table)
    return out


def _kernel_at(entry, tr, te=None):
    """Training and test blocks of one prepared kernel or feature matrix."""
    if isinstance(entry, tuple):
        F, spec, blocks = entry
        sc = channel_scale(F[tr], spec.n_bins, blocks=blocks)
        K_tr = autocorrelation_kernel(F[tr], scale=sc, n_bins=spec.n_bins, blocks=blocks)
        if te is None:
            return K_tr, None
        return K_tr, autocorrelation_kernel(F[tr], F[te], scale=sc, n_bins=spec.n_bins,
                                            blocks=blocks)
    K = entry
    if te is None:
        return K[np.ix_(tr, tr)], None
    return K[np.ix_(tr, tr)], K[np.ix_(te, tr)]


def nested(kernels, y, design, label=""):
    """Out-of-fold predictions with the ridge and the field width chosen inside each fold."""
    pred = np.full(len(y), np.nan)
    per_fold = []
    t0 = time.time()
    for k, (tr, te) in enumerate(design):
        best = None
        for alpha, entry in kernels.items():
            K_tr, _ = _kernel_at(entry, tr)
            for lam in LAMS:
                p = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=INNER, seed=SEED)
                err = rmse(y[tr], p)
                if best is None or err < best[0]:
                    best = (err, alpha, lam)
        _e, alpha, lam = best
        K_tr, K_te = _kernel_at(kernels[alpha], tr, te)
        learner = _Learner("krr", {"lam": lam}, True).fit(K_tr, y[tr])
        pred[te] = learner.predict(K_te)
        per_fold.append({"fold": k, "alpha": alpha, "lam": lam, "n_test": int(len(te)),
                         "mae": float(np.mean(np.abs(pred[te] - y[te])))})
    maes = np.array([f["mae"] for f in per_fold])
    rec = {"n": int(len(y)), "mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=0)),
           "mae_pooled": float(np.mean(np.abs(pred - y))),
           "r2_pooled": float(r2_score(y, pred)), "per_fold": per_fold,
           "abs_errors": [float(v) for v in np.abs(pred - y)],
           "elapsed_s": time.time() - t0}
    print(f"    {label:<22} MAE {rec['mae_mean']:.3f} +/- {rec['mae_sd']:.3f} eV, "
          f"R2 {rec['r2_pooled']:+.3f}  ({rec['elapsed_s']:.0f} s)", flush=True)
    return rec


def cofield_reading(mols, names, y, alpha=0.5, lam=0.1, n_report=40):
    """Where the co-field of the complex model has its weight: the item's own question."""
    fs = FieldSet.create(physchem=list(FIELDS), alpha=alpha, charge_model="gasteiger")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": lam})
    model.fit(table, y, cv_folds=0)
    out = {"alpha": alpha, "lam": lam, "fields": list(FIELDS), "per_field": []}
    grid = make_grid(np.vstack([m.coords for m in mf[:n_report]]), margin=2.0, spacing=0.5)
    for f, spec in enumerate(fs.specs):
        volume = cofield_volume(model, f, grid).ravel()
        k = int(np.argmax(np.abs(volume)))
        point = grid.points()[k]
        # the ion is at the origin of every framed complex, so the distance to it is the norm
        donor_distance = []
        for m, mol in zip(mf[:n_report], mols[:n_report]):
            donors, _d = S.coordination(mol, 0, ION, SHELL)
            if len(donors):
                donor_distance.append(float(np.min(np.linalg.norm(
                    m.coords[donors] - point, axis=1))))
        out["per_field"].append({
            "field": spec.label, "extremum": float(volume[k]),
            "distance_from_the_ion": float(np.linalg.norm(point)),
            "distance_from_the_nearest_donor": (float(np.mean(donor_distance))
                                                if donor_distance else float("nan")),
            "point": [float(v) for v in point]})
        print(f"    co-field {spec.label:<14} largest at {np.linalg.norm(point):.2f} A from the "
              f"ion, {np.mean(donor_distance) if donor_distance else float('nan'):.2f} A from "
              f"the nearest donor", flush=True)
    return out


def gate_stage(limit=0):
    """The four models of the gate, and the co-field reading."""
    complexes, _solvents, y, names = reference_complexes(limit)
    print(f"  {len(complexes)} complexes, target {y.min():.2f} to {y.max():.2f} eV, "
          f"mean {y.mean():.2f}, sd {y.std(ddof=0):.2f}", flush=True)
    quality = [S.frame_quality(m, 0, ION, SHELL) for m in complexes]
    donors = np.array([q["donors"] for q in quality])
    ratio = np.array([q["moment_ratio_xy"] for q in quality])
    print(f"  the polyhedron: {donors.mean():.2f} +/- {donors.std():.2f} donors inside "
          f"{SHELL} A; the two smallest moments agree to within 1 % in "
          f"{float(np.mean(np.abs(ratio - 1.0) < 0.01)):.1%} of the complexes, where the frame "
          f"leaves a rotation free", flush=True)

    framed = [S.with_frame(m, 0, ION, SHELL) for m in complexes]
    framed_solvent = [without_ion(m) for m in framed]
    free_solvent = [without_ion(m) for m in complexes]
    design = folds(len(y))

    t0 = time.time()
    models = {}
    models["solvent, no frame"] = nested(_invariant_kernels(free_solvent, names), y, design,
                                         "solvent, no frame")
    models["complex, no frame"] = nested(_invariant_kernels(complexes, names), y, design,
                                         "complex, no frame")
    models["complex, cation frame"] = nested(_overlap_kernels(framed, names), y, design,
                                             "complex, cation frame")
    models["solvent, cation frame"] = nested(_overlap_kernels(framed_solvent, names), y, design,
                                             "solvent, cation frame")
    out = {"n": int(len(y)), "mean": float(y.mean()), "sd": float(y.std(ddof=0)),
           "mae_of_the_mean": float(np.mean(np.abs(y - y.mean()))),
           "donors_mean": float(donors.mean()), "donors_sd": float(donors.std(ddof=0)),
           "frame_free_rotation_share": float(np.mean(np.abs(ratio - 1.0) < 0.01)),
           "models": models, "elapsed_s": time.time() - t0,
           "shell": SHELL, "fields": list(FIELDS), "alphas": list(ALPHAS), "lams": list(LAMS)}
    pairs = (("complex frame beats solvent no frame", "complex, cation frame",
              "solvent, no frame"),
             ("complex no frame beats solvent no frame", "complex, no frame",
              "solvent, no frame"),
             ("the frame alone", "solvent, cation frame", "solvent, no frame"),
             ("the ion inside the frame", "complex, cation frame", "solvent, cation frame"))
    out["paired"] = {label: paired_interval(models[a]["abs_errors"], models[b]["abs_errors"])
                     for label, a, b in pairs}
    for label, v in out["paired"].items():
        print(f"    {label:<40} {v['mean']:+.3f} eV [{v['low']:+.3f}, {v['high']:+.3f}] "
              f"{'excludes' if v['excludes_zero'] else 'includes'} zero", flush=True)
    # the co-field is read at the setting the folds chose most often, not at a guess
    chosen = models["complex, cation frame"]["per_fold"]
    alpha = max(set(f["alpha"] for f in chosen),
                key=lambda a: sum(1 for f in chosen if f["alpha"] == a))
    lam = max(set(f["lam"] for f in chosen),
              key=lambda l: sum(1 for f in chosen if f["lam"] == l))
    out["cofield"] = cofield_reading(framed, names, y, alpha=alpha, lam=lam)
    return out


# ------------------------------------------------------------------------ stage 3: the series


def series_stage(engine=None, sizes=SHELL_SIZES):
    """The carbonate, ether and fluorinated-ether series the item names, computed here."""
    engine = engine or S.Engine(workdir=WORK)
    e_ion = S.ion_energy(ION, engine)
    rows, failed = [], []
    t0 = time.time()
    index = 0
    for family, members in SERIES.items():
        for label, smiles in members:
            index += 1
            try:
                free = S._embed(Chem.MolFromSmiles(smiles), seed=index)
                _m, e_free = S.optimise(free, f"solvent_{label}", engine, index=1000 + index)
            except Exception as exc:                                    # noqa: BLE001
                failed.append({"solvent": label, "n": 0,
                               "error": f"{type(exc).__name__}: {exc}"})
                continue
            energies = {}
            for n in sizes:
                try:
                    built = S.place_shell([Chem.MolFromSmiles(smiles)] * n, ion=ION, seed=index)
                    mol, e = S.optimise(built.mol, f"{label}_{n}", engine,
                                        index=2000 + 10 * index + n)
                    got, dist = S.coordination(mol, 0, ION, SHELL)
                    energies[n] = e
                    rows.append({
                        "family": family, "solvent": label, "smiles": smiles, "n": n,
                        "binding_ev": S.binding_energy(e, e_ion, [e_free] * n),
                        "binding_per_solvent_ev": S.binding_energy(e, e_ion, [e_free] * n) / n,
                        "donors": int(len(got)),
                        "nearest": float(dist[0]) if len(dist) else float("nan"),
                        "mean_distance": float(dist.mean()) if len(dist) else float("nan"),
                        "desolvation_ev": (S.desolvation_energy(e, energies[n - 1], e_free)
                                           if (n - 1) in energies else float("nan")),
                        "solvent_ev": e_free, "complex_ev": e})
                except Exception as exc:                                # noqa: BLE001
                    failed.append({"solvent": label, "n": n,
                                   "error": f"{type(exc).__name__}: {exc}"})
            print(f"    {label:<12} {family:<18} "
                  + "  ".join(f"n={r['n']} {r['binding_ev']:+.2f} eV"
                              for r in rows if r["solvent"] == label), flush=True)
    out = {"rows": rows, "failed": failed, "level": engine.level(), "ion_energy_ev": e_ion,
           "sizes": list(sizes), "shell": SHELL, "elapsed_s": time.time() - t0}
    for family in SERIES:
        got = [r for r in rows if r["family"] == family]
        if got:
            first = [r["binding_ev"] for r in got if r["n"] == 1]
            out[family] = {"n_solvents": len(set(r["solvent"] for r in got)),
                           "binding_at_1_mean": float(np.mean(first)) if first else float("nan"),
                           "binding_at_1_sd": float(np.std(first)) if first else float("nan")}
    return out


# ------------------------------------------------------------------------------- the report


def write_table(path=None):
    path = path or RESULTS
    out = ["# A cation with its first shell as the sample (plan item BT4)", ""]
    p = os.path.join(path, "builder.json")
    if os.path.exists(p):
        b = json.load(open(p, encoding="utf-8"))
        out += ["## The builder against a published relaxation", "",
                f"{b['n']} of the Li(+)-ether complexes of `examples/ether/`, {b['level']}. "
                f"The first shell is counted inside {b['shell']} A.", "",
                "| Quantity | The published relaxation | xTB from that geometry | "
                "The builder's own complex |", "|---|---|---|---|"]
        rows = (("oxygens reaching the ion", "dft_donors", "xtb_donors_from_dft_geometry",
                 "xtb_donors_from_the_builder"),
                ("nearest oxygen (A)", "dft_nearest", "xtb_nearest_from_dft_geometry",
                 "xtb_nearest_from_the_builder"),
                ("binding energy (eV)", "dft_ebind", "xtb_ebind_from_dft_geometry",
                 "xtb_ebind_from_the_builder"))
        for label, a, c, d in rows:
            def cell(key):
                v = b.get(key, {})
                return (f"{v.get('mean', float('nan')):.2f} +/- {v.get('sd', float('nan')):.2f}"
                        if v else "-")
            out.append(f"| {label} | {cell(a)} | {cell(c)} | {cell(d)} |")
        out.append("")
        for key, label in (("xtb_ebind_from_dft_geometry_vs_dft",
                            "xTB on the published geometry"),
                           ("xtb_ebind_from_the_builder_vs_dft", "the builder's own complex")):
            v = b.get(key)
            if v:
                out.append(f"* Against the published binding energy, {label}: correlation "
                           f"{v['r']:+.3f}, mean error {v['mean_error']:+.3f} eV, "
                           f"mean absolute error {v['mae']:.3f} eV on {v['n']} molecules.")
        if b.get("failed"):
            out.append(f"* {len(b['failed'])} molecules failed and are named in `builder.json`.")
        out.append("")
    p = os.path.join(path, "gate.json")
    if os.path.exists(p):
        g = json.load(open(p, encoding="utf-8"))
        out += ["## The gate: does the shell carry what the solvent alone does not", "",
                f"{g['n']} complexes, the published binding energy as the target, five outer "
                f"folds with the ridge and the field width chosen inside each. The ion has "
                f"{g['donors_mean']:.2f} +/- {g['donors_sd']:.2f} donors inside {g['shell']} A. "
                f"Predicting the mean of the set costs {g['mae_of_the_mean']:.3f} eV.", "",
                "| Model | MAE (eV) | R2 |", "|---|---|---|"]
        for label, rec in g["models"].items():
            out.append(f"| {label} | {rec['mae_mean']:.3f} +/- {rec['mae_sd']:.3f} | "
                       f"{rec['r2_pooled']:+.3f} |")
        out.append("")
        for label, v in g["paired"].items():
            out.append(f"* Paired over the same molecules, {label}: {v['mean']:+.3f} eV "
                       f"[{v['low']:+.3f}, {v['high']:+.3f}], "
                       f"{'excludes' if v['excludes_zero'] else 'includes'} zero.")
        out += ["", f"In {g['frame_free_rotation_share']:.1%} of the complexes the two smallest "
                f"moments of the polyhedron agree to within one per cent, which leaves the frame "
                f"free to rotate about an axis.", ""]
        cf = g.get("cofield")
        if cf:
            out += ["### What the co-field names", "",
                    "The fitted co-field of the complex model, evaluated on a grid: where its "
                    "largest magnitude lies.", "",
                    "| Field | Distance from the ion (A) | Distance from the nearest donor (A) |",
                    "|---|---|---|"]
            for rec in cf["per_field"]:
                out.append(f"| {rec['field']} | {rec['distance_from_the_ion']:.2f} | "
                           f"{rec['distance_from_the_nearest_donor']:.2f} |")
            out.append("")
    p = os.path.join(path, "series.json")
    if os.path.exists(p):
        s = json.load(open(p, encoding="utf-8"))
        out += ["## The series computed here", "",
                f"Every complex built by `cmfqsar/solvation.py` and optimised with {s['level']}. "
                f"The binding energy is E(complex) - E(ion) - n E(solvent) and the desolvation "
                f"energy E(n-1) + E(solvent) - E(n), both in eV.", "",
                "| Family | Solvent | n | Binding (eV) | Per solvent (eV) | Desolvation (eV) | "
                "Oxygens at the ion | Nearest O (A) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in s["rows"]:
            des = ("-" if not np.isfinite(r["desolvation_ev"])
                   else f"{r['desolvation_ev']:+.2f}")
            out.append(f"| {r['family']} | {r['solvent']} | {r['n']} | {r['binding_ev']:+.2f} | "
                       f"{r['binding_per_solvent_ev']:+.2f} | {des} | {r['donors']} | "
                       f"{r['nearest']:.2f} |")
        out.append("")
        if s.get("failed"):
            out.append(f"{len(s['failed'])} complexes failed to build or optimise and are named "
                       f"in `series.json`.")
            out.append("")
    text = "\n".join(out) + "\n"
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="A cation with its first shell as the sample (BT4).")
    ap.add_argument("--stage", choices=("builder", "gate", "series", "all"), default="all")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only this many reference complexes (0: all)")
    ap.add_argument("--builder-limit", type=int, default=150,
                    help="how many complexes the builder stage rebuilds (it runs xTB three "
                         "times per molecule)")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        print(write_table())
        return 0
    engine = S.Engine(workdir=WORK)
    stages = ("builder", "gate", "series") if args.stage == "all" else (args.stage,)
    if "builder" in stages:
        print("stage builder: the placed complex against the published relaxation", flush=True)
        json.dump(builder_stage(args.builder_limit, engine),
                  open(os.path.join(RESULTS, "builder.json"), "w", encoding="utf-8"), indent=1)
    if "gate" in stages:
        print("stage gate: the complex against the free solvent", flush=True)
        json.dump(gate_stage(args.limit),
                  open(os.path.join(RESULTS, "gate.json"), "w", encoding="utf-8"), indent=1)
    if "series" in stages:
        print("stage series: carbonates, ethers and fluorinated ethers", flush=True)
        json.dump(series_stage(engine),
                  open(os.path.join(RESULTS, "series.json"), "w", encoding="utf-8"), indent=1)
    write_table()
    print(f"wrote {os.path.join(RESULTS, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
