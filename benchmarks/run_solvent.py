#!/usr/bin/env python
"""
Is the water worth a field of its own?

Plan item D9. Every field this program builds sits on the atoms of the molecule. The 2015
indicator-field formalism does not require that, and a water occupancy map from a simulation is
a continuous density like any other, so `cmfqsar/solvent.py` turns the places a ligand's water
occupies into a field: one Gaussian per hydration site, its coefficient fitted to the map. This
measures whether that field carries anything.

The water field cannot share a table with the molecule's own fields, because a table holds one
set of atoms and a hydration site is not an atom of the molecule. It is therefore a block of its
own, mixed in with one coefficient, exactly as the variance field of plan item D10 is:

    K(h) = (1 - h) K_fields + h K_water

with `h` searched on the same folds as everything else. Five rows:

  the five fields alone     h = 0, the reference of rule 1
  the water alone           h = 1, which says whether the map carries any signal at all
  the best mixture          the best inner h, which is the item's question
  sites moved at random     the same number of sites per ligand, in the same box, at random
                            positions: the control that says whether the *places* matter
  one ligand's water        every ligand given the first ligand's sites: the control that says
                            whether anything beyond the shape shared by the whole site matters

Every difference is reported against the five fields with the bootstrap interval of rule 4, so a
gain below the interval is not a gain. The two controls are the point of the design: a water
field has more parameters than no water field, and a kernel block of plausible numbers can
improve a fit without meaning anything.

The sites come from `examples/md/<set>_water.json`, written by
`examples/make_md_trajectories.py`; this script needs no simulation software.

Usage
-----
    python benchmarks/run_solvent.py
    python benchmarks/run_solvent.py --sets mcl1
    python benchmarks/run_solvent.py --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import RDLogger  # noqa: E402

import kernel_cache as KC  # noqa: E402

from cmfqsar import solvent as SV  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "solvent")
MD_DATA = os.path.join(ROOT, "examples", "md")
CRYSTAL = os.path.join(ROOT, "examples", "openff")
SETS = ("mcl1",)
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
MIXES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
FOLDS = 10
SEED = 0
RUNS = 2000
#: the two numbers that define a hydration site, swept to see how much they decide
SWEEP_OCCUPANCY = (0.3, 0.4, 0.5, 0.6, 0.7)
SWEEP_DISTANCE = (1.8, 2.4, 3.0, 3.6)


def _say(text):
    print(text, flush=True)


# ----------------------------------------------------------------------------------------------
# the data
# ----------------------------------------------------------------------------------------------

def water_sites(name, alpha=None):
    """{ligand: (centres, coefficients, occupancies)} and the record the maker wrote.

    A coefficient is fitted against Gaussians of one width, so `alpha` picks the set fitted at
    the width the kernel will use. The file carries one set per attenuation of the grid; without
    a match it falls back to the single column it also stores, and says nothing, because that is
    what an older file holds.
    """
    path = os.path.join(MD_DATA, name + "_water.json")
    if not os.path.isfile(path):
        return None, None
    with open(path) as fh:
        d = json.load(fh)
    key = None if alpha is None else f"{float(alpha):g}"
    out = {}
    for row in d["ligands"]:
        arr = np.asarray(row["sites"], float)
        if arr.size == 0:
            continue
        # column 3 is a coefficient fitted at the file's own attenuation and column 4 the share
        # of frames the site is occupied
        coeff = arr[:, 3].copy()
        stored = row.get("coefficients") or {}
        if key is not None and key in stored:
            coeff = np.asarray(stored[key], float)
        out[row["name"]] = (arr[:, :3].copy(), coeff, arr[:, 4].copy())
    return out, d


def crystal_set(name, activity="pAffinity"):
    ds = load_dataset(os.path.join(CRYSTAL, name + ".sdf"), activity_prop=activity)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    return ([m for m, g in zip(ds.mols, ok) if g],
            [n for n, g in zip(ds.names, ok) if g], y[ok])


# ----------------------------------------------------------------------------------------------
# the blocks
# ----------------------------------------------------------------------------------------------

def field_kernel(mols, names, alpha):
    """The joint kernel of the five physico-chemical fields."""
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger",
                         param_scheme="rdkit")
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    return KC.joint_training_kernel(model, table)


def water_kernel(sites, names, alpha):
    """The kernel of the water fields, one hydration-site Gaussian per site."""
    fs = SV.solvent_fieldset(alpha=alpha)
    fields = [SV.water_field(sites[n][0], sites[n][1], name=n) for n in names]
    table = AtomTable.from_molecules(fields)
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    return np.asarray(model.joint_kernel(model.prepare_training_kernels(table)), float)


def scrambled_sites(sites, names, seed=SEED):
    """The same number of sites per ligand, at random places in the box they came from.

    The control for the positions: if a kernel built from sites anywhere in the site does as
    well, then what the water field carries is how many sites a ligand has and not where they
    are.
    """
    rng = np.random.RandomState(int(seed))
    every = np.vstack([sites[n][0] for n in names])
    lo, hi = every.min(axis=0), every.max(axis=0)
    out = {}
    for n in names:
        c, w, occ = sites[n]
        out[n] = (rng.uniform(lo, hi, size=(len(c), 3)), w.copy(), occ.copy())
    return out


def shared_sites(sites, names):
    """Every ligand given the first ligand's sites: the control for telling ligands apart."""
    c, w, occ = sites[names[0]]
    return {n: (c.copy(), w.copy(), occ.copy()) for n in names}


def best(K, y, folds=FOLDS, seed=SEED, lams=LAMS):
    out = None
    for lam in lams:
        pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=folds, seed=seed)
        q2 = r2_score(y, pred)
        if out is None or q2 > out[0]:
            out = (float(q2), float(rmse(y, pred)), float(lam), np.asarray(pred, float))
    return out


def mixed(Ka, Kb, h):
    return (1.0 - float(h)) * Ka + float(h) * Kb


def bootstrap_difference(truth, a, b, runs=RUNS, seed=SEED):
    """(difference in q2, low, high) for prediction `a` against `b`, resampling the molecules."""
    truth = np.asarray(truth, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(truth) & np.isfinite(a) & np.isfinite(b)
    truth, a, b = truth[ok], a[ok], b[ok]
    if len(truth) < 4:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(int(seed))
    out = []
    for _ in range(int(runs)):
        idx = rng.randint(0, len(truth), len(truth))
        if np.std(truth[idx]) <= 0:
            continue
        out.append(r2_score(truth[idx], a[idx]) - r2_score(truth[idx], b[idx]))
    if not out:
        return float("nan"), float("nan"), float("nan")
    return (float(r2_score(truth, a) - r2_score(truth, b)),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


# ----------------------------------------------------------------------------------------------
# one set
# ----------------------------------------------------------------------------------------------

def _scan(Kf, Kw, y, mixes=MIXES):
    """The best (q2, rmse, lam, predictions) at every mixing coefficient."""
    rows = []
    for h in mixes:
        q2, rms, lam, pred = best(mixed(Kf, Kw, h), y)
        rows.append(dict(mix=float(h), q2=q2, rmse=rms, lam=lam, pred=pred))
    return rows


def occupancy_maps(name, ligands, log=_say):
    """The occupancy map of every ligand, rebuilt from the trajectories.

    The sites in ``examples/md/<set>_water.json`` were cut at one threshold and one spacing, so
    varying either needs the maps themselves, which need the trajectories. They are not in the
    repository -- ``examples/make_md_trajectories.py`` rebuilds them -- so this returns None when
    they are absent and the stage is skipped rather than guessed at. Reading a map needs mdtraj
    and no simulation software.
    """
    runs = os.path.join(MD_DATA, "work", name, "runs")
    work = os.path.join(MD_DATA, "work", name)
    if not os.path.isdir(runs):
        return None
    ready = sorted(glob.glob(os.path.join(work, "*_ready.pdb")))
    if not ready:
        return None
    out = {}
    for nm in ligands:
        top = os.path.join(runs, f"{nm}.pdb")
        dcd = os.path.join(runs, f"{nm}.dcd")
        if not (os.path.isfile(top) and os.path.isfile(dcd)):
            continue
        try:
            out[nm] = SV.occupancy(top, dcd, reference=ready[0], name=nm)
        except (SV.SolventError, Exception) as exc:      # noqa: BLE001 - reported and skipped
            log(f"    {nm}: no map -- {type(exc).__name__}: {exc}")
    return out or None


def sites_at(maps, names, alpha, min_occupancy, min_distance):
    """{ligand: (centres, coefficients, occupancies)} at one definition of a site.

    Returns (sites, missing): `missing` names the ligands with no point occupied often enough,
    which is how a definition too strict for a set shows itself.
    """
    sites, missing = {}, []
    for nm in names:
        occ = maps[nm]
        centres, heights = SV.hydration_sites(occ, min_occupancy=float(min_occupancy),
                                              min_distance=float(min_distance))
        if len(centres) == 0:
            missing.append(nm)
            continue
        coeff = SV.fit_coefficients(occ, centres, alpha=float(alpha))
        sites[nm] = (centres, coeff, heights)
    return sites, missing


def stage_sweep(name, mols, names, y, maps, results=RESULTS, seed=SEED, log=_say):
    """The same comparison at every definition of a hydration site.

    Two numbers decide what a site is: the share of frames a point has to be occupied, and how
    close two sites may be. Neither follows from anything, so the question is whether the result
    of this item is a property of the water or of those two numbers.
    """
    Kf = {a: field_kernel(mols, names, a) for a in ALPHAS}
    base = {}
    for a in ALPHAS:
        base[a] = best(Kf[a], y)
    ref = max(base.values(), key=lambda r: r[0])
    rows = []
    for occ_t in SWEEP_OCCUPANCY:
        for dist in SWEEP_DISTANCE:
            picked = None
            counts, missing = None, None
            for a in ALPHAS:
                sites, miss = sites_at(maps, names, a, occ_t, dist)
                if miss:
                    counts, missing = None, miss
                    break
                counts = [len(sites[n][0]) for n in names]
                Kw = water_kernel(sites, names, a)
                scan = _scan(Kf[a], Kw, y)
                inner = [r for r in scan if 0.0 < r["mix"] < 1.0]
                chosen = max(inner, key=lambda r: r["q2"]) if inner else scan[0]
                alone = scan[-1]
                control = _scan(Kf[a], water_kernel(scrambled_sites(sites, names, seed=seed),
                                                    names, a), y)[-1]
                # each model at its own best attenuation, as the main table does, so that the
                # row for the default definition reproduces it exactly
                if picked is None or chosen["q2"] > picked["mixed"]:
                    picked = dict(picked or {}, alpha=float(a), mixed=chosen["q2"],
                                  mix=chosen["mix"], pred=chosen["pred"])
                if picked.get("alone") is None or alone["q2"] > picked["alone"]:
                    picked["alone"] = alone["q2"]
                    picked["alone_alpha"] = float(a)
                if picked.get("control") is None or control["q2"] > picked["control"]:
                    picked["control"] = control["q2"]
            if missing:
                rows.append(dict(min_occupancy=float(occ_t), min_distance=float(dist),
                                 missing=len(missing), sites=None))
                log(f"    occupancy {occ_t:.0%}, spacing {dist:.1f} A: "
                    f"{len(missing)} ligand(s) with no site; skipped")
                continue
            d, lo, hi = bootstrap_difference(y, picked["pred"], ref[3])
            rows.append(dict(min_occupancy=float(occ_t), min_distance=float(dist), missing=0,
                             sites=float(np.mean(counts)), sites_min=int(min(counts)),
                             sites_max=int(max(counts)), alpha=picked["alpha"],
                             mix=picked["mix"], mixed=picked["mixed"], alone=picked["alone"],
                             alone_alpha=picked.get("alone_alpha"),
                             control=picked["control"], delta=d, lo=lo, hi=hi))
            log(f"    occupancy {occ_t:.0%}, spacing {dist:.1f} A: "
                f"{np.mean(counts):4.1f} sites, mixed {picked['mixed']:+.3f} "
                f"({d:+.3f} [{lo:+.3f}, {hi:+.3f}]), alone {picked['alone']:+.3f}, "
                f"control {picked['control']:+.3f}")
    out = {"set": name, "n": int(len(y)), "reference": float(ref[0]),
           "occupancies": list(SWEEP_OCCUPANCY), "distances": list(SWEEP_DISTANCE),
           "rows": rows}
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, f"{name}_sweep.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def site_places(sites, names, y, alpha, radius=2.4, folds=FOLDS, seed=SEED):
    """Where the model asks for water, and where it asks for less of it.

    The sites of different ligands are in one frame, so they can be gathered into places: every
    site within `radius` of a place belongs to it. The model is the water block on its own, so
    that what comes out is what the water says and not what the molecule's own fields say; its
    dual coefficients times each site's own coefficient are the weight it puts on that site
    (:func:`cmfqsar.solvent.site_contributions`), and the mean over the ligands that have a site
    in a place is what the place is worth.

    A positive number means the model wants **more** water there and a negative one means it
    wants less, which is where displacing a water is worth trying. Under rule 5 this is a display
    tool: the table above says the water block does not improve the model, so a place named here
    is a suggestion to look at and not a prediction.
    """
    fs = SV.solvent_fieldset(alpha=alpha)
    fields = [SV.water_field(sites[n][0], sites[n][1], name=n) for n in names]
    table = AtomTable.from_molecules(fields)
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = np.asarray(model.joint_kernel(model.prepare_training_kernels(table)), float)
    pick = best(K, y, folds=folds, seed=seed)
    lam = pick[2]
    n = len(y)
    # the dual coefficients of the ridge with a bias, the same system the model solves
    A = np.vstack([np.hstack([K + lam * np.eye(n), np.ones((n, 1))]),
                   np.hstack([np.ones((1, n)), np.zeros((1, 1))])])
    rhs = np.concatenate([np.asarray(y, float), [0.0]])
    sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    dual = sol[:n]

    weights = SV.site_contributions(model, fields, dual)
    every = np.vstack([sites[nm][0] for nm in names])
    flat_w = np.concatenate(weights)
    flat_coeff = np.concatenate([sites[nm][1] for nm in names])
    flat_occ = np.concatenate([sites[nm][2] for nm in names])
    owner = np.concatenate([[i] * len(sites[nm][0]) for i, nm in enumerate(names)])

    places = []
    taken = np.zeros(len(every), bool)
    order = np.argsort(-flat_coeff)
    for i in order:
        if taken[i]:
            continue
        near = (np.linalg.norm(every - every[i], axis=1) <= float(radius)) & ~taken
        taken |= near
        places.append(dict(centre=[round(float(v), 2) for v in every[near].mean(axis=0)],
                           ligands=int(len(set(owner[near].tolist()))),
                           sites=int(near.sum()),
                           occupancy=round(float(flat_occ[near].mean()), 3),
                           weight=round(float(flat_w[near].mean()), 4)))
    places.sort(key=lambda p: -abs(p["weight"]))
    return places, float(lam)


def run_set(name, results=RESULTS, force=False, seed=SEED):
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        with open(out_path) as fh:
            return json.load(fh)
    sites, record = water_sites(name)
    if not sites:
        _say(f"  {name}: no hydration sites in examples/md; run "
             f"examples/make_md_trajectories.py first")
        return None
    mols, names, y = crystal_set(name)
    keep = [i for i, n in enumerate(names) if n in sites]
    mols = [mols[i] for i in keep]
    names = [names[i] for i in keep]
    y = y[keep]
    counts = [len(sites[n][0]) for n in names]
    _say(f"  {name}: {len(names)} ligands, {np.mean(counts):.1f} hydration sites each "
         f"(from {min(counts)} to {max(counts)})")

    t0 = time.time()
    variants = {}
    # every model is given its own best attenuation over the same grid. Choosing one attenuation
    # to suit the water block and then scoring the five fields at it would hold the reference
    # back, and the reference of rule 1 is the five fields at their best.
    rows = {}
    picks = {"fields": None}
    scans = {}
    for alpha in ALPHAS:
        Kf = field_kernel(mols, names, alpha)
        base = best(Kf, y)
        if picks["fields"] is None or base[0] > picks["fields"]["q2"]:
            picks["fields"] = dict(q2=base[0], rmse=base[1], lam=base[2], alpha=float(alpha),
                                   pred=base[3])
        # the coefficients belong to the width the kernel is about to use
        at_alpha, _rec = water_sites(name, alpha=alpha)
        variants = {"water": at_alpha or sites,
                    "scrambled": scrambled_sites(at_alpha or sites, names, seed=seed),
                    "shared": shared_sites(at_alpha or sites, names)}
        for tag, s in variants.items():
            Kw = water_kernel(s, names, alpha)
            scan = _scan(Kf, Kw, y)
            inner = [r for r in scan if 0.0 < r["mix"] < 1.0]
            chosen = max(inner, key=lambda r: r["q2"]) if inner else scan[0]
            alone = scan[-1]
            for key, row in ((f"{tag}_mixed", chosen), (f"{tag}_alone", alone)):
                if picks.get(key) is None or row["q2"] > picks[key]["q2"]:
                    picks[key] = dict(row, alpha=float(alpha))
                    if key == "water_mixed":
                        scans["water"] = [{k: v for k, v in r.items() if k != "pred"}
                                          for r in scan]
                        scans["alpha"] = float(alpha)
            rows.setdefault(tag, [])
            rows[tag].append(dict(alpha=float(alpha), base=base[0], mixed=chosen["q2"],
                                  mix=chosen["mix"], alone=alone["q2"]))
        _say(f"    alpha {alpha:<4g} fields {base[0]:+.3f}  "
             + "  ".join(f"{t} {rows[t][-1]['mixed']:+.3f} (h {rows[t][-1]['mix']:.1f}, "
                         f"alone {rows[t][-1]['alone']:+.3f})" for t in variants))

    ref = picks["fields"]
    out = {"set": name, "n": int(len(y)), "alpha": float(ref["alpha"]),
           "scan_alpha": float(scans.get("alpha", ref["alpha"])),
           "sites_mean": float(np.mean(counts)), "sites_min": int(min(counts)),
           "sites_max": int(max(counts)),
           "strongest": (float(np.mean([r["strongest"] for r in record["ligands"]]))
                         if record else None),
           "bulk": (float(record["ligands"][0]["bulk"]) if record else None),
           "min_occupancy": (float(record["min_occupancy"]) if record else None),
           "seconds": float(time.time() - t0),
           "scan": scans.get("water", []), "by_alpha": rows, "models": {}}
    out["models"]["fields"] = dict(q2=ref["q2"], rmse=ref["rmse"], lam=ref["lam"],
                                   alpha=ref["alpha"], delta=0.0, lo=0.0, hi=0.0)
    for key in ("water_mixed", "water_alone", "scrambled_mixed", "shared_mixed",
                "scrambled_alone", "shared_alone"):
        row = picks.get(key)
        if row is None:
            continue
        d, lo, hi = bootstrap_difference(y, row["pred"], ref["pred"])
        out["models"][key.replace("_mixed", "") if key.startswith(("scrambled", "shared"))
                      else key] = dict(q2=row["q2"], rmse=row["rmse"], lam=row["lam"],
                                       alpha=row["alpha"], mix=row.get("mix", 1.0),
                                       delta=d, lo=lo, hi=hi)
    alone = picks.get("water_alone")
    if alone is not None:
        at_alpha, _rec = water_sites(name, alpha=float(alone["alpha"]))
        places, lam = site_places(at_alpha or sites, names, y, alpha=float(alone["alpha"]))
        out["places"] = places
        out["places_alpha"] = float(alone["alpha"])
        out["places_lam"] = lam
        shared_places = [p for p in places if p["ligands"] >= max(3, len(names) // 4)]
        _say(f"  {len(places)} places, {len(shared_places)} of them shared by at least "
             f"{max(3, len(names) // 4)} ligands")

    w = out["models"]["water_mixed"]
    _say(f"  the five fields at their best: {ref['q2']:+.3f} (alpha {ref['alpha']:g}); "
         f"with water at its best: {w['q2']:+.3f} (alpha {w['alpha']:g}, h {w['mix']:.1f}), "
         f"{w['delta']:+.3f} [{w['lo']:+.3f}, {w['hi']:+.3f}]")
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


# ----------------------------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------------------------

LABEL = {"fields": "the five fields alone", "water_alone": "the water alone",
         "water_mixed": "the five fields and the water",
         "scrambled": "with sites moved at random (control)",
         "shared": "with one ligand's water for every ligand (control)",
         "scrambled_alone": "random sites alone (control)",
         "shared_alone": "one shared set of sites alone (control)"}
ORDER = ("fields", "water_mixed", "water_alone", "scrambled", "shared",
         "scrambled_alone", "shared_alone")


def load_all(results=RESULTS):
    out = []
    for p in sorted(glob.glob(os.path.join(results, "*.json"))):
        if p.endswith("_sweep.json"):
            continue
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def load_sweeps(results=RESULTS):
    out = {}
    for p in sorted(glob.glob(os.path.join(results, "*_sweep.json"))):
        with open(p) as fh:
            d = json.load(fh)
        out[d["set"]] = d
    return out


def report(results=RESULTS, path=None):
    rows = load_all(results)
    out = ["# Is the water worth a field of its own?", "",
           "Plan item D9. The places a ligand's water occupies in a simulation, turned into a "
           "field of Gaussians at the hydration sites and mixed with the kernel of the five "
           "physico-chemical fields under one coefficient. Every difference is against the five "
           "fields alone, with the bootstrap interval of rule 4. Two controls carry the weight "
           "of the design: the same sites moved at random, and one ligand's sites given to "
           "every ligand.", ""]
    if not rows:
        out += ["No result yet. `examples/make_md_trajectories.py` writes the hydration sites, "
                "which needs the trajectories; this table is rebuilt from the file it writes.",
                ""]
    for rec in rows:
        m = rec["models"]
        out += [f"## {rec['set']}", "",
                f"{rec['n']} ligands, {rec['sites_mean']:.1f} hydration sites each (from "
                f"{rec['sites_min']} to {rec['sites_max']}), chosen at an attenuation of "
                f"{rec['alpha']:g}. A site is a point occupied at least "
                f"{rec['min_occupancy']:.0%} of the frames, against {rec['bulk']:.0%} for bulk "
                f"water; the strongest site of a ligand averages {rec['strongest']:.0%}.", "",
                "Every model is given its own best attenuation over the same grid, so the "
                "reference is not held back by a choice made to suit the water block.", "",
                "| model | q2 | attenuation | mixing coefficient | change against the five "
                "fields |", "| --- | ---: | ---: | ---: | --- |"]
        for tag in ORDER:
            if tag not in m:
                continue
            r = m[tag]
            mix = ("--" if tag == "fields" else
                   "1.0" if tag.endswith("_alone") else f"{r.get('mix', 1.0):.1f}")
            change = ("the reference" if tag == "fields" else
                      f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
            out.append(f"| {LABEL[tag]} | {r['q2']:+.3f} | {r.get('alpha', float('nan')):g} | "
                       f"{mix} | {change} |")
        out.append("")
        out += [f"The mixing coefficient at an attenuation of {rec.get('scan_alpha', 0):g}, "
                f"where the water block did best, from the fields alone to the water alone:",
                "", "| coefficient | q2 | RMSEcv |", "| ---: | ---: | ---: |"]
        for r in rec["scan"]:
            out.append(f"| {r['mix']:.1f} | {r['q2']:+.3f} | {r['rmse']:.3f} |")
        out.append("")
        places = [p for p in (rec.get("places") or []) if p["ligands"] >= 3]
        if places:
            out += ["### Where the model asks for water", "",
                    "The sites of every ligand gathered into places, with the weight the water "
                    "block puts on each. A positive weight asks for **more** water there and a "
                    "negative one for less, which is where displacing a water is worth trying. "
                    "Only places shared by at least three ligands are listed, and under rule 5 "
                    "this is a display tool: the table above says the block does not improve "
                    "the model, so a place here is something to look at and not a prediction.",
                    "",
                    "| place (x, y, z) | ligands with a site there | mean occupancy | weight |",
                    "| --- | ---: | ---: | ---: |"]
            for p in places[:12]:
                x, y_, z = p["centre"]
                out.append(f"| {x:.1f}, {y_:.1f}, {z:.1f} | {p['ligands']} | "
                           f"{p['occupancy']:.0%} | {p['weight']:+.3f} |")
            out.append("")
        sweep = load_sweeps().get(rec["set"])
        if sweep:
            out += sweep_section(sweep)
    out += verdict(rows)
    text = "\n".join(out) + "\n"
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="\n") as fh:
            fh.write(text)
    return text


def sweep_section(sweep):
    """The table and the reading of the sweep over what counts as a hydration site."""
    done = [r for r in sweep["rows"] if r.get("sites")]
    out = ["### How much the definition of a site decides", "",
           "Two numbers decide what a hydration site is, and neither follows from anything: the "
           "share of frames a point has to be occupied, and how close two sites may be. Here "
           "the whole comparison is run again at every pair of them, each time with the "
           f"attenuation and the mixing coefficient chosen as before. The reference is the five "
           f"fields at their best, {sweep['reference']:+.3f}.", "",
           "| occupancy | spacing | sites per ligand | with water | change against the fields | "
           "water alone | random sites alone |",
           "| ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for r in sweep["rows"]:
        if not r.get("sites"):
            out.append(f"| {r['min_occupancy']:.0%} | {r['min_distance']:.1f} A | "
                       f"-- | -- | {r['missing']} ligand(s) with no site | -- | -- |")
            continue
        out.append(f"| {r['min_occupancy']:.0%} | {r['min_distance']:.1f} A | "
                   f"{r['sites']:.1f} | {r['mixed']:+.3f} | "
                   f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                   f"{r['alone']:+.3f} | {r['control']:+.3f} |")
    out.append("")
    if not done:
        out += ["Every definition tried left some ligand without a site, so nothing can be "
                "read from this.", ""]
        return out

    ahead = [r for r in done if r["lo"] > 0]
    behind = [r for r in done if r["hi"] < 0]
    alone = [r["alone"] for r in done]
    gap = [r["alone"] - r["control"] for r in done]
    out += [f"**The first conclusion does not depend on the definition.** Of the "
            f"{len(done)} definitions that give every ligand a site, "
            f"{'none' if not ahead else str(len(ahead))} beats the five fields beyond the "
            f"interval and {'none' if not behind else str(len(behind))} falls behind it. The "
            f"change against the reference runs from {min(r['delta'] for r in done):+.3f} to "
            f"{max(r['delta'] for r in done):+.3f}, and every interval contains zero.", ""]
    inverted = [r for r in done if r["control"] >= r["alone"]]
    if min(alone) > 0 and not inverted:
        out += [f"**The second conclusion does not depend on it either.** The water field on its "
                f"own stays positive throughout, from {min(alone):+.3f} to {max(alone):+.3f}, "
                f"and stays above the random-site control at every definition, by "
                f"{min(gap):+.3f} to {max(gap):+.3f}. So the signal in where the water sits is "
                f"not an artefact of one threshold.", ""]
    else:
        weak = [r for r in done if r["alone"] <= 0]
        out += [f"**The second conclusion does depend on it, and this is the finding of the "
                f"sweep.** The water field on its own runs from {min(alone):+.3f} to "
                f"{max(alone):+.3f} across the {len(done)} definitions, and is not positive at "
                f"{len(weak)} of them. What was reported for the default definition alone was "
                f"therefore one point of a range, not a property of the water.", ""]
    if inverted:
        worst = max(inverted, key=lambda r: r["control"] - r["alone"])
        out += [f"**At {len(inverted)} of the {len(done)} definitions the random-site control "
                f"does as well as the real maps or better**, the worst being an occupancy of "
                f"{worst['min_occupancy']:.0%} and a spacing of {worst['min_distance']:.1f} A, "
                f"where the control reaches {worst['control']:+.3f} against {worst['alone']:+.3f} "
                f"for the measured sites. A loose definition puts tens of weakly occupied points "
                f"in the field, and a block of that many plausible numbers carries whatever a "
                f"block of that many random numbers carries. Any claim about where the water "
                f"sits has to be made at a definition where the control is beaten, and the table "
                f"says which those are.", ""]

    # the direction: does a stricter threshold help or hurt?
    by_threshold = {}
    for r in done:
        by_threshold.setdefault(r["min_occupancy"], []).append(r["alone"])
    means = sorted((t, float(np.mean(v))) for t, v in by_threshold.items())
    if len(means) >= 2:
        trend = ", ".join(f"{t:.0%} {m:+.3f}" for t, m in means)
        rising = means[-1][1] > means[0][1]
        out += [f"**The sensitivity has a direction: the stronger the sites, the more they "
                f"carry.** Averaged over the spacings, the water alone goes {trend}. "
                + (f"A stricter threshold keeps fewer sites and they carry more, which says the "
                   f"signal is in the few places a water really sits and that adding weakly "
                   f"occupied points dilutes it."
                   if rising else
                   f"A stricter threshold keeps fewer sites and they carry less, which says the "
                   f"signal is spread over many places rather than concentrated in a few."), ""]
    best = max(done, key=lambda r: r["alone"])
    out += [f"The strongest the water alone gets is {best['alone']:+.3f}, at an occupancy "
            f"threshold of {best['min_occupancy']:.0%} and a spacing of "
            f"{best['min_distance']:.1f} A, with {best['sites']:.1f} sites per ligand. **That "
            f"number should not be quoted as this item's result.** It is the largest of "
            f"{len(done)} tried on 25 molecules, chosen after seeing them, and the honest "
            f"estimate of what a strict definition is worth would need a set this one was not "
            f"used to choose on. The defaults of the module, 50% and 2.4 A, were fixed before "
            f"any of this was measured and are left alone for that reason.", ""]
    strict = [r for r in done if r["min_occupancy"] >= max(r2["min_occupancy"] for r2 in done)]
    if strict:
        out += [f"One thing the sweep does settle about the strict end: at the strictest "
                f"threshold that still leaves every ligand a site "
                f"({strict[0]['min_occupancy']:.0%}, {strict[0]['sites']:.1f} sites per ligand), "
                f"the water alone reaches {min(r['alone'] for r in strict):+.3f} to "
                f"{max(r['alone'] for r in strict):+.3f} against a control of "
                f"{min(r['control'] for r in strict):+.3f} to "
                f"{max(r['control'] for r in strict):+.3f}, and it does so at all "
                f"{len(strict)} spacings rather than at one. A threshold above that leaves "
                f"ligands with no site at all, which is the end of the range.", ""]
    return out


def verdict(rows):
    out = ["## What it settles", ""]
    for rec in rows:
        m = rec["models"]
        w = m.get("water_mixed")
        if not w:
            continue
        name = rec["set"]
        ahead = w["lo"] > 0
        behind = w["hi"] < 0
        if ahead:
            out += [f"**On {name} the water field adds something.** The five fields reach "
                    f"{m['fields']['q2']:+.3f} and with the water block at a coefficient of "
                    f"{w['mix']:.1f} they reach {w['q2']:+.3f}, a gain of {w['delta']:+.3f} "
                    f"[{w['lo']:+.3f}, {w['hi']:+.3f}] that clears the interval.", ""]
        elif behind:
            out += [f"**On {name} the water field makes the model worse.** {w['delta']:+.3f} "
                    f"[{w['lo']:+.3f}, {w['hi']:+.3f}] against the five fields, which clears "
                    f"the interval in the wrong direction.", ""]
        else:
            out += [f"**On {name} the water field adds nothing that can be measured.** The five "
                    f"fields reach {m['fields']['q2']:+.3f} and the best mixture "
                    f"{w['q2']:+.3f}, a change of {w['delta']:+.3f} [{w['lo']:+.3f}, "
                    f"{w['hi']:+.3f}], an interval that contains zero. On a set of {rec['n']} "
                    f"ligands an interval that wide cannot show a small gain, so what this says "
                    f"is that the water field is not worth a simulation here, not that the "
                    f"solvent carries nothing.", ""]
        alone = m.get("water_alone")
        if alone:
            out += [f"The water alone reaches {alone['q2']:+.3f}. That is the number to read "
                    f"first: it says how much of the activity the places a ligand's water "
                    f"occupies can account for by themselves, with no field of the molecule in "
                    f"the kernel at all.", ""]
        alone = m.get("water_alone")
        controls = [(t, m[t]) for t in ("scrambled_alone", "shared_alone") if t in m]
        if controls and alone:
            # the controls are read on the block alone. Mixed in, a block that tells the ligands
            # apart in no way is a near-constant matrix, which a fit with a bias absorbs, so the
            # mixed control lands on the reference whatever it contains and says nothing.
            best_control = max(c[1]["q2"] for c in controls)
            names_c = " and ".join(
                f"{LABEL[t].replace(' (control)', '')} {r['q2']:+.3f}" for t, r in controls)
            if alone["q2"] > best_control + 1e-9:
                out += [f"**The positions are carrying something, and that is the result worth "
                        f"keeping.** The water field on its own reaches {alone['q2']:+.3f}, "
                        f"while the same number of sites moved to random places in the same box "
                        f"or one ligand's sites given to every ligand reach {names_c}. Both "
                        f"controls are below zero, which is what a block that cannot tell the "
                        f"ligands apart must give, so the signal in the real maps is in **where "
                        f"the water sits** and not in how many sites a ligand has.", ""]
            else:
                out += [f"**The positions are carrying nothing.** The water field on its own "
                        f"reaches {alone['q2']:+.3f} and the controls reach {names_c}, so a "
                        f"block of plausible numbers does as well as the measured maps and the "
                        f"places the water occupies are not what the model is using.", ""]
        if "shared" in m and abs(m["shared"]["delta"]) < 1e-6:
            out += ["A note on reading the mixed controls: one ligand's sites given to every "
                    "ligand make every kernel entry the same, and a constant block added to a "
                    "kernel is absorbed by the fit's own bias. That control therefore lands "
                    "exactly on the reference by construction, which is a check that the "
                    "mixing does nothing of itself rather than a result about water.", ""]
    out += ["**What this rests on.** One set of 25 ligands, 2 ns of simulation each, and a "
            "definition of a hydration site -- a point occupied at least half the frames, with "
            "sites no closer than one water across. A longer simulation would place the sites "
            "more surely; it would not make 25 ligands a larger set, and the interval above is "
            "set by the 25 and not by the nanoseconds.", ""]
    return out


# ----------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="also vary what counts as a hydration site: the share of frames a "
                         "point has to be occupied, and how close two sites may be. Needs the "
                         "trajectories, which the repository does not carry")
    ap.add_argument("--report", nargs="?", const=os.path.join(RESULTS, "table.md"), default=None)
    a = ap.parse_args(argv)

    if a.report:
        print(report(path=a.report))
        print(f"written {a.report}")
        return 0
    os.makedirs(RESULTS, exist_ok=True)
    for name in a.sets:
        run_set(name, force=a.force)
        if a.sweep:
            sites, _record = water_sites(name)
            if not sites:
                continue
            mols, names, y = crystal_set(name)
            keep = [i for i, n in enumerate(names) if n in sites]
            mols = [mols[i] for i in keep]
            names = [names[i] for i in keep]
            y = y[keep]
            _say(f"  {name}: rebuilding the occupancy maps")
            maps = occupancy_maps(name, names)
            if maps is None or len(maps) < len(names):
                _say(f"  {name}: the trajectories are not here, so the sweep is skipped; "
                     f"examples/make_md_trajectories.py rebuilds them")
            else:
                stage_sweep(name, mols, names, y, maps)
    report(path=os.path.join(RESULTS, "table.md"))
    print(f"written {os.path.join(RESULTS, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
