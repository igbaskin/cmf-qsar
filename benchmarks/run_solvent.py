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
#: the definition given as a count of sites rather than a threshold, the same on every set,
#: fixed before the third set was run
COUNT_SITES = 5
COUNT_OCCUPANCY = 0.3
COUNT_DISTANCE = 2.4
#: the set the count was not chosen on, which is the only one that can test the hypothesis
COUNT_HELD_OUT = "hif2a"
#: the floors the rule of the next stage chooses between, in the steps the sweep used
RULE_FLOORS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


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


def sites_at(maps, names, alpha, min_occupancy, min_distance, max_sites=SV.MAX_SITES):
    """{ligand: (centres, coefficients, occupancies)} at one definition of a site.

    Returns (sites, missing): `missing` names the ligands with no point occupied often enough,
    which is how a definition too strict for a set shows itself. `max_sites` keeps only the
    strongest sites, so a definition can be given as a count instead of a threshold.
    """
    sites, missing = {}, []
    for nm in names:
        occ = maps[nm]
        centres, heights = SV.hydration_sites(occ, min_occupancy=float(min_occupancy),
                                              min_distance=float(min_distance),
                                              max_sites=int(max_sites))
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


def stage_count(name, mols, names, y, maps, max_sites=COUNT_SITES,
                min_occupancy=COUNT_OCCUPANCY, min_distance=COUNT_DISTANCE,
                results=RESULTS, seed=SEED, log=_say):
    """The same comparison with the number of sites fixed instead of the threshold.

    The sweep on two sets left a hypothesis rather than a result. What appeared to govern
    whether the water field carries anything was not the occupancy threshold but how many
    strongly occupied sites survive it, because the same threshold leaves a different number of
    sites on two targets: 11.4 on one and 3.6 on the other. If that is right, the definition to
    fix across sets is the count, and this stage fixes it at `max_sites` for every set. The
    threshold is kept low so that it is the count that decides, and the spacing is the module's
    own.
    """
    Kf, refs = {}, []
    for a in ALPHAS:
        Kf[a] = field_kernel(mols, names, a)
        refs.append((best(Kf[a], y), float(a)))
    ref, ref_alpha = max(refs, key=lambda r: r[0][0])
    picked, counts, occupancies = {}, None, None
    for a in ALPHAS:
        sites, miss = sites_at(maps, names, a, min_occupancy, min_distance, max_sites=max_sites)
        if miss:
            log(f"    {len(miss)} ligand(s) with no site at all; the stage is skipped")
            return None
        counts = [len(sites[n][0]) for n in names]
        occupancies = [float(sites[n][2][0]) for n in names]
        Kw = water_kernel(sites, names, a)
        scan = _scan(Kf[a], Kw, y)
        inner = [r for r in scan if 0.0 < r["mix"] < 1.0]
        # every model at its own best attenuation, as the main table does
        rows = {"mixed": max(inner, key=lambda r: r["q2"]) if inner else scan[0],
                "alone": scan[-1],
                "scrambled": _scan(Kf[a], water_kernel(
                    scrambled_sites(sites, names, seed=seed), names, a), y)[-1],
                "shared": _scan(Kf[a], water_kernel(
                    shared_sites(sites, names), names, a), y)[-1]}
        for key, row in rows.items():
            if picked.get(key) is None or row["q2"] > picked[key]["q2"]:
                picked[key] = dict(row, alpha=float(a))
    d, lo, hi = bootstrap_difference(y, picked["mixed"]["pred"], ref[3])
    out = {"set": name, "n": int(len(y)), "max_sites": int(max_sites),
           "min_occupancy": float(min_occupancy), "min_distance": float(min_distance),
           "sites": float(np.mean(counts)), "sites_min": int(min(counts)),
           "sites_max": int(max(counts)), "strongest": float(np.mean(occupancies)),
           "reference": float(ref[0]), "reference_alpha": ref_alpha,
           "mixed": float(picked["mixed"]["q2"]), "mix": float(picked["mixed"]["mix"]),
           "mixed_alpha": picked["mixed"]["alpha"],
           "alone": float(picked["alone"]["q2"]), "alone_alpha": picked["alone"]["alpha"],
           "scrambled": float(picked["scrambled"]["q2"]),
           "shared": float(picked["shared"]["q2"]),
           "delta": d, "lo": lo, "hi": hi}
    log(f"    {name}: {out['sites']:.1f} sites a ligand, the five fields "
        f"{out['reference']:+.3f}, with the water {out['mixed']:+.3f} "
        f"({d:+.3f} [{lo:+.3f}, {hi:+.3f}]), the water alone {out['alone']:+.3f}, "
        f"random sites {out['scrambled']:+.3f}, one shared set {out['shared']:+.3f}")
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, f"{name}_count.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def strictest_floor(maps, names, floors=RULE_FLOORS, min_distance=COUNT_DISTANCE):
    """The strictest occupancy floor at which every ligand still has at least one site.

    This is the rule that picked both definitions where the water field carried anything, 60% on
    one set and 70% on the other. It is stated here on its own, and it reads no activity and no
    q2: it looks only at whether a definition leaves a ligand with nothing. A rule of that kind
    can be applied to a set it was not read from, which a number cannot.

    Returns None when even the loosest floor leaves a ligand without a site.
    """
    out = None
    for floor in sorted(float(f) for f in floors):
        every = all(len(SV.hydration_sites(maps[nm], min_occupancy=floor,
                                           min_distance=float(min_distance))[0]) > 0
                    for nm in names)
        if not every:
            break
        out = floor
    return out


def stage_rule(name, mols, names, y, maps, floors=RULE_FLOORS, min_distance=COUNT_DISTANCE,
               results=RESULTS, seed=SEED, log=_say):
    """The comparison at the definition the rule above picks, measured once.

    The count-based reading of the sweep was refuted. What is left of it is a rule rather than a
    number: both definitions where the water alone carried anything were the strictest floor that
    still left every ligand a site. The floor is chosen first, from the maps alone, and the five
    models are then measured once at that floor. Nothing is searched over after the choice, so
    the row is an estimate on a set the rule was not read from.
    """
    floor = strictest_floor(maps, names, floors, min_distance)
    if floor is None:
        log(f"    {name}: even the loosest floor leaves a ligand with no site; skipped")
        return None
    Kf, refs = {}, []
    for a in ALPHAS:
        Kf[a] = field_kernel(mols, names, a)
        refs.append((best(Kf[a], y), float(a)))
    ref, ref_alpha = max(refs, key=lambda r: r[0][0])
    picked, counts = {}, None
    for a in ALPHAS:
        sites, miss = sites_at(maps, names, a, floor, min_distance)
        if miss:                                  # the floor was chosen so that this cannot happen
            log(f"    {name}: {len(miss)} ligand(s) lost their site at the chosen floor")
            return None
        counts = [len(sites[n][0]) for n in names]
        scan = _scan(Kf[a], water_kernel(sites, names, a), y)
        inner = [r for r in scan if 0.0 < r["mix"] < 1.0]
        rows = {"mixed": max(inner, key=lambda r: r["q2"]) if inner else scan[0],
                "alone": scan[-1],
                "scrambled": _scan(Kf[a], water_kernel(
                    scrambled_sites(sites, names, seed=seed), names, a), y)[-1],
                "shared": _scan(Kf[a], water_kernel(
                    shared_sites(sites, names), names, a), y)[-1]}
        for key, row in rows.items():
            if picked.get(key) is None or row["q2"] > picked[key]["q2"]:
                picked[key] = dict(row, alpha=float(a))
    d, lo, hi = bootstrap_difference(y, picked["mixed"]["pred"], ref[3])
    out = {"set": name, "n": int(len(y)), "floor": float(floor),
           "min_distance": float(min_distance), "floors": [float(f) for f in floors],
           "sites": float(np.mean(counts)), "sites_min": int(min(counts)),
           "sites_max": int(max(counts)),
           "reference": float(ref[0]), "reference_alpha": ref_alpha,
           "mixed": float(picked["mixed"]["q2"]), "mix": float(picked["mixed"]["mix"]),
           "mixed_alpha": picked["mixed"]["alpha"],
           "alone": float(picked["alone"]["q2"]), "alone_alpha": picked["alone"]["alpha"],
           "scrambled": float(picked["scrambled"]["q2"]),
           "shared": float(picked["shared"]["q2"]),
           "delta": d, "lo": lo, "hi": hi}
    log(f"    {name}: the rule picks a floor of {floor:.0%}, {out['sites']:.1f} sites a ligand; "
        f"the five fields {out['reference']:+.3f}, with the water {out['mixed']:+.3f} "
        f"({d:+.3f} [{lo:+.3f}, {hi:+.3f}]), the water alone {out['alone']:+.3f}, "
        f"random sites {out['scrambled']:+.3f}, one shared set {out['shared']:+.3f}")
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, f"{name}_rule.json"), "w") as fh:
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
        if p.endswith(("_sweep.json", "_count.json", "_rule.json")):
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


def load_counts(results=RESULTS):
    out = []
    for p in sorted(glob.glob(os.path.join(results, "*_count.json"))):
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def load_rules(results=RESULTS):
    out = []
    for p in sorted(glob.glob(os.path.join(results, "*_rule.json"))):
        with open(p) as fh:
            out.append(json.load(fh))
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
    out += count_section(load_counts(results), load_sweeps(results))
    out += rule_section(load_rules(results), load_sweeps(results))
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
    span = (f"The change against the reference runs from "
            f"{min(r['delta'] for r in done):+.3f} to {max(r['delta'] for r in done):+.3f}")
    if not ahead and not behind:
        out += [f"**Whether the water field improves the model does not depend on the "
                f"definition: it does not, at any of them.** Of the {len(done)} definitions "
                f"that give every ligand a site, none beats the five fields beyond the interval "
                f"and none falls behind it. {span}, and every interval contains zero.", ""]
    elif ahead:
        strongest = max(ahead, key=lambda r: r["delta"])
        out += [f"**Whether the water field improves the model does depend on the definition.** "
                f"Of the {len(done)} definitions that give every ligand a site, **{len(ahead)} "
                f"beat the five fields beyond the interval** and "
                f"{'none' if not behind else str(len(behind))} fall behind it. {span}. The "
                f"strongest is an occupancy of {strongest['min_occupancy']:.0%} and a spacing "
                f"of {strongest['min_distance']:.1f} A, {strongest['delta']:+.3f} "
                f"[{strongest['lo']:+.3f}, {strongest['hi']:+.3f}], with "
                f"{strongest['sites']:.1f} sites per ligand. Those definitions are the strict "
                f"end of the sweep, and they are the only place in this item where a water "
                f"block has cleared the interval.", ""]
    else:
        out += [f"**The water field makes the model worse at some definitions and at none does "
                f"it help.** Of the {len(done)} that give every ligand a site, none beats the "
                f"five fields beyond the interval and {len(behind)} fall behind it. {span}.", ""]
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
            f"{len(done)} tried on {sweep['n']} molecules, chosen after seeing them, and an "
            f"honest estimate of what a strict definition is worth needs a set it was not "
            f"chosen on. The defaults of the module, 50% and 2.4 A, were fixed before any of "
            f"this was measured and are left as they are for that reason.", ""]
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
    # what the two numbers really stand for: how few sites survive, which is not the same
    # threshold on two targets
    # what the two numbers really stand for: how few sites survive. A threshold is a proxy for
    # that count and the proxy is target-specific, so the count is the quantity to report.
    clear = [r for r in done if r["alone"] > 0.15 and r["alone"] > r["control"] + 0.15]
    marginal = [r for r in done if r not in clear and r["alone"] > 0.05]
    if clear:
        sites = sorted(set(round(r["sites"], 1) for r in clear))
        thresholds = sorted(set(r["min_occupancy"] for r in clear))
        other = [r for r in done if r not in clear]
        out += [f"**What the threshold is standing in for is the number of sites.** The water "
                f"alone clearly carries something at {len(clear)} of the {len(done)} "
                f"definitions, and all of them are at an occupancy of "
                f"{', '.join(f'{t:.0%}' for t in thresholds)}, leaving "
                f"{min(sites):.1f} to {max(sites):.1f} sites per ligand. Every other definition "
                f"leaves {min(r['sites'] for r in other):.1f} to "
                f"{max(r['sites'] for r in other):.1f} and gives at best "
                f"{max(r['alone'] for r in other):+.3f}"
                + (f", with {len(marginal)} of them marginal rather than flat" if marginal
                   else "") + ". A threshold is a proxy for how few sites survive, and the "
                f"proxy is target-specific: the same share of frames leaves a different number "
                f"of sites on a site that holds much ordered water than on one that holds "
                f"little. The count is the quantity to compare across targets, not the "
                f"threshold.", ""]
    elif marginal:
        out += [f"The water alone never clearly carries anything here: its best is "
                f"{max(r['alone'] for r in done):+.3f}, at {len(marginal)} definitions it is "
                f"weakly positive, and at the rest it is flat or negative.", ""]
    return out


def count_section(counts, sweeps=None, held_out=COUNT_HELD_OUT):
    """The table and the reading of the definition given as a count of sites.

    The reading follows the numbers rather than the expectation. It counts the sets where the
    water block beats the five fields beyond the interval and the sets where the water alone
    beats its own random-site control. Where a set also has a sweep it compares the count with
    the best the same set reached at a threshold, because a count that does not reproduce a
    set's own result is refuted by that set and not only by the held-out one.
    """
    if not counts:
        return []
    first = counts[0]
    out = ["## The definition given as a count of sites", "",
           f"The sweep above left a hypothesis and not a result: what appeared to decide "
           f"whether the water field carries anything was **how many strongly occupied sites "
           f"survive**, not the occupancy threshold, because the same threshold leaves a "
           f"different number of sites on two targets. A threshold is therefore a "
           f"target-specific proxy for a count, and a count is what can mean the same thing on "
           f"two sets. This section fixes it: the **{first['max_sites']} most occupied sites a "
           f"ligand**, an occupancy floor of {first['min_occupancy']:.0%} so that the count and "
           f"not the floor decides, and the module's own spacing of "
           f"{first['min_distance']:.1f} A. Everything else is unchanged.", ""]
    if any(r["set"] == held_out for r in counts):
        out += [f"**{held_out}** is the set the count was not chosen on. Its definition was "
                f"written down before the simulations were started, so its row is a test that "
                f"could fail. The other rows are the sets the hypothesis was read from, which "
                f"cannot confirm it -- but they can still refute it, by giving less at the "
                f"fixed count than they gave at their own threshold.", ""]
    out += ["| set | ligands | sites a ligand | the five fields | with the water | "
            "change against the five fields | the water alone | random sites alone | "
            "one shared set alone |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for r in counts:
        label = f"**{r['set']}**" if r["set"] == held_out else r["set"]
        out.append(f"| {label} | {r['n']} | {r['sites']:.1f} | {r['reference']:+.3f} | "
                   f"{r['mixed']:+.3f} | {r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                   f"{r['alone']:+.3f} | {r['scrambled']:+.3f} | {r['shared']:+.3f} |")
    out.append("")

    ahead = [r for r in counts if r["lo"] > 0]
    behind = [r for r in counts if r["hi"] < 0]
    carries = [r for r in counts if r["alone"] > max(0.05, r["scrambled"] + 0.05)]
    held = [r for r in counts if r["set"] == held_out]

    if ahead:
        out += [f"**The water block improves the model beyond the interval on "
                f"{len(ahead)} of the {len(counts)} sets** at this definition "
                f"({', '.join(r['set'] for r in ahead)}), by "
                f"{min(r['delta'] for r in ahead):+.3f} to "
                f"{max(r['delta'] for r in ahead):+.3f}.", ""]
    else:
        out += [f"**The water block improves no set beyond the interval** at this definition. "
                f"The change runs from {min(r['delta'] for r in counts):+.3f} to "
                f"{max(r['delta'] for r in counts):+.3f}, and "
                + (f"on {len(behind)} of the {len(counts)} sets "
                   f"({', '.join(r['set'] for r in behind)}) the water block makes the model "
                   f"**worse** beyond the interval, which no definition tried before this one "
                   f"had done."
                   if behind else "every interval contains zero.")
                + " A fixed count does not make the water field worth adding.", ""]
    if carries:
        out += [f"The water field **on its own** carries something on "
                f"{len(carries)} of the {len(counts)} sets "
                f"({', '.join(r['set'] for r in carries)}): "
                + "; ".join(f"{r['set']} {r['alone']:+.3f} against "
                            f"{r['scrambled']:+.3f} for the same number of sites moved at "
                            f"random" for r in carries) + ".", ""]
    else:
        out += [f"The water field on its own carries nothing on any of the "
                f"{len(counts)} sets at this definition: it reaches "
                f"{min(r['alone'] for r in counts):+.3f} to "
                f"{max(r['alone'] for r in counts):+.3f} against random-site controls of "
                f"{min(r['scrambled'] for r in counts):+.3f} to "
                f"{max(r['scrambled'] for r in counts):+.3f}.", ""]

    if held:
        h = held[0]
        passed = h["alone"] > max(0.05, h["scrambled"] + 0.05)
        out += [f"**What the held-out set says.** On {h['set']} the water alone reaches "
                f"{h['alone']:+.3f} against {h['scrambled']:+.3f} for random sites, and the "
                f"mixture {h['delta']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}] against the five "
                f"fields. "
                + (f"That is the direction the hypothesis predicted, on a set it was not read "
                   f"from, so the count-based statement now has one test behind it that could "
                   f"have failed and did not."
                   if passed else
                   f"The hypothesis predicted that a count of {h['max_sites']} would carry "
                   f"something here, and it does not. The count-based statement is therefore "
                   f"**not supported** on the one set that could test it.") , ""]
    # the count against each set's own best threshold: a reformulation that loses a set's own
    # result is refuted by that set, whether or not the held-out set agrees
    lost = []
    for r in counts:
        sweep = (sweeps or {}).get(r["set"])
        rows = [w for w in (sweep or {}).get("rows", []) if w.get("sites")]
        if not rows:
            continue
        top = max(rows, key=lambda w: w["alone"])
        if top["alone"] > r["alone"] + 0.05:
            lost.append((r, top))
    if lost:
        parts = "; ".join(f"{r['set']} {r['alone']:+.3f} at {r['sites']:.1f} sites against "
                          f"{t['alone']:+.3f} at {t['sites']:.1f} sites "
                          f"({t['min_occupancy']:.0%} occupancy)" for r, t in lost)
        out += [f"**The count does not reproduce what the threshold gave, on "
                f"{len(lost)} of the {len(counts)} sets**: {parts}. That is the clearest "
                f"result here. A fixed count was put forward as the definition the two sets "
                f"agreed on, and it gives less on both of the sets it was read from, so the "
                f"reformulation was a mistake in reading the sweep rather than a property that "
                f"failed to transfer. What the working definitions have in common is not how "
                f"many sites survive but that **every site kept is strongly occupied**; the "
                f"number of survivors is a consequence of that floor and differs between "
                f"targets. Fixing the count forces a ligand with one strongly held water to "
                f"take four weak ones as well, which is why a count behaves like a loose "
                f"threshold.", ""]
    out += ["> **What is fixed and what is not.** The count, the floor and the spacing were "
            "written down before the held-out set was simulated, so its row is an estimate and "
            "not the best of several. Why one target needs a floor of 60% and another 70% is "
            "still unexplained, and a floor chosen per target is not a definition anyone can "
            "fix in advance. The honest reading is in the verdict below.", ""]
    return out


#: a margin over the random-site control below this is reported as small, not as a result
RULE_MARGIN = 0.2


def rule_section(rules, sweeps=None, held_out=COUNT_HELD_OUT):
    """The table and the reading of the strictest-floor rule.

    The reading follows the numbers. It says which floor the rule chose on each set, whether the
    water alone beat its own random-site control there and by how much, and whether the mixture
    cleared the interval; and it separates the set the rule was not read from, which is the only
    one whose row is an estimate. Where a set has a sweep it also checks the rule's own premise,
    that the floor the rule picks is the floor at which that set's water field did best, because
    a rule that does not pick its own evidence is refuted before it is carried anywhere.
    """
    if not rules:
        return []
    out = ["## The rule that was meant to pick both working definitions", "",
           "A count was the wrong reading of the sweep. What was left of it was a **rule** "
           "rather than a number: the definitions where the water alone carried anything looked "
           "like the **strictest occupancy floor that still leaves every ligand at least one "
           "site**. A rule of that form can be carried to a new set, which a number cannot. It "
           "is applied here as a rule: the floor is chosen from the occupancy maps alone, "
           "reading no activity and no q2, and the five models are then measured once at that "
           "floor. Nothing is searched over after the choice.", ""]
    if any(r["set"] == held_out for r in rules):
        out += [f"**{held_out}** is again the set nothing was read from, so its row is an "
                f"estimate. The other rows are the sets the rule was read from: they can only "
                f"show whether the rule picks the definition it was meant to pick, which is a "
                f"check on the rule and not evidence for it.", ""]
    out += ["| set | ligands | floor the rule picks | sites a ligand | the five fields | "
            "with the water | change against the five fields | the water alone | "
            "random sites alone |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for r in rules:
        label = f"**{r['set']}**" if r["set"] == held_out else r["set"]
        out.append(f"| {label} | {r['n']} | {r['floor']:.0%} | {r['sites']:.1f} | "
                   f"{r['reference']:+.3f} | {r['mixed']:+.3f} | "
                   f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['alone']:+.3f} | "
                   f"{r['scrambled']:+.3f} |")
    out.append("")

    # the premise first: a rule that does not pick the definition it was read from is refuted
    # by its own evidence, whatever the held-out set then says
    missed = []
    for r in rules:
        rows = [w for w in (sweeps or {}).get(r["set"], {}).get("rows", []) if w.get("sites")]
        if not rows:
            continue
        top = max(rows, key=lambda w: w["alone"])
        if abs(float(top["min_occupancy"]) - r["floor"]) > 1e-9 \
                and top["alone"] > r["alone"] + 0.05:
            missed.append((r, top))
    if missed:
        parts = "; ".join(f"on {r['set']} it picks {r['floor']:.0%} and the water alone reaches "
                          f"{r['alone']:+.3f} there, while the sweep's best was "
                          f"{top['alone']:+.3f} at {top['min_occupancy']:.0%}"
                          for r, top in missed)
        out += [f"**The rule does not pick the definition it was read from, on "
                f"{len(missed)} of the {len(rules)} sets**: {parts}. That refutes the rule "
                f"before any held-out set is consulted, and it shows where the rule came from: "
                f"the sweep's grid stopped at 70%, so a floor that was the strictest **tried** "
                f"was taken for the strictest **possible**. Carried out properly, the rule "
                f"keeps raising the floor while every ligand still has a site, and it goes past "
                f"the definition the sweep had found.", ""]

    clear = [r for r in rules if r["alone"] > 0.05
             and r["alone"] - r["scrambled"] >= RULE_MARGIN]
    small = [r for r in rules if r["alone"] > 0.05
             and 0.0 < r["alone"] - r["scrambled"] < RULE_MARGIN]
    ahead = [r for r in rules if r["lo"] > 0]
    behind = [r for r in rules if r["hi"] < 0]
    held = [r for r in rules if r["set"] == held_out]

    margins = sorted(r["alone"] - r["scrambled"] for r in clear)
    out += [f"**The water alone clearly beats its own random-site control on "
            f"{len(clear)} of the {len(rules)} sets**"
            + (f" ({', '.join(r['set'] for r in clear)}), by "
               + (f"{margins[0]:+.3f}" if len(margins) == 1 else
                  f"{margins[0]:+.3f} to {margins[-1]:+.3f}") + "." if clear else
               f". It reaches {min(r['alone'] for r in rules):+.3f} to "
               f"{max(r['alone'] for r in rules):+.3f} against random-site controls of "
               f"{min(r['scrambled'] for r in rules):+.3f} to "
               f"{max(r['scrambled'] for r in rules):+.3f}.")
            + (f" On {'one further set' if len(small) == 1 else str(len(small)) + ' further sets'}"
               f" ({', '.join(r['set'] for r in small)}) it is ahead by less than "
               f"{RULE_MARGIN:.2f}, which on sets of this size is not a difference worth "
               f"reading: "
               + "; ".join(f"{r['set']} {r['alone']:+.3f} against {r['scrambled']:+.3f}"
                           for r in small) + "." if small else ""), ""]
    if ahead:
        out += [f"The water block **improves the model beyond the interval** on "
                f"{len(ahead)} of the {len(rules)} sets "
                f"({', '.join(r['set'] for r in ahead)}), by "
                f"{min(r['delta'] for r in ahead):+.3f} to "
                f"{max(r['delta'] for r in ahead):+.3f}.", ""]
    else:
        out += [f"The water block improves no set beyond the interval even here: the change "
                f"runs from {min(r['delta'] for r in rules):+.3f} to "
                f"{max(r['delta'] for r in rules):+.3f}"
                + (f", and on {len(behind)} of the {len(rules)} sets "
                   f"({', '.join(r['set'] for r in behind)}) it makes the model worse beyond "
                   f"the interval." if behind else ", and every interval contains zero."), ""]
    if held:
        h = held[0]
        passed = h["alone"] > max(0.05, h["scrambled"] + 0.05)
        out += [f"**What the held-out set says.** The rule picks a floor of {h['floor']:.0%} on "
                f"{h['set']}, which leaves {h['sites']:.1f} sites a ligand. There the water "
                f"alone reaches {h['alone']:+.3f} against {h['scrambled']:+.3f} for random "
                f"sites, and the mixture {h['delta']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}] "
                f"against the five fields. "
                + (f"The rule therefore holds on a set it was not read from: at the definition "
                   f"it chooses, where the water sits carries information about activity. That "
                   f"is one test that could have failed and did not, and it is the only "
                   f"statement in this item with a held-out measurement behind it."
                   if passed else
                   f"The rule does not hold on the set it was not read from: at the definition "
                   f"it chooses, the water field carries nothing that random sites do not. "
                   f"Both readings of the sweep have now failed a test on a set they were "
                   f"not read from, and what the sweep found on two sets is best treated as a "
                   f"property of those two sets."),
                ""]
    out += ["> **What this does not settle.** A floor chosen by this rule still depends on the "
            "set, so two targets get different definitions and the rule gives no number anyone "
            "can quote. The rule itself was read from two sets and is tested here on one. And "
            "whether the water block is worth adding to a model is answered by the change "
            "column, not by the water alone: that column is where rule 1 applies.", ""]
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
    ap.add_argument("--count", action="store_true",
                    help=f"also measure the definition given as a count of sites "
                         f"({COUNT_SITES} a ligand) rather than as an occupancy threshold, "
                         f"which is the one definition that can mean the same on two targets. "
                         f"Needs the trajectories, as --sweep does")
    ap.add_argument("--rule", action="store_true",
                    help="also measure the definition the strictest-floor rule picks: the "
                         "strictest occupancy floor that still leaves every ligand at least "
                         "one site, chosen from the maps alone and then measured once. Needs "
                         "the trajectories, as --sweep does")
    ap.add_argument("--report", nargs="?", const=os.path.join(RESULTS, "table.md"), default=None)
    a = ap.parse_args(argv)

    if a.report:
        print(report(path=a.report))
        print(f"written {a.report}")
        return 0
    os.makedirs(RESULTS, exist_ok=True)
    for name in a.sets:
        run_set(name, force=a.force)
        if a.sweep or a.count or a.rule:
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
                _say(f"  {name}: the trajectories are not here, so the stage is skipped; "
                     f"examples/make_md_trajectories.py rebuilds them")
            else:
                if a.sweep:
                    stage_sweep(name, mols, names, y, maps)
                if a.count:
                    stage_count(name, mols, names, y, maps)
                if a.rule:
                    stage_rule(name, mols, names, y, maps)
    report(path=os.path.join(RESULTS, "table.md"))
    print(f"written {os.path.join(RESULTS, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
