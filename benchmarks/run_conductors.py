#!/usr/bin/env python
"""
From DFT barriers to the activation energy of a conductor, with the error carried through
(the first half of plan item MS8).

The chain
---------
1. The hop model is fitted on the 1,652 hop barriers of nebDFT2k, which come from paths
   relaxed with density functional theory (``run_nebdft.py``).  The model is the random
   forest on the invariant field about the midpoint of a hop: it is the most accurate model
   of that benchmark, and the spread of its trees is the one standard deviation that is
   large where the error is large (``run_nebdft_rule.py``).
2. The model predicts the barrier of every hop of the thirteen disordered conductors of plan
   item M2 (``examples/hops/disordered``), each with a standard deviation.
3. The hops of a structure form a network.  Its *percolation barrier* is the lowest level
   that a lithium ion has to reach to cross the cell: the hops are opened in the order of
   their barriers until a path wraps around the cell (`cmfqsar.hops.percolation_thresholds`).
   It is given for a path in one direction and for paths in all three directions.
4. The standard deviations are carried through by drawing the barriers many times and
   taking the percolation barrier of each draw.
5. The result is compared with measured activation energies of lithium conduction.

How the barriers are drawn
--------------------------
The errors of the model are measured first, on nebDFT2k itself with folds that hold out whole
structures.  Two numbers are taken from there: the factor that puts the spread of the trees
on the right scale, and the share of the squared error that the hops of one structure have
in common.  A draw then gives every structure one common shift and every distinct hop its own
shift, in those proportions, each times the calibrated standard deviation of the hop.  Hops
that are equivalent by symmetry get the same barrier.  A barrier is not drawn below zero.

What the chain leaves out
-------------------------
An activation energy is compared, and not a conductivity: a conductivity needs the attempt
frequency and the number of carriers, which this program does not have.  The model was
fitted on hops into a vacancy of an ordered structure; a conductor has partly occupied sites,
and four of its elements (Ge, La, N, Zr) are not among the element fields of the training
set.  The errors of the model are measured inside its training distribution, so the
intervals are a lower bound of the real uncertainty.

Usage
-----
    python benchmarks/run_nebdft.py          # first: the kernels and rows of the hops
    python benchmarks/run_conductors.py      # this benchmark, then the table
    python benchmarks/run_conductors.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_nebdft as N  # noqa: E402
import run_nebdft_rule as RULE  # noqa: E402

RESULTS = os.path.join(HERE, "results", "conductors")
FOLDS = 10
DRAWS = 400
SEED = 0
#: the standard deviations of the barriers are also made smaller by these factors, to see
#: what error of one barrier would give a useful interval
SCALES = (0.5, 0.25, 0.1)
SCALE_DRAWS = 200
#: measured activation energies of lithium conduction, in eV: the structure of the set, the
#: range of the value, what was measured, the source, and how far the source was checked.
#: "page read" = the page of the source was opened and the value read there; "search result"
#: = the value comes from the summary of a search and the source was not opened.
MEASURED = {
    "cod_7215190": {
        "low": 0.22, "high": 0.28, "sample": "polycrystals, impedance",
        "source": "R. Iwasaki et al., ISSP Activity Report 2018, on the earlier measurements "
                  "(N. Kamaya et al., Nat. Mater. 10, 682 (2011) and later)",
        "checked": "page read"},
    "cod_1545083": {
        "low": 0.41, "high": 0.47, "sample": "single crystals of Ta-doped cubic garnet",
        "source": "a search summary of single-crystal studies of Li6.5La3Zr1.5Ta0.5O12; "
                  "pellets of the same compound: 0.41 eV",
        "checked": "search result"},
    "cod_1552152": {
        "low": 0.41, "high": 0.47, "sample": "the same family, another Ta content",
        "source": "as for cod_1545083; not measured on this composition",
        "checked": "search result"},
    "cod_1570310": {
        "low": 0.356, "high": 0.356, "sample": "recrystallised pellets, impedance",
        "source": "US patent 8,597,838 of the group of Z. Liu et al. (J. Am. Chem. Soc. 135, "
                  "975 (2013))",
        "checked": "search result"},
    "cod_1000437": {
        "low": 0.40, "high": 0.40, "sample": "bulk part of a polycrystal, impedance",
        "source": "S. Stramare, V. Thangadurai and W. Weppner, Chem. Mater. 15, 3974 (2003), "
                  "on Li0.34La0.51TiO2.94 (Y. Inaguma et al., Solid State Commun. 86, 689 "
                  "(1993))",
        "checked": "search result"},
    "cod_4002499": {
        "low": 0.25, "high": 0.25, "sample": "single crystal, in the Li2N planes",
        "source": "U. v. Alpen, A. Rabenau and G. H. Talat, Appl. Phys. Lett. 30, 621 (1977)",
        "checked": "search result"},
}


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# The error of the model inside its training distribution
# --------------------------------------------------------------------------- #
def shared_share(z: np.ndarray, groups: np.ndarray) -> float:
    """The share of the variance of `z` that the members of a group have in common (the
    intraclass correlation from a one-way analysis of variance), between 0 and 1."""
    ids, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    n, k = len(z), len(ids)
    if k < 2 or n <= k:
        return 0.0
    means = np.bincount(inverse, weights=z) / counts
    grand = float(np.mean(z))
    between = float(np.sum(counts * (means - grand) ** 2)) / (k - 1)
    within = float(np.sum((z - means[inverse]) ** 2)) / (n - k)
    size = (n - float(np.sum(counts ** 2)) / n) / (k - 1)      # the effective group size
    var_between = max((between - within) / size, 0.0)
    total = var_between + within
    return float(var_between / total) if total > 0 else 0.0


def error_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = FOLDS,
                seed: int = SEED) -> dict:
    """The forest on folds that hold out whole structures: its error, the factor that puts
    the spread of its trees on the right scale, and the share of the squared error that the
    hops of one structure have in common."""
    rng = np.random.default_rng(seed)
    pred, sd = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for te in N.group_folds(groups, folds, rng):
        tr = np.setdiff1d(np.arange(len(y)), te)
        pred[te], sd[te] = RULE.forest_spread(X, y, tr, te, seed)
    z = (y - pred) / sd
    factor = float(np.sqrt(np.mean(z ** 2)))
    return {"predicted": pred,
            "mae": float(np.mean(np.abs(y - pred))),
            "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
            "mean_sd": float(np.mean(sd)), "factor": factor,
            "shared": shared_share(z / factor, groups),
            "inside_2sd": float(np.mean(np.abs(z / factor) <= 2.0))}


# --------------------------------------------------------------------------- #
# The conductors
# --------------------------------------------------------------------------- #
def conductor_rows(fs_train, say=log):
    """The conductors as a HopSet, the indices of the hops that are distinct by symmetry, and
    the invariant rows of those hops in the layout of the training rows."""
    import run_hops as H
    from cmfqsar.localenv import local_features
    from cmfqsar.periodic import inorganic_fieldset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hc = H.load_disordered()
        cif = [s.name[4:] for s in fs_train.specs if s.is_cif and s.name != "cif:other"]
        fs = inorganic_fieldset(cif, physchem=H.PHYSCHEM, charge_model="oxidation",
                                other=any(s.name == "cif:other" for s in fs_train.specs))
        if [s.name for s in fs.specs] != [s.name for s in fs_train.specs]:
            raise ValueError("the fields of the conductors are not those of the training set")
        crystals = hc.crystals(fs)
        reps = hc.annotate(crystals)
        table = os.path.join(H.DISORDERED, "hops_bvse.csv")
        if os.path.isfile(table):
            read_saddles(table, hc)
        envs = hc.environments(crystals, rows=reps, rho=N.RHO, r_max=N.R_MAX)
        rows = local_features(envs)
    elements = sorted({a for s in hc.structures for a in s.get_chemical_symbols()})
    say(f"  {len(hc.hops)} hops in {len(hc.structures)} conductors, {len(reps)} distinct; "
        f"elements without a field of their own: "
        f"{', '.join(e for e in elements if e not in cif) or 'none'}")
    return hc, list(reps), rows, [e for e in elements if e not in cif]


def read_saddles(path: str, hc) -> int:
    """Put the site energies and the saddle energy of the bond-valence landscape on the hops,
    from the table that ``cmf hops --bvse --out`` wrote; returns how many hops were found.
    `cmfqsar.hops.read_hop_labels` reads the barriers of that table and not these columns."""
    import csv

    names = {n: k for k, n in enumerate(hc.names)}
    index = {(h.structure, h.a, h.b, tuple(h.offset)): h for h in hc.hops}
    n = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (names.get(row["structure"].strip(), -1), int(row["a"]), int(row["b"]),
                   (int(row["offset_x"]), int(row["offset_y"]), int(row["offset_z"])))
            h = index.get(key)
            if h is None or not row["saddle"].strip():
                continue
            h.site_a, h.site_b = float(row["site_a"]), float(row["site_b"])
            h.saddle = float(row["saddle"])
            n += 1
    return n


def spread_values(hc, reps: Sequence[int], values: np.ndarray) -> Dict[int, float]:
    """One value for every hop from the values of the distinct hops."""
    out = {int(r): float(v) for r, v in zip(reps, values)}
    for i, h in enumerate(hc.hops):
        if i not in out and h.representative in out:
            out[i] = out[h.representative]
    return out


def thresholds(hc, k: int, values: Dict[int, float]) -> Tuple[float, float]:
    """The percolation barrier of structure `k` for a path in one direction and for paths in
    all three directions, with `values` as the barrier of every hop."""
    from cmfqsar.hops import structure_percolation

    p = structure_percolation(hc, k, values, relative=False)
    return float(p["E_1D"]), float(p["E_3D"])


def drawn_thresholds(hc, reps: Sequence[int], pred: np.ndarray, sd: np.ndarray,
                     shared: float, draws: int = DRAWS, seed: int = SEED
                     ) -> Dict[int, np.ndarray]:
    """For every structure, the percolation barriers (one direction, three directions) of
    `draws` draws of the barriers: an array of shape (draws, 2).

    `sd` is the calibrated standard deviation of every distinct hop, and `shared` the share
    of the variance that the hops of one structure have in common.
    """
    rng = np.random.default_rng(seed)
    of = np.array([hc.hops[r].structure for r in reps])
    out = {k: np.full((draws, 2), np.nan) for k in range(len(hc.structures))}
    for d in range(draws):
        common = rng.normal(size=len(hc.structures))
        own = rng.normal(size=len(reps))
        z = np.sqrt(shared) * common[of] + np.sqrt(1.0 - shared) * own
        values = spread_values(hc, reps, np.maximum(pred + sd * z, 0.0))
        for k in out:
            if hc.of_structure(k):
                out[k][d] = thresholds(hc, k, values)
    return out


def sensitivity(hc, reps: Sequence[int], pred: np.ndarray, sd: np.ndarray, shared: float,
                full: Dict[int, np.ndarray], scales: Sequence[float] = SCALES,
                draws: int = SCALE_DRAWS) -> List[dict]:
    """How the error of one barrier moves the percolation barrier: for the standard
    deviations as they are (`full`, the draws already made) and multiplied by each of
    `scales`, the mean standard deviation of a barrier and the mean width of the 95 %
    interval of the percolation barrier in one direction, over the structures with hops."""
    def width(drawn):
        w = [np.percentile(v[:, 0], 97.5) - np.percentile(v[:, 0], 2.5)
             for v in drawn.values() if np.isfinite(v[:, 0]).all()]
        return float(np.mean(w)) if w else float("nan")

    out = [{"scale": 1.0, "barrier_sd": float(np.mean(sd)), "interval_width": width(full)}]
    for s in scales:
        drawn = drawn_thresholds(hc, reps, pred, sd * s, shared, draws)
        out.append({"scale": float(s), "barrier_sd": float(np.mean(sd) * s),
                    "interval_width": width(drawn)})
    return out


def _interval(v: np.ndarray) -> List[float]:
    v = v[np.isfinite(v)]
    if not len(v):
        return [float("nan")] * 3
    return [float(np.percentile(v, q)) for q in (2.5, 50.0, 97.5)]


def compare(structures: Sequence[dict]) -> dict:
    """How the percolation barriers stand against the measured activation energies: for the
    forest and for the bond-valence barriers, the mean absolute difference to the middle of
    the measured range, and for the forest how many measured ranges meet its interval."""
    rows_ = [s for s in structures if s.get("measured") and np.isfinite(s["forest"]["one"][1])]
    if not rows_:
        return {"n": 0}
    mid = np.array([0.5 * (s["measured"]["low"] + s["measured"]["high"]) for s in rows_])
    f1 = np.array([s["forest"]["one"][1] for s in rows_])
    f3 = np.array([s["forest"]["three"][1] for s in rows_])
    p1 = np.array([s["forest"]["point"][0] for s in rows_])
    p3 = np.array([s["forest"]["point"][1] for s in rows_])
    out = {"n": len(rows_),
           "mean_of_the_measured": float(np.mean(np.abs(mid - mid.mean()))),
           "predicted_one_direction": float(np.mean(np.abs(p1 - mid))),
           "predicted_three_directions": float(np.mean(np.abs(p3 - mid))),
           "forest_one_direction": float(np.mean(np.abs(f1 - mid))),
           "forest_three_directions": float(np.mean(np.abs(f3 - mid))),
           "interval_meets_measured": int(sum(
               s["forest"]["one"][0] <= s["measured"]["high"]
               and s["forest"]["three"][2] >= s["measured"]["low"] for s in rows_)),
           "mean_interval_width": float(np.mean([s["forest"]["one"][2] - s["forest"]["one"][0]
                                                 for s in rows_]))}
    b = [(s["bond_valence"]["one"], s["bond_valence"]["three"], m)
         for s, m in zip(rows_, mid) if s.get("bond_valence")
         and np.isfinite(s["bond_valence"]["one"])]
    if b:
        out["bond_valence_n"] = len(b)
        out["bond_valence_one_direction"] = float(np.mean([abs(x[0] - x[2]) for x in b]))
        out["bond_valence_three_directions"] = float(np.mean([abs(x[1] - x[2]) for x in b]))
    return out


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def run(results: str = RESULTS, draws: int = DRAWS, folds: int = FOLDS,
        cache: str = N.CACHE, scales: Sequence[float] = SCALES) -> dict:
    import run_hops as H
    from cmfqsar.hops import structure_percolation
    from sklearn.ensemble import RandomForestRegressor

    t0 = time.time()
    hs, doc, _, _ = N.load(N.PREPARED, 0)
    t = hs.targets()
    y = np.asarray(t.y, float)
    groups = np.asarray(t.molecule, int)
    _, rows = N.descriptions(hs, 8, cache, "")
    X = rows["invariant"]
    lengths = np.array([hs.hops[int(r)].length for r in t.atom])
    errors = error_model(X, y, groups, folds)
    bvse_train = np.array([h["bvse"] for r in doc["structures"] for h in r["hops"]],
                          float)[np.asarray(t.atom, int)]
    errors["bond_valence_correlation"] = float(np.corrcoef(errors.pop("predicted"),
                                                           bvse_train)[0, 1])
    log(f"  the forest on folds: mean absolute error {errors['mae']:.3f} eV, factor "
        f"{errors['factor']:.2f}, shared within a structure {errors['shared']:.2f}")
    hc, reps, Xc, unseen = conductor_rows(H.fieldset_for(hs, "file"))
    forest = RandomForestRegressor(n_estimators=N.FOREST_TREES, n_jobs=-1,
                                   random_state=SEED).fit(X, y)
    trees = np.array([e.predict(Xc) for e in forest.estimators_])
    pred = trees.mean(axis=0)
    sd = np.maximum(trees.std(axis=0), 1e-6) * errors["factor"]
    point = spread_values(hc, reps, pred)
    drawn = drawn_thresholds(hc, reps, pred, sd, errors["shared"], draws)
    rep_of = np.array([hc.hops[r].structure for r in reps])
    rep_len = np.array([hc.hops[r].length for r in reps])
    names = {f"cod_{cid}": (name, what) for cid, name, what in H.COD}
    structures = []
    for k, label in enumerate(hc.names):
        srows = hc.of_structure(k)
        mine = rep_of == k
        entry = {"structure": label, "compound": names[label][0],
                 "disorder": names[label][1], "hops": len(srows),
                 "distinct": int(mine.sum()), "measured": MEASURED.get(label)}
        if srows:
            one, three = thresholds(hc, k, point)
            entry.update({
                "mean_predicted": float(pred[mine].mean()),
                "mean_sd": float(sd[mine].mean()),
                "shorter_than_training": float(np.mean(rep_len[mine] < lengths.min())),
                "forest": {"point": [one, three], "one": _interval(drawn[k][:, 0]),
                           "three": _interval(drawn[k][:, 1])}})
            if all(np.isfinite(hc.hops[r].saddle) for r in srows):
                b = structure_percolation(hc, k)
                entry["bond_valence"] = {"one": float(b["E_1D"]), "three": float(b["E_3D"])}
        structures.append(entry)
    labelled = np.array([hc.hops[r].y for r in reps])
    ok = np.isfinite(labelled)
    out = {"training": {"hops": int(len(y)), "structures": int(len(np.unique(groups))),
                        "columns": int(X.shape[1]), "shortest_hop": float(lengths.min()),
                        **errors},
           "conductors": {"structures": len(hc.structures), "hops": len(hc.hops),
                          "distinct": len(reps), "elements_without_a_field": unseen,
                          "mean_predicted": float(pred.mean()),
                          "mean_sd": float(sd.mean()),
                          "bond_valence_correlation": float(
                              np.corrcoef(labelled[ok], pred[ok])[0, 1]) if ok.sum() > 2
                          else float("nan")},
           "draws": draws, "structures": structures, "comparison": compare(structures),
           "sensitivity": sensitivity(hc, reps, pred, sd, errors["shared"], drawn, scales),
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "conductors.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def _num(v: float) -> str:
    return "none" if not np.isfinite(v) else f"{v:.2f}"


def _range(v: Sequence[float]) -> str:
    return "none" if not np.isfinite(v[1]) else f"{v[1]:.2f} [{v[0]:.2f}, {v[2]:.2f}]"


def report(results: str = RESULTS) -> str:
    """The table of the benchmark as text; it is also written to ``table.md``."""
    path = os.path.join(results, "conductors.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    tr, c = d["training"], d["conductors"]
    lines = ["# From DFT barriers to the activation energy of a conductor",
             "",
             f"The model: a random forest on the invariant field about the midpoint of a hop "
             f"({tr['columns']} columns), fitted on {tr['hops']} DFT barriers of "
             f"{tr['structures']} structures (nebDFT2k). On folds that hold out whole "
             f"structures its mean absolute error is {tr['mae']:.3f} eV. The spread of its "
             f"trees is multiplied by {tr['factor']:.2f} to have the right size "
             f"({100 * tr['inside_2sd']:.0f} % of the barriers inside two standard "
             f"deviations), and {100 * tr['shared']:.0f} % of the squared error is common to "
             "the hops of one structure.",
             "",
             f"The conductors: {c['hops']} hops in {c['structures']} structures, "
             f"{c['distinct']} distinct by symmetry. Elements without a field of their own in "
             f"the training set: {', '.join(c['elements_without_a_field']) or 'none'}. The "
             f"predicted barriers have the mean {c['mean_predicted']:.2f} eV and a mean "
             f"standard deviation of {c['mean_sd']:.2f} eV.",
             "",
             "Does the model carry over? No DFT barrier of a conductor is known, so this "
             "cannot be measured directly. One check costs nothing: the predictions should "
             "correlate with the bond-valence barriers of the same hops as they do inside "
             f"the training set. Inside the training set that correlation is r = "
             f"{tr['bond_valence_correlation']:+.2f} (predictions for structures that the "
             f"fit has not seen). For the conductors it is r = "
             f"{c['bond_valence_correlation']:+.2f}.",
             "",
             "A percolation barrier is the lowest level that an ion has to reach to cross the "
             f"cell. An interval is the 95 % interval over {d['draws']} draws of the "
             "barriers; the number before it is the median. All values in eV.",
             "",
             "| Structure | Compound | Hops (distinct) | Share of hops shorter than any "
             "training hop | From the predicted barriers as they are: one / three "
             "directions | With the errors drawn: one direction | With the errors drawn: all "
             "three directions | Bond-valence barriers: one / three directions | Measured "
             "activation energy |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in d["structures"]:
        m = s.get("measured")
        measured = "" if not m else (f"{m['low']:.2f}" if m["low"] == m["high"]
                                     else f"{m['low']:.2f} to {m['high']:.2f}")
        if "forest" not in s:
            lines.append(f"| {s['structure']} | {s['compound']} | 0 | | no hop | no hop | "
                         f"no hop | | {measured} |")
            continue
        b = s.get("bond_valence")
        bv = "" if not b else f"{_num(b['one'])} / {_num(b['three'])}"
        pt = s["forest"]["point"]
        lines.append(f"| {s['structure']} | {s['compound']} | {s['hops']} "
                     f"({s['distinct']}) | {100 * s['shorter_than_training']:.0f} % | "
                     f"{_num(pt[0])} / {_num(pt[1])} | "
                     f"{_range(s['forest']['one'])} | {_range(s['forest']['three'])} | {bv} | "
                     f"{measured} |")
    cmp_ = d["comparison"]
    lines += ["",
              "## Against the measured activation energies",
              ""]
    if cmp_.get("n"):
        lines += [f"{cmp_['n']} structures have a measured value. The mean absolute "
                  "difference between the percolation barrier and the middle of the measured "
                  f"range is {cmp_['predicted_one_direction']:.2f} eV (one direction) and "
                  f"{cmp_['predicted_three_directions']:.2f} eV (three directions) for the "
                  "predicted barriers as they are, and "
                  f"{cmp_['forest_one_direction']:.2f} and "
                  f"{cmp_['forest_three_directions']:.2f} eV for the medians of the draws. "
                  "A prediction of the mean of the measured values for every structure would "
                  f"give {cmp_['mean_of_the_measured']:.2f} eV. For comparison"
                  + (f", the bond-valence barriers give "
                     f"{cmp_['bond_valence_one_direction']:.2f} and "
                     f"{cmp_['bond_valence_three_directions']:.2f} eV "
                     f"({cmp_['bond_valence_n']} structures)"
                     if "bond_valence_n" in cmp_ else ", no bond-valence value is at hand")
                  + ".",
                  f"The interval of the forest, from its lower end for one direction to its "
                  f"upper end for three directions, meets the measured range in "
                  f"{cmp_['interval_meets_measured']} of the {cmp_['n']} structures. The "
                  f"interval for one direction is {cmp_['mean_interval_width']:.2f} eV wide "
                  "on average.",
                  "",
                  "| Structure | Measured, eV | What was measured | Source | How far the "
                  "source was checked |",
                  "|---|---|---|---|---|"]
        for s in d["structures"]:
            m = s.get("measured")
            if m:
                value = f"{m['low']:.2f}" if m["low"] == m["high"] \
                    else f"{m['low']:.2f} to {m['high']:.2f}"
                lines.append(f"| {s['structure']} ({s['compound']}) | {value} | {m['sample']} "
                             f"| {m['source']} | {m['checked']} |")
    if d.get("sensitivity"):
        lines += ["",
                  "## How the error of one barrier moves the result",
                  "",
                  "The standard deviations of the barriers as they are, and made smaller. The "
                  "width is that of the 95 % interval of the percolation barrier in one "
                  "direction, as the mean over the structures.",
                  "",
                  "| Standard deviation of a barrier, mean, eV | Width of the interval of the "
                  "percolation barrier, eV |",
                  "|---|---|"]
        for s in d["sensitivity"]:
            lines.append(f"| {s['barrier_sd']:.3f}"
                         + (" (the model)" if s["scale"] == 1.0 else "")
                         + f" | {s['interval_width']:.2f} |")
    lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args(argv)
    if not a.report:
        run(a.results, draws=a.draws)
    print(report(a.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
