#!/usr/bin/env python
"""
The spectrum as the bridge between a structure and a spectrum (plan item SP3).

The series is the 189 substituted benzoic acids of plan item SP2.  Every acid has an infrared
spectrum at two levels of theory: ORCA r2SCAN-3c (`examples/spectra/benzoic_acids_ir.json`)
and GFN2-xTB (`examples/spectra/benzoic_acids_ir_xtb.json`).  The lines of the two levels are
at different places, as the lines of a measured spectrum and of a computed one are.

**Part (b), the gate: which structure fits a spectrum.**  The spectra of one level are the
library, one spectrum for every candidate acid.  A spectrum of the other level is the query.
The candidates are ranked by the kernel of `cmfqsar/spectrum.py` between the query and their
library spectra, and the question is where the acid of the query stands.  The composition
cannot answer it: the meta and the para acid of one substituent hold the same atoms.

The two levels differ by more than noise, so the axis of the query is mapped onto the axis of
the library first: not at all, by one factor, or by two factors (one below and one above
2300 cm^-1, where no acid has a line).  The factors, the width of the broadening and the power
of the intensity are fitted on half of the substituents (the calibration acids, for which the
right answer is known) and used on the queries of the other half.  The split is repeated.

A second kind of query is closer to a measurement: the same spectrum as a curve on the grid of
an instrument, broadened to the resolution of the instrument and with noise added.

**Part (a): which part of the structure moves which part of the spectrum.**  A kernel canonical
correlation analysis (`cmfqsar/kcca.py`) with a structural kernel as one view and the kernel
of the ORCA spectra as the other.  The pairs of directions are judged by their correlation on
held-out substituents against a shuffled pairing of the two views, and read on the spectral
side by region, as the co-spectrum of plan item SP1 is read.

Usage
-----
    python benchmarks/run_spectra_bridge.py            # both parts, then the table
    python benchmarks/run_spectra_bridge.py --part b
    python benchmarks/run_spectra_bridge.py --report
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

from cmfqsar import kcca as KC  # noqa: E402
from cmfqsar import spectrum as SP  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
import run_band as B  # noqa: E402
import run_spectra_ir as IR  # noqa: E402

LIBRARY = IR.SPECTRA
QUERIES = os.path.join(ROOT, "examples", "spectra", "benzoic_acids_ir_xtb.json")
RESULTS = os.path.join(HERE, "results", "spectra_bridge")

RANGE = IR.RANGE
#: the widths of the broadening tried, in cm^-1.  The wide ones compare regions of the spectrum
#: and not single lines, which is what two levels of theory can agree on.
WIDTHS = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
POWERS = (1.0, 0.5)
#: where the axis is cut for the map with two factors; no acid has a line near it
SPLIT = 2300.0
#: the factors tried for the whole axis, and for the parts below and above the cut
ONE_FACTOR = tuple(round(v, 3) for v in np.arange(0.960, 1.0601, 0.005))
LOW_FACTOR = tuple(round(v, 3) for v in np.arange(0.970, 1.0001, 0.005))
HIGH_FACTOR = tuple(round(v, 3) for v in np.arange(1.030, 1.0701, 0.005))
REPEATS = 10
#: a measured-like query: the resolution of the instrument (standard deviation of its line
#: shape, cm^-1), the spacing of its grid (cm^-1) and the noise as a part of the largest value
INSTRUMENT = {"sigma": 4.0, "step": 4.0, "noise": 0.02}
CURVE_WIDTHS = (20.0, 50.0, 100.0, 200.0)
CURVE_LOW = tuple(round(v, 3) for v in np.arange(0.975, 0.9951, 0.010))
CURVE_HIGH = tuple(round(v, 3) for v in np.arange(1.040, 1.0701, 0.010))
#: the factors tried for a second density functional, whose lines are near those of the first
DFT_FACTOR = tuple(round(v, 3) for v in np.arange(0.980, 1.0201, 0.005))
QUERIES_DFT = os.path.join(ROOT, "examples", "spectra", "benzoic_acids_ir_b97-3c.json")
#: the levels of theory of the queries.  The library is always the ORCA r2SCAN-3c spectra.
LEVELS = {
    "xtb": {"path": QUERIES, "label": "GFN2-xTB",
            "about": "a faster and rougher quantum method, with its own optimised geometry",
            "one": ONE_FACTOR, "low": LOW_FACTOR, "high": HIGH_FACTOR,
            "curve_widths": CURVE_WIDTHS, "curve_low": CURVE_LOW, "curve_high": CURVE_HIGH},
    "b97-3c": {"path": QUERIES_DFT, "label": "B97-3c",
               "about": "a second density functional, with its own optimised geometry and its "
                        "own scaling factor of the wavenumbers (0.98)",
               "one": DFT_FACTOR, "low": DFT_FACTOR, "high": DFT_FACTOR,
               "curve_widths": (10.0, 20.0, 50.0, 100.0), "curve_low": DFT_FACTOR[::2],
               "curve_high": DFT_FACTOR[::2]},
}
#: the control: GFN2-xTB spectra that start from the ORCA geometries, as the spectra of the
#: second density functional do.  They are computed when the control is run and are not kept
#: in the repository.
CONTROL_CACHE = os.path.join(HERE, "cache", "spectra_bridge")
LEVELS["xtb, ORCA geometry"] = dict(
    LEVELS["xtb"], path=os.path.join(CONTROL_CACHE, "benzoic_acids_ir_xtb_orca_geometry.json"),
    label="GFN2-xTB, optimised from the ORCA geometry")
#: the region in which the carbonyl stretch of either level lies
CARBONYL = (1650.0, 1850.0)
#: part (a)
COMPONENTS = 4
PERMUTATIONS = 100
KCCA_FOLDS = 10
KCCA_SPECTRUM = (5.0, 1.0)            # the setting that plan item SP1 chose most often
MAPS = ("none", "one factor", "two factors")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# The data
# --------------------------------------------------------------------------- #
def read_names() -> Tuple[List[str], dict]:
    """The names of the acids in the order of the SD file, and their table of substituents
    and positions."""
    from rdkit import Chem

    names, table = [], {"substituent": [], "position": [], "formula": [], "sigma": []}
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    for mol in Chem.SDMolSupplier(B.DATA, removeHs=False):
        names.append(mol.GetProp("_Name"))
        table["substituent"].append(mol.GetProp("substituent"))
        table["position"].append(mol.GetProp("position"))
        table["formula"].append(CalcMolFormula(mol))
        table["sigma"].append(float(mol.GetProp("sigma")))
    return names, table


def mapped(spectrum: SP.Spectrum, low: float = 1.0, high: float = 1.0,
           split: float = SPLIT) -> SP.Spectrum:
    """The spectrum with its axis mapped: positions below `split` times `low`, the others
    times `high`."""
    x = np.where(spectrum.x < split, spectrum.x * low, spectrum.x * high)
    return SP.Spectrum(x, spectrum.y, spectrum.kind, spectrum.axis, spectrum.name)


def measured_like(spectrum: SP.Spectrum, sigma: float, step: float, noise: float,
                  rng: np.random.Generator) -> SP.Spectrum:
    """A stick spectrum as an instrument would record it: a curve on an even grid of spacing
    `step`, every line broadened to the standard deviation `sigma`, with Gaussian noise of
    the standard deviation `noise` times the largest value."""
    grid = SP.Grid(RANGE[0], RANGE[1], step)
    y = SP.broaden(spectrum, grid, sigma)
    y = y + rng.normal(0.0, noise * float(np.max(y)) if len(y) else 0.0, len(y))
    return SP.Spectrum(grid.x, y, "curve", spectrum.axis, spectrum.name)


# --------------------------------------------------------------------------- #
# Part (b): where the acid of a query stands among the candidates
# --------------------------------------------------------------------------- #
def ranks(similarity: np.ndarray) -> np.ndarray:
    """The rank of the right candidate for every query: one plus the number of candidates
    with a larger similarity, and half of the number with the same similarity.  Row i of
    `similarity` is query i against every candidate, and candidate i is the right one."""
    own = np.diag(similarity)[:, None]
    larger = (similarity > own).sum(axis=1)
    same = (similarity == own).sum(axis=1) - 1
    return 1.0 + larger + 0.5 * same


def setting_key(kind: str, sigma: float, power: float, low: float, high: float) -> str:
    return f"{kind}|{sigma:g}|{power:g}|{low:.3f}|{high:.3f}"


def unit_rows(F: np.ndarray) -> np.ndarray:
    return F / np.sqrt(np.maximum(np.einsum("ij,ij->i", F, F), 1e-300))[:, None]


def similarities(lib: np.ndarray, qry: np.ndarray, cal: Optional[np.ndarray] = None
                 ) -> np.ndarray:
    """The cosine kernel of every query (rows) against every library spectrum (columns),
    from their broadened spectra on one grid.

    With `cal`, both levels are centred first: the mean normalised spectrum of the library is
    taken from every library spectrum, and the mean normalised spectrum of the calibration
    queries from every query.  All acids of the series share the bands of benzoic acid, and
    the centring leaves what an acid has beyond them.  The mean of the queries uses the
    calibration queries only, because the test queries arrive one at a time.
    """
    L, Q = unit_rows(lib), unit_rows(qry)
    if cal is None:
        return Q @ L.T
    return unit_rows(Q - Q[cal].mean(axis=0)) @ unit_rows(L - L.mean(axis=0)).T


def scores(S: np.ndarray, rows: np.ndarray, twins: Dict[int, int],
           same_formula: Optional[np.ndarray] = None) -> dict:
    """What a set of queries scores: the share ranked first, the share among the first three,
    the mean reciprocal rank, the median rank, and the share of queries whose own acid is
    above the other isomer of the same substituent.  With `same_formula`, also the share
    ranked first when only the candidates with the composition of the query are ranked."""
    r = ranks(S)[rows]
    out = {"top1": float(np.mean(r <= 1.0)), "top3": float(np.mean(r <= 3.0)),
           "mrr": float(np.mean(1.0 / r)), "median": float(np.median(r))}
    pairs = [i for i in rows if int(i) in twins]
    out["isomer"] = (float(np.mean([1.0 if S[i, i] > S[i, twins[int(i)]] else
                                    (0.5 if S[i, i] == S[i, twins[int(i)]] else 0.0)
                                    for i in pairs])) if pairs else float("nan"))
    if same_formula is not None:
        narrowed = ranks(np.where(same_formula, S, -np.inf))[rows]
        out["top1_formula"] = float(np.mean(narrowed <= 1.0))
    return out


def split_substituents(table: dict, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Half of the substituents for the calibration and half for the test; both acids of a
    substituent are on the same side."""
    subs = sorted(set(table["substituent"]))
    rng = np.random.default_rng(seed)
    chosen = set(rng.permutation(subs)[:len(subs) // 2].tolist())
    cal = np.array([i for i, s in enumerate(table["substituent"]) if s in chosen], int)
    test = np.array([i for i, s in enumerate(table["substituent"]) if s not in chosen], int)
    return cal, test


def twin_rows(table: dict) -> Dict[int, int]:
    """For every acid whose substituent occurs at both positions, the row of the other acid."""
    pairs = B.twin_pairs(table)
    out = {}
    for m, p in pairs:
        out[int(m)], out[int(p)] = int(p), int(m)
    return out


def factor_grid(kind: str, one=ONE_FACTOR, low=LOW_FACTOR, high=HIGH_FACTOR):
    """The pairs of factors (below the cut, above the cut) of one map of the axis."""
    if kind == "none":
        return [(1.0, 1.0)]
    if kind == "one factor":
        return [(s, s) for s in one]
    return [(lo, hi) for lo in low for hi in high]


def halves(cal: np.ndarray, table: dict, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """The calibration acids divided in two by their substituents."""
    subs = sorted({table["substituent"][int(i)] for i in cal})
    first = set(np.random.default_rng(seed).permutation(subs)[:len(subs) // 2].tolist())
    one = np.array([i for i in cal if table["substituent"][int(i)] in first], int)
    two = np.array([i for i in cal if table["substituent"][int(i)] not in first], int)
    return one, two


def best_settings(library: Sequence[SP.Spectrum], queries: Sequence[SP.Spectrum],
                  splits: Sequence[Tuple[np.ndarray, np.ndarray]], twins: Dict[int, int],
                  same_formula: np.ndarray, table: dict, kinds: Sequence[str] = MAPS,
                  centred: Sequence[bool] = (False, True), widths=WIDTHS, powers=POWERS,
                  one=ONE_FACTOR, low=LOW_FACTOR, high=HIGH_FACTOR) -> Dict[str, List[dict]]:
    """For every map of the axis, with and without the centring, and for every split: the
    setting with the highest mean reciprocal rank on the calibration queries, and what it
    scores on the test queries.

    Every setting is scored as soon as its spectra are on the grid, so nothing but the best of
    each split is kept.  The key of a row is the map, with ", centred" where the levels are
    centred.

    A map with more factors fits the calibration queries better whether or not it is better.
    Each row therefore also carries ``inner``, an estimate from the calibration acids alone of
    what the map scores on new queries: the calibration acids are divided in two, the setting
    is chosen on one half and scored on the other, both ways, and the two scores are averaged.
    :func:`chosen_map` uses it to choose the map without the test queries.
    """
    names = [kind + (", centred" if c else "") for kind in kinds for c in centred]
    best: Dict[str, List[Optional[dict]]] = {nm: [None] * len(splits) for nm in names}
    half: Dict[str, List[list]] = {nm: [[(-1.0, 0.0), (-1.0, 0.0)] for _ in splits]
                                   for nm in names}
    inner = [halves(cal, table, k) for k, (cal, _test) in enumerate(splits)]
    for sigma in widths:
        grid = SP.Grid.for_width(RANGE[0], RANGE[1], sigma)
        for power in powers:
            lib = SP.feature_matrix(library, grid, sigma, power)
            for kind in kinds:
                for lo, hi in factor_grid(kind, one, low, high):
                    qry = SP.feature_matrix([mapped(q, lo, hi) for q in queries], grid, sigma,
                                            power)
                    plain = similarities(lib, qry) if False in centred else None
                    for k, (cal, test) in enumerate(splits):
                        for c in centred:
                            S = similarities(lib, qry, cal) if c else plain
                            reciprocal = 1.0 / ranks(S)
                            crit = float(np.mean(reciprocal[cal]))
                            name = kind + (", centred" if c else "")
                            if best[name][k] is None or crit > best[name][k]["criterion"]:
                                best[name][k] = dict(
                                    scores(S, test, twins, same_formula), criterion=crit,
                                    setting=setting_key(kind, sigma, power, lo, hi))
                            a, b = inner[k]
                            ca, cb = float(np.mean(reciprocal[a])), float(np.mean(reciprocal[b]))
                            if ca > half[name][k][0][0]:
                                half[name][k][0] = (ca, cb)
                            if cb > half[name][k][1][0]:
                                half[name][k][1] = (cb, ca)
    for name in names:
        for k in range(len(splits)):
            best[name][k]["inner"] = 0.5 * (half[name][k][0][1] + half[name][k][1][1])
    return best


def chosen_map(best: Dict[str, List[dict]], names: Sequence[str]) -> List[dict]:
    """For every split, the row of the map with the highest ``inner`` estimate among
    `names`; the row gains ``map``, the name of the map that was chosen."""
    out = []
    for k in range(len(best[names[0]])):
        name = max(names, key=lambda nm: best[nm][k]["inner"])
        out.append(dict(best[name][k], map=name))
    return out


def band_scores(library: Sequence[SP.Spectrum], queries: Sequence[SP.Spectrum],
                cal: np.ndarray, test: np.ndarray, twins: Dict[int, int]) -> dict:
    """The reference with one number: the candidates ranked by how near their carbonyl band
    is to the carbonyl band of the query, after the bands of the queries are put on the scale
    of the library by a linear regression fitted on the calibration acids.  The carbonyl band
    is taken as the strongest line of the carbonyl region."""
    def band(s):
        part = s.within(*CARBONYL)
        return float(part.x[int(np.argmax(part.y))]) if len(part) else float("nan")

    lib = np.array([band(s) for s in library])
    qry = np.array([band(s) for s in queries])
    ok = cal[np.isfinite(lib[cal]) & np.isfinite(qry[cal])]
    if np.ptp(qry[ok]) > 0:
        slope, intercept = np.polyfit(qry[ok], lib[ok], 1)
    else:                                   # one value only: a shift is all that can be fitted
        slope, intercept = 1.0, float(np.mean(lib[ok] - qry[ok]))
    moved = slope * qry + intercept
    return scores(-np.abs(moved[:, None] - lib[None, :]), test, twins)


def formula_scores(table: dict) -> dict:
    """What the composition can do: candidates with the formula of the query cannot be told
    apart, so the right one is first with a chance of one over their number."""
    formulas = list(table["formula"])
    size = np.array([formulas.count(f) for f in formulas], float)
    return {"top1": float(np.mean(1.0 / size)), "median": float(np.median((size + 1) / 2)),
            "isomer": 0.5, "largest_group": int(size.max()),
            "share_unique": float(np.mean(size == 1))}


def summarise(rows: Sequence[dict]) -> dict:
    """The mean and the standard deviation over the splits of every score, and the settings
    chosen, most frequent first."""
    keys = [k for k in rows[0] if k not in ("criterion", "setting", "inner", "map")]
    out = {k: {"mean": float(np.nanmean([r[k] for r in rows])),
               "sd": float(np.nanstd([r[k] for r in rows]))} for k in keys}
    if "setting" in rows[0]:
        chosen = [r["setting"] for r in rows]
        out["chosen"] = sorted(set(chosen), key=lambda s: (-chosen.count(s), s))[:2]
    if "map" in rows[0]:
        maps = [r["map"] for r in rows]
        out["maps"] = {m: maps.count(m) for m in sorted(set(maps))}
    return out


def part_b(repeats: int = REPEATS, results: str = RESULTS, level: str = "xtb") -> dict:
    t0 = time.time()
    cfg = LEVELS[level]
    names, table = read_names()
    library = IR.read_spectra(LIBRARY, names)
    queries = IR.read_spectra(cfg["path"], names)
    twins = twin_rows(table)
    n = len(names)
    formulas = np.array(table["formula"])
    same = formulas[:, None] == formulas[None, :]
    splits = [split_substituents(table, r) for r in range(int(repeats))]

    def carbonyl(s):
        part = s.within(*CARBONYL)
        return float(part.x[int(np.argmax(part.y))])

    shift = np.array([carbonyl(q) - carbonyl(l) for q, l in zip(queries, library)])
    high = np.array([float(q.x[-1]) - float(l.x[-1]) for q, l in zip(queries, library)])
    out = {"level": level, "query": cfg["label"], "about": cfg["about"],
           "n": n, "repeats": int(repeats), "split": SPLIT, "widths": list(WIDTHS),
           "powers": list(POWERS), "instrument": dict(INSTRUMENT),
           "carbonyl_shift": {"mean": float(shift.mean()), "sd": float(shift.std())},
           "highest_line_shift": {"mean": float(high.mean()), "sd": float(high.std())},
           "formula": formula_scores(table), "chance_top1": 1.0 / n,
           "n_twins": len(twins), "rows": {}}
    log(f"=== part (b): {n} acids, the queries are {cfg['label']} spectra and the library is "
        f"ORCA r2SCAN-3c; "
        f"the carbonyl band of a query is {shift.mean():+.1f} cm^-1 from that of its library "
        f"spectrum (sd {shift.std():.1f}), its highest line {high.mean():+.1f} "
        f"(sd {high.std():.1f})")

    centred_maps = [m + ", centred" for m in MAPS]
    best = best_settings(library, queries, splits, twins, same, table, one=cfg["one"],
                         low=cfg["low"], high=cfg["high"])
    for name, rows in best.items():
        out["rows"][name] = summarise(rows)
    out["rows"]["chosen"] = summarise(chosen_map(best, centred_maps))
    log(f"  sticks, {time.time() - t0:.0f} s")
    out["rows"]["band"] = summarise([band_scores(library, queries, cal, test, twins)
                                     for cal, test in splits])
    # a measured-like query: a curve on the grid of an instrument, with noise
    t1 = time.time()
    rng = np.random.default_rng(0)
    curves = [measured_like(q, INSTRUMENT["sigma"], INSTRUMENT["step"], INSTRUMENT["noise"],
                            rng) for q in queries]
    measured = best_settings(library, curves, splits, twins, same, table, centred=(True,),
                             widths=cfg["curve_widths"], one=cfg["one"],
                             low=cfg["curve_low"], high=cfg["curve_high"])
    out["rows"]["measured-like"] = summarise(chosen_map(measured, centred_maps))
    log(f"  measured-like curves, {time.time() - t1:.0f} s")
    # the other direction: an ORCA spectrum as the query, the GFN2-xTB spectra as the library
    back = best_settings(queries, library, splits, twins, same, table, centred=(True,),
                         one=tuple(round(1.0 / v, 3) for v in cfg["one"]),
                         low=tuple(round(1.0 / v, 3) for v in cfg["low"][::2]),
                         high=tuple(round(1.0 / v, 3) for v in cfg["high"][::2]))
    out["rows"]["reverse"] = summarise(chosen_map(back, centred_maps))
    for name, row in out["rows"].items():
        log(f"  {name}: first in {100 * row['top1']['mean']:.1f} % of the test queries (sd "
            f"{100 * row['top1']['sd']:.1f}), median rank {row['median']['mean']:.1f}, the "
            f"right isomer above the other in {100 * row['isomer']['mean']:.1f} %"
            + (f", first among the candidates of its composition in "
               f"{100 * row['top1_formula']['mean']:.1f} %" if "top1_formula" in row else ""))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def part_control(repeats: int = REPEATS, results: str = RESULTS) -> Optional[dict]:
    """The control for the start geometry.  The spectra of the second density functional
    start from the ORCA geometries of the library, and the GFN2-xTB spectra start from the
    force-field geometries.  Here GFN2-xTB starts from the ORCA geometries too, and the same
    protocol is run.  Returns None when xtb is not installed."""
    from cmfqsar import xtb

    sys.path.insert(0, os.path.join(ROOT, "examples"))
    import make_band_examples as M

    path = LEVELS["xtb, ORCA geometry"]["path"]
    if not os.path.isfile(path):
        try:
            xtb.find_xtb(None)
        except xtb.XtbError:
            log("  the control needs xtb, which is not installed here; it is left out")
            return None
        os.makedirs(CONTROL_CACHE, exist_ok=True)
        failed = M.write_infrared_xtb(M.BANDS, path, jobs=8, log=log,
                                      workdir=os.path.join(CONTROL_CACHE, "xtb_work"))
        if failed:
            raise SystemExit(f"{failed} acids failed in the control run of xtb")
    full = part_b(repeats, results, "xtb, ORCA geometry")
    return {"query": full["query"], "repeats": full["repeats"],
            "carbonyl_shift": full["carbonyl_shift"],
            "rows": {k: full["rows"][k] for k in ("chosen", "none, centred", "measured-like")},
            "seconds": full["seconds"]}


# --------------------------------------------------------------------------- #
# Part (a): kernel canonical correlation of a structural kernel and the spectrum kernel
# --------------------------------------------------------------------------- #
def substituent_folds(table: dict, folds: int, seed: int) -> List[np.ndarray]:
    """Rows of the acids of every fold; a fold holds out whole substituents."""
    subs = sorted(set(table["substituent"]))
    order = np.random.default_rng(seed).permutation(len(subs))
    part = {subs[int(i)]: k % folds for k, i in enumerate(order)}
    which = np.array([part[s] for s in table["substituent"]])
    return [np.flatnonzero(which == k) for k in range(folds)]


def held_out_pairs(K: np.ndarray, L: np.ndarray, ridge: float, folds: Sequence[np.ndarray],
                   components: int, reference: Optional[KC.KCCAFit] = None):
    """The canonical variates of every acid from the directions fitted without its
    substituent, and the correlation of the two variates of every pair in every fold.

    The sign of a pair of directions is free.  With `reference`, the directions of every fold
    are turned so that their variates on the training acids correlate positively with the
    variates of the reference fit, which makes the held-out variates of the folds one column.
    """
    n = len(K)
    U, V = np.full((n, components), np.nan), np.full((n, components), np.nan)
    rho = []
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        fit = KC.kcca(K[np.ix_(tr, tr)], L[np.ix_(tr, tr)], ridge)
        r = min(components, fit.n_components)
        u = fit.project_a(K[np.ix_(te, tr)])[:, :r]
        v = fit.project_b(L[np.ix_(te, tr)])[:, :r]
        rho.append([KC._pearson(u[:, i], v[:, i]) for i in range(r)] + [np.nan] * (components - r))
        if reference is not None:
            for i in range(r):
                if np.corrcoef(fit.u[:, i], reference.u[tr, i])[0, 1] < 0:
                    u[:, i], v[:, i] = -u[:, i], -v[:, i]
        U[te, :r], V[te, :r] = u, v
    return U, V, np.array(rho, float)


def choose_ridge(K, L, folds, components, grid=KC.RIDGE_GRID) -> Tuple[float, Dict[str, float]]:
    """The ridge with the highest mean held-out correlation over the leading pairs."""
    seen = {}
    for c in grid:
        seen[float(c)] = float(np.nanmean(held_out_pairs(K, L, float(c), folds,
                                                         components)[2]))
    best = max(seen, key=seen.get)
    return best, {f"{k:g}": v for k, v in seen.items()}


def region_reading(Z: np.ndarray, x: np.ndarray, K: np.ndarray, ridge: float,
                   folds: Sequence[np.ndarray], reference: KC.KCCAFit,
                   components: int) -> List[List[float]]:
    """For every region of the spectrum and every pair: the correlation, over held-out acids,
    between what the region adds to the spectral variate and the structural variate.

    The kernel of the spectra is ``Z Z^T``, so a spectral direction with the dual coefficients
    ``b`` has the weight ``w = Z_c^T b`` over the axis, and the spectral variate of an acid is
    the sum over the axis of ``w`` times its centred intensity.  The sum over one region is
    what that region adds.
    """
    L = Z @ Z.T
    n = len(Z)
    masks = [(x >= lo) & (x <= hi) for lo, hi, _label in IR.REGIONS]
    added = np.zeros((len(masks), components, n))
    U = np.zeros((n, components))
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        fit = KC.kcca(K[np.ix_(tr, tr)], L[np.ix_(tr, tr)], ridge)
        r = min(components, fit.n_components)
        mean = Z[tr].mean(axis=0)
        b = fit.b[:, :r]
        w = (Z[tr] - mean).T @ b                               # weight over the axis
        u = fit.project_a(K[np.ix_(te, tr)])[:, :r]
        for i in range(r):
            sign = -1.0 if np.corrcoef(fit.u[:, i], reference.u[tr, i])[0, 1] < 0 else 1.0
            U[te, i] = sign * u[:, i]
            for k, cols in enumerate(masks):
                added[k, i, te] = sign * ((Z[np.ix_(te, cols)] - mean[cols]) @ w[cols, i])
    return [[float(np.corrcoef(added[k, i], U[:, i])[0, 1]) if np.std(added[k, i]) > 0 else 0.0
             for i in range(components)] for k in range(len(masks))]


def part_a(results: str = RESULTS, permutations: int = PERMUTATIONS) -> dict:
    t0 = time.time()
    mols, names, fields5, targets, _fs5, table = B.read(B.DATA, structures=B.STRUCTURES)
    spectra = IR.read_spectra(LIBRARY, names)
    spectra = [spectra[int(m)] for m in targets.molecule]
    sigma_w, power = KCCA_SPECTRUM
    grid = SP.Grid.for_width(RANGE[0], RANGE[1], sigma_w)
    F = SP.feature_matrix(spectra, grid, sigma_w, power)
    Z = F / np.sqrt(np.maximum(np.einsum("ij,ij->i", F, F), 1e-300))[:, None]
    L = Z @ Z.T
    qmols = B.with_quantum_charges(mols, names, B.QUANTUM)
    fs_q = FieldSet.create(physchem=("electrostatic",), charge_model="file")
    fields_q = [build_fields(m, fs_q, name=n) for m, n in zip(qmols, names)]
    views = {"five": ("the five fields, Gasteiger charges",
                      IR.structure_kernels(fields5, targets)),
             "quantum": ("the electrostatic field alone, quantum charges",
                         IR.structure_kernels(fields_q, targets))}
    band = np.asarray(targets.y, float)
    sigma = np.asarray(table["sigma"], float)
    para = np.array([p == "para" for p in table["position"]], float)
    heavy = np.array([m.GetNumHeavyAtoms() for m in mols], float)[
        np.asarray(targets.molecule, int)]
    lines = np.array([float(np.sum((s.x >= RANGE[0]) & (s.x <= RANGE[1]))) for s in spectra])
    props = {"the Hammett constant": sigma, "the position of the carbonyl band": band,
             "para (1) or meta (0)": para, "the number of heavy atoms": heavy,
             "the number of lines of the spectrum": lines}
    folds = substituent_folds(table, KCCA_FOLDS, 0)
    out = {"n": len(band), "folds": KCCA_FOLDS, "permutations": int(permutations),
           "components": COMPONENTS, "spectrum": {"width": sigma_w, "power": power},
           "regions": [r[2] for r in IR.REGIONS], "properties": list(props), "views": {}}
    rng = np.random.default_rng(0)
    for key, (label, kernels) in views.items():
        widths = {}
        for width, K in kernels.items():
            ridge, seen = choose_ridge(K, L, folds, COMPONENTS)
            _U, _V, rho = held_out_pairs(K, L, ridge, folds, COMPONENTS)
            widths[width] = {"ridge": ridge, "rho": [float(v) for v in np.nanmean(rho, axis=0)],
                             "ridges": seen}
        best = max(widths, key=lambda wdt: widths[wdt]["rho"][0])
        K, ridge = kernels[best], widths[best]["ridge"]
        ref = KC.kcca(K, L, ridge)
        U, V, rho = held_out_pairs(K, L, ridge, folds, COMPONENTS, reference=ref)
        null = []
        for _ in range(int(permutations)):
            p = rng.permutation(len(band))
            null.append(np.nanmean(held_out_pairs(K, L[np.ix_(p, p)], ridge, folds,
                                                  COMPONENTS)[2], axis=0))
        null = np.array(null)
        held = np.nanmean(rho, axis=0)
        q95 = np.nanpercentile(null, 95, axis=0)
        reading = {name: {"structure": [float(np.corrcoef(U[:, i], v)[0, 1])
                                        for i in range(COMPONENTS)],
                          "spectrum": [float(np.corrcoef(V[:, i], v)[0, 1])
                                       for i in range(COMPONENTS)]}
                   for name, v in props.items()}
        out["views"][key] = {
            "label": label, "widths": widths, "width": best, "ridge": ridge,
            "trained_rho": [float(np.corrcoef(ref.u[:, i], ref.v[:, i])[0, 1])
                            for i in range(COMPONENTS)],
            "held_out_rho": [float(v) for v in held],
            "held_out_sd": [float(v) for v in np.nanstd(rho, axis=0)],
            "null_mean": [float(v) for v in np.nanmean(null, axis=0)],
            "null_q95": [float(v) for v in q95],
            "above": [int(i + 1) for i in range(COMPONENTS) if held[i] > q95[i]],
            "reading": reading,
            "regions": region_reading(Z, grid.x, K, ridge, folds, ref, COMPONENTS)}
        log(f"  part (a), {label}: width {best}, ridge {ridge:g}; held-out correlations "
            + ", ".join(f"{v:+.2f}" for v in held) + "; shuffled pairing, 95 %: "
            + ", ".join(f"{v:+.2f}" for v in q95) + f"  ({time.time() - t0:.0f} s)")
    out["seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- #
# Running and the report
# --------------------------------------------------------------------------- #
def run(part: str = "all", repeats: int = REPEATS, results: str = RESULTS) -> dict:
    os.makedirs(results, exist_ok=True)
    path = os.path.join(results, "bridge.json")
    out = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            out = json.load(fh)
    if part in ("all", "b"):
        out["b"] = part_b(repeats, results, "xtb")
    if part in ("all", "dft") and os.path.isfile(QUERIES_DFT):
        out["b_dft"] = part_b(repeats, results, "b97-3c")
    if part in ("all", "control"):
        control = part_control(repeats, results)
        if control is not None:
            out["b_control"] = control
    if part in ("all", "a"):
        out["a"] = part_a(results)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def _pct(cell: dict) -> str:
    return f"{100 * cell['mean']:.1f} ({100 * cell['sd']:.1f})"


def shown(key: str) -> str:
    """A setting of part (b) in words."""
    kind, sigma, power, lo, hi = key.split("|")
    text = f"{float(sigma):g} cm^-1, power {float(power):g}"
    if kind == "one factor":
        return text + f", factor {float(lo):.3f}"
    if kind == "two factors":
        return text + f", factors {float(lo):.3f} and {float(hi):.3f}"
    return text


def report(results: str = RESULTS) -> str:
    """The table of the benchmark as text; it is also written to ``table.md``."""
    path = os.path.join(results, "bridge.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    lines = ["# The spectrum as the bridge between a structure and a spectrum (plan item SP3)",
             ""]
    if "b" in d:
        lines += _report_b(d["b"])
    if "b_dft" in d:
        lines += _report_b(d["b_dft"], again=True)
    if "a" in d:
        lines += _report_a(d["a"])
    if "a" in d and "b" in d:
        lines += _verdict(d)
    text = "\n".join(lines)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


B_ROWS = (("none", "spectrum kernel, the axis as it is"),
          ("one factor", "spectrum kernel, the axis times one factor"),
          ("two factors", "spectrum kernel, one factor below and one above 2300 cm^-1"),
          ("none, centred", "centred spectra, the axis as it is"),
          ("one factor, centred", "centred spectra, the axis times one factor"),
          ("two factors, centred", "centred spectra, one factor below and one above "
                                   "2300 cm^-1"),
          ("chosen", "centred spectra, the map of the axis chosen inside the calibration "
                     "acids"),
          ("measured-like", "the same, the query as a measured-like curve"),
          ("reverse", "the same in the other direction: an ORCA spectrum as the query, the "
                      "GFN2-xTB spectra as the library"),
          ("band", "the position of the carbonyl band alone, after a linear regression "
                   "between the two levels"))


def _report_b(b: dict, again: bool = False) -> List[str]:
    f = b["formula"]
    query = b.get("query", "GFN2-xTB")
    about = b.get("about", "a faster and rougher quantum method, with its own optimised "
                           "geometry")
    places = ("The two levels put the lines at different places: the carbonyl band of a query "
              f"is {b['carbonyl_shift']['mean']:+.1f} cm^-1 from that of its library spectrum "
              f"(standard deviation {b['carbonyl_shift']['sd']:.1f}), and its highest line, "
              "which is the O-H stretch in most acids, is "
              f"{b['highest_line_shift']['mean']:+.1f} cm^-1 from it (standard deviation "
              f"{b['highest_line_shift']['sd']:.1f}).")
    head = ["| how the candidates are ranked | first, % of the queries | among the first "
            "three, % | median rank | the right isomer above the other isomer of the same "
            "substituent, % | first among the candidates with the composition of the query, "
            "% | settings chosen most often |",
            "|---|---:|---:|---:|---:|---:|---|"]
    if again:
        return ([f"## Part (b) again: a spectrum from {query} as the query", "",
                 "The same library and the same protocol.  A query is now the spectrum of one "
                 f"acid from {query}, {about}.  {places}", ""] + head + _rows_b(b))
    lines = ["## Part (b): which acid fits a spectrum", "",
             f"The {b['n']} substituted benzoic acids of plan item SP2.  The library holds one "
             "infrared spectrum for every acid, from ORCA (r2SCAN-3c, wavenumbers times "
             f"0.97).  A query is the spectrum of one acid from {query}, {about}.  The "
             "candidates are ranked by the kernel between the query and their library "
             "spectra.  The rank of the acid of the query is 1 when it is first.", "",
             places + "  The axis of the query is therefore mapped first: not at all, by one "
             f"factor, or by one factor below {b['split']:g} cm^-1 and another above it.", "",
             "All acids share the bands of benzoic acid.  In the rows with centred spectra "
             "the mean normalised spectrum of each level is taken from its spectra before "
             "they are compared, which leaves what an acid has beyond those bands.", "",
             "A map with more factors fits the calibration acids better whether or not it "
             "is better.  One row therefore chooses the map inside the calibration acids: "
             "they are divided in two, every map is fitted on one half and scored on the "
             "other, and the map with the best score is taken.  This is the row that the "
             "reading uses.", "",
             "The factors, the width of the broadening (one of "
             + ", ".join(f"{w:g}" for w in b["widths"]) + " cm^-1), the power of the "
             "intensity and the mean spectrum of the queries are fitted on the acids of half "
             "of the substituents, where the right answer is used, and applied to the "
             f"queries of the other half.  The split is repeated {b['repeats']} times; a "
             "number is the mean over the splits, with the standard deviation over the "
             "splits in brackets.", "",
             "A measured-like query is the same spectrum as a curve on a grid of "
             f"{b['instrument']['step']:g} cm^-1, with every line broadened to a standard "
             f"deviation of {b['instrument']['sigma']:g} cm^-1 and with noise of "
             f"{100 * b['instrument']['noise']:g} % of the largest value.", ""]
    lines += head + _rows_b(b)
    lines += [f"{100 * (1 - f['share_unique']):.0f} % of the acids have the composition "
              "of at least one other acid of the series (the meta and the para acid of one "
              f"substituent, and other isomers; the largest group has {f['largest_group']} "
              "acids).  The composition alone therefore puts the right acid first in "
              f"{100 * f['top1']:.1f} % of the cases on average, by drawing among the acids "
              "that it cannot tell apart.  The last number column is the practical case: the "
              "composition is known, and the spectrum has to choose among the isomers.", ""]
    return lines


def _rows_b(b: dict) -> List[str]:
    """The rows of the table of part (b) for one level of the queries, and an empty line."""
    f = b["formula"]
    query = b.get("query", "GFN2-xTB")
    lines = [f"| by chance | {100 * b['chance_top1']:.1f} | | {(b['n'] + 1) / 2:.0f} | 50.0 "
             "| | |",
             f"| by the composition (the molecular formula) | {100 * f['top1']:.1f} | | "
             f"{f['median']:.1f} | 50.0 | {100 * f['top1']:.1f} | |"]
    for key, label in B_ROWS:
        r = b["rows"][key]
        narrowed = _pct(r["top1_formula"]) if "top1_formula" in r else ""
        chosen = "; ".join(shown(k) for k in r.get("chosen", []))
        if "maps" in r:
            chosen = ("map chosen: " + ", ".join(
                f"{m.replace(', centred', '')} in {c} of {b['repeats']} splits"
                for m, c in r["maps"].items()) + "; " + chosen)
        lines.append(f"| {label.replace('GFN2-xTB', query)} | {_pct(r['top1'])} | "
                     f"{_pct(r['top3'])} | {r['median']['mean']:.1f} | {_pct(r['isomer'])} | "
                     f"{narrowed} | {chosen} |")
    return lines + [""]


def _report_a(a: dict) -> List[str]:
    lines = ["## Part (a): which part of the structure moves which part of the spectrum", "",
             "A kernel canonical correlation analysis of a structural kernel and the kernel "
             f"of the ORCA spectra (width {a['spectrum']['width']:g} cm^-1, power "
             f"{a['spectrum']['power']:g}).  It finds pairs of directions, one in each "
             "description, whose values over the acids (the variates) correlate most.  The "
             "correlation on the acids that the directions were fitted on is close to 1 "
             "whatever the two descriptions hold, so a pair is judged by its correlation on "
             f"held-out acids: {a['folds']} folds that hold out whole substituents.  It is "
             f"compared with {a['permutations']} analyses in which the pairing of the two "
             "descriptions is shuffled.  The structural kernel is the invariant local kernel "
             "around the carbonyl carbon; the width of its envelope and the ridge are the "
             "ones with the highest held-out correlation.", ""]
    for key, v in a["views"].items():
        lines += [f"### {v['label']}", "",
                  "The held-out correlation of the first pair by the width of the envelope "
                  "around the carbonyl carbon: "
                  + ", ".join(f"{w} {c['rho'][0]:+.3f}" for w, c in v["widths"].items())
                  + f".  The table is for {v['width']}.", "",
                  "| pair | correlation of the two variates on the acids of the fit | "
                  "correlation on held-out "
                  "acids (sd over the folds) | shuffled pairing: mean | shuffled pairing: "
                  "95 % | above the shuffled pairing |", "|---:|---:|---:|---:|---:|---|"]
        for i in range(a["components"]):
            lines.append(f"| {i + 1} | {v['trained_rho'][i]:+.3f} | "
                         f"{v['held_out_rho'][i]:+.3f} ({v['held_out_sd'][i]:.3f}) | "
                         f"{v['null_mean'][i]:+.3f} | {v['null_q95'][i]:+.3f} | "
                         f"{'yes' if v['held_out_rho'][i] > v['null_q95'][i] else 'no'} |")
        lines += ["", "What the pairs follow: the correlation of the held-out variates with "
                  "five properties of the acid.  The sign of a pair is free; the two variates "
                  "of one pair have the same sign.", "",
                  "| property | " + " | ".join(
                      f"pair {i + 1}: structure | pair {i + 1}: spectrum"
                      for i in range(a["components"])) + " |",
                  "|---|" + "---:|---:|" * a["components"]]
        for name in a["properties"]:
            r = v["reading"][name]
            lines.append(f"| {name} | " + " | ".join(
                f"{r['structure'][i]:+.2f} | {r['spectrum'][i]:+.2f}"
                for i in range(a["components"])) + " |")
        lines += ["", "Which region of the spectrum carries a pair: the correlation, over "
                  "held-out acids, between what the region adds to the spectral variate and "
                  "the structural variate of the same pair.", "",
                  "| region | " + " | ".join(f"pair {i + 1}" for i in range(a["components"]))
                  + " |", "|---|" + "---:|" * a["components"]]
        for label, row in zip(a["regions"], v["regions"]):
            lines.append(f"| {label} | " + " | ".join(f"{c:+.2f}" for c in row) + " |")
        lines.append("")
    return lines


def _verdict(d: dict) -> List[str]:
    a, b = d["a"], d["b"]
    rows = b["rows"]
    best, plain, band = rows["chosen"], rows["two factors"], rows["band"]
    gate = best["top1"]["mean"] > 0.5
    narrowed = best["top1_formula"]["mean"] > 0.5
    out = ["## Reading", "",
           f"**Part (b).** With centred spectra and the map of the axis chosen inside the "
           f"calibration acids, the spectrum of the second level ranks its own acid first in "
           f"{100 * best['top1']['mean']:.1f} % of the queries and among the first three in "
           f"{100 * best['top3']['mean']:.1f} %.  It puts the right isomer above the other "
           f"isomer of the same substituent in {100 * best['isomer']['mean']:.1f} % of the "
           f"cases.  When only the candidates with the composition of the query are ranked, "
           f"it is first in {100 * best['top1_formula']['mean']:.1f} %; the composition alone "
           f"gives {100 * b['formula']['top1']:.1f} %.  Without the centring the share of "
           f"first places is {100 * rows['none']['top1']['mean']:.1f} % with the axis as it "
           f"is and {100 * plain['top1']['mean']:.1f} % with two factors.  With centred "
           f"spectra the three maps give {100 * rows['none, centred']['top1']['mean']:.1f} % "
           f"(none), {100 * rows['one factor, centred']['top1']['mean']:.1f} % (one factor) "
           f"and {100 * rows['two factors, centred']['top1']['mean']:.1f} % (two factors): "
           f"the map with the most freedom is the worst on new queries.  The position of the "
           f"carbonyl "
           f"band alone gives {100 * band['top1']['mean']:.1f} % and chance "
           f"{100 * b['chance_top1']:.1f} %.  A measured-like curve gives "
           f"{100 * rows['measured-like']['top1']['mean']:.1f} %, and the other direction "
           f"{100 * rows['reverse']['top1']['mean']:.1f} %.", "",
           "**The width that is chosen.** The calibration chooses "
           + " or ".join(shown(k) for k in best.get("chosen", [])) + ".  A width of this size "
           "compares regions of the spectrum and not single lines.  The two levels of theory "
           "do not agree line by line.  A map of the axis can move the carbonyl band and the "
           "O-H stretch onto their places, but the other lines stay tens of cm^-1 apart in a "
           "way that no smooth map repairs, and the calibration then prefers a width at which "
           "the places of single lines do not matter.", "",
           "**The gate of the item.** A spectrum computed at a second level of theory has to "
           "rank its own acid first among the series in most cases, where the composition "
           f"cannot tell the isomers apart.  With GFN2-xTB as the second level the gate is "
           f"{'met' if gate else 'not met'} among the whole series.  Among the candidates "
           f"with the composition of the query, which is the question of the isomers, it is "
           f"{'met' if narrowed else 'not met'}."
           + (f"  With {d['b_dft']['query']} as the second level it is "
              f"{'met' if d['b_dft']['rows']['chosen']['top1']['mean'] > 0.5 else 'not met'} "
              "among the whole series." if "b_dft" in d else ""), ""]
    if "b_dft" in d:
        dft = d["b_dft"]
        r2 = dft["rows"]["chosen"]
        gate2 = r2["top1"]["mean"] > 0.5
        out += [f"**Part (b) with a second density functional.** A spectrum from "
                f"{dft['query']} has its carbonyl band "
                f"{dft['carbonyl_shift']['mean']:+.1f} cm^-1 from that of the library "
                f"(standard deviation {dft['carbonyl_shift']['sd']:.1f}; GFN2-xTB "
                f"{b['carbonyl_shift']['mean']:+.1f}, {b['carbonyl_shift']['sd']:.1f}).  With "
                f"the same protocol it ranks its own acid first in "
                f"{100 * r2['top1']['mean']:.1f} % of the queries (GFN2-xTB "
                f"{100 * best['top1']['mean']:.1f} %) and among the first three in "
                f"{100 * r2['top3']['mean']:.1f} %.  It puts the right isomer above the other "
                f"in {100 * r2['isomer']['mean']:.1f} %, and it is first among the candidates "
                f"with the composition of the query in "
                f"{100 * r2['top1_formula']['mean']:.1f} %.  The calibration chooses "
                + " or ".join(shown(k) for k in r2.get("chosen", [])) + ".  A measured-like "
                f"curve gives {100 * dft['rows']['measured-like']['top1']['mean']:.1f} %.  "
                f"For this level the gate is {'met' if gate2 else 'not met'} among the whole "
                "series.  "
                + ("The share of GFN2-xTB was therefore a limit of that method and not of "
                   "the kernel: with a second level that is near the first, the kernel finds "
                   "the acid." if gate2 and not gate else
                   "The second functional does not change the answer of the gate."), ""]
        plain = dft["rows"]["none"]
        out += [f"**What is needed with a near level.** Without a map of the axis and without "
                f"centring the spectrum of the second functional is already first in "
                f"{100 * plain['top1']['mean']:.1f} % of the queries.  The centring, which "
                f"decided the result for GFN2-xTB, adds little here "
                f"({100 * dft['rows']['none, centred']['top1']['mean']:.1f} %).  The position "
                f"of the carbonyl band alone gives "
                f"{100 * dft['rows']['band']['top1']['mean']:.1f} %, so it is the whole "
                "spectrum that finds the acid and not one band.", ""]
    if "b_control" in d:
        c = d["b_control"]["rows"]["chosen"]
        out += [f"**Control: the start geometry.** The spectra of the second functional start "
                f"from the ORCA geometries of the library, and the GFN2-xTB spectra start "
                f"from the force-field geometries.  With GFN2-xTB started from the ORCA "
                f"geometries as well, the same protocol gives "
                f"{100 * c['top1']['mean']:.1f} % of first places (from the force-field "
                f"geometries {100 * best['top1']['mean']:.1f} %) and "
                f"{100 * c['top1_formula']['mean']:.1f} % among the candidates with the "
                f"composition of the query "
                f"({100 * best['top1_formula']['mean']:.1f} %).  The start geometry is not "
                "what separates the two levels; the method is.", ""]
    for key, v in a["views"].items():
        widths = v["widths"]
        narrow, wide = min(widths, key=lambda w: float(w.split()[0])), max(
            widths, key=lambda w: float(w.split()[0]))
        text = (f"**Part (a), {v['label']}.** The pairs above the shuffled pairing on held-out "
                f"substituents are: " + (", ".join(str(i) for i in v["above"]) or "none")
                + f" of the first {a['components']}.")
        for i in v["above"][:2]:
            follows = max(a["properties"], key=lambda p: abs(v["reading"][p]["structure"][i - 1]))
            order = np.argsort([-abs(row[i - 1]) for row in v["regions"]])[:2]
            r_s = v["reading"][follows]["structure"][i - 1]
            r_p = v["reading"][follows]["spectrum"][i - 1]
            text += (f"  Pair {i} has a held-out correlation of {v['held_out_rho'][i - 1]:+.3f}.  "
                     + (f"Of the five properties, its two variates follow {follows} most "
                        if abs(r_s) >= 0.5 else
                        f"Its two variates follow none of the five properties closely; the "
                        f"largest correlation is with {follows} ")
                     + f"(r {r_s:+.2f} for the structure and {r_p:+.2f} for the spectrum), "
                     "and the regions of the spectrum that carry it most are: "
                     + " and ".join(f"{a['regions'][k]} ({v['regions'][k][i - 1]:+.2f})"
                                    for k in order) + ".")
        text += (f"  The held-out correlation of the first pair is "
                 f"{widths[narrow]['rho'][0]:+.3f} with an envelope of {narrow} around the "
                 f"carbonyl carbon and {widths[wide]['rho'][0]:+.3f} with {wide}.")
        out += [text, ""]
    q, g = a["views"]["quantum"]["widths"], a["views"]["five"]["widths"]
    narrow = min(q, key=lambda w: float(w.split()[0]))
    out += ["**Where the first pair lives.** The plan expected the first structural direction "
            "on the neighbourhood of the carbonyl group and the first spectral direction on "
            "the carbonyl band.  On the spectral side the first pair is carried by the "
            "carbonyl stretch and by the O-H stretch about equally.  On the structural side "
            "the answer depends on the charges.  With quantum charges the pair is strongest "
            f"with the narrowest envelope ({narrow}: {q[narrow]['rho'][0]:+.3f}), which holds "
            "little beyond the carboxyl group: the quantum calculation has carried the "
            "effect of the substituent to that group.  With Gasteiger charges, which are the "
            "same on the carboxyl group of every acid, the pair vanishes there "
            f"({g[narrow]['rho'][0]:+.3f}) and needs an envelope that reaches the substituent.  "
            "This is the finding of plan item SP2, reached here without a target.", ""]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", choices=["all", "a", "b", "dft", "control"],
                    help="b: the GFN2-xTB queries; dft: the queries of a second density "
                         "functional; control: GFN2-xTB queries that start from the ORCA "
                         "geometries; a: the canonical correlation analysis")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--report", action="store_true", help="only write the table")
    a = ap.parse_args(argv)
    if not a.report:
        run(a.part, a.repeats)
    text = report()
    if not text:
        print("no results yet: run the script without --report first")
        return 1
    print(f"Report written to {os.path.join(RESULTS, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
