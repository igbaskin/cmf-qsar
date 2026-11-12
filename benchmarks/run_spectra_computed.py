#!/usr/bin/env python
"""
A measured infrared spectrum against a library of computed spectra (plan item SP3).

`run_spectra_measured.py` compared measured spectra with each other and gave the ceiling.
This benchmark asks the question of plan item SP3 with a measured query: which structure fits
a spectrum?  The queries are the measured spectra of the Chemotion IR set
(`prepare_chemotion_ir.py`).  The library holds one computed spectrum for every candidate
compound: a GFN2-xTB spectrum of one conformer of one molecule in the gas phase
(`prepare_chemotion_xtb.py`).  The candidates are ranked by the kernel of
`cmfqsar/spectrum.py` between the measured curve and their computed stick spectra, and the
question is where the compound of the query stands.

What is chosen, and how
-----------------------
A computed and a measured spectrum differ in more than two measured spectra do, so more is
chosen than in `run_spectra_measured.py`:

- the form of the kernel: plain, the mean of every spectrum taken off, or the slopes;
- the width of the broadening, the same for the curve and for the sticks;
- one factor on the computed wavenumbers (a harmonic calculation puts the lines too high);
- the region: the whole range, or 700 to 1800 cm^-1 alone, which leaves out the O-H and N-H
  stretch bands that a hydrogen bond in a solid moves and broadens;
- the power of the intensities (1, or 0.5 for the square root);
- whether the computed intensities are divided by the wavenumber.  Nearly all spectra of the
  set were measured by attenuated total reflectance (ATR).  The depth that the beam reaches
  in the sample grows with the wavelength, so a band at a low wavenumber is stronger in such
  a spectrum than the same band in a transmission spectrum;
- whether the spectra are centred: the mean spectrum of the library is taken from every
  library spectrum, and the mean of the queries from every query.  The two means are
  computed without the answers.

All of this is chosen on half of the compounds, for which the right answer is used, and scored
on the queries of the other half.  The split is repeated.  The library always holds every
candidate.

Two levels of theory
--------------------
`prepare_chemotion_dft.py` computes the compounds with at most 25 atoms again with a density
functional (B97-3c), from the geometry of xtb.  The option ``--subset`` makes the test above
twice on the compounds that have both spectra: the same candidates, the same queries and the
same splits, once with each library.  The difference between the two results is then the
effect of the level of theory alone.  The result is ``subset.json``, and the table gains a
section.

Usage
-----
    python benchmarks/prepare_chemotion_ir.py       # once: downloads 47.5 MB
    python benchmarks/prepare_chemotion_xtb.py --jobs 14   # once: the library, 81 minutes
    python benchmarks/run_spectra_computed.py       # the benchmark (17 minutes), then the table
    python benchmarks/prepare_chemotion_dft.py --jobs 14   # once: the small compounds at
                                                           # B97-3c, 13.5 hours
    python benchmarks/run_spectra_computed.py --subset     # the two levels on those compounds
    python benchmarks/run_spectra_computed.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import prepare_chemotion_ir as PREP  # noqa: E402
import prepare_chemotion_dft as DFT  # noqa: E402
import prepare_chemotion_xtb as XTB  # noqa: E402
import run_spectra_measured as M  # noqa: E402

RESULTS = os.path.join(HERE, "results", "spectra_computed")
CEILING = os.path.join(HERE, "results", "spectra_measured", "measured.json")
STEP = 2.0
#: the range of the kernel; the measured curves reach 50 cm^-1 further on both sides
RANGE = (700.0, 3850.0)
#: the sticks are put on a wider grid, so that a line just outside the range still adds the
#: part of its broadened shape that reaches into it
WIDE = (450.0, 4100.0)
FORMS = ("plain", "own mean removed", "slope")
WIDTHS = (4.0, 8.0, 16.0, 32.0, 64.0)
FACTORS = (0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01)
REGIONS = (("whole", 700.0, 3850.0), ("700-1800", 700.0, 1800.0))
POWERS = (1.0, 0.5)
#: computed intensities as they are, or divided by the wavenumber (see the module text)
DEPTHS = (False, True)
REPEATS = 20
SEED = 20261112


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Spectra on the grid
# --------------------------------------------------------------------------- #
def deposit(sticks: Sequence[Tuple[np.ndarray, np.ndarray]], factor: float = 1.0,
            power: float = 1.0, depth: bool = False, wide: Tuple[float, float] = WIDE,
            step: float = STEP) -> np.ndarray:
    """Stick spectra on an even grid, one row each, before any broadening.

    A stick at the position ``factor * x`` is shared between its two neighbouring grid points
    in proportion to its distance from them, which keeps its intensity and its mean position.
    The sharing makes a stick a little wider.  After a broadening of two grid steps the row
    differs from the exact broadened stick spectrum (`spectrum.broaden`) by up to three per
    cent of its largest value, and from four grid steps on by less than one per cent.
    With `depth` every intensity is first multiplied by 1000 over its wavenumber, which is
    how the intensity of an ATR measurement depends on the wavelength.  `power` is applied
    after that.
    """
    n = int(round((wide[1] - wide[0]) / step)) + 1
    out = np.zeros((len(sticks), n))
    for k, (x, y) in enumerate(sticks):
        place = float(factor) * np.asarray(x, float)
        pos = (place - wide[0]) / step
        w = np.asarray(y, float)
        if depth:
            w = w * 1000.0 / np.maximum(place, 1.0)
        w = np.sign(w) * np.abs(w) ** float(power) if power != 1.0 else w
        low = np.floor(pos).astype(int)
        share = pos - low
        for idx, part in ((low, 1.0 - share), (low + 1, share)):
            ok = (idx >= 0) & (idx < n)
            np.add.at(out[k], idx[ok], (w * part)[ok])
    return out


def powered_curves(A: np.ndarray, power: float) -> np.ndarray:
    """Measured absorbance curves with the power of the intensity applied.  For a power other
    than 1 the lowest value of every curve is taken off first, because a root of a negative
    absorbance has no meaning."""
    if power == 1.0:
        return np.asarray(A, float)
    low = A - A.min(axis=1, keepdims=True)
    return np.clip(low, 0.0, None) ** float(power)


def smooth(rows: np.ndarray, width: float, kind: str, step: float = STEP) -> np.ndarray:
    """Every row broadened with a Gaussian of standard deviation `width`.  Beyond the two ends
    a measured curve (`kind` "curve") is continued with its last value, so that a baseline
    does not fall to zero there; sticks (`kind` "sticks") have nothing beyond their ends."""
    mode = "nearest" if kind == "curve" else "constant"
    return gaussian_filter1d(np.asarray(rows, float), width / step, axis=1, mode=mode,
                             truncate=6.0)


def cut(rows: np.ndarray, first: float, lo: float, hi: float, step: float = STEP) -> np.ndarray:
    """The columns of rows on a grid that starts at `first`, from `lo` to `hi`."""
    a = int(round((lo - first) / step))
    b = int(round((hi - first) / step)) + 1
    return rows[:, a:b]


def formed(rows: np.ndarray, form: str, first: float, lo: float, hi: float,
           step: float = STEP) -> np.ndarray:
    """A form of the kernel on a region, as rows of unit length.  The slope is taken on the
    whole row before the region is cut out, so the cut makes no slope of its own."""
    if form == "slope":
        G = cut(np.gradient(rows, step, axis=1), first, lo, hi, step)
    elif form == "own mean removed":
        G = cut(rows, first, lo, hi, step)
        G = G - G.mean(axis=1, keepdims=True)
    elif form == "plain":
        G = cut(rows, first, lo, hi, step)
    else:
        raise ValueError(f"form must be one of {FORMS}; got '{form}'")
    return unit(G)


def unit(G: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.einsum("ij,ij->i", G, G))
    return G / np.where(norm > 1e-300, norm, 1.0)[:, None]


def centred(U: np.ndarray) -> np.ndarray:
    """Rows of unit length with their common mean taken off, and of unit length again."""
    return unit(U - U.mean(axis=0, keepdims=True))


# --------------------------------------------------------------------------- #
# The ranks
# --------------------------------------------------------------------------- #
def ranks(Q: np.ndarray, L: np.ndarray, owner: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For every query (rows of `Q`) the rank of its own compound among the library rows `L`
    (`owner` gives the library row of every query; 1 is the best, and a library spectrum as
    similar as the right one counts as better), and the library row that came first."""
    S = Q @ L.T
    own = S[np.arange(len(Q)), owner]
    rank = np.sum(S >= own[:, None], axis=1)        # the right one counts itself once
    return rank.astype(int), np.argmax(S, axis=1)


def setting_name(form: str, width: float, factor: float, region: str, power: float,
                 centre: bool, depth: bool = False) -> str:
    return f"{form}|{width:g}|{factor:g}|{region}|{power:g}|{int(centre)}|{int(depth)}"


def setting_values(key: str) -> dict:
    form, width, factor, region, power, centre, depth = key.split("|")
    return {"form": form, "width": float(width), "factor": float(factor), "region": region,
            "power": float(power), "centred": bool(int(centre)), "depth": bool(int(depth))}


def shown(key: str) -> str:
    """A setting as words for a table: a vertical bar would end the cell."""
    v = setting_values(key)
    region = ("the whole range" if v["region"] == "whole"
              else v["region"].replace("-", " to ") + " cm^-1")
    return (f"{v['form']}, width {v['width']:g} cm^-1, wavenumbers times {v['factor']:g}, "
            f"{region}, " + ("intensities" if v["power"] == 1 else "square root of the "
                             "intensities")
            + (", computed intensities over the wavenumber" if v["depth"] else "")
            + (", centred" if v["centred"] else ""))


def all_ranks(A: np.ndarray, x: np.ndarray, sticks: Sequence[Tuple[np.ndarray, np.ndarray]],
              owner: np.ndarray, forms=FORMS, widths=WIDTHS, factors=FACTORS,
              regions=REGIONS, powers=POWERS, centring=(False, True), depths=DEPTHS,
              keep_named: Optional[str] = None, say=None) -> Tuple[dict, Optional[np.ndarray]]:
    """The rank of the right compound for every query under every setting: a dictionary from
    the name of a setting to the ranks.  With `keep_named` (the name of one setting) the
    library rows that came first under that setting are returned too."""
    out: Dict[str, np.ndarray] = {}
    named = None
    for power in powers:
        curves = powered_curves(A, power)
        smooth_q = {w: smooth(curves, w, "curve") for w in widths}
        made_q: Dict[tuple, np.ndarray] = {}
        for factor in factors:
            for depth in depths:
                D = deposit(sticks, factor, power, depth)
                for width in widths:
                    Ls = smooth(D, width, "sticks")
                    for form in forms:
                        for region, lo, hi in regions:
                            at = (width, form, region)
                            if at not in made_q:
                                made_q[at] = formed(smooth_q[width], form, float(x[0]), lo, hi)
                            Qu = made_q[at]
                            Lu = formed(Ls, form, WIDE[0], lo, hi)
                            for centre in centring:
                                key = setting_name(form, width, factor, region, power,
                                                   centre, depth)
                                r, top = ranks(centred(Qu) if centre else Qu,
                                               centred(Lu) if centre else Lu, owner)
                                out[key] = r
                                if key == keep_named:
                                    named = top
            if say:
                say(f"  power {power:g}, factor {factor:g}: {len(out)} settings done")
    return out, named


def one_setting(A: np.ndarray, x: np.ndarray, sticks, owner: np.ndarray,
                key: str) -> Tuple[np.ndarray, np.ndarray]:
    """The ranks under one setting, and the library rows that came first."""
    v = setting_values(key)
    out, named = all_ranks(A, x, sticks, owner, forms=(v["form"],), widths=(v["width"],),
                           factors=(v["factor"],),
                           regions=tuple(r for r in REGIONS if r[0] == v["region"]),
                           powers=(v["power"],), centring=(v["centred"],),
                           depths=(v["depth"],), keep_named=key)
    return out[key], named


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def per_compound_scores(rank: np.ndarray, compound: np.ndarray, n_library: int) -> dict:
    """Scores per compound (a compound with several measured spectra counts once): the
    scores of `run_spectra_measured.scores` and the share within the best one per cent of
    the library."""
    s = M.scores(rank, compound)
    one = max(1, int(round(0.01 * n_library)))
    s["top1pct"] = M.per_compound((rank <= one).astype(float), compound)[1]
    return s


def descriptors(smiles: Sequence[str]) -> Dict[str, np.ndarray]:
    """What may limit a computed spectrum: the number of rotatable bonds (one conformer was
    computed) and the number of hydrogen-bond donors (one molecule in the gas phase was
    computed)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    rot, don, atoms = [], [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rot.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        don.append(rdMolDescriptors.CalcNumHBD(mol))
        atoms.append(Chem.AddHs(mol).GetNumAtoms())
    return {"rotatable_bonds": np.array(rot), "donors": np.array(don),
            "atoms": np.array(atoms)}


GROUPS = (("no hydrogen-bond donor", "donors", 0, 0),
          ("one or more hydrogen-bond donors", "donors", 1, 10 ** 6),
          ("at most 2 rotatable bonds", "rotatable_bonds", 0, 2),
          ("3 to 5 rotatable bonds", "rotatable_bonds", 3, 5),
          ("6 or more rotatable bonds", "rotatable_bonds", 6, 10 ** 6),
          ("at most 40 atoms", "atoms", 0, 40),
          ("41 to 70 atoms", "atoms", 41, 70),
          ("71 or more atoms", "atoms", 71, 10 ** 6))


def by_group(first: np.ndarray, top10: np.ndarray, desc: Dict[str, np.ndarray],
             keys: Sequence[str] = ()) -> dict:
    """The share of first places and of places among the first ten in groups of compounds;
    `keys` keeps the groups of these descriptors only."""
    out = {}
    for name, key, lo, hi in GROUPS:
        if keys and key not in keys:
            continue
        m = (desc[key] >= lo) & (desc[key] <= hi)
        if m.any():
            out[name] = {"compounds": int(m.sum()), "first": float(first[m].mean()),
                         "top10": float(top10[m].mean())}
    return out


def best_of_option(on_all: Dict[str, dict]) -> dict:
    """For every value of every option, the best share of first places over the settings
    that have this value (all compounds, no choice)."""
    out: Dict[str, Dict[str, float]] = {}
    for key, s in on_all.items():
        for option, value in setting_values(key).items():
            slot = out.setdefault(option, {})
            name = str(value)
            slot[name] = max(slot.get(name, 0.0), s["first"])
    return out


def run(results: str = RESULTS, repeats: int = REPEATS, **grid) -> dict:
    t0 = time.time()
    x, A, table = PREP.load()
    smiles = [r["smiles"] for r in table]
    keep, _ = M.without_copies(A, smiles)
    A = A[keep]
    smiles = [s for s, k in zip(smiles, keep) if k]
    head, computed = XTB.load()
    library = sorted(computed)
    row_of = {s: j for j, s in enumerate(library)}
    queries = np.array([i for i, s in enumerate(smiles) if s in row_of])
    owner = np.array([row_of[smiles[i]] for i in queries])
    sticks = [(np.array(computed[s]["wavenumbers"]), np.array(computed[s]["intensities"]))
              for s in library]
    log(f"{len(queries)} measured spectra as queries, {len(library)} computed spectra in the "
        f"library ({head['qm_level']})")

    rank_of, _ = all_ranks(A[queries], x, sticks, owner, say=log, **grid)
    ids = np.unique(owner)
    per = {k: per_compound_scores(r, owner, len(library)) for k, r in rank_of.items()}
    on_all = {k: {name: float(np.mean(v)) for name, v in s.items()} for k, s in per.items()}

    fits = {"all": M.nested(per, list(per), repeats, SEED)}
    parts = [(form, [k for k in per if setting_values(k)["form"] == form]) for form in FORMS]
    parts += [("region " + region, [k for k in per if setting_values(k)["region"] == region])
              for region, _, _ in REGIONS]
    for name, allowed in parts:
        if allowed:                         # a smaller grid may leave a form or a region out
            fits[name] = M.nested(per, allowed, repeats, SEED)
    differences = {name: M.mean_difference(fits["all"]["per_compound_first"],
                                           fits[name]["per_compound_first"])
                   for name in fits if name != "all"}
    for fit in fits.values():               # one number per compound and split: not kept
        fit.pop("per_compound_first")
        fit.pop("scored")

    best = next(iter(fits["all"]["chosen"]))
    rank = rank_of[best]
    _, named = one_setting(A[queries], x, sticks, owner, best)
    formulas, prints = M.structures(library)
    rng = np.random.default_rng(SEED)
    wrong = np.where(rank > 1)[0]
    t_named = [M.tanimoto(prints, int(owner[k]), int(named[k])) for k in wrong]
    t_random = [M.tanimoto(prints, int(owner[k]), int(rng.integers(len(library))))
                for k in wrong]
    desc = descriptors([library[j] for j in ids])
    imaginary = np.array([computed[library[j]]["n_imaginary"] for j in ids])

    out = {
        "source": PREP.DOI, "licence": PREP.LICENCE, "library_level": head["qm_level"],
        "xtb": head.get("xtb", ""), "library": len(library), "queries": int(len(queries)),
        "compounds_with_a_query": int(len(ids)),
        "library_left_out": head["left_out"], "library_failed": len(head["failed"]),
        "library_with_an_imaginary_mode": head["with_an_imaginary_mode"],
        "library_seconds": head["seconds"], "max_atoms": head["max_atoms"],
        "chance": 1.0 / len(library), "repeats": repeats, "settings": len(per),
        "nested": fits, "all_minus": differences, "setting_of_the_readings": best,
        "on_all_compounds_at_that_setting": on_all[best],
        "median_rank": float(np.median(M.per_compound(rank.astype(float), owner)[1])),
        "best_of_option": best_of_option(on_all),
        "groups": by_group(per[best]["first"], per[best]["top10"], desc),
        "with_an_imaginary_mode": {
            "compounds": int((imaginary > 0).sum()),
            "first": float(per[best]["first"][imaginary > 0].mean())
            if (imaginary > 0).any() else None,
            "top10": float(per[best]["top10"][imaginary > 0].mean())
            if (imaginary > 0).any() else None},
        "wrong_answers": {"n": 0} if not len(wrong) else {
            "n": int(len(wrong)), "tanimoto_named": float(np.mean(t_named)),
            "tanimoto_named_median": float(np.median(t_named)),
            "tanimoto_random": float(np.mean(t_random)),
            "share_similar": float(np.mean(np.array(t_named) >= 0.5)),
            "share_same_formula": float(np.mean([formulas[int(owner[k])]
                                                 == formulas[int(named[k])] for k in wrong]))},
        "seconds": round(time.time() - t0, 1),
    }
    if os.path.isfile(CEILING):
        with open(CEILING, encoding="utf-8") as fh:
            c = json.load(fh)
        out["ceiling"] = {"first": c["nested"]["all forms"]["first"],
                          "top10": c["nested"]["all forms"]["top10"],
                          "compounds": c["compounds"]}
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "computed.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# Two levels of theory on the same compounds
# --------------------------------------------------------------------------- #
def measured() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """The measured spectra without the copies: the axis, the curves and their compounds."""
    x, A, table = PREP.load()
    smiles = [r["smiles"] for r in table]
    keep, _ = M.without_copies(A, smiles)
    return x, A[keep], [s for s, k in zip(smiles, keep) if k]


def level_on(x: np.ndarray, A: np.ndarray, smiles: Sequence[str], computed: Dict[str, dict],
             candidates: Sequence[str], repeats: int = REPEATS, **grid) -> dict:
    """The test of :func:`run` for one library on a given list of candidates: the settings
    are chosen on half of the compounds and scored on the other half.  The splits depend only
    on the number of compounds and on the seed, so two libraries with the same candidates are
    scored on the same halves."""
    library = sorted(candidates)
    row_of = {s: j for j, s in enumerate(library)}
    queries = np.array([i for i, s in enumerate(smiles) if s in row_of])
    owner = np.array([row_of[smiles[i]] for i in queries])
    sticks = [(np.array(computed[s]["wavenumbers"]), np.array(computed[s]["intensities"]))
              for s in library]
    rank_of, _ = all_ranks(A[queries], x, sticks, owner, **grid)
    per = {k: per_compound_scores(r, owner, len(library)) for k, r in rank_of.items()}
    on_all = {k: {name: float(np.mean(v)) for name, v in s.items()} for k, s in per.items()}
    fit = M.nested(per, list(per), repeats, SEED)
    best = next(iter(fit["chosen"]))
    desc = descriptors([library[j] for j in np.unique(owner)])
    return {"per_compound": {"first": per[best]["first"], "top10": per[best]["top10"]},
            "groups": by_group(per[best]["first"], per[best]["top10"], desc,
                               keys=("donors", "rotatable_bonds")),
            "library": len(library), "queries": int(len(queries)),
            "compounds_with_a_query": int(len(np.unique(owner))),
            "nested": fit, "setting": best, "on_all_compounds_at_that_setting": on_all[best],
            "median_rank": float(np.median(
                M.per_compound(rank_of[best].astype(float), owner)[1])),
            "best_of_option": best_of_option(on_all),
            "with_an_imaginary_mode": int(sum(1 for s in library
                                              if computed[s]["n_imaginary"]))}


def run_subset(results: str = RESULTS, repeats: int = REPEATS, **grid) -> dict:
    """The GFN2-xTB library and the density-functional library on the compounds that have
    both: the same candidates, the same queries, the same halves."""
    t0 = time.time()
    x, A, smiles = measured()
    xhead, xtb = XTB.load()
    dhead, dft = DFT.load()
    candidates = sorted(set(xtb) & set(dft))
    levels = {}
    for name, computed in ((xhead["qm_level"], xtb), (dhead["qm_level"], dft)):
        levels[name] = level_on(x, A, smiles, computed, candidates, repeats, **grid)
        log(f"  {name}: first {levels[name]['nested']['first']:.3f}, among the first ten "
            f"{levels[name]['nested']['top10']:.3f}")
    low, high = (levels[xhead["qm_level"]], levels[dhead["qm_level"]])
    out = {"source": PREP.DOI, "licence": PREP.LICENCE,
           "levels": [xhead["qm_level"], dhead["qm_level"]],
           "max_atoms": dhead["max_atoms"], "compounds_asked": dhead["compounds"],
           "failed_at_the_higher_level": len(dhead["failed"]),
           "seconds_of_the_higher_level": dhead["seconds"],
           "job_seconds_of_the_higher_level": dhead.get("job_seconds"),
           "candidates": len(candidates), "chance": 1.0 / len(candidates),
           "repeats": repeats,
           "higher_minus_lower_first": M.mean_difference(
               high["nested"]["per_compound_first"], low["nested"]["per_compound_first"]),
           "seconds": round(time.time() - t0, 1)}
    out["found"] = {}                       # each level at the setting chosen most often
    for score in ("first", "top10"):
        a = high["per_compound"][score] > 0
        b = low["per_compound"][score] > 0
        out["found"][score] = {"both": int((a & b).sum()), "higher_only": int((a & ~b).sum()),
                               "lower_only": int((~a & b).sum()),
                               "neither": int((~a & ~b).sum())}
    for level in levels.values():
        level["nested"].pop("per_compound_first")
        level["nested"].pop("scored")
        level.pop("per_compound")
    out["results"] = levels
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "subset.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def _report_subset(results: str) -> List[str]:
    """The lines of the comparison of two levels, or nothing when it was not run."""
    path = os.path.join(results, "subset.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    first = d["results"][d["levels"][0]]
    lines = ["## Two levels of theory on the small compounds",
             "",
             f"The {d['candidates']} compounds with at most {d['max_atoms']} atoms that have "
             f"a spectrum at both levels are the candidates; {first['queries']} measured "
             f"spectra of {first['compounds_with_a_query']} of them are the queries. Both "
             "levels start from the same conformer. The settings are chosen for each level "
             f"on half of the compounds and scored on the other half, the same "
             f"{d['repeats']} splits for both.",
             "",
             "| Library | Compound is first | In the first three | In the first ten | "
             "Median rank of the right compound | Width chosen most often |",
             "|---|---|---|---|---|---|",
             f"| chance (one of {d['candidates']} compounds) | {_pct(d['chance'])} | "
             f"{_pct(3 * d['chance'])} | {_pct(10 * d['chance'])} | "
             f"{(d['candidates'] + 1) / 2:g} | |"]
    for name in d["levels"]:
        r = d["results"][name]
        f = r["nested"]
        width = setting_values(r["setting"])["width"]
        lines.append(f"| {name} | {_pct(f['first'])} +/- {100 * f['first_sd']:.1f} | "
                     f"{_pct(f['top3'])} +/- {100 * f['top3_sd']:.1f} | "
                     f"{_pct(f['top10'])} +/- {100 * f['top10_sd']:.1f} | "
                     f"{r['median_rank']:g} | {width:g} cm^-1 |")
    df = d["higher_minus_lower_first"]
    lines += ["",
              f"Difference in the share of first places, {d['levels'][1]} minus "
              f"{d['levels'][0]}, per compound: {100 * df['mean']:+.1f} "
              f"[{100 * df['low']:+.1f}, {100 * df['high']:+.1f}] points (95 % bootstrap "
              "interval over the compounds).",
              ""]
    for name in d["levels"]:
        r = d["results"][name]
        lines.append(f"- {name}: chosen most often: {shown(r['setting'])}. Best share of "
                     "first places by width, all compounds: "
                     + ", ".join(f"{float(w):g} cm^-1 {_pct(v)}"
                                 for w, v in r["best_of_option"]["width"].items()) + ".")
    low, high = (d["results"][name] for name in d["levels"])
    lines += ["",
              "Groups of compounds, each level at the setting chosen most often (all "
              "compounds):",
              "",
              f"| Compounds | Number | First, {d['levels'][0]} | First, {d['levels'][1]} | "
              f"In the first ten, {d['levels'][0]} | In the first ten, {d['levels'][1]} |",
              "|---|---|---|---|---|---|"]
    for name, g in high["groups"].items():
        g0 = low["groups"][name]
        lines.append(f"| {name} | {g['compounds']} | {_pct(g0['first'])} | {_pct(g['first'])} "
                     f"| {_pct(g0['top10'])} | {_pct(g['top10'])} |")
    f, t = d["found"]["first"], d["found"]["top10"]
    lines += ["",
              f"Of the {sum(f.values())} compounds, {f['both']} have a first place at both "
              f"levels, {f['higher_only']} only at {d['levels'][1]}, {f['lower_only']} only at "
              f"{d['levels'][0]} and {f['neither']} at neither. For a place among the first "
              f"ten the four numbers are {t['both']}, {t['higher_only']}, {t['lower_only']} "
              f"and {t['neither']}. A compound with several measured spectra is counted when "
              "one of them is found.",
              ""]
    return lines


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def _pct(v: float) -> str:
    return f"{100.0 * v:.1f} %"


def report(results: str = RESULTS) -> str:
    """The table of the benchmark as text; it is also written to ``table.md``."""
    path = os.path.join(results, "computed.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    n = d["nested"]
    left = "; ".join(f"{v}: {k}" for k, v in d["library_left_out"].items())
    lines = [
        "# A measured infrared spectrum against a library of computed spectra",
        "",
        f"Queries: {d['queries']} measured spectra of {d['compounds_with_a_query']} compounds "
        f"of the Chemotion IR set ({d['source']}, {d['licence']}; no file of the set is in "
        "this repository).",
        f"Library: {d['library']} computed spectra, one per candidate compound "
        f"({d['library_level']}, one conformer, one molecule in the gas phase; "
        f"`benchmarks/prepare_chemotion_xtb.py`). "
        f"{d['library_with_an_imaginary_mode']} of them keep an imaginary mode.",
        f"Compounds without a computed spectrum: {left}; {d['library_failed']} failed in the "
        "calculation.",
        f"The form of the kernel, the width, a factor on the computed wavenumbers, the "
        f"region, the power of the intensities, their division by the wavenumber and the "
        f"centring ({d['settings']} settings) "
        f"are chosen on half of the compounds and scored on the other half, {d['repeats']} "
        "random splits. A compound counts once. The number after the sign +/- is the "
        "standard deviation over the splits.",
        "",
        "## How often the compound of the query is found",
        "",
        "| What is compared | Compound is first | In the first three | In the first ten |",
        "|---|---|---|---|",
        f"| chance (one of {d['library']} compounds) | {_pct(d['chance'])} | "
        f"{_pct(3 * d['chance'])} | {_pct(10 * d['chance'])} |",
    ]
    rows = (("all", "all settings open to the choice"),
            ("slope", "only the kernel of the slopes"),
            ("own mean removed", "only the spectra with their mean taken off"),
            ("plain", "only the plain kernel"),
            ("region whole", "only the whole range, 700 to 3850 cm^-1"),
            ("region 700-1800", "only the region 700 to 1800 cm^-1"))
    rows = tuple(r for r in rows if r[0] in n)
    for key, label in rows:
        f = n[key]
        lines.append(f"| {label} | {_pct(f['first'])} +/- {100 * f['first_sd']:.1f} | "
                     f"{_pct(f['top3'])} +/- {100 * f['top3_sd']:.1f} | "
                     f"{_pct(f['top10'])} +/- {100 * f['top10_sd']:.1f} |")
    if "ceiling" in d:
        c = d["ceiling"]
        lines.append(f"| for comparison: a measured spectrum of another sample as the "
                     f"library spectrum, among {c['compounds']} compounds "
                     f"(`spectra_measured`) | {_pct(c['first'])} | | {_pct(c['top10'])} |")
    a = d["on_all_compounds_at_that_setting"]
    lines += ["",
              "Chosen settings over the splits: "
              + "; ".join(f"{shown(k)} in {v}" for k, v in n["all"]["chosen"].items()) + ".",
              "",
              f"At the setting chosen most often, on all compounds: first in "
              f"{_pct(a['first'])}, in the first ten in {_pct(a['top10'])}, in the best one "
              f"per cent of the library in {_pct(a['top1pct'])}; the median rank of the right "
              f"compound is {d['median_rank']:g} of {d['library']}.",
              "",
              "Difference in the share of first places, per compound, with a 95 % bootstrap "
              "interval over the compounds:",
              ""]
    for key, label in rows[1:]:
        df = d["all_minus"][key]
        if df["mean"] == 0 and df["low"] == 0 and df["high"] == 0:
            lines.append(f"- the open choice is the same as '{label}' in every split")
            continue
        lines.append(f"- the open choice minus '{label}': {100 * df['mean']:+.1f} "
                     f"[{100 * df['low']:+.1f}, {100 * df['high']:+.1f}] points")
    lines += ["",
              "## What every option is worth (all compounds, no choice, for the reader)",
              "",
              "The best share of first places over all settings that have the value.",
              "",
              "| Option | Value | Compound is first |",
              "|---|---|---|"]
    names = {"form": "form of the kernel", "width": "width, cm^-1",
             "factor": "factor on the computed wavenumbers", "region": "region",
             "power": "power of the intensities", "centred": "centred",
             "depth": "computed intensities over the wavenumber"}
    for option, values in d["best_of_option"].items():
        for value, share in values.items():
            lines.append(f"| {names[option]} | {value} | {_pct(share)} |")
    lines += ["",
              "## Which compounds are found (the setting chosen most often, all compounds)",
              "",
              "| Compounds | Number | Compound is first | In the first ten |",
              "|---|---|---|---|"]
    for name, g in d["groups"].items():
        lines.append(f"| {name} | {g['compounds']} | {_pct(g['first'])} | "
                     f"{_pct(g['top10'])} |")
    g = d["with_an_imaginary_mode"]
    if g["compounds"]:
        lines.append(f"| the computed geometry keeps an imaginary mode | {g['compounds']} | "
                     f"{_pct(g['first'])} | {_pct(g['top10'])} |")
    w = d["wrong_answers"]
    if w["n"]:                              # a kernel that is always right names no other
        lines += [
            "",
            f"When the kernel names another compound ({w['n']} queries), the Tanimoto "
            f"similarity of that compound to the right one is {w['tanimoto_named']:.2f} on "
            f"average (median {w['tanimoto_named_median']:.2f}), against "
            f"{w['tanimoto_random']:.2f} for a compound drawn at random. In "
            f"{_pct(w['share_similar'])} of these queries the similarity is 0.5 or more, "
            f"and in {_pct(w['share_same_formula'])} the named compound has the same "
            "formula (an isomer)."]
    lines.append("")
    lines += _report_subset(results)
    text = "\n".join(lines)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    ap.add_argument("--subset", action="store_true",
                    help="only the comparison of the two levels on the small compounds "
                         "(needs prepare_chemotion_dft.py)")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    a = ap.parse_args(argv)
    if a.subset:
        run_subset(repeats=a.repeats)
    elif not a.report:
        run(repeats=a.repeats)
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
