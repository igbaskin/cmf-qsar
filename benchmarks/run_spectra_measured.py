#!/usr/bin/env python
"""
A measured infrared spectrum against other measured spectra of the same compounds.

Plan items SP1 and SP3 used computed spectra only.  This benchmark is the first one on
measured spectra.  The set is Chemotion IR (`prepare_chemotion_ir.py` fetches it): infrared
spectra that chemists of research laboratories measured of their own compounds, with the
structure of every compound.  Some compounds were measured more than once, on different
samples.

**The question.**  One measured spectrum of a compound is the query.  The library holds one
spectrum for every compound of the set: for the compound of the query a spectrum of another
sample, and for every other compound its first spectrum.  The compounds are ranked by the
kernel of `cmfqsar/spectrum.py` between the query and their library spectra, and the question
is where the compound of the query stands.  Nothing is computed from a structure, so the
result is a ceiling: it says how well the kernel finds a compound when the only differences
are the sample and the measurement.

**Three forms of the kernel.**  The plain kernel of the absorbance; the same after the mean of
every spectrum is taken off (the cosine of two such spectra is their correlation coefficient);
and the kernel of the slopes of the spectra (`spectrum.slope`), which does not see a constant
baseline.  Each form has a width of the broadening.  The form and the width are chosen on half
of the compounds that have several spectra and are scored on the other half; the split is
repeated.

**Copies.**  Some files of the set are attached to two samples.  A curve that is the same as
another curve of the same compound is left out, because it would be found without any merit.
A curve that is attached to two different compounds is left out with both compounds' copies,
because at least one of the two structures is not the one that was measured.

Usage
-----
    python benchmarks/prepare_chemotion_ir.py          # once: downloads 47.5 MB
    python benchmarks/run_spectra_measured.py          # the benchmark, then the table
    python benchmarks/run_spectra_measured.py --report
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
import prepare_chemotion_ir as PREP  # noqa: E402

RESULTS = os.path.join(HERE, "results", "spectra_measured")
#: the grid of the kernel.  It ends 50 cm^-1 inside the prepared spectra, so the two ends of
#: a broadened curve do not reach into it (see `spectrum.slope`).
GRID = SP.Grid(700.0, 3850.0, 2.0)
FORMS = ("plain", "own mean removed", "slope")
#: standard deviations of the broadening in cm^-1; 0 leaves the measured curve as it is
WIDTHS = (0.0, 4.0, 8.0, 16.0)
REGIONS = (("700 to 1800 cm^-1 alone", 700.0, 1800.0),
           ("1800 to 3850 cm^-1 alone", 1800.0, 3850.0))
REPEATS = 50
SEED = 20261111


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# The set without copies
# --------------------------------------------------------------------------- #
def identical_sets(A: np.ndarray, tol: float = 1e-6) -> List[List[int]]:
    """The sets of rows of `A` that are one and the same curve (largest difference below
    `tol`), each as a sorted list of row numbers; rows that have no copy are not listed."""
    A = np.asarray(A, float)
    sq = np.einsum("ij,ij->i", A, A)
    D = sq[:, None] + sq[None, :] - 2.0 * (A @ A.T)
    label = np.arange(len(A))
    for a, b in np.argwhere(np.triu(D < tol * tol * A.shape[1] + 1e-9, 1)):
        if np.max(np.abs(A[a] - A[b])) < tol and label[a] != label[b]:
            label[label == label[b]] = label[a]
    sets: Dict[int, List[int]] = {}
    for i, lab in enumerate(label):
        sets.setdefault(int(lab), []).append(i)
    return [v for v in sets.values() if len(v) > 1]


def without_copies(A: np.ndarray, compounds: Sequence[str]) -> Tuple[np.ndarray, dict]:
    """Which rows stay, and what was left out.

    Of a set of identical curves of one compound the first stays.  A set of identical curves
    that belongs to more than one compound is left out as a whole.
    """
    keep = np.ones(len(A), bool)
    same, different = 0, []
    for rows in identical_sets(A):
        if len({compounds[i] for i in rows}) == 1:
            keep[rows[1:]] = False
            same += len(rows) - 1
        else:
            keep[rows] = False
            different.append(rows)
    return keep, {"copies_within_a_compound": same,
                  "sets_shared_by_different_compounds": different}


def grouped(compounds: Sequence[str]) -> List[List[int]]:
    """The row numbers of every compound, in the order in which the compounds first appear."""
    groups: Dict[str, List[int]] = {}
    for i, c in enumerate(compounds):
        groups.setdefault(c, []).append(i)
    return list(groups.values())


# --------------------------------------------------------------------------- #
# The three forms of the kernel
# --------------------------------------------------------------------------- #
def broadened(A: np.ndarray, x: np.ndarray, width: float, grid: SP.Grid = GRID) -> np.ndarray:
    """The measured curves (rows of `A` on the axis `x`) on the grid of the kernel, broadened
    to the width `width`; 0 only puts them on the grid."""
    spectra = [SP.Spectrum(x, row, "curve", "wavenumber") for row in A]
    return SP.feature_matrix(spectra, grid, width)


def formed(F: np.ndarray, form: str, grid: SP.Grid = GRID) -> np.ndarray:
    """The rows of a form of the kernel, each divided by its length, so that a dot product of
    two rows is the cosine kernel of that form."""
    if form == "plain":
        G = np.asarray(F, float)
    elif form == "own mean removed":
        G = F - F.mean(axis=1, keepdims=True)
    elif form == "slope":
        G = SP.slope(F, grid)
    else:
        raise ValueError(f"form must be one of {FORMS}; got '{form}'")
    norm = np.sqrt(np.einsum("ij,ij->i", G, G))
    return G / np.where(norm > 1e-300, norm, 1.0)[:, None]


def setting_name(form: str, width: float) -> str:
    return f"{form}|{width:g}"


def shown(key: str) -> str:
    """A setting as words for a table: a vertical bar would end the cell."""
    form, width = key.split("|")
    return f"{form}, not broadened" if float(width) == 0 else f"{form}, width {width} cm^-1"


# --------------------------------------------------------------------------- #
# The ranks
# --------------------------------------------------------------------------- #
def pair_ranks(U: np.ndarray, groups: Sequence[Sequence[int]]) -> Dict[str, np.ndarray]:
    """Where the compound of a query stands in the library.

    `U` holds one row per spectrum, of unit length.  For every compound with two or more
    spectra and every ordered pair of them, the first is the query and the second is that
    compound's spectrum in the library; every other compound is in the library with its first
    spectrum.  Returns, per pair: the number of the compound, the rank of the compound (1 is
    the best; a library spectrum as similar as the right one counts as better), the row of
    the library spectrum that came first, and the rows of the query and of the compound's
    own library spectrum (the target).
    """
    first = np.array([g[0] for g in groups])
    L = U[first]
    comp, rank, named, query, target = [], [], [], [], []
    for c, g in enumerate(groups):
        if len(g) < 2:
            continue
        for q in g:
            sims = L @ U[q]
            sims[c] = -np.inf
            top = int(np.argmax(sims))
            for t in g:
                if t == q:
                    continue
                own = float(U[t] @ U[q])
                r = 1 + int(np.sum(sims >= own))
                comp.append(c)
                rank.append(r)
                named.append(t if r == 1 else int(first[top]))
                query.append(q)
                target.append(t)
    return {"compound": np.array(comp, int), "rank": np.array(rank, int),
            "named": np.array(named, int), "query": np.array(query, int),
            "target": np.array(target, int)}


def misses(ranks: Dict[str, np.ndarray], unit: np.ndarray, beyond: int = 10) -> dict:
    """The pairs in which the compound is not among the first `beyond`: how many, of how many
    compounds, and how alike the query and its target are.  `unit` holds the spectra with
    their own mean taken off and of unit length, so a dot product of two rows is the
    correlation coefficient of two spectra."""
    far = np.where(ranks["rank"] > beyond)[0]
    r = [float(unit[ranks["query"][k]] @ unit[ranks["target"][k]]) for k in far]
    return {"beyond": beyond, "pairs": int(len(far)),
            "compounds": int(len(set(ranks["compound"][far].tolist()))),
            "correlation_median": float(np.median(r)) if r else None,
            "correlation_highest": float(np.max(r)) if r else None}


def per_compound(values: np.ndarray, compound: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The mean of `values` over the pairs of every compound: the compounds and their means.
    A compound with seven spectra has 42 pairs, and would otherwise count 21 times as much as
    a compound with two."""
    ids = np.unique(compound)
    return ids, np.array([np.mean(values[compound == c]) for c in ids])


def scores(rank: np.ndarray, compound: np.ndarray) -> Dict[str, np.ndarray]:
    """Per compound: how often it is first, in the top three, in the top ten, and its mean
    reciprocal rank."""
    out = {}
    for name, v in (("first", rank == 1), ("top3", rank <= 3), ("top10", rank <= 10),
                    ("rr", 1.0 / rank)):
        out[name] = per_compound(v.astype(float), compound)[1]
    return out


def nested(per: Dict[str, Dict[str, np.ndarray]], allowed: Sequence[str],
           repeats: int = REPEATS, seed: int = SEED) -> dict:
    """Choose a setting on half of the compounds and score it on the other half.

    `per` maps a setting to the per-compound scores of :func:`scores`.  In every repeat the
    compounds are split at random into two halves; the setting of `allowed` with the highest
    mean reciprocal rank on the first half is scored on the second.  Returns the mean and the
    standard deviation over the repeats, how often every setting was chosen, and for every
    compound its share of first places over the repeats in which it was scored.
    """
    allowed = list(allowed)
    n = len(per[allowed[0]]["first"])
    rng = np.random.default_rng(seed)
    got = {k: [] for k in ("first", "top3", "top10")}
    chosen: Dict[str, int] = {}
    total, count = np.zeros(n), np.zeros(n)
    for _ in range(repeats):
        order = rng.permutation(n)
        cal, test = order[: n // 2], order[n // 2:]
        best = max(allowed, key=lambda s: float(np.mean(per[s]["rr"][cal])))
        chosen[best] = chosen.get(best, 0) + 1
        for k in got:
            got[k].append(float(np.mean(per[best][k][test])))
        total[test] += per[best]["first"][test]
        count[test] += 1
    out = {k: float(np.mean(v)) for k, v in got.items()}
    out.update({k + "_sd": float(np.std(v)) for k, v in got.items()})
    out["chosen"] = dict(sorted(chosen.items(), key=lambda kv: -kv[1]))
    out["per_compound_first"] = (total / np.maximum(count, 1)).tolist()
    out["scored"] = (count > 0).tolist()
    return out


def mean_difference(a: Sequence[float], b: Sequence[float], n_boot: int = 4000,
                    seed: int = SEED) -> dict:
    """The mean of ``a - b`` over the compounds with a 95 % bootstrap interval; the compounds
    are drawn with replacement."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {"mean": float(d.mean()), "low": float(np.percentile(draws, 2.5)),
            "high": float(np.percentile(draws, 97.5))}


# --------------------------------------------------------------------------- #
# The structures, for the reading of the wrong answers
# --------------------------------------------------------------------------- #
def structures(smiles: Sequence[str]) -> Tuple[List[str], list]:
    """The molecular formula and a Morgan fingerprint (radius 2, 2048 bits) of every SMILES;
    an empty formula and ``None`` for a SMILES that RDKit does not read."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    formulas, prints = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        formulas.append(CalcMolFormula(mol) if mol is not None else "")
        prints.append(gen.GetFingerprint(mol) if mol is not None else None)
    return formulas, prints


def tanimoto(prints: list, a: int, b: int) -> Optional[float]:
    """The Tanimoto similarity of two fingerprints: the number of shared bits over the number
    of bits that either has.  ``None`` when a structure was not read."""
    from rdkit import DataStructs
    if prints[a] is None or prints[b] is None:
        return None
    return float(DataStructs.TanimotoSimilarity(prints[a], prints[b]))


def wrong_answers(ranks: Dict[str, np.ndarray], first: np.ndarray, formulas: List[str],
                  prints: list, seed: int = SEED) -> dict:
    """What the kernel names when it does not name the compound of the query: how similar
    that compound is to the right one, against a compound drawn at random from the library."""
    rng = np.random.default_rng(seed)
    wrong = np.where(ranks["rank"] > 1)[0]
    named, random, same = [], [], []
    for k in wrong:
        q, w = int(ranks["query"][k]), int(ranks["named"][k])
        t = tanimoto(prints, q, w)
        r = tanimoto(prints, q, int(rng.choice(first)))
        if t is None or r is None:
            continue
        named.append(t)
        random.append(r)
        same.append(formulas[q] == formulas[w])
    if not named:
        return {"n": 0}
    return {"n": len(named), "tanimoto_named": float(np.mean(named)),
            "tanimoto_named_median": float(np.median(named)),
            "tanimoto_random": float(np.mean(random)),
            "share_similar": float(np.mean(np.array(named) >= 0.5)),
            "share_same_formula": float(np.mean(same))}


def nearest_other(U: np.ndarray, first: np.ndarray, prints: list, seed: int = SEED) -> dict:
    """For every compound of the library, the other compound with the most similar spectrum:
    how similar the two structures are, against a pairing at random."""
    L = U[first]
    S = L @ L.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(axis=1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(first))
    near = [tanimoto(prints, int(first[a]), int(first[b])) for a, b in enumerate(nn)]
    rand = [tanimoto(prints, int(first[a]), int(first[b]))
            for a, b in enumerate(perm) if a != b]
    near = np.array([v for v in near if v is not None])
    rand = np.array([v for v in rand if v is not None])
    return {"n": int(len(near)), "tanimoto_nearest": float(near.mean()),
            "tanimoto_nearest_median": float(np.median(near)),
            "tanimoto_random": float(rand.mean()),
            "share_similar": float(np.mean(near >= 0.5)),
            "share_similar_random": float(np.mean(rand >= 0.5))}


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def run(results: str = RESULTS, repeats: int = REPEATS) -> dict:
    t0 = time.time()
    x, A, table = PREP.load()
    smiles = [r["smiles"] for r in table]
    keep, copies = without_copies(A, smiles)
    shared = [[table[i]["sample_id"] for i in rows]
              for rows in copies["sets_shared_by_different_compounds"]]
    A = A[keep]
    table = [r for r, k in zip(table, keep) if k]
    smiles = [r["smiles"] for r in table]
    groups = grouped(smiles)
    first = np.array([g[0] for g in groups])
    several = [g for g in groups if len(g) > 1]
    log(f"{len(table)} spectra of {len(groups)} compounds; {len(several)} compounds have "
        f"two or more spectra")

    mean_free = A - A.mean(axis=1, keepdims=True)
    unit = mean_free / np.linalg.norm(mean_free, axis=1, keepdims=True)
    replicate_r = [float(unit[a] @ unit[b]) for g in several
                   for i, a in enumerate(g) for b in g[i + 1:]]

    per, on_all, ranks_of = {}, {}, {}
    for width in WIDTHS:
        F = broadened(A, x, width)
        for form in FORMS:
            key = setting_name(form, width)
            ranks_of[key] = pair_ranks(formed(F, form), groups)
            per[key] = scores(ranks_of[key]["rank"], ranks_of[key]["compound"])
            on_all[key] = {k: float(np.mean(v)) for k, v in per[key].items()}
            log(f"  {shown(key):44s} first {on_all[key]['first']:.3f}  "
                f"top ten {on_all[key]['top10']:.3f}")

    fits = {"all forms": nested(per, list(per), repeats)}
    for form in FORMS:
        fits[form] = nested(per, [setting_name(form, w) for w in WIDTHS], repeats)
    differences = {form: mean_difference(fits["all forms"]["per_compound_first"],
                                         fits[form]["per_compound_first"])
                   for form in FORMS}

    best = next(iter(fits["all forms"]["chosen"]))
    form, width = best.split("|")
    U = formed(broadened(A, x, float(width)), form)
    gx = GRID.x
    regions = {}
    for name, lo, hi in REGIONS:
        cols = (gx >= lo) & (gx <= hi)
        part = formed(broadened(A, x, float(width))[:, cols], form)
        r = pair_ranks(part, groups)
        regions[name] = {k: float(np.mean(v))
                         for k, v in scores(r["rank"], r["compound"]).items()}
    formulas, prints = structures(smiles)
    n_formula: Dict[str, int] = {}
    for i in first:
        n_formula[formulas[i]] = n_formula.get(formulas[i], 0) + 1
    composition = float(np.mean([1.0 / n_formula[formulas[g[0]]] for g in several]))

    out = {
        "source": PREP.DOI, "licence": PREP.LICENCE,
        "spectra": len(table), "compounds": len(groups),
        "compounds_with_several_spectra": len(several),
        "spectra_per_compound": {str(n): int(sum(len(g) == n for g in several))
                                 for n in sorted({len(g) for g in several})},
        "pairs": int(len(ranks_of[best]["rank"])),
        "copies_within_a_compound": copies["copies_within_a_compound"],
        "sample_ids_of_curves_shared_by_different_compounds": shared,
        "replicate_correlation_median": float(np.median(replicate_r)),
        "replicate_correlation_tenth": float(np.percentile(replicate_r, 10)),
        "grid": [GRID.x_min, GRID.x_max, GRID.step], "repeats": repeats,
        "chance": 1.0 / len(groups), "composition_alone": composition,
        "settings_on_all_compounds": on_all, "nested": fits,
        "chosen_minus_form": differences, "setting_of_the_readings": best,
        "regions": regions,
        "wrong_answers": wrong_answers(ranks_of[best], first, formulas, prints),
        "misses": misses(ranks_of[best], unit),
        "nearest_other_compound": nearest_other(U, first, prints),
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "measured.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def _pct(v: float) -> str:
    return f"{100.0 * v:.1f} %"


def report(results: str = RESULTS) -> str:
    """The table of the benchmark as text; it is also written to ``table.md``."""
    path = os.path.join(results, "measured.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    n = d["nested"]
    per = ", ".join(f"{v} with {k}" for k, v in d["spectra_per_compound"].items())
    lines = [
        "# A measured infrared spectrum against other measured spectra of the same compounds",
        "",
        f"Set: Chemotion IR ({d['source']}, {d['licence']}), fetched by "
        "`benchmarks/prepare_chemotion_ir.py`; no file of the set is in this repository.",
        f"After the copies were left out: {d['spectra']} measured spectra of "
        f"{d['compounds']} compounds. {d['compounds_with_several_spectra']} compounds have "
        f"two or more spectra of different samples ({per}); they give {d['pairs']} pairs of "
        "a query and a library spectrum.",
        f"Left out: {d['copies_within_a_compound']} curves that are a copy of another curve "
        "of the same compound, and "
        f"{sum(len(s) for s in d['sample_ids_of_curves_shared_by_different_compounds'])} "
        "curves that are "
        f"{len(d['sample_ids_of_curves_shared_by_different_compounds'])} files, each "
        "attached to the samples of two different compounds.",
        "The library holds one spectrum of every compound. A compound counts once, whatever "
        "its number of spectra. The form of the kernel and the width are chosen on half of "
        f"the compounds with several spectra and scored on the other half, {d['repeats']} "
        "random splits; the number after the sign +/- is the standard deviation over the "
        "splits.",
        "",
        "## How often the compound of the query is found",
        "",
        "| Description of the spectrum | Compound is first | In the first three | "
        "In the first ten |",
        "|---|---|---|---|",
        f"| chance (one of {d['compounds']} compounds) | {_pct(d['chance'])} | "
        f"{_pct(3 * d['chance'])} | {_pct(10 * d['chance'])} |",
    ]
    rows = (("plain", "plain kernel of the absorbance, width chosen"),
            ("own mean removed", "the mean of every spectrum taken off (the correlation "
                                 "coefficient of two spectra), width chosen"),
            ("slope", "kernel of the slopes of the spectra, width chosen"),
            ("all forms", "form and width chosen"))
    for key, label in rows:
        f = n[key]
        lines.append(f"| {label} | {_pct(f['first'])} +/- {100 * f['first_sd']:.1f} | "
                     f"{_pct(f['top3'])} +/- {100 * f['top3_sd']:.1f} | "
                     f"{_pct(f['top10'])} +/- {100 * f['top10_sd']:.1f} |")
    for name, r in d["regions"].items():
        lines.append(f"| the chosen form on {name} (all compounds, no choice) | "
                     f"{_pct(r['first'])} | {_pct(r['top3'])} | {_pct(r['top10'])} |")
    lines += ["",
              "Chosen settings over the splits: "
              + "; ".join(f"{shown(k)} in {v}" for k, v in n["all forms"]["chosen"].items())
              + ".",
              "",
              "Difference in the share of first places, per compound, with a 95 % bootstrap "
              "interval over the compounds:",
              ""]
    for form in FORMS:
        df = d["chosen_minus_form"][form]
        if df["mean"] == 0 and df["low"] == 0 and df["high"] == 0:
            lines.append(f"- the choice among all forms took the form '{form}' in every "
                         "split, so the two rows are the same")
            continue
        lines.append(f"- chosen form minus the form '{form}': {100 * df['mean']:+.1f} "
                     f"[{100 * df['low']:+.1f}, {100 * df['high']:+.1f}] points")
    lines += ["",
              "## Every setting on all compounds (no choice, for the reader)",
              "",
              "| Form and width | Compound is first | In the first three | "
              "In the first ten |",
              "|---|---|---|---|"]
    for key, r in d["settings_on_all_compounds"].items():
        lines.append(f"| {shown(key)} | {_pct(r['first'])} | {_pct(r['top3'])} | "
                     f"{_pct(r['top10'])} |")
    w, o, m = d["wrong_answers"], d["nearest_other_compound"], d["misses"]
    lines += ["",
              "## What the numbers say about the set and about the kernel",
              "",
              f"- Two spectra of one compound from different samples have a correlation "
              f"coefficient of {d['replicate_correlation_median']:.3f} (median); a tenth of "
              f"the pairs is below {d['replicate_correlation_tenth']:.3f}.",
              f"- The molecular formula alone would name the compound in "
              f"{_pct(d['composition_alone'])} of the queries: the compounds of the set are "
              "diverse, and few share a formula. The formula is not known from a spectrum, "
              "so this is a statement about the library and not a rival of the kernel.",
              f"- Over the whole library ({o['n']} compounds), the other compound with the "
              f"most similar spectrum has a Tanimoto similarity of "
              f"{o['tanimoto_nearest']:.2f} on average (median "
              f"{o['tanimoto_nearest_median']:.2f}) and of 0.5 or more in "
              f"{_pct(o['share_similar'])}; a pairing at random gives "
              f"{o['tanimoto_random']:.2f} and {_pct(o['share_similar_random'])}.",
              ""]
    if w["n"]:                          # a kernel that is always right names no other compound
        lines.insert(-1, (
            f"- When the kernel ({shown(d['setting_of_the_readings'])}, all compounds) "
            f"names another compound ({w['n']} pairs), the Tanimoto similarity of that "
            f"compound to the right one is {w['tanimoto_named']:.2f} on average (median "
            f"{w['tanimoto_named_median']:.2f}), against {w['tanimoto_random']:.2f} for a "
            f"compound drawn at random. In {_pct(w['share_similar'])} of these pairs the "
            f"similarity is 0.5 or more, and in {_pct(w['share_same_formula'])} the named "
            "compound has the same formula (an isomer)."))
    if m["pairs"]:
        lines.insert(-1, (
            f"- The {m['pairs']} pairs in which the compound is not among the first "
            f"{m['beyond']} belong to {m['compounds']} compounds. The two spectra of such "
            f"a pair have a correlation coefficient of {m['correlation_median']:.2f} "
            f"(median; the highest is {m['correlation_highest']:.2f}), against "
            f"{d['replicate_correlation_median']:.2f} for all pairs: the two samples did "
            "not give the same spectrum, whatever the reason."))
    lines += _verdict(d)
    text = "\n".join(lines)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def _verdict(d: dict) -> List[str]:
    n = d["nested"]
    best, plain = n["all forms"], n["plain"]
    gain = d["chosen_minus_form"]["plain"]
    clear = gain["low"] > 0
    return ["## Reading",
            "",
            f"A measured spectrum finds its own compound among {d['compounds']} compounds "
            f"in {_pct(best['first'])} of the queries and puts it into the first ten in "
            f"{_pct(best['top10'])}. This is the ceiling for a query against a computed "
            "library of the same compounds: there the difference between a calculation and "
            "a measurement is added to the difference between two samples.",
            f"The plain kernel finds {_pct(plain['first'])}. The difference is "
            f"{100 * gain['mean']:+.1f} points "
            f"[{100 * gain['low']:+.1f}, {100 * gain['high']:+.1f}]"
            + (", so the form of the kernel matters on measured spectra: a measured curve "
               "has a baseline, and the plain kernel counts it as intensity."
               if clear else ", which the interval does not separate from zero."),
            ""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    a = ap.parse_args(argv)
    if not a.report:
        run(repeats=a.repeats)
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
