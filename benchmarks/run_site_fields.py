#!/usr/bin/env python
"""
Can a field of the site beat a count of its neighbours?  (plan item PP3)

Plan item PP1 found that what the field of a residue's surroundings adds to a substitution
table, for the change of folding free energy of a point mutant, is how buried the site is, and
that one count of the atoms around the site gives the same: a Gaussian kernel on that count and
three substitution columns reaches q2 +0.458 on 280 positions, ahead of the field.  PP3 tries
the descriptions of the surroundings that a count cannot imitate, against the references a
biochemist would reach for.

The samples are those of the second stage of `benchmarks/run_mutants.py`: every measured
position of the four domains of the mega-scale set (Tsuboyama et al., Nature 2023), four
substitutions drawn per position (two draws), the surroundings taken from the wild-type
structure (the alpha carbon's environment under the envelope rho = 0.1 per square angstrom, the
residue's own side chain left out), and the target ddG in kcal/mol.  Every description is joined
with the same three substitution columns (the change of volume, hydropathy and charge).

The descriptions
----------------
References:

* **one count** -- a Gaussian kernel on the number of atoms in the envelope and the three
  columns: the reference PP1 ends with;
* **burial columns** -- PP1's counts and soft counts at three envelopes, as a Gaussian kernel;
* **accessible area** -- the solvent-accessible surface area of the wild-type residue and of
  its side chain (Shrake and Rupley, van der Waals radii, 1.4 A probe, by mdtraj), as a
  Gaussian kernel: the quantity a biochemist would reach for, which PP3 has to add.

Descriptions a count cannot imitate:

* **typed fields, cosine** -- the five field kernels of the surroundings (electrostatic, steric,
  hydrophobic, hydrogen-bond acceptor and donor) each normalised to unit self-similarity, and
  offered alone or as their mean: a count does not know an acceptor from a carbon.  The mean is
  PP1's context kernel;
* **typed fields, raw** -- the same kernels without the normalisation, each divided by the mean
  of its diagonal, so that the amount of field stays in.  PP1 left open whether the cosine
  normalisation, which divides out the size of a field and with it the count, is why the field
  only equals the count; this tests it;
* **anisotropy** -- for every field type, the amount of field, its centre in the residue frame
  and the three principal second moments about that centre: a count has no direction.  Also
  without the amounts, so that no count enters;
* **anchored points** -- the value and the size of the gradient of every field at seven points
  of the residue frame along and around the direction of the side chain (the virtual beta
  carbon), as plan item EC3 builds them.

Every kernel on columns is a Gaussian on standardised columns with the width the reference
uses; the families of field kernels are joined with the column kernel at the two weights PP1
offers, and every choice (member of a family, weight, ridge) is made in inner folds of the same
kind as the outer ones.  Hold-outs: nine consecutive residues (segment) and whole domains, with
positions for continuity; ten folds (four for domains), three repeats.  Gate: a description
beats both references -- one count, and the better of the burial columns and the accessible
area -- by more than the interval of rule 4, with segments and with domains held out.

Usage
-----
    python benchmarks/run_site_fields.py             # about an hour
    python benchmarks/run_site_fields.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_mutants as M  # noqa: E402
from cmfqsar import mutants as MU  # noqa: E402
from cmfqsar.descriptors import field_at_points, field_gradient_at_points  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet  # noqa: E402
from cmfqsar.kernels import kernel_matrices, normalize_kernels, self_kernels  # noqa: E402

RESULTS = os.path.join(ROOT, "benchmarks", "results", "site_fields")
RHO = M.SITE_RHO
ALPHA = M.CONTEXT_ALPHA
#: distances of the anchored points from the alpha carbon along the side chain, angstrom
ANCHOR_ALONG = (2.0, 4.0, 6.0)
#: the four points around the middle one, this far off the axis
ANCHOR_SIDE = 2.0
#: main-chain atom names, whose area is not the side chain's
MAIN_CHAIN = {"N", "CA", "C", "O", "H", "HA", "H1", "H2", "H3", "OXT", "HA2", "HA3"}
HOLDOUTS = ("segment", "domain", "position")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Descriptions of a position
# --------------------------------------------------------------------------- #
def fieldset() -> FieldSet:
    fs = FieldSet.create(charge_model="file", alpha=ALPHA)
    fs.set_alpha(float(ALPHA))
    return fs


def typed_kernels(contexts, keys) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(cosine, raw) kernels of every field type over the positions, and the field names.

    The raw kernel of a field is divided by the mean of its diagonal, so the five are on one
    scale while each keeps how much field a site has."""
    fs = fieldset()
    fields = [contexts[(float(RHO), d, p)] for d, p in keys]
    table = AtomTable.from_molecules(fields)
    raw = np.asarray(kernel_matrices(table, table, fs.specs, symmetric=True), float)
    diag = np.asarray(self_kernels(table, fs.specs), float)
    cos = np.asarray(normalize_kernels(raw.copy(), diag, diag), float)
    scaled = np.stack([raw[f] / max(float(np.mean(np.diag(raw[f]))), 1e-300)
                       for f in range(raw.shape[0])])
    return cos, scaled, [s.name for s in fs.specs]


def anisotropy_rows(contexts, keys) -> Tuple[np.ndarray, List[str]]:
    """Per field type: the amount, the centre in the residue frame and the principal second
    moments about it.  The weights are taken as their magnitudes, so a field of mixed sign
    (the electrostatic one) is described by where it is, not by its net value."""
    fs = fieldset()
    names = [s.name for s in fs.specs]
    rows = []
    for d, p in keys:
        c = contexts[(float(RHO), d, p)]
        X = np.asarray(c.coords, float)
        W = np.abs(np.asarray(c.weights, float))
        row = []
        for f in range(W.shape[1]):
            m = W[:, f]
            total = float(m.sum())
            if total <= 0:
                row += [0.0] * 7
                continue
            centre = (m[:, None] * X).sum(axis=0) / total
            D = X - centre
            S = (m[:, None, None] * D[:, :, None] * D[:, None, :]).sum(axis=0) / total
            row += [total, *centre.tolist(), *np.sort(np.linalg.eigvalsh(S))[::-1].tolist()]
        rows.append(row)
    labels = [f"{n}:{k}" for n in names for k in ("amount", "x", "y", "z", "m1", "m2", "m3")]
    return np.asarray(rows, float), labels


def anchor_points() -> np.ndarray:
    """Seven points of the residue frame: along the direction of a virtual beta carbon from the
    ideal backbone (N 1.458 A from CA along x, C 1.525 A at 111 degrees in the xy plane), and
    four around the middle one."""
    n = np.array([1.458, 0.0, 0.0])
    ang = np.radians(111.0)
    c = 1.525 * np.array([np.cos(ang), np.sin(ang), 0.0])
    b, cc = -n, c                                   # CA - N and C - CA, with CA at the origin
    a = np.cross(b, cc)
    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc
    u = cb / np.linalg.norm(cb)
    v = np.cross(u, [0.0, 0.0, 1.0])
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    pts = [u * t for t in ANCHOR_ALONG]
    mid = u * ANCHOR_ALONG[1]
    pts += [mid + ANCHOR_SIDE * s * e for e in (v, w) for s in (1.0, -1.0)]
    return np.asarray(pts, float)


def anchored_rows(contexts, keys) -> Tuple[np.ndarray, List[str]]:
    fs = fieldset()
    pts = anchor_points()
    rows = []
    for d, p in keys:
        c = contexts[(float(RHO), d, p)]
        row = []
        for f, spec in enumerate(fs.specs):
            row += field_at_points(pts, c, spec, f).tolist()
            row += np.linalg.norm(field_gradient_at_points(pts, c, spec, f), axis=1).tolist()
        rows.append(row)
    labels = [f"{s.name}:{k}:{i}" for s in fs.specs for k in ("value", "grad")
              for i in range(len(pts))]
    return np.asarray(rows, float), labels


def accessible_area(domains=M.DOMAINS) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """(residue, side chain) solvent-accessible area of every residue of the wild types, in
    square angstrom: Shrake and Rupley with mdtraj's van der Waals radii and a 1.4 A probe."""
    import mdtraj as md
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for d in domains:
            path = os.path.join(tmp, d + ".pdb")
            MU.prepare_wild_type(os.path.join(M.EXAMPLES, d, d + ".pdb"), path)
            t = md.load(path)
            atom = md.shrake_rupley(t, probe_radius=0.14, mode="atom")[0] * 100.0
            for res in t.topology.residues:
                total = sum(float(atom[a.index]) for a in res.atoms)
                side = sum(float(atom[a.index]) for a in res.atoms if a.name not in MAIN_CHAIN)
                out[(d, int(res.resSeq))] = (total, side)
    return out


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def expand(keys, X: np.ndarray, s) -> np.ndarray:
    index = {k: i for i, k in enumerate(keys)}
    rows = np.array([index[(d, p)] for d, p in zip(s.domain, s.position)], int)
    return X[rows]


def family(kernels: Dict[str, np.ndarray], Kc3: np.ndarray) -> Dict[str, np.ndarray]:
    """Every member of a family of position kernels joined with the column kernel."""
    return {f"{name}, weight {w}": M.mixed(Kc3, K, w) for name, K in kernels.items()
            for w in M.MIXES}


COMPARE_TO = ("one count", "burial columns", "accessible area")
CANDIDATES = ("typed fields, cosine", "typed fields, raw", "anisotropy",
              "anisotropy without the amounts", "anchored points")


def measure(s, pos, seeds, holdouts) -> dict:
    """Every description under every hold-out, with the differences the gate reads."""
    keys, cos, raw, names, X_aniso, X_anchor, area = pos
    X_three = M.MU.substitution_features(s.mutation)[0][:, :3]
    Kc3 = M.gaussian_columns(X_three)
    expand_k = (lambda K: M.expand_position_kernel(keys, K, s))
    cos_s = {n: expand_k(cos[f]) for f, n in enumerate(names)}
    cos_s["mean of the five"] = expand_k(cos.mean(axis=0))
    raw_s = {n: expand_k(raw[f]) for f, n in enumerate(names)}
    raw_s["mean of the five"] = expand_k(raw.mean(axis=0))
    aniso = expand(keys, X_aniso, s)
    no_amount = np.delete(aniso, [7 * f for f in range(len(names))], axis=1)
    X_area = np.array([area[(d, p)] for d, p in zip(s.domain, s.position)], float)
    kernels = {
        "one count": {"one count": M.gaussian_columns(M.burial_count_rows(s))},
        "burial columns": {"burial": M.gaussian_columns(M.burial_rows(s))},
        "accessible area": {"area": M.gaussian_columns(np.c_[X_area, X_three])},
        "three columns kernel": {"columns": Kc3},
        "typed fields, cosine": family(cos_s, Kc3),
        "typed fields, raw": family(raw_s, Kc3),
        "anisotropy": {"anisotropy": M.gaussian_columns(np.c_[aniso, X_three])},
        "anisotropy without the amounts": {
            "directions": M.gaussian_columns(np.c_[no_amount, X_three])},
        "anchored points": {"anchored": M.gaussian_columns(
            np.c_[expand(keys, X_anchor, s), X_three])},
    }
    out = {}
    for by in holdouts:
        tg = s.targets(by)
        n_groups = int(tg.molecule.max()) + 1
        folds = min(M.FOLDS, n_groups) if by != "domain" else min(len(set(s.domain)), 4)
        rows, keep = {}, {}
        res, chosen = M.mean_reference(tg, seeds, folds)
        rows["mean"], keep["mean"] = M.row_of(res, chosen, tg), res
        for name, fam in kernels.items():
            res, chosen = M.nested(fam, tg, seeds, name, folds)
            rows[name], keep[name] = M.row_of(res, chosen, tg), res
            log(f"    {by:8s} {name:32s} q2 {rows[name]['q2']:+.3f}")
        diffs = {}
        for a in CANDIDATES:
            for b in COMPARE_TO:
                diffs[f"{a} - {b}"] = M.compare(keep[a], keep[b], tg)
        diffs["typed fields, raw - typed fields, cosine"] = M.compare(
            keep["typed fields, raw"], keep["typed fields, cosine"], tg)
        diffs["accessible area - one count"] = M.compare(keep["accessible area"],
                                                         keep["one count"], tg)
        out[by] = {"rows": rows, "differences": diffs, "n_groups": n_groups, "folds": folds}
    return out


def run(draws=M.SITE_DRAWS, repeats: int = M.REPEATS, holdouts=HOLDOUTS,
        results: str = RESULTS) -> dict:
    os.makedirs(results, exist_ok=True)
    t0 = time.time()
    contexts = M.site_contexts()
    keys = sorted({(d, p) for (r, d, p) in contexts if r == float(RHO)})
    log(f"{len(keys)} positions, contexts after {time.time() - t0:.0f} s")
    cos, raw, names = typed_kernels(contexts, keys)
    log(f"  typed kernels after {time.time() - t0:.0f} s")
    X_aniso, aniso_labels = anisotropy_rows(contexts, keys)
    X_anchor, anchor_labels = anchored_rows(contexts, keys)
    area = accessible_area()
    log(f"  anisotropy, anchored points and accessible area after {time.time() - t0:.0f} s")
    # how far each description of a position is from a count of its atoms
    count = np.array([float(contexts[(float(RHO), d, p)].n_atoms) for d, p in keys])
    side = np.array([area[k][1] for k in keys])
    total = np.array([area[k][0] for k in keys])
    out = {"positions": len(keys), "rho": RHO, "alpha": ALPHA, "fields": names,
           "anchor_points": anchor_points().tolist(), "repeats": repeats,
           "correlation_with_the_count": {
               "residue area": float(np.corrcoef(count, total)[0, 1]),
               "side-chain area": float(np.corrcoef(count, side)[0, 1]),
               **{f"amount of {n}": float(np.corrcoef(count, X_aniso[:, 7 * f])[0, 1])
                  for f, n in enumerate(names)}},
           "draws": {}}
    pos = (keys, cos, raw, names, X_aniso, X_anchor, area)
    seeds = list(range(repeats))
    for draw in draws:
        s = M.SiteSamples(contexts, M.DOMAINS, M.MAX_PER_POSITION, draw)
        log(f"--- draw {draw}: {len(s)} substitutions, ddG sd {s.y.std():.2f} kcal/mol")
        out["draws"][str(draw)] = {"n": len(s), "sd_y": round(float(s.y.std()), 4),
                                   "holdout": measure(s, pos, seeds, holdouts)}
        with open(os.path.join(results, "site_fields.json"), "w", encoding="utf-8",
                  newline="\n") as fh:
            json.dump(out, fh, indent=1)
    out["seconds"] = round(time.time() - t0, 1)
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(results, "site_fields.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(out, fh, indent=1)
    log(f"done after {out['seconds']:.0f} s")
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
ROWS = ("mean", "three columns kernel", "one count", "burial columns", "accessible area",
        "typed fields, cosine", "typed fields, raw", "anisotropy",
        "anisotropy without the amounts", "anchored points")


def _iv(d: Sequence[float]) -> str:
    txt = f"{d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f})"
    return f"**{txt}**" if d[1] > 0 or d[2] < 0 else txt


def report(results: str = RESULTS) -> str:
    path = os.path.join(results, "site_fields.json")
    if not os.path.exists(path):
        return "# PP3: no result yet\n"
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    L = ["# Can a field of the site beat a count of its neighbours? (plan item PP3)\n",
         f"`python benchmarks/run_site_fields.py`, written {d.get('written', '?')}, "
         f"{d.get('seconds', 0) / 60:.0f} minutes.\n",
         f"{d['positions']} positions of four domains of the mega-scale set, four substitutions "
         "drawn at each, the surroundings from the wild-type structure (envelope "
         f"{d['rho']} per square angstrom), every description joined with the change of "
         "volume, hydropathy and charge. q2 in cross-validation, the mean of "
         f"{d['repeats']} repeats; every choice is made in inner folds of the same kind.\n",
         "How close a description of the position is to the count of atoms in the envelope "
         "(Pearson correlation over the positions):\n",
         "| Description | Correlation with the count |", "|---|---:|"]
    for k, v in d["correlation_with_the_count"].items():
        L.append(f"| {k} | {v:+.3f} |")
    L.append("")
    for draw, part in d["draws"].items():
        L.append(f"## Draw {draw}: {part['n']} substitutions, sd of ddG {part['sd_y']:.2f} "
                 "kcal/mol\n")
        hs = list(part["holdout"])
        L.append("| Description | " + " | ".join(f"q2, {h}s held out" for h in hs) + " |")
        L.append("|---|" + "---:|" * len(hs))
        for name in ROWS:
            if all(name in part["holdout"][h]["rows"] for h in hs):
                L.append(f"| {name} | " + " | ".join(
                    f"{part['holdout'][h]['rows'][name]['q2']:+.3f}" for h in hs) + " |")
        L.append("")
        L.append("Differences in q2, with the 95 % interval over the repeats and the folds; "
                 "bold where it excludes zero.\n")
        L.append("| Difference | " + " | ".join(f"{h}s held out" for h in hs) + " |")
        L.append("|---|" + "---|" * len(hs))
        for key in part["holdout"][hs[0]]["differences"]:
            L.append(f"| {key} | " + " | ".join(
                _iv(part["holdout"][h]["differences"][key]) for h in hs) + " |")
        L.append("")
    return "\n".join(L)


def write_table(results: str = RESULTS) -> str:
    path = os.path.join(results, "table.md")
    text = report(results)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repeats", type=int, default=M.REPEATS)
    p.add_argument("--draws", type=int, nargs="*", default=list(M.SITE_DRAWS))
    p.add_argument("--holdout", nargs="*", default=list(HOLDOUTS))
    p.add_argument("--report", action="store_true")
    args = p.parse_args(argv)
    if not args.report:
        run(args.draws, args.repeats, tuple(args.holdout))
    print(write_table(RESULTS))


if __name__ == "__main__":
    main()
