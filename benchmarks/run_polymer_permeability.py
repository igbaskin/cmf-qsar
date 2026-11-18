#!/usr/bin/env python
"""
Plan item PL5: what the 3D field sees in a permeable polymer.

PL2 (``run_polymer_packing.py``) found that on the measured gas permeability of 742 polymers the
field kernel of the trimer's middle unit plus eight counts beats the same channels with distances
counted in bonds plus the counts, by +0.06 to +0.08 in q2 with clusters of similar polymers held
out.  It did not say what the 3D distances carry.  This script asks whether one number does the
work, as the count references of the other items did: each candidate is added as a column of the
counts kernel, and the field has to beat the bond-count kernel plus the counts plus that number.

The candidates, one number per polymer
--------------------------------------
* **span per mass** -- the end-to-end distance of the extended trimer (between its first and last
  backbone atoms, after the optimisation that holds it) divided by the trimer's molar mass: how
  far a stiff chain reaches per gram.
* **radius of gyration per mass** -- the radius of gyration of all atoms of the trimer, divided by
  its molar mass.
* **side-chain fraction** -- the fraction of the repeat unit's heavy atoms that are not on the
  backbone.  The backbone is the PolyMetriX rule: the shortest path between the two atoms that
  carry the attachment points, every ring that shares an atom with it (repeated, so that fused
  rings join), and every heavy atom with one neighbour that hangs on the backbone.
* **largest side group** -- the number of heavy atoms of the largest connected piece of the side
  chains.
* **van der Waals volume per mass** -- RDKit's ``ComputeMolVolume`` of the trimer (a grid over the
  van der Waals spheres) divided by its molar mass: the bulk the atoms take per gram, the
  quantity a group-contribution estimate of free volume starts from.

And all five together.  The first two and the last need the 3D trimer; the side-chain numbers
need only the graph.

The question, fixed before the measurement
------------------------------------------
For each gas and fold scheme (those of PL2: five random folds and five folds of ECFP4 Butina
clusters, each repeated ten times):

1. does a candidate add to the graph reference -- bond-count kernel + counts + candidate minus
   bond-count kernel + counts;
2. does the field's margin survive it -- central unit + counts minus bond-count kernel + counts +
   candidate.  If this interval reaches zero where PL2's did not, the field was worth that number;
3. for all five together, central unit + counts + five minus bond-count kernel + counts + five.

The counts kernel with a candidate is the Gaussian kernel of the eight counts on nine (or
thirteen) standardised columns.  Each part of a sum keeps the setting its single model chose on
the same training block; h and the ridge are chosen by the evidence.  Every evaluated setting
is kept in the cache of PL2, so the PL2 models are read, not refitted.

The distance ranges (``--ranges``, after the candidates)
---------------------------------------------------------
The central-unit field kernel is cut into the channels below 3 A (bonded pairs and pairs two
bonds apart) and those from 3 A on (pairs further apart along the chain, and pairs across a
fold of the side groups).  Each part plus the eight counts is compared with the bond-count
kernel plus the counts, and the full field plus the counts with each part.  If the short range
alone keeps the field's margin, the field gains its lead by typing near neighbours by their
exact distances, not by contacts through space.

Usage
-----
    python benchmarks/run_polymer_permeability.py
    python benchmarks/run_polymer_permeability.py --ranges
    python benchmarks/run_polymer_permeability.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

import run_polymer_packing as Q  # noqa: E402  (sets the globals of run_polymer_tg)

P, L = Q.P, Q.L
RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "polymer_permeability")
CANDIDATES = ("span per mass", "radius of gyration per mass", "side-chain fraction",
              "largest side group", "van der Waals volume per mass")
SETS = tuple((c, (c,)) for c in CANDIDATES) + (("all five", CANDIDATES),)
GASES = tuple(s for s in Q.TARGETS if s[0] != "density")
REF = ("central unit + eight counts", "bond-count kernel + eight counts")
log = L.log


# --------------------------------------------------------------------------- the candidates

def backbone_split(psmiles: str):
    """(side-chain fraction, size of the largest side group) of a repeat unit, by heavy atoms."""
    from cmfqsar.polymer import parse_repeat_unit
    unit = parse_repeat_unit(psmiles)
    m = unit.mol
    a = m.GetAtomWithIdx(unit.head).GetNeighbors()[0].GetIdx()
    b = m.GetAtomWithIdx(unit.tail).GetNeighbors()[0].GetIdx()
    back = set(Chem.GetShortestPath(m, a, b)) if a != b else {a}
    rings = [set(r) for r in m.GetRingInfo().AtomRings()]
    grown = True
    while grown:                                     # rings on the path, then rings fused to them
        grown = False
        for r in rings:
            if r & back and not r <= back:
                back |= r
                grown = True
    heavy = {x.GetIdx() for x in m.GetAtoms() if x.GetAtomicNum() > 1}
    for i in list(back):                             # a single atom hanging on the backbone
        for nb in m.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() in heavy and nb.GetDegree() == 1:
                back.add(nb.GetIdx())
    side = heavy - back
    sizes, seen = [], set()
    for s in side:
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            x = stack.pop()
            if x not in comp:
                comp.add(x)
                stack += [n.GetIdx() for n in m.GetAtomWithIdx(x).GetNeighbors()
                          if n.GetIdx() in side]
        seen |= comp
        sizes.append(len(comp))
    return len(side) / max(len(heavy), 1), (max(sizes) if sizes else 0)


def candidates(rows, keep) -> np.ndarray:
    """The five candidates for every built repeat unit, in the order of `keep`; cached."""
    from rdkit.Chem import AllChem, Descriptors
    path = os.path.join(Q.CACHE, "pl5_candidates.npz")
    ids = np.array([rows[i]["id"] for i in keep])
    if os.path.isfile(path):
        z = np.load(path)
        if z["ids"].shape == ids.shape and np.all(z["ids"] == ids):
            return z["X"]
    z = np.load(os.path.join(Q.CACHE, f"build_{Q.TAG}.npz"))
    sup = Chem.SDMolSupplier(os.path.join(Q.CACHE, f"trimers_{Q.TAG}.sdf"), removeHs=False)
    mols = {m.GetProp("_Name"): m for m in sup if m is not None}
    X = np.zeros((len(keep), len(CANDIDATES)))
    for j, i in enumerate(keep):
        r = rows[i]
        m = mols[r["id"]]
        mass = Descriptors.MolWt(m)
        xyz = m.GetConformer().GetPositions()
        rg = float(np.sqrt(((xyz - xyz.mean(axis=0)) ** 2).sum(axis=1).mean()))
        side, largest = backbone_split(r["psmiles"])
        X[j] = [z["span"][i] / mass, rg / mass, side, largest,
                AllChem.ComputeMolVolume(m) / mass]
    np.savez(path, ids=ids, X=X)
    return X


# --------------------------------------------------------------------------- the run

def target(spec, pos):
    """Rows, targets, groups, blocks and union indices of one gas, made exactly as PL2 makes them."""
    name, table, col, group_col, unit = spec
    rows = [r for r in Q.read(table) if r[col] != ""]
    keys = [Chem.MolToSmiles(Chem.MolFromSmiles(r["psmiles"])) for r in rows]
    have = [k for k, key in enumerate(keys) if key in pos]
    rows = [rows[k] for k in have]
    idx = np.array([pos[keys[k]] for k in have])
    y = np.array([float(r[col]) for r in rows])
    groups = Q.butina_groups([r["psmiles"] for r in rows])
    return rows, y, groups, Q.make_blocks(len(y), groups), idx


def run_gas(spec, kern, Zsets, pos, t0) -> dict:
    from cmfqsar.model import r2_score
    name = spec[0]
    rows, y, groups, blocks, idx = target(spec, pos)
    nb = len(blocks)
    sub = lambda K: K[np.ix_(idx, idx)]                       # noqa: E731
    P.Model.cache_dir = os.path.join(Q.CACHE, "settings")
    # the PL2 single models and sums, read from the cache under the names PL2 gave them
    single = {}
    for mname, grid, fmt, key in (("field kernel, central unit", P.SIGMAS, "sigma {:g}", "central"),
                                  ("bond-count kernel, central unit", P.TOPO_MAX,
                                   "up to {} bonds", "topo"),
                                  ("eight counts", P.COUNT_WIDTHS, "width {:g}", "counts")):
        m = P.Model(mname)
        for a in grid:
            m.add(fmt.format(a), lambda a=a, key=key: sub(kern.get((key, a))), y, blocks)
        single[mname] = [m.best_label(k) for k in range(nb)]
    models = {}

    def add_sum(mname, first, fmt, key, grid, cw, kc, label_fmt="{}, width {:g}, h {:g}"):
        m = P.Model(mname)
        for a in grid:
            for w in P.COUNT_WIDTHS:
                only = {k for k in range(nb) if single[first][k] == fmt.format(a)
                        and cw[k] == f"width {w:g}"}
                if not only:
                    continue
                for h in P.HS:
                    m.add(label_fmt.format(fmt.format(a), w, h),
                          lambda a=a, w=w, h=h: sub(kern.get((key, a))) + h * sub(kc(w)),
                          y, blocks, only)
        models[mname] = m

    cw0 = single["eight counts"]
    add_sum("central unit + eight counts", "field kernel, central unit", "sigma {:g}", "central",
            P.SIGMAS, cw0, lambda w: kern.get(("counts", w)))
    add_sum("bond-count kernel + eight counts", "bond-count kernel, central unit",
            "up to {} bonds", "topo", P.TOPO_MAX, cw0, lambda w: kern.get(("counts", w)))
    for sname, _ in SETS:
        Z = Zsets[sname]
        cm = P.Model(f"eight counts + {sname}")
        for w in P.COUNT_WIDTHS:
            cm.add(f"width {w:g}", lambda w=w, Z=Z: sub(L.gaussian_at(Z, w)), y, blocks)
        models[f"eight counts + {sname}"] = cm
        cw = [cm.best_label(k) for k in range(nb)]
        kc = (lambda w, Z=Z: L.gaussian_at(Z, w))
        add_sum(f"bond-count kernel + eight counts + {sname}", "bond-count kernel, central unit",
                "up to {} bonds", "topo", P.TOPO_MAX, cw, kc)
        if sname == "all five":
            add_sum(f"central unit + eight counts + {sname}", "field kernel, central unit",
                    "sigma {:g}", "central", P.SIGMAS, cw, kc)
        log(f"{name}: {sname} done ({time.time() - t0:.0f} s)")
    preds, res = {}, {}
    for mname, m in models.items():
        pred, chosen = m.select(len(y), blocks)
        res[mname] = {}
        for scheme in Q.SCHEMES:
            q2 = [float(r2_score(y, pred[scheme][r])) for r in range(Q.REPEATS)]
            res[mname][scheme] = {"q2_mean": float(np.mean(q2)), "q2_range": float(np.ptp(q2)),
                                  "chosen": chosen[scheme]}
            preds[(mname, scheme)] = pred[scheme].mean(axis=0)
    comps = []

    def compare(a, b):
        for scheme in Q.SCHEMES:
            pa, pb = preds[(a, scheme)], preds[(b, scheme)]
            g = (L.RS.paired(y, pa, pb) if scheme == "random"
                 else P.paired_groups(y, pa, pb, groups))
            comps.append({"model": a, "against": b, "scheme": scheme, **g})
            log(f"  {name:4s} {a} - {b} [{scheme}]: {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]")

    compare(*REF)
    for sname, _ in SETS:
        compare(f"bond-count kernel + eight counts + {sname}", REF[1])
        compare(REF[0], f"bond-count kernel + eight counts + {sname}")
    compare("central unit + eight counts + all five", "bond-count kernel + eight counts + all five")
    cors = {c: float(np.corrcoef(Zsets["all five"][idx, 8 + j], y)[0, 1])
            for j, c in enumerate(CANDIDATES)}
    return {"polymers": int(len(y)), "groups": int(len(set(groups.tolist()))),
            "models": res, "comparisons": comps, "correlation_with_target": cors}


def run() -> dict:
    t0 = time.time()
    rows, keys = Q.union_rows()
    z = np.load(os.path.join(Q.CACHE, f"build_{Q.TAG}.npz"))
    keep = np.where(z["ok"])[0]
    pos = {keys[i]: j for j, i in enumerate(keep)}
    feat = {"ok": z["ok"], "topo": z["topo"], "counts": z["counts"]}
    counts = L.standardise(z["counts"][keep])
    kern = Q.Kernels(feat, keep, [rows[i]["psmiles"] for i in keep], counts)
    X = candidates(rows, keep)
    log(f"candidates of {len(keep)} repeat units ({time.time() - t0:.0f} s)")
    Zsets = {s: L.standardise(np.hstack([z["counts"][keep], X[:, [CANDIDATES.index(c) for c in cols]]]))
             for s, cols in SETS}
    out = {spec[0]: run_gas(spec, kern, Zsets, pos, t0) for spec in GASES}
    payload = {"gases": out, "candidates": list(CANDIDATES), "repeats": Q.REPEATS,
               "seconds": round(time.time() - t0, 1)}
    L.jdump(payload, os.path.join(RESULTS, "polymer_permeability.json"))
    report(payload)
    return payload


# --------------------------------------------------------------------------- the distance ranges

#: where the short range ends: bonded pairs (up to 2.4 A for Si-Si) and pairs two bonds apart
#: (up to about 3.1 A around silicon) lie below it, pairs three or more bonds apart mostly above
R_CUT = 3.0
#: the widths the PL2 permeability models chose (0.005 A in most blocks, 0.01 A in the rest),
#: widened by one step on each side after a first run (0.005 to 0.02 A) chose both ends in 17 to
#: 20 % of the blocks
RANGE_SIGMAS = (0.0025, 0.005, 0.01, 0.02, 0.05)
RANGES = ("within 3 A", "beyond 3 A")


def range_kernel(F: np.ndarray, sigma: float, which: str) -> np.ndarray:
    """The cosine field kernel on the channels of one distance range only.

    The channels of each field pair are cut to the distances below (or from) :data:`R_CUT`; each
    pair's block is scaled by its own root mean square, as in the full kernel, and the rows are
    then normalised.
    """
    from cmfqsar.invariant import AutocorrelationSpec
    spec = AutocorrelationSpec(d_max=P.D_MAX, step=sigma, sigma=sigma)
    centres = spec.centres
    keep_bins = centres < R_CUT if which == "within 3 A" else centres >= R_CUT
    nb = spec.n_bins
    blocks = []
    for b in range(F.shape[1] // nb):
        blk = np.asarray(F[:, b * nb:(b + 1) * nb][:, keep_bins], np.float64)
        rms = float(np.sqrt(np.mean(blk ** 2)))
        if rms > 0:
            blocks.append((blk / rms).astype(np.float32))
    A = np.hstack(blocks)
    K = (A @ A.T).astype(np.float64)
    d = np.sqrt(np.maximum(np.diag(K), 1e-300))
    return K / np.outer(d, d)


def run_ranges() -> dict:
    """The central-unit field kernel cut to short or to long distances, alone and with counts."""
    t0 = time.time()
    rows, keys = Q.union_rows()
    z = np.load(os.path.join(Q.CACHE, f"build_{Q.TAG}.npz"))
    keep = np.where(z["ok"])[0]
    pos = {keys[i]: j for j, i in enumerate(keep)}
    feat = {"ok": z["ok"], "topo": z["topo"], "counts": z["counts"]}
    kern = Q.Kernels(feat, keep, [rows[i]["psmiles"] for i in keep],
                     L.standardise(z["counts"][keep]))
    store = {}

    def rk(sigma, which):
        if (sigma, which) not in store:
            F = P.autocorr_rows(feat, "central", sigma, Q.TAG, keep)
            store[(sigma, which)] = range_kernel(F, sigma, which)
        return store[(sigma, which)]

    out = {}
    for spec in GASES:
        from cmfqsar.model import r2_score
        name = spec[0]
        rows_t, y, groups, blocks, idx = target(spec, pos)
        nb = len(blocks)
        sub = lambda K: K[np.ix_(idx, idx)]                   # noqa: E731
        P.Model.cache_dir = os.path.join(Q.CACHE, "settings")
        cm = P.Model("eight counts")
        for w in P.COUNT_WIDTHS:
            cm.add(f"width {w:g}", lambda w=w: sub(kern.get(("counts", w))), y, blocks)
        cw = [cm.best_label(k) for k in range(nb)]
        models = {}
        for ref in REF:                                          # PL2's sums, from the cache
            first, fmt, key, grid = (("field kernel, central unit", "sigma {:g}", "central",
                                      P.SIGMAS) if ref == REF[0] else
                                     ("bond-count kernel, central unit", "up to {} bonds", "topo",
                                      P.TOPO_MAX))
            sm = P.Model(first)
            for a in grid:
                sm.add(fmt.format(a), lambda a=a, key=key: sub(kern.get((key, a))), y, blocks)
            chosen = [sm.best_label(k) for k in range(nb)]
            m = P.Model(ref)
            for a in grid:
                for w in P.COUNT_WIDTHS:
                    only = {k for k in range(nb) if chosen[k] == fmt.format(a)
                            and cw[k] == f"width {w:g}"}
                    if only:
                        for h in P.HS:
                            m.add(f"{fmt.format(a)}, width {w:g}, h {h:g}",
                                  lambda a=a, w=w, h=h, key=key: sub(kern.get((key, a)))
                                  + h * sub(kern.get(("counts", w))), y, blocks, only)
            models[ref] = m
        for which in RANGES:
            single = P.Model(f"field kernel, central unit, {which}")
            for s in RANGE_SIGMAS:
                single.add(f"sigma {s:g}", lambda s=s, which=which: sub(rk(s, which)), y, blocks)
            models[single.name] = single
            chosen = [single.best_label(k) for k in range(nb)]
            m = P.Model(f"central unit, {which} + eight counts")
            for s in RANGE_SIGMAS:
                for w in P.COUNT_WIDTHS:
                    only = {k for k in range(nb) if chosen[k] == f"sigma {s:g}"
                            and cw[k] == f"width {w:g}"}
                    if only:
                        for h in P.HS:
                            m.add(f"sigma {s:g}, width {w:g}, h {h:g}",
                                  lambda s=s, w=w, h=h, which=which: sub(rk(s, which))
                                  + h * sub(kern.get(("counts", w))), y, blocks, only)
            models[m.name] = m
        preds, res = {}, {}
        for mname, m in models.items():
            pred, chosen = m.select(len(y), blocks)
            res[mname] = {}
            for scheme in Q.SCHEMES:
                q2 = [float(r2_score(y, pred[scheme][r])) for r in range(Q.REPEATS)]
                res[mname][scheme] = {"q2_mean": float(np.mean(q2)), "chosen": chosen[scheme]}
                preds[(mname, scheme)] = pred[scheme].mean(axis=0)
        comps = []
        for a, b in [(f"central unit, {w} + eight counts", REF[1]) for w in RANGES] + \
                    [(REF[0], f"central unit, {w} + eight counts") for w in RANGES]:
            for scheme in Q.SCHEMES:
                pa, pb = preds[(a, scheme)], preds[(b, scheme)]
                g = (L.RS.paired(y, pa, pb) if scheme == "random"
                     else P.paired_groups(y, pa, pb, groups))
                comps.append({"model": a, "against": b, "scheme": scheme, **g})
                log(f"  {name:4s} {a} - {b} [{scheme}]: {g['mean']:+.3f} "
                    f"[{g['lo']:+.3f}, {g['hi']:+.3f}]")
        out[name] = {"models": res, "comparisons": comps}
        log(f"{name}: distance ranges done ({time.time() - t0:.0f} s)")
    payload = {"gases": out, "r_cut": R_CUT, "sigmas": list(RANGE_SIGMAS),
               "seconds": round(time.time() - t0, 1)}
    L.jdump(payload, os.path.join(RESULTS, "ranges.json"))
    report()
    return payload


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "polymer_permeability.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    lines = ["Plan item PL5: does one number carry what the field adds on gas permeability? Each "
             "candidate is a column added to the eight counts. Five random folds and five folds of "
             "ECFP4 Butina clusters, each repeated %d times, as in PL2." % payload["repeats"], ""]
    for gas in [s[0] for s in GASES if s[0] in payload["gases"]]:
        g = payload["gases"][gas]
        comp = {(c["model"], c["against"], c["scheme"]): c for c in g["comparisons"]}
        fmt = lambda c: f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"   # noqa: E731
        lines += [f"**{gas}** ({g['polymers']} polymers, {g['groups']} clusters)", "",
                  "| Candidate added to the counts | q², bond-count + counts + candidate, random / "
                  "clusters | candidate minus none, random | candidate minus none, clusters | "
                  "field margin over it, random | field margin over it, clusters |",
                  "|---|---|---|---|---|---|"]
        base = [comp[(REF[0], REF[1], s)] for s in Q.SCHEMES]
        lines.append(f"| none (PL2) | {g['models'][REF[1]]['random']['q2_mean']:+.3f} / "
                     f"{g['models'][REF[1]]['group']['q2_mean']:+.3f} | | | {fmt(base[0])} | "
                     f"{fmt(base[1])} |")
        for sname, _ in SETS:
            ref = f"bond-count kernel + eight counts + {sname}"
            r = g["models"][ref]
            add = [comp[(ref, REF[1], s)] for s in Q.SCHEMES]
            mar = [comp[(REF[0], ref, s)] for s in Q.SCHEMES]
            lines.append(f"| {sname} | {r['random']['q2_mean']:+.3f} / {r['group']['q2_mean']:+.3f} "
                         f"| {fmt(add[0])} | {fmt(add[1])} | {fmt(mar[0])} | {fmt(mar[1])} |")
        both = [comp[("central unit + eight counts + all five",
                      "bond-count kernel + eight counts + all five", s)] for s in Q.SCHEMES]
        lines += ["", f"Central unit + counts + all five minus bond-count kernel + counts + all "
                  f"five: {fmt(both[0])} (random), {fmt(both[1])} (clusters).",
                  "Correlation of each candidate with the target, over all polymers: " +
                  ", ".join(f"{c} {v:+.2f}" for c, v in g["correlation_with_target"].items()) + ".",
                  ""]
    path = os.path.join(RESULTS, "ranges.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            ranges = json.load(fh)
        fmt = lambda c: f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"   # noqa: E731
        lines += [f"**Distance ranges.** The central-unit field kernel with only the channels below "
                  f"{ranges['r_cut']:g} A (bonded pairs and pairs two bonds apart) or only those "
                  f"from {ranges['r_cut']:g} A on (widths "
                  f"{', '.join(f'{s:g}' for s in ranges['sigmas'])} A), each plus the eight counts.",
                  "", "| Gas | Range | q², random / clusters | minus bond-count + counts, random | "
                  "minus bond-count + counts, clusters | full field + counts minus it, random | "
                  "full field + counts minus it, clusters |", "|---|---|---|---|---|---|---|"]
        for gas in [s[0] for s in GASES if s[0] in ranges["gases"]]:
            g = ranges["gases"][gas]
            comp = {(c["model"], c["against"], c["scheme"]): c for c in g["comparisons"]}
            for which in RANGES:
                mname = f"central unit, {which} + eight counts"
                r = g["models"][mname]
                over = [comp[(mname, REF[1], s)] for s in Q.SCHEMES]
                under = [comp[(REF[0], mname, s)] for s in Q.SCHEMES]
                lines.append(f"| {gas} | {which} | {r['random']['q2_mean']:+.3f} / "
                             f"{r['group']['q2_mean']:+.3f} | {fmt(over[0])} | {fmt(over[1])} | "
                             f"{fmt(under[0])} | {fmt(under[1])} |")
        lines.append("")
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    ap.add_argument("--ranges", action="store_true",
                    help="the field cut to short and to long distances (after the candidates)")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    if a.ranges:
        run_ranges()
        return 0
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
