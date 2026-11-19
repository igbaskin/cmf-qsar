#!/usr/bin/env python
"""
Plan item PL3: the periodic chain against the trimer.  The field of one unit of an infinite chain
is compared with the field of the middle unit of the trimer that PL2 built, on PL2's targets
(density from molecular dynamics, measured permeability to five gases) and folds.

The chain
---------
The plan asked for a builder that takes the repeat distance from the extended trimer, so that the
periodic chain of item M1 gets a cell.  An extended trimer does not give one: its three units
have different conformations.  Moving the middle unit by the vector between two head atoms
misses the next unit by 1.7 to 11 A (root mean square) on four test polymers, and puts atoms of
neighbouring copies 0.5 A apart.  The best rigid fit of one unit onto the next overlaps atoms in
57 % of a sample of 432 PL2 trimers.

The chain here is built by :func:`cmfqsar.polymer.chain_screw`: copies of the middle unit,
related by the screw (rotation plus translation) that carries the junction before the middle
unit onto the junction after it.  Every junction of that chain has the bond, the bond angles and
the torsions of the trimer's junction, and the middle unit keeps its conformation, so the chain
differs from the trimer only by the copies beyond the nearest neighbours and by the absence of
caps.  Such a helix can still meet itself a few units further on.  A chain with two atoms closer
than 1.3 A (:data:`cmfqsar.polymer.CHAIN_CONTACT`) is reported and left out, and every model is
compared on the polymers whose chain is valid.  A screw that turns by less than 5 degrees per
unit is a translation, for which M1's periodic cell exists (:func:`cmfqsar.polymer.periodic_chain`);
how many there are is reported.

The question, fixed before the measurement
------------------------------------------
Does the field of the infinite chain describe a polymer better than the field of the trimer's
middle unit?  The gate of PL3 is

    periodic chain + eight counts  minus  central unit + eight counts  >  0,

with the 95 % paired bootstrap interval above zero, on a target and fold scheme.  The chain and
the central unit alone are compared too, and so is each with the bond-count kernel plus counts
(the PL2 gate, on the same polymers).  The correlation of the two field kernels over all pairs of
polymers says how much of the description the copies change.

Added after the first run, which found the chain worse than the central unit on every target:

* *The chain cut to its nearest copies* ("nearest copies only": one copy on each side, as in a
  trimer, but copies of the middle unit and no caps).  It separates the two differences between
  the chain and the trimer: the copies beyond the nearest neighbours, and the idealised
  neighbours.
* *A narrower width for the chain*: it chose 0.001 A, the narrowest of PL2's list, in a third of
  the blocks, so 0.0005 A is offered as well.
* *The ridge at the edge of its grid*: every model chose the smallest ridge (1e-8) in about half
  of its blocks, as in PL2; each model is selected again with the ridges below 1e-4 removed,
  from the stored predictions (``q2_ridge_floor``).

Usage
-----
    python benchmarks/run_polymer_chain.py --build    # the screws and the chain rows (cached)
    python benchmarks/run_polymer_chain.py            # the models; builds what is missing
    python benchmarks/run_polymer_chain.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

import run_polymer_packing as Q  # noqa: E402  (sets the globals of run_polymer_tg)

P, L = Q.P, Q.L
RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "polymer_chain")
TAG = "chain"
SCREW_FIELDS = ("turn", "rise", "frame_residual", "rmsd_next", "junction", "contact")
MODELS = ("field kernel, central unit", "field kernel, periodic chain",
          "field kernel, nearest copies only", "eight counts",
          "bond-count kernel, central unit", "central unit + eight counts",
          "periodic chain + eight counts", "bond-count kernel + eight counts")
GATE = ("periodic chain + eight counts", "central unit + eight counts")
COMPARE = (GATE, ("field kernel, periodic chain", "field kernel, central unit"),
           ("field kernel, nearest copies only", "field kernel, central unit"),
           ("field kernel, periodic chain", "field kernel, nearest copies only"),
           ("periodic chain + eight counts", "bond-count kernel + eight counts"),
           ("central unit + eight counts", "bond-count kernel + eight counts"))
#: the widths at which the kernels of the chain and the central unit are correlated
CORRELATE_AT = (0.005, 0.1)
log = L.log


# --------------------------------------------------------------------------- the chain rows

#: the chain chose the narrowest width of PL2's list, 0.001 A, in a third of the blocks of the
#: first run, so it is offered 0.0005 A as well
CHAIN_SIGMAS = (0.0005,) + tuple(P.SIGMAS)
#: the rows kept per kind: the widths, and how many copies on each side (None: all within reach)
KINDS = {"chain": (CHAIN_SIGMAS, None), "near": (tuple(P.SIGMAS), 1)}


def rows_path(kind: str, sigma: float) -> str:
    return os.path.join(Q.CACHE, f"{kind}_{TAG}_{sigma:g}.npy")


def _chain_one(task):
    """Worker: one stored trimer to its screw record and, for a chain that does not overlap, the
    rows asked for: {kind: [rows at each width]}."""
    k, block, psmiles, wanted = task
    try:
        from cmfqsar.fields import FieldSet, build_fields
        from cmfqsar.invariant import AutocorrelationSpec
        from cmfqsar.polymer import (UNIT_PROP, chain_autocorrelation_rows, chain_screw,
                                     oligomer, unit_labels)
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromMolBlock(block, removeHs=False)
        labels = unit_labels(oligomer(psmiles, n=P.UNITS))
        for a, v in zip(mol.GetAtoms(), labels):
            a.SetIntProp(UNIT_PROP, int(v))
        screw = chain_screw(mol)
        rec = screw.record()
        if screw.overlaps or not wanted:
            return k, rec, None, ""
        fs = FieldSet.create(physchem=P.FIELDS, alpha=1.0, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = build_fields(mol, fs, f"p{k}")
        out = {}
        for kind, sigmas, copies in wanted:
            specs = [AutocorrelationSpec(d_max=P.D_MAX, step=s, sigma=s) for s in sigmas]
            out[kind] = [r.astype(np.float32) for r in
                         chain_autocorrelation_rows(mol, mf, specs, screw, copies=copies)]
        return k, rec, out, ""
    except Exception as exc:                     # noqa: BLE001 - reported per polymer
        return k, {}, None, f"{type(exc).__name__}: {exc}"


def build_chains(rows, jobs: int) -> dict:
    """The screw of every trimer PL2 built, and the rows of every kind of :data:`KINDS` for the
    chains that do not overlap.  Cached; a width or a kind added later is computed alone."""
    from multiprocessing import get_context

    from cmfqsar.invariant import AutocorrelationSpec
    meta = os.path.join(Q.CACHE, f"build_{TAG}.npz")
    ids = np.array([r["psmiles"] for r in rows])
    n = len(rows)
    z = np.load(os.path.join(Q.CACHE, f"build_{Q.TAG}.npz"))
    m = None
    if os.path.isfile(meta):
        m = np.load(meta)
        if not (m["ids"].shape == ids.shape and bool(np.all(m["ids"] == ids))):
            m = None
    missing = [(kind, s) for kind, (sig, _) in KINDS.items() for s in sig
               if m is None or not os.path.isfile(rows_path(kind, s))]
    if m is not None and not missing:
        log("chain rows read from the cache")
        return {k: m[k] for k in m.files}
    wanted = [(kind, tuple(s for kk, s in missing if kk == kind), KINDS[kind][1])
              for kind in KINDS if any(kk == kind for kk, _ in missing)]
    sup = Chem.SDMolSupplier(os.path.join(Q.CACHE, f"trimers_{Q.TAG}.sdf"), removeHs=False,
                             sanitize=False)
    blocks = {mm.GetProp("_Name"): Chem.MolToMolBlock(mm) for mm in sup}
    todo = [k for k in range(n) if z["ok"][k] and (m is None or m["ok"][k])]
    tasks = [(k, blocks[rows[k]["id"]], rows[k]["psmiles"], wanted) for k in todo]
    out = {}
    for kind, sigmas, _ in wanted:
        for s in sigmas:
            w = 15 * AutocorrelationSpec(d_max=P.D_MAX, step=s, sigma=s).n_bins
            out[(kind, s)] = np.lib.format.open_memmap(rows_path(kind, s) + ".part", mode="w+",
                                                       dtype=np.float32, shape=(n, w))
    screw = m["screw"].copy() if m is not None else np.full((n, len(SCREW_FIELDS)), np.nan)
    ok = m["ok"].copy() if m is not None else np.zeros(n, bool)
    errors = {}
    t0 = time.time()
    log(f"chain rows to compute: {missing} for {len(tasks)} trimers")
    with get_context("spawn").Pool(int(jobs)) as pool:
        for done, (k, rec, rr, err) in enumerate(pool.imap_unordered(_chain_one, tasks,
                                                                      chunksize=4)):
            if err:
                errors[rows[k]["id"]] = err
            if rec:
                screw[k] = [rec[f] for f in SCREW_FIELDS]
            if rr is not None:
                for kind, sigmas, _ in wanted:
                    for s, r in zip(sigmas, rr[kind]):
                        out[(kind, s)][k] = r
                ok[k] = True
            if (done + 1) % 250 == 0 or done + 1 == len(tasks):
                log(f"  chains {done + 1}/{len(tasks)} ({time.time() - t0:.0f} s, "
                    f"{len(errors)} failed)")
    keys = list(out)
    for key in keys:
        out[key].flush()
    out.clear()
    for kind, s in keys:
        os.replace(rows_path(kind, s) + ".part", rows_path(kind, s))
    np.savez(meta, ids=ids, ok=ok, screw=screw, built=z["ok"])
    if m is None:
        with open(os.path.join(Q.CACHE, f"errors_{TAG}.json"), "w", encoding="utf-8") as fh:
            json.dump(errors, fh, indent=1)
    return {"ids": ids, "ok": ok, "screw": screw, "built": z["ok"]}


def screw_summary(ch: dict, idx: np.ndarray) -> dict:
    S = ch["screw"][idx]
    good = ~np.isnan(S[:, 0])
    S = S[good]
    out = {"trimers": int(len(idx)), "screws": int(good.sum()),
           "overlapping": int(np.sum(S[:, 5] < 1.3)),
           "translations": int(np.sum(S[:, 0] < 5.0)),
           "translations_valid": int(np.sum((S[:, 0] < 5.0) & (S[:, 5] >= 1.3)))}
    for j, f in enumerate(SCREW_FIELDS):
        v = S[:, j]
        out[f] = {"median": float(np.median(v)), "p05": float(np.percentile(v, 5)),
                  "p95": float(np.percentile(v, 95))}
    return out


# --------------------------------------------------------------------------- the models

class Kernels:
    def __init__(self, keep: np.ndarray, topo: np.ndarray, counts: np.ndarray):
        self.keep, self.topo, self.counts = keep, topo, counts
        self.store = {}

    def get(self, key):
        if key not in self.store:
            kind, arg = key
            if kind == "central":
                F = np.load(os.path.join(Q.CACHE, f"central_{Q.TAG}_{arg:g}.npy"),
                            mmap_mode="r")
                K = P.field_kernel(np.asarray(F[self.keep]), arg)
            elif kind in KINDS:
                F = np.load(rows_path(kind, arg), mmap_mode="r")
                K = P.field_kernel(np.asarray(F[self.keep]), arg)
            elif kind == "topo":
                K = P.topo_kernel(self.topo, arg)
            elif kind == "counts":
                K = L.gaussian_at(self.counts, arg)
            else:
                raise KeyError(kind)
            self.store[key] = K
        return self.store[key]


#: the smallest ridge kept by :func:`ridge_floor`
RIDGE_FLOOR = 1e-4


def ridge_floor(model, y, blocks, scheme: str) -> float:
    """q2 of a model re-selected with every ridge below :data:`RIDGE_FLOOR` removed, from the
    stored predictions of every ridge: whether the choice of the grid's smallest ridge matters."""
    from cmfqsar.model import r2_score
    first = int(np.searchsorted(np.asarray(P.LAMS), RIDGE_FLOOR))
    pred = np.full((Q.REPEATS, len(y)), np.nan)
    for k, (sch, rep, _, te) in enumerate(blocks):
        if sch != scheme:
            continue
        ev = np.array([c["ev"][k] for c in model.cands], float)
        ev[:, :first] = -np.inf
        c, j = np.unravel_index(int(np.nanargmax(ev)), ev.shape)
        pred[rep, te] = model.cands[c]["pred"][k][j]
    return float(np.mean([r2_score(y, pred[r]) for r in range(Q.REPEATS)]))


def kernel_correlation(A: np.ndarray, B: np.ndarray) -> float:
    iu = np.triu_indices(len(A), 1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def run_target(spec, kern: Kernels, pos: Dict[str, int], t0: float) -> dict:
    from cmfqsar.model import r2_score, rmse
    name, table, col, group_col, unit = spec
    rows = [r for r in Q.read(table) if r[col] != ""]
    keys = [Chem.MolToSmiles(Chem.MolFromSmiles(r["psmiles"])) for r in rows]
    have = [k for k, key in enumerate(keys) if key in pos]
    rows = [rows[k] for k in have]
    idx = np.array([pos[keys[k]] for k in have])
    y = np.array([float(r[col]) for r in rows])
    groups = (np.array([int(r[group_col]) for r in rows]) if group_col
              else Q.butina_groups([r["psmiles"] for r in rows]))
    blocks = Q.make_blocks(len(y), groups)
    nb = len(blocks)
    sub = lambda key: kern.get(key)[np.ix_(idx, idx)]          # noqa: E731
    P.Model.cache_dir = os.path.join(Q.CACHE, "settings_chain")
    models = {m: P.Model(m) for m in MODELS}
    for s in P.SIGMAS:
        models["field kernel, central unit"].add(f"sigma {s:g}", lambda s=s: sub(("central", s)),
                                                 y, blocks)
        models["field kernel, nearest copies only"].add(f"sigma {s:g}",
                                                        lambda s=s: sub(("near", s)), y, blocks)
    for s in CHAIN_SIGMAS:
        models["field kernel, periodic chain"].add(f"sigma {s:g}", lambda s=s: sub(("chain", s)),
                                                   y, blocks)
    for dm in P.TOPO_MAX:
        models["bond-count kernel, central unit"].add(f"up to {dm} bonds",
                                                      lambda dm=dm: sub(("topo", dm)), y, blocks)
    for w in P.COUNT_WIDTHS:
        models["eight counts"].add(f"width {w:g}", lambda w=w: sub(("counts", w)), y, blocks)
    cwidth = [models["eight counts"].best_label(k) for k in range(nb)]
    sums = (("central unit + eight counts", "field kernel, central unit", "central", P.SIGMAS,
             "sigma {:g}"),
            ("periodic chain + eight counts", "field kernel, periodic chain", "chain",
             CHAIN_SIGMAS, "sigma {:g}"),
            ("bond-count kernel + eight counts", "bond-count kernel, central unit", "topo",
             P.TOPO_MAX, "up to {} bonds"))
    for combo, single, kind, grid, fmt in sums:
        chosen = [models[single].best_label(k) for k in range(nb)]
        for a in grid:
            for w in P.COUNT_WIDTHS:
                only = {k for k in range(nb) if chosen[k] == fmt.format(a)
                        and cwidth[k] == f"width {w:g}"}
                if not only:
                    continue
                for h in P.HS:
                    models[combo].add(f"{fmt.format(a)}, width {w:g}, h {h:g}",
                                      lambda a=a, w=w, h=h, kind=kind:
                                      sub((kind, a)) + h * sub(("counts", w)), y, blocks, only)
    log(f"{name}: {len(y)} polymers, {len(set(groups.tolist()))} groups, models done "
        f"({time.time() - t0:.0f} s)")
    res, preds = {}, {}
    for m, model in models.items():
        pred, chosen = model.select(len(y), blocks)
        r = {}
        for scheme in Q.SCHEMES:
            p = pred[scheme]
            q2 = [float(r2_score(y, p[rep])) for rep in range(Q.REPEATS)]
            r[scheme] = {"q2_mean": float(np.mean(q2)), "q2_range": float(np.ptp(q2)),
                         "rmse_mean": float(np.mean([rmse(y, p[rep])
                                                     for rep in range(Q.REPEATS)])),
                         "q2_ridge_floor": ridge_floor(model, y, blocks, scheme),
                         "chosen": chosen[scheme]}
            preds[(m, scheme)] = p.mean(axis=0)
        res[m] = r
        log(f"  {name:8s} {m:34s} random q2 {r['random']['q2_mean']:+.3f} | group q2 "
            f"{r['group']['q2_mean']:+.3f}")
    comps = []
    for a, b in COMPARE:
        for scheme in Q.SCHEMES:
            pa, pb = preds[(a, scheme)], preds[(b, scheme)]
            g = (L.RS.paired(y, pa, pb) if scheme == "random"
                 else P.paired_groups(y, pa, pb, groups))
            comps.append({"model": a, "against": b, "scheme": scheme, **g})
            log(f"  {name:8s} {a} - {b} [{scheme}]: {g['mean']:+.3f} [{g['lo']:+.3f}, "
                f"{g['hi']:+.3f}]")
    return {"polymers": int(len(y)), "groups": int(len(set(groups.tolist()))), "unit": unit,
            "y": {"mean": float(y.mean()), "sd": float(y.std())}, "models": res,
            "comparisons": comps}


def run(jobs: int = 16, build_only: bool = False) -> Optional[dict]:
    t0 = time.time()
    L.selfcheck()
    rows, keys = Q.union_rows()
    ch = build_chains(rows, jobs)
    built = np.where(ch["built"])[0]
    summary = screw_summary(ch, built)
    log(f"{summary['screws']} screws of {summary['trimers']} trimers: {summary['overlapping']} "
        f"chains overlap, {summary['translations']} screws are translations "
        f"({summary['translations_valid']} of them valid) ({time.time() - t0:.0f} s)")
    if build_only:
        return None
    keep = np.where(ch["ok"])[0]
    pos = {keys[i]: j for j, i in enumerate(keep)}
    z = np.load(os.path.join(Q.CACHE, f"build_{Q.TAG}.npz"))
    kern = Kernels(keep, np.asarray(z["topo"][keep], float), L.standardise(z["counts"][keep]))
    rise = ch["screw"][keep, 1]
    tert = np.percentile(rise, [100 / 3, 200 / 3])
    corr = {}
    for s in CORRELATE_AT:
        A, B = kern.get(("central", s)), kern.get(("chain", s))
        part = {"all": kernel_correlation(A, B)}
        for lab, m in (("shortest third of rises", rise <= tert[0]),
                       ("longest third of rises", rise > tert[1])):
            ii = np.where(m)[0]
            part[lab] = kernel_correlation(A[np.ix_(ii, ii)], B[np.ix_(ii, ii)])
        corr[f"{s:g}"] = part
        log(f"kernel correlation, central unit against chain, at {s:g} A: {part}")
    out = {spec[0]: run_target(spec, kern, pos, t0) for spec in Q.TARGETS}
    payload = {"targets": out, "screws": summary, "kernel_correlation": corr,
               "rise_tertiles": [float(t) for t in tert], "valid": int(len(keep)),
               "not_built": {k: v for k, v in json.load(open(os.path.join(
                   Q.CACHE, f"errors_{TAG}.json"), encoding="utf-8")).items()},
               "repeats": Q.REPEATS, "folds": P.FOLDS, "sigmas": list(P.SIGMAS),
               "chain_sigmas": list(CHAIN_SIGMAS), "ridge_floor": RIDGE_FLOOR,
               "topo_max": list(P.TOPO_MAX), "count_widths": list(P.COUNT_WIDTHS),
               "hs": list(P.HS), "lams": list(P.LAMS), "seconds": round(time.time() - t0, 1)}
    L.jdump(payload, os.path.join(RESULTS, "polymer_chain.json"))
    report(payload)
    return payload


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "polymer_chain.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    s = payload["screws"]
    lines = [f"Plan item PL3: the periodic chain against the trimer, on the targets and folds of "
             f"PL2. The chain is made of copies of the trimer's middle unit, related by the screw "
             f"that carries the junction before the middle unit onto the junction after it.", "",
             f"Screws of {s['screws']} trimers: {s['overlapping']} chains have two atoms closer "
             f"than 1.3 A and are left out; {s['translations']} screws turn by less than 5 degrees "
             f"per unit (a translation; {s['translations_valid']} of them valid). Turn per unit: "
             f"median {s['turn']['median']:.0f} degrees; rise per unit: median "
             f"{s['rise']['median']:.2f} A (5 to 95 %: {s['rise']['p05']:.2f} to "
             f"{s['rise']['p95']:.2f} A); the next unit of the trimer is "
             f"{s['rmsd_next']['median']:.2f} A (median root mean square) from the image of the "
             f"middle unit. {payload['valid']} polymers have a valid chain.", "",
             "| Correlation of the two field kernels over pairs of polymers | all | shortest third "
             "of rises | longest third of rises |", "|---|---:|---:|---:|"]
    for w, c in payload["kernel_correlation"].items():
        lines.append(f"| width {w} A | {c['all']:.3f} | {c['shortest third of rises']:.3f} | "
                     f"{c['longest third of rises']:.3f} |")
    lines.append("")
    for name in [t[0] for t in Q.TARGETS if t[0] in payload["targets"]]:
        t = payload["targets"][name]
        lines += [f"**{name}** ({t['polymers']} polymers, {t['groups']} groups, {t['unit']}, "
                  f"spread {t['y']['sd']:.3f})", "",
                  "| Model | q², random folds | RMSE, random folds | q², groups held out |",
                  "|---|---:|---:|---:|"]
        for m in MODELS:
            r = t["models"][m]
            lines.append(f"| {m} | {r['random']['q2_mean']:+.3f} | {r['random']['rmse_mean']:.3f} "
                         f"| {r['group']['q2_mean']:+.3f} |")
        lines += ["", "| Difference of q² | Folds | Mean [95 % interval] |", "|---|---|---|"]
        for c in t["comparisons"]:
            lines.append(f"| {c['model']} minus {c['against']} | {c['scheme']} | "
                         f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] |")
        lines.append("")
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16, help="processes for the chain rows")
    ap.add_argument("--build", action="store_true", help="the screws and chain rows only")
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    run(a.jobs, a.build)
    return 0


if __name__ == "__main__":
    sys.exit(main())
