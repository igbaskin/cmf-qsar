#!/usr/bin/env python
"""
Plan item PL4: a copolymer as the weighted set of its local environments, against a weighted
mean of descriptors.

The data
--------
``examples/copolymers/vipea.csv`` (made by ``prepare_copolymers.py``): the electron affinity and
the ionization potential, computed with xTB, of 1,575 binary copolymers of 9 aryl diboronic acids
(A) and 25 aryl dibromides (B), each as an alternating copolymer (1:1) and as a random and a block
copolymer at A:B = 1:1, 1:3 and 3:1 (Aldeghi and Coley 2022, MIT licence).  The values were
computed on octamers and averaged over the sequences of each copolymer.

The representation
------------------
A unit inside a chain is described by the field of the middle unit of a trimer
(:func:`cmfqsar.polymer.build_sequence`, :func:`cmfqsar.polymer.autocorrelation_rows`).  A
copolymer has units in several local environments, a unit with a neighbour on each side.  The
environments are counted over the octamers the labels were computed on, with the rules of the
source's ``make-polymers.py``: every distinct order of the units for a random copolymer (a
sequence and its reverse once), AAAABBBB, AABBBBBB or AAAAAABB for a block copolymer and
ABABABAB for the alternating one; inner units only, every chain read both ways
(:func:`cmfqsar.polymer.environment_weights`).  The copolymer's row is the weighted mean of the
rows of its environment trimers.  This is the mean-map principle of item MS5 applied to the
autocorrelation, which is quadratic in the field: the rows of the members are averaged, so no
pair of atoms from two different members is counted.

The models, fixed before the measurement:

* *field kernel, sequence ensemble*: the mean rows above;
* *field kernel, mean of the homopolymer units*: the rows of the middle unit of the AAA and the
  BBB trimer, weighted by the fractions -- it knows the composition, not the neighbours;
* *bond-count kernel, sequence ensemble*: the same ensemble with the distance counted in bonds;
* *weighted counts*: the eight counts of PL1 of each unit, weighted by the fractions -- the
  weighted mean of descriptors that the plan names;
* *weighted counts + architecture*: the same with three columns that name the architecture,
  because a weighted mean of descriptors cannot see it otherwise;
* *weighted ECFP4*: the count fingerprint (radius 2, 2048 bits) of each unit weighted by the
  fractions, in the min/max (Tanimoto) kernel, the input of the source's random-forest baseline;
* the three structure kernels plus the weighted counts.

The gate of PL4 is

    field kernel of the sequence ensemble + weighted counts  minus  weighted counts + architecture

with the 95 % interval above zero with whole dibromides held out.  Two more comparisons: the
sequence ensemble against the mean of the homopolymer units (do the neighbours matter?), and
against the bond-count ensemble (does 3D add to the graph?).  The architecture check takes, for
each pair of monomers and each composition, the differences between the architectures (the
value minus the pair's mean over its architectures) and compares the predicted with the computed
differences; a model that cannot see the architecture predicts none.

The folds: five random folds and five folds of whole dibromides (25 dibromides, five per fold),
each repeated five times (not ten as in PL1 and PL2: the kernels are 1,575 wide).  Random folds
put copolymers of the same two monomers on both sides; the dibromide folds do not.

The ridge floor (added after the first run)
-------------------------------------------
Each training block holds seven copolymers of every pair of monomers (three architectures,
three compositions), whose labels are close.  The evidence treats them as independent and in
some blocks chooses a ridge of 1e-4 to 1e-3.  On a held-out dibromide such a model can predict
far outside the range of the labels: "weighted counts + architecture" missed dibromide 658 by
up to 9 eV on the electron affinity, and its mean q2 with dibromides held out was -0.33.  Every
model is therefore selected a second time with the ridges below :data:`RIDGE_FLOOR` (0.1)
removed, from the stored predictions, and the comparisons are repeated.  Both selections are
reported; the floor was chosen after the first run, from the values 1e-4, 1e-3, 1e-2 and 0.1.

Usage
-----
    python benchmarks/run_copolymers.py --build    # the trimers and their rows (cached)
    python benchmarks/run_copolymers.py            # the models; builds what is missing
    python benchmarks/run_copolymers.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from itertools import permutations
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

import run_logp as L  # noqa: E402
import run_polymer_tg as P  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "copolymers", "vipea.csv")
RESULTS = os.path.join(HERE, "results", "copolymers")
CACHE = os.path.join(HERE, "cache", "copolymers")
TARGETS = (("electron affinity", "ea_ev"), ("ionization potential", "ip_ev"))
SIGMAS = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
REPEATS = 5
SCHEMES = ("random", "group")
MODELS = ("field kernel, sequence ensemble", "field kernel, mean of the homopolymer units",
          "bond-count kernel, sequence ensemble", "weighted counts",
          "weighted counts + architecture", "weighted ECFP4",
          "sequence ensemble + weighted counts", "homopolymer mean + weighted counts",
          "bond-count ensemble + weighted counts")
GATE = ("sequence ensemble + weighted counts", "weighted counts + architecture")
COMPARE = (GATE, ("sequence ensemble + weighted counts", "weighted counts"),
           ("field kernel, sequence ensemble", "field kernel, mean of the homopolymer units"),
           ("sequence ensemble + weighted counts", "homopolymer mean + weighted counts"),
           ("sequence ensemble + weighted counts", "bond-count ensemble + weighted counts"),
           ("sequence ensemble + weighted counts", "weighted ECFP4"))
ARCHITECTURES = ("alternating", "random", "block")
#: the smallest ridge of the second selection (see the module docstring)
RIDGE_FLOOR = 0.1
SELECTIONS = ("evidence", "ridge floor")
log = L.log

# the shared code of PL1 reads these module globals
P.REPEATS = REPEATS
P.SCHEMES = SCHEMES


def read() -> List[dict]:
    with open(DATA, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- the environments

def chains(architecture: str, n_a: int, n_b: int) -> List[str]:
    """The octamers of one copolymer, by the rules of the source's make-polymers.py."""
    if architecture == "alternating":
        return ["AB" * n_a]
    if architecture == "block":
        return ["A" * n_a + "B" * n_b]
    if architecture == "random":
        out, seen = [], set()
        for p in sorted(set(permutations("A" * n_a + "B" * n_b))):
            s = "".join(p)
            if s[::-1] not in seen:
                seen.add(s)
                out.append(s)
        return out
    raise ValueError(architecture)


def environments(rows) -> Dict[str, Dict[tuple, float]]:
    """{copolymer id: {(left, unit, right) as SMILES: weight}}."""
    from cmfqsar.polymer import environment_weights
    out = {}
    for r in rows:
        name = {"A": r["unit_a"], "B": r["unit_b"]}
        w = environment_weights(chains(r["architecture"], int(r["n_a"]), int(r["n_b"])))
        out[r["id"]] = {tuple(name[x] for x in k): v for k, v in w.items()}
    return out


# --------------------------------------------------------------------------- the trimers

def _trimer_one(task):
    """Worker: one trimer (three unit SMILES) to the central-unit rows at every width and the
    bond-count rows."""
    k, units, sigmas = task
    try:
        from cmfqsar.fields import FieldSet, build_fields
        from cmfqsar.invariant import AutocorrelationSpec
        from cmfqsar.polymer import autocorrelation_rows, build_sequence, central_unit
        RDLogger.DisableLog("rdApp.*")
        mol, info = build_sequence(list(units), conformers=P.CONFORMERS, seed=P.BUILD_SEED)
        centre = central_unit(mol)
        fs = FieldSet.create(physchem=P.FIELDS, alpha=1.0, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = build_fields(mol, fs, f"t{k}")
        specs = [AutocorrelationSpec(d_max=P.D_MAX, step=s, sigma=s) for s in sigmas]
        rows = autocorrelation_rows(mf, specs, centre=centre)
        topo = P.topological_rows(mol, np.asarray(mf.weights, float), centre, max(P.TOPO_MAX))
        return (k, [r.astype(np.float32) for r in rows], topo.astype(np.float32),
                Chem.MolToMolBlock(mol), info, "")
    except Exception as exc:                     # noqa: BLE001 - reported per trimer
        return k, None, None, "", {}, f"{type(exc).__name__}: {exc}"


def build_trimers(trimers: List[tuple], jobs: int) -> dict:
    """Rows of every trimer, cached in ``benchmarks/cache/copolymers``."""
    from cmfqsar.invariant import AutocorrelationSpec
    from multiprocessing import get_context
    ids = np.array(["|".join(t) for t in trimers])
    meta = os.path.join(CACHE, "trimers.npz")
    if os.path.isfile(meta):
        z = np.load(meta)
        if z["ids"].shape == ids.shape and np.all(z["ids"] == ids) and all(
                os.path.isfile(os.path.join(CACHE, f"central_{s:g}.npy")) for s in SIGMAS):
            log("trimer rows read from the cache")
            return {k: z[k] for k in z.files}
    os.makedirs(CACHE, exist_ok=True)
    n = len(trimers)
    rows = [np.lib.format.open_memmap(os.path.join(CACHE, f"central_{s:g}.npy.part"), mode="w+",
                                      dtype=np.float32,
                                      shape=(n, 15 * AutocorrelationSpec(
                                          d_max=P.D_MAX, step=s, sigma=s).n_bins))
            for s in SIGMAS]
    topo = np.zeros((n, 15 * (max(P.TOPO_MAX) + 1)), np.float32)
    ok = np.zeros(n, bool)
    span = np.full(n, np.nan)
    errors, blocks = {}, [""] * n
    t0 = time.time()
    tasks = [(k, t, SIGMAS) for k, t in enumerate(trimers)]
    with get_context("spawn").Pool(int(jobs)) as pool:
        for done, (k, rr, tp, block, info, err) in enumerate(
                pool.imap_unordered(_trimer_one, tasks, chunksize=4)):
            if err:
                errors[ids[k]] = err
            else:
                for j, r in enumerate(rr):
                    rows[j][k] = r
                topo[k], ok[k], blocks[k] = tp, True, block
                span[k] = info.get("span", np.nan)
            if (done + 1) % 200 == 0 or done + 1 == n:
                log(f"  trimers {done + 1}/{n} ({time.time() - t0:.0f} s, {len(errors)} failed)")
    for j in range(len(rows)):
        rows[j].flush()
    rows.clear()                 # Windows cannot rename a file that a memory map holds open
    for s in SIGMAS:
        os.replace(os.path.join(CACHE, f"central_{s:g}.npy.part"),
                   os.path.join(CACHE, f"central_{s:g}.npy"))
    with open(os.path.join(CACHE, "trimers.sdf"), "w", encoding="utf-8", newline="\n") as fh:
        for k, b in enumerate(blocks):
            if b:
                fh.write(f"t{k}" + b[b.index("\n"):] + f"> <units>\n{ids[k]}\n\n$$$$\n")
    np.savez(meta, ids=ids, ok=ok, topo=topo, span=span)
    with open(os.path.join(CACHE, "errors.json"), "w", encoding="utf-8") as fh:
        json.dump(errors, fh, indent=1)
    return {"ids": ids, "ok": ok, "topo": topo, "span": span}


def unit_features(units: List[str]):
    """The eight counts of PL1 and the ECFP4 count fingerprint of every unit, capped by hydrogen
    as the monomer of PL1's counts is."""
    from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
    from cmfqsar.polymer import oligomer, parse_repeat_unit
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    C, F = [], []
    for u in units:
        mono = Chem.RemoveHs(oligomer(parse_repeat_unit(u), n=1, add_hydrogens=False))
        C.append([Descriptors.MolWt(mono), Lipinski.NumHDonors(mono),
                  Lipinski.NumHAcceptors(mono), rdMolDescriptors.CalcNumRotatableBonds(mono),
                  rdMolDescriptors.CalcNumRings(mono), rdMolDescriptors.CalcNumAromaticRings(mono),
                  rdMolDescriptors.CalcTPSA(mono), rdMolDescriptors.CalcFractionCSP3(mono)])
        F.append(gen.GetCountFingerprintAsNumPy(mono).astype(float))
    return np.array(C), np.array(F)


def minmax_kernel(X: np.ndarray) -> np.ndarray:
    """The min/max (generalised Tanimoto) kernel of non-negative count vectors."""
    n = len(X)
    K = np.empty((n, n))
    for i in range(n):
        mn = np.minimum(X[i], X).sum(axis=1)
        mx = np.maximum(X[i], X).sum(axis=1)
        K[i] = np.where(mx > 0, mn / np.maximum(mx, 1e-300), 1.0)
    return K


# --------------------------------------------------------------------------- the models

class Kernels:
    def __init__(self, Wens, Wmean, topo, counts, counts_arch, fp):
        self.Wens, self.Wmean = Wens, Wmean
        self.topo, self.counts, self.counts_arch, self.fp = topo, counts, counts_arch, fp
        self.store = {}

    def get(self, key):
        if key not in self.store:
            kind, arg = key
            if kind in ("ensemble", "mean"):
                T = np.load(os.path.join(CACHE, f"central_{arg:g}.npy"), mmap_mode="r")
                W = self.Wens if kind == "ensemble" else self.Wmean
                K = P.field_kernel((W @ np.asarray(T, np.float64)).astype(np.float32), arg)
            elif kind == "topo":
                K = P.topo_kernel(self.topo, arg)
            elif kind == "counts":
                K = L.gaussian_at(self.counts, arg)
            elif kind == "counts_arch":
                K = L.gaussian_at(self.counts_arch, arg)
            elif kind == "ecfp":
                K = minmax_kernel(self.fp)
            else:
                raise KeyError(kind)
            self.store[key] = K
        return self.store[key]


def make_blocks(n: int, groups: np.ndarray):
    out = []
    for rep in range(REPEATS):
        for te in L.RW.folds(n, P.FOLDS, P.SEED + rep):
            out.append(("random", rep, np.setdiff1d(np.arange(n), te), te))
    for rep in range(REPEATS):
        for te in P.class_folds(groups.astype(str), P.FOLDS, P.SEED + rep):
            out.append(("group", rep, np.setdiff1d(np.arange(n), te), te))
    return out


def architecture_check(rows, y, pred) -> dict:
    """Predicted against computed differences between the architectures of one pair and
    composition (the value minus the mean of its architectures)."""
    key = [(r["pair"], r["n_a"], r["n_b"]) for r in rows]
    groups: Dict[tuple, List[int]] = {}
    for i, k in enumerate(key):
        groups.setdefault(k, []).append(i)
    dy, dp = [], []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        idx = np.array(idx)
        dy.extend(y[idx] - y[idx].mean())
        dp.extend(pred[idx] - pred[idx].mean())
    dy, dp = np.array(dy), np.array(dp)
    corr = float(np.corrcoef(dy, dp)[0, 1]) if np.std(dp) > 1e-12 else 0.0
    return {"sd_computed": float(dy.std()), "sd_predicted": float(dp.std()),
            "correlation": corr, "rmse": float(np.sqrt(np.mean((dy - dp) ** 2)))}


def select(model, n: int, blocks, floor: float = 0.0):
    """The predictions of a model with the setting and the ridge chosen by the evidence on each
    training block, among the ridges at least `floor`; and how often each choice was made."""
    ok = np.asarray(P.LAMS) >= floor
    pred = {s: np.full((REPEATS, n), np.nan) for s in SCHEMES}
    chosen = {s: {} for s in SCHEMES}
    for k, (scheme, rep, _, te) in enumerate(blocks):
        if model.fixed is not None:
            pred[scheme][rep, te] = model.fixed[k]
            continue
        ev = np.array([c["ev"][k] for c in model.cands], float)
        ev[:, ~ok] = -np.inf
        c, j = np.unravel_index(int(np.nanargmax(ev)), ev.shape)
        pred[scheme][rep, te] = model.cands[c]["pred"][k][j]
        key = f"{model.labels[c]}, ridge {P.LAMS[j]:g}"
        chosen[scheme][key] = chosen[scheme].get(key, 0) + 1
    return pred, chosen


def run_target(name, col, rows, kern: Kernels, groups, t0) -> dict:
    from cmfqsar.model import r2_score, rmse
    y = np.array([float(r[col]) for r in rows])
    blocks = make_blocks(len(y), groups)
    nb = len(blocks)
    P.Model.cache_dir = os.path.join(CACHE, "settings")
    models = {m: P.Model(m) for m in MODELS}
    for s in SIGMAS:
        models["field kernel, sequence ensemble"].add(f"sigma {s:g}",
                                                      lambda s=s: kern.get(("ensemble", s)),
                                                      y, blocks)
        models["field kernel, mean of the homopolymer units"].add(
            f"sigma {s:g}", lambda s=s: kern.get(("mean", s)), y, blocks)
    for dm in P.TOPO_MAX:
        models["bond-count kernel, sequence ensemble"].add(f"up to {dm} bonds",
                                                           lambda dm=dm: kern.get(("topo", dm)),
                                                           y, blocks)
    for w in P.COUNT_WIDTHS:
        models["weighted counts"].add(f"width {w:g}", lambda w=w: kern.get(("counts", w)),
                                      y, blocks)
        models["weighted counts + architecture"].add(
            f"width {w:g}", lambda w=w: kern.get(("counts_arch", w)), y, blocks)
    models["weighted ECFP4"].add("min/max", lambda: kern.get(("ecfp", 0)), y, blocks)
    cwidth = [models["weighted counts"].best_label(k) for k in range(nb)]
    sums = (("sequence ensemble + weighted counts", "field kernel, sequence ensemble",
             "ensemble", SIGMAS, "sigma {:g}"),
            ("homopolymer mean + weighted counts", "field kernel, mean of the homopolymer units",
             "mean", SIGMAS, "sigma {:g}"),
            ("bond-count ensemble + weighted counts", "bond-count kernel, sequence ensemble",
             "topo", P.TOPO_MAX, "up to {} bonds"))
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
                                      kern.get((kind, a)) + h * kern.get(("counts", w)),
                                      y, blocks, only)
    log(f"{name}: {len(y)} copolymers, models done ({time.time() - t0:.0f} s)")
    res, preds = {}, {}
    for m, model in models.items():
        res[m] = {}
        for sel in SELECTIONS:
            pred, chosen = select(model, len(y), blocks,
                                  RIDGE_FLOOR if sel == "ridge floor" else 0.0)
            r = {}
            for scheme in SCHEMES:
                p = pred[scheme]
                q2 = [float(r2_score(y, p[rep])) for rep in range(REPEATS)]
                mean_pred = p.mean(axis=0)
                r[scheme] = {"q2_mean": float(np.mean(q2)), "q2_range": float(np.ptp(q2)),
                             "rmse_mean": float(np.mean([rmse(y, p[rep])
                                                         for rep in range(REPEATS)])),
                             "architecture": architecture_check(rows, y, mean_pred),
                             "chosen": chosen[scheme]}
                preds[(m, sel, scheme)] = mean_pred
            if sel == "evidence":
                res[m].update(r)
            else:
                res[m]["ridge_floor"] = r
            log(f"  {name[:10]:10s} {sel:11s} {m:46s} random q2 {r['random']['q2_mean']:+.3f} | "
                f"dibromides held out {r['group']['q2_mean']:+.3f} | architecture r "
                f"{r['group']['architecture']['correlation']:+.2f}")
    comps = []
    for sel in SELECTIONS:
        for a, b in COMPARE:
            for scheme in SCHEMES:
                pa, pb = preds[(a, sel, scheme)], preds[(b, sel, scheme)]
                g = (L.RS.paired(y, pa, pb) if scheme == "random"
                     else P.paired_groups(y, pa, pb, groups))
                comps.append({"model": a, "against": b, "scheme": scheme, "selection": sel, **g})
                log(f"  {name[:10]:10s} {sel:11s} {a} - {b} [{scheme}]: {g['mean']:+.3f} "
                    f"[{g['lo']:+.3f}, {g['hi']:+.3f}]")
    return {"copolymers": int(len(y)), "y": {"mean": float(y.mean()), "sd": float(y.std())},
            "models": res, "comparisons": comps}


def run(jobs: int = 12, build_only: bool = False) -> Optional[dict]:
    t0 = time.time()
    L.selfcheck()
    rows = read()
    env = environments(rows)
    units = sorted({r["unit_a"] for r in rows} | {r["unit_b"] for r in rows})
    homo = {u: (u, u, u) for u in units}
    trimers = sorted(set(homo.values()) | {t for e in env.values() for t in e})
    log(f"{len(rows)} copolymers, {len(units)} units, {len(trimers)} trimers")
    tr = build_trimers(trimers, jobs)
    bad = [i for i, ok in zip(tr["ids"], tr["ok"]) if not ok]
    if bad:
        log(f"trimers not built: {bad}")
    if build_only:
        return None
    pos = {t: k for k, t in enumerate(trimers)}
    ok_t = tr["ok"]
    keep = [i for i, r in enumerate(rows)
            if all(ok_t[pos[t]] for t in env[r["id"]]) and ok_t[pos[homo[r["unit_a"]]]]
            and ok_t[pos[homo[r["unit_b"]]]]]
    rows = [rows[i] for i in keep]
    n = len(rows)
    Wens = np.zeros((n, len(trimers)))
    Wmean = np.zeros((n, len(trimers)))
    for i, r in enumerate(rows):
        for t, w in env[r["id"]].items():
            Wens[i, pos[t]] += w
        Wmean[i, pos[homo[r["unit_a"]]]] += float(r["frac_a"])
        Wmean[i, pos[homo[r["unit_b"]]]] += float(r["frac_b"])
    C, F = unit_features(units)
    upos = {u: k for k, u in enumerate(units)}
    fa = np.array([float(r["frac_a"]) for r in rows])[:, None]
    fb = np.array([float(r["frac_b"]) for r in rows])[:, None]
    ia = np.array([upos[r["unit_a"]] for r in rows])
    ib = np.array([upos[r["unit_b"]] for r in rows])
    counts = fa * C[ia] + fb * C[ib]
    arch = np.array([[r["architecture"] == a for a in ARCHITECTURES] for r in rows], float)
    fp = fa * F[ia] + fb * F[ib]
    topo = Wens @ np.asarray(tr["topo"], np.float64)
    kern = Kernels(Wens, Wmean, topo, L.standardise(counts),
                   L.standardise(np.hstack([counts, arch])), fp)
    groups = np.array([int(r["b"]) for r in rows])
    log(f"{n} copolymers with every trimer built ({time.time() - t0:.0f} s)")
    out = {name: run_target(name, col, rows, kern, groups, t0) for name, col in TARGETS}
    payload = {"targets": out, "copolymers": n, "trimers": len(trimers),
               "trimers_not_built": bad, "units": len(units), "repeats": REPEATS,
               "folds": P.FOLDS, "sigmas": list(SIGMAS), "topo_max": list(P.TOPO_MAX),
               "count_widths": list(P.COUNT_WIDTHS), "hs": list(P.HS), "lams": list(P.LAMS),
               "ridge_floor": RIDGE_FLOOR,
               "span": {"median": float(np.nanmedian(tr["span"]))},
               "seconds": round(time.time() - t0, 1)}
    L.jdump(payload, os.path.join(RESULTS, "copolymers.json"))
    report(payload)
    return payload


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "copolymers.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    lines = [f"Plan item PL4: {payload['copolymers']} copolymers of the VIPEA set (computed "
             f"electron affinity and ionization potential, eV), each described by the "
             f"{payload['trimers']} trimers of its local environments. Five random folds and five "
             f"folds of whole dibromides, each repeated {payload['repeats']} times; the setting and "
             f"the ridge are chosen by the evidence on the training block, among all ridges "
             f"('evidence') or among the ridges of at least {payload['ridge_floor']:g} ('ridge "
             f"floor'). 'Architecture r' is the correlation of the predicted with the computed "
             f"differences between the architectures of one pair of monomers at one "
             f"composition, with dibromides held out.", ""]
    for name, _ in TARGETS:
        t = payload["targets"][name]
        for sel in SELECTIONS:
            lines += [f"**{name}, {sel}** ({t['copolymers']} copolymers, spread "
                      f"{t['y']['sd']:.3f} eV)", "",
                      "| Model | q², random folds | RMSE, random folds | q², dibromides held out "
                      "| RMSE, dibromides held out | architecture r |",
                      "|---|---:|---:|---:|---:|---:|"]
            for m in MODELS:
                r = t["models"][m] if sel == "evidence" else t["models"][m]["ridge_floor"]
                lines.append(f"| {m} | {r['random']['q2_mean']:+.3f} | "
                             f"{r['random']['rmse_mean']:.3f} | {r['group']['q2_mean']:+.3f} | "
                             f"{r['group']['rmse_mean']:.3f} | "
                             f"{r['group']['architecture']['correlation']:+.2f} |")
            lines += ["", "| Difference of q² | Folds | Mean [95 % interval] |", "|---|---|---|"]
            for c in t["comparisons"]:
                if c["selection"] != sel:
                    continue
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
    ap.add_argument("--jobs", type=int, default=12, help="processes for building the trimers")
    ap.add_argument("--build", action="store_true", help="build the trimers and rows only")
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    run(a.jobs, a.build)
    return 0


if __name__ == "__main__":
    sys.exit(main())
