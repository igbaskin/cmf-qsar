#!/usr/bin/env python
"""
Plan item PL6: ladder polymers -- the polymers of intrinsic microporosity and their relatives --
built as trimers joined by two bonds per junction, and what the permeability models of PL2 say
about them.

The data
--------
``examples/polymer_packing/permeability_pl6.csv``, made by ``prepare_polymer_packing.py``: the
measured permeability (log10 Barrer) of 769 polymers to CH4, CO2, H2, N2 and O2.  It is the table of
PL2 with two changes.  The 27 ladder polymers are kept (PL2 had no builder for four attachment
points), and two faults of the source are repaired: 99 rows had a benzene ring written as a
cyclohexadiene and 12 rows a ``=CH2`` where a carbonyl oxygen belongs.  The 742 linear polymers are
those of PL2; 101 of them have a repaired structure.

The questions, fixed before the measurement
-------------------------------------------
1. *Does the builder make the ladders?*  Every ladder trimer is checked with
   :func:`cmfqsar.orca.geometry_faults` (bond lengths and contacts), and the end-to-end distance
   is reported.
2. *What does the repair change?*  The models of PL2 on the 742 linear polymers with the repaired
   structures, against the numbers of PL2.
3. *Do the models predict a ladder polymer from linear polymers alone?*  The ladders are the
   most permeable class and the most rigid one.  Each model is trained on the 742 linear polymers
   (the setting and the ridge chosen by the evidence on them) and predicts the 27 ladders.  The
   gate of PL6 is

       RMSE on the ladders of central unit + eight counts  <  that of bond-count kernel + counts,

   with the 95 % bootstrap interval of the difference (ladders resampled) below zero.  A 3D
   field sees the rigid, kinked shape of a ladder; the bond graph sees a long run of fused rings.
4. *Does a ladder in the training set help?*  The error on the ladders inside the
   cross-validation of all 769 polymers, where other ladders are in the training block, against
   the error with all ladders held out.

The models are those of PL2 without the whole trimer: the field kernel of the middle unit, the
bond-count kernel of the middle unit, the eight counts, ECFP4 of the repeat unit, the sums of the
two structure kernels with the counts, and the reference of PL5 -- the counts with the side-chain
fraction and the size of the largest side group as two more columns, alone and summed with each
structure kernel.  Folds as in PL2: five random folds and five folds of ECFP4 Butina clusters,
ten repeats each.

The trimers of the units PL2 built are read from its cache (``benchmarks/cache/polymer_packing``,
tag ``packing``); the ladders and the repaired units are built here (tag ``pl6``).

Usage
-----
    python benchmarks/run_polymer_ladders.py --build    # the new trimers and their features
    python benchmarks/run_polymer_ladders.py            # the models; builds what is missing
    python benchmarks/run_polymer_ladders.py --report
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
import run_polymer_permeability as R  # noqa: E402

P, L = Q.P, Q.L
RDLogger.DisableLog("rdApp.*")

TABLE = "permeability_pl6.csv"
RESULTS = os.path.join(HERE, "results", "polymer_ladders")
TAG = "pl6"
GASES = ("CH4", "CO2", "H2", "N2", "O2")
MODELS = ("field kernel, central unit", "bond-count kernel, central unit", "eight counts",
          "ECFP4 of the repeat unit", "central unit + eight counts",
          "bond-count kernel + eight counts", "eight counts + side groups",
          "central unit + eight counts + side groups",
          "bond-count kernel + eight counts + side groups")
GATE = ("central unit + eight counts", "bond-count kernel + eight counts")
COMPARE = (GATE, ("central unit + eight counts", "ECFP4 of the repeat unit"),
           ("central unit + eight counts + side groups",
            "bond-count kernel + eight counts + side groups"),
           ("central unit + eight counts + side groups", "ECFP4 of the repeat unit"))
DRAWS = 2000
log = L.log


def canon(s: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))


# --------------------------------------------------------------------------- the structures

class Features:
    """Rows, counts and side-group numbers of every PL6 unit, from the PL2 cache or built here."""

    def __init__(self, jobs: int):
        self.rows = Q.read(TABLE)
        old_rows, old_keys = Q.union_rows()
        z = np.load(os.path.join(Q.CACHE, f"build_{Q.TAG}.npz"))
        old_pos = {k: i for i, k in enumerate(old_keys) if z["ok"][i]}
        # a unit PL2 tried and could not embed stays out, so that the linear set is PL2's: the
        # embedding has a time limit, and a second try under another load can succeed
        self.pl2_failed = {k for i, k in enumerate(old_keys) if not z["ok"][i]}
        keys = [canon(r["psmiles"]) for r in self.rows]
        new = sorted({k for k in keys if k not in old_pos and k not in self.pl2_failed})
        first = {}
        for r, k in zip(self.rows, keys):
            first.setdefault(k, r)
        new_rows = [{"id": first[k]["id"], "psmiles": first[k]["psmiles"]} for k in new]
        log(f"{len(self.rows)} polymers; {sum(k in old_pos for k in keys)} trimers from the PL2 "
            f"cache, {sum(k in self.pl2_failed for k in keys)} that PL2 could not embed, "
            f"{len(new_rows)} to build")
        t0 = time.time()
        feat = P.build_features(new_rows, jobs, use_cache=True, tag=TAG)
        P.extend_widths(new_rows, jobs, TAG)
        self.errors = P._errors(TAG)
        if self.errors:
            log(f"not built: {self.errors}")
        log(f"built ({time.time() - t0:.0f} s)")
        new_pos = {k: i for i, k in enumerate(new) if feat["ok"][i]}
        # (tag, index in that tag's arrays) of every polymer that has a trimer
        self.src = []
        for k in keys:
            self.src.append((Q.TAG, old_pos[k]) if k in old_pos else
                            (TAG, new_pos[k]) if k in new_pos else None)
        self.ok = np.array([s is not None for s in self.src])
        self.old, self.new = z, np.load(os.path.join(Q.CACHE, f"build_{TAG}.npz"))
        self.new_rows = new_rows

    def _take(self, name):
        out = []
        for s in self.src:
            if s is not None:
                out.append((self.old if s[0] == Q.TAG else self.new)[name][s[1]])
        return np.array(out)

    def topo(self):
        return np.asarray(self._take("topo"), float)

    def counts(self):
        return self._take("counts")

    def central(self, sigma: float) -> np.ndarray:
        arrays = {t: np.load(os.path.join(Q.CACHE, f"central_{t}_{sigma:g}.npy"), mmap_mode="r")
                  for t in (Q.TAG, TAG)}
        return np.stack([np.asarray(arrays[s[0]][s[1]]) for s in self.src if s is not None])


def side_groups(rows) -> np.ndarray:
    """(side-chain fraction, largest side group) of each repeat unit, by the rule of PL5."""
    return np.array([R.backbone_split(r["psmiles"]) for r in rows], float)


def ring_planarity(mol: Chem.Mol) -> float:
    """The largest distance of an atom of an aromatic ring of carbon and nitrogen from that
    ring's best plane (A).

    Rings with oxygen are left out: read back from an SD file, RDKit marks the 1,4-dioxin ring at
    the junction of a PIM as aromatic, and that ring folds along its O-O axis at little cost (0.2
    to 0.3 A out of plane in four of the ladder trimers, with every benzene ring within 0.08 A).
    """
    X = mol.GetConformer().GetPositions()
    worst = 0.0
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic()
               and mol.GetAtomWithIdx(i).GetAtomicNum() in (6, 7) for i in ring):
            Y = X[list(ring)] - X[list(ring)].mean(axis=0)
            worst = max(worst, float(np.abs(Y @ np.linalg.svd(Y)[2][-1]).max()))
    return worst


def audit(feat: Features) -> dict:
    """Geometry checks of the trimers built here, ladders and repaired linear units apart."""
    from cmfqsar.orca import geometry_faults
    sup = Chem.SDMolSupplier(os.path.join(Q.CACHE, f"trimers_{TAG}.sdf"), removeHs=False)
    mols = {m.GetProp("_Name"): m for m in sup if m is not None}
    lad = {r["id"] for r in feat.rows if r["ladder"] == "1"}
    out = {"ladders": [], "repaired": [], "others": []}
    rep = {r["id"] for r in feat.rows if r["repair"]}
    for k, r in enumerate(feat.new_rows):
        m = mols.get(r["id"])
        if m is None:
            continue
        rec = {"id": r["id"], "atoms": m.GetNumAtoms(), "span": float(feat.new["span"][k]),
               "faults": geometry_faults(m), "ring_planarity": round(ring_planarity(m), 3)}
        out["ladders" if r["id"] in lad else "repaired" if r["id"] in rep else "others"].append(rec)
    for part in out:
        bad = [x["id"] for x in out[part] if x["faults"]]
        log(f"audit, {part}: {len(out[part])} trimers, {len(bad)} with faults {bad}; aromatic "
            f"rings within {max([x['ring_planarity'] for x in out[part]] or [0]):.3f} A of "
            f"their planes")
    return out


# --------------------------------------------------------------------------- the models

class Kernels:
    """The kernels on all PL6 units with a trimer, each made once and kept."""

    def __init__(self, feat: Features, rows, Z0: np.ndarray, Zs: np.ndarray):
        self.feat, self.rows, self.Z0, self.Zs = feat, rows, Z0, Zs
        self.store = {}

    def get(self, key):
        if key not in self.store:
            kind, arg = key
            if kind == "central":
                K = P.field_kernel(self.feat.central(arg), arg)
            elif kind == "topo":
                K = P.topo_kernel(self.feat.topo(), arg)
            elif kind == "counts":
                K = L.gaussian_at(self.Z0, arg)
            elif kind == "side":
                K = L.gaussian_at(self.Zs, arg)
            elif kind == "ecfp":
                from cmfqsar.descriptors import fingerprint_kernel
                K = fingerprint_kernel([Chem.MolFromSmiles(r["psmiles"]) for r in self.rows])
            else:
                raise KeyError(kind)
            self.store[key] = K
        return self.store[key]


def fit_models(y, blocks, sub) -> Dict[str, "P.Model"]:
    """Every model of :data:`MODELS` on the blocks, the settings chosen by the evidence.

    The settings go to a cache of their own.  A cache key is made of the model, the setting, the
    targets and the folds, not of the kernel, and the linear polymers of PL6 have the targets and
    the random folds of PL2 but repaired structures: in PL2's cache they would find PL2's
    predictions.  For the same reason this cache has to be deleted when a trimer is rebuilt.
    """
    P.Model.cache_dir = os.path.join(Q.CACHE, "settings_pl6")
    nb = len(blocks)
    models = {m: P.Model(m) for m in MODELS}
    for s in P.SIGMAS:
        models["field kernel, central unit"].add(f"sigma {s:g}", lambda s=s: sub(("central", s)),
                                                 y, blocks)
    for dm in P.TOPO_MAX:
        models["bond-count kernel, central unit"].add(f"up to {dm} bonds",
                                                      lambda dm=dm: sub(("topo", dm)), y, blocks)
    for w in P.COUNT_WIDTHS:
        models["eight counts"].add(f"width {w:g}", lambda w=w: sub(("counts", w)), y, blocks)
        models["eight counts + side groups"].add(f"width {w:g}", lambda w=w: sub(("side", w)),
                                                 y, blocks)
    models["ECFP4 of the repeat unit"].add("Tanimoto", lambda: sub(("ecfp", 0)), y, blocks)
    best = {m: [models[m].best_label(k) for k in range(nb)]
            for m in ("field kernel, central unit", "bond-count kernel, central unit",
                      "eight counts", "eight counts + side groups")}
    sums = (("central unit + eight counts", "field kernel, central unit", "central", P.SIGMAS,
             "sigma {:g}", "eight counts", "counts"),
            ("bond-count kernel + eight counts", "bond-count kernel, central unit", "topo",
             P.TOPO_MAX, "up to {} bonds", "eight counts", "counts"),
            ("central unit + eight counts + side groups", "field kernel, central unit", "central",
             P.SIGMAS, "sigma {:g}", "eight counts + side groups", "side"),
            ("bond-count kernel + eight counts + side groups", "bond-count kernel, central unit",
             "topo", P.TOPO_MAX, "up to {} bonds", "eight counts + side groups", "side"))
    for combo, single, kind, grid, fmt, cmodel, ckind in sums:
        for a in grid:
            for w in P.COUNT_WIDTHS:
                only = {k for k in range(nb) if best[single][k] == fmt.format(a)
                        and best[cmodel][k] == f"width {w:g}"}
                if not only:
                    continue
                for h in P.HS:
                    models[combo].add(f"{fmt.format(a)}, width {w:g}, h {h:g}",
                                      lambda a=a, w=w, h=h, kind=kind, ckind=ckind:
                                      sub((kind, a)) + h * sub((ckind, w)), y, blocks, only)
    return models


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def ladder_stats(y, p) -> dict:
    from scipy.stats import spearmanr
    from cmfqsar.model import r2_score
    return {"rmse": rmse(y, p), "mean_error": float(np.mean(p - y)),
            "spearman": float(spearmanr(y, p)[0]), "q2": float(r2_score(y, p))}


def paired_rmse(y, a, b, seed: int = P.SEED) -> dict:
    """RMSE of a minus RMSE of b on the same polymers, with a bootstrap interval over them."""
    rng = np.random.RandomState(seed)
    n = len(y)
    d = []
    for _ in range(DRAWS):
        i = rng.randint(0, n, n)
        d.append(rmse(y[i], a[i]) - rmse(y[i], b[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": rmse(y, a) - rmse(y, b), "lo": float(lo), "hi": float(hi)}


def run_gas(gas, kern: Kernels, ladder: np.ndarray, t0: float) -> dict:
    from cmfqsar.model import r2_score
    col = f"log10_{gas}"
    have = np.array([r[col] != "" for r in kern.rows])
    out = {}
    for part in ("linear", "all"):
        idx = np.where(have & (~ladder if part == "linear" else True))[0]
        rows = [kern.rows[i] for i in idx]
        y = np.array([float(r[col]) for r in rows])
        lad = ladder[idx]
        groups = Q.butina_groups([r["psmiles"] for r in rows])
        blocks = Q.make_blocks(len(y), groups)
        schemes = ["random", "group"]
        if part == "all":
            blocks.append(("ladders", 0, np.where(~lad)[0], np.where(lad)[0]))
            schemes.append("ladders")
        P.SCHEMES = tuple(schemes)
        sub = lambda key, idx=idx: kern.get(key)[np.ix_(idx, idx)]      # noqa: E731
        models = fit_models(y, blocks, sub)
        log(f"{gas} {part}: {len(y)} polymers ({int(lad.sum())} ladders), {len(set(groups))} "
            f"clusters, models done ({time.time() - t0:.0f} s)")
        res, preds = {}, {}
        for m, model in models.items():
            pred, chosen = model.select(len(y), blocks)
            r = {}
            for scheme in ("random", "group"):
                p = pred[scheme]
                q2 = [float(r2_score(y, p[rep])) for rep in range(Q.REPEATS)]
                r[scheme] = {"q2_mean": float(np.mean(q2)), "q2_range": float(np.ptp(q2)),
                             "rmse_mean": float(np.mean([rmse(y, p[rep])
                                                         for rep in range(Q.REPEATS)])),
                             "chosen": chosen[scheme]}
                preds[(m, scheme)] = p.mean(axis=0)
                if part == "all" and lad.any():
                    r[scheme]["ladders"] = ladder_stats(y[lad], p.mean(axis=0)[lad])
            if part == "all" and lad.any():
                pl = pred["ladders"][0][lad]
                r["ladders"] = {**ladder_stats(y[lad], pl), "chosen": chosen["ladders"]}
                preds[(m, "ladders")] = pl
            res[m] = r
            line = (f"  {gas:4s} {part:6s} {m:46s} random {r['random']['q2_mean']:+.3f} | "
                    f"group {r['group']['q2_mean']:+.3f}")
            if "ladders" in r:
                line += (f" | ladders held out RMSE {r['ladders']['rmse']:.3f} "
                         f"(bias {r['ladders']['mean_error']:+.3f}), in CV "
                         f"{r['random']['ladders']['rmse']:.3f}")
            log(line)
        comps = []
        for a, b in COMPARE:
            for scheme in ("random", "group"):
                pa, pb = preds[(a, scheme)], preds[(b, scheme)]
                g = (L.RS.paired(y, pa, pb) if scheme == "random"
                     else P.paired_groups(y, pa, pb, groups))
                comps.append({"model": a, "against": b, "scheme": scheme, "measure": "q2", **g})
            if part == "all" and lad.any():
                g = paired_rmse(y[lad], preds[(a, "ladders")], preds[(b, "ladders")])
                comps.append({"model": a, "against": b, "scheme": "ladders held out",
                              "measure": "RMSE", **g})
        for c in comps:
            log(f"  {gas:4s} {part:6s} {c['model']} - {c['against']} [{c['scheme']}, "
                f"{c['measure']}]: {c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")
        out[part] = {"polymers": int(len(y)), "ladders": int(lad.sum()),
                     "clusters": int(len(set(groups.tolist()))),
                     "y": {"mean": float(y.mean()), "sd": float(y.std()),
                           "ladders_mean": float(y[lad].mean()) if lad.any() else None,
                           "linear_mean": float(y[~lad].mean())},
                     "models": res, "comparisons": comps}
    P.SCHEMES = Q.SCHEMES
    return out


def run(jobs: int = 16, build_only: bool = False) -> Optional[dict]:
    t0 = time.time()
    L.selfcheck()
    feat = Features(jobs)
    checks = audit(feat)
    if build_only:
        return None
    rows = [r for r, ok in zip(feat.rows, feat.ok) if ok]
    ladder = np.array([r["ladder"] == "1" for r in rows])
    counts = feat.counts()
    Zs = L.standardise(np.hstack([counts, side_groups(rows)]))
    kern = Kernels(feat, rows, L.standardise(counts), Zs)
    log(f"{len(rows)} polymers with a trimer, {int(ladder.sum())} ladders "
        f"({time.time() - t0:.0f} s)")
    out = {g: run_gas(g, kern, ladder, t0) for g in GASES}
    payload = {"gases": out, "audit": checks, "not_built": feat.errors,
               "polymers": len(rows), "ladders": int(ladder.sum()),
               "repaired": int(sum(1 for r in rows if r["repair"])),
               "repeats": Q.REPEATS, "folds": P.FOLDS, "sigmas": list(P.SIGMAS),
               "topo_max": list(P.TOPO_MAX), "count_widths": list(P.COUNT_WIDTHS),
               "hs": list(P.HS), "lams": list(P.LAMS), "draws": DRAWS,
               "seconds": round(time.time() - t0, 1)}
    L.jdump(payload, os.path.join(RESULTS, "polymer_ladders.json"))
    report(payload)
    return payload


# --------------------------------------------------------------------------- the report

def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "polymer_ladders.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    au = payload["audit"]
    lines = [f"Plan item PL6: ladder polymers. The measured permeability (log10 Barrer) of "
             f"{payload['polymers']} polymers, {payload['ladders']} of them ladder polymers; "
             f"{payload['repaired']} repeat units with a repaired structure. Five random folds and "
             f"five folds of ECFP4 Butina clusters, each repeated {payload['repeats']} times; for "
             f"the ladders held out, the models are trained on the linear polymers only. The "
             f"setting and the ridge are chosen by the evidence on the training block.", "",
             f"Built here: {len(au['ladders'])} ladder trimers "
             f"({sum(1 for x in au['ladders'] if x['faults'])} with geometry faults) and "
             f"{len(au['repaired'])} repaired linear trimers "
             f"({sum(1 for x in au['repaired'] if x['faults'])} with faults). End-to-end distance "
             f"of the ladder trimers: {min(x['span'] for x in au['ladders']):.1f} to "
             f"{max(x['span'] for x in au['ladders']):.1f} A.", ""]
    for gas in GASES:
        g = payload["gases"][gas]
        lin, al = g["linear"], g["all"]
        lines += [f"**{gas}** ({lin['polymers']} linear polymers, {al['ladders']} ladders; mean "
                  f"{al['y']['linear_mean']:.2f} for the linear polymers and "
                  f"{al['y']['ladders_mean']:.2f} for the ladders)", "",
                  "| Model | q², linear, random folds | q², linear, clusters held out | "
                  "q², all, random folds | q², all, clusters held out | RMSE on the ladders, "
                  "held out | mean error on the ladders, held out | RMSE on the ladders, "
                  "random folds of all |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m in MODELS:
            a, b = lin["models"][m], al["models"][m]
            lines.append(f"| {m} | {a['random']['q2_mean']:+.3f} | {a['group']['q2_mean']:+.3f} | "
                         f"{b['random']['q2_mean']:+.3f} | {b['group']['q2_mean']:+.3f} | "
                         f"{b['ladders']['rmse']:.3f} | {b['ladders']['mean_error']:+.3f} | "
                         f"{b['random']['ladders']['rmse']:.3f} |")
        lines += ["", "| Difference | Polymers | Folds | Measure | Mean [95 % interval] |",
                  "|---|---|---|---|---|"]
        for part in ("linear", "all"):
            for c in g[part]["comparisons"]:
                lines.append(f"| {c['model']} minus {c['against']} | {part} | {c['scheme']} | "
                             f"{c['measure']} | {c['mean']:+.3f} [{c['lo']:+.3f}, "
                             f"{c['hi']:+.3f}] |")
        lines.append("")
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16, help="processes for building the trimers")
    ap.add_argument("--build", action="store_true", help="build the new trimers only")
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    run(a.jobs, a.build)
    return 0


if __name__ == "__main__":
    sys.exit(main())
