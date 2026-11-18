#!/usr/bin/env python
"""
Plan item PL2: two polymer properties that depend on how chains pack -- density from molecular
dynamics and measured gas permeability -- and whether the 3D field sees more there than the bond
graph does.

The sets are made by ``benchmarks/prepare_polymer_packing.py``:

* ``examples/polymer_packing/pi1070.csv``: the density at 300 K of 1,077 amorphous homopolymers
  from RadonPy PI1070 (all-atom molecular dynamics, one protocol for all; BSD 3-Clause), with the
  polymer class RadonPy gives (20 classes).
* ``examples/polymer_packing/permeability.csv``: log10 of the permeability in Barrer of 742
  polymers to CH4, CO2, H2, N2 and O2 (450 to 728 labels per gas; measured; MIT, from
  torch-molecule).

The question, fixed before the measurement
------------------------------------------
On the glass transition temperature (PL1, ``run_polymer_tg.py``) the field kernel of the trimer
plus eight counts equalled the same channels with distances counted in bonds plus the counts, to
the third decimal: Tg is a property of the whole repeat unit and the graph carries it.  Density
and permeability are set by how chains pack, which is closer to what a 3D field describes.  The
gate of PL2 is

    central unit + eight counts  minus  bond-count kernel + eight counts  >  0,

with the 95 % paired bootstrap interval above zero, on each target and fold scheme.

The run of 18 November 2026 answered it: on gas permeability the gate is met with groups held
out for all five gases (+0.059 to +0.084) and on random folds for CH4, N2 and O2; on density it
is not met on either test.  ECFP4 is the best permeability model.

The sample and the models are those of ``run_polymer_tg.py``: a trimer capped with hydrogen, the
stereocentres fixed once ("isotactic"), the most extended of four ETKDG conformations, MMFF94 with
the end-to-end distance held; the five fields; kernel ridge regression with the setting and the
ridge chosen by the evidence on each training block.  The models are the field kernel of the
whole trimer and of its middle unit, the bond-count kernel of the middle unit, the eight counts,
ECFP4 of the repeat unit, and the sums of each of the three structure kernels with the eight
counts.  The PolyMetriX descriptors of PL1 are not used: they came with the Tg data.

The folds: five random folds repeated ten times, and five folds of whole groups repeated ten times,
the groups dealt in a random order to the smallest fold.  A group is a RadonPy polymer class for
the density, and a Butina cluster of the repeat units' ECFP4 fingerprints (Tanimoto distance 0.6)
for the permeability, which has no classes.  The bootstrap of the second scheme draws whole
groups.

Usage
-----
    python benchmarks/run_polymer_packing.py --build   # the trimers and their features (cached)
    python benchmarks/run_polymer_packing.py           # the models; builds what is missing
    python benchmarks/run_polymer_packing.py --report
"""
from __future__ import annotations

import argparse
import csv
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

import run_logp as L  # noqa: E402
import run_polymer_tg as P  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "polymer_packing")
RESULTS = os.path.join(HERE, "results", "polymer_packing")
CACHE = os.path.join(HERE, "cache", "polymer_packing")
TAG = "packing"
REPEATS = 10
CLUSTER_DISTANCE = 0.6
#: (target, table, label column, group column or None for Butina clusters, unit)
TARGETS = (("density", "pi1070.csv", "density", "polymer_class", "g/cm3"),
           ("CH4", "permeability.csv", "log10_CH4", None, "log10 Barrer"),
           ("CO2", "permeability.csv", "log10_CO2", None, "log10 Barrer"),
           ("H2", "permeability.csv", "log10_H2", None, "log10 Barrer"),
           ("N2", "permeability.csv", "log10_N2", None, "log10 Barrer"),
           ("O2", "permeability.csv", "log10_O2", None, "log10 Barrer"))
MODELS = ("field kernel, whole trimer", "field kernel, central unit",
          "bond-count kernel, central unit", "eight counts", "ECFP4 of the repeat unit",
          "whole trimer + eight counts", "central unit + eight counts",
          "bond-count kernel + eight counts")
GATE = ("central unit + eight counts", "bond-count kernel + eight counts")
SCHEMES = ("random", "group")
log = L.log

# the shared code of PL1 reads these module globals
P.CACHE = CACHE
P.REPEATS = REPEATS
P.SCHEMES = SCHEMES
#: the first full run chose 0.005 A, the narrowest of PL1's list, in most blocks for every gas,
#: and q2 at a fixed width still rose from 0.01 to 0.005 A, so PL2 goes down to 0.001 A
P.SIGMAS = (0.001, 0.0025) + tuple(P.SIGMAS)


def read(name: str) -> List[dict]:
    with open(os.path.join(DATA, name), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def union_rows():
    """One row per distinct repeat unit over both tables, in a fixed order."""
    seen: Dict[str, str] = {}
    for name in ("pi1070.csv", "permeability.csv"):
        for r in read(name):
            key = Chem.MolToSmiles(Chem.MolFromSmiles(r["psmiles"]))
            seen.setdefault(key, r["psmiles"])
    keys = sorted(seen)
    return [{"id": f"u{k + 1:04d}", "psmiles": seen[key]} for k, key in enumerate(keys)], keys


def butina_groups(psmiles: List[str]) -> np.ndarray:
    """A Butina cluster number for each repeat unit, from ECFP4 at Tanimoto distance 0.6."""
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in psmiles]
    dists = []
    for i in range(1, len(fps)):
        dists.extend(1.0 - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])))
    clusters = Butina.ClusterData(dists, len(fps), CLUSTER_DISTANCE, isDistData=True)
    out = np.zeros(len(fps), int)
    for c, members in enumerate(clusters):
        out[list(members)] = c
    return out


def make_blocks(n: int, groups: np.ndarray):
    out = []
    for rep in range(REPEATS):
        for te in L.RW.folds(n, P.FOLDS, P.SEED + rep):
            out.append(("random", rep, np.setdiff1d(np.arange(n), te), te))
    for rep in range(REPEATS):
        for te in P.class_folds(groups.astype(str), P.FOLDS, P.SEED + rep):
            out.append(("group", rep, np.setdiff1d(np.arange(n), te), te))
    return out


class Kernels:
    """The kernels on all repeat units, each made once and kept; a target takes its rows."""

    def __init__(self, feat, keep, units, counts):
        self.feat, self.keep, self.units, self.counts = feat, keep, units, counts
        self.store = {}

    def get(self, key):
        if key not in self.store:
            kind, arg = key
            if kind in ("whole", "central"):
                K = P.field_kernel(P.autocorr_rows(self.feat, kind, arg, TAG, self.keep), arg)
            elif kind == "topo":
                K = P.topo_kernel(np.asarray(self.feat["topo"][self.keep], float), arg)
            elif kind == "counts":
                K = L.gaussian_at(self.counts, arg)
            elif kind == "ecfp":
                from cmfqsar.descriptors import fingerprint_kernel
                K = fingerprint_kernel([Chem.MolFromSmiles(s) for s in self.units])
            else:
                raise KeyError(kind)
            self.store[key] = K
        return self.store[key]


def run_target(spec, kern: Kernels, pos: Dict[str, int], t0: float) -> dict:
    from cmfqsar.model import r2_score, rmse
    name, table, col, group_col, unit = spec
    rows = [r for r in read(table) if r[col] != ""]
    keys = [Chem.MolToSmiles(Chem.MolFromSmiles(r["psmiles"])) for r in rows]
    have = [k for k, key in enumerate(keys) if key in pos]
    rows = [rows[k] for k in have]
    idx = np.array([pos[keys[k]] for k in have])
    y = np.array([float(r[col]) for r in rows])
    groups = (np.array([int(r[group_col]) for r in rows]) if group_col
              else butina_groups([r["psmiles"] for r in rows]))
    blocks = make_blocks(len(y), groups)
    sub = lambda key: kern.get(key)[np.ix_(idx, idx)]          # noqa: E731
    P.Model.cache_dir = os.path.join(CACHE, "settings")
    models = {m: P.Model(m) for m in MODELS}
    for s in P.SIGMAS:
        models["field kernel, whole trimer"].add(f"sigma {s:g}", lambda s=s: sub(("whole", s)),
                                                 y, blocks)
        models["field kernel, central unit"].add(f"sigma {s:g}", lambda s=s: sub(("central", s)),
                                                 y, blocks)
    for dm in P.TOPO_MAX:
        models["bond-count kernel, central unit"].add(f"up to {dm} bonds",
                                                      lambda dm=dm: sub(("topo", dm)), y, blocks)
    for w in P.COUNT_WIDTHS:
        models["eight counts"].add(f"width {w:g}", lambda w=w: sub(("counts", w)), y, blocks)
    models["ECFP4 of the repeat unit"].add("Tanimoto", lambda: sub(("ecfp", 0)), y, blocks)
    nb = len(blocks)
    cwidth = [models["eight counts"].best_label(k) for k in range(nb)]
    pairs = (("whole trimer + eight counts", "field kernel, whole trimer", "whole", P.SIGMAS,
              "sigma {:g}"),
             ("central unit + eight counts", "field kernel, central unit", "central", P.SIGMAS,
              "sigma {:g}"),
             ("bond-count kernel + eight counts", "bond-count kernel, central unit", "topo",
              P.TOPO_MAX, "up to {} bonds"))
    for combo, single, kind, grid, fmt in pairs:
        chosen = [models[single].best_label(k) for k in range(nb)]
        for a in grid:
            for w in P.COUNT_WIDTHS:
                only = {k for k in range(nb) if chosen[k] == fmt.format(a)
                        and cwidth[k] == f"width {w:g}"}
                if not only:
                    continue
                for h in P.HS:
                    models[combo].add(f"{fmt.format(a)}, width {w:g}, h {h:g}",
                                      lambda a=a, w=w, h=h: sub((kind, a)) + h * sub(("counts", w)),
                                      y, blocks, only)
    log(f"{name}: {len(y)} polymers, {len(set(groups.tolist()))} groups, models done "
        f"({time.time() - t0:.0f} s)")
    res, preds = {}, {}
    for m, model in models.items():
        pred, chosen = model.select(len(y), blocks)
        r = {}
        for scheme in SCHEMES:
            p = pred[scheme]
            q2 = [float(r2_score(y, p[rep])) for rep in range(REPEATS)]
            r[scheme] = {"q2_mean": float(np.mean(q2)), "q2_by_repeat": q2,
                         "q2_range": float(np.max(q2) - np.min(q2)),
                         "rmse_mean": float(np.mean([rmse(y, p[rep]) for rep in range(REPEATS)])),
                         "chosen": chosen[scheme]}
            preds[(m, scheme)] = p.mean(axis=0)
        res[m] = r
        log(f"  {name:8s} {m:34s} random q2 {r['random']['q2_mean']:+.3f} "
            f"(RMSE {r['random']['rmse_mean']:.3f}) | group q2 {r['group']['q2_mean']:+.3f}")
    comps = []
    for a, b in (GATE, ("whole trimer + eight counts", "bond-count kernel + eight counts"),
                 ("field kernel, central unit", "bond-count kernel, central unit"),
                 ("central unit + eight counts", "eight counts"),
                 ("central unit + eight counts", "ECFP4 of the repeat unit"),
                 ("field kernel, central unit", "field kernel, whole trimer")):
        for scheme in SCHEMES:
            pa, pb = preds[(a, scheme)], preds[(b, scheme)]
            g = (L.RS.paired(y, pa, pb) if scheme == "random"
                 else P.paired_groups(y, pa, pb, groups))
            comps.append({"model": a, "against": b, "scheme": scheme, **g})
            log(f"  {name:8s} {a} - {b} [{scheme}]: {g['mean']:+.3f} "
                f"[{g['lo']:+.3f}, {g['hi']:+.3f}]")
    return {"polymers": int(len(y)), "groups": int(len(set(groups.tolist()))), "unit": unit,
            "y": {"mean": float(y.mean()), "sd": float(y.std())}, "models": res,
            "comparisons": comps}


def run(jobs: int = 16, build_only: bool = False) -> Optional[dict]:
    t0 = time.time()
    L.selfcheck()
    rows, keys = union_rows()
    feat = P.build_features(rows, jobs, use_cache=True, tag=TAG)
    P.extend_widths(rows, jobs, TAG)
    keep = np.where(feat["ok"])[0]
    errors = P._errors(TAG)
    log(f"{len(keep)} of {len(rows)} trimers built ({time.time() - t0:.0f} s); "
        f"{len(errors)} could not be embedded")
    if build_only:
        return None
    pos = {keys[i]: j for j, i in enumerate(keep)}
    counts = L.standardise(feat["counts"][keep])
    kern = Kernels(feat, keep, [rows[i]["psmiles"] for i in keep], counts)
    out = {}
    for spec in TARGETS:
        out[spec[0]] = run_target(spec, kern, pos, t0)
    payload = {"targets": out, "built": int(len(keep)), "units": int(len(rows)),
               "not_embedded": errors, "repeats": REPEATS, "folds": P.FOLDS,
               "cluster_distance": CLUSTER_DISTANCE, "sigmas": list(P.SIGMAS),
               "topo_max": list(P.TOPO_MAX), "count_widths": list(P.COUNT_WIDTHS),
               "hs": list(P.HS), "lams": list(P.LAMS), "seconds": round(time.time() - t0, 1)}
    L.jdump(payload, os.path.join(RESULTS, "polymer_packing.json"))
    report(payload)
    return payload


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "polymer_packing.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    lines = [f"Plan item PL2: density at 300 K from molecular dynamics (RadonPy PI1070) and the "
             f"measured gas permeability of polymers (log10 Barrer). Each repeat unit is built as "
             f"a trimer capped with hydrogen. Five random folds and five folds of whole groups "
             f"(RadonPy polymer classes for the density, ECFP4 Butina clusters for the "
             f"permeability), each repeated {payload['repeats']} times; the setting and the ridge "
             f"are chosen by the evidence on the training block.", ""]
    # in the order of TARGETS: the json stores the targets with sorted keys
    for name in [s[0] for s in TARGETS if s[0] in payload["targets"]]:
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
    ap.add_argument("--jobs", type=int, default=16, help="processes for building the trimers")
    ap.add_argument("--build", action="store_true", help="build the trimers and features only")
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    run(a.jobs, a.build)
    return 0


if __name__ == "__main__":
    sys.exit(main())
