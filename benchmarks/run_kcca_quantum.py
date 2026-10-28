#!/usr/bin/env python
"""
What the quantum fields see that shape and charge do not: kernel CCA of the two field views.

The conceptual-DFT fields -- the Fukui functions f+ and f- and the dual descriptor, one Gaussian
per atom weighted by the condensed index -- were measured as an *addition* to the five empirical
fields (``run_corrosion_qm.py``, ``run_xtb.py``): they raise q² on a minority of series and cost
the rest a little. That measurement cannot say *why*. Kernel canonical correlation analysis
(``cmfqsar/kcca.py``, plan item F6) can: it finds the directions the two views share and the
directions of the quantum view that the empirical view does not contain, and asks whether that
own part carries the activity. If it does not, the reactivity fields fail to help because shape
and charge already carry what they know about these molecules; if it does, they fail for another
reason, such as noise or the attenuation.

Two sources of quantum fields, both already computed and cached:

* **ORCA** on four corrosion series (``results/corrosion_qm/<set>/<set>_qm.sdf``): Hirshfeld
  condensed indices from a density-functional single point, 20 to 29 molecules each.
* **GFN2-xTB** on every series of the repository large enough to cross-validate
  (``results/xtb/work``, 52 series): the cheap stage of plan item D3, reused from its cache.

View A is the three reactivity fields, view B the five empirical fields, both cosine-normalised at
the attenuation the evidence of the empirical model picks for the series. The analysis is the one
of ``run_kcca.py``: ridge by held-out correlation, a shuffled pairing as the null, a direction of
view A counted as contained in view B when B explains more than half of its variance on held-out
molecules, and kernel ridge regression on the contained and on the own part of A with the
regularisation chosen inside the folds. The reverse direction, what shape and charge see that the
reactivity fields do not, is recorded as well. The two views have different fields, so the
co-field cosine of gate (iii) does not apply here.

Usage
-----
    python benchmarks/run_kcca_quantum.py                 # ORCA sets, then every xTB series
    python benchmarks/run_kcca_quantum.py --source orca
    python benchmarks/run_kcca_quantum.py --sets fep__bace sutherland__THR --force
    python benchmarks/run_kcca_quantum.py --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

import kernel_cache as KCACHE  # noqa: E402
from cmfqsar import kcca as KC  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "kcca_quantum")
ORCA_RESULTS = os.path.join(HERE, "results", "corrosion_qm")
XTB_RESULTS = os.path.join(HERE, "results", "xtb")
EMPIRICAL = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
QUANTUM = ["fukui_plus", "fukui_minus", "dual"]
ALPHAS = (0.3, 0.5, 1.0, 2.0)
LAMS = (1e-2, 0.1, 1.0)
COMPONENTS = 3
PERMUTATIONS = 200
FOLDS = 5
SEED = 0
MIN_MOLECULES = 20


def _kernel(mols, names, physchem, props, alpha):
    fs = FieldSet.create(physchem=physchem, alpha=float(alpha), charge_model="gasteiger",
                         param_scheme="rdkit", atom_properties=props)
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    probe = CMFModel(fieldset=fs, method="krr", normalize=True)
    return probe.joint_kernel(KCACHE.training_kernels(probe, table))


def _choose_alpha(mols, names, y):
    best = None
    for a in ALPHAS:
        K = _kernel(mols, names, EMPIRICAL, [], a)
        for lam in LAMS:
            e = log_evidence(K, y, float(lam), fit_bias=True)
            if np.isfinite(e) and (best is None or e > best[0]):
                best = (e, float(a))
    return best[1]


# --------------------------------------------------------------------------- the molecules


def orca_series():
    """{tag: (molecules, names, y)} of the corrosion series with a finished ORCA run."""
    import run_corrosion_qm as Q

    out = {}
    for sdf in sorted(glob.glob(os.path.join(ORCA_RESULTS, "*", "*_qm.sdf"))):
        set_name = os.path.basename(os.path.dirname(sdf))
        annotated = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(sdf, removeHs=False)
                     if m is not None}
        try:
            mols, names, y, _core, _ref, _rmsd = Q.prepare(os.path.join(Q.C.DATA, set_name + ".csv"))
        except Exception as exc:                        # noqa: BLE001 - a set that cannot be prepared
            print(f"  {set_name}: cannot prepare the series ({exc}); skipped")
            continue
        keep = [i for i, n in enumerate(names) if n in annotated]
        if len(keep) < MIN_MOLECULES:
            continue
        out[f"orca__{set_name}"] = ([annotated[names[i]] for i in keep],
                                    [names[i] for i in keep], np.asarray(y, float)[keep])
    return out


def xtb_series(only=None):
    """{tag: (molecules, names, y)} of every series the xTB benchmark computed, from its cache."""
    import run_xtb as X

    out = {}
    cols = X.collections()
    for coll, items in cols.items():
        for name, path in items:
            tag = f"{coll}__{name}"
            if only and tag not in only:
                continue
            if not os.path.isfile(os.path.join(XTB_RESULTS, "sets", f"{tag}.json")):
                continue
            got = X.series_of(path)
            if got is None:
                continue
            mols, names, y = got
            good, _failed, _fresh = X.annotate(mols, names, os.path.join(XTB_RESULTS, "work", tag))
            keep = sorted(good)
            if len(keep) < MIN_MOLECULES:
                continue
            out[tag] = ([good[i] for i in keep], [names[i] for i in keep],
                        np.asarray(y, float)[keep])
    return out


# --------------------------------------------------------------------------- the analysis


def run_series(tag, mols, names, y, force=False, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{tag}.json")
    if os.path.exists(out_path) and not force:
        print(f"{tag}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    alpha = _choose_alpha(mols, names, y)
    KE = _kernel(mols, names, EMPIRICAL, [], alpha)
    KQ = _kernel(mols, names, [], QUANTUM, alpha)
    print(f"=== {tag}: {len(mols)} molecules, alpha {alpha:g}", flush=True)
    res_q = KC.analyse(KQ, KE, y=y, ridge=None, components=COMPONENTS, folds=FOLDS, seed=SEED,
                       permutations=PERMUTATIONS)
    res_e = KC.analyse(KE, KQ, y=y, ridge=None, components=COMPONENTS, folds=FOLDS, seed=SEED,
                       permutations=PERMUTATIONS)
    q = res_q.summary()
    e = res_e.summary()
    d, lo, hi = q["own_minus_contained"]
    print(f"  quantum view: held-out rho {np.round(q['rho_held_out'][:3], 3)}, {q['real']} of "
          f"{q['pairs_held_out']} real, {q['contained']} of {q['pairs']} contained; q2 whole "
          f"{q['q2_all']:+.3f} contained {q['q2_contained']:+.3f} own {q['q2_own']:+.3f} "
          f"({d:+.3f} [{lo:+.3f}, {hi:+.3f}]); empirical whole {e['q2_all']:+.3f}, own "
          f"{e['q2_own']:+.3f}", flush=True)
    out = {"set": tag, "source": tag.split("__")[0], "n": len(mols), "alpha": alpha,
           "quantum": q, "empirical": e, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def report(results=RESULTS):
    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No results yet; run `python benchmarks/run_kcca_quantum.py`."
    data = [json.load(open(f)) for f in files]
    orca = sorted([d for d in data if d["source"] == "orca"], key=lambda d: -d["n"])
    xtb = sorted([d for d in data if d["source"] != "orca"], key=lambda d: -d["n"])
    lines = ["# What the quantum fields see that shape and charge do not", "",
             "View A is the three reactivity fields (f+, f− and the dual descriptor, one Gaussian "
             "per atom weighted by the condensed index), view B the five empirical fields, both "
             "at the attenuation the evidence picks for the empirical model. *Held-out ρ* is the "
             "mean canonical correlation over all pairs on held-out molecules; *real* counts the "
             "pairs above the 95th percentile of a shuffled pairing; *contained* counts the "
             "directions of the quantum view of which the empirical view explains more than half "
             "the variance on held-out molecules. The q² values are kernel ridge regression on "
             "the whole quantum view, on its contained part and on its own part, with the "
             "regularisation chosen inside the folds; the last two columns are the empirical "
             "view's whole and its own part relative to the quantum view.", ""]

    def table(rows, title):
        if not rows:
            return []
        out = [f"## {title}", "",
               "| series | n | α | mean ρ | real | contained | q² quantum | q² contained | "
               "q² quantum's own | own − contained [95 %] | q² empirical | q² empirical's own |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
        for d in rows:
            q, e = d["quantum"], d["empirical"]
            dd, lo, hi = q["own_minus_contained"]
            out.append(f"| {d['set'].split('__', 1)[1]} | {d['n']} | {d['alpha']:g} | "
                       f"{q['rho_held_out_mean']:+.3f} | {q['real']}/{q['pairs_held_out']} | "
                       f"{q['contained']}/{q['pairs']} | {q['q2_all']:+.3f} | "
                       f"{q['q2_contained']:+.3f} | {q['q2_own']:+.3f} | {dd:+.3f} [{lo:+.3f}, "
                       f"{hi:+.3f}] | {e['q2_all']:+.3f} | {e['q2_own']:+.3f} |")
        qa = np.array([d["quantum"]["q2_all"] for d in rows])
        qc = np.array([d["quantum"]["q2_contained"] for d in rows])
        qo = np.array([d["quantum"]["q2_own"] for d in rows])
        ea = np.array([d["empirical"]["q2_all"] for d in rows])
        eo = np.array([d["empirical"]["q2_own"] for d in rows])
        dd = np.array([d["quantum"]["own_minus_contained"][0] for d in rows])
        out.append(f"| **mean** |  |  |  |  |  | **{np.nanmean(qa):+.3f}** | **{np.nanmean(qc):+.3f}** "
                   f"| **{np.nanmean(qo):+.3f}** | **{np.nanmean(dd):+.3f}** | "
                   f"**{np.nanmean(ea):+.3f}** | **{np.nanmean(eo):+.3f}** |")
        above = sum(1 for d in rows if d["quantum"]["own_minus_contained"][1] > 0)
        below = sum(1 for d in rows if d["quantum"]["own_minus_contained"][2] < 0)
        share = np.mean([d["quantum"]["contained"] / max(d["quantum"]["pairs"], 1) for d in rows])
        real = np.mean([d["quantum"]["real"] / max(d["quantum"]["pairs_held_out"], 1) for d in rows])
        out += ["",
                f"Over {len(rows)} series: {100 * real:.0f} % of the canonical pairs hold out above "
                f"the shuffled pairing, and the empirical view contains {100 * share:.0f} % of the "
                f"quantum view's directions on average. The quantum view's own part predicts the "
                f"activity at q² **{np.nanmean(qo):+.3f}** on average against **{np.nanmean(qc):+.3f}** "
                f"for its contained part and **{np.nanmean(qa):+.3f}** for the whole quantum view; the "
                f"difference (own minus contained) is **{np.nanmean(dd):+.3f}**, with its interval "
                f"above zero on {above} and below zero on {below} of the {len(rows)} series. The "
                f"empirical view alone reaches **{np.nanmean(ea):+.3f}**, and its own part relative to "
                f"the quantum view **{np.nanmean(eo):+.3f}**.", ""]
        return out

    lines += table(orca, "ORCA: Hirshfeld indices on the corrosion series")
    lines += table(xtb, "GFN2-xTB: every series the cheap quantum stage was run on")
    rows = orca + xtb
    if rows:
        qo = np.nanmean([d["quantum"]["q2_own"] for d in rows])
        qc = np.nanmean([d["quantum"]["q2_contained"] for d in rows])
        eo = np.nanmean([d["empirical"]["q2_own"] for d in rows])
        ea = np.nanmean([d["empirical"]["q2_all"] for d in rows])
        if qc > qo and ea > qo:
            verdict = ("So what the reactivity fields know about the activity of these series is the "
                       "part of them that shape and charge already contain: the directions that are "
                       "theirs alone carry little. That is why adding them to the five fields gains "
                       "little in the earlier measurements, and it is a statement about the "
                       "condensed, atom-centred form of the fields on these series, not about "
                       "conceptual DFT.")
        else:
            verdict = ("So the reactivity fields carry activity in directions the empirical fields "
                       "do not contain, and the earlier measurements' small gains have another "
                       "cause than redundancy.")
        lines += ["## Verdict", "", verdict, ""]
    text = "\n".join(lines)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["orca", "xtb", "both"], default="both")
    ap.add_argument("--sets", nargs="*", default=None, help="series tags, e.g. fep__bace")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        series = {}
        if a.source in ("orca", "both"):
            series.update(orca_series())
        if a.source in ("xtb", "both"):
            series.update(xtb_series(only=set(a.sets) if a.sets else None))
        for tag, (mols, names, y) in series.items():
            if a.sets and tag not in a.sets:
                continue
            try:
                run_series(tag, mols, names, y, force=a.force)
            except Exception as exc:                     # noqa: BLE001 - one series must not stop the rest
                print(f"  {tag}: FAILED -- {type(exc).__name__}: {exc}", flush=True)
    print(report())


if __name__ == "__main__":
    main()
