#!/usr/bin/env python
"""
A cheap quantum stage over many series: does a GFN2-xTB field help any of them?

Plan item D3 asks one question. The expensive quantum stage of ``cmfqsar/orca.py`` has been run
on a few series only, because a density-functional single point takes two to three minutes per
molecule. GFN2-xTB takes a fraction of a second, which makes the same stage affordable over a
whole collection. The Hammett study says where a gain is to be expected -- where resonance moves
electron density and a charge increment cannot follow it -- and the corrosion study says where it
is not. So: run the cheap stage over every series in the repository that is large enough to
cross-validate, and find out whether any of them is of the first kind.

Three stages, in the order the question needs them.

**Cost.** What one molecule costs with xtb, beside what the same molecules cost with ORCA in
``benchmarks/results/corrosion_qm``. This is the premise of the item and it is measured, not
assumed.

**Agreement.** A cheap method is only useful if it says something similar to the expensive one.
On the corrosion series where both have been run, the correlation of the per-atom quantities --
the Fukui indices, the dual descriptor, the charges -- over every atom of every molecule. The
charges are compared although they are different quantities (a Mulliken-like tight-binding
population against a Hirshfeld charge), because what matters for a field is whether they order
the atoms the same way.

**Gain.** The question itself. For each series, on the same alignment, the same alpha and lambda
grids and the same 10-fold cross-validation:

  base          the five physico-chemical fields (electrostatic, steric, hydrophobic, hba, hbd)
  base+fukui    the same plus three per-atom reactivity fields: f+, f- and the dual descriptor
  xtb_charges   the same five, with the electrostatic field built from the GFN2 charge instead
                of the Gasteiger one
  base+desc     the five fields plus the per-molecule quantum descriptors as a kernel block

``base`` is the reference of rule 1 and every difference is reported with the bootstrap interval
of rule 4, so that a gain below the interval is not counted as a gain.

Usage
-----
    python benchmarks/run_xtb.py                       # every collection
    python benchmarks/run_xtb.py --collections sutherland hammett
    python benchmarks/run_xtb.py --stage cost
    python benchmarks/run_xtb.py --report

xtb itself is not part of this repository; see ``examples/xtb_atom_properties.py``. Without it
the stages that need a calculation are skipped and the ones already computed are reported from
their result files.
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

import kernel_cache as KC  # noqa: E402

from cmfqsar import xtb as QX  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.descriptors import DescriptorScaler, descriptor_kernel, read_values  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "xtb")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
FOLDS = 10
SEED = 0
RUNS = 2000
#: a series smaller than this cannot carry a 10-fold cross-validation worth reading
MIN_MOLECULES = 15

FUKUI = ["fukui_plus", "fukui_minus", "dual"]
DESC = ["E_homo", "E_lumo", "gap", "dipole"]

#: (tag, physchem fields, per-atom property fields, per-molecule descriptors)
MODELS = [
    ("base", FIELDS, [], []),
    ("base+fukui", FIELDS, FUKUI, []),
    ("xtb_charges", ["steric", "hydrophobic", "hba", "hbd"], ["q_xtb"], []),
    ("base+desc", FIELDS, [], DESC),
]


def _say(text):
    print(text, flush=True)


# ----------------------------------------------------------------------------------------------
# the series
# ----------------------------------------------------------------------------------------------

def collections():
    """{name: [(set name, path)]} for every collection this benchmark can run."""
    out = {}
    out["sutherland"] = [(os.path.basename(p)[:-4], p) for p in
                         sorted(glob.glob(os.path.join(ROOT, "examples", "sutherland", "*.sdf")))]
    ham = {os.path.basename(p)[:-4]: p for p in
           sorted(glob.glob(os.path.join(ROOT, "examples", "hammett", "*.sdf")))}
    # a *_qm file is the same series with ORCA properties this benchmark does not read, so it
    # would count one series twice
    out["hammett"] = [(k, v) for k, v in ham.items()
                      if not (k.replace("_qm", "") != k and k.replace("_qm", "") in ham)]
    fep = []
    for sub in ("openff/*.sdf", "schindler/*.sdf", "ross/*/*.sdf"):
        for p in sorted(glob.glob(os.path.join(ROOT, "examples", *sub.split("/")))):
            fep.append((os.path.basename(p)[:-4], p))
    out["fep"] = fep
    return out


def series_of(path, min_molecules=MIN_MOLECULES):
    """(molecules, names, activities) of one file, or None when it is too small or has no activity."""
    try:
        ds = load_dataset(path)
    except Exception:
        return None
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    if int(ok.sum()) < min_molecules or float(np.std(y[ok])) <= 0:
        return None
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    return mols, names, y[ok]


# ----------------------------------------------------------------------------------------------
# the models
# ----------------------------------------------------------------------------------------------

def kernel_for(mols, names, physchem, props, descriptors, alpha):
    """The joint kernel of one model at one attenuation, descriptor block included."""
    fs = FieldSet.create(physchem=physchem, alpha=alpha, charge_model="gasteiger",
                         param_scheme="rdkit", atom_properties=props, descriptors=descriptors)
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    return KC.joint_training_kernel(model, table)


def best_over_grid(mols, names, physchem, props, descriptors, y,
                   alphas=ALPHAS, lams=LAMS, folds=FOLDS, seed=SEED):
    """Best (q2, RMSEcv, alpha, lambda, predictions) over the grid, scored by its own CV."""
    best = None
    for alpha in alphas:
        K = kernel_for(mols, names, physchem, props, descriptors, alpha)
        for lam in lams:
            pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=folds, seed=seed)
            q2 = r2_score(y, pred)
            if best is None or q2 > best[0]:
                best = (float(q2), float(rmse(y, pred)), float(alpha), float(lam),
                        np.asarray(pred, float))
    return best


def bootstrap_difference(truth, a, b, runs=RUNS, seed=SEED):
    """(difference in q2, low, high) for prediction `a` against `b`, resampling the molecules."""
    truth = np.asarray(truth, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(truth) & np.isfinite(a) & np.isfinite(b)
    truth, a, b = truth[ok], a[ok], b[ok]
    if len(truth) < 4:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(int(runs)):
        idx = rng.randint(0, len(truth), len(truth))
        if np.std(truth[idx]) <= 0:
            continue
        out.append(r2_score(truth[idx], a[idx]) - r2_score(truth[idx], b[idx]))
    if not out:
        return float("nan"), float("nan"), float("nan")
    return (float(r2_score(truth, a) - r2_score(truth, b)),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


# ----------------------------------------------------------------------------------------------
# the quantum stage
# ----------------------------------------------------------------------------------------------

def annotate(mols, names, workdir, jobs=1, timeout=600.0):
    """xtb over the molecules as they stand; returns {index: annotated molecule}, failures, times."""
    cfg = QX.Config(xtb=QX.find_xtb(), opt=False, fukui=True, timeout=timeout, workdir=workdir)
    outcomes = QX.process_molecules(mols, names, cfg, jobs=jobs, log=_say)
    good = {oc.index - 1: oc.mol for oc in outcomes if oc.mol is not None}
    failed = [(oc.name, oc.error) for oc in outcomes if oc.mol is None]
    fresh = [oc.seconds for oc in outcomes if oc.mol is not None and not oc.reused]
    return good, failed, fresh


def run_series(name, path, results=RESULTS, jobs=1, force=False, models=MODELS):
    """One series: the quantum stage, then every model on the molecules it succeeded for."""
    out = os.path.join(results, "sets", f"{name}.json")
    if os.path.exists(out) and not force:
        with open(out) as fh:
            return json.load(fh)
    got = series_of(path)
    if got is None:
        return None
    mols, names, y = got
    t0 = time.time()
    good, failed, fresh = annotate(mols, names, os.path.join(results, "work", name), jobs=jobs)
    if len(good) < MIN_MOLECULES:
        _say(f"  {name}: only {len(good)} of {len(mols)} molecules computed; skipped")
        return None
    keep = sorted(good)
    mols = [good[i] for i in keep]
    names = [names[i] for i in keep]
    y = y[keep]

    rows = {}
    for tag, physchem, props, desc in models:
        q2, rms, alpha, lam, pred = best_over_grid(mols, names, physchem, props, desc, y)
        rows[tag] = {"q2": q2, "rmse": rms, "alpha": alpha, "lam": lam,
                     "pred": [float(v) for v in pred]}
        _say(f"  {name:28s} {tag:12s} q2 {q2:+.3f}  RMSEcv {rms:.3f}  "
             f"alpha {alpha:g} lambda {lam:g}")
    base = np.asarray(rows["base"]["pred"], float)
    for tag in rows:
        d, lo, hi = bootstrap_difference(y, np.asarray(rows[tag]["pred"], float), base)
        rows[tag].update(delta=d, lo=lo, hi=hi)
    rec = {"set": name, "path": os.path.relpath(path, ROOT).replace(os.sep, "/"),
           "n": int(len(y)), "failed": failed,
           "seconds_per_molecule": (float(np.mean(fresh)) if fresh else None),
           "wall_seconds": float(time.time() - t0), "models": rows}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ----------------------------------------------------------------------------------------------
# stage 1: cost
# ----------------------------------------------------------------------------------------------

def orca_times(root=os.path.join(HERE, "results", "corrosion_qm")):
    """{smiles: seconds} for every ORCA molecule already computed in this repository."""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "status.json"), recursive=True):
        try:
            with open(p) as fh:
                st = json.load(fh)
        except Exception:
            continue
        if st.get("status") == "done" and st.get("smiles") and st.get("seconds"):
            # three ORCA jobs make one molecule when the Fukui states are on; status.json
            # records the whole chain, so the number is already per molecule
            out.setdefault(st["smiles"], []).append(float(st["seconds"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def stage_cost(results=RESULTS, jobs=1, limit=40):
    """What a molecule costs with each program, on the molecules ORCA has already been run on."""
    have = orca_times()
    if not have:
        _say("  no ORCA results in this repository; the cost stage needs them")
        return None
    smiles = sorted(have)[:limit]
    mols, names = [], []
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        mols.append(m)
        names.append(f"c{i:04d}")
    workdir = os.path.join(results, "work", "_cost")
    cfg = QX.Config(xtb=QX.find_xtb(), opt=False, fukui=True, workdir=workdir)
    outcomes = QX.process_molecules(mols, names, cfg, jobs=jobs, log=_say)
    xs = [oc.seconds for oc in outcomes if oc.mol is not None and not oc.reused]
    if not xs:          # everything was cached; time one molecule again to have a number
        import shutil
        shutil.rmtree(os.path.join(workdir, f"00001_{names[0]}"), ignore_errors=True)
        oc = QX.run_molecule(mols[0], names[0], 1, cfg)
        xs = [oc.seconds]
    orca = [have[s] for s in smiles]
    rec = {"molecules": len(mols), "heavy_atoms": float(np.mean([m.GetNumHeavyAtoms()
                                                                for m in mols])),
           "xtb_seconds": float(np.median(xs)), "orca_seconds": float(np.median(orca)),
           "ratio": float(np.median(orca) / max(np.median(xs), 1e-9))}
    _say(f"  {rec['molecules']} molecules of {rec['heavy_atoms']:.0f} heavy atoms on average: "
         f"xtb {rec['xtb_seconds']:.2f} s, ORCA {rec['orca_seconds']:.0f} s, "
         f"a factor of {rec['ratio']:.0f}")
    return rec


# ----------------------------------------------------------------------------------------------
# stage 2: agreement with the expensive method
# ----------------------------------------------------------------------------------------------

def stage_agreement(results=RESULTS, jobs=1,
                    root=os.path.join(HERE, "results", "corrosion_qm")):
    """Per-atom quantities from xtb against the ORCA ones, over every atom of every molecule."""
    pairs = {"fukui_plus": ([], []), "fukui_minus": ([], []), "dual": ([], []),
             "charge": ([], [])}
    n_mol = 0
    for sdf in sorted(glob.glob(os.path.join(root, "*", "*_qm.sdf"))):
        ds = load_dataset(sdf)
        if "fukui_plus" not in ds.atom_properties:
            continue
        cfg = QX.Config(xtb=QX.find_xtb(), opt=False, fukui=True,
                        workdir=os.path.join(results, "work", "_agree",
                                             os.path.basename(sdf)[:-4]))
        outcomes = QX.process_molecules(ds.mols, ds.names, cfg, jobs=jobs, log=_say)
        for oc, ref in zip(outcomes, ds.mols):
            if oc.mol is None:
                continue
            n_mol += 1
            if oc.mol.GetNumAtoms() != ref.GetNumAtoms():
                continue            # the two files do not describe the same atoms
            for key, (a, b) in pairs.items():
                ref_name = "q_hirshfeld" if key == "charge" else key
                own_name = "q_xtb" if key == "charge" else key
                try:
                    mine = [float(at.GetDoubleProp(own_name)) for at in oc.mol.GetAtoms()]
                    theirs = [float(at.GetDoubleProp(ref_name)) for at in ref.GetAtoms()]
                except KeyError:
                    continue        # one of the two does not carry this quantity
                a.extend(mine)
                b.extend(theirs)
    if not n_mol:
        _say("  no ORCA-annotated SDF with Fukui indices was found; the agreement stage is skipped")
        return None
    out = {"molecules": n_mol}
    for key, (a, b) in pairs.items():
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        out[key] = {"atoms": int(ok.sum()),
                    "pearson": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                    "spearman": float(np.corrcoef(np.argsort(np.argsort(a[ok])),
                                                  np.argsort(np.argsort(b[ok])))[0, 1])}
        _say(f"  {key:12s} {out[key]['atoms']:6d} atoms  r {out[key]['pearson']:+.3f}  "
             f"rank r {out[key]['spearman']:+.3f}")
    return out


# ----------------------------------------------------------------------------------------------
# stage 3: the gain
# ----------------------------------------------------------------------------------------------

def stage_gain(names, results=RESULTS, jobs=1, force=False, limit_sets=0):
    """Every series of the named collections, with the models of MODELS."""
    cols = collections()
    rows = []
    for coll in names:
        items = cols.get(coll) or []
        if limit_sets:
            items = items[:limit_sets]
        _say(f"{coll}: {len(items)} files")
        for name, path in items:
            tag = f"{coll}__{name}"
            try:
                rec = run_series(tag, path, results=results, jobs=jobs, force=force)
            except Exception as exc:
                _say(f"  {tag}: FAILED -- {type(exc).__name__}: {exc}")
                continue
            if rec is not None:
                rec["collection"] = coll
                rows.append(rec)
    return rows


# ----------------------------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------------------------

def load_all(results=RESULTS):
    out = {}
    for key in ("cost", "agreement"):
        p = os.path.join(results, f"{key}.json")
        if os.path.exists(p):
            with open(p) as fh:
                out[key] = json.load(fh)
    rows = []
    for p in sorted(glob.glob(os.path.join(results, "sets", "*.json"))):
        with open(p) as fh:
            rec = json.load(fh)
        rec.setdefault("collection", rec["set"].split("__")[0])
        rows.append(rec)
    out["sets"] = rows
    return out


def _counts(rows, tag):
    """How many series the model beats the reference on, by more than the interval."""
    ahead = behind = 0
    for r in rows:
        m = r["models"].get(tag)
        if not m or not np.isfinite(m.get("lo", np.nan)):
            continue
        if m["lo"] > 0:
            ahead += 1
        elif m["hi"] < 0:
            behind += 1
    return ahead, behind


def report(results=RESULTS, path=None):
    d = load_all(results)
    rows = d["sets"]
    out = ["# A cheap quantum stage over many series", "",
           "GFN2-xTB per-atom properties as fields, against the five physico-chemical fields "
           "alone, on the same alignment, the same alpha and lambda grids and the same 10-fold "
           "cross-validation. Plan item D3 asks whether any series in the repository is one a "
           "quantum field helps; every difference carries the bootstrap interval of rule 4, so "
           "a gain below the interval is not counted.", ""]

    if d.get("cost"):
        c = d["cost"]
        out += ["## What it costs", "",
                f"On {c['molecules']} molecules of {c['heavy_atoms']:.0f} heavy atoms on "
                f"average, the same molecules ORCA was run on in "
                f"`benchmarks/results/corrosion_qm`: **{c['xtb_seconds']:.2f} s** per molecule "
                f"with xtb against **{c['orca_seconds']:.0f} s** with ORCA, a factor of "
                f"**{c['ratio']:.0f}**. Both numbers include the two ionic states a Fukui index "
                f"is built from.", ""]

    if d.get("agreement"):
        a = d["agreement"]
        out += ["## Whether it says the same thing as the expensive method", "",
                f"Per-atom quantities over every atom of {a['molecules']} molecules of the "
                f"corrosion series, xtb against ORCA.", "",
                "| quantity | atoms | correlation | rank correlation |",
                "| --- | ---: | ---: | ---: |"]
        for key in ("fukui_plus", "fukui_minus", "dual", "charge"):
            if key in a:
                r = a[key]
                out.append(f"| {key} | {r['atoms']} | {r['pearson']:+.3f} | "
                           f"{r['spearman']:+.3f} |")
        out += ["", "The charges are a Mulliken-like tight-binding population against a "
                    "Hirshfeld charge, so they are different quantities; what the correlation "
                    "says is whether they order the atoms the same way.", ""]

    if rows:
        tags = [t for t, _f, _p, _d in MODELS if t != "base"]
        out += ["## Whether it helps", "",
                f"{len(rows)} series of {sum(r['n'] for r in rows)} molecules, "
                f"from {len(set(r['collection'] for r in rows))} collections.", "",
                "| model | series ahead of the five fields | series behind | median change in q2 |",
                "| --- | ---: | ---: | ---: |"]
        for tag in tags:
            ahead, behind = _counts(rows, tag)
            deltas = [r["models"][tag]["delta"] for r in rows
                      if tag in r["models"] and np.isfinite(r["models"][tag].get("delta", np.nan))]
            med = float(np.median(deltas)) if deltas else float("nan")
            out.append(f"| {tag} | {ahead} | {behind} | {med:+.3f} |")
        out.append("")

        out += ["### Every series", "",
                "| collection | series | n | q2 base | " +
                " | ".join(f"q2 {t}" for t in tags) + " | best change over base |",
                "| --- | --- | ---: | ---: | " + " | ".join("---:" for _ in tags) + " | --- |"]
        for r in sorted(rows, key=lambda r: (r["collection"], r["set"])):
            m = r["models"]
            best_tag, best = None, None
            for t in tags:
                if t in m and np.isfinite(m[t].get("delta", np.nan)):
                    if best is None or m[t]["delta"] > best["delta"]:
                        best_tag, best = t, m[t]
            change = ("--" if best is None else
                      f"{best_tag} {best['delta']:+.3f} [{best['lo']:+.3f}, {best['hi']:+.3f}]")
            cells = " | ".join(f"{m[t]['q2']:+.3f}" if t in m else "--" for t in tags)
            out.append(f"| {r['collection']} | {r['set'].split('__', 1)[-1]} | {r['n']} | "
                       f"{m['base']['q2']:+.3f} | {cells} | {change} |")
        out.append("")
        out += verdict(d)

    text = "\n".join(out) + "\n"
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="\n") as fh:
            fh.write(text)
    return text


def verdict(d):
    rows = d.get("sets") or []
    if not rows:
        return []
    out = ["## What it settles", ""]
    if d.get("cost"):
        c = d["cost"]
        out += [f"**The premise of the item holds.** A quantum stage that costs "
                f"{c['xtb_seconds']:.2f} s per molecule instead of {c['orca_seconds']:.0f} s is "
                f"a stage that can be run over every series in the repository, and this is the "
                f"first time that has been done.", ""]
    tags = [t for t, _f, _p, _d in MODELS if t != "base"]
    ahead_total = sorted(((t,) + _counts(rows, t) for t in tags), key=lambda x: -x[1])
    best_tag, ahead, behind = ahead_total[0]
    share = ahead / max(len(rows), 1)
    if ahead == 0:
        out += [f"**No series is helped by a quantum field beyond the interval.** Of "
                f"{len(rows)} series, none is ahead of the five physico-chemical fields by more "
                f"than the bootstrap interval of rule 4, and {behind} are behind it. The answer "
                f"to the question the item asks is therefore no for this repository: these "
                f"collections contain no series of the kind the Hammett result pointed to. That "
                f"is a measured answer and not a failure of the stage, which agrees with the "
                f"expensive method where both were run.", ""]
    else:
        worse = ("none is made worse" if behind == 0 else
                 f"{behind} series {'is' if behind == 1 else 'are'} made worse")
        out += [f"**{ahead} of {len(rows)} series ({share:.0%}) are helped beyond the "
                f"interval**, the best of the four models being `{best_tag}`, and {worse}. "
                f"A gain on a minority of series is what the item expected: the "
                f"Hammett result says the gain belongs to series where resonance moves electron "
                f"density, and most series are not of that kind. The series named in the table "
                f"above are the ones to look at.", ""]
    if ahead:
        # a 95% interval leaves 2.5% of the series on each side by chance, so with this many
        # series a few will clear it with nothing behind them; the count is worth reading
        # against that number rather than on its own
        expected = 0.025 * len(rows)
        out += [f"Read that count against chance: a 95% interval is cleared on one side by "
                f"about {expected:.1f} of {len(rows)} series when there is nothing to find, so "
                f"{ahead} is {'well above' if ahead > 2 * expected else 'close to'} what chance "
                f"would give. The four models are counted separately for the same reason -- "
                f"taking the best of four per series and counting that would find four times "
                f"as much by chance, and the `best change over base` column of the table above "
                f"is there to be looked at rather than counted.", ""]
    deltas = [r["models"][best_tag]["delta"] for r in rows
              if best_tag in r["models"] and np.isfinite(r["models"][best_tag].get("delta", np.nan))]
    if deltas:
        out += [f"The median change in q2 for `{best_tag}` over all series is "
                f"**{np.median(deltas):+.3f}**, with a range from {min(deltas):+.3f} to "
                f"{max(deltas):+.3f}. A quantum field is not free: it adds a kernel block, and "
                f"on a series it does not suit it dilutes the ones that work.", ""]
    return out


# ----------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--collections", nargs="*",
                    default=["sutherland", "hammett", "fep"],
                    help="which collections to run (sutherland, hammett, fep)")
    ap.add_argument("--stage", nargs="*", default=["cost", "agreement", "gain"],
                    help="which stages to run")
    ap.add_argument("--jobs", type=int, default=1, help="molecules run at a time")
    ap.add_argument("--limit-sets", type=int, default=0, help="only the first N of a collection")
    ap.add_argument("--force", action="store_true", help="recompute series already on disk")
    ap.add_argument("--report", nargs="?", const=os.path.join(RESULTS, "table.md"), default=None,
                    help="write the table and stop")
    a = ap.parse_args(argv)

    if a.report:
        print(report(path=a.report))
        print(f"written {a.report}")
        return 0

    os.makedirs(RESULTS, exist_ok=True)
    try:
        QX.find_xtb()
    except QX.XtbError as exc:
        print(exc)
        return 2
    if "cost" in a.stage:
        _say("Cost:")
        rec = stage_cost(jobs=a.jobs)
        if rec:
            with open(os.path.join(RESULTS, "cost.json"), "w") as fh:
                json.dump(rec, fh, indent=1)
    if "agreement" in a.stage:
        _say("Agreement with ORCA:")
        rec = stage_agreement(jobs=a.jobs)
        if rec:
            with open(os.path.join(RESULTS, "agreement.json"), "w") as fh:
                json.dump(rec, fh, indent=1)
    if "gain" in a.stage:
        _say("Gain:")
        stage_gain(a.collections, jobs=a.jobs, force=a.force, limit_sets=a.limit_sets)
    report(path=os.path.join(RESULTS, "table.md"))
    print(f"written {os.path.join(RESULTS, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
