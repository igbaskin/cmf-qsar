#!/usr/bin/env python
"""
Selectivity fields: the AMPA / glycine-NMDA set of the 2003 paper, modelled three ways.

    Baskin, Tikhonova, Palyulin, Zefirov, J. Med. Chem. 46 (2003) 4063.

The paper's claim is that the *difference* of two affinities is better modelled directly than
as the difference of two models: on 52 training compounds, CoMFA gave q2 = 0.52 (AMPA), 0.53
(glycine) and 0.61 for the difference, and on the five test compounds the selectivity model was
closer than the difference of the two single models.  The co-field of the difference model is the
"selectivity field".  ``examples/selectivity/ampa_glycine.sdf`` is that data set, superimposed by
the paper's rule (see ``examples/make_selectivity_examples.py``).

What is measured
----------------
For every setting below, on the published split (52 training, 5 test):

  single        three single-endpoint CMF models, of pKi(AMPA), pKi(glycine) and of their
                difference Delta = pKi(AMPA) - pKi(glycine), each with alpha and lambda chosen by
                cross-validation (``cmf build``)
  difference    the difference of the AMPA and glycine single models, scored against Delta: on
                the out-of-fold predictions and on the test set.  Against the direct Delta model
                this is the paper's comparison
  two-task      both affinities on one kernel, tied by a task correlation chosen by the marginal
                likelihood (``cmfqsar.multitask``), at the alpha and lambda of the AMPA model:
                the q2 of each endpoint with and without the sharing, and the q2 of the predicted
                difference, all with whole molecules held out; and the test set

Settings
--------
  comfa     electrostatic and steric fields only, as in CoMFA, leave-one-out cross-validation,
            which is what the paper's q2 is
  default   the five physico-chemical fields of the program, ten-fold cross-validation
  file      as default, with the Gasteiger-Hueckel charges of the Sybyl files instead of RDKit's
            Gasteiger charges.  The file charges of sixteen compounds do not sum to the formal
            charge (see the example builder), so this is the setting to mistrust

Each setting also runs with the two duplicated compounds removed (``-dedup``: 50 training
compounds; the published training set carries compounds 8 and 9 twice).

Usage
-----
    python benchmarks/run_selectivity.py              # every setting; results/selectivity/table.md
    python benchmarks/run_selectivity.py --settings comfa default --no-dedup
    python benchmarks/run_selectivity.py --report
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

import cmf_cli  # noqa: E402
import kernel_cache as KC  # noqa: E402
from cmfqsar import multitask as MT  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, r2_score, rmse  # noqa: E402

DATA = os.path.join(ROOT, "examples", "selectivity", "ampa_glycine.sdf")
RESULTS = os.path.join(HERE, "results", "selectivity")
ENDPOINTS = {"AMPA": "pKi_AMPA", "glycine": "pKi_glycine", "Delta": "delta_AMPA_glycine"}
SETTINGS = {
    "comfa": {"fields": ["electrostatic", "steric"], "charges": "gasteiger", "loo": True},
    "default": {"fields": ["electrostatic", "steric", "hydrophobic", "hba", "hbd"],
                "charges": "gasteiger", "loo": False},
    "file": {"fields": ["electrostatic", "steric", "hydrophobic", "hba", "hbd"],
             "charges": "file", "loo": False},
}
FOLDS = 10
SEED = 0
#: Table 1 of the paper: CoMFA, leave-one-out, (q2, R2, s, number of components)
PAPER = {"AMPA": (0.52, 0.877, 0.258, 4), "glycine": (0.528, 0.890, 0.276, 5),
         "Delta": (0.608, 0.894, 0.308, 5)}
#: Table 2 of the paper, the five test compounds: observed Delta, predicted by the selectivity
#: model, and the difference of the AMPA and glycine predictions
PAPER_TEST = {"cpd_53": (-0.63, -0.22, -0.19), "cpd_54": (-0.30, -0.10, 0.07),
              "cpd_55": (0.05, 0.16, 0.39), "cpd_56": (0.12, 0.30, 0.50),
              "cpd_57": (-0.21, -0.23, 0.73)}


def dataset_file(dedup, tmp_dir):
    """The example file, or a copy without the two duplicated training compounds."""
    if not dedup:
        return DATA
    path = os.path.join(tmp_dir, "ampa_glycine_dedup.sdf")
    if not os.path.exists(path):
        w = Chem.SDWriter(path)
        for m in Chem.SDMolSupplier(DATA, removeHs=False):
            if not m.HasProp("duplicate_of"):
                w.write(m)
        w.close()
    return path


def build_single(path, endpoint, setting, tag, n_train, results, force):
    out = os.path.join(results, f"{tag}_{endpoint}.json")
    if os.path.exists(out) and not force:
        return json.load(open(out))
    s = SETTINGS[setting]
    argv = ["build", path, "--activity", ENDPOINTS[endpoint], "--charges", s["charges"],
            "--fields", ",".join(s["fields"]), "--folds", str(n_train if s["loo"] else FOLDS),
            "--seed", str(SEED), "--json", out]
    print(f"--- cmf {' '.join(argv)}", flush=True)
    cmf_cli.main(argv)
    return json.load(open(out))


def split(res):
    sub = res["subset"]
    tr = [i for i, x in enumerate(sub) if x == "train"]
    te = [i for i, x in enumerate(sub) if x == "test"]
    y = np.asarray(res["y"], float)
    return tr, te, y


def _by_name(res, key, idx):
    """The predictions `key` of a build result (a name -> value map) for the molecules `idx`."""
    return np.array([float(res[key][res["names"][i]]) for i in idx])


def difference_of_models(res):
    """The AMPA model minus the glycine model, against the observed Delta and the Delta model."""
    a, g, d = res["AMPA"], res["glycine"], res["Delta"]
    tr, te, y_d = split(d)
    truth_cv = y_d[tr]
    diff_cv = _by_name(a, "y_cv", tr) - _by_name(g, "y_cv", tr)
    delta_cv = _by_name(d, "y_cv", tr)
    truth_te = y_d[te]
    diff_te = _by_name(a, "y_test_pred", te) - _by_name(g, "y_test_pred", te)
    delta_te = _by_name(d, "y_test_pred", te)
    return {
        "difference": {"q2": r2_score(truth_cv, diff_cv), "rmse_cv": rmse(truth_cv, diff_cv),
                       "rmse_test": rmse(truth_te, diff_te)},
        "delta_model": {"q2": r2_score(truth_cv, delta_cv), "rmse_cv": rmse(truth_cv, delta_cv),
                        "rmse_test": rmse(truth_te, delta_te)},
        "test": {d["names"][i]: {"observed": float(truth_te[k]), "delta_model": float(delta_te[k]),
                                 "difference": float(diff_te[k])} for k, i in enumerate(te)},
    }


def two_task(path, setting, alpha, lam, folds):
    """Both affinities on one kernel, at the AMPA model's alpha and lambda."""
    s = SETTINGS[setting]
    ds = load_dataset(path, activity_prop=ENDPOINTS["AMPA"])
    tr, te = list(ds.train_indices), list(ds.test_indices)
    fs = FieldSet.create(physchem=s["fields"], alpha=alpha, charge_model=s["charges"],
                         param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    table = AtomTable.from_molecules([mf[i] for i in tr])
    K = KC.joint_training_kernel(model, table)
    K_test = model.test_kernel(AtomTable.from_molecules([mf[i] for i in te]))
    Y = np.array([[float(m.GetProp(ENDPOINTS["AMPA"])), float(m.GetProp(ENDPOINTS["glycine"]))]
                  for m in ds.mols], float)
    obs = MT.Observations.from_columns(Y[tr], names=("AMPA", "glycine"))
    chosen = MT.evidence_rho(K, obs, lams=[lam])
    rho = float(chosen.correlations[0, 1])
    T = MT.task_covariance(2, rho)
    joint = MT.score_by_task(obs, MT.cv_predict(K, obs, lam=lam, T=T, folds=folds, seed=SEED))
    alone = MT.score_by_task(obs, MT.cv_predict(K, obs, rho=0.0, lam=lam, folds=folds, seed=SEED))
    sel = MT.score_selectivity(K, obs, 0, 1, rho=rho, lam=lam, folds=folds, seed=SEED)
    fit = MT.fit(K, obs, lam=lam, T=T)
    truth = Y[te, 0] - Y[te, 1]
    pred = fit.predict_selectivity(K_test, 0, 1)
    out = {"rho": rho, "lam": lam, "alpha": alpha, "folds": folds, "evidence": float(chosen.evidence),
           "tasks": [{"task": j["task"], "q2_shared": j["q2"], "q2_alone": al["q2"],
                      "rmse_cv_shared": j["rmse_cv"], "rmse_cv_alone": al["rmse_cv"],
                      "rmse_test": rmse(Y[te, t], fit.predict(K_test, t))}
                     for t, (j, al) in enumerate(zip(joint, alone))],
           "selectivity": {"q2": sel["q2"], "rmse_cv": sel["rmse_cv"], "rmse_test": rmse(truth, pred)},
           "test": {ds.names[i]: float(pred[k]) for k, i in enumerate(te)}}
    return out


def run(setting, dedup, results=RESULTS, force=False, tmp_dir=None):
    tag = setting + ("-dedup" if dedup else "")
    out_path = os.path.join(results, f"{tag}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    t0 = time.time()
    path = dataset_file(dedup, tmp_dir)
    n_train = sum(1 for m in Chem.SDMolSupplier(path) if m.GetProp("subset") == "train")
    print(f"\n=== {tag}: {n_train} training compounds, {path}", flush=True)
    res = {"setting": setting, "dedup": dedup, "tag": tag, "n_train": n_train,
           "fields": SETTINGS[setting]["fields"], "charges": SETTINGS[setting]["charges"],
           "folds": n_train if SETTINGS[setting]["loo"] else FOLDS, "single": {}}
    singles = {}
    for ep in ENDPOINTS:
        r = build_single(path, ep, setting, tag, n_train, results, force)
        singles[ep] = r
        res["single"][ep] = {k: r[k] for k in ("best_alpha", "r2", "rmse_fit", "q2", "rmse_cv",
                                                "r2_pred", "rmse_pred")}
        res["single"][ep]["lam"] = float(r["best_params"]["lam"])
        print(f"    {ep:8s} alpha {r['best_alpha']:g} lambda {r['best_params']['lam']:g}  "
              f"r2 {r['r2']:.3f}  q2 {r['q2']:+.3f}  RMSEcv {r['rmse_cv']:.3f}  "
              f"R2pred {r['r2_pred']:+.3f}  RMSEpred {r['rmse_pred']:.3f}", flush=True)
    res["comparison"] = difference_of_models(singles)
    c = res["comparison"]
    print(f"    Delta model     q2 {c['delta_model']['q2']:+.3f}  RMSEcv {c['delta_model']['rmse_cv']:.3f}  "
          f"test RMSE {c['delta_model']['rmse_test']:.3f}", flush=True)
    print(f"    AMPA - glycine  q2 {c['difference']['q2']:+.3f}  RMSEcv {c['difference']['rmse_cv']:.3f}  "
          f"test RMSE {c['difference']['rmse_test']:.3f}", flush=True)
    a = res["single"]["AMPA"]
    res["two_task"] = two_task(path, setting, a["best_alpha"], a["lam"], res["folds"])
    tt = res["two_task"]
    print(f"    two-task rho {tt['rho']:+.3f}: " + ", ".join(
        f"{t['task']} q2 {t['q2_shared']:+.3f} shared / {t['q2_alone']:+.3f} alone" for t in tt["tasks"])
          + f"; selectivity q2 {tt['selectivity']['q2']:+.3f}, test RMSE {tt['selectivity']['rmse_test']:.3f}",
          flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def _load(results):
    rows = []
    if os.path.isdir(results):
        for fn in sorted(os.listdir(results)):
            if fn.endswith(".json") and "_" not in fn:
                rows.append(json.load(open(os.path.join(results, fn))))
    order = {k: i for i, k in enumerate(SETTINGS)}
    rows.sort(key=lambda d: (order.get(d["setting"], 9), d["dedup"]))
    return rows


def report(results=RESULTS):
    rows = _load(results)
    out = ["## Selectivity fields: the AMPA / glycine-NMDA set of the 2003 paper", "",
           "57 tricyclic antagonists (52 training, 5 test; the training set carries two compounds "
           "twice, as published) superimposed on the tricyclic core. Three single-endpoint models "
           "per setting, alpha and lambda chosen by cross-validation; `comfa` is the two CoMFA "
           "fields with leave-one-out, `default` the five fields with ten folds, `file` the five "
           "fields with the Sybyl charges of the source files (unreliable for sixteen compounds); "
           "`-dedup` removes the two duplicates. The paper's CoMFA row is Table 1 of the paper "
           "(leave-one-out PLS, steric and electrostatic fields).", ""]
    out += ["| setting | endpoint | alpha | lambda | r2 | q2 | RMSEcv | R2pred (5) | RMSEpred (5) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ep, (q2, r2, s, noc) in PAPER.items():
        out.append(f"| paper (CoMFA) | {ep} | | {noc} components | {r2:.3f} | {q2:+.3f} | {s:.3f} | | |")
    for d in rows:
        for ep, r in d["single"].items():
            out.append(f"| {d['tag']} | {ep} | {r['best_alpha']:g} | {r['lam']:g} | {r['r2']:.3f} | "
                       f"{r['q2']:+.3f} | {r['rmse_cv']:.3f} | {r['r2_pred']:+.3f} | {r['rmse_pred']:.3f} |")
    out += ["", "The selectivity three ways: the direct model of Delta, the difference of the AMPA "
                "and glycine models, and the two-task fit (both affinities on one kernel, "
                "correlation rho by the marginal likelihood, the alpha and lambda of the AMPA "
                "model). q2 and RMSEcv on the training set with whole molecules held out; RMSE on "
                "the five test compounds.", "",
            "| setting | Delta model q2 / RMSEcv / RMSEtest | AMPA - glycine q2 / RMSEcv / RMSEtest | "
            "two-task rho | two-task AMPA q2 (alone) | glycine q2 (alone) | two-task Delta q2 / RMSEcv / RMSEtest |",
            "|---|---|---|---:|---|---|---|"]
    for d in rows:
        c, t = d["comparison"], d["two_task"]
        ta, tg = t["tasks"]
        out.append(f"| {d['tag']} | {c['delta_model']['q2']:+.3f} / {c['delta_model']['rmse_cv']:.3f} / "
                   f"{c['delta_model']['rmse_test']:.3f} | {c['difference']['q2']:+.3f} / "
                   f"{c['difference']['rmse_cv']:.3f} / {c['difference']['rmse_test']:.3f} | "
                   f"{t['rho']:+.2f} | {ta['q2_shared']:+.3f} ({ta['q2_alone']:+.3f}) | "
                   f"{tg['q2_shared']:+.3f} ({tg['q2_alone']:+.3f}) | {t['selectivity']['q2']:+.3f} / "
                   f"{t['selectivity']['rmse_cv']:.3f} / {t['selectivity']['rmse_test']:.3f} |")
    out += ["", "The five test compounds (Table 2 of the paper): observed Delta, the paper's "
                "selectivity model and difference of models, and the same from the `comfa` and "
                "`default` settings here.", "",
            "| compound | observed | paper: Delta model | paper: difference | comfa: Delta model | "
            "comfa: difference | comfa: two-task | default: Delta model | default: difference | default: two-task |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    by_tag = {d["tag"]: d for d in rows}
    for name, (obs, p_sel, p_diff) in PAPER_TEST.items():
        line = f"| {name} | {obs:+.2f} | {p_sel:+.2f} | {p_diff:+.2f} |"
        for tag in ("comfa", "default"):
            d = by_tag.get(tag)
            if d is None:
                line += " | | |"
                continue
            c = d["comparison"]["test"].get(name, {})
            line += (f" {c.get('delta_model', float('nan')):+.2f} | {c.get('difference', float('nan')):+.2f} | "
                     f"{d['two_task']['test'].get(name, float('nan')):+.2f} |")
        out.append(line)
    if rows:
        def rm(vals):
            return float(np.sqrt(np.mean(np.square(vals))))
        obs = np.array([v[0] for v in PAPER_TEST.values()])
        line = (f"| RMSE | | {rm(np.array([v[1] for v in PAPER_TEST.values()]) - obs):.2f} | "
                f"{rm(np.array([v[2] for v in PAPER_TEST.values()]) - obs):.2f} |")
        for tag in ("comfa", "default"):
            d = by_tag.get(tag)
            if d is None:
                line += " | | |"
                continue
            c, t = d["comparison"], d["two_task"]
            line += (f" {c['delta_model']['rmse_test']:.2f} | {c['difference']['rmse_test']:.2f} | "
                     f"{t['selectivity']['rmse_test']:.2f} |")
        out.append(line)
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--no-dedup", action="store_true", help="only the published training set")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    table = os.path.join(RESULTS, "table.md")
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    os.makedirs(RESULTS, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for setting in a.settings:
            for dedup in ((False,) if a.no_dedup else (False, True)):
                run(setting, dedup, force=a.force, tmp_dir=tmp)
    text = report()
    with io.open(table, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print()
    print(text)


if __name__ == "__main__":
    main()
