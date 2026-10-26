"""Three series worked through end to end, with the interpretation tools (plan item F1).

Shapley decompositions, field differences, pharmacophores and pocket complementarity all have
tests, and none of them had a worked example: a reader could see that the code runs but not what
it says about a real series.  This script runs one of three congeneric series from start to
finish and prints what every step found, so that the numbers in the manual's case-study sections
can be reproduced and argued with.

The three series are free-energy benchmark sets that come with the crystal structure their ligands
were posed in, which is what makes the complementarity step possible.  They were chosen by
measuring every set in ``examples/openff`` and ``examples/schindler`` that has a pocket, and
taking the three best models among different target families, so that the interpretation has
something to interpret:

===============  ===  ==============  =============================================
set              n    q2cv (survey)   target
===============  ===  ==============  =============================================
openff/ptp1b      22  +0.66           a phosphatase: a small, charged, polar site
openff/mcl1       25  +0.55           a protein-protein interaction groove
schindler/cdk8    33  +0.51           a kinase, and the widest activity range (4.2)
===============  ===  ==============  =============================================

What it does, in order
----------------------
1. builds a model: the five physico-chemical fields, with the attenuation and the ridge chosen
   by cross-validated grid search **on two thirds of the set**, and the other third -- every
   third molecule by activity rank, so that it spans the range instead of sampling one end --
   held out as an external test.  The model that is then interpreted is refitted on the whole set
   with those same hyper-parameters, because a co-field read from two thirds of a series is a
   co-field of two thirds of a series;
2. checks it the three ways the program offers -- y-randomisation, the applicability domain and a
   conformal interval -- because an interpretation of a model that has not been checked is an
   interpretation of noise;
3. reads the co-fields, masked at two standard deviations (so what is drawn is not the
   measurement error), and turns them into a pharmacophore;
4. decomposes the best and the worst prediction into atom groups with Shapley values;
5. takes the field difference between the most and the least active molecule: what the weak one
   is missing;
6. scores every molecule's complementarity with the crystal pocket, and asks whether it correlates
   with activity at all;
7. writes the model report, the pharmacophore query and a CSV of everything, and prints a summary.

What it does not do
-------------------
It does not tell you whether the pharmacophore is chemically right.  The plan item asks for these
cases to be judged by a medicinal chemist, and that judgement is not in this file: what is here is
every number such a judgement would need, and the places where the program contradicts itself are
pointed out rather than smoothed over.

Usage
-----
    python examples/case_studies.py                       # all three cases
    python examples/case_studies.py --case ptp1b
    python examples/case_studies.py --case mcl1 --out /tmp/mcl1 --no-report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field as _field
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import RDLogger  # noqa: E402

from cmfqsar.cofields import cofield_with_error, make_grid, noise_level  # noqa: E402
from cmfqsar.complementarity import complementarity, pocket_fields  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fielddiff import difference_features  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, grid_search  # noqa: E402
from cmfqsar.pharmacophore import extract, summary, write_csv, write_query  # noqa: E402
from cmfqsar.protein import read_structure  # noqa: E402
from cmfqsar.shapley import AtomGame, shapley  # noqa: E402
from cmfqsar.validation import domain, scramble_kernel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0, 2.0, 3.0)
LAMS = (1e-2, 0.1, 1.0, 10.0)
FOLDS = 10
SEED = 0
OUT = os.path.join(HERE, "case_studies")


@dataclass
class Case:
    key: str
    dataset: str
    pocket: str
    target: str
    why: str


CASES = (
    Case("ptp1b", "examples/openff/ptp1b.sdf", "examples/pockets/openff/ptp1b.pdb",
         "protein tyrosine phosphatase 1B",
         "a small, charged, polar site, where the electrostatic field should carry the model"),
    Case("mcl1", "examples/openff/mcl1.sdf", "examples/pockets/openff/mcl1.pdb",
         "Mcl-1, a Bcl-2 family protein",
         "a protein-protein interaction groove, where shape and lipophilicity should carry it"),
    Case("cdk8", "examples/schindler/cdk8.sdf", "examples/pockets/schindler/cdk8.pdb",
         "cyclin-dependent kinase 8",
         "a kinase hinge, and the widest activity range of the three"),
)


def case_of(key: str) -> Case:
    for c in CASES:
        if c.key == key:
            return c
    raise SystemExit(f"unknown case {key!r}; use one of {[c.key for c in CASES]}")


def split_by_rank(y: np.ndarray, every: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Every `every`-th molecule by activity rank to the test set.

    A random split of twenty molecules can put the whole top of the range in the training set and
    leave a test set that spans one log unit, which makes R2pred a statement about the split.
    """
    order = np.argsort(y)
    test = order[::every]
    train = np.array([i for i in range(len(y)) if i not in set(test.tolist())], int)
    return train, test


def build(case: Case, verbose: bool = True) -> Dict:
    """Step 1 and 2: a model, and the three checks around it."""
    from cmfqsar.model import r2_score, rmse

    ds = load_dataset(os.path.join(ROOT, case.dataset))
    y = np.asarray(ds.activity, float)
    fs = FieldSet.create(physchem=FIELDS, alpha=0.5)
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    table = AtomTable.from_molecules(mf)
    tr, te = split_by_rank(y)
    train_tab = AtomTable.from_molecules([mf[i] for i in tr])
    test_tab = AtomTable.from_molecules([mf[i] for i in te])

    # the hyper-parameters are chosen on the training two thirds only, so that the external
    # third is external in the sense the number claims
    picker = CMFModel(fieldset=fs.copy(), method="krr", normalize=True)
    res = grid_search(train_tab, y[tr], picker, alphas=ALPHAS, lams=LAMS, folds=FOLDS, seed=SEED)
    picker.fit(train_tab, y[tr], cv_folds=0)
    pred = np.asarray(picker.predict(test_tab), float)
    alpha = float(res.best_alpha)
    lam = float(res.best_params.get("lam", float("nan")))

    # and the model that is interpreted is the one fitted on the whole series
    fs_full = fs.copy()
    fs_full.set_alpha(alpha)
    model = CMFModel(fieldset=fs_full, method="krr", normalize=True,
                     params={**res.best_params})
    model.fit(table, y, cv_folds=FOLDS, cv_seed=SEED)
    out = {
        "case": case.key, "target": case.target, "why": case.why,
        "dataset": case.dataset, "pocket": case.pocket,
        "n": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
        "activity_range": float(np.ptp(y)), "activity_sd": float(np.std(y)),
        "alpha": alpha, "lam": lam,
        "q2_cv_train_part": float(res.best_q2), "rmse_cv_train_part": float(res.best_rmse),
        "r2_pred": float(r2_score(y[te], pred)), "rmse_pred": float(rmse(y[te], pred)),
        "q2_cv": float(model.stats.q2) if model.stats is not None else float("nan"),
        "rmse_cv": float(model.stats.rmse_cv) if model.stats is not None else float("nan"),
        "r2_fit": float(model.stats.r2) if model.stats is not None else float("nan"),
    }
    sub, tr_y = table, y
    # the three checks around the q2: is it chance, is the test set inside the domain, and how
    # wide is an interval that covers what it claims to cover
    K_train = model.joint_kernel(model.K_fields_train, model.train.descriptors)
    sc = scramble_kernel(K_train, tr_y, method="krr", params=dict(model.params),
                         folds=FOLDS, cv_seed=SEED, runs=30, seed=SEED)
    out["y_random_mean_q2"] = float(sc.mean)
    out["y_random_max_q2"] = float(np.max(sc.scrambled))
    out["y_random_p"] = float(sc.p_value)
    K_cross = model.test_kernel(test_tab)
    dom = domain(K_cross, model.joint_self(test_tab), K_train)
    out["test_mean_similarity"] = float(np.mean(dom.mean_top))
    out["test_inside_domain"] = float(np.mean(dom.mean_top >= dom.threshold))
    try:
        model.calibrate_conformal(level=0.9)
        _p, lo, hi = model.predict_interval(test_tab)
        out["conformal_width"] = float(np.mean(np.asarray(hi) - np.asarray(lo)))
        out["conformal_covers"] = float(np.mean((y[te] >= np.asarray(lo))
                                                & (y[te] <= np.asarray(hi))))
    except Exception as exc:                            # noqa: BLE001 - not every model has one
        out["conformal_width"] = float("nan")
        out["conformal_note"] = str(exc)[:160]
    if verbose:
        print(f"  model: alpha {out['alpha']:g}, lambda {out['lam']:g}; "
              f"q2cv {out['q2_cv']:+.3f} on all {out['n']}, "
              f"R2pred {out['r2_pred']:+.3f} on the {out['n_test']} held out "
              f"(selected on the other {out['n_train']}, q2cv {out['q2_cv_train_part']:+.3f}); "
              f"y-randomised q2 {out['y_random_mean_q2']:+.3f} mean, "
              f"{out['y_random_max_q2']:+.3f} worst of 30, p {out['y_random_p']:.3f}", flush=True)
    return {"summary": out, "model": model, "fields": mf, "table": table, "y": y,
            "train": tr, "test": te, "fieldset": fs, "names": list(ds.names)}


def fields_that_matter(ctx, verbose: bool = True) -> Dict:
    """Which fields the series needs, by removing each one and refitting.

    A kernel norm cannot answer this: the kernels are scaled to unit mean diagonal before they
    are mixed, so every field looks equally large whatever it contributes.  Removing one and
    measuring what the cross-validated q2 does is the answer, and with the same attenuation and
    ridge as the full model it is one fit per field.
    """
    mf, y = ctx["fields"], ctx["y"]
    full = ctx["model"]
    names = [sp.name for sp in full.fieldset.specs]
    alpha = float(full.fieldset.specs[0].alpha)
    base = float(full.stats.q2) if full.stats is not None else float("nan")
    out = {"all": base, "without": {}}
    for drop in names:
        keep = [n for n in names if n != drop]
        fs = FieldSet.create(physchem=keep, alpha=alpha)
        tab = AtomTable.from_molecules([build_fields(m, fs, n) for m, n
                                        in zip(_molecules(ctx), ctx["names"])])
        m = CMFModel(fieldset=fs, method="krr", normalize=True, params=dict(full.params))
        m.fit(tab, y, cv_folds=FOLDS, cv_seed=SEED)
        q2 = float(m.stats.q2) if m.stats is not None else float("nan")
        out["without"][drop] = {"q2_cv": q2, "drop": base - q2}
    worst = max(out["without"], key=lambda k: out["without"][k]["drop"])
    out["most_needed"] = worst
    if verbose:
        print("  fields, by what removing one costs: "
              + ", ".join(f"{k} {v['drop']:+.3f}" for k, v in out["without"].items()), flush=True)
    return out


def _molecules(ctx):
    """The RDKit molecules of the case, reloaded once for the leave-one-field-out refits."""
    if "mols" not in ctx:
        ds = load_dataset(os.path.join(ROOT, ctx["summary"]["dataset"]))
        ctx["mols"] = list(ds.mols)
        for m, n in zip(ctx["mols"], ctx["names"]):
            m.SetProp("_Name", n)
    return ctx["mols"]


def cofield_story(ctx, verbose=True) -> Dict:
    """Step 3: the co-fields, masked at two standard deviations, and the pharmacophore."""
    model, fs = ctx["model"], ctx["model"].fieldset
    grid = make_grid(model.train.coords, margin=3.0, spacing=0.6)
    sigma, sigma_from = noise_level(model)
    kept = {}
    for f, spec in enumerate(fs.specs):
        ce = cofield_with_error(model, f, grid, sigma=sigma, z=2.0)
        vol, sd = ce.mean, ce.sd
        strong = np.abs(vol) > 2.0 * sd
        # of the grid points the training molecules put any field on at all, how many survive
        covered = sd > 0
        kept[spec.name] = {
            "share_above_2sd": float(np.mean(strong[covered])) if covered.any() else 0.0,
            "peak": float(np.max(np.abs(vol))),
            "signed_peak": float(vol.flat[int(np.argmax(np.abs(vol)))]),
            "peak_z": float(np.max(np.abs(ce.zscore))),
        }
    feats = extract(model, grid=grid, level=0.5, max_features=12, min_z=2.0)
    out = {"noise_level": float(sigma), "noise_from": sigma_from, "per_field": kept,
           "n_features": len(feats),
           "features": [{"field": f.field, "kind": f.kind, "sign": int(f.sign),
                         "importance": float(f.importance), "radius": float(f.radius),
                         "meaning": f.meaning,
                         "centre": [round(float(v), 2) for v in f.centre]} for f in feats]}
    if verbose:
        print(f"  co-fields: noise level {sigma:.3f}, "
              + ", ".join(f"{n} {d['share_above_2sd']:.0%} above 2 sd"
                          for n, d in kept.items()), flush=True)
        print(f"  pharmacophore: {len(feats)} features — "
              + "; ".join(f"{f.kind}{'+' if f.sign > 0 else '-'} w={f.importance:+.2f}"
                          for f in feats[:6]), flush=True)
    return out


def shapley_story(ctx, verbose=True) -> Dict:
    """Step 4: where the best and the worst prediction come from."""
    model, mf, y = ctx["model"], ctx["fields"], ctx["y"]
    te = ctx["test"]
    pred = np.asarray(model.predict(AtomTable.from_molecules([mf[i] for i in te])), float)
    err = np.abs(pred - y[te])
    picks = {"best predicted": int(te[int(np.argmin(err))]),
             "worst predicted": int(te[int(np.argmax(err))]),
             "most active": int(np.argmax(y))}
    out = {}
    for label, idx in picks.items():
        mol = mf[idx]
        groups, labels = _ring_groups(mol)
        game = AtomGame(model, mol, groups=groups, labels=labels)
        r = shapley(game, samples=300, seed=SEED)
        order = np.argsort(-np.abs(r.values))
        out[label] = {
            "molecule": mol.name, "activity": float(y[idx]),
            "prediction": float(r.prediction), "baseline": float(r.baseline),
            "players": int(len(r.values)), "exact": bool(r.exact),
            "efficiency_error": float(r.efficiency_error),
            "top": [{"group": r.labels[int(k)], "value": float(r.values[int(k)]),
                     "share": float(abs(r.values[int(k)]) / max(np.sum(np.abs(r.values)), 1e-12))}
                    for k in order[:5]],
        }
        if verbose:
            top = out[label]["top"]
            print(f"  shapley, {label} ({mol.name}, measured {y[idx]:+.2f}, "
                  f"predicted {r.prediction:+.2f}): "
                  + ", ".join(f"{t['group']} {t['value']:+.2f}" for t in top[:3]), flush=True)
    return out


def _ring_groups(mol) -> Tuple[List[List[int]], List[str]]:
    """Atoms grouped into rings and the chains between them, so the players are chemical parts.

    One player per atom is exact only up to fourteen atoms and is not what a chemist reads; the
    rings of the molecule and the connected pieces left over are.
    """
    from rdkit import Chem
    rd = Chem.MolFromMolBlock(mol.molblock, removeHs=False, sanitize=False) \
        if mol.molblock else None
    if rd is None or rd.GetNumAtoms() != mol.n_atoms:
        step = max(1, mol.n_atoms // 10)
        groups = [list(range(i, min(i + step, mol.n_atoms)))
                  for i in range(0, mol.n_atoms, step)]
        return groups, [f"atoms {g[0] + 1}-{g[-1] + 1}" for g in groups]
    ri = rd.GetRingInfo()
    taken = set()
    groups, labels = [], []
    for k, ring in enumerate(ri.AtomRings()):
        g = [a for a in ring if a not in taken]
        if not g:
            continue
        taken.update(g)
        groups.append(sorted(g))
        labels.append(f"ring {k + 1}")
    rest = [a for a in range(rd.GetNumAtoms()) if a not in taken]
    # the leftover atoms as connected pieces, which are the substituents and the linkers
    seen = set()
    for a in rest:
        if a in seen:
            continue
        stack, piece = [a], []
        while stack:
            b = stack.pop()
            if b in seen or b in taken:
                continue
            seen.add(b)
            piece.append(b)
            for nb in rd.GetAtomWithIdx(b).GetNeighbors():
                if nb.GetIdx() in rest and nb.GetIdx() not in seen:
                    stack.append(nb.GetIdx())
        if piece:
            groups.append(sorted(piece))
            syms = "".join(sorted({rd.GetAtomWithIdx(i).GetSymbol() for i in piece}))
            labels.append(f"{syms}{len(piece)} at {min(piece) + 1}")
    if len(groups) > 14:                 # keep the decomposition exact
        order = sorted(range(len(groups)), key=lambda i: -len(groups[i]))[:14]
        groups = [groups[i] for i in sorted(order)]
        labels = [labels[i] for i in sorted(order)]
    return groups, labels


def difference_story(ctx, verbose=True) -> Dict:
    """Step 5: what the least active molecule is missing, against the most active one."""
    mf, y, fs = ctx["fields"], ctx["y"], ctx["model"].fieldset
    hi, lo = int(np.argmax(y)), int(np.argmin(y))
    feats = difference_features(mf[lo], mf[hi], fs, level=0.5, max_features=10, spacing=0.5)
    out = {"strong": mf[hi].name, "weak": mf[lo].name,
           "gap": float(y[hi] - y[lo]), "n_features": len(feats),
           "features": [{"field": f.field, "kind": f.kind, "sign": int(f.sign),
                         "importance": float(f.importance), "radius": float(f.radius),
                         "meaning": f.meaning} for f in feats[:8]]}
    if verbose:
        print(f"  difference ({mf[lo].name} against {mf[hi].name}, "
              f"{y[hi] - y[lo]:+.2f} log units apart): "
              + ", ".join(f"{f.kind} {abs(f.importance):.2f}" for f in feats[:4]),
              flush=True)
    return out


def pocket_story(ctx, case: Case, verbose=True) -> Dict:
    """Step 6: complementarity with the crystal pocket, and whether it knows anything."""
    from scipy.stats import spearmanr
    mf, y, fs = ctx["fields"], ctx["y"], ctx["model"].fieldset
    prot = read_structure(os.path.join(ROOT, case.pocket), keep_waters=False)
    pocket = pocket_fields(prot, fs)
    rows = [complementarity(m, pocket, fs) for m in mf]
    score = np.array([float(r.score) for r in rows])
    burial = np.array([float(r.burial) for r in rows])
    clash = np.array([float(r.clashes) for r in rows])
    rho = float(spearmanr(score, y).statistic)
    rho_burial = float(spearmanr(burial, y).statistic)
    out = {"pocket_atoms": int(pocket.n_atoms), "residues": int(prot.n_residues),
           "mean_score": float(np.mean(score)), "mean_burial": float(np.mean(burial)),
           "mean_clashes": float(np.mean(clash)),
           "spearman_with_activity": rho, "spearman_burial_activity": rho_burial,
           "terms": {k: float(np.mean([r.terms.get(k, float("nan")) for r in rows]))
                     for k in (rows[0].terms if rows else {})}}
    if verbose:
        print(f"  pocket: {pocket.n_atoms} atoms of {prot.n_residues} residues; "
              f"complementarity {np.mean(score):+.3f} mean, burial {np.mean(burial):.3f}, "
              f"{np.mean(clash):.1f} clashes; Spearman with activity {rho:+.3f} "
              f"(burial alone {rho_burial:+.3f})", flush=True)
    return out


def run_case(case: Case, out_dir: str, want_report: bool = True, verbose: bool = True) -> Dict:
    t0 = time.time()
    if verbose:
        print(f"=== {case.key}: {case.target}\n    {case.why}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    ctx = build(case, verbose)
    result = {"summary": ctx["summary"], "fields": fields_that_matter(ctx, verbose)}
    result["cofields"] = cofield_story(ctx, verbose)
    result["shapley"] = shapley_story(ctx, verbose)
    result["difference"] = difference_story(ctx, verbose)
    result["pocket"] = pocket_story(ctx, case, verbose)

    feats = extract(ctx["model"], level=0.5, max_features=12, min_z=2.0)
    write_csv(os.path.join(out_dir, f"{case.key}_pharmacophore.csv"), feats)
    try:
        write_query(os.path.join(out_dir, f"{case.key}_pharmacophore.json"), feats, enable_top=4)
    except Exception as exc:                            # noqa: BLE001
        result["query_note"] = str(exc)[:120]
    if want_report:
        from cmfqsar.report import ReportSpec, write_report
        path = os.path.join(out_dir, f"{case.key}_report.html")
        write_report(ctx["model"], path, ReportSpec(title=f"{case.target} ({case.key})"))
        result["report"] = os.path.relpath(path, ROOT).replace("\\", "/")
    result["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, f"{case.key}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    if verbose:
        print(f"    written {out_dir} ({result['seconds']:.0f} s)", flush=True)
    return result


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                        # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs="*", default=[c.key for c in CASES],
                    choices=[c.key for c in CASES])
    ap.add_argument("--out", default=OUT, help="where to write the artefacts")
    ap.add_argument("--no-report", action="store_true", help="skip the HTML report, which is slow")
    a = ap.parse_args()
    for key in a.case:
        run_case(case_of(key), os.path.join(a.out, key), want_report=not a.no_report)


if __name__ == "__main__":
    main()
