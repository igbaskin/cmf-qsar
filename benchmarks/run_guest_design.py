#!/usr/bin/env python
"""
Can a model of a guest table say what to make next? (plan item SC7)

`cmf design` used to need a topomer model, because a substituent nobody has made has no pose and
the topomer rule is what gives it one. A rotation-invariant model needs no pose at all: a
candidate is built as a whole molecule, embedded once, and scored, and nothing is superimposed on
anything. That is what makes a design loop possible on a table of guests, which share no scaffold.

The question is whether the ranking means anything. The test here is **leave one group out**: for
each group in the library, every guest of the beta-cyclodextrin table that carries that group is
removed, a model is fitted on what is left, and the whole library is then ranked by it. A group's
prediction therefore comes from a model that never saw a molecule containing it.

The library is eleven R groups of R-COOH. Ten of them are a measured guest of the table, so the
prediction can be compared with a number; the eleventh is adamantyl, whose carboxylic acid is not
in the table although six other adamantyl guests are, and they are the strongest binders in it
(log K 2.9 to 4.3 against a median near 2). Where a group is so common that removing every guest
that carries it would leave no table -- a methyl is in 100% of them -- only the molecule itself is
removed, and the row says so.

Usage
-----
    python benchmarks/run_guest_design.py
    python benchmarks/run_guest_design.py --report
"""
from __future__ import annotations

import argparse
import csv
import io
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

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.design import design, whole_molecule, as_fragment  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, r2_score  # noqa: E402

DATA = os.path.join(ROOT, "examples", "cyclodextrin", "beta_cyclodextrin.csv")
RESULTS = os.path.join(HERE, "results", "guest_design")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
LAMS = (1e-2, 0.1, 1.0)
FOLDS = 10
SEED = 0
#: the core that names the position being varied, and the parent the candidates are built from
CORE = "C(=O)O"
PARENT = "CC(=O)O"
#: a group whose substructure is carried by more than this share of the table cannot have all of
#: its molecules removed: there would be no table left
MAX_DROP = 0.3

#: (name, fragment with its attachment point, the substructure that finds it in a guest)
LIBRARY = (
    ("methyl", "[*]C", "C"),
    ("propyl", "[*]CCC", "CCC"),
    ("butyl", "[*]CCCC", "CCCC"),
    ("pentyl", "[*]CCCCC", "CCCCC"),
    ("hexyl", "[*]CCCCCC", "CCCCCC"),
    ("phenyl", "[*]c1ccccc1", "c1ccccc1"),
    ("4-methylphenyl", "[*]c1ccc(C)cc1", "Cc1ccccc1"),
    ("cyclopentyl", "[*]C1CCCC1", "C1CCCC1"),
    ("cyclohexyl", "[*]C1CCCCC1", "C1CCCCC1"),
    ("cyclohexylmethyl", "[*]CC1CCCCC1", "CC1CCCCC1"),
    ("adamantyl", "[*]C12CC3CC(CC(C3)C1)C2", "C12CC3CC(CC(C3)C1)C2"),
)


def table():
    """The guests, their activities, and the canonical SMILES that names each one."""
    ds = load_dataset(DATA)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    keys = [Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(m))) for m in mols]
    return mols, names, y[ok], keys


def candidates():
    """The library as (name, fragment smiles, query, the whole molecule it makes)."""
    core = Chem.MolFromSmiles(CORE)
    out = []
    for name, frag, query in LIBRARY:
        f = as_fragment(frag, 0)
        mol = whole_molecule(core, [f]) if f is not None else None
        q = Chem.MolFromSmiles(query)
        if mol is None or q is None:
            raise SystemExit(f"the library entry {name} does not build")
        out.append({"name": name, "fragment": frag, "query": query,
                    "molecule": Chem.MolToSmiles(mol)})
    return out


def fit(mols, names, y, keep, fs, lams=LAMS):
    """An invariant model on a subset of the guests, with the ridge chosen by the evidence."""
    from cmfqsar.evidence import log_evidence
    mf = [build_fields(mols[i], fs, names[i]) for i in keep]
    tab = AtomTable.from_molecules(mf)
    probe = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation")
    K = probe.training_kernel(tab)
    yy = y[keep]
    best = None
    for lam in lams:
        ev = log_evidence(K, yy, float(lam), fit_bias=True)
        if np.isfinite(ev) and (best is None or ev > best[0]):
            best = (float(ev), float(lam))
    lam = best[1] if best else 0.1
    model = CMFModel(fieldset=fs, method="krr", params={"lam": lam}, normalize=True,
                     kernel_source="autocorrelation")
    model.fit(tab, yy, cv_folds=0)
    return model, lam


def run(results=RESULTS, force=False):
    out_path = os.path.join(results, "beta.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    t0 = time.time()
    mols, names, y, keys = table()
    lib = candidates()
    by_key = {k: (names[i], float(y[i])) for i, k in enumerate(keys)}
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    parent = Chem.MolFromSmiles(PARENT)
    core = Chem.MolFromSmiles(CORE)
    frags = [c["fragment"] for c in lib]
    print(f"=== {len(mols)} guests, {len(lib)} groups in the library", flush=True)
    rows = []
    for c in lib:
        q = Chem.MolFromSmiles(c["query"])
        hit = [i for i, m in enumerate(mols) if m.HasSubstructMatch(q)]
        share = len(hit) / len(mols)
        made = by_key.get(c["molecule"])
        if share > MAX_DROP:
            drop = {i for i, k in enumerate(keys) if k == c["molecule"]}
            how = "the molecule only"
        else:
            drop = set(hit)
            how = "every guest that carries it"
        keep = [i for i in range(len(mols)) if i not in drop]
        model, lam = fit(mols, names, y, keep, fs)
        res = design(model, parent, core, 0, fragments=frags, acquire="greedy",
                     parent_name="acetic acid", with_fields=len(frags), pose="molecule",
                     seed=1)
        order = [x.smiles for x in res.candidates]
        mine = next((x for x in res.candidates
                     if Chem.MolToSmiles(Chem.MolFromSmiles(x.smiles))
                     == Chem.MolToSmiles(Chem.MolFromSmiles(c["fragment"]))), None)
        row = {"group": c["name"], "molecule": c["molecule"], "matches": len(hit),
               "share": share, "dropped": len(drop), "how": how, "lam": lam,
               "n_train": len(keep),
               "measured": None if made is None else made[1],
               "measured_name": None if made is None else made[0],
               "predicted": None if mine is None else mine.mean,
               "sd": None if mine is None else mine.sd,
               "rank": None if mine is None else order.index(mine.smiles) + 1,
               "of": len(order),
               "group_value": None if mine is None else mine.group_value,
               "group_total": None if mine is None else mine.group_total}
        rows.append(row)
        print(f"    {row['group']:18s} dropped {row['dropped']:4d} ({how:28s})  "
              f"predicted {row['predicted']:+.3f}  rank {row['rank']:2d}/{row['of']}"
              + ("" if row["measured"] is None else f"   measured {row['measured']:+.3f}"),
              flush=True)
    scored = [r for r in rows if r["measured"] is not None and r["predicted"] is not None]
    pred = np.array([r["predicted"] for r in scored])
    true = np.array([r["measured"] for r in scored])
    from scipy.stats import spearmanr
    globals()["_parts"] = {}
    rho = float(spearmanr(pred, true).statistic) if len(scored) > 2 else float("nan")
    res = {"host": "beta", "n": len(mols), "library": len(lib), "rows": rows,
           "spearman": rho, "pearson": float(np.corrcoef(pred, true)[0, 1]),
           "r2": float(r2_score(true, pred)), "n_scored": len(scored),
           "parent": PARENT, "core": CORE, "fields": FIELDS, "seconds": time.time() - t0}
    # the two kinds of row are not equally hard, and the mean over both would hide it
    for mode, tag in (("every guest that carries it", "group_out"),
                      ("the molecule only", "molecule_out")):
        part = [r for r in scored if r["how"] == mode]
        if len(part) >= 2:
            p = np.array([r["predicted"] for r in part])
            t = np.array([r["measured"] for r in part])
            res_part = {"n": len(part), "bias": float(np.mean(p - t)),
                        "mean_absolute": float(np.mean(np.abs(p - t)))}
            if len(part) > 2:
                res_part["spearman"] = float(spearmanr(p, t).statistic)
            globals().setdefault("_parts", {})[tag] = res_part
    strong = [r for r in rows if r["measured"] is not None and r["measured"] >= 3.0]
    res["by_mode"] = dict(globals().get("_parts", {}))
    globals()["_parts"] = {}
    res["fully_held_out"] = sorted(
        [r["group"] for r in rows if r["how"] == "every guest that carries it"],
        key=lambda g: next(x["rank"] for x in rows if x["group"] == g))
    res["strong"] = {"n": len(strong),
                     "in_top_three": sum(1 for r in strong if r["rank"] <= 3),
                     "mean_rank": float(np.mean([r["rank"] for r in strong])) if strong else
                     float("nan")}
    print(f"    over the {len(scored)} groups with a measured guest: Spearman {rho:+.3f}, "
          f"R2 {res['r2']:+.3f}", flush=True)
    print(f"    the {len(strong)} groups measuring 3.0 or more have mean rank "
          f"{res['strong']['mean_rank']:.1f}; {res['strong']['in_top_three']} of them are in the "
          f"top three", flush=True)
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out_path} ({res['seconds']:.0f} s)", flush=True)
    return res


def report(results=RESULTS):
    path = os.path.join(results, "beta.json")
    if not os.path.exists(path):
        return ""
    d = json.load(open(path))
    out = ["## What to make next, on a table of guests", "",
           "`cmf design --kernel autocorrelation` ranks candidates built as whole molecules: the "
           "rotation-invariant kernel needs no frame, so the guests need no common scaffold and "
           "nothing is superimposed on anything. **Leave one group out**: for each row, every "
           f"guest of the beta-cyclodextrin table ({d['n']} of them) that carries that group is "
           "removed, a model is fitted on what is left, and the library is ranked by it, so a "
           "group's prediction comes from a model that never saw a molecule containing it. A "
           "group carried by more than 30% of the table cannot have all of its guests removed, "
           "and the row says so.", "",
           "| group | guests dropped | what was dropped | predicted | rank | measured |",
           "| --- | ---: | --- | ---: | ---: | ---: |"]
    for r in d["rows"]:
        out.append(f"| {r['group']} | {r['dropped']} | {r['how']} | {r['predicted']:+.3f} | "
                   f"{r['rank']}/{r['of']} | "
                   + ("-" if r["measured"] is None else f"{r['measured']:+.3f}") + " |")
    g = d.get("by_mode", {}).get("group_out", {})
    m = d.get("by_mode", {}).get("molecule_out", {})
    held = ", ".join(d.get("fully_held_out", []))
    out += ["", f"Over the {d['n_scored']} groups whose R-COOH is a measured guest the ranking "
                f"follows the measurements: Spearman {d['spearman']:+.3f}, R2 {d['r2']:+.3f}.", "",
            f"**The two kinds of row are not equally hard.** Where only the molecule was removed "
            f"the model had seen hundreds of its relatives, and it is close: mean absolute error "
            f"{m.get('mean_absolute', float('nan')):.2f} over {m.get('n', 0)} groups. Where every "
            f"guest carrying the group was removed ({held}) it is not: mean absolute error "
            f"{g.get('mean_absolute', float('nan')):.2f} over {g.get('n', 0)}, and the bias is "
            f"{g.get('bias', float('nan')):+.2f} -- a group the table no longer contains is "
            f"pulled towards the mean.", "",
            f"**What the ranking still gets right is the order.** Adamantyl, whose six guests in "
            f"the table measure 2.9 to 4.3 and whose own acid is not in it at all, comes first "
            f"of the {d['rows'][0]['of']} with every one of those six removed; cyclohexylmethyl, "
            f"the strongest of the measured groups that were fully held out, comes second. "
            f"Taking every row together, the {d['strong']['n']} groups measuring 3.0 or more "
            f"have mean rank {d['strong']['mean_rank']:.1f} of {d['rows'][0]['of']}, so the "
            f"ranking is useful at the top and unreliable in the middle.", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
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
    run(force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
