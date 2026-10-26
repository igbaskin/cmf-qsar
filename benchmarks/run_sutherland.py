#!/usr/bin/env python
"""
Benchmark of the cmfqsar implementation against Table 2 of

    Baskin I.I., Zhokhova N.I., J. Comput.-Aided Mol. Des. 27 (2013) 427-442

on the eight Sutherland et al. (2004) 3D-QSAR data sets (ACE, AChE, BZR, COX-2, DHFR, GPB,
THER, THR; aligned MOL2 structures with MNDO ESP (THER: Gasteiger) charges, published
train/test split).  The data sets are *not* part of this repository: set ``--data-root`` to
a folder containing ``<set>_data/<prefix>-all.mol2`` and ``<set>_data/<set>-data-sybyl-all.txt``
as distributed with the supplementary R scripts of the paper.

Every run is a ``cmf_cli.py build`` invocation; its JSON output is stored in
``benchmarks/results`` and ``--report`` turns the collected results into a Markdown table.

Modes
-----
paper    exact protocol of the 2013 paper: Sybyl-type increments of the original R scripts
         including their hydrogen-counting bug (--param tripos-legacy), MOL2 charges, raw
         (unscaled) kernels with h_f = 1, interleaved unshuffled 10-fold CV, alpha grid of the
         scripts, lambda minimised continuously in [0.01, 20] (Brent), external 5-fold CV on all
         molecules.  ACE is additionally run without the bias term (as in the paper).
tripos   as "paper" but with correctly counted hydrogens (--param tripos)
rdkit    as "paper" but with the RDKit-based field parameterisation (Crippen logP, built-in
         H-bond increments, element-based vdW parameters)
rdkit-scaled  as "rdkit" but with every field kernel scaled to unit mean diagonal before summation
         (the program's default convention instead of the raw h_f = 1 sum of the paper)
default  the program's default settings (RDKit parameterisation, field kernels scaled to unit
         mean diagonal, shuffled folds, lambda grid), MOL2 charges

Continuous indicator fields (Sitnikov et al., JCAMD 29 (2015) 233-247)
------------------------------------------------------------------------
``--cif-root`` points at the folder of the original CMF R scripts that holds the three data
sets of the 2015 paper (``ligands-all.mol2`` + ``activity-all.txt``: 88 thrombin inhibitors,
``AmEu/_AmEu1-all.mol2`` + ``_AmEu1-all.txt``: 46 Am3+/Eu3+ ligands, ``gleb/aff12_aligned_babel-p5.mol2``
+ ``gleb/aff.txt``: 30 azo dyes).  Activity tables with the published train/test splits are
derived into ``benchmarks/results/cif_data`` and the CIF runs (``cif*`` / ``pc*`` tags, see
``CIF_RUNS``) go to ``benchmarks/results/cif``; ``--cif-report`` writes their comparison table.
The protocol of the 2015 paper is: one indicator field per Sybyl atom type occurring in at least
five training molecules, h_t = 1, raw kernel sum, alpha and lambda optimised simultaneously with
R's Nelder-Mead simplex from (0.25, 5) on kernels interpolated between the alpha grid points,
10-fold interleaved CV, external 5-fold CV repeated 100 times with reshuffling and evaluated on
the mean prediction.

Usage
-----
    python benchmarks/run_sutherland.py --data-root <folder> [--modes paper,tripos,rdkit,default] [--sets ACE,THR]
    python benchmarks/run_sutherland.py --report benchmarks/results/table.md
    python benchmarks/run_sutherland.py --cif-root <folder of the R scripts> [--sets THR,AmEu,DYES] [--tags cif,cif-grid]
    python benchmarks/run_sutherland.py --cif-report benchmarks/results/cif/table.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cmf_cli  # noqa: E402

RESULTS = os.path.join(HERE, "results")

SETS = {  # name: (folder, file prefix)
    "ACE": ("ace_data", "ace-3dqsar-mndo"),
    "AChE": ("ache_data", "ache-3dqsar-mndo"),
    "BZR": ("bzr_data", "bzr-3dqsar-mndo"),
    "COX2": ("cox2_data", "cox2-3dqsar-mndo"),
    "DHFR": ("dhfr_data", "dhfr-3dqsar-mndo"),
    "GPB": ("gpb_data", "gpb-3dqsar-mndo"),
    "THER": ("ther_data", "ther-3dqsar-gasteiger"),
    "THR": ("thr_data", "thr-3dqsar-mndo"),
}

# Table 2 of the 2013 paper, CMF column: q2, RMSEcv, R2pred, RMSEpred, q2ex, RMSEcvex
PAPER = {
    "ACE": (0.72, 1.24, 0.65, 1.24, 0.67, 1.31),
    "AChE": (0.58, 0.79, 0.64, 0.77, 0.54, 0.84),
    "BZR": (0.40, 0.51, 0.20, 0.79, 0.27, 0.65),
    "COX2": (0.57, 0.67, 0.14, 1.23, 0.41, 0.89),
    "DHFR": (0.67, 0.73, 0.65, 0.80, 0.67, 0.75),
    "GPB": (0.69, 0.60, 0.51, 0.84, 0.59, 0.71),
    "THER": (0.60, 1.18, 0.31, 1.86, 0.40, 1.56),
    "THR": (0.73, 0.50, 0.63, 0.66, 0.71, 0.54),
}
STATS = ("q2", "rmse_cv", "r2_pred", "rmse_pred", "q2_ex", "rmse_ex")
STAT_LABELS = ("q2", "RMSEcv", "R2pred", "RMSEpred", "q2ex", "RMSEcvex")

R_ALPHAS = "0.01,0.05,0.1,0.25,0.5,0.75,1,2,3,5,10"
PAPER_PROTOCOL = ["--charges", "file", "--raw-scale", "--cv-no-shuffle", "--lam-search", "brent",
                  "--alphas", R_ALPHAS, "--lams", "0.01,20", "--nested", "--nested-on", "all"]
MODES = {
    "paper": ["--param", "tripos-legacy"] + PAPER_PROTOCOL,
    "tripos": ["--param", "tripos"] + PAPER_PROTOCOL,
    "rdkit": ["--param", "rdkit"] + PAPER_PROTOCOL,
    "rdkit-scaled": ["--param", "rdkit"] + [a for a in PAPER_PROTOCOL if a != "--raw-scale"],
    "default": ["--param", "rdkit", "--charges", "file", "--nested", "--nested-on", "all"],
}
MODE_LABELS = {
    "paper": "this work, paper protocol",
    "paper-nobias": "this work, paper protocol, b = 0",
    "tripos": "this work, corrected H counts",
    "rdkit": "this work, RDKit parameterisation",
    "rdkit-scaled": "this work, RDKit param., scaled field kernels",
    "default": "this work, program defaults",
}


def run_cli(mol2: str, act: str, argv_mode, out: str, label: str, force: bool = False) -> str:
    if os.path.exists(out) and not force:
        print(f"[skip] {out} exists")
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    argv = ["build", mol2, "--activity-file", act] + list(argv_mode) + ["--json", out]
    print(f"\n=== {label}: cmf_cli.py {' '.join(argv)}", flush=True)
    t0 = time.time()
    cmf_cli.main(argv)
    print(f"=== {label} finished in {time.time() - t0:.0f} s", flush=True)
    return out


def run_one(data_root: str, name: str, mode: str, extra=(), tag: str | None = None, force: bool = False) -> str:
    folder, prefix = SETS[name]
    mol2 = os.path.join(data_root, folder, f"{prefix}-all.mol2")
    act = os.path.join(data_root, folder, f"{folder.split('_')[0]}-data-sybyl-all.txt")
    tag = tag or mode
    out = os.path.join(RESULTS, f"{name}_{tag}.json")
    return run_cli(mol2, act, MODES[mode] + list(extra), out, f"{name} / {tag}", force)


# --------------------------------------------------------------------------- #
# Continuous indicator fields: the three data sets of Sitnikov et al. (2015)
# --------------------------------------------------------------------------- #

RESULTS_CIF = os.path.join(RESULTS, "cif")
CIF_DATA = os.path.join(RESULTS, "cif_data")
CIF_SETS = {  # name: (MOL2 relative to --cif-root, derived activity table)
    "THR": ("ligands-all.mol2", "thrombin-activity.txt"),
    "AmEu": ("AmEu/_AmEu1-all.mol2", "ameu-activity.txt"),
    "DYES": ("gleb/aff12_aligned_babel-p5.mol2", "dyes-activity.txt"),
}
# Test-set molecules of the Am3+/Eu3+ set (0-based positions in _AmEu1-all.mol2), identified by
# matching the coordinates of _AmEu1-test.mol2; the thrombin test set is CPD_73 ... CPD_88.
AMEU_TEST = {2, 7, 8, 17, 21, 24, 33, 38, 41, 43}

# Tables 1-3 of the 2015 paper: r2, RMSE(fit), q2, RMSEcv, R2ext (test variance), R2ext (training
# variance), RMSEext, q2ext, RMSEext-cv
PAPER_CIF = {
    ("THR", "paper, CIF/CMF (Table 1)"): (0.956, 0.21, 0.772, 0.49, 0.311, 0.431, 0.75, 0.712, 0.54),
    ("THR", "paper, PC/CMF (Table 1)"): (0.929, 0.27, 0.747, 0.52, 0.280, 0.404, 0.77, 0.764, 0.49),
    ("AmEu", "paper (Table 2, text)"): (0.906, None, 0.707, 0.24, 0.745, 0.767, 0.20, 0.668, 0.25),
    ("DYES", "paper, all atom types (Table 3)"): (1.000, None, 0.956, 0.69, None, None, None, 0.909, 0.99),
    ("DYES", "paper, hydrogens only (Table 3)"): (0.999, None, 0.940, 0.80, None, None, None, 0.822, 1.39),
}
CIF_STATS = ("r2", "rmse_fit", "q2", "rmse_cv", "r2_pred", "r2_pred_train_var", "rmse_pred", "q2_ex_aggr",
             "rmse_ex_aggr")
CIF_STAT_LABELS = ("r2", "RMSE", "q2", "RMSEcv", "R2ext", "R2ext(tr)", "RMSEext", "q2ext", "RMSEext-cv")
CIF_ND = (3, 2, 3, 2, 3, 3, 2, 3, 2)   # printed decimals, as in the paper's tables

CIF_COMMON = ["--raw-scale", "--cv-no-shuffle", "--alphas", R_ALPHAS, "--nested", "--nested-on", "all",
              "--outer-folds", "5", "--repeats", "100", "--seed", "0"]
CIF_FIELDS = ["--fields", ",", "--cif", "--cif-min", "5"]
NM = ["--alpha-search", "nelder-mead"]
GRID = ["--lam-search", "brent", "--lams", "0.01,20"]
PC_LEGACY = ["--param", "tripos-legacy", "--charges", "file"]
CIF_RUNS = [  # (set, tag, cmf_cli arguments)
    ("THR", "cif", CIF_FIELDS + NM + CIF_COMMON),
    ("THR", "cif-grid", CIF_FIELDS + GRID + CIF_COMMON),
    ("THR", "pc", PC_LEGACY + NM + CIF_COMMON),
    ("THR", "pc-grid", PC_LEGACY + GRID + CIF_COMMON),
    ("AmEu", "cif", CIF_FIELDS + NM + CIF_COMMON),
    ("AmEu", "cif-nobias", CIF_FIELDS + NM + CIF_COMMON + ["--no-bias"]),
    ("AmEu", "cif-grid", CIF_FIELDS + GRID + CIF_COMMON),
    ("DYES", "cif", CIF_FIELDS + NM + CIF_COMMON),
    ("DYES", "cif-nm11", CIF_FIELDS + NM + CIF_COMMON + ["--nm-start", "1,1"]),
    ("DYES", "cif-h", CIF_FIELDS + NM + CIF_COMMON + ["--cif-types", "H"]),
    ("DYES", "cif-h-nm11", CIF_FIELDS + NM + CIF_COMMON + ["--cif-types", "H", "--nm-start", "1,1"]),
    ("DYES", "cif-grid", CIF_FIELDS + GRID + CIF_COMMON),
]
CIF_LABELS = {
    "cif": "this work, paper protocol (Nelder-Mead from (0.25, 5))",
    "cif-nobias": "this work, paper protocol, b = 0",
    "cif-grid": "this work, alpha grid + Brent lambda (2013 protocol)",
    "cif-nm11": "this work, paper protocol, Nelder-Mead from (1, 1)",
    "cif-h": "this work, hydrogens only, paper protocol",
    "cif-h-nm11": "this work, hydrogens only, Nelder-Mead from (1, 1)",
    "pc": "this work, physico-chemical fields, Nelder-Mead",
    "pc-grid": "this work, physico-chemical fields, alpha grid + Brent lambda",
}


def _lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def prepare_cif_data(cif_root: str) -> None:
    """Derive 'name activity set' tables with the published splits from the original files."""
    os.makedirs(CIF_DATA, exist_ok=True)
    rows = [ln.split(",") for ln in _lines(os.path.join(cif_root, "activity-all.txt"))[1:] if "," in ln]
    with open(os.path.join(CIF_DATA, "thrombin-activity.txt"), "w", encoding="utf-8") as fh:
        fh.write("NAME\tACTIVITY\tSET\n")
        for r in rows:
            name = r[0].strip().strip('"')
            fh.write(f"{name}\t{r[1].strip()}\t{1 if int(name.split('_')[1]) <= 72 else 2}\n")
    vals = [ln.strip() for ln in _lines(os.path.join(cif_root, "AmEu/_AmEu1-all.txt"))[1:] if ln.strip()]
    with open(os.path.join(CIF_DATA, "ameu-activity.txt"), "w", encoding="utf-8") as fh:
        fh.write("NAME\tACTIVITY\tSET\n")
        for i, v in enumerate(vals):
            fh.write(f"L{i + 1:02d}\t{v}\t{2 if i in AMEU_TEST else 1}\n")
    vals = [ln.split("\t")[0].strip() for ln in _lines(os.path.join(cif_root, "gleb/aff.txt"))[1:] if ln.strip()]
    with open(os.path.join(CIF_DATA, "dyes-activity.txt"), "w", encoding="utf-8") as fh:
        fh.write("NAME\tACTIVITY\n")
        for i, v in enumerate(vals):
            fh.write(f"D{i + 1:02d}\t{v}\n")


def run_cif(cif_root: str, names, tags=None, force: bool = False) -> None:
    prepare_cif_data(cif_root)
    for name, tag, argv in CIF_RUNS:
        if name not in names or (tags and tag not in tags):
            continue
        mol2 = os.path.join(cif_root, CIF_SETS[name][0])
        act = os.path.join(CIF_DATA, CIF_SETS[name][1])
        run_cli(mol2, act, argv, os.path.join(RESULTS_CIF, f"{name}_{tag}.json"), f"{name} / {tag}", force)


def report_cif(path: str) -> str:
    res = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_CIF, "*.json"))):
        name, tag = os.path.basename(f)[:-5].split("_", 1)
        with open(f) as fh:
            res[(name, tag)] = json.load(fh)
    lines = ["| Set | Model | " + " | ".join(CIF_STAT_LABELS) + " | alpha | lambda |",
             "|---|---|" + "---|" * (len(CIF_STATS) + 2)]
    for name in CIF_SETS:
        first = True
        for (n, label), vals in PAPER_CIF.items():
            if n == name:
                lines.append(f"| {'**' + name + '**' if first else ''} | {label} | "
                             + " | ".join(fmt(v, nd) for v, nd in zip(vals, CIF_ND)) + " | | |")
                first = False
        for tag in [t for n, t, _ in CIF_RUNS if n == name]:
            r = res.get((name, tag))
            if r is None:
                continue
            vals = [r.get(k) for k in CIF_STATS]
            lam = r.get("best_params", {}).get("lam")
            lines.append(f"| | {CIF_LABELS.get(tag, tag)} | " + " | ".join(fmt(v, nd) for v, nd in zip(vals, CIF_ND))
                         + f" | {fmt(r.get('best_alpha'), 3)} | {fmt(lam, 3)} |")
    text = "\n".join(lines) + "\n"
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def load_results():
    res = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        base = os.path.basename(f)[:-5]
        name, tag = base.split("_", 1)
        with open(f) as fh:
            res[(name, tag)] = json.load(fh)
    return res


def fmt(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return f"{v:.{nd}f}" if v == v else "-"


def report(path: str, tags=("paper", "paper-nobias", "tripos", "rdkit", "rdkit-scaled", "default")) -> str:
    res = load_results()
    lines = []
    lines.append("| Set | Model | q2 | RMSEcv | R2pred | RMSEpred | q2ex | RMSEcvex | alpha | lambda |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in SETS:
        p = PAPER[name]
        lines.append(f"| **{name}** | paper (Table 2) | " + " | ".join(fmt(v) for v in p) + " | | |")
        for tag in tags:
            r = res.get((name, tag))
            if r is None:
                continue
            vals = [r.get(k) for k in STATS]
            lam = r.get("best_params", {}).get("lam")
            lines.append(f"| | {MODE_LABELS.get(tag, tag)} | " + " | ".join(fmt(v) for v in vals)
                         + f" | {fmt(r.get('best_alpha'), 2)} | {fmt(lam, 3)} |")
    # mean absolute deviation from the paper per tag
    lines.append("")
    lines.append("Mean absolute deviation from the paper over the eight sets:")
    lines.append("")
    lines.append("| Model | " + " | ".join(STAT_LABELS) + " |")
    lines.append("|---|" + "---|" * len(STATS))
    for tag in tags:
        devs = {k: [] for k in STATS}
        for name in SETS:
            r = res.get((name, tag))
            if r is None:
                continue
            for k, pv in zip(STATS, PAPER[name]):
                v = r.get(k)
                if v is not None and v == v:
                    devs[k].append(abs(float(v) - pv))
        if any(devs.values()):
            lines.append(f"| {MODE_LABELS.get(tag, tag)} | " + " | ".join(
                fmt(sum(d) / len(d), 3) if d else "-" for d in devs.values()) + " |")
    text = "\n".join(lines) + "\n"
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="folder with the <set>_data sub-folders")
    ap.add_argument("--modes", default="paper,tripos,rdkit,default")
    ap.add_argument("--sets", default=None, help="comma-separated data sets (default: all of the benchmark)")
    ap.add_argument("--force", action="store_true", help="recompute existing results")
    ap.add_argument("--report", default=None, help="write the Markdown comparison table to this file")
    ap.add_argument("--cif-root", default=None,
                    help="folder of the original R scripts with the three data sets of the 2015 CIF paper")
    ap.add_argument("--tags", default=None, help="comma-separated CIF run tags to run (default: all)")
    ap.add_argument("--cif-report", default=None, help="write the Markdown table of the CIF benchmark to this file")
    args = ap.parse_args(argv)
    if args.cif_root:
        names = [s for s in args.sets.split(",") if s in CIF_SETS] if args.sets else list(CIF_SETS)
        t0 = time.time()
        run_cif(args.cif_root, names, tags=[t for t in args.tags.split(",") if t] if args.tags else None,
                force=args.force)
        print(f"\nCIF runs finished in {(time.time() - t0) / 60:.1f} min")
    if args.cif_report:
        print(report_cif(args.cif_report))
    if args.data_root:
        modes = [m for m in args.modes.split(",") if m]
        names = [s for s in args.sets.split(",") if s in SETS] if args.sets else list(SETS)
        t0 = time.time()
        for mode in modes:
            for name in names:
                run_one(args.data_root, name, mode, force=args.force)
                if mode == "paper" and name == "ACE":
                    run_one(args.data_root, name, mode, extra=["--no-bias"], tag="paper-nobias", force=args.force)
        print(f"\nAll runs finished in {(time.time() - t0) / 60:.1f} min")
    if args.report or not (args.data_root or args.cif_root or args.cif_report):
        print(report(args.report))


if __name__ == "__main__":
    main()
