#!/usr/bin/env python
"""
One entry point for the benchmarks: what there is, rebuild every table, record what it rests on.

Every benchmark script under ``benchmarks/`` keeps its measurements as JSON under
``benchmarks/results/<name>/`` and can rebuild its Markdown table from them.  This script knows
them all and does four things:

    python benchmarks/run_all.py                 # --list: every benchmark, its inputs and its state
    python benchmarks/run_all.py --tables        # rebuild every table.md from the result files,
                                                 # and splice the tables the documents carry
    python benchmarks/run_all.py --manifest      # benchmarks/manifest.json: versions,
                                                 # platform, commit, and the hash of every data
                                                 # file and result file
    python benchmarks/run_all.py --check         # --tables, then fail if any table or document
                                                 # differs from what is committed, or any data
                                                 # file differs from the manifest
    python benchmarks/run_all.py --compute NAME [NAME ...] | all   [--data-root D] [--cif-root C] [--orca]

``--compute`` runs the scripts themselves, in the order listed, each as a subprocess.  Several
need what is not in the repository: the Sutherland MOL2 originals (``--data-root``), the data of
the 2015 CIF paper (``--cif-root``), or an ORCA installation for the quantum stages (``--orca``
says it is there).  A benchmark whose inputs are missing is skipped with a line that says so; the
tables are rebuilt afterwards from whatever results exist.  The kernels every script computes
are cached by ``kernel_cache.py``, so a second run of a script that builds a set another already
built pays nothing for the kernels.

The tables in the documents.  ``docs/benchmark.md`` carries two tables between markers
(``<!-- TABLE -->`` and ``<!-- CIF-TABLE -->``) that are the Sutherland and CIF result tables
verbatim, and the README carries the two-endpoint table between ``<!-- MULTITASK-TABLE -->``;
``--tables`` rewrites those three in place.  The other result tables of the README are written
by hand from the result files named in the registry below, and ``--list`` says which; the
manifest records the hashes of those files, so that a changed result is told apart from a
changed sentence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
MANIFEST = os.path.join(HERE, "manifest.json")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

PACKAGES = ("cmfqsar", "numpy", "scipy", "scikit-learn", "scikit-image", "rdkit", "joblib",
            "PySide6", "pyqtgraph", "PyOpenGL", "reportlab", "matplotlib", "pytest")


@dataclass
class Benchmark:
    name: str
    what: str
    script: str                                  # file under benchmarks/
    results: str                                 # folder under benchmarks/results/
    compute: List[List[str]] = field(default_factory=list)   # argument lists, run in order
    needs: Optional[str] = None                  # "data-root" | "cif-root" | "orca" | None
    minutes: str = "?"                           # what a full run costs on the author's machine
    report: Optional[Callable[[], None]] = None  # rebuilds results/<name>/table.md
    tables: List[str] = field(default_factory=list)          # files --tables writes
    readme: List[str] = field(default_factory=list)          # README sections written by hand from it


def _mod(name):
    return importlib.import_module(name)


def _r(path):
    return os.path.join(RESULTS, path)


def registry() -> List[Benchmark]:
    return [
        Benchmark("sutherland", "Table 2 of the 2013 paper on the eight Sutherland sets, and the "
                  "continuous-indicator-field models of the 2015 paper",
                  "run_sutherland.py", "",
                  compute=[["--data-root", "{data-root}"], ["--cif-root", "{cif-root}"]],
                  needs="data-root", minutes="60",
                  report=lambda: (_mod("run_sutherland").report(_r("table.md")),
                                  _mod("run_sutherland").report_cif(_r("cif/table.md"))),
                  tables=["table.md", "cif/table.md"],
                  readme=["docs/benchmark.md (first and second part, spliced)"]),
        Benchmark("alignment", "a curated alignment against an automatic one, nine targets",
                  "run_alignment.py", "alignment", compute=[[]], minutes="15",
                  report=lambda: _mod("run_alignment").report(_r("alignment/table.md")),
                  tables=["alignment/table.md"],
                  readme=["Topomer fields: the same measurement where the alignment is automatic"]),
        Benchmark("cyclodextrin", "the three kernels where no frame exists: cyclodextrin guest "
                  "sets, neutral and at pH 7",
                  "run_cyclodextrin.py", "cyclodextrin", compute=[[]], minutes="20",
                  report=lambda: _mod("run_cyclodextrin").report(_r("cyclodextrin/table.md")),
                  tables=["cyclodextrin/table.md"]),
        Benchmark("topomer", "aligned CMF against topomer CMF on the Sutherland sets",
                  "run_topomer.py", "topomer", compute=[[]], minutes="20",
                  report=lambda: _mod("run_topomer").report(
                      _r("topomer/table.md"), results=_r("topomer"), title="on the Sutherland sets"),
                  tables=["topomer/table.md"], readme=["Topomer fields"]),
        Benchmark("topomer_mittal", "the same on the Mittal sets", "run_topomer.py",
                  "topomer_mittal", compute=[["--mittal"]], minutes="90",
                  report=lambda: _mod("run_topomer").report(
                      _r("topomer_mittal/table.md"), results=_r("topomer_mittal"),
                      title="on the Mittal sets, whose alignments are automatic"),
                  tables=["topomer_mittal/table.md"], readme=["Topomer fields"]),
        Benchmark("corrosion", "frames, fields and baselines on the corrosion-inhibitor series",
                  "run_corrosion.py", "corrosion", compute=[[]], minutes="30",
                  report=lambda: _mod("run_corrosion").report(
                      _r("corrosion/table.md"), results=_r("corrosion"),
                      title="on the corrosion-inhibitor series"),
                  tables=["corrosion/table.md"], readme=["Heterogeneous series"]),
        Benchmark("corrosion_hcl", "the same, protonated for 1 M HCl", "run_corrosion.py",
                  "corrosion_hcl", compute=[["--hcl"]], minutes="30",
                  report=lambda: _mod("run_corrosion").report(
                      _r("corrosion_hcl/table.md"), results=_r("corrosion_hcl"),
                      title="on the corrosion-inhibitor series, protonated for 1 M HCl"),
                  tables=["corrosion_hcl/table.md"]),
        Benchmark("corrosion_qm", "quantum-chemical fields and descriptors on four corrosion series",
                  "run_corrosion_qm.py", "corrosion_qm", compute=[[]], needs="orca", minutes="300",
                  report=lambda: _mod("run_corrosion_qm").report(
                      _r("corrosion_qm"), _r("corrosion_qm/table.md")),
                  tables=["corrosion_qm/table.md"]),
        Benchmark("cdft_kernels", "conceptual-DFT fields on a grid, fitted and condensed",
                  "run_cdft_kernels.py", "cdft_kernels", compute=[[]], needs="orca", minutes="15",
                  report=lambda: _mod("run_cdft_kernels").report(_r("cdft_kernels")),
                  tables=["cdft_kernels/table.md"]),
        Benchmark("hammett", "a field-based Hammett equation, several charge models and ensembles",
                  "run_hammett.py", "hammett",
                  compute=[[], ["--sdf", "examples/hammett/hammett_sym.sdf"],
                           ["--sdf", "examples/hammett/hammett_qm.sdf", "--charges-prop", "q_hirshfeld"],
                           ["--sdf", "examples/hammett/hammett_qm_sym.sdf", "--charges-prop", "q_hirshfeld"]],
                  minutes="10",
                  report=lambda: _mod("run_hammett").report(_r("hammett")),
                  tables=["hammett/table.md"],
                  readme=["A field-based Hammett equation (two tables)"]),
        Benchmark("am_eu", "the Am/Eu separation set: classic, indicator and quantum fields",
                  "run_am_eu.py", "am_eu",
                  compute=[["--stage", "fields"], ["--stage", "orca"], ["--stage", "quantum"],
                           ["--stage", "cdft"]],
                  needs="orca", minutes="180",
                  report=lambda: _mod("run_am_eu").report(_r("am_eu/am_eu.json")),
                  tables=["am_eu/table.md"]),
        Benchmark("evidence", "can the marginal likelihood choose alpha and lambda",
                  "run_evidence.py", "evidence", compute=[[]],
                  minutes="10",
                  report=lambda: _mod("run_evidence").report(_r("evidence")),
                  tables=["evidence/table.md"]),
        Benchmark("mixing", "a posterior over the field weights, and what it is right about",
                  "run_mixing.py", "mixing", compute=[[]],
                  minutes="30",
                  report=lambda: _mod("run_mixing").report(_r("mixing")),
                  tables=["mixing/table.md"]),
        Benchmark("transfer", "does the co-field cosine say whether the SAR transfers",
                  "run_transfer.py", "transfer", compute=[[]],
                  minutes="15",
                  report=lambda: _mod("run_transfer").report(_r("transfer")),
                  tables=["transfer/table.md"]),
        Benchmark("design", "would the design loop have suggested the right substituent",
                  "run_design.py", "design", compute=[[]],
                  minutes="25",
                  report=lambda: _mod("run_design").report(_r("design")),
                  tables=["design/table.md"]),
        Benchmark("active", "which molecule to make next, against picking at random",
                  "run_active.py", "active", compute=[[]],
                  minutes="20",
                  report=lambda: _mod("run_active").report(_r("active")),
                  tables=["active/table.md"]),
        Benchmark("cofield_error", "does the co-field error bar know which lobes are real",
                  "run_cofield_error.py", "cofield_error", compute=[[]],
                  minutes="20",
                  report=lambda: _mod("run_cofield_error").report(_r("cofield_error")),
                  tables=["cofield_error/table.md"]),
        Benchmark("errorpred", "what knows when a prediction is bad",
                  "run_errorpred.py", "errorpred", compute=[[]],
                  minutes="15",
                  report=lambda: _mod("run_errorpred").report(_r("errorpred")),
                  tables=["errorpred/table.md"]),
        Benchmark("perfield", "one attenuation per field, or one shared by all of them",
                  "run_perfield.py", "perfield", compute=[[]],
                  minutes="10",
                  report=lambda: _mod("run_perfield").report(_r("perfield")),
                  tables=["perfield/table.md"]),
        Benchmark("chirality", "do the signed three-point channels buy anything",
                  "run_chirality.py", "chirality", compute=[[]],
                  minutes="20",
                  report=lambda: _mod("run_chirality").report(_r("chirality")),
                  tables=["chirality/table.md"]),
        Benchmark("structure", "is this set that set, and what are its real groups",
                  "run_structure.py", "structure", compute=[[]],
                  minutes="20",
                  report=lambda: _mod("run_structure").report(_r("structure")),
                  tables=["structure/table.md"]),
        Benchmark("ranking", "does fitting the order beat fitting the values",
                  "run_ranking.py", "ranking", compute=[["--repeats", "10"]],
                  minutes="15",
                  report=lambda: _mod("run_ranking").report(_r("ranking")),
                  tables=["ranking/table.md"]),
        Benchmark("conformal", "do the conformal intervals keep the coverage they promise",
                  "run_conformal.py", "conformal", compute=[["--repeats", "10"]],
                  minutes="3",
                  report=lambda: _mod("run_conformal").report(_r("conformal")),
                  tables=["conformal/table.md"]),
        Benchmark("diagnose", "the data-set diagnosis, on sets whose answers are known",
                  "run_diagnose.py", "diagnose", compute=[["--family", "all", "--force"]],
                  minutes="15",
                  report=lambda: _mod("run_diagnose").report(_r("diagnose")),
                  tables=["diagnose/table.md"]),
        Benchmark("exposure", "fields weighted by solvent exposure against the plain fields",
                  "run_exposure.py", "exposure",
                  compute=[["--family", "openff"], ["--family", "sutherland"]],
                  minutes="20",   # examples/sutherland and examples/openff, both in the repo
                  report=lambda: _mod("run_exposure").report(_r("exposure")),
                  tables=["exposure/table.md"]),
        Benchmark("bound", "ranking by a confidence bound against ranking by the prediction",
                  "run_bound.py", "bound", compute=[[]], minutes="15",
                  report=lambda: _mod("run_bound").report(_r("bound/table.md")),
                  tables=["bound/table.md"],
                  readme=["Ranking by a bound rather than by the prediction"]),
        Benchmark("fusion", "data-fusion rules against the one-class descriptions, both "
                            "representations",
                  "run_fusion.py", "fusion", compute=[[]], minutes="2",
                  report=lambda: _mod("run_fusion").report(_r("fusion/table.md")),
                  tables=["fusion/table.md"],
                  readme=["Several actives and no inactives: data fusion"]),
        Benchmark("poses", "alignment methods judged against the crystal poses of sixteen FEP sets",
                  "run_poses.py", "poses",
                  compute=[["--sets", "all", "--force"],
                           ["--sets", "all", "--flex", "--force"]],
                  minutes="5",
                  report=lambda: _mod("run_poses").report(_r("poses")),
                  tables=["poses/table.md"]),
        Benchmark("multitask", "two endpoints on one kernel, on the corrosion series",
                  "run_multitask.py", "multitask", compute=[[]], minutes="5",
                  report=lambda: _mod("run_multitask").report(_r("multitask")),
                  tables=["multitask/table.md"],
                  readme=["Two endpoints on one kernel (spliced)"]),
        Benchmark("matbench_phonons", "continuous molecular fields of crystals: the periodic "
                  "autocorrelation kernel on the Matbench phonons task (downloads the 460 kB task "
                  "file into benchmarks/cache on the first run)",
                  "run_matbench_phonons.py", "matbench_phonons", compute=[["--force"]], minutes="25",
                  report=lambda: _mod("run_matbench_phonons").report(_r("matbench_phonons/table.md")),
                  tables=["matbench_phonons/table.md"],
                  readme=["Crystals: the periodic autocorrelation"]),
    ]


# ----------------------------------------------------------------------------- the documents

#: (document, marker, how to get the text)
SPLICES = [
    (os.path.join(ROOT, "docs", "benchmark.md"), "TABLE", lambda: _read(_r("table.md"))),
    (os.path.join(ROOT, "docs", "benchmark.md"), "CIF-TABLE", lambda: _read(_r("cif/table.md"))),
    (os.path.join(ROOT, "README.md"), "MULTITASK-TABLE",
     lambda: _mod("run_multitask").readme_table(_r("multitask"))),
]


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def splice(doc: str, tag: str, text: str) -> bool:
    """Put `text` between the markers of `tag` in `doc`; True when the document changed."""
    src = _read(doc)
    pattern = re.compile(rf"(<!-- {re.escape(tag)} -->\n).*?(<!-- /{re.escape(tag)} -->)", re.S)
    if not pattern.search(src):
        raise SystemExit(f"{os.path.relpath(doc, ROOT)} has no <!-- {tag} --> markers")
    body = text if text.endswith("\n") else text + "\n"
    new = pattern.sub(lambda m: m.group(1) + body + m.group(2), src, count=1)
    if new != src:
        with open(doc, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(new)
        return True
    return False


def rebuild_tables(quiet=True):
    """Every table.md from its result files, then the documents.

    Returns ``(written, changed)``: every file the rebuild wrote, and those whose content is not
    what was on disk before.  ``--check`` is the statement that the second list is empty: the
    committed tables and documents are what the result files give.
    """
    written, changed = [], []
    devnull = open(os.devnull, "w", encoding="utf-8")
    for b in registry():
        if b.report is None:
            continue
        before = {t: (_read(_r(t)) if os.path.exists(_r(t)) else None) for t in b.tables}
        try:
            if quiet:
                out, sys.stdout = sys.stdout, devnull
                try:
                    b.report()
                finally:
                    sys.stdout = out
            else:
                b.report()
        except SystemExit as exc:          # a report with nothing to report says so and exits
            print(f"  {b.name}: {exc}")
            continue
        for t in b.tables:
            written.append(_r(t))
            if os.path.exists(_r(t)) and _read(_r(t)) != before[t]:
                changed.append(_r(t))
    for doc, tag, text in SPLICES:
        if splice(doc, tag, text()):
            written.append(doc)
            changed.append(f"{doc} <!-- {tag} -->")
    return written, changed


# ----------------------------------------------------------------------------- the manifest

def _sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tracked(folder) -> List[str]:
    """Files under `folder` that git tracks, or every file when git is not there."""
    out = _git("ls-files", "--", folder)
    if out is not None:
        return sorted(p for p in out.splitlines() if p)
    found = []
    for base, _, files in os.walk(os.path.join(ROOT, folder)):
        for f in files:
            found.append(os.path.relpath(os.path.join(base, f), ROOT).replace(os.sep, "/"))
    return sorted(found)


def versions():
    from importlib import metadata
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_manifest() -> dict:
    data = {p: {"sha256": _sha256(os.path.join(ROOT, p)),
                "bytes": os.path.getsize(os.path.join(ROOT, p))} for p in tracked("examples")}
    results = {}
    for p in tracked("benchmarks/results"):
        results[p] = _sha256(os.path.join(ROOT, p))
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", "cmfqsar", "benchmarks", "examples")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions(),
        "data": data,
        "results": results,
    }


def write_manifest():
    m = build_manifest()
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"manifest: {len(m['data'])} data files, {len(m['results'])} result files, "
          f"commit {m['commit'] and m['commit'][:9]}, cmfqsar {m['packages'].get('cmfqsar')}")
    return m


def check_manifest() -> List[str]:
    """Data files whose hash differs from the manifest, or that it does not know."""
    if not os.path.exists(MANIFEST):
        return ["no manifest: run --manifest"]
    old = json.load(open(MANIFEST, encoding="utf-8"))
    problems = []
    for p in tracked("examples"):
        h = _sha256(os.path.join(ROOT, p))
        if p not in old["data"]:
            problems.append(f"not in the manifest: {p}")
        elif old["data"][p]["sha256"] != h:
            problems.append(f"changed since the manifest: {p}")
    for p in old["data"]:
        if not os.path.exists(os.path.join(ROOT, p)):
            problems.append(f"in the manifest but gone: {p}")
    return problems


# ----------------------------------------------------------------------------- listing, computing

def list_benchmarks():
    print(f"{'benchmark':16s} {'results':>8s} {'table':>6s}  needs      minutes  what")
    for b in registry():
        folder = _r(b.results)
        n = (len([f for f in os.listdir(folder) if f.endswith((".json", ".csv"))])
             if os.path.isdir(folder) else 0)
        have = all(os.path.exists(_r(t)) for t in b.tables)
        print(f"{b.name:16s} {n:8d} {'yes' if have else 'no':>6s}  {b.needs or '-':10s} "
              f"{b.minutes:>7s}  {b.what}")
        for r in b.readme:
            print(f"{'':16s} {'':8s} {'':6s}  README: {r}")
    print("\nresult tables: " + ", ".join(f"results/{t}" for b in registry() for t in b.tables))
    print("spliced into the documents: " + ", ".join(f"{os.path.relpath(d, ROOT)} <!-- {t} -->"
                                                     for d, t, _ in SPLICES))


def compute(names, data_root=None, cif_root=None, orca=False):
    wanted = registry() if names == ["all"] else [b for b in registry() if b.name in names]
    unknown = set(names) - {b.name for b in registry()} - {"all"}
    if unknown:
        raise SystemExit("unknown benchmark(s): " + ", ".join(sorted(unknown)))
    for b in wanted:
        for argv in b.compute:
            args = []
            skip = None
            for a in argv:
                if a == "{data-root}":
                    if not data_root:
                        skip = "needs --data-root (the Sutherland MOL2 originals)"
                    a = data_root or ""
                elif a == "{cif-root}":
                    if not cif_root:
                        skip = "needs --cif-root (the data of the 2015 CIF paper)"
                    a = cif_root or ""
                args.append(a)
            if b.needs == "orca" and not orca and any(x in ("orca", "quantum", "cdft") for x in argv) \
                    or (b.needs == "orca" and not orca and b.name in ("corrosion_qm", "cdft_kernels")):
                skip = "needs ORCA (--orca)"
            if skip:
                print(f"--- {b.name} {' '.join(argv)}: skipped, {skip}", flush=True)
                continue
            cmd = [sys.executable, os.path.join(HERE, b.script), *args]
            print(f"=== {b.name}: {' '.join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd[1:])}",
                  flush=True)
            t0 = time.time()
            rc = subprocess.call(cmd, cwd=ROOT)
            print(f"--- {b.name}: exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
    rebuild_tables()
    write_manifest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compute", nargs="+", metavar="NAME")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--cif-root", default=None)
    ap.add_argument("--orca", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.compute:
        compute(args.compute, args.data_root, args.cif_root, args.orca)
    changed = []
    if args.tables or args.check:
        written, changed = rebuild_tables()
        print("tables rebuilt: " + ", ".join(os.path.relpath(w, ROOT) for w in written))
        if changed:
            print("changed by the rebuild: " + ", ".join(os.path.relpath(c, ROOT) for c in changed))
    if args.manifest:
        write_manifest()
    if args.check:
        problems = check_manifest()
        problems += ["table or document was not what its result files give: "
                     + os.path.relpath(c, ROOT) for c in changed]
        if problems:
            print("CHECK FAILED")
            for p in problems:
                print("  " + p)
            return 1
        print("check passed: tables and documents match their result files, data matches the manifest")
    if not (args.compute or args.tables or args.manifest or args.check) or args.list:
        list_benchmarks()
    return 0


if __name__ == "__main__":
    sys.exit(main())
