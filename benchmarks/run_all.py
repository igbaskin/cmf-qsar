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
    prepare: List[str] = field(default_factory=list)         # the command that fetches or builds
    # the input the script reads, as a command line under benchmarks/.  It is not run by
    # --compute: it needs the network or hours, and its result is kept in the repository.


def _mod(name):
    return importlib.import_module(name)


def _write_table(name, text):
    """Write `text` as `table.md` of the results folder `name`.

    For a script whose `report` returns the text of its table and leaves the writing to its
    `main`.  The line break at the end is the one that the script's `main` adds.
    """
    with open(_r(os.path.join(name, "table.md")), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _shown(path):
    """`path` relative to the repository, for a message.  A path on another drive (Windows)
    has no relative form and is given as it is."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def _report(b):
    """Call the report of benchmark `b` and make sure that it wrote every table of `b.tables`.

    Each table is moved aside before the call.  If the report leaves no file in its place, the
    table is put back and an error is raised.  Without this, a report that only returns its
    text would leave the committed table in place, and ``--check`` would compare the table
    with itself and pass.
    """
    aside = {_r(t): _r(t) + ".before" for t in b.tables if os.path.exists(_r(t))}
    for path, kept in aside.items():
        os.replace(path, kept)
    missing = []
    try:
        b.report()
    finally:
        for path, kept in aside.items():
            if os.path.exists(path):
                os.remove(kept)
            else:                          # nothing was written: the table is put back
                os.replace(kept, path)
                missing.append(path)
    if missing:
        raise RuntimeError(
            f"the report of benchmark '{b.name}' did not write "
            + ", ".join(_shown(m) for m in missing)
            + ". The table cannot be compared with its result files. The report has to write "
              "the table; returning its text is not enough.")


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
                  # One command per stored summary file, and no two of them write the same file.
                  compute=[[],
                           ["--sdf", "examples/hammett/hammett_qm_sym.sdf",
                            "--charges-prop", "q_hirshfeld"],
                           ["--sdf", "examples/hammett/hammett_ens.sdf"],
                           ["--sdf", "examples/hammett/hammett_ens.sdf", "--weighting", "boltzmann"],
                           ["--sdf", "examples/hammett/hammett_ens_qm.sdf",
                            "--charges-prop", "q_hirshfeld"],
                           ["--sdf", "examples/hammett/hammett_ens_qm.sdf",
                            "--charges-prop", "q_hirshfeld", "--weighting", "boltzmann"]],
                  minutes="8",
                  report=lambda: _mod("run_hammett").report(_r("hammett")),
                  tables=["hammett/table.md"],
                  readme=["A field-based Hammett equation (two tables)"]),
        Benchmark("denmark",
                  "a kinetic selectivity as the activity: 43 chiral phosphoric acids on 25 "
                  "substrate combinations, the frame measured against the invariant kernel",
                  "run_denmark.py", "denmark", compute=[[]], minutes="5",
                  report=lambda: _mod("run_denmark").write_table(
                      _load(_r("denmark/summary.json")), _r("denmark/table.md")),
                  tables=["denmark/table.md"],
                  readme=["A kinetic selectivity as the activity (two tables)"]),
        Benchmark("nonlinearity", "the molecule-dependent co-field of the Gaussian layer: its "
                  "spread and the non-linearity index on the Hammett set",
                  "run_nonlinearity.py", "nonlinearity", compute=[[]], minutes="3",
                  report=lambda: _mod("run_nonlinearity").report(_r("nonlinearity")),
                  tables=["nonlinearity/table.md"]),
        Benchmark("interactions", "second derivatives of the Gaussian layer on para-disubstituted "
                  "benzenes with a known pair term: is the pair-interaction field on the pair",
                  "run_interactions.py", "interactions", compute=[[]], minutes="5",
                  report=lambda: _mod("run_interactions").report(_r("interactions")),
                  tables=["interactions/table.md"]),
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
        Benchmark("query", "is the co-field pharmacophore a usable screening query",
                  "run_query.py", "query", compute=[[]],
                  minutes="20",
                  report=lambda: _mod("run_query").report(_r("query")),
                  tables=["query/table.md"]),
        Benchmark("pharmfields", "six pharmacophore features against the Sybyl indicators",
                  "run_pharmfields.py", "pharmfields", compute=[[]],
                  minutes="45",
                  report=lambda: _mod("run_pharmfields").report(_r("pharmfields")),
                  tables=["pharmfields/table.md"]),
        Benchmark("pharmfields_mittal",
                  "the same six features on the 38 Mittal sets, 15 of them small",
                  "run_pharmfields.py", "pharmfields_mittal", compute=[["--mittal"]],
                  minutes="40",
                  report=lambda: _mod("run_pharmfields").report(_r("pharmfields_mittal")),
                  tables=["pharmfields_mittal/table.md"]),
        Benchmark("host_pocket",
                  "does complementarity with a cyclodextrin cavity predict the constant",
                  "run_host_pocket.py", "host_pocket", compute=[[]],
                  minutes="170",
                  report=lambda: _write_table(
                      "host_pocket", _mod("run_host_pocket").report(_r("host_pocket"))),
                  tables=["host_pocket/table.md"]),
        Benchmark("guest_design",
                  "does a model with no frame name the group worth making next",
                  "run_guest_design.py", "guest_design", compute=[[]],
                  minutes="5",
                  report=lambda: _write_table(
                      "guest_design", _mod("run_guest_design").report(_r("guest_design"))),
                  tables=["guest_design/table.md"]),
        Benchmark("two_endpoints",
                  "two cyclodextrin cavities in one model, and the selectivity between them",
                  "run_two_endpoints.py", "two_endpoints",
                  compute=[[], ["--pair", "beta", "gamma"], ["--pair", "alpha", "gamma"]],
                  minutes="5",
                  report=lambda: _write_table(
                      "two_endpoints", _mod("run_two_endpoints").report(_r("two_endpoints"))),
                  tables=["two_endpoints/table.md"]),
        Benchmark("selectivity",
                  "the AMPA / glycine-NMDA set of the 2003 selectivity-fields paper: the "
                  "difference of two affinities modelled directly, as a difference of models, "
                  "and as a two-task fit",
                  "run_selectivity.py", "selectivity", compute=[[]], minutes="1",
                  report=lambda: _write_table(
                      "selectivity", _mod("run_selectivity").report(_r("selectivity"))),
                  tables=["selectivity/table.md"],
                  readme=["Several endpoints on one kernel: the selectivity co-field"]),
        Benchmark("guest_atoms",
                  "does an alignment-free model name the group that enters the cavity",
                  "run_guest_atoms.py", "guest_atoms", compute=[[]],
                  minutes="5",
                  report=lambda: _write_table(
                      "guest_atoms", _mod("run_guest_atoms").report(_r("guest_atoms"))),
                  tables=["guest_atoms/table.md"]),
        Benchmark("local",
                  "does the field around an atom predict a shift or an acidity of that atom",
                  "run_local.py", "local", compute=[[]],
                  minutes="45",
                  report=lambda: _mod("run_local").write_table(_r("local")),
                  tables=["local/table.md"]),
        Benchmark("local_cofield",
                  "a co-field around one atom: does it sit where substituent effects do",
                  "run_local_cofield.py", "local_cofield", compute=[[]],
                  minutes="15",
                  report=lambda: _mod("run_local_cofield").write_table(_r("local_cofield")),
                  tables=["local_cofield/table.md"]),
        Benchmark("local_scale",
                  "how far the local kernel goes: the primal route, a learning curve, landmarks",
                  "run_local_scale.py", "local_scale", compute=[[]],
                  needs="data-root", minutes="20",
                  report=lambda: _mod("run_local_scale").write_table(_r("local_scale")),
                  tables=["local_scale/table.md"]),
        Benchmark("soap_gate",
                  "is the angular term what the local kernel lacks: SOAP as the gate",
                  "run_soap_gate.py", "soap_gate", compute=[[]],
                  minutes="35",
                  report=lambda: _mod("run_soap_gate").write_table(_r("soap_gate")),
                  tables=["soap_gate/table.md"]),
        Benchmark("triangles",
                  "the three-distance autocorrelation against the two-point kernel and SOAP",
                  "run_triangles.py", "triangles", compute=[[]],
                  minutes="40",
                  report=lambda: _mod("run_triangles").write_table(_r("triangles")),
                  tables=["triangles/table.md"]),
        Benchmark("hops",
                  "does the field around a Li hop predict its barrier, across structure types, "
                  "and does the same model read from another program give the same numbers",
                  "run_hops.py", "hops",
                  # One command per stored result file, and no two of them write the same file:
                  # the first writes public.json, hosts.json and disordered.json, the second the
                  # fine co-field profile the manual draws, the third the reading of the model
                  # through the interfaces of plan item MS7.
                  compute=[[], ["--fine-profile"], ["--interfaces"]],
                  prepare=["prepare_hops.py"],
                  minutes="135",       # measured: 67 for the first, 40 for the second, 25 third
                  report=lambda: _mod("run_hops").write_table(_r("hops")),
                  tables=["hops/table.md"]),
        Benchmark("litraj",
                  "the hop models on a published benchmark of the same labels: 122,421 "
                  "bond-valence barriers of the LiTraj collection, the same descriptions "
                  "fitted the way the published models are fitted, and what a width on the "
                  "kernel, the angle at the midpoint, the published models' own strongest "
                  "feature and the fitting method itself recover of the difference",
                  "run_litraj.py", "litraj",
                  compute=[[]],
                  prepare=["prepare_litraj.py --fetch"],
                  # measured: 124.7, of which the two forest arms on the three-distance rows
                  # took 39 and 27 minutes and the overlap kernel 30.  The same kernel took 10
                  # minutes for the same numbers on a quieter machine, so this figure depends on
                  # what else is running.
                  minutes="125",
                  report=lambda: _mod("run_litraj").write_table(_r("litraj")),
                  tables=["litraj/table.md"]),
        Benchmark("battery",
                  "the average voltage of an insertion electrode from the cell, against a "
                  "composition baseline",
                  "run_battery.py", "battery",
                  compute=[["--stage", "fetch"], ["--stage", "check"], ["--stage", "voltage"],
                           ["--stage", "difference"]],
                  minutes="120",
                  report=lambda: _mod("run_battery").write_table(_r("battery")),
                  tables=["battery/table.md"]),
        Benchmark("ether",
                  "an ion probe field: where it puts the ion, and the binding energy and the "
                  "selectivity of three cations on 1,050 ethers",
                  "run_ether.py", "ether",
                  compute=[["--stage", "probe"], ["--stage", "bind"], ["--stage", "select"]],
                  minutes="60",
                  report=lambda: _mod("run_ether").write_table(_r("ether")),
                  tables=["ether/table.md"]),
        Benchmark("solvation",
                  "a cation with its first shell: the builder against a published relaxation, "
                  "and the complex against the free solvent",
                  "run_solvation.py", "solvation",
                  compute=[["--stage", "builder", "--builder-limit", "150"],
                           ["--stage", "gate"], ["--stage", "series"]],
                  minutes="45",
                  report=lambda: _mod("run_solvation").write_table(_r("solvation")),
                  tables=["solvation/table.md"]),
        Benchmark("shells",
                  "solvation shells sampled from a bulk electrolyte against one optimised "
                  "complex, on measured conductivities",
                  "run_shells.py", "shells",
                  compute=[["--stage", "data"], ["--stage", "gate"], ["--stage", "cofield"]],
                  minutes="25",
                  report=lambda: _mod("run_shells").write_table(_r("shells")),
                  tables=["shells/table.md"]),
        Benchmark("phenazine",
                  "quantum fields (Hirshfeld charge, Fukui indices, the spin of the radical "
                  "anion) against the five ordinary fields and against the global quantum "
                  "numbers, on 69 phenazine reduction potentials",
                  "run_phenazine.py", "phenazine",
                  compute=[["--stage", "data"], ["--stage", "gate"], ["--stage", "cofield"]],
                  minutes="3",
                  report=lambda: _mod("run_phenazine").write_table(_r("phenazine")),
                  tables=["phenazine/table.md"]),
        Benchmark("slab",
                  "a metal slab as the receptor: does docking constrained to the surface recover "
                  "the UMA-relaxed poses, and does a complementarity score track the adsorption "
                  "energy",
                  "run_slab.py", "slab",
                  compute=[["--stage", "receptor"], ["--stage", "pose"], ["--stage", "score"],
                           ["--stage", "corrosion"]],
                  minutes="32",
                  report=lambda: _mod("run_slab").write_table(_r("slab")),
                  tables=["slab/table.md"]),
        Benchmark("anchors",
                  "the field and its gradient read at a few anchored points of a shared frame, "
                  "against the field model that reads them everywhere, on the Hammett constants "
                  "and on 25 nickel complexes",
                  "run_anchors.py", "anchors",
                  compute=[["--stage", "check"], ["--stage", "hammett"], ["--stage", "nickel"]],
                  minutes="2",
                  report=lambda: _mod("run_anchors").write_table(_r("anchors")),
                  tables=["anchors/table.md"]),
        Benchmark("efield",
                  "the electric-field kernel, which pairs two molecules by the gradients of "
                  "their fields, against the scalar electrostatic kernel on the Hammett "
                  "constants and on 25 nickel complexes",
                  "run_efield.py", "efield",
                  compute=[["--stage", "check"], ["--stage", "hammett"], ["--stage", "nickel"]],
                  minutes="2",
                  report=lambda: _mod("run_efield").write_table(_r("efield")),
                  tables=["efield/table.md"]),
        Benchmark("metal",
                  "the fields of a coordination complex against its composition and its "
                  "population analysis, on 2216 iron redox potentials and on 25 nickel "
                  "complexes in a shared frame",
                  "run_metal.py", "metal",
                  compute=[["--stage", "data"], ["--stage", "iron"], ["--stage", "nickel"],
                           ["--stage", "cofield"]],
                  minutes="60",
                  report=lambda: _mod("run_metal").write_table(_r("metal")),
                  tables=["metal/table.md"]),
        Benchmark("sac",
                  "the field around a single-atom site against its H, O, OH and OOH binding energies",
                  "run_sac.py", "sac", compute=[[]],
                  minutes="15",
                  report=lambda: _mod("run_sac").write_table(_r("sac")),
                  tables=["sac/table.md"]),
        Benchmark("fluctuation",
                  "is the variance of a field across conformations worth a block of its own",
                  "run_fluctuation.py", "fluctuation", compute=[[]],
                  minutes="15",
                  report=lambda: _mod("run_fluctuation").write_table(_r("fluctuation")),
                  tables=["fluctuation/table.md"]),
        Benchmark("designloop",
                  "a loop with two objectives: does its front hold what the authors made next",
                  "run_designloop.py", "designloop", compute=[[]],
                  minutes="5",
                  report=lambda: _mod("run_designloop").write_table(_r("designloop")),
                  tables=["designloop/table.md"]),
        Benchmark("antitarget",
                  "a potency and an anti-target on the same compounds: the difference co-field",
                  "run_antitarget.py", "antitarget", compute=[[]],
                  minutes="5",
                  report=lambda: _mod("run_antitarget").write_table(_r("antitarget")),
                  tables=["antitarget/table.md"]),
        Benchmark("batch",
                  "ten compounds at once: the posterior covariance against a similarity filter",
                  "run_batch.py", "batch", compute=[[]],
                  minutes="20",
                  report=lambda: _mod("run_batch").write_table(_r("batch")),
                  tables=["batch/table.md"]),
        Benchmark("fraggrow",
                  "can a field score say what to grow a bound fragment into",
                  "run_fraggrow.py", "fraggrow", compute=[[]],
                  minutes="10",
                  report=lambda: _mod("run_fraggrow").write_table(_r("fraggrow")),
                  tables=["fraggrow/table.md"]),
        Benchmark("transformfield",
                  "does a transformation carry its effect from one series to another",
                  "run_transformfield.py", "transformfield", compute=[[]],
                  minutes="25",
                  report=lambda: _mod("run_transformfield").write_table(_r("transformfield")),
                  tables=["transformfield/table.md"]),
        Benchmark("bioisostere",
                  "which groups are interchangeable: the co-field against shape and fingerprints",
                  "run_bioisostere.py", "bioisostere", compute=[[]],
                  minutes="5",
                  report=lambda: _mod("run_bioisostere").write_table(_r("bioisostere")),
                  tables=["bioisostere/table.md"]),
        Benchmark("maskedmatch",
                  "a scaffold the model never saw, scored only where the co-field has weight",
                  "run_maskedmatch.py", "maskedmatch", compute=[["--force"]],
                  minutes="2",
                  report=lambda: _mod("run_maskedmatch").report(_r("maskedmatch")),
                  tables=["maskedmatch/table.md"]),
        Benchmark("restriction",
                  "does a ring closure remove the field that moves, and does potency follow",
                  "run_restriction.py", "restriction", compute=[[]],
                  minutes="10",
                  report=lambda: _mod("run_restriction").write_table(_r("restriction")),
                  tables=["restriction/table.md"]),
        Benchmark("metabolism",
                  "which atom a cytochrome P450 attacks, against a count over bonds",
                  "run_metabolism.py", "metabolism", compute=[[]],
                  minutes="5",
                  report=lambda: _mod("run_metabolism").write_table(_r("metabolism")),
                  tables=["metabolism/table.md"]),
        Benchmark("increments",
                  "do increments fitted on one series transfer to another",
                  "run_increments.py", "increments", compute=[[]],
                  minutes="90",
                  report=lambda: _write_table(
                      "increments", _mod("run_increments").report(_r("increments"))),
                  tables=["increments/table.md"]),
        Benchmark("protonation",
                  "does a pKa-weighted protonation ensemble beat one chosen state",
                  "run_protonation.py", "protonation",
                  compute=[["--ph", "0"], ["--ph", "3"]],
                  minutes="15",
                  report=lambda: _write_table(
                      "protonation", _mod("run_protonation").report(_r("protonation"))),
                  tables=["protonation/table.md"]),
        Benchmark("anisotropic", "can a direction on every atom represent a Fukui function",
                  "run_anisotropic.py", "anisotropic", compute=[[]],
                  needs="orca", minutes="90",
                  report=lambda: _mod("run_anisotropic").report(_r("anisotropic")),
                  tables=["anisotropic/table.md"]),
        Benchmark("cases", "three series worked through with the interpretation tools",
                  "run_cases.py", "cases", compute=[[]],
                  minutes="3",
                  report=lambda: _mod("run_cases").report(_r("cases")),
                  tables=["cases/table.md"]),
        Benchmark("speed", "what the kernel costs and what each shortcut saves",
                  "run_speed.py", "speed", compute=[[]],
                  minutes="25",
                  report=lambda: _mod("run_speed").report(_r("speed")),
                  tables=["speed/table.md"]),
        Benchmark("lowrank",
                  "fitting in the space of the landmarks: is it the same model, and what does "
                  "it save",
                  "run_lowrank.py", "lowrank", compute=[[]],
                  minutes="25",
                  report=lambda: _mod("run_lowrank").write_table(_r("lowrank")),
                  tables=["lowrank/table.md"]),
        Benchmark("periodic_overlap",
                  "the periodic overlap kernel of crystals against the periodic "
                  "autocorrelation, and the co-field inside the cell: 1,500 perovskite cells of "
                  "matbench_perovskites, in the cells as they come and on one lattice "
                  "(downloads about 30 MB into benchmarks/cache on the first run)",
                  "run_periodic_overlap.py", "periodic_overlap", compute=[["--limit", "1500"]],
                  minutes="40",
                  report=lambda: _mod("run_periodic_overlap").report(),
                  tables=["periodic_overlap/table.md"]),
        Benchmark("2dmd",
                  "a defect series on one host lattice: the periodic overlap of difference "
                  "fields with the host symmetry against the periodic autocorrelation, on "
                  "2,000 point-defect configurations each of MoS2 and WSe2 from the 2DMD set "
                  "(downloads 5 MB into benchmarks/cache on the first run; needs Numba for the "
                  "symmetry average at this size)",
                  "run_2dmd.py", "2dmd", compute=[["--limit", "2000", "--engine", "numba"]],
                  minutes="120",
                  report=lambda: _mod("run_2dmd").report(),
                  tables=["2dmd/table.md"]),
        Benchmark("sites",
                  "a site of a periodic cell as the sample, and a whole-cell model read site by "
                  "site: the split of the periodic autocorrelation against the cut cluster of "
                  "the single-atom catalysts and against counts, on the Li site energies of "
                  "the three electrolytes, and the split as a map against the site model's own "
                  "ranking",
                  "run_sites.py", "sites", compute=[[]],
                  minutes="30",
                  report=lambda: _mod("run_sites").report(),
                  tables=["sites/table.md"]),
        Benchmark("rowlength",
                  "what the row length is worth, where the invariant kernel divides it out: the "
                  "same kernel with the length divided out, with it kept, with it added as one "
                  "more channel and on its own, over five series -- the Li site energies, the "
                  "single-atom catalysts, the prepared hop sample of LiTraj, the 2DMD defect "
                  "series and the beta-cyclodextrin guests (the defect stage reads the same "
                  "cache as the 2dmd benchmark)",
                  "run_rowlength.py", "rowlength",
                  compute=[["--lithium"], ["--sac"], ["--molecules"], ["--hops"],
                           ["--defects"]],
                  minutes="90",
                  report=lambda: _mod("run_rowlength").report(),
                  tables=["rowlength/table.md"]),
        Benchmark("colweights",
                  "which columns of the invariant kernel count: one weight per pair of field "
                  "types, one per column, the best columns alone and a forest's own importances, "
                  "each chosen inside the training part, against the equal weights the program "
                  "uses, with a forest on the same rows as the gap to close -- on the hop "
                  "environments, the beta-cyclodextrin guests and the single-atom catalysts",
                  "run_colweights.py", "colweights",
                  compute=[["--hops"], ["--molecules"], ["--sac"]],
                  minutes="60",
                  report=lambda: _mod("run_colweights").report(),
                  tables=["colweights/table.md"]),
        Benchmark("cell_view",
                  "a periodic structure in the application: what one cell with its images "
                  "costs to show, and whether the site colouring and the co-field inside the "
                  "cell add up (the window is driven without a screen)",
                  "run_cell_view.py", "cell_view", compute=[[]],
                  minutes="3",
                  report=lambda: _mod("run_cell_view").write_table(_r("cell_view")),
                  tables=["cell_view/table.md"]),
        Benchmark("surface_view",
                  "the surface side of the application: what a patch of slab behind the "
                  "molecules costs to draw, whether the height and the tilt the panel states "
                  "agree with what the relaxation wrote, whether the arrows of the vector "
                  "co-field stand on the largest gradients and how far their direction depends "
                  "on the grid, whether the co-pattern chart of a kernel without a "
                  "co-field holds the same numbers as its channel table, and whether the bonds "
                  "the crystal view perceives from the distances are the molecule's own bonds, "
                  "in the open and in a cell that cuts the molecule in two",
                  "run_surface_view.py", "surface_view", compute=[[]],
                  minutes="1",
                  report=lambda: _mod("run_surface_view").write_table(_r("surface_view")),
                  tables=["surface_view/table.md"]),
        Benchmark("reciprocal",
                  "reciprocal-space channels of a crystal against the real-space "
                  "autocorrelation: the typed structure factors added to a short and a long "
                  "cutoff, on the formation energy of the perovskites and the two elastic "
                  "moduli of Matbench, with the identity that ties the two descriptions "
                  "together and the width of the field read as a resolution",
                  "run_reciprocal.py", "reciprocal", compute=[[]],
                  minutes="70",
                  report=lambda: _mod("run_reciprocal").report(),
                  tables=["reciprocal/table.md"]),
        Benchmark("scattering",
                  "a measured scattering curve as the sample: the reduced pair distribution "
                  "function as the untyped version of the periodic autocorrelation, what "
                  "summing away the field types costs on three Matbench tasks, and how much of "
                  "an instrument's blurring, noise and baseline error the model survives",
                  "run_scattering.py", "scattering", compute=[[]],
                  minutes="120",
                  report=lambda: _mod("run_scattering").report(),
                  tables=["scattering/table.md"]),
        Benchmark("microstructure",
                  "a microstructure as the sample: the field-overlap kernels on a periodic "
                  "voxel grid (zero-shift, rotation-averaged, and the overlap of the two-point "
                  "autocorrelations) against the two-point statistics, principal components "
                  "and polynomial or Gaussian-process regression of the MKS homogenisation "
                  "workflow, on the effective conductivity of synthetic two-phase cells at "
                  "training sizes 20 to 200 (the data are generated and solved by the script "
                  "itself; the PyMKS elastic label needs prepare_elastic.py in a PyMKS "
                  "environment); then a property of two scales, the coefficient map over the "
                  "lag vector, and the UHCS steel micrographs against CNN features (downloads "
                  "270 MB into benchmarks/cache/uhcs on the first run; needs torchvision)",
                  "run_microstructure.py", "microstructure",
                  # the pilot (five draws), the gate (twenty draws, equal budgets, the GP
                  # baseline, every label the cache holds), and the other three tests
                  compute=[[],
                           ["--protocol", "gate", "--jobs", "8"],
                           ["--stage", "twoscale", "--protocol", "gate", "--jobs", "8"],
                           ["--stage", "map"],
                           ["--stage", "uhcs", "--protocol", "gate", "--jobs", "8"]],
                  prepare=["prepare_elastic.py benchmarks/cache/microstructure/<set>.npz "
                           "(inside the PyMKS environment)"],
                  minutes="120",
                  report=lambda: _mod("run_microstructure").report(),
                  tables=["microstructure/table.md"]),
        Benchmark("threebody",
                  "three atoms at a time: the triangle around every atom of a cell and the "
                  "signed volume that separates a mirror image, measured on a pair of "
                  "structures a distance histogram cannot tell apart, on three Matbench tasks "
                  "and on the 2DMD defect series where the arrangement is the signal",
                  "run_threebody.py", "threebody", compute=[[]],
                  minutes="90",
                  report=lambda: _mod("run_threebody").report(),
                  tables=["threebody/table.md"]),
        Benchmark("prototypes",
                  "inverse design on prototypes: a candidate generator over a prototype cell, a "
                  "test for which candidates are the same structure, and a loop that proposes, "
                  "computes and refits -- measured on a published pool of perovskites, on the "
                  "2DMD defect series, with the model's ranking ability beside it, and as a "
                  "closed loop with a calculator",
                  "run_prototypes.py", "prototypes", compute=[[]],
                  minutes="60",
                  report=lambda: _mod("run_prototypes").report(),
                  tables=["prototypes/table.md"]),
        Benchmark("fieldae",
                  "a decoder of pseudo-atoms against kernel PCA: can a nonlinear autoencoder "
                  "reconstruct a molecular field, and is its error an applicability domain "
                  "(needs PyTorch and the stored COMPAS-1D kernels of the compas benchmark)",
                  "run_fieldae.py", "fieldae", compute=[[]],
                  minutes="90",
                  report=lambda: _mod("run_fieldae").report(),
                  tables=["fieldae/table.md"]),
        Benchmark("solvent",
                  "is a ligand's water worth a field of its own",
                  "run_solvent.py", "solvent", compute=[[]],
                  minutes="20",
                  report=lambda: _mod("run_solvent").report(path=_r("solvent/table.md")),
                  tables=["solvent/table.md"]),
        Benchmark("md",
                  "does a ligand's trajectory in its pocket beat its crystal pose",
                  "run_md.py", "md",
                  compute=[[], ["--normalisation"], ["--docked"]],
                  minutes="20",
                  report=lambda: _mod("run_md").report(path=_r("md/table.md")),
                  tables=["md/table.md"]),
        Benchmark("trajectory",
                  "a simulation read as a conformational ensemble: what the thinning loses",
                  "run_trajectory.py", "trajectory", compute=[[]],
                  minutes="15",
                  report=lambda: _mod("run_trajectory").report(path=_r("trajectory/table.md")),
                  tables=["trajectory/table.md"]),
        Benchmark("xtb",
                  "a cheap quantum stage over many series: does a GFN2-xTB field help any of "
                  "them",
                  "run_xtb.py", "xtb", compute=[[]],
                  minutes="90",
                  report=lambda: _mod("run_xtb").report(path=_r("xtb/table.md")),
                  tables=["xtb/table.md"]),
        Benchmark("nystrom", "can a low-rank kernel reproduce the exact model",
                  "run_nystrom.py", "nystrom", compute=[[]],
                  minutes="10",
                  report=lambda: _mod("run_nystrom").write_table(_r("nystrom")),
                  tables=["nystrom/table.md"]),
        Benchmark("tasks", "seven endpoints at once, and one that was never measured",
                  "run_tasks.py", "tasks", compute=[[]],
                  minutes="60",
                  report=lambda: _mod("run_tasks").report(_r("tasks")),
                  tables=["tasks/table.md"]),
        Benchmark("screening", "screening measured against inactives that were measured too",
                  "run_screening.py", "screening", compute=[[]],
                  minutes="40",
                  report=lambda: _mod("run_screening").report(_r("screening")),
                  tables=["screening/table.md"]),
        Benchmark("mixing", "a posterior over the field weights, and what it is right about",
                  "run_mixing.py", "mixing", compute=[[]],
                  minutes="30",
                  report=lambda: _mod("run_mixing").report(_r("mixing")),
                  tables=["mixing/table.md"]),
        Benchmark("pooled_mixing", "the field weights of a series learned from the other "
                  "series (a prior pooled across the Mittal sets)",
                  "run_pooled_mixing.py", "pooled_mixing", compute=[[]],
                  minutes="40",
                  report=lambda: _mod("run_pooled_mixing").report(_r("pooled_mixing")),
                  tables=["pooled_mixing/table.md"]),
        Benchmark("transfer", "does the co-field cosine say whether the SAR transfers",
                  "run_transfer.py", "transfer", compute=[[]],
                  minutes="15",
                  report=lambda: _mod("run_transfer").report(_r("transfer")),
                  tables=["transfer/table.md"]),
        Benchmark("kcca", "what the field sees that the fingerprint does not (kernel CCA)",
                  "run_kcca.py", "kcca", compute=[[]],
                  minutes="20",
                  report=lambda: _mod("run_kcca").report(_r("kcca")),
                  tables=["kcca/table.md"],
                  readme=["What the field sees that a fingerprint does not"]),
        Benchmark("kcca_quantum", "what the quantum fields see that shape and charge do not",
                  "run_kcca_quantum.py", "kcca_quantum", compute=[[]],
                  minutes="40",
                  report=lambda: _mod("run_kcca_quantum").report(_r("kcca_quantum")),
                  tables=["kcca_quantum/table.md"],
                  readme=["What the field sees that a fingerprint does not"]),
        Benchmark("kcca_pocket", "what the ligand field carries that its pocket contacts do not",
                  "run_kcca_pocket.py", "kcca_pocket", compute=[[]],
                  minutes="20",
                  report=lambda: _mod("run_kcca_pocket").report(_r("kcca_pocket")),
                  tables=["kcca_pocket/table.md"],
                  readme=["What the field sees that a fingerprint does not"]),
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
        Benchmark("gntk", "the graph neural tangent kernel of the bond graph as a kernel view "
                  "under the weights of the evidence, against the fields alone and against the "
                  "fields with the invariant kernel, on the eight Sutherland sets",
                  "run_gntk.py", "gntk", compute=[[]],
                  minutes="10",
                  report=lambda: _write_table(
                      "gntk", _mod("run_gntk").report(_r("gntk"))),
                  tables=["gntk/table.md"]),
        Benchmark("gntk_compas",
                  "the same view on a random subset of 2,000 molecules of COMPAS-1D",
                  "run_gntk.py", "gntk_compas", compute=[["--compas"]],
                  minutes="130",
                  report=lambda: _write_table(
                      "gntk_compas", _mod("run_gntk").report(_r("gntk_compas"))),
                  tables=["gntk_compas/table.md"]),
        Benchmark("band",
                  "a vibrational band as a local target: the carbonyl stretch of substituted "
                  "benzoic acids from the field around the carbonyl carbon, against the "
                  "atom-centred count and a linear regression on the Hammett constant, with "
                  "the co-field at the ring positions, and with one indicator field per atom "
                  "type on folds that hold out whole acids or whole substituents",
                  "run_band.py", "band", compute=[[]],
                  minutes="3",
                  report=lambda: _mod("run_band").report(_r("band")),
                  tables=["band/table.md"]),
        Benchmark("warped", "a monotone transform of the activity learned with the GP: q2, "
                  "coverage and width of its interval against the plain process and the "
                  "conformal interval, on the eight Sutherland sets",
                  "run_warped.py", "warped", compute=[[]],
                  minutes="35",        # measured: 34, with the warp refitted in every fold
                  report=lambda: _write_table(
                      "warped", _mod("run_warped").report(_r("warped"))),
                  tables=["warped/table.md"]),
        Benchmark("warped_mittal",
                  "the same transform on the 38 Mittal sets, 15 of them small",
                  "run_warped.py", "warped_mittal", compute=[["--mittal"]],
                  minutes="55",        # measured: 55, the warp refitted in every fold
                  report=lambda: _write_table(
                      "warped_mittal", _mod("run_warped").report(_r("warped_mittal"))),
                  tables=["warped_mittal/table.md"]),
        Benchmark("surrogate", "is the predictive standard deviation calibrated, and what "
                  "a tolerance rule does: the model as the fast level of a hierarchy, and the "
                  "same deviation on the low-rank path that never builds the n by n kernel",
                  "run_surrogate.py", "surrogate", compute=[[], ["--hops"], ["--low-rank"]],
                  minutes="15",
                  report=lambda: _write_table(
                      "surrogate", _mod("run_surrogate").report_all(_r("surrogate"))),
                  tables=["surrogate/table.md"]),
        Benchmark("perfield", "one attenuation per field, or one shared by all of them",
                  "run_perfield.py", "perfield", compute=[[]],
                  minutes="10",
                  report=lambda: _mod("run_perfield").report(_r("perfield")),
                  tables=["perfield/table.md"]),
        Benchmark("chirality", "do the signed three-point channels add anything",
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
        Benchmark("fe110", "the planar aligner against the relaxed poses on the Fe(110) "
                           "adsorption sets, and the overlap kernel in the rule frame against the "
                           "surface kernel",
                  "run_fe110.py", "fe110",
                  compute=[["--node", "benzimidazole"], ["--node", "quinoline"]],
                  minutes="25",   # examples/corrosion_fe110, in the repo
                  report=lambda: _mod("run_fe110").report(_r("fe110")),
                  tables=["fe110/table.md"],
                  readme=["Posed by rule: the surface frame without the slab"]),
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
                           ["--sets", "all", "--flex", "--force"],
                           # the files measured before the search was widened: 2 restarts of a
                           # turn of at most 30 degrees, which cannot recover an orientation
                           ["--sets", "all", "--restarts", "2", "--max-rotation-deg", "30",
                            "--force"],
                           ["--sets", "all", "--flex", "--restarts", "2",
                            "--max-rotation-deg", "30", "--force"],
                           ["--sets", "mcl1", "p38", "thrombin", "tnks2", "--pocket", "--force"],
                           ["--sets", "mcl1", "p38", "thrombin", "tnks2", "--pocket",
                            "--restarts", "2", "--max-rotation-deg", "30", "--force"],
                           # the control of the correction: the number of restarts and the
                           # turn they may make, varied separately
                           ["--sets", "mcl1", "thrombin", "--restarts", "20",
                            "--max-rotation-deg", "30", "--force"],
                           ["--sets", "mcl1", "thrombin", "--restarts", "2",
                            "--max-rotation-deg", "180", "--force"],
                           ["--sets", "mcl1", "--pocket", "--restarts", "20",
                            "--max-rotation-deg", "30", "--force"],
                           # how much of the receptor term the pose can stand, at the narrow search
                           ["--sets", "p38", "tnks2", "--pocket", "--pocket-weight", "0",
                            "--restarts", "2", "--max-rotation-deg", "30", "--force"],
                           ["--sets", "p38", "tnks2", "--pocket", "--pocket-weight", "0.25",
                            "--restarts", "2", "--max-rotation-deg", "30", "--force"],
                           ["--sets", "p38", "tnks2", "--pocket", "--pocket-weight", "4",
                            "--restarts", "2", "--max-rotation-deg", "30", "--force"],
                           # the same three weights at the corrected search. The weight changes the
                           # joint aligner and no other method, so only that one is computed here
                           ["--sets", "p38", "tnks2", "--pocket", "--pocket-weight", "0",
                            "--methods", "pocket+variance", "--force"],
                           ["--sets", "p38", "tnks2", "--pocket", "--pocket-weight", "0.25",
                            "--methods", "pocket+variance", "--force"],
                           ["--sets", "p38", "tnks2", "--pocket", "--pocket-weight", "4",
                            "--methods", "pocket+variance", "--force"]],
                  minutes="960",
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
        Benchmark("jarvis2d", "two-dimensional materials: the periodic autocorrelation of the "
                  "1,103 JARVIS-DFT 2D monolayers, flagged as crystals against flagged as layers "
                  "(background per area, no pair across the vacuum) and with the height channels "
                  "of plan item M1, on the band gap, the formation energy and the exfoliation "
                  "energy (downloads the 8 MB data file into benchmarks/cache on the first run)",
                  "run_jarvis2d.py", "jarvis2d", compute=[["--force"]], minutes="60",
                  report=lambda: _mod("run_jarvis2d").report(_r("jarvis2d/table.md")),
                  tables=["jarvis2d/table.md"],
                  readme=["Crystals: the periodic autocorrelation"]),
        Benchmark("compas", "the HOMO-LUMO gap of the polybenzenoid hydrocarbons of COMPAS-1D in a "
                  "frame given by rule: steric, electrostatic, atom-class, ring-class, Clar-sextet and "
                  "HOMA ring fields, the Gaussian kernel layer, baselines, co-fields, per-ring "
                  "contributions and the Shapley reading of the layered model, which has no "
                  "co-field; the layer-width grid measured below its own edge; Delta-learning from the GFN2-xTB gap; NICS(1) of the small molecules "
                  "(ORCA); the landmark model on COMPAS-1x; the peri-condensed set COMPAS-3D and a "
                  "10,000-molecule subset of the hetero-atom set COMPAS-2D (downloads about 400 MB of "
                  "COMPAS into benchmarks/cache on the first run; needs xtb for the charges, or "
                  "--charges gasteiger)",
                  "run_compas.py", "compas",
                  compute=[["--jobs", "12"],
                           ["--stage", "fields", "--sets", "ABCDF,ABCDEF", "--layer", "gauss",
                            "--splits", "random0,random1,random2,size8,size9,size10"],
                           ["--stage", "fields", "--sets", "ABCDE", "--layer", "gauss", "--targets",
                            "DELTA_eV", "--splits", "random0,random1,random2,size8,size9,size10"],
                           ["--stage", "baselines", "--force"],
                           ["--stage", "resolution"],
                           ["--stage", "resolution", "--wide-gammas"],
                           ["--stage", "ringshapley"],
                           ["--stage", "nics", "--jobs", "10"],
                           ["--stage", "scaling", "--set", "1x", "--jobs", "12"],
                           ["--set", "3D", "--stage", "prepare", "--jobs", "12"],
                           ["--set", "3D", "--stage", "kernels"],
                           ["--set", "3D", "--stage", "fields", "--layer", "gauss",
                            "--splits", "random0,random1,random2,size8,size9,size10"],
                           ["--set", "3D", "--stage", "baselines"],
                           ["--set", "3D", "--stage", "interpret"],
                           ["--set", "2D", "--stage", "prepare", "--jobs", "12"],
                           ["--set", "2D", "--stage", "kernels"],
                           ["--set", "2D", "--stage", "fields", "--layer", "gauss",
                            "--splits", "random0,random1,random2,size8,size9,size10"],
                           ["--set", "2D", "--stage", "baselines"]],
                  minutes="900",
                  report=lambda: _mod("run_compas").report(_r("compas/table.md")),
                  tables=["compas/table.md"],
                  readme=["A frame by rule: the gap of polybenzenoid hydrocarbons"]),
        Benchmark("kpls", "kernel partial least squares against kernel ridge regression on the "
                          "Sutherland sets, each choosing its own setting by cross-validation",
                  "run_kpls.py", "kpls", compute=[[]], minutes="30",
                  report=lambda: _mod("run_kpls").report(_r("kpls/table.md")),
                  tables=["kpls/table.md"],
                  readme=["The learners (manual), Learning row of the feature table"]),
        Benchmark("kfda", "kernel Fisher discriminant analysis against C-SVC, the Gaussian-process "
                          "classifier and kernel ridge regression on the LIT-PCBA classification "
                          "sets, and the two rules for where to cut its score on the Sutherland "
                          "sets",
                  "run_kfda.py", "kfda", compute=[[]], minutes="15",
                  report=lambda: _mod("run_kfda").report(_r("kfda/table.md")),
                  tables=["kfda/table.md"],
                  readme=["Classification (manual), Learning row of the feature table"]),
        Benchmark("klr", "kernel logistic regression against the Laplace Gaussian-process "
                         "classifier (same likelihood, amplitude by cross-validation and by the "
                         "marginal likelihood), C-SVC and the kernel Fisher discriminant, on the "
                         "LIT-PCBA classification sets",
                  "run_klr.py", "klr", compute=[[]], minutes="5",
                  report=lambda: _mod("run_klr").report(_r("klr/table.md")),
                  tables=["klr/table.md"],
                  readme=["Classification (manual), Learning row of the feature table"]),
        Benchmark("lapkrr", "Laplacian-regularised kernel ridge regression: molecules without an "
                            "activity brought into the fit, measured on the eight Sutherland "
                            "sets with half the training activities hidden and on the LIT-PCBA "
                            "screening sets with the library unlabelled",
                  "run_lapkrr.py", "lapkrr", compute=[["--part", "all"]], minutes="25",
                  report=lambda: _mod("run_lapkrr").report(_r("lapkrr/table.md")),
                  tables=["lapkrr/table.md"],
                  readme=["Molecules without an activity (manual), Learning row of the "
                          "feature table"]),
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
                    _report(b)
                finally:
                    sys.stdout = out
            else:
                _report(b)
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

def _is_binary(first_bytes) -> bool:
    """Whether a file is binary, by the rule git itself uses: a zero byte near the start.

    Git looks at the first 8000 bytes of a file and treats it as binary if one of them is zero.
    The same rule decides here whether a file is hashed byte for byte or with its newlines
    normalised, so the files whose newlines git converts are exactly the files normalised here.
    """
    return b"\0" in first_bytes[:8000]


def _content_sha256(path, block=1 << 20):
    """The hash of a file's content, with CRLF counted as LF in a text file.

    Git stores a text file with LF newlines and checks it out with CRLF on Windows, so one commit
    gives different bytes on different machines. A file written by a script keeps the newlines
    that script wrote, whatever the platform. Hashing the raw bytes therefore records the history
    of the working tree rather than the content of the commit, and no fresh clone reproduces it.
    Counting CRLF as LF removes both differences: the hash of a text file is the same on every
    platform and for every way the file arrived. A binary file is hashed byte for byte.

    Returns the hash and the number of bytes hashed, which is the file's size with LF newlines.
    """
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        chunk = fh.read(max(block, 8000))
        binary = _is_binary(chunk)
        pending = b""                      # a CR held back: its LF may be in the next block
        while chunk:
            if binary:
                piece = chunk
            else:
                chunk = pending + chunk
                pending = b"\r" if chunk.endswith(b"\r") else b""
                piece = (chunk[:-1] if pending else chunk).replace(b"\r\n", b"\n")
            h.update(piece)
            size += len(piece)
            chunk = fh.read(block)
        if pending:                        # a CR at the end of the file is content, not a newline
            h.update(pending)
            size += 1
    return h.hexdigest(), size


def _text_sha256(path):
    """The hash of a text file with CRLF read as LF: what docs/make_manual.py records for the
    manual, so that a Windows checkout and a Linux one agree on whether the PDF is current."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


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
    data = {}
    for p in tracked("examples"):
        sha, size = _content_sha256(os.path.join(ROOT, p))
        data[p] = {"sha256": sha, "bytes": size}
    results = {}
    for p in tracked("benchmarks/results"):
        results[p] = _content_sha256(os.path.join(ROOT, p))[0]
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hashes": "sha256 of the file content, with CRLF counted as LF in text files; "
                  "'bytes' is the size with LF newlines",
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
        h = _content_sha256(os.path.join(ROOT, p))[0]
        if p not in old["data"]:
            problems.append(f"not in the manifest: {p}")
        elif old["data"][p]["sha256"] != h:
            problems.append(f"changed since the manifest: {p}")
    for p in old["data"]:
        if not os.path.exists(os.path.join(ROOT, p)):
            problems.append(f"in the manifest but gone: {p}")
    return problems


def check_documents() -> List[str]:
    """What the manual is missing: a benchmark it never names, or a PDF older than its Markdown.

    Plan item G4 is a working rule -- a feature lands with its manual section and its benchmark
    script in the same commit -- and this is the part of it a program can check. A benchmark the
    manual never mentions is a measurement nobody can find; a PDF built from an older manual is a
    document that disagrees with the one in the repository.
    """
    problems = []
    manual = os.path.join(ROOT, "docs", "manual.md")
    if not os.path.exists(manual):
        return ["docs/manual.md is missing"]
    text = _read(manual)
    for b in registry():
        if not any(m in text for m in (b.name, f"results/{b.results}/", b.script)):
            problems.append(f"the manual never names the benchmark {b.name}: neither its name, "
                            f"nor results/{b.results}/, nor {b.script}")
    # the stamp docs/make_manual.py writes: comment lines, then "<sha256>  manual.md"
    pdf = os.path.join(ROOT, "docs", "CMF_manual.pdf")
    stamp = os.path.join(ROOT, "docs", "CMF_manual.sha256")
    if not os.path.exists(pdf) or not os.path.exists(stamp):
        problems.append("docs/CMF_manual.pdf or its .sha256 is missing: "
                        "run python docs/make_manual.py")
    else:
        recorded = [ln.split()[0] for ln in _read(stamp).splitlines()
                    if ln.strip() and not ln.startswith("#")]
        if not recorded or recorded[0] != _text_sha256(manual):
            problems.append("docs/CMF_manual.pdf was built from an older docs/manual.md: "
                            "run python docs/make_manual.py")
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
        if b.prepare:
            print(f"{'':16s} {'':8s} {'':6s}  input prepared by: python benchmarks/"
                  + " ".join(b.prepare) + " (not run by --compute)")
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
                    or (b.needs == "orca" and not orca and b.name in ("corrosion_qm", "cdft_kernels", "anisotropic")):
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
        problems = check_manifest() + check_documents()
        problems += ["table or document was not what its result files give: "
                     + os.path.relpath(c, ROOT) for c in changed]
        if problems:
            print("CHECK FAILED")
            for p in problems:
                print("  " + p)
            return 1
        print("check passed: tables and documents match their result files, every benchmark is "
              "named in the manual, the manual PDF is current, data matches the manifest")
    if not (args.compute or args.tables or args.manifest or args.check) or args.list:
        list_benchmarks()
    return 0


if __name__ == "__main__":
    sys.exit(main())
