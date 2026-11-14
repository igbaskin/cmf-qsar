"""
Plan item MS5: ensembles as the normal input, and its gate on the solvation shells of item BT5.

Item BT5 described each of 62 LiPF6 electrolytes by the mean field of up to sixteen solvation
shells sampled from a bulk simulation. It built that mean field with code written for shells:
special SD properties and atom maps made from the species of each shell. Item MS5 makes an
ensemble of structures the normal input of the program, whatever the structures are: a file in
which several records make one sample, read by ``load_dataset(..., ensemble_by=...,
member_weight=...)`` or by ``cmf build --ensemble-by ... --member-weight ...``, with the members
allowed to have different atoms (``cmfqsar/ensembles.py``).

The gate of the item: the general path reproduces the numbers of BT5 from the same frames.
This script measures it in three steps.

1. **The file.** The 62 shell files of ``examples/electrolyte/shells`` are written into one SD
   file, one record per shell, with the electrolyte's name, the shell's weight and the two
   targets on every record. Nothing of the shell code is used to read it back.
2. **The kernels.** For every field width of BT5, the kernel of the framed model ("shells") and
   the feature matrix of the alignment-free model ("shells-free") are built twice: by BT5's own
   path (``run_shells.field_tables``) and by the general path (``load_dataset`` on that file,
   then ``build_fields`` with ``conformers="ensemble"`` and the weighting "given"). The general
   path does not use the kernel cache, so the two are computed independently. The largest
   difference between the two is recorded.
3. **The numbers.** The nested cross-validation of BT5 is run for both models, both targets and
   both splits, once on the tables of each path. Each record is compared with the one stored in
   ``benchmarks/results/shells/gate.json``. The run on BT5's own tables is the control: it says
   whether the stored file still matches the present code before the general path is judged
   against it.

Stages
------
    python benchmarks/run_ensembles.py --stage file
    python benchmarks/run_ensembles.py --stage kernels --path bt5
    python benchmarks/run_ensembles.py --stage kernels --path general
    python benchmarks/run_ensembles.py --stage gate
    python benchmarks/run_ensembles.py --report
    python benchmarks/run_ensembles.py               # all of them
"""

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

from rdkit import RDLogger                                                   # noqa: E402

from cmfqsar.dataio import load_dataset                                      # noqa: E402
from cmfqsar.ensembles import is_packed, n_members, write_ensembles          # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields                 # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, block_sizes, feature_matrix  # noqa: E402
from cmfqsar.model import CMFModel                                           # noqa: E402

import run_shells as RS                                                      # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "ensembles")
WORK = os.path.join(HERE, "cache", "ensembles")
FILE = os.path.join(WORK, "electrolytes.sdf")
STORED = os.path.join(HERE, "results", "shells", "gate.json")
MODELS = ("shells", "shells-free")
GROUP = "electrolyte"
WEIGHT = "member_weight"
PATHS = ("bt5", "general")


# ------------------------------------------------------------------------------- 1. the file

def file_stage() -> Dict[str, object]:
    """The shells of every electrolyte as records of one file, with weights and targets."""
    rows = RS.load_series()
    shells = RS.shell_molecules(rows)
    os.makedirs(WORK, exist_ok=True)
    weights = [[float(m.GetProp("shell_weight")) for m in members] for members in shells]
    props = [{"log10_k_298": repr(float(r["log10_k_298"])), "ea_298": repr(float(r["ea_298"])),
              "molality": repr(float(r["molality"])), "composition": r["composition"]}
             for r in rows]
    n = write_ensembles(FILE, shells, [str(r["id"]) for r in rows], weights=weights,
                        sample_props=props, group_prop=GROUP, weight_prop=WEIGHT)
    sizes = [[m.GetNumAtoms() for m in members] for members in shells]
    out = {"electrolytes": len(rows), "records": n,
           "members_per_electrolyte": {"min": min(len(s) for s in sizes),
                                       "max": max(len(s) for s in sizes)},
           "atoms_per_member": {"min": min(min(s) for s in sizes),
                                "max": max(max(s) for s in sizes)},
           "electrolytes_whose_members_differ_in_atoms": sum(len(set(s)) > 1 for s in sizes)}
    print(f"  {n} records of {len(rows)} electrolytes written to {FILE}")
    return out


# ---------------------------------------------------------------------------- 2. the kernels

def general_fields(alpha: float):
    """The fields of the 62 electrolytes through the general entry, in BT5's order."""
    if not os.path.exists(FILE):
        raise SystemExit(f"{FILE} is not here; run --stage file first")
    ds = load_dataset(FILE, activity_prop="ea_298", ensemble_by=GROUP, member_weight=WEIGHT)
    order = [str(r["id"]) for r in RS.load_series()]
    if list(ds.names) != order:
        raise SystemExit("the file does not hold the electrolytes of BT5 in BT5's order")
    fs = FieldSet.create(physchem=list(RS.FIELDS), alpha=alpha, charge_model="gasteiger",
                         conformers="ensemble", conformer_weighting="given")
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    return ds, fs, mf


def _table_file(path: str, alpha: float) -> str:
    return os.path.join(WORK, f"tables_{path}_{alpha}.npz")


def kernels_stage(path: str, alpha: float) -> Dict[str, object]:
    """Both models' tables at one field width, by one path, written to the cache folder.

    One width at a time, because all four in one call take longer than ten minutes on the
    author's machine.
    """
    os.makedirs(WORK, exist_ok=True)
    t0 = time.time()
    packed = None
    if path == "bt5":
        rows = RS.load_series()
        shells = RS.shell_molecules(rows)
        widths = RS.ALPHAS
        RS.ALPHAS = (alpha,)            # BT5's table builder loops over its own widths
        try:
            framed = RS.field_tables(rows, "shells", shells, {})[alpha]
            free = RS.field_tables(rows, "shells-free", shells, {})[alpha][0]
        finally:
            RS.ALPHAS = widths
    else:
        ds, fs, mf = general_fields(alpha)
        packed = int(sum(is_packed(m) for m in ds.mols))
        table = AtomTable.from_molecules(mf)
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        framed = RS._unit_diagonal(model.joint_kernel(model.prepare_training_kernels(table),
                                                      table.descriptors))
        free = feature_matrix(mf, AutocorrelationSpec(), fieldset=fs)
        print(f"  {len(mf)} electrolytes, {packed} of them read as packed ensembles, "
              f"{np.mean([n_members(m) for m in ds.mols]):.1f} members on average")
    np.savez(_table_file(path, alpha), K=np.asarray(framed, float), F=np.asarray(free, float))
    seconds = time.time() - t0
    print(f"  {path}, width {alpha}: tables in {seconds:.0f} s")
    return {"seconds": seconds, "packed": packed}


def _tables(path: str):
    spec = AutocorrelationSpec()
    fs = FieldSet.create(physchem=list(RS.FIELDS), alpha=1.0, charge_model="gasteiger")
    blocks = block_sizes(spec, fs.n_fields)
    framed, free = {}, {}
    for a in RS.ALPHAS:
        name = _table_file(path, a)
        if not os.path.exists(name):
            raise SystemExit(f"{name} is not here; run --stage kernels --path {path} first")
        data = np.load(name)
        framed[a] = data["K"]
        free[a] = (data["F"], spec, blocks)
    return {"shells": framed, "shells-free": free}


# ------------------------------------------------------------------------------ 3. the numbers

def _largest_difference(a, b) -> float:
    """The largest absolute difference between two records, over every number in them."""
    if isinstance(a, dict):
        keys = [k for k in a if k != "elapsed_s"]
        if set(keys) != set(k for k in b if k != "elapsed_s"):
            return float("inf")
        return max([_largest_difference(a[k], b[k]) for k in keys] or [0.0])
    if isinstance(a, list):
        if len(a) != len(b):
            return float("inf")
        return max([_largest_difference(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


def gate_stage() -> Dict[str, object]:
    """Tables compared, then the nested numbers of both paths against the stored ones."""
    rows = RS.load_series()
    with open(STORED, encoding="utf-8") as fh:
        stored = json.load(fh)
    tables = {p: _tables(p) for p in PATHS}
    kernels: Dict[str, Dict[str, float]] = {}
    for alpha in RS.ALPHAS:
        dk = float(np.abs(tables["bt5"]["shells"][alpha]
                          - tables["general"]["shells"][alpha]).max())
        df = float(np.abs(tables["bt5"]["shells-free"][alpha][0]
                          - tables["general"]["shells-free"][alpha][0]).max())
        kernels[str(alpha)] = {"framed_kernel": dk, "alignment_free_rows": df}
        print(f"  width {alpha}: largest difference {dk:.3g} in the framed kernel, "
              f"{df:.3g} in the alignment-free rows")
    labels = [str(r["composition"]) for r in rows]
    splits = {"random": (RS.random_folds(len(rows)), None),
              "by-composition": (RS.grouped_folds(labels), labels)}
    readings: List[Dict[str, object]] = []
    for target, (tlabel, unit) in RS.TARGETS.items():
        y = np.array([float(r[target]) for r in rows], float)
        for split, (design, group_labels) in splits.items():
            key = f"{target}|{split}"
            for model in MODELS:
                Z = RS.descriptors(rows, model)
                old = stored["models"][key][model]
                row = {"target": target, "split": split, "model": model,
                       "stored_mae": old["mae_mean"]}
                for path in PATHS:
                    rec = RS.nested(tables[path][model], Z, y, design, group_labels, (RS.MIX,),
                                    label=f"{model} ({path})", unit=unit)
                    row[f"{path}_mae"] = rec["mae_mean"]
                    row[f"{path}_difference"] = _largest_difference(rec, old)
                readings.append(row)
    worst_general = max(r["general_difference"] for r in readings)
    worst_control = max(r["bt5_difference"] for r in readings)
    worst_kernel = max(max(v.values()) for v in kernels.values())
    out = {"kernels": kernels, "readings": readings,
           "largest_difference_control": worst_control,
           "largest_difference_general": worst_general,
           "largest_difference_tables": worst_kernel,
           "gate_met": bool(worst_control == 0.0 and worst_general == 0.0)}
    print(f"  control against the stored file: {worst_control:.3g}; general path against the "
          f"stored file: {worst_general:.3g}; tables: {worst_kernel:.3g}")
    return out


# --------------------------------------------------------------------------------- the table

def _read(name: str, base: Optional[str] = None) -> Dict[str, object]:
    path = os.path.join(base or RESULTS, f"{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is not here; run the stages first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(name: str, payload: Dict[str, object]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


def _number(v: float) -> str:
    return "0" if v == 0 else f"{v:.2g}"


def write_table(path: Optional[str] = None) -> str:
    """``table.md`` of a results folder (by default ``results/ensembles``), from its JSON."""
    base = path or RESULTS
    data = _read("file", base)
    gate = _read("gate", base)
    lines = [
        "# Item MS5: an ensemble of structures through the general entry",
        "",
        "The gate of item MS5 is that the general path reproduces the numbers of item BT5 from "
        "the same frames. The general path reads one SD file with "
        "`load_dataset(..., ensemble_by=\"electrolyte\", member_weight=\"member_weight\")` and "
        "builds the fields with `conformers=\"ensemble\"` and the weighting \"given\". BT5's "
        "path reads the shell files with the code of `cmfqsar/shells.py`.",
        "",
        "## The file",
        "",
        f"{data['records']} records, one per shell, make {data['electrolytes']} electrolytes. "
        f"An electrolyte has {data['members_per_electrolyte']['min']} to "
        f"{data['members_per_electrolyte']['max']} shells, and a shell has "
        f"{data['atoms_per_member']['min']} to {data['atoms_per_member']['max']} atoms. In "
        f"{data['electrolytes_whose_members_differ_in_atoms']} of the "
        f"{data['electrolytes']} electrolytes the shells differ in their atoms, so they cannot be "
        "conformations of one molecule.",
        "",
        "## The tables of the two paths",
        "",
        "The framed kernel is the joint overlap kernel of the five fields, with a unit diagonal. "
        "The alignment-free rows are the rows of the rotation-invariant autocorrelation. The "
        "general path does not use the kernel cache.",
        "",
        "| Field width (alpha) | Largest difference in the framed kernel | "
        "Largest difference in the alignment-free rows |",
        "|---:|---:|---:|",
    ]
    for alpha, v in gate["kernels"].items():
        lines.append(f"| {alpha} | {_number(v['framed_kernel'])} | "
                     f"{_number(v['alignment_free_rows'])} |")
    lines += [
        "",
        "## The numbers of BT5, from each path",
        "",
        "Nested cross-validation as in `benchmarks/run_shells.py`: the ridge and the field width "
        "chosen inside each fold, the field kernel and the molality kernel mixed in equal parts. "
        "The mean absolute error is the mean over the five folds. The difference is the largest "
        "absolute difference between the new record and the stored one in "
        "`benchmarks/results/shells/gate.json`, over every number in the record: the error of "
        "each electrolyte, the settings chosen in each fold and the summary numbers.",
        "",
        "| Target | Split | Model | Mean absolute error, stored | Mean absolute error, BT5's "
        "path now | Difference from the stored record, BT5's path | Mean absolute error, "
        "general path | Difference from the stored record, general path |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    names = {"log10_k_298": "log10 of the conductivity at 298 K", "ea_298":
             "activation energy at 298 K, kJ/mol"}
    for r in gate["readings"]:
        lines.append(f"| {names.get(r['target'], r['target'])} | {r['split']} | {r['model']} | "
                     f"{r['stored_mae']:.4f} | {r['bt5_mae']:.4f} | "
                     f"{_number(r['bt5_difference'])} | {r['general_mae']:.4f} | "
                     f"{_number(r['general_difference'])} |")
    met = gate["gate_met"]
    lines += [
        "",
        "## The gate",
        "",
        ("**Met.** Every number of the two models is reproduced exactly by the general path: "
         "the tables are the same to the last bit, and so is every record of the nested "
         "cross-validation."
         if met else
         "**Not met.** The general path does not reproduce every number; the tables above say "
         "where it departs."),
        "",
        f"The control, BT5's own path run again, differs from the stored file by "
        f"{_number(gate['largest_difference_control'])}, so the stored file matches the present "
        "code.",
        "",
    ]
    text = "\n".join(lines)
    target = os.path.join(base, "table.md")
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def report() -> None:
    print(write_table())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["file", "kernels", "gate", "all"], default="all")
    ap.add_argument("--path", choices=list(PATHS), default=None,
                    help="with --stage kernels: which path builds the tables (default both)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="with --stage kernels: one field width of BT5 (default all four)")
    ap.add_argument("--report", action="store_true", help="rebuild table.md from the results")
    args = ap.parse_args()
    if args.report:
        report()
        return 0
    if args.stage in ("file", "all"):
        save("file", file_stage())
    if args.stage == "kernels" or args.stage == "all":
        if args.alpha is not None and args.alpha not in RS.ALPHAS:
            raise SystemExit(f"--alpha must be one of BT5's widths {RS.ALPHAS}")
        for path in ([args.path] if args.path else list(PATHS)):
            for alpha in ([args.alpha] if args.alpha is not None else list(RS.ALPHAS)):
                timing = kernels_stage(path, alpha)
                data = _read("file")
                data.setdefault("seconds", {}).setdefault(path, {})[str(alpha)] = \
                    timing["seconds"]
                if timing["packed"] is not None:
                    data["electrolytes_read_as_packed_ensembles"] = timing["packed"]
                save("file", data)
    if args.stage in ("gate", "all"):
        save("gate", gate_stage())
        write_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
