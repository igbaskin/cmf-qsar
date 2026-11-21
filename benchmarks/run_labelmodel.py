"""
Plan item BT9, gate (b): does a model of the simulated label work for a new solvent mixture?

Item BT9 simulates the salt diffusion coefficient and the conductivity of an electrolyte from
molecular dynamics (``run_diffusion.py``). That costs about 20 minutes of a graphics card per
electrolyte. Gate (b) of the item asks whether a model can give the same label for an
electrolyte whose solvent mixture it has not seen, so that the simulation is needed only for the
training set. The model must state its uncertainty, and the statement must be right.

The data are the 62 LiPF6 electrolytes of CALiSol-23 (``examples/electrolyte/``) in 22 solvent
mixtures, each simulated at its own molality with the same settings: the label of every one is
read from ``benchmarks/results/diffusion/run_*.json``. The description of an electrolyte is the
ensemble of solvation shells that item BT5 sampled for it.

Two labels
----------
``ln D``
    the natural logarithm of the salt diffusion coefficient with the ideal-solution formula,
    2 D+ D- / (D+ + D-), in m2/s. This is the quantity that BT9 supplies to the cell model.
``ln sigma``
    the natural logarithm of the simulated conductivity, in S/m.

Models, all Gaussian processes with the ridge and the kernel chosen inside every fold
----------------------------------------------------------------------------------------
===================  ======================================================================
``conditions``       the molality alone: the control of item BT5
``composition``      the molality and the solvent weight fractions: the control that items
                     BT1 to BT3 earned
``shells``           the sampled shells through the overlap kernel in the cation frame, mixed
                     in equal parts with a kernel on the molality, as item BT5 does
``shells-free``      the same shells through the alignment-free kernel. This is the model of
                     the gate: items BT4 and BT5 found it the better of the two kernels
``reference``        the Nernst-Einstein reference of the plan. It is given the measured
                     conductivity of the test electrolyte (and, for ln D, the concentration
                     of the simulated box): one straight line fitted on the training part. It
                     knows something that no model of a new mixture knows, so a model that
                     beats it has a clear advantage
===================  ======================================================================

The reading of gate (b), fixed before the models were run
----------------------------------------------------------
Folds hold out whole solvent mixtures (the gate) and are repeated with five seeds. The label is
``ln D``. The gate holds if both parts hold for ``shells-free``:

1. *The deviation is calibrated:* the root mean square of the standardised error (z rms) is
   between 0.8 and 1.25.
2. *The model is ahead of the reference:* the 95 % interval of the paired difference of the
   absolute errors, from resampling the solvent mixtures, is above zero.

The comparison with ``composition`` is reported, and it is not part of the gate: the plan does
not expect a static shell to beat the composition on the first reading. The same readings are
reported for ``ln sigma``. Random folds are reported as the easier question.

The value of the simulation
---------------------------
A second question is not part of the gate and matters more for the use. If the solvent mixture
is new, is it worth simulating it? The target is then the measured conductivity of CALiSol-23.
``composition`` predicts it from the composition alone, ``simulated`` from the simulated
conductivity alone (one straight line), and ``composition + simulated`` from both. The gain of
the last over ``composition`` is what the simulation adds.

Stages
------
    python benchmarks/run_labelmodel.py --stage data     # the labels and the kernels
    python benchmarks/run_labelmodel.py --stage gate     # gate (b)
    python benchmarks/run_labelmodel.py --stage value    # what the simulation adds
    python benchmarks/run_labelmodel.py --stage report   # the table
    python benchmarks/run_labelmodel.py                  # all of them
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_cellmodel as RC                                                   # noqa: E402
import run_diffusion as RD                                                   # noqa: E402
import run_shells as RS                                                      # noqa: E402
from cmfqsar.surrogate import Calibration                                    # noqa: E402

RESULTS = os.path.join(HERE, "results", "labelmodel")
CACHE = os.path.join(HERE, "cache", "labelmodel")
SEEDS = (0, 1, 2, 3, 4)
#: the share of the field kernel in the mixed kernels; the rest is the kernel of the molality
MIX = 0.5
#: the calibration the gate asks for, and the interval of the paired difference
Z_LOW, Z_HIGH = 0.8, 1.25
GATE_MODEL = "shells-free"
LABELS = {"ln_D": ("natural logarithm of the salt diffusion coefficient, ideal solution",
                   "ln(m2/s)"),
          "ln_sigma": ("natural logarithm of the simulated conductivity", "ln(S/m)")}
MODELS = ("conditions", "composition", "shells", "shells-free", "reference")


# ------------------------------------------------------------------------------------ the data


def label_table() -> List[Dict[str, object]]:
    """The 62 electrolytes with their simulated labels and their measured conductivity.

    A formulation with two starts (the reference electrolyte, which is the series box) is the
    mean of its starts. The conductivity of CALiSol-23 is in S/m, as in ``run_diffusion.py``.
    """
    measured = {r["id"]: r for r in RD.calisol_rows()}
    names = sorted({n for runs in RD.ALL_FORMULATION_RUNS.values() for n in runs})
    got = {r["run"]: r for r in RD.read_results(names)}
    out = []
    for fid, runs in RD.ALL_FORMULATION_RUNS.items():
        rows = [got[n] for n in runs if n in got]
        if not rows:
            raise SystemExit(f"no simulated result for {fid}: run "
                             "`python benchmarks/run_diffusion.py --stage analyse` first")
        out.append({
            "id": fid, "molality": float(measured[fid]["molality"]),
            "sigma": float(np.mean([r["whole"]["conductivity"]["sigma"] for r in rows])),
            "salt": float(np.mean([r["whole"]["salt"] for r in rows])),
            "molarity": float(np.mean([r["whole"]["molarity"] for r in rows])),
            "measured_sigma": float(measured[fid]["conductivity"]),
            "exponent": float(np.mean([r["whole"]["cation"]["exponent"] for r in rows])),
            "starts": len(rows)})
    return out


def series() -> List[Dict[str, object]]:
    """The electrolytes of item BT5 that have a simulated label, in the order of the labels."""
    labels = {r["id"]: r for r in label_table()}
    rows = [r for r in RS.load_series() if r["id"] in labels]
    missing = sorted(set(labels) - {r["id"] for r in rows})
    if missing:
        raise SystemExit("no shells for " + ", ".join(missing))
    for r in rows:
        r.update(labels[str(r["id"])])
    return rows


def label_vector(rows: Sequence[Dict[str, object]], label: str) -> np.ndarray:
    key = {"ln_D": "salt", "ln_sigma": "sigma"}[label]
    return np.log(np.array([float(r[key]) for r in rows], float))


# ---------------------------------------------------------------------------------- the kernels


def _condition_kernels(Z: np.ndarray) -> Dict[str, np.ndarray]:
    return RC.descriptor_kernels(Z)


def _field_kernels(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, np.ndarray]]:
    """The kernels of the two shell models, mixed with the kernel of the molality.

    They take about 3 minutes to build, so they are kept in ``benchmarks/cache/labelmodel`` under
    the hash of the shell files, the mixture and the field widths. Nothing here reads a label.
    """
    import hashlib

    h = hashlib.sha256(f"{MIX}|{RS.ALPHAS}|{RS.FIELDS}".encode())
    for r in rows:
        with open(str(r["sdf"]), "rb") as fh:
            h.update(str(r["id"]).encode() + fh.read())
        h.update(repr(float(r["molality"])).encode())
    path = os.path.join(CACHE, f"fields_{h.hexdigest()[:20]}.npz")
    if os.path.isfile(path):
        with np.load(path) as z:
            out: Dict[str, Dict[str, np.ndarray]] = {}
            for key in z.files:
                model, name = key.split("|", 1)
                out.setdefault(model, {})[name] = z[key]
            return out
    every = np.arange(len(rows))
    shells = RS.shell_molecules(rows)
    molality = RC.descriptor_kernels(RS.descriptors(rows, "conditions"))["gaussian, width 1"]
    out = {}
    for model in ("shells", "shells-free"):
        table = RS.field_tables(rows, model, shells, {})
        kernels = {}
        for alpha, entry in table.items():
            K = RS._unit_diagonal(RS._kernel_at(entry, every)[0])
            kernels[f"field width {alpha:g}"] = RC.unit_kernel(MIX * K + (1.0 - MIX) * molality)
        out[model] = kernels
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **{f"{m}|{n}": K for m, g in out.items() for n, K in g.items()})
    return out


def model_kernels(rows: Sequence[Dict[str, object]], target: str) -> Dict[str, Dict[str, np.ndarray]]:
    """The whole kernel matrices of every model of one label, by the name of the kernel.

    Nothing here reads a simulated label. A kernel on all 62 rows takes its width from the
    descriptors of all of them, which is unsupervised.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    out["conditions"] = _condition_kernels(RS.descriptors(rows, "conditions"))
    out["composition"] = _condition_kernels(RS.descriptors(rows, "composition"))
    out.update(_field_kernels(rows))
    if target == "ln_D":
        x = np.log([float(r["measured_sigma"]) for r in rows])             - np.log([float(r["molarity"]) for r in rows])
    else:
        x = np.log([float(r["measured_sigma"]) for r in rows])
    out["reference"] = {"line": RC.descriptor_kernels(np.asarray(x, float)[:, None])["linear"]}
    return out


# ------------------------------------------------------------------------------------ the gate


def paired_group_interval(err_a, err_b, groups: Sequence[str], seed: int = 0,
                          draws: int = 2000) -> Dict[str, float]:
    """The mean of ``err_a - err_b`` with a 95 % interval from resampling whole groups.

    A positive mean means that ``b`` has the smaller error. Formulations of one solvent mixture
    share their solvents, so they are resampled together.
    """
    a, b = np.asarray(err_a, float), np.asarray(err_b, float)
    names = sorted(set(map(str, groups)))
    index = {g: [i for i, v in enumerate(groups) if str(v) == g] for g in names}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        pick = rng.integers(0, len(names), len(names))
        idx = np.concatenate([index[names[p]] for p in pick])
        boot.append(float(np.mean(a[idx] - b[idx])))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mean = float(np.mean(a - b))
    return {"mean": mean, "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def run_models(rows: Sequence[Dict[str, object]], y: np.ndarray,
               kernels: Dict[str, Dict[str, np.ndarray]], groups: Sequence[str],
               models: Sequence[str] = MODELS, label: str = "") -> Dict[str, object]:
    """Out-of-fold errors of every model over the seeds, with the calibration of each."""
    out: Dict[str, object] = {}
    for model in models:
        errs, zs, covs, sds, chosen = [], [], [], [], []
        t0 = time.time()
        for seed in SEEDS:
            fit = RC.nested_surrogate(kernels[model], y, groups, seed=seed)
            errs.append(np.abs(fit["predicted"] - y))
            cal: Calibration = fit["calibration"]
            zs.append(cal.z_rms)
            covs.append((cal.coverage_1sd, cal.coverage_2sd))
            sds.append(cal.mean_sd)
            chosen.extend(c["kernel"] for c in fit["chosen"])
        err = np.mean(errs, axis=0)
        out[model] = {"abs_errors": [float(v) for v in err], "mae": float(np.mean(err)),
                      "mae_by_seed": [float(np.mean(e)) for e in errs],
                      "z_rms": float(np.mean(zs)), "z_rms_by_seed": [float(v) for v in zs],
                      "coverage_1sd": float(np.mean([c[0] for c in covs])),
                      "coverage_2sd": float(np.mean([c[1] for c in covs])),
                      "mean_sd": float(np.mean(sds)),
                      "kernels_chosen": {k: chosen.count(k) for k in sorted(set(chosen))}}
        print(f"    {label}{model:<12} MAE {out[model]['mae']:.3f}, z rms "
              f"{out[model]['z_rms']:.2f}, mean sd {out[model]['mean_sd']:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    return out


def gate_stage(rows: List[Dict[str, object]]) -> Dict[str, object]:
    composition = [str(r["composition"]) for r in rows]
    ids = [str(r["id"]) for r in rows]
    splits = {"by-composition": composition, "random": ids}
    out: Dict[str, object] = {"n": len(rows), "seeds": list(SEEDS), "models": {},
                              "comparisons": {}, "gate": {}}
    for target, (tname, unit) in LABELS.items():
        y = label_vector(rows, target)
        kernels = model_kernels(rows, target)
        out.setdefault("spread", {})[target] = float(np.mean(np.abs(y - np.mean(y))))
        for split, groups in splits.items():
            print(f"  {tname}, {split} folds (the mean costs "
                  f"{out['spread'][target]:.3f} {unit})", flush=True)
            recs = run_models(rows, y, kernels, groups)
            key = f"{target}|{split}"
            out["models"][key] = recs
            pairs = {"shells-free vs reference": ("shells-free", "reference"),
                     "shells vs reference": ("shells", "reference"),
                     "shells-free vs composition": ("shells-free", "composition"),
                     "shells vs composition": ("shells", "composition"),
                     "composition vs conditions": ("composition", "conditions"),
                     "shells-free vs conditions": ("shells-free", "conditions"),
                     "the frame on the shells": ("shells", "shells-free")}
            comp = {}
            for name, (a, b) in pairs.items():
                comp[name] = paired_group_interval(recs[b]["abs_errors"], recs[a]["abs_errors"],
                                                   composition)
                iv = comp[name]
                print(f"      {name:<30} {iv['mean']:+.4f} [{iv['low']:+.4f}, {iv['high']:+.4f}]",
                      flush=True)
            out["comparisons"][key] = comp
            if split == "by-composition":
                z = recs[GATE_MODEL]["z_rms"]
                iv = comp[f"{GATE_MODEL} vs reference"]
                out["gate"][target] = {
                    "model": GATE_MODEL, "z_rms": z,
                    "calibrated": bool(Z_LOW <= z <= Z_HIGH),
                    "ahead_of_the_reference": bool(iv["low"] > 0),
                    "holds": bool(Z_LOW <= z <= Z_HIGH and iv["low"] > 0)}
    return out


# ------------------------------------------------------------------------- the value of a simulation


def value_stage(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Does the simulated conductivity add to the composition for the measured conductivity?"""
    composition = [str(r["composition"]) for r in rows]
    ids = [str(r["id"]) for r in rows]
    y = np.log([float(r["measured_sigma"]) for r in rows])
    comp = RC.descriptor_kernels(RS.descriptors(rows, "composition"))
    cond = RC.descriptor_kernels(RS.descriptors(rows, "conditions"))
    sim_x = np.log([float(r["sigma"]) for r in rows])[:, None]
    sim = RC.descriptor_kernels(sim_x)
    kernels = {
        "conditions": cond,
        "composition": comp,
        "simulated": {"line": sim["linear"]},
        "composition + simulated": {f"{k} + simulated line": RC.unit_kernel(comp[k] + sim["linear"])
                                    for k in comp}}
    out: Dict[str, object] = {"n": len(rows), "models": {}, "comparisons": {}}
    spread = float(np.mean(np.abs(y - np.mean(y))))
    out["spread"] = spread
    for split, groups in {"by-composition": composition, "random": ids}.items():
        print(f"  the measured conductivity, {split} folds (the mean costs {spread:.3f} ln(S/m))",
              flush=True)
        recs = run_models(rows, y, kernels, groups, models=tuple(kernels), label="")
        out["models"][split] = recs
        pairs = {"composition + simulated vs composition":
                 ("composition + simulated", "composition"),
                 "simulated vs composition": ("simulated", "composition"),
                 "composition vs conditions": ("composition", "conditions"),
                 "composition + simulated vs simulated": ("composition + simulated", "simulated")}
        comp_out = {}
        for name, (a, b) in pairs.items():
            comp_out[name] = paired_group_interval(recs[b]["abs_errors"], recs[a]["abs_errors"],
                                                   composition)
            iv = comp_out[name]
            print(f"      {name:<40} {iv['mean']:+.4f} [{iv['low']:+.4f}, {iv['high']:+.4f}]",
                  flush=True)
        out["comparisons"][split] = comp_out
    return out


# ----------------------------------------------------------------------------------- the report


def _f(v, digits=3):
    return f"{v:.{digits}f}"


def write_table(results: str = RESULTS) -> str:
    gate = _read(results, "gate")
    value = _read(results, "value")
    data = _read(results, "data")
    out: List[str] = ["# Item BT9, gate (b): a model of the simulated label", ""]
    w = out.append
    w("The label of every one of the 62 electrolytes of CALiSol-23 comes from its own simulation "
      "(`run_diffusion.py`). A model of the label is read on solvent mixtures that it has not "
      "seen: the folds hold out whole mixtures, and five seeds of the folds are averaged. The "
      "errors are in the natural logarithm of the label, so an error of 0.1 is about 10 %. "
      "The reading rule of the gate was fixed before the models were run (see the docstring of "
      "`benchmarks/run_labelmodel.py`).")
    w("")
    w("The labels have a scatter that no model can remove. One start of 4 ns scatters by "
      f"{100 * data['scatter']:.0f} % in the logarithm of the conductivity and by "
      f"{100 * data['scatter_D']:.0f} % in the logarithm of the salt diffusion coefficient. "
      "Both come from the six pairs of starts of the concentration series. A model that "
      "reproduced the true value exactly would therefore have a mean absolute error of about "
      f"{_f(0.8 * data['scatter'])} for the conductivity and about "
      f"{_f(0.8 * data['scatter_D'])} for the diffusion coefficient.")
    w("")
    for target, (tname, unit) in LABELS.items():
        w(f"## {tname[0].upper() + tname[1:]}")
        w("")
        for split, head in (("by-composition", "New solvent mixtures (the gate)"),
                            ("random", "Random folds, the easier question")):
            key = f"{target}|{split}"
            if key not in gate["models"]:
                continue
            w(f"**{head}.** Predicting the mean costs {_f(gate['spread'][target])} {unit}.")
            w("")
            w("| Model | Mean absolute error | Standardised error, root mean square | "
              "Inside one standard deviation | Mean standard deviation |")
            w("|---|---:|---:|---:|---:|")
            for m in MODELS:
                r = gate["models"][key][m]
                w(f"| {m} | {_f(r['mae'])} | {_f(r['z_rms'], 2)} | "
                  f"{100 * r['coverage_1sd']:.0f} % | {_f(r['mean_sd'])} |")
            w("")
            w("A positive difference means that the first model has the smaller error. The "
              "interval is 95 %, from resampling solvent mixtures.")
            w("")
            w("| Comparison | Difference of the mean absolute error | Interval |")
            w("|---|---:|---|")
            for name, iv in gate["comparisons"][key].items():
                w(f"| {name} | {iv['mean']:+.3f} | {iv['low']:+.3f} to {iv['high']:+.3f}"
                  f"{'' if iv['excludes_zero'] else ' (includes zero)'} |")
            w("")
        g = gate["gate"][target]
        w(f"**Gate (b) for this label, model {g['model']}.** The deviation is "
          + ("calibrated" if g["calibrated"] else "not calibrated")
          + f" (standardised error {_f(g['z_rms'], 2)}, asked for {Z_LOW} to {Z_HIGH}). The model is "
          + ("ahead of" if g["ahead_of_the_reference"] else "not clearly ahead of")
          + " the reference with the measured conductivity. The gate "
          + ("holds." if g["holds"] else "does not hold."))
        w("")
        vs = gate["comparisons"][f"{target}|by-composition"][f"{GATE_MODEL} vs composition"]
        comp = gate["models"][f"{target}|by-composition"]["composition"]
        if vs["high"] < 0:
            w(f"**The composition is the better model.** The composition alone has a smaller error "
              f"than the shells, by {_f(-vs['mean'])} ({_f(-vs['high'])} to {_f(-vs['low'])}). "
              "The description by shells adds nothing to it, as the plan expected. Its "
              f"standardised error is {_f(comp['z_rms'], 2)} (asked for {Z_LOW} to {Z_HIGH}); "
              "a value below the range means that the model is less sure than it could be.")
            w("")
    w("## What the simulation adds to the composition")
    w("")
    w("The target is the measured conductivity of CALiSol-23, in S/m, as the natural logarithm. "
      "The test mixture is new. Predicting the mean costs "
      f"{_f(value['spread'])} ln(S/m).")
    w("")
    for split, head in (("by-composition", "New solvent mixtures"),
                        ("random", "Random folds")):
        if split not in value["models"]:
            continue
        w(f"**{head}.**")
        w("")
        w("| Model | Mean absolute error | Standardised error, root mean square |")
        w("|---|---:|---:|")
        for m, r in value["models"][split].items():
            w(f"| {m} | {_f(r['mae'])} | {_f(r['z_rms'], 2)} |")
        w("")
        w("| Comparison | Difference of the mean absolute error | Interval |")
        w("|---|---:|---|")
        for name, iv in value["comparisons"][split].items():
            w(f"| {name} | {iv['mean']:+.3f} | {iv['low']:+.3f} to {iv['high']:+.3f}"
              f"{'' if iv['excludes_zero'] else ' (includes zero)'} |")
        w("")
    path = os.path.join(results, "table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def _read(base: str, name: str) -> Dict[str, object]:
    with open(os.path.join(base, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def save(name: str, payload: Dict[str, object], base: str = RESULTS) -> None:
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


def data_stage(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """The labels as they are used, and the scatter of one start from the series boxes."""
    name_runs = [r for v in RD.CONCENTRATION_RUNS.values() for r in v]
    ser = RD.read_results(name_runs)
    scatter = RD.start_scatter(ser)
    sim = np.array([float(r["sigma"]) for r in rows])
    meas = np.array([float(r["measured_sigma"]) for r in rows])
    by_run = {r["run"]: r for r in ser}
    d = [np.log(by_run[a]["whole"]["salt"] / by_run[b]["whole"]["salt"])
         for a, b in RD.CONCENTRATION_RUNS.values() if a in by_run and b in by_run]
    scatter_d = float(np.sqrt(np.mean(np.square(d)) / 2.0)) if d else float("nan")
    return {"n": len(rows), "scatter": float(scatter) if scatter is not None else float("nan"),
            "scatter_D": scatter_d,
            "spearman_simulated_measured": RD.spearman(meas, sim),
            "short": [str(r["id"]) for r in rows if float(r["exponent"]) < 0.8],
            "rows": [{k: r[k] for k in ("id", "composition", "molality", "sigma", "salt",
                                        "molarity", "measured_sigma", "exponent", "starts")}
                     for r in rows]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("data", "gate", "value", "report", "all"),
                    default="all")
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args(argv)
    stages = ("data", "gate", "value", "report") if args.stage == "all" else (args.stage,)
    rows = series() if any(s != "report" for s in stages) else []
    for stage in stages:
        if stage == "data":
            save("data", data_stage(rows), args.results)
        elif stage == "gate":
            save("gate", gate_stage(rows), args.results)
        elif stage == "value":
            save("value", value_stage(rows), args.results)
        else:
            print("the table:", write_table(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
