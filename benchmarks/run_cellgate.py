"""
Plan item BT9, gate (c): does the predicted diffusion coefficient hold the capacity of the cell?

Gate (c) of item BT9 asks that, in the stress stage of item MS9, the interval of the capacity that
follows from the predicted salt diffusion coefficient holds the capacity of the measured
electrolyte under the eight conditions. This benchmark makes that test with the model of gate (b).

What is predicted
-----------------
The test electrolyte is LiPF6 in EC:EMC 3:7, the electrolyte of the cell. Every formulation of
that solvent mixture is held out. A Gaussian process on the molality and the solvent weight
fractions (the model ``composition`` of ``run_labelmodel.py``, with the kernel and the ridge
chosen inside the training part on folds that hold out whole mixtures) is fitted to the natural
logarithm of the simulated salt diffusion coefficient of the other 21 mixtures. It predicts that
logarithm and its standard deviation for EC:EMC 3:7 at a grid of concentrations from 0.1 to
3.0 mol/L. The molality of the grid is read from the molarity of the boxes of the concentration
series.

The salt diffusion coefficient that the cell needs is the ideal one multiplied by the
thermodynamic factor, which item BT11 does not yet predict, so the measured factor of the
published set is used, as the plan says. The predicted coefficient is then a ratio to the
measured one at every concentration, and the cell is run with the measured function multiplied
by that ratio. The temperature dependence is that of the measured function: the label is made at
298 K only, so the activation energy of this coefficient is not predicted. In the cold cells the
test is therefore of the level of the coefficient and not of its temperature dependence.

The reading of the gate, fixed before the cell was run
------------------------------------------------------
Everything that is drawn in the model is one standard normal number z, because one coefficient is
predicted and the draws at the different concentrations are taken as fully correlated. The
capacity is therefore a function of z, and the 95 % interval is the capacity at z = -1.96 and at
z = +1.96; no sampling is needed. For each of the eight conditions of the stress stage (the full
model at 1, 1.5, 2 and 2.5C at 298 K, and at 263 K and 253 K at 1 and 1.5C):

* the **truth** is the capacity of the cell with all four transport parameters of the electrolyte
  as Landesfeind and Gasteiger measured them (stored by the stress stage as ``all_four``);
* the interval **holds** the truth if the truth lies between its ends;
* it is **useful** if it is narrower than 5 % of the capacity of the published cell, the width
  that item MS9 calls useful.

Gate (c) holds if the interval holds the truth under all eight conditions. Usefulness is
reported and is not part of the gate. Two intervals are read. *Reading A* uses the standard
deviation of the model alone. *Reading B* adds the deviation of the label from the measurement,
the standard deviation of the logarithm of the ratio of the simulated to the measured
coefficient over the six concentrations of the series (gate (a)), because the model predicts the
simulation and not the measurement. Gate (c) is read on A; B shows what the label's own
error does.

Stages
------
    python benchmarks/run_cellgate.py --stage predict    # the coefficient, 1 minute
    python benchmarks/run_cellgate.py --stage cell       # the cell, needs pybamm, minutes
    python benchmarks/run_cellgate.py --stage report     # the table
    python benchmarks/run_cellgate.py                    # all of them
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
import run_labelmodel as LM                                                  # noqa: E402
import run_shells as RS                                                      # noqa: E402
from cmfqsar.surrogate import Surrogate                                      # noqa: E402

RESULTS = os.path.join(HERE, "results", "cellgate")
#: the test mixture: the solvent weight fractions of the cell's electrolyte
TEST_SOLVENTS = {"EC": 0.3, "EMC": 0.7}
TEST_PREFIX = "LiPF6_EC30_EMC70_"
#: the concentrations of the grid, in mol/m3, and the ones that close it on both sides
GRID = (100.0, 250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 2000.0, 2500.0, 3000.0)
#: the two-sided 95 % interval of a standard normal number, and the points that are run
Z95 = 1.96
Z_POINTS = (-Z95, 0.0, Z95)
LAMS = tuple(10.0 ** e for e in (-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0))
INNER = 4
SEED = 0


# ------------------------------------------------------------------------------ the prediction


def molality_at(molarity: float, series: Sequence[dict]) -> float:
    """The molality of EC:EMC 3:7 at a molarity, interpolated between the boxes of the series."""
    pts = sorted((float(r["molarity"]), float(r["realised_molality"])) for r in series)
    return float(np.interp(molarity, [p[0] for p in pts], [p[1] for p in pts]))


def label_deviation(series: Sequence[dict]) -> float:
    """The standard deviation of the logarithm of the ratio of the simulated to the measured salt
    diffusion coefficient (with the measured thermodynamic factor) over the series boxes."""
    by_conc = {}
    for key, names in RD.CONCENTRATION_RUNS.items():
        got = [r for r in series if r["run"] in names]
        if got:
            by_conc[key] = float(np.mean([r["ratio_with_factor"] for r in got]))
    return float(np.std(np.log(list(by_conc.values())), ddof=1))


def fit_best(kernels: Dict[str, np.ndarray], y: np.ndarray, train: np.ndarray,
             groups: Sequence[str]):
    """The kernel and the ridge that predict the held-out mixtures of the training part best."""
    best = None
    inner = RC.group_parts([groups[i] for i in train], INNER, SEED + 1)
    for name, K in kernels.items():
        for lam in LAMS:
            pred = np.full(len(train), np.nan)
            for part in inner:
                tr = np.setdiff1d(np.arange(len(train)), part)
                s = Surrogate.fit(K[np.ix_(train[tr], train[tr])], y[train[tr]], lam)
                pred[part], _ = s.predict(K[np.ix_(train[part], train[tr])],
                                          np.diag(K)[train[part]])
            err = float(np.sqrt(np.mean((y[train] - pred) ** 2)))
            if best is None or err < best[0]:
                best = (err, name, lam)
    return best


def predict_stage() -> Dict[str, object]:
    from cmfqsar import cellmodel as CM

    rows = LM.series()
    ids = [str(r["id"]) for r in rows]
    test = np.array([i.startswith(TEST_PREFIX) for i in ids])
    if not test.any():
        raise SystemExit(f"no formulation starts with {TEST_PREFIX}")
    train = np.where(~test)[0]
    y = LM.label_vector(rows, "ln_D")
    series_rows = RD.read_results([n for v in RD.CONCENTRATION_RUNS.values() for n in v])
    four = CM.measured_electrolyte(coldest=None)
    grid_rows, grid_c = [], []
    for c in GRID:
        m = molality_at(c / 1000.0, series_rows)
        grid_rows.append({"molality": m, "solvents": dict(TEST_SOLVENTS)})
        grid_c.append(c)
    every = list(rows) + grid_rows
    Z = RS.descriptors(every, "composition")
    kernels = RC.descriptor_kernels(Z)
    groups = [str(r["composition"]) for r in rows]
    err, kname, lam = fit_best(kernels, y, train, groups)
    K = kernels[kname]
    s = Surrogate.fit(K[np.ix_(train, train)], y[train], lam)
    te = np.arange(len(rows), len(every))
    mean, sd = s.predict(K[np.ix_(te, train)], np.diag(K)[te])
    out = {"kernel": kname, "ridge": lam, "inner_rmse": err, "n_train": int(len(train)),
           "held_out": [ids[i] for i in np.where(test)[0]], "grid": []}
    for c, mu, sg, row in zip(grid_c, mean, sd, grid_rows):
        tdf = RC.electrolyte_value(four[RC.THERMODYNAMIC], 298.15, c)
        d_meas = RC.electrolyte_value(four[RC.DIFFUSIVITY], 298.15, c)
        out["grid"].append({"c": float(c), "molality": float(row["molality"]),
                            "ln_d_ideal": float(mu), "sd": float(sg),
                            "tdf_measured": float(tdf), "d_measured": float(d_meas),
                            "d_predicted": float(np.exp(mu) * tdf),
                            "ln_ratio": float(np.log(np.exp(mu) * tdf / d_meas))})
    out["label_deviation"] = label_deviation(
        [r for r in series_rows if "ratio_with_factor" in r])
    for g in out["grid"]:
        print(f"  c {g['c']:6.0f}: predicted {g['d_predicted']:.3e}, measured "
              f"{g['d_measured']:.3e}, ratio {np.exp(g['ln_ratio']):.2f}, sd of ln {g['sd']:.3f}",
              flush=True)
    print(f"  kernel {kname}, ridge {lam:g}; deviation of the label {out['label_deviation']:.2f}",
          flush=True)
    return out


# ---------------------------------------------------------------------------------- the cell


def _scaled(function, c_grid: np.ndarray, ratio: np.ndarray):
    """`function(c, T)` multiplied by a ratio that is a piecewise linear function of the
    concentration, flat beyond the grid."""
    from cmfqsar import cellmodel as CM

    pybamm = CM._pybamm()
    x = np.concatenate([[0.0], c_grid, [10000.0]])
    y = np.concatenate([[ratio[0]], ratio, [ratio[-1]]])

    def scaled(c_e, T):
        return function(c_e, T) * pybamm.Interpolant(x, y, c_e, name="predicted ratio",
                                                     interpolator="linear")
    return scaled


def read_interval(low: float, median: float, high: float, truth: float,
                  published: float) -> Dict[str, object]:
    """One interval read against the truth: whether it holds it, whether it is useful (narrower
    than :data:`run_cellmodel.TOL_SHARE` of the published capacity), and how far its median is
    from the truth as a share of the published capacity. A capacity that is not a number (the
    cell did not finish) holds nothing."""
    lo, hi = sorted([low, high])
    finite = bool(np.isfinite(lo) and np.isfinite(hi) and np.isfinite(truth))
    return {"low": float(lo), "high": float(hi), "median": float(median),
            "holds": bool(finite and lo <= truth <= hi),
            "useful": bool(finite and hi - lo <= RC.TOL_SHARE * published),
            "width_share": float((hi - lo) / published),
            "median_minus_truth_share": float((median - truth) / published)}


def cell_stage(prediction: Dict[str, object]) -> Dict[str, object]:
    from cmfqsar import cellmodel as CM

    grid = prediction["grid"]
    c = np.array([g["c"] for g in grid])
    mu = np.array([g["ln_ratio"] for g in grid])
    sd = np.array([g["sd"] for g in grid])
    s_label = float(prediction["label_deviation"])
    four = CM.measured_electrolyte()
    p = CM.published_parameters(RC.BASE)
    out: Dict[str, object] = {"conditions": []}
    for temperature, rate in RC.stress_conditions():
        t0 = time.time()
        with open(RC.stress_path(RC.RESULTS, temperature, rate), encoding="utf-8") as fh:
            stored = json.load(fh)
        common = dict(cycles=1, base=RC.BASE, kind=RC.STRESS_MODEL, cycle=CM.discharge(rate),
                      temperature=temperature)
        entry = {"temperature": float(temperature), "rate": float(rate),
                 "published": float(stored["published"]),
                 "truth": float(stored["landesfeind"]["all_four"]),
                 "truth_ok": bool(stored["landesfeind"]["all_four_ok"]), "reading": {}}
        for reading, spread in (("A", sd), ("B", np.sqrt(sd ** 2 + s_label ** 2))):
            caps = {}
            for z in Z_POINTS:
                if reading == "B" and z == 0.0 and "A" in entry["reading"]:
                    caps[z] = entry["reading"]["A"]["median"]
                    continue
                ratio = np.exp(mu + z * spread)
                run = CM.run_cell(dict(four, **{RC.DIFFUSIVITY: _scaled(
                    four[RC.DIFFUSIVITY], c, ratio)}), parameters=p, **common)
                caps[z] = run.final if run.ok else float("nan")
            entry["reading"][reading] = read_interval(caps[-Z95], caps[0.0], caps[Z95],
                                                      entry["truth"], entry["published"])
        entry["seconds"] = round(time.time() - t0, 1)
        a, b = entry["reading"]["A"], entry["reading"]["B"]
        print(f"  {temperature:.0f} K, {rate:g}C: truth {entry['truth']:.3f} A.h; A "
              f"{a['low']:.3f} to {a['high']:.3f} ({'holds' if a['holds'] else 'misses'}); B "
              f"{b['low']:.3f} to {b['high']:.3f} ({'holds' if b['holds'] else 'misses'}) "
              f"({entry['seconds']:.0f} s)", flush=True)
        out["conditions"].append(entry)
    out["gate_holds"] = bool(all(e["reading"]["A"]["holds"] for e in out["conditions"]))
    out["gate_holds_with_label_deviation"] = bool(
        all(e["reading"]["B"]["holds"] for e in out["conditions"]))
    return out


# ----------------------------------------------------------------------------------- the report


def save(name: str, payload: Dict[str, object], base: str = RESULTS) -> None:
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


def _read(base: str, name: str) -> Dict[str, object]:
    with open(os.path.join(base, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def write_table(results: str = RESULTS) -> str:
    pred = _read(results, "prediction")
    cell = _read(results, "cell")
    out: List[str] = ["# Item BT9, gate (c): the capacity of the cell with the predicted "
                      "diffusion coefficient", ""]
    w = out.append
    w("The salt diffusion coefficient of LiPF6 in EC:EMC 3:7 is predicted from the composition "
      "with every formulation of that mixture held out. The cell is the full model of item MS9. "
      "The truth is the capacity of the cell with all four transport parameters as Landesfeind and "
      "Gasteiger measured them. The reading of the gate was fixed before the cell was run (see "
      "the docstring of `benchmarks/run_cellgate.py`).")
    w("")
    w(f"The model uses the kernel `{pred['kernel']}` with the ridge {pred['ridge']:g}, chosen on "
      f"{pred['n_train']} formulations of other mixtures. The deviation of the label from the "
      f"measurement, over the six concentrations of the series, is {pred['label_deviation']:.2f} "
      "in the natural logarithm.")
    w("")
    w("| Concentration, mol/L | Molality, mol/kg | Predicted salt diffusion coefficient, m2/s "
      "| Measured, m2/s | Ratio | Standard deviation of the logarithm |")
    w("|---:|---:|---:|---:|---:|---:|")
    for g in pred["grid"]:
        w(f"| {g['c'] / 1000:.2f} | {g['molality']:.2f} | {g['d_predicted']:.2e} | "
          f"{g['d_measured']:.2e} | {np.exp(g['ln_ratio']):.2f} | {g['sd']:.3f} |")
    w("")
    w("Capacity of the last discharge in A.h. Reading A uses the standard deviation of the model "
      "alone, reading B adds the deviation of the label.")
    w("")
    w("| Temperature, K | Rate, C | Published cell | Truth | Median of the prediction | "
      "Interval A | A holds the truth | A useful | Interval B | B holds the truth |")
    w("|---:|---:|---:|---:|---:|---|---|---|---|---|")
    for e in cell["conditions"]:
        a, b = e["reading"]["A"], e["reading"]["B"]
        yes = lambda v: "yes" if v else "no"                                  # noqa: E731
        w(f"| {e['temperature']:.0f} | {e['rate']:g} | {e['published']:.3f} | {e['truth']:.3f} | "
          f"{a['median']:.3f} | {a['low']:.3f} to {a['high']:.3f} | {yes(a['holds'])} | "
          f"{yes(a['useful'])} | {b['low']:.3f} to {b['high']:.3f} | {yes(b['holds'])} |")
    w("")
    n = len(cell["conditions"])
    held = sum(e["reading"]["A"]["holds"] for e in cell["conditions"])
    heldb = sum(e["reading"]["B"]["holds"] for e in cell["conditions"])
    useful = sum(e["reading"]["A"]["useful"] for e in cell["conditions"])
    off = max(abs(e["reading"]["A"]["median_minus_truth_share"]) for e in cell["conditions"])
    w(f"**Gate (c).** The interval of reading A holds the truth under {held} of the {n} "
      f"conditions, and the interval of reading B under {heldb}. The gate "
      + ("holds." if cell["gate_holds"] else "does not hold.")
      + f" The interval of reading A is useful (narrower than {100 * RC.TOL_SHARE:.0f} % of the "
      f"published capacity) under {useful} of {n}. The median of the prediction is at most "
      f"{100 * off:.1f} % of the published capacity away from the truth.")
    w("")
    path = os.path.join(results, "table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("predict", "cell", "report", "all"), default="all")
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args(argv)
    stages = ("predict", "cell", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "predict":
            save("prediction", predict_stage(), args.results)
        elif stage == "cell":
            save("cell", cell_stage(_read(args.results, "prediction")), args.results)
        else:
            print("the table:", write_table(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
