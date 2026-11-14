#!/usr/bin/env python
"""
The cell model of plan item MS9 with a field model in every channel.

What this benchmark does
------------------------
``run_cellmodel.py`` passes four predicted material parameters to a continuum cell model, each
with its standard deviation. It takes for every parameter the model that had the smallest error
in the benchmark of that parameter. For two of the four that is a kernel on the composition, and
for one it is a random forest. This benchmark asks a narrower question: does the whole scheme
run when **every** channel is a kernel on continuous fields, which is the method this program
is named after? It builds the same parameter set with these models:

1. **The diffusion coefficient in the positive particles.**  Kernel ridge regression with the
   cosine kernel on the invariant field about the midpoint of a lithium hop, fitted on the 1,652
   density-functional barriers of nebDFT2k (plan item BT7). ``run_cellmodel.py`` gives the same
   rows to a random forest.
2. **The open-circuit voltage of the positive electrode.**  The autocorrelation kernel on the
   periodic fields of the discharged cell of each insertion pair (plan item BT3), at three
   widths of the field. ``run_cellmodel.py`` uses a kernel on the composition.
3. **The conductivity of the electrolyte.**  The kernel on the fields of the solvation shells
   that were sampled from a simulated box of each electrolyte (plan item BT5), mixed in equal
   parts with a kernel on the molality, as in that item. The molality is a descriptor that is
   not a field. ``run_cellmodel.py`` uses a kernel on the composition and the molality.
4. **The tortuosity of the porous electrode.**  The kernel on the radial autocorrelation of a
   two-phase field. This is the model of ``run_cellmodel.py`` itself, because that channel was
   a field kernel from the start.

The exchange-current density is not predicted in either benchmark and carries a stated spread.

What is measured
----------------
* For every channel: the error of the field model on held-out groups, beside the error of the
  model that ``run_cellmodel.py`` uses, and whether the deviation of the field model is
  calibrated (the root mean square of the standardised residual, which is near 1 for a
  deviation of the right size).
* For every parameter: the interval of the capacity when that parameter is drawn from its
  deviation and the others stay published, and whether the interval holds the published curve.
* All five parameters drawn together, which is the scheme run as a whole.

What this benchmark does not claim
----------------------------------
It does not claim that the field models are the better ones. In three channels they have the
larger error, and the verdict of ``run_cellmodel.py`` on the usefulness of the intervals does
not change. The claim is that the scheme is implemented with fields in every channel, and that
the deviations reach the capacity of the cell.

Usage
-----
    python benchmarks/run_cellmodel.py            # once: the other benchmark, for the comparison
    python benchmarks/run_cellmodel_fields.py     # this benchmark, then the table
    python benchmarks/run_cellmodel_fields.py --stage joint
    python benchmarks/run_cellmodel_fields.py --report

Every stage writes its own file under ``benchmarks/results/cellmodel_fields``, and a stage whose
file is there is not run again. The field rows of the insertion pairs are kept in
``benchmarks/cache/cellmodel_fields``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_cellmodel as RC                                                    # noqa: E402

RESULTS = os.path.join(HERE, "results", "cellmodel_fields")
#: the results of ``run_cellmodel.py``, which the table sets beside its own
OTHER = RC.RESULTS
CACHE = os.path.join(HERE, "cache", "cellmodel_fields")
STAGES = ("channels", "reference", "sample", "joint", "finish")
#: the two kernels on the sampled shells that the fold chooses between, with their names
SHELL_MODELS = (("shells-free", "fields of the sampled shells, no alignment"),
                ("shells", "fields of the sampled shells, in the frame of the cation"))
#: how each channel is described in the table: (the field model, the model of run_cellmodel.py)
MODELS = {
    "Positive particle diffusivity [m2.s-1]":
        ("kernel ridge regression on the invariant field of the hop",
         "random forest on the same field rows"),
    "Positive electrode OCP [V]":
        ("autocorrelation kernel on the periodic fields of the cell",
         "kernel on the composition"),
    "Electrolyte conductivity [S.m-1]":
        ("kernel on the fields of the sampled solvation shells, with the molality",
         "kernel on the composition and the molality"),
    "Positive electrode Bruggeman coefficient (electrolyte)":
        ("kernel on the radial autocorrelation of the two-phase field",
         "the same model"),
    "Positive electrode exchange-current density [A.m-2]":
        ("not predicted: a stated spread", "not predicted: a stated spread"),
}
#: the units of the error of each channel, as the table prints them
ERROR_UNITS = {"Positive particle diffusivity [m2.s-1]": "eV",
               "Positive electrode OCP [V]": "V",
               "Electrolyte conductivity [S.m-1]": "log10",
               "Positive electrode Bruggeman coefficient (electrolyte)": "ln"}

log = RC.log


# --------------------------------------------------------------------------- #
# A fit on every label, for a structure outside the labelled set
# --------------------------------------------------------------------------- #
def fit_whole(kernels: Dict[str, np.ndarray], y: Sequence[float], groups: Sequence[object],
              lams: Sequence[float] = RC.LAMS, inner: int = RC.INNER, seed: int = RC.SEED
              ) -> Tuple[str, float, object]:
    """The kernel and the ridge chosen on folds of the whole set that hold out whole groups,
    and the fit on the whole set with them.

    ``run_cellmodel.nested_surrogate`` makes this choice inside every outer fold, for the
    labelled structures. A structure that carries no label, such as the cathode of the cell, is
    predicted by one fit on all labels, and this function makes the same choice for that fit.
    Returns the name of the kernel, the ridge and the fitted surrogate.
    """
    from cmfqsar.surrogate import Surrogate

    y = np.asarray(y, float).ravel()
    n = len(y)
    everything = np.arange(n)
    parts = RC.group_parts(list(groups), inner, seed)
    best = None
    for name, K in kernels.items():
        diag = np.diag(K)
        for lam in lams:
            p = np.full(n, np.nan)
            for te in parts:
                tr = np.setdiff1d(everything, te)
                if len(tr) < 3:
                    continue
                s = Surrogate.fit(K[np.ix_(tr, tr)], y[tr], lam)
                p[te], _ = s.predict(K[np.ix_(te, tr)], diag[te])
            ok = np.isfinite(p)
            if ok.sum() < 3:
                continue
            err = float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))
            if best is None or err < best[0]:
                best = (err, name, float(lam))
    if best is None:
        raise RuntimeError("no fold of these kernels could be fitted")
    _, name, lam = best
    return name, lam, Surrogate.fit(kernels[name], y, lam)


# --------------------------------------------------------------------------- #
# Channel 1: the diffusion coefficient, from a kernel on the field of a hop
# --------------------------------------------------------------------------- #
def barrier_fields(draws: int = RC.DRAWS, folds: int = 10) -> Tuple[object, dict]:
    """The diffusion coefficient of the positive particles, from hop barriers that kernel ridge
    regression on the invariant field predicts."""
    import run_conductors as C
    import run_nebdft as N

    from cmfqsar.localenv import local_kernel

    if not os.path.isfile(N.PREPARED):
        raise SystemExit(
            f"{N.PREPARED} is missing: run 'python benchmarks/prepare_nebdft.py --fetch' and "
            "'python benchmarks/run_nebdft.py' once, which leave the barriers and the rows "
            "this channel is fitted on")
    hs, _doc, _, _ = N.load(N.PREPARED, 0)
    t = hs.targets()
    y = np.asarray(t.y, float)
    groups = np.asarray(t.molecule, int)
    _, rows = N.descriptions(hs, 8, N.CACHE, "")
    X = rows["invariant"]
    kernels = {"invariant field of the hop": np.asarray(local_kernel(X), float)}
    fit = RC.nested_surrogate(kernels, y, list(groups), folds=folds)
    # the deviation is put on the scale of the held-out errors, as the forest's is
    factor = float(fit["z_rms"])
    z = (y - fit["predicted"]) / fit["sd"]
    errors = {"mae": fit["mae"], "rmse": fit["rmse"], "mean_sd": fit["mean_sd"],
              "factor": factor, "shared": C.shared_share(z / factor, groups),
              "inside_2sd": float(np.mean(np.abs(z / factor) <= 2.0))}
    log(f"  the kernel on {len(y)} barriers, whole structures held out: mean absolute error "
        f"{errors['mae']:.3f} eV, z rms {factor:.2f}, shared within a structure "
        f"{errors['shared']:.2f}")

    hc, reps, Xc, record = RC.cathode_rows(hs)
    name, lam, s = fit_whole(kernels, y, list(groups))
    cross = np.asarray(local_kernel(X, Xc), float)
    if cross.shape != (len(Xc), len(X)):
        raise RuntimeError(f"the kernel of the cathode hops against the training hops has the "
                           f"shape {cross.shape}, not {(len(Xc), len(X))}")
    # the kernel is cosine-normalised, so the kernel of a hop with itself is 1
    pred, sd = s.predict(cross, np.ones(len(Xc)))
    sd = np.maximum(sd, 1e-6) * factor
    parameter, details = RC.barrier_parameter(
        hc, reps, record, np.asarray(pred, float), sd, errors, len(y), len(np.unique(groups)),
        "kernel ridge regression on the invariant field", draws)
    details["chosen"] = {"kernel": name, "lam": lam}
    return parameter, details


# --------------------------------------------------------------------------- #
# Channel 2: the open-circuit voltage, from the periodic fields of the cell
# --------------------------------------------------------------------------- #
def cached_rows(name: str, keys: Sequence[str], build) -> np.ndarray:
    """A matrix of field rows, kept in the cache under `name` together with the names of its
    rows. It is built again when the names are not the ones that were stored."""
    path = os.path.join(CACHE, f"{name}.npz")
    keys = [str(k) for k in keys]
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as z:
            if [str(k) for k in z["keys"]] == keys:
                return np.asarray(z["rows"], float)
    rows = np.asarray(build(), float)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, rows=rows, keys=np.asarray(keys))
    return rows


def voltage_field_kernels(structures, names, elements) -> Dict[str, np.ndarray]:
    """The autocorrelation kernels of the periodic fields of the insertion pairs, one per width
    of the field, as ``run_battery.py`` builds them.

    The scale of the channels is computed on all rows. It uses the fields and no voltage."""
    import run_battery as B

    fs = B.inorganic_fieldset(elements, physchem=B.INORGANIC_FIELDS, charge_model="oxidation")
    out = {}
    for sigma in B.SIGMAS:
        spec = B.PeriodicSpec(r_max=B.R_MAX, step=sigma, sigma=sigma, taper=B.TAPER)
        t0 = time.time()
        F = cached_rows(
            f"voltage_sigma_{sigma:g}", names,
            lambda: B.periodic_feature_matrix(
                [B.crystal_fields(a, fs, nm) for a, nm in zip(structures, names)], spec,
                n_fields=fs.n_fields))
        blocks = B.periodic_block_sizes(spec, fs.n_fields)
        scale = B.channel_scale(F, spec.n_bins, blocks=blocks)
        K = B.autocorrelation_kernel(F, scale=scale, n_bins=spec.n_bins, blocks=blocks)
        out[f"periodic fields, width {sigma:g} A"] = RC.unit_kernel(K)
        log(f"  periodic fields at a width of {sigma:g} A: {F.shape[1]} channels, "
            f"{time.time() - t0:.0f} s")
    return out


def voltage_fields() -> Tuple[object, dict]:
    """A shift of the published open-circuit voltage, from the kernel on the periodic fields."""
    return RC.voltage_channel(voltage_field_kernels, "the continuous fields of the discharged cell")


# --------------------------------------------------------------------------- #
# Channel 3: the conductivity, from the fields of the sampled solvation shells
# --------------------------------------------------------------------------- #
def mixed_kernel(field: np.ndarray, condition: np.ndarray, mix: float) -> np.ndarray:
    """A field kernel and a kernel on the conditions, each with a unit diagonal, added with the
    weights ``1 - mix`` and ``mix``."""
    def unit(K):
        d = np.sqrt(np.clip(np.diag(np.asarray(K, float)), 1e-12, None))
        return np.asarray(K, float) / np.outer(d, d)
    if not 0.0 <= float(mix) <= 1.0:
        raise ValueError("the weight of the condition kernel has to be between 0 and 1")
    return (1.0 - float(mix)) * unit(field) + float(mix) * unit(condition)


def shell_kernels(rows) -> Dict[str, np.ndarray]:
    """The kernels on the fields of the sampled solvation shells of plan item BT5, at every
    width of the field, each mixed with the kernel on the molality as in that item.

    The scale of the channels and the width of the molality kernel are computed on all rows.
    They use the fields and the molality, and no conductivity."""
    import run_shells as S

    shells = S.shell_molecules(rows)
    everything = np.arange(len(rows))
    condition, _ = S.condition_kernel(S.descriptors(rows, "shells"), everything)
    out = {}
    for model, label in SHELL_MODELS:
        t0 = time.time()
        for alpha, entry in S.field_tables(rows, model, shells, {}).items():
            field, _ = S._kernel_at(entry, everything)
            out[f"{label}, width {alpha:g}"] = mixed_kernel(field, condition, S.MIX)
        log(f"  {label}: {time.time() - t0:.0f} s")
    return out


def electrolyte_fields() -> Tuple[object, dict]:
    """A factor on the published conductivity, from the kernel on the fields of the shells."""
    return RC.electrolyte_channel(
        shell_kernels, "the continuous fields of the sampled solvation shells and the molality")


CHANNELS = (("diffusion coefficient", barrier_fields),
            ("open-circuit voltage", voltage_fields),
            ("electrolyte conductivity", electrolyte_fields),
            ("electrode tortuosity", RC.microstructure_channel),
            ("exchange current", RC.exchange_channel))


# --------------------------------------------------------------------------- #
# All parameters drawn together
# --------------------------------------------------------------------------- #
def joint_values(pset, draws: int, seed: int = RC.SEED) -> Dict[str, np.ndarray]:
    """`draws` values of every parameter of the set, each drawn from its own deviation and
    independently of the others."""
    rng = np.random.default_rng(int(seed))
    return {p.name: np.asarray(p.draw(int(draws), rng), float) for p in pset}


def stage_joint(results: str, draws: int = RC.DRAWS, cycles: int = RC.CYCLES) -> dict:
    """The cell model run once per draw of all parameters together."""
    from cmfqsar import cellmodel as CM

    pset, _ = RC.read_set(results)
    reference = RC.read_reference(results)
    published = CM.published_parameters(RC.BASE)
    values = joint_values(pset, draws)
    log(f"all {len(pset)} parameters drawn together: {draws} draws")
    t0 = time.time()
    runs = []
    for i in range(int(draws)):
        overrides = {p.name: p.apply(published[p.name], float(values[p.name][i])) for p in pset}
        runs.append(CM.run_cell(overrides, cycles=cycles, base=RC.BASE, kind=RC.KIND,
                                parameters=published))
        if (i + 1) % 25 == 0 or i + 1 == draws:
            log(f"    {i + 1} of {draws} draws, {time.time() - t0:.0f} s")
    b = CM.band(runs)
    held = CM.contains(b, reference["curve"])
    out = {"draws": int(draws), "cycles": int(cycles),
           "interval_of_the_last_cycle": [float(b["low"][-1]), float(b["high"][-1])]
           if b["n"] > 1 else [float("nan")] * 2,
           "median_of_the_last_cycle": float(b["median"][-1]) if b["n"] > 1 else float("nan"),
           "runs_that_completed": int(b["n"]), "refused": float(b["refused"]),
           "holds_the_published_curve": held,
           "final": [r.final for r in runs], "ok": [bool(r.ok) for r in runs],
           "seconds": round(time.time() - t0, 1)}
    log(f"  the last cycle: {out['interval_of_the_last_cycle'][0]:.3f} to "
        f"{out['interval_of_the_last_cycle'][1]:.3f} A.h, the published curve is "
        f"{'inside' if held['all_inside'] else 'outside'} it, "
        f"{100 * out['refused']:.1f} % of the runs refused")
    with open(os.path.join(results, "joint.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# The summary and the table
# --------------------------------------------------------------------------- #
def channel_error(name: str, info: dict) -> Optional[dict]:
    """The error of one channel on held-out groups, from the record of that channel: the mean
    absolute error, that of predicting the mean where it was stored, and the z rms."""
    if name not in ERROR_UNITS:
        return None
    if "training" in info:                              # the barriers
        t = info["training"]
        return {"mae": float(t["mae"]), "mean": None, "z_rms": float(t["factor"])}
    key = "mae_log" if "mae_log" in info else "mae"
    return {"mae": float(info[key]), "mean": float(info["mae_of_the_mean"]),
            "z_rms": float(info["z_rms"])}


def central_value(parameter) -> float:
    """The value of a parameter as the table prints it: the median of its drawn sample when it
    carries one, and its value otherwise."""
    if parameter.distribution == "sample":
        return float(np.median(parameter.sample))
    return float(parameter.value)


def published_value(parameter) -> Optional[float]:
    """The published value in the terms of the parameter: 1 for a factor, 0 for a shift, and
    the published number for a value that replaces it."""
    if parameter.kind == "scale":
        return 1.0
    if parameter.kind == "shift":
        return 0.0
    return None if parameter.published is None else float(parameter.published)


def read_other(other: str = OTHER) -> Optional[dict]:
    """What ``run_cellmodel.py`` stored, for the comparison; None when it was not run."""
    path = os.path.join(other, "cellmodel.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def summarise(results: str = RESULTS, other: str = OTHER) -> dict:
    """Everything the table prints, from the files of the stages."""
    pset, details = RC.read_set(results)
    reference = RC.read_reference(results)
    theirs = read_other(other)
    their_set = RC.read_set(other)[0] if theirs is not None else None
    rows = []
    for p in pset:
        path = os.path.join(results, f"sample_{RC.slug(p.name)}.json")
        if not os.path.isfile(path):
            raise SystemExit(f"{path} is missing: run this benchmark with --stage sample first")
        with open(path, encoding="utf-8") as fh:
            sample = json.load(fh)
        lo, hi = p.interval()
        row = {"name": p.name, "model": MODELS[p.name][0], "other_model": MODELS[p.name][1],
               "predicted": bool(p.predicted), "value": central_value(p), "units": p.units,
               "published": published_value(p),
               "kind": p.kind, "interval": [float(lo), float(hi)],
               "calibrated": bool(p.calibrated) if p.calibration is not None else None,
               "error": channel_error(p.name, details[p.name]),
               "chosen": details[p.name].get("chosen"),
               "capacity": sample["interval_of_the_last_cycle"],
               "refused": float(sample["refused"]),
               "holds": bool(sample["holds_the_published_curve"]["all_inside"]),
               "variance": float(np.var([v for v, ok in zip(sample["final"], sample["ok"])
                                         if ok and np.isfinite(v)]))}
        if theirs is not None and p.name in theirs["per_parameter"]:
            q = theirs["per_parameter"][p.name]
            row["other"] = {"error": channel_error(p.name, theirs["channels"][p.name]),
                            "value": central_value(their_set[p.name]),
                            "capacity": q["interval_of_the_last_cycle"],
                            "holds": bool(q["holds_the_published_curve"]["all_inside"])}
        rows.append(row)
    total = sum(r["variance"] for r in rows)
    for r in rows:
        r["share"] = r["variance"] / total if total > 0 else float("nan")
    joint_path = os.path.join(results, "joint.json")
    joint = None
    if os.path.isfile(joint_path):
        with open(joint_path, encoding="utf-8") as fh:
            j = json.load(fh)
        joint = {k: v for k, v in j.items() if k not in ("final", "ok")}
    out = {"base": RC.BASE, "model": RC.KIND, "published": float(reference["final"]),
           "cycles": int(reference["cycles"]), "rows": rows, "joint": joint,
           "compared": theirs is not None}
    with open(os.path.join(results, "cellmodel_fields.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def _n(v, digits: int = 3) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def _g(v) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.2g}"


def table_lines(res: dict) -> List[str]:
    """The table of this benchmark as lines of Markdown."""
    out: List[str] = []
    w = out.append
    rows = res["rows"]
    predicted = [r for r in rows if r["predicted"]]
    w("# Item MS9 with a field model in every channel")
    w("")
    w(f"The cell model is PyBaMM's {res['model']} on the published parameter set "
      f"`{res['base']}`, as in `benchmarks/results/cellmodel/table.md`. That table takes for "
      "every parameter the model with the smallest error. This one takes a kernel on "
      "continuous fields for every parameter that is predicted, and asks whether the scheme "
      "still runs from the structures to the capacity of the cell. It does not ask which "
      "model is better.")
    w("")
    w("## The models and their errors on held-out groups")
    w("")
    w("The error is the mean absolute error with whole groups held out: structures for the "
      "barriers, frameworks for the voltage, solvent mixtures for the conductivity. z rms is "
      "the root mean square of the standardised residual; a deviation of the right size has a "
      "z rms near 1, and the rule of plan item MS1 accepts 0.8 to 1.25.")
    w("")
    w("| Parameter | Field model | Error of the field model | z rms | Model of the other table "
      "| Its error | Unit of the error |")
    w("|---|---|---:|---:|---|---:|---|")
    for r in predicted:
        e = r["error"] or {}
        o = (r.get("other") or {}).get("error") or {}
        w(f"| {r['name']} | {r['model']} | {_n(e.get('mae'))} | {_n(e.get('z_rms'), 2)} | "
          f"{r['other_model']} | {_n(o.get('mae'))} | {ERROR_UNITS[r['name']]} |")
    w("")
    w("The unit `log10` is the base-ten logarithm of the conductivity in mS/cm, and `ln` is "
      "the natural logarithm of the ratio of the effective conductivity to that of the "
      "electrolyte.")
    w("")
    w("## The parameters")
    w("")
    w("A value replaces the published one, multiplies it or is added to it. The column of the "
      "published value is in the same terms: the published number, 1 for a factor, 0 for a "
      "shift.")
    w("")
    w("| Parameter | Value | 95 % interval | What the value does to the published one | "
      "Published value in the same terms | Value in the other table | Deviation calibrated |")
    w("|---|---:|---|---|---:|---:|---|")
    what = {"value": "replaces it", "scale": "multiplies it", "shift": "is added to it"}
    for r in rows:
        cal = "--" if r["calibrated"] is None else ("yes" if r["calibrated"] else "no")
        w(f"| {r['name']} | {_g(r['value'])} | {_g(r['interval'][0])} to "
          f"{_g(r['interval'][1])} | {what[r['kind']]} | {_g(r.get('published'))} | "
          f"{_g((r.get('other') or {}).get('value'))} | {cal} |")
    w("")
    w("## What each deviation does to the capacity")
    w("")
    w(f"Each parameter is drawn from its deviation while the others stay published. The "
      f"published values give {_n(res['published'])} A.h at the last of {res['cycles']} "
      "cycles. The share is that of the summed variance of the capacity over the five "
      "parameters.")
    w("")
    w("| Parameter | Capacity of the last cycle, A.h | Holds the published curve | Draws that "
      "the cell model refused | Share of the variance | The same interval in the other table "
      "| Holds there |")
    w("|---|---|---|---:|---:|---|---|")
    for r in sorted(rows, key=lambda r: -r["share"] if np.isfinite(r["share"]) else 1.0):
        o = r.get("other")
        w(f"| {r['name']} | {_n(r['capacity'][0])} to {_n(r['capacity'][1])} | "
          f"{'yes' if r['holds'] else 'no'} | {100 * r['refused']:.1f} % | "
          f"{100 * r['share']:.1f} % | "
          + (f"{_n(o['capacity'][0])} to {_n(o['capacity'][1])} | "
             f"{'yes' if o['holds'] else 'no'} |" if o else "-- | -- |"))
    w("")
    j = res.get("joint")
    if j is not None:
        lo, hi = j["interval_of_the_last_cycle"]
        w("## All parameters drawn together")
        w("")
        w(f"All five parameters are drawn at once, {j['draws']} times, and the cell model is "
          f"run once per draw. {j['runs_that_completed']} runs completed, and the cell model "
          f"refused {100 * j['refused']:.1f} % of the draws: a drawn voltage far from the "
          "published one leaves the cell no capacity inside its voltage limits. Over the runs "
          "that completed, "
          f"the capacity of the last cycle is between {_n(lo)} and {_n(hi)} A.h (95 % "
          f"interval), with a median of {_n(j['median_of_the_last_cycle'])} A.h. The published "
          f"curve is {'inside' if j['holds_the_published_curve']['all_inside'] else 'outside'} "
          "this interval.")
        w("")
    w("## What this shows and what it does not")
    w("")
    w("* The scheme is implemented with a field model in every predicted channel: fields of a "
      "hop in a crystal, fields of a periodic cell, fields of solvation shells and the field "
      "of a two-phase structure. Each gives a value with a deviation, a conversion turns it "
      "into a parameter of the cell model, and the deviation reaches the capacity.")
    w("* It does not show that the intervals are useful. An interval of the capacity is "
      "useful only if it is narrow and holds the published value, and the columns above say "
      "where it is neither. The reason is the size of two deviations. A hop barrier is known "
      "to this program to a few tenths of an electronvolt, which is several orders of "
      "magnitude in a diffusion coefficient at room temperature. A voltage is known to a "
      "large fraction of a volt, where a measured curve is known to millivolts.")
    w("")
    return out


def report(results: str = RESULTS) -> str:
    """Write the table from the stored files. Returns the path of the table."""
    path = os.path.join(results, "cellmodel_fields.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run this benchmark first")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    out = os.path.join(results, "table.md")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(table_lines(res)) + "\n")
    return out


def run(results: str = RESULTS, draws: int = RC.DRAWS, cycles: int = RC.CYCLES,
        stages: Sequence[str] = STAGES, force: bool = False) -> Optional[dict]:
    """The benchmark, in stages. Returns the summary when the last stage runs."""
    os.makedirs(results, exist_ok=True)
    if "channels" in stages:
        if force or not os.path.isfile(os.path.join(results, "parameters.json")):
            RC.stage_channels(results, CHANNELS)
        else:
            log("the parameter set is there already; --force rebuilds it")
    pset, _ = RC.read_set(results)
    if "reference" in stages:
        if force or not os.path.isfile(os.path.join(results, "reference.json")):
            RC.stage_reference(results, cycles)
        else:
            log("the published curve is there already")
    if "sample" in stages:
        for p in pset:
            path = os.path.join(results, f"sample_{RC.slug(p.name)}.json")
            if os.path.isfile(path) and not force:
                log(f"{p.name}: drawn already")
                continue
            RC.stage_sample(results, p.name, draws, cycles)
    if "joint" in stages:
        if force or not os.path.isfile(os.path.join(results, "joint.json")):
            stage_joint(results, draws, cycles)
        else:
            log("the joint draws are there already")
    if "finish" not in stages:
        return None
    return summarise(results)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--report", action="store_true",
                    help="write the table from the stored result and stop")
    ap.add_argument("--draws", type=int, default=RC.DRAWS,
                    help="how many times each parameter is drawn")
    ap.add_argument("--cycles", type=int, default=RC.CYCLES, help="how many cycles per run")
    ap.add_argument("--results", default=RESULTS, help="where the result is written")
    ap.add_argument("--stage", action="append", choices=STAGES,
                    help="run only these stages; a stage whose file is there is not run again "
                         "(default: all of them)")
    ap.add_argument("--force", action="store_true",
                    help="run the stages again even where their files are there")
    args = ap.parse_args(argv)
    if not args.report:
        done = run(args.results, args.draws, args.cycles, tuple(args.stage or STAGES),
                   args.force)
        if done is None:
            return 0
    log(f"the table: {report(args.results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
