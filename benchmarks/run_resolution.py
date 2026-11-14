"""
Plan item MS4: a kernel that knows the resolution of its description.

Two questions, both on the eight Sutherland series (``examples/sutherland``), with the five
ordinary fields, charges from the files, and the field kernels cosine-normalised and summed with
equal weights at each width, as ``run_wavelet.py`` builds them.

1. **Several widths in one model** (stage ``widths``). Three models under the same repeated
   cross-validation (five folds, three repeats), everything chosen on the training block only:

   ===================  ====================================================================
   ``one width``        one width of seven (0.05 to 4) and the ridge, chosen by the evidence.
                        This is the single width the gate compares with.
   ``equal widths``     the sum of the kernels at all seven widths, the ridge by the evidence.
   ``evidence widths``  the weights of the seven widths fitted by the evidence
                        (``resolution.fit_width_mixture``), then the ridge by the evidence.
   ===================  ====================================================================

   **The gate, fixed before the measurement:** no loss of q² against the single width means
   that on no series does the 95 % interval of the q² difference (paired bootstrap over the
   molecules, on the predictions averaged over the repeats) lie wholly below zero, and that the
   mean difference over the eight series is not below -0.01.

2. **Beads** (stage ``beads``). Each molecule is also described by beads
   (``resolution.beads_of``): ``united`` puts each heavy atom with its hydrogens in one bead,
   ``four`` makes beads of up to four bonded heavy atoms. At each of the seven widths the bead
   kernel is compared with the all-atom kernel at the same width, by the Pearson correlation of
   the off-diagonal entries (the reading of item EC5) and by the centred kernel alignment; the
   all-atom width that the bead kernel matches best is recorded too. The same cross-validation
   then fits a model on each bead description.

   **The reading, fixed before the measurement:** item EC5 found that a kernel correlation of
   0.98 or more with the existing kernel predicted that no model built on the new kernel would
   differ. A bead kernel that correlates 0.98 or more with the all-atom kernel at the same
   width is read as the same description at a lower cost; below that it is a different
   description, and the q² of its model says whether different means worse.

Stages
------
    python benchmarks/run_resolution.py --stage widths --sets ACE AChE ...
    python benchmarks/run_resolution.py --stage beads --sets ACE AChE ...
    python benchmarks/run_resolution.py --report
    python benchmarks/run_resolution.py               # every stage on every series
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from cmfqsar import resolution as R                                          # noqa: E402
from cmfqsar.evidence import log_evidence                                    # noqa: E402
from cmfqsar.model import krr_fit, r2_score, rmse                            # noqa: E402

import run_wavelet as RW                                                     # noqa: E402

RESULTS = os.path.join(HERE, "results", "resolution")
SETS = RW.SETS
ALPHAS = RW.ALPHAS
LAMS = RW.LAMS
FOLDS = 5
REPEATS = 3
SEED = 0
DRAWS = 2000
RULES = ("united", "four")
#: the kernel correlation at and above which item EC5 found no model to differ
SAME = 0.98
#: the mean q2 difference over the series below which the gate counts a loss
MEAN_LOSS = -0.01


# --------------------------------------------------------------------------------- the fits

def select_ridge(K: np.ndarray, y: np.ndarray, tr: np.ndarray) -> float:
    """The ridge with the highest log evidence on the training block."""
    best = None
    Ktr = K[np.ix_(tr, tr)]
    for lam in LAMS:
        e = log_evidence(Ktr, y[tr], float(lam), fit_bias=True)
        if np.isfinite(e) and (best is None or e > best[0]):
            best = (e, float(lam))
    return best[1]


def select_width(stack: np.ndarray, y: np.ndarray, tr: np.ndarray):
    """The width and ridge with the highest log evidence on the training block."""
    best = None
    for w in range(stack.shape[0]):
        Ktr = stack[w][np.ix_(tr, tr)]
        for lam in LAMS:
            e = log_evidence(Ktr, y[tr], float(lam), fit_bias=True)
            if np.isfinite(e) and (best is None or e > best[0]):
                best = (e, w, float(lam))
    return best[1], best[2]


def predict(K: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, lam: float):
    a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
    return K[np.ix_(te, tr)] @ a + b


def cross_validate(models: Dict[str, object], y: np.ndarray) -> Dict[str, Dict[str, object]]:
    """Every model on the same folds; a model is ("one", stack), ("sum", K) or ("mix", stack)."""
    n = len(y)
    preds = {m: np.zeros((REPEATS, n)) for m in models}
    chosen: Dict[str, List[object]] = {m: [] for m in models}
    for rep in range(REPEATS):
        for te in RW.folds(n, FOLDS, SEED + rep):
            tr = np.setdiff1d(np.arange(n), te)
            for m, (how, data) in models.items():
                if how == "one":
                    w, lam = select_width(data, y, tr)
                    K = data[w]
                    chosen[m].append(float(ALPHAS[w]))
                elif how == "sum":
                    K = data
                    lam = select_ridge(K, y, tr)
                    chosen[m].append(lam)
                else:
                    sub = data[:, tr][:, :, tr]
                    h = R.fit_width_mixture(sub, y[tr])
                    K = R.mixed_kernel(data, h)
                    lam = select_ridge(K, y, tr)
                    chosen[m].append([float(v) for v in h / h.sum()])
                preds[m][rep, te] = predict(K, y, tr, te, lam)
    out = {}
    for m in models:
        r2s = [float(r2_score(y, preds[m][r])) for r in range(REPEATS)]
        out[m] = {"q2_mean": float(np.mean(r2s)), "q2_sd": float(np.std(r2s)),
                  "rmse_mean": float(np.mean([rmse(y, preds[m][r]) for r in range(REPEATS)])),
                  "chosen": chosen[m], "_pred": preds[m].mean(axis=0)}
    return out


def paired(y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = SEED) -> Dict[str, float]:
    """q2 of a minus q2 of b, with a paired bootstrap interval over the molecules."""
    rng = np.random.RandomState(seed)
    n = len(y)
    diffs = []
    for _ in range(DRAWS):
        idx = rng.randint(0, n, n)
        diffs.append(r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(r2_score(y, a) - r2_score(y, b)), "lo": float(lo), "hi": float(hi)}


def _strip(models: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {m: {k: v for k, v in r.items() if not k.startswith("_")} for m, r in models.items()}


# ------------------------------------------------------------------------------- the stages

def load(name: str):
    """A series as ``run_wavelet.load_set`` reads it, with the fields of each molecule kept."""
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    ds = load_dataset(os.path.join(RW.DATA, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    fs = FieldSet.create(physchem=RW.FIELDS, alpha=RW.ALPHA0, charge_model="file")
    mf = [build_fields(m, fs, n) for m, n, k in zip(ds.mols, ds.names, keep) if k]
    return AtomTable.from_molecules(mf), fs, y[keep], mf


def widths_set(name: str) -> Dict[str, object]:
    t0 = time.time()
    table, fs, y, _mf = load(name)
    stack = R.width_kernels(table, fs, ALPHAS)
    models = {"one width": ("one", stack), "equal widths": ("sum", stack.sum(axis=0)),
              "evidence widths": ("mix", stack)}
    res = cross_validate(models, y)
    ref = res["one width"]["_pred"]
    for m in ("equal widths", "evidence widths"):
        res[m]["q2_gain_vs_one_width"] = paired(y, res[m]["_pred"], ref)
    weights = np.mean(np.asarray(res["evidence widths"]["chosen"]), axis=0)
    out = {"set": name, "n": int(len(y)), "alphas": list(ALPHAS), "models": _strip(res),
           "mean_width_weights": [float(v) for v in weights], "seconds": time.time() - t0}
    for m, r in out["models"].items():
        g = r.get("q2_gain_vs_one_width")
        extra = f"  {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]" if g else ""
        print(f"  {name} {m:16s} q2 {r['q2_mean']:+.3f} +/- {r['q2_sd']:.3f}{extra}", flush=True)
    return out


def beads_set(name: str) -> Dict[str, object]:
    t0 = time.time()
    table, fs, y, mf = load(name)
    atom_stack = R.width_kernels(table, fs, ALPHAS)
    descriptions = {"atoms": atom_stack}
    beads_info = {}
    for rule in RULES:
        beads = [R.beads_of(m, rule) for m in mf]
        descriptions[rule] = np.asarray([R.bead_joint_kernel(beads, fs, a) for a in ALPHAS])
        beads_info[rule] = {"beads_per_molecule": float(np.mean([b.n_beads for b in beads])),
                            "atoms_per_molecule": float(np.mean([m.n_atoms for m in mf]))}
    agreement = {}
    for rule in RULES:
        rows = []
        for w, a in enumerate(ALPHAS):
            Kb = descriptions[rule][w]
            corr = [R.kernel_correlation(atom_stack[v], Kb) for v in range(len(ALPHAS))]
            rows.append({"alpha": float(a),
                         "correlation": R.kernel_correlation(atom_stack[w], Kb),
                         "alignment": R.centred_alignment(atom_stack[w], Kb),
                         "best_all_atom_alpha": float(ALPHAS[int(np.nanargmax(corr))]),
                         "best_correlation": float(np.nanmax(corr))})
        agreement[rule] = rows
    per_field = {}
    for rule in RULES:
        beads = [R.beads_of(m, rule) for m in mf]
        Kf_atoms = R.cosine_field_kernels(table, R._specs_at(fs, 1.0))
        Kf_beads = R.bead_kernels(beads, fs, 1.0)
        per_field[rule] = {s.name: R.kernel_correlation(Kf_atoms[f], Kf_beads[f])
                           for f, s in enumerate(fs.specs)}
    models = {d: ("one", st) for d, st in descriptions.items()}
    res = cross_validate(models, y)
    for rule in RULES:
        res[rule]["q2_gain_vs_atoms"] = paired(y, res[rule]["_pred"], res["atoms"]["_pred"])
    out = {"set": name, "n": int(len(y)), "alphas": list(ALPHAS), "beads": beads_info,
           "agreement": agreement, "per_field_at_alpha_1": per_field, "models": _strip(res),
           "seconds": time.time() - t0}
    for rule in RULES:
        best = max(r["correlation"] for r in agreement[rule])
        g = res[rule]["q2_gain_vs_atoms"]
        print(f"  {name} {rule:6s} {beads_info[rule]['beads_per_molecule']:.1f} beads, "
              f"best correlation {best:+.3f}, q2 {res[rule]['q2_mean']:+.3f} against "
              f"{res['atoms']['q2_mean']:+.3f}: {g['mean']:+.3f} [{g['lo']:+.3f}, "
              f"{g['hi']:+.3f}]", flush=True)
    return out


# --------------------------------------------------------------------------------- the table

def _read(base: str, stage: str, name: str) -> Optional[Dict[str, object]]:
    path = os.path.join(base, f"{stage}_{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def gate(rows: Sequence[Dict[str, object]], model: str = "evidence widths") -> Dict[str, object]:
    gains = [r["models"][model]["q2_gain_vs_one_width"] for r in rows]
    below = [r["set"] for r, g in zip(rows, gains) if g["hi"] < 0]
    mean = float(np.mean([g["mean"] for g in gains])) if gains else float("nan")
    return {"sets": len(rows), "interval_wholly_below_zero": below, "mean_difference": mean,
            "met": bool(rows) and not below and mean >= MEAN_LOSS}


def _iv(g: Dict[str, float]) -> str:
    return f"{g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]"


def write_table(path: Optional[str] = None) -> str:
    """``table.md`` of a results folder (by default ``results/resolution``), from its JSON."""
    base = path or RESULTS
    widths = [r for r in (_read(base, "widths", s) for s in SETS) if r]
    beads = [r for r in (_read(base, "beads", s) for s in SETS) if r]
    lines = ["# Item MS4: a kernel that knows the resolution of its description", "",
             "Eight Sutherland series, five fields with the charges of the files, field kernels "
             "cosine-normalised and summed with equal weights; five folds repeated three times, "
             "every choice made on the training block. Generated by "
             "`benchmarks/run_resolution.py`.", ""]
    if widths:
        lines += ["## Several widths in one model", "",
                  "The differences are q² of the model minus q² of the single width, with the "
                  "95 % interval of a paired bootstrap over the molecules.", "",
                  "| Series | Molecules | q², one width | q², equal sum of seven widths | "
                  "Difference | q², widths weighted by the evidence | Difference | "
                  "Largest weight, at width |",
                  "|---|---:|---:|---:|---|---:|---|---|"]
        for r in widths:
            m = r["models"]
            wts = r["mean_width_weights"]
            k = int(np.argmax(wts))
            lines.append(f"| {r['set']} | {r['n']} | {m['one width']['q2_mean']:+.3f} | "
                         f"{m['equal widths']['q2_mean']:+.3f} | "
                         f"{_iv(m['equal widths']['q2_gain_vs_one_width'])} | "
                         f"{m['evidence widths']['q2_mean']:+.3f} | "
                         f"{_iv(m['evidence widths']['q2_gain_vs_one_width'])} | "
                         f"{wts[k]:.2f} at {r['alphas'][k]} |")
        g = gate(widths)
        eq = gate(widths, "equal widths")
        lines += ["",
                  f"**The gate** (no series with the interval wholly below zero, and a mean "
                  f"difference not below {MEAN_LOSS}): "
                  + ("**met**" if g["met"] else "**not met**")
                  + f" for the widths weighted by the evidence, over {g['sets']} series, mean "
                  f"difference {g['mean_difference']:+.3f}"
                  + (f", interval below zero on {', '.join(g['interval_wholly_below_zero'])}"
                     if g["interval_wholly_below_zero"] else "")
                  + f". The equal sum: mean difference {eq['mean_difference']:+.3f}"
                  + (f", interval below zero on {', '.join(eq['interval_wholly_below_zero'])}."
                     if eq["interval_wholly_below_zero"] else "; no interval below zero."), ""]
    if beads:
        lines += ["## Beads", "",
                  "`united`: each heavy atom with its hydrogens. `four`: up to four bonded heavy "
                  "atoms per bead. The correlation is that of the off-diagonal entries of the "
                  "bead kernel and the all-atom kernel at the same width; item EC5 found that "
                  f"{SAME} or more predicts that no model differs.", "",
                  "| Series | Rule | Atoms per molecule | Beads per molecule | Correlation at "
                  "width 0.2 | Correlation at width 1 | Correlation at width 4 | Electrostatic "
                  "field alone, width 1 | q², all atoms | q², beads | Difference |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in beads:
            for rule in RULES:
                ag = {row["alpha"]: row for row in r["agreement"][rule]}
                lines.append(
                    f"| {r['set']} | {rule} | {r['beads'][rule]['atoms_per_molecule']:.1f} | "
                    f"{r['beads'][rule]['beads_per_molecule']:.1f} | "
                    f"{ag[0.2]['correlation']:+.3f} | {ag[1.0]['correlation']:+.3f} | "
                    f"{ag[4.0]['correlation']:+.3f} | "
                    f"{r['per_field_at_alpha_1'][rule]['electrostatic']:+.3f} | "
                    f"{r['models']['atoms']['q2_mean']:+.3f} | "
                    f"{r['models'][rule]['q2_mean']:+.3f} | "
                    f"{_iv(r['models'][rule]['q2_gain_vs_atoms'])} |")
        lines.append("")
        for rule in RULES:
            same = sum(all(row["correlation"] >= SAME for row in r["agreement"][rule])
                       for r in beads)
            gains = [r["models"][rule]["q2_gain_vs_atoms"] for r in beads]
            lines.append(f"* `{rule}`: the bead kernel correlates {SAME} or more with the "
                         f"all-atom kernel at every width on {same} of {len(beads)} series; "
                         f"its model is worse than the all-atom one with the interval below zero "
                         f"on {sum(g['hi'] < 0 for g in gains)} series and better with the "
                         f"interval above zero on {sum(g['lo'] > 0 for g in gains)}; mean "
                         f"difference {np.mean([g['mean'] for g in gains]):+.3f}.")
        lines.append("")
    text = "\n".join(lines)
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def save(stage: str, name: str, payload: Dict[str, object]) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{stage}_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["widths", "beads", "all"], default="all")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--report", action="store_true", help="rebuild table.md from the results")
    args = ap.parse_args(argv)
    if args.report:
        print(write_table())
        return 0
    for name in args.sets:
        if args.stage in ("widths", "all"):
            save("widths", name, widths_set(name))
        if args.stage in ("beads", "all"):
            save("beads", name, beads_set(name))
    write_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
