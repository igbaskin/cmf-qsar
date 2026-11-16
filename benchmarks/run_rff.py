#!/usr/bin/env python
"""
Random Fourier features of the field kernel, against the exact model and against Nyström.

The field kernel of two molecules is an overlap integral, so by Parseval's theorem it is also an
integral over frequencies of the two fields' Fourier transforms.  Sampling that integral at D
frequencies gives every molecule 2 D explicit features per field,
``sum_l w_l a_l(omega) cos(omega . r_l)`` and the same with a sine (:mod:`cmfqsar.rff`).  They
need no landmarks and no training molecules.

Stage ``rank``: every Sutherland set at the attenuation and the ridge the exact grid search chose
(read from ``results/nystrom/rank.json``, and checked here against a fresh exact kernel).  The
joint kernel built from the features is scored exactly as the Nyström kernels of plan item B6
were: the relative Frobenius error of the joint training kernel, and the 10-fold q2 on the same
folds, with the exact model's ridge and with a ridge re-selected on the approximate kernel.
Four ways of choosing the frequencies are compared: independent draws (``mc``), orthogonal
triples (``orthogonal``), a scrambled Sobol sequence (``qmc``), each with three seeds, and a
cubic lattice of frequencies (``lattice``), which is deterministic.

Stage ``width``: COX2 and THR at attenuations from 0.1 to 5, because a narrow Gaussian has a
wide spectrum, and the Nyström grid of Gaussian centres failed there (plan item B6).

Usage
-----
    python benchmarks/run_rff.py                 # every stage, then the table
    python benchmarks/run_rff.py --stage rank
    python benchmarks/run_rff.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from run_nystrom import (best_lam, gaussian_centres, joint_of, nystrom, q2_at,  # noqa: E402
                         rel_error, specs_at, table_of)

from cmfqsar import rff  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402

RESULTS = os.path.join(HERE, "results", "rff")
NYSTROM = os.path.join(HERE, "results", "nystrom", "rank.json")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
#: frequencies per field; each gives two feature columns
FREQS = (16, 32, 64, 128, 256, 512, 1024)
SAMPLERS = ("mc", "orthogonal", "qmc")
SEEDS = (0, 1, 2)
#: tolerances of the lattice: the share of its peak at which a spectrum is cut, and the overlap
#: at which a periodic image no longer matters
LATTICE_TOLS = (1e-3, 1e-6)
WIDTH_SETS = ("COX2", "THR")
WIDTH_ALPHAS = (0.1, 0.3, 1.0, 2.0, 5.0)
WIDTH_FREQS = (128, 512)
WIDTH_LANDMARKS = (40, 80)
WIDTH_SPACING = 2.0


def _fs_at(fs, alpha):
    fs2 = fs.copy()
    fs2.specs = specs_at(fs, alpha)
    return fs2


def _lattice(fs2, T, tol):
    """A lattice map at tolerance `tol`, each field with its own period at the same tolerance."""
    specs = [s for s in fs2.specs if s.enabled]
    ff = rff.build(fs2, T, sampler="lattice", tol=tol)
    return ff, max(rff.default_period(T, s, tol) for s in specs)


def _score(fs2, T, y, lam, K_ex, K_fields, seconds):
    K = joint_of(fs2, T, K_fields)
    q2, r = q2_at(K, y, lam)
    lam_own, q2_own = best_lam(K, y)
    return {"rel_error": rel_error(K_ex, K), "q2": q2, "rmse": r, "lam_own": lam_own,
            "q2_own": q2_own, "seconds": round(seconds, 3)}


def stage_rank(force=False):
    path = os.path.join(RESULTS, "rank.json")
    if os.path.exists(path) and not force:
        print("rank: already done")
        return json.load(open(path))
    chosen = {r["set"]: r for r in json.load(open(NYSTROM))["sets"]}
    out = {"freqs": list(FREQS), "samplers": list(SAMPLERS), "seeds": list(SEEDS),
           "lattice_tols": list(LATTICE_TOLS), "sets": []}
    for name in SETS:
        c = chosen[name]
        alpha, lam = float(c["alpha"]), float(c["lam"])
        fs, T, y = table_of(name, alpha)
        fs2 = _fs_at(fs, alpha)
        specs = [s for s in fs2.specs if s.enabled]
        t0 = time.time()
        K_fields = kernel_matrices(T, T, specs)
        t_exact = time.time() - t0
        K_ex = joint_of(fs2, T, K_fields)
        q2_ex, _ = q2_at(K_ex, y, lam)
        print(f"{name}: n {T.n_mol}, alpha {alpha:g}, lambda {lam:g}, exact q2 {q2_ex:+.4f} "
              f"(stored {c['q2_exact']:+.4f}), exact kernel {t_exact:.2f} s", flush=True)
        row = {"set": name, "n_mol": int(T.n_mol), "alpha": alpha, "lam": lam, "q2_exact": q2_ex,
               "seconds_exact": round(t_exact, 3), "random": [], "lattice": []}
        for sampler in SAMPLERS:
            for D in FREQS:
                draws = []
                for seed in SEEDS:
                    ff = rff.build(fs2, T, D, sampler, seed=seed)
                    t0 = time.time()
                    Ka = ff.field_kernels(T, normalize=False)
                    draws.append(_score(fs2, T, y, lam, K_ex, Ka, time.time() - t0))
                row["random"].append({"sampler": sampler, "freqs": D, "draws": draws})
                q = [d["q2"] for d in draws]
                print(f"   {sampler:10s} D {D:5d}  error "
                      f"{np.mean([d['rel_error'] for d in draws]):.1e}  q2 {np.mean(q):+.4f} "
                      f"[{min(q):+.4f}, {max(q):+.4f}]  {draws[0]['seconds']:.2f} s", flush=True)
        for tol in LATTICE_TOLS:
            ff, period = _lattice(fs2, T, tol)
            t0 = time.time()
            Ka = ff.field_kernels(T, normalize=False)
            s = _score(fs2, T, y, lam, K_ex, Ka, time.time() - t0)
            s.update({"tol": tol, "period": round(period, 2),
                      "freqs_per_field": [len(ff.omegas[g]) for g in ff.group]})
            row["lattice"].append(s)
            print(f"   lattice tol {tol:g} period {period:.1f} A  freqs {s['freqs_per_field']}  "
                  f"error {s['rel_error']:.1e}  q2 {s['q2']:+.4f}  {s['seconds']:.2f} s",
                  flush=True)
        out["sets"].append(row)
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(out, open(path, "w"), indent=1)
    return out


def stage_width(force=False):
    path = os.path.join(RESULTS, "width.json")
    if os.path.exists(path) and not force:
        print("width: already done")
        return json.load(open(path))
    out = {"alphas": list(WIDTH_ALPHAS), "freqs": list(WIDTH_FREQS),
           "landmarks": list(WIDTH_LANDMARKS), "spacing": WIDTH_SPACING, "sets": []}
    for name in WIDTH_SETS:
        fs, T, y = table_of(name)
        for alpha in WIDTH_ALPHAS:
            fs2 = _fs_at(fs, alpha)
            specs = [s for s in fs2.specs if s.enabled]
            K_fields = kernel_matrices(T, T, specs)
            K_ex = joint_of(fs2, T, K_fields)
            lam, q2_ex = best_lam(K_ex, y)
            row = {"set": name, "alpha": alpha, "lam": lam, "q2_exact": q2_ex, "qmc": [],
                   "molecules": [], "centres": None}
            for D in WIDTH_FREQS:
                draws = []
                for seed in SEEDS:
                    ff = rff.build(fs2, T, D, "qmc", seed=seed)
                    draws.append(_score(fs2, T, y, lam, K_ex,
                                        ff.field_kernels(T, normalize=False), 0.0))
                row["qmc"].append({"freqs": D, "draws": draws})
            for m in WIDTH_LANDMARKS:
                draws = []
                for seed in SEEDS:
                    idx = np.sort(np.random.RandomState(seed).choice(T.n_mol, m, replace=False))
                    Ka = nystrom(K_fields[:, :, idx], K_fields[:, idx][:, :, idx])
                    draws.append(_score(fs2, T, y, lam, K_ex, Ka, 0.0))
                row["molecules"].append({"m": m, "draws": draws})
            Z = gaussian_centres(T, WIDTH_SPACING, len(specs))
            Ka = nystrom(kernel_matrices(T, Z, specs, symmetric=False),
                         kernel_matrices(Z, Z, specs, symmetric=True))
            row["centres"] = {"m": int(Z.n_mol), **_score(fs2, T, y, lam, K_ex, Ka, 0.0)}
            q = lambda rows: " ".join(f"{np.mean([d['q2'] for d in r['draws']]):+.3f}"
                                      for r in rows)
            print(f"{name} alpha {alpha:g}: exact {q2_ex:+.3f}  qmc {q(row['qmc'])}  "
                  f"molecules {q(row['molecules'])}  centres {row['centres']['q2']:+.3f}",
                  flush=True)
            out["sets"].append(row)
            os.makedirs(RESULTS, exist_ok=True)
            json.dump(out, open(path, "w"), indent=1)
    return out


# ------------------------------------------------------------------------------ the report


def _mean(draws, key):
    return float(np.mean([d[key] for d in draws]))


def report(results=RESULTS):
    parts = {}
    for name in ("rank", "width"):
        p = os.path.join(results, f"{name}.json")
        if os.path.exists(p):
            parts[name] = json.load(open(p))
    if not parts:
        return "No results yet; run `python benchmarks/run_rff.py`."
    L = ["# Random Fourier features of the field kernel", "",
         "The field kernel is an overlap integral, so it is also an integral of the two fields' "
         "Fourier transforms over frequencies. Each of D frequencies gives a molecule two "
         "features per field (`cmfqsar/rff.py`). Every q² is 10-fold cross-validated on the "
         "folds of the Nyström benchmark (`results/nystrom`); `error` is the relative Frobenius "
         "error of the joint training kernel. `columns` is the number of feature columns per "
         "field: 2 D for frequencies, m for m Nyström landmarks.", ""]
    if "rank" in parts:
        k = parts["rank"]
        nys = {r["set"]: r for r in json.load(open(NYSTROM))["sets"]}
        L += ["## At the selected attenuation, against the exact model and Nyström", "",
              "Random frequencies: the mean over 3 seeds. `mc`: independent draws; "
              "`orthogonal`: orthogonal triples; `qmc`: a scrambled Sobol sequence. "
              "`lattice`: a cubic lattice of frequencies, cut where the spectrum falls below the "
              "tolerance, with a period at which the periodic images overlap the molecules by "
              "less than the same tolerance. Nyström rows are copied from "
              "`results/nystrom/table.md` (landmark molecules: mean of 3 random draws).", ""]
        summary = []
        for row in k["sets"]:
            L += [f"**{row['set']}** -- {row['n_mol']} molecules; alpha {row['alpha']:g}, "
                  f"lambda {row['lam']:g}, exact q² **{row['q2_exact']:+.4f}**, exact kernel "
                  f"{row['seconds_exact']:.2f} s", "",
                  "| features | columns per field | error | q² | q² (own ridge) | time |",
                  "|---|---:|---:|---:|---:|---:|"]
            for r in row["random"]:
                if r["freqs"] not in (32, 128, 512, 1024):
                    continue
                d = r["draws"]
                L.append(f"| Fourier, {r['sampler']} | {2 * r['freqs']} | "
                         f"{_mean(d, 'rel_error'):.1e} | {_mean(d, 'q2'):+.4f} | "
                         f"{_mean(d, 'q2_own'):+.4f} | {_mean(d, 'seconds'):.2f} s |")
            for r in row["lattice"]:
                cols = "/".join(sorted({str(2 * c) for c in r["freqs_per_field"]}, key=int))
                L.append(f"| Fourier, lattice (tolerance {r['tol']:g}) | {cols} | "
                         f"{r['rel_error']:.1e} | {r['q2']:+.4f} | {r['q2_own']:+.4f} | "
                         f"{r['seconds']:.2f} s |")
            for r in nys[row["set"]]["molecules"]:
                if r["m"] >= row["n_mol"]:
                    continue
                d = r["draws"]
                L.append(f"| Nyström, molecules | {r['m']} | {_mean(d, 'rel_error'):.1e} | "
                         f"{_mean(d, 'q2'):+.4f} | {_mean(d, 'q2_own'):+.4f} | "
                         f"~{r['seconds_kernels_estimate']:.2f} s |")
            for r in nys[row["set"]]["space"]:
                if r["spacing"] not in (2.0, 1.5):
                    continue
                L.append(f"| Nyström, Gaussian centres {r['spacing']:g} A | {r['m']} | "
                         f"{r['rel_error']:.1e} | {r['q2']:+.4f} | {r['q2_own']:+.4f} | "
                         f"{r['seconds']:.2f} s |")
            L.append("")
            # the fewest columns that bring q2 within 0.01 of the exact value
            def first(rows, cols_of, q_of):
                for r in rows:
                    if q_of(r) >= row["q2_exact"] - 0.01:
                        return cols_of(r)
                return None
            ent = {"set": row["set"]}
            for s in k["samplers"]:
                ent[s] = first([r for r in row["random"] if r["sampler"] == s],
                               lambda r: 2 * r["freqs"], lambda r: _mean(r["draws"], "q2"))
            ent["molecules"] = first(nys[row["set"]]["molecules"], lambda r: r["m"],
                                     lambda r: _mean(r["draws"], "q2"))
            ent["centres"] = first(sorted(nys[row["set"]]["space"], key=lambda r: r["m"]),
                                   lambda r: r["m"], lambda r: r["q2"])
            summary.append(ent)
        L += ["## The fewest columns per field that bring q² within 0.01 of the exact model", "",
              "The exact model's ridge is used throughout. A dash means that no size tried reached "
              f"it (Fourier: up to {2 * max(k['freqs'])} columns; Nyström molecules: up to 160 or "
              "the whole set; centres: down to 1.25 A).", "",
              "| set | Fourier, mc | Fourier, orthogonal | Fourier, qmc | Nyström, molecules | "
              "Nyström, Gaussian centres |", "|---|---:|---:|---:|---:|---:|"]
        for e in summary:
            L.append("| " + " | ".join([e["set"]] + [str(e[c]) if e[c] else "-" for c in
                                                     (*k["samplers"], "molecules", "centres")])
                     + " |")
        L.append("")
    if "width" in parts:
        w = parts["width"]
        L += ["## Narrow fields", "",
              "The same sets at attenuations from broad to narrow, each with the ridge that is best "
              "for the exact kernel. Fourier features with Sobol frequencies, Nyström with "
              "landmark molecules (mean of 3 draws) and with Gaussian centres "
              f"{w['spacing']:g} A apart.", "",
              "| set | alpha | exact q² | " +
              " | ".join(f"Fourier, {2 * D} columns" for D in w["freqs"]) + " | " +
              " | ".join(f"Nyström, {m} molecules" for m in w["landmarks"]) +
              " | Nyström, centres (columns) |",
              "|---|---:|---:|" + "---:|" * (len(w["freqs"]) + len(w["landmarks"]) + 1)]
        for r in w["sets"]:
            cells = [f"{_mean(x['draws'], 'q2'):+.3f}" for x in r["qmc"]]
            cells += [f"{_mean(x['draws'], 'q2'):+.3f}" for x in r["molecules"]]
            cells.append(f"{r['centres']['q2']:+.3f} ({r['centres']['m']})")
            L.append(f"| {r['set']} | {r['alpha']:g} | {r['q2_exact']:+.3f} | " +
                     " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


def write_table(results=RESULTS) -> str:
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                        # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="*", default=["rank", "width"], choices=["rank", "width"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    if "rank" in a.stage:
        stage_rank(a.force)
    if "width" in a.stage:
        stage_width(a.force)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
