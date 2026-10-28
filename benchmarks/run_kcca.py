#!/usr/bin/env python
"""
What the field sees that a fingerprint does not: kernel canonical correlation analysis (F6).

`cmfqsar/kcca.py` finds, for two kernels over the same molecules, the pairs of directions whose
projections correlate most strongly, judged on held-out molecules and against a shuffled pairing
of the two views.  The pairing measured first is the one that puts CMF's own premise to the
test: the five-field kernel against the Tanimoto kernel of 2048-bit ECFP4.  The field is
three-dimensional and the fingerprint is not, so the directions of the field that the fingerprint
does not contain are the part of the description that is genuinely three-dimensional, and the
question is whether that part is where the activity is.

Three gates, each on the eight Sutherland sets:

(i)   **The correlations are real.**  The canonical correlations on held-out molecules stay above
      the ones a shuffled pairing of the two views produces.  Without this the correlations are
      the degeneracy of the method (every correlation is 1 when the ridge is too small), not
      chemistry.
(ii)  **The field's own part carries activity.**  The field kernel is split into the part the
      fingerprint contains (the directions of which it explains more than half the variance on
      held-out molecules) and the rest; kernel ridge regression on each, with the regularisation
      chosen inside the folds, and the difference of the two q2 with its bootstrap interval.
      Beside it, a control: a random subspace of the same dimension as the contained part, so
      that "the contained part predicts" can be told apart from "any part of that size predicts".
(iii) **The components are readable.**  A fingerprint has no co-field, so this gate needs two
      field views on the same fields in one frame: the molecules as posed in the file against
      the same molecules with every pose moved by a small random rigid motion (up to 10 degrees
      and 0.5 A).  The co-fields of the two sides of one shared component should resemble each
      other, by the cosine of plan item F3, more than the co-fields of two different components
      do.

The attenuation of each set is the one the evidence of a kernel ridge regression picks for the
whole set, as `run_transfer.py` does, so that the field view is the model's own view.  Nothing
else is tuned: the ridge of the analysis is chosen by the held-out correlation inside the
analysis itself.

Usage
-----
    python benchmarks/run_kcca.py
    python benchmarks/run_kcca.py --sets THR COX2 --force
    python benchmarks/run_kcca.py --report
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

from rdkit import Chem, RDLogger  # noqa: E402

import kernel_cache as KCACHE  # noqa: E402
from cmfqsar import kcca as KC  # noqa: E402
from cmfqsar.align import random_rigid_transform  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.descriptors import fingerprint_kernel  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "kcca")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0, 2.0)
LAMS = (1e-2, 0.1, 1.0)
COMPONENTS = 3          # leading pairs reported, and co-fields compared in gate (iii)
PERMUTATIONS = 200
FOLDS = 5
SEED = 0
RANDOM_SUBSPACES = 5    # draws of the control subspace of gate (ii)
MOVE_ROTATION = 10.0    # degrees, the pose perturbation of gate (iii)
MOVE_TRANSLATION = 0.5  # angstroms


def _field_view(mols, names, alpha):
    """The cosine-normalised five-field kernel at `alpha`, through the cache; returns the probe."""
    fs = FieldSet.create(physchem=FIELDS, alpha=float(alpha), charge_model="file")
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    probe = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = probe.joint_kernel(KCACHE.training_kernels(probe, table))
    return probe, K


def _choose_alpha(mols, names, y):
    best = None
    for a in ALPHAS:
        _probe, K = _field_view(mols, names, a)
        for lam in LAMS:
            e = log_evidence(K, y, float(lam), fit_bias=True)
            if np.isfinite(e) and (best is None or e > best[0]):
                best = (e, float(a))
    return best[1]


def _moved(mols, seed=SEED):
    rng = np.random.RandomState(seed)
    out = []
    for m in mols:
        mm = Chem.Mol(m)
        conf = mm.GetConformer()
        X = random_rigid_transform(np.array(conf.GetPositions(), float), rng,
                                   max_rotation=MOVE_ROTATION, max_translation=MOVE_TRANSLATION)
        for i, xyz in enumerate(X):
            conf.SetAtomPosition(i, [float(v) for v in xyz])
        out.append(mm)
    return out


def _random_subspace_q2(fit, Kc, y, k, draws=RANDOM_SUBSPACES, seed=SEED):
    """q2 of kernel ridge regression on `draws` random sets of `k` canonical directions."""
    if k <= 0 or k >= fit.n_components:
        return float("nan")
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(int(draws)):
        idx = np.sort(rng.choice(fit.n_components, size=int(k), replace=False))
        q, _cv, _lam = KC.activity_on_kernel(KC.projected_kernel(fit, idx), y, folds=FOLDS,
                                             seed=seed)
        out.append(q)
    return float(np.nanmean(out))


def run_set(name, force=False, data=DATA, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(os.path.join(data, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y = y[ok]
    alpha = _choose_alpha(mols, names, y)
    probe, K = _field_view(mols, names, alpha)
    print(f"=== {name}: {len(mols)} molecules, alpha {alpha:g}", flush=True)

    # ---- pairing 1: the field against the fingerprint
    L = fingerprint_kernel(mols)
    res = KC.analyse(K, L, y=y, ridge=None, components=COMPONENTS, folds=FOLDS, seed=SEED,
                     permutations=PERMUTATIONS)
    held = res.held_out.mean
    n_contained = res.n_contained()
    Kc = KC.centre(K)
    q2_random = _random_subspace_q2(res.fit, Kc, y, n_contained)
    d, lo, hi = res.q2_difference
    fingerprint = {
        "ridge": res.fit.ridge, "ridge_scores": res.ridge_choice.scores,
        "pairs": int(res.fit.n_components), "pairs_held_out": int(len(held)),
        "rho_held_out": [float(v) for v in held[:10]],
        "rho_held_out_mean": float(np.nanmean(held)),
        "null_95": [float(v) for v in res.null_quantile()[:10]],
        "real": int(res.real_components()),
        "contained": n_contained,
        "q2_all": res.q2_all, "q2_contained": res.q2_shared, "q2_own": res.q2_private,
        "q2_random_same_size": q2_random,
        "own_minus_contained": [d, lo, hi],
        "rows": res.rows(),
    }
    print(f"  fingerprint: ridge {res.fit.ridge:g}; held-out rho of the first three "
          f"{np.round(held[:3], 3)}; {fingerprint['real']} of {len(held)} real; "
          f"{n_contained} of {res.fit.n_components} contained", flush=True)
    print(f"  q2 whole {res.q2_all:+.3f}  contained {res.q2_shared:+.3f}  own {res.q2_private:+.3f}"
          f"  random of the same size {q2_random:+.3f}  own-contained {d:+.3f} [{lo:+.3f}, {hi:+.3f}]",
          flush=True)

    # ---- pairing 2: the field against the same field in moved poses (gate iii)
    probe_b, Kb = _field_view(_moved(mols), names, alpha)
    res_b = KC.analyse(K, Kb, y=None, ridge=None, components=COMPONENTS, folds=FOLDS, seed=SEED,
                       permutations=max(50, PERMUTATIONS // 4))
    k = min(COMPONENTS, res_b.fit.n_components)
    ma = [KC.component_model(probe, res_b.fit.a[:, i], res_b.fit.u[:, i]) for i in range(k)]
    mb = [KC.component_model(probe_b, res_b.fit.b[:, i], res_b.fit.v[:, i]) for i in range(k)]
    M = KC.component_cosines(ma, mb)
    diag, off = KC.cosine_summary(M)
    held_b = res_b.held_out.mean
    poses = {"ridge": res_b.fit.ridge, "rho_held_out": [float(v) for v in held_b[:10]],
             "rho_held_out_mean": float(np.nanmean(held_b)), "real": int(res_b.real_components()),
             "pairs_held_out": int(len(held_b)), "cosines": M.tolist(),
             "cosine_same": diag, "cosine_other": off}
    print(f"  moved poses: held-out rho of the first three {np.round(held_b[:3], 3)}; co-field "
          f"cosine of the same component {diag:+.3f}, of different components {off:.3f}",
          flush=True)

    out = {"set": name, "n": len(mols), "alpha": alpha, "fingerprint": fingerprint,
           "poses": poses, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def report(results=RESULTS):
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No KCCA results yet; run `python benchmarks/run_kcca.py`."
    data = sorted([json.load(open(f)) for f in files], key=lambda d: -d["n"])
    lines = ["# What the field sees that the fingerprint does not", "",
             "Each Sutherland set, the five-field kernel at the attenuation the evidence picks "
             "for the set against the Tanimoto kernel of 2048-bit ECFP4. *Held-out ρ* is the "
             "mean canonical correlation over all pairs on molecules the directions were not "
             "fitted on; *real* counts the pairs above the 95th percentile of a shuffled "
             "pairing; *contained* counts the directions of the field of which the fingerprint "
             "explains more than half the variance on held-out molecules. The three q² values "
             "are kernel ridge regression on the whole field kernel, on its part inside the "
             "contained directions, and on the rest, the regularisation chosen inside the folds; "
             "*random* is a subspace of the same dimension as the contained part, drawn at "
             f"random {RANDOM_SUBSPACES} times.", "",
             "| set | n | α | ridge | held-out ρ (first three) | mean ρ | real | contained | "
             "q² whole | q² contained | q² own | q² random, same size | own − contained [95 %] |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    diffs, owns, conts, wholes, rands = [], [], [], [], []
    above = below = 0
    for d in data:
        f = d["fingerprint"]
        dd, lo, hi = f["own_minus_contained"]
        if np.isfinite(lo) and lo > 0:
            above += 1
        if np.isfinite(hi) and hi < 0:
            below += 1
        diffs.append(dd)
        owns.append(f["q2_own"])
        conts.append(f["q2_contained"])
        wholes.append(f["q2_all"])
        rands.append(f["q2_random_same_size"])
        first = " ".join(f"{v:+.2f}" for v in f["rho_held_out"][:3])
        lines.append(f"| {d['set']} | {d['n']} | {d['alpha']:g} | {f['ridge']:g} | {first} | "
                     f"{f['rho_held_out_mean']:+.3f} | {f['real']}/{f['pairs_held_out']} | "
                     f"{f['contained']}/{f['pairs']} | {f['q2_all']:+.3f} | "
                     f"{f['q2_contained']:+.3f} | {f['q2_own']:+.3f} | "
                     f"{f['q2_random_same_size']:+.3f} | {dd:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines.append(f"| **mean** |  |  |  |  |  |  |  | **{np.nanmean(wholes):+.3f}** | "
                 f"**{np.nanmean(conts):+.3f}** | **{np.nanmean(owns):+.3f}** | "
                 f"**{np.nanmean(rands):+.3f}** | **{np.nanmean(diffs):+.3f}** |")
    n = len(data)
    reals = [d["fingerprint"]["real"] / max(d["fingerprint"]["pairs_held_out"], 1) for d in data]
    shares = [d["fingerprint"]["contained"] / max(d["fingerprint"]["pairs"], 1) for d in data]
    lines += ["",
              f"**(i) The correlations are real.** On average {100 * np.mean(reals):.0f} % of "
              f"the canonical pairs hold out above the shuffled pairing, and the leading pairs "
              f"hold out at correlations near one on every set: the field and the fingerprint "
              f"describe the same molecules and agree about most of what varies between them. "
              f"The fingerprint *contains* -- explains more than half the variance of -- "
              f"{100 * np.mean(shares):.0f} % of the field's directions on average; the rest of "
              f"the field is its own.", "",
              f"**(ii) Where the activity is.** The part of the field the fingerprint does not "
              f"contain predicts the activity at q² **{np.nanmean(owns):+.3f}** on average, "
              f"against **{np.nanmean(conts):+.3f}** for the part it contains and "
              f"**{np.nanmean(wholes):+.3f}** for the whole field; a random subspace of the "
              f"contained part's size reaches {np.nanmean(rands):+.3f}. The difference (own minus "
              f"contained) is **{np.nanmean(diffs):+.3f}** on average, with its interval above "
              f"zero on {above} of {n} sets and below zero on {below} of {n}."]
    if above > n / 2:
        lines.append("So the field directions the fingerprint does not contain are where most of "
                     "the activity sits: the three-dimensional part of the description earns its "
                     "place on these sets.")
    elif below > n / 2:
        lines.append("So on these sets the activity sits mostly in the field directions the "
                     "fingerprint already contains: the three-dimensional part of the description "
                     "adds less than the shared part carries. That is a measured answer of the "
                     "same kind as D5, and it says what the item asked to be told either way.")
    else:
        lines.append("So on most sets the interval covers zero: the two parts carry comparable "
                     "shares of the activity, and neither claim -- that the three-dimensional "
                     "part is where the activity is, or that the fingerprint already holds it -- "
                     "is a result under rule 4 on sets of this size.")
    lines += ["", "## The components are readable: the same field in moved poses", "",
              "A fingerprint has no co-field, so the third gate is measured on two field views "
              "of the same molecules on the same fields: the poses of the file against the same "
              f"poses moved by a random rigid motion of up to {MOVE_ROTATION:g} degrees and "
              f"{MOVE_TRANSLATION:g} Å. *Same* is the mean F3 cosine between the co-fields of the "
              f"two sides of one component over the first {COMPONENTS}; *different* is the mean "
              "absolute cosine between the co-fields of two different components.", "",
              "| set | ridge | held-out ρ (first three) | real | co-field cosine, same component | "
              "different components |",
              "|---|---:|---|---:|---:|---:|"]
    same, diff = [], []
    for d in data:
        p = d["poses"]
        first = " ".join(f"{v:+.2f}" for v in p["rho_held_out"][:3])
        lines.append(f"| {d['set']} | {p['ridge']:g} | {first} | {p['real']}/{p['pairs_held_out']} "
                     f"| {p['cosine_same']:+.3f} | {p['cosine_other']:.3f} |")
        same.append(p["cosine_same"])
        diff.append(p["cosine_other"])
    lines.append(f"| **mean** |  |  |  | **{np.nanmean(same):+.3f}** | **{np.nanmean(diff):.3f}** |")
    wins = int(np.sum(np.asarray(same) > np.asarray(diff)))
    lines += ["",
              f"**(iii)** The two co-fields of one component agree at a cosine of "
              f"**{np.nanmean(same):+.3f}** on average, against {np.nanmean(diff):.3f} between "
              f"different components, on {wins} of {n} sets. "
              + ("A canonical component is one picture, drawn from either side." if wins == n
                 else "On the sets where it fails the leading directions were not stable under "
                      "the motion, and the component is not one picture.")]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:                       # the report is written as UTF-8; a console may not be
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:          # noqa: BLE001 - an output stream that cannot be reconfigured
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        for name in a.sets:
            run_set(name, force=a.force)
    print(report())


if __name__ == "__main__":
    main()
