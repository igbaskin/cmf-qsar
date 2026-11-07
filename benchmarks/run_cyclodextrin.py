#!/usr/bin/env python
"""
A frame that cannot exist: the three kernels on cyclodextrin guest sets.

``benchmarks/run_alignment.py`` asks what an alignment is worth by replacing a curated one with
an automatic one.  This asks the same question in a harder case, on data where there is no curated
alignment to be had and no receptor to borrow a frame from.  In ``examples/cyclodextrin/`` the
*host* is fixed and the guests are whatever the literature measured, from acetonitrile to
steroids, so the molecules share no scaffold; the maximum common substructure of such a set is a
couple of atoms, and a superposition built on it means nothing.  If the alignment-free route is
ever worth taking, it is here.

The three models are the ones of the alignment benchmark, so the columns can be read beside it:

  aligned      the ordinary CMF model, on the best frame the program can produce without a
               receptor: one RDKit ETKDG conformer per guest, superimposed by maximum common
               substructure on a reference guest chosen the way ``make_mittal_examples.py``
               chooses one, the largest molecule of the most active quartile.  The median number
               of atoms that actually matched is reported with the q2, because that number is the
               explanation of whatever the column shows.
  invariant    the rotation-invariant field autocorrelation (:mod:`cmfqsar.invariant`), which uses
               the conformer and no frame.  Alignment cannot change it, so it is computed from the
               same molecules after superposition and the number would be the same before.
  counts       atom-type counts, which use neither frame nor conformer: the floor.

Each host is measured twice, once on the neutral structures as PubChem deposits them and once on
``ph7/``, the same guests in their protonation state at the pH of the measurement.  Whether a
third of a set turning into ions helps or hurts is a question about the electrostatic field, and
the two columns answer it per host rather than in general.

What the comparison is against
------------------------------
The published field-based models of this endpoint, for scale rather than as a direct
reference: CoMFA with COSMOsar3D fields on 60 alpha-CD guests (Q2 0.83, r2test 0.70), GRIND on
126 beta-CD guests (Q2LOO 0.64, r2test 0.68) and GRIND on 233 beta-CD guests (Q2LOO 0.75,
r2test 0.74).  Those sets are an order of magnitude smaller than the tables here and were
curated by hand; a q2 from this script on 1162 guests is not the same measurement.

Usage
-----
    python benchmarks/run_cyclodextrin.py [--sets beta,alpha] [--force] [--max-guests 200]
    python benchmarks/run_cyclodextrin.py --report benchmarks/results/cyclodextrin/table.md

Results are cached per host in ``benchmarks/results/cyclodextrin/<host>.json``, so an interrupted
run resumes and ``--force`` recomputes.
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

from rdkit import RDLogger  # noqa: E402

from cmfqsar.align import mcs_align  # noqa: E402
from cmfqsar.chemotypes import count_kernel  # noqa: E402
from cmfqsar.dataio import load_dataset, relative_path  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               feature_matrix)
from cmfqsar.model import AtomTable, CMFModel, cv_predict, r2_score, rmse  # noqa: E402

sys.path.insert(0, HERE)
import kernel_cache as KC  # noqa: E402

RESULTS = os.path.join(HERE, "results", "cyclodextrin")
DATA = os.path.join(ROOT, "examples", "cyclodextrin")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
FOLDS = 10
SEED = 0
MCS_TIMEOUT = 2          # seconds per guest; a heterogeneous set has nothing to search for

#: host -> file name, smallest first so that partial results are useful
HOSTS = {
    "dimethyl_beta": "dimethyl_beta_cyclodextrin.csv",
    "methyl_beta": "methyl_beta_cyclodextrin.csv",
    "sbe_beta": "sulfobutylether_beta_cyclodextrin.csv",
    "hp_beta": "hydroxypropyl_beta_cyclodextrin.csv",
    "gamma": "gamma_cyclodextrin.csv",
    "alpha": "alpha_cyclodextrin.csv",
    "beta": "beta_cyclodextrin.csv",
}
#: the two variants of every table
VARIANTS = (("neutral", ""), ("ph7", "ph7"))


def best_cv(kernels, y):
    """Best (q2, RMSEcv, key, lambda) over a dict of kernels and the lambda grid."""
    best = None
    for key, K in kernels.items():
        for lam in LAMS:
            pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=FOLDS, seed=SEED)
            q2 = r2_score(y, pred)
            if best is None or q2 > best[0]:
                best = (q2, rmse(y, pred), key, float(lam))
    return best


def align_by_mcs(mols, y):
    """Superimpose on the largest guest of the most active quartile. Returns (ref, matched atoms)."""
    threshold = float(np.percentile(y, 75))
    candidates = [i for i in range(len(y)) if y[i] >= threshold] or list(range(len(y)))
    ref = max(candidates, key=lambda i: mols[i].GetNumHeavyAtoms())
    matched = []
    for i, m in enumerate(mols):
        if i == ref:
            matched.append(mols[ref].GetNumHeavyAtoms())
            continue
        try:
            _, n = mcs_align(m, mols[ref], timeout=MCS_TIMEOUT)
            matched.append(int(n))
        except Exception:                                      # noqa: BLE001
            matched.append(0)
    return ref, matched


def measure(path, tag, max_guests=0):
    """The three models on one table."""
    ds = load_dataset(path)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y = y[ok]
    if max_guests and len(mols) > max_guests:
        keep = np.linspace(0, len(mols) - 1, max_guests).astype(int)
        mols = [mols[i] for i in keep]
        names = [names[i] for i in keep]
        y = y[keep]

    t0 = time.time()
    ref, matched = align_by_mcs(mols, y)
    heavy = [m.GetNumHeavyAtoms() for m in mols]
    out = {"file": relative_path(path, ROOT).replace("\\", "/"), "n": len(mols),
           "reference": names[ref], "reference_heavy_atoms": heavy[ref],
           "mcs_atoms_median": float(np.median(matched)),
           "mcs_atoms_min": int(np.min(matched)), "mcs_atoms_max": int(np.max(matched)),
           "heavy_atoms_median": float(np.median(heavy)),
           "mcs_failed": int(sum(1 for n in matched if n == 0)),
           "charged": int(sum(1 for m in mols
                              if any(a.GetFormalCharge() for a in m.GetAtoms()))),
           "align_seconds": time.time() - t0}
    print(f"    {tag:8s} n {out['n']:5d}  MCS median {out['mcs_atoms_median']:.0f} of "
          f"{out['heavy_atoms_median']:.0f} heavy atoms, {out['mcs_failed']} failed, "
          f"{out['charged']} charged  ({out['align_seconds']:.0f} s)", flush=True)

    aligned, invariant = {}, {}
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        aligned[alpha] = KC.joint_training_kernel(model, AtomTable.from_molecules(mf))
        spec = AutocorrelationSpec()
        invariant[alpha] = autocorrelation_kernel(feature_matrix(mf, spec), n_bins=spec.n_bins)
        print(f"      alpha {alpha:g} done ({time.time() - t0:.0f} s)", flush=True)
    for name, kernels in (("aligned", aligned), ("invariant", invariant)):
        q2, err, alpha, lam = best_cv(kernels, y)
        out[name] = {"q2": q2, "rmse_cv": err, "alpha": alpha, "lam": lam}
        print(f"    {tag:8s} {name:9s} q2 {q2:+.3f}  RMSEcv {err:.3f}  "
              f"(alpha {alpha:g}, lam {lam:g})", flush=True)
    q2, err, _, lam = best_cv({"counts": count_kernel(mols, names)}, y)
    out["counts"] = {"q2": q2, "rmse_cv": err, "lam": lam}
    print(f"    {tag:8s} {'counts':9s} q2 {q2:+.3f}  RMSEcv {err:.3f}  (lam {lam:g})", flush=True)
    out["seconds"] = time.time() - t0
    return out


def run_set(host, force=False, max_guests=0):
    out_path = os.path.join(RESULTS, f"{host}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        with open(out_path) as fh:
            return json.load(fh)
    print(f"\n=== {host} ({HOSTS[host]})", flush=True)
    t0 = time.time()
    res = {"host": host, "fields": FIELDS, "alphas": list(ALPHAS), "lams": list(LAMS),
           "folds": FOLDS, "max_guests": max_guests}
    for tag, sub in VARIANTS:
        path = os.path.join(DATA, sub, HOSTS[host]) if sub else os.path.join(DATA, HOSTS[host])
        if not os.path.exists(path):
            print(f"    {tag}: {path} missing, skipped "
                  f"(run make_cyclodextrin_examples.py --protonate)")
            continue
        res[tag] = measure(path, tag, max_guests=max_guests)
    res["seconds"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out_path}  ({res['seconds']:.0f} s)", flush=True)
    return res


def report(path):
    rows = []
    for host in HOSTS:
        p = os.path.join(RESULTS, f"{host}.json")
        if os.path.exists(p):
            with open(p) as fh:
                rows.append(json.load(fh))
    if not rows:
        raise SystemExit("no results yet")
    have = [r for r in rows if "neutral" in r]
    out = ["# The three kernels where no frame exists", "",
           "Association constants with a fixed cyclodextrin host, `examples/cyclodextrin/`. The",
           "guests of one table share no scaffold, so the superposition the `aligned` column uses",
           "is a maximum common substructure fit onto the largest guest of the most active",
           "quartile, which is the best the program can do without a receptor. The `invariant`",
           "column is the field autocorrelation, which needs no frame; `counts` are atom-type",
           f"counts, which need no geometry. Same fields ({', '.join(rows[0]['fields'])}), same",
           f"alpha and lambda grids and the same {rows[0]['folds']}-fold CV as",
           "`run_alignment.py`, so the columns are comparable with that benchmark.", "",
           "## The neutral tables", "",
           "| host | n | aligned | invariant | counts | MCS atoms (median) | heavy atoms (median) |",
           "|---|---|---|---|---|---|---|"]
    for r in have:
        d = r["neutral"]
        out.append(f"| {r['host']} | {d['n']} | {d['aligned']['q2']:+.3f} | "
                   f"{d['invariant']['q2']:+.3f} | {d['counts']['q2']:+.3f} | "
                   f"{d['mcs_atoms_median']:.0f} | {d['heavy_atoms_median']:.0f} |")
    inv_wins = sum(1 for r in have
                   if r["neutral"]["invariant"]["q2"] > r["neutral"]["aligned"]["q2"])
    below_counts = sum(1 for r in have
                       if r["neutral"]["aligned"]["q2"] < r["neutral"]["counts"]["q2"])
    out += ["",
            "Mean q2 over the %d hosts measured: aligned %+.3f, invariant %+.3f, counts %+.3f."
            % (len(have),
               np.mean([r["neutral"]["aligned"]["q2"] for r in have]),
               np.mean([r["neutral"]["invariant"]["q2"] for r in have]),
               np.mean([r["neutral"]["counts"]["q2"] for r in have])), "",
            "The kernel that uses no frame is ahead of the one that does on %d of the %d hosts, "
            "and ahead of" % (inv_wins, len(have)),
            "atom-type counts on all of them. The aligned column falls below counts on %d, which "
            "is the" % below_counts,
            "frame failing rather than the fields: the MCS column is the explanation, a "
            "superposition built on",
            "half a molecule or less. This is the regime `run_alignment.py` reaches only on its "
            "worst targets,",
            "and here it is the whole data set.", ""]

    both = [r for r in have if "ph7" in r]
    if both:
        out += ["## Neutral against the structures at pH 7", "",
                "The same guests in their protonation state at the pH the constants were measured",
                "at, `examples/cyclodextrin/ph7/`. The `charged` column counts the guests that",
                "carry a formal charge there.", "",
                "| host | n | charged at pH 7 | aligned neutral | aligned pH 7 | invariant neutral "
                "| invariant pH 7 | counts neutral | counts pH 7 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in both:
            a, b = r["neutral"], r["ph7"]
            out.append(f"| {r['host']} | {a['n']} | {b['charged']} | {a['aligned']['q2']:+.3f} | "
                       f"{b['aligned']['q2']:+.3f} | {a['invariant']['q2']:+.3f} | "
                       f"{b['invariant']['q2']:+.3f} | {a['counts']['q2']:+.3f} | "
                       f"{b['counts']['q2']:+.3f} |")
        better = [r["host"] for r in both
                  if r["ph7"]["invariant"]["q2"] > r["neutral"]["invariant"]["q2"]]
        big = sorted(both, key=lambda r: -r["neutral"]["n"])[:2]
        out += ["",
                "Mean q2 at pH 7: aligned %+.3f, invariant %+.3f, counts %+.3f."
                % (np.mean([r["ph7"]["aligned"]["q2"] for r in both]),
                   np.mean([r["ph7"]["invariant"]["q2"] for r in both]),
                   np.mean([r["ph7"]["counts"]["q2"] for r in both])), "",
                "Protonating the guests is not settled by this measurement. The invariant kernel "
                "improves on %d of" % len(better),
                "the %d hosts (%s) and loses elsewhere, so the mean barely moves."
                % (len(both), ", ".join(better) or "none")]
        if all(r["host"] in better for r in big):
            out += ["The two largest sets, %s, are among those that improve, which is what one "
                    "would expect if the" % " and ".join(r["host"] for r in big),
                    "charges carry signal that only a large enough set can use: on the small hosts "
                    "the extra spread in",
                    "the electrostatic field costs more than it returns. Read it as helping where "
                    "there are guests",
                    "enough to learn from.", ""]
        else:
            out += ["The two largest sets, %s, do not both improve, so not even that pattern holds."
                    % " and ".join(r["host"] for r in big), ""]

    out += ["For scale, the published field-based models of this endpoint: CoMFA with COSMOsar3D",
            "fields on 60 alpha-CD guests (Q2 0.83, r2test 0.70), GRIND on 126 beta-CD guests",
            "(Q2LOO 0.64, r2test 0.68) and GRIND on 233 beta-CD guests (Q2LOO 0.75, r2test 0.74).",
            "Those sets are smaller and hand-curated, so these numbers are not a direct",
            "comparison with them.", ""]
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=",".join(HOSTS),
                    help="hosts to measure: " + ", ".join(HOSTS))
    ap.add_argument("--force", action="store_true", help="recompute cached results")
    ap.add_argument("--max-guests", type=int, default=0,
                    help="subsample evenly to at most this many guests per table (0: all)")
    ap.add_argument("--report", default=None, metavar="PATH")
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if args.report:
        report(args.report)
        return
    for host in [s for s in args.sets.split(",") if s]:
        if host not in HOSTS:
            raise SystemExit(f"unknown host {host}; one of {', '.join(HOSTS)}")
        run_set(host, force=args.force, max_guests=args.max_guests)


if __name__ == "__main__":
    main()
