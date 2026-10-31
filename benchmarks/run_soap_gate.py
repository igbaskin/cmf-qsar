#!/usr/bin/env python
"""
Is the angular term what the local kernel lacks?  SOAP as the gate (follow-up to plan items C5 and LP5)

Plan item LP5 found the rotation-invariant local kernel to be representation-limited: sixty times
the data took the carbon-shift error from 9.75 to only 7.76 ppm, so the route to a better local
model is a richer descriptor and not more data.  The descriptor in `cmfqsar/invariant.py` is a
two-point quantity -- a histogram of neighbour-neighbour distances inside the envelope -- and it
leaves the triangle formed by the centre and two neighbours unresolved.  SOAP (Smooth Overlap of
Atomic Positions, Bartok, Kondor and Csanyi 2013) resolves that triangle: its power spectrum at
angular order l > 0 carries the two radii and the angle at the centre.  At l = 0 it carries the
radii alone and is a two-body descriptor like ours.

So the question of whether the angular term is the missing piece can be answered without writing
any of it: compute the SOAP power spectrum with the dscribe library on the same atoms, put it
through the same ridge on the same folds holding out whole molecules, and compare three things.

1. **SOAP at its best against the two-point kernel at its best.**  If SOAP is not clearly better,
   a richer invariant descriptor is not where the error is, and nothing more needs building.
2. **SOAP at l_max = 0 against SOAP at l_max > 0.**  The ladder over l isolates the angular term
   from every other difference between the two descriptors (element channels against field
   channels, a hard cutoff against a Gaussian envelope, a radial basis against distance bins).
   The gate passes only when the gain appears on this ladder.
3. **On the aromatic carbons of plan item LP3, SOAP against the posed overlap kernel.**  That is
   the model with a frame, and it says whether an invariant descriptor can reach what the frame
   gives when a frame is available.

Everything is measured as the other local items are: the ridge chosen on the same grid, q2 and the
root-mean-square error out of fold, and differences with a bootstrap interval over molecules, so
a difference inside its interval is not a result (rule 4 of the plan).

The SOAP here has one channel per element.  A CMF-native version would have one channel per
field, with continuous weights; that is the stage this gate decides on, and it is not measured
here.

Usage
-----
    python benchmarks/run_soap_gate.py                 # both sets
    python benchmarks/run_soap_gate.py --only shifts   # the 5096 carbons of the repository set
    python benchmarks/run_soap_gate.py --only aromatic # the 1225 ring carbons of plan item LP3
    python benchmarks/run_soap_gate.py --max-molecules 40   # a quick look
    python benchmarks/run_soap_gate.py --report

Needs the `dscribe` package (``pip install dscribe``), which is not a dependency of the program.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import RDLogger  # noqa: E402

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.localalign import overlap_kernel, posed_environments, ring_frames  # noqa: E402
from cmfqsar.localenv import (choose_lam, count_features, count_kernel, environments,  # noqa: E402
                              local_features, local_kernel, paired_interval, types_of)

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "local", "shifts_cdcl3.sdf")
RESULTS = os.path.join(HERE, "results", "soap_gate")

#: the elements of the shift set; a SOAP has one density per element
SPECIES = ("H", "C", "N", "O", "S", "F", "Cl", "Br", "I")
#: the cutoff radii searched first, in A, at the full angular resolution
CUTOFFS = (3.0, 4.0, 5.0, 6.0)
#: the angular ladder at the best cutoff; 0 is the two-body SOAP
L_LADDER = (0, 1, 2, 4, 6, 8)
#: kernel exponents and density widths tried at the best cutoff and angular order
ZETAS = (1, 2, 4)
SIGMAS = (0.3, 0.5)
N_MAX = 8
L_MAX = 6
SIGMA = 0.5
ZETA = 2
#: envelope widths of the two-point kernel: the ones the earlier items found best for each set
RHOS = dict(shifts=(0.5, 1.0, 2.0), aromatic=(0.25, 0.5, 1.0))
#: the posed overlap kernel of plan item LP3 at the width that was best there
POSED_RHO = 0.25


def _say(text):
    print(text, flush=True)


# --------------------------------------------------------------------------- #
# the samples
# --------------------------------------------------------------------------- #

def shift_samples(path=DATA, prop="shift", max_molecules=0):
    """All measured carbons of the repository set, as plan items LP2 and LP5 use them."""
    ds = load_dataset(path, atom_activity=prop)
    mols, names = list(ds.mols), list(ds.names)
    if max_molecules:
        mols, names = mols[:max_molecules], names[:max_molecules]
    targets = AtomTargets.from_molecules(mols, prop, names=names)
    return mols, names, targets


def aromatic_ring_samples(max_molecules=0):
    """The ring carbons of plan item LP3, by the rule of that script."""
    from run_local_cofield import aromatic_samples
    return aromatic_samples(max_molecules=max_molecules)


# --------------------------------------------------------------------------- #
# SOAP rows and kernel
# --------------------------------------------------------------------------- #

def to_ase(mol):
    from ase import Atoms
    return Atoms(symbols=[a.GetSymbol() for a in mol.GetAtoms()],
                 positions=mol.GetConformer().GetPositions())


def soap_rows(mols, targets, r_cut, n_max=N_MAX, l_max=L_MAX, sigma=SIGMA, species=SPECIES):
    """The SOAP power spectrum of every measured atom, one row per sample of `targets`."""
    from dscribe.descriptors import SOAP
    soap = SOAP(species=list(species), r_cut=float(r_cut), n_max=int(n_max), l_max=int(l_max),
                sigma=float(sigma), periodic=False, sparse=False)
    rows = np.zeros((len(targets), soap.get_number_of_features()))
    for k in sorted({int(m) for m in targets.molecule}):
        r = targets.of_molecule(k)
        centres = [int(targets.atom[i]) for i in r]
        rows[r] = soap.create(to_ase(mols[k]), centers=centres)
    return rows


def soap_kernel(X, zeta=ZETA):
    """The SOAP kernel: the cosine of two power spectra, raised to `zeta`."""
    Z = np.asarray(X, float)
    d = np.sqrt(np.maximum(np.sum(Z * Z, axis=1), 1e-300))
    Z = Z / d[:, None]
    K = Z @ Z.T
    return np.clip(K, -1.0, 1.0) ** int(zeta) if int(zeta) != 1 else K


def _row(result, **extra):
    out = dict(q2=result.q2, error=result.error, mae=result.mae, lam=result.lam,
               note=result.note)
    out.update(extra)
    return out


def _interval(a, b, targets, seed):
    return [float(v) for v in paired_interval(a, b, targets, runs=2000, seed=seed)]


# --------------------------------------------------------------------------- #
# one set
# --------------------------------------------------------------------------- #

def run_set(which, folds=10, seed=0, max_molecules=0, results=RESULTS):
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} is missing; run examples/make_local_examples.py first")
    _say(f"=== {which}")
    if which == "shifts":
        mols, names, targets = shift_samples(max_molecules=max_molecules)
    else:
        mols, names, targets = aromatic_ring_samples(max_molecules=max_molecules)
    _say(f"{len(targets)} carbons in {targets.n_measured_molecules} molecules")
    fieldset = FieldSet.create()
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(mols, names)]

    # the reference of rule 1, and the two-point kernel at the widths the earlier items used
    types = types_of(fields, min_atoms=10)
    ref = choose_lam(count_kernel(count_features(fields, targets, types)), targets, folds=folds,
                     seed=seed, name="atom-centred counts")
    ref.note = f"{len(types)} atom types in 6 shells"
    _say("  " + ref.summary())
    two_point = []
    for rho in RHOS[which]:
        t0 = time.time()
        envs = environments(fields, targets, rho=rho)
        r = choose_lam(local_kernel(local_features(envs)), targets, folds=folds, seed=seed,
                       name=f"two-point local, rho {rho:g}")
        r.note = f"255 channels, {time.time() - t0:.0f} s"
        _say("  " + r.summary())
        two_point.append((rho, r))
    best_two = max(two_point, key=lambda t: t[1].q2)

    posed = None
    if which == "aromatic":
        t0 = time.time()
        envs = posed_environments(fields, targets, rho=POSED_RHO, frames_of=ring_frames)
        _model, _table, K = overlap_kernel(envs, fieldset)
        posed = choose_lam(K, targets, folds=folds, seed=seed,
                           name=f"posed overlap, rho {POSED_RHO:g}")
        posed.note = f"{time.time() - t0:.0f} s"
        _say("  " + posed.summary())

    def soap(r_cut, l_max=L_MAX, zeta=ZETA, sigma=SIGMA, n_max=N_MAX):
        t0 = time.time()
        X = soap_rows(mols, targets, r_cut, n_max=n_max, l_max=l_max, sigma=sigma)
        built = time.time() - t0
        r = choose_lam(soap_kernel(X, zeta), targets, folds=folds, seed=seed,
                       name=f"SOAP r_cut {r_cut:g} l_max {l_max} zeta {zeta} sigma {sigma:g}")
        r.note = f"{X.shape[1]} channels, rows {built:.0f} s, folds {time.time() - t0 - built:.0f} s"
        _say("  " + r.summary())
        return r

    # 1. the cutoff, at the full angular resolution
    _say("cutoff radius")
    cutoffs = [(c, soap(c)) for c in CUTOFFS]
    best_cut = max(cutoffs, key=lambda t: t[1].q2)[0]

    # 2. the angular ladder at that cutoff
    _say(f"angular ladder at r_cut {best_cut:g}")
    ladder = [(l, soap(best_cut, l_max=l)) for l in L_LADDER]
    best_l = max(ladder, key=lambda t: t[1].q2)[0]
    two_body_soap = ladder[0][1]

    # 3. the exponent and the density width at that cutoff and order
    _say(f"exponent and width at r_cut {best_cut:g}, l_max {best_l}")
    tuned = []
    for zeta in ZETAS:
        for sigma in SIGMAS:
            tuned.append((zeta, sigma, soap(best_cut, l_max=best_l, zeta=zeta, sigma=sigma)))
    best = max([t[2] for t in tuned] + [l[1] for l in ladder] + [c[1] for c in cutoffs],
               key=lambda r: r.q2)
    best_angular = max([l[1] for l in ladder if l[0] > 0], key=lambda r: r.q2)

    diffs = dict(
        best_soap_against_two_point=_interval(best, best_two[1], targets, seed),
        best_soap_against_counts=_interval(best, ref, targets, seed),
        two_body_soap_against_two_point=_interval(two_body_soap, best_two[1], targets, seed),
        best_angular_against_two_body_soap=_interval(best_angular, two_body_soap, targets, seed),
    )
    if posed is not None:
        diffs["best_soap_against_posed"] = _interval(best, posed, targets, seed)
    _say("differences in q2, bootstrapped over molecules")
    for k, v in diffs.items():
        _say(f"  {k:40s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")

    out = dict(set=which, n_samples=len(targets), n_molecules=targets.n_measured_molecules,
               folds=folds, seed=seed, species=list(SPECIES), n_max=N_MAX,
               reference=_row(ref),
               two_point=[_row(r, rho=float(rho)) for rho, r in two_point],
               best_two_point_rho=float(best_two[0]),
               posed=(_row(posed, rho=POSED_RHO) if posed is not None else None),
               cutoffs=[_row(r, r_cut=float(c), l_max=L_MAX, zeta=ZETA, sigma=SIGMA)
                        for c, r in cutoffs],
               best_cutoff=float(best_cut),
               ladder=[_row(r, r_cut=float(best_cut), l_max=int(l), zeta=ZETA, sigma=SIGMA)
                       for l, r in ladder],
               best_l_max=int(best_l),
               tuned=[_row(r, r_cut=float(best_cut), l_max=int(best_l), zeta=int(z), sigma=float(s))
                      for z, s, r in tuned],
               best=_row(best), differences=diffs)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, f"{which}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #

def _load(results, which):
    path = os.path.join(results, f"{which}.json")
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(d):
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"


def report(results=RESULTS):
    lines = ["# SOAP as the gate for a richer invariant local kernel", ""]
    lines += ["SOAP with one channel per element (dscribe), against the two-point local kernel of "
              "`cmfqsar/invariant.py`, on the same atoms, the same ridge grid and the same folds "
              "holding out whole molecules. Errors are root-mean-square, out of fold, in ppm. "
              "Differences are in q2 with a 95% bootstrap interval over molecules.", ""]
    for which, title in (("shifts", "All measured carbons of the repository set (plan item LP2)"),
                         ("aromatic", "Aromatic ring carbons (plan item LP3)")):
        d = _load(results, which)
        if d is None:
            continue
        lines += [f"## {title}", "",
                  f"{d['n_samples']} carbons in {d['n_molecules']} molecules, {d['folds']} folds.", "",
                  "| model | q2 | error (ppm) | ridge |", "|---|---|---|---|"]
        r = d["reference"]
        lines.append(f"| atom-centred counts | {r['q2']:+.3f} | {r['error']:.2f} | {r['lam']:g} |")
        for r in d["two_point"]:
            lines.append(f"| two-point local kernel, rho {r['rho']:g} | {r['q2']:+.3f} | "
                         f"{r['error']:.2f} | {r['lam']:g} |")
        if d.get("posed"):
            r = d["posed"]
            lines.append(f"| posed overlap kernel, rho {r['rho']:g} | {r['q2']:+.3f} | "
                         f"{r['error']:.2f} | {r['lam']:g} |")
        for r in d["cutoffs"]:
            lines.append(f"| SOAP r_cut {r['r_cut']:g} A, l_max {r['l_max']}, zeta {r['zeta']} | "
                         f"{r['q2']:+.3f} | {r['error']:.2f} | {r['lam']:g} |")
        for r in d["ladder"]:
            lines.append(f"| SOAP r_cut {r['r_cut']:g} A, l_max {r['l_max']}, zeta {r['zeta']} | "
                         f"{r['q2']:+.3f} | {r['error']:.2f} | {r['lam']:g} |")
        for r in d["tuned"]:
            lines.append(f"| SOAP r_cut {r['r_cut']:g} A, l_max {r['l_max']}, zeta {r['zeta']}, "
                         f"sigma {r['sigma']:g} | {r['q2']:+.3f} | {r['error']:.2f} | {r['lam']:g} |")
        lines += ["", "Differences in q2:", ""]
        names = dict(best_soap_against_two_point="best SOAP against the best two-point kernel",
                     best_soap_against_counts="best SOAP against the counts",
                     two_body_soap_against_two_point="two-body SOAP (l_max 0) against the best "
                                                     "two-point kernel",
                     best_angular_against_two_body_soap="best SOAP with l_max > 0 against "
                                                        "two-body SOAP, same cutoff",
                     best_soap_against_posed="best SOAP against the posed overlap kernel")
        for k, v in d["differences"].items():
            lines.append(f"- {names.get(k, k)}: {_fmt(v)}")
        lines.append("")
    return "\n".join(lines)


def write_table(results=RESULTS):
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="all", choices=["all", "shifts", "aromatic"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-molecules", type=int, default=0)
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    args = ap.parse_args()
    if args.report is not None:
        text = write_table()
        if args.report == "-":
            print(text)
        else:
            with io.open(args.report, "w", encoding="utf-8") as fh:
                fh.write(text)
        return
    try:
        import dscribe  # noqa: F401
    except ImportError:
        raise SystemExit("this benchmark needs the dscribe package: pip install dscribe")
    for which in (("shifts", "aromatic") if args.only == "all" else (args.only,)):
        run_set(which, folds=args.folds, seed=args.seed, max_molecules=args.max_molecules)
    write_table()


if __name__ == "__main__":
    main()
