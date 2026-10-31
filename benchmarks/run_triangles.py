#!/usr/bin/env python
"""
The three-distance autocorrelation against the two-point kernel and against SOAP (plan item C5)

`benchmarks/run_soap_gate.py` showed that what the rotation-invariant local kernel lacks is the
angle at the centre: SOAP, which resolves the triangle of the centre and two neighbours, lowers
the carbon-shift error by a third, and a SOAP restricted to its two-body part does not.
`cmfqsar/triangles.py` resolves that triangle in the terms of this program -- the pair sum of the
environment over the two distances from the centre and the distance between the pair, with the
field weights as the channels -- and this script measures it on the same atoms, the same ridge
grid and the same folds holding out whole molecules.

Three questions, in order.

1. **Does the three-distance descriptor reach SOAP?**  The field channels against the best SOAP
   of the gate, and against the two-point kernel it extends.
2. **Which channels are better, the fields or the elements?**  SOAP has one channel per element;
   this descriptor can carry either, on the same geometry and the same envelope, so the
   comparison isolates the channels from the basis.  Element channels in this basis against SOAP
   is the test of the basis alone.
3. **Do the fields add to the elements?**  Both together, against each alone.

Along the way the script settles the hyper-parameters in order -- the cutoff and the envelope,
the window width, the scaling of the channel blocks (per field, per pair of fields, or none;
see `triangles.block_scales`), the kernel exponent -- each at the best of the one before, and
every set of channels is run under every scaling, since what suits five fields of different
units need not suit element indicators.

Everything is reported with q2 and the root-mean-square error out of fold, and differences come
with a bootstrap interval over molecules (rule 4 of the plan).

Usage
-----
    python benchmarks/run_triangles.py                 # both sets
    python benchmarks/run_triangles.py --only shifts   # the 5096 carbons of the repository set
    python benchmarks/run_triangles.py --only aromatic # the 1225 ring carbons of plan item LP3
    python benchmarks/run_triangles.py --max-molecules 40   # a quick look
    python benchmarks/run_triangles.py --report

The SOAP row needs the `dscribe` package and `benchmarks/results/soap_gate/`; without them the
script runs and that row is left out.
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

from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.localenv import (choose_lam, count_features, count_kernel, environments,  # noqa: E402
                              local_features, local_kernel, paired_interval, types_of)
from cmfqsar.triangles import (SCALINGS, TriangleSpec, element_weights,  # noqa: E402
                               triangle_kernel_matrix, with_weights)
from run_soap_gate import (DATA, RESULTS as SOAP_RESULTS, aromatic_ring_samples,  # noqa: E402
                           shift_samples, soap_kernel, soap_rows)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "triangles")
#: the cutoffs of the radial grid searched first, in A, with the step and width of the two-point
#: distance grid and the plain kernel
R_MAX = (3.0, 4.0, 5.0)
#: envelope widths: a weak envelope leaves the radial grid to do the cutting, a strong one damps
RHOS = (0.1, 0.5)
STEP = 0.5
SIGMAS = (0.35, 0.7)            # tried at the best cutoff and envelope, against the default 0.5
POWERS = (2, 4)                 # kernel exponents tried at the best grid
#: the elements that get a channel of their own; every other element shares one
ELEMENTS = ("H", "C", "N", "O")
#: the two-point kernel's best envelope on each set, from plan items LP2 and LP3
TWO_POINT_RHO = dict(shifts=1.0, aromatic=0.5)


def _say(text):
    print(text, flush=True)


def _row(result, **extra):
    out = dict(q2=result.q2, error=result.error, mae=result.mae, lam=result.lam,
               note=result.note)
    out.update(extra)
    return out


def _interval(a, b, targets, seed):
    return [float(v) for v in paired_interval(a, b, targets, runs=2000, seed=seed)]


def soap_reference(which, mols, targets, folds, seed):
    """The best SOAP of the gate, refitted so its predictions are there for the intervals."""
    path = os.path.join(SOAP_RESULTS, f"{which}.json")
    if not os.path.exists(path):
        return None, None
    try:
        import dscribe  # noqa: F401
    except ImportError:
        return None, None
    with io.open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    rows = d["cutoffs"] + d["ladder"] + d["tuned"]
    best = max(rows, key=lambda r: r["q2"])
    t0 = time.time()
    X = soap_rows(mols, targets, best["r_cut"], l_max=best["l_max"], sigma=best["sigma"])
    r = choose_lam(soap_kernel(X, best["zeta"]), targets, folds=folds, seed=seed,
                   name=f"SOAP r_cut {best['r_cut']:g} l_max {best['l_max']} zeta "
                        f"{best['zeta']} sigma {best['sigma']:g}")
    r.note = f"{X.shape[1]} channels, {time.time() - t0:.0f} s"
    return r, best


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
    n_fields = len(fieldset.specs)

    types = types_of(fields, min_atoms=10)
    ref = choose_lam(count_kernel(count_features(fields, targets, types)), targets, folds=folds,
                     seed=seed, name="atom-centred counts")
    ref.note = f"{len(types)} atom types in 6 shells"
    _say("  " + ref.summary())
    rho2 = TWO_POINT_RHO[which]
    t0 = time.time()
    two = choose_lam(local_kernel(local_features(environments(fields, targets, rho=rho2))),
                     targets, folds=folds, seed=seed, name=f"two-point local, rho {rho2:g}")
    two.note = f"255 channels, {time.time() - t0:.0f} s"
    _say("  " + two.summary())
    soap, soap_params = soap_reference(which, mols, targets, folds, seed)
    if soap is not None:
        _say("  " + soap.summary())
    else:
        _say("  no SOAP row: dscribe or benchmarks/results/soap_gate/ is missing")

    cache = {}

    def envs_at(rho):
        if rho not in cache:
            cache[rho] = environments(fields, targets, rho=rho)
        return cache[rho]

    def run(envs, spec, power, rho, channels, nf, scaling="field"):
        t0 = time.time()
        K = triangle_kernel_matrix(envs, spec, n_fields=nf, power=power, scaling=scaling)
        r = choose_lam(K, targets, folds=folds, seed=seed,
                       name=f"three-distance, {channels}, r_max {spec.r_max:g} step "
                            f"{spec.step:g} sigma {spec.sigma:g} rho {rho:g} scaling {scaling} "
                            f"power {power}")
        r.note = f"{spec.n_channels(nf)} channels, {time.time() - t0:.0f} s"
        _say("  " + r.summary())
        return r

    # 1. the cutoff and the envelope, field channels, plain kernel, scaling per field
    _say("cutoff and envelope, field channels")
    grid = []
    for r_max in R_MAX:
        for rho in RHOS:
            spec = TriangleSpec(r_max=r_max, step=STEP, sigma=0.5)
            grid.append((r_max, rho, spec, 1, run(envs_at(rho), spec, 1, rho, "fields", n_fields)))
    best = max(grid, key=lambda g: g[4].q2)
    r_best, rho_best = best[0], best[1]

    # 2. the window width at that cutoff and envelope
    _say(f"window width at r_max {r_best:g}, rho {rho_best:g}")
    widths = [(0.5, best[4])]
    for sigma in SIGMAS:
        spec = TriangleSpec(r_max=r_best, step=STEP, sigma=sigma)
        widths.append((sigma, run(envs_at(rho_best), spec, 1, rho_best, "fields", n_fields)))
    sigma_best = max(widths, key=lambda w: w[1].q2)[0]
    spec_best = TriangleSpec(r_max=r_best, step=STEP, sigma=sigma_best)

    # 3. the scaling of the blocks, then the kernel power
    _say(f"block scaling at sigma {sigma_best:g}")
    scalings = [("field", max(widths, key=lambda w: w[1].q2)[1])]
    for sc in ("pair", "none"):
        scalings.append((sc, run(envs_at(rho_best), spec_best, 1, rho_best, "fields", n_fields,
                                 scaling=sc)))
    scaling_best = max(scalings, key=lambda w: w[1].q2)[0]
    _say(f"kernel power with scaling {scaling_best}")
    powers = [(1, max(scalings, key=lambda w: w[1].q2)[1])]
    for p in POWERS:
        powers.append((p, run(envs_at(rho_best), spec_best, p, rho_best, "fields", n_fields,
                              scaling=scaling_best)))
    power_best, fields_best = max(powers, key=lambda w: w[1].q2)

    # 4. the channels: the elements, and the fields with the elements, each under every scaling
    _say(f"channels at r_max {r_best:g}, sigma {sigma_best:g}, rho {rho_best:g}, power "
         f"{power_best}")
    base = envs_at(rho_best)
    n_el = len(ELEMENTS) + 1
    el_envs = [with_weights(e, element_weights(e, ELEMENTS, rho=rho_best)) for e in base]
    el_rows = [(sc, run(el_envs, spec_best, power_best, rho_best, "elements", n_el, scaling=sc))
               for sc in SCALINGS]
    del el_envs
    both_envs = [with_weights(e, np.hstack([e.weights, element_weights(e, ELEMENTS, rho=rho_best)]))
                 for e in base]
    both_rows = [(sc, run(both_envs, spec_best, power_best, rho_best, "fields and elements",
                          n_fields + n_el, scaling=sc)) for sc in SCALINGS]
    del both_envs
    el_scaling, elements = max(el_rows, key=lambda w: w[1].q2)
    both_scaling, both = max(both_rows, key=lambda w: w[1].q2)

    diffs = dict(
        fields_against_two_point=_interval(fields_best, two, targets, seed),
        fields_against_counts=_interval(fields_best, ref, targets, seed),
        elements_against_fields=_interval(elements, fields_best, targets, seed),
        both_against_fields=_interval(both, fields_best, targets, seed),
        both_against_elements=_interval(both, elements, targets, seed),
    )
    if soap is not None:
        diffs["fields_against_soap"] = _interval(fields_best, soap, targets, seed)
        diffs["elements_against_soap"] = _interval(elements, soap, targets, seed)
        diffs["both_against_soap"] = _interval(both, soap, targets, seed)
    _say("differences in q2, bootstrapped over molecules")
    for k, v in diffs.items():
        _say(f"  {k:28s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")

    out = dict(set=which, n_samples=len(targets), n_molecules=targets.n_measured_molecules,
               folds=folds, seed=seed, step=STEP, element_channels=list(ELEMENTS),
               reference=_row(ref), two_point=_row(two, rho=rho2),
               soap=(_row(soap, **soap_params) if soap is not None else None),
               grid=[_row(r, r_max=float(rm), rho=float(rho), sigma=0.5, power=1,
                          channels="fields") for rm, rho, _s, _p, r in grid],
               widths=[_row(r, r_max=float(r_best), rho=float(rho_best), sigma=float(s), power=1,
                            channels="fields") for s, r in widths[1:]],
               scalings=[_row(r, r_max=float(r_best), rho=float(rho_best),
                              sigma=float(sigma_best), power=1, scaling=sc, channels="fields")
                         for sc, r in scalings[1:]],
               powers=[_row(r, r_max=float(r_best), rho=float(rho_best), sigma=float(sigma_best),
                            power=int(p), scaling=scaling_best, channels="fields")
                       for p, r in powers[1:]],
               best=dict(r_max=float(r_best), rho=float(rho_best), sigma=float(sigma_best),
                         power=int(power_best), scaling=scaling_best),
               fields=_row(fields_best, scaling=scaling_best),
               elements=_row(elements, scaling=el_scaling), both=_row(both, scaling=both_scaling),
               element_rows=[_row(r, scaling=sc) for sc, r in el_rows],
               both_rows=[_row(r, scaling=sc) for sc, r in both_rows],
               differences=diffs)
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


NAMES = dict(fields_against_two_point="field channels against the two-point kernel",
             fields_against_counts="field channels against the counts",
             fields_against_soap="field channels against the best SOAP of the gate",
             elements_against_fields="element channels against field channels",
             elements_against_soap="element channels against SOAP (the basis alone)",
             both_against_fields="fields and elements against the fields",
             both_against_elements="fields and elements against the elements",
             both_against_soap="fields and elements against SOAP")


def report(results=RESULTS):
    lines = ["# The three-distance autocorrelation against the two-point kernel and SOAP", ""]
    lines += ["`cmfqsar/triangles.py` on the same carbons, ridge grid and folds as the SOAP gate. "
              "Errors are root-mean-square, out of fold, in ppm. Differences are in q2 with a 95% "
              "bootstrap interval over molecules. `rho` is the envelope width parameter, `r_max` "
              "the cutoff of the radial grid, `sigma` the window width and `power` the kernel "
              "exponent.", ""]
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
        r = d["two_point"]
        lines.append(f"| two-point local kernel, rho {r['rho']:g} | {r['q2']:+.3f} | "
                     f"{r['error']:.2f} | {r['lam']:g} |")
        if d.get("soap"):
            r = d["soap"]
            lines.append(f"| SOAP (elements), r_cut {r['r_cut']:g} A, l_max {r['l_max']}, zeta "
                         f"{r['zeta']}, sigma {r['sigma']:g} | {r['q2']:+.3f} | {r['error']:.2f} | "
                         f"{r['lam']:g} |")
        for r in d["grid"] + d["widths"] + d["scalings"] + d["powers"]:
            lines.append(f"| three-distance, fields, r_max {r['r_max']:g} A, rho {r['rho']:g}, "
                         f"sigma {r['sigma']:g}, scaling {r.get('scaling', 'field')}, power "
                         f"{r['power']} | {r['q2']:+.3f} | {r['error']:.2f} | {r['lam']:g} |")
        b = d["best"]
        tail = (f"r_max {b['r_max']:g} A, rho {b['rho']:g}, sigma {b['sigma']:g}, power "
                f"{b['power']}")
        for key, label in (("element_rows", "elements"), ("both_rows", "fields and elements")):
            for r in d[key]:
                lines.append(f"| three-distance, {label}, {tail}, scaling {r['scaling']} | "
                             f"{r['q2']:+.3f} | {r['error']:.2f} | {r['lam']:g} |")
        lines += ["", "Differences in q2:", ""]
        for k, v in d["differences"].items():
            lines.append(f"- {NAMES.get(k, k)}: {_fmt(v)}")
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
    for which in (("shifts", "aromatic") if args.only == "all" else (args.only,)):
        run_set(which, folds=args.folds, seed=args.seed, max_molecules=args.max_molecules)
    write_table()


if __name__ == "__main__":
    main()
