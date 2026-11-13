#!/usr/bin/env python
"""
A parameter set for a cell model, with its deviations (plan item MS9).

What this benchmark does
------------------------
A continuum cell model -- here PyBaMM's single-particle model with electrolyte, with the growth
of the passivating layer that makes the capacity fall from cycle to cycle -- needs a value for
every material parameter it holds.  Four of those parameters are quantities that benchmarks in
this repository predict, and each prediction comes with a standard deviation in the units of the
parameter (plan item MS1).  This benchmark converts the four predictions into the parameters the
cell model reads, draws each one from its deviation, runs the cell model once per draw, and says
which deviation moves the capacity most.

The four channels, and where each value comes from
--------------------------------------------------
1. **The diffusion coefficient in the positive particles.**  The forest on the invariant field,
   fitted on the 1,652 density-functional barriers of nebDFT2k (plan item BT7, ``run_nebdft.py``),
   predicts the barrier of every lithium hop of a layered nickel oxide with a standard deviation.
   The hops form a network, and its percolation barrier -- the lowest level at which lithium can
   cross the cell (plan item MS7) -- goes through an Arrhenius law into a diffusion coefficient.
2. **The open-circuit voltage of the positive electrode.**  The insertion ladder of plan item BT3
   reconstructs the average voltage of a layered nickel oxide from public formation energies, and
   the kernel model on the composition predicts it with a deviation.  What transfers to a cell
   model is the difference between the prediction and the published value, as a shift of the
   cell model's own curve.
3. **The conductivity of the electrolyte.**  The model of plan item BT5, fitted on the 62
   measured conductivities of CALiSol-23, predicts the conductivity of the cell's own
   electrolyte -- lithium hexafluorophosphate in ethylene carbonate and ethyl methyl carbonate,
   3 to 7 -- with the whole solvent mixture held out.  The prediction becomes a factor on the
   function the cell model holds.
4. **The tortuosity of the porous electrode.**  The kernel on a two-phase field (the
   microstructure pilot) predicts the effective conductivity of a porous electrode at the cell's
   own porosity, which is a Bruggeman exponent.  The structures are generated, not measured: what
   this channel contributes is the deviation of such a model, not the value for this cell.

A fifth parameter, the exchange-current density of the positive electrode, is **not** predicted
here.  It carries a stated spread instead, because the gate asks where it ranks, and a ranking
needs every parameter in it.

What is measured
----------------
* The parameter set itself, written out with one record per parameter
  (``results/cellmodel/parameters.json`` and ``parameters.csv``).
* For every parameter, the interval of the capacity-against-cycle curve over the draws, and
  whether it holds the curve the published value gives.
* Which parameter's deviation explains the largest share of the variance of the last cycle's
  capacity, drawn one at a time.
* The same ranking without the deviations: how far the capacity moves when each parameter is
  changed by a tenth of itself, which is the reading the cell-model literature states.
* How much more accurate the barrier would have to be for the capacity interval to be useful,
  which is the rule of plan item MS1 in the units of a cell.
* The electrolyte where it can matter (``--stage stress``): at the one-hour rate the cell hardly
  responds to its electrolyte, so the conductivity and the Bruggeman exponent are drawn again
  at faster discharges with the full Doyle-Fuller-Newman model, and the conductivity in colder
  cells, carried there by the activation energy the same model of item BT5 predicts. The
  published electrolyte functions have no temperature dependence, so the curve a prediction is
  judged against is that of the electrolyte as CALiSol-23 measured it.
* How accurate the three parameters of the electrolyte that are *not* predicted would have to
  be (``--stage tolerance``; plan items BT9, BT10 and BT11): the salt diffusion coefficient, the
  cation transference number and the thermodynamic factor. Each is changed over a fixed set of
  points around its measured function, the full model is run once per point, and the range
  inside which the capacity stays within 1 and within 5 per cent is read from that curve. The
  cheapest reference of each item is run as well: the Nernst-Einstein value from the
  conductivity, and the two constants of the published set.

Usage
-----
    python benchmarks/run_nebdft.py            # once: the rows of the hops
    python benchmarks/run_cellmodel.py         # this benchmark, then the table (50 minutes)
    python benchmarks/run_cellmodel.py --stage stress    # one stage; each writes its own file
    python benchmarks/run_cellmodel.py --stage tolerance # the three unpredicted parameters
    python benchmarks/run_cellmodel.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

RESULTS = os.path.join(HERE, "results", "cellmodel")
#: the published parameter set: a graphite / nickel-manganese-cobalt oxide cell
BASE = "Chen2020"
#: which reduction of the cell model to run, and how many cycles of one ampere-hour rate
KIND = "SPMe"
CYCLES = 10
#: cycles for the three runs that look at the fade alone, where ten cycles show almost none
FADE_CYCLES = 100
DRAWS = 200
SEED = 0
FOLDS = 5
INNER = 4
LAMS = tuple(10.0 ** e for e in (-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0))
#: the layered nickel oxide the diffusion coefficient is predicted for: the entry of the
#: Materials Project lithium-oxide set that lies on the convex hull
POSITIVE_STRUCTURE = "mp-2348641"
#: the attempt frequency of one jump, in jumps per second; not predicted by this program
ATTEMPT_FREQUENCY = 1e13
#: no drawn diffusion coefficient above this, in m^2/s: an ion in a solid does not move faster
#: than one in a liquid, and a cell model spends minutes on a value that fast
FASTEST_DIFFUSION = 1e-10
#: the solvent mixture of the cell's electrolyte in the measured series, and the molality closest
#: to the one mole per litre the cell model starts from
ELECTROLYTE_COMPOSITION = "EC0.30+EMC0.70"
ELECTROLYTE_ID = "LiPF6_EC30_EMC70_m1p110"
#: the microstructure channel: how many generated structures, on what grid, and the porosity of
#: the cell's positive electrode
MICRO_N = 200
MICRO_GRID = 16
MICRO_TEST = 24
#: the conductivity of the solid phase relative to the electrolyte in the pores: a cathode
#: particle carries almost no ionic current, so the ratio is small
MICRO_CONTRAST = 1e-3
MICRO_SMOOTH = (0.0, 1.0, 2.0)
#: the published spread of an exchange-current density for one material, as a factor
EXCHANGE_SPREAD = 10.0
#: the factors the barrier's deviation is made smaller by, for the tolerance rule
TOL_SCALES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02)
TOL_DRAWS = 60
TOL_CYCLES = 3
#: an interval of the last cycle's capacity narrower than this share of the published capacity
#: counts as useful
TOL_SHARE = 0.05
#: the electrolyte under stress: the full model, faster discharges at room temperature, and
#: colder cells, where the conductivity follows the predicted activation energy
STRESS_MODEL = "DFN"
STRESS_RATES = (1.0, 1.5, 2.0, 2.5)
STRESS_COLD = (263.15, 253.15)
STRESS_COLD_RATES = (1.0, 1.5)
STRESS_DRAWS = 100
#: the coldest temperature, in kelvin, that Landesfeind and Gasteiger measured; a colder
#: cell reads their fitted functions outside the range they were fitted on
LANDESFEIND_COLDEST = 263.15
ROOM = 298.15
#: the cell model's names of the four transport parameters of the electrolyte
CONDUCTIVITY = "Electrolyte conductivity [S.m-1]"
DIFFUSIVITY = "Electrolyte diffusivity [m2.s-1]"
TRANSFERENCE = "Cation transference number"
THERMODYNAMIC = "Thermodynamic factor"
#: the three of them that this program does not predict (plan items BT9, BT10 and BT11), and
#: the points of the response curve of each. A "scale" multiplies the measured function by the
#: value and a "shift" adds the value to it, so 1 and 0 are the measured function unchanged.
TOLERANCE_PARAMETERS = (
    {"item": "BT9", "name": DIFFUSIVITY, "label": "salt diffusion coefficient",
     "kind": "scale",
     "values": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0,
                3.0, 5.0, 10.0, 20.0)},
    {"item": "BT10", "name": TRANSFERENCE, "label": "cation transference number",
     "kind": "shift",
     "values": (-0.4, -0.3, -0.2, -0.15, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2,
                0.3, 0.4)},
    {"item": "BT11", "name": THERMODYNAMIC, "label": "thermodynamic factor", "kind": "scale",
     "values": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0)},
)
#: the shares of the capacity a tolerance is stated for: the capacity may move by this much
TOLERANCE_SHARES = (0.01, 0.05)
#: the condition at which the reading of the response curve is checked against drawn values
TOLERANCE_CHECK = (ROOM, 2.5)
TOLERANCE_CHECK_DRAWS = 100
#: the concentrations, in mol/m3, between which Landesfeind and Gasteiger measured
MEASURED_CONCENTRATIONS = (100.0, 3000.0)
#: below this concentration, in mol/m3, the electrolyte counts as exhausted somewhere in the cell
EXHAUSTED = 1.0
#: the stages, in order: fit the models and convert, run the published values, draw one
#: parameter at a time, the electrolyte under stress, the tolerance of the cell to the three
#: parameters that are not predicted, then put the result together
STAGES = ("channels", "reference", "sample", "stress", "tolerance", "finish")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Folds and a fit that carries a deviation
# --------------------------------------------------------------------------- #
def group_parts(groups: Sequence[object], k: int, seed: int = SEED) -> List[np.ndarray]:
    """`k` held-out index sets in which no group is split, the largest groups dealt first."""
    labels = [str(v) for v in groups]
    members: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels):
        members.setdefault(lab, []).append(i)
    rng = np.random.default_rng(seed)
    order = sorted(members)
    order = [order[i] for i in rng.permutation(len(order))]
    order.sort(key=lambda g: -len(members[g]))
    buckets: List[List[int]] = [[] for _ in range(max(int(k), 1))]
    for g in order:
        buckets[int(np.argmin([len(b) for b in buckets]))].extend(members[g])
    return [np.asarray(sorted(b), int) for b in buckets if b]


def unit_kernel(K: np.ndarray) -> np.ndarray:
    """A kernel divided by the mean of its diagonal, so that one ridge grid fits every kernel."""
    d = float(np.mean(np.diag(np.asarray(K, float))))
    return np.asarray(K, float) / (d if d > 0 else 1.0)


def descriptor_kernels(Z: np.ndarray, gammas: Sequence[float] = (0.25, 1.0, 4.0)
                       ) -> Dict[str, np.ndarray]:
    """The kernels of a table of descriptors: the linear one and Gaussian ones of several widths.

    The widths are given as multiples of one over the median squared distance between the rows,
    so that the same numbers work whatever the descriptors are. The fold chooses among them, so
    a channel is not held to a linear model when the quantity is not linear in its descriptors.
    """
    Z = np.asarray(Z, float)
    Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)
    out = {"linear": unit_kernel(Z @ Z.T)}
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    median = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    for g in gammas:
        out[f"gaussian, width {g:g}"] = np.exp(-float(g) * d2 / median)
    return out


def nested_surrogate(kernels: Dict[str, np.ndarray], y: Sequence[float],
                     groups: Sequence[object], lams: Sequence[float] = LAMS,
                     folds: int = FOLDS, inner: int = INNER, seed: int = SEED
                     ) -> Dict[str, object]:
    """Out-of-fold prediction and standard deviation, with the kernel and the ridge chosen inside
    every fold (plan item MS1 read on held-out groups).

    `kernels` maps a name to a whole kernel matrix; the fold picks one of them together with the
    ridge, on inner folds of its own training part that hold out whole groups as well. Returns
    the predictions, the deviations, the calibration and what every fold chose.
    """
    from cmfqsar.surrogate import Calibration, Surrogate

    y = np.asarray(y, float).ravel()
    n = len(y)
    names = list(kernels)
    diag = {m: np.diag(kernels[m]) for m in names}
    mean, sd = np.full(n, np.nan), np.full(n, np.nan)
    chosen = []
    for te in group_parts(groups, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        inner_parts = [tr[p] for p in group_parts([groups[i] for i in tr], inner, seed + 1)]
        best = None
        for m in names:
            K = kernels[m]
            for lam in lams:
                p = np.full(n, np.nan)
                for ip in inner_parts:
                    itr = np.setdiff1d(tr, ip)
                    if len(itr) < 3:
                        continue
                    s = Surrogate.fit(K[np.ix_(itr, itr)], y[itr], lam)
                    p[ip], _ = s.predict(K[np.ix_(ip, itr)], diag[m][ip])
                ok = np.isfinite(p[tr])
                if ok.sum() < 3:
                    continue
                err = float(np.sqrt(np.mean((y[tr][ok] - p[tr][ok]) ** 2)))
                if best is None or err < best[0]:
                    best = (err, m, lam)
        if best is None:
            raise RuntimeError("no fold of this kernel could be fitted")
        _, m, lam = best
        s = Surrogate.fit(kernels[m][np.ix_(tr, tr)], y[tr], lam)
        mean[te], sd[te] = s.predict(kernels[m][np.ix_(te, tr)], diag[m][te])
        chosen.append({"kernel": m, "lam": float(lam), "n_test": int(len(te))})
    cal = Calibration.of(y, mean, sd)
    return {"predicted": mean, "sd": sd, "calibration": cal,
            "mae": float(np.mean(np.abs(y - mean))),
            "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
            "z_rms": float(cal.z_rms), "mean_sd": float(np.mean(sd)), "chosen": chosen,
            "mae_of_the_mean": float(np.mean(np.abs(y - np.mean(y))))}


# --------------------------------------------------------------------------- #
# Channel 1: the diffusion coefficient in the positive particles
# --------------------------------------------------------------------------- #
def layered_oxide(structure: str = POSITIVE_STRUCTURE):
    """The layered nickel oxide as an ASE structure, from the set run_hops.py cached."""
    from ase import Atoms
    path = os.path.join(HERE, "cache", "hops", "mp_li_oxides.json")
    if not os.path.isfile(path):
        raise SystemExit(
            f"{path} is missing: run 'python benchmarks/run_hops.py --public --fetch' once, "
            "which caches the lithium-oxide structures this channel takes its cathode from")
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)["structures"]
    mine = [r for r in records if r["id"] == str(structure)]
    if not mine:
        raise SystemExit(f"the cached set holds no structure called {structure!r}")
    r = mine[0]
    return Atoms(symbols=r["symbols"], positions=np.asarray(r["positions"], float),
                 cell=np.asarray(r["cell"], float), pbc=True), r


def percolation_by_dimension(hc, k: int, values: Dict[int, float]) -> Dict[str, float]:
    """The percolation barrier of one structure in one, two and three directions."""
    from cmfqsar.hops import structure_percolation
    p = structure_percolation(hc, k, values, relative=False)
    return {d: float(p[f"E_{d}D"]) for d in (1, 2, 3)}


def drawn_barriers(hc, reps: Sequence[int], pred: np.ndarray, sd: np.ndarray, shared: float,
                   dimension: int, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """The percolation barrier in `dimension` directions, once per draw of the hop barriers.

    The draw is the one of plan item MS8: every structure gets one shift it shares and every
    distinct hop one of its own, in the proportions the errors of the model have, and a barrier
    is not drawn below zero. Draws in which the network does not percolate in that many
    directions are left out, and how many were left out is reported next to the interval.
    """
    import run_conductors as C
    rng = np.random.default_rng(int(seed))
    out = []
    for _ in range(int(draws)):
        z = (np.sqrt(shared) * rng.normal() +
             np.sqrt(max(1.0 - shared, 0.0)) * rng.normal(size=len(reps)))
        values = C.spread_values(hc, reps, np.maximum(pred + sd * z, 0.0))
        e = percolation_by_dimension(hc, 0, values)[int(dimension)]
        if np.isfinite(e):
            out.append(float(e))
    if len(out) < 2:
        raise SystemExit(f"the hop network percolates in {dimension} directions in fewer than "
                         "two draws, so no diffusion coefficient can be carried through")
    return np.asarray(out, float)


def barrier_channel(draws: int = DRAWS, folds: int = 10) -> Tuple[object, dict]:
    """The diffusion coefficient of the positive particles, from predicted hop barriers."""
    import run_conductors as C
    import run_hops as H
    import run_nebdft as N
    from sklearn.ensemble import RandomForestRegressor

    from cmfqsar.hops import HopSet
    from cmfqsar.interfaces import diffusivity_from_barrier
    from cmfqsar.localenv import local_features
    from cmfqsar.periodic import inorganic_fieldset

    if not os.path.isfile(N.PREPARED):
        raise SystemExit(
            f"{N.PREPARED} is missing: run 'python benchmarks/prepare_nebdft.py --fetch' and "
            "'python benchmarks/run_nebdft.py' once, which leave the barriers and the rows "
            "this channel is fitted on")
    hs, doc, _, _ = N.load(N.PREPARED, 0)
    t = hs.targets()
    y = np.asarray(t.y, float)
    groups = np.asarray(t.molecule, int)
    _, rows = N.descriptions(hs, 8, N.CACHE, "")
    X = rows["invariant"]
    errors = C.error_model(X, y, groups, folds)
    log(f"  the forest on {len(y)} barriers, whole structures held out: mean absolute error "
        f"{errors['mae']:.3f} eV, z rms {errors['factor']:.2f}, shared within a structure "
        f"{errors['shared']:.2f}")

    atoms, record = layered_oxide()
    fs_train = H.fieldset_for(hs, "file")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hc = HopSet.build([atoms], names=[record["formula"]], mobile="Li", d_min=2.0, d_max=5.0)
        cif = [s.name[4:] for s in fs_train.specs if s.is_cif and s.name != "cif:other"]
        fs = inorganic_fieldset(cif, physchem=H.PHYSCHEM, charge_model="oxidation",
                                other=any(s.name == "cif:other" for s in fs_train.specs))
        if [s.name for s in fs.specs] != [s.name for s in fs_train.specs]:
            raise SystemExit("the fields of the cathode are not those of the training set")
        crystals = hc.crystals(fs)
        reps = list(hc.annotate(crystals))
        envs = hc.environments(crystals, rows=reps, rho=N.RHO, r_max=N.R_MAX)
        Xc = local_features(envs)
    forest = RandomForestRegressor(n_estimators=N.FOREST_TREES, n_jobs=-1,
                                   random_state=SEED).fit(X, y)
    trees = np.array([e.predict(Xc) for e in forest.estimators_])
    pred = trees.mean(axis=0)
    sd = np.maximum(trees.std(axis=0), 1e-6) * errors["factor"]
    point = C.spread_values(hc, reps, pred)
    by_dim = percolation_by_dimension(hc, 0, point)
    dimension = max([d for d in (1, 2, 3) if np.isfinite(by_dim[d])] or [1])
    lengths = np.array([hc.hops[r].length for r in reps], float)
    carrying = lengths[pred <= by_dim[dimension] + 1e-9]
    hop_length = float(np.median(carrying)) if len(carrying) else float(np.median(lengths))
    log(f"  {record['formula']}: {len(hc.hops)} lithium hops, {len(reps)} distinct, predicted "
        f"barrier {pred.mean():.2f} +/- {sd.mean():.2f} eV; the network percolates in "
        f"{dimension} of three directions at {by_dim[dimension]:.3f} eV, over hops of "
        f"{hop_length:.2f} A")

    barriers = drawn_barriers(hc, reps, pred, sd, errors["shared"], dimension, draws)
    log(f"  {len(barriers)} of {draws} draws percolate in {dimension} directions: barrier "
        f"{np.median(barriers):.3f} eV, 95 % interval {np.percentile(barriers, 2.5):.3f} to "
        f"{np.percentile(barriers, 97.5):.3f} eV")
    published = 4e-15
    parameter = diffusivity_from_barrier(
        barriers, hop_length, "Positive particle diffusivity [m2.s-1]",
        attempt_frequency=ATTEMPT_FREQUENCY, dimension=dimension, published=published,
        high=FASTEST_DIFFUSION,
        source=f"the forest on the invariant field, fitted on the {len(y)} density-functional "
               f"barriers of nebDFT2k (plan item BT7), read as the percolation barrier of the "
               f"hop network of {record['formula']} ({record['id']}) in {dimension} directions",
        calibration=float(errors["factor"]), in_domain=True,
        note=f"the attempt frequency is {ATTEMPT_FREQUENCY:.0e} per second and the hop length "
             f"{hop_length:.2f} angstrom, neither of them predicted; the cathode of the cell is "
             "a nickel-manganese-cobalt oxide and the structure used is the ordered layered "
             "nickel oxide, which is the closest the public set holds")
    details = {"training": {"hops": int(len(y)), "structures": int(len(np.unique(groups))),
                            **{k: v for k, v in errors.items() if k != "predicted"}},
               "structure": {"id": record["id"], "formula": record["formula"],
                             "hops": len(hc.hops), "distinct": len(reps),
                             "mean_barrier": float(pred.mean()), "mean_sd": float(sd.mean()),
                             "hop_length": hop_length, "dimension": dimension,
                             "percolation": {str(k): (v if np.isfinite(v) else None)
                                             for k, v in by_dim.items()}},
               "barrier_draws": {"n": int(len(barriers)),
                                 "median": float(np.median(barriers)),
                                 "interval": [float(np.percentile(barriers, 2.5)),
                                              float(np.percentile(barriers, 97.5))]},
               "barriers": [float(b) for b in barriers],
               "shared": float(errors["shared"])}
    return parameter, details


# --------------------------------------------------------------------------- #
# Channel 2: the open-circuit voltage of the positive electrode
# --------------------------------------------------------------------------- #
def voltage_channel() -> Tuple[object, dict]:
    """A shift of the published open-circuit voltage, from the insertion ladder of item BT3."""
    import run_battery as B

    from cmfqsar import battery
    from cmfqsar.interfaces import ocp_shift

    pairs = B.load_pairs(verbose=False)
    structures, y, names = battery.pair_table(pairs, frame="discharged")
    frameworks = [p.framework for p in pairs]
    elements = sorted(B.elements_in(structures))
    X = np.nan_to_num(B.composition_features(structures, elements), nan=0.0)
    fit = nested_surrogate(descriptor_kernels(X), y, frameworks)
    log(f"  the voltage of {len(y)} insertion pairs over {len(set(frameworks))} frameworks, "
        f"frameworks held out: mean absolute error {fit['mae']:.3f} V against "
        f"{fit['mae_of_the_mean']:.3f} V for the mean, z rms {fit['z_rms']:.2f}")

    published = {f: v for f, v, _ in B.PUBLISHED}
    rows = B.cathode_rows(pairs)
    layered = [r for r in rows if r["framework"] == "NiO2"]
    if not layered:
        raise SystemExit("the insertion set holds no layered nickel oxide to read the voltage of")
    ladder = layered[0]
    # the same average the ladder takes: over the steps of the first insertion, each weighted by
    # how much of the working ion it takes in
    mine = [i for i, p in enumerate(pairs)
            if p.framework == "NiO2" and p.x_discharged <= B.CHECK_WINDOW + 1e-9]
    if not mine:
        raise SystemExit("no insertion pair of the layered nickel oxide was fitted")
    weights = np.array([pairs[i].x_discharged - pairs[i].x_charged for i in mine], float)
    weights = weights / weights.sum()
    predicted = float(np.sum(weights * fit["predicted"][mine]))
    # the pairs of one framework are held out together, so their errors move together; the
    # weighted mean of their deviations is the conservative reading and the one taken here
    sd = float(np.sum(weights * fit["sd"][mine]))
    reference = float(published["NiO2"])
    log(f"  the layered nickel oxide: the ladder reconstructs {ladder['voltage']:.2f} V over "
        f"{ladder['steps']} steps, the model predicts {predicted:.2f} +/- {sd:.2f} V over the "
        f"same {len(mine)} steps, the published value is {reference:.2f} V")
    parameter = ocp_shift(
        predicted, reference, "Positive electrode OCP [V]", sd=sd,
        source=f"the kernel model on the composition of {len(y)} insertion pairs reconstructed "
               "from public formation energies (plan item BT3), with the frameworks held out",
        calibration=float(fit["z_rms"]), in_domain=True,
        note="the shift is the model's own error on the layered nickel oxide, which is the "
             "closest material to the cell's cathode that the set holds; the ladder "
             f"reconstructs {ladder['voltage']:.2f} V and the model on the composition "
             f"{predicted:.2f} V, against {reference:.2f} V published")
    details = {"n": int(len(y)), "frameworks": int(len(set(frameworks))),
               "mae": fit["mae"], "rmse": fit["rmse"], "z_rms": fit["z_rms"],
               "mae_of_the_mean": fit["mae_of_the_mean"],
               "coverage_2sd": float(fit["calibration"].coverage_2sd),
               "error_units": "V", "chosen": fit["chosen"],
               "ladder": {k: (float(v) if isinstance(v, (int, float)) else v)
                          for k, v in ladder.items()
                          if k in ("framework", "label", "published", "voltage", "steps")},
               "predicted": predicted, "sd": sd, "published": reference,
               "steps_averaged": int(len(mine)),
               "measured_of_the_steps": float(np.sum(weights * y[mine])),
               "pairs": [str(names[i]) for i in mine]}
    return parameter, details


# --------------------------------------------------------------------------- #
# Channel 3: the conductivity of the electrolyte
# --------------------------------------------------------------------------- #
def electrolyte_channel() -> Tuple[object, dict]:
    """A factor on the published conductivity of the electrolyte, from the model of item BT5."""
    import run_shells as S

    from cmfqsar.cellmodel import published_parameters
    from cmfqsar.interfaces import conductivity_scale_from_log10

    rows = S.load_series()
    y = np.array([float(r["log10_k_298"]) for r in rows], float)
    compositions = [str(r["composition"]) for r in rows]
    fit = nested_surrogate(descriptor_kernels(S.descriptors(rows, "composition")), y,
                           compositions)
    log(f"  the conductivity of {len(y)} electrolytes over {len(set(compositions))} solvent "
        f"mixtures, mixtures held out: mean absolute error {fit['mae']:.3f} on the logarithmic "
        f"scale against {fit['mae_of_the_mean']:.3f} for the mean, z rms {fit['z_rms']:.2f}")

    mine = [i for i, r in enumerate(rows) if str(r["id"]) == ELECTROLYTE_ID]
    if not mine:
        raise SystemExit(f"the measured series holds no electrolyte called {ELECTROLYTE_ID!r}")
    i = mine[0]
    predicted, sd = float(fit["predicted"][i]), float(fit["sd"][i])
    p = published_parameters(BASE)
    reference = _conductivity_of(p, 1000.0, 298.15)
    log(f"  {ELECTROLYTE_ID}: measured {y[i]:+.3f}, predicted {predicted:+.3f} +/- {sd:.3f}, "
        f"the cell model holds {reference:+.3f} (all base-ten logarithms of mS/cm)")
    parameter = conductivity_scale_from_log10(
        predicted, reference, "Electrolyte conductivity [S.m-1]", sd=sd,
        source=f"the kernel model on the composition of {len(y)} measured conductivities of "
               "CALiSol-23 (plan item BT5), with the solvent mixture of this electrolyte held out",
        calibration=float(fit["z_rms"]), in_domain=True,
        note="the cell model holds the conductivity as a function of concentration and "
             "temperature, so a prediction of one condition can only be a factor on it; the "
             f"condition is {ELECTROLYTE_COMPOSITION} at "
             f"{float(rows[i]['molality']):.2f} mol/kg against the one mole per litre the cell "
             "model starts from")
    details = {"n": int(len(y)), "mixtures": int(len(set(compositions))), "mae": fit["mae"],
               "rmse": fit["rmse"], "z_rms": fit["z_rms"],
               "mae_of_the_mean": fit["mae_of_the_mean"],
               "coverage_2sd": float(fit["calibration"].coverage_2sd),
               "error_units": "on the base-ten logarithmic scale",
               "chosen": fit["chosen"], "id": ELECTROLYTE_ID,
               "composition": ELECTROLYTE_COMPOSITION,
               "molality": float(rows[i]["molality"]), "measured": float(y[i]),
               "predicted": predicted, "sd": sd, "cell_model": reference}
    return parameter, details


def _conductivity_of(parameters, concentration: float, temperature: float) -> float:
    """The base-ten logarithm of the cell model's own electrolyte conductivity, in mS/cm."""
    import pybamm
    f = parameters["Electrolyte conductivity [S.m-1]"]
    v = f(pybamm.Scalar(float(concentration)), pybamm.Scalar(float(temperature)))
    value = float(v.evaluate()) if hasattr(v, "evaluate") else float(v)
    return float(np.log10(10.0 * value))


# --------------------------------------------------------------------------- #
# Channel 4: the tortuosity of the porous electrode
# --------------------------------------------------------------------------- #
def microstructure_channel(n: int = MICRO_N, grid: int = MICRO_GRID, test: int = MICRO_TEST
                           ) -> Tuple[object, dict]:
    """A Bruggeman exponent, from a kernel on the two-phase field of a porous electrode."""
    import run_microstructure as M

    from cmfqsar.cellmodel import published_parameters
    from cmfqsar.interfaces import bruggeman_from_conductivity_ratio

    porosity = float(published_parameters(BASE)["Positive electrode porosity"])
    rng = np.random.default_rng(SEED)
    phases, ratios, kinds = [], [], []
    for i in range(int(n) + int(test)):
        # the training structures spread over a window of porosities, the test ones sit at the
        # porosity of the cell's own electrode
        eps = (porosity if i >= int(n)
               else float(rng.uniform(porosity - 0.085, porosity + 0.085)))
        rot = M._rotation(3, rng)
        ph = M.generate(3, int(grid), 1.0 - eps, float(rng.uniform(2.0, 5.0)),
                        float(rng.uniform(1.0, 2.0)), rot, rng)
        tensor = M.effective_conductivity(ph, MICRO_CONTRAST)
        phases.append(ph)
        ratios.append(float(np.mean(np.diag(tensor))))
        kinds.append("test" if i >= int(n) else "train")
    phase = np.stack(phases)
    y = np.log(np.asarray(ratios, float))          # the logarithm, since the exponent needs it
    fractions = 1.0 - phase.reshape(len(phase), -1).mean(axis=1)
    kernels = {}
    for s in MICRO_SMOOTH:
        R = M.radial_profile(M.autocorrelations(M.smoothed_fields(phase, s)))
        R = R.reshape(len(phase), -1)
        kernels[f"radial autocorrelation, smoothing {s:g}"] = unit_kernel(R @ R.T)
    groups = [f"{k}{i // 8}" for i, k in enumerate(kinds)]
    fit = nested_surrogate(kernels, y, groups)
    is_test = np.array([k == "test" for k in kinds])
    predicted = float(np.median(np.exp(fit["predicted"][is_test])))
    sd_log = float(np.mean(fit["sd"][is_test]))
    measured = float(np.median(np.exp(y[is_test])))
    log(f"  {len(y)} generated two-phase structures on a {grid}^3 grid, porosity "
        f"{fractions.min():.2f} to {fractions.max():.2f}: the kernel predicts the logarithm of "
        f"the effective conductivity to {fit['mae']:.3f} against {fit['mae_of_the_mean']:.3f} "
        f"for the mean, z rms {fit['z_rms']:.2f}")
    log(f"  at the porosity of the cell's positive electrode ({porosity:.3f}): predicted ratio "
        f"{predicted:.3f}, true ratio {measured:.3f}")
    parameter = bruggeman_from_conductivity_ratio(
        predicted, porosity, "Positive electrode Bruggeman coefficient (electrolyte)",
        sd=predicted * sd_log, published=1.5,
        source=f"the kernel on the radial autocorrelation of a two-phase field, fitted on "
               f"{int(n)} generated structures with their effective conductivity solved on the "
               "grid (the microstructure pilot)",
        calibration=float(fit["z_rms"]), in_domain=False,
        note="the structures are generated and not measured, so what this channel gives is the "
             "deviation such a model has, not the exponent of this cell's electrode; the "
             "training structures cover the porosity of the cell, as a model fitted for one "
             "cell would, and the structures read off are held out; the solid phase carries "
             f"{MICRO_CONTRAST:g} of the conductivity of the electrolyte in the pores")
    details = {"n_train": int(n), "n_test": int(test), "grid": int(grid),
               "contrast": MICRO_CONTRAST, "porosity": porosity,
               "porosity_window": [float(fractions.min()), float(fractions.max())],
               "mae_log": fit["mae"], "rmse_log": fit["rmse"], "z_rms": fit["z_rms"],
               "mae_of_the_mean": fit["mae_of_the_mean"],
               "coverage_2sd": float(fit["calibration"].coverage_2sd),
               "error_units": "on the natural logarithm of the conductivity ratio",
               "chosen": fit["chosen"], "predicted_ratio": predicted,
               "true_ratio": measured, "sd_of_the_logarithm": sd_log,
               "true_exponent": float(np.log(measured) / np.log(porosity))}
    return parameter, details


# --------------------------------------------------------------------------- #
# Channel 5: the exchange-current density, which this program does not predict
# --------------------------------------------------------------------------- #
def exchange_channel() -> Tuple[object, dict]:
    """A factor on the published exchange-current density, with a stated spread."""
    from cmfqsar.interfaces import CellParameter
    sd = float(np.log(EXCHANGE_SPREAD) / 1.96)
    parameter = CellParameter(
        name="Positive electrode exchange-current density [A.m-2]", value=1.0, sd=sd,
        kind="scale", distribution="lognormal", published=1.0, units="factor",
        source="not predicted by this program: a stated spread",
        calibration=None, predicted=False, in_domain=True, low=0.0,
        note=f"a 95 % interval of a factor of {EXCHANGE_SPREAD:g}, which is the order the "
             "published values for one electrode material differ by; it is here because the "
             "ranking of the parameters needs every parameter in it")
    return parameter, {"spread": EXCHANGE_SPREAD, "sd_of_the_logarithm": sd}


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
CHANNELS = (("diffusion coefficient", barrier_channel),
            ("open-circuit voltage", voltage_channel),
            ("electrolyte conductivity", electrolyte_channel),
            ("electrode tortuosity", microstructure_channel),
            ("exchange current", exchange_channel))


def build_set() -> Tuple[object, Dict[str, dict]]:
    """The parameter set and what every channel measured on the way to it."""
    from cmfqsar.interfaces import CellParameterSet
    pset = CellParameterSet(
        base=BASE, temperature=298.15,
        note="the parameters a cell model reads, as the benchmarks of this repository predict "
             "them, each with a standard deviation in the unit of the parameter")
    details = {}
    for label, channel in CHANNELS:
        log(f"{label}:")
        t0 = time.time()
        parameter, info = channel()
        info["seconds"] = round(time.time() - t0, 1)
        info["label"] = label
        details[parameter.name] = info
        pset.add(parameter)
        log(f"  -> {parameter.describe()}")
    return pset, details


def fade_edges(parameter, cycles: int = FADE_CYCLES) -> dict:
    """What the parameter's two-sided interval does to the fade over many cycles.

    Ten cycles of the published set lose almost no capacity, so the interval over ten cycles
    tests the level of the curve and not its slope. Three runs per parameter -- at the middle of
    its interval and at both edges -- say whether the deviation changes the slope at all.
    """
    from cmfqsar import cellmodel as CM
    p = CM.published_parameters(BASE)
    lo, hi = parameter.interval()
    mid = (parameter.value if parameter.distribution != "sample"
           else float(np.median(parameter.sample)))
    out = {}
    for label, v in (("low", lo), ("middle", mid), ("high", hi)):
        run = CM.run_cell({parameter.name: parameter.apply(p[parameter.name], v)},
                          cycles=cycles, base=BASE, kind=KIND, parameters=p)
        out[label] = {"value": float(v), "ok": bool(run.ok),
                      "first": float(run.capacity[0]) if len(run.capacity) else float("nan"),
                      "final": run.final, "fade": run.fade, "message": run.message}
    return out


def slug(name: str) -> str:
    """A file name for one parameter: its own name with everything unusual removed."""
    keep = [c if (c.isalnum() or c in " -") else " " for c in name.lower()]
    return "_".join("".join(keep).split())


def stage_channels(results: str) -> object:
    """The first stage: fit the models, convert, and write the parameter set."""
    from cmfqsar.interfaces import CellParameterSet
    os.makedirs(results, exist_ok=True)
    pset, details = build_set()
    pset.write(os.path.join(results, "parameters.json"))
    pset.write_table(os.path.join(results, "parameters.csv"))
    with open(os.path.join(results, "channels.json"), "w", encoding="utf-8") as fh:
        json.dump(details, fh, indent=1)
    assert CellParameterSet.read(os.path.join(results, "parameters.json")).names == pset.names
    return pset


def read_set(results: str):
    """The parameter set and the channel records a previous stage wrote."""
    from cmfqsar.interfaces import CellParameterSet
    path = os.path.join(results, "parameters.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run this benchmark with --stage channels first")
    with open(os.path.join(results, "channels.json"), encoding="utf-8") as fh:
        details = json.load(fh)
    return CellParameterSet.read(path), details


def stage_reference(results: str, cycles: int = CYCLES) -> dict:
    """The curve the published values give, which every interval is compared with."""
    from cmfqsar import cellmodel as CM
    log(f"the cell model ({KIND}, {cycles} cycles) at the published values:")
    run = CM.run_cell(cycles=cycles, base=BASE, kind=KIND)
    if not run.ok:
        raise SystemExit(f"the cell model does not run at its own published values: "
                         f"{run.message}")
    log(f"  capacity {run.capacity[0]:.4f} A.h at the first cycle and {run.final:.4f} at the "
        f"last, {100 * (1 - run.fade):.3f} % lost")
    out = {"first": float(run.capacity[0]), "final": run.final, "fade": run.fade,
           "curve": [float(v) for v in run.capacity], "cycles": int(cycles)}
    with open(os.path.join(results, "reference.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def read_reference(results: str) -> dict:
    path = os.path.join(results, "reference.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run this benchmark with --stage reference first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def stage_sample(results: str, name: str, draws: int = DRAWS, cycles: int = CYCLES) -> dict:
    """One parameter: its draws, the interval they give, and the two sensitivity readings.

    One stage per parameter, each written to its own file, so that a cell model that fails on one
    drawn value costs that parameter and not the whole run.
    """
    from cmfqsar import cellmodel as CM
    pset, _ = read_set(results)
    p = pset[name]
    reference = read_reference(results)
    log(f"{p.name}: {draws} draws")
    t1 = time.time()
    done = [0]

    def progress(i: int, n: int) -> None:
        if i % 25 == 0 or i == n:
            log(f"    {i} of {n} draws, {time.time() - t1:.0f} s")
        done[0] = i

    these = CM.sample(p, draws=draws, cycles=cycles, base=BASE, kind=KIND, seed=SEED,
                      progress=progress)
    b = CM.band(these)
    held = CM.contains(b, reference["curve"])
    sens = CM.local_sensitivity(p, cycles=cycles, base=BASE, kind=KIND)
    out = {"name": p.name, "draws": int(draws), "cycles": int(cycles),
           "interval_of_the_last_cycle": [float(b["low"][-1]), float(b["high"][-1])]
           if b["n"] > 1 else [float("nan")] * 2,
           "median_of_the_last_cycle": float(b["median"][-1]) if b["n"] > 1 else float("nan"),
           "runs_that_completed": int(b["n"]), "refused": float(b["refused"]),
           "holds_the_published_curve": held, "elasticity": sens, "fade": fade_edges(p),
           "final": [r.final for r in these], "ok": [bool(r.ok) for r in these],
           "curve": {"low": [float(v) for v in b["low"]],
                     "median": [float(v) for v in b["median"]],
                     "high": [float(v) for v in b["high"]]},
           "seconds": round(time.time() - t1, 1)}
    log(f"  the last cycle: {out['interval_of_the_last_cycle'][0]:.3f} to "
        f"{out['interval_of_the_last_cycle'][1]:.3f} A.h, the published curve is "
        f"{'inside' if held['all_inside'] else 'outside'} it, elasticity "
        f"{sens['elasticity']:+.3f}, {out['seconds']:.0f} s")
    with open(os.path.join(results, f"sample_{slug(p.name)}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def electrolyte_fits() -> Dict[str, dict]:
    """The conductivity at 298 K and its activation energy for the cell's own electrolyte, each
    predicted by the composition model of item BT5 with the solvent mixture held out."""
    import run_shells as S
    rows = S.load_series()
    compositions = [str(r["composition"]) for r in rows]
    kernels = descriptor_kernels(S.descriptors(rows, "composition"))
    mine = [i for i, r in enumerate(rows) if str(r["id"]) == ELECTROLYTE_ID]
    if not mine:
        raise SystemExit(f"the measured series holds no electrolyte called {ELECTROLYTE_ID!r}")
    i = mine[0]
    out = {}
    for target, unit in (("log10_k_298", "log10 mS/cm"), ("ea_298", "kJ/mol")):
        y = np.array([float(r[target]) for r in rows], float)
        fit = nested_surrogate(kernels, y, compositions)
        out[target] = {"unit": unit, "measured": float(y[i]),
                       "predicted": float(fit["predicted"][i]), "sd": float(fit["sd"][i]),
                       "mae": fit["mae"], "mae_of_the_mean": fit["mae_of_the_mean"],
                       "z_rms": fit["z_rms"],
                       "coverage_2sd": float(fit["calibration"].coverage_2sd)}
        log(f"  {target}: {ELECTROLYTE_ID} measured {y[i]:.3f}, predicted "
            f"{fit['predicted'][i]:.3f} +/- {fit['sd'][i]:.3f} {unit}; over the {len(y)} "
            f"electrolytes, mixtures held out, mean absolute error {fit['mae']:.3f} against "
            f"{fit['mae_of_the_mean']:.3f} for the mean, z rms {fit['z_rms']:.2f}")
    return out


def stress_conditions() -> List[Tuple[float, float]]:
    """(temperature, rate) of every condition of the stress stage."""
    return ([(ROOM, r) for r in STRESS_RATES]
            + [(t, r) for t in STRESS_COLD for r in STRESS_COLD_RATES])


def stress_path(results: str, temperature: float, rate: float) -> str:
    return os.path.join(results, f"stress_{temperature:.0f}K_{rate:g}C.json")


def stage_stress(results: str, draws: int = STRESS_DRAWS, force: bool = False) -> None:
    """The electrolyte where it can matter: faster discharges, and colder cells.

    For every condition three curves are compared. The **published** one is the cell model's
    own; its electrolyte functions have no temperature dependence, so in the cold it is the cell
    with a warm electrolyte. The **measured** one replaces the conductivity by that of the
    electrolyte as CALiSol-23 measured it -- its conductivity at 298 K carried to the cell's
    temperature by its measured activation energy -- which is the curve a prediction should be
    judged against. The **predicted** interval comes from drawing the predicted conductivity and,
    in the cold, the predicted activation energy. At room temperature the Bruggeman exponent is
    drawn as well, because the two parameters together are what the electrolyte in a porous
    electrode is described by.

    The **second reference** is the electrolyte as Landesfeind and Gasteiger (2019) measured it,
    with all four of its transport parameters and their dependence on concentration and
    temperature (see :func:`_second_reference`). A condition written before the second reference
    existed keeps its draws and gets the second reference added.
    """
    from cmfqsar import cellmodel as CM

    pset, _ = read_set(results)
    cond = pset["Electrolyte conductivity [S.m-1]"]
    brug = pset["Positive electrode Bruggeman coefficient (electrolyte)"]
    p = CM.published_parameters(BASE)
    fits_path = os.path.join(results, "stress_fits.json")
    if force or not os.path.isfile(fits_path):
        log("the electrolyte's conductivity and activation energy:")
        fits = electrolyte_fits()
        with open(fits_path, "w", encoding="utf-8") as fh:
            json.dump(fits, fh, indent=1)
    with open(fits_path, encoding="utf-8") as fh:
        fits = json.load(fh)
    k, ea = fits["log10_k_298"], fits["ea_298"]
    for temperature, rate in stress_conditions():
        path = stress_path(results, temperature, rate)
        common = dict(cycles=1, base=BASE, kind=STRESS_MODEL, cycle=CM.discharge(rate),
                      temperature=temperature)
        entry = None
        if os.path.isfile(path) and not force:
            with open(path, encoding="utf-8") as fh:
                entry = json.load(fh)
            if "landesfeind" in entry:
                log(f"{temperature:.0f} K, {rate:g}C: done already")
                continue
        if entry is None:
            entry = _stress_condition(temperature, rate, draws, p, cond, brug, k, ea, common)
        entry["landesfeind"] = _second_reference(entry, p, cond.name, brug, draws, common)
        s = entry["landesfeind"]
        log(f"  the electrolyte Landesfeind and Gasteiger measured: conductivity alone "
            f"{s['conductivity_only']:.3f} A.h, all four parameters {s['all_four']:.3f}"
            + (f", Bruggeman {s['bruggeman']['low']:.3f} to {s['bruggeman']['high']:.3f}"
               if "bruggeman" in s else "") + f" ({s['seconds']:.0f} s)")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=1)


def _stress_condition(temperature: float, rate: float, draws: int, p, cond, brug,
                      k: dict, ea: dict, common: dict) -> dict:
    """One condition of the stress stage against the first reference, the electrolyte as
    CALiSol-23 measured it."""
    from cmfqsar import cellmodel as CM
    from cmfqsar.interfaces import conductivity_scale_at_temperature

    per_kelvin = 1000.0 / (8.314462618 * np.log(10.0))
    t0 = time.time()
    published = CM.run_cell(parameters=p, **common)
    reference = _conductivity_of(p, 1000.0, temperature)
    shift = 1.0 / temperature - 1.0 / ROOM
    measured_factor = 10.0 ** (k["measured"] - ea["measured"] * per_kelvin * shift
                               - reference)
    measured = CM.run_cell({cond.name: cond.apply(p[cond.name], measured_factor)},
                           parameters=p, **common)
    if temperature == ROOM:
        conductivity = cond
    else:
        conductivity = conductivity_scale_at_temperature(
            k["predicted"], k["sd"], ea["predicted"], ea["sd"], reference, temperature,
            cond.name, draws=2000, seed=SEED, source=cond.source + ", carried to "
            f"{temperature:.0f} K by the activation energy the same model predicts")
    entry = {"temperature": float(temperature), "rate": float(rate),
             "model": STRESS_MODEL, "draws": int(draws),
             "published": published.final, "published_ok": bool(published.ok),
             "measured": measured.final, "measured_ok": bool(measured.ok),
             "measured_factor": float(measured_factor),
             "predicted_factor": float(conductivity.value),
             "factor_interval": conductivity.interval(), "parameters": {}}
    runs = {}
    for parameter in ((conductivity, brug) if temperature == ROOM else (conductivity,)):
        these = CM.sample(parameter, draws=draws, seed=SEED, **common)
        runs[parameter.name] = these
        b = CM.band(these)
        lo, hi = ((float(b["low"][-1]), float(b["high"][-1])) if b["n"] > 1
                  else (float("nan"), float("nan")))
        entry["parameters"][parameter.name] = {
            "low": lo, "high": hi, "median": float(b["median"][-1]) if b["n"] > 1
            else float("nan"), "refused": float(b["refused"]),
            "holds_published": bool(lo <= published.final <= hi),
            "holds_measured": (bool(lo <= measured.final <= hi)
                               if parameter.name == cond.name else None)}
    shares = CM.variance_shares(runs)
    for name, s in shares.items():
        entry["parameters"][name]["share"] = s["share"]
        entry["parameters"][name]["sd"] = s["sd"]
    entry["seconds"] = round(time.time() - t0, 1)
    c = entry["parameters"][cond.name]
    log(f"{temperature:.0f} K, {rate:g}C: published {published.final:.3f} A.h, measured "
        f"electrolyte {measured.final:.3f}, predicted conductivity {c['low']:.3f} to "
        f"{c['high']:.3f}" + (f", Bruggeman {entry['parameters'][brug.name]['low']:.3f} to "
                              f"{entry['parameters'][brug.name]['high']:.3f}"
                              if brug.name in entry["parameters"] else "")
        + f" ({entry['seconds']:.0f} s)")
    return entry


def _second_reference(entry: dict, p, cond_name: str, brug, draws: int,
                      common: dict) -> dict:
    """The same condition against the second reference, the electrolyte as Landesfeind and
    Gasteiger (2019) measured it, with all four of its transport parameters.

    Two curves are run. *Conductivity only* replaces the conductivity alone; it is the fair
    reference for the predicted conductivity, which is the only one of the four that is
    predicted. *All four* replaces every transport parameter, with the transference number and
    the thermodynamic factor read at +20 degrees Celsius in colder cells (below it the study's
    values of these two are less precise, and its fit gives a negative transference number from
    0 degrees down). At room temperature the Bruggeman exponent is drawn
    again with all four in place, because its interval against the published electrolyte says
    nothing about how it would do in a cell with the real one.
    """
    from cmfqsar import cellmodel as CM

    t0 = time.time()
    temperature = float(entry["temperature"])
    only = CM.run_cell(CM.measured_electrolyte((cond_name,)), parameters=p, **common)
    four = CM.measured_electrolyte()
    full = CM.run_cell(four, parameters=p, **common)
    c = entry["parameters"][cond_name]
    out = {"source": f"{CM.MEASURED_ELECTROLYTE} in PyBaMM: Landesfeind and Gasteiger (2019)",
           "conductivity_only": only.final, "conductivity_only_ok": bool(only.ok),
           "all_four": full.final, "all_four_ok": bool(full.ok),
           "inside_the_measured_temperatures": bool(temperature >= LANDESFEIND_COLDEST - 1e-6),
           "transference_held_at": (CM.COLDEST_TRUSTED if temperature < CM.COLDEST_TRUSTED
                                    else None),
           "conductivity_holds_it": bool(c["low"] <= only.final <= c["high"])}
    if temperature == ROOM:
        runs = CM.sample(brug, draws=draws, seed=SEED, fixed=four, **common)
        b = CM.band(runs)
        lo, hi = ((float(b["low"][-1]), float(b["high"][-1])) if b["n"] > 1
                  else (float("nan"), float("nan")))
        out["bruggeman"] = {"low": lo, "high": hi,
                            "median": float(b["median"][-1]) if b["n"] > 1 else float("nan"),
                            "refused": float(b["refused"]),
                            "holds_it": bool(lo <= full.final <= hi)}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def read_stress(results: str) -> Optional[dict]:
    """What the stress stage wrote, or None when it has not been run."""
    fits_path = os.path.join(results, "stress_fits.json")
    if not os.path.isfile(fits_path):
        return None
    with open(fits_path, encoding="utf-8") as fh:
        fits = json.load(fh)
    rows = []
    for temperature, rate in stress_conditions():
        path = stress_path(results, temperature, rate)
        if not os.path.isfile(path):
            raise SystemExit(f"{path} is missing: run this benchmark with --stage stress")
        with open(path, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    return {"fits": fits, "conditions": rows}


# --------------------------------------------------------------------------- #
# The tolerance of the cell to the three parameters that are not predicted
# --------------------------------------------------------------------------- #
def electrolyte_value(f, temperature: float, concentration: float = 1000.0) -> float:
    """A transport parameter of the electrolyte, a number or a function, at one concentration
    (mol/m3) and one temperature (K)."""
    if not callable(f):
        return float(f)
    import pybamm
    v = f(pybamm.Scalar(float(concentration)), pybamm.Scalar(float(temperature)))
    return float(v.evaluate()) if hasattr(v, "evaluate") else float(v)


def tolerance_references(p, four) -> List[dict]:
    """The values each of the three items would use if nothing were predicted.

    `cheapest` marks the one that needs no data beyond what this program already has; rule 1 of
    the three items says that an item ends if that one is inside the tolerance. The other two
    for the diffusion coefficient are there to say where its error comes from.
    """
    from cmfqsar import cellmodel as CM

    t_pub = float(p[TRANSFERENCE])
    return [
        {"item": "BT9", "key": "nernst_einstein", "parameter": DIFFUSIVITY, "cheapest": True,
         "label": "Nernst-Einstein value from the measured conductivity, ideal solution",
         "value": CM.nernst_einstein_diffusivity(four[CONDUCTIVITY], t_pub)},
        {"item": "BT9", "key": "nernst_einstein_factor", "parameter": DIFFUSIVITY,
         "cheapest": False,
         "label": "the same, multiplied by the measured thermodynamic factor",
         "value": CM.nernst_einstein_diffusivity(four[CONDUCTIVITY], t_pub,
                                                 factor=four[THERMODYNAMIC])},
        {"item": "BT9", "key": "published_function", "parameter": DIFFUSIVITY,
         "cheapest": False,
         "label": "the function of the published set, which has no temperature in it",
         "value": p[DIFFUSIVITY]},
        {"item": "BT10", "key": "published_constant", "parameter": TRANSFERENCE,
         "cheapest": True, "label": f"the constant of the published set, {t_pub:g}",
         "value": t_pub},
        {"item": "BT11", "key": "ideal_solution", "parameter": THERMODYNAMIC, "cheapest": True,
         "label": "the ideal solution, which is the constant 1 of the published set",
         "value": float(p[THERMODYNAMIC])},
    ]


def tolerance_path(results: str, temperature: float, rate: float) -> str:
    return os.path.join(results, f"tolerance_{temperature:.0f}K_{rate:g}C.json")


def _usable(run) -> bool:
    """A run whose capacity can be compared: it completed, and its discharge ended at the
    voltage limit."""
    return bool(run.ok and run.at_cutoff and np.isfinite(run.final))


def _tolerance_condition(temperature: float, rate: float, p, four) -> dict:
    """One condition: the response curve of each of the three parameters around its measured
    function, the tolerated ranges read from it, and the cheapest references."""
    from cmfqsar import cellmodel as CM
    from cmfqsar.interfaces import CellParameter

    t0 = time.time()
    common = dict(cycles=1, base=BASE, kind=STRESS_MODEL, cycle=CM.discharge(rate),
                  temperature=temperature, parameters=p)
    ref = CM.run_cell(four, details=True, **common)
    if not _usable(ref):
        raise SystemExit(f"{temperature:.0f} K, {rate:g}C: the cell with the measured "
                         f"electrolyte gave no usable discharge ({ref.message or ref.termination})")
    entry = {"temperature": float(temperature), "rate": float(rate), "model": STRESS_MODEL,
             "reference": ref.final, "reference_concentration": list(ref.concentration),
             "at_1000": {name: electrolyte_value(four[name], temperature)
                         for name in CM.ELECTROLYTE_ENTRIES},
             "parameters": {}, "references": {}}
    for spec in TOLERANCE_PARAMETERS:
        scale = spec["kind"] == "scale"
        centre = 1.0 if scale else 0.0
        parameter = CellParameter(name=spec["name"], value=centre, kind=spec["kind"])
        runs = CM.response(parameter, spec["values"], fixed=four, **common)
        usable = [_usable(r) for r in runs]
        capacity = [r.final if u else float("nan") for r, u in zip(runs, usable)]
        at_centre = capacity[list(spec["values"]).index(centre)]
        if not np.isclose(at_centre, ref.final, rtol=0, atol=1e-6):
            raise SystemExit(f"{spec['name']}: the unchanged point gives {at_centre!r} and the "
                             f"reference run {ref.final!r}; they have to be the same run")
        ranges = {f"{share:g}": CM.tolerated_range(spec["values"], capacity, centre,
                                                   share * ref.final, log=scale)
                  for share in TOLERANCE_SHARES}
        entry["parameters"][spec["name"]] = {
            "item": spec["item"], "label": spec["label"], "kind": spec["kind"],
            "values": [float(v) for v in spec["values"]],
            "capacity": [float(r.final) for r in runs], "usable": usable,
            "ended": [r.termination for r in runs],
            "lowest_concentration": [float(r.concentration[0]) for r in runs],
            "highest_concentration": [float(r.concentration[1]) for r in runs],
            "ranges": ranges}
    for r in tolerance_references(p, four):
        run = CM.run_cell({**four, r["parameter"]: r["value"]}, details=True, **common)
        ok = _usable(run)
        entry["references"][r["key"]] = {
            "item": r["item"], "parameter": r["parameter"], "label": r["label"],
            "cheapest": bool(r["cheapest"]), "usable": ok, "ended": run.termination,
            "capacity": float(run.final) if ok else float("nan"),
            "difference": float(run.final - ref.final) if ok else float("nan"),
            "share": float((run.final - ref.final) / ref.final) if ok else float("nan"),
            "at_1000": electrolyte_value(r["value"], temperature)}
    entry["seconds"] = round(time.time() - t0, 1)
    return entry


def curve_values(values: Sequence[float], capacities: Sequence[float], at: Sequence[float],
                 log: bool = False) -> np.ndarray:
    """The capacity a response curve gives at the values `at`, read by linear interpolation
    between its points (on the logarithm of a factor). Points without a capacity are left out,
    and outside the outermost points the curve is read as flat."""
    x = np.asarray(values, float)
    q = np.asarray(capacities, float)
    keep = np.isfinite(q)
    x, q = x[keep], q[keep]
    order = np.argsort(x)
    x, q = x[order], q[order]
    at = np.asarray(at, float)
    return np.interp(np.log(at) if log else at, np.log(x) if log else x, q)


def _tolerance_check(entry: dict, p, four, draws: int = TOLERANCE_CHECK_DRAWS) -> dict:
    """Does the response curve say what a run says?

    A tolerance is read from a curve of a few points, with straight lines between them. For
    each parameter the deviation tolerated at the widest share is taken, values are drawn with
    it, and the cell is run once per draw. Each run is compared with the capacity the curve
    gives at the same drawn value: the largest difference is the error of reading between
    points. The two 95 per cent intervals, of the runs and of the curve at the same draws, are
    given as well, and so is the share of the draws whose run left the limit. That share
    should be near 2.5 per cent, since the deviation is set by the nearer end of the range and
    an error of that size passes that end in one draw of forty.
    """
    from cmfqsar import cellmodel as CM
    from cmfqsar.interfaces import CellParameter

    temperature, rate = float(entry["temperature"]), float(entry["rate"])
    common = dict(cycles=1, base=BASE, kind=STRESS_MODEL, cycle=CM.discharge(rate),
                  temperature=temperature)
    share = f"{max(TOLERANCE_SHARES):g}"
    reference = float(entry["reference"])
    out = {"temperature": temperature, "rate": rate, "draws": int(draws), "share": float(share),
           "reference": reference, "parameters": {}}
    for spec in TOLERANCE_PARAMETERS:
        rec = entry["parameters"][spec["name"]]
        scale = spec["kind"] == "scale"
        sd = float(rec["ranges"][share]["sd"])
        width = float(rec["ranges"][share]["width"])
        if not np.isfinite(sd) or sd <= 0:
            out["parameters"][spec["name"]] = {"sd": sd, "skipped": True}
            continue
        parameter = CellParameter(name=spec["name"], value=1.0 if scale else 0.0, sd=sd,
                                  kind=spec["kind"],
                                  distribution="lognormal" if scale else "normal")
        runs = CM.sample(parameter, draws=draws, seed=SEED, fixed=four, **common)
        drawn = np.array([r.overrides[spec["name"]] for r in runs], float)
        got = np.array([r.final if r.ok else float("nan") for r in runs], float)
        capacity = [c if u else float("nan") for c, u in zip(rec["capacity"], rec["usable"])]
        read = curve_values(rec["values"], capacity, drawn, log=scale)
        ok = np.isfinite(got)
        difference = read[ok] - got[ok]
        out["parameters"][spec["name"]] = {
            "sd": sd, "skipped": False, "width": width, "refused": float(np.mean(~ok)),
            "drawn_low": float(np.percentile(got[ok], 2.5)),
            "drawn_high": float(np.percentile(got[ok], 97.5)),
            "curve_low": float(np.percentile(read[ok], 2.5)),
            "curve_high": float(np.percentile(read[ok], 97.5)),
            "largest_difference": float(np.max(np.abs(difference))),
            "mean_difference": float(np.mean(difference)),
            "outside": float(np.mean(np.abs(got[ok] - reference) > width))}
    return out


def stage_tolerance(results: str, force: bool = False) -> None:
    """How accurate the three parameters of the electrolyte that are not predicted have to be
    (rule 1 of plan items BT9, BT10 and BT11).

    The cell holds the electrolyte with all four transport parameters as Landesfeind and
    Gasteiger (2019) measured them. One parameter at a time is then multiplied by a factor (the
    diffusion coefficient, the thermodynamic factor) or shifted (the transference number) over
    a fixed set of points, and the full model is run once per point, under the eight conditions
    of the stress stage. The range of the parameter inside which the capacity stays within a
    stated share of the capacity with the measured function is read from that curve. The
    cheapest reference of each item is run under the same conditions.
    """
    from cmfqsar import cellmodel as CM

    p = CM.published_parameters(BASE)
    four = CM.measured_electrolyte()
    for temperature, rate in stress_conditions():
        path = tolerance_path(results, temperature, rate)
        if os.path.isfile(path) and not force:
            log(f"{temperature:.0f} K, {rate:g}C: the tolerance is there already")
            continue
        entry = _tolerance_condition(temperature, rate, p, four)
        wide = f"{max(TOLERANCE_SHARES):g}"
        parts = []
        for spec in TOLERANCE_PARAMETERS:
            r = entry["parameters"][spec["name"]]["ranges"][wide]
            parts.append(f"{spec['label']} {_range(r, spec['kind'])}")
        log(f"{temperature:.0f} K, {rate:g}C: {entry['reference']:.3f} A.h with the measured "
            f"electrolyte; inside {100 * float(wide):.0f} %: " + "; ".join(parts)
            + f" ({entry['seconds']:.0f} s)")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=1)
    check_path = os.path.join(results, "tolerance_check.json")
    if force or not os.path.isfile(check_path):
        with open(tolerance_path(results, *TOLERANCE_CHECK), encoding="utf-8") as fh:
            entry = json.load(fh)
        check = _tolerance_check(entry, p, four)
        for name, c in check["parameters"].items():
            if not c["skipped"]:
                log(f"  check, {name}: runs {c['drawn_low']:.3f} to {c['drawn_high']:.3f} "
                    f"A.h, the curve at the same draws {c['curve_low']:.3f} to "
                    f"{c['curve_high']:.3f}, largest difference "
                    f"{c['largest_difference']:.3f}, {100 * c['outside']:.0f} % outside")
        with open(check_path, "w", encoding="utf-8") as fh:
            json.dump(check, fh, indent=1)


def read_tolerance(results: str) -> Optional[dict]:
    """What the tolerance stage wrote, or None when it has not been run."""
    paths = [tolerance_path(results, t, r) for t, r in stress_conditions()]
    if not any(os.path.isfile(path) for path in paths):
        return None
    rows = []
    for path in paths:
        if not os.path.isfile(path):
            raise SystemExit(f"{path} is missing: run this benchmark with --stage tolerance")
        with open(path, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    check = None
    check_path = os.path.join(results, "tolerance_check.json")
    if os.path.isfile(check_path):
        with open(check_path, encoding="utf-8") as fh:
            check = json.load(fh)
    return {"conditions": rows, "check": check}


def _range(r: dict, kind: str) -> str:
    """A tolerated range as text. "<" or ">" before an end says that the points ran out there
    with the capacity still inside, so the range is at least that wide; "*" after an end says
    that the next point gave no capacity."""
    sign = kind == "shift"

    def end(v, ends, low):
        text = _n(v, 2, sign)
        if ends == "open":
            return ("< " if low else "> ") + text
        return text + ("*" if ends == "refused" else "")
    return f"{end(r['low'], r['low_ends'], True)} to {end(r['high'], r['high_ends'], False)}"


def _sd(r: dict) -> str:
    return (">= " if r["sd_is_a_lower_bound"] else "") + _n(r["sd"], 3)


def _where(row: dict) -> str:
    return f"{row['temperature']:.0f} K and {row['rate']:g}C"


def tolerance_lines(tol: dict) -> List[str]:
    """The section of the table on the three parameters that are not predicted."""
    rows = tol["conditions"]
    n = len(rows)
    out = ["### How accurate the three parameters without a prediction have to be", ""]
    out.append(
        "This program does not predict the salt diffusion coefficient, the cation transference "
        "number or the thermodynamic factor of the electrolyte (plan items BT9, BT10 and BT11). "
        "Before any of them is simulated, this stage asks how accurate each one has to be. The "
        "cell holds the electrolyte with all four transport parameters as Landesfeind and "
        "Gasteiger (2019) measured them. One parameter at a time is then multiplied by a factor "
        "(the diffusion coefficient and the thermodynamic factor) or shifted by a constant (the "
        "transference number) over a fixed set of points, and the full model is run once per "
        "point. Nothing is drawn: the capacity for any spread of the parameter can be read "
        "from such a curve.")
    out.append("")
    out.append(
        "The *tolerated range* is the connected range around the measured function inside "
        "which the discharge capacity stays within the stated share of the capacity with the "
        "measured function. The *tolerated deviation* is the largest standard deviation of an "
        "error whose 95 % interval stays inside that range: of the natural logarithm for a "
        "factor, of the number itself for a shift. The nearer end of the range sets it. An end "
        "written with < or > is the outermost point that was run, so the range is at least "
        "that wide, and a deviation written with >= is a lower bound for the same reason. An "
        "end written with * is the last point before one at which the model gave no usable "
        "discharge.")
    out.append("")
    for share in sorted(TOLERANCE_SHARES):
        key = f"{share:g}"
        out.append(f"**The capacity may move by {100 * share:g} %.**")
        out.append("")
        out.append("| Temperature, K | Rate | Capacity with the measured electrolyte, A.h | "
                   "Diffusion coefficient: tolerated factor | Tolerated deviation of its "
                   "logarithm | Transference number: tolerated shift | Tolerated deviation | "
                   "Thermodynamic factor: tolerated factor | Tolerated deviation of its "
                   "logarithm |")
        out.append("|---:|---:|---:|---|---:|---|---:|---|---:|")
        for row in rows:
            cells = [f"{row['temperature']:.0f}", f"{row['rate']:g}C", _n(row["reference"], 3)]
            for spec in TOLERANCE_PARAMETERS:
                r = row["parameters"][spec["name"]]["ranges"][key]
                cells += [_range(r, spec["kind"]), _sd(r)]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    room = next((r for r in rows if r["temperature"] == ROOM), rows[0])
    v = room["at_1000"]
    out.append(
        f"For scale: at 1 mol/L and {room['temperature']:.0f} K the measured diffusion "
        f"coefficient is {_n(v[DIFFUSIVITY])} m2/s, the measured transference number "
        f"{_n(v[TRANSFERENCE], 3)} and the measured thermodynamic factor "
        f"{_n(v[THERMODYNAMIC], 2)}.")
    out.append("")

    # the least tolerance of each parameter, over the conditions
    wide = f"{max(TOLERANCE_SHARES):g}"
    parts = []
    for spec in TOLERANCE_PARAMETERS:
        tight = min(rows, key=lambda r: r["parameters"][spec["name"]]["ranges"][wide]["sd"])
        r = tight["parameters"][spec["name"]]["ranges"][wide]
        if spec["kind"] == "scale":
            meaning = (f"a factor between {_n(np.exp(-1.96 * r['sd']), 2)} and "
                       f"{_n(np.exp(1.96 * r['sd']), 2)}")
        else:
            meaning = f"an error within {_n(1.96 * r['sd'], 3)} either way"
        parts.append(f"the {spec['label']} at {_where(tight)}, where the tolerated deviation "
                     f"is {_sd(r)} ({meaning})")
    out.append(f"**Where each parameter has to be most accurate, at "
               f"{100 * float(wide):g} % of the capacity:** " + "; ".join(parts) + ".")
    out.append("")

    warm = [r for r in rows if r["temperature"] == ROOM]
    ranges = [r["parameters"][DIFFUSIVITY]["ranges"][wide] for r in warm]
    if warm and all(r["high_ends"] == "open" for r in ranges):
        out.append(
            "**At room temperature only a diffusion coefficient that is too small is "
            f"punished.** A larger one costs less than {100 * float(wide):g} % of the capacity "
            f"at every factor that was tried, up to {_n(max(r['high'] for r in ranges), 0)}. "
            "The lower end of the tolerated factor rises with the rate: "
            + ", ".join(f"{_n(r['low'], 2)} at {row['rate']:g}C"
                        for row, r in zip(warm, ranges))
            + ". A prediction of this parameter should therefore err on the high side.")
        out.append("")

    out.append("### What the cheapest reference of each item costs")
    out.append("")
    out.append(
        "Each of the three items names a value that needs no new data. For the diffusion "
        "coefficient it is the Nernst-Einstein value, D = 2 t (1 - t) k R T / (F^2 c), from the "
        "measured conductivity k and the transference number t of the published set: the "
        "coefficient of an ideal solution of fully dissociated ions. For the other two it is "
        "the constant of the published set. Each is put in place of the measured function, "
        "with the other three as measured, and the difference of the capacity is given in A.h "
        "and as a share of the capacity with the measured electrolyte. Two more values of the "
        "diffusion coefficient are run to say where the error of the first comes from.")
    out.append("")
    keys = list(rows[0]["references"])
    head = {"nernst_einstein": "Diffusion coefficient: Nernst-Einstein, ideal solution",
            "nernst_einstein_factor": "Diffusion coefficient: Nernst-Einstein times the "
                                      "measured thermodynamic factor",
            "published_function": "Diffusion coefficient: function of the published set",
            "published_constant": "Transference number: constant of the published set",
            "ideal_solution": "Thermodynamic factor: 1"}
    out.append("| Temperature, K | Rate | " + " | ".join(head.get(k, k) for k in keys) + " |")
    out.append("|---:|---:|" + "---:|" * len(keys))
    for row in rows:
        cells = [f"{row['temperature']:.0f}", f"{row['rate']:g}C"]
        for k in keys:
            r = row["references"][k]
            cells.append(f"{_n(r['difference'], 3, True)} ({_n(100 * r['share'], 1, True)} %)"
                         if r["usable"] else "no usable discharge")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    ne = room["references"]["nernst_einstein"]["at_1000"]
    nef = room["references"]["nernst_einstein_factor"]["at_1000"]
    out.append(
        f"At 1 mol/L and {room['temperature']:.0f} K the Nernst-Einstein value is {_n(ne)} "
        f"m2/s, {_n(v[DIFFUSIVITY] / ne, 1)} times below the measured {_n(v[DIFFUSIVITY])}. "
        f"Multiplied by the measured thermodynamic factor it is {_n(nef)}, "
        f"{_n(v[DIFFUSIVITY] / nef, 1)} times below. The function of the published set gives "
        f"{_n(room['references']['published_function']['at_1000'])} at every temperature.")
    out.append("")

    def verdict(key: str, name: str) -> str:
        inside = {s: sum(1 for r in rows if r["references"][key]["usable"]
                         and abs(r["references"][key]["share"]) <= s)
                  for s in TOLERANCE_SHARES}
        usable = [r for r in rows if r["references"][key]["usable"]]
        unusable = [r for r in rows if not r["references"][key]["usable"]]
        ends = all(v == n for v in inside.values())
        text = (f"**{rows[0]['references'][key]['item']}, the {name}.** The reference is inside "
                + " and inside ".join(f"{100 * s:g} % of the capacity under {inside[s]} of {n} "
                                      f"conditions" for s in sorted(TOLERANCE_SHARES)) + ". ")
        if usable:
            worst = max(usable, key=lambda r: abs(r["references"][key]["share"]))
            w = worst["references"][key]
            text += (f"Its largest difference is {_n(w['difference'], 3, True)} A.h "
                     f"({_n(100 * w['share'], 1, True)} %) at {_where(worst)}. ")
        if unusable:
            text += (f"At {', '.join(_where(r) for r in unusable)} the cell model gave no "
                     "usable discharge with this reference: the discharge did not end at the "
                     "voltage limit, so there is no capacity to compare, and the condition is "
                     "counted as outside. ")
        if ends:
            return text + ("The reference is inside the tolerance under every condition, so by "
                           "rule 1 the item ends here.")
        if inside[max(TOLERANCE_SHARES)] == 0:
            return text + ("The reference is outside the tolerance under every condition, so "
                           "the item does not end here.")
        return text + ("The reference is not inside the tolerance under every condition, so "
                       "the item does not end here; the conditions where it is inside are the "
                       "ones where this parameter does not need a prediction.")
    out.append(verdict("nernst_einstein", "salt diffusion coefficient"))
    out.append("")
    out.append(verdict("published_constant", "cation transference number"))
    out.append("")
    out.append(verdict("ideal_solution", "thermodynamic factor"))
    out.append("")

    # where the measured functions are read outside what was measured
    lo, hi = MEASURED_CONCENTRATIONS
    outside = [r for r in rows if r["reference_concentration"][0] < lo
               or r["reference_concentration"][1] > hi]
    exhausted = [r for r in rows if r["reference_concentration"][0] < EXHAUSTED]
    if outside:
        far = max(outside, key=lambda r: r["reference_concentration"][1])
        out.append(
            f"**The reference itself leaves the measured range under {len(outside)} of {n} "
            f"conditions.** The study measured between {lo:g} and {hi:g} mol/m3. With the "
            f"measured electrolyte in place, the concentration somewhere in the cell goes "
            f"outside that range at " + ", ".join(_where(r) for r in outside)
            + f"; the highest is {_n(far['reference_concentration'][1], 0)} mol/m3 at "
            f"{_where(far)}. "
            + (f"Under {len(exhausted)} of these the electrolyte is exhausted somewhere in the "
               f"cell (below {EXHAUSTED:g} mol/m3), which is what ends the discharge early. "
               if exhausted else "")
            + "There the fitted functions are read outside the measurements they were fitted "
              "to, so the capacity called measured is the capacity of an extrapolation.")
        out.append("")
    cold = [r for r in rows if r["temperature"] < 293.15]
    if cold:
        out.append(
            "In the colder cells the transference number and the thermodynamic factor called "
            "measured are the values at 293 K, as in the second reference, because the study "
            "calls its colder values of these two less precise. A tolerance of these two in "
            "the cold is therefore a tolerance around an assumed value.")
        out.append("")
    check = tol.get("check")
    if check:
        done = {k: c for k, c in check["parameters"].items() if not c.get("skipped")}
        if done:
            labels = {spec["name"]: spec["label"] for spec in TOLERANCE_PARAMETERS}
            out.append(
                f"**The reading of the curve, checked by drawing.** A tolerance is read from a "
                f"curve of a few points with straight lines between them. At "
                f"{check['temperature']:.0f} K and {check['rate']:g}C each parameter was drawn "
                f"{check['draws']} times with the deviation tolerated at "
                f"{100 * check['share']:g} %, and the cell was run once per draw. Each run is "
                f"compared with the capacity the curve gives at the same drawn value.")
            out.append("")
            out.append("| Parameter | Deviation drawn with | Largest difference between curve "
                       "and run, A.h | 95 % interval of the runs, A.h | 95 % interval of the "
                       "curve at the same draws, A.h | Draws whose run left the limit |")
            out.append("|---|---:|---:|---|---|---:|")
            for k, c in done.items():
                out.append(f"| {labels[k][0].upper()}{labels[k][1:]} | {_n(c['sd'], 3)} | "
                           f"{_n(c['largest_difference'], 3)} | {_n(c['drawn_low'], 3)} to "
                           f"{_n(c['drawn_high'], 3)} | {_n(c['curve_low'], 3)} to "
                           f"{_n(c['curve_high'], 3)} | {_n(100 * c['outside'], 0)} % |")
            out.append("")
            worst = max(done.values(), key=lambda c: c["largest_difference"])
            limit = check["share"] * check["reference"]
            out.append(
                f"The capacity with the measured electrolyte is {_n(check['reference'], 3)} "
                f"A.h, so the limit of {100 * check['share']:g} % is {_n(limit, 3)} A.h. The "
                f"largest error of reading between points is "
                f"{_n(worst['largest_difference'], 3)} A.h, which is "
                f"{_n(100 * worst['largest_difference'] / limit, 0)} % of that limit. About "
                f"2.5 % of the draws are expected to leave the limit, because the deviation "
                f"is set by the nearer end of the range.")
            out.append("")
    return out


def read_samples(results: str, pset) -> Dict[str, dict]:
    """What the sampling stages wrote, one parameter at a time."""
    out = {}
    for p in pset:
        path = os.path.join(results, f"sample_{slug(p.name)}.json")
        if not os.path.isfile(path):
            raise SystemExit(f"{path} is missing: run this benchmark with "
                             f"--stage sample --parameter '{p.name}' first")
        with open(path, encoding="utf-8") as fh:
            out[p.name] = json.load(fh)
    return out


def shares_of(samples: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    """The share of the variance of the last cycle's capacity each parameter explains."""
    from cmfqsar import cellmodel as CM
    runs = {}
    for name, s in samples.items():
        runs[name] = [CM.CellRun(capacity=np.array([v], float), cycles=1, ok=bool(ok))
                      for v, ok in zip(s["final"], s["ok"])]
    return CM.variance_shares(runs)


def run(results: str = RESULTS, draws: int = DRAWS, cycles: int = CYCLES,
        stages: Sequence[str] = STAGES,
        only: Optional[str] = None, force: bool = False) -> Optional[dict]:
    """The benchmark, in stages. Returns the whole result when the last stage runs."""
    from cmfqsar import cellmodel as CM

    t0 = time.time()
    os.makedirs(results, exist_ok=True)
    if "channels" in stages:
        if force or not os.path.isfile(os.path.join(results, "parameters.json")):
            stage_channels(results)
        else:
            log("the parameter set is there already; --force rebuilds it")
    pset, details = read_set(results)
    if "reference" in stages:
        if force or not os.path.isfile(os.path.join(results, "reference.json")):
            stage_reference(results, cycles)
        else:
            log("the published curve is there already")
    if "sample" in stages:
        for p in pset:
            if only is not None and p.name != only:
                continue
            path = os.path.join(results, f"sample_{slug(p.name)}.json")
            if os.path.isfile(path) and not force:
                log(f"{p.name}: drawn already")
                continue
            stage_sample(results, p.name, draws, cycles)
    if "stress" in stages:
        stage_stress(results, force=force)
    if "tolerance" in stages:
        stage_tolerance(results, force=force)
    if "finish" not in stages:
        return None

    reference = read_reference(results)
    samples = read_samples(results, pset)
    per_parameter = {name: {k: v for k, v in s.items() if k not in ("final", "ok")}
                     for name, s in samples.items()}
    shares = shares_of(samples)
    width = TOL_SHARE * float(reference["final"])
    log(f"the tolerance rule: how accurate the barrier has to be for an interval narrower than "
        f"{width:.3f} A.h ({100 * TOL_SHARE:.0f} % of the published capacity)")
    tolerance = CM.tolerance_for_capacity(
        pset["Positive particle diffusivity [m2.s-1]"], width, TOL_SCALES,
        published=float(reference["final"]), draws=TOL_DRAWS, cycles=TOL_CYCLES, base=BASE,
        kind=KIND)
    barrier_sd = float(np.std(np.asarray(
        details["Positive particle diffusivity [m2.s-1]"]["barriers"], float)))
    for row in tolerance:
        row["barrier_sd"] = float(barrier_sd * row["scale"])
        log(f"  the barrier known to {row['barrier_sd']:.3f} eV: interval "
            f"{row['low']:.3f} to {row['high']:.3f} A.h, "
            f"{'narrow' if row['narrow'] else 'too wide'} and "
            f"{'holds' if row['holds'] else 'does not hold'} the published capacity")

    out = {"base": BASE, "model": KIND, "cycles": int(reference["cycles"]),
           "draws": int(max(s["draws"] for s in samples.values())),
           "published": {k: v for k, v in reference.items() if k != "cycles"},
           "parameters": pset.as_dict(), "channels": {
               k: {kk: vv for kk, vv in v.items() if kk != "barriers"}
               for k, v in details.items()},
           "per_parameter": per_parameter, "variance_shares": shares,
           "tolerance": tolerance, "tolerance_width": width,
           "stress": read_stress(results),
           "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(results, "cellmodel.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def _n(v, nd: int = 3, sign: bool = False) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "none"
    if not np.isfinite(x):
        return "none"
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f"{x:.2e}"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def stress_lines(stress: dict, params: Dict[str, dict]) -> List[str]:
    """The section of the table on the electrolyte at faster discharge and in the cold."""
    cond = "Electrolyte conductivity [S.m-1]"
    brug = "Positive electrode Bruggeman coefficient (electrolyte)"
    k, ea = stress["fits"]["log10_k_298"], stress["fits"]["ea_298"]
    rows = stress["conditions"]
    model = rows[0]["model"] if rows else STRESS_MODEL
    out = ["## The electrolyte at faster discharge and in the cold", ""]
    out.append(
        "At the one-hour rate the cell hardly responds to its electrolyte, so the two "
        "electrolyte-side parameters are drawn again where it can matter: faster discharges at "
        f"room temperature, and colder cells. Each run is one discharge of the full {model} "
        f"model, {rows[0]['draws'] if rows else STRESS_DRAWS} draws per parameter, because the "
        "single-particle model approximates the electrolyte and is not meant for high rates.")
    out.append("")
    out.append("The published electrolyte functions have **no temperature dependence**; of the "
               "whole parameter set only the two exchange-current densities depend on "
               "temperature. A cold run of the published set is therefore a cold cell with a "
               "warm electrolyte. The program supplies what is missing: the composition model "
               "of item BT5 predicts the activation energy of the conductivity as well, with the "
               "solvent mixture held out.")
    out.append("")
    out.append("| quantity, for the cell's electrolyte | measured | predicted | over the 62 "
               "electrolytes: error, and for the mean | z rms |")
    out.append("|---|---:|---:|---|---:|")
    for label, f in (("conductivity at 298 K, log10 mS/cm", k),
                     ("activation energy, kJ/mol", ea)):
        out.append(f"| {label} | {_n(f['measured'], 3)} | {_n(f['predicted'], 3)} +/- "
                   f"{_n(f['sd'], 3)} | {_n(f['mae'], 3)} against "
                   f"{_n(f['mae_of_the_mean'], 3)} | {_n(f['z_rms'], 2)} |")
    out.append("")
    out.append("Three capacities are compared under every condition. *Published* is the cell "
               "model's own. *Measured electrolyte* replaces its conductivity by the one "
               "CALiSol-23 measured for this electrolyte, carried to the cell's temperature by "
               "the measured activation energy: that is the curve a prediction is judged "
               "against. The intervals come from drawing the predicted parameters.")
    out.append("")
    out.append("| temperature, K | rate | published, A.h | measured electrolyte, A.h | "
               "interval from the conductivity, A.h | holds the measured electrolyte? | "
               "interval from the Bruggeman exponent, A.h | holds the published curve? | "
               "share of the Bruggeman exponent |")
    out.append("|---:|---:|---:|---:|---|---|---|---|---:|")
    for r in rows:
        c = r["parameters"][cond]
        b = r["parameters"].get(brug)
        out.append(
            f"| {r['temperature']:.0f} | {r['rate']:g}C | {_n(r['published'], 3)} | "
            f"{_n(r['measured'], 3)} | {_n(c['low'], 3)} to {_n(c['high'], 3)} | "
            f"{'yes' if c['holds_measured'] else 'no'} | "
            + (f"{_n(b['low'], 3)} to {_n(b['high'], 3)} | "
               f"{'yes' if b['holds_published'] else 'no'} | "
               f"{_n(100 * b['share'], 1)} % |" if b else "-- | -- | -- |"))
    out.append("")
    warm = [r for r in rows if r["temperature"] == ROOM]
    cold = [r for r in rows if r["temperature"] != ROOM]
    held = sum(bool(r["parameters"][cond]["holds_measured"]) for r in rows)
    cold_out = sum(not (r["parameters"][cond]["low"] <= r["published"]
                        <= r["parameters"][cond]["high"]) for r in cold)
    worst = max(warm, key=lambda r: r["published"] - r["parameters"][brug]["high"]) if warm \
        else None
    out.append(
        f"**The conductivity is right, now that it can be seen.** The predicted interval holds "
        f"the capacity of the measured electrolyte under {held} of {len(rows)} conditions. In "
        f"the cold it also excludes the published curve under {cold_out} of {len(cold)}: the "
        "published set, whose electrolyte does not get colder, gives too much capacity, and the "
        "predicted activation energy corrects it by about as much as the measured one does. "
        "The effect is modest -- a few hundredths of an ampere-hour, at most about 1 % -- "
        "because the conductivity is the only electrolyte property that is carried to the cold; "
        "the electrolyte diffusivity and the two solid diffusivities stay at their room values "
        "in this parameter set.")
    out.append("")
    if worst is not None:
        b = worst["parameters"][brug]
        slow = min(warm, key=lambda r: r["rate"])
        slow_miss = slow["published"] - slow["parameters"][brug]["high"]
        out.append(
            "**The tortuosity takes over at faster discharge, and its error grows with it.** "
            f"At {worst['rate']:g}C the Bruggeman exponent explains {100 * b['share']:.1f} % of "
            f"the variance, its interval runs from {_n(b['low'], 3)} to {_n(b['high'], 3)} A.h, "
            f"and it misses the published {_n(worst['published'], 3)} A.h by "
            f"{_n(worst['published'] - b['high'], 3)} A.h. At {slow['rate']:g}C the same miss "
            f"was {_n(slow_miss, 4)} A.h. The generated structures are more tortuous than the "
            "1.5 the cell's authors fitted. How much that costs at a fast discharge depends on "
            "the electrolyte the cell is given, as the second reference below shows.")
        out.append("")
    if rows and all("landesfeind" in r for r in rows):
        out += second_reference_lines(rows)
    return out


def second_reference_lines(rows: List[dict]) -> List[str]:
    """The part of the stress section on the second reference: the electrolyte with all four
    of its transport parameters measured (Landesfeind and Gasteiger, 2019)."""
    cond = "Electrolyte conductivity [S.m-1]"
    brug = "Positive electrode Bruggeman coefficient (electrolyte)"
    held_at = next((r["landesfeind"]["transference_held_at"] for r in rows
                    if r["landesfeind"]["transference_held_at"]), None)
    out = ["### The second reference: all four transport parameters measured", ""]
    out.append(
        "The published cell takes its conductivity and electrolyte diffusivity from one study "
        "at room temperature and holds the transference number (the share of the current the "
        "lithium ion carries) and the thermodynamic factor constant. Landesfeind and Gasteiger "
        "(2019) measured all four for the same electrolyte, LiPF6 in EC:EMC 3:7 by weight, "
        f"between {LANDESFEIND_COLDEST:.0f} and 323 K and between 0.1 and 3.0 mol/L; PyBaMM "
        "ships their fitted functions in its ORegan2022 parameter set. *Conductivity only* puts "
        "their conductivity alone into the cell, which is the fair reference for the predicted "
        "conductivity. *All four* puts in every transport parameter"
        + (f"; in colder cells the transference number and the thermodynamic factor are read "
           f"at {held_at:.0f} K, because the study says these two are less precise below "
           "+20 degrees Celsius and its fit gives a negative transference number from 0 degrees "
           "down" if held_at else "") + ". At room temperature the Bruggeman exponent is drawn "
        "again with all four in place.")
    out.append("")
    out.append("| temperature, K | rate | published, A.h | measured conductivity only, A.h | "
               "interval from the predicted conductivity, A.h | holds it? | all four measured, "
               "A.h | interval from the Bruggeman exponent with all four measured, A.h | "
               "holds it? |")
    out.append("|---:|---:|---:|---:|---|---|---:|---|---|")
    for r in rows:
        s, c = r["landesfeind"], r["parameters"][cond]
        b = s.get("bruggeman")
        mark = "" if s["inside_the_measured_temperatures"] else " (outside the measured range)"
        out.append(
            f"| {r['temperature']:.0f}{mark} | {r['rate']:g}C | {_n(r['published'], 3)} | "
            f"{_n(s['conductivity_only'], 3)} | {_n(c['low'], 3)} to {_n(c['high'], 3)} | "
            f"{'yes' if s['conductivity_holds_it'] else 'no'} | {_n(s['all_four'], 3)} | "
            + (f"{_n(b['low'], 3)} to {_n(b['high'], 3)} | {'yes' if b['holds_it'] else 'no'} |"
               if b else "-- | -- |"))
    out.append("")
    held = [r for r in rows if r["landesfeind"]["conductivity_holds_it"]]
    missed = [r for r in rows if not r["landesfeind"]["conductivity_holds_it"]]

    def miss(r):
        v, c = r["landesfeind"]["conductivity_only"], r["parameters"][cond]
        return max(c["low"] - v, v - c["high"])

    words = "; ".join(
        f"{r['temperature']:.0f} K and {r['rate']:g}C by {_n(miss(r), 3)} A.h"
        + ("" if r["landesfeind"]["inside_the_measured_temperatures"]
           else ", below the coldest measured temperature") for r in missed)
    out.append(
        f"**The predicted conductivity against the measured temperature dependence.** The "
        f"predicted interval holds the capacity of the measured conductivity under {len(held)} "
        f"of {len(rows)} conditions" + (f". It misses at {words}." if missed else ".")
        + " A single activation energy, which is what the prediction carries to the cold, "
        "describes the conductivity near room temperature; the measured conductivity falls "
        "faster the colder it gets.")
    out.append("")
    warm = [r for r in rows if r["temperature"] == ROOM]
    if warm:
        fast = max(warm, key=lambda r: r["rate"])
        s, b = fast["landesfeind"], fast["landesfeind"]["bruggeman"]
        old = fast["parameters"][brug]
        gap_old = fast["published"] - old["high"]
        gap_new = s["all_four"] - b["high"] if not b["holds_it"] else 0.0
        out.append(
            f"**At fast discharge the published electrolyte decides the verdict on the "
            f"tortuosity.** At {fast['rate']:g}C the cell with all four measured parameters "
            f"gives {_n(s['all_four'], 3)} A.h against the published {_n(fast['published'], 3)}, "
            "mostly because the measured diffusivity is larger. With the measured electrolyte "
            f"in place the Bruggeman interval runs from {_n(b['low'], 3)} to {_n(b['high'], 3)} "
            f"A.h and misses by {_n(gap_new, 3)} A.h, where it missed the published curve by "
            f"{_n(gap_old, 3)}. The exponent is still too large, but most of the earlier miss "
            "came from the published electrolyte, not from the tortuosity.")
        out.append("")
    cold = [r for r in rows if r["temperature"] != ROOM]
    if cold:
        worst = min(cold, key=lambda r: r["landesfeind"]["all_four"])
        s = worst["landesfeind"]
        out.append(
            "**In the cold the parameters that are not predicted decide the capacity.** At "
            f"{worst['temperature']:.0f} K and {worst['rate']:g}C the measured conductivity alone "
            f"gives {_n(s['conductivity_only'], 3)} A.h and all four measured parameters give "
            f"{_n(s['all_four'], 3)}, against the published {_n(worst['published'], 3)}. The "
            "difference between the two measured capacities comes from the diffusivity, the "
            "transference number and the thermodynamic factor, for which this program has no "
            "prediction.")
        out.append("")
    return out


def report(results: str = RESULTS) -> str:
    """Write the table from the stored result. Returns the path of the table."""
    with open(os.path.join(results, "cellmodel.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    pub = res["published"]
    params = {p["name"]: p for p in res["parameters"]["parameters"]}
    lines: List[str] = []
    w = lines.append
    w("# Item MS9: a parameter set for a cell model, with its deviations")
    w("")
    w(f"The cell model is PyBaMM's {res['model']} with the growth of the passivating layer, on "
      f"the published parameter set of a graphite / nickel-manganese-cobalt oxide cell "
      f"(`{res['base']}`). One cycle is a discharge at the one-hour rate to 2.5 V, a charge back "
      f"to 4.2 V and a hold. At the published values the cell gives "
      f"{_n(pub['first'], 3)} A.h at the first of {res['cycles']} cycles and "
      f"{_n(pub['final'], 3)} A.h at the last, losing "
      f"{_n(100 * (1 - pub['fade']), 3)} % of its capacity.")
    w("")
    w("## What this program predicts, and how sure it is")
    w("")
    w("| parameter | what the program gives | 95 % interval | deviation calibrated? | from |")
    w("|---|---|---|---|---|")
    for name, p in params.items():
        lo, hi = p["interval_95"]
        value = f"{_n(p['value'], 4)} {p['units']}"
        cal = ("not predicted" if not p["predicted"] else
               f"yes, z rms {_n(p['calibration'], 2)}" if p["calibrated"] else
               f"no, z rms {_n(p['calibration'], 2)}")
        short = res["channels"].get(name, {}).get("label", "")
        w(f"| {name} | {value} | {_n(lo, 4)} to {_n(hi, 4)} | {cal} | {short} |")
    w("")
    w("The kind of each number matters: a diffusion coefficient and a Bruggeman exponent replace "
      "the published value, a conductivity and an exchange current multiply it, and a voltage is "
      "added to it. A conductivity and an exchange-current density are functions of "
      "concentration and of state of charge in a cell model, so a prediction of one condition "
      "can only be a factor on the whole function.")
    w("")
    w("## The capacity-against-cycle curve under each deviation")
    w("")
    w("Each parameter is drawn from its own deviation while the others keep their published "
      f"values, and the cell model runs once per draw ({res['draws']} draws). The interval is "
      "that of the capacity of the last cycle.")
    w("")
    w("| parameter | interval of the last cycle, A.h | holds the published curve? | runs the "
      "cell model refused | share of the variance | elasticity |")
    w("|---|---|---|---:|---:|---:|")
    order = sorted(res["per_parameter"],
                   key=lambda k: -(res["variance_shares"][k]["share"]
                                   if np.isfinite(res["variance_shares"][k]["share"]) else -1))
    for name in order:
        d = res["per_parameter"][name]
        lo, hi = d["interval_of_the_last_cycle"]
        held = d["holds_the_published_curve"]
        verdict = ("yes" if held["all_inside"] else
                   f"no, {held['inside']} of {held['cycles']} cycles")
        w(f"| {name} | {_n(lo, 3)} to {_n(hi, 3)} | {verdict} | "
          f"{_n(100 * d['refused'], 0)} % | "
          f"{_n(100 * res['variance_shares'][name]['share'], 1)} % | "
          f"{_n(d['elasticity']['elasticity'], 3, sign=True)} |")
    w("")
    w("The two last columns answer two different questions. The share of the variance is what "
      "this program's own deviations do: it ranks the parameters by how much better they would "
      "have to be known. The elasticity is the relative change of the capacity for a relative "
      "change of one tenth in the parameter, around the published value, which does not depend "
      "on this program at all and is the reading the cell-model literature states.")
    w("")
    w("The two orders are not the same, and neither is the one the item expected, which is the "
      "solid diffusion coefficient and the exchange current ahead of the voltage curve. The "
      "open-circuit voltage comes first in both. The reason is the same in both columns and it "
      "is a matter of scale: a tenth of a voltage is 0.38 V, which empties a cell, while a tenth "
      "of a diffusion coefficient changes almost nothing, and the deviation this program has on "
      "a voltage is about one volt, where a measured open-circuit curve is known to millivolts. "
      "The literature's ranking is for a cell whose voltage curve has been measured. Between the "
      "two transport parameters the expected order does come out: the diffusion coefficient is "
      "ahead of the exchange current in both columns.")
    w("")
    w("## How accurate the barrier would have to be")
    w("")
    w(f"The deviation of the predicted barrier is made smaller by each factor below and the cell "
      f"model is run again, {TOL_DRAWS} draws over {TOL_CYCLES} cycles. An interval is worth "
      f"using only if it is both narrow -- inside {_n(res['tolerance_width'], 3)} A.h, which is "
      f"{100 * TOL_SHARE:.0f} % of the published capacity -- and still holds the capacity the "
      f"published value gives.")
    w("")
    w("| the barrier known to, eV | interval of the capacity, A.h | narrow? | holds the "
      "published capacity? |")
    w("|---|---|---|---|")
    for row in res["tolerance"]:
        w(f"| {_n(row['barrier_sd'], 3)} | {_n(row.get('low'), 3)} to {_n(row.get('high'), 3)} | "
          f"{'yes' if row.get('narrow') else 'no'} | "
          f"{'yes' if row.get('holds') else 'no'} |")
    w("")
    w("No row is both. A narrower deviation gives a narrower interval, but the interval closes "
      "on the capacity the median prediction gives, and that prediction is wrong: the barrier is "
      "over-predicted, so the diffusion coefficient comes out several orders of magnitude too "
      "small. The error of this channel is a bias and not a spread, and a tolerance rule cannot "
      "repair a bias. What this parameter needs is not a smaller deviation but a better "
      "prediction, which for a cathode of this kind means the barrier from the fine level "
      "itself.")
    w("")
    stress = res.get("stress")
    if stress:
        lines.extend(stress_lines(stress, params))
    tolerance = read_tolerance(results)
    if tolerance:
        lines.extend(tolerance_lines(tolerance))
    w("## What the gate asked, and what came out")
    w("")
    failed = [n for n, p in params.items()
              if p["calibrated"] and not res["per_parameter"][n]
              ["holds_the_published_curve"]["all_inside"]]
    held = [n for n in params if res["per_parameter"][n]["holds_the_published_curve"]
            ["all_inside"]]
    w(f"**The first part is not met.** The interval holds the published curve for {len(held)} of "
      f"{len(params)} parameters, and the gate asks for every parameter whose deviation means "
      f"what it says. " +
      (f"It fails for {', '.join(failed)}, whose deviation is calibrated "
       f"(z rms {_n(params[failed[0]]['calibration'], 2)}). The interval misses by "
       f"{_n(res['per_parameter'][failed[0]]['holds_the_published_curve']['largest_gap'], 4)} "
       "A.h, because the structures this channel was fitted on are more tortuous than the "
       "exponent of 1.5 the authors of the cell fitted; the channel says so itself, since its "
       "structures are generated and it is marked as outside what the model was fitted on."
       if failed else "No calibrated parameter fails it."))
    w("")
    w("**The second part is not met either.** The ranking does not put the solid diffusion "
      "coefficient and the exchange current ahead of the voltage curve, under either reading, "
      "for the reason given above: this program's deviation on a voltage is about a volt.")
    w("")
    w("**What carries over is the interface.** The writer, the sampler and the report the item "
      "describes are built: the deviation is carried through the conversions into the units a "
      "cell model reads, and the report "
      "says which parameter's deviation moves the capacity most. The tolerance rule says the "
      "useful thing: for the diffusion coefficient no deviation is small enough, because the "
      "error is a bias; and the electrolyte conductivity, the one channel predicted for the "
      "cell's own material under its own conditions, agrees with the published parameter set "
      "to 1.3 %.")
    w("")
    w("## What every channel measured")
    w("")
    for name, info in res["channels"].items():
        source = params[name]["source"]
        w(f"**{name}** ({info.get('label', '')}, {_n(info.get('seconds'), 1)} s). "
          f"{source[0].upper()}{source[1:]}.")
        if "z_rms" in info:
            mae = info.get("mae", info.get("mae_log"))
            w(f"Mean absolute error {_n(mae, 3)} {info.get('error_units', '')} against "
              f"{_n(info.get('mae_of_the_mean'), 3)} for predicting the mean; the deviation has "
              f"z rms {_n(info.get('z_rms'), 2)} and holds "
              f"{_n(100 * info.get('coverage_2sd', float('nan')), 0)} % of the activities inside "
              f"two standard deviations, where 95 % is expected.")
        elif "training" in info:
            tr = info["training"]
            s = info["structure"]
            w(f"The forest predicts the {tr['hops']} training barriers to "
              f"{_n(tr['mae'], 3)} eV with whole structures held out, its deviation has z rms "
              f"{_n(tr['factor'], 2)} and holds {_n(100 * tr['inside_2sd'], 0)} % of them inside "
              f"two standard deviations. {s['formula']} ({s['id']}) has {s['hops']} lithium "
              f"hops, {s['distinct']} of them distinct, and its network percolates in "
              f"{s['dimension']} of three directions, which is the two-dimensional diffusion a "
              f"layered oxide is known for.")
        if params[name]["note"]:
            w(f"*{params[name]['note'][0].upper()}{params[name]['note'][1:]}.*")
        w("")
    path = os.path.join(results, "table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[1])
    ap.add_argument("--report", action="store_true",
                    help="write the table from the stored result and stop")
    ap.add_argument("--draws", type=int, default=DRAWS,
                    help="how many times each parameter is drawn")
    ap.add_argument("--cycles", type=int, default=CYCLES, help="how many cycles per run")
    ap.add_argument("--results", default=RESULTS, help="where the result is written")
    ap.add_argument("--stage", action="append", choices=STAGES,
                    help="run only these stages; every stage writes its own file and a stage "
                         "whose file is there is not run again, so the benchmark can be "
                         "continued after a failure (default: all of them)")
    ap.add_argument("--parameter", default=None,
                    help="with --stage sample: draw only this parameter")
    ap.add_argument("--force", action="store_true",
                    help="run the stages again even where their files are there")
    args = ap.parse_args(argv)
    if not args.report:
        done = run(args.results, args.draws, args.cycles, tuple(args.stage or STAGES),
                   args.parameter, args.force)
        if done is None:
            return 0
    log(f"the table: {report(args.results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
