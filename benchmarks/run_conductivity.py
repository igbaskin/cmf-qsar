#!/usr/bin/env python
"""
From DFT barriers to the conductivity of a conductor, with the error carried through (the
second half of plan item MS8).

The first half (``run_conductors.py``) stops at the percolation barrier, an activation energy.
This script goes on to the conductivity at 298 K, which is what the gate of the item asks for.

The chain
---------
1. The forest on the invariant field, fitted on the 1,652 DFT barriers of nebDFT2k, predicts
   every hop of the thirteen disordered conductors with a calibrated standard deviation, as in
   the first half (the same model, the same error model, the same way of drawing).
2. **The hop simulation.** The hops of a structure form a periodic network of lithium sites.
   The long-time diffusion coefficient of one ion on that network is computed exactly
   (``cmfqsar.conduction.network_diffusion``), which is the limit a kinetic Monte Carlo
   simulation approaches; the tests check the two against each other. The two directions of a
   hop get the mean of their barriers, so every site is equally likely, as a network without
   site energies says; the polycrystal value is a third of the trace of the tensor.
3. **The prefactor.** A hop of barrier E has the rate nu exp(-E/kT). The attempt frequency nu
   is drawn between 10^12.5 and 10^13.5 Hz (uniform in its logarithm), the range of a lithium
   vibration; it is the one number of the chain that is assumed rather than computed.
4. **The carriers.** The lithium sites of the structure file and their mean occupancy c give
   the conductivity of a lattice gas, sigma = q^2 (N/V) c (1 - c) D / kT
   (``cmfqsar.conduction.lattice_gas_conductivity``).
5. The barriers and the attempt frequency are drawn together; the 95 % interval of the
   conductivity is read from the draws and set beside the measured conductivity.

**The gate, as the plan states it:** the conductivity with an interval, against the measured one
of at least three of the disordered conductors. Read with it, fixed before the measurement: the
interval holds the measured value or does not, the point prediction is reported in decades from
it, and the same network with the bond-valence barriers is the reference that needs no model.

**What the deviation of MS1 does to the final number** is reported as in the first half: the
standard deviations of the barriers are multiplied by 0.5, 0.25, 0.1 and 0.05, and the width of
the interval of log10(sigma) is read for each.

Usage
-----
    python benchmarks/run_nebdft.py          # first: the kernels and rows of the hops
    python benchmarks/run_conductivity.py    # this benchmark, then the table
    python benchmarks/run_conductivity.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_conductors as RC  # noqa: E402
import run_nebdft as N  # noqa: E402

from cmfqsar import conduction as C  # noqa: E402

RESULTS = os.path.join(HERE, "results", "conductivity")
TEMPERATURE = 298.15
NU = (10 ** 12.5, 10 ** 13.5)
NU_POINT = 1e13
DRAWS = 400
SCALE_DRAWS = 200
SCALES = (0.5, 0.25, 0.1, 0.05)
SEED = 0
#: measured conductivities near room temperature, in S/cm, with how far the source was read.
MEASURED = {
    "cod_7215190": {
        "sigma": 1.2e-2, "temperature": "room temperature", "what": "total",
        "sample": "not stated in the abstract",
        "source": "N. Kamaya et al., Nat. Mater. 10, 682 (2011), doi:10.1038/nmat3066",
        "checked": "abstract read on the publisher's page"},
    "cod_1545083": {
        "sigma": 8.4e-4, "temperature": "297 K", "what": "total (bulk and grain boundary "
        "not separated)", "sample": "sintered pellet of Li6.5La3Zr1.5Ta0.5O12, tetragonal and "
        "cubic phases together",
        "source": "N. Hamao, K. Kataoka, N. Kijima, J. Akimoto, J. Ceram. Soc. Jpn. 124, 678 "
                  "(2016), doi:10.2109/jcersj2.16019; the same composition as a single crystal: "
                  "1.27e-3 S/cm (cited in K. Kataoka, J. Akimoto, J. Ceram. Soc. Jpn. 127, 521 "
                  "(2019))",
        "checked": "full text read (open access PDF)"},
    "cod_1000437": {
        "sigma": 1.0e-3, "temperature": "room temperature", "what": "not stated in the abstract",
        "sample": "the x of about 0.1 member of Li3xLa2/3-xTiO3, which is this composition "
                  "(La0.557Li0.33TiO3, x = 0.11)",
        "source": "S. Stramare, V. Thangadurai, W. Weppner, Chem. Mater. 15, 3974 (2003), "
                  "doi:10.1021/cm0300516",
        "checked": "abstract read (through OpenAlex; the publisher's page shows a bot check)"},
}


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- the networks

class Network:
    """The lithium network of one structure: sites, hops, occupancy and cell volume."""

    def __init__(self, hc, k: int):
        from cmfqsar.hops import mobile_occupancy

        atoms = hc.structures[k]
        occ = mobile_occupancy(atoms)
        sites = np.flatnonzero(occ > 0)
        index = {int(s): i for i, s in enumerate(sites)}
        self.rows = hc.of_structure(k)
        self.n_sites = int(len(sites))
        self.occupancy = float(occ[sites].mean()) if len(sites) else 0.0
        self.volume = float(atoms.get_volume())
        self.edges = [(index[hc.hops[r].a], index[hc.hops[r].b],
                       tuple(int(x) for x in hc.hops[r].offset)) for r in self.rows]
        self.vectors = np.array([np.asarray(hc.hops[r].vector, float) for r in self.rows]
                                ).reshape(-1, 3)
        # the bond-valence landscape: an energy on every site and on every saddle
        self.site_energy = np.full(self.n_sites, np.nan)
        for r, (a, _b, _o) in zip(self.rows, self.edges):
            if np.isfinite(hc.hops[r].site_a):
                self.site_energy[a] = hc.hops[r].site_a
        self.saddle = np.array([hc.hops[r].saddle for r in self.rows], float)

    def diffusion(self, values: Dict[int, float]) -> float:
        """A third of the trace of D, in A^2/s per Hz of attempt frequency."""
        if not self.rows:
            return 0.0
        b = np.array([values.get(r, np.nan) for r in self.rows], float)
        rates = C.rates_from_barriers(C.pair_barriers(self.edges, b), 1.0, TEMPERATURE)
        D = C.network_diffusion(self.n_sites, self.edges, self.vectors, rates)
        return float(np.trace(D)) / 3.0

    def bond_valence_diffusion(self) -> Optional[float]:
        """A third of the trace of D on the bond-valence landscape, per Hz; None without one."""
        if (not self.rows or not np.isfinite(self.saddle).all()
                or not np.isfinite(self.site_energy).any()):
            return None
        rates, pi = C.landscape_rates(self.edges, self.site_energy, self.saddle, 1.0,
                                      TEMPERATURE)
        D = C.network_diffusion(self.n_sites, self.edges, self.vectors, rates,
                                site_weights=pi)
        return float(np.trace(D)) / 3.0

    def sigma(self, d_per_hz: float, nu: float) -> float:
        return C.lattice_gas_conductivity(d_per_hz * nu, self.n_sites, self.occupancy,
                                          self.volume, TEMPERATURE)


def _log10(v: float) -> float:
    return float(np.log10(v)) if v > 0 else float("-inf")


def draw_sigma(hc, nets: Dict[int, Network], reps, pred, sd, shared, draws: int,
               seed: int = SEED, nu_fixed: Optional[float] = None) -> Dict[int, np.ndarray]:
    """log10 sigma of every structure for `draws` joint draws of barriers and attempt frequency."""
    rng = np.random.default_rng(seed)
    of = np.array([hc.hops[r].structure for r in reps])
    out = {k: np.full(draws, np.nan) for k in nets}
    for d in range(draws):
        common = rng.normal(size=len(hc.structures))
        own = rng.normal(size=len(reps))
        z = np.sqrt(shared) * common[of] + np.sqrt(1.0 - shared) * own
        values = RC.spread_values(hc, reps, np.maximum(pred + sd * z, 0.0))
        nu = nu_fixed if nu_fixed is not None else 10 ** rng.uniform(*np.log10(NU))
        for k, net in nets.items():
            out[k][d] = _log10(net.sigma(net.diffusion(values), nu))
    return out


def _interval(v: np.ndarray) -> List[float]:
    """2.5 %, 50 % and 97.5 % points of the draws; a draw with no path counts as minus
    infinity, so a point that falls among those draws is minus infinity too."""
    s = np.sort(np.where(np.isfinite(v), v, -np.inf))
    n = len(s)
    low = [float(s[int(np.floor(q * (n - 1)))]) for q in (0.025, 0.5)]
    return low + [float(s[int(np.ceil(0.975 * (n - 1)))])]


def _width(drawn: Dict[int, np.ndarray]) -> float:
    w = [np.percentile(v, 97.5) - np.percentile(v, 2.5) for v in drawn.values()
         if np.isfinite(v).all()]
    return float(np.mean(w)) if w else float("nan")


# ------------------------------------------------------------------------------ the run

def run(results: str = RESULTS, draws: int = DRAWS, cache: str = N.CACHE) -> dict:
    import run_hops as H
    from sklearn.ensemble import RandomForestRegressor

    t0 = time.time()
    hs, _doc, _, _ = N.load(N.PREPARED, 0)
    t = hs.targets()
    y = np.asarray(t.y, float)
    groups = np.asarray(t.molecule, int)
    _, rows = N.descriptions(hs, 8, cache, "")
    X = rows["invariant"]
    errors = RC.error_model(X, y, groups, RC.FOLDS)
    errors.pop("predicted", None)
    log(f"  the forest on folds: mean absolute error {errors['mae']:.3f} eV, factor "
        f"{errors['factor']:.2f}, shared within a structure {errors['shared']:.2f}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hc, reps, Xc, _unseen = RC.conductor_rows(H.fieldset_for(hs, "file"))
    forest = RandomForestRegressor(n_estimators=N.FOREST_TREES, n_jobs=-1,
                                   random_state=SEED).fit(X, y)
    trees = np.array([e.predict(Xc) for e in forest.estimators_])
    pred = trees.mean(axis=0)
    sd = np.maximum(trees.std(axis=0), 1e-6) * errors["factor"]
    nets = {k: Network(hc, k) for k in range(len(hc.structures))}
    point = RC.spread_values(hc, reps, pred)
    drawn = draw_sigma(hc, nets, reps, pred, sd, errors["shared"], draws)
    nu_only = draw_sigma(hc, nets, reps, pred, sd * 0.0, errors["shared"], SCALE_DRAWS,
                         seed=SEED + 1)
    barriers_only = draw_sigma(hc, nets, reps, pred, sd, errors["shared"], SCALE_DRAWS,
                               seed=SEED + 2, nu_fixed=NU_POINT)
    names = {f"cod_{cid}": (name, what) for cid, name, what in H.COD}
    structures = []
    for k, label in enumerate(hc.names):
        net = nets[k]
        d_point = net.diffusion(point)
        entry = {"structure": label, "compound": names[label][0], "disorder": names[label][1],
                 "sites": net.n_sites, "occupancy": net.occupancy, "volume": net.volume,
                 "hops": len(net.rows), "measured": MEASURED.get(label),
                 "diffusion_point": d_point * NU_POINT,
                 "log10_sigma_point": _log10(net.sigma(d_point, NU_POINT)),
                 "log10_sigma_interval": _interval(drawn[k])}
        d_bv = net.bond_valence_diffusion()
        if d_bv is not None:
            entry["log10_sigma_bond_valence"] = _log10(net.sigma(d_bv, NU_POINT))
        structures.append(entry)
    sensitivity = [{"scale": 1.0, "barrier_sd": float(np.mean(sd)), "width": _width(drawn)}]
    for s in SCALES:
        dd = draw_sigma(hc, nets, reps, pred, sd * s, errors["shared"], SCALE_DRAWS,
                        seed=SEED + 3)
        sensitivity.append({"scale": float(s), "barrier_sd": float(np.mean(sd) * s),
                            "width": _width(dd)})
        log(f"  barrier sd {np.mean(sd) * s:.3f} eV: interval of log10 sigma "
            f"{sensitivity[-1]['width']:.1f} decades wide")
    out = {"temperature": TEMPERATURE, "attempt_frequency": list(NU), "draws": draws,
           "training": {"hops": int(len(y)), **errors},
           "structures": structures, "sensitivity": sensitivity,
           "width_attempt_frequency_only": _width(nu_only),
           "width_barriers_only": _width(barriers_only),
           "comparison": compare(structures), "seconds": round(time.time() - t0, 1)}
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "conductivity.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def compare(structures: Sequence[dict]) -> dict:
    """The measured structures: inside the interval or not, and the distance in decades."""
    rows = [s for s in structures if s.get("measured")]
    logs = np.array([np.log10(s["measured"]["sigma"]) for s in rows])
    out = {"structures": len(rows), "inside": 0, "rows": []}
    for i, s in enumerate(rows):
        m = float(logs[i])
        lo, mid, hi = s["log10_sigma_interval"]
        rest = np.delete(logs, i)
        r = {"structure": s["structure"], "measured": m, "point": s["log10_sigma_point"],
             "interval": [lo, hi], "inside": bool(lo <= m <= hi),
             "point_minus_measured": s["log10_sigma_point"] - m,
             "others_mean_minus_measured": float(rest.mean() - m) if rest.size else None}
        if "log10_sigma_bond_valence" in s:
            r["bond_valence_minus_measured"] = s["log10_sigma_bond_valence"] - m
        out["inside"] += int(r["inside"])
        out["rows"].append(r)

    def mean_abs(key):
        v = [abs(r[key]) for r in out["rows"] if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")

    out["mean_decades_off"] = {"forest": mean_abs("point_minus_measured"),
                               "bond_valence": mean_abs("bond_valence_minus_measured"),
                               "mean_of_the_others": mean_abs("others_mean_minus_measured")}
    out["gate_met"] = bool(len(rows) >= 3 and all(np.isfinite(r["interval"][1])
                                                   for r in out["rows"]))
    return out


# ---------------------------------------------------------------------------- the table

def _fmt(v: Optional[float]) -> str:
    if v is None or not np.isfinite(v):
        return "no path" if v is not None and v < 0 else "--"
    return f"{v:+.1f}"


def report(results: str = RESULTS) -> str:
    path = os.path.join(results, "conductivity.json")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    lines = ["# Item MS8, second half: from DFT barriers to the conductivity of a conductor", "",
             f"Conductivity at {d['temperature']:.0f} K as log10 of S/cm. The forest on the "
             "invariant field predicts the barriers (fitted on the DFT barriers of nebDFT2k); "
             "the hop network of each structure gives the diffusion coefficient of one ion; the "
             f"attempt frequency is drawn from 10^12.5 to 10^13.5 Hz; the lattice gas of the "
             f"lithium sites gives the conductivity. {d['draws']} joint draws of the barriers "
             "and the attempt frequency give the interval. Generated by "
             "`benchmarks/run_conductivity.py`.", "",
             "| Structure | Compound | Lithium sites | Mean occupancy | log10 sigma, forest | "
             "95 % interval | log10 sigma, bond-valence landscape | Measured log10 sigma |",
             "|---|---|---:|---:|---:|---|---:|---:|"]
    for s in d["structures"]:
        lo, _mid, hi = s["log10_sigma_interval"]
        meas = s.get("measured")
        lines.append(f"| {s['structure']} | {s['compound']} | {s['sites']} | "
                     f"{s['occupancy']:.2f} | {_fmt(s['log10_sigma_point'])} | "
                     f"{_fmt(lo)} to {_fmt(hi)} | {_fmt(s.get('log10_sigma_bond_valence'))} | "
                     f"{np.log10(meas['sigma']):+.1f} |" if meas else
                     f"| {s['structure']} | {s['compound']} | {s['sites']} | "
                     f"{s['occupancy']:.2f} | {_fmt(s['log10_sigma_point'])} | "
                     f"{_fmt(lo)} to {_fmt(hi)} | {_fmt(s.get('log10_sigma_bond_valence'))} | "
                     "-- |")
    c = d["comparison"]
    lines += ["", "## Against the measured conductivities", "",
              "| Structure | Measured | Forest, point minus measured, decades | Inside the "
              "95 % interval | Bond-valence landscape minus measured | Mean of the other "
              "measured values minus this one |",
              "|---|---:|---:|---|---:|---:|"]
    for r in c["rows"]:
        lines.append(f"| {r['structure']} | {r['measured']:+.1f} | "
                     f"{_fmt(r['point_minus_measured'])} | {'yes' if r['inside'] else 'no'} | "
                     f"{_fmt(r.get('bond_valence_minus_measured'))} | "
                     f"{_fmt(r['others_mean_minus_measured'])} |")
    m = c["mean_decades_off"]
    lines += ["",
              f"The interval holds the measured value for {c['inside']} of {c['structures']} "
              f"structures. Mean distance in decades: forest {m["forest"]:.1f}, bond-valence "
              f"landscape {m["bond_valence"]:.1f}, the mean of the other measured values "
              f"{m['mean_of_the_others']:.1f}.", "",
              "## What the deviation of a barrier does to the conductivity", "",
              "| Standard deviation of one barrier, eV | Width of the 95 % interval of log10 "
              "sigma, decades |", "|---:|---:|"]
    for s in d["sensitivity"]:
        lines.append(f"| {s['barrier_sd']:.3f} | {s['width']:.1f} |")
    lines += ["",
              f"With the barriers fixed at their prediction, the attempt frequency alone gives "
              f"an interval {d['width_attempt_frequency_only']:.1f} decades wide; with the "
              f"attempt frequency fixed at 10^13 Hz the barriers alone give "
              f"{d['width_barriers_only']:.1f}.", "",
              "## The measured values and how far their sources were read", ""]
    for label, mm in MEASURED.items():
        lines.append(f"* {label}: {mm['sigma']:.2g} S/cm, {mm['temperature']}, {mm['what']}; "
                     f"{mm['sample']}. {mm['source']}. Checked: {mm['checked']}.")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", action="store_true", help="rebuild table.md from the results")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    if not args.report:
        run(draws=args.draws)
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
