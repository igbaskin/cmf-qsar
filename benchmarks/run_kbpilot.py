"""
Plan item BT11, a free pilot: does a thermodynamic factor show in the boxes that item BT9 stored?

Item BT11 needs the thermodynamic factor of the salt, that is how far the salt is from an ideal
solution. The plan takes it from the Kirkwood-Buff integrals of the simulated boxes and warns
that this needs boxes several times larger than those of BT5 and long runs. Before that cost is
paid, this pilot asks whether the boxes of item BT9 already show a usable signal. They hold the
centre of every molecule once per picosecond for 4 ns (``centres.npz``) and nothing else, and
each has 1,000 solvent molecules.

The route is the number fluctuation. The molecules in a cube that is smaller than the box are
counted in every frame. The covariance of the counts of two species, divided by the volume of the
cube, is the same quantity as the Kirkwood-Buff matrix of the species, and a ratio of such
quantities gives the factor without any radial distribution function.

The definition, with the units of the cell model
------------------------------------------------
Let the solvents be 1 to k and let the salt be one species with the number N_s = (N_+ + N_-)/2
and the density rho_s, which is the concentration of the salt. Let B be the covariance of the
counts of these species in the cube divided by its volume. In the grand canonical ensemble,
d rho_i = (B_ij / kT) d mu_j. At constant pressure the Gibbs-Duhem relation is
rho_s d mu_s + sum_j rho_j d mu_j = 0, with mu_s = mu_+ + mu_-. The solvents keep their
number ratios, so d rho_j = rho_j eps for one eps. These equations are linear in the unknowns
d mu_s, d mu_j and eps. The factor is

    Gamma = d ln(a_+ a_-)^(1/2) / d ln rho_s = rho_s d mu_s / (2 kT d rho_s),

which is 1 for an ideal solution and 1/2 if every cation is bound to one anion (then the pairs
are the neutral species). The molality, the weight ratio and the number ratio of the solvents
differ a little in what they keep fixed; this pilot keeps the number ratio.

What the pilot does not do
--------------------------
The cube is open and the box is closed, so the fluctuations in a cube that is a large share of
the box are too small by a factor near one minus that share. The pilot reports the cubes of
one quarter, one third and one half of the side and does not correct them. Molecules are points
at their centres, so the solvent structure is coarse. The run is 4 ns, so the counts of the
ions, of which there are 9 to 70 pairs, are correlated over long times. The factor in the
measured function that the cell model reads may include a term for the motion of the solvent;
the comparison with it is therefore a comparison of trends and not of the level.

Stages
------
    python benchmarks/run_kbpilot.py --stage series    # the six concentrations, two starts each
    python benchmarks/run_kbpilot.py --stage all62     # the 62 formulations
    python benchmarks/run_kbpilot.py --stage charges   # the older boxes with other charge factors
    python benchmarks/run_kbpilot.py --stage report    # the table
    python benchmarks/run_kbpilot.py                   # all of them
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

import run_diffusion as RD                                                   # noqa: E402

RESULTS = os.path.join(HERE, "results", "kbpilot")

#: the cubes, as shares of the side of the box
CUBES = (0.25, 1.0 / 3.0, 0.5)
#: every STRIDE-th frame is used (the frames are one picosecond apart)
STRIDE = 5
#: random cubes per frame
ORIGINS = 40
BLOCKS = 8
SEED = 0
CATION, ANION = "LI", "ANI"


# ------------------------------------------------------------------------------------ the data


def load_box(path: str):
    """The wrapped centres (frames, molecules, 3), the species name of every molecule and the
    side of the box, in angstrom."""
    with np.load(path) as z:
        centres = np.asarray(z["centres"], float)
        names = [str(n) for n in z["names"]]
        side = float(z["side"])
    return np.mod(centres, side), names, side


def species_of(names: Sequence[str]):
    """The index of the species of every molecule and the species names: the cation, the anion,
    then the solvents in sorted order."""
    solvents = sorted({n for n in names if n not in (CATION, ANION)})
    order = [CATION, ANION] + solvents
    index = {n: i for i, n in enumerate(order)}
    return np.array([index[n] for n in names]), order


def cube_counts(centres: np.ndarray, species: np.ndarray, n_species: int, side: float,
                share: float, origins: int = ORIGINS, stride: int = STRIDE,
                seed: int = SEED) -> np.ndarray:
    """The numbers of every species in random cubes of side `share` times the box, for every
    `stride`-th frame. Shape (frames, origins, species). The cubes wrap around the box."""
    rng = np.random.default_rng(seed)
    frames = centres[::stride]
    onehot = np.zeros((len(species), n_species))
    onehot[np.arange(len(species)), species] = 1.0
    ell = share * side
    out = np.zeros((len(frames), origins, n_species))
    for f, x in enumerate(frames):
        o = rng.uniform(0.0, side, (origins, 3))
        d = np.mod(x[None, :, :] - o[:, None, :], side)
        inside = np.all(d < ell, axis=2).astype(float)
        out[f] = inside @ onehot
    return out


# ---------------------------------------------------------------------------------- the factor


def thermodynamic_factor(counts: np.ndarray, volume: float) -> float:
    """The factor Gamma from the counts of the species in cubes of the given volume.

    `counts` has shape (samples, species) with the cation first, the anion second and the
    solvents after them. See the docstring of this file for the definition.
    """
    counts = np.asarray(counts, float)
    n_s = 0.5 * (counts[:, 0] + counts[:, 1])
    z = np.column_stack([n_s, counts[:, 2:]])
    rho = z.mean(axis=0) / volume
    B = np.atleast_2d(np.cov(z, rowvar=False, bias=True)) / volume
    k = z.shape[1] - 1
    # unknowns: d mu_s, d mu_1 .. d mu_k (in kT), eps; the right side fixes d rho_s = 1
    A = np.zeros((k + 2, k + 2))
    b = np.zeros(k + 2)
    A[0, :k + 1] = B[0]
    b[0] = 1.0
    for j in range(k):
        A[1 + j, :k + 1] = B[1 + j]
        A[1 + j, k + 1] = -rho[1 + j]
    A[k + 1, :k + 1] = rho
    x = np.linalg.solve(A, b)
    return float(0.5 * rho[0] * x[0])


def block_error(counts: np.ndarray, volume: float, blocks: int = BLOCKS) -> Dict[str, float]:
    """The factor of the whole run and its standard error from `blocks` blocks of frames.

    `counts` has shape (frames, origins, species).
    """
    whole = thermodynamic_factor(counts.reshape(-1, counts.shape[-1]), volume)
    parts = np.array_split(np.arange(counts.shape[0]), blocks)
    vals = [thermodynamic_factor(counts[p].reshape(-1, counts.shape[-1]), volume)
            for p in parts]
    return {"gamma": whole,
            "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1
            else float("nan")}


def box_factors(path: str, shares: Sequence[float] = CUBES) -> Dict[str, object]:
    centres, names, side = load_box(path)
    species, order = species_of(names)
    out = {"side": side, "species": order, "n": [int(np.sum(species == i))
                                                  for i in range(len(order))],
           "cubes": {}}
    for share in shares:
        counts = cube_counts(centres, species, len(order), side, share)
        out["cubes"][f"{share:.3f}"] = block_error(counts, (share * side) ** 3)
    return out


# ---------------------------------------------------------------------------------- the stages


def series_stage() -> Dict[str, object]:
    out: Dict[str, object] = {"boxes": {}}
    for key, names in RD.CONCENTRATION_RUNS.items():
        for name in names:
            path = os.path.join(RD.run_dir(name), "centres.npz")
            if not os.path.isfile(path):
                continue
            t0 = time.time()
            res = box_factors(path)
            with open(os.path.join(RD.RESULTS, f"run_{name}.json"), encoding="utf-8") as fh:
                run = json.load(fh)
            res["run"] = name
            res["molarity"] = float(run["molarity"])
            res["measured_factor"] = float(run["measured"]["thermodynamic_factor"])
            out["boxes"][name] = res
            print(f"  {name}: " + ", ".join(f"{k} {v['gamma']:.2f}"
                                             for k, v in res["cubes"].items())
                  + f" (measured {res['measured_factor']:.2f}, {time.time() - t0:.0f} s)",
                  flush=True)
    return out


def all62_stage() -> Dict[str, object]:
    out: Dict[str, object] = {"boxes": {}}
    for fid, names in RD.ALL_FORMULATION_RUNS.items():
        for name in names:
            path = os.path.join(RD.run_dir(name), "centres.npz")
            if not os.path.isfile(path):
                print(f"  {name}: no stored centres", flush=True)
                continue
            res = box_factors(path)
            res["id"] = fid
            res["run"] = name
            out["boxes"][name] = res
            print(f"  {name}: " + ", ".join(f"{k} {v['gamma']:.2f}"
                                             for k, v in res["cubes"].items()), flush=True)
    return out


def charges_stage() -> Dict[str, object]:
    """The older boxes of item BT9 at the reference electrolyte: factors of 1, 0.8 and 0.7 on the
    charges of the ions, and several sizes and starts at 0.8."""
    out: Dict[str, object] = {"boxes": {}}
    for name, run in RD.RUNS.items():
        if name.startswith(("form_", "conc_")) or "settings" not in run:
            continue
        path = os.path.join(RD.run_dir(name), "centres.npz")
        if not os.path.isfile(path):
            continue
        res = box_factors(path)
        res["run"] = name
        res["ion_charge_scale"] = float(run["settings"].get("ion_charge_scale", 1.0))
        res["n_solvent"] = int(run["settings"].get("n_solvent", 250))
        out["boxes"][name] = res
        print(f"  {name}: scale {res['ion_charge_scale']:g}, {res['n_solvent']} solvent molecules: "
              + ", ".join(f"{k} {v['gamma']:.2f}" for k, v in res["cubes"].items()), flush=True)
    return out


def save(name: str, payload: Dict[str, object], base: str = RESULTS) -> None:
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


def _read(base: str, name: str) -> Dict[str, object]:
    with open(os.path.join(base, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _spearman(a, b):
    return RD.spearman(a, b)


def write_table(results: str = RESULTS) -> str:
    series = _read(results, "series")
    out: List[str] = ["# Item BT11, a pilot: the thermodynamic factor from the stored boxes", ""]
    w = out.append
    w("The factor is read from the fluctuations of the numbers of molecules in cubes, in the boxes "
      "of item BT9 (1,000 solvent molecules, 4 ns, charges of the ions multiplied by 0.7). The "
      "definition and what the pilot leaves out are in the docstring of "
      "`benchmarks/run_kbpilot.py`. The cubes are 1/4, 1/3 and 1/2 of the side of the box; they "
      "are not corrected for the closed box. The error after +/- is the standard error over "
      f"{BLOCKS} blocks of frames.")
    w("")
    w("## The six concentrations of the series")
    w("")
    w("| Run | Molarity, mol/L | Measured factor | Cube 1/4 | Cube 1/3 | Cube 1/2 |")
    w("|---|---:|---:|---:|---:|---:|")
    boxes = sorted(series["boxes"].values(), key=lambda r: (r["molarity"], r["run"]))
    for r in boxes:
        cells = " | ".join(f"{v['gamma']:.2f} +/- {v['se']:.2f}" for v in r["cubes"].values())
        w(f"| {r['run']} | {r['molarity']:.2f} | {r['measured_factor']:.2f} | {cells} |")
    w("")
    mol = [r["molarity"] for r in boxes]
    meas = [r["measured_factor"] for r in boxes]
    for key in boxes[0]["cubes"]:
        g = [r["cubes"][key]["gamma"] for r in boxes]
        w(f"* Cube {float(key):.2f} of the side: the rank correlation with the measured factor "
          f"is {_spearman(meas, g):+.2f}, and the rank correlation with the molarity "
          f"{_spearman(mol, g):+.2f}.")
    w("")
    pathc = os.path.join(results, "charges.json")
    if os.path.isfile(pathc):
        c = _read(results, "charges")["boxes"]
        w("## The factor on the charges of the ions")
        w("")
        w("The older boxes of item BT9 at the reference electrolyte (1.11 mol/kg). The measured "
          "factor at this concentration is 2.37. The boxes of the series at 0.7 are in the first "
          "table.")
        w("")
        w("| Run | Factor on the charges | Solvent molecules | Cube 1/4 | Cube 1/3 | Cube 1/2 |")
        w("|---|---:|---:|---:|---:|---:|")
        for r in sorted(c.values(), key=lambda r: (-r["ion_charge_scale"], r["n_solvent"], r["run"])):
            cells = " | ".join(f"{v['gamma']:.2f} +/- {v['se']:.2f}" for v in r["cubes"].values())
            w(f"| {r['run']} | {r['ion_charge_scale']:g} | {r['n_solvent']} | {cells} |")
        w("")
    path62 = os.path.join(results, "all62.json")
    if os.path.isfile(path62):
        a = _read(results, "all62")["boxes"]
        w("## The 62 formulations")
        w("")
        w("The cube of 1/2 of the side. There is no measured factor for these electrolytes.")
        w("")
        w("| Formulation | Run | Factor, cube 1/2 | Standard error |")
        w("|---|---|---:|---:|")
        for r in sorted(a.values(), key=lambda r: (r["id"], r["run"])):
            v = r["cubes"][f"{CUBES[-1]:.3f}"]
            w(f"| {r['id']} | {r['run']} | {v['gamma']:.2f} | {v['se']:.2f} |")
        w("")
    path = os.path.join(results, "table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("series", "all62", "charges", "report", "all"), default="all")
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args(argv)
    stages = ("series", "all62", "charges", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "series":
            save("series", series_stage(), args.results)
        elif stage == "all62":
            save("all62", all62_stage(), args.results)
        elif stage == "charges":
            save("charges", charges_stage(), args.results)
        else:
            print("the table:", write_table(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
