"""
Plan item BT10, a free pilot: does the transference number show in the boxes that item BT9 stored?

The cell model reads the cation transference number counted against the motion of the solvent.
It includes the motion of ions together: a cation and an anion that move as a pair carry no
current. The ratio of the self-diffusion coefficients, D+ / (D+ + D-), ignores that and is the
first reference of the item. The label of the item is the number from the Onsager coefficients,
the correlations of the collective displacements. They converge slowly, because a box gives one
sample of a collective displacement. Before paying for long runs, this pilot reads them from the
centres that item BT9 stored (one frame per picosecond for 4 ns, 1,000 solvent molecules) and
asks how large the error is.

The definition
--------------
Let X_i(t) be the sum of the unwrapped positions of the molecules of species i (the cations or the
anions), and R_s(t) the centre of mass of the solvent molecules. The collective coordinate
relative to the solvent is Y_i = X_i - N_i R_s, with N_i the number of molecules of i. The
Onsager coefficient is L_ij = lim d <dY_i . dY_j> / dt over the time origins, divided by
6 V k T. For singly charged ions the conductivity is sigma = e^2 (L_++ + L_-- - 2 L_+-), and the
share of the current that the cations carry is

    t+ = (L_++ - L_+-) / (L_++ + L_-- - 2 L_+-)

(Wheeler and Newman, Gudla et al.). The common factor of the three coefficients cancels. The
three slopes come from the mean squared displacement of three single trajectories, Y_+, Y_- and
Y_+ - Y_-, with the polarisation identity for the cross term:
<dY_+ . d(Y_+ - Y_-)> = (<dY_+^2> + <d(Y_+ - Y_-)^2> - <dY_-^2>) / 2.
The conductivity is that of ``cmfqsar.transport.conductivity`` (the number of cations equals
that of anions, so the frame of the solvent does not change it), and the pilot checks that.

The window and the error
------------------------
The slopes are fitted in the default window of ``cmfqsar.transport.default_window`` (from a
fortieth to a tenth of the run), the same window as the conductivity of item BT9. The error is
the standard error over the five blocks of time origins of ``cmfqsar.transport.diffusion``,
and the scatter between the two starts of the concentration series is read next to it.

The reading of the pilot, fixed before the boxes were read
-----------------------------------------------------------
The number is usable at room temperature if one start of 4 ns scatters by no more than 0.05 in t+
(the root mean square difference of the two starts divided by the square root of two, over the
six concentrations). The tolerance stage of item MS9 tolerates a shift of -0.13 or wider at 298 K
for a loss of 5 % of the capacity, so 0.05 is about a third of that. The pilot makes no claim
for the cold cells, where the tolerated shift is 0.03 to 0.06.

Stages
------
    python benchmarks/run_onsager.py --stage series    # the six concentrations, both starts, and the 20 ns runs
    python benchmarks/run_onsager.py --stage all62     # the 62 formulations
    python benchmarks/run_onsager.py --stage report    # the table
    python benchmarks/run_onsager.py                   # all of them
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_diffusion as RD                                                   # noqa: E402
from cmfqsar import transport as T                                           # noqa: E402

RESULTS = os.path.join(HERE, "results", "onsager")
#: one start of 4 ns may scatter by at most this much in t+ for the number to be usable at 298 K
USABLE_SCATTER = 0.05
BLOCKS = 5


# --------------------------------------------------------------------------------- the analysis


def collective_trajectories(centres: np.ndarray, names: Sequence[str], masses: Sequence[float],
                            cation: str, anion: str):
    """The collective coordinates Y_+ and Y_- relative to the centre of mass of the solvent, each
    of shape (frames, 3), with the numbers of cations and anions."""
    centres = np.asarray(centres, float)
    names = np.asarray(list(names))
    masses = np.asarray(masses, float).ravel()
    if centres.ndim != 3 or centres.shape[1] != len(names):
        raise T.TransportError(f"{centres.shape} centres and {len(names)} names")
    is_c, is_a = names == cation, names == anion
    solvent = ~(is_c | is_a)
    if not is_c.any() or not is_a.any():
        raise T.TransportError("the run holds no cation or no anion under the names it gives")
    if not solvent.any():
        raise T.TransportError("the run holds no solvent to be the frame")
    ms = masses[solvent]
    r_solvent = (centres[:, solvent, :] * ms[None, :, None]).sum(axis=1) / ms.sum()
    n_c, n_a = int(is_c.sum()), int(is_a.sum())
    y_c = centres[:, is_c, :].sum(axis=1) - n_c * r_solvent
    y_a = centres[:, is_a, :].sum(axis=1) - n_a * r_solvent
    return y_c, y_a, n_c, n_a


def transference(y_c: np.ndarray, y_a: np.ndarray, dt_ps: float,
                 window: Optional[Sequence[float]] = None, blocks: int = BLOCKS
                 ) -> Dict[str, object]:
    """The transference number of the cation from the collective coordinates, with its error over
    blocks of time origins, and the three slopes it is made of (in the units of
    :func:`cmfqsar.transport.diffusion`, which divides a slope by six).

    The number is nan where the slope of the charge displacement is not above zero.
    """
    win = None if window is None else tuple(window)
    pp = T.diffusion(y_c[:, None, :], dt_ps, window=win, blocks=blocks)
    mm = T.diffusion(y_a[:, None, :], dt_ps, window=win, blocks=blocks)
    cc = T.diffusion((y_c - y_a)[:, None, :], dt_ps, window=win, blocks=blocks)

    def number(a, b, c):
        den = c
        # the cross term <dY+ . d(Y+ - Y-)> = (L++ + L(+-,+-) - L--) / 2
        return float((a + c - b) / (2.0 * den)) if den > 1e-12 * max(abs(a), abs(b), 1e-30) \
            else float("nan")
    t = number(pp["d"], mm["d"], cc["d"])
    per_block = np.array([number(a, b, c) for a, b, c in zip(pp["blocks"], mm["blocks"],
                                                              cc["blocks"])])
    per_block = per_block[np.isfinite(per_block)]
    se = (float(np.std(per_block, ddof=1) / np.sqrt(len(per_block)))
          if len(per_block) > 1 else float("nan"))
    return {"t": t, "se": se, "L_pp": pp["d"], "L_mm": mm["d"], "L_charge": cc["d"],
            "exponent_charge": cc["exponent"], "window_ps": cc["window_ps"],
            "length_ps": cc["length_ps"], "blocks": [float(v) for v in per_block]}


def box_numbers(path: str, cation: str = "LI", anion: str = "ANI") -> Dict[str, object]:
    """The numbers of one stored box: the transference number from the Onsager coefficients, the
    self-diffusion ratio, and the conductivity both ways, as a check of the frame."""
    data = T.read_centres(path)
    dt = data["dt_ps"]
    y_c, y_a, n_c, n_a = collective_trajectories(data["centres"], data["names"],
                                                 data["masses"], cation, anion)
    out = transference(y_c, y_a, dt)
    centres = T.remove_drift(data["centres"], data["masses"])
    names = np.array(data["names"])
    dp = T.diffusion(centres[:, names == cation, :], dt, blocks=BLOCKS)
    dm = T.diffusion(centres[:, names == anion, :], dt, blocks=BLOCKS)
    out["t_self"] = T.transport_number(dp["d"], dm["d"])
    out["d_cation"], out["d_anion"] = float(dp["d"]), float(dm["d"])
    out["pairs"] = n_c
    out["side"] = float(data["side"])
    return out


# ---------------------------------------------------------------------------------- the stages


def _path(name: str) -> str:
    return os.path.join(RD.run_dir(name), "centres.npz")


def series_stage() -> Dict[str, object]:
    out: Dict[str, object] = {"boxes": {}}
    names = [n for v in RD.CONCENTRATION_RUNS.values() for n in v] + list(RD.LONG_RUNS.values())
    for name in names:
        if not os.path.isfile(_path(name)):
            continue
        res = box_numbers(_path(name))
        with open(os.path.join(RD.RESULTS, f"run_{name}.json"), encoding="utf-8") as fh:
            run = json.load(fh)
        res["run"] = name
        res["molarity"] = float(run["molarity"])
        res["measured_t"] = float(run["measured"]["transference"])
        res["sigma_boxes_stage"] = float(run["whole"]["conductivity"]["sigma"])
        # the conductivity from the charge displacement of this analysis, as a check of the frame
        volume = float(res["side"]) ** 3 * 1.0e-30
        scale = T.ELEMENTARY_CHARGE ** 2 / (volume * T.BOLTZMANN * float(run["temperature"]))
        res["sigma_check_ratio"] = float(res["L_charge"] * scale / res["sigma_boxes_stage"])
        out["boxes"][name] = res
        print(f"  {name}: t+ {res['t']:.3f} +/- {res['se']:.3f}, self ratio {res['t_self']:.3f}, "
              f"measured {res['measured_t']:.3f}", flush=True)
    return out


def all62_stage() -> Dict[str, object]:
    out: Dict[str, object] = {"boxes": {}}
    for fid, names in RD.ALL_FORMULATION_RUNS.items():
        for name in names:
            if not os.path.isfile(_path(name)):
                print(f"  {name}: no stored centres", flush=True)
                continue
            res = box_numbers(_path(name))
            res["id"], res["run"] = fid, name
            out["boxes"][name] = res
            print(f"  {name}: t+ {res['t']:.3f} +/- {res['se']:.3f}, self ratio "
                  f"{res['t_self']:.3f}", flush=True)
    return out


def start_scatter(series: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    """The scatter of one start in t+: the root mean square difference of the two starts of the
    series, divided by the square root of two, over the concentrations that have both."""
    d = []
    for names in RD.CONCENTRATION_RUNS.values():
        a, b = (series.get(n) for n in names)
        if a and b and np.isfinite(a["t"]) and np.isfinite(b["t"]):
            d.append(a["t"] - b["t"])
    if not d:
        return {"scatter": float("nan"), "pairs": 0}
    return {"scatter": float(np.sqrt(np.mean(np.square(d)) / 2.0)), "pairs": len(d)}


def save(name: str, payload: Dict[str, object], base: str = RESULTS) -> None:
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


def _read(base: str, name: str) -> Dict[str, object]:
    with open(os.path.join(base, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def write_table(results: str = RESULTS) -> str:
    series = _read(results, "series")["boxes"]
    sc = start_scatter(series)
    out: List[str] = ["# Item BT10, a pilot: the transference number from the stored boxes", ""]
    w = out.append
    w("The cation transference number is read from the Onsager coefficients in the frame of the "
      "solvent, from the stored centres of the boxes of item BT9 (1,000 solvent molecules, charges "
      "of the ions multiplied by 0.7, 4 ns unless the name says long). The definition, the window "
      "and the reading are in the docstring of `benchmarks/run_onsager.py`. The error is the "
      "standard error over 5 blocks of time origins. The self ratio is D+ / (D+ + D-).")
    w("")
    w("## The six concentrations of the series")
    w("")
    w("| Run | Molarity, mol/L | Transference number from the Onsager coefficients | Self "
      "ratio | Measured | Exponent of the charge displacement |")
    w("|---|---:|---:|---:|---:|---:|")
    for r in sorted(series.values(), key=lambda r: (r["molarity"], r["run"])):
        w(f"| {r['run']} | {r['molarity']:.2f} | {r['t']:.3f} +/- {r['se']:.3f} | "
          f"{r['t_self']:.3f} | {r['measured_t']:.3f} | {r['exponent_charge']:.2f} |")
    w("")
    short = [r for r in series.values() if "long" not in r["run"]]
    w(f"* **Scatter of one start.** The two starts of {sc['pairs']} concentrations differ by an "
      f"amount that corresponds to {sc['scatter']:.3f} in t+ for one start. The pilot asked for "
      f"at most {USABLE_SCATTER}. This is "
      + ("within that." if sc["scatter"] <= USABLE_SCATTER else "above that."))
    ok = [r for r in short if np.isfinite(r["t"])]
    if ok:
        diff = np.array([r["t"] - r["t_self"] for r in ok])
        w(f"* **Collective against self.** The number from the Onsager coefficients is on average "
          f"{diff.mean():+.3f} from the self ratio (standard deviation {diff.std():.3f}, "
          f"{len(ok)} boxes).")
        meas = np.array([r["measured_t"] for r in ok])
        w(f"* **Against the measured number.** The mean difference from the measured number is "
          f"{np.mean([r['t'] for r in ok] - meas):+.3f} for the Onsager number and "
          f"{np.mean([r['t_self'] for r in ok] - meas):+.3f} for the self ratio.")
    check = [r["sigma_check_ratio"] for r in series.values()]
    w(f"* **Check of the frame.** The conductivity from the charge displacement of this analysis "
      f"is {min(check):.6f} to {max(check):.6f} times that of item BT9, which uses the frame of the "
      "whole box. The number of cations equals that of anions, so the two have to agree.")
    path62 = os.path.join(results, "all62.json")
    if os.path.isfile(path62):
        a = _read(results, "all62")["boxes"]
        t = np.array([r["t"] for r in a.values()])
        s = np.array([r["t_self"] for r in a.values()])
        ok = np.isfinite(t)
        w("")
        w("## The 62 formulations")
        w("")
        w(f"The transference number from the Onsager coefficients runs from {np.nanmin(t):.2f} to "
          f"{np.nanmax(t):.2f} over {int(ok.sum())} boxes (a mean block error of "
          f"{np.nanmean([r['se'] for r in a.values()]):.2f}); the self ratio runs from "
          f"{s.min():.2f} to {s.max():.2f}. Their rank correlation is "
          f"{RD.spearman(t[ok], s[ok]):+.2f}.")
        w("")
        w("| Formulation | Run | Onsager | Block error | Self ratio |")
        w("|---|---|---:|---:|---:|")
        for r in sorted(a.values(), key=lambda r: (r["id"], r["run"])):
            w(f"| {r['id']} | {r['run']} | {r['t']:.3f} | {r['se']:.3f} | {r['t_self']:.3f} |")
        w("")
    path = os.path.join(results, "table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("series", "all62", "report", "all"), default="all")
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args(argv)
    stages = ("series", "all62", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "series":
            save("series", series_stage(), args.results)
        elif stage == "all62":
            save("all62", all62_stage(), args.results)
        else:
            print("the table:", write_table(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
