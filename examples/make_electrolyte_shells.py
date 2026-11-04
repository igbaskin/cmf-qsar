"""
Lithium solvation shells sampled from bulk electrolyte simulations, for plan item BT5.

The conductivities are measured values from

    de Blasio P. V. F., Elsborg J. T., Vegge T., Flores E., Bhowmik A.
    "CALiSol-23: Experimental electrolyte conductivity data for various Li-salts and solvent
    combinations", Scientific Data 11 (2024) 1-11.  https://doi.org/10.1038/s41597-024-03575-8
    Data at https://doi.org/10.11583/DTU.24559960, licence CC BY 4.0.

The shells are simulated here: :mod:`cmfqsar.electrolyte` builds and runs one box per
formulation and :mod:`cmfqsar.shells` cuts the shells out of it.

Why this set is here
--------------------
Item BT4 builds one lithium complex and optimises it. Item BT5 asks whether an ensemble of
shells sampled from a liquid describes an electrolyte better, and the target has to be a
transport quantity, because that is what an ensemble of shells could plausibly carry and a single
optimised complex could not. CALiSol-23 is the open collection of measured conductivities, so
the target is measured and only the descriptor is computed.

Two targets are read off each formulation rather than one:

* **log10 of the conductivity at 298.15 K**, in mS/cm. This is what an electrolyte is chosen for.
* **The activation energy at 298.15 K**, in kJ/mol. This is the temperature dependence, and it is
  the quantity a solvation shell has the better claim on: an ion moves by leaving one shell and
  entering the next, so the barrier should depend on how tightly the shell holds it.

Both come from a fit of log10 k = a + b/T + c/T^2 to that formulation's own measurements, which
is a smooth interpolation inside the measured range and is never extrapolated: a formulation is
kept only if its temperatures span at least 20 K and contain 298.15 K.

How the series is chosen, and why each rule is there
----------------------------------------------------
* **One salt, LiPF6.** The force field decides this. GAFF has no boron, so LiBF4 and LiBOB --
  together the larger part of the collection -- cannot be simulated here, and they are refused
  by name rather than approximated.
* **Molality and weight fractions only.** Those two units fix the number of molecules in the box
  exactly. Molarity and volume fractions would need a density, which is an assumption about the
  mixture rather than a measurement of it.
* **Solvents the module can parameterise**, which are the carbonates, the two acetates, the
  ethers and acetonitrile listed in ``cmfqsar.electrolyte.SOLVENTS``.
* **At least four temperatures spanning 20 K and containing 298.15 K**, so that the two targets
  are interpolated and not extrapolated, and a fit whose root-mean-square residual is at most
  0.03 in log10 k.
* **One source paper is dropped.** Its conductivities near room temperature are about fifty times
  below every other paper's, which is a unit or a transcription difference rather than chemistry.
  The rule is written as a rule and not as a name: a paper whose median conductivity between
  0.5 and 1.6 mol of salt and within 12 K of room temperature is below a tenth of the
  collection's median there is dropped, and the script prints which papers that removes.
* **Up to three molalities per solvent composition**, the lowest, a middle one and the highest.
  The concentration axis is kept on purpose: it is where a sampled shell can differ from an
  optimised complex, because the share of shells that hold an anion rises with concentration and
  a single optimised complex has no way to show that.

What is written
---------------
``formulations.csv`` holds one row per formulation with its two targets. ``shells/<id>.sdf``
holds the representative shells of that formulation, each with its population weight, its species
and their atom counts, which is what ``cmfqsar.shells.ensemble_from_molecules`` needs to rebuild
the mean field. ``shells.json`` holds what each run did: the realised molality, the density the
barostat settled at, the composition of the shells, the mean coordination number, the
block-averaged error of the mean field and the speed.

The trajectories themselves are not written here. A nanosecond of a box of three thousand atoms
is tens of megabytes, and the shells are what the model reads, so the trajectories stay in the
cache directory, which is not tracked.

What this set is not
--------------------
The shells are a model of the liquid and not a measurement of one. A fixed-charge force field has
no polarisation and is known to bind a small cation to a carbonate too strongly, the box holds a
few hundred molecules, and the sampling is one nanosecond. ``shells.json`` reports the density
the box settled at and the block-averaged error of the mean field so that both can be judged.

Usage
-----
    python examples/make_electrolyte_shells.py --fetch --select
    python examples/make_electrolyte_shells.py --simulate            # needs the md environment
    python examples/make_electrolyte_shells.py --simulate --limit 1  # one, to time it
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar import shells as SH                                             # noqa: E402
from cmfqsar.electrolyte import (Formulation, Settings, SOLVENTS, run,       # noqa: E402
                                 shell_templates)
from cmfqsar.fields import FieldSet                                          # noqa: E402

SOURCE = "https://ndownloader.figshare.com/files/43151344"
SOURCE_NAME = "calisol23.csv"
SOURCE_SHA256 = "850ff20eaad7abfe86fdd455b9cbf661e1b393dbdf9bb33e47769e00c2f5f617"

OUT = os.path.join(HERE, "electrolyte")
UPSTREAM = os.path.join(OUT, "upstream")
SHELL_DIR = os.path.join(OUT, "shells")
CACHE = os.path.join(HERE, "..", "benchmarks", "cache", "electrolyte")

SALT = "LiPF6"
R_KJ = 8.314462618e-3
REFERENCE_T = 298.15
MIN_POINTS = 4
MIN_SPAN = 20.0
MAX_FIT_RMS = 0.03
MAX_PER_COMPOSITION = 3
SCALE_FLOOR = 0.1                 # a paper below this share of the collection's median is dropped

#: five physico-chemical fields, as every other benchmark here uses
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")

META = ("", "doi", "k", "T", "c", "salt", "c units", "solvent ratio type")


# ------------------------------------------------------------------------------- the upstream


def fetch_upstream() -> str:
    """Download the conductivity table and check its hash."""
    os.makedirs(UPSTREAM, exist_ok=True)
    path = os.path.join(UPSTREAM, SOURCE_NAME)
    if not os.path.exists(path):
        print(f"downloading {SOURCE}")
        with urllib.request.urlopen(SOURCE, timeout=120) as src:
            data = src.read()
        with open(path, "wb") as fh:
            fh.write(data)
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != SOURCE_SHA256:
        raise SystemExit(f"{path} has hash {got}, not the expected {SOURCE_SHA256}")
    print(f"{path}: hash checks out")
    return path


def _num(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def read_rows(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    solvents = [c for c in rows[0] if c not in META]
    return rows, solvents


def paper_scales(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """Each paper's median conductivity in a common window, as a share of the collection's."""
    near: Dict[str, List[float]] = {}
    for r in rows:
        k, T, c = _num(r, "k"), _num(r, "T"), _num(r, "c")
        if k is None or T is None or c is None or k <= 0:
            continue
        if abs(T - REFERENCE_T) > 12.0 or not (0.5 <= c <= 1.6):
            continue
        near.setdefault(r["doi"], []).append(k)
    if not near:
        return {}
    median = float(np.median([v for vs in near.values() for v in vs]))
    return {doi: float(np.median(vs)) / median for doi, vs in near.items()}


def composition_of(row: Dict[str, str], solvents: List[str]) -> Dict[str, float]:
    out = {}
    for s in solvents:
        v = _num(row, s)
        if v is not None and v > 1e-9:
            out[s] = v
    total = sum(out.values())
    return {s: v / total for s, v in out.items()} if total > 0 else {}


def fit_targets(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """(log10 k at 298.15 K, activation energy there in kJ/mol, fit residual in log10 k)."""
    T = np.array([p[0] for p in points], float)
    y = np.log10(np.array([p[1] for p in points], float))
    A = np.vstack([np.ones_like(T), 1.0 / T, 1.0 / T ** 2]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rms = float(np.sqrt(np.mean((y - A @ coef) ** 2)))
    log_k = float(coef[0] + coef[1] / REFERENCE_T + coef[2] / REFERENCE_T ** 2)
    slope = float(coef[1] + 2.0 * coef[2] / REFERENCE_T)      # d(log10 k)/d(1/T)
    return log_k, float(-R_KJ * np.log(10.0) * slope), rms


def select(path: str) -> List[Dict[str, object]]:
    """The series, with its two targets, by the rules in this file's docstring."""
    rows, solvents = read_rows(path)
    scales = paper_scales(rows)
    dropped = sorted(doi for doi, s in scales.items() if s < SCALE_FLOOR)
    for doi in dropped:
        print(f"dropping {doi}: its conductivity scale is {scales[doi]:.3f} of the collection's")

    groups: Dict[Tuple, List[Tuple[float, float, str]]] = {}
    for r in rows:
        if r["salt"].strip() != SALT or r["c units"] != "mol/kg":
            continue
        if r["solvent ratio type"] != "w" or r["doi"] in dropped:
            continue
        k, T, c = _num(r, "k"), _num(r, "T"), _num(r, "c")
        if k is None or T is None or c is None or k <= 0:
            continue
        comp = composition_of(r, solvents)
        if not comp or any(s not in SOLVENTS for s in comp):
            continue
        key = (tuple(sorted((s, round(v, 3)) for s, v in comp.items())), round(c, 3))
        groups.setdefault(key, []).append((T, k, r["doi"]))
    print(f"{len(groups)} formulations of {SALT} in solvents this module can parameterise")

    usable: Dict[Tuple, Dict[str, object]] = {}
    for key, pts in groups.items():
        temps = sorted(p[0] for p in pts)
        if len(pts) < MIN_POINTS or temps[-1] - temps[0] < MIN_SPAN:
            continue
        if not temps[0] <= REFERENCE_T <= temps[-1]:
            continue
        log_k, ea, rms = fit_targets([(p[0], p[1]) for p in pts])
        if rms > MAX_FIT_RMS or not 0.0 < ea < 40.0:
            continue
        usable[key] = {"log10_k_298": log_k, "ea_298": ea, "fit_rms": rms,
                       "n_points": len(pts), "t_min": temps[0], "t_max": temps[-1],
                       "dois": sorted({p[2] for p in pts})}
    print(f"{len(usable)} pass the temperature and fit checks")

    by_comp: Dict[Tuple, List[float]] = {}
    for comp, c in usable:
        by_comp.setdefault(comp, []).append(c)
    chosen: List[Dict[str, object]] = []
    for comp in sorted(by_comp):
        cs = sorted(by_comp[comp])
        if len(cs) <= MAX_PER_COMPOSITION:
            take = cs
        else:
            take = [cs[0], cs[len(cs) // 2], cs[-1]]
        for c in take:
            info = dict(usable[(comp, c)])
            form = Formulation(salt=SALT, solvents={s: v for s, v in comp}, molality=c)
            info.update({"id": form.name, "salt": SALT, "molality": c,
                         "solvents": dict(form.solvents)})
            chosen.append(info)
    print(f"{len(chosen)} chosen, over {len(by_comp)} solvent compositions")
    return sorted(chosen, key=lambda d: str(d["id"]))


def write_formulations(chosen: List[Dict[str, object]]) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "formulations.csv")
    names = sorted({s for row in chosen for s in row["solvents"]})
    header = (["id", "salt", "molality", "log10_k_298", "ea_298", "fit_rms", "n_points",
               "t_min", "t_max", "dois"] + [f"w_{s}" for s in names])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in chosen:
            w.writerow([row["id"], row["salt"], f"{row['molality']:.3f}",
                        f"{row['log10_k_298']:.4f}", f"{row['ea_298']:.3f}",
                        f"{row['fit_rms']:.4f}", row["n_points"],
                        f"{row['t_min']:.2f}", f"{row['t_max']:.2f}",
                        ";".join(row["dois"])]
                       + [f"{row['solvents'].get(s, 0.0):.3f}" for s in names])
    print(f"wrote {path}: {len(chosen)} formulations")
    return path


def read_formulations(path: str) -> List[Dict[str, object]]:
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            solvents = {k[2:]: float(v) for k, v in row.items()
                        if k.startswith("w_") and float(v) > 0}
            out.append({"id": row["id"], "salt": row["salt"],
                        "molality": float(row["molality"]), "solvents": solvents,
                        "log10_k_298": float(row["log10_k_298"]),
                        "ea_298": float(row["ea_298"])})
    return out


# ---------------------------------------------------------------------- simulate and extract


def one_formulation(row: Dict[str, object], settings: Settings, fieldset: FieldSet,
                    max_members: int, blocks: int) -> Dict[str, object]:
    """Run one box, cut its shells, write the members, and return what happened."""
    form = Formulation(salt=str(row["salt"]), solvents=dict(row["solvents"]),
                       molality=float(row["molality"]), name=str(row["id"]))
    workdir = os.path.abspath(os.path.join(CACHE, form.name))
    info = run(form, workdir, settings, progress=lambda s: print("   ", s, flush=True))

    from cmfqsar.trajectory import load
    t0 = time.time()
    traj = load(info["topology"], info["trajectory"])
    tmpl = shell_templates(form)
    SH.check_templates(traj, tmpl)
    found = SH.extract_shells(traj, form.cation, min_molecules=1)
    read_s = time.time() - t0

    t0 = time.time()
    ens = SH.shell_ensemble(found, tmpl, fieldset, max_members=max_members, name=form.name)
    conv = SH.convergence(found, tmpl, fieldset, n_blocks=blocks, max_members=max_members)
    build_s = time.time() - t0

    os.makedirs(SHELL_DIR, exist_ok=True)
    path = os.path.join(SHELL_DIR, f"{form.name}.sdf")
    writer = Chem.SDWriter(path)
    for mol, shell, weight in zip(ens.members, ens.chosen, ens.weights):
        out = Chem.Mol(mol)
        out.SetProp("_Name", form.name)
        for key, value in SH.member_properties(shell, weight).items():
            out.SetProp(key, value)
        writer.write(out)
    writer.close()

    return {
        "id": form.name, "atoms": info["atoms"], "frames": info["frames"],
        "realised_molality": info["realised_molality"], "density": info["density"],
        "start_density": info["start_density"], "final_side": info["final_side"],
        "ns_per_day": info["ns_per_day"], "run_seconds": info["seconds"],
        "read_seconds": read_s, "build_seconds": build_s,
        "shells": len(found), "members": len(ens.members),
        "mean_coordination": ens.mean_coordination,
        "mean_molecules": float(np.mean([s.n_molecules for s in found])),
        "ion_pair_share": float(np.mean([("ANI" in s.species) for s in found])),
        "composition": [[" ".join(f"{s}:{n}" for s, n in comp), share]
                        for comp, share in ens.composition[:8]],
        "block_error": conv["block_error"], "thinning_error": conv["thinning_error"],
        "sampled_for_convergence": conv["sampled"],
        "sdf": os.path.relpath(path, OUT),
    }


def simulate(rows: List[Dict[str, object]], settings: Settings, max_members: int,
             blocks: int, limit: int = 0) -> List[Dict[str, object]]:
    os.makedirs(CACHE, exist_ok=True)
    fieldset = FieldSet.create(physchem=list(FIELDS), alpha=0.3, charge_model="gasteiger")
    path = os.path.join(OUT, "shells.json")
    done: Dict[str, Dict[str, object]] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            done = {d["id"]: d for d in json.load(fh)["runs"]}
    todo = [r for r in rows if r["id"] not in done]
    if limit:
        todo = todo[:int(limit)]
    print(f"{len(done)} already done, {len(todo)} to run now")
    for i, row in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {row['id']}", flush=True)
        t0 = time.time()
        try:
            done[str(row["id"])] = one_formulation(row, settings, fieldset, max_members, blocks)
        except Exception as err:                 # noqa: BLE001 - one box must not stop the rest
            print(f"    failed: {type(err).__name__}: {err}", flush=True)
            done[str(row["id"])] = {"id": row["id"], "failed": f"{type(err).__name__}: {err}"}
        print(f"    {time.time() - t0:.0f} s", flush=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"settings": vars(settings), "max_members": max_members,
                       "blocks": blocks,
                       "runs": [done[k] for k in sorted(done)]}, fh, indent=1, default=str)
    return [done[k] for k in sorted(done)]


README = """\
# Lithium solvation shells from bulk electrolyte simulations

For plan item BT5, which asks whether an ensemble of solvation shells sampled from a liquid
describes an electrolyte better than the single optimised complex of item BT4.

## What is here

| File | What it holds |
|---|---|
| `formulations.csv` | one row per electrolyte: the salt, the solvent weight fractions, the molality, and the two measured targets |
| `shells/<id>.sdf` | the representative solvation shells of that electrolyte, each with its population weight |
| `shells.json` | what each simulation did and how well it is converged |

`formulations.csv` has one row per formulation. `log10_k_298` is the base-ten logarithm of the
ionic conductivity in mS/cm at 298.15 K and `ea_298` is the activation energy in kJ/mol at the
same temperature. Both are read off a fit of log10 k = a + b/T + c/T^2 to that formulation's own
measurements, which is an interpolation: a formulation is listed only when its measured
temperatures span at least 20 K and contain 298.15 K. `fit_rms` is the residual of that fit in
log10 k, `n_points` the number of measurements behind it, and `dois` the papers they come from.

Each file in `shells/` holds one record per representative shell, under the formulation's name.
The properties `shell_weight`, `shell_species` and `shell_sizes` are what
`cmfqsar.shells.ensemble_from_molecules` needs to rebuild the mean field, so a model is built
from these files directly:

```python
from rdkit import Chem
from cmfqsar.fields import FieldSet
from cmfqsar.shells import ensemble_from_molecules

fs = FieldSet.create(alpha=0.3, charge_model="gasteiger")
members = [m for m in Chem.SDMolSupplier("shells/LiPF6_DMC100_m1p000.sdf", removeHs=False)]
fields = ensemble_from_molecules(members, fs, name="DMC at 1 mol/kg")
```

The ion is always atom 0 of every record, and every record is in the cation frame of item BT4.

## Where the numbers come from

The conductivities are measurements, taken from CALiSol-23:

> de Blasio P. V. F., Elsborg J. T., Vegge T., Flores E., Bhowmik A. "CALiSol-23: Experimental
> electrolyte conductivity data for various Li-salts and solvent combinations", *Scientific Data*
> **11** (2024) 1-11. <https://doi.org/10.1038/s41597-024-03575-8>
> Data: <https://doi.org/10.11583/DTU.24559960>, licence CC BY 4.0.

The shells are computed here, not measured: one box per formulation, packed with packmol, the
solvents and the anion typed with GAFF-2.11 and charged with the OpenFF graph network, the
lithium ion with the Amber ion parameters, and run with OpenMM under pressure coupling. The
shells are then cut by `cmfqsar.shells`. `examples/make_electrolyte_shells.py` is the whole
recipe and `benchmarks/run_shells.py` the measurement.

## What these shells are not

A model of the liquid, not a measurement of one. Three limits are worth stating:

* **No polarisation.** A fixed-charge force field is known to bind a small cation to a carbonate
  too strongly, so the coordination numbers here are more likely too high than too low.
* **A small box and a short run.** A few hundred solvent molecules and one nanosecond.
  `shells.json` reports the `block_error`, the standard error of the mean field over blocks of
  frames as a share of the field's own size, so that the sampling can be judged rather than
  assumed.
* **The density is a check, not an input.** `density` is what the barostat settled at and
  `start_density` the ideal-mixing value the box was built from. A large difference is a result
  about the force field.

## Licence of this copy

The conductivities come from a CC BY 4.0 dataset and are redistributed here under the same
licence, with the citation above. The shell files are produced by this repository's own code.
Under the three-word scheme of `docs/public_3dqsar_datasets.md` that makes these files **free**:
the licence is explicit, so the copy may travel with the program, and the one obligation it
carries is the citation above.
"""


def write_readme() -> None:
    path = os.path.join(OUT, "README.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(README)
    print(f"wrote {path}")


def summarise(runs: List[Dict[str, object]]) -> None:
    ok = [r for r in runs if "failed" not in r]
    bad = [r for r in runs if "failed" in r]
    print(f"\n{len(ok)} runs, {len(bad)} failures")
    if not ok:
        return
    for key, label, fmt in (("atoms", "atoms", "{:.0f}"),
                            ("density", "density settled at, g/cm3", "{:.3f}"),
                            ("mean_molecules", "molecules in a shell", "{:.2f}"),
                            ("mean_coordination", "donors at the ion", "{:.2f}"),
                            ("ion_pair_share", "shells holding an anion", "{:.3f}"),
                            ("shells", "shells cut", "{:.0f}"),
                            ("members", "members kept", "{:.1f}"),
                            ("block_error", "block error of the mean field", "{:.5f}"),
                            ("thinning_error", "cost of keeping 16 members", "{:.5f}"),
                            ("ns_per_day", "ns per day", "{:.0f}"),
                            ("run_seconds", "seconds per box", "{:.0f}")):
        vs = np.array([float(r[key]) for r in ok if r.get(key) is not None], float)
        vs = vs[np.isfinite(vs)]
        if len(vs) == 0:
            continue
        print(f"  {label:32s} " + fmt.format(float(np.mean(vs)))
              + " +/- " + fmt.format(float(np.std(vs)))
              + "   (" + fmt.format(float(vs.min())) + " to "
              + fmt.format(float(vs.max())) + ")")
    for r in bad:
        print(f"  failed: {r['id']}: {r['failed']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fetch", action="store_true", help="download the conductivity table")
    ap.add_argument("--select", action="store_true", help="write formulations.csv and the README")
    ap.add_argument("--simulate", action="store_true", help="run the boxes and cut the shells")
    ap.add_argument("--limit", type=int, default=0, help="run at most this many formulations")
    ap.add_argument("--production-ps", type=float, default=1000.0)
    ap.add_argument("--equilibration-ps", type=float, default=150.0)
    ap.add_argument("--frame-ps", type=float, default=5.0)
    ap.add_argument("--solvents", type=int, default=250, help="solvent molecules in a box")
    ap.add_argument("--members", type=int, default=16, help="representative shells kept")
    ap.add_argument("--blocks", type=int, default=5, help="blocks for the error of the mean")
    ap.add_argument("--platform", default="CUDA")
    args = ap.parse_args()
    if not (args.fetch or args.select or args.simulate):
        ap.error("nothing to do: pass --fetch, --select or --simulate")

    table = os.path.join(UPSTREAM, SOURCE_NAME)
    if args.fetch:
        table = fetch_upstream()
    if args.select:
        if not os.path.exists(table):
            raise SystemExit(f"{table} is not here; run with --fetch first")
        write_formulations(select(table))
        write_readme()
    if args.simulate:
        path = os.path.join(OUT, "formulations.csv")
        if not os.path.exists(path):
            raise SystemExit(f"{path} is not here; run with --select first")
        settings = Settings(production_ps=args.production_ps,
                            equilibration_ps=args.equilibration_ps,
                            frame_ps=args.frame_ps, n_solvent=args.solvents,
                            platform=args.platform)
        runs = simulate(read_formulations(path), settings, args.members, args.blocks,
                        limit=args.limit)
        summarise(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
