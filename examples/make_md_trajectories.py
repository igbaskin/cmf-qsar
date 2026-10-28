"""Short ligand-in-pocket simulations, written as conformational ensembles (plan item D8).

This is the data-preparation step behind `benchmarks/run_md.py`. For one free-energy set it

1. finds the PDB entry the set's pocket file names and downloads it from RCSB (entries there are
   in the public domain under CC0),
2. prepares the whole protein -- one copy, no water or other heteroatoms, missing side-chain
   atoms built, hydrogens added at pH 7,
3. simulates every ligand of the set in that site with GAFF in TIP3P water, the ligand placed
   exactly where the collection puts it, so nothing is docked,
4. reads each trajectory back with `cmfqsar/trajectory.py` -- solvent stripped, every frame
   superimposed on the binding-site backbone, the frames thinned by clustering, each kept
   conformation weighted by its cluster population,
5. writes one SDF of those ensembles, one record per conformation under the ligand's own title,
   which `--ensemble-by name` collapses again.

The SDF is the only thing the benchmark needs, so the table can be rebuilt on a machine with no
simulation software. The trajectories themselves stay in the working directory and are not part
of the repository.

Usage
-----
    python examples/make_md_trajectories.py --set mcl1
    python examples/make_md_trajectories.py --set mcl1 --ns 2 --max-frames 20 --jobs 1
    python examples/make_md_trajectories.py --set mcl1 --read-only     # skip the simulations

The stack is conda-forge only (OpenMM, AmberTools, openff-toolkit, openmmforcefields, pdbfixer).
On a machine without it, run this inside such an environment; everything downstream runs
anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

NEWLINE = chr(10)
from typing import Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar import md as MD  # noqa: E402
from cmfqsar import solvent as SV  # noqa: E402
from cmfqsar import trajectory as TJ  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT_DIR = os.path.join(HERE, "md")
POCKETS = os.path.join(HERE, "pockets", "openff")
LIGANDS = os.path.join(HERE, "openff")

README = """# Short ligand-in-pocket simulations

Conformational ensembles of free-energy-set ligands, taken from molecular-dynamics simulations
of each ligand inside its own crystal site. Made for plan item D8, which asks whether such an
ensemble models a series better than the single crystal pose it started from;
`benchmarks/run_md.py` is the measurement and `docs/development_plan.md` holds the answer.

## What each file is

One SDF per set, named `<set>_md.sdf`. It holds one record per kept conformation, every record
under the ligand's own title, which is how a multi-conformer file is read back:

```bash
cmf build examples/md/mcl1_md.sdf --activity pAffinity --ensemble-by name \\
    --conformations ensemble --conformer-weighting given
```

Each molecule carries `conformer_weights`, the share of simulation frames that fell in that
conformation's cluster, and the activity of the ligand as the source collection gives it.

Beside it, `<set>_water.json` holds the **hydration sites** of every ligand, for plan item D9:
the positions where a water sits in the site, each with the share of frames it is occupied and
the coefficient fitted to the occupancy map. `cmfqsar/solvent.py` turns those into a field and
`benchmarks/run_solvent.py` measures whether that field is worth a sixth kernel block.

## How they were made

{recipe}

`examples/make_md_trajectories.py` is the script, and it is the whole recipe: nothing was done
by hand. The trajectories themselves are not in the repository -- {size} -- but the script
rebuilds them from the data here and the entries named below.

## Where the data comes from

* The ligands and their activities: the Open Force Field protein-ligand-benchmark, free under
  MIT, already in `examples/openff/` with its licence; see `docs/public_3dqsar_datasets.md`.
* The receptors: the PDB entries named below, downloaded from RCSB. The wwPDB distributes its
  holdings in the public domain under CC0, so they carry no restrictions.

{entries}
"""


def say(text):
    print(text, flush=True)


def pose_drift(ensemble, crystal):
    """Median heavy-atom distance of the kept conformations from the crystal pose.

    Measured where they stand, with no further fitting: the ensemble is already in the frame of
    the structure every run was built from, so this is how far the simulation moved the ligand
    inside its site. A number of a few tenths of an angstrom means the pose was kept; a number
    of several angstroms means the ligand went somewhere else and is no longer the molecule the
    comparison is about.
    """
    heavy = [a.GetIdx() for a in crystal.GetAtoms() if a.GetAtomicNum() > 1]
    ref = np.asarray(crystal.GetConformer().GetPositions(), float)[heavy]
    out = []
    for conf in ensemble.GetConformers():
        xyz = np.asarray(conf.GetPositions(), float)[heavy]
        out.append(float(np.sqrt(((xyz - ref) ** 2).sum(axis=1).mean())))
    return float(np.median(out)) if out else float("nan")


def build_set(name, settings, out_dir=OUT_DIR, work=None, max_frames=20, cutoff=0.75,
              stride=1, limit=0, read_only=False, chain_radius=5.0, keep_chains=(),
              max_drift=4.0, common_reference=True, water=True, water_alpha=0.5,
              water_alphas=(0.1, 0.3, 0.5, 1.0, 2.0),
              min_occupancy=SV.MIN_OCCUPANCY, max_sites=SV.MAX_SITES, log=say):
    """One set: download, prepare, simulate, read back, write the ensemble SDF."""
    pocket = os.path.join(POCKETS, name + ".pdb")
    ligands = os.path.join(LIGANDS, name + ".sdf")
    for path in (pocket, ligands):
        if not os.path.isfile(path):
            raise SystemExit(f"{path} is missing; this script works on the openff collection")
    code = MD.entry_of(pocket)
    if not code:
        raise SystemExit(f"{pocket} does not name a source entry in its REMARK 400 lines")
    work = work or os.path.join(out_dir, "work", name)
    os.makedirs(work, exist_ok=True)

    ds = load_dataset(ligands)
    mols, names = list(ds.mols), list(ds.names)
    if limit:
        mols, names = mols[:limit], names[:limit]
    log(f"{name}: {len(mols)} ligands, receptor {code}")

    receptor = os.path.join(work, f"{code}_ready.pdb")
    chains = list(keep_chains)
    if not read_only:
        raw = MD.fetch_entry(code, os.path.join(work, "entries"))
        if not chains:
            chains = MD.chains_near(raw, mols, radius=chain_radius)
        if not os.path.isfile(receptor):
            log(f"  preparing {code}, chain(s) {', '.join(chains)} of "
                f"{len(set(MD.chains_of(raw)))} in the entry")
            MD.prepare_receptor(raw, receptor, keep_chains=chains)
        outcomes = MD.simulate_set(mols, names, receptor, os.path.join(work, "runs"),
                                   settings=settings, log=log)
        failed = [o for o in outcomes if o.error]
        if failed:
            log(f"  {len(failed)} ligand(s) failed: " +
                ", ".join(f"{o.name} ({o.error})" for o in failed[:3]))
        speeds = [o.ns_per_day for o in outcomes if o.ns_per_day > 0]
        if speeds:
            log(f"  {np.median(speeds):.0f} ns/day on this machine")

    runs = os.path.join(work, "runs")
    kept, skipped, left = [], [], []
    drifts = {}
    for m, nm in zip(mols, names):
        top = os.path.join(runs, f"{nm}.pdb")
        dcd = os.path.join(runs, f"{nm}.dcd")
        if not (os.path.isfile(top) and os.path.isfile(dcd)):
            skipped.append(nm)
            continue
        template = Chem.AddHs(Chem.Mol(m), addCoords=True)
        try:
            # every run is superimposed on the one structure they were all built from, so
            # that the ligands come back in a single frame instead of each in its own
            e = TJ.read_trajectory(top, dcd, template=template, max_frames=max_frames,
                                   cutoff=cutoff, stride=stride, name=nm,
                                   reference=(receptor if common_reference
                                              and os.path.isfile(receptor) else None))
        except TJ.TrajectoryError as exc:
            log(f"  {nm}: left out -- {exc}")
            skipped.append(nm)
            left.append(nm)
            continue
        drift = pose_drift(e.mol, template)
        drifts[nm] = drift
        # the distance from the crystal pose only means anything in the crystal pose's own
        # frame, so the check belongs to the common-reference reading alone
        if common_reference and max_drift and drift > float(max_drift):
            log(f"  {nm}: left out -- its conformations sit {drift:.1f} A from the crystal "
                f"pose, past the {max_drift:g} A allowed, so the simulation did not keep it "
                f"in the pose the comparison is about")
            skipped.append(nm)
            left.append(nm)
            continue
        for prop in ds.properties:
            if m.HasProp(prop):
                e.mol.SetProp(prop, m.GetProp(prop))
        e.mol.SetProp("_Name", nm)
        log(f"  {nm}: {e.describe()}, {drift:.2f} A from the crystal pose")
        kept.append(e)
    if not kept:
        raise SystemExit(f"no trajectory of {name} could be read; "
                         f"{len(skipped)} ligand(s) have none")

    os.makedirs(out_dir, exist_ok=True)
    suffix = "_md" if common_reference else "_md_ownframe"
    out = os.path.join(out_dir, f"{name}{suffix}.sdf")
    records = 0
    with Chem.SDWriter(out) as w:
        for e in kept:
            for conf in e.mol.GetConformers():
                w.write(e.mol, confId=conf.GetId())
                records += 1
    log(f"  {len(kept)} ligands in {records} records -> {out}")

    sites = []
    if water and common_reference:
        # the occupancy map needs the water, which the ensemble file does not carry, so it is
        # read from the trajectories a second time; it also needs the common frame, which is why
        # it is skipped for the own-frame reading
        for e in kept:
            nm = e.mol.GetProp("_Name")
            top = os.path.join(runs, f"{nm}.pdb")
            dcd = os.path.join(runs, f"{nm}.dcd")
            try:
                field, occ, heights = SV.field_for(
                    top, dcd, reference=receptor, alpha=float(water_alpha), name=nm,
                    stride=stride, min_occupancy=float(min_occupancy),
                    max_sites=int(max_sites))
            except (SV.SolventError, TJ.TrajectoryError) as exc:
                log(f"  {nm}: no water field -- {exc}")
                continue
            row = {"name": nm,
                   "sites": [[round(float(v), 4) for v in c] + [round(float(w), 5),
                                                                round(float(h), 4)]
                             for c, w, h in zip(field.coords,
                                                np.asarray(field.weights).ravel(), heights)],
                   # a coefficient belongs to one attenuation, so one set per attenuation a
                   # measurement will use; without them a model at another width would be
                   # built from a fit made at this one
                   "coefficients": {k: [round(v, 5) for v in vs] for k, vs in
                                    SV.coefficients_at(occ, field.coords,
                                                       water_alphas).items()},
                   "strongest": round(float(occ.values.max()), 4),
                   "bulk": round(float(occ.bulk), 4)}
            for prop in ds.properties:
                if e.mol.HasProp(prop):
                    row[prop] = e.mol.GetProp(prop)
            sites.append(row)
            log(f"  {nm}: {len(row['sites'])} hydration sites, strongest "
                f"{row['strongest']:.0%} against {row['bulk']:.0%} for bulk water")
        if sites:
            path = os.path.join(out_dir, f"{name}_water.json")
            with open(path, "w", newline=NEWLINE) as fh:
                json.dump({"set": name, "entry": code, "alpha": float(water_alpha),
                           "alphas": [float(a) for a in water_alphas],
                           "min_occupancy": float(min_occupancy),
                           "max_sites": int(max_sites),
                           "probe": SV.PROBE, "spacing": SV.SPACING,
                           "min_distance": SV.MIN_DISTANCE, "ligands": sites}, fh, indent=1)
            log(f"  {len(sites)} water fields -> {path}")
    held = [v for k, v in drifts.items() if k not in left]
    return {"set": name, "entry": code, "chains": chains, "ligands": len(kept),
            "water_fields": len(sites),
            "hydration_sites": (float(np.mean([len(r["sites"]) for r in sites]))
                                if sites else None),
            "records": records, "skipped": skipped, "left_the_site": left,
            "max_frames": max_frames, "cutoff": cutoff, "max_drift": float(max_drift),
            "drift_median": (float(np.median(held)) if held else None),
            "drift_max": (float(np.max(held)) if held else None),
            "trajectory_mb": work_size(work),
            "settings": {k: v for k, v in vars(settings).items()}}


def _left_out(r):
    """Why each ligand of a set is missing, with the two reasons kept apart.

    A ligand can be missing for two different reasons and the note has to say which. Either the
    simulation moved it out of the site, which is a result about the ligand, or there is no
    trajectory for it at all, which is a failure of the preparation and says nothing about the
    ligand. Reporting the second as the first would be a wrong statement about the data.
    """
    left = list(r.get("left_the_site") or [])
    none = [nm for nm in (r.get("skipped") or []) if nm not in left]
    out = ""
    if left:
        out += (f" {len(left)} ligand(s) are left out because the simulation did not keep them "
                f"in the site: {', '.join(left[:6])}.")
    if none:
        out += (f" {len(none)} ligand(s) have no trajectory at all, because the preparation "
                f"could not build a system for them: {', '.join(none[:6])}.")
    return out


def work_size(path):
    """Megabytes of trajectory left in the working directory, for the data statement."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return round(total / 1e6, 1)


def merge_records(records, out_dir=OUT_DIR):
    """This run's records, plus the records of the sets it did not build, sorted by set."""
    path = os.path.join(out_dir, "provenance.json")
    old = []
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                old = json.load(fh)
        except (OSError, ValueError):
            old = []
    fresh = {r["set"] for r in records}
    kept = [r for r in old if isinstance(r, dict) and r.get("set") not in fresh]
    return sorted(records + kept, key=lambda r: r.get("set", ""))


def write_readme(records, out_dir=OUT_DIR):
    """The data statement, built from what was actually run."""
    if not records:
        return
    s = records[0]["settings"]
    recipe = (f"Every ligand was simulated for **{s['nanoseconds']:g} ns** at "
              f"{s['temperature']:g} K and {s['pressure']:g} bar, after "
              f"{s['equilibration_ps']:g} ps of equilibration with the alpha carbons restrained "
              f"at {s['restraint_k']:g} kcal/mol/A^2. The force fields are "
              f"{s['small_molecule_forcefield']} for the ligand with AM1-BCC charges, "
              f"{s['protein_forcefield']} for the protein and TIP3P water with "
              f"{s['ionic_strength']:g} M of ions, in a box with "
              f"{s['padding']:g} A of padding; the time step is {s['timestep_fs']:g} fs with "
              f"hydrogen mass repartitioning and a frame was written every {s['frame_ps']:g} ps. "
              f"The frames were then superimposed on the binding-site backbone and thinned to at "
              f"most {records[0]['max_frames']} conformations at a clustering radius of "
              f"{records[0]['cutoff']:g} A. Where the crystal entry holds several copies "
              f"of the protein, only the chain lining the site the ligands occupy was kept.")
    entries = "\n".join(
        f"* `{r['set']}_md.sdf`: {r['ligands']} ligands, {r['records']} records, receptor "
        f"**{r['entry']}**"
        + (f", chain {' and '.join(r['chains'])} of it" if r.get("chains") else "")
        + "." + _left_out(r)
        + (f" The kept ligands sit {r['drift_median']:.2f} A from their crystal pose on "
           f"average, at most {r['drift_max']:.2f} A." if r.get("drift_median") else "")
        for r in records)
    total = sum(r.get("trajectory_mb") or 0 for r in records)
    size = (f"they come to {total / 1000:.1f} GB" if total >= 1000
            else f"they come to {total:.0f} MB")
    with open(os.path.join(out_dir, "README.md"), "w", newline="\n") as fh:
        fh.write(README.format(recipe=recipe, entries=entries, size=size))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="sets", action="append", default=None,
                    help="a set of examples/openff (repeat for several); default mcl1")
    ap.add_argument("--ns", type=float, default=2.0, help="nanoseconds per ligand (default 2)")
    ap.add_argument("--equilibration-ps", type=float, default=200.0)
    ap.add_argument("--max-frames", type=int, default=20,
                    help="conformations kept per ligand (default 20)")
    ap.add_argument("--cutoff", type=float, default=0.75,
                    help="clustering radius in angstrom (default 0.75)")
    ap.add_argument("--stride", type=int, default=1, help="read every Nth frame")
    ap.add_argument("--limit", type=int, default=0, help="only the first N ligands")
    ap.add_argument("--platform", default="CUDA", help="OpenMM platform (CUDA, OpenCL, CPU)")
    ap.add_argument("--restraint-k", type=float, default=MD.DEFAULT_RESTRAINT)
    ap.add_argument("--chains", default="",
                    help="chains of the entry to keep, comma separated; by default the ones "
                         "whose atoms line the site the ligands occupy")
    ap.add_argument("--chain-radius", type=float, default=5.0,
                    help="how near a chain must come to a ligand to be kept (default 5 A)")
    ap.add_argument("--no-common-reference", action="store_true",
                    help="superimpose each trajectory on its own first frame instead of on the "
                         "structure they were all built from, and write the result as "
                         "<set>_md_ownframe.sdf. This is not how an ensemble should be made -- "
                         "it leaves every ligand in a frame of its own -- and the file is there "
                         "so that benchmarks/run_md.py can measure what that costs")
    ap.add_argument("--water-max-sites", type=int, default=SV.MAX_SITES,
                    help=f"keep at most this many hydration sites per ligand, the most "
                         f"occupied first (default {SV.MAX_SITES}). Giving a small number makes "
                         f"the count the definition of a site rather than the threshold, which "
                         f"is what plan item D9 found transfers between targets")
    ap.add_argument("--no-water", action="store_true",
                    help="skip the water occupancy maps of plan item D9")
    ap.add_argument("--water-alpha", type=float, default=0.5,
                    help="the attenuation the hydration-site Gaussians are fitted at "
                         "(default 0.5, the one the benchmarks use)")
    ap.add_argument("--min-occupancy", type=float, default=SV.MIN_OCCUPANCY,
                    help=f"a point has to be occupied this share of the frames to be a "
                         f"hydration site (default {SV.MIN_OCCUPANCY:g}; bulk water is about "
                         f"0.14 at a 1 A probe)")
    ap.add_argument("--max-drift", type=float, default=4.0,
                    help="leave out a ligand whose conformations sit further than this from its "
                         "crystal pose, in angstrom; 0 keeps every ligand (default 4)")
    ap.add_argument("--read-only", action="store_true",
                    help="do not simulate; read the trajectories already in the working "
                         "directory and write the SDF")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)

    if not a.read_only:
        have = MD.available()
        if not all(have.values()):
            print("the simulation stack is incomplete here: "
                  + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in have.items()))
            print("run this inside an environment that has it, or pass --read-only")
            return 2
    settings = MD.Settings(nanoseconds=a.ns, equilibration_ps=a.equilibration_ps,
                           restraint_k=a.restraint_k, platform=a.platform)
    t0 = time.time()
    records = []
    for name in (a.sets or ["mcl1"]):
        records.append(build_set(name, settings, out_dir=a.out, max_frames=a.max_frames,
                                 cutoff=a.cutoff, stride=a.stride, limit=a.limit,
                                 read_only=a.read_only, chain_radius=a.chain_radius,
                                 keep_chains=[c for c in a.chains.split(",") if c],
                                 max_drift=a.max_drift,
                                 common_reference=not a.no_common_reference,
                                 water=not a.no_water, water_alpha=a.water_alpha,
                                 min_occupancy=a.min_occupancy,
                                 max_sites=a.water_max_sites))
    # the directory holds one set per run and the notes have to describe all of them, so a run
    # replaces the records of the sets it built and leaves the others as they are
    records = merge_records(records, out_dir=a.out)
    if not a.no_common_reference:
        write_readme(records, out_dir=a.out)
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(records, fh, indent=1)
    print(f"total wall time {(time.time() - t0) / 60:.1f} min")
    print("  next:  python benchmarks/run_md.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
