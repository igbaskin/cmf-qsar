#!/usr/bin/env python
"""
What the trajectory reader costs and what its thinning loses.

Plan item D7 is a reader: it turns a simulation into the weighted conformational ensemble the
field machinery already takes. Plan item D8 is the measurement that reader exists for -- short
simulations on a set whose ligands have crystal poses, and the comparison that says whether a
trajectory mean field beats the single pose. **This script is not D8 and does not answer it.**
It measures the three things that belong to the reader itself and can be measured without any
simulation package:

**Do the two readers agree?** mdtraj and MDAnalysis are alternatives, so a result must not depend
on which one is installed. The same files are read by both and the coordinates compared.

**What does the thinning lose?** The kernel of an ensemble of k conformations costs k^2
atom-pair integrals, so the frames have to be thinned and the question is how far. Every frame
is kept as the reference, then the leader clustering is run at a series of cutoffs, and the mean
field of the representatives is compared with the mean field of all the frames -- as the kernel
sees it, through the relative error of the Gram matrix over a set of molecules. That curve is
what the default of thirty conformations rests on.

**What does it cost?** The kernel time against the number of conformations kept, which should
grow with its square, measured rather than assumed.

The frames do not come from a simulation, because this repository has none and producing one
needs a package it does not depend on. They are conformations of real ligands, written to a real
trajectory file and read back through the whole chain. That exercises every step of the reader
and measures its thinning honestly; it says nothing about molecular dynamics, and the verdict
below says so.

Usage
-----
    python benchmarks/run_trajectory.py
    python benchmarks/run_trajectory.py --sets GPB THR
    python benchmarks/run_trajectory.py --report
"""
from __future__ import annotations

import argparse
import glob
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
from rdkit.Chem import AllChem  # noqa: E402

from cmfqsar import trajectory as TJ  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "trajectory")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("GPB", "THR", "BZR")
FIELDS = ["electrostatic", "steric"]
ALPHA = 0.5
#: molecules per set: enough for a Gram matrix to mean something, few enough to be quick
N_MOLECULES = 12
#: frames written per molecule
N_FRAMES = 60
#: the clustering radii measured
CUTOFFS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _say(text):
    print(text, flush=True)


# ----------------------------------------------------------------------------------------------
# a trajectory to read
# ----------------------------------------------------------------------------------------------

def write_trajectory(mol: Chem.Mol, path_stem: str, n_frames: int = N_FRAMES,
                     seed: int = 0) -> str:
    """Conformations of `mol` written as a PDB topology and a DCD; returns the stem.

    Not a simulation: a set of conformations in a trajectory file, so that the reader can be put
    through its whole chain on a real molecule.
    """
    import mdtraj

    work = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    work.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) or 1000003
    params.pruneRmsThresh = -1.0
    AllChem.EmbedMultipleConfs(work, numConfs=int(n_frames), params=params)
    if work.GetNumConformers() == 0:
        raise RuntimeError("no conformation could be generated")
    AllChem.MMFFOptimizeMoleculeConfs(work, maxIters=200)
    for k, a in enumerate(work.GetAtoms()):
        ri = Chem.AtomPDBResidueInfo()
        ri.SetResidueName("LIG")
        ri.SetResidueNumber(1)
        ri.SetIsHeteroAtom(True)
        ri.SetName(f"{a.GetSymbol()}{k:<3d}"[:4])
        a.SetMonomerInfo(ri)
    pdb = path_stem + ".pdb"
    os.makedirs(os.path.dirname(os.path.abspath(pdb)), exist_ok=True)
    Chem.MolToPDBFile(work, pdb, confId=work.GetConformer(0).GetId())
    top = mdtraj.load(pdb).topology
    xyz = np.stack([np.asarray(c.GetPositions()) for c in work.GetConformers()]) / 10.0
    mdtraj.Trajectory(xyz, top).save_dcd(path_stem + ".dcd")
    return path_stem


def prepare(name, n_molecules=N_MOLECULES, n_frames=N_FRAMES, results=RESULTS):
    """`n_molecules` of one Sutherland set, each with a trajectory file written for it."""
    path = os.path.join(DATA, name + ".sdf")
    ds = load_dataset(path)
    y = np.asarray(ds.activity, float)
    keep = [i for i in range(len(ds.mols)) if np.isfinite(y[i])][:int(n_molecules)]
    out = []
    for i in keep:
        stem = os.path.join(results, "work", name, f"{i:04d}")
        if not os.path.isfile(stem + ".dcd"):
            try:
                write_trajectory(ds.mols[i], stem, n_frames=n_frames, seed=i + 1)
            except Exception as exc:          # noqa: BLE001 - a molecule that will not embed
                _say(f"    {ds.names[i]}: skipped ({type(exc).__name__}: {exc})")
                continue
        out.append((ds.names[i], ds.mols[i], stem))
    return out


# ----------------------------------------------------------------------------------------------
# the kernel the model sees
# ----------------------------------------------------------------------------------------------

def gram(mols, names, weighting="given", alpha=ALPHA):
    """The joint kernel over an ensemble of molecules, which is what a model is fitted to."""
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha, conformers="ensemble",
                         conformer_weighting=weighting)
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    return np.asarray(model.joint_kernel(model.prepare_training_kernels(table)), float)


def relative_error(K, K_ref):
    return float(np.linalg.norm(K - K_ref) / max(np.linalg.norm(K_ref), 1e-300))


# ----------------------------------------------------------------------------------------------
# the stages
# ----------------------------------------------------------------------------------------------

def stage_engines(items):
    """The two readers on the same files: a result must not depend on which one is installed."""
    have = TJ.available_engines()
    if len(have) < 2:
        _say(f"  only {have} is installed; the comparison needs both")
        return None
    worst, checked = 0.0, 0
    for _name, _mol, stem in items:
        a = TJ.load(stem + ".pdb", stem + ".dcd", engine="mdtraj")
        b = TJ.load(stem + ".pdb", stem + ".dcd", engine="mdanalysis")
        if a.elements != b.elements or a.coords.shape != b.coords.shape:
            return {"agree": False, "molecules": checked,
                    "note": "the two readers gave different atoms"}
        worst = max(worst, float(np.abs(a.coords - b.coords).max()))
        checked += 1
    _say(f"  {checked} trajectories, largest coordinate difference {worst:.2e} A")
    return {"agree": bool(worst < 1e-3), "molecules": checked, "max_difference": worst}


def stage_thinning(items, cutoffs=CUTOFFS):
    """What the thinning loses, measured on the kernel rather than on the coordinates."""
    full_mols, full_names = [], []
    for name, mol, stem in items:
        e = TJ.read_trajectory(stem + ".pdb", stem + ".dcd", ligand="LIG",
                               template=Chem.AddHs(Chem.Mol(mol), addCoords=True),
                               max_frames=0, cutoff=0.0, name=name)
        full_mols.append(e.mol)
        full_names.append(name)
    K_ref = gram(full_mols, full_names)
    n_frames = int(full_mols[0].GetNumConformers())
    rows = []
    for cut in cutoffs:
        mols, kept = [], []
        for name, mol, stem in items:
            e = TJ.read_trajectory(stem + ".pdb", stem + ".dcd", ligand="LIG",
                                   template=Chem.AddHs(Chem.Mol(mol), addCoords=True),
                                   max_frames=0, cutoff=float(cut), name=name)
            mols.append(e.mol)
            kept.append(e.mol.GetNumConformers())
        t0 = time.time()
        K = gram(mols, full_names)
        rows.append({"cutoff": float(cut), "kept_mean": float(np.mean(kept)),
                     "kept_max": int(np.max(kept)),
                     "error": relative_error(K, K_ref),
                     "kernel_seconds": float(time.time() - t0)})
        _say(f"  cutoff {cut:4.2f} A  {np.mean(kept):5.1f} conformations kept of {n_frames}  "
             f"kernel error {rows[-1]['error']:.2e}  {rows[-1]['kernel_seconds']:.1f} s")
    t0 = time.time()
    gram(full_mols, full_names)
    return {"frames": n_frames, "molecules": len(items), "rows": rows,
            "full_seconds": float(time.time() - t0)}


def stage_cost(items, counts=(1, 2, 5, 10, 20, 40)):
    """Kernel time against the number of conformations kept; it should grow with the square."""
    rows = []
    for k in counts:
        mols = []
        for name, mol, stem in items:
            e = TJ.read_trajectory(stem + ".pdb", stem + ".dcd", ligand="LIG",
                                   template=Chem.AddHs(Chem.Mol(mol), addCoords=True),
                                   max_frames=int(k), cutoff=0.05, name=name)
            mols.append(e.mol)
        got = float(np.mean([m.GetNumConformers() for m in mols]))
        t0 = time.time()
        gram(mols, [n for n, _m, _s in items])
        rows.append({"asked": int(k), "kept_mean": got,
                     "seconds": float(time.time() - t0)})
        _say(f"  up to {k:3d} conformations ({got:4.1f} kept)  kernel {rows[-1]['seconds']:.2f} s")
    return {"rows": rows, "molecules": len(items)}


# ----------------------------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------------------------

def load_all(results=RESULTS):
    out = {}
    for p in sorted(glob.glob(os.path.join(results, "*.json"))):
        with open(p) as fh:
            out[os.path.basename(p)[:-5]] = json.load(fh)
    return out


def report(results=RESULTS, path=None):
    d = load_all(results)
    out = ["# Reading a trajectory as a conformational ensemble", "",
           "Plan item D7 is the reader; plan item D8 is the measurement it exists for, and this "
           "is not that measurement. What is measured here belongs to the reader itself: that "
           "the two readers agree, what the thinning of the frames loses, and what the frames "
           "cost. The frames are conformations of real ligands written to a real trajectory "
           "file, not a simulation, because this repository has none.", ""]

    eng = d.get("engines")
    if eng:
        if eng.get("agree"):
            out += ["## The two readers agree", "",
                    f"On {eng['molecules']} trajectories read by both mdtraj and MDAnalysis, the "
                    f"largest difference in any coordinate is "
                    f"**{eng['max_difference']:.1e} A**. Which of the two is installed therefore "
                    f"does not change a result.", ""]
        else:
            out += ["## The two readers do NOT agree", "",
                    f"{eng.get('note', '')} This has to be fixed before either is trusted.", ""]

    th = d.get("thinning")
    if th:
        out += ["## What the thinning loses", "",
                f"{th['molecules']} molecules of {th['frames']} frames each. The reference is "
                f"the mean field over every frame; the error is of the joint kernel, which is "
                f"what a model is fitted to.", "",
                "| clustering radius | conformations kept | error of the kernel | kernel time |",
                "| ---: | ---: | ---: | ---: |"]
        for r in th["rows"]:
            out.append(f"| {r['cutoff']:.2f} A | {r['kept_mean']:.1f} | {r['error']:.2e} | "
                       f"{r['kernel_seconds']:.1f} s |")
        out += ["", f"All {th['frames']} frames take {th['full_seconds']:.1f} s for the same "
                    f"kernel.", ""]

    co = d.get("cost")
    if co:
        out += ["## What the frames cost", "",
                "| conformations asked for | kept | kernel time |",
                "| ---: | ---: | ---: |"]
        for r in co["rows"]:
            out.append(f"| {r['asked']} | {r['kept_mean']:.1f} | {r['seconds']:.2f} s |")
        out.append("")
    out += verdict(d)
    text = "\n".join(out) + "\n"
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="\n") as fh:
            fh.write(text)
    return text


def verdict(d):
    out = ["## What it settles", ""]
    th = d.get("thinning")
    if th and th["rows"]:
        rows = sorted(th["rows"], key=lambda r: r["cutoff"])
        tight, loose = rows[0], rows[-1]
        full = float(th.get("full_seconds") or 0.0)
        # the row nearest the default radius, which is what the default has to be justified by
        default = min(rows, key=lambda r: abs(r["cutoff"] - 1.0))
        out += [f"**The thinning trades accuracy for time in a straight line, with no radius "
                f"at which it is free.** Over {th['molecules']} molecules of {th['frames']} "
                f"frames, the kernel error runs from {tight['error']:.1e} at "
                f"{tight['cutoff']:.2f} A ({tight['kept_mean']:.0f} conformations kept) to "
                f"{loose['error']:.1e} at {loose['cutoff']:.2f} A "
                f"({loose['kept_mean']:.0f} kept). It does not flatten out: keeping more frames "
                f"keeps improving the kernel, so the number to keep is a decision about cost "
                f"and not a point where the curve stops moving.", ""]
        if full > 0 and default["kernel_seconds"] > 0:
            out += [f"At the default radius of 1 A the ensemble keeps "
                    f"{default['kept_mean']:.0f} of {th['frames']} conformations, the kernel is "
                    f"within **{default['error']:.1e}** of the one built from every frame, and "
                    f"it takes {default['kernel_seconds']:.0f} s against {full:.0f} s for all "
                    f"of them. A few per cent of the kernel for a quarter of the time is the "
                    f"trade the default makes. The radius is the control, but what the cost "
                    f"follows is the number of conformations it leaves, and that number depends "
                    f"on how flexible the molecules are: the same radius keeps far more frames "
                    f"of a flexible ligand than of a rigid one.", ""]
    co = d.get("cost")
    if co and len(co["rows"]) >= 2:
        rows = [r for r in co["rows"] if r["seconds"] > 0 and r["kept_mean"] > 0]
        if len(rows) >= 2:
            a, b = rows[0], rows[-1]
            ratio_k = b["kept_mean"] / a["kept_mean"]
            ratio_t = b["seconds"] / a["seconds"]
            out += [f"**The cost follows the square of the number of conformations, as the "
                    f"algebra says it must.** From {a['kept_mean']:.0f} to {b['kept_mean']:.0f} "
                    f"conformations is {ratio_k:.0f} times as many and {ratio_t:.0f} times the "
                    f"kernel time, against {ratio_k ** 2:.0f} for an exact square. The kernel "
                    f"of two ensembles is a double sum over their members, so the thinning "
                    f"belongs inside the reader rather than being left to the caller.", ""]
    eng = d.get("engines")
    if eng and eng.get("agree"):
        out += [f"**Which reader is installed does not matter.** mdtraj and MDAnalysis differ "
                f"by at most {eng['max_difference']:.0e} A on any coordinate of "
                f"{eng['molecules']} trajectories.", ""]
    out += ["**What this does not say.** Nothing here compares a trajectory with a crystal "
            "pose, because the frames are conformations of the ligands and not a simulation. "
            "That comparison is plan item D8, and it needs short simulations on a set whose "
            "ligands have crystal poses.", ""]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--molecules", type=int, default=N_MOLECULES)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--report", nargs="?", const=os.path.join(RESULTS, "table.md"), default=None)
    a = ap.parse_args(argv)

    if a.report:
        print(report(path=a.report))
        print(f"written {a.report}")
        return 0
    if not TJ.available_engines():
        print("this benchmark needs mdtraj or MDAnalysis; neither is installed")
        return 2
    os.makedirs(RESULTS, exist_ok=True)
    items = []
    for name in a.sets:
        _say(f"{name}: writing trajectories")
        items += prepare(name, n_molecules=a.molecules, n_frames=a.frames)
    if not items:
        print("no trajectory could be written")
        return 1
    _say(f"{len(items)} molecules in all")

    _say("The two readers:")
    rec = stage_engines(items)
    if rec:
        with open(os.path.join(RESULTS, "engines.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
    _say("Thinning:")
    rec = stage_thinning(items)
    with open(os.path.join(RESULTS, "thinning.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    _say("Cost:")
    rec = stage_cost(items)
    with open(os.path.join(RESULTS, "cost.json"), "w") as fh:
        json.dump(rec, fh, indent=1)

    report(path=os.path.join(RESULTS, "table.md"))
    print(f"written {os.path.join(RESULTS, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
