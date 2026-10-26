#!/usr/bin/env python
"""
Does complementarity with a cyclodextrin cavity predict the constant? (plan item SC3)

The complementarity score asks a one-body question: how well do a molecule's fields match what
the pocket offers where the molecule sits. Every measurement of it so far has been on a protein
set with a few dozen ligands in one pocket. A cyclodextrin table is the largest test there is:
one host, one cavity, and over a thousand guests with a measured association constant.

What is done
------------
* The host is read from a molecule file with its sugars as the residues
  (`cmfqsar.protein.read_molecule_host`), built by `examples/make_host_structures.py`, which
  keeps the most symmetric of forty conformations -- a cyclodextrin is a ring of identical
  sugars, so symmetry is the rule, and the chosen structures have cavity radii of 2.6, 3.0 and
  3.9 A for alpha, beta and gamma, against published values near 2.4, 3.1 and 4.0.
* Every guest is put in the cavity: its centroid on the cavity centre, its longest axis along the
  cavity axis, which is how a guest threads a cyclodextrin, and then the local rigid-body
  refinement of `cmfqsar.docking` inside the site.
* The score and its terms are correlated with log K.

The gate the item sets is whether the score correlates with log K at all. A one-body score cannot
contain the desolvation that drives this endpoint, so a weak result is the expected one and is
reported as the limit of the score rather than as a failure of the structures.

Usage
-----
    python benchmarks/run_host_pocket.py                 # beta, then alpha and gamma
    python benchmarks/run_host_pocket.py --hosts beta --max-guests 200
    python benchmarks/run_host_pocket.py --report
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

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.complementarity import complementarity, pocket_fields, series_scores  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.docking import dock_coords  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.protein import cavity_frame, read_molecule_host  # noqa: E402

DATA = os.path.join(ROOT, "examples", "cyclodextrin")
HOSTS = os.path.join(DATA, "hosts")
RESULTS = os.path.join(HERE, "results", "host_pocket")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 1.0
SEED = 0
FILES = {"beta": "beta_cyclodextrin", "alpha": "alpha_cyclodextrin",
         "gamma": "gamma_cyclodextrin"}


def place(mol: Chem.Mol, centre: np.ndarray, axis: np.ndarray) -> Chem.Mol:
    """Put a guest in the cavity: centroid on the centre, longest axis along the channel."""
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    pos = np.array(conf.GetPositions(), float)
    pos = pos - pos.mean(axis=0)
    if len(pos) > 2:
        _u, _s, vt = np.linalg.svd(pos, full_matrices=False)
        long_axis = vt[0]
        v = np.cross(long_axis, axis)
        c = float(np.dot(long_axis, axis))
        if np.linalg.norm(v) > 1e-8:
            k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            R = np.eye(3) + k + k @ k * (1.0 / (1.0 + c)) if c > -1 + 1e-8 else -np.eye(3)
            pos = pos @ R.T
    pos = pos + centre
    for i in range(out.GetNumAtoms()):
        conf.SetAtomPosition(i, [float(x) for x in pos[i]])
    return out


def run_host(host_key, results=RESULTS, force=False, max_guests=0, restarts=2):
    out_path = os.path.join(results, f"{host_key}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    t0 = time.time()
    stem = FILES[host_key]
    host = read_molecule_host(os.path.join(HOSTS, stem + ".sdf"))
    centre, axis = cavity_frame(host)
    fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA, charge_model="gasteiger",
                         param_scheme="rdkit")
    pocket = pocket_fields(host, fs)
    ds = load_dataset(os.path.join(DATA, stem + ".csv"))
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
    print(f"\n=== {host_key}: {host.n_residues} units, {host.n_atoms} atoms; "
          f"{len(mols)} guests", flush=True)
    posed, moved, clashes = [], [], []
    for i, (m, nm) in enumerate(zip(mols, names)):
        guest = place(m, centre, axis)
        mf = build_fields(guest, fs, nm)
        coords, res = dock_coords(mf, pocket, fs, restarts=restarts, seed=SEED)
        mf.coords = coords
        posed.append(mf)
        moved.append(res.moved)
        clashes.append(res.clash_after)
        if (i + 1) % 100 == 0:
            print(f"    posed {i + 1}/{len(mols)} ({time.time() - t0:.0f} s)", flush=True)
    scores, stats = series_scores(posed, pocket, fs, activity=y)
    heavy = np.array([m.GetNumHeavyAtoms() for m in mols], float)
    score = np.array([c.score for c in scores], float)
    burial = np.array([c.burial for c in scores], float)
    res = {"host": host_key, "units": host.n_residues, "n": len(mols),
           "fields": FIELDS, "alpha": ALPHA, "restarts": restarts,
           "moved_median": float(np.median(moved)),
           "clash_median": float(np.median(clashes)),
           "r_score": stats.get("r_score"), "r_burial": stats.get("r_burial"),
           "terms": {k[2:]: v for k, v in stats.items()
                     if k.startswith("r_") and k not in ("r_score", "r_burial")},
           "r_heavy_atoms": float(np.corrcoef(heavy, y)[0, 1]),
           "r_score_heavy": float(np.corrcoef(heavy, score)[0, 1]),
           "r_burial_heavy": float(np.corrcoef(heavy, burial)[0, 1]),
           "score_mean": float(np.nanmean(score)), "burial_mean": float(np.nanmean(burial)),
           "seconds": time.time() - t0}
    # the size of the guest explains much of a one-body score, so the question is what is left
    res["partial"] = partial_correlation(score, y, heavy)
    res["partial_burial"] = partial_correlation(burial, y, heavy)
    print(f"    complementarity against log K: r {res['r_score']:+.3f}; "
          f"burial r {res['r_burial']:+.3f}", flush=True)
    print(f"    the guest's heavy-atom count against log K: r {res['r_heavy_atoms']:+.3f}; "
          f"with it held constant the score gives {res['partial']:+.3f} and burial "
          f"{res['partial_burial']:+.3f}", flush=True)
    for k, v in sorted(res["terms"].items(), key=lambda kv: -abs(kv[1] or 0.0)):
        print(f"      {k:16s} r {v:+.3f}", flush=True)
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out_path} ({res['seconds']:.0f} s)", flush=True)
    return res


def partial_correlation(a, y, z):
    """Correlation of `a` with `y` once `z` is taken out of both, which is what it adds."""
    a, y, z = np.asarray(a, float), np.asarray(y, float), np.asarray(z, float)
    ok = np.isfinite(a) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 5:
        return float("nan")
    a, y, z = a[ok], y[ok], z[ok]
    Z = np.column_stack([np.ones(len(z)), z])
    ra = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if np.std(ra) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, ry)[0, 1])


def report(results=RESULTS):
    rows = []
    if os.path.isdir(results):
        for fn in sorted(os.listdir(results)):
            if fn.endswith(".json"):
                rows.append(json.load(open(os.path.join(results, fn))))
    order = {"beta": 0, "alpha": 1, "gamma": 2}
    rows.sort(key=lambda d: order.get(d["host"], 9))
    out = ["## A cyclodextrin cavity as a pocket", "",
           "The complementarity score on the largest set it has had: one host, one cavity and "
           "over a thousand guests with a measured constant. The host is a molecule file whose "
           "sugars are its residues; every guest is put in the cavity, centroid on the centre "
           "and longest axis along the channel, and refined there by the local rigid-body "
           "placement. `r` is the Pearson correlation with log K over the guests.", ""]
    if rows:
        out += ["| host | units | guests | score vs log K | burial vs log K | heavy atoms vs "
                "log K | score, size held constant | burial, size held constant |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for d in rows:
            out.append(f"| {d['host']} | {d['units']} | {d['n']} | {d['r_score']:+.3f} | "
                       f"{d['r_burial']:+.3f} | {d['r_heavy_atoms']:+.3f} | "
                       f"{d['partial']:+.3f} | {d['partial_burial']:+.3f} |")
        out += ["", "| host | " + " | ".join(sorted(rows[0]["terms"])) + " |",
                "| --- | " + " | ".join("---:" for _ in rows[0]["terms"]) + " |"]
        for d in rows:
            out.append(f"| {d['host']} | "
                       + " | ".join(f"{d['terms'][k]:+.3f}" for k in sorted(d["terms"])) + " |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", nargs="*", default=["beta", "alpha", "gamma"])
    ap.add_argument("--max-guests", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    os.makedirs(RESULTS, exist_ok=True)
    for host in a.hosts:
        run_host(host, force=a.force, max_guests=a.max_guests, restarts=a.restarts)
    print()
    print(report())


if __name__ == "__main__":
    main()
