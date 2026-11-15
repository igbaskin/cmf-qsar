#!/usr/bin/env python
"""
Rigid porous hosts as pockets: the periodic route against the finite host (plan item M7).

`cmfqsar/frameworks.py` reads a framework with its periodic images, finds its pores, and cuts the
images around a pore into a finite pocket, so that complementarity, the interaction profile and
the rigid-body placement of the pocket code work on a zeolite or a MOF as they do on a protein.
The plan sets one test: the cyclodextrin example (`examples/cyclodextrin`) is the finite host
that the periodic route must reproduce as a limit.

Stage limit (the gate)
----------------------
Each cyclodextrin host (alpha, beta, gamma) is put alone in a cubic box at its own coordinates.
The box is a framework whose images are a box length apart. The pocket is cut at 12 A from the
cavity centre, the frame is the cavity frame of the host's sugars (as in
`run_host_pocket.py`), and every guest of the host's table is posed in it. For box sides from
just above the host's size upwards, the complementarity score of every guest is compared with
the finite host's score: before the images leave the cut, they change the scores; after, the
gate asks that every score be the finite host's to rounding (1e-12). The same is then asked of
the rigid-body placement (`dock_coords`) for every tenth guest of beta-cyclodextrin: the poses
and the scores after placement must agree to rounding.

Stage zeolites (the pore finder on real frameworks)
---------------------------------------------------
Three frameworks of the IZA Structure Commission database (LTA, FAU, MFI; fetched by stage
fetch, not stored in the repository). The largest sphere that can be included is compared with
the value the database gives on each framework page, for the oxygen radius that reproduces it
and for the van der Waals radius. Then four small guests are posed at the largest pore of each
and scored against the framework pocket, with the profile resolved by framework atom type. That
part is a demonstration that the ligand path runs on a framework: it has no measured property.

Usage
-----
    python benchmarks/run_frameworks.py --stage fetch      # three CIF files from the IZA site
                                                           # (stage zeolites fetches them too)
    python benchmarks/run_frameworks.py --stage limit      # about 10 minutes
    python benchmarks/run_frameworks.py --stage zeolites   # about 1 minute
    python benchmarks/run_frameworks.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from cmfqsar import frameworks as F  # noqa: E402
from cmfqsar.complementarity import (complementarity, interaction_profile,  # noqa: E402
                                     pocket_fields, pocket_residues)
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.docking import dock_coords  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.protein import MONOMER_RESNAME, cavity_frame, read_molecule_host  # noqa: E402

import run_host_pocket as HP  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "frameworks")
CACHE = os.path.join(HERE, "cache", "frameworks")
IZA_CIF = "https://europe.iza-structure.org/IZA-SC/cif/{code}.cif"
#: the largest sphere that can be included, in angstrom, as the framework pages of the IZA
#: database give it (read 15 November 2026)
IZA_INCLUDED = {"LTA": 11.05, "FAU": 11.24, "MFI": 6.36}
#: the oxygen radius that reproduces those values (the pages do not state one), and the van der
#: Waals radius beside it
O_RADII = (1.32, 1.52)
HOSTS = ("alpha", "beta", "gamma")
#: box sides, in angstrom; the smallest is just above the largest host
BOXES = (18.0, 19.0, 20.0, 21.0, 22.0, 24.0, 28.0, 32.0)
CUT = 12.0
DOCK_EVERY = 10
EXACT = 1e-12
GUESTS = {"methane": "C", "benzene": "c1ccccc1", "methanol": "CO", "carbon dioxide": "O=C=O"}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fieldset() -> FieldSet:
    return FieldSet.create(physchem=HP.FIELDS, alpha=HP.ALPHA, charge_model="gasteiger",
                           param_scheme="rdkit")


# --------------------------------------------------------------------------- #
# Stage fetch
# --------------------------------------------------------------------------- #

def stage_fetch(force: bool = False) -> None:
    """Download the three framework files of the IZA database into the git-ignored cache."""
    import urllib.request

    os.makedirs(CACHE, exist_ok=True)
    for code in IZA_INCLUDED:
        path = os.path.join(CACHE, f"{code}.cif")
        if os.path.isfile(path) and not force:
            log(f"{path} is there")
            continue
        with urllib.request.urlopen(IZA_CIF.format(code=code), timeout=60) as r:
            data = r.read()
        if not data.startswith(b"data_"):
            raise RuntimeError(f"{code}: the server did not return a CIF file")
        with open(path, "wb") as fh:
            fh.write(data)
        log(f"{code}: {len(data)} bytes to {path}")


# --------------------------------------------------------------------------- #
# Stage limit: the cyclodextrin in a box against the finite cyclodextrin
# --------------------------------------------------------------------------- #

def host_guests(key: str):
    host = read_molecule_host(os.path.join(HP.HOSTS, HP.FILES[key] + ".sdf"))
    ds = load_dataset(os.path.join(HP.DATA, HP.FILES[key] + ".csv"))
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    return host, mols, names, y[ok]


def boxed_pocket(host, box: float, centre: np.ndarray, fs: FieldSet):
    atoms = F.boxed_host(host.coords, host.elements, box)
    pocket = F.framework_pocket(atoms, centre, radius=CUT, units=host.resids - 1,
                                unit_name=MONOMER_RESNAME, name=host.name)
    fields = pocket_fields(pocket, fs)        # the finite host's own route, defaults and all
    return pocket, fields


def limit_host(key: str) -> Dict[str, object]:
    t0 = time.time()
    fs = fieldset()
    host, mols, names, y = host_guests(key)
    centre, axis = cavity_frame(host)
    ref_pocket = pocket_fields(host, fs)
    posed = [build_fields(HP.place(m, centre, axis), fs, n) for m, n in zip(mols, names)]
    ref = np.array([complementarity(g, ref_pocket, fs).score for g in posed])
    ref_terms = [complementarity(g, ref_pocket, fs).terms for g in posed]
    span = float(np.max(host.coords.max(axis=0) - host.coords.min(axis=0)))
    reach = float(np.max(np.linalg.norm(host.coords - centre, axis=1)))
    rows = []
    for box in BOXES:
        if box <= span:
            continue
        pocket, fields = boxed_pocket(host, box, centre, fs)
        sc = np.array([complementarity(g, fields, fs).score for g in posed])
        terms = [complementarity(g, fields, fs).terms for g in posed]
        dterm = max(abs(t[k] - r[k]) for t, r in zip(terms, ref_terms) for k in r)
        n_img = int(np.sum(np.any(pocket.source_image != 0, axis=1)))
        rows.append({"box": box, "pocket_atoms": int(pocket.n_atoms), "image_atoms": n_img,
                     "max_score_difference": float(np.max(np.abs(sc - ref))),
                     "max_term_difference": float(dterm),
                     "r_score_logK": float(np.corrcoef(sc, y)[0, 1]),
                     "same_fields": bool(fields.n_atoms == ref_pocket.n_atoms
                                         and np.array_equal(fields.coords, ref_pocket.coords)
                                         and np.array_equal(fields.weights,
                                                            ref_pocket.weights))})
        log(f"  {key} box {box:g}: {pocket.n_atoms} pocket atoms ({n_img} from images), "
            f"largest score difference {rows[-1]['max_score_difference']:.2e}")
    exact = [r["box"] for r in rows if r["max_score_difference"] <= EXACT]
    return {"host": key, "atoms": int(host.n_atoms), "units": int(host.n_residues),
            "guests": len(posed), "span": span, "reach": reach, "cut": CUT,
            "r_score_logK_finite": float(np.corrcoef(ref, y)[0, 1]),
            "boxes": rows, "exact_from": min(exact) if exact else None,
            "expected_from": reach + CUT, "seconds": round(time.time() - t0, 1)}


def limit_docking(key: str = "beta", box: float = 32.0) -> Dict[str, object]:
    """The rigid-body placement in the box against the placement in the finite host."""
    t0 = time.time()
    fs = fieldset()
    host, mols, names, y = host_guests(key)
    centre, axis = cavity_frame(host)
    ref_pocket = pocket_fields(host, fs)
    _pocket, fields = boxed_pocket(host, box, centre, fs)
    small = min(b for b in BOXES if b > float(np.max(host.coords.max(0) - host.coords.min(0))))
    _p2, near = boxed_pocket(host, small, centre, fs)
    pick = list(range(0, len(mols), DOCK_EVERY))
    d_pose, d_score, d_small = 0.0, 0.0, []
    for i in pick:
        g = build_fields(HP.place(mols[i], centre, axis), fs, names[i])
        c1, _ = dock_coords(g, ref_pocket, fs, restarts=2, seed=HP.SEED)
        c2, _ = dock_coords(g, fields, fs, restarts=2, seed=HP.SEED)
        c3, _ = dock_coords(g, near, fs, restarts=2, seed=HP.SEED)
        d_pose = max(d_pose, float(np.max(np.abs(c1 - c2))))
        g1, g2, g3 = (build_fields(HP.place(mols[i], centre, axis), fs, names[i])
                      for _ in range(3))
        g1.coords, g2.coords, g3.coords = c1, c2, c3
        s1 = complementarity(g1, ref_pocket, fs).score
        s2 = complementarity(g2, fields, fs).score
        s3 = complementarity(g3, near, fs).score
        d_score = max(d_score, abs(s1 - s2))
        d_small.append(abs(s1 - s3))
    log(f"  {key} docking, {len(pick)} guests: box {box:g} pose difference {d_pose:.1e} A, "
        f"score {d_score:.1e}; box {small:g} score difference up to {max(d_small):.3f}")
    return {"host": key, "box": box, "guests": len(pick), "max_pose_difference": d_pose,
            "max_score_difference": d_score, "small_box": small,
            "small_box_max_score_difference": float(max(d_small)),
            "small_box_mean_score_difference": float(np.mean(d_small)),
            "seconds": round(time.time() - t0, 1)}


def stage_limit() -> Dict[str, object]:
    out = {"hosts": [limit_host(k) for k in HOSTS], "docking": limit_docking()}
    out["gate"] = bool(all(h["exact_from"] is not None for h in out["hosts"])
                       and out["docking"]["max_pose_difference"] <= 1e-9
                       and out["docking"]["max_score_difference"] <= EXACT)
    jdump(out, os.path.join(RESULTS, "limit.json"))
    return out


# --------------------------------------------------------------------------- #
# Stage zeolites: the pores of three real frameworks, and guests in one
# --------------------------------------------------------------------------- #

def guest(smiles: str, name: str):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=0)
    if m.GetNumAtoms() > 1:
        AllChem.MMFFOptimizeMolecule(m)
    m.SetProp("_Name", name)
    return m


def stage_zeolites() -> Dict[str, object]:
    fs = fieldset()
    out = []
    for code, published in IZA_INCLUDED.items():
        path = os.path.join(CACHE, f"{code}.cif")
        if not os.path.isfile(path):
            stage_fetch()                  # the files are fetched, not kept in the repository
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = F.read_framework(path)
        t0 = time.time()
        found = {}
        for r_o in O_RADII:
            pores = F.find_pores(atoms, spacing=0.4, radii={"O": r_o, "Si": -1.0},
                                 min_radius=1.5)
            distinct = []
            for p in pores:
                if not any(abs(p.diameter - d["diameter"]) < 0.05 and p.kind == d["kind"]
                           for d in distinct):
                    distinct.append({"diameter": round(p.diameter, 3), "kind": p.kind,
                                     "anisotropy": round(p.anisotropy, 2)})
            found[str(r_o)] = {"largest": pores[0].diameter, "pores": len(pores),
                               "distinct": distinct}
        pores = F.find_pores(atoms, spacing=0.4, radii={"O": O_RADII[0], "Si": -1.0},
                             min_radius=1.5)
        big = pores[0]
        types = F.atom_types(atoms)
        pocket = F.framework_pocket(atoms, big.centre, radius=CUT, types=types, name=code)
        fields = F.framework_pocket_fields(pocket, fs, atoms=atoms, charges="oxidation")
        labels, order = pocket_residues(pocket, fields)
        scored = []
        for gname, smi in GUESTS.items():
            m = guest(smi, gname)
            placed = HP.place(m, big.centre, big.axis)
            g = build_fields(placed, fs, gname)
            coords, res = dock_coords(g, fields, fs, restarts=2, seed=0)
            g.coords = coords
            c = complementarity(g, fields, fs)
            prof = interaction_profile(g, fields, fs, labels, order)
            by_type = F.type_profile(prof, pocket)
            totals = prof.total()
            sums = {lab: sum(v[lab] for v in by_type.values()) for lab in prof.labels}
            # a term the framework cannot have (no donor in pure silica) is undefined, not zero
            defined = [k for k in totals if np.isfinite(totals[k])]
            scored.append({"guest": gname, "score": c.score, "burial": c.burial,
                           "terms": c.terms, "clashes": c.clashes, "moved": res.moved,
                           "by_type": by_type,
                           "undefined_terms": sorted(k for k in totals if k not in defined),
                           "profile_sum_error": max(abs(sums[k] - totals[k])
                                                    for k in defined)})
        out.append({"code": code, "atoms": len(atoms), "cell": atoms.cell.cellpar().tolist(),
                    "types": sorted(set(types)), "published_included": published,
                    "found": found, "largest_pore": {"diameter": big.diameter,
                                                     "kind": big.kind,
                                                     "fractional": big.fractional.tolist()},
                    "pocket_atoms": int(pocket.n_atoms), "guests": scored,
                    "seconds": round(time.time() - t0, 1)})
        log(f"{code}: largest sphere {found[str(O_RADII[0])]['largest']:.3f} A with O "
            f"{O_RADII[0]} (published {published}), "
            f"{found[str(O_RADII[1])]['largest']:.3f} A with O {O_RADII[1]}")
    jdump({"frameworks": out}, os.path.join(RESULTS, "zeolites.json"))
    return {"frameworks": out}


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def report(path: Optional[str] = None) -> str:
    lines = ["# Rigid porous hosts as pockets (plan item M7)", ""]
    lp = os.path.join(RESULTS, "limit.json")
    if os.path.isfile(lp):
        L = jload(lp)
        lines += ["## The finite host as the limit of the periodic one", "",
                  f"Each cyclodextrin alone in a cubic box, the pocket cut at {CUT:g} A from the "
                  "cavity centre, every guest posed in the cavity frame of the host's sugars. "
                  "The difference is the largest over the guests of the complementarity score "
                  "through the periodic route minus the score against the finite host.", "",
                  "| Host | Atoms | Guests | Box side, A | Pocket atoms | From images | "
                  "Largest score difference | r(score, log K) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for h in L["hosts"]:
            for r in h["boxes"]:
                lines.append(f"| {h['host']} | {h['atoms']} | {h['guests']} | {r['box']:g} | "
                             f"{r['pocket_atoms']} | {r['image_atoms']} | "
                             f"{r['max_score_difference']:.1e} | {r['r_score_logK']:+.3f} |")
            lines.append(f"| {h['host']} | {h['atoms']} | {h['guests']} | finite | "
                         f"{h['atoms']} | 0 | 0 | {h['r_score_logK_finite']:+.3f} |")
        lines += ["", "| Host | Exact (1e-12) from a box of, A | Host reach from the centre "
                  "plus the cut, A |", "|---|---:|---:|"]
        for h in L["hosts"]:
            lines.append(f"| {h['host']} | {h['exact_from']:g} | {h['expected_from']:.1f} |")
        d = L["docking"]
        lines += ["", f"Rigid-body placement of {d['guests']} {d['host']}-cyclodextrin guests "
                  f"(every tenth): in a box of {d['box']:g} A the poses differ from the finite "
                  f"host's by at most {d['max_pose_difference']:.1e} A and the scores by "
                  f"{d['max_score_difference']:.1e}; in a box of {d['small_box']:g} A, where "
                  f"images are inside the cut, the scores differ by up to "
                  f"{d['small_box_max_score_difference']:.3f} (mean "
                  f"{d['small_box_mean_score_difference']:.3f}). Gate met: "
                  f"{'yes' if L['gate'] else 'no'}.", ""]
    zp = os.path.join(RESULTS, "zeolites.json")
    if os.path.isfile(zp):
        Z = jload(zp)["frameworks"]
        lines += ["## Pores of three zeolites", "",
                  "The largest sphere that can be included, found as the largest maximum of "
                  "the free distance (silicon left out, oxygen at the radius shown), against "
                  "the value of the IZA database.", "",
                  "| Framework | Atoms | Published, A | Found, O radius 1.32 A | Found, O radius "
                  "1.52 A | Distinct pores found (diameter, kind), O radius 1.32 A |",
                  "|---|---:|---:|---:|---:|---|"]
        for z in Z:
            f1, f2 = z["found"][str(O_RADII[0])], z["found"][str(O_RADII[1])]
            dist = "; ".join(f"{d['diameter']:.2f} {d['kind']}" for d in f1["distinct"][:5])
            lines.append(f"| {z['code']} | {z['atoms']} | {z['published_included']:.2f} | "
                         f"{f1['largest']:.3f} | {f2['largest']:.3f} | {dist} |")
        lines += ["", "Guests posed at the largest pore of each framework and placed there; "
                  "framework charges from formal oxidation states. A demonstration that the "
                  "ligand path runs, not a measurement. Pure silica has no hydrogen-bond donor, "
                  "so the acceptor-to-donor term is undefined and left out.", "",
                  "| Framework | Guest | Score | Burial | Largest profile entry by atom type | "
                  "Profile sum minus score terms |", "|---|---|---:|---:|---|---:|"]
        for z in Z:
            for g in z["guests"]:
                best = max(((t, k, v) for t, row in g["by_type"].items()
                            for k, v in row.items() if np.isfinite(v)),
                           key=lambda e: abs(e[2]))
                lines.append(f"| {z['code']} | {g['guest']} | {g['score']:+.3f} | "
                             f"{g['burial']:.3f} | {best[0]} {best[1]} {best[2]:+.3f} | "
                             f"{g['profile_sum_error']:.1e} |")
        lines.append("")
    text = "\n".join(lines)
    path = path or os.path.join(RESULTS, "table.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("fetch", "limit", "zeolites"), action="append",
                    help="stage(s) to run; the report always follows")
    ap.add_argument("--force", action="store_true", help="fetch the files again")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    for st in args.stage or []:
        if st == "fetch":
            stage_fetch(force=args.force)
        elif st == "limit":
            stage_limit()
        elif st == "zeolites":
            stage_zeolites()
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
