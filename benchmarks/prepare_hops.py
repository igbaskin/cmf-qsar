#!/usr/bin/env python
"""
Prepare the public hop set: ternary compounds of lithium and oxygen (Li, O and one more element) of the Materials Project with bond-valence
barriers on every symmetry-distinct Li hop (plan item BT7).

The structures come through the OPTIMADE interface of the Materials Project, which needs no
key (``https://optimade.materialsproject.org``): every ordered ternary Li-M-O compound with 6 to
30 sites, with its energy above the convex hull.  This script keeps the ones within
``--hull`` eV/atom of the hull, enumerates the Li hops of each (1.5 to 4.5 A), reduces them by
symmetry, labels the distinct ones with the bond-valence site-energy barrier (``bvlain``) and
writes everything ``benchmarks/run_hops.py`` needs to ``benchmarks/results/hops/prepared_mp.json``,
so that the benchmark runs without the network and without ``bvlain``.

Usage
-----
    python benchmarks/prepare_hops.py                   # fetch, label, write
    python benchmarks/prepare_hops.py --max 50          # a quick look
    python benchmarks/prepare_hops.py --no-fetch        # reuse benchmarks/cache/hops/mp_li_oxides.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar.hops import HopSet, bvse_barriers  # noqa: E402
from cmfqsar.periodic import inorganic_fieldset, oxidation_states  # noqa: E402

CACHE = os.path.join(HERE, "cache", "hops", "mp_li_oxides.json")
OUT = os.path.join(HERE, "results", "hops", "prepared_mp.json")
BASE = "https://optimade.materialsproject.org/v1/structures"
FILTER = 'elements HAS "Li" AND elements HAS "O" AND nelements=3 AND nsites<=30 AND nsites>=6'
FIELDS = ("chemical_formula_reduced,nsites,lattice_vectors,cartesian_site_positions,"
          "species_at_sites,species,_mp_stability")


def fetch() -> list:
    rows, offset = [], 0
    while True:
        url = BASE + "?" + urllib.parse.urlencode({"filter": FILTER, "page_limit": 100,
                                                   "page_offset": offset,
                                                   "response_fields": FIELDS})
        req = urllib.request.Request(url, headers={"User-Agent": "cmfqsar-hops/0.1"})
        for attempt in range(4):
            try:
                d = json.load(urllib.request.urlopen(req, timeout=120))
                break
            except Exception as ex:
                print("retry", attempt, ex, flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit("the Materials Project did not answer")
        for e in d["data"]:
            a = e["attributes"]
            st = a.get("_mp_stability") or {}
            hull = None
            for k in ("gga_gga+u_r2scan", "gga_gga+u", "r2scan"):
                if k in st and st[k].get("energy_above_hull") is not None:
                    hull = st[k]["energy_above_hull"]
                    break
            if any(len(s.get("chemical_symbols", [])) != 1 for s in a["species"]):
                continue
            rows.append({"id": e["id"], "formula": a["chemical_formula_reduced"],
                         "nsites": a["nsites"], "hull": hull, "cell": a["lattice_vectors"],
                         "positions": a["cartesian_site_positions"],
                         "symbols": a["species_at_sites"]})
        print(f"  fetched {len(rows)}", flush=True)
        if not d.get("links", {}).get("next") or not d["data"]:
            break
        offset += 100
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump({"filter": FILTER, "source": BASE, "fetched": time.strftime("%Y-%m-%d"),
               "structures": rows}, open(CACHE, "w"))
    return rows


def to_atoms(row: dict):
    from ase import Atoms
    return Atoms(symbols=row["symbols"], positions=np.asarray(row["positions"], float),
                 cell=np.asarray(row["cell"], float), pbc=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hull", type=float, default=0.05,
                    help="keep structures within this energy of the hull, eV/atom (default 0.05)")
    ap.add_argument("--max", type=int, default=0, help="stop after this many structures")
    ap.add_argument("--no-fetch", action="store_true", help="use the cached download")
    ap.add_argument("--resolution", type=float, default=0.25,
                    help="grid spacing of the bond-valence landscape, A (default 0.25)")
    ap.add_argument("--max-label", type=int, default=12,
                    help="label at most this many distinct hops per structure, drawn at random "
                         "(default 12); a structure with more keeps the rest unlabelled, and the "
                         "percolation check of the benchmark uses only fully labelled structures")
    ap.add_argument("--spacing", type=float, default=0.25,
                    help="grid spacing of the box around each path, A (default 0.25)")
    ap.add_argument("--pad", type=float, default=2.0,
                    help="how far the box reaches beyond the sites and to the sides, A (default 2)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if args.no_fetch and os.path.exists(CACHE):
        rows = json.load(open(CACHE))["structures"]
    else:
        rows = fetch()
    rows = [r for r in rows if r["hull"] is not None and r["hull"] <= args.hull]
    rows.sort(key=lambda r: r["id"])
    if args.max:
        rows = rows[:args.max]
    print(f"{len(rows)} structures within {args.hull} eV/atom of the hull", flush=True)

    fs = inorganic_fieldset([], physchem=("steric",))      # geometry only: for the symmetry step
    out, skipped = [], []
    rng = np.random.RandomState(0)
    t0 = time.time()
    for n, row in enumerate(rows):
        atoms = to_atoms(row)
        ox = oxidation_states(atoms.get_chemical_symbols())
        if ox is None or ox.get("Li", 0) != 1:
            skipped.append((row["id"], "no oxidation states"))
            continue
        try:
            hs = HopSet.build([atoms], [row["id"]], "Li")
            if not hs.hops:
                skipped.append((row["id"], "no Li hops between 1.5 and 4.5 A"))
                continue
            reps = hs.annotate(hs.crystals(fs))
            label = reps
            if args.max_label and len(reps) > args.max_label:
                label = sorted(rng.choice(reps, args.max_label, replace=False).tolist())
            perc = bvse_barriers(hs, rows=label, oxidation=ox, resolution=args.resolution,
                                 spacing=args.spacing, pad=args.pad, grid_percolation=False)
        except Exception as ex:
            skipped.append((row["id"], str(ex)[:100]))
            continue
        hops = [{"a": h.a, "b": h.b, "offset": list(h.offset), "length": round(h.length, 4),
                 "y": h.y, "y_reverse": h.y_reverse, "site_a": h.site_a, "site_b": h.site_b,
                 "saddle": h.saddle, "bottleneck": h.bottleneck,
                 "bottleneck_atoms": h.bottleneck_atoms, "representative": h.representative,
                 "multiplicity": h.multiplicity}
                for h in hs.hops]
        bad = sum(1 for h in hops if not np.isfinite(h["y"]))
        for h in hops:                                   # JSON has no NaN
            for key in ("y", "y_reverse", "site_a", "site_b", "saddle", "bottleneck"):
                if not np.isfinite(h[key]):
                    h[key] = None
        out.append({"id": row["id"], "formula": row["formula"], "hull": row["hull"],
                    "cell": row["cell"], "positions": row["positions"], "symbols": row["symbols"],
                    "oxidation": ox, "percolation": perc.get(row["id"], {}), "hops": hops,
                    "n_distinct": len(reps)})
        print(f"  {n + 1:4d}/{len(rows)} {row['id']:12s} {row['formula']:14s} {len(hops):4d} hops, "
              f"{len(reps):3d} distinct, {bad} unlabelled, {time.time() - t0:.0f} s", flush=True)
        if (len(out) % 20 == 0) or n == len(rows) - 1:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            json.dump({"source": BASE, "filter": FILTER, "hull": args.hull,
                       "resolution": args.resolution, "spacing": args.spacing, "pad": args.pad,
                       "max_label": args.max_label, "prepared": time.strftime("%Y-%m-%d"),
                       "structures": out, "skipped": skipped}, open(args.out, "w"))
    print(f"{len(out)} structures prepared, {len(skipped)} skipped, "
          f"{sum(len(s['hops']) for s in out)} hops, "
          f"{sum(s['n_distinct'] for s in out)} distinct; written to {args.out}")


if __name__ == "__main__":
    main()
