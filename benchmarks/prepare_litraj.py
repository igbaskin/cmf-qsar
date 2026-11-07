#!/usr/bin/env python
"""
Prepare a sample of the published nebBVSE122k hop benchmark, as the external check of plan item BT7.

nebBVSE122k is one part of the LiTraj collection of Dembitskiy et al., npj Computational
Materials 11, 131 (2025), published under the MIT licence at
https://github.com/AIRI-Institute/LiTraj.  It holds 122,421 bond-valence migration barriers of
single Li hops in 11,290 crystal structures of the Materials Project, over 1,644 chemical
systems, with a split that holds out whole structures.  Its label is the same quantity item BT7
predicts -- the barrier one Li meets between two sites of a rigid framework, from the
bond-valence site-energy model -- computed by relaxing a straight path with the nudged elastic
band method rather than by the minimax path on a grid that
:func:`cmfqsar.hops.bvse_barriers` takes.  The set is therefore the one place where the models
of item BT7 can be compared with published numbers on the same kind of label.

Two differences from the stored public set of item BT7 are worth naming.  The collection covers
every chemistry that holds lithium, not ternary lithium oxides alone, and its barriers spread
2.4 times as widely (standard deviation 1.97 eV against 0.81 eV), so the same absolute error
gives a much higher q2 on it.  Both are reported with the result.

A kernel model holds one number per pair of samples, so 122,421 hops are out of reach: this
script draws a sample of structures and, from each, a few of its hops, which is how the public
set of item BT7 was drawn as well (at most six distinct hops per structure).  Only the index of
the collection is needed, since it carries the barrier of every hop; the structures themselves
come from the OPTIMADE interface of the Materials Project, which needs no key, and the atom
indices of an ``edge_id`` are indices into that structure.

Usage
-----
    python benchmarks/prepare_litraj.py --fetch      # download the index, fetch structures, write
    python benchmarks/prepare_litraj.py              # reuse what is already downloaded
    python benchmarks/prepare_litraj.py --structures 50   # a quick look

The archive the index comes in is 191 MB and is not ours to redistribute, so it is downloaded
into ``benchmarks/cache/litraj``, which is not in the repository; the prepared sample this script
writes is, so the benchmark runs without the network.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar.dataio import relative_path  # noqa: E402
from cmfqsar.periodic import oxidation_states  # noqa: E402

CACHE = os.path.join(HERE, "cache", "litraj")
INDEX = os.path.join(CACHE, "nebBVSE122k", "nebBVSE122k_index.csv")
STRUCTURES = os.path.join(CACHE, "mp_structures.json")
OUT = os.path.join(HERE, "results", "litraj", "prepared.json")
ARCHIVE = "https://bs3u.obs.ru-moscow-1.hc.sbercloud.ru/litraj/nebBVSE122k.zip"
SOURCE = "https://github.com/AIRI-Institute/LiTraj"
BASE = "https://optimade.materialsproject.org/v1/structures"
FIELDS = ("chemical_formula_reduced,nsites,lattice_vectors,cartesian_site_positions,"
          "species_at_sites,species")
#: ids asked for in one OPTIMADE request
BATCH = 25
#: a hop whose length falls outside this range is left out, as a guard against an index of the
#: collection that does not mean what this script takes it to mean
LENGTH_RANGE = (1.0, 12.0)


def download_index() -> str:
    """Download the nebBVSE122k archive and take the index out of it; returns the index path."""
    os.makedirs(CACHE, exist_ok=True)
    zip_path = os.path.join(CACHE, "nebBVSE122k.zip")
    if not os.path.exists(zip_path):
        print(f"downloading {ARCHIVE} (191 MB)", flush=True)
        req = urllib.request.Request(ARCHIVE, headers={"User-Agent": "cmfqsar-hops/0.1"})
        with urllib.request.urlopen(req, timeout=600) as r, open(zip_path, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    member = "nebBVSE122k/nebBVSE122k_index.csv"
    with zipfile.ZipFile(zip_path) as z:
        z.extract(member, CACHE)
    print(f"index: {INDEX}", flush=True)
    return INDEX


def fetch_structures(ids, cache: str = STRUCTURES) -> dict:
    """The Materials Project structures of `ids`, by id, through OPTIMADE; cached on disk."""
    have = {}
    if os.path.exists(cache):
        have = json.load(open(cache))
    missing = sorted(set(ids) - set(have))
    for i in range(0, len(missing), BATCH):
        part = missing[i:i + BATCH]
        flt = " OR ".join(f'id="{m}"' for m in part)
        url = BASE + "?" + urllib.parse.urlencode({"filter": flt, "page_limit": BATCH,
                                                   "response_fields": FIELDS})
        req = urllib.request.Request(url, headers={"User-Agent": "cmfqsar-hops/0.1"})
        for attempt in range(4):
            try:
                d = json.load(urllib.request.urlopen(req, timeout=180))
                break
            except Exception as ex:
                print("retry", attempt, ex, flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit("the Materials Project did not answer")
        for e in d["data"]:
            a = e["attributes"]
            if any(len(s.get("chemical_symbols", [])) != 1 for s in a["species"]):
                continue                      # a site shared by two species: not an ordered cell
            have[e["id"]] = {"formula": a["chemical_formula_reduced"], "nsites": a["nsites"],
                             "cell": a["lattice_vectors"],
                             "positions": a["cartesian_site_positions"],
                             "symbols": a["species_at_sites"]}
        print(f"  fetched {i + len(part)}/{len(missing)}", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(have, fh)
    return have


def parse_edge(edge_id: str):
    """(source, target, offset) of an ``edge_id`` of the collection.

    An id reads ``<material id>_<source>_<target>_<offset x>_<offset y>_<offset z>``, and the two
    numbers are indices into the Materials Project structure of that material.
    """
    parts = edge_id.split("_")
    return int(parts[-5]), int(parts[-4]), tuple(int(v) for v in parts[-3:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true",
                    help="download the archive of the collection and take its index out of it")
    ap.add_argument("--index", default=INDEX, help="the index of the collection (a .csv)")
    ap.add_argument("--structures", type=int, default=400,
                    help="how many structures to draw (default 400)")
    ap.add_argument("--max-per-structure", type=int, default=6,
                    help="hops kept per structure, drawn at random (default 6, as the public set "
                         "of item BT7 keeps)")
    ap.add_argument("--seed", type=int, default=0, help="the draw (default 0)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    import pandas as pd

    index_path = download_index() if args.fetch else args.index
    if not os.path.exists(index_path):
        raise SystemExit(f"no index at {index_path}; run with --fetch to download it")
    idx = pd.read_csv(index_path)
    y_all = idx.em.values.astype(float)
    print(f"{len(idx)} hops of {idx.material_id.nunique()} structures and "
          f"{idx.chemsys.nunique()} chemical systems in the collection; "
          f"barriers mean {y_all.mean():.3f}, standard deviation {y_all.std(ddof=1):.3f}, "
          f"range {y_all.min():.3f} to {y_all.max():.3f} eV", flush=True)

    rng = np.random.RandomState(args.seed)
    ids = np.array(sorted(idx.material_id.unique()))
    drawn = sorted(rng.choice(ids, min(args.structures, len(ids)), replace=False).tolist())
    rows = idx[idx.material_id.isin(set(drawn))]
    print(f"drew {len(drawn)} structures with {len(rows)} hops", flush=True)

    have = fetch_structures(drawn)
    out, skipped, dropped = [], [], 0
    for mp_id in drawn:
        row = have.get(mp_id)
        if row is None:
            skipped.append((mp_id, "not in the Materials Project answer"))
            continue
        symbols = list(row["symbols"])
        ox = oxidation_states(symbols)
        if ox is None or ox.get("Li", 0) != 1:
            skipped.append((mp_id, "no oxidation states"))
            continue
        cell = np.asarray(row["cell"], float)
        pos = np.asarray(row["positions"], float)
        edges = rows[rows.material_id == mp_id]
        hops = []
        for edge_id, em, split in zip(edges.edge_id, edges.em, edges._split):
            a, b, off = parse_edge(edge_id)
            if max(a, b) >= len(symbols) or symbols[a] != "Li" or symbols[b] != "Li":
                dropped += 1
                continue
            vec = pos[b] + np.asarray(off, float) @ cell - pos[a]
            length = float(np.linalg.norm(vec))
            if not LENGTH_RANGE[0] <= length <= LENGTH_RANGE[1]:
                dropped += 1
                continue
            hops.append({"edge_id": edge_id, "a": a, "b": b, "offset": list(off),
                         "length": round(length, 4), "y": float(em), "split": str(split)})
        if not hops:
            skipped.append((mp_id, "no usable hop"))
            continue
        if len(hops) > args.max_per_structure:
            keep = rng.choice(len(hops), args.max_per_structure, replace=False)
            hops = [hops[i] for i in sorted(keep)]
        out.append({"id": mp_id, "formula": row["formula"], "nsites": row["nsites"],
                    "cell": row["cell"], "positions": row["positions"], "symbols": symbols,
                    "oxidation": ox, "hops": hops})
    n_hops = sum(len(r["hops"]) for r in out)
    y = np.array([h["y"] for r in out for h in r["hops"]], float)
    lengths = np.array([h["length"] for r in out for h in r["hops"]], float)
    print(f"{len(out)} structures and {n_hops} hops kept, {len(skipped)} structures and "
          f"{dropped} hops left out", flush=True)
    print(f"  barriers mean {y.mean():.3f}, standard deviation {y.std(ddof=1):.3f}, "
          f"range {y.min():.3f} to {y.max():.3f} eV", flush=True)
    print(f"  hop lengths {lengths.min():.2f} to {lengths.max():.2f} A, "
          f"mean {lengths.mean():.2f} A", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doc = {"source": SOURCE, "dataset": "nebBVSE122k", "archive": ARCHIVE,
           "reference": "Dembitskiy et al., npj Comput. Mater. 11, 131 (2025)",
           "licence": "MIT", "structures_source": BASE,
           "index": relative_path(index_path, ROOT).replace(os.sep, "/"),
           "prepared": time.strftime("%Y-%m-%d"),
           "n_index_hops": int(len(idx)), "n_index_structures": int(idx.material_id.nunique()),
           "n_index_chemsys": int(idx.chemsys.nunique()),
           "index_y_mean": float(y_all.mean()), "index_y_sd": float(y_all.std(ddof=1)),
           "drawn": len(drawn), "max_per_structure": args.max_per_structure, "seed": args.seed,
           "structures": out, "skipped": skipped, "dropped_hops": dropped}
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
