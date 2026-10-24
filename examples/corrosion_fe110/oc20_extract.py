"""Pull the systems on one pure-metal surface out of the OC20 IS2RE LMDBs into a compact file.

Runs in the WSL environment (torch and torch_geometric unpickle the LMDB records); the result
is a .npz that the kernel experiments read on any Python:

    python examples/corrosion_fe110/oc20_extract.py --metal Fe \\
        --lmdb ~/data/oc20/is2re/all/train/data.lmdb ~/data/oc20/is2re/all/val_id/data.lmdb ... \\
        --mapping ~/data/oc20/oc20_data_mapping.pkl -o examples/corrosion_fe110/oc20_Fe.npz

For every system whose bulk is the pure metal and whose relaxation is not flagged anomalous:
the DFT-relaxed positions of the adsorbate atoms (tag 2) in the surface frame (z = 0 at the
highest surface atom, tag 1; x, y centred on the adsorbate's centroid), their elements, the
relaxed adsorption energy y_relaxed (eV, OC20's reference: clean slab plus gas-phase
references), the adsorbate formula, the Miller index and the system id.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys

import numpy as np


def elements(symbols: str):
    return set(re.findall(r"[A-Z][a-z]?", symbols))


def read_lmdb(path: str):
    import lmdb
    env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
    with env.begin() as txn:
        with txn.cursor() as cur:                 # the OC20 LMDBs have no length key: walk them
            for key, raw in cur:
                if key == b"length":
                    continue
                try:
                    d = pickle.loads(raw)
                except Exception:               # noqa: BLE001
                    continue
                # OC20's records were pickled by an old PyG whose attributes live in the instance
                # dictionary; the modern accessors refuse them, the dictionary does not
                rec = dict(getattr(d, "__dict__", {}))
                if "pos_relaxed" in rec:
                    yield rec
    env.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metal", required=True)
    ap.add_argument("--lmdb", nargs="+", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--max", type=int, default=0, help="stop after this many systems (0 = all)")
    a = ap.parse_args(argv)
    mapping = pickle.load(open(os.path.expanduser(a.mapping), "rb"))
    wanted = {sid for sid, v in mapping.items()
              if elements(v["bulk_symbols"]) == {a.metal} and v.get("anomaly", 0) == 0}
    print(f"{len(wanted)} non-anomalous systems on pure {a.metal} in the mapping", flush=True)
    out = []
    for path in a.lmdb:
        path = os.path.expanduser(path)
        n_seen = n_hit = 0
        for d in read_lmdb(path):
            n_seen += 1
            sid = f"random{int(d['sid'])}"
            if sid not in wanted:
                continue
            pos = np.asarray(d["pos_relaxed"], float)
            z = np.asarray(d["atomic_numbers"], int)
            tags = np.asarray(d["tags"], int)
            ads = tags == 2
            surf = tags == 1
            if ads.sum() == 0 or surf.sum() == 0:
                continue
            z_top = pos[surf, 2].max()
            xyz = pos[ads].copy()
            xyz[:, 2] -= z_top
            xyz[:, :2] -= xyz[:, :2].mean(axis=0)
            meta = mapping[sid]
            out.append({"sid": sid, "y": float(d["y_relaxed"]), "z": z[ads], "xyz": xyz,
                        "ads": meta["ads_symbols"], "miller": tuple(int(k) for k in meta["miller_index"]),
                        "split": meta.get("split", ""), "n_slab": int((~ads).sum())})
            n_hit += 1
            if a.max and len(out) >= a.max:
                break
        print(f"{path}: {n_seen} records read, {n_hit} on {a.metal}", flush=True)
        if a.max and len(out) >= a.max:
            break
    n = len(out)
    np.savez_compressed(a.out,
                        sid=np.array([o["sid"] for o in out]), y=np.array([o["y"] for o in out]),
                        ads=np.array([o["ads"] for o in out]), split=np.array([o["split"] for o in out]),
                        miller=np.array([o["miller"] for o in out]).reshape(n, 3),
                        n_atoms=np.array([len(o["z"]) for o in out]),
                        numbers=np.concatenate([o["z"] for o in out]) if out else np.zeros(0, int),
                        xyz=np.concatenate([o["xyz"] for o in out]) if out else np.zeros((0, 3)),
                        n_slab=np.array([o["n_slab"] for o in out]))
    ys = np.array([o["y"] for o in out])
    print(f"-> {a.out}: {n} systems, {len(set(o['ads'] for o in out))} adsorbates, E_ads {ys.min():.2f}..{ys.max():.2f} eV "
          f"(sd {ys.std():.2f})" if n else "no systems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
