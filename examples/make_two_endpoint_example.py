#!/usr/bin/env python
"""Merge two cyclodextrin guest tables into one two-endpoint table (plan item SC1).

``build --second-endpoint COLUMN`` reads both endpoints from one file, and the guest tables of
``examples/cyclodextrin`` are one file per host.  This script joins two of them on the canonical
SMILES of the guest, so that a molecule measured against both cavities carries both numbers:

    python examples/make_two_endpoint_example.py

writes ``examples/cyclodextrin/beta_with_alpha.csv``: every guest of the beta table, with the
alpha constant in a column of its own where the same guest appears in the alpha table.  The
primary endpoint is the one every row carries, because a molecule with no primary activity is
dropped before a model is built; the second endpoint is missing for most rows, which is the case
a two-endpoint model exists for.

It needs nothing but the files already in the repository, so the result can be rebuilt by anyone.
"""
from __future__ import annotations

import argparse
import csv
import io
import os

from rdkit import Chem, RDLogger

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "cyclodextrin")
#: the columns carried over from the primary table, in order
KEEP = ("smiles", "name", "activity", "logK", "n_values", "spread", "cid", "pH", "T", "host")


def read(path):
    """(canonical smiles -> row) for one host table, keeping the first row of a repeated guest."""
    out = {}
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            m = Chem.MolFromSmiles(r.get("smiles", ""))
            if m is None:
                continue
            try:
                float(r["activity"])
            except (TypeError, ValueError, KeyError):
                continue
            out.setdefault(Chem.MolToSmiles(m), r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primary", default="beta_cyclodextrin.csv",
                    help="the table whose activity becomes the primary endpoint")
    ap.add_argument("--second", default="alpha_cyclodextrin.csv",
                    help="the table whose activity becomes the second endpoint")
    ap.add_argument("--column", default="alpha_logK", help="the name of the second column")
    ap.add_argument("--out", default=os.path.join(DATA, "beta_with_alpha.csv"))
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    first = read(os.path.join(DATA, a.primary))
    second = read(os.path.join(DATA, a.second))
    shared = set(first) & set(second)
    rows = []
    for key, r in first.items():
        row = {c: r.get(c, "") for c in KEEP}
        row[a.column] = second[key]["activity"] if key in shared else ""
        row["second_host"] = second[key]["host"] if key in shared else ""
        rows.append(row)
    with io.open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(KEEP) + [a.column, "second_host"])
        w.writeheader()
        w.writerows(rows)
    print(f"{a.out}: {len(rows)} guests of {a.primary}, {len(shared)} of them also measured "
          f"against {a.second}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
