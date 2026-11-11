#!/usr/bin/env python
"""
Fetch the Chemotion IR set of measured infrared spectra and put it on one grid.

The set (Jung, Tremouilhac, Punjabi et al., Karlsruhe Institute of Technology, 2024,
https://doi.org/10.22000/OGoEQGlsZGElrgst) holds 4,183 JCAMP-DX files for 2,116 infrared
measurements of 1,926 compounds that were made in research laboratories, with the SMILES of
every compound.  Its licence is CC BY-SA 4.0.  A file that is passed on has to carry that
licence, so nothing of the set is copied into this repository: this script downloads it into
``benchmarks/cache/chemotion_ir`` (47.5 MB, not tracked) and writes there what the benchmark
reads.

What it writes
--------------
``spectra.npz``: one row per measurement.  ``absorbance`` holds the spectrum as absorbance on
an even grid of wavenumbers (``grid``); a spectrum that the file gives as transmittance is
converted.  ``table.json`` beside it holds, for every row, the SMILES of the compound, the
numbers of the sample and of the analysis in the Chemotion repository, the instrument, the
kind of measurement and the name of the file that was read.

A measurement has up to three files: the spectrum as the instrument wrote it, and copies in
which a chemist marked the peaks.  The first block of every file is the spectrum itself, and
the first file that can be read is used.  A measurement is left out when no file holds a
curve, when the curve does not cover the grid, or when it is not an infrared spectrum.

Usage
-----
    python benchmarks/prepare_chemotion_ir.py            # download if needed, then prepare
    python benchmarks/prepare_chemotion_ir.py --check    # only say what is in the cache
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import sys
import tarfile
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar import spectrum as SP  # noqa: E402

DOI = "https://doi.org/10.22000/OGoEQGlsZGElrgst"
URL = ("https://radar4chem.radar-service.eu/radar-backend/archives/"
       "OGoEQGlsZGElrgst/versions/1/content")
LICENCE = "CC BY-SA 4.0"
CACHE = os.path.join(HERE, "cache", "chemotion_ir")
ARCHIVE = os.path.join(CACHE, "download", "chemotion_ir.tar")
#: the two files of the archive that hold the data, with their SHA-256 sums.  The outer
#: archive is not checked as a whole, because the repository may pack it again.
MEMBERS = {
    "dataset/meta_data.json":
        "187bb351465409ff84d960aad5aa3837d25e5c062030daafde5e3326f6eb6336",
    "JCAMP-DX Files/IR_data.tar.xz":
        "f6b8cdc7d0cfc53eb5ec2d34cd9f49b315ddc7343745074a62cc4f43e32de9a0",
}
#: the grid of the prepared spectra, in cm^-1.  Nearly every spectrum of the set covers it.
GRID = (650.0, 3900.0, 2.0)
SPECTRA = os.path.join(CACHE, "spectra.npz")
TABLE = os.path.join(CACHE, "table.json")


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch(archive: str = ARCHIVE) -> str:
    """Download the archive when it is not in the cache; returns its path."""
    if os.path.isfile(archive):
        return archive
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    log(f"downloading {URL} (47.5 MB, {LICENCE}) ...")
    request = urllib.request.Request(URL, headers={"User-Agent": "cmfqsar-benchmark"})
    with urllib.request.urlopen(request, timeout=600) as r, \
            open(archive + ".part", "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(archive + ".part", archive)
    return archive


def members(archive: str = ARCHIVE, check: bool = True) -> Tuple[list, Dict[str, bytes]]:
    """The metadata and the JCAMP-DX files of the archive, read into memory.

    Returns the list of the metadata and the text files by the last part of their name.
    Nothing is written to disk, so a name inside the archive cannot reach a path outside it.
    With `check`, the two data files have to have the SHA-256 sums of :data:`MEMBERS`.
    """
    raw: Dict[str, bytes] = {}
    with tarfile.open(archive) as outer:
        for m in outer.getmembers():
            for suffix in MEMBERS:
                if m.isfile() and m.name.endswith(suffix):
                    raw[suffix] = outer.extractfile(m).read()
    missing = [s for s in MEMBERS if s not in raw]
    if missing:
        raise SystemExit(f"{archive} does not hold {missing[0]}")
    if check:
        for suffix, want in MEMBERS.items():
            got = hashlib.sha256(raw[suffix]).hexdigest()
            if got != want:
                raise SystemExit(f"{suffix} of {archive} has the SHA-256 sum {got}; the "
                                 f"script was written for {want}.  The set has changed.")
    meta = json.loads(raw["dataset/meta_data.json"])
    files: Dict[str, bytes] = {}
    inner_bytes = lzma.decompress(raw["JCAMP-DX Files/IR_data.tar.xz"])
    with tarfile.open(fileobj=io.BytesIO(inner_bytes)) as inner:
        for m in inner.getmembers():
            if m.isfile():
                files[m.name.split("/")[-1]] = inner.extractfile(m).read()
    return meta, files


def curve_of(entry: dict, files: Dict[str, bytes]) -> Tuple[Optional[SP.Spectrum], str]:
    """The measured curve of one entry of the metadata as an absorbance spectrum, and the
    name of the file it was read from; ``(None, reason)`` when no file gives one."""
    reason = "no file of the measurement was found"
    for ds in entry.get("datasets", []):
        for att in ds.get("attacments", []):                # the spelling of the metadata
            data = files.get(str(att.get("identifier", "")).split("/")[-1])
            if data is None:
                continue
            blocks = [b for b in SP.parse_jcamp(data.decode("utf-8", errors="replace"))
                      if b["kind"] == "curve"]
            if not blocks:
                reason = "no curve in the file"
                continue
            lab = blocks[0]["labels"]
            if "1/CM" not in lab.get("XUNITS", "").upper().replace(" ", ""):
                reason = f"the axis is in {lab.get('XUNITS', '?')}"
                continue
            spec = SP.Spectrum(blocks[0]["x"], blocks[0]["y"], "curve", "wavenumber",
                               str(att.get("filename", "")),
                               {"yunits": lab.get("YUNITS", "")})
            try:
                return SP.absorbance(spec), str(att.get("filename", ""))
            except ValueError:
                reason = f"the intensity is in {lab.get('YUNITS', '?')}"
    return None, reason


def prepare(archive: str = ARCHIVE, cache: str = CACHE, grid=GRID, check: bool = True) -> dict:
    """Read every measurement, put its absorbance on the grid, and write the two files."""
    meta, files = members(archive, check)
    gx = np.arange(grid[0], grid[1] + 1e-9, grid[2])
    rows, table, left_out = [], [], {}
    for entry in meta:
        kind = str(entry.get("kind", ""))
        if "infrared" not in kind.lower() or "UV-VIS" in kind.upper():
            left_out["not an infrared spectrum"] = left_out.get(
                "not an infrared spectrum", 0) + 1
            continue
        spec, note = curve_of(entry, files)
        if spec is None:
            left_out[note] = left_out.get(note, 0) + 1
            continue
        if spec.x[0] > grid[0] or spec.x[-1] < grid[1] or not np.all(np.isfinite(spec.y)):
            left_out["the curve does not cover the grid"] = left_out.get(
                "the curve does not cover the grid", 0) + 1
            continue
        rows.append(np.interp(gx, spec.x, spec.y).astype(np.float32))
        instruments = sorted({str(ds.get("instrument") or "") for ds in entry["datasets"]})
        table.append({"smiles": entry["cano_smiles"], "sample_id": entry["sample_id"],
                      "analysis_id": entry["analysis_id"], "kind": kind,
                      "instrument": "; ".join(i for i in instruments if i),
                      "file": note, "step": float(np.median(np.diff(spec.x))),
                      "atr": "ATR" in str(entry.get("content") or "") or "ATR" in kind})
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(os.path.join(cache, "spectra.npz"), grid=gx,
                        absorbance=np.vstack(rows))
    with open(os.path.join(cache, "table.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": DOI, "licence": LICENCE, "left_out": left_out, "rows": table},
                  fh, indent=1)
    return {"entries": len(meta), "kept": len(table), "left_out": left_out,
            "compounds": len({r["smiles"] for r in table})}


def load(cache: str = CACHE) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """The grid, the absorbance spectra (one row each) and the table of the prepared set."""
    spectra, table = os.path.join(cache, "spectra.npz"), os.path.join(cache, "table.json")
    if not (os.path.isfile(spectra) and os.path.isfile(table)):
        raise SystemExit(f"{spectra} is missing: run benchmarks/prepare_chemotion_ir.py, which "
                         f"downloads the set (47.5 MB, {LICENCE})")
    with np.load(spectra) as z:
        grid, A = z["grid"], z["absorbance"].astype(float)
    with open(table, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    return grid, A, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only say what is in the cache")
    a = ap.parse_args(argv)
    if a.check:
        if not os.path.isfile(SPECTRA):
            print(f"{SPECTRA} is not there")
            return 1
        grid, A, rows = load()
        print(f"{len(rows)} spectra of {len({r['smiles'] for r in rows})} compounds on "
              f"{len(grid)} grid points from {grid[0]:g} to {grid[-1]:g} cm^-1")
        return 0
    done = prepare(fetch())
    print(f"{done['kept']} of {done['entries']} measurements kept, {done['compounds']} "
          f"compounds; left out: {done['left_out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
