#!/usr/bin/env python
"""
Fetch the two sets of fluorescence maps of plan item SP4.

A fluorescence *excitation-emission matrix* is a spectrum with two axes: the sample is excited
at one wavelength after another, and for each of them the emitted light is recorded over a
range of emission wavelengths.  The result is a map of intensity over the excitation and the
emission wavelength.

Olive oils during ageing
    Venturini, Fluri and Baumgartner, Data 8, 81 (2023), https://doi.org/10.3390/data8050081;
    the files are at https://doi.org/10.17632/g6y69g8gwm.1 (Mendeley Data, CC BY 4.0).  24
    commercial extra virgin olive oils were kept at 60 degrees C, and three samples of every
    oil were measured at each of 10 steps of ageing: 720 maps with 35 excitation wavelengths
    (300 to 640 nm) and 251 emission wavelengths (300 to 800 nm).  For one of the three
    samples the absorbance of the diluted oil at 232, 264, 268 and 272 nm was measured, the
    quantities by which the European Union regulation grades an olive oil.
Honey
    Lenhardt et al., Applied Spectroscopy 68, 557 (2014); the file is on the data page of the
    chemometrics group of the University of Copenhagen (https://ucphchemometrics.com/honey/),
    which says that its data are "available for public use" and names no licence.  110
    honeys in eight classes: acacia, linden, meadow and sunflower honeys, and six "false"
    honeys from bees that were fed a sucrose solution.  52 excitation wavelengths (240 to
    495 nm) and 741 emission wavelengths (270 to 640 nm).

Nothing of the two sets is copied into this repository: this script downloads the archives
(48.4 MB and 28.5 MB) into ``benchmarks/cache/fluorescence`` (not tracked) and writes there
what the benchmark reads.

What it writes
--------------
``olive.npz``: ``maps`` (720 samples x 35 excitation x 251 emission wavelengths, on the even
emission grid of 2 nm), ``excitation``, ``emission``, and for every sample ``oil`` (a letter
from A to X), ``replicate`` (0, 1 or 2) and ``step`` (0 to 9); ``absorbance`` (720 x 4, with
NaN for the samples that were not measured), ``absorbance_wavelength`` and ``days`` (the
days of ageing of the ten steps).

``honey.npz``: ``maps`` (110 x 52 x 741; a value that the authors removed is NaN),
``excitation``, ``emission``, ``label`` (the class of every honey) and ``false`` (true for
the six false honeys).

Usage
-----
    python benchmarks/prepare_fluorescence.py            # download if needed, then prepare
    python benchmarks/prepare_fluorescence.py --check    # only say what is in the cache
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys
import urllib.request
import zipfile
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "fluorescence")
#: set -> (address, file in the cache, SHA-256 sum, size as text, terms)
ARCHIVES = {
    "olive": ("https://data.mendeley.com/public-api/zip/g6y69g8gwm/download/1",
              os.path.join("olive", "olive_oil_ageing.zip"),
              "75759d5c2dc71395b777aaab2590f143b9ea28f3d46bff17e7d0438c535dd553", "48.4 MB",
              "CC BY 4.0"),
    "honey": ("https://sid.erda.dk/share_redirect/fGlqQtWWut/HoneyEEM.zip",
              os.path.join("honey", "HoneyEEM.zip"),
              "ee73d60267b02e62f4f685b63df938fbf93624232004e1cdcf9c5c32200266ed", "28.5 MB",
              "available for public use, no licence named"),
}
SOURCES = {"olive": "https://doi.org/10.17632/g6y69g8gwm.1",
           "honey": "https://ucphchemometrics.com/honey/"}
#: the days at 60 degrees C of the ten ageing steps (Table 3 of the data paper)
OLIVE_EMISSION = (300.0, 800.0, 2.0)
OLIVE_NAME = re.compile(r"(\d{8})_(\d{4})_AS(\d)_((?:[A-X][0-2])+)\.csv$")
ABSORBANCE_WAVELENGTHS = (232.0, 264.0, 268.0, 272.0)
ABSORBANCE_ROW = re.compile(r"^([A-X][0-2])\s*,[^,]*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,"
                            r"\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,")


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch(which: str, cache: str = CACHE) -> str:
    """Download one archive when it is not in the cache; returns its path."""
    url, rel, _, size, terms = ARCHIVES[which]
    path = os.path.join(cache, rel)
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f"downloading {url} ({size}, {terms}) ...")
    request = urllib.request.Request(url, headers={"User-Agent": "cmfqsar-benchmark"})
    with urllib.request.urlopen(request, timeout=900) as r, open(path + ".part", "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(path + ".part", path)
    return path


def checked(path: str, want: str = "") -> bytes:
    """The bytes of an archive; with `want` they have to have that SHA-256 sum."""
    with open(path, "rb") as fh:
        raw = fh.read()
    got = hashlib.sha256(raw).hexdigest()
    if want and got != want:
        raise SystemExit(f"{path} has the SHA-256 sum {got}; the script was written for "
                         f"{want}.  The set has changed.")
    return raw


# --------------------------------------------------------------------------- #
# Olive oil
# --------------------------------------------------------------------------- #
def olive_members(raw: bytes) -> Dict[str, bytes]:
    """The CSV files of the olive-oil archive by their path, read into memory.  The archive
    holds one more archive with the files; nothing is unpacked to disk."""
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        inner = [i.filename for i in outer.infolist() if i.filename.lower().endswith(".zip")]
        source = zipfile.ZipFile(io.BytesIO(outer.read(inner[0]))) if inner else outer
        return {i.filename: source.read(i) for i in source.infolist()
                if i.filename.lower().endswith(".csv")}


def fluorescence_file(text: str, grid: np.ndarray) -> Dict[str, np.ndarray]:
    """The maps of one file of the instrument: sample -> (excitation, map on `grid`).

    Line 1 names every pair of columns, as ``C2_EX_300.00`` for sample C2 excited at 300 nm;
    line 2 says which column is the wavelength and which the intensity; the block of numbers
    ends at the first empty line, and the log of the instrument follows it.  The emission
    wavelengths of the instrument are not exactly even (302.03, 304.06, ...), so every curve
    is interpolated onto the even `grid`.
    """
    lines = text.splitlines()
    names = lines[0].split(",")[0::2]
    rows = []
    for line in lines[2:]:
        if not line.strip():
            break
        rows.append(line.split(","))
    block = np.array([[float(v) if v.strip() else np.nan for v in r[:2 * len(names)]]
                      for r in rows])
    curves: Dict[str, List[Tuple[float, np.ndarray]]] = {}
    for k, name in enumerate(names):
        m = re.match(r"(\w+)_EX_([\d.]+)$", name.strip())
        if not m:
            continue
        x, y = block[:, 2 * k], block[:, 2 * k + 1]
        ok = np.isfinite(x) & np.isfinite(y)
        order = np.argsort(x[ok])
        curve = np.interp(grid, x[ok][order], y[ok][order])
        curves.setdefault(m.group(1), []).append((float(m.group(2)), curve))
    out = {}
    for sample, pairs in curves.items():
        pairs.sort(key=lambda p: p[0])
        out[sample] = (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
    return out


def absorbance_file(text: str) -> Dict[str, np.ndarray]:
    """The four absorbances of every sample of one file of the absorption instrument."""
    out = {}
    for line in text.splitlines():
        m = ABSORBANCE_ROW.match(line)
        if m:
            out[m.group(1)] = np.array([float(m.group(k)) for k in (2, 3, 4, 5)])
    return out


def olive(files: Dict[str, bytes]) -> Dict[str, np.ndarray]:
    """The olive-oil set from its CSV files.

    A file whose name does not have the pattern of the set is left out (the archive holds
    one measurement of a white wine).  A sample that was measured twice keeps the later
    measurement.  Returns the arrays of ``olive.npz`` and, under ``notes``, what was left
    out.
    """
    grid = np.arange(OLIVE_EMISSION[0], OLIVE_EMISSION[1] + 0.5 * OLIVE_EMISSION[2],
                     OLIVE_EMISSION[2])
    maps: Dict[Tuple[int, str], np.ndarray] = {}
    absorb: Dict[Tuple[int, str], np.ndarray] = {}
    excitation = None
    odd, twice = [], []
    for path in sorted(files, key=lambda p: p.split("/")[-1]):     # by date and time
        name = path.split("/")[-1]
        m = OLIVE_NAME.match(name)
        if not m:
            odd.append(name)
            continue
        step = int(m.group(3))
        text = files[path].decode("utf-8", errors="replace")
        if "/Absorption/" in "/" + path:
            for sample, values in absorbance_file(text).items():
                absorb[(step, sample)] = values
            continue
        for sample, (ex, block) in fluorescence_file(text, grid).items():
            if excitation is None:
                excitation = ex
            if len(ex) != len(excitation) or not np.allclose(ex, excitation):
                raise ValueError(f"{name}: sample {sample} has other excitation wavelengths")
            if (step, sample) in maps:
                twice.append(f"step {step}, sample {sample}")
            maps[(step, sample)] = block
    keys = sorted(maps)
    A = np.full((len(keys), 4), np.nan)
    for k, key in enumerate(keys):
        if key in absorb:
            A[k] = absorb[key]
    return {"maps": np.array([maps[k] for k in keys], dtype=np.float32),
            "excitation": excitation, "emission": grid,
            "oil": np.array([k[1][0] for k in keys]),
            "replicate": np.array([int(k[1][1]) for k in keys]),
            "step": np.array([k[0] for k in keys]),
            "absorbance": A, "absorbance_wavelength": np.array(ABSORBANCE_WAVELENGTHS),
            "notes": np.array([f"left out: {', '.join(odd) or 'nothing'}",
                               f"measured twice, the later kept: {', '.join(twice) or 'none'}"])}


# --------------------------------------------------------------------------- #
# Honey
# --------------------------------------------------------------------------- #
def honey(raw: bytes, member: str = "HoneyEEM.mat") -> Dict[str, np.ndarray]:
    """The honey set from its archive: the "DataSet" object ``X`` of a MATLAB file, read with
    scipy (see `prepare_calibration_transfer.py`).  The table is samples x emission x
    excitation and is turned to samples x excitation x emission, the layout of the olive
    set."""
    from scipy.io import loadmat

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        found = loadmat(io.BytesIO(z.read(member)))
    r = found["X"][0, 0]
    data = np.asarray(r["data"], dtype=float)
    emission = np.asarray(r["axisscale"][1, 0], dtype=float).ravel()
    excitation = np.asarray(r["axisscale"][2, 0], dtype=float).ravel()
    if data.shape[1:] != (len(emission), len(excitation)):
        raise ValueError(f"a table of shape {data.shape} for {len(emission)} emission and "
                         f"{len(excitation)} excitation wavelengths")
    number = np.asarray(r["class"][0, 0]).ravel().astype(int)
    lookup = r["classlookup"][0, 0]
    names = {int(np.asarray(lookup[k, 0]).ravel()[0]): str(np.asarray(lookup[k, 1]).ravel()[0])
             for k in range(lookup.shape[0])}
    label = np.array([names[n].strip().lower() for n in number])
    return {"maps": data.transpose(0, 2, 1).astype(np.float32), "excitation": excitation,
            "emission": emission, "label": label,
            "false": np.array([s.startswith("false") for s in label])}


# --------------------------------------------------------------------------- #
def prepare(cache: str = CACHE) -> None:
    os.makedirs(cache, exist_ok=True)
    o = olive(olive_members(checked(fetch("olive", cache), ARCHIVES["olive"][2])))
    np.savez_compressed(os.path.join(cache, "olive.npz"), **o)
    for note in o["notes"]:
        log(f"olive: {note}")
    h = honey(checked(fetch("honey", cache), ARCHIVES["honey"][2]))
    np.savez_compressed(os.path.join(cache, "honey.npz"), **h)


def load(which: str, cache: str = CACHE) -> Dict[str, np.ndarray]:
    """The prepared set ``"olive"`` or ``"honey"`` as a dictionary of arrays."""
    path = os.path.join(cache, which + ".npz")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run benchmarks/prepare_fluorescence.py first "
                         "(it downloads 77 MB).")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only say what is in the cache")
    a = ap.parse_args(argv)
    if not a.check:
        prepare()
    for which in ("olive", "honey"):
        if os.path.isfile(os.path.join(CACHE, which + ".npz")):
            d = load(which)
            log(f"{which}: maps {d['maps'].shape}, excitation {d['excitation'][0]:g} to "
                f"{d['excitation'][-1]:g} nm, emission {d['emission'][0]:g} to "
                f"{d['emission'][-1]:g} nm")
        else:
            log(f"{which}: not prepared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
