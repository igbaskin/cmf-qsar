#!/usr/bin/env python
"""
Fetch the two near-infrared sets of plan item SP5: the same samples on several instruments.

Corn
    80 samples of corn on three spectrometers (``m5``, ``mp5``, ``mp6``), 1100 to 2498 nm in
    steps of 2 nm, with the moisture, oil, protein and starch of every sample.  Measured at
    Cargill.
Tablets
    655 pharmaceutical tablets on two spectrometers, 600 to 1898 nm in steps of 2 nm, with the
    weight, the hardness and the content of the active ingredient (``assay``).  The set of
    the "shoot-out" of the International Diffuse Reflectance Conference of 2002.  It comes
    divided into a calibration part (155 tablets), a validation part (40) and a test part
    (460).

Both files are in the data archive of Eigenvector Research
(https://eigenvector.com/resources/data-sets/).  The page states no licence, so nothing of
the sets is copied into this repository: this script downloads the two archives (1.1 MB and
5.4 MB) into ``benchmarks/cache/calibration_transfer`` (not tracked) and writes there what
the benchmark reads.

The files are MATLAB files that hold "DataSet" objects of the PLS_Toolbox.
``scipy.io.loadmat`` reads such an object as a record with the fields ``data``, ``label`` and
``axisscale``; MATLAB is not needed.

What it writes
--------------
``corn.npz``: ``wavelength`` (700), ``spectra`` (3 instruments x 80 samples x 700),
``instruments``, ``properties`` (80 x 4), ``property_names``, and the glass standards of every
instrument (``standards_m5`` and so on).

``tablets.npz``: ``wavelength`` (650), ``spectra`` (2 instruments x 655 tablets x 650),
``instruments``, ``properties`` (655 x 3), ``property_names`` and ``part`` (for every tablet
``calibrate``, ``validate`` or ``test``, as the set was published).

Usage
-----
    python benchmarks/prepare_calibration_transfer.py            # download if needed, prepare
    python benchmarks/prepare_calibration_transfer.py --check    # only say what is in the cache
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PAGE = "https://eigenvector.com/resources/data-sets/"
BASE = "https://eigenvector.com/wp-content/uploads/2019/06/"
LICENCE = "no licence stated"
CACHE = os.path.join(HERE, "cache", "calibration_transfer")
#: archive -> (member that holds the data, SHA-256 sum of the archive, size as text)
ARCHIVES = {
    "corn.mat_.zip": (
        "corn.mat", "8a2d1a03648b6ad334caaafa5d8377bf945ba1477a5082c4963d705d07cca795",
        "1.1 MB"),
    "nir_shootout_2002.mat_.zip": (
        "nir_shootout_2002.mat",
        "04c03a4314ab8e8f63e0ecc788b657b5d1b5450791ce465ad41ad90cb5d52125", "5.4 MB"),
}
CORN = os.path.join(CACHE, "corn.npz")
TABLETS = os.path.join(CACHE, "tablets.npz")
CORN_INSTRUMENTS = ("m5", "mp5", "mp6")
TABLET_PARTS = ("calibrate", "validate", "test")


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch(name: str, cache: str = CACHE) -> str:
    """Download one archive when it is not in the cache; returns its path."""
    path = os.path.join(cache, "download", name)
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f"downloading {BASE + name} ({ARCHIVES[name][2]}, {LICENCE}) ...")
    request = urllib.request.Request(BASE + name, headers={"User-Agent": "cmfqsar-benchmark"})
    with urllib.request.urlopen(request, timeout=600) as r, open(path + ".part", "wb") as fh:
        fh.write(r.read())
    os.replace(path + ".part", path)
    return path


def records(path: str, member: str, want: str = "") -> Dict[str, np.ndarray]:
    """The variables of the MATLAB file `member` of the archive `path`.

    The member is read into memory and nothing is unpacked to disk, so a name inside the
    archive cannot reach a path outside it.  With `want`, the archive has to have that
    SHA-256 sum.
    """
    from scipy.io import loadmat

    with open(path, "rb") as fh:
        raw = fh.read()
    got = hashlib.sha256(raw).hexdigest()
    if want and got != want:
        raise SystemExit(f"{path} has the SHA-256 sum {got}; the script was written for "
                         f"{want}.  The set has changed.")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        data = z.read(member)
    found = loadmat(io.BytesIO(data))
    return {k: v for k, v in found.items() if not k.startswith("__")}


def dataset(record: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """The table of one "DataSet" object, the scale of its columns and the names of its
    columns.  A scale or a list of names that the object does not have is empty."""
    r = record[0, 0]
    data = np.asarray(r["data"], dtype=float)
    scale = np.asarray(r["axisscale"][1, 0], dtype=float).ravel()
    names = [str(s).strip() for s in np.asarray(r["label"][1, 0]).ravel()]
    if scale.size not in (0, data.shape[1]):
        raise ValueError(f"a scale of {scale.size} values for {data.shape[1]} columns")
    return data, scale, names


def corn(found: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The corn set from the variables of its file."""
    spectra, scale = [], None
    for name in CORN_INSTRUMENTS:
        data, axis, _ = dataset(found[name + "spec"])
        if scale is not None and not np.array_equal(axis, scale):
            raise ValueError(f"instrument {name} has another wavelength scale")
        scale = axis
        spectra.append(data)
    props, _, names = dataset(found["propvals"])
    out = {"wavelength": scale, "spectra": np.array(spectra),
           "instruments": np.array(CORN_INSTRUMENTS), "properties": props,
           "property_names": np.array([n.lower() for n in names])}
    for name in CORN_INSTRUMENTS:
        out["standards_" + name] = dataset(found[name + "nbs"])[0]
    return out


def tablets(found: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The tablet set from the variables of its file, the three parts one after another."""
    spectra: List[List[np.ndarray]] = [[], []]
    props, part, scale, names = [], [], None, []
    for name in TABLET_PARTS:
        for k in (0, 1):
            data, axis, _ = dataset(found[f"{name}_{k + 1}"])
            if scale is not None and not np.array_equal(axis, scale):
                raise ValueError(f"part {name}, instrument {k + 1}: another wavelength scale")
            scale = axis
            spectra[k].append(data)
        y, _, names = dataset(found[name + "_Y"])
        if len(y) != len(spectra[0][-1]) or len(y) != len(spectra[1][-1]):
            raise ValueError(f"part {name}: the numbers of tablets differ")
        props.append(y)
        part += [name] * len(y)
    return {"wavelength": scale,
            "spectra": np.array([np.vstack(s) for s in spectra]),
            "instruments": np.array(["1", "2"]), "properties": np.vstack(props),
            "property_names": np.array(names), "part": np.array(part)}


def prepare(cache: str = CACHE) -> None:
    os.makedirs(cache, exist_ok=True)
    made = {}
    for (name, (member, want, _)), build, out in zip(ARCHIVES.items(), (corn, tablets),
                                                     (CORN, TABLETS)):
        made[name] = build(records(fetch(name, cache), member, want))
        np.savez_compressed(os.path.join(cache, os.path.basename(out)), **made[name])
    for name, d in made.items():
        log(f"{name}: {d['spectra'].shape[0]} instruments, {d['spectra'].shape[1]} samples, "
            f"{d['spectra'].shape[2]} points from {d['wavelength'][0]:g} to "
            f"{d['wavelength'][-1]:g} nm; properties: {', '.join(d['property_names'])}")


def load(which: str, cache: str = CACHE) -> Dict[str, np.ndarray]:
    """The prepared set ``"corn"`` or ``"tablets"`` as a dictionary of arrays."""
    path = os.path.join(cache, which + ".npz")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run benchmarks/prepare_calibration_transfer.py "
                         "first (it downloads 6.5 MB).")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only say what is in the cache")
    a = ap.parse_args(argv)
    if not a.check:
        prepare()
    for which in ("corn", "tablets"):
        path = os.path.join(CACHE, which + ".npz")
        if os.path.isfile(path):
            d = load(which)
            log(f"{which}: spectra {d['spectra'].shape}, properties {d['properties'].shape}")
        else:
            log(f"{which}: not prepared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
