"""
One real micrograph set for `run_microstructure.py`: the Ultrahigh Carbon Steel micrographs.

The data (DeCost, Hecht, Francis, Webler, Picard and Holm, UHCSDB, Integrating Materials and
Manufacturing Innovation 2017; CC BY 3.0 US) are 961 scanning electron micrographs of one steel
after different heat treatments, with the annealing temperature, the annealing time and the
cooling method for 598 of them and a primary microconstituent class for all. The original host
(materialsdata.nist.gov, handle 11256/940) no longer serves the files; the copy used here is
the mirror on Zenodo (record 23050464), which keeps the original licence and provenance.

Everything in the data folder is treated as untrusted input: this module reads image files and
one CSV from it and executes nothing from it. The 38-pixel information bar at the bottom of
every micrograph, which prints the specimen label (and so the heat treatment) as text, is cut
off before any descriptor is computed.

A micrograph is one window on a medium that does not repeat, so the two-point autocorrelation
is computed with zero padding and a mask-normalised denominator: the sum over the overlapping
pixel pairs at a lag divided by the number of such pairs.
"""
from __future__ import annotations

import csv
import glob
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "cache", "uhcs")
ZENODO_URL = ("https://zenodo.org/api/records/23050464/files/"
              "rydia001/UHCS_micrographs_mirror-release.zip/content")

IMAGE_ROWS = 484              # the rows above the information bar
REDUCE = 4                    # block mean: 645 x 484 becomes 161 x 121
MAX_LAG = 30                  # lags kept, in reduced pixels, in each direction
CLASSES = ("spheroidite", "network", "pearlite", "pearlite+spheroidite",
           "spheroidite+widmanstatten", "martensite", "pearlite+widmanstatten")


def fetch() -> None:
    """Download the mirror archive (270 MB) into the data folder and extract it there.

    The archive is extracted with the zip module into ``extract/`` after a check that no
    member path leaves that folder; nothing in it is executed.
    """
    import urllib.request
    import zipfile
    os.makedirs(CACHE, exist_ok=True)
    archive = os.path.join(CACHE, "mirror-release.zip")
    if not os.path.isfile(archive):
        print(f"  downloading {ZENODO_URL} (270 MB) ...", flush=True)
        urllib.request.urlretrieve(ZENODO_URL, archive)
    target = os.path.join(CACHE, "extract")
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            dest = os.path.realpath(os.path.join(target, name))
            if not dest.startswith(os.path.realpath(target) + os.sep):
                raise SystemExit(f"refusing to extract {name!r}: it would leave {target}")
        z.extractall(target)


def data_folder() -> str:
    found = sorted(glob.glob(os.path.join(CACHE, "extract", "*", "metadata.csv")))
    if not found:
        fetch()
        found = sorted(glob.glob(os.path.join(CACHE, "extract", "*", "metadata.csv")))
    if not found:
        raise SystemExit(
            "the UHCS micrographs are not under benchmarks/cache/uhcs/extract: download the "
            f"mirror archive {ZENODO_URL} (270 MB, CC BY 3.0 US) into benchmarks/cache/uhcs/ "
            "and extract it into benchmarks/cache/uhcs/extract/")
    return os.path.dirname(found[0])


def metadata() -> List[dict]:
    """One record per micrograph: the file, the class, the specimen, the treatment, the scale."""
    folder = data_folder()
    rows = []
    with open(os.path.join(folder, "metadata.csv"), encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            temp = float(r["anneal_temperature"]) if r["anneal_temperature"] else np.nan
            t = float(r["anneal_time"]) if r["anneal_time"] else np.nan
            if r["anneal_time_unit"] == "M":
                t = t / 60.0
            bar = float(r["micron_bar"]) if r["micron_bar"] else np.nan
            if r["micron_bar_units"] == "nm":
                bar = bar / 1000.0
            px = float(r["micron_bar_px"]) if r["micron_bar_px"] else np.nan
            rows.append({"id": int(r["micrograph_id"]),
                         "path": os.path.join(folder, "micrographs", os.path.basename(r["path"])),
                         "cls": r["primary_microconstituent"],
                         "specimen": r["sample_key"] or r["label"],
                         "temperature": temp, "time_h": t, "cool": r["cool_method"],
                         "um_per_px": bar / px if px else np.nan})
    return rows


def load_images(rows: Sequence[dict]) -> np.ndarray:
    """Every micrograph as a float array in [0, 1], cropped above the bar and block-averaged."""
    from PIL import Image
    path = os.path.join(CACHE, f"images_r{REDUCE}.npy")
    if os.path.isfile(path):
        arr = np.load(path)
        if arr.shape[0] == len(rows):
            return arr
    out = None
    t0 = time.time()
    for i, r in enumerate(rows):
        im = np.asarray(Image.open(r["path"]).convert("L"), dtype=np.float64) / 255.0
        im = im[:IMAGE_ROWS]
        h, w = (im.shape[0] // REDUCE) * REDUCE, (im.shape[1] // REDUCE) * REDUCE
        small = im[:h, :w].reshape(h // REDUCE, REDUCE, w // REDUCE, REDUCE).mean(axis=(1, 3))
        if out is None:
            out = np.zeros((len(rows),) + small.shape, np.float32)
        out[i] = small
    np.save(path, out)
    print(f"  {len(rows)} micrographs read and reduced to {out.shape[1:]} in "
          f"{time.time() - t0:.0f} s", flush=True)
    return out


def smooth(images: np.ndarray, s: float) -> np.ndarray:
    if s <= 0:
        return images.astype(np.float64)
    from scipy.ndimage import gaussian_filter
    return np.stack([gaussian_filter(im.astype(np.float64), s, mode="reflect") for im in images])


def two_level(images: np.ndarray) -> np.ndarray:
    """The indicator of the bright phase, by Otsu's threshold on each image."""
    from skimage.filters import threshold_otsu
    return np.stack([(im > threshold_otsu(im)).astype(np.float64) for im in images])


def windowed_autocorrelation(fields: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """S(r) of each field of a non-periodic window, lags -max_lag..max_lag in each direction.

    Zero padding to twice the size makes the circular correlation a linear one; the count of
    overlapping pixel pairs at each lag (the autocorrelation of the all-ones mask) is the
    denominator. The field's own mean is removed first, so that this is the covariance of the
    grey level (for an indicator, the covariance of the phase) and the window's mean grey
    level does not enter through the lag-zero term alone.
    """
    n, h, w = fields.shape
    H, W = 2 * h, 2 * w
    mask = np.zeros((H, W))
    mask[:h, :w] = 1.0
    counts = np.fft.ifft2(np.abs(np.fft.fft2(mask)) ** 2).real
    counts = np.maximum(np.rint(counts), 1.0)
    padded = np.zeros((n, H, W))
    centred = fields - fields.mean(axis=(1, 2), keepdims=True)
    padded[:, :h, :w] = centred
    F = np.fft.fft2(padded, axes=(1, 2))
    S = np.fft.ifft2(np.abs(F) ** 2, axes=(1, 2)).real / counts
    lags = np.r_[0:max_lag + 1, H - max_lag:H]
    lags_w = np.r_[0:max_lag + 1, W - max_lag:W]
    return S[:, lags][:, :, lags_w]          # (n, 2 max_lag + 1, 2 max_lag + 1), lag 0 first


def radial_profile(S: np.ndarray) -> np.ndarray:
    """The autocorrelation binned by the length of the lag, weighted by the root of the count."""
    L = S.shape[-1]
    k = np.fft.fftfreq(L) * L
    coords = np.meshgrid(k, k, indexing="ij")
    r = np.sqrt(coords[0] ** 2 + coords[1] ** 2)
    shell = np.floor(r + 1e-9).astype(int)
    n_bins = int(shell.max()) + 1
    counts = np.bincount(shell.ravel(), minlength=n_bins).astype(float)
    flat = S.reshape(S.shape[0], -1)
    out = np.stack([np.bincount(shell.ravel(), weights=f, minlength=n_bins) for f in flat])
    return out / np.maximum(counts, 1.0) * np.sqrt(counts)


def cnn_features(images: np.ndarray, rows: Sequence[dict]) -> np.ndarray:
    """ResNet-18 trained on ImageNet, the 512 values after global average pooling, on each
    micrograph cropped above the bar and resized to 224 x 224 (the DeCost 2017 recipe with a
    pretrained network and a linear model on top)."""
    path = os.path.join(CACHE, "resnet18_avgpool.npy")
    if os.path.isfile(path):
        arr = np.load(path)
        if arr.shape[0] == len(rows):
            return arr
    import torch
    import torchvision
    from PIL import Image
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval()
    pre = weights.transforms()
    out = np.zeros((len(rows), 512), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            batch = []
            for r in rows[i:i + 16]:
                im = Image.open(r["path"]).convert("L").crop((0, 0, 645, IMAGE_ROWS)).convert("RGB")
                batch.append(pre(im))
            out[i:i + len(batch)] = net(torch.stack(batch)).numpy()
    np.save(path, out)
    print(f"  CNN features of {len(rows)} micrographs in {time.time() - t0:.0f} s", flush=True)
    return out


def grouped_draw(groups: np.ndarray, size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Whole specimens into the training part until it holds at least `size` images; the
    images of every other specimen are the test part."""
    rng = np.random.RandomState(seed)
    keys = np.unique(groups)
    rng.shuffle(keys)
    train_keys, count = [], 0
    for k in keys:
        if count >= size:
            break
        train_keys.append(k)
        count += int(np.sum(groups == k))
    tr = np.flatnonzero(np.isin(groups, train_keys))
    te = np.flatnonzero(~np.isin(groups, train_keys))
    return tr, te
