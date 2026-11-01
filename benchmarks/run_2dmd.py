#!/usr/bin/env python
"""
A defect series on one host lattice: the periodic overlap kernel with the host symmetry (M14).

Plan item M4 built the periodic overlap kernel and measured it on `matbench_perovskites`, where
it did not beat the periodic autocorrelation because that set is 18,928 different compounds
with no common frame. Item M14 asks for the data the kernel was built for: one host, one
supercell, one lattice, and a different arrangement of defects in every cell. The 2DMD set
(Huang et al., npj 2D Materials and Applications 7, 6 (2023)) is that data, public and already
labelled: 5,933 configurations of one to three point defects -- vacancies of either species and
substitutions by the heavier (or lighter) homologue -- on an 8 x 8 supercell of MoS2, and the
same number on WSe2, each with its DFT formation energy and HOMO-LUMO gap. The structures are
given unrelaxed on the ideal supercell, so the frame is exact: lattice spread zero, asymmetry
zero.

Two things a defect series taught the kernel
--------------------------------------------
**The difference field.** Every cell is the host plus a change on one to three sites. The
overlap kernel is bilinear, so the kernel of two cells is the kernel of the host with itself,
plus the host against each difference, plus the two differences: a model with a bias on the
difference fields is the same model as one on the whole cells, at a cost set by the number of
defects and not by the 190 atoms of the supercell (`cmfqsar.periodic.defect_table`).

**The host symmetry.** Two arrangements related by a symmetry operation of the host -- the same
vacancy pair two sites along -- have the same energy but fields that do not overlap in a fixed
frame. Averaging over the 768 operations of the supercell (64 translations times the twelve
point operations of the monolayer) makes them one structure. The linear average is additive
over the defects, so it is a kernel of the defect **counts** and knows nothing of the
arrangement; averaging the Gaussian layer instead keeps the pair and triple structure
(`cmfqsar.periodic.symmetry_averaged_kernels`). Both are measured, and the fixed frame with
them, so that the table says what the symmetry is worth and what the nonlinearity is worth.

What is compared
----------------
Four kernels on the same cells, folds and ridge grid, each with a linear and a Gaussian layer:

* **fixed frame** -- the periodic overlap of the difference fields as they stand in the cell;
* **counts** -- the periodic overlap averaged over the host symmetry, linear in the fields;
* **arrangement** -- the Gaussian layer of the periodic overlap averaged over the host
  symmetry, raw and cosine-normalised;
* **autocorrelation** -- the periodic field autocorrelation of the whole cells, the invariant
  kernel the module already had and the thing M14's gate names to beat.

The gate of M14, in the form this set gives it: (a) the symmetry-averaged overlap beats the
autocorrelation by more than the paired bootstrap interval; (b, c) the model names the defects
-- the per-defect contributions of the counts model are the formation-energy increments of
each kind of defect; (d) trained on the cells with one and two defects, it predicts the cells
with three, which is the extrapolation a dopant study makes.

The published errors on this set are the reference: with the test split of Kazeev et al.
(npj Computational Materials 9, 113 (2023)), the formation energy per defect site of MoS2 is
predicted to an MAE of 2.6 meV by their sparse graph network, 12 meV by gradient boosting on
descriptors and 34 to 48 meV by three whole-structure graph networks. Those are a different
split from the folds here, so they are context, not a column of this table.

The data file (5 MB) is downloaded once to ``benchmarks/cache/2dmd``. The structures are
reduced to difference records against the host and kept there as JSON; the kernels of a run
are kept there too, because the arrangement kernel of the full set costs about an hour per
attenuation.

Usage
-----
    python benchmarks/run_2dmd.py                         # both materials, 2,000 cells each
    python benchmarks/run_2dmd.py --material MoS2 --limit 500
    python benchmarks/run_2dmd.py --stage fit --kernels counts arrangement
    python benchmarks/run_2dmd.py --stage reading          # the per-defect contributions
    python benchmarks/run_2dmd.py --report
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar import periodic  # noqa: E402
from cmfqsar.fields import AtomTable  # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel, channel_scale  # noqa: E402
from cmfqsar.kcca import paired_difference  # noqa: E402
from cmfqsar.model import (CMFModel, _Learner, cv_predict, median_gamma_scale,  # noqa: E402
                           r2_score, rmse)

URL = ("https://constructor.app/platform/open/2d-materials-point-defects/attachments/"
       "b62e17e599364a12941a9ae494a14736?name=2d-materials-point-defects-all.zip")
CACHE = os.path.join(HERE, "cache", "2dmd")
ZIP = os.path.join(CACHE, "raw", "2d-materials-point-defects-all.zip")
RESULTS = os.path.join(HERE, "results", "2dmd")
MATERIALS = {"MoS2": "low_density_defects/MoS2", "WSe2": "low_density_defects/WSe2"}
ELEMENTS = ("Mo", "S", "W", "Se")
#: the charge of every species on the host lattice: a substituent takes the charge of the site
#: it occupies, so the electrostatic difference is as local as the structural one
HOST_CHARGE = {"Mo": 4.0, "W": 4.0, "S": -2.0, "Se": -2.0}
TARGET = "formation_energy"
SPLIT_SEED = 0
OUTER, INNER = 5, 5
#: the ridge, relative to the mean diagonal of the kernel (every kernel is scaled to a mean
#: self-kernel of one before the fit, so that one grid serves kernels of any magnitude)
LAMS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1)
#: attenuations of the overlap kernels (the Gaussian width per atom); the nearest-neighbour
#: distances of the host are 2.4 A (Mo-S) and 3.2 A (S-S)
ALPHAS = (0.5, 1.0, 2.0)
#: bandwidths of the Gaussian layer, in units of the median squared distance of the fixed frame
GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0)
#: distance grid of the autocorrelation, as `run_periodic_overlap.py` uses it
AC_SIGMAS = (0.35, 0.5, 0.75)
AC_R_MAX, AC_TAPER = 10.0, 2.0
LIMIT = 2000
KERNELS = ("fixed", "counts", "arrangement", "autocorrelation")
#: the fields: the three of INORGANIC_FIELDS, the inverse square root of the mass, and one
#: indicator per element of the series
FIELDS = tuple(periodic.INORGANIC_FIELDS) + ("prop:inverse_sqrt_mass",)
#: the cells of the site union that define the host: enough that every site is occupied in
#: most of them
HOST_SAMPLE = 60
READING_SAMPLE = 400


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def result(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def jdump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=_default)


def jload(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# The data
# --------------------------------------------------------------------------- #

def fetch() -> str:
    if not os.path.isfile(ZIP):
        os.makedirs(os.path.dirname(ZIP), exist_ok=True)
        log(f"downloading {URL}")
        urllib.request.urlretrieve(URL, ZIP)
    return ZIP


def read_cif(text: str):
    """Cell (3 x 3), fractional positions (N x 3) and symbols of a pymatgen-written P1 CIF.

    The files of this set are all written by pymatgen in one layout, so a reader of the few
    tags they use is faster than a general one by a factor of fifty and raises no warnings
    about coordinates that are not exactly 1/24.
    """
    from ase.geometry import cellpar_to_cell

    par = {}
    rows = []
    in_loop = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("_cell_"):
            key, val = s.split()
            par[key] = float(val)
        elif s.startswith("_atom_site_occupancy"):
            in_loop = True
        elif in_loop and s and not s.startswith("_") and not s.startswith("loop_"):
            p = s.split()
            if len(p) >= 7:
                rows.append((p[0], float(p[3]), float(p[4]), float(p[5])))
    cell = cellpar_to_cell([par["_cell_length_a"], par["_cell_length_b"], par["_cell_length_c"],
                            par["_cell_angle_alpha"], par["_cell_angle_beta"],
                            par["_cell_angle_gamma"]])
    frac = np.array([[r[1], r[2], r[3]] for r in rows], float)
    return np.asarray(cell, float), frac, [r[0] for r in rows]


def _csv(z, name: str):
    import csv
    raw = z.open(name).read()
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def prepare(material: str, force: bool = False) -> dict:
    """Reduce the structures of one material to difference records against the host.

    The host is the union of the sites over the first cells (every site is occupied in most of
    them), with the majority species on each; a structure is then the host minus the sites it
    leaves empty plus the atoms that sit on a site with another species. The record of a cell
    is those two lists and its targets, which is what the kernel needs and a hundredth of the
    size of the CIFs.
    """
    path = os.path.join(CACHE, f"{material}.json")
    if os.path.isfile(path) and not force:
        prep = jload(path)
        if "n_classes" in prep:
            return prep
        label_classes(prep)
        jdump(prep, path)
        return prep
    z = zipfile.ZipFile(fetch())
    base = MATERIALS[material]
    targets = {r["_id"]: r for r in _csv(z, base + "/targets.csv.gz")}
    desc = {r["_id"]: r for r in _csv(z, base + "/descriptors.csv")}
    tar = tarfile.open(fileobj=io.BytesIO(z.open(base + "/initial.tar.gz").read()))
    members = [m for m in tar.getmembers() if m.name.endswith(".cif")]
    log(f"{material}: {len(members)} cells, {len(targets)} target rows")
    parsed = []
    for m in members:
        _id = os.path.basename(m.name)[:-4]
        if _id not in targets:
            continue
        cell, frac, syms = read_cif(tar.extractfile(m).read().decode("utf-8"))
        parsed.append((_id, cell, frac, syms))
    A = parsed[0][1]
    for _id, cell, _f, _s in parsed:
        if np.max(np.abs(cell - A)) > 1e-4:
            raise ValueError(f"{material}: cell of {_id} differs from the first")
    # the host: union of sites, majority species
    sites, votes = [], []
    for _id, _c, frac, syms in parsed[:HOST_SAMPLE]:
        for p, s in zip(frac, syms):
            hit = None
            if sites:
                d = np.asarray(sites) - p[None, :]
                d -= np.round(d)
                dist = np.linalg.norm(d @ A, axis=1)
                k = int(np.argmin(dist))
                if dist[k] < 0.1:
                    hit = k
            if hit is None:
                sites.append(p % 1.0)
                votes.append({s: 1})
            else:
                votes[hit][s] = votes[hit].get(s, 0) + 1
    host_frac = np.asarray(sites)
    host_syms = [max(v, key=v.get) for v in votes]
    log(f"{material}: host of {len(host_syms)} sites: "
        + ", ".join(f"{host_syms.count(s)} {s}" for s in sorted(set(host_syms))))
    # the records
    records = []
    for _id, _c, frac, syms in parsed:
        d = frac[:, None, :] - host_frac[None, :, :]
        d -= np.round(d)
        dist = np.linalg.norm(d @ A, axis=2)
        site = np.argmin(dist, axis=1)
        if np.any(dist[np.arange(len(frac)), site] > 0.1):
            raise ValueError(f"{material}: an atom of {_id} sits on no site of the host")
        occupied = set(int(s) for s in site)
        removed = sorted(set(range(len(host_syms))) - occupied)
        added = [[syms[i], int(site[i])] for i in range(len(frac)) if syms[i] != host_syms[site[i]]]
        t = targets[_id]
        dd = desc.get(t["descriptor_id"], {})
        records.append({"id": _id, "descriptor": dd.get("description", ""),
                        "n_defects": len(removed) + len(added),
                        "removed": removed, "added": added,
                        "formation_energy": float(t["formation_energy"]),
                        "formation_energy_per_site": float(t["formation_energy_per_site"]),
                        "homo_lumo_gap": float(t["homo_lumo_gap"]),
                        "energy": float(t["energy"])})
    out = {"material": material, "cell": A.tolist(), "host_frac": host_frac.tolist(),
           "host_symbols": host_syms, "n": len(records), "records": records}
    label_classes(out)
    os.makedirs(CACHE, exist_ok=True)
    jdump(out, path)
    counts = {}
    for r in records:
        counts[r["n_defects"]] = counts.get(r["n_defects"], 0) + 1
    log(f"{material}: {len(records)} records; cells by number of defects {counts}; "
        f"{out['n_classes']} arrangements distinct under the host symmetry")
    return out


def label_classes(prep: dict) -> None:
    """Give every record the class of its arrangement under the host symmetry.

    Two cells whose defects are carried onto each other by an operation of the host have the
    same energy, and the symmetry-averaged kernels treat them as one structure.  The set does
    hold such pairs, so a fold that puts one in the training part and the other in the test
    part is not a test of the model; the class lets the benchmark report the error on the
    arrangements the model has not seen separately.  The class is the smallest, over the
    operations, of the sorted list of (site, change) after the operation has permuted the
    sites.
    """
    host = host_atoms(prep)
    rot, tr = periodic.host_symmetry(host)
    A = np.asarray(prep["cell"], float)
    frac = np.asarray(prep["host_frac"], float)
    perm = np.zeros((len(rot), len(frac)), int)
    for g, (R, t) in enumerate(zip(rot, tr)):
        moved = frac @ R.T + t[None, :]
        d = moved[:, None, :] - frac[None, :, :]
        d -= np.round(d)
        dist = np.linalg.norm(d @ A, axis=2)
        perm[g] = np.argmin(dist, axis=1)
        if np.any(dist[np.arange(len(frac)), perm[g]] > 0.05) or len(set(perm[g])) != len(frac):
            raise ValueError("a symmetry operation of the host does not permute its sites")
    classes = {}
    for rec in prep["records"]:
        changes = [(int(s), "-") for s in rec["removed"]] + [(int(k), "+" + s) for s, k in rec["added"]]
        best = None
        for g in range(len(rot)):
            key = tuple(sorted((int(perm[g][s]), c) for s, c in changes))
            if best is None or key < best:
                best = key
        rec["arrangement_class"] = classes.setdefault(best, len(classes))
    prep["n_classes"] = len(classes)


def host_atoms(prep: dict):
    from ase import Atoms
    host = Atoms(symbols=prep["host_symbols"], scaled_positions=np.asarray(prep["host_frac"]),
                 cell=np.asarray(prep["cell"]), pbc=True)
    host.set_initial_charges([HOST_CHARGE[s] for s in prep["host_symbols"]])
    host.name = f"{prep['material']} 8x8"
    return host


def cell_of(prep: dict, rec: dict):
    """The ASE structure of one record: the host minus its vacancies plus its substituents,
    every atom charged by the host rule."""
    from ase import Atoms
    frac = np.asarray(prep["host_frac"])
    syms = list(prep["host_symbols"])
    for s, k in rec["added"]:
        syms[k] = s
    keep = [i for i in range(len(syms)) if i not in set(rec["removed"])]
    a = Atoms(symbols=[syms[i] for i in keep], scaled_positions=frac[keep],
              cell=np.asarray(prep["cell"]), pbc=True)
    a.set_initial_charges([HOST_CHARGE[s] for s in a.get_chemical_symbols()])
    return a


def pick(prep: dict, limit: int, seed: int = SPLIT_SEED) -> list:
    """A subset of the records of the size asked for, or all of them.

    Every cell with one or two defects is kept (there are 131 of 5,933 in MoS2), because the
    extrapolation of gate (d) trains on them; the rest are a random sample of the cells with
    three.
    """
    recs = prep["records"]
    if limit and limit < len(recs):
        few = [i for i, r in enumerate(recs) if r["n_defects"] <= 2]
        rest = [i for i, r in enumerate(recs) if r["n_defects"] > 2]
        rng = np.random.RandomState(seed)
        take = max(limit - len(few), 0)
        idx = sorted(few + list(rng.choice(rest, min(take, len(rest)), replace=False)))
        recs = [recs[i] for i in idx]
    return recs


def fieldset(alpha: float):
    return periodic.inorganic_fieldset(elements=ELEMENTS, physchem=FIELDS, charge_model="file",
                                       other=False, alpha=alpha)


def folds(n: int):
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=OUTER, shuffle=True, random_state=SPLIT_SEED).split(np.arange(n)))


# --------------------------------------------------------------------------- #
# The kernels
# --------------------------------------------------------------------------- #

def kernel_path(material: str, n: int, kind: str, setting: str) -> str:
    os.makedirs(os.path.join(CACHE, "kernels"), exist_ok=True)
    safe = setting.replace(" ", "_").replace(",", "")
    return os.path.join(CACHE, "kernels", f"{material}_{n}_{kind}_{safe}.npy")


def gauss_layer(K: np.ndarray, gamma_rel: float, scale: float) -> np.ndarray:
    d = np.diag(K)
    d2 = np.maximum(d[:, None] + d[None, :] - 2.0 * K, 0.0)
    return np.exp(-gamma_rel * scale * d2)


def cosine(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(K), 1e-300))
    return K / d[:, None] / d[None, :]


class Series:
    """The structures of one run, built once per attenuation."""

    def __init__(self, prep: dict, recs: list):
        self.prep = prep
        self.recs = recs
        self.host = host_atoms(prep)
        self.symmetry = periodic.host_symmetry(self.host)
        self.y = np.array([r[TARGET] for r in recs], float)
        self.n_defects = np.array([r["n_defects"] for r in recs], int)
        self.classes = np.array([r["arrangement_class"] for r in recs], int)
        self.names = [r["id"] for r in recs]
        self._cells = None

    @property
    def cells(self):
        if self._cells is None:
            self._cells = [cell_of(self.prep, r) for r in self.recs]
        return self._cells

    def differences(self, alpha: float) -> AtomTable:
        fs = fieldset(alpha)
        hostf = periodic.crystal_fields(self.host, fs, self.host.name)
        return periodic.defect_table(self.cells, hostf, fs, names=self.names)

    def whole(self, alpha: float = 1.0) -> AtomTable:
        return periodic.crystal_table(self.cells, fieldset(alpha), names=self.names)


def overlap_kernels(series: Series, alpha: float, engine=None) -> dict:
    """Every setting of the three overlap kernels at one attenuation, from one evaluation of
    the compiled loop: the fixed frame, the counts and the arrangement."""
    fs = fieldset(alpha)
    specs = [s for s in fs.specs if s.enabled]
    table = series.differences(alpha)
    t0 = time.time()
    d = periodic.periodic_self_overlaps(table, specs)
    mixing = np.array([1.0 / max(float(np.mean(d[f])), 1e-300) for f in range(len(specs))])
    dp = np.tensordot(mixing, d, axes=(0, 0))
    K_fixed = np.tensordot(mixing, periodic.periodic_overlap_kernels(table, table, specs,
                                                                     engine=engine),
                           axes=(0, 0))
    scale = median_gamma_scale(K_fixed)
    log(f"  alpha {alpha:g}: fixed frame {time.time() - t0:.0f} s, gamma scale {scale:.3g}")
    t0 = time.time()
    gammas = [g * scale for g in GAMMAS]
    lin, lay = periodic.symmetry_averaged_kernels(table, table, specs, series.symmetry,
                                                  gammas=gammas, mixing=mixing, self_p=dp,
                                                  engine=engine)
    log(f"  alpha {alpha:g}: symmetry average over {len(series.symmetry[0])} operations "
        f"{time.time() - t0:.0f} s")
    K_counts = np.tensordot(mixing, lin, axes=(0, 0))
    out = {"fixed": {f"alpha {alpha:g}, linear": K_fixed},
           "counts": {f"alpha {alpha:g}, linear": K_counts},
           "arrangement": {}}
    for g in GAMMAS:
        out["fixed"][f"alpha {alpha:g}, gauss {g:g}"] = gauss_layer(K_fixed, g, scale)
        out["counts"][f"alpha {alpha:g}, gauss {g:g}"] = gauss_layer(K_counts, g,
                                                                     median_gamma_scale(K_counts))
    for g, K in zip(GAMMAS, lay):
        out["arrangement"][f"alpha {alpha:g}, gamma {g:g}"] = K
        out["arrangement"][f"alpha {alpha:g}, gamma {g:g}, cosine"] = cosine(K)
    return out


def autocorrelation_kernels(series: Series) -> dict:
    table = series.whole()
    out = {}
    for sigma in AC_SIGMAS:
        t0 = time.time()
        spec = periodic.PeriodicSpec(r_max=AC_R_MAX, step=sigma, sigma=sigma, taper=AC_TAPER,
                                     background=True)
        F = periodic.periodic_feature_matrix(table.molecules, spec,
                                             n_fields=table.weights.shape[1])
        blocks = periodic.periodic_block_sizes(spec, table.weights.shape[1])
        sc = channel_scale(F, spec.n_bins, blocks=blocks)
        K = autocorrelation_kernel(F, scale=sc, n_bins=spec.n_bins, blocks=blocks)
        out[f"sigma {sigma:g}, linear"] = K
        scale = median_gamma_scale(K)
        for g in GAMMAS:
            out[f"sigma {sigma:g}, gauss {g:g}"] = gauss_layer(K, g, scale)
        log(f"  autocorrelation sigma {sigma:g}: {time.time() - t0:.0f} s")
    return out


def kernels_for(kind: str, series: Series, material: str, engine=None,
                force: bool = False) -> dict:
    """The settings of one kernel kind, from the cache when they are there."""
    n = len(series.y)
    if kind == "autocorrelation":
        names = [f"sigma {s:g}, {lay}" for s in AC_SIGMAS
                 for lay in ["linear"] + [f"gauss {g:g}" for g in GAMMAS]]
    elif kind == "arrangement":
        names = [f"alpha {a:g}, gamma {g:g}{c}" for a in ALPHAS for g in GAMMAS
                 for c in ("", ", cosine")]
    else:
        names = [f"alpha {a:g}, {lay}" for a in ALPHAS
                 for lay in ["linear"] + [f"gauss {g:g}" for g in GAMMAS]]
    paths = {s: kernel_path(material, n, kind, s) for s in names}
    if all(os.path.isfile(p) for p in paths.values()) and not force:
        return {s: np.load(p) for s, p in paths.items()}
    if kind == "autocorrelation":
        got = autocorrelation_kernels(series)
        for s, K in got.items():
            np.save(paths[s], K)
        return got
    for alpha in ALPHAS:
        if all(os.path.isfile(paths[s]) for s in names if s.startswith(f"alpha {alpha:g},")) \
                and not force:
            continue
        got = overlap_kernels(series, alpha, engine=engine)
        for k, settings in got.items():
            for s, K in settings.items():
                np.save(kernel_path(material, n, k, s), K)
    return {s: np.load(p) for s, p in paths.items()}


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #

def unit_diagonal(Ks: dict) -> dict:
    """Every kernel scaled to a mean self-kernel of one, so that the ridge grid is relative."""
    return {s: K / max(float(np.mean(np.diag(K))), 1e-300) for s, K in Ks.items()}


def choose(Ks: dict, y: np.ndarray, tr: np.ndarray):
    """The setting and ridge with the smallest inner cross-validated RMSE on `tr`."""
    best = None
    for setting, K in Ks.items():
        K_tr = K[np.ix_(tr, tr)]
        for lam in LAMS:
            p = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=INNER, seed=0)
            err = rmse(y[tr], p)
            if best is None or err < best[0]:
                best = (err, setting, lam)
    return best


def evaluate(Ks: dict, y: np.ndarray, n_defects: np.ndarray, classes: np.ndarray,
             splits) -> dict:
    """Nested cross-validation over the settings and the ridge on the given splits.

    Besides the usual errors, the error on the test cells whose arrangement class has no
    member in the training part ("unseen"): the others are, to a symmetry-averaged kernel,
    structures it was trained on.
    """
    pred = np.full(len(y), np.nan)
    seen = np.zeros(len(y), bool)
    per_fold = []
    for k, (tr, te) in enumerate(splits):
        err, setting, lam = choose(Ks, y, tr)
        K = Ks[setting]
        learner = _Learner("krr", {"lam": lam}, True).fit(K[np.ix_(tr, tr)], y[tr])
        pred[te] = learner.predict(K[np.ix_(te, tr)])
        seen[te] = np.isin(classes[te], classes[tr])
        e = np.abs(pred[te] - y[te])
        es = e / n_defects[te]
        new = ~seen[te]
        per_fold.append({"fold": k, "setting": setting, "lam": lam, "inner_rmse": float(err),
                         "n_train": int(len(tr)), "n_test": int(len(te)),
                         "n_unseen": int(new.sum()),
                         "mae": float(np.mean(e)), "mae_site": float(np.mean(es)),
                         "mae_site_unseen": float(np.mean(es[new])) if new.any() else float("nan"),
                         "rmse": float(rmse(y[te], pred[te])),
                         "r2": float(r2_score(y[te], pred[te]))})
        log(f"  fold {k}: {setting}, lam {lam:g}, R2 {per_fold[-1]['r2']:+.3f}, "
            f"MAE {per_fold[-1]['mae']:.4f} eV, {1000 * per_fold[-1]['mae_site']:.1f} meV per "
            f"site; {int(new.sum())} of {len(te)} arrangements unseen, "
            f"{1000 * per_fold[-1]['mae_site_unseen']:.1f} meV per site on them")
    ok = np.isfinite(pred)
    e = np.abs(pred[ok] - y[ok])
    es = e / n_defects[ok]
    new = ~seen[ok]
    return {"r2": float(r2_score(y[ok], pred[ok])), "rmse": float(rmse(y[ok], pred[ok])),
            "mae": float(np.mean(e)), "mae_site": float(np.mean(es)),
            "n_unseen": int(new.sum()), "n_tested": int(ok.sum()),
            "mae_site_unseen": float(np.mean(es[new])) if new.any() else float("nan"),
            "mae_site_seen": float(np.mean(es[~new])) if (~new).any() else float("nan"),
            "r2_unseen": float(r2_score(y[ok][new], pred[ok][new])) if new.sum() > 2 else float("nan"),
            "folds": per_fold, "predictions": pred.tolist(), "seen": seen.tolist()}


def fit(materials, kinds=KERNELS, limit: int = LIMIT, engine=None, force: bool = False) -> dict:
    path = result("fit.json")
    done = jload(path) if os.path.isfile(path) and not force else {}
    for material in materials:
        prep = prepare(material)
        series = Series(prep, pick(prep, limit))
        n = len(series.y)
        few = series.n_defects <= 2
        splits_extra = [(np.where(few)[0], np.where(~few)[0])]
        log(f"=== {material}: {n} cells in {len(set(series.classes))} arrangement classes, "
            f"{int(few.sum())} with one or two defects, target {TARGET} mean "
            f"{series.y.mean():.3f} sd {series.y.std():.3f} eV; {len(series.symmetry[0])} "
            f"symmetry operations")
        for kind in kinds:
            key = f"{material} | {kind} | {n}"
            if key in done and "extrapolation" in done[key]:
                continue
            log(f"--- {key}")
            t0 = time.time()
            Ks = unit_diagonal(kernels_for(kind, series, material, engine=engine))
            out = evaluate(Ks, series.y, series.n_defects, series.classes, folds(n))
            log(f"  random folds: R2 {out['r2']:+.4f}, MAE {out['mae']:.4f} eV, "
                f"{1000 * out['mae_site']:.1f} meV per site; {out['n_unseen']} unseen "
                f"arrangements at {1000 * out['mae_site_unseen']:.1f} meV per site")
            log("  one and two defects -> three:")
            extra = evaluate(Ks, series.y, series.n_defects, series.classes, splits_extra)
            out.update({"material": material, "kernel": kind, "n": n, "seconds": time.time() - t0,
                        "settings": list(Ks), "fields": [s.name for s in fieldset(1.0).specs],
                        "grid": {"alphas": list(ALPHAS), "gammas": list(GAMMAS),
                                 "sigmas": list(AC_SIGMAS)},
                        "n_defects": series.n_defects.tolist(), "y": series.y.tolist(),
                        "extrapolation": extra})
            done[key] = out
            jdump(done, path)
    return done


# --------------------------------------------------------------------------- #
# Reading the counts model: what each kind of defect costs
# --------------------------------------------------------------------------- #

def reading(materials, alpha: float = 2.0, lam: float = 1e-3, sample: int = READING_SAMPLE,
            force: bool = False) -> dict:
    """The per-defect contributions of the symmetry-averaged linear model (rule 3).

    Fitted through `CMFModel` on a sample of cells, so that the contributions come from the
    program's own decomposition and not from a side calculation.  Under the host symmetry the
    contribution of a defect depends on its kind and nothing else, so the table is one number
    per kind, with the spread over the cells as a check that this is so.
    """
    path = result("reading.json")
    done = jload(path) if os.path.isfile(path) and not force else {}
    for material in materials:
        if material in done:
            continue
        prep = prepare(material)
        series = Series(prep, pick(prep, sample, seed=SPLIT_SEED + 1))
        fs = fieldset(alpha)
        table = series.differences(alpha)
        t0 = time.time()
        model = CMFModel(fieldset=fs, method="krr", params={"lam": lam}, normalize=False,
                         pbc_overlap=True, pbc_symmetry=series.symmetry)
        model.fit(table, series.y, cv_folds=0)
        pred = model.predict(table)
        per_kind = {}
        for i, mol in enumerate(table.molecules):
            c = model.atom_contributions(mol)
            for t, v in zip(mol.atom_types, c):
                per_kind.setdefault(t, []).append(float(v))
        # a substitution is two rows on one site: report it as one
        kinds = {}
        for t, vals in per_kind.items():
            kinds[t] = {"n": len(vals), "mean": float(np.mean(vals)), "sd": float(np.std(vals))}
        done[material] = {"alpha": alpha, "lam": lam, "n": len(series.y), "bias": float(model.bias),
                          "r2_train": float(r2_score(series.y, pred)),
                          "mae_train": float(np.mean(np.abs(pred - series.y))),
                          "kinds": kinds, "seconds": time.time() - t0}
        log(f"{material}: contributions by kind " +
            ", ".join(f"{t} {v['mean']:+.3f} +- {v['sd']:.3f} ({v['n']})"
                      for t, v in sorted(kinds.items())))
        jdump(done, path)
    return done


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def _interval(y, pa, pb, n_defects):
    """R2 difference with its interval, and the MAE-per-site difference with its interval."""
    d, lo, hi = paired_difference(y, pa, pb)
    y = np.asarray(y, float)
    ea = np.abs(np.asarray(pa) - y) / n_defects
    eb = np.abs(np.asarray(pb) - y) / n_defects
    rng = np.random.RandomState(0)
    diffs = []
    for _ in range(2000):
        i = rng.randint(0, len(y), len(y))
        diffs.append(float(np.mean(ea[i]) - np.mean(eb[i])))
    return ((d, lo, hi), (float(np.mean(ea) - np.mean(eb)), float(np.quantile(diffs, 0.025)),
                          float(np.quantile(diffs, 0.975))))


def report(out: str = "") -> str:
    fits = jload(result("fit.json")) if os.path.isfile(result("fit.json")) else {}
    read = jload(result("reading.json")) if os.path.isfile(result("reading.json")) else {}
    lines = ["# A defect series on one host lattice: the periodic overlap with the host "
             "symmetry (item M14)", ""]
    lines += ["2DMD (Huang et al. 2023), the low-density sets: one to three point defects on an "
              "8 x 8 supercell of MoS2 or WSe2, formation energy of the cell in eV from DFT. The "
              "structures are the unrelaxed ideal supercell, so every cell stands on one "
              "lattice exactly. Five random folds; the setting (attenuation or distance width, "
              "and the bandwidth of the Gaussian layer) and the ridge chosen by five-fold "
              "cross-validation inside each training part. MAE per site is the error of the "
              "cell divided by its number of defects, which is the quantity the literature on "
              "this set reports.", ""]
    by_material = {}
    for key, r in fits.items():
        by_material.setdefault((r["material"], r["n"]), {})[r["kernel"]] = r
    for (material, n), kinds in sorted(by_material.items()):
        any_r = next(iter(kinds.values()))
        grid = any_r.get("grid")
        lines += [f"## {material}: {n} cells", ""]
        if grid:
            lines += ["Settings searched: attenuation alpha in {" + ", ".join(
                f"{a:g}" for a in grid["alphas"]) + "}; bandwidth of the Gaussian layer gamma "
                "in {" + ", ".join(f"{g:g}" for g in grid["gammas"]) + "} times the median "
                "squared distance; autocorrelation width sigma in {" + ", ".join(
                f"{s:g}" for s in grid["sigmas"]) + "} A; ridge in {" + ", ".join(
                f"{l:g}" for l in LAMS) + "} times the mean self-kernel.", ""]
        lines += [f"Of the {any_r.get('n_tested', n)} cells, "
                  f"{any_r.get('n_unseen', 0)} have an arrangement "
                  f"that no cell of their training fold shares under the host symmetry "
                  f"(\"unseen\"); the rest are, to a symmetry-averaged kernel, structures the "
                  f"model was trained on, and their error is the reproducibility of the DFT "
                  f"labels rather than a prediction.", "",
                  "| kernel | R2 | MAE, eV per cell | MAE, meV per site | MAE on unseen "
                  "arrangements, meV per site | R2 on unseen | settings chosen |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for kind in KERNELS:
            if kind not in kinds:
                continue
            r = kinds[kind]
            chosen = sorted({f["setting"] for f in r["folds"]})
            lines.append(f"| {kind} | {r['r2']:+.3f} | {r['mae']:.4f} | "
                         f"{1000 * r['mae_site']:.1f} | "
                         f"{1000 * r.get('mae_site_unseen', float('nan')):.1f} | "
                         f"{r.get('r2_unseen', float('nan')):+.3f} | {', '.join(chosen)} |")
        lines.append("")
        if "autocorrelation" in kinds:
            ref = kinds["autocorrelation"]
            lines += ["Against the autocorrelation, paired over the cells (bootstrap, 95 %): "
                      "the difference in R2 and in MAE per site.", "",
                      "| kernel | R2 difference | MAE per site difference, meV |",
                      "|---|---:|---:|"]
            nd = np.asarray(ref["n_defects"], float)
            for kind in KERNELS:
                if kind not in kinds or kind == "autocorrelation":
                    continue
                r = kinds[kind]
                (d, lo, hi), (m, mlo, mhi) = _interval(ref["y"], r["predictions"],
                                                       ref["predictions"], nd)
                lines.append(f"| {kind} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                             f"{1000 * m:+.1f} [{1000 * mlo:+.1f}, {1000 * mhi:+.1f}] |")
            lines.append("")
        lines += ["**From one and two defects to three.** Trained on the cells with one or two "
                  "defects, predicting those with three.", "",
                  "| kernel | cells trained on | R2 | MAE, eV per cell | MAE, meV per site | "
                  "setting |", "|---|---:|---:|---:|---:|---|"]
        for kind in KERNELS:
            if kind not in kinds:
                continue
            e = kinds[kind]["extrapolation"]
            f0 = e["folds"][0]
            lines.append(f"| {kind} | {f0['n_train']} | {e['r2']:+.3f} | {e['mae']:.4f} | "
                         f"{1000 * e['mae_site']:.1f} | {f0['setting']}, lam {f0['lam']:g} |")
        lines.append("")
        if material in read:
            rd = read[material]
            lines += [f"**What each kind of defect costs** (the per-defect contributions of the "
                      f"symmetry-averaged linear model, decomposition rule 3; {rd['n']} cells, "
                      f"alpha {rd['alpha']:g}, training R2 {rd['r2_train']:+.3f}). A vacancy is "
                      f"one row (the species removed); a substitution is two rows on one site, "
                      f"the species removed and the species added, whose sum is its cost. The "
                      f"spread is over the cells and should be zero: under the host symmetry a "
                      f"defect's contribution depends on its kind only.", "",
                      "| row | cells | contribution, eV | spread |", "|---|---:|---:|---:|"]
            for t, v in sorted(rd["kinds"].items()):
                lines.append(f"| {t} | {v['n']} | {v['mean']:+.3f} | {v['sd']:.4f} |")
            lines.append("")
    lines += ["## Context: the published errors on this set", "",
              "Kazeev et al. 2023, their own test split (not these folds), MAE of the formation "
              "energy per site in meV: MoS2 -- sparse graph network 2.6, gradient boosting on "
              "descriptors 11.8, whole-structure graph networks 34 to 48; WSe2 -- 4.8, 14.7 and "
              "31 to 55.", ""]
    text = "\n".join(lines)
    out = out or result("table.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return out


def main():
    global ALPHAS, GAMMAS, AC_SIGMAS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("prepare", "fit", "reading"), default=None)
    ap.add_argument("--material", nargs="+", default=list(MATERIALS))
    ap.add_argument("--kernels", nargs="+", default=list(KERNELS))
    ap.add_argument("--limit", type=int, default=LIMIT)
    ap.add_argument("--engine", default=None, help="kernel engine: numpy, numba or auto")
    ap.add_argument("--alpha", type=float, default=2.0, help="attenuation of the reading model")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="attenuations searched (default %s)" % (ALPHAS,))
    ap.add_argument("--gammas", type=float, nargs="+", default=None,
                    help="layer bandwidths searched (default %s)" % (GAMMAS,))
    ap.add_argument("--sigmas", type=float, nargs="+", default=None,
                    help="autocorrelation widths searched (default %s)" % (AC_SIGMAS,))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.alphas:
        ALPHAS = tuple(args.alphas)
    if args.gammas:
        GAMMAS = tuple(args.gammas)
    if args.sigmas:
        AC_SIGMAS = tuple(args.sigmas)
    if args.report:
        report()
        return 0
    stages = (args.stage,) if args.stage else ("prepare", "fit", "reading")
    for stage in stages:
        log(f"=== {stage}")
        if stage == "prepare":
            for m in args.material:
                prepare(m, force=args.force)
        elif stage == "fit":
            fit(args.material, kinds=args.kernels, limit=args.limit, engine=args.engine,
                force=args.force)
        elif stage == "reading":
            reading(args.material, alpha=args.alpha, force=args.force)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
