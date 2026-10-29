#!/usr/bin/env python
"""
Continuous molecular fields of polybenzenoid hydrocarbons: the HOMO-LUMO gap of COMPAS-1.

COMPAS-1 (Wahab, Pfuderer, Paenurk, Gershoni-Poranne, J. Chem. Inf. Model. 2022, 62, 3704) is
the exhaustive enumeration of the cata-condensed polybenzenoid hydrocarbons of one to eleven
rings: 8,678 molecules optimised at B3LYP-D3BJ/def2-SVP (COMPAS-1D) and 34,072 at GFN2-xTB
(COMPAS-1x), each with its HOMO and LUMO energies and their gap.  The molecules are rigid and
nearly planar, so the frame that a 3D-QSAR model needs is given by a rule
(:func:`cmfqsar.pbh.canonical_frame`) and the kernel ridge regression on the overlap of the
molecular fields can be judged on the fields alone.

Fields and field sets
---------------------
    A   steric (Tripos well depths, radius-scaled widths; hydrogens included)
    B   electrostatic: GFN2-xTB partial charges (``prop:q_xtb``), Gasteiger on request
    C   continuous indicator fields of the three carbon classes of a benzenoid:
        ``cif:C.H`` (rim carbon), ``cif:C.2r`` (two rings), ``cif:C.3r`` (three rings; absent
        in a cata-condensed set and dropped when no molecule has one)
    D   ring-centred indicator fields of the annulation classes: ``ring:terminal``,
        ``ring:linear``, ``ring:angular``, ``ring:branched`` (one Gaussian per ring centroid)
    E   ``ring:sextet``: the ring centroids weighted by their share of the Clar sextets, a
        placeholder for an aromaticity field (a NICS-weighted field would need a DFT run per ring)

Field sets: {A}, {A,B}, {C}, {D}, {A,B,C}, {A,B,C,D}, {C,D} and {A,B,C,D,E}.

Protocol
--------
Random splits: 80/10/10 train/validation/test for five seeds.  Size-extrapolation splits: train
on the molecules with fewer than N rings, test on the molecules with N rings, N = 8, 9, 10.
For every field set and split:

1. the attenuation shared by the groups of the set, then one attenuation per group by
   coordinate descent, both chosen by 5-fold cross-validation (RMSE over the ridge grid) on a
   random subset of 2,500 training molecules;
2. the ridge for each of the two attenuation choices by 5-fold cross-validation on the whole
   training part;
3. the choice between the shared and the per-group attenuation by the RMSE on the validation
   part (by the cross-validation RMSE where there is none);
4. the final fit on the whole training part; R2, RMSE and MAE on the test part.

Every field kernel is scaled to unit mean diagonal over the whole set (the field-scaling
convention of the program, with the constant taken over all molecules rather than the training
part so that one kernel per attenuation serves every split), the kernels of a group are summed,
and the group kernels are stored once per attenuation in ``benchmarks/cache/compas/kernels``
(float32).  Kernel ridge regression with the bias term solved as in Eq. 12 of the 2013 paper
(the augmented system is the same as ridge regression on the centred kernel, which is what the
eigendecomposition inside the cross-validation uses; the script checks the two agree).

Baselines on the same splits: ridge regression on the counts of the atom and ring classes (the
attenuation-to-zero limit of the indicator and ring fields, an additivity scheme), a random
forest on RDKit descriptors, and a straight line in the number of rings.

Interpretation: the {A,B,C,D} model of seed 0 is refitted as a :class:`cmfqsar.model.CMFModel`
and its co-fields are drawn in the molecular plane of a reference molecule, the per-ring
contributions (decomposition rule 3 summed over the atoms of every ring) are drawn for four test
molecules, and a short text lists which rings and zones raise or lower the gap.

The data are downloaded once to ``benchmarks/cache/compas`` (CC-BY-NC-SA, not part of the
repository); results go to ``benchmarks/results/compas``.

Usage
-----
    python benchmarks/run_compas.py                      # every stage on COMPAS-1D, then the tables
    python benchmarks/run_compas.py --stage prepare      # download, parse, align, xTB charges, splits
    python benchmarks/run_compas.py --stage kernels      # the group kernels on the attenuation grid
    python benchmarks/run_compas.py --stage fields       # the field sets on every split
    python benchmarks/run_compas.py --stage baselines
    python benchmarks/run_compas.py --stage interpret
    python benchmarks/run_compas.py --stage scaling --set 1x --limit 2000   # the Nystroem path
    python benchmarks/run_compas.py --report
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar import pbh  # noqa: E402
from cmfqsar.align import set_coords  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, FieldSpec, MoleculeFields  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.model import CMFModel, _kfold_indices, krr_fit, mae, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

CACHE = os.path.join(HERE, "cache", "compas")
RESULTS = os.path.join(HERE, "results", "compas")
BASE_URL = "https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/"
FILES = {
    "1D": ("compas-1D_b3lyp_def2-svp.csv", "compas-1D_b3lyp_def2-svp.tar.gz"),
    "1x": ("compas-1x.csv", "compas-1x.tar.gz"),
}
LEVELS = {"1D": "B3LYP-D3BJ/def2-SVP", "1x": "GFN2-xTB"}

#: attenuation grid, in the convention of the program (field = w exp(-alpha/2 r^2), alpha in A^-2;
#: the prompt's alpha' in exp(-alpha'^2 r^2) is sqrt(alpha/2): 0.22 to 1.6 A^-1 here)
ALPHAS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
LAMS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)
#: the resolution stage: attenuations below, between and above the grid, a finer and smaller
#: ridge, and the widths of the Gaussian kernel layer relative to the median squared distance
ALPHAS_EXT = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 20.0)
LAMS_FINE = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SEEDS = (0, 1, 2, 3, 4)
FOLDS = 5
SUBSAMPLE = 2500                     # training molecules used to choose the attenuations
SIZE_TESTS = (8, 9, 10)
TARGET = "GAP_eV"

GROUP_FIELDS = {
    "A": ["steric"],
    "B": ["prop:q_xtb"],
    "C": ["cif:C.H", "cif:C.2r", "cif:C.3r"],
    "D": list(pbh.RING_FIELDS),
    "E": [pbh.SEXTET_FIELD],
}
GROUP_LABELS = {"A": "steric", "B": "electrostatic", "C": "atom-class indicator",
                "D": "ring-class indicator", "E": "Clar-sextet ring"}
FIELD_SETS = ("A", "AB", "C", "D", "ABC", "ABCD", "CD", "ABCDE")
MAIN_SET = "ABCD"
NYSTROEM_M = (500, 1000, 2000)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def jdump(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def results_path(set_key: str, name: str) -> str:
    return os.path.join(RESULTS, f"{name}_{set_key}.json")


# --------------------------------------------------------------------------- #
# Stage: prepare
# --------------------------------------------------------------------------- #

def download(set_key: str) -> Dict[str, dict]:
    """Fetch the csv and the geometry archive of one set; record URL, size and checksum."""
    ensure_dir(CACHE)
    record_path = os.path.join(CACHE, "checksums.json")
    record = jload(record_path) if os.path.isfile(record_path) else {}
    for name in FILES[set_key] + ("README.md",):
        path = os.path.join(CACHE, name)
        url = BASE_URL + name
        if not os.path.isfile(path):
            log(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        with open(path, "rb") as fh:
            head = fh.read(2)
        if name.endswith(".gz") and head != b"\x1f\x8b":
            raise SystemExit(f"{path} is not a gzip archive (a Git-LFS pointer?): ask the user "
                             f"for the file")
        if name not in record:
            record[name] = {"url": url, "bytes": os.path.getsize(path), "sha256": sha256(path),
                            "downloaded": _dt.date.today().isoformat()}
    jdump(record, record_path)
    return record


def extract(set_key: str) -> str:
    folder = os.path.join(CACHE, f"xyz{set_key}")
    if os.path.isdir(folder) and any(os.scandir(folder)):
        return folder
    archive = os.path.join(CACHE, FILES[set_key][1])
    log(f"extracting {archive}")
    with tarfile.open(archive, "r:*") as tar:
        try:
            tar.extractall(folder, filter="data")
        except TypeError:                       # Python < 3.12
            tar.extractall(folder)
    return folder


def prepared_sdf(set_key: str) -> str:
    return os.path.join(CACHE, f"compas-{set_key}.sdf")


def prepare(set_key: str, charges: str = "xtb", jobs: int = 8, limit: Optional[int] = None,
            force: bool = False) -> dict:
    """Download, parse, align, compute the charges, write the SDF, make the splits."""
    from cmfqsar import xtb

    meta_path = results_path(set_key, "prepared")
    if os.path.isfile(meta_path) and os.path.isfile(prepared_sdf(set_key)) and not force:
        return jload(meta_path)
    t0 = time.time()
    record = download(set_key)
    folder = extract(set_key)
    csv_path = os.path.join(CACHE, FILES[set_key][0])
    log(f"reading {csv_path}")
    ds = pbh.read_compas(csv_path, folder, limit=limit,
                         progress=lambda k, n: log(f"  {k}/{n} molecules read"))
    log(f"{len(ds)} molecules; frame and rings")
    mismatches, peri = [], []
    has_c3r = False
    for m in ds.mols:
        set_coords(m, pbh.canonical_frame(m))
        rings = pbh.ring_classes(m)
        rc = pbh.ring_class_counts(rings)
        if any(r.cls == "peri" for r in rings):
            peri.append(m.GetProp("_Name"))
        lin, ang = pbh.annulation_counts(m.GetProp("lalas") if m.HasProp("lalas") else "")
        if len(rings) != int(m.GetProp("n_rings")) or rc["linear"] != lin \
                or rc["angular"] + rc["branched"] != ang:
            mismatches.append(m.GetProp("_Name"))
        if any(pbh.pbh_atom_type(a) == "C.3r" for a in m.GetAtoms()):
            has_c3r = True
    if peri:
        raise SystemExit(f"{len(peri)} molecules have a peri-fused ring (first: {peri[0]}); "
                         "this study is for cata-condensed sets only")
    if mismatches:
        log(f"WARNING: {len(mismatches)} molecules disagree with their annulation string "
            f"(first: {mismatches[0]})")
    substitutions: List[str] = []
    failures: List[dict] = []
    charge_source = "GFN2-xTB Mulliken charges (cmfqsar.xtb, no Fukui run, input geometry)"
    xtb_version = ""
    if charges == "xtb":
        cfg = xtb.Config(fukui=False, workdir=os.path.join(CACHE, f"xtb_{set_key}"))
        try:
            xtb_version = xtb.xtb_version(xtb.find_xtb())
        except Exception as exc:
            raise SystemExit(f"xtb is not available ({exc}); pass --charges gasteiger to go on "
                             "without it (the substitution is recorded)")
        log(f"GFN2-xTB charges for {len(ds)} molecules, {jobs} at a time ({xtb_version})")
        outcomes = xtb.process_molecules(ds.mols, ds.names, cfg, jobs=jobs, log=log)
        keep = []
        for k, oc in enumerate(outcomes):
            if oc.mol is None:
                failures.append({"name": oc.name, "error": oc.error})
            else:
                ds.mols[k] = oc.mol
                keep.append(k)
        if failures:
            log(f"{len(failures)} molecules failed in xtb and are dropped")
            substitutions.append(f"{len(failures)} molecules dropped: xtb failed on them")
            ds.remove([k for k in range(len(ds)) if k not in set(keep)])
    else:
        charge_source = "Gasteiger charges (RDKit); SUBSTITUTION for the GFN2-xTB charges"
        substitutions.append("electrostatic field from Gasteiger charges instead of GFN2-xTB")
        log("SUBSTITUTION: Gasteiger charges instead of GFN2-xTB")
    if not has_c3r:
        substitutions.append("cif:C.3r dropped: no carbon of three rings in a cata-condensed set")
    out = prepared_sdf(set_key)
    with Chem.SDWriter(out) as w:
        for m in ds.mols:
            w.write(m)
    n_rings = [int(m.GetProp("n_rings")) for m in ds.mols]
    meta = {
        "set": set_key, "level": LEVELS[set_key], "n": len(ds), "names": ds.names,
        "n_rings": n_rings, "gap": [float(v) for v in ds.activity],
        "homo": [float(m.GetProp("HOMO_eV")) for m in ds.mols],
        "lumo": [float(m.GetProp("LUMO_eV")) for m in ds.mols],
        "charges": charges, "charge_source": charge_source, "xtb_version": xtb_version,
        "failures": failures, "substitutions": substitutions, "has_C3r": has_c3r,
        "annulation_mismatches": mismatches, "files": {k: record[k] for k in FILES[set_key]},
        "frame": "cmfqsar.pbh.canonical_frame (heavy-atom inertia tensor, sign rule)",
        "seconds": time.time() - t0, "sdf": out, "date": _dt.date.today().isoformat(),
    }
    jdump(meta, meta_path)
    splits = make_splits(n_rings)
    jdump(splits, results_path(set_key, "splits"))
    log(f"prepared {len(ds)} molecules in {time.time() - t0:.0f} s -> {out}")
    return meta


def make_splits(n_rings: Sequence[int]) -> Dict[str, dict]:
    """The random and the size-extrapolation splits, as lists of indices."""
    n = len(n_rings)
    out: Dict[str, dict] = {}
    for seed in SEEDS:
        s = pbh.random_split(n, seed)
        out[f"random{seed}"] = {"kind": "random", "seed": seed,
                                **{k: v.tolist() for k, v in s.items()}}
    for nr in SIZE_TESTS:
        s = pbh.size_split(n_rings, nr)
        if len(s["train"]) and len(s["test"]):
            out[f"size{nr}"] = {"kind": "size", "seed": 0, "test_rings": nr,
                                **{k: v.tolist() for k, v in s.items()}}
    return out


def load_prepared(set_key: str):
    meta = jload(results_path(set_key, "prepared"))
    ds = load_dataset(prepared_sdf(set_key), activity_prop=TARGET)
    if meta["charges"] == "xtb" and "q_xtb" not in ds.atom_properties:
        raise SystemExit(f"{prepared_sdf(set_key)} carries no q_xtb atom property")
    if ds.names != meta["names"]:
        raise SystemExit("the SDF and prepared.json list different molecules; rerun --stage "
                         "prepare --force")
    return ds, meta


# --------------------------------------------------------------------------- #
# Fields and tables
# --------------------------------------------------------------------------- #

def group_fields(group: str, meta: dict) -> List[str]:
    names = list(GROUP_FIELDS[group])
    if group == "B" and meta["charges"] != "xtb":
        names = ["electrostatic"]
    if group == "C" and not meta.get("has_C3r", False):
        names = [n for n in names if n != "cif:C.3r"]
    return names


def make_fieldset(groups: str, meta: dict, alpha_by_group: Optional[Dict[str, float]] = None,
                  h_by_field: Optional[Dict[str, float]] = None) -> FieldSet:
    specs = []
    for g in groups:
        a = (alpha_by_group or {}).get(g, 1.0)
        for name in group_fields(g, meta):
            kind = "steric" if name == "steric" else "gauss"
            specs.append(FieldSpec(name=name, kind=kind, alpha=float(a),
                                   h=float((h_by_field or {}).get(name, 1.0))))
    return FieldSet(specs=specs, charge_model="gasteiger", cif_scheme="pbh")


def build_table(ds, fieldset: FieldSet, ring_cache: Optional[dict] = None) -> AtomTable:
    mfs = []
    want_sextet = any(s.name == pbh.SEXTET_FIELD for s in fieldset.specs)
    for m, name in zip(ds.mols, ds.names):
        rings = None
        if ring_cache is not None:
            rings = ring_cache.get(name)
            if rings is None:
                rings = pbh.ring_classes(m)
                if want_sextet:
                    pbh.clar_weights(m, rings)
                ring_cache[name] = rings
        mfs.append(pbh.build_pbh_fields(m, fieldset, name=name, rings=rings))
    return AtomTable.from_molecules(mfs)


def restrict_table(table: AtomTable, columns: Sequence[int]) -> AtomTable:
    """The table with only the atoms that carry weight in `columns`, and only those columns."""
    cols = list(columns)
    mfs = []
    for mf in table.molecules:
        keep = np.flatnonzero(np.abs(mf.weights[:, cols]).sum(axis=1) > 0)
        if len(keep) == 0:
            keep = np.array([0])
        mfs.append(MoleculeFields(name=mf.name, coords=mf.coords[keep], weights=mf.weights[np.ix_(keep, cols)],
                                  radii=mf.radii[keep], symbols=[mf.symbols[i] for i in keep],
                                  atom_types=[mf.atom_types[i] for i in keep],
                                  bonds=np.zeros((0, 2), int), charges=mf.charges[keep]))
    return AtomTable.from_molecules(mfs)


# --------------------------------------------------------------------------- #
# Stage: kernels
# --------------------------------------------------------------------------- #

def kernel_path(set_key: str, group: str, alpha: float) -> str:
    return os.path.join(CACHE, "kernels", f"{set_key}_{group}_{alpha:g}.npy")


def kernels(set_key: str, groups: str = "ABCDE", engine: Optional[str] = "auto",
            force: bool = False, alphas: Sequence[float] = ALPHAS) -> dict:
    """The group kernels on the whole set for every attenuation of the grid, stored as float32."""
    ds, meta = load_prepared(set_key)
    rec_path = results_path(set_key, "kernels")
    rec = jload(rec_path) if os.path.isfile(rec_path) and not force else {}
    fs_all = make_fieldset(groups, meta)
    log(f"building the fields of {len(ds)} molecules")
    table = build_table(ds, fs_all, ring_cache={})
    log(f"{table.coords.shape[0]} atoms including {sum(1 for m in table.molecules for s in m.symbols if s == pbh.RING_SYMBOL)} ring centres")
    for g in groups:
        cols = [fs_all.names.index(n) for n in group_fields(g, meta)]
        part = restrict_table(table, cols)
        for alpha in alphas:
            key = f"{g}_{alpha:g}"
            path = kernel_path(set_key, g, alpha)
            if os.path.isfile(path) and key in rec and not force:
                continue
            specs = [FieldSpec(name=fs_all.specs[c].name, kind=fs_all.specs[c].kind, alpha=alpha)
                     for c in cols]
            t0 = time.time()
            K = kernel_matrices(part, part, specs, engine=engine)
            diag_mean = [float(np.mean(np.diag(K[f]))) for f in range(len(specs))]
            scale = [1.0 / d if d > 1e-300 else 0.0 for d in diag_mean]
            joint = np.zeros(K.shape[1:], dtype=np.float32)
            for f in range(len(specs)):
                joint += (scale[f] * K[f]).astype(np.float32)
            del K
            ensure_dir(os.path.dirname(path))
            np.save(path, joint)
            rec[key] = {"group": g, "alpha": alpha, "fields": [s.name for s in specs],
                        "diag_mean": diag_mean, "scale": scale, "atoms": int(part.coords.shape[0]),
                        "seconds": time.time() - t0, "engine": engine}
            jdump(rec, rec_path)
            log(f"kernel {key}: {part.coords.shape[0]} atoms, {time.time() - t0:.0f} s")
    return rec


class KernelStore:
    """The stored group kernels of one set, read into memory on request and summed.

    A kernel is read whole (300 MB in float32 for 8,678 molecules) rather than memory-mapped:
    gathering the rows and columns of a split from a memory map is many small reads and was the
    slowest part of a split.  The last `max_loaded` kernels stay in memory.
    """

    def __init__(self, set_key: str, max_loaded: int = 8):
        from collections import OrderedDict
        self.set_key = set_key
        self.rec = jload(results_path(set_key, "kernels"))
        self.max_loaded = int(max_loaded)
        self._maps: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def block(self, group: str, alpha: float) -> np.ndarray:
        key = f"{group}_{alpha:g}"
        arr = self._maps.get(key)
        if arr is None:
            while len(self._maps) >= self.max_loaded:
                self._maps.popitem(last=False)
            arr = np.load(kernel_path(self.set_key, group, alpha))
            self._maps[key] = arr
        else:
            self._maps.move_to_end(key)
        return arr

    def joint(self, alpha_by_group: Dict[str, float], rows: np.ndarray,
              cols: Optional[np.ndarray] = None) -> np.ndarray:
        cols = rows if cols is None else cols
        out = None
        for g, a in alpha_by_group.items():
            part = np.asarray(self.block(g, a)[np.ix_(rows, cols)], dtype=np.float64)
            out = part if out is None else out + part
        return out

    def diag(self, alpha_by_group: Dict[str, float], rows: np.ndarray) -> np.ndarray:
        """The diagonal of the joint kernel on `rows` (the self-overlaps, for the Gaussian layer)."""
        out = np.zeros(len(rows))
        for g, a in alpha_by_group.items():
            out += np.asarray(np.diagonal(self.block(g, a))[rows], dtype=np.float64)
        return out

    def scales(self, alpha_by_group: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for g, a in alpha_by_group.items():
            r = self.rec[f"{g}_{a:g}"]
            out.update(dict(zip(r["fields"], r["scale"])))
        return out


# --------------------------------------------------------------------------- #
# Kernel ridge regression with the bias, cross-validated over the ridge grid
# --------------------------------------------------------------------------- #

def centred(K: np.ndarray) -> np.ndarray:
    r = K.mean(axis=1, keepdims=True)
    c = K.mean(axis=0, keepdims=True)
    return K - r - c + K.mean()


def cv_predictions(K: np.ndarray, y: np.ndarray, folds: Sequence[np.ndarray],
                   lams: Sequence[float]) -> np.ndarray:
    """Out-of-fold predictions for every ridge of `lams`: (len(lams), n).

    One eigendecomposition of the centred training kernel per fold serves every ridge.  The
    solution is that of the augmented system of :func:`cmfqsar.model.krr_fit` (bias b, sum of the
    coefficients zero), which :func:`check_krr_equivalence` verifies.
    """
    n = len(y)
    out = np.zeros((len(lams), n))
    all_idx = np.arange(n)
    for te in folds:
        tr = np.setdiff1d(all_idx, te)
        Ktr = K[np.ix_(tr, tr)]
        ytr = y[tr]
        s, V = np.linalg.eigh(centred(Ktr))
        Vy = V.T @ (ytr - ytr.mean())
        Kte = K[np.ix_(te, tr)]
        for li, lam in enumerate(lams):
            a = V @ (Vy / (s + lam))
            b = float(np.mean(ytr - Ktr @ a))
            out[li, te] = Kte @ a + b
    return out


def median_d2(K: np.ndarray) -> float:
    """The median squared distance between the molecules of a training kernel, off the diagonal."""
    d = np.diag(K)
    D2 = d[:, None] + d[None, :] - 2.0 * K
    n = K.shape[0]
    off = D2[~np.eye(n, dtype=bool)]
    m = float(np.median(np.maximum(off, 0.0)))
    return m if m > 1e-300 else 1.0


def gauss_layer(K: np.ndarray, d_rows: np.ndarray, d_cols: np.ndarray, gamma: float,
                med: float) -> np.ndarray:
    """The Gaussian kernel layer of the program on top of the linear overlap kernel:
    ``exp(-gamma |x_i - x_j|^2 / med)`` with the distances taken in the field space, so that
    gamma = 1 is the median-distance bandwidth (:func:`cmfqsar.model.median_gamma_scale`)."""
    D2 = d_rows[:, None] + d_cols[None, :] - 2.0 * K
    return np.exp(-float(gamma) * np.maximum(D2, 0.0) / med)


def check_krr_equivalence() -> None:
    rng = np.random.RandomState(1)
    X = rng.randn(40, 5)
    K = X @ X.T
    y = rng.randn(40)
    folds = [np.arange(0, 40, 4)]
    tr = np.setdiff1d(np.arange(40), folds[0])
    a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], 0.3)
    ref = K[np.ix_(folds[0], tr)] @ a + b
    got = cv_predictions(K, y, folds, [0.3])[0, folds[0]]
    if np.abs(ref - got).max() > 1e-8:
        raise RuntimeError("the centred-kernel solution does not match krr_fit")


def best_lambda(K: np.ndarray, y: np.ndarray, folds, lams=LAMS) -> Tuple[float, float, np.ndarray]:
    preds = cv_predictions(K, y, folds, lams)
    errs = [rmse(y, p) for p in preds]
    k = int(np.argmin(errs))
    return float(lams[k]), float(errs[k]), preds[k]


def fit_predict(store: KernelStore, alpha_by_group: Dict[str, float], lam: float,
                y_tr: np.ndarray, idx_tr: np.ndarray, others: Dict[str, np.ndarray]
                ) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
    K = store.joint(alpha_by_group, idx_tr)
    a, b = krr_fit(K, y_tr, lam)
    preds = {}
    for name, idx in others.items():
        preds[name] = store.joint(alpha_by_group, idx, idx_tr) @ a + b if len(idx) else np.zeros(0)
    return a, b, preds


def metrics(y: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
    if len(y) == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"), "n": 0}
    return {"r2": r2_score(y, yp), "rmse": rmse(y, yp), "mae": mae(y, yp), "n": int(len(y))}


# --------------------------------------------------------------------------- #
# Stage: fields
# --------------------------------------------------------------------------- #

def evaluate_set(store: KernelStore, groups: str, split: dict, y: np.ndarray,
                 target: str = TARGET, alphas: Sequence[float] = ALPHAS,
                 lams: Sequence[float] = LAMS, layer: Optional[str] = None,
                 gammas: Sequence[float] = GAMMAS) -> dict:
    """One field set on one split: the attenuations, the ridge, the fit and the test metrics.

    With ``layer="gauss"`` a third configuration is added: the per-group attenuations of the
    linear search with the Gaussian kernel layer on top, its width chosen on the subsample.
    """
    t0 = time.time()
    ALPHAS_ = tuple(alphas)
    idx_tr = np.asarray(split["train"], int)
    idx_va = np.asarray(split["valid"], int)
    idx_te = np.asarray(split["test"], int)
    seed = int(split["seed"])
    rng = np.random.RandomState(seed)
    sub = np.sort(rng.choice(idx_tr, size=min(SUBSAMPLE, len(idx_tr)), replace=False))
    folds_sub = _kfold_indices(len(sub), FOLDS, seed)
    folds_tr = _kfold_indices(len(idx_tr), FOLDS, seed)
    trace: List[dict] = []
    # the subsample block of every group kernel on the grid, gathered once per split
    sub_blocks = {(g, a): np.asarray(store.block(g, a)[np.ix_(sub, sub)], dtype=np.float64)
                  for g in groups for a in ALPHAS_}

    def score(alpha_by_group: Dict[str, float], gamma: Optional[float] = None) -> Tuple[float, float]:
        K = sum(sub_blocks[(g, a)] for g, a in alpha_by_group.items())
        if gamma is not None:
            d = np.diag(K).copy()
            K = gauss_layer(K, d, d, gamma, median_d2(K))
        lam, err, _ = best_lambda(K, y[sub], folds_sub, lams)
        trace.append({"alphas": dict(alpha_by_group), "gamma": gamma, "lam": lam, "rmse_cv_sub": err})
        return err, lam

    # 1. the shared attenuation
    shared_best = None
    for alpha in ALPHAS_:
        err, lam = score({g: alpha for g in groups})
        if shared_best is None or err < shared_best[0]:
            shared_best = (err, alpha, lam)
    shared = {g: shared_best[1] for g in groups}
    # 2. one attenuation per group, by coordinate descent from the shared one
    per_group = dict(shared)
    best_err = shared_best[0]
    if len(groups) > 1:
        for _pass in range(2):
            changed = False
            for g in groups:
                for alpha in ALPHAS_:
                    if alpha == per_group[g]:
                        continue
                    cand = dict(per_group)
                    cand[g] = alpha
                    err, _lam = score(cand)
                    if err < best_err - 1e-12:
                        best_err, per_group, changed = err, cand, True
            if not changed:
                break
    # 3. the ridge on the whole training part, for both choices
    configs: Dict[str, Tuple[Dict[str, float], Optional[float]]] = {"shared": (shared, None)}
    if per_group != shared:
        configs["per-group"] = (per_group, None)
    if layer == "gauss":
        # the Gaussian layer on the per-group kernel, its width chosen on the subsample
        best_g = None
        for gamma in gammas:
            err, _lam = score(per_group, gamma=gamma)
            if best_g is None or err < best_g[0]:
                best_g = (err, gamma)
        configs["gauss"] = (per_group, best_g[1])
    results = {}
    for label, (abg, gamma) in configs.items():
        K_tr = store.joint(abg, idx_tr)
        K_va = store.joint(abg, idx_va, idx_tr) if len(idx_va) else np.zeros((0, len(idx_tr)))
        K_te = store.joint(abg, idx_te, idx_tr)
        if gamma is not None:
            d_tr = store.diag(abg, idx_tr)
            med = median_d2(K_tr)
            K_tr = gauss_layer(K_tr, d_tr, d_tr, gamma, med)
            if len(idx_va):
                K_va = gauss_layer(K_va, store.diag(abg, idx_va), d_tr, gamma, med)
            K_te = gauss_layer(K_te, store.diag(abg, idx_te), d_tr, gamma, med)
        lam, err_cv, pred_cv = best_lambda(K_tr, y[idx_tr], folds_tr, lams)
        a, b = krr_fit(K_tr, y[idx_tr], lam)
        del K_tr
        pv = K_va @ a + b if len(idx_va) else np.zeros(0)
        pt = K_te @ a + b
        results[label] = {
            "alphas": abg, "gamma": gamma, "lam": lam, "q2": r2_score(y[idx_tr], pred_cv),
            "rmse_cv": err_cv, "valid": metrics(y[idx_va], pv), "test": metrics(y[idx_te], pt),
            "y_test_pred": pt, "y_valid_pred": pv,
        }
    # 4. the choice between them
    if len(idx_va):
        chosen = min(results, key=lambda k: results[k]["valid"]["rmse"])
    else:
        chosen = min(results, key=lambda k: results[k]["rmse_cv"])
    out = {
        "groups": groups, "fields": sum((list(store.rec[f"{g}_{ALPHAS_[0]:g}"]["fields"]) for g in groups), []),
        "split": split.get("name", ""), "kind": split["kind"], "seed": seed, "target": target,
        "n_train": int(len(idx_tr)), "n_valid": int(len(idx_va)), "n_test": int(len(idx_te)),
        "subsample": int(len(sub)), "chosen": chosen, "configs": results,
        "alphas_grid": list(ALPHAS_), "lams_grid": list(lams), "layer": layer,
        "search_trace": trace, "seconds": time.time() - t0,
    }
    return out


def fields_files(set_key: str) -> List[str]:
    """The result files of the field-set stage: ``fields_<set>.json`` and one
    ``fields_<set>_<groups>.json`` per field set, so that several field sets can run at once."""
    if not os.path.isdir(RESULTS):
        return []
    return sorted(os.path.join(RESULTS, f) for f in os.listdir(RESULTS)
                  if f.startswith("fields_") and f.endswith(f"_{set_key}.json"))


def load_fields(set_key: str) -> dict:
    """Every finished (target, field set, split) of the stage, merged over its files."""
    done: dict = {}
    for path in fields_files(set_key):
        done.update(jload(path))
    return done


def fields(set_key: str, sets: Sequence[str] = FIELD_SETS, splits_wanted: Optional[Sequence[str]] = None,
           targets: Sequence[str] = (TARGET,), force: bool = False) -> dict:
    check_krr_equivalence()
    ds, meta = load_prepared(set_key)
    store = KernelStore(set_key, max_loaded=4)
    splits = jload(results_path(set_key, "splits"))
    done = {} if force else load_fields(set_key)
    ys = {TARGET: np.asarray(meta["gap"], float), "HOMO_eV": np.asarray(meta["homo"], float),
          "LUMO_eV": np.asarray(meta["lumo"], float)}
    for target in targets:
        for groups in sets:
            path = results_path(set_key, f"fields_{groups}")
            for sname, split in splits.items():
                if splits_wanted and sname not in splits_wanted:
                    continue
                key = f"{target}|{groups}|{sname}"
                if key in done:
                    continue
                split = dict(split, name=sname)
                res = evaluate_set(store, groups, split, ys[target], target=target)
                res["y_test"] = ys[target][np.asarray(split["test"], int)]
                done[key] = res
                part = jload(path) if os.path.isfile(path) and not force else {}
                part[key] = res
                jdump(part, path)
                c = res["configs"][res["chosen"]]
                log(f"{key}: {res['chosen']} alphas {c['alphas']} lam {c['lam']:g} "
                    f"q2 {c['q2']:.3f} test r2 {c['test']['r2']:.3f} rmse {c['test']['rmse']:.3f} "
                    f"({res['seconds']:.0f} s)")
    return done


# --------------------------------------------------------------------------- #
# Stage: baselines
# --------------------------------------------------------------------------- #

def descriptor_matrix(mols) -> Tuple[np.ndarray, List[str]]:
    from rdkit.Chem import Descriptors
    rows = []
    names = None
    for m in mols:
        d = Descriptors.CalcMolDescriptors(Chem.RemoveHs(m))
        if names is None:
            names = sorted(d)
        rows.append([float(d[k]) if d[k] is not None else np.nan for k in names])
    X = np.array(rows, float)
    ok = np.isfinite(X).all(axis=0) & (np.nanstd(X, axis=0) > 0)
    return X[:, ok], [n for n, k in zip(names, ok) if k]


def baselines(set_key: str, force: bool = False) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    ds, meta = load_prepared(set_key)
    splits = jload(results_path(set_key, "splits"))
    path = results_path(set_key, "baselines")
    if os.path.isfile(path) and not force:
        return jload(path)
    y = np.asarray(meta["gap"], float)
    log("count features and RDKit descriptors")
    counts = np.array([pbh.count_features(m) for m in ds.mols])
    X_desc, desc_names = descriptor_matrix(ds.mols)
    n_rings = np.asarray(meta["n_rings"], float)[:, None]
    out = {"count_names": list(pbh.COUNT_NAMES), "n_descriptors": len(desc_names), "splits": {}}
    ridge_grid = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
    for sname, split in splits.items():
        tr = np.asarray(split["train"], int)
        te = np.asarray(split["test"], int)
        seed = int(split["seed"])
        folds = _kfold_indices(len(tr), FOLDS, seed)
        rec = {}
        # ridge on the counts, the ridge by the same 5-fold CV
        sc = StandardScaler().fit(counts[tr])
        Xtr, Xte = sc.transform(counts[tr]), sc.transform(counts[te])
        best = None
        for lam in ridge_grid:
            pred = np.zeros(len(tr))
            for f in folds:
                rest = np.setdiff1d(np.arange(len(tr)), f)
                mdl = Ridge(alpha=lam).fit(Xtr[rest], y[tr][rest])
                pred[f] = mdl.predict(Xtr[f])
            err = rmse(y[tr], pred)
            if best is None or err < best[0]:
                best = (err, lam, r2_score(y[tr], pred))
        mdl = Ridge(alpha=best[1]).fit(Xtr, y[tr])
        rec["count_ridge"] = {"lam": best[1], "rmse_cv": best[0], "q2": best[2],
                              "test": metrics(y[te], mdl.predict(Xte)),
                              "coef": dict(zip(pbh.COUNT_NAMES, (mdl.coef_ / sc.scale_).tolist()))}
        # a straight line in the number of rings
        lin = LinearRegression().fit(n_rings[tr], y[tr])
        rec["rings_line"] = {"test": metrics(y[te], lin.predict(n_rings[te])),
                             "slope": float(lin.coef_[0]), "intercept": float(lin.intercept_)}
        # random forest on RDKit descriptors
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed, min_samples_leaf=1)
        rf.fit(X_desc[tr], y[tr])
        rec["rf_descriptors"] = {"test": metrics(y[te], rf.predict(X_desc[te])),
                                 "seconds": time.time() - t0}
        out["splits"][sname] = rec
        log(f"baselines {sname}: counts r2 {rec['count_ridge']['test']['r2']:.3f}  "
            f"rings {rec['rings_line']['test']['r2']:.3f}  "
            f"RF {rec['rf_descriptors']['test']['r2']:.3f}")
        jdump(out, path)
    return out


# --------------------------------------------------------------------------- #
# Stage: interpret
# --------------------------------------------------------------------------- #

def chosen_config(fields_done: dict, groups: str, split_name: str, target: str = TARGET) -> dict:
    res = fields_done[f"{target}|{groups}|{split_name}"]
    return res["configs"][res["chosen"]]


def refit_model(set_key: str, groups: str = MAIN_SET, split_name: str = "random0",
                engine: Optional[str] = "auto"):
    """The chosen model of one field set and split as a fitted CMFModel (for the co-fields)."""
    ds, meta = load_prepared(set_key)
    store = KernelStore(set_key)
    fields_done = load_fields(set_key)
    cfg = chosen_config(fields_done, groups, split_name)
    split = jload(results_path(set_key, "splits"))[split_name]
    abg = {g: float(a) for g, a in cfg["alphas"].items()}
    # the field scales are carried as mixing coefficients so that the refit is the model of the
    # table exactly (scale_fields would recompute them over the training part)
    fs = make_fieldset(groups, meta, alpha_by_group=abg, h_by_field=store.scales(abg))
    ring_cache: dict = {}
    table = build_table(ds, fs, ring_cache=ring_cache)
    idx_tr = np.asarray(split["train"], int)
    idx_te = np.asarray(split["test"], int)
    y = np.asarray(meta["gap"], float)
    model = CMFModel(fieldset=fs, method="krr", params={"lam": float(cfg["lam"])},
                     scale_fields=False, engine=engine)
    t0 = time.time()
    log(f"refitting {groups} on {len(idx_tr)} molecules (alphas {abg}, lam {cfg['lam']:g})")
    model.fit(table.subset(idx_tr), y[idx_tr], cv_folds=FOLDS, cv_seed=int(split["seed"]))
    log(f"refit in {time.time() - t0:.0f} s; r2 {model.stats.r2:.3f} q2 {model.stats.q2:.3f}")
    pt = model.predict(table.subset(idx_te))
    diff = np.abs(pt - np.asarray(cfg["y_test_pred"]))
    k = int(np.argmax(diff))
    big = np.asarray(meta["n_rings"], int)[idx_te] >= 3
    dev_big = float(diff[big].max()) if big.any() else float("nan")
    log(f"largest difference to the table's test predictions: {diff[k]:.2e} eV at "
        f"{ds.names[idx_te[k]]}, {dev_big:.2e} eV over the molecules of three rings and more "
        f"(float32 kernels, ridge {cfg['lam']:g})")
    return model, table, ds, meta, split, ring_cache, {
        "test": metrics(y[idx_te], pt), "max_dev": float(diff[k]), "max_dev_molecule": ds.names[idx_te[k]],
        "max_dev_three_rings_and_more": dev_big}


def pick_reference(ds, meta: dict, split: dict) -> int:
    """A test molecule of five or six rings with at least one angular and one linear ring."""
    te = [int(i) for i in split["test"]]
    best = None
    for i in te:
        nr = int(meta["n_rings"][i])
        if nr not in (5, 6):
            continue
        lalas = ds.mols[i].GetProp("lalas") if ds.mols[i].HasProp("lalas") else ""
        lin, ang = pbh.annulation_counts(lalas)
        score = (nr == 6, lin >= 1 and ang >= 1, lin + ang)
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        # fall back to the smallest test molecule with an angular ring
        cands = sorted(te, key=lambda i: int(meta["n_rings"][i]))
        best = (None, cands[0])
    return best[1]


def cofield_plane(model: CMFModel, f: int, xs: np.ndarray, ys: np.ndarray, z: float) -> np.ndarray:
    from cmfqsar.cofields import evaluate_gaussians
    centres, coeffs, widths = model.cofield_coefficients(f)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    # small chunks of points: the steric Gaussians are wide (b = alpha / R^2) and every point
    # touches tens of thousands of the 460k training Gaussians, so a large chunk runs out of memory
    v = evaluate_gaussians(pts, centres, coeffs, widths, chunk=256)
    v *= model.fieldset.specs[f].h * (model.scale[f] if model.scale is not None else 1.0)
    return v.reshape(X.shape)


def draw_skeleton(ax, mf: MoleculeFields, rings: Sequence[pbh.Ring], lw: float = 1.2,
                  centres: bool = True) -> None:
    """The heavy-atom bonds of a molecule, and a cross at every ring centroid when `centres`."""
    for a, b in np.asarray(mf.bonds, int).reshape(-1, 2):
        if mf.symbols[a] == "H" or mf.symbols[b] == "H":
            continue
        ax.plot([mf.coords[a, 0], mf.coords[b, 0]], [mf.coords[a, 1], mf.coords[b, 1]],
                color="black", lw=lw, zorder=4)
    if centres:
        for r in rings:
            ax.plot(r.centroid[0], r.centroid[1], marker="+", color="black", ms=5, mew=0.8, zorder=5)


def plot_cofields(model: CMFModel, mf: MoleculeFields, rings: Sequence[pbh.Ring], meta: dict,
                  path: str, planes: Sequence[float] = (0.0, 1.0), title: str = "") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pad = 3.0
    xs = np.arange(mf.coords[:, 0].min() - pad, mf.coords[:, 0].max() + pad, 0.15)
    ys = np.arange(mf.coords[:, 1].min() - pad, mf.coords[:, 1].max() + pad, 0.15)
    # one panel per field group: the co-fields of different fields are not on one scale (a
    # charge of 0.03 and an indicator of 1 meet co-fields of different size), so no sum is drawn
    groups = [g for g in "ABCDE" if any(n in model.fieldset.names for n in group_fields(g, meta))]
    cols = [(g, z) for z in planes for g in groups]
    fig, axes = plt.subplots(len(planes), len(groups),
                             figsize=(3.3 * len(groups), 2.6 * len(planes)), squeeze=False)
    slices: Dict[Tuple[str, float], np.ndarray] = {}
    for g, z in cols:
        v = np.zeros((len(ys), len(xs)))
        for name in group_fields(g, meta):
            if name in model.fieldset.names:
                v += cofield_plane(model, model.fieldset.names.index(name), xs, ys, z)
        slices[(g, z)] = v
    extremes = {}
    for (g, z), v in slices.items():
        i, j = np.unravel_index(np.argmax(v), v.shape)
        k, l = np.unravel_index(np.argmin(v), v.shape)
        extremes[f"{g}|z={z:g}"] = {"max": float(v[i, j]), "max_xy": [float(xs[j]), float(ys[i])],
                                   "min": float(v[k, l]), "min_xy": [float(xs[l]), float(ys[k])]}
    for row, z in enumerate(planes):
        for col, g in enumerate(groups):
            ax = axes[row, col]
            v = slices[(g, z)]
            vmax = float(np.abs(v).max()) or 1.0
            levels = np.linspace(-vmax, vmax, 21)
            ax.contourf(xs, ys, v, levels=levels, cmap="RdBu_r", extend="both")
            ax.contour(xs, ys, v, levels=[0.0], colors="gray", linewidths=0.5)
            draw_skeleton(ax, mf, rings)
            ax.set_title(f"{GROUP_LABELS[g]}, z = {z:g} A\nmax |C| = {vmax:.2g} eV/A^3", fontsize=8)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title or "Co-fields of the HOMO-LUMO gap (red: raises the gap, blue: lowers it)",
                 fontsize=10)
    fig.tight_layout()
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return extremes


def plot_ring_contributions(items: Sequence[dict], path: str) -> None:
    """Skeletons with every ring filled by its contribution to the gap (eV)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
    vmax = max(float(np.abs(it["per_ring"]).max()) for it in items) or 1.0
    cmap = plt.get_cmap("RdBu_r")
    for ax, it in zip(axes[0], items):
        mf, rings = it["mf"], it["rings"]
        for r, val in zip(rings, it["per_ring"]):
            poly = mf.coords[list(r.atoms), :2]
            ax.add_patch(Polygon(poly, closed=True, facecolor=cmap(0.5 + 0.5 * val / vmax),
                                 edgecolor="none", zorder=2))
            ax.text(r.centroid[0], r.centroid[1], f"{val:+.2f}", ha="center", va="center",
                    fontsize=6.5, zorder=6)
        draw_skeleton(ax, mf, rings, centres=False)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{it['name']}\ngap {it['y']:.2f} eV, predicted {it['yp']:.2f} eV\n"
                     f"bias {it['bias']:.2f} + rings {it['per_ring'].sum():+.2f}", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, label="contribution to the gap (eV)")
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def export_isosurface_html(model: CMFModel, mf: MoleculeFields, mol: Chem.Mol, path: str,
                           cube_path: str) -> bool:
    """A cube file of the total co-field around the reference molecule and, when py3Dmol is
    installed, an HTML page with its isosurfaces.  Returns whether the HTML was written."""
    from cmfqsar.cofields import evaluate_gaussians, export_cube, make_grid
    n_real = pbh.n_real_atoms(mf)
    grid = make_grid(mf.coords[:n_real], margin=2.5, spacing=0.5)
    pts = grid.points()
    # the ring-class co-fields only: the co-fields of different field groups are not on one
    # scale, and the ring fields are the ones that carry the gap
    total = np.zeros(len(pts))
    for f, spec in enumerate(model.fieldset.specs):
        if not spec.name.startswith("ring:"):
            continue
        centres, coeffs, widths = model.cofield_coefficients(f)
        v = evaluate_gaussians(pts, centres, coeffs, widths, chunk=512)
        total += v * spec.h * (model.scale[f] if model.scale is not None else 1.0)
    total = total.reshape(grid.shape)
    atoms_only = MoleculeFields(name=mf.name, coords=mf.coords[:n_real], weights=mf.weights[:n_real],
                                radii=mf.radii[:n_real], symbols=list(mf.symbols[:n_real]),
                                atom_types=list(mf.atom_types[:n_real]), bonds=mf.bonds,
                                charges=mf.charges[:n_real])
    export_cube(cube_path, grid, total, atoms_only)
    try:
        import py3Dmol
    except ImportError:
        log("SUBSTITUTION: py3Dmol is not installed; the 3D view is left out (the cube file is written)")
        return False
    with open(cube_path) as fh:
        cube = fh.read()
    level = 0.3 * float(np.abs(total).max())
    view = py3Dmol.view(width=700, height=500)
    view.addModel(Chem.MolToMolBlock(mol), "mol")
    view.setStyle({"stick": {"radius": 0.12}})
    view.addVolumetricData(cube, "cube", {"isoval": level, "color": "red", "opacity": 0.6})
    view.addVolumetricData(cube, "cube", {"isoval": -level, "color": "blue", "opacity": 0.6})
    view.zoomTo()
    html = view._make_html()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return True


def interpret(set_key: str, groups: str = MAIN_SET, split_name: str = "random0",
              engine: Optional[str] = "auto", force: bool = False) -> dict:
    path = results_path(set_key, "interpret")
    if os.path.isfile(path) and not force:
        return jload(path)
    model, table, ds, meta, split, ring_cache, check = refit_model(set_key, groups, split_name, engine)
    y = np.asarray(meta["gap"], float)
    figdir = ensure_dir(os.path.join(RESULTS, "figures"))
    ref = pick_reference(ds, meta, split)
    ref_name = ds.names[ref]
    mf_ref = table.molecules[ref]
    rings_ref = ring_cache[ref_name]
    log(f"reference molecule {ref_name} ({meta['n_rings'][ref]} rings, gap {y[ref]:.2f} eV)")
    fig_path = os.path.join(figdir, f"cofield_gap_{ref_name}.png")
    extremes = plot_cofields(model, mf_ref, rings_ref, meta, fig_path,
                             title=f"Co-fields of the gap over {ref_name} "
                                   f"({meta['n_rings'][ref]} rings, gap {y[ref]:.2f} eV)")
    # per-ring contributions: the reference and four test molecules (two largest, two smallest gaps)
    te = np.asarray(split["test"], int)
    big = [int(i) for i in te if int(meta["n_rings"][i]) >= 4]
    order = sorted(big, key=lambda i: y[i])
    picks = [order[0], order[1], order[-2], order[-1]]
    items = []
    for i in [ref] + picks:
        mf = table.molecules[i]
        rings = ring_cache[ds.names[i]]
        per_ring, contrib, lost = pbh.ring_contributions(model, mf, rings)
        yp = float(model.predict(table.subset([i]))[0])
        assert abs(per_ring.sum() + lost + model.bias - yp) < 1e-6
        items.append({"name": ds.names[i], "mf": mf, "rings": rings, "per_ring": per_ring,
                      "y": float(y[i]), "yp": yp, "bias": float(model.bias), "lost": lost,
                      "classes": [r.cls for r in rings], "sextet": [r.sextet for r in rings],
                      "centroids": [r.centroid.tolist() for r in rings], "n_rings": len(rings)})
    plot_ring_contributions(items[1:], os.path.join(figdir, "ring_contributions_gap.png"))
    plot_ring_contributions(items[:1], os.path.join(figdir, f"ring_contributions_{ref_name}.png"))
    # rule 2 for the reference: the training molecules that weigh most
    tc = model.training_contributions(mf_ref)
    top = np.argsort(-np.abs(tc))[:8]
    rule2 = [{"name": model.train.names[j], "contribution": float(tc[j]),
              "gap": float(model.y[j])} for j in top]
    cube_path = os.path.join(figdir, f"cofield_gap_{ref_name}.cube")
    html_path = os.path.join(figdir, f"cofield_gap_{ref_name}.html")
    html_ok = export_isosurface_html(model, mf_ref, ds.mols[ref], html_path, cube_path)
    narrative = write_narrative(items[0], extremes, rule2, meta, os.path.join(RESULTS, f"narrative_{ref_name}.md"))
    out = {
        "groups": groups, "split": split_name, "reference": ref_name,
        "reference_rings": meta["n_rings"][ref], "reference_gap": float(y[ref]),
        "refit_check": check, "figure": os.path.relpath(fig_path, ROOT),
        "ring_figure": os.path.relpath(os.path.join(figdir, "ring_contributions_gap.png"), ROOT),
        "html": os.path.relpath(html_path, ROOT) if html_ok else "",
        "cube": os.path.relpath(cube_path, ROOT), "extremes": extremes, "rule2": rule2,
        "molecules": [{k: v for k, v in it.items() if k not in ("mf", "rings")} for it in items],
        "narrative": narrative, "bias": float(model.bias),
        "alphas": {s.name: s.alpha for s in model.fieldset.specs},
        "h": {s.name: s.h for s in model.fieldset.specs}, "lam": model.params["lam"],
        "r2_fit": model.stats.r2, "q2": model.stats.q2,
    }
    jdump(out, path)
    return out


def write_narrative(item: dict, extremes: dict, rule2: Sequence[dict], meta: dict, path: str) -> str:
    """A short text on where the gap of the reference molecule comes from."""
    lines = [f"# Where the predicted gap of {item['name']} comes from", ""]
    lines.append(f"Predicted gap {item['yp']:.2f} eV (computed {item['y']:.2f} eV) = bias "
                 f"{item['bias']:.2f} eV + the sum of the ring contributions {item['per_ring'].sum():+.2f} eV.")
    lines.append("")
    lines.append("Per-ring contributions (decomposition rule 3 summed over the atoms of every ring; "
                 "a positive value raises the gap):")
    lines.append("")
    lines.append("| ring | class | Clar share | centroid x, y (A) | contribution (eV) |")
    lines.append("|---|---|---|---|---|")
    order = np.argsort(-np.asarray(item["per_ring"]))
    for k in order:
        cx, cy, _ = item["centroids"][k]
        lines.append(f"| {k + 1} | {item['classes'][k]} | {item['sextet'][k]:.2f} | {cx:+.1f}, {cy:+.1f} | "
                     f"{item['per_ring'][k]:+.3f} |")
    pos = [k for k in order if item["per_ring"][k] > 0]
    neg = [k for k in order if item["per_ring"][k] < 0]
    lines.append("")
    if pos:
        lines.append("Rings that raise the gap: " + ", ".join(f"{k + 1} ({item['classes'][k]})" for k in pos) + ".")
    if neg:
        lines.append("Rings that lower the gap: " + ", ".join(f"{k + 1} ({item['classes'][k]})" for k in neg) + ".")
    lines.append("")
    lines.append("Co-field extremes in the molecular plane (z = 0) and 1 A above it, by field group "
                 "(where a positive field weight raises the gap most, and lowers it most):")
    lines.append("")
    lines.append("| field group | plane | most positive at x, y (A) | value | most negative at x, y (A) | value |")
    lines.append("|---|---|---|---|---|---|")
    for key, e in extremes.items():
        g, z = key.split("|")
        label = "all fields" if g == "all" else GROUP_LABELS[g]
        lines.append(f"| {label} | {z} | {e['max_xy'][0]:+.1f}, {e['max_xy'][1]:+.1f} | {e['max']:+.3g} | "
                     f"{e['min_xy'][0]:+.1f}, {e['min_xy'][1]:+.1f} | {e['min']:+.3g} |")
    lines.append("")
    lines.append("Training molecules that weigh most in this prediction (decomposition rule 2, a_j K(X, X_j)):")
    lines.append("")
    lines.append("| training molecule | gap (eV) | contribution (eV) |")
    lines.append("|---|---|---|")
    for r in rule2:
        lines.append(f"| {r['name']} | {r['gap']:.2f} | {r['contribution']:+.3f} |")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


# --------------------------------------------------------------------------- #
# Stage: scaling (Nystroem on COMPAS-1x)
# --------------------------------------------------------------------------- #

RESOLUTION_RUNS = (("ABCDE", "random0"), ("ABCDE", "random1"), ("ABCDE", "random2"),
                   ("ABCD", "random0"))


def resolution(set_key: str = "1D", runs: Sequence[Tuple[str, str]] = RESOLUTION_RUNS,
               engine: Optional[str] = "auto", force: bool = False) -> dict:
    """How much of the gap to the published graph networks is resolution.

    The same protocol as the field stage with the attenuation grid extended below, between and
    above its points (:data:`ALPHAS_EXT`), the ridge grid extended down to 1e-6
    (:data:`LAMS_FINE`), and a third configuration with the Gaussian kernel layer on top of the
    per-group kernel; on the random splits of the main field sets.  The extra kernels are built
    on demand.
    """
    check_krr_equivalence()
    groups_needed = "".join(sorted(set("".join(g for g, _s in runs))))
    kernels(set_key, groups=groups_needed, engine=engine, alphas=ALPHAS_EXT)
    ds, meta = load_prepared(set_key)
    store = KernelStore(set_key, max_loaded=6)
    splits = jload(results_path(set_key, "splits"))
    path = results_path(set_key, "resolution")
    done = jload(path) if os.path.isfile(path) and not force else {}
    y = np.asarray(meta["gap"], float)
    for groups, sname in runs:
        key = f"{TARGET}|{groups}|{sname}"
        if key in done or sname not in splits:
            continue
        split = dict(splits[sname], name=sname)
        res = evaluate_set(store, groups, split, y, alphas=ALPHAS_EXT, lams=LAMS_FINE,
                           layer="gauss")
        res["y_test"] = y[np.asarray(split["test"], int)]
        done[key] = res
        jdump(done, path)
        for label, c in res["configs"].items():
            log(f"{key} {label}: alphas {c['alphas']} gamma {c['gamma']} lam {c['lam']:g} "
                f"q2 {c['q2']:.3f} test r2 {c['test']['r2']:.3f} rmse {c['test']['rmse']:.3f} "
                f"mae {c['test']['mae']:.3f}")
        log(f"{key}: chosen {res['chosen']} ({res['seconds']:.0f} s)")
    return done


def report_resolution(set_key: str = "1D") -> List[str]:
    """The lines of the resolution section of table.md."""
    path = results_path(set_key, "resolution")
    if not os.path.isfile(path):
        return []
    done = jload(path)
    coarse = load_fields(set_key)
    lines = ["### Resolution: a wider attenuation grid, a smaller ridge and the Gaussian kernel layer",
             "",
             f"Attenuation grid {ALPHAS_EXT}, ridge grid down to {min(LAMS_FINE):g}, Gaussian layer widths "
             f"{GAMMAS} (relative to the median squared distance); random splits, mean +- sd over the seeds.",
             "",
             "| field set | configuration | alpha per group (A^-2), gamma | lambda | q2 | test R2 | "
             "test RMSE (eV) | test MAE (eV) |",
             "|---|---|---|---|---|---|---|---|"]
    for groups in ("ABCDE", "ABCD"):
        keys = [k for k in done if k.startswith(f"{TARGET}|{groups}|random")]
        if not keys:
            continue
        ckeys = [k for k in coarse if k.startswith(f"{TARGET}|{groups}|random")]
        if ckeys:
            cs = [coarse[k]["configs"][coarse[k]["chosen"]] for k in ckeys]
            lines.append(f"| {{{','.join(groups)}}} | coarse grid, as in the main table | - | - | "
                         f"{_mean_std([c['q2'] for c in cs])} | {_mean_std([c['test']['r2'] for c in cs])} | "
                         f"{_mean_std([c['test']['rmse'] for c in cs])} | {_mean_std([c['test']['mae'] for c in cs])} |")
        for label in ("shared", "per-group", "gauss"):
            cs = [done[k]["configs"][label] for k in keys if label in done[k]["configs"]]
            if not cs:
                continue
            first = cs[0]
            a = ", ".join(f"{g}: {v:g}" for g, v in first["alphas"].items())
            if first.get("gamma") is not None:
                a += f"; gamma {first['gamma']:g}"
            lams_ = sorted(set(f"{c['lam']:g}" for c in cs))
            lines.append(f"| {{{','.join(groups)}}} | {label} (first seed's choice shown) | {a} | "
                         f"{'/'.join(lams_)} | {_mean_std([c['q2'] for c in cs])} | "
                         f"{_mean_std([c['test']['r2'] for c in cs])} | "
                         f"{_mean_std([c['test']['rmse'] for c in cs])} | "
                         f"{_mean_std([c['test']['mae'] for c in cs])} |")
    lines.append("")
    return lines


def scaling(set_key: str = "1x", groups: str = MAIN_SET, limit: Optional[int] = None,
            ms: Sequence[int] = NYSTROEM_M, charges: str = "xtb", jobs: int = 8,
            engine: Optional[str] = "auto", force: bool = False, seed: int = 0) -> dict:
    """Kernel ridge regression in the space of m landmark molecules (the Nystroem path of
    ``cmfqsar.lowrank``, as ``CMFModel.fit_low_rank`` uses it): accuracy and wall time against
    m, on a random 80/10/10 split of the set.  The compiled kernel engine is switched on for the
    whole process when Numba is installed, because the feature map evaluates the kernel of every
    molecule against every landmark."""
    from cmfqsar import fastkernel
    from cmfqsar import kernels as KN
    from cmfqsar import lowrank as LR
    from cmfqsar.kernels import self_kernels

    if engine != "numpy" and fastkernel.available():
        KN.ENGINE = "numba"
    log(f"kernel engine for the scaling stage: {KN.ENGINE}")
    tag = f"scaling_{set_key}" + (f"_n{limit}" if limit else "")
    path = os.path.join(RESULTS, tag + ".json")
    if os.path.isfile(path) and not force:
        return jload(path)
    meta = prepare(set_key, charges=charges, jobs=jobs, limit=limit, force=False) \
        if not os.path.isfile(results_path(set_key, "prepared")) else jload(results_path(set_key, "prepared"))
    if limit and meta["n"] > limit:
        raise SystemExit("the prepared set is larger than --limit; prepare a separate set first")
    ds, meta = load_prepared(set_key)
    y = np.asarray(meta["gap"], float)
    # the attenuations and the ridge of the main model on COMPAS-1D, when it exists
    fields_1d = load_fields("1D")
    if f"{TARGET}|{groups}|random0" in fields_1d:
        cfg = chosen_config(fields_1d, groups, "random0")
        abg = {g: float(a) for g, a in cfg["alphas"].items()}
        lam = float(cfg["lam"])
    else:
        abg = {g: 0.5 for g in groups}
        lam = 0.01
    fs = make_fieldset(groups, meta, alpha_by_group=abg)
    log(f"building fields of {len(ds)} molecules of COMPAS-{set_key}")
    table = build_table(ds, fs, ring_cache={})
    split = pbh.random_split(len(ds), seed)
    tr, te = split["train"], split["test"]
    train, test = table.subset(tr), table.subset(te)
    out = {"set": set_key, "n": len(ds), "groups": groups, "alphas": abg, "lam": lam,
           "n_train": int(len(tr)), "n_test": int(len(te)), "rows": []}
    # the field scales of the exact model: unit mean diagonal over the training part
    specs = [s for s in fs.specs if s.enabled]
    d = np.asarray(self_kernels(train, specs), float)
    scale = np.array([1.0 / v if v > 1e-300 else 0.0 for v in d.mean(axis=1)])
    folds = _kfold_indices(len(tr), FOLDS, seed)
    for m in ms:
        if m >= len(tr):
            continue
        t0 = time.time()
        landmarks, _rows = LR.landmark_molecules(train, m, seed=seed)
        low = LR.build(landmarks, fs, scale=scale, normalize=False, kind="landmarks")
        t_build = time.time() - t0
        try:
            # the features of the training part once (fit_low_rank would build them three
            # times: for the Gram, for the cross-validation and for the fitted values)
            Phi = low.transform(train, block=256)
        except MemoryError:
            log(f"Nystroem m={m}: not run, the feature matrix does not fit in memory")
            out["rows"].append({"m": int(m), "rank": int(low.rank), "skipped": "memory"})
            jdump(out, path)
            continue
        t_features = time.time() - t0 - t_build
        y_tr = y[tr]
        PtP = Phi.T @ Phi
        g = LR.Gram(PtP=PtP, Pty=Phi.T @ y_tr, Pt1=Phi.sum(axis=0), n=len(y_tr),
                    sum_y=float(y_tr.sum()), yty=float(y_tr @ y_tr))
        w, b = LR.fit(g, lam, fit_bias=True)
        t_fit = time.time() - t0
        pred_cv = np.zeros(len(y_tr))
        for held in folds:
            Pf = Phi[held]
            keep = np.setdiff1d(np.arange(len(y_tr)), held)
            gf = LR.Gram(PtP=PtP - Pf.T @ Pf, Pty=Phi.T @ y_tr - Pf.T @ y_tr[held],
                         Pt1=Phi.sum(axis=0) - Pf.sum(axis=0), n=len(keep),
                         sum_y=float(y_tr[keep].sum()), yty=float(y_tr[keep] @ y_tr[keep]))
            wf, bf = LR.fit(gf, lam, fit_bias=True)
            pred_cv[held] = LR.predict(Pf, wf, bf)
            del Pf, gf
        pt = LR.predict(low.transform(test, block=256), w, b)
        row = {"m": int(m), "rank": int(low.rank), "seconds_build": t_build,
               "seconds_features": t_features, "seconds_fit": t_fit,
               "seconds_total": time.time() - t0, "q2": r2_score(y_tr, pred_cv),
               "rmse_cv": rmse(y_tr, pred_cv), "test": metrics(y[te], pt)}
        out["rows"].append(row)
        log(f"Nystroem m={m}: rank {low.rank}, q2 {row['q2']:.3f} test r2 {row['test']['r2']:.3f} "
            f"rmse {row['test']['rmse']:.3f} (features {t_features:.0f} s, fit {t_fit:.0f} s)")
        jdump(out, path)
        del Phi, PtP
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _mean_std(values: Sequence[float]) -> str:
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    return f"{v.mean():.3f} +- {v.std(ddof=1):.3f}"


def report(path: str, set_key: str = "1D") -> str:
    """Rebuild table.md from the JSON results."""
    lines: List[str] = []
    prep_path = results_path(set_key, "prepared")
    if not os.path.isfile(prep_path):
        raise SystemExit("nothing to report: run the stages first")
    meta = jload(prep_path)
    lines.append(f"## COMPAS-{set_key} ({meta['level']}): the HOMO-LUMO gap, {meta['n']} molecules")
    lines.append("")
    lines.append(f"Charges: {meta['charge_source']}.  Target: {TARGET} (eV).  "
                 f"Random splits 80/10/10, {len(SEEDS)} seeds (mean +- standard deviation); "
                 f"size-extrapolation splits train on fewer than N rings and test on N rings.")
    lines.append("")
    done = load_fields(set_key)
    if done:
        splits = jload(results_path(set_key, "splits"))
        size_names = [s for s in splits if splits[s]["kind"] == "size"]
        lines.append("### Field sets")
        lines.append("")
        head = "| field set | fields | q2 (5-fold CV, train) | test R2 | test RMSE (eV) | test MAE (eV) |"
        for s in size_names:
            n_tr, n_te = len(splits[s]["train"]), len(splits[s]["test"])
            head += f" N={splits[s]['test_rings']} R2 (train {n_tr}, test {n_te}) | N={splits[s]['test_rings']} RMSE |"
        lines.append(head)
        lines.append("|" + "---|" * (head.count("|") - 1))
        for groups in FIELD_SETS:
            keys = [k for k in done if k.startswith(f"{TARGET}|{groups}|random")]
            if not keys:
                continue
            chosen = [done[k]["configs"][done[k]["chosen"]] for k in keys]
            row = (f"| {{{','.join(groups)}}} | {', '.join(done[keys[0]]['fields'])} | "
                   f"{_mean_std([c['q2'] for c in chosen])} | {_mean_std([c['test']['r2'] for c in chosen])} | "
                   f"{_mean_std([c['test']['rmse'] for c in chosen])} | {_mean_std([c['test']['mae'] for c in chosen])} |")
            for s in size_names:
                k = f"{TARGET}|{groups}|{s}"
                if k in done:
                    c = done[k]["configs"][done[k]["chosen"]]
                    row += f" {c['test']['r2']:.3f} | {c['test']['rmse']:.3f} |"
                else:
                    row += " - | - |"
            lines.append(row)
        lines.append("")
        # the smallest molecules: a random test part that holds benzene or naphthalene asks the
        # model to extrapolate to a size it has one or no example of, and one such molecule can
        # set the RMSE of a seed; the table above keeps them, this line says what they cost
        n_rings = np.asarray(meta["n_rings"], int)
        small = []
        for groups in (MAIN_SET, "ABCDE"):
            keys = [k for k in done if k.startswith(f"{TARGET}|{groups}|random")]
            if not keys:
                continue
            vals = []
            for k in keys:
                r = done[k]
                c = r["configs"][r["chosen"]]
                te = np.asarray(splits[r["split"]]["test"], int)
                keep = n_rings[te] >= 3
                vals.append(rmse(np.asarray(r["y_test"])[keep], np.asarray(c["y_test_pred"])[keep]))
            small.append(f"{{{','.join(groups)}}} {_mean_std(vals)}")
        held = [(s, [meta["names"][i] for i in splits[s]["test"] if n_rings[i] < 3])
                for s in splits if splits[s]["kind"] == "random"]
        held = [f"{s}: {', '.join(n)}" for s, n in held if n]
        lines.append("Test RMSE (eV) over the molecules of three rings and more, random splits: "
                     + "; ".join(small) + ".  The one- and two-ring molecules in a random test part "
                     "(" + ("; ".join(held) if held else "none") + ") are predicted by extrapolation "
                     "from a training part that holds at most one molecule of their size.")
        lines.append("")
        # the chosen hyperparameters of the main set
        lines.append(f"### Attenuations and ridge chosen for {{{','.join(MAIN_SET)}}}")
        lines.append("")
        lines.append("| split | choice | alpha per group (A^-2) | lambda | q2 | valid R2 | test R2 |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in splits:
            k = f"{TARGET}|{MAIN_SET}|{s}"
            if k not in done:
                continue
            r = done[k]
            c = r["configs"][r["chosen"]]
            a = ", ".join(f"{g}: {v:g}" for g, v in c["alphas"].items())
            v = c["valid"]["r2"]
            lines.append(f"| {s} | {r['chosen']} | {a} | {c['lam']:g} | {c['q2']:.3f} | "
                         f"{'-' if not np.isfinite(v) else f'{v:.3f}'} | {c['test']['r2']:.3f} |")
        lines.append("")
        other = [t for t in ("HOMO_eV", "LUMO_eV") if any(k.startswith(t + "|") for k in done)]
        if other:
            lines.append("### HOMO and LUMO energies with the same fields")
            lines.append("")
            lines.append("| target | field set | q2 | test R2 | test RMSE (eV) | test MAE (eV) |")
            lines.append("|---|---|---|---|---|---|")
            for t in other:
                for groups in FIELD_SETS:
                    keys = [k for k in done if k.startswith(f"{t}|{groups}|random")]
                    if not keys:
                        continue
                    chosen = [done[k]["configs"][done[k]["chosen"]] for k in keys]
                    lines.append(f"| {t} | {{{','.join(groups)}}} | {_mean_std([c['q2'] for c in chosen])} | "
                                 f"{_mean_std([c['test']['r2'] for c in chosen])} | "
                                 f"{_mean_std([c['test']['rmse'] for c in chosen])} | "
                                 f"{_mean_std([c['test']['mae'] for c in chosen])} |")
            lines.append("")
    bpath = results_path(set_key, "baselines")
    if os.path.isfile(bpath):
        b = jload(bpath)
        splits = jload(results_path(set_key, "splits"))
        size_names = [s for s in splits if splits[s]["kind"] == "size"]
        lines.append("### Baselines on the same splits")
        lines.append("")
        head = "| model | test R2 | test RMSE (eV) | test MAE (eV) |"
        for s in size_names:
            head += f" N={splits[s]['test_rings']} R2 | N={splits[s]['test_rings']} RMSE |"
        lines.append(head)
        lines.append("|" + "---|" * (head.count("|") - 1))
        labels = {"count_ridge": "ridge on atom-class and ring-class counts (additivity scheme)",
                  "rings_line": "straight line in the number of rings",
                  "rf_descriptors": f"random forest on {b['n_descriptors']} RDKit descriptors"}
        for key, label in labels.items():
            rand = [b["splits"][s][key]["test"] for s in b["splits"] if s.startswith("random")]
            row = (f"| {label} | {_mean_std([m['r2'] for m in rand])} | {_mean_std([m['rmse'] for m in rand])} | "
                   f"{_mean_std([m['mae'] for m in rand])} |")
            for s in size_names:
                if s in b["splits"]:
                    m = b["splits"][s][key]["test"]
                    row += f" {m['r2']:.3f} | {m['rmse']:.3f} |"
                else:
                    row += " - | - |"
            lines.append(row)
        lines.append("")
        lines.append("### Published models of the gap on the same molecules (random splits; no "
                     "size-extrapolation numbers are published)")
        lines.append("")
        lines.append("| model | source | protocol | test MAE (eV) | test RMSE (eV) | test R2 |")
        lines.append("|---|---|---|---|---|---|")
        lines.append("| SE(3)-transformer on a graph of rings (SE(3)-GOR) | Weiss, Wahab, Bronstein, "
                     "Gershoni-Poranne, J. Org. Chem. 2023, 88, 9645, Table 1 | COMPAS-1D, 8,678 molecules, "
                     "random 81/9/10, seed 42 | 0.012 +- 0.002 | - | - |")
        lines.append("| SE(3)-transformer on a graph of atoms (SE(3)-GOA) | same | same | 0.011 +- 0.004 | - | - |")
        lines.append("| SphereNet / DimeNet++ / SchNet (geometric graph networks) | same | same | "
                     "0.013 / 0.015 / 0.022 | - | - |")
        lines.append("| CHEM-BERT (SMILES language model) / AttentiveFP (topological graph) | same | same | "
                     "0.033 / 0.073 | - | - |")
        lines.append("| RNN on one-hot augmented LALA annulation strings | Fite, Wahab, Paenurk, Gross, "
                     "Gershoni-Poranne, J. Phys. Org. Chem. 2023, 36, e4458, Fig. 3 | COMPAS-1D, 7,674 training "
                     "molecules (90 %) | 0.05 (all model/representation pairs: 0.04 to 0.26) | - | - |")
        lines.append("| extra-trees / random forest on selected RDKit descriptors | Zheng, Zhu, Lu, Yan, Jiang, "
                     "Sun, Chem. Phys. Lett. 2023, 814, 140358 | COMPAS-1D without benzene and naphthalene "
                     "(8,676), random 90/10, 100 repeats | - | about 0.12 (MSE about 0.014) | about 0.91 |")
        lines.append("")
        lines.append("The random forest of this study (0.074 eV MAE, 0.117 eV RMSE, R2 0.91) reproduces the "
                     "last row.  The field models of this study (MAE 0.13 eV for {A,B,C,D}, 0.11 eV with the "
                     "sextet field) sit at the lower end of the annulation-string models and an order of "
                     "magnitude above the geometric graph networks on these random splits.")
        lines.append("")
    rows = []
    for name in sorted(os.listdir(RESULTS)) if os.path.isdir(RESULTS) else []:
        if name.startswith("scaling_") and name.endswith(".json"):
            rows.append(jload(os.path.join(RESULTS, name)))
    lines.extend(report_resolution(set_key))
    lines.append("### Nystroem approximation (landmark molecules)")
    lines.append("")
    if rows:
        lines.append("| set | molecules | m landmarks | rank | q2 | test R2 | test RMSE (eV) | fit time (s) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for sc in rows:
            for r in sc["rows"]:
                lines.append(f"| COMPAS-{sc['set']} | {sc['n']} | {r['m']} | {r['rank']} | {r['q2']:.3f} | "
                             f"{r['test']['r2']:.3f} | {r['test']['rmse']:.3f} | {r['seconds_fit']:.0f} |")
    else:
        lines.append("Nystroem on COMPAS-1x: pending (second milestone).")
    lines.append("")
    ipath = results_path(set_key, "interpret")
    if os.path.isfile(ipath):
        it = jload(ipath)
        lines.append("### Interpretation")
        lines.append("")
        fig = str(it["figure"]).replace("\\", "/")
        ring_fig = str(it["ring_figure"]).replace("\\", "/")
        rt = it["refit_check"]["test"]
        lines.append(f"Reference molecule {it['reference']} ({it['reference_rings']} rings, gap "
                     f"{it['reference_gap']:.2f} eV); co-field figure `{fig}`; per-ring "
                     f"contributions `{ring_fig}`.  The CMFModel refitted in float64 for the figures "
                     f"reproduces the table's test predictions (float32 kernels) to "
                     f"{it['refit_check']['max_dev']:.1e} eV at worst; its own test R2 is {rt['r2']:.3f}, "
                     f"RMSE {rt['rmse']:.3f} eV.")
        lines.append("")
    text = "\n".join(lines) + "\n"
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def write_notes(set_key: str = "1D") -> str:
    """NOTES.md: the files, their checksums, every substitution, and the decisions taken."""
    meta = jload(results_path(set_key, "prepared"))
    lines = ["# Notes of the COMPAS-1 pilot", ""]
    lines.append(f"Written by benchmarks/run_compas.py on {_dt.date.today().isoformat()}.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append("| file | URL | bytes | SHA-256 | downloaded |")
    lines.append("|---|---|---|---|---|")
    rec_path = os.path.join(CACHE, "checksums.json")
    rec = jload(rec_path) if os.path.isfile(rec_path) else meta["files"]
    for name, r in rec.items():
        lines.append(f"| {name} | {r['url']} | {r['bytes']} | {r['sha256']} | {r['downloaded']} |")
    lines.append("")
    lines.append(f"COMPAS-{set_key}: {meta['n']} molecules at {meta['level']}; the GitLab repository "
                 "porannegroup/compas, branch main (the raw-file endpoint carries no commit hash; the "
                 "checksums above identify the files).  Licence CC-BY-NC-SA: the files stay in "
                 "benchmarks/cache and are not committed.")
    lines.append("")
    lines.append("## Substitutions and deviations from the task text")
    lines.append("")
    subs = list(meta["substitutions"])
    subs += [
        "No separate package or YAML configuration: the study is this benchmark script with the "
        "library module cmfqsar/pbh.py, as every other study of the repository.",
        "The steric field is the program's (Tripos well depths, widths scaled by the van der Waals "
        "radius, hydrogens included), not a unit-weight heavy-atom field.",
        "The field scales (unit mean kernel diagonal) are taken over the whole set rather than the "
        "training part, so that one stored kernel per attenuation serves every split; the scale is "
        "a constant per field and the difference is a rescaling of the mixing coefficients.",
        "The attenuations are chosen on a random subset of 2,500 training molecules (5-fold CV); the "
        "ridge is then chosen by 5-fold CV on the whole training part.",
        "The aromaticity field E is the Clar-sextet share of every ring (placeholder); a "
        "NICS-weighted ring field is a follow-up that needs a DFT calculation per ring.",
        "Literature numbers for the gap on COMPAS-1 are not filled in (table row 'TBD').",
        "The Nystroem scaling experiment on COMPAS-1x (--stage scaling) uses the attenuations and "
        "the ridge of the main COMPAS-1D model (no search on the 34k set), one random 80/10/10 split, "
        "and the landmark path of cmfqsar.lowrank with the compiled kernel engine switched on for the "
        "process; the feature map of the training part is built once and the cross-validation reuses "
        "it, which CMFModel.fit_low_rank does not do.",
    ]
    for s in subs:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    lines.append("- Frame: heavy-atom inertia tensor with atomic masses; x along the smallest moment, z "
                 "along the largest; the signs of x and y by the moments sum x^3, sum y^3, sum x^2 y, "
                 "sum x^3 y, sum x y^3 in that order (cmfqsar.pbh.resolve_signs); molecules with equal "
                 "in-plane moments take x along the largest third (else fourth) moment.")
    lines.append("- The kernel-based choice among the four frames (alignment by prediction confidence) "
                 "is implemented (cmfqsar.pbh.choose_frame_by_kernel) and off by default; the rule "
                 "leaves nothing to choose for a planar molecule without symmetry.")
    lines.append("- Ring classes from the fused-neighbour graph of the six-membered carbon rings; "
                 "checked against the 'lalas' annulation strings of the table "
                 f"({len(meta['annulation_mismatches'])} disagreements).")
    lines.append("- Kernel ridge regression with the bias of Eq. 12 (2013); inside the cross-validation "
                 "the same solution is obtained from the eigendecomposition of the centred kernel "
                 "(checked against krr_fit at every run).")
    lines.append(f"- Attenuation grid {ALPHAS} A^-2 in the program's convention (exponent alpha/2); "
                 f"ridge grid {LAMS}.")
    lines.append("- GFN2-xTB charges with xtb " + (meta.get("xtb_version") or "(not run)") +
                 "; the input geometry is kept (no optimisation), no Fukui run.")
    if meta["failures"]:
        lines.append(f"- xtb failed on {len(meta['failures'])} molecules, which were dropped: "
                     + ", ".join(f["name"] for f in meta["failures"][:10]))
    lines.append("")
    lines.append("## Open questions")
    lines.append("")
    lines.append("- Whether the gap of the few non-planar molecules (helicene-like fjords) is served by "
                 "the planar frame; the frame is defined for them but the two faces differ.")
    lines.append("- The size-extrapolation split trains on few molecules (the sets are dominated by "
                 "ten-ring molecules), so its numbers mix extrapolation with a small training set.")
    lines.append("- Published models on the same molecules (table.md, last table): the geometric graph "
                 "networks of Weiss et al. 2023 reach a test MAE of 0.011 to 0.022 eV on a random 81/9/10 "
                 "split (their code: gitlab.com/porannegroup/PBHs-interpretability), the sequence model on "
                 "augmented annulation strings of Fite et al. 2023 0.05 eV (gitlab.com/porannegroup/lalas), "
                 "and the descriptor forests of Zheng et al. 2023 R2 0.91. The field models here (MAE 0.11 "
                 "to 0.13 eV) are an order of magnitude behind the graph networks on interpolation among "
                 "isomers. The gap is set by the ring arrangement to a precision the overlap of wide "
                 "Gaussians does not resolve: the chosen widths are the widest of the grid and the ridge "
                 "the smallest, so the model asks for more resolution than it was given. Nothing published "
                 "measures the size-extrapolation split, where the ring fields are positive and the "
                 "descriptor forest is not.")
    lines.append("- The resolution stage (table.md) answers where the resolution is: not in the width. "
                 "The attenuation grid extended to 0.02-20 A^-2 and the ridge down to 1e-6 leave the linear "
                 "{A,B,C,D,E} model at 0.11 eV MAE; the Gaussian kernel layer on the same kernel halves it "
                 "to 0.056 +- 0.004 eV (R2 0.964 +- 0.005, three seeds), level with the published "
                 "annulation-string network and ahead of the descriptor forests, three to five times "
                 "behind the graph networks. The layered model is no longer linear in the field: the "
                 "co-field and rule 3 belong to the linear model, and Shapley values are the reading of "
                 "the layered one. Its width came out at gamma = 0.25-0.5 of the median squared distance, "
                 "the widest of the grid, and a wider grid of gammas is one more run.")
    lines.append("- The chosen hyperparameters sit at the edges of the grids: the steric attenuation at "
                 "the top (its width is scaled by the van der Waals radius, so the same grid value is a "
                 "wider Gaussian) and the ridge at the bottom (1e-4) for the sets with the ring fields. "
                 "A wider steric grid and a smaller ridge are worth one more run; the float32 kernels "
                 "limit the ridge to about 1e-5.")
    lines.append("- Seed 0 of the random splits puts benzene, the only one-ring molecule, in the test "
                 "part; the main model extrapolates it to 10-13 eV (computed 6.72), which is the whole "
                 "difference between that seed and the other four (table.md gives the RMSE over the "
                 "molecules of three rings and more).")
    lines.append("- The Clar-sextet ring field (E), meant as a placeholder, is the single largest gain "
                 "on top of {A,B,C,D}; the NICS-weighted field should be compared against it, not "
                 "against the set without it.")
    lines.append("- On COMPAS-1x the landmark approximation converges at 500 landmark molecules: 1,000 "
                 "landmarks give the same q2 and test R2 to three decimals and the rank of the feature "
                 "map grows only from 2,244 to 2,570 (9 fields x 1,000 landmarks would allow 9,000). "
                 "The fields of these planar molecules in one frame span a few thousand directions, "
                 "which is also why the exact 8,678 x 8,678 model of COMPAS-1D is not needed for them.")
    lines.append("- The per-ring contributions (rule 3) are large and cancel: a ring on the +x side of "
                 "the frame counts positive and one on the -x side negative, whatever its class, "
                 "because the co-field has a smooth component across the frame and a ridge of 1e-4 "
                 "lets the coefficients grow. Their sum is the prediction; the individual values read "
                 "position as much as chemistry. A decomposition against a reference (the Shapley "
                 "values of cmfqsar/shapley.py) is the next thing to try for this series.")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS, "NOTES.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

STAGES = ("prepare", "kernels", "fields", "baselines", "interpret", "scaling", "resolution", "report")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=STAGES + ("all",), default="all")
    ap.add_argument("--set", default="1D", choices=("1D", "1x"))
    ap.add_argument("--charges", default="xtb", choices=("xtb", "gasteiger"))
    ap.add_argument("--jobs", type=int, default=8, help="xtb runs at a time")
    ap.add_argument("--limit", type=int, default=None, help="first molecules only (smoke tests)")
    ap.add_argument("--sets", default=",".join(FIELD_SETS), help="field sets, comma separated")
    ap.add_argument("--splits", default="", help="split names, comma separated (default: all)")
    ap.add_argument("--targets", default=TARGET, help="targets, comma separated")
    ap.add_argument("--engine", default="auto", help="kernel engine: auto, numba or numpy")
    ap.add_argument("--landmarks", default=",".join(str(m) for m in NYSTROEM_M))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="only rebuild table.md and NOTES.md")
    args = ap.parse_args(argv)
    ensure_dir(RESULTS)
    engine = None if args.engine == "numpy" else args.engine
    if args.report:
        print(report(os.path.join(RESULTS, "table.md"), args.set))
        write_notes(args.set)
        return
    stages = STAGES[:-1] if args.stage == "all" else (args.stage,)
    if args.stage == "all":
        stages = ("prepare", "kernels", "fields", "baselines", "interpret")
    for stage in stages:
        log(f"== stage {stage}")
        if stage == "prepare":
            prepare(args.set, charges=args.charges, jobs=args.jobs, limit=args.limit, force=args.force)
        elif stage == "kernels":
            kernels(args.set, engine=engine, force=args.force)
        elif stage == "fields":
            fields(args.set, sets=[s for s in args.sets.split(",") if s],
                   splits_wanted=[s for s in args.splits.split(",") if s] or None,
                   targets=[t for t in args.targets.split(",") if t], force=args.force)
        elif stage == "baselines":
            baselines(args.set, force=args.force)
        elif stage == "interpret":
            interpret(args.set, engine=engine, force=args.force)
        elif stage == "scaling":
            scaling(args.set if args.set != "1D" else "1x", limit=args.limit,
                    ms=[int(m) for m in args.landmarks.split(",")], charges=args.charges,
                    jobs=args.jobs, engine=engine, force=args.force)
        elif stage == "resolution":
            resolution(args.set, engine=engine, force=args.force)
        elif stage == "report":
            pass
    print(report(os.path.join(RESULTS, "table.md"), "1D" if os.path.isfile(results_path("1D", "prepared")) else args.set))
    if os.path.isfile(results_path("1D", "prepared")):
        write_notes("1D")


if __name__ == "__main__":
    main()
