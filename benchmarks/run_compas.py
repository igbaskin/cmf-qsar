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
from cmfqsar.dataio import load_dataset, relative_path  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, FieldSpec, MoleculeFields  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.model import CMFModel, _kfold_indices, krr_fit, mae, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

CACHE = os.path.join(HERE, "cache", "compas")
RESULTS = os.path.join(HERE, "results", "compas")
BASE_URL = "https://gitlab.com/porannegroup/compas/-/raw/main/COMPAS-1/"
#: the raw-file folders of the three COMPAS sets used here, by family (the first character of a set key)
BASE_URLS = {"1": BASE_URL, "2": BASE_URL.replace("COMPAS-1", "COMPAS-2"),
             "3": BASE_URL.replace("COMPAS-1", "COMPAS-3")}
README = {"1": "README.md", "2": "readme.md", "3": "README.md"}
FILES = {
    "1D": ("compas-1D_b3lyp_def2-svp.csv", "compas-1D_b3lyp_def2-svp.tar.gz"),
    "1x": ("compas-1x.csv", "compas-1x.tar.gz"),
    "3D": ("compas-3D.csv", "compas-3D.tar.gz"),
    "3x": ("compas-3x.csv", "compas-3x.tar.gz"),
    "2D": ("compas-2D.csv", "compas-2D.sdf.gz"),
}
LEVELS = {"1D": "B3LYP-D3BJ/def2-SVP", "1x": "GFN2-xTB",
          "3D": "CAM-B3LYP-D3BJ/def2-SVP geometries (xyz headers), energies as in the COMPAS-3 paper",
          "3x": "GFN2-xTB",
          "2D": "CAM-B3LYP-D3BJ/def2-SVP (COMPAS-2 paper, Sci. Data 2024, 11, 97)"}
#: the sets with a peri-condensed ring (COMPAS-3) and with hetero atoms (COMPAS-2)
PERI_FAMILIES = ("2", "3")
HETERO_FAMILIES = ("2",)
#: COMPAS-2D holds 52,000 molecules; the stored-kernel protocol runs on a random subset of this size
SUBSET_2D = 10000

#: attenuation grid, in the convention of the program (field = w exp(-alpha/2 r^2), alpha in A^-2;
#: the prompt's alpha' in exp(-alpha'^2 r^2) is sqrt(alpha/2): 0.22 to 1.6 A^-1 here)
ALPHAS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
LAMS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)
#: the resolution stage: attenuations below, between and above the grid, a finer and smaller
#: ridge, and the widths of the Gaussian kernel layer relative to the median squared distance
ALPHAS_EXT = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 20.0)
LAMS_FINE = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
#: the layer widths, relative to the median squared distance. The first measurement chose the
#: smallest point of the grid, 0.25, on nine of ten splits, and the smallest point is the widest
#: layer: as gamma falls to zero the Gaussian layer tends to the linear kernel, which the
#: `per-group` configuration measures. Three wider layers were therefore measured as well
#: (``--stage resolution --wide-gammas``, stored in resolution_wide_<set>.json and reported
#: beside this grid). They found no better width under the edge: the selection criterion is flat
#: from 0.03 to 0.25 on all ten splits, and on the size-extrapolation splits the wider points
#: make the choice worse, because there the criterion cannot tell the layer from the linear
#: kernel while the held-out larger molecules can. The default grid therefore keeps its edge.
GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
#: the default grid with three wider layers added, for the measurement described above
WIDE_GAMMAS = (0.03, 0.06, 0.125) + GAMMAS
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
    "F": [pbh.HOMA_FIELD],
    "R": [],                           # the ring-type fields of a hetero-atom set, from its table
}
GROUP_LABELS = {"A": "steric", "B": "electrostatic", "C": "atom-class indicator",
                "D": "ring-class indicator", "E": "Clar-sextet ring", "F": "HOMA-weighted ring",
                "R": "ring-type indicator"}
FIELD_SETS = ("A", "AB", "C", "D", "ABC", "ABCD", "CD", "ABCDE", "ABCDF", "ABCDEF")
#: the field sets run on the later sets (three random seeds and the size splits)
FIELD_SETS_BY_FAMILY = {"1": FIELD_SETS, "3": ("D", "ABCD", "ABCDE", "ABCDEF"),
                        "2": ("C", "D", "R", "ABCD", "ABCDR", "ABCDFR")}
MAIN_SET = "ABCD"
#: the gap minus its linear estimate from the GFN2-xTB gap (slope and intercept fitted on the
#: training part of every split): the target of the Delta-learning rows
DELTA = "DELTA_eV"
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
    fam = set_key[0]
    readme_local = "README.md" if fam == "1" else f"README-COMPAS-{fam}.md"
    for name, local in [(n, n) for n in FILES[set_key]] + [(README[fam], readme_local)]:
        path = os.path.join(CACHE, local)
        url = BASE_URLS[fam] + name
        if not os.path.isfile(path):
            log(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        with open(path, "rb") as fh:
            head = fh.read(2)
        if name.endswith(".gz") and head != b"\x1f\x8b":
            raise SystemExit(f"{path} is not a gzip archive (a Git-LFS pointer?): ask the user "
                             f"for the file")
        if local not in record:
            record[local] = {"url": url, "bytes": os.path.getsize(path), "sha256": sha256(path),
                             "downloaded": _dt.date.today().isoformat()}
    jdump(record, record_path)
    return record


def extract(set_key: str) -> str:
    archive = os.path.join(CACHE, FILES[set_key][1])
    if archive.endswith(".sdf.gz"):
        return archive                      # read in place by pbh.read_compas2
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


def subset_names(csv_path: str, size: int, seed: int = 0) -> List[str]:
    """A random subset of the neutral molecules of a COMPAS-2 table, by name."""
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as fh:
        names = [r["name"] for r in _csv.DictReader(fh) if r.get("charge") == "0"]
    rng = np.random.RandomState(seed)
    pick = np.sort(rng.choice(len(names), size=min(size, len(names)), replace=False))
    return [names[i] for i in pick]


def prepare(set_key: str, charges: str = "xtb", jobs: int = 8, limit: Optional[int] = None,
            force: bool = False, subset: Optional[int] = None) -> dict:
    """Download, parse, align, compute the charges, write the SDF, make the splits.

    COMPAS-3D (peri-condensed) and COMPAS-2D (hetero atoms) go through the same steps; the
    hetero-atom set is read from its SDF, on a random subset of `subset` molecules (default
    :data:`SUBSET_2D`), with every ring of four to six atoms and the ring types of its table.
    """
    from cmfqsar import xtb

    meta_path = results_path(set_key, "prepared")
    if os.path.isfile(meta_path) and os.path.isfile(prepared_sdf(set_key)) and not force:
        return jload(meta_path)
    fam = set_key[0]
    carbon_only = fam not in HETERO_FAMILIES
    t0 = time.time()
    record = download(set_key)
    folder = extract(set_key)
    csv_path = os.path.join(CACHE, FILES[set_key][0])
    log(f"reading {csv_path}")
    chosen_names = None
    if fam in HETERO_FAMILIES:
        size = subset if subset is not None else SUBSET_2D
        chosen_names = subset_names(csv_path, size)
        ds = pbh.read_compas2(csv_path, folder, limit=limit, names=chosen_names,
                              progress=lambda k, n: log(f"  {k}/{n} molecules read"))
    else:
        ds = pbh.read_compas(csv_path, folder, limit=limit,
                             progress=lambda k, n: log(f"  {k}/{n} molecules read"))
    log(f"{len(ds)} molecules; frame and rings")
    mismatches, peri, flat = [], [], []
    types_seen: Dict[str, int] = {}
    classes_seen: Dict[str, int] = {}
    ring_types_seen: Dict[str, int] = {}
    has_c3r = False
    for m in ds.mols:
        set_coords(m, pbh.canonical_frame(m))
        rings = pbh.ring_classes(m, carbon_only=carbon_only)
        rc = pbh.ring_class_counts(rings)
        for r in rings:
            classes_seen[r.cls] = classes_seen.get(r.cls, 0) + 1
            ring_types_seen[r.ring_type] = ring_types_seen.get(r.ring_type, 0) + 1
        if any(r.cls == pbh.PERI_CLASS for r in rings):
            peri.append(m.GetProp("_Name"))
        if m.HasProp("lalas"):
            lin, ang = pbh.annulation_counts(m.GetProp("lalas"))
            if len(rings) != int(m.GetProp("n_rings")) or rc["linear"] != lin \
                    or rc["angular"] + rc["branched"] != ang:
                mismatches.append(m.GetProp("_Name"))
        elif len(rings) != int(m.GetProp("n_rings")):
            mismatches.append(m.GetProp("_Name"))
        for a in m.GetAtoms():
            t = pbh.pbh_atom_type(a)
            types_seen[t] = types_seen.get(t, 0) + 1
            if t == "C.3r":
                has_c3r = True
        pos = m.GetConformer().GetPositions()
        flat.append(float(np.abs(pos[pbh.heavy_mask(m), 2]).max()))
    if peri and fam not in PERI_FAMILIES:
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
    if chosen_names is not None:
        substitutions.append(f"COMPAS-{set_key}: a random subset of {len(chosen_names)} of the "
                             f"neutral molecules (seed 0) instead of the whole set, so that the "
                             f"stored-kernel protocol of COMPAS-1D applies unchanged")
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
        "family": fam, "carbon_only": carbon_only, "subset": chosen_names is not None,
        "atom_types": [t for t in pbh.PBH_TYPES + pbh.HETERO_TYPES if types_seen.get(t, 0) > 0],
        "atom_type_counts": types_seen,
        "ring_classes": [c for c in pbh.RING_CLASSES_PERI if classes_seen.get(c, 0) > 0],
        "ring_class_counts": classes_seen,
        "ring_types": [t for t in pbh.RING_TYPES if ring_types_seen.get(t, 0) > 0] if not carbon_only else [],
        "ring_type_counts": ring_types_seen, "n_peri_molecules": len(peri),
        "max_out_of_plane_A": {"median": float(np.median(flat)), "p95": float(np.percentile(flat, 95)),
                               "max": float(np.max(flat))},
    }
    unknown = [t for t in ring_types_seen if t not in pbh.RING_TYPES]
    if unknown and not carbon_only:
        log(f"WARNING: ring types outside the COMPAS-2 list: {unknown}")
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


def xtb_gaps(set_key: str, names: Sequence[str]) -> Optional[np.ndarray]:
    """The GFN2-xTB gap of every molecule of a DFT set from the xTB table of the same family
    (COMPAS-1x for 1D, COMPAS-3x for 3D), or None when that table is not in the cache."""
    import csv as _csv
    xkey = set_key[0] + "x"
    if xkey not in FILES:
        return None
    path = os.path.join(CACHE, FILES[xkey][0])
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        gaps = {r["molecule"]: float(r["GAP_eV"]) for r in _csv.DictReader(fh)}
    if any(n not in gaps for n in names):
        return None
    return np.array([gaps[n] for n in names], float)


def default_groups(meta: dict) -> str:
    """The kernel groups of a set: A to F for the benzenoids, A to D, F and the ring types for
    the hetero-atom sets (the Clar sextets are defined for carbon rings only)."""
    if meta.get("ring_types"):
        return "ABCDFR"
    return "ABCDEF"


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
    if group == "C":
        if meta.get("atom_types"):
            names = ["cif:" + t for t in meta["atom_types"]]
        elif not meta.get("has_C3r", False):
            names = [n for n in names if n != "cif:C.3r"]
    if group == "D" and meta.get("ring_classes"):
        names = ["ring:" + c for c in meta["ring_classes"]]
    if group == "R":
        names = ["ring:" + t for t in meta.get("ring_types", [])]
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


def build_table(ds, fieldset: FieldSet, ring_cache: Optional[dict] = None,
                carbon_only: bool = True) -> AtomTable:
    mfs = []
    want_sextet = any(s.name == pbh.SEXTET_FIELD for s in fieldset.specs)
    if ring_cache is None:
        ring_cache = {}
    for m, name in zip(ds.mols, ds.names):
        rings = None
        if ring_cache is not None:
            rings = ring_cache.get(name)
            if rings is None:
                rings = pbh.ring_classes(m, carbon_only=carbon_only)
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


def kernels(set_key: str, groups: Optional[str] = None, engine: Optional[str] = "auto",
            force: bool = False, alphas: Sequence[float] = ALPHAS) -> dict:
    """The group kernels on the whole set for every attenuation of the grid, stored as float32."""
    ds, meta = load_prepared(set_key)
    groups = groups or default_groups(meta)
    rec_path = results_path(set_key, "kernels")
    rec = jload(rec_path) if os.path.isfile(rec_path) and not force else {}
    fs_all = make_fieldset(groups, meta)
    log(f"building the fields of {len(ds)} molecules")
    table = build_table(ds, fs_all, ring_cache={}, carbon_only=meta.get("carbon_only", True))
    log(f"{table.coords.shape[0]} atoms including {sum(1 for m in table.molecules for s in m.symbols if s == pbh.RING_SYMBOL)} ring centres")
    n_mol = len(ds)
    for g in groups:
        cols = [fs_all.names.index(n) for n in group_fields(g, meta)]
        if not cols:
            continue
        # one table per field, holding only the atoms that carry weight in it: a field kernel is
        # then built alone (one float64 n x n matrix at a time) and summed into the group kernel
        parts = [restrict_table(table, [c]) for c in cols]
        n_atoms = sum(int(p.coords.shape[0]) for p in parts)
        for alpha in alphas:
            key = f"{g}_{alpha:g}"
            path = kernel_path(set_key, g, alpha)
            if os.path.isfile(path) and key in rec and not force:
                continue
            specs = [FieldSpec(name=fs_all.specs[c].name, kind=fs_all.specs[c].kind, alpha=alpha)
                     for c in cols]
            t0 = time.time()
            joint = np.zeros((n_mol, n_mol), dtype=np.float32)
            diag_mean, scale = [], []
            for part, spec in zip(parts, specs):
                Kf = kernel_matrices(part, part, [spec], engine=engine)[0]
                d = float(np.mean(np.diag(Kf)))
                s = 1.0 / d if d > 1e-300 else 0.0
                diag_mean.append(d)
                scale.append(s)
                joint += (s * Kf).astype(np.float32)
                del Kf
            ensure_dir(os.path.dirname(path))
            np.save(path, joint)
            rec[key] = {"group": g, "alpha": alpha, "fields": [s.name for s in specs],
                        "diag_mean": diag_mean, "scale": scale, "atoms": n_atoms,
                        "seconds": time.time() - t0, "engine": engine}
            jdump(rec, rec_path)
            log(f"kernel {key}: {n_atoms} atoms, {time.time() - t0:.0f} s")
    return rec


class KernelStore:
    """The stored group kernels of one set, read into memory on request and summed.

    A kernel is read whole (300 MB in float32 for 8,678 molecules) rather than memory-mapped:
    gathering the rows and columns of a split from a memory map is many small reads and was the
    slowest part of a split.  The last `max_loaded` kernels stay in memory (four: a field
    process then holds about 3 GB, and several of them run at once).
    """

    def __init__(self, set_key: str, max_loaded: int = 4):
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


def delta_target(gap: np.ndarray, gap_xtb: np.ndarray, train: np.ndarray
                 ) -> Tuple[np.ndarray, Tuple[float, float]]:
    """The gap minus its linear estimate from the xTB gap, the line fitted on `train` only."""
    A = np.c_[gap_xtb[train], np.ones(len(train))]
    coef, *_ = np.linalg.lstsq(A, gap[train], rcond=None)
    return gap - (coef[0] * gap_xtb + coef[1]), (float(coef[0]), float(coef[1]))


def fields(set_key: str, sets: Sequence[str] = FIELD_SETS, splits_wanted: Optional[Sequence[str]] = None,
           targets: Sequence[str] = (TARGET,), force: bool = False, layer: Optional[str] = None) -> dict:
    """Every (target, field set, split): the attenuations, the ridge and the test metrics.

    ``layer="gauss"`` adds the Gaussian kernel layer as a third configuration of every run
    (chosen against the linear ones on the validation part; the report shows both).  The target
    :data:`DELTA` is the gap minus its linear xTB estimate, fitted on the training part of each
    split, and needs the xTB table of the family in the cache.
    """
    check_krr_equivalence()
    ds, meta = load_prepared(set_key)
    store = KernelStore(set_key, max_loaded=4)
    splits = jload(results_path(set_key, "splits"))
    done = {} if force else load_fields(set_key)
    ys = {TARGET: np.asarray(meta["gap"], float), "HOMO_eV": np.asarray(meta["homo"], float),
          "LUMO_eV": np.asarray(meta["lumo"], float)}
    gx = xtb_gaps(set_key, meta["names"]) if DELTA in targets else None
    if DELTA in targets and gx is None:
        raise SystemExit(f"the xTB gaps of COMPAS-{set_key} are not available (the x table of the "
                         "family must be in the cache and list every molecule)")
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
                line = None
                if target == DELTA:
                    y, line = delta_target(ys[TARGET], gx, np.asarray(split["train"], int))
                else:
                    y = ys[target]
                res = evaluate_set(store, groups, split, y, target=target, layer=layer)
                res["y_test"] = y[np.asarray(split["test"], int)]
                if line is not None:
                    res["xtb_line"] = {"slope": line[0], "intercept": line[1]}
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
    carbon_only = meta.get("carbon_only", True)
    atom_types = tuple(meta.get("atom_types") or pbh.PBH_TYPES)
    ring_classes_ = tuple(meta.get("ring_classes") or pbh.RING_CLASSES)
    ring_types = tuple(meta.get("ring_types") or ())
    names_c = pbh.count_names(atom_types, ring_classes_, ring_types, sextet=carbon_only, homa=True)
    counts = np.array([pbh.count_features(m, atom_types=atom_types, ring_classes_=ring_classes_,
                                          ring_types=ring_types, sextet=carbon_only, homa=True,
                                          carbon_only=carbon_only) for m in ds.mols])
    X_desc, desc_names = descriptor_matrix(ds.mols)
    n_rings = np.asarray(meta["n_rings"], float)[:, None]
    gx = xtb_gaps(set_key, meta["names"])
    out = {"count_names": list(names_c), "n_descriptors": len(desc_names), "splits": {},
           "xtb_gaps": gx is not None}
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
                              "coef": dict(zip(names_c, (mdl.coef_ / sc.scale_).tolist()))}
        if gx is not None:
            # the linear xTB-to-DFT correction, fitted on the training part: the Delta baseline
            _d, (slope, icpt) = delta_target(y, gx, tr)
            rec["xtb_linear"] = {"slope": slope, "intercept": icpt,
                                 "test": metrics(y[te], slope * gx[te] + icpt)}
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
# Stage: nics (the magnetic aromaticity index at the ring centres of the small molecules)
# --------------------------------------------------------------------------- #

NICS_LEVEL = "B3LYP D3BJ def2-SVP NMR TightSCF"
NICS_MAX_RINGS = 7


def _nics_input(mol: Chem.Mol, rings: Sequence[pbh.Ring], height: float = 1.0) -> str:
    """An ORCA NMR input with one ghost hydrogen 1 A above every ring centroid (the molecule
    is in its planar frame, so the normal is z); the shielding at a ghost atom is minus the
    nucleus-independent chemical shift NICS(1)."""
    pos = mol.GetConformer().GetPositions()
    lines = ["! " + NICS_LEVEL, "%maxcore 3000", "* xyz 0 1"]
    for a, p in zip(mol.GetAtoms(), pos):
        lines.append(f"{a.GetSymbol():2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}")
    for r in rings:
        c = r.centroid + np.array([0.0, 0.0, height])
        lines.append(f"H: {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}")
    lines.append("*")
    return "\n".join(lines) + "\n"


def _parse_nics(text: str, n_real: int, n_rings: int) -> Optional[Dict[str, List[float]]]:
    """The isotropic shielding and the zz component of the shielding tensor at the ghost
    nuclei (the last `n_rings` nuclei), as NICS values (minus the shielding)."""
    if "ORCA TERMINATED NORMALLY" not in text:
        return None
    lines = text.splitlines()
    iso: Dict[int, float] = {}
    zz: Dict[int, float] = {}
    for i, line in enumerate(lines):
        if line.startswith(" Nucleus") and line.rstrip().endswith(":"):
            try:
                k = int(line.split()[1].rstrip("H:").rstrip("CH"))
            except ValueError:
                continue
            for j in range(i, min(i + 40, len(lines))):
                if lines[j].startswith("Total shielding tensor"):
                    zz[k] = float(lines[j + 3].split()[2])
                if lines[j].startswith(" Total") and "iso=" in lines[j]:
                    iso[k] = float(lines[j].split("iso=")[1])
                    break
    ghosts = list(range(n_real, n_real + n_rings))
    if any(k not in iso or k not in zz for k in ghosts):
        return None
    return {"nics1_iso": [-iso[k] for k in ghosts], "nics1_zz": [-zz[k] for k in ghosts]}


def nics(set_key: str = "1D", max_rings: int = NICS_MAX_RINGS, jobs: int = 10,
         force: bool = False) -> dict:
    """NICS(1) and NICS(1)zz at every ring centre of the molecules of up to `max_rings` rings
    (179 molecules of COMPAS-1D), with ORCA, serial jobs run `jobs` at a time; the HOMA index,
    the Clar-sextet share and the annulation class of every ring are stored beside them, so
    that the geometric and the magnetic aromaticity indices can be compared ring by ring.
    A full-set NICS field would need about a week of this machine (five minutes per six-ring
    molecule, a quarter of an hour per ten-ring molecule, one core each)."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    from cmfqsar.orca import find_orca

    path = results_path(set_key, "nics")
    if os.path.isfile(path) and not force:
        return jload(path)
    ds, meta = load_prepared(set_key)
    exe = find_orca()
    work = ensure_dir(os.path.join(CACHE, f"nics_{set_key}"))
    pick = [i for i, nr in enumerate(meta["n_rings"]) if int(nr) <= max_rings]
    log(f"NICS(1) for {len(pick)} molecules of up to {max_rings} rings, {jobs} ORCA jobs at a time")
    rings_of = {}
    for i in pick:
        rings = pbh.ring_classes(ds.mols[i], carbon_only=meta.get("carbon_only", True))
        pbh.clar_weights(ds.mols[i], rings)
        rings_of[i] = rings

    def run(i: int) -> Tuple[int, Optional[dict], float]:
        name = ds.names[i]
        d = ensure_dir(os.path.join(work, name))
        out = os.path.join(d, "nics.out")
        t0 = time.time()
        if not (os.path.isfile(out) and "ORCA TERMINATED NORMALLY" in open(out, errors="replace").read()):
            with open(os.path.join(d, "nics.inp"), "w") as fh:
                fh.write(_nics_input(ds.mols[i], rings_of[i]))
            with open(out, "w") as fh:
                subprocess.run([exe, "nics.inp"], cwd=d, stdout=fh, stderr=subprocess.STDOUT)
        parsed = _parse_nics(open(out, errors="replace").read(), ds.mols[i].GetNumAtoms(), len(rings_of[i]))
        return i, parsed, time.time() - t0

    t0 = time.time()
    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for k, (i, parsed, dt) in enumerate(pool.map(run, pick)):
            if parsed is None:
                failed.append(ds.names[i])
                continue
            rings = rings_of[i]
            rows.append({"name": ds.names[i], "n_rings": int(meta["n_rings"][i]), "gap": float(meta["gap"][i]),
                         "classes": [r.cls for r in rings], "sextet": [float(r.sextet) for r in rings],
                         "homa": [float(r.homa) for r in rings], "seconds": dt, **parsed})
            if k % 20 == 0:
                log(f"  {k + 1}/{len(pick)} molecules, {time.time() - t0:.0f} s")
    out = {"set": set_key, "level": NICS_LEVEL, "height_A": 1.0, "max_rings": max_rings,
           "n_molecules": len(rows), "failed": failed, "jobs": jobs, "seconds": time.time() - t0,
           "molecules": rows}
    # the ring-by-ring agreement of the indices
    h = np.array([v for r in rows for v in r["homa"]])
    s = np.array([v for r in rows for v in r["sextet"]])
    zi = np.array([v for r in rows for v in r["nics1_iso"]])
    zzv = np.array([v for r in rows for v in r["nics1_zz"]])
    cls = [c for r in rows for c in r["classes"]]
    out["rings"] = int(len(h))
    out["correlations"] = {"homa_vs_nics1_zz": float(np.corrcoef(h, zzv)[0, 1]),
                           "homa_vs_nics1_iso": float(np.corrcoef(h, zi)[0, 1]),
                           "sextet_vs_nics1_zz": float(np.corrcoef(s, zzv)[0, 1]),
                           "sextet_vs_homa": float(np.corrcoef(s, h)[0, 1])}
    out["by_class"] = {c: {"n": int(sum(1 for x in cls if x == c)),
                           "nics1_zz_mean": float(np.mean([v for v, x in zip(zzv, cls) if x == c])),
                           "homa_mean": float(np.mean([v for v, x in zip(h, cls) if x == c]))}
                       for c in sorted(set(cls))}
    jdump(out, path)
    log(f"NICS done: {len(rows)} molecules, {len(h)} rings, r(HOMA, NICS(1)zz) = "
        f"{out['correlations']['homa_vs_nics1_zz']:.3f}, {time.time() - t0:.0f} s")
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
    table = build_table(ds, fs, ring_cache=ring_cache, carbon_only=meta.get("carbon_only", True))
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
        # no annulation strings (COMPAS-2, COMPAS-3): a six-ring test molecule, else the smallest
        six = [i for i in te if int(meta["n_rings"][i]) == 6]
        cands = six or sorted(te, key=lambda i: int(meta["n_rings"][i]))
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
    ring_fig_name = "ring_contributions_gap.png" if set_key == "1D" else f"ring_contributions_gap_{set_key}.png"
    plot_ring_contributions(items[1:], os.path.join(figdir, ring_fig_name))
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
        "refit_check": check, "figure": relative_path(fig_path, ROOT),
        "ring_figure": relative_path(os.path.join(figdir, ring_fig_name), ROOT),
        "html": relative_path(html_path, ROOT) if html_ok else "",
        "cube": relative_path(cube_path, ROOT), "extremes": extremes, "rule2": rule2,
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
                   ("ABCD", "random0"),
                   ("ABCDE", "size8"), ("ABCDE", "size9"), ("ABCDE", "size10"),
                   ("ABCD", "size8"), ("ABCD", "size9"), ("ABCD", "size10"))


def resolution(set_key: str = "1D", runs: Sequence[Tuple[str, str]] = RESOLUTION_RUNS,
               engine: Optional[str] = "auto", force: bool = False, wide: bool = False) -> dict:
    """How much of the gap to the published graph networks is resolution.

    The same protocol as the field stage with the attenuation grid extended below, between and
    above its points (:data:`ALPHAS_EXT`), the ridge grid extended down to 1e-6
    (:data:`LAMS_FINE`), and a third configuration with the Gaussian kernel layer on top of the
    per-group kernel; on the random splits of the main field sets.  The extra kernels are built
    on demand.

    With ``wide=True`` the layer-width grid is :data:`WIDE_GAMMAS` instead of :data:`GAMMAS`,
    three points wider than the default edge, and the result is stored apart under
    ``resolution_wide_<set>.json`` so that the two grids can be compared without either
    overwriting the other (plan item PBH6).
    """
    check_krr_equivalence()
    groups_needed = "".join(sorted(set("".join(g for g, _s in runs))))
    kernels(set_key, groups=groups_needed, engine=engine, alphas=ALPHAS_EXT)
    ds, meta = load_prepared(set_key)
    store = KernelStore(set_key, max_loaded=6)
    splits = jload(results_path(set_key, "splits"))
    path = results_path(set_key, "resolution_wide" if wide else "resolution")
    done = jload(path) if os.path.isfile(path) and not force else {}
    y = np.asarray(meta["gap"], float)
    for groups, sname in runs:
        key = f"{TARGET}|{groups}|{sname}"
        if key in done or sname not in splits:
            continue
        split = dict(splits[sname], name=sname)
        res = evaluate_set(store, groups, split, y, alphas=ALPHAS_EXT, lams=LAMS_FINE,
                           layer="gauss", gammas=WIDE_GAMMAS if wide else GAMMAS)
        res["y_test"] = y[np.asarray(split["test"], int)]
        done[key] = res
        jdump(done, path)
        for label, c in res["configs"].items():
            log(f"{key} {label}: alphas {c['alphas']} gamma {c['gamma']} lam {c['lam']:g} "
                f"q2 {c['q2']:.3f} test r2 {c['test']['r2']:.3f} rmse {c['test']['rmse']:.3f} "
                f"mae {c['test']['mae']:.3f}")
        log(f"{key}: chosen {res['chosen']} ({res['seconds']:.0f} s)")
    return done


def ring_shapley(set_key: str = "1D", groups: str = MAIN_SET, split_name: str = "random0",
                 engine: Optional[str] = "auto", force: bool = False,
                 samples: int = 400) -> dict:
    """The per-ring reading of the layered model, which has no co-field (plan item PBH6).

    The Gaussian layer halves the error of the field model on this series and gives up rule 2 for
    it: a model that is not linear in the field has no co-field and no exact per-atom
    decomposition.  What it does have is Shapley values, and `cmfqsar.shapley.atom_shapley` takes
    groups of atoms as the players, so one ring is one player.  For a benzenoid of at most
    fourteen rings the values are exact, every one of the 2**P coalitions being evaluated.

    Three readings of the same molecules are stored side by side:

      exact        `cmfqsar.pbh.ring_contributions` of the linear model, which is rule 3 summed
                   per ring and adds up to the prediction minus the bias
      linear       Shapley values of the same linear model, so that the two decompositions of one
                   model can be compared and the Shapley reading can be judged
      layered      Shapley values of the layered model, which is the reading the item asks for

    The layered model is the linear model's kernel with the layer width and the ridge the
    resolution stage chose, and its own test metrics are stored with it, so a reader can see what
    the model being read is worth.
    """
    path = results_path(set_key, "ringshapley")
    if os.path.isfile(path) and not force:
        return jload(path)
    from scipy.stats import spearmanr

    from cmfqsar.shapley import atom_shapley
    model, table, ds, meta, split, ring_cache, check = refit_model(set_key, groups, split_name,
                                                                   engine)
    y = np.asarray(meta["gap"], float)
    idx_tr = np.asarray(split["train"], int)
    idx_te = np.asarray(split["test"], int)
    res_path = results_path(set_key, "resolution")
    if not os.path.isfile(res_path):
        raise SystemExit("run --stage resolution first: the layer width comes from it")
    key = f"{TARGET}|{groups}|{split_name}"
    done_res = jload(res_path)
    if key not in done_res:
        raise SystemExit(f"the resolution stage has no {key}")
    cfg = done_res[key]["configs"]["gauss"]
    t0 = time.time()
    layered = CMFModel(fieldset=model.fieldset, method="krr", kernel_layer="gauss",
                       params={"lam": float(cfg["lam"]), "gamma": float(cfg["gamma"])},
                       scale_fields=False, engine=engine)
    layered.fit(table.subset(idx_tr), y[idx_tr], cv_folds=0)
    pt_lin = model.predict(table.subset(idx_te))
    pt_lay = layered.predict(table.subset(idx_te))
    log(f"the layered model: gamma {cfg['gamma']:g}, ridge {cfg['lam']:g}, fitted in "
        f"{time.time() - t0:.0f} s; test mae {metrics(y[idx_te], pt_lay)['mae']:.4f} eV against "
        f"{metrics(y[idx_te], pt_lin)['mae']:.4f} eV for the linear model")
    ref = pick_reference(ds, meta, split)
    big = [int(i) for i in idx_te if int(meta["n_rings"][i]) >= 4]
    order = sorted(big, key=lambda i: y[i])
    picks = [order[0], order[1], order[-2], order[-1]]
    out = {"set": set_key, "groups": groups, "split": split_name,
           "gamma": float(cfg["gamma"]), "lam": float(cfg["lam"]),
           "samples": int(samples), "exact_limit": 14,
           "test": {"linear": metrics(y[idx_te], pt_lin), "layered": metrics(y[idx_te], pt_lay)},
           "molecules": []}
    for i in [ref] + picks:
        name = ds.names[i]
        mf = table.molecules[i]
        rings = ring_cache[name]
        # every atom in exactly one ring's group, so the full coalition is the molecule
        players = pbh.ring_players(mf, rings)
        labels = [f"{k + 1} {r.cls}" for k, r in enumerate(rings)]
        exact, _contrib, lost = pbh.ring_contributions(model, mf, rings)
        t1 = time.time()
        sh_lin = atom_shapley(model, mf, groups=players, labels=labels, samples=samples,
                             exact_limit=14)
        sh_lay = atom_shapley(layered, mf, groups=players, labels=labels, samples=samples,
                             exact_limit=14)
        row = {"name": name, "n_rings": len(rings), "classes": [r.cls for r in rings],
               "y": float(y[i]), "rings_exact": exact.tolist(), "lost": float(lost),
               "linear": {"values": sh_lin.values.tolist(), "baseline": float(sh_lin.baseline),
                          "prediction": float(sh_lin.prediction), "exact": bool(sh_lin.exact),
                          "evaluations": int(sh_lin.n_evaluations),
                          "efficiency_error": float(sh_lin.efficiency_error)},
               "layered": {"values": sh_lay.values.tolist(), "baseline": float(sh_lay.baseline),
                           "prediction": float(sh_lay.prediction), "exact": bool(sh_lay.exact),
                           "evaluations": int(sh_lay.n_evaluations),
                           "efficiency_error": float(sh_lay.efficiency_error)},
               "seconds": time.time() - t1}
        if len(rings) > 2:
            row["spearman_layered_against_exact"] = float(spearmanr(sh_lay.values, exact)[0])
            row["spearman_linear_against_exact"] = float(spearmanr(sh_lin.values, exact)[0])
            row["spearman_layered_against_linear"] = float(spearmanr(sh_lay.values,
                                                                     sh_lin.values)[0])
        out["molecules"].append(row)
        log(f"{name}: {len(rings)} rings, {sh_lay.n_evaluations} coalitions, Spearman of the "
            f"layered reading against the linear decomposition "
            f"{row.get('spearman_layered_against_exact', float('nan')):+.3f} "
            f"({row['seconds']:.0f} s)")
    jdump(out, path)
    return out


def _cancellation(values, prediction: float, baseline: float) -> float:
    """How much of a decomposition cancels: the absolute values over what they explain.

    A reading of a prediction adds up to the prediction minus its baseline. If the single values
    are much larger than that sum, they cancel, and each one says less about its own part of the
    molecule than its size suggests. One means no cancellation.
    """
    total = float(np.sum(np.abs(np.asarray(values, float))))
    explained = abs(float(prediction) - float(baseline))
    return total / explained if explained > 0 else float("nan")


def report_ring_shapley(set_key: str = "1D") -> List[str]:
    """The lines of the ring-Shapley section of table.md (plan item PBH6)."""
    path = results_path(set_key, "ringshapley")
    if not os.path.isfile(path):
        return []
    d = jload(path)
    rows = d["molecules"]
    conv = [float(np.max(np.abs(np.asarray(m["linear"]["values"], float)
                                - np.asarray(m["rings_exact"], float)))) for m in rows]
    lines = ["### The per-ring reading of the layered model (Shapley values)", "",
             f"The layered model of field set {{{','.join(d['groups'])}}} on split "
             f"{d['split']}, layer width {d['gamma']:g} and ridge {d['lam']:g}: test MAE "
             f"{d['test']['layered']['mae']:.4f} eV against "
             f"{d['test']['linear']['mae']:.4f} eV for the linear model it is built on. The "
             "layered model is not linear in the field, so it has no co-field and no exact "
             "per-atom decomposition; Shapley values are its reading, with one ring as one "
             "player, and for molecules of at most fourteen rings every coalition is evaluated, "
             "so the values are exact. The baseline both readings start from is the prediction "
             "for a molecule with no atoms.", "",
             "For the linear model the Shapley values are not a different reading: they are "
             "decomposition rule 3's own per-atom shares summed over the same players, which the "
             "tests hold to 1e-10 eV. The only difference to the per-ring decomposition stored "
             "elsewhere in this table is that rule 3 can give half of a shared atom to each of "
             "two rings while a player cannot be shared, and the last column says what that "
             "convention alone moves a ring's value by.", "",
             "| molecule | rings | gap (eV) | coalitions | largest ring value, layered reading "
             "(eV) | largest ring value, linear reading (eV) | cancellation, layered reading | "
             "cancellation, linear reading | Spearman of the two readings | shift from the "
             "shared-atom convention (eV) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m, cv in zip(rows, conv):
        la = np.asarray(m["layered"]["values"], float)
        li = np.asarray(m["linear"]["values"], float)
        s2 = m.get("spearman_layered_against_linear")
        lines.append(f"| {m['name']} | {m['n_rings']} | {m['y']:.3f} | "
                     f"{m['layered']['evaluations']} | "
                     f"{la[int(np.argmax(np.abs(la)))]:+.3f} | "
                     f"{li[int(np.argmax(np.abs(li)))]:+.3f} | "
                     f"{_cancellation(la, m['layered']['prediction'], m['layered']['baseline']):.1f} | "
                     f"{_cancellation(li, m['linear']['prediction'], m['linear']['baseline']):.1f} | "
                     f"{'--' if s2 is None else f'{s2:+.3f}'} | {cv:.3f} |")
    lines.append("")
    worst = max((m["layered"]["efficiency_error"] for m in rows), default=0.0)
    cl = [_cancellation(m["layered"]["values"], m["layered"]["prediction"],
                        m["layered"]["baseline"]) for m in rows]
    cn = [_cancellation(m["linear"]["values"], m["linear"]["prediction"],
                        m["linear"]["baseline"]) for m in rows]
    sp = [m["spearman_layered_against_linear"] for m in rows
          if m.get("spearman_layered_against_linear") is not None]
    lines.append(f"Cancellation is the sum of the absolute ring values divided by what they "
                 f"explain, which is the prediction minus the baseline; one means no "
                 f"cancellation. The layered reading cancels {min(cl):.1f} to {max(cl):.1f} "
                 f"times over, the linear one {min(cn):.1f} to {max(cn):.1f} times. That is the "
                 "clearest result of this section: the linear model's per-ring values are far "
                 "larger than the prediction they add up to, because a small ridge lets the "
                 "coefficients grow and a ring counts positive or negative by its position in "
                 "the frame, while the layered model's values stay close to what they explain. "
                 f"The two readings agree only in part, Spearman {min(sp):+.2f} to "
                 f"{max(sp):+.2f} over these molecules, so the layer changes which ring the "
                 "model gives the gap to and not only how well it predicts. Every reading is "
                 f"exact: the values add up to the prediction minus the baseline to "
                 f"{worst:.1e} eV.")
    lines.append("")
    return lines


def _criterion_by_width(record: dict) -> Dict[Optional[float], float]:
    """The best cross-validated RMSE of the subsample at each layer width of one search.

    The search trace holds one entry per (attenuations, width, ridge) tried.  The minimum over
    everything else at a fixed width is the curve the width is chosen on, and ``None`` is the
    linear kernel, which the search tries as well.
    """
    best: Dict[Optional[float], float] = {}
    for row in record.get("search_trace", []):
        g = row["gamma"]
        err = float(row["rmse_cv_sub"])
        if g not in best or err < best[g]:
            best[g] = err
    return best


def report_resolution_wide(set_key: str = "1D") -> List[str]:
    """The lines of the wider-layer section of table.md (plan item PBH6).

    Nine of ten splits chose the smallest width of the default grid, which is its widest layer,
    so the question was whether a better width lay under that edge.  This section answers it from
    the two stored searches: the curve the width is chosen on, and what the choice is worth on the
    held-out molecules.
    """
    wide_path = results_path(set_key, "resolution_wide")
    base_path = results_path(set_key, "resolution")
    if not (os.path.isfile(wide_path) and os.path.isfile(base_path)):
        return []
    wide, base = jload(wide_path), jload(base_path)
    keys = [k for k in base if k in wide]
    if not keys:
        return []
    added = [g for g in WIDE_GAMMAS if g not in GAMMAS]
    lines = ["### Is a wider layer better? The grid below its own edge",
             "",
             f"The default width grid is {GAMMAS}, and nine of its ten searches chose its "
             f"smallest point, which is its widest layer. The same searches were repeated with "
             f"{tuple(added)} added below that point ({WIDE_GAMMAS}), stored apart. "
             "The first two columns are the curve the width is chosen on: the cross-validated "
             "error of the training subsample at the widest added layer and at the default edge, "
             "in eV. The last two columns are what the choice is worth on the held-out "
             "molecules.",
             "",
             "| field set and split | criterion at the widest added layer | criterion at the "
             "default edge | width chosen, default grid | width chosen, wider grid | test MAE, "
             "default grid (eV) | test MAE, wider grid (eV) |",
             "|---|---|---|---|---|---|---|"]
    moved, worse_of_moved, best_diff = 0, 0, 0.0
    for k in keys:
        gb = base[k]["configs"]["gauss"]
        gw = wide[k]["configs"]["gauss"]
        if gw["gamma"] < min(GAMMAS):
            moved += 1
            d = gw["test"]["mae"] - gb["test"]["mae"]
            if d > 0:
                worse_of_moved += 1
            best_diff = max(best_diff, d)
    for k in keys:
        cw = _criterion_by_width(wide[k])
        gb = base[k]["configs"]["gauss"]
        gw = wide[k]["configs"]["gauss"]
        lines.append(f"| {{{','.join(base[k]['groups'])}}} {base[k]['split']} | "
                     f"{cw.get(min(added), float('nan')):.3f} | "
                     f"{cw.get(min(GAMMAS), float('nan')):.3f} | {gb['gamma']:g} | "
                     f"{gw['gamma']:g} | {gb['test']['mae']:.4f} | {gw['test']['mae']:.4f} |")
    lines.append("")
    flat = []
    for k in keys:
        cw = _criterion_by_width(wide[k])
        inside = [cw[g] for g in WIDE_GAMMAS if g <= min(GAMMAS) and g in cw]
        if len(inside) > 1:
            flat.append(max(inside) - min(inside))
    rise = []
    for k in keys:
        cw = _criterion_by_width(wide[k])
        if min(GAMMAS) in cw and max(GAMMAS) in cw:
            rise.append(cw[max(GAMMAS)] - cw[min(GAMMAS)])
    all_worse = "every one of them" if worse_of_moved == moved else f"{worse_of_moved} of them"
    lines.append(f"The criterion is flat under the edge: from the widest added layer to the "
                 f"default edge it changes by at most {max(flat, default=float('nan')):.3f} eV, "
                 f"while from that edge to the narrowest layer of the grid it grows by "
                 f"{min(rise, default=float('nan')):.3f} to {max(rise, default=float('nan')):.3f} "
                 f"eV. The wider grid moved the choice below the edge on {moved} of "
                 f"{len(keys)} searches, and on {all_worse} the held-out error got worse, by up "
                 f"to {best_diff:.3f} eV. So nothing better was hidden under the edge. On the "
                 "size-extrapolation splits the two disagree by construction: the criterion is "
                 "cross-validation inside the training sizes, which are the smaller molecules, "
                 "and there a layer wide enough to be nearly the linear kernel loses little, "
                 "while the held-out larger molecules are exactly what the layer is worth. The "
                 "default grid therefore keeps its lower edge, and this measurement is the "
                 "reason, not the limit of a grid.")
    lines.append("")
    return lines


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
    size_keys = [k for k in done if "|size" in k]
    if size_keys:
        lines.append("Size-extrapolation splits (train on fewer than N rings, test on N), the same three "
                     "configurations against the coarse-grid linear model of the main table:")
        lines.append("")
        lines.append("| field set | split | coarse linear R2 / RMSE | wide-grid linear R2 / RMSE | "
                     "Gaussian layer R2 / RMSE | layer MAE (eV) | gamma |")
        lines.append("|---|---|---|---|---|---|---|")
        for groups in ("ABCDE", "ABCD"):
            for sname in ("size8", "size9", "size10"):
                k = f"{TARGET}|{groups}|{sname}"
                if k not in done:
                    continue
                r = done[k]
                lin = r["configs"].get("per-group", r["configs"]["shared"])
                ga = r["configs"].get("gauss")
                ck = coarse.get(k)
                cc = ck["configs"][ck["chosen"]] if ck else None
                cell = lambda c: f"{c['test']['r2']:.3f} / {c['test']['rmse']:.3f}" if c else "-"
                lines.append(f"| {{{','.join(groups)}}} | {sname} | {cell(cc)} | {cell(lin)} | {cell(ga)} | "
                             f"{ga['test']['mae']:.3f} | {ga['gamma']:g} |" if ga else
                             f"| {{{','.join(groups)}}} | {sname} | {cell(cc)} | {cell(lin)} | - | - | - |")
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


def linear_cfg(res: dict) -> dict:
    """The chosen configuration among the linear ones (shared or per-group attenuations),
    by the rule of evaluate_set: the validation RMSE, else the cross-validated RMSE."""
    cands = {k: v for k, v in res["configs"].items() if k != "gauss"}
    if res.get("n_valid", 0):
        return min(cands.values(), key=lambda c: c["valid"]["rmse"])
    return min(cands.values(), key=lambda c: c["rmse_cv"])


def set_label(groups: str) -> str:
    return "{" + ",".join(groups) + "}"


def report_fields(set_key: str, field_sets: Sequence[str], target: str = TARGET) -> List[str]:
    """The field-set table of one set: one row per field set (the linear model) and, where the
    Gaussian kernel layer was run, a second row for the layered model."""
    done = load_fields(set_key)
    splits = jload(results_path(set_key, "splits"))
    size_names = [s for s in splits if splits[s]["kind"] == "size"]
    lines: List[str] = []
    head = "| field set | fields | q2 (5-fold CV, train) | test R2 | test RMSE (eV) | test MAE (eV) |"
    for s in size_names:
        n_tr, n_te = len(splits[s]["train"]), len(splits[s]["test"])
        head += f" N={splits[s]['test_rings']} R2 (train {n_tr}, test {n_te}) | N={splits[s]['test_rings']} RMSE |"
    lines.append(head)
    lines.append("|" + "---|" * (head.count("|") - 1))
    for groups in field_sets:
        keys = [k for k in done if k.startswith(f"{target}|{groups}|random")]
        if not keys:
            continue
        for layered in (False, True):
            if layered and not any("gauss" in done[k]["configs"] for k in keys):
                continue
            pick = (lambda r: r["configs"].get("gauss")) if layered else linear_cfg
            chosen = [pick(done[k]) for k in keys if pick(done[k]) is not None]
            label = set_label(groups) + (" + Gaussian layer" if layered else "")
            if len(chosen) != len(SEEDS):
                label += f" ({len(chosen)} seeds)"
            fields_txt = ", ".join(done[keys[0]]["fields"]) if not layered else "the same kernel, exp(-gamma d^2)"
            row = (f"| {label} | {fields_txt} | "
                   f"{_mean_std([c['q2'] for c in chosen])} | {_mean_std([c['test']['r2'] for c in chosen])} | "
                   f"{_mean_std([c['test']['rmse'] for c in chosen])} | {_mean_std([c['test']['mae'] for c in chosen])} |")
            for s in size_names:
                k = f"{target}|{groups}|{s}"
                c = pick(done[k]) if k in done else None
                row += f" {c['test']['r2']:.3f} | {c['test']['rmse']:.3f} |" if c else " - | - |"
            lines.append(row)
    lines.append("")
    return lines


def report_baselines(set_key: str) -> List[str]:
    bpath = results_path(set_key, "baselines")
    if not os.path.isfile(bpath):
        return []
    b = jload(bpath)
    splits = jload(results_path(set_key, "splits"))
    size_names = [s for s in splits if splits[s]["kind"] == "size"]
    lines = ["### Baselines on the same splits", ""]
    head = "| model | test R2 | test RMSE (eV) | test MAE (eV) |"
    for s in size_names:
        head += f" N={splits[s]['test_rings']} R2 | N={splits[s]['test_rings']} RMSE |"
    lines.append(head)
    lines.append("|" + "---|" * (head.count("|") - 1))
    labels = {"count_ridge": f"ridge on {len(b['count_names'])} counts of atom classes, ring classes"
                             + (", ring types" if any(n[2:] in pbh.RING_TYPES for n in b["count_names"]) else "")
                             + " and HOMA (additivity scheme)" if "sum_homa" in b["count_names"] else
                             "ridge on atom-class and ring-class counts (additivity scheme)",
              "rings_line": "straight line in the number of rings",
              "rf_descriptors": f"random forest on {b['n_descriptors']} RDKit descriptors",
              "xtb_linear": "linear correction of the GFN2-xTB gap (slope and intercept fitted on the training part)"}
    for key, label in labels.items():
        rand = [b["splits"][s][key]["test"] for s in b["splits"] if s.startswith("random") and key in b["splits"][s]]
        if not rand:
            continue
        row = (f"| {label} | {_mean_std([m['r2'] for m in rand])} | {_mean_std([m['rmse'] for m in rand])} | "
               f"{_mean_std([m['mae'] for m in rand])} |")
        for s in size_names:
            if s in b["splits"] and key in b["splits"][s]:
                m = b["splits"][s][key]["test"]
                row += f" {m['r2']:.3f} | {m['rmse']:.3f} |"
            else:
                row += " - | - |"
        lines.append(row)
    lines.append("")
    return lines


def report_interpret(set_key: str = "1D") -> List[str]:
    """The interpretation lines of a set: the reference molecule, the figures, the refit check."""
    ipath = results_path(set_key, "interpret")
    if not os.path.isfile(ipath):
        return []
    it = jload(ipath)
    fig = str(it["figure"]).replace("\\", "/")
    ring_fig = str(it["ring_figure"]).replace("\\", "/")
    rt = it["refit_check"]["test"]
    lines = ["### Interpretation", "",
             f"Reference molecule {it['reference']} ({it['reference_rings']} rings, gap "
             f"{it['reference_gap']:.2f} eV); co-field figure `{fig}`; per-ring "
             f"contributions `{ring_fig}`.  The CMFModel refitted in float64 for the figures "
             f"reproduces the table's test predictions (float32 kernels) to "
             f"{it['refit_check']['max_dev']:.1e} eV at worst; its own test R2 is {rt['r2']:.3f}, "
             f"RMSE {rt['rmse']:.3f} eV."]
    if it["refit_check"]["max_dev"] > 0.2:
        lines.append("A difference of that size means the linear model at this ridge is ill-conditioned on "
                     "this set: the contributions of decomposition rule 2 are large and cancel, and the "
                     "per-ring reading is not reliable.")
    lines.append("")
    return lines


def report_delta(set_key: str = "1D") -> List[str]:
    """The Delta-learning rows: the field model on the gap minus its linear xTB estimate."""
    done = load_fields(set_key)
    keys = [k for k in done if k.startswith(DELTA + "|")]
    if not keys:
        return []
    lines = [f"### Delta-learning: the DFT gap from the GFN2-xTB gap and the fields (COMPAS-{set_key})", ""]
    lines.append("Target: the DFT gap minus a straight line in the GFN2-xTB gap, the line fitted on the "
                 "training part of each split (the xTB gaps from the x table of the same family).  The "
                 "metrics are on the gap scale, since adding the line back changes no difference.  The "
                 "first row is the line alone.")
    lines.append("")
    sets_here = sorted({k.split("|")[1] for k in keys}, key=len)
    lines.extend(report_fields(set_key, sets_here, target=DELTA))
    bpath = results_path(set_key, "baselines")
    if os.path.isfile(bpath):
        b = jload(bpath)
        rand = [b["splits"][s]["xtb_linear"]["test"] for s in b["splits"]
                if s.startswith("random") and "xtb_linear" in b["splits"][s]]
        sizes = [(s, b["splits"][s]["xtb_linear"]["test"]) for s in b["splits"]
                 if s.startswith("size") and "xtb_linear" in b["splits"][s]]
        if rand:
            lines.append("The line alone (every random seed): test R2 " + _mean_std([m["r2"] for m in rand])
                         + ", RMSE " + _mean_std([m["rmse"] for m in rand]) + " eV, MAE "
                         + _mean_std([m["mae"] for m in rand]) + " eV; size splits: "
                         + "; ".join(f"{s} R2 {m['r2']:.3f}, RMSE {m['rmse']:.3f}" for s, m in sizes) + ".")
            lines.append("")
    return lines


def report_nics(set_key: str = "1D") -> List[str]:
    path = results_path(set_key, "nics")
    if not os.path.isfile(path):
        return []
    n = jload(path)
    c = n["correlations"]
    lines = [f"### NICS(1) against the HOMA index and the Clar-sextet share, ring by ring (COMPAS-{set_key})", ""]
    lines.append(f"ORCA, {n['level']}, one ghost atom 1 A above every ring centroid, {n['n_molecules']} "
                 f"molecules of up to {n['max_rings']} rings ({n['rings']} rings; {n['seconds'] / 60:.0f} min "
                 f"with {n['jobs']} serial jobs).  Correlation coefficients over the rings: HOMA against "
                 f"NICS(1)zz {c['homa_vs_nics1_zz']:.3f}, HOMA against NICS(1) {c['homa_vs_nics1_iso']:.3f}, "
                 f"Clar share against NICS(1)zz {c['sextet_vs_nics1_zz']:.3f}, Clar share against HOMA "
                 f"{c['sextet_vs_homa']:.3f}.")
    lines.append("")
    lines.append("| ring class | rings | mean NICS(1)zz (ppm) | mean HOMA |")
    lines.append("|---|---|---|---|")
    for cls, v in n["by_class"].items():
        lines.append(f"| {cls} | {v['n']} | {v['nics1_zz_mean']:.1f} | {v['homa_mean']:.3f} |")
    lines.append("")
    return lines


SET_TITLES = {"3D": "the peri-condensed polybenzenoid hydrocarbons of COMPAS-3D",
              "2D": "the cata-condensed hetero-polycyclic systems of COMPAS-2D"}


def report_other_set(set_key: str) -> List[str]:
    """The section of a later set: what it holds, the field-set table, the baselines."""
    meta = jload(results_path(set_key, "prepared"))
    fam = set_key[0]
    lines = [f"## COMPAS-{set_key} ({meta['level']}): {SET_TITLES.get(set_key, 'the gap')}, "
             f"{meta['n']} molecules", ""]
    rc = meta.get("ring_class_counts", {})
    txt = (f"Charges: {meta['charge_source']}.  Rings by annulation class: "
           + ", ".join(f"{k} {v}" for k, v in sorted(rc.items())) + ".")
    if meta.get("ring_types"):
        rt = meta.get("ring_type_counts", {})
        txt += "  Rings by type: " + ", ".join(f"{k} {v}" for k, v in sorted(rt.items(), key=lambda kv: -kv[1])) + "."
        txt += "  Atom types of the indicator fields: " + ", ".join(meta["atom_types"]) + "."
    if meta.get("subset"):
        txt += f"  A random subset (seed 0) of the {('52,000' if set_key == '2D' else '')} neutral molecules."
    oop = meta.get("max_out_of_plane_A", {})
    if oop:
        txt += (f"  Largest heavy-atom distance from the molecular plane: median {oop['median']:.2f} A, "
                f"95th percentile {oop['p95']:.2f} A, maximum {oop['max']:.2f} A.")
    lines.append(txt)
    lines.append("")
    lines.append("### Field sets (three random seeds and the size-extrapolation splits)")
    lines.append("")
    lines.extend(report_fields(set_key, FIELD_SETS_BY_FAMILY.get(fam, FIELD_SETS)))
    lines.extend(report_baselines(set_key))
    lines.extend(report_interpret(set_key))
    return lines


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
        lines.append("Field group F is the ring field weighted by the HOMA index of every ring (from the bond "
                     "lengths of the DFT geometry).  A row '+ Gaussian layer' is the same kernel under "
                     "exp(-gamma d^2) (the resolution section explains it); the sets added later were run on "
                     "three seeds.")
        lines.append("")
        lines.extend(report_fields(set_key, FIELD_SETS))
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
                c = linear_cfg(r)
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
        lines.extend(report_baselines(set_key))
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
    lines.extend(report_resolution_wide(set_key))
    lines.extend(report_ring_shapley(set_key))
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
    lines.extend(report_interpret(set_key))
    lines.extend(report_delta(set_key))
    lines.extend(report_nics(set_key))
    for other in ("3D", "2D"):
        if other != set_key and os.path.isfile(results_path(other, "prepared")):
            lines.extend(report_other_set(other))
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
    lines.append("- Whether the gap of the non-planar molecules is served by the planar frame.  Most of the "
                 "ten-ring molecules hold a helicene-like fjord and are twisted: 89 % of the COMPAS-1D "
                 "molecules have a heavy atom more than 0.5 A from the best plane (median of the largest "
                 "distance 1.48 A, 95th percentile 2.91 A).  The frame is defined for them, the two faces "
                 "differ, and a mirror image lands in the mirrored frame, so the model is not symmetric "
                 "under reflection although the gap is.")
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
                 "which is the smallest point of the grid and so its widest layer. Three wider layers were "
                 "measured afterwards (--stage resolution --wide-gammas, stored in resolution_wide_1D.json) "
                 "and nothing better was hidden under that edge: the criterion the width is chosen on "
                 "changes by at most 0.016 eV between gamma 0.03 and 0.25 on all ten searches, and on the "
                 "seven searches where the wider grid moved the choice the held-out error got worse, by up "
                 "to 0.038 eV. On the size-extrapolation splits that is no surprise: the criterion is "
                 "cross-validation inside the training sizes, which are the smaller molecules, so it "
                 "cannot see what the layer is worth on the larger ones it is tested on. On the "
                 "size-extrapolation splits the layer gains as much: the five-group model trained on the "
                 "molecules of up to nine rings predicts the 6,408 ten-ring molecules with R2 0.92 and RMSE "
                 "0.113 eV (linear: 0.70 and 0.213), where the descriptor forest is negative.")
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
    lines.append("")
    lines.append("## The later sets and the follow-ups of the plan (30 October 2026)")
    lines.append("")
    lines.append("- HOMA ring field (group F): the harmonic-oscillator model of aromaticity of every ring from "
                 "the bond lengths of the DFT geometry (Krygowski 1993 parameters for C-C, C-N, C-O, C-S, "
                 "N-N, N-O; a ring with a bond to boron has no parameters and weight 0).  It stands in for "
                 "the NICS field of the task text, which would need about a week of serial ORCA on this "
                 "machine; --stage nics computes NICS(1) and NICS(1)zz (B3LYP-D3BJ/def2-SVP, GIAO, ghost "
                 "atom 1 A above each centroid) for the 179 molecules of up to seven rings and reports the "
                 "ring-by-ring correlation with HOMA and with the Clar share (table.md).")
    lines.append("- Delta-learning (target DELTA_eV): the DFT gap minus a straight line in the GFN2-xTB gap of "
                 "the same molecule (COMPAS-1x lists every COMPAS-1D molecule), the line fitted on the training "
                 "part of every split; the field model learns the residual.  The 'corrected' columns of "
                 "COMPAS-1x are not used: they are fitted to another functional and are 2.2 eV from the "
                 "B3LYP gaps.  The multi-task form (cmfqsar/multitask.py) was not run: the residual form is "
                 "the standard one and the xTB value exists for every molecule.")
    lines.append("- COMPAS-3D (peri-condensed): the same reader (xyz archive) and frame; a fifth annulation "
                 "class 'peri' (a ring two of whose shared bonds touch, or fused to four or more rings); the "
                 "C.3r indicator field is present; Clar sextets as before (the enumeration is over all "
                 "independent ring sets, fine for up to ten rings).  Three seeds and the three size splits; "
                 "sets D, ABCD, ABCDE, ABCDEF, each with the Gaussian layer as a third configuration.")
    lines.append("- COMPAS-2D (hetero atoms): read from the gzipped SDF (neutral rows), a random subset of "
                 "10,000 of the 52,000 molecules (seed 0) so that the stored-kernel protocol applies; every "
                 "ring of four to six atoms is a ring; the atom types add N, N.H, O, S, B, B.H; group R is "
                 "one ring-centred field per ring type of the table (benzene, pyridine, pyrazine, borinine, "
                 "1,4-diborinine, 1,2-dihydro-1,2-diborinine, pyrrole, furan, thiophene, borole, "
                 "cyclobutadiene); no Clar field (undefined for hetero rings).  Sets C, D, R, ABCD, ABCDR, "
                 "ABCDFR, three seeds and the size splits.")
    oop = []
    for s in ("1D", "3D", "2D"):
        pth = results_path(s, "prepared")
        if os.path.isfile(pth):
            d = jload(pth).get("max_out_of_plane_A") or {}
            if d:
                oop.append(f"COMPAS-{s}: median {d['median']:.2f} A, 95th percentile {d['p95']:.2f} A, "
                           f"maximum {d['max']:.2f} A")
    lines.append("- Non-planarity: the frame is defined for every molecule, and the largest distance of a heavy "
                 "atom from the molecular plane is recorded in prepared_<set>.json ("
                 + "; ".join(oop) + ").  The peri-condensed molecules with a cove or fjord "
                 "region are twisted, and the two faces of their frame differ; the frame rule keeps the sign "
                 "convention of the planar case and does not choose between the faces.")
    lines.append("- Open: NICS on the full set (a week of this machine, or a cluster); the multi-task form of "
                 "Delta-learning; HOMA parameters for boron; a frame from the skeleton rather than from the "
                 "masses for the hetero-atom set, and then the 52,000 molecules of COMPAS-2 beyond the subset "
                 "(the landmark path of --stage scaling applies to them unchanged); a kernel averaged over "
                 "the two faces of the frame, so that a mirror image gets the same prediction.")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS, "NOTES.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

STAGES = ("prepare", "kernels", "fields", "baselines", "interpret", "scaling", "resolution",
          "ringshapley", "nics",
          "report")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=STAGES + ("all",), default="all")
    ap.add_argument("--set", default="1D", choices=tuple(FILES))
    ap.add_argument("--charges", default="xtb", choices=("xtb", "gasteiger"))
    ap.add_argument("--jobs", type=int, default=8, help="xtb runs at a time")
    ap.add_argument("--limit", type=int, default=None, help="first molecules only (smoke tests)")
    ap.add_argument("--sets", default=",".join(FIELD_SETS), help="field sets, comma separated")
    ap.add_argument("--splits", default="", help="split names, comma separated (default: all)")
    ap.add_argument("--targets", default=TARGET,
                    help=f"targets, comma separated ({TARGET}, HOMO_eV, LUMO_eV, {DELTA})")
    ap.add_argument("--layer", default="", choices=("", "gauss"),
                    help="add the Gaussian kernel layer as a configuration of the field stage")
    ap.add_argument("--subset", type=int, default=None,
                    help=f"molecules of COMPAS-2D to use (default {SUBSET_2D})")
    ap.add_argument("--engine", default="auto", help="kernel engine: auto, numba or numpy")
    ap.add_argument("--landmarks", default=",".join(str(m) for m in NYSTROEM_M))
    ap.add_argument("--wide-gammas", action="store_true",
                    help="resolution stage: three wider layers under the default grid's edge, "
                         "stored apart (plan item PBH6)")
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
            prepare(args.set, charges=args.charges, jobs=args.jobs, limit=args.limit, force=args.force,
                    subset=args.subset)
        elif stage == "kernels":
            kernels(args.set, engine=engine, force=args.force)
        elif stage == "fields":
            sets = [s for s in args.sets.split(",") if s]
            if args.sets == ",".join(FIELD_SETS):
                sets = list(FIELD_SETS_BY_FAMILY.get(args.set[0], FIELD_SETS))
            fields(args.set, sets=sets,
                   splits_wanted=[s for s in args.splits.split(",") if s] or None,
                   targets=[t for t in args.targets.split(",") if t], force=args.force,
                   layer=args.layer or None)
        elif stage == "baselines":
            baselines(args.set, force=args.force)
        elif stage == "interpret":
            interpret(args.set, engine=engine, force=args.force)
        elif stage == "scaling":
            scaling(args.set if args.set != "1D" else "1x", limit=args.limit,
                    ms=[int(m) for m in args.landmarks.split(",")], charges=args.charges,
                    jobs=args.jobs, engine=engine, force=args.force)
        elif stage == "resolution":
            resolution(args.set, engine=engine, force=args.force, wide=args.wide_gammas)
        elif stage == "ringshapley":
            chosen = args.sets.split(",")[0] if args.sets != ",".join(FIELD_SETS) else MAIN_SET
            ring_shapley(args.set, groups=chosen or MAIN_SET, engine=engine, force=args.force)
        elif stage == "nics":
            nics(args.set, jobs=args.jobs, force=args.force)
        elif stage == "report":
            pass
    print(report(os.path.join(RESULTS, "table.md"), "1D" if os.path.isfile(results_path("1D", "prepared")) else args.set))
    if os.path.isfile(results_path("1D", "prepared")):
        write_notes("1D")


if __name__ == "__main__":
    main()
