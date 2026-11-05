#!/usr/bin/env python
"""
A single-atom site as the sample: the field around the metal against what H, O, OH and OOH
bind for (plan item EC8, the pilot).

A metal atom held by four atoms of a graphene divacancy, nitrogen or carbon, is the active
site of the metal-nitrogen-carbon catalysts of the oxygen reduction and the hydrogen evolution
reactions.  Its activity is read, at the thermodynamic level, from four binding energies: of H
(the hydrogen evolution descriptor), and of OH, O and OOH (the oxygen reduction descriptors,
which the computational hydrogen electrode turns into a limiting potential).  A catalyst that
does both reactions is one whose H binding is near zero *and* whose OH binding sits on the top
of the oxygen reduction volcano, so the question is a two-endpoint one, and the quantity a
design wants is the difference between the two co-fields.

The data are the 140 catalysts of He, Wang and Ouyang (Chem. Mater. 2023, DOI
10.1021/acs.chemmater.3c02549; raw VASP files at github.com/Jeff-oakley/SAC_HOER_data): 28
metals in the five motifs M@C4, M@C3N, M@C2N2, M@CN3 and M@N4 of an 8x8 graphene sheet, PBE+D2,
each with the bare sheet and the four adsorbed states relaxed.  ``--prepare`` reads that clone
once and writes ``examples/sac``; everything else reads ``examples/sac``.

What is measured
----------------
The sample is the **bare site**: the environment of the metal cut from the relaxed sheet
without any adsorbate (:func:`cmfqsar.hops.point_environment`, the periodic environment of a
point), in a frame that needs no alignment because every sheet of the set is the same 8x8
template with the metal at the same lattice position: origin under the metal in the plane of
the sheet, normal along z, the metal on the +z side.  The targets are the four electronic
binding energies with the gas references H2 and H2O computed at the same settings
(``examples/sac/gas_references.json``), one model per target and the four together.

Three designs of folds, because the set is a grid of metals by motifs and the question a
chemist asks depends on which axis is new:

  random      ten folds over the 139 catalysts: a catalyst whose metal and motif are both
              known from other members of the set
  metal-out   one fold per metal (28): a metal the model has never seen, in known motifs
  motif-out   one fold per motif (5): a coordination the model has never seen

and in each design every method chooses its settings in an inner cross-validation of the same
kind, so the comparison favours none.  The references under rule 1: a one-hot of the metal and
the motif with no structure at all; tabulated properties of the metal with the nitrogen count
(the "d-band periodicity and local hybridization" of the source paper, as a kernel); the
periodic autocorrelation of the whole cell (the reference of plan items M1 to M14); and the
rotation-invariant autocorrelation of the same site environment (plan item LP2), which is what
the overlap kernel has to beat to justify its frame and its co-field.

Then, on the overlap kernel of the site: the four endpoints in one multi-task fit against four
separate ones, the predicted difference of the H and OH binding (the bifunctional selectivity),
the limiting potentials and overpotentials derived from the out-of-fold binding energies, the
GP variance of a metal the model has not seen against its error, and the co-fields of the H
binding, the OH binding and their difference drawn in the plane of the sheet.

Usage
-----
    python benchmarks/run_sac.py --prepare   # from the clone in benchmarks/cache/sac_hoer/repo
    python benchmarks/run_sac.py             # the measurement, from examples/sac
    python benchmarks/run_sac.py --report    # the table from the stored results
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import io
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar import multitask as MT  # noqa: E402
from cmfqsar.cofields import evaluate_gaussians  # noqa: E402
from cmfqsar.fields import AtomTable, MoleculeFields  # noqa: E402
from cmfqsar.hops import point_environment  # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel  # noqa: E402
from cmfqsar.localenv import _ridge_path, local_features, local_kernel  # noqa: E402
from cmfqsar.model import CMFModel, _Learner, krr_fit, r2_score, rmse  # noqa: E402
from cmfqsar.periodic import (PeriodicSpec, crystal_fields, inorganic_fieldset,  # noqa: E402
                              periodic_feature_matrix)

RESULTS = os.path.join(HERE, "results", "sac")
EXAMPLES = os.path.join(ROOT, "examples", "sac")
CLONE = os.path.join(HERE, "cache", "sac_hoer", "repo", "RawData")
STRUCTURES = os.path.join(EXAMPLES, "substrates.extxyz")
TABLE = os.path.join(EXAMPLES, "sac_hoer.csv")
GAS = os.path.join(EXAMPLES, "gas_references.json")
RECOMPUTED = os.path.join(EXAMPLES, "substrates_pbe.csv")

MOTIFS = ("0N0", "1N1", "2N2", "3N3", "4N4")
MOTIF_NAMES = {"0N0": "C4", "1N1": "C3N", "2N2": "C2N2", "3N3": "CN3", "4N4": "N4"}
ADSORBATES = ("H", "O", "OH", "OOH")
#: the gas references of the computational hydrogen electrode: E(*A) - E(*) - a E(H2O) - b E(H2)
REFERENCE = {"H": (0.0, 0.5), "O": (1.0, -1.0), "OH": (1.0, -0.5), "OOH": (2.0, -1.5)}
#: zero-point and entropy corrections from binding energy to binding free energy, in eV: the
#: values in common use for these intermediates on metals and on M-N-C sites (Norskov et al.
#: 2004, 2005), not the source paper's own, which it does not tabulate
FREE_ENERGY_CORRECTION = {"H": 0.24, "O": 0.05, "OH": 0.35, "OOH": 0.40}
#: PBE gas energies in common use, only for a run without examples/sac/gas_references.json
APPROXIMATE_GAS = {"H2": -6.77, "H2O": -14.22}
WATER_FORMATION = 4.92        # 2 H2O from O2 and 2 H2 at the electrode, in eV
EQUILIBRIUM = 1.23            # the oxygen reduction equilibrium potential, in V

TARGETS = [f"dE_{a}" for a in ADSORBATES]
PHYSCHEM = ("electrostatic", "steric", "prop:electronegativity")
EXTENDED = PHYSCHEM + ("prop:d_count", "prop:valence", "prop:ionisation", "prop:radius")
FIELDSETS = ("classic", "extended", "indicator")
RHOS = (0.05, 0.1, 0.2)
R_MAX = 7.0
ALPHAS = (0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
FOLDS = 10
INNER = 5
SEED = 0
RUNS = 2000
PERIODIC_SPEC = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5)
TOP_K = 10


# ----------------------------------------------------------------------------- preparation


def _last_energy(path: str) -> Tuple[float, int]:
    """The energy E0 of the last ionic step of an OSZICAR, and how many steps there were."""
    e, n = float("nan"), 0
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "E0=" in line:
                n += 1
                e = float(line.split("E0=")[1].split()[0])
    return e, n


def _metal_index(atoms) -> int:
    sym = atoms.get_chemical_symbols()
    metals = [i for i, s in enumerate(sym) if s not in ("C", "N", "H", "O")]
    if len(metals) != 1:
        raise ValueError(f"{atoms.get_chemical_formula()}: expected one metal atom, found {len(metals)}")
    return metals[0]


def prepare(clone: str = CLONE, log=print) -> None:
    """Read the clone once: the bare sheets as one extxyz, the energies as one CSV."""
    from ase.io import read, write
    os.makedirs(EXAMPLES, exist_ok=True)
    rows, structures = [], []
    for motif in MOTIFS:
        for sub in sorted(glob.glob(os.path.join(clone, f"{motif}_*_substrate"))):
            metal = os.path.basename(sub).split("_")[1]
            name = f"{metal}@{MOTIF_NAMES[motif]}"
            contcar = os.path.join(sub, "CONTCAR")
            if not os.path.exists(contcar):
                log(f"  {name}: no relaxed bare sheet in the clone, dropped")
                continue
            atoms = read(contcar, format="vasp")
            e_sub, n_sub = _last_energy(os.path.join(sub, "OSZICAR"))
            incar = io.open(os.path.join(sub, "INCAR"), encoding="utf-8", errors="replace").read()
            ldau = int(any(line.strip().upper().startswith("LDAU") and ".TRUE." in line.upper()
                           for line in incar.splitlines()))
            row = {"name": name, "metal": metal, "motif": MOTIF_NAMES[motif],
                   "n_N": int(motif[0]), "E_sub": e_sub, "steps_sub": n_sub, "ldau_sub": ldau,
                   "sub_ok": 1, "sub_note": ""}
            m = _metal_index(atoms)
            pos = atoms.get_positions()
            inplane = np.hypot(*(pos[:, :2] - pos[m, :2]).T)
            far = [i for i in range(len(atoms)) if i != m and inplane[i] > 5.0]
            z0 = float(np.mean(pos[far, 2]))
            row["height"] = abs(float(pos[m, 2]) - z0)
            d = atoms.get_distances(m, [i for i in range(len(atoms)) if i != m], mic=True)
            row["d_first_shell"] = float(np.mean(np.sort(d)[:4]))
            for ads in ADSORBATES:
                folder = os.path.join(clone, f"{motif}_{metal}_{ads}")
                e_ads, n_ads = _last_energy(os.path.join(folder, "OSZICAR"))
                intact = 1
                try:
                    a2 = read(os.path.join(folder, "CONTCAR"), format="vasp")
                    m2 = _metal_index(a2)
                    sym2 = a2.get_chemical_symbols()
                    probe = [i for i, s in enumerate(sym2) if s in ("H", "O")]
                    dmin = float(min(a2.get_distances(m2, probe, mic=True))) if probe else float("nan")
                    # did the adsorbate stay what it was?  OOH with its O-O bond and its O-H
                    # bond, OH with its O-H bond; otherwise the energy is that of another state
                    oxy = [i for i, s in enumerate(sym2) if s == "O"]
                    hyd = [i for i, s in enumerate(sym2) if s == "H"]
                    if ads == "OOH" and len(oxy) == 2:
                        if float(a2.get_distance(oxy[0], oxy[1], mic=True)) > 1.65:
                            intact = 0
                    if ads in ("OH", "OOH") and oxy and hyd:
                        if float(min(a2.get_distances(hyd[0], oxy, mic=True))) > 1.25:
                            intact = 0
                except Exception as exc:  # a malformed or missing CONTCAR
                    log(f"  {name} {ads}: {exc}; the geometry is missing, the energy is kept "
                        f"only if its OSZICAR has one")
                    dmin = float("nan")
                row[f"E_{ads}"] = e_ads
                row[f"steps_{ads}"] = n_ads
                row[f"d_metal_{ads}"] = dmin
                row[f"intact_{ads}"] = intact
            # a bare-sheet energy that cannot be subtracted from the adsorbed states: relaxed
            # with DFT+U (they were not), or so far from them that every binding energy comes
            # out below -3.5 eV (Os@C4, whose source run recorded a single ionic step)
            if ldau:
                row["sub_note"] = "dft+u"
            else:
                rough = [row[f"E_{a}"] - e_sub - REFERENCE[a][0] * APPROXIMATE_GAS["H2O"]
                         - REFERENCE[a][1] * APPROXIMATE_GAS["H2"] for a in ADSORBATES
                         if np.isfinite(row[f"E_{a}"])]
                if rough and max(rough) < -3.5:
                    row["sub_note"] = "an energy inconsistent with its adsorbed states"
            row["sub_ok"] = int(not row["sub_note"])
            atoms.info["name"] = name
            structures.append(atoms)
            rows.append(row)
    write(STRUCTURES, structures, format="extxyz")
    keys = list(rows[0])
    with io.open(TABLE, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    log(f"  wrote {len(rows)} catalysts to {TABLE} and their bare sheets to {STRUCTURES}")


# ----------------------------------------------------------------------------- the data


@dataclasses.dataclass
class SacSet:
    names: List[str]
    metals: List[str]
    motifs: List[str]
    n_N: np.ndarray
    structures: list                       # ASE Atoms of the bare sheets
    centres: np.ndarray                    # (n, 3) the point under the metal in the sheet plane
    flip: np.ndarray                       # (n,) True where the metal sits on the -z side
    height: np.ndarray
    dE: Dict[str, np.ndarray]              # binding energies by adsorbate, eV
    dG: Dict[str, np.ndarray]
    gas: dict
    notes: List[str]

    def __len__(self) -> int:
        return len(self.names)

    @property
    def eta_her(self) -> np.ndarray:
        return np.abs(self.dG["H"])

    @property
    def eta_orr(self) -> np.ndarray:
        return orr_overpotential(self.dG["OH"], self.dG["O"], self.dG["OOH"])


def orr_limiting_potential(g_oh, g_o, g_ooh) -> np.ndarray:
    """The limiting potential of the four-electron associative path, in V."""
    g_oh, g_o, g_ooh = (np.asarray(v, float) for v in (g_oh, g_o, g_ooh))
    steps = np.stack([g_ooh - WATER_FORMATION, g_o - g_ooh, g_oh - g_o, -g_oh])
    return -np.max(steps, axis=0)


def orr_overpotential(g_oh, g_o, g_ooh) -> np.ndarray:
    return EQUILIBRIUM - orr_limiting_potential(g_oh, g_o, g_ooh)


def free_energies(dE: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {a: dE[a] + FREE_ENERGY_CORRECTION[a] for a in ADSORBATES}


def load_set(log=print) -> SacSet:
    from ase.io import read
    structures = read(STRUCTURES, index=":", format="extxyz")
    rows = list(csv.DictReader(io.open(TABLE, encoding="utf-8")))
    if len(rows) != len(structures):
        raise ValueError(f"{TABLE} has {len(rows)} rows and {STRUCTURES} {len(structures)} structures")
    notes = []
    if os.path.exists(GAS):
        gas = json.load(io.open(GAS, encoding="utf-8"))
        e_h2 = gas["molecules"]["H2"]["energy_eV"]
        e_h2o = gas["molecules"]["H2O"]["energy_eV"]
    else:
        gas = {"approximate": True, "molecules": {k: {"energy_eV": v} for k, v in APPROXIMATE_GAS.items()}}
        e_h2, e_h2o = APPROXIMATE_GAS["H2"], APPROXIMATE_GAS["H2O"]
        notes.append("gas references are the approximate PBE constants, not computed ones")
    # the bare sheets the clone relaxed with DFT+U cannot be subtracted from adsorbed states
    # relaxed without it; they are replaced by the plain-PBE energies of examples/sac/
    # substrates_pbe.csv where that run has finished, and left without a target otherwise
    recomputed = {}
    if os.path.exists(RECOMPUTED):
        for r in csv.DictReader(io.open(RECOMPUTED, encoding="utf-8")):
            motif, metal = r["name"].split("_")[:2]
            if r["E_sub_pbe"] not in ("", "nan"):
                recomputed[f"{metal}@{MOTIF_NAMES[motif]}"] = float(r["E_sub_pbe"])
    e_sub, bad, fixed = [], {}, {}
    for r in rows:
        e = float(r["E_sub"])
        if not int(r.get("sub_ok", 1)):
            why = r.get("sub_note", "") or "unusable"
            bad[why] = bad.get(why, 0) + 1
            if r["name"] in recomputed:
                e = recomputed[r["name"]]
                fixed[why] = fixed.get(why, 0) + 1
            else:
                e = float("nan")
        e_sub.append(e)
    for why, n in bad.items():
        what = ("were relaxed with DFT+U in the source data while every adsorbed state was relaxed "
                "without it" if why == "dft+u" else f"have a source run with {why}")
        notes.append(f"{n} bare sheets {what}; {fixed.get(why, 0)} of them have a plain-PBE energy "
                     f"recomputed here and {n - fixed.get(why, 0)} carry no binding energy")
    dE = {}
    broken = []
    for a in ADSORBATES:
        ca, cb = REFERENCE[a]
        vals = []
        for r, es in zip(rows, e_sub):
            e = float(r[f"E_{a}"]) if r[f"E_{a}"] not in ("", "nan") else float("nan")
            if not int(r.get(f"intact_{a}", 1)):
                broken.append(f"{r['name']} {a}")
                e = float("nan")
            vals.append(e - es - ca * e_h2o - cb * e_h2)
        dE[a] = np.array(vals)
    if broken:
        notes.append(f"{len(broken)} adsorbed states ended as another species (an O-O or O-H bond "
                     f"broken) and carry no binding energy: {', '.join(broken)}")
    centres, flip, height = [], [], []
    for atoms in structures:
        m = _metal_index(atoms)
        pos = atoms.get_positions()
        inplane = np.hypot(*(pos[:, :2] - pos[m, :2]).T)
        far = [i for i in range(len(atoms)) if i != m and inplane[i] > 5.0]
        z0 = float(np.mean(pos[far, 2]))
        centres.append([pos[m, 0], pos[m, 1], z0])
        flip.append(pos[m, 2] < z0)
        height.append(abs(pos[m, 2] - z0))
    for r in rows:
        for a in ADSORBATES:
            d = r.get(f"d_metal_{a}", "nan")
            if d not in ("", "nan") and float(d) > 2.6:
                notes.append(f"{r['name']}: {a} ended {float(d):.2f} A from the metal, on the sheet "
                             f"rather than on the site; kept")
    return SacSet(names=[r["name"] for r in rows], metals=[r["metal"] for r in rows],
                  motifs=[r["motif"] for r in rows], n_N=np.array([int(r["n_N"]) for r in rows]),
                  structures=structures, centres=np.array(centres), flip=np.array(flip),
                  height=np.array(height), dE=dE, dG=free_energies(dE), gas=gas, notes=notes)


# ----------------------------------------------------------------------------- metal properties


_PROPS: Dict[str, Dict[str, float]] = {}


def element_properties(symbol: str) -> Dict[str, float]:
    """Tabulated properties of a neutral atom (pymatgen's tables)."""
    if symbol not in _PROPS:
        import warnings
        from pymatgen.core import Element
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = Element(symbol)
            d_count = sum(occ for n, l, occ in e.full_electronic_structure
                          if l == "d" and n == e.row - 1)
            valence = e.group if e.group <= 12 else e.group - 10
            _PROPS[symbol] = {
                "Z": float(e.Z), "group": float(e.group), "period": float(e.row),
                "electronegativity": float(e.X), "d_count": float(d_count),
                "valence": float(valence), "ionisation": float(e.ionization_energy),
                "affinity": float(e.electron_affinity if e.electron_affinity is not None else 0.0),
                "radius": float(e.atomic_radius) if e.atomic_radius is not None else 1.5}
    return _PROPS[symbol]


def with_property_arrays(atoms):
    """The structure with the per-atom property arrays the extended field set reads."""
    a = atoms.copy()
    for key in ("d_count", "valence", "ionisation", "radius"):
        a.set_array(key, np.array([element_properties(s)[key] for s in a.get_chemical_symbols()]))
    return a


def fieldset_for(kind: str, metals: Sequence[str], alpha: float = 1.0):
    if kind == "classic":
        return inorganic_fieldset((), physchem=PHYSCHEM, charge_model="electronegativity",
                                  other=False, alpha=alpha)
    if kind == "extended":
        return inorganic_fieldset((), physchem=EXTENDED, charge_model="electronegativity",
                                  other=False, alpha=alpha)
    if kind == "indicator":
        return inorganic_fieldset(["C", "N"] + sorted(set(metals)), physchem=PHYSCHEM,
                                  charge_model="electronegativity", other=False, alpha=alpha)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- environments


def site_environments(data: SacSet, fs, rho: float, r_max: float = R_MAX) -> List[MoleculeFields]:
    """The bare site of every catalyst in the shared frame: origin under the metal in the plane
    of the sheet, the metal on the +z side."""
    out = []
    for k, atoms in enumerate(data.structures):
        crystal = crystal_fields(with_property_arrays(atoms), fs, name=data.names[k])
        env = point_environment(crystal, data.centres[k], rho=rho, r_max=r_max, name=data.names[k])
        if data.flip[k]:
            coords = env.coords.copy()
            coords[:, 2] *= -1.0
            env = dataclasses.replace(env, coords=coords)
        out.append(env)
    return out


def overlap_kernel(table: AtomTable, fs, normalize: bool) -> np.ndarray:
    model = CMFModel(fieldset=fs, normalize=normalize)
    K_fields = model.prepare_training_kernels(table)
    return model.joint_kernel(K_fields)


def crystals_for(data: SacSet, fs):
    return [crystal_fields(with_property_arrays(a), fs, name=n)
            for a, n in zip(data.structures, data.names)]


# ----------------------------------------------------------------------------- cross-validation


def random_folds(n: int, k: int, seed: int) -> List[np.ndarray]:
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    return [np.sort(p) for p in np.array_split(order, k)]


def group_folds(groups: Sequence[str], k: Optional[int] = None, seed: int = 0) -> List[np.ndarray]:
    """One fold per group, or the groups dealt into `k` folds."""
    labels = sorted(set(groups))
    if k is None or k >= len(labels):
        return [np.where(np.asarray(groups) == g)[0] for g in labels]
    order = list(labels)
    np.random.RandomState(seed).shuffle(order)
    parts = np.array_split(np.arange(len(order)), k)
    g = np.asarray(groups)
    return [np.sort(np.concatenate([np.where(g == order[i])[0] for i in p])) for p in parts]


def inner_folds_for(design: str, data: SacSet, rows: np.ndarray, seed: int) -> List[np.ndarray]:
    """Folds *within* a working set, of the same kind as the outer design (indices into rows)."""
    if design == "random":
        return random_folds(len(rows), INNER, seed)
    key = data.metals if design == "metal-out" else data.motifs
    return group_folds([key[i] for i in rows], k=INNER if design == "metal-out" else None, seed=seed)


def outer_folds_for(design: str, data: SacSet, seed: int) -> List[np.ndarray]:
    if design == "random":
        return random_folds(len(data), FOLDS, seed)
    if design == "metal-out":
        return group_folds(data.metals)
    if design == "motif-out":
        return group_folds(data.motifs)
    raise ValueError(design)


def _cv_q2(K: np.ndarray, y: np.ndarray, folds: List[np.ndarray], lams: Sequence[float]) -> np.ndarray:
    """q2 of every ridge value, from the folds given (indices into K)."""
    n = len(y)
    pred = np.full((len(lams), n), np.nan)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        if len(tr) < 3:
            continue
        for j, (a, b) in enumerate(_ridge_path(K[np.ix_(tr, tr)], y[tr], lams)):
            pred[j, te] = K[np.ix_(te, tr)] @ a + b
    ok = np.all(np.isfinite(pred), axis=0)
    return np.array([r2_score(y[ok], p[ok]) for p in pred])


@dataclasses.dataclass
class CVResult:
    name: str
    design: str
    target: str
    predicted: np.ndarray
    q2: float
    rmse: float
    mae: float
    chosen: List[Tuple[str, float]]
    variance: Optional[np.ndarray] = None    # GP predictive variance of every held-out sample
    seconds: float = 0.0

    def summary(self) -> str:
        return (f"{self.name:>34s} | {self.design:9s} | {self.target:6s} | q2 {self.q2:+.3f} "
                f"rmse {self.rmse:.3f} mae {self.mae:.3f} eV")

    def entry(self) -> dict:
        counts: Dict[str, int] = {}
        for k, lam in self.chosen:
            counts[f"{k} lam={lam:g}"] = counts.get(f"{k} lam={lam:g}", 0) + 1
        return {"name": self.name, "design": self.design, "target": self.target, "q2": self.q2,
                "rmse": self.rmse, "mae": self.mae, "chosen": counts, "seconds": self.seconds,
                "predicted": [None if not np.isfinite(v) else float(v) for v in self.predicted]}


def cross_validate(name: str, kernels: Dict[str, np.ndarray], y: np.ndarray, design: str,
                   target: str, data: SacSet, seed: int = SEED, lams: Sequence[float] = LAMS,
                   variance: bool = False) -> CVResult:
    """Out-of-fold predictions with the kernel and the ridge chosen inside every working set.

    The candidates are a family of kernels (the attenuations, widths and normalisations of one
    method) by the ridge values; the inner folds are of the same kind as the outer ones, so a
    metal-out model chooses what predicts unseen metals.  Only samples with a finite target
    take part.
    """
    t0 = time.time()
    y = np.asarray(y, float)
    have = np.where(np.isfinite(y))[0]
    n = len(y)
    pred = np.full(n, np.nan)
    var = np.full(n, np.nan)
    chosen = []
    for te_all in outer_folds_for(design, data, seed):
        te = np.intersect1d(te_all, have)
        tr = np.setdiff1d(have, te_all)
        if len(te) == 0 or len(tr) < 5:
            continue
        inner = inner_folds_for(design, data, tr, seed)
        best = None
        for key, K in kernels.items():
            Ktr = K[np.ix_(tr, tr)]
            scores = _cv_q2(Ktr, y[tr], inner, lams)
            j = int(np.argmax(scores))
            if best is None or scores[j] > best[0]:
                best = (scores[j], key, lams[j])
        _, key, lam = best
        K = kernels[key]
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], float(lam))
        pred[te] = K[np.ix_(te, tr)] @ a + b
        chosen.append((key, float(lam)))
        if variance:
            learner = _Learner("gp", {"lam": float(lam)})
            learner.K_train_ = K[np.ix_(tr, tr)]
            v = learner.predict_variance(K[np.ix_(te, tr)], np.diag(K)[te])
            # the amplitude of the kernel in the units of the target, from the working set
            yc = y[tr] - np.mean(y[tr])
            s2 = float(yc @ np.linalg.solve(K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), yc)) / len(tr)
            var[te] = s2 * (v + lam)
    ok = np.isfinite(pred) & np.isfinite(y)
    return CVResult(name=name, design=design, target=target, predicted=pred,
                    q2=r2_score(y[ok], pred[ok]), rmse=rmse(y[ok], pred[ok]),
                    mae=float(np.mean(np.abs(y[ok] - pred[ok]))), chosen=chosen,
                    variance=var if variance else None, seconds=time.time() - t0)


def bootstrap_difference(y: np.ndarray, a: np.ndarray, b: np.ndarray, runs: int = RUNS,
                         seed: int = SEED) -> Tuple[float, float, float]:
    """(difference in q2, low, high) for prediction `a` against `b`, resampling the catalysts."""
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(y)
    for _ in range(runs):
        idx = rng.randint(0, n, n)
        if np.var(y[idx]) <= 0:
            continue
        diffs.append(r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
    d = r2_score(y, a) - r2_score(y, b)
    return float(d), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(a[ok], b[ok])[0])


# ----------------------------------------------------------------------------- the methods


def composition_kernel(data: SacSet) -> Dict[str, np.ndarray]:
    metals = sorted(set(data.metals))
    motifs = sorted(set(data.motifs))
    Z = np.zeros((len(data), len(metals) + len(motifs)))
    for i in range(len(data)):
        Z[i, metals.index(data.metals[i])] = 1.0
        Z[i, len(metals) + motifs.index(data.motifs[i])] = 1.0
    return {"one-hot": Z @ Z.T}


def descriptor_matrix(data: SacSet) -> Tuple[np.ndarray, List[str]]:
    keys = ["Z", "group", "period", "electronegativity", "d_count", "valence", "ionisation",
            "affinity", "radius"]
    X = np.array([[element_properties(m)[k] for k in keys] for m in data.metals], float)
    X = np.column_stack([X, data.n_N.astype(float)])
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    return X, keys + ["n_N"]


def descriptor_kernels(data: SacSet) -> Dict[str, np.ndarray]:
    X, _ = descriptor_matrix(data)
    D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    med = float(np.median(D2[D2 > 0]))
    out = {"linear": X @ X.T}
    for factor in (0.25, 1.0, 4.0):
        out[f"gauss x{factor:g}"] = np.exp(-D2 / (factor * med))
    return out


def periodic_kernels(data: SacSet) -> Dict[str, np.ndarray]:
    out = {}
    for kind in FIELDSETS:
        fs = fieldset_for(kind, data.metals)
        F = periodic_feature_matrix(crystals_for(data, fs), PERIODIC_SPEC, n_fields=fs.n_fields)
        out[kind] = autocorrelation_kernel(F)
    return out


def site_kernels(data: SacSet, kind: str, log=print) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """(overlap kernels, invariant kernels, tables) of the site environments of one field set,
    over the envelope widths, attenuations and normalisations."""
    overlap, invariant, tables = {}, {}, {}
    for rho in RHOS:
        fs = fieldset_for(kind, data.metals)
        envs = site_environments(data, fs, rho)
        table = AtomTable.from_molecules(envs)
        tables[rho] = table
        invariant[f"rho={rho:g}"] = local_kernel(local_features(envs))
        for alpha in ALPHAS:
            fs_a = fieldset_for(kind, data.metals, alpha=alpha)
            for normalize in (False, True):
                key = f"rho={rho:g} alpha={alpha:g} {'cosine' if normalize else 'raw'}"
                overlap[key] = overlap_kernel(table, fs_a, normalize)
        log(f"    {kind}: rho {rho:g}, {np.mean([e.n_atoms for e in envs]):.0f} atoms per "
            f"environment, {len(overlap)} overlap kernels so far")
    return overlap, invariant, tables


# ----------------------------------------------------------------------------- the measurement


def run(args, log) -> dict:
    t0 = time.time()
    data = load_set(log)
    log(f"  {len(data)} catalysts, {len(set(data.metals))} metals, {len(set(data.motifs))} motifs; "
        f"gas references {'approximate' if data.gas.get('approximate') else 'computed'}")
    for note in data.notes:
        log("  note: " + note)
    for a in ADSORBATES:
        v = data.dE[a]
        log(f"  dE_{a}: {np.sum(np.isfinite(v))} values, mean {np.nanmean(v):+.2f}, "
            f"sd {np.nanstd(v):.2f}, range {np.nanmin(v):+.2f} to {np.nanmax(v):+.2f} eV")
    out = {"n": len(data), "metals": sorted(set(data.metals)), "motifs": sorted(set(data.motifs)),
           "gas": data.gas, "notes": data.notes, "targets": {}, "models": [], "pairs": [],
           "folds": FOLDS, "inner": INNER, "rhos": RHOS, "alphas": ALPHAS, "lams": LAMS,
           "free_energy_correction": FREE_ENERGY_CORRECTION}
    for a in ADSORBATES:
        out["targets"][f"dE_{a}"] = {"values": [None if not np.isfinite(v) else float(v) for v in data.dE[a]],
                                     "sd": float(np.nanstd(data.dE[a]))}
    out["names"] = data.names
    out["eta_her"] = data.eta_her.tolist()
    out["eta_orr"] = [None if not np.isfinite(v) else float(v) for v in data.eta_orr]
    out["height"] = data.height.tolist()

    # the kernels, once
    log("  kernels:")
    methods: Dict[str, Dict[str, np.ndarray]] = {
        "one-hot metal and motif": composition_kernel(data),
        "metal properties and N count": descriptor_kernels(data),
        "periodic autocorrelation, whole cell": periodic_kernels(data),
    }
    tables = {}
    for kind in FIELDSETS:
        overlap, invariant, tabs = site_kernels(data, kind, log)
        methods[f"site invariant, {kind}"] = invariant
        methods[f"site overlap, {kind}"] = overlap
        tables[kind] = tabs
    out["methods"] = {m: list(k) for m, k in methods.items()}
    log(f"  {sum(len(k) for k in methods.values())} kernels in {time.time() - t0:.0f} s")

    # the models, per design and target
    designs = ("random", "metal-out", "motif-out")
    results: Dict[Tuple[str, str, str], CVResult] = {}
    for design in designs:
        for target in TARGETS:
            y = data.dE[target[3:]]
            for name, kernels in methods.items():
                r = cross_validate(name, kernels, y, design, target, data,
                                   variance=name.startswith("site overlap"))
                results[(name, design, target)] = r
                out["models"].append(r.entry())
                log("  " + r.summary())
    # the best overlap field set: mean q2 over the targets and over all three fold designs.
    # Every design is an out-of-fold measurement. The random design alone chose the indicator
    # fields on the full set (5 November 2026) by a margin inside the interval, and those fields
    # collapse on an unseen metal as a one-hot does, so a choice made on random folds alone
    # would hand every deliverable the one set that cannot generalise over metals.
    def mean_q2(name, design):
        return float(np.mean([results[(name, design, t)].q2 for t in TARGETS]))

    def mean_all(name):
        return float(np.mean([mean_q2(name, d) for d in designs]))
    best_kind = max(FIELDSETS, key=lambda k: mean_all(f"site overlap, {k}"))
    best_name = f"site overlap, {best_kind}"
    out["best_overlap"] = best_name
    log("  best overlap field set by the mean over the fold designs: " + best_kind + " ("
        + ", ".join(f"{k} {mean_all(f'site overlap, {k}'):+.3f}" for k in FIELDSETS) + ")")
    # paired intervals against every reference, per design and target
    refs = [m for m in methods if not m.startswith("site overlap")]
    for design in designs:
        for target in TARGETS:
            y = data.dE[target[3:]]
            a = results[(best_name, design, target)]
            for ref in refs:
                b = results[(ref, design, target)]
                d, lo, hi = bootstrap_difference(y, a.predicted, b.predicted, runs=args.bootstrap)
                out["pairs"].append({"a": best_name, "b": ref, "design": design, "target": target,
                                     "diff": d, "low": lo, "high": hi})
    # the GP variance of an unseen metal against the error it made
    out["variance"] = []
    for design in ("metal-out", "random"):
        for target in TARGETS:
            r = results[(best_name, design, target)]
            y = data.dE[target[3:]]
            ok = np.isfinite(r.predicted) & np.isfinite(y) & np.isfinite(r.variance)
            z = (y[ok] - r.predicted[ok]) / np.sqrt(np.maximum(r.variance[ok], 1e-12))
            out["variance"].append({"design": design, "target": target,
                                    "coverage_95": float(np.mean(np.abs(z) < 1.96)),
                                    "spearman_sd_vs_error": spearman(np.sqrt(r.variance[ok]),
                                                                     np.abs(y[ok] - r.predicted[ok]))})
            log(f"  GP variance, {design}, {target}: 95% interval covers "
                f"{out['variance'][-1]['coverage_95']:.0%}, rank correlation of the predicted sd "
                f"with the error {out['variance'][-1]['spearman_sd_vs_error']:+.2f}")

    # the four endpoints together: multi-task on the kernel the random design chose most often
    pick = {}
    for target in TARGETS:
        for key, lam in results[(best_name, "random", target)].chosen:
            pick[key] = pick.get(key, 0) + 1
    key = max(pick, key=pick.get)
    K = methods[best_name][key]
    out["multitask"] = {"kernel": key}
    Y = np.column_stack([data.dE[a] for a in ADSORBATES])
    obs = MT.Observations.from_columns(Y, names=tuple(TARGETS))
    rho_mt, lam_mt, score, per = MT.search(K, obs, rhos=(0.0, 0.3, 0.5, 0.7, 0.9), lams=LAMS,
                                           folds=FOLDS, seed=SEED)
    _, lam_alone, score_alone, per_alone = MT.search(K, obs, rhos=(0.0,), lams=LAMS,
                                                     folds=FOLDS, seed=SEED)
    out["multitask"].update({"rho": rho_mt, "lam": lam_mt, "mean_q2_shared": score,
                             "mean_q2_alone": score_alone, "per_task_shared": per,
                             "per_task_alone": per_alone})
    log(f"  multi-task on {key}: rho {rho_mt:g}, mean q2 {score:+.3f} shared against "
        f"{score_alone:+.3f} alone")
    iH, iOH = TARGETS.index("dE_H"), TARGETS.index("dE_OH")
    sel = MT.score_selectivity(K, obs, iH, iOH, rho_mt, lam_mt, folds=FOLDS, seed=SEED)
    diff_truth = data.dE["H"] - data.dE["OH"]
    diff_single = results[(best_name, "random", "dE_H")].predicted - results[(best_name, "random", "dE_OH")].predicted
    ok = np.isfinite(diff_truth) & np.isfinite(diff_single)
    out["selectivity"] = {"q2_shared": sel["q2"], "rmse_shared": sel["rmse_cv"], "n": sel["n"],
                          "q2_single": r2_score(diff_truth[ok], diff_single[ok]),
                          "rmse_single": rmse(diff_truth[ok], diff_single[ok]),
                          "sd": float(np.nanstd(diff_truth))}
    log(f"  dE_H - dE_OH: q2 {sel['q2']:+.3f} from the shared fit, "
        f"{out['selectivity']['q2_single']:+.3f} from the two single fits (sd {np.nanstd(diff_truth):.2f} eV)")

    # the overpotentials from the out-of-fold binding energies
    out["overpotential"] = {}
    for design in designs:
        pred = {a: results[(best_name, design, f"dE_{a}")].predicted for a in ADSORBATES}
        gp = free_energies(pred)
        eta_her_p = np.abs(gp["H"])
        eta_orr_p = orr_overpotential(gp["OH"], gp["O"], gp["OOH"])
        entry = {}
        for label, truth, p in (("HER", data.eta_her, eta_her_p), ("ORR", data.eta_orr, eta_orr_p)):
            ok = np.isfinite(truth) & np.isfinite(p)
            entry[label] = {"mae": float(np.mean(np.abs(truth[ok] - p[ok]))),
                            "spearman": spearman(truth, p),
                            "sd": float(np.nanstd(truth))}
        both = np.maximum(data.eta_her, data.eta_orr)
        both_p = np.maximum(eta_her_p, eta_orr_p)
        ok = np.isfinite(both) & np.isfinite(both_p)
        true_top = set(np.where(ok)[0][np.argsort(both[ok])[:TOP_K]].tolist())
        pred_top = set(np.where(ok)[0][np.argsort(both_p[ok])[:TOP_K]].tolist())
        entry["bifunctional"] = {"spearman": spearman(both, both_p),
                                 "top_k": TOP_K, "recovered": len(true_top & pred_top),
                                 "true_top": [data.names[i] for i in sorted(true_top, key=lambda i: both[i])],
                                 "predicted_top": [data.names[i] for i in sorted(pred_top, key=lambda i: both_p[i])]}
        out["overpotential"][design] = entry
        log(f"  overpotentials, {design}: HER mae {entry['HER']['mae']:.2f} V (rank r "
            f"{entry['HER']['spearman']:+.2f}), ORR mae {entry['ORR']['mae']:.2f} V (rank r "
            f"{entry['ORR']['spearman']:+.2f}); {len(true_top & pred_top)}/{TOP_K} of the best "
            f"bifunctional sites recovered")

    # the co-fields of the H and the OH binding and their difference, in the plane of the sheet
    parts = key.split()
    rho_best = float(parts[0].split("=")[1])
    alpha_best = float(parts[1].split("=")[1])
    norm_best = parts[2] == "cosine"
    table = tables[best_kind][rho_best]
    fs_best = fieldset_for(best_kind, data.metals, alpha=alpha_best)
    cof_models = {}
    for a in ("H", "OH", "OOH"):
        y = data.dE[a]
        have = np.where(np.isfinite(y))[0]
        sub = table.subset(have) if len(have) < len(y) else table
        lam = results[(best_name, "random", f"dE_{a}")].chosen
        lam = float(np.median([l for _, l in lam]))
        m = CMFModel(fieldset=fs_best.copy(), normalize=norm_best, method="gp", params={"lam": lam})
        m.fit(sub, y[have], cv_folds=0)
        cof_models[a] = m
    out["cofield"] = {"fieldset": best_kind, "rho": rho_best, "alpha": alpha_best,
                      "normalize": norm_best, "points": {}}
    ref_env = table.molecules[data.names.index("Fe@N4")] if "Fe@N4" in data.names else table.molecules[0]
    points = named_points(ref_env, float(np.mean(data.height)))
    out["cofield"]["named_points"] = {k: v.tolist() for k, v in points.items()}
    for fname in fs_best.names:
        if fname.startswith("cif:"):
            continue
        f = fs_best.names.index(fname)
        entry = {}
        for a, m in cof_models.items():
            entry[a] = {k: float(np.mean(cofield_at(m, f, v))) for k, v in points.items()}
        entry["H - OH"] = {k: entry["H"][k] - entry["OH"][k] for k in points}
        out["cofield"]["points"][fname] = entry
    os.makedirs(RESULTS, exist_ok=True)
    plot_cofields(cof_models, [n for n in fs_best.names if not n.startswith("cif:")], ref_env,
                  os.path.join(RESULTS, "cofields.png"), planes=(0.0, 1.0))
    plot_scatter(results, best_name, data, os.path.join(RESULTS, "cv_scatter.png"))
    out["seconds"] = time.time() - t0
    return out


def cofield_at(model: CMFModel, f: int, points: np.ndarray) -> np.ndarray:
    centres, coeffs, widths = model.cofield_coefficients(f)
    v = evaluate_gaussians(np.asarray(points, float).reshape(-1, 3), centres, coeffs, widths)
    return v * model.fieldset.specs[f].h * model.scale[f]


def named_points(env: MoleculeFields, metal_height: float) -> Dict[str, np.ndarray]:
    """Places a chemist can name, read from one site: the metal, the first and the second shell
    of the sheet, and the point an adsorbate occupies above the metal."""
    xy = np.hypot(env.coords[:, 0], env.coords[:, 1])
    sheet = np.array([s in ("C", "N") for s in env.symbols])
    first = env.coords[sheet & (xy < 2.4)]
    second = env.coords[sheet & (xy >= 2.4) & (xy < 3.3)]
    return {"metal": np.array([[0.0, 0.0, metal_height]]),
            "first shell": first, "second shell": second,
            "2 A above the metal": np.array([[0.0, 0.0, metal_height + 2.0]])}


# ----------------------------------------------------------------------------- figures


def plot_cofields(models: Dict[str, CMFModel], fields: Sequence[str], ref_env: MoleculeFields,
                  path: str, planes: Sequence[float] = (0.0, 1.0)) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(-5.0, 5.0, 0.1)
    ys = np.arange(-5.0, 5.0, 0.1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    columns = [("H", "dE_H"), ("OH", "dE_OH"), ("H - OH", "dE_H - dE_OH")]
    rows = [(f, z) for f in fields for z in planes]
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(3.4 * len(columns), 3.1 * len(rows)),
                             squeeze=False)
    for i, (fname, z) in enumerate(rows):
        pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
        slices = {}
        for a in ("H", "OH"):
            m = models[a]
            f = m.fieldset.names.index(fname)
            slices[a] = cofield_at(m, f, pts).reshape(X.shape)
        slices["H - OH"] = slices["H"] - slices["OH"]
        vmax = max(np.abs(s).max() for s in slices.values()) or 1.0
        for j, (a, label) in enumerate(columns):
            ax = axes[i, j]
            im = ax.imshow(slices[a], origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            for k in range(1, ref_env.n_atoms):
                s = ref_env.symbols[k]
                c = "black" if s not in ("C", "N") else ("tab:blue" if s == "N" else "0.4")
                ax.plot(ref_env.coords[k, 0], ref_env.coords[k, 1], "o", ms=4 if s != "C" else 2.5,
                        color=c, mfc="none" if s == "C" else c, mew=0.8)
            spec = models["H"].fieldset.specs[models["H"].fieldset.names.index(fname)]
            ax.set_title(f"{label}: {spec.label}, z = {z:g} A", fontsize=8)
            ax.set_xlabel("x / A", fontsize=7)
            if j == 0:
                ax.set_ylabel("y / A", fontsize=7)
            ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.8, pad=0.02).ax.tick_params(labelsize=6)
    fig.suptitle("Co-fields of the binding energies at the site (red: more of the field here "
                 "raises the binding energy, that is weakens the binding); Fe@N4 drawn, N blue",
                 fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(results, best_name: str, data: SacSet, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    designs = ("random", "metal-out", "motif-out")
    fig, axes = plt.subplots(len(designs), len(TARGETS), figsize=(3.6 * len(TARGETS), 3.4 * len(designs)),
                             squeeze=False)
    for i, design in enumerate(designs):
        for j, target in enumerate(TARGETS):
            r = results[(best_name, design, target)]
            y = data.dE[target[3:]]
            ok = np.isfinite(y) & np.isfinite(r.predicted)
            ax = axes[i, j]
            ax.scatter(y[ok], r.predicted[ok], s=12, color="tab:blue")
            lo, hi = min(y[ok].min(), r.predicted[ok].min()), max(y[ok].max(), r.predicted[ok].max())
            ax.plot([lo, hi], [lo, hi], color="gray", lw=0.8)
            err = np.abs(y - r.predicted)
            err[~ok] = -1
            for k in np.argsort(-err)[:5]:
                ax.annotate(data.names[k], (y[k], r.predicted[k]), fontsize=6, xytext=(3, 3),
                            textcoords="offset points")
            ax.set_title(f"{target}, {design}: q2 {r.q2:.2f}, rmse {r.rmse:.2f} eV", fontsize=8)
            ax.set_xlabel("DFT / eV", fontsize=7)
            ax.set_ylabel("out of fold / eV", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(best_name, fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- the table


def _fmt(x, nd=2, sign=False):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def write_table(results_dir: str = RESULTS) -> str:
    out = json.load(io.open(os.path.join(results_dir, "results.json"), encoding="utf-8"))
    lines = ["# A single-atom site as the sample: the field around the metal against the H, O, "
             "OH and OOH binding energies", ""]
    gas = out["gas"]
    lines += [f"The 139 catalysts of He, Wang and Ouyang (Chem. Mater. 2023) with a relaxed bare sheet: "
              f"{len(out['metals'])} metals ({', '.join(out['metals'])}) in the motifs "
              f"{', '.join(out['motifs'])}. The targets are electronic binding energies against "
              f"{'computed' if not gas.get('approximate') else 'approximate'} gas references "
              f"(H2 {gas['molecules']['H2']['energy_eV']:.3f} eV, H2O {gas['molecules']['H2O']['energy_eV']:.3f} eV"
              f"{'' if gas.get('approximate') else ', PBE+D2 at the settings of the set'}). "
              f"Folds: {out['folds']} random, one per metal, one per motif; every method chooses its "
              f"kernel and ridge in an inner cross-validation of the same kind. Differences carry a "
              f"95% bootstrap interval over catalysts.", ""]
    for note in out["notes"]:
        lines.append(f"* {note}")
    if out["notes"]:
        lines.append("")
    lines += ["## The targets", "", "| target | n | mean, eV | sd, eV |", "|---|---:|---:|---:|"]
    for t, v in out["targets"].items():
        vals = np.array([x for x in v["values"] if x is not None])
        lines.append(f"| {t} | {len(vals)} | {vals.mean():+.2f} | {vals.std():.2f} |")
    lines.append("")
    models = out["models"]
    names = []
    for m in models:
        if m["name"] not in names:
            names.append(m["name"])
    for design in ("random", "metal-out", "motif-out"):
        lines += [f"## q² by method, {design} folds", "",
                  "| method | " + " | ".join(TARGETS) + " | mean |",
                  "|---|" + "---:|" * (len(TARGETS) + 1)]
        for name in names:
            q = [next(m["q2"] for m in models if m["name"] == name and m["design"] == design
                      and m["target"] == t) for t in TARGETS]
            lines.append(f"| {name} | " + " | ".join(_fmt(x, sign=True) for x in q)
                         + f" | {np.mean(q):+.2f} |")
        lines.append("")
        lines += [f"RMSE in eV, {design} folds:", "",
                  "| method | " + " | ".join(TARGETS) + " |", "|---|" + "---:|" * len(TARGETS)]
        for name in names:
            q = [next(m["rmse"] for m in models if m["name"] == name and m["design"] == design
                      and m["target"] == t) for t in TARGETS]
            lines.append(f"| {name} | " + " | ".join(_fmt(x) for x in q) + " |")
        lines.append("")
    best = out["best_overlap"]
    lines += [f"## {best} against the references: difference in q² with its interval", "",
              "| design | reference | " + " | ".join(TARGETS) + " |", "|---|---|" + "---:|" * len(TARGETS)]
    for design in ("random", "metal-out", "motif-out"):
        refs = []
        for p in out["pairs"]:
            if p["design"] == design and p["b"] not in refs:
                refs.append(p["b"])
        for ref in refs:
            cells = []
            for t in TARGETS:
                p = next(p for p in out["pairs"] if p["design"] == design and p["b"] == ref and p["target"] == t)
                cells.append(f"{p['diff']:+.2f} [{p['low']:+.2f}, {p['high']:+.2f}]")
            lines.append(f"| {design} | {ref} | " + " | ".join(cells) + " |")
    lines.append("")
    lines += ["## What the overlap models chose", ""]
    for design in ("random", "metal-out", "motif-out"):
        for t in TARGETS:
            m = next(m for m in models if m["name"] == best and m["design"] == design and m["target"] == t)
            top = sorted(m["chosen"].items(), key=lambda kv: -kv[1])[:3]
            lines.append(f"* {design}, {t}: " + "; ".join(f"{k} ({v} folds)" for k, v in top))
    lines.append("")
    lines += ["## The GP variance against the error", "",
              "| design | target | 95% interval covers | rank correlation of predicted sd with error |",
              "|---|---|---:|---:|"]
    for v in out["variance"]:
        lines.append(f"| {v['design']} | {v['target']} | {v['coverage_95']:.0%} | {v['spearman_sd_vs_error']:+.2f} |")
    lines.append("")
    mt = out["multitask"]
    lines += [f"## The four endpoints together (kernel {mt['kernel']})", "",
              f"Multi-task fit with the task correlation chosen by cross-validation: rho {mt['rho']:g}, "
              f"ridge {mt['lam']:g}. Mean q² over the four endpoints {mt['mean_q2_shared']:+.3f} shared "
              f"against {mt['mean_q2_alone']:+.3f} with four separate fits.", "",
              "| endpoint | q² shared | q² alone |", "|---|---:|---:|"]
    for a, b in zip(mt["per_task_shared"], mt["per_task_alone"]):
        lines.append(f"| {a['task']} | {a['q2']:+.3f} | {b['q2']:+.3f} |")
    sel = out["selectivity"]
    lines += ["", f"The bifunctional difference dE_H - dE_OH (sd {sel['sd']:.2f} eV, {sel['n']} catalysts): "
              f"q² {sel['q2_shared']:+.3f} (RMSE {sel['rmse_shared']:.2f} eV) from the shared fit, "
              f"{sel['q2_single']:+.3f} (RMSE {sel['rmse_single']:.2f} eV) as the difference of the "
              f"two single-endpoint predictions.", ""]
    lines += ["## Overpotentials from the out-of-fold binding energies", "",
              f"Free energies with the corrections {out['free_energy_correction']} eV; eta_HER = |dG_H|, "
              f"eta_ORR = 1.23 V minus the limiting potential of the four-electron associative path. "
              f"The bifunctional score of a site is the larger of its two overpotentials.", "",
              "| design | HER mae, V | HER rank r | ORR mae, V | ORR rank r | bifunctional rank r | "
              f"best {TOP_K} recovered |", "|---|---:|---:|---:|---:|---:|---:|"]
    for design, e in out["overpotential"].items():
        lines.append(f"| {design} | {e['HER']['mae']:.2f} | {e['HER']['spearman']:+.2f} | "
                     f"{e['ORR']['mae']:.2f} | {e['ORR']['spearman']:+.2f} | "
                     f"{e['bifunctional']['spearman']:+.2f} | {e['bifunctional']['recovered']}/{TOP_K} |")
    e = out["overpotential"]["random"]["bifunctional"]
    lines += ["", f"Best {TOP_K} bifunctional sites by DFT: {', '.join(e['true_top'])}.",
              f"Best {TOP_K} by the random-fold predictions: {', '.join(e['predicted_top'])}.", ""]
    cof = out["cofield"]
    lines += [f"## The co-fields at named points ({cof['fieldset']} fields, rho {cof['rho']:g}, "
              f"alpha {cof['alpha']:g}, {'cosine' if cof['normalize'] else 'raw'} kernel)", "",
              "Value of the co-field per unit of field weight, in eV, averaged over the atoms of a "
              "shell (Fe@N4 as the reference geometry). A positive value says more of that field at "
              "that place raises the binding energy, that is weakens the binding.", "",
              "| field | place | H | OH | OOH | H - OH |", "|---|---|---:|---:|---:|---:|"]
    for fname, entry in cof["points"].items():
        for place in entry["H"]:
            lines.append(f"| {fname} | {place} | {entry['H'][place]:+.3f} | {entry['OH'][place]:+.3f} | "
                         f"{entry['OOH'][place]:+.3f} | {entry['H - OH'][place]:+.3f} |")
    lines += ["", "![co-fields](cofields.png)", "", "![cross-validation](cv_scatter.png)", "",
              f"Run time {out['seconds'] / 60:.1f} min."]
    text = "\n".join(lines) + "\n"
    io.open(os.path.join(results_dir, "table.md"), "w", encoding="utf-8").write(text)
    return text


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prepare", action="store_true", help="read the clone and write examples/sac")
    ap.add_argument("--clone", default=CLONE)
    ap.add_argument("--report", action="store_true", help="the table from the stored results")
    ap.add_argument("--bootstrap", type=int, default=RUNS)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    log_path = os.path.join(RESULTS, "run.log")
    log_fh = io.open(log_path, "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        log_fh.write(s + "\n")
        log_fh.flush()

    if args.prepare:
        prepare(args.clone, log)
        return
    if args.report:
        print(write_table())
        return
    log(f"== run_sac {time.strftime('%Y-%m-%d %H:%M')}")
    out = run(args, log)
    json.dump(out, io.open(os.path.join(RESULTS, "results.json"), "w", encoding="utf-8"), indent=1)
    print(write_table())


if __name__ == "__main__":
    main()
