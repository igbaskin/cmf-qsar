#!/usr/bin/env python
"""
The molecule that reacts first at an electrode: reduction and oxidation potentials of the species
of the LIBE data set (plan item BT13, part (a)).

The solid electrolyte interphase (SEI) starts with the reduction of a solvent, an anion or an
additive. The cathode electrolyte interphase (CEI) starts with an oxidation. The label of this
item is the potential of that first electron transfer, computed from two free energies of the same
species.

The data are the Lithium-Ion Battery Electrolyte set (LIBE; Spotte-Smith and co-workers,
Scientific Data 2021; figshare 10.6084/m9.figshare.14226464, licence CC BY 4.0). It holds 17,190
species. Every species has a geometry, free energies at one charge, atomic charges and atomic
spin populations. The set does not publish potentials, so this script forms them.

Stage ``series``
----------------
1. Read the set from ``benchmarks/cache/libe/libe.json``.
2. Give every species a graph key: its elements and its bonds, compared by a Weisfeiler-Lehman
   hash (a hash that depends only on the connectivity of the atoms, not on their numbering).
3. Pair species that have the same graph key and charges that differ by one. The reactant is a
   closed-shell species (spin multiplicity 1). Its partner is the radical ion of the same graph:
   one more electron for a reduction, one less for an oxidation. A reduction that breaks a bond
   gives a different graph and is not a pair. It is a decomposition product, and the set holds it
   as another species.
4. The potential is the free-energy difference in volts, on the scale of Li/Li+:

       reduction    E = -(G(charge - 1) - G(charge)) - 1.4 V
       oxidation    E =  (G(charge + 1) - G(charge)) - 1.4 V

   The 1.4 V is the difference between the absolute potential of the standard hydrogen electrode
   (4.44 V) and the potential of Li/Li+ against it (-3.04 V). It is one constant. It changes no
   rank and no regression slope, and it is stated because the plan asks that the reference is
   fixed and stated.
5. Where several entries have the same graph, charge and spin multiplicity, the one of lowest free
   energy is kept.
6. Every pair is given a chemical family, so that whole families can be held out together.

Stage ``xtb``
-------------
A GFN2 xtb calculation (ALPB solvent: acetone) on the geometry of every reactant gives the orbital
energies (HOMO and LUMO) and the Fukui functions. The set has no orbital energies. Species that
xtb cannot read are listed in ``benchmarks/cache/interphase/xtb_failed.json`` and are left out.

Stage ``models``
----------------
Models of the potential, each tested on species that it has not seen: the mean, the charge
state, the orbital energy (the reference of the item), the global quantum numbers, element
counts, and four sets of fields, each alone and joined to the orbital energy and charge state.
The kernel and the ridge constant are chosen on inner folds, inside every training set. Two
designs: whole chemical families held out in turn, and whole formulae held out (isomers of one
formula stay together). The result is stored in ``results/interphase/models_<kind>.json``.

Stage ``report``
----------------
Paired differences with a 95 % interval from resampling whole formulae, and the table.

Usage
-----
    python benchmarks/run_interphase.py --stage series
    python benchmarks/run_interphase.py --stage xtb
    python benchmarks/run_interphase.py --stage models --kind reduction
    python benchmarks/run_interphase.py --stage models --kind oxidation
    python benchmarks/run_interphase.py --stage report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

LIBE = os.path.join(HERE, "cache", "libe", "libe.json")
CACHE = os.path.join(HERE, "cache", "interphase")
RESULTS = os.path.join(HERE, "results", "interphase")

#: the free-energy column of the set that is used: it is stated in the table, and the two others
#: are kept as a sensitivity check
FREE_ENERGY = "quasi_rrho_eV"
OTHER_FREE_ENERGIES = ("shifted_rrho_eV", "eV")
#: absolute potential of the standard hydrogen electrode minus the potential of Li/Li+ against it
LI_REFERENCE_V = 4.44 - 3.04
#: the rounds of the Weisfeiler-Lehman hash
WL_ROUNDS = 4

#: the chemical families, in the order in which a species is given the first that fits
FAMILIES = ("phosphorus", "sulfur", "fluorinated", "carbonate", "other")


# --------------------------------------------------------------------------------------- graph key
def atom_labels(species: List[str], bonds: List[List[int]], rounds: int = WL_ROUNDS) -> List[str]:
    """One label per atom, from its element and the labels of its neighbours (Weisfeiler-Lehman).

    Two atoms that the connectivity cannot tell apart get the same label. The labels do not depend
    on the numbering of the atoms.
    """
    n = len(species)
    nbrs: List[List[int]] = [[] for _ in range(n)]
    for a, b in bonds:
        nbrs[a].append(b)
        nbrs[b].append(a)
    labels = list(species)
    for _ in range(rounds):
        labels = [hashlib.md5((labels[i] + "|" + ",".join(sorted(labels[j] for j in nbrs[i])))
                              .encode()).hexdigest()[:12] for i in range(n)]
    return labels


def graph_key(species: List[str], bonds: List[List[int]], rounds: int = WL_ROUNDS) -> str:
    """A hash of the elements and the bonds that does not depend on the numbering of the atoms.

    The key is the formula and a hash of the sorted atom labels. It is not a proof of identity,
    but a clash needs two different graphs with the same formula and the same neighbour structure
    after four rounds.
    """
    labels = atom_labels(species, bonds, rounds)
    formula = "".join(f"{e}{c}" for e, c in sorted(Counter(species).items()))
    return formula + ":" + hashlib.md5(",".join(sorted(labels)).encode()).hexdigest()[:16]


def family_of(species: List[str], bonds: List[List[int]]) -> str:
    """The chemical family of a species, for holding whole families out together.

    phosphorus   holds P (PF6- and its fragments)
    sulfur       holds S (the sulfonimide anions and their fragments)
    fluorinated  holds F and neither P nor S
    carbonate    a carbon that is bonded to three oxygen atoms (the carbonate group)
    other        the rest
    """
    present = set(species)
    if "P" in present:
        return "phosphorus"
    if "S" in present:
        return "sulfur"
    if "F" in present:
        return "fluorinated"
    n = len(species)
    nbrs: List[List[int]] = [[] for _ in range(n)]
    for a, b in bonds:
        nbrs[a].append(b)
        nbrs[b].append(a)
    for i, e in enumerate(species):
        if e == "C" and sum(1 for j in nbrs[i] if species[j] == "O") >= 3:
            return "carbonate"
    return "other"


# --------------------------------------------------------------------------------------- reading
def read_libe(path: str = LIBE) -> List[dict]:
    """The records of the set, reduced to the fields that this item reads."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for r in data:
        thermo = r["thermo"]
        energies = {}
        for name in (FREE_ENERGY,) + OTHER_FREE_ENERGIES:
            block = thermo.get(name)
            energies[name] = None if not block else block.get("free_energy")
        out.append({
            "id": r["molecule_id"],
            "charge": int(r["charge"]),
            "mult": int(r["spin_multiplicity"]),
            "species": list(r["species"]),
            "xyz": r["xyz"],
            "bonds": [list(map(int, b)) for b in r["bonds"]],
            "g": energies,
            "resp": (r.get("partial_charges") or {}).get("resp"),
            "mulliken": (r.get("partial_charges") or {}).get("mulliken"),
            "spin": (r.get("partial_spins") or {}).get("mulliken"),
        })
    del data
    return out


def potential(g_from: float, g_to: float, kind: str) -> float:
    """The potential in volts against Li/Li+ for one electron transferred. See the module text."""
    if kind == "reduction":
        return -(g_to - g_from) - LI_REFERENCE_V
    return (g_to - g_from) - LI_REFERENCE_V


def pairs(records: List[dict]) -> List[dict]:
    """The closed-shell species with a partner of the same graph one electron away."""
    best: Dict[Tuple[str, int, int], dict] = {}
    for r in records:
        g = r["g"].get(FREE_ENERGY)
        if g is None:
            continue
        r["key"] = graph_key(r["species"], r["bonds"])
        k = (r["key"], r["charge"], r["mult"])
        if k not in best or g < best[k]["g"][FREE_ENERGY]:
            best[k] = r
    out = []
    for (key, charge, mult), r in best.items():
        if mult != 1:
            continue
        fam = family_of(r["species"], r["bonds"])
        for kind, dq in (("reduction", -1), ("oxidation", +1)):
            partner = best.get((key, charge + dq, 2))
            if partner is None:
                continue
            row = {
                "kind": kind, "key": key, "family": fam, "charge": charge,
                "has_li": "Li" in r["species"], "id": r["id"], "partner": partner["id"],
                "formula": key.split(":")[0], "n_atoms": len(r["species"]),
                "E": potential(r["g"][FREE_ENERGY], partner["g"][FREE_ENERGY], kind),
            }
            for other in OTHER_FREE_ENERGIES:
                a, b = r["g"].get(other), partner["g"].get(other)
                row["E_" + other] = None if a is None or b is None else potential(a, b, kind)
            out.append(row)
    return out


# ---------------------------------------------------------------------------------------- stages
def series_stage() -> dict:
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    records = read_libe()
    prs = pairs(records)
    by_id = {r["id"]: r for r in records}
    # keep the geometry and the atomic numbers of the pairs in the cache, so that later stages do
    # not read the large file again
    keep = {}
    for p in prs:
        for i in (p["id"], p["partner"]):
            if i not in keep:
                r = by_id[i]
                keep[i] = {k: r[k] for k in ("charge", "mult", "species", "xyz", "bonds",
                                             "resp", "mulliken", "spin")}
    with open(os.path.join(CACHE, "pairs.json"), "w") as fh:
        json.dump({"pairs": prs, "species": keep}, fh)
    summary = {"n_records": len(records), "n_with_energy": sum(
        1 for r in records if r["g"].get(FREE_ENERGY) is not None)}
    for kind in ("reduction", "oxidation"):
        sel = [p for p in prs if p["kind"] == kind]
        summary[kind] = {
            "n": len(sel),
            "families": dict(Counter(p["family"] for p in sel)),
            "with_li": sum(p["has_li"] for p in sel),
            "charges": dict(sorted(Counter(p["charge"] for p in sel).items())),
            "E_range": [min(p["E"] for p in sel), max(p["E"] for p in sel)] if sel else None,
            "n_atoms_max": max(p["n_atoms"] for p in sel) if sel else None,
        }
    with open(os.path.join(RESULTS, "series.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


#: the solvent of the quantum stage: the set uses a continuum of permittivity 18.5, and the nearest
#: ALPB solvent of xtb is acetone (permittivity 20.7)
XTB_SOLVENT = "acetone"
XTB_JOBS = os.path.join(CACHE, "xtb_jobs")
XTB_FILE = os.path.join(CACHE, "xtb.json")


def _write_xyz(path: str, species: List[str], xyz: List[List[float]]) -> None:
    with open(path, "w") as fh:
        fh.write(f"{len(species)}\n\n")
        for e, (x, y, z) in zip(species, xyz):
            fh.write(f"{e:2s} {x:.8f} {y:.8f} {z:.8f}\n")


def _xtb_one(args: Tuple[str, dict]) -> Tuple[str, Optional[dict], str]:
    """xtb on one reactant: the orbital energies, the charges and the Fukui functions."""
    from cmfqsar import xtb as QX
    sid, rec = args
    cfg = QX.Config(solvent=XTB_SOLVENT, fukui=True, timeout=300.0)
    job = os.path.join(XTB_JOBS, sid)
    os.makedirs(job, exist_ok=True)
    geom = os.path.join(job, "geom.xyz")
    if not os.path.isfile(geom):
        _write_xyz(geom, rec["species"], rec["xyz"])
    try:
        res = QX.run_job(job, cfg, rec["charge"], 1)
    except Exception as exc:                                     # noqa: BLE001
        return sid, None, str(exc)[:200]
    atom, mol = QX.derive_properties(res, rec["charge"], 1)
    out = {k: float(v) for k, v in mol.items()}
    out.update({k: np.asarray(v, float).tolist() for k, v in atom.items()
                if k in ("q_xtb", "fukui_plus", "fukui_minus", "dual")})
    return sid, out, ""


def xtb_stage(workers: int = 12, limit: Optional[int] = None) -> dict:
    """GFN2-xTB single points of every reactant at the geometry of the set.

    The reference of the plan is a regression on the energy of the lowest empty orbital (for a
    reduction) or the highest filled orbital (for an oxidation), and the set does not hold
    orbital energies. The same runs give the Fukui functions and the charges of the reactant,
    which are the per-atom fields of the models. The runs are resumable: a finished job is read
    again and not run again.
    """
    from concurrent.futures import ThreadPoolExecutor
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with open(os.path.join(CACHE, "pairs.json")) as fh:
        data = json.load(fh)
    ids = sorted({p["id"] for p in data["pairs"]})
    if limit:
        ids = ids[:limit]
    done: Dict[str, dict] = {}
    if os.path.isfile(XTB_FILE):
        with open(XTB_FILE) as fh:
            done = json.load(fh)
    todo = [(i, data["species"][i]) for i in ids if i not in done]
    failed: Dict[str, str] = {}
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sid, out, err in pool.map(_xtb_one, todo):
            n += 1
            if out is None:
                failed[sid] = err
            else:
                done[sid] = out
            if n % 200 == 0:
                print(f"{n}/{len(todo)} done, {len(failed)} failed", flush=True)
                with open(XTB_FILE, "w") as fh:
                    json.dump(done, fh)
    with open(XTB_FILE, "w") as fh:
        json.dump(done, fh)
    with open(os.path.join(CACHE, "xtb_failed.json"), "w") as fh:
        json.dump(failed, fh, indent=1)
    return {"requested": len(ids), "finished": len(done), "failed": len(failed)}


# ------------------------------------------------------------------------------------- the models
ELEMENTS = ("C", "H", "O", "F", "P", "S", "N", "Li")
#: columns of the per-atom table of a species
COLUMNS = ("resp", "steric") + tuple("el_" + e for e in ELEMENTS) + ("fukui", "spin")
#: the field sets: the columns that each one carries
FIELD_SETS = {
    "fields": ("resp", "steric") + tuple("el_" + e for e in ELEMENTS),
    "fields + Fukui": ("resp", "steric") + tuple("el_" + e for e in ELEMENTS) + ("fukui",),
    "fields + spin": ("resp", "steric") + tuple("el_" + e for e in ELEMENTS) + ("spin",),
    "fields + Fukui + spin": COLUMNS,
}
#: the two grids of the distance channels (d_max, step = width of the smoothing), in angstrom
SPECS = {"fine": (8.0, 0.5), "coarse": (8.0, 1.0)}
#: the models that need no geometry
REFERENCES = ("mean", "charge state", "orbital energy", "orbital energy + charge state",
              "global quantum numbers", "element counts")
MIX = 0.5
LAMS = tuple(10.0 ** e for e in (-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0))
FOLDS, INNER = 5, 3
GROUP_REPEATS = (0, 1, 2)
#: the orbital that decides the first step: the lowest empty one takes the electron of a
#: reduction, the highest filled one gives the electron of an oxidation
ORBITAL = {"reduction": "E_lumo", "oxidation": "E_homo"}
FUKUI = {"reduction": "fukui_plus", "oxidation": "fukui_minus"}


def _load_series() -> Tuple[List[dict], Dict[str, dict], Dict[str, dict]]:
    with open(os.path.join(CACHE, "pairs.json")) as fh:
        data = json.load(fh)
    with open(XTB_FILE) as fh:
        xtb = json.load(fh)
    return data["pairs"], data["species"], xtb


def kind_rows(kind: str) -> List[dict]:
    """The pairs of one kind that have an xtb result for the reactant."""
    prs, _, xtb = _load_series()
    return [p for p in prs if p["kind"] == kind and p["id"] in xtb]


def species_table(row: dict, species: Dict[str, dict], xtb: Dict[str, dict], kind: str
                  ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(coordinates, per-atom columns, element symbols) of the reactant of one pair.

    ``resp`` is the RESP charge of the reactant. ``steric`` is the van der Waals well depth of the
    element. ``el_<symbol>`` is one for the atoms of that element. ``fukui`` is the Fukui
    function of xtb that belongs to the step (f+ to add an electron, f- to remove one). ``spin``
    is the Mulliken spin population of the radical ion that the step gives, which says where
    the electron went or where it was taken from. The radical ion is numbered in its own way, so
    its population is averaged over the atoms that the connectivity cannot tell apart, and every
    atom of the reactant takes the average of its class.
    """
    from cmfqsar.atomtypes import vdw_params
    r, q = species[row["id"]], species[row["partner"]]
    sym = r["species"]
    n = len(sym)
    cols = np.zeros((n, len(COLUMNS)))
    cols[:, 0] = r["resp"]
    cols[:, 1] = [vdw_params(s)[1] for s in sym]
    for k, e in enumerate(ELEMENTS):
        cols[:, 2 + k] = [1.0 if s == e else 0.0 for s in sym]
    cols[:, COLUMNS.index("fukui")] = xtb[row["id"]][FUKUI[kind]]
    lab_r = atom_labels(sym, r["bonds"])
    lab_q = atom_labels(q["species"], q["bonds"])
    sums: Dict[str, List[float]] = defaultdict(list)
    for lab, v in zip(lab_q, q["spin"]):
        sums[lab].append(v)
    mean = {lab: float(np.mean(v)) for lab, v in sums.items()}
    cols[:, COLUMNS.index("spin")] = [mean.get(lab, 0.0) for lab in lab_r]
    return np.asarray(r["xyz"], float), cols, sym


def feature_rows(kind: str, rows: List[dict]) -> Dict[str, Dict[str, np.ndarray]]:
    """The autocorrelation features of every field set on every grid: {set: {grid: matrix}}."""
    from cmfqsar.fields import MoleculeFields
    from cmfqsar import invariant as INV
    _, species, xtb = _load_series()
    tables = [species_table(r, species, xtb, kind) for r in rows]
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, cols in FIELD_SETS.items():
        idx = [COLUMNS.index(c) for c in cols]
        mols = [MoleculeFields(name=r["id"], coords=xyz, weights=W[:, idx],
                               radii=np.ones(len(sym)), symbols=list(sym),
                               atom_types=list(sym), bonds=np.zeros((0, 2), int),
                               charges=W[:, 0], charge_source="file")
                for r, (xyz, W, sym) in zip(rows, tables)]
        out[name] = {}
        for grid, (d_max, step) in SPECS.items():
            spec = INV.AutocorrelationSpec(d_max=d_max, step=step, sigma=step)
            out[name][grid] = INV.feature_matrix(mols, spec)
    return out


def descriptor_table(kind: str, rows: List[dict]) -> Dict[str, np.ndarray]:
    """The tables of the models without geometry."""
    _, species, xtb = _load_series()
    q = np.array([r["charge"] for r in rows], float)
    onehot = np.stack([(q == c).astype(float) for c in sorted(set(q))], axis=1)
    li = np.array([1.0 if r["has_li"] else 0.0 for r in rows])[:, None]
    orb = np.array([xtb[r["id"]][ORBITAL[kind]] for r in rows])[:, None]
    glob = np.array([[xtb[r["id"]][k] for k in ("E_homo", "E_lumo", "gap", "dipole")]
                     for r in rows])
    counts = np.array([[species[r["id"]]["species"].count(e) for e in ELEMENTS] for r in rows],
                      float)
    return {"charge state": np.hstack([onehot, li]),
            "orbital energy": orb,
            "orbital energy + charge state": np.hstack([orb, onehot, li]),
            "global quantum numbers": np.hstack([glob, onehot, li]),
            "element counts": np.hstack([counts, onehot, li])}


def build_kernels(kind: str, rows: List[dict]) -> Dict[str, Dict[str, np.ndarray]]:
    """{model: {kernel name: kernel matrix}} for every model of the study."""
    import run_cellmodel as RC
    from cmfqsar import invariant as INV
    desc = descriptor_table(kind, rows)
    kernels: Dict[str, Dict[str, np.ndarray]] = {}
    n = len(rows)
    kernels["mean"] = {"constant": np.zeros((n, n))}
    for name, Z in desc.items():
        kernels[name] = RC.descriptor_kernels(Z)
    line = RC.descriptor_kernels(desc["orbital energy + charge state"])["linear"]
    feats = feature_rows(kind, rows)
    for name, grids in feats.items():
        alone, plus = {}, {}
        for grid, X in grids.items():
            cosine = INV.autocorrelation_kernel(X, n_bins=1)
            plain = RC.unit_kernel(INV.autocorrelation_kernel(X, n_bins=1, normalize=False))
            for kn, K in (("cosine", cosine), ("not normalised", plain)):
                alone[f"{grid}, {kn}"] = K
                plus[f"{grid}, {kn}"] = RC.unit_kernel(MIX * RC.unit_kernel(K)
                                                       + (1.0 - MIX) * line)
        kernels[name] = alone
        kernels[name + " + orbital energy and charge state"] = plus
    return kernels


def nested_ridge(kernels: Dict[str, np.ndarray], y: np.ndarray, groups: List[str], folds: int,
                 inner: int, seed: int) -> Dict[str, object]:
    """Predictions for every row from models fitted without its group, the kernel and the ridge
    chosen on inner folds of the training part. The eigen-decomposition of every inner training
    block is made once and used for every ridge, so that the grid is cheap."""
    import run_cellmodel as RC
    n = len(y)
    pred = np.full(n, np.nan)
    chosen = []
    for te in RC.group_parts(groups, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        parts = [tr[p] for p in RC.group_parts([groups[i] for i in tr], inner, seed + 1)]
        best = None
        for name, K in kernels.items():
            sse = np.zeros(len(LAMS))
            cnt = 0
            for ip in parts:
                itr = np.setdiff1d(tr, ip)
                if len(itr) < 3:
                    continue
                w, V = np.linalg.eigh(K[np.ix_(itr, itr)])
                w = np.maximum(w, 0.0)
                m = float(np.mean(y[itr]))
                c = V.T @ (y[itr] - m)
                Kt = K[np.ix_(ip, itr)] @ V
                for j, lam in enumerate(LAMS):
                    p = m + Kt @ (c / (w + lam))
                    sse[j] += float(np.sum((y[ip] - p) ** 2))
                cnt += len(ip)
            if cnt == 0:
                continue
            j = int(np.argmin(sse))
            if best is None or sse[j] / cnt < best[0]:
                best = (sse[j] / cnt, name, LAMS[j])
        _, name, lam = best
        K = kernels[name]
        w, V = np.linalg.eigh(K[np.ix_(tr, tr)])
        w = np.maximum(w, 0.0)
        m = float(np.mean(y[tr]))
        c = V.T @ (y[tr] - m)
        pred[te] = m + (K[np.ix_(te, tr)] @ V) @ (c / (w + lam))
        chosen.append({"kernel": name, "lam": float(lam), "n_test": int(len(te))})
    return {"predicted": pred, "chosen": chosen}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def models_stage(kind: str, designs: Tuple[str, ...] = ("families", "formulae")) -> dict:
    import time
    rows = kind_rows(kind)
    y = np.array([r["E"] for r in rows])
    fam = [r["family"] for r in rows]
    form = [r["formula"] for r in rows]
    print(f"{kind}: {len(rows)} species", flush=True)
    kernels = build_kernels(kind, rows)
    out: dict = {"kind": kind, "n": len(rows), "ids": [r["id"] for r in rows],
                 "families": fam, "formulae": form, "y": y.tolist(), "designs": {}}
    for design in designs:
        res: Dict[str, dict] = {}
        for model, ks in kernels.items():
            t0 = time.time()
            errs, preds, chosen = [], [], []
            seeds = (0,) if design == "families" else GROUP_REPEATS
            for seed in seeds:
                if design == "families":
                    fit = nested_ridge(ks, y, fam, folds=len(set(fam)), inner=len(set(fam)) - 1,
                                       seed=seed)
                else:
                    fit = nested_ridge(ks, y, form, folds=FOLDS, inner=INNER, seed=seed)
                preds.append(fit["predicted"])
                errs.append(np.abs(fit["predicted"] - y))
                chosen.extend(c["kernel"] for c in fit["chosen"])
            err = np.mean(errs, axis=0)
            pred = np.mean(preds, axis=0)
            res[model] = {"abs_errors": err.tolist(), "mae": float(err.mean()),
                          "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
                          "rank": spearman(pred, y),
                          "kernels_chosen": {k: chosen.count(k) for k in sorted(set(chosen))}}
            print(f"  {design:<9} {model:<52} MAE {err.mean():.3f} V  rank "
                  f"{res[model]['rank']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        out["designs"][design] = res
    with open(os.path.join(RESULTS, f"models_{kind}.json"), "w") as fh:
        json.dump(out, fh)
    return out


# ------------------------------------------------------------------------------------- the report
#: the comparisons that are read: (name, reference model, model that is tested). The first is the
#: one of the gate. A positive difference means that the tested model has the smaller error.
GATE_REFERENCE = "orbital energy + charge state"
GATE_MODEL = "fields + Fukui + spin"
COMPARISONS = (
    ("gate: the fields against the orbital energy", GATE_REFERENCE, GATE_MODEL),
    ("the fields from the reactant alone against the orbital energy", GATE_REFERENCE,
     "fields + Fukui"),
    ("the fields against the global quantum numbers", "global quantum numbers", GATE_MODEL),
    ("the fields added to the orbital energy", GATE_REFERENCE,
     GATE_MODEL + " + orbital energy and charge state"),
    ("the spin population added to the fields", "fields", "fields + spin"),
    ("the Fukui function added to the fields", "fields", "fields + Fukui"),
    ("the fields against the element counts", "element counts", "fields"),
)


def read_models(kind: str) -> dict:
    with open(os.path.join(RESULTS, f"models_{kind}.json")) as fh:
        return json.load(fh)


def comparisons(data: dict, design: str, draws: int = 2000) -> List[dict]:
    import run_labelmodel as RL
    res = data["designs"][design]
    out = []
    for name, ref, model in COMPARISONS:
        iv = RL.paired_group_interval(res[ref]["abs_errors"], res[model]["abs_errors"],
                                      data["formulae"], seed=0, draws=draws)
        out.append({"name": name, "reference": ref, "model": model, **iv})
    return out


def by_family(data: dict, design: str, models: Tuple[str, ...]) -> Dict[str, Dict[str, float]]:
    res = data["designs"][design]
    fam = np.array(data["families"])
    return {f: {m: float(np.mean(np.array(res[m]["abs_errors"])[fam == f])) for m in models}
            for f in sorted(set(fam))}


def report_stage() -> dict:
    out = {}
    for kind in ("reduction", "oxidation"):
        if not os.path.isfile(os.path.join(RESULTS, f"models_{kind}.json")):
            continue
        data = read_models(kind)
        out[kind] = {d: {"comparisons": comparisons(data, d),
                         "gate_holds": None} for d in data["designs"]}
        for d, v in out[kind].items():
            gate = v["comparisons"][0]
            v["gate_holds"] = bool(gate["low"] > 0)
    with open(os.path.join(RESULTS, "report.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def _v(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def write_table() -> str:
    """The table of results in plain words."""
    rep = report_stage()
    lines = ["# Part (a) of item BT13: the first electron transfer of the electrolyte species", ""]
    for kind in ("reduction", "oxidation"):
        if kind not in rep:
            continue
        data = read_models(kind)
        lines += [f"## {kind.capitalize()}: {data['n']} species", "",
                  "Mean absolute error of the potential in volts, on species that the model has "
                  "not seen. In the design 'whole families held out' every chemical family is "
                  "left out in turn. In the design 'whole formulae held out' the isomers of one "
                  "formula are left out together, five folds, three repeats.", "",
                  "| Model | Whole families held out, error in V | Whole formulae held out, "
                  "error in V | Rank correlation, formulae |", "|---|---:|---:|---:|"]
        fam, frm = data["designs"]["families"], data["designs"]["formulae"]
        for m in fam:
            lines.append(f"| {m} | {_v(fam[m]['mae'])} | {_v(frm[m]['mae'])} | "
                         f"{_v(frm[m]['rank'])} |")
        lines.append("")
        for d, title in (("families", "whole families held out"),
                         ("formulae", "whole formulae held out")):
            lines += [f"Differences, {title}. A positive difference means that the tested model "
                      "has the smaller error. The interval is 95 %, from resampling whole "
                      "formulae.", "",
                      "| Comparison | Reference model | Tested model | Difference, V | "
                      "Interval, V | Interval excludes zero |", "|---|---|---|---:|---|---|"]
            for c in rep[kind][d]["comparisons"]:
                lines.append(f"| {c['name']} | {c['reference']} | {c['model']} | {_v(c['mean'])} "
                             f"| {_v(c['low'])} to {_v(c['high'])} | "
                             f"{'yes' if c['excludes_zero'] else 'no'} |")
            lines.append("")
        fams = by_family(data, "families", (GATE_REFERENCE, GATE_MODEL))
        lines += ["Error of the gate model and of its reference in every family held out, in V.", "",
                  "| Family held out | Orbital energy and charge state | Fields, Fukui and spin |",
                  "|---|---:|---:|"]
        for f, v in fams.items():
            lines.append(f"| {f} | {_v(v[GATE_REFERENCE])} | {_v(v[GATE_MODEL])} |")
        lines.append("")
    path = os.path.join(RESULTS, "table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="series", choices=["series", "xtb", "models", "report"])
    ap.add_argument("--kind", default="reduction", choices=["reduction", "oxidation"])
    ap.add_argument("--designs", default="families,formulae")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.stage == "series":
        print(json.dumps(series_stage(), indent=1))
    elif args.stage == "xtb":
        print(json.dumps(xtb_stage(args.workers, args.limit), indent=1))
    elif args.stage == "models":
        models_stage(args.kind, tuple(args.designs.split(",")))
    elif args.stage == "report":
        print(write_table())


if __name__ == "__main__":
    main()
