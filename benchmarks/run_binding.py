#!/usr/bin/env python
"""
Binding, not folding: the frame of a residue at a protein interface (plan item PP6).

PP1 to PP5 measured the change of folding stability of point mutants.  The same frame and the
same descriptions of a site apply to a mutation at an interface, where the target is the change
of binding free energy and the surroundings that matter are the partner protein.  SKEMPI 2.0
(Jankauskaite et al., Bioinformatics 35 (2019) 462; CC BY 4.0) is the public collection.

The samples
-----------
Every single substitution of SKEMPI 2.0 whose two affinities are numbers (not a bound such as
"> 1E-3").  The target is ddG = R T ln(Kd of the mutant / Kd of the wild type) in kcal/mol, at
the temperature of the measurement (298 K where none is given); a positive value means weaker
binding.  Repeated measurements of one substitution are averaged.  The structure is the deposited
complex from the PDB, with only the chains SKEMPI names, completed by
:func:`cmfqsar.mutants.prepare_complex`; residues are found by SKEMPI's PDB numbering.

The descriptions, every one joined with the three substitution columns (the change of volume,
hydropathy and charge)
-----------------------------------------------------------------------------------------
* **three columns kernel** -- the substitution alone;
* **one count** -- the number of atoms of the complex around the site (PP1);
* **two counts** -- the same, and the number of those atoms that belong to the partner;
* **interface class** -- SKEMPI's own class of the site (core, rim, support, surface,
  interior; Levy, J. Mol. Biol. 403 (2010) 660), as five one-hot columns;
* **accessible area** -- the solvent-accessible area of the residue and of its side chain in
  the complex (PP3);
* **buried area** -- the same, and the area of each that binding buries: the area in the
  partner's absence minus the area in the complex;
* **field of the surroundings** -- PP1's context kernel over the complex (envelope rho 0.1,
  cosine), joined with the column kernel at the weights 0.25 and 0.5, as PP2;
* **field of the partner** -- the same context with only the partner's atoms, joined with the
  two counts.  Two forms are offered and the inner folds choose: the cosine kernel, which
  divides out how much of the partner there is, and the raw kernel, which keeps it;
* **buried area and field of the partner**, **buried area and field of the surroundings** --
  the field joined with the buried area instead.

Folds: whole complexes, and whole families of complexes (complexes SKEMPI lists as related, or
that share a protein name), eight folds, three repeats; every choice (member of a family,
ridge) is made in inner folds of the same kind.  The plan's question is whether a field of the
site beats a count and the buried area.

Usage
-----
    python benchmarks/run_binding.py --fetch                 # the table and the PDB entries
    python benchmarks/run_binding.py --prepare --workers 8   # the complexes, about 10 minutes
    python benchmarks/run_binding.py                         # the measurement
    python benchmarks/run_binding.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_mutants as M  # noqa: E402
from cmfqsar import mutants as MU  # noqa: E402
from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, MoleculeFields  # noqa: E402
from cmfqsar.kernels import kernel_matrices, normalize_kernels, self_kernels  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "skempi")
SKEMPI_CSV = os.path.join(CACHE, "raw", "skempi_v2.csv")
#: sha256 of skempi_v2.csv as it was fetched (Zenodo record 23716666, a copy of SKEMPI 2.0)
SKEMPI_SHA256 = "76f8e60d09eaa4f05fb265063886859a86895c0b2fb225e31fe82928a883972d"
PDB_DIR = os.path.join(CACHE, "pdb")
PREPARED = os.path.join(CACHE, "prepared")
RESULTS = os.path.join(ROOT, "benchmarks", "results", "binding")
#: the gas constant in kcal/(mol K)
R_GAS = 1.98720425864083e-3
DEFAULT_T = 298.0
RHO = M.SITE_RHO
ALPHA = M.CONTEXT_ALPHA
FOLDS = 8
HOLDOUTS = ("complex", "family")
LOCATIONS = ("COR", "RIM", "SUP", "SUR", "INT")
#: complexes with at least this many substitutions get a correlation of their own
PER_COMPLEX_MIN = 15
MAIN_CHAIN = {"N", "CA", "C", "O", "H", "HA", "H1", "H2", "H3", "OXT", "HA2", "HA3"}

_MUTATION = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+[a-zA-Z]?)([A-Z])$")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def parse_temperature(text: str) -> float:
    """The temperature of a SKEMPI row in kelvin: '298', '298(assumed)', or empty for 298."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)", text or "")
    return float(m.group(1)) if m else DEFAULT_T


def parse_mutation(text: str) -> Tuple[str, str, str, str]:
    """(wild residue, chain, residue number as deposited, new residue) of SKEMPI's PDB-numbered
    form, such as 'LI45G' or 'RD100bA' (residue 100B of chain D)."""
    m = _MUTATION.match(text.strip())
    if not m:
        raise ValueError(f"{text!r} is not a single substitution written as LI45G")
    wild, chain, resid, new = m.groups()
    return wild, chain, resid.upper(), new


def read_skempi(path: str = SKEMPI_CSV) -> Tuple[List[dict], Counter]:
    """Every single substitution with two numeric affinities, one row per measurement."""
    out, skipped = [], Counter()
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if "," in r["Mutation(s)_PDB"]:
                continue
            try:
                k_mut, k_wt = float(r["Affinity_mut (M)"]), float(r["Affinity_wt (M)"])
            except ValueError:
                skipped["an affinity is a bound or missing"] += 1
                continue
            if k_mut <= 0 or k_wt <= 0:
                skipped["an affinity is zero"] += 1
                continue
            wild, chain, resid, new = parse_mutation(r["Mutation(s)_PDB"])
            if wild == new or wild not in MU.AA3 or new not in MU.AA3:
                skipped["not a substitution of two standard residues"] += 1
                continue
            t = parse_temperature(r["Temperature"])
            out.append({"complex": r["#Pdb"], "chain": chain, "resid": resid, "wild": wild,
                        "new": new, "ddg": R_GAS * t * math.log(k_mut / k_wt),
                        "location": r["iMutation_Location(s)"],
                        "related": r["Hold_out_proteins"],
                        "proteins": (r["Protein 1"], r["Protein 2"])})
    return out, skipped


def complex_chains(name: str) -> Tuple[str, str, str]:
    """(PDB code, chains of the first partner, chains of the second) of '1DAN_HL_UT'."""
    code, a, b = name.split("_")
    return code, a, b


def substitutions(rows: Sequence[dict]) -> List[dict]:
    """One row per distinct substitution of a complex, the measured ddG averaged."""
    by: Dict[tuple, List[dict]] = {}
    for r in rows:
        by.setdefault((r["complex"], r["chain"], r["resid"], r["wild"], r["new"]), []).append(r)
    out = []
    for key in sorted(by):
        group = by[key]
        out.append(dict(group[0], ddg=float(np.mean([g["ddg"] for g in group])),
                        measurements=len(group)))
    return out


def _norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def families(rows: Sequence[dict]) -> Dict[str, str]:
    """Complex -> family: complexes joined when SKEMPI lists one as related to the other
    (its column Hold_out_proteins), or when they share a protein name.  The families are the
    strictest hold-out: a model never sees the same protein, or a close relative, with
    another partner."""
    parent: Dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    first: Dict[str, str] = {}
    for r in rows:
        find(r["complex"])
        for h in r["related"].split(","):
            if "_" in h:
                union(r["complex"], h.strip())
        for p in r["proteins"]:
            n = _norm_name(p)
            if n:
                if n in first:
                    union(r["complex"], first[n])
                else:
                    first[n] = r["complex"]
    names = sorted({r["complex"] for r in rows})
    root = {c: find(c) for c in names}
    label = {}
    for c in names:                              # name a family by its first complex
        label.setdefault(root[c], c)
    return {c: label[root[c]] for c in names}


SKEMPI_URL = "https://zenodo.org/api/records/23716666/files/skempi_v2.zip/content"


def fetch() -> None:
    """Download the SKEMPI 2.0 table (a CC BY 4.0 copy in a Zenodo record, since the SKEMPI
    site does not always answer) and the PDB entries of its complexes (from RCSB)."""
    import hashlib
    import io
    import urllib.request
    import zipfile
    from cmfqsar.protein import fetch_pdb
    os.makedirs(os.path.dirname(SKEMPI_CSV), exist_ok=True)
    if not os.path.exists(SKEMPI_CSV):
        with urllib.request.urlopen(SKEMPI_URL, timeout=120) as r:  # noqa: S310 - fixed host
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            text = z.read("skempi_v2.csv")
        with open(SKEMPI_CSV, "wb") as fh:
            fh.write(text)
    with open(SKEMPI_CSV, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    if got != SKEMPI_SHA256:
        raise SystemExit(f"{SKEMPI_CSV} has sha256 {got}, not the {SKEMPI_SHA256} this "
                         "benchmark was measured on")
    codes = sorted({complex_chains(r["complex"])[0] for r in read_skempi()[0]})
    failed = []
    for code in codes:
        try:
            fetch_pdb(code, PDB_DIR)
        except RuntimeError as exc:
            failed.append(str(exc))
    log(f"{len(codes) - len(failed)} of {len(codes)} PDB entries in {PDB_DIR}")
    if failed:
        raise SystemExit("\n".join(failed))


# --------------------------------------------------------------------------- #
# The complexes
# --------------------------------------------------------------------------- #
def _paths(name: str) -> Dict[str, str]:
    base = os.path.join(PREPARED, name)
    return {"pdb": base + ".pdb", "structure": base + ".npz", "sites": base + "_sites.json",
            "contexts": base + "_contexts.npz"}


def _pack(m: MoleculeFields, prefix: str) -> Dict[str, np.ndarray]:
    return {prefix + k: v for k, v in M._pack(m).items()}


def prepare_one(name: str, sites: Sequence[Tuple[str, str]]) -> dict:
    """Complete one complex and write, for every site of `sites` ((chain, residue as
    deposited)), its number in the prepared file, its areas, its two counts and its two
    contexts (the whole complex and the partner alone)."""
    import mdtraj as md
    from cmfqsar.fields import build_fields
    os.makedirs(PREPARED, exist_ok=True)
    p = _paths(name)
    code, a, b = complex_chains(name)
    st, numbering = MU.prepare_complex(os.path.join(PDB_DIR, code + ".pdb"), p["pdb"], a + b)
    st.save(p["structure"])
    mol = MU.protein_of(st)
    chain_of = np.array([at.GetPDBResidueInfo().GetChainId().strip() for at in mol.GetAtoms()])
    fields = build_fields(mol, FieldSet.create(charge_model="file"), name=name)
    t = md.load(p["pdb"])
    area = md.shrake_rupley(t, probe_radius=0.14, mode="atom")[0] * 100.0
    by_res = {int(r.resSeq): r for r in t.topology.residues}
    side_of = {c: (a if c in a else b) for c in a + b}
    unbound_area = {}
    for group in (a, b):
        idx = [x.index for x in t.topology.atoms if x.residue.chain.chain_id in group]
        if not idx:
            continue
        alone = md.shrake_rupley(t.atom_slice(idx), probe_radius=0.14, mode="atom")[0] * 100.0
        for k, i in enumerate(idx):
            unbound_area[i] = float(alone[k])
    packed, out = {}, {}
    for chain, resid in sites:
        key = f"{chain}{resid}"
        number = numbering.get((chain, resid))
        if number is None:
            out[key] = {"error": "residue not in the structure"}
            continue
        res = by_res[number]
        one = MU.AA1.get(res.name, "X")
        own = side_of[chain]
        partner = b if own == a else a
        atoms = list(res.atoms)
        side = [x.index for x in atoms if x.name not in MAIN_CHAIN]
        whole = MU.site_context(fields, mol, number, float(RHO), name=f"{name}:{key}")
        src = np.asarray(whole.member_atom_map, int)
        keep = np.array([k for k, s in enumerate(src) if chain_of[s] in partner], int)
        part = MU._subset(whole, keep)
        out[key] = {
            "number": number, "residue": one,
            "area": [float(sum(area[x.index] for x in atoms)), float(sum(area[i] for i in side))],
            "unbound": [float(sum(unbound_area[x.index] for x in atoms)),
                        float(sum(unbound_area[i] for i in side))],
            "count": int(whole.n_atoms), "partner_count": int(part.n_atoms)}
        packed.update(_pack(whole, f"{key}|whole|"))
        packed.update(_pack(part, f"{key}|partner|"))
    np.savez_compressed(p["contexts"], **packed)
    with open(p["sites"], "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    return out


def prepare_share(shard: int, shards: int) -> None:
    rows = substitutions(read_skempi()[0])
    sites: Dict[str, set] = {}
    for r in rows:
        sites.setdefault(r["complex"], set()).add((r["chain"], r["resid"]))
    names = sorted(sites)[shard::shards]
    os.makedirs(PREPARED, exist_ok=True)
    failed, t0 = [], time.time()
    for k, name in enumerate(names, 1):
        if os.path.exists(_paths(name)["sites"]):
            continue
        try:
            prepare_one(name, sorted(sites[name]))
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
        if k % 10 == 0 or k == len(names):
            log(f"  share {shard}: {k}/{len(names)} after {time.time() - t0:.0f} s, "
                f"{len(failed)} failed")
    if failed:
        with open(os.path.join(PREPARED, f"failed_{shard}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(failed) + "\n")


def prepare(workers: int = 8) -> None:
    env = dict(os.environ, OPENMM_CPU_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    procs = [subprocess.Popen([sys.executable, "-P", os.path.abspath(__file__),
                               "--prepare-shard", str(k), "--shards", str(workers)], env=env)
             for k in range(workers)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"preparing: exit codes {codes}")


# --------------------------------------------------------------------------- #
# The samples
# --------------------------------------------------------------------------- #
class BindingSamples:
    """Every substitution whose site was prepared, in one row order, with what the kernels
    need: the substitution, the site's numbers and its two contexts."""

    def __init__(self, rows: Optional[Sequence[dict]] = None):
        raw, self.skipped = read_skempi()
        rows = substitutions(raw) if rows is None else rows
        family = families(raw)
        self.complex, self.family, self.site, self.location = [], [], [], []
        self.mutation, self.y, self.measurements = [], [], []
        self.info: Dict[Tuple[str, str], dict] = {}
        self.contexts: Dict[Tuple[str, str, str], MoleculeFields] = {}
        self.missing = Counter()
        loaded: Dict[str, dict] = {}
        for r in rows:
            name = r["complex"]
            if name not in loaded:
                path = _paths(name)["sites"]
                loaded[name] = {}
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as fh:
                        loaded[name] = json.load(fh)
            key = f"{r['chain']}{r['resid']}"
            site = loaded[name].get(key)
            if not loaded[name]:
                self.missing["complex not prepared"] += 1
                continue
            if site is None or "error" in site:
                self.missing["residue not in the structure"] += 1
                continue
            if site["residue"] != r["wild"]:
                self.missing["residue in the structure differs from SKEMPI"] += 1
                continue
            self.info[(name, key)] = site
            self.complex.append(name)
            self.family.append(family[name])
            self.site.append(key)
            self.location.append(r["location"])
            self.mutation.append(MU.Mutation(r["wild"], int(site["number"]), r["new"]))
            self.y.append(float(r["ddg"]))
            self.measurements.append(int(r["measurements"]))
        self.y = np.asarray(self.y, float)
        # domain and position, the names the helpers of PP1 use
        self.domain = self.complex
        self.position = [m.position for m in self.mutation]

    def __len__(self) -> int:
        return len(self.y)

    def load_contexts(self) -> None:
        for name in sorted(set(self.complex)):
            with np.load(_paths(name)["contexts"], allow_pickle=False) as z:
                keys = {k.rsplit("|", 1)[0] for k in z.files}
                for key in keys:
                    site, kind = key.split("|")
                    if (name, site) in self.info:
                        self.contexts[(name, site, kind)] = M._unpack(z, key + "|",
                                                                       f"{name}:{site}")

    def positions(self) -> List[Tuple[str, str]]:
        return sorted(set(zip(self.complex, self.site)))

    def rows_of(self, keys: Sequence[Tuple[str, str]]) -> np.ndarray:
        index = {k: i for i, k in enumerate(keys)}
        return np.array([index[k] for k in zip(self.complex, self.site)], int)

    def group(self, by: str) -> np.ndarray:
        if by == "complex":
            keys = list(self.complex)
        elif by == "family":
            keys = list(self.family)
        else:
            raise ValueError(f"hold out 'complex' or 'family', not {by!r}")
        order = {k: i for i, k in enumerate(sorted(set(keys)))}
        return np.array([order[k] for k in keys], int)

    def targets(self, by: str) -> AtomTargets:
        g = self.group(by)
        return AtomTargets(molecule=g, atom=np.arange(len(self)), y=np.asarray(self.y, float),
                           n_mol=int(g.max()) + 1, name=f"binding ddG, {by} held out")

    def columns(self) -> Dict[str, np.ndarray]:
        """The number columns of every description, per sample."""
        info = [self.info[k] for k in zip(self.complex, self.site)]
        area = np.array([i["area"] for i in info], float)
        unbound = np.array([i["unbound"] for i in info], float)
        return {
            "three": MU.substitution_features(self.mutation)[0][:, :3],
            "count": np.array([[i["count"]] for i in info], float),
            "partner_count": np.array([[i["partner_count"]] for i in info], float),
            "area": area, "buried": unbound - area,
            "location": np.array([[1.0 if loc == c else 0.0 for c in LOCATIONS]
                                  for loc in self.location], float)}


# --------------------------------------------------------------------------- #
# The kernels
# --------------------------------------------------------------------------- #
def context_kernels(fields: Sequence[MoleculeFields], alpha: float = ALPHA
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """(cosine, raw) sum of the five field kernels over `fields`.

    The cosine kernel divides every pair by the size of both fields, so it compares shapes
    only.  The raw kernel keeps the size, divided by the mean size so that its scale is that
    of the cosine one.  A field with no atoms (a site with no partner atom inside the
    envelope) has a raw kernel of zero with everything; in the cosine kernel such fields are
    identical to each other and unlike every other field.
    """
    n = len(fields)
    full = np.array([f.n_atoms > 0 for f in fields])
    cos, raw = np.zeros((n, n)), np.zeros((n, n))
    idx = np.where(full)[0]
    if len(idx):
        fs = FieldSet.create(charge_model="file", alpha=alpha)
        fs.set_alpha(float(alpha))
        table = AtomTable.from_molecules([fields[i] for i in idx])
        # the compiled engine gives the NumPy kernel to 1e-13 about twenty times faster
        K = kernel_matrices(table, table, fs.specs, symmetric=True, engine="auto")
        d = self_kernels(table, fs.specs)
        raw_sub = K.sum(axis=0) / K.shape[0]
        scale = float(np.mean(np.diag(raw_sub)))
        raw[np.ix_(idx, idx)] = raw_sub / scale
        cos[np.ix_(idx, idx)] = normalize_kernels(K, d, d).sum(axis=0) / K.shape[0]
    empty = np.where(~full)[0]
    cos[np.ix_(empty, empty)] = 1.0
    return cos, raw


def position_kernels(s: BindingSamples, kind: str) -> Tuple[List[Tuple[str, str]], np.ndarray,
                                                             np.ndarray]:
    """The (cosine, raw) context kernels over every position, kept in the cache with the
    positions they cover and recomputed when those change."""
    keys = s.positions()
    names = np.array([f"{c}:{k}" for c, k in keys])
    cache = os.path.join(CACHE, f"position_kernel_{kind}.npz")
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as z:
            if z["names"].shape == names.shape and (z["names"] == names).all():
                return keys, np.asarray(z["cos"], float), np.asarray(z["raw"], float)
    t0 = time.time()
    cos, raw = context_kernels([s.contexts[(c, k, kind)] for c, k in keys])
    log(f"  {kind} context kernel over {len(keys)} positions in {time.time() - t0:.0f} s")
    np.savez_compressed(cache, names=names, cos=cos.astype(np.float32),
                        raw=raw.astype(np.float32))
    return keys, cos, raw


DESCRIPTIONS = ("three columns kernel", "one count", "two counts", "interface class",
                "accessible area", "buried area", "field of the surroundings",
                "field of the partner", "buried area and field of the partner",
                "buried area and field of the surroundings")
COMPARISONS = (("one count", "three columns kernel"),
               ("buried area", "three columns kernel"),
               ("two counts", "one count"),
               ("buried area", "accessible area"),
               ("buried area", "two counts"),
               ("buried area", "interface class"),
               ("field of the surroundings", "one count"),
               ("field of the partner", "two counts"),
               ("field of the partner", "buried area"),
               ("buried area and field of the partner", "buried area"),
               ("buried area and field of the surroundings", "buried area"))


def family_of(name: str, X: Dict[str, np.ndarray], K: Dict[str, np.ndarray]
              ) -> Dict[str, np.ndarray]:
    """The kernels one description offers to the inner folds."""
    g = M.gaussian_columns
    three = X["three"]
    if name == "three columns kernel":
        return {"columns": g(three)}
    if name == "one count":
        return {"count": g(np.c_[X["count"], three])}
    if name == "two counts":
        return {"counts": g(np.c_[X["count"], X["partner_count"], three])}
    if name == "interface class":
        return {"class": g(np.c_[X["location"], three])}
    if name == "accessible area":
        return {"area": g(np.c_[X["area"], three])}
    if name == "buried area":
        return {"area": g(np.c_[X["area"], X["buried"], three])}
    if name == "field of the surroundings":
        Kc3 = g(three)
        return {f"weight {w}": M.mixed(Kc3, K["whole"], w) for w in M.MIXES}
    if name == "field of the partner":
        Kn = g(np.c_[X["count"], X["partner_count"], three])
        return {f"{form}, weight {w}": M.mixed(Kn, K[f"partner {form}"], w)
                for form in ("cosine", "raw") for w in M.MIXES}
    if name == "buried area and field of the partner":
        Ka = g(np.c_[X["area"], X["buried"], three])
        return {f"{form}, weight {w}": M.mixed(Ka, K[f"partner {form}"], w)
                for form in ("cosine", "raw") for w in M.MIXES}
    if name == "buried area and field of the surroundings":
        Ka = g(np.c_[X["area"], X["buried"], three])
        return {f"weight {w}": M.mixed(Ka, K["whole"], w) for w in M.MIXES}
    raise ValueError(f"no description {name!r}")


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def per_complex(results, s: BindingSamples, minimum: int = PER_COMPLEX_MIN) -> Dict[str, float]:
    """The correlation of the held-out predictions with ddG inside each complex that has at
    least `minimum` substitutions, averaged over the repeats."""
    y = np.asarray(s.y, float)
    cx = np.array(s.complex)
    out = {}
    for c, n in sorted(Counter(s.complex).items()):
        if n < minimum:
            continue
        idx = cx == c
        if y[idx].std() == 0:
            continue
        r = [np.corrcoef(res.predicted[idx], y[idx])[0, 1] for res in results]
        out[c] = round(float(np.mean(r)), 3)
    return out


def measure(s: BindingSamples, X, K, seeds, holdouts, descriptions=DESCRIPTIONS) -> dict:
    out = {}
    for by in holdouts:
        tg = s.targets(by)
        n_groups = int(tg.molecule.max()) + 1
        folds = min(FOLDS, n_groups)
        rows, keep = {}, {}
        res, chosen = M.mean_reference(tg, seeds, folds)
        rows["mean"], keep["mean"] = M.row_of(res, chosen, tg), res
        for name in descriptions:
            t0 = time.time()
            fam = family_of(name, X, K)
            res, chosen = M.nested(fam, tg, seeds, name, folds)
            del fam
            rows[name], keep[name] = M.row_of(res, chosen, tg), res
            rows[name]["per_complex"] = per_complex(res, s)
            log(f"    {by:8s} {name:42s} q2 {rows[name]['q2']:+.3f}  "
                f"({time.time() - t0:.0f} s)")
        diffs = {f"{a} - {b}": M.compare(keep[a], keep[b], tg)
                 for a, b in COMPARISONS if a in keep and b in keep}
        out[by] = {"rows": rows, "differences": diffs, "n_groups": n_groups, "folds": folds}
    return out


def describe(s: BindingSamples) -> dict:
    X = s.columns()
    loc = Counter(s.location)
    return {"n": len(s), "positions": len(s.positions()), "complexes": len(set(s.complex)),
            "families": len(set(s.family)), "sd_y": round(float(s.y.std()), 4),
            "mean_y": round(float(s.y.mean()), 4),
            "repeated": int(sum(m > 1 for m in s.measurements)),
            "skipped": dict(s.skipped), "missing": dict(s.missing),
            "locations": {c: [int(loc[c]),
                              round(float(s.y[np.array(s.location) == c].mean()), 3)]
                          for c in LOCATIONS if loc[c]},
            "no_partner_atoms": int((X["partner_count"][:, 0] == 0).sum()),
            "largest_family": max(Counter(s.family).values())}


def run(repeats: int = M.REPEATS, holdouts=HOLDOUTS, results: str = RESULTS) -> dict:
    os.makedirs(results, exist_ok=True)
    t0 = time.time()
    s = BindingSamples()
    s.load_contexts()
    log(f"{len(s)} substitutions at {len(s.positions())} positions of "
        f"{len(set(s.complex))} complexes ({sum(s.missing.values())} left out: "
        f"{dict(s.missing)}); loaded after {time.time() - t0:.0f} s")
    keys, cw, _rw = position_kernels(s, "whole")
    keys2, cp, rp = position_kernels(s, "partner")
    rows = s.rows_of(keys)
    K = {"whole": cw[np.ix_(rows, rows)], "partner cosine": cp[np.ix_(rows, rows)],
         "partner raw": rp[np.ix_(rows, rows)]}
    del cw, cp, rp, _rw
    out = {"samples": describe(s), "repeats": repeats, "folds": FOLDS,
           "skempi_sha256": SKEMPI_SHA256}
    out["holdout"] = measure(s, s.columns(), K, list(range(repeats)), holdouts)
    out["seconds"] = round(time.time() - t0, 1)
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(results, "binding.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    log(f"done after {time.time() - t0:.0f} s")
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _iv(d: Sequence[float]) -> str:
    txt = f"{d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f})"
    return f"**{txt}**" if d[1] > 0 or d[2] < 0 else txt


HOLDOUT_NAMES = {"complex": "complexes", "family": "families of complexes"}
LOCATION_NAMES = {"COR": "core of the interface", "RIM": "rim of the interface",
                  "SUP": "support of the interface", "SUR": "surface away from the interface",
                  "INT": "interior of a protein"}


def report(results: str = RESULTS) -> str:
    path = os.path.join(results, "binding.json")
    if not os.path.exists(path):
        return "# PP6: no result yet\n"
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    sm = d["samples"]
    L = ["# Binding, not folding: the frame of a residue at an interface (plan item PP6)\n",
         f"`python benchmarks/run_binding.py`, written {d['written']}, "
         f"{d['seconds'] / 60:.0f} minutes. SKEMPI 2.0, every single substitution with two "
         f"numeric affinities: {sm['n']} substitutions at {sm['positions']} positions of "
         f"{sm['complexes']} complexes in {sm['families']} families (the largest family has "
         f"{sm['largest_family']} substitutions). The target is the change of binding free "
         f"energy, mean {sm['mean_y']:+.2f} and sd {sm['sd_y']:.2f} kcal/mol; "
         f"{sm['repeated']} substitutions were measured more than once and are averaged. "
         f"{sm['no_partner_atoms']} sites have no partner atom inside the envelope. "
         f"Left out: {sm['skipped']}, {sm['missing']}.\n",
         "Every description is joined with the change of volume, hydropathy and charge. "
         f"{d['folds']} folds, {d['repeats']} repeats; every choice is made in inner folds of "
         "the same kind.\n",
         "| Class of the site (SKEMPI) | Substitutions | Mean ddG, kcal/mol |",
         "|---|---:|---:|"]
    for c, (n, m) in sm["locations"].items():
        L.append(f"| {LOCATION_NAMES[c]} | {n} | {m:+.2f} |")
    L.append("")
    for h, hd in d["holdout"].items():
        L.append(f"## Whole {HOLDOUT_NAMES.get(h, h)} held out ({hd['n_groups']} groups)\n")
        L.append(f"q2, and the median over the complexes with at least {PER_COMPLEX_MIN} "
                 "substitutions of the correlation of the held-out predictions with ddG inside "
                 "a complex:\n")
        L.append("| Description | q2 | RMSE, kcal/mol | Median r inside a complex | "
                 "Member chosen most often |")
        L.append("|---|---:|---:|---:|---|")
        for name, row in hd["rows"].items():
            pc = row.get("per_complex")
            med = f"{np.median(list(pc.values())):+.3f}" if pc else ""
            ch = row.get("chosen") or {}
            pick = max(ch, key=ch.get) if len(ch) > 1 else ""
            L.append(f"| {name} | {row['q2']:+.3f} | {row['rmse']:.3f} | {med} | {pick} |")
        L.append("")
        L.append("Differences in q2, with the 95 % interval over the repeats and the folds; "
                 "bold where it excludes zero.\n")
        L.append("| Difference | Value (interval) |")
        L.append("|---|---|")
        for key, v in hd["differences"].items():
            L.append(f"| {key} | {_iv(v)} |")
        L.append("")
    return "\n".join(L)


def write_table(results: str = RESULTS) -> str:
    os.makedirs(results, exist_ok=True)
    path = os.path.join(results, "table.md")
    text = report(results)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--fetch", action="store_true",
                   help="only download the SKEMPI table and the PDB entries of its complexes")
    p.add_argument("--prepare", action="store_true",
                   help="complete the complexes and write the sites before measuring")
    p.add_argument("--workers", type=int, default=8,
                   help="how many processes the preparation runs side by side")
    p.add_argument("--prepare-shard", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--shards", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--repeats", type=int, default=M.REPEATS)
    p.add_argument("--holdout", nargs="*", default=list(HOLDOUTS))
    p.add_argument("--report", action="store_true", help="only rewrite the table")
    args = p.parse_args(argv)
    if args.fetch:
        fetch()
        return
    if args.prepare_shard is not None:
        prepare_share(args.prepare_shard, args.shards)
        return
    if args.prepare:
        prepare(args.workers)
    if not args.report:
        run(args.repeats, tuple(args.holdout))
    print(write_table(RESULTS))


if __name__ == "__main__":
    main()
