"""
Polybenzenoid hydrocarbons: a planar frame, ring classes, ring-centred fields and per-ring
contributions.

A cata-condensed polybenzenoid hydrocarbon (PBH) is a set of fused benzene rings in which no
carbon belongs to three rings.  The COMPAS-1 data set of the Gershoni-Poranne group (Wahab et
al., J. Chem. Inf. Model. 2022, 62, 3704) enumerates every such molecule of one to eleven rings
with optimised geometries and frontier-orbital energies.  It is a good test of the overlap kernel
for two reasons: the property is a quantum-chemical one rather than a binding affinity, and the
molecules are rigid and nearly planar, so the alignment that usually decides a 3D-QSAR model is
reduced to a rule.  This module holds what that study needs beyond the ordinary machinery:

* :func:`read_compas` builds RDKit molecules from the SMILES of the table and the coordinates
  of the xyz files, and checks that the two describe the same atoms;
* :func:`canonical_frame` puts a molecule into its inertial frame (x along the long axis, z
  along the normal) and resolves the signs of the in-plane axes by a fixed rule, so that two
  molecules that are alike in shape are alike in position;
* :func:`pbh_atom_type` is the atom typing of the continuous indicator fields: a ring carbon
  bonded to a hydrogen, a carbon shared by two rings, a carbon shared by three;
* :func:`ring_classes` perceives the rings and classifies them by annulation -- terminal,
  linear, angular, branched -- and :func:`clar_weights` gives every ring its share of the Clar
  sextets;
* :func:`build_pbh_fields` adds one Gaussian per ring centroid to the molecular field, carried
  by the ``ring:<class>`` fields, which is what ``ring:terminal`` and the rest mean in a field set;
* :func:`ring_contributions` sums decomposition rule 3 over the atoms of every ring.

The same machinery serves the later COMPAS sets: the peri-condensed benzenoids of COMPAS-3
(a fifth annulation class, ``peri``, for a ring whose shared bonds touch) and the hetero-atom
systems of COMPAS-2 (rings of four to six atoms with B, N, O or S, typed by ring type; read by
:func:`read_compas2` from the SDF of that set).  :func:`ring_classes` also gives every ring its
HOMA index, the harmonic-oscillator measure of aromaticity computed from the bond lengths
alone, which the ``ring:homa`` field uses as weight.

Everything here is read by ``benchmarks/run_compas.py``; see the manual, section "A frame by
rule: the gap of polybenzenoid hydrocarbons".
"""
from __future__ import annotations

import csv
import gzip
import math
import os
import tarfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from .fields import AtomTable, FieldSet, FieldSpec, MoleculeFields, build_fields

#: the annulation classes of a ring: how many rings are fused to it, and where
RING_CLASSES = ("terminal", "linear", "angular", "branched")
#: the field names that carry one Gaussian per ring centroid
RING_FIELDS = tuple("ring:" + c for c in RING_CLASSES)
#: the ring field weighted by the Clar-sextet share of the ring (a placeholder for an
#: aromaticity field: a NICS-weighted field needs a DFT calculation per ring)
SEXTET_FIELD = "ring:sextet"
#: the carbon classes of the indicator fields
PBH_TYPES = ("C.H", "C.2r", "C.3r")
#: columns of the COMPAS-1 tables kept as molecule properties
COMPAS_NUMERIC = ("HOMO_eV", "LUMO_eV", "GAP_eV", "n_rings", "Erel_eV", "Dipmom_Debye")
COMPAS_TEXT = ("smiles", "lalas", "augmented_lalas", "balaban_notation")
#: symbol and radius of a ring-centre pseudo-atom
RING_SYMBOL = "Du"
RING_RADIUS = 1.7
#: the fifth annulation class, of peri-condensed systems: a ring two of whose shared bonds touch
#: (so that one of its carbons belongs to three rings), or a ring fused to four or more rings
PERI_CLASS = "peri"
RING_CLASSES_PERI = RING_CLASSES + (PERI_CLASS,)
#: the ring field weighted by the HOMA index of the ring: the harmonic-oscillator model of
#: aromaticity, 1 for the bond lengths of benzene and 0 for a Kekulé structure, from the
#: optimised geometry alone (Krygowski, J. Chem. Inf. Comput. Sci. 1993, 33, 70)
HOMA_FIELD = "ring:homa"
#: HOMA parameters per bond type: the optimal length in Å and the normalisation constant
HOMA_PARAMS = {("C", "C"): (1.388, 257.7), ("C", "N"): (1.334, 93.52), ("C", "O"): (1.265, 157.38),
               ("C", "S"): (1.677, 94.09), ("N", "N"): (1.309, 130.33), ("N", "O"): (1.248, 57.21)}
#: the ring types of COMPAS-2, named as in its table (``14diborinine`` is 1,4-diborinine,
#: ``dhdiborinine`` the 1,2-dihydro-1,2-diborinine with two bonded borons)
RING_TYPES = ("benzene", "pyridine", "pyrazine", "borinine", "14diborinine", "dhdiborinine",
              "pyrrole", "furan", "thiophene", "borole", "cyclobutadiene")
#: the atom types of the hetero-atom sets beyond :data:`PBH_TYPES`
HETERO_TYPES = ("N", "N.H", "O", "S", "B", "B.H")


# --------------------------------------------------------------------------- #
# Reading COMPAS-1
# --------------------------------------------------------------------------- #

def parse_xyz(text: str) -> Tuple[List[str], np.ndarray]:
    """Symbols and coordinates of one xyz block (count line, comment line, atoms)."""
    lines = text.splitlines()
    n = int(lines[0].split()[0])
    rows = [ln.split() for ln in lines[2:2 + n]]
    if len(rows) != n:
        raise ValueError(f"xyz block announces {n} atoms and holds {len(rows)}")
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def mol_from_smiles_xyz(smiles: str, symbols: Sequence[str], coords: np.ndarray,
                        name: str = "", max_bond: float = 1.8) -> Chem.Mol:
    """An RDKit molecule with the bonds of `smiles` and the coordinates of the xyz atoms.

    The xyz file and the SMILES list the atoms in different orders.  Bonds are perceived from the
    coordinates, the two graphs are matched (elements and connectivity, bond orders ignored), and
    the coordinates are copied over the match.  A mismatch -- different elements, a graph that
    does not match, a bonded pair further apart than `max_bond` angstroms after the copy -- raises
    ``ValueError`` naming the molecule.
    """
    label = name or smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"{label}: the SMILES {smiles!r} cannot be parsed")
    mol = Chem.AddHs(mol)
    want = Counter(a.GetSymbol() for a in mol.GetAtoms())
    have = Counter(str(s).capitalize() for s in symbols)
    if want != have:
        raise ValueError(f"{label}: the SMILES has the atoms {dict(want)} and the xyz file "
                         f"{dict(have)}")
    coords = np.asarray(coords, float)
    block = [str(len(symbols)), label] + [f"{s} {x:.6f} {y:.6f} {z:.6f}"
                                          for s, (x, y, z) in zip(symbols, coords)]
    xyz_mol = Chem.MolFromXYZBlock("\n".join(block) + "\n")
    if xyz_mol is None:
        raise ValueError(f"{label}: the xyz block cannot be read")
    rdDetermineBonds.DetermineConnectivity(xyz_mol)
    # the template as a query with generic bonds: the xyz graph has no bond orders
    query = Chem.RWMol(mol)
    Chem.Kekulize(query, clearAromaticFlags=True)
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(query, params)
    match = xyz_mol.GetSubstructMatch(query, useChirality=False)
    if len(match) != mol.GetNumAtoms():
        raise ValueError(f"{label}: the bonds perceived from the coordinates do not match "
                         f"the SMILES {smiles!r}")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, j in enumerate(match):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in coords[j]]))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    pos = mol.GetConformer().GetPositions()
    for b in mol.GetBonds():
        d = float(np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]))
        if d > max_bond:
            raise ValueError(f"{label}: atoms {b.GetBeginAtomIdx()} and {b.GetEndAtomIdx()} are "
                             f"bonded in the SMILES and {d:.2f} A apart in the geometry")
    if name:
        mol.SetProp("_Name", name)
    return mol


class _XyzSource:
    """The xyz files of a COMPAS archive, read from a folder or straight from the tar.gz."""

    def __init__(self, path: str):
        self.path = path
        self.tar = None
        self.members: Dict[str, str] = {}
        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for f in files:
                    if f.endswith(".xyz"):
                        self.members[f[:-4]] = os.path.join(root, f)
        elif tarfile.is_tarfile(path):
            self.tar = tarfile.open(path, "r:*")
            for m in self.tar.getmembers():
                base = os.path.basename(m.name)
                if m.isfile() and base.endswith(".xyz"):
                    self.members[base[:-4]] = m.name
        else:
            raise ValueError(f"{path}: not a folder of xyz files and not a tar archive")

    def __contains__(self, name: str) -> bool:
        return name in self.members

    def __len__(self) -> int:
        return len(self.members)

    def read(self, name: str) -> str:
        if name not in self.members:
            raise KeyError(name)
        if self.tar is not None:
            fh = self.tar.extractfile(self.members[name])
            return fh.read().decode("utf-8", errors="replace")
        with open(self.members[name], encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()


def read_compas_table(csv_path: str) -> List[Dict[str, str]]:
    """The rows of a COMPAS-1 csv file, as dictionaries keyed by the column names."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    needed = ("molecule", "smiles", "GAP_eV", "n_rings")
    missing = [c for c in needed if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{csv_path}: columns missing: {missing}")
    return rows


def read_compas(csv_path: str, geometries: str, limit: Optional[int] = None,
                names: Optional[Sequence[str]] = None, progress=None):
    """The molecules of a COMPAS-1 set as a :class:`cmfqsar.dataio.Dataset`.

    `geometries` is the folder of xyz files or the tar.gz archive itself.  Every row of the table
    becomes one molecule with the columns of :data:`COMPAS_NUMERIC` and :data:`COMPAS_TEXT` as
    properties and ``GAP_eV`` as the activity.  `limit` keeps the first rows only; `names`
    keeps the listed molecules only.  A molecule whose geometry does not match its SMILES stops
    the reading with a ``ValueError``.
    """
    from .dataio import Dataset

    rows = read_compas_table(csv_path)
    if names is not None:
        keep = set(names)
        rows = [r for r in rows if r["molecule"] in keep]
    if limit:
        rows = rows[:int(limit)]
    src = _XyzSource(geometries)
    try:
        missing = [r["molecule"] for r in rows if r["molecule"] not in src]
        if missing:
            raise ValueError(f"{len(missing)} molecules of {csv_path} have no xyz file in "
                             f"{geometries}; the first is {missing[0]}")
        mols, out_names, y = [], [], []
        for k, r in enumerate(rows):
            if progress is not None and k % 500 == 0:
                progress(k, len(rows))
            symbols, coords = parse_xyz(src.read(r["molecule"]))
            mol = mol_from_smiles_xyz(r["smiles"], symbols, coords, name=r["molecule"])
            for col in COMPAS_NUMERIC:
                if col in r and r[col] != "":
                    mol.SetProp(col, r[col])
            for col in COMPAS_TEXT:
                if col in r:
                    mol.SetProp(col, r[col])
            mols.append(mol)
            out_names.append(r["molecule"])
            y.append(float(r["GAP_eV"]))
    finally:
        src.close()
    ds = Dataset(mols=mols, names=out_names, activity=np.array(y, float),
                 subset=np.array(["train"] * len(mols), dtype=object),
                 properties=[c for c in COMPAS_NUMERIC if mols and mols[0].HasProp(c)],
                 path=csv_path, activity_prop="GAP_eV")
    return ds


def read_compas2(csv_path: str, sdf_gz: str, limit: Optional[int] = None,
                 names: Optional[Sequence[str]] = None, progress=None):
    """The neutral molecules of a COMPAS-2 set as a :class:`cmfqsar.dataio.Dataset`.

    COMPAS-2 stores one row per molecule and charge state (-1, 0, +1) and the geometries in one
    gzipped SDF with ``name`` and ``charge`` tags.  The neutral rows are kept, the geometry of
    the neutral state is taken, and the SDF graph is checked against the SMILES of the table.
    The columns ``gap``, ``homo``, ``lumo`` and ``rings`` become ``GAP_eV``, ``HOMO_eV``,
    ``LUMO_eV`` and ``n_rings``, so that the molecules carry the same properties as COMPAS-1.
    """
    from .dataio import Dataset

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("charge") == "0"]
    needed = ("name", "smiles", "gap", "rings")
    missing = [c for c in needed if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{csv_path}: columns missing: {missing}")
    if names is not None:
        keep = set(names)
        rows = [r for r in rows if r["name"] in keep]
    if limit:
        rows = rows[:int(limit)]
    wanted = {r["name"]: r for r in rows}
    found: Dict[str, Chem.Mol] = {}
    with gzip.open(sdf_gz, "rb") as fh:
        supplier = Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=True)
        for k, m in enumerate(supplier):
            if progress is not None and k % 5000 == 0:
                progress(len(found), len(wanted))
            if m is None or not m.HasProp("name") or m.GetProp("charge") != "0":
                continue
            name = m.GetProp("name")
            if name in wanted and name not in found:
                found[name] = m
                if len(found) == len(wanted):
                    break
    lost = [n for n in wanted if n not in found]
    if lost:
        raise ValueError(f"{len(lost)} molecules of {csv_path} have no neutral geometry in "
                         f"{sdf_gz}; the first is {lost[0]}")
    mols, out_names, y = [], [], []
    for name, r in wanted.items():
        m = Chem.Mol(found[name])
        ref = Chem.MolFromSmiles(r["smiles"])
        if ref is None or Chem.MolToSmiles(Chem.RemoveHs(m)) != Chem.MolToSmiles(ref):
            raise ValueError(f"{name}: the SDF graph does not match the SMILES of the table")
        if m.GetNumConformers() != 1 or not m.GetConformer().Is3D():
            raise ValueError(f"{name}: no 3D geometry in the SDF")
        m.SetProp("_Name", name)
        for col, prop in (("gap", "GAP_eV"), ("homo", "HOMO_eV"), ("lumo", "LUMO_eV"),
                          ("rings", "n_rings"), ("heteroatoms", "n_heteroatoms"),
                          ("aromatic_rings", "n_aromatic_rings")):
            if col in r and r[col] != "":
                m.SetProp(prop, r[col])
        for col in ("smiles", "formula", "inchi"):
            if col in r:
                m.SetProp(col, r[col])
        mols.append(m)
        out_names.append(name)
        y.append(float(r["gap"]))
    ds = Dataset(mols=mols, names=out_names, activity=np.array(y, float),
                 subset=np.array(["train"] * len(mols), dtype=object),
                 properties=["HOMO_eV", "LUMO_eV", "GAP_eV", "n_rings"],
                 path=csv_path, activity_prop="GAP_eV")
    return ds


# --------------------------------------------------------------------------- #
# The planar frame
# --------------------------------------------------------------------------- #

def heavy_mask(mol: Chem.Mol) -> np.ndarray:
    return np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()], dtype=bool)


def atomic_masses(mol: Chem.Mol) -> np.ndarray:
    pt = Chem.GetPeriodicTable()
    return np.array([pt.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()], float)


def inertial_frame(coords: np.ndarray, masses: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre of mass, the rows x, y, z of the inertial frame, and the three moments.

    x is the axis of the smallest moment of inertia (the largest extent), z the axis of the
    largest moment (the normal of a planar molecule); the frame is right-handed.
    """
    coords = np.asarray(coords, float)
    m = np.ones(len(coords)) if masses is None else np.asarray(masses, float)
    com = (coords * m[:, None]).sum(axis=0) / m.sum()
    r = coords - com
    r2 = (r ** 2).sum(axis=1)
    tensor = np.einsum("i,i,jk->jk", m, r2, np.eye(3)) - np.einsum("i,ij,ik->jk", m, r, r)
    moments, vecs = np.linalg.eigh(tensor)
    axes = vecs.T.copy()                     # rows, by increasing moment
    axes[2] = np.cross(axes[0], axes[1])     # right-handed
    return com, axes, moments


def _direction_moment(xy: np.ndarray, power: int, step_deg: float = 0.5
                      ) -> Tuple[float, float, float]:
    """The in-plane direction that maximises sum((x cos t + y sin t)^power): a grid over the
    angle, then a bounded refinement around the best grid point, then Newton steps on the
    derivative.  Returns (angle, value, the spread of the value over the angles relative to its
    mean magnitude); a spread near zero means the moment does not depend on the direction and
    cannot fix it.

    The last step is what makes the angle, and with it the frame, the same on every machine.  A
    search that only compares values of the moment cannot do better than the square root of the
    machine precision, about 1e-8 radians, because the moment is flat at its maximum: a change
    of 1e-8 radians moves it by less than one part in 1e16.  The derivative is not flat there,
    so solving ``derivative = 0`` by Newton's method places the angle to the last digit.  Before
    this step the frame of a molecule with a three-fold axis differed by up to 5e-8 A between
    runs on different machines, and applying the frame twice was not the identity.
    """
    from scipy.optimize import minimize_scalar

    def projections(t):
        return (xy[:, 0] * math.cos(t) + xy[:, 1] * math.sin(t),
                -xy[:, 0] * math.sin(t) + xy[:, 1] * math.cos(t))

    def value(t):
        return float((projections(t)[0] ** power).sum())

    def derivatives(t):
        """The first two derivatives of the moment with respect to the angle."""
        p, d = projections(t)
        first = power * float((p ** (power - 1) * d).sum())
        second = power * float(((power - 1) * p ** (power - 2) * d ** 2 - p ** power).sum())
        return first, second

    t = np.radians(np.arange(0.0, 360.0, step_deg))
    proj = xy[:, 0][:, None] * np.cos(t)[None, :] + xy[:, 1][:, None] * np.sin(t)[None, :]
    s = (proj ** power).sum(axis=0)
    k = int(np.argmax(s))
    step = math.radians(step_deg)
    res = minimize_scalar(lambda u: -value(u), bounds=(t[k] - step, t[k] + step), method="bounded",
                          options={"xatol": 1e-10})
    best = float(res.x) if res.success and -res.fun >= s[k] else float(t[k])
    # Newton on the derivative, from the angle the search found.  A step is taken only while it
    # points at a maximum (a negative second derivative) and stays inside the grid cell the
    # search worked in, so a flat or badly behaved moment leaves the search result untouched.
    for _ in range(8):
        first, second = derivatives(best)
        if not (second < 0.0) or not np.isfinite(first):
            break
        shift = -first / second
        if not np.isfinite(shift) or abs(shift) > step:
            break
        best += shift
        if abs(shift) < 1e-15:
            break
    # the spread is measured against the largest value the moment could take, so that a moment
    # that vanishes by symmetry (the third of a centrosymmetric molecule) is not mistaken for a
    # direction-dependent one because of the noise in the coordinates
    full = float((np.sqrt((xy ** 2).sum(axis=1)) ** power).sum()) + 1e-300
    spread = float((s.max() - s.min()) / full)
    return best, value(best), spread


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sign_key(xy: np.ndarray) -> np.ndarray:
    """The five moments that decide the signs of the in-plane axes.

    Under x -> -x the first and the last two change sign; under y -> -y the second, third and
    the last two do.  A molecule for which all five vanish is symmetric under the flip in
    question, and its fields are then the same in either frame.
    """
    x, y = xy[:, 0], xy[:, 1]
    return np.array([(x ** 3).sum(), (y ** 3).sum(), (x * x * y).sum(),
                     (x ** 3 * y).sum(), (x * y ** 3).sum()])


#: the four proper rotations that keep the inertial axes: (sign of x, sign of y); z follows
FLIPS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))


def resolve_signs(coords: np.ndarray, heavy: Optional[np.ndarray] = None,
                  tol: float = 1e-6) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Among the four sign choices of a frame, the one with the largest :func:`sign_key`.

    The keys are compared in order, each within a tolerance relative to the size of the
    molecule; candidates still tied after the fifth key are equivalent by symmetry and the first
    in :data:`FLIPS` is taken.  Returns the coordinates and the flip applied.
    """
    sel = coords[heavy] if heavy is not None and heavy.any() else coords
    base = sign_key(sel[:, :2])
    scale = float(np.abs(sel[:, :2]).sum() ** 3) / max(len(sel), 1) ** 2 + 1e-12
    signs = np.array([[sx, sy, sy, sx * sy, sx * sy] for sx, sy in FLIPS])
    keys = signs * base[None, :]
    alive = list(range(len(FLIPS)))
    for k in range(5):
        best = max(keys[i, k] for i in alive)
        alive = [i for i in alive if keys[i, k] >= best - tol * scale]
        if len(alive) == 1:
            break
    sx, sy = FLIPS[alive[0]]
    return coords * np.array([sx, sy, sx * sy]), (sx, sy)


def canonical_frame(mol: Chem.Mol, coords: Optional[np.ndarray] = None,
                    degenerate_tol: float = 1e-3) -> np.ndarray:
    """The coordinates of `mol` in its canonical planar frame.

    The frame is that of the heavy-atom inertia tensor: origin at the centre of mass, x along the
    axis of the smallest moment (the long axis), z along the largest (the normal).  When the two
    in-plane moments are equal within `degenerate_tol` of their mean (benzene, triphenylene and
    the other molecules with a three-fold or higher axis), x is put along the in-plane direction
    with the largest third moment of the heavy atoms, or the largest fourth or sixth moment when
    the lower ones do not depend on the direction.  The signs of x and y are then fixed by
    :func:`resolve_signs`.  Hydrogens follow the heavy atoms and take no part in the rule.
    """
    pos = np.asarray(mol.GetConformer().GetPositions() if coords is None else coords, float)
    heavy = heavy_mask(mol)
    if not heavy.any():
        heavy = np.ones(len(pos), bool)
    com, axes, moments = inertial_frame(pos[heavy], atomic_masses(mol)[heavy])
    out = (pos - com) @ axes.T
    if abs(moments[1] - moments[0]) <= degenerate_tol * max(0.5 * (moments[0] + moments[1]), 1e-12):
        # a three-fold axis leaves the third moment direction-dependent; a six-fold one
        # (benzene) makes every moment up to the fifth isotropic, so the sixth is needed
        xy = out[heavy, :2]
        theta = 0.0
        for power in (3, 4, 6):
            theta, _value, spread = _direction_moment(xy, power)
            if spread > 1e-4:
                break
        out = out @ _rot_z(-theta).T
    out, _flip = resolve_signs(out, heavy)
    return out


def frame_candidates(coords: np.ndarray, z_flip: bool = False) -> List[np.ndarray]:
    """The four (eight with `z_flip`) frames that share the inertial axes of `coords`."""
    out = [coords * np.array([sx, sy, sx * sy]) for sx, sy in FLIPS]
    if z_flip:
        out += [c * np.array([1.0, 1.0, -1.0]) for c in out]
    return out


def choose_frame_by_kernel(mol: Chem.Mol, fieldset: FieldSet, reference: AtomTable,
                           scale: Optional[np.ndarray] = None, z_flip: bool = False,
                           engine=None) -> Tuple[np.ndarray, np.ndarray]:
    """Alignment by prediction confidence: of the frames that share the inertial axes, the one
    whose field is on average most similar to the molecules of `reference`.

    The similarity is the cosine-normalised joint kernel, with the field scales `scale` of the
    model when given.  Returns the chosen coordinates and the mean similarity of every candidate
    in the order of :func:`frame_candidates`.
    """
    from .kernels import combine_kernels, kernel_matrices, self_kernels

    base = canonical_frame(mol)
    specs = [s for s in fieldset.specs if s.enabled]
    h = np.array([s.h for s in specs])
    c = np.ones(len(specs)) if scale is None else np.asarray(scale, float)[:len(specs)]
    d_ref = np.asarray(self_kernels(reference, specs), float)          # (F, n_ref)
    ref_self = (d_ref.T * (h * c)).sum(axis=1)
    scores = []
    cands = frame_candidates(base, z_flip=z_flip)
    for cand in cands:
        m = Chem.Mol(mol)
        conf = m.GetConformer()
        for i, xyz in enumerate(cand):
            conf.SetAtomPosition(i, Point3D(*[float(v) for v in xyz]))
        mf = build_pbh_fields(m, fieldset)
        t = AtomTable.from_molecules([mf])
        K = kernel_matrices(t, reference, specs, engine=engine)
        joint = combine_kernels(K, h, c)[0]
        own = float((np.asarray(self_kernels(t, specs), float)[:, 0] * h * c).sum())
        sim = joint / np.sqrt(np.maximum(own * ref_self, 1e-300))
        scores.append(float(sim.mean()))
    k = int(np.argmax(scores))
    return cands[k], np.array(scores)


# --------------------------------------------------------------------------- #
# Atom types and rings
# --------------------------------------------------------------------------- #

def pbh_atom_type(atom: Chem.Atom) -> str:
    """The carbon classes of a benzenoid: ``C.H`` on the rim (one ring, bonded to a hydrogen),
    ``C.2r`` shared by two rings, ``C.3r`` shared by three (absent in a cata-condensed set).
    Hydrogens are ``H``; a hetero atom is its element symbol, with ``.H`` appended when it
    carries a hydrogen (``N.H`` of pyrrole, ``B.H`` of borole and borinine); ``C.1r`` is a ring
    carbon without a hydrogen and ``C`` a carbon outside the rings."""
    z = atom.GetAtomicNum()
    if z == 1:
        return "H"
    if z != 6:
        sym = atom.GetSymbol()
        return sym + ".H" if atom.GetTotalNumHs(includeNeighbors=True) > 0 else sym
    n = atom.GetOwningMol().GetRingInfo().NumAtomRings(atom.GetIdx())
    if n >= 3:
        return "C.3r"
    if n == 2:
        return "C.2r"
    if n == 1:
        return "C.H" if atom.GetTotalNumHs(includeNeighbors=True) > 0 else "C.1r"
    return "C"


@dataclass
class Ring:
    """One benzene ring of a benzenoid."""
    atoms: Tuple[int, ...]          # the six carbons, in cyclic order
    centroid: np.ndarray            # (3,)
    neighbours: Tuple[int, ...]     # indices of the rings fused to it
    cls: str                        # one of RING_CLASSES_PERI
    sextet: float = 0.0             # the Clar-sextet share, set by clar_weights
    homa: float = float("nan")      # the HOMA index from the bond lengths (nan: no parameters)
    ring_type: str = "benzene"      # one of RING_TYPES, or "ring:<size>:<hetero atoms>"


def _cyclic_order(mol: Chem.Mol, atoms: Sequence[int]) -> Tuple[int, ...]:
    """The atoms of a ring in the order in which they are bonded."""
    inside = set(atoms)
    order = [atoms[0]]
    prev = None
    while len(order) < len(atoms):
        cur = order[-1]
        nxt = [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
               if n.GetIdx() in inside and n.GetIdx() != prev and n.GetIdx() not in order]
        if not nxt:
            raise ValueError("ring atoms do not form a cycle")
        prev, cur = cur, nxt[0]
        order.append(cur)
    return tuple(order)


def ring_type_of(mol: Chem.Mol, atoms: Sequence[int]) -> str:
    """The ring type of COMPAS-2 from the ring size and its hetero atoms, or a generic
    ``ring:<size>:<symbols>`` name for anything else."""
    syms = [mol.GetAtomWithIdx(int(i)).GetSymbol() for i in atoms]
    hetero = sorted(s for s in syms if s != "C")
    n = len(atoms)
    if n == 6:
        if not hetero:
            return "benzene"
        if hetero == ["N"]:
            return "pyridine"
        if hetero == ["N", "N"]:
            return "pyrazine"
        if hetero == ["B"]:
            return "borinine"
        if hetero == ["B", "B"]:
            borons = [int(i) for i in atoms if mol.GetAtomWithIdx(int(i)).GetSymbol() == "B"]
            bonded = mol.GetBondBetweenAtoms(borons[0], borons[1]) is not None
            return "dhdiborinine" if bonded else "14diborinine"
    if n == 5 and len(hetero) == 1:
        return {"N": "pyrrole", "O": "furan", "S": "thiophene", "B": "borole"}.get(hetero[0],
                                                                                  "")
    if n == 4 and not hetero:
        return "cyclobutadiene"
    return f"ring:{n}:{''.join(hetero) or 'C'}"


def ring_homa(mol: Chem.Mol, pos: np.ndarray, atoms: Sequence[int]) -> float:
    """The HOMA index of one ring from its bond lengths, ``nan`` when a bond of the ring has
    no parameters (a bond to boron)."""
    total = 0.0
    n = len(atoms)
    for k in range(n):
        a, b = int(atoms[k]), int(atoms[(k + 1) % n])
        key = tuple(sorted((mol.GetAtomWithIdx(a).GetSymbol(), mol.GetAtomWithIdx(b).GetSymbol())))
        if key not in HOMA_PARAMS:
            return float("nan")
        r_opt, alpha = HOMA_PARAMS[key]
        total += alpha * (r_opt - float(np.linalg.norm(pos[a] - pos[b]))) ** 2
    return 1.0 - total / n


def ring_classes(mol: Chem.Mol, coords: Optional[np.ndarray] = None,
                 carbon_only: bool = True) -> List[Ring]:
    """The rings of `mol` with their centroids, annulation classes, HOMA indices and types.

    With `carbon_only` (the default) only six-membered carbon rings are taken, as in a
    benzenoid; otherwise every ring of four to six atoms, which is what the hetero-atom systems
    of COMPAS-2 need.  Two rings are fused when they share a bond.  A ring with one fused
    neighbour (or none: benzene) is ``terminal``; with two, it is ``linear`` when the two shared
    bonds face each other across the ring (anthracene's middle ring) and ``angular`` when they
    do not (phenanthrene's; any five-membered ring); with three it is ``branched``
    (triphenylene's centre).  Two shared bonds that touch make a carbon of three rings, which is
    peri-condensation: such a ring, and a ring fused to four or more, is ``peri``.
    """
    pos = np.asarray(mol.GetConformer().GetPositions() if coords is None else coords, float)
    if carbon_only:
        rings_raw = [tuple(r) for r in mol.GetRingInfo().AtomRings()
                     if len(r) == 6 and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in r)]
    else:
        rings_raw = [tuple(r) for r in mol.GetRingInfo().AtomRings() if 4 <= len(r) <= 6]
    ordered = [_cyclic_order(mol, r) for r in rings_raw]
    sets = [frozenset(r) for r in ordered]
    rings: List[Ring] = []
    for i, atoms in enumerate(ordered):
        n = len(atoms)
        bonds = [frozenset((atoms[k], atoms[(k + 1) % n])) for k in range(n)]
        nbrs, shared = [], []
        for j in range(len(ordered)):
            pair = sets[i] & sets[j]
            if j != i and len(pair) == 2 and frozenset(pair) in bonds:
                nbrs.append(j)
                shared.append(bonds.index(frozenset(pair)))
        touching = any((shared[a] - shared[b]) % n in (1, n - 1)
                       for a in range(len(shared)) for b in range(a))
        if len(nbrs) <= 1:
            cls = "terminal"
        elif touching or len(nbrs) > 3:
            cls = PERI_CLASS
        elif len(nbrs) == 2:
            d = (shared[1] - shared[0]) % n
            cls = "linear" if n % 2 == 0 and d == n // 2 else "angular"
        else:
            cls = "branched"
        rings.append(Ring(atoms=atoms, centroid=pos[list(atoms)].mean(axis=0),
                          neighbours=tuple(nbrs), cls=cls, homa=ring_homa(mol, pos, atoms),
                          ring_type=ring_type_of(mol, atoms)))
    return rings


def ring_class_counts(rings: Sequence[Ring]) -> Dict[str, int]:
    out = {c: 0 for c in RING_CLASSES}
    for r in rings:
        out[r.cls] = out.get(r.cls, 0) + 1
    return out


def annulation_counts(lalas: str) -> Tuple[int, int]:
    """(linear, angular-or-branched) rings announced by a COMPAS annulation string.

    The ``lalas`` column writes one letter per inner ring: ``L`` for a linearly annulated ring,
    ``A`` for an angular one; a branch is written in parentheses after the ring it hangs from,
    and that ring is also written ``A``.  Terminal rings are not written.
    """
    s = str(lalas or "")
    return s.count("L"), s.count("A")


# --------------------------------------------------------------------------- #
# Clar sextets
# --------------------------------------------------------------------------- #

def _has_perfect_matching(n: int, edges: Sequence[Tuple[int, int]], nodes: Sequence[int]) -> bool:
    """Whether the subgraph on `nodes` has a perfect matching (bipartite augmenting paths; the
    carbon skeleton of a benzenoid is bipartite)."""
    nodes = list(nodes)
    if len(nodes) % 2:
        return False
    if not nodes:
        return True
    inside = set(nodes)
    adj: Dict[int, List[int]] = {v: [] for v in nodes}
    for a, b in edges:
        if a in inside and b in inside:
            adj[a].append(b)
            adj[b].append(a)
    colour: Dict[int, int] = {}
    for s in nodes:                                  # two-colouring, component by component
        if s in colour:
            continue
        colour[s] = 0
        stack = [s]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if w not in colour:
                    colour[w] = 1 - colour[v]
                    stack.append(w)
                elif colour[w] == colour[v]:
                    return False                     # not bipartite: not a benzenoid skeleton
    left = [v for v in nodes if colour[v] == 0]
    match: Dict[int, int] = {}

    def augment(v: int, seen: set) -> bool:
        for w in adj[v]:
            if w in seen:
                continue
            seen.add(w)
            if w not in match or augment(match[w], seen):
                match[w] = v
                return True
        return False

    size = sum(1 for v in left if augment(v, set()))
    return 2 * size == len(nodes)


def clar_structures(mol: Chem.Mol, rings: Sequence[Ring]) -> List[Tuple[int, ...]]:
    """Every Clar structure with the largest number of sextets, as tuples of ring indices.

    A Clar structure places aromatic sextets on rings that share no atom and covers every other
    carbon by a double bond.  All independent sets of rings are enumerated (at most eleven rings
    in COMPAS-1) and those whose remaining carbons admit a perfect matching are kept; the ones of
    the largest size are returned.
    """
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
             if b.GetBeginAtom().GetAtomicNum() == 6 and b.GetEndAtom().GetAtomicNum() == 6]
    sets = [set(r.atoms) for r in rings]
    best: List[Tuple[int, ...]] = []
    best_size = -1

    def rec(start: int, chosen: List[int], used: set) -> None:
        nonlocal best, best_size
        rest = [c for c in carbons if c not in used]
        if _has_perfect_matching(mol.GetNumAtoms(), edges, rest):
            if len(chosen) > best_size:
                best, best_size = [tuple(chosen)], len(chosen)
            elif len(chosen) == best_size:
                best.append(tuple(chosen))
        for i in range(start, len(rings)):
            if sets[i] & used:
                continue
            rec(i + 1, chosen + [i], used | sets[i])

    rec(0, [], set())
    return best


def clar_weights(mol: Chem.Mol, rings: Sequence[Ring]) -> np.ndarray:
    """The share of the maximal Clar structures in which every ring carries a sextet, in [0, 1];
    also written into ``Ring.sextet``.  Naphthalene gives 0.5 and 0.5, anthracene 1/3 each,
    phenanthrene 1, 0, 1."""
    structures = clar_structures(mol, rings)
    w = np.zeros(len(rings))
    for s in structures:
        for i in s:
            w[i] += 1.0
    if structures:
        w /= len(structures)
    for r, v in zip(rings, w):
        r.sextet = float(v)
    return w


# --------------------------------------------------------------------------- #
# Ring-centred fields
# --------------------------------------------------------------------------- #

def ring_field_weight(ring: Ring, field_name: str) -> float:
    """The weight of one ring in a ``ring:<class>``, ``ring:<type>``, ``ring:sextet`` or
    ``ring:homa`` field."""
    cls = field_name[5:]
    if cls == "sextet":
        return float(ring.sextet)
    if cls == "homa":
        return 0.0 if math.isnan(ring.homa) else float(ring.homa)
    return 1.0 if cls in (ring.cls, ring.ring_type) else 0.0


def with_ring_centres(mf: MoleculeFields, rings: Sequence[Ring], fieldset: FieldSet) -> MoleculeFields:
    """`mf` with one pseudo-atom per ring centroid appended, carrying the ``ring:`` fields.

    The pseudo-atoms have weight zero in every atom field and the real atoms weight zero in the
    ring fields, so the kernel of a ring field is a sum over ring pairs only.  They are appended
    in the order of `rings`, which :func:`ring_contributions` relies on.
    """
    ring_cols = [k for k, s in enumerate(fieldset.specs) if s.name.startswith("ring:")]
    if not ring_cols or not rings:
        return mf
    n_r = len(rings)
    W = np.zeros((n_r, mf.weights.shape[1]))
    for k in ring_cols:
        name = fieldset.specs[k].name
        W[:, k] = [ring_field_weight(r, name) for r in rings]
    return MoleculeFields(
        name=mf.name,
        coords=np.vstack([mf.coords, np.array([r.centroid for r in rings])]),
        weights=np.vstack([mf.weights, W]),
        radii=np.concatenate([mf.radii, np.full(n_r, RING_RADIUS)]),
        symbols=list(mf.symbols) + [RING_SYMBOL] * n_r,
        atom_types=list(mf.atom_types) + ["ring:" + r.cls for r in rings],
        bonds=mf.bonds, charges=np.concatenate([mf.charges, np.zeros(n_r)]),
        molblock=mf.molblock, descriptors=mf.descriptors, probe_coeffs=mf.probe_coeffs,
        charge_source=mf.charge_source)


def build_pbh_fields(mol: Chem.Mol, fieldset: FieldSet, name: str = "",
                     rings: Optional[Sequence[Ring]] = None) -> MoleculeFields:
    """:func:`cmfqsar.fields.build_fields` followed by the ring-centre pseudo-atoms when the
    field set has ``ring:`` fields.  The Clar weights are computed when ``ring:sextet`` asks."""
    mf = build_fields(mol, fieldset, name=name)
    if not any(s.name.startswith("ring:") for s in fieldset.specs):
        return mf
    if rings is None:
        rings = ring_classes(mol)
        if any(s.name == SEXTET_FIELD for s in fieldset.specs):
            clar_weights(mol, rings)
    return with_ring_centres(mf, rings, fieldset)


def n_real_atoms(mf: MoleculeFields) -> int:
    """Atoms of `mf` that are not ring-centre pseudo-atoms."""
    return sum(1 for s in mf.symbols if s != RING_SYMBOL)


def ring_contributions(model, mf: MoleculeFields, rings: Sequence[Ring]
                       ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Decomposition rule 3 summed per ring.

    Every atom's contribution (:meth:`cmfqsar.model.CMFModel.atom_contributions`) goes to the
    rings it belongs to, shared equally when it belongs to two; a hydrogen goes with the heavy
    atom it is bonded to; a ring-centre pseudo-atom goes to its ring.  Returns the per-ring sums,
    the per-atom contributions they were summed from, and what could not be placed (atoms in no
    ring), which is zero for a benzenoid.  The per-ring sums plus the unplaced part equal the
    prediction minus the bias.
    """
    contrib = np.asarray(model.atom_contributions(mf), float)
    n_real = n_real_atoms(mf)
    members: List[List[int]] = [[] for _ in range(n_real)]
    for k, r in enumerate(rings):
        for a in r.atoms:
            members[a].append(k)
    heavy_of: Dict[int, int] = {}
    for a, b in np.asarray(mf.bonds, int).reshape(-1, 2):
        if mf.symbols[a] == "H" and mf.symbols[b] != "H":
            heavy_of[int(a)] = int(b)
        elif mf.symbols[b] == "H" and mf.symbols[a] != "H":
            heavy_of[int(b)] = int(a)
    per_ring = np.zeros(len(rings))
    lost = 0.0
    for a in range(n_real):
        owners = members[a] if members[a] else members[heavy_of[a]] if a in heavy_of else []
        if owners:
            for k in owners:
                per_ring[k] += contrib[a] / len(owners)
        else:
            lost += contrib[a]
    for k in range(len(contrib) - n_real):
        if k < len(rings):
            per_ring[k] += contrib[n_real + k]
        else:
            lost += contrib[n_real + k]
    return per_ring, contrib, float(lost)


def ring_players(mf: MoleculeFields, rings: Sequence[Ring]) -> List[List[int]]:
    """A partition of every atom of `mf` into one group per ring, for a Shapley reading.

    :func:`ring_contributions` gives an atom that belongs to two rings half to each, which an
    exact decomposition of a linear model allows.  A Shapley game cannot share a player: a
    coalition either holds an atom or it does not.  So here an atom of two rings goes to the
    first of them, a hydrogen goes with the heavy atom it is bonded to, and the ring-centre
    pseudo-atom of ring *k* goes to ring *k*.

    Every atom of `mf` ends in exactly one group, which is what makes the reading add up: the
    coalition of all players is the whole molecule, so the values sum to its prediction minus the
    prediction of the empty molecule.  A model that is not linear in the field -- the Gaussian
    kernel layer, for one -- has no co-field and no exact per-atom decomposition, and this is how
    it is read instead (plan item PBH6).
    """
    n_real = n_real_atoms(mf)
    owner: Dict[int, int] = {}
    for k, r in enumerate(rings):
        for a in r.atoms:
            owner.setdefault(int(a), k)
    for a, b in np.asarray(mf.bonds, int).reshape(-1, 2):
        a, b = int(a), int(b)
        if mf.symbols[a] == "H" and mf.symbols[b] != "H" and b in owner:
            owner.setdefault(a, owner[b])
        elif mf.symbols[b] == "H" and mf.symbols[a] != "H" and a in owner:
            owner.setdefault(b, owner[a])
    players: List[List[int]] = [[] for _ in rings]
    for a in range(n_real):
        if a in owner:
            players[owner[a]].append(a)
    for k in range(len(mf.symbols) - n_real):
        if k < len(rings):
            players[k].append(n_real + k)
    placed = sum(len(p) for p in players)
    if placed != len(mf.symbols):
        missing = sorted(set(range(len(mf.symbols))) - {a for p in players for a in p})
        raise ValueError(
            f"{mf.name or 'the molecule'} has {len(mf.symbols) - placed} atom(s) in no ring and "
            f"bonded to none: {missing[:8]}. A Shapley reading over rings needs every atom in one "
            f"group, so this molecule is not a benzenoid the reading applies to.")
    return players


# --------------------------------------------------------------------------- #
# Count descriptors (the additivity baseline) and splits
# --------------------------------------------------------------------------- #

def count_names(atom_types: Sequence[str] = PBH_TYPES, ring_classes_: Sequence[str] = RING_CLASSES,
                ring_types: Sequence[str] = (), sextet: bool = True, homa: bool = False) -> Tuple[str, ...]:
    """The names of :func:`count_features` for the given classes."""
    names = ["n_" + t.replace(".", "") for t in atom_types] + ["n_" + c for c in ring_classes_]
    names += ["n_" + t for t in ring_types] + ["n_rings"]
    if sextet:
        names.append("n_sextet")
    if homa:
        names.append("sum_homa")
    return tuple(names)


COUNT_NAMES = count_names()


def count_features(mol: Chem.Mol, rings: Optional[Sequence[Ring]] = None,
                   atom_types: Sequence[str] = PBH_TYPES, ring_classes_: Sequence[str] = RING_CLASSES,
                   ring_types: Sequence[str] = (), sextet: bool = True, homa: bool = False,
                   carbon_only: bool = True) -> np.ndarray:
    """Counts of the atom classes, the ring classes (and ring types), the rings and the Clar
    sextets, and the sum of the HOMA indices when asked: what the indicator and ring fields
    reduce to when their attenuation goes to zero."""
    if rings is None:
        rings = ring_classes(mol, carbon_only=carbon_only)
        if sextet:
            clar_weights(mol, rings)
    types = Counter(pbh_atom_type(a) for a in mol.GetAtoms())
    rc = Counter(r.cls for r in rings)
    rt = Counter(r.ring_type for r in rings)
    out = [types.get(t, 0) for t in atom_types] + [rc.get(c, 0) for c in ring_classes_]
    out += [rt.get(t, 0) for t in ring_types] + [len(rings)]
    if sextet:
        out.append(float(sum(r.sextet for r in rings)))
    if homa:
        out.append(float(sum(0.0 if math.isnan(r.homa) else r.homa for r in rings)))
    return np.array(out, float)


def random_split(n: int, seed: int, fractions: Sequence[float] = (0.8, 0.1, 0.1)
                 ) -> Dict[str, np.ndarray]:
    """Indices of a shuffled train / validation / test split with the given fractions."""
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    return {"train": np.sort(perm[:n_tr]), "valid": np.sort(perm[n_tr:n_tr + n_va]),
            "test": np.sort(perm[n_tr + n_va:])}


def size_split(n_rings: Sequence[int], test_rings: int) -> Dict[str, np.ndarray]:
    """Train on the molecules with fewer than `test_rings` rings, test on those with exactly
    `test_rings`: the size-extrapolation split."""
    r = np.asarray(n_rings, int)
    return {"train": np.flatnonzero(r < test_rings), "valid": np.zeros(0, int),
            "test": np.flatnonzero(r == test_rings)}
