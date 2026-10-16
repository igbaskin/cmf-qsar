"""Compatibility of molecular fields and co-fields with the fields of a protein pocket.

A pocket is a set of atoms, so the parameterisation that builds the fields of a ligand builds the
fields of the protein without any new physics, and the overlap of two Gaussian fields has the same
closed form whichever side it comes from.  Three questions can then be answered with numbers rather
than with a glance at a picture:

1. **How complementary is a molecule to the pocket?**  Overlap integrals between the ligand fields
   and the pocket fields, normalised to a cosine so that size drops out.  Complementarity is not
   similarity: the electrostatic term counts with a minus sign, because a positive ligand region
   should meet a negative protein region, while the hydrophobic term counts with a plus and the
   hydrogen-bond terms pair the donors of one side with the acceptors of the other.
2. **Where on the molecule does it hold?**  The same quantities on the surface of the molecule,
   which gives a score per vertex to colour the surface with, and, for electrostatics, the
   correlation of the two potentials over the surface, which is the published electrostatic
   complementarity score (Bauer & Mackey, J. Comput. Aided Mol. Des. 2009; J. Med. Chem. 62 (2019)
   3036).  Across a congeneric series this score is reported to track activity.
3. **Does the model agree with the protein?**  A co-field says where more field raises activity; the
   pocket says what it offers.  Their weighted correlation over the space the series occupies is a
   test of a model that never saw the protein.  A strong co-field lobe where the pocket offers
   nothing is a sign that the model is fitting the alignment or a correlated substituent instead of
   an interaction.

What this is not: a binding energy.  Desolvation, entropy, bridging waters and side-chain motion are
all absent, and the co-field is unconstrained wherever no molecule of the series reaches, which is
why the agreement is weighted by occupancy.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .fields import FieldSet, MoleculeFields, build_fields
from .protein import Protein, write_pdb

# field pairs that make up the score: (ligand field, pocket field, sign, label)
COMPLEMENTARY_PAIRS = [
    ("electrostatic", "electrostatic", -1.0, "electrostatic"),
    ("hydrophobic", "hydrophobic", +1.0, "hydrophobic"),
    ("hbd", "hba", +1.0, "donor to acceptor"),
    ("hba", "hbd", +1.0, "acceptor to donor"),
]


# --------------------------------------------------------------------------- #
# The pocket as a molecule
# --------------------------------------------------------------------------- #

#: formal charges of the ionisable side chains at physiological pH, by residue and atom name
IONISABLE = {("ASP", "OD2"): -1, ("GLU", "OE2"): -1, ("LYS", "NZ"): +1, ("", "OXT"): -1}


def ionise(mol) -> int:
    """Give the standard side chains their charge at pH 7, in place; returns how many were set.

    A PDB file carries no charges, so a reader leaves aspartate and glutamate protonated and
    lysine neutral.  Left alone that misplaces both the electrostatic field and, once hydrogens
    are added, the donor field, which is exactly what a complementarity score is made of.
    Histidine is left neutral, the usual assumption.
    """
    from rdkit import Chem
    n = 0
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        res = info.GetResidueName().strip().upper()
        name = info.GetName().strip().upper()
        want = IONISABLE.get((res, name), IONISABLE.get(("", name)))
        if want is not None and atom.GetFormalCharge() != want:
            atom.SetFormalCharge(want)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            n += 1
    # arginine: the positive charge belongs on the nitrogen double-bonded to CZ
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetResidueName().strip().upper() != "ARG":
            continue
        if info.GetName().strip().upper() != "CZ":
            continue
        nitrogens = [b for b in atom.GetBonds()
                     if b.GetOtherAtom(atom).GetSymbol() == "N"
                     and b.GetOtherAtom(atom).GetPDBResidueInfo() is not None
                     and b.GetOtherAtom(atom).GetPDBResidueInfo().GetName().strip().upper() != "NE"]
        if not nitrogens:
            continue
        dbl = [b for b in nitrogens if b.GetBondType() == Chem.BondType.DOUBLE]
        if not dbl:
            nitrogens[0].SetBondType(Chem.BondType.DOUBLE)
            dbl = [nitrogens[0]]
        target = dbl[0].GetOtherAtom(atom)
        if target.GetFormalCharge() != 1:
            target.SetFormalCharge(1)
            n += 1
    mol.UpdatePropertyCache(strict=False)
    return n


def pocket_molecule(protein: Protein, add_hydrogens: bool = True, ionise_residues: bool = True):
    """Build an RDKit molecule from a structure so that the field code can parameterise it.

    Crystal structures rarely carry hydrogens, and the donor field lives on the polar hydrogen, so
    they are added with coordinates unless the structure already has them.
    """
    import tempfile
    import os
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fd, path = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    try:
        write_pdb(protein, path)
        mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False, proximityBonding=True)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if mol is None:
        raise ValueError("RDKit could not read the structure")
    try:
        Chem.SanitizeMol(mol)
    except Exception:                      # noqa: BLE001 - truncated residues are expected
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                         catchErrors=True)
    if ionise_residues:
        ionise(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception:                  # noqa: BLE001 - keep what we have
            mol.UpdatePropertyCache(strict=False)
    n_h = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    if add_hydrogens and n_h < 0.2 * mol.GetNumAtoms():
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:                  # noqa: BLE001 - keep the heavy atoms if this fails
            pass
    return mol


def pocket_fields(protein: Protein, fieldset: FieldSet, add_hydrogens: bool = True,
                  ionise_residues: bool = True) -> MoleculeFields:
    """Fields of the pocket, in the same parameterisation as the molecules of the data set."""
    mol = pocket_molecule(protein, add_hydrogens=add_hydrogens, ionise_residues=ionise_residues)
    return build_fields(mol, fieldset, name=protein.name or "pocket")


# --------------------------------------------------------------------------- #
# Overlap integrals between two field sets
# --------------------------------------------------------------------------- #

def _widths(mol: MoleculeFields, spec) -> np.ndarray:
    return (np.full(mol.n_atoms, float(spec.alpha)) if spec.kind == "gauss"
            else float(spec.alpha) / mol.radii ** 2)


def field_overlap(a: MoleculeFields, spec_a, fa: int, b: MoleculeFields, spec_b, fb: int) -> float:
    """Integral over space of field `fa` of `a` times field `fb` of `b`.

    Both fields are sums of Gaussians, so the integral is a sum over atom pairs of
    (2 pi / (b1 + b2))^(3/2) exp(-b1 b2 / (2 (b1 + b2)) |x1 - x2|^2), the same closed form the
    CMF kernel uses; no grid is involved and the result is exact.
    """
    wa, wb = a.weights[:, fa], b.weights[:, fb]
    ia, ib = np.flatnonzero(np.abs(wa) > 1e-12), np.flatnonzero(np.abs(wb) > 1e-12)
    if len(ia) == 0 or len(ib) == 0:
        return 0.0
    ba, bb = _widths(a, spec_a)[ia], _widths(b, spec_b)[ib]
    s = ba[:, None] + bb[None, :]
    d2 = cdist(a.coords[ia], b.coords[ib], "sqeuclidean")
    g = (2.0 * np.pi / s) ** 1.5 * np.exp(-(ba[:, None] * bb[None, :]) / (2.0 * s) * d2)
    return float(wa[ia] @ g @ wb[ib])


def clash_count(mol: MoleculeFields, pocket: MoleculeFields, tolerance: float = 0.5) -> Tuple[int, float]:
    """Heavy-atom pairs closer than the sum of their radii minus `tolerance`, and the worst overlap.

    A complementarity score says nothing about whether the molecule fits at all; this does.
    """
    ia = [i for i, s in enumerate(mol.symbols) if s != "H"]
    ib = [i for i, s in enumerate(pocket.symbols) if s != "H"]
    if not ia or not ib:
        return 0, 0.0
    d = cdist(mol.coords[ia], pocket.coords[ib])
    lim = mol.radii[ia][:, None] + pocket.radii[ib][None, :] - tolerance
    bad = d < lim
    worst = float(np.max(lim - d)) if bad.any() else 0.0
    return int(bad.sum()), worst


@dataclass
class Complementarity:
    """Scores of one molecule against one pocket."""
    name: str
    terms: Dict[str, float] = _field(default_factory=dict)        # normalised, sign applied
    raw: Dict[str, float] = _field(default_factory=dict)          # plain overlap integrals
    burial: float = float("nan")                                  # steric overlap, normalised
    clashes: int = 0
    worst_clash: float = 0.0
    surface: Dict[str, float] = _field(default_factory=dict)      # correlations over the surface
    n_surface: int = 0

    @property
    def score(self) -> float:
        """Mean of the electrostatic, hydrophobic and hydrogen-bond terms, in [-1, 1]."""
        vals = [v for k, v in self.terms.items() if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def describe(self) -> str:
        parts = ", ".join(f"{k} {v:+.3f}" for k, v in self.terms.items())
        s = f"{self.name}: score {self.score:+.3f} ({parts}); burial {self.burial:.3f}"
        if self.clashes:
            s += f"; {self.clashes} clashes, worst {self.worst_clash:.2f} Å"
        if self.surface:
            s += "; surface " + ", ".join(f"{k} {v:+.3f}" for k, v in self.surface.items())
        return s


def complementarity(mol: MoleculeFields, pocket: MoleculeFields, fieldset: FieldSet,
                    clash_tolerance: float = 0.5) -> Complementarity:
    """Field complementarity of one molecule with the pocket, from exact overlap integrals.

    Every term is divided by the norms of the two fields, so it is a cosine in [-1, 1] and does
    not grow with the size of the molecule.
    """
    names = [s.name for s in fieldset.specs]
    specs = fieldset.specs
    out = Complementarity(name=mol.name)
    norm_l, norm_p = {}, {}
    for f, s in enumerate(specs):
        norm_l[s.name] = np.sqrt(max(field_overlap(mol, s, f, mol, s, f), 0.0))
        norm_p[s.name] = np.sqrt(max(field_overlap(pocket, s, f, pocket, s, f), 0.0))
    for lig, poc, sign, label in COMPLEMENTARY_PAIRS:
        if lig not in names or poc not in names:
            continue
        fa, fb = names.index(lig), names.index(poc)
        raw = field_overlap(mol, specs[fa], fa, pocket, specs[fb], fb)
        out.raw[label] = raw
        denom = norm_l[lig] * norm_p[poc]
        out.terms[label] = sign * raw / denom if denom > 1e-30 else float("nan")
    if "steric" in names:
        f = names.index("steric")
        raw = field_overlap(mol, specs[f], f, pocket, specs[f], f)
        out.raw["burial"] = raw
        denom = norm_l["steric"] * norm_p["steric"]
        out.burial = raw / denom if denom > 1e-30 else float("nan")
    out.clashes, out.worst_clash = clash_count(mol, pocket, tolerance=clash_tolerance)
    return out


# --------------------------------------------------------------------------- #
# On the surface of the molecule
# --------------------------------------------------------------------------- #

def shape_surface(mol: MoleculeFields, spacing: float = 0.4, level: float = 0.5,
                  margin: float = 3.0, heavy_only: bool = True):
    """A smooth van der Waals surface of a molecule, as (vertices, faces).

    The density is a sum of per-atom Gaussians scaled so that an isolated atom has the chosen
    level exactly on its van der Waals sphere, which puts the surface where a chemist expects it
    and closes smoothly between bonded atoms.
    """
    from .cofields import Grid, isosurface
    idx = [i for i in range(mol.n_atoms) if not heavy_only or mol.symbols[i] != "H"]
    if not idx:
        return None
    coords, radii = mol.coords[idx], mol.radii[idx]
    k = -np.log(max(min(level, 0.99), 1e-3))
    lo = coords.min(axis=0) - (radii.max() + margin)
    hi = coords.max(axis=0) + (radii.max() + margin)
    shape = np.maximum(np.ceil((hi - lo) / spacing).astype(int) + 1, 2)
    grid = Grid(origin=lo.astype(float), spacing=float(spacing), shape=tuple(int(s) for s in shape))
    pts = grid.points()
    d2 = cdist(pts, coords, "sqeuclidean")
    dens = np.exp(-k * d2 / (radii ** 2)[None, :]).sum(axis=1)
    surf = isosurface(dens.reshape(grid.shape), level, grid)
    if surf is None:
        return None
    return surf[0], surf[1]


def coulomb_potential(points: np.ndarray, coords: np.ndarray, charges: np.ndarray,
                      dielectric: str = "distance", eps: float = 4.0,
                      min_distance: float = 1.0) -> np.ndarray:
    """Electrostatic potential of point charges, in arbitrary but consistent units.

    ``dielectric="distance"`` uses the distance-dependent dielectric eps*r customary in
    complementarity scoring, which damps the long range; ``"constant"`` divides by eps instead.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    q = np.asarray(charges, float)
    keep = np.abs(q) > 1e-12
    if not keep.any() or len(pts) == 0:
        return np.zeros(len(pts))
    d = np.maximum(cdist(pts, np.asarray(coords, float)[keep]), min_distance)
    if dielectric == "distance":
        return (q[keep][None, :] / (eps * d * d)).sum(axis=1)
    return (q[keep][None, :] / (eps * d)).sum(axis=1)


def _pearson(a: np.ndarray, b: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    ok = np.isfinite(a) & np.isfinite(b) & (w > 0)
    if ok.sum() < 3:
        return float("nan")
    a, b, w = a[ok], b[ok], w[ok]
    wa = a - np.average(a, weights=w)
    wb = b - np.average(b, weights=w)
    va = np.average(wa * wa, weights=w)
    vb = np.average(wb * wb, weights=w)
    if va <= 1e-300 or vb <= 1e-300:
        return float("nan")
    return float(np.average(wa * wb, weights=w) / np.sqrt(va * vb))


def surface_complementarity(mol: MoleculeFields, pocket: MoleculeFields, fieldset: FieldSet,
                            spacing: float = 0.4, level: float = 0.5,
                            charge_field: str = "electrostatic"):
    """Per-vertex and overall complementarity on the surface of the molecule.

    Returns (vertices, faces, per-vertex score in [-1, 1], correlations per field).  The
    electrostatic part compares potentials rather than the Gaussian charge densities, because the
    potential is what the other side actually feels; the reported value is minus the correlation,
    so that a positive number means complementary.
    """
    surf = shape_surface(mol, spacing=spacing, level=level)
    if surf is None:
        return None
    verts, faces = surf
    names = [s.name for s in fieldset.specs]
    corr: Dict[str, float] = {}
    per_vertex = np.zeros(len(verts))
    n_terms = 0
    if charge_field in names:
        f = names.index(charge_field)
        v_lig = coulomb_potential(verts, mol.coords, mol.weights[:, f])
        v_poc = coulomb_potential(verts, pocket.coords, pocket.weights[:, f])
        corr["electrostatic"] = -_pearson(v_lig, v_poc)
        s = -v_lig * v_poc
        scale = np.percentile(np.abs(s), 95) or 1.0
        per_vertex += np.clip(s / scale, -1.0, 1.0)
        n_terms += 1
    for lig, poc, sign, label in COMPLEMENTARY_PAIRS[1:]:
        if lig not in names or poc not in names:
            continue
        fa, fb = names.index(lig), names.index(poc)
        a = _gauss_field_at(verts, mol, fieldset.specs[fa], fa)
        b = _gauss_field_at(verts, pocket, fieldset.specs[fb], fb)
        corr[label] = sign * _pearson(a, b)
        s = sign * a * b
        scale = np.percentile(np.abs(s), 95) or 1.0
        per_vertex += np.clip(s / scale, -1.0, 1.0)
        n_terms += 1
    if n_terms:
        per_vertex /= n_terms
    return verts, faces, per_vertex, corr


def _gauss_field_at(points: np.ndarray, mol: MoleculeFields, spec, f: int) -> np.ndarray:
    w = mol.weights[:, f]
    keep = np.flatnonzero(np.abs(w) > 1e-12)
    if len(keep) == 0:
        return np.zeros(len(points))
    b = _widths(mol, spec)[keep]
    d2 = cdist(np.asarray(points, float), mol.coords[keep], "sqeuclidean")
    return (w[keep][None, :] * np.exp(-0.5 * b[None, :] * d2)).sum(axis=1)


# --------------------------------------------------------------------------- #
# The model against the protein
# --------------------------------------------------------------------------- #

def occupancy(mols: Sequence[MoleculeFields], grid, k: float = 0.693) -> np.ndarray:
    """Where the series actually puts atoms, as a density on the grid.

    The co-field is unconstrained by the data wherever this is zero, so it is the weight to use
    when the co-field is compared with anything.
    """
    pts = grid.points()
    dens = np.zeros(len(pts))
    for m in mols:
        idx = [i for i in range(m.n_atoms) if m.symbols[i] != "H"]
        if not idx:
            continue
        d2 = cdist(pts, m.coords[idx], "sqeuclidean")
        dens += np.exp(-k * d2 / (m.radii[idx] ** 2)[None, :]).sum(axis=1)
    return dens.reshape(grid.shape)


def _cosine(a: np.ndarray, b: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Weighted cosine between two fields, without removing their means.

    Fields have a meaningful zero: no field means no interaction.  Mean-centring would throw that
    away and report disagreement between two maps that are positive in the same places merely
    because their gradients differ, so the cosine is the right statistic for comparing them.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    ok = np.isfinite(a) & np.isfinite(b) & (w > 0)
    if ok.sum() < 3:
        return float("nan")
    a, b, w = a[ok], b[ok], w[ok]
    na, nb = float(np.sum(w * a * a)), float(np.sum(w * b * b))
    if na <= 1e-300 or nb <= 1e-300:
        return float("nan")
    return float(np.sum(w * a * b) / np.sqrt(na * nb))


def _sign_agreement(a: np.ndarray, b: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Magnitude-weighted agreement of sign, in [-1, 1].

    A co-field is read by its sign: green means add this here.  This statistic asks only whether
    the model and the pocket want the same sign in the same place, weighted by how strongly each
    of them wants it, so two maps that agree everywhere in sign score 1 even if their shapes
    differ.  The cosine, by contrast, also demands that the shapes match.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    m = np.where(np.isfinite(w) & (w > 0), w, 0.0) * np.abs(a) * np.abs(b)
    ok = np.isfinite(a) & np.isfinite(b) & (m > 0)
    if ok.sum() < 3 or not np.isfinite(m[ok]).any():
        return float("nan")
    tot = float(np.sum(m[ok]))
    if tot <= 1e-300:
        return float("nan")
    return float(np.sum(m[ok] * np.sign(a[ok] * b[ok])) / tot)


def cofield_agreement(model, pocket: MoleculeFields, grid, weights: np.ndarray,
                      charge_field: str = "electrostatic",
                      statistic: str = "sign") -> Dict[str, float]:
    """Weighted correlation between each co-field and what the pocket offers there.

    For every field type the pocket has a preferred ligand field: the opposite of its own charge,
    the same hydrophobicity, acceptors where the ligand should donate, and space where the ligand
    should be bulky.  The co-field says what the *model* prefers.  Their agreement over the
    occupied region, in [-1, 1], is how far the model agrees with the protein.

    Three statistics are available, all in [-1, 1].  ``"sign"`` (the default) is the
    magnitude-weighted agreement of sign, which is how a co-field is actually read; ``"cosine"``
    also requires the two maps to have the same shape; ``"pearson"`` removes the means first and
    so asks only whether the variations track each other.
    """
    stat = {"sign": _sign_agreement, "cosine": _cosine, "pearson": _pearson}[statistic]
    from .cofields import cofield_volume
    names = [s.name for s in model.fieldset.specs]
    pts = grid.points()
    w = np.asarray(weights, float).reshape(-1)
    out: Dict[str, float] = {}
    pairs = [(n, n, -1.0) if n == charge_field else
             (n, {"hbd": "hba", "hba": "hbd"}.get(n, n), 1.0)
             for n in names]
    for f, (lig, poc, sign) in enumerate(pairs):
        if poc not in names:
            continue
        g = names.index(poc)
        C = cofield_volume(model, f, grid).reshape(-1)
        if lig == charge_field:
            P = coulomb_potential(pts, pocket.coords, pocket.weights[:, g])
        elif lig == "steric":
            # the steric field of a protein is positive everywhere, so comparing with it would
            # only report the sign of the co-field; what the pocket offers for bulk is *space*
            P = free_space(pts, pocket)
        else:
            P = _gauss_field_at(pts, pocket, model.fieldset.specs[g], g)
        out[model.fieldset.specs[f].label] = sign * stat(C, P, w)
    return out


def free_space(points: np.ndarray, pocket: MoleculeFields, contact: float = 3.4,
               reach: float = 3.0) -> np.ndarray:
    """How much room the pocket leaves at each point, in [-1, 1].

    Positive where a ligand atom would sit at a comfortable contact distance or further, negative
    where it would clash with the protein.  This is what a steric co-field should be compared
    with: a protein offers space, not steric field.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    heavy = [i for i, s in enumerate(pocket.symbols) if s != "H"]
    if not heavy or len(pts) == 0:
        return np.zeros(len(pts))
    from scipy.spatial import cKDTree
    d = cKDTree(pocket.coords[heavy]).query(pts, k=1)[0]
    return np.clip((d - contact) / max(reach, 1e-6), -1.0, 1.0)


def series_scores(mols: Sequence[MoleculeFields], pocket: MoleculeFields, fieldset: FieldSet,
                  activity: Optional[Sequence[float]] = None, surface: bool = False,
                  spacing: float = 0.5, progress=None) -> Tuple[List[Complementarity], Dict[str, float]]:
    """Complementarity of every molecule of a series, and how it relates to the activity.

    The correlation between the score and the activity is the honest test of whether
    complementarity means anything for this series and this structure.
    """
    res: List[Complementarity] = []
    for i, m in enumerate(mols):
        if progress:
            progress(f"Complementarity {i + 1}/{len(mols)}", i / max(len(mols), 1))
        c = complementarity(m, pocket, fieldset)
        if surface:
            s = surface_complementarity(m, pocket, fieldset, spacing=spacing)
            if s is not None:
                c.surface = s[3]
                c.n_surface = len(s[0])
        res.append(c)
    stats: Dict[str, float] = {}
    if activity is not None:
        y = np.asarray(activity, float)
        ok = np.isfinite(y)
        if ok.sum() >= 3:
            score = np.array([c.score for c in res])
            stats["r_score"] = _pearson(score[ok], y[ok])
            stats["r2_score"] = stats["r_score"] ** 2 if np.isfinite(stats["r_score"]) else float("nan")
            for key in res[0].terms:
                v = np.array([c.terms.get(key, np.nan) for c in res])
                stats["r_" + key] = _pearson(v[ok], y[ok])
            if np.isfinite([c.burial for c in res]).any():
                stats["r_burial"] = _pearson(np.array([c.burial for c in res])[ok], y[ok])
            if surface and res[0].surface:
                for key in res[0].surface:
                    v = np.array([c.surface.get(key, np.nan) for c in res])
                    stats["r_surface_" + key] = _pearson(v[ok], y[ok])
    return res, stats
