"""
Molecular co-fields (fields of regression coefficients), molecular fields and
their overlap evaluated on a rectangular grid for visualisation (isosurfaces)
and export.  Also: Gaussian-cube export.

Co-field of field type f (Baskin & Zhokhova 2013, Eq. 9; "molecular co-field"
in Baskin & Zhokhova 2019):

    C_f(r) = sum_j a_j sum_m w_fjm exp(-b_jm/2 |r - r_jm|^2)

Primal form of the model (Eq. 7):  y = sum_f h_f  INT C_f(r) X_f(r) d^3r + b
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .fields import MoleculeFields
from .model import CMFModel

BOHR = 0.52917721092


@dataclass
class Grid:
    origin: np.ndarray            # (3,) coordinates of the first grid point
    spacing: float
    shape: Tuple[int, int, int]

    @property
    def n_points(self) -> int:
        return int(np.prod(self.shape))

    def axes(self):
        return [self.origin[k] + self.spacing * np.arange(self.shape[k]) for k in range(3)]

    def points(self) -> np.ndarray:
        ax = self.axes()
        g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
        return g.reshape(-1, 3)

    @property
    def extent(self):
        return self.origin, self.origin + self.spacing * (np.array(self.shape) - 1)


def make_grid(coords: np.ndarray, margin: float = 3.0, spacing: float = 0.5, max_points: int = 1_500_000) -> Grid:
    """Rectangular grid enclosing `coords` with a margin; the spacing is coarsened if needed."""
    lo = coords.min(axis=0) - margin
    hi = coords.max(axis=0) + margin
    spacing = float(spacing)
    while True:
        shape = tuple(int(np.ceil((hi[k] - lo[k]) / spacing)) + 1 for k in range(3))
        if np.prod(shape) <= max_points:
            break
        spacing *= 1.25
    return Grid(origin=lo.astype(float), spacing=spacing, shape=shape)


def evaluate_gaussians(points: np.ndarray, centers: np.ndarray, coeffs: np.ndarray, widths: np.ndarray,
                       eps: float = 1e-5, chunk: int = 20000) -> np.ndarray:
    """sum_m coeffs[m] exp(-widths[m]/2 |p - centers[m]|^2) for every point p (sparse evaluation)."""
    centers = np.asarray(centers, float)
    coeffs = np.asarray(coeffs, float)
    widths = np.asarray(widths, float)
    keep = np.abs(coeffs) > 1e-14
    centers, coeffs, widths = centers[keep], coeffs[keep], widths[keep]
    out = np.zeros(len(points))
    if len(centers) == 0:
        return out
    bmin = max(float(widths.min()), 1e-6)
    cutoff = np.sqrt(2.0 * np.log(1.0 / eps) / bmin)
    tree_atoms = cKDTree(centers)
    for s in range(0, len(points), chunk):
        pts = points[s:s + chunk]
        tree_pts = cKDTree(pts)
        pairs = tree_pts.sparse_distance_matrix(tree_atoms, cutoff, output_type="ndarray")
        if len(pairs) == 0:
            continue
        i = pairs["i"]
        j = pairs["j"]
        d = pairs["v"]
        vals = coeffs[j] * np.exp(-0.5 * widths[j] * d * d)
        out[s:s + chunk] = np.bincount(i, weights=vals, minlength=len(pts))
    return out


def cofield_volume(model: CMFModel, f: int, grid: Grid, include_scale: bool = False) -> np.ndarray:
    """C_f(r) on the grid (shape = grid.shape).  If include_scale, multiply by h_f c_f."""
    centers, coeffs, widths = model.cofield_coefficients(f)
    v = evaluate_gaussians(grid.points(), centers, coeffs, widths)
    if include_scale:
        v *= model.fieldset.specs[f].h * model.scale[f]
    return v.reshape(grid.shape)


def _centres_and_widths(mol: MoleculeFields, spec, f: int, coeff=None):
    """(centres, coefficients, widths) of the Gaussians one field of one molecule is made of.

    An ordinary field has one Gaussian per atom; a probe field has several, so its atoms are
    repeated once per term of the expansion.  Everything that draws a field goes through
    here, which is why a probe field can be drawn, exported as a cube and subtracted from
    another molecule exactly like any other.
    """
    if getattr(spec, "is_probe", False):
        from .probes import basis_widths
        C = mol.probe_coeffs.get(spec.name)
        if C is None:
            return mol.coords, np.zeros(mol.n_atoms), np.full(mol.n_atoms, 1.0)
        b = basis_widths(C.shape[1])
        n_terms = C.shape[1]
        centres = np.repeat(mol.coords, n_terms, axis=0)
        widths = np.tile(b, mol.n_atoms)
        weights = C.reshape(-1)
        if coeff is not None:
            # a per-atom coefficient (an atom contribution, say) multiplies all of its terms
            weights = weights * np.repeat(np.asarray(coeff, float), n_terms)
        return centres, weights, widths
    widths = (np.full(mol.n_atoms, spec.alpha) if spec.kind == "gauss"
              else spec.alpha / mol.radii ** 2)
    weights = mol.weights[:, f] if coeff is None else np.asarray(coeff, float)
    return mol.coords, weights, widths


def field_volume(mol: MoleculeFields, spec, f: int, grid: Grid) -> np.ndarray:
    """Molecular field X_f(r) of `mol` (field f described by FieldSpec `spec`) on the grid; no model needed."""
    centres, weights, widths = _centres_and_widths(mol, spec, f)
    v = evaluate_gaussians(grid.points(), centres, weights, widths)
    return v.reshape(grid.shape)


def molfield_volume(model: CMFModel, mol: MoleculeFields, f: int, grid: Grid, primal_scale: bool = False) -> np.ndarray:
    """Molecular field X_f(r) of `mol` on the grid."""
    v = field_volume(mol, model.fieldset.specs[f], f, grid)
    if primal_scale:
        v = v * model.field_scale_for(f, mol)
    return v


def difference_volume(candidate: MoleculeFields, reference: MoleculeFields, spec, f: int,
                      grid: Grid) -> np.ndarray:
    """X_f(candidate) - X_f(reference) on the grid.

    Positive where the candidate puts field the reference does not have, negative where it fails
    to reproduce field the reference has.  Both molecules have to be in the same frame already,
    which is the usual condition of everything here.
    """
    return field_volume(candidate, spec, f, grid) - field_volume(reference, spec, f, grid)


def shapley_volume(mol: MoleculeFields, spec, values: np.ndarray, grid: Grid,
                   normalized: bool = True) -> np.ndarray:
    """Shapley field: the per-atom values of one field type smeared over its own Gaussians.

    Phi(r) = sum_l phi_l (b_l / 2 pi)^(3/2) exp(-b_l/2 |r - r_l|^2)

    With `normalized` each Gaussian integrates to one, so the integral of the map over space is
    the sum of the values, which is the part of the prediction attributed to this field: the same
    relation that holds between partial charges and the charge density they generate.  Unlike a
    co-field this map belongs to one molecule, because a model with a nonlinear kernel has no
    co-field to belong to.
    """
    centres, coeffs, widths = _centres_and_widths(mol, spec, 0, coeff=values)
    if normalized:
        coeffs = coeffs * (widths / (2.0 * np.pi)) ** 1.5
    v = evaluate_gaussians(grid.points(), centres, coeffs, widths)
    return v.reshape(grid.shape)


def overlap_volume(model: CMFModel, mol: MoleculeFields, f: int, grid: Grid) -> np.ndarray:
    """Integrand h_f c_f C_f(r) X~_f(r) whose integral is the contribution of field f to y."""
    C = cofield_volume(model, f, grid, include_scale=True)
    X = molfield_volume(model, mol, f, grid, primal_scale=True)
    return C * X


def isosurface(volume: np.ndarray, level: float, grid: Grid):
    """Marching cubes.  Returns (verts, faces, normals) in world coordinates or None."""
    from skimage import measure
    vmin, vmax = float(volume.min()), float(volume.max())
    if not (vmin < level < vmax):
        return None
    # pad with zeros so that surfaces crossing the grid boundary are closed
    padded = np.pad(volume, 1, mode="constant", constant_values=0.0)
    try:
        verts, faces, normals, _ = measure.marching_cubes(padded, level=level,
                                                          spacing=(grid.spacing,) * 3)
    except (ValueError, RuntimeError):
        return None
    verts = verts + (grid.origin - grid.spacing)[None, :]
    return verts, faces, normals


def export_cube(path: str, grid: Grid, volume: np.ndarray, mol: Optional[MoleculeFields] = None,
                title: str = "CMF co-field") -> None:
    """Write a Gaussian cube file (coordinates converted to Bohr)."""
    from rdkit.Chem import GetPeriodicTable
    pt = GetPeriodicTable()
    o = grid.origin / BOHR
    s = grid.spacing / BOHR
    natoms = mol.n_atoms if mol is not None else 0
    with open(path, "w") as fh:
        fh.write(f"{title}\n")
        fh.write("generated by cmfqsar\n")
        fh.write(f"{natoms:5d}{o[0]:12.6f}{o[1]:12.6f}{o[2]:12.6f}\n")
        fh.write(f"{grid.shape[0]:5d}{s:12.6f}{0.0:12.6f}{0.0:12.6f}\n")
        fh.write(f"{grid.shape[1]:5d}{0.0:12.6f}{s:12.6f}{0.0:12.6f}\n")
        fh.write(f"{grid.shape[2]:5d}{0.0:12.6f}{0.0:12.6f}{s:12.6f}\n")
        if mol is not None:
            for sym, xyz in zip(mol.symbols, mol.coords):
                z = pt.GetAtomicNumber(sym)
                b = xyz / BOHR
                fh.write(f"{z:5d}{float(z):12.6f}{b[0]:12.6f}{b[1]:12.6f}{b[2]:12.6f}\n")
        flat = volume.reshape(grid.shape)
        for ix in range(grid.shape[0]):
            for iy in range(grid.shape[1]):
                row = flat[ix, iy, :]
                for k in range(0, len(row), 6):
                    fh.write("".join(f"{v:13.5E}" for v in row[k:k + 6]) + "\n")
