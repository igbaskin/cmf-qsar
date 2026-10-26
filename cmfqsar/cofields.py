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


# --------------------------------------------------------------------------- #
# How much of a co-field is noise
# --------------------------------------------------------------------------- #

@dataclass
class CofieldError:
    """A co-field with its error bar: the mean map, its standard deviation, and the ratio."""

    mean: np.ndarray            # C_f(r) on the grid
    sd: np.ndarray              # standard deviation of C_f(r), same shape
    sigma: float                # the noise level in the activities this was computed from
    sigma_from: str             # where that number came from
    z: float = 2.0              # the threshold the mask uses

    @property
    def zscore(self) -> np.ndarray:
        """C_f(r) / sd(r), zero where the training molecules put no field at all."""
        return np.divide(self.mean, self.sd, out=np.zeros_like(self.mean), where=self.sd > 0)

    @property
    def masked(self) -> np.ndarray:
        """The co-field with everything below `z` standard deviations set to zero."""
        return mask_noise(self.mean, self.sd, self.z)

    @property
    def surviving(self) -> float:
        """The fraction of the drawn volume the mask keeps.

        "Drawn" is the part a picture shows at all -- a tenth of the peak and above, which is
        well below any isosurface anyone draws -- so this is the fraction of what a reader would
        have seen that is more than `z` standard deviations from zero.
        """
        peak = float(np.max(np.abs(self.mean))) if self.mean.size else 0.0
        drawn = np.abs(self.mean) >= 0.1 * (peak or 1.0)
        return float(np.mean(np.abs(self.zscore)[drawn] >= self.z)) if drawn.any() else 0.0

    def describe(self) -> str:
        return (f"co-field error from sigma = {self.sigma:.3f} ({self.sigma_from}): "
                f"{100 * self.surviving:.0f} % of the drawn volume lies more than "
                f"{self.z:g} standard deviations from zero")


def dual_covariance_factor(model: CMFModel) -> np.ndarray:
    """G, with Cov[a] = sigma^2 G G^T, for the dual coefficients of a kernel ridge fit.

    The coefficients are a linear map of the activities -- with a bias, the bordered system of
    :func:`cmfqsar.model.krr_fit`, whose inverse's top-left block is this G; without one, simply
    (K + lam I)^-1.  So activities carrying independent noise of variance sigma^2 give
    coefficients carrying covariance sigma^2 G G^T, and every linear functional of them carries
    the matching variance.  Nothing here is a Bayesian posterior: see :func:`cofield_variance`.
    """
    if model.method not in ("krr", "gp"):
        raise ValueError(f"the dual coefficients of a '{model.method}' model come out of a "
                         "quadratic program and are not a linear map of the activities, so this "
                         "covariance does not describe them; fit krr or gp for co-field errors")
    if model.train is None:
        raise ValueError("the model carries no training molecules to propagate noise through")
    if model.K_fields_train is None:
        # a model read from a file stores its training atoms, not their kernels: compute them
        # again, with the stored self-kernels and scale untouched, as predict_with_variance does
        model.K_fields_train = model._field_kernels(model.train, model.train,
                                                    model.self_diag, model.self_diag)
    K = model.joint_kernel(model.K_fields_train, model.train.descriptors)
    n = K.shape[0]
    lam = float(model.params.get("lam", 1.0))
    if not model.fit_bias:
        A, rhs = K + lam * np.eye(n), np.eye(n)
    else:
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = K + lam * np.eye(n)
        A[:n, n] = 1.0
        A[n, :n] = 1.0
        rhs = np.vstack([np.eye(n), np.zeros((1, n))])
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return np.asarray(sol[:n, :], float)


def noise_level(model: CMFModel, sigma: Optional[float] = None) -> Tuple[float, str]:
    """The noise level to propagate, and where it came from.

    The out-of-fold error is the default and the reliable estimate: it is what the model's own
    predictions miss by, and it includes the part of the error that is the model's fault rather
    than the assay's, which errs in the conservative direction.  The residual of the fit would be
    far too small -- a model with a small lambda reproduces its training set almost exactly,
    which says nothing about the noise in it.
    """
    if sigma is not None:
        return float(sigma), "given"
    st = getattr(model, "stats", None)
    for attr, how in (("rmse_cv", "RMSEcv, the out-of-fold error"),
                      ("rmse_fit", "RMSE of the fit, which understates the noise")):
        v = float(getattr(st, attr, np.nan)) if st is not None else float("nan")
        if np.isfinite(v) and v > 0:
            return v, how
    raise ValueError("no noise level to propagate: cross-validate the model, or pass sigma")


def per_molecule_fields(model: CMFModel, f: int, points: np.ndarray,
                        chunk: int = 0) -> np.ndarray:
    """(n_train, n_points): the normalised field of every training molecule at every point.

    This is the basis the co-field is a combination of: ``C_f = coef @ X``, and the variance of
    the co-field is read off the same matrix, so the two cannot disagree.
    """
    centres, coeffs, widths, index = model.cofield_basis(f)
    centres = np.asarray(centres, float)
    coeffs = np.asarray(coeffs, float)
    widths = np.asarray(widths, float)
    index = np.asarray(index, int)
    n = model.train.n_mol
    points = np.atleast_2d(np.asarray(points, float))
    out = np.zeros((n, len(points)))
    keep = np.abs(coeffs) > 1e-14
    centres, coeffs, widths, index = centres[keep], coeffs[keep], widths[keep], index[keep]
    if len(centres) == 0 or len(points) == 0:
        return out
    cutoff = np.sqrt(2.0 * np.log(1e5) / max(float(widths.min()), 1e-6))
    tree_atoms = cKDTree(centres)
    if chunk <= 0:
        chunk = _point_chunk(n)
    for s in range(0, len(points), chunk):
        pts = points[s:s + chunk]
        pairs = cKDTree(pts).sparse_distance_matrix(tree_atoms, cutoff, output_type="ndarray")
        if len(pairs) == 0:
            continue
        i, j, d = pairs["i"], pairs["j"], pairs["v"]
        vals = coeffs[j] * np.exp(-0.5 * widths[j] * d * d)
        flat = np.bincount(index[j] * len(pts) + i, weights=vals, minlength=n * len(pts))
        out[:, s:s + chunk] = flat.reshape(n, len(pts))
    return out


def _point_chunk(n_mol: int) -> int:
    """Points per block, so that one (n_train, points) block stays a few tens of megabytes."""
    return int(max(1000, min(20000, 4.0e6 / max(n_mol, 1))))


def cofield_variance(model: CMFModel, f: int, grid: Grid, sigma: Optional[float] = None,
                     include_scale: bool = False, chunk: int = 0) -> np.ndarray:
    """Var[C_f(r)] on the grid: how far the co-field moves when the activities carry noise.

    The co-field is linear in the dual coefficients, ``C_f(r) = sum_j a_j x_j(r)``, and the
    coefficients are a linear map of the activities, so with ``Cov[a] = sigma^2 G G^T``

        Var[C_f(r)] = sigma^2 || G^T x(r) ||^2

    in closed form: one matrix product per block of grid points, no resampling and no refitting.

    **What this is and is not.**  It is tempting to call it the posterior variance of the
    co-field, and the plan item that asked for it did.  It is not, and the reason is worth
    stating.  Under the Gaussian process whose kernel this is, the prior on the field weight
    function is white noise on L^2: its value at a point is not a random variable at all, and the
    pointwise posterior variance is infinite everywhere.  What the prior does allow is the
    variance of ``<w, g>`` for a probe field g of non-zero width, which smooths the point into a
    Gaussian -- that is exactly the GP variance of a prediction for a one-atom molecule placed
    there, and :meth:`CMFModel.predict_with_variance` already computes it.

    What is computed here is the other half of the question, with the training molecules and the
    kernel held fixed and the noise in the activities propagated.  It answers *how much of this
    lobe would survive the error in the measurements*, which is the question a co-field picture
    raises.  It says nothing about regions where the training set put no field at all:
    there the co-field and its error are both zero, and the ratio of the two is not informative.
    """
    s, _src = noise_level(model, sigma)
    G = dual_covariance_factor(model)
    pts = grid.points()
    var = np.zeros(len(pts))
    if chunk <= 0:
        chunk = _point_chunk(model.train.n_mol)
    for a in range(0, len(pts), chunk):
        V = G.T @ per_molecule_fields(model, f, pts[a:a + chunk], chunk=chunk)
        var[a:a + chunk] = (s * s) * np.einsum("ij,ij->j", V, V)
    if include_scale:
        var *= (model.fieldset.specs[f].h * model.scale[f]) ** 2
    return var.reshape(grid.shape)


def cofield_with_error(model: CMFModel, f: int, grid: Grid, sigma: Optional[float] = None,
                       include_scale: bool = False, z: float = 2.0,
                       chunk: int = 0) -> CofieldError:
    """The co-field, its standard deviation and the threshold that decides what is drawn."""
    s, src = noise_level(model, sigma)
    mean = cofield_volume(model, f, grid, include_scale=include_scale)
    var = cofield_variance(model, f, grid, sigma=s, include_scale=include_scale, chunk=chunk)
    return CofieldError(mean=mean, sd=np.sqrt(np.maximum(var, 0.0)), sigma=s, sigma_from=src,
                        z=float(z))


def mask_noise(volume: np.ndarray, sd: np.ndarray, z: float = 2.0) -> np.ndarray:
    """`volume` with everything below `z` standard deviations of zero set to zero.

    Zero rather than nan, so the result is still a field an isosurface can be drawn on: a lobe
    keeps its shape where it is significant and is cut where it is not.  Where the error is zero
    the value is kept, because there the co-field is zero as well and there is nothing to cut.
    """
    volume = np.asarray(volume, float)
    sd = np.asarray(sd, float)
    return np.where((sd <= 0) | (np.abs(volume) >= z * sd), volume, 0.0)
