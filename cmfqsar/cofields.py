"""
Molecular co-fields (fields of regression coefficients), molecular fields and
their overlap evaluated on a rectangular grid for visualisation (isosurfaces)
and export.  Also: Gaussian-cube export.

Co-field of field type f (Baskin & Zhokhova 2013, Eq. 9; "molecular co-field"
in Baskin & Zhokhova 2019)::

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


#: pairs held in one neighbour list, which scipy returns as (point, centre, distance) in
#: 24 bytes each: this cap is about half a gigabyte.  It exists because the chunk below counts
#: *points*, and what costs memory is points times the centres each one sees -- a few hundred
#: for a molecule's co-field, over a hundred thousand for the co-field of a periodic model,
#: whose basis is the image-expanded training table.
MAX_PAIRS = 20_000_000


def points_per_chunk(points: np.ndarray, tree, cutoff: float, chunk: int,
                     sample: int = 16) -> int:
    """How many points to put in one neighbour list, from how many centres a point sees.

    The count is measured on a few points rather than assumed, because it is the one quantity
    that differs by orders of magnitude between a molecular co-field and a periodic one.
    """
    n = len(points)
    if n == 0:
        return max(1, int(chunk))
    step = max(1, n // int(sample))
    probe = np.asarray(points, float)[::step][:int(sample)]
    seen = tree.query_ball_point(probe, cutoff, return_length=True)
    per_point = max(1, int(np.mean(np.asarray(seen, float))))
    return max(1, min(int(chunk), int(MAX_PAIRS // per_point)))


def evaluate_gaussians(points: np.ndarray, centers: np.ndarray, coeffs: np.ndarray, widths: np.ndarray,
                       eps: float = 1e-5, chunk: int = 20000) -> np.ndarray:
    """``sum_m coeffs[m] exp(-widths[m]/2 |p - centers[m]|^2)`` for every point p (sparse evaluation).

    `chunk` is an upper bound on the points evaluated at once; the real one is
    :data:`MAX_PAIRS`, which bounds the neighbour list of a chunk and so the memory this needs,
    whatever the density of the centres.
    """
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
    chunk = points_per_chunk(points, tree_atoms, cutoff, chunk)
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
    """Shapley field: the per-atom values of one field type smeared over its own Gaussians::

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

    The Laplacian-regularised ridge (method ``lapkrr``) is the same kind of map, with two
    differences: its system matrix carries the smoothness penalty, and only the molecules that
    have an activity carry noise, so G has one column per labelled molecule rather than per
    training molecule.  ``G G^T`` is the same size either way, and so is everything downstream.
    """
    if model.method not in ("krr", "gp", "lapkrr"):
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
    if model.method == "lapkrr":
        from .spectral import knn_laplacian
        lab = np.isfinite(np.asarray(model.y, float))
        j = lab.astype(float)
        M = np.zeros((n, n))
        M[lab] = K[lab]
        M[np.diag_indices(n)] += lam
        gamma = float(model.params.get("manifold", 1.0))
        if gamma != 0.0:
            M += gamma * (knn_laplacian(
                K, neighbours=int(model.params.get("neighbours", 10))) @ K)
        cols = np.flatnonzero(lab)                   # only a measured activity carries noise
        E = np.zeros((n, len(cols)))
        E[cols, np.arange(len(cols))] = 1.0
        if not model.fit_bias:
            A, rhs = M, E
        else:
            A = np.zeros((n + 1, n + 1))
            A[:n, :n] = M
            A[:n, n] = j
            A[n, :n] = j @ K
            A[n, n] = float(np.count_nonzero(lab))
            rhs = np.vstack([E, np.ones((1, len(cols)))])
    elif not model.fit_bias:
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
                        chunk: int = 0, basis=None) -> np.ndarray:
    """(n_train, n_points): the normalised field of every training molecule at every point.

    This is the basis the co-field is a combination of: ``C_f = coef @ X``, and the variance of
    the co-field is read off the same matrix, so the two cannot disagree.  `basis` replaces the
    Gaussians of :meth:`CMFModel.cofield_basis` by another set over the same training molecules,
    which is how :func:`atom_effect` reads the same quantity convolved with one atom.
    """
    centres, coeffs, widths, index = (model.cofield_basis(f) if basis is None else basis)
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


def cofield_at_points(model: CMFModel, f: int, points: np.ndarray,
                      include_scale: bool = True) -> np.ndarray:
    """C_f at a list of points rather than on a grid.

    A co-field read at a few places a chemist can name -- the ortho, meta and para positions of a
    ring, the four corners of a pocket -- is a table, not a picture, and this is what fills it.
    """
    centres, coeffs, widths = model.cofield_coefficients(f)
    v = evaluate_gaussians(np.asarray(points, float), centres, coeffs, widths)
    if include_scale:
        v = v * model.fieldset.specs[f].h * model.scale[f]
    return v


def cofield_covariance_at(model: CMFModel, f: int, points: np.ndarray,
                          sigma: Optional[float] = None, include_scale: bool = True,
                          chunk: int = 0, factor: Optional[np.ndarray] = None) -> np.ndarray:
    """Cov[C_f] at a list of points, under noise of level `sigma` in the activities.

    The diagonal is what :func:`cofield_variance` gives on a grid.  The off-diagonal terms are
    what a *difference* between two places needs: the co-field at two points a few angstrom
    apart is built from the same training fields, so the two values are correlated and the error
    of their difference is not the two errors added -- usually it is far smaller.  Without this
    a comparison between two positions would be judged against the wrong error bar.

    `factor` is the matrix :func:`dual_covariance_factor` returns.  It does not depend on the
    field, and computing it costs a solve with as many right-hand sides as there are training
    molecules, so a caller reading the co-fields of twenty fields at the same points should
    compute it once and pass it here instead of paying for it twenty times.
    """
    s, _src = noise_level(model, sigma)
    G = dual_covariance_factor(model) if factor is None else np.asarray(factor, float)
    V = G.T @ per_molecule_fields(model, f, np.asarray(points, float), chunk=chunk)
    cov = (s * s) * (V.T @ V)
    if include_scale:
        cov = cov * (model.fieldset.specs[f].h * model.scale[f]) ** 2
    return cov


def contrast_at(model: CMFModel, f: int, points: np.ndarray, weights: Sequence[float],
                sigma: Optional[float] = None, include_scale: bool = True,
                factor: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """A weighted combination of the co-field at a few points, and its standard deviation.

    ``weights`` picks the comparison: (0, 0, -1, 1) at the four ring positions is "para minus
    meta".  The standard deviation uses the whole covariance of :func:`cofield_covariance_at`, so
    the correlation between the points is taken into account rather than ignored.  `factor` is
    passed through, so several comparisons on one model share the one expensive solve.
    """
    w = np.asarray(weights, float)
    v = cofield_at_points(model, f, points, include_scale=include_scale)
    cov = cofield_covariance_at(model, f, points, sigma=sigma, include_scale=include_scale,
                                factor=factor)
    return float(w @ v), float(np.sqrt(max(float(w @ cov @ w), 0.0)))


def probe_basis(centres, coeffs, widths, width: float, weight: float = 1.0):
    """The same Gaussian basis, convolved with one atom's own Gaussian of width `width`.

    The overlap of two Gaussians is a Gaussian, which is what makes every integral in this
    program closed form:

        int exp(-b/2 |r - c|^2) exp(-bp/2 |r - p|^2) dr
            = (2 pi / (b + bp))^(3/2) exp(-(b bp) / (2 (b + bp)) |p - c|^2),

    the same expression :mod:`cmfqsar.kernels` sums over atom pairs.  So a basis evaluated at a
    point becomes the overlap with an atom placed there by scaling every coefficient and widening
    every width, and nothing else in this module has to change.
    """
    b = np.asarray(widths, float)
    bp = float(width)
    s = b + bp
    return (np.asarray(centres, float),
            np.asarray(coeffs, float) * float(weight) * (2.0 * np.pi / s) ** 1.5,
            b * bp / s)


def _probe_width(model: CMFModel, f: int, width: Optional[float]) -> float:
    if width is not None:
        return float(width)
    spec = model.fieldset.specs[f]
    if spec.kind != "gauss":
        raise ValueError(f"the '{spec.name}' field puts a different Gaussian on every atom "
                         f"({spec.kind}), so there is no single width for a probe atom: pass one")
    return float(spec.alpha)


def atom_effect(model: CMFModel, f: int, points: np.ndarray, width: Optional[float] = None,
                weight: float = 1.0, include_scale: bool = True) -> np.ndarray:
    """What one atom placed at each point would add to the prediction.

    A co-field says what the prediction gains per unit of field at a point, which is not a number
    a chemist can check against anything: a field is not something one can put somewhere.  An atom
    is.  This is the overlap of the co-field with one atom's own Gaussian, so the result is in the
    units of the measured value -- ppm for a shift, pKa units for an acidity -- and can be read
    beside a table of measured substituent effects.

    For a continuous indicator field the weight of an atom of its type is 1, so the default
    ``weight=1`` is one such atom; for the electrostatic field the weight is a partial charge and
    the caller should pass it.  `width` defaults to the field's own attenuation, which is the
    Gaussian the kernel puts on an atom of that field.
    """
    centres, coeffs, widths = model.cofield_coefficients(f)
    c, k, b = probe_basis(centres, coeffs, widths, _probe_width(model, f, width), weight)
    v = evaluate_gaussians(np.asarray(points, float), c, k, b)
    if include_scale:
        v = v * model.fieldset.specs[f].h * model.scale[f]
    return v


def atom_effect_covariance(model: CMFModel, f: int, points: np.ndarray,
                           width: Optional[float] = None, weight: float = 1.0,
                           sigma: Optional[float] = None, include_scale: bool = True,
                           chunk: int = 0, factor: Optional[np.ndarray] = None) -> np.ndarray:
    """Cov of :func:`atom_effect` under noise of level `sigma` in the measured values."""
    s, _src = noise_level(model, sigma)
    G = dual_covariance_factor(model) if factor is None else np.asarray(factor, float)
    centres, coeffs, widths, index = model.cofield_basis(f)
    c, k, b = probe_basis(centres, coeffs, widths, _probe_width(model, f, width), weight)
    X = per_molecule_fields(model, f, np.asarray(points, float), chunk=chunk,
                            basis=(c, k, b, index))
    V = G.T @ X
    cov = (s * s) * (V.T @ V)
    if include_scale:
        cov = cov * (model.fieldset.specs[f].h * model.scale[f]) ** 2
    return cov


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


# --------------------------------------------------------------------------- #
# A mask over the co-field, and the co-field restricted to it (plan item MC5)
# --------------------------------------------------------------------------- #

@dataclass
class CofieldMask:
    """A soft region of space: a sum of Gaussians on the atoms the model rewards most.

    Everything else in this program compares two molecules over the whole of both of them, which
    is right for a congeneric series and wrong for a replacement scaffold.  A candidate that
    carries the same field where the co-field has weight is the one worth making, and what it
    does elsewhere is its own business -- that is the whole content of plan item MC5, and this is
    the region "elsewhere" is measured against.

    `centres`, `weights` and `widths` are a Gaussian expansion like any other in this module, so
    the mask can be drawn, exported as a cube and multiplied into a co-field with
    :func:`cmfqsar.kernels.gaussian_product`.  `atoms` and `contributions` say which atoms it was
    built from and what each contributed, because a mask nobody can explain is not useful.
    """
    centres: np.ndarray
    weights: np.ndarray
    widths: np.ndarray
    atoms: np.ndarray = None              # (k,) atom index within its molecule
    molecules: np.ndarray = None          # (k,) which molecule each atom came from
    contributions: np.ndarray = None      # (k,) the contribution that selected it

    def expansion(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.centres, self.weights, self.widths

    def volume(self, grid: Grid) -> np.ndarray:
        """The mask itself on a grid, so that it can be drawn beside the co-field it cuts."""
        return evaluate_gaussians(grid.points(), self.centres, self.weights,
                                  self.widths).reshape(grid.shape)

    def describe(self) -> str:
        lines = [f"a mask of {len(self.centres)} Gaussians, width "
                 f"{np.mean(self.widths):.3g}, peak {self.peak():.3g}"]
        if self.contributions is not None:
            for i in np.argsort(-np.abs(self.contributions)):
                lines.append(f"  molecule {int(self.molecules[i])} atom {int(self.atoms[i])}: "
                             f"contribution {self.contributions[i]:+.4f}")
        return "\n".join(lines)

    def peak(self) -> float:
        """The largest value the mask takes at one of its own centres."""
        if not len(self.centres):
            return 0.0
        return float(np.max(evaluate_gaussians(self.centres, self.centres, self.weights,
                                               self.widths)))


def mask_width(model: CMFModel, width: Optional[float] = None) -> float:
    """The width of the mask's Gaussians: by default the attenuation the kernel itself uses.

    A mask is a region and not a field, so it has one width for every field of the model.  The
    default is the mean attenuation of the fields that put one Gaussian of a fixed width on an
    atom, which is the Gaussian the kernel compares atoms with; a field whose width is per atom
    (the radius-scaled steric field) has no single value to contribute and is left out.
    """
    if width is not None:
        return float(width)
    alphas = [float(s.alpha) for s in model.fieldset.specs if s.kind == "gauss"]
    return float(np.mean(alphas)) if alphas else 1.0


def top_atom_mask(model: CMFModel, mols, k: int = 10, width: Optional[float] = None,
                  normalise: bool = True, per_field: bool = False) -> CofieldMask:
    """A mask on the `k` atoms of `mols` whose contribution to the prediction is largest.

    `mols` is one :class:`~cmfqsar.fields.MoleculeFields` or several -- the training molecules the
    mask should describe, usually the series the model was fitted on.  Atoms are ranked by the
    absolute value of their contribution (decomposition rule 3), pooled over the molecules given,
    so a superimposed series votes on the region rather than one molecule choosing it.

    With `normalise` the weights are scaled so that the mask's largest value is 1, which keeps a
    masked overlap in the units of the unmasked one and makes the two comparable.
    """
    if isinstance(mols, MoleculeFields):
        mols = [mols]
    mols = list(mols)
    if not mols:
        raise ValueError("a mask needs at least one molecule to read the contributions of")
    coords, contrib, atom_ix, mol_ix = [], [], [], []
    for j, mol in enumerate(mols):
        c = model.atom_contributions(mol, per_field=False)
        coords.append(np.asarray(mol.coords, float))
        contrib.append(np.asarray(c, float))
        atom_ix.append(np.arange(len(c)))
        mol_ix.append(np.full(len(c), j))
    coords = np.vstack(coords)
    contrib = np.concatenate(contrib)
    atom_ix = np.concatenate(atom_ix)
    mol_ix = np.concatenate(mol_ix)
    k = int(min(max(k, 1), len(contrib)))
    pick = np.argsort(-np.abs(contrib))[:k]
    b = mask_width(model, width)
    mask = CofieldMask(centres=coords[pick], weights=np.ones(k), widths=np.full(k, b),
                       atoms=atom_ix[pick], molecules=mol_ix[pick],
                       contributions=contrib[pick])
    if normalise:
        peak = mask.peak()
        if peak > 1e-300:
            mask.weights = mask.weights / peak
    return mask


def masked_cofield(model: CMFModel, f: int, mask: CofieldMask, tol: float = 1e-9):
    """The Gaussians of ``C_f(r) m(r)``: the co-field where the mask is, and nothing elsewhere.

    A product of two Gaussian expansions is a Gaussian expansion
    (:func:`cmfqsar.kernels.gaussian_product`), so the masked co-field is the same kind of object
    as the co-field and every reader of one works on it.  `tol` drops the product terms whose
    weight is a negligible fraction of the largest, which removes the pairs that are far apart.
    """
    from .kernels import gaussian_product

    centres, coeffs, widths = model.cofield_coefficients(f)
    mc, mw, mb = mask.expansion()
    return gaussian_product(centres, coeffs, widths, mc, mw, mb, tol=tol)
