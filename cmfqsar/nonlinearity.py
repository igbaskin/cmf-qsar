"""
The co-field of a model with a nonlinear kernel layer: one per molecule, and how far they differ.

A model with a kernel layer (:attr:`CMFModel.kernel_layer`) predicts

    y(x) = sum_j a_j g( K(x, x_j), K(x, x), K(x_j, x_j) ) + b,                              (1)

where K is the linear joint kernel -- the overlap integrals of the normalised fields, mixed over
the field types -- and g is the layer: the identity for the linear model, exp(-gamma d^2) for
the Gaussian layer, the arc-cosine function of Cho and Saul for the arc-cosine layers.  The
prediction is then no longer the overlap of the field with one fixed co-field, which is why
:meth:`CMFModel.cofield_coefficients` refuses such a model.  It is still a differentiable
function of the field, and its derivative with respect to the normalised field of type f, taken
at molecule x, is a field:

    C_fx(r) = sum_j a_j g_s(x, j) x_fj(r)  +  2 [ sum_j a_j g_n(x, j) ] x_fx(r),              (2)

with g_s = dg/dK(x, x_j), g_n = dg/dK(x, x), x_fj the normalised field of training molecule j
(the basis of the ordinary co-field, :meth:`CMFModel.training_field_basis`) and x_fx the
normalised field of x itself.  This is the **local co-field**: the co-field of the linear model
tangent to (1) at x.  The prediction of a molecule close to x changes by the overlap of C_fx
with the change of the normalised field, summed over the field types with the usual factors
h_f c_f.  For the linear layer g_s = 1 and g_n = 0, and (2) is the ordinary co-field whatever x
is; for the Gaussian layer g_s = 2 gamma g and g_n = -gamma g, so the training molecules are
weighted by their kernel similarity to x and the field is taken relative to x's own.

The second term of (2) lies along x's own field.  With cosine-normalised kernels (the default)
K(x, x) does not change when the normalised field moves, so the term has no effect on any
prediction and is left out -- as the ordinary co-field leaves out the dependence of the
normalisation on the field.  Without normalisation it is kept.

Two readings of (2) are built on top of the map itself:

* the **spread** of the local co-fields over the training molecules (:func:`cofield_spread`):
  their mean, which is the one co-field to draw for a nonlinear model, and their standard
  deviation point by point, which is a map of where the effect of field depends on the rest of
  the molecule -- zero where the model is additive, large where it is not;
* a **non-linearity index** per field type (:func:`nonlinearity_indices`), the share of the
  squared norm of the local co-fields that their differences from the mean make up, computed in
  closed form from the training kernel matrix: 0 for a linear model, 1 when the local co-fields
  have nothing in common.  The mean pairwise cosine between them is reported beside it.

What the maps are not.  They describe the tangent models, not the chemistry by themselves: the
Gaussian layer makes every model nonlinear to a degree set by its bandwidth, and the spread is
nonzero even when the measured relation is additive, as long as the bandwidth is finite.  Read
the spread against the linear model's cross-validated error (if the linear kernel predicts as
well, the spread is a property of the kernel) and check that it survives the folds, as the noise
band of :mod:`cmfqsar.cofields` is checked.

Second derivatives -- how the effect of field at one place depends on the field at another --
are a function of two places, not a field.  The second half of this module takes them with
respect to the amplitudes of probe atoms, which keeps them finite and exact under the kernel
normalisation: the **pair-interaction field** (one probe fixed, the other scanned over a grid),
the **curvature field** (its diagonal), and the **atom interaction matrix** of one molecule with
its eigenmodes, the patterns of atoms the model reads together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .fields import AtomTable, MoleculeFields
from .model import CMFModel

__all__ = ["layer_derivatives", "local_duals", "local_cofield_coefficients",
           "local_cofield_volume", "local_cofield_at_points", "CofieldSpread", "cofield_spread",
           "NonlinearityIndex", "nonlinearity_indices",
           "layer_second_derivatives", "ProbeSet", "probe_set", "atom_probes", "Tangent",
           "probe_gradient", "probe_hessian", "pair_interaction_field", "gradient_field",
           "curvature_field",
           "atom_interactions", "InteractionMode", "interaction_modes"]


# ------------------------------------------------------------------------------ the layer

def _arccos_derivatives(K: np.ndarray, dp: np.ndarray, dq: np.ndarray):
    """The degree-1 arc-cosine function and its two partial derivatives.

    With n = K(x, x), m = K(x_j, x_j), s = K(x, x_j) and cos(theta) = s / sqrt(n m)::

        A(s, n, m) = sqrt(n m) / pi * ( sin(theta) + (pi - theta) cos(theta) )
        dA/ds      = (pi - theta) / pi
        dA/dn      = sqrt(m / n) sin(theta) / (2 pi)

    `dp` is a column (the rows' self-kernels n), `dq` a row (the columns' self-kernels m), as
    :meth:`CMFModel.apply_layer` takes them.
    """
    npn = np.sqrt(np.maximum(dp, 1e-300))
    nqn = np.sqrt(np.maximum(dq, 1e-300))
    cos = np.clip(K / (npn * nqn), -1.0, 1.0)
    th = np.arccos(cos)
    sin = np.sin(th)
    val = (npn * nqn) / np.pi * (sin + (np.pi - th) * cos)
    d_s = (np.pi - th) / np.pi
    d_n = (nqn / npn) * sin / (2.0 * np.pi)
    return val, d_s, d_n


def layer_derivatives(model: CMFModel, K: np.ndarray, diag_p: np.ndarray,
                      diag_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The layer g on a block of the linear kernel, and its derivatives g_s and g_n of Eq. (2).

    `K` holds the linear joint kernel between the rows (the molecules the derivative is taken
    at, self-kernels `diag_p`) and the columns (the training molecules, self-kernels `diag_q`).
    The first array equals ``model.apply_layer(K, diag_p, diag_q)``; the second is the
    derivative with respect to the entry K(x, x_j); the third the derivative with respect to the
    row's own self-kernel K(x, x), the other arguments held fixed.
    """
    K = np.asarray(K, float)
    dp = np.asarray(diag_p, float)[:, None]
    dq = np.asarray(diag_q, float)[None, :]
    if model.is_linear:
        return K.copy(), np.ones_like(K), np.zeros_like(K)
    layer = model.kernel_layer
    if layer == "gauss":
        g = float(model.params.get("gamma", 1.0)) * float(getattr(model, "gamma_scale_", None) or 1.0)
        d2 = np.maximum(dp + dq - 2.0 * K, 0.0)
        val = np.exp(-g * d2)
        return val, 2.0 * g * val, -g * val
    if layer in ("arccos1", "arccos2"):
        val, v_s, v_n = _arccos_derivatives(K, dp, dq)
        if layer == "arccos2":
            # the second application takes the first's value where the kernel was, with the
            # same self-kernels (the arc-cosine self-kernel equals the base one)
            val2, w_s, w_n = _arccos_derivatives(val, dp, dq)
            return val2, w_s * v_s, w_s * v_n + w_n
        return val, v_s, v_n
    raise ValueError(f"unknown kernel layer {layer}")


# ------------------------------------------------------------------------- the coefficients

def _require_expansion(model: CMFModel, what: str) -> None:
    """The local co-field needs a score that is a sum over the training molecules, and fields."""
    model._require_fields(what)
    if model.is_low_rank:
        raise ValueError(f"{what} are not available for a model fitted in the space of its "
                         "landmarks, which has no coefficient per training molecule")
    oc = getattr(model.learner_, "oc_", None) if model.learner_ is not None else None
    if oc is not None and not oc.is_linear:
        raise ValueError(f"{what} are defined for a score that is a sum over the training "
                         f"molecules; '{oc.label}' is not one ({oc.why_not_linear})")
    if model.coef is None or model.train is None:
        raise RuntimeError("model is not fitted")


def local_duals(model: CMFModel, table: Optional[AtomTable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The coefficients of Eq. (2) for the molecules of `table`, or for the training set.

    Returns ``(B, b0)``: ``B[x, j] = a_j g_s(x, j)`` weighs the normalised field of training
    molecule j in the local co-field at x, and ``b0[x] = 2 sum_j a_j g_n(x, j)`` weighs x's own
    normalised field.  ``b0`` is zero for a model with normalised kernels (see the module
    docstring) and for the linear layer, where ``B`` has the dual coefficients in every row.
    The same two arrays serve every field type, because the layer acts on the joint kernel.
    """
    _require_expansion(model, "local co-fields")
    coef = np.asarray(model.coef, float)
    dq = getattr(model, "base_diag_", None)
    if table is None:
        K_lin = model._linear_train_kernel()
        dp = np.diag(K_lin).copy() if dq is None else np.asarray(dq, float)
    else:
        Kf = model.test_kernels(table)
        K_lin = model.joint_kernel(Kf, table.descriptors)
        dp = model.joint_self(table)
    if dq is None:
        dq = np.diag(model._linear_train_kernel()).copy()
    _g, g_s, g_n = layer_derivatives(model, K_lin, dp, dq)
    B = g_s * coef[None, :]
    if model.normalize:
        b0 = np.zeros(K_lin.shape[0])
    else:
        b0 = 2.0 * (g_n @ coef)
    return B, b0


def local_cofield_coefficients(model: CMFModel, f: int, mol: MoleculeFields
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centres, coefficients and widths of the Gaussians composing the local co-field C_fx.

    The same three arrays :meth:`CMFModel.cofield_coefficients` returns, in the same
    convention: the field-scaling constant c_f and the mixing coefficient h_f are not included.
    For a linear model the result is the ordinary co-field whatever `mol` is.  Without
    normalisation the molecule's own Gaussians are appended with the weight ``b0`` of Eq. (2).
    """
    table = AtomTable.from_molecules([mol])
    B, b0 = local_duals(model, table)
    centres, coeffs, widths, index = model.training_field_basis(f)
    c = np.asarray(coeffs, float) * B[0][np.asarray(index, int)]
    centres = np.asarray(centres, float)
    widths = np.asarray(widths, float)
    if abs(float(b0[0])) > 0.0:
        from .cofields import _centres_and_widths
        own_c, own_w, own_b = _centres_and_widths(mol, model.fieldset.specs[f], f)
        s = model.field_scale_for(f, mol)
        centres = np.vstack([centres, np.asarray(own_c, float)])
        c = np.concatenate([c, float(b0[0]) * s * np.asarray(own_w, float)])
        widths = np.concatenate([widths, np.asarray(own_b, float)])
    return centres, c, widths


def local_cofield_volume(model: CMFModel, f: int, mol: MoleculeFields, grid,
                         include_scale: bool = False) -> np.ndarray:
    """C_fx on a grid (shape = grid.shape); with `include_scale`, multiplied by h_f c_f."""
    from .cofields import evaluate_gaussians
    centres, coeffs, widths = local_cofield_coefficients(model, f, mol)
    v = evaluate_gaussians(grid.points(), centres, coeffs, widths)
    if include_scale:
        v *= model.fieldset.specs[f].h * model.scale[f]
    return v.reshape(grid.shape)


def local_cofield_at_points(model: CMFModel, f: int, mol: MoleculeFields, points: np.ndarray,
                            include_scale: bool = True) -> np.ndarray:
    """C_fx at a list of points rather than on a grid."""
    from .cofields import evaluate_gaussians
    centres, coeffs, widths = local_cofield_coefficients(model, f, mol)
    v = evaluate_gaussians(np.atleast_2d(np.asarray(points, float)), centres, coeffs, widths)
    if include_scale:
        v = v * model.fieldset.specs[f].h * model.scale[f]
    return v


# ---------------------------------------------------------------------------------- the spread

@dataclass
class CofieldSpread:
    """The local co-fields of the training molecules, summarised point by point."""
    mean: np.ndarray        # their mean on the grid: the co-field to draw for a nonlinear model
    sd: np.ndarray          # their standard deviation on the grid: where the model is not additive
    n: int                  # how many training molecules the two are taken over


def _training_duals(model: CMFModel) -> np.ndarray:
    """B of :func:`local_duals` for the training set, with b0 folded onto the diagonal.

    Training molecule x's own normalised field is row x of the basis, so the second term of
    Eq. (2) is one more weight on that row.
    """
    B, b0 = local_duals(model)
    return B + np.diag(b0)


def cofield_spread(model: CMFModel, f: int, grid, include_scale: bool = False,
                   chunk: int = 0) -> CofieldSpread:
    """Mean and standard deviation of the local co-fields C_fx over the training molecules x.

    Both are read off the same matrix the co-field noise band uses -- the normalised fields of
    the training molecules at the grid points -- multiplied by the coefficient matrix of
    :func:`local_duals`, one block of points at a time.  For a linear model the standard
    deviation is zero everywhere and the mean is the co-field.
    """
    from .cofields import _point_chunk, per_molecule_fields
    Bt = _training_duals(model)
    basis = model.training_field_basis(f)
    pts = grid.points()
    n = Bt.shape[0]
    mean = np.zeros(len(pts))
    sd = np.zeros(len(pts))
    if chunk <= 0:
        chunk = _point_chunk(n)
    for a in range(0, len(pts), chunk):
        X = per_molecule_fields(model, f, pts[a:a + chunk], chunk=chunk, basis=basis)
        C = Bt @ X
        mean[a:a + chunk] = C.mean(axis=0)
        sd[a:a + chunk] = C.std(axis=0)
    if include_scale:
        s = model.fieldset.specs[f].h * model.scale[f]
        mean *= s
        sd *= abs(s)
    return CofieldSpread(mean=mean.reshape(grid.shape), sd=sd.reshape(grid.shape), n=int(n))


# ----------------------------------------------------------------------------------- the index

@dataclass
class NonlinearityIndex:
    """How far the local co-fields of one field type differ over the training molecules."""
    field: str
    #: share of the mean squared norm of the local co-fields that their spread around the mean
    #: local co-field makes up: 0 for a linear model, 1 when they have nothing in common
    spread: float
    #: mean cosine between the local co-fields of two training molecules: 1 for a linear model
    cosine: float
    n: int

    def __str__(self) -> str:
        return (f"{self.field}: spread {self.spread:.3f} of the squared norm, mean cosine "
                f"{self.cosine:.3f} over {self.n} training molecules")


def _index_from_gram(name: str, G: np.ndarray) -> NonlinearityIndex:
    n = G.shape[0]
    d = np.diag(G)
    total = float(d.mean())
    if not np.isfinite(total) or total <= 0.0:
        return NonlinearityIndex(name, 0.0, float("nan"), n)
    spread = (total - float(G.mean())) / total
    spread = float(min(max(spread, 0.0), 1.0))
    norms = np.sqrt(np.maximum(d, 0.0))
    ok = norms > 1e-300
    cos = float("nan")
    if ok.sum() > 1:
        C = G[np.ix_(ok, ok)] / np.outer(norms[ok], norms[ok])
        off = ~np.eye(int(ok.sum()), dtype=bool)
        cos = float(np.mean(C[off]))
    return NonlinearityIndex(name, spread, cos, n)


def nonlinearity_indices(model: CMFModel, joint: bool = True) -> List[NonlinearityIndex]:
    """The non-linearity index of every enabled field type, and of all of them together.

    The local co-fields of the training molecules are combinations of the training fields,
    ``C_x = sum_j Bt[x, j] x_j``, so their Gram matrix in the Hilbert space of the fields is
    ``Bt K_f Bt^T`` with ``K_f`` the (normalised) training kernel of field f, which the model
    keeps.  The index is read off that matrix: the mean squared norm is the mean of its
    diagonal, the squared norm of the mean local co-field is the mean of all its entries, and
    the spread is one minus their ratio.  Nothing is evaluated on a grid.  With `joint` a last
    entry named ``joint`` takes all the fields together, weighted as the joint kernel weighs
    them, which is the index of the model as a whole.
    """
    _require_expansion(model, "the non-linearity index")
    Bt = _training_duals(model)
    if model.K_fields_train is None:
        model._linear_train_kernel()
    Kf = np.asarray(model.K_fields_train, float)
    out = []
    for f, spec in enumerate(model.fieldset.specs):
        if not spec.enabled:
            continue
        out.append(_index_from_gram(spec.name, Bt @ Kf[f] @ Bt.T))
    if joint:
        out.append(_index_from_gram("joint", Bt @ model.joint_kernel(Kf) @ Bt.T))
    return out


# ------------------------------------------------------------------- second derivatives
#
# The maps above are first derivatives.  How the effect of field at one place depends on the
# field at another is a second derivative, and it is taken here with respect to the amplitudes
# of *probe atoms*: Gaussians of a given width added to the molecule's raw field, one per
# place.  A probe has a width, so the second derivative is finite (with respect to the field
# at a point it carries a Dirac delta on the diagonal); and the normalisation of the kernels
# is differentiated exactly rather than argued away, which is what makes the finite-difference
# test of tests/test_nonlinearity.py pass to four figures.
#
# With X_f(e) = X_f + sum_k e_k p_k the raw field with probes, the linear joint kernel with
# training molecule j is S_j(e) = sum_f h_f c_f r_fj(e) + D_j, where r_fj = <X_f(e), x_fj> /
# sqrt(N_f(e)) for normalised kernels (N_f = <X_f, X_f>) and r_fj = <X_f(e), x_fj> without,
# and the self-kernel n(e) is constant for normalised kernels and sum_f h_f c_f N_f(e) + D
# without.  Everything below is the chain rule through g(S_j, n, m_j):
#
#     dy/de_k      = sum_j a_j [ g_s S_jk + g_n n_k ]
#     d2y/de_k de_l = sum_j a_j [ g_ss S_jk S_jl + g_sn (S_jk n_l + n_k S_jl) + g_nn n_k n_l
#                                + g_s S_jkl + g_n n_kl ],
#
# with the overlaps u_k = <p_k, X_f>, v_jk = <p_k, x_fj>, W_kl = <p_k, p_l>, A_j = <X_f, x_fj>:
#
#     normalised:    S_jk  = h c N^-1/2 (v_jk - A_j u_k / N),  n_k = 0,  n_kl = 0,
#                    S_jkl = h c N^-3/2 [ -(u_l v_jk + u_k v_jl) - A_j W_kl + 3 A_j u_k u_l / N ]
#     unnormalised:  S_jk  = h c v_jk,  S_jkl = 0,  n_k = 2 h c u_k,  n_kl = 2 h c W_kl,
#
# the W terms only between probes of the same field type.  A linear model without
# normalisation has a zero Hessian; with cosine normalisation it has not, because the
# normalisation itself is a nonlinearity (a field cannot grow without bound), and the maps show
# that part too.


def layer_second_derivatives(model: CMFModel, K: np.ndarray, diag_p: np.ndarray,
                             diag_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(g_ss, g_sn, g_nn): the second derivatives of the layer, as :func:`layer_derivatives`.

    Defined for the linear kernel (all zero) and the Gaussian layer.  The arc-cosine layers are
    refused: their kernel has a cusp where a molecule meets a training molecule (the angle is
    zero), and the second derivative diverges there, so a training molecule has no
    pair-interaction field under them.
    """
    K = np.asarray(K, float)
    dp = np.asarray(diag_p, float)[:, None]
    dq = np.asarray(diag_q, float)[None, :]
    if model.is_linear:
        z = np.zeros_like(K)
        return z, z.copy(), z.copy()
    if model.kernel_layer == "gauss":
        g = float(model.params.get("gamma", 1.0)) * float(getattr(model, "gamma_scale_", None) or 1.0)
        d2 = np.maximum(dp + dq - 2.0 * K, 0.0)
        val = np.exp(-g * d2)
        return 4.0 * g * g * val, -2.0 * g * g * val, g * g * val
    raise ValueError(
        f"the second derivatives of the '{model.kernel_layer}' layer are not defined everywhere: "
        "the arc-cosine kernel has a cusp where two molecules coincide, so the pair-interaction "
        "field and the interaction modes exist for the Gaussian layer and the linear kernel only")


@dataclass
class ProbeSet:
    """Probe atoms of one field type: Gaussians added to the molecule's raw field.

    ``centres`` (P, 3), ``widths`` (P,) the exponents b of exp(-b/2 |r - c|^2), and ``weights``
    (P,) the amount of field one unit of the probe's amplitude puts there: one for an indicator
    field, a charge for the electrostatic field.  The derivatives below are per unit amplitude.
    """
    field: int
    centres: np.ndarray
    widths: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return len(self.centres)


def probe_set(model: CMFModel, f: int, centres, widths=None, weights=None,
              radius: float = 1.7) -> ProbeSet:
    """Probes in field `f` at `centres`, with the field's own width unless `widths` is given.

    A field with one Gaussian per atom (``kind == "gauss"``) has the width ``alpha``; the steric
    kind scales it by the atom's radius, for which `radius` stands in.
    """
    spec = model.fieldset.specs[f]
    c = np.atleast_2d(np.asarray(centres, float))
    P = len(c)
    if widths is None:
        b = spec.alpha if spec.kind == "gauss" else spec.alpha / float(radius) ** 2
        widths = np.full(P, float(b))
    w = np.ones(P) if weights is None else np.asarray(weights, float).reshape(-1)
    return ProbeSet(int(f), c, np.asarray(widths, float).reshape(-1), w)


def atom_probes(model: CMFModel, f: int, mol: MoleculeFields) -> ProbeSet:
    """The atoms of `mol` as probes of field `f`: a unit change of each atom's weight."""
    spec = model.fieldset.specs[f]
    b = (np.full(mol.n_atoms, spec.alpha) if spec.kind == "gauss"
         else spec.alpha / np.asarray(mol.radii, float) ** 2)
    return ProbeSet(int(f), np.asarray(mol.coords, float), b, np.ones(mol.n_atoms))


def _pair_overlaps(a: ProbeSet, b: ProbeSet) -> np.ndarray:
    """<p_k, p_l> for probes of the same field: the overlap of two Gaussians, in closed form."""
    from scipy.spatial.distance import cdist
    s = a.widths[:, None] + b.widths[None, :]
    d2 = cdist(a.centres, b.centres, "sqeuclidean")
    return ((2.0 * np.pi / s) ** 1.5 * np.exp(-(a.widths[:, None] * b.widths[None, :])
                                              / (2.0 * s) * d2)
            * a.weights[:, None] * b.weights[None, :])


class Tangent:
    """The derivatives of one model's prediction at one molecule, to second order.

    Holding the layer's derivatives, the kernel row and the molecule's self-overlaps, it answers
    for any set of probes: what each probe adds per unit amplitude (:meth:`gradient`), and how
    the effect of one depends on the amplitude of another (:meth:`hessian`).  The maps and the
    modes below are built on these two.
    """

    def __init__(self, model: CMFModel, mol: MoleculeFields):
        _require_expansion(model, "second derivatives")
        if getattr(model, "pbc_overlap", False):
            raise ValueError("the probe derivatives are not implemented for the periodic overlap "
                             "kernel, whose molecule has images the probes would need too")
        self.model, self.mol = model, mol
        table = AtomTable.from_molecules([mol])
        self.Kf = np.asarray(model.test_kernels(table), float)          # (F, 1, n)
        K_lin = model.joint_kernel(self.Kf, table.descriptors)
        n_self = model.joint_self(table)
        dq = getattr(model, "base_diag_", None)
        if dq is None:
            dq = np.diag(model._linear_train_kernel()).copy()
        coef = np.asarray(model.coef, float)
        _g, g_s, g_n = layer_derivatives(model, K_lin, n_self, dq)
        self.A_s = coef * g_s[0]
        self.A_n = coef * g_n[0]
        self._second = None
        self._dq = dq
        self._K_lin, self._n_self = K_lin, n_self
        self.N = np.asarray(model._self_kernels(table), float)[:, 0]      # raw <X_f, X_f>
        self.hc = np.asarray(model.fieldset.mixing_vector(), float) * np.asarray(model.scale, float)
        self.normalize = bool(model.normalize)

    # -- pieces

    def second(self):
        if self._second is None:
            coef = np.asarray(self.model.coef, float)
            g_ss, g_sn, g_nn = layer_second_derivatives(self.model, self._K_lin, self._n_self,
                                                        self._dq)
            self._second = (coef * g_ss[0], coef * g_sn[0], coef * g_nn[0])
        return self._second

    def check_field(self, f: int) -> None:
        if self.normalize and not self.N[f] > 1e-300:
            name = self.model.fieldset.specs[f].name
            raise ValueError(
                f"{self.mol.name} has no '{name}' field at all, and with cosine-normalised kernels "
                f"the derivative with respect to that field is not defined there: any amount of "
                f"it would be normalised to unit norm")

    def A_tilde(self, f: int) -> np.ndarray:
        """<X_f, x_fj> over the training molecules, with x_fj the (normalised) basis field."""
        row = self.Kf[f, 0]
        return row * np.sqrt(self.N[f]) if self.normalize else row

    def terms(self, p: ProbeSet) -> Tuple[np.ndarray, np.ndarray]:
        """V (n_train, P): <p_k, x_fj>; u (P,): <p_k, X_f>, the molecule's own raw field."""
        from .cofields import _centres_and_widths, evaluate_gaussians, per_molecule_fields, probe_basis
        f = p.field
        self.check_field(f)
        basis = self.model.training_field_basis(f)
        own = _centres_and_widths(self.mol, self.model.fieldset.specs[f], f)
        n = self.model.train.n_mol
        V = np.zeros((n, len(p)))
        u = np.zeros(len(p))
        for b in np.unique(p.widths):
            idx = np.flatnonzero(p.widths == b)
            c0, k0, w0, index = basis
            cb, kb, wb = probe_basis(c0, k0, w0, float(b))
            V[:, idx] = (per_molecule_fields(self.model, f, p.centres[idx], basis=(cb, kb, wb, index))
                         * p.weights[idx][None, :])
            oc, ok, ow = probe_basis(*own, float(b))
            u[idx] = evaluate_gaussians(p.centres[idx], oc, ok, ow) * p.weights[idx]
        return V, u

    def first(self, p: ProbeSet, V: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """S (n_train, P) = dS_j/de_k and n1 (P,) = dn/de_k."""
        f = p.field
        hc = self.hc[f]
        if self.normalize:
            N = self.N[f]
            S = hc / np.sqrt(N) * (V - np.outer(self.A_tilde(f), u) / N)
            return S, np.zeros(len(p))
        return hc * V, 2.0 * hc * u

    # -- the answers

    def gradient(self, p: ProbeSet) -> np.ndarray:
        """dy/de_k: what one unit of each probe adds to the prediction, to first order."""
        V, u = self.terms(p)
        S, n1 = self.first(p, V, u)
        return self.A_s @ S + float(self.A_n.sum()) * n1

    def hessian(self, a: ProbeSet, b: Optional[ProbeSet] = None,
                terms_a: Optional[tuple] = None) -> np.ndarray:
        """d2y/de_k de_l between the probes of `a` (rows) and `b` (columns; `a` when None)."""
        if b is None:
            b = a
        Va, ua = self.terms(a) if terms_a is None else terms_a
        Vb, ub = (Va, ua) if b is a else self.terms(b)
        Sa, na = self.first(a, Va, ua)
        Sb, nb = self.first(b, Vb, ub)
        A_ss, A_sn, A_nn = self.second()
        H = Sa.T @ (A_ss[:, None] * Sb)
        if not self.normalize:
            H += np.outer(Sa.T @ A_sn, nb) + np.outer(na, Sb.T @ A_sn)
            H += float(A_nn.sum()) * np.outer(na, nb)
        if a.field == b.field:
            W = _pair_overlaps(a, b)
            f = a.field
            hc = self.hc[f]
            if self.normalize:
                N = self.N[f]
                AsA = float(self.A_s @ self.A_tilde(f))
                AsVa = self.A_s @ Va
                AsVb = self.A_s @ Vb
                H += hc * N ** -1.5 * (-(np.outer(AsVa, ub) + np.outer(ua, AsVb)) - AsA * W
                                       + 3.0 * AsA * np.outer(ua, ub) / N)
            else:
                H += 2.0 * hc * float(self.A_n.sum()) * W
        return H

    def diagonal(self, b: ProbeSet) -> np.ndarray:
        """d2y/de_k^2 for every probe: the curvature along each probe alone."""
        Vb, ub = self.terms(b)
        Sb, nb = self.first(b, Vb, ub)
        A_ss, A_sn, A_nn = self.second()
        d = np.einsum("j,jk,jk->k", A_ss, Sb, Sb)
        if not self.normalize:
            d += 2.0 * nb * (A_sn @ Sb) + float(A_nn.sum()) * nb * nb
        W = (np.pi / b.widths) ** 1.5 * b.weights ** 2            # <p_k, p_k>
        f = b.field
        hc = self.hc[f]
        if self.normalize:
            N = self.N[f]
            AsA = float(self.A_s @ self.A_tilde(f))
            AsV = self.A_s @ Vb
            d += hc * N ** -1.5 * (-2.0 * ub * AsV - AsA * W + 3.0 * AsA * ub * ub / N)
        else:
            d += 2.0 * hc * float(self.A_n.sum()) * W
        return d


def probe_gradient(model: CMFModel, mol: MoleculeFields, probes: ProbeSet) -> np.ndarray:
    """What one unit of each probe adds to the prediction of `mol`, to first order."""
    return Tangent(model, mol).gradient(probes)


def probe_hessian(model: CMFModel, mol: MoleculeFields, a: ProbeSet,
                  b: Optional[ProbeSet] = None) -> np.ndarray:
    """How the effect of each probe of `a` depends on the amplitude of each probe of `b`."""
    return Tangent(model, mol).hessian(a, b)


def pair_interaction_field(model: CMFModel, mol: MoleculeFields, f: int, centre, grid,
                           f_probe: Optional[int] = None, width: Optional[float] = None,
                           weight: float = 1.0, chunk: int = 0) -> np.ndarray:
    """The pair-interaction field: one probe fixed at `centre`, the other scanned over the grid.

    The value at r is the second derivative of the prediction with respect to the amplitude of a
    probe of field `f_probe` (default `f`) at `centre` and of a probe of field `f` at r: how
    much adding field at r changes what the fixed probe does.  Positive where the two reinforce
    each other, negative where one undoes part of the other, zero where the model treats the two
    places additively.  With the fixed probe on an atom of `mol`, `weight` set to that atom's
    weight gives the field for *that atom*; with weight one it is per unit of field.
    """
    from .cofields import _point_chunk
    T = Tangent(model, mol)
    a = probe_set(model, f if f_probe is None else f_probe, [centre],
                  widths=None if width is None else [width], weights=[weight])
    terms_a = T.terms(a)
    pts = grid.points()
    out = np.zeros(len(pts))
    if chunk <= 0:
        chunk = _point_chunk(model.train.n_mol)
    for s in range(0, len(pts), chunk):
        b = probe_set(model, f, pts[s:s + chunk])
        out[s:s + chunk] = T.hessian(a, b, terms_a=terms_a)[0]
    return out.reshape(grid.shape)


def gradient_field(model: CMFModel, mol: MoleculeFields, f: int, grid, chunk: int = 0) -> np.ndarray:
    """What one unit of a probe of field `f` adds to the prediction of `mol`, at every point.

    The first-derivative companion of the maps below: under a nonlinear layer it is what
    :func:`cmfqsar.cofields.atom_effect` is for a linear model, the local co-field convolved
    with the probe's own Gaussian, in the units of the activity.
    """
    from .cofields import _point_chunk
    T = Tangent(model, mol)
    pts = grid.points()
    out = np.zeros(len(pts))
    if chunk <= 0:
        chunk = _point_chunk(model.train.n_mol)
    for s in range(0, len(pts), chunk):
        out[s:s + chunk] = T.gradient(probe_set(model, f, pts[s:s + chunk]))
    return out.reshape(grid.shape)


def curvature_field(model: CMFModel, mol: MoleculeFields, f: int, grid, chunk: int = 0) -> np.ndarray:
    """The diagonal of the pair-interaction field: the second derivative along one probe.

    Negative where the model saturates (a second unit of field there adds less than the first),
    positive where it accelerates.  For a linear model without normalisation it is zero; with
    cosine normalisation it is not, because the normalisation itself saturates.
    """
    from .cofields import _point_chunk
    T = Tangent(model, mol)
    pts = grid.points()
    out = np.zeros(len(pts))
    if chunk <= 0:
        chunk = _point_chunk(model.train.n_mol)
    for s in range(0, len(pts), chunk):
        out[s:s + chunk] = T.diagonal(probe_set(model, f, pts[s:s + chunk]))
    return out.reshape(grid.shape)


def atom_interactions(model: CMFModel, mol: MoleculeFields, f: int) -> Tuple[np.ndarray, np.ndarray]:
    """The interaction matrix of the atoms of `mol` in field `f`, and their first-order effects.

    ``H[k, l]`` is the second derivative of the prediction with respect to the weights of atoms k
    and l in that field (two partial charges, say), ``g[k]`` the first derivative: what a unit
    more of weight on atom k adds, which under a nonlinear layer takes the place of the atom
    contribution.  ``H`` is symmetric; its diagonal is the curvature along one atom, its
    off-diagonal part says which atoms the model reads together rather than one by one.
    """
    T = Tangent(model, mol)
    p = atom_probes(model, f, mol)
    V, u = T.terms(p)
    S, n1 = T.first(p, V, u)
    g = T.A_s @ S + float(T.A_n.sum()) * n1
    return T.hessian(p, terms_a=(V, u)), g


@dataclass
class InteractionMode:
    """One eigenpair of the atom interaction matrix."""
    value: float          # the eigenvalue: the curvature along the mode, per unit of its norm
    vector: np.ndarray    # one weight per atom, of unit norm; the pattern of changes that moves together


def interaction_modes(model: CMFModel, mol: MoleculeFields, f: int, k: int = 3) -> List[InteractionMode]:
    """The `k` eigenpairs of :func:`atom_interactions` of largest absolute eigenvalue.

    A mode is a pattern of weight changes over the atoms whose effect on the prediction is pure
    curvature: changing the weights along it by t moves the prediction by value t^2 / 2 beyond
    the linear term.  A positive value means the changes in the pattern reinforce one another,
    a negative one that they saturate.  The patterns of the largest absolute values are where
    the model is least additive; a mode spread over two sites is a pair the model reads
    together.
    """
    H, _g = atom_interactions(model, mol, f)
    vals, vecs = np.linalg.eigh(0.5 * (H + H.T))
    order = np.argsort(-np.abs(vals))[:max(int(k), 0)]
    return [InteractionMode(float(vals[i]), vecs[:, i].copy()) for i in order]
