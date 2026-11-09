"""
The derivative of a prediction with respect to the atoms of the molecule being predicted.

A CMF model predicts an overlap integral.  The prediction of molecule *x* is

    y(x) = sum_j a_j K(x, j) + b,

and the kernel is a sum of Gaussian overlaps over pairs of atoms, one atom from *x* and one from
training molecule *j*.  A Gaussian overlap is an analytic function of the distance between the two
atoms, so ``y`` is an analytic function of the coordinates of the atoms of *x* and of the per-atom
weights that carry its fields.  This module differentiates it.

What the derivative is for
--------------------------
Three things, all of which otherwise have to be done by trying one structure at a time:

* **Guiding a generative model.**  A three-dimensional generative model proposes atom positions.
  With a gradient, a fitted CMF model can be used as a guidance term inside that proposal: the
  gradient says which way every atom would have to move to raise the predicted activity.  Nothing
  has to be enumerated first.
* **Refining a pose or a conformation by predicted activity.**  :mod:`cmfqsar.varalign` and
  :mod:`cmfqsar.flexalign` already move a molecule by *confidence*, with their own gradients.
  This module supplies the other objective, the prediction itself, in the same analytic form.
* **Reading the model at an atom.**  The derivative with respect to an atom's weight is a local
  statement: how much the prediction would change if that one atom carried a little more charge,
  or were a little more hydrophobic.  That is the question a chemist asks of a co-field picture,
  answered as a number per atom instead of as a surface.

The derivatives
---------------
Write the raw kernel of field *f* as a double sum over atoms, with ``g_f`` the Gaussian overlap of
two atoms as a function of their squared distance:

    K_f(x, j) = sum_l sum_m w_fl w_fjm g_f(|r_l - r_jm|^2)

and the molecule's own self-kernel as ``d_f(x) = K_f(x, x)``.  Then

    dK_f(x, j) / dr_l   = 2 sum_m w_fl w_fjm dg_f (r_l - r_jm)
    dK_f(x, j) / dw_fl  = sum_m w_fjm g_f(|r_l - r_jm|^2)
    dd_f(x) / dr_l      = 2 (the same sum, over the atoms of x itself)
    dd_f(x) / dw_fl     = 2 sum_m w_fm g_f(|r_l - r_m|^2)

where ``dg_f`` is the derivative of the overlap with respect to the squared distance.  The factor
of two in the last two lines is there because atom *l* appears in both arguments of the
self-kernel.  The ratio of that derivative to the overlap itself is a constant per field
(``-alpha_f/4``) for a field whose Gaussians share one width, and a constant per atom pair for the
radius-scaled steric field; that is the part :func:`cmfqsar.kernels.kernel_gradient` already did,
and the coordinate part of this module is a chain rule on top of it.

Everything the model puts between the raw kernel and the prediction is differentiated with it:

* the **cosine normalisation**, through ``s_f(x) = d_f(x) ** -0.5`` and its derivative
  ``-0.5 s_f ** 3 dd_f``.  This term is what makes the gradient of a normalised model different
  from the gradient of its kernel row: moving an atom changes what the molecule is divided by, as
  well as its overlap with the training set;
* the **field scaling and mixing**, ``h_f c_f``, which are constants of the fitted model;
* the **nonlinear kernel layer** (``gauss``, ``arccos1``, ``arccos2``), which depends on the
  linear kernel and on the molecule's own self-kernel, so it contributes through both;
* the **non-field descriptors**, which contribute nothing: they are computed from the molecular
  graph or read from a file property, so they do not change when an atom moves.

What the derivative is not
--------------------------
**A weight is not a molecule.**  The weight of an atom in the electrostatic field is its partial
charge, and the charges of a real molecule are not free: they sum to its total charge and they
follow from its structure.  A positive derivative with respect to one atom's weight says the model
would like more of that quantity there.  It does not say that a molecule exists in which only that
atom's charge has changed.  The same holds for the other fields, whose weights are table
increments of an atom type.  Treat the weight gradient as a direction in the field
representation, to be realised by choosing a different atom or a different substituent, not as a
molecule.

**The gradient is local and the model has a domain.**  A step along the gradient leaves the
training set, which is where a CMF model is least reliable.  The posterior variance
(:meth:`cmfqsar.model.CMFModel.predict_with_variance`), the conformal interval
(:mod:`cmfqsar.conformal`) and the two-sample test (:mod:`cmfqsar.mmd`) are the three things that
say how far a step can be trusted; the lower confidence bound of :mod:`cmfqsar.active` is the
objective to follow when the aim is a molecule to make rather than a number to maximise.

**Maximising an overlap is not designing a molecule.**  The overlap of a field with a positive
lobe of a co-field grows without limit as atoms are piled into that lobe.  Nothing in this
gradient knows about valence, bond lengths or sterics.  It is a term to add to a generator or a
force field that does, not an optimiser on its own.

**Probe fields have no per-atom weight.**  In a GRID-like probe field every atom carries several
Gaussians whose coefficients come from the expansion of a probe interaction curve
(:mod:`cmfqsar.probes`), not from one weight.  The coordinate gradient covers them; the weight
gradient reports zero for their columns and names them in
:attr:`PredictionGradient.weightless_fields`.

Which models are covered
------------------------
Any fitted model whose prediction is a sum over the training molecules plus a constant: the
regression learners, the classifiers (the gradient is then of the decision value, not of the
probability) and the ranking SVM, on either of two kernels.  On the overlap-integral kernel the
derivative runs over the pairs of one atom of the molecule and one atom of a training molecule.
On the rotation-invariant field autocorrelation, whose channels are built from the distances
*inside* the molecule alone, the derivative runs over the pairs of its own atoms and through the
row length the cosine normalisation divides out; a model on that kernel has no frame, so its
gradient moves a conformation and not a pose (plan item MS3).

The other invariant kernels -- the surface autocorrelation with its heights, the periodic one
with its images and background, the reciprocal-space structure factors and the three-body
channels -- are refused, because each builds its channels differently and needs a derivative of
its own.  So is the periodic *overlap* kernel: it is this same overlap taken against an
image-expanded table, so moving one atom moves every image of it, and that term is not here
(found while writing the ASE calculator of plan item MS7).  So are the one-class and data-fusion
scores, whose decision is not a sum over the training molecules.  Every case that is not covered
is refused with a reason rather than approximated.

Checked against finite differences
----------------------------------
``tests/test_gradients.py`` holds the analytic gradient to central finite differences of
:meth:`cmfqsar.model.CMFModel.predict` itself, over the combinations that change the chain rule:
normalised and unnormalised kernels, the steric field with its per-atom widths, probe fields,
interpolated attenuations, the three nonlinear layers, a descriptor block and a conformational
ensemble; and, on the invariant kernel, every field of the set, the three layers, the channel
scale with fitted block weights, and the distance grid with and without its self terms.  The
remaining disagreement is that of the finite difference, not of the gradient.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .fields import AtomTable, MoleculeFields
from .kernels import atom_kernel_block, kernel_gradient
from .model import CMFModel

#: self-kernels at or below this are treated as a field the molecule does not have, exactly as
#: :func:`cmfqsar.kernels.normalize_kernels` does, so that a molecule with no donor atom gets a
#: zero row and a zero gradient for that field rather than a division by nothing
TINY = 1e-300


@dataclass
class PredictionGradient:
    """The prediction of one molecule and its derivatives.

    Attributes
    ----------
    value : the prediction itself, the number :meth:`cmfqsar.model.CMFModel.predict` gives.
    coords : (N, 3) derivative with respect to the atom positions, in activity units per
        angstrom, or None when it was not asked for.  For a molecule that is an ensemble of
        conformations or states, the rows are the atoms of all of its members, one member after
        another, in the order the ensemble was built.
    weights : (N, F) derivative with respect to the per-atom field weights, in activity units per
        unit of weight, one column per field of the field set, or None when it was not asked for.
    field_names : the field of every column of `weights`.
    weightless_fields : fields whose weight column is zero because they have no single per-atom
        weight to differentiate (the probe fields).
    """
    value: float
    coords: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    field_names: List[str] = field(default_factory=list)
    weightless_fields: List[str] = field(default_factory=list)

    @property
    def forces(self) -> Optional[np.ndarray]:
        """The negative of the coordinate derivative: the direction in which every atom would
        have to move to *lower* the prediction, which is the sign convention of a force."""
        return None if self.coords is None else -self.coords

    def largest_coordinate_terms(self, n: int = 5) -> List[Tuple[int, float]]:
        """The `n` atoms whose position matters most, as (atom index, length of its derivative)."""
        if self.coords is None:
            return []
        size = np.linalg.norm(self.coords, axis=1)
        order = np.argsort(size)[::-1][:int(max(0, n))]
        return [(int(i), float(size[i])) for i in order]

    def largest_weight_terms(self, n: int = 5) -> List[Tuple[int, str, float]]:
        """The `n` largest weight derivatives, as (atom index, field name, derivative)."""
        if self.weights is None:
            return []
        flat = np.argsort(np.abs(self.weights).reshape(-1))[::-1][:int(max(0, n))]
        out = []
        for k in flat:
            atom, f = divmod(int(k), self.weights.shape[1])
            out.append((atom, self.field_names[f] if f < len(self.field_names) else str(f),
                        float(self.weights[atom, f])))
        return out


# --------------------------------------------------------------------------- #
# What the model is allowed to be
# --------------------------------------------------------------------------- #

def _check_model(model: CMFModel) -> None:
    """Refuse, with a reason, every model whose prediction this gradient does not describe."""
    if model.train is None or model.coef is None:
        raise RuntimeError("the model must be fitted before its gradient can be taken")
    if bool(getattr(model, "pbc_overlap", False)):
        raise ValueError(
            "the periodic overlap kernel evaluates the overlap against an image-expanded table "
            "(plan item M4), so moving one atom moves every image of it and the derivative gains "
            "a term over the images that this one does not have. It is refused rather than "
            "approximated, as the other periodic kernels are (plan item MS3).")
    if not model.uses_fields:
        _check_invariant_model(model)
    oc = getattr(model.learner_, "oc_", None) if model.learner_ is not None else None
    if oc is not None:
        raise ValueError(
            f"the gradient is implemented for a prediction that is a sum over the training "
            f"molecules plus a constant: the regression learners, the classifiers and the "
            f"ranking SVM. '{model.method}' scores a molecule against the training set as a "
            f"whole, so its decision is not of that form. To move a molecule towards the "
            f"training set, minimise the posterior variance instead (cmfqsar.varalign).")


def _working_copy(mol: MoleculeFields, coords: Optional[np.ndarray],
                  weights: Optional[np.ndarray]) -> MoleculeFields:
    """`mol` with the given coordinates and weights, so that a gradient can be taken away from the
    molecule's own geometry without altering it."""
    if coords is None and weights is None:
        return mol
    X = mol.coords if coords is None else np.asarray(coords, float)
    W = mol.weights if weights is None else np.asarray(weights, float)
    if X.shape != mol.coords.shape:
        raise ValueError(f"coords has shape {X.shape}, the molecule has {mol.coords.shape}")
    if W.shape != mol.weights.shape:
        raise ValueError(f"weights has shape {W.shape}, the molecule has {mol.weights.shape}")
    return dataclasses.replace(mol, coords=X, weights=W)


def _spec_terms(model: CMFModel) -> List[Tuple[np.ndarray, list]]:
    """The field sets the kernel is actually evaluated on, with their per-field coefficients.

    One term, with coefficients of one, for an ordinary model.  A model whose attenuation lies
    between the points of ``alpha_grid`` evaluates its kernel as a linear combination of the
    kernels at the two neighbouring grid points (``CMFModel._raw_kernels``), and because that
    combination is linear the gradient is the same combination of the two gradients.
    """
    it = model._interp_specs()
    if it is None:
        return [(np.ones(len(model.fieldset.specs)), list(model.fieldset.specs))]
    (c_lo, lo), (c_hi, hi) = it
    terms = [(np.asarray(c_lo, float), list(lo))]
    if hi is not None:
        terms.append((np.asarray(c_hi, float), list(hi)))
    return terms


def _layer_terms(model: CMFModel, K_lin: np.ndarray, d_self: float):
    """(the kernel row the learner sees, its derivative in the linear kernel, its derivative in
    the molecule's own self-kernel).

    The linear kernel passes through unchanged and contributes nothing through the self-kernel.
    Each nonlinear layer is differentiated as it is implemented in
    :meth:`cmfqsar.model.CMFModel.apply_layer`, including its clipping, so that the gradient is
    the derivative of the prediction the program actually makes.
    """
    K_lin = np.asarray(K_lin, float)
    if model.is_linear:
        return K_lin, np.ones_like(K_lin), np.zeros_like(K_lin)
    d_train = np.asarray(model.base_diag_, float)
    if model.kernel_layer == "gauss":
        g = (float(model.params.get("gamma", 1.0))
             * float(getattr(model, "gamma_scale_", None) or 1.0))
        raw = d_self + d_train - 2.0 * K_lin
        L = np.exp(-g * np.maximum(raw, 0.0))
        live = (raw > 0.0).astype(float)      # the layer clamps a negative squared distance
        return L, 2.0 * g * L * live, -g * L * live
    if model.kernel_layer in ("arccos1", "arccos2"):
        p = np.sqrt(max(float(d_self), TINY))
        q = np.sqrt(np.maximum(d_train, TINY))

        def step(K_in):
            cos = np.clip(K_in / (p * q), -1.0, 1.0)
            th = np.arccos(cos)
            out = (p * q) / np.pi * (np.sin(th) + (np.pi - th) * cos)
            return out, (np.pi - th) / np.pi, q * np.sin(th) / (2.0 * p * np.pi)

        out1, dK1, dp1 = step(K_lin)
        if model.kernel_layer == "arccos1":
            return out1, dK1, dp1
        out2, dK2, dp2 = step(out1)
        return out2, dK2 * dK1, dp2 + dK2 * dp1
    raise ValueError(f"unknown kernel layer {model.kernel_layer}")


# --------------------------------------------------------------------------- #
# The gradient of a prediction
# --------------------------------------------------------------------------- #

def prediction_gradient(model: CMFModel, mol: MoleculeFields,
                        coords: Optional[np.ndarray] = None,
                        weights: Optional[np.ndarray] = None, wrt: str = "both",
                        chunk: int = 4000) -> PredictionGradient:
    """The prediction of `mol` and its analytic derivatives with respect to its atoms.

    Parameters
    ----------
    model : a fitted :class:`~cmfqsar.model.CMFModel` on the overlap-integral kernel.  Its
        training poses define the frame, exactly as they do for a prediction.
    mol : the molecule, as :class:`~cmfqsar.fields.MoleculeFields` from
        :func:`cmfqsar.fields.build_fields`, in the frame of the training set.
    coords : positions to differentiate at, (N, 3), instead of the molecule's own.  Nothing is
        written back to `mol`, which is what lets a caller follow a gradient without rebuilding
        the fields at every step.
    weights : per-atom weights to differentiate at, (N, F), instead of the molecule's own.
    wrt : "coords", "weights" or "both".  Asking for less costs less.
    chunk : training atoms per block of the coordinate sum, which bounds the memory it uses.

    Returns
    -------
    :class:`PredictionGradient`

    Notes
    -----
    The per-atom weights of a field are not free parameters of a molecule, and a step along the
    gradient leaves the training set.  Read the limits in the module docstring before using
    either derivative for design.
    """
    _check_model(model)
    if wrt not in ("coords", "weights", "both"):
        raise ValueError(f"wrt must be 'coords', 'weights' or 'both', not {wrt!r}")
    want_coords = wrt in ("coords", "both")
    want_weights = wrt in ("weights", "both")
    if not model.uses_fields:
        # the invariant kernel has no training atom to overlap with: its own chain rule
        return _invariant_prediction_gradient(model, mol, coords, weights, wrt)

    work = _working_copy(mol, coords, weights)
    one = AtomTable.from_molecules([work])
    specs = list(model.fieldset.specs)
    n_fields = len(specs)
    terms = _spec_terms(model)
    ci = work.probe_coeffs or None
    hc = model.fieldset.mixing_vector() * (np.ones(n_fields) if model.scale is None
                                           else np.asarray(model.scale, float))

    # ----- the values the chain rule is built from.  The raw per-field kernel row of the moving
    # molecule against the training set, and its own per-field self-kernels, both evaluated
    # exactly as a prediction evaluates them, interpolated attenuations included.
    K_raw = model._atom_block(work).sum(axis=1)                 # (F, n_train)
    d_x = model._self_kernels(one)[:, 0]                        # (F,)
    if model.normalize:
        if model.self_diag is None:
            raise RuntimeError("a normalised model must carry the self-kernels of its "
                               "training set")
        s_x = np.where(d_x > TINY, 1.0 / np.sqrt(np.maximum(d_x, TINY)), 0.0)
        s_tr = np.where(model.self_diag > TINY,
                        1.0 / np.sqrt(np.maximum(model.self_diag, TINY)), 0.0)
        K_fields = K_raw * s_x[:, None] * s_tr
    else:
        s_x = np.ones(n_fields)
        s_tr = np.ones_like(K_raw)
        K_fields = K_raw

    K_lin = model.joint_kernel(K_fields[:, None, :], one.descriptors)[0]        # (n_train,)
    d_joint = float(model.joint_self(one)[0])
    K_layer, dL_dK, dL_dself = _layer_terms(model, K_lin, d_joint)
    coef = np.asarray(model.coef, float)
    value = float(K_layer @ coef + model.bias)

    # ----- the two multipliers every derivative is a combination of: u[f, j] multiplies the
    # derivative of K_f(x, j), and v[f] multiplies the derivative of the self-kernel d_f(x)
    t = coef * dL_dK                                            # (n_train,)
    u = hc[:, None] * t[None, :]
    if model.normalize:
        u = u * s_x[:, None] * s_tr
        # the molecule's own normalisation: K_f(x, j) s_f(x) s_f(j), with s_f(x) = d_f(x) ** -0.5
        v_norm = -0.5 * s_x ** 2 * np.einsum("fj,fj->f", u, K_raw)
        # a cosine-normalised self-kernel is the constant one, so the nonlinear layer's dependence
        # on it contributes nothing; that is why v_self below is for an unnormalised model only
        v_self = np.zeros(n_fields)
    else:
        v_norm = np.zeros(n_fields)
        v_self = hc * float(coef @ dL_dself)

    g_coords = None
    if want_coords:
        g_coords = np.zeros((work.n_atoms, 3))
        for c, sp in terms:
            g_coords += kernel_gradient(work.coords, work.weights, work.radii, model.train, sp,
                                        u * c[:, None], chunk=chunk, ci=ci)
            if np.any(v_norm):
                # the self-kernel term: the atom appears in both arguments, hence the two
                g_coords += kernel_gradient(work.coords, work.weights, work.radii, one, sp,
                                            (2.0 * v_norm * c)[:, None], chunk=chunk, ci=ci)
        if np.any(v_self):
            # the layer reads the self-kernel through CMFModel.joint_self, which uses the nominal
            # attenuations rather than the interpolated ones; the gradient follows it there
            g_coords += kernel_gradient(work.coords, work.weights, work.radii, one, specs,
                                        (2.0 * v_self)[:, None], chunk=chunk, ci=ci)

    g_weights = None
    if want_weights:
        g_weights = np.zeros((work.n_atoms, n_fields))
        unit = np.ones_like(work.weights)
        for c, sp in terms:
            # with unit weights on the moving molecule, the atom block is exactly the derivative
            # of K_f(x, j) with respect to the weight of atom l
            B = atom_kernel_block(work.coords, unit, work.radii, model.train, sp, ci=ci)
            g_weights += np.einsum("flj,fj->lf", B, u * c[:, None])
            if np.any(v_norm):
                S = atom_kernel_block(work.coords, unit, work.radii, one, sp, ci=ci)
                g_weights += (2.0 * v_norm * c)[None, :] * S[:, :, 0].T
        if np.any(v_self):
            S = atom_kernel_block(work.coords, unit, work.radii, one, specs, ci=ci)
            g_weights += (2.0 * v_self)[None, :] * S[:, :, 0].T
        for f, spec in enumerate(specs):
            if getattr(spec, "is_probe", False):
                g_weights[:, f] = 0.0

    return PredictionGradient(value=value, coords=g_coords, weights=g_weights,
                              field_names=[s.name for s in specs],
                              weightless_fields=[s.name for s in specs
                                                 if getattr(s, "is_probe", False)])


def numerical_prediction_gradient(model: CMFModel, mol: MoleculeFields, h: float = 1e-5,
                                  wrt: str = "both") -> PredictionGradient:
    """The same derivatives by central finite differences of :meth:`CMFModel.predict` (slow).

    The reference the analytic gradient is tested against, and the way to check a model this
    module has not been tried on.  It costs two predictions per degree of freedom, so it is for
    small molecules and for tests, not for a design loop.
    """
    _check_model(model)
    want_coords = wrt in ("coords", "both")
    want_weights = wrt in ("weights", "both")

    def predict(X, W):
        table = AtomTable.from_molecules([_working_copy(mol, X, W)])
        return float(model.predict(table)[0])

    X0 = np.array(mol.coords, float)
    W0 = np.array(mol.weights, float)
    value = predict(X0, W0)
    g_coords = None
    if want_coords:
        g_coords = np.zeros_like(X0)
        for atom in range(X0.shape[0]):
            for k in range(3):
                plus, minus = X0.copy(), X0.copy()
                plus[atom, k] += h
                minus[atom, k] -= h
                g_coords[atom, k] = (predict(plus, W0) - predict(minus, W0)) / (2.0 * h)
    g_weights = None
    if want_weights:
        g_weights = np.zeros_like(W0)
        for atom in range(W0.shape[0]):
            for f in range(W0.shape[1]):
                plus, minus = W0.copy(), W0.copy()
                plus[atom, f] += h
                minus[atom, f] -= h
                g_weights[atom, f] = (predict(X0, plus) - predict(X0, minus)) / (2.0 * h)
    return PredictionGradient(value=value, coords=g_coords, weights=g_weights,
                              field_names=[s.name for s in model.fieldset.specs],
                              weightless_fields=[s.name for s in model.fieldset.specs
                                                 if getattr(s, "is_probe", False)])


# --------------------------------------------------------------------------- #
# The co-field as a potential
# --------------------------------------------------------------------------- #

def _gaussian_terms(model: CMFModel, f: int, width: Optional[float]):
    """The Gaussians to evaluate: the co-field's own, or the co-field smoothed by one atom.

    `width` is the ``b`` of an atom's Gaussian.  The convolution of two Gaussians is a Gaussian,
    and its integral is the atom-pair overlap this program's kernel is made of, so smoothing the
    co-field by an atom's own Gaussian turns it into the function whose value at the atom's centre
    is that atom's contribution to the prediction, per unit of weight.
    """
    centres, coeffs, widths = model.cofield_coefficients(f)
    centres = np.asarray(centres, float)
    coeffs = np.asarray(coeffs, float)
    widths = np.asarray(widths, float)
    if width is None:
        return centres, coeffs, widths
    b = float(width)
    total = widths + b
    return centres, coeffs * (2.0 * np.pi / total) ** 1.5, widths * b / total


def _cofield_sum(centres: np.ndarray, coeffs: np.ndarray, widths: np.ndarray,
                 points: np.ndarray, want_gradient: bool, chunk: int = 2048):
    """The sum of the Gaussians at every point and, when asked, its gradient.

    Dense and exact, in blocks of points so that the memory stays bounded.  Dropping distant
    Gaussians, as :func:`cmfqsar.cofields.evaluate_gaussians` does for drawing, would make the
    gradient an approximation of unstated accuracy; a gradient is usually wanted at a few points
    rather than on a grid, so the exact sum is affordable.
    """
    n = len(points)
    value = np.zeros(n)
    grad = np.zeros((n, 3)) if want_gradient else None
    if len(centres) == 0:
        return value, grad
    keep = np.abs(coeffs) > 1e-14
    if not np.any(keep):
        return value, grad
    centres, coeffs, widths = centres[keep], coeffs[keep], widths[keep]
    step = max(1, int(chunk))
    for s in range(0, n, step):
        pts = points[s:s + step]
        diff = pts[:, None, :] - centres[None, :, :]                 # (M, K, 3)
        d2 = np.einsum("mkd,mkd->mk", diff, diff)
        E = coeffs[None, :] * np.exp(-0.5 * widths[None, :] * d2)
        value[s:s + step] = E.sum(axis=1)
        if want_gradient:
            grad[s:s + step] = -np.einsum("mk,mkd->md", E * widths[None, :], diff)
    return value, grad


def smoothed_cofield(model: CMFModel, f: int, points: np.ndarray, width: Optional[float] = None,
                     include_scale: bool = True) -> np.ndarray:
    """The co-field of field `f` at `points`, optionally smoothed by an atom's own Gaussian.

    With `width` of None this is the co-field itself, which is what
    :func:`cmfqsar.cofields.cofield_at_points` gives, computed exactly rather than with a
    neighbour cutoff.  With `width` set to the ``b`` of an atom's Gaussian (``alpha`` for an
    ordinary field, ``alpha / R_vdw ** 2`` for the steric one), the value at a point is what an
    atom of unit weight centred there would contribute to the prediction, before the mixing
    coefficient and the field scaling that `include_scale` applies.
    """
    centres, coeffs, widths = _gaussian_terms(model, f, width)
    value, _ = _cofield_sum(centres, coeffs, widths, np.atleast_2d(np.asarray(points, float)),
                            want_gradient=False)
    if include_scale:
        value = value * model.fieldset.specs[f].h * model.scale[f]
    return value


def cofield_gradient(model: CMFModel, f: int, points: np.ndarray, width: Optional[float] = None,
                     include_scale: bool = True) -> np.ndarray:
    """The gradient of that same function, (M, 3).

    This is the co-field read as a potential: at a point in the binding site it says which way a
    piece of field of this type would have to move to raise the prediction.  It is the quantity a
    generative model can be steered by, and it needs no molecule, only a place to stand.

    For an unnormalised linear model the two gradients are the same statement: the derivative of
    the prediction with respect to the position of atom *l* is ``h_f c_f w_fl`` times this
    gradient, taken at the atom's own position and smoothed by the atom's own Gaussian, summed
    over the fields.  :func:`prediction_gradient` is the form to use on a molecule, because it
    also carries the normalisation and the nonlinear layer; this one is the form to use at a point
    where no atom is yet.
    """
    centres, coeffs, widths = _gaussian_terms(model, f, width)
    _, grad = _cofield_sum(centres, coeffs, widths, np.atleast_2d(np.asarray(points, float)),
                           want_gradient=True)
    if include_scale:
        grad = grad * (model.fieldset.specs[f].h * model.scale[f])
    return grad


# --------------------------------------------------------------------------- #
# The rotation-invariant kernel (plan item MS3)
# --------------------------------------------------------------------------- #

def _check_invariant_model(model: CMFModel) -> None:
    """The invariant sources this gradient covers, with a reason for each one it does not."""
    if model.kernel_source != "autocorrelation":
        raise ValueError(
            f"the gradient of an invariant kernel is implemented for the field "
            f"autocorrelation; this model uses the '{model.kernel_source}' kernel, whose "
            f"channels are built differently -- the heights above a surface, the images and the "
            f"uniform background of a periodic cell, the structure factors of reciprocal space, "
            f"the angles of a triple -- and each of those needs a derivative of its own.")
    if any(getattr(sp, "is_probe", False) for sp in model.fieldset.specs):
        raise ValueError(
            "a probe field carries its weights per Gaussian term and not per atom, and the "
            "invariant channels are built from those expanded terms, so there is no one weight "
            "per atom to differentiate. Fit the model without probe fields for its gradient.")
    if getattr(model, "ac_train_", None) is None:
        raise RuntimeError("the model carries no channel rows of its training set: refit it")


def _invariant_prediction_gradient(model: CMFModel, mol: MoleculeFields,
                                   coords: Optional[np.ndarray],
                                   weights: Optional[np.ndarray],
                                   wrt: str) -> PredictionGradient:
    """The derivatives of a prediction made on the rotation-invariant autocorrelation kernel.

    The kernel is the cosine of the angle between two scaled channel rows, so the prediction is

        y(x) = sum_j a_j (u(x) . p_j) / (L(x) |p_j|) + b,    u(x) = s * F(x),  L = |u(x)|,

    and the sum over the training molecules collapses into one vector in channel space,
    c = sum_j a_j p_j / |p_j|, which is what :func:`cmfqsar.invariant.channel_coefficients`
    returns.  The derivative in the channels is then

        dy/dF_k = s_k [ c_k / L - (u . c) u_k / L^3 ],

    where the second term is the row length the cosine divides out -- the one number plan item
    M15 measured -- and the channels are differentiated with respect to the atoms by
    :func:`cmfqsar.invariant.autocorrelation_gradient`.

    A nonlinear kernel layer multiplies every dual coefficient by the layer's derivative at that
    entry, so the collapse above holds with a reweighted dual vector.  The layer also depends on
    the molecule's own self-kernel, which for a cosine kernel is one for every molecule and
    therefore contributes nothing, exactly as it contributes nothing to a normalised overlap
    kernel.
    """
    from .invariant import autocorrelation_gradient

    work = _working_copy(mol, coords, weights)
    one = AtomTable.from_molecules([work])
    specs = list(model.fieldset.specs)

    s = np.asarray(model.ac_column_scale, float)
    u = np.asarray(model._ac_features(one), float)[0] * s
    L = float(np.sqrt(max(float(u @ u), TINY)))
    A = np.asarray(model.ac_train_, float) * s
    na = np.sqrt(np.maximum(np.einsum("jc,jc->j", A, A), TINY))
    A = A / na[:, None]                                  # the training rows the kernel sees

    K_lin = A @ (u / L)                                  # (n_train,), the cosine kernel row
    K_layer, dL_dK, _ = _layer_terms(model, K_lin, 1.0)
    coef = np.asarray(model.coef, float)
    value = float(K_layer @ coef + model.bias)

    c = A.T @ (coef * dL_dK)
    g_chan = s * (c / L - float(u @ c) * u / L ** 3)
    g_coords, g_weights = autocorrelation_gradient(work, g_chan, spec=model._ac_spec(), wrt=wrt)
    return PredictionGradient(value=value, coords=g_coords, weights=g_weights,
                              field_names=[sp.name for sp in specs])
