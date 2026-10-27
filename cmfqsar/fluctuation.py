"""The variance of a field across the frames of an ensemble (plan item D10).

The ensemble mode of this program represents a molecule by the **mean** of the fields of its
conformations (:func:`cmfqsar.fields.combine_members`), which is the occupancy averaging of
4D-QSAR.  A mean throws away how much the field moved, and that is a flexibility map: where a
molecule's field is the same in every conformation, and where it is not.

The variance is a field of the same class, which is what makes it usable without a new kernel.
With the Gaussians written as ``exp(-b/2 |r - c|^2)``, as everywhere else in this program,

    X_m(r) = sum_i w_i exp(-b/2 |r - r_mi|^2)

for conformation *m*, and the variance across conformations is

    V(r) = (1/k) sum_m X_m(r)^2  -  ( (1/k) sum_m X_m(r) )^2.

Every term is a product of two Gaussians, and a product of two Gaussians is a Gaussian:

    exp(-b/2 |r-p|^2) exp(-b/2 |r-q|^2) = exp(-b |r - (p+q)/2|^2) exp(-b |p-q|^2 / 4),

so V is again a sum of Gaussians -- at the **midpoints of atom pairs**, with **twice the width**,
and with a coefficient that falls off with how far apart the two atoms are.  The same-conformation
pairs enter with weight 1/k - 1/k^2 and the pairs from different conformations with -1/k^2, which
is the two terms of the variance written out.

:func:`variance_field` builds that field, and :func:`variance_fieldset` gives the field set it has
to be used with: the widths are doubled, so a variance field must not be mixed with an ordinary
one in the same kernel call.  Pairs whose coefficient is below a floor are dropped, which is what
keeps the construction from growing with the square of the atoms; :func:`kept_share` says what the
floor cost.

**What this is for.** The plan expects it to be a display tool -- a map of where a series is
rigid and where it is not -- and asks for it to be measured as a block of its own under multiple
kernel learning, so that the mixing coefficient says whether it carries anything the mean does
not.  `benchmarks/run_fluctuation.py` does that.

**What it is not.** Only fields whose Gaussians share one width can be written this way, which is
every "gauss" field of a field set at one attenuation.  The steric field puts a different width on
every atom (b = alpha / R^2), so a pair of atoms of different radii gives a product Gaussian whose
width depends on the pair, which a single field set cannot express; :func:`variance_fieldset`
refuses it rather than approximating.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .fields import FieldSet, FieldSpec, MoleculeFields

#: pairs whose Gaussian overlap factor is below this are left out of the variance field
PAIR_FLOOR = 1e-3


def variance_fieldset(fieldset: FieldSet) -> FieldSet:
    """The field set a variance field has to be evaluated with: the same fields at twice the width.

    A product of two Gaussians of width `b` is a Gaussian of width `2b`, so the variance field
    lives at twice the attenuation of the field it came from.  A field whose width is not the same
    on every atom cannot be written this way and is refused.
    """
    out = fieldset.copy()
    for spec in out.specs:
        if not spec.enabled:
            continue
        if spec.kind != "gauss":
            raise ValueError(
                f"the '{spec.name}' field puts a different Gaussian on every atom ({spec.kind}), "
                f"so the variance of it across conformations is not a field of one width; drop "
                f"it from the field set, or use the fields that are 'gauss'")
        spec.alpha = 2.0 * float(spec.alpha)
    return out


def _pairs(coords_a: np.ndarray, coords_b: np.ndarray, b: float,
           floor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, cols, factors) of the atom pairs whose product Gaussian survives the floor.

    Every ordered pair is enumerated, including both (i, j) and (j, i), because the square
    of a sum has both and halving them would halve the field.
    """
    d2 = np.sum((coords_a[:, None, :] - coords_b[None, :, :]) ** 2, axis=-1)
    factor = np.exp(-0.25 * float(b) * d2)
    i, j = np.where(factor >= float(floor))
    return i, j, factor[i, j]


def variance_field(members: Sequence[MoleculeFields], fieldset: FieldSet,
                   weights: Optional[Sequence[float]] = None, floor: float = PAIR_FLOOR,
                   name: str = "") -> MoleculeFields:
    """The variance of the fields of `members` across the ensemble, as a field.

    `members` are the fields of one molecule in its conformations, in the same atom order.  The
    result is a :class:`~cmfqsar.fields.MoleculeFields` whose atoms are pseudo-atoms at the
    midpoints of atom pairs; it must be used with :func:`variance_fieldset`, because its Gaussians
    are twice as wide as the ones it was built from.

    The weights need not be uniform: a Boltzmann-weighted ensemble has the same algebra with
    ``1/k`` replaced by the member's weight.
    """
    if len(members) < 2:
        raise ValueError("the variance across an ensemble needs at least two conformations")
    n = members[0].n_atoms
    if any(m.n_atoms != n for m in members):
        raise ValueError("the conformations of one molecule must have the same atoms")
    specs = [s for s in fieldset.specs if s.enabled]
    if any(s.kind != "gauss" for s in specs):
        raise ValueError("only fields of one width per field have a variance field; see "
                         "variance_fieldset")
    k = len(members)
    w = (np.ones(k) if weights is None else np.asarray(weights, float))
    if w.min() < 0 or w.sum() <= 0:
        raise ValueError("the member weights have to be non-negative and not all zero")
    w = w / w.sum()
    b = float(specs[0].alpha)
    if any(abs(float(s.alpha) - b) > 1e-12 for s in specs):
        raise ValueError("the variance field needs one attenuation for every field; set_alpha "
                         "with no names gives that")

    coords = [np.asarray(m.coords, float) for m in members]
    W = np.asarray(members[0].weights, float)
    centres: List[np.ndarray] = []
    pieces: List[np.ndarray] = []
    for m in range(k):
        for q in range(k):
            # the mean-of-squares term, which exists only within one conformation, minus the
            # square-of-the-mean term, which runs over every pair of conformations
            scale = (w[m] if m == q else 0.0) - w[m] * w[q]
            if abs(scale) < 1e-15:
                continue
            i, j, factor = _pairs(coords[m], coords[q], b, floor)
            if not len(i):
                continue
            mid = 0.5 * (coords[m][i] + coords[q][j])
            # a pseudo-atom carries the product of the two atoms' weights in every field
            coeff = (W[i] * W[j]) * (scale * factor)[:, None]
            centres.append(mid)
            pieces.append(coeff)
    if not centres:
        raise ValueError("every pair fell below the floor, which leaves no variance field")
    coords_out = np.vstack(centres)
    weights_out = np.vstack(pieces)
    return MoleculeFields(
        name=name or f"{members[0].name or 'molecule'} variance",
        coords=coords_out, weights=weights_out,
        radii=np.full(len(coords_out), float(np.mean(members[0].radii))),
        symbols=["X"] * len(coords_out), atom_types=["X"] * len(coords_out),
        bonds=np.zeros((0, 2), int), charges=np.zeros(len(coords_out)),
        charge_source=members[0].charge_source)


def kept_share(members: Sequence[MoleculeFields], fieldset: FieldSet,
               floor: float = PAIR_FLOOR) -> float:
    """The share of the pair factors the floor keeps, in [0, 1]."""
    specs = [s for s in fieldset.specs if s.enabled]
    b = float(specs[0].alpha)
    total = kept = 0.0
    for a in members:
        for c in members:
            d2 = np.sum((np.asarray(a.coords, float)[:, None, :]
                         - np.asarray(c.coords, float)[None, :, :]) ** 2, axis=-1)
            f = np.exp(-0.25 * b * d2)
            total += float(f.sum())
            kept += float(f[f >= float(floor)].sum())
    return kept / total if total > 0 else 1.0


def evaluate(field: MoleculeFields, fieldset: FieldSet, f: int,
             points: np.ndarray) -> np.ndarray:
    """One field of a :class:`~cmfqsar.fields.MoleculeFields` at a list of points.

    Used to check a variance field against the variance it is supposed to be, and to draw one.
    """
    from .cofields import evaluate_gaussians

    spec = [s for s in fieldset.specs if s.enabled][f]
    w = np.asarray(field.weights, float)[:, f]
    b = np.full(len(w), float(spec.alpha))
    return evaluate_gaussians(np.asarray(points, float), np.asarray(field.coords, float), w, b)
