"""What one molecule's fields have that another's do not.

A model says how well a candidate scores; it does not say what is wrong with it.  For scaffold
hopping and peptidomimetic work the useful readout is the difference between the fields of the
molecule being imitated and those of the candidate: where the candidate fails to reproduce what
the reference presents, and where it puts something the reference never had.

Two views of the same quantity, both exact:

* **Per field type, in closed form.**  The squared distance between two fields is an overlap
  integral like any other,

      ||X_fa - X_fb||^2 = K_f(a, a) + K_f(b, b) - 2 K_f(a, b),

  so the mismatch of each field needs no grid at all, and the cosine of the two fields comes
  from the same three numbers.  This is what :func:`mismatch` reports.
* **Where in space**, on a grid: `X_f(candidate) - X_f(reference)`, whose negative lobes are
  what the candidate is missing and positive lobes what it adds.  :func:`difference_features`
  turns those lobes into the same typed features with tolerance radii that
  :mod:`cmfqsar.pharmacophore` reads out of a co-field, so the report says "missing acceptor at
  (x, y, z), radius 1.4 A" rather than only "cosine 0.71".

Both molecules must already be in the same frame: a difference map of two molecules that were
never aligned measures the alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cofields import Grid, difference_volume, make_grid
from .fields import AtomTable, FieldSet, MoleculeFields
from .kernels import kernel_matrices, self_kernels


@dataclass
class FieldMismatch:
    """How one field type of a candidate compares with the same field of a reference."""
    field: str
    label: str
    cosine: float                  # 1.0 = the same field, 0.0 = no overlap at all
    distance: float                # ||X_cand - X_ref||, in the units of the field
    norm_ref: float                # ||X_ref||
    norm_cand: float               # ||X_cand||
    missing: float = float("nan")  # integral of the field the candidate fails to reproduce
    added: float = float("nan")    # integral of the field it adds

    def describe(self) -> str:
        extra = ""
        if np.isfinite(self.missing):
            extra = f"  missing {self.missing:8.3f}  added {self.added:8.3f}"
        return (f"{self.label:18s} cosine {self.cosine:+6.3f}  distance {self.distance:8.3f}"
                f"  |ref| {self.norm_ref:8.3f}  |cand| {self.norm_cand:8.3f}{extra}")


def _pair_kernels(candidate: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet):
    """(K_f(cand, ref), K_f(cand, cand), K_f(ref, ref)) for every field, analytically."""
    specs = fieldset.specs
    ta = AtomTable.from_molecules([candidate])
    tb = AtomTable.from_molecules([reference])
    k_ab = kernel_matrices(ta, tb, specs)[:, 0, 0]
    d_a = self_kernels(ta, specs)[:, 0]
    d_b = self_kernels(tb, specs)[:, 0]
    return k_ab, d_a, d_b


def mismatch(candidate: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet,
             grid: Optional[Grid] = None, spacing: float = 0.5,
             margin: float = 3.0) -> List[FieldMismatch]:
    """Per-field comparison of a candidate with the molecule it is meant to imitate.

    The cosine and the distance are exact overlap integrals.  `missing` and `added` are the
    integrals of the negative and positive parts of the difference map and need a grid; pass
    ``grid=False`` to skip them when only the closed-form numbers are wanted.
    """
    k_ab, d_a, d_b = _pair_kernels(candidate, reference, fieldset)
    rows: List[FieldMismatch] = []
    use_grid = grid is not False
    if use_grid and grid is None:
        coords = np.vstack([candidate.coords, reference.coords])
        grid = make_grid(coords, margin=margin, spacing=spacing)
    cell = (grid.spacing ** 3) if use_grid else 0.0
    for f, spec in enumerate(fieldset.specs):
        na, nb = float(np.sqrt(max(d_a[f], 0.0))), float(np.sqrt(max(d_b[f], 0.0)))
        d2 = float(d_a[f] + d_b[f] - 2.0 * k_ab[f])
        cos = float(k_ab[f] / (na * nb)) if na > 1e-300 and nb > 1e-300 else float("nan")
        row = FieldMismatch(field=spec.name, label=spec.label, cosine=cos,
                            distance=float(np.sqrt(max(d2, 0.0))), norm_ref=nb, norm_cand=na)
        if use_grid:
            d = difference_volume(candidate, reference, spec, f, grid)
            row.added = float(np.clip(d, 0.0, None).sum() * cell)
            row.missing = float(np.clip(-d, 0.0, None).sum() * cell)
        rows.append(row)
    return rows


def overall_similarity(candidate: MoleculeFields, reference: MoleculeFields,
                       fieldset: FieldSet) -> float:
    """Cosine of the two molecules over all fields together, with the mixing coefficients."""
    k_ab, d_a, d_b = _pair_kernels(candidate, reference, fieldset)
    h = np.asarray(fieldset.mixing_vector(), float)
    num = float(h @ k_ab)
    den = float(np.sqrt(max(h @ d_a, 0.0) * max(h @ d_b, 0.0)))
    return num / den if den > 1e-300 else float("nan")


def difference_features(candidate: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet,
                        grid: Optional[Grid] = None, level: float = 0.5,
                        min_radius: float = 0.6, max_features: int = 10,
                        spacing: float = 0.5, margin: float = 3.0):
    """The lobes of the difference map, as typed features: what is missing, and what is extra.

    The kinds read "missing <field>" where the candidate fails to reproduce the reference and
    "extra <field>" where it adds something; the weight is the integral of the difference over
    the lobe, negative for what is missing.
    """
    from .pharmacophore import Feature, features_from_volume
    if grid is None:
        coords = np.vstack([candidate.coords, reference.coords])
        grid = make_grid(coords, margin=margin, spacing=spacing)
    out: List[Feature] = []
    for f, spec in enumerate(fieldset.specs):
        d = difference_volume(candidate, reference, spec, f, grid)
        for sign in (+1, -1):
            for feat in features_from_volume(d, grid, spec.name, sign, level=level,
                                             min_radius=min_radius):
                feat.kind = f"{'extra' if sign > 0 else 'missing'} {spec.label.lower()}"
                feat.favourable = sign > 0
                feat.importance = abs(feat.importance) * (1 if sign > 0 else -1)
                feat.meaning = ("the candidate puts field here that the reference does not have"
                                if sign > 0 else
                                "the candidate fails to reproduce the reference here")
                out.append(feat)
    out.sort(key=lambda ft: -abs(ft.importance))
    return out[:max_features]


def rank_against(reference: MoleculeFields, candidates: Sequence[MoleculeFields],
                 fieldset: FieldSet) -> List[Tuple[int, float]]:
    """(index, overall cosine) of every candidate against the reference, best first."""
    out = [(i, overall_similarity(c, reference, fieldset)) for i, c in enumerate(candidates)]
    out.sort(key=lambda t: -(t[1] if np.isfinite(t[1]) else -np.inf))
    return out
