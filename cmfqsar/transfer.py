"""Comparing two models by their co-fields (plan item F3).

A co-field is the model as a function of space: where more of a field raises the predicted
activity and where it lowers it.  Two models that were fitted on molecules sharing a frame
therefore have two functions on the same space, and the overlap integral of those two functions
is a similarity between the *models* -- not between their molecules, and not between their
predictions, but between what they ask the chemistry for.

It needs no grid.  With

    C_f^A(r) = sum_j a_j s_fj X_fj(r),      C_f^B(r) = sum_k b_k s_fk X_fk(r)

the overlap is a sum of overlap integrals of atom-centred Gaussians, which is the field kernel
between the two training sets:

    <C_f^A, C_f^B> = sum_jk a_j b_k s_fj s_fk K_f^raw(j, k) = a^T K_f(A, B) b

one matrix product on a kernel block the program computes for a prediction anyway.  The cosine

    cos_f = <C_f^A, C_f^B> / (||C_f^A|| ||C_f^B||)

is the number to read, because it is invariant to the scale of either co-field -- to the units of
either activity, to the mixing coefficients, to the field scaling -- and so compares the *shapes*
of two demands rather than their sizes.

What it is for, in the order the question usually arises:

* **Two targets.**  A positive cosine says the two targets want the same thing at the same place,
  which is why selectivity is hard; a negative one says they want opposite things, which is where
  selectivity comes from.  The multi-task selectivity co-field is the difference of the two
  co-fields, and this is its inner product -- the same algebra, read as an angle.
* **Two series on one target.**  Agreement means the SAR transfers; disagreement means two
  chemotypes are being fitted by two different stories and a model pooled over both will be an
  average of them.
* **One series across rounds.**  A model refitted after new compounds should move its co-field
  only where the new compounds taught it something; the cosine with the previous round says how
  much was learned and the per-field cosines say in which field.

**What it cannot do.**  Two co-fields are functions on one space only if their molecules were
posed in one frame: comparing models fitted in different frames compares two different coordinate
systems and means nothing, which is why :func:`incompatible` checks what it can (the same fields,
the same attenuations, overlapping boxes) and the caller is told rather than quietly given a
number.  And a cosine is not a prediction: that two models want the same thing does not mean
either is right.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fields import AtomTable


@dataclass
class FieldOverlap:
    """One field's share of the comparison."""

    field: str
    label: str
    cosine: float
    overlap: float
    norm_a: float
    norm_b: float

    def describe(self) -> str:
        return (f"{self.label:18s} cos {self.cosine:+6.3f}   overlap {self.overlap:+10.4g}"
                f"   |A| {self.norm_a:9.4g}  |B| {self.norm_b:9.4g}")


@dataclass
class ModelOverlap:
    """How much two models ask the same thing of the chemistry."""

    name_a: str
    name_b: str
    cosine: float                       # over all fields together, with each model's own scaling
    fields: List[FieldOverlap] = _field(default_factory=list)
    notes: List[str] = _field(default_factory=list)

    def by_field(self) -> Dict[str, float]:
        return {f.field: f.cosine for f in self.fields}

    def describe(self) -> str:
        agree = ("ask the same thing" if self.cosine > 0.3 else
                 "ask opposite things" if self.cosine < -0.3 else
                 "ask largely unrelated things")
        return (f"{self.name_a} against {self.name_b}: cosine {self.cosine:+.3f} "
                f"-- they {agree}")


def incompatible(a, b) -> Optional[str]:
    """Why these two models cannot be compared by their co-fields, or None if they can.

    The checks are the ones that can be made from the model files: both linear, both on the
    overlap kernel, the same fields with the same attenuations, and training boxes that actually
    overlap.  What cannot be checked is whether the two alignments mean the same thing, and that
    is the assumption the whole comparison rests on.
    """
    for m, which in ((a, "the first"), (b, "the second")):
        if not m.is_linear:
            return (f"{which} model has a '{m.kernel_layer}' layer, so it is not linear in the "
                    "field and has no co-field")
        if not m.uses_fields:
            return f"{which} model uses the '{m.kernel_source}' kernel, which has no co-field"
        if m.train is None or m.coef is None:
            return f"{which} model carries no fitted co-field"
    na = [s.name for s in a.fieldset.specs if s.enabled]
    nb = [s.name for s in b.fieldset.specs if s.enabled]
    if na != nb:
        return (f"different fields: {', '.join(na)} against {', '.join(nb)}; two co-fields can "
                "only be compared field by field")
    for sa, sb in zip(a.fieldset.specs, b.fieldset.specs):
        if abs(float(sa.alpha) - float(sb.alpha)) > 1e-9:
            return (f"the {sa.label} field has alpha {sa.alpha:g} in one model and "
                    f"{sb.alpha:g} in the other: two co-fields drawn with different widths are "
                    "different functions, and their cosine would mix the two effects")
    lo_a, hi_a = a.train.coords.min(axis=0), a.train.coords.max(axis=0)
    lo_b, hi_b = b.train.coords.min(axis=0), b.train.coords.max(axis=0)
    if np.any(np.minimum(hi_a, hi_b) < np.maximum(lo_a, lo_b)):
        return ("the training molecules of the two models do not occupy the same region of "
                "space at all, so they were not posed in one frame and their co-fields are "
                "functions on different coordinate systems")
    return None


def _norms(model) -> np.ndarray:
    """||C_f|| for every field, from the model's own training kernels."""
    if model.K_fields_train is None:
        model.K_fields_train = model._field_kernels(model.train, model.train, model.self_diag,
                                                    model.self_diag)
    a = np.asarray(model.coef, float)
    K = np.asarray(model.K_fields_train, float)
    return np.sqrt(np.maximum(np.einsum("i,fij,j->f", a, K, a), 0.0))


def cofield_overlap(model_a, model_b, name_a: str = "A", name_b: str = "B",
                    include_scale: bool = True, strict: bool = True) -> ModelOverlap:
    """The co-field cosine of two models, field by field and over all fields together.

    `include_scale` weights the fields by each model's own mixing coefficients and field
    scaling when the overall cosine is formed, which is how the models themselves weigh them;
    the per-field cosines do not depend on it. With `strict` off, a reason the two should not be
    compared becomes a note on the result instead of an exception, which is what a matrix over
    many models wants.
    """
    why = incompatible(model_a, model_b)
    notes = []
    if why:
        if strict:
            raise ValueError(why)
        return ModelOverlap(name_a=name_a, name_b=name_b, cosine=float("nan"), notes=[why])

    # the cross kernel between the two training sets, in the first model's (shared) fields
    K_ab = model_a._field_kernels(model_a.train, model_b.train,
                                  model_a.self_diag, model_b.self_diag)
    a = np.asarray(model_a.coef, float)
    b = np.asarray(model_b.coef, float)
    over = np.einsum("i,fij,j->f", a, np.asarray(K_ab, float), b)
    na, nb = _norms(model_a), _norms(model_b)

    rows = []
    for f, spec in enumerate(model_a.fieldset.specs):
        denom = na[f] * nb[f]
        rows.append(FieldOverlap(field=spec.name, label=spec.label,
                                 cosine=float(over[f] / denom) if denom > 1e-300 else float("nan"),
                                 overlap=float(over[f]), norm_a=float(na[f]),
                                 norm_b=float(nb[f])))
    wa = (model_a.fieldset.mixing_vector() * np.asarray(model_a.scale, float)
          if include_scale else np.ones(len(rows)))
    wb = (model_b.fieldset.mixing_vector() * np.asarray(model_b.scale, float)
          if include_scale else np.ones(len(rows)))
    num = float(np.sum(wa * wb * over))
    den = float(np.sqrt(np.sum((wa * na) ** 2)) * np.sqrt(np.sum((wb * nb) ** 2)))
    if np.any(~np.isfinite([r.cosine for r in rows])):
        notes.append("a field with no co-field at all was left out of its own cosine")
    return ModelOverlap(name_a=name_a, name_b=name_b,
                        cosine=float(num / den) if den > 1e-300 else float("nan"),
                        fields=rows, notes=notes)


def overlap_matrix(models: Sequence, names: Sequence[str],
                   include_scale: bool = True) -> Tuple[np.ndarray, List[ModelOverlap]]:
    """The full table: every model against every other, by the cosine of their co-fields.

    The diagonal is one by construction, and the matrix is symmetric, so only the upper triangle
    is computed. A pair that cannot be compared leaves nan and a note.
    """
    n = len(models)
    M = np.full((n, n), np.nan)
    pairs: List[ModelOverlap] = []
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            res = cofield_overlap(models[i], models[j], names[i], names[j],
                                  include_scale=include_scale, strict=False)
            pairs.append(res)
            M[i, j] = M[j, i] = res.cosine
    return M, pairs


def describe_matrix(M: np.ndarray, names: Sequence[str], width: int = 7) -> str:
    """The matrix as a text table, for the command line."""
    head = " " * 20 + "".join(f"{n[:width]:>{width + 1}s}" for n in names)
    lines = [head]
    for i, n in enumerate(names):
        row = "".join(f"{M[i, j]:+{width + 1}.3f}" if np.isfinite(M[i, j])
                      else f"{'--':>{width + 1}s}" for j in range(len(names)))
        lines.append(f"{n[:19]:20s}{row}")
    return "\n".join(lines)
