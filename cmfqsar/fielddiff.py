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

The cosine weighs the two molecules alike, which is the wrong question for half of this module's
purpose.  A fragment cut from a lead reproduces part of its fields and nothing else, so the
cosine divides what it does reproduce by the norm of the whole lead and reports a poor match for
a perfect sub-structure.  :func:`tversky` weights the missing part and the added part separately
and so can be asked "how much of the candidate does the reference account for" instead, which is
the measure for growing a fragment or screening smaller molecules against a lead.

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
                 fieldset: FieldSet, alpha: Optional[float] = None,
                 beta: Optional[float] = None) -> List[Tuple[int, float]]:
    """(index, similarity) of every candidate against the reference, best first.

    The similarity is the overall cosine by default.  Give `alpha` and `beta` to rank by
    Tversky's index instead, which is what a library of molecules smaller than the reference
    needs; see :func:`tversky` for what the two weights mean.
    """
    if alpha is None and beta is None:
        score = lambda c: overall_similarity(c, reference, fieldset)       # noqa: E731
    else:
        a = 1.0 if alpha is None else float(alpha)
        b = 1.0 if beta is None else float(beta)
        score = lambda c: tversky_similarity(c, reference, fieldset, a, b)  # noqa: E731
    out = [(i, score(c)) for i, c in enumerate(candidates)]
    out.sort(key=lambda t: -(t[1] if np.isfinite(t[1]) else -np.inf))
    return out


# --------------------------------------------------------------------------- #
# Asymmetric similarity: Tversky's index on the overlap integrals
# --------------------------------------------------------------------------- #

@dataclass
class TverskyScore:
    """Tversky's index of one field type, with the two one-sided parts it is built from."""
    field: str
    label: str
    index: float                   # the index itself; see the note on values above one
    overlap: float                 # O_cr, the shared part
    missing: float                 # O_rr - O_cr, what the reference has and the candidate lacks
    added: float                   # O_cc - O_cr, what the candidate has and the reference lacks

    def describe(self) -> str:
        return (f"{self.label:18s} T {self.index:+6.3f}  shared {self.overlap:9.3f}"
                f"  missing {self.missing:9.3f}  added {self.added:9.3f}")


def _tversky_ratio(overlap: float, self_candidate: float, self_reference: float,
                   alpha: float, beta: float) -> float:
    """O / (O + alpha * (O_rr - O) + beta * (O_cc - O)), the overlap form of Eq. 8.44.

    The two differences are taken as they come, without clamping them at zero.  That is what
    makes the identities exact: the expression is Tanimoto at alpha = beta = 1 and Dice at
    alpha = beta = 1/2 whatever the signs of the three integrals.  It is also why a one-sided
    index can exceed one, which the containment note in :func:`tversky` explains.
    """
    missing = float(self_reference) - float(overlap)
    added = float(self_candidate) - float(overlap)
    den = float(overlap) + float(alpha) * missing + float(beta) * added
    return float(overlap) / den if abs(den) > 1e-300 else float("nan")


def tversky(candidate: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet,
            alpha: float = 1.0, beta: float = 1.0) -> List[TverskyScore]:
    """Asymmetric field similarity per field type: Tversky's index on the overlap integrals.

    The cosine of :func:`mismatch` treats the two molecules alike, so a fragment can never
    resemble the lead it was cut from: whatever the fragment does reproduce is divided by the
    norm of the whole reference.  Tversky's index weights the two one-sided parts separately,

        T = O_cr / (O_cr + alpha * (O_rr - O_cr) + beta * (O_cc - O_cr))

    where `O_cr` is the overlap integral of the two fields and `O_cc`, `O_rr` the self-overlaps.
    `alpha` therefore weights what the **reference** presents and the candidate fails to
    reproduce, and `beta` what the **candidate** adds, which is the vocabulary
    :func:`mismatch` already reports as `missing` and `added`.  Four settings are worth knowing:

    ``alpha = beta = 1``
        the Tanimoto index of the two fields, and the default.
    ``alpha = beta = 0.5``
        the Dice index.
    ``alpha = 0, beta = 1``
        `O_cr / O_cc`: how much of the candidate the reference accounts for.  This is the
        setting for growing a fragment or screening a library of smaller molecules against a
        lead, because it charges the candidate for nothing that is merely absent from it.
    ``alpha = 1, beta = 0``
        `O_cr / O_rr`: how much of the reference the candidate covers, which asks the opposite
        question and rewards a candidate for leaving nothing out.

    The cosine of :func:`mismatch` is exactly the geometric mean of those last two, since their
    product is `O_cr^2 / (O_cc O_rr)`.  That is the precise sense in which it splits the
    difference between the two questions and answers neither.

    **A one-sided index can exceed one, and that is the containment signal rather than a bug.**
    An overlap integral is an inner product, not the measure of an intersection, so the
    set identity `|A - B| >= 0` has no counterpart here: where the reference field exceeds the
    candidate's everywhere, `O_cr` exceeds `O_cc` and `O_cr / O_cc` passes one.  A fragment of
    the reference, in the frame it was cut from, therefore scores one or a little above, and the
    amount above measures how much reference field surrounds it.  The shape literature reports
    Tversky scores the same way.  Nothing is clamped, because clamping would buy a tidier range
    at the price of the Tanimoto and Dice identities above.

    One further honesty note.  Tversky's index was defined on sets, and a field with signed
    weights is not one: the electrostatic overlap of two molecules can be negative while both
    self-overlaps are positive, and the index is then a formal ratio rather than a shared
    fraction.  The reading above holds for the non-negative fields, which is the steric,
    hydrophobic and hydrogen-bond ones and every continuous indicator field.
    """
    k_ab, d_a, d_b = _pair_kernels(candidate, reference, fieldset)
    out: List[TverskyScore] = []
    for f, spec in enumerate(fieldset.specs):
        o, occ, orr = float(k_ab[f]), float(d_a[f]), float(d_b[f])
        out.append(TverskyScore(field=spec.name, label=spec.label,
                                index=_tversky_ratio(o, occ, orr, alpha, beta),
                                overlap=o, missing=orr - o, added=occ - o))
    return out


def tversky_similarity(candidate: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet,
                       alpha: float = 1.0, beta: float = 1.0) -> float:
    """Tversky's index over all fields together, with the mixing coefficients applied.

    The counterpart of :func:`overall_similarity`, which is the cosine of the same three
    quantities; `alpha = beta = 1` gives the Tanimoto index of the mixed fields.
    """
    k_ab, d_a, d_b = _pair_kernels(candidate, reference, fieldset)
    h = np.asarray(fieldset.mixing_vector(), float)
    return _tversky_ratio(float(h @ k_ab), float(h @ d_a), float(h @ d_b), alpha, beta)
