"""Scoring a conformational restriction (plan item MC9).

Rigidifying a molecule -- tying two parts of it together into a ring so that it can no longer
rotate freely -- is one of the few moves in medicinal chemistry whose effect is meant to be purely
conformational.  The fourth-dimension verdict of this repository is that an ensemble of
conformations does not predict better than one geometry (plan item D9), and plan item D10 found
that the variance of the field across an ensemble adds nothing as a block of the kernel either.
The variance is still a map of **where a molecule places field only part of the time**, and this
module asks the one question that map is suited to: when a chemist closes a ring, how much of the
field that matters for activity stops moving?

Two numbers describe a restriction
----------------------------------
The co-field :math:`C_f(r)` of a fitted model says how much the field of type *f* at point *r*
matters for the activity.  Its **square** is therefore a weight: positive everywhere, large where
the model reads the field and near zero where it does not.  With that weight, one molecule gets
two amounts:

* **relevant** :math:`\\int C_f(r)^2 \\bar{X}_f(r)^2 dr`, how much activity-relevant field the
  molecule places, averaged over its conformations;
* **intermittent** :math:`\\int C_f(r)^2 V_f(r) dr`, how much of that field is there only part of
  the time, with :math:`V_f` the variance field of :mod:`cmfqsar.fluctuation`.

Both are summed over the field types.  A pair of a flexible molecule **A** and its restricted
analogue **B** is then scored by two shares:

* **kept** = relevant(B) / relevant(A) -- did the restriction keep the field that matters, or did
  closing the ring move it away?
* **removed** = 1 - intermittent(B) / intermittent(A) -- how much of the intermittent part the
  restriction took out.

A restriction that works should keep the first and raise the second.  The gate of plan item MC9
is that **removed** tracks the measured potency gain over a set of such pairs;
`benchmarks/run_restriction.py` measures it.

Both members of a pair must come from the same conformer generator with the same settings: the
comparison is between two ensembles, not between two molecules, and a different number of
conformations changes the variance by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cofields import Grid, field_volume, make_grid, per_molecule_fields
from .fields import MoleculeFields

#: default grid spacing, in angstroms; the amounts are integrals and do not need a fine grid
DEFAULT_SPACING = 0.75
#: default margin around the molecules, in angstroms
DEFAULT_MARGIN = 3.0
#: how much of the smaller member the common substructure must cover
MCS_SHARE = 0.8
#: how many heavy atoms the restricted analogue may add
MAX_EXTRA_ATOMS = 2
#: how far the two atom compositions may differ, counted element by element
MAX_COMPOSITION = 2


# --------------------------------------------------------------------------- #
# Finding the pairs
# --------------------------------------------------------------------------- #


@dataclass
class Pair:
    """A flexible molecule and the analogue that ties part of it into a ring."""
    flexible: int
    restricted: int
    common: float
    rotatable_removed: int
    gain: float
    names: Tuple[str, str] = ("", "")

    def __str__(self) -> str:
        return (f"{self.names[0]} -> {self.names[1]}: {self.rotatable_removed} rotatable bonds "
                f"fewer, {self.common:.0%} of the atoms in common, potency {self.gain:+.2f}")


def _profile(mol):
    from collections import Counter

    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    m = Chem.RemoveHs(mol)
    return dict(rings=rdMolDescriptors.CalcNumRings(m),
                rotatable=rdMolDescriptors.CalcNumRotatableBonds(m),
                heavy=m.GetNumHeavyAtoms(), mol=m,
                composition=Counter(a.GetSymbol() for a in m.GetAtoms()))


def _composition_distance(a, b) -> int:
    """How many atoms the two molecules differ by, counted element by element."""
    ca, cb = a["composition"], b["composition"]
    return int(sum(abs(ca.get(k, 0) - cb.get(k, 0)) for k in set(ca) | set(cb)))


def _common_share(a, b, timeout: int = 2) -> float:
    """The share of the smaller molecule that the two have in common, by atom count."""
    from rdkit.Chem import rdFMCS
    res = rdFMCS.FindMCS([a["mol"], b["mol"]], timeout=int(timeout),
                         ringMatchesRingOnly=False, completeRingsOnly=False)
    if res.canceled or res.numAtoms <= 0:
        return 0.0
    return float(res.numAtoms) / float(min(a["heavy"], b["heavy"]))


def matched_pairs(mols: Sequence, y: Sequence[float], names: Optional[Sequence[str]] = None,
                  mcs_share: float = MCS_SHARE, max_extra: int = MAX_EXTRA_ATOMS,
                  max_composition: int = MAX_COMPOSITION, timeout: int = 2) -> List[Pair]:
    """Every pair in a set where the second molecule is the first with one more ring.

    The rule is deliberately mechanical, so that the pairs are not chosen by eye and the
    measurement cannot be steered.  The restricted member has exactly one ring more, at least one
    rotatable bond fewer, at most `max_extra` heavy atoms more, an atom composition differing by
    at most `max_composition` atoms counted element by element, and a maximum common substructure
    covering at least `mcs_share` of the smaller of the two.  The last condition is the one that
    decides, and a ring closure that also changes a substituent fails it; the ones before it are
    cheap, and keep the common-substructure search off most of the pairs in a set.

    A fingerprint is **not** used here, although it is the usual way to ask whether two molecules
    are alike.  Closing a ring changes the environment of every atom near the new bond, so
    butylphenol and 2-methylchroman -- the same eleven heavy atoms, one bond more -- are only
    0.15 alike as Morgan fingerprints, and a threshold high enough to be useful would throw away
    exactly the pairs this item is about.  The atom composition has no such blind spot.
    """
    y = np.asarray(y, float)
    rows = [None if m is None or not np.isfinite(y[i]) else _profile(m)
            for i, m in enumerate(mols)]
    out: List[Pair] = []
    for i, a in enumerate(rows):
        if a is None:
            continue
        for j, b in enumerate(rows):
            if b is None or i == j:
                continue
            if b["rings"] != a["rings"] + 1 or b["rotatable"] >= a["rotatable"]:
                continue
            if not 0 <= b["heavy"] - a["heavy"] <= int(max_extra):
                continue
            if _composition_distance(a, b) > int(max_composition):
                continue
            share = _common_share(a, b, timeout)
            if share < mcs_share:
                continue
            out.append(Pair(flexible=i, restricted=j, common=share,
                            rotatable_removed=int(a["rotatable"] - b["rotatable"]),
                            gain=float(y[j] - y[i]),
                            names=(names[i] if names else str(i),
                                   names[j] if names else str(j))))
    return out


# --------------------------------------------------------------------------- #
# The two amounts
# --------------------------------------------------------------------------- #


@dataclass
class Amounts:
    """Per molecule: the activity-relevant field it places, and how much of it is intermittent."""
    relevant: np.ndarray                      # (n_molecules,)
    intermittent: np.ndarray                  # (n_molecules,)
    per_field: np.ndarray = _field(default_factory=lambda: np.zeros((0, 0, 2)))
    fields: List[str] = _field(default_factory=list)
    grid: Optional[Grid] = None

    @property
    def intermittent_share(self) -> np.ndarray:
        """How much of the activity-relevant field of each molecule is there only part of time."""
        return np.where(self.relevant > 0, self.intermittent / np.maximum(self.relevant, 1e-300),
                        np.nan)


def training_fields(model, grid: Grid) -> List[np.ndarray]:
    """One (training molecules, grid points) array per enabled field: the co-field basis.

    Every co-field of this model, with any coefficients, is a combination of these rows, so the
    grid work is done once and the leave-one-pair-out co-fields below cost a matrix product each.
    """
    points = grid.points()
    return [np.asarray(per_molecule_fields(model, f, points), float)
            for f, _s in enumerate([s for s in model.fieldset.specs if s.enabled])]


def cofield_from(basis: Sequence[np.ndarray], model, rows: Optional[Sequence[int]] = None,
                 coef: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """The co-field of every field on the grid points, with its scales included.

    `rows` and `coef` refit nothing; they choose which training molecules the co-field is built
    from and with what dual coefficients.  That is how a co-field that has not seen a pair is
    made: solve the ridge without the two members, and pass the coefficients here with their rows.
    """
    specs = [s for s in model.fieldset.specs if s.enabled]
    a = np.asarray(model.coef if coef is None else coef, float)
    out = []
    for f, spec in enumerate(specs):
        X = basis[f] if rows is None else basis[f][np.asarray(rows, int)]
        if len(a) != len(X):
            raise ValueError(f"{len(a)} coefficients for {len(X)} training molecules")
        out.append((a @ X) * float(spec.h) * float(model.scale[f]))
    return out


def cofield_on_grid(model, grid: Grid, rows: Optional[Sequence[int]] = None,
                    coef: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """The co-field on the grid, evaluating the basis on the spot."""
    return cofield_from(training_fields(model, grid), model, rows, coef)


def grid_volumes(means: Sequence[MoleculeFields], variances: Sequence[MoleculeFields],
                 specs, vspecs, grid: Grid, rows: Optional[Sequence[int]] = None,
                 progress=None) -> np.ndarray:
    """(molecules, fields, 2, grid points): the mean field and the variance field of each.

    The two slices of the third axis are the mean field and the variance field.  They are held as
    32-bit numbers because they are only ever summed against a weight, and a set of a few hundred
    molecules would otherwise need gigabytes.
    """
    rows = list(range(len(means))) if rows is None else [int(r) for r in rows]
    out = np.zeros((len(rows), len(specs), 2, grid.n_points), dtype=np.float32)
    for k, m in enumerate(rows):
        for f in range(len(specs)):
            out[k, f, 0] = field_volume(means[m], specs[f], f, grid).reshape(-1)
            out[k, f, 1] = np.maximum(field_volume(variances[m], vspecs[f], f, grid), 0.0
                                      ).reshape(-1)
        if progress is not None:
            progress(f"Fields on the grid {k + 1}/{len(rows)}", (k + 1) / max(len(rows), 1))
    return out


def amounts_from(cofields: Sequence[np.ndarray], volumes: np.ndarray, grid: Grid,
                 fields: Optional[Sequence[str]] = None) -> Amounts:
    """The two amounts of every molecule, from grid volumes that are already built."""
    n, n_f = volumes.shape[0], volumes.shape[1]
    per = np.zeros((n, n_f, 2))
    cell = float(grid.spacing) ** 3
    for f in range(n_f):
        w = np.asarray(cofields[f], float).reshape(-1) ** 2
        xbar = volumes[:, f, 0, :]
        per[:, f, 0] = (xbar.astype(float) ** 2) @ w * cell
        per[:, f, 1] = volumes[:, f, 1, :].astype(float) @ w * cell
    return Amounts(relevant=per[:, :, 0].sum(axis=1), intermittent=per[:, :, 1].sum(axis=1),
                   per_field=per, fields=list(fields or [str(i) for i in range(n_f)]), grid=grid)


def weighted_amounts(cofields: Sequence[np.ndarray], means: Sequence[MoleculeFields],
                     variances: Sequence[MoleculeFields], specs, vspecs, grid: Grid,
                     progress=None) -> Amounts:
    """The two amounts of every molecule, from its mean field and its variance field.

    `cofields` holds one array per field over the grid points, as `cofield_on_grid` returns them;
    `specs` and `vspecs` are the field descriptions of the mean fields and of the variance fields
    (the variance of a Gaussian field is a Gaussian field of twice the width, which is why the
    second set is needed; see :func:`cmfqsar.fluctuation.variance_fieldset`).
    """
    volumes = grid_volumes(means, variances, specs, vspecs, grid, progress=progress)
    return amounts_from(cofields, volumes, grid,
                        [getattr(s, "name", str(i)) for i, s in enumerate(specs)])


def grid_for(means: Sequence[MoleculeFields], margin: float = DEFAULT_MARGIN,
             spacing: float = DEFAULT_SPACING) -> Grid:
    """One grid wide enough for every molecule of the set."""
    coords = np.vstack([m.coords for m in means])
    return make_grid(coords, margin=margin, spacing=spacing)


# --------------------------------------------------------------------------- #
# The score of a pair
# --------------------------------------------------------------------------- #


@dataclass
class Score:
    """What one restriction did, as two shares."""
    kept: float
    removed: float
    gain: float
    names: Tuple[str, str] = ("", "")

    def __str__(self) -> str:
        return (f"{self.names[0]} -> {self.names[1]}: kept {self.kept:.2f} of the relevant "
                f"field, removed {self.removed:+.2f} of the intermittent part, "
                f"potency {self.gain:+.2f}")


def score_pair(amounts: Amounts, pair: Pair) -> Score:
    """The two shares of one pair."""
    a, b = int(pair.flexible), int(pair.restricted)
    rel_a = float(amounts.relevant[a])
    int_a = float(amounts.intermittent[a])
    kept = float(amounts.relevant[b] / rel_a) if rel_a > 0 else float("nan")
    removed = float(1.0 - amounts.intermittent[b] / int_a) if int_a > 0 else float("nan")
    return Score(kept=kept, removed=removed, gain=float(pair.gain), names=pair.names)


def score_pairs(amounts: Amounts, pairs: Sequence[Pair]) -> List[Score]:
    return [score_pair(amounts, p) for p in pairs]


def correlation(x: Sequence[float], y: Sequence[float], runs: int = 2000,
                seed: int = 0) -> Tuple[float, float, float]:
    """Pearson correlation of two columns with a bootstrap interval over the pairs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.RandomState(int(seed))
    got = []
    for _ in range(int(runs)):
        i = rng.randint(0, len(x), len(x))
        if np.std(x[i]) <= 0 or np.std(y[i]) <= 0:
            continue
        got.append(float(np.corrcoef(x[i], y[i])[0, 1]))
    if not got:
        return (point, float("nan"), float("nan"))
    return (point, float(np.percentile(got, 2.5)), float(np.percentile(got, 97.5)))
