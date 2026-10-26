"""Handedness as a property of a centre, not of a shape.

C5 of the development plan added signed three-point channels to the field autocorrelation: the
triple product of three displacement vectors, summed over every triple of atoms in the molecule
and resolved over radial shells.  Measured, it does exactly what it claims about one molecule --
the channels of a molecule and its mirror image are exact negatives -- and nothing at all about
two molecules, which is what a model needs.  On twenty pairs the cosine of that block was -0.016
between molecules of the *same* handedness and +0.016 between molecules of opposite handedness,
a difference of 0.03 against a spread of 0.55: handedness was not a direction in that space, so
a kernel method had nothing to transfer, and an activity that was pure handedness could not be
learnt at any setting.

The failure is in the summation, not in the triple product.  A sum over every triple of atoms is
a property of the whole shape, and two molecules of the same configuration have no reason to
share one.  The configuration itself is local: it is the order in which three substituents stand
around a centre, which is what a chemist reads when assigning R or S, and what this module
measures.

For every atom with at least three neighbours, and for every field used as a priority:

1. order the neighbours by their weight in that field, largest first;
2. take the unit vectors from the centre to the first three;
3. the signed volume of that ordered frame is +1 or -1 in character and changes sign under
   reflection, is unchanged by any rotation or translation, and is the same number for two
   molecules whose substituents stand in the same order around a comparable centre.

Point 3 is the whole difference from C5, and the tests assert it directly.

Using the field weights as the priority rather than the CIP rules keeps the descriptor a
property of the fields -- the same quantities the rest of the program is built from -- and not of
a toolkit's implementation of a convention.  The cost is that two substituents of nearly equal
weight have no reliable order, and an unreliable order is an unreliable sign; such centres are
dropped rather than guessed at, and :func:`stereo_matrix` reports how many survived.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["StereoSpec", "channel_labels", "stereo_centres", "stereo_channels", "stereo_matrix"]


@dataclass(frozen=True)
class StereoSpec:
    """How a centre is read.

    `min_separation` is the margin, in units of the field's own spread over the molecule, that
    two neighbours' weights must differ by before their order is believed.  It matters more than
    it seems: at zero every atom with three neighbours contributes a sign, including the ones
    whose substituents are the same and whose order is therefore noise, and the handedness of two
    molecules separates by 0.18 against a spread of 0.53.  At 0.1 to 0.3 the tied centres drop
    out and the separation is 0.86 against 0.70.  Above 0.6 so few centres survive that it falls
    away again, so this is a real parameter and not a formality.
    """
    min_neighbours: int = 3
    min_separation: float = 0.2
    include_hydrogens: bool = True


def stereo_centres(mol, spec: StereoSpec = StereoSpec()) -> List[Tuple[int, np.ndarray]]:
    """(atom, its neighbours) for every atom that has enough of them to define a frame."""
    bonds = np.asarray(getattr(mol, "bonds", np.zeros((0, 2), int)), int).reshape(-1, 2)
    n = len(mol.coords)
    if not len(bonds) or n == 0:
        return []
    keep = np.ones(n, bool)
    if not spec.include_hydrogens:
        keep = np.array([s.upper() != "H" for s in mol.symbols], bool)
    neigh: List[List[int]] = [[] for _ in range(n)]
    for a, b in bonds:
        if keep[b]:
            neigh[a].append(int(b))
        if keep[a]:
            neigh[b].append(int(a))
    out = []
    for i in range(n):
        if keep[i] and len(neigh[i]) >= spec.min_neighbours:
            out.append((i, np.array(sorted(set(neigh[i])), dtype=int)))
    return out


def _ordered_volume(coords: np.ndarray, centre: int, neighbours: np.ndarray,
                    priority: np.ndarray, min_separation: float) -> float:
    """The signed volume of the three highest-priority neighbours, or 0 when the order is a tie.

    The vectors are normalised, so the value lies in [-1, 1] and says how the three stand
    around the centre rather than how far away they are.
    """
    if len(neighbours) < 3:
        return 0.0
    w = priority[neighbours]
    order = np.argsort(-w)
    top = neighbours[order[:3]]
    best = w[order[:3]]
    if min_separation > 0.0:
        spread = float(np.std(priority)) or 1.0
        if (best[0] - best[1]) < min_separation * spread or \
                (best[1] - best[2]) < min_separation * spread:
            return 0.0                      # an order this close is not an order
    v = coords[top] - coords[centre]
    norms = np.linalg.norm(v, axis=1)
    if np.any(norms < 1e-9):
        return 0.0
    u = v / norms[:, None]
    return float(np.linalg.det(u))


def stereo_channels(mol, spec: StereoSpec = StereoSpec(),
                    n_fields: Optional[int] = None) -> np.ndarray:
    """One number per (priority field, weighting field): the molecule's signed centres.

    The priority field decides the order of the substituents around a centre; the weighting
    field says how much that centre counts, through the centre's own weight in it.  A molecule
    with no usable centre returns zeros, which is the right answer: it has no handedness that
    these fields can see.
    """
    W = np.asarray(mol.weights, float)
    F = int(n_fields if n_fields is not None else W.shape[1])
    out = np.zeros((F, F))
    centres = stereo_centres(mol, spec)
    if not centres:
        return out.reshape(-1)
    coords = np.asarray(mol.coords, float)
    for f in range(F):
        priority = W[:, f]
        for i, neighbours in centres:
            v = _ordered_volume(coords, i, neighbours, priority, spec.min_separation)
            if v:
                out[f] += v * W[i, :F]
    return out.reshape(-1)


def channel_labels(fieldset, spec: StereoSpec = StereoSpec()) -> List[str]:
    names = [s.label for s in fieldset.specs]
    return [f"{a} order x {b} weight" for a in names for b in names]


def stereo_matrix(mols: Sequence, spec: StereoSpec = StereoSpec(),
                  n_fields: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """The channels of a set of molecules, and how many centres they were read from."""
    mols = list(mols)
    if not mols:
        return np.zeros((0, 0)), {"molecules": 0, "with_centre": 0, "centres": 0}
    F = int(n_fields if n_fields is not None else np.asarray(mols[0].weights).shape[1])
    X = np.vstack([stereo_channels(m, spec, n_fields=F) for m in mols])
    used = int(np.sum(np.any(np.abs(X) > 1e-12, axis=1)))
    centres = int(np.sum([len(stereo_centres(m, spec)) for m in mols]))
    return X, {"molecules": len(mols), "with_centre": used, "centres": centres}
