"""Three-body channels and handedness for a periodic cell (plan item M10).

The periodic autocorrelation of :mod:`cmfqsar.periodic` is a **two-point** descriptor: it counts
how much field weight sits at each distance from how much other field weight.  Two structures
with the same multiset of interatomic distances therefore give exactly the same channels, however
differently their atoms are arranged.  Two such structures are called **homometric**, and they are
not a curiosity: a defect series on one host lattice is full of arrangements that share their
distance histogram, which is why `benchmarks/run_2dmd.py` finds the autocorrelation near the
accuracy of guessing on a set where the arrangement of the defects is the whole of the signal.
A mirror image is the other blind spot: reflecting a structure leaves every distance unchanged, so
no two-point descriptor can tell a left-handed crystal from a right-handed one.

This module closes both, in the terms of this program.

**The three-body block.**  Every atom of the cell is taken as a centre in turn.  Around a centre,
two neighbours and the centre form a triangle, and a triangle is fixed by its three sides: the two
distances from the centre and the distance between the neighbours.  Writing ``r_a`` for the
distance of neighbour *a* from the centre and ``d_ab`` for the distance between two neighbours::

    Phi_fg(k, k', m) = (1/N) sum over centres l of
                       sum over neighbours a, b of l of
                       w_af w_bg R_k(r_a) R_k'(r_b) D_m(d_ab),

with ``R_k`` a Gaussian window on a grid of radii from the centre, ``D_m`` one on a grid of
distances between neighbours, and ``N`` the atoms in the cell.  The neighbourhood of a centre holds
the centre itself (at radius zero, so its own field types enter the channels) and every atom of
every periodic image within the cutoff.  The channel layout is the one of
:mod:`cmfqsar.triangles`, which builds the same block around one atom of a molecule (plan item C5):
one block per pair of field types, and inside a block one column per pair of radial windows and
distance window.  Dividing by ``N`` makes a supercell give exactly the same channels as the cell it
repeats.

**The handedness block.**  Three distances fix a triangle only up to its mirror image, so the block
above is still blind to a reflection.  The signed quantity that is not is the triple product of
three vectors.  Around a centre::

    X_fgh(s) = (1/N) sum over centres l of det[ A_f(l, s), A_g(l, s), A_h(l, s) ],
    A_f(l, s) = sum over neighbours a of l of w_af G_s(r_a) (r_a - r_l),

with ``G_s`` a Gaussian window on a grid of radii, which keeps the handedness resolved by distance
from the centre rather than reduced to one number.  The development plan expected this block to
cost a sum over triples of atoms, because the factorisation that :func:`cmfqsar.invariant.chirality`
uses for a molecule needs an origin, and a periodic structure has no centroid to use.  With the
centre atom as that origin the factorisation survives in full: the determinant is linear in each of
its three arguments, so the sum over triples of neighbours **is** the determinant of three summed
vectors, and the block costs one pass over the neighbour list, not a cubic sum.  Under a proper
rotation all three vectors turn together and the determinant does not change; under a reflection it
changes sign; a centrosymmetric neighbourhood gives zero, as an achiral environment must.

**When the handedness block is empty.**  A determinant of three vectors is zero whenever they lie
in a plane, and this is not a rare case.  A field weight that depends only on the species of the
atom makes every ``A_f`` a combination of one vector per species, so a structure of two species
gives vectors in a plane at most and the whole block is identically zero; three species are the
minimum, and their vector sums must not be coplanar.  A site with a rotation axis through it also
gives zero, because a sum that is unchanged by the rotation can only lie along the axis.  The
block is therefore a sharp test for handedness and not a general-purpose descriptor: it is off by
default, and a structure that gives zero in every channel is telling the truth about itself.

**What it costs.**  One pass over the pairs of neighbours of every atom.  Measured by
`benchmarks/run_threebody.py --probe`: a cell of four atoms with a 4 angstrom reach and a
0.5 angstrom grid takes about 0.05 seconds and gives 11,322 channels, and a defect supercell of
191 atoms with a 0.75 angstrom grid takes about 0.26 seconds and gives 14,112.  The number of
channels grows as the square of (fields times radii), so the radial grid is coarse and the cutoff
short; the kernel works in the dual and does not mind the width.

**What is not covered.**  A disordered cell is refused: the mean of a product of three weights on
one site is not the product of their means, and this module tabulates no third moments.  No uniform
background is subtracted, unlike the two-point descriptor, so these channels carry how dense the
structure is as well as how it is arranged; a cosine-normalised kernel removes most of that, and
the two-point channels are there to carry the density properly.  The cutoff is sharp in the sense
that an atom beyond ``r_max + reach * sigma`` is left out, where it contributes less than
``exp(-reach^2 / 2)`` of a window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fields import FieldSet
from .invariant import chirality_triples, field_pairs
from .triangles import TriangleSpec

#: how far beyond the radial grid a neighbour still enters the sum, in widths of the window
DEFAULT_REACH = 4.0


@dataclass(frozen=True)
class ThreeBodySpec:
    """The grids the three-body channels of a cell are evaluated on.

    `r_max` is how far from a centre the radial grid reaches, in angstrom, and is the cutoff of the
    descriptor; the grid of distances between two neighbours reaches twice as far, which is the
    longest third side a triangle inside the cutoff can have.  `step` is the spacing of both grids
    and `sigma` the width of the Gaussian window on every grid point.  `include_self` keeps the
    terms in which the two neighbours are the same atom, which carry the radial density itself.
    With `chirality` on, the handedness block is added, resolved over `chirality_shells` radial
    windows of width `chirality_sigma` about the centre.
    """
    r_max: float = 4.0
    step: float = 0.5
    sigma: float = 0.35
    include_self: bool = True
    reach: float = DEFAULT_REACH
    chirality: bool = False
    chirality_shells: int = 4
    chirality_sigma: float = 0.5

    @property
    def triangle(self) -> TriangleSpec:
        """The channel layout of :mod:`cmfqsar.triangles`, which this block shares."""
        return TriangleSpec(r_max=float(self.r_max), step=float(self.step),
                            sigma=float(self.sigma), include_self=bool(self.include_self),
                            reach=float(self.reach))

    @property
    def radii(self) -> np.ndarray:
        return self.triangle.radii

    @property
    def distances(self) -> np.ndarray:
        return self.triangle.distances

    @property
    def n_radii(self) -> int:
        return self.triangle.n_radii

    @property
    def n_distances(self) -> int:
        return self.triangle.n_distances

    @property
    def cutoff(self) -> float:
        """Beyond this distance from a centre an atom is not in the sum."""
        return self.triangle.cutoff

    @property
    def n_bins(self) -> int:
        """The width of the innermost grid, which is the grid of distances between neighbours.

        The helpers that scale the channels and build the kernel
        (:func:`cmfqsar.invariant.channel_scale`, :func:`cmfqsar.invariant.autocorrelation_kernel`)
        ask a spec for one block width to fall back on.  The blocks of this descriptor are not all
        of one width, so they are always passed explicitly and this value is only the fallback.
        """
        return self.n_distances

    @property
    def n_shells(self) -> int:
        """Radial windows the handedness block is resolved over; one for a single number."""
        return max(int(self.chirality_shells), 1) if self.chirality else 0

    @property
    def shell_centres(self) -> np.ndarray:
        """Where the windows of the handedness block sit, measured from the centre atom."""
        if not self.chirality:
            return np.zeros(0)
        s = self.n_shells
        if s <= 1:
            return np.zeros(1)
        return np.linspace(0.0, float(self.r_max), s)

    def radial_pairs(self, same: bool) -> List[Tuple[int, int]]:
        return self.triangle.radial_pairs(same)

    def block_sizes(self, n_fields: int) -> List[int]:
        """Channels per block: the pair blocks first, then the handedness blocks."""
        out = self.triangle.block_sizes(int(n_fields))
        if self.chirality:
            out = out + [self.n_shells] * len(chirality_triples(int(n_fields)))
        return out

    def n_channels(self, n_fields: int) -> int:
        return int(sum(self.block_sizes(int(n_fields))))


def _windows(x: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-((x[..., None] - grid) ** 2) / (2.0 * float(sigma) ** 2))


def neighbourhoods(crystal, cutoff: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """For every atom of the cell, the vectors to its neighbours and their field weights.

    The neighbourhood of an atom holds the atom itself, at the vector zero, and every atom of
    every periodic image whose distance is below `cutoff`.  The vectors are measured from the
    centre, so they do not change when the structure is moved, and they turn with it when it is
    turned.  The periodicity flags of the cell are respected: a layer has neighbours in two
    directions and a chain in one.
    """
    from ase.neighborlist import neighbor_list

    W = np.asarray(crystal.weights, float)
    n_atoms = W.shape[0]
    if n_atoms == 0:
        return []
    i, j, D = neighbor_list("ijD", crystal.to_atoms(), float(cutoff))
    i = np.asarray(i, int)
    order = np.argsort(i, kind="stable")
    i, j, D = i[order], np.asarray(j, int)[order], np.asarray(D, float)[order]
    starts = np.searchsorted(i, np.arange(n_atoms))
    ends = np.searchsorted(i, np.arange(n_atoms), side="right")
    out = []
    for c in range(n_atoms):
        s, e = int(starts[c]), int(ends[c])
        vectors = np.vstack([np.zeros((1, 3)), D[s:e]])
        weights = np.vstack([W[c:c + 1], W[j[s:e]]])
        out.append((vectors, weights))
    return out


def _refuse_disordered(crystal) -> None:
    if getattr(crystal, "site_species", None) is not None:
        raise ValueError(
            f"{getattr(crystal, 'name', '') or 'this structure'} has sites shared by more than "
            f"one species. A three-body channel is a product of three weights, and the mean of "
            f"that product over the species is not the product of their means, which is all this "
            f"module has. Use an ordered structure, or the two-point channels of periodic.py, "
            f"which carry the second moments a shared site needs.")


def _centres_of(crystal) -> Optional[List[int]]:
    """Which atoms are summed over: all of them, or the one site a record stands for (M5)."""
    centre = getattr(crystal, "centre", None)
    return None if centre is None else [int(centre)]


def threebody_autocorrelation(crystal, spec: ThreeBodySpec = ThreeBodySpec(),
                              pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """The three-body channels of one cell, flattened: ``(n_pair_blocks * channels,)``.

    The sum runs over the atoms of the cell as centres and over ordered pairs of their
    neighbours, so a pair of different field types is counted both ways round, as the two-point
    descriptor counts it.  For a site record (plan item M5) only that site is a centre, and the
    channels are the three-body profile of that one site.
    """
    _refuse_disordered(crystal)
    W = np.asarray(crystal.weights, float)
    n_atoms, n_fields = W.shape
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    if any(f >= n_fields or g >= n_fields for f, g in pairs):
        raise ValueError(f"{getattr(crystal, 'name', '') or 'this structure'} carries {n_fields} "
                         f"fields, fewer than the channels asked for")
    n_r, n_d = spec.n_radii, spec.n_distances
    width = sum(len(spec.radial_pairs(f == g)) * n_d for f, g in pairs)
    if n_atoms == 0:
        return np.zeros(width)
    radii, distances, sigma = spec.radii, spec.distances, float(spec.sigma)
    total = np.zeros((n_fields, n_r, n_fields, n_r, n_d))
    wanted = _centres_of(crystal)
    hoods = neighbourhoods(crystal, spec.cutoff)
    taken = 0
    for c, (X, Wc) in enumerate(hoods):
        if wanted is not None and c not in wanted:
            continue
        taken += 1
        r = np.linalg.norm(X, axis=1)
        keep = r <= spec.cutoff
        X, Wc, r = X[keep], Wc[keep], r[keep]
        R = _windows(r, radii, sigma)                                   # (n, n_r)
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        if not spec.include_self:
            np.fill_diagonal(d, np.inf)
        Dm = _windows(d, distances, sigma)                              # (n, n, n_d)
        A = Wc[:, :, None] * R[:, None, :]                              # (n, F, n_r)
        total += np.einsum("afk,abm,bgl->fkglm", A, Dm, A, optimize=True)
    out = []
    iu = np.triu_indices(n_r)
    double = (iu[0] != iu[1]).astype(float) + 1.0
    for f, g in pairs:
        block = total[f, :, g, :, :]
        if f == g:
            out.append((block[iu] * double[:, None]).ravel())
        else:
            out.append((2.0 * block).ravel())
    return np.concatenate(out) / max(taken, 1)


def periodic_chirality(crystal, spec: ThreeBodySpec = ThreeBodySpec(),
                       triples: Optional[Sequence[Tuple[int, int, int]]] = None) -> np.ndarray:
    """The handedness channels of one cell, flattened: ``(n_triples * n_shells,)``.

    Each channel is the signed volume spanned by three field-weighted vectors, summed over the
    atoms of the cell as centres and divided by their number.  A structure and its mirror image
    give the same number with the opposite sign, and a centrosymmetric structure gives zero.
    """
    _refuse_disordered(crystal)
    W = np.asarray(crystal.weights, float)
    n_atoms, n_fields = W.shape
    trips = list(triples) if triples is not None else chirality_triples(n_fields)
    shells = spec.shell_centres
    n_s = max(len(shells), 1)
    if n_atoms == 0 or not trips or not spec.chirality:
        return np.zeros(len(trips) * (len(shells) if spec.chirality else 0))
    out = np.zeros((len(trips), n_s))
    wanted = _centres_of(crystal)
    taken = 0
    for c, (X, Wc) in enumerate(neighbourhoods(crystal, spec.cutoff)):
        if wanted is not None and c not in wanted:
            continue
        taken += 1
        r = np.linalg.norm(X, axis=1)
        keep = r <= spec.cutoff
        X, Wc, r = X[keep], Wc[keep], r[keep]
        if spec.n_shells <= 1:
            G = np.ones((len(r), 1))
        else:
            G = _windows(r, shells, float(spec.chirality_sigma))
        A = np.einsum("af,as,ad->fsd", Wc, G, X, optimize=True)         # (F, n_s, 3)
        for t, (f, g, h) in enumerate(trips):
            out[t] += np.einsum("sd,sd->s", A[f], np.cross(A[g], A[h]))
    return (out / max(taken, 1)).ravel()


def threebody_features(crystal, spec: ThreeBodySpec = ThreeBodySpec(),
                       pairs: Optional[Sequence[Tuple[int, int]]] = None,
                       triples: Optional[Sequence[Tuple[int, int, int]]] = None) -> np.ndarray:
    """The channels of one cell: the three-body block, and the handedness block after it."""
    rows = [threebody_autocorrelation(crystal, spec, pairs)]
    if spec.chirality:
        rows.append(periodic_chirality(crystal, spec, triples))
    return np.concatenate(rows)


def threebody_feature_matrix(crystals: Sequence, spec: ThreeBodySpec = ThreeBodySpec(),
                             n_fields: Optional[int] = None, progress=None,
                             dtype=float) -> np.ndarray:
    """Three-body features of a list of cells, one row each."""
    if not len(crystals):
        return np.zeros((0, 0))
    n_fields = int(n_fields) if n_fields is not None else crystals[0].weights.shape[1]
    pairs = field_pairs(n_fields)
    trips = chirality_triples(n_fields) if spec.chirality else []
    width = spec.n_channels(n_fields)
    out = np.zeros((len(crystals), width), dtype=dtype)
    for i, c in enumerate(crystals):
        if c.weights.shape[1] != n_fields:
            raise ValueError(
                f"{getattr(c, 'name', '') or f'structure {i}'} carries {c.weights.shape[1]} "
                f"fields and the others carry {n_fields}. Build the fields of every structure "
                f"from one field set.")
        row = threebody_features(c, spec, pairs, trips)
        if len(row) != width:
            raise ValueError(f"{getattr(c, 'name', '') or f'structure {i}'} gives {len(row)} "
                             f"channels where the set gives {width}")
        out[i] = row
        if progress is not None and i % 50 == 0:
            progress(f"Three-body channels {i + 1}/{len(crystals)}", i / max(len(crystals), 1))
    return out


def threebody_block_sizes(spec: ThreeBodySpec, n_fields: int) -> List[int]:
    """How many columns each channel block takes, in the order the features are laid out."""
    return spec.block_sizes(int(n_fields))


def threebody_channel_labels(fieldset: FieldSet, spec: ThreeBodySpec) -> List[str]:
    """One label per channel: which two field types, at which two radii and how far apart."""
    names = [sp.name for sp in fieldset.specs]
    radii, distances = spec.radii, spec.distances
    out: List[str] = []
    for f, g in field_pairs(len(names)):
        for k, kk in spec.radial_pairs(f == g):
            for d in distances:
                out.append(f"{names[f]} @ {radii[k]:.2f} x {names[g]} @ {radii[kk]:.2f}, "
                           f"{d:.2f} A apart")
    if spec.chirality:
        for f, g, h in chirality_triples(len(names)):
            for s in spec.shell_centres:
                out.append(f"handedness {names[f]} x {names[g]} x {names[h]} @ {s:.2f} A")
    return out


# --------------------------------------------------------------------------- #
# Reading a fitted model
# --------------------------------------------------------------------------- #
def channel_spread(features: np.ndarray, blocks: Optional[Sequence[int]] = None) -> np.ndarray:
    """The standard deviation of every channel over a set of structures.

    A ridge weight alone says nothing: a channel that hardly varies can carry a large weight and
    move no prediction.  What a reading needs is the weight times the spread, which is the
    convention :func:`cmfqsar.periodic.top_channels` uses.
    """
    X = np.asarray(features, float)
    return X.std(axis=0) if len(X) else np.zeros(X.shape[1] if X.ndim == 2 else 0)


def pattern_table(weights: Sequence[float], features: np.ndarray, labels: Sequence[str],
                  top: int = 20) -> List[Dict[str, float]]:
    """The channels that move the prediction most, largest first.

    Each row carries the label of the channel, the weight the model gives it, the spread of the
    channel over the structures and their product, which is how much of the prediction that
    channel accounts for.
    """
    w = np.asarray(weights, float)
    spread = channel_spread(features)
    if len(w) != len(spread) or len(w) != len(labels):
        raise ValueError(f"{len(w)} weights, {len(spread)} channels and {len(labels)} labels")
    effect = w * spread
    order = np.argsort(-np.abs(effect))[:int(top)]
    return [{"channel": str(labels[k]), "weight": float(w[k]), "spread": float(spread[k]),
             "effect": float(effect[k])} for k in order]
