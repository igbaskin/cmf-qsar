"""The three-distance autocorrelation of an environment: the angle at the centre (plan item C5).

The rotation-invariant local kernel of :mod:`cmfqsar.localenv` describes an atom's surroundings by
the two-point autocorrelation of :mod:`cmfqsar.invariant`: for every pair of field types, how much
field weight sits at each distance from how much other field weight.  On an environment that is a
histogram of the distances between neighbours of the centre, and it leaves something open.  Two
neighbours and the centre form a triangle; the two-point descriptor resolves the side between the
neighbours and not the two sides that meet at the centre, so it does not know the angle at the
centre.  `benchmarks/run_soap_gate.py` measured what that costs: the SOAP descriptor (Smooth
Overlap of Atomic Positions, Bartok, Kondor and Csanyi 2013), which resolves the triangle, lowers
the error of a carbon-shift model by a third, and a SOAP restricted to its two-body part does not.

This module resolves the triangle in the terms of this program.  For an environment whose first
atom is the centre, with ``r_a`` the distance of atom *a* from the centre and ``d_ab`` the distance
between atoms *a* and *b*, the descriptor is ::

    Phi_fg(k, k', m) = sum over atoms a, b of
                       w_af w_bg R_k(r_a) R_k'(r_b) D_m(d_ab),

with ``R_k`` a Gaussian window on a grid of radii from the centre and ``D_m`` one on a grid of
distances.  A triangle is fixed by its three sides, so this carries the same information as the
SOAP power spectrum, which resolves the two radii and the angle between them; the basis differs
and nothing else.  The channels of the two-point descriptor are the sum of these over the two
radii.

**Channels.**  The sum is symmetric under swapping (a, f, k) with (b, g, k'), so one channel is
kept for each unordered pair of (field type, radial window) labels and the two orderings are
added, as the two-point descriptor adds a pair both ways round.  The channels come in blocks, one
per pair of field types, in the order of :func:`cmfqsar.invariant.field_pairs`; a block of one
field type with itself holds the radial pairs ``k <= k'`` and a block of two different types holds
all of them.  Before the cosine-normalised linear kernel is taken the blocks are scaled, by
default per field (:func:`block_scales`): the two-point kernel's scaling per pair of fields gives
a block of two rare channels the weight of a block of two common ones, and with element channels
that costs accuracy.  :func:`triangle_kernel` can raise the kernel to an integer power, which is
what SOAP does with its exponent zeta; a power above one has no finite feature map, so it is for
the dual form only.

**The channels are the fields.**  SOAP has one density per element; this descriptor has one
weight per field, so the channels are the charges, the steric bulk, the hydrophobicity and the
hydrogen-bond strengths of the atoms around the centre, and a channel pair says which two
properties, at which two radii and which distance apart, the model reads.  An element indicator
is itself a field, so SOAP's channels are one setting of this descriptor and the two can be mixed.

**Cost.**  One pass over the atom pairs of the environment, with a window product per pair and
grid point: for forty atoms, nine radii and seventeen distances it is a few milliseconds.  The
number of channels grows as the square of (fields times radii), which is why the radial grid is
coarse and the cutoff short; the cross-validation of :mod:`cmfqsar.localenv` works in the dual
and does not mind.

**What is not resolved.**  Three distances fix a triangle up to its mirror image, so an
environment and its reflection give the same channels.  That is the blind spot plan item C5
measured on whole molecules, and it is unchanged here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .fields import FieldSet, MoleculeFields
from .invariant import autocorrelation_kernel, field_pairs


@dataclass(frozen=True)
class TriangleSpec:
    """The two grids the three-distance autocorrelation is evaluated on.

    `r_max` is how far from the centre the radial grid reaches, in angstrom, and is the cutoff of
    the descriptor; the distance grid reaches twice as far, which is the longest side a triangle
    inside the cutoff can have.  `step` is the spacing of both grids and `sigma` the width of the
    Gaussian window on every grid point.  Atoms further than ``r_max + reach * sigma`` from the
    centre contribute less than ``exp(-reach^2 / 2)`` of a window and are left out of the pair
    sum.
    """
    r_max: float = 4.0
    step: float = 0.5
    sigma: float = 0.35
    include_self: bool = True        # the a = b terms, which carry the radial density itself
    reach: float = 4.0

    @property
    def radii(self) -> np.ndarray:
        n = max(int(round(self.r_max / self.step)) + 1, 1)
        return np.arange(n, dtype=float) * self.step

    @property
    def distances(self) -> np.ndarray:
        n = max(int(round(2.0 * self.r_max / self.step)) + 1, 1)
        return np.arange(n, dtype=float) * self.step

    @property
    def n_radii(self) -> int:
        return len(self.radii)

    @property
    def n_distances(self) -> int:
        return len(self.distances)

    @property
    def cutoff(self) -> float:
        """Beyond this distance from the centre an atom is not in the sum."""
        return float(self.r_max + self.reach * self.sigma)

    def radial_pairs(self, same: bool) -> List[Tuple[int, int]]:
        """The (k, k') radial windows of one block: ``k <= k'`` for a field type with itself."""
        n = self.n_radii
        if same:
            return [(k, kk) for k in range(n) for kk in range(k, n)]
        return [(k, kk) for k in range(n) for kk in range(n)]

    def block_sizes(self, n_fields: int) -> List[int]:
        """Channels per block, one block per pair of field types."""
        return [len(self.radial_pairs(f == g)) * self.n_distances
                for f, g in field_pairs(int(n_fields))]

    def n_channels(self, n_fields: int) -> int:
        return int(sum(self.block_sizes(n_fields)))


def _windows(x: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-((x[..., None] - grid) ** 2) / (2.0 * float(sigma) ** 2))


def triangle_features(env: MoleculeFields, spec: TriangleSpec = TriangleSpec(),
                      pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """The three-distance autocorrelation of one environment, flattened.

    The first atom of `env` is the centre, as :func:`cmfqsar.localenv.environment` builds it;
    radii are measured from it.  The result depends on the arrangement of the atoms about the
    centre and on nothing else: not on where the environment sits or how it is turned.
    """
    W = np.asarray(env.weights, float)
    n_atoms, n_fields = W.shape
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    if any(f >= n_fields or g >= n_fields for f, g in pairs):
        raise ValueError(f"{env.name or 'an environment'} carries {n_fields} fields, fewer than "
                         f"the channels asked for")
    n_r, n_d = spec.n_radii, spec.n_distances
    width = sum(len(spec.radial_pairs(f == g)) * n_d for f, g in pairs)
    if n_atoms == 0:
        return np.zeros(width)
    X = np.asarray(env.coords, float)
    X = X - X[0]
    r = np.linalg.norm(X, axis=1)
    keep = r <= spec.cutoff
    X, W, r = X[keep], W[keep], r[keep]
    R = _windows(r, spec.radii, spec.sigma)                                  # (N, n_r)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    if not spec.include_self:
        np.fill_diagonal(d, np.inf)
    D = _windows(d, spec.distances, spec.sigma)                              # (N, N, n_d)
    A = W[:, :, None] * R[:, None, :]                                        # (N, F, n_r)
    T = np.einsum("afk,abm,bgl->fkglm", A, D, A, optimize=True)              # (F,n_r,F,n_r,n_d)
    out = []
    iu = np.triu_indices(n_r)
    double = (iu[0] != iu[1]).astype(float) + 1.0
    for f, g in pairs:
        block = T[f, :, g, :, :]
        if f == g:
            out.append((block[iu] * double[:, None]).ravel())
        else:
            out.append((2.0 * block).ravel())
    return np.concatenate(out)


def triangle_feature_matrix(envs: Sequence[MoleculeFields], spec: TriangleSpec = TriangleSpec(),
                            n_fields: Optional[int] = None, progress=None) -> np.ndarray:
    """Three-distance features of a list of environments, one row each."""
    if not len(envs):
        return np.zeros((0, 0))
    n_fields = n_fields if n_fields is not None else envs[0].weights.shape[1]
    pairs = field_pairs(n_fields)
    width = spec.n_channels(n_fields)
    out = np.zeros((len(envs), width))
    for i, e in enumerate(envs):
        if e.weights.shape[1] != n_fields:
            raise ValueError(
                f"{e.name or f'environment {i}'} carries {e.weights.shape[1]} fields and the "
                f"others carry {n_fields}. Build the fields of every molecule from one field set.")
        row = triangle_features(e, spec, pairs)
        if len(row) != width:
            raise ValueError(f"{e.name or f'environment {i}'} gives {len(row)} channels where "
                             f"the set gives {width}")
        out[i] = row
        if progress is not None and i % 200 == 0:
            progress(f"Three-distance autocorrelation {i + 1}/{len(envs)}",
                     i / max(len(envs), 1))
    return out


#: how the channel blocks are scaled before the kernel: by field, by pair of fields, or not
SCALINGS = ("field", "pair", "none")


def block_scales(rms: Sequence[float], n_fields: int, blocks: Sequence[int],
                 scaling: str = "field") -> np.ndarray:
    """One scale per channel from the root mean square of every block.

    ``"pair"`` divides every block by its own root mean square, as
    :func:`cmfqsar.invariant.channel_scale` does for the two-point kernel.  ``"field"`` divides
    the block of fields f and g by the geometric mean of the root mean squares of the blocks of f
    with itself and of g with itself, which is the same as giving every field unit scale and
    scaling nothing else.  ``"none"`` leaves the channels as they are.

    The difference matters when the channels differ greatly in how much they carry.  With one
    indicator per element the block of two rare elements holds a few pairs and the block of two
    carbons holds most of the molecule; scaled per pair the rare block counts as much as the
    carbon block and the kernel is mostly noise, which on the aromatic carbons of
    `benchmarks/run_triangles.py` costs 0.4 ppm.  Per field, every element is given the same
    weight as an atom and the pairs keep their proportions; that is the default.  For the
    physico-chemical fields, whose units differ, per-field and per-pair scaling are close and
    no scaling is worse.
    """
    if scaling not in SCALINGS:
        raise ValueError(f"scaling has to be one of {SCALINGS}, not {scaling!r}")
    rms = [float(r) for r in rms]
    blocks = [int(b) for b in blocks]
    pairs = field_pairs(int(n_fields))
    if len(rms) != len(pairs) or len(blocks) != len(pairs):
        raise ValueError(f"{len(rms)} block values and {len(blocks)} block sizes for "
                         f"{len(pairs)} pairs of {n_fields} fields")
    width = int(sum(blocks))
    scale = np.ones(width)
    if scaling == "none":
        return scale
    floor = max(1e-300, (max(rms) if rms else 0.0) * 1e-9)       # as channel_scale floors it
    own = {f: r for (f, g), r in zip(pairs, rms) if f == g}
    starts = np.cumsum([0] + blocks)
    for (f, g), r, s, e in zip(pairs, rms, starts[:-1], starts[1:]):
        d = r if scaling == "pair" else float(np.sqrt(own[f] * own[g]))
        scale[s:e] = 1.0 / d if d > floor else 0.0
    return scale


def _block_rms(X: np.ndarray, blocks: Sequence[int]) -> List[float]:
    starts = np.cumsum([0] + [int(b) for b in blocks])
    return [float(np.sqrt(np.mean(X[:, s:e] ** 2))) for s, e in zip(starts[:-1], starts[1:])]


def triangle_scale(train: np.ndarray, spec: TriangleSpec, n_fields: int,
                   scaling: str = "field") -> np.ndarray:
    """The channel scales a training set gives, to apply to its test rows as well."""
    blocks = spec.block_sizes(n_fields)
    return block_scales(_block_rms(np.asarray(train, float), blocks), n_fields, blocks, scaling)


def triangle_kernel(train: np.ndarray, test: Optional[np.ndarray] = None, *,
                    spec: TriangleSpec = TriangleSpec(), n_fields: int,
                    scale: Optional[np.ndarray] = None, scaling: str = "field",
                    power: int = 1, normalize: bool = True) -> np.ndarray:
    """Cosine-normalised linear kernel on the three-distance features, to an integer power.

    The blocks are scaled by `scaling` (see :func:`block_scales`) unless `scale` is given, and
    the normalisation is of the joint kernel.  `power` above one is the exponent SOAP calls zeta:
    it sharpens the kernel, keeps it positive definite, and leaves no finite feature map, so the
    primal route of :mod:`cmfqsar.locallarge` needs ``power=1``.
    """
    if scale is None:
        scale = triangle_scale(train, spec, n_fields, scaling)
    K = autocorrelation_kernel(train, test, scale=scale, blocks=spec.block_sizes(n_fields),
                               normalize=normalize)
    p = int(power)
    if p < 1:
        raise ValueError(f"the power of the kernel has to be a positive integer, not {power}")
    return K if p == 1 else np.clip(K, -1.0, 1.0) ** p if normalize else K ** p


def triangle_kernel_matrix(envs: Sequence[MoleculeFields], spec: TriangleSpec = TriangleSpec(),
                           n_fields: Optional[int] = None, power: int = 1, chunk: int = 500,
                           scaling: str = "field", progress=None) -> np.ndarray:
    """The kernel of a set of environments among themselves, in two passes over the set.

    The same number as ``triangle_kernel(triangle_feature_matrix(envs))``, to single precision,
    without ever holding the features in double precision: the first pass collects the block
    scales, the second builds the scaled and normalised rows in single precision, chunk by chunk,
    and the kernel is their product.  With many fields and a long cutoff the feature matrix is
    the largest thing in the computation, and this halves it.
    """
    n = len(envs)
    if n == 0:
        return np.zeros((0, 0))
    n_fields = n_fields if n_fields is not None else envs[0].weights.shape[1]
    width = spec.n_channels(n_fields)
    blocks = spec.block_sizes(n_fields)
    starts = np.cumsum([0] + blocks)
    chunk = max(int(chunk), 1)
    if scaling not in SCALINGS:
        raise ValueError(f"scaling has to be one of {SCALINGS}, not {scaling!r}")

    def rows(lo: int, hi: int, what: str) -> np.ndarray:
        if progress is not None:
            progress(f"{what} {lo + 1}/{n}", lo / n)
        return triangle_feature_matrix(envs[lo:hi], spec, n_fields)

    ss = np.zeros(width)
    for lo in range(0, n, chunk):
        X = rows(lo, min(lo + chunk, n), "Three-distance scales")
        ss += np.sum(X * X, axis=0)
    rms = [float(np.sqrt(ss[s:e].sum() / (n * (e - s)))) for s, e in zip(starts[:-1], starts[1:])]
    scale = block_scales(rms, n_fields, blocks, scaling)
    Z = np.zeros((n, width), np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        X = rows(lo, hi, "Three-distance rows") * scale
        d = np.sqrt(np.maximum(np.sum(X * X, axis=1), 1e-300))
        Z[lo:hi] = (X / d[:, None]).astype(np.float32)
    K = np.clip((Z @ Z.T).astype(float), -1.0, 1.0)
    p = int(power)
    if p < 1:
        raise ValueError(f"the power of the kernel has to be a positive integer, not {power}")
    return K if p == 1 else K ** p


def triangle_labels(fieldset: FieldSet, spec: TriangleSpec = TriangleSpec()) -> List[str]:
    """One label per channel: the two field types, their two radii and the distance between."""
    names = [s.label for s in fieldset.specs]
    radii, dists = spec.radii, spec.distances
    out = []
    for f, g in field_pairs(len(names)):
        for k, kk in spec.radial_pairs(f == g):
            for m in range(len(dists)):
                out.append(f"{names[f]} at {radii[k]:.2f} A x {names[g]} at {radii[kk]:.2f} A, "
                           f"{dists[m]:.2f} A apart")
    return out


def with_weights(env: MoleculeFields, weights: np.ndarray, name: str = "") -> MoleculeFields:
    """The same environment with other field weights, for channels that are not the fields.

    The weights of an environment already carry the envelope factor of each atom; weights given
    here are taken as they are, so a caller that wants the envelope multiplies it in
    (:func:`envelope_factors`).
    """
    W = np.asarray(weights, float)
    if W.shape[0] != env.n_atoms:
        raise ValueError(f"{W.shape[0]} rows of weights for {env.n_atoms} atoms")
    return MoleculeFields(name=name or env.name, coords=env.coords, weights=W, radii=env.radii,
                          symbols=list(env.symbols), atom_types=list(env.atom_types),
                          bonds=env.bonds, charges=env.charges,
                          member_atom_map=env.member_atom_map, charge_source=env.charge_source)


def envelope_factors(env: MoleculeFields, rho: float) -> np.ndarray:
    """The envelope factor ``exp(-rho r^2)`` of every atom of an environment, from its radius."""
    X = np.asarray(env.coords, float)
    r2 = np.sum((X - X[0]) ** 2, axis=1)
    return np.exp(-float(rho) * r2)


def element_weights(env: MoleculeFields, elements: Sequence[str], rho: Optional[float] = None,
                    other: bool = True) -> np.ndarray:
    """One indicator weight per element, which makes the descriptor SOAP's channels.

    `elements` are the symbols that get a channel of their own; with `other`, every other symbol
    shares one more.  With `rho` the envelope factor is multiplied in, as the field weights of an
    environment carry it.
    """
    cols = list(elements) + (["other"] if other else [])
    W = np.zeros((env.n_atoms, len(cols)))
    for a, s in enumerate(env.symbols):
        j = cols.index(s) if s in elements else (len(cols) - 1 if other else -1)
        if j >= 0:
            W[a, j] = 1.0
    if rho is not None:
        W = W * envelope_factors(env, rho)[:, None]
    return W
