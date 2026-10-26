"""Alignment-free but geometry-aware kernels: autocorrelation of the molecular fields.

Everything else in this program compares two molecules by the overlap integral of their fields,
which needs them to share a frame.  That is the strength of the approach and its one hard
requirement, and the heterogeneous-series work shows what it costs when the frame is in doubt:
the alignment-free comparator there is the count kernel, which is pose-free only because it has
thrown the geometry away entirely.

This module fills the gap between those two.  For every pair of field types it takes the distance
distribution of the field weights inside one molecule,

    Phi_fg(d) = sum_l sum_m w_lf w_mg exp(-(|r_l - r_m| - d)^2 / 2 sigma^2),

evaluated on a grid of distances d_k.  A rigid motion moves every atom and leaves every
interatomic distance where it was, so the whole descriptor is invariant to rotation and
translation by construction -- no superposition, no reference molecule, no common frame -- while
still describing the three-dimensional arrangement: where a donor sits relative to an acceptor,
how far the positive charge is from the hydrophobic bulk.

The connection to the CMF kernel is exact at one point.  The overlap integral of two Gaussian
fields of the same molecule is

    K_fg(A, A) = (pi/alpha)^{3/2} sum_lm w_lf w_mg exp(-alpha |r_l - r_m|^2 / 4),

which is this descriptor's channel at d = 0 with sigma^2 = 2/alpha, up to the prefactor.  The
further channels resolve what that single number collapses: the overlap integral reports how much
the fields of a molecule coincide, the autocorrelation reports at which distances they do.

A distance distribution is the same for a molecule and its mirror image, so the two-point form
above cannot detect chirality.  The signed quantity that is not is the triple product of three
displacement vectors, and :func:`chirality` adds it as a second block of channels.  Summing the
signed volume over every triple of atoms would cost O(N^3), but the determinant is trilinear, so
the sum factorises into the triple product of three field-weighted first moments and costs one
pass over the atoms instead.  It is off by default, so a model built before it existed is
unchanged; a test asserts that.

What is lost with the frame is the co-field: a map in space needs a space to live in.  The
decomposition that survives is the one in the descriptor's own terms -- which pair of field
types at which distance carries the weight -- and :func:`channel_labels` names those channels so
a fitted model can be read that way.

The channels come in blocks, one per pair of field types, and every block is scaled to unit root
mean square before the blocks are taken together in one cosine-normalised dot product.  That is
the right default for a few fields and the wrong one for sixty: with one indicator field
per element the pairs across the elements outnumber the physico-chemical pairs a hundred to one
and, equally scaled, overwhelm them.  The overlap kernel has its mixing coefficients h_f for this;
the functions under "Mixing coefficients over the channel blocks" below give the invariant
kernels theirs -- one weight per block, set by the kind of block (within a group of fields, across
two groups) or by centred kernel alignment of the per-block kernels, as :func:`model.align_mixing`
sets the h_f.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .fields import FieldSet, MoleculeFields


@dataclass(frozen=True)
class AutocorrelationSpec:
    """The distance grid the field autocorrelation is evaluated on."""
    d_max: float = 12.0
    step: float = 0.75
    sigma: float = 0.75              # width of the smoothing along the distance axis
    include_self: bool = True        # the l = m terms, which carry the plain count of each type
    chirality: bool = False          # append the three-point signed channels; off by default,
    #                                  so a model built before they existed is unchanged
    chirality_shells: int = 4        # radial shells the signed volume is resolved over; 1 is
    #                                  the single global invariant per field triple

    @property
    def centres(self) -> np.ndarray:
        n = max(int(round(self.d_max / self.step)) + 1, 1)
        return np.arange(n, dtype=float) * self.step

    @property
    def n_bins(self) -> int:
        return len(self.centres)

    @property
    def shell_centres(self) -> np.ndarray:
        """Where the radial windows of the chirality channels sit, measured from the centroid.

        Radii from the centroid reach about half the longest interatomic distance, so the shells
        span `d_max / 2` rather than `d_max`; one shell means no window at all.
        """
        s = max(int(self.chirality_shells), 1)
        if s == 1:
            return np.zeros(1)
        return np.linspace(0.0, 0.5 * self.d_max, s)

    @property
    def shell_sigma(self) -> float:
        s = self.shell_centres
        return float(s[1] - s[0]) if len(s) > 1 else 0.0

    @property
    def n_shells(self) -> int:
        return len(self.shell_centres)


@dataclass(frozen=True)
class SurfaceSpec:
    """The in-plane distance grid and the height grid of the surface autocorrelation.

    A molecule on a surface has one direction that is not like the others: the normal.  Heights
    above the surface are physical and stay as they are; what is arbitrary is where the molecule
    sits in the plane and how it is turned about the normal.  The descriptor is the two-point
    distribution of the field weights over the *in-plane* distance rho, resolved by the heights
    of both atoms,

        Phi_fg(rho_k, z_i, z_j) = sum_l sum_m w_lf w_mg exp(-(rho_lm - rho_k)^2 / 2 sigma^2)
                                              exp(-(z_l - z_i)^2 / 2 sigma_z^2)
                                              exp(-(z_m - z_j)^2 / 2 sigma_z^2),

    which is invariant to translations in the plane and rotations about the normal, and to
    nothing else.  Averaged over the direction of rho it is, by the Wiener-Khinchin theorem,
    the same information as the in-plane power spectrum |F_f(k, z_i) F_g(k, z_j)| of the fields:
    a description of the molecule over the surface in reciprocal space in the plane and in
    real space along the normal.  The surface is at z = 0 and the normal is +z, which is how
    examples/uma_fe110_adsorption.py writes its poses.
    """
    d_max: float = 12.0
    step: float = 0.75
    sigma: float = 0.75              # width along the in-plane distance axis
    z_min: float = 1.0
    z_max: float = 6.0
    z_step: float = 1.0
    z_sigma: float = 1.0             # width along the height axis
    include_self: bool = True

    @property
    def centres(self) -> np.ndarray:
        n = max(int(round(self.d_max / self.step)) + 1, 1)
        return np.arange(n, dtype=float) * self.step

    @property
    def heights(self) -> np.ndarray:
        n = max(int(round((self.z_max - self.z_min) / self.z_step)) + 1, 1)
        return self.z_min + np.arange(n, dtype=float) * self.z_step

    @property
    def n_bins(self) -> int:
        """Channels per pair of field types: distances times heights squared."""
        return len(self.centres) * len(self.heights) ** 2


def surface_autocorrelation(mol: MoleculeFields, spec: SurfaceSpec = SurfaceSpec(),
                            pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """Surface autocorrelation of one molecule, flattened: (n_pairs * n_rho * n_z * n_z,).

    For a pair of different field types both orders are summed (f at z_i with g at z_j, and
    g at z_i with f at z_j), as the three-dimensional descriptor counts a pair both ways round.
    """
    W = np.asarray(mol.weights, float)
    n_atoms, n_fields = W.shape
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    centres, heights = spec.centres, spec.heights
    if n_atoms == 0:
        return np.zeros(len(pairs) * spec.n_bins)
    xy = np.asarray(mol.coords, float)[:, :2]
    z = np.asarray(mol.coords, float)[:, 2]
    rho = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    if not spec.include_self:
        np.fill_diagonal(rho, np.inf)
    B = np.exp(-((rho[:, :, None] - centres[None, None, :]) ** 2) / (2.0 * spec.sigma ** 2))
    Z = np.exp(-((z[:, None] - heights[None, :]) ** 2) / (2.0 * spec.z_sigma ** 2))
    out = np.empty((len(pairs), len(centres), len(heights), len(heights)))
    for i, (f, g) in enumerate(pairs):
        t = np.einsum("a,b,abk,ai,bj->kij", W[:, f], W[:, g], B, Z, Z, optimize=True)
        if f != g:
            t = t + np.einsum("a,b,abk,ai,bj->kij", W[:, g], W[:, f], B, Z, Z, optimize=True)
        out[i] = t
    return out.ravel()


def surface_block_sizes(spec: SurfaceSpec, n_fields: int) -> List[int]:
    return [spec.n_bins] * len(field_pairs(n_fields))


def surface_channel_labels(fieldset: FieldSet, spec: SurfaceSpec) -> List[str]:
    """One label per feature: the field types, the in-plane distance and the two heights."""
    names = [s.label for s in fieldset.specs]
    out = []
    for f, g in field_pairs(len(names)):
        pair = names[f] if f == g else f"{names[f]} x {names[g]}"
        for d in spec.centres:
            for zi in spec.heights:
                for zj in spec.heights:
                    out.append(f"{pair} @ rho {d:.2f} A, z {zi:.1f} / {zj:.1f} A")
    return out


def surface_feature_matrix(mols: Sequence[MoleculeFields], spec: SurfaceSpec = SurfaceSpec(),
                           n_fields: Optional[int] = None, progress=None) -> np.ndarray:
    """Surface autocorrelation features of a set of molecules, one row each."""
    if not len(mols):
        return np.zeros((0, 0))
    n_fields = n_fields if n_fields is not None else mols[0].weights.shape[1]
    pairs = field_pairs(n_fields)
    width = sum(surface_block_sizes(spec, n_fields))
    out = np.zeros((len(mols), width))
    for i, m in enumerate(mols):
        if m.weights.shape[1] != n_fields:
            raise ValueError(
                f"{m.name or f'molecule {i}'} carries {m.weights.shape[1]} fields where the set "
                f"carries {n_fields}: build the fields of every molecule from one field set.")
        row = surface_autocorrelation(m, spec, pairs)
        out[i] = row
        if progress is not None and i % 25 == 0:
            progress(f"Surface autocorrelation {i + 1}/{len(mols)}", i / max(len(mols), 1))
    return out


def field_pairs(n_fields: int) -> List[Tuple[int, int]]:
    """The unordered pairs of field types, diagonal first: (0,0), (1,1), ..., then (0,1), ..."""
    return ([(f, f) for f in range(n_fields)]
            + [(f, g) for f in range(n_fields) for g in range(f + 1, n_fields)])


def chirality_triples(n_fields: int) -> List[Tuple[int, int, int]]:
    """The ordered triples f < g < h of distinct field types.

    The signed volume is antisymmetric in its three arguments, so it vanishes whenever two of
    them are the same field and merely changes sign when two are swapped: the triples with
    f < g < h are all the independent ones there are.
    """
    return [(f, g, h) for f in range(n_fields)
            for g in range(f + 1, n_fields) for h in range(g + 1, n_fields)]


def block_sizes(spec: AutocorrelationSpec, n_fields: int) -> List[int]:
    """How many columns each channel block occupies, in the order the features are laid out.

    The pair blocks all have `n_bins` columns and the chirality blocks `n_shells`, so the blocks
    are no longer of one size once chirality is on and the scaling has to be told where they are.
    """
    out = [spec.n_bins] * len(field_pairs(n_fields))
    if spec.chirality:
        out += [spec.n_shells] * len(chirality_triples(n_fields))
    return out


def channel_labels(fieldset: FieldSet, spec: AutocorrelationSpec) -> List[str]:
    """One label per feature: which field types, and at which distance or radius."""
    names = [s.label for s in fieldset.specs]
    out = []
    for f, g in field_pairs(len(names)):
        pair = names[f] if f == g else f"{names[f]} x {names[g]}"
        for d in spec.centres:
            out.append(f"{pair} @ {d:.2f} A")
    if spec.chirality:
        out += chirality_labels(fieldset, spec)
    return out


def chirality_labels(fieldset: FieldSet, spec: AutocorrelationSpec) -> List[str]:
    """One label per chirality feature: which three field types, in which radial shell."""
    names = [s.label for s in fieldset.specs]
    out = []
    for f, g, h in chirality_triples(len(names)):
        trio = f"{names[f]} x {names[g]} x {names[h]}"
        for r in spec.shell_centres:
            out.append(f"{trio} (signed)" if spec.n_shells == 1
                       else f"{trio} (signed) at r {r:.2f} A")
    return out


def chirality(mol: MoleculeFields, spec: AutocorrelationSpec = AutocorrelationSpec(),
              triples: Optional[Sequence[Tuple[int, int, int]]] = None) -> np.ndarray:
    """The three-point signed channels of one molecule, flattened: (n_triples * n_shells,).

    A distance distribution is the same for a molecule and its mirror image, so everything above
    cannot detect chirality.  The signed quantity that is not is the triple product of three
    displacement vectors, and the obvious way to use it -- summing the signed volume over every
    triple of atoms -- would cost O(N^3) per field triple.  It does not have to.  The determinant
    is trilinear, so the sum over triples factorises exactly:

        sum_{l,m,n} w_lf w_mg w_nh  det[r_l - c, r_m - c, r_n - c]
            =  det[ A_f, A_g, A_h ],        A_f = sum_l w_lf (r_l - c)

    with `c` the centroid of the atoms.  So the whole block costs one pass over the atoms, and
    each channel is the signed volume spanned by three field-weighted vectors -- the first moment
    of each field about the centroid.  Under a proper rotation all three are rotated and the
    determinant is unchanged; under a reflection the determinant changes sign; a translation
    moves the centroid with the atoms and cancels.  That is exactly the behaviour chirality
    needs, and it is why a planar molecule gives zero in every channel: three vectors in a plane
    span no volume, and a planar molecule is achiral.

    `chirality_shells` resolves the invariant over radial windows about the centroid, since one
    number per field triple describes the handedness of the molecule as a whole and cannot say
    at which radius it is located.  The windows depend on `|r_l - c|`, which is itself reflection-
    invariant, so every channel keeps the sign behaviour above.
    """
    W = np.asarray(mol.weights, float)
    n_atoms, n_fields = W.shape
    trips = list(triples) if triples is not None else chirality_triples(n_fields)
    shells = spec.shell_centres
    if n_atoms == 0 or not trips:
        return np.zeros(len(trips) * len(shells))
    X = np.asarray(mol.coords, float)
    X = X - X.mean(axis=0)
    if spec.n_shells <= 1:
        G = np.ones((n_atoms, 1))
    else:
        r = np.linalg.norm(X, axis=1)
        G = np.exp(-((r[:, None] - shells[None, :]) ** 2) / (2.0 * spec.shell_sigma ** 2))
    # A[f, s, :] = sum_l W[l, f] G[l, s] (r_l - c)
    A = np.einsum("lf,ls,ld->fsd", W, G, X, optimize=True)
    out = np.empty((len(trips), len(shells)))
    for i, (f, g, h) in enumerate(trips):
        out[i] = np.einsum("sd,sd->s", A[f], np.cross(A[g], A[h]))
    return out.ravel()


def autocorrelation(mol: MoleculeFields, spec: AutocorrelationSpec = AutocorrelationSpec(),
                    pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """Field autocorrelation of one molecule, flattened: (n_pairs * n_bins,).

    The result depends on the molecule's conformation and on nothing else: not on where it sits,
    not on how it is turned, not on any other molecule.
    """
    W = np.asarray(mol.weights, float)                       # (n_atoms, n_fields)
    n_atoms, n_fields = W.shape
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    centres = spec.centres
    if n_atoms == 0:
        return np.zeros(len(pairs) * len(centres))
    d = np.linalg.norm(mol.coords[:, None, :] - mol.coords[None, :, :], axis=-1)
    if not spec.include_self:
        np.fill_diagonal(d, np.inf)
    # (n_atoms, n_atoms, n_bins): how much each atom pair contributes to each distance channel
    B = np.exp(-((d[:, :, None] - centres[None, None, :]) ** 2) / (2.0 * spec.sigma ** 2))
    out = np.empty((len(pairs), len(centres)))
    for i, (f, g) in enumerate(pairs):
        out[i] = np.einsum("a,abk,b->k", W[:, f], B, W[:, g], optimize=True)
        if f != g:
            out[i] *= 2.0                                    # the pair counts both ways round
    flat = out.ravel()
    if spec.chirality:
        flat = np.concatenate([flat, chirality(mol, spec)])
    return flat


def feature_matrix(mols: Sequence[MoleculeFields],
                   spec: AutocorrelationSpec = AutocorrelationSpec(),
                   n_fields: Optional[int] = None,
                   progress=None) -> np.ndarray:
    """Autocorrelation features of a set of molecules, one row each."""
    if not len(mols):
        return np.zeros((0, 0))
    n_fields = n_fields if n_fields is not None else mols[0].weights.shape[1]
    pairs = field_pairs(n_fields)
    width = sum(block_sizes(spec, n_fields))
    out = np.zeros((len(mols), width))
    for i, m in enumerate(mols):
        row = autocorrelation(m, spec, pairs)
        if len(row) != width:
            # padding or truncating here would quietly put one molecule's channels in another's
            # columns, which is worse than refusing: every molecule must carry the same fields
            raise ValueError(
                f"{m.name or f'molecule {i}'} gives {len(row)} channels where the set gives "
                f"{width}: it carries {m.weights.shape[1]} fields and the others carry "
                f"{n_fields}. Build the fields of every molecule from one field set.")
        out[i] = row
        if progress is not None and i % 25 == 0:
            progress(f"Field autocorrelation {i + 1}/{len(mols)}", i / max(len(mols), 1))
    return out


def channel_scale(train: np.ndarray, n_bins: int,
                  blocks: Optional[Sequence[int]] = None) -> np.ndarray:
    """One scale per channel block, so that no block dominates by its units alone.

    The electrostatic field carries charges and the steric one carries ones, so their
    autocorrelations differ by orders of magnitude; each block is divided by its root mean square
    over the training molecules, which is what `scale_fields` does for the kernels.  The signed
    chirality channels need this more than anything else does, since a volume carries the cube of
    a length.  `blocks` gives the column counts explicitly, which is required once they are not
    all `n_bins` wide; without it the blocks are taken to be uniform, as they were before the
    chirality channels existed.
    """
    X = np.asarray(train, float)
    sizes = ([int(n_bins)] * (X.shape[1] // max(int(n_bins), 1)) if blocks is None
             else [int(b) for b in blocks])
    slices, rms = [], []
    start = 0
    for width in sizes:
        if width <= 0 or start >= X.shape[1]:
            continue
        sl = slice(start, min(start + width, X.shape[1]))
        slices.append(sl)
        rms.append(float(np.sqrt(np.mean(X[:, sl] ** 2))))
        start = sl.stop
    # A block that is zero for every training molecule gets no scale.  "Zero" has to allow for
    # round-off: a channel that cancels exactly in theory -- the charge-weighted count of a
    # neighbour shell in a crystal whose atoms all see the same shells -- comes out at 1e-14, and
    # dividing by its root mean square would turn that noise into the largest feature there is.
    floor = max(1e-300, (max(rms) if rms else 0.0) * 1e-9)
    s = np.ones(X.shape[1])
    for sl, r in zip(slices, rms):
        s[sl] = 1.0 / r if r > floor else 0.0
    return s


def autocorrelation_kernel(train: np.ndarray, test: Optional[np.ndarray] = None,
                           scale: Optional[np.ndarray] = None, n_bins: int = 1,
                           normalize: bool = True,
                           blocks: Optional[Sequence[int]] = None) -> np.ndarray:
    """Cosine-normalised linear kernel on the autocorrelation features.

    The normalisation is applied to the joint kernel, not block by block: normalising each pair
    of field types on its own would divide out the very magnitudes that distinguish molecules,
    the same source of error that affects the alignment-free count kernel when it is normalised
    field by field.
    """
    A = np.asarray(train, float)
    B = A if test is None else np.asarray(test, float)
    if scale is None:
        scale = channel_scale(A, n_bins, blocks=blocks)
    A, B = A * scale, B * scale
    K = B @ A.T
    if not normalize:
        return K
    da = np.sqrt(np.maximum(np.sum(A * A, axis=1), 1e-300))
    db = np.sqrt(np.maximum(np.sum(B * B, axis=1), 1e-300))
    return K / np.outer(db, da)


# --------------------------------------------------------------------------- #
# Mixing coefficients over the channel blocks
# --------------------------------------------------------------------------- #
PHYSCHEM_GROUP = "physchem"
INDICATOR_GROUP = "indicator"
BALANCED = "balanced"
ALIGN_MODES = ("align", "alignf")


def block_fields(n_fields: int, chirality: bool = False) -> List[Tuple[int, ...]]:
    """The field types of every channel block, in the order the features are laid out: the pairs
    of :func:`field_pairs`, then, with chirality, the triples of :func:`chirality_triples`.  The
    surface and periodic descriptors have the pair blocks only."""
    out: List[Tuple[int, ...]] = [tuple(p) for p in field_pairs(n_fields)]
    if chirality:
        out += [tuple(t) for t in chirality_triples(n_fields)]
    return out


def default_block_groups(names: Iterable[str]) -> Dict[str, str]:
    """The grouping of the fields the block weights use unless told otherwise: the continuous
    indicator fields (``cif:<type>``, one per element or atom type) in one group, every other
    field -- the physico-chemical and property fields -- in the other."""
    return {n: (INDICATOR_GROUP if n.startswith("cif:") else PHYSCHEM_GROUP) for n in names}


def group_key(groups: Iterable[str]) -> str:
    """The name of a kind of block from the groups of its fields: the group's name for a block
    within one group, the names joined by ``*`` in alphabetical order for a block across groups,
    so that ``"a*b"`` and ``"b*a"`` are the same key."""
    return "*".join(sorted({str(g) for g in groups}))


def block_group_keys(names: Sequence[str], groups: Optional[Mapping[str, str]] = None,
                     chirality: bool = False) -> List[str]:
    """The kind of every block of a field set with these field names, in layout order.

    ``groups`` maps field names to group names; a field it does not name falls back to the
    default grouping of :func:`default_block_groups`.
    """
    names = list(names)
    g = default_block_groups(names)
    if groups:
        g.update({k: str(v) for k, v in groups.items() if k in g})
    return [group_key(g[names[f]] for f in blk) for blk in block_fields(len(names), chirality)]


def group_block_weights(keys: Sequence[str],
                        weights: Union[str, Mapping[str, float]]) -> np.ndarray:
    """One weight per block, by the kind of block.

    ``weights`` is either a mapping from kinds (keys as :func:`group_key` writes them; either
    order of the groups of a cross key is accepted) to weights, with 1 for every kind it does not
    name, or the word ``"balanced"``: every kind present gets ``1 / sqrt(its number of blocks)``,
    so that the kinds contribute equally to the squared norm of a molecule's channel vector --
    the pairs within the four physico-chemical fields as much as the fifteen hundred pairs across
    the element indicators -- however many blocks each kind has.  The weights multiply the scaled
    features, so a block's share of the kernel goes as the square of its weight, and a weight of
    zero removes the block exactly: it enters neither the dot products nor the norms.
    """
    keys = [str(k) for k in keys]
    if isinstance(weights, str):
        if weights != BALANCED:
            raise ValueError(f"unknown block weighting {weights!r}: give a mapping from kinds of "
                             f"block to weights, '{BALANCED}', or one of {ALIGN_MODES}")
        counts: Dict[str, int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        return np.array([1.0 / np.sqrt(counts[k]) for k in keys])
    table = {group_key(str(k).split("*")): float(v) for k, v in dict(weights).items()}
    w = np.array([table.get(k, 1.0) for k in keys], float)
    if (w < 0).any():
        raise ValueError("block weights cannot be negative")
    return w


def expand_block_weights(w: Sequence[float], blocks: Sequence[int]) -> np.ndarray:
    """The per-block weights spread over the columns of their blocks: one multiplier per feature,
    to be applied with (or folded into) the channel scale."""
    w = np.asarray(w, float)
    sizes = np.asarray([int(b) for b in blocks], int)
    if len(w) != len(sizes):
        raise ValueError(f"{len(w)} block weights for {len(sizes)} blocks")
    return np.repeat(w, sizes)


def _block_starts(blocks: Sequence[int], n_cols: int) -> np.ndarray:
    starts = np.concatenate([[0], np.cumsum([int(b) for b in blocks])]).astype(int)
    if starts[-1] != n_cols:
        raise ValueError(f"the blocks cover {starts[-1]} columns, the features have {n_cols}")
    return starts


def _block_gram_frobenius(Z: np.ndarray, starts: np.ndarray, max_chunk_cols: int) -> np.ndarray:
    """``M[a, b] = <K_a, K_b>_F`` for the linear kernels ``K_b = Z_b Z_b^T`` of the column
    blocks of ``Z``, which is ``||Z_a^T Z_b||_F^2``: computed from the Gram matrix of the columns
    in chunks of rows, so that the (D, D) Gram matrix never exists in full."""
    B = len(starts) - 1
    widths = np.diff(starts)
    M = np.empty((B, B))
    i = 0
    while i < B:
        j, ncol = i, 0
        while j < B and (j == i or ncol + widths[j] <= max_chunk_cols):
            ncol += widths[j]
            j += 1
        P = Z[:, starts[i]:starts[j]].T @ Z                 # (ncol, D)
        P *= P
        S = np.add.reduceat(P, starts[:-1], axis=1)         # (ncol, B): summed over block b
        M[i:j] = np.add.reduceat(S, starts[i:j] - starts[i], axis=0)
        i = j
    return M


def align_block_weights(X: np.ndarray, y: np.ndarray, blocks: Sequence[int], mode: str = "align",
                        rho: float = 1.0, max_chunk_cols: int = 2048) -> np.ndarray:
    """Block weights by centred kernel alignment: :func:`model.align_mixing` for the channel
    blocks of an invariant kernel.

    ``X`` holds the scaled channel features of the training molecules (after
    :func:`channel_scale`), ``blocks`` the column count of every block.  Every block carries a
    linear kernel ``K_b = X_b X_b^T``; its centred form is ``K_b^c = X_b^c X_b^c^T`` with the
    columns centred over the molecules, so everything the alignment needs is in the feature
    space and the kernels themselves are never formed: ``<K_b^c, y^c y^c^T>_F = ||X_b^c^T y^c||^2``
    and ``||K_b^c||_F = ||X_b^c^T X_b^c||_F``.

      ``mode="align"``  weights every block by its own alignment,
        ``h_b ~ <K_b^c, y^c y^c^T> / ||K_b^c||_F`` (Cristianini et al. 2001);
      ``mode="alignf"`` maximises the alignment of the combination,
        ``min_{h >= 0} || sum_b h_b K_b^c - y^c y^c^T ||_F`` (Cortes et al. 2012), with the
        Frobenius Gram matrix of the block kernels formed block pair by block pair and the
        solution shrunk towards equal weights with strength ``rho``, as for the fields.

    The kernel weights ``h_b`` are rescaled so that the mean diagonal of the joint kernel is
    what it was with every ``h_b = 1``, and the returned weights are their square roots, since
    they multiply the features and the kernel sees them squared.  A block that is constant over
    the training molecules -- a zero block, or one that does not vary -- gets no weight.
    """
    X = np.asarray(X, float)
    n, D = X.shape
    starts = _block_starts(blocks, D)
    B = len(starts) - 1
    yc = np.asarray(y, float)
    if len(yc) != n:
        raise ValueError(f"{len(yc)} activities for {n} molecules")
    yc = yc - yc.mean()
    Xc = X - X.mean(axis=0)
    proj = Xc.T @ yc
    v = np.empty(B)                                  # <K_b^c, y^c y^c^T>
    nrm = np.empty(B)                                # ||K_b^c||_F
    md = np.empty(B)                                 # mean diagonal of K_b
    for b in range(B):
        sl = slice(starts[b], starts[b + 1])
        v[b] = float(proj[sl] @ proj[sl])
        G = Xc[:, sl].T @ Xc[:, sl]
        nrm[b] = float(np.sqrt(np.sum(G * G)))
        md[b] = float(np.mean(np.sum(X[:, sl] ** 2, axis=1)))
    floor = max(1e-300, float(nrm.max()) * 1e-12) if B else 1e-300
    live = np.flatnonzero(nrm > floor)
    h = np.zeros(B)
    if live.size:
        if mode == "align":
            h[live] = np.maximum(v[live], 0.0) / nrm[live]
        elif mode == "alignf":
            from scipy.linalg import cholesky, solve_triangular
            from scipy.optimize import nnls
            cols = np.concatenate([np.arange(starts[b], starts[b + 1]) for b in live])
            lstarts = np.concatenate([[0], np.cumsum([starts[b + 1] - starts[b] for b in live])])
            M = _block_gram_frobenius(Xc[:, cols], lstarts.astype(int), max_chunk_cols)
            vv = v[live].copy()
            sc = float(np.mean(nrm[live]))
            if rho > 0:
                denom = float(M.sum())
                h0 = float(vv.sum()) / denom if denom > 1e-300 else 1.0
                M = M + rho * sc * sc * np.eye(live.size)
                vv = vv + rho * sc * sc * h0
            else:
                M = M + 1e-10 * sc * sc * np.eye(live.size)
            # min_h>=0 ||A h - b||^2 with A^T A = M and A^T b = vv is min ||R h - R^-T vv||^2
            # for the Cholesky factor R of M, which is all the NNLS solver needs
            R = cholesky(M, lower=False)
            rhs = solve_triangular(R, vv, trans="T", lower=False)
            h[live], _ = nnls(R, rhs)
        else:
            raise ValueError(f"unknown alignment mode {mode!r}; use one of {ALIGN_MODES}")
    if not np.isfinite(h).all() or h.sum() <= 1e-300:
        h = np.zeros(B)
        h[live] = 1.0                                # degenerate: fall back to equal weights
    ref, cur = float(md[live].sum()), float((h * md).sum())
    if cur > 1e-300 and ref > 1e-300:
        h = h * (ref / cur)
    return np.sqrt(h)


def block_weight_summary(keys: Sequence[str], w: Sequence[float]) -> List[Dict[str, float]]:
    """The weights by kind of block: how many blocks of each kind there are and the mean,
    smallest and largest weight among them, plus the kind's share of the squared norm of a
    channel vector whose blocks all have unit root mean square (``sum w^2`` over the kind,
    normalised over the kinds)."""
    keys = [str(k) for k in keys]
    w = np.asarray(w, float)
    kinds = sorted(set(keys), key=lambda k: (k.count("*"), k))
    total = float(np.sum(w ** 2)) or 1.0
    out = []
    for k in kinds:
        ws = w[[i for i, kk in enumerate(keys) if kk == k]]
        out.append({"kind": k, "n_blocks": int(len(ws)), "mean": float(ws.mean()),
                    "min": float(ws.min()), "max": float(ws.max()),
                    "share": float(np.sum(ws ** 2) / total)})
    return out


# --------------------------------------------------------------------------- #
# What the model says about one molecule, atom by atom (plan item SC6)
# --------------------------------------------------------------------------- #


def channel_coefficients(train: np.ndarray, scale: np.ndarray, dual: Sequence[float],
                         normalize: bool = True) -> np.ndarray:
    """The model as one vector in channel space.

    A kernel ridge or support-vector model on this kernel predicts

        y(x) = sum_j a_j K(x, x_j) + b,

    and the kernel is a dot product of the scaled, length-normalised channel rows, so the sum
    over the training molecules collapses into one vector::

        c = sum_j a_j (s * F_j) / |s * F_j|      and      y(x) - b = u(x) . c

    with ``u(x)`` the scaled row of the molecule being predicted, divided by its own length.
    Everything the model has to say about a new molecule is that one dot product, which is what
    makes the decomposition below exact rather than an attribution method.
    """
    A = np.asarray(train, float) * np.asarray(scale, float)
    if normalize:
        norms = np.sqrt(np.maximum(np.sum(A * A, axis=1), 1e-300))
        A = A / norms[:, None]
    return A.T @ np.asarray(dual, float)


def atom_contributions(mol: MoleculeFields, coefficients: Sequence[float],
                       scale: Optional[Sequence[float]] = None,
                       spec: AutocorrelationSpec = AutocorrelationSpec(),
                       pairs: Optional[Sequence[Tuple[int, int]]] = None,
                       normalize: bool = True,
                       per_pair: bool = False) -> np.ndarray:
    """Split what the invariant model predicts for one molecule over its atoms (item SC6).

    Returns (n_atoms,), or (n_pairs, n_atoms) with `per_pair`.  The numbers sum to the prediction
    without the bias, exactly, so an atom's number is its share of the predicted value and not a
    score of its own.

    **Why it is exact.** Each channel of the autocorrelation is a double sum over atom pairs,

        F[(f,g), k] = sum over atoms a, b of  W[a,f] B[a,b,k] W[b,g],

    and the prediction is a fixed linear functional of those channels
    (:func:`channel_coefficients`).  A pair (a, b) is therefore one term of the prediction, and
    the only choice to make is how to divide a pair between its two atoms; it is divided equally,
    which is the only split that treats the two the same way.  A self term (a = a) is one atom's
    alone.  Nothing is left over: the atom numbers add up to the prediction minus the bias.

    `coefficients` is the vector :func:`channel_coefficients` returns, `scale` what every channel
    was multiplied by before the kernel was taken, and `normalize` says whether the kernel was
    cosine-normalised, which divides the molecule's row by its own length.
    """
    W = np.asarray(mol.weights, float)
    n_atoms, n_fields = W.shape
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    centres = spec.centres
    n_bins = len(centres)
    out = np.zeros((len(pairs), n_atoms))
    if n_atoms == 0:
        return out if per_pair else out.sum(axis=0)
    c = np.asarray(coefficients, float)
    s = np.ones(len(c)) if scale is None else np.asarray(scale, float)
    need = len(pairs) * n_bins
    if len(c) < need or len(s) < len(c):
        raise ValueError(f"the model carries {len(c)} channels and this molecule needs {need}: "
                         "the fields of the two do not match")
    if len(c) > need:
        # the signed three-point channels are a sum over triples, not pairs, so they have no
        # place in a split over two atoms; refusing is better than quietly dropping them
        raise ValueError("atom contributions are not defined for the signed three-point "
                         "channels: build the model without --ac-chirality to decompose it")
    gamma = (c * s)[:need].reshape(len(pairs), n_bins)

    d = np.linalg.norm(mol.coords[:, None, :] - mol.coords[None, :, :], axis=-1)
    if not spec.include_self:
        np.fill_diagonal(d, np.inf)
    B = np.exp(-((d[:, :, None] - centres[None, None, :]) ** 2) / (2.0 * spec.sigma ** 2))
    # T[k] = B[:, :, k] @ W: for every atom and every field, what the rest of the molecule puts
    # into that distance channel
    T = np.einsum("abk,bf->kaf", B, W, optimize=True)
    for i, (f, g) in enumerate(pairs):
        mult = 2.0 if f != g else 1.0
        # a as the first index of the pair, and a as the second; half of each
        share = 0.5 * mult * (W[:, f][None, :] * T[:, :, g] + W[:, g][None, :] * T[:, :, f])
        out[i] = gamma[i] @ share
    if normalize:
        row = autocorrelation(mol, spec, pairs)[:need] * s[:need]
        length = float(np.sqrt(max(np.sum(row * row), 1e-300)))
        out /= length
    return out if per_pair else out.sum(axis=0)


def pair_contributions(mol: MoleculeFields, coefficients: Sequence[float],
                       scale: Optional[Sequence[float]] = None,
                       spec: AutocorrelationSpec = AutocorrelationSpec(),
                       fieldset: Optional[FieldSet] = None,
                       normalize: bool = True) -> List[Tuple[str, float]]:
    """The same split taken over the channel blocks instead of the atoms.

    One number per pair of field types -- "steric with steric", "electrostatic with hba" -- which
    says which kind of channel the prediction came from.  The numbers add up to the same total as
    :func:`atom_contributions`.
    """
    n_fields = np.asarray(mol.weights, float).shape[1]
    per = atom_contributions(mol, coefficients, scale=scale, spec=spec, normalize=normalize,
                             per_pair=True)
    labels = []
    for i, (f, g) in enumerate(field_pairs(n_fields)):
        if fieldset is not None:
            name = (f"{fieldset.specs[f].name} with {fieldset.specs[g].name}" if f != g
                    else f"{fieldset.specs[f].name}")
        else:
            name = f"{f} with {g}" if f != g else str(f)
        labels.append((name, float(per[i].sum())))
    return labels
