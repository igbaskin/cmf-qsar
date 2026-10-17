"""Alignment-free but geometry-aware kernels: autocorrelation of the molecular fields.

Everything else in this program compares two molecules by the overlap integral of their fields,
which needs them to share a frame.  That is the strength of the approach and its one hard
requirement, and the heterogeneous-series work shows what it costs when the frame is in doubt:
the alignment-free comparator there is the count kernel, which is pose-free only because it has
thrown the geometry away entirely.

This module is the missing middle.  For every pair of field types it takes the distance
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

One thing the descriptor cannot see: a distance distribution is the same for a molecule and its
mirror image, so enantiomers are indistinguishable here.  Chirality needs a signed quantity, the
triple product of three displacement vectors, which is what three-point pharmacophores add; the
distance form below is the two-point case.

What is lost with the frame is the co-field: a map in space needs a space to live in.  The
decomposition that survives is the one in the descriptor's own currency -- which pair of field
types at which distance carries the weight -- and :func:`channel_labels` names those channels so
a fitted model can be read that way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .fields import FieldSet, MoleculeFields


@dataclass(frozen=True)
class AutocorrelationSpec:
    """The distance grid the field autocorrelation is evaluated on."""
    d_max: float = 12.0
    step: float = 0.75
    sigma: float = 0.75              # width of the smoothing along the distance axis
    include_self: bool = True        # the l = m terms, which carry the plain count of each type

    @property
    def centres(self) -> np.ndarray:
        n = max(int(round(self.d_max / self.step)) + 1, 1)
        return np.arange(n, dtype=float) * self.step

    @property
    def n_bins(self) -> int:
        return len(self.centres)


def field_pairs(n_fields: int) -> List[Tuple[int, int]]:
    """The unordered pairs of field types, diagonal first: (0,0), (1,1), ..., then (0,1), ..."""
    return ([(f, f) for f in range(n_fields)]
            + [(f, g) for f in range(n_fields) for g in range(f + 1, n_fields)])


def channel_labels(fieldset: FieldSet, spec: AutocorrelationSpec) -> List[str]:
    """One label per feature: which two field types, at which distance."""
    names = [s.label for s in fieldset.specs]
    out = []
    for f, g in field_pairs(len(names)):
        pair = names[f] if f == g else f"{names[f]} x {names[g]}"
        for d in spec.centres:
            out.append(f"{pair} @ {d:.2f} A")
    return out


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
    return out.ravel()


def feature_matrix(mols: Sequence[MoleculeFields],
                   spec: AutocorrelationSpec = AutocorrelationSpec(),
                   n_fields: Optional[int] = None,
                   progress=None) -> np.ndarray:
    """Autocorrelation features of a set of molecules, one row each."""
    if not len(mols):
        return np.zeros((0, 0))
    n_fields = n_fields if n_fields is not None else mols[0].weights.shape[1]
    pairs = field_pairs(n_fields)
    out = np.zeros((len(mols), len(pairs) * spec.n_bins))
    for i, m in enumerate(mols):
        out[i] = autocorrelation(m, spec, pairs)
        if progress is not None and i % 25 == 0:
            progress(f"Field autocorrelation {i + 1}/{len(mols)}", i / max(len(mols), 1))
    return out


def channel_scale(train: np.ndarray, n_bins: int) -> np.ndarray:
    """One scale per field-type pair, so that no pair dominates by its units alone.

    The electrostatic field carries charges and the steric one carries ones, so their
    autocorrelations differ by orders of magnitude; each block is divided by its root mean square
    over the training molecules, which is what `scale_fields` does for the kernels.
    """
    X = np.asarray(train, float)
    blocks = X.shape[1] // n_bins
    s = np.ones(X.shape[1])
    for b in range(blocks):
        sl = slice(b * n_bins, (b + 1) * n_bins)
        rms = float(np.sqrt(np.mean(X[:, sl] ** 2)))
        s[sl] = 1.0 / rms if rms > 1e-300 else 0.0
    return s


def autocorrelation_kernel(train: np.ndarray, test: Optional[np.ndarray] = None,
                           scale: Optional[np.ndarray] = None, n_bins: int = 1,
                           normalize: bool = True) -> np.ndarray:
    """Cosine-normalised linear kernel on the autocorrelation features.

    The normalisation is applied to the joint kernel, not block by block: normalising each pair
    of field types on its own would divide out the very magnitudes that distinguish molecules,
    the same trap the alignment-free count kernel falls into when it is normalised field by
    field.
    """
    A = np.asarray(train, float)
    B = A if test is None else np.asarray(test, float)
    if scale is None:
        scale = channel_scale(A, n_bins)
    A, B = A * scale, B * scale
    K = B @ A.T
    if not normalize:
        return K
    da = np.sqrt(np.maximum(np.sum(A * A, axis=1), 1e-300))
    db = np.sqrt(np.maximum(np.sum(B * B, axis=1), 1e-300))
    return K / np.outer(db, da)
