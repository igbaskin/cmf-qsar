"""Reciprocal-space channels of a crystal: the typed structure factors (plan item M6).

The periodic autocorrelation of ``periodic.py`` is a sum over a neighbour list, so it stops at
the cutoff radius ``r_max``.  Order that extends further -- a superstructure with a long repeat,
the long-range part of the electrostatic interaction, a modulation -- leaves no mark on it.  This
module adds the channels that do carry such order, and it pays nothing for the distance: they are
sums over the atoms of **one cell** with no neighbour list at all.

The structure factor
--------------------
For a field f with weight ``w_lf`` on atom l at position ``r_l``, and a vector **G** of the
reciprocal lattice (the lattice of vectors with ``G . T = 2 pi n`` for every lattice translation
T), the *typed structure factor* is the sum over the atoms of one cell ::

    S_f(G) = sum_l w_lf exp(i G . r_l).

It is the Fourier transform of the field of the whole infinite crystal, evaluated at G, and it
is exactly what a diffraction experiment measures: ``|S_f(G)|^2`` is the intensity of the Bragg
reflection at G when field f is the scattering power.  A single value of S_f(G) already holds
information about the whole structure, however far the order extends, which is why this
descriptor has no cutoff radius.

The channels
------------
For a pair of field types (f, g) and a channel at the length ``q_k``, ::

    Psi_fg(q_k) = (1/N) { (2 pi)^3 / (V I_k) sum_{G != 0} A_fg(G) F(|G|) B_k(|G|)
                          + (e_fg - c_fg) <F>_k },

with

* ``A_fg(G) = |S_f(G)|^2`` for one field type and ``2 Re[S_f(G) conj(S_g(G))]`` for two, which
  counts a pair of different types in both orders exactly as the real-space descriptor does;
* ``c_fg`` the part of ``A_fg`` that comes from an atom with itself, ``sum_l w_lf w_lg`` times
  the same factor, and ``e_fg`` the same sum of the *second moments* of a disordered site.  The
  two are equal for an ordered structure, so the term vanishes there; where they differ it is
  the correction that partial occupancy asks for (plan item M2).  Neither is summed over the
  reciprocal lattice: a quantity that does not depend on G contributes ``<F>_k``, the mean of
  the form factor over the channel, and taking it from that integral rather than from the grid
  is what makes the descriptor independent of the choice of cell *exactly* (see
  :meth:`ReciprocalSpec.form_average`).  ``include_self=False`` subtracts the self scattering
  altogether, which leaves the distinct scattering of Faber and Ziman;
* ``F(q) = exp(-a^2 q^2)`` is the *form factor*: the two fields are Gaussians of width ``a``
  (``form_sigma``), so their product in reciprocal space carries this factor.  It falls with q,
  which makes ``a`` a **resolution**: a wide field sees only the first few reflections, a narrow
  one sees fine detail.  This is the reciprocal-space reading of the width that the real-space
  descriptor uses along the distance axis;
* ``B_k(q) = exp(-(q - q_k)^2 / 2 sigma^2) w(q)`` is the channel, a Gaussian about ``q_k`` with
  the same tapered window as the real-space cutoff, and ``I_k`` its shell integral
  ``INT 4 pi q^2 B_k(q) dq``;
* ``V`` is the volume of the cell and ``N`` the number of atoms in it (the total occupancy for a
  disordered cell).

Dividing by ``V I_k / (2 pi)^3`` -- the number of reciprocal-lattice points a channel is
expected to contain -- makes the channel the *spherical average of the intensity*, which is the
powder diffraction profile of that pair of field types.  It also makes the descriptor exactly
independent of the choice of cell: a supercell has as many times more reciprocal-lattice points
as it has cells, and the new ones carry no intensity, so numerator and denominator grow by the
same factor.  Nothing here depends on the origin or the orientation either, so the descriptor
has the same invariances as the real-space one and adds no new assumption.

Two further things follow from the definition and are worth stating, because the real-space
module needs both and has them only as approximations:

* **The uniform background is the G = 0 term.**  ``A_fg(0) = (sum_l w_lf)(sum_l w_lg)``, the
  forward scattering, and that is the quantity the real-space descriptor subtracts as the
  uniform-density term.  Here it is one term of the sum, left out by ``include_zero=False``, and
  no integral has to be approximated to remove it.
* **The two descriptions are one object.**  Summing the field of the crystal in real space and
  summing it over the reciprocal lattice give the same number (Poisson summation, the identity
  an Ewald sum is built on): for any width ``u``, ::

      sum_{l, m, T} w_lf w_mg exp(-|r_l - r_m - T|^2 / 2 u^2)
          = (2 pi u^2)^{3/2} / V  sum_G A_fg(G) exp(-u^2 |G|^2 / 2).

  :func:`parseval_residual` computes both sides and their difference.  The left side is this
  program's real-space pair sum and the right side is this module's sum over G, so the identity
  is a test of the two pieces of code against each other, and it is the statement that the
  reciprocal channels are the same descriptor read at a different resolution rather than a new
  one.

What this adds and what it does not
-----------------------------------
A reciprocal channel at small ``q`` describes the structure on a length scale ``2 pi / q``, so
the low-q channels carry exactly what a real-space cutoff cannot reach.  A cell with few atoms
has few reciprocal-lattice points below ``q_max``, and its profile is a short series of sharp
peaks whose positions are set by the lattice metric; a large cell gives a dense profile.  The
descriptor still cannot tell a structure from its mirror image, because the intensity of a
reflection is the same for both: that deficiency is shared with every two-point descriptor and
is the subject of plan item M10.

Only structures periodic in all three directions are accepted.  A layer or a chain has a
reciprocal lattice of lower dimension, and the intensity then depends on the component of G
across the non-periodic directions as well, which these channels do not resolve.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .fields import FieldSet
from .invariant import block_sizes, field_pairs

#: the largest number of reciprocal-lattice vectors held in memory at once.  The sum over G is
#: taken in chunks of this size, so that a large cell with a fine grid costs time and not memory:
#: one chunk is n_chunk * n_atoms complex numbers.
G_CHUNK = 8192


@dataclass(frozen=True)
class ReciprocalSpec:
    """The grid of reciprocal-space channels: where they sit, how wide they are, and the
    resolution the form factor sets.

    The layout matches :class:`cmfqsar.periodic.PeriodicSpec`, so the block and label helpers of
    ``invariant.py`` accept this spec unchanged: channels at 0, ``step``, 2 ``step``, ... up to
    ``q_max``, one block per pair of field types.  ``chirality`` exists only for those helpers
    and is always False.
    """
    #: longest reciprocal length resolved, in 1/angstrom.  2 pi / q_max is the shortest distance
    #: the channels distinguish, so q_max is a resolution limit and not a cutoff: every atom of
    #: the cell enters every channel whatever the distances in the structure.
    q_max: float = 8.0
    step: float = 0.25               # spacing of the channels (1/angstrom)
    sigma: float = 0.25              # width of a channel along the q axis (1/angstrom)
    #: width ``a`` of the Gaussian field in angstrom: the form factor is exp(-a^2 q^2).  It sets
    #: how much of the high-q detail reaches the descriptor, and is the resolution knob of the
    #: whole module.
    form_sigma: float = 0.3
    include_self: bool = True        # keep the self-scattering term, which carries the composition
    include_zero: bool = False       # keep G = 0, the uniform background (see the module docstring)
    taper: float = 1.0               # width of the cosine taper below q_max (0: hard cutoff)
    chirality: bool = False          # always False; see the class docstring

    @property
    def d_max(self) -> float:
        """The spec's own extent, for the helpers that ask every spec for one."""
        return self.q_max

    @property
    def has_heights(self) -> bool:
        return False

    @property
    def centres(self) -> np.ndarray:
        n = max(int(round(self.q_max / self.step)) + 1, 1)
        return np.arange(n, dtype=float) * self.step

    @property
    def n_bins(self) -> int:
        return len(self.centres)

    def window(self, q: np.ndarray) -> np.ndarray:
        """1 below q_max, a half-cosine over the last ``taper``, 0 beyond."""
        q = np.asarray(q, float)
        w = np.where(q < self.q_max, 1.0, 0.0)
        if self.taper > 0:
            q0 = self.q_max - self.taper
            inside = (q > q0) & (q < self.q_max)
            w = np.where(inside, 0.5 * (1.0 + np.cos(np.pi * (q - q0) / self.taper)), w)
        return w

    def form_factor(self, q: np.ndarray) -> np.ndarray:
        """``exp(-a^2 q^2)``: the product of the two Gaussian fields in reciprocal space."""
        q = np.asarray(q, float)
        return np.exp(-(float(self.form_sigma) ** 2) * q ** 2)

    def shell_integral(self, n_points: int = 4000) -> np.ndarray:
        """``INT 4 pi q^2 exp(-(q - q_k)^2 / 2 sigma^2) w(q) dq`` for every channel.

        The number of reciprocal-lattice points a channel is expected to hold, per unit density
        of those points.  It depends on the grid, the width and the window only, so it is
        computed once per spec.
        """
        return self._integral(1.0, n_points)

    def form_average(self, n_points: int = 4000) -> np.ndarray:
        """The mean of the form factor over every channel: the shell integral with ``F`` in it,
        divided by the shell integral without.

        It is what a quantity that does not depend on the direction or the length of G
        contributes to a channel, and the self-scattering term is such a quantity.  Computing
        that contribution from this integral rather than by summing it over the
        reciprocal-lattice points is what makes the descriptor exactly independent of the choice
        of cell: a small cell has few points per channel, so the sum would estimate a constant
        badly and differently for a supercell of the same structure.
        """
        return self._integral(None, n_points) / np.maximum(self._integral(1.0, n_points), 1e-300)

    def _integral(self, flat: Optional[float], n_points: int) -> np.ndarray:
        q = np.linspace(0.0, self.q_max, int(n_points))
        g = np.exp(-((q[:, None] - self.centres[None, :]) ** 2) / (2.0 * self.sigma ** 2))
        f = 4.0 * np.pi * q[:, None] ** 2 * g * self.window(q)[:, None]
        if flat is None:
            f = f * self.form_factor(q)[:, None]
        return np.trapezoid(f, q, axis=0)


@functools.lru_cache(maxsize=32)
def _shell_integral(spec: ReciprocalSpec) -> np.ndarray:
    out = spec.shell_integral()
    out.flags.writeable = False
    return out


@functools.lru_cache(maxsize=32)
def _form_average(spec: ReciprocalSpec) -> np.ndarray:
    out = spec.form_average()
    out.flags.writeable = False
    return out


# --------------------------------------------------------------------------- #
# The reciprocal lattice
# --------------------------------------------------------------------------- #
def reciprocal_cell(cell: np.ndarray) -> np.ndarray:
    """The reciprocal lattice vectors as rows: ``b_i . a_j = 2 pi delta_ij``."""
    A = np.asarray(cell, float)
    if abs(float(np.linalg.det(A))) < 1e-12:
        raise ValueError("the cell has no volume, so it has no reciprocal lattice")
    return 2.0 * np.pi * np.linalg.inv(A).T


def reciprocal_vectors(cell: np.ndarray, q_max: float,
                       include_zero: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """The reciprocal-lattice vectors with ``|G| <= q_max``, and their lengths.

    A vector ``G = sum_i h_i b_i`` has ``h_i = G . a_i / 2 pi``, so ``|h_i| <= q_max |a_i| / 2 pi``
    bounds the search exactly, whatever the shape of the cell.
    """
    A = np.asarray(cell, float)
    B = reciprocal_cell(A)
    lengths = np.linalg.norm(A, axis=1)
    n = np.floor(float(q_max) * lengths / (2.0 * np.pi) + 1e-9).astype(int)
    ranges = [np.arange(-k, k + 1) for k in np.maximum(n, 0)]
    h = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
    G = h @ B
    q = np.linalg.norm(G, axis=1)
    keep = q <= float(q_max)
    if not include_zero:
        keep &= q > 1e-9
    return G[keep], q[keep]


def structure_factors(crystal, G: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """``S_f(G)`` for every vector of ``G`` and every field: a complex ``(len(G), n_fields)``.

    ``weights`` overrides the field weights of the structure, which is what the Parseval test
    and the per-field sums need.
    """
    W = np.asarray(crystal.weights if weights is None else weights, float)
    R = np.asarray(crystal.coords, float)
    G = np.asarray(G, float).reshape(-1, 3)
    if not len(G) or not len(R):
        return np.zeros((len(G), W.shape[1]), complex)
    out = np.zeros((len(G), W.shape[1]), complex)
    for start in range(0, len(G), G_CHUNK):
        sl = slice(start, start + G_CHUNK)
        phase = np.exp(1j * (G[sl] @ R.T))          # (chunk, n_atoms)
        out[sl] = phase @ W
    return out


# --------------------------------------------------------------------------- #
# The descriptor
# --------------------------------------------------------------------------- #
def reciprocal_autocorrelation(crystal, spec: ReciprocalSpec = ReciprocalSpec(),
                               pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """Reciprocal-space channels of one structure, flattened: ``(n_pairs * n_bins,)``.

    The formula is the one in the module docstring.  The sum runs over the reciprocal-lattice
    vectors of the cell out to ``q_max`` and over the atoms of one cell; there is no neighbour
    list and no cutoff in real space.
    """
    if getattr(crystal, "centre", None) is not None:
        raise ValueError(
            "a site record (plan item M5) has no reciprocal-space channels of its own: a "
            "structure factor is a sum over the whole cell, so the intensity cannot be divided "
            "among the sites the way the pair sum can. Describe the site in real space.")
    pbc = tuple(bool(p) for p in getattr(crystal, "pbc", (True, True, True)))
    if sum(pbc) != 3:
        raise ValueError(
            f"the reciprocal-space channels need a structure periodic in all three directions; "
            f"{getattr(crystal, 'name', 'this structure')} is periodic in {sum(pbc)}. A layer or "
            f"a chain has a reciprocal lattice of lower dimension and the intensity then depends "
            f"on the direction across the vacuum as well (see the module docstring).")
    W = np.asarray(crystal.weights, float)
    n_atoms, n_fields = W.shape
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    centres = spec.centres
    if n_atoms == 0:
        return np.zeros(len(pairs) * len(centres))
    fi = np.array([f for f, _ in pairs], int)
    gi = np.array([g for _, g in pairs], int)
    mult = np.where(fi == gi, 1.0, 2.0)            # a pair of different types, counted both ways

    # the self-scattering constant of every pair: the second moments on a disordered site, where
    # the mean of a product is not the product of the means (plan item M2)
    if crystal.second_moments is not None:
        M = np.asarray(crystal.second_moments, float).sum(axis=0)
        n_norm = (float(np.sum(crystal.occupancy)) if crystal.occupancy is not None
                  else float(n_atoms))
    else:
        M = W.T @ W
        n_norm = float(n_atoms)
    C = W.T @ W                                    # the same sum from the mean weights
    c_self = mult * M[fi, gi]
    c_distinct = mult * C[fi, gi]

    G, q = reciprocal_vectors(crystal.cell, spec.q_max, include_zero=spec.include_zero)
    rows = np.zeros((len(pairs), len(centres)))
    if len(G):
        inv = 2.0 * spec.sigma ** 2
        for start in range(0, len(G), G_CHUNK):
            sl = slice(start, start + G_CHUNK)
            qs = q[sl]
            S = structure_factors(crystal, G[sl])
            A = mult[:, None] * np.real(S[:, fi] * np.conj(S[:, gi])).T       # A_fg(G), per pair
            B = (np.exp(-((qs[:, None] - centres[None, :]) ** 2) / inv)
                 * (spec.form_factor(qs) * spec.window(qs))[:, None])
            rows += A @ B
        volume = float(abs(np.linalg.det(np.asarray(crystal.cell, float))))
        rows *= (2.0 * np.pi) ** 3 / max(volume, 1e-12)
        rows /= np.maximum(_shell_integral(spec), 1e-300)[None, :]
    # Everything that does not depend on G is added here and not summed over the grid (see
    # ReciprocalSpec.form_average).  On a disordered site the mean of a product is not the
    # product of the means, so the self-scattering term carries the second moments while the
    # sum above carries the mean weights: the difference is the correction (plan item M2).
    flat = _form_average(spec)[None, :]
    if spec.include_self:
        rows = rows + (c_self - c_distinct)[:, None] * flat
    else:
        rows = rows - c_distinct[:, None] * flat
    return (rows / max(n_norm, 1e-12)).ravel()


def reciprocal_feature_matrix(crystals: Sequence, spec: ReciprocalSpec = ReciprocalSpec(),
                              n_fields: Optional[int] = None, progress=None) -> np.ndarray:
    """Reciprocal-space channels of a set of structures, one row each."""
    if not len(crystals):
        return np.zeros((0, 0))
    n_fields = n_fields if n_fields is not None else crystals[0].weights.shape[1]
    pairs = field_pairs(n_fields)
    out = np.zeros((len(crystals), sum(reciprocal_block_sizes(spec, n_fields))))
    for k, c in enumerate(crystals):
        if c.weights.shape[1] != n_fields:
            raise ValueError(
                f"{c.name or f'structure {k}'} carries {c.weights.shape[1]} fields where the set "
                f"carries {n_fields}: build the fields of every structure from one field set.")
        out[k] = reciprocal_autocorrelation(c, spec, pairs)
        if progress is not None and k % 25 == 0:
            progress(f"Reciprocal channels {k + 1}/{len(crystals)}", k / max(len(crystals), 1))
    return out


def reciprocal_block_sizes(spec: ReciprocalSpec, n_fields: int) -> List[int]:
    return block_sizes(spec, n_fields)


def reciprocal_channel_labels(fieldset: FieldSet, spec: ReciprocalSpec) -> List[str]:
    """One label per feature: which pair of field types, at which reciprocal length."""
    names = [s.label for s in fieldset.specs]
    out = []
    for f, g in field_pairs(len(names)):
        pair = names[f] if f == g else f"{names[f]} x {names[g]}"
        for q in spec.centres:
            out.append(f"{pair} @ q {q:.2f} 1/A")
    return out


def joint_features(crystals: Sequence, real_spec, spec: ReciprocalSpec = ReciprocalSpec(),
                   n_fields: Optional[int] = None, progress=None) -> Tuple[np.ndarray, List[int]]:
    """The real-space and reciprocal-space channels side by side, and the block widths.

    The two descriptions of the same structure in one feature matrix: the pair blocks of the
    periodic autocorrelation first, then the pair blocks of this module.  Each block keeps its
    own scale (:func:`cmfqsar.invariant.channel_scale` takes the widths), so neither description
    dominates the kernel by its units.  Mixing the two as separate *kernels* is the other way,
    and the one the benchmark uses when it wants to measure how much weight each deserves.
    """
    from .periodic import periodic_block_sizes, periodic_feature_matrix

    n_fields = n_fields if n_fields is not None else crystals[0].weights.shape[1]
    A = periodic_feature_matrix(crystals, real_spec, n_fields=n_fields, progress=progress)
    B = reciprocal_feature_matrix(crystals, spec, n_fields=n_fields, progress=progress)
    widths = (list(periodic_block_sizes(real_spec, n_fields))
              + list(reciprocal_block_sizes(spec, n_fields)))
    return np.hstack([A, B]), widths


def joint_labels(fieldset: FieldSet, real_spec, spec: ReciprocalSpec) -> List[str]:
    """The labels of :func:`joint_features`, in the same order."""
    from .periodic import periodic_channel_labels

    return (list(periodic_channel_labels(fieldset, real_spec))
            + list(reciprocal_channel_labels(fieldset, spec)))


# --------------------------------------------------------------------------- #
# The identity that ties the two descriptions together
# --------------------------------------------------------------------------- #
def gaussian_lattice_sum(crystal, width: float,
                         pairs: Optional[Sequence[Tuple[int, int]]] = None,
                         r_cut: float = 8.0) -> np.ndarray:
    """``sum_{l, m, T} w_lf w_mg exp(-|r_l - r_m - T|^2 / 2 u^2)`` for every pair of field types.

    The real-space side of the identity in the module docstring: a sum over the neighbour list of
    the cell, with the atom-with-itself term at distance zero added, out to ``r_cut`` widths.
    """
    from .periodic import _neighbour_pairs

    W = np.asarray(crystal.weights, float)
    n_fields = W.shape[1]
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    fi = np.array([f for f, _ in pairs], int)
    gi = np.array([g for _, g in pairs], int)
    mult = np.where(fi == gi, 1.0, 2.0)
    i, j, d = _neighbour_pairs(crystal, float(r_cut) * float(width))
    T = np.zeros((n_fields, n_fields))
    if len(d):
        b = np.exp(-(d ** 2) / (2.0 * float(width) ** 2))
        T = (W[i] * b[:, None]).T @ W[j]
    C = W.T @ W
    return mult * (T[fi, gi] + C[fi, gi])


def gaussian_reciprocal_sum(crystal, width: float,
                            pairs: Optional[Sequence[Tuple[int, int]]] = None,
                            q_cut: float = 8.0) -> np.ndarray:
    """``(2 pi u^2)^{3/2} / V sum_G A_fg(G) exp(-u^2 |G|^2 / 2)`` for every pair of field types.

    The reciprocal side of the same identity, over the reciprocal lattice out to ``q_cut / u``.
    """
    W = np.asarray(crystal.weights, float)
    n_fields = W.shape[1]
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    fi = np.array([f for f, _ in pairs], int)
    gi = np.array([g for _, g in pairs], int)
    mult = np.where(fi == gi, 1.0, 2.0)
    u = float(width)
    G, q = reciprocal_vectors(crystal.cell, float(q_cut) / u, include_zero=True)
    out = np.zeros(len(pairs))
    for start in range(0, len(G), G_CHUNK):
        sl = slice(start, start + G_CHUNK)
        S = structure_factors(crystal, G[sl])
        damp = np.exp(-(u ** 2) * q[sl] ** 2 / 2.0)
        out += mult * (np.real(S[:, fi] * np.conj(S[:, gi])) * damp[:, None]).sum(axis=0)
    volume = float(abs(np.linalg.det(np.asarray(crystal.cell, float))))
    return out * (2.0 * np.pi * u ** 2) ** 1.5 / max(volume, 1e-12)


def parseval_residual(crystal, width: float = 0.5,
                      pairs: Optional[Sequence[Tuple[int, int]]] = None,
                      cut: float = 8.0) -> dict:
    """The two sides of the identity of the module docstring, and how far apart they are.

    Returns the real-space sum, the reciprocal-space sum and the largest relative difference
    over the pairs of field types.  The two are the same quantity computed by two independent
    pieces of code -- a neighbour list over the images, and a sum over the reciprocal lattice --
    so a small residual is evidence that both are right, and it is the test the theory section
    of the publication plan asks for.
    """
    real = gaussian_lattice_sum(crystal, width, pairs, r_cut=cut)
    recip = gaussian_reciprocal_sum(crystal, width, pairs, q_cut=cut)
    # A pair of field types whose own sum nearly cancels is measured against the largest sum of
    # the structure, not against its own: a relative difference on a quantity that is itself
    # round-off would say nothing about the agreement of the two routes.
    floor = 1e-6 * float(np.max(np.abs(real)) if len(real) else 0.0)
    scale = np.maximum(np.maximum(np.abs(real), np.abs(recip)), floor)
    use = scale > 1e-12
    rel = np.abs(real - recip)[use] / scale[use] if use.any() else np.zeros(0)
    return {"width": float(width), "real": real, "reciprocal": recip,
            "relative": rel, "worst": float(np.max(rel)) if len(rel) else 0.0}


# --------------------------------------------------------------------------- #
# Reading a fitted model: the pattern in q
# --------------------------------------------------------------------------- #
def channel_spread(model) -> np.ndarray:
    """The standard deviation of every scaled, normalised channel over the training structures.

    A weight only matters where the channel varies, so this is the other half of reading a
    fitted model; :func:`cmfqsar.periodic.top_channels` is where the program defines it.
    """
    Z = np.asarray(model.ac_train_, float) * np.asarray(model.ac_column_scale, float)
    Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
    return Z.std(axis=0)


def reciprocal_pattern(model, n_fields: Optional[int] = None,
                       spec: Optional[ReciprocalSpec] = None,
                       effect: bool = False) -> np.ndarray:
    """The reciprocal channels of a fitted model read back: ``(n_pairs, n_bins)``.

    The co-pattern of the real-space descriptor says which pair of field types at which distance
    raises the property.  This is the same reading in reciprocal space: which pair of field types
    at which reciprocal length, that is, at which spacing of planes in the structure.  A positive
    value at ``q`` means that more intensity there raises the property.

    ``effect=True`` multiplies each weight by the spread of its channel over the training
    structures, which is how much that channel actually moves the prediction and is how the
    program ranks channels elsewhere.  The weights themselves change sign from one channel to
    the next, because neighbouring channels of a Gaussian grid are strongly correlated and a
    ridge splits a single feature among them; the product with the spread is the quantity to
    read.
    """
    from .periodic import channel_weights

    w = np.asarray(channel_weights(model), float)
    if effect:
        w = w * channel_spread(model)
    spec = spec if spec is not None else model._ac_spec()
    n_bins = int(spec.n_bins)
    if n_fields is None:
        n_fields = len(model.fieldset.specs)
    n_pairs = len(field_pairs(int(n_fields)))
    if w.size < n_pairs * n_bins:
        raise ValueError(f"the model carries {w.size} channel weights, fewer than the "
                         f"{n_pairs * n_bins} reciprocal channels of this spec")
    return w[-n_pairs * n_bins:].reshape(n_pairs, n_bins)


def pattern_table(model, fieldset: FieldSet, spec: Optional[ReciprocalSpec] = None,
                  n: int = 10) -> List[Tuple[str, float, float]]:
    """The ``n`` reciprocal channels that move the prediction most.

    (label, weight times the spread of the channel, the reciprocal length q), ranked by the
    size of the first, which is the ranking :func:`cmfqsar.periodic.top_channels` uses.
    """
    spec = spec if spec is not None else model._ac_spec()
    P = reciprocal_pattern(model, len(fieldset.specs), spec, effect=True)
    labels = reciprocal_channel_labels(fieldset, spec)
    flat = P.ravel()
    order = np.argsort(-np.abs(flat))[:int(n)]
    centres = np.tile(spec.centres, P.shape[0])
    return [(labels[int(k)], float(flat[int(k)]), float(centres[int(k)])) for k in order]


def spacings(spec: ReciprocalSpec) -> np.ndarray:
    """The plane spacing ``2 pi / q`` of every channel, in angstrom; infinity for the first.

    What a channel means as a length: a channel at ``q`` describes the structure on the scale of
    ``2 pi / q``, which is how the reciprocal channels are compared with the cutoff radius of the
    real-space descriptor.
    """
    q = spec.centres
    with np.errstate(divide="ignore"):
        return np.where(q > 0, 2.0 * np.pi / np.maximum(q, 1e-300), np.inf)
