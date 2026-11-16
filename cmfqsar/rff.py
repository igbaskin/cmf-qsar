"""Random Fourier features of the field kernel: explicit features without landmarks.

**Why they apply.**  The kernel of two molecules for one field is the overlap integral of their
fields, ``K(A, B) = INT rho_A(r) rho_B(r) dr``.  By Parseval's theorem the same integral is an
integral over frequencies of the two Fourier transforms,

    K(A, B) = (2 pi)^-3 INT F_A(omega) conj(F_B(omega)) d omega,

and for a field made of atom-centred Gaussians ``w_l exp(-b_l |r - r_l|^2 / 2)`` the transform is
a sum over the atoms,

    F_A(omega) = sum_l w_l a_l(omega) exp(-i omega . r_l),   a_l(omega) = (2 pi / b_l)^{3/2}
                                                                       exp(-|omega|^2 / (2 b_l)).

So the field kernel is a Gaussian kernel on positions in R^3, summed over the atom pairs with the
atom weights.  That kernel is shift-invariant, and the random features of Rahimi and Recht
(NIPS 2007, "Random features for large-scale kernel machines") apply to it: draw D frequencies
from a density p, and the integral becomes a mean over them,

    K(A, B) ~ (1/D) sum_k Re[F_A(omega_k) conj(F_B(omega_k))] / ((2 pi)^3 p(omega_k)).

Each frequency gives a molecule two features, the real and the imaginary part of ``F_A``,

    sum_l w_l a_l(omega_k) cos(omega_k . r_l)   and   sum_l w_l a_l(omega_k) sin(omega_k . r_l),

which need no grid, no landmarks and no training molecules.  The estimate is unbiased for every
pair of molecules.  An earlier assessment in the plan (item B6) said that random Fourier
features do not fit the CMF kernel because it has no vector input.  The atoms are the vector
input: the molecule kernel is a weighted sum of the kernel on R^3.

**The density of the frequencies.**  For a field whose Gaussians share one width b (the
electrostatic, hydrophobic and donor and acceptor fields), p is the normal density with variance
``b / 2`` per axis.  Then ``a(omega)^2 / p(omega)`` is a constant, every frequency carries the
same weight, and the features are the classical ones.  The steric field has a width per atom
(``b_l = alpha / R_l^2``); p is then matched to the narrowest Gaussian, so that the ratio stays
bounded for every atom (importance sampling).  The same rule covers the probe fields, whose atoms
carry several Gaussians, and the electric-field kind of plan item EC5, whose kernel integrates
the gradients of the fields: a gradient multiplies the transform by ``i omega``, so its features
are the ordinary ones times ``|omega|``.

**Four ways to choose the frequencies.**

* ``"mc"``: independent draws from p (Monte Carlo); the error of a kernel value falls as
  ``1 / sqrt(D)``.
* ``"orthogonal"``: the orthogonal random features of Yu et al. (NIPS 2016): the directions come
  in triples that are exactly orthogonal, the lengths are drawn from the right distribution.
  Still unbiased, with a smaller variance.
* ``"qmc"``: a scrambled Sobol sequence mapped through the normal quantile function (quasi-Monte
  Carlo, Avron et al., JMLR 17(120), 2016).  In three dimensions its error can fall almost as
  ``1 / D``.
* ``"lattice"``: a cubic lattice of frequencies inside a ball, with spacing ``2 pi / P``.  By the
  Poisson summation formula this sum is the exact kernel of the molecules repeated periodically
  with period P, so the error is that of the periodic images only, and it is negligible once P
  exceeds the size of the frame by a few widths.  The number of features is not chosen; it
  follows from P and from the width of the field, as ``(P sqrt(b))^3``.  These are the structure
  factors of plan item M6 for a box around the molecules.

**What stays exact.**  As in :mod:`cmfqsar.lowrank`, the self-kernel of a molecule is computed
exactly and the cosine normalisation divides by it.  The feature map has the duck type of
:class:`cmfqsar.lowrank.LowRank`, so :func:`cmfqsar.lowrank.gram`, :func:`cmfqsar.lowrank.fit` and
:func:`cmfqsar.lowrank.cross_validate` fit a model on it unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fields import AtomTable, FieldSet, FieldSpec
from .kernels import cutoff_radius, normalize_kernels, self_kernels

SAMPLERS = ("mc", "orthogonal", "qmc", "lattice")
#: atoms times frequencies held at once while the features are built
CHUNK = 4_000_000
#: molecules per block when the features are streamed
BLOCK = 256
#: a lattice keeps the frequencies at which the product of two spectra is above this share of
#: its peak; the same tolerance as the cutoff of the kernel code (plan item B2)
LATTICE_TOL = 1e-6


def _kind(spec: FieldSpec) -> str:
    if getattr(spec, "is_probe", False):
        return "probe"
    if spec.kind in ("gauss", "efield"):
        return spec.kind
    return "steric"


def _widths(spec: FieldSpec, radii: np.ndarray) -> np.ndarray:
    """The width b of every term: one per atom, or the shared widths of a probe expansion."""
    if _kind(spec) == "probe":
        from .probes import basis_widths
        return np.asarray(basis_widths(), float)
    if _kind(spec) == "steric":
        return float(spec.alpha) / np.asarray(radii, float) ** 2
    return np.array([float(spec.alpha)])


def proposal_variance(spec: FieldSpec, radii: Sequence[float]) -> float:
    """The variance per axis of the normal density the frequencies are drawn from.

    It is half the largest width b present, which is the exact choice for a field of one width
    and keeps the importance weight of every atom at or below one for a field of several.
    """
    return 0.5 * float(np.max(_widths(spec, np.asarray(radii, float))))


# --------------------------------------------------------------------------- #
# The frequencies
# --------------------------------------------------------------------------- #


def frequencies(n: int, sd: float, sampler: str = "orthogonal",
                seed: Optional[int] = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(n, 3) frequencies from a normal density with standard deviation `sd` per axis, and the
    log of the weight each carries in the mean, ``1 / (n (2 pi)^3 p(omega))``."""
    if sampler not in ("mc", "orthogonal", "qmc"):
        raise ValueError(f"unknown sampler {sampler!r}; use one of {SAMPLERS}")
    n = int(n)
    if n < 1:
        raise ValueError("at least one frequency is needed")
    rng = np.random.default_rng(seed)
    if sampler == "mc":
        W = rng.standard_normal((n, 3))
    elif sampler == "orthogonal":
        blocks = []
        for _ in range(-(-n // 3)):
            Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
            Q = Q * np.sign(np.diag(R))[None, :]
            lengths = np.sqrt(rng.chisquare(3, size=3))
            blocks.append(Q.T * lengths[:, None])
        W = np.vstack(blocks)[:n]
    else:
        from scipy.stats import norm, qmc
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # Sobol prefers a power of two
            U = qmc.Sobol(d=3, scramble=True, seed=seed).random(n)
        W = norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))
    W = W * float(sd)
    logp = -1.5 * math.log(2 * math.pi * sd * sd) - 0.5 * (W * W).sum(axis=1) / (sd * sd)
    logq = -math.log(n) - 3 * math.log(2 * math.pi) - logp
    return W, logq


def lattice_frequencies(period: float, b_max: float,
                        tol: float = LATTICE_TOL) -> Tuple[np.ndarray, np.ndarray]:
    """The frequencies of a cubic lattice with spacing ``2 pi / period`` inside the ball where
    the spectrum of the narrowest Gaussian is not negligible, one of every pair (omega, -omega).

    The weight of a frequency is ``(spacing / 2 pi)^3``, doubled for every frequency whose
    partner was dropped.
    """
    step = 2 * math.pi / float(period)
    w_max = math.sqrt(float(b_max) * -math.log(tol))   # exp(-|w|^2 / b) = tol
    m = int(math.floor(w_max / step))
    ax = np.arange(-m, m + 1)
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
    G = G[(G * G).sum(axis=1) * step * step <= w_max * w_max]
    # keep one of each pair: the first non-zero index positive
    first = np.where(G[:, 0] != 0, G[:, 0], np.where(G[:, 1] != 0, G[:, 1], G[:, 2]))
    G = G[first >= 0]
    W = G * step
    mult = np.where(first[first >= 0] == 0, 1.0, 2.0)
    logq = np.log(mult) + 3 * math.log(step / (2 * math.pi))
    return W.astype(float), logq


def default_period(table: AtomTable, spec: FieldSpec, tol: float = LATTICE_TOL) -> float:
    """The longest extent of the atoms plus the distance at which the field's broadest overlap
    falls to `tol` of its peak.  Two molecules are never further apart than the extent, so the
    nearest periodic image of either is at least that distance away, and it adds less than `tol`.
    """
    ext = float((table.coords.max(axis=0) - table.coords.min(axis=0)).max())
    b_min = float(np.min(_widths(spec, table.radii)))
    return ext + cutoff_radius(0.25 * b_min, tol)


# --------------------------------------------------------------------------- #
# The feature map
# --------------------------------------------------------------------------- #


@dataclass
class FourierFeatures:
    """Frequencies per field and the weights that turn a molecule into a row of features.

    Fields that draw from the same density share one set of frequencies, so the cosines and sines
    of a block of atoms are computed once for all of them.
    """
    specs: Sequence[FieldSpec]
    omegas: List[np.ndarray]                  # one (D, 3) per frequency group
    logq: List[np.ndarray]                    # one (D,) per frequency group
    group: List[int]                          # frequency group of every field
    weights: np.ndarray                       # h_f c_f, one per field
    normalize: bool = True
    sampler: str = "orthogonal"
    kind: str = "fourier"

    @property
    def sizes(self) -> List[int]:
        """Feature columns per field: two per frequency."""
        return [2 * len(self.omegas[g]) for g in self.group]

    @property
    def rank(self) -> int:
        return int(sum(self.sizes))

    def field_features(self, part: AtomTable) -> List[np.ndarray]:
        """Unscaled, unnormalised features of every field: a list of (molecules, 2 D) arrays
        whose inner products estimate the field kernels."""
        n = part.n_mol
        A = len(part.coords)
        starts = np.asarray(part.offsets[:-1], int)
        out = [np.zeros((n, s)) for s in self.sizes]
        cs = part.probe_coeffs if any(_kind(s) == "probe" for s in self.specs) else {}
        for g, (W, lq) in enumerate(zip(self.omegas, self.logq)):
            fids = [f for f, gg in enumerate(self.group) if gg == g]
            if not fids:
                continue
            D = len(W)
            w2 = (W * W).sum(axis=1)
            step = max(1, CHUNK // max(A, 1))
            for lo in range(0, D, step):
                hi = min(lo + step, D)
                phase = part.coords @ W[lo:hi].T                     # (A, d)
                C, S = np.cos(phase), np.sin(phase)
                for f in fids:
                    amp = self._amplitude(f, part, cs, w2[lo:hi], lq[lo:hi])   # (A, d)
                    if _kind(self.specs[f]) == "probe":
                        wf = amp                  # the probe coefficients are the weights
                    else:
                        wf = np.asarray(part.weights[:, f], float)[:, None] * amp
                    out[f][:, lo:hi] = np.add.reduceat(wf * C, starts, axis=0)
                    out[f][:, D + lo:D + hi] = np.add.reduceat(wf * S, starts, axis=0)
        return out

    def _amplitude(self, f: int, part: AtomTable, cs: Dict[str, np.ndarray], w2: np.ndarray,
                   lq: np.ndarray) -> np.ndarray:
        """``a_l(omega) sqrt(q(omega))`` for every atom and frequency, built in logarithms."""
        spec = self.specs[f]
        k = _kind(spec)
        half = 0.5 * lq[None, :]
        if k in ("gauss", "efield"):
            b = float(spec.alpha)
            out = np.exp(1.5 * math.log(2 * math.pi / b) - w2[None, :] / (2 * b) + half)
            out = np.broadcast_to(out, (len(part.coords), len(w2)))
            if k == "efield":
                out = out * np.sqrt(w2)[None, :]
            return out
        if k == "steric":
            b = float(spec.alpha) / np.asarray(part.radii, float) ** 2          # (A,)
            return np.exp(1.5 * np.log(2 * math.pi / b)[:, None]
                          - w2[None, :] / (2 * b[:, None]) + half)
        from .probes import basis_widths
        Cf = cs.get(spec.name)
        if Cf is None:
            return np.zeros((len(part.coords), len(w2)))
        bk = np.asarray(basis_widths(Cf.shape[1]), float)
        E = np.exp(1.5 * np.log(2 * math.pi / bk)[:, None] - w2[None, :] / (2 * bk[:, None])
                   + half)                                                       # (K, d)
        return Cf @ E

    def _self(self, part: AtomTable) -> np.ndarray:
        return np.asarray(self_kernels(part, self.specs), float)

    def _block(self, part: AtomTable) -> np.ndarray:
        feats = self.field_features(part)
        if self.normalize:
            d = self._self(part)
            feats = [Z / np.where(d[f] > 1e-300, np.sqrt(np.maximum(d[f], 1e-300)), np.inf)[:, None]
                     for f, Z in enumerate(feats)]
        cols = [math.sqrt(max(float(self.weights[f]), 0.0)) * Z for f, Z in enumerate(feats)]
        return np.hstack(cols) if cols else np.zeros((part.n_mol, 0))

    def transform(self, table: AtomTable, block: int = BLOCK, progress=None) -> np.ndarray:
        """The feature rows of every molecule of `table`, a block of molecules at a time."""
        out = np.zeros((table.n_mol, self.rank))
        for lo, hi, part in _blocks(table, block):
            out[lo:hi] = self._block(part)
            if progress is not None:
                progress(f"Features {hi}/{table.n_mol}", hi / max(table.n_mol, 1))
        return out

    def joint_self(self, table: AtomTable, block: int = BLOCK) -> np.ndarray:
        """The exact joint self-kernel, as :meth:`cmfqsar.lowrank.LowRank.joint_self`."""
        out = np.zeros(table.n_mol)
        for lo, hi, part in _blocks(table, block):
            d = self._self(part)
            if self.normalize:
                d = np.where(d > 1e-300, 1.0, 0.0)
            out[lo:hi] = (d.T * np.asarray(self.weights, float)[:len(self.specs)]).sum(axis=1)
        return out

    def field_kernels(self, P: AtomTable, Q: Optional[AtomTable] = None,
                      normalize: Optional[bool] = None) -> np.ndarray:
        """The estimated field kernels, (F, nP, nQ), with or without the cosine normalisation
        by the exact self-kernels.  For tests and for measuring the error."""
        normalize = self.normalize if normalize is None else bool(normalize)
        fp = self.field_features(P)
        fq = fp if Q is None else self.field_features(Q)
        K = np.stack([a @ b.T for a, b in zip(fp, fq)])
        if normalize:
            dp = self._self(P)
            dq = dp if Q is None else self._self(Q)
            K = normalize_kernels(K, dp, dq)
        return K


def _blocks(table: AtomTable, block: int):
    n = table.n_mol
    step = max(int(block), 1)
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        yield lo, hi, (table if (lo == 0 and hi == n) else table.subset(np.arange(lo, hi)))


def build(fieldset: FieldSet, table: AtomTable, n_frequencies: int = 512,
          sampler: str = "orthogonal", seed: Optional[int] = 0,
          scale: Optional[Sequence[float]] = None, normalize: bool = True,
          period: Optional[float] = None, tol: float = LATTICE_TOL) -> FourierFeatures:
    """The feature map of a field set.

    `table` is read for two things only: the smallest atom radius, which sets the density of the
    steric field's frequencies, and, for the lattice, the extent of the frame that sets the
    default `period`.  The frequencies themselves do not depend on any molecule.
    `n_frequencies` is per field (each gives two feature columns); the lattice ignores it and
    takes its size from `period` (by default :func:`default_period` of each field) and from `tol`,
    the share of its peak at which a spectrum is cut.
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; use one of {SAMPLERS}")
    specs = [s for s in fieldset.specs if s.enabled]
    radii = np.asarray(table.radii, float)
    keys: Dict[tuple, int] = {}
    omegas: List[np.ndarray] = []
    logq: List[np.ndarray] = []
    group: List[int] = []
    for s in specs:
        var = proposal_variance(s, radii)
        # the gradient field of the electric-field kind has the spectrum |omega|^2 times the
        # ordinary one and an overlap with a polynomial factor, so both fall more slowly; its
        # lattice is cut at the square of the tolerance and is a group of its own
        grad = _kind(s) == "efield" and sampler == "lattice"
        key = (round(var, 12), grad)
        if key not in keys:
            keys[key] = len(omegas)
            if sampler == "lattice":
                t = tol * tol if grad else tol
                P = default_period(table, s, t) if period is None else float(period)
                W, lq = lattice_frequencies(P, 2.0 * var, t)
            else:
                # one seed per density, so that two field sets with the same widths share draws
                W, lq = frequencies(n_frequencies, math.sqrt(var), sampler,
                                    None if seed is None else int(seed) + len(omegas))
            omegas.append(W)
            logq.append(lq)
        group.append(keys[key])
    h = np.asarray(fieldset.mixing_vector(), float)
    sc = np.ones(len(specs)) if scale is None else np.asarray(scale, float)
    return FourierFeatures(specs=specs, omegas=omegas, logq=logq, group=group,
                           weights=h[:len(specs)] * sc[:len(specs)], normalize=normalize,
                           sampler=sampler)
