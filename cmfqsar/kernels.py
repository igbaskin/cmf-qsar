"""
Molecular-field kernels.

For two atoms with Gaussian field contributions ::

    rho_l(r) = w_l exp(-b_l/2 |r - r_l|^2),   rho_m(r) = w_m exp(-b_m/2 |r - r_m|^2)

the overlap integral over R^3 is analytic::

    k(l, m) = w_l w_m (2 pi / (b_l + b_m))^{3/2} exp(-(b_l b_m) / (2 (b_l + b_m)) |r_l - r_m|^2)

For b_l = b_m = alpha this reduces to Eq. 5 of Baskin & Zhokhova (2013)::

    k = w_l w_m (pi/alpha)^{3/2} exp(-alpha/4 |r_l - r_m|^2)

and for the radius-scaled steric field (b = alpha / R_vdw^2) to Eq. 11.

The kernel of molecules i, j for field f is the sum over all atom pairs
(Eq. 2); the joint kernel is the h_f-weighted sum over fields (Eq. 1).

Three things make this fast enough to use on more than one series (plan item B1).  They were
chosen by measuring one 40-atom molecule against a 200-molecule table, 318 000 atom pairs:

* **The training kernel is computed once, not twice.**  K_f(i, j) = K_f(j, i), so for a table
  against itself only the upper triangle is evaluated and then mirrored.  This is the largest of
  the three, because it removes half of the work rather than part of it, and it is exact: the two
  triangles used to differ by about 4e-13, which was round-off from summing the atom pairs in two
  different orders.

* **The radius-scaled prefactor uses a square root, not a power.**  The steric field has a width
  per atom, so ``(2 pi / (b_l + b_m))^{3/2}`` has a different value for every atom pair, and
  ``x ** 1.5`` on 318 000 elements took 2.3 ms of the block's 9.2 ms.  ``z * sqrt(z)`` is the same
  number -- the two agree to 1.4e-14 -- and costs a third as much, because a square root is a
  machine instruction and a general power is a library call.

* **The distances are computed once per molecule, not once per alpha** (what B1 asked for).  The
  attenuation enters only through the exponential, so a grid search can share the squared
  distances: :func:`kernel_matrices_multi` takes several field sets and evaluates them all from
  one distance block.  This is the smallest of the three; the distance matrix is 3 % of the time.

A fourth shortcut, the cutoff of plan item B2, is **off unless it is worth switching on**.  A
Gaussian overlap falls below 10^-6 of its peak beyond ``sqrt(-4 ln tol / b)``, so atom pairs
further apart than that can be dropped and the sum run over a neighbour list instead of over
every pair.  Whether that saves anything depends on the molecules, and for the series this
program is usually given it does not: the molecules of a congeneric series are superimposed in
one frame about 14 A across, and at the usual alpha = 0.5 the cutoff is 10.5 A, which keeps 97 %
of the atom pairs of the electrostatic field and 100 % of the steric one.  On 60 decapeptides
spread over 33 A the same cutoff keeps 62 %, and at alpha = 2 it keeps 14 %, where the neighbour
list is 2.7 times faster than the dense block.  :func:`atom_kernel_block` therefore takes
``cutoff="auto"``, which compares the cutoff sphere with the bounding box of the molecules and
uses the neighbour list only when it expects to drop most of the pairs.  The dense path stays as
the exact reference, and ``tests/test_kernel_speed.py`` holds the two to the tolerance the cutoff
was derived from.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .fields import AtomTable, FieldSpec


class Cancelled(Exception):
    pass


def use_compiled(engine=None) -> bool:
    """Whether this call should use the compiled loop.

    `engine` of None means the module default :data:`ENGINE`.  "numba" asks for it explicitly and
    raises when Numba is not installed, which is what a benchmark wants; "auto" falls back.
    """
    from . import fastkernel

    e = ENGINE if engine is None else str(engine)
    if e not in ENGINES:
        raise ValueError(f"unknown kernel engine {e!r}; use one of {ENGINES}")
    if e == "numpy":
        return False
    if e == "numba" and not fastkernel.available():
        raise ValueError("the 'numba' kernel engine needs Numba installed: pip install numba "
                         "(or pip install -e .[fast]); the default 'numpy' engine needs nothing")
    return fastkernel.available()


def _compiled_cutoff(specs: Sequence[FieldSpec], cutoff, radii_max: float):
    """The cutoff radius to give the compiled loop, which has no use for "auto".

    Inside the loop a cutoff is one comparison per atom pair and needs no neighbour list, so
    there is nothing to decide; "auto" and True become the largest radius any field group would
    have used, which can only drop pairs that every group agrees are negligible.
    """
    if cutoff is None or cutoff is False:
        return None
    if isinstance(cutoff, (int, float)) and cutoff is not True:
        return float(cutoff)
    gauss, steric, _probe = _group_fields(specs)
    radii = []
    for alpha in gauss:
        radii.append(cutoff_radius(0.25 * alpha))
    for _f, alpha in steric:
        radii.append(cutoff_radius(0.25 * alpha / max(radii_max, 1e-6) ** 2))
    return max(radii) if radii else None


def _group_fields(specs: Sequence[FieldSpec]):
    """Fields by the shape of their Gaussians: shared width, per-atom width, or an expansion."""
    gauss = {}
    steric = []
    probe = []
    for f, s in enumerate(specs):
        if getattr(s, "is_probe", False):
            probe.append((f, s.name))
        elif s.kind == "gauss":
            gauss.setdefault(float(s.alpha), []).append(f)
        else:
            steric.append((f, float(s.alpha)))
    return gauss, steric, probe


#: which implementation of the atom-pair sum to use (plan item B5).  "numpy" is the default and
#: the reference: every table in this repository was produced with it.  "numba" is a compiled,
#: threaded loop (:mod:`cmfqsar.fastkernel`) which is six to ten times faster on a many-core
#: machine and differs from the NumPy path by about 4e-15 relative, from summing in another order.
#: "auto" uses it when Numba is installed and falls back silently when it is not.
ENGINE = "numpy"
ENGINES = ("numpy", "numba", "auto")

#: how far below its peak a Gaussian overlap has to fall before the pair is dropped (item B2)
CUTOFF_TOL = 1e-6
#: the neighbour list is used only when this small a share of the atom pairs survives the cutoff.
#: The value is measured, not chosen: on 200 superimposed COX2 inhibitors a cutoff that keeps 24 %
#: of the pairs makes the block 1.4 times *slower*, because scipy's sparse_distance_matrix costs
#: more per surviving pair than the dense exponential does per pair; at 4 % it is 2.6 times faster.
#: at 4 % it is 2.6 times faster.  The share that pays depends on how big the block is, because
#: the query has a fixed cost: a small block has to drop almost everything, a large one only most.
CUTOFF_SHARE = 0.10
CUTOFF_SHARE_LARGE = 0.30
#: where "large" is, in atom pairs per block
CUTOFF_LARGE_BLOCK = 1_000_000
#: and below this a block is never worth a neighbour list at all
CUTOFF_MIN_PAIRS = 200_000


def cutoff_radius(coefficient: float, tol: float = CUTOFF_TOL) -> float:
    """The distance beyond which ``exp(-coefficient * d^2)`` is below `tol` of its peak.

    `coefficient` is what multiplies the squared distance in the exponent: ``alpha / 4`` for a
    field whose Gaussians share one width, and ``b_min / 4`` for the radius-scaled steric field,
    where ``b_min = alpha / R_max^2`` is the widest Gaussian present -- the widest one decides,
    because it is the one that still contributes at long range.
    """
    c = max(float(coefficient), 1e-300)
    return float(np.sqrt(-np.log(max(float(tol), 1e-300)) / c))


def surviving_share(xi: np.ndarray, coords: np.ndarray, radius: float, sample: int = 16,
                    seed: int = 0) -> float:
    """The share of atom pairs closer than `radius`, estimated from a sample of them.

    A geometric estimate from the bounding box is not good enough here, because the molecules of a
    series are superimposed and their atoms are concentrated in the middle of the box rather than
    spread through it, so the volume ratio underestimates the share by a factor of three or more.
    A sample of at most ``sample`` x 8 ``sample`` real distances answers the question directly.
    The default is 16 x 128 = 2048 pairs, which estimates a share of 0.2 to within 0.01 and costs
    about a hundredth of what the block costs; the first version sampled nine times as many pairs
    and that showed up as a 12 % overhead on the sets where the answer is "do not bother".
    """
    ni, nq = xi.shape[0], coords.shape[0]
    if ni == 0 or nq == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    ri = xi if ni <= sample else xi[rng.choice(ni, sample, replace=False)]
    m = 8 * sample
    rq = coords if nq <= m else coords[rng.choice(nq, m, replace=False)]
    d2 = cdist(ri, rq, "sqeuclidean")
    return float(np.mean(d2 <= radius * radius))


def _worth_a_neighbour_list(radius: float, xi: np.ndarray, coords: np.ndarray) -> bool:
    """Whether a cutoff of `radius` would drop enough of the pairs to pay for the neighbour list.

    Two conditions, both measured rather than assumed: most of the pairs have to be dropped
    (:data:`CUTOFF_SHARE`), and the block has to be large enough for the query's fixed cost to be
    worth paying (:data:`CUTOFF_MIN_PAIRS`).
    """
    pairs = xi.shape[0] * coords.shape[0]
    if coords.shape[0] < 2 or pairs < CUTOFF_MIN_PAIRS:
        return False
    share = CUTOFF_SHARE_LARGE if pairs >= CUTOFF_LARGE_BLOCK else CUTOFF_SHARE
    return surviving_share(xi, coords, radius) < share


def _neighbour_pairs(xi: np.ndarray, coords: np.ndarray, radius: float):
    """(row, col, squared distance) for every atom pair closer than `radius`.

    Pairs at distance exactly zero are kept, which matters because an atom's overlap with itself
    is the largest term of all and a molecule's self-kernel is made of them.
    """
    # the unbalanced build is about twice as cheap as the default and the query is what costs
    small = cKDTree(xi, compact_nodes=False, balanced_tree=False)
    big = cKDTree(coords, compact_nodes=False, balanced_tree=False)
    coo = small.sparse_distance_matrix(big, float(radius), output_type="coo_matrix")
    return coo.row, coo.col, coo.data ** 2


def _pow32(z: np.ndarray) -> np.ndarray:
    """``z ** 1.5`` as ``z * sqrt(z)``, which is the same number and three times faster.

    A general power is a library call per element; a square root is one machine instruction.  On
    318 000 atom pairs the two agree to 1.4e-14 and the second costs 1.4 ms instead of 4.3 ms,
    which matters because the radius-scaled steric field evaluates this for every atom pair.
    """
    return z * np.sqrt(z)


def pair_distances(xi: np.ndarray, Q: AtomTable, first_mol: int = 0) -> np.ndarray:
    """Squared distances from the atoms of one molecule to the atoms of table Q.

    The only part of a field kernel that does not depend on the attenuation, which is why
    :func:`kernel_matrices_multi` computes it once for a whole grid of alphas.  `first_mol`
    restricts the columns to the molecules from that index onwards.
    """
    return cdist(xi, Q.coords[int(Q.offsets[first_mol]):], "sqeuclidean")


def atom_kernel_block(xi: np.ndarray, wi: np.ndarray, ri: np.ndarray, Q: AtomTable,
                      specs: Sequence[FieldSpec], ci=None, cq=None, first_mol: int = 0,
                      D2: Optional[np.ndarray] = None, cutoff=None,
                      tol: float = CUTOFF_TOL) -> np.ndarray:
    """Kernels between the atoms of one molecule and every molecule of table Q.

    Returns an array of shape (F, Ni, nQ): the contribution of every atom of
    the probe molecule to the field kernels with every molecule in Q.

    `first_mol` restricts the columns to the molecules of Q from that index onwards, which is what
    the symmetric case needs: for a table against itself only molecules j >= i are evaluated.
    `D2` is a squared-distance block from :func:`pair_distances` for the same `first_mol`; passing
    it in saves recomputing the distances for every attenuation of a grid.

    `cutoff` controls the neighbour list of plan item B2.  None or False computes every atom pair,
    which is exact and is the reference the tests compare against.  ``"auto"`` uses a neighbour
    list for each group of fields whose cutoff radius is small against the size of the molecules,
    True uses one whatever the geometry, and a number is used as the radius itself.  A cutoff
    makes the kernel an approximation: every dropped pair contributed less than `tol` of a peak
    overlap, and the measured error on a 60-molecule table was 4e-7 relative.  A cutoff cannot be
    combined with a precomputed `D2`, which is a dense block by construction.
    """
    F = len(specs)
    Ni = len(xi)
    a0 = int(Q.offsets[first_mol])
    nQ = Q.n_mol - first_mol
    out = np.zeros((F, Ni, nQ))
    gauss, steric, probe = _group_fields(specs)
    wq = Q.weights[a0:]
    starts = Q.offsets[first_mol:-1] - a0
    coords = Q.coords[a0:]
    want_cutoff = cutoff is not None and cutoff is not False and D2 is None
    counts = np.diff(Q.offsets[first_mol:])
    mol_of = np.repeat(np.arange(nQ), counts) if want_cutoff else None

    def dense():
        return D2 if D2 is not None else pair_distances(xi, Q, first_mol)

    def use_list(radius: float) -> bool:
        if not want_cutoff:
            return False
        if isinstance(cutoff, str):
            return _worth_a_neighbour_list(radius, xi, coords)
        return True

    def accumulate(row, col, values):
        """Sum one field's atom-pair contributions into (atom of i, molecule of Q)."""
        key = row * nQ + mol_of[col]
        return np.bincount(key, weights=values, minlength=Ni * nQ).reshape(Ni, nQ)

    dense_D2 = None
    for alpha, fids in gauss.items():
        radius = float(cutoff) if isinstance(cutoff, (int, float)) and cutoff is not True \
            else cutoff_radius(0.25 * alpha, tol)
        if use_list(radius):
            row, col, d2 = _neighbour_pairs(xi, coords, radius)
            E = np.exp(-0.25 * alpha * d2) * (np.pi / alpha) ** 1.5
            for f in fids:
                out[f] = accumulate(row, col, wi[row, f] * E * wq[col, f])
            continue
        dense_D2 = dense() if dense_D2 is None else dense_D2
        E = np.exp(-0.25 * alpha * dense_D2)
        E *= (np.pi / alpha) ** 1.5
        for f in fids:
            M = (wi[:, f][:, None] * E) * wq[:, f][None, :]          # (Ni, AQ)
            out[f] = np.add.reduceat(M, starts, axis=1)
    for f, alpha in steric:
        rq = Q.radii[a0:]
        bi = alpha / ri ** 2
        bq = alpha / rq ** 2
        b_min = float(min(bi.min(), bq.min()))
        radius = float(cutoff) if isinstance(cutoff, (int, float)) and cutoff is not True \
            else cutoff_radius(0.25 * b_min, tol)
        if use_list(radius):
            row, col, d2 = _neighbour_pairs(xi, coords, radius)
            bs = bi[row] + bq[col]
            E = np.exp(-(bi[row] * bq[col]) / (2.0 * bs) * d2) * _pow32(2.0 * np.pi / bs)
            out[f] = accumulate(row, col, wi[row, f] * E * wq[col, f])
            continue
        dense_D2 = dense() if dense_D2 is None else dense_D2
        bs = bi[:, None] + bq[None, :]
        E = np.exp(-(bi[:, None] * bq[None, :]) / (2.0 * bs) * dense_D2)
        E *= _pow32(2.0 * np.pi / bs)
        M = (wi[:, f][:, None] * E) * wq[:, f][None, :]
        out[f] = np.add.reduceat(M, starts, axis=1)
    if probe:
        # Every atom of a probe field carries several Gaussians, so the overlap runs over
        # pairs of terms as well as pairs of atoms.  The widths are shared, so each pair of
        # terms is one pass over the distance matrix under the Gaussian-product rule --
        # the same formula as the steric field, with the two widths constant.
        from .probes import basis_widths
        cq = cq if cq is not None else Q.probe_coeffs
        dense_D2 = dense() if dense_D2 is None else dense_D2
        D2 = dense_D2
        for f, name in probe:
            Ci = None if ci is None else ci.get(name)
            Cq = cq.get(name)
            if Ci is None or Cq is None:
                continue
            Cq = Cq[a0:]
            widths = basis_widths(Ci.shape[1])
            M = np.zeros_like(D2)
            for k1, b1 in enumerate(widths):
                for k2, b2 in enumerate(widths):
                    bs = b1 + b2
                    E = np.exp(-(b1 * b2) / (2.0 * bs) * D2) * (2.0 * np.pi / bs) ** 1.5
                    M += (Ci[:, k1][:, None] * E) * Cq[:, k2][None, :]
            out[f] = np.add.reduceat(M, starts, axis=1)
    return out


def _mirror(K: np.ndarray) -> np.ndarray:
    """Fill the lower triangle of every field's matrix from the upper one."""
    for f in range(K.shape[0]):
        up = np.triu(K[f])
        K[f] = up + np.triu(up, 1).T
    return K


def kernel_matrices(P: AtomTable, Q: AtomTable, specs: Sequence[FieldSpec],
                    progress: Optional[Callable[[int, int], None]] = None,
                    cancel: Optional[Callable[[], bool]] = None,
                    symmetric: Optional[bool] = None, cutoff=None, engine=None) -> np.ndarray:
    """Per-field kernel matrices K_f(i, j) for molecules i in P and j in Q.

    Returns an array of shape (F, nP, nQ).

    `symmetric` evaluates only the pairs with j >= i and mirrors them, which halves the work and
    is exact because the atom-pair overlap is symmetric.  It defaults to true when Q is P -- the
    training kernel, where it applies -- and must not be set for two different tables.

    `cutoff` is passed to :func:`atom_kernel_block`: ``"auto"`` for the neighbour list of item B2
    where the geometry makes it worth using, None for the exact dense sum.

    `engine` chooses the implementation (item B5): None for the module default, "numba" for the
    compiled threaded loop, "auto" for that where it is installed.  The compiled loop cannot take
    probe fields and declines small problems, and the call then falls back to NumPy by itself.
    """
    sym = (Q is P) if symmetric is None else bool(symmetric)
    if sym and (P.n_mol != Q.n_mol or P.coords.shape != Q.coords.shape):
        raise ValueError("a symmetric kernel needs one table against itself")
    F = len(specs)
    if use_compiled(engine):
        from . import fastkernel
        got = fastkernel.kernel_matrices(
            P, Q, specs, symmetric=sym,
            cutoff=_compiled_cutoff(specs, cutoff, float(np.max(Q.radii)) if len(Q.radii) else 1.0))
        if got is not None:
            if progress is not None:
                progress(P.n_mol, P.n_mol)
            return _mirror(got) if sym else got
    K = np.zeros((F, P.n_mol, Q.n_mol))
    # concatenated once rather than per molecule: the property rebuilds them every call
    cq = Q.probe_coeffs
    cp = P.probe_coeffs if cq else {}
    for i in range(P.n_mol):
        if cancel is not None and cancel():
            raise Cancelled()
        sl = P.slice(i)
        ci = {k: v[sl] for k, v in cp.items()} if cp else None
        first = i if sym else 0
        blk = atom_kernel_block(P.coords[sl], P.weights[sl], P.radii[sl], Q, specs,
                                ci=ci, cq=cq, first_mol=first, cutoff=cutoff)
        K[:, i, first:] = blk.sum(axis=1)
        if progress is not None and (i % 10 == 0 or i == P.n_mol - 1):
            progress(i + 1, P.n_mol)
    return _mirror(K) if sym else K


def kernel_matrices_multi(P: AtomTable, Q: AtomTable, spec_sets: Sequence[Sequence[FieldSpec]],
                          progress: Optional[Callable[[int, int], None]] = None,
                          cancel: Optional[Callable[[], bool]] = None,
                          symmetric: Optional[bool] = None, engine=None) -> List[np.ndarray]:
    """The kernels of several field sets, sharing each molecule's distance block (plan item B1).

    A grid search over the attenuation computes one kernel per alpha, and the squared interatomic
    distances are the same for all of them.  This evaluates every field set of `spec_sets` from
    one distance block per molecule, and returns one (F, nP, nQ) array per field set, in order.

    The field sets must describe the same fields in the same order and differ only in their
    widths; nothing here checks that, because the caller builds them from one field set.
    """
    sym = (Q is P) if symmetric is None else bool(symmetric)
    if sym and (P.n_mol != Q.n_mol or P.coords.shape != Q.coords.shape):
        raise ValueError("a symmetric kernel needs one table against itself")
    if use_compiled(engine):
        # the compiled loop computes each distance per attenuation rather than sharing it, which
        # costs 3 % and buys the threads: on a many-core machine that trade is worth taking
        got = [kernel_matrices(P, Q, sp, symmetric=sym, engine=engine) for sp in spec_sets]
        if all(g is not None for g in got):
            return got
    outs = [np.zeros((len(specs), P.n_mol, Q.n_mol)) for specs in spec_sets]
    cq = Q.probe_coeffs
    cp = P.probe_coeffs if cq else {}
    for i in range(P.n_mol):
        if cancel is not None and cancel():
            raise Cancelled()
        sl = P.slice(i)
        ci = {k: v[sl] for k, v in cp.items()} if cp else None
        first = i if sym else 0
        D2 = pair_distances(P.coords[sl], Q, first)
        for k, specs in enumerate(spec_sets):
            blk = atom_kernel_block(P.coords[sl], P.weights[sl], P.radii[sl], Q, specs,
                                    ci=ci, cq=cq, first_mol=first, D2=D2)
            outs[k][:, i, first:] = blk.sum(axis=1)
        if progress is not None and (i % 10 == 0 or i == P.n_mol - 1):
            progress(i + 1, P.n_mol)
    return [_mirror(K) if sym else K for K in outs]


def self_kernels(P: AtomTable, specs: Sequence[FieldSpec]) -> np.ndarray:
    """K_f(i, i) for every molecule i of P. Shape (F, nP)."""
    return self_kernels_multi(P, [specs])[0]


def self_kernels_multi(P: AtomTable, spec_sets: Sequence[Sequence[FieldSpec]]) -> List[np.ndarray]:
    """``self_kernels`` for several field sets, sharing each molecule's distance block."""
    out = [np.zeros((len(specs), P.n_mol)) for specs in spec_sets]
    for i in range(P.n_mol):
        sub = P.subset([i])
        cs = sub.probe_coeffs
        D2 = pair_distances(sub.coords, sub, 0)
        for k, specs in enumerate(spec_sets):
            blk = atom_kernel_block(sub.coords, sub.weights, sub.radii, sub, specs,
                                    ci=cs or None, cq=cs or None, D2=D2)
            out[k][:, i] = blk.sum(axis=1)[:, 0]
    return out


def normalize_kernels(K: np.ndarray, dP: np.ndarray, dQ: np.ndarray) -> np.ndarray:
    """Cosine-normalised kernels K_f(i,j) / sqrt(K_f(i,i) K_f(j,j)); zero where undefined."""
    Kn = np.zeros_like(K)
    for f in range(K.shape[0]):
        sp = np.sqrt(np.maximum(dP[f], 0.0))
        sq = np.sqrt(np.maximum(dQ[f], 0.0))
        denom = sp[:, None] * sq[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            Kn[f] = np.where(denom > 1e-300, K[f] / denom, 0.0)
    return Kn


def combine_kernels(K: np.ndarray, h: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Joint kernel K = sum_f h_f * scale_f * K_f."""
    return np.tensordot(np.asarray(h, dtype=float) * np.asarray(scale, dtype=float), K, axes=(0, 0))


# --------------------------------------------------------------------------- #
# Reference implementation (numerical quadrature) used in the unit tests
# --------------------------------------------------------------------------- #

def expansion_overlap(x1, w1, b1, x2, w2, b2) -> float:
    """The overlap integral of two Gaussian expansions, in closed form.

    The expansions are given the way the rest of this program gives them: centres ``(n, 3)``,
    weights ``(n,)`` and widths ``(n,)``.  The integral is the sum of the pair overlaps at the top
    of this module, so this is the same formula :func:`atom_kernel_block` evaluates over atom
    pairs, written for two bags of Gaussians that need not be molecules at all -- a co-field, a
    mask, a masked co-field (plan item MC5).
    """
    x1 = np.atleast_2d(np.asarray(x1, float))
    x2 = np.atleast_2d(np.asarray(x2, float))
    w1 = np.asarray(w1, float).reshape(-1)
    w2 = np.asarray(w2, float).reshape(-1)
    b1 = np.asarray(b1, float).reshape(-1)
    b2 = np.asarray(b2, float).reshape(-1)
    if len(x1) == 0 or len(x2) == 0:
        return 0.0
    s = b1[:, None] + b2[None, :]
    d2 = cdist(x1, x2, "sqeuclidean")
    k = _pow32(2.0 * np.pi / s) * np.exp(-(b1[:, None] * b2[None, :]) / (2.0 * s) * d2)
    return float(w1 @ k @ w2)


def gaussian_product(x1, w1, b1, x2, w2, b2, tol: float = 0.0):
    """The Gaussian expansion of the pointwise *product* of two Gaussian expansions.

    The product of two Gaussians is a Gaussian -- the same fact that makes every overlap in this
    program closed form, used on the integrand instead of on the integral::

        exp(-b/2 |r - A|^2) exp(-c/2 |r - B|^2)
            = exp(-(b c) / (2 (b + c)) |A - B|^2) exp(-(b + c)/2 |r - C|^2),
            C = (b A + c B) / (b + c)

    So a co-field multiplied by a mask is again a sum of Gaussians, and anything that reads a
    Gaussian expansion -- the grid drawing, the cube export, :func:`expansion_overlap` -- works on
    it unchanged (plan item MC5).  The result has ``n1 * n2`` terms; `tol` drops the terms whose
    weight is below that fraction of the largest, which is worth doing because two Gaussians far
    apart make a term that is numerically zero.
    """
    x1 = np.atleast_2d(np.asarray(x1, float))
    x2 = np.atleast_2d(np.asarray(x2, float))
    w1 = np.asarray(w1, float).reshape(-1)
    w2 = np.asarray(w2, float).reshape(-1)
    b1 = np.asarray(b1, float).reshape(-1)
    b2 = np.asarray(b2, float).reshape(-1)
    if len(x1) == 0 or len(x2) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    s = b1[:, None] + b2[None, :]                                  # (n1, n2)
    d2 = cdist(x1, x2, "sqeuclidean")
    weights = (w1[:, None] * w2[None, :]
               * np.exp(-(b1[:, None] * b2[None, :]) / (2.0 * s) * d2))
    centres = ((b1[:, None, None] * x1[:, None, :] + b2[None, :, None] * x2[None, :, :])
               / s[:, :, None])
    centres = centres.reshape(-1, 3)
    weights = weights.reshape(-1)
    widths = s.reshape(-1)
    if tol > 0.0 and len(weights):
        keep = np.abs(weights) >= tol * np.max(np.abs(weights))
        centres, weights, widths = centres[keep], weights[keep], widths[keep]
    return centres, weights, widths


def numerical_overlap(x1, w1, b1, x2, w2, b2, spacing=0.1, extent=6.0) -> float:
    """Numerically integrate the product of two Gaussian atom fields (slow; for tests)."""
    lo = np.minimum(x1, x2) - extent
    hi = np.maximum(x1, x2) + extent
    axes = [np.arange(lo[k], hi[k], spacing) for k in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    f1 = w1 * np.exp(-0.5 * b1 * ((g - x1) ** 2).sum(1))
    f2 = w2 * np.exp(-0.5 * b2 * ((g - x2) ** 2).sum(1))
    return float((f1 * f2).sum() * spacing ** 3)


# --------------------------------------------------------------------------- #
# Derivatives of the field kernels with respect to the probe atom positions
# --------------------------------------------------------------------------- #

def kernel_gradient(xi: np.ndarray, wi: np.ndarray, ri: np.ndarray, Q: AtomTable,
                    specs: Sequence[FieldSpec], u: np.ndarray, chunk: int = 4000,
                    ci=None) -> np.ndarray:
    """Gradient of ``sum_f sum_j u[f, j] K_f(i, j)`` with respect to the atom positions of i.

    The kernels are sums of Gaussians in the squared interatomic distances, so the
    derivative is a weighted sum of the same Gaussians times the interatomic vectors::

        d/dr_il  w_fil w_fjm g(|r_il - r_jm|^2)  =  2 g'(d^2) w_fil w_fjm (r_il - r_jm)

    with ``g'/g = -alpha_f/4`` for ordinary fields and ``-b_il b_jm / (2 (b_il + b_jm))``
    for the radius-scaled steric field.

    Parameters
    ----------
    u : array of shape (F, nQ), the multiplier of every field/molecule pair.
    ci : the moving molecule's probe-field expansion coefficients, ``{name: (Ni, n_terms)}``,
        needed only when the field set contains probe fields.

    Returns
    -------
    array of shape (Ni, 3)
    """
    grad = np.zeros((len(xi), 3))
    u_atom = np.repeat(np.asarray(u, dtype=float), np.diff(Q.offsets), axis=1)   # (F, AQ)
    gauss, steric, probe = _group_fields(specs)
    if probe:
        from .probes import basis_widths
        cq_all = Q.probe_coeffs
        missing = [name for _f, name in probe
                   if (ci or {}).get(name) is None or cq_all.get(name) is None]
        if missing:
            raise ValueError(f"probe-field coefficients missing for {missing}; pass ci= from the "
                             f"moving molecule and build the table with probe fields")
    for s0 in range(0, len(Q.coords), chunk):
        sl = slice(s0, min(s0 + chunk, len(Q.coords)))
        diff = xi[:, None, :] - Q.coords[None, sl, :]            # (Ni, M, 3)
        D2 = np.einsum("lmk,lmk->lm", diff, diff)
        for alpha, fids in gauss.items():
            E = np.exp(-0.25 * alpha * D2) * (np.pi / alpha) ** 1.5
            for f in fids:
                M = (wi[:, f][:, None] * E) * (Q.weights[sl, f] * u_atom[f, sl])[None, :]
                grad += (-0.5 * alpha) * np.einsum("lm,lmk->lk", M, diff)
        for f, alpha in steric:
            bi = alpha / ri ** 2
            bq = alpha / Q.radii[sl] ** 2
            bs = bi[:, None] + bq[None, :]
            fac = (bi[:, None] * bq[None, :]) / bs
            E = np.exp(-0.5 * fac * D2) * (2.0 * np.pi / bs) ** 1.5
            M = (wi[:, f][:, None] * E) * (Q.weights[sl, f] * u_atom[f, sl])[None, :]
            grad += -np.einsum("lm,lmk->lk", M * fac, diff)
        for f, name in probe:
            # every atom of a probe field carries several Gaussians, so the gradient runs over
            # pairs of terms as well as pairs of atoms -- the steric formula with both widths
            # constant, summed over the term pairs
            Ci, Cq = ci[name], cq_all[name]
            widths = basis_widths(Ci.shape[1])
            for k1, b1 in enumerate(widths):
                for k2, b2 in enumerate(widths):
                    bs = b1 + b2
                    fac = (b1 * b2) / bs
                    E = np.exp(-0.5 * fac * D2) * (2.0 * np.pi / bs) ** 1.5
                    M = (Ci[:, k1][:, None] * E) * (Cq[sl, k2] * u_atom[f, sl])[None, :]
                    grad += -fac * np.einsum("lm,lmk->lk", M, diff)
    return grad


def so3_right_jacobian(omega: np.ndarray) -> np.ndarray:
    """Right Jacobian of the SO(3) exponential map (identity at omega = 0)."""
    w = np.asarray(omega, dtype=float)
    th = float(np.linalg.norm(w))
    W = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-8:
        return np.eye(3) - 0.5 * W
    a = (1.0 - np.cos(th)) / th ** 2
    b = (th - np.sin(th)) / th ** 3
    return np.eye(3) - a * W + b * (W @ W)
