"""Fitting the atomic increments to the data instead of taking them from the table (item D5).

The hydrophobic field and the two hydrogen-bond fields are built from one-atom increments: a
number per (Sybyl atom type, number of attached hydrogens), taken from the regressions of the 2013
scripts (:mod:`cmfqsar.increments`).  Those numbers were fitted to logP and to Abraham's solute
constants, not to the activity being modelled, and the program's other parameterisation ("rdkit")
gives a q2 that differs by 0.20 on one Sutherland set.  That difference says the increments are
not settled.  This module fits them to the data and asks whether what it learns is a property of
the atom types or of the series.

How the fitting is possible
---------------------------
For a Gaussian field the kernel is a double sum over atom pairs::

    K(A, B) = sum over i in A, j in B of  w_i w_j E(|r_i - r_j|)

with E the overlap integral of two Gaussians, which depends on the geometry and the attenuation
alpha but not on the weights.  An increment gives every atom of one type the same weight, so with
`c_t` the increment of type `t`::

    K(A, B) = sum over types t, u of  c_t c_u M_tu(A, B)
    M_tu(A, B) = sum over i in A of type t, j in B of type u of  E(|r_i - r_j|)

The kernel is a quadratic form in the increments, and `M` does not depend on them.  So `M` is
built once per attenuation, every candidate kernel is one small contraction, and the derivative of
anything computed from the kernel can be carried back to the increments exactly.  That is what
makes a gradient method affordable here.

What is optimised
-----------------
The loss is the k-fold cross-validated squared error of kernel ridge regression -- the number the
model is selected on -- plus a penalty on the distance from the published increments::

    L(c) = sum over folds of || y_test - K_test,train (K_train + lam I)^-1 y_train ||^2
           + rho * || c - c_published ||^2

The gradient is analytic through the fold solves, the field scaling and the cosine normalisation.
One warning about the parameterisation: a cosine-normalised kernel does not change when a field's
increments are all multiplied by the same number, so only the *relative* increments of a field are
identifiable.  The fitted vectors are therefore rescaled to the length of the published ones
before they are reported or compared, and the penalty is applied to the rescaled vector.

Usage
-----
    from cmfqsar.learnincrements import fit_increments, increment_types

    fit = fit_increments(mols, names, y, alpha=0.5, lam=0.1)
    fs.increments = fit.increments        # a field set now builds the fields with them
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

from .fields import AtomTable, FieldSet, MoleculeFields, _hydrogen_counts, build_fields
from .increments import tripos_increments
from .atomtypes import assign_atom_types

#: the three fields built from increments; the others (steric, electrostatic) have no increments
INCREMENT_FIELDS = ("hydrophobic", "hba", "hbd")
#: the increment of `tripos_increments` each of them reads
INCREMENT_KEY = {"hydrophobic": "hydrophobic", "hba": "hba", "hbd": "hbd"}
#: types kept apart at most.  The tensor below costs (molecules x types)^2 numbers, so a
#: vocabulary that is too large has to be cut; on the eight Sutherland sets, which carry 21 to 29
#: types, nothing is cut and the largest tensor is 380 MB.
MAX_TYPES = 64
#: a type carried by fewer atoms than this is merged into the "other" group.  One by default,
#: which merges nothing: a type on two atoms is a free parameter fitted to almost nothing, but
#: the penalty on the distance from the published value is what holds it there, and merging it
#: away would change the kernel of the published increments as well.
MIN_ATOMS = 1
#: the name of that group
OTHER = "other"
#: strength of the penalty on the distance from the published increments
RHO = 1.0


def increment_key(syb: str, n_h: int) -> str:
    """The key the increments are stored under: the Sybyl type and the hydrogen count."""
    return f"{syb}/{int(n_h)}"


def increment_types(mol: Chem.Mol, legacy: bool = False) -> List[str]:
    """The increment key of every atom of a molecule; hydrogens get an empty key."""
    syb = assign_atom_types(mol, "sybyl")
    nh = _hydrogen_counts(mol, legacy=legacy)
    out = []
    for i, a in enumerate(mol.GetAtoms()):
        out.append("" if a.GetAtomicNum() == 1 else increment_key(syb[i], int(nh[i])))
    return out


def published_increments(keys: Sequence[str]) -> np.ndarray:
    """The published increment of every key, as (len(keys), 3) in the order of INCREMENT_FIELDS."""
    out = np.zeros((len(keys), len(INCREMENT_FIELDS)))
    for i, key in enumerate(keys):
        if key == OTHER or "/" not in key:
            continue
        syb, nh = key.rsplit("/", 1)
        inc = tripos_increments(syb, int(nh))
        for f, name in enumerate(INCREMENT_FIELDS):
            out[i, f] = inc[INCREMENT_KEY[name]]
    return out


def types_in(mols: Sequence[Chem.Mol], min_atoms: int = MIN_ATOMS, max_types: int = MAX_TYPES,
             legacy: bool = False) -> Tuple[List[str], Dict[str, int]]:
    """The increment types worth fitting in a series, and how many atoms carry each.

    A type on too few atoms cannot be fitted and would only add a free parameter, so the rare ones
    are merged into one group called "other", whose increment is fitted like any other.  The
    published value of that group is zero, which is what a type the table does not know would get.
    """
    counts: Dict[str, int] = {}
    for m in mols:
        for key in increment_types(m, legacy=legacy):
            if key:
                counts[key] = counts.get(key, 0) + 1
    kept = [k for k, n in counts.items() if n >= min_atoms]
    kept.sort(key=lambda k: -counts[k])
    if len(kept) > max_types:
        kept = kept[:max_types]
    kept.sort()
    if len(kept) < len(counts):
        kept.append(OTHER)
        counts[OTHER] = sum(n for k, n in counts.items() if k not in kept)
    return kept, counts


def type_index(mols: Sequence[Chem.Mol], keys: Sequence[str],
               legacy: bool = False) -> List[np.ndarray]:
    """For every molecule, the index into `keys` of each atom, or -1 for an atom with no increment.

    An atom whose own type was merged away lands in the "other" group when there is one.
    """
    where = {k: i for i, k in enumerate(keys)}
    other = where.get(OTHER, -1)
    out = []
    for m in mols:
        idx = np.full(m.GetNumAtoms(), -1, int)
        for i, key in enumerate(increment_types(m, legacy=legacy)):
            if key:
                idx[i] = where.get(key, other)
        out.append(idx)
    return out


# ------------------------------------------------------------------- the quadratic-form basis


@dataclass
class IncrementBasis:
    """What a series' increment kernels are built from, at one attenuation.

    `M` has shape (n_mol, T, n_mol, T): the overlap integral summed over the atom pairs of two
    molecules whose types are t and u.  Any increment vector `c` gives the field kernel
    ``K = einsum('t,u,mtnu->mn', c, c, M)``, so every candidate costs one contraction instead of a
    pass over the atom pairs.
    """
    keys: List[str]
    alpha: float
    M: np.ndarray                      # (n_mol, T, n_mol, T), float32 to keep it in memory
    n_mol: int
    counts: Dict[str, int] = _field(default_factory=dict)

    @property
    def n_types(self) -> int:
        return len(self.keys)

    @property
    def megabytes(self) -> float:
        return self.M.nbytes / 1e6

    def _over_u(self, c: np.ndarray) -> np.ndarray:
        """``sum_u c_u M[m, t, n, u]``, as (n_mol, T, n_mol).

        `tensordot` on the last axis is a matrix product, so this runs through BLAS; the same
        contraction written with `einsum` is ten times slower, which matters because the
        optimiser asks for it a few hundred times.  The increments are cast to the tensor's own
        type for the same reason.
        """
        return np.tensordot(self.M, np.asarray(c, self.M.dtype), axes=([3], [0]))

    def kernel(self, c: np.ndarray) -> np.ndarray:
        """The field kernel of one increment vector, (n_mol, n_mol)."""
        half = self._over_u(c)
        return np.tensordot(half, np.asarray(c, self.M.dtype), axes=([1], [0])).astype(float)

    def half(self, c: np.ndarray) -> np.ndarray:
        """``S[t][m, n] = sum_u c_u M[m, t, n, u]``, the half-contraction the gradient needs."""
        return np.ascontiguousarray(np.transpose(self._over_u(c), (1, 0, 2))).astype(float)


def basis_from(mols: Sequence[MoleculeFields], index: Sequence[np.ndarray], alpha: float,
               keys: Sequence[str], counts: Optional[Dict[str, int]] = None,
               dtype=np.float32) -> IncrementBasis:
    """Build the (molecule, type) overlap tensor of a series at one attenuation.

    `mols` are the built fields (only their coordinates are used) and `index` the per-atom type
    index of :func:`type_index`, in the same order.
    """
    T = len(keys)
    n = len(mols)
    coords = np.concatenate([np.asarray(m.coords, float) for m in mols], axis=0)
    group = []
    for mol_i, (m, idx) in enumerate(zip(mols, index)):
        if len(idx) != m.n_atoms:
            raise ValueError(f"molecule {mol_i} has {m.n_atoms} atoms and {len(idx)} type indices")
        group.append(np.where(idx >= 0, mol_i * T + idx, -1))
    group = np.concatenate(group)
    keep = group >= 0
    coords, group = coords[keep], group[keep]
    G = n * T
    M = np.zeros((G, G), dtype)
    # the overlap integral of two s Gaussians of the same width: (pi/alpha)^1.5 exp(-alpha d^2/4)
    pref = (np.pi / alpha) ** 1.5
    # the atoms are summed into their (molecule, type) group with a sparse indicator matrix, so
    # that a block of atom pairs costs one sparse product on each side instead of a scatter
    from scipy import sparse
    N = len(coords)
    A = sparse.csr_matrix((np.ones(N), (group, np.arange(N))), shape=(G, N))
    sq = np.sum(coords * coords, axis=1)
    block = 256
    for start in range(0, N, block):
        stop = min(start + block, N)
        d2 = sq[start:stop, None] + sq[None, :] - 2.0 * (coords[start:stop] @ coords.T)
        np.maximum(d2, 0.0, out=d2)
        E = pref * np.exp(-0.25 * alpha * d2)
        cols = A.dot(E.T).T                                  # (block, G): columns summed
        M += np.asarray(A[:, start:stop].dot(cols), dtype)   # rows summed, into (G, G)
    return IncrementBasis(keys=list(keys), alpha=float(alpha),
                          M=M.reshape(n, T, n, T), n_mol=n, counts=dict(counts or {}))


def build_basis(mols: Sequence[Chem.Mol], names: Sequence[str], alpha: float,
                keys: Optional[Sequence[str]] = None, fieldset: Optional[FieldSet] = None,
                legacy: bool = False, dtype=np.float32) -> IncrementBasis:
    """The basis of a series given the molecules themselves."""
    if keys is None:
        keys, counts = types_in(mols, legacy=legacy)
    else:
        keys, counts = list(keys), {}
    fs = fieldset or FieldSet.create(physchem=["steric"], alpha=alpha)
    built = [build_fields(m, fs, nm, _single=True) for m, nm in zip(mols, names)]
    return basis_from(built, type_index(mols, keys, legacy=legacy), alpha, keys, counts,
                      dtype=dtype)


# ---------------------------------------------------------------- the loss and its gradient


def normalise(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine-normalise a kernel, returning it and the square roots of its diagonal."""
    d = np.sqrt(np.maximum(np.diag(K), 0.0))
    safe = np.where(d > 1e-12, d, 1.0)
    return K / np.outer(safe, safe), safe


def folds_of(n: int, k: int, seed: int = 0) -> List[np.ndarray]:
    """The folds the model's own cross-validation uses, so the loss is the number it reports."""
    from .model import _kfold_indices
    return [np.asarray(te, int) for te in _kfold_indices(n, max(2, min(k, n)), seed)]


def cv_loss_and_grad(K: np.ndarray, y: np.ndarray, lam: float,
                     folds: Sequence[np.ndarray],
                     fit_bias: bool = True) -> Tuple[float, np.ndarray]:
    """The cross-validated squared error of kernel ridge regression, and its gradient in K.

    The learner is the one the model uses: kernel ridge regression with the bias term of Eq. 12
    of Baskin and Zhokhova (2013), which solves the bordered system

        [ K + lam I   1 ] [ a ]   [ y ]
        [ 1^T         0 ] [ b ] = [ 0 ]

    so the fitted increments are fitted against the model as it is, not against a simpler one.

    Returns (loss, G) with ``G[i, j] = dL / dK[i, j]``, symmetric, so that the change of the loss
    under a symmetric change of the kernel is the sum of G times that change.
    """
    n = len(y)
    G = np.zeros((n, n))
    loss = 0.0
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        m = len(tr)
        A = K[np.ix_(tr, tr)] + lam * np.eye(m)
        Kte = K[np.ix_(te, tr)]
        if fit_bias:
            # the bordered system, and the test block gets the column of ones with it
            B = np.zeros((m + 1, m + 1))
            B[:m, :m] = A
            B[:m, m] = 1.0
            B[m, :m] = 1.0
            z = np.concatenate([y[tr], [0.0]])
            u = np.linalg.solve(B, z)
            a = u[:m]
            P = np.hstack([Kte, np.ones((len(te), 1))])
            r = P @ u - y[te]
            v = np.linalg.solve(B, P.T @ r)[:m]
        else:
            u = np.linalg.solve(A, y[tr])
            a = u
            r = Kte @ a - y[te]
            v = np.linalg.solve(A, Kte.T @ r)
        loss += float(r @ r)
        # dL/dK_te,tr = 2 r a^T, kept symmetric as r a^T on each side of the diagonal;
        # dL/dK_tr,tr = -(v a^T + a v^T)
        outer = np.outer(r, a)
        G[np.ix_(te, tr)] += outer
        G[np.ix_(tr, te)] += outer.T
        inner = np.outer(v, a)
        G[np.ix_(tr, tr)] -= (inner + inner.T)
    return loss, 0.5 * (G + G.T)


def _normalised_gradient(G: np.ndarray, Kraw: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Carry a gradient in the cosine-normalised kernel back to the raw one.

    With ``Kn[m, n] = Kraw[m, n] / (d_m d_n)`` and ``d_m = sqrt(Kraw[m, m])``::

        dL/dKraw[m, n] = G[m, n] / (d_m d_n)   for m != n
        dL/dKraw[m, m] also collects  -sum_n G[m, n] Kn[m, n] / Kraw[m, m]

    The second term appears twice, once for the row and once for the column, and both give the
    same number because G and the normalised kernel are symmetric.
    """
    out = G / np.outer(d, d)
    Kn = Kraw / np.outer(d, d)
    diag = -np.sum(G * Kn, axis=1) / np.maximum(d ** 2, 1e-30)
    out[np.diag_indices_from(out)] += diag
    return out


@dataclass
class IncrementFit:
    """What one fitting produced."""
    keys: List[str]
    values: np.ndarray                 # (T, 3), in the order of INCREMENT_FIELDS
    published: np.ndarray              # (T, 3)
    alpha: float
    lam: float
    rho: float
    loss: float                        # the cross-validated squared error at the fitted values
    loss_published: float              # the same number for the published increments
    iterations: int
    message: str = ""
    counts: Dict[str, int] = _field(default_factory=dict)

    @property
    def increments(self) -> Dict[str, Dict[str, float]]:
        """The fitted increments as a table a field set can use."""
        out = {}
        for i, key in enumerate(self.keys):
            out[key] = {name: float(self.values[i, f])
                        for f, name in enumerate(INCREMENT_FIELDS)}
        return out

    def cosine(self) -> Dict[str, float]:
        """How much each fitted field still looks like the published one, in [-1, 1]."""
        out = {}
        for f, name in enumerate(INCREMENT_FIELDS):
            a, b = self.values[:, f], self.published[:, f]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            out[name] = float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0
        return out

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"keys": self.keys, "fields": list(INCREMENT_FIELDS),
                       "values": self.values.tolist(), "alpha": self.alpha, "lam": self.lam,
                       "rho": self.rho, "loss": self.loss,
                       "loss_published": self.loss_published,
                       "cosine": self.cosine(), "counts": self.counts}, fh, indent=1)


def read_increments(path: str) -> Dict[str, Dict[str, float]]:
    """Read a table written by :meth:`IncrementFit.to_json` (or any table of the same shape)."""
    with open(path) as fh:
        d = json.load(fh)
    if "values" in d and "keys" in d:
        fields = d.get("fields", list(INCREMENT_FIELDS))
        return {k: {name: float(v) for name, v in zip(fields, row)}
                for k, row in zip(d["keys"], d["values"])}
    return {k: {name: float(x) for name, x in v.items()} for k, v in d.items()}


def _rescale(values: np.ndarray, published: np.ndarray) -> np.ndarray:
    """Give every fitted field the length of the published one, and the same overall sign.

    A cosine-normalised kernel is unchanged when a field's increments are multiplied by any
    non-zero number, so this fixes a direction that the data cannot determine; it changes no
    kernel and makes the numbers comparable with the table.
    """
    out = np.array(values, float)
    for f in range(out.shape[1]):
        na = np.linalg.norm(out[:, f])
        nb = np.linalg.norm(published[:, f])
        if na < 1e-12 or nb < 1e-12:
            continue
        sign = 1.0 if out[:, f] @ published[:, f] >= 0 else -1.0
        out[:, f] = out[:, f] * (sign * nb / na)
    return out


def fit_from_basis(bases: Sequence[IncrementBasis], other: Optional[np.ndarray], y: np.ndarray,
                   lam: float = 0.1, rho: float = RHO, folds: int = 10, seed: int = 0,
                   start: Optional[np.ndarray] = None, max_iter: int = 200,
                   mixing: Optional[Sequence[float]] = None,
                   use: Optional[Sequence[int]] = None) -> IncrementFit:
    """Fit the increments of the three fields on a prepared basis.

    `bases` holds one :class:`IncrementBasis` per increment field, in the order of
    INCREMENT_FIELDS, all with the same types.  `other` is the summed, already normalised kernel
    of the fields that have no increments (steric, electrostatic), or None.  The kernels are
    cosine-normalised and added with `mixing` (ones by default), which is what the model does.

    `use` fits on a subset of the molecules of the basis -- the training set, when the basis was
    built over a whole series -- and `y` then holds that subset's activities.
    """
    from scipy.optimize import minimize
    keys = list(bases[0].keys)
    T = len(keys)
    F = len(bases)
    pub = published_increments(keys)
    c0 = np.array(start if start is not None else pub, float).reshape(T, F)
    n = len(y)
    idx = None if use is None else np.asarray(use, int)
    if idx is not None and len(idx) != n:
        raise ValueError(f"{len(idx)} molecules selected and {n} activities")
    parts = folds_of(n, folds, seed)
    if other is None:
        base = np.zeros((n, n))
    else:
        base = np.asarray(other, float)
        if idx is not None:
            base = base[np.ix_(idx, idx)]
    h = np.ones(F) if mixing is None else np.asarray(mixing, float)

    def take(K):
        return K if idx is None else K[np.ix_(idx, idx)]

    def value_and_grad(flat):
        # the increments are optimised as they are: the penalty, which is not scale invariant,
        # is what fixes the length the kernel cannot see.  The rescaling to the published length
        # happens once at the end and changes no kernel.
        c = flat.reshape(T, F)
        K = base.copy()
        raw, dsq = [], []
        for f in range(F):
            Kf = take(bases[f].kernel(c[:, f]))
            Kn, d = normalise(Kf)
            K += h[f] * Kn
            raw.append(Kf)
            dsq.append(d)
        loss, G = cv_loss_and_grad(K, y, lam, parts)
        grad = np.zeros((T, F))
        for f in range(F):
            Graw = _normalised_gradient(h[f] * G, raw[f], dsq[f])
            S = bases[f].half(c[:, f])                 # (T, n, n) over the whole basis
            if idx is not None:
                S = S[:, idx][:, :, idx]
            grad[:, f] = 2.0 * np.tensordot(S, Graw, axes=([1, 2], [0, 1]))
        pen = float(rho * np.sum((c - pub) ** 2))
        grad += 2.0 * rho * (c - pub)
        return loss + pen, grad.ravel()

    def cv_only(c):
        K = base.copy()
        for f in range(F):
            K += h[f] * normalise(take(bases[f].kernel(c[:, f])))[0]
        return float(cv_loss_and_grad(K, y, lam, parts)[0])

    base_loss = cv_only(pub)
    res = minimize(value_and_grad, c0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": int(max_iter)})
    raw = np.asarray(res.x, float).reshape(T, F)
    values = _rescale(raw, pub)
    return IncrementFit(keys=keys, values=values, published=pub, alpha=bases[0].alpha, lam=lam,
                        rho=rho, loss=cv_only(raw), loss_published=base_loss,
                        iterations=int(res.nit), message=str(res.message),
                        counts=dict(bases[0].counts))


def fit_increments(mols: Sequence[Chem.Mol], names: Sequence[str], y: np.ndarray,
                   alpha: float = 0.5, lam: float = 0.1, rho: float = RHO, folds: int = 10,
                   seed: int = 0, keys: Optional[Sequence[str]] = None,
                   fieldset: Optional[FieldSet] = None, legacy: bool = False,
                   max_iter: int = 200) -> IncrementFit:
    """Fit the increments of one series, including the fields that have no increments.

    The other fields of `fieldset` (steric, electrostatic and any others) are built once and
    enter the kernel as a constant, so what is fitted is only the three increment fields against
    the rest of the model as it stands.
    """
    fs = fieldset or FieldSet.create(alpha=alpha, param_scheme="tripos")
    if keys is None:
        keys, counts = types_in(mols, legacy=legacy)
    else:
        keys, counts = list(keys), {}
    built = [build_fields(m, fs, nm, _single=True) for m, nm in zip(mols, names)]
    index = type_index(mols, keys, legacy=legacy)
    bases, cache = [], {}
    for name in INCREMENT_FIELDS:
        spec = next((s for s in fs.specs if s.name == name), None)
        a = float(spec.alpha) if spec is not None else float(alpha)
        if a not in cache:                 # the tensor depends on the width, not on the field
            cache[a] = basis_from(built, index, a, keys, counts)
        bases.append(cache[a])
    other = other_kernel(built, fs)
    return fit_from_basis(bases, other, np.asarray(y, float), lam=lam, rho=rho, folds=folds,
                          seed=seed, max_iter=max_iter)


def other_kernel(built: Sequence[MoleculeFields], fieldset: FieldSet) -> Optional[np.ndarray]:
    """The normalised, summed kernel of the fields that carry no increments."""
    from .kernels import kernel_matrices, normalize_kernels, self_kernels
    specs = [s for s in fieldset.specs if s.name not in INCREMENT_FIELDS]
    if not specs:
        return None
    table = AtomTable.from_molecules(list(built))
    keep = [i for i, s in enumerate(fieldset.specs) if s.name not in INCREMENT_FIELDS]
    sub = AtomTable(coords=table.coords, weights=np.ascontiguousarray(table.weights[:, keep]),
                    radii=table.radii, offsets=table.offsets, names=list(table.names),
                    molecules=list(table.molecules))
    K = kernel_matrices(sub, sub, specs)
    d = self_kernels(sub, specs)
    K = normalize_kernels(K, d, d)
    h = np.array([getattr(s, "h", 1.0) for s in specs], float)
    return np.tensordot(h, K, axes=(0, 0))


def coverage(table: Dict[str, Dict[str, float]], mols: Sequence[Chem.Mol],
             legacy: bool = False) -> Dict[str, float]:
    """How much of a series an increment table covers.

    Returns the share of atoms whose own type is in the table, the share that falls back to its
    "other" entry and the share that gets nothing.  A table fitted on one series and applied to
    another is only as good as this says it is.
    """
    has_other = OTHER in table
    exact = fallback = missing = total = 0
    for m in mols:
        for key in increment_types(m, legacy=legacy):
            if not key:
                continue
            total += 1
            if key in table:
                exact += 1
            elif has_other:
                fallback += 1
            else:
                missing += 1
    if total == 0:
        return {"exact": 0.0, "other": 0.0, "missing": 0.0, "atoms": 0}
    return {"exact": exact / total, "other": fallback / total, "missing": missing / total,
            "atoms": total}


def spread(fits: Sequence[IncrementFit]) -> Dict[str, float]:
    """How much the increments fitted on different series agree with each other.

    The mean cosine between every pair of fitted vectors, per field, over the types the two fits
    have in common.  One means the series learned the same increments; zero means they learned
    unrelated ones, and the increments are then a property of the series rather than of the atom
    types.
    """
    out = {}
    for f, name in enumerate(INCREMENT_FIELDS):
        vals = []
        for i in range(len(fits)):
            for j in range(i + 1, len(fits)):
                shared = [k for k in fits[i].keys if k in fits[j].keys and k != OTHER]
                if len(shared) < 3:
                    continue
                a = np.array([fits[i].values[fits[i].keys.index(k), f] for k in shared])
                b = np.array([fits[j].values[fits[j].keys.index(k), f] for k in shared])
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-12 and nb > 1e-12:
                    vals.append(float(a @ b / (na * nb)))
        out[name] = float(np.mean(vals)) if vals else float("nan")
    return out
