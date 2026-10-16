"""Shapley decomposition of a CMF prediction.

The decomposition rules of the papers are exact but they need the model to be *linear* in the
molecular field: rule 3 splits a prediction into atoms because the field is a sum of atom-centred
Gaussians and the kernel is an overlap integral, hence additive over those atoms.  A nonlinear
kernel layer destroys that additivity, and with it the co-field and the atom contributions.

The Shapley value restores an additive attribution for any kernel.  Its usual obstacle in machine
learning is that "feature absent" has no definition, so a background sample and an imputation rule
have to be invented.  Continuous molecular fields do not have that problem: the field of a subset
of atoms is again a field, and because the kernel is bilinear its value for any subset is a sum of
entries of one precomputed array.  Every coalition value here is therefore the model's own
prediction for a genuine field, with no surrogate model and no background set:

    v(S) = sum_j a_j kappa( K(X_S, X_j), K(X_S, X_S), K(X_j, X_j) ) + b

with K(X_S, X_j) = sum_{l in S} blk[l, j] and K(X_S, X_S) = sum_{l, m in S} B[l, m].

Two games are provided.  In the *atom* game the players are atoms, or groups of atoms such as the
substituents of a common scaffold; the scaffold can be held fixed, so that the attribution explains
the difference from the reference compound rather than from an empty region of space.  In the
*field* game the players are the field types, which replaces the per-field overlap integral that
only a linear model can report.

For the linear kernel without normalisation the atom game is additive, so the Shapley value of an
atom equals its contribution under rule 3 exactly (:func:`cmfqsar.model.CMFModel.atom_contributions`)
and all interaction indices vanish; the tests assert this.  What a nonlinear model needs beyond the
values themselves is the second-order Shapley interaction index (Grabisch & Roubens 1999), which
measures the part of the prediction that two groups produce only together.

References: Shapley (1953); Grabisch & Roubens (1999); Lundberg & Lee (2017) for the use of these
values as model explanations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fields import MoleculeFields
from .kernels import Cancelled

ProgressFn = Callable[[str, float], None]
CancelFn = Callable[[], bool]


# --------------------------------------------------------------------------- #
# Games
# --------------------------------------------------------------------------- #

class _Game:
    """A coalition game: `value(masks)` evaluates the model for a batch of coalitions."""

    labels: List[str] = []
    members: List[List[int]] = []
    kind: str = ""

    @property
    def n_players(self) -> int:
        return len(self.labels)

    def value(self, masks: np.ndarray) -> np.ndarray:        # pragma: no cover - interface
        raise NotImplementedError


class AtomGame(_Game):
    """Players are atoms, or groups of atoms; the field of a coalition is the field of its atoms.

    `core` holds atoms that belong to every coalition (the common scaffold), so the empty
    coalition is the reference compound rather than empty space.
    """

    kind = "atoms"

    def __init__(self, model, mol: MoleculeFields, groups: Optional[Sequence[Sequence[int]]] = None,
                 core: Sequence[int] = (), labels: Optional[Sequence[str]] = None):
        self.model = model
        self.mol = mol
        n = mol.n_atoms
        self.core = np.zeros(n, bool)
        self.core[list(core)] = True
        if groups is None:
            groups = [[i] for i in range(n) if not self.core[i]]
        self.members = [list(map(int, g)) for g in groups]
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [(f"{mol.symbols[g[0]]}{g[0] + 1}" if len(g) == 1
                            else f"{mol.symbols[g[0]]}{g[0] + 1}+{len(g) - 1}") for g in self.members]
        self.blk = model._atom_block(mol)                    # (F, N, n_train), raw kernels
        self.B = model._atom_self_block(mol)                 # (F, N, N), raw kernels
        self.mix = model.fieldset.mixing_vector() * model.scale
        self.normalize = bool(model.normalize)
        self.d_train = model.self_diag if self.normalize else None      # raw (F, n_train)
        self.coef, self.bias = model.coef, model.bias
        self.base_diag = getattr(model, "base_diag_", None)
        if self.base_diag is None:         # a model stored before the layer existed
            self.base_diag = model.joint_self(model.train)
        F, _, nt = self.blk.shape
        self.chunk = max(1, int(2.0e6 / max(F * nt, 1)))
        # atoms of every player, as a (P, N) membership matrix
        self.memb = np.zeros((len(self.members), n), bool)
        for p, g in enumerate(self.members):
            self.memb[p, g] = True

    def atoms_of(self, masks: np.ndarray) -> np.ndarray:
        """Atom membership (m, N) of the coalitions in `masks` (m, P), including the core."""
        sel = np.asarray(masks, bool).astype(np.int8) @ self.memb.astype(np.int8)
        return (sel > 0) | self.core[None, :]

    def value(self, masks: np.ndarray) -> np.ndarray:
        masks = np.atleast_2d(np.asarray(masks, bool))
        out = np.empty(len(masks))
        for s in range(0, len(masks), self.chunk):
            A = self.atoms_of(masks[s:s + self.chunk]).astype(float)       # (m, N)
            Kf = np.einsum("ma,fan->fmn", A, self.blk, optimize=True)      # (F, m, n_train)
            q = np.einsum("ma,fab,mb->fm", A, self.B, A, optimize=True)    # (F, m) self-kernels
            if self.normalize:
                denom = np.sqrt(np.maximum(q, 0.0))[:, :, None] * \
                    np.sqrt(np.maximum(self.d_train, 0.0))[:, None, :]
                with np.errstate(divide="ignore", invalid="ignore"):
                    Kf = np.where(denom > 1e-300, Kf / denom, 0.0)
                q = np.where(q > 1e-300, 1.0, 0.0)
            K = np.einsum("f,fmn->mn", self.mix, Kf, optimize=True)
            if not self.model.is_linear:
                K = self.model.apply_layer(K, self.mix @ q, self.base_diag)
            out[s:s + self.chunk] = K @ self.coef + self.bias
        return out


class AtomFieldGame(_Game):
    """Players are (atom or group, field type) pairs: what each atom contributes to each field.

    This is the game behind a *Shapley field*.  Its values can be laid back onto the atoms of one
    field type and smeared over space with the Gaussians of that field, giving a map per field
    whose integral is the attributed part of the prediction, so that a nonlinear model gets one
    picture per field type where a linear one has a co-field.
    """

    kind = "atom-fields"

    def __init__(self, model, mol: MoleculeFields, groups: Optional[Sequence[Sequence[int]]] = None,
                 core: Sequence[int] = (), labels: Optional[Sequence[str]] = None):
        base = AtomGame(model, mol, groups=groups, core=core, labels=labels)
        self.base = base
        self.model, self.mol = model, mol
        self.fields = [i for i, s in enumerate(model.fieldset.specs) if s.enabled]
        self.field_labels = [model.fieldset.specs[i].label for i in self.fields]
        self.n_groups = len(base.members)
        self.labels = [f"{g} / {fl}" for fl in self.field_labels for g in base.labels]
        self.members = [list(base.members[g]) for _ in self.fields for g in range(self.n_groups)]
        self.field_of = [f for f in range(len(self.fields)) for _ in range(self.n_groups)]
        F, _, nt = base.blk.shape
        self.chunk = max(1, int(1.0e6 / max(len(self.fields) * max(nt, 1), 1)))

    def value(self, masks: np.ndarray) -> np.ndarray:
        b = self.base
        masks = np.atleast_2d(np.asarray(masks, bool))
        out = np.empty(len(masks))
        fidx = np.array(self.fields, int)
        for s in range(0, len(masks), self.chunk):
            m = masks[s:s + self.chunk]
            sel = m.reshape(len(m), len(self.fields), self.n_groups)          # (m, F, G)
            A = np.einsum("mfg,ga->mfa", sel.astype(np.int8), b.memb.astype(np.int8)) > 0
            A = (A | b.core[None, None, :]).astype(float)                     # (m, F, N)
            Kf = np.einsum("mfa,fan->fmn", A, b.blk[fidx], optimize=True)     # (F, m, n_train)
            q = np.einsum("mfa,fab,mfb->fm", A, b.B[fidx], A, optimize=True)
            if b.normalize:
                denom = np.sqrt(np.maximum(q, 0.0))[:, :, None] * \
                    np.sqrt(np.maximum(b.d_train[fidx], 0.0))[:, None, :]
                with np.errstate(divide="ignore", invalid="ignore"):
                    Kf = np.where(denom > 1e-300, Kf / denom, 0.0)
                q = np.where(q > 1e-300, 1.0, 0.0)
            mix = b.mix[fidx]
            K = np.einsum("f,fmn->mn", mix, Kf, optimize=True)
            if not self.model.is_linear:
                K = self.model.apply_layer(K, mix @ q, b.base_diag)
            out[s:s + self.chunk] = K @ b.coef + b.bias
        return out


class FieldGame(_Game):
    """Players are the field types: what each field contributes to the prediction.

    For a linear model this is the integral of the overlap of co-field and molecular field
    (Eq. 7); for a nonlinear one no such integral exists and the Shapley value takes its place.
    """

    kind = "fields"

    def __init__(self, model, mol: MoleculeFields):
        from .fields import AtomTable
        self.model = model
        self.mol = mol
        table = AtomTable.from_molecules([mol])
        self.Kf = model.test_kernels(table)[:, 0, :]                 # (F, n_train), as the model uses
        d = model._self_kernels(table)[:, 0]                         # raw per-field self-kernels
        self.sf = np.where(d > 1e-300, 1.0, 0.0) if model.normalize else d
        if model.K_fields_train is not None:
            self.dtr = np.diagonal(model.K_fields_train, axis1=1, axis2=2).copy()   # (F, n_train)
        else:                      # a model loaded from a file keeps no kernel matrices
            d_raw = model._self_kernels(model.train)
            self.dtr = np.where(d_raw > 1e-300, 1.0, 0.0) if model.normalize else d_raw
        self.mix = model.fieldset.mixing_vector() * model.scale
        self.coef, self.bias = model.coef, model.bias
        on = [i for i, s in enumerate(model.fieldset.specs) if s.enabled]
        self.members = [[i] for i in on]
        self.labels = [model.fieldset.specs[i].label for i in on]
        self.idx = np.array(on, int)
        F, nt = self.Kf.shape
        self.chunk = max(1, int(2.0e6 / max(F * nt, 1)))

    def value(self, masks: np.ndarray) -> np.ndarray:
        masks = np.atleast_2d(np.asarray(masks, bool))
        out = np.empty(len(masks))
        w_all = self.mix[self.idx]
        for s in range(0, len(masks), self.chunk):
            M = masks[s:s + self.chunk].astype(float) * w_all[None, :]      # (m, P) weights in use
            K = M @ self.Kf[self.idx]                                       # (m, n_train)
            if not self.model.is_linear:
                diag_p = M @ self.sf[self.idx]
                diag_q = M @ self.dtr[self.idx]                             # (m, n_train)
                K = _apply_layer_rowwise(self.model, K, diag_p, diag_q)
            out[s:s + self.chunk] = K @ self.coef + self.bias
        return out


def _apply_layer_rowwise(model, K: np.ndarray, diag_p: np.ndarray, diag_q: np.ndarray) -> np.ndarray:
    """Apply the kernel layer when every coalition has its own training diagonal.

    Dropping a field changes the self-kernel of the *training* molecules as well, so `diag_q`
    is a matrix here and :meth:`CMFModel.apply_layer` cannot be used directly.
    """
    if model.is_linear:
        return K
    if model.kernel_layer == "gauss":
        g = float(model.params.get("gamma", 1.0)) * float(getattr(model, "gamma_scale_", None) or 1.0)
        d2 = diag_p[:, None] + diag_q - 2.0 * K
        return np.exp(-g * np.maximum(d2, 0.0))
    if model.kernel_layer in ("arccos1", "arccos2"):
        def arccos(Kx):
            npn = np.sqrt(np.maximum(diag_p, 1e-300))[:, None]
            nqn = np.sqrt(np.maximum(diag_q, 1e-300))
            cos = np.clip(Kx / (npn * nqn), -1.0, 1.0)
            th = np.arccos(cos)
            return (npn * nqn) / np.pi * (np.sin(th) + (np.pi - th) * cos)
        out = arccos(K)
        return arccos(out) if model.kernel_layer == "arccos2" else out
    raise ValueError(f"unknown kernel layer {model.kernel_layer}")


# --------------------------------------------------------------------------- #
# Shapley values and interaction indices
# --------------------------------------------------------------------------- #

@dataclass
class ShapleyResult:
    """Shapley decomposition of one prediction."""
    kind: str                       # "atoms" | "fields"
    labels: List[str]
    members: List[List[int]]        # atom (or field) indices behind every player
    values: np.ndarray              # (P,) Shapley values
    baseline: float                 # v(empty coalition): the reference prediction
    prediction: float               # v(all players): the prediction of the model
    exact: bool
    n_evaluations: int
    stderr: Optional[np.ndarray] = None       # standard error of the sampled estimate
    interactions: Optional[np.ndarray] = None  # (P, P) second-order indices, diagonal zero
    interactions_exact: bool = False
    player_fields: Optional[List[int]] = None  # field index of each player (atom-field game only)

    @property
    def efficiency_error(self) -> float:
        """How far the values are from adding up to prediction - baseline (zero if exact)."""
        return float(abs(self.values.sum() - (self.prediction - self.baseline)))

    def atom_values(self, n_atoms: int) -> np.ndarray:
        """Spread the values of the players over the atoms they contain (atom game only)."""
        out = np.zeros(n_atoms)
        for v, g in zip(self.values, self.members):
            if g:
                out[list(g)] = v / len(g)
        return out

    def field_atom_values(self, n_atoms: int, n_fields: int) -> np.ndarray:
        """Values laid back onto (field, atom), ready to be smeared into a Shapley field."""
        if self.player_fields is None:
            raise ValueError("this decomposition has no field axis: use the atom-field game")
        out = np.zeros((n_fields, n_atoms))
        for v, g, f in zip(self.values, self.members, self.player_fields):
            if g:
                out[f, list(g)] = v / len(g)
        return out

    def table(self) -> List[Tuple[str, float, float]]:
        se = self.stderr if self.stderr is not None else np.full(len(self.values), np.nan)
        order = np.argsort(-np.abs(self.values))
        return [(self.labels[i], float(self.values[i]), float(se[i])) for i in order]

    def top_interactions(self, k: int = 10) -> List[Tuple[str, str, float]]:
        if self.interactions is None:
            return []
        iu = np.triu_indices(len(self.labels), 1)
        vals = self.interactions[iu]
        order = np.argsort(-np.abs(vals))[:k]
        return [(self.labels[iu[0][o]], self.labels[iu[1][o]], float(vals[o])) for o in order]


def _weights(P: int) -> np.ndarray:
    """Shapley weights s!(P-s-1)!/P! for every coalition size s."""
    return np.array([factorial(s) * factorial(P - s - 1) / factorial(P) for s in range(P)])


def _exact_values(game: _Game, progress=None, cancel=None) -> Tuple[np.ndarray, np.ndarray, int]:
    """All 2^P coalition values, and the Shapley values derived from them."""
    P = game.n_players
    idx = np.arange(1 << P)
    bits = ((idx[:, None] >> np.arange(P)) & 1).astype(bool)
    v = np.empty(len(idx))
    chunk = max(1, getattr(game, "chunk", 4096))
    for s in range(0, len(idx), chunk):
        if cancel and cancel():
            raise Cancelled()
        e = min(s + chunk, len(idx))
        v[s:e] = game.value(bits[s:e])
        if progress:
            progress(f"coalitions {e}/{len(idx)}", e / len(idx))
    w = _weights(P)
    sizes = bits.sum(axis=1)
    phi = np.zeros(P)
    for i in range(P):
        without = np.flatnonzero(~bits[:, i])
        phi[i] = float(np.sum(w[sizes[without]] * (v[without + (1 << i)] - v[without])))
    return phi, v, len(idx)


def _sampled_values(game: _Game, samples: int, seed: Optional[int], progress=None,
                    cancel=None) -> Tuple[np.ndarray, np.ndarray, int]:
    """Permutation sampling: each permutation gives one unbiased sample per player."""
    P = game.n_players
    rng = np.random.RandomState(seed)
    contrib = np.zeros((samples, P))
    batch = max(1, min(samples, int(4096 / (P + 1)) or 1))
    done = 0
    for s in range(0, samples, batch):
        if cancel and cancel():
            raise Cancelled()
        m = min(batch, samples - s)
        masks = np.zeros((m * (P + 1), P), bool)
        perms = np.empty((m, P), int)
        for k in range(m):
            perm = rng.permutation(P)
            perms[k] = perm
            row = np.zeros(P, bool)
            masks[k * (P + 1)] = row
            for t, p in enumerate(perm):
                row = row.copy()
                row[p] = True
                masks[k * (P + 1) + t + 1] = row
        v = game.value(masks).reshape(m, P + 1)
        d = np.diff(v, axis=1)
        for k in range(m):
            contrib[s + k, perms[k]] = d[k]
        done += m * (P + 1)
        if progress:
            progress(f"permutations {s + m}/{samples}", (s + m) / samples)
    phi = contrib.mean(axis=0)
    se = contrib.std(axis=0, ddof=1) / np.sqrt(samples) if samples > 1 else np.zeros(P)
    return phi, se, done


def _exact_interactions(game: _Game, v: np.ndarray) -> np.ndarray:
    """Second-order Shapley interaction indices from the table of all coalition values."""
    P = game.n_players
    idx = np.arange(1 << P)
    bits = ((idx[:, None] >> np.arange(P)) & 1).astype(bool)
    sizes = bits.sum(axis=1)
    w = np.array([factorial(s) * factorial(P - s - 2) / factorial(P - 1) for s in range(max(P - 1, 1))])
    out = np.zeros((P, P))
    for i in range(P):
        for j in range(i + 1, P):
            sel = np.flatnonzero(~bits[:, i] & ~bits[:, j])
            delta = (v[sel + (1 << i) + (1 << j)] - v[sel + (1 << i)]
                     - v[sel + (1 << j)] + v[sel])
            out[i, j] = out[j, i] = float(np.sum(w[sizes[sel]] * delta))
    return out


def _sampled_interactions(game: _Game, samples: int, seed: Optional[int], progress=None,
                          cancel=None) -> Tuple[np.ndarray, int]:
    """Unbiased estimate of the interaction indices.

    The Shapley weights make the coalition *size* uniform on 0 ... P-2, so a sample is drawn by
    picking a size uniformly and then a subset of that size among the other players.
    """
    P = game.n_players
    rng = np.random.RandomState(seed)
    pairs = [(i, j) for i in range(P) for j in range(i + 1, P)]
    out = np.zeros((P, P))
    done = 0
    for c, (i, j) in enumerate(pairs):
        if cancel and cancel():
            raise Cancelled()
        others = np.array([k for k in range(P) if k not in (i, j)], int)
        masks = np.zeros((samples * 4, P), bool)
        for t in range(samples):
            size = rng.randint(0, P - 1)              # 0 ... P-2
            S = rng.choice(others, size=size, replace=False) if size else np.empty(0, int)
            base = np.zeros(P, bool)
            base[S] = True
            masks[4 * t] = base
            m_i = base.copy(); m_i[i] = True
            m_j = base.copy(); m_j[j] = True
            m_ij = m_i.copy(); m_ij[j] = True
            masks[4 * t + 1] = m_i
            masks[4 * t + 2] = m_j
            masks[4 * t + 3] = m_ij
        v = game.value(masks).reshape(samples, 4)
        out[i, j] = out[j, i] = float(np.mean(v[:, 3] - v[:, 1] - v[:, 2] + v[:, 0]))
        done += samples * 4
        if progress:
            progress(f"pair {c + 1}/{len(pairs)}", (c + 1) / max(len(pairs), 1))
    return out, done


def shapley(game: _Game, samples: int = 400, seed: Optional[int] = 0, exact_limit: int = 14,
            interactions: bool = False, interaction_samples: int = 64,
            interaction_limit: int = 24, progress: Optional[ProgressFn] = None,
            cancel: Optional[CancelFn] = None) -> ShapleyResult:
    """Shapley values of a coalition game, exactly for few players and by sampling otherwise."""
    P = game.n_players
    if P == 0:
        raise ValueError("the game has no players: nothing to decompose")
    exact = P <= exact_limit
    v_table = None
    if exact:
        phi, v_table, n_eval = _exact_values(game, progress=progress, cancel=cancel)
        se = None
    else:
        phi, se, n_eval = _sampled_values(game, samples, seed, progress=progress, cancel=cancel)
    empty = float(game.value(np.zeros((1, P), bool))[0])
    full = float(game.value(np.ones((1, P), bool))[0])
    inter, inter_exact = None, False
    if interactions and P >= 2:
        if P > interaction_limit:
            raise ValueError(f"interaction indices are computed for at most {interaction_limit} "
                             f"players ({P} here); group the atoms, for example into the "
                             f"substituents of a common scaffold")
        if v_table is not None:
            inter, inter_exact = _exact_interactions(game, v_table), True
        else:
            if progress:
                progress("interaction indices", 0.0)
            inter, extra = _sampled_interactions(game, interaction_samples, seed,
                                                 progress=progress, cancel=cancel)
            n_eval += extra
    pf = None
    if getattr(game, "field_of", None) is not None:
        pf = [int(game.fields[k]) for k in game.field_of]
    return ShapleyResult(kind=game.kind, labels=list(game.labels), members=list(game.members),
                         values=phi, baseline=empty, prediction=full, exact=exact,
                         n_evaluations=n_eval + 2, stderr=se, interactions=inter,
                         interactions_exact=inter_exact, player_fields=pf)


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #

def atom_shapley(model, mol: MoleculeFields, groups: Optional[Sequence[Sequence[int]]] = None,
                 core: Sequence[int] = (), labels: Optional[Sequence[str]] = None,
                 samples: int = 400, seed: Optional[int] = 0, exact_limit: int = 14,
                 interactions: bool = False, interaction_samples: int = 64,
                 progress: Optional[ProgressFn] = None,
                 cancel: Optional[CancelFn] = None) -> ShapleyResult:
    """Decompose a prediction over atoms, or over groups of atoms.

    `core` atoms are present in every coalition, so the baseline is the prediction for that
    fragment alone; with the maximum common substructure as the core the values explain the
    difference between this compound and the scaffold.
    """
    game = AtomGame(model, mol, groups=groups, core=core, labels=labels)
    return shapley(game, samples=samples, seed=seed, exact_limit=exact_limit,
                   interactions=interactions, interaction_samples=interaction_samples,
                   progress=progress, cancel=cancel)


def atom_field_shapley(model, mol: MoleculeFields, groups: Optional[Sequence[Sequence[int]]] = None,
                       core: Sequence[int] = (), labels: Optional[Sequence[str]] = None,
                       samples: int = 400, seed: Optional[int] = 0, exact_limit: int = 14,
                       interactions: bool = False, interaction_samples: int = 64,
                       progress: Optional[ProgressFn] = None,
                       cancel: Optional[CancelFn] = None) -> ShapleyResult:
    """Decompose a prediction over (atom, field type) pairs: the basis of a Shapley field."""
    game = AtomFieldGame(model, mol, groups=groups, core=core, labels=labels)
    return shapley(game, samples=samples, seed=seed, exact_limit=exact_limit,
                   interactions=interactions, interaction_samples=interaction_samples,
                   progress=progress, cancel=cancel)


def field_shapley(model, mol: MoleculeFields, interactions: bool = False, samples: int = 400,
                  seed: Optional[int] = 0, exact_limit: int = 14,
                  interaction_samples: int = 64, progress: Optional[ProgressFn] = None,
                  cancel: Optional[CancelFn] = None) -> ShapleyResult:
    """Decompose a prediction over the field types."""
    game = FieldGame(model, mol)
    return shapley(game, samples=samples, seed=seed, exact_limit=exact_limit,
                   interactions=interactions, interaction_samples=interaction_samples,
                   progress=progress, cancel=cancel)


# --------------------------------------------------------------------------- #
# Scaffold baseline: the common core and the substituents hanging from it
# --------------------------------------------------------------------------- #

def common_core(mol, references: Sequence, timeout: int = 10,
                ring_matches_ring_only: bool = True) -> List[int]:
    """Atoms of `mol` that belong to its maximum common substructure with `references`.

    Hydrogens attached to a core atom are part of the core.  Returns atom indices in the order
    of `mol`, which is the order of the fields built from it.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    params = rdFMCS.MCSParameters()
    params.Timeout = timeout
    params.AtomCompareParameters.RingMatchesRingOnly = ring_matches_ring_only
    params.BondCompareParameters.RingMatchesRingOnly = ring_matches_ring_only
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.BondTyper = rdFMCS.BondCompare.CompareOrder
    mols = [Chem.RemoveHs(Chem.Mol(m)) for m in [mol] + list(references)]
    res = rdFMCS.FindMCS(mols, params)
    if res.numAtoms < 3:
        raise RuntimeError("the common substructure is smaller than three atoms; "
                           "use the empty baseline instead")
    patt = Chem.MolFromSmarts(res.smartsString)
    match = mol.GetSubstructMatch(patt)
    if not match:
        raise RuntimeError("the common substructure does not match this molecule")
    core = set(int(i) for i in match)
    for i in list(core):                       # hydrogens of a core atom belong to the core
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetAtomicNum() == 1:
                core.add(int(nb.GetIdx()))
    return sorted(core)


def substituent_groups(mol, core: Sequence[int]) -> Tuple[List[List[int]], List[str]]:
    """Group the atoms outside the core into connected substituents, with readable labels."""
    core_set = set(int(i) for i in core)
    n = mol.GetNumAtoms()
    seen = set()
    groups: List[List[int]] = []
    for start in range(n):
        if start in core_set or start in seen or mol.GetAtomWithIdx(start).GetAtomicNum() == 1:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            a = stack.pop()
            comp.append(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                k = int(nb.GetIdx())
                if k in core_set or k in seen or nb.GetAtomicNum() == 1:
                    continue
                seen.add(k)
                stack.append(k)
        groups.append(sorted(comp))
    # every hydrogen follows the heavy atom it is attached to
    owner = {}
    for g, comp in enumerate(groups):
        for a in comp:
            owner[a] = g
    for a in range(n):
        at = mol.GetAtomWithIdx(a)
        if at.GetAtomicNum() != 1 or a in core_set:
            continue
        for nb in at.GetNeighbors():
            k = int(nb.GetIdx())
            if k in owner:
                groups[owner[k]].append(a)
                break
    labels = []
    for comp in groups:
        heavy = [a for a in comp if mol.GetAtomWithIdx(a).GetAtomicNum() != 1]
        sym = "".join(sorted(mol.GetAtomWithIdx(a).GetSymbol() for a in heavy))
        labels.append(f"{sym} at {min(heavy) + 1}" if heavy else f"H{min(comp) + 1}")
    return [sorted(g) for g in groups], labels


def scaffold_decomposition(model, mol: MoleculeFields, rdmol, references: Sequence,
                           **kwargs) -> ShapleyResult:
    """Shapley values of the substituents of a congeneric series over their common scaffold."""
    core = common_core(rdmol, references)
    groups, labels = substituent_groups(rdmol, core)
    if not groups:
        raise RuntimeError("this molecule is the scaffold itself: nothing to decompose")
    return atom_shapley(model, mol, groups=groups, core=core, labels=labels, **kwargs)
