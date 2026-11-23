"""Atom amplitudes of the fields from a network on the atoms (plan item ND14a).

A molecular field of CMF is a sum of Gaussians, one on every atom, and every Gaussian has an
amplitude: the partial charge of the atom in the electrostatic field, a well depth in the
steric field, a hydrogen-bond strength, a one in an indicator field.  These amplitudes are
fixed numbers of a force field or of a rule.  This module lets a small network produce them.
The network reads every atom of the molecular graph with shared weights, and its output is a
correction to the fixed amplitude:

    a_f(i) = w_f(i) + s_f d_f(i),

with ``w_f(i)`` the fixed amplitude of atom ``i`` in field ``f``, ``s_f`` the scale of the
field (the root mean square of ``w_f`` over the atoms that carry the field) and ``d_f(i)`` the
output of the network.  Only the atoms that carry the field are corrected: an atom with the
fixed amplitude zero keeps it.  The last layer of the network starts at zero, so the model at
the start is the model with the fixed amplitudes.

The network is the first level of the neural device of Baskin, Palyulin and Zefirov (J. Chem.
Inf. Comput. Sci. 37 (1997) 715): one network with shared weights on every atom.  With
``rounds=0`` it reads the atom alone.  With ``rounds=k`` every atom also reads the sum of the
hidden values of its bonded neighbours, ``k`` times, so that its output depends on the atoms up
to ``k`` bonds away (message passing).  The input of an atom is its atom type (one column per
type), its fixed amplitude in every field divided by the scale of the field, and the numbers of
its bonded non-hydrogen atoms and hydrogen atoms.

What stays
----------
The model stays the kernel model of :mod:`cmfqsar.learnable`:

    y(x) = sum_z c_z K(x, z) + b,        K(x, z) = sum_f h_f K_f(x, z),

with ``K_f`` the overlap of the fields ``f`` of two molecules, now with the amplitudes above.
The coefficients ``c`` and the bias are the closed-form ridge solution for the current
amplitudes; they are not trained by gradient.  The model is linear in the fields, so it has a
regression-coefficient field (co-field), :func:`cofield_gaussians`, and
:func:`atom_weights` returns the amplitudes in the layout of an atom table, so that every
other part of the program can use them as the weights of the atoms.

The objective
-------------
It is the criterion of :mod:`cmfqsar.learnable`: the squared error of the out-of-fold
predictions inside the training part, divided by the variance of the activity, plus a penalty
on the corrections, ``penalty`` times the mean of ``d^2`` over the corrected atoms of the
training molecules (averaged over the fields).  The penalty holds the amplitudes near the
fixed values.  A quasi-Newton method (L-BFGS-B) minimises the sum over the weights of the
network and, with ``learn_lam``, the logarithm of the ridge.  With validation rows, the
evaluation with the lowest validation error is kept (early stopping).

How the gradient reaches the network
------------------------------------
A field kernel is ``K_f(x, z) = sum_ij a_i a_j E(i, j)`` with ``E`` the overlap of two atom
Gaussians.  The array ``E`` of all atom pairs is too large to keep, so every evaluation makes
two passes over the atom pairs.  The first pass computes the kernels for the current
amplitudes.  The criterion is then a function of the kernels, and automatic differentiation
gives its derivative ``G`` with respect to every kernel entry.  The second pass computes, from
``G`` and the same overlaps, the derivative with respect to the amplitude of every atom,

    dL / da_i = sum_z G(x, z) sum_j E(i, j) a_j          (atom i of molecule x),

and the same sum for the atoms of the landmark molecules.  This derivative is exact, and it is
handed to the network by one backward step.  No graph of the atom pairs is kept.

Precision
---------
On a GPU the kernels are computed in single precision unless ``dtype`` says otherwise; the
ridge solution is always computed in double precision.  With a small ridge the criterion is
sensitive to the last digits of the kernels.  On a part of COMPAS-1D (1,500 training molecules,
243 landmarks) the criterion at the start was 0.1340 in single and 0.1347 in double precision,
and the two trainings took the same course.  On a series of a few hundred molecules double
precision costs about twice the time and removes this difference; the benchmark uses it there.

PyTorch is needed (the optional dependency of :mod:`cmfqsar.fieldae`).  Without it the module
imports, :func:`available` is false and :func:`pack_graph` still works.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import learnable as L

try:                                                    # pragma: no cover - import guard
    import torch
except ImportError:                                     # pragma: no cover
    torch = None

#: length of the first step of the training, in the trained numbers
FIRST_STEP = 0.3
#: bound of every weight of the network, as a distance from its start
WEIGHT_BOUND = 50.0
TINY = L.TINY


def available() -> bool:
    """Whether PyTorch is installed, so that the amplitudes can be trained."""
    return L.HAVE_TORCH


# --------------------------------------------------------------------------- #
# The atoms of a table: packed blocks per field, and the molecular graph
# --------------------------------------------------------------------------- #

def pack_graph(table, specs, columns: Optional[Sequence[int]] = None,
               like: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """The packed atoms of :func:`cmfqsar.learnable.pack` and the graph the network reads.

    The result holds everything :func:`cmfqsar.learnable.pack` returns and, in addition:

    ``atom[f]``    (molecules, atoms) for every field: the index of the packed atom in the
                   atom list of the whole table, or -1 where the block is padded;
    ``x``          (atoms of the table, columns): the input of the network for every atom;
    ``edges``      (2, bonds times 2): the bonded atom pairs, in both directions;
    ``offsets``    the first atom of every molecule;
    ``scale``      one number per field: the root mean square of the fixed amplitude over the
                   atoms that carry the field;
    ``types``      the atom types that have a column, and ``columns_x`` the name of every column.

    `table` is an atom table whose molecules carry ``atom_types`` and ``bonds``
    (:class:`cmfqsar.fields.AtomTable`).  For the table of other molecules pass the graph of
    the training table as `like`: the atom types and the scales are then taken from it, so
    that the columns mean the same.  An atom type that `like` does not have gets no column.
    """
    cols = list(range(len(specs))) if columns is None else [int(c) for c in columns]
    out = L.pack(table, specs, cols)
    W = np.asarray(table.weights, float)
    offsets = np.asarray(table.offsets, int)
    n = len(offsets) - 1
    mol_of = np.repeat(np.arange(n), np.diff(offsets))
    out["atom"] = []
    for f, col in enumerate(cols):
        keep = np.flatnonzero(W[:, col] != 0.0)
        m = mol_of[keep]
        counts = np.bincount(m, minlength=n)
        start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        rank = np.arange(len(keep)) - start[m]
        atom = np.full(out["w"][f].shape, -1, dtype=np.int64)
        atom[m, rank] = keep
        out["atom"].append(atom)
    if like is None:
        scale = np.array([float(np.sqrt(np.mean(W[W[:, c] != 0.0, c] ** 2)))
                          if np.any(W[:, c] != 0.0) else 1.0 for c in cols])
        types = sorted({str(t) for mf in table.molecules for t in mf.atom_types})
    else:
        scale = np.asarray(like["scale"], float)
        types = list(like["types"])
        if len(scale) != len(cols):
            raise ValueError("the graph given as `like` has another number of fields")
    col_of = {t: k for k, t in enumerate(types)}
    N = len(W)
    T = len(types)
    x = np.zeros((N, T + len(cols) + 2))
    x[:, T:T + len(cols)] = W[:, cols] / scale[None, :]
    edges = []
    for k, mf in enumerate(table.molecules):
        base = int(offsets[k])
        for i, t in enumerate(mf.atom_types):
            c = col_of.get(str(t))
            if c is not None:
                x[base + i, c] = 1.0
        bonds = np.asarray(mf.bonds, int).reshape(-1, 2)
        hydrogen = np.array([s == "H" for s in mf.symbols], bool)
        for a, b in bonds:
            x[base + a, T + len(cols) + int(hydrogen[b])] += 0.25
            x[base + b, T + len(cols) + int(hydrogen[a])] += 0.25
        if len(bonds):
            edges.append(bonds + base)
    e = np.concatenate(edges, axis=0) if edges else np.zeros((0, 2), int)
    out["x"] = x
    out["edges"] = np.concatenate([e.T, e.T[::-1]], axis=1).astype(np.int64)
    out["offsets"] = offsets.copy()
    out["scale"] = scale
    out["types"] = types
    out["columns_x"] = ([f"type {t}" for t in types] + [f"amplitude {s.name}" for s in specs]
                        + ["bonded non-hydrogen atoms / 4", "bonded hydrogen atoms / 4"])
    return out


# --------------------------------------------------------------------------- #
# The network on the atoms
# --------------------------------------------------------------------------- #

def _network(n_in: int, hidden: int, rounds: int, n_out: int, seed: int):
    """The network of one atom: a first layer, `rounds` layers that also read the sum over
    the bonded neighbours, and a linear output that starts at zero."""
    L.require()

    class AtomNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(int(seed))
            self.first = torch.nn.Linear(int(n_in), int(hidden))
            self.own = torch.nn.ModuleList(
                [torch.nn.Linear(int(hidden), int(hidden)) for _ in range(int(rounds))])
            self.neighbours = torch.nn.ModuleList(
                [torch.nn.Linear(int(hidden), int(hidden), bias=False)
                 for _ in range(int(rounds))])
            self.out = torch.nn.Linear(int(hidden), int(n_out))
            torch.nn.init.zeros_(self.out.weight)
            torch.nn.init.zeros_(self.out.bias)

        def forward(self, x, edges):
            h = torch.tanh(self.first(x))
            for own, neighbours in zip(self.own, self.neighbours):
                total = torch.zeros_like(h).index_add_(0, edges[0], h[edges[1]])
                h = torch.tanh(own(h) + neighbours(total))
            return self.out(h)

    return AtomNetwork().double()


# --------------------------------------------------------------------------- #
# The two passes over the atom pairs
# --------------------------------------------------------------------------- #

def _pair_overlap(xp, cp, xq, cq, alpha: float, kind: str):
    """The overlap of every atom of a block with every atom of the landmarks:
    (atoms of the block, landmarks, atoms of a landmark)."""
    b, A = xp.shape[:2]
    m, B = xq.shape[:2]
    P = xp.reshape(b * A, 3)
    Q = xq.reshape(m * B, 3)
    g = torch.mm(P, Q.T)
    g.mul_(-2.0).add_((P * P).sum(1)[:, None]).add_((Q * Q).sum(1)[None, :]).clamp_(min=0.0)
    if kind == "steric":
        ci, cj = cp.reshape(b * A, 1), cq.reshape(1, m * B)
        s = ci + cj
        g.mul_(alpha * ci * cj / (2.0 * s))
        z = (2.0 * math.pi / alpha) / s
        E = g.neg_().exp_().mul_(z * torch.sqrt(z))
    else:
        g.mul_(0.25 * alpha)
        E = g.neg_().exp_().mul_((math.pi / alpha) ** 1.5)
    return E.view(b * A, m, B)


def _self_overlap(x, c, alpha: float, kind: str):
    d2 = ((x[:, :, None, :] - x[:, None, :, :]) ** 2).sum(-1)
    if kind == "steric":
        return L._overlap(d2, alpha, c[:, :, None], c[:, None, :])[0]
    return L._overlap(d2, alpha)[0]


def _field_pass(xp, ap, cp, xq, aq, cq, alpha: float, kind: str, budget: float,
                G=None, Gd=None):
    """One field: the kernels of the rows against the landmarks, or their pull-back.

    `xp`, `ap`, `cp` are the coordinates, amplitudes and width factors of the rows, (rows,
    atoms, ...); `xq`, `aq`, `cq` those of the landmarks.  Without `G` the result is ``(K, d)``:
    the kernel (rows, landmarks) and the self-kernel of every row.  With `G` (the derivative
    of a function with respect to ``K``) and `Gd` (with respect to ``d``) the result is the
    derivative of that function with respect to the amplitudes: ``(rows, atoms)`` for the atoms
    of the rows and ``(landmarks, atoms)`` for the atoms of the landmarks.
    """
    n, A = ap.shape
    m, B = aq.shape
    forward = G is None
    if forward:
        K = torch.zeros((n, m), dtype=ap.dtype, device=ap.device)
        d = torch.zeros(n, dtype=ap.dtype, device=ap.device)
    else:
        gp = torch.zeros_like(ap)
        gq = torch.zeros_like(aq)
    step = max(1, int(budget // max(A * m * B, 1)))
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        b = hi - lo
        E = _pair_overlap(xp[lo:hi], cp[lo:hi], xq, cq, alpha, kind)
        if forward:
            T = torch.einsum("izj,zj->iz", E, aq)
            K[lo:hi] = (T * ap[lo:hi].reshape(b * A, 1)).view(b, A, m).sum(1)
        else:
            T = torch.einsum("izj,zj->iz", E, aq).view(b, A, m)
            gp[lo:hi] = (T * G[lo:hi, None, :]).sum(2)
            S = (ap[lo:hi, :, None] * G[lo:hi, None, :]).reshape(b * A, m)
            gq += torch.einsum("iz,izj->zj", S, E)
        del E, T
    step = max(1, int(budget // max(A * A, 1)))
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        Ea = torch.einsum("bij,bj->bi", _self_overlap(xp[lo:hi], cp[lo:hi], alpha, kind),
                          ap[lo:hi])
        if forward:
            d[lo:hi] = (Ea * ap[lo:hi]).sum(1)
        else:
            gp[lo:hi] += 2.0 * Gd[lo:hi, None] * Ea
    return (K, d) if forward else (gp, gq)


# --------------------------------------------------------------------------- #
# The trained amplitudes
# --------------------------------------------------------------------------- #

@dataclass
class TrainedAmplitudes:
    """The network a training found and the kernel model fitted with its amplitudes.

    ``learned`` names the fields whose amplitudes the network corrects.  ``state`` holds the
    weights of the network, ``sizes`` its input width, hidden width and rounds.  ``coef`` and
    ``bias`` are the model over the landmarks ``landmarks`` (rows of the packed table);
    ``land`` holds the atoms of the landmarks for every field (coordinates, width factors and
    amplitudes), so that the model predicts other molecules without the training table.
    ``correction`` is the root mean square of the network output over the corrected atoms of
    the training molecules, one number per learned field, in units of the field scale.
    ``pred`` holds the prediction of the wanted molecules, ``history`` one record per
    evaluation.
    """
    names: List[str]
    kinds: List[str]
    learned: List[str]
    alpha: np.ndarray
    h: np.ndarray
    lam: float
    gamma: Optional[float]
    med: float
    normalize: str
    layer: Optional[str]
    landmarks: np.ndarray
    coef: np.ndarray
    bias: float
    pred: np.ndarray
    scale: np.ndarray
    sizes: Dict[str, int]
    state: Dict[str, np.ndarray]
    land: Dict[str, List[np.ndarray]]
    dz: np.ndarray                              # (fields, landmarks) self-kernels
    norms: np.ndarray                           # (fields, landmarks) factors of the co-field
    correction: Dict[str, float] = field(default_factory=dict)
    penalty: float = 0.0
    history: List[dict] = field(default_factory=list)
    start: Dict[str, object] = field(default_factory=dict)
    evaluations: int = 0
    seconds: float = 0.0
    pass_seconds: float = 0.0
    #: the model at the lowest objective, when the training kept it beside the selected one
    by_criterion: Optional["TrainedAmplitudes"] = None

    def settings(self) -> dict:
        return {"alpha": dict(zip(self.names, map(float, self.alpha))),
                "h": dict(zip(self.names, map(float, self.h))), "lam": float(self.lam),
                "gamma": None if self.gamma is None else float(self.gamma),
                "correction": dict(self.correction)}


def _net_of(trained: TrainedAmplitudes, dev):
    s = trained.sizes
    net = _network(s["inputs"], s["hidden"], s["rounds"], len(trained.learned), 0).to(dev)
    net.load_state_dict({k: torch.as_tensor(v, dtype=torch.float64, device=dev)
                         for k, v in trained.state.items()})
    return net


def _delta(net, graph, dev):
    """The output of the network for every atom of the table: (atoms, learned fields)."""
    x = torch.as_tensor(np.asarray(graph["x"], float), dtype=torch.float64, device=dev)
    edges = torch.as_tensor(np.asarray(graph["edges"]), dtype=torch.long, device=dev)
    return net(x, edges)


def _amplitude(graph, f: int, rows, delta_f, scale_f: float, dev):
    """The amplitudes of field `f` for the molecules `rows`: the fixed ones plus the scaled
    output of the network on the atoms that carry the field; (rows, atoms), double."""
    w = torch.as_tensor(np.asarray(graph["w"][f], float)[rows], dtype=torch.float64, device=dev)
    if delta_f is None:
        return w, None, None
    atom = torch.as_tensor(np.asarray(graph["atom"][f])[rows], dtype=torch.long, device=dev)
    mask = (atom >= 0).to(torch.float64)
    d = delta_f[atom.clamp(min=0)] * mask
    return w + float(scale_f) * d, d, mask


def train(graph, y, train, landmarks, valid=None, fields: Optional[Sequence[str]] = None,
          alpha=1.0, h=None, lam: float = 1e-2, gamma: Optional[float] = None,
          hidden: int = 16, rounds: int = 2, penalty: float = 0.1, learn_lam: bool = True,
          normalize: str = "cosine", layer: Optional[str] = None, steps: int = 30,
          folds: int = 5, seed: int = 0, select: Optional[str] = None, predict=None,
          keep_criterion: bool = False, first_step: float = FIRST_STEP, device=None,
          dtype=None, budget: float = L.BUDGET,
          log: Optional[Callable[[str], None]] = None) -> TrainedAmplitudes:
    """Train the network that corrects the atom amplitudes, and fit the ridge model with them.

    `graph` is the result of :func:`pack_graph`, `y` one activity per molecule, `train` and
    `valid` rows of the table, `landmarks` rows of the training part that carry the model.
    `fields` names the fields whose amplitudes are corrected (default: every field); an empty
    list trains no network, and the function then fits the model with the fixed amplitudes
    (with `learn_lam`, the ridge is still trained).  `alpha`, `h`, `lam` and `gamma` are the
    attenuations, the field weights, the start of the ridge and the width of the Gaussian
    kernel layer (``layer="gauss"``); they are not trained here, except the ridge.

    `hidden` is the width of the network and `rounds` the number of times an atom reads the
    sum over its bonded neighbours.  `penalty` multiplies the mean squared correction.
    `steps` bounds the iterations of the quasi-Newton method, which makes at most
    ``steps + 10`` evaluations; every evaluation makes two passes over the atom pairs of the
    learned fields.  `select` is ``"valid"`` (keep the evaluation with the lowest validation
    error; the default when `valid` is given) or ``"cv"`` (keep the lowest objective).
    `predict` lists the rows whose predictions are wanted (default: every molecule).
    """
    L.require()
    from scipy.optimize import minimize

    dev = L._device(device)
    dt = L._store_dtype(dev, dtype)
    f64 = torch.float64
    t_start = time.time()
    say = log or (lambda _msg: None)
    names = list(graph["names"])
    kinds = list(graph["kinds"])
    F = len(names)
    y = np.asarray(y, float)
    tr = np.asarray(train, int)
    va = np.asarray(valid, int) if valid is not None and len(valid) else np.zeros(0, int)
    land = np.asarray(landmarks, int)
    where = {int(r): k for k, r in enumerate(tr)}
    if any(int(z) not in where for z in land):
        raise ValueError("every landmark must be a molecule of the training part")
    pos_land = torch.as_tensor([where[int(z)] for z in land], dtype=torch.long, device=dev)
    wanted = names if fields is None else [str(v) for v in fields]
    unknown = [v for v in wanted if v not in names]
    if unknown:
        raise ValueError(f"no field {unknown[0]!r}; the fields are {', '.join(names)}")
    learned = [f for f in range(F) if names[f] in wanted]
    fixed = [f for f in range(F) if f not in learned]
    if layer not in (None, "gauss"):
        raise ValueError(f"layer is None or 'gauss', not {layer!r}")
    if layer == "gauss" and gamma is None:
        gamma = 1.0
    select = select or ("valid" if len(va) else "cv")
    alpha_v = np.broadcast_to(np.asarray(alpha, float), (F,)).astype(float).copy()
    h_v = np.ones(F) if h is None else np.broadcast_to(np.asarray(h, float), (F,)).astype(float)
    scale = np.asarray(graph["scale"], float)
    rows = np.concatenate([tr, va])
    n_tr = len(tr)
    y_t = torch.as_tensor(y[tr], dtype=f64, device=dev)
    y_va = torch.as_tensor(y[va], dtype=f64, device=dev)
    var = float(np.var(y[tr])) or 1.0
    fold_idx = [torch.as_tensor(k, dtype=torch.long, device=dev)
                for k in L._kfold(n_tr, folds, seed)]
    tr_idx = torch.arange(n_tr, device=dev)
    va_idx = torch.arange(n_tr, n_tr + len(va), device=dev)
    order = learned + fixed                           # the fields as the kernels are stacked
    h_t = torch.as_tensor(h_v[order], dtype=f64, device=dev)

    # the atoms of the rows, once on the device; the centre keeps single precision exact enough
    centre = [np.asarray(graph["xyz"][f], float)[land].reshape(-1, 3).mean(axis=0)
              for f in range(F)]
    blocks = {}

    def block(f, which):
        key = (f, id(which))
        if key not in blocks:
            blocks[key] = (
                torch.as_tensor(np.asarray(graph["xyz"][f], float)[which] - centre[f],
                                dtype=dt, device=dev),
                torch.as_tensor(np.asarray(graph["c"][f], float)[which], dtype=dt, device=dev))
        return blocks[key]

    counters = {"pass_seconds": 0.0}

    def kernels_of(f, amp_rows, G=None, Gd=None):
        """The pass of field `f` over the rows, forward or backward.  The backward pass
        covers the training rows only: the criterion does not depend on the other rows."""
        t0 = time.time()
        xp, cp = block(f, rows)
        a = amp_rows.detach().to(dt)
        xq, aq, cq = xp[pos_land], a[pos_land], cp[pos_land]
        if G is None:
            out = _field_pass(xp, a, cp, xq, aq, cq, float(alpha_v[f]), kinds[f], budget)
        else:
            gp = torch.zeros_like(a)
            gp[:n_tr], gq = _field_pass(xp[:n_tr], a[:n_tr], cp[:n_tr], xq, aq, cq,
                                        float(alpha_v[f]), kinds[f], budget, G[:n_tr],
                                        Gd[:n_tr])
            out = (gp, gq)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        counters["pass_seconds"] += time.time() - t0
        return out

    with torch.no_grad():
        fixed_K = [kernels_of(f, _amplitude(graph, f, rows, None, 1.0, dev)[0]) for f in fixed]

    net = _network(graph["x"].shape[1], hidden, rounds, len(learned), seed).to(dev) \
        if learned else None
    params = list(net.parameters()) if net is not None else []
    n_net = sum(p.numel() for p in params)
    trains_lam = bool(learn_lam)
    x0 = np.concatenate([
        torch.nn.utils.parameters_to_vector(params).detach().cpu().numpy() if n_net
        else np.zeros(0), [math.log(lam)] if trains_lam else []])
    lower = np.concatenate([x0[:n_net] - WEIGHT_BOUND,
                            [math.log(L.LAM_BOUNDS[0])] if trains_lam else []])
    upper = np.concatenate([x0[:n_net] + WEIGHT_BOUND,
                            [math.log(L.LAM_BOUNDS[1])] if trains_lam else []])
    state = {"med": None}
    history: List[dict] = []
    best = {"score": math.inf, "x": x0.copy()}
    lowest = {"score": math.inf, "x": x0.copy()}

    def evaluate(x, want_grad=True, record=True):
        x = np.asarray(x, float)
        if n_net:
            torch.nn.utils.vector_to_parameters(
                torch.as_tensor(x[:n_net], dtype=f64, device=dev), params)
            for p in params:
                p.grad = None
        ll = torch.as_tensor([x[n_net] if trains_lam else math.log(lam)], dtype=f64,
                             device=dev).requires_grad_(trains_lam and want_grad)
        grad_on = want_grad and len(x) > 0
        amps, pen, sizes = [], None, {}
        with torch.set_grad_enabled(grad_on and bool(n_net)):
            delta = _delta(net, graph, dev) if net is not None else None
            for k, f in enumerate(learned):
                a, d, mask = _amplitude(graph, f, rows, delta[:, k], scale[f], dev)
                amps.append(a)
                msq = (d[:n_tr] ** 2).sum() / mask[:n_tr].sum().clamp(min=1.0)
                sizes[names[f]] = float(torch.sqrt(msq.detach()))
                pen = msq if pen is None else pen + msq
            if pen is not None:
                pen = float(penalty) * pen / len(learned)
        with torch.no_grad():
            passes = [kernels_of(f, a) for f, a in zip(learned, amps)]
        leaves = [(K.requires_grad_(grad_on), d.requires_grad_(grad_on)) for K, d in passes]
        with torch.set_grad_enabled(grad_on):
            Kall = torch.stack([K for K, _d in leaves] + [K for K, _d in fixed_K])
            dall = torch.stack([d for _K, d in leaves] + [d for _K, d in fixed_K])
            Kz = Kall[:, pos_land, :]
            kern = L.Kernels(alpha=alpha_v[order], K=Kall, Kzz=0.5 * (Kz + Kz.transpose(1, 2)),
                             dx=dall, dz=dall[:, pos_land])
            Kxz, Kzz, kxx, kzz = L._joint(kern, h_t, None, normalize)
            if layer == "gauss":
                if state["med"] is None:
                    state["med"] = L._median_d2(Kzz.detach(), kzz.detach())
                Kxz, Kzz = L._layer(Kxz, Kzz, kxx, kzz, float(gamma), state["med"])
            lam_t = torch.exp(ll)[0]
            pred_cv, (G, s, v, n, sy) = L._cv_predictions(Kxz[tr_idx], y_t, Kzz, lam_t,
                                                         fold_idx)
            loss = ((pred_cv - y_t) ** 2).mean() / var
        grad = np.zeros(len(x))
        if grad_on:
            loss.backward()
            if n_net:
                surrogate = pen
                with torch.no_grad():
                    back = [kernels_of(f, a, K.grad, d.grad)
                            for f, a, (K, d) in zip(learned, amps, leaves)]
                for a, (gp, gq) in zip(amps, back):
                    g = gp.to(f64)
                    g[pos_land] += gq.to(f64)
                    surrogate = surrogate + (a * g).sum()
                surrogate.backward()
                grad[:n_net] = torch.nn.utils.parameters_to_vector(
                    [p.grad for p in params]).cpu().numpy()
            if trains_lam:
                grad[n_net] = float(ll.grad[0])
        with torch.no_grad():
            c, b = L._solve(G.detach(), s.detach(), v.detach(), n, sy, Kzz.detach(),
                            lam_t.detach())
            fitted = Kxz.detach() @ c + b
            rmse_va = float(torch.sqrt(((fitted[va_idx] - y_va) ** 2).mean())) \
                if len(va) else float("nan")
        cv = float(loss.detach())
        value = cv + (float(pen.detach()) if pen is not None else 0.0)
        score = rmse_va if select == "valid" else value
        if record:
            history.append({"evaluation": len(history), "cv": cv, "objective": value,
                            "valid_rmse": rmse_va, "lam": float(lam_t.detach()),
                            "correction": sizes, "gradient_norm": float(np.linalg.norm(grad)),
                            "seconds": time.time() - t_start})
            if np.isfinite(value) and value < lowest["score"]:
                lowest.update(score=value, x=x.copy())
            if np.isfinite(score) and score < best["score"]:
                best.update(score=score, x=x.copy())
            say(f"    evaluation {len(history) - 1}: criterion {cv:.5f}, objective "
                f"{value:.5f}, validation RMSE {rmse_va:.5f}")
        extra = {"c": c, "b": b, "amps": [a.detach() for a in amps], "sizes": sizes,
                 "dz": dall.detach()[:, pos_land]}
        return value, grad, extra

    first_grad = evaluate(x0, want_grad=len(x0) > 0)[1]
    start = dict(history[0])
    start["stop"] = "nothing trained"
    start["weights"] = int(n_net)
    if len(x0) and steps > 0:
        # L-BFGS-B takes the gradient itself as its first step when every number has bounds;
        # the numbers are scaled so that this step has the length `first_step` (see
        # cmfqsar.learnable.train for the fault that a small gradient causes otherwise)
        norm = float(start["gradient_norm"])
        sigma = math.sqrt(float(first_step) / norm) if norm > 1e-12 else 1.0

        first = {"value": start["objective"], "grad": first_grad}

        def fun(z):
            if first is not None and not np.any(z):
                return first["value"], sigma * first["grad"]     # the start, already known
            value, grad, _ = evaluate(np.clip(x0 + sigma * np.asarray(z, float), lower, upper))
            return value, sigma * grad
        res = minimize(fun, np.zeros(len(x0)), jac=True, method="L-BFGS-B",
                       bounds=list(zip((lower - x0) / sigma, (upper - x0) / sigma)),
                       options={"maxiter": int(steps), "maxfun": int(steps + 10), "maxls": 10,
                                "ftol": 1e-6, "gtol": 1e-8})
        start["stop"] = str(res.message)
    want = np.arange(graph["n"]) if predict is None else np.asarray(predict, int)

    def finish(x) -> TrainedAmplitudes:
        """The model fitted with the numbers `x`, and its predictions."""
        _value, _grad, extra = evaluate(x, want_grad=False, record=False)
        lam_x = float(np.exp(x[n_net])) if trains_lam else float(lam)
        dz = np.zeros((F, len(land)))
        dz[order] = extra["dz"].to(f64).cpu().numpy()
        if normalize == "cosine":
            norms = np.where(dz > TINY, 1.0 / np.sqrt(np.maximum(dz, TINY)), 0.0)
        else:
            norms = np.broadcast_to(1.0 / np.maximum(dz.mean(axis=1, keepdims=True), TINY),
                                    dz.shape).copy()
        land_a = [None] * F
        for f in fixed:
            land_a[f] = np.asarray(graph["w"][f], float)[land]
        for f, a in zip(learned, extra["amps"]):
            land_a[f] = a[pos_land].cpu().numpy()
        out = TrainedAmplitudes(
            names=names, kinds=kinds, learned=[names[f] for f in learned], alpha=alpha_v.copy(),
            h=h_v.copy(), lam=lam_x, gamma=None if layer is None else float(gamma),
            med=float(state["med"]) if state["med"] is not None else 1.0, normalize=normalize,
            layer=layer, landmarks=land, coef=extra["c"].cpu().numpy(), bias=float(extra["b"]),
            pred=np.full(graph["n"], np.nan), scale=scale.copy(),
            sizes={"inputs": int(graph["x"].shape[1]), "hidden": int(hidden),
                   "rounds": int(rounds)},
            state={k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
            if net is not None else {},
            land={"xyz": [np.asarray(graph["xyz"][f], float)[land] for f in range(F)],
                  "c": [np.asarray(graph["c"][f], float)[land] for f in range(F)],
                  "a": land_a},
            dz=dz, norms=norms, correction=dict(extra["sizes"]), penalty=float(penalty),
            history=history, start=start)
        if len(want):
            out.pred[want] = _predict(out, graph, want, device=dev, dtype=dtype, budget=budget)
        return out

    out = finish(best["x"])
    if keep_criterion:
        same = np.array_equal(best["x"], lowest["x"])
        out.by_criterion = out if same else finish(lowest["x"])
    for res_ in (out, out.by_criterion):
        if res_ is not None:
            res_.evaluations = len(history)
            res_.seconds = time.time() - t_start
            res_.pass_seconds = counters["pass_seconds"]
    return out


# --------------------------------------------------------------------------- #
# The trained model on any molecules, and its reading
# --------------------------------------------------------------------------- #

def amplitudes(trained: TrainedAmplitudes, graph, rows=None, device=None) -> List[np.ndarray]:
    """The amplitudes of every field for the molecules `rows` of `graph`: one (molecules,
    atoms) array per field, in the layout of the packed blocks.  A field that the network
    does not correct returns its fixed amplitudes."""
    L.require()
    dev = L._device(device)
    rows = np.arange(graph["n"]) if rows is None else np.asarray(rows, int)
    out = [np.asarray(graph["w"][f], float)[rows] for f in range(len(trained.names))]
    if trained.learned:
        with torch.no_grad():
            delta = _delta(_net_of(trained, dev), graph, dev)
            for k, name in enumerate(trained.learned):
                f = trained.names.index(name)
                out[f] = _amplitude(graph, f, rows, delta[:, k], trained.scale[f],
                                    dev)[0].cpu().numpy()
    return out


def atom_weights(trained: TrainedAmplitudes, graph, weights, columns=None,
                 device=None) -> np.ndarray:
    """A copy of the atom weights of a table with the trained amplitudes written into it.

    `weights` is ``table.weights`` of the table that `graph` was packed from, (atoms, fields
    of the table); `columns` are the columns of the fields of the model (default 0, 1, 2,
    ...).  The result can replace the weights of the atoms anywhere in the program.
    """
    W = np.array(weights, float, copy=True)
    cols = list(range(len(trained.names))) if columns is None else [int(c) for c in columns]
    amps = amplitudes(trained, graph, device=device)
    for name in trained.learned:
        f = trained.names.index(name)
        atom = np.asarray(graph["atom"][f])
        W[atom[atom >= 0], cols[f]] = amps[f][atom >= 0]
    return W


def predict(trained: TrainedAmplitudes, graph, rows=None, device=None, dtype=None,
            budget: float = L.BUDGET) -> np.ndarray:
    """The predictions of a trained model for the molecules `rows` of `graph`.

    `graph` may be the training table or the table of other molecules packed with
    ``pack_graph(..., like=<graph of the training table>)``.
    """
    L.require()
    dev = L._device(device)
    dt = L._store_dtype(dev, dtype)
    f64 = torch.float64
    rows = np.arange(graph["n"]) if rows is None else np.asarray(rows, int)
    F = len(trained.names)
    amps = amplitudes(trained, graph, rows, device=dev)
    K, d = [], []
    with torch.no_grad():
        for f in range(F):
            centre = trained.land["xyz"][f].reshape(-1, 3).mean(axis=0)

            def t(a):
                return torch.as_tensor(np.asarray(a, float), dtype=dt, device=dev)
            k, s = _field_pass(t(np.asarray(graph["xyz"][f], float)[rows] - centre), t(amps[f]),
                               t(np.asarray(graph["c"][f], float)[rows]),
                               t(trained.land["xyz"][f] - centre), t(trained.land["a"][f]),
                               t(trained.land["c"][f]), float(trained.alpha[f]),
                               trained.kinds[f], budget)
            K.append(k)
            d.append(s)
        kern = L.Kernels(alpha=trained.alpha, K=torch.stack(K), Kzz=None, dx=torch.stack(d),
                         dz=torch.as_tensor(trained.dz, dtype=f64, device=dev))
        h_t = torch.as_tensor(trained.h, dtype=f64, device=dev)
        Kxz, _zz, kxx, kzz = L._joint(kern, h_t, None, trained.normalize, want_zz=False)
        if trained.layer == "gauss":
            Kxz, _ = L._layer(Kxz, None, kxx, kzz, float(trained.gamma), float(trained.med))
        c = torch.as_tensor(trained.coef, dtype=f64, device=dev)
        return (Kxz @ c + trained.bias).cpu().numpy()


def cofield_gaussians(trained: TrainedAmplitudes, f: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centres, coefficients, widths) of the co-field of field `f` of a trained linear model.

    The three arrays have the form of :func:`cmfqsar.lowrank.cofield_gaussians`, so
    :func:`cmfqsar.cofields.evaluate_gaussians` draws the map and
    :func:`cmfqsar.cofields.export_cube` writes it.  The coefficients hold the trained
    amplitudes of the landmark atoms.  The overlap integral of this field with field `f` of a
    molecule, built with the trained amplitudes of that molecule, is the part of the
    prediction that field `f` carries, before the division by the molecule's own norm under
    the cosine convention.
    """
    if trained.layer is not None:
        raise ValueError("a model with the Gaussian kernel layer is not linear in the fields "
                         "and has no co-field")
    f = int(f)
    a = np.asarray(trained.land["a"][f], float)
    keep = a != 0.0
    per_landmark = trained.coef * float(trained.h[f]) * trained.norms[f]
    coefficients = (per_landmark[:, None] * a)[keep]
    c = np.asarray(trained.land["c"][f], float)[keep]
    return trained.land["xyz"][f][keep], coefficients, float(trained.alpha[f]) * c


#: the function above under a name that the argument `predict` of :func:`train` does not hide
_predict = predict


# --------------------------------------------------------------------------- #
# A graph in a file
# --------------------------------------------------------------------------- #

def save_graph(path: str, graph: Dict[str, object], **extra) -> None:
    """Write the result of :func:`pack_graph` to a NumPy ``.npz`` file; `extra` arrays are
    stored beside it."""
    arrays = {f"atom_{f}": np.asarray(a, np.int64) for f, a in enumerate(graph["atom"])}
    arrays.update(graph_x=np.asarray(graph["x"], np.float64),
                  graph_edges=np.asarray(graph["edges"], np.int64),
                  graph_offsets=np.asarray(graph["offsets"], np.int64),
                  graph_scale=np.asarray(graph["scale"], np.float64),
                  graph_types=np.array(list(graph["types"])),
                  graph_columns_x=np.array(list(graph["columns_x"])))
    L.save_packed(path, graph, **arrays, **extra)


def load_graph(path: str) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Read :func:`save_graph`: (the graph, the extra arrays)."""
    graph, extra = L.load_packed(path)
    F = len(graph["names"])
    graph["atom"] = [np.asarray(extra.pop(f"atom_{f}"), np.int64) for f in range(F)]
    graph["x"] = np.asarray(extra.pop("graph_x"), float)
    graph["edges"] = np.asarray(extra.pop("graph_edges"), np.int64)
    graph["offsets"] = np.asarray(extra.pop("graph_offsets"), np.int64)
    graph["scale"] = np.asarray(extra.pop("graph_scale"), float)
    graph["types"] = [str(v) for v in extra.pop("graph_types")]
    graph["columns_x"] = [str(v) for v in extra.pop("graph_columns_x")]
    return graph, extra
