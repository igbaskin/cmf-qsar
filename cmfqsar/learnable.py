"""A learnable CMF layer: the width and the weight of every field trained by gradient (item ND2).

A CMF model fixes two things before it sees an activity: the attenuation ``alpha`` of every
field, which sets the width of the Gaussian on every atom, and the mixing weight ``h`` of every
field in the joint kernel ``K = sum_f h_f K_f``.  The program chooses the attenuation from a
grid of a few values and leaves the weights equal.  This module makes both continuous
parameters and trains them by gradient, with the ridge and, where it is used, the width of the
Gaussian kernel layer.

What is trained and what is not
-------------------------------
The model stays the kernel model.  Its landmarks ``Z`` are molecules of the training part, and
its prediction is

    y(x) = sum_z c_z K(x, z) + b,        K(x, z) = sum_f h_f K_f(x, z; alpha_f).

The coefficients ``c`` and the bias are not trained by gradient.  They are the closed-form
ridge solution for the current ``alpha`` and ``h``, so the model at every step is the model the
rest of the program would fit with these settings.  Only the settings move: one ``log alpha``
and one weight per field, the ridge, and the layer width.  The weights are written
``h = T softmax(u)``, so that their sum ``T`` stays what it was and the ridge keeps its meaning.

Because the model is linear in the fields, its regression-coefficient field (the co-field) is
kept: :func:`cofield_gaussians` returns it as atom-centred Gaussians, in the form
:func:`cmfqsar.cofields.evaluate_gaussians` draws and :func:`cmfqsar.cofields.export_cube`
writes.  :func:`fieldset_of` writes the trained settings into a field set, so that every other
part of the program can build its own model from them.

The objective
-------------
The settings are chosen by the criterion the grid search uses: the squared error of the
out-of-fold predictions inside the training part (``folds`` folds).  The ridge solution is a
differentiable function of the kernel, and the kernel is an analytic function of ``alpha``, so
the criterion has an exact gradient.  A quasi-Newton method (L-BFGS-B) minimises it.  Every
evaluation is exact: the field kernels are recomputed at the attenuations it asks for.  When
validation rows are given, the evaluation with the lowest validation error is kept, which is
early stopping; without them the one with the lowest criterion is kept.

How the gradient reaches the attenuation
----------------------------------------
A field kernel is a sum over atom pairs of ``w_i w_j E(d_ij)``, and for an atom-centred
Gaussian field

    E = (2 pi / (alpha s))^(3/2) exp(-alpha t d^2),   s = c_i + c_j,   t = c_i c_j / (2 s),

with ``c = 1`` for an ordinary field and ``c = 1 / R^2`` for the steric field.  Its derivative
is ``dE / dlog alpha = -E (3/2 + alpha t d^2)``, so the kernel and its derivative come from the
same pass over the atom pairs, with one more sum.  No graph of the atom pairs is kept.  The
pass returns ``K_f`` and ``dK_f / dlog alpha_f`` as two matrices, and the rest of the
computation sees ``K_f + dK_f (theta_f - theta_f0)`` with ``theta = log alpha``: the value is
exact at ``theta0`` and so is the gradient.

A network on the layer
----------------------
:func:`train_network` puts a multilayer network on the kernel rows of the layer, optionally
joined with other descriptors of the molecule, and trains the network and the settings of the
layer together by stochastic gradient descent.  Here the kernel rows are recomputed only when
an attenuation has moved by more than ``trust`` in ``log alpha``; between two such passes the
first-order form above is used.  A last phase holds the settings fixed, recomputes the kernels
exactly and trains the network alone, so that the reported model uses exact kernels.  A
network has no co-field; this route is the comparison, not the model to read.

PyTorch is needed (the optional dependency of :mod:`cmfqsar.fieldae`).  Without it the module
imports, :func:`available` is false and :func:`pack` still works.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                                    # pragma: no cover - import guard
    import torch
    HAVE_TORCH = True
except ImportError:                                     # pragma: no cover
    torch = None
    HAVE_TORCH = False

INSTALL = ("the learnable layer needs PyTorch, which is not installed; install it with "
           "pip install \"cmfqsar[fieldae]\" (the processor build is enough for small sets)")
KINDS = ("gauss", "steric")
LEARNABLE = ("alpha", "weights", "lam", "gamma")
#: bounds of the trained settings: attenuation, ridge, layer width and the weight logits
ALPHA_BOUNDS = (1e-3, 1e2)
LAM_BOUNDS = (1e-9, 1e4)
GAMMA_BOUNDS = (1e-4, 1e3)
LOGIT_BOUND = 8.0
#: atom pairs held at once in one pass (elements of one temporary array)
BUDGET = 4e7
JITTER = 1e-10
#: length of the first step of the training, in the logarithms of the trained numbers
FIRST_STEP = 0.5
TINY = 1e-300


def available() -> bool:
    """Whether PyTorch is installed, so that the layer can be trained."""
    return HAVE_TORCH


def require() -> None:
    """Raise with an installation message when PyTorch is missing."""
    if not HAVE_TORCH:
        raise ImportError(INSTALL)


# --------------------------------------------------------------------------- #
# The atoms of a table, one padded block per field
# --------------------------------------------------------------------------- #

def pack(table, specs, columns: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """The atoms of every molecule of `table`, as one padded block per field.

    A field uses only the atoms that carry weight in it.  For field ``f`` the result holds
    ``xyz[f]`` (molecules, atoms, 3), ``w[f]`` (molecules, atoms) and ``c[f]`` (molecules,
    atoms), padded with zero weights.  ``c`` is 1 for an ordinary field and ``1 / R^2`` for the
    steric field, so that the Gaussian exponent of an atom is ``alpha c``.

    `table` needs ``coords``, ``weights``, ``radii`` and ``offsets`` only
    (:class:`cmfqsar.fields.AtomTable`); `specs` are the fields, each with ``name`` and
    ``kind``; `columns` are their columns in ``table.weights`` (default: 0, 1, 2, ...).
    """
    coords = np.asarray(table.coords, float)
    W = np.asarray(table.weights, float)
    radii = np.asarray(table.radii, float)
    offsets = np.asarray(table.offsets, int)
    n = len(offsets) - 1
    mol_of = np.repeat(np.arange(n), np.diff(offsets))
    cols = list(range(len(specs))) if columns is None else [int(c) for c in columns]
    out = {"n": int(n), "names": [str(s.name) for s in specs],
           "kinds": [str(s.kind) for s in specs], "xyz": [], "w": [], "c": []}
    for spec, col in zip(specs, cols):
        if spec.kind not in KINDS:
            raise ValueError(f"the learnable layer has no kernel for the field kind "
                             f"{spec.kind!r} of {spec.name!r}; it has {', '.join(KINDS)}")
        keep = np.flatnonzero(W[:, col] != 0.0)
        m = mol_of[keep]
        counts = np.bincount(m, minlength=n)
        width = max(int(counts.max()) if len(keep) else 0, 1)
        start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        rank = np.arange(len(keep)) - start[m]
        xyz = np.zeros((n, width, 3))
        w = np.zeros((n, width))
        c = np.ones((n, width))
        xyz[m, rank] = coords[keep]
        w[m, rank] = W[keep, col]
        if spec.kind == "steric":
            c[m, rank] = 1.0 / radii[keep] ** 2
        out["xyz"].append(xyz)
        out["w"].append(w)
        out["c"].append(c)
    return out


def subset(packed: Dict[str, object], rows: Sequence[int]) -> Dict[str, object]:
    """The packed atoms of the molecules `rows`."""
    idx = np.asarray(rows, int)
    return {"n": int(len(idx)), "names": list(packed["names"]), "kinds": list(packed["kinds"]),
            "xyz": [a[idx] for a in packed["xyz"]], "w": [a[idx] for a in packed["w"]],
            "c": [a[idx] for a in packed["c"]]}


def save_packed(path: str, packed: Dict[str, object], **extra) -> None:
    """Write packed atoms to a NumPy ``.npz`` file; `extra` arrays are stored beside them."""
    arrays = {"names": np.array(packed["names"]), "kinds": np.array(packed["kinds"])}
    for f in range(len(packed["names"])):
        arrays[f"xyz_{f}"] = np.asarray(packed["xyz"][f], np.float64)
        arrays[f"w_{f}"] = np.asarray(packed["w"][f], np.float64)
        arrays[f"c_{f}"] = np.asarray(packed["c"][f], np.float64)
    arrays.update(extra)
    np.savez_compressed(path, **arrays)


def load_packed(path: str) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Read :func:`save_packed`: (the packed atoms, the extra arrays)."""
    z = np.load(path, allow_pickle=False)
    names = [str(v) for v in z["names"]]
    F = len(names)
    packed = {"names": names, "kinds": [str(v) for v in z["kinds"]],
              "xyz": [np.asarray(z[f"xyz_{f}"], float) for f in range(F)],
              "w": [np.asarray(z[f"w_{f}"], float) for f in range(F)],
              "c": [np.asarray(z[f"c_{f}"], float) for f in range(F)]}
    packed["n"] = int(packed["w"][0].shape[0]) if F else 0
    used = {"names", "kinds"} | {f"{k}_{f}" for k in ("xyz", "w", "c") for f in range(F)}
    return packed, {k: z[k] for k in z.files if k not in used}


# --------------------------------------------------------------------------- #
# The field kernels and their derivative in log alpha
# --------------------------------------------------------------------------- #

def _device(device=None):
    require()
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _store_dtype(dev, dtype=None):
    if dtype is not None:
        return dtype
    return torch.float32 if dev.type == "cuda" else torch.float64


def _overlap(d2, alpha: float, ci=None, cj=None):
    """(E, g) of every atom pair: E the overlap and g with dE / dlog alpha = -E (3/2 + g)."""
    if ci is None:
        g = d2 * (0.25 * alpha)
        E = torch.exp(-g) * (math.pi / alpha) ** 1.5
        return E, g
    s = ci + cj
    g = d2 * (alpha * ci * cj / (2.0 * s))
    z = (2.0 * math.pi / alpha) / s
    E = torch.exp(-g) * (z * torch.sqrt(z))
    return E, g


def _cross(xp, wp, cp, xq, wq, cq, alpha: float, kind: str, derivative: bool):
    """K and dK / dlog alpha between the molecules of two padded blocks: (b, m) each.

    Two arrays of (atoms of the block) by (atoms of the landmarks) are held: the exponent ``g``
    and the overlap ``E``.  Everything else is done in place, and the sums over the atoms of a
    landmark are one batched product with its weights.
    """
    b, A = wp.shape
    m, B = wq.shape
    P = xp.reshape(b * A, 3)
    Q = xq.reshape(m * B, 3)
    g = torch.mm(P, Q.T)
    g.mul_(-2.0).add_((P * P).sum(1)[:, None]).add_((Q * Q).sum(1)[None, :]).clamp_(min=0.0)
    if kind == "steric":
        ci, cj = cp.reshape(b * A, 1), cq.reshape(1, m * B)
        s = ci + cj
        g.mul_(alpha * ci * cj / (2.0 * s))
        z = (2.0 * math.pi / alpha) / s
        E = g.neg().exp_().mul_(z * torch.sqrt(z))
        del z, s
    else:
        g.mul_(0.25 * alpha)
        E = g.neg().exp_().mul_((math.pi / alpha) ** 1.5)
    wp_ = wp.reshape(b * A, 1)
    K = (torch.einsum("izj,zj->iz", E.view(b * A, m, B), wq) * wp_).view(b, A, m).sum(1)
    if not derivative:
        return K, None
    E.mul_(g)
    S = (torch.einsum("izj,zj->iz", E.view(b * A, m, B), wq) * wp_).view(b, A, m).sum(1)
    return K, -1.5 * K - S


def _self(x, w, c, alpha: float, kind: str, derivative: bool):
    """The self-kernel of every molecule of one padded block and its derivative: (b,) each."""
    d2 = ((x[:, :, None, :] - x[:, None, :, :]) ** 2).sum(-1)
    if kind == "steric":
        E, g = _overlap(d2, alpha, c[:, :, None], c[:, None, :])
    else:
        E, g = _overlap(d2, alpha)
    ww = w[:, :, None] * w[:, None, :]
    E = E * ww
    K = E.sum((1, 2))
    if not derivative:
        return K, None
    return K, -1.5 * K - (E * g).sum((1, 2))


@dataclass
class Kernels:
    """The field kernels of some molecules against the landmarks at given attenuations.

    ``K`` and ``dK`` are (fields, molecules, landmarks), ``Kzz`` and ``dKzz`` are (fields,
    landmarks, landmarks), ``dx`` and ``ddx`` are the self-kernels (fields, molecules).  The
    ``d...`` arrays are the derivatives with respect to ``log alpha`` of the same field, or
    None when the pass was made without them.
    """
    alpha: np.ndarray
    K: object
    Kzz: object
    dx: object
    dz: object
    dK: object = None
    dKzz: object = None
    ddx: object = None
    ddz: object = None
    seconds: float = 0.0


def kernel_pass(packed, rows, landmarks, alpha, derivative: bool = True, device=None,
                dtype=None, budget: float = BUDGET, zz: bool = True) -> Kernels:
    """The kernels of every field between the molecules `rows` and the molecules `landmarks`.

    `packed` is the result of :func:`pack`, `alpha` one attenuation per field.  With
    `derivative` the derivative with respect to ``log alpha`` is returned too.  The pass is made
    a block of molecules at a time, so its memory does not grow with the number of rows; it is
    set by `budget`, the number of atom pairs held at once.  ``zz=False`` leaves out the
    kernels between the landmarks and computes their self-kernels only.
    """
    dev = _device(device)
    dt = _store_dtype(dev, dtype)
    t0 = time.time()
    rows = np.asarray(rows, int)
    land = np.asarray(landmarks, int)
    alpha = np.asarray(alpha, float).ravel()
    F = len(packed["names"])
    n, m = len(rows), len(land)
    K = torch.zeros((F, n, m), dtype=dt, device=dev)
    dK = torch.zeros((F, n, m), dtype=dt, device=dev) if derivative else None
    Kzz = torch.zeros((F, m, m), dtype=dt, device=dev) if zz else None
    dKzz = torch.zeros((F, m, m), dtype=dt, device=dev) if derivative and zz else None
    dx = torch.zeros((F, n), dtype=dt, device=dev)
    ddx = torch.zeros((F, n), dtype=dt, device=dev) if derivative else None
    dz = torch.zeros((F, m), dtype=dt, device=dev)
    ddz = torch.zeros((F, m), dtype=dt, device=dev) if derivative else None
    with torch.no_grad():
        for f in range(F):
            kind = packed["kinds"][f]
            a = float(alpha[f])
            centre = packed["xyz"][f].reshape(-1, 3).mean(axis=0)

            def block(idx):
                return (torch.as_tensor(packed["xyz"][f][idx] - centre, dtype=dt, device=dev),
                        torch.as_tensor(packed["w"][f][idx], dtype=dt, device=dev),
                        torch.as_tensor(packed["c"][f][idx], dtype=dt, device=dev))

            xq, wq, cq = block(land)
            A, B = packed["w"][f].shape[1], wq.shape[1]
            step = max(1, int(budget // max(A * m * B, 1)))
            targets = [(rows, K, dK)] + ([(land, Kzz, dKzz)] if zz else [])
            for idx, out, dout in targets:
                for lo in range(0, len(idx), step):
                    xp, wp, cp = block(idx[lo:lo + step])
                    k, dk = _cross(xp, wp, cp, xq, wq, cq, a, kind, derivative)
                    out[f, lo:lo + len(k)] = k
                    if derivative:
                        dout[f, lo:lo + len(k)] = dk
            if zz:
                Kzz[f] = 0.5 * (Kzz[f] + Kzz[f].T)
                if derivative:
                    dKzz[f] = 0.5 * (dKzz[f] + dKzz[f].T)
            step = max(1, int(budget // max(A * A, 1)))
            for idx, out, dout in ((rows, dx, ddx), (land, dz, ddz)):
                for lo in range(0, len(idx), step):
                    xp, wp, cp = block(idx[lo:lo + step])
                    k, dk = _self(xp, wp, cp, a, kind, derivative)
                    out[f, lo:lo + len(k)] = k
                    if derivative:
                        dout[f, lo:lo + len(k)] = dk
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return Kernels(alpha=alpha.copy(), K=K, Kzz=Kzz, dx=dx, dz=dz, dK=dK, dKzz=dKzz,
                   ddx=ddx, ddz=ddz, seconds=time.time() - t0)


# --------------------------------------------------------------------------- #
# The joint kernel, the Gaussian layer and the ridge solution
# --------------------------------------------------------------------------- #

def _first_order(value, slope, delta_f):
    return value if slope is None or delta_f is None else value + slope * delta_f


def _joint(kern: Kernels, h, delta=None, normalize: str = "cosine", rows=None,
           want_zz: bool = True):
    """The joint kernel in double precision: (K_xz, K_zz, k_xx, k_zz).

    `h` holds one weight per field, `delta` the move of every ``log alpha`` away from the
    attenuations of the pass (zeros that carry a gradient), `rows` a selection of molecules.
    ``k_xx`` and ``k_zz`` are the joint self-kernels, which the Gaussian layer needs.
    """
    F = kern.K.shape[0]
    f64 = torch.float64
    Kxz = Kzz = kxx = kzz = None
    for f in range(F):
        d_f = None if delta is None else delta[f]
        K0 = kern.K[f] if rows is None else kern.K[f][rows]
        dK0 = None if kern.dK is None else (kern.dK[f] if rows is None else kern.dK[f][rows])
        dx0 = kern.dx[f] if rows is None else kern.dx[f][rows]
        ddx0 = None if kern.ddx is None else (kern.ddx[f] if rows is None else kern.ddx[f][rows])
        K = _first_order(K0.to(f64), None if dK0 is None else dK0.to(f64), d_f)
        dx = _first_order(dx0.to(f64), None if ddx0 is None else ddx0.to(f64), d_f)
        dz0 = kern.dz[f].to(f64)
        dz = _first_order(dz0, None if kern.ddz is None else kern.ddz[f].to(f64), d_f)
        if normalize == "cosine":
            ix = torch.where(dx0.to(f64) > TINY, dx.clamp(min=TINY).rsqrt(), torch.zeros_like(dx))
            iz = torch.where(dz0 > TINY, dz.clamp(min=TINY).rsqrt(), torch.zeros_like(dz))
            Kn = K * ix[:, None] * iz[None, :]
            sx = (dx0.to(f64) > TINY).to(f64)
            sz = (dz0 > TINY).to(f64)
        elif normalize == "mean":
            scale = 1.0 / dz.mean().clamp(min=TINY)
            Kn = K * scale
            sx, sz = dx * scale, dz * scale
            ix = iz = None
        else:
            raise ValueError(f"normalize is 'cosine' or 'mean', not {normalize!r}")
        Kxz = h[f] * Kn if Kxz is None else Kxz + h[f] * Kn
        kxx = h[f] * sx if kxx is None else kxx + h[f] * sx
        kzz = h[f] * sz if kzz is None else kzz + h[f] * sz
        if want_zz:
            Z = _first_order(kern.Kzz[f].to(f64),
                             None if kern.dKzz is None else kern.dKzz[f].to(f64), d_f)
            Zn = Z * iz[:, None] * iz[None, :] if normalize == "cosine" else Z * scale
            Kzz = h[f] * Zn if Kzz is None else Kzz + h[f] * Zn
    return Kxz, Kzz, kxx, kzz


def _layer(Kxz, Kzz, kxx, kzz, gamma, med: float):
    """The Gaussian kernel layer on the joint kernel: ``exp(-gamma |x - z|^2 / med)``."""
    D2 = (kxx[:, None] + kzz[None, :] - 2.0 * Kxz).clamp(min=0.0)
    out = torch.exp(-gamma * D2 / med)
    if Kzz is None:
        return out, None
    Dz = (kzz[:, None] + kzz[None, :] - 2.0 * Kzz).clamp(min=0.0)
    return out, torch.exp(-gamma * Dz / med)


def _median_d2(Kzz, kzz) -> float:
    D2 = (kzz[:, None] + kzz[None, :] - 2.0 * Kzz).clamp(min=0.0)
    m = D2.shape[0]
    off = D2[~torch.eye(m, dtype=torch.bool, device=D2.device)]
    med = float(off.median()) if off.numel() else 1.0
    return med if med > TINY else 1.0


def _solve(G, s, v, n: float, sy, Kzz, lam, jitter: float = JITTER):
    """The ridge solution in the space of the landmarks, with a bias that is not penalised.

    It minimises ``|K c + b - y|^2 + lam c' K_zz c`` from ``G = K'K``, ``s = K'1``, ``v = K'y``,
    the number of rows `n` and the sum `sy` of their activities.  With every training molecule
    a landmark this is the kernel ridge regression of :func:`cmfqsar.model.krr_fit`.
    """
    A = G - torch.outer(s, s) / n + lam * Kzz
    A = A + (jitter * A.diagonal().mean().detach().clamp(min=TINY)) \
        * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    rhs = (v - s * (sy / n))[:, None]
    L, info = torch.linalg.cholesky_ex(A)
    c = torch.cholesky_solve(rhs, L) if int(info) == 0 else torch.linalg.solve(A, rhs)
    c = c[:, 0]
    return c, (sy - s @ c) / n


def _cv_predictions(K, y, Kzz, lam, folds):
    """Out-of-fold predictions of the ridge model for the rows of `K`."""
    parts = []
    for idx in folds:
        Kk = K[idx]
        parts.append((idx, Kk, Kk.T @ Kk, Kk.sum(0), Kk.T @ y[idx], float(len(idx)),
                      y[idx].sum()))
    G = sum(p[2] for p in parts)
    s = sum(p[3] for p in parts)
    v = sum(p[4] for p in parts)
    n = float(K.shape[0])
    sy = y.sum()
    pred = torch.zeros_like(y)
    for idx, Kk, Gk, sk, vk, nk, syk in parts:
        c, b = _solve(G - Gk, s - sk, v - vk, n - nk, sy - syk, Kzz, lam)
        pred = pred.index_put((idx,), Kk @ c + b)
    return pred, (G, s, v, n, sy)


def _kfold(n: int, folds: int, seed: int) -> List[np.ndarray]:
    order = np.random.RandomState(int(seed)).permutation(n)
    return [np.sort(order[k::folds]) for k in range(max(2, min(int(folds), n)))]


# --------------------------------------------------------------------------- #
# The trained settings
# --------------------------------------------------------------------------- #

@dataclass
class Trained:
    """The settings a training found and the model fitted with them.

    ``alpha`` and ``h`` hold one value per field, ``lam`` is the ridge and ``gamma`` the width
    of the Gaussian layer (None for the linear model).  ``coef`` and ``bias`` are the model
    over the landmarks ``landmarks`` (rows of the packed table).  ``pred`` holds the prediction
    of every molecule of the packed table, ``history`` one record per evaluation.
    """
    names: List[str]
    kinds: List[str]
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
    learned: Tuple[str, ...]
    history: List[dict] = field(default_factory=list)
    start: Dict[str, object] = field(default_factory=dict)
    passes: int = 0
    seconds: float = 0.0
    kernel_seconds: float = 0.0
    norms: Optional[np.ndarray] = None          # (fields, landmarks) factors of the co-field
    #: the model at the lowest criterion, when the training kept it beside the selected one
    by_criterion: Optional["Trained"] = None

    def settings(self) -> dict:
        return {"alpha": dict(zip(self.names, map(float, self.alpha))),
                "h": dict(zip(self.names, map(float, self.h))), "lam": float(self.lam),
                "gamma": None if self.gamma is None else float(self.gamma)}


def _predict_rows(packed, rows, landmarks, alpha, h, normalize, layer, gamma, med, coef, bias,
                  device, dtype, budget, block: int = 2048):
    """Predictions and joint kernel rows of `rows`, with exact kernels."""
    rows = np.asarray(rows, int)
    h_t = None
    pred = np.zeros(len(rows))
    for lo in range(0, len(rows), block):
        kern = kernel_pass(packed, rows[lo:lo + block], landmarks, alpha, derivative=False,
                           device=device, dtype=dtype, budget=budget, zz=False)
        if h_t is None:
            h_t = torch.as_tensor(np.asarray(h, float), dtype=torch.float64, device=kern.K.device)
            c_t = torch.as_tensor(coef, dtype=torch.float64, device=kern.K.device)
        with torch.no_grad():
            Kxz, Kzz, kxx, kzz = _joint(kern, h_t, None, normalize, want_zz=False)
            if layer == "gauss":
                Kxz, _ = _layer(Kxz, None, kxx, kzz, float(gamma), float(med))
            pred[lo:lo + len(Kxz)] = (Kxz @ c_t + bias).cpu().numpy()
    return pred


def train(packed, y, train, landmarks, valid=None, alpha=1.0, h=None, lam: float = 1e-2,
          gamma: Optional[float] = None, learn: Sequence[str] = ("alpha", "weights", "lam"),
          normalize: str = "cosine", layer: Optional[str] = None, steps: int = 40,
          folds: int = 5, seed: int = 0, select: Optional[str] = None, predict=None,
          keep_criterion: bool = False, device=None, dtype=None, budget: float = BUDGET,
          log: Optional[Callable[[str], None]] = None) -> Trained:
    """Train the settings of the fields and fit the ridge model with them.

    `packed` holds the atoms of every molecule (:func:`pack`), `y` one activity per molecule,
    `train` and `valid` rows of the table, `landmarks` rows that carry the model (molecules of
    the training part).  `alpha`, `h`, `lam` and `gamma` are the start: one number, or one per
    field for the first two.  `learn` names what is trained: any of ``"alpha"``, ``"weights"``,
    ``"lam"`` and ``"gamma"`` (the last needs ``layer="gauss"``); an empty `learn` or
    ``steps=0`` fits the model at the start and trains nothing.

    `normalize` is ``"cosine"`` (every field kernel divided by the square roots of the two
    self-kernels, the convention of the program) or ``"mean"`` (every field kernel divided by
    the mean self-kernel of the landmarks).  `steps` bounds the iterations of the quasi-Newton
    method.  `select` is ``"valid"`` (keep the evaluation with the lowest validation error;
    the default when `valid` is given) or ``"cv"`` (keep the lowest criterion).  `predict`
    lists the rows whose predictions are wanted (default: every molecule).  With
    `keep_criterion` the model at the lowest criterion is returned too, as ``by_criterion``
    of the result, so that one training gives the model with and without early stopping.
    ``start`` of the result holds the criterion at the start and its gradient with respect to
    the trained numbers (``log alpha`` of every field, the weight logits, ``log lam``,
    ``log gamma``, in the order of ``start["layout"]``).
    The method makes at most ``steps + 10`` evaluations.
    """
    require()
    from scipy.optimize import minimize

    dev = _device(device)
    t_start = time.time()
    say = log or (lambda _msg: None)
    F = len(packed["names"])
    y = np.asarray(y, float)
    tr = np.asarray(train, int)
    va = np.asarray(valid, int) if valid is not None and len(valid) else np.zeros(0, int)
    land = np.asarray(landmarks, int)
    learn = tuple(k for k in LEARNABLE if k in set(learn))
    if "gamma" in learn and layer != "gauss":
        raise ValueError("the layer width can be trained only with layer='gauss'")
    if layer not in (None, "gauss"):
        raise ValueError(f"layer is None or 'gauss', not {layer!r}")
    if layer == "gauss" and gamma is None:
        gamma = 1.0
    select = select or ("valid" if len(va) else "cv")
    alpha0 = np.broadcast_to(np.asarray(alpha, float), (F,)).astype(float).copy()
    h0 = np.ones(F) if h is None else np.broadcast_to(np.asarray(h, float), (F,)).astype(float)
    total = float(h0.sum())
    u0 = np.log(np.maximum(h0, 1e-12))
    rows = np.concatenate([tr, va])
    n_tr = len(tr)
    f64 = torch.float64
    y_t = torch.as_tensor(y[tr], dtype=f64, device=dev)
    var = float(np.var(y[tr])) or 1.0
    fold_idx = [torch.as_tensor(k, dtype=torch.long, device=dev) for k in _kfold(n_tr, folds, seed)]
    tr_idx = torch.arange(n_tr, device=dev)
    va_idx = torch.arange(n_tr, n_tr + len(va), device=dev)

    # the vector of trained numbers: log alpha, weight logits, log ridge, log layer width
    layout, x0, bounds = [], [], []
    if "alpha" in learn:
        layout.append(("alpha", F))
        x0 += list(np.log(alpha0))
        bounds += [tuple(np.log(ALPHA_BOUNDS))] * F
    if "weights" in learn:
        layout.append(("weights", F))
        x0 += list(u0)
        bounds += [(float(u0.mean()) - LOGIT_BOUND, float(u0.mean()) + LOGIT_BOUND)] * F
    if "lam" in learn:
        layout.append(("lam", 1))
        x0.append(math.log(lam))
        bounds.append(tuple(np.log(LAM_BOUNDS)))
    if "gamma" in learn:
        layout.append(("gamma", 1))
        x0.append(math.log(gamma))
        bounds.append(tuple(np.log(GAMMA_BOUNDS)))
    x0 = np.asarray(x0, float)

    def unpack(x):
        out = {"alpha": np.log(alpha0), "weights": u0, "lam": np.array([math.log(lam)]),
               "gamma": np.array([math.log(gamma if gamma is not None else 1.0)])}
        at = 0
        for key, size in layout:
            out[key] = np.asarray(x[at:at + size], float)
            at += size
        return out

    state = {"kern": None, "theta": None, "passes": 0, "kernel_seconds": 0.0, "med": None}
    history: List[dict] = []
    best = {"score": math.inf, "x": x0.copy()}
    lowest = {"score": math.inf, "x": x0.copy()}

    def kernels_at(theta):
        if state["theta"] is None or not np.array_equal(theta, state["theta"]):
            state["kern"] = None                      # free the previous pass first
            state["kern"] = kernel_pass(packed, rows, land, np.exp(theta),
                                        derivative="alpha" in learn, device=dev, dtype=dtype,
                                        budget=budget)
            state["theta"] = theta.copy()
            state["passes"] += 1
            state["kernel_seconds"] += state["kern"].seconds
        return state["kern"]

    def evaluate(x, want_grad=True, record=True):
        p = unpack(x)
        kern = kernels_at(p["alpha"])
        theta = torch.as_tensor(p["alpha"], dtype=f64, device=dev).requires_grad_("alpha" in learn)
        u = torch.as_tensor(p["weights"], dtype=f64, device=dev).requires_grad_("weights" in learn)
        ll = torch.as_tensor(p["lam"], dtype=f64, device=dev).requires_grad_("lam" in learn)
        lg = torch.as_tensor(p["gamma"], dtype=f64, device=dev).requires_grad_("gamma" in learn)
        delta = theta - theta.detach() if "alpha" in learn else None
        h_t = total * torch.softmax(u, dim=0)
        with torch.set_grad_enabled(want_grad and bool(layout)):
            Kxz, Kzz, kxx, kzz = _joint(kern, h_t, delta, normalize)
            if layer == "gauss":
                if state["med"] is None:
                    state["med"] = _median_d2(Kzz.detach(), kzz.detach())
                Kxz, Kzz = _layer(Kxz, Kzz, kxx, kzz, torch.exp(lg)[0], state["med"])
            lam_t = torch.exp(ll)[0]
            K_tr = Kxz[tr_idx]
            pred_cv, (G, s, v, n, sy) = _cv_predictions(K_tr, y_t, Kzz, lam_t, fold_idx)
            loss = ((pred_cv - y_t) ** 2).mean() / var
        grad = np.zeros(len(x))
        if want_grad and layout:
            loss.backward()
            parts = {"alpha": theta, "weights": u, "lam": ll, "gamma": lg}
            grad = np.concatenate([parts[k].grad.detach().cpu().numpy().ravel()
                                   for k, _size in layout])
        with torch.no_grad():
            c, b = _solve(G.detach(), s.detach(), v.detach(), n, sy, Kzz.detach(),
                          lam_t.detach())
            fitted = Kxz.detach() @ c + b
            rmse_va = float(torch.sqrt(((fitted[va_idx] - torch.as_tensor(
                y[va], dtype=f64, device=dev)) ** 2).mean())) if len(va) else float("nan")
        value = float(loss.detach())
        score = rmse_va if select == "valid" else value
        if record:
            history.append({"evaluation": len(history), "cv": value, "valid_rmse": rmse_va,
                            "gradient": grad.tolist() if want_grad and layout else None,
                            "alpha": [float(a) for a in np.exp(p["alpha"])],
                            "h": [float(a) for a in h_t.detach().cpu().numpy()],
                            "lam": float(lam_t.detach()), "gamma": float(torch.exp(lg.detach())[0])
                            if layer == "gauss" else None,
                            "seconds": time.time() - t_start})
            if np.isfinite(value) and value < lowest["score"]:
                lowest.update(score=value, x=np.asarray(x, float).copy())
            if np.isfinite(score) and score < best["score"]:
                best.update(score=score, x=np.asarray(x, float).copy())
            say(f"    evaluation {len(history) - 1}: criterion {value:.5f}, validation RMSE "
                f"{rmse_va:.5f}, pass {state['passes']}")
        return value, grad, (c, b, fitted)

    evaluate(x0, want_grad=bool(layout))
    start = dict(history[0])
    start["layout"] = [[key, size] for key, size in layout]
    start["stop"] = "nothing trained"
    if layout and steps > 0:
        # With bounds on every number, L-BFGS-B takes the gradient itself as its first step.
        # The gradient of the criterion can be 1e-4; such a step lowers the criterion by less
        # than the stopping tolerance, and the method stops where it started.  The numbers are
        # therefore scaled so that the first step has the length FIRST_STEP in the logarithms.
        norm = float(np.linalg.norm(np.asarray(start["gradient"], float)))
        sigma = math.sqrt(FIRST_STEP / norm) if norm > 1e-12 else 1.0
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        def fun(z):
            value, grad, _ = evaluate(np.clip(x0 + sigma * np.asarray(z, float), lower, upper))
            return value, sigma * grad
        res = minimize(fun, np.zeros(len(x0)), jac=True, method="L-BFGS-B",
                       bounds=list(zip((lower - x0) / sigma, (upper - x0) / sigma)),
                       options={"maxiter": int(steps), "maxfun": int(steps + 10), "maxls": 10,
                                "ftol": 1e-6, "gtol": 1e-8})
        start["stop"] = str(res.message)
    want = np.arange(packed["n"]) if predict is None else np.asarray(predict, int)

    def finish(x) -> Trained:
        """The model fitted with the settings `x`, and its predictions with exact kernels."""
        _value, _grad, (c, b, _fitted) = evaluate(x, want_grad=False, record=False)
        p = unpack(x)
        alpha_x = np.exp(p["alpha"])
        e = np.exp(p["weights"] - p["weights"].max())
        h_x = total * e / e.sum()
        gamma_x = float(np.exp(p["gamma"][0])) if layer == "gauss" else None
        with torch.no_grad():
            dz = state["kern"].dz.to(f64)
            if normalize == "cosine":
                norms = torch.where(dz > TINY, dz.clamp(min=TINY).rsqrt(), torch.zeros_like(dz))
            else:
                norms = (1.0 / dz.mean(dim=1, keepdim=True).clamp(min=TINY)).expand_as(dz)
            norms = norms.cpu().numpy()
        coef, bias = c.cpu().numpy(), float(b)
        med = float(state["med"]) if state["med"] is not None else 1.0
        pred = np.full(packed["n"], np.nan)
        if len(want):
            pred[want] = _predict_rows(packed, want, land, alpha_x, h_x, normalize, layer,
                                       gamma_x, med, coef, bias, dev, dtype, budget)
        return Trained(names=list(packed["names"]), kinds=list(packed["kinds"]), alpha=alpha_x,
                       h=h_x, lam=float(np.exp(p["lam"][0])), gamma=gamma_x, med=med,
                       normalize=normalize, layer=layer, landmarks=land, coef=coef, bias=bias,
                       pred=pred, learned=learn, history=history, start=start, norms=norms)

    out = finish(best["x"])
    if keep_criterion:
        same = np.array_equal(best["x"], lowest["x"])
        out.by_criterion = out if same else finish(lowest["x"])
    state["kern"] = None
    for res in (out, out.by_criterion):
        if res is not None:
            res.passes = state["passes"]
            res.seconds = time.time() - t_start
            res.kernel_seconds = state["kernel_seconds"]
    return out


# --------------------------------------------------------------------------- #
# Back to the objects of the program: the field set and the co-field
# --------------------------------------------------------------------------- #

def fieldset_of(trained: Trained, fieldset):
    """A copy of `fieldset` with the trained attenuation and weight of every field."""
    out = fieldset.copy()
    by_name = {n: (float(a), float(w)) for n, a, w in zip(trained.names, trained.alpha, trained.h)}
    for spec in out.specs:
        if spec.name in by_name:
            spec.alpha, spec.h = by_name[spec.name]
    return out


def cofield_gaussians(trained: Trained, landmarks_table, f: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centres, coefficients, widths) of the co-field of field `f` of a trained linear model.

    `landmarks_table` is the atom table of the landmark molecules, in the order of
    ``trained.landmarks``.  The three arrays have the form of
    :func:`cmfqsar.lowrank.cofield_gaussians`, so :func:`cmfqsar.cofields.evaluate_gaussians`
    draws the map and :func:`cmfqsar.cofields.export_cube` writes it.  The overlap integral of
    this field with field `f` of a molecule is the part of the prediction that field `f`
    carries, before the division by the molecule's own norm under the cosine convention.
    """
    if trained.layer is not None:
        raise ValueError("a model with the Gaussian kernel layer is not linear in the fields "
                         "and has no co-field")
    f = int(f)
    Z = landmarks_table
    coords = np.asarray(Z.coords, float)
    offsets = np.asarray(Z.offsets, int)
    if len(offsets) - 1 != len(trained.landmarks):
        raise ValueError("the table does not hold the landmarks of this model")
    a = float(trained.alpha[f])
    widths = (np.full(len(coords), a) if trained.kinds[f] == "gauss"
              else a / np.asarray(Z.radii, float) ** 2)
    per_landmark = trained.coef * float(trained.h[f]) * np.asarray(trained.norms, float)[f]
    per_atom = np.repeat(per_landmark, np.diff(offsets))
    return coords, per_atom * np.asarray(Z.weights, float)[:, f], widths


# --------------------------------------------------------------------------- #
# A network on the layer, trained with it
# --------------------------------------------------------------------------- #

def whitener(Kzz: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """``W`` with ``W W' = K_zz^+`` (the same as :func:`cmfqsar.lowrank.whitener`)."""
    A = 0.5 * (np.asarray(Kzz, float) + np.asarray(Kzz, float).T)
    w, V = np.linalg.eigh(A)
    keep = w > max(float(w.max()), 0.0) * float(rcond)
    return V[:, keep] / np.sqrt(w[keep])


@dataclass
class TrainedNetwork:
    """A network trained on the kernel rows of the layer, and the settings of the layer."""
    names: List[str]
    alpha: np.ndarray
    h: np.ndarray
    pred: np.ndarray
    learned: Tuple[str, ...]
    history: List[dict] = field(default_factory=list)
    epochs: int = 0
    passes: int = 0
    seconds: float = 0.0
    kernel_seconds: float = 0.0
    valid_rmse: float = float("nan")

    def settings(self) -> dict:
        return {"alpha": dict(zip(self.names, map(float, self.alpha))),
                "h": dict(zip(self.names, map(float, self.h)))}


def _mlp(n_in: int, hidden: Sequence[int], seed: int):
    torch.manual_seed(int(seed))
    layers, last = [], int(n_in)
    for width in hidden:
        layers += [torch.nn.Linear(last, int(width)), torch.nn.ReLU()]
        last = int(width)
    layers.append(torch.nn.Linear(last, 1))
    return torch.nn.Sequential(*layers).double()


def train_network(packed, y, train, valid, landmarks, alpha=1.0, h=None, extra=None,
                  learn: Sequence[str] = ("alpha", "weights"), normalize: str = "cosine",
                  hidden: Sequence[int] = (128, 64), l2: float = 1e-4, epochs: int = 200,
                  batch: int = 256, lr: float = 1e-3, lr_fields: float = 2e-3,
                  trust: float = 0.1, patience: int = 25, seed: int = 0, predict=None,
                  device=None, dtype=None, budget: float = BUDGET,
                  log: Optional[Callable[[str], None]] = None) -> TrainedNetwork:
    """Train a multilayer network on the kernel rows of the layer, and the layer with it.

    The input of the network is ``(K(x, Z) W0 - mu0) / s0``: the joint kernel of the molecule
    with the landmarks, times the whitening matrix of the start, centred on the training rows
    and divided by one number.  ``W0``, ``mu0`` and ``s0`` are computed at the start and do not
    change, so the input at the start is the scaled CMF vector of plan item ND1.  `extra` is an
    optional (molecules, columns) block of other descriptors, already scaled, that is joined
    to it.  `learn` names what moves in the layer (``"alpha"``, ``"weights"`` or neither); with
    neither, the network alone is trained and the kernels are computed once.

    The kernels are recomputed when an attenuation has moved by more than `trust` in
    ``log alpha`` and in a last phase with the settings held fixed (see the module text).
    Early stopping uses the validation rows, with `patience` epochs.  There are two phases of
    at most `epochs` epochs each, and the second starts from the best state of the first with
    a new optimiser; with a fixed layer both phases train the network alone.
    """
    require()
    dev = _device(device)
    t_start = time.time()
    say = log or (lambda _msg: None)
    f64 = torch.float64
    F = len(packed["names"])
    y = np.asarray(y, float)
    tr = np.asarray(train, int)
    va = np.asarray(valid, int)
    land = np.asarray(landmarks, int)
    learn = tuple(k for k in ("alpha", "weights") if k in set(learn))
    alpha0 = np.broadcast_to(np.asarray(alpha, float), (F,)).astype(float).copy()
    h0 = np.ones(F) if h is None else np.broadcast_to(np.asarray(h, float), (F,)).astype(float)
    total = float(h0.sum())
    rows = np.concatenate([tr, va])
    n_tr, n_va = len(tr), len(va)
    y_mean, y_sd = float(y[tr].mean()), float(y[tr].std()) or 1.0
    y_t = torch.as_tensor((y[rows] - y_mean) / y_sd, dtype=f64, device=dev)
    X = None if extra is None else torch.as_tensor(np.asarray(extra, float), dtype=f64,
                                                   device=dev)
    theta = torch.tensor(np.log(alpha0), dtype=f64, device=dev, requires_grad="alpha" in learn)
    u = torch.tensor(np.log(np.maximum(h0, 1e-12)), dtype=f64, device=dev,
                     requires_grad="weights" in learn)
    counters = {"passes": 0, "kernel_seconds": 0.0}

    def new_pass(which_rows, derivative):
        kern = kernel_pass(packed, which_rows, land, np.exp(theta.detach().cpu().numpy()),
                           derivative=derivative, device=dev, dtype=dtype, budget=budget)
        counters["passes"] += 1
        counters["kernel_seconds"] += kern.seconds
        return kern

    cur = {"kern": new_pass(rows, "alpha" in learn)}
    theta0 = theta.detach().clone()
    with torch.no_grad():
        Kxz, Kzz, _kxx, _kzz = _joint(cur["kern"], total * torch.softmax(u, 0), None, normalize)
        W0 = torch.as_tensor(whitener(Kzz.cpu().numpy()), dtype=f64, device=dev)
        Phi = Kxz[:n_tr] @ W0
        mu0 = Phi.mean(0)
        s0 = torch.sqrt(((Phi - mu0) ** 2).sum(1).mean()).clamp(min=TINY)
        del Kxz, Kzz, Phi
    n_in = W0.shape[1] + (0 if X is None else X.shape[1])
    net = _mlp(n_in, hidden, seed).to(dev)
    rows_t = torch.as_tensor(rows, dtype=torch.long, device=dev)

    def inputs(pos, with_fields):
        """The network input of the rows at positions `pos` of the current pass."""
        kern_ = cur["kern"]
        delta = theta - theta0 if (with_fields and kern_.dK is not None) else None
        h_t = total * torch.softmax(u, 0)
        Kxz_, _z, _a, _b = _joint(kern_, h_t if with_fields else h_t.detach(), delta, normalize,
                                  rows=pos, want_zz=False)
        z = (Kxz_ @ W0 - mu0) / s0
        return z if X is None else torch.cat([z, X[rows_t[pos]]], dim=1)

    def validation():
        with torch.no_grad():
            out = []
            for lo in range(n_tr, n_tr + n_va, 4096):
                pos = torch.arange(lo, min(lo + 4096, n_tr + n_va), device=dev)
                out.append(net(inputs(pos, True))[:, 0])
            pv = torch.cat(out)
            return float(torch.sqrt(((pv - y_t[n_tr:]) ** 2).mean())) * y_sd

    def snapshot():
        return {"net": {k: v.detach().clone() for k, v in net.state_dict().items()},
                "theta": theta.detach().clone(), "u": u.detach().clone()}

    rng = np.random.RandomState(int(seed))
    history: List[dict] = []

    def run_phase(fields: bool, max_epochs: int, label: str, first=None):
        groups = [{"params": list(net.parameters()), "lr": float(lr)}]
        if fields:
            groups.append({"params": [t for t in (theta, u) if t.requires_grad],
                           "lr": float(lr_fields)})
        opt = torch.optim.Adam(groups)
        best_rmse = validation() if first is None else first
        best_state, since = snapshot(), 0
        for epoch in range(int(max_epochs)):
            order = rng.permutation(n_tr)
            for lo in range(0, n_tr, int(batch)):
                pos = torch.as_tensor(order[lo:lo + int(batch)], dtype=torch.long, device=dev)
                opt.zero_grad()
                out = net(inputs(pos, fields))[:, 0]
                loss = ((out - y_t[pos]) ** 2).mean()
                pen = sum((p ** 2).sum() for name, p in net.named_parameters()
                          if name.endswith("weight"))
                (loss + 0.5 * float(l2) * pen / len(pos)).backward()
                opt.step()
                if fields and "alpha" in learn:
                    with torch.no_grad():
                        theta.clamp_(math.log(ALPHA_BOUNDS[0]), math.log(ALPHA_BOUNDS[1]))
                    if float((theta.detach() - theta0).abs().max()) > float(trust):
                        cur["kern"] = None
                        cur["kern"] = new_pass(rows, True)
                        theta0.copy_(theta.detach())
            rmse_va = validation()
            history.append({"phase": label, "epoch": epoch, "valid_rmse": rmse_va,
                            "alpha": [float(a) for a in torch.exp(theta.detach()).cpu()],
                            "passes": counters["passes"], "seconds": time.time() - t_start})
            if rmse_va < best_rmse - 1e-9:
                best_rmse, best_state, since = rmse_va, snapshot(), 0
            else:
                since += 1
            if epoch % 10 == 0 or since == 0:
                say(f"    {label} epoch {epoch}: validation RMSE {rmse_va:.5f} "
                    f"(best {best_rmse:.5f}), passes {counters['passes']}")
            if since >= int(patience):
                break
        net.load_state_dict(best_state["net"])
        with torch.no_grad():
            theta.copy_(best_state["theta"])
            u.copy_(best_state["u"])
        return best_rmse

    if learn:
        run_phase(True, epochs, "joint")
        cur["kern"] = None
        cur["kern"] = new_pass(rows, False)          # exact kernels at the chosen settings
        theta0.copy_(theta.detach())
        best_rmse = run_phase(False, epochs, "network alone")
    else:
        # the same two phases with the layer fixed, so that the control differs from the
        # trained layer in nothing but the layer
        run_phase(False, epochs, "first phase", first=math.inf)
        best_rmse = run_phase(False, epochs, "network alone")
    cur["kern"] = None
    alpha_best = np.exp(theta.detach().cpu().numpy())
    h_best = (total * torch.softmax(u.detach(), 0)).cpu().numpy()
    want = np.arange(packed["n"]) if predict is None else np.asarray(predict, int)
    pred = np.full(packed["n"], np.nan)
    with torch.no_grad():
        h_t = torch.as_tensor(h_best, dtype=f64, device=dev)
        for lo in range(0, len(want), 2048):
            part = want[lo:lo + 2048]
            k2 = kernel_pass(packed, part, land, alpha_best, derivative=False, device=dev,
                             dtype=dtype, budget=budget, zz=False)
            Kxz, _z, _a, _b = _joint(k2, h_t, None, normalize, want_zz=False)
            z = (Kxz @ W0 - mu0) / s0
            if X is not None:
                z = torch.cat([z, X[torch.as_tensor(part, dtype=torch.long, device=dev)]], 1)
            pred[part] = (net(z)[:, 0] * y_sd + y_mean).cpu().numpy()
            del k2
    return TrainedNetwork(names=list(packed["names"]), alpha=alpha_best, h=h_best, pred=pred,
                          learned=learn, history=history, epochs=len(history),
                          passes=counters["passes"], seconds=time.time() - t_start,
                          kernel_seconds=counters["kernel_seconds"], valid_rmse=best_rmse)
