"""The co-field as an atom feature, and a network that sums over the atoms (item ND14c).

A linear CMF model has a regression-coefficient field (the co-field) for every field type.  The
prediction for a molecule is the overlap of the fields of the molecule with these co-fields, and
a ridge regression has fitted them.  This module keeps the co-fields and replaces the last step.
The co-field of every field is read at the position of every atom, in the common frame of the
aligned molecules, and the values become features of the atom.  A network with shared weights
then reads every atom, the outputs are summed over the atoms, and a second network maps the sum
to the activity.  This structure is the neural device of Baskin, Palyulin and Zefirov (J. Chem.
Inf. Comput. Sci. 37 (1997) 715): a network on every atom, a sum, and a second network.

What is read at an atom
-----------------------
The value is the overlap of the co-field of field ``f`` with the Gaussian that the field puts on
the atom, for a weight of one:

    v_f(a) = h_f sum_z c_z n_fz sum_{b in z} w_fb E_f(a, b),

with ``z`` the landmark molecules of the model, ``c_z`` their regression coefficients, ``n_fz``
the factor of the kernel normalisation, ``w_fb`` the weight of atom ``b`` in the field and
``E_f`` the overlap of two atom Gaussians.  It is the quantity
:func:`cmfqsar.cofields.atom_effect` draws: what one atom of weight one at this place adds to
the prediction.  The linear model is the special case in which the values are multiplied by the
weights of the atoms and summed:

    y(x) = b + sum_f sum_{a in x} w_fa v_f(a)            ("mean" normalisation),

and with the "cosine" normalisation every field term is divided by the square root of the
self-kernel of the molecule (``molecule_norm=True`` does this division).

Out-of-fold values
------------------
The co-field is fitted to the activities of the training molecules.  If the co-field of the
full training part is read at a training molecule, the feature holds a part of the activity of
that same molecule, and a network trained on such features learns to trust them more than it
can for a new molecule.  :func:`fold_models` therefore fits one model per fold of the training
part, each without the molecules of its fold, and one model on the whole training part.
:func:`atom_cofield` takes, for every molecule, the model to read: a training molecule reads
the model of its fold (an out-of-fold value), and any other molecule reads the model of the
whole training part.  The models share the landmarks and the kernel, so the folds cost only
one more ridge solution each.

The atom network
----------------
The manual and the result tables call this model the *atom network*.  It has three parts: one
network with shared weights that reads the numbers of every atom, a sum of its outputs over the
atoms of the molecule, and a second network that maps the sums to the activity.  It sees no
bonds: a molecule is a set of atoms with their numbers.  :func:`train_device` trains the two
networks together by stochastic gradient descent, with early stopping on validation molecules.
The sum over the atoms does not depend on the order of the atoms.
The sums are called *collectors*, the name of the 1997 paper; ``collectors`` of the result holds
them for every molecule.  A network has no co-field: the reading of the model is kept at the
input (the co-fields that were read) and lost after it.

The low-rank model of the command line
---------------------------------------
:func:`lowrank_fold_weights` and :func:`lowrank_atom_cofield` do the same for the model of
:mod:`cmfqsar.lowrank`, which ``cmf lowrank`` fits: the weights of the model without each
fold, and its co-field at every atom of a table.  They run on the processor and need no
PyTorch; ``cmf lowrank --atom-cofield FILE`` writes their result.

PyTorch is needed for everything else (the optional dependency of :mod:`cmfqsar.fieldae`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from . import learnable as L

try:                                                    # pragma: no cover - import guard
    import torch
except ImportError:                                     # pragma: no cover
    torch = None


def available() -> bool:
    """Whether PyTorch is installed, so that the functions of this module can run."""
    return L.HAVE_TORCH


# --------------------------------------------------------------------------- #
# One linear model per fold of the training part
# --------------------------------------------------------------------------- #

@dataclass
class FoldModels:
    """Linear models over the same landmarks: one per fold, and one of all training rows.

    ``coef`` is (folds + 1, landmarks) and ``bias`` (folds + 1,): row ``k`` is the model fitted
    without fold ``k``, and the last row is the model of the whole training part.  ``fold_of``
    gives the fold of every training row, in the order of ``train``.  ``norms`` are the factors
    of the kernel normalisation, (fields, landmarks).  ``cv`` holds the out-of-fold prediction
    of every training row and ``fit`` the prediction of the model of the whole training part.
    """
    names: List[str]
    kinds: List[str]
    alpha: np.ndarray
    h: np.ndarray
    lam: float
    normalize: str
    landmarks: np.ndarray
    train: np.ndarray
    fold_of: np.ndarray
    coef: np.ndarray
    bias: np.ndarray
    norms: np.ndarray
    cv: np.ndarray
    fit: np.ndarray
    seconds: float = 0.0

    @property
    def folds(self) -> int:
        return len(self.coef) - 1

    def versions(self, rows: Sequence[int], out_of_fold: bool = True) -> np.ndarray:
        """The model every molecule of `rows` reads: its fold for a training molecule (with
        `out_of_fold`), the model of the whole training part (-1) for every other molecule."""
        rows = np.asarray(rows, int)
        out = np.full(len(rows), -1, dtype=int)
        if out_of_fold:
            where = {int(r): int(k) for r, k in zip(self.train, self.fold_of)}
            out = np.array([where.get(int(r), -1) for r in rows], dtype=int)
        return out


def fold_models(packed, y, train, landmarks, alpha=1.0, h=None, lam: float = 1e-2,
                normalize: str = "cosine", folds: int = 5, seed: int = 0, device=None,
                dtype=None, budget: float = L.BUDGET) -> FoldModels:
    """Fit the linear model of the training part, and the same model without each fold.

    `packed` holds the atoms of every molecule (:func:`cmfqsar.learnable.pack`), `y` one
    activity per molecule, `train` the training rows and `landmarks` the rows that carry the
    model.  `alpha` and `h` are the attenuation and the weight of every field (one number, or
    one per field), `lam` the ridge and `normalize` the kernel normalisation, all as in
    :func:`cmfqsar.learnable.train`.  The folds are those of that function for the same `seed`.
    """
    L.require()
    t0 = time.time()
    dev = L._device(device)
    F = len(packed["names"])
    y = np.asarray(y, float)
    tr = np.asarray(train, int)
    land = np.asarray(landmarks, int)
    alpha_ = np.broadcast_to(np.asarray(alpha, float), (F,)).astype(float).copy()
    h_ = (np.ones(F) if h is None
          else np.broadcast_to(np.asarray(h, float), (F,)).astype(float).copy())
    kern = L.kernel_pass(packed, tr, land, alpha_, derivative=False, device=dev, dtype=dtype,
                         budget=budget)
    f64 = torch.float64
    parts = L._kfold(len(tr), folds, seed)
    fold_of = np.zeros(len(tr), dtype=int)
    coef, bias = [], []
    with torch.no_grad():
        K, Kzz, _kxx, _kzz = L._joint(kern, torch.as_tensor(h_, dtype=f64, device=dev), None,
                                      normalize)
        y_t = torch.as_tensor(y[tr], dtype=f64, device=dev)
        G, s, v = K.T @ K, K.sum(0), K.T @ y_t
        n, sy = float(len(tr)), y_t.sum()
        cv = torch.zeros_like(y_t)
        for k, idx in enumerate(parts):
            fold_of[idx] = k
            i = torch.as_tensor(idx, dtype=torch.long, device=dev)
            Kk = K[i]
            c, b = L._solve(G - Kk.T @ Kk, s - Kk.sum(0), v - Kk.T @ y_t[i], n - len(idx),
                            sy - y_t[i].sum(), Kzz, float(lam))
            cv[i] = Kk @ c + b
            coef.append(c.cpu().numpy())
            bias.append(float(b))
        c, b = L._solve(G, s, v, n, sy, Kzz, float(lam))
        fit = (K @ c + b).cpu().numpy()
        coef.append(c.cpu().numpy())
        bias.append(float(b))
        dz = kern.dz.to(f64)
        if normalize == "cosine":
            norms = torch.where(dz > L.TINY, dz.clamp(min=L.TINY).rsqrt(), torch.zeros_like(dz))
        else:
            norms = (1.0 / dz.mean(dim=1, keepdim=True).clamp(min=L.TINY)).expand_as(dz)
        norms = norms.cpu().numpy()
        cv = cv.cpu().numpy()
    return FoldModels(names=list(packed["names"]), kinds=list(packed["kinds"]), alpha=alpha_,
                      h=h_, lam=float(lam), normalize=normalize, landmarks=land, train=tr,
                      fold_of=fold_of, coef=np.array(coef), bias=np.array(bias), norms=norms,
                      cv=cv, fit=fit, seconds=time.time() - t0)


# --------------------------------------------------------------------------- #
# The co-field of every field, read at every atom
# --------------------------------------------------------------------------- #

def atom_cofield(packed, models: FoldModels, rows, points, mask=None, radii=None, version=None,
                 molecule_norm: bool = False, device=None, dtype=None,
                 budget: float = L.BUDGET) -> np.ndarray:
    """The co-field of every field at every atom: (molecules, atoms, fields).

    `rows` are the molecules, as rows of `packed`.  `points` is (molecules, atoms, 3): the
    positions of the atoms that are read, in the frame of `packed`, padded to a common number
    of atoms.  `mask` is (molecules, atoms), true for a real atom; a padded place gets zero.
    `radii` is (molecules, atoms): the radius of every atom, which the steric field needs for
    the width of its Gaussian.  `version` names the model every molecule reads (see
    :meth:`FoldModels.versions`); the default is the model of the whole training part.

    With `molecule_norm` and the cosine normalisation, the values of a molecule are divided by
    the square root of its own self-kernel in the field, so that the sum of the values times
    the weights of the atoms is the share of the field in the prediction.
    """
    L.require()
    dev = L._device(device)
    dt = L._store_dtype(dev, dtype)
    f64 = torch.float64
    rows = np.asarray(rows, int)
    points = np.asarray(points, float)
    if points.ndim != 3 or points.shape[0] != len(rows) or points.shape[2] != 3:
        raise ValueError("points must be (molecules, atoms, 3), one block per molecule of rows")
    n, A = points.shape[:2]
    mask = np.ones((n, A), dtype=bool) if mask is None else np.asarray(mask, bool)
    version = (np.full(n, -1, dtype=int) if version is None else np.asarray(version, int))
    version = np.where(version < 0, models.folds, version)
    if version.max(initial=0) > models.folds:
        raise ValueError("version names a fold that the models do not have")
    F = len(packed["names"])
    if "steric" in packed["kinds"] and radii is None:
        raise ValueError("the steric field needs the radius of every atom that is read; "
                         "pass radii")
    land = models.landmarks
    m = len(land)
    out = np.zeros((n, A, F))
    safe = np.where(mask[:, :, None], points, 0.0)
    cp_all = np.ones((n, A))
    if radii is not None:
        cp_steric = np.where(mask, 1.0 / np.maximum(np.asarray(radii, float), 1e-6) ** 2, 1.0)
    with torch.no_grad():
        for f in range(F):
            kind = packed["kinds"][f]
            a = float(models.alpha[f])
            centre = packed["xyz"][f].reshape(-1, 3).mean(axis=0)
            xq = torch.as_tensor(packed["xyz"][f][land] - centre, dtype=dt, device=dev)
            wq = torch.as_tensor(packed["w"][f][land], dtype=dt, device=dev)
            cq = torch.as_tensor(packed["c"][f][land], dtype=dt, device=dev)
            B = wq.shape[1]
            C = torch.as_tensor(models.norms[f][:, None] * models.coef.T * float(models.h[f]),
                                dtype=f64, device=dev)                    # (landmarks, models)
            cp_f = cp_steric if kind == "steric" else cp_all
            step = max(1, int(budget // max(A * m * B, 1)))
            for lo in range(0, n, step):
                hi = min(lo + step, n)
                b = hi - lo
                xp = torch.as_tensor(safe[lo:hi] - centre, dtype=dt, device=dev)
                cp = torch.as_tensor(cp_f[lo:hi], dtype=dt, device=dev)
                K, _ = L._cross(xp.reshape(b * A, 1, 3),
                                torch.ones((b * A, 1), dtype=dt, device=dev),
                                cp.reshape(b * A, 1), xq, wq, cq, a, kind, False)
                vals = (K.to(f64) @ C).view(b, A, -1)                     # (b, atoms, models)
                pick = torch.as_tensor(version[lo:hi], dtype=torch.long, device=dev)
                vals = vals.gather(2, pick[:, None, None].expand(b, A, 1))[:, :, 0]
                if molecule_norm and models.normalize == "cosine":
                    idx = rows[lo:hi]
                    dx, _ = L._self(
                        torch.as_tensor(packed["xyz"][f][idx] - centre, dtype=dt, device=dev),
                        torch.as_tensor(packed["w"][f][idx], dtype=dt, device=dev),
                        torch.as_tensor(packed["c"][f][idx], dtype=dt, device=dev),
                        a, kind, False)
                    dx = dx.to(f64)
                    inv = torch.where(dx > L.TINY, dx.clamp(min=L.TINY).rsqrt(),
                                      torch.zeros_like(dx))
                    vals = vals * inv[:, None]
                out[lo:hi, :, f] = vals.cpu().numpy()
    out[~mask] = 0.0
    return out


# --------------------------------------------------------------------------- #
# The same for the low-rank model of cmfqsar.lowrank
# --------------------------------------------------------------------------- #

def lowrank_fold_weights(low, Phi, y, lam: float, folds: int = 10, seed: Optional[int] = 0):
    """The weights of a low-rank model without each fold, and with every molecule.

    `low` is the feature map (:func:`cmfqsar.lowrank.build`), `Phi` the feature rows of the
    molecules and `y` their activities.  Returns (weights (folds + 1, columns), biases
    (folds + 1,), the fold of every molecule).  Row ``k`` is the model fitted without fold
    ``k``; the last row is the model of all molecules.  The folds are those of
    :func:`cmfqsar.lowrank.cross_validate` for the same `seed`.
    """
    from . import lowrank as LR
    from .model import _kfold_indices

    y = np.asarray(y, float).ravel()
    Phi = np.asarray(Phi, float)
    parts = _kfold_indices(len(y), folds, seed)
    which = np.zeros(len(y), dtype=int)
    W, B = np.zeros((len(parts) + 1, low.rank)), np.zeros(len(parts) + 1)

    def solve(rows):
        part = Phi[rows]
        g = LR.Gram(PtP=part.T @ part, Pty=part.T @ y[rows], Pt1=part.sum(axis=0),
                    n=len(rows), sum_y=float(y[rows].sum()), yty=float(y[rows] @ y[rows]))
        return LR.fit(g, float(lam))

    for k, test in enumerate(parts):
        which[test] = k
        W[k], B[k] = solve(np.setdiff1d(np.arange(len(y)), test))
    W[-1], B[-1] = solve(np.arange(len(y)))
    return W, B, which


def lowrank_atom_cofield(low, w, table) -> np.ndarray:
    """The co-field of a low-rank model at every atom of `table`: (atoms, fields).

    `w` holds the weights of the model over the columns of `low`.  The value is the overlap of
    the co-field of the field with the Gaussian that the field puts on the atom, for a weight
    of one.  The sum over the atoms of a molecule of the values times the weights of the
    atoms, divided by the square root of the self-kernel of the molecule in the field when the
    kernels are normalised, is the share of the field in the prediction.
    """
    from . import lowrank as LR
    from .cofields import evaluate_gaussians, probe_basis

    points = np.asarray(table.coords, float)
    radii = np.asarray(table.radii, float)
    out = np.zeros((len(points), len(low.specs)))
    for f, spec in enumerate(low.specs):
        centres, coeffs, widths = LR.cofield_gaussians(low, w, f)
        if spec.kind == "gauss":
            own = np.full(len(points), float(spec.alpha))
        elif spec.kind == "steric":
            own = float(spec.alpha) / np.maximum(radii, 1e-6) ** 2
        else:
            raise ValueError(f"no atom Gaussian is defined for the field kind {spec.kind!r} "
                             f"of {spec.name!r}")
        for width in np.unique(own):
            at = own == width
            out[at, f] = evaluate_gaussians(points[at],
                                            *probe_basis(centres, coeffs, widths, width),
                                            eps=1e-9)
    return out


# --------------------------------------------------------------------------- #
# A network on every atom, a sum, and a second network
# --------------------------------------------------------------------------- #

@dataclass
class TrainedDevice:
    """A trained network of :func:`train_device` and what it gives for the molecules it saw.

    ``pred`` holds the prediction of every molecule that was given, ``collectors`` the sums
    over the atoms, (molecules, collectors).  ``valid_rmse`` is the lowest error on the
    validation rows, and the returned network is the one of that epoch.
    """
    pred: np.ndarray
    collectors: np.ndarray
    valid_rmse: float
    epochs: int
    seconds: float
    history: List[dict] = field(default_factory=list)
    net: object = None
    scaling: dict = field(default_factory=dict)

    def predict(self, X, mask=None, extra=None, return_collectors: bool = False):
        """Predictions for other molecules: `X`, `mask` and `extra` as in :func:`train_device`."""
        return _apply(self.net, self.scaling, X, mask, extra, return_collectors)


def _mlp(n_in: int, hidden: Sequence[int], n_out: int):
    layers, last = [], int(n_in)
    for width in hidden:
        layers += [torch.nn.Linear(last, int(width)), torch.nn.ReLU()]
        last = int(width)
    layers.append(torch.nn.Linear(last, int(n_out)))
    return torch.nn.Sequential(*layers)


def _build(n_atom: int, hidden, collectors: int, n_extra: int, brain, seed: int):
    torch.manual_seed(int(seed))
    net = torch.nn.ModuleDict({"atom": _mlp(n_atom, hidden, collectors),
                               "brain": _mlp(collectors + n_extra, brain, 1)})
    return net.double()


def _forward(net, X, mask, extra, per_molecule: float):
    """(predictions, collectors) for scaled inputs: X (b, atoms, d), mask (b, atoms)."""
    z = net["atom"](X) * mask[:, :, None]
    col = z.sum(1) / per_molecule
    inp = col if extra is None else torch.cat([col, extra], dim=1)
    return net["brain"](inp)[:, 0], col


def _apply(net, sc, X, mask, extra, return_collectors: bool, block: int = 4096):
    dev = next(net.parameters()).device
    X = np.asarray(X, float)
    mask = np.ones(X.shape[:2], dtype=bool) if mask is None else np.asarray(mask, bool)
    preds, cols = [], []
    with torch.no_grad():
        for lo in range(0, len(X), block):
            x = torch.as_tensor((X[lo:lo + block] - sc["mean"]) / sc["sd"], dtype=torch.float64,
                                device=dev)
            mk = torch.as_tensor(mask[lo:lo + block], dtype=torch.float64, device=dev)
            ex = None
            if sc["extra"]:
                if extra is None:
                    raise ValueError("this network was trained with extra columns; pass extra")
                ex = torch.as_tensor(np.asarray(extra, float)[lo:lo + block],
                                     dtype=torch.float64, device=dev)
            p, c = _forward(net, x, mk, ex, sc["per_molecule"])
            preds.append((p * sc["y_sd"] + sc["y_mean"]).cpu().numpy())
            cols.append(c.cpu().numpy())
    pred, col = np.concatenate(preds), np.concatenate(cols)
    return (pred, col) if return_collectors else pred


def train_device(X, mask, y, train, valid, extra=None, hidden: Sequence[int] = (64, 64),
                 collectors: int = 16, brain: Sequence[int] = (64,), l2: float = 1e-4,
                 epochs: int = 400, batch: int = 128, lr: float = 1e-3, patience: int = 40,
                 lr_decay: float = 0.5, seed: int = 0, device=None,
                 log: Optional[Callable[[str], None]] = None) -> TrainedDevice:
    """Train the atom network: a network on every atom, a sum over the atoms, a second network.

    `X` is (molecules, atoms, features): the features of every atom, padded to a common number
    of atoms; `mask` is (molecules, atoms), true for a real atom.  `y` holds one activity per
    molecule, `train` and `valid` are rows.  `extra` is an optional (molecules, columns) block
    of descriptors of the whole molecule, already scaled; it is joined to the sums before the
    second network.

    The first network has the hidden layers `hidden` and `collectors` outputs, the second the
    hidden layers `brain` and one output.  Every feature is centred and divided by its standard
    deviation over the atoms of the training molecules, and the sums are divided by the mean
    number of atoms of a training molecule.  The training is Adam with the rate `lr`, batches
    of `batch` molecules and the L2 penalty `l2` on the weights.  When the error on the
    validation rows has not fallen for a third of `patience` epochs, the rate is multiplied by
    `lr_decay` (1 keeps it fixed).  The training stops when that error has not fallen for
    `patience` epochs, and the network of the best epoch is returned.
    """
    L.require()
    dev = L._device(device)
    say = log or (lambda _msg: None)
    t0 = time.time()
    f64 = torch.float64
    X = np.asarray(X, float)
    if X.ndim != 3:
        raise ValueError("X must be (molecules, atoms, features)")
    mask = np.ones(X.shape[:2], dtype=bool) if mask is None else np.asarray(mask, bool)
    y = np.asarray(y, float)
    tr, va = np.asarray(train, int), np.asarray(valid, int)
    if not len(tr) or not len(va):
        raise ValueError("the device needs training rows and validation rows")
    atoms = X[tr][mask[tr]]
    mean = atoms.mean(axis=0)
    sd = atoms.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    sc = {"mean": mean, "sd": sd, "per_molecule": float(max(mask[tr].sum(1).mean(), 1.0)),
          "y_mean": float(y[tr].mean()), "y_sd": float(y[tr].std()) or 1.0,
          "extra": extra is not None}
    Xt = torch.as_tensor((X - mean) / sd, dtype=f64, device=dev)
    Mt = torch.as_tensor(mask, dtype=f64, device=dev)
    Et = None if extra is None else torch.as_tensor(np.asarray(extra, float), dtype=f64,
                                                    device=dev)
    yt = torch.as_tensor((y - sc["y_mean"]) / sc["y_sd"], dtype=f64, device=dev)
    net = _build(X.shape[2], hidden, int(collectors), 0 if Et is None else Et.shape[1], brain,
                 seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    va_t = torch.as_tensor(va, dtype=torch.long, device=dev)

    def validation() -> float:
        with torch.no_grad():
            p, _ = _forward(net, Xt[va_t], Mt[va_t], None if Et is None else Et[va_t],
                            sc["per_molecule"])
            return float(torch.sqrt(((p - yt[va_t]) ** 2).mean())) * sc["y_sd"]

    rng = np.random.RandomState(int(seed))
    best, since, since_decay = float("inf"), 0, 0
    wait = max(int(patience) // 3, 1)
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    history: List[dict] = []
    for epoch in range(int(epochs)):
        order = rng.permutation(len(tr))
        for lo in range(0, len(tr), int(batch)):
            pos = torch.as_tensor(tr[order[lo:lo + int(batch)]], dtype=torch.long, device=dev)
            opt.zero_grad()
            p, _ = _forward(net, Xt[pos], Mt[pos], None if Et is None else Et[pos],
                            sc["per_molecule"])
            loss = ((p - yt[pos]) ** 2).mean()
            pen = sum((w ** 2).sum() for name, w in net.named_parameters()
                      if name.endswith("weight"))
            (loss + 0.5 * float(l2) * pen / len(pos)).backward()
            opt.step()
        rmse_va = validation()
        history.append({"epoch": epoch, "valid_rmse": rmse_va, "seconds": time.time() - t0,
                        "rate": float(opt.param_groups[0]["lr"])})
        if rmse_va < best - 1e-9:
            best, since, since_decay = rmse_va, 0, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            since += 1
            since_decay += 1
            if since_decay >= wait and float(lr_decay) < 1.0:
                since_decay = 0
                for group in opt.param_groups:
                    group["lr"] *= float(lr_decay)
        if epoch % 20 == 0 or since == 0:
            say(f"    epoch {epoch}: validation RMSE {rmse_va:.5f} (best {best:.5f})")
        if since >= int(patience):
            break
    net.load_state_dict(best_state)
    pred, col = _apply(net, sc, X, mask, extra, True)
    return TrainedDevice(pred=pred, collectors=col, valid_rmse=best, epochs=len(history),
                         seconds=time.time() - t0, history=history, net=net, scaling=sc)
