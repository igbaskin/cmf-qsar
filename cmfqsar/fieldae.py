"""A field autoencoder with an analytic decoder (plan item E11).

Kernel principal component analysis on the CMF kernel is the linear autoencoder of the field
space, and this program already has it twice: as the coordinates of the generative topographic
map (:mod:`cmfqsar.gtm`) and as the ``ockpca`` applicability-domain score of
:mod:`cmfqsar.oneclass`.  What kernel PCA cannot give is a **pre-image**.  A field reconstructed
from its principal components is a weighted sum of training fields.  It can be drawn on a grid,
but it is not a set of atoms, nothing can be sampled from it, and no design step can take it as
a target.  This module adds a nonlinear autoencoder that never leaves the field space and whose
output *is* a set of atoms.

The three pieces
----------------
**The encoder** maps a molecule to a code of a few dimensions.  Its input is the whitened kernel
against a few hundred landmarks -- the feature vector of :mod:`cmfqsar.lowrank`, built here by
:func:`landmark_features` -- so the encoder sees a molecule exactly as the kernel does and needs
neither a grid nor an atom order nor a fixed atom count.

**The decoder** maps a code to a fixed number of *pseudo-atoms*, each with a position and one
weight per field type.  That is the same form as every other field in the program, so a decoded
field can be handed to the overlap kernel, to a co-field reader or to a matcher without any
conversion (:meth:`FieldAutoencoder.pre_image`).

**The loss** is the squared distance between the input field and the decoded field in the field
Hilbert space::

    ||f - f'||^2 = k(f, f) - 2 k(f, f') + k(f', f')

and all three terms are sums of overlap integrals of atom-centred Gaussians, the same closed
form as every kernel in :mod:`cmfqsar.kernels`.  The loss is therefore exact and differentiable
in the positions and the weights of the pseudo-atoms, and **rule 3 holds: no grid enters the
training**.  :func:`overlap` is that formula written for PyTorch tensors, and
``tests/test_fieldae.py`` holds it to :func:`cmfqsar.kernels.kernel_matrices`, which stays the
reference.

What does not change: the activity model
----------------------------------------
The autoencoder is never the regressor.  A code that enters a prediction enters it as one more
kernel view, so the predictive model stays a kernel model, rule 2 holds, and the co-field, the
Shapley values and every interpretation tool are untouched.  That is why this is not the
convolutional network that the plan's "What is not planned" excludes: there a grid replaces the
kernel and a network replaces the model; here the loss *is* the closed-form kernel and the model
is the one the program already has.

What is measured, and what is not
---------------------------------
`benchmarks/run_fieldae.py` measures version (i) of the item, the plain autoencoder, against the
two gates that decide whether the nonlinearity is worth anything:

(a) the relative reconstruction error ``||f - f'|| / ||f||`` on held-out molecules, against
    kernel PCA with the same number of latent dimensions (:func:`kpca_relative_errors`);
(b) the reconstruction error as an applicability-domain score, against the five scores of
    :mod:`cmfqsar.oneclass` under the protocol of item E5 (:mod:`cmfqsar.errorpred`).

Versions (ii) denoising, (iii) variational and (iv) semi-supervised are not here.
:meth:`FieldAutoencoder.sample` draws a code from the Gaussian fitted to the training codes,
which is the pre-image entry point the design items need; it is not the variational prior of
version (iii), and it is labelled as what it is.

Using it
--------
PyTorch is an optional dependency, in the way Numba, ORCA and xTB are: this module imports
without it, :func:`available` is then false and :func:`fit` raises with a message that says what
to install.  Install it with ``pip install torch`` or ``pip install -e .[fieldae]``.

    >>> from cmfqsar import fieldae
    >>> geom = fieldae.geometry(fieldset, scale=fieldae.field_scales(table, fieldset))
    >>> tens = fieldae.tensors(table, geom)
    >>> feats, low = fieldae.landmark_features(table, fieldset, n_landmarks=256)
    >>> ae = fieldae.FieldAutoencoder(geom, feats.shape[1], code=8, n_pseudo=32)
    >>> hist = fieldae.fit(ae, feats, tens, train_rows, valid_rows)
    >>> err = fieldae.relative_errors(ae, feats, tens, test_rows)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field as _field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fields import AtomTable, FieldSet, FieldSpec, MoleculeFields
from .kernels import kernel_matrices, self_kernels

try:                                            # pragma: no cover - depends on the installation
    import torch
    HAVE_TORCH = True
except Exception:                               # pragma: no cover - PyTorch is optional
    torch = None                                # type: ignore[assignment]
    HAVE_TORCH = False

#: symbol and radius given to a pseudo-atom.  The radius is needed by the radius-scaled steric
#: field, which has a width per atom; it is the same value :mod:`cmfqsar.pbh` gives a ring centre.
PSEUDO_SYMBOL = "Du"
PSEUDO_RADIUS = 1.7
#: what to install when PyTorch is missing
INSTALL = ("this needs PyTorch, which is an optional dependency: pip install torch "
           "(or pip install -e .[fieldae])")


def available() -> bool:
    """Whether PyTorch is installed, so that the autoencoder can be built and fitted."""
    return HAVE_TORCH


def require() -> None:
    """Raise with an installation message when PyTorch is missing."""
    if not HAVE_TORCH:
        raise ImportError(INSTALL)


# --------------------------------------------------------------------------- #
# The field set, as the closed-form overlap needs to see it
# --------------------------------------------------------------------------- #


@dataclass
class FieldGeometry:
    """What the overlap of two sets of Gaussian atoms needs to know about a field set.

    `kinds` is "gauss" (one width for every atom, ``b = alpha``) or "steric" (a width per atom,
    ``b = alpha / R^2``).  `scale` is the constant in front of each field in the joint kernel,
    ``h_f`` times whatever field scaling the caller uses; :func:`field_scales` gives the
    convention of this program, one over the mean self-kernel of the field over the whole set.
    """
    names: List[str]
    kinds: List[str]
    alphas: np.ndarray
    scale: np.ndarray

    @property
    def n_fields(self) -> int:
        return len(self.names)

    def groups(self) -> Tuple[Dict[float, List[int]], List[int]]:
        """The fields of shared width grouped by that width, and the radius-scaled ones.

        The grouping is what makes the overlap one pass over the distance block per distinct
        width rather than one per field, exactly as :func:`cmfqsar.kernels.atom_kernel_block`
        does it.
        """
        gauss: Dict[float, List[int]] = {}
        steric: List[int] = []
        for f, kind in enumerate(self.kinds):
            if kind == "steric":
                steric.append(f)
            else:
                gauss.setdefault(float(self.alphas[f]), []).append(f)
        return gauss, steric


def geometry(fieldset: FieldSet, scale: Optional[Sequence[float]] = None) -> FieldGeometry:
    """The :class:`FieldGeometry` of an enabled field set.

    Probe fields and descriptor blocks are refused rather than ignored: a probe field puts
    several Gaussians on every atom and a descriptor block is not a field at all, so neither can
    be reproduced by pseudo-atoms that carry one weight per field.
    """
    specs = [s for s in fieldset.specs if s.enabled]
    if not specs:
        raise ValueError("the field set has no enabled field")
    bad = [s.name for s in specs if getattr(s, "is_probe", False)]
    if bad:
        raise ValueError(f"a probe field puts several Gaussians on one atom, so a pseudo-atom "
                         f"cannot carry it: {', '.join(bad)}")
    if getattr(fieldset, "descriptors", ()):
        raise ValueError("a descriptor block is not a field, so the autoencoder cannot "
                         "reconstruct it; drop the descriptors from the field set")
    h = np.array([float(s.h) for s in specs])
    s = np.ones(len(specs)) if scale is None else np.asarray(scale, float)[:len(specs)]
    return FieldGeometry(names=[s_.name for s_ in specs],
                         kinds=[("steric" if s_.kind == "steric" else "gauss") for s_ in specs],
                         alphas=np.array([float(s_.alpha) for s_ in specs]),
                         scale=h * s)


def field_scales(table: AtomTable, fieldset: FieldSet) -> np.ndarray:
    """One over the mean self-kernel of every field over the whole table.

    This is the field-scaling convention of the program: each field kernel is brought to unit
    mean diagonal, so that fields of very different magnitude -- a steric field and a charge
    field -- enter the joint kernel on comparable terms.
    """
    specs = [s for s in fieldset.specs if s.enabled]
    d = np.asarray(self_kernels(table, specs), float)
    mean = d.mean(axis=1)
    return np.where(mean > 0, 1.0 / np.maximum(mean, 1e-300), 1.0)


def joint_kernel(P: AtomTable, Q: AtomTable, fieldset: FieldSet,
                 geom: FieldGeometry, symmetric: Optional[bool] = None) -> np.ndarray:
    """The scaled joint kernel of two tables, ``sum_f scale_f K_f``, from the NumPy kernels."""
    specs = [s for s in fieldset.specs if s.enabled]
    K = np.asarray(kernel_matrices(P, Q, specs, symmetric=symmetric), float)
    return np.tensordot(np.asarray(geom.scale, float), K, axes=(0, 0))


def joint_self(table: AtomTable, fieldset: FieldSet, geom: FieldGeometry) -> np.ndarray:
    """``k(f, f)`` of every molecule of the table, in the same scaling."""
    specs = [s for s in fieldset.specs if s.enabled]
    d = np.asarray(self_kernels(table, specs), float)
    return np.asarray(geom.scale, float) @ d


# --------------------------------------------------------------------------- #
# The molecules, as padded arrays
# --------------------------------------------------------------------------- #


@dataclass
class FieldTensors:
    """Every molecule of a table as padded arrays, plus its own field norm.

    A padded atom carries zero weight in every field, and a zero-weight atom contributes nothing
    to an overlap, so a batch of molecules of different size needs **no mask**: the padding is
    exact.  Its position is set to the origin and its radius to one, because a radius of zero
    would divide by zero in the width of the steric field.
    """
    coords: np.ndarray            # (n, A, 3)
    weights: np.ndarray           # (n, A, F)
    radii: np.ndarray             # (n, A)
    self_k: np.ndarray            # (n,) k(f, f) in the scaling of the geometry
    counts: np.ndarray            # (n,) real atoms per molecule
    names: List[str]

    @property
    def n_mol(self) -> int:
        return len(self.names)

    @property
    def max_atoms(self) -> int:
        return self.coords.shape[1]

    def box(self, margin: float = 1.1) -> np.ndarray:
        """Half the extent of the frame along each axis, which bounds the pseudo-atoms."""
        if not len(self.coords):
            return np.ones(3)
        real = np.abs(self.weights).sum(axis=2) > 0
        pts = self.coords[real]
        if not len(pts):
            return np.ones(3)
        return np.maximum(np.abs(pts).max(axis=0) * float(margin), 1e-3)

    def weight_scale(self, quantile: float = 0.99) -> np.ndarray:
        """A typical size of an atom weight in each field, used to condition the decoder."""
        real = np.abs(self.weights).sum(axis=2) > 0
        w = np.abs(self.weights[real])
        if not len(w):
            return np.ones(self.weights.shape[2])
        q = np.quantile(w, float(quantile), axis=0)
        return np.where(q > 0, q, 1.0)


def tensors(table: AtomTable, fieldset: FieldSet, geom: FieldGeometry) -> FieldTensors:
    """Pad every molecule of `table` to the largest atom count and take its field norm."""
    mols = table.molecules
    if not mols:
        raise ValueError("the table holds no molecules")
    n_f = geom.n_fields
    if table.weights.shape[1] != n_f:
        raise ValueError(f"the table carries {table.weights.shape[1]} fields and the geometry "
                         f"{n_f}; they must be the same field set in the same order")
    counts = np.array([m.coords.shape[0] for m in mols], int)
    A = int(counts.max())
    coords = np.zeros((len(mols), A, 3))
    weights = np.zeros((len(mols), A, n_f))
    radii = np.ones((len(mols), A))
    for i, m in enumerate(mols):
        k = m.coords.shape[0]
        coords[i, :k] = m.coords
        weights[i, :k] = m.weights
        radii[i, :k] = np.maximum(m.radii, 1e-6)
    return FieldTensors(coords=coords, weights=weights, radii=radii,
                        self_k=joint_self(table, fieldset, geom), counts=counts,
                        names=list(table.names))


# --------------------------------------------------------------------------- #
# The closed-form overlap, for PyTorch tensors
# --------------------------------------------------------------------------- #


def overlap(xa, wa, ra, xb, wb, rb, geom: FieldGeometry):
    """The joint overlap integral of two sets of Gaussian atoms, in closed form.

    `xa` is (..., Na, 3), `wa` is (..., Na, F), `ra` is (..., Na), and the same for the second
    set, with one shared batch shape.  Returns (...,): ``sum_f scale_f K_f``, the same number
    :func:`joint_kernel` gives for two tables.

    The distances are formed by subtraction rather than by ``torch.cdist``, whose gradient is
    not defined where two atoms coincide -- which is exactly where a pseudo-atom may be pushed.
    """
    require()
    d2 = (xa.unsqueeze(-2) - xb.unsqueeze(-3)).pow(2).sum(-1)       # (..., Na, Nb)
    gauss, steric = geom.groups()
    out = None
    for alpha, fids in gauss.items():
        E = torch.exp(-0.25 * alpha * d2) * (math.pi / alpha) ** 1.5
        for f in fids:
            term = float(geom.scale[f]) * torch.einsum(
                "...i,...ij,...j->...", wa[..., f], E, wb[..., f])
            out = term if out is None else out + term
    for f in steric:
        alpha = float(geom.alphas[f])
        ba = alpha / ra.pow(2)
        bb = alpha / rb.pow(2)
        s = ba.unsqueeze(-1) + bb.unsqueeze(-2)
        E = torch.exp(-(ba.unsqueeze(-1) * bb.unsqueeze(-2)) / (2.0 * s) * d2) \
            * (2.0 * math.pi / s) ** 1.5
        term = float(geom.scale[f]) * torch.einsum(
            "...i,...ij,...j->...", wa[..., f], E, wb[..., f])
        out = term if out is None else out + term
    if out is None:                             # pragma: no cover - a geometry has fields
        raise ValueError("the geometry carries no field")
    return out


def pseudo_table(positions: np.ndarray, weights: np.ndarray,
                 radius: float = PSEUDO_RADIUS, name: str = "pseudo") -> AtomTable:
    """One set of pseudo-atoms as a one-molecule :class:`~cmfqsar.fields.AtomTable`.

    This is what lets the NumPy kernels be the reference for the formula above, and what lets a
    decoded field be used by anything in the program that takes a table.
    """
    return AtomTable.from_molecules([pre_image_fields(positions, weights, radius, name)])


def pre_image_fields(positions: np.ndarray, weights: np.ndarray,
                     radius: float = PSEUDO_RADIUS, name: str = "pseudo") -> MoleculeFields:
    """One set of pseudo-atoms as :class:`~cmfqsar.fields.MoleculeFields`.

    It is not a molecule: the pseudo-atoms have no elements and no bonds, and their weights can
    be of either sign in any field.  It is the field a molecule would have to place.
    """
    x = np.asarray(positions, float).reshape(-1, 3)
    w = np.asarray(weights, float).reshape(len(x), -1)
    return MoleculeFields(name=name, coords=x, weights=w,
                          radii=np.full(len(x), float(radius)),
                          symbols=[PSEUDO_SYMBOL] * len(x),
                          atom_types=["pseudo"] * len(x),
                          bonds=np.zeros((0, 2), int), charges=np.zeros(len(x)))


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #

if HAVE_TORCH:                                  # pragma: no cover - the stub is the other branch
    _Base = torch.nn.Module
else:
    class _Base:                                # type: ignore[no-redef]
        """Stands in for ``torch.nn.Module`` so that this module imports without PyTorch."""

        def __init__(self, *a, **k):
            raise ImportError(INSTALL)


def _mlp(sizes: Sequence[int], out: int):
    """A plain multi-layer perceptron with GELU between the layers and nothing after the last."""
    layers: List[object] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [torch.nn.Linear(int(a), int(b)), torch.nn.GELU()]
    layers.append(torch.nn.Linear(int(sizes[-1]), int(out)))
    return torch.nn.Sequential(*layers)


class LandmarkEncoder(_Base):
    """The whitened landmark kernel of a molecule, mapped to a code of a few dimensions.

    The input columns are standardised by the training mean and spread, held as buffers so that
    they travel with the fitted model; without it the columns of a whitened kernel differ by
    orders of magnitude and the first layer cannot see the small ones.
    """

    def __init__(self, n_features: int, code: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.net = _mlp([int(n_features), *[int(h) for h in hidden]], int(code))
        self.register_buffer("mean_", torch.zeros(int(n_features)))
        self.register_buffer("sd_", torch.ones(int(n_features)))

    def set_standardisation(self, features: np.ndarray) -> None:
        f = np.asarray(features, float)
        mean = f.mean(axis=0)
        sd = f.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        with torch.no_grad():
            self.mean_.copy_(torch.as_tensor(mean, dtype=self.mean_.dtype))
            self.sd_.copy_(torch.as_tensor(sd, dtype=self.sd_.dtype))

    def forward(self, features):
        return self.net((features - self.mean_) / self.sd_)


class PseudoAtomDecoder(_Base):
    """A code, mapped to a fixed number of pseudo-atoms with a position and a weight per field.

    The positions are ``box * tanh(raw)``, so a pseudo-atom cannot be pushed out of the frame
    the training molecules occupy; the weights are a per-field scale times a linear output, of
    either sign, because a charge field has both.
    """

    def __init__(self, code: int, n_pseudo: int, n_fields: int,
                 hidden: Sequence[int] = (256, 256),
                 box: Sequence[float] = (10.0, 10.0, 5.0),
                 weight_scale: Optional[Sequence[float]] = None):
        super().__init__()
        self.n_pseudo = int(n_pseudo)
        self.n_fields = int(n_fields)
        self.net = _mlp([int(code), *[int(h) for h in hidden]],
                        self.n_pseudo * (3 + self.n_fields))
        self.register_buffer("box_", torch.as_tensor(np.asarray(box, float), dtype=torch.float32))
        ws = np.ones(self.n_fields) if weight_scale is None else np.asarray(weight_scale, float)
        self.register_buffer("wscale_", torch.as_tensor(ws, dtype=torch.float32))

    def forward(self, code):
        raw = self.net(code).reshape(code.shape[0], self.n_pseudo, 3 + self.n_fields)
        positions = self.box_.to(raw.dtype) * torch.tanh(raw[..., :3])
        weights = self.wscale_.to(raw.dtype) * raw[..., 3:]
        return positions, weights


class FieldAutoencoder(_Base):
    """The encoder, the decoder and the field geometry that makes the loss closed form."""

    def __init__(self, geom: FieldGeometry, n_features: int, code: int = 8, n_pseudo: int = 32,
                 hidden: Sequence[int] = (256, 256), radius: float = PSEUDO_RADIUS,
                 box: Sequence[float] = (10.0, 10.0, 5.0),
                 weight_scale: Optional[Sequence[float]] = None, seed: int = 0):
        require()
        if seed is not None:
            torch.manual_seed(int(seed))
        super().__init__()
        self.geom = geom
        self.code_dim = int(code)
        self.radius = float(radius)
        self.encoder = LandmarkEncoder(n_features, code, hidden)
        self.decoder = PseudoAtomDecoder(code, n_pseudo, geom.n_fields, hidden, box, weight_scale)
        self.register_buffer("code_mean_", torch.zeros(int(code)))
        self.register_buffer("code_sd_", torch.ones(int(code)))

    # ------------------------------------------------------------------ forward
    @property
    def n_pseudo(self) -> int:
        return self.decoder.n_pseudo

    def forward(self, features):
        """(positions, weights) of the pseudo-atoms of every row of `features`."""
        return self.decoder(self.encoder(features))

    def _radii(self, positions):
        return torch.full(positions.shape[:-1], self.radius, dtype=positions.dtype,
                          device=positions.device)

    def squared_error(self, features, coords, weights, radii, self_k):
        """``||f - f'||^2`` of every molecule of the batch, in the field Hilbert space.

        The three terms are each of the size of ``k(f, f)`` and their sum is much smaller, so the
        subtraction loses precision as the reconstruction improves: in single precision the
        result is reliable while the *relative* error is above about 1e-3, which is two orders
        of magnitude below anything measured here.  A fit that has to go below that must be run
        in double precision.
        """
        positions, w = self(features)
        rp = self._radii(positions)
        cross = overlap(coords, weights, radii, positions, w, rp, self.geom)
        own = overlap(positions, w, rp, positions, w, rp, self.geom)
        return self_k - 2.0 * cross + own, positions, w

    def relative_error(self, features, coords, weights, radii, self_k):
        """``||f - f'|| / ||f||``, the quantity gate (a) of the item compares."""
        sq, positions, w = self.squared_error(features, coords, weights, radii, self_k)
        return torch.sqrt(torch.clamp(sq, min=0.0) / torch.clamp(self_k, min=1e-30)), positions, w

    # ------------------------------------------------------------------ the pre-image
    def codes(self, features, batch: int = 512) -> np.ndarray:
        """The code of every row of `features`, with no gradient."""
        require()
        f = np.asarray(features, float)
        out = np.zeros((len(f), self.code_dim))
        self.eval()
        with torch.no_grad():
            for lo in range(0, len(f), int(batch)):
                hi = min(lo + int(batch), len(f))
                x = torch.as_tensor(f[lo:hi], dtype=torch.float32)
                out[lo:hi] = self.encoder(x).detach().cpu().numpy()
        return out

    def pre_image(self, code: Sequence[float], name: str = "pseudo") -> MoleculeFields:
        """The pseudo-atoms a code decodes to, as fields the rest of the program can read."""
        require()
        c = np.atleast_2d(np.asarray(code, float))
        self.eval()
        with torch.no_grad():
            positions, weights = self.decoder(torch.as_tensor(c, dtype=torch.float32))
        return pre_image_fields(positions[0].cpu().numpy(), weights[0].cpu().numpy(),
                                self.radius, name)

    def set_code_distribution(self, codes: np.ndarray) -> None:
        """Remember the mean and the spread of the training codes, for :meth:`sample`."""
        c = np.asarray(codes, float)
        with torch.no_grad():
            self.code_mean_.copy_(torch.as_tensor(c.mean(axis=0), dtype=self.code_mean_.dtype))
            sd = c.std(axis=0)
            self.code_sd_.copy_(torch.as_tensor(np.where(sd > 1e-12, sd, 1.0),
                                                dtype=self.code_sd_.dtype))

    def sample(self, n: int = 1, seed: Optional[int] = 0,
               spread: float = 1.0) -> List[MoleculeFields]:
        """`n` decoded fields from codes drawn around the training codes.

        This is **not** the variational prior of version (iii) of the item: the codes are drawn
        from an independent Gaussian fitted to the training codes, which says nothing about
        whether the region between two codes decodes to anything sensible.  It is the pre-image
        entry point the design items need, and it is labelled as what it is.
        """
        require()
        rng = np.random.RandomState(seed if seed is not None else 0)
        mean = self.code_mean_.detach().cpu().numpy()
        sd = self.code_sd_.detach().cpu().numpy() * float(spread)
        out = []
        for i in range(int(n)):
            out.append(self.pre_image(mean + sd * rng.randn(len(mean)), name=f"sample{i}"))
        return out


# --------------------------------------------------------------------------- #
# The features of the encoder
# --------------------------------------------------------------------------- #


def landmark_features(table: AtomTable, fieldset: FieldSet, n_landmarks: int = 256,
                      seed: int = 0, molecules: int = 24, block: int = 256,
                      progress: Optional[Callable[[str, float], None]] = None):
    """(features, the feature map) of every molecule, from the landmarks of item B6.

    The landmarks are the atoms of a few training molecules, thinned so that no two stand on top
    of each other (:func:`cmfqsar.lowrank.landmark_atoms`).  That is the kind of landmark a
    narrow field wants, and it keeps the feature vector to a few hundred columns per field
    whatever the size of the set.
    """
    from . import lowrank

    specs = [s for s in fieldset.specs if s.enabled]
    marks = lowrank.landmark_atoms(table, len(specs), molecules=int(molecules), seed=seed,
                                  max_landmarks=int(n_landmarks))
    low = lowrank.build(marks, fieldset)
    return low.transform(table, block=int(block), progress=progress), low


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


@dataclass
class Training:
    """What a fit did: the error after every epoch and the state that was kept."""
    epochs: int = 0
    best_epoch: int = 0
    train_error: float = float("nan")
    valid_error: float = float("nan")
    history: List[Dict[str, float]] = _field(default_factory=list)
    seconds: float = 0.0
    min_distance: float = float("nan")
    stopped_early: bool = False


def _repulsion(positions, min_distance: float):
    """A penalty on pseudo-atoms closer together than `min_distance`.

    The loss is invariant to the order of the pseudo-atoms, which is right, but it lets several
    of them collapse onto one position, where they act as one wider atom and the rest of the set
    is wasted.  This is the term the item names as the likely remedy.

    The penalty is written in the *squared* distance, ``relu(d_min^2 - d^2)^2``, because the
    derivative of a distance is not defined where two atoms coincide -- which is the very place
    this term has to push them apart from.
    """
    d2 = (positions.unsqueeze(-2) - positions.unsqueeze(-3)).pow(2).sum(-1)
    big = torch.eye(positions.shape[-2], dtype=positions.dtype,
                    device=positions.device) * 1e6
    gap = torch.clamp(float(min_distance) ** 2 - (d2 + big), min=0.0)
    return gap.pow(2).sum(dim=(-2, -1)) * 0.5


def min_pseudo_distance(positions: np.ndarray) -> float:
    """The smallest distance between two pseudo-atoms of a set, which says whether they collapsed."""
    x = np.asarray(positions, float).reshape(-1, positions.shape[-2], 3)
    best = np.inf
    for row in x:
        d = np.linalg.norm(row[:, None, :] - row[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        best = min(best, float(d.min()))
    return best


def fit(ae: "FieldAutoencoder", features: np.ndarray, tens: FieldTensors,
        train_rows: Sequence[int], valid_rows: Optional[Sequence[int]] = None,
        epochs: int = 200, batch: int = 128, lr: float = 1e-3, weight_decay: float = 0.0,
        repulsion: float = 0.0, min_distance: float = 0.75, patience: int = 20,
        loss: str = "relative", seed: int = 0,
        progress: Optional[Callable[[str, float], None]] = None) -> Training:
    """Fit the autoencoder by Adam on the closed-form reconstruction loss.

    `loss` is "relative" -- the mean of ``||f - f'||^2 / ||f||^2`` -- or "absolute", the mean of
    ``||f - f'||^2`` as the item writes it.  The relative form is the default because the
    molecules of a large set differ in size by an order of magnitude, so an absolute loss is
    almost entirely the largest molecules, and because the relative error is what gate (a)
    measures.  `repulsion` weights the penalty of :func:`_repulsion`, which keeps pseudo-atoms
    from collapsing onto one position.

    The state of the best epoch on `valid_rows` is restored at the end, so the returned model is
    the one the validation part chose and not the one the last epoch left.
    """
    require()
    if loss not in ("relative", "absolute"):
        raise ValueError(f"unknown loss {loss!r}; use 'relative' or 'absolute'")
    torch.manual_seed(int(seed))
    rows = np.asarray(train_rows, int)
    val = None if valid_rows is None or not len(valid_rows) else np.asarray(valid_rows, int)
    feats = np.asarray(features, float)
    ae.encoder.set_standardisation(feats[rows])
    opt = torch.optim.Adam(ae.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.RandomState(int(seed))
    hist: List[Dict[str, float]] = []
    best = (np.inf, 0, None)
    t0 = time.time()
    stopped = False

    def batches(idx, size):
        for lo in range(0, len(idx), int(size)):
            yield idx[lo:lo + int(size)]

    for epoch in range(int(epochs)):
        ae.train()
        order = rows[rng.permutation(len(rows))]
        total = 0.0
        seen = 0
        for part in batches(order, batch):
            opt.zero_grad(set_to_none=True)
            x, c, w, r, k = _batch(ae, feats, tens, part)
            sq, positions, _pw = ae.squared_error(x, c, w, r, k)
            value = sq / k if loss == "relative" else sq
            objective = value.mean()
            if repulsion:
                objective = objective + float(repulsion) * _repulsion(positions,
                                                                      min_distance).mean()
            objective.backward()
            opt.step()
            total += float(torch.clamp(sq, min=0.0).div(k).sqrt().sum().item())
            seen += len(part)
        train_err = total / max(seen, 1)
        row = {"epoch": epoch, "train": train_err}
        if val is not None:
            row["valid"] = float(np.mean(relative_errors(ae, feats, tens, val)))
        hist.append(row)
        watch = row.get("valid", train_err)
        if watch < best[0] - 1e-6:
            best = (watch, epoch, {k_: v.detach().clone() for k_, v in ae.state_dict().items()})
        if progress is not None:
            progress(f"epoch {epoch + 1}/{epochs}: {watch:.4f}", (epoch + 1) / max(epochs, 1))
        if patience and epoch - best[1] >= int(patience):
            stopped = True
            break
    if best[2] is not None:
        ae.load_state_dict(best[2])
    ae.set_code_distribution(ae.codes(feats[rows]))
    out = Training(epochs=len(hist), best_epoch=best[1], history=hist,
                   seconds=time.time() - t0, stopped_early=stopped)
    out.train_error = float(np.mean(relative_errors(ae, feats, tens, rows)))
    if val is not None:
        out.valid_error = float(np.mean(relative_errors(ae, feats, tens, val)))
    pos, _w = decode_rows(ae, feats, rows[:min(256, len(rows))])
    out.min_distance = min_pseudo_distance(pos)
    return out


def _batch(ae: "FieldAutoencoder", features: np.ndarray, tens: FieldTensors,
           rows: Sequence[int]):
    """The tensors of one batch of molecules, trimmed to the atoms those molecules really have."""
    idx = np.asarray(rows, int)
    a = int(tens.counts[idx].max())
    f32 = torch.float32
    x = torch.as_tensor(np.asarray(features, float)[idx], dtype=f32)
    c = torch.as_tensor(tens.coords[idx, :a], dtype=f32)
    w = torch.as_tensor(tens.weights[idx, :a], dtype=f32)
    r = torch.as_tensor(tens.radii[idx, :a], dtype=f32)
    k = torch.as_tensor(tens.self_k[idx], dtype=f32)
    return x, c, w, r, k


def relative_errors(ae: "FieldAutoencoder", features: np.ndarray, tens: FieldTensors,
                    rows: Optional[Sequence[int]] = None, batch: int = 256) -> np.ndarray:
    """``||f - f'|| / ||f||`` of every molecule, with no gradient."""
    require()
    idx = np.arange(tens.n_mol) if rows is None else np.asarray(rows, int)
    out = np.zeros(len(idx))
    ae.eval()
    with torch.no_grad():
        for lo in range(0, len(idx), int(batch)):
            part = idx[lo:lo + int(batch)]
            x, c, w, r, k = _batch(ae, features, tens, part)
            rel, _p, _w = ae.relative_error(x, c, w, r, k)
            out[lo:lo + len(part)] = rel.detach().cpu().numpy()
    return out


def decode_rows(ae: "FieldAutoencoder", features: np.ndarray, rows: Sequence[int],
                batch: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """The pseudo-atom positions and weights of several molecules, as NumPy arrays."""
    require()
    idx = np.asarray(rows, int)
    f = np.asarray(features, float)
    pos = np.zeros((len(idx), ae.n_pseudo, 3))
    wts = np.zeros((len(idx), ae.n_pseudo, ae.geom.n_fields))
    ae.eval()
    with torch.no_grad():
        for lo in range(0, len(idx), int(batch)):
            part = idx[lo:lo + int(batch)]
            x = torch.as_tensor(f[part], dtype=torch.float32)
            p, w = ae(x)
            pos[lo:lo + len(part)] = p.cpu().numpy()
            wts[lo:lo + len(part)] = w.cpu().numpy()
    return pos, wts


# --------------------------------------------------------------------------- #
# The floor: what M pseudo-atoms can represent at all
# --------------------------------------------------------------------------- #


def fit_pseudo_atoms(tens: FieldTensors, rows: Sequence[int], geom: FieldGeometry,
                     n_pseudo: int = 32, steps: int = 400, lr: float = 0.05,
                     repulsion: float = 0.0, min_distance: float = 0.75, seed: int = 0,
                     progress: Optional[Callable[[str, float], None]] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit pseudo-atoms to each of several fields directly, with no code in between.

    This is the **floor** of the autoencoder: the reconstruction error that `n_pseudo`
    pseudo-atoms can reach for one field when nothing is shared between molecules and no
    bottleneck restricts them.  It separates the two ways a decoder can fail -- the code is too
    narrow, or the pseudo-atoms cannot represent the field at all -- and the plan's gate (a)
    cannot be read without it.

    It is also useful on its own: it is the pre-image of a *given* field, so it answers "what
    set of atoms would place this field?" for a co-field or a decoded field as well as for a
    molecule.

    Returns (relative errors, positions, weights).  The positions start on the molecule's own
    atoms, which is the best starting point available and keeps the fit from having to find the
    molecule before it can fit it.  With at least as many pseudo-atoms as atoms the start is the
    molecule itself, the spare pseudo-atoms carrying no weight, so the error starts at zero for
    every field of shared width and the fit has only to keep it there.
    """
    require()
    idx = np.asarray(rows, int)
    rng = np.random.RandomState(int(seed))
    M = int(n_pseudo)
    F = geom.n_fields
    pos0 = np.zeros((len(idx), M, 3))
    w0 = np.zeros((len(idx), M, F))
    for i, r in enumerate(idx):
        n = int(tens.counts[r])
        if n >= M:
            # M of the atoms, their weights raised by n/M so that the integral is kept
            pick = rng.choice(n, M, replace=False)
            pos0[i] = tens.coords[r, pick]
            w0[i] = tens.weights[r, pick] * (n / M)
        else:
            pos0[i, :n] = tens.coords[r, :n]
            w0[i, :n] = tens.weights[r, :n]
            spare = rng.choice(n, M - n, replace=True)
            pos0[i, n:] = tens.coords[r, spare] + 0.3 * rng.randn(M - n, 3)
    positions = torch.tensor(pos0, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    a = int(tens.counts[idx].max())
    c = torch.as_tensor(tens.coords[idx, :a], dtype=torch.float64)
    w = torch.as_tensor(tens.weights[idx, :a], dtype=torch.float64)
    r_mol = torch.as_tensor(tens.radii[idx, :a], dtype=torch.float64)
    k_aa = torch.as_tensor(tens.self_k[idx], dtype=torch.float64)
    opt = torch.optim.Adam([positions, weights], lr=float(lr))
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        rp = torch.full(positions.shape[:-1], PSEUDO_RADIUS, dtype=positions.dtype)
        cross = overlap(c, w, r_mol, positions, weights, rp, geom)
        own = overlap(positions, weights, rp, positions, weights, rp, geom)
        sq = k_aa - 2.0 * cross + own
        objective = (sq / k_aa).mean()
        if repulsion:
            objective = objective + float(repulsion) * _repulsion(positions, min_distance).mean()
        objective.backward()
        opt.step()
        if progress is not None and (step % 25 == 0 or step == steps - 1):
            progress(f"step {step + 1}/{steps}", (step + 1) / max(steps, 1))
    with torch.no_grad():
        rp = torch.full(positions.shape[:-1], PSEUDO_RADIUS, dtype=positions.dtype)
        cross = overlap(c, w, r_mol, positions, weights, rp, geom)
        own = overlap(positions, weights, rp, positions, weights, rp, geom)
        sq = torch.clamp(k_aa - 2.0 * cross + own, min=0.0)
        rel = torch.sqrt(sq / k_aa).cpu().numpy()
    return rel, positions.detach().cpu().numpy(), weights.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Drawing and exporting a decoded field
# --------------------------------------------------------------------------- #


def field_width(geom: FieldGeometry, f: int, radius: float = PSEUDO_RADIUS) -> float:
    """The Gaussian width ``b`` of field `f` for a pseudo-atom of this radius."""
    alpha = float(geom.alphas[f])
    return alpha / float(radius) ** 2 if geom.kinds[f] == "steric" else alpha


def decoded_volume(mf: MoleculeFields, geom: FieldGeometry, f: int, grid) -> np.ndarray:
    """Field `f` of a set of pseudo-atoms (or of a molecule) on a grid, for display only.

    A grid never enters the loss -- that is rule 3 -- but a chemist has to see the field, and
    this is the same evaluation :mod:`cmfqsar.cofields` uses for a co-field.
    """
    from .cofields import evaluate_gaussians

    x = np.asarray(mf.coords, float)
    w = np.asarray(mf.weights, float)[:, int(f)]
    if geom.kinds[int(f)] == "steric":
        widths = float(geom.alphas[int(f)]) / np.maximum(np.asarray(mf.radii, float), 1e-6) ** 2
    else:
        widths = np.full(len(x), float(geom.alphas[int(f)]))
    return evaluate_gaussians(grid.points(), x, w, widths).reshape(grid.shape)


def export_decoded(path: str, mf: MoleculeFields, geom: FieldGeometry, f: int,
                   spacing: float = 0.5, margin: float = 3.0,
                   title: str = "CMF decoded field") -> str:
    """Write field `f` of a decoded set of pseudo-atoms as a Gaussian cube file.

    The pseudo-atoms are written as dummy atoms of atomic number zero, because they are not
    elements; a viewer shows them as points with the field around them.
    """
    import dataclasses

    from .cofields import export_cube, make_grid

    grid = make_grid(np.asarray(mf.coords, float), margin=margin, spacing=spacing)
    volume = decoded_volume(mf, geom, f, grid)
    dummy = dataclasses.replace(mf, symbols=["*"] * len(mf.coords))
    export_cube(path, grid, volume, dummy, title=f"{title}: {geom.names[int(f)]}")
    return path


# --------------------------------------------------------------------------- #
# What the autoencoder is compared with: kernel PCA
# --------------------------------------------------------------------------- #


def kpca_errors(K_train: np.ndarray, K_test: np.ndarray, k_self: np.ndarray,
                components: Sequence[int]) -> Dict[int, np.ndarray]:
    """:func:`kpca_relative_errors` for several numbers of components, from one eigendecomposition.

    The eigendecomposition of the centred training kernel is the whole cost of kernel PCA -- a
    minute and a half on seven thousand molecules -- and it does not depend on how many
    components are then kept, so a comparison over a grid of latent dimensions must do it once.
    """
    Ktr = np.asarray(K_train, float)
    Kt = np.atleast_2d(np.asarray(K_test, float))
    d = np.asarray(k_self, float)
    n = Ktr.shape[0]
    one = np.full((n, n), 1.0 / n)
    Kc = Ktr - one @ Ktr - Ktr @ one + one @ Ktr @ one
    w, V = np.linalg.eigh(Kc)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    kmean = Ktr.mean(axis=1)
    ktotal = float(Ktr.mean())
    kc = Kt - kmean[None, :] - Kt.mean(axis=1)[:, None] + ktotal
    centred_self = d - 2.0 * Kt.mean(axis=1) + ktotal
    out: Dict[int, np.ndarray] = {}
    for q in components:
        q = int(max(0, min(int(q), n - 1)))
        wq, Vq = w[:q], V[:, :q]
        keep = wq > (max(float(w[0]), 1e-300) * 1e-12 if q else 0.0)
        vecs = Vq[:, keep] / np.sqrt(np.maximum(wq[keep], 1e-300)) if q else np.zeros((n, 0))
        proj = kc @ vecs
        err2 = np.maximum(centred_self - np.sum(proj * proj, axis=1), 0.0)
        out[q] = np.sqrt(err2 / np.maximum(d, 1e-30))
    return out


def kpca_relative_errors(K_train: np.ndarray, K_test: np.ndarray, k_self: np.ndarray,
                         components: int) -> np.ndarray:
    """``||f - f'|| / ||f||`` of kernel PCA with `components` latent dimensions.

    The reconstruction is the training mean plus the projection on the first `components`
    principal directions of the centred training kernel (Hoffmann 2007, the ``ockpca`` score of
    :mod:`cmfqsar.oneclass` without its offset).  This is the number gate (a) of item E11
    compares the decoder with, and the comparison is not quite even: kernel PCA gets the
    training mean for free on top of its `components` directions, and its reconstruction is a
    sum of training fields, which is not a set of atoms and cannot be sampled from.

    `K_train` is the training kernel, `K_test` the test-against-training block, `k_self` the
    test molecules' own ``k(x, x)``; all in one scaling.
    """
    q = int(max(0, min(int(components), np.asarray(K_train).shape[0] - 1)))
    return kpca_errors(K_train, K_test, k_self, [q])[q]
