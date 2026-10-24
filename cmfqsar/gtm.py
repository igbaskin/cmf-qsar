"""
Chemical space and activity landscapes from CMF kernels: kernel PCA + Generative Topographic
Mapping.

GTM (Bishop, Svensen & Williams, Neural Comput. 10 (1998) 215) is a probabilistic latent-variable
map: a square grid of K nodes in a two-dimensional latent space is mapped into data space by
y(x) = W phi(x) with radial basis functions phi, and the data are modelled as a constrained mixture
of Gaussians centred on the mapped nodes, fitted by expectation-maximisation.  Each molecule
therefore gets a posterior distribution over the nodes (its "responsibility"), from which the
chemical-space map, the activity or class landscapes and the applicability domain follow
(Gaspar, Baskin, Marcou, Horvath & Varnek, J. Chem. Inf. Model. 55 (2015) 84; Mol. Inf. 34 (2015) 348).

Molecular fields are not vectors, so GTM cannot be applied to them directly; only the kernel is
available.  The CMF kernel is an inner product in the Hilbert space of the fields, so kernel PCA on
it is functional PCA, and its leading components are coordinates that reproduce the feature-space
distances exactly.  Running the ordinary GTM in those coordinates is equivalent to a kernelised GTM
(cf. Olier, Vellido & Giraldo, ESANN 2010) and keeps the out-of-sample projection a one-liner, which
a map needs in order to place new molecules.

Two practical points, both learned the hard way elsewhere in this program:

* use cosine-normalised kernels, otherwise the leading components follow molecular size;
* a CMF kernel between molecules that are not aligned in a common frame is meaningless, so a map
  spanning several series would partly show alignment quality.  The escape is the limit of small
  attenuation: as alpha -> 0 a continuous indicator field tends to the count of atoms of that type
  (Sitnikov et al. 2015), so the kernel becomes an alignment-free fragment-descriptor kernel and the
  map can span a whole library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Kernel PCA (functional PCA of the molecular fields)
# --------------------------------------------------------------------------- #

@dataclass
class KernelPCA:
    """Principal components of a set of molecules described by a kernel matrix."""
    n_components: int = 20
    alphas: Optional[np.ndarray] = None       # (N, d) eigenvectors scaled to unit length
    lambdas: Optional[np.ndarray] = None      # (d,) eigenvalues
    K_fit_rows_: Optional[np.ndarray] = None  # row means of the training kernel
    K_fit_all_: float = 0.0                   # grand mean of the training kernel

    def fit(self, K: np.ndarray) -> "KernelPCA":
        K = np.asarray(K, dtype=float)
        n = K.shape[0]
        self.K_fit_rows_ = K.mean(axis=0)
        self.K_fit_all_ = float(K.mean())
        Kc = K - self.K_fit_rows_[None, :] - self.K_fit_rows_[:, None] + self.K_fit_all_
        w, v = np.linalg.eigh((Kc + Kc.T) / 2.0)
        order = np.argsort(-w)
        w, v = w[order], v[:, order]
        keep = min(self.n_components, int(np.sum(w > 1e-10)))
        keep = max(keep, 1)
        self.lambdas = w[:keep]
        # dual coefficients: alpha_k = u_k / sqrt(lambda_k), so that projecting a centred kernel
        # row gives sqrt(lambda_k) u_k both in and out of sample
        self.alphas = v[:, :keep] / np.sqrt(np.maximum(w[:keep], 1e-300))[None, :]
        return self

    def transform(self, K_new: Optional[np.ndarray] = None) -> np.ndarray:
        """Coordinates of molecules whose kernels against the training set are `K_new` (M, N).

        With ``K_new=None`` the coordinates of the training molecules themselves are returned.
        """
        if K_new is None:
            return self.alphas * self.lambdas[None, :]       # = sqrt(lambda_k) u_k
        K_new = np.atleast_2d(np.asarray(K_new, dtype=float))
        Kc = (K_new - self.K_fit_rows_[None, :]
              - K_new.mean(axis=1)[:, None] + self.K_fit_all_)
        return Kc @ self.alphas

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        tot = float(np.sum(np.maximum(self.lambdas, 0.0)))
        return np.maximum(self.lambdas, 0.0) / tot if tot > 0 else np.zeros_like(self.lambdas)


# --------------------------------------------------------------------------- #
# Generative Topographic Mapping
# --------------------------------------------------------------------------- #

def _grid(n: int) -> np.ndarray:
    a = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(a, a, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


@dataclass
class GTM:
    """Generative Topographic Mapping fitted by expectation-maximisation.

    Parameters
    ----------
    grid : the map is grid x grid nodes.
    rbf : rbf x rbf radial basis functions constrain the mapping (its smoothness).
    rbf_width : width of those functions, in units of the spacing between their centres.
    reg : weight-decay regularisation of the mapping (the variational/regularised GTM of
        Olier & Vellido guards against overfitting; this is its simple quadratic form).
    """
    grid: int = 20
    rbf: int = 5
    rbf_width: float = 1.0
    reg: float = 0.1
    n_iter: int = 100
    tol: float = 1e-6
    seed: int = 0

    W: Optional[np.ndarray] = None            # (M, D) mapping weights
    beta: float = 1.0                         # inverse noise variance
    Phi: Optional[np.ndarray] = None          # (K, M) basis functions on the latent grid
    latent: Optional[np.ndarray] = None       # (K, 2) latent coordinates of the nodes
    log_likelihood_: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------ setup
    def _basis(self) -> np.ndarray:
        self.latent = _grid(self.grid)
        centres = _grid(self.rbf)
        spacing = 2.0 / max(self.rbf - 1, 1)
        sigma = self.rbf_width * spacing
        d2 = ((self.latent[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        Phi = np.exp(-d2 / (2.0 * sigma ** 2))
        return np.column_stack([Phi, np.ones(len(Phi))])      # bias term

    def _node_positions(self) -> np.ndarray:
        return self.Phi @ self.W

    # ------------------------------------------------------------------ fitting
    def fit(self, T: np.ndarray, progress: Optional[Callable[[int, int, float], None]] = None) -> "GTM":
        T = np.asarray(T, dtype=float)
        n, d = T.shape
        self.Phi = self._basis()
        k, m = self.Phi.shape

        # initialise the mapping so that the latent grid lies on the first two principal axes
        mean = T.mean(axis=0)
        X = T - mean
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        axes = vt[:2].T * (s[:2] / max(np.sqrt(n), 1.0))      # (d, 2)
        target = self.latent @ axes.T + mean                  # (k, d)
        self.W = np.linalg.lstsq(self.Phi, target, rcond=None)[0]
        y = self._node_positions()
        # noise level: the larger of the third principal variance and half the node spacing
        var3 = float(s[2] ** 2 / n) if len(s) > 2 else 0.0
        dist = np.sqrt(((y[:, None, :] - y[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        self.beta = 1.0 / max(var3, (0.5 * float(dist.min())) ** 2, 1e-12)

        self.log_likelihood_ = []
        prev = -np.inf
        for it in range(self.n_iter):
            R, ll = self._responsibilities(T, y, return_ll=True)
            self.log_likelihood_.append(ll)
            G = R.sum(axis=1)                                  # (k,)
            A = self.Phi.T @ (G[:, None] * self.Phi) + (self.reg / self.beta) * np.eye(m)
            B = self.Phi.T @ (R @ T)
            self.W = np.linalg.solve(A, B)
            y = self._node_positions()
            d2 = ((T[None, :, :] - y[:, None, :]) ** 2).sum(axis=2)
            self.beta = (n * d) / max(float(np.sum(R * d2)), 1e-300)
            if progress is not None and (it % 5 == 0 or it == self.n_iter - 1):
                progress(it + 1, self.n_iter, ll)
            if it > 0 and abs(ll - prev) < self.tol * max(1.0, abs(prev)):
                break
            prev = ll
        return self

    # ------------------------------------------------------------------ inference
    def _responsibilities(self, T: np.ndarray, y: Optional[np.ndarray] = None, return_ll: bool = False):
        if y is None:
            y = self._node_positions()
        n, d = T.shape
        k = y.shape[0]
        d2 = ((T[None, :, :] - y[:, None, :]) ** 2).sum(axis=2)        # (k, n)
        logp = -0.5 * self.beta * d2
        mx = logp.max(axis=0, keepdims=True)
        e = np.exp(logp - mx)
        tot = e.sum(axis=0, keepdims=True)
        R = e / np.maximum(tot, 1e-300)
        if not return_ll:
            return R
        ll = float(np.sum(mx + np.log(np.maximum(tot, 1e-300)))
                   + n * (0.5 * d * np.log(self.beta / (2.0 * np.pi)) - np.log(k)))
        return R, ll

    def responsibilities(self, T: np.ndarray) -> np.ndarray:
        """Posterior probability of every node for every molecule, shape (K, N)."""
        return self._responsibilities(np.atleast_2d(np.asarray(T, dtype=float)))

    def transform(self, T: np.ndarray) -> np.ndarray:
        """Posterior mean position of every molecule in the latent square, shape (N, 2)."""
        R = self.responsibilities(T)
        return R.T @ self.latent

    def modes(self, T: np.ndarray) -> np.ndarray:
        """Most probable node of every molecule, shape (N, 2)."""
        R = self.responsibilities(T)
        return self.latent[np.argmax(R, axis=0)]


# --------------------------------------------------------------------------- #
# Landscapes
# --------------------------------------------------------------------------- #

def density(R: np.ndarray) -> np.ndarray:
    """Cumulated responsibility of every node: the data support, i.e. applicability domain."""
    return R.sum(axis=1)


def landscape(R: np.ndarray, values: np.ndarray, min_density: float = 1e-3) -> np.ndarray:
    """Responsibility-weighted mean of `values` at every node; NaN where there is no support."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    num = R[:, ok] @ v[ok]
    den = R[:, ok].sum(axis=1)
    out = np.full(R.shape[0], np.nan)
    good = den > min_density
    out[good] = num[good] / den[good]
    return out


def class_landscape(R: np.ndarray, classes: np.ndarray, min_density: float = 1e-3) -> np.ndarray:
    """Responsibility-weighted fraction of the positive class at every node (NaN without support)."""
    c = np.asarray(classes, dtype=float)
    return landscape(R, np.where(np.isfinite(c), (c > 0).astype(float), np.nan), min_density)


def predict_from_landscape(R_new: np.ndarray, land: np.ndarray) -> np.ndarray:
    """Predict a property of new molecules as their responsibility-weighted landscape value."""
    ok = np.isfinite(land)
    num = R_new[ok].T @ land[ok]
    den = R_new[ok].sum(axis=0)
    out = np.full(R_new.shape[1], np.nan)
    good = den > 1e-12
    out[good] = num[good] / den[good]
    return out


# --------------------------------------------------------------------------- #
# Putting it together for CMF
# --------------------------------------------------------------------------- #

@dataclass
class CMFMap:
    """A chemical-space map of a set of molecules described by CMF kernels.

    Workflow: ``fit(table, fieldset)`` computes the kernel matrix, its principal components and the
    GTM; ``project(other)`` places further molecules on the same map.
    """
    n_components: int = 20
    grid: int = 20
    rbf: int = 5
    rbf_width: float = 1.0
    reg: float = 0.1
    n_iter: int = 100
    normalize: bool = True
    seed: int = 0

    kpca: Optional[KernelPCA] = None
    gtm: Optional[GTM] = None
    train_: Optional[object] = None           # AtomTable of the fitted molecules
    fieldset_: Optional[object] = None
    coords_: Optional[np.ndarray] = None      # latent coordinates of the fitted molecules
    R_: Optional[np.ndarray] = None           # their responsibilities
    scores_: Optional[np.ndarray] = None      # their functional principal component scores

    # ------------------------------------------------------------------ kernels
    def _kernel(self, P, Q=None):
        from .kernels import kernel_matrices, normalize_kernels, self_kernels
        specs = self.fieldset_.specs
        Q = P if Q is None else Q
        K = kernel_matrices(P, Q, specs)
        if self.normalize:
            K = normalize_kernels(K, self_kernels(P, specs), self_kernels(Q, specs))
        h = self.fieldset_.mixing_vector()
        return np.tensordot(h / max(h.sum(), 1e-12), K, axes=(0, 0))

    def fit_pca(self, table, fieldset,
                progress: Optional[Callable[[str, float], None]] = None) -> "CMFMap":
        """The kernel and its functional principal components, without the map.

        The score plot needs no GTM, and the GTM is the slow half -- a hundred EM iterations
        against one eigendecomposition.  It also leaves ``gtm`` and ``R_`` as None, so a map
        fitted this far cannot be mistaken for one that can carry a landscape.
        """
        self.train_, self.fieldset_ = table, fieldset
        if progress:
            progress("Computing the kernel matrix", 0.2)
        K = self._kernel(table)
        if progress:
            progress("Functional principal components", 0.7)
        self.kpca = KernelPCA(n_components=self.n_components).fit(K)
        self.scores_ = self.kpca.transform()
        return self

    def scores(self, i: int = 0, j: int = 1) -> np.ndarray:
        """The fitted molecules on two principal components, as an (N, 2) array.

        Unlike the latent coordinates of the map, these are distances one may read: the
        projection is linear and orthogonal, so the plane is a faithful shadow of the kernel
        geometry as far as its two eigenvalues reach.
        """
        if self.scores_ is None:
            raise RuntimeError("fit the map first")
        d = self.scores_.shape[1]
        if not (0 <= i < d and 0 <= j < d):
            raise IndexError(f"only {d} components were kept")
        return np.column_stack([self.scores_[:, i], self.scores_[:, j]])

    def fit(self, table, fieldset, progress: Optional[Callable[[str, float], None]] = None) -> "CMFMap":
        self.fit_pca(table, fieldset,
                     progress=(lambda m, f: progress(m, 0.6 * f)) if progress else None)
        T = self.scores_
        if progress:
            progress("Fitting the map", 0.6)
        self.gtm = GTM(grid=self.grid, rbf=self.rbf, rbf_width=self.rbf_width, reg=self.reg,
                       n_iter=self.n_iter, seed=self.seed)
        self.gtm.fit(T, progress=(lambda i, n, ll: progress(f"GTM iteration {i}/{n} (log L {ll:.1f})",
                                                            0.6 + 0.4 * i / n)) if progress else None)
        self.R_ = self.gtm.responsibilities(T)
        self.coords_ = self.gtm.transform(T)
        if progress:
            progress("Done", 1.0)
        return self

    def project(self, table):
        """(latent coordinates, responsibilities) of further molecules on the fitted map."""
        K = self._kernel(table, self.train_)
        T = self.kpca.transform(K)
        R = self.gtm.responsibilities(T)
        return self.gtm.transform(T), R

    # ------------------------------------------------------------------ landscapes
    def activity_landscape(self, values, min_density: float = 1e-3) -> np.ndarray:
        return landscape(self.R_, values, min_density)

    def class_landscape(self, classes, min_density: float = 1e-3) -> np.ndarray:
        return class_landscape(self.R_, classes, min_density)

    def density(self) -> np.ndarray:
        return density(self.R_)

    def as_image(self, node_values: np.ndarray) -> np.ndarray:
        """Reshape per-node values into the grid, ready to be drawn."""
        return np.asarray(node_values, dtype=float).reshape(self.grid, self.grid)
