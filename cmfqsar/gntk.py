"""The graph neural tangent kernel of the bond graph, as a kernel view (plan item ND7).

The field kernel of this program is the kernel of an infinitely wide radial-basis network on
the aligned fields. The same limit exists for a network on the bond graph: the graph neural
tangent kernel (Du, Hou, Salakhutdinov, Poczos, Wang and Xu, NeurIPS 2019) is the kernel of an
infinitely wide graph network *at initialisation*, computed in closed form from the graph with
no training at all. Added to the field kernels as one more view, and weighed with them by the
multiple kernel learning of item E6, it contributes what the bond graph knows and the aligned
field does not -- which atom is bonded to which -- at no data cost, and the model stays a kernel
model.

**The recursion.** Two molecules with ``n`` and ``m`` atoms give an ``n x m`` matrix of atom
pairs. Each atom starts with a feature vector (one hot over its atom type), so the covariance of
the pair ``(u, v)`` starts at the inner product of their features. One *block* of the network
does two things to every pair: it aggregates over the bonded neighbours of ``u`` and of ``v``,
which is the adjacency with self-loops applied on both sides, and it passes through a layer of
rectified units, whose infinite-width expectations have the closed form of the arc-cosine kernel
of Cho and Saul. The tangent kernel is carried through the same steps as the sum of the
covariance and the previous tangent kernel times the derivative term, as for any network. After
``blocks`` such blocks the graph kernel is the sum over every atom pair, and it is normalised
to a cosine so that it compares with the cosine-normalised field kernels of the model.

The expectations need each atom's own variance at every step, which is the same recursion run
on the molecule against itself; it is done once per molecule and reused for every pair.

**What this is not.** A graph kernel knows nothing of where the atoms are, so it brings no
co-field and no atom-wise split in space; the co-field of a mixed model is read from the field
views alone, and the Shapley value of the graph view is per molecule. It is one more term under
the weights of item E6, and the gate of item ND7 is whether that term adds anything the invariant
kernel of item C5 does not already give.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: how many blocks of aggregation and rectification the infinitely wide network has
DEFAULT_BLOCKS = 2
#: the variance kept by a rectified layer: 2 E[relu(a) relu(b)], so that a unit variance is a
#: unit variance after the layer (the He convention)
C_SIGMA = 2.0


# --------------------------------------------------------------------------- #
# One molecule as a graph
# --------------------------------------------------------------------------- #
class Graph:
    """The bond graph of one molecule: its adjacency with self-loops, scaled for the mean over
    the neighbourhood, and its atoms' features; with the within-molecule recursion cached."""

    def __init__(self, features: np.ndarray, bonds: np.ndarray, name: str = ""):
        X = np.asarray(features, float)
        n = X.shape[0]
        A = np.eye(n)
        for a, b in np.asarray(bonds, int).reshape(-1, 2):
            if 0 <= a < n and 0 <= b < n and a != b:
                A[a, b] = A[b, a] = 1.0
        # the mean over the neighbourhood (self included), so that a heavy atom with four
        # neighbours and a hydrogen with one are on the same scale
        self.A = A / A.sum(axis=1, keepdims=True)
        self.X = X
        self.name = name
        self.n = n
        self._self: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    def self_terms(self, blocks: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """The within-molecule recursion: (covariance, tangent kernel) of every atom pair of
        this molecule after each aggregation, whose diagonals the expectations need."""
        if self._self is None or len(self._self) != blocks:
            self._self = _recurse(self, self, int(blocks), keep=True)
        return self._self


def _arccos_terms(sigma: np.ndarray, var_u: np.ndarray, var_v: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """E[relu(a) relu(b)] and E[relu'(a) relu'(b)] for Gaussian (a, b) with covariance `sigma`
    and variances `var_u`, `var_v`, in the C_SIGMA convention: the arc-cosine kernel of Cho and
    Saul of degree one, and its derivative term."""
    root = np.sqrt(np.maximum(var_u[:, None] * var_v[None, :], 1e-300))
    cos = np.clip(sigma / root, -1.0, 1.0)
    theta = np.arccos(cos)
    new_sigma = C_SIGMA * root / (2.0 * np.pi) * (np.sin(theta) + (np.pi - theta) * cos)
    dot_sigma = C_SIGMA * (np.pi - theta) / (2.0 * np.pi)
    return new_sigma, dot_sigma


def _recurse(g: Graph, h: Graph, blocks: int, keep: bool = False):
    """The GNTK recursion between two molecules.

    Returns the final (covariance, tangent kernel) over the atom pairs, or with `keep` the list
    of the two after every aggregation, which is what a molecule against itself stores: the
    diagonals of those are the variances the expectations between two molecules need.
    """
    sigma = g.X @ h.X.T
    theta = sigma.copy()
    kept = []
    same = g is h
    if not same:
        own_g = g.self_terms(blocks)
        own_h = h.self_terms(blocks)
    for b in range(blocks):
        # aggregation over the neighbourhoods on both sides
        sigma = g.A @ sigma @ h.A.T
        theta = g.A @ theta @ h.A.T
        if keep:
            kept.append((sigma.copy(), theta.copy()))
        if same:
            var_u = var_v = np.diag(sigma)
        else:
            var_u = np.diag(own_g[b][0])
            var_v = np.diag(own_h[b][0])
        new_sigma, dot_sigma = _arccos_terms(sigma, var_u, var_v)
        theta = theta * dot_sigma + new_sigma
        sigma = new_sigma
    return kept if keep else (sigma, theta)


def graph_kernel(g: Graph, h: Graph, blocks: int = DEFAULT_BLOCKS) -> float:
    """The unnormalised graph neural tangent kernel of two molecules: the sum of the tangent
    kernel over every atom pair after `blocks` blocks."""
    _sigma, theta = _recurse(g, h, int(blocks))
    return float(theta.sum())


# --------------------------------------------------------------------------- #
# The features of the atoms, and the kernel over a table
# --------------------------------------------------------------------------- #
def atom_feature_table(types: Sequence[str]) -> Dict[str, int]:
    """One column per atom type seen, in alphabetical order."""
    return {t: i for i, t in enumerate(sorted(set(str(x) for x in types)))}


def graphs_of(table, blocks: int = DEFAULT_BLOCKS,
              columns: Optional[Dict[str, int]] = None) -> Tuple[List[Graph], Dict[str, int]]:
    """One :class:`Graph` per molecule of an :class:`~cmfqsar.fields.AtomTable`.

    The atom features are one hot over the atom types the table carries (the Sybyl types the
    fields are built with, which say element, hybridisation and aromaticity at once); `columns`
    fixes the columns to those of another table, so that a test set is featurised like the
    training set it is compared with, and a type the training set never saw gets no column.
    """
    mols = list(table.molecules)
    if columns is None:
        columns = atom_feature_table([t for m in mols for t in m.atom_types])
    out = []
    for m in mols:
        X = np.zeros((m.n_atoms, len(columns)))
        for i, t in enumerate(m.atom_types):
            j = columns.get(str(t))
            if j is not None:
                X[i, j] = 1.0
        g = Graph(X, np.asarray(m.bonds, int), name=m.name)
        g.self_terms(int(blocks))
        out.append(g)
    return out, columns


def gntk_matrix(graphs_a: Sequence[Graph], graphs_b: Optional[Sequence[Graph]] = None,
                blocks: int = DEFAULT_BLOCKS, normalize: bool = True,
                progress=None) -> np.ndarray:
    """The kernel matrix between two lists of graphs (or one list against itself).

    With `normalize` every entry is divided by the square root of the two self-kernels, which
    puts the view on the same scale as the cosine-normalised field kernels it is mixed with.
    """
    A = list(graphs_a)
    symmetric = graphs_b is None
    B = A if symmetric else list(graphs_b)
    K = np.zeros((len(A), len(B)))
    for i, g in enumerate(A):
        start = i if symmetric else 0
        for j in range(start, len(B)):
            K[i, j] = graph_kernel(g, B[j], blocks)
            if symmetric:
                K[j, i] = K[i, j]
        if progress is not None and i % 25 == 0:
            progress(f"Graph kernel {i + 1}/{len(A)}", i / max(len(A), 1))
    if not normalize:
        return K
    da = np.array([graph_kernel(g, g, blocks) for g in A]) if not symmetric else np.diag(K).copy()
    db = da if symmetric else np.array([graph_kernel(h, h, blocks) for h in B])
    denom = np.sqrt(np.maximum(da[:, None] * db[None, :], 1e-300))
    return K / denom


def gntk_kernel(table, blocks: int = DEFAULT_BLOCKS, normalize: bool = True,
                progress=None) -> np.ndarray:
    """The graph neural tangent kernel over the molecules of one table, ``(n, n)``."""
    graphs, _cols = graphs_of(table, blocks)
    return gntk_matrix(graphs, None, blocks=blocks, normalize=normalize, progress=progress)


def gntk_cross(train, test, blocks: int = DEFAULT_BLOCKS, normalize: bool = True) -> np.ndarray:
    """The kernel of the molecules of `test` against those of `train`, ``(n_test, n_train)``,
    with the test molecules featurised by the training table's atom types."""
    g_train, cols = graphs_of(train, blocks)
    g_test, _ = graphs_of(test, blocks, columns=cols)
    return gntk_matrix(g_test, g_train, blocks=blocks, normalize=normalize)
