"""The graph neural tangent kernel as a kernel view (plan item ND7).

What has to be true, each checked against something the kernel does not compute itself:

* the kernel is a kernel: symmetric, positive definite, one on the diagonal once normalised;
* it is a function of the graph and not of the atom order, and two molecules with the same
  bond graph and types are identical to it whatever their coordinates;
* the recursion is the one written down: with no blocks it is the inner product of the atom
  features summed over the pairs, and the rectified layer's expectations are the arc-cosine
  terms of Cho and Saul, checked against a Monte Carlo estimate;
* a test table is featurised with the training table's atom types, so a type never seen gets
  no column and the cross kernel is the matching block of the symmetric one;
* stacked under the mixing of item E6, the view gets a weight when it explains the activities
  and is reduced when it does not.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cmfqsar import gntk as G  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.mixing import evidence_mixing, joint_kernel, with_views  # noqa: E402

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O", "CCCCO",
          "CCCCN", "CC(C)CO", "c1ccc(F)cc1O", "c1ccc(Cl)cc1N", "CCOCC", "CCCCC"]


def _table(smiles=SMILES, seed=1):
    fs = FieldSet.create(physchem=["steric"], alpha=1.0)
    mols = []
    for k, s in enumerate(smiles):
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        AllChem.EmbedMolecule(m, randomSeed=seed + k)
        mols.append(build_fields(m, fs, f"m{k}"))
    return AtomTable.from_molecules(mols)


# ------------------------------------------------------------------ a kernel
def test_the_kernel_is_a_kernel():
    K = G.gntk_kernel(_table())
    assert K.shape == (len(SMILES), len(SMILES))
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K).min() > 0.0
    assert np.all(K <= 1.0 + 1e-12) and np.all(K > 0.0)
    raw = G.gntk_kernel(_table(), normalize=False)
    assert np.all(np.diag(raw) > 1.0)           # the sum over atom pairs, before normalising


def test_the_kernel_depends_on_the_graph_and_not_on_the_order_or_the_coordinates():
    table = _table()
    graphs, _cols = G.graphs_of(table)
    g = graphs[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(g.n)
    inv = np.argsort(perm)
    relabelled = G.Graph(g.X[perm], inv[np.asarray(table.molecules[0].bonds, int)])
    assert G.graph_kernel(relabelled, relabelled) == pytest.approx(G.graph_kernel(g, g))
    assert G.graph_kernel(relabelled, graphs[1]) == pytest.approx(G.graph_kernel(g, graphs[1]))
    # the same molecules embedded with other coordinates give the same kernel
    K1 = G.gntk_kernel(_table(seed=1))
    K2 = G.gntk_kernel(_table(seed=7))
    assert np.abs(K1 - K2).max() < 1e-12


def test_like_molecules_are_closer_than_unlike_ones():
    K = G.gntk_kernel(_table())
    names = SMILES
    phenol, aniline, butanol, pentane = (names.index(s) for s in
                                         ("c1ccccc1O", "c1ccccc1N", "CCCCO", "CCCCC"))
    assert K[phenol, aniline] > K[phenol, butanol]
    assert K[butanol, pentane] > K[butanol, phenol]


# ------------------------------------------------------------------ the recursion
def test_no_blocks_is_the_sum_of_the_feature_inner_products():
    table = _table(SMILES[:3])
    graphs, _cols = G.graphs_of(table, blocks=0)
    for g, h in ((graphs[0], graphs[1]), (graphs[2], graphs[2])):
        assert G.graph_kernel(g, h, blocks=0) == pytest.approx(float((g.X @ h.X.T).sum()))


def test_the_rectified_layers_expectations_are_the_arc_cosine_terms():
    """E[relu(a) relu(b)] and E[relu'(a) relu'(b)] for a Gaussian pair, by Monte Carlo."""
    rng = np.random.default_rng(0)
    var_u, var_v, cov = 1.5, 0.8, 0.6
    L = np.linalg.cholesky(np.array([[var_u, cov], [cov, var_v]]))
    z = rng.normal(size=(400000, 2)) @ L.T
    a, b = z[:, 0], z[:, 1]
    mc_sigma = G.C_SIGMA * np.mean(np.maximum(a, 0) * np.maximum(b, 0))
    mc_dot = G.C_SIGMA * np.mean((a > 0) & (b > 0))
    new_sigma, dot_sigma = G._arccos_terms(np.array([[cov]]), np.array([var_u]),
                                           np.array([var_v]))
    assert new_sigma[0, 0] == pytest.approx(mc_sigma, rel=0.02)
    assert dot_sigma[0, 0] == pytest.approx(mc_dot, rel=0.02)
    # a unit variance stays a unit variance through the layer
    s, _d = G._arccos_terms(np.array([[1.0]]), np.array([1.0]), np.array([1.0]))
    assert s[0, 0] == pytest.approx(1.0)


def test_more_blocks_see_further_along_the_graph():
    """Two molecules that differ only far from most atoms look more alike to a shallow kernel."""
    table = _table(["CCCCCCCCO", "CCCCCCCCN"])
    shallow = G.gntk_kernel(table, blocks=1)[0, 1]
    deep = G.gntk_kernel(table, blocks=4)[0, 1]
    assert 0.0 < deep < 1.0 and 0.0 < shallow < 1.0
    assert deep != pytest.approx(shallow)


# ------------------------------------------------------------------ tables
def test_a_test_table_is_featurised_like_the_training_table():
    train = _table(SMILES[:8])
    test = _table(SMILES[8:], seed=3)
    K = G.gntk_kernel(_table(SMILES, seed=1))
    graphs, cols = G.graphs_of(train)
    g_test, cols2 = G.graphs_of(test, columns=cols)
    assert cols2 is cols
    # the fluorine of the ninth molecule is a type the training part never saw: no column
    assert "F" not in "".join(cols) or all("F" not in c for c in cols)
    cross = G.gntk_cross(train, test)
    assert cross.shape == (len(SMILES) - 8, 8)
    # where no new type is involved the cross kernel is the block of the symmetric one
    sym = G.gntk_matrix(G.graphs_of(_table(SMILES, seed=1))[0])
    assert cross[2, 0] == pytest.approx(sym[10, 0], abs=1e-9)   # ether against phenol
    assert np.all(cross > 0.0) and np.all(cross <= 1.0 + 1e-12)
    assert K.shape == (len(SMILES), len(SMILES))


# ------------------------------------------------------------------ as a view
def test_the_view_gets_a_weight_when_it_explains_the_activities():
    rng = np.random.default_rng(0)
    n, d = 60, 6
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Z = rng.normal(size=(n, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    fields = np.stack([X @ X.T])
    # the activities depend on the view alone: the view takes the weight
    y = Z @ rng.normal(size=d) + rng.normal(scale=0.2, size=n)
    stack, names, scale = with_views(fields, {"gntk": Z @ Z.T}, names=["steric"])
    assert names == ["steric", "gntk"] and stack.shape == (2, n, n)
    assert np.allclose(scale, 1.0)                 # unit diagonals, so unit scales
    h = evidence_mixing(stack, y, scale=scale)
    assert h[1] > 3.0 * h[0]
    assert joint_kernel(stack, h, scale).shape == (n, n)
    # and the other way round
    y2 = X @ rng.normal(size=d) + rng.normal(scale=0.2, size=n)
    h2 = evidence_mixing(stack, y2, scale=scale)
    assert h2[0] > 3.0 * h2[1]


def test_what_a_view_must_be():
    K = np.eye(5)
    with pytest.raises(ValueError, match="names for"):
        with_views(np.stack([K]), {}, names=["a", "b"])
    with pytest.raises(ValueError, match="over 5 molecules"):
        with_views(np.stack([K]), {"v": np.eye(4)}, names=["a"])
    with pytest.raises(ValueError, match="already in the stack"):
        with_views(np.stack([K]), {"a": K}, names=["a"])
    stack, names, scale = with_views(K, {"v": 2.0 * K})
    assert names == ["field 0", "v"] and scale[1] == pytest.approx(0.5)
