"""The fast paths of the field kernel must give the same numbers as the plain one (item B1).

Three shortcuts are tested here, and all three are exact rather than approximate, so each test
compares against the slow route it replaces instead of against a tolerance chosen to pass:

* the symmetric training kernel, which evaluates only the pairs with j >= i;
* ``z * sqrt(z)`` in place of ``z ** 1.5`` in the radius-scaled prefactor;
* one distance block per molecule shared by a grid of attenuations.
"""
import math
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, FieldSpec, build_fields  # noqa: E402
from cmfqsar.kernels import (_pow32, atom_kernel_block, kernel_matrices,  # noqa: E402
                             kernel_matrices_multi, pair_distances, self_kernels,
                             self_kernels_multi)

SMILES = ["c1ccccc1C(=O)O", "c1ccccc1CCN", "c1ccc(Cl)cc1C(=O)N", "CC(C)c1ccccc1O",
          "c1ccncc1CC(=O)O", "FC(F)(F)c1ccccc1N", "c1ccsc1C(=O)OC", "OCc1ccc(Br)cc1"]


def _table(fields=("electrostatic", "steric", "hydrophobic", "hba", "hbd"), alpha=0.5):
    fs = FieldSet.create(physchem=list(fields), alpha=alpha)
    mols = []
    for k, smi in enumerate(SMILES):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = k + 1
        AllChem.EmbedMolecule(m, p)
        mols.append(m)
    return fs, AtomTable.from_molecules([build_fields(m, fs, f"m{i}")
                                         for i, m in enumerate(mols)])


def _sets(fs, alphas):
    return [[FieldSpec(**{**vars(s), "alpha": a}) for s in fs.specs] for a in alphas]


# ----------------------------------------------------------------- the symmetric training kernel


def test_the_symmetric_kernel_equals_the_full_one():
    fs, T = _table()
    sym = kernel_matrices(T, T, fs.specs, symmetric=True)
    full = kernel_matrices(T, T, fs.specs, symmetric=False)
    assert np.allclose(sym, full, rtol=0, atol=1e-10)


def test_a_table_against_itself_takes_the_symmetric_path_by_default():
    fs, T = _table()
    auto = kernel_matrices(T, T, fs.specs)
    assert np.allclose(auto, kernel_matrices(T, T, fs.specs, symmetric=True), rtol=0, atol=0)


def test_the_result_is_exactly_symmetric_which_the_full_path_was_not():
    """Summing the atom pairs in two orders left the two triangles differing by round-off."""
    fs, T = _table()
    sym = kernel_matrices(T, T, fs.specs, symmetric=True)
    assert np.array_equal(sym, np.transpose(sym, (0, 2, 1)))
    full = kernel_matrices(T, T, fs.specs, symmetric=False)
    assert not np.array_equal(full, np.transpose(full, (0, 2, 1)))


def test_two_different_tables_are_not_treated_as_symmetric():
    fs, T = _table()
    A, B = T.subset([0, 1, 2]), T.subset([3, 4, 5])
    K = kernel_matrices(A, B, fs.specs)
    assert K.shape == (len(fs.specs), 3, 3)
    assert np.allclose(K, kernel_matrices(A, B, fs.specs, symmetric=False))
    with pytest.raises(ValueError, match="one table against itself"):
        kernel_matrices(A, T.subset([0, 1]), fs.specs, symmetric=True)


def test_a_block_restricted_to_later_molecules_is_a_slice_of_the_whole_block():
    fs, T = _table()
    sl = T.slice(2)
    whole = atom_kernel_block(T.coords[sl], T.weights[sl], T.radii[sl], T, fs.specs)
    part = atom_kernel_block(T.coords[sl], T.weights[sl], T.radii[sl], T, fs.specs, first_mol=2)
    assert np.allclose(part, whole[:, :, 2:], rtol=0, atol=1e-12)


# ------------------------------------------------------------------ the square-root prefactor


def test_the_square_root_form_is_the_three_halves_power():
    z = np.logspace(-6, 6, 2000)
    assert np.allclose(_pow32(z), z ** 1.5, rtol=1e-14, atol=0)


def test_the_steric_kernel_is_unchanged_by_it(monkeypatch):
    fs, T = _table(fields=("steric",))
    fast = kernel_matrices(T, T, fs.specs)
    import cmfqsar.kernels as kn
    monkeypatch.setattr(kn, "_pow32", lambda z: z ** 1.5)
    slow = kernel_matrices(T, T, fs.specs)
    assert np.allclose(fast, slow, rtol=1e-12, atol=0)


# --------------------------------------------------------------------- the shared distances


def test_distance_blocks_are_what_the_kernel_would_have_computed():
    fs, T = _table()
    sl = T.slice(1)
    D2 = pair_distances(T.coords[sl], T, 0)
    assert D2.shape == (len(T.coords[sl]), T.coords.shape[0])
    direct = np.sum((T.coords[sl][:, None, :] - T.coords[None, :, :]) ** 2, axis=2)
    assert np.allclose(D2, direct, rtol=0, atol=1e-9)
    later = pair_distances(T.coords[sl], T, 3)
    assert np.allclose(later, D2[:, int(T.offsets[3]):], rtol=0, atol=0)


def test_a_grid_of_attenuations_gives_the_same_kernels_one_at_a_time():
    fs, T = _table()
    alphas = (0.1, 0.5, 2.0)
    sets = _sets(fs, alphas)
    shared = kernel_matrices_multi(T, T, sets)
    assert len(shared) == len(alphas)
    for K, specs in zip(shared, sets):
        assert np.allclose(K, kernel_matrices(T, T, specs), rtol=0, atol=1e-10)


def test_the_shared_path_is_symmetric_for_a_table_against_itself():
    fs, T = _table()
    sets = _sets(fs, (0.3, 1.0))
    for K in kernel_matrices_multi(T, T, sets):
        assert np.array_equal(K, np.transpose(K, (0, 2, 1)))
    A, B = T.subset([0, 1, 2]), T.subset([4, 5])
    for K in kernel_matrices_multi(A, B, sets):
        assert K.shape[1:] == (3, 2)
    with pytest.raises(ValueError, match="one table against itself"):
        kernel_matrices_multi(A, B, sets, symmetric=True)


def test_shared_self_kernels_match_the_single_ones():
    fs, T = _table()
    sets = _sets(fs, (0.2, 0.5, 1.5))
    shared = self_kernels_multi(T, sets)
    for d, specs in zip(shared, sets):
        assert np.allclose(d, self_kernels(T, specs), rtol=0, atol=1e-12)


def test_an_atom_block_takes_a_precomputed_distance_matrix():
    fs, T = _table()
    sl = T.slice(0)
    D2 = pair_distances(T.coords[sl], T, 0)
    with_d2 = atom_kernel_block(T.coords[sl], T.weights[sl], T.radii[sl], T, fs.specs, D2=D2)
    without = atom_kernel_block(T.coords[sl], T.weights[sl], T.radii[sl], T, fs.specs)
    assert np.allclose(with_d2, without, rtol=0, atol=0)


# ------------------------------------------------------- the grid search gets the same answer


def test_the_grid_search_selects_what_it_selected_before():
    """The shortcut is in grid_search's path, so the selection itself is the thing to pin."""
    from cmfqsar.model import CMFModel, grid_search
    fs, T = _table()
    rng = np.random.default_rng(0)
    K = kernel_matrices(T, T, fs.specs).sum(axis=0)
    y = K @ rng.normal(size=T.n_mol) * 0.1 + rng.normal(scale=0.05, size=T.n_mol)
    alphas = (0.3, 0.5, 1.0)

    def search(model):
        return grid_search(T, y, model, alphas=alphas, lams=(0.1,), folds=4, seed=0)

    fast = search(CMFModel(fieldset=fs.copy(), method="krr", normalize=True))
    plain = CMFModel(fieldset=fs.copy(), method="krr", normalize=True)
    plain.prepare_training_kernels_grid = lambda *a, **k: None       # the per-alpha path
    slow = search(plain)
    assert fast.best_alpha == slow.best_alpha
    assert fast.best_q2 == pytest.approx(slow.best_q2, abs=1e-12)


def test_the_grid_shortcut_declines_what_it_cannot_do():
    from cmfqsar.model import CMFModel
    fs, T = _table()
    m = CMFModel(fieldset=fs.copy(), method="krr", normalize=True)
    assert m.prepare_training_kernels_grid(T, [0.5]) is None          # one alpha: nothing to share
    assert m.prepare_training_kernels_grid(T, [None, None]) is None   # the table's own alphas
    got = m.prepare_training_kernels_grid(T, [0.3, 1.0])
    assert sorted(got) == [0.3, 1.0]
    for alpha, (Kf, diag) in got.items():
        assert Kf.shape == (len(fs.specs), T.n_mol, T.n_mol)
        assert diag is not None and diag.shape == (len(fs.specs), T.n_mol)
    # the field set is left as it was, not at the last alpha of the grid
    assert all(sp.alpha == 0.5 for sp in m.fieldset.specs)


# ------------------------------------------------------------------ the cutoff (plan item B2)


def test_the_cutoff_radius_is_where_the_gaussian_reaches_the_tolerance():
    from cmfqsar.kernels import CUTOFF_TOL, cutoff_radius
    for coeff in (0.025, 0.125, 0.5, 2.0):
        r = cutoff_radius(coeff)
        assert math.exp(-coeff * r * r) == pytest.approx(CUTOFF_TOL, rel=1e-9)
        assert math.exp(-coeff * (0.9 * r) ** 2) > CUTOFF_TOL
    # a looser tolerance gives a shorter radius
    assert cutoff_radius(0.125, 1e-3) < cutoff_radius(0.125, 1e-9)


def test_a_forced_cutoff_agrees_with_the_dense_sum_to_its_tolerance():
    fs, T = _table(alpha=4.0)
    exact = kernel_matrices(T, T, fs.specs)
    cut = kernel_matrices(T, T, fs.specs, cutoff=True)
    rel = float(np.max(np.abs(cut - exact)) / np.max(np.abs(exact)))
    assert rel < 1e-5                      # the dropped pairs were below 1e-6 of a peak overlap
    assert not np.array_equal(cut, exact)  # and it really is the approximate path


def test_auto_declines_on_a_small_table_and_is_then_exact():
    fs, T = _table()
    assert np.array_equal(kernel_matrices(T, T, fs.specs, cutoff="auto"),
                          kernel_matrices(T, T, fs.specs))


def test_a_radius_given_as_a_number_is_used_as_the_radius():
    fs, T = _table(fields=("steric",))
    wide = kernel_matrices(T, T, fs.specs, cutoff=1e4)
    assert np.allclose(wide, kernel_matrices(T, T, fs.specs), rtol=1e-12, atol=0)
    narrow = kernel_matrices(T, T, fs.specs, cutoff=0.5)
    off_diagonal = narrow[0] - np.diag(np.diag(narrow[0]))
    assert np.max(np.abs(off_diagonal)) < np.max(np.abs(narrow[0]))


def test_pairs_at_zero_distance_are_kept():
    """An atom's overlap with itself is the largest term there is; a self-kernel is made of them."""
    from cmfqsar.kernels import _neighbour_pairs
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    row, col, d2 = _neighbour_pairs(x, x, 1.0)
    assert sorted(zip(row.tolist(), col.tolist())) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    assert np.allclose(d2, 0.0)


def test_the_surviving_share_is_a_share():
    from cmfqsar.kernels import surviving_share
    fs, T = _table()
    assert surviving_share(T.coords[T.slice(0)], T.coords, 1e6) == pytest.approx(1.0)
    assert surviving_share(T.coords[T.slice(0)], T.coords, 1e-9) < 0.05
    assert surviving_share(np.zeros((0, 3)), T.coords, 5.0) == 1.0


def test_a_cutoff_is_refused_together_with_a_precomputed_distance_block():
    """A dense block is dense; asking for a cutoff as well would silently ignore one of them."""
    fs, T = _table()
    sl = T.slice(0)
    D2 = pair_distances(T.coords[sl], T, 0)
    with_both = atom_kernel_block(T.coords[sl], T.weights[sl], T.radii[sl], T, fs.specs,
                                  D2=D2, cutoff=True)
    assert np.array_equal(with_both,
                          atom_kernel_block(T.coords[sl], T.weights[sl], T.radii[sl], T,
                                            fs.specs, D2=D2))
