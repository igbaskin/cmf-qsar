"""A field is a kernel mean embedding: the identities of the manual section that says so.

A field is the sum of the Gaussians of its atoms, each multiplied by the atom's weight. In the
language of kernel methods it is the embedding of the weighted atoms (a signed measure) by the
Gaussian of the field, and the overlap kernel is the inner product of two such embeddings, which
is the set kernel. These tests check the identities the section states, on the program's own
functions:

* the overlap kernel is the mean-map kernel, a double sum of a base kernel over pairs of atoms;
* the steric field, with a width per atom, is the same with the width as an extra coordinate;
* the field distance is the length of the difference field, and the MMD between two sets of
  molecules is the length of the difference of their mean fields;
* the witness function of the two-sample test is the difference of the mean fields seen through
  the kernel;
* two atoms at one place are one atom with the summed weight, because only the measure counts;
* cosine normalisation divides out the scale of the weights, which the raw kernel keeps;
* a field whose weights sum to zero, as the charges of a neutral molecule do, still has a length.
"""
import os

import numpy as np
import pytest

from cmfqsar import kernels as KN
from cmfqsar import mmd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bag(rng, n, signed=True):
    x = rng.normal(scale=2.0, size=(n, 3))
    w = rng.normal(size=n) if signed else rng.uniform(0.2, 1.5, size=n)
    return x, w


def mean_map_kernel(x1, w1, x2, w2, alpha):
    """The set kernel with weights: the sum over pairs of points of w w k, with the base kernel
    k(r, r') = (pi / alpha)^(3/2) exp(-alpha |r - r'|^2 / 4) that two Gaussians of width alpha
    give when their product is integrated over space."""
    d2 = ((x1[:, None, :] - x2[None, :, :]) ** 2).sum(-1)
    k = (np.pi / alpha) ** 1.5 * np.exp(-alpha * d2 / 4.0)
    return float(w1 @ k @ w2)


def overlap(x1, w1, x2, w2, alpha):
    return KN.expansion_overlap(x1, w1, np.full(len(x1), alpha), x2, w2, np.full(len(x2), alpha))


def test_the_overlap_kernel_is_the_mean_map_kernel():
    rng = np.random.default_rng(1)
    xa, wa = _bag(rng, 6)
    xb, wb = _bag(rng, 9)
    for alpha in (0.3, 1.0, 3.0):
        assert overlap(xa, wa, xb, wb, alpha) == pytest.approx(
            mean_map_kernel(xa, wa, xb, wb, alpha), rel=1e-12)


def test_the_steric_widths_are_one_more_coordinate():
    # with a width per atom the base kernel takes the widths as arguments too; it is still the
    # integral of the product of two Gaussians, so it is still an inner product
    rng = np.random.default_rng(2)
    xa, wa = _bag(rng, 5, signed=False)
    xb, wb = _bag(rng, 4, signed=False)
    ba, bb = rng.uniform(0.3, 2.0, 5), rng.uniform(0.3, 2.0, 4)
    s = ba[:, None] + bb[None, :]
    d2 = ((xa[:, None, :] - xb[None, :, :]) ** 2).sum(-1)
    k = (2 * np.pi / s) ** 1.5 * np.exp(-(ba[:, None] * bb[None, :]) / (2 * s) * d2)
    assert KN.expansion_overlap(xa, wa, ba, xb, wb, bb) == pytest.approx(float(wa @ k @ wb),
                                                                        rel=1e-12)
    # and the matrix of the base kernel over all atoms is positive semidefinite
    x = np.vstack([xa, xb])
    b = np.concatenate([ba, bb])
    s = b[:, None] + b[None, :]
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    G = (2 * np.pi / s) ** 1.5 * np.exp(-(b[:, None] * b[None, :]) / (2 * s) * d2)
    assert np.linalg.eigvalsh(G).min() > -1e-10 * np.abs(G).max()


def test_the_field_distance_is_the_length_of_the_difference_field():
    rng = np.random.default_rng(3)
    xa, wa = _bag(rng, 6)
    xb, wb = _bag(rng, 8)
    a = 0.7
    d2 = overlap(xa, wa, xa, wa, a) - 2 * overlap(xa, wa, xb, wb, a) + overlap(xb, wb, xb, wb, a)
    x, w = np.vstack([xa, xb]), np.concatenate([wa, -wb])     # field A minus field B
    assert d2 == pytest.approx(overlap(x, w, x, w, a), rel=1e-10)
    assert d2 > 0


def _sets(rng, n_x, n_y):
    return [_bag(rng, int(rng.integers(3, 8))) for _ in range(n_x + n_y)]


def test_the_mmd_of_two_sets_is_the_distance_of_their_mean_fields():
    rng = np.random.default_rng(4)
    mols = _sets(rng, 4, 5)
    a = 1.0
    K = np.array([[overlap(*m, *n, a) for n in mols] for m in mols])
    mask = np.array([True] * 4 + [False] * 5)
    # the mean field of a set: every molecule's atoms with its weights divided by the set size
    x = np.vstack([m[0] for m in mols])
    w = np.concatenate([m[1] / (4 if i < 4 else -5) for i, m in enumerate(mols)])
    assert mmd.mmd2(K, mask, unbiased=False) == pytest.approx(overlap(x, w, x, w, a), rel=1e-10)


def test_the_witness_is_the_difference_of_the_mean_fields():
    rng = np.random.default_rng(5)
    mols = _sets(rng, 3, 4)
    a = 0.5
    K = np.array([[overlap(*m, *n, a) for n in mols] for m in mols])
    mask = np.array([True] * 3 + [False] * 4)
    x = np.vstack([m[0] for m in mols])
    w = np.concatenate([m[1] / (3 if i < 3 else -4) for i, m in enumerate(mols)])
    got = mmd.witness(K, mask)
    # the witness at a molecule is the overlap of its field with the difference field
    want = np.array([overlap(*m, x, w, a) for m in mols])
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)


def test_two_atoms_at_one_place_are_one_atom_with_the_summed_weight():
    rng = np.random.default_rng(6)
    xa, wa = _bag(rng, 5)
    split_x = np.vstack([xa, xa[:1]])
    split_w = np.concatenate([wa, [0.0]])
    split_w[0], split_w[-1] = 0.3 * wa[0], 0.7 * wa[0]
    other = _bag(rng, 7)
    a = 1.3
    assert overlap(split_x, split_w, *other, a) == pytest.approx(overlap(xa, wa, *other, a),
                                                                rel=1e-12)
    x, w = np.vstack([xa, split_x]), np.concatenate([wa, -split_w])
    assert abs(overlap(x, w, x, w, a)) < 1e-10                # the same measure, no distance
    # moving one atom by 0.1 A gives a different measure and a distance that is not zero
    moved = xa.copy()
    moved[0] += 0.1
    x, w = np.vstack([xa, moved]), np.concatenate([wa, -wa])
    assert overlap(x, w, x, w, a) > 1e-6


def test_cosine_normalisation_divides_out_the_scale_of_the_weights():
    rng = np.random.default_rng(7)
    xa, wa = _bag(rng, 6, signed=False)
    xb, wb = _bag(rng, 6, signed=False)
    a = 1.0

    def cosine(w1, w2):
        return overlap(xa, w1, xb, w2, a) / np.sqrt(overlap(xa, w1, xa, w1, a)
                                                     * overlap(xb, w2, xb, w2, a))

    assert overlap(xa, 3 * wa, xb, wb, a) == pytest.approx(3 * overlap(xa, wa, xb, wb, a))
    assert cosine(3 * wa, wb) == pytest.approx(cosine(wa, wb), rel=1e-12)


def test_a_field_whose_weights_sum_to_zero_still_has_a_length():
    rng = np.random.default_rng(8)
    x, w = _bag(rng, 7)
    w = w - w.mean()                                          # a neutral set of charges
    assert abs(w.sum()) < 1e-12
    assert overlap(x, w, x, w, 1.0) > 0


def test_the_program_kernel_of_two_molecules_is_the_mean_map_kernel():
    rdkit = pytest.importorskip("rdkit.Chem")
    from cmfqsar.fields import AtomTable, FieldSet, build_fields

    supplier = rdkit.SDMolSupplier(os.path.join(ROOT, "examples", "benzamides_synthetic.sdf"),
                                   removeHs=False)
    mols = [m for m in supplier if m is not None][:2]
    fs = FieldSet.create(alpha=0.6)
    table = AtomTable.from_molecules([build_fields(m, fs, str(i)) for i, m in enumerate(mols)])
    specs = [s for s in fs.specs if s.enabled]
    K = KN.kernel_matrices(table, table, specs)
    f = next(i for i, s in enumerate(specs) if s.name == "electrostatic")
    a, b = table.slice(0), table.slice(1)
    xa, wa = table.coords[a], table.weights[a, f]
    xb, wb = table.coords[b], table.weights[b, f]
    assert K[f, 0, 1] == pytest.approx(mean_map_kernel(xa, wa, xb, wb, specs[f].alpha), rel=1e-9)
