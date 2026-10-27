"""The variance of a field across an ensemble (plan item D10).

The claim the whole item rests on is that the variance of a field across conformations is itself
a field of the same class -- a sum of Gaussians, because a product of two Gaussians is a Gaussian
-- so that no new kernel is needed. What these tests fix is that the field built here really is
that variance, that it is used with the right widths, and that what is dropped for cost is
dropped knowingly.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.conformers import generate_conformers, single_conformer  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.fluctuation import (PAIR_FLOOR, evaluate, kept_share, variance_field,  # noqa: E402
                                 variance_fieldset)

FIELDS = ["electrostatic", "hydrophobic"]


def _members(smiles="CCCCc1ccccc1", n=4, seed=1, alpha=1.0):
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha)
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    m.SetProp("_Name", "m")
    ens = generate_conformers(m, n=n, seed=seed)
    k = ens.GetNumConformers()
    return fs, [build_fields(single_conformer(ens, i), fs, name=f"c{i}") for i in range(k)]


def _points(members, n=40, seed=0):
    rng = np.random.RandomState(seed)
    base = np.asarray(members[0].coords, float)
    return base[rng.randint(0, len(base), n)] + rng.randn(n, 3) * 0.7


# ----------------------------------------------------------------- it is the variance


def test_the_field_is_the_variance_it_claims_to_be():
    """The whole item rests on this: evaluated anywhere, it is the variance across the frames."""
    fs, members = _members()
    assert len(members) > 1
    V = variance_field(members, fs, floor=1e-8)
    vfs = variance_fieldset(fs)
    pts = _points(members)
    for f in range(len(FIELDS)):
        per_frame = np.array([evaluate(m, fs, f, pts) for m in members])
        want = (per_frame ** 2).mean(axis=0) - per_frame.mean(axis=0) ** 2
        got = evaluate(V, vfs, f, pts)
        # measured against the size of the field, not point by point: the variance is near zero
        # in most of the space and a relative tolerance there would be a tolerance on nothing
        assert np.max(np.abs(got - want)) < 1e-3 * np.max(np.abs(want))
        assert np.abs(want).max() > 0


def test_weights_that_are_not_uniform_are_the_weighted_variance():
    fs, members = _members()
    k = len(members)
    w = np.linspace(1.0, 2.0, k)
    V = variance_field(members, fs, weights=w, floor=1e-8)
    vfs = variance_fieldset(fs)
    pts = _points(members)
    p = w / w.sum()
    per_frame = np.array([evaluate(m, fs, 0, pts) for m in members])
    want = (p[:, None] * per_frame ** 2).sum(axis=0) - ((p[:, None] * per_frame).sum(axis=0)) ** 2
    got = evaluate(V, vfs, 0, pts)
    assert np.max(np.abs(got - want)) < 1e-3 * np.max(np.abs(want))


def test_identical_conformations_have_no_variance():
    fs, members = _members()
    same = [members[0], members[0], members[0]]
    V = variance_field(same, fs, floor=1e-10)
    vfs = variance_fieldset(fs)
    pts = _points(members)
    assert evaluate(V, vfs, 0, pts) == pytest.approx(np.zeros(len(pts)), abs=1e-9)


# -------------------------------------------------------------------- the widths it needs


def test_the_variance_field_is_twice_as_wide():
    fs, _members_ = _members(alpha=0.7)
    vfs = variance_fieldset(fs)
    assert [s.alpha for s in vfs.specs] == pytest.approx([1.4] * len(fs.specs))
    assert [s.name for s in vfs.specs] == [s.name for s in fs.specs]
    assert [s.alpha for s in fs.specs] == pytest.approx([0.7] * len(fs.specs))   # not in place


def test_a_field_with_a_width_per_atom_is_refused():
    fs = FieldSet.create(physchem=["electrostatic", "steric"])
    with pytest.raises(ValueError, match="different Gaussian on every atom"):
        variance_fieldset(fs)
    with pytest.raises(ValueError, match="one width per field"):
        variance_field(_members(alpha=1.0)[1], FieldSet.create(physchem=["steric"]))


def test_fields_at_different_attenuations_are_refused():
    fs, members = _members()
    fs.specs[0].alpha = 3.0
    with pytest.raises(ValueError, match="one attenuation"):
        variance_field(members, fs)


# ------------------------------------------------------------- what the floor drops, knowingly


def test_a_higher_floor_drops_pairs_and_says_so():
    fs, members = _members()
    small = variance_field(members, fs, floor=1e-6)
    large = variance_field(members, fs, floor=PAIR_FLOOR)
    assert large.n_atoms < small.n_atoms
    assert kept_share(members, fs, 1e-6) > kept_share(members, fs, PAIR_FLOOR)
    assert 0.0 < kept_share(members, fs, PAIR_FLOOR) <= 1.0
    # and a floor high enough to cut the field really does cut it
    assert kept_share(members, fs, 2e-1) < kept_share(members, fs, PAIR_FLOOR)
    pts = _points(members)
    vfs = variance_fieldset(fs)
    exact = evaluate(small, vfs, 0, pts)
    coarse = evaluate(large, vfs, 0, pts)
    assert np.max(np.abs(coarse - exact)) > 0                 # it really is an approximation
    assert np.corrcoef(coarse, exact)[0, 1] > 0.9             # and still the same picture


def test_everything_below_the_floor_is_refused_rather_than_returned_empty():
    fs, members = _members()
    with pytest.raises(ValueError, match="no variance field"):
        variance_field(members, fs, floor=2.0)


def test_one_conformation_has_no_variance_to_take():
    fs, members = _members()
    with pytest.raises(ValueError, match="at least two conformations"):
        variance_field(members[:1], fs)


def test_conformations_of_different_molecules_are_refused():
    fs, members = _members()
    small = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    p = AllChem.ETKDGv3()
    p.randomSeed = 1
    AllChem.EmbedMolecule(small, p)
    other = build_fields(small, fs, name="other")
    with pytest.raises(ValueError, match="same atoms"):
        variance_field([members[0], other], fs)


# ------------------------------------------------------------------ it is a field like any other


def test_the_result_is_a_field_the_rest_of_the_program_can_take():
    from cmfqsar.fields import AtomTable
    from cmfqsar.kernels import kernel_matrices

    fs, members = _members()
    V = variance_field(members, fs, floor=PAIR_FLOOR)
    vfs = variance_fieldset(fs)
    assert V.n_atoms > 0 and V.weights.shape[1] == len(FIELDS)
    assert len(V.symbols) == len(V.atom_types) == V.n_atoms
    table = AtomTable.from_molecules([V, V])
    K = kernel_matrices(table, table, [s for s in vfs.specs if s.enabled], symmetric=True)
    assert K.shape == (len(FIELDS), 2, 2)
    assert np.isfinite(K).all()
    for f in range(len(FIELDS)):
        assert K[f][0, 0] == pytest.approx(K[f][1, 1])
        assert K[f][0, 1] == pytest.approx(K[f][0, 0])        # the same field twice
