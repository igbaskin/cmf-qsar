"""The three-distance autocorrelation of an environment (plan item C5).

What these tests fix: that the descriptor has the shape its grid says, that it is invariant to
where the environment sits and how it is turned, that it sees what the two-point descriptor
cannot (the angle at the centre, through a pair of arrangements with the same distance histogram),
that it does not see handedness, that the two routes to the kernel agree, and that the command
reaches it.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.fields import FieldSet, MoleculeFields, build_fields  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, autocorrelation  # noqa: E402
from cmfqsar.localenv import environment, environments  # noqa: E402
from cmfqsar.triangles import (TriangleSpec, element_weights, envelope_factors,  # noqa: E402
                               triangle_feature_matrix, triangle_features, triangle_kernel,
                               triangle_kernel_matrix, triangle_labels, with_weights)

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "CCOc1ccccc1N", "CC(C)c1ccccc1O", "CCCCc1ccccc1",
          "CC(=O)Nc1ccc(O)cc1", "COc1ccc(C=O)cc1"]
SPEC = TriangleSpec(r_max=3.0, step=0.5, sigma=0.5)


def _mol(smiles, name="m", seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    m.SetProp("_Name", name)
    return m


@pytest.fixture(scope="module")
def built():
    fieldset = FieldSet.create()
    mols, fields = [], []
    for i, s in enumerate(SMILES):
        m = _mol(s, f"m{i}", seed=i + 1)
        mols.append(m)
        fields.append(build_fields(m, fieldset, name=f"m{i}"))
    for m in mols:
        for a in m.GetAtoms():
            if a.GetSymbol() == "C":
                a.SetDoubleProp("nh", float(sum(1 for nb in a.GetNeighbors()
                                                if nb.GetSymbol() == "H")))
    targets = AtomTargets.from_molecules(mols, "nh", names=[f"m{i}" for i in range(len(mols))])
    return mols, fields, targets, fieldset


def _line(points, scale=0.25):
    """An environment of unit weights on one field, atoms on a line, the first at the centre."""
    x = np.asarray(points, float) * scale
    n = len(x)
    coords = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    return MoleculeFields(name="line", coords=coords, weights=np.ones((n, 1)),
                          radii=np.full(n, 1.7), symbols=["C"] * n, atom_types=["C.3"] * n,
                          bonds=np.zeros((0, 2), int), charges=np.zeros(n))


# ------------------------------------------------------------------------------------ the grids


def test_the_grids_and_the_channel_count_are_what_the_spec_says():
    s = TriangleSpec(r_max=3.0, step=0.5, sigma=0.5)
    assert s.radii.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert len(s.distances) == 13 and s.distances[-1] == pytest.approx(6.0)
    assert len(s.radial_pairs(True)) == 7 * 8 // 2
    assert len(s.radial_pairs(False)) == 49
    # two fields, in the order of field_pairs: (0, 0), (0, 1), (1, 1)
    assert s.block_sizes(2) == [28 * 13, 49 * 13, 28 * 13] or \
        s.block_sizes(2) == [28 * 13, 28 * 13, 49 * 13]
    assert sorted(s.block_sizes(2)) == [28 * 13, 28 * 13, 49 * 13]
    assert s.n_channels(2) == sum(s.block_sizes(2))


def test_the_features_have_the_shape_of_the_spec(built):
    _mols, fields, targets, fieldset = built
    envs = environments(fields, targets, rho=0.25)[:10]
    X = triangle_feature_matrix(envs, SPEC)
    n_fields = len(fieldset.specs)
    assert X.shape == (10, SPEC.n_channels(n_fields))
    assert np.isfinite(X).all()
    assert len(triangle_labels(fieldset, SPEC)) == X.shape[1]
    assert "electrostatic at 0.00 a x electrostatic at 0.00 a, 0.00 a apart" in \
        triangle_labels(fieldset, SPEC)[0].lower()


# -------------------------------------------------------------------------------- invariance


def test_moving_and_turning_the_molecule_changes_nothing(built):
    _mols, fields, _t, _fs = built
    m = fields[0]
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned = type(m)(name=m.name, coords=m.coords @ rot.T + 7.0, weights=m.weights,
                     radii=m.radii, symbols=m.symbols, atom_types=m.atom_types, bonds=m.bonds,
                     charges=m.charges)
    a = triangle_features(environment(m, 2, 0.25), SPEC)
    b = triangle_features(environment(turned, 2, 0.25), SPEC)
    assert a == pytest.approx(b, abs=1e-9)


def test_a_mirror_image_is_the_same_environment(built):
    """Three distances fix a triangle up to its reflection: the known blind spot stays."""
    _mols, fields, _t, _fs = built
    m = fields[1]
    mirrored = type(m)(name=m.name, coords=m.coords * np.array([1.0, 1.0, -1.0]),
                       weights=m.weights, radii=m.radii, symbols=m.symbols,
                       atom_types=m.atom_types, bonds=m.bonds, charges=m.charges)
    a = triangle_features(environment(m, 3, 0.25), SPEC)
    b = triangle_features(environment(mirrored, 3, 0.25), SPEC)
    assert a == pytest.approx(b, abs=1e-9)


def test_an_atom_beyond_the_cutoff_is_not_in_the_sum():
    base = _line([0, 1, 4, 10], scale=0.25)
    far = _line([0, 1, 4, 10, 60], scale=0.25)       # 15 A away, beyond r_max + reach * sigma
    assert triangle_features(base, SPEC) == pytest.approx(triangle_features(far, SPEC))


# -------------------------------------------------- what the two-point descriptor cannot see


def test_two_arrangements_with_one_distance_histogram_are_told_apart():
    """{0, 1, 4, 10, 12, 17} and {0, 1, 8, 11, 13, 17} have the same multiset of differences, so
    the two-point autocorrelation cannot tell them apart; their radii from the centre differ, so
    the three-distance one can."""
    a = _line([0, 1, 4, 10, 12, 17])
    b = _line([0, 1, 8, 11, 13, 17])
    two = AutocorrelationSpec(d_max=5.0, step=0.5, sigma=0.5)
    assert autocorrelation(a, two) == pytest.approx(autocorrelation(b, two), abs=1e-9)
    spec = TriangleSpec(r_max=4.5, step=0.5, sigma=0.5)
    fa, fb = triangle_features(a, spec), triangle_features(b, spec)
    assert np.linalg.norm(fa - fb) > 0.05 * np.linalg.norm(fa)


def test_the_angle_at_the_centre_moves_the_channels():
    """Two neighbours at fixed radii: turning one about the centre changes the descriptor, and
    more so the further it turns."""
    def env(theta):
        c = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                      [2.0 * np.cos(theta), 2.0 * np.sin(theta), 0.0]])
        return MoleculeFields(name="v", coords=c, weights=np.ones((3, 1)), radii=np.full(3, 1.7),
                              symbols=["C"] * 3, atom_types=["C.3"] * 3,
                              bonds=np.zeros((0, 2), int), charges=np.zeros(3))
    f0 = triangle_features(env(0.5), SPEC)
    gaps = [np.linalg.norm(triangle_features(env(0.5 + t), SPEC) - f0) for t in (0.3, 0.6, 1.2)]
    assert gaps[0] > 1e-3 and gaps[0] < gaps[1] < gaps[2]


# ---------------------------------------------------------------------------------- the kernel


def test_the_kernel_is_a_cosine_and_the_power_is_its_power(built):
    _mols, fields, targets, fieldset = built
    envs = environments(fields, targets, rho=0.25)[:12]
    n_fields = len(fieldset.specs)
    X = triangle_feature_matrix(envs, SPEC)
    K = triangle_kernel(X, spec=SPEC, n_fields=n_fields)
    assert K.shape == (12, 12)
    assert np.diag(K) == pytest.approx(np.ones(12))
    assert K == pytest.approx(K.T)
    assert K.max() <= 1.0 + 1e-9
    K2 = triangle_kernel(X, spec=SPEC, n_fields=n_fields, power=2)
    assert K2 == pytest.approx(K ** 2)
    with pytest.raises(ValueError):
        triangle_kernel(X, spec=SPEC, n_fields=n_fields, power=0)


def test_the_block_scalings_mean_what_they_say(built):
    from cmfqsar.triangles import block_scales, triangle_scale
    _mols, fields, targets, fieldset = built
    envs = environments(fields, targets, rho=0.25)[:12]
    nf = len(fieldset.specs)
    X = triangle_feature_matrix(envs, SPEC)
    blocks = SPEC.block_sizes(nf)
    starts = np.cumsum([0] + blocks)
    pair = triangle_scale(X, SPEC, nf, "pair")
    field = triangle_scale(X, SPEC, nf, "field")
    none = triangle_scale(X, SPEC, nf, "none")
    assert none == pytest.approx(np.ones(X.shape[1]))
    from cmfqsar.invariant import channel_scale, field_pairs
    assert pair == pytest.approx(channel_scale(X, 1, blocks=blocks))
    own = {}
    for (f, g), s, e in zip(field_pairs(nf), starts[:-1], starts[1:]):
        if f == g:
            own[f] = pair[s]
            assert field[s] == pytest.approx(pair[s])      # a field with itself: the same
    for (f, g), s, e in zip(field_pairs(nf), starts[:-1], starts[1:]):
        if f != g:
            assert field[s] == pytest.approx(np.sqrt(own[f] * own[g]))
    with pytest.raises(ValueError):
        block_scales([1.0] * len(blocks), nf, blocks, "other")
    K = triangle_kernel(X, spec=SPEC, n_fields=nf, scaling="none")
    assert np.diag(K) == pytest.approx(np.ones(12))


def test_the_two_routes_to_the_kernel_agree(built):
    _mols, fields, targets, fieldset = built
    envs = environments(fields, targets, rho=0.25)[:17]
    n_fields = len(fieldset.specs)
    K = triangle_kernel(triangle_feature_matrix(envs, SPEC), spec=SPEC, n_fields=n_fields,
                        power=2)
    K2 = triangle_kernel_matrix(envs, SPEC, power=2, chunk=5)
    assert K2 == pytest.approx(K, abs=1e-5)


def test_every_environment_must_carry_the_same_fields(built):
    _mols, fields, targets, _fs = built
    envs = environments(fields, targets, rho=0.25)[:3]
    odd = with_weights(envs[1], envs[1].weights[:, :2])
    with pytest.raises(ValueError):
        triangle_feature_matrix([envs[0], odd, envs[2]], SPEC)


# --------------------------------------------------------------------- channels that are not fields


def test_element_channels_are_indicators_under_the_envelope(built):
    _mols, fields, targets, _fs = built
    e = environments(fields, targets, rho=0.25)[0]
    W = element_weights(e, ("H", "C", "N", "O"))
    assert W.shape == (e.n_atoms, 5)
    assert W.sum(axis=1) == pytest.approx(np.ones(e.n_atoms))
    assert W[0, 1] == 1.0                              # the centre is a carbon
    Wr = element_weights(e, ("H", "C", "N", "O"), rho=0.25)
    assert Wr.sum(axis=1) == pytest.approx(envelope_factors(e, 0.25))
    assert envelope_factors(e, 0.25)[0] == 1.0
    swapped = with_weights(e, Wr)
    assert swapped.n_atoms == e.n_atoms and swapped.weights.shape[1] == 5
    with pytest.raises(ValueError):
        with_weights(e, Wr[:-1])


# --------------------------------------------------------------------------------- the command


def test_the_command_reaches_the_three_distance_descriptor(tmp_path, capsys):
    from cmfqsar.cli import main
    path = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(path) as w:
        for i, smiles in enumerate(SMILES):
            m = _mol(smiles, f"m{i}", seed=i + 1)
            v = []
            for a in m.GetAtoms():
                if a.GetSymbol() == "C":
                    v.append("%.3f" % sum(1 for nb in a.GetNeighbors()
                                          if nb.GetSymbol() == "H"))
                else:
                    v.append("n/a")
            m.SetProp("atom.dprop.nh", " ".join(v))
            w.write(m)
    capsys.readouterr()
    main(["local", path, "--target", "nh", "--rho", "0.25", "--triangles", "--r-max", "3",
          "--power", "2", "--folds", "3", "--bootstrap", "50"])
    text = capsys.readouterr().out
    assert "three-distance autocorrelation" in text and "three-distance field" in text
    with pytest.raises(SystemExit):
        main(["local", path, "--target", "nh", "--rho", "0.25", "--triangles", "--primal",
              "--power", "2", "--folds", "3"])
