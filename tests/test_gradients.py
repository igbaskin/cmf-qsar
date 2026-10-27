"""The analytic gradient of a prediction: is it the derivative it claims to be?

Every test here compares the analytic gradient with central finite differences of
``CMFModel.predict`` itself, so what is checked is the derivative of the prediction the program
makes, not of a formula written twice.  The cases are the ones that change the chain rule:
normalisation, the steric field with a width per atom, probe fields, an attenuation between the
points of the interpolation grid, the three nonlinear layers, a descriptor block, a
conformational ensemble and a classifier.
"""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar.cofields import cofield_at_points
from cmfqsar.fields import AtomTable, FieldSet, build_fields
from cmfqsar.gradients import (PredictionGradient, cofield_gradient,
                               numerical_prediction_gradient, prediction_gradient,
                               smoothed_cofield)
from cmfqsar.model import CMFModel

SMILES = ["CCCCO", "CCCCN", "CCCOC", "CCCCC", "CCOCC", "CCCCF", "CCCCCl", "CCC(C)O"]
#: the step that puts the finite difference at its best: the disagreement falls as its square
#: down to about 1e-9 relative and rises again below it, from round-off in the predictions
STEP = 1e-4
#: the relative agreement the finite difference itself can reach at that step
TOL = 1e-6


def _mol(smiles, seed=1, conformers=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if conformers:
        AllChem.EmbedMultipleConfs(m, numConfs=conformers, randomSeed=seed)
        AllChem.MMFFOptimizeMoleculeConfs(m)
    else:
        AllChem.EmbedMolecule(m, randomSeed=seed)
        AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)              # MMFF set-up alters the aromaticity flags in place
    m.SetProp("_Name", smiles)
    return m


def _fitted(fields=("steric", "electrostatic"), alpha=0.5, n=8, conformers=0,
            fieldset_args=None, y=None, **model_args):
    """A small fitted model, its field set and a test molecule that is not in it."""
    mols = [_mol(s, seed=i + 1, conformers=conformers) for i, s in enumerate(SMILES[:n])]
    fs = FieldSet.create(physchem=list(fields), alpha=alpha, **(fieldset_args or {}))
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}")
                                      for i, m in enumerate(mols)])
    if y is None:
        y = np.array([4.0 + 0.2 * m.GetNumHeavyAtoms() + 0.05 * i
                      for i, m in enumerate(mols)])
    model = CMFModel(fieldset=fs, **model_args)
    if fs.descriptors:
        model.prepare_descriptors(table)
    model.fit(table, y, cv_folds=0)
    probe = build_fields(_mol("CCCCS", seed=11, conformers=conformers), fs, "probe")
    return model, fs, probe


def _agree(model, probe, what="both", step=STEP):
    """(relative disagreement in the coordinates, the same in the weights)."""
    g = prediction_gradient(model, probe, wrt=what)
    num = numerical_prediction_gradient(model, probe, h=step, wrt=what)
    out = []
    for got, ref in ((g.coords, num.coords), (g.weights, num.weights)):
        if got is None or ref is None:
            out.append(0.0)
            continue
        out.append(float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12)))
    return tuple(out)


# --------------------------------------------------------------------------- #
# The derivative, against finite differences
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("normalize", [False, True])
def test_the_gradient_matches_finite_differences(normalize):
    """The plain case, which is most models: two fields, one of them radius-scaled."""
    model, _fs, probe = _fitted(method="krr", normalize=normalize, params={"lam": 0.1})
    ec, ew = _agree(model, probe)
    assert ec < TOL, f"coordinate gradient off by {ec}"
    assert ew < TOL, f"weight gradient off by {ew}"


@pytest.mark.parametrize("layer", ["gauss", "arccos1", "arccos2"])
@pytest.mark.parametrize("normalize", [False, True])
def test_the_nonlinear_layers_are_differentiated_too(layer, normalize):
    """A layer depends on the molecule's own self-kernel as well as on the kernel row, and
    without normalisation both paths contribute."""
    model, _fs, probe = _fitted(method="krr", normalize=normalize, kernel_layer=layer,
                                params={"lam": 0.1, "gamma": 1.0})
    ec, ew = _agree(model, probe)
    assert ec < TOL and ew < TOL, f"{layer}, normalize={normalize}: {ec}, {ew}"


@pytest.mark.parametrize("normalize", [False, True])
def test_probe_fields(normalize):
    """A probe field puts several Gaussians on every atom, so its gradient runs over pairs of
    terms as well as pairs of atoms."""
    model, _fs, probe = _fitted(fields=("steric",), method="krr", normalize=normalize,
                                params={"lam": 0.1},
                                fieldset_args={"probes": ["water", "methyl"]})
    ec, ew = _agree(model, probe)
    assert ec < TOL, f"coordinate gradient off by {ec}"
    assert ew < TOL, f"weight gradient off by {ew}"


def test_the_weight_column_of_a_probe_field_is_zero_and_said_to_be():
    """A probe field has no single weight per atom, and the result says which fields those are
    instead of returning a number that means nothing."""
    model, fs, probe = _fitted(fields=("steric",), method="krr", normalize=True,
                               params={"lam": 0.1},
                               fieldset_args={"probes": ["water"]})
    g = prediction_gradient(model, probe)
    assert g.weightless_fields == ["probe:water"]
    f = fs.names.index("probe:water")
    assert np.all(g.weights[:, f] == 0.0)
    assert np.any(g.weights[:, fs.names.index("steric")] != 0.0)


@pytest.mark.parametrize("normalize", [False, True])
def test_an_attenuation_between_the_grid_points(normalize):
    """With `alpha_grid` the kernel is a linear combination of two kernels, and so is the
    gradient; the self-kernel of the nonlinear layer is the one place the two differ."""
    model, _fs, probe = _fitted(alpha=0.37, method="krr", normalize=normalize,
                                kernel_layer="gauss", params={"lam": 0.1, "gamma": 1.0})
    model.alpha_grid = [0.25, 0.5, 1.0]
    model.fit(model.train, model.y, cv_folds=0)
    ec, ew = _agree(model, probe)
    assert ec < TOL and ew < TOL, f"{ec}, {ew}"


def test_a_descriptor_block_shifts_the_prediction_and_not_the_gradient():
    """Non-field descriptors are computed from the graph, so they add a constant to the
    prediction and nothing to either derivative."""
    model, _fs, probe = _fitted(method="krr", normalize=True, params={"lam": 0.1},
                                fieldset_args={"descriptors": ["calc:mw", "calc:clogp"]})
    assert model.desc_train_ is not None
    ec, ew = _agree(model, probe)
    assert ec < TOL and ew < TOL, f"{ec}, {ew}"


def test_a_conformational_ensemble():
    """The rows are the atoms of every member of the ensemble, and each member moves on its own."""
    model, _fs, probe = _fitted(conformers=3, method="krr", normalize=True, params={"lam": 0.1},
                                fieldset_args={"conformers": "ensemble", "max_conformers": 3})
    assert probe.is_ensemble and probe.n_conformers == 3
    g = prediction_gradient(model, probe, wrt="coords")
    assert g.coords.shape == (probe.n_atoms, 3)
    num = numerical_prediction_gradient(model, probe, h=STEP, wrt="coords")
    ec = np.abs(g.coords - num.coords).max() / np.abs(num.coords).max()
    assert ec < TOL, f"coordinate gradient off by {ec}"


def test_a_classifier_differentiates_its_decision_value():
    """For a classifier the derivative is of the decision value, which is the quantity that is a
    sum over the training molecules; the probability is a monotone function of it."""
    classes = np.array([1.0, -1.0] * 4)
    model, _fs, probe = _fitted(task="classification", method="svc", normalize=True,
                                params={"C": 10.0}, y=classes)
    ec, ew = _agree(model, probe)
    assert ec < TOL and ew < TOL, f"{ec}, {ew}"


def test_the_value_is_the_prediction():
    """The gradient is taken of the number the program predicts, and reports it."""
    model, _fs, probe = _fitted(method="gp", normalize=True, params={"lam": 0.1})
    g = prediction_gradient(model, probe, wrt="coords")
    direct = float(model.predict(AtomTable.from_molecules([probe]))[0])
    assert g.value == pytest.approx(direct, abs=1e-12)


def test_a_step_along_the_gradient_raises_the_prediction():
    """What the gradient is for: it points the way up, and a small step goes there."""
    model, _fs, probe = _fitted(method="krr", normalize=True, params={"lam": 0.1})
    g = prediction_gradient(model, probe, wrt="coords")
    size = np.linalg.norm(g.coords)
    assert size > 0
    moved = probe.coords + 0.02 * g.coords / size
    after = prediction_gradient(model, probe, coords=moved, wrt="coords")
    assert after.value > g.value


def test_asking_for_one_derivative_leaves_the_other_out():
    model, _fs, probe = _fitted(method="krr", normalize=True, params={"lam": 0.1})
    only_coords = prediction_gradient(model, probe, wrt="coords")
    only_weights = prediction_gradient(model, probe, wrt="weights")
    both = prediction_gradient(model, probe, wrt="both")
    assert only_coords.weights is None and only_weights.coords is None
    np.testing.assert_allclose(only_coords.coords, both.coords, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(only_weights.weights, both.weights, rtol=1e-12, atol=1e-14)


def test_the_molecule_is_not_altered_by_differentiating_somewhere_else():
    """A design loop evaluates the gradient away from the molecule's own geometry; the molecule
    it was given has to come back unchanged."""
    model, _fs, probe = _fitted(method="krr", normalize=True, params={"lam": 0.1})
    before = probe.coords.copy()
    elsewhere = probe.coords + 0.3
    g = prediction_gradient(model, probe, coords=elsewhere)
    np.testing.assert_allclose(probe.coords, before, atol=0)
    direct = float(model.predict(AtomTable.from_molecules(
        [build_fields(_mol("CCCCS", seed=11), _fs, "probe")]))[0])
    assert g.value != direct, "the gradient was taken at the molecule's own pose after all"


# --------------------------------------------------------------------------- #
# The co-field as a potential
# --------------------------------------------------------------------------- #

def test_the_cofield_is_the_potential_the_atoms_move_in():
    """The identity that ties the two halves of the module together.

    For an unnormalised linear model the derivative of the prediction with respect to the
    position of an atom is the atom's weight times the gradient of the co-field, smoothed by that
    atom's own Gaussian, summed over the fields.  The prediction itself is the same sum of
    values plus the bias.
    """
    model, fs, probe = _fitted(method="krr", normalize=False, params={"lam": 0.1})
    g = prediction_gradient(model, probe, wrt="coords")
    expect = np.zeros_like(g.coords)
    total = 0.0
    for f, spec in enumerate(fs.specs):
        b = (np.full(probe.n_atoms, spec.alpha) if spec.kind == "gauss"
             else spec.alpha / probe.radii ** 2)
        for atom in range(probe.n_atoms):
            point = probe.coords[atom:atom + 1]
            w = probe.weights[atom, f]
            expect[atom] += w * cofield_gradient(model, f, point, width=float(b[atom]))[0]
            total += w * float(smoothed_cofield(model, f, point, width=float(b[atom]))[0])
    np.testing.assert_allclose(g.coords, expect, rtol=1e-9, atol=1e-12)
    assert total + model.bias == pytest.approx(g.value, rel=1e-9)


def test_the_unsmoothed_cofield_is_the_one_the_rest_of_the_program_draws():
    """With no width this is the co-field itself, computed exactly where
    `cofields.cofield_at_points` drops distant Gaussians for speed."""
    model, _fs, probe = _fitted(method="krr", normalize=True, params={"lam": 0.1})
    points = probe.coords[:6]
    mine = smoothed_cofield(model, 0, points)
    theirs = cofield_at_points(model, 0, points)
    np.testing.assert_allclose(mine, theirs, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("probes", [(), ("water",)])
def test_the_cofield_gradient_matches_finite_differences_of_the_cofield(probes):
    """Including a probe field, whose co-field carries several Gaussians per atom and so has a
    width per Gaussian rather than one for the field."""
    model, fs, probe = _fitted(fields=("steric", "electrostatic"), method="krr", normalize=True,
                               params={"lam": 0.1},
                               fieldset_args={"probes": list(probes)})
    points = probe.coords[:4] + 0.4
    for f in range(len(fs.specs)):
        for width in (None, 0.5):
            g = cofield_gradient(model, f, points, width=width)
            num = np.zeros_like(g)
            for k in range(3):
                step = np.zeros(3)
                step[k] = STEP
                num[:, k] = (smoothed_cofield(model, f, points + step, width=width)
                             - smoothed_cofield(model, f, points - step, width=width)) / (2 * STEP)
            rel = np.abs(g - num).max() / max(np.abs(num).max(), 1e-12)
            assert rel < TOL, f"field {fs.names[f]}, width={width}: {rel}"


def test_the_cofield_gradient_needs_a_linear_kernel():
    """With a nonlinear layer the prediction is no longer an overlap integral with a co-field, so
    there is no potential to take a gradient of."""
    model, _fs, probe = _fitted(method="krr", normalize=True, kernel_layer="gauss",
                                params={"lam": 0.1, "gamma": 1.0})
    with pytest.raises(ValueError, match="only for the linear kernel"):
        cofield_gradient(model, 0, probe.coords[:2])


# --------------------------------------------------------------------------- #
# What it refuses, and why
# --------------------------------------------------------------------------- #

def test_an_unfitted_model_is_refused():
    fs = FieldSet.create(physchem=["steric"], alpha=0.5)
    probe = build_fields(_mol("CCCCS", seed=11), fs, "probe")
    with pytest.raises(RuntimeError, match="fitted"):
        prediction_gradient(CMFModel(fieldset=fs), probe)


def test_a_kernel_that_is_not_an_overlap_integral_is_refused():
    """The invariant, surface and periodic kernels are built from the distances inside each
    molecule, so there is no atom of the training set to overlap with."""
    model, _fs, probe = _fitted(method="krr", normalize=True, params={"lam": 0.1})
    model.kernel_source = "autocorrelation"
    with pytest.raises(ValueError, match="overlap-integral kernel"):
        prediction_gradient(model, probe)


def test_a_one_class_score_is_refused_and_says_what_to_use_instead():
    model, _fs, probe = _fitted(task="oneclass", method="ocmean", normalize=True,
                                params={"nu": 0.1})
    with pytest.raises(ValueError, match="posterior variance"):
        prediction_gradient(model, probe)


def test_the_shapes_are_checked():
    model, _fs, probe = _fitted(method="krr", normalize=True, params={"lam": 0.1})
    with pytest.raises(ValueError, match="coords has shape"):
        prediction_gradient(model, probe, coords=probe.coords[:3])
    with pytest.raises(ValueError, match="weights has shape"):
        prediction_gradient(model, probe, weights=probe.weights[:, :1])
    with pytest.raises(ValueError, match="wrt must be"):
        prediction_gradient(model, probe, wrt="everything")


# --------------------------------------------------------------------------- #
# Reading the result
# --------------------------------------------------------------------------- #

def test_the_result_names_the_atoms_and_fields_that_matter_most():
    model, fs, probe = _fitted(method="krr", normalize=True, params={"lam": 0.1})
    g = prediction_gradient(model, probe)
    assert g.field_names == fs.names
    top_atoms = g.largest_coordinate_terms(3)
    assert len(top_atoms) == 3
    sizes = [s for _i, s in top_atoms]
    assert sizes == sorted(sizes, reverse=True)
    assert top_atoms[0][1] == pytest.approx(np.linalg.norm(g.coords, axis=1).max())
    top_weights = g.largest_weight_terms(4)
    assert len(top_weights) == 4
    assert all(name in fs.names for _atom, name, _v in top_weights)
    assert abs(top_weights[0][2]) == pytest.approx(np.abs(g.weights).max())
    np.testing.assert_allclose(g.forces, -g.coords, atol=0)
    assert PredictionGradient(value=1.0).largest_coordinate_terms() == []
    assert PredictionGradient(value=1.0).largest_weight_terms() == []


# --------------------------------------------------------------------------- #
# The block the gradient shares with the atom decomposition
# --------------------------------------------------------------------------- #

def test_a_probe_field_reaches_the_atom_decomposition():
    """A regression test for the block this module and `CMFModel.atom_contributions` share.

    The block is built from the atoms of the molecule being predicted, and a probe field's
    Gaussians live in its expansion coefficients rather than in one weight per atom.  When those
    coefficients were not passed on, a probe field contributed nothing to the block, and the atom
    contributions of a model with probe fields summed to the wrong number -- on the small model
    here, to the wrong sign.
    """
    model, _fs, probe = _fitted(fields=("steric",), method="krr", normalize=True,
                                params={"lam": 0.1},
                                fieldset_args={"probes": ["water", "methyl"]})
    predicted = float(model.predict(AtomTable.from_molecules([probe]))[0])
    assert model.atom_contributions(probe).sum() + model.bias == pytest.approx(predicted,
                                                                               rel=1e-10)
    per_field = model.atom_contributions(probe, per_field=True)
    assert per_field.shape == (3, probe.n_atoms)
    assert np.any(per_field[1] != 0.0), "the probe field contributed nothing"
