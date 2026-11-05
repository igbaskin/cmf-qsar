"""The electric-field kernel: the integral of two fields' gradients (plan item EC5).

Every other field in the program is a scalar and its kernel is the overlap of two scalars. This
one pairs the *gradients*, which is the quantity a reaction centre feels, and the pair integral
is in closed form. Three things therefore have to be checked rather than assumed.

* The closed form is the integral it claims to be. It is compared against a grid integration,
  for one pair of Gaussians and for two whole molecules, because an algebra slip gives a
  perfectly well-behaved kernel for the wrong quantity.
* The kernel is still a kernel. The pair integral changes sign -- two charges far apart have
  gradients that point against each other -- which no other pair integral here does, so the
  positive semidefiniteness of the matrix is checked instead of inferred.
* The compiled loop does not implement this kind, so it must decline the whole call rather than
  handle the field halfway.
"""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar import fastkernel
from cmfqsar.cofields import (Grid, cofield_arrows, cofield_gradient, cofield_volume,
                              evaluate_gaussian_gradients, make_grid)
from cmfqsar.fields import (EFIELD_FIELD, FIELD_LABELS, PHYSCHEM_AVAILABLE, PHYSCHEM_FIELDS,
                            AtomTable, FieldSet, build_fields)
from cmfqsar.kernels import (atom_kernel_block, efield_pair_factor, kernel_gradient,
                             kernel_matrices, self_kernels)
from cmfqsar.model import CMFModel

SMILES = ("c1ccccc1N", "c1ccccc1O", "c1ccccc1C", "c1ccccc1F", "c1ccccc1Cl", "c1ccccc1C#N")


def mol(smiles: str, seed: int = 1) -> Chem.Mol:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def series(n: int = 5):
    mols = [mol(s, seed=i + 1) for i, s in enumerate(SMILES[:n])]
    names = [f"m{i}" for i in range(len(mols))]
    return mols, names


def fieldset(names=("electrostatic", EFIELD_FIELD), alpha: float = 1.0) -> FieldSet:
    return FieldSet.create(physchem=list(names), alpha=alpha, charge_model="gasteiger",
                           param_scheme="rdkit")


# ------------------------------------------------------------------ the closed form

def grid_gradient_overlap(centres_a, weights_a, centres_b, weights_b, alpha,
                          half_width=9.0, n=140):
    """The integral of the two fields' gradients, by quadrature on a cubic grid.

    The field of a molecule is ``sum_i w_i exp(-(alpha/2) |r - r_i|^2)``, which is the
    convention the kernels of this program use, and the integral is of the dot product of the
    two gradients over all space.
    """
    x = np.linspace(-half_width, half_width, n)
    h = x[1] - x[0]
    grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    ga = evaluate_gaussian_gradients(grid, np.asarray(centres_a, float),
                                     np.asarray(weights_a, float),
                                     np.full(len(centres_a), alpha))
    gb = evaluate_gaussian_gradients(grid, np.asarray(centres_b, float),
                                     np.asarray(weights_b, float),
                                     np.full(len(centres_b), alpha))
    return float(np.einsum("ij,ij->", ga, gb) * h ** 3)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("d", [0.0, 1.0, 2.0, 3.0])
def test_the_pair_integral_is_the_integral_it_claims(alpha, d):
    """One pair of Gaussians, closed form against quadrature."""
    got = efield_pair_factor(alpha, np.array([d * d]))[0]
    want = grid_gradient_overlap([[0.0, 0.0, 0.0]], [1.0], [[d, 0.0, 0.0]], [1.0], alpha)
    assert got == pytest.approx(want, rel=2e-4, abs=1e-6)


def test_the_pair_integral_changes_sign_where_the_formula_says():
    """It is zero at d^2 = 6 / alpha, positive inside and negative outside."""
    for alpha in (0.5, 1.0, 2.0):
        zero = 6.0 / alpha
        assert efield_pair_factor(alpha, np.array([zero]))[0] == pytest.approx(0.0, abs=1e-12)
        assert efield_pair_factor(alpha, np.array([0.5 * zero]))[0] > 0
        assert efield_pair_factor(alpha, np.array([2.0 * zero]))[0] < 0


def test_the_pair_integral_is_the_overlap_times_a_factor():
    """At d = 0 it is 3 alpha / 2 times the ordinary overlap of the same two Gaussians."""
    for alpha in (0.5, 1.0, 2.0):
        overlap = (np.pi / alpha) ** 1.5
        assert efield_pair_factor(alpha, np.array([0.0]))[0] == pytest.approx(1.5 * alpha
                                                                             * overlap)


def test_the_kernel_of_two_molecules_is_the_integral_of_their_gradients():
    """The whole kernel, not one pair: closed form against quadrature on two molecules."""
    alpha = 1.0
    mols, names = series(2)
    fs = fieldset((EFIELD_FIELD,), alpha=alpha)
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    # both molecules are moved onto the origin, so that one grid covers them both
    for f in mf:
        f.coords = f.coords - f.coords.mean(axis=0)
    table = AtomTable.from_molecules(mf)
    K = kernel_matrices(table, table, fs.specs)
    want = grid_gradient_overlap(mf[0].coords, mf[0].weights[:, 0],
                                 mf[1].coords, mf[1].weights[:, 0], alpha)
    assert K[0, 0, 1] == pytest.approx(want, rel=5e-3)
    self_a = grid_gradient_overlap(mf[0].coords, mf[0].weights[:, 0],
                                   mf[0].coords, mf[0].weights[:, 0], alpha)
    assert K[0, 0, 0] == pytest.approx(self_a, rel=5e-3)


# ------------------------------------------------------------------ it is still a kernel

def test_the_kernel_matrix_is_positive_semidefinite():
    mols, names = series()
    fs = fieldset()
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    K = kernel_matrices(table, table, fs.specs)
    for f in range(K.shape[0]):
        assert np.allclose(K[f], K[f].T, atol=1e-10)
        assert np.linalg.eigvalsh(K[f]).min() > -1e-9


def test_the_self_kernel_is_positive():
    """A field's own gradient integral is the squared norm of a vector field, so it cannot be
    negative, however many atom pairs contribute a negative term."""
    mols, names = series()
    fs = fieldset()
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    diag = self_kernels(table, fs.specs)
    assert np.all(diag > 0)
    # the electric-field kind and the electrostatic field are different quantities, so their
    # self-kernels differ; if they came out equal the kind would not have been applied
    assert not np.allclose(diag[0], diag[1])


def test_the_kernel_does_not_change_with_the_frame():
    """Both fields are rotation invariant, since the kernel depends only on atom distances."""
    mols, names = series(3)
    fs = fieldset()
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    K = kernel_matrices(table, table, fs.specs)
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    for f in mf:
        f.coords = f.coords @ q + np.array([3.0, -2.0, 1.0])
    turned = AtomTable.from_molecules(mf)
    assert np.allclose(K, kernel_matrices(turned, turned, fs.specs), atol=1e-9)


# ------------------------------------------------------------------ the field set

def test_the_field_carries_the_charges_of_the_electrostatic_field():
    mols, names = series(1)
    fs = fieldset()
    mf = build_fields(mols[0], fs, names[0])
    w = np.asarray(mf.weights)
    assert np.allclose(w[:, 0], w[:, 1])
    assert abs(w[:, 1].sum()) < 1e-6              # a neutral molecule


def test_the_field_is_offered_but_not_a_default():
    assert EFIELD_FIELD not in PHYSCHEM_FIELDS
    assert EFIELD_FIELD in PHYSCHEM_AVAILABLE
    assert EFIELD_FIELD in FIELD_LABELS
    fs = fieldset()
    assert [s.kind for s in fs.specs] == ["gauss", "efield"]
    assert FieldSet.create().specs[0].kind == "gauss"


def test_the_compiled_loop_declines_the_whole_field_set():
    """It does not implement this kind, so it must hand the call back rather than do half."""
    fs = fieldset()
    assert fastkernel.plan(fs.specs) is None
    assert fastkernel.plan(FieldSet.create().specs) is not None or not fastkernel.HAVE_NUMBA


def test_the_two_engines_agree_on_a_field_set_without_the_new_kind():
    """The fall-back must not change the ordinary fields it is handed alongside."""
    mols, names = series(3)
    fs = fieldset()
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    numpy_only = kernel_matrices(table, table, fs.specs, engine="numpy")
    either = kernel_matrices(table, table, fs.specs, engine="auto")
    assert np.allclose(numpy_only, either, atol=1e-12)


# ------------------------------------------------------------------ the derivatives

def test_the_kernel_gradient_matches_a_finite_difference():
    mols, names = series(2)
    fs = fieldset((EFIELD_FIELD,))
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    probe = mf[0]
    u = np.ones((1, table.n_mol))

    def total(coords):
        blk = atom_kernel_block(coords, probe.weights, probe.radii, table, fs.specs)
        return float(blk.sum())

    analytic = kernel_gradient(probe.coords, probe.weights, probe.radii, table, fs.specs, u)
    step = 1e-5
    for atom in (0, 3, len(probe.coords) - 1):
        for axis in range(3):
            plus = probe.coords.copy()
            minus = probe.coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numeric = (total(plus) - total(minus)) / (2 * step)
            assert analytic[atom, axis] == pytest.approx(numeric, rel=1e-4, abs=1e-7)


def test_the_cofield_gradient_matches_a_finite_difference_of_the_cofield():
    mols, names = series()
    y = np.array([0.1, 0.4, -0.2, 0.7, -0.5])
    fs = fieldset((EFIELD_FIELD,))
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), y, cv_folds=0)
    centre = mf[0].coords.mean(axis=0)
    step = 1e-4
    point = Grid(origin=centre, spacing=1.0, shape=(1, 1, 1))
    analytic = cofield_gradient(model, 0, point)[:, 0, 0, 0]
    for axis in range(3):
        lo = centre.copy()
        hi = centre.copy()
        lo[axis] -= step
        hi[axis] += step
        v_lo = cofield_volume(model, 0, Grid(origin=lo, spacing=1.0, shape=(1, 1, 1)))[0, 0, 0]
        v_hi = cofield_volume(model, 0, Grid(origin=hi, spacing=1.0, shape=(1, 1, 1)))[0, 0, 0]
        assert analytic[axis] == pytest.approx((v_hi - v_lo) / (2 * step), rel=1e-3, abs=1e-9)


def test_the_arrows_are_ordered_and_separated():
    mols, names = series()
    y = np.array([0.1, 0.4, -0.2, 0.7, -0.5])
    fs = fieldset((EFIELD_FIELD,))
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), y, cv_folds=0)
    grid = make_grid(np.vstack([m.coords for m in mf]), margin=2.0, spacing=0.6)
    points, vectors, lengths = cofield_arrows(model, 0, grid, count=6, min_separation=1.5)
    assert len(points) == len(vectors) == len(lengths) == 6
    assert np.all(np.diff(lengths) <= 1e-12)        # strongest first
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            assert np.linalg.norm(points[i] - points[j]) >= 1.5 - 1e-9
    assert np.allclose(np.linalg.norm(vectors, axis=1), lengths)


# ------------------------------------------------------------------ end to end

def test_a_model_with_the_new_field_trains_and_predicts():
    mols, names = series()
    y = np.array([0.1, 0.4, -0.2, 0.7, -0.5])
    fs = fieldset()
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.01})
    model.fit(table, y, cv_folds=0)
    pred = model.predict(table)
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))
    # the Shapley decomposition is unchanged by the new kind: it is still a sum over atoms
    centres, coeffs, widths = model.cofield_coefficients(1)
    assert len(centres) == len(coeffs) == len(widths)
    assert np.all(widths > 0)
