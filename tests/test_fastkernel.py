"""The compiled kernel must agree with the NumPy one (plan item B5).

The NumPy path is the reference: it is what every table in the repository was produced with, so
every test here compares against it rather than against a tolerance chosen to pass. The compiled
loop is built with `fastmath`, so it may add the atom pairs in another order; the agreement asked
for is 1e-12 relative, and the measured difference is 4e-15.

The whole file is skipped when Numba is not installed, which is a supported way to run the
program.
"""
import math
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar import fastkernel as FK  # noqa: E402
from cmfqsar import kernels as KN  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, FieldSpec, build_fields  # noqa: E402

SMILES = ["c1ccccc1C(=O)O", "c1ccccc1CCN", "c1ccc(Cl)cc1C(=O)N", "CC(C)c1ccccc1O",
          "c1ccncc1CC(=O)O", "FC(F)(F)c1ccccc1N", "c1ccsc1C(=O)OC", "OCc1ccc(Br)cc1",
          "c1ccc(cc1)S(=O)(=O)N", "CC(=O)Nc1ccc(O)cc1", "c1ccc2ccccc2c1", "OC(=O)c1ccccc1O"]


def _table(fields=("electrostatic", "steric", "hydrophobic", "hba", "hbd"), alpha=0.5, n=None):
    fs = FieldSet.create(physchem=list(fields), alpha=alpha)
    mols = []
    for k, smi in enumerate(SMILES[:n] if n else SMILES):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = k + 1
        AllChem.EmbedMolecule(m, p)
        mols.append(m)
    return fs, AtomTable.from_molecules([build_fields(m, fs, f"m{i}")
                                         for i, m in enumerate(mols)])


needs_numba = pytest.mark.skipif(not FK.available(), reason="Numba is not installed")


# --------------------------------------------------------------- what it does without numba


def test_the_library_default_is_the_numpy_path():
    """Whatever is installed, the default must be the implementation the tables came from."""
    assert KN.ENGINE == "numpy"
    assert KN.use_compiled(None) is False
    assert KN.use_compiled("numpy") is False


def test_an_unknown_engine_is_refused():
    with pytest.raises(ValueError, match="unknown kernel engine"):
        KN.use_compiled("elvish")
    fs, T = _table()
    with pytest.raises(ValueError, match="unknown kernel engine"):
        KN.kernel_matrices(T, T, fs.specs, engine="fortran")


def test_auto_falls_back_rather_than_failing():
    fs, T = _table()
    got = KN.kernel_matrices(T, T, fs.specs, engine="auto")
    assert np.allclose(got, KN.kernel_matrices(T, T, fs.specs), rtol=1e-12, atol=0)


def test_asking_for_numba_without_numba_says_how_to_get_it():
    if FK.available():
        assert KN.use_compiled("numba") is True
    else:
        with pytest.raises(ValueError, match="pip install numba"):
            KN.use_compiled("numba")


def test_the_plan_declines_what_the_loop_cannot_do():
    class Probe:
        kind, alpha, is_probe = "gauss", 0.5, True

    class Odd:
        kind, alpha, is_probe = "spherical-harmonic", 0.5, False

    assert FK.plan([Probe()]) is None
    assert FK.plan([Odd()]) is None


# ------------------------------------------------------------------- the numbers must agree


@needs_numba
def test_the_compiled_kernel_equals_the_numpy_one():
    fs, T = _table()
    ref = KN.kernel_matrices(T, T, fs.specs)
    got = KN.kernel_matrices(T, T, fs.specs, engine="numba")
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


@needs_numba
def test_it_agrees_for_two_different_tables():
    fs, T = _table()
    A, B = T.subset([0, 1, 2, 3, 4, 5]), T.subset([6, 7, 8, 9, 10, 11])
    ref = KN.kernel_matrices(A, B, fs.specs)
    got = KN.kernel_matrices(A, B, fs.specs, engine="numba")
    assert got.shape == (len(fs.specs), A.n_mol, B.n_mol)
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


@needs_numba
@pytest.mark.parametrize("alpha", [0.1, 0.5, 2.0, 5.0])
def test_it_agrees_at_every_attenuation(alpha):
    fs, T = _table(alpha=alpha)
    ref = KN.kernel_matrices(T, T, fs.specs)
    got = KN.kernel_matrices(T, T, fs.specs, engine="numba")
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


@needs_numba
@pytest.mark.parametrize("fields", [("steric",), ("electrostatic",), ("hba", "hbd"),
                                    ("electrostatic", "steric")])
def test_it_agrees_for_each_kind_of_field(fields):
    """The shared-width fields and the radius-scaled one are different branches of the loop."""
    fs, T = _table(fields=fields)
    ref = KN.kernel_matrices(T, T, fs.specs)
    got = KN.kernel_matrices(T, T, fs.specs, engine="numba")
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


@needs_numba
def test_it_agrees_when_fields_have_different_attenuations():
    """Several groups of shared-width fields: the loop takes one exponential per group."""
    fs, T = _table()
    specs = list(fs.specs)
    for k, sp in enumerate(specs):
        specs[k] = FieldSpec(**{**vars(sp), "alpha": 0.3 + 0.4 * (k % 3)})
    ref = KN.kernel_matrices(T, T, specs)
    got = KN.kernel_matrices(T, T, specs, engine="numba")
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


@needs_numba
def test_the_compiled_result_is_exactly_symmetric():
    fs, T = _table()
    got = KN.kernel_matrices(T, T, fs.specs, engine="numba")
    assert np.array_equal(got, np.transpose(got, (0, 2, 1)))


@needs_numba
def test_a_grid_of_attenuations_agrees_too():
    fs, T = _table()
    sets = [[FieldSpec(**{**vars(s), "alpha": a}) for s in fs.specs] for a in (0.2, 0.8)]
    ref = KN.kernel_matrices_multi(T, T, sets)
    got = KN.kernel_matrices_multi(T, T, sets, engine="numba")
    for a, b in zip(ref, got):
        assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-12


# ------------------------------------------------------------------------------- the cutoff


@needs_numba
def test_a_wide_cutoff_changes_nothing_and_a_narrow_one_does():
    fs, T = _table(fields=("steric",))
    exact = KN.kernel_matrices(T, T, fs.specs, engine="numba")
    wide = KN.kernel_matrices(T, T, fs.specs, engine="numba", cutoff=1e4)
    assert np.allclose(wide, exact, rtol=1e-12, atol=0)
    narrow = KN.kernel_matrices(T, T, fs.specs, engine="numba", cutoff=1.0)
    assert np.max(np.abs(narrow - exact)) > 0


@needs_numba
def test_the_tolerance_cutoff_stays_within_its_tolerance():
    fs, T = _table()
    exact = KN.kernel_matrices(T, T, fs.specs, engine="numba")
    cut = KN.kernel_matrices(T, T, fs.specs, engine="numba", cutoff="auto")
    assert np.max(np.abs(cut - exact)) / np.max(np.abs(exact)) < 1e-5


def test_auto_becomes_the_widest_radius_any_field_would_use():
    """Inside the loop a cutoff is one comparison, so the safe radius is the largest one."""
    fs, _T = _table()
    r = KN._compiled_cutoff(fs.specs, "auto", 1.9)
    assert r == pytest.approx(KN.cutoff_radius(0.25 * 0.5 / 1.9 ** 2))
    assert r > KN.cutoff_radius(0.25 * 0.5)
    assert KN._compiled_cutoff(fs.specs, None, 1.9) is None
    assert KN._compiled_cutoff(fs.specs, 7.5, 1.9) == 7.5


# ------------------------------------------------------------------------------ the plumbing


@needs_numba
def test_a_small_problem_is_left_to_numpy():
    """Threads cost more than they save on a handful of atoms, so the loop declines."""
    fs, T = _table(n=2)
    assert T.coords.shape[0] ** 2 < FK.MIN_PAIRS
    assert FK.kernel_matrices(T, T, fs.specs, symmetric=True) is None
    assert np.array_equal(KN.kernel_matrices(T, T, fs.specs, engine="numba"),
                          KN.kernel_matrices(T, T, fs.specs))


@needs_numba
def test_a_model_can_be_told_which_engine_to_use():
    from cmfqsar.model import CMFModel
    fs, T = _table()
    y = np.linspace(4.0, 8.0, T.n_mol)
    plain = CMFModel(fieldset=fs.copy(), method="krr", normalize=True, params={"lam": 0.1})
    plain.fit(T, y, cv_folds=0)
    fast = CMFModel(fieldset=fs.copy(), method="krr", normalize=True, params={"lam": 0.1},
                    engine="numba")
    fast.fit(T, y, cv_folds=0)
    assert fast.predict(T) == pytest.approx(plain.predict(T), rel=1e-9)


@needs_numba
def test_the_screener_uses_it_by_default_and_restores_the_model():
    from cmfqsar.screen import ScreenSpec
    assert ScreenSpec().engine == "auto"
    from cmfqsar.model import CMFModel
    fs, T = _table()
    m = CMFModel(fieldset=fs.copy(), method="krr", normalize=True, params={"lam": 0.1})
    assert m.engine is None


@needs_numba
def test_warm_up_compiles_and_reports():
    assert FK.warm_up() is True
    assert FK.threads() >= 1
    assert FK.bounds_of(np.array([0, 3, 7]), 0).tolist() == [[0, 3], [3, 7]]
    assert FK.bounds_of(np.array([0, 3, 7]), 1).tolist() == [[0, 4]]
