"""Co-alignment as EM: the objective, the acceptance test, the rollback and the contraction stop."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar.coalign import CoalignResult, coalign, leave_fold_out_error
from cmfqsar.fields import AtomTable, FieldSet, build_fields
from cmfqsar.model import CMFModel


def _set(n=12, seed=0):
    """A small aligned series with an activity that depends on its geometry."""
    from cmfqsar.align import align_dataset

    subs = ["", "C", "CC", "F", "Cl", "O", "OC", "N", "NC", "C#N", "CO", "Br", "CCC", "I"][:n]
    mols, names = [], []
    for i, s in enumerate(subs):
        m = Chem.AddHs(Chem.MolFromSmiles(f"c1ccc(cc1){s}" if s else "c1ccccc1"))
        AllChem.EmbedMolecule(m, randomSeed=30 + i)
        AllChem.MMFFOptimizeMolecule(m)
        Chem.SanitizeMol(m)
        m.SetProp("_Name", f"m{i}")
        mols.append(m)
        names.append(f"m{i}")
    align_dataset(mols, 0)
    rng = np.random.default_rng(seed)
    y = np.array([4.0 + 0.25 * m.GetNumHeavyAtoms() for m in mols]) + 0.1 * rng.normal(size=len(mols))
    return mols, names, y


def _template(fs):
    return CMFModel(fieldset=fs, method="gp", normalize=True, params={"lam": 0.1})


def _fields(mols, names, fs):
    return [build_fields(m, fs, n) for m, n in zip(mols, names)]


def test_the_objective_uses_no_activity_of_the_molecule_it_scores():
    """Every molecule is predicted by a model fitted without it, so the error is honest."""
    mols, names, y = _set()
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    mf = _fields(mols, names, fs)
    err = leave_fold_out_error(mf, y, _template(fs), lam=0.1, folds=4, seed=0)
    assert np.isfinite(err) and err > 0
    # scrambling the activities must destroy it, which a leaky implementation would not
    shuffled = np.random.RandomState(0).permutation(y)
    err_shuffled = leave_fold_out_error(mf, shuffled, _template(fs), lam=0.1, folds=4, seed=0)
    assert err_shuffled > err


def test_a_round_that_does_not_help_is_rolled_back_exactly():
    """The poses must come back bit for bit, or a failed run quietly damages the training set."""
    mols, names, y = _set()
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    mf = _fields(mols, names, fs)
    before = [np.array(m.coords, float) for m in mf]
    out = coalign(mf, y, _template(fs), rounds=2, folds=4, max_rmsd=0.5, seed=0)
    assert isinstance(out, CoalignResult)
    if out.best_round == 0:                      # nothing was accepted
        for m, x in zip(mf, before):
            np.testing.assert_allclose(m.coords, x, atol=1e-12)
        assert out.improved <= 0.0 or not np.isfinite(out.improved)
    else:
        assert out.loo_rmse <= out.loo_rmse_start + 1e-12


def test_an_accepted_round_improved_the_objective_and_a_rejected_one_stopped_the_run():
    mols, names, y = _set(14)
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    mf = _fields(mols, names, fs)
    out = coalign(mf, y, _template(fs), rounds=4, folds=4, max_rmsd=1.0, seed=1)
    assert out.rounds, "the iteration must report its rounds"
    for r in out.rounds:
        if r.accepted:
            assert r.loo_rmse < out.loo_rmse_start + 1e-9
    kept = [r for r in out.rounds if r.accepted]
    if len(kept) < len(out.rounds):              # a rejection ends the run
        assert not out.rounds[-1].accepted
        assert out.stopped
    assert all(np.isfinite(r.similarity) for r in out.rounds)


def test_the_ridge_is_annealed_downwards():
    mols, names, y = _set(14)
    fs = FieldSet.create(physchem=["steric"], alpha=0.5)
    mf = _fields(mols, names, fs)
    out = coalign(mf, y, _template(fs), rounds=3, folds=4, max_rmsd=1.0,
                  lam_start=1.6, lam_factor=0.5, seed=2)
    lams = [r.lam for r in out.rounds]
    assert lams[0] == pytest.approx(1.6)
    assert all(b <= a + 1e-12 for a, b in zip(lams, lams[1:])), "lambda must not rise"
    assert min(lams) >= 0.1 - 1e-12, "it must not fall below the model's own ridge"


def test_a_contracting_ensemble_stops_the_run():
    """The degeneracy of the criterion is that everything piles up; the tolerance catches it."""
    mols, names, y = _set(12)
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    mf = _fields(mols, names, fs)
    before = [np.array(m.coords, float) for m in mf]
    out = coalign(mf, y, _template(fs), rounds=3, folds=4, max_rmsd=None,
                  max_similarity=0.0, seed=0)          # any similarity at all is "too much"
    assert out.best_round == 0
    assert "contract" in out.stopped
    for m, x in zip(mf, before):                       # and the poses are untouched
        np.testing.assert_allclose(m.coords, x, atol=1e-12)


def test_the_flexible_step_needs_the_molecules_and_says_so():
    mols, names, y = _set(12)
    fs = FieldSet.create(physchem=["steric"], alpha=0.5)
    mf = _fields(mols, names, fs)
    with pytest.raises(ValueError):
        coalign(mf, y, _template(fs), rounds=1, flexible=True, rdmols=None)
    with pytest.raises(ValueError):
        coalign(mf, y, _template(fs), rounds=1, flexible=True, rdmols=mols[:3])


def test_too_small_a_set_is_refused():
    mols, names, y = _set(12)
    fs = FieldSet.create(physchem=["steric"], alpha=0.5)
    mf = _fields(mols, names, fs)
    with pytest.raises(ValueError):
        coalign(mf[:4], y[:4], _template(fs), rounds=1)


def test_the_report_names_the_objective_and_the_reason_it_stopped():
    mols, names, y = _set(12)
    fs = FieldSet.create(physchem=["steric"], alpha=0.5)
    mf = _fields(mols, names, fs)
    out = coalign(mf, y, _template(fs), rounds=2, folds=4, max_rmsd=1.0, seed=3)
    text = out.describe()
    assert "leave-fold-out RMSE" in text and "similarity" in text
    assert out.stopped and out.stopped in text or "kept round" in text
