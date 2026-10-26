"""One attenuation per field, chosen by the evidence (plan item D6).

The cheap tests use a synthetic cache of per-field kernels, because what the descent has to get
right is combinatorial rather than chemical: take the right field's matrix from the right width,
never mix a field normalised at one width with its kernel at another, and come back with something
no worse than the shared width it started from.  Two tests then run the real computation end to end.
"""
import os
import sys
import types

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, grid_search  # noqa: E402
from cmfqsar.perfield import assemble, descend, tunable_fields  # noqa: E402

ALPHAS = [0.25, 1.0, 4.0]


def _cache(n=24, fields=3, alphas=ALPHAS, frozen=(), seed=0):
    """A cache shaped like grid_search's: width -> (K_fields, scale, self_diag).

    Every (field, width) gets its own positive-definite matrix, so a field taken from the wrong
    width is detectable; the diagonals are tagged with the same pair, so a mismatched
    normalisation is detectable too.  Fields listed in `frozen` get the same matrix at every
    width, which is what a probe field does.
    """
    rng = np.random.default_rng(seed)
    base = {(f, a): rng.normal(size=(n, n // 2)) for f in range(fields) for a in alphas}
    for f in frozen:
        for a in alphas[1:]:
            base[(f, a)] = base[(f, alphas[0])]
    cache = {}
    for a in alphas:
        Kf = np.stack([base[(f, a)] @ base[(f, a)].T / n for f in range(fields)])
        sd = np.stack([np.full(n, 10 * f + a) for f in range(fields)])
        cache[float(a)] = (Kf, np.ones(fields), sd)
    return cache


def _model(fields=3, **kw):
    fs = FieldSet.create(physchem=["electrostatic", "steric", "hydrophobic", "hba", "hbd"][:fields],
                         alpha=1.0)
    return CMFModel(fieldset=fs, method="krr", normalize=False, scale_fields=False, **kw)


def _train():
    return types.SimpleNamespace(descriptors=None)


def test_assemble_takes_each_field_from_its_own_width():
    cache = _cache()
    K, sd = assemble(cache, [4.0, 0.25, 1.0])
    for f, a in enumerate([4.0, 0.25, 1.0]):
        assert np.allclose(K[f], cache[a][0][f])
        assert np.allclose(sd[f], cache[a][2][f])     # the matching normalisation, not another's


def test_a_field_whose_kernel_ignores_the_width_is_not_searched():
    """A probe field's widths come from its own expansion; alpha does nothing to it."""
    cache = _cache(fields=3, frozen=(1,))
    tune, skip = tunable_fields(cache, ALPHAS, [True, True, True])
    assert tune == [0, 2] and skip == [1]


def test_a_field_multiplied_out_of_the_kernel_is_not_searched():
    cache = _cache(fields=3)
    tune, skip = tunable_fields(cache, ALPHAS, [True, False, True])
    assert tune == [0, 2] and skip == [1]


def test_the_descent_never_ends_below_the_shared_width_it_started_from():
    cache, train = _cache(), _train()
    rng = np.random.default_rng(3)
    y = rng.normal(size=24)
    for start in ALPHAS:
        res = descend(train, y, _model(), cache, ALPHAS, [{"lam": 0.1}, {"lam": 1.0}], start,
                      {"lam": 1.0})
        assert res.log_evidence >= res.shared_log_evidence - 1e-9
        assert res.gain >= -1e-9


def test_the_descent_finds_the_best_assignment_the_grid_holds():
    """On a space small enough to enumerate, the coordinate descent is not merely an improvement."""
    cache, train = _cache(fields=2), _train()
    rng = np.random.default_rng(1)
    K = cache[1.0][0][0] + cache[4.0][0][1]
    y = np.linalg.cholesky(K + 1e-3 * np.eye(len(K))) @ rng.normal(size=len(K))
    grid = [{"lam": 0.01}, {"lam": 0.1}, {"lam": 1.0}]
    best = max(log_evidence(assemble(cache, [a, b])[0].sum(0), y, p["lam"], fit_bias=True)
               for a in ALPHAS for b in ALPHAS for p in grid)
    res = descend(train, y, _model(fields=2), cache, ALPHAS, grid, 1.0, {"lam": 1.0}, passes=3)
    assert res.log_evidence == pytest.approx(best, abs=1e-6)
    assert res.alphas == {"electrostatic": 1.0, "steric": 4.0}


def test_the_model_is_left_holding_the_widths_that_were_chosen():
    cache, train = _cache(), _train()
    rng = np.random.default_rng(5)
    y = rng.normal(size=24)
    model = _model()
    res = descend(train, y, model, cache, ALPHAS, [{"lam": 0.1}, {"lam": 2.0}], 1.0, {"lam": 2.0})
    assert {s.name: s.alpha for s in model.fieldset.specs} == res.alphas
    assert model.params["lam"] == res.params["lam"]
    assert model.K_fields_train.shape[0] == len(res.alphas)
    for f, a in enumerate(res.alphas.values()):
        assert np.allclose(model.K_fields_train[f], cache[a][0][f])


def test_a_width_outside_the_shared_grid_cannot_be_assembled():
    cache, train = _cache(), _train()
    with pytest.raises(ValueError, match="cached kernels"):
        descend(train, np.zeros(24), _model(), cache, ALPHAS + [7.0], [{"lam": 1.0}], 1.0,
                {"lam": 1.0})


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _set():
    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
              "c1ccccc1OC", "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
              "c1ccc(Br)cc1O", "c1ccc(N)cc1C"]
    fs = FieldSet.create(alpha=1.0)
    table = AtomTable.from_molecules([build_fields(_mol(s), fs, f"m{i}")
                                      for i, s in enumerate(smiles)])
    y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.4])
    return table, y, fs


def test_the_search_reports_one_width_per_field_and_improves_on_the_shared_one():
    table, y, fs = _set()
    alphas, lams = [0.3, 1.0, 2.0], [0.01, 0.1, 1.0]
    shared = grid_search(table, y, CMFModel(fieldset=fs.copy()), alphas, lams=lams,
                         select="evidence")
    model = CMFModel(fieldset=fs.copy())
    res = grid_search(table, y, model, alphas, lams=lams, select="evidence",
                      alpha_search="per-field")
    assert set(res.best_alphas) == set(fs.names)
    assert res.best_q2 >= shared.best_q2 - 1e-9          # the evidence it was selected on
    assert res.per_field.gain >= -1e-9
    assert res.per_field.n_eval > 0
    assert any(row.get("per_field") for row in res.table)
    # the widths the descent chose are the ones a prediction will use
    assert {s.name: s.alpha for s in model.fieldset.specs} == res.best_alphas
    model.fit(table, y, cv_folds=4, cv_seed=None, reuse_kernels=True)
    assert np.isfinite(model.predict(table)).all()


def test_the_per_field_search_refuses_what_it_cannot_honestly_do():
    table, y, fs = _set()
    with pytest.raises(ValueError, match="searched by the evidence"):
        grid_search(table, y, CMFModel(fieldset=fs.copy()), [0.5, 1.0], lams=[0.1],
                    alpha_search="per-field")
    with pytest.raises(ValueError, match="at least two widths"):
        grid_search(table, y, CMFModel(fieldset=fs.copy()), [1.0], lams=[0.1],
                    select="evidence", alpha_search="per-field")
    with pytest.raises(ValueError, match="alpha-fields"):
        grid_search(table, y, CMFModel(fieldset=fs.copy()), [0.5, 1.0], lams=[0.1],
                    select="evidence", alpha_search="per-field", alpha_fields=["steric"])
