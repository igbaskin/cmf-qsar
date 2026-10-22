"""The on-disk kernel cache of the benchmarks must be invisible: same kernels, same model state."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import kernel_cache as KC  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402


def _table(alpha=0.5, n=6):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccc(F)cc1O", "c1ccc(C)cc1O"]
    fs = FieldSet.create(alpha=alpha)
    mf = []
    for i, s in enumerate(smiles[:n]):
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        p = AllChem.ETKDGv3()
        p.randomSeed = 7 + i
        AllChem.EmbedMolecule(m, p)
        mf.append(build_fields(m, fs, f"m{i}"))
    return fs, AtomTable.from_molecules(mf)


@pytest.fixture
def cache(tmp_path):
    KC.enable(str(tmp_path))
    KC.stats.update(hits=0, misses=0)
    yield tmp_path
    KC.enable(None)


def test_cold_and_warm_calls_give_the_direct_kernels_and_the_same_model_state(cache):
    fs, table = _table()
    direct = CMFModel(fieldset=fs, method="krr", normalize=True)
    K_direct = np.array(direct.prepare_training_kernels(table))
    J_direct = direct.joint_kernel(K_direct, table.descriptors)

    cold = CMFModel(fieldset=fs, method="krr", normalize=True)
    J_cold = KC.joint_training_kernel(cold, table)
    assert KC.stats == {"hits": 0, "misses": 1}
    assert len([f for f in os.listdir(cache) if f.endswith(".npz")]) == 1
    np.testing.assert_allclose(J_cold, J_direct)

    warm = CMFModel(fieldset=fs, method="krr", normalize=True)
    warm.prepare_training_kernels = lambda *a, **k: pytest.fail("the warm call computed")
    J_warm = KC.joint_training_kernel(warm, table)
    assert KC.stats == {"hits": 1, "misses": 1}
    np.testing.assert_allclose(J_warm, J_direct)
    np.testing.assert_allclose(warm.K_fields_train, K_direct)
    np.testing.assert_allclose(warm.self_diag, direct.self_diag)
    np.testing.assert_allclose(warm.scale, direct.scale)
    assert warm.train is table


def test_the_key_follows_the_attenuation_the_normalisation_and_the_atoms(cache):
    fs, table = _table(alpha=0.5)
    fs2, table2 = _table(alpha=0.8)
    m = CMFModel(fieldset=fs, method="krr", normalize=True)
    k0 = KC.key_for(m, table)
    assert KC.key_for(CMFModel(fieldset=fs2, method="krr", normalize=True), table2) != k0
    assert KC.key_for(CMFModel(fieldset=fs, method="krr", normalize=False), table) != k0
    moved = AtomTable(coords=table.coords + 0.1, weights=table.weights, radii=table.radii,
                      offsets=table.offsets, names=table.names, molecules=table.molecules)
    assert KC.key_for(m, moved) != k0
    assert KC.key_for(CMFModel(fieldset=fs, method="svr", normalize=True), table) == k0, \
        "the learner does not change the kernel, so it must not change the key"


def test_the_cache_can_be_switched_off(cache, monkeypatch):
    fs, table = _table()
    monkeypatch.setenv("CMF_KERNEL_CACHE", "0")
    KC.enable(None)
    m = CMFModel(fieldset=fs, method="krr", normalize=True)
    KC.joint_training_kernel(m, table)
    assert KC.cache_dir() is None
    assert not [f for f in os.listdir(cache) if f.endswith(".npz")]
