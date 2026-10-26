"""Non-field descriptors: one number per molecule, beside the fields."""
import numpy as np
import pytest
from rdkit import Chem

from cmfqsar.descriptors import (DescriptorScaler, contributions, descriptor_kernel,
                                 descriptor_self, read_values)
from cmfqsar.fields import AtomTable, FieldSet, build_fields
from cmfqsar.model import CMFModel


def _mol(smiles, name, **props):
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=0xF00D)
    m.SetProp("_Name", name)
    for k, v in props.items():
        m.SetProp(k, str(v))
    return m


def _series(n=10):
    """A series whose activity is a field effect plus a descriptor effect."""
    mols, y = [], []
    for i in range(n):
        smiles = "c1ccccc1" + "C" * (i % 3)
        m = _mol(smiles, f"m{i}", logp=0.5 * i, medium=float(i % 2))
        mols.append(m)
        y.append(0.3 * (i % 3) + 0.7 * (i % 2))
    return mols, np.array(y, float)


def test_scaler_uses_training_statistics_only():
    sc = DescriptorScaler.fit(np.array([[1.0], [2.0], [3.0]]), ["d"])
    assert sc.mean[0] == pytest.approx(2.0)
    # a test molecule is standardised with the training mean and spread, not its own
    z = sc.transform(np.array([[2.0], [100.0]]))
    assert z[0, 0] == pytest.approx(0.0)
    assert z[1, 0] > 100


def test_missing_values_are_refused_or_imputed():
    V = np.array([[1.0], [np.nan], [3.0]])
    with pytest.raises(ValueError, match="missing or unreadable"):
        DescriptorScaler.fit(V, ["d"], missing="error")
    sc = DescriptorScaler.fit(V, ["d"], missing="mean")
    assert sc.transform(V)[1, 0] == pytest.approx(0.0)   # the training mean, i.e. no information


def test_a_constant_descriptor_is_inert():
    sc = DescriptorScaler.fit(np.array([[5.0], [5.0], [5.0]]), ["flat"])
    assert sc.constant == ["flat"]
    assert np.allclose(sc.transform(np.array([[5.0], [9.0]])), 0.0)


def test_read_values_tolerates_text_and_absence():
    m = _mol("CC", "x", good="1.5", bad="not a number")
    v = read_values(m, ["good", "bad", "absent"])
    assert v[0] == pytest.approx(1.5)
    assert np.isnan(v[1]) and np.isnan(v[2])


def test_block_is_a_dot_product_and_splits_exactly():
    rng = np.random.RandomState(0)
    Z = rng.randn(6, 3)
    a = rng.randn(6)
    z = rng.randn(3)
    whole = (descriptor_kernel(z[None, :], Z) @ a).item()
    parts = contributions(z, Z, a)
    assert parts.sum() == pytest.approx(whole)
    assert len(parts) == 3
    assert descriptor_self(Z)[0] == pytest.approx(float(Z[0] @ Z[0]))


def test_descriptors_reach_the_model_and_change_it():
    mols, y = _series()
    names = [m.GetProp("_Name") for m in mols]
    plain = FieldSet.create(physchem=["steric"], alpha=0.5)
    withd = FieldSet.create(physchem=["steric"], alpha=0.5, descriptors=("logp", "medium"))

    def fit(fs):
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        table = AtomTable.from_molecules(mf)
        model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.01})
        model.fit(table, y, cv_folds=0)
        return model, table, mf

    m0, t0, _ = fit(plain)
    m1, t1, mf1 = fit(withd)
    assert t0.descriptors is None and t1.descriptors.shape == (len(mols), 2)
    assert m1.desc_train_ is not None and m1.desc_scale_ > 0
    assert not np.allclose(m0.predict(t0), m1.predict(t1)), "the block has to change something"
    # the block really is part of the kernel the model used
    K_fields_only = m1.joint_kernel(m1.K_fields_train)
    K_with = m1.joint_kernel(m1.K_fields_train, t1.descriptors)
    assert not np.allclose(K_fields_only, K_with)
    assert np.allclose(K_with - K_fields_only, m1.descriptor_block(t1.descriptors))


def test_prediction_equals_fields_plus_descriptor_contributions():
    mols, y = _series()
    names = [m.GetProp("_Name") for m in mols]
    fs = FieldSet.create(physchem=["steric"], alpha=0.5, descriptors=("logp",))
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.01})
    model.fit(table, y, cv_folds=0)

    one = AtomTable.from_molecules([mf[0]])
    pred = float(model.predict(one)[0])
    field_part = (model.joint_kernel(model.test_kernels(one)) @ model.coef).item() + model.bias
    desc_part = sum(v for _, v in model.descriptor_contributions(mf[0]))
    assert pred == pytest.approx(field_part + desc_part, abs=1e-9)


def test_h_zero_switches_the_block_off():
    mols, y = _series()
    names = [m.GetProp("_Name") for m in mols]
    off = FieldSet.create(physchem=["steric"], alpha=0.5, descriptors=("logp",), descriptor_h=0.0)
    plain = FieldSet.create(physchem=["steric"], alpha=0.5)

    def predict(fs):
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        table = AtomTable.from_molecules(mf)
        model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.01})
        model.fit(table, y, cv_folds=0)
        return model.predict(table)

    assert np.allclose(predict(off), predict(plain))


def test_field_kernels_alone_refuse_to_predict_a_descriptor_model():
    mols, y = _series()
    names = [m.GetProp("_Name") for m in mols]
    fs = FieldSet.create(physchem=["steric"], alpha=0.5, descriptors=("logp",))
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.01})
    model.fit(table, y, cv_folds=0)
    with pytest.raises(ValueError, match="non-field descriptors"):
        model.predict_from_field_kernels(model.K_fields_train)


def test_computed_descriptors_match_rdkit():
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    from cmfqsar.descriptors import compute, computed_names

    m = _mol("CCOc1ccc(CCN)cc1", "x")
    assert computed_names() == ["calc:mw", "calc:clogp", "calc:tpsa", "calc:rotatable"]
    assert compute(m, "calc:mw") == pytest.approx(Descriptors.MolWt(m))
    assert compute(m, "calc:clogp") == pytest.approx(Crippen.MolLogP(m))
    assert compute(m, "calc:tpsa") == pytest.approx(Descriptors.TPSA(m))
    assert compute(m, "calc:rotatable") == pytest.approx(Lipinski.NumRotatableBonds(m))


def test_an_unknown_computed_descriptor_is_named_rather_than_guessed():
    from cmfqsar.descriptors import compute
    with pytest.raises(KeyError, match="unknown computed descriptor"):
        compute(_mol("CC", "x"), "calc:enthalpy")


def test_computed_descriptors_need_nothing_from_the_file():
    """Their purpose: a file carrying no numeric tags still provides descriptors."""
    mols, y = _series()
    for m in mols:
        for prop in list(m.GetPropNames()):
            m.ClearProp(prop)                     # remove every property from the file
    names = [f"m{i}" for i in range(len(mols))]
    fs = FieldSet.create(physchem=["steric"], alpha=0.5,
                         descriptors=("calc:mw", "calc:clogp"))
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    assert table.descriptors.shape == (len(mols), 2)
    assert np.isfinite(table.descriptors).all()
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.01})
    model.fit(table, y, cv_folds=0)
    parts = dict(model.descriptor_contributions(mf[0]))
    assert set(parts) == {"calc:mw", "calc:clogp"}


def test_any_rdkit_descriptor_can_be_named():
    from rdkit.Chem import Descriptors
    from cmfqsar.descriptors import compute, rdkit_descriptors

    table = rdkit_descriptors()
    assert len(table) > 150, "RDKit should expose a couple of hundred descriptors"
    m = _mol("CCOc1ccc(CCN)cc1", "x")
    for name in ("NumHAcceptors", "FractionCSP3", "BalabanJ", "RingCount"):
        assert compute(m, f"calc:{name}") == pytest.approx(float(table[name](m)))
    # RDKit's names are CamelCase; asking in another case still works
    assert compute(m, "calc:numhacceptors") == compute(m, "calc:NumHAcceptors")


def test_a_mistyped_descriptor_suggests_the_right_one():
    from cmfqsar.descriptors import compute
    with pytest.raises(KeyError, match="did you mean NumHDonors"):
        compute(_mol("CC", "x"), "calc:NumHDonor")


def test_registry_descriptors_reach_a_model():
    mols, y = _series()
    names = [f"m{i}" for i in range(len(mols))]
    fs = FieldSet.create(physchem=["steric"], alpha=0.5,
                         descriptors=("calc:NumHAcceptors", "calc:FractionCSP3"))
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    assert np.isfinite(table.descriptors).all()
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.01})
    model.fit(table, y, cv_folds=0)
    assert len(model.descriptor_contributions(mf[0])) == 2



def test_copying_a_field_set_keeps_the_descriptors():
    """The window copies the field set on its way to the model; a copy that dropped the
    descriptors built a model without them and said nothing about it."""
    import dataclasses

    fs = FieldSet.create(physchem=["steric"], descriptors=["logP", "calc:mw"], descriptor_h=0.3,
                         probes=["water"], probe_terms=5)
    fs.prop_missing = "zero"
    copy = fs.copy()
    for f in dataclasses.fields(FieldSet):
        if f.name == "specs":
            continue
        assert getattr(copy, f.name) == getattr(fs, f.name), f"copy() lost {f.name}"
    assert [s.name for s in copy.specs] == [s.name for s in fs.specs]
    assert copy.specs[0] is not fs.specs[0]      # retuning the copy must not retune the original
    copy.specs[0].alpha = 0.123
    assert fs.specs[0].alpha != 0.123


def test_descriptors_survive_the_copy_into_a_model(tmp_path):
    """The whole path a descriptor takes in the window: chosen, copied, read, fitted."""
    mols = [_mol("CCO", "a", logP=-0.3), _mol("CCCO", "b", logP=0.2), _mol("CCCCO", "c", logP=0.8)]
    y = np.array([1.0, 2.0, 3.0])
    fs = FieldSet.create(physchem=["steric"], descriptors=["logP"]).copy()
    mf = [build_fields(m, fs, m.GetProp("_Name")) for m in mols]
    assert all(m.descriptors is not None for m in mf)
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 1e-6})
    model.fit(table, y, cv_folds=0)
    assert [n for n, _ in model.descriptor_contributions(mf[0])] == ["logP"]
