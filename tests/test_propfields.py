"""Tests for fields whose atomic weights are read from a property of the molecule."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import (AtomTable, FieldSet, MissingAtomProperty,  # noqa: E402
                            atom_property, atom_property_names, build_fields)
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

SMILES = ["CCOc1ccccc1N", "CCCOc1ccccc1C(=O)NCC", "NCCc1ccc(O)cc1", "CC(C)COc1ccccc1C(=O)NC",
          "CCOc1ccc(C)cc1CC(=O)N"]


def _mol(smiles, seed=1, name="m", charges=True):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    m.SetProp("_Name", name)
    if charges:
        AllChem.ComputeGasteigerCharges(m)
        for a in m.GetAtoms():
            q = a.GetDoubleProp("_GasteigerCharge")
            a.SetDoubleProp("q", 0.0 if q != q else float(q))
    return m


def test_both_conventions_are_read(tmp_path):
    from rdkit import Chem
    m = _mol(SMILES[0])
    n = m.GetNumAtoms()
    want = np.array([m.GetAtomWithIdx(i).GetDoubleProp("q") for i in range(n)])
    assert np.allclose(atom_property(m, "q"), want)

    m.SetProp("fukui", " ".join(f"{0.1 * i:.4f}" for i in range(n)))
    assert np.allclose(atom_property(m, "fukui"), 0.1 * np.arange(n))
    m.SetProp("commas", ",".join(f"{0.2 * i:.4f}" for i in range(n)))
    assert np.allclose(atom_property(m, "commas"), 0.2 * np.arange(n))
    assert set(atom_property_names(m)) >= {"q", "fukui", "commas"}

    # and the properties survive a round trip through an SDF, which is the point of them
    Chem.CreateAtomDoublePropertyList(m, "q")
    path = str(tmp_path / "x.sdf")
    w = Chem.SDWriter(path)
    w.write(m)
    w.close()
    back = Chem.SDMolSupplier(path, removeHs=False)[0]
    assert np.allclose(atom_property(back, "q"), want, atol=1e-6)


def test_values_for_the_heavy_atoms_alone_land_on_them():
    m = _mol(SMILES[0])
    heavy = [i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).GetAtomicNum() > 1]
    m.SetProp("heavy_only", " ".join(f"{1.0 + k:.1f}" for k in range(len(heavy))))
    got = atom_property(m, "heavy_only")
    assert len(got) == m.GetNumAtoms()
    assert np.allclose(got[heavy], 1.0 + np.arange(len(heavy)))
    assert np.allclose(np.delete(got, heavy), 0.0)          # hydrogens left at zero


def test_a_wrong_length_is_an_error_that_names_the_molecule():
    m = _mol(SMILES[0], name="probe")
    m.SetProp("short", "1.0 2.0 3.0")
    with pytest.raises(MissingAtomProperty, match="probe"):
        atom_property(m, "short")
    with pytest.raises(MissingAtomProperty, match="no per-atom property"):
        atom_property(m, "absent")
    assert np.allclose(atom_property(m, "absent", policy="zero"), 0.0)
    m.SetProp("words", "alpha beta gamma")
    with pytest.raises(MissingAtomProperty, match="not a list of numbers"):
        atom_property(m, "words")


def test_a_property_field_of_the_charges_is_the_electrostatic_field():
    """The strong test: same numbers in, same field and same kernel out, to the last bit."""
    mols = [_mol(s, seed=i + 1, name=f"m{i}") for i, s in enumerate(SMILES)]
    elec = FieldSet.create(physchem=["electrostatic"], alpha=0.5)
    prop = FieldSet.create(physchem=[], atom_properties=["q"], alpha=0.5)
    a = [build_fields(m, elec, m.GetProp("_Name")) for m in mols]
    b = [build_fields(m, prop, m.GetProp("_Name")) for m in mols]
    for x, y in zip(a, b):
        assert np.array_equal(x.weights[:, 0], y.weights[:, 0])
    ka = kernel_matrices(AtomTable.from_molecules(a), AtomTable.from_molecules(a), elec.specs)
    kb = kernel_matrices(AtomTable.from_molecules(b), AtomTable.from_molecules(b), prop.specs)
    assert np.allclose(ka, kb, atol=0, rtol=0)
    assert prop.specs[0].is_prop and prop.specs[0].prop == "q"
    assert prop.specs[0].label == "q" and not prop.specs[0].is_cif


def test_a_model_on_property_fields_fits_saves_and_explains(tmp_path):
    from cmfqsar.dataio import load_model, save_model
    mols = [_mol(s, seed=i + 1, name=f"m{i}") for i, s in enumerate(SMILES)]
    fs = FieldSet.create(physchem=["steric"], atom_properties=["q"], alpha=0.5)
    mf = [build_fields(m, fs, m.GetProp("_Name")) for m in mols]
    y = np.array([5.0, 5.7, 6.2, 6.6, 7.1])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), y, cv_folds=0)
    pred = model.predict(AtomTable.from_molecules(mf[:2]))
    assert np.all(np.isfinite(pred))

    # the interpretation machinery does not care where the weights came from
    contrib = model.atom_contributions(mf[0])
    assert contrib.shape == (mf[0].n_atoms,)
    assert float(contrib.sum() + model.bias) == pytest.approx(float(model.predict(
        AtomTable.from_molecules([mf[0]]))[0]), abs=1e-8)
    centres, coef, widths = model.cofield_coefficients(1)       # the property field
    assert len(centres) == len(coef) == len(widths)

    path = str(tmp_path / "prop.cmf")
    save_model(path, model)
    back = load_model(path)
    back = back[0] if isinstance(back, tuple) else back
    assert back.fieldset.specs[1].prop == "q"
    assert np.allclose(back.predict(AtomTable.from_molecules(mf[:2])), pred)


def test_a_molecule_without_the_property_is_refused_or_zeroed():
    good = _mol(SMILES[0], name="has_it")
    bare = _mol(SMILES[1], name="has_none", charges=False)
    fs = FieldSet.create(physchem=[], atom_properties=["q"], alpha=0.5)
    with pytest.raises(MissingAtomProperty, match="has_none"):
        build_fields(bare, fs, "has_none")
    lenient = FieldSet.create(physchem=[], atom_properties=["q"], alpha=0.5, prop_missing="zero")
    assert float(np.abs(build_fields(bare, lenient, "has_none").weights).max()) == 0.0
    assert float(np.abs(build_fields(good, lenient, "has_it").weights).max()) > 0.0
