"""The reader that turns a relaxed ion-ether complex into a molecule (plan item EC7)."""
import io
import os
import pickle
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import make_ether_examples as M  # noqa: E402


def _complex(smiles="COCCOC", seed=3, ion_at=(0.0, 0.0, 0.0)):
    """An ether with its coordinates and a lithium put at `ion_at`, as the distribution has it."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(mol, p)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.SanitizeMol(mol)
    coords = mol.GetConformer().GetPositions()
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], int)
    # the ion first, as the geometry file has it
    numbers = np.concatenate([[3], numbers])
    coords = np.vstack([np.asarray(ion_at, float), coords])
    return smiles, numbers, coords, mol


def test_the_ion_is_removed_and_its_position_kept():
    smiles, numbers, coords, mol = _complex(ion_at=(1.5, -2.0, 0.5))
    out, ion = M.ether_in_complex_geometry(smiles, numbers, coords)
    assert out.GetNumAtoms() == mol.GetNumAtoms(), "the ether keeps every atom but the ion"
    assert not any(a.GetSymbol() == "Li" for a in out.GetAtoms())
    assert np.allclose(ion, [1.5, -2.0, 0.5])


def test_the_atom_order_is_the_order_of_the_smiles():
    """So that a property written per atom belongs to the atom the SMILES names."""
    smiles, numbers, coords, mol = _complex()
    out, _ion = M.ether_in_complex_geometry(smiles, numbers, coords)
    assert ([a.GetSymbol() for a in out.GetAtoms()]
            == [a.GetSymbol() for a in mol.GetAtoms()])
    # the coordinates are the ones given, up to the matching of identical atoms
    got = out.GetConformer().GetPositions()
    want = coords[1:]
    assert sorted(np.round(got.sum(axis=1), 6).tolist()) == \
        sorted(np.round(want.sum(axis=1), 6).tolist())


def test_a_geometry_that_does_not_match_the_smiles_is_refused():
    smiles, numbers, coords, _mol = _complex()
    with pytest.raises(ValueError, match="atoms"):
        M.ether_in_complex_geometry("COCCOCCOC", numbers, coords)


def test_a_complex_without_exactly_one_ion_is_refused():
    smiles, numbers, coords, _mol = _complex()
    numbers = numbers.copy()
    numbers[0] = 6                                    # the lithium replaced by a carbon
    with pytest.raises(ValueError, match="lithium"):
        M.ether_in_complex_geometry(smiles, numbers, coords)


def test_the_geometry_reader_admits_numpy_and_refuses_everything_else():
    """The distributed geometries are a pickle, and a pickle runs what it names."""
    good = pickle.dumps({"a.[Li+]": {"ebind": -1.0, "xyz": np.zeros((2, 4))}})
    out = M._ArraysOnly(io.BytesIO(good)).load()
    assert out["a.[Li+]"]["xyz"].shape == (2, 4)
    bad = pickle.dumps({"x": os.getcwd})
    with pytest.raises(pickle.UnpicklingError, match="refuses"):
        M._ArraysOnly(io.BytesIO(bad)).load()


def _bench():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    import run_ether as E
    return E


def _placed(smiles="COCCOCCOC", seed=5):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(mol, p)
    return mol


def test_contact_features_count_the_oxygens():
    E = _bench()
    mol = _placed()
    pos = mol.GetConformer().GetPositions()
    oxygens = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O"]
    # an ion put 2.0 A from the first oxygen, along a direction away from the molecule's centre
    out = pos[oxygens[0]] - pos.mean(axis=0)
    ion = pos[oxygens[0]] + 2.0 * out / np.linalg.norm(out)
    n, mean_d, inv = E.contact_features(mol, ion)
    d = np.array([np.linalg.norm(pos[i] - ion) for i in oxygens])
    assert n == float(np.sum(d < E.CONTACT_CUT)) and n >= 1
    assert np.isclose(mean_d, d[d < E.CONTACT_CUT].mean())
    assert np.isclose(inv, np.sum(1.0 / d[d < E.CONTACT_REACH]))
    # far from every oxygen: no contact, and the mean distance falls back to the nearest oxygen
    far = pos.mean(axis=0) + np.array([50.0, 0.0, 0.0])
    n0, m0, i0 = E.contact_features(mol, far)
    assert n0 == 0.0 and i0 == 0.0
    assert np.isclose(m0, min(np.linalg.norm(pos[i] - far) for i in oxygens))


def test_gaussian_nested_fits_a_smooth_function_and_reports_paired_errors():
    E = _bench()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1] ** 2 + rng.normal(scale=0.05, size=120)
    folds = E.random_folds(len(y))
    rec = E.gaussian_nested(X, y, folds)
    assert rec["n"] == 120 and len(rec["abs_errors"]) == 120
    assert rec["r2_pooled"] > 0.8                       # a Gaussian kernel follows a smooth nonlinear target
    assert all(f["gamma"] in E.GAMMAS and f["lam"] in E.LAMS for f in rec["per_fold"])
    lin = E.nested("counts", {}, (), y, folds, X)
    assert rec["mae_mean"] < lin["mae_mean"]            # and does better than the linear kernel on it


def test_contacts_section_reads_a_result():
    E = _bench()
    rec = {"mae_mean": 0.2, "mae_sd": 0.01, "r2_pooled": 0.5}
    stored = {b: {"mae_mean": 0.1, "r2_pooled": 0.9} for b in ("counts", "fields", "probe", "fields+probe")}
    iv = {"mean": 0.05, "low": 0.04, "high": 0.06, "excludes_zero": True}
    design = {"models": {m: rec for m in E.CONTACT_MODELS}, "stored": stored,
              "paired": {f"{m} minus {b}": iv for m in E.CONTACT_MODELS for b in ("counts", "fields", "probe", "fields+probe")}}
    c = {"cut": 2.5, "reach": 3.5, "models": list(E.CONTACT_MODELS), "agreement": {"same_count": 0.9},
         "ions": {ion: {"n": 10, "designs": {"random": design, "one oxygen count held out": design}} for ion in E.ION_ORDER}}
    text = "\n".join(E.contacts_section(c))
    assert "## The oxygens at the ion as a reference" in text and "90%" in text
    assert "contacts at the probe minimum, Gaussian minus fields: +0.050 eV [+0.040, +0.060], excludes zero." in text
    assert text.count("### ") == 6

