"""Solvent exposure as a weight on the fields: the factor, the scaling, and what it leaves alone."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar.exposure import (atom_exposure, buried_fraction, exposed_fields, exposed_set,
                              weight_by_exposure)
from cmfqsar.fields import AtomTable, FieldSet, build_fields


def _mol(smiles="CC(=O)Nc1ccc(O)cc1", seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def test_exposure_is_a_fraction_and_the_inside_is_less_exposed_than_the_outside():
    m = _mol()
    e = atom_exposure(m)
    assert e.shape == (m.GetNumAtoms(),)
    assert np.all(e >= 0.0) and np.all(e <= 1.0)
    # the two oxygens stick out; the ring carbon bearing the amide does not
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    oxygens = [i for i, s in enumerate(sym) if s == "O"]
    assert min(e[i] for i in oxygens) > 0.05
    # a quaternary carbon in a crowded molecule is nearly invisible
    # a quaternary carbon carries no hydrogens, so it is its own group and is genuinely shielded
    crowded = _mol("CC(C)(C)c1ccccc1C(C)(C)C", seed=3)
    ec = atom_exposure(crowded)
    quaternary = [a.GetIdx() for a in crowded.GetAtoms()          # four heavy neighbours, no H
                  if a.GetSymbol() == "C" and a.GetTotalNumHs(includeNeighbors=True) == 0
                  and a.GetDegree() == 4]
    methyls = [a.GetIdx() for a in crowded.GetAtoms()
               if a.GetSymbol() == "C" and a.GetTotalNumHs(includeNeighbors=True) == 3]
    assert quaternary and methyls
    assert max(ec[i] for i in quaternary) < min(ec[i] for i in methyls)


def test_the_floor_and_the_power_do_what_they_say():
    m = _mol()
    plain = atom_exposure(m)
    floored = atom_exposure(m, floor=0.2)
    assert floored.min() >= 0.2 - 1e-12
    assert np.all(floored >= plain - 1e-12)
    softer = atom_exposure(m, power=0.5)
    assert np.all(softer >= plain - 1e-12)          # x^0.5 >= x on [0, 1]
    harsher = atom_exposure(m, power=2.0)
    assert np.all(harsher <= plain + 1e-12)


def test_weighting_scales_every_field_of_every_atom_and_copies():
    m = _mol()
    fs = FieldSet.create(physchem=["steric", "electrostatic", "hydrophobic"], alpha=0.5)
    base = build_fields(m, fs, "x")
    e = atom_exposure(m)
    out = weight_by_exposure(base, e)
    np.testing.assert_allclose(out.weights, base.weights * e[:, None], atol=1e-12)
    assert out.weights is not base.weights          # the original is left alone
    np.testing.assert_allclose(out.coords, base.coords, atol=1e-12)
    with pytest.raises(ValueError):
        weight_by_exposure(base, e[:-1])


def test_an_exposure_of_one_everywhere_changes_nothing():
    """The weighting must reduce to the plain field, or any comparison with it is confounded."""
    m = _mol()
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    base = build_fields(m, fs, "x")
    same = weight_by_exposure(base, np.ones(m.GetNumAtoms()))
    np.testing.assert_allclose(same.weights, base.weights, atol=1e-15)
    full = atom_exposure(m, floor=1.0)
    np.testing.assert_allclose(full, np.ones(m.GetNumAtoms()), atol=1e-12)


def test_the_kernel_of_exposure_weighted_fields_is_still_a_kernel():
    mols = [_mol(s, seed=i) for i, s in enumerate(["CCO", "CCCO", "c1ccccc1O", "CC(=O)O"])]
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    names = [f"m{i}" for i in range(len(mols))]
    table = AtomTable.from_molecules(exposed_set(mols, fs, names))
    from cmfqsar.model import CMFModel

    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = model.joint_kernel(model.prepare_training_kernels(table))
    assert K.shape == (len(mols), len(mols))
    np.testing.assert_allclose(K, K.T, rtol=1e-10)
    assert np.all(np.linalg.eigvalsh(K) > -1e-8)    # positive semi-definite
    assert np.all(np.diag(K) > 0)


def test_the_weighting_actually_changes_the_kernel():
    """If buried atoms were rare or the scaling cancelled, the measurement would be empty."""
    from cmfqsar.model import CMFModel

    mols = [_mol(s, seed=i) for i, s in enumerate(
        ["CC(C)(C)c1ccccc1C(C)(C)C", "c1ccccc1O", "CCCCCCCC", "c1ccc2ccccc2c1"])]
    names = [f"m{i}" for i in range(len(mols))]
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)

    def kernel(fields):
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        return model.joint_kernel(model.prepare_training_kernels(AtomTable.from_molecules(fields)))

    plain = kernel([build_fields(m, fs, n) for m, n in zip(mols, names)])
    exposed = kernel(exposed_set(mols, fs, names))
    off = ~np.eye(len(mols), dtype=bool)
    assert np.abs(plain[off] - exposed[off]).max() > 1e-3, "the weighting did nothing"
    # it changes the kernel by modulating exposure between about 0.2 and 0.6, not by switching
    # a buried core off: there is no buried core in molecules this size
    assert max(buried_fraction(m) for m in mols) == pytest.approx(0.0, abs=1e-9)


def test_a_small_molecule_has_no_interior_to_discount():
    """Measured by group, a ligand is all surface -- which is why little is expected of this.

    Per atom it looks otherwise, but that is a methyl carbon shielded by its own hydrogens, not
    burial, and weighting a field by it would be measuring hydrogen counts.
    """
    assert buried_fraction(_mol("CCO")) == pytest.approx(0.0, abs=1e-9)
    assert buried_fraction(_mol("CC(C)(C)c1ccccc1C(C)(C)C", seed=3)) == pytest.approx(0.0, abs=1e-9)
    assert buried_fraction(_mol("CCO"), united=False) > 0.5
    e = atom_exposure(_mol("CC(=O)Nc1ccc(O)cc1"))
    heavy = [a.GetIdx() for a in _mol("CC(=O)Nc1ccc(O)cc1").GetAtoms() if a.GetAtomicNum() > 1]
    assert 0.2 < float(np.median(e[heavy])) < 0.7
