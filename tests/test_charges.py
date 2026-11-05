"""The charge model that no element and no valence state can void (plan item D11).

The Gasteiger scheme answers "not a number" in two ways, and before this item both became a zero
charge, which is not an error and not a charge:

* an element it has no parameters for at all voids **every atom of the molecule**, so the
  molecule's electrostatic field was identically zero.  Item EC4 found this on the metal of a
  coordination complex and repaired it for the d block; item EC3 found it again on a selenoether;
* a valence state it has no parameters for voids **that atom alone**, so an ion looked neutral to
  the field.  The phosphorus of hexafluorophosphate is the case that matters here, because 717 of
  the 989 solvation shells of `examples/electrolyte/` contain one.

The tests below pin the covered element list (which was measured, not taken from a paper), both
repair paths, and the thing that must not change: a molecule the scheme can do is answered exactly
as before.
"""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar.fields import (GASTEIGER_ELEMENTS, FieldSet, _gasteiger_with_unknown_atoms,
                            _partial_charges, build_fields)


def mol(smiles: str, seed: int = 1, embed: bool = True) -> Chem.Mol:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if embed:
        AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


def plain_gasteiger(m: Chem.Mol) -> np.ndarray:
    work = Chem.Mol(m)
    AllChem.ComputeGasteigerCharges(work)
    return np.array([float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge")
                     else np.nan for a in work.GetAtoms()])


# ------------------------------------------------------------------ the covered elements

def test_the_covered_elements_are_the_fourteen_that_were_measured():
    assert GASTEIGER_ELEMENTS == {"H", "B", "C", "N", "O", "F", "Mg", "Al", "Si", "P",
                                  "S", "Cl", "Br", "I"}


def test_magnesium_is_covered_and_lithium_is_not():
    """The part of the list nobody would guess, and the reason it was measured."""
    assert "Mg" in GASTEIGER_ELEMENTS and "Al" in GASTEIGER_ELEMENTS
    assert "Li" not in GASTEIGER_ELEMENTS and "Na" not in GASTEIGER_ELEMENTS
    assert "Fe" not in GASTEIGER_ELEMENTS and "Se" not in GASTEIGER_ELEMENTS


@pytest.mark.parametrize("element", ["Li", "Se", "As", "Sn", "Zn", "Fe"])
def test_an_uncovered_element_really_does_void_the_whole_molecule(element):
    """The fault this item repairs, demonstrated on the scheme itself.

    Without this test the repair could be fixing something that no longer happens.
    """
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(6))
    atom = Chem.Atom(element)
    atom.SetNoImplicit(True)
    rw.AddAtom(atom)
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    assert not np.any(np.isfinite(plain_gasteiger(m)))


# ------------------------------------------------------------------ nothing else changes

@pytest.mark.parametrize("smiles", ["c1ccccc1N", "CC(=O)Oc1ccccc1C(=O)O", "C[Mg]Br",
                                    "CC[Si](C)(C)C", "c1ccc(S(=O)(=O)[O-])cc1"])
def test_a_molecule_the_scheme_can_do_is_answered_exactly_as_before(smiles):
    m = mol(smiles)
    q, source = _partial_charges(m, "gasteiger")
    assert source == "gasteiger"
    assert np.allclose(q, plain_gasteiger(m), atol=0.0, rtol=0.0)


def test_the_repair_declines_a_molecule_it_has_nothing_to_do_with():
    assert _gasteiger_with_unknown_atoms(mol("c1ccccc1N")) is None


# ------------------------------------------------------------------ an uncovered element

def test_a_selenoether_gets_a_field_instead_of_nothing():
    """One of the two corrosion molecules that found this fault outside the d block."""
    m = mol("C[Se]c1ccc(NC(=S)Nc2ccccc2)cc1")
    assert not np.any(np.isfinite(plain_gasteiger(m)))        # the scheme gives nothing
    q, source = _partial_charges(m, "gasteiger")
    assert source == "gasteiger+formal"
    assert np.all(np.isfinite(q))
    assert np.abs(q).sum() > 1.0
    selenium = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "Se"]
    assert q[selenium[0]] == pytest.approx(0.0)               # its formal charge, which is zero


def test_an_arsonium_cation_keeps_its_charge():
    """The cyclodextrin guest that found the same fault, and it is an ion."""
    m = mol("c1ccc([As+](c2ccccc2)(c2ccccc2)c2ccccc2)cc1")
    q, source = _partial_charges(m, "gasteiger")
    assert source == "gasteiger+formal"
    arsenic = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "As"]
    assert q[arsenic[0]] == pytest.approx(1.0)
    assert q.sum() > 0.5                                      # the cation is positive overall


def test_the_net_charge_is_about_the_formal_charge():
    """Removing a *bonded* atom leaves its neighbours an open valence, so the remaining fragment
    is charged as a radical and the sum is near the formal charge rather than exactly it.  That is
    the approximation the repair makes, and it is bounded here rather than hidden."""
    for smiles, formal in (("C[Se]c1ccccc1", 0.0),
                           ("c1ccc([As+](c2ccccc2)(c2ccccc2)c2ccccc2)cc1", 1.0)):
        q, _source = _partial_charges(mol(smiles), "gasteiger")
        assert q.sum() == pytest.approx(formal, abs=0.25)


def test_a_lone_ion_needs_no_repair_at_all():
    """The plain scheme does give a lone ion its charge; the fault needs a *bonded* atom.

    This is the distinction that cost two wrong scope statements for item D11 before it was
    measured, so it is pinned here.
    """
    q, source = _partial_charges(Chem.MolFromSmiles("[Li+]"), "gasteiger")
    assert source == "gasteiger"
    assert q == pytest.approx([1.0])
    # and the repair, asked directly, answers with the formal charges
    assert _gasteiger_with_unknown_atoms(Chem.MolFromSmiles("[Li+]")) == pytest.approx([1.0])


# ------------------------------------------------------------------ an uncovered valence state

def test_hexafluorophosphate_stops_looking_neutral():
    """The phosphorus of PF6- comes back as "not a number" although phosphorus is covered.

    Before this item that atom was set to zero, so the anion carried no charge at all and a
    neutral lithium shell presented a net charge of +1 to the electrostatic field.
    """
    m = mol("F[P-](F)(F)(F)(F)F")
    before = plain_gasteiger(m)
    phosphorus = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "P"][0]
    assert not np.isfinite(before[phosphorus])                # the atom the scheme cannot do
    q, source = _partial_charges(m, "gasteiger")
    assert source == "gasteiger+formal"
    assert q[phosphorus] == pytest.approx(-1.0)
    assert q.sum() == pytest.approx(-1.0, abs=0.3)            # the anion is an anion


def test_a_lithium_shell_with_the_anion_adds_up():
    """A neutral ion pair must present a net charge of zero to the field."""
    pair = Chem.AddHs(Chem.MolFromSmiles("[Li+].F[P-](F)(F)(F)(F)F"))
    AllChem.EmbedMolecule(pair, randomSeed=1)
    q, source = _partial_charges(pair, "gasteiger")
    assert source == "gasteiger+formal"
    assert q.sum() == pytest.approx(0.0, abs=0.3)
    symbols = [a.GetSymbol() for a in pair.GetAtoms()]
    assert q[symbols.index("Li")] == pytest.approx(1.0)
    assert q[symbols.index("P")] == pytest.approx(-1.0)


# ------------------------------------------------------------------ the field that follows

def test_the_electrostatic_field_of_a_selenoether_is_not_zero():
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0,
                         charge_model="gasteiger", param_scheme="rdkit")
    mf = build_fields(mol("C[Se]c1ccc(NC(=S)Nc2ccccc2)cc1"), fs, "selenoether")
    weights = np.asarray(mf.weights)
    assert mf.charge_source == "gasteiger+formal"
    assert np.abs(weights[:, 0]).sum() > 1.0


def test_the_metal_rule_of_item_ec4_still_holds():
    """The general rule has to subsume the special one: a metal is an uncovered element."""
    from cmfqsar.complexes import build_complex, metal_index
    from test_complexes import dichloro_tetraaqua
    symbols, coords = dichloro_tetraaqua()
    complexed, report = build_complex(symbols, coords, total_charge=0,
                                      ligand_charges=[-1, -1, 0, 0, 0, 0])
    assert report.ok
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0,
                         charge_model="gasteiger", param_scheme="rdkit")
    mf = build_fields(complexed, fs, "complex")
    weights = np.asarray(mf.weights)
    assert mf.charge_source == "gasteiger+formal"
    assert weights[metal_index(complexed), 0] == pytest.approx(2.0)
    assert weights[:, 0].sum() == pytest.approx(0.0, abs=1e-6)
