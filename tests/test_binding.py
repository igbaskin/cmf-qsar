"""Plan item PP6, the parts of the binding measurement that need no download and no fit.

* SKEMPI's PDB-numbered substitutions and temperatures are read as the table writes them.
* Repeated measurements of one substitution are averaged into one sample.
* Complexes that SKEMPI lists as related, or that share a protein name, form one family.
* A site with no partner atom gets a raw kernel of zero and a cosine kernel that sets it
  apart from every site that has partner atoms.
* Proximity bonding across two residues is undone where there is no bond, and a terminal
  oxygen on top of the carbonyl oxygen is moved to where a carboxylate puts it.
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "benchmarks")):
    sys.path.insert(0, p)

pytest.importorskip("rdkit")

import run_binding as R                                                 # noqa: E402
from cmfqsar import mutants as MU                                       # noqa: E402
from cmfqsar.fields import MoleculeFields                               # noqa: E402


def test_the_pdb_numbered_form_and_the_temperature_are_read():
    assert R.parse_mutation("LI45G") == ("L", "I", "45", "G")
    assert R.parse_mutation("RD100bA") == ("R", "D", "100B", "A")
    assert R.parse_mutation("KA-3E") == ("K", "A", "-3", "E")
    with pytest.raises(ValueError):
        R.parse_mutation("LI45G,KA3E")
    assert R.parse_temperature("298(assumed)") == 298.0
    assert R.parse_temperature("310") == 310.0
    assert R.parse_temperature("") == R.DEFAULT_T


def _row(complex_, resid, new, ddg, related="", proteins=("A", "B")):
    return {"complex": complex_, "chain": "A", "resid": resid, "wild": "L", "new": new,
            "ddg": ddg, "location": "COR", "related": related, "proteins": proteins}


def test_repeated_measurements_are_averaged():
    rows = [_row("1ABC_A_B", "45", "G", 1.0), _row("1ABC_A_B", "45", "G", 2.0),
            _row("1ABC_A_B", "45", "A", 0.5)]
    out = R.substitutions(rows)
    assert len(out) == 2
    g = [r for r in out if r["new"] == "G"][0]
    assert g["ddg"] == pytest.approx(1.5) and g["measurements"] == 2


def test_related_complexes_and_shared_names_form_one_family():
    rows = [_row("1AAA_A_B", "1", "G", 0.0, related="1BBB_A_B", proteins=("x", "y")),
            _row("1BBB_A_B", "1", "G", 0.0, proteins=("p", "q")),
            _row("1CCC_A_B", "1", "G", 0.0, proteins=("Y ", "r")),
            _row("1DDD_A_B", "1", "G", 0.0, related="Pr/PI", proteins=("s", "t"))]
    fam = R.families(rows)
    assert fam["1AAA_A_B"] == fam["1BBB_A_B"] == fam["1CCC_A_B"]
    assert fam["1DDD_A_B"] == "1DDD_A_B"         # a class label such as Pr/PI joins nothing


def test_the_binding_energy_has_the_sign_of_weaker_binding():
    # ten times weaker binding at 298 K is +1.36 kcal/mol
    assert R.R_GAS * 298.0 * math.log(10.0) == pytest.approx(1.364, abs=0.001)


def _field(n, shift=0.0):
    rng = np.random.RandomState(n)
    return MoleculeFields(name=str(n), coords=rng.normal(size=(n, 3)) + shift,
                          weights=np.ones((n, 5)), radii=np.full(n, 1.5),
                          symbols=["C"] * n, atom_types=["C.3"] * n,
                          bonds=np.zeros((0, 2), int), charges=np.zeros(n),
                          charge_source="file", member_atom_map=np.arange(n))


def test_a_site_without_partner_atoms_is_set_apart():
    empty = MU._subset(_field(3), np.zeros(0, int))
    fields = [_field(4), empty, _field(5), empty]
    cos, raw = R.context_kernels(fields)
    assert raw[1].tolist() == [0.0] * 4 and raw[3].tolist() == [0.0] * 4
    assert cos[1, 3] == 1.0 and cos[1, 1] == 1.0
    assert cos[1, 0] == 0.0 and cos[3, 2] == 0.0
    assert cos[0, 0] == pytest.approx(1.0) and cos[2, 2] == pytest.approx(1.0)
    assert np.mean([raw[0, 0], raw[2, 2]]) == pytest.approx(1.0)


PDB_TWO_RESIDUES = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY B   1       0.300   2.900   0.000  1.00  0.00           N
ATOM      6  CA  GLY B   1      -0.500   4.100   0.000  1.00  0.00           C
END
"""


def test_a_contact_between_two_chains_is_not_a_bond():
    from rdkit import Chem
    mol = Chem.MolFromPDBBlock(PDB_TWO_RESIDUES, removeHs=False, sanitize=False,
                               proximityBonding=True)
    rw = Chem.RWMol(mol)
    before = rw.GetBondBetweenAtoms(3, 4)
    assert before is not None                    # O of chain A and N of chain B, 1.1 A apart
    assert MU._drop_contact_bonds(rw) == 1
    assert rw.GetBondBetweenAtoms(3, 4) is None
    assert rw.GetBondBetweenAtoms(1, 2) is not None


def test_only_the_bonds_of_the_topology_are_kept(tmp_path):
    from rdkit import Chem
    mol = Chem.MolFromPDBBlock(PDB_TWO_RESIDUES, removeHs=False, sanitize=False,
                               proximityBonding=True)
    rw = Chem.RWMol(mol)
    assert MU._keep_topology_bonds(rw, np.array([[0, 1], [1, 2], [2, 3]])) >= 1
    kept = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in rw.GetBonds()}
    assert kept == {(0, 1), (1, 2), (2, 3)}
    st = MU.Structure("x", PDB_TWO_RESIDUES, np.zeros(6), bonds=np.array([[0, 1], [2, 3]]))
    back = MU.Structure.load(st.save(str(tmp_path / "x.npz")))
    assert back.bonds.tolist() == [[0, 1], [2, 3]]
    old = MU.Structure("y", PDB_TWO_RESIDUES, np.zeros(6))
    assert MU.Structure.load(old.save(str(tmp_path / "y.npz"))).bonds is None


def test_a_terminal_oxygen_on_the_carbonyl_oxygen_is_moved():
    pytest.importorskip("openmm")
    from openmm import app, unit
    from openmm import openmm as mm
    top = app.Topology()
    chain = top.addChain("A")
    res = top.addResidue("GLY", chain)
    xyz = {"N": (0.0, 0.0, 0.0), "CA": (1.458, 0.0, 0.0), "C": (2.009, 1.420, 0.0),
           "O": (1.251, 2.390, 0.0), "OXT": (1.40, 2.10, 0.0)}
    for name in xyz:
        top.addAtom(name, app.element.get_by_symbol(name[0]), res)
    pos = [mm.Vec3(*(np.array(v) * 0.1)) for v in xyz.values()] * unit.nanometer
    new = np.array(MU._repair_terminal_oxygens(top, pos, unit, mm).value_in_unit(unit.angstrom))
    c, o, oxt = new[2], new[3], new[4]
    assert np.linalg.norm(oxt - c) == pytest.approx(1.25, abs=1e-6)
    assert np.linalg.norm(oxt - o) > 2.0
    ca = new[1]
    angle = lambda a, b: np.degrees(np.arccos((a @ b) / np.linalg.norm(a) / np.linalg.norm(b)))
    assert angle(ca - c, oxt - c) == pytest.approx(angle(ca - c, o - c), abs=1e-6)
