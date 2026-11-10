"""The frequency run of cmfqsar.orca: the parsers, the band picker and the input (plan item SP2).

The fixture is the tail of a real ORCA 6.1 output for formic acid at r2SCAN-3c (optimisation
and frequencies), cut to the blocks the parsers read.
"""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar import orca

FORMIC = """\
---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  C      0.043513    0.425587   -0.000000
  O      1.109690    0.979798    0.000000
  O     -1.154344    1.046907   -0.000000
  H     -0.108387   -0.665701    0.000000
  H     -0.980472    2.003408    0.000000

-----------------------
VIBRATIONAL FREQUENCIES
-----------------------

Scaling factor for frequencies =  1.000000000  (already applied!)

     0:       0.00 cm**-1
     1:       0.00 cm**-1
     2:       0.00 cm**-1
     3:       0.00 cm**-1
     4:       0.00 cm**-1
     5:       0.00 cm**-1
     6:     631.88 cm**-1
     7:     685.75 cm**-1
     8:    1035.07 cm**-1
     9:    1112.38 cm**-1
    10:    1303.44 cm**-1
    11:    1401.13 cm**-1
    12:    1809.32 cm**-1
    13:    3030.12 cm**-1
    14:    3720.89 cm**-1


------------
NORMAL MODES
------------

These modes are the Cartesian displacements weighted by the diagonal matrix
M(i,i)=1/sqrt(m[i]) where m[i] is the mass of the displaced atom
Thus, these vectors are normalized but *not* orthogonal

                  0          1          2          3          4          5    
      0       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
      1       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
      2       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
      3       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
      4       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
      5       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
      6       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
      7       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
      8       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
      9       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
     10       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
     11       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
     12       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
     13       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
     14       0.000000   0.000000   0.000000   0.000000   0.000000   0.000000
                  6          7          8          9         10         11    
      0       0.043434   0.000000   0.000000   0.174177  -0.124451   0.010816
      1       0.231267   0.000000  -0.000000  -0.215905   0.172227  -0.016100
      2      -0.000000   0.090067  -0.212075  -0.000000  -0.000000  -0.000000
      3       0.244399   0.000000   0.000000   0.099153   0.034048   0.057088
      4      -0.135516  -0.000000  -0.000000   0.050821  -0.039153   0.052997
      5       0.000000  -0.045804   0.055467   0.000000   0.000000   0.000000
      6      -0.218606  -0.000000  -0.000000  -0.304967  -0.017255  -0.026006
      7      -0.056800  -0.000000   0.000000   0.129193  -0.084477  -0.041358
      8       0.000000  -0.094653   0.033783   0.000000  -0.000000   0.000000
      9      -0.060599  -0.000000  -0.000000   0.472892   0.339360  -0.935547
     10       0.241314   0.000000   0.000000  -0.244232   0.122853   0.097675
     11      -0.000000   0.182762   0.964042  -0.000000   0.000000  -0.000000
     12      -0.866338  -0.000001  -0.000000   0.718358   0.877030   0.313334
     13       0.055422  -0.000000   0.000000  -0.040293  -0.212798  -0.090566
     14      -0.000002   0.973360   0.146389  -0.000000  -0.000000  -0.000000
                 12         13         14    
      0      -0.558296   0.015270   0.000687
      1      -0.263716   0.085935  -0.003987
      2      -0.000000  -0.000000   0.000000
      3       0.370049  -0.003481  -0.000817
      4       0.197251  -0.002401   0.000102
      5       0.000000  -0.000000   0.000000
      6       0.055533  -0.001190  -0.009487
      7       0.022841  -0.000905  -0.060245
      8      -0.000000  -0.000000  -0.000000
      9       0.286722  -0.120511   0.003637
     10      -0.443912  -0.988627   0.015704
     11      -0.000000   0.000000  -0.000000
     12      -0.389103   0.012700   0.151728
     13       0.092960   0.017141   0.986399
     14      -0.000000  -0.000000   0.000000


-----------
IR SPECTRUM
-----------

 Mode   freq       eps      Int      T**2         TX        TY        TZ
       cm**-1   L/(mol*cm) km/mol    a.u.
----------------------------------------------------------------------------
  6:    631.88   0.007641   38.61  0.003774  (-0.032737  0.051980 -0.000000)
  7:    685.75   0.025821  130.49  0.011750  ( 0.000000  0.000000  0.108398)
  8:   1035.07   0.000736    3.72  0.000222  ( 0.000000  0.000000  0.014894)
  9:   1112.38   0.049963  252.49  0.014016  ( 0.103254 -0.057922 -0.000000)
 10:   1303.44   0.001040    5.26  0.000249  (-0.006740  0.014270 -0.000000)
 11:   1401.13   0.000353    1.78  0.000079  (-0.000713 -0.008839 -0.000000)
 12:   1809.32   0.067327  340.24  0.011612  (-0.107013 -0.012668  0.000000)
 13:   3030.12   0.007935   40.10  0.000817  ( 0.007130  0.027683  0.000000)
 14:   3720.89   0.010746   54.30  0.000901  (-0.005857  0.029444  0.000000)

* The epsilon (eps) is given for a Dirac delta lineshape.

****ORCA TERMINATED NORMALLY****
"""


@pytest.fixture(scope="module")
def result():
    return orca.parse_orca_output(FORMIC)


def test_frequencies_are_read_in_order(result):
    assert result.terminated
    assert result.frequencies is not None and len(result.frequencies) == 15
    assert np.all(result.frequencies[:6] == 0.0)
    assert result.frequencies[12] == pytest.approx(1809.32)
    assert result.frequencies[14] == pytest.approx(3720.89)


def test_normal_modes_have_one_column_per_frequency(result):
    assert result.modes is not None
    assert result.modes.shape == (15, 15)
    # the six zero modes are empty columns; the vibrations are unit vectors (ORCA's own
    # normalisation, as the block's header says)
    assert np.all(result.modes[:, :6] == 0.0)
    norms = np.linalg.norm(result.modes[:, 6:], axis=0)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_ir_intensities_follow_the_mode_numbers(result):
    assert result.ir_intensities is not None
    assert result.ir_intensities[12] == pytest.approx(340.24)
    assert result.ir_intensities[6] == pytest.approx(38.61)
    assert np.all(result.ir_intensities[:6] == 0.0)


def test_coordinates_are_the_last_block(result):
    assert result.symbols == ["C", "O", "O", "H", "H"]
    assert result.coords.shape == (5, 3)


def test_the_carbonyl_stretch_is_picked_by_motion(result):
    mode, nu, inten, share = orca.bond_stretch_mode(result, 0, 1, 1600.0, 1900.0)
    assert mode == 12 and nu == pytest.approx(1809.32) and inten == pytest.approx(340.24)
    assert 0.3 < share <= 1.0
    # without a window the same bond's largest stretch is still the carbonyl mode
    assert orca.bond_stretch_mode(result, 0, 1)[0] == 12
    # the hydroxyl bond's largest stretch is the O-H mode at 3721
    assert orca.bond_stretch_mode(result, 2, 4)[0] == 14


def test_an_empty_window_is_refused(result):
    with pytest.raises(orca.OrcaError):
        orca.bond_stretch_mode(result, 0, 1, 2000.0, 2500.0)


def test_a_result_without_modes_is_refused():
    r = orca.parse_orca_output("ORCA TERMINATED NORMALLY")
    assert r.frequencies is None and r.modes is None
    with pytest.raises(orca.OrcaError):
        orca.bond_stretch_mode(r, 0, 1)


def test_frequency_scale_is_by_the_first_keyword():
    assert orca.frequency_scale("r2SCAN-3c TightSCF") == pytest.approx(0.97)
    assert orca.frequency_scale("B3LYP D3BJ def2-SVP") == pytest.approx(0.96)
    assert orca.frequency_scale("wB97X-D4 def2-TZVP") == 1.0


def test_frequency_input_asks_for_optimisation_and_frequencies():
    text = orca.make_frequency_input("r2SCAN-3c TightSCF", 0, 1, "geom.xyz")
    assert text.splitlines()[0] == "! r2SCAN-3c TightSCF Opt Freq"
    assert "* xyzfile 0 1 geom.xyz" in text
    assert "UKS" not in text
    only = orca.make_frequency_input("r2SCAN-3c", 0, 2, "g.xyz", opt=False)
    assert only.splitlines()[0] == "! r2SCAN-3c Freq UKS"


def test_bond_from_smarts_names_the_carbonyl_of_an_acid():
    m = Chem.AddHs(Chem.MolFromSmiles("OC(=O)c1ccccc1"))
    i, j = orca.bond_from_smarts(m, "[CX3](=O)[OX2H1]")
    assert m.GetAtomWithIdx(i).GetSymbol() == "C"
    assert m.GetAtomWithIdx(j).GetSymbol() == "O"
    assert m.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() == 2.0
    assert orca.bond_from_smarts(Chem.MolFromSmiles("c1ccccc1"), "[CX3](=O)[OX2H1]") is None
    with pytest.raises(ValueError):
        orca.bond_from_smarts(m, "[C]")


def test_a_step_along_a_mode_moves_the_largest_atom_by_the_step():
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mode = np.array([0.0, 0.0, 0.1, 0.0, 0.0, -0.4, 0.0, 0.0, 0.2])
    moved = orca.displaced_along_mode(coords, mode, step=0.2)
    shift = moved - coords
    assert np.max(np.linalg.norm(shift, axis=1)) == pytest.approx(0.2)
    assert np.allclose(shift[:, 2], [0.05, -0.2, 0.1]) and np.allclose(shift[:, :2], 0.0)
    with pytest.raises(ValueError):
        orca.displaced_along_mode(coords, mode[:6])
    with pytest.raises(ValueError):
        orca.displaced_along_mode(coords, np.zeros(9))


def test_a_molecule_without_an_imaginary_frequency_is_not_retried():
    oc = orca.BandOutcome(index=1, name="x", mol=None, n_imaginary=1)
    assert orca.retry_band(oc, orca.Config(), (0, 1)) is oc      # no result to start from
    ok = orca.BandOutcome(index=1, name="x", mol=object(), n_imaginary=0)
    assert orca.retry_band(ok, orca.Config(), (0, 1)) is ok


def test_a_continued_optimisation_asks_for_a_computed_hessian_and_more_cycles():
    plain = orca.make_frequency_input("r2SCAN-3c TightSCF", 0, 1, "geom.xyz")
    assert "%geom" not in plain                      # the default input does not change
    more = orca.make_frequency_input("r2SCAN-3c TightSCF", 0, 1, "freq.xyz", calc_hess=True,
                                     max_cycles=200)
    assert "%geom Calc_Hess true MaxIter 200 end" in more.splitlines()
    assert more.splitlines()[-1] == "* xyzfile 0 1 freq.xyz"
    only = orca.make_frequency_input("r2SCAN-3c TightSCF", 0, 1, "geom.xyz", max_cycles=90)
    assert "%geom MaxIter 90 end" in only.splitlines()
    single = orca.make_frequency_input("r2SCAN-3c TightSCF", 0, 1, "geom.xyz", opt=False,
                                       calc_hess=True)
    assert "%geom" not in single                     # nothing to optimise


def test_the_cycle_limit_is_recognised_only_for_the_same_input(tmp_path):
    inp = orca.make_frequency_input("r2SCAN-3c TightSCF", 0, 1, "geom.xyz")
    d = str(tmp_path)
    assert not orca.optimisation_hit_cycle_limit(d, "freq", inp)        # nothing was run
    (tmp_path / "freq.inp").write_text(inp)
    (tmp_path / "freq.xyz").write_text("1\n\nH 0 0 0\n")
    (tmp_path / "freq.out").write_text(
        "                                   ERROR !!!\n"
        "       The optimization did not converge but reached the maximum \n"
        "       number of optimization cycles.\n"
        "                             ****ORCA TERMINATED NORMALLY****\n")
    assert orca.optimisation_hit_cycle_limit(d, "freq", inp)
    assert not orca.optimisation_hit_cycle_limit(d, "freq", inp + "! another job\n")
    (tmp_path / "freq.out").write_text("SCF NOT CONVERGED AFTER 300 CYCLES\n")
    assert not orca.optimisation_hit_cycle_limit(d, "freq", inp)        # another failure


def test_bond_length_ratios_are_near_one_for_a_whole_molecule():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    ratios = orca.bond_length_ratios(mol)
    assert len(ratios) == mol.GetNumBonds()
    assert 0.75 < ratios.min() and ratios.max() < 1.1
    assert orca.changed_bonds(mol) == []


def test_a_broken_bond_is_named_with_its_ratio():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    coords = mol.GetConformer().GetPositions().copy()
    h = next(a.GetIdx() for a in mol.GetAtoms()
             if a.GetSymbol() == "H" and a.GetNeighbors()[0].GetSymbol() == "O")
    o = mol.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
    coords[h] = coords[o] + 3.0 * (coords[h] - coords[o])          # the O-H bond, three times as long
    wrong = orca.changed_bonds(mol, coords)
    assert len(wrong) == 1
    assert wrong[0].startswith(f"O{o + 1}-H{h + 1} ") or wrong[0].startswith(f"H{h + 1}-O{o + 1} ")
    assert float(wrong[0].split()[-1]) > 2.5
    assert orca.changed_bonds(mol) == []                           # the conformer itself is whole
