"""Tests for cmfqsar/orca.py: the ORCA parser, the derived descriptors, the input
writer, the cache key, the SDF round trip, and (when ORCA is installed) one real water run."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar import orca as oap  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import atom_property  # noqa: E402

# A trimmed ORCA 6.1 output of a closed-shell water single point: the blocks the script reads, in
# ORCA's own layout, with an earlier (stale) Mulliken block first to check that the LAST one wins.
CLOSED_SHELL = """
---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  O      0.000000    0.000000    0.117300
  H      0.000000    0.757200   -0.469200
  H      0.000000   -0.757200   -0.469200

-----------------------
MULLIKEN ATOMIC CHARGES
-----------------------
   0 O :   -0.999999
   1 H :    0.500000
   2 H :    0.499999
Sum of atomic charges:    0.0000000

----------------
ORBITAL ENERGIES
----------------

  NO   OCC          E(Eh)            E(eV)
   0   2.0000     -19.114098      -520.1211
   1   2.0000      -0.978887       -26.6369
   2   2.0000      -0.513826       -13.9819
   3   2.0000      -0.362267        -9.8578
   4   2.0000      -0.288390        -7.8475
   5   0.0000       0.049987         1.3602
   6   0.0000       0.129017         3.5107
*Only the first 10 virtual orbitals were printed.

                    ********************************
                    * MULLIKEN POPULATION ANALYSIS *
                    ********************************

-----------------------
MULLIKEN ATOMIC CHARGES
-----------------------
   0 O :   -0.288924
   1 H :    0.144462
   2 H :    0.144462
Sum of atomic charges:    0.0000000

--------------------------------
MULLIKEN REDUCED ORBITAL CHARGES
--------------------------------
  0 O s       :     3.674581  s :     3.674581
      pz      :     1.505874  p :     4.603304

--------------------------------
MULLIKEN ATOM POPULATIONS PER MO
--------------------------------
THRESHOLD FOR PRINTING IS 0.1%%
                      0         1         2         3         4         5
                 -19.11410  -0.97889  -0.51383  -0.36227  -0.28839   0.04999
                   2.00000   2.00000   2.00000   2.00000   2.00000   0.00000
                  --------  --------  --------  --------  --------  --------
  0 O               100.1      71.9      57.3      87.2      98.0      -8.1
  1 H                -0.0      14.1      21.3       6.4       1.0      54.1
  2 H                -0.0      14.1      21.3       6.4       1.0      54.1

                      6         7
                   0.12902   0.56831
                   0.00000   0.00000
                  --------  --------
  0 O                 7.9      18.7
  2 H                46.1      40.7


                     *******************************
                     * LOEWDIN POPULATION ANALYSIS *
                     *******************************

----------------------
LOEWDIN ATOMIC CHARGES
----------------------
   0 O :   -0.140942
   1 H :    0.070471
   2 H :    0.070471

------------------
HIRSHFELD ANALYSIS
------------------

Total integrated alpha density =      4.999998185
Total integrated beta density  =      4.999998185

  ATOM     CHARGE      SPIN
   0 O   -0.318890    0.000000
   1 H    0.159447    0.000000
   2 H    0.159447    0.000000

  TOTAL   0.000004    0.000000

CHELPG Charges
--------------------------------
  0   O   :      -0.720154
  1   H   :       0.359811
  2   H   :       0.360343
--------------------------------
Total charge:     0.000000
--------------------------------

-------------------------   --------------------
FINAL SINGLE POINT ENERGY       -76.321663220492
-------------------------   --------------------

Total Dipole Moment    :      0.000000000      -0.000000000      -0.779472966
                        -----------------------------------------
Magnitude (a.u.)       :      0.779472966
Magnitude (Debye)      :      1.981262827

                             ****ORCA TERMINATED NORMALLY****
"""

# the water cation (doublet): charges carry a spin column, orbitals come per spin
OPEN_SHELL = """
----------------
ORBITAL ENERGIES
----------------
                 SPIN UP ORBITALS
  NO   OCC          E(Eh)            E(eV)
   0   1.0000     -19.700000      -536.0000
   1   1.0000      -1.300000       -35.3000
   2   1.0000      -0.900000       -24.5000
   3   1.0000      -0.881855       -23.9965
   4   1.0000      -0.870042       -23.6751
   5   0.0000      -0.278459        -7.5773
*Only the first 10 virtual orbitals were printed.
                 SPIN DOWN ORBITALS
  NO   OCC          E(Eh)            E(eV)
   0   1.0000     -19.600000      -533.0000
   1   1.0000      -1.200000       -32.6000
   2   1.0000      -0.880000       -23.9000
   3   1.0000      -0.838594       -22.8000
   4   0.0000      -0.614181       -16.7000
*Only the first 10 virtual orbitals were printed.

--------------------------------------------
MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS
--------------------------------------------
   0 O :    0.407637    1.047993
   1 H :    0.296182   -0.023996
   2 H :    0.296182   -0.023996
Sum of atomic charges         :    1.0000000
Sum of atomic spin populations:    1.0000000

-------------------------------------------
LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS
-------------------------------------------
   0 O :    0.613942    0.900000
   1 H :    0.193029    0.050000
   2 H :    0.193029    0.050000

------------------
HIRSHFELD ANALYSIS
------------------

Total integrated alpha density =      4.999998185
Total integrated beta density  =      3.999999106

  ATOM     CHARGE      SPIN
   0 O    0.346280    0.938255
   1 H    0.326861    0.030872
   2 H    0.326861    0.030872

  TOTAL   1.000003    0.999999

-------------------------   --------------------
FINAL SINGLE POINT ENERGY       -75.869184386651
-------------------------   --------------------

Magnitude (Debye)      :      2.464202502

                             ****ORCA TERMINATED NORMALLY****
"""


def test_parse_closed_shell_fragment():
    r = oap.parse_orca_output(CLOSED_SHELL)
    assert r.terminated and not r.converged
    assert r.energy == pytest.approx(-76.321663220492)
    assert r.dipole == pytest.approx(1.981262827)
    # the last Mulliken block, not the stale one
    assert np.allclose(r.mulliken, [-0.288924, 0.144462, 0.144462])
    assert np.allclose(r.loewdin, [-0.140942, 0.070471, 0.070471])
    assert np.allclose(r.hirshfeld, [-0.318890, 0.159447, 0.159447])
    assert np.allclose(r.chelpg, [-0.720154, 0.359811, 0.360343])
    assert r.homo_index == 4 and r.homo == pytest.approx(-0.288390)
    assert r.lumo_index == 5 and r.lumo == pytest.approx(0.049987)
    assert np.allclose(r.mo_population(4, 3), [0.980, 0.010, 0.010])
    assert np.allclose(r.mo_population(5, 3), [-0.081, 0.541, 0.541])
    # a row ORCA left out of a chunk (atom 1 in MO 6) reads as zero, the others as given
    assert np.allclose(r.mo_population(6, 3), [0.079, 0.0, 0.461])
    assert r.symbols == ["O", "H", "H"]
    assert np.allclose(r.coords[1], [0.0, 0.7572, -0.4692])


def test_parse_open_shell_fragment():
    r = oap.parse_orca_output(OPEN_SHELL)
    assert r.terminated
    assert np.allclose(r.mulliken, [0.407637, 0.296182, 0.296182])     # charge column, not spin
    assert np.allclose(r.loewdin, [0.613942, 0.193029, 0.193029])
    assert np.allclose(r.hirshfeld, [0.346280, 0.326861, 0.326861])
    assert r.chelpg is None
    # the highest occupied orbital is the beta HOMO, the lowest empty one the beta hole
    assert r.homo == pytest.approx(-0.838594)
    assert r.lumo == pytest.approx(-0.614181)
    assert r.energy == pytest.approx(-75.869184386651)


def _result(hirsh, mull, energy):
    r = oap.OrcaResult(terminated=True, energy=energy)
    r.hirshfeld = np.array(hirsh, float)
    r.mulliken = np.array(mull, float)
    r.loewdin = r.hirshfeld.copy()
    r.chelpg = r.hirshfeld.copy()
    r.homo, r.lumo, r.homo_index, r.lumo_index = -0.25, 0.05, 4, 5
    r.mo_populations = {4: {0: 0.9, 1: 0.05, 2: 0.05}, 5: {0: 0.1, 2: 0.9}}
    return r


def test_derived_fukui_arithmetic():
    n = _result([-0.30, 0.15, 0.15], [-0.20, 0.10, 0.10], -76.30)
    np1 = _result([-0.90, -0.05, -0.05], [-0.80, -0.10, -0.10], -76.32)   # anion: one more electron
    nm1 = _result([0.40, 0.30, 0.30], [0.50, 0.25, 0.25], -75.85)         # cation
    atom, mol = oap.derive_properties(n, np1, nm1, charge=0, mult=1)

    assert np.allclose(atom["fukui_plus"], [0.60, 0.20, 0.20])
    assert np.allclose(atom["fukui_minus"], [0.70, 0.15, 0.15])
    assert np.allclose(atom["fukui_zero"], [0.65, 0.175, 0.175])
    assert np.allclose(atom["dual"], [-0.10, 0.05, 0.05])
    assert np.allclose(atom["fukui_plus_mulliken"], [0.60, 0.20, 0.20])
    assert np.allclose(atom["fukui_minus_mulliken"], [0.70, 0.15, 0.15])
    assert np.allclose(atom["dual_mulliken"], [-0.10, 0.05, 0.05])
    # Fukui functions integrate to the one electron that moved
    assert atom["fukui_plus"].sum() == pytest.approx(1.0)
    assert atom["fukui_minus"].sum() == pytest.approx(1.0)

    ev = oap.HARTREE_EV
    ip, ea = 0.45 * ev, 0.02 * ev
    eta = (ip - ea) / 2
    s = 1 / (2 * eta)
    mu = -(ip + ea) / 2
    omega = mu ** 2 / (2 * eta)
    assert mol["IP"] == pytest.approx(ip)
    assert mol["EA"] == pytest.approx(ea)
    assert mol["hardness"] == pytest.approx(eta)
    assert mol["softness"] == pytest.approx(s)
    assert mol["chemical_potential"] == pytest.approx(mu)
    assert mol["electrophilicity"] == pytest.approx(omega)
    assert np.allclose(atom["softness_plus"], s * atom["fukui_plus"])
    assert np.allclose(atom["softness_minus"], s * atom["fukui_minus"])
    assert np.allclose(atom["local_electrophilicity"], omega * atom["fukui_plus"])
    assert np.allclose(atom["homo_pop"], [0.9, 0.05, 0.05])
    assert np.allclose(atom["lumo_pop"], [0.1, 0.0, 0.9])
    assert mol["E_homo"] == pytest.approx(-0.25 * ev)
    assert mol["gap"] == pytest.approx(0.30 * ev)
    assert mol["E_total"] == pytest.approx(-76.30)
    assert mol["charge"] == 0 and mol["multiplicity"] == 1

    # without the ionic states only the N-electron quantities are there
    atom_n, mol_n = oap.derive_properties(n)
    assert set(atom_n) == set(oap.ATOM_PROPS_N)
    assert "IP" not in mol_n


def test_input_writer():
    inp = oap.make_input("B3LYP D3BJ def2-SVP RIJCOSX def2/J TightSCF", 0, 1, "geom.xyz", nprocs=4)
    first = inp.splitlines()[0]
    assert first.startswith("! B3LYP D3BJ def2-SVP RIJCOSX def2/J TightSCF")
    assert "CHELPG" in first and "UKS" not in first and "Opt" not in first
    assert "%pal nprocs 4 end" in inp
    assert "Print[P_Hirshfeld] 1" in inp and "Print[P_AtPopMO_M] 1" in inp
    assert inp.rstrip().endswith("* xyzfile 0 1 geom.xyz")

    anion = oap.make_input("r2SCAN-3c", -1, 2, "opt.xyz", nprocs=1, chelpg=False)
    assert "UKS" in anion.splitlines()[0] and "CHELPG" not in anion
    assert "%pal" not in anion
    assert "* xyzfile -1 2 opt.xyz" in anion

    opt = oap.make_input("r2SCAN-3c", 1, 1, "geom.xyz", opt=True, populations=False, chelpg=False)
    assert " Opt" in opt.splitlines()[0] and "%output" not in opt and "CHELPG" not in opt
    assert "* xyzfile 1 1 geom.xyz" in opt


def test_multiplicity_from_electron_count():
    from rdkit import Chem
    water = Chem.AddHs(Chem.MolFromSmiles("O"))
    assert oap.multiplicity_for(water, 0) == 1
    assert oap.multiplicity_for(water, 1) == 2
    assert oap.multiplicity_for(water, -1) == 2
    cation = Chem.AddHs(Chem.MolFromSmiles("C[NH3+]"))
    assert oap.multiplicity_for(cation, Chem.GetFormalCharge(cation)) == 1
    assert oap.multiplicity_for(cation, 0) == 2      # the N+1 state of the protonated amine


def test_cache_key():
    level = "B3LYP D3BJ def2-SVP"
    k = oap.cache_key("OCC", 0, level, False)
    assert k == oap.cache_key("C(O)C", 0, level, False)            # same molecule, other spelling
    assert k == oap.cache_key("OCC", 0, "B3LYP  D3BJ def2-SVP", False)   # whitespace is not a level
    assert k != oap.cache_key("OCC", 1, level, False)
    assert k != oap.cache_key("OCC", 0, "r2SCAN-3c", False)
    assert k != oap.cache_key("OCC", 0, level, True)
    assert k != oap.cache_key("OCCO", 0, level, False)
    xyz = np.zeros((3, 3))
    assert oap.cache_key("O", 0, level, False, xyz) == oap.cache_key("O", 0, level, False, xyz + 1e-5)
    assert oap.cache_key("O", 0, level, False, xyz) != oap.cache_key("O", 0, level, False, xyz + 0.1)


def test_sdf_round_trip(tmp_path):
    from rdkit import Chem
    from cmfqsar.align import embed_3d
    mol = embed_3d(Chem.MolFromSmiles("CO"), seed=3)
    mol.SetProp("_Name", "methanol")
    mol.SetProp("activity", "4.2")
    n = mol.GetNumAtoms()
    rng = np.random.default_rng(0)
    names = oap.ATOM_PROPS_N + oap.ATOM_PROPS_FUKUI
    atom = {k: rng.normal(size=n) for k in names}
    molp = {"E_total": -115.5, "E_homo": -7.1, "E_lumo": 1.2, "gap": 8.3, "dipole": 1.7,
            "charge": 0.0, "multiplicity": 1.0, "IP": 10.5, "EA": -1.0, "hardness": 5.75,
            "softness": 1 / 11.5, "electrophilicity": 1.96, "chemical_potential": -4.75}
    coords = mol.GetConformer().GetPositions() + 0.25        # "optimised" geometry
    ann = oap.annotate(mol, atom, molp, coords=coords, extra={"qm_level": "test"})

    path = str(tmp_path / "qm.sdf")
    w = Chem.SDWriter(path)
    w.write(ann)
    w.close()

    ds = load_dataset(path)
    assert len(ds) == 1
    assert set(names) <= set(ds.atom_properties)
    back = ds.mols[0]
    assert back.GetNumAtoms() == n                        # hydrogens kept
    for k in names:
        assert np.allclose(atom_property(back, k), atom[k], atol=1e-6), k
    assert np.allclose(back.GetConformer().GetPositions(), coords, atol=1e-4)
    assert back.GetProp("activity") == "4.2"
    assert ds.activity[0] == pytest.approx(4.2)
    assert back.GetProp("charge") == "0" and back.GetProp("multiplicity") == "1"
    assert float(back.GetProp("IP")) == pytest.approx(10.5)
    assert back.GetProp("qm_level") == "test"
    assert ds.names[0] == "methanol"
    assert oap.verify_sdf(path, names)


def test_finished_job_is_not_rerun(tmp_path):
    """A job whose output already says TERMINATED NORMALLY is parsed, not run: the resume path."""
    jobdir = str(tmp_path)
    inp = oap.make_input("B3LYP", 0, 1, "geom.xyz")
    with open(os.path.join(jobdir, "sp_n.inp"), "w") as fh:
        fh.write(inp)
    with open(os.path.join(jobdir, "sp_n.out"), "w") as fh:
        fh.write(CLOSED_SHELL)
    cfg = oap.Config(orca=os.path.join(jobdir, "no_such_orca.exe"))
    r = oap.run_job(jobdir, "sp_n", inp, cfg)
    assert r.energy == pytest.approx(-76.321663220492)
    # a changed input invalidates the output, and running is then attempted (and fails here)
    with pytest.raises(Exception):
        oap.run_job(jobdir, "sp_n", oap.make_input("r2SCAN-3c", 0, 1, "geom.xyz"), cfg)
    assert not oap._output_ok(os.path.join(jobdir, "sp_n.out"), False)


@pytest.mark.skipif(not os.path.isfile(oap.DEFAULT_ORCA), reason="ORCA is not installed here")
def test_water_end_to_end(tmp_path):
    """One real run: water, N and N+-1 states, serial, a few seconds per state."""
    csv = tmp_path / "water.csv"
    csv.write_text("smiles,name,activity\nO,water,1.0\n")
    out = str(tmp_path / "water_qm.sdf")
    cfg = oap.Config(orca=oap.DEFAULT_ORCA, level=oap.DEFAULT_LEVEL, nprocs=1, timeout=240,
                     workdir=str(tmp_path / "orca"))
    outcomes = oap.process_dataset(str(csv), out, cfg, jobs=1, log=lambda *a: None)
    assert len(outcomes) == 1 and outcomes[0].mol is not None, outcomes[0].error
    ds = load_dataset(out)
    assert set(oap.ATOM_PROPS_N + oap.ATOM_PROPS_FUKUI) <= set(ds.atom_properties)
    m = ds.mols[0]
    assert m.GetNumAtoms() == 3
    assert atom_property(m, "fukui_plus").sum() == pytest.approx(1.0, abs=0.01)
    assert atom_property(m, "fukui_minus").sum() == pytest.approx(1.0, abs=0.01)
    assert abs(atom_property(m, "q_chelpg").sum()) < 0.01
    assert atom_property(m, "fukui_minus")[0] > 0.5           # the electron leaves from the oxygen
    assert float(m.GetProp("IP")) > 8.0                        # water: ~12.6 eV
    assert float(m.GetProp("gap")) > 5.0
    assert m.GetProp("activity") == "1.0"
    # a second call reuses everything
    again = oap.process_dataset(str(csv), out, cfg, jobs=1, log=lambda *a: None)
    assert again[0].reused and again[0].seconds < 5


def test_missing_properties_finds_the_molecules_still_to_run(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    done = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(done, randomSeed=1)
    annotated = oap.annotate(done, {"q_mulliken": np.zeros(done.GetNumAtoms())},
                             {"E_total": -1.0})
    todo = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(todo, randomSeed=1)
    assert oap.missing_properties([annotated, todo]) == [1]


def test_estimate_warns_about_the_cost_before_the_run():
    one = oap.estimate_seconds(1, fukui=False, jobs=1, per_job=120)
    assert one == pytest.approx(120)
    assert oap.estimate_seconds(1, fukui=True, jobs=1, per_job=120) == pytest.approx(360)
    assert oap.estimate_seconds(12, fukui=True, jobs=4, per_job=120) == pytest.approx(1080)


def test_cancelling_runs_nothing_and_keeps_what_was_done(tmp_path):
    """A cancel asked before a molecule starts must not launch ORCA for it."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mols = []
    for smi in ("CO", "CC"):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=1)
        mols.append(m)
    cfg = oap.Config(orca="definitely-not-an-executable", workdir=str(tmp_path / "work"),
                     fukui=False)
    out = oap.annotate_molecules(mols, ["a", "b"], cfg, cancel=lambda: True)
    assert [o.error for o in out] == ["cancelled", "cancelled"]
    assert all(o.mol is None for o in out)
