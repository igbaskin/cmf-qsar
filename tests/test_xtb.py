"""Tests for cmfqsar/xtb.py: the parser, the derived properties, the command line it builds,
the cache, the SDF round trip, and -- when xtb is installed -- real runs on water that fix the
Fukui convention against the program's own charges.

The claim the module rests on is that its Fukui indices mean the same as `cmfqsar/orca.py`'s, so
that one ``prop:`` field name can be filled by either engine. That is not a matter of taste: f+
is defined as q(N) - q(N+1), and `test_the_fukui_convention_is_the_same_as_orcas` computes both
sides from separate xtb runs of the three charge states and compares them.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar import xtb as QX  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import atom_property  # noqa: E402

#: A trimmed GFN2-xTB 6.7.1 output of a water single point with --vfukui: the blocks the module
#: reads, in xtb's own layout, with an earlier (stale) per-atom table first to check that the
#: LAST one is used.
WATER_OUT = """
     #   Z          covCN         q      C6AA        α(0)
     1   8 O        1.611    -9.999    24.329     9.999
     2   1 H        0.806     9.999     0.778     9.999
     3   1 H        0.806     9.999     0.778     9.999

Fukui index Calculation

Run single point for reduced species

 iter      E             dE          RMSdq      gap      omega  full diag
   1     -4.8452323 -0.484523E+01  0.421E+00    4.23       0.0  T

Run single point for oxidized species

 iter      E             dE          RMSdq      gap      omega  full diag
   1     -4.3873892 -0.438739E+01  0.741E+00   14.63       0.0  T

Fukui functions:
     #        f(+)     f(-)     f(0)
     1O       0.234    0.598    0.416
     2H       0.383    0.201    0.292
     3H       0.383    0.201    0.292

           -------------------------------------------------
          |                Property Printout                |
           -------------------------------------------------

    * Orbital Energies and Occupations

         #    Occupation            Energy/Eh            Energy/eV
      -------------------------------------------------------------
         1        2.0000           -0.6816642             -18.5490
         2        2.0000           -0.5664674             -15.4144
         3        2.0000           -0.5147063             -14.0059
         4        2.0000           -0.4474927             -12.1769 (HOMO)
         5                          0.0901379               2.4528 (LUMO)
         6                          0.2456096               6.6834
      -------------------------------------------------------------
                  HL-Gap            0.5376306 Eh           14.6297 eV
             Fermi-level           -0.1786774 Eh           -4.8621 eV

     #   Z          covCN         q      C6AA        α(0)
     1   8 O        1.611    -0.564    24.329     6.657
     2   1 H        0.806     0.282     0.778     1.386
     3   1 H        0.806     0.282     0.778     1.386

 Mol. C6AA /au·bohr⁶  :         44.526912

molecular dipole:
                 x           y           z       tot (Debye)
 q only:        0.000       0.000      -0.625
   full:       -0.000       0.000      -0.894       2.271
molecular quadrupole (traceless):
                xx          xy          yy          xz          yz          zz
   full:       -1.250       0.000       1.417      -0.000      -0.000      -0.167

           -------------------------------------------------
          | TOTAL ENERGY               -5.070370761845 Eh   |
          | GRADIENT NORM               0.007221585093 Eh/α |
          | HOMO-LUMO GAP              14.629673576893 eV   |
           -------------------------------------------------

normal termination of xtb
"""

HAVE_XTB = True
try:
    QX.find_xtb()
except QX.XtbError:
    HAVE_XTB = False

needs_xtb = pytest.mark.skipif(not HAVE_XTB, reason="xtb is not installed here")


def _water():
    m = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(m)
    m.SetProp("_Name", "water")
    return m


# --------------------------------------------------------------------------- the parser


def test_the_parser_reads_the_per_atom_table():
    res = QX.parse_output(WATER_OUT)
    assert res.symbols == ["O", "H", "H"]
    assert np.allclose(res.charges, [-0.564, 0.282, 0.282])
    assert np.allclose(res.polarisability, [6.657, 1.386, 1.386])


def test_the_last_per_atom_table_is_the_one_that_counts():
    """xtb prints the table again for a later state; the final one describes the molecule."""
    res = QX.parse_output(WATER_OUT)
    assert not np.any(np.isclose(res.charges, 9.999)), "the stale first table was used"


def test_the_parser_reads_the_molecule_properties():
    res = QX.parse_output(WATER_OUT)
    assert res.energy == pytest.approx(-5.070370761845)
    assert res.homo == pytest.approx(-12.1769)
    assert res.lumo == pytest.approx(2.4528)
    assert res.gap == pytest.approx(14.629673576893)
    assert res.dipole == pytest.approx(2.271), "the dipole, not the quadrupole, is read"


def test_the_parser_reads_the_fukui_table():
    res = QX.parse_output(WATER_OUT)
    assert np.allclose(res.fukui_plus, [0.234, 0.383, 0.383])
    assert np.allclose(res.fukui_minus, [0.598, 0.201, 0.201])
    assert np.allclose(res.fukui_zero, [0.416, 0.292, 0.292])


def test_an_unfinished_run_is_refused():
    with pytest.raises(QX.XtbError, match="normal"):
        QX.parse_output(WATER_OUT.replace("normal termination of xtb", "abnormal termination"))


def test_an_output_with_no_atom_table_is_refused():
    text = "\n".join(line for line in WATER_OUT.splitlines()
                     if "O " not in line and "H " not in line)
    with pytest.raises(QX.XtbError, match="per-atom"):
        QX.parse_output(text)


def test_a_run_without_the_fukui_states_has_no_fukui_values():
    text = WATER_OUT[:WATER_OUT.index("Fukui index Calculation")] + \
        WATER_OUT[WATER_OUT.index("           ------"):]
    res = QX.parse_output(text)
    assert res.fukui_plus is None
    atom, _mol = QX.derive_properties(res)
    assert "fukui_plus" not in atom and "q_xtb" in atom


# ------------------------------------------------------------------ the derived properties


def test_the_dual_descriptor_is_the_difference_of_the_two_fukui_indices():
    res = QX.parse_output(WATER_OUT)
    atom, mol = QX.derive_properties(res, charge=0, mult=1)
    assert np.allclose(atom["dual"], res.fukui_plus - res.fukui_minus)
    assert np.allclose(atom["dual"][0], 0.234 - 0.598)
    assert set(QX.ATOM_PROPS + QX.ATOM_PROPS_FUKUI) <= set(atom)
    assert set(QX.MOL_PROPS) <= set(mol)


def test_the_charge_carries_a_name_of_its_own():
    """q_xtb is a tight-binding population and q_hirshfeld is not; they must not share a name."""
    atom, _mol = QX.derive_properties(QX.parse_output(WATER_OUT))
    assert "q_xtb" in atom and "q_hirshfeld" not in atom


# ----------------------------------------------------------------------- the command line


def _stand_in_exe(tmp_path):
    """A file that stands in for the program.

    Building an argument list needs a path, not a working xtb, so these tests make an empty file
    and never run it. They used to pass the made-up name ``"x"``, which `find_xtb` quietly
    replaced with the installed xtb, so they passed here and failed wherever xtb is absent.
    """
    exe = tmp_path / ("xtb.exe" if os.name == "nt" else "xtb")
    exe.write_text("")
    return str(exe)


def test_the_command_carries_the_charge_and_the_spin(tmp_path):
    exe = _stand_in_exe(tmp_path)
    args = QX.command(QX.Config(xtb=exe), "geom.xyz", charge=-1, mult=2)
    assert args[0] == exe, "the program given is the one the command names"
    assert "--chrg" in args and args[args.index("--chrg") + 1] == "-1"
    assert "--uhf" in args and args[args.index("--uhf") + 1] == "1"
    assert "--vfukui" in args and "--opt" not in args


def test_the_options_reach_the_command(tmp_path):
    cfg = QX.Config(xtb=_stand_in_exe(tmp_path), gfn=1, opt=True, fukui=False, solvent="water",
                    accuracy=0.5)
    args = QX.command(cfg, "geom.xyz", 0, 1)
    assert args[args.index("--gfn") + 1] == "1"
    assert args[args.index("--alpb") + 1] == "water"
    assert args[args.index("--acc") + 1] == "0.5"
    assert "--opt" in args and "--vfukui" not in args


def test_the_level_string_names_what_was_done():
    assert QX.Config(xtb="x").level() == "GFN2-xTB acc=1"
    assert "alpb=water" in QX.Config(xtb="x", solvent="water").level()
    assert "opt" in QX.Config(xtb="x", opt=True).level()


def test_a_missing_executable_says_where_to_get_one(monkeypatch):
    monkeypatch.delenv("XTB_EXE", raising=False)
    monkeypatch.setattr(QX.shutil, "which", lambda _name: None)
    monkeypatch.setattr(QX, "DEFAULT_XTB", os.path.join(HERE, "no_such_xtb.exe"))
    with pytest.raises(QX.XtbError, match="github.com/grimme-lab/xtb"):
        QX.find_xtb()


def test_a_given_path_that_names_no_file_is_refused(monkeypatch, tmp_path):
    """And is not replaced by another xtb, which is the point of the check.

    The installed program is put where every fallback would find it, so the test fails if the
    search walks past the path it was given instead of stopping at it.
    """
    installed = tmp_path / "installed_xtb"
    installed.write_text("")
    monkeypatch.setenv("XTB_EXE", str(installed))
    # only the bare name is on the path, which is what `shutil.which` does with a path
    monkeypatch.setattr(QX.shutil, "which",
                        lambda name: str(installed) if name == "xtb" else None)
    monkeypatch.setattr(QX, "DEFAULT_XTB", str(installed))

    missing = str(tmp_path / "nowhere" / "xtb.exe")
    with pytest.raises(QX.XtbError, match="nowhere"):
        QX.find_xtb(missing)


def test_a_wrong_xtb_exe_is_refused_too(monkeypatch, tmp_path):
    """``XTB_EXE`` is also a path the user gave, so a stale one is an error, not a fallback."""
    installed = tmp_path / "installed_xtb"
    installed.write_text("")
    monkeypatch.setenv("XTB_EXE", str(tmp_path / "stale" / "xtb.exe"))
    # only the bare name is on the path, which is what `shutil.which` does with a path
    monkeypatch.setattr(QX.shutil, "which",
                        lambda name: str(installed) if name == "xtb" else None)
    monkeypatch.setattr(QX, "DEFAULT_XTB", str(installed))
    with pytest.raises(QX.XtbError, match="XTB_EXE"):
        QX.find_xtb()


def test_a_bare_name_is_looked_up_on_the_path(monkeypatch, tmp_path):
    """`--xtb xtb` still works: a name that is not a file is searched for on the path."""
    installed = tmp_path / "xtb"
    installed.write_text("")
    monkeypatch.setattr(QX.shutil, "which", lambda name: str(installed) if name == "xtb" else None)
    assert QX.find_xtb("xtb") == str(installed)


def test_a_path_that_exists_wins_over_the_environment(monkeypatch, tmp_path):
    """The order is unchanged: the argument first, then ``XTB_EXE``, then the path."""
    given = tmp_path / "given_xtb"
    given.write_text("")
    other = tmp_path / "other_xtb"
    other.write_text("")
    monkeypatch.setenv("XTB_EXE", str(other))
    assert QX.find_xtb(str(given)) == str(given)
    assert QX.find_xtb() == str(other)


# ------------------------------------------------------------------------------- the runs


@needs_xtb
def test_water_end_to_end(tmp_path):
    cfg = QX.Config(workdir=str(tmp_path / "work"))
    oc = QX.run_molecule(_water(), "water", 1, cfg)
    assert oc.error == "" and oc.mol is not None
    o = oc.mol.GetAtomWithIdx(0)
    assert o.GetSymbol() == "O"
    assert o.GetDoubleProp("q_xtb") < -0.3, "the oxygen of water carries a negative charge"
    assert o.GetDoubleProp("alpha_xtb") > 1.0
    assert float(oc.mol.GetProp("gap")) > 5.0


@needs_xtb
def test_the_fukui_convention_is_the_same_as_orcas(tmp_path):
    """f+ = q(N) - q(N+1) and f- = q(N-1) - q(N), which is how cmfqsar/orca.py defines them.

    The three charge states are run separately and the differences of their charges are compared
    with what --vfukui printed. This is the fact that lets a model use one ``prop:fukui_minus``
    field whichever program filled it.
    """
    mol = _water()
    base = QX.Config(workdir=str(tmp_path / "work"))
    printed = QX.run_job(_job(mol, tmp_path, "n", base, 0, 1), base, 0, 1)

    plain = QX.Config(workdir=str(tmp_path / "work"), fukui=False)
    neutral = QX.run_job(_job(mol, tmp_path, "q0", plain, 0, 1), plain, 0, 1)
    anion = QX.run_job(_job(mol, tmp_path, "qm", plain, -1, 2), plain, -1, 2)
    cation = QX.run_job(_job(mol, tmp_path, "qp", plain, 1, 2), plain, 1, 2)

    assert np.allclose(printed.fukui_plus, neutral.charges - anion.charges, atol=2e-3)
    assert np.allclose(printed.fukui_minus, cation.charges - neutral.charges, atol=2e-3)


def _job(mol, tmp_path, tag, cfg, charge, mult):
    """A job directory holding the molecule's geometry, ready for run_job."""
    from cmfqsar.orca import xyz_block
    jobdir = str(tmp_path / "work" / tag)
    os.makedirs(jobdir, exist_ok=True)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    with open(os.path.join(jobdir, "geom.xyz"), "w", newline="\n") as fh:
        fh.write(xyz_block(symbols, mol.GetConformer().GetPositions(), tag))
    return jobdir


@needs_xtb
def test_a_finished_job_is_not_rerun(tmp_path):
    cfg = QX.Config(workdir=str(tmp_path / "work"))
    mol = _water()
    first = QX.run_molecule(mol, "water", 1, cfg)
    second = QX.run_molecule(mol, "water", 1, cfg)
    assert first.error == "" and second.error == ""
    assert second.reused and second.seconds < max(first.seconds, 0.02)


def test_a_reused_job_gives_the_charges_of_its_first_run(tmp_path):
    """The first run takes the charges from the ``charges`` file, which has 8 decimals; the
    table of the output has 3.  A job that is read back from its directory must take them
    from the same file, or the second build of a data file differs from the first."""
    jobdir = str(tmp_path / "job")
    os.makedirs(jobdir)
    with open(os.path.join(jobdir, "xtb.out"), "w", encoding="utf-8") as fh:
        fh.write(WATER_OUT)
    cfg = QX.Config(xtb="no such program", workdir=str(tmp_path))
    printed = QX.run_job(jobdir, cfg, 0, 1)
    assert np.allclose(printed.charges, [-0.564, 0.282, 0.282])
    with open(os.path.join(jobdir, "charges"), "w") as fh:
        fh.write("-0.56412345\n0.28206172\n0.28206173\n")
    reused = QX.run_job(jobdir, cfg, 0, 1)
    assert np.array_equal(reused.charges, [-0.56412345, 0.28206172, 0.28206173])
    # a file for another number of atoms is not this job's file
    with open(os.path.join(jobdir, "charges"), "w") as fh:
        fh.write("0.1\n0.2\n")
    assert np.allclose(QX.run_job(jobdir, cfg, 0, 1).charges, [-0.564, 0.282, 0.282])


@needs_xtb
def test_a_finished_job_keeps_its_charges_when_it_is_read_again(tmp_path):
    from cmfqsar.fields import atom_property

    cfg = QX.Config(workdir=str(tmp_path / "work"))
    first = QX.run_molecule(_water(), "water", 1, cfg)
    second = QX.run_molecule(_water(), "water", 1, cfg)
    assert second.reused
    for name in ("q_xtb", "fukui_plus", "fukui_minus"):
        assert np.array_equal(atom_property(first.mol, name), atom_property(second.mol, name))
    # more digits than the table of the output prints
    q = atom_property(first.mol, "q_xtb")
    assert np.any(np.abs(q - np.round(q, 3)) > 1e-6)


@needs_xtb
def test_a_different_molecule_in_the_same_slot_is_computed_again(tmp_path):
    """The cache key is the molecule, so slot 1 holding another compound must not be reused."""
    cfg = QX.Config(workdir=str(tmp_path / "work"))
    QX.run_molecule(_water(), "water", 1, cfg)
    other = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(other, randomSeed=2)
    oc = QX.run_molecule(other, "methanol", 1, cfg)
    assert oc.error == "" and oc.mol is not None
    assert not oc.reused
    assert oc.mol.GetNumAtoms() == other.GetNumAtoms()


@needs_xtb
def test_a_failing_molecule_is_reported_and_not_raised(tmp_path):
    cfg = QX.Config(workdir=str(tmp_path / "work"), timeout=1e-6)
    oc = QX.run_molecule(_water(), "water", 1, cfg)
    assert oc.mol is None and oc.error, "a failure must come back as an outcome, not an exception"


@needs_xtb
def test_the_sdf_round_trip_gives_the_program_its_fields(tmp_path):
    path = str(tmp_path / "set.sdf")
    with Chem.SDWriter(path) as w:
        for k, smi in enumerate(["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"]):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(m, randomSeed=k + 1)
            AllChem.MMFFOptimizeMolecule(m)
            m.SetProp("_Name", f"m{k}")
            m.SetProp("activity", f"{4.0 + k:.2f}")
            w.write(m)
    out = str(tmp_path / "set_xtb.sdf")
    cfg = QX.Config(workdir=str(tmp_path / "work"))
    outcomes = QX.process_dataset(path, out, cfg, log=lambda *_a: None)
    assert all(oc.mol is not None for oc in outcomes)

    ds = load_dataset(out)
    for name in QX.ATOM_PROPS + QX.ATOM_PROPS_FUKUI:
        assert name in ds.atom_properties, f"{name} did not survive the file"
    vals = atom_property(ds.mols[0], "fukui_minus")
    assert len(vals) == ds.mols[0].GetNumAtoms() and np.all(np.isfinite(vals))
    assert ds.mols[0].GetProp("activity") == "4.00", "the input's own properties are kept"
