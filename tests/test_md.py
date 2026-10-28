"""Tests for cmfqsar/md.py and the comparison of benchmarks/run_md.py (plan item D8).

The simulations themselves need OpenMM and the OpenFF stack, which is conda-forge only, so the
tests that need them are skipped where they are not installed. Everything that decides what a
run *is* -- how long it runs, which entry it uses, how the frames are counted -- is arithmetic
and file reading, and that is tested everywhere.

The comparison is tested on predictions made up for the purpose, because its job is to turn
predictions into a verdict and that job must be right whatever produced them.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from cmfqsar import md as MD  # noqa: E402

HAVE_STACK = all(MD.available().values())


# ----------------------------------------------------------------------------- the settings


def test_the_length_of_a_run_is_what_the_settings_say():
    s = MD.Settings(nanoseconds=2.0, timestep_fs=4.0)
    assert s.production_steps == 500_000, "2 ns at 4 fs"
    s = MD.Settings(nanoseconds=0.5, timestep_fs=2.0)
    assert s.production_steps == 250_000


def test_the_equilibration_and_the_frames_are_counted_in_the_same_units():
    s = MD.Settings(equilibration_ps=200.0, frame_ps=10.0, timestep_fs=4.0)
    assert s.equilibration_steps == 50_000
    assert s.frame_steps == 2_500
    assert s.production_steps // s.frame_steps == 200, "2 ns at a frame every 10 ps"


def test_a_frame_interval_shorter_than_a_step_still_writes_frames():
    s = MD.Settings(frame_ps=0.0, timestep_fs=4.0)
    assert s.frame_steps == 1


# ------------------------------------------------------------------------------ the receptor


def test_the_source_entry_is_read_from_a_pocket_file():
    """The pocket files of examples/pockets name the entry they were cut from."""
    path = os.path.join(ROOT, "examples", "pockets", "openff", "mcl1.pdb")
    if not os.path.isfile(path):
        pytest.skip("the example pockets are not in this checkout")
    assert MD.entry_of(path) == "4HW3"


def test_a_file_with_no_source_entry_gives_none(tmp_path):
    path = str(tmp_path / "p.pdb")
    with open(path, "w") as fh:
        fh.write("REMARK 400 something else\nATOM      1  N   ALA A   1       0.0   0.0   0.0\n")
    assert MD.entry_of(path) is None


def test_the_remark_is_not_looked_for_past_the_coordinates(tmp_path):
    """A REMARK after the atoms is not a header, and reading the whole file would be wasteful."""
    path = str(tmp_path / "p.pdb")
    with open(path, "w") as fh:
        fh.write("ATOM      1  N   ALA A   1       0.0   0.0   0.0\n"
                 "REMARK 400 Source entry 1ABC from RCSB\n")
    assert MD.entry_of(path) is None


def test_an_entry_already_downloaded_is_not_fetched_again(tmp_path):
    d = str(tmp_path / "entries")
    os.makedirs(d)
    path = os.path.join(d, "4HW3.pdb")
    with open(path, "w") as fh:
        fh.write("X" * 2000)
    assert MD.fetch_entry("4hw3", d) == path, "the code is matched without regard to case"
    with open(path) as fh:
        assert fh.read() == "X" * 2000, "the file on disk was kept"


# -------------------------------------------------------------------------------- the stack


def test_available_reports_every_part():
    have = MD.available()
    assert set(have) == {"openmm", "pdbfixer", "openmmforcefields", "openff"}
    assert all(isinstance(v, bool) for v in have.values())


@pytest.mark.skipif(HAVE_STACK, reason="the stack is installed here")
def test_a_missing_stack_says_how_to_get_one():
    with pytest.raises(MD.MDError, match="conda-forge"):
        MD.require()


@pytest.mark.skipif(not HAVE_STACK, reason="the simulation stack is not installed here")
def test_the_stack_builds_a_system(tmp_path):
    """One real build, small and short, where the stack is there."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    pocket = os.path.join(ROOT, "examples", "pockets", "openff", "mcl1.pdb")
    if not os.path.isfile(pocket):
        pytest.skip("the example pockets are not in this checkout")
    lig = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    AllChem.EmbedMolecule(lig, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(lig)
    ready = MD.prepare_receptor(pocket, str(tmp_path / "ready.pdb"))
    topology, system, positions, _rd = MD.build(ready, lig, MD.Settings(padding=6.0))
    assert topology.getNumAtoms() > lig.GetNumAtoms()
    assert system.getNumParticles() == topology.getNumAtoms()
    _idx, n = MD.add_restraints(system, topology, positions, 5.0)
    assert n > 0, "a protein has alpha carbons to restrain"


# ------------------------------------------------------------------------- the comparison


def _rows(deltas, n=24, seed=0):
    """A result record whose models differ from the reference by the given amounts in q2."""
    import run_md

    rng = np.random.RandomState(seed)
    y = rng.normal(size=n) * 2.0
    base = y + rng.normal(size=n) * 1.0
    models = {"crystal": {"q2": 0.5, "rmse": 1.0, "alpha": 0.5, "lam": 0.1,
                          "pred": base.tolist()}}
    for tag, d in deltas.items():
        # a prediction closer to (or further from) the truth than the reference by construction
        mix = (1.0 - d) * base + d * y if d >= 0 else (1.0 + abs(d)) * base - abs(d) * y
        models[tag] = {"q2": 0.5 + d, "rmse": 1.0, "alpha": 0.5, "lam": 0.1,
                       "pred": np.asarray(mix, float).tolist()}
    rec = {"set": "toy", "n": n, "frames_mean": 18.0, "frames_min": 12, "frames_max": 20,
           "seconds": 1.0, "models": models}
    ref = np.asarray(models["crystal"]["pred"], float)
    for tag in models:
        dd, lo, hi = run_md.bootstrap_difference(y, np.asarray(models[tag]["pred"], float), ref)
        models[tag].update(delta=dd, lo=lo, hi=hi)
    return rec


def test_the_difference_against_the_reference_is_zero_for_the_reference():
    import run_md

    rng = np.random.RandomState(1)
    y = rng.normal(size=30)
    p = y + rng.normal(size=30) * 0.5
    d, lo, hi = run_md.bootstrap_difference(y, p, p)
    assert d == pytest.approx(0.0)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0)


def test_a_better_prediction_gives_a_positive_difference():
    import run_md

    rng = np.random.RandomState(2)
    y = rng.normal(size=40)
    worse = y + rng.normal(size=40) * 1.0
    better = y + rng.normal(size=40) * 0.2
    d, lo, hi = run_md.bootstrap_difference(y, better, worse)
    assert d > 0 and lo > 0, "a clearly better prediction must clear the interval"


def test_too_few_molecules_give_no_interval():
    import run_md

    d, lo, hi = run_md.bootstrap_difference([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])
    assert np.isnan(d) and np.isnan(lo) and np.isnan(hi)


def test_the_verdict_says_the_trajectory_wins_when_it_does():
    import run_md

    rec = _rows({"conformers": 0.0, "md_uniform": 0.45, "md_population": 0.4, "md_mil": 0.3})
    text = " ".join(run_md.verdict([rec]))
    # the headline sentence, not the phrase, because the closing paragraph names it too
    assert "On toy the trajectory beats the crystal pose" in text
    assert "worse than the crystal pose" not in text
    assert "What follows for the plan" not in text, "nothing follows when the trajectory wins"


def test_the_verdict_says_the_trajectory_loses_when_it_does():
    import run_md

    rec = _rows({"conformers": 0.0, "md_uniform": -0.45, "md_population": -0.5,
                 "md_mil": -0.6})
    text = " ".join(run_md.verdict([rec]))
    assert "On toy the trajectory is worse than the crystal pose" in text
    assert "On toy the trajectory beats the crystal pose" not in text
    assert "What follows for the plan" in text, "a loss has a consequence for D9"


def test_the_verdict_says_nothing_was_settled_when_the_interval_contains_zero():
    """A difference that is noise rather than a shift: better on some molecules, worse on others."""
    import run_md

    rng = np.random.RandomState(7)
    n = 24
    y = rng.normal(size=n) * 2.0
    base = y + rng.normal(size=n)
    models = {"crystal": {"q2": 0.5, "rmse": 1.0, "alpha": 0.5, "lam": 0.1,
                          "pred": base.tolist()}}
    for tag in ("conformers", "md_uniform", "md_population", "md_mil"):
        jitter = base + rng.normal(size=n) * 0.4      # no systematic gain in either direction
        models[tag] = {"q2": 0.5, "rmse": 1.0, "alpha": 0.5, "lam": 0.1,
                       "pred": jitter.tolist()}
    ref = np.asarray(base, float)
    for tag in models:
        d, lo, hi = run_md.bootstrap_difference(y, np.asarray(models[tag]["pred"], float), ref)
        models[tag].update(delta=d, lo=lo, hi=hi)
    assert models["md_uniform"]["lo"] < 0 < models["md_uniform"]["hi"], "the case being tested"
    rec = {"set": "toy", "n": n, "frames_mean": 18.0, "frames_min": 12, "frames_max": 20,
           "seconds": 1.0, "models": models}

    text = " ".join(run_md.verdict([rec]))
    assert "neither beats the crystal pose nor loses to it" in text
    assert "contains zero" in text


def test_the_verdict_always_names_the_control_and_the_scope():
    import run_md

    rec = _rows({"conformers": 0.05, "md_uniform": 0.1, "md_population": 0.1, "md_mil": 0.1})
    text = " ".join(run_md.verdict([rec]))
    assert "generated conformations" in text, "the control must be reported either way"
    assert "not a change of binding mode" in text, "the scope must be stated"


def test_the_report_says_what_is_missing_when_nothing_has_been_run(tmp_path):
    import run_md

    text = run_md.report(results=str(tmp_path), path=str(tmp_path / "t.md"))
    assert "make_md_trajectories.py" in text
    assert os.path.isfile(str(tmp_path / "t.md"))


def test_the_report_renders_a_result(tmp_path):
    import run_md

    rec = _rows({"conformers": 0.02, "md_uniform": 0.2, "md_population": 0.2, "md_mil": 0.15})
    with open(str(tmp_path / "toy.json"), "w") as fh:
        json.dump(rec, fh)
    text = run_md.report(results=str(tmp_path), path=str(tmp_path / "t.md"))
    for label in ("the crystal pose", "generated conformations",
                  "trajectory, clusters equal", "trajectory, multi-instance"):
        assert label in text
    assert "| the crystal pose | +0.500 |" in text


def test_only_the_ligands_both_sets_have_are_compared():
    """A ligand whose simulation failed must drop out of every model, not only of that one."""
    import run_md
    from rdkit import Chem

    def mol(smi):
        return Chem.MolFromSmiles(smi)

    crystal = ([mol("CCO"), mol("CCC"), mol("CCN")], ["a", "b", "c"], np.array([1.0, 2.0, 3.0]))
    md = ([mol("CCN"), mol("CCO")], ["c", "a"], np.array([3.0, 1.0]))
    (cm, cn, cy), (mm, mn, my) = run_md.matched(crystal, md)
    assert cn == ["a", "c"] == mn
    assert np.allclose(cy, [1.0, 3.0]) and np.allclose(my, [1.0, 3.0])
    assert len(cm) == len(mm) == 2


def test_the_charge_toolkit_is_reported():
    """Either a toolkit that can do AM1-BCC is registered, or there is none; both are answers."""
    got = MD.charge_toolkit()
    assert got is None or isinstance(got, str)


@pytest.mark.skipif(not HAVE_STACK, reason="the simulation stack is not installed here")
def test_a_run_without_a_charge_toolkit_says_what_to_do(monkeypatch):
    """The common failure: every package present, antechamber not on PATH."""
    monkeypatch.setattr(MD, "charge_toolkit", lambda: None)
    with pytest.raises(MD.MDError, match="PATH"):
        MD.require()
    MD.require(charges=False)        # preparing the receptor needs no charges
