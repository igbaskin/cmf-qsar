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


def test_one_set_is_not_compared_with_itself():
    """The across-sets paragraph needs two sets, so a single set gets none."""
    import run_md

    rec = _rows({"conformers": 0.0, "md_uniform": -0.2, "md_population": -0.3, "md_mil": -0.25})
    assert run_md.across_sets([rec]) == []


def test_the_sets_are_reported_as_agreeing_when_none_of_them_gains():
    import run_md

    one = _rows({"conformers": 0.0, "md_uniform": -0.2, "md_population": -0.3, "md_mil": -0.25})
    two = _rows({"conformers": 0.0, "md_uniform": -0.3, "md_population": -0.4, "md_mil": -0.35},
                seed=3)
    two["set"] = "other"
    text = " ".join(run_md.across_sets([one, two]))
    assert "No set shows a gain from the trajectory" in text
    assert "on each of the 2 sets" in text.lower()
    assert "not a property of that set" in text


def test_the_sets_are_reported_as_disagreeing_when_one_of_them_gains():
    import run_md

    one = _rows({"conformers": 0.0, "md_uniform": -0.2, "md_population": -0.3, "md_mil": -0.25})
    two = _rows({"conformers": 0.0, "md_uniform": 0.45, "md_population": 0.4, "md_mil": 0.3},
                seed=3)
    two["set"] = "other"
    text = " ".join(run_md.across_sets([one, two]))
    assert "The sets disagree" in text
    assert "on 1 of the 2 sets (other)" in text
    assert "not a method yet" in text


def _across_row(name, n, deltas):
    """A record for the across-sets paragraph: {tag: (delta, lo, hi)}."""
    models = {"crystal": dict(q2=0.3, rmse=1.0, alpha=0.5, lam=0.1, delta=0.0, lo=0.0, hi=0.0)}
    for tag, (d, lo, hi) in deltas.items():
        models[tag] = dict(q2=0.3 + d, rmse=1.0, alpha=0.5, lam=0.1, delta=d, lo=lo, hi=hi)
        if tag == "md_mil":
            models[tag]["beta"] = 0.0
    return {"set": name, "n": n, "frames_mean": 10.0, "frames_min": 8, "frames_max": 12,
            "seconds": 1.0, "models": models}


def test_a_set_counts_as_ahead_when_any_reading_clears_the_interval():
    """The largest gain and the gain that clears the interval need not be the same reading."""
    import run_md

    one = _across_row("flat", 29, {"md_uniform": (-0.02, -0.31, 0.30),
                                   "md_mil": (0.195, -0.13, 0.52)})
    two = _across_row("edge", 36, {"md_population": (0.192, -0.07, 0.56),
                                   "md_mil": (0.178, 0.003, 0.47)})
    text = " ".join(run_md.across_sets([one, two]))
    assert "The sets disagree" in text
    assert "on 1 of the 2 sets (edge)" in text, "the set with a clearing reading is named"
    assert "md_population" not in text, "the reading is named in words, not by its tag"


def test_an_interval_that_only_just_excludes_zero_is_reported_as_an_edge_result():
    import run_md

    one = _across_row("flat", 29, {"md_mil": (0.195, -0.13, 0.52)})
    two = _across_row("edge", 36, {"md_population": (0.192, -0.07, 0.56),
                                   "md_mil": (0.178, 0.003, 0.47)})
    text = " ".join(run_md.across_sets([one, two]))
    assert "Read that carefully" in text
    assert "3 readings of a trajectory were compared" in text, "every comparison is counted"
    assert "about what chance produces" in text


def test_a_clear_gain_is_not_reported_as_an_edge_result():
    import run_md

    one = _across_row("flat", 29, {"md_mil": (0.195, -0.13, 0.52)})
    two = _across_row("wide", 36, {"md_mil": (0.40, 0.21, 0.60)})
    text = " ".join(run_md.across_sets([one, two]))
    assert "Read that carefully" not in text


def test_the_set_paragraph_names_a_reading_that_clears_when_the_largest_does_not():
    import run_md

    rec = _across_row("edge", 36, {"conformers": (-0.03, -0.25, 0.43),
                                   "md_uniform": (0.129, -0.076, 0.365),
                                   "md_population": (0.192, -0.07, 0.56),
                                   "md_mil": (0.178, 0.003, 0.47)})
    text = " ".join(run_md.verdict([rec]))
    assert "neither beats the crystal pose nor loses to it" in text, "the largest gain is flat"
    assert "A different reading of the same frames does clear the interval" in text
    assert "was not fixed before the measurement" in text
    assert "which is the same model" in text, "beta = 0 is the uniform ensemble"
    assert "+0.129" in text, "and that model's own interval is given"
    assert "where the kernel is normalised" in text


def test_a_clearing_reading_that_does_weight_the_frames_says_so():
    """With beta above zero the multi-instance fit really does weight the frames."""
    import run_md

    rec = _across_row("edge", 36, {"md_uniform": (0.05, -0.20, 0.30),
                                   "md_population": (0.192, -0.07, 0.56),
                                   "md_mil": (0.178, 0.003, 0.47)})
    rec["models"]["md_mil"]["beta"] = 1.0
    text = " ".join(run_md.verdict([rec]))
    assert "The two differ in how the frames are weighted" in text
    assert "differ in where the kernel is normalised" not in text


def test_the_consequence_says_the_gate_is_no_longer_a_plain_no():
    import run_md

    lost = _across_row("flat", 29, {"md_mil": (0.195, -0.13, 0.52)})
    won = _across_row("edge", 36, {"md_mil": (0.178, 0.003, 0.47)})
    text = " ".join(run_md.verdict([lost, won]))
    assert "What follows for the plan" in text
    assert "no longer a plain no" in text
    assert "it failed on its own evidence" in text, "D9's own verdict is not reopened"


def test_the_closing_text_counts_the_sets_it_rests_on():
    import run_md

    one = _rows({"conformers": 0.0, "md_uniform": -0.2, "md_population": -0.3, "md_mil": -0.25})
    two = _rows({"conformers": 0.0, "md_uniform": -0.3, "md_population": -0.4, "md_mil": -0.35},
                seed=3)
    two["set"] = "other"
    text = " ".join(run_md.verdict([one, two]))
    assert "What this rests on.** 2 sets, each" in text
    assert "One set, simulated" not in text
    assert "on 2 of the 2 sets (toy, other)" in text, "the consequence names the sets"


def _norm_row(name, n, ens, mil_, ens_plain, mil_plain):
    return {"set": name, "n": n, "alpha": 0.3, "lam": 0.1,
            "q2": {"ensemble_normalised": ens, "mil_normalised": mil_,
                   "ensemble_plain": ens_plain, "mil_plain": mil_plain}}


def test_the_normalisation_section_is_empty_without_a_measurement():
    import run_md

    assert run_md.norm_section([]) == []


def test_the_normalisation_section_puts_the_difference_on_the_normalisation():
    """The two paths are one model, so a gap that closes without normalisation is its doing."""
    import run_md

    rows = [_norm_row("one", 36, 0.471, 0.520, 0.550, 0.552),
            _norm_row("two", 25, 0.391, 0.463, 0.470, 0.471)]
    text = " ".join(run_md.norm_section(rows))
    assert "where the kernel is normalised" in text
    assert "0.049" in text and "0.072" in text, "both gaps are given"
    assert "0.001" in text and "0.002" in text, "and both gaps with the normalisation off"
    assert "on 2 of the 2 sets" in text, "how many sets the gap narrows on"
    assert "not all of it" in text, "the leftover is not hidden"


def test_a_normalisation_record_is_not_read_as_a_set_result(tmp_path):
    """The two kinds of file live in one directory, so the table must not read the check."""
    import json

    import run_md

    rec = _rows({"conformers": 0.0, "md_uniform": -0.2, "md_population": -0.3, "md_mil": -0.25})
    with open(str(tmp_path / "toy.json"), "w") as fh:
        json.dump(rec, fh)
    with open(str(tmp_path / "toy_norm.json"), "w") as fh:
        json.dump(_norm_row("toy", 24, 0.4, 0.45, 0.5, 0.5), fh)
    got = run_md.load_all(results=str(tmp_path))
    assert [r["set"] for r in got] == ["toy"], "only the set result is a row of the table"
    assert len(run_md.load_norms(results=str(tmp_path))) == 1


# ----------------------------------------------------------------- poses that are not known


def _docked_rec(name="toy", n=25, uniform=(0.33, 0.055, -0.08, 0.21), pop=None, mil=None,
                poses=(1.12, 8.26, 0.36, 0.56), repair=(0.90, 1.30)):
    """A pose-not-known record; each reading is (q2, delta, low, high) against the found pose."""
    def row(q2, d, lo, hi, vs=(-0.30, -0.62, -0.04)):
        return dict(q2=q2, rmse=1.0, alpha=0.5, lam=0.1, delta=d, lo=lo, hi=hi,
                    vs_crystal=dict(delta=vs[0], lo=vs[1], hi=vs[2]))

    models = {"crystal": row(0.575, 0.300, 0.080, 0.520),
              "docked": row(0.275, 0.0, 0.0, 0.0)}
    models["docked_uniform"] = row(*uniform)
    if pop is not None:
        models["docked_population"] = row(*pop)
    if mil is not None:
        models["docked_mil"] = row(*mil)
    med, mx, w1, w2 = poses
    return {"set": name, "n": n, "method": "pocket", "frames_mean": 12.0, "frames_min": 8,
            "frames_max": 20, "seconds": 1.0,
            "poses": {"median_rmsd": med, "max_rmsd": mx, "within_1A": w1, "within_2A": w2},
            "repair": {"nearest_median": repair[0], "conformation_median": repair[1]},
            "models": models}


def test_the_pose_not_known_section_is_empty_without_a_result():
    import run_md

    assert run_md.docked_section([]) == []


def test_the_pose_not_known_section_reports_the_poses_it_started_from():
    import run_md

    text = " ".join(run_md.docked_section([_docked_rec()]))
    assert "Poses that are not known" in text
    assert "median 1.12 A from the crystal poses" in text
    assert "36% within 1 A" in text
    assert "no crystal frame of any kind" in text, "how the poses were made is stated"
    assert "| the pose that was found | +0.275 | the reference |" in text


def test_an_ensemble_that_beats_the_found_pose_is_reported_as_worth_something():
    import run_md

    text = " ".join(run_md.docked_section([_docked_rec(uniform=(0.45, 0.175, 0.02, 0.33))]))
    assert "the ensemble is worth something" in text
    assert "an ensemble covers a pose that is wrong" in text


def test_an_ensemble_that_does_not_beat_the_found_pose_is_reported_plainly():
    import run_md

    text = " ".join(run_md.docked_section([_docked_rec(uniform=(0.33, 0.055, -0.08, 0.21))]))
    assert "gains nothing here either" in text
    assert "does not show it helping" in text


def test_an_ensemble_worse_than_the_found_pose_is_reported_as_worse():
    import run_md

    text = " ".join(run_md.docked_section([_docked_rec(uniform=(0.12, -0.155, -0.33, -0.02))]))
    assert "worse than the single pose it started from" in text


def test_the_largest_of_the_three_readings_is_named_when_it_is_not_the_fixed_one():
    import run_md

    rec = _docked_rec(uniform=(0.33, 0.055, -0.08, 0.21), pop=(0.41, 0.135, -0.05, 0.30))
    text = " ".join(run_md.docked_section([rec]))
    assert "trajectory from it, by population" in text
    assert "taking the largest of three is what the section above measured the cost of" in text


def test_the_section_says_whether_the_simulation_brought_the_pose_nearer():
    import run_md

    near = " ".join(run_md.docked_section([_docked_rec(repair=(0.90, 1.30))]))
    assert "bring the ligand nearer to the pose it should have had" in near
    far = " ".join(run_md.docked_section([_docked_rec(repair=(1.80, 2.20))]))
    assert "bring the ligand no nearer to the pose it should have had" in far


def test_the_nearest_conformation_is_not_read_as_repair_on_its_own():
    """A smallest of several is below a single pose even when nothing moved nearer."""
    import run_md

    text = " ".join(run_md.docked_section([_docked_rec(repair=(0.79, 1.51))]))
    assert "chosen after the fact" in text
    assert "a smallest of several and not repair" in text
    assert "did not move towards it" in text, "the ensemble drifted away, and the text says so"
    near = " ".join(run_md.docked_section([_docked_rec(repair=(0.79, 0.95))]))
    assert "did not move towards it" not in near


def test_a_pose_not_known_record_is_not_read_as_a_set_result(tmp_path):
    import run_md

    rec = _rows({"conformers": 0.0, "md_uniform": -0.2, "md_population": -0.3, "md_mil": -0.25})
    with open(str(tmp_path / "toy.json"), "w") as fh:
        json.dump(rec, fh)
    with open(str(tmp_path / "toy_docked.json"), "w") as fh:
        json.dump(_docked_rec(), fh)
    got = run_md.load_all(results=str(tmp_path))
    assert [r["set"] for r in got] == ["toy"], "only the set result is a row of the table"
    assert len(run_md.load_docked(results=str(tmp_path))) == 1


def test_the_conformations_are_measured_against_the_pose_without_superposition():
    """A conformation translated by 1 angstrom is 1 angstrom away, not zero."""
    import run_md
    from rdkit import Chem
    from rdkit.Chem import AllChem

    ref = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(ref, randomSeed=0)
    ref = Chem.RemoveHs(ref)
    many = Chem.Mol(ref)
    many.RemoveAllConformers()
    for shift in (0.0, 1.0):
        conf = Chem.Conformer(ref.GetConformer())
        for i in range(ref.GetNumAtoms()):
            x, y, z = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (x + shift, y, z))
        many.AddConformer(conf, assignId=True)
    got = run_md._conformer_rmsds(many, ref)
    assert len(got) == 2
    assert got[0] == pytest.approx(0.0, abs=1e-6)
    assert got[1] == pytest.approx(1.0, abs=1e-6)


def test_a_set_without_found_poses_has_no_stage(tmp_path):
    import run_md

    assert run_md.docked_set("not_a_set") is None
    assert run_md.stage_docked("not_a_set", results=str(tmp_path)) is None


# ------------------------------------------------------------- the reading fixed in advance


def _fixed_row(name, n, delta, seed=0, lo=None, hi=None, extra=None):
    """A record with the activities stored, so that one reading can be combined over sets."""
    import run_md

    rng = np.random.RandomState(seed)
    y = rng.normal(size=n) * 2.0
    base = y + rng.normal(size=n) * 1.0

    def shift(d):
        return (1.0 - d) * base + d * y if d >= 0 else (1.0 + abs(d)) * base - abs(d) * y

    models = {"crystal": dict(q2=float(run_md.r2_score(y, base)), rmse=1.0, alpha=0.5, lam=0.1,
                              pred=base.tolist(), delta=0.0, lo=0.0, hi=0.0)}
    for tag, d in [("md_uniform", delta)] + list((extra or {}).items()):
        pred = shift(d)
        dd, blo, bhi = run_md.bootstrap_difference(y, pred, base)
        models[tag] = dict(q2=float(run_md.r2_score(y, pred)), rmse=1.0, alpha=0.5, lam=0.1,
                           pred=np.asarray(pred, float).tolist(), delta=dd, lo=blo, hi=bhi)
    if lo is not None:
        models["md_uniform"]["lo"] = lo
    if hi is not None:
        models["md_uniform"]["hi"] = hi
    return {"set": name, "n": n, "frames_mean": 10.0, "frames_min": 8, "frames_max": 12,
            "seconds": 1.0, "activity": y.tolist(), "models": models}


def test_the_fixed_reading_section_is_empty_without_a_result():
    import run_md

    assert run_md.fixed_section([]) == []


def test_the_fixed_reading_is_reported_on_every_set():
    import run_md

    text = " ".join(run_md.fixed_section([_fixed_row("one", 25, 0.1, seed=1),
                                          _fixed_row("two", 29, 0.2, seed=2)]))
    assert "One reading fixed in advance" in text
    assert "trajectory, clusters equal" in text
    assert "| one | 25 |" in text and "| two | 29 |" in text
    assert "Combined over the 2 sets" in text
    assert "This is not a pre-registration" in text, "the limit of the rule is stated"


def test_the_combined_difference_weights_the_sets_by_their_molecules():
    import run_md

    rows = [_fixed_row("small", 12, 0.3, seed=3), _fixed_row("large", 36, 0.0, seed=4)]
    got = run_md.combined_difference(rows)
    want = 0.0
    for rec in rows:
        y = np.asarray(rec["activity"], float)
        a = np.asarray(rec["models"]["md_uniform"]["pred"], float)
        b = np.asarray(rec["models"]["crystal"]["pred"], float)
        want += len(y) / 48.0 * (run_md.r2_score(y, a) - run_md.r2_score(y, b))
    assert got["sets"] == 2 and got["n"] == 48
    assert abs(got["delta"] - want) < 1e-9, "the larger set weighs more"
    assert got["lo"] < got["delta"] < got["hi"]


def test_one_set_alone_gives_no_combined_difference():
    import run_md

    assert run_md.combined_difference([_fixed_row("one", 25, 0.2)]) is None


def test_a_set_whose_activities_were_not_stored_is_left_out_of_the_combined_estimate():
    """Records written before the activities were kept must not silently shift the average."""
    import run_md

    rows = [_fixed_row("one", 25, 0.2, seed=5), _fixed_row("two", 29, 0.2, seed=6),
            _fixed_row("old", 36, 0.2, seed=7)]
    del rows[2]["activity"]
    got = run_md.combined_difference(rows)
    assert got["sets"] == 2 and got["n"] == 54


def test_a_reading_that_gains_nothing_fixed_in_advance_is_reported_as_no_gain():
    import run_md

    text = " ".join(run_md.fixed_section([_fixed_row("one", 25, 0.0, seed=8),
                                          _fixed_row("two", 29, 0.0, seed=9)]))
    assert "shows no gain" in text
    assert "taking the best of several" in text


def test_a_reading_that_still_gains_fixed_in_advance_is_reported_as_a_gain():
    import run_md

    text = " ".join(run_md.fixed_section([_fixed_row("one", 25, 0.5, seed=10),
                                          _fixed_row("two", 29, 0.5, seed=11)]))
    assert "still shows a gain" in text
    assert "does not depend on choosing the reading after the sets were seen" in text


def test_the_fixed_reading_says_whether_it_is_also_the_largest_gain():
    import run_md

    smaller = {"md_mil": 0.05}
    text = " ".join(run_md.fixed_section([_fixed_row("one", 25, 0.3, seed=12, extra=smaller),
                                          _fixed_row("two", 29, 0.3, seed=13, extra=smaller)]))
    assert "also the largest gain on every one of these sets" in text
    larger = {"md_mil": 0.6}
    text = " ".join(run_md.fixed_section([_fixed_row("one", 25, 0.1, seed=14, extra=larger),
                                          _fixed_row("two", 29, 0.1, seed=15, extra=larger)]))
    assert "not the largest gain on any of the 2 sets" in text


# ---------------------------------------------------------------------- the own-frame control


def _own_row(name, n, uniform, own, interval=None):
    """A record with both readings of the same frames: one common frame, and each in its own."""
    models = {"crystal": dict(q2=0.3, rmse=1.0, alpha=0.5, lam=0.1, delta=0.0, lo=0.0, hi=0.0),
              "md_uniform": dict(q2=uniform, rmse=1.0, alpha=0.5, lam=0.1,
                                 delta=uniform - 0.3, lo=-0.1, hi=0.3),
              "md_ownframe": dict(q2=own, rmse=1.0, alpha=0.5, lam=0.1,
                                  delta=own - 0.3, lo=-0.5, hi=0.1)}
    if interval is not None:
        d, lo, hi = interval
        models["md_ownframe"]["vs_uniform"] = dict(delta=d, lo=lo, hi=hi)
    return {"set": name, "n": n, "frames_mean": 10.0, "frames_min": 8, "frames_max": 12,
            "seconds": 1.0, "models": models}


def test_the_own_frame_section_is_empty_without_the_control():
    import run_md

    rec = _across_row("one", 25, {"md_uniform": (0.1, -0.1, 0.3)})
    assert run_md.ownframe_section([rec]) == []


def test_the_own_frame_section_reports_the_loss_on_every_set():
    import run_md

    text = " ".join(run_md.ownframe_section([_own_row("one", 25, 0.391, 0.277),
                                             _own_row("two", 36, 0.471, 0.400)]))
    assert "What the common frame is worth" in text
    assert "| one | 25 | +0.391 | +0.277 |" in text
    assert "| two | 36 | +0.471 | +0.400 |" in text
    assert "between 0.071 and 0.114 of q2" in text, "the range of the loss over the sets"
    assert "all 2 sets" in text


def test_the_own_frame_section_names_the_sets_whose_loss_clears_the_interval():
    import run_md

    text = " ".join(run_md.ownframe_section(
        [_own_row("one", 25, 0.391, 0.277, interval=(-0.114, -0.28, -0.01)),
         _own_row("two", 36, 0.471, 0.400, interval=(-0.071, -0.21, 0.06))]))
    assert "[-0.280, -0.010]" in text, "the interval against the common frame is in the table"
    assert "excludes zero on 1 of the 2 sets (one)" in text


def test_an_own_frame_reading_that_does_not_lose_is_not_called_a_loss():
    import run_md

    text = " ".join(run_md.ownframe_section([_own_row("one", 25, 0.391, 0.277),
                                             _own_row("two", 36, 0.400, 0.460)]))
    assert "does not change q2 in one direction" in text
    assert "The common frame is what makes any of this measurable" not in text


def test_the_own_frame_control_is_not_counted_among_the_readings_that_were_tried():
    """It is built to be wrong, so counting it would make the comparisons look more numerous."""
    import run_md

    one = _across_row("flat", 29, {"md_mil": (0.195, -0.13, 0.52),
                                   "md_ownframe": (-0.30, -0.67, -0.03)})
    two = _across_row("edge", 36, {"md_population": (0.192, -0.07, 0.56),
                                   "md_mil": (0.178, 0.003, 0.47),
                                   "md_ownframe": (-0.25, -0.60, -0.02)})
    text = " ".join(run_md.across_sets([one, two]))
    assert "3 readings of a trajectory were compared" in text, "the control is left out"
    assert "own frame" not in text


def test_a_set_is_not_ahead_because_the_own_frame_control_clears_the_interval():
    import run_md

    rec = _across_row("one", 25, {"md_uniform": (-0.18, -0.52, 0.13),
                                  "md_ownframe": (0.30, 0.10, 0.50)})
    assert run_md._loses(rec), "only a candidate reading can put a set ahead"


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
