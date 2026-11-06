"""One documented command per stored result, for the external check of the hop models (item G6).

``benchmarks/run_litraj.py`` measures the hop models of plan item BT7 on a sample of the
published nebBVSE122k benchmark. It is run with options, so the two rules plan item G6 sets must
hold for it as they hold for ``benchmarks/run_hops.py``:

* every option that changes a number enters the names of the files the run writes, so that two
  runs never write over each other; and
* the registry in ``benchmarks/run_all.py`` lists one command for each stored result, so that
  ``cmf benchmark --compute litraj`` rebuilds what the repository holds and nothing else.
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "benchmarks")
RESULTS = os.path.join(BENCH, "results", "litraj")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

import run_all                                                      # noqa: E402
import run_litraj as rl                                             # noqa: E402


def tag_of(argv):
    return rl.run_tag(rl.build_parser().parse_args(argv))


def benchmark():
    return next(b for b in run_all.registry() if b.name == "litraj")


def files_of(argv):
    """The files in results/litraj that this command writes, by their names alone."""
    tag = tag_of(argv)
    return {f"public{tag}.json", f"log{tag}.txt", f"table{tag}.md", f"run{tag}.json"}


def documented_files():
    """Every file a command of the registry writes, the prepared input included."""
    b = benchmark()
    files = {os.path.basename(rl.PREPARED)} if b.prepare else set()
    for argv in b.compute:
        files |= files_of(argv)
    return files


def stored_files():
    """The files of results/litraj the repository keeps, from git rather than from the disk."""
    out = subprocess.check_output(["git", "ls-files", "benchmarks/results/litraj"],
                                  cwd=ROOT, text=True)
    return {f.split("/")[-1] for f in out.split()}


# ------------------------------------------------------------------ the tag names the run

def test_the_stored_measurement_has_the_empty_tag():
    assert tag_of([]) == ""


def test_the_prepared_sample_named_by_its_path_is_still_the_default():
    """Two ways of naming one input give one tag."""
    assert tag_of(["--prepared", rl.PREPARED]) == ""
    assert tag_of(["--prepared", os.path.relpath(rl.PREPARED, ROOT)]) == ""


@pytest.mark.parametrize("argv,expected", [
    (["--max-structures", "40"], "_n40"),
    (["--rho", "0.1"], "_rho0.1"),
    (["--rho", "0.1", "0.25"], "_rho0.1-0.25"),
    (["--r-max", "8"], "_r8"),
    (["--site-r", "4"], "_siter4"),
    (["--folds", "5"], "_folds5"),
    (["--seed", "1"], "_seed1"),
    (["--bootstrap", "200"], "_boot200"),
    (["--prepared", "benchmarks/results/litraj/prepared_all.json"], "_prepared_all"),
])
def test_an_option_that_changes_the_numbers_changes_the_tag(argv, expected):
    assert tag_of(argv) == expected
    assert tag_of(argv) != tag_of([])


def test_the_quick_look_of_the_usage_lines_does_not_write_over_the_measurement():
    assert files_of(["--max-structures", "40"]).isdisjoint(files_of([]))


@pytest.mark.parametrize("argv", [["--jobs", "4"], ["--report"]])
def test_choosing_which_parts_run_leaves_the_tag_alone(argv):
    """These options change no number, so they write the files of the run they belong to."""
    assert tag_of(argv) == ""


def test_the_full_option_set_gives_one_readable_name():
    argv = ["--max-structures", "40", "--rho", "0.1", "--folds", "5", "--seed", "3"]
    assert tag_of(argv) == "_n40_rho0.1_folds5_seed3"


# ------------------------------------------------------------------ one command per stored result

def test_no_two_documented_commands_write_the_same_file():
    commands = benchmark().compute
    for i, first in enumerate(commands):
        for second in commands[i + 1:]:
            shared = files_of(first) & files_of(second)
            assert not shared, (f"`{' '.join(first) or 'no arguments'}` and "
                                f"`{' '.join(second) or 'no arguments'}` both write "
                                + ", ".join(sorted(shared)))


def test_every_stored_result_has_a_documented_command():
    missing = sorted(stored_files() - documented_files())
    assert not missing, "stored results no documented command rebuilds: " + ", ".join(missing)


def test_every_documented_command_has_stored_results():
    extra = sorted(documented_files() - stored_files())
    assert not extra, ("documented commands whose results the repository does not keep: "
                       + ", ".join(extra))


# ------------------------------------------------------------------ the provenance file

def test_the_stored_results_say_which_command_produced_them():
    prov = rl.run_provenance(RESULTS)
    assert prov is not None, "results/litraj/run.json is missing"
    assert tag_of(prov["argv"]) == "", "run.json records a command with another tag"
    assert prov["prepared"] == "benchmarks/results/litraj/prepared.json"
    assert prov["folds"] == rl.FOLDS and prov["rho"] == list(rl.RHOS)


def test_the_provenance_matches_the_public_result_it_describes():
    prov = rl.run_provenance(RESULTS)
    with open(os.path.join(RESULTS, "public.json"), encoding="utf-8") as fh:
        pub = json.load(fh)
    assert pub["folds"] == prov["folds"]
    assert pub["r_max"] == prov["r_max"]
    assert pub["site_r"] == prov["site_r"]
    assert sorted({m["rho"] for m in pub["models"] if m.get("rho") is not None}) \
        == sorted(prov["rho"])


def test_the_table_prints_the_command_that_produced_it(tmp_path):
    """A reader of the table can repeat the run, and a table with no command says so."""
    out = tmp_path / "litraj"
    out.mkdir()
    with open(os.path.join(RESULTS, "public.json"), encoding="utf-8") as fh:
        (out / "public.json").write_text(fh.read(), encoding="utf-8")
    text = rl.write_table(str(out))
    assert "The command that produced these result files was not recorded." in text
    with open(out / "run.json", "w", encoding="utf-8") as fh:
        json.dump({"argv": [], "tag": "", "prepared": "benchmarks/results/litraj/prepared.json",
                   "seed": 0, "bootstrap": 1000, "folds": 10}, fh)
    text = rl.write_table(str(out))
    assert "`python benchmarks/run_litraj.py`" in text
    assert "`benchmarks/results/litraj/prepared.json`" in text


# ------------------------------------------------------------------ the sample it reads

def test_the_prepared_sample_names_its_source_and_licence():
    """The labels are someone else's work, so the file that holds them says whose."""
    with open(rl.PREPARED, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["source"] == "https://github.com/AIRI-Institute/LiTraj"
    assert doc["licence"] == "MIT"
    assert "npj Comput. Mater." in doc["reference"]
    assert doc["n_index_hops"] == 122421
