"""One documented command per stored result, for the hop benchmark (plan item G6).

Rule 6 of the plan says that one command rebuilds every table. For a script that is run several
times with different options, that holds only if two things are true:

* every option that changes a number enters the names of the files the run writes, so that two
  runs never write over each other; and
* the registry in ``benchmarks/run_all.py`` lists one command for each stored result, so that
  ``cmf benchmark --compute hops`` rebuilds what the repository holds and nothing else.

Neither was true for ``benchmarks/run_hops.py`` before this item. Every run wrote the same file
names, so ``--max-structures 60``, which the script's own usage lines offer as a quick look, wrote
its small numbers over the stored measurement; and the registry listed one command where the
repository keeps the results of two, so ``profile_fine.json``, the fine co-field profile the
manual draws, was rebuilt by nothing.
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "benchmarks")
RESULTS = os.path.join(BENCH, "results", "hops")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

import run_all                                                      # noqa: E402
import run_hops as rh                                               # noqa: E402


def tag_of(argv):
    return rh.run_tag(rh.build_parser().parse_args(argv))


def benchmark():
    return next(b for b in run_all.registry() if b.name == "hops")


def files_of(argv):
    """The files in results/hops that this command writes, by their names alone."""
    a = rh.build_parser().parse_args(argv)
    tag = rh.run_tag(a)
    if a.fine_profile:
        return {f"profile_fine{tag}.json", f"run_fine_profile{tag}.json"}
    return {f"public{tag}.json", f"hosts{tag}.json", f"disordered{tag}.json",
            f"log{tag}.txt", f"table{tag}.md", f"run{tag}.json"}


def documented_files():
    """Every file a command of the registry writes, the prepared input included."""
    b = benchmark()
    files = {os.path.basename(rh.PREPARED)} if b.prepare else set()
    for argv in b.compute:
        files |= files_of(argv)
    return files


def stored_files():
    """The files of results/hops the repository keeps, from git rather than from the disk.

    A file left behind by an exploratory run would otherwise look like a stored result.
    """
    out = subprocess.check_output(["git", "ls-files", "benchmarks/results/hops"],
                                  cwd=ROOT, text=True)
    return {f.split("/")[-1] for f in out.split()}


# ------------------------------------------------------------------ the tag names the run

def test_the_stored_measurement_has_the_empty_tag():
    """`public.json` keeps its name, so the stored file names do not change with this item."""
    assert tag_of([]) == ""


def test_the_prepared_set_named_by_its_path_is_still_the_default():
    """Two ways of naming one input give one tag."""
    assert tag_of(["--prepared", rh.PREPARED]) == ""
    assert tag_of(["--prepared", os.path.relpath(rh.PREPARED, ROOT)]) == ""


@pytest.mark.parametrize("argv,expected", [
    (["--max-structures", "60"], "_n60"),
    (["--max-per-structure", "12"], "_per12"),
    (["--rho", "0.25"], "_rho0.25"),
    (["--r-max", "8"], "_r8"),
    (["--folds", "5"], "_folds5"),
    (["--seed", "1"], "_seed1"),
    (["--bootstrap", "200"], "_boot200"),
    (["--site-r", "6"], "_siter6"),
    (["--prepared", "benchmarks/results/hops/prepared_sulfides.json"], "_sulfides"),
])
def test_an_option_that_changes_the_numbers_changes_the_tag(argv, expected):
    assert tag_of(argv) == expected
    assert tag_of(argv) != tag_of([])


def test_the_quick_look_of_the_usage_lines_no_longer_writes_over_the_measurement():
    """The fault: ``--max-structures 60`` wrote its ``public.json`` over the stored one."""
    assert files_of(["--max-structures", "60"]).isdisjoint(files_of([]))


@pytest.mark.parametrize("argv", [["--no-hosts"], ["--reuse-public"], ["--jobs", "4"]])
def test_choosing_which_parts_run_leaves_the_tag_alone(argv):
    """These options change no number, so they write the files of the run they belong to."""
    assert tag_of(argv) == ""


def test_the_full_option_set_gives_one_readable_name():
    argv = ["--max-structures", "60", "--rho", "0.1", "0.25", "--folds", "5", "--seed", "3"]
    assert tag_of(argv) == "_n60_rho0.1-0.25_folds5_seed3"


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
    prov = rh.run_provenance(RESULTS)
    assert prov is not None, "results/hops/run.json is missing"
    assert tag_of(prov["argv"]) == "", "run.json records a command with another tag"
    assert prov["prepared"] == "benchmarks/results/hops/prepared_mp.json"
    assert prov["folds"] == rh.FOLDS and prov["rho"] == list(rh.RHOS)


def test_the_fine_profile_says_which_command_produced_it():
    prov = rh.run_provenance(RESULTS, name="run_fine_profile")
    assert prov is not None, "results/hops/run_fine_profile.json is missing"
    assert "--fine-profile" in prov["argv"]
    assert tag_of(prov["argv"]) == ""


def test_the_provenance_matches_the_public_result_it_describes():
    prov = rh.run_provenance(RESULTS)
    with open(os.path.join(RESULTS, "public.json"), encoding="utf-8") as fh:
        pub = json.load(fh)
    assert pub["folds"] == prov["folds"]
    assert pub["r_max"] == prov["r_max"]
    assert sorted({m["rho"] for m in pub["models"] if m.get("rho") is not None}) \
        == sorted(prov["rho"])


def test_the_table_prints_the_command_that_produced_it(tmp_path):
    """A reader of the table can repeat the run, and a table with no command says so."""
    out = tmp_path / "hops"
    out.mkdir()
    with open(os.path.join(RESULTS, "public.json"), encoding="utf-8") as fh:
        (out / "public.json").write_text(fh.read(), encoding="utf-8")
    text = rh.write_table(str(out))
    assert "The command that produced these result files was not recorded." in text
    with open(out / "run.json", "w", encoding="utf-8") as fh:
        json.dump({"argv": [], "tag": "", "prepared": "benchmarks/results/hops/prepared_mp.json",
                   "seed": 0, "bootstrap": 1000, "folds": 10}, fh)
    text = rh.write_table(str(out))
    assert "`python benchmarks/run_hops.py`" in text
    assert "`benchmarks/results/hops/prepared_mp.json`" in text
    assert "The command that produced these result files was not recorded." not in text
