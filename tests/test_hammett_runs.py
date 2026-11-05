"""One documented command per stored result, for the Hammett benchmark (plan item G6).

Rule 6 of the plan says that one command rebuilds every table. For a script that is run several
times with different options, that holds only if two things are true:

* every option that changes a number enters the name of the files the run writes, so that two runs
  never write over each other; and
* the registry in ``benchmarks/run_all.py`` lists one command for each stored result, so that
  ``cmf benchmark --compute hammett`` rebuilds what the repository holds and nothing else.

Neither was true before this item. ``--sdf examples/hammett/hammett.sdf`` was exempt from the tag,
so it wrote over the default run's ``summary.csv``; the protocol options did not enter the tag at
all, so a run over one target replaced the summary of a run over three; and of the four commands in
the registry two were the same command, one wrote a file the repository does not keep, and the four
ensemble results in the table came from commands the registry did not list.
"""
import csv
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "benchmarks")
RESULTS = os.path.join(BENCH, "results", "hammett")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

import run_all                                                      # noqa: E402
import run_hammett as rh                                            # noqa: E402


def tag_of(argv):
    return rh.run_tag(rh.build_parser().parse_args(argv))


def registry_tags():
    bench = next(b for b in run_all.registry() if b.name == "hammett")
    return [tag_of(argv) for argv in bench.compute]


def stored_tags():
    """The tags of the summary files the repository keeps, from git rather than from the disk.

    A file left behind by an exploratory run would otherwise look like a stored result.
    """
    import subprocess
    out = subprocess.check_output(["git", "ls-files", "benchmarks/results/hammett"],
                                  cwd=ROOT, text=True)
    return sorted(f.split("/")[-1][len("summary"):-len(".csv")]
                  for f in out.split() if "/summary" in f and f.endswith(".csv"))


# ------------------------------------------------------------------ the tag names the run

def test_the_default_run_has_the_empty_tag():
    """`summary.csv` keeps its name, so the stored file names do not change with this item."""
    assert tag_of([]) == ""


def test_the_two_names_of_the_single_conformer_set_give_one_tag():
    """``--single`` and the path of the same file are the same data set, so one tag."""
    assert tag_of(["--single"]) == tag_of(["--sdf", "examples/hammett/hammett.sdf"]) == "_single"


def test_the_single_conformer_set_no_longer_writes_over_the_default():
    """The fault: by its path, the single-conformer set used to take the default's empty tag."""
    assert tag_of(["--sdf", "examples/hammett/hammett.sdf"]) != tag_of([])


def test_the_default_data_set_named_by_its_path_is_still_the_default():
    assert tag_of(["--sdf", "examples/hammett/hammett_sym.sdf"]) == ""


@pytest.mark.parametrize("argv,expected", [
    (["--targets", "sigma_m"], "_sigma_m"),
    (["--targets", "sigma_m,pi"], "_sigma_m-pi"),
    (["--folds", "5"], "_folds5"),
    (["--no-ablation"], "_noablation"),
    (["--alphas", "1.0"], "_grid"),
    (["--lams", "0.1,1"], "_grid"),
])
def test_an_option_that_changes_the_numbers_changes_the_tag(argv, expected):
    assert tag_of(argv) == expected
    assert tag_of(argv) != tag_of([])


def test_the_full_option_set_gives_one_readable_name():
    argv = ["--sdf", "examples/hammett/hammett_ens_qm.sdf", "--charges-prop", "q_hirshfeld",
            "--weighting", "boltzmann", "--kt", "1.2"]
    assert tag_of(argv) == "_ens_qm_q_hirshfeld_boltzmann_kt1.2"


# ------------------------------------------------------------------ one command per stored result

def test_no_two_documented_commands_write_the_same_files():
    tags = registry_tags()
    assert len(tags) == len(set(tags)), f"two commands share a tag: {sorted(tags)}"


def test_every_stored_summary_has_a_documented_command():
    missing = sorted(set(stored_tags()) - set(registry_tags()))
    assert not missing, ("stored results no documented command rebuilds: "
                         + ", ".join(f"summary{t}.csv" for t in missing))


def test_every_documented_command_has_a_stored_summary():
    extra = sorted(set(registry_tags()) - set(stored_tags()))
    assert not extra, ("documented commands whose results the repository does not keep: "
                       + ", ".join(f"summary{t}.csv" for t in extra))


# ------------------------------------------------------------------ the provenance file

def test_every_stored_summary_says_which_command_produced_it():
    for tag in stored_tags():
        prov = rh.run_provenance(RESULTS, tag)
        assert prov is not None, f"summary{tag}.csv has no run{tag}.json"
        assert tag_of(prov["argv"]) == tag, f"run{tag}.json records a command with another tag"


def test_the_provenance_counts_the_rows_of_its_summary():
    for tag in stored_tags():
        prov = rh.run_provenance(RESULTS, tag)
        with open(os.path.join(RESULTS, f"summary{tag}.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert prov["rows"] == len(rows)
        assert sorted({r["target"] for r in rows}) == sorted(prov["targets"])


def test_the_table_prints_the_command_of_every_block(tmp_path):
    """A reader of the table can repeat any block, and a block with no command says so."""
    out = tmp_path / "hammett"
    out.mkdir()
    for name in ("summary.csv", "summary_ens.csv"):
        with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
            (out / name).write_text(fh.read(), encoding="utf-8")
    with open(out / "run_ens.json", "w", encoding="utf-8") as fh:
        json.dump({"argv": ["--sdf", "examples/hammett/hammett_ens.sdf"], "tag": "_ens",
                   "dataset": "examples/hammett/hammett_ens.sdf", "ensemble": True,
                   "charges": "gasteiger", "weighting": "uniform", "folds": 10}, fh)
    text = rh.report(str(out), str(out / "table.md"))
    assert "`python benchmarks/run_hammett.py --sdf examples/hammett/hammett_ens.sdf`" in text
    assert ("Data set `examples/hammett/hammett_ens.sdf`, charges gasteiger, uniform weights over "
            "the records of each substituent, 10-fold cross-validation.") in text
    assert "The command that produced this file was not recorded." in text
