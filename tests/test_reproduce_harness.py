"""The reproduction run: what it measures, what it compares, and what makes it fail.

The run of 7 November 2026 was green and had measured almost nothing. Three faults together
produced that: most scripts keep the result files a clone carries and recompute only what is
missing, ``--compute`` rewrote the tables and the manifest before the check that was meant to
compare them, and a script that failed left the committed results in place with no more than a
line in the log. These tests are what makes the repaired run trustworthy: that ``--fresh``
reaches the scripts that have the option, that ``--no-rebuild`` leaves the comparison to be made,
that a failed script stops the run, and that the groups of the run fit the job they run in.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import run_all  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "reproduce.yml")


def workflow() -> str:
    return run_all._read(WORKFLOW)


def fake(name, script, compute=None, **kw):
    """A registry entry that names a real script, so that --force is read from a real file."""
    return run_all.Benchmark(name, "a benchmark for a test", script, name,
                             compute=compute if compute is not None else [[]],
                             minutes="5", **kw)


# --------------------------------------------------------------------------- the groups

def test_every_benchmark_the_rule_admits_is_in_exactly_one_group():
    """The fault the computed split removes: a benchmark added to the registry and to no group.

    Seven of them had collected that way -- anchors, cell_view, denmark, efield, phenazine,
    shells, surface_view -- while the groups were a list of names in the workflow. The rule is
    the same one step 7 of docs/release_checklist.md states.
    """
    groups = run_all.reproduce_groups()
    placed = [n for g in sorted(groups) for n in groups[g]]
    assert len(placed) == len(set(placed)), "a benchmark is in more than one group"
    for b in run_all.registry():
        try:
            short = float(b.minutes) <= run_all.REPRODUCE_MINUTES
        except ValueError:
            short = False
        admitted = b.needs is None and (short or b.name in run_all.REPRODUCE_ALSO)
        assert admitted == (b.name in placed), (
            f"{b.name}: needs={b.needs}, minutes={b.minutes}, "
            f"{'in' if b.name in placed else 'not in'} the run")


def test_no_group_holds_a_benchmark_the_runner_cannot_measure():
    """A benchmark that needs ORCA, xtb or data outside the repository would be skipped, not
    measured, and a skipped benchmark in a reproduction run is a table nobody compared."""
    needs = {b.name: b.needs for b in run_all.registry()}
    for g, names in sorted(run_all.reproduce_groups().items()):
        for n in names:
            assert needs[n] is None, f"group {g}: {n} needs {needs[n]}"


def test_the_same_registry_gives_the_same_groups():
    assert run_all.reproduce_groups() == run_all.reproduce_groups()


def test_a_group_fits_the_job_twice_over():
    """The minutes are the author's machine, so a group has to leave room for a slower one."""
    m = re.search(r"timeout-minutes:\s*(\d+)", workflow())
    assert m, "the workflow sets no timeout"
    timeout = int(m.group(1))
    assert timeout <= 360, "a GitHub runner stops a job after six hours"
    for g in sorted(run_all.reproduce_groups()):
        minutes = run_all.group_minutes(g)
        assert minutes > 0, f"group {g} has no recorded minutes"
        assert 2 * minutes <= timeout, (f"group {g} is {minutes:.0f} minutes, which does not fit "
                                        f"twice in {timeout}")


def test_the_workflow_runs_every_group_the_way_a_reproduction_run_has_to_be_run():
    text = workflow()
    m = re.search(r"matrix:\s*\n\s*group:\s*\[([0-9,\s]+)\]", text)
    assert m, "the workflow has no group matrix"
    groups = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
    assert groups == sorted(run_all.reproduce_groups()), \
        "the workflow and run_all.py do not agree on how many groups there are"
    assert "--reproduce ${{ matrix.group }}" in text, "the groups are not run by --reproduce"


def test_reproduce_group_refuses_a_group_that_does_not_exist():
    for key in (0, 99, "every", None):
        with pytest.raises(SystemExit):
            run_all.reproduce_group(key)


# --------------------------------------------------------------------------- --fresh

def test_accepts_force_reads_the_option_from_the_script():
    assert run_all.accepts_force("run_corrosion.py"), "run_corrosion.py has --force"
    assert not run_all.accepts_force("run_fluctuation.py"), "run_fluctuation.py has no --force"
    assert not run_all.accepts_force("run_no_such_script_xyz.py")


def commands(monkeypatch, benchmarks):
    """Let compute() run over `benchmarks` without running them; return the arguments it passed."""
    calls = []
    monkeypatch.setattr(run_all, "registry", lambda: list(benchmarks))
    monkeypatch.setattr(run_all.subprocess, "call", lambda cmd, cwd=None: calls.append(cmd) or 0)
    monkeypatch.setattr(run_all, "rebuild_tables", lambda *a, **k: ([], [], {}))
    monkeypatch.setattr(run_all, "write_manifest", lambda: None)
    return calls


def test_fresh_passes_force_to_the_scripts_that_have_it(monkeypatch):
    bench = [fake("a", "run_corrosion.py"), fake("b", "run_fluctuation.py"),
             fake("c", "run_corrosion.py", compute=[["--force"]])]
    calls = commands(monkeypatch, bench)
    run_all.compute(["a", "b", "c"], fresh=True, rebuild=False)
    assert [cmd[2:] for cmd in calls] == [["--force"], [], ["--force"]]


def test_without_fresh_the_scripts_are_left_to_reuse_what_is_on_disk(monkeypatch):
    calls = commands(monkeypatch, [fake("a", "run_corrosion.py")])
    run_all.compute(["a"], rebuild=False)
    assert [cmd[2:] for cmd in calls] == [[]]


def test_no_rebuild_leaves_the_tables_and_the_manifest_for_the_check(monkeypatch):
    done = []
    monkeypatch.setattr(run_all, "registry", lambda: [fake("a", "run_corrosion.py")])
    monkeypatch.setattr(run_all.subprocess, "call", lambda cmd, cwd=None: 0)
    monkeypatch.setattr(run_all, "rebuild_tables",
                        lambda *a, **k: (done.append("tables"), ([], [], {}))[1])
    monkeypatch.setattr(run_all, "write_manifest", lambda: done.append("manifest"))
    run_all.compute(["a"], rebuild=False)
    assert done == [], "--no-rebuild rebuilt something"
    run_all.compute(["a"])
    assert done == ["tables", "manifest"], "an ordinary run has to rebuild both"


def test_a_script_that_fails_fails_the_run(monkeypatch):
    """Its results are the committed ones, which would look like a measurement that did not move."""
    monkeypatch.setattr(run_all, "registry", lambda: [fake("a", "run_corrosion.py")])
    monkeypatch.setattr(run_all.subprocess, "call", lambda cmd, cwd=None: 3)
    monkeypatch.setattr(run_all, "rebuild_tables", lambda *a, **k: ([], [], {}))
    monkeypatch.setattr(run_all, "write_manifest", lambda: None)
    with pytest.raises(SystemExit) as exc:
        run_all.compute(["a"], rebuild=False)
    assert "exit 3" in str(exc.value)


def test_reproduce_measures_again_and_leaves_the_comparison(monkeypatch):
    """--reproduce GROUP is --compute with both options, which is what the workflow relies on."""
    seen = {}
    monkeypatch.setattr(run_all, "reproduce_groups", lambda: {1: ["a"]})
    monkeypatch.setattr(run_all, "registry", lambda: [fake("a", "run_corrosion.py")])
    monkeypatch.setattr(run_all, "compute",
                        lambda names, *a, **kw: seen.update(names=names, kw=kw))
    assert run_all.main(["--reproduce", "1"]) == 0
    assert seen["names"] == ["a"]
    assert seen["kw"] == {"fresh": True, "rebuild": False}


def test_fresh_without_compute_is_refused():
    with pytest.raises(SystemExit):
        run_all.main(["--fresh"])


# --------------------------------------------------------------------------- how far a table moved

def test_moved_numbers_counts_what_moved_and_gives_the_largest_difference():
    line = run_all._moved_numbers("| q2 | 0.300 |\n| rmse | 1.000 |\n",
                                  "| q2 | 0.278 |\n| rmse | 1.400 |\n")
    assert "2 of 3 numbers moved" in line, line
    assert "0.4" in line, line


def test_moved_numbers_notices_a_table_of_another_shape():
    assert "shape changed" in run_all._moved_numbers("1 2 3\n", "1 2 3 4\n")


def test_moved_numbers_tells_a_changed_sentence_from_a_changed_number():
    assert "no number moved" in run_all._moved_numbers("the steric field 1\n",
                                                       "the electrostatic field 1\n")


def test_moved_numbers_tells_a_new_table_apart():
    assert "did not exist" in run_all._moved_numbers(None, "1\n")


def test_the_rebuild_reports_how_far_the_table_of_a_benchmark_moved(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "RESULTS", str(tmp_path))
    monkeypatch.setattr(run_all, "SPLICES", [])
    table = tmp_path / "fake" / "table.md"
    table.parent.mkdir()
    table.write_text("| q2 | 0.300 |\n", encoding="utf-8")
    bench = fake("fake", "run_fake.py", tables=["fake/table.md"],
                 report=lambda: table.write_text("| q2 | 0.278 |\n", encoding="utf-8"))
    monkeypatch.setattr(run_all, "registry", lambda: [bench])
    written, changed, moved = run_all.rebuild_tables()
    assert [os.path.basename(c) for c in changed] == ["table.md"]
    assert "1 of 2 numbers moved" in moved[changed[0]], moved
    assert "0.022" in moved[changed[0]], moved
# --------------------------------------------------------------------------- cmf benchmark

def benchmark_argv(monkeypatch, argv):
    """What ``cmf benchmark ...`` would put on the command line of benchmarks/run_all.py."""
    import runpy

    from cmfqsar import cli

    seen = {}

    def instead_of_running(path, run_name=None):
        seen["argv"] = list(sys.argv)
        raise SystemExit(0)

    monkeypatch.setattr(runpy, "run_path", instead_of_running)
    cli.main(["benchmark", *argv])
    return seen["argv"][1:]


def test_the_names_reach_compute(monkeypatch):
    """The manual documents `cmf benchmark alignment poses`; the names were passed as positional
    arguments, which run_all.py does not take, and the command ended in "unrecognized
    arguments"."""
    assert benchmark_argv(monkeypatch, ["alignment", "poses"]) == \
        ["--compute", "alignment", "poses"]


def test_readme_asks_for_the_tables(monkeypatch):
    """--readme passed an option run_all.py has never had. The spliced passages, the README's
    among them, are rebuilt by --tables."""
    assert benchmark_argv(monkeypatch, ["--readme"]) == ["--tables"]


def test_the_options_of_a_reproduction_run_are_passed_through(monkeypatch):
    assert benchmark_argv(monkeypatch, ["--reproduce", "3"]) == ["--reproduce", "3"]
    assert benchmark_argv(monkeypatch, ["alignment", "--fresh", "--no-rebuild"]) == \
        ["--compute", "alignment", "--fresh", "--no-rebuild"]


# ------------------------------------------ the comparison with what the repository carries

def test_the_committed_text_is_what_the_repository_carries():
    text = run_all._committed(os.path.join(ROOT, "benchmarks", "run_all.py"))
    if text is None:
        pytest.skip("not a git checkout")
    assert "def reproduce_groups" in text


def test_a_file_the_repository_does_not_carry_has_no_committed_text(tmp_path):
    assert run_all._committed(str(tmp_path / "nothing.md")) is None


def test_the_rebuild_compares_with_the_commit_and_not_with_the_file_the_script_wrote(
        tmp_path, monkeypatch):
    """The fault the run of the night of 7 November 2026 showed.

    Most scripts write their own results/<name>/table.md while they measure, so by the time the
    check reads that file it is already the new table: a comparison with it compares the new
    numbers with themselves. Thirty of the 66 tables moved in that run and the comparison with
    the files on disk named six of them.
    """
    monkeypatch.setattr(run_all, "RESULTS", str(tmp_path))
    monkeypatch.setattr(run_all, "SPLICES", [])
    table = tmp_path / "fake" / "table.md"
    table.parent.mkdir()
    table.write_text("| q2 | 0.278 |\n", encoding="utf-8")   # the script has already run
    bench = fake("fake", "run_fake.py", tables=["fake/table.md"],
                 report=lambda: table.write_text("| q2 | 0.278 |\n", encoding="utf-8"))
    monkeypatch.setattr(run_all, "registry", lambda: [bench])
    monkeypatch.setattr(run_all, "_committed", lambda path: "| q2 | 0.300 |\n")
    assert run_all.rebuild_tables()[1] == [], "the file on disk was its own baseline"
    written, changed, moved = run_all.rebuild_tables(baseline="committed")
    assert [os.path.basename(c) for c in changed] == ["table.md"]
    assert "1 of 2 numbers moved" in moved[changed[0]], moved
    assert "0.022" in moved[changed[0]], moved


def test_a_baseline_the_rebuild_does_not_have_is_refused():
    with pytest.raises(ValueError):
        run_all.rebuild_tables(baseline="the last release")


def test_the_rebuild_says_so_when_the_repository_carries_no_version_of_a_table(
        tmp_path, monkeypatch, capsys):
    """A silent fallback to the file on disk would be the fault this option removes."""
    monkeypatch.setattr(run_all, "RESULTS", str(tmp_path))
    monkeypatch.setattr(run_all, "SPLICES", [])
    table = tmp_path / "fake" / "table.md"
    table.parent.mkdir()
    table.write_text("| q2 | 0.278 |\n", encoding="utf-8")
    bench = fake("fake", "run_fake.py", tables=["fake/table.md"],
                 report=lambda: table.write_text("| q2 | 0.278 |\n", encoding="utf-8"))
    monkeypatch.setattr(run_all, "registry", lambda: [bench])
    monkeypatch.setattr(run_all, "_committed", lambda path: None)
    run_all.rebuild_tables(baseline="committed")
    assert "carries no version of this table" in capsys.readouterr().out


def test_check_tables_finds_a_table_that_is_not_the_one_the_repository_carries(
        tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "RESULTS", str(tmp_path))
    (tmp_path / "fake").mkdir()
    table = tmp_path / "fake" / "table.md"
    table.write_text("| q2 | 0.278 |\n", encoding="utf-8")
    monkeypatch.setattr(run_all, "_committed", lambda path: "| q2 | 0.300 |\n")
    problems = run_all.check_tables()
    assert len(problems) == 1 and "0.022" in problems[0], problems
    assert run_all.check_tables(skip={run_all._label(str(table))}) == [], \
        "a table rebuild_tables has already reported is reported twice"


def test_a_table_that_differs_only_in_its_newlines_is_the_same_table(tmp_path, monkeypatch):
    """The same table written on Windows and on Linux is one table.

    benchmarks/results/threebody/table.md was committed with CRLF newlines, so on a Linux
    runner it showed as changed in all seven jobs although not one of its numbers had moved.
    """
    monkeypatch.setattr(run_all, "RESULTS", str(tmp_path))
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "table.md").write_text("| q2 | 0.278 |\n", encoding="utf-8",
                                                newline="\n")
    monkeypatch.setattr(run_all, "_committed", lambda path: "| q2 | 0.278 |\r\n")
    assert run_all.check_tables() == []


def test_a_table_the_repository_does_not_carry_is_not_a_difference(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "RESULTS", str(tmp_path))
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "table.md").write_text("| q2 | 0.278 |\n", encoding="utf-8")
    monkeypatch.setattr(run_all, "_committed", lambda path: None)
    assert run_all.check_tables() == []


def test_every_committed_table_has_the_same_newlines():
    """What made the CRLF table of threebody show as changed in every job of the run.

    Only the Markdown tables are compared as text, so only they have to agree on this.
    """
    try:
        done = run_all.subprocess.run(
            ["git", "-C", ROOT, "ls-files", "--eol", "benchmarks/results"],
            stdout=run_all.subprocess.PIPE, stderr=run_all.subprocess.DEVNULL)
    except OSError:
        pytest.skip("no git here")
    if done.returncode != 0:
        pytest.skip("not a git checkout")
    crlf = [ln.split("\t")[-1] for ln in done.stdout.decode("utf-8", "replace").splitlines()
            if ln.startswith("i/crlf") and ln.rstrip().endswith(".md")]
    assert crlf == [], ("these tables are committed with CRLF newlines, so a Linux runner "
                        "rewrites them whole: " + ", ".join(crlf))


def test_the_check_compares_with_the_commit_only_when_it_is_asked(monkeypatch):
    seen = {}

    def rebuild(*a, **kw):
        seen["kw"] = kw
        return [], [], {}

    monkeypatch.setattr(run_all, "rebuild_tables", rebuild)
    monkeypatch.setattr(run_all, "check_manifest", lambda: [])
    monkeypatch.setattr(run_all, "check_documents", lambda: [])
    monkeypatch.setattr(run_all, "check_tables", lambda skip=(): ["a table moved"])
    assert run_all.main(["--check"]) == 0, "a plain --check asks its own question"
    assert seen["kw"] == {"baseline": "disk"}
    assert run_all.main(["--check", "--against-commit"]) == 1, \
        "--against-commit did not reach the tables of benchmarks that rebuild none"
    assert seen["kw"] == {"baseline": "committed"}


def test_against_commit_is_an_option_of_the_check():
    with pytest.raises(SystemExit):
        run_all.main(["--against-commit"])


def test_the_workflow_compares_the_tables_with_the_commit():
    assert "--check --against-commit" in workflow(), \
        "the workflow would compare the new tables with themselves"


def test_the_run_has_what_the_two_benchmarks_that_failed_for_want_of_a_package_need():
    """protonation needed Dimorphite-DL and trajectory a trajectory reader; neither was there.

    Both are in the reproduction run, so the versions the run installs have to carry them.
    """
    names = [n for g in run_all.reproduce_groups().values() for n in g]
    assert "protonation" in names and "trajectory" in names
    lock = run_all._read(os.path.join(ROOT, "requirements-lock.txt")).lower()
    for package in ("dimorphite-dl", "mdtraj", "mdanalysis"):
        assert package in lock, f"the locked versions do not install {package}"


def test_the_cli_passes_the_comparison_through(monkeypatch):
    assert benchmark_argv(monkeypatch, ["--check", "--against-commit"]) == \
        ["--check", "--against-commit"]


def test_a_label_for_a_path_that_has_no_relative_form(monkeypatch):
    """What broke the Windows jobs: two drives have no relative path between them.

    pytest's ``tmp_path`` is on C: on a Windows runner and the checkout is on D:, so
    ``os.path.relpath`` raises and every caller that only wanted something to print failed with
    it. The raise is forced here, so the test asks the question on every platform.
    """
    def no_relative(path, start=None):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")

    monkeypatch.setattr(run_all.os.path, "relpath", no_relative)
    elsewhere = os.path.join(os.sep, "elsewhere", "table.md")
    assert run_all._label(elsewhere) == elsewhere
    assert run_all._label("results/table.md") == "results/table.md"
