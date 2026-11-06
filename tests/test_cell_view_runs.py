"""One documented command per stored result, for the cell-view benchmark (plan item G6).

Rule 6 of the plan says that one command rebuilds every table.  For ``benchmarks/run_cell_view.py``
that means two things, checked here without running the benchmark: every option that changes a
number enters the names of the files a run writes, so that a quick run never writes over the
stored measurement; and the registry in ``benchmarks/run_all.py`` lists one command for every
stored result, so that ``cmf benchmark --compute cell_view`` rebuilds what the repository holds.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "benchmarks")
RESULTS = os.path.join(BENCH, "results", "cell_view")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

import run_all                                                      # noqa: E402
import run_cell_view as rc                                          # noqa: E402


def tag_of(argv):
    return rc.run_tag(rc.build_parser().parse_args(argv))


def benchmark():
    return next(b for b in run_all.registry() if b.name == "cell_view")


def files_of(argv):
    tag = tag_of(argv)
    return {f"draw{tag}.json", f"read{tag}.json", f"log{tag}.txt", f"run{tag}.json",
            f"table{tag}.md"}


def test_the_stored_run_has_the_empty_tag():
    assert tag_of([]) == ""


def test_every_option_that_changes_a_number_changes_the_file_names():
    assert tag_of(["--cells", "2"]) == "_cells2"
    assert tag_of(["--limit", "40"]) == "_n40"
    assert tag_of(["--spacing", "0.3"]) == "_sp0.3"
    assert tag_of(["--limit", "40", "--spacing", "0.3"]) == "_n40_sp0.3"


def test_the_options_that_change_no_number_leave_the_names_alone():
    assert tag_of(["--stage", "draw"]) == ""
    assert tag_of(["--force"]) == ""
    assert tag_of(["--report"]) == ""


def test_the_registry_lists_one_command_per_stored_result():
    b = benchmark()
    documented = set()
    for argv in b.compute:
        documented |= files_of(argv)
    stored = {f for f in os.listdir(RESULTS)} if os.path.isdir(RESULTS) else set()
    stored = {f for f in stored if f.endswith((".json", ".md", ".txt"))}
    assert stored, "the benchmark has no stored result"
    assert stored <= documented, f"stored by no documented command: {sorted(stored - documented)}"
    assert "cell_view/table.md" in b.tables
