"""The documents, checked against the program (plan item G4).

The rule of item G4 is that a feature lands with its manual section and its benchmark script in
the same commit. Two parts of that rule can be checked by a program, and these tests are what
makes the check itself trustworthy: that it passes on the repository as it stands, and that it
fails when a benchmark is not named in the manual or when the typeset manual is older than the
Markdown it was built from.
"""
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import run_all  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_documents_check_passes_on_the_repository():
    problems = run_all.check_documents()
    assert not problems, "\n".join(problems)


def test_every_benchmark_is_named_in_the_manual():
    """The same rule, stated as the question it answers, so a failure names the benchmark."""
    text = run_all._read(os.path.join(ROOT, "docs", "manual.md"))
    unnamed = [b.name for b in run_all.registry()
               if not any(m in text for m in (b.name, f"results/{b.results}/", b.script))]
    assert not unnamed, f"the manual names no measurement of: {unnamed}"


def test_the_check_notices_a_benchmark_the_manual_does_not_name(monkeypatch):
    fake = run_all.Benchmark("no_such_benchmark_xyz", "a benchmark nobody wrote about",
                             "run_no_such_benchmark_xyz.py", "no_such_benchmark_xyz")
    monkeypatch.setattr(run_all, "registry", lambda: [fake])
    problems = run_all.check_documents()
    assert any("no_such_benchmark_xyz" in p for p in problems), problems


def test_the_check_notices_a_manual_pdf_older_than_its_markdown(tmp_path, monkeypatch):
    """The stamp is the only thing that can say this: a PDF carries a build date of its own."""
    pytest.importorskip("reportlab")
    sys.path.insert(0, os.path.join(ROOT, "docs"))
    import make_manual

    source = str(tmp_path / "manual.md")
    out = str(tmp_path / "CMF_manual.pdf")
    with open(source, "w", encoding="utf-8") as fh:
        fh.write("# Title\n\nSubtitle\n\nA paragraph.\n")
    shutil.copyfile(os.path.join(ROOT, "docs", "manual.md"), out)   # any file will do as the PDF
    assert make_manual.check_stamp(source, out), "no stamp yet, and the check said nothing"

    make_manual.write_stamp(source, out)
    assert make_manual.check_stamp(source, out) == []

    with open(source, "a", encoding="utf-8") as fh:
        fh.write("\nOne more paragraph, after the PDF was made.\n")
    problems = make_manual.check_stamp(source, out)
    assert problems and "make_manual" in problems[0], problems


def test_the_registry_knows_where_every_table_is():
    """A benchmark with a report writes tables, and --check compares them with the result files."""
    for b in run_all.registry():
        if b.report is None:
            continue
        assert b.tables, f"{b.name} has a report and no table"
        for t in b.tables:
            assert os.path.exists(os.path.join(ROOT, "benchmarks", "results", t)), f"{b.name}: {t}"
