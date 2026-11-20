"""The runner for boxes on another machine, ``benchmarks/colab/colab_transport.py``, its bundle
script and its notebook, without a simulation.

A box is replaced by a small folder with a record, so that the order, the skipping, the copy,
the time limit and the log are tested on any machine.
"""
import json
import os
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "colab"))

import colab_transport as C  # noqa: E402
import make_bundle as B  # noqa: E402
import run_diffusion as R  # noqa: E402

NOTEBOOK = os.path.join(ROOT, "benchmarks", "colab", "cmf_transport_colab.ipynb")


def fake_simulate(cache, clock=None, step=0.0):
    """A function that makes the folder and the record of a box, and moves a fake clock."""
    made = []

    def simulate(name):
        directory = R.run_dir(name, str(cache))
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "centres.npz"), "wb") as fh:
            fh.write(b"centres")
        with open(os.path.join(directory, "run.json"), "w", encoding="utf-8") as fh:
            json.dump({"seconds": 900.0, "equilibration_seconds": 120.0, "ns_per_day": 380.0,
                       "atoms": 15000, "pairs": 100, "production_ps": 4000.0}, fh)
        made.append(name)
        if clock is not None:
            clock[0] += step
    return simulate, made


def test_the_transfer_list_has_the_21_boxes_and_the_order_is_reversed():
    names = C.names_for("transfer")
    assert len(names) == 21 and names == list(R.FORMULATION_NAMES)
    with pytest.raises(SystemExit, match="unknown run name"):
        C.names_for("form_EMC100,no_such_run")
    assert C.names_for("form_EMC100,form_DMC100") == ["form_EMC100", "form_DMC100"]
    assert C.names_for("rest") == list(R.REST_NAMES) and len(C.names_for("rest")) == 40
    assert C.names_for("all") == list(R.FORMULATION_NAMES) + list(R.REST_NAMES)


def test_a_box_in_the_store_is_left_out(tmp_path):
    store = tmp_path / "store"
    names = ["form_EMC100", "form_DMC100", "form_EC50_EMC50"]
    done = store / C.box_dirname("form_DMC100")
    done.mkdir(parents=True)
    (done / "run.json").write_text("{}")
    # a folder without a record is not a finished box
    half = store / C.box_dirname("form_EC50_EMC50")
    half.mkdir()
    todo, there = C.plan(names, str(store))
    assert there == ["form_DMC100"]
    assert todo == ["form_EC50_EMC50", "form_EMC100"]            # reversed
    todo, _ = C.plan(names, str(store), reverse=False)
    assert todo == ["form_EMC100", "form_EC50_EMC50"]


def test_a_copy_is_complete_or_not_there(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "run.json").write_text("{}")
    (src / "centres.npz").write_bytes(b"x" * 100)
    # an old, unfinished copy and a leftover of a cut-off copy are both replaced
    dst.mkdir()
    (dst / "old.txt").write_text("old")
    part = tmp_path / "dst.partial"
    part.mkdir()
    (part / "junk").write_text("junk")
    C.copy_box(str(src), str(dst))
    assert sorted(os.listdir(dst)) == ["centres.npz", "run.json"]
    assert not part.exists()


def test_boxes_are_run_copied_and_logged(tmp_path):
    store, cache = tmp_path / "store", tmp_path / "cache"
    simulate, made = fake_simulate(cache)
    names = ["form_EMC100", "form_DMC100", "form_EC50_EMC50"]
    results, left = C.run_boxes(names, str(cache), str(store), simulate=simulate, gpu="Test GPU")
    assert made == ["form_EC50_EMC50", "form_DMC100", "form_EMC100"] and left == []
    for n in names:
        assert C.is_done(str(store / C.box_dirname(n)))
    lines = [json.loads(x) for x in (store / C.LOG_NAME).read_text().splitlines()]
    assert [x["run"] for x in lines] == made
    assert lines[0]["gpu"] == "Test GPU" and lines[0]["ns_per_day"] == 380.0
    assert lines[0]["atoms"] == 15000 and lines[0]["production_s"] == 900.0
    # a second session finds everything there and runs nothing
    simulate2, made2 = fake_simulate(cache)
    again, left2 = C.run_boxes(names, str(tmp_path / "cache2"), str(store), simulate=simulate2,
                               gpu="")
    assert again == [] and made2 == [] and left2 == []


def test_the_number_of_boxes_and_the_time_are_limited(tmp_path):
    names = ["form_EMC100", "form_DMC100", "form_EC50_EMC50", "form_EC60_PC40"]
    store, cache = tmp_path / "store", tmp_path / "cache"
    simulate, made = fake_simulate(cache)
    results, left = C.run_boxes(names, str(cache), str(store), max_boxes=2, simulate=simulate,
                                gpu="")
    assert len(results) == 2 and len(left) == 2
    # one box takes an hour on a fake clock; 2.5 hours allow two boxes: after the second, the
    # time used (2 h) plus a fifth of the longest box more than the whole hour is over the limit
    store2, cache2 = tmp_path / "store2", tmp_path / "cache2"
    clock = [0.0]
    simulate2, made2 = fake_simulate(cache2, clock, step=3600.0)
    results, left = C.run_boxes(names, str(cache2), str(store2), max_hours=2.5,
                                simulate=simulate2, clock=lambda: clock[0], gpu="")
    assert len(results) == 2 and len(made2) == 2 and len(left) == 2
    # the first box is always started, whatever the limit
    store3, cache3 = tmp_path / "store3", tmp_path / "cache3"
    clock = [0.0]
    simulate3, made3 = fake_simulate(cache3, clock, step=7200.0)
    results, _ = C.run_boxes(names, str(cache3), str(store3), max_hours=1.0, simulate=simulate3,
                             clock=lambda: clock[0], gpu="")
    assert len(results) == 1


def test_the_dry_run_names_what_it_would_do(tmp_path, capsys):
    store = tmp_path / "store"
    assert C.main(["--store", str(store), "--runs", "form_EMC100,form_DMC100",
                   "--dry-run"]) == 0
    said = capsys.readouterr().out
    assert "2 runs: 0 already in the store, 2 to run" in said
    assert said.index("to run: form_DMC100") < said.index("to run: form_EMC100")
    assert not store.exists()                                   # a dry run writes nothing


def test_the_bundle_holds_what_the_benchmark_needs(tmp_path):
    out, n, size = B.make_bundle(str(tmp_path / "bundle.zip"))
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        info = json.loads(z.read("bundle_info.json"))
    for needed in ("cmfqsar/__init__.py", "cmfqsar/transport.py", "cmfqsar/electrolyte.py",
                   "benchmarks/run_diffusion.py", "benchmarks/colab/colab_transport.py",
                   "examples/electrolyte/formulations.csv"):
        assert needed in names, needed
    assert not any("__pycache__" in x or x.endswith(".pyc") for x in names)
    assert not any(x.startswith(("tests/", "docs/", "benchmarks/cache/", "benchmarks/results/"))
                   for x in names)
    assert n == len(names) - 1 and size < 20e6
    assert info["files"] == n and "commit" in info


def test_the_notebook_is_valid_and_its_code_compiles():
    with open(NOTEBOOK, encoding="utf-8") as fh:
        nb = json.load(fh)
    assert nb["nbformat"] == 4 and nb["metadata"]["accelerator"] == "GPU"
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) >= 7
    ids = [c["id"] for c in nb["cells"]]
    assert len(set(ids)) == len(ids)
    for c in code:
        source = "".join(c["source"])
        if source.lstrip().startswith("%%"):                     # a shell cell
            assert "micromamba" in source and "openmm=8.3.1" in source
            continue
        compile(source, "<cell>", "exec")
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    # the pins match the environment of the author's machine
    for pin in ("openmm=8.3.1", "openmmforcefields=0.15.1", "openff-toolkit=0.18.0",
                "ambertools=24.8"):
        assert pin in text
    # the settings that the later cells read are all defined in the first code cell
    first = "".join(code[0]["source"])
    for name in ("DRIVE_FOLDER", "RUNS", "MAX_BOXES", "MAX_HOURS", "UNITS_PER_HOUR",
                 "END_RUNTIME_WHEN_DONE", "CUDA_VERSION"):
        assert name + " =" in first or name + "=" in first
