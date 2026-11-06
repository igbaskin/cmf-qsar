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


# ------------------------------------------------------------------ the forest arms

def test_the_forest_is_given_the_same_folds_as_the_kernel():
    """The forest arms answer what the fitting method is worth, so only the fitting may differ."""
    import numpy as np

    from cmfqsar.atomtargets import AtomTargets
    from cmfqsar.localenv import choose_lam
    rng = np.random.RandomState(0)
    mol = np.repeat(np.arange(12), 3)
    X = rng.normal(size=(len(mol), 4))
    y = 2.0 * X[:, 0] + X[:, 1]
    t = AtomTargets(molecule=mol, atom=np.arange(len(mol)), y=y, n_mol=12, name="barrier")
    forest = rl.forest_result(X, t, folds=4, seed=0, name="forest")
    kernel = choose_lam(X @ X.T, t, folds=4, seed=0, name="kernel")
    assert forest.folds == kernel.folds
    assert forest.n_samples == kernel.n_samples == len(y)
    assert np.array_equal(forest.molecule, kernel.molecule)
    assert np.isnan(forest.lam), "a forest chooses no ridge"
    assert np.isfinite(forest.predicted).all(), "every hop is predicted out of its own fold"


def test_the_boosted_forest_is_a_second_fitting_method():
    import numpy as np

    from cmfqsar.atomtargets import AtomTargets
    rng = np.random.RandomState(1)
    mol = np.repeat(np.arange(10), 4)
    X = rng.normal(size=(len(mol), 3))
    y = X[:, 0] ** 2
    t = AtomTargets(molecule=mol, atom=np.arange(len(mol)), y=y, n_mol=10, name="barrier")
    a = rl.forest_result(X, t, folds=5, seed=0, name="a", kind="random forest")
    b = rl.forest_result(X, t, folds=5, seed=0, name="b", kind="boosted forest")
    assert not np.allclose(a.predicted, b.predicted), "the two methods are the same code path"


def test_a_learning_curve_too_short_to_read_makes_no_claim():
    """q2 cannot pass 1, so a rate that would carry the model past it says only that it cannot."""
    assert rl.projection([{"structures": 200, "q2": 0.55}]) is None
    per, steps, arrives = rl.projection([{"structures": 200, "q2": 0.50},
                                         {"structures": 400, "q2": 0.53}])
    assert per == pytest.approx(0.03)
    assert arrives == pytest.approx(0.53 + 0.03 * steps)
    assert rl.projection([{"structures": 25, "q2": 0.54},
                          {"structures": 50, "q2": 0.80}])[2] > 1.0


def test_the_table_does_not_print_an_impossible_q2(tmp_path):
    out = tmp_path / "litraj"
    out.mkdir()
    with open(os.path.join(RESULTS, "public.json"), encoding="utf-8") as fh:
        pub = json.load(fh)
    pub["learning_curve"] = [{"structures": 25, "q2": 0.54, "rmse": 1.2},
                             {"structures": 50, "q2": 0.80, "rmse": 1.0}]
    with open(out / "public.json", "w", encoding="utf-8") as fh:
        json.dump(pub, fh)
    text = rl.write_table(str(out))
    assert "too short to read" in text
    assert "carries the model to about q2 +1" not in text


def test_the_forest_is_handed_exactly_the_rows_the_kernel_sees():
    """The arm that isolates the fitting method may not be handed more than the kernel had."""
    import numpy as np

    from cmfqsar.localenv import count_kernel
    rng = np.random.RandomState(0)
    X = np.abs(rng.normal(size=(6, 5)))
    ones = np.ones(X.shape[1])
    rows = rl.kernel_rows(X, scale=ones)
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0), "every row is divided by its length"
    assert np.allclose(rows @ rows.T, count_kernel(X)), \
        "the product of these rows is the kernel itself, so a forest on them sees no more"
    stretched = X.copy()
    stretched[0] *= 3.0
    assert np.allclose(rl.kernel_rows(stretched, scale=ones)[0], rows[0]), \
        "the kernel cannot tell a row from a multiple of it, so neither may this"


def test_every_description_is_fitted_both_ways():
    """A forest arm for each description the table compares, and its paired interval beside it."""
    with open(os.path.join(RESULTS, "public.json"), encoding="utf-8") as fh:
        pub = json.load(fh)
    forests = pub["forests"]
    keys = {m["key"] for m in pub["models"]}
    pairs = {(p["a"], p["b"]) for p in pub["pairs"]}
    on = {f"invariant {pub['best_rho']}", "counts", "geometry"}
    assert {f["on"] for f in forests} == on
    for f in forests:
        assert f["on"] in keys, "a forest arm names a description the table does not hold"
        assert (f["key"], f["on"]) in pairs, "a forest arm has no paired interval"
        assert f["lam"] is None
        assert f["method"] in {name for name, _ in rl.FORESTS}
        assert f["rows"] in rl.FOREST_ROWS
    assert (pub["best_forest"], f"overlap {pub['best_rho']}") in pairs
    for key in on:
        arms = [f for f in forests if f["on"] == key]
        assert {f["method"] for f in arms} == {name for name, _ in rl.FORESTS}
        forms = {f["rows"] for f in arms}
        assert "as the kernel" in forms, \
            "without the kernel's own rows the arm does not separate the fitting method"
        assert len(arms) == len(forms) * len(rl.FORESTS)
    cosine = {f["rows"] for f in forests if f["on"].startswith("invariant")}
    assert cosine == set(rl.FOREST_ROWS), \
        "the local kernel divides out the row length, so both forms of its rows are measured"


def test_the_stored_result_is_strict_json():
    """A result file with a not-a-number in it cannot be read by anything but Python."""
    with open(os.path.join(RESULTS, "public.json"), encoding="utf-8") as fh:
        pub = json.load(fh)
    json.dumps(pub, allow_nan=False)


# ------------------------------------------------------------------ the nested kernel arms

def test_the_gaussian_kernel_is_a_function_of_the_linear_one():
    """The rows are unit vectors, so no feature matrix is needed to put a width on them."""
    import numpy as np

    from cmfqsar.localenv import count_kernel
    rng = np.random.RandomState(0)
    X = np.abs(rng.normal(size=(7, 4)))
    K = count_kernel(X)
    rows = rl.kernel_rows(X, scale=np.ones(X.shape[1]))
    for gamma in (0.25, 1.0, 4.0):
        G = rl.gaussian_kernel(K, gamma)
        d2 = ((rows[:, None, :] - rows[None, :, :]) ** 2).sum(axis=2)
        assert np.allclose(G, np.exp(-gamma * d2)), \
            "2 - 2K is the squared distance only for rows of unit length"
        assert np.allclose(np.diag(G), 1.0)
        assert (G > 0).all() and (G <= 1.0 + 1e-12).all()


def test_the_nested_arm_chooses_nothing_on_the_fold_it_reports():
    """A width chosen on the test fold would make the Gaussian row meaningless."""
    import numpy as np

    from cmfqsar.atomtargets import AtomTargets
    rng = np.random.RandomState(0)
    mol = np.repeat(np.arange(20), 2)
    X = rng.normal(size=(len(mol), 3))
    y = 2.0 * X[:, 0] + X[:, 1]
    t = AtomTargets(molecule=mol, atom=np.arange(len(mol)), y=y, n_mol=20, name="barrier")
    K = X @ X.T
    poisoned = K.copy()
    held = t.folds(5, 0)[0]
    # a setting that is useless everywhere except on one fold's own rows cannot be picked,
    # because the choice is made before that fold is touched
    arms = {"plain": K, "poisoned": poisoned}
    poisoned[np.ix_(held, held)] += 1e6
    r, picked = rl.nested_result(arms, t, folds=5, seed=0, name="nested", inner=3)
    assert {c["setting"] for c in picked} <= set(arms)
    assert len(picked) == r.folds
    assert np.isnan(r.lam), "the ridge differs from fold to fold, so no single one is reported"
    assert np.isfinite(r.predicted).all()
    one, _ = rl.nested_result({"plain": K}, t, folds=5, seed=0, name="one", inner=3)
    assert one.q2 > 0.9, "a linear target on a linear kernel is predicted well"


def test_the_nested_linear_arm_is_the_honest_form_of_a_first_table_row():
    """Choosing the ridge on the folds it reports is optimistic; the nested arm is not."""
    with open(os.path.join(RESULTS, "public.json"), encoding="utf-8") as fh:
        pub = json.load(fh)
    nested = pub["nested"]
    pairs = {(p["a"], p["b"]) for p in pub["pairs"]}
    keys = {m["key"] for m in pub["models"]}
    on = {f"invariant {pub['best_rho']}", "counts"}
    assert {m["on"] for m in nested} == on
    for m in nested:
        assert m["on"] in keys
        assert (m["key"], m["on"]) in pairs, "a nested arm has no paired interval"
        assert m["lam"] is None
        assert m["kernel"] in {"linear", "Gaussian"}
        assert len(m["chosen"]) == m["folds"], "one choice per outer fold"
        for c in m["chosen"]:
            assert c["setting"] == "linear" or c["setting"].startswith("gamma ")
    for key in on:
        assert (f"nested Gaussian {key}", f"nested linear {key}") in pairs, \
            "what the width is worth is the number this section exists for"
    widths = {c["setting"] for m in nested if m["kernel"] == "Gaussian" for c in m["chosen"]}
    assert widths <= {f"gamma {g:g}" for g in rl.GAMMAS}
