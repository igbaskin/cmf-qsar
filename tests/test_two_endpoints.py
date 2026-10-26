"""Two endpoints on the command line (plan item SC1).

`cmfqsar/multitask.py` could be reached only from a benchmark script or from Python. These tests
drive the command line: one file with two columns, one model that carries both endpoints, and the
selectivity that is the difference of the two. The properties worth fixing are that the saved
model predicts the second endpoint from the same kernel as the first, that the selectivity is
exactly the difference of the two predictions, and that a co-field asked for by endpoint is the
co-field of that endpoint -- for the selectivity, the difference of the other two, which follows
from the model being linear in the field.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.cli import main  # noqa: E402
from cmfqsar.dataio import load_dataset, load_model  # noqa: E402
from cmfqsar.fields import AtomTable, build_fields  # noqa: E402

SMILES = ["c1ccccc1C(=O)O", "c1ccc(Cl)cc1C(=O)O", "c1ccc(C)cc1C(=O)O", "c1ccc(O)cc1C(=O)O",
          "c1ccc(N)cc1C(=O)O", "c1ccc(F)cc1C(=O)O", "c1ccc(CC)cc1C(=O)O",
          "c1ccc(OC)cc1C(=O)O", "c1ccc(Br)cc1C(=O)O", "c1ccc(C(C)C)cc1C(=O)O"]


def _set(path, second_on=None):
    """A small series with a primary activity and a second endpoint on `second_on` molecules."""
    if second_on is None:
        second_on = range(len(SMILES))
    with Chem.SDWriter(path) as w:
        for k, smi in enumerate(SMILES):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            p = AllChem.ETKDGv3()
            p.randomSeed = k + 1
            AllChem.EmbedMolecule(m, p)
            AllChem.MMFFOptimizeMolecule(m)
            m.SetProp("_Name", f"m{k}")
            m.SetProp("activity", f"{4.0 + 0.5 * k:.2f}")
            if k in second_on:
                # correlated with the first, but not identical to it
                m.SetProp("other", f"{3.5 + 0.4 * k + 0.1 * (k % 3):.2f}")
            w.write(m)
    return path


def _build(tmp, extra=(), second_on=None, name="m.cmf"):
    data = _set(os.path.join(str(tmp), "train.sdf"), second_on=second_on)
    out = os.path.join(str(tmp), name)
    main(["build", data, "--activity", "activity", "--fields", "electrostatic,steric",
          "--alphas", "0.5", "--lams", "0.1", "--folds", "3", "--test-fraction", "0",
          "--save", out] + list(extra))
    return data, out


# ------------------------------------------------------------------- the model carries both


def test_a_model_built_with_a_second_endpoint_carries_it(tmp_path, capsys):
    data, path = _build(tmp_path, ["--second-endpoint", "other"])
    text = capsys.readouterr().out
    assert "Second endpoint 'other'" in text
    assert "task correlation" in text
    model, extra = load_model(path)
    second = extra["second_endpoint"]
    assert second["column"] == "other"
    assert second["names"][1] == "other"
    assert len(second["coef"]) == model.train.n_mol
    assert len(second["selectivity_coef"]) == model.train.n_mol
    assert -1.0 <= second["rho"] <= 1.0
    assert [t["task"] for t in second["scores"]] == ["activity", "other"]


def test_the_second_endpoint_predicts_from_the_same_kernel(tmp_path):
    data, path = _build(tmp_path, ["--second-endpoint", "other"])
    model, extra = load_model(path)
    second = extra["second_endpoint"]
    ds = load_dataset(data)
    mf = [build_fields(m, model.fieldset, n) for m, n in zip(ds.mols, ds.names)]
    table = AtomTable.from_molecules(mf)
    K = model.test_kernel(table)
    y1 = model.predict(table)
    y2 = K @ np.asarray(second["coef"], float) + second["bias"]
    sel = K @ np.asarray(second["selectivity_coef"], float) + second["selectivity_bias"]
    # the selectivity is the difference of the two, by construction and to the last digit
    assert sel == pytest.approx(y1 - y2, rel=1e-9, abs=1e-12)


def test_the_correlation_can_be_given_instead_of_fitted(tmp_path, capsys):
    _build(tmp_path, ["--second-endpoint", "other", "--task-rho", "0.5"])
    text = capsys.readouterr().out
    assert "task correlation +0.500, given" in text


def test_a_correlation_of_zero_is_the_two_single_fits(tmp_path):
    """With no correlation the two endpoints decouple, so the primary fit must not move."""
    _d, a = _build(tmp_path, ["--second-endpoint", "other", "--task-rho", "0.0"], name="a.cmf")
    _d, b = _build(tmp_path, name="b.cmf")
    ma, _ = load_model(a)
    mb, _ = load_model(b)
    assert np.asarray(ma.coef) == pytest.approx(np.asarray(mb.coef), rel=1e-6, abs=1e-9)
    assert ma.bias == pytest.approx(mb.bias, rel=1e-6, abs=1e-9)


def test_an_endpoint_that_is_not_there_is_refused_by_name(tmp_path):
    with pytest.raises(SystemExit, match="only 0 training molecules carry it"):
        _build(tmp_path, ["--second-endpoint", "nothing_like_this"])


def test_too_few_measurements_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="carry it"):
        _build(tmp_path, ["--second-endpoint", "other"], second_on=[0, 1])


def test_a_partly_measured_second_endpoint_is_the_point_of_the_thing(tmp_path, capsys):
    """Most molecules carry one endpoint only; that is what the two-task model exists for."""
    _build(tmp_path, ["--second-endpoint", "other"], second_on=[0, 2, 4, 6, 8])
    text = capsys.readouterr().out
    assert "5 of 10 training molecules carry it" in text


# ------------------------------------------------------------------------- predict and cofield


def test_predict_reports_both_endpoints_and_their_difference(tmp_path, capsys):
    data, path = _build(tmp_path, ["--second-endpoint", "other"])
    out = os.path.join(str(tmp_path), "p.csv")
    capsys.readouterr()
    main(["predict", path, data, "--out", out])
    text = capsys.readouterr().out
    assert "Both endpoints" in text and "difference" in text
    with open(out) as fh:
        header = fh.readline().strip().split(",")
        first = fh.readline().strip().split(",")
    assert "other" in header and "difference" in header
    cols = dict(zip(header, first))
    assert float(cols["difference"]) == pytest.approx(
        float(cols["predicted"]) - float(cols["other"]), abs=1e-3)


def test_a_single_endpoint_model_predicts_as_before(tmp_path, capsys):
    data, path = _build(tmp_path)
    capsys.readouterr()
    main(["predict", path, data])
    assert "Both endpoints" not in capsys.readouterr().out


def test_the_cofield_of_the_selectivity_is_the_difference_of_the_two(tmp_path, capsys):
    """The model is linear in the field, so the maps subtract exactly as the predictions do."""
    from cmfqsar.cofields import cofield_volume, make_grid
    data, path = _build(tmp_path, ["--second-endpoint", "other"])
    model, extra = load_model(path)
    second = extra["second_endpoint"]
    grid = make_grid(model.train.coords, margin=2.0, spacing=1.5)
    f = [sp.name for sp in model.fieldset.specs].index("steric")
    primary = cofield_volume(model, f, grid)
    model.coef = np.asarray(second["coef"], float)
    other = cofield_volume(model, f, grid)
    model.coef = np.asarray(second["selectivity_coef"], float)
    selectivity = cofield_volume(model, f, grid)
    assert selectivity == pytest.approx(primary - other, rel=1e-8, abs=1e-10)


def test_the_cofield_command_takes_an_endpoint(tmp_path, capsys):
    data, path = _build(tmp_path, ["--second-endpoint", "other"])
    capsys.readouterr()
    main(["cofield", path, "--fields", "steric", "--spacing", "1.5", "--margin", "2",
          "--z", "0", "--endpoint", "selectivity"])
    text = capsys.readouterr().out
    assert "co-fields of 'activity - other'" in text
    main(["cofield", path, "--fields", "steric", "--spacing", "1.5", "--margin", "2",
          "--z", "0", "--endpoint", "second"])
    assert "co-fields of 'other'" in capsys.readouterr().out


def test_an_endpoint_asked_of_a_single_endpoint_model_is_refused(tmp_path):
    _data, path = _build(tmp_path)
    with pytest.raises(SystemExit, match="needs a model built with --second-endpoint"):
        main(["cofield", path, "--fields", "steric", "--spacing", "1.5", "--margin", "2",
              "--endpoint", "second"])


def test_the_example_table_can_be_rebuilt_from_the_shipped_files(tmp_path):
    """The two-endpoint example is a join of two tables that are already in the repository."""
    import runpy
    root = os.path.join(os.path.dirname(__file__), "..")
    out = os.path.join(str(tmp_path), "joined.csv")
    sys.argv = ["make_two_endpoint_example.py", "--out", out,
                "--primary", "gamma_cyclodextrin.csv", "--second", "alpha_cyclodextrin.csv",
                "--column", "alpha_logK"]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(os.path.join(root, "examples", "make_two_endpoint_example.py"),
                       run_name="__main__")
    assert exc.value.code == 0
    with open(out) as fh:
        header = fh.readline().strip().split(",")
        rows = fh.readlines()
    assert "alpha_logK" in header and "activity" in header
    assert len(rows) > 100
    assert sum(1 for r in rows if r.strip().split(",")[header.index("alpha_logK")]) > 10
