"""The co-pattern of an invariant kernel, as a curve (plan item NT9).

The invariant kernels have no co-field, so there is no map to look at.  What they have is a
weight per channel, and a channel is one pair of field types at one distance.  The chart draws
those weights against distance for one pair at a time, and these tests check the three things
that make it a reading rather than a picture:

* the weights it draws are the model's own: the bias plus the sum over every channel of weight
  times channel is the prediction, exactly;
* the curve holds the weights the ranked channel table names, and lies on the kernel's own
  distance grid;
* where a kernel resolves a distance further -- the surface kernel resolves it by the heights of
  both atoms -- a point is the sum over what is left, and the curve says how many were summed.

A kernel whose channels are not indexed by one distance -- the three-body channels, the overlap
kernel of the fields -- is refused with a message that says what to read instead; that is tested
too, because a chart that quietly draws the wrong slice is worse than no chart.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdkit import Chem, RDLogger                                        # noqa: E402

from cmfqsar.align import embed_3d                                      # noqa: E402
from cmfqsar.copattern import (VALUES, copattern, copattern_rows,       # noqa: E402
                               has_copattern, pair_labels)
from cmfqsar.fields import AtomTable, FieldSet, build_fields            # noqa: E402
from cmfqsar.model import CMFModel                                      # noqa: E402
from cmfqsar.periodic import channel_weights, top_channels              # noqa: E402

RDLogger.DisableLog("rdApp.*")

SMILES = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1O", "c1ccccc1N", "c1ccccc1Cl", "c1ccccc1F",
          "c1ccc(C)cc1C", "c1ccccc1CCC", "c1ccccc1OC", "c1ccccc1C(=O)O"]


def molecules():
    return [embed_3d(Chem.AddHs(Chem.MolFromSmiles(s))) for s in SMILES]


@pytest.fixture(scope="module")
def fitted():
    """A model on the molecular autocorrelation: the plainest kernel that has no co-field."""
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(molecules())]
    y = np.arange(len(mf), dtype=float)
    model = CMFModel(fieldset=fs, method="krr", normalize=True,
                     kernel_source="autocorrelation", ac_sigma=0.5, ac_d_max=6.0,
                     params={"lam": 1e-3})
    table = AtomTable.from_molecules(mf)
    model.fit(table, y, cv_folds=0)
    return model, fs, table, y


@pytest.fixture(scope="module")
def surface_fitted():
    """A model on the surface kernel, which resolves a distance by the two heights."""
    from cmfqsar.planealign import plane_align
    mols = molecules()
    for m in mols:
        plane_align(m, pose="flat", height=2.0)
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    y = np.arange(len(mf), dtype=float)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="surface",
                     ac_sigma=0.75, ac_d_max=6.0, surf_z_min=1.0, surf_z_max=4.0,
                     surf_z_step=1.0, params={"lam": 1e-3})
    table = AtomTable.from_molecules(mf)
    model.fit(table, y, cv_folds=0)
    return model, fs, table, y


# --------------------------------------------------------------------------- #
# What the curve is
# --------------------------------------------------------------------------- #
def test_the_pairs_are_the_diagonal_ones_first(fitted):
    model, fs, _table, _y = fitted
    assert pair_labels(model, fs) == ["Electrostatic", "Steric", "Electrostatic x Steric"]


def test_the_curve_lies_on_the_kernels_own_distance_grid(fitted):
    model, fs, _table, _y = fitted
    cp = copattern(model, fs, 0)
    assert np.allclose(cp.x, model._ac_spec().centres)
    assert len(cp.y) == len(cp.x)
    assert cp.grouped == 1
    assert "distance" in cp.x_label and "A" in cp.x_label


def test_the_curve_holds_the_weights_the_table_names(fitted):
    """Point by point, the curve is the channel weight the ranked table prints."""
    model, fs, _table, _y = fitted
    w = channel_weights(model)
    centres = model._ac_spec().centres
    for pair in range(len(pair_labels(model, fs))):
        cp = copattern(model, fs, pair)
        expected = w[pair * len(centres):(pair + 1) * len(centres)]
        assert np.allclose(cp.weight, expected)


def test_the_ranked_table_and_the_curves_agree_on_the_strongest_channel(fitted):
    """The channel `top_channels` ranks first is a point of one of the curves, with its value."""
    model, fs, _table, _y = fitted
    from cmfqsar.invariant import channel_labels
    labels = channel_labels(fs, model._ac_spec())
    label, weight, spread = top_channels(model, labels, n=1)[0]
    found = []
    for pair in range(len(pair_labels(model, fs))):
        cp = copattern(model, fs, pair)
        for x, w, s in zip(cp.x, cp.weight, cp.spread):
            if f"@ {x:.2f} A" in label and cp.pair == label.split(" @ ")[0]:
                found.append((w, s))
    assert found, f"the ranked channel {label!r} is in no curve"
    assert any(w == pytest.approx(weight) and s == pytest.approx(spread) for w, s in found)


def test_the_weights_reproduce_the_prediction(fitted):
    """The identity that makes the weights a reading: bias + sum of weight times channel."""
    model, _fs, table, _y = fitted
    w = np.asarray(channel_weights(model), float)
    Z = np.asarray(model.ac_train_, float) * np.asarray(model.ac_column_scale, float)
    Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
    assert np.allclose(model.bias + Z @ w, model.predict(table), atol=1e-8)


def test_the_two_quantities_differ_and_are_both_carried(fitted):
    model, fs, _table, _y = fitted
    a = copattern(model, fs, 0, "weight")
    b = copattern(model, fs, 0, "weight x spread")
    assert np.allclose(a.weight, b.weight)              # the same model either way
    assert np.allclose(a.y, a.weight)
    assert np.allclose(b.y, b.moved)
    assert not np.allclose(a.y, b.y)                    # the spread is not one
    assert "weight" in a.y_label and "spread" in b.y_label


def test_an_unknown_quantity_is_refused(fitted):
    model, fs, _table, _y = fitted
    with pytest.raises(ValueError):
        copattern(model, fs, 0, "whatever")
    assert VALUES == ("weight", "weight x spread")


def test_a_pair_index_outside_the_set_is_refused(fitted):
    model, fs, _table, _y = fitted
    with pytest.raises(ValueError):
        copattern(model, fs, 99)


def test_the_rows_are_the_curve(fitted):
    model, fs, _table, _y = fitted
    cp = copattern(model, fs, 1)
    cols, rows = copattern_rows(model, fs, 1)
    assert len(rows) == len(cp.x)
    assert cols[0] == cp.x_label
    assert float(rows[0][0]) == pytest.approx(cp.x[0])
    assert float(rows[0][1]) == pytest.approx(cp.weight[0], abs=5e-5)


def test_the_description_names_the_largest_point(fitted):
    model, fs, _table, _y = fitted
    cp = copattern(model, fs, 0)
    k = int(np.argmax(np.abs(cp.y)))
    assert f"{cp.x[k]:.2f}" in cp.describe()
    assert cp.pair in cp.describe()


# --------------------------------------------------------------------------- #
# A kernel that resolves a distance further
# --------------------------------------------------------------------------- #
def test_the_surface_curve_sums_the_height_channels(surface_fitted):
    model, fs, _table, _y = surface_fitted
    spec = model._ac_spec()
    n_z = len(spec.heights)
    cp = copattern(model, fs, 0)
    assert cp.grouped == n_z * n_z > 1
    assert len(cp.x) == len(spec.centres)
    assert "in-plane distance" in cp.x_label
    assert "height channels" in cp.note
    # the point is the sum of the weights of the height channels at that distance
    w = np.asarray(channel_weights(model), float)
    block = w[:len(spec.centres) * cp.grouped].reshape(len(spec.centres), cp.grouped)
    assert np.allclose(cp.weight, block.sum(axis=1))


def test_the_surface_weights_also_reproduce_the_prediction(surface_fitted):
    model, _fs, table, _y = surface_fitted
    w = np.asarray(channel_weights(model), float)
    Z = np.asarray(model.ac_train_, float) * np.asarray(model.ac_column_scale, float)
    Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
    assert np.allclose(model.bias + Z @ w, model.predict(table), atol=1e-8)


# --------------------------------------------------------------------------- #
# A kernel that has no curve
# --------------------------------------------------------------------------- #
def test_the_overlap_kernel_has_a_co_field_and_no_curve():
    """The kernel of the fields is read as a map; the chart is for the ones that are not."""
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(molecules())]
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 1e-3})
    model.fit(AtomTable.from_molecules(mf), np.arange(len(mf), dtype=float), cv_folds=0)
    assert has_copattern(model, fs) is False
    with pytest.raises(ValueError, match="co-field"):
        copattern(model, fs, 0)


def test_the_three_body_channels_are_refused_with_a_reason():
    """A channel indexed by two radii and a separation is not a curve against one distance."""
    pytest.importorskip("ase", reason="the three-body channels are a crystal descriptor")
    from ase import Atoms

    from cmfqsar.periodic import crystal_table, inorganic_fieldset

    def rocksalt(a, cation):
        pos = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
                        [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]])
        return Atoms(symbols=[cation] * 4 + ["Cl"] * 4, scaled_positions=pos,
                     cell=[a, a, a], pbc=True)

    cells = [rocksalt(a, c) for a, c in [(4.0, "Na"), (4.2, "Na"), (4.4, "Na"),
                                         (4.1, "K"), (4.3, "K"), (4.5, "K")]]
    names = [f"c{i}" for i in range(len(cells))]
    fs = inorganic_fieldset([], physchem=["steric"], charge_model="electronegativity")
    table = crystal_table(cells, fs, names)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="threebody",
                     tb_r_max=4.0, tb_step=1.0, tb_sigma=1.0, params={"lam": 1e-3})
    model.fit(table, np.arange(len(cells), dtype=float), cv_folds=0)
    assert has_copattern(model, fs) is False
    with pytest.raises(ValueError, match="two radii"):
        copattern(model, fs, 0)


def test_an_unfitted_model_has_no_curve():
    fs = FieldSet.create(physchem=["steric"], alpha=1.0)
    model = CMFModel(fieldset=fs, method="krr", kernel_source="autocorrelation")
    assert has_copattern(model, fs) is False
    assert has_copattern(None, fs) is False


def test_a_nonlinear_kernel_layer_has_no_channel_weights(fitted):
    """A Gaussian layer is not linear in the channels, so the channels have no weights."""
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(molecules())]
    model = CMFModel(fieldset=fs, method="krr", normalize=True,
                     kernel_source="autocorrelation", ac_sigma=0.5, ac_d_max=6.0,
                     kernel_layer="gauss", params={"lam": 1e-3, "gamma": 1.0})
    model.fit(AtomTable.from_molecules(mf), np.arange(len(mf), dtype=float), cv_folds=0)
    assert has_copattern(model, fs) is False


# --------------------------------------------------------------------------- #
# The chart in the window
# --------------------------------------------------------------------------- #
def test_the_chart_fills_its_boxes_and_draws_the_curve(fitted, qt_application):
    pytest.importorskip("pyqtgraph")

    from cmfqsar.gui.copattern_panel import CoPatternChart
    model, fs, _table, _y = fitted
    chart = CoPatternChart()
    assert chart.set_model(model, fs) is True
    assert [chart.cmb_pair.itemText(i) for i in range(chart.cmb_pair.count())] == \
        pair_labels(model, fs)
    x, y = chart.curve.getData()
    cp = copattern(model, fs, 0)
    assert np.allclose(x, cp.x)
    assert np.allclose(y, cp.y)
    chart.cmb_value.setCurrentIndex(1)
    _x2, y2 = chart.curve.getData()
    assert np.allclose(y2, copattern(model, fs, 0, "weight x spread").y)
    assert chart.current_value() == "weight x spread"
    chart.clear("nothing to show")
    assert chart.cmb_pair.count() == 0
    assert chart.note.text() == "nothing to show"


def test_the_chart_says_why_an_overlap_model_has_none(qt_application):
    pytest.importorskip("pyqtgraph")

    from cmfqsar.gui.copattern_panel import CoPatternChart
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(molecules())]
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 1e-3})
    model.fit(AtomTable.from_molecules(mf), np.arange(len(mf), dtype=float), cv_folds=0)
    chart = CoPatternChart()
    assert chart.set_model(model, fs) is False
    assert "co-field" in chart.note.text()
    assert chart.curve.getData()[0] is None or len(chart.curve.getData()[0]) == 0
