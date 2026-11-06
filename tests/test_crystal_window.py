"""A periodic structure as a sample in the application (plan item M13).

The window is tested through the things a user would check by looking at it, each against
something independent of the drawing:

* a folder of CIF files becomes a list of structures with their cells;
* the cell box is drawn, the atoms of the cell are drawn, and the periodic images are the ones
  the kernel sums over -- their number grows with the radius and agrees with the expansion the
  kernel itself uses;
* the site colouring is the per-site split of the prediction, so the numbers behind the colours
  sum to the prediction less the bias, exactly;
* the co-field grid covers the cell of a crystal and stops at the atoms of a layer, where the
  rest of the cell is vacuum;
* the reading tables hold what their titles say.

The window runs its long steps in a worker thread; these tests build it with
``background=False``, so every step finishes before the call returns and no event loop is needed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("ase", reason="the crystal path needs ASE")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ase import Atoms                                                   # noqa: E402
from PySide6.QtWidgets import QApplication                              # noqa: E402

from cmfqsar import periodic                                            # noqa: E402
from cmfqsar.gui.crystal_window import (CrystalWindow, cell_grid, reading_rows, site_rows,
                                        site_contributions_of)          # noqa: E402
from cmfqsar.gui.viewer3d import color_name, element_color, signed_spheres   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIFS = os.path.join(ROOT, "examples", "hops", "disordered")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def rocksalt(a=4.2, cation="Na", anion="Cl"):
    """One cubic rock-salt cell of eight atoms: a small set that fits on one lattice."""
    pos = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                    [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]])
    return Atoms(symbols=[cation] * 4 + [anion] * 4, scaled_positions=pos, cell=[a, a, a],
                 pbc=True)


def layer(a=4.0, height=1.2):
    """A two-dimensional cell with vacuum along the third vector: four atoms in a plane."""
    at = Atoms(symbols=["Mo", "S", "S", "Mo"],
               positions=[[0, 0, 10.0], [a / 2, 0, 10.0 + height],
                          [0, a / 2, 10.0 - height], [a / 2, a / 2, 10.0]],
               cell=[a, a, 20.0], pbc=(True, True, False))
    return at


@pytest.fixture(scope="module")
def small_set():
    """Six rock-salt cells with a target that is a function of the lattice constant."""
    cells, names, y = [], [], []
    for k, (a, cation) in enumerate([(4.0, "Na"), (4.2, "Na"), (4.4, "Na"),
                                     (4.1, "K"), (4.3, "K"), (4.5, "K")]):
        cells.append(rocksalt(a, cation))
        names.append(f"{cation}Cl_{a:g}")
        y.append(10.0 * a + (1.0 if cation == "K" else 0.0))
    return cells, names, np.array(y, float)


@pytest.fixture()
def one_lattice(app):
    """Six cells with the same lattice constant: the case the overlap kernel is for."""
    cells, names, y = [], [], []
    for k, (cation, anion) in enumerate([("Na", "Cl"), ("Na", "Br"), ("K", "Cl"),
                                         ("K", "Br"), ("Li", "Cl"), ("Li", "Br")]):
        cells.append(rocksalt(4.2, cation, anion))
        names.append(f"{cation}{anion}")
        y.append(1.0 + 0.7 * k)
    w = CrystalWindow(background=False)
    w.atoms, w.names, w.y = cells, names, np.array(y, float)
    w.populate()
    w.table.setCurrentCell(0, 0)
    return w


@pytest.fixture()
def window(app, small_set):
    """A window holding the six cells with their values, nothing fitted yet."""
    cells, names, y = small_set
    w = CrystalWindow(background=False)
    w.atoms, w.names, w.y = list(cells), list(names), y.copy()
    w.subset = ["train"] * len(names)
    w.populate()
    w.table.setCurrentCell(0, 0)
    w.update_display()
    return w


def fitted(w, kernel: int = 0):
    """Fit the window's model on its cells with the smallest settings that still mean something."""
    w.cmb_kernel.setCurrentIndex(kernel)
    w.cmb_charges.setCurrentText("electronegativity")
    w.spin_min_structures.setValue(1)
    w.spin_cutoff.setValue(6.0)
    w.spin_sigma.setValue(1.0)
    w.spin_folds.setValue(3)
    w.ed_alphas.setText("2" if kernel else "1")
    w.ed_lams.setText("1e-3")
    w.build_model()
    return w.model


# ----------------------------------------------------------------- reading a folder of files
def test_a_folder_of_cifs_is_a_data_set(app):
    w = CrystalWindow(background=False)
    w.load(CIFS)
    assert len(w.names) == 13
    assert "cod_1545083" in w.names                 # named after the file, as a database set is
    assert all(len(a) > 0 for a in w.atoms)
    assert "13 structures" in w.note.text()


def test_without_measured_values_the_window_says_what_to_do(app):
    w = CrystalWindow(background=False)
    w.load(CIFS)
    w.build_model()                                  # no values: nothing is fitted
    assert w.model is None


def test_the_measured_values_come_from_a_column_of_a_table(app):
    """A prepared data set is one file with one column per property, not a name,value pair."""
    w = CrystalWindow(background=False)
    path = os.path.join(ROOT, "examples", "sac", "sac_hoer.csv")
    if not os.path.isfile(path):
        pytest.skip("the single-atom example set is not in this checkout")
    atoms, names = periodic.read_crystal_set(os.path.join(ROOT, "examples", "sac",
                                                          "substrates.extxyz"))
    w.atoms, w.names = atoms[:6], names[:6]
    w.y = np.zeros(0)
    w.import_targets(path, column="height")
    assert np.isfinite(w.y).all()
    assert "height" in w.note.text()


# ------------------------------------------------------------------------------- the drawing
def test_the_cell_and_its_atoms_are_drawn(window):
    groups = window.viewer._groups
    assert groups["cell"], "the lattice box is not drawn"
    assert groups["atoms"], "the atoms of the cell are not drawn"
    assert "cell 4.00 x 4.00 x 4.00 A" in window.legend.text()


def test_the_images_are_the_ones_the_kernel_sums_over(window):
    """The dots outside the box are the expansion the periodic kernel itself uses."""
    window.spin_images.setValue(6.0)
    window.update_display()
    crystal = window._crystal(0)
    coords, _index = periodic.cell_image_indices(crystal, 6.0)
    expected = len(coords) - crystal.n_atoms
    assert f"{expected} image atoms" in window.legend.text()
    assert expected > crystal.n_atoms


def test_a_larger_radius_draws_more_images(window):
    window.spin_images.setValue(4.0)
    window.update_display()
    near = int(window.legend.text().split(" image atoms")[0].split("|")[-1])
    window.spin_images.setValue(8.0)
    window.update_display()
    far = int(window.legend.text().split(" image atoms")[0].split("|")[-1])
    assert far > near


def test_the_images_can_be_switched_off(window):
    window.chk_images.setChecked(False)
    window.update_display()
    assert not window.viewer._groups.get("images")
    assert "image atoms" not in window.legend.text()


def test_a_layer_has_no_images_through_the_vacuum(app):
    """A structure periodic in two directions is never paired with a copy of itself above it."""
    w = CrystalWindow(background=False)
    w.atoms, w.names, w.y = [layer()], ["MoS2"], np.array([1.0])
    w.subset = ["train"]
    w.populate()
    w.table.setCurrentCell(0, 0)
    w.update_display()
    crystal = w._crystal(0)
    coords, _ = periodic.cell_image_indices(crystal, 8.0)
    assert np.allclose(coords[:, 2].max(), np.asarray(crystal.coords)[:, 2].max())
    assert "periodic in 2 directions" in w.legend.text()


def test_the_elements_of_a_solid_have_their_own_colours():
    assert element_color("Li") != element_color("W")
    assert element_color("C") == (0.45, 0.45, 0.45)      # as the molecular views draw it
    assert element_color("Nonsense") == (0.6, 0.6, 0.6)


def test_a_colour_has_a_name_a_legend_can_use():
    assert color_name((0.90, 0.15, 0.15)) == "red"
    assert color_name((0.15, 0.30, 0.95)) == "blue"


# ----------------------------------------------------------------------- the co-field grid
def test_the_grid_covers_the_cell_of_a_crystal(window):
    crystal = window._crystal(0)
    grid = cell_grid(crystal, spacing=0.5)
    lo, hi = grid.origin, grid.origin + grid.spacing * (np.array(grid.shape) - 1)
    assert np.all(lo <= 1e-9) and np.all(hi >= 3.9)


def test_the_grid_of_a_layer_stops_at_the_atoms(app):
    """Most of a layer's cell is vacuum, and a co-field there would be points spent on nothing."""
    crystal = periodic.crystal_fields(layer(), periodic.inorganic_fieldset(
        ["Mo", "S"], physchem=["steric"]), "MoS2")
    grid = cell_grid(crystal, spacing=0.5, margin=3.0)
    height = grid.spacing * (grid.shape[2] - 1)
    assert height < 12.0, "the vacuum of the cell is still in the grid"
    z = np.asarray(crystal.coords)[:, 2]
    assert grid.origin[2] <= z.min() and grid.origin[2] + height >= z.max()


# -------------------------------------------------------------------- the model and its reading
def test_the_autocorrelation_model_is_fitted_and_read(window):
    model = fitted(window, kernel=0)
    assert model is not None
    assert model.kernel_source == "periodic"
    assert len(window.predictions) == len(window.names)
    cols, rows = reading_rows(model, model.fieldset, n=5)
    assert cols[0] == "Channel" and len(rows) == 5
    assert all("@" in r[0] for r in rows)            # a channel is a pair of fields at a distance


def test_the_site_contributions_sum_to_the_prediction(window):
    """The colouring is a decomposition, so its numbers have to add up exactly."""
    model = fitted(window, kernel=0)
    crystal = window._crystal(0, model.fieldset)
    values, how = site_contributions_of(model, crystal)
    assert len(values) == crystal.n_atoms
    predicted = model.predict(periodic.crystal_table([window.atoms[0]], model.fieldset,
                                                     [window.names[0]]))[0]
    assert float(np.sum(values)) + float(model.bias) == pytest.approx(predicted, abs=1e-8)
    assert "plan item M5" in how


def test_the_colouring_states_its_scale(window):
    fitted(window, kernel=0)
    window.chk_sites.setChecked(True)
    window.update_display()
    assert window.viewer._groups.get("sites"), "the sites are not drawn"
    values, _ = window._site_values(0)
    radii, colors, vmax = signed_spheres(values)
    assert vmax == pytest.approx(float(np.abs(values).max()))
    assert window.scale.vmax == pytest.approx(vmax)
    assert window.scale.lab_hi.text().startswith("+")
    # the largest site is the largest sphere, and the sign decides the colour
    assert radii[int(np.argmax(np.abs(values)))] == pytest.approx(max(radii))
    up = {c for c, v in zip(colors, values) if v > 0}
    down = {c for c, v in zip(colors, values) if v <= 0}
    assert len(up) <= 1 and len(down) <= 1, "one colour per sign"
    assert not (up & down), "the two signs are drawn in the same colour"


def test_the_site_table_names_the_largest_sites_first(window):
    model = fitted(window, kernel=0)
    crystal = window._crystal(0, model.fieldset)
    values, _ = site_contributions_of(model, crystal)
    cols, rows = site_rows(values, crystal, n=4)
    assert cols[:3] == ["Site", "Element", "Contribution"]
    got = [abs(float(r[2])) for r in rows]
    assert got == sorted(got, reverse=True)
    assert rows[0][1] in crystal.symbols


def test_the_autocorrelation_has_no_co_field_and_says_so(window):
    """It is a distribution of distances, so it cannot say where anything is."""
    fitted(window, kernel=0)
    assert not window.chk_cofield.isEnabled()
    window.compute_cofield()
    assert window._volume is None


def test_the_overlap_model_has_a_co_field_in_the_cell(one_lattice):
    window = one_lattice
    model = fitted(window, kernel=1)
    assert model is not None and model.pbc_overlap
    window.table.setCurrentCell(0, 0)
    window.spin_spacing.setValue(0.5)
    window.compute_cofield()
    index, name, label, volume, grid = window._volume
    assert index == 0 and volume.shape == tuple(grid.shape)
    assert np.abs(volume).max() > 0
    window.chk_cofield.setChecked(True)
    window.update_display()
    assert window.viewer._groups.get("cofield"), "no isosurface was drawn"
    assert f"co-field {label} inside the cell" in window.legend.text()


def test_the_overlap_model_splits_a_prediction_over_the_atoms(one_lattice):
    """On one lattice the split over the atoms is the prediction, to machine precision."""
    w = one_lattice
    model = fitted(w, kernel=1)
    crystal = w._crystal(0, model.fieldset)
    values, how = site_contributions_of(model, crystal)
    assert float(np.sum(values)) + float(model.bias) == pytest.approx(w.predictions[0], abs=1e-6)
    assert "rule 3" in how


def test_a_set_that_is_not_one_lattice_is_said_to_be_one(window):
    """The overlap kernel compares two cells point by point; off one lattice the split is not
    the prediction, and the window has to say so rather than draw a sum that does not add up."""
    fitted(window, kernel=1)
    window.chk_sites.setChecked(True)
    window.update_display()
    assert "do not stand exactly on one lattice" in window.legend.text()

# --------------------------------------------------------------------------- #
# How the atoms are drawn: the four representations, and the bonds
# --------------------------------------------------------------------------- #
def test_the_four_representations_are_offered(window):
    from cmfqsar.gui.crystal_window import ATOM_STYLES
    offered = [window.cmb_atoms.itemText(i) for i in range(window.cmb_atoms.count())]
    assert offered == list(ATOM_STYLES)
    assert offered == ["balls", "dots", "ball and stick", "sticks"]


def test_the_unbonded_styles_draw_the_atoms_and_no_bonds(window):
    window.table.setCurrentCell(0, 0)
    for style in ("balls", "dots"):
        window.cmb_atoms.setCurrentText(style)
        window.update_display()
        assert window.viewer.has_group("atoms"), style
        assert not window.viewer.has_group("bonds"), style


def test_a_cell_of_metal_atoms_gets_no_bonds_and_falls_back_to_balls(window):
    """The rock-salt cells have only sodium, potassium and chlorine: no pair is close enough."""
    window.table.setCurrentCell(0, 0)
    window.cmb_atoms.setCurrentText("ball and stick")
    window.update_display()
    assert not window.viewer.has_group("bonds")
    assert window.viewer.has_group("atoms")
    assert "no bonds" in window.legend.text()
    assert "drawn as balls instead" in window.legend.text()


def test_the_polyhedra_box_draws_the_shell_of_every_metal(window):
    """What a metal gets instead of bonds: the rock-salt cation is six-coordinate."""
    window.table.setCurrentCell(0, 0)
    window.chk_polyhedra.setChecked(True)
    window.update_display()
    assert window.viewer.has_group("polyhedra")
    assert "6-coordinate" in window.legend.text()
    window.chk_polyhedra.setChecked(False)
    window.update_display()
    assert not window.viewer.has_group("polyhedra")


def test_an_organic_molecule_in_a_cell_is_drawn_with_its_bonds(app):
    """The case the bonds exist for: a molecule whose rings a cloud of spheres cannot show."""
    import math

    from cmfqsar.gui.crystal_window import CrystalWindow
    r = 1.39
    ring = [(r * math.cos(k * math.pi / 3), r * math.sin(k * math.pi / 3), 4.0)
            for k in range(6)]
    at = Atoms(symbols=["C"] * 6, positions=np.array(ring) + np.array([4.0, 4.0, 0.0]),
               cell=[8.0, 8.0, 8.0], pbc=True)
    # the same ring put over the corner of the cell, so that the box cuts it in two
    cut = Atoms(symbols=["C"] * 6, positions=np.array(ring) - np.array([0.0, 0.0, 4.0]),
                cell=[8.0, 8.0, 8.0], pbc=True)
    cut.wrap()
    w = CrystalWindow(background=False)
    try:
        w.atoms = [at, cut]
        w.names = ["benzene", "benzene over the corner"]
        w.y = np.array([1.0, 2.0])
        w.populate()
        w.table.setCurrentCell(0, 0)
        for style in ("ball and stick", "sticks"):
            w.cmb_atoms.setCurrentText(style)
            w.update_display()
            assert w.viewer.has_group("bonds"), style
            assert f"{style}: 6 bonds" in w.legend.text()
            assert "perceived from the distances for drawing only" in w.legend.text()
        assert "crossing the cell boundary" not in w.legend.text()
        # the ring the box cuts is still a ring, and the image atoms it needs are drawn with it
        w.table.setCurrentCell(1, 0)
        w.update_display()
        assert "6 bonds" in w.legend.text()
        assert "crossing the cell boundary" in w.legend.text()
        assert "image atoms drawn with them" in w.legend.text()
    finally:
        w.close()
