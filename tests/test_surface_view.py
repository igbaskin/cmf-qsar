"""The surface side of the program on screen (plan item EC10).

Three things the electrochemistry items built were invisible in the application, and each is
tested here through what a user would check by looking at it, against something independent of
the drawing:

* the **slab** behind the molecules: the patch grows with the radius, it is cut in the plane and
  not in space, every drawn atom lies in or below the top layer, and the atoms drawn are the
  atoms the complementarity receptor of :mod:`cmfqsar.slab` is built from;
* the **surface frame**: the height and the tilt the panel states are computed from the
  coordinates, so they are checked against the height the relaxation wrote into the Fe(110) data
  set, and against poses built by hand whose two numbers are known in advance;
* the **vector co-field**: the arrows stand on the largest gradients of the co-field, and the
  box says why it draws nothing for a scalar field.

The window runs its long steps in a worker thread; these tests build it with
``background=False``, so every step finishes before the call returns and no event loop is needed.
Nothing here opens a modal dialog, which would wait for a person that is not there.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("ase", reason="a slab is an ASE structure")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cmfqsar.fields import FieldSet                                     # noqa: E402
from cmfqsar.gui.surface_view import (build_bcc110, drawn_atoms, pose_line,  # noqa: E402
                                      pose_table, site_points)
from cmfqsar.planealign import pose_geometry                            # noqa: E402
from cmfqsar.slab import slab_receptor                                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE110_SDF = os.path.join(ROOT, "examples", "corrosion_fe110", "157_benzimidazole_uma.sdf")
FE110_CSV = os.path.join(ROOT, "examples", "corrosion_fe110", "157_benzimidazole_uma.csv")
HAMMETT = os.path.join(ROOT, "examples", "hammett", "hammett.sdf")


@pytest.fixture(scope="module")
def app(qt_application):
    return qt_application


@pytest.fixture(scope="module")
def surface(app):
    return build_bcc110(element="Fe", nx=6, ny=6, layers=4, a=2.78, vacuum=15.0)


# --------------------------------------------------------------------------- #
# The slab
# --------------------------------------------------------------------------- #
def test_the_slab_is_built_in_the_surface_frame(surface):
    """The normal is z and the top atomic layer is z = 0, which is the frame the poses use."""
    z = np.asarray(surface.atoms.positions, float)[:, 2]
    assert z.max() == pytest.approx(0.0, abs=1e-9)
    assert z.min() < -1.0                       # the slab goes down, never up
    assert len(surface.layers) == 4
    assert surface.spacing == pytest.approx(1.97, abs=0.05)


def test_an_odd_number_of_repeats_is_rounded_and_named(app):
    """ASE needs an even count along the second surface direction; the name says what was built."""
    s = build_bcc110(nx=3, ny=3, layers=2)
    assert "3x4x2" in s.name


def test_the_patch_grows_with_the_radius(surface):
    points = np.array([[0.0, 0.0, 2.0]])
    counts = [len(drawn_atoms(surface, points, radius=r)[0]) for r in (4.0, 8.0, 16.0)]
    assert counts[0] < counts[1] < counts[2]
    # the area of the patch grows as the square of the radius, so so does the count
    assert counts[2] / max(counts[0], 1) > 3.0


def test_the_patch_is_cut_in_the_plane_and_not_in_space(surface):
    """A molecule 8 A above the surface still has a surface under it.

    The cut has to be in the plane: a cut in space would thin the surface out to nothing under a
    molecule that sits high above it, and the picture would say the surface ends there.
    """
    low = drawn_atoms(surface, np.array([[0.0, 0.0, 2.0]]), radius=6.0)[0]
    high = drawn_atoms(surface, np.array([[0.0, 0.0, 8.0]]), radius=6.0)[0]
    assert len(high) == len(low) > 0


def test_every_drawn_atom_is_an_atom_of_the_receptor(surface):
    """The picture shows what a complementarity score against the surface is computed from."""
    fs = FieldSet.create(physchem=["steric"], alpha=1.0)
    receptor = slab_receptor(surface.atoms, fs, top_layers=2, cutoff=24.0)
    known = {tuple(np.round(p, 4)) for p in receptor.coords}
    coords, symbols, radii = drawn_atoms(surface, np.array([[0.0, 0.0, 2.0]]), radius=10.0)
    assert len(coords) == len(symbols) == len(radii) > 0
    assert all(tuple(np.round(p, 4)) in known for p in coords)
    assert coords[:, 2].max() <= 1e-9           # nothing is drawn above the top layer
    assert np.all(radii > 0)                    # every drawn atom has a size to draw it with


def test_the_three_kinds_of_surface_site_are_found(surface):
    labels, points = site_points(surface, np.array([[0.0, 0.0, 2.0]]), radius=8.0)
    assert set(labels) == {"top", "bridge", "hollow"}
    assert len(points) == len(labels)
    assert np.allclose(points[:, 2], 0.0)       # the sites lie in the top layer


# --------------------------------------------------------------------------- #
# The surface frame: the two numbers the panel states
# --------------------------------------------------------------------------- #
def test_a_flat_molecule_lying_on_the_surface_has_no_tilt():
    xyz = np.array([[0.0, 0.0, 2.0], [1.4, 0.0, 2.0], [0.7, 1.2, 2.0], [-0.7, 1.2, 2.0]])
    g = pose_geometry(xyz, ["C"] * 4)
    assert g.height == pytest.approx(2.0)
    assert g.centre_height == pytest.approx(2.0)
    assert g.tilt == pytest.approx(0.0, abs=1e-6)
    assert g.flat is True


def test_the_same_molecule_stood_on_edge_is_tilted_by_ninety_degrees():
    xyz = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.4], [0.0, 1.2, 2.7], [0.0, -1.2, 2.7]])
    g = pose_geometry(xyz, ["C"] * 4)
    assert g.tilt == pytest.approx(90.0, abs=1e-6)
    assert g.height == pytest.approx(2.0)


def test_hydrogens_do_not_set_the_height():
    """The height is quoted on the heavy atoms, as the relaxed poses report it."""
    xyz = np.array([[0.0, 0.0, 2.0], [1.4, 0.0, 2.0], [0.7, 1.2, 2.0], [0.7, 0.6, 1.1]])
    assert pose_geometry(xyz, ["C", "C", "C", "H"]).height == pytest.approx(2.0)
    assert pose_geometry(xyz, ["C", "C", "C", "C"]).height == pytest.approx(1.1)


def test_a_bulky_molecule_is_told_apart_from_a_flat_one():
    rng = np.random.default_rng(0)
    xyz = rng.normal(scale=1.5, size=(20, 3)) + np.array([0.0, 0.0, 5.0])
    g = pose_geometry(xyz, ["C"] * 20)
    assert g.flat is False
    assert g.thickness > 1.0
    assert "no flat plane" in g.describe()


def test_the_pose_line_names_the_molecule_and_both_numbers():
    xyz = np.array([[0.0, 0.0, 2.0], [1.4, 0.0, 2.0], [0.7, 1.2, 2.0]])
    line = pose_line(xyz, ["C"] * 3, "benzene")
    assert line.startswith("benzene:")
    assert "2.00 A above the surface" in line
    assert "tilt" in line


def test_the_pose_table_has_one_row_per_molecule():
    xyz = np.array([[0.0, 0.0, 2.0], [1.4, 0.0, 2.0], [0.7, 1.2, 2.0]])
    cols, rows = pose_table([xyz, xyz + 1.0], [["C"] * 3] * 2, ["a", "b"])
    assert len(rows) == 2
    assert len(cols) == len(rows[0])
    assert rows[0][0] == "a" and rows[1][0] == "b"
    assert float(rows[1][1]) - float(rows[0][1]) == pytest.approx(1.0)


@pytest.mark.skipif(not os.path.exists(FE110_SDF), reason="the Fe(110) set is not in the tree")
def test_the_height_agrees_with_what_the_relaxation_wrote():
    """The panel's height is computed from the coordinates; the relaxation wrote its own.

    They agree only if the window reads the frame the poses are given in, so this is the one
    check of the surface frame that does not rest on the program's own arithmetic.
    """
    import csv

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    written = {}
    with open(FE110_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                written[row["name"]] = float(row["height_A"])
            except (KeyError, TypeError, ValueError):
                continue
    checked = 0
    for mol in Chem.SDMolSupplier(FE110_SDF, removeHs=False, sanitize=True):
        if mol is None or not mol.HasProp("_Name"):
            continue
        target = written.get(mol.GetProp("_Name"))
        if target is None:
            continue
        g = pose_geometry(np.array(mol.GetConformer().GetPositions(), float),
                          [a.GetSymbol() for a in mol.GetAtoms()])
        assert g.height == pytest.approx(target, abs=0.01)
        checked += 1
    assert checked > 50


# --------------------------------------------------------------------------- #
# The window: the slab behind the molecules, and the arrows
# --------------------------------------------------------------------------- #
@pytest.fixture()
def window(app, surface):
    from cmfqsar.gui.main_window import MainWindow
    if not os.path.exists(FE110_SDF):
        pytest.skip("the Fe(110) set is not in the tree")
    w = MainWindow(background=False)
    w.load_dataset_file(FE110_SDF, "train", activity="E_ads_kJmol")
    w._install_surface(surface)
    yield w
    w.close()


def test_installing_a_surface_turns_the_box_on_and_names_it(window, surface):
    assert window.surface is surface
    assert window.controls.chk_slab.isChecked()
    assert surface.name in window.controls.lbl_slab.text()


def test_the_window_draws_the_slab_the_frame_and_the_sites(window):
    vc = window.controls
    vc.chk_slab.setChecked(True)
    vc.chk_slab_frame.setChecked(True)
    vc.chk_slab_sites.setChecked(True)
    window.dataset_panel.select_index(0)
    window.update_display()
    assert window.viewer.has_group("slab")
    assert window.viewer.has_group("slabframe")
    assert window.viewer.has_group("slabsites")
    legend = vc.lbl_legend.text()
    assert "top layer at z = 0" in legend
    assert "surface frame" in legend
    assert "above the surface" in legend          # the pose of the selected molecule


def test_without_a_surface_the_legend_says_what_to_do(window):
    window.clear_surface()
    window.controls.chk_slab.setChecked(True)
    window.update_display()
    assert not window.viewer.has_group("slab")
    assert "load one or build one" in window.controls.lbl_legend.text()


def test_posing_on_the_plane_puts_every_molecule_at_the_asked_height(window):
    window.pose_on_plane(pose="flat", height=2.2)
    assert window.plane_poses is not None and len(window.plane_poses) == len(window.ds.names)
    heights = [pose_geometry(m.coords, m.symbols).height for m in window.molfields]
    assert np.allclose(heights, 2.2, atol=1e-6)
    tilts = [pose_geometry(m.coords, m.symbols).tilt for m in window.molfields]
    assert max(tilts) < 1e-3                      # the flat rule leaves no tilt
    assert window.results_panel.tbl_poses.rowCount() == len(window.ds.names)


def test_the_pose_step_can_be_undone(window):
    before = [m.coords.copy() for m in window.molfields]
    window.pose_on_plane(pose="flat", height=2.2)
    assert not np.allclose(window.molfields[0].coords, before[0])
    window.undo_alignment()
    assert np.allclose(window.molfields[0].coords, before[0], atol=1e-6)


@pytest.mark.skipif(not os.path.exists(HAMMETT), reason="the Hammett set is not in the tree")
def test_the_arrows_stand_on_the_largest_gradients_of_the_co_field(app):
    """The drawn arrows are the vector co-field, and the longest one is the largest gradient."""
    from cmfqsar.cofields import cofield_gradient
    from cmfqsar.gui.main_window import MainWindow

    w = MainWindow(background=False)
    try:
        w.load_dataset_file(HAMMETT, "train", activity="sigma_m")
        for name, box in w.model_panel.chk_fields.items():
            box.setChecked(name == "efield")
        w.model_panel.chk_cif.setChecked(False)
        w.refresh_fieldset()
        w.build_model()
        w.compute_cofields()
        vc = w.controls
        vc.chk_cofield.setChecked(False)
        vc.chk_arrows.setChecked(True)
        w.dataset_panel.select_index(0)
        w.update_display()
        assert w.viewer.has_group("arrows")
        legend = vc.lbl_legend.text()
        assert "vector co-field" in legend
        assert "per unit of the field per" in legend          # the unit is stated
        _key, points, vectors = w._arrows
        assert len(points) == len(vectors) == vc.arrow_count()
        g = cofield_gradient(w.model, 0, w.grid)
        magnitude = np.linalg.norm(np.stack([g[k].ravel() for k in range(3)], axis=1), axis=1)
        assert np.linalg.norm(vectors, axis=1).max() == pytest.approx(magnitude.max(), rel=1e-9)
        # the arrows spread over the molecule instead of crowding one peak
        from scipy.spatial.distance import pdist
        assert pdist(points).min() >= 1.5 - 1e-9
    finally:
        w.close()


def test_a_scalar_field_has_no_arrows_and_the_legend_says_why(window):
    """Every other field's co-field is a value at a place, and the isosurface is its drawing."""
    w = window
    w.model_panel.chk_cif.setChecked(False)
    w.refresh_fieldset()
    w.build_model()
    w.compute_cofields()
    w.controls.chk_arrows.setChecked(True)
    w.dataset_panel.select_index(0)
    w.update_display()
    assert not w.viewer.has_group("arrows")
    assert "is a scalar" in w.controls.lbl_legend.text()
