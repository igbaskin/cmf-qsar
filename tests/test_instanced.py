"""Copies of one shape drawn at once, for structures of ten thousand atoms (plan item NT8).

The graphics card cannot be asked from a test without a screen, so the tests check what decides
the picture before it gets there:

* the vertex normals computed with array operations are pyqtgraph's own, to rounding;
* a copy of the unit sphere or the unit cylinder, moved by its instance row, lands on exactly
  the vertices the old per-atom mesh had, and its normals are the ones pyqtgraph would compute
  for that mesh, so the shading is the same;
* the viewer and the crystal window hand their spheres and sticks to the instanced item, with
  one row per atom or half bond, and the window centres a particle on its atoms.

The pictures themselves are compared pixel by pixel in ``benchmarks/run_large_view.py``.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.opengl import MeshData                                   # noqa: E402

from cmfqsar.gui import instanced as ins                                # noqa: E402
from cmfqsar.gui.viewer3d import Viewer3D, _rotation_to                 # noqa: E402


def _angles(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    cos = np.clip(np.sum(a * b, axis=1), -1, 1)
    return np.degrees(np.arccos(cos))


def _pyqtgraph_normals(verts, faces):
    return MeshData(vertexes=np.asarray(verts, np.float32), faces=np.asarray(faces)).vertexNormals()


def test_vertex_normals_are_pyqtgraphs():
    rng = np.random.default_rng(0)
    sv, _, sf = ins.unit_sphere(10, 18)
    meshes = [(sv * 1.7 + 3.0, sf)]
    pts = rng.normal(size=(60, 3))
    faces = rng.integers(0, 60, size=(150, 3))
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                  & (faces[:, 0] != faces[:, 2])]
    meshes.append((pts, faces))
    for v, f in meshes:
        ours = ins.vertex_normals(v, f)
        theirs = _pyqtgraph_normals(v, f)
        used = np.unique(f)
        assert _angles(ours[used], theirs[used]).max() < 0.1


def test_a_vertex_without_faces_gets_the_zero_vector():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], float)
    n = ins.vertex_normals(v, np.array([[0, 1, 2]]))
    assert np.allclose(n[3], 0.0)
    assert np.allclose(np.abs(n[:3, 2]), 1.0)


def test_the_unit_shapes_keep_pyqtgraphs_sign():
    """The normals point the way pyqtgraph's would, which is into the sphere: the lighting of the
    'shaded' program is set for that sign, and the opposite one draws dark spheres."""
    v, n, f = ins.unit_sphere(10, 18)
    theirs = _pyqtgraph_normals(v, f)
    assert _angles(n, theirs).max() < 0.1
    assert np.sum(n * v, axis=1).mean() < 0                  # into the sphere, as pyqtgraph has it
    cv, cn, cf = ins.unit_cylinder(8)
    assert _angles(cn, _pyqtgraph_normals(cv, cf)).max() < 0.1
    assert np.allclose(cn[:, 2], 0.0, atol=1e-6)


def test_spheres_land_on_the_old_mesh():
    rng = np.random.default_rng(1)
    centers = rng.uniform(-5, 5, (40, 3))
    radii = rng.uniform(0.2, 0.9, 40)
    radii[3] = 0.0                                           # left out, as the old code did
    cols = rng.uniform(0, 1, (40, 3))
    inst = ins.sphere_instances(centers, radii, cols, alpha=0.5)
    assert inst.shape == (39, ins.INSTANCE_WIDTH)
    sv, sn, sf = ins.unit_sphere(10, 18)
    V, N, F, C = ins.expand_instances(sv, sn, sf, inst)
    old = np.concatenate([sv * r + c for c, r in zip(centers, radii) if r > 1e-4])
    assert np.abs(V - old).max() < 1e-5
    assert np.allclose(C[:, 3], 0.5)
    assert np.allclose(C[0, :3], cols[0])
    # shading: the normals of the copies are pyqtgraph's normals of the merged mesh
    assert _angles(N, _pyqtgraph_normals(V, F)).max() < 0.1


def test_sticks_land_on_the_old_mesh():
    rng = np.random.default_rng(2)
    starts = rng.uniform(-5, 5, (30, 3))
    ends = starts + rng.normal(0, 1.5, (30, 3))
    ends[0] = starts[0] + [0, 0, 1.2]                         # along z
    ends[1] = starts[1] - [0, 0, 1.2]                         # against z
    ends[2] = starts[2]                                       # no length: left out
    r = 0.15
    inst = ins.cylinder_instances(starts, ends, r, (0.4, 0.5, 0.6))
    assert len(inst) == 29
    cv, cn, cf = ins.unit_cylinder(8)
    V, N, F, _ = ins.expand_instances(cv, cn, cf, inst)
    old_cv = MeshData.cylinder(rows=1, cols=8, radius=[r, r], length=1.0).vertexes()
    old = []
    for p, q in zip(starts, ends):
        d = q - p
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        old.append((old_cv * np.array([1.0, 1.0, L])) @ _rotation_to(d / L).T + p)
    assert np.abs(V - np.concatenate(old)).max() < 1e-5
    assert _angles(N, _pyqtgraph_normals(V, F)).max() < 0.1


def test_the_sphere_gets_coarser_with_the_count():
    assert ins.sphere_detail(500) == (10, 18)
    assert ins.sphere_detail(2000) == (10, 18)
    assert ins.sphere_detail(2001) == (8, 14)
    assert ins.sphere_detail(50000) == (6, 10)


def test_the_merged_fallback_is_built_from_the_same_rows():
    item = ins.spheres_item(np.zeros((2, 3)) + [[0, 0, 0], [3, 0, 0]], [0.5, 0.7],
                            [(1, 0, 0), (0, 1, 0)])
    assert item.n_instances == 2 and item.mode == ""
    v, n, f = ins.unit_sphere(*ins.sphere_detail(2))
    V, _, F, _ = ins.expand_instances(item.base_verts, item.base_normals, item.base_faces,
                                      item.instances)
    assert len(V) == 2 * len(v) and len(F) == 2 * len(f)
    assert ins.spheres_item(np.zeros((1, 3)), [0.0], [(1, 1, 1)]) is None


@pytest.fixture(scope="module")
def app(qt_application):
    return qt_application


def test_the_viewer_draws_atoms_and_sticks_as_copies(app):
    v = Viewer3D()
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-10, 10, (5000, 3))
    v.add_atoms(xyz, ["Pt"] * 5000, "atoms", radii=np.full(5000, 1.36))
    (item,) = v._groups["atoms"]
    assert isinstance(item, ins.InstancedMeshItem) and item.n_instances == 5000
    assert np.allclose(item.instances[:, 3], 1.36 * 0.45, atol=1e-6)
    # a coarser sphere above 2,000 atoms
    assert len(item.base_verts) == len(ins.unit_sphere(*ins.sphere_detail(5000))[0])
    bonds = np.array([[0, 1], [1, 2], [2, 3]])
    v.add_structure(xyz[:4], bonds, ["C", "O", "C", "N"], "bonds", style="sticks")
    (sticks,) = v._groups["bonds"]
    assert isinstance(sticks, ins.InstancedMeshItem) and sticks.n_instances == 6
    # the first half of each bond starts at the first atom, the second ends at the second
    assert np.allclose(sticks.instances[0, :3], xyz[0], atol=1e-5)
    assert np.allclose(sticks.instances[3, :3], 0.5 * (xyz[0] + xyz[1]), atol=1e-5)


def test_meshes_carry_their_normals(app):
    """Every mesh the viewer draws comes with its normals, so pyqtgraph's loop never runs."""
    v = Viewer3D()
    sv, _, sf = ins.unit_sphere(12, 20)
    v.add_isosurface(sv * 3.0, sf, (0.2, 0.8, 0.2), "iso")
    v.add_scalar_surface(sv * 2.0, sf, sv[:, 0], "surf")
    v.add_arrows(np.zeros((3, 3)), np.eye(3), "arrows")
    for group in ("iso", "surf", "arrows"):
        (item,) = v._groups[group]
        assert item.opts["meshdata"]._vertexNormals is not None, group


def test_the_crystal_window_offers_images_as_balls(app):
    pytest.importorskip("ase")
    from ase.build import bulk
    from cmfqsar.gui.crystal_window import IMAGE_STYLES, CrystalWindow
    assert IMAGE_STYLES == ("dots", "balls")
    w = CrystalWindow(background=False)
    w.atoms, w.names, w.y = [bulk("NaCl", "rocksalt", a=5.64, cubic=True)], ["NaCl"], \
        np.array([1.0])
    w.populate()
    w.table.setCurrentCell(0, 0)
    w.chk_images.setChecked(True)
    w.spin_images.setValue(4.0)
    w.cmb_images.setCurrentText("dots")
    w.update_display()
    (dots,) = w.viewer._groups["images"]
    n_images = len(dots.pos)
    w.cmb_images.setCurrentText("balls")
    w.update_display()
    (balls,) = w.viewer._groups["images"]
    assert isinstance(balls, ins.InstancedMeshItem)
    assert balls.n_instances == n_images > 0


def test_a_particle_is_centred_on_its_atoms(app):
    pytest.importorskip("ase")
    from ase.cluster import Icosahedron
    from cmfqsar.gui.crystal_window import CrystalWindow
    at = Icosahedron("Pt", 3, latticeconstant=3.92)
    at.positions += [40.0, -7.0, 3.0]
    at.pbc = False
    w = CrystalWindow(background=False)
    w.atoms, w.names, w.y = [at], ["Pt55"], np.array([np.nan])
    w.populate()
    w.table.setCurrentCell(0, 0)
    w.center_current()
    c = w.viewer.view.opts["center"]
    assert np.allclose([c.x(), c.y(), c.z()], at.positions.mean(axis=0), atol=1e-4)


# ------------------------------------------------------------------- the benchmark of NT8
BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")


def _bench():
    if BENCH not in sys.path:
        sys.path.insert(0, BENCH)
    import run_all
    import run_large_view
    return run_all, run_large_view


def test_the_registry_lists_the_benchmark_and_its_files():
    run_all, rl = _bench()
    b = next(b for b in run_all.registry() if b.name == "large_view")
    assert b.compute == [[]] and b.needs is None
    assert "large_view/table.md" in b.tables
    stored = set(os.listdir(rl.RESULTS)) if os.path.isdir(rl.RESULTS) else set()
    assert stored, "the benchmark has no stored result"
    assert stored <= {"normals.json", "draw.json", "log.txt", "run.json", "table.md"}


def test_without_a_screen_the_drawing_is_kept(monkeypatch):
    _, rl = _bench()
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    lines = []
    assert rl.draw_stage(log=lines.append) is None
    assert "kept" in lines[0]


def test_the_table_is_rebuilt_from_the_stored_results(tmp_path):
    import shutil
    _, rl = _bench()
    for name in ("normals.json", "draw.json", "run.json"):
        shutil.copy(os.path.join(rl.RESULTS, name), tmp_path / name)
    rl.write_table(str(tmp_path))
    with open(os.path.join(rl.RESULTS, "table.md"), encoding="utf-8") as fh:
        stored = fh.read().replace("\r\n", "\n")
    assert (tmp_path / "table.md").read_text(encoding="utf-8") == stored
