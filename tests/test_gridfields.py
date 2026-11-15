"""Tests of the grid fields of plan item M9 (`cmfqsar/gridfields.py`)."""
import numpy as np
import pytest

from cmfqsar import gridfields as G
from cmfqsar.cdft import BOHR


def gaussian_grid(centres, coeffs, beta, cell, shape, periodic=True, origin=(0, 0, 0)):
    """A sum of Gaussians on a grid, with the periodic images summed in when `periodic`."""
    g = G.GridField(np.zeros(shape), cell, origin, periodic, ["X"] * len(centres),
                    np.asarray(centres, float), "density")
    pts = g.points()
    vals = np.zeros(len(pts))
    shifts = [np.zeros(3)]
    if periodic:
        shifts = [np.array([i, j, k], float) @ np.asarray(cell, float)
                  for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2) for k in (-2, -1, 0, 1, 2)]
    for c, w in zip(centres, coeffs):
        for s in shifts:
            vals += w * np.exp(-0.5 * beta * np.sum((pts - np.asarray(c) - s) ** 2, axis=1))
    return g.with_values(vals.reshape(shape))


def cubic(a=4.0):
    return np.eye(3) * a


# --------------------------------------------------------------------------- #
# The representation and the readers
# --------------------------------------------------------------------------- #

def test_volume_element_and_integral():
    g = G.GridField(np.ones((4, 5, 6)), np.diag([2.0, 3.0, 4.0]))
    assert g.volume == pytest.approx(24.0)
    assert g.dV == pytest.approx(24.0 / 120)
    assert g.integral() == pytest.approx(24.0)
    assert g.points().shape == (120, 3)
    assert np.allclose(g.points()[1], [0.0, 0.0, 4.0 / 6])


def test_vasp_round_trip_divides_out_the_volume(tmp_path):
    g = gaussian_grid([[0, 0, 0], [2, 2, 2]], [1.0, 0.5], 2.0, cubic(), (12, 12, 12))
    g.symbols = ["Na", "Cl"]
    path = str(tmp_path / "CHGCAR.gz")
    G.write_vasp_grid(g, path)
    r = G.read_vasp_grid(path)
    assert r.kind == "density" and r.symbols == ["Na", "Cl"]
    assert np.allclose(r.values, g.values, rtol=1e-9, atol=1e-12)
    assert r.integral() == pytest.approx(g.integral(), rel=1e-9)
    assert np.allclose(r.positions, g.positions, atol=1e-7)


def test_vasp_kind_from_the_name():
    assert G.vasp_kind("run/CHGCAR.gz") == "density"
    assert G.vasp_kind("AECCAR0") == "density"
    assert G.vasp_kind("ELFCAR") == "elf"
    assert G.vasp_kind("x/LOCPOT.gz") == "potential"
    with pytest.raises(ValueError, match="cannot tell the kind"):
        G.vasp_kind("density.dat")


def test_vasp_second_block_after_augmentation(tmp_path):
    """A spin-polarised CHGCAR: augmentation lines, then the magnetisation grid."""
    shape = (2, 3, 4)
    total = np.arange(24, dtype=float).reshape(shape) + 1.0
    mag = -np.arange(24, dtype=float).reshape(shape)
    lines = ["Fe", "1.0", "2.0 0 0", "0 2.0 0", "0 0 2.0", "Fe", "1", "Direct", "0 0 0", "",
             " 2 3 4"]
    for arr in (total,):
        flat = arr.transpose(2, 1, 0).ravel() * 8.0
        lines += [" ".join(f"{v:.6E}" for v in flat[i:i + 5]) for i in range(0, 24, 5)]
    lines += ["augmentation occupancies 1 2", " 0.1 0.2", "   0.0", " 2 3 4"]
    flat = mag.transpose(2, 1, 0).ravel() * 8.0
    lines += [" ".join(f"{v:.6E}" for v in flat[i:i + 5]) for i in range(0, 24, 5)]
    path = tmp_path / "CHGCAR"
    path.write_text("\n".join(lines) + "\n")
    assert np.allclose(G.read_vasp_grid(str(path)).values, total)
    assert np.allclose(G.read_vasp_grid(str(path), block=1).values, mag)
    with pytest.raises(ValueError, match="no grid block 2"):
        G.read_vasp_grid(str(path), block=2)


def test_elf_file_keeps_its_values_and_cartesian_coordinates_are_read(tmp_path):
    """An ELFCAR is not divided by the volume; a POSCAR in Cartesian form is read as such."""
    vals = np.linspace(0.0, 1.0, 8).reshape(2, 2, 2)
    lines = ["O", "1.0", "3.0 0 0", "0 3.0 0", "0 0 3.0", "O", "1", "Cartesian",
             "1.5 0.0 0.0", "", " 2 2 2",
             " ".join(f"{v:.6E}" for v in vals.transpose(2, 1, 0).ravel())]
    path = tmp_path / "ELFCAR"
    path.write_text("\n".join(lines) + "\n")
    g = G.read_vasp_grid(str(path))
    assert g.kind == "elf"
    assert np.allclose(g.values, vals)
    assert np.allclose(g.positions, [[1.5, 0.0, 0.0]])


def test_summary_and_the_writer_refuse_nothing_they_should_accept(tmp_path):
    g = G.GridField(np.full((2, 2, 2), 0.5), np.eye(3) * 2.0, kind="density", name="x")
    s = G.summary(g)
    assert s["integral"] == pytest.approx(4.0) and s["shape"] == [2, 2, 2] and s["periodic"]
    with pytest.raises(ValueError, match="only a periodic grid"):
        G.write_vasp_grid(G.GridField(np.zeros((2, 2, 2)), np.eye(3), periodic=False),
                          str(tmp_path / "CHGCAR"))


def test_cube_reader_converts_a_density_and_keeps_a_potential(tmp_path):
    from cmfqsar.cofields import Grid, export_cube

    grid = Grid(origin=np.array([-1.0, -1.0, -1.0]), spacing=0.5, shape=(5, 5, 5))
    vol = np.linspace(0.0, 1.0, 125)
    path = str(tmp_path / "x.cube")
    export_cube(path, grid, vol)
    d = G.read_cube(path, kind="density")
    p = G.read_cube(path, kind="potential")
    assert not d.periodic and d.shape == (5, 5, 5)
    assert np.allclose(d.origin, [-1.0, -1.0, -1.0], atol=1e-5)
    assert np.allclose(d.steps, np.eye(3) * 0.5, atol=1e-5)
    assert np.allclose(p.values.ravel(), vol, atol=1e-4)
    assert np.allclose(d.values.ravel(), vol / BOHR ** 3, rtol=1e-4, atol=1e-4)


def test_unknown_kind_and_bad_shapes_are_refused():
    with pytest.raises(ValueError, match="unknown kind"):
        G.GridField(np.zeros((2, 2, 2)), np.eye(3), kind="spin")
    with pytest.raises(ValueError, match="3D array"):
        G.GridField(np.zeros((2, 2)), np.eye(3))
    g = G.GridField(np.zeros((2, 2, 2)), np.eye(3))
    with pytest.raises(ValueError, match="values of shape"):
        g.with_values(np.zeros((3, 3, 3)))


# --------------------------------------------------------------------------- #
# Common points
# --------------------------------------------------------------------------- #

def test_fourier_resample_is_exact_for_a_band_limited_field():
    n = 16
    x = np.arange(n) / n
    f = (1.0 + np.cos(2 * np.pi * x)[:, None, None] * np.sin(4 * np.pi * x)[None, :, None]
         + 0.3 * np.cos(2 * np.pi * x)[None, None, :])
    up = G.fourier_resample(f, (32, 32, 32))
    x2 = np.arange(32) / 32
    want = (1.0 + np.cos(2 * np.pi * x2)[:, None, None] * np.sin(4 * np.pi * x2)[None, :, None]
            + 0.3 * np.cos(2 * np.pi * x2)[None, None, :])
    assert np.max(np.abs(up - want)) < 1e-12
    down = G.fourier_resample(up, (16, 16, 16))
    assert np.max(np.abs(down - f)) < 1e-12
    assert G.fourier_resample(f, (8, 10, 12)).mean() == pytest.approx(f.mean())


def test_common_box_interpolates_a_linear_field_exactly():
    g = G.GridField(np.zeros((10, 10, 10)), np.eye(3) * 5.0, (-2.5, -2.5, -2.5), False)
    pts = g.points()
    g = g.with_values((1.0 + pts @ [0.3, -0.2, 0.1]).reshape(10, 10, 10))
    origin, shape = G.common_box([g], spacing=0.4)
    assert np.allclose(origin, [-2.5, -2.5, -2.5])
    b = G.on_common_box(g, origin, (8, 8, 8), 0.4)
    inside = b.points()
    assert np.allclose(b.values.ravel(), 1.0 + inside @ [0.3, -0.2, 0.1], atol=1e-12)


def test_frame_functions_refuse_the_wrong_kind_of_grid():
    per = G.GridField(np.zeros((4, 4, 4)), np.eye(3))
    mol = G.GridField(np.zeros((4, 4, 4)), np.eye(3), periodic=False)
    with pytest.raises(ValueError, match="for periodic grids"):
        G.resample_fractional(mol, (2, 2, 2))
    with pytest.raises(ValueError, match="do not repeat"):
        G.on_common_box(per, np.zeros(3), (2, 2, 2))
    with pytest.raises(ValueError, match="same grid points"):
        G.grid_overlap_kernel([per, G.GridField(np.zeros((5, 5, 5)), np.eye(3))])


# --------------------------------------------------------------------------- #
# The overlap kernel on a shared frame
# --------------------------------------------------------------------------- #

def test_grid_overlap_equals_the_closed_form_gaussian_overlap():
    beta = 2.0
    box = (np.eye(3) * 8.0, (-4.0, -4.0, -4.0))
    a = gaussian_grid([[0, 0, 0]], [1.0], beta, box[0], (64, 64, 64), False, box[1])
    b = gaussian_grid([[0.8, 0, 0]], [1.0], beta, box[0], (64, 64, 64), False, box[1])
    K = G.grid_overlap_kernel([a, b], normalize=False)
    want = (np.pi / beta) ** 1.5
    assert K[0, 0] == pytest.approx(want, rel=1e-6)
    assert K[0, 1] == pytest.approx(want * np.exp(-beta * 0.64 / 4.0), rel=1e-6)
    Kn = G.grid_overlap_kernel([a, b])
    assert np.allclose(np.diag(Kn), 1.0)
    assert Kn[0, 1] == pytest.approx(np.exp(-beta * 0.64 / 4.0), rel=1e-6)
    rect = G.grid_overlap_kernel([a, b], [b])
    assert rect.shape == (1, 2) and rect[0, 1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The radial correlations without a frame
# --------------------------------------------------------------------------- #

def test_radial_profile_of_one_gaussian_matches_the_analytic_curve():
    """C(t) = (pi/beta)^1.5 exp(-beta t^2 / 4) for one Gaussian; A(r) = 4 pi r^2 C(r)."""
    beta = 3.0
    g = gaussian_grid([[0, 0, 0]], [1.0], beta, np.eye(3) * 8.0, (48, 48, 48), False,
                      (-4.0, -4.0, -4.0))
    spec = G.RadialSpec(r_max=3.0, step=0.5, sigma=0.1)
    prof = G.radial_correlations([g], spec)
    rr = np.linspace(0.0, 6.0, 6001)
    curve = 4 * np.pi * rr ** 2 * (np.pi / beta) ** 1.5 * np.exp(-beta * rr ** 2 / 4)
    for k, r in enumerate(spec.centres):
        w = np.exp(-0.5 * ((rr - r) / spec.sigma) ** 2) / (spec.sigma * np.sqrt(2 * np.pi))
        want = np.trapezoid(curve * w, rr)
        assert prof[k] == pytest.approx(want, rel=0.02, abs=1e-3 * curve.max())


def test_radial_profile_does_not_change_when_a_molecule_is_turned_or_moved():
    beta = 2.0
    centres = [[-1.0, 0.3, 0.0], [0.8, -0.5, 0.6], [0.2, 1.1, -0.7]]
    g = gaussian_grid(centres, [1.0, -0.6, 0.4], beta, np.eye(3) * 8.0, (40, 40, 40), False,
                      (-4.0, -4.0, -4.0))
    turned = g.with_values(np.transpose(g.values, (1, 2, 0))[::-1, :, :].copy())
    moved = G.GridField(g.values, g.cell, g.origin + 1.7, False)
    spec = G.RadialSpec(r_max=4.0, step=0.5, sigma=0.4)
    p = G.radial_correlations([g], spec)
    assert np.allclose(G.radial_correlations([turned], spec), p, rtol=1e-10, atol=1e-12)
    assert np.allclose(G.radial_correlations([moved], spec), p, rtol=1e-12, atol=1e-14)


def test_radial_profile_of_a_crystal_is_the_same_for_a_supercell():
    g = gaussian_grid([[0, 0, 0], [2.0, 1.5, 2.5]], [1.0, 0.4], 2.5, cubic(4.0), (16, 16, 16))
    other = g.with_values(np.sqrt(np.abs(g.values)), kind="elf")
    big = [G.GridField(np.tile(x.values, (2, 1, 3)), np.diag([8.0, 4.0, 12.0]), kind=x.kind)
           for x in (g, other)]
    spec = G.RadialSpec(r_max=6.0, step=0.5, sigma=0.5)
    p1 = G.radial_correlations([g, other], spec)
    p2 = G.radial_correlations(big, spec)
    assert p1.shape == (3 * spec.n_bins,)
    assert np.max(np.abs(p1 - p2)) < 1e-12 * np.max(np.abs(p1))


def test_cross_pair_is_symmetric_and_centring_removes_a_constant():
    g = gaussian_grid([[0, 0, 0], [1.0, 2.0, 0.5]], [1.0, -0.7], 2.0, cubic(4.0), (16, 16, 16))
    h = g.with_values(g.values ** 2 + 0.1, kind="potential")
    spec = G.RadialSpec(r_max=4.0, step=0.5, sigma=0.5)
    fg = G.radial_correlations([g, h], spec, pairs=[(0, 1)])
    gf = G.radial_correlations([g, h], spec, pairs=[(1, 0)])
    assert np.allclose(fg, gf, rtol=1e-10)
    shifted = h.with_values(h.values + 5.0)
    a = G.radial_correlations([h], spec, centre=[True])
    b = G.radial_correlations([shifted], spec, centre=[True])
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12)


def test_histogram_serves_several_widths_and_refuses_a_wider_one():
    g = gaussian_grid([[0, 0, 0]], [1.0], 2.0, cubic(4.0), (12, 12, 12))
    h = G.radial_histogram([g], reach=7.0)
    for s in (0.35, 0.5, 0.75):
        spec = G.RadialSpec(r_max=4.0, step=s, sigma=s)
        # the longer histogram also holds the window tails beyond four widths
        assert np.allclose(G.radial_profiles(h, spec), G.radial_correlations([g], spec),
                           rtol=1e-4)
    with pytest.raises(ValueError, match="reach past"):
        G.radial_profiles(h, G.RadialSpec(r_max=6.0, step=0.5, sigma=0.5))
    with pytest.raises(ValueError, match="one structure on one grid"):
        G.radial_histogram([g, G.GridField(np.zeros((6, 6, 6)), cubic(4.0))])


def test_labels_and_block_sizes_agree_with_the_profile():
    spec = G.RadialSpec(r_max=2.0, step=1.0, sigma=0.5)
    labels = G.radial_channel_labels(["density", "elf"], spec)
    assert len(labels) == sum(G.radial_block_sizes(spec, 2)) == 3 * 3
    assert labels[0] == "density x density @ 0 A" and labels[-1] == "density x elf @ 2 A"


# --------------------------------------------------------------------------- #
# The atom-centred fit
# --------------------------------------------------------------------------- #

def test_fit_recovers_a_field_made_of_its_own_gaussians_in_a_crystal():
    cell = cubic(4.0)
    g = G.GridField(np.zeros((16, 16, 16)), cell, symbols=["A", "B"],
                    positions=[[0.2, 0.1, 0.0], [2.1, 1.9, 2.2]])
    widths = (6.0, 1.5)
    cols = G._gaussian_columns(g, widths, reach=8.0)
    coef = np.array([0.7, 0.2, -0.3, 0.5])
    g = g.with_values((cols @ coef).reshape(g.shape))
    fit, c, err = G.fit_atom_gaussians(g, widths, ridge=1e-12, reach=8.0)
    assert err < 1e-6
    assert np.allclose(c.ravel(), coef, atol=1e-5)
    assert np.allclose(fit.values, g.values, atol=1e-6)


def test_fit_leaves_an_error_on_a_field_that_is_not_atom_centred():
    g = G.GridField(np.zeros((16, 16, 16)), cubic(4.0), symbols=["A"],
                    positions=[[0.0, 0.0, 0.0]])
    pts = g.points()
    lobe = np.exp(-2.0 * np.sum((pts - [2.0, 2.0, 2.0]) ** 2, axis=1))   # between images
    g = g.with_values(lobe.reshape(g.shape))
    _fit, _c, err = G.fit_atom_gaussians(g, (8.0, 2.0, 0.5))
    assert err > 0.3
