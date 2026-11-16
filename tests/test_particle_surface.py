"""Plan item NT5: the surface kernel on the surface of a particle (ParticleSurfaceSpec).

A flat core reproduces the flat surface kernel exactly; a sphere and a cylinder make the channels
invariant to turning and moving the whole particle; the heights and the distances along the
surface are those of the geometry; the model takes the kernel with kernel_source="particle".
"""
import numpy as np
import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("scipy")
from ase import Atoms  # noqa: E402
from ase.build import fcc100, fcc111, molecule  # noqa: E402
from ase.cluster import Icosahedron  # noqa: E402

from cmfqsar.invariant import (CoreSurface, ParticleSurfaceSpec, SurfaceSpec,  # noqa: E402
                               fit_core_surface, particle_surface_autocorrelation,
                               particle_surface_feature_matrix, surface_autocorrelation,
                               with_core)
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.periodic import crystal_table, inorganic_fieldset  # noqa: E402

GRID = dict(d_max=8.0, step=1.0, sigma=1.0, z_min=2.0, z_max=8.0, z_step=2.0, z_sigma=2.0)


def _fs():
    return inorganic_fieldset([], physchem=("steric", "prop:electronegativity"))


def _rec(atoms, name="x"):
    return crystal_table([atoms], _fs(), [name]).molecules[0]


def _turn(X, angles=(0.4, 1.1, -0.7), shift=(3.0, -2.0, 1.0)):
    from scipy.spatial.transform import Rotation
    return X @ Rotation.from_euler("xyz", angles).as_matrix().T + np.array(shift)


@pytest.mark.parametrize("build", [fcc111, fcc100])
def test_a_flat_core_gives_the_flat_surface_kernel_exactly(build):
    slab = build("Au", size=(5, 5, 3), vacuum=0.0)
    slab.positions[:, 2] -= slab.positions[:, 2].max()
    m = molecule("CH3CH2OH")
    m.positions += [5.0, 4.0, 2.5 - m.positions[:, 2].min()]
    rec = _rec(m)
    flat = surface_autocorrelation(rec, SurfaceSpec(**GRID))
    on_plane = particle_surface_autocorrelation(with_core(rec, slab),
                                                ParticleSurfaceSpec(shape="plane", **GRID))
    assert np.array_equal(flat, on_plane)


def test_the_surface_kernel_is_the_formula():
    m = Atoms("CON", positions=[[0.0, 0.0, 2.0], [1.5, 0.0, 3.0], [0.0, 2.0, 4.5]])
    rec = _rec(m)
    spec = SurfaceSpec(**GRID)
    W, X = np.asarray(rec.weights), np.asarray(rec.coords)
    expect = np.zeros((spec.centres.size, spec.heights.size, spec.heights.size))
    for a in range(3):
        for b in range(3):
            rho = np.hypot(*(X[a, :2] - X[b, :2]))
            g = np.exp(-(rho - spec.centres) ** 2 / (2 * spec.sigma ** 2))
            za = np.exp(-(X[a, 2] - spec.heights) ** 2 / (2 * spec.z_sigma ** 2))
            zb = np.exp(-(X[b, 2] - spec.heights) ** 2 / (2 * spec.z_sigma ** 2))
            expect += W[a, 0] * W[b, 0] * g[:, None, None] * za[None, :, None] * zb[None, None, :]
    got = surface_autocorrelation(rec, spec)[:expect.size].reshape(expect.shape)
    assert np.allclose(got, expect, rtol=1e-12, atol=0)


@pytest.mark.parametrize("shape", ["sphere", "cylinder"])
def test_turning_and_moving_the_particle_changes_nothing(shape):
    if shape == "sphere":
        core = Icosahedron("Au", 4).positions
        core = core - core.mean(axis=0)
        place = [0.0, 0.0, 10.0]
    else:
        from ase.build import bulk
        core = bulk("Au", "fcc", a=4.08, cubic=True).repeat((3, 3, 6)).positions
        core = core - core.mean(axis=0)
        core = core[np.hypot(core[:, 0], core[:, 1]) <= 5.0]
        place = [9.0, 0.0, 1.0]
    m = molecule("CH3CH2OH")
    m.positions += np.array(place) - m.positions.mean(axis=0)
    spec = ParticleSurfaceSpec(shape=shape, **GRID)
    a = particle_surface_autocorrelation(_rec(m), spec, core=core)
    m2 = m.copy()
    m2.positions = _turn(m.positions)
    b = particle_surface_autocorrelation(_rec(m2), spec, core=_turn(core))
    assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(a))


def test_heights_and_arcs_of_a_sphere():
    s = CoreSurface("sphere", centre=np.zeros(3), axis=np.array([0.0, 0.0, 1.0]), radius=10.0)
    X = np.array([[0.0, 0.0, 12.0], [13.0, 0.0, 0.0], [0.0, 0.0, -10.5]])
    assert np.allclose(s.heights(X), [2.0, 3.0, 0.5])
    A = s.along(X)
    assert np.isclose(A[0, 1], 10.0 * np.pi / 2) and np.isclose(A[0, 2], 10.0 * np.pi)
    assert np.allclose(np.diag(A), 0.0)


def test_heights_and_paths_of_a_cylinder():
    s = CoreSurface("cylinder", centre=np.zeros(3), axis=np.array([0.0, 0.0, 1.0]), radius=4.0)
    X = np.array([[6.0, 0.0, 0.0], [6.0, 0.0, 3.0], [-5.0, 0.0, 0.0]])
    assert np.allclose(s.heights(X), [2.0, 2.0, 1.0])
    A = s.along(X)
    assert np.isclose(A[0, 1], 3.0) and np.isclose(A[0, 2], 4.0 * np.pi)


def test_the_fitted_sphere_and_plane():
    core = Icosahedron("Au", 5).positions
    surf = fit_core_surface(core + [1.0, 2.0, 3.0], "sphere")
    assert np.allclose(surf.centre, core.mean(axis=0) + [1.0, 2.0, 3.0])
    radii = np.linalg.norm(core - core.mean(axis=0), axis=1)
    assert radii.max() * 0.85 < surf.radius <= radii.max()
    slab = fcc111("Au", size=(4, 4, 3), vacuum=0.0)
    below = fit_core_surface(slab.positions, "plane",
                             towards=np.array([[0.0, 0.0, -20.0]]))
    assert np.allclose(below.axis, [0.0, 0.0, -1.0])        # the normal faces the ligands
    with pytest.raises(ValueError, match="unknown core shape"):
        fit_core_surface(core, "cone")


def test_the_kernel_tends_to_the_flat_one_on_a_large_sphere():
    rng = np.random.default_rng(1)
    patch = np.column_stack([rng.uniform(-4, 4, 12), rng.uniform(-4, 4, 12),
                             rng.uniform(2.5, 7.0, 12)])
    atoms = Atoms("C" * 6 + "O" * 6, positions=patch)
    flat = surface_autocorrelation(_rec(atoms), SurfaceSpec(**GRID))
    diffs = []
    for radius in (20.0, 200.0):
        wrapped = []
        for x, y, z in patch:
            a, az = np.hypot(x, y) / radius, np.arctan2(y, x)
            d = np.array([np.sin(a) * np.cos(az), np.sin(a) * np.sin(az), np.cos(a)])
            wrapped.append((radius + z) * d)
        k = np.arange(2000) + 0.5
        phi, th = np.arccos(1 - 2 * k / 2000), np.pi * (1 + 5 ** 0.5) * k
        core = radius * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                                  np.cos(phi)], 1)
        row = particle_surface_autocorrelation(_rec(Atoms("C" * 6 + "O" * 6,
                                                          positions=np.array(wrapped))),
                                               ParticleSurfaceSpec(**GRID), core=core)
        diffs.append(np.max(np.abs(row - flat)) / np.max(np.abs(flat)))
    assert diffs[1] < diffs[0] / 30.0 and diffs[1] < 1e-3


def test_a_sample_without_a_core_is_refused():
    with pytest.raises(ValueError, match="with_core"):
        particle_surface_feature_matrix([_rec(molecule("H2O"))], ParticleSurfaceSpec(**GRID))


def _shells():
    core = Icosahedron("Au", 3).positions
    core = core - core.mean(axis=0)
    out = []
    rng = np.random.default_rng(0)
    for k in range(10):
        n = 3 + k % 4
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        pos = np.vstack([d * (6.5 + 1.4 * j) for d in dirs for j in range(2)])
        out.append(with_core(_rec(Atoms("CO" * n, positions=pos), f"s{k}"), core))
    return out


def test_the_model_takes_the_particle_kernel():
    from cmfqsar.fields import AtomTable
    recs = _shells()
    table = AtomTable.from_molecules(recs)
    y = np.array([len(r.coords) / 2.0 for r in recs])
    m = CMFModel(fieldset=_fs(), method="krr", normalize=True, kernel_source="particle",
                 ac_d_max=8.0, ac_sigma=1.0, surf_z_min=2.0, surf_z_max=8.0, surf_z_step=2.0,
                 params={"lam": 1e-3})
    m.fit(table, y, cv_folds=5)
    assert np.isfinite(m.stats.q2)
    expect = particle_surface_feature_matrix(recs, m._ac_spec())
    assert np.allclose(m._ac_features(table), expect)
    back = CMFModel.from_state(m.state())
    assert back.particle_shape == "sphere" and back.kernel_source == "particle"
    assert np.allclose(back.predict(table), m.predict(table))
    assert CMFModel(fieldset=_fs(), particle_shape="cylinder").fresh_copy().particle_shape \
        == "cylinder"
