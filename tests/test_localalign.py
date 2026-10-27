"""Local alignment: an environment posed on its centre and first shell (plan item LP3).

The sample is an atom in a molecule, and for one site -- an aromatic carbon of a six-membered
ring, a carboxyl carbon -- a frame can be fixed by a rule instead of fitted. What these tests fix
is that the rule is a rule: that it does not depend on where the molecule happens to lie or how it
is turned, that it averages the ambiguity it cannot settle instead of picking a side, that it
refuses a centre it cannot pose, and that the positions a co-field is read at are where the
chemistry says they are.
"""
import io
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.localalign import (BOND_MAX, RING_POSITIONS, RING_RADIUS, frame_from,  # noqa: E402
                                images_of, local_frames, overlap_kernel, pose, posed,
                                posed_environments, ring_frames, ring_positions, well_posed)
from cmfqsar.localenv import choose_lam, environment  # noqa: E402


def _mol(smiles, name="m", seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m, maxIters=400)
    m.SetProp("_Name", name)
    return m


@pytest.fixture(scope="module")
def anisole():
    fs = FieldSet.create()
    m = _mol("COc1ccccc1", "anisole")
    return m, build_fields(m, fs, name="anisole"), fs


def _rotate(fields, angle=0.7, shift=(3.0, -2.0, 1.0)):
    """The same molecule turned and moved: nothing about the site has changed."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R = R @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    out = type(fields)(**{k: v for k, v in vars(fields).items()})
    out.coords = np.asarray(fields.coords, float) @ R.T + np.asarray(shift, float)
    return out


# ----------------------------------------------------------------------------- the frame itself


def test_a_frame_is_orthonormal_and_right_handed():
    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-0.7, 1.2, 0.0]])
    R = frame_from(coords, 1, 2)
    assert R is not None
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-12)
    assert float(np.linalg.det(R)) == pytest.approx(1.0)
    assert R[0] == pytest.approx([1.0, 0.0, 0.0])


def test_three_atoms_on_a_line_have_no_frame():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert frame_from(coords, 1, 2) is None
    assert frame_from(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 1, 2) is None


def test_turning_and_moving_the_molecule_does_not_change_the_pose(anisole):
    """The pose is a rule about the site, so it cannot depend on how the molecule lies."""
    _m, fields, _fs = anisole
    turned = _rotate(fields)
    for atom in (2, 4, 1):
        a = posed(environment(fields, atom, rho=0.1))
        b = posed(environment(turned, atom, rho=0.1))
        assert a.n_atoms == b.n_atoms
        assert a.coords == pytest.approx(b.coords, abs=1e-6)
        assert a.weights == pytest.approx(b.weights, abs=1e-9)


def test_the_key_leaves_two_poses_for_a_symmetric_centre_and_three_for_a_methyl():
    fs = FieldSet.create()
    benzene = build_fields(_mol("c1ccccc1", "benzene"), fs, name="benzene")
    assert len(local_frames(environment(benzene, 0, rho=0.1))) == 2
    toluene = build_fields(_mol("Cc1ccccc1", "toluene"), fs, name="toluene")
    methyl = [i for i, t in enumerate(toluene.atom_types) if t == "C.3"][0]
    assert len(local_frames(environment(toluene, methyl, rho=0.1))) == 3


def test_a_centre_with_nothing_bonded_to_it_has_no_pose():
    fs = FieldSet.create()
    f = build_fields(_mol("CCO", "e"), fs, name="e")
    lone = type(f)(name="lone", coords=np.zeros((1, 3)), weights=f.weights[:1],
                   radii=f.radii[:1], symbols=[f.symbols[0]], atom_types=[f.atom_types[0]],
                   bonds=np.zeros((0, 2), int), charges=f.charges[:1])
    got = local_frames(lone)
    assert len(got) == 1 and np.allclose(got[0], np.eye(3))
    with pytest.raises(ValueError, match="no frame"):
        posed(lone, frames=[])


# ------------------------------------------------------------------------------- the ring site


def test_the_ring_rule_poses_an_aromatic_carbon_and_refuses_anything_else(anisole):
    m, fields, _fs = anisole
    for a in m.GetAtoms():
        env = environment(fields, a.GetIdx(), rho=0.1)
        frames = ring_frames(env)
        assert len(frames) == (2 if a.GetIsAromatic() else 0)
    fs = FieldSet.create()
    naph = _mol("c1ccc2ccccc2c1", "naphthalene")
    nf = build_fields(naph, fs, name="naphthalene")
    fused = [a.GetIdx() for a in naph.GetAtoms()
             if a.GetIsAromatic() and sum(1 for nb in a.GetNeighbors() if nb.GetIsAromatic()) == 3]
    assert fused, "naphthalene has two carbons where the rings meet"
    for i in fused:
        assert ring_frames(environment(nf, i, rho=0.1)) == []


def test_the_ring_pose_puts_the_substituent_on_x_and_the_ring_in_the_plane(anisole):
    m, fields, _fs = anisole
    ipso = [a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()
            and any(nb.GetSymbol() == "O" for nb in a.GetNeighbors())][0]
    env = environment(fields, ipso, rho=0.1)
    R = ring_frames(env)[0]
    c = np.asarray(env.coords, float) @ R.T
    near = [k for k in range(1, env.n_atoms) if np.linalg.norm(env.coords[k]) < BOND_MAX]
    oxygen = [k for k in near if env.symbols[k] == "O"][0]
    ring = [k for k in near if str(env.atom_types[k]).startswith("C.ar")]
    assert c[oxygen][0] > 1.2                      # the substituent is on +x
    assert abs(c[oxygen][1]) < 0.3 and abs(c[oxygen][2]) < 0.3
    assert len(ring) == 2
    for k in ring:
        assert c[k][0] < 0                         # the ring is on the other side
        assert abs(c[k][2]) < 0.1                  # and in the xy plane
    assert c[ring[0]][1] * c[ring[1]][1] < 0       # one neighbour each side of the x axis


def test_the_positions_a_cofield_is_read_at_are_where_the_atoms_are():
    """In the frame of one carbon of 1,4-dimethoxybenzene the far oxygen sits at 'para'."""
    fs = FieldSet.create()
    m = _mol("COc1ccc(OC)cc1", "dimethoxybenzene")
    fields = build_fields(m, fs, name="dimethoxybenzene")
    ipso = [a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()
            and any(nb.GetSymbol() == "O" for nb in a.GetNeighbors())][0]
    env = environment(fields, ipso, rho=0.05)
    R = ring_frames(env)[0]
    c = np.asarray(env.coords, float) @ R.T
    oxygens = [k for k in range(env.n_atoms) if env.symbols[k] == "O"]
    far = max(oxygens, key=lambda k: np.linalg.norm(env.coords[k]))
    assert np.linalg.norm(c[far] - RING_POSITIONS["para"]) < 0.4
    near = min(oxygens, key=lambda k: np.linalg.norm(env.coords[k]))
    assert np.linalg.norm(c[near] - RING_POSITIONS["ipso"]) < 0.4


def test_the_four_positions_follow_the_hexagon():
    p = ring_positions()
    assert set(p) == {"ipso", "ortho", "meta", "para"}
    assert p["ipso"] == pytest.approx([1.4, 0.0, 0.0])
    assert p["para"] == pytest.approx([-(RING_RADIUS + RING_RADIUS + 1.4), 0.0, 0.0])
    # the distance from the centre grows with the position, and para is further than meta
    d = {k: float(np.linalg.norm(v)) for k, v in p.items()}
    assert d["ipso"] < d["ortho"] < d["meta"] < d["para"]
    assert all(abs(v[2]) < 1e-12 for v in p.values())      # all in the ring plane


# ------------------------------------------------------------------- averaging the ambiguity


def test_two_poses_are_kept_as_one_mean_field(anisole):
    _m, fields, _fs = anisole
    env = environment(fields, 4, rho=0.1)
    e = posed(env, frames=ring_frames(env))
    assert e.n_conformers == 2
    assert e.n_atoms == 2 * env.n_atoms
    assert e.is_ensemble
    # the mean field halves every weight, since the two poses share it
    assert float(np.abs(e.weights).sum()) == pytest.approx(float(np.abs(env.weights).sum()),
                                                           rel=1e-9)
    one = posed(env, frames=ring_frames(env)[:1])
    assert not one.is_ensemble and one.n_atoms == env.n_atoms


def test_one_posed_environment_per_measured_atom(anisole):
    m, fields, _fs = anisole
    rows = [a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()]
    t = AtomTargets(np.zeros(len(rows), int), np.asarray(rows, int),
                    np.arange(len(rows), dtype=float), 1, "x", [np.zeros(m.GetNumAtoms())])
    envs = posed_environments([fields], t, rho=0.1, frames_of=ring_frames)
    assert len(envs) == len(rows)
    assert all(e.n_conformers == 2 for e in envs)
    counted = well_posed([fields], t, rho=0.1, frames_of=ring_frames)
    assert counted == {2: len(rows)}
    assert images_of(environment(fields, rows[0], rho=0.1), frames_of=ring_frames) == 2


def test_posing_an_environment_moves_only_its_coordinates(anisole):
    _m, fields, _fs = anisole
    env = environment(fields, 3, rho=0.1)
    R = ring_frames(env)[0]
    out = pose(env, R, name="turned")
    assert out.name == "turned"
    assert out.weights is env.weights                      # nothing else is touched
    assert out.coords == pytest.approx(np.asarray(env.coords, float) @ R.T)
    assert np.linalg.norm(out.coords, axis=1) == pytest.approx(
        np.linalg.norm(env.coords, axis=1))                # a rotation keeps every distance


# ------------------------------------------------------- the overlap kernel on posed samples


@pytest.fixture(scope="module")
def posed_set():
    """Aromatic carbons of a few substituted benzenes, posed, with a learnable target."""
    fs = FieldSet.create()
    smiles = ["COc1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Clc1ccccc1", "c1ccccc1", "Nc1ccccc1"]
    mols, fields = [], []
    for i, s in enumerate(smiles):
        m = _mol(s, f"m{i}", seed=i + 1)
        mols.append(m)
        fields.append(build_fields(m, fs, name=f"m{i}"))
    mol_i, atom_i, vals, per = [], [], [], []
    for k, m in enumerate(mols):
        v = np.full(m.GetNumAtoms(), np.nan)
        for a in m.GetAtoms():
            if not a.GetIsAromatic():
                continue
            # the target: does this carbon carry the substituent?  In the posed frame the answer
            # is at a fixed place -- 1.4 A along +x -- so a model over posed environments has to
            # get it right, and one with no frame has to work much harder
            ipso = any(nb.GetSymbol() not in ("H",) and not nb.GetIsAromatic()
                       for nb in a.GetNeighbors())
            v[a.GetIdx()] = 1.0 if ipso else 0.0
            mol_i.append(k)
            atom_i.append(a.GetIdx())
            vals.append(v[a.GetIdx()])
        per.append(v)
    t = AtomTargets(np.asarray(mol_i, int), np.asarray(atom_i, int), np.asarray(vals, float),
                    len(mols), "ipso", per)
    envs = posed_environments(fields, t, rho=0.1, frames_of=ring_frames)
    return fields, t, envs, fs


def test_the_overlap_kernel_of_posed_environments_is_a_kernel(posed_set):
    _fields, _t, envs, fs = posed_set
    _model, table, K = overlap_kernel(envs[:20], fs)
    assert K.shape == (20, 20)
    assert K == pytest.approx(K.T, abs=1e-9)
    assert np.isfinite(K).all()
    assert (np.diag(K) > 0).all()
    assert table.n_mol == 20
    w, _v = np.linalg.eigh(K)
    assert w.min() > -1e-6                                  # positive semi-definite


def test_turning_a_molecule_does_not_change_its_kernel_row(posed_set):
    """The pose is equivariant, so the kernel of the posed environments is invariant."""
    fields, t, envs, fs = posed_set
    turned = [_rotate(f, angle=1.1) for f in fields]
    other = posed_environments(turned, t, rho=0.1, frames_of=ring_frames)
    _m1, _t1, K = overlap_kernel(envs[:12], fs)
    _m2, _t2, K2 = overlap_kernel(other[:12], fs)
    assert K2 == pytest.approx(K, rel=1e-6, abs=1e-8)


def test_a_posed_model_learns_a_target_the_pose_makes_visible(posed_set):
    _fields, t, envs, fs = posed_set
    _model, _table, K = overlap_kernel(envs, fs)
    r = choose_lam(K, t, folds=3, seed=0, name="posed overlap")
    assert r.n_samples == len(t)
    assert r.q2 > 0.2


def test_the_cofield_is_linear_in_the_activities(posed_set):
    """Doubling every target doubles the co-field: the model is linear in the field."""
    from cmfqsar.cofields import evaluate_gaussians

    _fields, t, envs, fs = posed_set
    model, table, _K = overlap_kernel(envs, fs)
    pts = np.array([RING_POSITIONS[k] for k in ("ipso", "ortho", "meta", "para")])

    def read(y):
        model.fit(table, y, cv_folds=3, cv_seed=0, reuse_kernels=True)
        centres, coeffs, widths = model.cofield_coefficients(0)
        return evaluate_gaussians(pts, centres, coeffs, widths)

    y = np.asarray(t.y, float)
    one = read(y - y.mean())
    two = read(2.0 * (y - y.mean()))
    assert np.abs(one).max() > 0
    assert two == pytest.approx(2.0 * one, rel=1e-6)

# ------------------------------------------------- the co-field at places a chemist can name


def test_the_cofield_at_points_is_the_cofield_on_the_grid(posed_set):
    from cmfqsar.cofields import cofield_at_points, cofield_volume, make_grid

    _fields, t, envs, fs = posed_set
    model, table, _K = overlap_kernel(envs, fs)
    model.fit(table, np.asarray(t.y, float), cv_folds=3, cv_seed=0, reuse_kernels=True)
    grid = make_grid(np.array([[-2.0, -2.0, 0.0], [2.0, 2.0, 0.0]]), margin=0.0, spacing=1.0)
    volume = cofield_volume(model, 0, grid, include_scale=True)
    pts = grid.points()
    at = cofield_at_points(model, 0, pts, include_scale=True)
    assert at == pytest.approx(volume.ravel(), rel=1e-9, abs=1e-12)


def test_the_error_of_a_difference_is_smaller_than_the_two_errors_added(posed_set):
    """Two nearby positions move together, which is why the covariance cannot be ignored."""
    from cmfqsar.cofields import cofield_covariance_at, contrast_at

    _fields, t, envs, fs = posed_set
    model, table, _K = overlap_kernel(envs, fs)
    model.fit(table, np.asarray(t.y, float), cv_folds=3, cv_seed=0, reuse_kernels=True)
    pts = np.array([RING_POSITIONS["meta"], RING_POSITIONS["para"]])
    cov = cofield_covariance_at(model, 0, pts, sigma=1.0)
    assert cov == pytest.approx(cov.T)
    assert (np.diag(cov) >= 0).all()
    w, _v = np.linalg.eigh(cov)
    assert w.min() > -1e-9
    sd = np.sqrt(np.diag(cov))
    _v1, sd_one = contrast_at(model, 0, pts, (1.0, 0.0), sigma=1.0)
    assert sd_one == pytest.approx(sd[0], rel=1e-9)
    _d, sd_diff = contrast_at(model, 0, pts, (-1.0, 1.0), sigma=1.0)
    assert 0 < sd_diff < sd[0] + sd[1]


def test_a_contrast_is_the_weighted_sum_it_says_it_is(posed_set):
    from cmfqsar.cofields import cofield_at_points, contrast_at

    _fields, t, envs, fs = posed_set
    model, table, _K = overlap_kernel(envs, fs)
    model.fit(table, np.asarray(t.y, float), cv_folds=3, cv_seed=0, reuse_kernels=True)
    pts = np.array([RING_POSITIONS[k] for k in ("ipso", "ortho", "meta", "para")])
    v = cofield_at_points(model, 1, pts)
    w = (0.0, 0.5, -1.0, 0.5)
    got, sd = contrast_at(model, 1, pts, w, sigma=0.5)
    assert got == pytest.approx(float(np.asarray(w) @ v))
    assert sd > 0


def test_the_command_fits_the_posed_model_and_writes_a_cofield(tmp_path, capsys):
    from cmfqsar.cli import main

    path = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(path) as w:
        for i, smiles in enumerate(["COc1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Clc1ccccc1",
                                    "CCc1ccccc1"]):
            m = _mol(smiles, f"m{i}", seed=i + 1)
            v = []
            for a in m.GetAtoms():
                v.append("%.2f" % (120.0 + a.GetIdx()) if a.GetIsAromatic() else "n/a")
            m.SetProp("atom.dprop.shift", " ".join(v))
            w.write(m)
    cube = os.path.join(str(tmp_path), "co.cube")
    capsys.readouterr()
    main(["local", path, "--target", "shift", "--rho", "0.1", "--align", "--site", "ring",
          "--folds", "3", "--cofield", cube])
    text = capsys.readouterr().out
    assert "posed overlap" in text and "poses averaged" in text
    assert os.path.exists(cube)
    with io.open(cube, encoding="utf-8") as fh:
        assert "co-field" in fh.readline().lower()
