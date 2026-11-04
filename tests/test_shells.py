"""Tests for cmfqsar/shells.py: solvation shells cut from a bulk electrolyte trajectory.

Every trajectory here is built for the purpose, so the tests need no simulation package and no
downloaded data. The geometry is chosen so that each claim can fail: a molecule is placed with
its donor atom at an exact distance from the ion, which makes the coordination number and the
distances assertions rather than impressions.

The claim that matters most is the periodic repair. A bulk simulation writes every atom wrapped
into the cell, so an ion near a face has its shell on the other side of the box and a molecule
across a face arrives in two pieces. Both produce a geometry rather than an error, which is how
a periodic-boundary artefact becomes a wrong conclusion about an electrolyte.
`test_a_shell_across_a_face_is_found_whole` and `test_a_split_molecule_comes_back_whole` are the
tests that fail if those two repairs are not made.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar import shells as SH                                             # noqa: E402
from cmfqsar import solvation as SV                                          # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet                               # noqa: E402
from cmfqsar.kernels import kernel_matrices, self_kernels                    # noqa: E402
from cmfqsar.trajectory import Trajectory                                    # noqa: E402

BOX = 24.0
DME = "COCCOC"
DMC = "COC(=O)OC"
PF6 = "F[P-](F)(F)(F)(F)F"


# ------------------------------------------------------------------- building a box by hand


def _template(smiles, seed=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    return m


TEMPLATES = {"DME": _template(DME), "DMC": _template(DMC), "PF6": _template(PF6)}


def _between(a, b):
    """The rotation that takes unit vector `a` onto unit vector `b`."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def _placed(name, donor_at, ion_at=(12.0, 12.0, 12.0)):
    """The template with its best donor at `donor_at` and its lone pair pointing at the ion.

    Turning the molecule is not decoration. A molecule merely translated so that one oxygen is
    two angstrom from the ion can have its other atoms closer still, which is not a solvation
    shell and would make the distances in these tests meaningless.
    """
    t = TEMPLATES[name]
    pos = np.asarray(t.GetConformer().GetPositions(), float)
    idx = SV.best_donor(t)
    lone = SV.donor_direction(t, idx, pos)
    donor_at = np.asarray(donor_at, float)
    towards_ion = np.asarray(ion_at, float) - donor_at
    rot = _between(lone, towards_ion)
    return (pos - pos[idx]) @ rot.T + donor_at


def _rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _trajectory(members, frames=1, jitter=0.0, box=BOX, wrap=True, seed=0,
                rotate_each_frame=None):
    """One box. `members` is a list of (residue name, coordinates) with "LI" for a bare ion."""
    rng = np.random.default_rng(seed)
    elements, names, residues, resids, bonds = [], [], [], [], []
    blocks = []
    for r, (name, coords) in enumerate(members):
        start = len(elements)
        coords = np.atleast_2d(np.asarray(coords, float))
        if name == "LI":
            els = ["Li"]
            bs = []
        else:
            t = TEMPLATES[name]
            els = [a.GetSymbol() for a in t.GetAtoms()]
            bs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in t.GetBonds()]
        assert len(els) == coords.shape[0], f"{name}: {len(els)} atoms, {coords.shape[0]} rows"
        elements.extend(els)
        names.extend(f"{name}{i}" for i in range(len(els)))
        residues.extend([name] * len(els))
        resids.extend([r] * len(els))
        bonds.extend((start + i, start + j) for i, j in bs)
        blocks.append(coords)
    base = np.vstack(blocks)

    out = []
    for k in range(frames):
        xyz = base.copy()
        if rotate_each_frame is not None and k > 0:
            R = _rotation([0.3, 0.7, 0.5], rotate_each_frame * k)
            centre = xyz.mean(axis=0)
            xyz = (xyz - centre) @ R.T + centre
        if jitter > 0:
            xyz = xyz + rng.normal(scale=jitter, size=xyz.shape)
        out.append(np.mod(xyz, box) if wrap else xyz)
    cell = np.tile(np.eye(3) * box, (frames, 1, 1))
    return Trajectory(coords=np.asarray(out, float), elements=elements, atom_names=names,
                      residues=residues, resids=np.asarray(resids, int), bonds=bonds,
                      box=cell, engine="constructed")


def _simple_box(ion_at=(12.0, 12.0, 12.0), distances=(2.0, 2.1, 2.2), name="DME", **kw):
    ion = np.asarray(ion_at, float)
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    members = [("LI", ion[None, :])]
    for k, d in enumerate(distances):
        members.append((name, _placed(name, ion + dirs[k % len(dirs)] * d, ion_at=ion)))
    return _trajectory(members, **kw)


FIELDS = FieldSet.create(alpha=0.3, charge_model="gasteiger")


def _cosine(a, b):
    ta, tb = AtomTable.from_molecules([a]), AtomTable.from_molecules([b])
    kaa = float(np.sum(self_kernels(ta, FIELDS.specs)))
    kbb = float(np.sum(self_kernels(tb, FIELDS.specs)))
    kab = float(np.sum(kernel_matrices(ta, tb, FIELDS.specs)))
    return kab / np.sqrt(kaa * kbb)


# ------------------------------------------------------------------------------ the pieces


def test_the_ion_atoms_are_found():
    traj = _simple_box()
    assert SH.ion_atoms(traj, "Li").tolist() == [0]
    assert SH.ion_atoms(traj, "Na").tolist() == []


def test_residues_come_back_in_the_order_they_appear():
    traj = _simple_box(distances=(2.0, 2.1))
    groups = SH.residue_groups(traj)
    assert len(groups) == 3
    assert groups[0].tolist() == [0]
    assert [traj.residues[g[0]] for g in groups] == ["LI", "DME", "DME"]
    assert sum(len(g) for g in groups) == traj.n_atoms


def test_the_cutoff_is_the_coordination_distance_of_item_bt4():
    assert SH.shell_cutoff("Li") == pytest.approx(SV.ION_OXYGEN_DISTANCE["Li"] * 1.3)
    assert SH.shell_cutoff("Na", 1.0) == pytest.approx(SV.ION_OXYGEN_DISTANCE["Na"])
    with pytest.raises(SH.ShellError):
        SH.shell_cutoff("Fe")


def test_an_unknown_ion_is_refused_by_name():
    traj = _simple_box()
    with pytest.raises(SH.ShellError):
        SH.extract_shells(traj, "Cs")


# ------------------------------------------------------------------------------- the cut


def test_the_molecules_inside_the_cutoff_are_the_ones_kept():
    # three at 2.0 to 2.2 angstrom are inside the 2.61 cutoff, one at 4.0 is outside
    traj = _simple_box(distances=(2.0, 2.1, 2.2, 4.0))
    sh = SH.extract_shells(traj, "Li")[0]
    assert sh.n_molecules == 3
    assert sh.coordination == 3
    assert sorted(round(d, 3) for d in sh.donor_distances) == [2.0, 2.1, 2.2]


def test_the_shell_is_ordered_with_the_closest_molecule_first():
    traj = _simple_box(distances=(2.2, 2.0, 2.1))
    sh = SH.extract_shells(traj, "Li")[0]
    assert [round(d, 3) for d in sh.donor_distances] == [2.0, 2.1, 2.2]


def test_a_shell_across_a_face_is_found_whole():
    """An ion beside a face has its shell wrapped to the other side of the box.

    Without reimage the molecules look a box length away and the shell comes back empty.
    """
    traj = _simple_box(ion_at=(23.5, 12.0, 12.0), distances=(2.0, 2.1, 2.2))
    sh = SH.extract_shells(traj, "Li")[0]
    assert sh.n_molecules == 3
    assert sorted(round(d, 3) for d in sh.donor_distances) == [2.0, 2.1, 2.2]


def test_a_split_molecule_comes_back_whole():
    """A molecule straddling a face arrives in two pieces; make_whole puts it back.

    The test is on a bond length: a piece a box length away would make one bond 24 angstrom.
    """
    traj = _simple_box(ion_at=(0.2, 12.0, 12.0), distances=(2.0,))
    sh = SH.extract_shells(traj, "Li")[0]
    mol = SH.shell_molecule(sh, TEMPLATES)
    pos = np.asarray(mol.GetConformer().GetPositions(), float)
    lengths = [float(np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]))
               for b in mol.GetBonds()]
    assert max(lengths) < 2.0, f"a bond of {max(lengths):.1f} angstrom means a split molecule"


def test_an_anion_that_comes_close_is_part_of_the_shell():
    ion = np.array([12.0, 12.0, 12.0])
    members = [("LI", ion[None, :]),
               ("DMC", _placed("DMC", ion + np.array([2.0, 0.0, 0.0]))),
               ("PF6", _placed("PF6", ion + np.array([0.0, 2.0, 0.0])))]
    traj = _trajectory(members)
    sh = SH.extract_shells(traj, "Li")[0]
    assert sorted(sh.species) == ["DMC", "PF6"]
    mol = SH.shell_molecule(sh, TEMPLATES)
    assert Chem.GetFormalCharge(mol) == 0, "Li(+) with PF6(-) is a neutral contact ion pair"


def test_max_molecules_keeps_the_closest():
    traj = _simple_box(distances=(2.0, 2.1, 2.2))
    sh = SH.extract_shells(traj, "Li", max_molecules=2)[0]
    assert sh.n_molecules == 2
    assert sorted(round(d, 3) for d in sh.donor_distances) == [2.0, 2.1]


def test_a_shell_with_too_few_molecules_is_dropped():
    traj = _simple_box(distances=(5.0,))
    assert SH.extract_shells(traj, "Li", min_molecules=1) == []
    kept = SH.extract_shells(traj, "Li", min_molecules=0)
    assert len(kept) == 1 and kept[0].n_molecules == 0


def test_every_frame_and_every_ion_gives_a_shell():
    ion_a = np.array([6.0, 6.0, 6.0])
    ion_b = np.array([18.0, 18.0, 18.0])
    members = [("LI", ion_a[None, :]), ("LI", ion_b[None, :]),
               ("DME", _placed("DME", ion_a + np.array([2.0, 0.0, 0.0]), ion_at=ion_a)),
               ("DME", _placed("DME", ion_b + np.array([2.0, 0.0, 0.0]), ion_at=ion_b))]
    traj = _trajectory(members, frames=5, jitter=0.02)
    sh = SH.extract_shells(traj, "Li")
    assert len(sh) == 10
    assert {s.frame for s in sh} == {0, 1, 2, 3, 4}
    assert {s.ion_atom for s in sh} == {0, 1}


def test_the_stride_takes_every_nth_frame():
    traj = _simple_box(distances=(2.0,), frames=10, jitter=0.02)
    assert len(SH.extract_shells(traj, "Li", stride=5)) == 2


# ------------------------------------------------------------------- a shell as a molecule


def test_a_shell_becomes_a_molecule_with_the_ion_first():
    traj = _simple_box(distances=(2.0, 2.1))
    sh = SH.extract_shells(traj, "Li")[0]
    mol = SH.shell_molecule(sh, TEMPLATES)
    assert mol.GetAtomWithIdx(0).GetSymbol() == "Li"
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 1
    assert mol.GetNumAtoms() == 1 + 2 * TEMPLATES["DME"].GetNumAtoms()
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)).count("COCCOC") == 2


def test_the_shell_molecule_keeps_the_distances_the_box_had():
    traj = _simple_box(distances=(2.0, 2.2))
    sh = SH.extract_shells(traj, "Li")[0]
    mol = SH.shell_molecule(sh, TEMPLATES)
    pos = np.asarray(mol.GetConformer().GetPositions(), float)
    found, _ = SV.coordination(mol, 0, "Li")
    d = sorted(float(np.linalg.norm(pos[int(i)] - pos[0])) for i in found)
    assert d == pytest.approx([2.0, 2.2], abs=1e-6)


def test_a_species_without_a_template_is_refused_by_name():
    traj = _simple_box(distances=(2.0,))
    sh = SH.extract_shells(traj, "Li")[0]
    with pytest.raises(SH.ShellError) as err:
        SH.shell_molecule(sh, {"DMC": TEMPLATES["DMC"]})
    assert "DME" in str(err.value)


def test_a_template_with_the_wrong_atoms_is_refused():
    traj = _simple_box(distances=(2.0,))
    with pytest.raises(SH.ShellError) as err:
        SH.check_templates(traj, {"DME": TEMPLATES["DMC"]})
    assert "atoms" in str(err.value) or "differ at" in str(err.value)
    SH.check_templates(traj, TEMPLATES)                     # the right ones pass


# ------------------------------------------------------------------------- the thinning


def test_the_fingerprint_does_not_depend_on_the_order_of_the_molecules():
    ion = np.array([12.0, 12.0, 12.0])
    a = _trajectory([("LI", ion[None, :]),
                     ("DME", _placed("DME", ion + np.array([2.0, 0.0, 0.0]))),
                     ("DMC", _placed("DMC", ion + np.array([0.0, 2.2, 0.0])))])
    b = _trajectory([("LI", ion[None, :]),
                     ("DMC", _placed("DMC", ion + np.array([0.0, 2.2, 0.0]))),
                     ("DME", _placed("DME", ion + np.array([2.0, 0.0, 0.0])))])
    fa = SH.extract_shells(a, "Li")[0].fingerprint()
    fb = SH.extract_shells(b, "Li")[0].fingerprint()
    assert fa.shape == fb.shape
    assert np.allclose(np.sort(fa), np.sort(fb), atol=1e-6)


def test_identical_shells_collapse_to_one_member():
    traj = _simple_box(distances=(2.0, 2.1), frames=6, jitter=0.0)
    sh = SH.extract_shells(traj, "Li")
    reps, w = SH.cluster_shells(sh, max_members=8)
    assert len(reps) == 1
    assert w.tolist() == [1.0]


def test_the_weights_are_populations():
    """Four shells of one composition and one of another give weights 0.8 and 0.2."""
    ion = np.array([12.0, 12.0, 12.0])
    one = [("LI", ion[None, :]), ("DME", _placed("DME", ion + np.array([2.0, 0.0, 0.0])))]
    two = one + [("DMC", _placed("DMC", ion + np.array([0.0, 2.1, 0.0])))]
    many = SH.extract_shells(_trajectory(one, frames=4), "Li")
    few = SH.extract_shells(_trajectory(two, frames=1), "Li")
    reps, w = SH.cluster_shells(many + few, max_members=8)
    assert len(reps) == 2
    assert sorted(round(float(v), 6) for v in w) == [0.2, 0.8]


def test_the_richest_composition_comes_first():
    ion = np.array([12.0, 12.0, 12.0])
    one = [("LI", ion[None, :]), ("DME", _placed("DME", ion + np.array([2.0, 0.0, 0.0])))]
    two = one + [("DMC", _placed("DMC", ion + np.array([0.0, 2.1, 0.0])))]
    shells_ = (SH.extract_shells(_trajectory(one, frames=9), "Li")
               + SH.extract_shells(_trajectory(two, frames=1), "Li"))
    reps, _ = SH.cluster_shells(shells_, max_members=8)
    assert len(shells_[reps[0]].composition) == 2, "the two-species shell has to be first"


def test_the_member_cap_is_respected():
    traj = _simple_box(distances=(2.0, 2.4, 2.5), frames=40, jitter=0.25, seed=3)
    sh = SH.extract_shells(traj, "Li")
    reps, w = SH.cluster_shells(sh, max_members=5, radius=0.05)
    assert 1 <= len(reps) <= 5
    assert float(w.sum()) == pytest.approx(1.0)


def test_the_composition_table_sums_to_one():
    traj = _simple_box(distances=(2.0, 2.5, 2.6), frames=20, jitter=0.2, seed=1)
    sh = SH.extract_shells(traj, "Li")
    table = SH.composition_table(sh)
    assert sum(share for _, share in table) == pytest.approx(1.0)
    assert table == sorted(table, key=lambda t: (-t[1], t[0]))


# ---------------------------------------------------------------------------- the atom map


def test_the_atom_map_sends_the_ion_to_the_ion_and_a_missing_species_to_minus_one():
    ion = np.array([12.0, 12.0, 12.0])
    rich = SH.extract_shells(_trajectory(
        [("LI", ion[None, :]),
         ("DME", _placed("DME", ion + np.array([2.0, 0.0, 0.0])))], frames=1), "Li")[0]
    other = SH.extract_shells(_trajectory(
        [("LI", ion[None, :]),
         ("DMC", _placed("DMC", ion + np.array([2.0, 0.0, 0.0])))], frames=1), "Li")[0]
    maps = SH.atom_maps([rich, other], TEMPLATES)
    assert maps[0][0] == 0 and maps[1][0] == 0
    assert maps[0].tolist() == list(range(rich.n_atoms))
    assert set(maps[1][1:].tolist()) == {-1}, "DMC has no counterpart in a DME shell"


def test_a_second_molecule_of_a_species_maps_onto_the_first():
    ion = np.array([12.0, 12.0, 12.0])
    sh = SH.extract_shells(_trajectory(
        [("LI", ion[None, :]),
         ("DME", _placed("DME", ion + np.array([2.0, 0.0, 0.0]))),
         ("DME", _placed("DME", ion + np.array([0.0, 2.1, 0.0])))], frames=1), "Li")[0]
    amap = SH.atom_maps([sh], TEMPLATES)[0]
    n = TEMPLATES["DME"].GetNumAtoms()
    assert amap[1:1 + n].tolist() == amap[1 + n:1 + 2 * n].tolist()


# ------------------------------------------------------------------------------ the ensemble


def test_the_ensemble_is_an_ensemble_of_members_with_their_own_atoms():
    traj = _simple_box(distances=(2.0, 2.5, 2.6), frames=16, jitter=0.2, seed=5)
    sh = SH.extract_shells(traj, "Li")
    ens = SH.shell_ensemble(sh, TEMPLATES, FIELDS, max_members=6, radius=0.1)
    f = ens.fields
    assert f.is_ensemble
    assert f.member_kind == "shells"
    assert f.n_conformers == len(ens.weights)
    assert float(np.sum(f.conformer_weights)) == pytest.approx(1.0)
    assert int(np.sum(f.sizes)) == f.n_atoms
    assert ens.n_shells == len(sh)


def test_an_ensemble_of_one_shape_is_that_shape():
    """Every shell the same geometry: the mean field must be the field of that geometry."""
    traj = _simple_box(distances=(2.0, 2.1), frames=5, jitter=0.0)
    sh = SH.extract_shells(traj, "Li")
    ens = SH.shell_ensemble(sh, TEMPLATES, FIELDS, max_members=6)
    single = SH.shell_molecule(sh[0], TEMPLATES)
    from cmfqsar.fields import build_fields
    one = build_fields(SH._in_cation_frame(single), FIELDS, name="one")
    assert _cosine(ens.fields, one) == pytest.approx(1.0, abs=1e-6)


def test_the_frame_makes_a_rotated_box_give_the_same_field():
    """The same shell seen rotated is the same chemistry, so the framed field must match."""
    ion = np.array([12.0, 12.0, 12.0])
    members = [("LI", ion[None, :]),
               ("DME", _placed("DME", ion + np.array([2.0, 0.0, 0.0]))),
               ("DMC", _placed("DMC", ion + np.array([0.0, 2.2, 0.0]))),
               ("PF6", _placed("PF6", ion + np.array([0.0, 0.0, 2.3])))]
    plain = _trajectory(members, frames=1)
    turned = _trajectory(members, frames=2, rotate_each_frame=0.7)
    a = SH.shell_ensemble(SH.extract_shells(plain, "Li"), TEMPLATES, FIELDS, max_members=1)
    b = SH.shell_ensemble([SH.extract_shells(turned, "Li")[1]], TEMPLATES, FIELDS,
                          max_members=1)
    assert _cosine(a.fields, b.fields) > 0.99


def test_the_block_error_is_small_for_a_steady_ensemble():
    traj = _simple_box(distances=(2.0, 2.1, 2.2), frames=20, jitter=0.03, seed=11)
    sh = SH.extract_shells(traj, "Li")
    err = SH.block_error(sh, TEMPLATES, FIELDS, n_blocks=4, max_members=4)
    assert 0.0 <= err < 0.02, f"a steady box should not report {err:.3f}"


def test_the_block_error_notices_a_drifting_ensemble():
    """Frames whose shell keeps changing have a larger error of the mean than steady ones."""
    ion = np.array([12.0, 12.0, 12.0])
    steady, drifting = [], []
    for k in range(16):
        d = 2.0 + 0.004 * k
        steady.append(_trajectory([("LI", ion[None, :]),
                                   ("DME", _placed("DME", ion + np.array([d, 0.0, 0.0])))],
                                  frames=1))
        dd = 1.9 + 0.045 * k
        drifting.append(_trajectory([("LI", ion[None, :]),
                                     ("DME", _placed("DME", ion + np.array([dd, 0.0, 0.0])))],
                                    frames=1))

    def collect(boxes):
        out = []
        for k, t in enumerate(boxes):
            for s in SH.extract_shells(t, "Li"):
                s.frame = k
                out.append(s)
        return out

    a = SH.block_error(collect(steady), TEMPLATES, FIELDS, n_blocks=4, max_members=4,
                       radius=0.01)
    b = SH.block_error(collect(drifting), TEMPLATES, FIELDS, n_blocks=4, max_members=4,
                       radius=0.01)
    assert b > a, f"the drifting box reported {b:.5f} against {a:.5f} for the steady one"


def test_the_thinning_error_is_zero_when_nothing_is_thinned():
    """Keeping every distinct shape as a member leaves no thinning error to report."""
    traj = _simple_box(distances=(2.0, 2.1), frames=6, jitter=0.0)
    sh = SH.extract_shells(traj, "Li")
    out = SH.convergence(sh, TEMPLATES, FIELDS, n_blocks=3, sample=6, max_members=8)
    assert out["thinning_error"] == pytest.approx(0.0, abs=1e-8)
    assert out["members"] == 1.0


def test_the_thinning_error_grows_when_fewer_members_are_kept():
    traj = _simple_box(distances=(2.0, 2.4, 2.5), frames=24, jitter=0.3, seed=4)
    sh = SH.extract_shells(traj, "Li")
    coarse = SH.convergence(sh, TEMPLATES, FIELDS, n_blocks=3, sample=24, max_members=1,
                            radius=0.02)
    fine = SH.convergence(sh, TEMPLATES, FIELDS, n_blocks=3, sample=24, max_members=12,
                          radius=0.02)
    assert coarse["thinning_error"] >= fine["thinning_error"]


def test_too_few_frames_for_the_blocks_is_refused():
    traj = _simple_box(distances=(2.0,), frames=3, jitter=0.02)
    sh = SH.extract_shells(traj, "Li")
    with pytest.raises(SH.ShellError):
        SH.block_error(sh, TEMPLATES, FIELDS, n_blocks=5)
    with pytest.raises(SH.ShellError):
        SH.block_error(sh, TEMPLATES, FIELDS, n_blocks=1)


def test_the_ensemble_reports_the_shells_its_members_were_made_from():
    """The caller must not have to repeat the thinning to learn which shells were kept."""
    traj = _simple_box(distances=(2.0, 2.5, 2.6), frames=14, jitter=0.2, seed=7)
    sh = SH.extract_shells(traj, "Li")
    ens = SH.shell_ensemble(sh, TEMPLATES, FIELDS, max_members=5, radius=0.1)
    assert len(ens.chosen) == len(ens.members) == len(ens.weights)
    for shell, mol in zip(ens.chosen, ens.members):
        assert mol.GetNumAtoms() == shell.n_atoms
        assert mol.GetProp("shell_composition") == " ".join(
            f"{s}:{n}" for s, n in shell.composition)


def test_an_ensemble_survives_a_round_trip_through_a_file(tmp_path):
    """Written out with its three properties and read back, the mean field has to be the same.

    This is what lets the repository hold the shells rather than the trajectory they came from.
    """
    traj = _simple_box(distances=(2.0, 2.5, 2.6), frames=12, jitter=0.15, seed=9)
    sh = SH.extract_shells(traj, "Li")
    ens = SH.shell_ensemble(sh, TEMPLATES, FIELDS, max_members=5, radius=0.1, name="round")

    path = str(tmp_path / "members.sdf")
    writer = Chem.SDWriter(path)
    for mol, shell, weight in zip(ens.members, ens.chosen, ens.weights):
        out = Chem.Mol(mol)
        for key, value in SH.member_properties(shell, weight).items():
            out.SetProp(key, value)
        writer.write(out)
    writer.close()

    back = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    assert len(back) == len(ens.members)
    rebuilt = SH.ensemble_from_molecules(back, FIELDS, name="round")
    assert rebuilt.n_atoms == ens.fields.n_atoms
    assert _cosine(rebuilt, ens.fields) == pytest.approx(1.0, abs=1e-6)


def test_a_member_without_its_properties_is_refused(tmp_path):
    traj = _simple_box(distances=(2.0,), frames=1)
    sh = SH.extract_shells(traj, "Li")
    mol = SH.shell_molecule(sh[0], TEMPLATES)
    with pytest.raises(SH.ShellError) as err:
        SH.ensemble_from_molecules([mol], FIELDS)
    assert "shell_weight" in str(err.value)


def test_no_energy_is_read_from_a_frame():
    """The weights are populations, as item D7 requires; nothing here may be a Boltzmann factor.

    A shell carries no energy at all, so the test is that the dataclass has no such field.
    """
    traj = _simple_box(distances=(2.0,), frames=2)
    sh = SH.extract_shells(traj, "Li")[0]
    banned = [n for n in vars(sh) if "energ" in n.lower() or n.lower() in ("kt", "beta")]
    assert banned == [], f"a shell must not carry {banned}"
