"""Tests for field complementarity with a protein pocket."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.complementarity import (clash_count, cofield_agreement, complementarity,  # noqa: E402
                                     consensus_profile, coulomb_potential, field_overlap,
                                     field_overlap_by_atom, interaction_profile, ionise,
                                     occupancy, pocket_fields, pocket_molecule,
                                     pocket_residues, profile_matrix, profile_similarity,
                                     series_scores, shape_surface, surface_complementarity)
from cmfqsar.fields import FieldSet, FieldSpec, MoleculeFields  # noqa: E402
from cmfqsar.protein import read_pdb  # noqa: E402

# aspartate, lysine and arginine side chains, enough for the ionisation rules
PDB = """\
ATOM      1  N   ASP A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ASP A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  CB  ASP A   1       2.000   1.420   0.000  1.00  0.00           C
ATOM      4  CG  ASP A   1       3.520   1.500   0.000  1.00  0.00           C
ATOM      5  OD1 ASP A   1       4.180   0.440   0.000  1.00  0.00           O
ATOM      6  OD2 ASP A   1       4.060   2.640   0.000  1.00  0.00           O
ATOM      7  C   ASP A   1       2.000  -0.780   1.190  1.00  0.00           C
ATOM      8  O   ASP A   1       1.400  -0.900   2.260  1.00  0.00           O
ATOM      9  N   LYS A   2       3.230  -1.290   1.080  1.00  0.00           N
ATOM     10  CA  LYS A   2       3.900  -2.030   2.140  1.00  0.00           C
ATOM     11  CB  LYS A   2       5.380  -2.180   1.790  1.00  0.00           C
ATOM     12  CG  LYS A   2       6.150  -2.950   2.860  1.00  0.00           C
ATOM     13  CD  LYS A   2       7.630  -3.060   2.510  1.00  0.00           C
ATOM     14  CE  LYS A   2       8.390  -3.830   3.580  1.00  0.00           C
ATOM     15  NZ  LYS A   2       9.830  -3.930   3.230  1.00  0.00           N
ATOM     16  C   LYS A   2       3.300  -3.420   2.330  1.00  0.00           C
ATOM     17  O   LYS A   2       3.100  -4.160   1.370  1.00  0.00           O
ATOM     18  N   ARG A   3       3.010  -3.790   3.570  1.00  0.00           N
ATOM     19  CA  ARG A   3       2.430  -5.090   3.900  1.00  0.00           C
ATOM     20  CB  ARG A   3       2.260  -5.220   5.410  1.00  0.00           C
ATOM     21  CG  ARG A   3       1.610  -6.530   5.840  1.00  0.00           C
ATOM     22  CD  ARG A   3       1.470  -6.620   7.350  1.00  0.00           C
ATOM     23  NE  ARG A   3       0.840  -7.880   7.760  1.00  0.00           N
ATOM     24  CZ  ARG A   3       0.580  -8.240   9.010  1.00  0.00           C
ATOM     25  NH1 ARG A   3       0.890  -7.440  10.020  1.00  0.00           N
ATOM     26  NH2 ARG A   3       0.010  -9.410   9.260  1.00  0.00           N
ATOM     27  C   ARG A   3       1.080  -5.280   3.220  1.00  0.00           C
ATOM     28  O   ARG A   3       0.460  -4.320   2.760  1.00  0.00           O
ATOM     29  OXT ARG A   3       0.610  -6.420   3.140  1.00  0.00           O
END
"""


def _point(coords, charges, name="probe", radius=1.7):
    n = len(coords)
    w = np.zeros((n, 1))
    w[:, 0] = charges
    return MoleculeFields(name=name, coords=np.array(coords, float), weights=w,
                          radii=np.full(n, radius), symbols=["C"] * n, atom_types=["C.3"] * n,
                          bonds=np.zeros((0, 2), int), charges=np.array(charges, float))


ONE_FIELD = FieldSet(specs=[FieldSpec(name="electrostatic", kind="gauss", alpha=0.5)])


def test_field_overlap_matches_numerical_integration():
    """The closed form is the same integral a grid would give."""
    a = _point([[0.0, 0, 0]], [1.0])
    b = _point([[1.3, 0.4, -0.2]], [-2.0])
    spec = ONE_FIELD.specs[0]
    exact = field_overlap(a, spec, 0, b, spec, 0)
    g = np.arange(-8.0, 8.0, 0.1)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    fa = 1.0 * np.exp(-0.5 * spec.alpha * ((pts - a.coords[0]) ** 2).sum(1))
    fb = -2.0 * np.exp(-0.5 * spec.alpha * ((pts - b.coords[0]) ** 2).sum(1))
    numeric = float((fa * fb).sum() * 0.1 ** 3)
    assert exact == pytest.approx(numeric, rel=2e-3)
    # an empty field contributes nothing
    assert field_overlap(_point([[0, 0, 0]], [0.0]), spec, 0, b, spec, 0) == 0.0


def test_complementarity_has_the_sign_of_the_chemistry():
    """A cation facing an anion is complementary; facing a cation it is not."""
    lig = _point([[0.0, 0, 0]], [+1.0], "cation")
    anion = _point([[4.0, 0, 0]], [-1.0], "anion")
    cation = _point([[4.0, 0, 0]], [+1.0], "cation pocket")
    good = complementarity(lig, anion, ONE_FIELD)
    bad = complementarity(lig, cation, ONE_FIELD)
    assert good.terms["electrostatic"] > 0 and bad.terms["electrostatic"] < 0
    assert good.terms["electrostatic"] == pytest.approx(-bad.terms["electrostatic"])
    assert good.score > 0 > bad.score
    # the normalised term does not grow when the same field is simply scaled up
    strong = _point([[0.0, 0, 0]], [+5.0], "stronger cation")
    assert complementarity(strong, anion, ONE_FIELD).terms["electrostatic"] == pytest.approx(
        good.terms["electrostatic"])
    assert "electrostatic" in good.describe()


def test_clashes_are_counted_separately_from_the_score():
    lig = _point([[0.0, 0, 0]], [1.0], radius=1.7)
    far = _point([[8.0, 0, 0]], [-1.0], radius=1.7)
    near = _point([[1.5, 0, 0]], [-1.0], radius=1.7)
    assert clash_count(lig, far)[0] == 0
    n, worst = clash_count(lig, near)
    assert n == 1 and worst == pytest.approx(1.7 + 1.7 - 0.5 - 1.5)


def test_coulomb_potential_falls_off_and_sums():
    q = coulomb_potential(np.array([[2.0, 0, 0], [4.0, 0, 0]]), np.zeros((1, 3)), [1.0])
    assert q[0] > q[1] > 0
    both = coulomb_potential(np.array([[0.0, 0, 3.0]]), np.array([[0.0, 0, 0], [1.0, 0, 0]]),
                             [1.0, -1.0])
    assert abs(float(both[0])) < 0.05           # a dipole nearly cancels at a distance
    assert coulomb_potential(np.zeros((0, 3)), np.zeros((1, 3)), [1.0]).shape == (0,)
    const = coulomb_potential(np.array([[2.0, 0, 0]]), np.zeros((1, 3)), [1.0],
                              dielectric="constant")
    assert const[0] > q[0]                      # 1/r is larger than 1/r^2 at this distance


def test_surface_is_closed_and_carries_a_score_per_vertex():
    lig = _point([[0.0, 0, 0]], [+1.0], radius=1.7)
    anion = _point([[4.5, 0, 0]], [-1.0], radius=1.7)
    dipole = _point([[0.0, 0, 0], [2.4, 0, 0]], [-1.0, +1.0], "dipole", radius=1.7)
    verts, faces = shape_surface(lig, spacing=0.25)
    assert len(verts) > 100 and len(faces) > 100
    r = np.linalg.norm(verts - lig.coords[0], axis=1)
    assert np.allclose(r, 1.7, atol=0.15)       # the surface sits on the van der Waals sphere
    out = surface_complementarity(lig, anion, ONE_FIELD, spacing=0.25)
    assert out is not None
    v, f, per_vertex, corr = out
    assert len(per_vertex) == len(v)
    assert np.all(np.abs(per_vertex) <= 1.0 + 1e-9)
    # the side facing the anion is the complementary one
    facing = v[:, 0] > lig.coords[0][0]
    assert per_vertex[facing].mean() > per_vertex[~facing].mean()
    # the correlation needs a ligand whose own potential varies over its surface; one sphere with
    # one charge has the same potential everywhere on it
    out2 = surface_complementarity(dipole, anion, ONE_FIELD, spacing=0.25)
    assert out2[3]["electrostatic"] > 0.5


def test_ionise_sets_the_side_chain_charges(tmp_path):
    path = tmp_path / "site.pdb"
    path.write_text(PDB)
    prot = read_pdb(str(path))
    mol = pocket_molecule(prot, add_hydrogens=False, ionise_residues=False)
    assert all(a.GetFormalCharge() == 0 for a in mol.GetAtoms())
    n = ionise(mol)
    assert n >= 3
    charged = {}
    for a in mol.GetAtoms():
        if a.GetFormalCharge():
            info = a.GetPDBResidueInfo()
            charged[(info.GetResidueName().strip(), info.GetName().strip())] = a.GetFormalCharge()
    assert charged.get(("ASP", "OD2")) == -1
    assert charged.get(("LYS", "NZ")) == +1
    assert charged.get(("ARG", "OXT")) == -1
    assert any(res == "ARG" and name.startswith("NH") and q == 1
               for (res, name), q in charged.items())
    # the fields built from it then carry those charges, with hydrogens on the donors
    fs = FieldSet.create(alpha=0.5)
    pf = pocket_fields(prot, fs)
    assert pf.n_atoms > prot.n_atoms                      # hydrogens were added
    f = [s.name for s in fs.specs].index("electrostatic")
    assert pf.weights[:, f].min() < -0.3 and pf.weights[:, f].max() > 0.1
    assert np.isfinite(pf.weights).all()


def test_series_scores_and_their_relation_to_activity():
    """Three ligands at the same place with different charges, against an anionic pocket."""
    anion = _point([[6.0, 0, 0]], [-1.0], "pocket")
    mols = [_point([[x, 0, 0]], [+1.0], f"x{x:.0f}") for x in (0.0, 1.0, 2.0)]
    activity = [1.0, 2.0, 3.0]                   # the closer to the anion, the more active
    res, stats = series_scores(mols, anion, ONE_FIELD, activity=activity)
    assert [r.name for r in res] == ["x0", "x1", "x2"]
    assert res[2].score > res[1].score > res[0].score > 0
    assert stats["r_score"] > 0.95 and stats["r2_score"] > 0.9
    # without activities only the scores come back
    res2, stats2 = series_scores(mols, anion, ONE_FIELD)
    assert stats2 == {}
    assert res2[2].score > res2[0].score
    # a molecule whose field is identically zero has no defined complementarity
    dead = series_scores([_point([[0.0, 0, 0]], [0.0], "neutral")], anion, ONE_FIELD)[0][0]
    assert not np.isfinite(dead.score)


def test_cofield_agreement_sees_a_model_that_matches_the_protein():
    """A model trained so that positive charge raises activity should agree with an anionic site."""
    from cmfqsar.cofields import make_grid
    from cmfqsar.fields import AtomTable
    from cmfqsar.model import CMFModel
    rng = np.random.RandomState(0)
    mols, y = [], []
    for i in range(8):
        q = -1.0 + 2.0 * i / 7.0
        mols.append(_point([[0.0, 0, 0], [1.5, 0, 0]], [q, 0.0], f"m{i}"))
        y.append(q)                              # activity follows the charge exactly
    fs = FieldSet(specs=[FieldSpec(name="electrostatic", kind="gauss", alpha=0.5)])
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 1e-6})
    model.fit(AtomTable.from_molecules(mols), np.array(y), cv_folds=0)
    grid = make_grid(model.train.coords, margin=4.0, spacing=0.5)
    w = occupancy(mols, grid).reshape(-1)
    anion = _point([[5.0, 0, 0]], [-1.0], "anion")
    cation = _point([[5.0, 0, 0]], [+1.0], "cation")
    good = cofield_agreement(model, anion, grid, w)["Electrostatic"]
    bad = cofield_agreement(model, cation, grid, w)["Electrostatic"]
    assert good > 0.99 and bad < -0.99          # the signs agree everywhere, or nowhere
    assert good == pytest.approx(-bad, abs=1e-6)
    # the cosine also asks for the shapes to match, and a localised co-field against a potential
    # that grows towards the anion does not match in shape, so it scores lower
    shape = cofield_agreement(model, anion, grid, w, statistic="cosine")["Electrostatic"]
    assert 0 < shape < good
    # the mean-centred variant asks whether the variations track, which here they do not
    trend = cofield_agreement(model, anion, grid, w, statistic="pearson")["Electrostatic"]
    assert trend < shape


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_the_per_atom_split_sums_to_the_whole_integral():
    a = _point([[0.0, 0, 0], [1.1, 0.3, 0]], [+1.0, +0.4], "lig")
    b = _point([[4.0, 0, 0], [4.2, 1.6, 0], [7.0, 0, 0]], [-1.0, -0.5, -0.3], "pocket")
    spec = ONE_FIELD.specs[0]
    per_atom = field_overlap_by_atom(a, spec, 0, b, spec, 0)
    assert per_atom.shape == (b.n_atoms,)
    assert per_atom.sum() == pytest.approx(field_overlap(a, spec, 0, b, spec, 0), rel=1e-12)
    # the nearest pocket atom carries the most of it
    assert abs(per_atom[0]) > abs(per_atom[2])
    # an empty field splits into zeros rather than failing
    assert field_overlap_by_atom(_point([[0, 0, 0]], [0.0]), spec, 0, b, spec, 0).tolist() == \
        [0.0, 0.0, 0.0]


def test_profile_columns_sum_to_the_global_complementarity_terms():
    """The decomposition rule: residue by residue, the profile adds up to the global score."""
    lig = _point([[0.0, 0, 0], [1.2, 0, 0]], [+1.0, +0.4], "lig")
    pocket = _point([[4.0, 0, 0], [4.0, 1.5, 0], [7.5, 0, 0], [7.5, 1.5, 0]],
                    [-1.0, -0.5, -0.3, -0.2], "pocket")
    labels = ["ASP1.A", "ASP1.A", "LYS2.A", "LYS2.A"]
    prof = interaction_profile(lig, pocket, ONE_FIELD, labels)
    glob = complementarity(lig, pocket, ONE_FIELD)
    assert prof.residues == ["ASP1.A", "LYS2.A"]
    assert prof.labels == ["electrostatic"]
    assert prof.terms.shape == (2, 1)
    for lab, total in prof.total().items():
        assert total == pytest.approx(glob.terms[lab], rel=1e-10)
    # the near residue carries more of the score than the far one, and both are complementary
    per = prof.per_residue()
    assert per["ASP1.A"] > per["LYS2.A"] > 0
    assert prof.contacts[0] < prof.contacts[1]
    assert prof.top(1) == [("ASP1.A", "electrostatic", pytest.approx(per["ASP1.A"]))]
    assert "ASP1.A" in prof.describe()


def test_a_ligand_out_of_reach_has_an_empty_profile():
    pocket = _point([[0.0, 0, 0], [3.0, 0, 0]], [-1.0, -1.0], "pocket")
    far = _point([[60.0, 0, 0]], [+1.0], "far")
    prof = interaction_profile(far, pocket, ONE_FIELD, ["A1", "B2"], ["A1", "B2"])
    assert np.allclose(prof.terms, 0.0, atol=1e-12)
    assert prof.bits().sum() == 0.0
    assert "no favourable interaction" in prof.describe()


def test_profiles_compare_and_a_consensus_marks_the_conserved_contact():
    pocket = _point([[0.0, 0, 0], [12.0, 0, 0]], [-1.0, -1.0], "pocket")
    order = ["A1", "B2"]
    at_a = interaction_profile(_point([[3.0, 0, 0]], [+1.0], "a"), pocket, ONE_FIELD, order, order)
    also_a = interaction_profile(_point([[3.0, 1.0, 0]], [+1.0], "a2"), pocket, ONE_FIELD,
                                 order, order)
    at_b = interaction_profile(_point([[9.0, 0, 0]], [+1.0], "b"), pocket, ONE_FIELD, order, order)
    assert profile_similarity(at_a, at_a) == pytest.approx(1.0)
    assert profile_similarity(at_a, at_a, binary=True) == pytest.approx(1.0)
    # two ligands at opposite ends of the pocket touch different residues
    assert profile_similarity(at_a, also_a) > profile_similarity(at_a, at_b)
    assert profile_similarity(at_a, at_b, binary=True) == pytest.approx(0.0)
    # the contact two of the three complexes share is the one the consensus keeps
    freq, strength = consensus_profile([at_a, also_a, at_b])
    assert freq.tolist() == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert strength[0] > strength[1]
    with pytest.raises(ValueError):
        profile_similarity(at_a, interaction_profile(_point([[3.0, 0, 0]], [+1.0], "c"), pocket,
                                                     ONE_FIELD, ["X9", "Y8"], ["X9", "Y8"]))


def _point2(coords, charges, steric, name="probe", radius=1.7):
    """A probe carrying both fields of TWO_FIELDS, so it can be scored against a real pocket."""
    n = len(coords)
    w = np.zeros((n, 2))
    w[:, 0] = charges
    w[:, 1] = steric
    return MoleculeFields(name=name, coords=np.array(coords, float), weights=w,
                          radii=np.full(n, radius), symbols=["C"] * n, atom_types=["C.3"] * n,
                          bonds=np.zeros((0, 2), int), charges=np.array(charges, float))


TWO_FIELDS = FieldSet(specs=[FieldSpec(name="electrostatic", kind="gauss", alpha=0.5),
                             FieldSpec(name="steric", kind="steric", alpha=0.5)])


def test_pocket_residues_places_every_field_atom_in_a_residue(tmp_path):
    """The hydrogens pocket_fields adds get the residue of the atom they were added to."""
    path = tmp_path / "site.pdb"
    path.write_text(PDB)
    prot = read_pdb(str(path))
    fs = TWO_FIELDS
    pf = pocket_fields(prot, fs)
    labels, order = pocket_residues(prot, pf)
    assert len(labels) == pf.n_atoms
    assert pf.n_atoms > prot.n_atoms                      # hydrogens were added
    assert len(order) == prot.n_residues == 3
    assert [o.split(".")[0][:3] for o in order] == ["ASP", "LYS", "ARG"]
    assert all(lab in order for lab in labels)            # nothing fell outside a residue
    assert all(labels.count(o) > 0 for o in order)
    # and the profile of a real pocket decomposes the real score, burial included
    lig = _point2([[4.5, 1.0, 1.0]], [+1.0], [1.0], "cation")
    prof = interaction_profile(lig, pf, fs, labels, order)
    assert prof.labels == ["electrostatic", "burial"]
    glob = complementarity(lig, pf, fs)
    assert prof.residues == order
    for lab, total in prof.total().items():
        want = glob.burial if lab == "burial" else glob.terms[lab]
        assert total == pytest.approx(want, rel=1e-9)
    assert profile_matrix([prof, prof]).shape == (2, len(order) * len(prof.labels))
