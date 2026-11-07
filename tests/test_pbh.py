"""Tests of cmfqsar/pbh.py: the COMPAS reader, the planar frame, ring classes, Clar sextets,
ring-centred fields and the per-ring decomposition (run with: python -m pytest tests)."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar import pbh  # noqa: E402
from cmfqsar.align import set_coords  # noqa: E402
from cmfqsar.atomtypes import assign_atom_types  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, atom_type_counts  # noqa: E402
from cmfqsar.kernels import kernel_matrices, numerical_overlap  # noqa: E402
from cmfqsar.model import CMFModel, krr_fit, mae  # noqa: E402

SMILES = {
    "benzene": "c1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "phenanthrene": "c1ccc2c(c1)ccc1ccccc12",
    "triphenylene": "c1ccc2c(c1)c1ccccc1c1ccccc21",
    "chrysene": "c1ccc2c(c1)ccc1c3ccccc3ccc21",
    "tetracene": "c1ccc2cc3cc4ccccc4cc3cc2c1",
    "benz[a]anthracene": "c1ccc2cc3c(ccc4ccccc43)cc2c1",
}


def _mol(name, seed=1, framed=True):
    m = Chem.AddHs(Chem.MolFromSmiles(SMILES[name]))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    if framed:
        set_coords(m, pbh.canonical_frame(m))
    m.SetProp("_Name", name)
    return m


def _same_point_set(a, b, symbols, tol=1e-6):
    """Every atom of `a` has an atom of the same element of `b` at the same place."""
    for s in set(symbols):
        pa = a[[i for i, x in enumerate(symbols) if x == s]]
        pb = b[[i for i, x in enumerate(symbols) if x == s]]
        d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
        if d.min(axis=1).max() > tol:
            return False
    return True


# --------------------------------------------------------------------------- the reader

def test_reader_matches_smiles_to_xyz_in_another_atom_order():
    m = _mol("phenanthrene", framed=False)
    pos = m.GetConformer().GetPositions()
    symbols = [a.GetSymbol() for a in m.GetAtoms()]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(symbols))
    built = pbh.mol_from_smiles_xyz(SMILES["phenanthrene"], [symbols[i] for i in perm], pos[perm],
                                    name="phen")
    assert built.GetNumAtoms() == m.GetNumAtoms()
    assert built.GetProp("_Name") == "phen"
    # the geometry is the same point set, and the bonds of the SMILES join bonded atoms
    out = built.GetConformer().GetPositions()
    assert _same_point_set(pos, out, symbols)
    for b in built.GetBonds():
        d = np.linalg.norm(out[b.GetBeginAtomIdx()] - out[b.GetEndAtomIdx()])
        assert d < 1.6
    assert Chem.MolToSmiles(Chem.RemoveHs(built)) == Chem.MolToSmiles(Chem.MolFromSmiles(SMILES["phenanthrene"]))


def test_reader_rejects_wrong_elements_and_wrong_graph():
    m = _mol("naphthalene", framed=False)
    pos = m.GetConformer().GetPositions()
    symbols = [a.GetSymbol() for a in m.GetAtoms()]
    with pytest.raises(ValueError, match="atoms"):
        pbh.mol_from_smiles_xyz(SMILES["anthracene"], symbols, pos, name="x")
    # the same atoms with the geometry of a different isomer: azulene from naphthalene atoms
    with pytest.raises(ValueError):
        pbh.mol_from_smiles_xyz("c1ccc2cccc-2cc1", symbols, pos, name="azulene?")


def test_parse_xyz_and_table_reader(tmp_path):
    text = "3\ncomment\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n"
    syms, xyz = pbh.parse_xyz(text)
    assert syms == ["C", "H", "H"] and xyz.shape == (3, 3) and xyz[1, 0] == 1.0
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("molecule,smiles,n_rings,HOMO_eV,LUMO_eV,GAP_eV,lalas\n"
                        "m1,c1ccc2ccccc2c1,2,-5.9,-1.2,4.7,\n")
    folder = tmp_path / "xyz"
    folder.mkdir()
    m = _mol("naphthalene", framed=False)
    pos = m.GetConformer().GetPositions()
    lines = [str(m.GetNumAtoms()), "m1"] + [f"{a.GetSymbol()} {x} {y} {z}"
                                           for a, (x, y, z) in zip(m.GetAtoms(), pos)]
    (folder / "m1.xyz").write_text("\n".join(lines) + "\n")
    ds = pbh.read_compas(str(csv_path), str(folder))
    assert len(ds) == 1 and ds.names == ["m1"] and ds.activity[0] == 4.7
    assert ds.mols[0].GetProp("GAP_eV") == "4.7" and ds.mols[0].GetProp("n_rings") == "2"


@pytest.mark.skipif(not os.path.isfile(os.path.join(os.path.dirname(__file__), "..", "benchmarks",
                                                    "cache", "compas", "compas-1D_b3lyp_def2-svp.csv")),
                    reason="COMPAS-1D is not downloaded")
def test_compas_rows_agree_with_their_annulation_strings():
    root = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "cache", "compas")
    geoms = os.path.join(root, "xyz1D")
    if not os.path.isdir(geoms):
        geoms = os.path.join(root, "compas-1D_b3lyp_def2-svp.tar.gz")
    ds = pbh.read_compas(os.path.join(root, "compas-1D_b3lyp_def2-svp.csv"), geoms, limit=60)
    assert len(ds) == 60
    for m in ds.mols:
        rings = pbh.ring_classes(m)
        assert len(rings) == int(m.GetProp("n_rings"))
        lin, ang = pbh.annulation_counts(m.GetProp("lalas"))
        rc = pbh.ring_class_counts(rings)
        assert rc["linear"] == lin
        assert rc["angular"] + rc["branched"] == ang


# --------------------------------------------------------------------------- the frame

@pytest.mark.parametrize("name", ["phenanthrene", "chrysene", "benz[a]anthracene", "triphenylene"])
def test_frame_is_idempotent_and_rotation_invariant(name):
    from scipy.spatial.transform import Rotation
    m = _mol(name)
    c = m.GetConformer().GetPositions()
    symbols = [a.GetSymbol() for a in m.GetAtoms()]
    again = pbh.canonical_frame(m, coords=c)
    # already aligned: the identity.  The bound is 1e-10 and not 1e-8 because the angle of the
    # frame of a molecule with a three-fold axis is now solved for, not searched for: before
    # that, triphenylene came back 3.9e-8 A away on one runner and 8.5e-14 away here, since a
    # search that only compares values of a flat maximum cannot place the angle better than
    # 1e-8 radians.  Measured after the fix: 1.8e-15 for triphenylene, 7.5e-16 for the others.
    assert np.abs(again - c).max() < 1e-10
    rot = Rotation.random(random_state=7).as_matrix()
    moved = pbh.canonical_frame(m, coords=c @ rot.T + np.array([3.0, -2.0, 11.0]))
    assert _same_point_set(c, moved, symbols, tol=1e-6)
    # the plane is z = 0 (the molecule is planar), the long axis is x (triphenylene has
    # equal in-plane moments, so it has no long axis and x follows the third-moment rule)
    assert np.abs(c[:, 2]).max() < 0.2
    if name != "triphenylene":
        assert np.ptp(c[:, 0]) >= np.ptp(c[:, 1]) - 1e-9
    assert np.abs(c[[a.GetAtomicNum() > 1 for a in m.GetAtoms()]].mean(axis=0)).max() < 1e-6


def test_frame_of_symmetric_molecules_is_the_same_point_set():
    for name in ("naphthalene", "anthracene", "benzene"):
        m = _mol(name)
        c = m.GetConformer().GetPositions()
        symbols = [a.GetSymbol() for a in m.GetAtoms()]
        for seed in (1, 2):
            from scipy.spatial.transform import Rotation
            rot = Rotation.random(random_state=seed).as_matrix()
            moved = pbh.canonical_frame(m, coords=c @ rot.T)
            # an embedded geometry is symmetric only to about 1e-5 A, and the frame of a
            # nearly symmetric molecule is decided at that level; the point sets agree to it
            assert _same_point_set(c, moved, symbols, tol=1e-4), name


def test_mirror_image_lands_in_the_mirrored_frame():
    m = _mol("benz[a]anthracene")
    c = m.GetConformer().GetPositions()
    symbols = [a.GetSymbol() for a in m.GetAtoms()]
    mirrored = pbh.canonical_frame(m, coords=c * np.array([1.0, 1.0, -1.0]))
    assert _same_point_set(c[:, :2], mirrored[:, :2], symbols, tol=1e-6)
    # a mirror through a plane that is not the molecular plane: the same in-plane picture
    mirrored = pbh.canonical_frame(m, coords=c * np.array([-1.0, 1.0, 1.0]))
    assert _same_point_set(c[:, :2], mirrored[:, :2], symbols, tol=1e-6)


def test_sign_rule_separates_a_molecule_from_its_in_plane_flips():
    # benz[a]anthracene has no in-plane symmetry: its four flips are four different point sets,
    # and the rule picks one of them whatever the starting orientation
    m = _mol("benz[a]anthracene")
    c = m.GetConformer().GetPositions()
    symbols = [a.GetSymbol() for a in m.GetAtoms()]
    cands = pbh.frame_candidates(c)
    assert sum(_same_point_set(c, k, symbols) for k in cands) == 1
    for k in cands:
        back = pbh.canonical_frame(m, coords=k)
        assert _same_point_set(c, back, symbols, tol=1e-6)


def test_kernel_frame_choice_agrees_with_the_rule_on_an_aligned_set():
    fs = FieldSet.create(physchem=("steric",), cif_types=pbh.PBH_TYPES, cif_scheme="pbh",
                         ring_classes=pbh.RING_CLASSES, alpha=0.5)
    ref = AtomTable.from_molecules([pbh.build_pbh_fields(_mol(n), fs)
                                    for n in ("phenanthrene", "chrysene", "triphenylene")])
    m = _mol("benz[a]anthracene")
    c = m.GetConformer().GetPositions()
    chosen, scores = pbh.choose_frame_by_kernel(m, fs, ref)
    assert len(scores) == 4 and np.isfinite(scores).all()
    assert np.abs(chosen - pbh.frame_candidates(c)[int(np.argmax(scores))]).max() < 1e-12


# --------------------------------------------------------------------------- atoms and rings

def test_pbh_atom_types_and_scheme():
    m = _mol("phenanthrene")
    types = assign_atom_types(m, "pbh")
    assert types.count("C.H") == 10 and types.count("C.2r") == 4 and types.count("H") == 10
    counts = atom_type_counts([m, _mol("triphenylene")], scheme="pbh")
    assert counts["C.2r"] == 2 and "C.3r" not in counts


def test_ring_classes_and_clar_weights():
    expect = {
        "benzene": (["terminal"], [1.0]),
        "naphthalene": (["terminal", "terminal"], [0.5, 0.5]),
        "anthracene": (["terminal", "linear", "terminal"], [1 / 3, 1 / 3, 1 / 3]),
        "phenanthrene": (["terminal", "angular", "terminal"], [1.0, 0.0, 1.0]),
        "triphenylene": (["terminal", "terminal", "terminal", "branched"], [1, 1, 1, 0]),
    }
    for name, (classes, sextets) in expect.items():
        m = _mol(name)
        rings = pbh.ring_classes(m)
        assert sorted(r.cls for r in rings) == sorted(classes), name
        w = pbh.clar_weights(m, rings)
        assert sorted(np.round(w, 6)) == sorted(np.round(sextets, 6)), name
        assert all(r.sextet == v for r, v in zip(rings, w))
        # the centroid of a ring is the mean of its six carbons, in the plane
        for r in rings:
            assert np.allclose(r.centroid, m.GetConformer().GetPositions()[list(r.atoms)].mean(0))
    m = _mol("chrysene")
    rings = pbh.ring_classes(m)
    assert sorted(r.cls for r in rings) == ["angular", "angular", "terminal", "terminal"]
    assert sorted(np.round(pbh.clar_weights(m, rings), 4)) == [0.3333, 0.3333, 0.6667, 0.6667]
    assert pbh.annulation_counts("ALA") == (1, 2) and pbh.annulation_counts("A()") == (0, 1)


def test_count_features_and_splits():
    f = pbh.count_features(_mol("anthracene"))
    assert list(f) == [10, 4, 0, 2, 1, 0, 0, 3, 1]
    s = pbh.random_split(100, seed=3)
    allidx = np.concatenate([s["train"], s["valid"], s["test"]])
    assert len(s["train"]) == 80 and len(s["valid"]) == 10 and len(s["test"]) == 10
    assert sorted(allidx) == list(range(100))
    assert np.array_equal(pbh.random_split(100, seed=3)["test"], s["test"])
    z = pbh.size_split([3, 4, 5, 5, 6], test_rings=5)
    assert list(z["train"]) == [0, 1] and list(z["test"]) == [2, 3]


# --------------------------------------------------------------------------- the fields

def _fieldset(alpha=0.5):
    return FieldSet.create(physchem=("steric", "electrostatic"), cif_types=pbh.PBH_TYPES,
                           cif_scheme="pbh", ring_classes=pbh.RING_CLASSES + ("sextet",),
                           alpha=alpha)


def test_ring_fields_are_pseudo_atoms_with_the_right_weights():
    fs = _fieldset()
    m = _mol("anthracene")
    mf = pbh.build_pbh_fields(m, fs)
    assert mf.n_atoms == m.GetNumAtoms() + 3
    assert pbh.n_real_atoms(mf) == m.GetNumAtoms()
    k = fs.names.index("ring:linear")
    assert mf.weights[:m.GetNumAtoms(), k].sum() == 0            # atoms carry no ring field
    assert mf.weights[m.GetNumAtoms():, k].sum() == 1            # one linear ring
    assert mf.weights[m.GetNumAtoms():, fs.names.index("ring:terminal")].sum() == 2
    assert abs(mf.weights[m.GetNumAtoms():, fs.names.index("ring:sextet")].sum() - 1.0) < 1e-12
    assert mf.weights[m.GetNumAtoms():, fs.names.index("steric")].sum() == 0   # and no steric
    assert fs.specs[k].label == "Ring linear" and fs.specs[k].is_ring


def test_ring_kernel_matches_numerical_integration():
    fs = _fieldset(alpha=0.5)
    a = pbh.build_pbh_fields(_mol("naphthalene"), fs)
    b = pbh.build_pbh_fields(_mol("phenanthrene"), fs)
    K = kernel_matrices(AtomTable.from_molecules([a]), AtomTable.from_molecules([b]), fs.specs)
    for fname in ("ring:terminal", "cif:C.H"):
        f = fs.names.index(fname)
        ia = np.flatnonzero(a.weights[:, f])
        ib = np.flatnonzero(b.weights[:, f])
        num = sum(numerical_overlap(a.coords[i], a.weights[i, f], 0.5, b.coords[j], b.weights[j, f],
                                    0.5, spacing=0.1, extent=5.0) for i in ia for j in ib)
        assert abs(num - K[f, 0, 0]) < 1e-3 * abs(K[f, 0, 0]), fname


def test_ring_contributions_sum_to_the_prediction():
    fs = _fieldset()
    names = ["naphthalene", "anthracene", "phenanthrene", "triphenylene", "chrysene", "tetracene",
             "benz[a]anthracene"]
    mols = [_mol(n) for n in names]
    mfs = [pbh.build_pbh_fields(m, fs) for m in mols]
    y = np.array([4.78, 3.56, 4.70, 4.87, 4.06, 2.75, 3.74])
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.05})
    model.fit(AtomTable.from_molecules(mfs), y, cv_folds=3)
    for m, mf in zip(mols, mfs):
        rings = pbh.ring_classes(m)
        pbh.clar_weights(m, rings)
        per_ring, contrib, lost = pbh.ring_contributions(model, mf, rings)
        yp = model.predict(AtomTable.from_molecules([mf]))[0]
        assert lost == 0.0
        assert abs(per_ring.sum() + model.bias - yp) < 1e-8
        assert abs(contrib.sum() + model.bias - yp) < 1e-8
    # the same molecule unaligned gives the same prediction only up to the frame, so the
    # per-ring values are frame-dependent by construction; what must hold is the sum
    assert len(per_ring) == 4


def test_indicator_model_at_small_alpha_is_the_count_model():
    """alpha -> 0: every Gaussian becomes flat, the overlap of two indicator fields becomes the
    product of their counts times a constant, and KRR on that kernel is ridge regression on the
    counts.  With the fields scaled to unit mean diagonal the constant drops out."""
    names = ["naphthalene", "anthracene", "phenanthrene", "triphenylene", "chrysene", "tetracene",
             "benz[a]anthracene", "benzene"]
    mols = [_mol(n) for n in names]
    y = np.array([4.78, 3.56, 4.70, 4.87, 4.06, 2.75, 3.74, 6.72])
    fs = FieldSet.create(physchem=(), cif_types=("C.H", "C.2r"), cif_scheme="pbh",
                         ring_classes=("terminal", "linear", "angular"), alpha=1e-6)
    mfs = [pbh.build_pbh_fields(m, fs) for m in mols]
    table = AtomTable.from_molecules(mfs)
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.1}, scale_fields=True)
    model.fit(table, y, cv_folds=2)
    pred_field = model.predict(table)
    # the count kernel: sum_f c_f n_f(i) n_f(j), c_f = 1 / mean_i n_f(i)^2
    counts = np.array([[mf.weights[:, f].sum() for f in range(fs.n_fields)] for mf in mfs])
    c = 1.0 / np.mean(counts ** 2, axis=0)
    K = (counts * c) @ counts.T
    a, b = krr_fit(K, y, 0.1)
    pred_count = K @ a + b
    # alpha = 1e-6 leaves exp(-alpha d^2 / 4) within 3e-5 of 1 across a molecule of 10 A
    assert np.abs(pred_field - pred_count).max() < 1e-4


def test_mae():
    assert mae([1.0, 2.0, 4.0], [2.0, 2.0, 1.0]) == pytest.approx(4.0 / 3.0)


# --------------------------------------------------------------------------- later COMPAS sets

HETERO_SMILES = {
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "coronene": "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61",
    "indole-fused": "c1ccc2c(c1)cc1[nH]ccc1c2",
    "dibenzothiophene": "c1ccc2c(c1)sc1ccccc12",
    "borole": "C1=CC=CB1",
    "1,4-diborinine": "B1=CC=BC=C1",
    "1,2-dihydro-1,2-diborinine": "B1C=CC=CB1",
    "pyrazine": "c1cnccn1",
    "cyclobutadiene": "C1=CC=C1",
}


def _hetero(name, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(HETERO_SMILES[name]))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    m.SetProp("_Name", name)
    return m


def test_peri_class_and_homa_index():
    # every ring of pyrene and of coronene has two shared bonds that touch: peri
    for name, n in (("pyrene", 4), ("coronene", 7)):
        rings = pbh.ring_classes(_hetero(name))
        assert len(rings) == n and all(r.cls == pbh.PERI_CLASS for r in rings), name
    assert pbh.ring_class_counts(pbh.ring_classes(_hetero("pyrene")))["peri"] == 4
    # the cata-condensed classes are unchanged
    assert sorted(r.cls for r in pbh.ring_classes(_mol("phenanthrene"))) == ["angular", "terminal", "terminal"]
    # HOMA: 1 for a regular hexagon of 1.388 A, negative for alternating 1.33 / 1.47 A
    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    conf = m.GetConformer()
    ring = list(m.GetRingInfo().AtomRings()[0])
    order = pbh._cyclic_order(m, ring)
    radius = 1.388
    for k, a in enumerate(order):
        ang = 2 * np.pi * k / 6
        conf.SetAtomPosition(a, Point3D(radius * np.cos(ang), radius * np.sin(ang), 0.0))
    rings = pbh.ring_classes(m)
    assert abs(rings[0].homa - 1.0) < 1e-9
    pos = conf.GetPositions()
    kek = []
    for k, a in enumerate(order):
        kek.append(1.33 if k % 2 == 0 else 1.47)
    # place the atoms on a hexagon with alternating edge lengths (approximately, by scaling angles)
    r_alt = np.zeros((6, 3))
    theta = 0.0
    r_alt[0] = [0.0, 0.0, 0.0]
    heading = 0.0
    for k in range(1, 6):
        heading += np.pi / 3
        r_alt[k] = r_alt[k - 1] + kek[k - 1] * np.array([np.cos(heading), np.sin(heading), 0.0])
    for k, a in enumerate(order):
        conf.SetAtomPosition(a, Point3D(*r_alt[k]))
    expect = 1 - 257.7 / 6 * sum((1.388 - d) ** 2 for d in [1.33, 1.47, 1.33, 1.47, 1.33, kek[5]])
    rings = pbh.ring_classes(m)
    assert rings[0].homa < 0 and abs(rings[0].homa - expect) < 0.05
    # a ring with a bond to boron has no HOMA parameters: nan, and weight 0 in the field
    b = pbh.ring_classes(_hetero("borole"), carbon_only=False)[0]
    assert np.isnan(b.homa) and pbh.ring_field_weight(b, pbh.HOMA_FIELD) == 0.0


def test_hetero_ring_types_and_atom_types():
    expect = {"indole-fused": {"benzene": 2, "pyrrole": 1}, "dibenzothiophene": {"benzene": 2, "thiophene": 1},
              "borole": {"borole": 1}, "1,4-diborinine": {"14diborinine": 1},
              "1,2-dihydro-1,2-diborinine": {"dhdiborinine": 1}, "pyrazine": {"pyrazine": 1},
              "cyclobutadiene": {"cyclobutadiene": 1}}
    for name, types in expect.items():
        rings = pbh.ring_classes(_hetero(name), carbon_only=False)
        seen = {}
        for r in rings:
            seen[r.ring_type] = seen.get(r.ring_type, 0) + 1
        assert seen == types, name
    # carbon-only perception ignores the hetero rings
    assert len(pbh.ring_classes(_hetero("dibenzothiophene"))) == 2
    # five-membered rings fused on two sides are angular; the thiophene of dibenzothiophene
    assert [r.cls for r in pbh.ring_classes(_hetero("dibenzothiophene"), carbon_only=False)
            if r.ring_type == "thiophene"] == ["angular"]
    types = {pbh.pbh_atom_type(a) for a in _hetero("indole-fused").GetAtoms()}
    assert "N.H" in types and "N" not in types
    types = {pbh.pbh_atom_type(a) for a in _hetero("1,4-diborinine").GetAtoms()}
    assert "B" in types and "B.H" not in types
    assert "B.H" in {pbh.pbh_atom_type(a) for a in _hetero("borole").GetAtoms()}
    # the generalised count features: hetero atom types, the peri class, the ring types
    names = pbh.count_names(pbh.PBH_TYPES + pbh.HETERO_TYPES, pbh.RING_CLASSES_PERI, pbh.RING_TYPES,
                            sextet=False, homa=True)
    f = pbh.count_features(_hetero("indole-fused"), atom_types=pbh.PBH_TYPES + pbh.HETERO_TYPES,
                           ring_classes_=pbh.RING_CLASSES_PERI, ring_types=pbh.RING_TYPES,
                           sextet=False, homa=True, carbon_only=False)
    d = dict(zip(names, f))
    assert d["n_NH"] == 1 and d["n_pyrrole"] == 1 and d["n_benzene"] == 2 and d["n_rings"] == 3
    assert d["n_linear"] == 1 and d["n_terminal"] == 2 and d["n_peri"] == 0
    # the default call is unchanged
    assert list(pbh.count_features(_mol("anthracene"))) == [10, 4, 0, 2, 1, 0, 0, 3, 1]


def test_ring_type_fields_are_pseudo_atoms():
    fs = FieldSet.create(physchem=("steric",), ring_classes=("pyrrole", "benzene", "angular"), alpha=0.5)
    m = _hetero("indole-fused")
    rings = pbh.ring_classes(m, carbon_only=False)
    mf = pbh.build_pbh_fields(m, fs, rings=rings)
    assert mf.n_atoms == m.GetNumAtoms() + 3
    tail = mf.weights[m.GetNumAtoms():]
    assert tail[:, fs.names.index("ring:pyrrole")].sum() == 1
    assert tail[:, fs.names.index("ring:benzene")].sum() == 2
    assert tail[:, fs.names.index("ring:angular")].sum() == 0


def test_read_compas2(tmp_path):
    import csv
    import gzip
    rows, blocks = [], []
    for k, (name, smi) in enumerate((("C2M000001", "c1ccc2[nH]ccc2c1"), ("C2M000002", "c1ccc2occc2c1"))):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=1)
        for charge in (-1, 0, 1):
            rows.append({"name": name, "charge": str(charge), "smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smi)),
                         "gap": "" if charge else f"{4.0 + k:.3f}", "homo": "-6", "lumo": "-2",
                         "rings": "2", "heteroatoms": "1", "formula": "x"})
            mm = Chem.Mol(m)
            mm.SetProp("name", name)
            mm.SetProp("charge", str(charge))
            blocks.append(mm)
    csv_path = tmp_path / "t.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sdf_path = tmp_path / "t.sdf.gz"
    with gzip.open(sdf_path, "wt") as fh:
        wr = Chem.SDWriter(fh)
        for b in blocks:
            wr.write(b)
        wr.close()
    ds = pbh.read_compas2(str(csv_path), str(sdf_path))
    assert ds.names == ["C2M000001", "C2M000002"] and list(ds.activity) == [4.0, 5.0]
    assert ds.mols[0].GetProp("n_rings") == "2" and ds.mols[1].GetProp("GAP_eV") == "5.000"
    assert ds.mols[0].GetNumAtoms() == 16              # hydrogens kept
    ds1 = pbh.read_compas2(str(csv_path), str(sdf_path), names=["C2M000002"])
    assert ds1.names == ["C2M000002"]
    # a SMILES that does not match the SDF graph is refused
    rows[4]["smiles"] = "c1ccc2sccc2c1"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError):
        pbh.read_compas2(str(csv_path), str(sdf_path))
