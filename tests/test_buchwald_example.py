"""The Buchwald-Hartwig yield set: labels, frames, the built files and the benchmark's algebra."""
import csv
import os
import sys

import numpy as np
import pytest
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import make_buchwald_examples as M  # noqa: E402
import run_buchwald as B  # noqa: E402

DATA = os.path.join(ROOT, "examples", "buchwald")
BUILT = all(os.path.exists(os.path.join(DATA, f)) for f in
            ("reactions.csv", "components.csv", "splits.csv", "aryl_halides.sdf", "additives.sdf"))


def test_ligand_and_base_labels():
    assert M.ligand_label("CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2") == "XPhos"
    assert M.ligand_label("CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2") == "t-BuXPhos"
    assert M.ligand_label("CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC") \
        == "t-BuBrettPhos"
    # the adamantyl rings are cyclohexane rings too: AdBrettPhos must not be read as XPhos
    ad = ("CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)"
          "[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC")
    assert M.ligand_label(ad) == "AdBrettPhos"
    assert M.base_label("CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC") == "P2Et"
    assert M.base_label("CN1CCCN2C1=NCCC2") == "MTBD"
    assert M.base_label("CC(C)(C)/N=C(N(C)C)/N(C)C") == "BTMG"


def test_aryl_halide_labels():
    assert M.aryl_label("ClC1=NC=CC=C1") == "2-pyridyl-Cl"
    assert M.aryl_label("IC1=CN=CC=C1") == "3-pyridyl-I"
    assert M.aryl_label("BrC1=CC=C(C(F)(F)F)C=C1") == "4-CF3-phenyl-Br"
    assert M.aryl_label("IC1=CC=C(OC)C=C1") == "4-MeO-phenyl-I"
    assert M.aryl_label("ClC1=CC=C(CC)C=C1") == "4-Et-phenyl-Cl"


@pytest.mark.parametrize("smiles", ["CC1=CC(C)=NO1", "C12=CON=C1C=CC=C2", "C12=C(C=CC=C2)ON=C1",
                                    "CCOC(C1=CON=C1)=O"])
def test_isoxazole_bond_is_found_in_every_ring_type(smiles):
    mol = Chem.MolFromSmiles(smiles)
    o, n, ring = M.isoxazole_bond(mol)
    assert mol.GetAtomWithIdx(o).GetSymbol() == "O" and mol.GetAtomWithIdx(n).GetSymbol() == "N"
    assert mol.GetBondBetweenAtoms(o, n) is not None
    assert len(ring) == 5 and o in ring and n in ring


def test_additive_frame():
    mol = M.embed_lowest("CC1=CC(C(OCC)=O)=NO1", n=4)
    images = M.place_additive(mol, "test")
    assert [m.GetProp("image") for m in images] == ["identity", "mirror z"]
    o, n, ring = M.isoxazole_bond(mol)
    pos = images[0].GetConformer().GetPositions()
    assert pos[n, 0] > 0.5 and pos[o, 0] < -0.5                       # N on +x, O on -x
    assert np.allclose(pos[[o, n]][:, 1:], 0.0, atol=1e-8)            # the bond on the x axis
    assert pos[ring].mean(axis=0)[1] > 0.5                            # the ring centre on +y
    assert np.abs(pos[ring, 2]).max() < 0.05                          # the ring in the xy plane
    mirror = images[1].GetConformer().GetPositions()
    assert np.allclose(mirror[:, 2], -pos[:, 2]) and np.allclose(mirror[:, :2], pos[:, :2])


def test_aryl_halide_frame():
    mol = M.embed_lowest("BrC1=NC=CC=C1", n=4)
    images = M.place_aryl_halide(mol, "test")
    assert len(images) == 4
    ipso, x = M.reacting_bond(mol)
    ring = [r for r in mol.GetRingInfo().AtomRings() if ipso in r][0]
    for m in images:
        pos = m.GetConformer().GetPositions()
        assert np.allclose(pos[ipso], 0.0, atol=1e-8)
        assert pos[x, 0] > 1.5 and np.allclose(pos[x, 1:], 0.0, atol=1e-8)
        assert np.abs(pos[list(ring), 2]).max() < 0.05
    # the four images: identity, half turn about x, and the two mirrors
    p = [m.GetConformer().GetPositions() for m in images]
    assert np.allclose(p[1][:, 1:], -p[0][:, 1:])
    assert np.allclose(p[2][:, 2], -p[0][:, 2])
    assert np.allclose(p[3][:, 1], -p[0][:, 1])


def test_ridge_path_matches_a_direct_solve():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 5))
    K = X @ X.T + 1.0
    y = rng.normal(size=30)
    tr, te = np.arange(20), np.arange(20, 30)
    lams = [0.1, 1.0]
    p = B.ridge_path(K[np.ix_(tr, tr)], y[tr], K[np.ix_(te, tr)], lams)
    for k, lam in enumerate(lams):
        c = np.linalg.solve(K[np.ix_(tr, tr)] + lam * np.eye(20), y[tr] - y[tr].mean())
        assert np.allclose(p[k], y[tr].mean() + K[np.ix_(te, tr)] @ c)


@pytest.mark.skipif(not BUILT, reason="examples/buchwald has not been built")
def test_built_files_are_complete():
    with open(os.path.join(DATA, "reactions.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3955
    counts = {r: len({row[r] for row in rows}) for r in B.ROLES}
    assert counts == {"ligand": 4, "base": 3, "aryl_halide": 15, "additive": 22}
    images = {"aryl_halides.sdf": 4 * 15, "additives.sdf": 2 * 22}
    for f, n in images.items():
        assert len([m for m in Chem.SDMolSupplier(os.path.join(DATA, f), removeHs=False)]) == n


@pytest.mark.skipif(not BUILT, reason="examples/buchwald has not been built")
def test_published_splits_and_the_anova_kernel():
    data = B.Reactions()
    sizes = {k: int(m.sum()) for k, m in data.published.items()}
    assert all(sizes[f"random_{i:02d}"] == 1188 for i in range(1, 11))
    assert [sizes[f"additive_test{i}"] for i in range(1, 5)] == [898, 900, 897, 900]
    for i in range(1, 5):
        m = data.published[f"additive_test{i}"]
        held = set(data.index["additive"][m])
        assert len(held) == 5 and not held & set(data.index["additive"][~m])
    # the ANOVA kernel of identities: 2**(number of shared components)
    comps = {r: np.eye(len(data.labels[r])) for r in B.ROLES}
    sub = np.arange(40)
    K = B.reaction_kernel(data, comps, "anova")[np.ix_(sub, sub)]
    shared = sum((data.index[r][sub][:, None] == data.index[r][sub][None, :]).astype(int) for r in B.ROLES)
    assert np.allclose(K, 2.0 ** shared)
    # grouped inner folds never split an additive
    train = np.where(~data.published["additive_test1"])[0]
    folds = B.inner_folds(data, train, "additive", 0)
    groups = [set(data.index["additive"][train[f]]) for f in folds]
    assert sum(len(f) for f in folds) == len(train)
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])


@pytest.mark.skipif(not BUILT, reason="examples/buchwald has not been built")
def test_arene_designs_hold_out_all_three_halides_together():
    data = B.Reactions()
    assert data.arenes == ["2-pyridyl", "3-pyridyl", "4-CF3-phenyl", "4-Et-phenyl", "4-MeO-phenyl"]
    folds = B.arene_designs(data)["one arene out"]
    assert len(folds) == 5
    seen = np.zeros(data.n, int)
    for name, mask in folds:
        arenes = {data.labels["aryl_halide"][i].rsplit("-", 1)[0] for i in data.index["aryl_halide"][mask]}
        assert arenes == {name}                      # one arene, with all its halides, in each test part
        halogens = {data.labels["aryl_halide"][i].rsplit("-", 1)[1] for i in data.index["aryl_halide"][mask]}
        assert halogens == {"Cl", "Br", "I"}
        seen += mask
    assert (seen == 1).all()                         # every reaction is held out exactly once
    # the inner folds of the new design never split an arene
    train = np.where(~folds[0][1])[0]
    parts = B.inner_folds(data, train, "arene", 0)
    groups = [set(data.index["arene"][train[f]]) for f in parts]
    assert sum(len(f) for f in parts) == len(train)
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])


@pytest.mark.skipif(not BUILT, reason="examples/buchwald has not been built")
def test_parts_kernels_and_what_a_new_arene_can_use():
    data = B.Reactions()
    lab = data.labels["aryl_halide"]
    arene = np.array([l.rsplit("-", 1)[0] for l in lab])
    halogen = np.array([l.rsplit("-", 1)[1] for l in lab])
    Ks = B.parts_kernel(data, "parts_sum")
    Kp = B.parts_kernel(data, "parts_product")
    # the added form is a Free-Wilson model in three dimensions: 5 arenes + 3 halogens - 1
    assert np.linalg.matrix_rank(Ks) == 7
    # with the product it is the identity kernel of the aryl halide plus the two parts
    assert np.linalg.matrix_rank(Kp) == 15
    same_arene = (arene[:, None] == arene[None, :]).astype(float)
    same_halogen = (halogen[:, None] == halogen[None, :]).astype(float)
    assert np.allclose(Ks, same_arene + same_halogen)
    assert np.allclose(Kp, same_arene + same_halogen + same_arene * same_halogen)
    # a member of a new arene shares nothing with the training part but its halogen, in both forms
    new = arene == "2-pyridyl"
    for K in (Ks, Kp):
        cross = K[np.ix_(new, ~new)]
        assert np.allclose(cross, same_halogen[np.ix_(new, ~new)])
    # the stage uses the additive by name: the kernels of the parts method are identities elsewhere
    comps = B.component_kernels(data, "parts_product", {})
    assert np.allclose(comps["additive"], np.eye(22))
    assert np.allclose(comps["aryl_halide"], Kp)
    assert np.allclose(comps["ligand"], np.eye(4)) and np.allclose(comps["base"], np.eye(3))

