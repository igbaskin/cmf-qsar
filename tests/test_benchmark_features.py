"""Tests for the features added for the literature benchmark: Tripos increments, tolerant
MOL2 parsing, activity tables, KRR without bias, interleaved folds, Brent lambda search and
the precomputed-kernel nested CV."""
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.atomtypes import vdw_params  # noqa: E402
from cmfqsar.dataio import (Dataset, _mol2_fallback, apply_activity_table, mol_from_mol2_block,  # noqa: E402
                            read_activity_table)
from cmfqsar.fields import AtomTable, FieldSet, _hydrogen_counts, build_fields  # noqa: E402
from cmfqsar.increments import tripos_increments  # noqa: E402
from cmfqsar.model import CMFModel, _kfold_indices, grid_search, krr_fit, nested_cv  # noqa: E402


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _mol2_text(mol, name="mol", types=None, charges=None, bond_types=None):
    """Minimal MOL2 writer for the tests (heavy atom first in every bond record)."""
    conf = mol.GetConformer()
    n, nb = mol.GetNumAtoms(), mol.GetNumBonds()
    lines = ["@<TRIPOS>MOLECULE", name, f"{n:5d}{nb:5d}     1     0     0", "SMALL", "USER_CHARGES", "", "@<TRIPOS>ATOM"]
    for a in mol.GetAtoms():
        i = a.GetIdx()
        pos = conf.GetAtomPosition(i)
        t = types[i] if types else a.GetSymbol()
        q = charges[i] if charges else 0.0
        lines.append(f"{i + 1:7d} {a.GetSymbol()}{i + 1:<4d} {pos.x:10.4f} {pos.y:10.4f} {pos.z:10.4f} {t:<6s} 1 LIG {q:10.4f}")
    lines.append("@<TRIPOS>BOND")
    for b in mol.GetBonds():
        i1, i2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i1).GetAtomicNum() == 1:
            i1, i2 = i2, i1
        bt = bond_types[b.GetIdx()] if bond_types else {1.0: "1", 2.0: "2", 3.0: "3", 1.5: "ar"}[b.GetBondTypeAsDouble()]
        lines.append(f"{b.GetIdx() + 1:6d}{i1 + 1:6d}{i2 + 1:6d} {bt}")
    return "\n".join(lines) + "\n"


def test_tripos_increments_match_original_scripts():
    assert abs(tripos_increments("O.3", 1)["hbd"] - 0.445333) < 1e-9
    assert abs(tripos_increments("O.3", 0)["hbd"] - (-0.035354)) < 1e-9
    assert abs(tripos_increments("C.3", 3)["hydrophobic"] - (0.041270 + 0.369764)) < 1e-9
    assert abs(tripos_increments("S.O2", 0)["hydrophobic"] - (-0.077480 + 0.027576)) < 1e-9  # case-insensitive
    assert all(v == 0.0 for v in tripos_increments("H", 0).values())
    assert abs(tripos_increments("N.ar", 0)["hba"] - 0.282666) < 1e-9
    # vdW parameters by Sybyl type (sulfone sulfur has R = 1.7 in the Tripos table)
    assert vdw_params("S", "S.o2") == (1.70, 0.314)
    assert vdw_params("S") == (1.80, 0.314)
    assert vdw_params("C", "C.ar") == vdw_params("C")


def test_tripos_parameterisation_of_ethanol():
    m = _mol("CCO")
    fs = FieldSet.create(param_scheme="tripos")
    mf = build_fields(m, fs, "ethanol")
    f = {s.name: k for k, s in enumerate(fs.specs)}
    # heavy atoms: CH3 (C.3, 3 H), CH2 (C.3, 2 H), OH (O.3, 1 H); hydrogens carry nothing
    assert abs(mf.weights[0, f["hydrophobic"]] - 0.411034) < 1e-6
    assert abs(mf.weights[1, f["hydrophobic"]] - 0.249558) < 1e-6
    assert abs(mf.weights[2, f["hydrophobic"]] - (-0.224286 - 0.102250 - 0.279404)) < 1e-6
    assert abs(mf.weights[2, f["hbd"]] - 0.445333) < 1e-6
    assert abs(mf.weights[2, f["hba"]] - (0.147962 + 0.014953)) < 1e-6
    assert np.all(mf.weights[3:, f["hydrophobic"]] == 0) and np.all(mf.weights[3:, f["hbd"]] == 0)
    # legacy hydrogen counting: RDKit writes heavy atom first -> no hydrogens counted
    nh = _hydrogen_counts(m, legacy=True)
    assert nh[:3].tolist() == [0, 0, 0]
    assert _hydrogen_counts(m)[:3].tolist() == [3, 2, 1]
    mf_leg = build_fields(m, FieldSet.create(param_scheme="tripos-legacy"), "ethanol")
    assert abs(mf_leg.weights[2, f["hydrophobic"]] - (-0.224286 - 0.102250 - 0.010867)) < 1e-6
    # the default (RDKit) scheme is unchanged and puts the donor weight on the hydrogen
    mf_rd = build_fields(m, FieldSet.create(), "ethanol")
    assert mf_rd.weights[2, f["hbd"]] == 0 and mf_rd.weights[:, f["hbd"]].max() > 0


def test_mol2_fallback_parser():
    m = _mol("CCO")
    types = ["C.3", "C.3", "O.3"] + ["H"] * 6
    charges = [-0.065, 0.039, -0.397, 0.03, 0.03, 0.03, 0.06, 0.06, 0.21]
    text = _mol2_text(m, "ethanol", types, charges)
    fb = _mol2_fallback(text)
    assert fb is not None and fb.GetNumAtoms() == 9 and fb.GetNumBonds() == 8
    assert fb.GetProp("_Name") == "ethanol"
    assert [a.GetProp("_TriposAtomType") for a in fb.GetAtoms()] == types
    assert abs(fb.GetAtomWithIdx(2).GetDoubleProp("_TriposPartialCharge") + 0.397) < 1e-9
    assert np.allclose(fb.GetConformer().GetPositions(), m.GetConformer().GetPositions(), atol=1e-3)
    # the fallback keeps the bond order of the file (needed for the legacy hydrogen count)
    assert fb.GetBondWithIdx(0).GetBeginAtom().GetAtomicNum() > 1
    # the robust reader accepts a block that RDKit's sanitising reader rejects (pentavalent carbon)
    bad = "\n".join([
        "@<TRIPOS>MOLECULE", "bad", "    6     5     1     0     0", "SMALL", "NO_CHARGES", "", "@<TRIPOS>ATOM",
        "      1 C1    0.0000   0.0000   0.0000 C.3   1 LIG  0.0000",
        "      2 H1    1.0900   0.0000   0.0000 H     1 LIG  0.0000",
        "      3 H2   -1.0900   0.0000   0.0000 H     1 LIG  0.0000",
        "      4 H3    0.0000   1.0900   0.0000 H     1 LIG  0.0000",
        "      5 H4    0.0000  -1.0900   0.0000 H     1 LIG  0.0000",
        "      6 H5    0.0000   0.0000   1.0900 H     1 LIG  0.0000",
        "@<TRIPOS>BOND",
        "     1 1 2 1", "     2 1 3 1", "     3 1 4 1", "     4 1 5 1", "     5 1 6 1", ""])
    assert Chem.MolFromMol2Block(bad, removeHs=False, sanitize=True) is None
    mm = mol_from_mol2_block(bad)
    assert mm is not None and mm.GetNumAtoms() == 6 and mm.GetNumBonds() == 5
    assert mm.GetNumConformers() == 1 and mm.GetAtomWithIdx(0).GetProp("_TriposAtomType") == "C.3"
    # the RDKit route is used when it works and also sets the Tripos properties
    ok = mol_from_mol2_block(text)
    assert ok is not None and ok.GetAtomWithIdx(0).HasProp("_TriposAtomType")


def test_activity_table(tmp_path):
    p = tmp_path / "act.txt"
    p.write_text('NAME\tACTIVITY\tSET\tMin_dist\n"MOL_1"\t7.92\t1\t999.9\n"mol_2"\t5.60\t2\t0.06\n"MOL_3"\tNA\t1\t1\nfooter line\n')
    rows = read_activity_table(str(p))
    assert rows == [("MOL_1", 7.92, "train"), ("mol_2", 5.6, "test")]
    ds = Dataset(mols=[None, None, None], names=["mol_1", "MOL_2", "mol_3"], activity=np.full(3, np.nan),
                 subset=np.array(["train"] * 3, dtype=object))
    n = apply_activity_table(ds, rows)
    assert n == 2
    assert np.isnan(ds.activity[2]) and ds.activity[0] == 7.92 and ds.activity[1] == 5.6
    assert list(ds.subset) == ["train", "test", "train"]
    # header-less comma file, positional fallback when names do not match
    p2 = tmp_path / "act.csv"
    p2.write_text("a,1.0,1\nb,2.0,2\nc,3.0,1\n")
    rows2 = read_activity_table(str(p2))
    assert [r[1] for r in rows2] == [1.0, 2.0, 3.0] and rows2[1][2] == "test"
    ds2 = Dataset(mols=[None] * 3, names=["x", "y", "z"], activity=np.full(3, np.nan),
                  subset=np.array(["train"] * 3, dtype=object))
    apply_activity_table(ds2, rows2)
    assert ds2.activity.tolist() == [1.0, 2.0, 3.0] and list(ds2.subset) == ["train", "test", "train"]


def test_krr_without_bias_and_interleaved_folds():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 3)
    K = np.exp(-0.5 * ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    y = rng.randn(12)
    a, b = krr_fit(K, y, 0.1, fit_bias=False)
    assert b == 0.0 and np.allclose((K + 0.1 * np.eye(12)) @ a, y)
    a2, b2 = krr_fit(K, y, 0.1, fit_bias=True)
    assert abs(a2.sum()) < 1e-10 and b2 != 0.0
    folds = _kfold_indices(10, 3, None)
    assert [f.tolist() for f in folds] == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]


def _small_set():
    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O", "c1ccccc1OC",
              "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C", "c1ccc(Br)cc1O", "c1ccc(N)cc1C"]
    mols = [_mol(s, i) for i, s in enumerate(smiles)]
    y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.4])
    return mols, y


def test_brent_lambda_search_and_precomputed_nested_cv():
    mols, y = _small_set()
    fs = FieldSet.create(param_scheme="tripos", alpha=0.7)
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)])
    alphas, lams = [0.5, 1.0], [0.01, 10.0]

    model = CMFModel(fieldset=fs.copy(), scale_fields=False, fit_bias=False)
    res_grid = grid_search(table, y, model, alphas, lams=lams, folds=4, seed=None)
    model_b = CMFModel(fieldset=fs.copy(), scale_fields=False, fit_bias=False)
    res_b = grid_search(table, y, model_b, alphas, lams=lams, folds=4, seed=None, lam_search="brent")
    assert res_b.best_rmse <= res_grid.best_rmse + 1e-12
    assert any(row.get("brent") for row in res_b.table)
    assert lams[0] <= res_b.best_params["lam"] <= lams[1]
    model_b.fit(table, y, cv_folds=4, cv_seed=None, reuse_kernels=True)
    assert model_b.bias == 0.0

    for normalize in (False, True):
        tmpl = CMFModel(fieldset=fs.copy(), normalize=normalize, scale_fields=not normalize)
        r1 = nested_cv(table, y, tmpl, alphas, lams=lams, outer_folds=3, inner_folds=4, seed=None, precompute=True)
        r2 = nested_cv(table, y, tmpl, alphas, lams=lams, outer_folds=3, inner_folds=4, seed=None, precompute=False)
        assert np.allclose(r1.y_pred, r2.y_pred, atol=1e-8)
        assert abs(r1.q2_ex - r2.q2_ex) < 1e-8
