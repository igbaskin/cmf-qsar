"""The Hammett example: substituted benzenes in one frame, their constants, and a model on them."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

import make_hammett_examples as mh  # noqa: E402


def test_table_is_sane():
    labels = [row[0] for row in mh.SUBSTITUENTS]
    assert len(labels) >= 90
    assert len(set(labels)) == len(labels)
    for label, frag, sm, sp, pi in mh.SUBSTITUENTS:
        mol = Chem.MolFromSmiles(mh.smiles_for(frag))
        assert mol is not None, label
        assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 0, label
        assert -1.0 < sm < 1.3 and -1.0 < sp < 1.3, label
        if pi is not None:
            assert -2.5 < pi < 3.0, label
    by_label = {row[0]: row for row in mh.SUBSTITUENTS}
    assert by_label["H"][2:4] == (0.0, 0.0)
    assert by_label["NO2"][3] > 0.7 and by_label["N(CH3)2"][3] < -0.8     # the two ends


def _ring_frame_ok(coords):
    ring = coords[mh.RING_ATOMS]
    assert np.allclose(ring.mean(axis=0), 0.0, atol=1e-6)
    assert coords[mh.IPSO, 0] > 1.3 and abs(coords[mh.IPSO, 1]) < 1e-6 and abs(coords[mh.IPSO, 2]) < 1e-6
    assert np.abs(ring[:, 2]).max() < 0.1


def test_place_in_frame_undoes_a_rigid_motion():
    rng = np.random.RandomState(3)
    base = mh.build("OC", n_conformers=4)
    ref = np.array(base.GetConformer().GetPositions(), float)
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    moved = ref @ rot.T + np.array([3.0, -2.0, 7.0])
    back = mh.place_in_frame(moved)
    _ring_frame_ok(back)
    assert np.allclose(back, ref, atol=1e-5)


def test_built_molecule_sits_in_frame_and_leans_positive():
    mol = mh.build("OC", n_conformers=6)
    coords = np.array(mol.GetConformer().GetPositions(), float)
    _ring_frame_ok(coords)
    methyl = coords[7]                      # ring 0-5, O 6, C 7
    assert methyl[1] > -1e-6 and methyl[2] > -1e-6
    assert all(a.GetIsAromatic() for a in list(mol.GetAtoms())[:6])      # MMFF left no trace
    assert mol.HasProp("mmff_energy")


def test_symmetry_images_are_reflections_that_keep_properties(tmp_path):
    mol = mh.build("N(C)C", n_conformers=4)
    for a in mol.GetAtoms():
        a.SetDoubleProp("q_test", 0.01 * a.GetIdx())
    images = mh.symmetry_images(mol)
    assert len(images) == 4
    base = np.array(mol.GetConformer().GetPositions(), float)
    signs = set()
    for im in images:
        c = np.array(im.GetConformer().GetPositions(), float)
        sy = np.sign(np.sum(c[:, 1] * base[:, 1])) or 1.0
        sz = np.sign(np.sum(c[:, 2] * base[:, 2])) or 1.0
        assert np.allclose(c, base * np.array([1.0, sy, sz]), atol=1e-6)
        signs.add((sy, sz))
        assert [a.GetDoubleProp("q_test") for a in im.GetAtoms()] == \
               [a.GetDoubleProp("q_test") for a in mol.GetAtoms()]
    assert len(signs) == 4
    # the file round trip: the atom property survives symmetrisation of an SDF
    src = tmp_path / "one.sdf"
    mol.SetProp("_Name", "NMe2")
    mol.SetDoubleProp("sigma_p", -0.83)
    mh.write_sdf(str(src), [mol], atom_props=["q_test"])
    dst = tmp_path / "one_sym.sdf"
    assert mh.symmetrize_file(str(src), str(dst)) == 1
    out = [m for m in Chem.SDMolSupplier(str(dst), removeHs=False)]
    assert len(out) == 4 and all(m.GetProp("_Name") == "NMe2" for m in out)
    assert all(abs(m.GetDoubleProp("sigma_p") + 0.83) < 1e-9 for m in out)
    assert all(abs(m.GetAtomWithIdx(3).GetDoubleProp("q_test") - 0.03) < 1e-9 for m in out)


def test_conformers_are_distinct_in_frame_and_the_lowest_is_build():
    mol, kept = mh.build_conformers("CCCC", keep=6)          # n-butyl: several rotamers
    assert len(kept) >= 2
    energies = [e for _, e in kept]
    assert energies == sorted(energies)
    atoms = mh._shape_atoms(mol)
    for i, (c, _) in enumerate(kept):
        _ring_frame_ok(c)
        assert mh._distinct_in_frame(c, [k for k, _ in kept[:i]], atoms, 0.3)
    one = mh.build("CCCC")
    assert np.allclose(np.array(one.GetConformer().GetPositions()), kept[0][0], atol=1e-6)
    assert abs(one.GetDoubleProp("mmff_energy") - kept[0][1]) < 1e-9
    # benzamide has one conformer in MMFF: the twisted amide and its mirror image are images here
    _, amide = mh.build_conformers("C(N)=O", keep=6)
    assert len(amide) == 1
    # a molecule without MMFF parameters is still optimised (UFF), not left as embedded
    _, sf5 = mh.build_conformers("S(F)(F)(F)(F)F", keep=6)
    assert all(np.isfinite(e) and e != -1.0 for _, e in sf5)


def test_ensemble_file_round_trip_gives_boltzmann_weights(tmp_path):
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import FieldSet, conformer_weights
    rows = [r for r in mh.SUBSTITUENTS if r[0] in ("C2H5", "n-C4H9")]
    # a file with atom properties to copy, as the ORCA run would leave it
    src = tmp_path / "qm.sdf"
    singles = []
    for label, frag, sm, sp, pi in rows:
        m = mh.build(frag)
        mh._set_constants(m, label, sm, sp, pi)
        for a in m.GetAtoms():
            a.SetDoubleProp("q_test", 0.001 * a.GetIdx())
        singles.append(m)
    mh.write_sdf(str(src), singles, atom_props=["q_test"])
    records, counts, props = mh.make_ensemble(keep=4, rows=rows, atom_props_from=str(src))
    assert props == ["q_test"] and len(records) == 4 * sum(counts) and max(counts) >= 2
    ens = tmp_path / "ens.sdf"
    mh.write_sdf(str(ens), records, atom_props=props)
    ds = load_dataset(str(ens), activity_prop="sigma_p", ensemble_by="name")
    assert len(ds) == 2
    fs = FieldSet.create(physchem=["electrostatic"], atom_properties=["q_test"],
                         conformers="ensemble", conformer_weighting="boltzmann", conformer_kt=0.6)
    for m, n in zip(ds.mols, counts):
        assert m.GetNumConformers() == 4 * n
        assert len(m.GetProp("conformer_energies").split(",")) == 4 * n
        assert all(a.HasProp("q_test") for a in m.GetAtoms())
        w = conformer_weights(m, fs)
        assert abs(w.sum() - 1.0) < 1e-12
        if n > 1:
            assert w.max() > 1.0 / (4 * n) + 1e-6          # not uniform
            assert np.allclose(w[:4], w[0])                  # the four images weigh the same


def test_small_model_runs_end_to_end(tmp_path):
    """A dozen substituents through the benchmark script: the files it promises appear."""
    import run_hammett as rh
    keep = ["H", "CH3", "CF3", "F", "Cl", "OCH3", "NH2", "N(CH3)2", "NO2", "CN", "COOCH3", "SO2CH3"]
    rows = [r for r in mh.SUBSTITUENTS if r[0] in keep]
    mols = []
    for label, frag, sm, sp, pi in rows:
        m = mh.build(frag, n_conformers=3)
        m.SetProp("_Name", label)
        m.SetDoubleProp("sigma_p", sp)
        mols.append(m)
    images = []
    for m in mols:
        images.extend(mh.symmetry_images(m))
    sdf = tmp_path / "small_sym.sdf"
    mh.write_sdf(str(sdf), images)
    out = tmp_path / "res"
    rc = rh.main(["--sdf", str(sdf), "--targets", "sigma_p", "--alphas", "1.0", "--lams", "0.1,1",
                  "--folds", "4", "--no-ablation", "--out-dir", str(out)])
    assert rc == 0
    # a data set other than the shipped ones names the outputs after its file
    assert (out / "summary_small_sym.csv").exists() and (out / "predictions_sigma_p_small_sym.csv").exists()
    assert (out / "cofields_small_sym.png").exists() and (out / "cv_scatter_small_sym.png").exists()
    lines = (out / "predictions_sigma_p_small_sym.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + len(keep)
    # the ensemble of four images gives a co-field symmetric in y and z
    mols_, names, y = rh.load(str(sdf), "sigma_p", ensemble=True)
    fs = rh.field_set(["electrostatic"], ensemble=True)
    model, _ = rh.fit(mols_, names, y, fs, [1.0], [1.0], 4)
    xs = np.array([0.0, 2.0, 4.0])
    ys = np.array([-2.0, 2.0])
    s_up = rh.cofield_slice(model, 0, 1.0, xs, ys)
    s_dn = rh.cofield_slice(model, 0, -1.0, xs, ys)
    assert np.allclose(s_up, s_dn, atol=1e-8)
    assert np.allclose(s_up[0], s_up[1], atol=1e-8)
