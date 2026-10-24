"""Tests for the periodic field autocorrelation: continuous molecular fields of crystals."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ase = pytest.importorskip("ase")
pytest.importorskip("pymatgen.core")

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from cmfqsar.fields import AtomTable  # noqa: E402
from cmfqsar.invariant import autocorrelation, field_pairs  # noqa: E402
from cmfqsar.periodic import (CrystalFields, PeriodicSpec, channel_weights,  # noqa: E402
                              crystal_charges, crystal_fields, crystal_table, elements_in,
                              inorganic_fieldset, periodic_autocorrelation,
                              periodic_block_sizes, periodic_channel_labels,
                              periodic_feature_matrix, structures_from_matbench, top_channels)


@pytest.fixture(scope="module")
def world():
    fs = inorganic_fieldset(["Na", "Cl", "Mg", "O"])
    structures = {
        "NaCl": bulk("NaCl", "rocksalt", a=5.64),
        "MgO": bulk("MgO", "rocksalt", a=4.21),
        "CsCl": bulk("CsCl", "cesiumchloride", a=4.12),
        "Si": bulk("Si", "diamond", a=5.43),
        "Fe": bulk("Fe", "bcc", a=2.87),
        "KBr": bulk("KBr", "rocksalt", a=6.60),
    }
    crystals = {k: crystal_fields(a, fs, k) for k, a in structures.items()}
    return fs, structures, crystals


def test_the_grid_the_blocks_and_the_labels_agree(world):
    fs, _, crystals = world
    spec = PeriodicSpec(r_max=6.0, step=1.0, sigma=0.7)
    assert len(spec.centres) == 7 and spec.n_bins == 7 and spec.d_max == 6.0
    row = periodic_autocorrelation(crystals["NaCl"], spec)
    n_pairs = len(field_pairs(fs.n_fields))
    assert row.shape == (n_pairs * 7,)
    assert sum(periodic_block_sizes(spec, fs.n_fields)) == len(row)
    labels = periodic_channel_labels(fs, spec)
    assert len(labels) == len(row)
    assert labels[0].lower().startswith("electrostatic @ 0.00 a")
    assert any("CIF Na x CIF Cl" in lab for lab in labels)


def test_a_supercell_a_rotation_and_an_origin_shift_change_nothing(world):
    """The descriptor is per atom of the cell and summed over the neighbour list with images, so
    it does not know which cell was chosen, where the origin is or how the lattice is turned."""
    fs, structures, crystals = world
    spec = PeriodicSpec()
    for name in ("NaCl", "Si"):
        a = structures[name]
        here = periodic_autocorrelation(crystals[name], spec)
        sup = crystal_fields(a.repeat((2, 1, 3)), fs, "sup")
        rot = a.copy()
        rot.rotate(37.0, (1, 2, 3), rotate_cell=True)
        rot.translate((0.3, -1.2, 2.0))
        rot.wrap()
        shifted = crystal_fields(rot, fs, "rot")
        assert np.allclose(here, periodic_autocorrelation(sup, spec), rtol=1e-9, atol=1e-9)
        assert np.allclose(here, periodic_autocorrelation(shifted, spec), rtol=1e-9, atol=1e-9)
    conv = crystal_fields(bulk("NaCl", "rocksalt", a=5.64, cubic=True), fs, "conv")
    assert np.allclose(periodic_autocorrelation(crystals["NaCl"], spec),
                       periodic_autocorrelation(conv, spec), rtol=1e-9, atol=1e-9)


def test_but_the_structure_and_the_lattice_constant_do(world):
    fs, _, crystals = world
    spec = PeriodicSpec()
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(a)
    nacl = periodic_autocorrelation(crystals["NaCl"], spec)
    # the same composition on the other lattice at the same density
    other = crystal_fields(bulk("NaCl", "cesiumchloride", a=5.64 / 4 ** (1 / 3)), fs, "NaCl-B2")
    assert rel(nacl, periodic_autocorrelation(other, spec)) > 0.1
    bigger = crystal_fields(bulk("NaCl", "rocksalt", a=5.9), fs, "NaCl-big")
    assert rel(nacl, periodic_autocorrelation(bigger, spec)) > 0.05


def test_the_background_removes_the_uniform_term():
    """A box of randomly placed atoms is a uniform distribution: the raw channels follow the shell
    integral and the subtracted ones are nothing but noise about zero."""
    rng = np.random.default_rng(5)
    n, L = 3000, 36.0
    atoms = Atoms("Ar" * n, positions=rng.uniform(0, L, size=(n, 3)), cell=np.eye(3) * L, pbc=True)
    c = CrystalFields(name="gas", coords=atoms.get_positions(), weights=np.ones((n, 1)),
                      radii=np.ones(n), symbols=["Ar"] * n, cell=atoms.get_cell()[:])
    raw_spec = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, background=False, include_self=False)
    sub_spec = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, background=True, include_self=False)
    raw = periodic_autocorrelation(c, raw_spec)
    sub = periodic_autocorrelation(c, sub_spec)
    expect = n / L ** 3 * raw_spec.shell_integral()
    far = raw_spec.centres >= 2.0                       # where the count is large enough to be stable
    assert np.allclose(raw[far], expect[far], rtol=0.05)
    assert np.abs(sub[far]).max() < 0.05 * raw[far].max()
    # the subtraction is exactly the uniform term of this field pair
    assert np.allclose(raw - sub, expect, rtol=1e-9)


def test_the_taper_makes_the_cutoff_continuous():
    """Neighbours just inside a tapered cutoff contribute almost nothing; inside a hard one they
    contribute in full, so a pair walking out of the sphere would jump."""
    a = 7.99
    one = Atoms("Fe", positions=[[0, 0, 0]], cell=np.eye(3) * a, pbc=True)
    c = CrystalFields(name="sc", coords=one.get_positions(), weights=np.ones((1, 1)),
                      radii=np.ones(1), symbols=["Fe"], cell=one.get_cell()[:])
    hard = PeriodicSpec(r_max=8.0, step=1.0, sigma=0.5, taper=0.0, background=False, include_self=False)
    soft = PeriodicSpec(r_max=8.0, step=1.0, sigma=0.5, taper=2.0, background=False, include_self=False)
    h, s = periodic_autocorrelation(c, hard), periodic_autocorrelation(c, soft)
    assert h[-1] > 1.0                      # six neighbours at 7.99 A land in the 8 A channel
    assert s[-1] < 1e-3 * h[-1]
    w = soft.window(np.array([0.0, 5.9, 6.0, 7.0, 7.999, 8.0, 9.0]))
    assert np.allclose(w, [1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0], atol=1e-6)


def test_a_molecule_in_a_huge_box_is_the_molecular_descriptor():
    """With no neighbour closer than the cutoff, no background and a hard cutoff, the periodic sum
    is the molecular autocorrelation of the same atoms: the two agree exactly."""
    from cmfqsar.fields import FieldSet, build_fields
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccccc1C(=O)N"))
    AllChem.EmbedMolecule(m, randomSeed=3)
    fs = FieldSet.create(alpha=0.5)
    mf = build_fields(m, fs, "m")
    box = Atoms([a.GetSymbol() for a in m.GetAtoms()], positions=mf.coords, cell=np.eye(3) * 60.0, pbc=True)
    c = CrystalFields(name="m", coords=mf.coords, weights=mf.weights, radii=mf.radii,
                      symbols=box.get_chemical_symbols(), cell=box.get_cell()[:])
    from cmfqsar.invariant import AutocorrelationSpec
    mol_spec = AutocorrelationSpec(d_max=12.0, step=0.75, sigma=0.75)
    per_spec = PeriodicSpec(r_max=12.0, step=0.75, sigma=0.75, taper=0.0, background=False)
    want = autocorrelation(mf, mol_spec) / mf.n_atoms
    assert np.allclose(periodic_autocorrelation(c, per_spec), want, rtol=1e-9, atol=1e-9)


def test_the_inorganic_weights(world):
    fs, structures, crystals = world
    assert [s.label for s in fs.specs] == ["Electrostatic", "Steric", "electronegativity",
                                           "CIF Na", "CIF Cl", "CIF Mg", "CIF O", "CIF other"]
    nacl = crystals["NaCl"]
    assert nacl.charge_source == "oxidation"
    assert dict(zip(nacl.symbols, nacl.weights[:, 0])) == {"Na": 1.0, "Cl": -1.0}
    assert dict(zip(crystals["MgO"].symbols, crystals["MgO"].weights[:, 0])) == {"Mg": 2.0, "O": -2.0}
    assert crystals["Fe"].weights[0, 0] == 0.0                     # an elemental solid
    assert crystals["Fe"].weights[0, 7] == 1.0 and crystals["Fe"].weights[0, 3:7].sum() == 0  # "other"
    assert nacl.weights[0, 3] == 1.0 and nacl.weights[1, 4] == 1.0 and nacl.weights[:, 7].sum() == 0
    assert nacl.weights[1, 2] > nacl.weights[0, 2]                 # Cl is the more electronegative
    assert nacl.weights[0, 1] > nacl.weights[1, 1]                 # and Na the bigger atom
    q, src = crystal_charges(structures["NaCl"], "electronegativity")
    assert src == "electronegativity" and q[0] > 0 > q[1] and abs(q.sum()) < 1e-12
    with pytest.raises(ValueError, match="no per-atom"):
        crystal_charges(structures["NaCl"], "file")
    tagged = structures["NaCl"].copy()
    tagged.set_array("oxi_states", np.array([0.9, -0.9]))
    q, src = crystal_charges(tagged, "file")
    assert src == "oxi_states" and np.allclose(q, [0.9, -0.9])
    assert elements_in(structures.values(), min_structures=2) == ["Cl"]       # NaCl and CsCl
    assert elements_in(structures.values(), min_structures=1) == sorted(
        {"Na", "Cl", "Mg", "O", "Cs", "Si", "Fe", "K", "Br"})
    with pytest.raises(ValueError, match="not defined for a periodic solid"):
        crystal_fields(structures["NaCl"], inorganic_fieldset(physchem=["hydrophobic"]))
    with pytest.raises(ValueError, match="no per-atom array"):
        crystal_fields(structures["NaCl"], inorganic_fieldset(physchem=["prop:bader"]))
    tagged.set_array("bader", np.array([0.8, -0.8]))
    c = crystal_fields(tagged, inorganic_fieldset(physchem=["prop:bader"]))
    assert np.allclose(c.weights[:, 0], [0.8, -0.8])
    c = crystal_fields(structures["NaCl"], inorganic_fieldset(physchem=["prop:mass", "prop:inverse_sqrt_mass"]))
    assert np.allclose(c.weights[:, 0], [22.990, 35.45], atol=1e-2)
    assert np.allclose(c.weights[:, 1], c.weights[:, 0] ** -0.5)


def test_the_feature_matrix_refuses_mixed_field_sets(world):
    fs, structures, crystals = world
    other = crystal_fields(structures["NaCl"], inorganic_fieldset(["Na"]), "x")
    with pytest.raises(ValueError, match="one field set"):
        periodic_feature_matrix([crystals["NaCl"], other], PeriodicSpec())


def _series(fs):
    rows = [("NaCl", 5.64, 1.0), ("NaCl", 5.9, 1.3), ("NaCl", 5.4, 0.8), ("MgO", 4.21, 2.0),
            ("MgO", 4.4, 2.3), ("MgO", 4.0, 1.8), ("KBr", 6.6, 0.5), ("KBr", 6.9, 0.7)]
    crys = [crystal_fields(bulk(s, "rocksalt", a=a), fs, f"{s}-{a}") for s, a, _ in rows]
    return crys, np.array([v for _, _, v in rows])


def test_a_model_on_the_periodic_kernel_fits_predicts_and_round_trips(tmp_path, world):
    from cmfqsar.dataio import load_model, save_model
    from cmfqsar.model import CMFModel
    fs, _, _ = world
    crys, y = _series(fs)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_sigma=0.5, ac_d_max=9.0, pbc_taper=1.5, params={"lam": 0.01})
    model.fit(AtomTable.from_molecules(crys[:7]), y[:7], cv_folds=3)
    assert np.isfinite(model.stats.q2)
    here = model.predict(AtomTable.from_molecules(crys[7:]))
    assert here.shape == (1,)
    # a supercell of the test structure is the same material and gets the same prediction
    sup = crystal_fields(bulk("KBr", "rocksalt", a=6.9).repeat((2, 2, 1)), fs, "sup")
    assert np.allclose(model.predict(AtomTable.from_molecules([sup])), here, atol=1e-9)
    with pytest.raises(ValueError, match="invariant to rotation"):
        model.cofield_coefficients(0)
    path = str(tmp_path / "crystal.cmf")
    save_model(path, model)
    back = load_model(path)[0]
    assert back.kernel_source == "periodic" and back.pbc_taper == 1.5 and back.ac_d_max == 9.0
    assert np.allclose(back.predict(AtomTable.from_molecules(crys[7:])), here, atol=1e-9)
    fresh = model.fresh_copy()
    assert fresh.kernel_source == "periodic" and fresh.pbc_taper == 1.5 and fresh.pbc_background


def test_the_channel_weights_reproduce_the_prediction(world):
    from cmfqsar.model import CMFModel
    fs, _, _ = world
    crys, y = _series(fs)
    table = AtomTable.from_molecules(crys)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     params={"lam": 0.1})
    model.fit(table, y, cv_folds=0)
    w = channel_weights(model)
    Z = model.ac_train_ * model.ac_scale_
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    assert np.allclose(Z @ w + model.bias, model.predict(table), atol=1e-9)
    labels = periodic_channel_labels(fs, model._ac_spec())
    top = top_channels(model, labels, n=5)
    assert len(top) == 5 and all(lab in labels for lab, _, _ in top)
    assert np.isfinite([v for _, v, _ in top]).all()


def test_the_nested_cross_validation_keeps_the_kernel(world):
    from cmfqsar.model import CMFModel, nested_cv
    fs, _, _ = world
    crys, y = _series(fs)
    kw = dict(alphas=[1.0], lams=[0.01, 0.1, 1.0], outer_folds=4, inner_folds=2, repeats=1,
              seed=0, precompute=False)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic")
    res = nested_cv(AtomTable.from_molecules(crys), y, model, **kw)
    assert np.isfinite(res.rmse_ex) and res.y_pred.shape == (len(y),)
    sups = [crystal_fields(bulk(c.name.split("-")[0], "rocksalt", a=float(c.name.split("-")[1])).repeat((1, 2, 1)), fs, c.name)
            for c in crys]
    again = nested_cv(AtomTable.from_molecules(sups), y, model, **kw)
    assert np.allclose(res.y_pred, again.y_pred, atol=1e-8)


def test_a_table_can_be_built_straight_from_structures(world):
    fs, structures, _ = world
    tab = crystal_table(list(structures.values()), fs, list(structures))
    assert tab.n_mol == len(structures) and tab.names[0] == "NaCl"
    assert tab.weights.shape == (sum(len(a) for a in structures.values()), fs.n_fields)


def test_the_matbench_parser():
    table = {
        "index": ["mb-1", "mb-2"],
        "columns": ["structure", "last phdos peak"],
        "data": [
            [{"lattice": {"matrix": [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]},
              "sites": [{"species": [{"element": "Na", "occu": 1}], "xyz": [0, 0, 0]},
                        {"species": [{"element": "Cl", "occu": 1}], "xyz": [2, 2, 2]}]}, 150.0],
            [{"lattice": {"matrix": [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]},
              "sites": [{"species": [{"element": "Fe", "occu": 1}], "xyz": [0, 0, 0]}]}, 300.0],
        ],
    }
    atoms, y, names = structures_from_matbench(table)
    assert names == ["mb-1", "mb-2"] and np.allclose(y, [150.0, 300.0])
    assert atoms[0].get_chemical_symbols() == ["Na", "Cl"] and atoms[0].cell[0, 0] == 4.0
    assert atoms[1].pbc.all() and len(atoms[1]) == 1


# ----------------------------------------------------------------------------------------------
# mixing coefficients over the channel blocks: the indicator pairs against the rest
# ----------------------------------------------------------------------------------------------

def test_zero_weights_on_the_indicator_blocks_are_the_physico_chemical_model(world):
    """With the indicator pairs and the cross pairs weighed zero, the 8-field model is the
    3-field model exactly: that is what removing a block means."""
    from cmfqsar.model import CMFModel
    fs, _, _ = world
    crys, y = _series(fs)
    fs_pc = inorganic_fieldset([])
    crys_pc, _ = _series(fs_pc)
    full = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                    params={"lam": 0.01},
                    ac_block_weights={"indicator": 0.0, "indicator*physchem": 0.0})
    small = CMFModel(fieldset=fs_pc, method="krr", normalize=True, kernel_source="periodic",
                     params={"lam": 0.01})
    assert np.allclose(full.training_kernel(AtomTable.from_molecules(crys)),
                       small.training_kernel(AtomTable.from_molecules(crys_pc)), atol=1e-10)
    full.fit(AtomTable.from_molecules(crys[:7]), y[:7], cv_folds=0)
    small.fit(AtomTable.from_molecules(crys_pc[:7]), y[:7], cv_folds=0)
    assert np.allclose(full.predict(AtomTable.from_molecules(crys[7:])),
                       small.predict(AtomTable.from_molecules(crys_pc[7:])), atol=1e-9)
    kinds = {r["kind"]: r["n_blocks"] for r in full.block_weight_report()}
    assert kinds == {"physchem": 6, "indicator": 15, "indicator*physchem": 15}
    # the co-pattern sees the weights: the removed channels carry no weight
    w = channel_weights(full)
    labels = periodic_channel_labels(fs, full._ac_spec())
    assert all(w[i] == 0.0 for i, lab in enumerate(labels) if "CIF" in lab)
    assert np.abs(w).max() > 0


@pytest.mark.parametrize("mode", ["balanced", "align", "alignf"])
def test_weighted_models_fit_predict_and_round_trip(tmp_path, world, mode):
    from cmfqsar.dataio import load_model, save_model
    from cmfqsar.model import CMFModel
    fs, _, _ = world
    crys, y = _series(fs)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     params={"lam": 0.01}, ac_block_weights=mode, ac_block_rho=0.5)
    model.fit(AtomTable.from_molecules(crys[:7]), y[:7], cv_folds=3)
    assert np.isfinite(model.stats.q2)
    here = model.predict(AtomTable.from_molecules(crys[7:]))
    assert np.isfinite(here).all()
    sup = crystal_fields(bulk("KBr", "rocksalt", a=6.9).repeat((2, 2, 1)), fs, "sup")
    assert np.allclose(model.predict(AtomTable.from_molecules([sup])), here, atol=1e-9)
    rows = model.block_weight_report()
    assert {r["kind"] for r in rows} == {"physchem", "indicator", "indicator*physchem"}
    if mode == "balanced":
        assert all(r["share"] == pytest.approx(1.0 / 3) for r in rows)
    else:
        assert np.isfinite(model.ac_block_weights_).all() and (model.ac_block_weights_ >= 0).all()
    path = str(tmp_path / f"{mode}.cmf")
    save_model(path, model)
    back = load_model(path)[0]
    assert back.ac_block_weights == mode and back.ac_block_rho == 0.5
    assert back.block_weight_report() == rows
    assert np.allclose(back.predict(AtomTable.from_molecules(crys[7:])), here, atol=1e-9)
    assert back.fresh_copy().ac_block_weights == mode


def test_the_nested_cross_validation_carries_the_block_weights_into_every_fold(world):
    from cmfqsar.model import CMFModel, nested_cv
    fs, _, _ = world
    crys, y = _series(fs)
    fs_pc = inorganic_fieldset([])
    crys_pc, _ = _series(fs_pc)
    kw = dict(alphas=[1.0], lams=[0.01, 0.1, 1.0], outer_folds=4, inner_folds=2, repeats=1,
              seed=0, precompute=False)
    full = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                    ac_block_weights={"indicator": 0.0, "indicator*physchem": 0.0})
    small = CMFModel(fieldset=fs_pc, method="krr", normalize=True, kernel_source="periodic")
    a = nested_cv(AtomTable.from_molecules(crys), y, full, **kw)
    b = nested_cv(AtomTable.from_molecules(crys_pc), y, small, **kw)
    assert np.allclose(a.y_pred, b.y_pred, atol=1e-8)
    plain = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic")
    c = nested_cv(AtomTable.from_molecules(crys), y, plain, **kw)
    assert not np.allclose(a.y_pred, c.y_pred, atol=1e-6)      # the weights were not dropped
    aligned = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                       ac_block_weights="align")
    d = nested_cv(AtomTable.from_molecules(crys), y, aligned, **kw)
    assert d.y_pred.shape == (len(y),) and np.isfinite(d.y_pred).all()


def test_the_command_line_takes_the_block_weights(tmp_path, capsys):
    from ase.io import write
    from cmfqsar.cli import main
    from cmfqsar.dataio import load_model
    rows = [("NaCl", 5.64, 1.0), ("NaCl", 5.9, 1.3), ("NaCl", 5.4, 0.8), ("MgO", 4.21, 2.0),
            ("MgO", 4.4, 2.3), ("MgO", 4.0, 1.8), ("KBr", 6.6, 0.5), ("KBr", 6.9, 0.7),
            ("LiF", 4.0, 3.1), ("LiF", 4.2, 2.8)]
    atoms = []
    for s, a, v in rows:
        at = bulk(s, "rocksalt", a=a)
        at.info["name"], at.info["y"] = f"{s}-{a}", v
        atoms.append(at)
    path = str(tmp_path / "rocksalts.extxyz")
    write(path, atoms, format="extxyz")
    saved = str(tmp_path / "balanced.cmf")
    main(["build-crystals", path, "--target", "y", "--elements", "Na,Cl,Mg,O,K,Br",
          "--block-weights", "balanced", "--folds", "3", "--lams", "0.01,0.1", "--save", saved])
    out = capsys.readouterr().out
    assert "Block weights" in out and "indicator*physchem" in out
    back = load_model(saved)[0]
    assert back.ac_block_weights == "balanced" and back.ac_block_weights_ is not None
    # named kinds over named groups, with a pattern; the alignment with the fold-wise re-derivation
    main(["build-crystals", path, "--target", "y", "--elements", "Na,Cl,Mg,O,K,Br",
          "--block-weights", "alkali=2,alkali*physchem=0", "--block-groups", "alkali=cif:Na,cif:K",
          "--folds", "3", "--lams", "0.01"])
    out = capsys.readouterr().out
    assert "alkali" in out and "alkali*physchem" in out and "alkali*indicator" in out
    main(["build-crystals", path, "--target", "y", "--elements", "Na,Cl,Mg,O,K,Br",
          "--block-weights", "align", "--block-groups", "halide=cif:*l,cif:Br",
          "--folds", "3", "--lams", "0.01,0.1", "--external-cv", "3"])
    out = capsys.readouterr().out
    assert "halide" in out and "External 3-fold CV" in out
    with pytest.raises(SystemExit, match="no field matches"):
        main(["build-crystals", path, "--target", "y", "--block-weights", "balanced",
              "--block-groups", "x=cif:Xe"])
