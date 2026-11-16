"""Plan item NT3: size and shape as descriptors of structures.

Six computed descriptors (``calc:atoms``, ``calc:diameter``, ``calc:gyration_radius``,
``calc:aspect_ratio``, ``calc:surface_fraction``, ``calc:surface_area``), a table of measured
values read into the structures, and the descriptor block beside the invariant kernels, which
ignored it before.
"""
import os

import numpy as np
import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("scipy")
from ase.build import bulk, graphene, mx2, nanotube  # noqa: E402
from ase.cluster import Icosahedron, Octahedron  # noqa: E402

from cmfqsar.descriptors import (SHAPE, compute, read_values, shape_names,  # noqa: E402
                                 surface_access)
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.periodic import crystal_table, inorganic_fieldset  # noqa: E402
from cmfqsar.structures import attach_values  # noqa: E402

PHYSCHEM = ("steric", "prop:electronegativity")


def _open(a):
    a.cell[:] = 0.0
    a.pbc = False
    return a


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_the_surface_atoms_of_an_icosahedron_are_its_outer_shell(k):
    a = _open(Icosahedron("Cu", k))
    assert int(np.sum(surface_access(a) > 0)) == 10 * (k - 1) ** 2 + 2


def test_the_surface_atoms_of_an_octahedron_have_fewer_than_twelve_neighbours():
    from ase.neighborlist import neighbor_list
    a = _open(Octahedron("Cu", 9, cutoff=2))
    i, _, _ = neighbor_list("ijd", a, 0.85 * 3.61)
    low = np.bincount(i, minlength=len(a)) < 12
    assert np.array_equal(surface_access(a) > 0, low)


@pytest.mark.parametrize("n,m", [(6, 6), (10, 0), (8, 4)])
def test_the_diameter_of_a_nanotube_is_the_textbook_one(n, m):
    t = nanotube(n, m, length=2)
    d, rg, frac = read_values(t, ["calc:diameter", "calc:gyration_radius",
                                  "calc:surface_fraction"])
    formula = 1.42 * np.sqrt(3.0) * np.sqrt(n * n + n * m + m * m) / np.pi
    assert abs(d - formula) < 1e-2 and abs(2 * rg - formula) < 1e-2
    assert frac == 1.0                        # every atom of a tube is at a surface


def test_the_aspect_ratio_follows_the_sides_of_a_block():
    for L in (1, 2, 4):
        a = _open(bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3 * L)))
        assert abs(compute(a, "calc:aspect_ratio") - L) < 0.02 * L
    assert abs(compute(_open(Icosahedron("Cu", 4)), "calc:aspect_ratio") - 1.0) < 1e-6


def test_a_crystal_has_no_size_and_no_surface():
    c = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(2)
    v = dict(zip(shape_names(), read_values(c, shape_names())))
    assert v["calc:atoms"] == 32 and v["calc:surface_fraction"] == 0.0
    assert v["calc:surface_area"] == 0.0
    for key in ("calc:diameter", "calc:gyration_radius", "calc:aspect_ratio"):
        assert np.isnan(v[key])


def test_a_layer_has_a_thickness_even_split_across_the_cell():
    layer = mx2("MoS2", vacuum=8.0)
    layer.pbc = (True, True, False)
    thick = compute(layer, "calc:diameter")
    assert 3.0 < thick < 3.3                   # the two sulfur planes of MoS2
    split = layer.copy()
    split.translate([0.0, 0.0, -layer.positions[:, 2].min() - 0.5])
    split.wrap(pbc=(True, True, True))         # the bottom sulfur plane moves to the top
    assert abs(compute(split, "calc:diameter") - thick) < 1e-9
    assert np.isnan(compute(layer, "calc:aspect_ratio"))
    sheet = graphene(size=(3, 3, 1), vacuum=6.0)
    sheet.pbc = (True, True, False)
    assert compute(sheet, "calc:surface_fraction") == 1.0


def test_an_unknown_name_lists_the_shape_descriptors():
    with pytest.raises(KeyError, match="surface_fraction"):
        compute(_open(Icosahedron("Cu", 2)), "calc:surface_fractoin")
    assert set(SHAPE) == {k[5:] for k in shape_names()}


def _particles():
    atoms = []
    for el in ("Cu", "Ag"):
        for k in (2, 3, 4, 5):
            atoms.append(_open(Icosahedron(el, k)))
        for L in (4, 6, 8):
            atoms.append(_open(Octahedron(el, L, cutoff=(L - 1) // 3)))
    return atoms


def _model(desc, **kw):
    fs = inorganic_fieldset(["Ag", "Cu"], physchem=("steric",))
    fs.descriptors = tuple(desc)
    return fs, CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                        ac_d_max=6.0, params={"lam": 1e-3}, **kw)


def test_the_periodic_kernel_uses_the_descriptor_block():
    atoms = _particles()
    y = np.array([float(len(a)) for a in atoms])
    scores = {}
    for desc in ((), ("calc:atoms",)):
        fs, m = _model(desc)
        table = crystal_table(atoms, fs, [str(i) for i in range(len(atoms))])
        m.fit(table, y, cv_folds=5)
        scores[desc] = m.stats.q2
        if desc:
            # per atom of the cell, the channels cannot tell the size; the descriptor can
            assert np.allclose(m.predict(table), y, rtol=1e-3)
            d = m.joint_self(table)
            assert np.allclose(d, 1.0 + m.descriptor_self(table.descriptors))
            assert np.allclose(np.diag(m.training_kernel(table)), d)
    assert scores[()] < 0.5 < 0.99 < scores[("calc:atoms",)]


def test_a_saved_periodic_model_keeps_its_descriptor_block():
    atoms = _particles()
    y = np.array([float(len(a)) for a in atoms])
    fs, m = _model(("calc:atoms", "calc:surface_fraction"))
    table = crystal_table(atoms, fs, [str(i) for i in range(len(atoms))])
    m.fit(table, y, cv_folds=5)
    back = CMFModel.from_state(m.state())
    assert np.allclose(back.predict(table), m.predict(table))


def test_the_molecular_autocorrelation_kernel_uses_the_descriptor_block():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    mols = []
    for smi in ("CCO", "CCCO", "CCCCO", "CCCCCO", "OCCO", "OCCCO"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        mols.append(mol)
    kernels = []
    for desc in ((), ("calc:mw",)):
        fs = FieldSet.create(physchem=["steric"], alpha=0.5)
        fs.descriptors = desc
        table = AtomTable.from_molecules([build_fields(mol, fs, name=str(k))
                                          for k, mol in enumerate(mols)])
        m = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation")
        kernels.append(m.training_kernel(table))
    assert not np.allclose(kernels[0], kernels[1])
    assert np.allclose(np.diag(kernels[0]), 1.0)


def _table(tmp_path, text):
    path = tmp_path / "values.csv"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_table_of_measured_values_reaches_the_structures(tmp_path):
    atoms = [_open(Icosahedron("Cu", k)) for k in (2, 3, 4)]
    names = ["small", "middle", "large"]
    path = _table(tmp_path, "name,hydrodynamic_nm,zeta_mV,coating\n"
                            "small,2.5,-30,citrate\nlarge,6.0,-12,citrate\nother,9,1,none\n")
    got, unmatched = attach_values(atoms, names, path)
    assert got == ["hydrodynamic_nm", "zeta_mV"] and unmatched == ["other"]
    values = np.array([read_values(a, got) for a in atoms])
    assert np.allclose(values[0], [2.5, -30.0]) and np.allclose(values[2], [6.0, -12.0])
    assert np.isnan(values[1]).all()
    got, _ = attach_values(atoms, names, path, columns=["zeta_mV"])
    assert got == ["zeta_mV"]
    with pytest.raises(ValueError, match="no column size"):
        attach_values(atoms, names, path, columns=["size"])


def _particle_file(tmp_path):
    from ase.io import write
    atoms = _particles()
    for k, a in enumerate(atoms):
        a.info.update({"name": f"p{k}", "y": float(len(a)) ** (1 / 3)})
    path = str(tmp_path / "particles.extxyz")
    write(path, atoms)
    rows = "\n".join(f"p{k},{0.1 * len(a) ** (1 / 3) + 0.3 * (k % 3)}"
                     for k, a in enumerate(atoms))
    table = _table(tmp_path, "name,hydrodynamic_nm\n" + rows + "\n")
    return path, table


def test_build_crystals_takes_a_table_of_descriptors(tmp_path, capsys):
    from cmfqsar.cli import main
    path, table = _particle_file(tmp_path)
    main(["build-crystals", path, "--target", "y", "--fields", "steric", "--elements", "none",
          "--descriptor-file", table, "--descriptors", "hydrodynamic_nm,calc:atoms",
          "--cutoff", "6", "--folds", "4"])
    out = capsys.readouterr().out
    assert "read hydrodynamic_nm from" in out
    assert "descriptors beside the channels: hydrodynamic_nm, calc:atoms" in out


def test_the_crystal_window_reads_descriptors_from_a_table(qt_application, tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from cmfqsar.gui.crystal_window import CrystalWindow
    path, table = _particle_file(tmp_path)
    w = CrystalWindow(background=False)
    w.load(path)
    w.ed_descriptors.setText("calc:atoms")
    w.import_descriptors(table)
    assert w.ed_descriptors.text() == "calc:atoms,hydrodynamic_nm"
    assert w._fieldset(range(len(w.names))).descriptors == ("calc:atoms", "hydrodynamic_nm")
    w.build_model()
    assert w.model is not None and w.model.desc_train_ is not None
