"""Structures that are not molecules in the Dataset (plan item NT1, cmfqsar/structures.py)."""
import os

import numpy as np
import pytest

from cmfqsar import structures as S
from cmfqsar.dataio import Dataset, load_dataset

ase = pytest.importorskip("ase")
from ase.build import bulk, molecule, nanotube  # noqa: E402
from ase.io import write  # noqa: E402


def _halite(name="halite", **info):
    a = bulk("NaCl", "rocksalt", 5.64)
    a.info.update({"name": name, **info})
    return a


def _water(name="water", **info):
    m = molecule("H2O")
    m.info.update({"name": name, **info})
    return m


def test_a_database_keeps_the_names_and_values_of_its_rows(tmp_path):
    import ase.db
    from ase.io import read
    for ending in ("db", "json"):
        path = str(tmp_path / f"set.{ending}")
        db = ase.db.connect(path)
        db.write(_halite(), name="halite", gap=8.5, subset="test")
        db.write(_water(), name="water", gap=6.0)
        assert read(path, index=":")[0].info == {}      # what ASE's general reader keeps
        ds = load_dataset(path, accept_structures=True)
        assert ds.kind == "structures" and len(ds) == 2
        assert ds.names == ["halite", "water"]
        assert np.allclose(ds.activity, [8.5, 6.0]) and ds.activity_prop == "gap"
        assert list(ds.subset) == ["test", "train"]


def test_a_structure_with_no_cell_keeps_its_coordinates(tmp_path):
    path = str(tmp_path / "set.extxyz")
    water = _water()
    write(path, [_halite(), water], format="extxyz")
    ds = S.load_structures(path)
    got = ds.structures[1]
    assert not any(got.get_pbc()) and S.boundary(got) == 0
    assert np.allclose(got.positions, water.positions)
    from cmfqsar.periodic import centre_layer
    assert np.allclose(centre_layer(water).positions, water.positions)


def test_which_files_hold_structures(tmp_path):
    xyz_plain = tmp_path / "plain.xyz"
    write(str(xyz_plain), _water(), format="xyz")
    xyz_cell = tmp_path / "cell.xyz"
    write(str(xyz_cell), _halite(), format="extxyz")
    other = tmp_path / "other.json"
    other.write_text('{"a": 1}')
    import ase.db
    dbj = str(tmp_path / "rows.json")
    ase.db.connect(dbj).write(_halite())
    assert not S.is_structure_file(str(xyz_plain))
    assert S.is_structure_file(str(xyz_cell))
    assert not S.is_structure_file(str(other))
    assert S.is_structure_file(dbj)
    for name in ("x.cif", "POSCAR", "CONTCAR", "y.vasp", "z.db", "w.extxyz", "t.traj"):
        assert S.is_structure_file(name), name
    for name in ("a.sdf", "b.mol2", "c.smi", "d.csv"):
        assert not S.is_structure_file(name), name


def test_the_number_of_periodic_directions():
    assert S.boundary(_halite()) == 3
    assert S.boundary(_water()) == 0
    tube = nanotube(6, 0, length=1)
    assert S.boundary(tube) == 1 and S.boundary_name(tube) == "chain"
    sheet = _halite()
    sheet.set_pbc((True, True, False))
    assert S.boundary_name(sheet) == "layer"
    bad = _water()
    bad.set_pbc(True)
    with pytest.raises(ValueError, match="zero length"):
        S.boundary(bad)


def test_molecule_commands_refuse_a_structure_file(tmp_path):
    path = str(tmp_path / "halite.cif")
    write(path, _halite(), format="cif")
    with pytest.raises(ValueError, match="build-crystals"):
        load_dataset(path)
    ds = load_dataset(path, accept_structures=True)
    assert ds.kind == "structures" and ds.names == ["halite"] and ds.mols == []


def test_the_target_from_the_info_fields(tmp_path):
    path = str(tmp_path / "set.extxyz")
    a, b = _halite(gap=8.5, cod=1000041), _halite("other", gap=7.0, cod=1000042)
    write(path, [a, b], format="extxyz")
    ds = S.load_structures(path)
    assert ds.activity_prop == "gap" and np.allclose(ds.activity, [8.5, 7.0])
    assert S.numeric_info_keys(ds.structures, targets_only=True) == ["gap"]
    write(path, [_halite(gap=1.0, energy_above_hull=0.1), _halite("x", gap=2.0,
                                                                  energy_above_hull=0.0)],
          format="extxyz")
    ds = S.load_structures(path)
    assert ds.activity_prop == "" and np.all(np.isnan(ds.activity))
    ds = S.load_structures(path, target="energy_above_hull")
    assert np.allclose(ds.activity, [0.1, 0.0])


def test_an_energy_stored_as_a_calculator_result_is_a_target(tmp_path):
    from ase.calculators.singlepoint import SinglePointCalculator
    a = _halite()
    a.calc = SinglePointCalculator(a, energy=-7.5)
    path = str(tmp_path / "set.extxyz")
    write(path, [a], format="extxyz")
    ds = S.load_structures(path, target="energy")
    assert np.allclose(ds.activity, [-7.5])


def test_the_target_from_a_table(tmp_path):
    folder = tmp_path / "cifs"
    folder.mkdir()
    for nm in ("a", "b"):
        write(str(folder / f"{nm}.cif"), _halite(nm), format="cif")
    one = tmp_path / "one.csv"
    one.write_text("name,value\na,1.5\nb,2.5\n")
    ds = S.load_structures(str(folder), target_file=str(one))
    assert ds.names == ["a", "b"] and np.allclose(ds.activity, [1.5, 2.5])
    many = tmp_path / "many.csv"
    many.write_text("name,cod,mp_C,mp_low_C\na,11,1.5,1.4\nb,12,2.5,2.4\n")
    with pytest.raises(ValueError, match="name the one"):
        S.load_structures(str(folder), target_file=str(many))
    ds = S.load_structures(str(folder), target_file=str(many), target_column="mp_low_C")
    assert np.allclose(ds.activity, [1.4, 2.4])
    ids = tmp_path / "ids.csv"
    ids.write_text("name,cod,mp_C\na,11,1.5\nb,12,2.5\n")
    assert S.target_csv_column(str(ids)) == "mp_C"


def test_the_published_split_and_unique_names(tmp_path):
    path = str(tmp_path / "set.extxyz")
    write(path, [_halite("x", subset="train", y=1.0), _halite("x", subset="test", y=2.0),
                 _halite("x", y=3.0)], format="extxyz")
    ds = S.load_structures(path)
    assert ds.names == ["x", "x_2", "x_3"]
    assert list(ds.subset) == ["train", "test", "train"]
    assert ds.activity_prop == "y"


def test_a_dataset_of_structures_appends_and_removes(tmp_path):
    path = str(tmp_path / "set.extxyz")
    write(path, [_halite("a", y=1.0), _halite("b", y=2.0), _halite("c", y=3.0)],
          format="extxyz")
    ds = S.load_structures(path)
    more = S.load_structures(path)
    ds.append(more, subset="test")
    assert len(ds) == 6 and len(ds.structures) == 6 and list(ds.subset).count("test") == 3
    ds.remove([0, 4])
    assert ds.names == ["b", "c", "a", "c"] and len(ds.structures) == 4
    assert np.allclose(ds.activity, [2.0, 3.0, 1.0, 3.0])
    assert np.allclose(ds.classes_from_threshold(2.5), [-1, 1, -1, 1])
    from rdkit import Chem
    mols = Dataset(mols=[Chem.MolFromSmiles("C")], names=["m"], activity=np.zeros(1),
                   subset=np.array(["train"], dtype=object))
    with pytest.raises(ValueError, match="cannot add"):
        mols.append(ds)


def test_the_periodic_directions_given_for_a_whole_set(tmp_path):
    path = str(tmp_path / "sheet.extxyz")
    sheet = _halite("sheet", y=1.0)
    sheet.positions[:, 2] += 0.9 * sheet.cell[2, 2]      # split across the boundary along c
    write(path, [sheet], format="extxyz")
    ds = S.load_structures(path, pbc="ab")
    assert S.boundary_name(ds.structures[0]) == "layer"
    ds = S.load_structures(path, pbc="none")
    assert S.boundary_name(ds.structures[0]) == "open"


def test_the_old_reader_still_names_a_folder_by_its_files(tmp_path):
    from cmfqsar.periodic import read_crystal_set
    folder = tmp_path / "cifs"
    folder.mkdir()
    for nm in ("first", "second"):
        write(str(folder / f"{nm}.cif"), _halite("inside"), format="cif")
    atoms, names = read_crystal_set(str(folder))
    assert names == ["first", "second"]
    assert [a.info["name"] for a in atoms] == names
    assert os.path.basename(str(folder)) == "cifs"


# ------------------------------------------------------------------------------ the application
def _rocksalt_db(path, n=10):
    import ase.db
    db = ase.db.connect(path)
    for k in range(n):
        a = bulk("NaCl", "rocksalt", 5.4 + 0.05 * k, cubic=True)
        db.write(a, name=f"salt_{k}", y=float(k), subset="test" if k in (2, 7) else "train")


def test_the_crystal_window_reads_a_database_with_its_split(qt_application, tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from cmfqsar.gui.crystal_window import CrystalWindow
    path = str(tmp_path / "salts.db")
    _rocksalt_db(path)
    w = CrystalWindow(background=False)
    w.load(path)
    assert w.names == [f"salt_{k}" for k in range(10)]
    assert np.allclose(w.y, np.arange(10.0)) and w.target_name == "y"
    assert [k for k, s in enumerate(w.subset) if s == "test"] == [2, 7]
    w.build_model()
    assert w.model is not None
    assert [k for k, s in enumerate(w.subset) if s == "test"] == [2, 7]


def test_the_main_window_sends_a_structure_file_to_the_crystal_window(qt_application,
                                                                      tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from cmfqsar.gui.main_window import MainWindow
    path = str(tmp_path / "salts.db")
    _rocksalt_db(path, n=4)
    w = MainWindow(background=False)
    w.load_dataset_file(path)
    assert w.ds is None
    cw = w._crystal_window
    cw_names = cw.names
    assert cw_names == [f"salt_{k}" for k in range(4)]


def test_build_crystals_takes_the_one_numeric_field_as_the_target(tmp_path, capsys):
    from cmfqsar.cli import main
    rows = [("NaCl", 5.64, 1.0), ("NaCl", 5.9, 1.3), ("NaCl", 5.4, 0.8), ("MgO", 4.21, 2.0),
            ("MgO", 4.4, 2.3), ("MgO", 4.0, 1.8), ("KBr", 6.6, 0.5), ("KBr", 6.9, 0.7)]
    atoms = []
    for k, (s, a, v) in enumerate(rows):
        at = bulk(s, "rocksalt", a=a)
        at.info.update({"name": f"{s}-{a}", "gap": v, "cod": 1000000 + k})
        atoms.append(at)
    path = str(tmp_path / "salts.extxyz")
    write(path, atoms, format="extxyz")
    main(["build-crystals", path, "--elements", "Na,Cl,Mg,O,K,Br", "--folds", "3",
          "--lams", "0.01"])
    out = capsys.readouterr().out
    assert "the target is the info field 'gap'" in out and "q2 =" in out
    for a in atoms:
        a.info["volume"] = a.get_volume()
    write(path, atoms, format="extxyz")
    with pytest.raises(SystemExit, match="numeric fields gap, volume"):
        main(["build-crystals", path, "--elements", "Na,Cl,Mg,O,K,Br"])


def test_a_tube_and_a_cluster_have_images_only_where_they_repeat():
    from cmfqsar.periodic import cell_image_indices, crystal_fields, inorganic_fieldset
    fs = inorganic_fieldset(["C", "Cu"], physchem=("steric",))
    tube = nanotube(6, 0, length=1)
    coords, idx = cell_image_indices(crystal_fields(tube, fs, "tube"), 6.0)
    moved = coords[len(tube):] - tube.positions[idx[len(tube):]]
    assert len(moved) > 0
    assert np.allclose(moved[:, :2], 0.0)                          # along the axis only
    assert np.allclose(np.mod(moved[:, 2] / tube.cell[2, 2] + 0.5, 1.0), 0.5)
    from ase.cluster import Icosahedron
    cluster = Icosahedron("Cu", 2)
    coords, idx = cell_image_indices(crystal_fields(cluster, fs, "cluster"), 6.0)
    assert len(coords) == len(cluster)


def test_a_cluster_with_no_cell_is_the_cluster_alone_in_a_box():
    from dataclasses import replace
    from ase.cluster import Icosahedron
    from cmfqsar.periodic import (PeriodicSpec, crystal_table, inorganic_fieldset,
                                  periodic_feature_matrix)
    fs = inorganic_fieldset(["Cu", "Au"], physchem=("steric", "prop:electronegativity"))
    open_ = [Icosahedron("Cu", 2), Icosahedron("Au", 3)]
    boxed = []
    for a in open_:
        b = a.copy()
        b.set_cell([40.0, 40.0, 40.0])
        b.center()
        b.set_pbc(True)
        boxed.append(b)
    spec = PeriodicSpec(r_max=7.0)
    A = periodic_feature_matrix(crystal_table(open_, fs, ["a", "b"]).molecules, spec,
                                n_fields=len(fs.specs))
    B = periodic_feature_matrix(crystal_table(boxed, fs, ["a", "b"]).molecules,
                                replace(spec, background=False), n_fields=len(fs.specs))
    assert np.abs(A - B).max() <= 1e-12 * np.abs(B).max()


def test_a_file_name_with_an_at_sign_is_read(tmp_path):
    folder = tmp_path / "sheets"
    folder.mkdir()
    write(str(folder / "Ag@C4.cif"), _halite(), format="cif")
    ds = S.load_structures(str(folder))
    assert ds.names == ["Ag@C4"]
