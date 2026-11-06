"""`cmf predict` on a crystal, and what has to match between fitting and predicting (item M13).

`predict` built molecular fields for whatever it was given, so a model fitted on cells could not
be applied at all: the command read the file as molecules and failed, or worse, predicted from
fields that were not the model's.  These tests drive the command line itself, with the smallest
set that still exercises each branch, and check the three things that make the periodic path
usable:

* a model fitted with `build-crystals` predicts a folder of structure files, and the folder is
  read the same way it was read for fitting, so the names match the target file;
* the contract check asks the questions a crystal raises -- the lattice, the elements the fields
  can see, the frame of the overlap kernel -- instead of the ones a molecule raises, and refuses
  a prediction that would be silently wrong;
* the options that only mean something for a molecule (aligning a pose) are refused rather than
  ignored.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("ase", reason="the crystal path needs ASE")

from ase import Atoms                                              # noqa: E402
from ase.io import write as ase_write                              # noqa: E402

from cmfqsar import periodic                                       # noqa: E402
from cmfqsar.cli import main                                       # noqa: E402
from cmfqsar.contract import check_crystals                        # noqa: E402
from cmfqsar.dataio import load_model                              # noqa: E402

ELEMENTS = "Na,K,Cl,Br"


def rocksalt(a=4.2, cation="Na", anion="Cl"):
    pos = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                    [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]])
    return Atoms(symbols=[cation] * 4 + [anion] * 4, scaled_positions=pos, cell=[a, a, a],
                 pbc=True)


def a_set(folder: str, n: int = 6):
    """`n` rock-salt cells as one CIF file each, with a CSV of their lattice constants."""
    os.makedirs(folder, exist_ok=True)
    names, values = [], []
    for k in range(n):
        a = 4.0 + 0.1 * k
        cation, anion = ("Na", "Cl") if k % 2 else ("K", "Br")
        name = f"cell{k}"
        ase_write(os.path.join(folder, f"{name}.cif"), rocksalt(a, cation, anion))
        names.append(name)
        values.append(10.0 * a + (1.0 if cation == "K" else 0.0))
    csv = os.path.join(folder, "values.csv")
    with open(csv, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("name,lattice\n")
        for name, v in zip(names, values):
            fh.write(f"{name},{v}\n")
    return csv, names, np.array(values, float)


@pytest.fixture()
def built(tmp_path):
    """A crystal model fitted from the command line on a folder of CIF files."""
    folder = os.path.join(str(tmp_path), "cells")
    csv, names, y = a_set(folder)
    model_path = os.path.join(str(tmp_path), "cells.cmf")
    main(["build-crystals", folder, "--target-file", csv, "--target-column", "lattice",
          "--elements", ELEMENTS, "--cutoff", "6", "--sigma", "1", "--folds", "3",
          "--lams", "1e-3,1e-2", "--top", "3", "--save", model_path])
    return folder, csv, model_path, names, y


# ------------------------------------------------------------------ reading a folder as a set
def test_a_folder_of_structure_files_is_a_data_set(tmp_path):
    folder = os.path.join(str(tmp_path), "cells")
    a_set(folder, n=4)
    atoms, names = periodic.read_crystal_set(folder)
    assert names == ["cell0", "cell1", "cell2", "cell3"]        # named after their files
    assert all(len(a) == 8 for a in atoms)
    # the CSV in the same folder is not a structure file and is not read as one
    assert len(atoms) == 4


def test_the_periodic_directions_can_be_set_for_the_whole_set(tmp_path):
    folder = os.path.join(str(tmp_path), "cells")
    a_set(folder, n=2)
    atoms, _names = periodic.read_crystal_set(folder, pbc="ab")
    assert all(tuple(a.get_pbc()) == (True, True, False) for a in atoms)


# ----------------------------------------------------------------------------- the prediction
def test_predict_applies_a_crystal_model_to_a_folder_of_cells(built, capsys):
    folder, csv, model_path, names, y = built
    main(["predict", model_path, folder, "--activity-file", csv, "--activity-column", "lattice"])
    out = capsys.readouterr().out
    assert "periodic field autocorrelation" in out
    for name in names:
        assert name in out
    assert "R2pred" in out                       # the measured values were read and used


def test_the_predictions_are_written_where_asked(built, tmp_path):
    folder, csv, model_path, names, y = built
    out = os.path.join(str(tmp_path), "pred.csv")
    main(["predict", model_path, folder, "--activity-file", csv, "--activity-column", "lattice",
          "--out", out])
    lines = open(out, encoding="utf-8").read().splitlines()
    assert len(lines) == len(names) + 1
    assert all(name in "".join(lines) for name in names)


def test_a_crystal_model_states_how_its_cells_were_prepared(built, capsys):
    folder, csv, model_path, _names, _y = built
    main(["predict", model_path, folder])
    out = capsys.readouterr().out
    assert "training cells" in out and "atoms per cell" in out
    assert "elements" in out


def test_a_pose_cannot_be_aligned_for_a_crystal(built):
    """A crystal has a lattice, not a pose; the alignment options are molecular."""
    folder, csv, model_path, _names, _y = built
    with pytest.raises(SystemExit) as exc:
        main(["predict", model_path, folder, "--align-variance", "2"])
    assert "no pose to align" in str(exc.value)


# ------------------------------------------------------------------------------- the contract
def test_a_structure_without_a_lattice_is_refused(built):
    folder, csv, model_path, _names, _y = built
    model, _extra = load_model(model_path)
    loose = Atoms(symbols=["Na", "Cl"], positions=[[0, 0, 0], [2.1, 0, 0]])
    crystal = periodic.crystal_fields(rocksalt(), model.fieldset, "fine")
    crystal.cell = np.zeros((3, 3))
    findings = check_crystals(model, [crystal])
    assert any(f.level == "refuse" and f.code == "no-cell" for f in findings), findings


def test_an_element_the_fields_cannot_see_is_reported(built):
    folder, csv, model_path, _names, _y = built
    model, _extra = load_model(model_path)
    other = periodic.crystal_fields(rocksalt(4.2, "Rb", "I"), model.fieldset, "RbI")
    findings = check_crystals(model, [other])
    codes = {f.code for f in findings}
    assert "elements" in codes, findings           # Rb and I share the 'other' field
    assert "new-elements" in codes                 # and the training cells had neither


def test_cells_of_the_training_set_raise_nothing(built):
    folder, csv, model_path, names, _y = built
    model, _extra = load_model(model_path)
    atoms, got = periodic.read_crystal_set(folder)
    table = periodic.crystal_table(atoms, model.fieldset, got)
    findings = check_crystals(model, table.molecules)
    assert not [f for f in findings if f.level in ("refuse", "warn")], findings


def test_the_overlap_kernel_asks_for_one_lattice(tmp_path):
    """The kernel compares two cells point by point, so a cell on another lattice is a warning."""
    from cmfqsar.model import CMFModel

    fs = periodic.inorganic_fieldset(["Na", "Cl"], physchem=["steric"], alpha=2.0)
    cells = [rocksalt(4.2, "Na", "Cl"), rocksalt(4.21, "Na", "Cl"), rocksalt(4.19, "Na", "Cl"),
             rocksalt(4.2, "Na", "Cl")]
    table = periodic.crystal_table(cells, fs, [f"c{k}" for k in range(4)])
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 1e-3}, normalize=True,
                     pbc_overlap=True)
    model.fit(table, np.array([1.0, 1.1, 0.9, 1.05]), cv_folds=0)
    far = periodic.crystal_table([rocksalt(5.4, "Na", "Cl")], fs, ["stretched"])
    findings = check_crystals(model, far.molecules)
    assert any(f.code == "lattice" and f.level == "warn" for f in findings), findings
    near = periodic.crystal_table([rocksalt(4.2, "Na", "Cl")], fs, ["same"])
    assert not [f for f in check_crystals(model, near.molecules) if f.code == "lattice"]


def test_a_site_model_is_not_a_crystal_model(tmp_path):
    """A model fitted on sites of a cell takes sites, and `predict` says so instead of guessing."""
    from cmfqsar.model import CMFModel
    from cmfqsar.sites import site_table, site_targets

    fs = periodic.inorganic_fieldset(["Na", "Cl"], physchem=["steric"], alpha=1.0)
    crystals = [periodic.crystal_fields(rocksalt(4.0 + 0.1 * k), fs, f"c{k}") for k in range(4)]
    targets = site_targets([[float(k + i) for i in range(8)] for k in range(4)], "site value")
    table = site_table(crystals, targets)
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 1e-2}, normalize=True,
                     kernel_source="periodic", ac_d_max=6.0, ac_sigma=1.0)
    model.fit(table, np.asarray(targets.y, float), cv_folds=0)
    path = os.path.join(str(tmp_path), "sites.cmf")
    from cmfqsar.dataio import save_model
    save_model(path, model)
    folder = os.path.join(str(tmp_path), "cells")
    a_set(folder, n=2)
    with pytest.raises(SystemExit) as exc:
        main(["predict", path, folder])
    assert "sites of a cell" in str(exc.value)
