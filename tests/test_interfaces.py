"""Reading a fitted model from another program (plan item MS7).

What has to be true, each checked against the route the program already had:

* the calculator's energy is the model's own prediction, and its forces are the analytic gradient
  of item MS3 with the sign an ASE optimiser needs, so another program driving atoms through the
  ASE interface drives them with the model and nothing else;
* a model whose kernel has no gradient answers energies and says why it has no forces, instead of
  returning zeros;
* an ASE relaxation runs through the calculator and lowers what the calculator calls the energy;
* the barrier table of a structure the model has never seen holds the numbers the direct route
  gives, one row per symmetry-distinct path, with the equivalent paths carrying their
  representative's barrier so that a percolation step reads a complete network;
* the co-field cube of a model on one lattice covers the cell, with the lattice vectors as its
  axes, and a kernel with no co-field at all is refused by name.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ase = pytest.importorskip("ase")
pytest.importorskip("pymatgen.core")

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.gradients import prediction_gradient  # noqa: E402
from cmfqsar.interfaces import (BarrierTable, CMFCalculator, HopBarriers,  # noqa: E402
                                write_cofield_cube)
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.periodic import crystal_fields, inorganic_fieldset  # noqa: E402

SMILES = ["CCCCO", "CCCCN", "CCCOC", "CCCCC", "CCOCC", "CCCCF", "CCCCCl", "CCC(C)O"]
#: the two routes evaluate the same formula, so they agree to round-off
EXACT = 1e-9


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)              # MMFF set-up alters the aromaticity flags in place
    m.SetProp("_Name", smiles)
    return m


def _atoms_of(mol) -> Atoms:
    conf = mol.GetConformer()
    return Atoms(symbols=[a.GetSymbol() for a in mol.GetAtoms()],
                 positions=np.array([list(conf.GetAtomPosition(i))
                                     for i in range(mol.GetNumAtoms())]))


def _fitted(**model_args):
    """A small fitted model, a molecule it has not seen, and that molecule as an ASE object."""
    mols = [_mol(s, seed=i + 1) for i, s in enumerate(SMILES)]
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}")
                                      for i, m in enumerate(mols)])
    y = np.array([4.0 + 0.2 * m.GetNumHeavyAtoms() + 0.05 * i for i, m in enumerate(mols)])
    model = CMFModel(fieldset=fs, **model_args)
    model.fit(table, y, cv_folds=0)
    probe = _mol("CCCCS", seed=11)
    return model, fs, build_fields(probe, fs, "probe"), _atoms_of(probe)


# ------------------------------------------------------------------ the calculator
def test_the_energy_is_the_prediction():
    model, _, rec, atoms = _fitted()
    calc = CMFCalculator(model)
    direct = float(model.predict(AtomTable.from_molecules([rec]))[0])
    assert calc.get_potential_energy(atoms) == pytest.approx(direct, abs=EXACT)
    assert calc.results["prediction"] == pytest.approx(direct, abs=EXACT)
    assert calc.results["free_energy"] == pytest.approx(direct, abs=EXACT)
    # and through ASE's own route, with the calculator attached to the structure
    atoms.calc = calc
    assert atoms.get_potential_energy() == pytest.approx(direct, abs=EXACT)


def test_the_forces_are_the_gradient_with_the_sign_an_optimiser_needs():
    model, _, rec, atoms = _fitted()
    calc = CMFCalculator(model)
    assert calc.has_forces and "forces" in calc.implemented_properties
    g = prediction_gradient(model, rec, wrt="coords")
    f = calc.get_forces(atoms)
    assert f.shape == (rec.n_atoms, 3)
    assert np.abs(f + np.asarray(g.coords, float)).max() < EXACT


def test_the_sign_turns_the_objective_round():
    model, _, rec, atoms = _fitted()
    up, down = CMFCalculator(model), CMFCalculator(model, sign=-1.0)
    assert down.get_potential_energy(atoms) == pytest.approx(-up.get_potential_energy(atoms),
                                                             abs=EXACT)
    assert np.abs(down.get_forces(atoms) + up.get_forces(atoms)).max() < EXACT


@pytest.mark.parametrize("kernel", [
    pytest.param(dict(kernel_source="autocorrelation", ac_d_max=6.0), id="invariant"),
])
def test_the_kernels_item_ms3_differentiates_give_forces(kernel):
    model, _, rec, atoms = _fitted(**kernel)
    calc = CMFCalculator(model)
    assert calc.has_forces
    g = prediction_gradient(model, rec, wrt="coords")
    assert np.abs(calc.get_forces(atoms) + np.asarray(g.coords, float)).max() < 1e-8


def test_a_model_without_a_gradient_answers_energies_and_says_why():
    """A nonlinear layer still has a gradient (item MS3 carries it); the surface kernel has none."""
    model, _, rec, atoms = _fitted()
    model.kernel_source = "surface"              # the fitted model, asked as another kernel
    calc = CMFCalculator(model)
    assert not calc.has_forces
    assert calc.implemented_properties == ["energy", "free_energy"]
    assert "energies only" in repr(calc)
    with pytest.raises(Exception) as err:
        calc.get_forces(atoms)
    assert "energies only" in str(err.value) and "surface" in str(err.value)


def test_what_the_calculator_refuses_to_pretend():
    model, _, _, atoms = _fitted()
    calc = CMFCalculator(model)
    with pytest.raises(NotImplementedError) as err:
        calc.get_potential_energies(atoms)
    assert "no energy per atom" in str(err.value)
    with pytest.raises(NotImplementedError) as err:
        calc.get_stress(atoms)
    assert "with respect to the cell" in str(err.value)
    with pytest.raises(Exception):
        calc.get_property("dipole", atoms)
    with pytest.raises(ValueError):
        CMFCalculator(model).get_potential_energy()        # no structure given


def test_an_unfitted_model_is_not_a_calculator():
    fs = FieldSet.create(physchem=["steric"], alpha=0.5)
    with pytest.raises(ValueError) as err:
        CMFCalculator(CMFModel(fieldset=fs))
    assert "not fitted" in str(err.value)


def test_a_structure_whose_bonds_cannot_be_read_says_so():
    model, _, _, _ = _fitted()
    calc = CMFCalculator(model)
    # two atoms far apart are not a molecule RDKit can bond
    broken = Atoms(symbols=["C", "C"], positions=[[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    with pytest.raises(ValueError) as err:
        calc.get_potential_energy(broken)
    assert "bonds" in str(err.value)


def test_an_ase_relaxation_runs_through_the_calculator():
    """The point of the calculator: another program drives the atoms."""
    from ase.optimize import BFGS
    model, _, rec, atoms = _fitted()
    atoms.calc = CMFCalculator(model)
    before = atoms.get_potential_energy()
    BFGS(atoms, logfile=None).run(fmax=1e-3, steps=3)
    after = atoms.get_potential_energy()
    assert after <= before + 1e-9                     # an optimiser minimises what it is given
    assert not np.allclose(atoms.get_positions(), np.asarray(rec.coords, float))


# ------------------------------------------------------------------ periodic models
RHO, R_MAX = 0.25, 6.0


def _cells(constants=(3.45, 3.55, 3.65)):
    fs = inorganic_fieldset(["Li", "O"])
    cells = [bulk("Li", "bcc", a=a, cubic=True).repeat((2, 2, 2)) for a in constants]
    names = [f"Li-{a:g}" for a in constants]
    return fs, cells, names


def _periodic_model(fs, cells, names):
    table = AtomTable.from_molecules([crystal_fields(c, fs, n) for c, n in zip(cells, names)])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, pbc_overlap=True,
                     pbc_overlap_r_max=6.0, params={"lam": 0.05})
    model.fit(table, np.array([1.0, 1.3, 0.8]), cv_folds=0)
    return model


def test_a_periodic_model_gives_energies_and_refuses_forces():
    fs, cells, names = _cells()
    model = _periodic_model(fs, cells, names)
    calc = CMFCalculator(model)
    assert not calc.has_forces
    direct = float(model.predict(AtomTable.from_molecules(
        [crystal_fields(cells[0], fs, names[0])]))[0])
    assert calc.get_potential_energy(cells[0]) == pytest.approx(direct, abs=EXACT)
    with pytest.raises(Exception) as err:
        calc.get_forces(cells[0])
    assert "image-expanded" in str(err.value)


def test_the_cube_of_a_periodic_model_covers_the_cell(tmp_path):
    fs, cells, names = _cells()
    model = _periodic_model(fs, cells, names)
    path = str(tmp_path / "cell.cube")
    assert write_cofield_cube(model, path, field=0, spacing=1.0,
                              structure=cells[0], name=names[0]) == path
    lines = open(path, encoding="utf-8").read().splitlines()
    assert "one cell of" in lines[0]
    n_atoms = int(lines[2].split()[0])
    assert n_atoms == len(cells[0])
    # the three axis rows are the lattice vectors divided by the point counts, in bohr
    cell = np.asarray(cells[0].get_cell(), float)
    bohr = 1.0 / 0.529177210903
    for k in range(3):
        parts = lines[3 + k].split()
        count = int(parts[0])
        axis = np.array([float(v) for v in parts[1:4]]) / bohr
        assert np.allclose(axis * count, cell[k], atol=1e-6)
    assert len(lines) > 3 + 3 + n_atoms              # the values follow the atoms


def test_a_periodic_cube_needs_the_cell_and_an_invariant_kernel_has_no_cofield(tmp_path):
    fs, cells, names = _cells()
    model = _periodic_model(fs, cells, names)
    with pytest.raises(ValueError) as err:
        write_cofield_cube(model, str(tmp_path / "x.cube"), structure=None)
    assert "needs the structure" in str(err.value)
    table = AtomTable.from_molecules([crystal_fields(c, fs, n) for c, n in zip(cells, names)])
    invariant = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                         ac_sigma=0.5, ac_d_max=6.0, params={"lam": 0.05})
    invariant.fit(table, np.array([1.0, 1.3, 0.8]), cv_folds=0)
    with pytest.raises(ValueError) as err:
        write_cofield_cube(invariant, str(tmp_path / "y.cube"), structure=cells[0])
    assert "no co-field" in str(err.value) and "MS2" in str(err.value)


def test_the_cube_of_a_molecular_model_is_the_one_the_program_already_wrote(tmp_path):
    model, _, _, _ = _fitted()
    path = str(tmp_path / "mol.cube")
    write_cofield_cube(model, path, field=0, spacing=1.0, margin=2.0)
    head = open(path, encoding="utf-8").read().splitlines()[:3]
    assert int(head[2].split()[0]) == model.train.molecules[0].n_atoms


# ------------------------------------------------------------------ a barrier per path
def _hop_model(fs, cells, names):
    from cmfqsar.hops import HopModel, HopSet, framed_environments
    hs = HopSet.build(cells, names=names, mobile="Li", d_min=1.5, d_max=3.2)
    crystals = hs.crystals(fs)
    reps = hs.annotate(crystals, rho=RHO, r_max=R_MAX)
    envs = framed_environments(hs, crystals, rows=reps, rho=RHO, r_max=R_MAX)
    y = np.array([0.3 + 0.1 * (hs.hops[i].length - 2.5) for i in reps])
    return HopModel.build(envs, fs).fit(y, lam=0.01), hs, reps


def test_the_barrier_table_holds_what_the_direct_route_gives():
    from cmfqsar.hops import HopSet, framed_environments
    fs, cells, names = _cells((3.40, 3.50, 3.60, 3.64))
    model, _, _ = _hop_model(fs, cells[:2], names[:2])
    hb = HopBarriers(model, fs, rho=RHO, r_max=R_MAX, mobile="Li", d_min=1.5, d_max=3.2)
    assert "Li paths" in repr(hb)
    table = hb.table(cells[3], name=names[3])           # a cell the model has not seen
    assert isinstance(table, BarrierTable) and len(table) >= 1
    assert table.name == names[3] and table.units == "eV"
    # the same structure through the route the benchmark uses
    hs = HopSet.build([cells[3]], names=[names[3]], mobile="Li", d_min=1.5, d_max=3.2)
    reps = hs.annotate(hs.crystals(fs), rho=RHO, r_max=R_MAX)
    envs = framed_environments(hs, hs.crystals(fs), rows=reps, rho=RHO, r_max=R_MAX)
    direct = np.sort(np.asarray(model.predict(envs), float))
    assert np.abs(np.sort(table.values) - direct).max() < EXACT
    # every path of the structure carries a barrier, not only the distinct ones
    assert len(table.barriers) == len(hs.hops)
    assert sum(int(r["stands_for"]) for r in table.rows) == len(hs.hops)
    for r in table.rows:
        assert r["length"] > 0 and float(r["x"]) == pytest.approx(float(r["x"]))


def test_the_barrier_table_is_written_as_a_file_another_program_reads(tmp_path):
    import csv
    fs, cells, names = _cells()
    model, _, _ = _hop_model(fs, cells[:2], names[:2])
    hb = HopBarriers(model, fs, rho=RHO, r_max=R_MAX, mobile="Li", d_min=1.5, d_max=3.2)
    table = hb.table(cells[2], name=names[2])
    path = str(tmp_path / "barriers.csv")
    assert table.write_table(path) == path
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(table)
    assert rows[0]["structure"] == names[2]
    assert float(rows[0]["barrier"]) == pytest.approx(table.values[0], abs=EXACT)
    for key in ("from_site", "to_site", "length", "x", "y", "z", "stands_for"):
        assert rows[0][key] != ""


def test_the_percolation_thresholds_come_from_the_predicted_network():
    fs, cells, names = _cells()
    model, _, _ = _hop_model(fs, cells[:2], names[:2])
    hb = HopBarriers(model, fs, rho=RHO, r_max=R_MAX, mobile="Li", d_min=1.5, d_max=3.2)
    table = hb.table(cells[2], name=names[2])
    thresholds = table.percolation()
    assert sorted(thresholds) == ["E_1D", "E_2D", "E_3D"]
    # a body-centred lattice of one species percolates in three dimensions at its own barrier
    assert thresholds["E_1D"] == pytest.approx(float(table.values.min()), abs=1e-6)
    assert thresholds["E_3D"] >= thresholds["E_1D"]
    assert hb.percolation(cells[2], names[2]) == thresholds
    bare = BarrierTable(name="x", rows=[], barriers={})
    with pytest.raises(ValueError) as err:
        bare.percolation()
    assert "no path network" in str(err.value)


def test_a_structure_with_no_path_and_a_model_with_no_fit_are_refused():
    fs, cells, names = _cells()
    model, _, _ = _hop_model(fs, cells[:2], names[:2])
    hb = HopBarriers(model, fs, rho=RHO, r_max=R_MAX, mobile="Li", d_min=1.5, d_max=3.2)
    with pytest.raises(ValueError) as err:
        hb.table(bulk("Cu", "fcc", a=3.6, cubic=True), name="copper")
    assert "has no Li" in str(err.value)
    from cmfqsar.hops import HopModel
    unfitted = HopModel(fieldset=fs, train=[], K=np.zeros((0, 0)), diag=np.zeros((1, 0)))
    with pytest.raises(ValueError) as err:
        HopBarriers(unfitted, fs)
    assert "not fitted" in str(err.value)


def test_the_command_line_writes_the_cube_over_a_cell(tmp_path, monkeypatch):
    """`cmf cofield --cell`: the only way in for a model the command line cannot yet fit."""
    from ase.io import write as ase_write

    from cmfqsar.cli import main
    from cmfqsar.dataio import save_model
    fs, cells, names = _cells()
    model = _periodic_model(fs, cells, names)
    path = str(tmp_path / "m.cmf")
    save_model(path, model)
    folder = tmp_path / "cells"
    folder.mkdir()
    ase_write(str(folder / f"{names[0]}.cif"), cells[0])
    monkeypatch.chdir(tmp_path)
    main(["cofield", path, "--cell", str(folder), "--spacing", "1.0", "--out-prefix", "co"])
    written = sorted(p.name for p in tmp_path.glob("co_cell_*.cube"))
    assert written, "no cube was written"
    first = (tmp_path / written[0]).read_text(encoding="utf-8").splitlines()
    assert "one cell of" in first[0]
    assert int(first[2].split()[0]) == len(cells[0])
