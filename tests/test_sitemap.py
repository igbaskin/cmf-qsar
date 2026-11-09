"""A spatially resolved prediction: one number per atom, site or region (plan item MS2).

What has to be true, each checked against something the map does not compute itself:

* the split is exact. The places and the fitted constant add up to the prediction the model
  itself makes, for a molecule under the overlap kernel, for a molecule under the
  rotation-invariant kernel, and for a periodic cell;
* grouping places into regions keeps the total, whatever the grouping, and a place that would
  belong to two regions is refused rather than counted twice;
* the shells around a place are measured through the images of a periodic cell, so a neighbour
  across the cell boundary lands in the shell it belongs to;
* a model whose kernel has no split is refused by name, with the reason;
* a map of predictions per site is not a split and says so: it refuses the sum check, and a model
  fitted on whole structures refuses to give one;
* the written table holds what the map holds: its value column of one structure adds up to that
  structure's prediction.
"""
import json
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.sitemap import (CONSTANT_LABEL, CONTRIBUTION, PREDICTION, SiteMap,  # noqa: E402
                             by_element, cell_map, is_site_model, region_map, shells_around,
                             site_map, site_maps, site_predictions, splits_prediction,
                             write_maps)

SMILES = ["CCCCO", "CCCCN", "CCCOC", "CCCCC", "CCOCC", "CCCCF", "CCCCCl", "CCC(C)O"]
#: how exactly the split has to add up: the two sides are one matrix product apart
EXACT = 1e-9


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)              # MMFF set-up alters the aromaticity flags in place
    m.SetProp("_Name", smiles)
    return m


def _fitted(**model_args):
    """A small fitted model and a molecule that is not in its training set."""
    mols = [_mol(s, seed=i + 1) for i, s in enumerate(SMILES)]
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}")
                                      for i, m in enumerate(mols)])
    y = np.array([4.0 + 0.2 * m.GetNumHeavyAtoms() + 0.05 * i for i, m in enumerate(mols)])
    model = CMFModel(fieldset=fs, **model_args)
    model.fit(table, y, cv_folds=0)
    return model, fs, build_fields(_mol("CCCCS", seed=11), fs, "probe")


#: the two kernels whose channels are sums over atom pairs, which is what makes the split exact
KERNELS = [pytest.param({}, id="overlap"),
           pytest.param(dict(kernel_source="autocorrelation", ac_d_max=6.0), id="invariant")]


# ------------------------------------------------------------------ the split adds up
@pytest.mark.parametrize("kernel", KERNELS)
def test_the_places_and_the_constant_add_up_to_the_prediction(kernel):
    model, _, probe = _fitted(**kernel)
    m = site_map(model, probe, units="log units")
    assert m.quantity == CONTRIBUTION and m.kind == "atom"
    assert len(m) == probe.n_atoms
    predicted = float(model.predict(AtomTable.from_molecules([probe]))[0])
    assert m.prediction == pytest.approx(predicted, abs=EXACT)
    assert m.total == pytest.approx(predicted, abs=EXACT)
    assert abs(m.residual) < EXACT
    assert m.check() < EXACT                      # the check is what the gate of MS2 asks for
    assert m.coords is not None and m.coords.shape == (probe.n_atoms, 3)
    assert m.labels[0].endswith(probe.symbols[0])


def test_a_map_whose_prediction_was_changed_is_refused():
    model, _, probe = _fitted()
    m = site_map(model, probe)
    m.prediction = float(m.prediction) + 0.5
    with pytest.raises(ValueError) as err:
        m.check()
    assert "differ by" in str(err.value)


def test_the_shares_are_sizes_and_add_up_to_one():
    model, _, probe = _fitted()
    m = site_map(model, probe)
    s = m.share()
    assert s.min() >= 0.0 and float(s.sum()) == pytest.approx(1.0)
    # the largest share belongs to the place the top list names first
    assert m.labels[int(np.argmax(s))] == m.top(1)[0][0]


def test_several_structures_at_once():
    model, fs, _ = _fitted()
    mols = [build_fields(_mol(s, seed=20 + i), fs, f"p{i}") for i, s in enumerate(SMILES[:3])]
    maps = site_maps(model, mols)
    assert [m.name for m in maps] == ["p0", "p1", "p2"]
    for m in maps:
        assert abs(m.residual) < EXACT


# ------------------------------------------------------------------ regions
def test_grouping_by_element_keeps_the_total():
    model, _, probe = _fitted()
    m = site_map(model, probe)
    groups = by_element(probe)
    r = region_map(m, groups)
    assert r.kind == "region" and len(r) == len(groups)
    assert sorted(r.labels) == sorted(groups)
    assert r.total == pytest.approx(m.total, abs=EXACT)
    assert r.check() < EXACT
    # a region's number is the sum of its places, and its position their mean position
    k = r.labels.index("S")
    inside = groups["S"]
    assert r.values[k] == pytest.approx(float(m.values[inside].sum()), abs=1e-12)
    assert r.coords[k] == pytest.approx(m.coords[inside].mean(axis=0), abs=1e-9)


def test_a_place_in_two_regions_is_refused():
    model, _, probe = _fitted()
    m = site_map(model, probe)
    with pytest.raises(ValueError) as err:
        region_map(m, {"first": [0, 1], "second": [1, 2]})
    assert "belongs to one region at most" in str(err.value)


def test_the_places_left_over_become_one_region_or_are_dropped_with_a_note():
    model, _, probe = _fitted()
    m = site_map(model, probe)
    kept = region_map(m, {"the first two": [0, 1]})
    assert len(kept) == 2 and kept.check() < EXACT
    dropped = region_map(m, {"the first two": [0, 1]}, rest=None)
    assert len(dropped) == 1 and dropped.notes
    with pytest.raises(ValueError):
        dropped.check()


def test_a_region_map_cannot_be_grouped_again():
    model, _, probe = _fitted()
    r = region_map(site_map(model, probe), by_element(probe))
    with pytest.raises(ValueError) as err:
        region_map(r, {"all": [0]})
    assert "already grouped" in str(err.value)


def test_the_shells_around_a_place_start_with_that_place():
    model, _, probe = _fitted()
    m = site_map(model, probe)
    shells = shells_around(probe, 0, (2.0, 3.0))
    first = list(shells)[0]
    assert shells[first] == [0] and first.startswith("the centre")
    assert region_map(m, shells).check() < EXACT
    with pytest.raises(IndexError):
        shells_around(probe, probe.n_atoms, (2.0,))


def test_an_ensemble_is_refused():
    mols = [_mol(s, seed=i + 1) for i, s in enumerate(SMILES)]
    fs = FieldSet.create(physchem=["steric"], alpha=0.5)
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)])
    model = CMFModel(fieldset=fs)
    model.fit(table, np.arange(len(mols), dtype=float), cv_folds=0)
    probe = build_fields(_mol("CCCCS", seed=5), fs, "probe")
    probe.n_conformers = 2                      # claim an ensemble without building one
    with pytest.raises(ValueError) as err:
        site_map(model, probe)
    assert "ensemble" in str(err.value)


# ------------------------------------------------------------------ what has no split
@pytest.mark.parametrize("layer", ["gauss", "arccos1", "arccos2"])
def test_a_nonlinear_layer_has_no_split(layer):
    model, _, probe = _fitted(kernel_layer=layer)
    assert not splits_prediction(model)
    with pytest.raises(ValueError) as err:
        site_map(model, probe)
    assert "linear layer" in str(err.value) and "Shapley" in str(err.value)


@pytest.mark.parametrize("source,word", [("surface", "height"),
                                         ("reciprocal", "structure factors"),
                                         ("threebody", "triples")])
def test_the_kernels_without_a_split_are_refused_by_name(source, word):
    model, _, probe = _fitted()
    model.kernel_source = source                 # the fitted model, asked as another kernel
    assert not splits_prediction(model)
    with pytest.raises(ValueError) as err:
        site_map(model, probe)
    assert source in str(err.value) and word in str(err.value)


def test_the_periodic_overlap_kernel_is_refused():
    model, _, probe = _fitted()
    model.pbc_overlap = True
    with pytest.raises(ValueError) as err:
        site_map(model, probe)
    assert "point by point" in str(err.value)


# ------------------------------------------------------------------ the written table
def test_the_table_holds_what_the_map_holds(tmp_path):
    import csv
    model, _, probe = _fitted()
    m = site_map(model, probe)
    path = str(tmp_path / "map.csv")
    assert m.write_table(path) == path
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(m) + 1                           # the places and the constant
    assert rows[-1]["label"] == CONSTANT_LABEL and rows[-1]["place"] == "-1"
    total = sum(float(r["value"]) for r in rows)
    assert total == pytest.approx(m.prediction, abs=EXACT)   # a reader that sums gets it back
    assert all(r["structure"] == m.name for r in rows)
    assert all(r["quantity"] == CONTRIBUTION for r in rows)
    # without the constant the places alone are written
    m.write_table(path, include_constant=False)
    with open(path, encoding="utf-8", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == len(m)


def test_several_maps_in_one_table(tmp_path):
    import csv
    model, fs, _ = _fitted()
    mols = [build_fields(_mol(s, seed=30 + i), fs, f"q{i}") for i, s in enumerate(SMILES[:2])]
    maps = site_maps(model, mols)
    path = str(tmp_path / "all.csv")
    write_maps(maps, path)
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for m in maps:
        mine = [r for r in rows if r["structure"] == m.name]
        assert sum(float(r["value"]) for r in mine) == pytest.approx(m.prediction, abs=EXACT)


def test_the_json_file_carries_every_number(tmp_path):
    model, _, probe = _fitted()
    m = site_map(model, probe, units="log units")
    path = str(tmp_path / "map.json")
    m.write_json(path)
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    assert d["units"] == "log units" and d["quantity"] == CONTRIBUTION
    assert d["values"] == pytest.approx(list(m.values))
    assert d["prediction"] == pytest.approx(m.prediction)
    assert len(d["coords"]) == len(m)


def test_a_map_checks_its_own_shape():
    with pytest.raises(ValueError):
        SiteMap(name="x", labels=["a", "b"], values=np.array([1.0]))
    with pytest.raises(ValueError):
        SiteMap(name="x", labels=["a"], values=np.array([1.0]), quantity="something else")


# ------------------------------------------------------------------ periodic cells
ase = pytest.importorskip("ase")
pytest.importorskip("pymatgen.core")

from ase.build import bulk  # noqa: E402

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.periodic import crystal_fields, inorganic_fieldset  # noqa: E402
from cmfqsar.sites import one_site_targets, site_table  # noqa: E402


def _cells():
    fs = inorganic_fieldset(["Na", "Cl", "Mg", "O"])
    rows = [("NaCl", 5.64, 1.0), ("NaCl", 5.9, 1.3), ("NaCl", 5.4, 0.8), ("MgO", 4.21, 2.0),
            ("MgO", 4.4, 2.3), ("MgO", 4.0, 1.8)]
    crys = [crystal_fields(bulk(s, "rocksalt", a=a), fs, f"{s}-{a}") for s, a, _ in rows]
    return fs, crys, np.array([v for _, _, v in rows])


def _cell_model():
    fs, crys, y = _cells()
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_sigma=0.5, ac_d_max=6.0, params={"lam": 0.05})
    model.fit(AtomTable.from_molecules(crys), y, cv_folds=0)
    return model, crys


def test_a_cell_splits_over_its_sites():
    model, crys = _cell_model()
    predicted = model.predict(AtomTable.from_molecules(crys))
    for k, c in enumerate(crys):
        m = cell_map(model, c, units="eV")
        assert m.kind == "site" and len(m) == c.n_atoms
        assert m.prediction == pytest.approx(float(predicted[k]), abs=EXACT)
        assert m.check() < EXACT
    # the same through the common entry, which sees the lattice and chooses the site split
    m = site_map(model, crys[0])
    assert m.kind == "site" and m.check() < EXACT


def test_the_shells_of_a_cell_are_measured_through_the_images():
    model, crys = _cell_model()
    c = crys[0]
    m = site_map(model, c)
    shells = shells_around(c, 0, (3.0, 4.5))
    assert region_map(m, shells).check() < EXACT
    # every site of this cell is a neighbour of site 0 through some image, so nothing is left
    # in the outermost shell that a cell measured without images would have filled
    covered = sum(len(v) for v in shells.values())
    assert covered == c.n_atoms


def test_a_site_model_gives_one_prediction_per_site():
    fs, crys, y = _cells()
    sites = [0] * len(crys)
    index = one_site_targets(sites, y, [c.n_atoms for c in crys], name="energy")
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_sigma=0.5, ac_d_max=6.0, params={"lam": 0.05})
    model.fit(site_table(crys, index), y, cv_folds=0)
    assert is_site_model(model)
    m = site_predictions(model, crys[0], units="eV")
    assert m.quantity == PREDICTION and len(m) == crys[0].n_atoms
    assert np.all(np.isfinite(m.values))
    # a prediction map is not a split: it has no total to check
    with pytest.raises(ValueError) as err:
        m.check()
    assert "no total to check" in str(err.value)
    # the prediction at the measured site is the one the model makes for that site
    one = float(model.predict(site_table([crys[0]], AtomTargets.from_arrays(
        [np.where(np.arange(crys[0].n_atoms) == 0, 0.0, np.nan)], name="energy")))[0])
    assert m.values[0] == pytest.approx(one, abs=EXACT)
    # and only some of the sites can be asked for
    part = site_predictions(model, crys[0], sites=[1])
    assert len(part) == 1 and part.values[0] == pytest.approx(m.values[1], abs=EXACT)
    with pytest.raises(IndexError):
        site_predictions(model, crys[0], sites=[crys[0].n_atoms])


def test_a_whole_structure_model_has_no_answer_for_one_site():
    model, crys = _cell_model()
    assert not is_site_model(model)
    with pytest.raises(ValueError) as err:
        site_predictions(model, crys[0])
    assert "no answer for a single site" in str(err.value)


def test_one_site_of_a_cell_is_not_a_structure_to_split():
    """A site record carries that site's own profile, so there is nothing to share out."""
    from cmfqsar.sites import site_fields
    model, crys = _cell_model()
    one = site_fields(crys[0], 1)
    with pytest.raises(ValueError) as err:
        site_map(model, one)
    assert "one site of a cell" in str(err.value)
