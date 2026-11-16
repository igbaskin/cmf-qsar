"""Plan item NT7: site-resolved models on particles.

A table of per-site values (`read_site_values`), a site model fitted on folds that hold out whole
structures (`fit_site_model`), and the command line around them (`build-crystals --site-file`,
then `sites` on a new particle).
"""
import csv

import numpy as np
import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("scipy")
from ase.cluster import Icosahedron, Octahedron  # noqa: E402
from ase.io import write  # noqa: E402

from cmfqsar.cli import main  # noqa: E402
from cmfqsar.dataio import load_model  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.periodic import (coordination_numbers, crystal_table,  # noqa: E402
                              generalized_coordination, inorganic_fieldset)
from cmfqsar.sitemap import is_site_model, site_predictions  # noqa: E402
from cmfqsar.sites import fit_site_model, read_site_values, site_table  # noqa: E402

FIELDS = ("steric", "prop:coordination")


def _open(a, name):
    a.cell[:] = 0.0
    a.pbc = False
    a.info["name"] = name
    return a


def _particles():
    out = [_open(Icosahedron("Cu", 3), "ico55"), _open(Octahedron("Cu", 5, cutoff=1), "octa79"),
           _open(Octahedron("Cu", 5, cutoff=2), "cubo55"), _open(Octahedron("Cu", 4, cutoff=1),
                                                                 "octa38")]
    # two with atoms removed, so that the sites differ more
    for k, base in enumerate((Octahedron("Cu", 5, cutoff=1), Icosahedron("Cu", 3))):
        rng = np.random.default_rng(k)
        surf = np.flatnonzero(coordination_numbers(_open(base, "x")) < 12)
        del base[sorted(rng.choice(surf, 4, replace=False).tolist())]
        out.append(_open(base, f"defect{k}"))
    return out


def _site_rows(parts):
    """Every surface site of every particle, with its generalized coordination as the value."""
    rows = []
    for p in parts:
        cn = coordination_numbers(p)
        g = generalized_coordination(p)
        for s in np.flatnonzero(cn < 12):
            rows.append((p.info["name"], int(s), float(g[s])))
    return rows


def _write_table(path, rows, header=("structure", "site", "gcn")):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def test_a_site_table_is_read_by_name_and_index(tmp_path):
    path = str(tmp_path / "sites.csv")
    _write_table(path, [("a", 0, 1.5), ("a", 3, -2.0), ("b", 1, 7.0), ("zz", 0, 1.0)],
                 header=("structure", "site", "energy"))
    t, unmatched = read_site_values(path, ["a", "b", "c"], [4, 2, 5])
    assert unmatched == ["zz"] and t.name == "energy"
    assert t.molecule.tolist() == [0, 0, 1] and t.atom.tolist() == [0, 3, 1]
    assert t.y.tolist() == [1.5, -2.0, 7.0] and t.n_measured_molecules == 2


def test_a_site_table_chooses_its_column_by_title(tmp_path):
    path = str(tmp_path / "sites.csv")
    with open(path, "w", newline="") as fh:
        fh.write("structure,site,first,second\na,1,1.0,10.0\n")
    t, _ = read_site_values(path, ["a"], [3], column="second")
    assert t.y.tolist() == [10.0]
    with pytest.raises(ValueError, match="no column 'third'"):
        read_site_values(path, ["a"], [3], column="third")


@pytest.mark.parametrize("rows, message", [
    ([("a", 3, 1.0)], "site 3 does not exist"),
    ([("a", 1, 1.0), ("a", 1, 2.0)], "given twice"),
    ([("a", "x", 1.0)], "whole number"),
])
def test_a_wrong_site_table_is_refused(tmp_path, rows, message):
    path = str(tmp_path / "sites.csv")
    _write_table(path, rows)
    with pytest.raises(ValueError, match=message):
        read_site_values(path, ["a"], [3])


def _fit(parts, rows, lams=(1e-3, 1e-2, 0.1)):
    names = [p.info["name"] for p in parts]
    per = [np.full(len(p), np.nan) for p in parts]
    for name, s, v in rows:
        per[names.index(name)][s] = v
    from cmfqsar.atomtargets import AtomTargets
    targets = AtomTargets.from_arrays(per, name="gcn")
    fs = inorganic_fieldset(["Cu"], physchem=FIELDS)
    crystals = list(crystal_table(parts, fs, names).molecules)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_d_max=6.0, ac_sigma=0.5)
    return model, crystals, targets, fit_site_model(model, crystals, targets, lams=lams,
                                                    folds=3, seed=0)


def test_a_site_model_learns_the_generalized_coordination_of_unseen_particles():
    parts = _particles()
    model, crystals, targets, res = _fit(parts, _site_rows(parts))
    assert res["q2"] > 0.9 and res["folds"] == 3 and res["structures"] == len(parts)
    assert model.params["lam"] == res["lam"] and model.stats.q2 == res["q2"]
    assert model.stats.extra["cv_folds_hold_out"] == "whole structures"
    assert is_site_model(model)


def test_a_held_out_particle_is_predicted_without_its_own_sites():
    parts = _particles()
    rows = _site_rows(parts)
    _m, _c, targets, res = _fit(parts, rows, lams=(0.01,))
    # change every value of one particle: its own cross-validated predictions must not move
    shifted = [(n, s, v + 5.0 if n == "octa79" else v) for n, s, v in rows]
    _m, _c, _t, res2 = _fit(parts, shifted, lams=(0.01,))
    mine = targets.molecule == [p.info["name"] for p in parts].index("octa79")
    assert np.allclose(res["y_cv"][mine], res2["y_cv"][mine])
    assert not np.allclose(res["y_cv"][~mine], res2["y_cv"][~mine])


def test_equivalent_sites_of_a_new_particle_get_one_prediction():
    parts = _particles()
    model, _c, _t, _r = _fit(parts, _site_rows(parts))
    new = _open(Icosahedron("Cu", 4), "ico147")
    fs = model.fieldset
    crystal = crystal_table([new], fs, ["ico147"]).molecules[0]
    m = site_predictions(model, crystal)
    v = np.asarray(m.values, float)
    cn = coordination_numbers(new)
    for c in (6, 8, 9):              # vertices, edges, facets of an icosahedron
        assert np.ptp(v[cn == c]) < 1e-8
    # the vertices are the least coordinated, so the smallest predicted value
    assert v[cn == 6].mean() < v[cn == 8].mean() < v[cn == 9].mean()


def test_a_site_model_needs_two_structures():
    parts = _particles()[:1]
    with pytest.raises(ValueError, match="one structure"):
        _fit(parts, _site_rows(parts))


def test_build_crystals_fits_a_site_model_and_sites_predicts_a_new_particle(tmp_path, capsys):
    parts = _particles()
    data = str(tmp_path / "particles.extxyz")
    write(data, parts, format="extxyz")
    table = str(tmp_path / "sites.csv")
    _write_table(table, _site_rows(parts))
    saved = str(tmp_path / "sites.cmf")
    preds = str(tmp_path / "pred.csv")
    main(["build-crystals", data, "--site-file", table, "--fields", ",".join(FIELDS),
          "--elements", "Cu", "--cutoff", "6", "--sigma", "0.5", "--folds", "3",
          "--lams", "0.001,0.01", "--save", saved, "--predictions", preds])
    out = capsys.readouterr().out
    assert "measured sites" in out and "hold out whole structures" in out
    model = load_model(saved)[0]
    assert is_site_model(model) and model.stats.q2 > 0.9
    with open(preds) as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == len(_site_rows(parts)) and got[0]["predicted_cv"]
    new = str(tmp_path / "new.extxyz")
    write(new, [_open(Icosahedron("Cu", 4), "ico147")], format="extxyz")
    main(["sites", saved, new, "--top", "3"])
    out = capsys.readouterr().out
    assert "fitted on" in out and "sites of a cell" in out and "ico147" in out


def test_build_crystals_refuses_a_site_file_with_a_structure_target(tmp_path):
    parts = _particles()[:2]
    data = str(tmp_path / "particles.extxyz")
    write(data, parts, format="extxyz")
    table = str(tmp_path / "sites.csv")
    _write_table(table, _site_rows(parts))
    with pytest.raises(SystemExit, match="--test-fraction"):
        main(["build-crystals", data, "--site-file", table, "--test-fraction", "0.2"])


def test_the_contract_accepts_particles_and_tubes_and_warns_of_another_periodicity():
    from ase.build import bulk, nanotube

    from cmfqsar.contract import check_crystals
    parts = _particles()
    model, _c, _t, _r = _fit(parts, _site_rows(parts))
    fs = model.fieldset

    def codes(atoms, name):
        c = crystal_table([atoms], fs, [name]).molecules[0]
        return {(f.level, f.code) for f in check_crystals(model, [c])}

    particle = codes(_open(Icosahedron("Cu", 4), "ico147"), "ico147")
    assert not any(level == "refuse" for level, _ in particle)
    assert ("warn", "periodicity") not in particle
    tube = nanotube(4, 4, length=2, symbol="Cu")        # a lattice vector along the axis only
    tube.pbc = (False, False, True)
    got = codes(tube, "tube")
    assert ("refuse", "no-cell") not in got and ("warn", "periodicity") in got
    crystal = codes(bulk("Cu", "fcc", a=3.6), "bulk")
    assert ("warn", "periodicity") in crystal and ("refuse", "no-cell") not in crystal
    c = crystal_table([bulk("Cu", "fcc", a=3.6)], fs, ["broken"]).molecules[0]
    c.cell = np.array(c.cell, float)
    c.cell[2] = 0.0                                     # repeats along c with no vector there
    assert ("refuse", "no-cell") in {(f.level, f.code) for f in check_crystals(model, [c])}
