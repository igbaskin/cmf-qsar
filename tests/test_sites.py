"""A site of a periodic cell as the sample: the split of the autocorrelation (M5).

What has to be true, each checked against something independent:

* the split is exact: the mean of the site profiles is the descriptor of the cell, for a crystal,
  a layer and a chain, with and without the uniform background;
* a site record is a sample: the periodic autocorrelation of a record with a centre is that
  site's own profile, and a model fits a table of such records, predicts and survives a file;
* a site profile carries what a site is and nothing else: two sites related by a symmetry of the
  cell get the same profile, a supercell repeats the profiles of the primitive cell, an origin
  shift and a rotation of the frame change nothing, the vacuum of a layer changes nothing, and a
  change beyond the cutoff changes nothing while the same change inside it does;
* the targets are the data model of plan item LP1: the folds hold out whole structures;
* the contributions of a whole-cell model add up to its prediction, and equivalent sites get
  equal contributions.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ase = pytest.importorskip("ase")
pytest.importorskip("pymatgen.core")

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.fields import AtomTable  # noqa: E402
from cmfqsar.periodic import (PeriodicSpec, crystal_fields, inorganic_fieldset,  # noqa: E402
                              periodic_autocorrelation, periodic_channel_labels)
from cmfqsar.sites import (SHELLS, count_labels, elements_of, one_site_targets,  # noqa: E402
                           shell_count_features, shell_counts, site_autocorrelation,
                           site_contribution_table, site_contributions, site_features,
                           site_fields, site_kernel, site_names, site_table, site_targets,
                           site_top_channels, site_width)

SPEC = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5)


@pytest.fixture(scope="module")
def fs():
    return inorganic_fieldset(["Na", "Cl", "Mg", "O", "C", "H"])


@pytest.fixture(scope="module")
def nacl(fs):
    return crystal_fields(bulk("NaCl", "rocksalt", a=5.64), fs, "NaCl")


def graphene(vacuum: float = 20.0, n: int = 3):
    sheet = Atoms("C2", positions=[[0.0, 0.0, vacuum / 2], [1.23, 0.71, vacuum / 2]],
                  cell=[[2.46, 0.0, 0.0], [1.23, 2.13, 0.0], [0.0, 0.0, vacuum]],
                  pbc=[True, True, False])
    return sheet.repeat((n, n, 1))


def chain(n: int = 4):
    one = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                cell=[[2.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
                pbc=[True, False, False])
    return one.repeat((n, 1, 1))


# --------------------------------------------------------------------------- the split is exact
@pytest.mark.parametrize("background", [True, False])
def test_the_mean_of_the_site_profiles_is_the_cell_descriptor(fs, background):
    spec = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5, background=background)
    cases = [bulk("NaCl", "rocksalt", a=5.64), bulk("MgO", "rocksalt", a=4.21).repeat((2, 1, 1)),
             graphene(), chain()]
    for atoms in cases:
        c = crystal_fields(atoms, fs, atoms.get_chemical_formula())
        rows = site_autocorrelation(c, spec)
        cell = periodic_autocorrelation(c, spec)
        assert rows.shape == (c.n_atoms, site_width(spec, fs.n_fields))
        assert np.allclose(rows.mean(axis=0), cell, rtol=0, atol=1e-10 * max(np.abs(cell).max(), 1))


def test_a_disordered_cell_splits_as_exactly_as_an_ordered_one(fs):
    """The second moments of a shared site (plan item M2) belong to that site, so the sum still
    reproduces the cell -- here over the expected number of atoms rather than the rows."""
    atoms = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 1, 1))
    species = [{"Na": 0.5, "Mg": 0.5} if s == "Na" else {s: 1.0}
               for s in atoms.get_chemical_symbols()]
    atoms.info["site_species"] = species
    atoms.set_array("occupancy", np.array([sum(d.values()) for d in species]))
    neutral = inorganic_fieldset(["Na", "Mg", "Cl"],
                                 physchem=("steric", "prop:electronegativity"))
    c = crystal_fields(atoms, neutral, "mixed")
    assert c.second_moments is not None
    rows = site_autocorrelation(c, SPEC)
    cell = periodic_autocorrelation(c, SPEC)
    n_norm = float(np.sum(c.occupancy))
    assert np.allclose(rows.sum(axis=0) / n_norm, cell, atol=1e-10 * max(np.abs(cell).max(), 1))


# --------------------------------------------------------------------------- a site is a sample
def test_a_site_record_is_the_row_of_its_site(nacl):
    rows = site_autocorrelation(nacl, SPEC)
    for s in range(nacl.n_atoms):
        rec = site_fields(nacl, s)
        assert rec.centre == s and rec.name.endswith(f"#{s}")
        assert np.allclose(periodic_autocorrelation(rec, SPEC), rows[s], atol=0)
    with pytest.raises(IndexError, match="does not exist"):
        site_fields(nacl, nacl.n_atoms)


def test_the_feature_matrix_lines_up_with_the_targets(fs):
    crystals = [crystal_fields(bulk("NaCl", "rocksalt", a=a), fs, f"NaCl-{a}")
                for a in (5.4, 5.64, 5.9)]
    per = [np.array([1.0, np.nan]), np.array([np.nan, 2.0]), np.array([3.0, 4.0])]
    t = site_targets(per, name="per-site")
    assert len(t) == 4 and t.n_mol == 3
    X = site_features(crystals, t, SPEC)
    assert X.shape == (4, site_width(SPEC, fs.n_fields))
    for r in range(len(t)):
        k, s = int(t.molecule[r]), int(t.atom[r])
        assert np.allclose(X[r], site_autocorrelation(crystals[k], SPEC)[s], atol=0)
    names = site_names(crystals, t)
    assert names[0] == "NaCl-5.4#0 Na" and names[-1] == "NaCl-5.9#1 Cl"
    K = site_kernel(X)
    assert K.shape == (4, 4) and np.allclose(np.diag(K), 1.0, atol=1e-10)


def test_the_folds_hold_out_whole_structures(fs):
    per = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]),
           np.array([7.0, 8.0])]
    t = site_targets(per, name="per-site")
    folds = t.folds(k=4, seed=0)
    assert len(folds) == 4
    for te in folds:
        held = set(int(t.molecule[r]) for r in te)
        train = set(int(t.molecule[r]) for r in range(len(t)) if r not in set(te.tolist()))
        assert not (held & train)
    assert sum(len(f) for f in folds) == len(t)


def test_one_measured_site_per_cell_builds_the_same_index():
    t = one_site_targets([1, 0], [0.5, -0.5], [4, 3], name="binding")
    assert len(t) == 2 and t.n_mol == 2
    assert list(t.molecule) == [0, 1] and list(t.atom) == [1, 0]
    assert np.allclose(t.y, [0.5, -0.5])
    assert len(t.per_molecule[0]) == 4 and np.isnan(t.per_molecule[0][0])
    with pytest.raises(ValueError, match="cell sizes"):
        one_site_targets([0], [1.0, 2.0], [3], name="x")


def test_a_model_on_the_sites_fits_predicts_and_round_trips(tmp_path, fs):
    from cmfqsar.dataio import load_model, save_model
    from cmfqsar.model import CMFModel
    crystals = [crystal_fields(bulk("NaCl", "rocksalt", a=a), fs, f"NaCl-{a}")
                for a in (5.2, 5.4, 5.64, 5.9, 6.1, 6.3)]
    per = [np.array([float(k), float(k) + 0.5]) for k in range(len(crystals))]
    t = site_targets(per, name="per-site")
    table = site_table(crystals, t)
    assert table.n_mol == len(t)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_sigma=0.5, ac_d_max=6.0, params={"lam": 0.01})
    model.fit(table, t.y, cv_folds=3)
    assert np.isfinite(model.stats.q2)
    here = model.predict(table)
    assert here.shape == (len(t),)
    path = str(tmp_path / "sites.cmf")
    save_model(path, model)
    back = load_model(path)[0]
    assert np.allclose(back.predict(table), here, atol=1e-9)


# ------------------------------------------------- the profile carries the site and nothing else
def test_equivalent_sites_get_the_same_profile(fs):
    """Every sodium of a rocksalt supercell stands in the same surroundings."""
    c = crystal_fields(bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2)), fs, "NaCl")
    rows = site_autocorrelation(c, SPEC)
    na = [k for k, s in enumerate(c.symbols) if s == "Na"]
    cl = [k for k, s in enumerate(c.symbols) if s == "Cl"]
    assert len(na) == 8 and len(cl) == 8
    for group in (na, cl):
        spread = np.abs(rows[group] - rows[group[0]]).max()
        assert spread < 1e-9, spread
    # and the two kinds of site are not the same profile
    assert np.abs(rows[na[0]] - rows[cl[0]]).max() > 1e-3


def _worst_match(rows: np.ndarray, base: np.ndarray) -> float:
    """How far the least similar row of `rows` is from its nearest row of `base`.

    The sites of a cell come in whatever order the file gives, so two descriptions of one
    structure are compared row against nearest row and not position by position.
    """
    return max(float(min(np.abs(base - r).max(axis=1))) for r in rows)


def test_a_supercell_an_origin_shift_and_a_rotation_change_nothing(fs):
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    base = site_autocorrelation(crystal_fields(atoms, fs, "x"), SPEC)
    sup = site_autocorrelation(crystal_fields(atoms.repeat((2, 1, 1)), fs, "sup"), SPEC)
    # the supercell has every site of the primitive cell twice, with the same surroundings
    assert _worst_match(sup, base) < 1e-9
    shifted = atoms.copy()
    shifted.positions += np.array([1.3, -2.1, 0.7])
    moved = site_autocorrelation(crystal_fields(shifted, fs, "shift"), SPEC)
    assert _worst_match(moved, base) < 1e-9
    turned = atoms.copy()
    turned.rotate(37.0, "xyz", rotate_cell=True)
    spun = site_autocorrelation(crystal_fields(turned, fs, "turn"), SPEC)
    assert _worst_match(spun, base) < 1e-7


def test_the_vacuum_of_a_layer_changes_nothing(fs):
    thin = crystal_fields(graphene(vacuum=8.0), fs, "thin")
    thick = crystal_fields(graphene(vacuum=30.0), fs, "thick")
    a = site_autocorrelation(thin, SPEC)
    b = site_autocorrelation(thick, SPEC)
    assert np.allclose(a, b, atol=1e-9 * max(np.abs(a).max(), 1.0))


def test_the_pair_part_of_the_profile_is_local():
    """A site feels what is inside the cutoff.  The same change nearer than the cutoff moves it.

    The uniform background is a property of the whole cell (the mean field times the cell
    measure), so locality is a statement about the pair part: the spec here keeps the pairs and
    removes nothing.  The field set carries no charges either, because the oxidation-state
    charges are balanced over the cell and would change every site at once.
    """
    spec = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5, background=False)
    neutral = inorganic_fieldset(["Mg", "O", "Na"],
                                 physchem=("steric", "prop:electronegativity"))
    atoms = bulk("MgO", "rocksalt", a=4.21).repeat((4, 4, 4))
    rows = site_autocorrelation(crystal_fields(atoms, neutral, "host"), spec)
    d = atoms.get_distances(0, range(len(atoms)), mic=True)
    far = int(np.argmax(d))
    near = int(np.argsort(d)[1])
    assert d[far] > spec.r_max + 1.0 and atoms.get_chemical_symbols()[far] == "Mg"
    for k, expect_change in ((far, False), (near, True)):
        other = atoms.copy()
        sym = other.get_chemical_symbols()
        sym[k] = "Na"
        other.set_chemical_symbols(sym)
        changed = site_autocorrelation(crystal_fields(other, neutral, "changed"), spec)
        moved = np.abs(changed[0] - rows[0]).max() / max(np.abs(rows[0]).max(), 1e-30)
        if expect_change:
            assert moved > 1e-3, moved
        else:
            assert moved < 1e-12, moved


def test_the_height_channels_of_a_layer_are_refused(fs):
    c = crystal_fields(graphene(), fs, "sheet")
    with pytest.raises(ValueError, match="no site split"):
        site_autocorrelation(c, PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5, z_max=2.0))


# --------------------------------------------------------------------------- the count reference
def test_the_counts_around_a_site_are_the_neighbour_shells(fs):
    c = crystal_fields(bulk("NaCl", "rocksalt", a=5.64), fs, "NaCl")
    elements = elements_of([c])
    assert elements == ["Cl", "Na"]
    row = shell_counts(c, 0, elements, SHELLS).reshape(1 + len(SHELLS), len(elements))
    assert row[0, elements.index("Na")] == 1.0 and row[0, elements.index("Cl")] == 0.0
    # rocksalt: six chlorines at a/2 = 2.82 A (the 2.0 to 3.0 shell), then twelve sodiums at
    # 3.99 A (the 3.0 to 4.5 shell), and nothing closer than 2.0 A
    assert row[1 + 0].sum() == 0.0
    assert row[1 + 1, elements.index("Cl")] == 6.0
    assert row[1 + 2, elements.index("Na")] == 12.0
    t = site_targets([np.array([1.0, 2.0])], name="x")
    X = shell_count_features([c], t, elements)
    assert X.shape == (2, (1 + len(SHELLS)) * len(elements))
    labels = count_labels(elements, SHELLS)
    assert len(labels) == X.shape[1] and labels[0] == "the site is Cl"
    assert "between 2.0 and 3.0 A" in labels[(1 + 1) * len(elements)]


# ------------------------------------------------------- reading a whole-cell model site by site
def _cell_model(fs):
    from cmfqsar.model import CMFModel
    rows = [("NaCl", 5.64, 1.0), ("NaCl", 5.9, 1.3), ("NaCl", 5.4, 0.8), ("MgO", 4.21, 2.0),
            ("MgO", 4.4, 2.3), ("MgO", 4.0, 1.8)]
    crys = [crystal_fields(bulk(s, "rocksalt", a=a), fs, f"{s}-{a}") for s, a, _ in rows]
    y = np.array([v for _, _, v in rows])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_sigma=0.5, ac_d_max=6.0, params={"lam": 0.05})
    model.fit(AtomTable.from_molecules(crys), y, cv_folds=0)
    return model, crys


def test_the_site_contributions_add_up_to_the_prediction(fs):
    model, crys = _cell_model(fs)
    predicted = model.predict(AtomTable.from_molecules(crys))
    for k, c in enumerate(crys):
        v = site_contributions(model, c)
        assert v.shape == (c.n_atoms,)
        assert abs(float(v.sum()) + model.bias - predicted[k]) < 1e-9
    # equivalent sites carry equal amounts
    big = crystal_fields(bulk("NaCl", "rocksalt", a=5.64).repeat((2, 1, 1)), fs, "NaCl-sup")
    v = site_contributions(model, big)
    na = [k for k, s in enumerate(big.symbols) if s == "Na"]
    assert np.abs(v[na] - v[na[0]]).max() < 1e-12
    table = site_contribution_table(model, crys[0], n=2)
    assert len(table) == 2 and all(" " in lab for lab, _ in table)


def test_the_contributions_of_a_disordered_cell_add_up_too(fs):
    """A disordered cell divides by the expected number of atoms, not by the rows (M2)."""
    from cmfqsar.model import CMFModel
    neutral = inorganic_fieldset(["Na", "Mg", "Cl"],
                                 physchem=("steric", "prop:electronegativity"))
    crys, y = [], []
    for k, share in enumerate((0.2, 0.4, 0.6, 0.8)):
        atoms = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 1, 1))
        species = [{"Na": 1.0 - share, "Mg": share} if s == "Na" else {s: 1.0}
                   for s in atoms.get_chemical_symbols()]
        atoms.info["site_species"] = species
        atoms.set_array("occupancy", np.array([sum(d.values()) for d in species]))
        crys.append(crystal_fields(atoms, neutral, f"mixed-{share}"))
        y.append(float(share))
    model = CMFModel(fieldset=neutral, method="krr", normalize=True, kernel_source="periodic",
                     ac_sigma=0.5, ac_d_max=6.0, params={"lam": 0.05})
    model.fit(AtomTable.from_molecules(crys), np.array(y), cv_folds=0)
    predicted = model.predict(AtomTable.from_molecules(crys))
    for k, c in enumerate(crys):
        v = site_contributions(model, c)
        assert abs(float(v.sum()) + model.bias - predicted[k]) < 1e-9


def test_the_channels_of_one_site_add_up_to_its_contribution(fs):
    model, crys = _cell_model(fs)
    labels = periodic_channel_labels(fs, model._ac_spec())
    c = crys[0]
    v = site_contributions(model, c)
    n = len(labels)
    parts = site_top_channels(model, c, 0, labels, n=n)
    assert len(parts) == n
    assert abs(sum(amount for _, amount in parts) - v[0]) < 1e-9
    assert all(lab in labels for lab, _ in parts)


def test_a_model_that_is_not_linear_has_no_site_contributions(fs):
    from cmfqsar.model import CMFModel
    crys = [crystal_fields(bulk("NaCl", "rocksalt", a=a), fs, f"NaCl-{a}")
            for a in (5.4, 5.64, 5.9, 6.1)]
    y = np.array([1.0, 1.3, 0.8, 0.6])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     kernel_layer="gauss", params={"lam": 0.05, "gamma": 1.0})
    model.fit(AtomTable.from_molecules(crys), y, cv_folds=0)
    with pytest.raises(ValueError, match="linear"):
        site_contributions(model, crys[0])


def test_the_same_cell_is_split_once(fs, nacl):
    """The split of a cell gives every site at once, so a table of its sites costs one split."""
    import cmfqsar.sites as S
    rows = site_autocorrelation(nacl, SPEC)
    calls = {"n": 0}
    real = S._neighbour_pairs

    def counted(crystal, r_max):
        calls["n"] += 1
        return real(crystal, r_max)

    S._LAST = None                      # the split above is remembered; start from nothing
    S._neighbour_pairs = counted
    try:
        recs = [site_fields(nacl, s) for s in range(nacl.n_atoms)]
        got = np.vstack([periodic_autocorrelation(r, SPEC) for r in recs])
    finally:
        S._neighbour_pairs = real
    assert nacl.n_atoms > 1 and calls["n"] == 1, calls["n"]
    assert np.allclose(got, rows, atol=0)
