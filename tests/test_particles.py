"""Plan item NT2: particles of thousands of atoms.

A structure with an open direction and no lattice vector along it takes a k-d tree neighbour list
instead of ASE's, whose bins need a cell; the background term of a large chain or layer is summed
in blocks of rows; and the model refuses the overlap kernel for a structure above an atom count.
"""
import os

import numpy as np
import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("scipy")
from ase.build import bulk, graphene, nanotube  # noqa: E402
from ase.cluster import Icosahedron  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402

import cmfqsar.periodic as P  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

PHYSCHEM = ("steric", "prop:electronegativity")
SPEC = P.PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5, taper=2.0)


def _cluster(k=4, el="Cu", cell=False):
    a = Icosahedron(el, k)
    if cell:
        a.center(vacuum=6.0)
    else:
        a.cell[:] = 0.0
    a.pbc = False
    return a


def _record(atoms, name="s"):
    fs = P.inorganic_fieldset(sorted(set(atoms.get_chemical_symbols())), physchem=PHYSCHEM)
    return P.crystal_table([atoms], fs, [name]).molecules[0], fs


def _ase_pairs(c, r, what="ijd"):
    i, j, x = neighbor_list(what, c.to_atoms(), r)
    return np.asarray(i), np.asarray(j), np.asarray(x)


def _sorted(i, j, x):
    x = np.asarray(x, float).reshape(len(i), -1)
    order = np.lexsort(tuple(x.T[::-1]) + (j, i))
    return i[order], j[order], x[order]


@pytest.mark.parametrize("kind", ["cluster", "cluster in a box", "tube", "wide tube"])
def test_the_tree_finds_the_pairs_ase_finds(kind):
    s = {"cluster": lambda: _cluster(4),
         "cluster in a box": lambda: _cluster(4, cell=True),
         "tube": lambda: nanotube(6, 6, length=6),
         "wide tube": lambda: nanotube(12, 0, length=3)}[kind]()
    c, _ = _record(s)
    assert P._uses_tree(c)
    for vectors in (False, True):
        mine = _sorted(*P._neighbour_pairs(c, 6.0, vectors=vectors))
        ref = _sorted(*_ase_pairs(c, 6.0, "ijD" if vectors else "ijd"))
        assert np.array_equal(mine[0], ref[0]) and np.array_equal(mine[1], ref[1])
        assert np.allclose(mine[2], ref[2], atol=1e-12)


def test_crystals_and_layers_keep_the_ase_route():
    crystal, _ = _record(bulk("Cu", "fcc", a=3.6, cubic=True))
    sheet = graphene(size=(3, 3, 1), vacuum=6.0)
    sheet.pbc = (True, True, False)
    layer, _ = _record(sheet)
    tube = nanotube(6, 6, length=2)
    tube.center(vacuum=6.0, axis=(0, 1))       # a tube in a full box
    boxed, _ = _record(tube)
    assert not P._uses_tree(crystal) and not P._uses_tree(layer) and not P._uses_tree(boxed)


def test_the_autocorrelation_of_a_cluster_is_unchanged_by_the_tree(monkeypatch):
    c, _ = _record(_cluster(5))
    new = P.periodic_autocorrelation(c, SPEC)
    monkeypatch.setattr(P, "_uses_tree", lambda crystal: False)
    old = P.periodic_autocorrelation(c, SPEC)
    assert np.max(np.abs(new - old)) <= 1e-12 * np.max(np.abs(old))


def test_the_three_body_channels_of_a_cluster_are_unchanged_by_the_tree(monkeypatch):
    from cmfqsar.threebody import ThreeBodySpec, threebody_features
    c, _ = _record(_cluster(3))
    spec = ThreeBodySpec()
    new = threebody_features(c, spec)
    monkeypatch.setattr(P, "_uses_tree", lambda crystal: False)
    old = threebody_features(c, spec)
    assert np.max(np.abs(new - old)) <= 1e-12 * np.max(np.abs(old))


@pytest.mark.parametrize("kind", ["tube", "sheet"])
def test_the_background_in_blocks_equals_the_whole_matrix(kind, monkeypatch):
    if kind == "tube":
        s = nanotube(6, 6, length=8)
    else:
        s = graphene(size=(6, 6, 1), vacuum=6.0)
        s.pbc = (True, True, False)
    c, _ = _record(s)
    whole = P.periodic_autocorrelation(c, SPEC)
    monkeypatch.setattr(P, "BACKGROUND_BLOCK_ABOVE", 0)
    monkeypatch.setattr(P, "BACKGROUND_BLOCK", 37)        # blocks that do not divide n
    blocks = P.periodic_autocorrelation(c, SPEC)
    assert np.max(np.abs(blocks - whole)) <= 1e-12 * np.max(np.abs(whole))


def test_the_separations_of_some_rows_are_those_rows_of_the_matrix():
    s = nanotube(6, 6, length=3)
    whole = P._normal_separations(s.positions, s.cell.array, s.pbc)
    part = P._normal_separations(s.positions, s.cell.array, s.pbc, slice(5, 20))
    assert np.array_equal(part, whole[5:20])


def _particles(sizes=(2, 3, 4, 5)):
    atoms = []
    for k in sizes:
        for el in ("Cu", "Ag"):
            atoms.append(_cluster(k, el))
    fs = P.inorganic_fieldset(["Ag", "Cu"], physchem=PHYSCHEM)
    names = [f"{a.get_chemical_symbols()[0]}{len(a)}" for a in atoms]
    table = P.crystal_table(atoms, fs, names)
    y = np.array([len(a) ** (-1.0 / 3.0) for a in atoms])
    return table, fs, y


def test_the_model_refuses_the_overlap_kernel_above_the_atom_limit():
    table, fs, y = _particles()
    model = CMFModel(fieldset=fs, method="krr", normalize=True, max_overlap_atoms=100)
    with pytest.raises(ValueError, match=r"Cu309 has 309 atoms.*kernel_source='periodic'"):
        model.fit(table, y, cv_folds=4)
    for pbc in (False, True):
        ok = CMFModel(fieldset=fs, method="krr", normalize=True, pbc_overlap=pbc,
                      max_overlap_atoms=None)
        ok.fit(table, y, cv_folds=4)
        assert np.isfinite(ok.stats.q2)


def test_the_autocorrelation_model_is_not_limited():
    table, fs, y = _particles()
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_d_max=6.0, max_overlap_atoms=10)
    model.fit(table, y, cv_folds=4)
    assert np.isfinite(model.stats.q2)


def test_molecules_are_never_refused():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mols = []
    for smi in ("CCO", "CCCO", "CCCCO", "CCCCCO", "OCCO", "OCCCO"):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=1)
        mols.append(m)
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    table = AtomTable.from_molecules([build_fields(m, fs, name=str(k))
                                      for k, m in enumerate(mols)])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, max_overlap_atoms=1)
    model.fit(table, np.arange(6.0), cv_folds=3)
    assert np.isfinite(model.stats.q2)


def test_the_limit_is_kept_by_a_copy_and_by_a_saved_model():
    fs = FieldSet.create(physchem=["steric"], alpha=0.5)
    assert CMFModel(fieldset=fs).max_overlap_atoms == P.OVERLAP_ATOM_LIMIT
    model = CMFModel(fieldset=fs, max_overlap_atoms=None)
    assert model.fresh_copy().max_overlap_atoms is None
    assert CMFModel(fieldset=fs, max_overlap_atoms=500).fresh_copy().max_overlap_atoms == 500
    st = model.state()
    assert st["max_overlap_atoms"] is None
    assert CMFModel.from_state(st).max_overlap_atoms is None
    st.pop("max_overlap_atoms")                 # a model saved before this item
    assert CMFModel.from_state(st).max_overlap_atoms == P.OVERLAP_ATOM_LIMIT


def test_the_crystal_window_refuses_before_computing(qt_application, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ase.io import write

    from cmfqsar.gui.crystal_window import CrystalWindow
    atoms = []
    for k, el in [(2, "Cu"), (3, "Cu"), (2, "Ag"), (3, "Ag"), (4, "Cu")]:
        a = _cluster(k, el)
        a.info.update({"name": f"{el}{len(a)}", "y": float(len(a))})
        atoms.append(a)
    path = str(tmp_path / "particles.extxyz")
    write(path, atoms)
    monkeypatch.setattr(P, "OVERLAP_ATOM_LIMIT", 100)
    told = []
    w = CrystalWindow(background=False)
    w.load(path)
    monkeypatch.setattr(w, "_tell", lambda title, text: told.append((title, text)))
    w.cmb_kernel.setCurrentIndex(1)
    w.build_model()
    assert w.model is None
    assert told and "Cu147 has 147 atoms" in told[0][1]
