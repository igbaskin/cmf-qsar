"""The Map tab draws two different pictures, and must not confuse them."""
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cmfqsar.gui.panels import ResultsPanel                   # noqa: E402


@pytest.fixture(scope="module")
def app(qt_application):
    return qt_application


@pytest.fixture()
def panel(app):
    return ResultsPanel()


def _coords(n=12):
    return np.random.default_rng(0).normal(size=(n, 2))


def test_score_plot_draws_every_molecule(panel):
    c = _coords()
    panel.set_scatter(c, np.arange(12.0), [f"m{i}" for i in range(12)], list(range(12)),
                      title="kernel PCA score plot", xlabel="PC1 (40 %)", ylabel="PC2 (20 %)")
    assert panel.tabs.isTabVisible(panel.tab_map)
    assert len(panel.map_pts.data) == 12
    assert panel.map_plot.plotItem.getAxis("bottom").labelText == "PC1 (40 %)"


def test_the_score_plot_has_no_landscape_behind_it(panel):
    """A scatter must clear the image, or the previous map's landscape stays underneath it."""
    panel.set_map(np.arange(100.0).reshape(10, 10), _coords(), np.arange(12.0),
                  [f"m{i}" for i in range(12)], list(range(12)))
    assert panel.map_img.image is not None
    panel.set_scatter(_coords(), None, [f"m{i}" for i in range(12)], list(range(12)))
    assert panel.map_img.image is None


def test_a_map_after_a_score_plot_takes_its_axes_back(panel):
    panel.set_scatter(_coords(), None, [f"m{i}" for i in range(12)], list(range(12)),
                      xlabel="PC1 (40 %)", ylabel="PC2 (20 %)")
    panel.set_map(np.arange(100.0).reshape(10, 10), _coords(), np.arange(12.0),
                  [f"m{i}" for i in range(12)], list(range(12)))
    assert panel.map_plot.plotItem.getAxis("bottom").labelText == "latent coordinate 1"


def test_nothing_to_draw_hides_the_tab(panel):
    panel.set_scatter(np.zeros((0, 2)))
    assert not panel.tabs.isTabVisible(panel.tab_map)
    panel.set_scatter(None)
    assert not panel.tabs.isTabVisible(panel.tab_map)


def test_points_are_coloured_only_where_there_is_a_value(panel):
    v = np.array([1.0, np.nan, 3.0, 4.0])
    panel.set_scatter(_coords(4), v, ["a", "b", "c", "d"], [0, 1, 2, 3])
    assert len(panel.map_pts.data) == 4


def test_the_window_builds_a_score_plot_from_a_fitted_map(panel):
    """The connection between the map and the tab: axis labels, colours and the log line.

    Called unbound on a stub, because what is being checked is the connections -- that the method
    uses attributes that exist and hands the panel what it can draw -- and not the window.
    """
    from types import SimpleNamespace

    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    from cmfqsar.gtm import CMFMap
    from cmfqsar.gui.main_window import MainWindow

    smis = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1O", "c1ccccc1N", "c1ccccc1Cl",
            "c1ccccc1F", "c1ccc(C)cc1C", "c1ccc(O)cc1C"]
    mols = []
    for smi in smis:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=7)
        Chem.SanitizeMol(m)
        mols.append(m)
    fs = FieldSet.create(alpha=0.6)
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)])
    cmap = CMFMap(n_components=4).fit_pca(table, fs)

    lines = []
    stub = SimpleNamespace(
        ds=SimpleNamespace(names=[f"m{i}" for i in range(len(smis))], activity_prop="pIC50"),
        results_panel=panel, log=lines.append)
    act = np.arange(len(smis), dtype=float)
    MainWindow._show_score_plot(stub, cmap, list(range(len(smis))), act, 0, None, 0, 1)

    assert panel.tabs.isTabVisible(panel.tab_map)
    assert len(panel.map_pts.data) == len(smis)
    assert panel.map_plot.plotItem.getAxis("bottom").labelText.startswith("PC1 (")
    assert "pIC50" in panel.map_plot.plotItem.titleLabel.text
    assert lines and "kernel variance" in lines[0]


def test_a_component_beyond_the_rank_falls_back_to_the_first_two(panel):
    from types import SimpleNamespace

    from cmfqsar.gtm import CMFMap, KernelPCA
    from cmfqsar.gui.main_window import MainWindow

    rng = np.random.default_rng(1)
    X = rng.normal(size=(9, 3))
    K = X @ X.T
    cmap = CMFMap()
    cmap.kpca = KernelPCA(n_components=3).fit(K)
    cmap.scores_ = cmap.kpca.transform()

    lines = []
    stub = SimpleNamespace(ds=SimpleNamespace(names=[f"m{i}" for i in range(9)], activity_prop=""),
                           results_panel=panel, log=lines.append)
    MainWindow._show_score_plot(stub, cmap, list(range(9)), np.arange(9.0), 0, None, 0, 40)
    assert any("plotting the first two instead" in l for l in lines)
    assert panel.map_plot.plotItem.getAxis("bottom").labelText.startswith("PC1 (")
