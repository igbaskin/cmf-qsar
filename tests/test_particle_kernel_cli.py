"""The particle surface kernel and open boundaries on the command line and in the window (NT11).

Plan item NT5 built the particle surface kernel and measured it from Python only.  Plan item NT11
makes it a choice of `cmf build-crystals`, of `cmf predict` and of the crystal window.  The tests
check what that adds:

* the core rule finds the core ("metals", a list of elements, a per-atom array) and refuses a
  structure with no core or with nothing else, naming it;
* a table built from whole structures by the core rule gives exactly the kernel the benchmark of
  NT5 builds by hand from the shell and the core;
* the command line fits the model, saves the rule with it, and `cmf predict` applies the same
  split: its predictions on the training file are the fitted values;
* the options that belong to the periodic kernel are refused with this one, and `cmf sites`
  refuses the model;
* the window offers the kernel and open boundaries, shows only the controls that apply, fits
  the same model as the command line, and passes the descriptor weight to the field set.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("ase", reason="the structure path needs ASE")

from ase import Atoms                                                   # noqa: E402
from ase.io import read                                                 # noqa: E402

from cmfqsar.invariant import with_core                                 # noqa: E402
from cmfqsar.model import CMFModel                                      # noqa: E402
from cmfqsar.periodic import (core_mask, crystal_fields, inorganic_fieldset,  # noqa: E402
                              particle_table, split_core, structure_table)
from cmfqsar.fields import AtomTable                                    # noqa: E402

DATA = os.path.join(ROOT, "examples", "coated_particles", "particles.extxyz")


@pytest.fixture(scope="module")
def particles():
    return read(DATA, index=":")


def _cli(*args, cwd=ROOT):
    env = dict(os.environ, PYTHONPATH=ROOT, QT_QPA_PLATFORM="offscreen")
    return subprocess.run([sys.executable, "-m", "cmfqsar.cli", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=900)


def test_the_core_rules():
    a = Atoms(["Au", "Au", "S", "C", "H", "Cd", "Se"],
              positions=np.arange(21, dtype=float).reshape(7, 3))
    assert core_mask(a, "metals").tolist() == [True, True, False, False, False, True, False]
    assert core_mask(a, "Cd,Se").tolist() == [False] * 5 + [True, True]
    a.arrays["core"] = np.array([1, 0, 0, 0, 0, 0, 0])
    assert core_mask(a, "array:core").tolist() == [True] + [False] * 6
    with pytest.raises(ValueError, match="no per-atom array 'shell'"):
        core_mask(a, "array:shell")
    shell, core = split_core(a, "Au")
    assert shell.get_chemical_symbols() == ["S", "C", "H", "Cd", "Se"]
    assert np.allclose(core, a.positions[:2])


def test_a_structure_without_a_core_or_without_a_shell_is_named():
    a = Atoms(["C", "H"], positions=[[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="mol7: no atom matches the core rule"):
        split_core(a, "metals", "mol7")
    with pytest.raises(ValueError, match="every atom matches"):
        split_core(Atoms(["Au", "Au"], positions=[[0, 0, 0], [2.9, 0, 0]]), "metals")


def test_the_table_gives_the_kernel_the_benchmark_builds_by_hand(particles):
    # electronegativity charges: guessing oxidation states for a whole gold particle with its
    # ligands takes minutes, and the last check below builds whole particles
    fs = inorganic_fieldset([], physchem=["electrostatic", "steric", "prop:electronegativity"],
                            charge_model="electronegativity")
    some = particles[:8]
    names = [p.info["name"] for p in some]
    by_rule = particle_table(some, fs, names, core="Au")
    by_hand = []
    for p, nm in zip(some, names):
        mask = np.array([s == "Au" for s in p.get_chemical_symbols()])
        shell = p[np.flatnonzero(~mask)]
        by_hand.append(with_core(crystal_fields(shell, fs, nm), p.positions[mask]))
    kw = dict(kernel_source="particle", ac_d_max=12.0, ac_sigma=1.0, surf_z_min=2.0,
              surf_z_max=12.0, surf_z_step=2.0, particle_shape="sphere")
    K1 = CMFModel(fieldset=fs, method="krr", normalize=True, **kw).training_kernel(by_rule)
    K2 = CMFModel(fieldset=fs, method="krr", normalize=True,
                  **kw).training_kernel(AtomTable.from_molecules(by_hand))
    assert np.array_equal(K1, K2)
    # structure_table follows the model's kernel
    model = CMFModel(fieldset=fs, particle_core="Au", **kw)
    t = structure_table(model, some, names)
    assert all(hasattr(m, "core") for m in t.molecules)
    t2 = structure_table(CMFModel(fieldset=fs, kernel_source="periodic"), some, names)
    assert not any(hasattr(m, "core") for m in t2.molecules)


def test_the_command_line_fits_saves_and_predicts(tmp_path):
    model, pred, out = tmp_path / "m.cmf", tmp_path / "fit.csv", tmp_path / "pred.csv"
    r = _cli("build-crystals", DATA, "--kernel", "particle-surface", "--core", "Au",
             "--sigma", "1", "--cutoff", "12", "--height-step", "2", "--folds", "5",
             "--elements", "none", "--save", str(model), "--predictions", str(pred))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Kernel: particle surface" in r.stdout
    from cmfqsar.dataio import load_model
    m, _ = load_model(str(model))
    assert (m.kernel_source, m.particle_core, m.particle_shape) == ("particle", "Au", "sphere")
    assert (m.surf_z_min, m.surf_z_max, m.surf_z_step) == (2.0, 12.0, 2.0)
    r = _cli("predict", str(model), DATA, "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "particle surface kernel" in r.stdout
    import csv
    fit = {row["name"]: float(row["predicted"]) for row in csv.DictReader(open(pred))}
    new = {row["name"]: float(row["predicted"]) for row in csv.DictReader(open(out))}
    assert fit.keys() == new.keys() and len(fit) == 72
    assert max(abs(fit[k] - new[k]) for k in fit) < 1e-9
    r = _cli("sites", str(model), DATA)
    assert r.returncode != 0 and "no split into sites" in (r.stdout + r.stderr)


def test_options_of_the_periodic_kernel_are_refused():
    for extra in (["--heights", "4"], ["--no-background"], ["--site-file", "x.csv"],
                  ["--shell-heights", "5,3"]):
        r = _cli("build-crystals", DATA, "--kernel", "particle-surface", *extra)
        assert r.returncode != 0
        assert "particle-surface" in r.stderr or "--shell-heights" in r.stderr, extra
    r = _cli("build-crystals", DATA, "--kernel", "particle-surface", "--core", "Pt")
    assert r.returncode != 0 and "no atom matches the core rule 'Pt'" in r.stderr


def test_open_boundaries_on_the_command_line(tmp_path):
    a = Atoms(["Na", "Cl"], positions=[[0, 0, 0], [2.8, 0, 0]], cell=[5.6, 5.6, 5.6], pbc=True)
    from ase.io import write
    path = tmp_path / "x.extxyz"
    write(str(path), [a])
    from cmfqsar.periodic import read_crystal_set
    (got,), _ = read_crystal_set(str(path), pbc="none")
    assert not any(got.get_pbc())
    r = _cli("build-crystals", "--help")
    assert "'none' for a particle" in " ".join(r.stdout.split())


# ------------------------------------------------------------------------------- the window
@pytest.fixture(scope="module")
def app(qt_application):
    return qt_application


def test_the_window_offers_the_kernel_and_open_boundaries(app):
    from cmfqsar.gui.crystal_window import KERNELS, PBC_CHOICES, CrystalWindow
    w = CrystalWindow(background=False)
    assert [w.cmb_kernel.itemText(i) for i in range(w.cmb_kernel.count())] == list(KERNELS)
    assert "particle surface" in KERNELS[2]
    assert ("none: a particle or a molecule", "none") in PBC_CHOICES
    w.cmb_pbc.setCurrentIndex(len(PBC_CHOICES) - 1)
    assert w._pbc_text() == "none"
    assert not any(w._kernel_form.isRowVisible(x) for x in w._particle_rows)
    w.cmb_kernel.setCurrentIndex(2)
    assert all(w._kernel_form.isRowVisible(x) for x in w._particle_rows)
    assert not w.spin_taper.isEnabled() and not w.chk_background.isEnabled()
    w.cmb_kernel.setCurrentIndex(0)
    assert w.spin_taper.isEnabled()


def test_the_window_fits_the_model_of_the_command_line(app, particles):
    from cmfqsar.gui.crystal_window import CrystalWindow
    w = CrystalWindow(background=False)
    w.load(DATA, target_key="accessibility")
    w.cmb_kernel.setCurrentIndex(2)
    w.ed_core.setText("Au")
    w.spin_sigma.setValue(1.0)
    w.spin_cutoff.setValue(12.0)
    w.spin_zstep.setValue(2.0)
    w.cmb_elements.setCurrentIndex(1)
    w.spin_descriptor_h.setValue(0.5)
    assert w._fieldset(range(3)).descriptor_h == 0.5
    w.build_model()
    m = w.model
    assert (m.kernel_source, m.particle_core, m.ac_d_max, m.surf_z_step) == \
        ("particle", "Au", 12.0, 2.0)
    # the same settings on the command line: the same cross-validated q2
    fs = w.model_fieldset
    y = np.array([p.info["accessibility"] for p in particles])
    from cmfqsar.model import grid_search
    ref = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="particle",
                   ac_d_max=12.0, ac_sigma=1.0, surf_z_min=2.0, surf_z_max=12.0,
                   surf_z_step=2.0, particle_core="Au")
    table = particle_table(particles, fs, [p.info["name"] for p in particles], core="Au")
    res = grid_search(table, y, ref, [1.0], lams=[1e-4, 1e-3, 1e-2, 0.1, 1], folds=5, seed=0)
    assert abs(res.best_q2 - m.stats.q2) < 1e-9
    w.cmb_reading.setCurrentIndex(0)
    w.update_reading()
    assert w.reading.rowCount() > 0 and "rho" in w.reading.item(0, 0).text()
    w.chk_sites.setChecked(True)
    w.update_display()
    assert "no split of a prediction over the sites" in w.legend.text()
    # a core rule that finds nothing is refused before anything is computed
    w.ed_core.setText("Pt")
    w.build_model()
    assert w.model.particle_core == "Au"
