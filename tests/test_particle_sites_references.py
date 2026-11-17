"""The stage `references` of benchmarks/run_particle_sites.py (tutorial 32)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
sys.path.insert(0, os.path.join(HERE, ".."))

import run_particle_sites as PS  # noqa: E402


def _toy(n=60, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    metal = np.array(["Au", "Pt"] * (n // 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + (metal == "Pt") * 0.3 + 0.05 * rng.randn(n)
    groups = np.repeat(np.arange(n // 5), 5)
    return X, y, metal, groups


def test_the_gaussian_reference_equals_the_one_of_the_model_stage():
    import run_core_fields
    X, y, metal, groups = _toy()
    tr, te = np.arange(10, 60), np.arange(10)
    run_core_fields.CHOSEN.clear()
    old = run_core_fields._gaussian_counts(metal, X, y, tr, te, 0, groups, PS.COUNT_GAMMAS,
                                           PS.COUNT_LAMBDAS)
    new, chosen = PS.gaussian_per_metal(X, y, metal, tr, te, 0, groups)
    assert chosen == (run_core_fields.CHOSEN[-1]["gamma"], run_core_fields.CHOSEN[-1]["lambda"])
    assert np.allclose(new, old, atol=1e-8)


def test_the_kernel_is_zero_between_metals():
    X, y, metal, groups = _toy()
    tr = np.flatnonzero(metal == "Au")
    te = np.flatnonzero(metal == "Pt")[:5]
    p, _ = PS.gaussian_per_metal(X, y, metal, tr, te, 0, groups)
    # nothing of the gold sites reaches a platinum site: it gets the training mean
    assert np.allclose(p, y[tr].mean())


def test_the_paired_difference_is_the_difference_of_q2():
    rng = np.random.RandomState(1)
    y = rng.randn(40)
    a = y + 0.1 * rng.randn(40)
    b = y + 0.5 * rng.randn(40)
    particle = np.repeat(np.arange(8), 5).astype(str)
    g = PS.paired_over_particles(y, a, b, particle, draws=200)

    def q2(p):
        return 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
    assert g["mean"] == pytest.approx(q2(a) - q2(b))
    assert g["lo"] <= g["mean"] <= g["hi"]
    same = PS.paired_over_particles(y, a, a, particle, draws=50)
    assert same["lo"] == same["hi"] == 0.0


def test_the_numbers_of_a_site_hold_its_exposure_and_the_shell_sums():
    from cmfqsar.periodic import core_columns, crystal_table
    from cmfqsar.sites import SHELLS, shell_count_features
    D = PS.load()
    t = D["targets"]
    plain = PS.inorganic_fieldset(list(PS.METALS), physchem=PS.PHYSCHEM, other=False)
    crystals = list(crystal_table(D["particles"], plain, D["names"]).molecules)
    counts = shell_count_features(crystals, t, elements=list(PS.METALS))
    counts = counts[:, counts.any(axis=0)]
    X = PS.site_numbers(D, counts, SHELLS)
    widths = [X[name].shape[1] for name in PS.REFERENCES]
    assert widths == [2 + counts.shape[1], 3 + counts.shape[1],
                      3 + counts.shape[1] + 3 * len(SHELLS)]
    r = 0
    k, s = int(t.molecule[r]), int(t.atom[r])
    exposure = core_columns(D["particles"][k], ["prop:exposure"])["prop:exposure"][s]
    assert X[PS.REFERENCES[1]][r, -1] == pytest.approx(exposure)
    sums = X[PS.REFERENCES[2]][:, -3 * len(SHELLS):]
    assert np.all(sums >= 0)
    # no metal atom lies within 2 A of another, so the first shell is empty
    assert np.all(sums[:, :3] == 0)
