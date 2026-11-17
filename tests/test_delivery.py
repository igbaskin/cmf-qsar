"""What a predicted binding constant means for a formulation (plan item SC2).

No phase-solubility set is in the repository, so the quantities are checked against what
defines them, not against measurements:

* the solubility gain and the fraction bound are the mass balance of a 1:1 complex, solved here
  numerically by bisection and compared with the closed forms of ``cmfqsar.delivery``;
* the gain without an intrinsic solubility is the upper bound of the gain with any;
* the diffusion limit is 8RT/(3 eta), about 7.4e9 L/(mol s) in water at 25 C, and the half-life
  is ln 2 K / k_on;
* every quantity grows with K, so the interval of K gives the interval of each, and a free
  energy scale turns the ends round;
* `cmf predict --host-conc` prints and writes them, with the conformal interval when the model
  has one and a sentence saying so when it has not.
"""
import csv
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cmfqsar.delivery import (R_GAS, Delivery, bound_fraction, constant_from_activity,  # noqa: E402
                              diffusion_limit, format_time, half_life, rows, solubility_gain)

KS = (0.5, 12.0, 450.0, 2.0e4, 3.0e6)


def test_the_activity_scales():
    assert np.allclose(constant_from_activity([2.0], "log10K"), [100.0])
    assert np.allclose(constant_from_activity([math.log(50.0)], "lnK"), [50.0])
    assert np.allclose(constant_from_activity([7.0], "K"), [7.0])
    dG = -R_GAS * 300.0 * math.log(1000.0) / 1000.0           # kJ/mol for K = 1000 at 300 K
    assert np.allclose(constant_from_activity([dG], "dG", temperature=300.0), [1000.0])
    with pytest.raises(ValueError, match="unknown activity scale"):
        constant_from_activity([1.0], "pIC50")


def test_the_solubility_gain_is_the_mass_balance():
    """Solid guest in excess: free guest S0, complex C = K S0 Hf, host Ht = Hf + C."""
    for K in KS:
        for S0, Ht in ((1e-4, 0.05), (2e-2, 0.01), (1e-6, 0.2)):
            Hf = brentq(lambda h: h + K * S0 * h - Ht, 0.0, Ht, xtol=1e-300, rtol=1e-15)
            exact = (S0 + K * S0 * Hf) / S0
            assert solubility_gain(K, Ht, S0) == pytest.approx(exact, rel=1e-10)
            # without S0, the bound reached when K S0 is small
            assert solubility_gain(K, Ht) >= exact - 1e-12
    assert solubility_gain(450.0, 0.05, 1e-9) == pytest.approx(solubility_gain(450.0, 0.05),
                                                               rel=1e-6)


def test_the_fraction_bound_is_the_mass_balance():
    for K in KS:
        for G, H, D in ((1e-3, 1e-2, 1.0), (1e-2, 1e-2, 10.0), (5e-2, 1e-3, 50.0)):
            g, h = G / D, H / D
            C = brentq(lambda c: K * (g - c) * (h - c) - c, 0.0, min(g, h), xtol=1e-300,
                       rtol=1e-15)
            assert float(bound_fraction(K, G, H, D)) == pytest.approx(C / g, rel=1e-8)
        # a dilute guest: K H / (1 + K H)
        assert float(bound_fraction(K, None, 0.01, 4.0)) == pytest.approx(
            K * 0.0025 / (1 + K * 0.0025))
        # diluting more releases more
        f = [float(bound_fraction(K, 1e-3, 1e-2, d)) for d in (1, 10, 100, 1000)]
        assert all(a >= b for a, b in zip(f, f[1:]))
    with pytest.raises(ValueError, match="dilution"):
        bound_fraction(10.0, 1e-3, 1e-2, 0.5)


def test_the_diffusion_limit_and_the_half_life():
    k = diffusion_limit()
    assert k == pytest.approx(8 * R_GAS * 298.15 / (3 * 0.890e-3) * 1000.0)
    assert 7.3e9 < k < 7.5e9
    assert float(half_life(1000.0, k)) == pytest.approx(math.log(2) * 1000.0 / k)
    assert format_time(1.77e-7) == "177 ns"
    assert format_time(5400.0) == "1.5 h"


def test_the_interval_of_k_gives_the_interval_of_every_quantity():
    d = Delivery(host_total=0.05, intrinsic=1e-4, dilution=20.0)
    recs = rows(["a", "b"], [2.0, 3.5], d, low=[1.5, 3.0], high=[2.5, 4.0])
    for r in recs:
        for key in ("K", "solubility_gain", "solubility", "bound_fraction", "half_life_s"):
            assert r[key + "_lo"] < r[key] < r[key + "_hi"], key
    # on the free-energy scale the lower end of dG is the larger constant
    e = Delivery(host_total=0.05, scale="dG")
    (r,) = rows(["a"], [-20.0], e, low=[-25.0], high=[-15.0])
    assert r["K_lo"] < r["K"] < r["K_hi"]
    assert r["half_life_s_lo"] < r["half_life_s"] < r["half_life_s_hi"]


def _cli(*args):
    env = dict(os.environ, PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, "-m", "cmfqsar.cli", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    pytest.importorskip("rdkit")
    tmp = tmp_path_factory.mktemp("sc2")
    src = os.path.join(ROOT, "examples", "cyclodextrin", "hydroxypropyl_beta_cyclodextrin.csv")
    with open(src, encoding="utf-8") as fh:
        lines = fh.readlines()
    data = tmp / "hp.csv"
    data.write_text("".join(lines[:41]), encoding="utf-8")
    few = tmp / "few.csv"
    few.write_text("".join(lines[:4]), encoding="utf-8")
    out = {}
    for name, extra in (("conformal", ["--conformal", "0.9"]), ("plain", [])):
        path = tmp / f"{name}.cmf"
        r = _cli("build", str(data), "--kernel", "autocorrelation", "--folds", "5", *extra,
                 "--save", str(path))
        assert r.returncode == 0, r.stderr[-2000:]
        out[name] = path
    return out, few, tmp


def test_predict_reports_the_quantities_with_their_interval(models):
    paths, few, tmp = models
    out = tmp / "pred.csv"
    r = _cli("predict", str(paths["conformal"]), str(few), "--host-conc", "0.05",
             "--intrinsic-solubility", "1e-4", "--dilution", "50", "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Delivery (plan item SC2)" in r.stdout and "lower bound" in r.stdout
    recs = list(csv.DictReader(open(out)))
    assert len(recs) == 3
    for rec in recs:
        # the predicted log K is written with four decimals: K agrees to about 1e-4
        K = float(rec["K"])
        assert K == pytest.approx(10.0 ** float(rec["predicted"]), rel=5e-4)
        assert float(rec["solubility_gain"]) == pytest.approx(
            float(solubility_gain(K, 0.05, 1e-4)), rel=1e-5)
        assert float(rec["half_life_s"]) == pytest.approx(
            math.log(2) * K / diffusion_limit(), rel=1e-5)
        for key in ("K", "solubility_gain", "bound_fraction", "half_life_s"):
            assert float(rec[key + "_lo"]) <= float(rec[key]) <= float(rec[key + "_hi"])
        # the ends are those of the conformal interval of log K
        assert float(rec["K_lo"]) == pytest.approx(10.0 ** float(rec["lower"]), rel=1e-3)


def test_predict_without_a_calibration_says_so(models):
    paths, few, tmp = models
    r = _cli("predict", str(paths["plain"]), str(few), "--host-conc", "0.01")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "no conformal calibration" in r.stdout
    assert "upper bound 1 + K [host]" in r.stdout
    r = _cli("predict", str(paths["plain"]), str(few), "--host-conc", "0.01", "--dilution", "0.5")
    assert r.returncode != 0 and "dilution" in r.stderr
