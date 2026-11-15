"""The reference of the covariate-shift tutorial: the plain interval with no interval outside the
applicability domain (rule cv-ad of benchmarks/run_shift.py)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

RS = pytest.importorskip("run_shift")


@pytest.fixture(scope="module")
def one_split(tmp_path_factory):
    out = tmp_path_factory.mktemp("shift")
    return RS.run_set("THR", repeats=1, force=True, results=str(out),
                      scenarios=["random", "neighbours"]), str(out)


def test_the_domain_rule_is_the_plain_interval_with_the_far_molecules_refused(one_split):
    rec, _ = one_split
    assert "cv-ad" in RS.RULES and RS.RULES.index("cv-ad") == RS.RULES.index("cv-norm") + 1
    for scen in ("random", "neighbours"):
        refused = next(d["ad_refused"] for d in rec["diagnostics"] if d["scenario"] == scen)
        assert 0.0 <= refused <= 1.0
        for level in RS.LEVELS:
            row = {r["rule"]: r for r in rec["rows"]
                   if r["scenario"] == scen and r["level"] == level}
            ad, cv = row["cv-ad"], row["cv"]
            # an infinite interval covers, so refusing can only raise the coverage
            assert ad["coverage"] >= cv["coverage"] - 1e-12
            assert ad["infinite"] == pytest.approx(refused)
            assert cv["infinite"] == 0.0


def test_the_table_has_the_rule(one_split):
    _rec, out = one_split
    RS.report(results=out)
    text = open(os.path.join(out, "table.md"), encoding="utf-8").read()
    assert "`cv-ad`" in text
