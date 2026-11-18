"""The twenty domains plan item PP2 adds, and the parts of its measurement that need no fit.

* Every domain follows the rules of PP1: a natural protein named by a PDB code, a wild-type
  stability above 3 kcal/mol, a fold cluster that no other domain has, and a table and a model
  in the repository.
* A draw takes at most four substitutions at a position and keeps every position.
* The score inside a domain is the correlation of the held-out predictions with ddG there.
"""
import csv
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "benchmarks"), os.path.join(ROOT, "examples")):
    sys.path.insert(0, p)

pytest.importorskip("rdkit")

import make_mutant_examples as MX                                       # noqa: E402
import run_new_proteins as R                                            # noqa: E402

EXAMPLES = os.path.join(ROOT, "examples", "mutants")


def test_the_domains_follow_the_rules_of_pp1():
    assert len(MX.NEW_DOMAINS) == 20
    assert not set(MX.NEW_DOMAINS) & set(MX.DOMAINS)
    clusters = [c for c, _dg in MX.ALL_DOMAINS.values()]
    assert len(set(clusters)) == len(clusters) == 24
    for d, (_c, dg) in MX.ALL_DOMAINS.items():
        assert re.fullmatch(r"[0-9][A-Z0-9]{3}", d)
        assert dg > 3.0
        assert os.path.exists(os.path.join(EXAMPLES, d, f"{d}.pdb"))
        with open(os.path.join(EXAMPLES, d, "mutants.csv"), newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) > 1000
        assert all(-1.0 <= float(r["dG_ML"]) <= 5.0 for r in rows)
        assert len({r["label"] for r in rows}) == len(rows)
        with open(os.path.join(EXAMPLES, d, "wild_type.txt")) as fh:
            text = fh.read()
        assert float(re.search(r"dG_ML (\S+)", text).group(1)) == pytest.approx(dg, abs=0.006)


def test_a_draw_keeps_every_position_and_at_most_four_substitutions():
    import run_mutants as M
    s = M.SiteSamples({}, ("1AOY", "2KVS"), R.PER_POSITION, 0)
    for d in ("1AOY", "2KVS"):
        with open(os.path.join(EXAMPLES, d, "mutants.csv"), newline="") as fh:
            positions = {int(r["position"]) for r in csv.DictReader(fh)}
        here = [p for dd, p in zip(s.domain, s.position) if dd == d]
        assert set(here) == positions
        counts = np.bincount(here)
        assert counts.max() <= R.PER_POSITION
    other = M.SiteSamples({}, ("1AOY",), R.PER_POSITION, 1)
    first = [lab for dd, lab in zip(s.domain, s.label) if dd == "1AOY"]
    assert first != list(other.label)


def test_the_score_inside_a_domain():
    y = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 0.0, 1.0])
    s = SimpleNamespace(y=y, domain=["a"] * 4 + ["b"] * 4)
    perfect = SimpleNamespace(predicted=y.copy())
    turned = SimpleNamespace(predicted=np.r_[y[:4], -y[4:]])
    out = R.per_domain([perfect, turned], s)
    assert out["a"] == pytest.approx(1.0)
    assert out["b"] == pytest.approx(0.0)
