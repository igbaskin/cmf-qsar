"""The diagnosis: what it asks of a set, and the leak that would make it useless."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar.diagnose import (Diagnosis, counts_route, descriptor_route, diagnose,
                              frame_findings, paired_interval, Route)


def _series(n=16, seed=0, congeneric=True):
    from cmfqsar.align import align_dataset

    subs = ["", "C", "CC", "F", "Cl", "O", "OC", "N", "NC", "C#N", "CO", "Br", "CCC", "I",
            "C(C)C", "OCC"][:n]
    diverse = ["c1ccccc1C", "C1CCCCC1N", "c1ccncc1O", "C1CCOC1", "c1cc[nH]c1C", "CCCCCCO",
               "c1ccc2ccccc2c1", "C1CCNCC1", "Cc1cscn1", "CC(=O)Nc1ccccc1", "C1CC1CO",
               "c1ccc(cc1)S", "CCN(CC)CC", "c1ccoc1CC", "C1CCCC1", "CC(C)(C)O"][:n]
    smiles = [f"c1ccc(cc1){s}" if s else "c1ccccc1" for s in subs] if congeneric else diverse
    mols, names = [], []
    for i, smi in enumerate(smiles):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=40 + i)
        AllChem.MMFFOptimizeMolecule(m)
        Chem.SanitizeMol(m)
        m.SetProp("_Name", f"m{i}")
        mols.append(m)
        names.append(f"m{i}")
    if congeneric:
        align_dataset(mols, 0)
    rng = np.random.default_rng(seed)
    y = np.array([4.0 + 0.3 * m.GetNumHeavyAtoms() for m in mols]) + 0.2 * rng.normal(size=len(mols))
    return mols, names, y


def test_the_descriptor_baseline_refuses_the_activity_itself():
    """Most files store the activity as a property; handed it, this baseline returns q2 = 1.

    Then every set would be diagnosed "use descriptors", which is why the guard
    exists and why it is tested before anything else here.
    """
    mols, names, y = _series(14)
    for m, v in zip(mols, y):
        m.SetProp("pIC50", f"{v:.6f}")          # the activity, as a file would carry it
        m.SetProp("mw_copy", f"{m.GetNumHeavyAtoms():d}")
    leaky = descriptor_route(mols, y, folds=4, seed=0)
    assert "another name" in leaky.note or "activity itself" in leaky.note
    assert not (leaky.q2 > 0.999), "the activity was used as a descriptor"
    named = descriptor_route(mols, y, folds=4, seed=0, exclude=["pIC50"])
    assert named.q2 < 0.999


def test_a_perfect_descriptor_is_dropped_by_correlation_even_when_it_is_not_named():
    mols, names, y = _series(14)
    for m, v in zip(mols, y):
        m.SetProp("some_other_column", f"{v:.6f}")
    r = descriptor_route(mols, y, folds=4, seed=0)        # nothing excluded by name
    assert "some_other_column" in r.note
    assert not (r.q2 > 0.999)


def test_the_frame_questions_notice_a_diverse_set_and_a_congeneric_one():
    tight, names_t, _y = _series(14, congeneric=True)
    loose, names_l, _y2 = _series(14, congeneric=False)
    ft = {f.code: f for f in frame_findings(tight, names_t)}
    fl = {f.code: f for f in frame_findings(loose, names_l)}
    assert ft["chemotypes"].numbers["chemotypes"] <= fl["chemotypes"].numbers["chemotypes"]
    assert ft["superimposed"].level == "ok", "an aligned series must read as superimposed"
    # the unaligned set must not pass; with a small symmetric framework the test is allowed to
    # say it cannot decide, but never that molecules at their own origins share a frame
    assert fl["superimposed"].level in ("warn", "note")
    assert fl["superimposed"].numbers["core_rmsd"] > ft["superimposed"].numbers["core_rmsd"]


def test_the_paired_interval_is_wide_on_few_molecules():
    """A difference is only a difference against its interval, which is the purpose of rule 4."""
    y = np.linspace(0, 5, 12)
    a = Route("a", q2=0.6, predicted=y + np.random.RandomState(0).normal(0, 0.9, 12))
    b = Route("b", q2=0.55, predicted=y + np.random.RandomState(1).normal(0, 0.95, 12))
    diff, lo, hi = paired_interval(a, b, y, runs=500, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < diff < hi
    assert hi - lo > 0.1, "twelve molecules cannot resolve a small difference"


def test_a_set_with_nothing_in_it_is_refused_before_any_modelling():
    mols, names, _y = _series(12)
    flat = np.full(len(mols), 7.0)
    d = diagnose(mols, names, flat, routes=("counts",))
    assert d.recommended == "none"
    assert any(f.code == "size" for f in d.findings)
    assert not d.routes, "no route should be run on a set that cannot support one"


def test_the_diagnosis_runs_end_to_end_and_reports_itself():
    mols, names, y = _series(16)
    d = diagnose(mols, names, y, routes=("aligned", "counts"), folds=4,
                 alphas=(0.3, 1.0))
    assert isinstance(d, Diagnosis)
    assert set(d.routes) == {"aligned", "counts"}
    assert d.recommended
    text = d.report()
    for piece in ("molecules, activity from", "How well each route predicts",
                  "What the set says", "Recommended route"):
        assert piece in text
    js = d.to_dict()
    assert js["routes"]["counts"]["q2"] == pytest.approx(d.routes["counts"].q2)
    assert isinstance(js["findings"], list) and js["findings"]


def test_an_alpha_at_the_edge_of_the_grid_is_flagged():
    mols, names, y = _series(14)
    d = diagnose(mols, names, y, routes=("aligned",), folds=4, alphas=(0.5, 1.0))
    codes = {f.code for f in d.findings}
    assert "alpha-edge" in codes, "with two alphas the choice is always at an edge"
    flag = next(f for f in d.findings if f.code == "alpha-edge")
    assert flag.level == "warn" and "widen" in flag.text


def test_a_leak_that_only_several_columns_together_reconstruct_is_caught_by_the_outcome():
    """No per-column correlation test can see it, so the outcome is judged instead.

    The corrosion files carry the potency as `activity`, as `mwila_pooled_raw` and as two
    laboratory-adjusted variants; excluding the first two by name and by correlation still left
    a descriptor baseline at q2 = 0.99, because the remainder reconstruct it between them.
    """
    mols, names, y = _series(16)
    rng = np.random.default_rng(3)
    for m, v in zip(mols, y):                      # two columns, neither alone decisive
        m.SetProp("half_a", f"{0.5 * v + 0.4 * rng.normal():.6f}")
        m.SetProp("half_b", f"{0.5 * v - 0.4 * rng.normal():.6f}")
    d = diagnose(mols, names, y, routes=("aligned", "descriptors"), folds=4, alphas=(0.3, 1.0))
    desc = d.routes["descriptors"]
    if desc.q2 >= 0.95:
        assert any(f.code == "descriptor-leak" for f in d.findings)
        assert d.recommended != "descriptors"
        assert getattr(desc, "leaked", False)


def test_the_frame_test_declines_rather_than_guessing_when_it_cannot_tell():
    """Its two failure modes act in opposite directions, so it is allowed a third answer.

    A small symmetric framework matches a dozen ways and two rings near the origin coincide by
    chance; a framework that is a small part of the molecule may match a different copy in each.
    So a near-coincidence is a verdict on its own, a large deviation only when the framework is
    a large part of the molecule, and otherwise the answer is "look at it".
    """
    from cmfqsar.diagnose import _superimposed_finding

    mols, names, _y = _series(12, congeneric=True)
    core = Chem.MolFromSmiles("c1ccccc1")
    f = _superimposed_finding(mols, core)
    assert f.level == "ok" and f.numbers["core_rmsd"] < 0.5

    moved = [Chem.Mol(m) for m in mols]
    for i, m in enumerate(moved[1:], 1):            # scatter them, keeping the framework large
        conf = m.GetConformer()
        for a in range(m.GetNumAtoms()):
            p = conf.GetAtomPosition(a)
            conf.SetAtomPosition(a, (p.x + 3.0 * i, p.y, p.z))
    big = Chem.MolFromSmiles("c1ccc(cc1)C")
    f2 = _superimposed_finding(moved, big)
    assert f2.level in ("warn", "note") and f2.numbers["core_rmsd"] > 1.0
