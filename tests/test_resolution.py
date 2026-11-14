"""A kernel that knows the resolution of its description (plan item MS4).

The tests check that a bead of one atom is that atom's Gaussian exactly, that a bead of several
atoms has the closed-form overlap of its moment-matched Gaussian, that the mapping rules give
every atom one bead, that the width mixture prefers the kernel the target came from, and the
reading rules of the benchmark.
"""
import json
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar import resolution as R  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, MoleculeFields, build_fields  # noqa: E402

FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
SMILES = ["CC(=O)NCc1ccccc1", "OCCN(C)C(=O)c1ccncc1", "CCOC(=O)C1CCNCC1", "Nc1ccc(Cl)cc1O",
          "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "COc1ccc2[nH]ccc2c1"]


def _fields(smiles=SMILES, alpha=1.0):
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger")
    out = []
    for k, s in enumerate(smiles):
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        p = AllChem.ETKDGv3()
        p.randomSeed = k + 1
        assert AllChem.EmbedMolecule(m, p) == 0
        out.append(build_fields(m, fs, f"m{k}"))
    return fs, out


# ------------------------------------------------------------------------------------ beads


def test_a_bead_of_one_atom_is_that_atoms_gaussian():
    fs, mf = _fields()
    table = AtomTable.from_molecules(mf)
    beads = [R.beads_of(m, "atoms") for m in mf]
    for alpha in (0.2, 1.0, 4.0):
        atoms = R.cosine_field_kernels(table, R._specs_at(fs, alpha))
        assert np.allclose(R.bead_kernels(beads, fs, alpha), atoms, rtol=0, atol=1e-12)


def _single_bead(name, x, w, spread, r2=1.0):
    return R.Beads(name=name, coords=np.asarray([x], float), weights=np.asarray([w], float),
                   spread=np.asarray([spread]), radius2=np.asarray([r2]), groups=[[0]])


def test_a_bead_has_the_overlap_of_its_moment_matched_gaussian():
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5)
    a = _single_bead("a", [0.0, 0.0, 0.0], [0.7, 2.0], 0.3, r2=2.4)
    b = _single_bead("b", [1.1, -0.4, 0.8], [-0.2, 3.0], 0.9, r2=3.0)
    K = R.bead_kernels([a], fs, 0.5, [b], normalise=False)
    d2 = float(np.sum((a.coords[0] - b.coords[0]) ** 2))

    def overlap(wa, wb, ba, bb):
        bs = ba + bb
        return wa * wb * (2 * np.pi / bs) ** 1.5 * np.exp(-ba * bb / (2 * bs) * d2)

    # an ordinary field: the atom's variance 1/alpha plus the spread
    be_a, be_b = 1 / (1 / 0.5 + 0.3), 1 / (1 / 0.5 + 0.9)
    assert K[0, 0, 0] == pytest.approx(overlap(0.7, -0.2, be_a, be_b), rel=1e-12)
    # the steric field: the atom's variance R^2/alpha plus the spread
    bs_a, bs_b = 1 / (2.4 / 0.5 + 0.3), 1 / (3.0 / 0.5 + 0.9)
    assert K[1, 0, 0] == pytest.approx(overlap(2.0, 3.0, bs_a, bs_b), rel=1e-12)


def test_the_beads_keep_the_total_weight_and_find_the_centre():
    _fs, mf = _fields()
    for m in mf:
        for rule in R.RULES:
            b = R.beads_of(m, rule)
            assert np.allclose(b.weights.sum(axis=0), m.weights.sum(axis=0))
            for g, c in zip(b.groups, b.coords):
                assert np.allclose(c, m.coords[g].mean(axis=0))


def test_every_atom_is_in_one_bead_and_the_rules_hold():
    _fs, mf = _fields()
    for m in mf:
        heavy = [i for i, s in enumerate(m.symbols) if s != "H"]
        bonded = {tuple(sorted(map(int, p))) for p in np.asarray(m.bonds).reshape(-1, 2)}
        for rule in R.RULES:
            groups = R.bead_groups(m, rule)
            flat = sorted(i for g in groups for i in g)
            assert flat == list(range(m.n_atoms))
            assert R.bead_groups(m, rule) == groups            # the same every time
        assert len(R.bead_groups(m, "united")) == len(heavy)
        for g in R.bead_groups(m, "four"):
            hv = [i for i in g if m.symbols[i] != "H"]
            assert 1 <= len(hv) <= R.FOUR
            # the heavy atoms of a bead are connected through bonds inside the bead
            seen, todo = {hv[0]}, [hv[0]]
            while todo:
                cur = todo.pop()
                for j in hv:
                    if j not in seen and tuple(sorted((cur, j))) in bonded:
                        seen.add(j)
                        todo.append(j)
            assert seen == set(hv)


def test_coarser_beads_are_fewer():
    _fs, mf = _fields()
    for m in mf:
        n = [len(R.bead_groups(m, rule)) for rule in ("atoms", "united", "four")]
        assert n[0] > n[1] > n[2]


def test_what_beads_are_not_built_for():
    fs, mf = _fields()
    with pytest.raises(ValueError, match="not one of"):
        R.bead_groups(mf[0], "martini")
    ens = MoleculeFields(name="e", coords=np.zeros((2, 3)), weights=np.ones((2, 5)),
                         radii=np.ones(2), symbols=["C", "C"], atom_types=["C", "C"],
                         bonds=np.zeros((0, 2), int), charges=np.zeros(2))
    ens.n_conformers = 2
    ens.atoms_per_conformer = 1
    with pytest.raises(ValueError, match="ensemble"):
        R.bead_groups(ens, "four")
    efs = FieldSet.create(physchem=["efield"])
    assert [s.kind for s in efs.specs] == ["efield"]
    beads = [R.beads_of(m, "united") for m in mf[:2]]
    with pytest.raises(ValueError, match="ordinary and steric"):
        R.bead_kernels(beads, efs, 1.0)


# ----------------------------------------------------------------------------------- widths


def test_the_width_stack_and_its_mixture():
    fs, mf = _fields()
    table = AtomTable.from_molecules(mf)
    stack = R.width_kernels(table, fs, (0.2, 1.0))
    assert stack.shape == (2, len(mf), len(mf))
    assert np.allclose(np.diagonal(stack, axis1=1, axis2=2), 5.0)   # five fields, each cosine
    assert np.allclose(R.mixed_kernel(stack, [1.0, 0.0]), stack[0])


def test_the_evidence_prefers_the_width_the_target_came_from():
    rng = np.random.default_rng(3)
    n = 60
    X = rng.normal(size=(n, 3))
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    good = np.exp(-0.5 * d2)
    noise = rng.normal(size=(n, n))
    bad = noise @ noise.T / n + np.eye(n)
    bad /= np.sqrt(np.outer(np.diag(bad), np.diag(bad)))
    y = np.linalg.cholesky(good + 1e-8 * np.eye(n)) @ rng.normal(size=n)
    h = R.fit_width_mixture(np.stack([good, bad]), y)
    assert h[0] > h[1]


def test_the_two_agreement_readings():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 8))
    K = A @ A.T
    assert R.kernel_correlation(K, K) == pytest.approx(1.0)
    assert R.centred_alignment(K, 2.0 * K) == pytest.approx(1.0)
    L = rng.normal(size=(8, 8))
    L = L @ L.T
    assert R.centred_alignment(K, L) == pytest.approx(R.centred_alignment(L, K))


# ------------------------------------------------------------------------ the benchmark


def _bench():
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    import run_resolution as RR
    return RR


def _widths_row(name, mean, lo, hi):
    g = {"mean": mean, "lo": lo, "hi": hi}
    return {"set": name, "n": 50, "alphas": [0.5, 1.0], "mean_width_weights": [0.3, 0.7],
            "models": {"one width": {"q2_mean": 0.5},
                       "equal widths": {"q2_mean": 0.5, "q2_gain_vs_one_width": g},
                       "evidence widths": {"q2_mean": 0.5, "q2_gain_vs_one_width": g}}}


def test_the_gate_of_the_widths():
    RR = _bench()
    ok = [_widths_row("A", 0.01, -0.02, 0.04), _widths_row("B", -0.015, -0.04, 0.01)]
    assert RR.gate(ok)["met"]
    below = ok + [_widths_row("C", -0.05, -0.08, -0.01)]
    g = RR.gate(below)
    assert not g["met"] and g["interval_wholly_below_zero"] == ["C"]
    mean_loss = [_widths_row("A", -0.02, -0.05, 0.01), _widths_row("B", -0.03, -0.06, 0.01)]
    assert not RR.gate(mean_loss)["met"]


def test_the_table_writes_the_gate(tmp_path):
    RR = _bench()
    with open(tmp_path / "widths_ACE.json", "w") as fh:
        json.dump(_widths_row("ACE", 0.01, -0.02, 0.04), fh)
    text = RR.write_table(str(tmp_path))
    assert "**met**" in text and (tmp_path / "table.md").exists()
