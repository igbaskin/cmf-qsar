"""Flexible alignment by confidence: the torsions, the chain rule, and the strain prior."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar.fields import AtomTable, FieldSet, build_fields
from cmfqsar.flexalign import (FlexibleAligner, Torsion, _Strain, apply_torsions,
                               flexible_align, rotatable_torsions, torsion_gradient)
from cmfqsar.kernels import atom_kernel_block, kernel_gradient
from cmfqsar.model import CMFModel


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    m.SetProp("_Name", smiles)
    return m


def _fitted(fields=("steric", "electrostatic"), alpha=0.5, n=10):
    smiles = ["CCCCO", "CCCCN", "CCCOC", "CCCCC", "CCOCC", "CCCCF", "CCCCCl", "CCC(C)O",
              "CCCC=O", "CCCCBr"][:n]
    mols = [_mol(s, seed=i + 1) for i, s in enumerate(smiles)]
    names = [f"m{i}" for i in range(len(mols))]
    fs = FieldSet.create(physchem=list(fields), alpha=alpha)
    mf = [build_fields(m, fs, nm) for m, nm in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    y = np.array([4.0 + 0.2 * m.GetNumHeavyAtoms() for m in mols])
    model = CMFModel(fieldset=fs, method="gp", normalize=True, params={"lam": 0.1})
    model.fit(table, y, cv_folds=0)
    return model, fs, mols, names


def test_rotatable_bonds_move_the_smaller_side_and_are_ordered_from_the_root():
    m = _mol("CCCCCCOc1ccccc1")
    ts = rotatable_torsions(m)
    assert ts, "a hexyl phenyl ether has rotatable bonds"
    for t in ts:
        other = m.GetNumAtoms() - len(t)
        assert len(t) <= other, "the larger side must stay put"
        assert t.pivot not in set(t.moving.tolist())
        assert t.tip in set(t.moving.tolist())
    sizes = [len(t) for t in ts]
    assert sizes == sorted(sizes, reverse=True), "root-outward order is what the chain rule needs"


def test_a_ring_bond_never_turns():
    m = _mol("c1ccccc1")
    assert rotatable_torsions(m) == []
    m2 = _mol("C1CCCCC1C")
    for t in rotatable_torsions(m2):
        bond = m2.GetBondBetweenAtoms(t.pivot, t.tip)
        assert bond is not None and not bond.IsInRing()


def test_turning_a_torsion_keeps_the_molecule_intact():
    """Bond lengths and the angles inside each rigid part cannot change when a bond turns."""
    m = _mol("CCCCCCOc1ccccc1")
    ts = rotatable_torsions(m)
    X0 = np.asarray(m.GetConformer().GetPositions(), float)
    X = apply_torsions(X0, ts, [0.7] * len(ts))
    assert not np.allclose(X, X0)
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d0 = np.linalg.norm(X0[i] - X0[j])
        d1 = np.linalg.norm(X[i] - X[j])
        assert d1 == pytest.approx(d0, abs=1e-9), "a bond length changed"


def test_zero_angles_change_nothing_and_a_full_turn_returns():
    m = _mol("CCCCO")
    ts = rotatable_torsions(m)
    X0 = np.asarray(m.GetConformer().GetPositions(), float)
    np.testing.assert_allclose(apply_torsions(X0, ts, [0.0] * len(ts)), X0, atol=1e-12)
    full = apply_torsions(X0, ts, [2.0 * np.pi] * len(ts))
    np.testing.assert_allclose(full, X0, atol=1e-8)


def test_the_torsion_chain_rule_matches_finite_differences():
    """The whole point of the analytic gradient: it must be the derivative it claims to be."""
    model, fs, mols, names = _fitted()
    rd = _mol("CCCCCCOc1ccccc1", seed=7)
    mf = build_fields(rd, fs, "probe")
    ts = rotatable_torsions(rd)
    assert len(ts) >= 3
    X0 = np.asarray(rd.GetConformer().GetPositions(), float)
    u = np.ones((len(fs.specs), model.train.n_mol))

    def value(thetas):
        X = apply_torsions(X0, ts, thetas)
        blk = atom_kernel_block(X, mf.weights, mf.radii, model.train, fs.specs,
                                ci=mf.probe_coeffs)
        return float((u * blk.sum(axis=1)).sum())

    theta = np.linspace(0.1, 0.5, len(ts))
    X = apply_torsions(X0, ts, theta)
    g = kernel_gradient(X, mf.weights, mf.radii, model.train, fs.specs, u, ci=mf.probe_coeffs)
    analytic = torsion_gradient(X, ts, g)
    h = 1e-6
    numeric = np.zeros(len(ts))
    for k in range(len(ts)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        numeric[k] = (value(tp) - value(tm)) / (2 * h)
    scale = max(np.abs(numeric).max(), 1e-12)
    assert np.abs(analytic - numeric).max() / scale < 2e-4


def test_the_full_objective_gradient_matches_finite_differences():
    model, fs, mols, names = _fitted()
    rd = _mol("CCCCCCOc1ccccc1", seed=7)
    mf = build_fields(rd, fs, "probe")
    aligner = FlexibleAligner(model, max_rmsd=None, energy_weight=0.0)
    ts = rotatable_torsions(rd)
    strain = _Strain(rd, slack=10.0, weight=0.0)
    X0 = np.asarray(mf.coords, float)
    c = X0.mean(axis=0)
    sp, kself = aligner._self_terms(mf)
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.normal(scale=0.2, size=3), rng.normal(scale=0.3, size=3),
                        rng.normal(scale=0.2, size=len(ts))])
    val, grad = aligner.objective_flex(mf, rd, X0, c, ts, strain, sp, kself, p)
    h = 1e-6
    numeric = np.zeros_like(grad)
    for k in range(len(p)):
        pp, pm = p.copy(), p.copy()
        pp[k] += h
        pm[k] -= h
        numeric[k] = (aligner.objective_flex(mf, rd, X0, c, ts, strain, sp, kself, pp)[0]
                      - aligner.objective_flex(mf, rd, X0, c, ts, strain, sp, kself, pm)[0]) / (2 * h)
    scale = max(np.abs(numeric).max(), 1e-12)
    assert np.abs(grad - numeric).max() / scale < 1e-3


def test_the_strain_prior_is_one_sided_and_bites_when_the_geometry_is_broken():
    rd = _mol("CCCCCCOc1ccccc1")
    strain = _Strain(rd, slack=5.0, weight=1.0)
    assert strain.ok
    X0 = np.asarray(rd.GetConformer().GetPositions(), float)
    pen, grad = strain.penalty(X0)
    assert pen == pytest.approx(0.0), "the starting conformation must cost nothing"
    assert np.allclose(grad, 0.0)
    squashed = X0 * 0.7                      # atoms driven into each other
    pen2, grad2 = strain.penalty(squashed)
    assert pen2 > 0.0 and np.abs(grad2).max() > 0.0


def test_the_flexible_search_lowers_the_variance_and_keeps_the_molecule_whole():
    model, fs, mols, names = _fitted()
    rd = _mol("CCCCCCOc1ccccc1", seed=7)
    mf = build_fields(rd, fs, "probe")
    before = np.asarray(mf.coords, float).copy()
    out = flexible_align([mf], [rd], model, max_rmsd=None, restarts=1, seed=0)[0]
    assert out.variance_after <= out.variance_before + 1e-9
    assert out.n_eval > 0
    moved = np.asarray(mf.coords, float)
    assert not np.allclose(moved, before)
    for b in rd.GetBonds():                  # the conformation changed, the molecule did not
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        assert np.linalg.norm(moved[i] - moved[j]) == pytest.approx(
            float(np.linalg.norm(before[i] - before[j])), abs=1e-6)


def test_a_rigid_molecule_reduces_to_the_rigid_body_search():
    """With no rotatable bonds the flexible search must do exactly what the old one does."""
    model, fs, mols, names = _fitted()
    rd = _mol("c1ccccc1", seed=3)
    assert rotatable_torsions(rd) == []
    mf_a, mf_b = build_fields(rd, fs, "a"), build_fields(rd, fs, "b")
    flex = FlexibleAligner(model, max_rmsd=None, energy_weight=0.0)
    rigid_out = flex.align(mf_a, restarts=1, seed=0)
    flex_out = flex.align_flexible(mf_b, rd, restarts=1, seed=0)
    assert flex_out.variance_after == pytest.approx(rigid_out.variance_after, rel=1e-6)
    np.testing.assert_allclose(mf_a.coords, mf_b.coords, atol=1e-6)
