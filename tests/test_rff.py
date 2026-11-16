"""Tests of cmfqsar.rff: random Fourier features of the field kernel."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar import lowrank as LR  # noqa: E402
from cmfqsar import rff  # noqa: E402
from cmfqsar.fields import EFIELD_FIELD, AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import kernel_matrices, normalize_kernels, self_kernels  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, r2_score  # noqa: E402

SMILES = ("c1ccccc1N", "c1ccccc1O", "c1ccccc1C", "c1ccccc1F", "c1ccccc1Cl", "c1ccccc1C#N",
          "Nc1ccc(O)cc1", "Cc1ccc(F)cc1")


def _mol(smiles, seed):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _table(fs, n=len(SMILES)):
    mols = [_mol(s, i + 1) for i, s in enumerate(SMILES[:n])]
    return AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)])


def _fs(physchem=("electrostatic", "steric", "hydrophobic"), alpha=0.5, **kw):
    return FieldSet.create(physchem=list(physchem), alpha=alpha, charge_model="gasteiger",
                           param_scheme="rdkit", **kw)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("names", [("electrostatic", "steric", "hydrophobic"),
                                   ("electrostatic", EFIELD_FIELD)])
def test_the_lattice_reproduces_the_exact_kernel(names):
    """By Poisson summation the lattice sum is the kernel of the periodically repeated
    molecules, so with a large enough period it is the exact kernel to the tolerance."""
    fs = _fs(names, alpha=0.7)
    T = _table(fs)
    K = kernel_matrices(T, T, fs.specs)
    ff = rff.build(fs, T, sampler="lattice", normalize=False)
    Ka = ff.field_kernels(T)
    for f in range(len(fs.specs)):
        assert _rel(Ka[f], K[f]) < 1e-4, fs.specs[f].name


def test_the_lattice_reproduces_a_probe_field():
    fs = _fs(("steric",), alpha=0.5, probes=["water"], probe_terms=4)
    T = _table(fs, 4)
    K = kernel_matrices(T, T, fs.specs)
    Ka = rff.build(fs, T, sampler="lattice", normalize=False).field_kernels(T)
    for f in range(len(fs.specs)):
        assert _rel(Ka[f], K[f]) < 1e-4, fs.specs[f].name


@pytest.mark.parametrize("sampler", ["mc", "orthogonal", "qmc"])
def test_random_frequencies_converge_to_the_exact_kernel(sampler):
    fs = _fs()
    T = _table(fs)
    K = kernel_matrices(T, T, fs.specs)
    errs = []
    for D in (32, 2048):
        Ka = rff.build(fs, T, D, sampler, seed=3, normalize=False).field_kernels(T)
        errs.append(max(_rel(Ka[f], K[f]) for f in range(len(fs.specs))))
    assert errs[1] < errs[0]
    assert errs[1] < 0.06


def test_the_estimate_is_unbiased():
    """The mean of many independent estimates is closer to the exact kernel than any one."""
    fs = _fs(("steric",))
    T = _table(fs, 5)
    K = kernel_matrices(T, T, fs.specs)[0]
    est = [rff.build(fs, T, 64, "mc", seed=s, normalize=False).field_kernels(T)[0]
           for s in range(60)]
    one = np.mean([_rel(e, K) for e in est])
    mean = _rel(np.mean(est, axis=0), K)
    assert mean < one / 4


def test_one_width_gives_every_frequency_the_same_weight():
    """For a field of one width the density matches the spectrum, as in Rahimi and Recht."""
    fs = _fs(("electrostatic",), alpha=0.8)
    T = _table(fs, 2)
    ff = rff.build(fs, T, 50, "mc", seed=0, normalize=False)
    w2 = (ff.omegas[0] ** 2).sum(axis=1)
    amp = ff._amplitude(0, T, {}, w2, ff.logq[0])
    assert np.allclose(amp, amp[0, 0])
    # a(omega)^2 q(omega) = (pi / b)^{3/2} / D
    assert amp[0, 0] ** 2 == pytest.approx((np.pi / 0.8) ** 1.5 / 50, rel=1e-10)


def test_orthogonal_frequencies_come_in_orthogonal_triples():
    W, _ = rff.frequencies(9, 1.0, "orthogonal", seed=1)
    for k in range(3):
        B = W[3 * k:3 * k + 3]
        G = B @ B.T
        assert np.allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-12)


def test_a_common_shift_of_every_molecule_changes_nothing():
    fs = _fs()
    T = _table(fs, 5)
    ff = rff.build(fs, T, 128, "qmc", seed=0, normalize=False)
    K1 = ff.field_kernels(T)
    T2 = AtomTable(coords=T.coords + np.array([3.0, -7.0, 11.0]), weights=T.weights,
                   radii=T.radii, offsets=T.offsets, names=T.names, molecules=T.molecules)
    assert np.allclose(ff.field_kernels(T2), K1, rtol=1e-9, atol=1e-12)


def test_the_cosine_normalisation_uses_the_exact_self_kernel():
    fs = _fs()
    T = _table(fs, 5)
    ff = rff.build(fs, T, 256, "qmc", seed=0, normalize=True)
    raw = rff.build(fs, T, 256, "qmc", seed=0, normalize=False).field_kernels(T)
    d = self_kernels(T, fs.specs)
    assert np.allclose(ff.field_kernels(T), normalize_kernels(raw, d, d))


def test_the_low_rank_fit_runs_on_fourier_features():
    """The feature map has the duck type of lowrank.LowRank, and on the lattice the low-rank fit
    is the exact model."""
    fs = _fs(alpha=0.7)
    T = _table(fs)
    y = np.linspace(-1.0, 1.0, T.n_mol) + 0.3 * np.sin(np.arange(T.n_mol))
    m = CMFModel(fieldset=fs.copy(), method="krr")
    K_fields = kernel_matrices(T, T, fs.specs)
    m.set_training_kernels(T, K_fields)
    K = m.joint_kernel(K_fields, T.descriptors)
    scale = m._low_rank_scale(T)
    ff = rff.build(fs, T, sampler="lattice", scale=scale, normalize=m.normalize)
    pred, _ = LR.cross_validate(ff, T, y, 0.1, folds=4, seed=0)
    low = LR.build(T, fs, scale=scale, normalize=m.normalize)
    ref, _ = LR.cross_validate(low, T, y, 0.1, folds=4, seed=0)
    assert np.allclose(pred, ref, atol=1e-4)
    exact = cv_predict(K, y, "krr", {"lam": 0.1}, folds=4, seed=0)
    assert np.allclose(pred, exact, atol=1e-4)


def test_unknown_sampler_is_refused():
    fs = _fs(("steric",))
    T = _table(fs, 2)
    with pytest.raises(ValueError):
        rff.build(fs, T, 8, "grid")
