"""GRID-like probe fields: the expansion, the kernel it implies, and what it refuses to do."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar.fields import AtomTable, FieldSet, build_fields
from cmfqsar.kernels import kernel_matrices
from cmfqsar.model import CMFModel
from cmfqsar.probes import (PROBES, attraction, basis_widths, coefficients, fit, probe,
                            probe_names)


def _mol(smiles, seed=7, name="m"):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    m.SetProp("_Name", name)
    return m


def test_attraction_is_negative_monotone_and_tapered():
    r = np.linspace(0.5, 9.0, 400)
    e = attraction(r, probe("water"), "C")
    assert e.max() <= 1e-12, "the repulsive wall is deliberately not represented"
    assert np.all(np.diff(e) >= -1e-9), "capped terms must not deepen with distance"
    assert abs(e[-1]) < 1e-12, "the taper has to reach zero"
    assert e.min() < -0.05, "there has to be some attraction to represent"


def test_a_complementary_hydrogen_bond_deepens_the_well():
    r = np.linspace(0.5, 9.0, 400)
    plain = attraction(r, probe("carbonyl"), "N").min()
    donor = attraction(r, probe("carbonyl"), "N", hbond=1.0).min()   # acceptor probe, donor atom
    wrong = attraction(r, probe("carbonyl"), "N", hbond=-1.0).min()  # two acceptors: nothing
    assert donor < plain - 0.5
    assert wrong == pytest.approx(plain)


def test_the_expansion_reproduces_the_attraction_it_stands_for():
    """Within the bound the module documents, wherever there is an interaction worth fitting.

    Curves below 0.1 kcal/mol are skipped on purpose: they are the near-cancellation of a
    dispersion well against an electrostatic repulsion, they change sign, and a relative error
    computed against a norm that is almost zero says nothing useful.
    """
    r = np.linspace(0.2, 9.0, 700)
    vol = r ** 2
    worst, tested = 0.0, 0
    for name in probe_names():
        p = PROBES[name]
        for sym, q, hb in (("C", -0.05, 0.0), ("H", 0.06, 0.0), ("O", -0.4, -1.0),
                           ("N", -0.35, 0.0), ("S", -0.1, 0.0)):
            e = attraction(r, p, sym, charge=q, hbond=hb)
            if np.max(np.abs(e)) < 0.1:
                continue
            f = fit(p, sym, charge=q, hbond=hb)
            rel = np.sqrt(np.sum(vol * (f(r) - e) ** 2) / np.sum(vol * e ** 2))
            worst = max(worst, float(rel))
            tested += 1
    assert tested >= 15, "most combinations should carry a real interaction"
    assert worst < 0.30, f"worst relative error {worst:.1%}"


def test_the_kernel_matches_numerical_integration_of_the_fields():
    """The purpose of the whole construction: the overlap stays a closed form over atom pairs."""
    fs = FieldSet.create(physchem=[], probes=["water"], alpha=0.5)
    mols = [_mol("c1ccccc1O", 7, "phenol"), _mol("c1ccccc1N", 11, "aniline")]
    mf = [build_fields(m, fs, m.GetProp("_Name")) for m in mols]
    table = AtomTable.from_molecules(mf)
    K = kernel_matrices(table, table, fs.specs)[0]

    widths = basis_widths(mf[0].probe_coeffs["probe:water"].shape[1])

    def field_on(points, m):
        C = m.probe_coeffs["probe:water"]
        out = np.zeros(len(points))
        for l, xyz in enumerate(m.coords):
            d2 = np.sum((points - xyz) ** 2, axis=1)
            for k, b in enumerate(widths):
                out += C[l, k] * np.exp(-0.5 * b * d2)
        return out

    lo = np.min(np.vstack([m.coords for m in mf]), axis=0) - 10
    hi = np.max(np.vstack([m.coords for m in mf]), axis=0) + 10
    step = 0.45
    axes = [np.arange(lo[k], hi[k], step) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    vals = [field_on(grid, m) for m in mf]
    num = np.array([[float(np.sum(vals[i] * vals[j]) * step ** 3) for j in range(2)]
                    for i in range(2)])
    assert np.allclose(K, num, rtol=2e-4), f"analytic {K} against quadrature {num}"


def test_more_terms_do_not_change_the_field_much():
    """The expansion is a numerical device: the kernel it gives must not depend on K."""
    mols = [_mol("c1ccccc1O", 7), _mol("CCO", 3)]
    out = {}
    for k in (3, 5):
        fs = FieldSet.create(physchem=[], probes=["water"], alpha=0.5, probe_terms=k)
        table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}")
                                          for i, m in enumerate(mols)])
        K = kernel_matrices(table, table, fs.specs)[0]
        out[k] = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))      # compare the shape, not the scale
    assert np.allclose(out[3], out[5], atol=0.05)


def test_a_probe_field_fits_a_model_and_keeps_its_co_field():
    mols = [_mol(s, i, f"m{i}") for i, s in enumerate(["c1ccccc1O", "c1ccccc1N", "c1ccccc1C",
                                                       "c1ccccc1Cl", "c1ccccc1F", "c1ccccc1Br"])]
    y = np.array([1.0, 1.4, 0.3, 0.8, 0.6, 1.1])
    fs = FieldSet.create(physchem=["steric"], probes=["water"], alpha=0.5)
    mf = [build_fields(m, fs, m.GetProp("_Name")) for m in mols]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.01})
    model.fit(table, y, cv_folds=0)
    assert np.isfinite(model.predict(table)).all()
    # the co-field of a probe field exists: one Gaussian per atom per term
    centres, coeffs, widths = model.cofield_coefficients(1)
    k = mf[0].probe_coeffs["probe:water"].shape[1]
    assert len(centres) == len(table.coords) * k
    assert np.isfinite(coeffs).all() and (widths > 0).all()


def test_an_unknown_probe_is_named_rather_than_guessed():
    with pytest.raises(KeyError, match="unknown probe"):
        probe("plutonium")


def test_coefficients_are_one_row_per_atom():
    C = coefficients(["C", "O", "H"], [0.0, -0.4, 0.2], [0.0, -1.0, 1.0], probe("water"),
                     n_terms=4)
    assert C.shape == (3, 4)
    assert np.isfinite(C).all()
