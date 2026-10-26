"""Protonation ensembles: members that are different molecules (plan item D4).

Two things are tested here.  First the general builder, :func:`combine_members`: it has to reduce
to the conformational tiling it generalises, atom for atom, because the 4D mode is already
measured and must not move.  Second the protonation ensemble itself: the weights have to be the
Henderson-Hasselbalch fractions, the kernel of two ensembles has to be the weighted mean of the
kernels of their member pairs -- the property the whole construction rests on -- and the heavy
atoms have to stay where the alignment put them.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import (AtomTable, FieldSet, build_fields, combine_members,  # noqa: E402
                            conformer_weights, ensemble_fields)
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.protonation import (Ensemble, State, _fraction, available,  # noqa: E402
                                 ensemble_table, mixture_share, protonation_fields,
                                 protonation_states)

SMILES = {
    "aniline": "Nc1ccccc1",
    "toluidine": "Nc1ccc(C)cc1",
    "acetic acid": "CC(=O)O",
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "benzimidazole": "Cc1nc2ccccc2[nH]1",
}
#: the aniline pKa Dimorphite-DL carries, so that a pH equal to it gives a half-and-half mixture
ANILINE_PKA = 3.9

needs_dimorphite = pytest.mark.skipif(not available(),
                                      reason="Dimorphite-DL is not installed")


def _mol(smiles, seed=1, name="m"):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    m.SetProp("_Name", name)
    return m


def _conformers(smiles, n=3, seed=5):
    from cmfqsar.conformers import generate_conformers
    return generate_conformers(_mol(smiles, seed=seed, name="c"), n=n, seed=seed + 1,
                               prune_rmsd=1.0)


def _single(mol, conf_id, fieldset):
    one = Chem.Mol(mol)
    c = Chem.Conformer(one.GetConformer(conf_id))
    one.RemoveAllConformers()
    one.AddConformer(c, assignId=True)
    return build_fields(one, fieldset, name="c", _single=True)


# ------------------------------------- the general builder reduces to the conformational one


def test_the_general_builder_reproduces_the_conformer_tiling_exactly():
    """The gate the item sets: the 4D numbers must not move by a single bit."""
    mol = _conformers(SMILES["aniline"], n=4)
    fs_e = FieldSet.create(alpha=0.6, conformers="ensemble")
    fs_1 = FieldSet.create(alpha=0.6)
    tiled = ensemble_fields(mol, fs_e, name="x")
    w = conformer_weights(mol, fs_e)
    use = [i for i, x in enumerate(w) if x > 0]
    members = [_single(mol, i, fs_1) for i in use]
    general = combine_members(members, [w[i] for i in use], kind="conformers", name="x")
    assert np.array_equal(general.coords, tiled.coords)
    assert np.array_equal(general.weights, tiled.weights)
    assert np.array_equal(general.radii, tiled.radii)
    assert np.array_equal(general.bonds, tiled.bonds)
    assert np.array_equal(general.charges, tiled.charges)
    assert list(general.symbols) == list(tiled.symbols)
    assert list(general.atom_types) == list(tiled.atom_types)
    assert general.n_conformers == tiled.n_conformers
    assert general.atoms_per_conformer == tiled.atoms_per_conformer
    assert np.allclose(general.conformer_weights, tiled.conformer_weights)


def test_the_two_builders_give_the_same_kernel():
    mol = _conformers(SMILES["toluidine"], n=3)
    fs_e = FieldSet.create(alpha=0.5, conformers="ensemble")
    fs_1 = FieldSet.create(alpha=0.5)
    w = conformer_weights(mol, fs_e)
    use = [i for i, x in enumerate(w) if x > 0]
    tiled = ensemble_fields(mol, fs_e, name="x")
    general = combine_members([_single(mol, i, fs_1) for i in use], [w[i] for i in use])
    a = kernel_matrices(AtomTable.from_molecules([tiled]), AtomTable.from_molecules([tiled]),
                        fs_1.specs)
    b = kernel_matrices(AtomTable.from_molecules([general]), AtomTable.from_molecules([general]),
                        fs_1.specs)
    assert a == pytest.approx(b, rel=1e-12, abs=1e-15)


def test_equal_members_keep_the_old_layout_and_the_old_per_atom_sum():
    mol = _conformers(SMILES["aniline"], n=3)
    fs = FieldSet.create(alpha=0.5, conformers="ensemble")
    e = ensemble_fields(mol, fs, name="x")
    assert e.same_atoms and e.member_sizes is None and e.member_atom_map is None
    assert list(e.sizes) == [e.atoms_per_conformer] * e.n_conformers
    v = np.arange(e.n_atoms, dtype=float)
    expect = v.reshape(e.n_conformers, e.atoms_per_conformer).sum(axis=0)
    assert np.array_equal(e.per_atom(v), expect)
    assert e.unmapped_share(v) == 0.0
    assert e.member_slice(1) == slice(e.atoms_per_conformer, 2 * e.atoms_per_conformer)


def test_a_single_member_is_returned_as_itself():
    fs = FieldSet.create(alpha=0.5)
    one = build_fields(_mol(SMILES["benzene"]), fs, "b", _single=True)
    out = combine_members([one], [1.0])
    assert out is one
    # a member of weight zero is not a member
    two = build_fields(_mol(SMILES["pyridine"]), fs, "p", _single=True)
    assert combine_members([one, two], [1.0, 0.0]) is one


def test_the_weights_are_normalised_and_scale_the_atom_weights():
    fs = FieldSet.create(alpha=0.5)
    a = build_fields(_mol(SMILES["benzene"], seed=1), fs, "a", _single=True)
    b = build_fields(_mol(SMILES["benzene"], seed=2), fs, "b", _single=True)
    out = combine_members([a, b], [3.0, 1.0])
    assert np.allclose(out.conformer_weights, [0.75, 0.25])
    assert np.allclose(out.weights[:a.n_atoms], a.weights * 0.75)
    assert np.allclose(out.weights[a.n_atoms:], b.weights * 0.25)


def test_bad_members_are_refused_with_a_reason():
    fs = FieldSet.create(alpha=0.5)
    small = build_fields(_mol(SMILES["benzene"]), fs, "a", _single=True)
    big = build_fields(_mol(SMILES["toluidine"]), fs, "b", _single=True)
    with pytest.raises(ValueError, match="at least one member"):
        combine_members([], [])
    with pytest.raises(ValueError, match="weights"):
        combine_members([small, big], [1.0])
    with pytest.raises(ValueError, match="non-negative"):
        combine_members([small, big], [1.0, -1.0])
    with pytest.raises(ValueError, match="atom_maps"):
        combine_members([small, big], [1.0, 1.0])
    with pytest.raises(ValueError, match="atom map"):
        combine_members([small, big], [1.0, 1.0],
                        atom_maps=[np.arange(small.n_atoms), np.arange(3)])
    other = build_fields(_mol(SMILES["benzene"]), FieldSet.create(physchem=["steric"]), "c",
                         _single=True)
    with pytest.raises(ValueError, match="same field set"):
        combine_members([small, other], [1.0, 1.0])


def test_members_with_their_own_atoms_report_their_sizes_and_place_every_atom():
    fs = FieldSet.create(alpha=0.5)
    a = build_fields(_mol(SMILES["benzene"]), fs, "a", _single=True)
    b = build_fields(_mol(SMILES["toluidine"]), fs, "b", _single=True)
    # a deliberately crude map: the first atoms correspond, the rest are new
    amap_b = [i if i < a.n_atoms else -1 for i in range(b.n_atoms)]
    out = combine_members([a, b], [0.5, 0.5], kind="tautomers",
                          atom_maps=[list(range(a.n_atoms)), amap_b])
    assert not out.same_atoms
    assert list(out.sizes) == [a.n_atoms, b.n_atoms]
    assert out.member_kind == "tautomers"
    assert out.n_atoms == a.n_atoms + b.n_atoms
    assert out.member_slice(1) == slice(a.n_atoms, a.n_atoms + b.n_atoms)
    v = np.ones(out.n_atoms)
    got = out.per_atom(v)
    assert len(got) == a.n_atoms
    lost = b.n_atoms - a.n_atoms
    assert got.sum() == pytest.approx(out.n_atoms - lost)
    assert out.unmapped_share(v) == pytest.approx(lost / out.n_atoms)


# ----------------------------------------------------------------------- the pKa weights


def test_the_fractions_are_the_henderson_hasselbalch_ones():
    dep, pro = _fraction(5.0, 5.0)
    assert dep == pytest.approx(0.5) and pro == pytest.approx(0.5)
    dep, pro = _fraction(5.0, 6.0)
    assert dep == pytest.approx(10.0 / 11.0) and pro == pytest.approx(1.0 / 11.0)
    assert dep + pro == pytest.approx(1.0)


def test_an_extreme_pka_does_not_overflow():
    """The pattern table carries a pKa of -1000 for a group that is always deprotonated."""
    dep, pro = _fraction(-1000.0, 0.0)
    assert dep == pytest.approx(1.0) and pro < 1e-300


# ------------------------------------------------------------------- the states themselves


@needs_dimorphite
def test_a_molecule_with_no_ionisable_site_is_one_state():
    e = protonation_states(_mol(SMILES["benzene"]), ph=0.0)
    assert len(e) == 1 and not e.is_mixture
    assert e.states[0].weight == 1.0
    assert "no ionisable site" in e.note


@needs_dimorphite
def test_a_pH_at_the_pka_is_a_half_and_half_mixture():
    e = protonation_states(_mol(SMILES["aniline"]), ph=ANILINE_PKA)
    assert len(e) == 2 and e.is_mixture
    assert e.weights.sum() == pytest.approx(1.0)
    assert e.weights[0] == pytest.approx(0.5, abs=0.05)
    assert sorted(s.charge for s in e.states) == [0, 1]
    assert e.mean_charge == pytest.approx(0.5, abs=0.05)
    # heaviest first, always
    assert list(e.weights) == sorted(e.weights, reverse=True)


@needs_dimorphite
def test_far_from_the_pka_the_ensemble_is_one_state_again():
    low = protonation_states(_mol(SMILES["aniline"]), ph=0.0)
    high = protonation_states(_mol(SMILES["aniline"]), ph=10.0)
    assert len(low) == 1 and low.main.charge == 1        # the cation in acid
    assert len(high) == 1 and high.main.charge == 0      # the free base in base
    assert low.sites and low.sites[0].pka > 0


@needs_dimorphite
def test_the_sites_are_reported_with_their_pka_values():
    e = protonation_states(_mol(SMILES["acetic acid"]), ph=7.0)
    assert e.sites and any("arboxyl" in s.name for s in e.sites)
    assert 2.0 < e.sites[0].pka < 6.0
    assert e.main.charge == -1                           # deprotonated at pH 7
    assert "pH 7" in e.describe()


@needs_dimorphite
def test_the_heavy_atoms_keep_the_geometry_they_were_given():
    """A CMF series is aligned; building the states must not move it."""
    mol = _mol(SMILES["aniline"])
    heavy = Chem.RemoveHs(Chem.Mol(mol))
    want = np.array(heavy.GetConformer().GetPositions())
    e = protonation_states(mol, ph=ANILINE_PKA)
    assert len(e) == 2
    for st in e.states:
        got = np.array(Chem.RemoveHs(Chem.Mol(st.mol)).GetConformer().GetPositions())
        assert got.shape == want.shape
        assert np.allclose(np.sort(got, axis=0), np.sort(want, axis=0), atol=1e-6)


@needs_dimorphite
def test_a_molecule_without_geometry_comes_back_as_itself():
    flat = Chem.AddHs(Chem.MolFromSmiles(SMILES["aniline"]))
    e = protonation_states(flat, ph=ANILINE_PKA)
    assert len(e) == 1 and "no geometry" in e.note


# --------------------------------------------------------------------------- the field


@needs_dimorphite
def test_one_state_gives_the_same_field_as_building_that_state():
    fs = FieldSet.create(alpha=0.5)
    mol = _mol(SMILES["benzene"])
    a = protonation_fields(mol, fs, ph=0.0, name="b")
    b = build_fields(mol, fs, "b", _single=True)
    assert np.array_equal(a.coords, b.coords) and np.array_equal(a.weights, b.weights)
    assert not a.is_ensemble


@needs_dimorphite
def test_the_ensemble_field_is_the_weighted_mean_of_its_members():
    fs = FieldSet.create(alpha=0.5)
    mol = _mol(SMILES["aniline"])
    e = protonation_states(mol, ph=ANILINE_PKA)
    mf = protonation_fields(mol, fs, ph=ANILINE_PKA, name="a")
    assert mf.is_ensemble and mf.member_kind == "protonation"
    assert list(mf.sizes) == [s.mol.GetNumAtoms() for s in e.states]
    assert not mf.same_atoms                  # the states differ by one proton
    for i, st in enumerate(e.states):
        one = build_fields(st.mol, fs, "s", _single=True)
        assert np.allclose(mf.weights[mf.member_slice(i)], one.weights * st.weight)


@needs_dimorphite
def test_the_kernel_of_two_ensembles_is_the_mean_of_the_member_kernels():
    """The property the whole construction rests on, for members that are different molecules."""
    fs = FieldSet.create(alpha=0.6)
    mols = [_mol(SMILES["aniline"], seed=1), _mol(SMILES["toluidine"], seed=2)]
    ens = [protonation_states(m, ph=ANILINE_PKA) for m in mols]
    assert all(len(e) == 2 for e in ens)
    fields = [protonation_fields(m, fs, ph=ANILINE_PKA, name=f"m{i}")
              for i, m in enumerate(mols)]
    tab = AtomTable.from_molecules(fields)
    K = kernel_matrices(tab, tab, fs.specs)
    for i, a in enumerate(ens):
        for j, b in enumerate(ens):
            acc = np.zeros(len(fs.specs))
            for sa in a.states:
                for sb in b.states:
                    pair = AtomTable.from_molecules(
                        [build_fields(sa.mol, fs, "a", _single=True),
                         build_fields(sb.mol, fs, "b", _single=True)])
                    acc += sa.weight * sb.weight * kernel_matrices(pair, pair, fs.specs)[:, 0, 1]
            assert K[:, i, j] == pytest.approx(acc, rel=1e-9, abs=1e-12)


@needs_dimorphite
def test_atom_contributions_come_back_on_the_atoms_of_the_molecule():
    fs = FieldSet.create(alpha=0.5)
    mol = _mol(SMILES["aniline"])
    mf = protonation_fields(mol, fs, ph=ANILINE_PKA, name="a")
    v = np.ones(mf.n_atoms)
    got = mf.per_atom(v)
    assert len(got) == mf.sizes[0]
    # the proton that only one state has is reported on the nitrogen it sits on, so nothing is lost
    assert got.sum() == pytest.approx(mf.n_atoms)
    assert mf.unmapped_share(v) == 0.0


@needs_dimorphite
def test_the_field_set_carries_the_ph_and_build_fields_uses_it():
    fs = FieldSet.create(alpha=0.5, protonation_ph=ANILINE_PKA)
    assert fs.is_protonation_ensemble
    mf = build_fields(_mol(SMILES["aniline"]), fs, "a")
    assert mf.is_ensemble and mf.member_kind == "protonation"
    plain = build_fields(_mol(SMILES["aniline"]), FieldSet.create(alpha=0.5), "a")
    assert not plain.is_ensemble


@needs_dimorphite
def test_the_two_kinds_of_ensemble_cannot_be_combined_yet():
    fs = FieldSet.create(alpha=0.5, protonation_ph=0.0, conformers="ensemble")
    with pytest.raises(ValueError, match="cannot be combined"):
        build_fields(_conformers(SMILES["aniline"], n=3), fs, "a")


@needs_dimorphite
def test_a_mixed_series_trains_and_the_kernel_stays_positive_definite():
    from cmfqsar.model import CMFModel
    fs = FieldSet.create(alpha=0.5, protonation_ph=ANILINE_PKA)
    names = ["aniline", "toluidine", "benzene", "pyridine", "acetic acid", "benzimidazole"]
    mols = [_mol(SMILES[n], seed=i + 1) for i, n in enumerate(names)]
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    K = kernel_matrices(table, table, fs.specs)
    for f in range(K.shape[0]):
        w = np.linalg.eigvalsh(K[f])
        assert w.min() > -1e-8, f"field {f} is not positive semidefinite"
    y = np.linspace(4.0, 9.0, len(mols))
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(table, y, cv_folds=0)
    assert model.stats is not None and model.stats.r2 > 0.5


@needs_dimorphite
def test_a_whole_series_can_be_described_at_once():
    names = ["aniline", "benzene", "acetic acid"]
    mols = [_mol(SMILES[n], seed=i + 1) for i, n in enumerate(names)]
    ens = ensemble_table(mols, names, ph=ANILINE_PKA)
    assert len(ens) == 3
    # aniline (pKa 3.9) and acetic acid (pKa 3.5) are both mixtures at this pH; benzene is not
    assert mixture_share(ens) == pytest.approx(2.0 / 3.0)
    assert mixture_share([]) == 0.0
    assert all(isinstance(e, Ensemble) and isinstance(e.main, State) for e in ens)


@needs_dimorphite
def test_the_report_says_what_the_ensemble_was():
    fs = FieldSet.create(alpha=0.5)
    rep = {}
    protonation_fields(_mol(SMILES["aniline"]), fs, ph=ANILINE_PKA, name="a", report=rep)
    assert rep["n_states"] == 2
    assert sum(rep["weights"]) == pytest.approx(1.0)
    assert rep["charges"] == [0, 1] or rep["charges"] == [1, 0]
    assert rep["sites"] and "pka" in rep["sites"][0]
    assert rep["dropped"] < 0.01
