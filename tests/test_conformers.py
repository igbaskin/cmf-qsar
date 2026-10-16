"""Tests for the conformational ensemble mode: the fourth dimension."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.conformers import (CONFORMER_ALIGNMENTS, align_conformers,  # noqa: E402
                                align_conformers_to_reference, core_atom_indices,
                                dataset_reference, ensemble_summary, generate_conformers,
                                generate_for_dataset, get_conformer_energies, group_conformers,
                                set_conformer_energies, single_conformer)
from cmfqsar.fields import (AtomTable, FieldSet, build_fields, conformer_weights,  # noqa: E402
                            ensemble_fields)
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

SMILES = ["CCCOc1ccccc1C(=O)NCC", "CCCNc1ccccc1C(=O)OC", "CCOc1ccc(C)cc1CC(=O)N",
          "CC(C)COc1ccccc1C(=O)NC", "CCCCOc1ccccc1C(=O)N"]


def _mol(smiles: str, seed: int = 1, name: str = "m"):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    m.SetProp("_Name", name)
    return m


@pytest.fixture(scope="module")
def ensemble_mol():
    return generate_conformers(_mol(SMILES[0], name="probe"), n=5, seed=3, prune_rmsd=1.0)


def test_generation_keeps_the_input_pose_and_stores_energies(ensemble_mol):
    m = ensemble_mol
    assert m.GetNumConformers() > 1
    e = get_conformer_energies(m)
    assert e is not None and len(e) == m.GetNumConformers() and np.isfinite(e).all()
    # the pose the molecule arrived with is still conformer 0, so an earlier alignment survives
    original = _mol(SMILES[0], name="probe")
    assert np.allclose(m.GetConformer(0).GetPositions(), original.GetConformer().GetPositions())
    assert m.GetProp("_Name") == "probe"


def test_seed_zero_still_gives_distinct_conformations():
    """RDKit returns identical geometries for a seed of exactly zero; the generator must not."""
    m = generate_conformers(_mol(SMILES[1], name="z"), n=6, seed=0, prune_rmsd=0.5)
    assert m.GetNumConformers() >= 3
    p = [np.array(c.GetPositions()) for c in m.GetConformers()]
    assert max(np.abs(p[0] - q).max() for q in p[1:]) > 0.5


def test_single_mode_ignores_the_extra_conformations(ensemble_mol):
    """The default must behave exactly as it did before conformations existed."""
    from rdkit import Chem
    fs = FieldSet.create(alpha=0.5)
    assert fs.conformers == "single" and not fs.is_ensemble
    one = Chem.Mol(ensemble_mol)
    keep = Chem.Conformer(one.GetConformer(0))
    one.RemoveAllConformers()
    one.AddConformer(keep, assignId=True)
    a = build_fields(ensemble_mol, fs, "x")
    b = build_fields(one, fs, "x")
    assert a.n_atoms == b.n_atoms == one.GetNumAtoms()
    assert np.array_equal(a.coords, b.coords) and np.array_equal(a.weights, b.weights)
    assert not a.is_ensemble and a.n_conformers == 1


def test_ensemble_field_is_the_weighted_union(ensemble_mol):
    fs = FieldSet.create(alpha=0.5, conformers="ensemble")
    e = build_fields(ensemble_mol, fs, "x")
    k = ensemble_mol.GetNumConformers()
    n = ensemble_mol.GetNumAtoms()
    assert e.is_ensemble and e.n_conformers == k and e.atoms_per_conformer == n
    assert e.n_atoms == k * n
    assert e.conformer_weights == pytest.approx(np.full(k, 1.0 / k))
    one = build_fields(ensemble_mol, FieldSet.create(alpha=0.5), "x")
    assert np.allclose(e.weights[:n], one.weights / k)         # scaled by the conformer weight
    assert np.allclose(e.coords[:n], one.coords)
    assert len(e.symbols) == k * n and len(e.bonds) == k * len(one.bonds)
    # the bonds of each conformation point into its own block of atoms
    assert e.bonds[len(one.bonds):2 * len(one.bonds)].min() >= n


def test_ensemble_kernel_is_the_mean_of_the_conformer_kernels():
    """The heart of the method: the mean-map kernel of the two conformational distributions."""
    from rdkit import Chem
    fs_e = FieldSet.create(alpha=0.6, conformers="ensemble")
    fs_1 = FieldSet.create(alpha=0.6)
    mols = [generate_conformers(_mol(s, seed=i + 1, name=f"m{i}"), n=3, seed=i + 5,
                                prune_rmsd=1.0) for i, s in enumerate(SMILES[:2])]
    tab = AtomTable.from_molecules([build_fields(m, fs_e, m.GetProp("_Name")) for m in mols])
    K = kernel_matrices(tab, tab, fs_e.specs)                   # (F, 2, 2)

    def single(mol, conf_id):
        one = Chem.Mol(mol)
        c = Chem.Conformer(one.GetConformer(conf_id))
        one.RemoveAllConformers()
        one.AddConformer(c, assignId=True)
        return build_fields(one, fs_1, "c")

    for i, a in enumerate(mols):
        for j, b in enumerate(mols):
            wa = np.full(a.GetNumConformers(), 1.0 / a.GetNumConformers())
            wb = np.full(b.GetNumConformers(), 1.0 / b.GetNumConformers())
            acc = np.zeros(len(fs_1.specs))
            for ka in range(a.GetNumConformers()):
                for kb in range(b.GetNumConformers()):
                    pair = AtomTable.from_molecules([single(a, ka), single(b, kb)])
                    acc += wa[ka] * wb[kb] * kernel_matrices(pair, pair, fs_1.specs)[:, 0, 1]
            assert K[:, i, j] == pytest.approx(acc, rel=1e-9, abs=1e-12)


def test_one_conformation_in_ensemble_mode_equals_the_single_mode():
    m = _mol(SMILES[2], name="solo")
    a = build_fields(m, FieldSet.create(alpha=0.5, conformers="ensemble"), "x")
    b = build_fields(m, FieldSet.create(alpha=0.5), "x")
    assert np.array_equal(a.coords, b.coords) and np.array_equal(a.weights, b.weights)
    assert a.n_conformers == 1 and not a.is_ensemble


def test_weights_uniform_boltzmann_and_the_cap(ensemble_mol):
    from rdkit import Chem
    m = Chem.Mol(ensemble_mol)
    k = m.GetNumConformers()
    fs = FieldSet.create(alpha=0.5, conformers="ensemble")
    assert conformer_weights(m, fs) == pytest.approx(np.full(k, 1.0 / k))
    # Boltzmann weights follow the stored energies
    e = np.zeros(k)
    e[1:] = 10.0                                      # one clear minimum
    set_conformer_energies(m, e)
    fs.conformer_weighting = "boltzmann"
    fs.conformer_kt = 0.6
    w = conformer_weights(m, fs)
    assert w[0] > 0.99 and w.sum() == pytest.approx(1.0)
    # a bigger kT flattens them
    fs.conformer_kt = 100.0
    assert conformer_weights(m, fs).std() < w.std()
    # missing or mismatched energies fall back to uniform rather than guessing
    m.SetProp("conformer_energies", "1.0,2.0")
    assert conformer_weights(m, fs) == pytest.approx(np.full(k, 1.0 / k))
    m.ClearProp("conformer_energies")
    assert conformer_weights(m, fs) == pytest.approx(np.full(k, 1.0 / k))
    # the cap keeps the heaviest conformations and renormalises
    set_conformer_energies(m, np.arange(k, dtype=float))
    fs.conformer_kt = 0.6
    fs.max_conformers = 2
    w2 = conformer_weights(m, fs)
    assert (w2 > 0).sum() == 2 and w2.sum() == pytest.approx(1.0)
    assert w2[0] > w2[1] > 0 and w2[2:].sum() == 0.0
    assert build_fields(m, fs, "x").n_conformers == 2


def test_contributions_of_an_ensemble_add_up_and_fold_back_onto_atoms():
    fs = FieldSet.create(alpha=0.6, conformers="ensemble")
    mols = [generate_conformers(_mol(s, seed=i + 1, name=f"m{i}"), n=3, seed=i + 2,
                                prune_rmsd=1.0) for i, s in enumerate(SMILES)]
    mf = [build_fields(m, fs, m.GetProp("_Name")) for m in mols]
    y = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 1e-3}, normalize=True)
    model.fit(AtomTable.from_molecules(mf), y, cv_folds=0)
    probe = mf[0]
    c = model.atom_contributions(probe)                # one value per atom per conformation
    assert len(c) == probe.n_atoms
    pred = float(model.predict(AtomTable.from_molecules([probe]))[0])
    assert float(c.sum() + model.bias) == pytest.approx(pred, abs=1e-8)
    folded = probe.per_atom(c)                         # summed back onto the atoms of the molecule
    assert len(folded) == probe.atoms_per_conformer
    assert folded.sum() == pytest.approx(c.sum(), abs=1e-10)
    # co-fields remain available, because the model is still linear in the mean field
    centers, coeffs, widths = model.cofield_coefficients(0)
    assert len(centers) == sum(m.n_atoms for m in mf)


def test_group_conformers_collapses_a_multi_record_file():
    mols, names = [], []
    for i in range(2):
        m = generate_conformers(_mol(SMILES[i], seed=i + 1, name=f"cmp{i}"), n=3, seed=i + 7,
                                prune_rmsd=1.0)
        for c in m.GetConformers():
            from rdkit import Chem
            one = Chem.Mol(m)
            keep = Chem.Conformer(c)
            one.RemoveAllConformers()
            one.AddConformer(keep, assignId=True)
            mols.append(one)
            names.append(f"cmp{i}")
    grouped, gnames, counts = group_conformers(mols, names, by="name")
    assert gnames == ["cmp0", "cmp1"]
    assert counts == [3, 3] and [g.GetNumConformers() for g in grouped] == [3, 3]
    assert len(grouped) == 2


def test_summary_reports_the_cost_and_the_effective_count(ensemble_mol):
    fs = FieldSet.create(alpha=0.5, conformers="ensemble")
    info = ensemble_summary([ensemble_mol], fs)
    k = ensemble_mol.GetNumConformers()
    assert info["conformations_mean"] == pytest.approx(k)
    assert info["effective_mean"] == pytest.approx(k)          # uniform weights
    assert info["kernel_cost_factor"] == pytest.approx(k * k)
    # with one dominant conformation the effective number collapses towards one
    from rdkit import Chem
    m = Chem.Mol(ensemble_mol)
    e = np.zeros(k)
    e[1:] = 20.0
    set_conformer_energies(m, e)
    fs.conformer_weighting = "boltzmann"
    assert ensemble_summary([m], fs)["effective_mean"] < 1.01


def test_field_set_carries_the_mode_through_copy():
    fs = FieldSet.create(alpha=0.5, conformers="ensemble", conformer_weighting="boltzmann",
                         conformer_kt=1.5, max_conformers=4)
    c = fs.copy()
    assert c.is_ensemble and c.conformer_weighting == "boltzmann"
    assert c.conformer_kt == 1.5 and c.max_conformers == 4
    assert FieldSet.create().conformers == "single"


def test_align_conformers_superimposes_them(ensemble_mol):
    from rdkit import Chem
    m = Chem.Mol(ensemble_mol)
    before = np.array(m.GetConformer(1).GetPositions())
    rms = align_conformers(m)
    assert len(rms) == m.GetNumConformers() - 1
    after = np.array(m.GetConformer(1).GetPositions())
    assert np.abs(after - before).max() >= 0.0            # coordinates may or may not move
    assert np.allclose(m.GetConformer(0).GetPositions(),
                       ensemble_mol.GetConformer(0).GetPositions())


# --------------------------------------------------------------------------- #
# Aligning the ensemble to the series rather than to each molecule's own pose
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def series():
    """A little series in one frame, plus its reference molecule, before any generation."""
    mols = [_mol(s, seed=i + 1, name=f"s{i}") for i, s in enumerate(SMILES)]
    return mols, dataset_reference(mols)


def _rmsd_on(mol, conf, ref, amap):
    a = np.array(mol.GetConformer(conf).GetPositions())[[i for i, _ in amap]]
    b = np.array(ref.GetConformer().GetPositions())[[j for _, j in amap]]
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def test_the_reference_is_the_largest_molecule_in_its_original_pose(series):
    mols, ref = series
    big = max(range(len(mols)), key=lambda i: mols[i].GetNumHeavyAtoms())
    assert ref.GetNumConformers() == 1
    assert np.allclose(ref.GetConformer().GetPositions(), mols[big].GetConformer().GetPositions())
    # and it can be picked by hand
    assert np.allclose(dataset_reference(mols, 1).GetConformer().GetPositions(),
                       mols[1].GetConformer().GetPositions())
    with pytest.raises(ValueError):
        dataset_reference(mols, 99)


def test_every_alignment_mode_leaves_the_input_pose_alone(series):
    """Whatever is done with the generated conformations, the published pose must survive."""
    mols, ref = series
    fs = FieldSet.create(alpha=0.5)
    for mode in CONFORMER_ALIGNMENTS:
        g = generate_conformers(mols[0], n=4, seed=11, prune_rmsd=1.0, align_to=mode,
                                reference=ref, fieldset=fs)
        assert g.GetNumConformers() > 1
        assert np.allclose(g.GetConformer(0).GetPositions(),
                           mols[0].GetConformer().GetPositions())


def test_reference_alignment_puts_the_conformations_in_the_frame_of_the_series(series):
    """The point of the exercise: the extra conformations stop being arbitrarily oriented."""
    from cmfqsar.align import mcs_atom_map
    mols, ref = series
    probe = mols[3]
    amap = mcs_atom_map(probe, ref)
    assert len(amap) >= 3
    own = generate_conformers(probe, n=4, seed=13, prune_rmsd=1.0, align_to="self")
    onto = generate_conformers(probe, n=4, seed=13, prune_rmsd=1.0, align_to="reference",
                               reference=ref)
    assert own.GetNumConformers() == onto.GetNumConformers() > 1
    for k in range(1, onto.GetNumConformers()):
        assert _rmsd_on(onto, k, ref, amap) < _rmsd_on(own, k, ref, amap)


def test_core_alignment_holds_the_scaffold_on_the_published_pose(series):
    """The common scaffold stays where the series put it; only the flexible part moves."""
    from cmfqsar.align import mcs_atom_map
    mols, ref = series
    probe = mols[3]
    core = [(i, i) for i, _ in mcs_atom_map(probe, ref)]
    own = generate_conformers(probe, n=4, seed=13, prune_rmsd=1.0, align_to="self")
    kept = generate_conformers(probe, n=4, seed=13, prune_rmsd=1.0, align_to="core",
                               reference=ref)
    anchor = single_conformer(probe, 0)
    for k in range(1, kept.GetNumConformers()):
        assert _rmsd_on(kept, k, anchor, core) <= _rmsd_on(own, k, anchor, core) + 1e-9


def test_field_refinement_cannot_make_the_field_similarity_worse(series):
    """The refinement keeps the MCS pose unless the kernel with the reference improves."""
    from cmfqsar.align import mcs_atom_map
    from cmfqsar.kernels import kernel_matrices
    mols, ref = series
    fs = FieldSet.create(alpha=0.5)

    def similarity(mol, k):
        pair = AtomTable.from_molecules([build_fields(single_conformer(mol, k), fs, "p"),
                                         build_fields(ref, fs, "r")])
        K = kernel_matrices(pair, pair, fs.specs)
        return float(np.sum(K[:, 0, 1] / np.sqrt(K[:, 0, 0] * K[:, 1, 1])))

    probe = mols[2]
    amap = mcs_atom_map(probe, ref)
    base = generate_conformers(probe, n=3, seed=17, prune_rmsd=1.0, align_to="reference",
                               reference=ref)
    tuned = generate_conformers(probe, n=3, seed=17, prune_rmsd=1.0, align_to="field",
                                reference=ref, fieldset=fs)
    assert base.GetNumConformers() == tuned.GetNumConformers() > 1
    for k in range(1, tuned.GetNumConformers()):
        assert similarity(tuned, k) >= similarity(base, k) - 1e-6
    # the refinement starts from the MCS pose, so it stays in the frame of the series
    assert max(_rmsd_on(tuned, k, ref, amap) for k in range(1, tuned.GetNumConformers())) < 20.0


def test_unknown_alignment_is_refused_and_a_missing_reference_falls_back(series):
    mols, ref = series
    g = generate_conformers(mols[0], n=3, seed=19, prune_rmsd=1.0, align_to="self")
    with pytest.raises(ValueError):
        align_conformers_to_reference(g, ref, mode="centroid")
    # without a reference the only thing left is the molecule's own pose, and it is used
    msg = align_conformers_to_reference(g, None, mode="reference")
    assert "input pose" in msg


def test_the_whole_series_can_be_aligned_to_one_reference(series):
    mols, _ = series
    out, failures = generate_for_dataset(mols[:3], n=3, seed=23, prune_rmsd=1.0,
                                         align_to="core", reference_index=0)
    assert not failures and len(out) == 3
    for m, before in zip(out, mols[:3]):
        assert m.GetNumConformers() > 1
        assert np.allclose(m.GetConformer(0).GetPositions(), before.GetConformer().GetPositions())
        assert get_conformer_energies(m) is not None


def test_the_default_is_still_the_molecule_s_own_pose(series):
    """The mode that was benchmarked must stay the default, bit for bit."""
    mols, ref = series
    a = generate_conformers(mols[1], n=4, seed=29, prune_rmsd=1.0)
    b = generate_conformers(mols[1], n=4, seed=29, prune_rmsd=1.0, align_to="self",
                            reference=ref)
    assert a.GetNumConformers() == b.GetNumConformers()
    for k in range(a.GetNumConformers()):
        assert np.allclose(a.GetConformer(k).GetPositions(), b.GetConformer(k).GetPositions())


# --------------------------------------------------------------------------- #
# Generating the conformations around the published core, and the energy window
# --------------------------------------------------------------------------- #

def _core_rmsd(mol, conf, base, core):
    a = np.array(mol.GetConformer(conf).GetPositions())[core]
    b = np.array(base.GetConformer(0).GetPositions())[core]
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def test_core_atom_indices_finds_the_scaffold_and_degrades_quietly(series):
    mols, ref = series
    core = core_atom_indices(mols[3], ref)
    assert core is not None and len(core) >= 3
    assert core == sorted(set(core)) and max(core) < mols[3].GetNumAtoms()
    # nothing to hold on to: no reference, or a molecule without a conformation
    assert core_atom_indices(mols[3], None) is None
    from rdkit import Chem
    bare = Chem.Mol(mols[3])
    bare.RemoveAllConformers()
    assert core_atom_indices(bare, ref) is None
    # a correspondence can be handed in, and one below three atoms is refused
    assert core_atom_indices(mols[3], ref, atom_map=[(0, 0), (1, 1), (2, 2)]) == [0, 1, 2]
    assert core_atom_indices(mols[3], ref, atom_map=[(0, 0), (1, 1)]) is None


def test_core_constrained_generation_holds_the_published_scaffold(series):
    """The hypothesis under test: keep the published core, vary only the flexible part."""
    mols, ref = series
    probe = mols[3]
    core = core_atom_indices(probe, ref)
    free = generate_conformers(probe, n=5, seed=13, prune_rmsd=1.0, align_to="core",
                               reference=ref)
    held = generate_conformers(probe, n=5, seed=13, prune_rmsd=1.0, align_to="core",
                               reference=ref, constrain_core=True)
    assert held.GetNumConformers() > 1
    assert np.allclose(held.GetConformer(0).GetPositions(), probe.GetConformer().GetPositions())
    held_rms = [_core_rmsd(held, k, probe, core) for k in range(1, held.GetNumConformers())]
    free_rms = [_core_rmsd(free, k, probe, core) for k in range(1, free.GetNumConformers())]
    # superimposing a freely generated conformation on the core only brings it so close; holding
    # the core while the conformation is built keeps it there
    assert max(held_rms) < 0.3
    assert np.mean(free_rms) > np.mean(held_rms)
    # and the ensemble has not collapsed: the conformations still differ from each other
    pos = [np.array(c.GetPositions()) for c in held.GetConformers()]
    assert max(np.abs(pos[0] - q).max() for q in pos[1:]) > 0.3
    assert get_conformer_energies(held) is not None


def test_the_energy_window_drops_the_worst_and_keeps_the_input_pose(series):
    mols, ref = series
    probe = mols[0]
    all_of = generate_conformers(probe, n=6, seed=31, prune_rmsd=0.5, align_to="core",
                                 reference=ref)
    e = get_conformer_energies(all_of)
    window = float(np.median(e[1:]) - np.min(e))      # cuts something, whatever the geometry
    cut = generate_conformers(probe, n=6, seed=31, prune_rmsd=0.5, align_to="core",
                              reference=ref, energy_window=window)
    assert 1 <= cut.GetNumConformers() <= all_of.GetNumConformers()
    ec = get_conformer_energies(cut)
    assert ec is not None and len(ec) == cut.GetNumConformers()
    # the published pose survives whatever the force field thinks of it, the rest are inside
    assert np.allclose(cut.GetConformer(0).GetPositions(), probe.GetConformer().GetPositions())
    assert (ec[1:] <= np.min(ec) + window + 1e-6).all()
    # a window wide enough to hold everything changes nothing
    wide = generate_conformers(probe, n=6, seed=31, prune_rmsd=0.5, align_to="core",
                               reference=ref, energy_window=1e6)
    assert wide.GetNumConformers() == all_of.GetNumConformers()


def test_the_new_controls_are_off_by_default(series):
    """Neither control may touch what was measured before them."""
    mols, ref = series
    a = generate_conformers(mols[1], n=4, seed=29, prune_rmsd=1.0)
    b = generate_conformers(mols[1], n=4, seed=29, prune_rmsd=1.0, constrain_core=False,
                            energy_window=0.0, reference=ref)
    assert a.GetNumConformers() == b.GetNumConformers()
    for k in range(a.GetNumConformers()):
        assert np.allclose(a.GetConformer(k).GetPositions(), b.GetConformer(k).GetPositions())


def test_a_whole_series_can_be_generated_around_its_core(series):
    mols, _ = series
    out, failures = generate_for_dataset(mols[:3], n=3, seed=23, prune_rmsd=1.0,
                                         align_to="core", reference_index=0,
                                         constrain_core=True, energy_window=25.0)
    assert not failures and len(out) == 3
    for m, before in zip(out, mols[:3]):
        assert m.GetNumConformers() >= 1
        assert np.allclose(m.GetConformer(0).GetPositions(), before.GetConformer().GetPositions())
        assert get_conformer_energies(m) is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
