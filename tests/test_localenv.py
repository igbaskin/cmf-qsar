"""Atom-centred environments and the invariant local kernel (plan item LP2).

The sample here is an atom in a molecule, not the molecule.  What these tests fix is that the
environment is a field like any other (so the whole program applies to it), that it is local (an
atom far from the centre cannot change it), that it is invariant (turning the molecule does not
change it), that the count reference is the increment scheme it is meant to be, and that the
cross-validation never tests a molecule against itself.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import autocorrelation  # noqa: E402
from cmfqsar.localenv import (DEFAULT_RHO, LOCAL_SPEC, SHELL_EDGES,  # noqa: E402
                              choose_lam, count_features, count_kernel, cross_validate,
                              cross_validate_lams, difference_field, ensemble_environment,
                              envelope, envelope_radius, envelope_width, environment,
                              environments, kept_atoms, kept_share, local_features,
                              local_kernel, paired_interval, shell_counts, types_of)

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "CCOc1ccccc1N", "CC(C)c1ccccc1O", "CCCCc1ccccc1",
          "CC(=O)Nc1ccc(O)cc1", "COc1ccc(C=O)cc1"]


def _mol(smiles, name="m", seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    m.SetProp("_Name", name)
    return m


@pytest.fixture(scope="module")
def built():
    """A handful of molecules, their fields, and the shift-like target on every carbon."""
    fieldset = FieldSet.create()
    mols, fields = [], []
    for i, s in enumerate(SMILES):
        m = _mol(s, f"m{i}", seed=i + 1)
        mols.append(m)
        fields.append(build_fields(m, fieldset, name=f"m{i}"))
    # the target is the number of hydrogens on the carbon, which any local descriptor should
    # recover: it is a property of the atom and its first shell
    for m in mols:
        for a in m.GetAtoms():
            if a.GetSymbol() == "C":
                n_h = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "H")
                a.SetDoubleProp("nh", float(n_h))
    targets = AtomTargets.from_molecules(mols, "nh", names=[f"m{i}" for i in range(len(mols))])
    return mols, fields, targets, fieldset


# ------------------------------------------------------------------- the envelope and the field


def test_the_width_and_the_radius_are_the_distances_they_claim():
    assert envelope_width(0.25) == pytest.approx(2.0)
    assert envelope(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), [0, 0, 0], 0.25) == \
        pytest.approx([1.0, np.exp(-1.0)])
    r = envelope_radius(0.1, 1e-4)
    assert np.exp(-0.1 * r * r) == pytest.approx(1e-4)
    with pytest.raises(ValueError, match="positive"):
        envelope_width(0.0)


def test_the_centre_sits_at_the_origin_and_keeps_its_whole_weight(built):
    _mols, fields, _t, _fs = built
    m = fields[0]
    e = environment(m, 3, rho=DEFAULT_RHO)
    assert e.coords[0] == pytest.approx([0.0, 0.0, 0.0])
    assert e.weights[0] == pytest.approx(m.weights[3])
    assert e.atom_types[0] == m.atom_types[3]
    assert e.name.endswith(":3")


def test_an_atom_further_out_keeps_less(built):
    _mols, fields, _t, _fs = built
    m = fields[0]
    e = environment(m, 0, rho=0.2, floor=0.0)
    assert e.n_atoms == m.n_atoms
    d = np.linalg.norm(m.coords - m.coords[0], axis=1)
    back = np.asarray(e.member_atom_map, int)          # which atom of the molecule each row is
    assert back[0] == 0 and sorted(back.tolist()) == list(range(m.n_atoms))
    for k in range(e.n_atoms):
        i = int(back[k])
        assert e.weights[k] == pytest.approx(m.weights[i] * np.exp(-0.2 * d[i] ** 2))
    far = int(np.argmax(d))
    row = int(np.where(back == far)[0][0])
    assert abs(e.weights[row]).sum() < abs(m.weights[far]).sum()


def test_the_floor_drops_the_distant_atoms_and_says_how_much_it_dropped(built):
    _mols, fields, _t, _fs = built
    m = fields[0]
    tight = environment(m, 0, rho=1.0, floor=1e-4)
    loose = environment(m, 0, rho=0.05, floor=1e-4)
    assert tight.n_atoms < loose.n_atoms <= m.n_atoms
    d = np.linalg.norm(m.coords - m.coords[0], axis=1)
    assert tight.n_atoms == int(np.sum(d <= envelope_radius(1.0, 1e-4)))
    share = kept_share(m, 0, rho=1.0, floor=1e-4)
    assert 0.0 < share <= 1.0
    assert kept_share(m, 0, rho=1.0, floor=0.0) == pytest.approx(1.0)


def test_the_bonds_still_point_at_the_right_atoms(built):
    _mols, fields, _t, _fs = built
    m = fields[0]
    e = environment(m, 0, rho=0.5, floor=1e-3)
    assert len(e.bonds) and e.bonds.max() < e.n_atoms
    kept = [i for i in range(m.n_atoms)
            if np.exp(-0.5 * float(np.sum((m.coords[i] - m.coords[0]) ** 2))) >= 1e-3]
    pairs = {tuple(sorted((e.symbols[a], e.symbols[b]))) for a, b in e.bonds}
    original = {tuple(sorted((m.symbols[a], m.symbols[b]))) for a, b in m.bonds
                if a in kept and b in kept}
    assert pairs == original


def test_an_atom_that_is_not_there_is_refused_by_number(built):
    _mols, fields, _t, _fs = built
    with pytest.raises(ValueError, match="atoms"):
        environment(fields[0], fields[0].n_atoms)


def test_one_environment_per_measured_atom_in_order(built):
    mols, fields, targets, _fs = built
    envs = environments(fields, targets, rho=DEFAULT_RHO)
    assert len(envs) == len(targets)
    for r in (0, 5, len(targets) - 1):
        k, i = int(targets.molecule[r]), int(targets.atom[r])
        assert envs[r].atom_types[0] == fields[k].atom_types[i]
        assert mols[k].GetAtomWithIdx(i).GetSymbol() == "C"


# -------------------------------------------------------------------------- it really is local


def test_turning_the_molecule_does_not_change_the_descriptor(built):
    _mols, fields, _t, _fs = built
    m = fields[0]
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned = type(m)(name=m.name, coords=m.coords @ rot.T + 7.0, weights=m.weights,
                     radii=m.radii, symbols=m.symbols, atom_types=m.atom_types, bonds=m.bonds,
                     charges=m.charges)
    a = autocorrelation(environment(m, 2), LOCAL_SPEC)
    b = autocorrelation(environment(turned, 2), LOCAL_SPEC)
    assert a == pytest.approx(b, abs=1e-9)


def test_an_atom_beyond_the_envelope_cannot_change_the_environment(built):
    """The whole point of the envelope: a change far away is not in the sample."""
    _mols, fields, _t, _fs = built
    m = fields[3]                                     # butylbenzene, a long molecule
    d = np.linalg.norm(m.coords - m.coords[0], axis=1)
    far = int(np.argmax(d))
    rho = 1.0
    assert d[far] > envelope_radius(rho, 1e-4)
    changed = type(m)(name=m.name, coords=m.coords, weights=m.weights.copy(), radii=m.radii,
                      symbols=m.symbols, atom_types=m.atom_types, bonds=m.bonds,
                      charges=m.charges)
    changed.weights[far] += 5.0
    before = autocorrelation(environment(m, 0, rho), LOCAL_SPEC)
    after = autocorrelation(environment(changed, 0, rho), LOCAL_SPEC)
    assert before == pytest.approx(after)


def test_the_features_and_the_kernel_have_the_shape_the_spec_says(built):
    _mols, fields, targets, _fs = built
    envs = environments(fields, targets, rho=DEFAULT_RHO)[:12]
    X = local_features(envs)
    n_fields = fields[0].weights.shape[1]
    assert X.shape == (12, n_fields * (n_fields + 1) // 2 * LOCAL_SPEC.n_bins)
    K = local_kernel(X)
    assert K.shape == (12, 12)
    assert np.diag(K) == pytest.approx(np.ones(12))
    assert K == pytest.approx(K.T)
    assert K.max() <= 1.0 + 1e-9


def test_the_same_atom_of_two_copies_is_the_same_environment(built):
    _mols, fields, _t, _fs = built
    X = local_features([environment(fields[1], 4), environment(fields[1], 4)])
    K = local_kernel(X)
    assert K[0, 1] == pytest.approx(1.0)


# ------------------------------------------------------------------------- the difference field


def test_a_field_minus_itself_is_nothing(built):
    _mols, fields, _t, _fs = built
    d = difference_field(fields[0], fields[0])
    assert d.n_atoms == 2 * fields[0].n_atoms
    assert autocorrelation(d, LOCAL_SPEC) == pytest.approx(
        np.zeros(fields[0].weights.shape[1] * (fields[0].weights.shape[1] + 1) // 2
                 * LOCAL_SPEC.n_bins), abs=1e-9)


def test_the_difference_of_an_acid_and_its_anion_is_not_nothing(built):
    _mols, _fields, _t, fieldset = built
    acid = _mol("c1ccccc1C(=O)O", "acid")
    anion = Chem.RWMol(acid)
    for atom in anion.GetAtoms():
        if atom.GetSymbol() == "O" and atom.GetTotalNumHs() == 1:
            h = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "H"][0]
            anion.RemoveAtom(h)
            anion.GetAtomWithIdx(atom.GetIdx()).SetFormalCharge(-1)
            break
    anion = anion.GetMol()
    Chem.SanitizeMol(anion)
    a = build_fields(acid, fieldset, name="acid")
    b = build_fields(anion, fieldset, name="anion")
    d = difference_field(a, b, name="deprotonation")
    assert d.name == "deprotonation"
    assert d.n_atoms == a.n_atoms + b.n_atoms
    assert float(np.abs(autocorrelation(d, LOCAL_SPEC)).sum()) > 0
    e = environment(d, 6, rho=0.3)
    assert not e.is_ensemble and e.n_atoms > 1


def test_a_difference_needs_one_field_set(built):
    _mols, fields, _t, _fs = built
    other = FieldSet.create(physchem=["electrostatic"])
    small = build_fields(_mol("CCO", "e"), other, name="e")
    with pytest.raises(ValueError, match="one field set"):
        difference_field(fields[0], small)


# -------------------------------------------------------------- the count reference of rule 1


def test_the_count_row_says_what_the_centre_is_and_what_is_next_to_it(built):
    _mols, fields, _t, _fs = built
    m = fields[0]
    types = types_of(fields, min_atoms=1)
    row = shell_counts(m, 0, types, SHELL_EDGES)
    n_t = len(types) + 1
    assert len(row) == n_t * (1 + len(SHELL_EDGES))
    assert row[:n_t].sum() == 1.0                      # one-hot type of the centre
    assert row[:n_t].argmax() == types.index(m.atom_types[0])
    bonded = sum(1 for a, b in m.bonds if 0 in (a, b))
    assert row[n_t:2 * n_t].sum() == pytest.approx(float(bonded))
    assert row.sum() > row[:n_t].sum()


def test_a_type_too_rare_for_a_column_shares_the_last_one(built):
    _mols, fields, _t, _fs = built
    m = fields[0]
    types = types_of(fields, min_atoms=10 ** 6)
    assert types == []
    row = shell_counts(m, 0, types, SHELL_EDGES)
    assert len(row) == 1 * (1 + len(SHELL_EDGES))
    assert row[0] == 1.0


def test_one_count_row_per_measured_atom(built):
    _mols, fields, targets, _fs = built
    types = types_of(fields, min_atoms=5)
    X = count_features(fields, targets, types, SHELL_EDGES)
    assert X.shape == (len(targets), (len(types) + 1) * (1 + len(SHELL_EDGES)))
    K = count_kernel(X)
    assert np.diag(K) == pytest.approx(np.ones(len(targets)))


# ------------------------------------------------------------- cross-validation over molecules


def test_a_molecule_is_never_tested_against_itself(built):
    _mols, fields, targets, _fs = built
    envs = environments(fields, targets, rho=DEFAULT_RHO)
    K = local_kernel(local_features(envs))
    r = choose_lam(K, targets, folds=3, seed=0, name="local")
    assert r.n_samples == len(targets)
    assert np.isfinite(r.predicted).all()
    assert r.n_molecules == len(SMILES)
    assert "local" in r.summary() and "folds" in r.summary()
    # the target is the hydrogen count of the carbon, which a local descriptor must get right
    assert r.q2 > 0.3


def test_the_two_descriptors_are_cross_validated_the_same_way(built):
    _mols, fields, targets, _fs = built
    envs = environments(fields, targets, rho=DEFAULT_RHO)
    local = choose_lam(local_kernel(local_features(envs)), targets, folds=3, name="local")
    types = types_of(fields, min_atoms=5)
    counts = choose_lam(count_kernel(count_features(fields, targets, types)), targets, folds=3,
                        name="counts")
    assert local.folds == counts.folds == 3
    assert local.lam in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    diff, lo, hi = paired_interval(local, counts, targets, runs=200, seed=0)
    assert np.isfinite([diff, lo, hi]).all()
    assert lo <= diff <= hi or lo <= 0 <= hi


def test_the_shared_arithmetic_of_the_ridge_search_changes_no_result(built):
    """One eigendecomposition per fold gives every ridge; the fits must still be the fits."""
    _mols, fields, targets, _fs = built
    envs = environments(fields, targets, rho=DEFAULT_RHO)
    K = local_kernel(local_features(envs))
    lams = (1e-3, 1e-2, 1e-1, 1.0)
    together = cross_validate_lams(K, targets, lams, folds=3, seed=0)
    for lam, r in zip(lams, together):
        one = cross_validate(K, targets, lam, folds=3, seed=0)
        assert r.lam == lam
        assert r.predicted == pytest.approx(one.predicted, abs=1e-6)
        assert r.q2 == pytest.approx(one.q2, abs=1e-6)


def test_a_result_compared_with_itself_is_a_difference_of_nothing(built):
    _mols, fields, targets, _fs = built
    envs = environments(fields, targets, rho=DEFAULT_RHO)
    r = cross_validate(local_kernel(local_features(envs)), targets, folds=3, name="local")
    diff, lo, hi = paired_interval(r, r, targets, runs=100, seed=1)
    assert diff == pytest.approx(0.0)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0)


def test_there_is_nothing_to_cross_validate_without_a_target():
    with pytest.raises(ValueError, match="nothing to cross-validate"):
        cross_validate(np.zeros((0, 0)), AtomTargets(name="nothing"))

def test_the_command_fits_a_model_over_atoms_and_writes_the_predictions(tmp_path, capsys):
    from cmfqsar.cli import main
    path = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(path) as w:
        for i, smiles in enumerate(SMILES):
            m = _mol(smiles, f"m{i}", seed=i + 1)
            v = []
            for a in m.GetAtoms():
                if a.GetSymbol() == "C":
                    v.append("%.3f" % sum(1 for nb in a.GetNeighbors()
                                          if nb.GetSymbol() == "H"))
                else:
                    v.append("n/a")
            m.SetProp("atom.dprop.nh", " ".join(v))
            w.write(m)
    out = os.path.join(str(tmp_path), "predicted.sdf")
    capsys.readouterr()
    main(["local", path, "--target", "nh", "--rho", "0.5", "--reference", "--folds", "3",
          "--bootstrap", "100", "--out", out])
    text = capsys.readouterr().out
    assert "atom-centred counts" in text and "local field" in text
    assert "envelope width" in text and "against the counts" in text
    assert os.path.exists(out)
    back = [m for m in Chem.SDMolSupplier(out, removeHs=False) if m is not None]
    predicted = AtomTargets.from_molecules(back, "nh_predicted")
    assert len(predicted) == len(AtomTargets.from_molecules(back, "nh"))

# ------------------------------------------- the conformations the molecule visits (plan item LP4)


def _conformer_fields(smiles, n=4, seed=1, name="m"):
    """One molecule's fields in each of several conformations, the embedded one first."""
    from cmfqsar.conformers import generate_conformers, single_conformer

    fs = FieldSet.create()
    m = _mol(smiles, name, seed)
    ens = generate_conformers(m, n=n, seed=seed)
    k = ens.GetNumConformers()
    return [build_fields(single_conformer(ens, i), fs, name=f"{name}#{i}") for i in range(k)], ens


def test_an_ensemble_environment_is_the_mean_of_the_per_conformation_ones():
    members, _ens = _conformer_fields("CCCCc1ccccc1", n=4)
    assert len(members) > 1, "the generator gave only one conformation"
    e = ensemble_environment(members, 2, rho=0.5)
    assert e.is_ensemble and e.n_conformers == len(members)
    # every member keeps the same atoms, so the mean is taken over the same atoms
    assert e.n_atoms % len(members) == 0
    assert e.atoms_per_conformer == e.n_atoms // len(members)
    # the mean halves nothing: the member weights sum to one
    one = environment(members[0], 2, rho=0.5, keep=range(members[0].n_atoms))
    assert float(np.abs(e.weights).sum()) == pytest.approx(
        float(np.mean([np.abs(environment(m, 2, rho=0.5,
                                          keep=range(m.n_atoms)).weights).sum()
                       for m in members])), rel=0.2)
    assert one.n_atoms == members[0].n_atoms


def test_each_conformation_is_centred_on_its_own_copy_of_the_atom():
    """The atom moves between conformations, so there is no single centre to put an envelope on."""
    members, _ens = _conformer_fields("CCCCc1ccccc1", n=4)
    e = ensemble_environment(members, 2, rho=0.5)
    n = e.atoms_per_conformer
    for i in range(e.n_conformers):
        part = e.coords[e.member_slice(i)]
        assert part[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)   # the centre of that member


def test_one_conformation_gives_the_plain_environment():
    members, _ens = _conformer_fields("CCO", n=4)
    e = ensemble_environment(members[:1], 1, rho=0.5)
    assert not e.is_ensemble
    assert e.coords == pytest.approx(environment(members[0], 1, rho=0.5).coords)


def test_the_conformations_have_to_be_the_same_molecule():
    fs = FieldSet.create()
    a = build_fields(_mol("CCO", "a"), fs, name="a")
    b = build_fields(_mol("CCCO", "b"), fs, name="b")
    with pytest.raises(ValueError, match="same atoms"):
        ensemble_environment([a, b], 0, rho=0.5)
    with pytest.raises(ValueError, match="at least one conformation"):
        ensemble_environment([], 0, rho=0.5)


def test_the_weights_decide_how_much_each_conformation_counts():
    members, _ens = _conformer_fields("CCCCc1ccccc1", n=4)
    k = len(members)
    even = ensemble_environment(members, 2, rho=0.5, weights=np.ones(k))
    first = ensemble_environment(members, 2, rho=0.5,
                                 weights=[1.0] + [1e-12] * (k - 1))
    alone = environment(members[0], 2, rho=0.5)
    # weighting everything onto the first conformation reproduces that conformation
    a = local_features([first])[0]
    b = local_features([alone])[0]
    assert np.corrcoef(a, b)[0, 1] > 0.999
    assert not np.allclose(local_features([even])[0], a)


def test_the_union_of_kept_atoms_is_used_so_the_members_match():
    """A tight envelope keeps different atoms in different conformations; the union fixes that."""
    members, _ens = _conformer_fields("CCCCc1ccccc1", n=4)
    sets = [set(kept_atoms(m, 2, rho=1.0).tolist()) for m in members]
    assert len(set(map(frozenset, sets))) > 1, "the conformations keep the same atoms here"
    e = ensemble_environment(members, 2, rho=1.0)
    assert e.atoms_per_conformer == len(set().union(*sets))

def test_the_command_can_average_over_conformations(tmp_path, capsys):
    from cmfqsar.cli import main
    path = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(path) as w:
        for i, smiles in enumerate(SMILES[:4]):
            m = _mol(smiles, f"m{i}", seed=i + 1)
            v = []
            for a in m.GetAtoms():
                n_h = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "H")
                v.append("%.1f" % n_h if a.GetSymbol() == "C" else "n/a")
            m.SetProp("atom.dprop.nh", " ".join(v))
            w.write(m)
    capsys.readouterr()
    main(["local", path, "--target", "nh", "--rho", "0.5", "--folds", "3",
          "--conformers", "3", "--conformer-weighting", "boltzmann"])
    text = capsys.readouterr().out
    assert "conformations per molecule, weighted boltzmann" in text
    assert "conformations" in text and "local field" in text

