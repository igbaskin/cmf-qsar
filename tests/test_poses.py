"""The protocol that judges an alignment: the pose measure, the control, and the two references."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from cmfqsar.fields import FieldSet
from cmfqsar.poses import POSERS, PoseScore, judge, pose_rmsd, table


def _series(n=12):
    """A congeneric series in one frame, standing in for a set of crystal poses."""
    from cmfqsar.align import align_dataset

    subs = ["", "C", "CC", "F", "Cl", "O", "OC", "N", "NC", "C(F)(F)F", "C#N", "CO",
            "Br", "CCC"][:n]
    mols, names = [], []
    for i, s in enumerate(subs):
        m = Chem.AddHs(Chem.MolFromSmiles(f"c1ccc(cc1){s}" if s else "c1ccccc1"))
        AllChem.EmbedMolecule(m, randomSeed=20 + i)
        AllChem.MMFFOptimizeMolecule(m)
        m.SetProp("_Name", f"m{i}")
        mols.append(m)
        names.append(f"m{i}")
    align_dataset(mols, 0)                      # one frame: these are the "crystal" poses
    rng = np.random.default_rng(3)
    y = np.array([4.0 + 0.3 * m.GetNumHeavyAtoms() for m in mols]) + 0.2 * rng.normal(size=len(mols))
    return mols, names, y


def _shift(mol, by=(1.0, 0.0, 0.0)):
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + by[0], p.y + by[1], p.z + by[2]))
    return out


def test_pose_rmsd_measures_the_pose_and_not_the_conformation():
    """Superposing before measuring would report a perfect score for a displaced molecule.

    This is the single mistake that would make the whole protocol meaningless, so it is asserted
    directly against the function that makes it.
    """
    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    assert pose_rmsd(m, m) == pytest.approx(0.0, abs=1e-9)
    moved = _shift(m, (1.0, 0.0, 0.0))
    assert pose_rmsd(moved, m) == pytest.approx(1.0, abs=1e-6)
    assert rdMolAlign.GetBestRMS(Chem.RemoveHs(moved), Chem.RemoveHs(m)) == pytest.approx(0.0, abs=1e-6)


def test_pose_rmsd_is_symmetry_aware():
    """Benzene turned by 60 degrees about its own axis is in the same place, and must score 0."""
    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(m, randomSeed=2)
    AllChem.MMFFOptimizeMolecule(m)
    P = np.asarray(m.GetConformer().GetPositions(), float)
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    centre = P[heavy].mean(axis=0)
    normal = np.linalg.svd(P[heavy] - centre)[2][-1]
    th = np.radians(60.0)
    K = np.array([[0, -normal[2], normal[1]], [normal[2], 0, -normal[0]],
                  [-normal[1], normal[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    turned = Chem.Mol(m)
    conf = turned.GetConformer()
    for i, p in enumerate((P - centre) @ R.T + centre):
        conf.SetAtomPosition(i, p.tolist())
    assert pose_rmsd(turned, m) == pytest.approx(0.0, abs=1e-5)


def test_the_floor_is_the_scrambled_poses_and_the_ceiling_is_the_crystal_ones():
    mols, names, y = _series()
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    tr, te = list(range(8)), list(range(8, len(mols)))
    out = judge(mols, names, y, tr, te, fs, methods=("scrambled",), seed=0)
    assert set(out) == {"crystal", "scrambled"}
    assert np.allclose(out["crystal"].rmsd, 0.0)
    assert out["crystal"].within(1.0) == 1.0
    floor = out["scrambled"]
    assert floor.median_rmsd > 1.0, "the control must actually move the molecules"
    assert abs(floor.recovery) < 1e-9, "doing nothing recovers nothing"
    # and the crystal poses are not disturbed by the run
    assert np.allclose(out["crystal"].observed, y[te])


def test_a_method_that_restores_the_crystal_pose_scores_the_ceiling():
    """The protocol must give full marks to a perfect poser, and the floor to an idle one.

    Together with the previous test this pins both ends of the scale, which is what makes the
    numbers in between mean anything.
    """
    mols, names, y = _series()
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    tr, te = list(range(8)), list(range(8, len(mols)))
    crystal = {i: Chem.Mol(mols[i]) for i in te}
    order = list(te)

    def restore(probe, ctx, _order=order, _crystal=crystal):
        from cmfqsar.align import set_coords
        for p, i in zip(probe, _order):
            set_coords(p, np.asarray(_crystal[i].GetConformer().GetPositions(), float))

    out = judge(mols, names, y, tr, te, fs, methods=("scrambled", "perfect"),
                posers={**POSERS, "perfect": restore}, seed=0)
    perfect = out["perfect"]
    assert perfect.median_rmsd == pytest.approx(0.0, abs=1e-6)
    assert perfect.within(1.0) == 1.0
    assert perfect.recovery == pytest.approx(1.0, abs=1e-6)
    # identical poses must give identical predictions, so the ceiling is reached exactly
    assert perfect.r2_pred == pytest.approx(out["crystal"].r2_pred, abs=1e-9)
    assert out["scrambled"].median_rmsd > perfect.median_rmsd


def test_every_method_starts_from_scrambled_poses():
    """A poser is never handed the crystal poses, or doing nothing would look like success."""
    seen = {}

    def spy(probe, ctx):
        seen["start"] = [np.asarray(p.GetConformer().GetPositions(), float) for p in probe]

    mols, names, y = _series()
    fs = FieldSet.create(physchem=["steric"], alpha=0.5)
    tr, te = list(range(8)), list(range(8, len(mols)))
    crystal = [np.asarray(mols[i].GetConformer().GetPositions(), float) for i in te]
    judge(mols, names, y, tr, te, fs, methods=("spy",), posers={"spy": spy}, seed=0)
    assert "start" in seen
    moved = [np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))) for a, b in zip(seen["start"], crystal)]
    assert min(moved) > 0.5, "the poser was handed poses that had not been scrambled"


def test_the_named_methods_run_and_are_reported():
    mols, names, y = _series()
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    tr, te = list(range(8)), list(range(8, len(mols)))
    out = judge(mols, names, y, tr, te, fs, methods=("scrambled", "mcs"), seed=0, restarts=0)
    assert out["mcs"].n == len(te)
    assert np.all(np.isfinite(out["mcs"].rmsd))
    text = table(out, title="unit test")
    assert "median RMSD" in text and "mcs" in text and "crystal" in text
    assert text.count("\n") >= 5


def test_a_regenerated_conformation_can_still_be_matched_by_substructure():
    """MMFF set-up alters aromaticity in place, and the MCS then fails to match its own molecule.

    The failure is silent -- the aligner raises, the poser swallows it, and the molecule stays
    where it was, which reads as "substructure alignment achieves nothing" instead of "the
    substructure alignment never ran".  It cost five of sixteen sets in the first flexible run.
    """
    from cmfqsar.align import mcs_align
    from cmfqsar.poses import regenerate_conformation

    ref = Chem.AddHs(Chem.MolFromSmiles("c1ccc2[nH]c(nc2c1)C(=O)Nc1ccccc1"))
    AllChem.EmbedMolecule(ref, randomSeed=5)
    AllChem.MMFFOptimizeMolecule(ref)
    probe = Chem.AddHs(Chem.MolFromSmiles("c1ccc2[nH]c(nc2c1)C(=O)Nc1ccc(F)cc1"))
    AllChem.EmbedMolecule(probe, randomSeed=6)

    fresh = regenerate_conformation(probe, seed=1)
    assert fresh.GetNumConformers() == 1
    rmsd, n = mcs_align(fresh, ref)          # must not raise
    assert n > 10, "the common substructure of two benzimidazole amides is large"
    assert rmsd < 2.0

    # and the molecule really moved to the reference, which is what the silent failure prevented
    centre_ref = np.asarray(ref.GetConformer().GetPositions(), float).mean(axis=0)
    centre_new = np.asarray(fresh.GetConformer().GetPositions(), float).mean(axis=0)
    assert float(np.linalg.norm(centre_ref - centre_new)) < 3.0


def test_a_training_frame_is_judged_up_to_a_global_rigid_motion():
    """Rotating the whole set changes no kernel entry, so it must not count as pose error.

    Measured per molecule without this, a perfectly recovered mutual arrangement would score as
    badly as a destroyed one, and the C4 measurement would be meaningless.
    """
    from scipy.spatial.transform import Rotation

    from cmfqsar.poses import frame_rmsds, global_superpose

    mols, names, y = _series(10)
    ref = [Chem.Mol(m) for m in mols]
    turned = []
    R = Rotation.from_rotvec([0.4, -0.2, 0.7]).as_matrix()
    shift = np.array([3.0, -1.0, 2.0])
    for m in mols:                                  # one motion applied to the whole set
        p = Chem.Mol(m)
        X = np.asarray(p.GetConformer().GetPositions(), float)
        MW_set_coords(p, X @ R.T + shift)
        turned.append(p)
    assert pose_rmsd(turned[0], ref[0]) > 2.0, "the molecules did move"
    after = frame_rmsds(turned, ref)
    assert float(np.max(after)) < 1e-6, "a global motion is not pose error"

    # a single molecule displaced within the set is pose error, and survives the fit
    broken = [Chem.Mol(m) for m in mols]
    X = np.asarray(broken[3].GetConformer().GetPositions(), float)
    MW_set_coords(broken[3], X + np.array([2.5, 0.0, 0.0]))
    rms = frame_rmsds(broken, ref)
    assert float(np.median(rms)) < 0.5 and float(np.max(rms)) > 1.5


def MW_set_coords(mol, X):
    from cmfqsar.align import set_coords
    set_coords(mol, np.asarray(X, float))
