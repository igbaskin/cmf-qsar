"""The crown ether series of plan item SC5: the table, the files and the benchmark's own tools.

* The table holds 79 hosts; each named isomer is that isomer, and the file is the table.
* The ring counts read the formula: ring size, donors, fused aromatic rings, side arms, bicyclic.
* The two geometry files hold the same hosts, each in the frame of its cavity.
* Turning a host to the side of its ion is a half turn about x and nothing else.
* The ridge of the benchmark is the program's own kernel ridge regression.
* Nested cross-validation recovers a target that is linear in the kernel.
* The plane profile of a co-field finds the radius of a ring put there.
"""
import csv
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "examples"))

pytest.importorskip("rdkit")
from rdkit import Chem                                                  # noqa: E402

import make_crown_examples as MK                                        # noqa: E402
import run_crowns as R                                                  # noqa: E402
from cmfqsar import cavity as C                                         # noqa: E402
from cmfqsar.model import krr_fit                                       # noqa: E402

CSV = os.path.join(ROOT, "examples", "crowns", "crowns.csv")


def test_the_table_is_checked_and_the_file_is_the_table():
    MK.check()
    rows = MK.rows()
    assert len(rows) == 79
    with open(CSV, encoding="utf-8") as fh:
        stored = list(csv.DictReader(fh))
    assert [r["abbreviation"] for r in stored] == [r["abbreviation"] for r in rows]
    for a, b in zip(stored, rows):
        assert a["smiles"] == b["smiles"]
        assert float(a["logK_Na"]) == pytest.approx(b["logK_Na"])
        assert float(a["logK_K"]) == pytest.approx(b["logK_K"])
    by = {r["abbreviation"]: r for r in rows}
    # the median of an even number of values is the mean of the two in the middle
    assert by["18C6"]["logK_K"] == pytest.approx(6.09)
    assert by["18C6"]["logK_Na"] == pytest.approx(4.35)
    assert by["21C7"]["logK_Na"] == pytest.approx((1.73 + 2.54) / 2)
    assert by["2.2.2"]["medium"].startswith("methanol, 0.05 M")
    assert by["2.1.1"]["selectivity_K_Na"] < 0 < by["18C6"]["selectivity_K_Na"]


def test_the_ring_counts_read_the_formula():
    def counts(abbr):
        smiles = next(h[3] for h in MK.HOSTS if h[1] == abbr)
        return R.ring_counts(smiles)
    assert counts("18C6") == [18, 6, 0, 0, 0, 0, 0.0]
    assert counts("B2-18C6") == [18, 6, 0, 0, 2, 0, 0.0]
    assert counts("T18C6") == [18, 6, 0, 1, 0, 0, 0.0]
    assert counts("2.2.2") == [18, 8, 2, 0, 0, 0, 1.0]
    assert counts("MeOEA18C6") == [18, 6, 1, 0, 0, 1, 0.0]
    # the oxygens of the two ester carbonyls are not donors on a side arm
    assert counts("K2-18C6") == [18, 6, 0, 0, 0, 0, 0.0]
    # a fused pyridine ring is counted with the fused benzene rings
    assert counts("K2Py18C6") == [18, 6, 1, 0, 1, 0, 0.0]


def _files():
    return [os.path.join(ROOT, "examples", "crowns", f"hosts_{i}.sdf") for i in R.IONS]


@pytest.mark.skipif(not all(os.path.exists(p) for p in _files()),
                    reason="the geometries are not built")
def test_both_geometry_files_hold_the_hosts_in_their_cavity_frames():
    d = R.load()
    assert len(d["names"]) == 79
    assert d["names"] == [h[1] for h in MK.HOSTS]
    for ion in R.IONS:
        for m, p in zip(d["mols"][ion], d["ion_xyz"][ion]):
            xyz = m.GetConformer().GetPositions()
            donors = C.donor_atoms(m)
            assert np.allclose(xyz[donors].mean(axis=0), 0.0, atol=2e-3)
            # the stored frame is the one the align command gives
            assert np.allclose(C.cavity_frame(m), xyz, atol=2e-3)
            assert np.linalg.norm(p) < 6.0
    assert np.allclose(d["y"]["selectivity"], d["y"]["K"] - d["y"]["Na"])


def test_turning_to_the_ion_is_a_half_turn_about_x():
    m = Chem.AddHs(Chem.MolFromSmiles("C1COCCOCCOCCO1"))
    from rdkit.Chem import AllChem
    assert AllChem.EmbedMolecule(m, randomSeed=1) == 0
    xyz = m.GetConformer().GetPositions()
    below, above = np.array([0.3, 0.2, -1.0]), np.array([0.3, 0.2, 1.0])
    a, b = R.turned_to_ion([m, m], [below, above])
    assert np.allclose(a.GetConformer().GetPositions(), xyz * [1.0, -1.0, -1.0])
    assert np.allclose(b.GetConformer().GetPositions(), xyz)


def test_the_ridge_is_the_program_ridge_and_the_nested_loop_recovers_a_linear_target():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 3))
    K = X @ X.T
    y = X @ [1.0, -2.0, 0.5] + 0.7
    a, b = R.ridge(K, y, 0.1)
    a2, b2 = krr_fit(K, y, 0.1, True)
    assert np.allclose(a, a2) and b == pytest.approx(b2)
    noise = rng.normal(size=(40, 40))
    folds = R.k_folds(40, 5, 0)
    assert sorted(np.concatenate(folds).tolist()) == list(range(40))
    pred, ref, chosen = R.nested({("good",): K, ("bad",): noise @ noise.T}, y, folds, 0)
    assert all(c[0] == ["good"] for c in chosen)
    assert R.q2_of(y, pred, ref) > 0.99
    fam = ["a"] * 10 + ["b"] * 25 + ["c"] * 5
    held = R.design_folds("family", fam)
    assert [len(h) for h in held] == [10, 25, 5]


def test_the_plane_profile_finds_the_radius_of_a_ring():
    pts = R.cylinder_points()
    rho = np.hypot(pts[:, 0], pts[:, 1])
    C_ring = np.exp(-((rho - 2.75) ** 2 + pts[:, 2] ** 2) / 0.5)
    prof = R.plane_profile(C_ring)
    assert prof.shape == R.RHO.shape
    assert R.peak(prof, 1.0)[0] == pytest.approx(2.75)
    assert R.peak(-prof, -1.0)[0] == pytest.approx(2.75)
    # a profile that is negative inside 2.6 A and positive outside turns there, and one that
    # never becomes positive has no such radius
    line = R.RHO - 2.6
    assert R.sign_change(line) == pytest.approx(2.6)
    assert np.isnan(R.sign_change(-np.ones_like(R.RHO)))


def test_the_kernel_with_the_amounts_kept_gives_the_larger_host_the_longer_row():
    from rdkit.Chem import AllChem
    mols = []
    for smiles in ("C1COCCOCCOCCO1", "C1COCCOCCOCCOCCOCCO1", "C1COCCOCCOCCOCCO1"):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(m, randomSeed=3) == 0
        mols.append(C.with_cavity_frame(m))
    names = ["a", "b", "c"]
    K = R.amounts_kernel(mols, names, 1.0)
    assert K[1, 1] > K[2, 2] > K[0, 0]               # 18-crown-6, 15-crown-5, 12-crown-4
    Kn = R.field_kernels(mols, names, alpha=1.0)[0]
    # the cosine-normalised kernels have a unit diagonal where the field exists
    d = np.array([np.diag(k) for k in Kn])
    assert np.allclose(d[d > 0], 1.0)
