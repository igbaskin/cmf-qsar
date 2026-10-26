"""examples/uma_fe110_adsorption.py without the GPU or the UMA weights: the slab builder, the
anchor atom, the pose placement, and the whole run with a stub calculator, down to an SDF that
cmfqsar loads as a data set."""
import csv
import json
import os
import sys

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from rdkit import Chem  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))

import uma_fe110_adsorption as ufa  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402


class HeightSpring(Calculator):
    """A substitute for UMA: -4 eV per Fe, -1 eV per molecule atom, and for an adsorbed molecule a
    -0.5 eV bond plus a spring that pulls the molecule's centroid to `z0` A above the top Fe atom
    and a small penalty on its vertical extent (so the flat pose wins and the forces converge)."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, k=10.0, z0=2.0, spread=0.02):
        super().__init__()
        self.k, self.z0, self.spread = k, z0, spread
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        pos = atoms.positions
        is_fe = atoms.numbers == 26
        forces = np.zeros_like(pos)
        energy = -4.0 * is_fe.sum() - 1.0 * (~is_fe).sum()
        if is_fe.any() and (~is_fe).any():
            mol = np.flatnonzero(~is_fe)
            z_top = pos[is_fe, 2].max()
            dz = pos[mol, 2].mean() - z_top - self.z0
            energy += -0.5 + self.k * dz ** 2
            forces[mol, 2] = -2.0 * self.k * dz / len(mol)
            top, bottom = mol[np.argmax(pos[mol, 2])], mol[np.argmin(pos[mol, 2])]
            energy += self.spread * (pos[top, 2] - pos[bottom, 2])
            forces[top, 2] -= self.spread
            forces[bottom, 2] += self.spread
        self.results = {"energy": energy, "forces": forces}


def test_slab_geometry():
    slab = ufa.build_fe110_slab(6, 6, 4, a=2.83, vacuum=15.0, fixed_layers=2)
    assert len(slab) == 144 and set(slab.get_chemical_symbols()) == {"Fe"}
    assert sorted(set(slab.get_tags())) == [1, 2, 3, 4]
    lengths = slab.cell.lengths()
    assert lengths[0] == pytest.approx(6 * 2.83, abs=1e-6)
    assert lengths[1] == pytest.approx(6 * 2.83 / np.sqrt(2), abs=1e-6)
    rigid = ufa.build_fe110_slab()                                              # the default: all fixed
    assert len(rigid.constraints[0].index) == 144 and rigid.cell.lengths()[0] == pytest.approx(6 * 2.78)
    assert np.allclose(slab.cell.angles(), 90.0)
    thickness = slab.positions[:, 2].max() - slab.positions[:, 2].min()
    assert lengths[2] - thickness == pytest.approx(15.0, abs=1e-6)          # the vacuum between images
    fixed = slab.constraints[0].index
    assert len(fixed) == 72 and set(slab.get_tags()[fixed]) == {3, 4}       # the bottom two layers
    assert ufa.top_layer_z(slab) == pytest.approx(slab.positions[:, 2].max())
    assert np.allclose(slab.get_initial_magnetic_moments(), 2.2) and all(slab.pbc)


def test_cell_grows_with_the_footprint():
    assert ufa.cell_for_footprint((9.0, 4.0)) == (6, 6)                      # benzimidazole-sized: fits
    nx, ny = ufa.cell_for_footprint((14.0, 8.0))
    slab = ufa.build_fe110_slab(nx, ny)
    assert nx > 6 and ny > 6
    assert slab.cell.lengths()[0] >= 14.0 + 6.0 and slab.cell.lengths()[1] >= 8.0 + 6.0
    assert ufa.cell_for_footprint((14.0, 8.0), nx=8, ny=8) == (8, 8)        # never below the given size


@pytest.mark.parametrize("smiles,expected", [
    ("c1ccc2[nH]cnc2c1", "pyridine-type N"),       # benzimidazole: the N without the H
    ("Nc1nc2ccccc2[nH]1", "pyridine-type N"),      # 2-aminobenzimidazole: the ring N, not the amine
    ("CCN(CC)CC", "amine"),
    ("c1ccccc1S", "S"),
    ("c1ccccc1", None),
])
def test_anchor_atom(smiles, expected):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    idx = ufa.anchor_atom(mol)
    if expected is None:
        assert idx is None
        return
    atom = mol.GetAtomWithIdx(idx)
    if expected == "pyridine-type N":
        assert atom.GetSymbol() == "N" and atom.GetIsAromatic() and atom.GetTotalNumHs() == 0
    elif expected == "amine":
        assert atom.GetSymbol() == "N" and not atom.GetIsAromatic()
    else:
        assert atom.GetSymbol() == expected


def test_neutral_form_and_embedding():
    m = ufa.embed_smiles("C[NH3+]")                                           # a protonated amine
    assert sum(a.GetFormalCharge() for a in m.GetAtoms()) == 0
    assert m.GetNumConformers() == 1 and m.GetNumAtoms() == 7
    m = ufa.embed_smiles("O=[N+]([O-])c1ccccc1")                               # nitro: stays as it is
    assert sum(a.GetFormalCharge() for a in m.GetAtoms()) == 0
    assert all(a.GetIsAromatic() for a in m.GetAtoms() if a.GetAtomicNum() == 6)   # RDKit's model survives MMFF


def test_pose_orientations_and_placement():
    mol = ufa.embed_smiles("c1ccc2[nH]cnc2c1", seed=1)
    coords = ufa.mol_coords(mol)
    anchor = ufa.anchor_atom(mol)
    rng = np.random.default_rng(0)
    flat = ufa.orient(coords, "flat", anchor, rng)
    span = flat.max(axis=0) - flat.min(axis=0)
    assert span[2] < 0.3 and span[0] >= span[1] > 1.0          # in the xy plane, long axis along x
    assert ufa.footprint(flat) == pytest.approx((span[0], span[1]))
    up = ufa.orient(coords, "upright", anchor, rng, mol)
    assert np.argmin(up[:, 2]) == anchor                         # the N is the lowest atom
    assert np.allclose(up[anchor, :2], 0.0, atol=1e-6)           # and sits over the origin
    lp = ufa.lone_pair_direction(mol, up, anchor)
    angle = np.degrees(np.arccos(-lp[2] / np.linalg.norm(lp)))
    assert angle < 30.0                                          # lone pair roughly at the surface
    r1 = ufa.orient(coords, "random", anchor, np.random.default_rng(7))
    r2 = ufa.orient(coords, "random", anchor, np.random.default_rng(7))
    assert np.allclose(r1, r2)                                   # reproducible
    for arr in (flat, up, r1):                                   # rigid: all distances kept
        d0 = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
        d1 = np.linalg.norm(arr[:, None] - arr[None], axis=-1)
        assert np.allclose(d0, d1, atol=1e-6)
    with pytest.raises(ValueError):
        ufa.orient(coords, "upright", None, rng)

    slab = ufa.build_fe110_slab()
    system = ufa.place_on_slab(slab, mol, flat, height=2.5)
    assert len(system) == len(slab) + mol.GetNumAtoms()
    assert system.get_chemical_symbols()[len(slab):] == [a.GetSymbol() for a in mol.GetAtoms()]
    mol_pos = system.positions[len(slab):]
    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    assert mol_pos[heavy, 2].min() == pytest.approx(ufa.top_layer_z(slab) + 2.5)
    assert np.allclose(mol_pos[:, :2].mean(axis=0), slab.cell.lengths()[:2] / 2.0, atol=1e-6)
    upright = ufa.place_on_slab(slab, mol, up, height=2.0)
    assert np.allclose(upright.positions[len(slab) + anchor, :2], slab.cell.lengths()[:2] / 2.0, atol=1e-6)
    assert upright.positions[len(slab) + anchor, 2] == pytest.approx(ufa.top_layer_z(slab) + 2.0)
    assert len(system.constraints[0].index) == 144               # the fixed layers survive


def test_docking_stage_holds_then_releases_the_heavy_atoms():
    mol = ufa.embed_smiles("c1ccc2[nH]cnc2c1", seed=1)
    slab = ufa.build_fe110_slab()
    flat = ufa.orient(ufa.mol_coords(mol), "flat", None, np.random.default_rng(0), mol)
    system = ufa.place_on_slab(slab, mol, flat, height=3.0)
    n = len(slab)
    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    z0 = system.positions[n:][heavy, 2].copy()
    calc = HeightSpring()                                            # pulls the molecule to 2.0 A
    e, steps, conv = ufa.dock_and_relax(system, n, mol, calc, fmax=0.05, steps=0, dock_steps=30)
    assert np.allclose(system.positions[n:][heavy, 2], z0)          # held during the docking stage
    assert steps > 0 and len(system.constraints) == 1               # the plane constraint is gone
    e, steps2, conv = ufa.dock_and_relax(system, n, mol, calc, fmax=0.05, steps=60, dock_steps=0)
    assert conv and not np.allclose(system.positions[n:][heavy, 2], z0)         # released
    assert abs(system.positions[n:, 2].mean() - ufa.top_layer_z(system) - 2.0) < 0.1   # the spring dominates


def test_bonds_intact_flags_a_dissociated_molecule():
    mol = ufa.embed_smiles("CO")
    c = ufa.mol_coords(mol)
    assert ufa.bonds_intact(mol, c, c)
    broken = c.copy()
    broken[0] += [3.0, 0.0, 0.0]                                   # carbon pulled 3 A away
    assert not ufa.bonds_intact(mol, c, broken)


def _args(out, **over):
    args = ufa.build_parser().parse_args(["in.csv", "-o", out])
    for k, v in over.items():
        setattr(args, k, v)
    return ufa.prepare_args(args)


def test_run_with_a_stub_calculator(tmp_path):
    rows = [{"smiles": "c1ccc2[nH]cnc2c1", "name": "Benzimidazole", "activity": "5.61", "mw": "118.1"},
            {"smiles": "Cc1nc2ccccc2[nH]1", "name": "2-Methylbenzimidazole", "activity": "6.7", "mw": "132.2"}]
    out = str(tmp_path / "node.sdf")
    calc = HeightSpring()
    args = _args(out, steps=60, dock_steps=0)      # the stub has no bonds: docking is tested on its own
    records = ufa.run(rows, out, calc, args, columns=list(rows[0].keys()), log=lambda s: None)
    assert [r["name"] for r in records] == ["Benzimidazole", "2-Methylbenzimidazole"]
    for r in records:
        assert r["e_slab_eV"] == pytest.approx(-4.0 * 144)
        assert r["e_mol_eV"] == pytest.approx(-1.0 * Chem.AddHs(Chem.MolFromSmiles(r["smiles"])).GetNumAtoms())
        assert r["e_ads_eV"] < 0 and r["e_ads_eV"] == pytest.approx(-0.5, abs=0.05)
        assert r["e_ads_kJmol"] == pytest.approx(r["e_ads_eV"] * 96.485332)
        assert r["best"] == "flat"                                    # the smallest vertical extent
        assert {p["label"] for p in r["poses"]} == {"flat", "upright", "random"}
        assert all(p["intact"] and p["bound"] and p["converged"] for p in r["poses"])
        assert r["e_ads_eV"] == min(p["e_ads_eV"] for p in r["poses"])
        assert r["cell"] == [6, 6, 4]

    # the SDF: the adsorbed molecules in the surface frame, loadable as a labelled data set
    ds = load_dataset(out, activity_prop="E_ads_kJmol")
    assert ds.names == ["Benzimidazole", "2-Methylbenzimidazole"]
    assert np.allclose(ds.activity, [r["e_ads_kJmol"] for r in records], atol=0.01)
    for m, r in zip(ds.mols, records):
        assert m.GetProp("pose_label") == "flat" and int(m.GetProp("n_steps")) >= 1
        assert float(m.GetProp("E_ads_eV")) == pytest.approx(r["e_ads_eV"], abs=1e-4)
        assert m.GetProp("activity") == r["row"]["activity"] and m.GetProp("mw") == r["row"]["mw"]
        assert m.GetProp("model") == f"{args.model}/{args.task}"
        z = np.array(m.GetConformer().GetPositions())[:, 2]
        heavy = np.array([a.GetAtomicNum() > 1 for a in m.GetAtoms()])
        assert abs(z.mean() - 2.0) < 0.1                               # the spring's height, over z = 0
        assert z[heavy].max() - z[heavy].min() < 0.3                   # flat (the methyl H protrude)
        assert Chem.MolToSmiles(Chem.RemoveHs(m)) == Chem.MolToSmiles(Chem.MolFromSmiles(r["smiles"]))

    # the CSV summary and the per-molecule records
    with open(os.path.splitext(out)[0] + ".csv", newline="") as fh:
        table = list(csv.DictReader(fh))
    assert [t["name"] for t in table] == ds.names
    assert table[0]["pose_label"] == "flat" and float(table[0]["E_ads_eV"]) == pytest.approx(records[0]["e_ads_eV"], abs=1e-4)
    assert table[0]["activity"] == "5.61" and table[0]["poses_eV"].startswith("flat:") and table[0]["bound"] == "1"
    assert table[0]["gas_intact"] == "1" and ds.mols[0].GetProp("gas_intact") == "1"
    work = os.path.splitext(out)[0] + "_work"
    assert os.path.exists(os.path.join(work, "Benzimidazole.json"))
    assert os.path.exists(os.path.join(work, "slab_6x6x4_fixed4.json"))

    # a second run reuses the records without touching the calculator
    calls = calc.calls
    again = ufa.run(rows, out, calc, args, columns=list(rows[0].keys()), log=lambda s: None)
    assert calc.calls == calls
    assert [r["e_ads_eV"] for r in again] == [r["e_ads_eV"] for r in records]
    assert load_dataset(out, activity_prop="E_ads_kJmol").names == ds.names


def test_upright_pose_without_an_anchor_becomes_random(tmp_path):
    rows = [{"smiles": "c1ccccc1", "name": "Benzene"}]
    out = str(tmp_path / "benzene.sdf")
    args = _args(out, steps=60, dock_steps=0)
    records = ufa.run(rows, out, HeightSpring(), args, log=lambda s: None)
    labels = [p["label"] for p in records[0]["poses"]]
    assert labels == ["flat", "random", "random"]
    rec = json.load(open(os.path.join(os.path.splitext(out)[0] + "_work", "Benzene.json")))
    assert rec["anchor_atom"] is None and rec["best"] == "flat"
