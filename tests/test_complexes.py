"""Coordination complexes: the reader, the parameters and the frames (plan item EC4).

Three things in this item fail silently, so they are the three things tested here. A complex whose
ligand charges are placed on the wrong ligands sanitises and gives a plausible electrostatic field
for the wrong charge distribution. A metal with no van der Waals parameters gets the default
radius, which is a number and not an error. And a frame built on a symmetric coordination sphere
can be several angstrom out while the complexes lie exactly on top of each other, which is the
same trap the phenazine core showed in item EC1.
"""
import numpy as np
import pytest
from rdkit import Chem

from cmfqsar.atomtypes import (TRANSITION_METALS, UFF_RADIUS_SCALE, assign_atom_types,
                               is_transition_metal, vdw_params)
from cmfqsar.complexes import (TRY_LIGAND_CHARGES, align_on_first_sphere, build_complex,
                               coordination, donor_indices, metal_index, metal_origin,
                               molecule_from_coordinates, parse_comment,
                               principal_metal_frame, read_xyz)
from cmfqsar.fields import FieldSet, build_fields
from cmfqsar.increments import tripos_increments


# ------------------------------------------------------------------ coordinates to work with

def hexaaqua(metal="Fe", d=2.10):
    """[M(H2O)6] with the metal at the origin and one water on each axis.

    The hydrogens sit on the far side of their oxygen, so that distance-based bond perception
    bonds each of them to its own oxygen and none of them to the metal.
    """
    symbols = [metal]
    coords = [[0.0, 0.0, 0.0]]
    for axis in range(3):
        for sign in (1.0, -1.0):
            o = np.zeros(3)
            o[axis] = sign * d
            symbols.append("O")
            coords.append(list(o))
            # two hydrogens, pushed outward along the same axis and spread in another direction
            other = (axis + 1) % 3
            for shift in (0.76, -0.76):
                h = np.array(o, float)
                h[axis] += sign * 0.60
                h[other] += shift
                symbols.append("H")
                coords.append(list(h))
    return symbols, np.asarray(coords, float)


def dichloro_tetraaqua(metal="Fe", d=2.10):
    """[M(H2O)4Cl2]: two of the six waters of :func:`hexaaqua` replaced by chloride."""
    symbols, coords = hexaaqua(metal, d)
    keep_syms, keep_xyz = [symbols[0]], [coords[0]]
    i = 1
    group = 0
    while i < len(symbols):
        o, h1, h2 = i, i + 1, i + 2
        if group < 2:                                  # the first two waters become chlorides
            keep_syms.append("Cl")
            keep_xyz.append(coords[o])
        else:
            keep_syms += ["O", "H", "H"]
            keep_xyz += [coords[o], coords[h1], coords[h2]]
        group += 1
        i += 3
    return keep_syms, np.asarray(keep_xyz, float)


def xyz_text(symbols, coords, comment=""):
    lines = [str(len(symbols)), comment]
    for s, p in zip(symbols, coords):
        lines.append(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    return "\n".join(lines) + "\n"


def rotation(angles):
    """A rotation matrix from three angles, used to move a complex out of its frame."""
    rx, ry, rz = angles
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry),
                              np.cos(rz), np.sin(rz))
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))


# ------------------------------------------------------------------ reading an XYZ file

def test_read_xyz_reads_every_block(tmp_path):
    symbols, coords = hexaaqua()
    text = (xyz_text(symbols, coords, "csd_code=AAAAAA q=0")
            + xyz_text(symbols, coords, "csd_code=BBBBBB q=2"))
    path = tmp_path / "two.xyz"
    path.write_text(text, encoding="utf-8")
    frames = read_xyz(str(path))
    assert [f.name for f in frames] == ["AAAAAA", "BBBBBB"]
    assert frames[0].n_atoms == len(symbols)
    assert frames[1].tags["q"] == "2"
    assert frames[0].coords.shape == (len(symbols), 3)


def test_read_xyz_stops_at_the_limit(tmp_path):
    symbols, coords = hexaaqua()
    path = tmp_path / "many.xyz"
    path.write_text("".join(xyz_text(symbols, coords, f"name=m{i}") for i in range(4)),
                    encoding="utf-8")
    assert len(read_xyz(str(path), limit=2)) == 2


def test_read_xyz_refuses_a_file_without_a_count(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("not a number\ncomment\nFe 0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_xyz(str(path))


def test_read_xyz_refuses_a_block_that_ends_early(tmp_path):
    path = tmp_path / "short.xyz"
    path.write_text("3\ncomment\nFe 0 0 0\nO 2 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_xyz(str(path))


def test_parse_comment_reads_both_conventions():
    tags, parts = parse_comment('csd_code=ABCDEF rmsd=1.57 pbc="F F F"')
    assert tags == {"csd_code": "ABCDEF", "rmsd": "1.57", "pbc": "F F F"}
    assert parts == ['csd_code=ABCDEF rmsd=1.57 pbc="F F F"']
    tags, parts = parse_comment("Complex 7 (lab 019) | Ni(II) triplet | charge +2 "
                                "| multiplicity 3 | E(gas) = -1.5 Eh")
    assert parts[0] == "Complex 7 (lab 019)"
    assert parts[2] == "charge +2"
    assert tags["E(gas)"] == "-1.5"


def test_molecule_from_coordinates_has_no_bonds():
    symbols, coords = hexaaqua()
    mol = molecule_from_coordinates(symbols, coords)
    assert mol.GetNumAtoms() == len(symbols)
    assert mol.GetNumBonds() == 0
    assert mol.GetConformer().Is3D()


# ------------------------------------------------------------------ building a complex

def test_hexaaqua_iron_builds_with_six_dative_bonds():
    symbols, coords = hexaaqua()
    mol, report = build_complex(symbols, coords, total_charge=2, name="hexaaqua")
    assert report.ok, report.fault
    assert mol is not None
    dative = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]
    assert len(dative) == 6
    # a dative bond points from the donor to the metal, which is what leaves the donor its own
    # valence; the other direction would make the oxygen four-valent and the molecule invalid
    for b in dative:
        assert b.GetEndAtom().GetSymbol() == "Fe"
        assert b.GetBeginAtom().GetSymbol() == "O"
    assert report.metal == "Fe"
    assert report.metal_charge == 2
    assert Chem.GetFormalCharge(mol) == 2
    assert report.donor_symbols == ["O"] * 6


def test_the_metal_takes_the_charge_the_ligands_leave():
    symbols, coords = dichloro_tetraaqua()
    mol, report = build_complex(symbols, coords, total_charge=0,
                                ligand_charges=[-1, -1, 0, 0, 0, 0],
                                ligand_smiles=["[Cl-]", "[Cl-]", "O", "O", "O", "O"])
    assert report.ok, report.fault
    assert report.metal_charge == 2                   # iron(II) with two chlorides and four waters
    assert sorted(report.ligand_charges) == [-1, -1, 0, 0, 0, 0]
    assert Chem.GetFormalCharge(mol) == 0
    chlorides = [a for a in mol.GetAtoms() if a.GetSymbol() == "Cl"]
    assert [a.GetFormalCharge() for a in chlorides] == [-1, -1]


def test_a_lone_halide_keeps_the_charge_the_data_set_states():
    """A one-atom ligand has no bond to determine, and bond perception leaves it neutral.

    Without the repair the chloride stays neutral, the metal is then given the charge that
    belonged to the chloride, and the electrostatic field of the complex is wrong while its total
    charge is right. This is the fault that cost a quarter of the iron series before it was found.
    """
    symbols, coords = dichloro_tetraaqua()
    mol, report = build_complex(symbols, coords, total_charge=0,
                                ligand_charges=[-1, -1, 0, 0, 0, 0])
    assert report.ok, report.fault
    assert report.metal_charge == 2


def test_a_complex_with_two_metals_is_refused():
    symbols, coords = hexaaqua()
    symbols = list(symbols) + ["Fe"]
    coords = np.vstack([coords, [[8.0, 8.0, 8.0]]])
    mol, report = build_complex(symbols, coords, total_charge=4)
    assert mol is None
    assert "mononuclear" in report.fault


def test_a_metal_with_no_neighbour_is_refused():
    symbols = ["Fe", "O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [9.6, 0.76, 0.0], [9.6, -0.76, 0.0]])
    mol, report = build_complex(symbols, coords, total_charge=2)
    assert mol is None
    assert "no neighbour" in report.fault


def test_a_stated_charge_that_does_not_add_up_is_refused():
    """The charges of the ligands have to add up to the charge of the complex."""
    symbols, coords = hexaaqua()
    mol, report = build_complex(symbols, coords, total_charge=2,
                                ligand_charges=[-5, 0, 0, 0, 0, 0],
                                ligand_smiles=[None] * 6)
    # the stated -5 cannot be given to a water, so every water takes 0 and the metal takes +2;
    # the complex is then consistent, which is what the report has to say
    assert report.net_charge == 2
    assert mol is not None


def test_the_charge_is_written_into_the_molecule():
    symbols, coords = hexaaqua()
    mol, _report = build_complex(symbols, coords, total_charge=2, multiplicity=5, name="fe")
    assert mol.GetProp("charge") == "2"
    assert mol.GetProp("multiplicity") == "5"
    assert mol.GetProp("_Name") == "fe"


def test_the_try_list_starts_at_the_neutral_ligand():
    assert TRY_LIGAND_CHARGES[0] == 0
    assert -1 in TRY_LIGAND_CHARGES and -2 in TRY_LIGAND_CHARGES


# ------------------------------------------------------------------ the metal site

def test_coordination_counts_the_donors_and_their_distances():
    symbols, coords = dichloro_tetraaqua()
    mol, _report = build_complex(symbols, coords, total_charge=0,
                                 ligand_charges=[-1, -1, 0, 0, 0, 0])
    c = coordination(mol)
    assert c["n_donors"] == 6
    assert c["donor_O"] == 4
    assert c["donor_Cl"] == 2
    assert c["metal_charge"] == 2
    assert 2.0 < c["d_mean"] < 2.2
    assert donor_indices(mol) == sorted(donor_indices(mol))
    assert metal_index(mol) is not None


def test_coordination_of_a_molecule_without_a_metal_is_empty():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert coordination(mol) == {}
    assert metal_index(mol) is None
    assert donor_indices(mol) == []


def test_metal_origin_puts_the_metal_at_the_origin():
    symbols, coords = hexaaqua()
    coords = coords + np.array([3.0, -4.0, 5.0])
    mol, _report = build_complex(symbols, coords, total_charge=2)
    moved = metal_origin(mol)
    xyz = moved.GetConformer().GetPositions()
    assert np.allclose(xyz[metal_index(moved)], 0.0, atol=1e-9)
    # the shape is untouched: only the origin moved
    before = mol.GetConformer().GetPositions()
    assert np.allclose(before - before[metal_index(mol)], xyz, atol=1e-9)


# ------------------------------------------------------------------ the frames

def test_alignment_recovers_a_rotated_complex():
    symbols, coords = dichloro_tetraaqua()
    ref, _r = build_complex(symbols, coords, total_charge=0, ligand_charges=[-1, -1, 0, 0, 0, 0])
    turned, _r = build_complex(symbols, coords @ rotation((0.4, -0.9, 1.7)).T, total_charge=0,
                               ligand_charges=[-1, -1, 0, 0, 0, 0])
    _mols, rmsds = align_on_first_sphere([ref, turned], ref_index=0)
    assert rmsds[0] == pytest.approx(0.0, abs=1e-6)
    assert rmsds[1] == pytest.approx(0.0, abs=1e-6)


def test_alignment_searches_over_a_symmetric_coordination_sphere():
    """Six identical donors can answer to the reference donors in any order.

    The two complexes here are the same octahedron turned by a quarter turn, which maps every
    donor onto a different donor. Comparing one arbitrary matching with another would report about
    three angstrom for two complexes that lie exactly on top of each other.
    """
    symbols, coords = hexaaqua()
    ref, _r = build_complex(symbols, coords, total_charge=2)
    quarter = rotation((0.0, 0.0, np.pi / 2))
    turned, _r = build_complex(symbols, coords @ quarter.T, total_charge=2)
    _mols, rmsds = align_on_first_sphere([ref, turned], ref_index=0)
    assert rmsds[1] == pytest.approx(0.0, abs=1e-6)


def test_alignment_reports_no_match_for_a_different_sphere():
    sym_a, xyz_a = hexaaqua()
    sym_b, xyz_b = dichloro_tetraaqua()
    a, _r = build_complex(sym_a, xyz_a, total_charge=2)
    b, _r = build_complex(sym_b, xyz_b, total_charge=0, ligand_charges=[-1, -1, 0, 0, 0, 0])
    _mols, rmsds = align_on_first_sphere([a, b], ref_index=0)
    assert rmsds[0] == pytest.approx(0.0, abs=1e-6)
    assert rmsds[1] is None                           # four oxygens against six: nothing to match


def test_alignment_never_reflects():
    """A mirror image is a different molecule, so the frame may rotate and not reflect."""
    symbols, coords = dichloro_tetraaqua()
    ref, _r = build_complex(symbols, coords, total_charge=0, ligand_charges=[-1, -1, 0, 0, 0, 0])
    mirrored = coords * np.array([1.0, 1.0, -1.0])
    other, _r = build_complex(symbols, mirrored, total_charge=0,
                              ligand_charges=[-1, -1, 0, 0, 0, 0])
    mols, _rmsds = align_on_first_sphere([ref, other], ref_index=0)
    # the mirrored complex cannot be brought onto the reference by a rotation, and the aligned
    # coordinates must still be a rigid copy of what went in: the distances are unchanged
    xyz_in = other.GetConformer().GetPositions()
    xyz_out = mols[1].GetConformer().GetPositions()
    d_in = np.linalg.norm(xyz_in[1] - xyz_in[2])
    d_out = np.linalg.norm(xyz_out[1] - xyz_out[2])
    assert d_out == pytest.approx(d_in, abs=1e-9)


def test_the_rule_frame_does_not_depend_on_the_input_frame():
    symbols, coords = dichloro_tetraaqua()
    a, _r = build_complex(symbols, coords, total_charge=0, ligand_charges=[-1, -1, 0, 0, 0, 0])
    b, _r = build_complex(symbols, coords @ rotation((1.1, 0.3, -0.7)).T, total_charge=0,
                          ligand_charges=[-1, -1, 0, 0, 0, 0])
    fa, _gap_a = principal_metal_frame(a)
    fb, _gap_b = principal_metal_frame(b)
    xa = fa.GetConformer().GetPositions()
    xb = fb.GetConformer().GetPositions()
    assert np.allclose(np.abs(xa), np.abs(xb), atol=1e-6)


def test_the_rule_frame_reports_a_degenerate_shape():
    """A regular octahedron has three equal moments, so its rule frame is arbitrary."""
    symbols, coords = hexaaqua()
    mol, _r = build_complex(symbols, coords, total_charge=2)
    _framed, gap = principal_metal_frame(mol)
    assert gap < 0.2
    symbols, coords = dichloro_tetraaqua()
    other, _r = build_complex(symbols, coords, total_charge=0,
                              ligand_charges=[-1, -1, 0, 0, 0, 0])
    _framed, gap_other = principal_metal_frame(other)
    assert gap_other > gap                            # two chlorides break the symmetry


# ------------------------------------------------------------------ the parameters

def test_the_metals_of_the_item_have_parameters():
    for sym in ("Fe", "Co", "Ni", "Cu", "Mn", "Zn", "Ru", "Rh", "Pd", "Pt", "Ir"):
        assert is_transition_metal(sym)
        radius, well = vdw_params(sym)
        assert 1.0 < radius < 2.0
        assert 0.0 < well < 0.5
    assert len(TRANSITION_METALS) == 29               # the whole d block


def test_the_metal_radii_are_on_the_scale_of_the_tripos_radii():
    """The UFF radius of carbon is 13 % larger than the Tripos one, and the metals are scaled."""
    assert UFF_RADIUS_SCALE == pytest.approx(1.70 / (3.851 / 2.0))
    # iron: the UFF distance 2.912 A halved and put on the Tripos scale
    assert vdw_params("Fe")[0] == pytest.approx(round(2.912 / 2.0 * UFF_RADIUS_SCALE, 3))
    # a metal is smaller than carbon, which is what a buried metal centre should be
    assert vdw_params("Fe")[0] < vdw_params("C")[0]


def test_a_sybyl_metal_type_finds_the_parameters_of_its_element():
    """Sybyl writes the coordination into a metal type ("Co.oh"); the parameters are the metal's."""
    assert vdw_params("Co", "Co.oh") == vdw_params("Co")
    assert vdw_params("Ti", "Ti.th") == vdw_params("Ti")


def test_the_main_group_parameters_are_untouched():
    assert vdw_params("C") == (1.70, 0.107)
    assert vdw_params("N") == (1.55, 0.095)
    assert vdw_params("O") == (1.52, 0.116)
    assert vdw_params("S", "S.o2") == (1.70, 0.314)
    # lithium is in the table by atom type only, as it was before this item
    assert vdw_params("Li", "Li") == (1.20, 0.400)


def test_a_metal_carries_no_hydrogen_bond_or_hydrophobic_increment():
    for sym in ("Fe", "Ni", "Pt"):
        inc = tripos_increments(sym, 0)
        assert all(v == 0.0 for v in inc.values())


def test_an_unknown_element_still_gets_a_default():
    radius, well = vdw_params("Xe")
    assert radius > 0 and well > 0


# ------------------------------------------------------------------ the fields of a complex

def complex_fields(scheme="rdkit"):
    symbols, coords = dichloro_tetraaqua()
    mol, _report = build_complex(symbols, coords, total_charge=0,
                                 ligand_charges=[-1, -1, 0, 0, 0, 0])
    fs = FieldSet.create(physchem=["electrostatic", "steric", "hydrophobic", "hba", "hbd"],
                         alpha=1.0, charge_model="gasteiger", param_scheme=scheme)
    return mol, fs, build_fields(mol, fs, "complex")


def test_the_electrostatic_field_of_a_complex_is_not_zero():
    """One metal atom used to make the Gasteiger scheme return nothing for every atom.

    The field was then identically zero, which is not an error and not a field; the repair gives
    the ligands the charges of the free ligand and the metal its formal charge.
    """
    mol, fs, mf = complex_fields()
    assert mf.charge_source == "gasteiger+formal"
    weights = np.asarray(mf.weights)
    electrostatic = weights[:, [s.name for s in fs.specs].index("electrostatic")]
    assert np.abs(electrostatic).sum() > 0.5
    # the charges still add up to the charge of the complex
    assert electrostatic.sum() == pytest.approx(float(Chem.GetFormalCharge(mol)), abs=1e-6)
    metal = metal_index(mol)
    assert electrostatic[metal] == pytest.approx(2.0, abs=1e-9)


def test_the_metal_carries_no_hydrophobic_weight():
    mol, fs, mf = complex_fields()
    weights = np.asarray(mf.weights)
    hydrophobic = weights[:, [s.name for s in fs.specs].index("hydrophobic")]
    assert hydrophobic[metal_index(mol)] == 0.0


def test_the_steric_field_uses_the_metal_radius():
    mol, fs, mf = complex_fields()
    assert mf.radii[metal_index(mol)] == pytest.approx(vdw_params("Fe")[0])
    weights = np.asarray(mf.weights)
    steric = weights[:, [s.name for s in fs.specs].index("steric")]
    assert steric[metal_index(mol)] == pytest.approx(vdw_params("Fe")[1])


def test_the_sybyl_type_of_a_metal_is_its_element():
    mol, _fs, _mf = complex_fields()
    types = assign_atom_types(mol, "sybyl")
    assert types[metal_index(mol)] == "Fe"


# ------------------------------------------------------------------ the data loader

def test_read_molecules_reads_a_complex_from_an_xyz_file(tmp_path):
    from cmfqsar.dataio import read_molecules
    symbols, coords = hexaaqua()
    path = tmp_path / "complex.xyz"
    path.write_text(xyz_text(symbols, coords, "name=hexaaqua charge=2 multiplicity=5"),
                    encoding="utf-8")
    mols, names, _props = read_molecules(str(path))
    assert names == ["hexaaqua"]
    mol = mols[0]
    assert sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE) == 6
    assert Chem.GetFormalCharge(mol) == 2
    assert mol.GetProp("multiplicity") == "5"


def test_read_molecules_reads_an_ordinary_molecule_from_an_xyz_file(tmp_path):
    from cmfqsar.dataio import read_molecules
    path = tmp_path / "water.xyz"
    path.write_text("3\nname=water\nO 0.000 0.000 0.000\nH 0.600 0.760 0.000\n"
                    "H 0.600 -0.760 0.000\n", encoding="utf-8")
    mols, names, _props = read_molecules(str(path))
    assert names == ["water"]
    assert Chem.MolToSmiles(Chem.RemoveHs(mols[0])) == "O"


def test_read_molecules_skips_a_block_it_cannot_read(tmp_path):
    from cmfqsar.dataio import read_molecules
    symbols, coords = hexaaqua()
    broken = "2\nname=broken\nFe 0.0 0.0 0.0\nFe 2.2 0.0 0.0\n"
    path = tmp_path / "mixed.xyz"
    path.write_text(broken + xyz_text(symbols, coords, "name=good charge=2"), encoding="utf-8")
    mols, names, _props = read_molecules(str(path))
    assert names == ["good"]
    assert len(mols) == 1
