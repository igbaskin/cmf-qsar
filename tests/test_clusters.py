"""A cluster cut from a periodic sheet around its metal, capped, read as a complex (EC8).

Checked on the single-atom catalyst sheets of `examples/sac`:

* the cut keeps the atoms within the asked number of bonds, leaves no atom hanging by one bond,
  and caps every cut bond with one hydrogen at the C-H length along the bond;
* every atom of the cluster other than the metal ends with as many neighbours as it has in the
  sheet, the metal not counted (three for a carbon, two for a pyridine-like nitrogen);
* the cluster does not depend on where the site sits in the cell: a sheet moved so that the
  metal is at the corner gives the same cluster, unwrapped around the metal;
* the complex sanitises, it is bonded to its four donors by dative bonds, and the metal's
  formal charge is its number of carbon donors (each a carbanion of the closed-shell ligand),
  while the cluster as a whole is neutral;
* the origin and the turn over z put a series in one frame.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("ase")
pytest.importorskip("rdkit")

from ase.io import read                                                 # noqa: E402

from cmfqsar.clusters import CAP_LENGTH, cluster_complex, cut_cluster   # noqa: E402

SHEETS = os.path.join(ROOT, "examples", "sac", "substrates.extxyz")


@pytest.fixture(scope="module")
def sheets():
    return read(SHEETS, index=":")


def _metal(atoms):
    return next(i for i, s in enumerate(atoms.get_chemical_symbols()) if s not in ("C", "N"))


def _pick(sheets, motif_n):
    for a in sheets:
        n = sum(s == "N" for s in a.get_chemical_symbols())
        if n == motif_n:
            return a
    pytest.skip(f"no sheet with {motif_n} nitrogen atoms")


@pytest.mark.parametrize("hops, n_atoms", [(3, 25), (4, 45), (5, 57)])
def test_the_cut_and_the_caps(sheets, hops, n_atoms):
    a = _pick(sheets, 4)
    site = _metal(a)
    cl = cut_cluster(a, site, hops)
    assert cl.n_atoms == n_atoms and cl.kept[0] == site
    assert cl.symbols[0] == a[site].symbol
    from ase.neighborlist import neighbor_list
    i_s, j_s = neighbor_list("ij", a, 1.75)
    in_sheet = {k: int(np.sum((i_s == k) & (j_s != site))) for k in cl.kept[1:]}
    heavy = np.array([s != "H" for s in cl.symbols])
    X = cl.coords
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    for i in np.flatnonzero(heavy)[1:]:
        # the neighbours an atom has in the sheet: kept atoms or caps, the metal not counted
        near = [j for j in range(1, cl.n_atoms) if j != i and D[i, j] < 1.75]
        assert len(near) == in_sheet[cl.kept[i]], (cl.symbols[i], len(near))
    for i in np.flatnonzero(~heavy):
        j = int(np.argmin(np.where(np.arange(cl.n_atoms) == i, np.inf, D[i])))
        assert D[i, j] == pytest.approx(CAP_LENGTH[cl.symbols[j]], abs=1e-9)
    assert cl.n_caps == int((~heavy).sum())


def test_the_cluster_does_not_depend_on_where_the_site_is(sheets):
    a = _pick(sheets, 2)
    site = _metal(a)
    b = a.copy()
    b.positions += -b.positions[site] + 0.3       # the metal near the corner of the cell
    b.wrap()
    c1, c2 = cut_cluster(a, site, 4), cut_cluster(b, site, 4)
    assert c1.symbols == c2.symbols and c1.kept == c2.kept
    assert np.allclose(c1.coords - c1.coords[0], c2.coords - c2.coords[0], atol=1e-8)


def test_the_complex(sheets):
    from rdkit import Chem
    for n in (0, 4):
        a = _pick(sheets, n)
        site = _metal(a)
        centre = a.positions[site].copy()
        mol, cl, rep = cluster_complex(a, site, hops=4, origin=centre)
        assert mol is not None, rep.fault
        metal = mol.GetAtomWithIdx(0)
        dative = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE]
        assert len(dative) == 4
        carbons = sum(b.GetBeginAtom().GetSymbol() == "C" for b in dative)
        assert metal.GetSymbol() == a[site].symbol and metal.GetFormalCharge() == carbons
        assert rep.metal_charge == carbons and rep.ligand_charges == [-carbons]
        assert Chem.GetFormalCharge(mol) == 0
        assert np.allclose(mol.GetConformer().GetAtomPosition(0), (0.0, 0.0, 0.0))
    # turned over: the metal's height changes sign, the rest of the frame stays
    mol2, cl2, _ = cluster_complex(a, site, hops=4, origin=centre - [0, 0, 1.0], flip_z=True)
    assert cl2.coords[0][2] == pytest.approx(-1.0)
    assert np.allclose(cl2.coords[:, :2], cl.coords[:, :2])


def test_a_bad_request_is_refused(sheets):
    with pytest.raises(ValueError, match="hops"):
        cut_cluster(sheets[0], _metal(sheets[0]), 0)
