"""The slab generator of the adsorption benchmarks (plan item BT8), without the UMA model.

What the script adds to the Fe(110) script is the slab, so the slab is what is checked here:

* layers are found from the heights and numbered from the top, and the slab is rigid by default;
* the Fe(110) recipe is the Fe(110) script's own slab, atom for atom;
* the Li(100) and graphite recipes have full layers and no two atoms closer than a bond;
* a slab read from a file is repeated by its own cell;
* the cell grows for a large molecule in steps of the recipe (two rows at a time for bcc(110));
* the lattice at the minimum of the bulk is a minimum of the energy (checked with ASE's EMT
  potential on copper, which needs no GPU and no licence);
* the options the recipe sets are left unset by the parser.
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "examples"))

pytest.importorskip("ase")
pytest.importorskip("rdkit")

import run_slab_adsorption as S                                         # noqa: E402
import uma_fe110_adsorption as fe                                       # noqa: E402
from ase.neighborlist import neighbor_list                              # noqa: E402


def test_layers_are_numbered_from_the_top():
    tags = S.layer_tags(np.array([0.0, 0.1, 2.0, 2.3, 4.0, -1.5]))
    assert tags.tolist() == [3, 3, 2, 2, 1, 4]


def test_the_fe110_recipe_is_the_fe110_script_slab():
    ref = fe.build_fe110_slab(6, 6, 4, 2.78, 15.0, None)
    new = S.SURFACES["fe110"].build(6, 6, 4, 2.78, 15.0, None)
    assert np.allclose(new.positions, ref.positions, atol=1e-9)
    assert np.allclose(new.cell, ref.cell)
    assert (new.get_tags() == ref.get_tags()).all()
    assert len(new.constraints[0].index) == len(new)          # rigid
    assert S.top_layer_z(new) == pytest.approx(fe.top_layer_z(ref))


@pytest.mark.parametrize("name, a, per_layer, bond", [("li100", 3.44, 25, 2.9),
                                                      ("graphite0001", 2.46, 112, 1.42)])
def test_the_battery_recipes(name, a, per_layer, bond):
    sf = S.SURFACES[name]
    slab = sf.build(*sf.cell, sf.layers, a, 15.0, None)
    tags = slab.get_tags()
    assert tags.max() == sf.layers
    assert np.bincount(tags)[1:].tolist() == [per_layer] * sf.layers
    _, _, d = neighbor_list("ijd", slab, 3.5)
    assert d.min() == pytest.approx(bond, abs=0.1)
    ux, uy = sf.unit(a)
    assert slab.cell.lengths()[0] == pytest.approx(sf.cell[0] * ux)
    assert slab.cell.lengths()[1] == pytest.approx(sf.cell[1] * uy)


def test_graphite_has_three_neighbours_in_its_plane():
    sf = S.SURFACES["graphite0001"]
    slab = sf.build(*sf.cell, 2, S.GRAPHITE_A, 15.0, None)
    i, _, d = neighbor_list("ijd", slab, 1.6)
    assert (np.bincount(i, minlength=len(slab)) == 3).all()
    z = slab.positions[:, 2]
    assert z.max() - z.min() == pytest.approx(S.GRAPHITE_SPACING)


def test_a_slab_from_a_file_is_repeated_by_its_cell(tmp_path):
    from ase.build import fcc111
    from ase.io import write
    unit = fcc111("Cu", size=(1, 2, 3), a=3.6, vacuum=6.0, orthogonal=True)
    path = tmp_path / "cu111.extxyz"
    write(str(path), unit)
    sf = S.surface_from_file(str(path))
    assert sf.layers == 3
    slab = sf.build(4, 3, sf.layers, 1.0, 15.0, 1)
    assert len(slab) == 12 * len(unit)
    assert np.allclose(slab.cell.lengths()[:2], unit.cell.lengths()[:2] * [4, 3])
    # one fixed layer from the bottom
    assert len(slab.constraints[0].index) == np.sum(slab.get_tags() == 3)


def test_the_cell_grows_in_steps_of_the_recipe():
    p = S.build_parser()
    args = p.parse_args(["x.csv", "-o", "x.sdf", "--surface", "fe110"])
    assert args.lattice is None and args.cell is None and args.layers is None
    sf = S.SURFACES["fe110"]
    args.cell, args.layers, args.lattice = sf.cell, sf.layers, 2.78
    args = fe.prepare_args(args)
    saved = (fe.top_layer_z, fe.cell_for_footprint, fe.clean_slab_energy)
    try:
        S.install(sf, args)
        assert fe.cell_for_footprint((5.0, 3.0), 2.78, 6, 6, 6.0) == (6, 6)
        nx, ny = fe.cell_for_footprint((20.0, 9.0), 2.78, 6, 6, 6.0)
        assert nx * 2.78 >= 26.0 and (nx - 1) * 2.78 < 26.0
        assert ny % 2 == 0 and ny * 2.78 / math.sqrt(2) >= 15.0
        # the Fe(110) script's own rule gives the same cell
        assert (nx, ny) == saved[1]((20.0, 9.0), 2.78, 6, 6, 6.0)
    finally:
        fe.top_layer_z, fe.cell_for_footprint, fe.clean_slab_energy = saved


def test_the_lattice_at_the_minimum_of_the_bulk(tmp_path):
    from ase.build import bulk
    from ase.calculators.emt import EMT
    sf = S.Surface("cu", None, lambda a: (a, a), (1, 1), 1, None,
                   bulk=lambda a: bulk("Cu", "fcc", a=a, cubic=True))
    a0 = S.bulk_minimum(sf, EMT(), str(tmp_path), log=lambda *x: None)

    def energy(a):
        b = sf.bulk(a)
        b.calc = EMT()
        return b.get_potential_energy()
    assert energy(a0) < energy(1.02 * a0) and energy(a0) < energy(0.98 * a0)
    # cached on the second call
    assert S.bulk_minimum(sf, None, str(tmp_path), log=lambda *x: None) == pytest.approx(a0)
