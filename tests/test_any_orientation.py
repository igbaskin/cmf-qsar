"""The option that lets a restart draw any orientation, from the dialog down to the search.

The pose protocol of plan item C1 destroys an orientation completely, and a restart that turns the
arriving pose by at most about 63 degrees cannot recover one. The searches therefore take
`any_orientation`. What is fixed here is what that option promises: the bounds it keeps when it is
off, the orientations it reaches when it is on, that the three searches the protocol judges all
have it, and that the application and the protocol ask for it where they should.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.docking import restart_poses  # noqa: E402


def _turns(poses):
    """How far each restart turns the pose it starts from, in degrees."""
    return np.degrees([np.linalg.norm(p[:3]) for p in poses])


def test_the_arriving_pose_is_always_tried_first():
    """A restarting search must never be worse than the pose it was given."""
    poses = restart_poses(5, np.random.default_rng(0))
    assert len(poses) == 6
    assert np.allclose(poses[0], np.zeros(6))


def test_a_bounded_restart_stays_bounded():
    poses = restart_poses(200, np.random.default_rng(0), max_rotation_deg=30.0)
    assert _turns(poses).max() <= 30.0 + 1e-9


def test_the_unbounded_restart_reaches_every_orientation():
    """A uniformly random orientation turns by up to 180 degrees, and by 127 on average.

    That mean is the one the uniform distribution over orientations has, whose angle density is
    (1 - cos t) / pi; a bounded restart of 30 degrees averages 15.
    """
    turns = _turns(restart_poses(400, np.random.default_rng(0), max_rotation_deg=180.0)[1:])
    assert turns.max() > 170.0
    assert 115.0 < float(np.mean(turns)) < 140.0
    assert turns.min() < 40.0          # near orientations are still drawn, just not only them


def test_every_search_that_the_protocol_judges_takes_the_option():
    """Three searches had the same defect; the field refinement never had it."""
    import inspect

    from cmfqsar.flexalign import flexible_align
    from cmfqsar.pocketalign import pocket_align
    from cmfqsar.varalign import variance_align

    for fn in (variance_align, pocket_align, flexible_align):
        sig = inspect.signature(fn)
        assert "any_orientation" in sig.parameters, fn.__name__
        assert sig.parameters["any_orientation"].default is False, fn.__name__


def test_the_protocol_asks_for_any_orientation_by_default():
    """It destroys the orientation itself, so its default must be able to find one."""
    import inspect

    from cmfqsar.poses import judge

    p = inspect.signature(judge).parameters
    assert p["max_rotation_deg"].default >= 180.0
    assert p["restarts"].default == 20


def test_the_dialog_offers_it_and_sets_what_it_needs(qt_application):
    """Ticking the box in the application removes the guard and raises the restarts."""
    pytest.importorskip("pyqtgraph")

    from cmfqsar.gui.main_window import VarianceAlignDialog

    dlg = VarianceAlignDialog(5, 5)
    assert dlg.any_orientation() is False
    assert dlg.max_rmsd() == 2.0
    dlg.chk_any.setChecked(True)
    assert dlg.any_orientation() is True
    assert dlg.max_rmsd() is None, "a reorientation is a large displacement, so the guard must go"
    assert dlg.spin_restarts.value() >= 20


def _result(set_name="p38", weight=1.0, methods=("pocket", "pocket+variance")):
    """A result record of the shape the report reads, with the fields it groups by."""
    cell = {"median_rmsd": 0.5, "within_1A": 1.0, "within_2A": 1.0, "recovery": 0.9,
            "r2_pred": 0.2, "conformer_limit": 0.0, "seconds": 1.0}
    return {"set": set_name, "n": 29, "n_test": 9, "alpha": 0.5, "lam": 0.1, "seed": 0,
            "reconform": False, "conformers": 0, "max_rmsd": None, "restarts": 20,
            "max_rotation_deg": 180.0, "pocket_weight": weight, "pocket": "pocket.pdb",
            "methods": {m: dict(cell) for m in ("crystal",) + tuple(methods)},
            "seconds": 1.0}


def _written(tmp_path, rows):
    """Write these result records to a folder of their own and report on it."""
    import json

    from benchmarks.run_poses import report

    for i, row in enumerate(rows):
        with open(os.path.join(tmp_path, f"{row['set']}_{i}.json"), "w") as fh:
            json.dump(row, fh)
    return report(results=str(tmp_path), out=os.path.join(tmp_path, "table.md"))


def test_two_weights_of_the_receptor_term_are_two_tables(tmp_path, capsys):
    """A run at another weight finds other poses, so it is another result and not another row.

    Grouped together, the weight sweep gave one table with p38 in it three times and no column
    saying which row was which weight.
    """
    text = _written(tmp_path, [_result(weight=w) for w in (0.0, 0.25, 4.0)])
    capsys.readouterr()
    headings = [ln for ln in text.splitlines() if ln.startswith("## ")]
    assert len(headings) == 3, headings
    assert sum("weighted 0.25" in h for h in headings) == 1, headings
    # each table then has the set once, which is what makes its rows readable
    assert text.count("| p38 | 29 |") == 6   # three tables, the RMSD table and the R2pred table


def test_the_pocket_methods_reach_the_table(tmp_path, capsys):
    """They were measured and stored, and the report left them out of its columns."""
    from benchmarks.run_poses import POCKET_METHODS

    text = _written(tmp_path, [_result(methods=POCKET_METHODS)])
    capsys.readouterr()
    for method in POCKET_METHODS:
        assert f"| {method} |" in text or f" {method} |" in text, method
