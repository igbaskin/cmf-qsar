"""Screening a library in blocks (plan item B3).

The point of the module is that the answer does not depend on how the library was cut into
blocks, so most of these tests compare a streamed run with the ordinary all-at-once prediction.
"""
import csv
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.screen import ScreenSpec, screen, stream_molecules  # noqa: E402

TRAIN = ["c1ccccc1C(=O)O", "c1ccccc1CC(=O)O", "c1ccc(Cl)cc1C(=O)O", "c1ccc(C)cc1C(=O)O",
         "c1ccc(O)cc1C(=O)O", "c1ccc(N)cc1C(=O)O", "c1ccc(F)cc1C(=O)O", "c1ccc(Br)cc1C(=O)O"]
LIBRARY = ["c1ccccc1CCC(=O)O", "c1ccc(I)cc1C(=O)O", "c1ccc(CC)cc1C(=O)O", "OCc1ccccc1",
           "c1ccncc1C(=O)O", "c1ccc(C#N)cc1C(=O)O", "CCCCCC(=O)O", "c1ccsc1C(=O)O",
           "c1ccc(OC)cc1C(=O)O", "c1ccc(C(F)(F)F)cc1C(=O)O"]


def _mol(smi, seed):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    return m


def _model(method="gp"):
    fs = FieldSet.create(physchem=["electrostatic", "steric", "hydrophobic"], alpha=0.5)
    mols = [_mol(s, k + 1) for k, s in enumerate(TRAIN)]
    table = AtomTable.from_molecules([build_fields(m, fs, f"t{i}")
                                      for i, m in enumerate(mols)])
    y = np.linspace(4.0, 8.0, len(mols))
    m = CMFModel(fieldset=fs, method=method, normalize=True, params={"lam": 0.1})
    m.fit(table, y, cv_folds=0)
    return m, fs


def _library(tmp_path, smiles=LIBRARY, name="lib.sdf", broken=False):
    path = os.path.join(str(tmp_path), name)
    with Chem.SDWriter(path) as w:
        for k, s in enumerate(smiles):
            m = _mol(s, 100 + k)
            m.SetProp("_Name", f"lib{k}")
            w.write(m)
    if broken:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("not a molecule\n  rubbish\n\n  0  0\nM  END\n$$$$\n")
    return path


# ------------------------------------------------------------------------------- the stream


def test_the_stream_gives_every_molecule_once_whatever_the_block(tmp_path):
    path = _library(tmp_path)
    for chunk in (1, 3, 10, 100):
        blocks = list(stream_molecules(path, chunk))
        names = [n for b in blocks for _m, n in b]
        assert len(names) == len(LIBRARY)
        assert names == [f"lib{k}" for k in range(len(LIBRARY))]
        assert all(len(b) <= chunk for b in blocks)


def test_an_entry_the_reader_cannot_use_still_comes_through(tmp_path):
    """A library file holds rubbish too, and the stream must not stop at it."""
    path = _library(tmp_path, broken=True)
    got = [m for b in stream_molecules(path, 50) for m, _n in b]
    assert len(got) == len(LIBRARY) + 1
    # RDKit either refuses the entry or parses it into a molecule with no atoms
    assert got[-1] is None or got[-1].GetNumAtoms() == 0


def test_a_smiles_file_is_read_and_embedded(tmp_path):
    path = os.path.join(str(tmp_path), "lib.smi")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("smiles name\n")
        for k, s in enumerate(LIBRARY[:4]):
            fh.write(f"{s} s{k}\n")
        fh.write("not-a-smiles bad\n")
    blocks = list(stream_molecules(path, 2))
    rows = [(m, n) for b in blocks for m, n in b]
    assert [n for _m, n in rows] == ["s0", "s1", "s2", "s3", "bad"]
    assert rows[-1][0] is None
    assert all(m.GetNumConformers() == 1 for m, _n in rows[:4])


# ------------------------------------------------------------------------------ the scoring


def test_the_streamed_scores_are_the_all_at_once_scores(tmp_path):
    model, fs = _model()
    path = _library(tmp_path)
    blocks = list(stream_molecules(path, 1000))
    table = AtomTable.from_molecules([build_fields(m, fs, n) for b in blocks for m, n in b])
    direct = np.asarray(model.predict(table), float)
    for chunk in (1, 3, 1000):
        rep = screen(model, path, ScreenSpec(chunk=chunk, top=len(LIBRARY)))
        by_name = {n: s for s, n, _p, _sd in rep.best}
        got = np.array([by_name[f"lib{k}"] for k in range(len(LIBRARY))])
        assert got == pytest.approx(direct, abs=1e-9)
        assert rep.n_scored == len(LIBRARY)
        assert rep.blocks == int(np.ceil(len(LIBRARY) / chunk))


def test_the_csv_holds_every_molecule_and_is_readable(tmp_path):
    model, _fs = _model()
    path = _library(tmp_path)
    out = os.path.join(str(tmp_path), "scores.csv")
    rep = screen(model, path, ScreenSpec(chunk=4, top=3), out=out)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == len(LIBRARY)
    assert [r["name"] for r in rows] == [f"lib{k}" for k in range(len(LIBRARY))]
    assert all(r["sd"] for r in rows)                   # a GP model fills the spread
    best = max(rows, key=lambda r: float(r["score"]))
    assert best["name"] == rep.best[0][1]


def test_the_top_list_is_the_best_of_them_in_order(tmp_path):
    model, _fs = _model()
    path = _library(tmp_path)
    full = screen(model, path, ScreenSpec(chunk=2, top=len(LIBRARY))).best
    few = screen(model, path, ScreenSpec(chunk=3, top=3)).best
    assert [r[1] for r in few] == [r[1] for r in full[:3]]
    assert [r[0] for r in few] == sorted([r[0] for r in few], reverse=True)


def test_the_confidence_bound_demotes_what_the_model_does_not_know(tmp_path):
    model, _fs = _model()
    path = _library(tmp_path)
    plain = screen(model, path, ScreenSpec(chunk=5, top=len(LIBRARY)))
    bound = screen(model, path, ScreenSpec(chunk=5, top=len(LIBRARY), beta=1.0))
    by_plain = {n: (s, sd) for s, n, _p, sd in plain.best}
    for score, name, pred, sd in bound.best:
        assert score == pytest.approx(pred - sd, abs=1e-9)
        assert score <= by_plain[name][0] + 1e-12
    assert [r[1] for r in bound.best] != [r[1] for r in plain.best] or len(LIBRARY) < 3


def test_a_molecule_that_cannot_be_typed_is_counted_not_fatal(tmp_path):
    model, _fs = _model()
    path = _library(tmp_path, broken=True)
    rep = screen(model, path, ScreenSpec(chunk=3, top=2))
    assert rep.n_scored == len(LIBRARY)
    assert rep.n_failed == 1
    assert rep.failures and "describe" in dir(rep)
    assert "could not be read" in rep.describe()


def test_the_report_counts_what_it_did(tmp_path):
    model, _fs = _model()
    path = _library(tmp_path)
    rep = screen(model, path, ScreenSpec(chunk=4, top=2))
    assert rep.n_read == len(LIBRARY)
    assert rep.blocks == 3
    assert rep.peak_kernel_mb == pytest.approx(8.0 * 4 * model.train.n_mol / 1e6, rel=0.5)
    assert rep.per_second > 0
    assert "molecules scored" in rep.describe()


def test_the_run_leaves_the_model_as_it_found_it(tmp_path):
    model, _fs = _model()
    path = _library(tmp_path)
    screen(model, path, ScreenSpec(chunk=3, cutoff="auto"))
    assert model.cutoff is None
    model.cutoff = 12.0
    screen(model, path, ScreenSpec(chunk=3, cutoff="auto"))
    assert model.cutoff == 12.0


def test_cancelling_stops_the_run_with_what_it_had(tmp_path):
    model, _fs = _model()
    path = _library(tmp_path)
    seen = {"n": 0}

    def cancel():
        seen["n"] += 1
        return seen["n"] > 2

    rep = screen(model, path, ScreenSpec(chunk=2, top=5), cancel=cancel)
    assert 0 < rep.n_scored < len(LIBRARY)


def test_a_one_class_model_is_scored_by_its_decision_value(tmp_path):
    fs = FieldSet.create(physchem=["steric", "hydrophobic"], alpha=0.5)
    mols = [_mol(s, k + 1) for k, s in enumerate(TRAIN)]
    table = AtomTable.from_molecules([build_fields(m, fs, f"t{i}") for i, m in enumerate(mols)])
    m = CMFModel(fieldset=fs, task="oneclass", method="ocmean", normalize=True,
                 params={"nu": 0.2})
    m.fit(table, np.ones(len(mols)), cv_folds=0)
    path = _library(tmp_path)
    rep = screen(m, path, ScreenSpec(chunk=4, top=3, beta=1.0))
    assert rep.n_scored == len(LIBRARY)
    # there is no posterior here, so the spread column is empty and beta is ignored
    assert all(sd != sd for _s, _n, _p, sd in rep.best)
    assert all(s == p for s, _n, p, _sd in rep.best)
