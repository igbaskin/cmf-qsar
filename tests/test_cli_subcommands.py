"""The subcommands of plan item G1: `align`, `cofield` and `benchmark`.

Alignment used to be a handful of flags on `build` and `predict`; co-fields could only be exported
from the application; the benchmarks had their own entry point and no way in from the tool. These
tests drive the command line itself, with the smallest sets that still exercise each branch,
because an option that only works when it is typed by hand is not a command-line interface.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.cli import main  # noqa: E402

SMILES = ["c1ccccc1C(=O)O", "c1ccc(Cl)cc1C(=O)O", "c1ccc(C)cc1C(=O)O", "c1ccc(O)cc1C(=O)O",
          "c1ccc(N)cc1C(=O)O", "c1ccc(F)cc1C(=O)O"]


def _set(path, activities=True):
    with Chem.SDWriter(path) as w:
        for k, smi in enumerate(SMILES):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            p = AllChem.ETKDGv3()
            p.randomSeed = k + 1
            AllChem.EmbedMolecule(m, p)
            m.SetProp("_Name", f"m{k}")
            if activities:
                m.SetProp("activity", f"{4.0 + 0.6 * k:.2f}")
            w.write(m)
    return path


def _model(tmp, name="m.cmf"):
    data = _set(os.path.join(str(tmp), "train.sdf"))
    out = os.path.join(str(tmp), name)
    main(["build", data, "--activity", "activity", "--fields", "electrostatic,steric",
          "--alphas", "0.5", "--lams", "0.1", "--save", out])
    return data, out


# ----------------------------------------------------------------------------------- align


def test_align_by_substructure_writes_what_it_says(tmp_path, capsys):
    data = _set(os.path.join(str(tmp_path), "in.sdf"))
    out = os.path.join(str(tmp_path), "out.sdf")
    main(["align", data, "--by", "mcs", "--out", out, "--activity", "activity"])
    text = capsys.readouterr().out
    assert "maximum common substructure" in text
    assert "Moved: median" in text
    assert os.path.exists(out)
    got = [m for m in Chem.SDMolSupplier(out, removeHs=False) if m is not None]
    assert len(got) == len(SMILES)
    assert [m.GetProp("_Name") for m in got] == [f"m{k}" for k in range(len(SMILES))]


def test_align_keeps_nothing_without_an_output_file(tmp_path, capsys):
    data = _set(os.path.join(str(tmp_path), "in.sdf"))
    main(["align", data, "--by", "mcs", "--activity", "activity"])
    assert "Nothing written" in capsys.readouterr().out


def test_align_by_field_moves_the_molecules_into_one_frame(tmp_path, capsys):
    data = _set(os.path.join(str(tmp_path), "in.sdf"))
    out = os.path.join(str(tmp_path), "out.sdf")
    main(["align", data, "--by", "field", "--reference", "0", "--restarts", "0",
          "--out", out, "--activity", "activity"])
    text = capsys.readouterr().out
    assert "maximum field similarity" in text
    assert "similarity" in text
    before = [m for m in Chem.SDMolSupplier(data, removeHs=False)]
    after = [m for m in Chem.SDMolSupplier(out, removeHs=False)]
    moved = [float(np.sqrt(np.mean(np.sum(
        (b.GetConformer().GetPositions() - a.GetConformer().GetPositions()) ** 2, axis=1))))
        for b, a in zip(before, after)]
    assert max(moved) > 0.1               # something was actually moved


def test_align_by_confidence_lowers_the_variance(tmp_path, capsys):
    data, model = _model(tmp_path)
    scrambled = os.path.join(str(tmp_path), "scrambled.sdf")
    main(["unalign", data, scrambled, "--max-rotation", "25", "--max-translation", "1.0"])
    capsys.readouterr()
    main(["align", scrambled, "--by", "variance", "--model", model, "--max-rmsd", "2",
          "--out", os.path.join(str(tmp_path), "re.sdf")])
    text = capsys.readouterr().out
    assert "prediction confidence" in text
    rows = [ln for ln in text.splitlines() if "variance" in ln and "->" in ln]
    assert rows, text
    for ln in rows:
        before, after = [float(v) for v in ln.split("variance")[1].split("moved")[0].split("->")]
        assert after <= before + 1e-9       # the criterion never gets worse


def test_align_from_any_orientation_reaches_the_search(tmp_path, capsys, monkeypatch):
    """The option is useless if it stops at the command line, so follow it to the aligner."""
    from cmfqsar import varalign

    data, model = _model(tmp_path)
    seen = {}
    monkeypatch.setattr(varalign, "variance_align",
                        lambda mols, mdl, **kw: seen.update(kw) or [])
    main(["align", data, "--by", "variance", "--model", model, "--max-rmsd", "0",
          "--any-orientation", "--restarts", "3"])
    assert seen["any_orientation"] is True
    assert seen["restarts"] == 3
    seen.clear()
    main(["align", data, "--by", "variance", "--model", model, "--max-rmsd", "0"])
    assert seen["any_orientation"] is False


def test_align_says_the_guard_would_undo_any_orientation(tmp_path, capsys, monkeypatch):
    """A reorientation is a large displacement, so the two options work against each other."""
    from cmfqsar import varalign

    data, model = _model(tmp_path)
    monkeypatch.setattr(varalign, "variance_align", lambda mols, mdl, **kw: [])
    main(["align", data, "--by", "variance", "--model", model, "--max-rmsd", "2",
          "--any-orientation"])
    assert "--max-rmsd 0" in capsys.readouterr().out


def test_predict_passes_any_orientation_to_the_aligner(tmp_path, capsys, monkeypatch):
    from cmfqsar import varalign

    data, model = _model(tmp_path)
    seen = {}
    monkeypatch.setattr(varalign, "variance_align",
                        lambda mols, mdl, **kw: seen.update(kw) or [])
    main(["predict", model, data, "--align-variance", "0", "--align-any-orientation",
          "--align-restarts", "4"])
    assert seen["any_orientation"] is True
    assert seen["restarts"] == 4


def test_align_by_confidence_needs_a_model(tmp_path):
    data = _set(os.path.join(str(tmp_path), "in.sdf"))
    with pytest.raises(SystemExit, match="needs --model"):
        main(["align", data, "--by", "variance"])


def test_align_in_the_pocket_needs_a_structure(tmp_path):
    data, model = _model(tmp_path)
    with pytest.raises(SystemExit, match="needs --pocket"):
        main(["align", data, "--by", "pocket", "--model", model])


def test_align_refuses_a_reference_it_cannot_find(tmp_path):
    data = _set(os.path.join(str(tmp_path), "in.sdf"))
    with pytest.raises(SystemExit, match="no molecule called"):
        main(["align", data, "--by", "mcs", "--reference", "not-a-molecule"])


def test_align_takes_a_reference_by_name_or_index(tmp_path, capsys):
    data = _set(os.path.join(str(tmp_path), "in.sdf"))
    main(["align", data, "--by", "mcs", "--reference", "m2"])
    by_name = capsys.readouterr().out
    main(["align", data, "--by", "mcs", "--reference", "2"])
    by_index = capsys.readouterr().out
    assert "onto m2" in by_name and "onto m2" in by_index


# --------------------------------------------------------------------------------- cofield


def test_cofield_reports_every_field_and_writes_cubes(tmp_path, capsys):
    _data, model = _model(tmp_path)
    prefix = os.path.join(str(tmp_path), "cof")
    main(["cofield", model, "--out-prefix", prefix, "--spacing", "1.0", "--z", "0"])
    text = capsys.readouterr().out
    assert "training molecules, grid" in text
    for name in ("electrostatic", "steric"):
        assert name in text
        assert os.path.exists(f"{prefix}_cofield_{name}.cube")
    head = open(f"{prefix}_cofield_electrostatic.cube", encoding="utf-8").read(400)
    assert "CMF" in head


def test_cofield_masks_at_the_given_number_of_standard_deviations(tmp_path, capsys):
    _data, model = _model(tmp_path)
    main(["cofield", model, "--spacing", "1.0", "--z", "2",
          "--csv", os.path.join(str(tmp_path), "a.csv")])
    masked = capsys.readouterr().out
    assert "standard deviations" in masked
    assert "of the covered grid above 2 sd" in masked
    import csv
    rows = list(csv.DictReader(open(os.path.join(str(tmp_path), "a.csv"))))
    assert [r["field"] for r in rows] == ["electrostatic", "steric"]
    assert all(0.0 <= float(r["share_above_z"]) <= 1.0 for r in rows)


def test_cofield_writes_nothing_unless_asked(tmp_path, capsys):
    _data, model = _model(tmp_path)
    main(["cofield", model, "--spacing", "1.2"])
    assert "Nothing written" in capsys.readouterr().out


def test_cofield_can_be_restricted_to_one_field(tmp_path, capsys):
    _data, model = _model(tmp_path)
    main(["cofield", model, "--spacing", "1.2", "--fields", "steric"])
    text = capsys.readouterr().out
    assert "steric" in text
    assert "electrostatic   peak" not in text


def test_cofield_refuses_a_field_the_model_does_not_have(tmp_path):
    _data, model = _model(tmp_path)
    with pytest.raises(SystemExit, match="none of those fields"):
        main(["cofield", model, "--fields", "hydrophobic"])


def test_a_molecule_kind_needs_a_molecule(tmp_path):
    _data, model = _model(tmp_path)
    with pytest.raises(SystemExit, match="needs --molecule"):
        main(["cofield", model, "--kind", "molecule", "--spacing", "1.2"])


def test_cofield_against_one_molecule(tmp_path, capsys):
    data, model = _model(tmp_path)
    main(["cofield", model, "--kind", "overlap", "--molecule", data, "--spacing", "1.0",
          "--csv", os.path.join(str(tmp_path), "o.csv")])
    text = capsys.readouterr().out
    assert "against m0" in text
    assert os.path.exists(os.path.join(str(tmp_path), "o.csv"))


# ------------------------------------------------------------------------------- benchmark


def test_benchmark_lists_what_there_is(capsys):
    main(["benchmark", "--list"])
    text = capsys.readouterr().out
    assert "benchmark" in text and "minutes" in text
    for name in ("alignment", "topomer", "speed"):
        assert name in text


def test_benchmark_checks_the_tables(capsys):
    main(["benchmark", "--check"])
    text = capsys.readouterr().out
    assert "check passed" in text or "problem" in text.lower()
