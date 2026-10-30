"""The supported interface of the package (plan item G3).

`cmfqsar.__all__` is a promise: within a major version those names are not removed or renamed, and
the arguments recorded here keep their names, their order and their defaults.  New arguments may
be added after them, which is why this file records the arguments the manual and the tutorial
notebooks use rather than comparing whole signatures: a model gains a field almost every week, and
a test that fails on every new field would be deleted within a month.

The last two tests read `docs/manual.md` itself, so that an example in the manual cannot go on
importing a name the package no longer has.
"""
import dataclasses
import importlib
import inspect
import io
import os
import re
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cmfqsar  # noqa: E402
from cmfqsar import AtomTable, CMFModel, Dataset, FieldSet, ModelStats, build_fields  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUAL = os.path.join(ROOT, "docs", "manual.md")

# every name the package offers, and what the manual calls it
SURFACE = ("AtomTable", "CMFModel", "Dataset", "FieldSet", "FieldSpec", "ModelStats",
           "MoleculeFields", "build_fields", "load_dataset")

# the arguments each one promises, in the order a positional call may rely on
PROMISED = {
    "build_fields": ["mol", "fieldset", "name"],
    "load_dataset": ["path", "activity_prop", "add_hs", "subset"],
    "AtomTable.from_molecules": ["mols"],
    "FieldSet.create": ["physchem", "cif_types", "alpha", "charge_model"],
    "CMFModel": ["fieldset", "task", "method", "params"],
    "CMFModel.fit": ["train", "y", "cv_folds", "reuse_kernels"],
    "CMFModel.predict": ["test", "return_fields"],
    "CMFModel.predict_interval": ["test"],
    "CMFModel.calibrate_conformal": ["level"],
    "CMFModel.rank_score": ["test", "beta", "larger_is_better"],
}

# the defaults a caller may leave out and still get what the manual describes
DEFAULTS = {
    ("load_dataset", "add_hs"): True,
    ("load_dataset", "subset"): "train",
    ("FieldSet.create", "alpha"): 1.0,
    ("FieldSet.create", "charge_model"): "gasteiger",
    ("CMFModel", "task"): "regression",
    ("CMFModel", "method"): "krr",
    ("CMFModel.fit", "cv_folds"): 10,
    ("CMFModel.fit", "reuse_kernels"): False,
    ("CMFModel.predict", "return_fields"): False,
    ("CMFModel.rank_score", "beta"): 0.0,
    ("CMFModel.rank_score", "larger_is_better"): True,
    ("CMFModel.calibrate_conformal", "level"): 0.9,
}

SMILES = ["c1ccccc1C(=O)O", "c1ccc(Cl)cc1C(=O)O", "c1ccc(C)cc1C(=O)O", "c1ccc(O)cc1C(=O)O",
          "c1ccc(N)cc1C(=O)O", "c1ccc(F)cc1C(=O)O"]


def _resolve(dotted):
    """The callable a key of PROMISED names, and the signature a caller sees."""
    obj = cmfqsar
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj, inspect.signature(obj)


def test_the_surface_is_what_the_package_declares():
    assert sorted(cmfqsar.__all__) == sorted(SURFACE), cmfqsar.__all__
    for name in cmfqsar.__all__:
        assert hasattr(cmfqsar, name), name
        assert getattr(cmfqsar, name) is not None, name


@pytest.mark.parametrize("dotted", sorted(PROMISED))
def test_the_promised_arguments_are_still_there(dotted):
    """A promised argument keeps its name and its place; new ones go after it."""
    _, sig = _resolve(dotted)
    have = [p for p in sig.parameters if p != "self"]
    want = PROMISED[dotted]
    missing = [p for p in want if p not in have]
    assert not missing, f"{dotted} no longer takes {missing}"
    assert [p for p in have if p in want] == want, f"{dotted}: {want} are no longer in that order"


@pytest.mark.parametrize("dotted,arg", sorted(DEFAULTS))
def test_the_promised_defaults_are_unchanged(dotted, arg):
    _, sig = _resolve(dotted)
    assert sig.parameters[arg].default == DEFAULTS[(dotted, arg)], f"{dotted}({arg}=...)"


def _readable(cls):
    """Every name a script can read off an instance: the dataclass fields and the properties."""
    names = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
    return names | {n for n in dir(cls) if not n.startswith("_")}


def test_the_data_set_and_the_statistics_keep_their_readers():
    """What a script reads off a Dataset and off ModelStats after a fit."""
    have = _readable(Dataset)
    for attr in ("mols", "names", "activity", "subset", "path", "train_indices", "test_indices"):
        assert attr in have, attr
    have = _readable(ModelStats)
    for attr in ("n_train", "r2", "q2", "rmse_cv", "r2_pred", "summary"):
        assert attr in have, attr


def test_the_surface_alone_fits_and_predicts(tmp_path):
    """The whole of the manual's first example, with nothing imported from a private module."""
    mols = []
    for k, smi in enumerate(SMILES):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = k + 1
        AllChem.EmbedMolecule(m, p)
        mols.append(m)
    fs = FieldSet.create(physchem=("steric", "electrostatic"), alpha=0.5)
    mf = [build_fields(m, fs, f"m{k}") for k, m in enumerate(mols)]
    table = AtomTable.from_molecules(mf)
    y = np.array([1.0, 2.0, 1.5, 2.5, 3.0, 2.0])
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.1})
    model.fit(table, y, cv_folds=3)
    pred = model.predict(AtomTable.from_molecules(mf[:2]))
    assert len(pred) == 2 and np.all(np.isfinite(pred))
    assert isinstance(model.stats.summary(), str)


def _library_chapter():
    """The chapter's own text, ending at the next chapter heading and not at a code comment.

    A `# ` at the start of a line inside a fenced block is a Python comment, so the end of a
    chapter can only be found by tracking the fences.
    """
    lines = io.open(MANUAL, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.startswith("# Using it as a library")),
                 None)
    assert start is not None, "the manual has no library chapter"
    fenced, end = False, len(lines)
    for i, ln in enumerate(lines[start + 1:], start + 1):
        if ln.startswith("```"):
            fenced = not fenced
        elif not fenced and ln.startswith("# "):
            end = i
            break
    return "\n".join(lines[start:end])


def test_the_manual_imports_only_names_that_exist():
    """Every `from cmfqsar... import ...` of the library chapter resolves."""
    bad = []
    for module, names in re.findall(r"^from (cmfqsar[\w.]*) import ([^\n]+)", _library_chapter(),
                                    re.M):
        mod = importlib.import_module(module)
        for name in (n.strip() for n in names.split(",")):
            name = name.split(" as ")[0].strip()
            if hasattr(mod, name):
                continue
            try:                                   # `from cmfqsar import multitask as MT`
                importlib.import_module(f"{module}.{name}")
            except ImportError:
                bad.append(f"{module}.{name}")
    assert not bad, f"the manual imports names the package no longer has: {bad}"


def test_the_manual_names_the_stable_surface():
    """The chapter has to say which names are supported, or the promise is not documented."""
    chapter = _library_chapter()
    assert "stable surface" in chapter.lower(), "the library chapter does not say what is stable"
    for name in cmfqsar.__all__:
        assert name in chapter, f"{name} is on the surface and not in the manual chapter"
