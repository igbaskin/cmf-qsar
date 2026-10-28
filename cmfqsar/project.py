"""
Projects (plan item G2): a session that can be saved and reopened, the poses before every
alignment step so that a step can be undone, and two models compared on one set of molecules.

A project file holds the data set with its current coordinates and properties, the pose history
(one snapshot of every conformer of every molecule, taken before each step that moved them), the
settings of the application's panels, the fitted model if there is one with the indices it was
fitted and tested on, the hyper-parameter search table, the validation report and the log.  It is
a joblib dump of a plain dictionary, like a model file, and the molecules travel as RDKit binary
blobs with their properties, so nothing is lost between sessions.

Nothing here needs Qt.  The application calls :func:`snapshot_poses` before a step that moves
molecules, :class:`PoseHistory` keeps the snapshots in order, :func:`save_project` and
:func:`load_project` move a :class:`Project` to and from disk, and :func:`compare_models` scores
two models on the same molecules.  The widgets' values are captured and restored by the
application itself (``cmfqsar.gui.panels.capture_settings``); here they are a dictionary.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D

from .dataio import Dataset
from .fields import AtomTable, build_fields
from .model import CMFModel, r2_score, rmse

#: version of the project file; a file written by a newer program is refused
FILE_VERSION = 1
PROJECT_EXTENSION = ".cmfproj"


# --------------------------------------------------------------------------- #
# Poses: snapshots and the undo history
# --------------------------------------------------------------------------- #

def snapshot_poses(mols: Sequence[Chem.Mol], label: str) -> dict:
    """Every conformer of every molecule, as it is now, under a label that says what is about
    to move them ("Align to reference", "Unalign", ...)."""
    coords = []
    for m in mols:
        confs = [c.GetPositions().astype(float) for c in m.GetConformers()]
        coords.append(np.stack(confs) if confs else np.zeros((0, m.GetNumAtoms(), 3)))
    return {"label": str(label), "time": _dt.datetime.now().isoformat(timespec="seconds"),
            "coords": coords}


def restore_poses(mols: Sequence[Chem.Mol], snapshot: dict) -> None:
    """Put the molecules back where the snapshot has them.  The conformers are replaced, so a
    snapshot taken before conformers were generated removes them again."""
    coords = snapshot["coords"]
    if len(coords) != len(mols):
        raise ValueError(f"the snapshot holds {len(coords)} molecules, the data set {len(mols)}")
    for m, xyz in zip(mols, coords):
        if xyz.shape[0] and xyz.shape[1] != m.GetNumAtoms():
            raise ValueError(f"{m.GetProp('_Name') if m.HasProp('_Name') else 'a molecule'} has "
                             f"{m.GetNumAtoms()} atoms now and {xyz.shape[1]} in the snapshot")
        m.RemoveAllConformers()
        for k, pos in enumerate(xyz):
            conf = rdchem.Conformer(m.GetNumAtoms())
            conf.SetId(int(k))
            conf.Set3D(True)
            for a, p in enumerate(pos):
                conf.SetAtomPosition(a, Point3D(float(p[0]), float(p[1]), float(p[2])))
            m.AddConformer(conf, assignId=False)


def pose_displacement(mols: Sequence[Chem.Mol], snapshot: dict) -> np.ndarray:
    """How far each molecule's first conformer now stands from the snapshot, as the root mean
    square displacement of its atoms in angstrom (nan where the shapes do not match)."""
    out = np.full(len(mols), np.nan)
    for i, (m, xyz) in enumerate(zip(mols, snapshot["coords"])):
        if m.GetNumConformers() == 0 or xyz.shape[0] == 0 or xyz.shape[1] != m.GetNumAtoms():
            continue
        now = m.GetConformer().GetPositions()
        out[i] = float(np.sqrt(((now - xyz[0]) ** 2).sum(axis=1).mean()))
    return out


class PoseHistory:
    """The snapshots taken before each step that moved the molecules, oldest first.

    ``remember`` is called before a step, ``undo`` puts the molecules back to the latest
    snapshot and drops it.  The history is bounded so that a long session of trial alignments on
    a large set does not grow without limit; the oldest snapshots go first.
    """

    def __init__(self, steps: Optional[List[dict]] = None, limit: int = 50):
        self.steps: List[dict] = list(steps or [])
        self.limit = int(limit)

    def __len__(self) -> int:
        return len(self.steps)

    def remember(self, mols: Sequence[Chem.Mol], label: str) -> dict:
        snap = snapshot_poses(mols, label)
        self.steps.append(snap)
        if len(self.steps) > self.limit:
            del self.steps[: len(self.steps) - self.limit]
        return snap

    @property
    def last_label(self) -> Optional[str]:
        return self.steps[-1]["label"] if self.steps else None

    def undo(self, mols: Sequence[Chem.Mol]) -> Tuple[str, np.ndarray]:
        """Restore the latest snapshot; returns its label and the displacement it undid."""
        if not self.steps:
            raise ValueError("there is no alignment step to undo")
        snap = self.steps[-1]
        moved = pose_displacement(mols, snap)
        restore_poses(mols, snap)
        self.steps.pop()
        return snap["label"], moved

    def labels(self) -> List[str]:
        return [s["label"] for s in self.steps]


# --------------------------------------------------------------------------- #
# The project file
# --------------------------------------------------------------------------- #

@dataclass
class Project:
    """Everything a session holds that is worth reopening."""
    dataset: Dataset
    settings: Dict[str, object] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)
    model_state: Optional[dict] = None
    train_indices: List[int] = field(default_factory=list)
    test_indices: List[int] = field(default_factory=list)
    decoy_indices: List[int] = field(default_factory=list)
    search_rows: List[dict] = field(default_factory=list)
    validation: Optional[dict] = None
    log: str = ""
    protein_path: str = ""
    notes: str = ""
    #: the protein structure behind the molecules, if one was loaded (a cmfqsar.protein.Protein)
    protein: Optional[object] = None
    #: the target the model was fitted to when it is not the activity column (the classes of a
    #: classification model), one value per molecule of the data set
    target: Optional[np.ndarray] = None
    #: when the project was last saved, and by which program version (set on save)
    saved: str = ""
    program_version: str = ""

    @property
    def model(self) -> Optional[CMFModel]:
        return CMFModel.from_state(self.model_state) if self.model_state else None


def _mol_to_blob(m: Chem.Mol) -> bytes:
    # the properties travel with the molecule: the activity column, the subset, the SD fields
    # the descriptors read, everything the file came with
    return m.ToBinary(Chem.PropertyPickleOptions.AllProps)


def _mol_from_blob(b: bytes) -> Chem.Mol:
    return Chem.Mol(b)


def dataset_state(ds: Dataset) -> dict:
    return {
        "mols": [_mol_to_blob(m) for m in ds.mols],
        "names": list(ds.names),
        "activity": np.asarray(ds.activity, float),
        "subset": [str(s) for s in ds.subset],
        "properties": list(ds.properties),
        "atom_properties": list(ds.atom_properties),
        "path": ds.path,
        "activity_prop": ds.activity_prop,
        "atom_activity_prop": ds.atom_activity_prop,
    }


def dataset_from_state(st: dict) -> Dataset:
    mols = [_mol_from_blob(b) for b in st["mols"]]
    return Dataset(mols=mols, names=list(st["names"]), activity=np.asarray(st["activity"], float),
                   subset=np.array(st["subset"], dtype=object), properties=list(st.get("properties", [])),
                   atom_properties=list(st.get("atom_properties", [])), path=st.get("path", ""),
                   activity_prop=st.get("activity_prop", ""),
                   atom_activity_prop=st.get("atom_activity_prop", ""))


def project_state(project: Project) -> dict:
    from . import __version__
    return {
        "kind": "cmf-project",
        "version": FILE_VERSION,
        "program_version": __version__,
        "saved": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_state(project.dataset),
        "settings": dict(project.settings),
        "history": [dict(s) for s in project.history],
        "model": project.model_state,
        "train_indices": [int(i) for i in project.train_indices],
        "test_indices": [int(i) for i in project.test_indices],
        "decoy_indices": [int(i) for i in project.decoy_indices],
        "search_rows": list(project.search_rows),
        "validation": project.validation,
        "log": project.log,
        "protein_path": project.protein_path,
        "notes": project.notes,
        "protein": project.protein,
        "target": None if project.target is None else np.asarray(project.target, float),
    }


def project_from_state(st: dict) -> Project:
    if not isinstance(st, dict) or st.get("kind") != "cmf-project":
        raise ValueError("not a CMF project file")
    version = int(st.get("version", 1))
    if version > FILE_VERSION:
        raise ValueError(f"this project file is version {version}, written by a newer cmfqsar than "
                         f"this one, which reads up to version {FILE_VERSION}")
    model_state = st.get("model")
    if model_state is not None:
        CMFModel.from_state(model_state)          # refuses a model from a newer program now
    return Project(dataset=dataset_from_state(st["dataset"]), settings=dict(st.get("settings", {})),
                   history=list(st.get("history", [])), model_state=model_state,
                   train_indices=list(st.get("train_indices", [])),
                   test_indices=list(st.get("test_indices", [])),
                   decoy_indices=list(st.get("decoy_indices", [])),
                   search_rows=list(st.get("search_rows", [])), validation=st.get("validation"),
                   log=str(st.get("log", "")), protein_path=str(st.get("protein_path", "")),
                   notes=str(st.get("notes", "")), protein=st.get("protein"),
                   target=st.get("target"), saved=str(st.get("saved", "")),
                   program_version=str(st.get("program_version", "")))


def save_project(path: str, project: Project) -> None:
    """Write the project; the file is a compressed joblib dump of :func:`project_state`."""
    st = project_state(project)
    joblib.dump(st, path, compress=3)
    project.saved, project.program_version = st["saved"], st["program_version"]


def load_project(path: str) -> Project:
    """Read a project file; refuses one written by a newer version of the program."""
    try:
        st = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{path} could not be read as a project file: {exc}") from exc
    return project_from_state(st)


def is_project_file(path: str) -> bool:
    """Whether a file on disk is a project rather than a model, judged by its content."""
    try:
        st = joblib.load(path)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(st, dict) and st.get("kind") == "cmf-project"


# --------------------------------------------------------------------------- #
# Two models on the same molecules
# --------------------------------------------------------------------------- #

@dataclass
class ComparisonRow:
    index: int
    name: str
    subset: str
    y: float
    pred_a: float
    pred_b: float

    @property
    def difference(self) -> float:
        return self.pred_b - self.pred_a


@dataclass
class Comparison:
    """Two models' predictions for one set of molecules, and what they agree on."""
    label_a: str
    label_b: str
    rows: List[ComparisonRow]
    #: per subset ("train", "test", "all"): n, rmse and r2 of each model against the activity
    summary: Dict[str, dict]
    #: correlation of the two prediction vectors, their mean and largest absolute difference,
    #: the molecule where they differ most, and how often B is nearer the activity than A
    agreement: dict

    def lines(self) -> List[str]:
        """The comparison as text, for the log."""
        out = [f"Two models on {len(self.rows)} molecules: A = {self.label_a}, B = {self.label_b}"]
        for sub, s in self.summary.items():
            if s["n"] == 0:
                continue
            out.append(f"  {sub:5s} n = {s['n']:3d}   RMSE  A {s['rmse_a']:.3f}  B {s['rmse_b']:.3f}"
                       f"   r2  A {s['r2_a']:.3f}  B {s['r2_b']:.3f}")
        ag = self.agreement
        out.append(f"  predictions: correlation {ag['correlation']:.3f}, mean |B - A| "
                   f"{ag['mean_abs_difference']:.3f}, largest {ag['max_abs_difference']:.3f} "
                   f"({ag['most_different']})")
        if ag["n_with_activity"]:
            out.append(f"  B nearer the measured activity on {ag['b_closer']} of "
                       f"{ag['n_with_activity']} molecules with an activity")
        return out

    def to_csv(self, path: str) -> None:
        import csv
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "subset", "experimental", f"predicted_{self.label_a}",
                        f"predicted_{self.label_b}", "difference_B_minus_A"])
            for r in self.rows:
                w.writerow([r.name, r.subset, _num(r.y), _num(r.pred_a), _num(r.pred_b),
                            _num(r.difference)])


def _num(v) -> str:
    return "" if v is None or not math.isfinite(v) else f"{v:.4f}"


def predict_dataset(model: CMFModel, ds: Dataset, progress=None) -> np.ndarray:
    """One prediction per molecule of the data set, with the fields the model was fitted with."""
    mf = []
    for k, (m, n) in enumerate(zip(ds.mols, ds.names)):
        mf.append(build_fields(m, model.fieldset, n))
        if progress and k % 20 == 0:
            progress(f"fields {k + 1}/{len(ds)}", k / max(len(ds), 1))
    return np.asarray(model.predict(AtomTable.from_molecules(mf)), float)


def _scores(y, p) -> Tuple[float, float]:
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    return float(rmse(y[ok], p[ok])), float(r2_score(y[ok], p[ok]))


def compare_models(model_a: CMFModel, model_b: CMFModel, ds: Dataset,
                   labels: Tuple[str, str] = ("A", "B"), progress=None,
                   pred_a: Optional[np.ndarray] = None,
                   pred_b: Optional[np.ndarray] = None) -> Comparison:
    """Score two models on the same molecules.

    Both models see the molecules as they are now, in one frame: a model fitted on another
    alignment of the same series is compared on *this* alignment, which is the comparison the
    panel is for (did the realignment help?) and also its caveat.  Predictions already in hand
    can be passed in to save the fields.
    """
    y = np.asarray(ds.activity, float)
    if pred_a is None:
        pred_a = predict_dataset(model_a, ds, progress=progress)
    if pred_b is None:
        pred_b = predict_dataset(model_b, ds, progress=progress)
    pred_a, pred_b = np.asarray(pred_a, float), np.asarray(pred_b, float)
    subset = [str(s) for s in ds.subset]
    rows = [ComparisonRow(i, ds.names[i], subset[i], float(y[i]), float(pred_a[i]), float(pred_b[i]))
            for i in range(len(ds))]
    summary = {}
    for sub in ("train", "test", "all"):
        mask = np.array([s == sub or sub == "all" for s in subset]) & np.isfinite(y)
        ra, qa = _scores(y[mask], pred_a[mask])
        rb, qb = _scores(y[mask], pred_b[mask])
        summary[sub] = {"n": int(mask.sum()), "rmse_a": ra, "rmse_b": rb, "r2_a": qa, "r2_b": qb}
    ok = np.isfinite(pred_a) & np.isfinite(pred_b)
    d = np.abs(pred_b - pred_a)
    corr = float(np.corrcoef(pred_a[ok], pred_b[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    worst = int(np.nanargmax(np.where(ok, d, -1))) if ok.any() else -1
    with_y = ok & np.isfinite(y)
    agreement = {
        "correlation": corr,
        "mean_abs_difference": float(np.mean(d[ok])) if ok.any() else float("nan"),
        "max_abs_difference": float(d[worst]) if worst >= 0 else float("nan"),
        "most_different": ds.names[worst] if worst >= 0 else "",
        "n_with_activity": int(with_y.sum()),
        "b_closer": int(np.sum(np.abs(pred_b - y)[with_y] < np.abs(pred_a - y)[with_y])),
    }
    return Comparison(labels[0], labels[1], rows, summary, agreement)
