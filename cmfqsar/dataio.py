"""
Reading structure-activity data sets (SDF / MOL2 / SMILES), saving models and
exporting results.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from .model import CMFModel

MOL2_HEADER = "@<TRIPOS>MOLECULE"

_MOL2_BOND_TYPES = {
    "1": Chem.BondType.SINGLE, "2": Chem.BondType.DOUBLE, "3": Chem.BondType.TRIPLE,
    "am": Chem.BondType.SINGLE, "ar": Chem.BondType.AROMATIC, "du": Chem.BondType.SINGLE,
    "un": Chem.BondType.SINGLE, "nc": None,
}
_MOL2_DUMMY_TYPES = {"du", "du.c", "lp", "any", "hev", "het", "hal"}


def _try_sanitize(m: Chem.Mol) -> bool:
    """Full sanitisation, falling back to a partial one (no valence check / kekulisation)."""
    try:
        Chem.SanitizeMol(m)
        return True
    except Exception:
        pass
    try:
        ops = (Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
               ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.SanitizeMol(m, sanitizeOps=ops)
        m.UpdatePropertyCache(strict=False)
        return True
    except Exception:
        try:
            m.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(m)
            return True
        except Exception:
            return False


def _mol2_fallback(block: str) -> Optional[Chem.Mol]:
    """Tolerant MOL2 parser used when RDKit rejects a block (e.g. unusual valences).

    Atoms, coordinates, Sybyl atom types, partial charges and bonds are read
    literally; the molecule is sanitised as far as possible.  The Sybyl type and
    charge are stored in the atom properties ``_TriposAtomType`` and
    ``_TriposPartialCharge`` as RDKit's own reader does.
    """
    sections: Dict[str, List[str]] = {}
    cur = None
    for ln in block.splitlines():
        if ln.startswith("@<TRIPOS>"):
            cur = ln.strip()[9:].upper()
            sections.setdefault(cur, [])
            continue
        if cur is not None:
            sections[cur].append(ln)
    mol_lines = sections.get("MOLECULE", [])
    name = mol_lines[0].strip() if mol_lines else ""
    atoms = []
    for ln in sections.get("ATOM", []):
        p = ln.split()
        if len(p) < 6:
            continue
        try:
            aid = int(p[0])
            xyz = (float(p[2]), float(p[3]), float(p[4]))
        except ValueError:
            continue
        q = 0.0
        if len(p) >= 9:
            try:
                q = float(p[8])
            except ValueError:
                q = 0.0
        atoms.append((aid, p[1], xyz, p[5], q))
    if not atoms:
        return None
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    conf.Set3D(True)
    idmap: Dict[int, int] = {}
    for aid, aname, xyz, syb, q in atoms:
        el = syb.split(".")[0]
        el = el[0].upper() + el[1:].lower() if len(el) > 1 else el.upper()
        if syb.lower() in _MOL2_DUMMY_TYPES:
            atom = Chem.Atom(0)
        else:
            try:
                atom = Chem.Atom(el)
            except Exception:
                atom = Chem.Atom(0)
        atom.SetNoImplicit(True)
        if syb == "N.4":
            atom.SetFormalCharge(1)
        atom.SetProp("_TriposAtomType", syb)
        atom.SetProp("_TriposAtomName", aname)
        atom.SetDoubleProp("_TriposPartialCharge", q)
        idx = rw.AddAtom(atom)
        idmap[aid] = idx
        conf.SetAtomPosition(idx, Point3D(*xyz))
    aromatic = set()
    for ln in sections.get("BOND", []):
        p = ln.split()
        if len(p) < 4:
            continue
        try:
            a1, a2 = idmap[int(p[1])], idmap[int(p[2])]
        except (KeyError, ValueError):
            continue
        bt = _MOL2_BOND_TYPES.get(p[3].lower(), Chem.BondType.SINGLE)
        if bt is None or a1 == a2 or rw.GetBondBetweenAtoms(a1, a2) is not None:
            continue
        rw.AddBond(a1, a2, bt)
        if bt == Chem.BondType.AROMATIC:
            aromatic.update((a1, a2))
            rw.GetBondBetweenAtoms(a1, a2).SetIsAromatic(True)
    for i in aromatic:
        rw.GetAtomWithIdx(i).SetIsAromatic(True)
    m = rw.GetMol()
    m.AddConformer(conf, assignId=True)
    m.SetProp("_Name", name)
    _try_sanitize(m)
    return m


def mol_from_mol2_block(block: str) -> Optional[Chem.Mol]:
    """Parse one MOL2 block with RDKit; fall back to the tolerant parser on failure."""
    m = Chem.MolFromMol2Block(block, removeHs=False, sanitize=True)
    if m is not None:
        return m
    for cleanup in (True, False):
        m = Chem.MolFromMol2Block(block, removeHs=False, sanitize=False, cleanupSubstructures=cleanup)
        if m is not None and _try_sanitize(m):
            return m
    return _mol2_fallback(block)


@dataclass
class Dataset:
    mols: List[Chem.Mol] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    activity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    subset: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))   # "train" / "test"
    properties: List[str] = field(default_factory=list)
    atom_properties: List[str] = field(default_factory=list)   # one value per atom, for prop: fields
    path: str = ""
    activity_prop: str = ""

    def __len__(self) -> int:
        return len(self.mols)

    @property
    def train_indices(self) -> np.ndarray:
        return np.where(self.subset == "train")[0]

    @property
    def test_indices(self) -> np.ndarray:
        return np.where(self.subset == "test")[0]

    @property
    def decoy_indices(self) -> np.ndarray:
        """Background molecules, presumed inactive; never used for fitting."""
        return np.where(self.subset == "decoy")[0]

    def random_split(self, test_fraction: float = 0.25, seed: int = 0) -> None:
        n = len(self)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_test = int(round(test_fraction * n))
        self.subset = np.array(["train"] * n, dtype=object)
        self.subset[idx[:n_test]] = "test"

    def set_activity_from_property(self, prop: str) -> None:
        vals = []
        for m in self.mols:
            v = float("nan")
            if m.HasProp(prop):
                try:
                    v = float(str(m.GetProp(prop)).strip().split()[0])
                except (ValueError, IndexError):
                    v = float("nan")
            vals.append(v)
        self.activity = np.array(vals, dtype=float)
        self.activity_prop = prop

    # ------------------------------------------------------------------ classes
    def classes_from_threshold(self, threshold: float, active_above: bool = True) -> np.ndarray:
        """+1 / -1 class labels obtained by thresholding the activity (NaN stays NaN).

        Molecules of the "decoy" subset are presumed inactive whatever their activity.
        """
        y = np.asarray(self.activity, dtype=float)
        out = np.full(len(y), np.nan)
        ok = np.isfinite(y)
        if active_above:
            out[ok] = np.where(y[ok] >= threshold, 1.0, -1.0)
        else:
            out[ok] = np.where(y[ok] <= threshold, 1.0, -1.0)
        out[self.decoy_indices] = -1.0
        return out

    def classes_from_property(self, prop: str, positive=("1", "+1", "active", "yes", "true", "a", "p")) -> np.ndarray:
        """+1 / -1 class labels read from an SD property (numeric or a label such as active/inactive)."""
        pos = {str(p).strip().lower() for p in positive}
        out = np.full(len(self.mols), np.nan)
        for i, m in enumerate(self.mols):
            if not m.HasProp(prop):
                continue
            raw = str(m.GetProp(prop)).strip()
            low = raw.lower()
            if low in pos:
                out[i] = 1.0
                continue
            try:
                v = float(raw)
                out[i] = 1.0 if v > 0 else -1.0
            except ValueError:
                out[i] = -1.0
        out[self.decoy_indices] = -1.0
        return out

    def append(self, other: "Dataset", subset: str = "test") -> None:
        self.mols += other.mols
        self.names += other.names
        self.activity = np.concatenate([self.activity, other.activity])
        self.subset = np.concatenate([self.subset, np.array([subset] * len(other), dtype=object)])
        for p in other.properties:
            if p not in self.properties:
                self.properties.append(p)

    def remove(self, indices: Sequence[int]) -> None:
        keep = [i for i in range(len(self)) if i not in set(indices)]
        self.mols = [self.mols[i] for i in keep]
        self.names = [self.names[i] for i in keep]
        self.activity = self.activity[keep]
        self.subset = self.subset[keep]


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def _ensure_3d_with_hs(mol: Chem.Mol, add_hs: bool, seed: int = 42) -> Chem.Mol:
    """Add explicit hydrogens (with coordinates) and 3D coordinates when missing."""
    has_conf = mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()
    n_h = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    if not has_conf:
        from .align import embed_3d
        return embed_3d(mol, seed=seed)
    if add_hs and n_h == 0 and any(a.GetTotalNumHs() > 0 for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=True)
    return mol


def _mol2_blocks(text: str) -> List[str]:
    parts = text.split(MOL2_HEADER)
    return [MOL2_HEADER + p for p in parts[1:] if p.strip()]


def common_atom_properties(mols, limit: int = 25) -> List[str]:
    """Per-atom properties that every one of the first `limit` molecules carries.

    These are what a ``prop:`` field can be built from; only the properties present throughout
    are offered, since a field that exists for half a data set is not a field.
    """
    from .fields import atom_property_names
    shared = None
    for m in [x for x in mols if x is not None][:limit]:
        here = set(atom_property_names(m))
        shared = here if shared is None else (shared & here)
        if not shared:
            break
    return sorted(shared or [])


def read_molecules(path: str, add_hs: bool = True):
    """Return (mols, names, property_names). Supports .sdf/.sd/.mol, .mol2, .smi/.csv/.txt."""
    ext = os.path.splitext(path)[1].lower()
    mols: List[Chem.Mol] = []
    names: List[str] = []
    props: List[str] = []
    if ext in (".sdf", ".sd", ".mol", ".mdl"):
        suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
        for k, m in enumerate(suppl):
            if m is None:
                continue
            m = _ensure_3d_with_hs(m, add_hs, seed=k)
            mols.append(m)
            names.append(m.GetProp("_Name") if m.HasProp("_Name") and m.GetProp("_Name").strip() else f"mol_{k + 1}")
            for p in m.GetPropNames():
                if p not in props:
                    props.append(p)
    elif ext == ".mol2":
        with open(path, "r", errors="replace") as fh:
            text = fh.read()
        for k, block in enumerate(_mol2_blocks(text)):
            m = mol_from_mol2_block(block)
            if m is None:
                continue
            m = _ensure_3d_with_hs(m, add_hs, seed=k)
            mols.append(m)
            nm = block.splitlines()[1].strip() if len(block.splitlines()) > 1 else ""
            names.append(nm or f"mol_{k + 1}")
    elif ext in (".smi", ".smiles", ".csv", ".txt", ".tsv"):
        from .align import embed_3d
        rows = _read_smiles_table(path)
        for k, (smi, name, extra) in enumerate(rows):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            m = embed_3d(m, seed=k)
            m.SetProp("_Name", name)
            for key, val in extra.items():
                m.SetProp(key, val)
                if key not in props:
                    props.append(key)
            mols.append(m)
            names.append(name or f"mol_{k + 1}")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return mols, names, props


def _read_smiles_table(path: str):
    """Rows of (smiles, name, {other columns}). CSV with header or whitespace-separated .smi."""
    ext = os.path.splitext(path)[1].lower()
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        if ext in (".csv", ".tsv"):
            dialect = csv.excel_tab if ext == ".tsv" else csv.excel
            reader = csv.reader(fh, dialect)
            header = next(reader)
            lower = [h.strip().lower() for h in header]
            smi_col = next((i for i, h in enumerate(lower) if h in ("smiles", "smi")), 0)
            name_col = next((i for i, h in enumerate(lower) if h in ("name", "id", "title", "compound")), None)
            for k, r in enumerate(reader):
                if not r or not r[smi_col].strip():
                    continue
                name = r[name_col].strip() if name_col is not None and name_col < len(r) else f"mol_{k + 1}"
                extra = {header[i].strip(): r[i] for i in range(len(r))
                         if i not in (smi_col, name_col) and header[i].strip()}
                rows.append((r[smi_col].strip(), name, extra))
        else:
            for k, line in enumerate(fh):
                parts = line.split()
                if not parts or parts[0].startswith("#"):
                    continue
                name = parts[1] if len(parts) > 1 else f"mol_{k + 1}"
                extra = {f"col{i}": parts[i] for i in range(2, len(parts))}
                rows.append((parts[0], name, extra))
    return rows


def numeric_properties(mols: Sequence[Chem.Mol], min_fraction: float = 0.5) -> List[str]:
    """SD properties that are numeric for at least `min_fraction` of the molecules."""
    counts: Dict[str, int] = {}
    for m in mols:
        for p in m.GetPropNames():
            if p.startswith("atom."):      # per-atom property lists (atom.prop.*, atom.dprop.*)
                continue
            try:
                float(str(m.GetProp(p)).strip().split()[0])
                counts[p] = counts.get(p, 0) + 1
            except (ValueError, IndexError):
                pass
    n = max(1, len(mols))
    return [p for p, c in counts.items() if c / n >= min_fraction]


def load_dataset(path: str, activity_prop: Optional[str] = None, add_hs: bool = True,
                 subset: str = "train", activity_file: Optional[str] = None,
                 use_subset_column: bool = True, ensemble_by: Optional[str] = None) -> Dataset:
    """Load a structure file; activities come from an SD property or from a separate table.

    `activity_file` is a text table (tab/comma/whitespace separated, header optional) with the
    molecule name in the first column, the activity in the column whose header mentions the
    activity (else the second column) and, optionally, a set/subset column (1/"train", 2/"test")
    defining the published train/test split.

    `ensemble_by` collapses a multi-conformer file into one molecule per compound, carrying all
    of its conformations: "name" groups records that share a title, anything else is read as the
    name of a property to group by.  The conformations are used only in the ensemble mode of the
    field set; in the default single mode the first one of each compound is used, as before.
    """
    mols, names, props = read_molecules(path, add_hs=add_hs)
    if ensemble_by:
        from .conformers import group_conformers
        mols, names, counts = group_conformers(mols, names, by=ensemble_by)
        if counts:
            props = [p for p in props if any(m.HasProp(p) for m in mols)]
    ds = Dataset(mols=mols, names=names, activity=np.full(len(mols), np.nan),
                 subset=np.array([subset] * len(mols), dtype=object), properties=props, path=path,
                 atom_properties=common_atom_properties(mols))
    if activity_file:
        apply_activity_table(ds, read_activity_table(activity_file), use_subset=use_subset_column)
        ds.activity_prop = os.path.basename(activity_file)
        return ds
    if activity_prop is None:
        activity_prop = guess_activity_property(numeric_properties(mols))
    if activity_prop:
        ds.set_activity_from_property(activity_prop)
    if use_subset_column:
        # a per-molecule SD tag "subset" / "set" / "split" (train/test or 1/2) defines the published split
        tag = next((p for p in props if re.fullmatch(r"set|subset|split|training|train/test", p, re.I)), None)
        if tag is not None:
            labels = [_subset_label(m.GetProp(tag)) if m.HasProp(tag) else None for m in mols]
            if any(l is not None for l in labels):
                ds.subset = np.array([l or subset for l in labels], dtype=object)
    return ds


def guess_activity_property(cands: Sequence[str]) -> Optional[str]:
    """Pick the most plausible activity column among numeric SD properties."""
    if not cands:
        return None
    exact = ("pAffinity", "pactivity", "activity", "pIC50", "pKi", "pKd", "pEC50", "pic50", "pki", "pkd",
             "y", "target", "value")
    low = {c.lower(): c for c in cands}
    for e in exact:
        if e.lower() in low:
            return low[e.lower()]
    good = re.compile(r"(^|[^a-z])(p?ic50|p?ki|p?kd|p?ec50|pk$|activ|affin|potenc|log)", re.I)
    bad = re.compile(r"^[a-z]_|tautomer|ionization|penalty|probab|energy|score|charge|weight|mass|logp|mw", re.I)
    pref = [p for p in cands if good.search(p) and not bad.search(p)]
    if pref:
        return pref[0]
    rest = [p for p in cands if not bad.search(p)]
    return (rest or list(cands))[0]


def _subset_label(v: str) -> Optional[str]:
    v = v.strip().strip('"').lower()
    if v in ("1", "train", "training", "tr", "t"):
        return "train"
    if v in ("2", "test", "pred", "prediction", "validation", "ext", "external", "te", "p"):
        return "test"
    if v in ("3", "decoy", "decoys", "background", "library", "inactive_presumed"):
        return "decoy"
    return None


def read_activity_table(path: str):
    """Rows (name, activity, subset) of a 'name activity [set]' table (header optional).

    Separator is tab, comma or whitespace; quotes around names are removed; rows without a
    numeric activity are skipped.  The set column (header matching set/subset/split, or the
    third column of a header-less file) is mapped to "train" (1) / "test" (2).
    """
    with open(path, "r", errors="replace") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    if not lines:
        return []
    sep = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)

    def split(ln):
        return [c.strip().strip('"') for c in (ln.split(sep) if sep else ln.split())]

    def isnum(v):
        try:
            float(v)
            return True
        except ValueError:
            return False

    first = split(lines[0])
    header = None
    start = 0
    if len(first) >= 2 and not isnum(first[1]):
        header = [h.lower() for h in first]
        start = 1
    act_col, set_col = 1, None
    if header:
        for i, h in enumerate(header):
            if i > 0 and re.search(r"activ|^y$|^pic|^pki|^pkd|ic50|^act|^value", h):
                act_col = i
                break
        for i, h in enumerate(header):
            if re.fullmatch(r"set|subset|split|training|train/test", h):
                set_col = i
    else:
        if len(first) >= 3 and _subset_label(first[2]) is not None:
            set_col = 2
    rows = []
    for ln in lines[start:]:
        p = split(ln)
        if len(p) <= act_col or not isnum(p[act_col]):
            continue
        sub = _subset_label(p[set_col]) if set_col is not None and len(p) > set_col else None
        rows.append((p[0], float(p[act_col]), sub))
    return rows


def apply_activity_table(ds: Dataset, rows, use_subset: bool = True) -> int:
    """Assign activities (and train/test membership) from `read_activity_table` rows.

    Names are matched case-insensitively; if fewer than half of the molecules match by name
    but the table has exactly one row per molecule, the rows are applied in order.
    Returns the number of molecules that received an activity.
    """
    lookup = {}
    for name, act, sub in rows:
        lookup.setdefault(name.strip().lower(), (act, sub))
    y = np.full(len(ds), np.nan)
    subset = np.array(ds.subset, dtype=object)
    matched = sum(1 for n in ds.names if n.strip().lower() in lookup)
    if matched < 0.5 * len(ds) and len(rows) == len(ds):
        for i, (_, act, sub) in enumerate(rows):
            y[i] = act
            if sub:
                subset[i] = sub
        matched = len(ds)
    else:
        for i, n in enumerate(ds.names):
            hit = lookup.get(n.strip().lower())
            if hit is not None:
                y[i] = hit[0]
                if hit[1]:
                    subset[i] = hit[1]
    ds.activity = y
    if use_subset:
        ds.subset = subset
    return matched


def read_activities_csv(path: str) -> Dict[str, float]:
    """name,value CSV (header optional) -> dict."""
    out: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                out[row[0].strip()] = float(row[1])
            except ValueError:
                continue
    return out


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def save_model(path: str, model: CMFModel, extra: Optional[dict] = None) -> None:
    """Write the model with its provenance: how its training set was prepared, and by what.

    The file is a joblib dump of :meth:`CMFModel.state`, version 2: everything a prediction
    needs, plus the record that :mod:`cmfqsar.contract` checks new molecules against.
    """
    st = model.state()
    if extra:
        st["extra"] = extra
    joblib.dump(st, path, compress=3)


def load_model(path: str):
    """Read a model file; refuses one written by a newer version of the program."""
    st = joblib.load(path)
    if not isinstance(st, dict) or "fieldset" not in st:
        raise ValueError(f"{path} is not a CMF model file")
    return CMFModel.from_state(st), st.get("extra", {})


def write_predictions_csv(path: str, names: Sequence[str], y: Sequence[float], yp: Sequence[float],
                          subset: Sequence[str], y_cv: Optional[Sequence[float]] = None,
                          var: Optional[Sequence[float]] = None, value_header: str = "predicted",
                          domain: Optional[Sequence[float]] = None,
                          domain_threshold: Optional[float] = None,
                          truth_header: str = "experimental") -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["name", "subset", truth_header, value_header]
        if y_cv is not None:
            header.append("predicted_cv")
        if var is not None:
            header.append("gp_variance")
        if domain is not None:
            # the similarity itself and the verdict, so that an extrapolation stays marked
            # after the predictions leave the program
            header += ["ad_similarity", "in_domain"]
        w.writerow(header)
        for i, nm in enumerate(names):
            row = [nm, subset[i], "" if not np.isfinite(y[i]) else f"{y[i]:.4f}",
                   "" if yp[i] is None or not np.isfinite(yp[i]) else f"{yp[i]:.4f}"]
            if y_cv is not None:
                row.append("" if y_cv[i] is None or not np.isfinite(y_cv[i]) else f"{y_cv[i]:.4f}")
            if var is not None:
                row.append("" if var[i] is None or not np.isfinite(var[i]) else f"{var[i]:.4g}")
            if domain is not None:
                d = domain[i]
                row.append("" if d is None or not np.isfinite(d) else f"{d:.4f}")
                row.append("" if d is None or not np.isfinite(d) or domain_threshold is None
                           else ("yes" if d >= domain_threshold else "no"))
            w.writerow(row)


def write_sdf(path: str, ds: Dataset, activity_prop: str = "activity") -> None:
    w = Chem.SDWriter(path)
    for m, nm, y, sub in zip(ds.mols, ds.names, ds.activity, ds.subset):
        mm = Chem.Mol(m)
        mm.SetProp("_Name", nm)
        if np.isfinite(y):
            mm.SetProp(activity_prop, f"{y:.4f}")
        mm.SetProp("subset", str(sub))
        w.write(mm)
    w.close()
