"""Turn the "Iron Inhibitor Scaffolds" artifact into CMF-ready data sets, one per scaffold.

The artifact (https://claude.ai/artifact/8h4apgDS9F4Rfi6oZGSPRT) is a Schuffenhauer scaffold tree
of the corrosion inhibitors of iron-based alloys in the user's database.  Its page carries the
whole tree as JSON in ``<script type="application/json" id="data">``: ``nodes`` (one ring system
each: ``s`` scaffold SMILES, ``r`` depth, ``p`` parent index) and ``comps``, one record per
compound::

    [id, name, MW, n_rows, n_papers, raw, adj, hcl_raw, hcl_adj, [node indices root -> leaf],
     SMILES, thermo[9], h2so4_raw, h2so4_adj, diff]

where the potencies are MWILA = ln[theta / (C (1 - theta))], the median over the compound's
experiments, pooled over all media (``raw``, and ``adj`` after subtracting the molecular-weight
fit), restricted to 1 M HCl (``hcl_*``) or to plain sulfuric acid (``h2so4_*``), and
``thermo[0]`` is the Langmuir adsorption free energy in kJ/mol.

A CMF model needs one series: molecules that share a substructure to superimpose on and one
activity measured under one condition.  So the script

1. ranks the tree nodes by the number of compounds below them that have a value in the chosen
   medium (for 1 M HCl this is the page's ``hn``; it is recounted here from the compound records,
   which also serves the other media),
2. writes one CSV per selected node, ``<outdir>/<node_index>_<scaffold>.csv``, with the chosen
   MWILA in the ``activity`` column and every other MWILA variant present in the node as a
   further column, so the activity can be switched from the app's property picker,
3. prints a summary (node, depth, scaffold, compounds, activity range and quartiles) to pick the
   series from, and
4. checks that every CSV loads through :func:`cmfqsar.dataio.load_dataset` and that an MCS
   alignment of its first molecules on the node scaffold matches the whole scaffold.

The compounds are stored in the tree with their protonation charges neutralised, while in 1 M HCl
the basic nitrogens are protonated.  ``--protonate [PH]`` (default pH 0, that of 1 M HCl) sets the
protonation state with Dimorphite-DL (``pip install dimorphite_dl``), whose SMARTS pKa rules decide
site by site.  Taken at face value those rules over-protonate: at pH 0 they charge every ring
nitrogen of a triazole or oxadiazole and both nitrogens of 2-aminobenzothiazole, although the
first protonation makes the second site far less basic.  So the protons added per molecule are
capped (``--max-protons``, default 1): the pH is swept down from 14 towards the target and the
sites are protonated in the order their rule pKa is passed, until the cap (see
:class:`Protonator` for the ties).  The protonated SMILES go into the ``smiles`` column the app
reads and the neutral ones into ``smiles_neutral``, with the net ``charge`` as a further column;
swap the two columns to model the neutral set from the same file.

Usage::

    python examples/scaffold_tree_export.py iron_inhibitor_scaffolds.html [--outdir examples/corrosion]
        [--medium hcl|h2so4|all] [--adjusted] [--min-compounds 15] [--max-depth D]
        [--nodes 157,158] [--protonate [PH]] [--max-protons N] [--check-molecules 5] [--no-check]

The HTML is the file the Artifact tool saves when it reads the page; a bare ``.json`` holding
the same object is accepted too.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

#: medium -> (label, index of the raw MWILA in a compound record, index of the adjusted one)
MEDIA = {"hcl": ("1 M HCl", 7, 8),
         "h2so4": ("H2SO4", 12, 13),
         "all": ("all media pooled", 5, 6)}
#: the CSV column each MWILA variant is written to
MWILA_COLUMNS = {("hcl", False): "mwila_hcl_raw", ("hcl", True): "mwila_hcl_adj",
                 ("h2so4", False): "mwila_h2so4_raw", ("h2so4", True): "mwila_h2so4_adj",
                 ("all", False): "mwila_pooled_raw", ("all", True): "mwila_pooled_adj"}
#: fixed columns, in order; the MWILA variants other than the activity follow them
BASE_COLUMNS = ["smiles", "name", "activity", "mw", "n_papers", "dG_ads", "scaffold", "depth"]
TAIL_COLUMNS = ["leaf_scaffold", "n_rows", "db_id"]
#: added after the tail when the compounds are protonated
PROTONATION_COLUMNS = ["smiles_neutral", "charge"]

# positions in a compound record
C_ID, C_NAME, C_MW, C_NROWS, C_NPAPERS, C_PATH, C_SMILES, C_THERMO = 0, 1, 2, 3, 4, 9, 10, 11

#: typographic characters seen in the compound names, mapped to ASCII before NFKD drops the rest
_NAME_MAP = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-",
                           "—": "-", "−": "-", "′": "'", "‘": "'",
                           "’": "'", "“": '"', "”": '"', "α": "alpha",
                           "β": "beta", "γ": "gamma", "ω": "omega",
                           "Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "€": "(E)"})


# --------------------------------------------------------------------------- #
# the artifact
# --------------------------------------------------------------------------- #

def load_artifact(path: str) -> dict:
    """The tree JSON, from the saved artifact HTML or from a JSON file holding the same object."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if text.lstrip().startswith("{"):
        data = json.loads(text)
    else:
        m = re.search(r'<script[^>]*\bid="data"[^>]*>(.*?)</script>', text, re.S)
        if m is None:
            raise ValueError(f'{path}: no <script id="data"> block with the tree JSON')
        data = json.loads(m.group(1))
    for key in ("nodes", "comps"):
        if key not in data:
            raise ValueError(f"{path}: the tree JSON has no '{key}' entry")
    return data


def mwila(comp: Sequence, medium: str, adjusted: bool) -> Optional[float]:
    """The compound's MWILA for a medium, or None when it was not measured there."""
    _, raw, adj = MEDIA[medium]
    v = comp[adj if adjusted else raw]
    return None if v is None else float(v)


def clean_name(name: str, fallback: str) -> str:
    """A compound name as plain ASCII on one line; `fallback` when nothing printable is left."""
    s = " ".join(str(name).translate(_NAME_MAP).split())
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").strip()
    return s or fallback


def safe_scaffold(smiles: str, limit: int = 40) -> str:
    """A scaffold SMILES as a file-name fragment: runs of non-alphanumerics become one underscore."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", smiles).strip("_")
    return s[:limit].rstrip("_") or "scaffold"


# --------------------------------------------------------------------------- #
# protonation
# --------------------------------------------------------------------------- #

def net_charge(smiles: str) -> int:
    """The net formal charge of a SMILES (a nitro group counts as zero)."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"SMILES does not parse: {smiles}")
    return sum(a.GetFormalCharge() for a in m.GetAtoms())


def _positive_sites(mol) -> List[int]:
    """Atoms carrying a positive charge and at least one hydrogen, i.e. protonated ones."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() > 0 and a.GetTotalNumHs() > 0]


def _skeleton_query(mol):
    """A query matching `mol`'s heavy-atom skeleton whatever the charges, hydrogens and bond orders.

    Every atom becomes ``[#Z]`` and every bond ``~``, so a protonation state of the molecule,
    which may be written in another atom order and with other aromatic bonds, still matches.
    """
    from rdkit import Chem
    smarts = Chem.MolToSmarts(mol)
    atoms = re.findall(r"\[#\d+[^\]]*\]", smarts)
    parts = re.split(r"\[#\d+[^\]]*\]", smarts)
    out = []
    for i, p in enumerate(parts):
        out.append(re.sub(r"[:=#\-/\\]", "~", p))
        if i < len(atoms):
            out.append(re.match(r"\[#\d+", atoms[i]).group(0) + "]")
    return Chem.MolFromSmarts("".join(out))


def _ref_map(state, query, ref) -> Optional[Dict[int, int]]:
    """Atom index of `state` -> atom index of `ref`, through the skeleton query; None if no match."""
    m_ref = ref.GetSubstructMatch(query)
    m_state = state.GetSubstructMatch(query)
    if len(m_ref) != query.GetNumAtoms() or len(m_state) != query.GetNumAtoms():
        return None
    return {s: r for s, r in zip(m_state, m_ref)}


def _deprotonate(mol, atoms: Sequence[int]):
    """A copy of `mol` with a proton taken off each of `atoms`; None if the result does not sanitise."""
    from rdkit import Chem
    m = Chem.RWMol(mol)
    for i in atoms:
        a = m.GetAtomWithIdx(int(i))
        h = a.GetTotalNumHs()
        a.SetFormalCharge(a.GetFormalCharge() - 1)
        a.SetNumExplicitHs(h - 1)
        a.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(m)
    except Exception:                            # noqa: BLE001
        return None
    return m.GetMol()


class Protonator:
    """Protonation states from Dimorphite-DL with a cap on the number of protons added.

    Dimorphite-DL decides each ionisable site on its own, so at a low pH it charges every
    nitrogen whose rule pKa lies above the pH, however many there are in the molecule.  With
    `max_protons` set, the pH is lowered from `ph_top` towards `ph` and the sites are protonated
    in the order their rule pKa is passed, until the cap: the most basic sites get the protons
    and the rest stay neutral.  Sites with the same rule pKa flip together; when that jumps over
    the cap, as many of them as the cap allows keep their proton (the first ones in atom order,
    the only tie-break there is) and the others are deprotonated again.  A proton counts when it
    sits on a positively charged atom, so permanent charges of the input and the deprotonation
    of acids at a higher pH do not enter the count.  `max_protons=None` returns the uncapped
    Dimorphite-DL state.  `unmapped` collects the SMILES whose states could not be matched back
    to the input atoms, which were left at the first state within the cap.
    """

    def __init__(self, ph: float = 0.0, max_protons: Optional[int] = 1, ph_top: float = 14.0,
                 ph_step: float = 0.1):
        try:
            from dimorphite_dl import protonate_smiles
        except ImportError as e:
            raise ImportError("--protonate needs Dimorphite-DL: pip install dimorphite_dl") from e
        self._protonate = protonate_smiles
        self.ph, self.max_protons = float(ph), max_protons
        self.ph_top, self.ph_step = float(ph_top), float(ph_step)
        self._cache: Dict[str, str] = {}
        self.unmapped: List[str] = []

    def state_at(self, smiles: str, ph: float):
        """The molecule in its Dimorphite-DL state at one pH (the first variant if there are several)."""
        from rdkit import Chem
        out = self._protonate(smiles, ph_min=ph, ph_max=ph, precision=0.0)
        m = Chem.MolFromSmiles(out[0]) if out else None
        return Chem.MolFromSmiles(smiles) if m is None else m

    def __call__(self, smiles: str) -> str:
        from rdkit import Chem
        if smiles in self._cache:
            return self._cache[smiles]
        ref = Chem.MolFromSmiles(smiles)
        if ref is None:
            raise ValueError(f"SMILES does not parse: {smiles}")
        # protons added = protonated sites beyond those of the input; the net charge would not do,
        # since an acid losing its proton at a higher pH would hide a base gaining one
        base = len(_positive_sites(ref))
        added = (lambda m: len(_positive_sites(m)) - base)
        state = self.state_at(smiles, self.ph)
        cap = self.max_protons
        if cap is not None and added(state) > cap:
            # protons come off as the pH rises: bisect for the lowest pH that respects the cap
            lo, hi = self.ph, self.ph_top
            lo_state, hi_state = state, self.state_at(smiles, hi)
            if added(hi_state) <= cap:
                while hi - lo > self.ph_step:
                    mid = 0.5 * (lo + hi)
                    s = self.state_at(smiles, mid)
                    if added(s) > cap:
                        lo, lo_state = mid, s
                    else:
                        hi, hi_state = mid, s
                state = hi_state
                # the sites that flip between hi and lo share a rule pKa: keep what the cap allows
                room = cap - added(hi_state)
                if room > 0:
                    query = _skeleton_query(ref)
                    lo_map, hi_map = _ref_map(lo_state, query, ref), _ref_map(hi_state, query, ref)
                    if lo_map is None or hi_map is None:
                        self.unmapped.append(smiles)
                    else:
                        kept = {hi_map[i] for i in _positive_sites(hi_state)}
                        jump = [i for i in _positive_sites(lo_state) if lo_map[i] not in kept]
                        trimmed = _deprotonate(lo_state, jump[room:])
                        if trimmed is not None:
                            state = trimmed
            else:
                state = hi_state
        result = Chem.MolToSmiles(state)
        self._cache[smiles] = result
        return result


# --------------------------------------------------------------------------- #
# selecting the nodes
# --------------------------------------------------------------------------- #

@dataclass
class NodeSeries:
    """A scaffold-tree node and the compounds below it that have the chosen activity."""
    index: int
    depth: int
    scaffold: str
    compounds: List[Sequence] = field(default_factory=list)
    values: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n(self) -> int:
        return len(self.compounds)

    def file_name(self) -> str:
        return f"{self.index}_{safe_scaffold(self.scaffold)}.csv"

    def quartiles(self) -> Dict[str, float]:
        v = self.values
        return {"min": float(v.min()), "q1": float(np.percentile(v, 25)),
                "median": float(np.median(v)), "q3": float(np.percentile(v, 75)),
                "max": float(v.max())}


def node_series(data: dict, index: int, medium: str = "hcl", adjusted: bool = False) -> NodeSeries:
    """The compounds below node `index` whose MWILA in `medium` is known."""
    node = data["nodes"][index]
    comps = [c for c in data["comps"] if index in c[C_PATH] and mwila(c, medium, adjusted) is not None]
    vals = np.array([mwila(c, medium, adjusted) for c in comps], dtype=float)
    return NodeSeries(index, int(node["r"]), node["s"], comps, vals)


def rank_nodes(data: dict, medium: str = "hcl", adjusted: bool = False, min_compounds: int = 15,
               max_depth: Optional[int] = None, nodes: Optional[Sequence[int]] = None) -> List[NodeSeries]:
    """Nodes with at least `min_compounds` measured compounds, most compounds first.

    `nodes` names the nodes explicitly instead; they are still ordered by size but not filtered,
    except that a node with no measured compound at all is dropped.
    """
    if medium not in MEDIA:
        raise ValueError(f"medium must be one of {sorted(MEDIA)}, not {medium!r}")
    chosen = range(len(data["nodes"])) if nodes is None else list(nodes)
    out = []
    for i in chosen:
        node = data["nodes"][i]
        if nodes is None and max_depth is not None and int(node["r"]) > max_depth:
            continue
        s = node_series(data, i, medium, adjusted)
        if nodes is None and s.n < min_compounds:
            continue
        if s.n:
            out.append(s)
    out.sort(key=lambda s: (-s.n, s.index))
    return out


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #

def _fmt(v) -> str:
    return "" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def extra_mwila_columns(series: NodeSeries, medium: str, adjusted: bool) -> List[str]:
    """The MWILA variants, other than the activity, that at least one compound of the node has."""
    cols = []
    for (med, adj), col in MWILA_COLUMNS.items():
        if (med, adj) == (medium, adjusted):
            continue
        if any(mwila(c, med, adj) is not None for c in series.compounds):
            cols.append(col)
    return cols


def write_node_csv(path: str, data: dict, series: NodeSeries, medium: str = "hcl",
                   adjusted: bool = False, protonator: Optional[Protonator] = None) -> int:
    """Write the node's compounds as a CSV the app reads; returns the number of rows.

    With a `protonator` the ``smiles`` column holds the protonated form and the neutral one
    moves to ``smiles_neutral``, with the net ``charge`` beside it.
    """
    nodes = data["nodes"]
    extras = extra_mwila_columns(series, medium, adjusted)
    by_col = {col: key for key, col in MWILA_COLUMNS.items()}
    header = BASE_COLUMNS + extras + TAIL_COLUMNS + (PROTONATION_COLUMNS if protonator else [])
    with open(path, "w", newline="", encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for c in series.compounds:
            thermo = c[C_THERMO] or []
            smiles = c[C_SMILES]
            if protonator is not None:
                smiles = protonator(c[C_SMILES])
            row = {"smiles": smiles, "smiles_neutral": c[C_SMILES],
                   "charge": net_charge(smiles) if protonator else None,
                   "name": clean_name(c[C_NAME], f"cpd_{c[C_ID]}"),
                   "activity": mwila(c, medium, adjusted),
                   "mw": c[C_MW], "n_papers": c[C_NPAPERS],
                   "dG_ads": thermo[0] if thermo else None,
                   "scaffold": series.scaffold, "depth": series.depth,
                   "leaf_scaffold": nodes[c[C_PATH][-1]]["s"],
                   "n_rows": c[C_NROWS], "db_id": c[C_ID]}
            for col in extras:
                med, adj = by_col[col]
                row[col] = mwila(c, med, adj)
            w.writerow([_fmt(row[h]) for h in header])
    return series.n


def export(data: dict, outdir: str, selected: Sequence[NodeSeries], medium: str = "hcl",
           adjusted: bool = False, protonator: Optional[Protonator] = None) -> Dict[int, str]:
    """Write one CSV per selected node into `outdir`; returns node index -> path."""
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    for s in selected:
        p = os.path.join(outdir, s.file_name())
        write_node_csv(p, data, s, medium, adjusted, protonator)
        paths[s.index] = p
    return paths


def summary_table(selected: Sequence[NodeSeries]) -> str:
    """The table the series are picked from: one line per node."""
    lines = [f"{'node':>5} {'depth':>5} {'n':>4} {'min':>6} {'q1':>6} {'median':>6} {'q3':>6} {'max':>6}  scaffold"]
    for s in selected:
        q = s.quartiles()
        lines.append(f"{s.index:>5} {s.depth:>5} {s.n:>4} {q['min']:>6.2f} {q['q1']:>6.2f} "
                     f"{q['median']:>6.2f} {q['q3']:>6.2f} {q['max']:>6.2f}  {s.scaffold}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# checking what was written
# --------------------------------------------------------------------------- #

@dataclass
class CheckResult:
    n_loaded: int = 0
    n_missing_activity: int = 0
    scaffold_atoms: int = 0
    matched: List[int] = field(default_factory=list)      # MCS size per aligned molecule
    rmsd: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def mcs_short(self) -> bool:
        """True when some aligned molecule matched fewer atoms than the scaffold has."""
        return any(m < self.scaffold_atoms for m in self.matched)

    @property
    def ok(self) -> bool:
        return not self.errors and self.n_loaded > 0 and self.n_missing_activity == 0


def check_csv(path: str, scaffold: str, n_align: int = 5, activity_prop: str = "activity") -> CheckResult:
    """Load a written CSV as the app would and align its first molecules on the scaffold."""
    from rdkit import Chem
    from cmfqsar.align import embed_3d, mcs_align
    from cmfqsar.dataio import load_dataset

    res = CheckResult()
    try:
        ds = load_dataset(path, activity_prop=activity_prop)
    except Exception as e:                      # noqa: BLE001 - report, do not stop the run
        res.errors.append(f"load_dataset: {e}")
        return res
    res.n_loaded = len(ds)
    res.n_missing_activity = int(np.sum(~np.isfinite(ds.activity)))
    if activity_prop not in ds.properties:
        res.errors.append(f"column {activity_prop!r} not among the properties {ds.properties}")
    core = Chem.MolFromSmiles(scaffold)
    if core is None:
        res.errors.append(f"scaffold {scaffold!r} does not parse")
        return res
    res.scaffold_atoms = core.GetNumHeavyAtoms()
    try:
        ref = embed_3d(core)
    except Exception as e:                      # noqa: BLE001
        res.errors.append(f"scaffold embedding: {e}")
        return res
    for m in ds.mols[:n_align]:
        try:
            rmsd, n = mcs_align(m, ref)
        except Exception as e:                  # noqa: BLE001
            res.errors.append(f"mcs_align {m.GetProp('_Name')}: {e}")
            continue
        res.matched.append(int(n))
        res.rmsd.append(float(rmsd))
    return res


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #

def _parse_nodes(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    return [int(t) for t in re.split(r"[,\s]+", text.strip()) if t]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("artifact", help="the saved artifact HTML (or its JSON)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "corrosion"))
    ap.add_argument("--medium", choices=sorted(MEDIA), default="hcl",
                    help="which MWILA is the activity: 1 M HCl, H2SO4 or pooled over media")
    ap.add_argument("--adjusted", action="store_true", help="use the MW-adjusted MWILA (default raw)")
    ap.add_argument("--min-compounds", type=int, default=15)
    ap.add_argument("--max-depth", type=int, default=None, help="deepest tree level to consider")
    ap.add_argument("--nodes", default=None, help="node indices to export, e.g. 157,158 (overrides the ranking)")
    ap.add_argument("--protonate", nargs="?", const=0.0, default=None, type=float, metavar="PH",
                    help="protonate with Dimorphite-DL at this pH (default 0, i.e. 1 M HCl); "
                         "the neutral SMILES move to a smiles_neutral column")
    ap.add_argument("--max-protons", type=int, default=1,
                    help="cap on the protons added per molecule; sites are charged in pKa order "
                         "until the cap (default 1); a negative value removes the cap")
    ap.add_argument("--check-molecules", type=int, default=5, help="molecules aligned on the scaffold per CSV")
    ap.add_argument("--no-check", action="store_true", help="skip loading the CSVs back and the MCS check")
    a = ap.parse_args(argv)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    data = load_artifact(a.artifact)
    label = MEDIA[a.medium][0] + (", MW-adjusted" if a.adjusted else ", raw")
    protonator = None
    if a.protonate is not None:
        protonator = Protonator(a.protonate, None if a.max_protons < 0 else a.max_protons)
        label += (f"; protonated at pH {a.protonate:g}"
                  + (f", at most {a.max_protons} proton(s) added" if a.max_protons >= 0 else ", uncapped"))
    selected = rank_nodes(data, a.medium, a.adjusted, a.min_compounds, a.max_depth, _parse_nodes(a.nodes))
    print(f"{len(data['nodes'])} nodes, {len(data['comps'])} compounds (tree of {data.get('date', '?')}); "
          f"activity = MWILA, {label}")
    if not selected:
        print("no node passes the selection")
        return 1
    paths = export(data, a.outdir, selected, a.medium, a.adjusted, protonator)
    print(f"{len(paths)} CSV files written to {a.outdir}\n")
    if protonator is not None:
        charges = [net_charge(protonator(c[C_SMILES])) for s in selected for c in s.compounds]
        counts = {q: charges.count(q) for q in sorted(set(charges))}
        print("net charge of the written compounds: " + ", ".join(f"{q:+d}: {n}" for q, n in counts.items()))
        if protonator.unmapped:
            print(f"{len(protonator.unmapped)} compounds whose protonation states could not be mapped back "
                  "to the input atoms were left at the first state within the cap:")
            for smi in protonator.unmapped:
                print(f"    {smi}")
        print()
    print(summary_table(selected))

    if a.no_check:
        return 0
    print("\ncheck: load through cmfqsar.dataio.load_dataset, MCS-align the first molecules on the scaffold")
    bad = 0
    for s in selected:
        r = check_csv(paths[s.index], s.scaffold, n_align=a.check_molecules)
        flag = "ok" if r.ok and not r.mcs_short else "CHECK"
        bad += flag != "ok"
        mcs = ",".join(str(m) for m in r.matched) or "-"
        print(f"  {flag:5} node {s.index:>4} {os.path.basename(paths[s.index])}: {r.n_loaded} loaded, "
              f"{r.n_missing_activity} without activity, MCS {mcs} of {r.scaffold_atoms} scaffold atoms"
              + (f"; max rmsd {max(r.rmsd):.2f} A" if r.rmsd else ""))
        if r.mcs_short:
            print(f"        MCS smaller than the scaffold: a tautomer, charge or ring-fusion difference "
                  f"between {s.scaffold} and the compounds")
        for e in r.errors:
            print(f"        {e}")
    print(f"\n{len(selected) - bad} of {len(selected)} files pass"
          + ("" if bad else "; every node scaffold is matched in full"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
