"""Protein structures for display behind the molecules, their fields and the co-fields.

The free-energy benchmark collections give their ligands in the coordinate frame of a reference
crystal structure, and every record carries the identifier of that structure, so nothing has to be
superimposed: fetching the entry puts the protein exactly around the training set, and a co-field
computed from those molecules already lies in the same frame.

This module reads a structure (PDB or mmCIF), selects the residues around a set of points (the
binding site), and derives bonds from interatomic distances, which is all the viewer needs.  It is
deliberately independent of RDKit: crystal structures carry altLoc flags, missing elements and
non-standard residues that a chemistry toolkit would refuse to sanitise, and none of that matters
for drawing.

:func:`contact_distance` is the frame check.  Ligands posed in the site sit a few ångström from the
protein; a distance of tens of ångström means the molecules were realigned after the structure was
loaded, or that the structure does not belong to this series.
"""
from __future__ import annotations

import glob
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

WATER_RESIDUES = {"HOH", "WAT", "DOD", "H2O", "SOL"}

# covalent radii in angstrom, enough elements for a protein plus the usual cofactors and ions
COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05,
    "CL": 1.02, "BR": 1.20, "I": 1.39, "B": 0.84, "SE": 1.20, "SI": 1.11,
    "NA": 1.66, "MG": 1.41, "K": 2.03, "CA": 1.76, "ZN": 1.22, "FE": 1.32,
    "MN": 1.39, "CU": 1.32, "NI": 1.24, "CO": 1.26, "MO": 1.54, "CD": 1.44,
}
_DEFAULT_RADIUS = 0.77

# two-letter element symbols that may appear in a PDB atom name
_TWO_LETTER = {"CL", "BR", "ZN", "FE", "MG", "MN", "CU", "NA", "CA", "NI", "CO", "SE", "MO", "CD", "SI"}


def _element_from_name(name: str, resname: str = "") -> str:
    """Element symbol from a PDB atom name when the element columns are missing."""
    raw = name.strip()
    if not raw:
        return "C"
    if raw[0].isdigit():                       # "1HB " and the like: the digit is a branch index
        raw = raw[1:]
    head = raw[:2].upper()
    if head in _TWO_LETTER and resname.strip().upper() not in {"HOH", "WAT"}:
        return head.capitalize()
    return raw[0].upper()


@dataclass
class Protein:
    """Atoms of a structure, with everything needed to draw and to select a site."""
    name: str = ""
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    elements: List[str] = field(default_factory=list)
    atom_names: List[str] = field(default_factory=list)
    resnames: List[str] = field(default_factory=list)
    resids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    icodes: List[str] = field(default_factory=list)
    chains: List[str] = field(default_factory=list)
    hetatm: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    bonds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    path: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    @property
    def n_residues(self) -> int:
        return len(set(self.residue_keys()))

    def residue_keys(self) -> List[Tuple[str, int, str]]:
        return [(c, int(r), i) for c, r, i in zip(self.chains, self.resids, self.icodes)]

    def is_water(self) -> np.ndarray:
        return np.array([r.strip().upper() in WATER_RESIDUES for r in self.resnames], bool)

    def residue_label(self, i: int) -> str:
        ic = self.icodes[i].strip()
        return f"{self.resnames[i].strip()}{self.resids[i]}{ic}{('.' + self.chains[i].strip()) if self.chains[i].strip() else ''}"

    def subset(self, idx: Sequence[int], with_bonds: bool = True) -> "Protein":
        idx = np.asarray(sorted(set(int(i) for i in idx)), dtype=int)
        p = Protein(name=self.name, coords=self.coords[idx],
                    elements=[self.elements[i] for i in idx],
                    atom_names=[self.atom_names[i] for i in idx],
                    resnames=[self.resnames[i] for i in idx],
                    resids=self.resids[idx], icodes=[self.icodes[i] for i in idx],
                    chains=[self.chains[i] for i in idx], hetatm=self.hetatm[idx], path=self.path)
        if with_bonds:
            p.bonds = covalent_bonds(p.coords, p.elements)
        return p


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def read_pdb(path: str, keep_waters: bool = False, keep_hydrogens: bool = True,
             first_model_only: bool = True, with_bonds: bool = True) -> Protein:
    """Read a PDB file.  Alternate locations other than the first are dropped."""
    coords, elements, names, resnames, resids, icodes, chains, het = [], [], [], [], [], [], [], []
    seen_alt = set()
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            rec = line[:6]
            if rec == "ENDMDL" and first_model_only:
                break
            if rec not in ("ATOM  ", "HETATM"):
                continue
            alt = line[16]
            name = line[12:16]
            resname = line[17:20]
            chain = line[21]
            try:
                resid = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            icode = line[26]
            if alt not in (" ", ""):
                key = (chain, resid, icode, name.strip(), resname.strip())
                if key in seen_alt:
                    continue
                seen_alt.add(key)
            el = line[76:78].strip() if len(line) >= 78 and line[76:78].strip() else ""
            el = el.capitalize() if el else _element_from_name(name, resname)
            if not keep_waters and resname.strip().upper() in WATER_RESIDUES:
                continue
            if not keep_hydrogens and el.upper() in ("H", "D"):
                continue
            coords.append((x, y, z))
            elements.append(el)
            names.append(name.strip())
            resnames.append(resname.strip())
            resids.append(resid)
            icodes.append(icode)
            chains.append(chain)
            het.append(rec == "HETATM")
    return _finish(path, coords, elements, names, resnames, resids, icodes, chains, het, with_bonds)


def read_mmcif(path: str, keep_waters: bool = False, keep_hydrogens: bool = True,
               first_model_only: bool = True, with_bonds: bool = True) -> Protein:
    """Read the atom_site loop of an mmCIF file, which is what large entries are served as."""
    cols: Dict[str, int] = {}
    rows: List[List[str]] = []
    in_loop = False
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols, rows = True, {}, []
                cols[s.split(".", 1)[1].strip()] = len(cols)
                continue
            if in_loop:
                if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                    if rows:
                        break
                    continue
                rows.append(s.split())
    if not rows:
        raise ValueError(f"no atom_site records found in {path}")

    def col(row, key, default=""):
        i = cols.get(key)
        return row[i] if i is not None and i < len(row) else default

    coords, elements, names, resnames, resids, icodes, chains, het = [], [], [], [], [], [], [], []
    model0 = None
    for row in rows:
        if first_model_only:
            m = col(row, "pdbx_PDB_model_num", "1")
            if model0 is None:
                model0 = m
            elif m != model0:
                break
        alt = col(row, "label_alt_id", ".")
        if alt not in (".", "?", "", "A"):
            continue
        try:
            x, y, z = float(col(row, "Cartn_x")), float(col(row, "Cartn_y")), float(col(row, "Cartn_z"))
        except ValueError:
            continue
        resname = col(row, "auth_comp_id") or col(row, "label_comp_id")
        el = (col(row, "type_symbol") or "C").capitalize()
        if not keep_waters and resname.strip().upper() in WATER_RESIDUES:
            continue
        if not keep_hydrogens and el.upper() in ("H", "D"):
            continue
        rid = col(row, "auth_seq_id") or col(row, "label_seq_id") or "0"
        try:
            resid = int(rid)
        except ValueError:
            resid = 0
        coords.append((x, y, z))
        elements.append(el)
        names.append(col(row, "auth_atom_id") or col(row, "label_atom_id"))
        resnames.append(resname)
        resids.append(resid)
        icodes.append(col(row, "pdbx_PDB_ins_code", " ").replace("?", " ").replace(".", " ")[:1] or " ")
        chains.append((col(row, "auth_asym_id") or col(row, "label_asym_id") or " ")[:1])
        het.append(col(row, "group_PDB", "ATOM") == "HETATM")
    return _finish(path, coords, elements, names, resnames, resids, icodes, chains, het, with_bonds)


def _finish(path, coords, elements, names, resnames, resids, icodes, chains, het, with_bonds) -> Protein:
    p = Protein(name=os.path.splitext(os.path.basename(path))[0],
                coords=np.array(coords, float).reshape(-1, 3), elements=elements, atom_names=names,
                resnames=resnames, resids=np.array(resids, dtype=int), icodes=icodes, chains=chains,
                hetatm=np.array(het, bool), path=path)
    if p.n_atoms == 0:
        raise ValueError(f"no atoms read from {path}")
    if with_bonds:
        p.bonds = covalent_bonds(p.coords, p.elements)
    return p


#: a ring larger than this is not a monomer unit but the macrocycle that joins them
MAX_MONOMER_RING = 9
#: what a monomer unit is called when the file does not say
MONOMER_RESNAME = "MON"


def monomer_units(mol, max_ring: int = MAX_MONOMER_RING) -> List[int]:
    """Split a molecule into its monomer units, one number per atom (plan item SC3).

    A host that comes as a molecule file has no residues, and the profiles that make a
    complementarity score readable are per residue.  A macrocyclic host does have repeating
    units -- a glucose of a cyclodextrin, a glycoluril of a cucurbituril, a phenol of a
    calixarene -- and they are its small ring systems: rings that share an atom belong to the
    same unit, and a ring larger than `max_ring` is the macrocycle that joins the units rather
    than a unit itself, so it is left out of the grouping.

    Every atom that is in no ring is given to the unit it is bonded to, following the bonds
    outwards, so a hydroxyl goes with its own sugar.  An atom bridging two units -- the
    glycosidic oxygen -- goes to the one reached first, which makes the split deterministic.  A
    molecule with no rings at all comes back as one unit.
    """
    n = mol.GetNumAtoms()
    rings = [r for r in mol.GetRingInfo().AtomRings() if len(r) <= int(max_ring)]
    unit = [-1] * n
    groups: List[set] = []
    for ring in rings:
        here = set(ring)
        merged = [g for g in groups if g & here]
        for g in merged:
            here |= g
            groups.remove(g)
        groups.append(here)
    groups.sort(key=lambda g: min(g))
    for k, g in enumerate(groups):
        for i in g:
            unit[i] = k
    if not groups:
        return [0] * n
    # the atoms outside every unit join the first one they can reach
    frontier = [i for i in range(n) if unit[i] >= 0]
    while frontier:
        nxt = []
        for i in frontier:
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                j = nb.GetIdx()
                if unit[j] < 0:
                    unit[j] = unit[i]
                    nxt.append(j)
        frontier = nxt
    for i in range(n):                       # a disconnected piece: its own unit
        if unit[i] < 0:
            unit[i] = len(groups)
            groups.append({i})
    return unit


def read_molecule_host(path: str, add_hydrogens: bool = True, resname: str = MONOMER_RESNAME,
                       max_ring: int = MAX_MONOMER_RING, name: str = "",
                       keep_hydrogens: bool = True, keep_waters: bool = False,
                       with_bonds: bool = True) -> Protein:
    """A host given as a molecule file, with its monomer units as the residues (plan item SC3).

    `complement` was written for proteins, and a cyclodextrin, cucurbituril or calixarene comes
    as an SDF or a MOL file with no residues at all.  This reads one and names its repeating
    units, so that everything the pocket machinery does per residue -- the interaction profile,
    the contacts, the residue table -- is done per unit instead.

    `keep_hydrogens`, `keep_waters` and `with_bonds` are the arguments the PDB reader takes, so
    that :func:`read_structure` can pass whatever it was given: hydrogens are dropped when asked,
    a molecule file has no waters to drop, and the bonds come from the file rather than from the
    distances.
    """
    from rdkit import Chem
    ext = os.path.splitext(path)[1].lower()
    if ext in (".sdf", ".mol"):
        mol = next(iter(Chem.SDMolSupplier(path, removeHs=False)), None)
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(path, removeHs=False)
    elif ext in (".pdb", ".ent"):
        mol = Chem.MolFromPDBFile(path, removeHs=False)
    else:
        raise ValueError(f"{path}: a host molecule is read from .sdf, .mol, .mol2 or .pdb")
    if mol is None:
        raise ValueError(f"{path}: the molecule could not be read")
    if not keep_hydrogens:
        from rdkit import Chem as _Chem
        mol = _Chem.RemoveHs(mol)
    return host_from_molecule(mol, add_hydrogens=add_hydrogens and keep_hydrogens,
                              resname=resname, max_ring=max_ring,
                              name=name or os.path.splitext(os.path.basename(path))[0],
                              path=path)


def host_from_molecule(mol, add_hydrogens: bool = True, resname: str = MONOMER_RESNAME,
                       max_ring: int = MAX_MONOMER_RING, name: str = "",
                       path: str = "") -> Protein:
    """The same from a molecule already in hand, which is what a script or a test has."""
    from rdkit import Chem
    m = Chem.Mol(mol)
    if m.GetNumConformers() == 0:
        raise ValueError("a host needs a conformation: the pocket is its geometry")
    if add_hydrogens and not any(a.GetAtomicNum() == 1 for a in m.GetAtoms()):
        m = Chem.AddHs(m, addCoords=True)
    unit = monomer_units(m, max_ring=max_ring)
    pos = np.array(m.GetConformer().GetPositions(), float)
    elements = [a.GetSymbol() for a in m.GetAtoms()]
    counts: Dict[int, int] = {}
    names = []
    for i, a in enumerate(m.GetAtoms()):
        counts[unit[i]] = counts.get(unit[i], 0) + 1
        names.append(f"{a.GetSymbol()}{counts[unit[i]]}")
    p = Protein(name=name or (m.GetProp("_Name") if m.HasProp("_Name") else "host"),
                coords=pos, elements=elements, atom_names=names,
                resnames=[resname] * m.GetNumAtoms(),
                resids=np.array([u + 1 for u in unit], dtype=int),
                icodes=[""] * m.GetNumAtoms(), chains=["A"] * m.GetNumAtoms(),
                hetatm=np.zeros(m.GetNumAtoms(), bool), path=path)
    p.bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in m.GetBonds()],
                       dtype=int).reshape(-1, 2)
    return p


def cavity_frame(protein: Protein, max_ring: int = MAX_MONOMER_RING) -> Tuple[np.ndarray, np.ndarray]:
    """(centre, axis) of a macrocyclic host: where a guest goes and which way the cavity points.

    The centre is the centroid of the monomer units' own centroids, which for a ring of units is
    the middle of the ring whether or not the units are the same size; the axis is the direction
    in which the units vary least, that is the smallest principal axis of those centroids.
    """
    keys = sorted(set(int(r) for r in protein.resids))
    middles = np.array([protein.coords[protein.resids == k].mean(axis=0) for k in keys])
    centre = middles.mean(axis=0)
    if len(middles) < 3:
        return centre, np.array([0.0, 0.0, 1.0])
    u, sv, vt = np.linalg.svd(middles - centre, full_matrices=False)
    return centre, vt[-1] / max(float(np.linalg.norm(vt[-1])), 1e-12)


def read_structure(path: str, **kw) -> Protein:
    """Read a structure, choosing the parser by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".sdf", ".mol", ".mol2"):
        return read_molecule_host(path, **kw)
    if ext in (".cif", ".mmcif"):
        return read_mmcif(path, **kw)
    return read_pdb(path, **kw)


def write_pdb(protein: Protein, path: str, remarks: Sequence[str] = ()) -> None:
    """Write a structure as PDB, with free-text REMARK lines for provenance.

    Used to store a trimmed binding site next to a data set, so that the pocket stays with the
    molecules and needs no network access.
    """
    with open(path, "w", newline="\n") as fh:
        for line in remarks:
            for chunk in [line[i:i + 68] for i in range(0, max(len(line), 1), 68)]:
                fh.write(f"REMARK 400 {chunk}\n")
        for i in range(protein.n_atoms):
            rec = "HETATM" if protein.hetatm[i] else "ATOM  "
            name = protein.atom_names[i]
            el = protein.elements[i]
            # atom names of one-letter elements start in column 14, two-letter ones in column 13
            nm = f" {name:<3.3s}" if len(el) == 1 and len(name) < 4 else f"{name:<4.4s}"
            x, y, z = protein.coords[i]
            fh.write(f"{rec}{i + 1:5d} {nm}{' '}{protein.resnames[i]:>3.3s} "
                     f"{(protein.chains[i] or ' '):1.1s}{int(protein.resids[i]):4d}"
                     f"{(protein.icodes[i] or ' '):1.1s}   "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el.upper():>2.2s}\n")
        fh.write("END\n")


def find_pocket(dataset_path: str, code: Optional[str] = None) -> Optional[str]:
    """Locate a stored binding site that belongs to a data set.

    Searched in order: a structure beside the file, named after it; a ``pockets`` folder beside
    it; and a ``pockets`` folder further up the tree, which is where
    ``examples/make_pocket_examples.py`` writes them, by file name and then by PDB identifier.
    """
    if not dataset_path:
        return None
    folder = os.path.dirname(os.path.abspath(dataset_path))
    stem = os.path.splitext(os.path.basename(dataset_path))[0]
    names = [stem + ".pdb", stem + "_pocket.pdb", stem + ".cif"]
    if code:
        names += [code.upper() + ".pdb", code.upper() + ".cif"]
    for n in names:
        for d in (folder, os.path.join(folder, "pockets")):
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    up = folder
    for _ in range(3):
        up = os.path.dirname(up)
        if not up:
            break
        root = os.path.join(up, "pockets")
        if not os.path.isdir(root):
            continue
        rel = os.path.relpath(folder, up)        # the generator mirrors the data set's own folders
        for n in names:
            mirrored = os.path.join(root, rel, n)
            if os.path.isfile(mirrored):
                return mirrored
        for n in names:
            hits = sorted(glob.glob(os.path.join(root, "**", n), recursive=True))
            if hits:
                return hits[0]
    return None


def cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cmfqsar", "pdb")


def fetch_pdb(code: str, dest_dir: Optional[str] = None, timeout: float = 30.0,
              overwrite: bool = False) -> str:
    """Download an entry from RCSB and return the local path; cached between runs.

    The PDB format is tried first and mmCIF second, because very large entries are not
    distributed in the PDB format at all.
    """
    code = code.strip().upper()
    if not code or len(code) not in (4, 8) or not code[0].isalnum():
        raise ValueError(f"'{code}' is not a PDB identifier")
    dest_dir = dest_dir or cache_dir()
    os.makedirs(dest_dir, exist_ok=True)
    for ext in (".pdb", ".cif"):
        dest = os.path.join(dest_dir, code + ext)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0 and not overwrite:
            return dest
    last = None
    for ext in (".pdb", ".cif"):
        url = f"https://files.rcsb.org/download/{code}{ext}"
        dest = os.path.join(dest_dir, code + ext)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as fh:
                fh.write(r.read())
            if os.path.getsize(dest) > 0:
                return dest
        except Exception as exc:                  # noqa: BLE001 - reported to the user as text
            last = exc
            if os.path.isfile(dest):
                os.remove(dest)
    raise RuntimeError(f"could not download {code} from RCSB: {last}")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def covalent_bonds(coords: np.ndarray, elements: Sequence[str], tolerance: float = 0.45,
                   max_bond: float = 2.2) -> np.ndarray:
    """Bonds from interatomic distances: ``|r_ij| < R_i + R_j + tolerance``."""
    n = len(coords)
    if n < 2:
        return np.zeros((0, 2), dtype=int)
    radii = np.array([COVALENT_RADII.get(e.upper(), _DEFAULT_RADIUS) for e in elements])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=min(max_bond, 2.0 * radii.max() + tolerance), output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=int)
    d = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    lim = radii[pairs[:, 0]] + radii[pairs[:, 1]] + tolerance
    is_h = np.array([e.upper() in ("H", "D") for e in elements])
    both_h = is_h[pairs[:, 0]] & is_h[pairs[:, 1]]
    keep = (d < lim) & (d > 0.3) & ~both_h
    return pairs[keep].astype(int)


def residue_anchors(protein: Protein) -> Tuple[np.ndarray, List[str]]:
    """One labelling point and text per residue: the alpha carbon, or the centroid without one.

    Residues are returned in the order they appear, so a label list reads along the chain.
    """
    order: List[Tuple[str, int, str]] = []
    members: Dict[Tuple[str, int, str], List[int]] = {}
    for i, k in enumerate(protein.residue_keys()):
        if k not in members:
            members[k] = []
            order.append(k)
        members[k].append(i)
    one_chain = len({c.strip() for c in protein.chains}) <= 1      # then the chain is already known
    pos, txt = [], []
    for k in order:
        idx = members[k]
        ca = [i for i in idx if protein.atom_names[i].strip().upper() == "CA"
              and protein.elements[i].upper() == "C"]
        pos.append(protein.coords[ca[0]] if ca else protein.coords[idx].mean(axis=0))
        label = protein.residue_label(idx[0])
        txt.append(label.split(".")[0] if one_chain else label)
    return np.array(pos, float).reshape(-1, 3), txt


def contact_distance(protein: Protein, points: np.ndarray, polymer_only: bool = True) -> float:
    """Smallest distance between any of `points` and the structure.

    This is the frame check: ligands posed in the site give a few ångström, while a value of
    tens of ångström means the molecules and the structure are not in the same frame.  By
    default only polymer atoms count, because a co-crystallised ligand often *is* one of the
    molecules of the data set and would give a distance of zero.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    if protein.n_atoms == 0 or len(pts) == 0:
        return float("inf")
    coords = protein.coords
    if polymer_only and (~protein.hetatm).any():
        coords = coords[~protein.hetatm]
    return float(cKDTree(coords).query(pts, k=1)[0].min())


def binding_site(protein: Protein, points: np.ndarray, cutoff: float = 6.0,
                 whole_residues: bool = True, keep_waters: bool = False,
                 drop_ligand: bool = True, ligand_overlap: float = 2.0) -> Protein:
    """The part of the structure around `points`, as whole residues by default.

    `drop_ligand` removes the co-crystallised ligand, recognised as a non-water HETATM residue of
    more than one atom that *overlaps* the points, meaning it comes within `ligand_overlap` of
    them.  That is what the crystallographic ligand does when the data set contains it or a close
    analogue.  Ions and cofactors that line the site without occupying the same space are kept.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    if protein.n_atoms == 0 or len(pts) == 0:
        return protein.subset([])
    hits = cKDTree(protein.coords).query_ball_point(pts, r=cutoff)
    flat = [int(i) for h in hits for i in h]
    near = np.unique(np.asarray(flat, dtype=int)) if flat else np.zeros(0, dtype=int)
    if whole_residues and len(near):
        keys = protein.residue_keys()
        wanted = {keys[i] for i in near}
        near = np.array([i for i, k in enumerate(keys) if k in wanted], dtype=int)
    if len(near) == 0:
        return protein.subset([])
    water = protein.is_water()
    drop = np.zeros(protein.n_atoms, bool)
    if not keep_waters:
        drop |= water
    if drop_ligand:
        keys = protein.residue_keys()
        size: Dict[Tuple[str, int, str], int] = {}
        for k in keys:
            size[k] = size.get(k, 0) + 1
        by_res: Dict[Tuple[str, int, str], List[int]] = {}
        for i in near:
            by_res.setdefault(keys[i], []).append(i)
        ptree = cKDTree(pts)
        for k, idx in by_res.items():
            if not all(protein.hetatm[i] for i in idx) or water[idx[0]] or size[k] <= 1:
                continue                      # polymer, water or a single-atom ion: keep
            if ptree.query(protein.coords[idx], k=1)[0].min() <= ligand_overlap:
                drop[idx] = True              # occupies the same space as the molecules
    near = np.array([i for i in near if not drop[i]], dtype=int)
    return protein.subset(near)
