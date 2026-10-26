"""Topomer fields: a frame that comes from the topology, not from a superposition.

Everything else here needs the molecules to share a frame, and gets it by superimposing them --
which works for a congeneric series and silently stops meaning anything when the substituents are
unrelated, as the chemotype measurements show.  Cramer's topomer idea (topomer CoMFA, 2003)
removes the search altogether: cut the molecule at the bond that attaches a substituent, and
build that fragment into a pose that is a *deterministic function of its topology* -- standard
geometry, every rotatable torsion at a fixed value, the attachment atom at the origin and its
open valence along a fixed axis, the remaining roll fixed by a canonical rule.  The same R-group
then always ends in the same place, whatever molecule it came from and however its atoms were
numbered, so the fields of different substituents at one attachment point can be compared with no
alignment at all.

The pose is a convention, not a bioactive conformation, and the method is clear about that: it
compares comparable things at one attachment point, and says nothing about how the ligand really
sits.  What it gains is that the comparison is reproducible and instant.

Two things make this fit CMF better than it fits CoMFA.  There is no grid: a fragment's field is
its atoms and weights, and a fragment-fragment kernel is an analytic number.  And the fields are
additive, so a molecule's kernel with another decomposes into scaffold and substituent terms that
can be tabulated once per fragment and reused across a library -- which is the topomer-search
technique, obtained here without approximation.

The pose rule, in order:

1. the fragment is renumbered canonically, so two copies of the same R-group that were cut out of
   different molecules have the same atom order and therefore the same everything after it;
2. one conformation is embedded with a fixed seed and every rotatable torsion is then set to
   `torsion` degrees (180 by default, Cramer's anti convention), which replaces whatever the
   embedding happened to produce;
3. the attachment atom goes to the origin and the bond to its open valence along -x, so the
   fragment body extends towards +x;
4. the roll about x is fixed by the principal axis of the heavy atoms in the yz-plane, with the
   sign chosen by the lowest-ranked off-axis atom, and left alone when the fragment is too small
   or too symmetric for that to mean anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdFMCS, rdMolTransforms

from .fields import FieldSet, MoleculeFields, build_fields

#: a single acyclic bond between two non-terminal heavy atoms: where a substituent is cut off
ROTATABLE = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


@dataclass(frozen=True)
class TopomerSpec:
    """How a fragment is built into its topomer pose."""
    torsion: float = 180.0       # every rotatable torsion is set to this
    seed: int = 0xC0FFEE         # the embedding seed; fixed, so the geometry is reproducible
    optimise: bool = False       # relax after setting the torsions (off: the rule stays the rule)
    roll: bool = True            # fix the rotation about the attachment axis
    spacing: float = 100.0       # how far apart the attachment sites are placed, in angstrom


def canonical(frag: Chem.Mol) -> Chem.Mol:
    """The same fragment renumbered the same way, whatever molecule it was cut out of."""
    order = list(Chem.CanonicalRankAtoms(frag, breakTies=True))
    return Chem.RenumberAtoms(frag, [int(i) for i in np.argsort(order)])


def _attachment(frag: Chem.Mol) -> Tuple[int, int]:
    """(dummy atom, the atom it is attached to) -- the open valence of the fragment."""
    dummies = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        raise ValueError(f"a topomer fragment needs exactly one attachment point, found "
                         f"{len(dummies)}")
    d = dummies[0]
    nbrs = [n.GetIdx() for n in frag.GetAtomWithIdx(d).GetNeighbors()]
    if not nbrs:
        raise ValueError("the attachment point is not bonded to anything")
    return d, nbrs[0]


def _set_torsions(mol: Chem.Mol, spec: TopomerSpec) -> None:
    """Put every rotatable torsion at the convention angle, in canonical order."""
    conf = mol.GetConformer()
    for j, k in mol.GetSubstructMatches(ROTATABLE):
        a = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        d = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
        if not a or not d:
            continue
        try:
            rdMolTransforms.SetDihedralDeg(conf, min(a), j, k, min(d), float(spec.torsion))
        except Exception:                        # noqa: BLE001 - a torsion that cannot be set
            continue


def _frame(mol: Chem.Mol, dummy: int, root: int, spec: TopomerSpec) -> np.ndarray:
    """Coordinates in the topomer frame: root at the origin, open valence along -x."""
    xyz = np.array(mol.GetConformer().GetPositions(), float)
    xyz = xyz - xyz[root]
    v = xyz[dummy]
    n = np.linalg.norm(v)
    if n > 1e-9:
        v = v / n
        target = np.array([-1.0, 0.0, 0.0])
        axis = np.cross(v, target)
        s, c = np.linalg.norm(axis), float(np.dot(v, target))
        if s > 1e-9:
            axis = axis / s
            K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
            R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
            xyz = xyz @ R.T
        elif c < 0:                               # already antiparallel: turn it round
            xyz = xyz @ np.diag([-1.0, -1.0, 1.0]).T
    if not spec.roll:
        return xyz
    heavy = [i for i in range(mol.GetNumAtoms())
             if mol.GetAtomWithIdx(i).GetAtomicNum() > 1 and i not in (dummy, root)]
    if len(heavy) < 2:
        return xyz
    yz = xyz[heavy][:, 1:]
    if float(np.abs(yz).max()) < 1e-6:
        return xyz                                # everything on the axis: no roll to fix
    w, vec = np.linalg.eigh(yz.T @ yz)
    main = vec[:, int(np.argmax(w))]
    angle = float(np.arctan2(main[1], main[0]))   # bring the principal axis onto +y
    ca, sa = np.cos(np.pi / 2 - angle), np.sin(np.pi / 2 - angle)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    xyz = xyz @ rot.T
    ranked = sorted(heavy)                        # canonical order, so the sign is reproducible
    for i in ranked:
        if abs(xyz[i, 1]) > 1e-6:
            if xyz[i, 1] < 0:
                xyz = xyz @ np.diag([1.0, -1.0, -1.0]).T    # a rotation, so chirality survives
            break
    return xyz


def topomer_pose(frag: Chem.Mol, spec: TopomerSpec = TopomerSpec()) -> Chem.Mol:
    """Build one fragment into its topomer pose.  The result depends only on its topology."""
    work = canonical(Chem.Mol(frag))
    try:
        Chem.SanitizeMol(work)          # renumbering drops the ring information
    except Exception:                   # noqa: BLE001 - a fragment that will not sanitise
        Chem.FastFindRings(work)
    work = Chem.AddHs(work)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(spec.seed)
    # the open valence is a dummy atom and no force field has a type for one, so embedding and
    # optimising a fragment prints a warning per atom; it says nothing a caller can act on, and
    # a design loop poses thousands of fragments
    with rdBase.BlockLogs():
        if AllChem.EmbedMolecule(work, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(work, params) != 0:
                raise ValueError(f"cannot embed the fragment {Chem.MolToSmiles(frag)}")
        if spec.optimise:
            try:
                AllChem.MMFFOptimizeMolecule(work, maxIters=500)
                Chem.SanitizeMol(work)
            except Exception:                    # noqa: BLE001
                pass
    _set_torsions(work, spec)
    dummy, root = _attachment(work)
    xyz = _frame(work, dummy, root, spec)
    conf = work.GetConformer()
    for i in range(work.GetNumAtoms()):
        conf.SetAtomPosition(i, [float(v) for v in xyz[i]])
    work.SetProp("_Name", Chem.MolToSmiles(frag))
    return work


# --------------------------------------------------------------------------- #
# Cutting a series into a core and its substituents
# --------------------------------------------------------------------------- #

def framework_core(mols: Sequence[Chem.Mol], min_atoms: int = 3) -> Optional[Chem.Mol]:
    """The scaffold to cut a topomer series from: the framework of its largest chemotype.

    :func:`common_core` takes the maximum common substructure of everything handed to it, which is
    the right answer for a congeneric series and the wrong one for anything else.  A data set that
    mixes chemotypes has an MCS that is some small fragment -- ``CCO`` on the GPB set -- fused into
    every ring system, so it cuts nothing off and the topomer route loses every molecule.  The
    measurements in ``benchmarks/run_topomer.py`` are what made this the default.

    What a topomer model actually needs is the scaffold of *one* series, so this takes the
    Bemis-Murcko framework shared by the largest group of molecules.  Combine it with
    :func:`series_indices`, which collects every molecule that framework can be cut off cleanly --
    including molecules whose own framework is larger, because an extra ring on a substituent is
    part of the R-group and not of the scaffold.
    """
    from .chemotypes import chemotype_labels      # local: chemotypes is the heavier module
    good = [m for m in mols if m is not None]
    if len(good) < 2:
        return None
    labels, scaffolds = chemotype_labels(good)
    sizes = np.bincount(labels, minlength=len(scaffolds) + 1)
    for g in np.argsort(-sizes):
        smiles = scaffolds[g] if g < len(scaffolds) else ""
        if not smiles or smiles == "(acyclic)":
            continue                                # a chain is not a scaffold to cut from
        core = Chem.MolFromSmiles(smiles)
        if core is not None and core.GetNumAtoms() >= min_atoms:
            return core
    return None


def series_indices(mols: Sequence[Chem.Mol], core: Chem.Mol) -> List[int]:
    """Which molecules of a set this core can be cut off cleanly, i.e. which form the series."""
    return [i for i, m in enumerate(mols) if m is not None and cuttable(m, core)]


def choose_core(mols: Sequence[Chem.Mol], mode: str = "framework",
                timeout: int = 20) -> Tuple[Optional[Chem.Mol], str]:
    """(core, how it was found) for `mode` in {"framework", "mcs"}, with a fallback either way."""
    if mode == "mcs":
        core = common_core(mols, timeout=timeout)
        if core is not None:
            return core, "maximum common substructure"
        core = framework_core(mols)
        return core, "framework of the largest chemotype (no common substructure)"
    core = framework_core(mols)
    if core is not None:
        return core, "framework of the largest chemotype"
    core = common_core(mols, timeout=timeout)
    return core, "maximum common substructure (no ring framework)"

def common_core(mols: Sequence[Chem.Mol], timeout: int = 20,
                threshold: float = 1.0) -> Optional[Chem.Mol]:
    """The maximum common substructure of a series, which is what the R-groups are attached to."""
    # the loader gives every molecule explicit hydrogens; an MCS found over those carries [H]
    # atoms in its SMARTS and then fails to match molecules whose hydrogen count differs, so the
    # core is looked for on hydrogen-suppressed copies
    good = [Chem.RemoveHs(Chem.Mol(m)) for m in mols if m is not None]
    if len(good) < 2:
        return None
    res = rdFMCS.FindMCS(good, timeout=timeout, threshold=threshold,
                         ringMatchesRingOnly=True, completeRingsOnly=True,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrder)
    if res.canceled or not res.smartsString:
        return None
    return Chem.MolFromSmarts(res.smartsString)


def substituents(mol: Chem.Mol, core: Chem.Mol) -> List[Chem.Mol]:
    """The R-groups of one molecule, each carrying its attachment point."""
    rest = Chem.ReplaceCore(mol, core, labelByIndex=True)
    if rest is None:
        return []
    out = []
    for frag in Chem.GetMolFrags(rest, asMols=True, sanitizeFrags=False):
        try:
            Chem.SanitizeMol(frag)
        except Exception:                        # noqa: BLE001 - keep what can be kept
            continue
        if any(a.GetAtomicNum() == 0 for a in frag.GetAtoms()):
            out.append(frag)
    return out


def largest_substituent(mol: Chem.Mol, core: Chem.Mol) -> Optional[Chem.Mol]:
    """The R-group with the most heavy atoms: the one a single-cut topomer model uses."""
    subs = substituents(mol, core)
    if not subs:
        return None
    return max(subs, key=lambda f: sum(1 for a in f.GetAtoms() if a.GetAtomicNum() > 1))


def cuttable(mol: Chem.Mol, core: Chem.Mol) -> bool:
    """True when the core can be cut off this molecule at acyclic single bonds alone.

    A core fused to the rest of the molecule -- a benzene ring picked out of a bicyclic system,
    say -- cannot be removed without opening a ring, and whatever is left is not a substituent.
    The topomer construction needs cuts of the kind a chemist would call an R-group.
    """
    match = mol.GetSubstructMatch(core)
    if not match:
        return False
    inside = set(match)
    for idx in match:
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            other = bond.GetOtherAtomIdx(idx)
            if other in inside or mol.GetAtomWithIdx(other).GetAtomicNum() == 1:
                continue
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                return False
    return True


def cuttable_fraction(mols: Sequence[Chem.Mol], core: Chem.Mol) -> float:
    """How much of a series the core can be cut out of cleanly."""
    good = [m for m in mols if m is not None]
    if not good:
        return 0.0
    return sum(1 for m in good if cuttable(m, core)) / float(len(good))


def attachment_label(frag: Chem.Mol) -> int:
    """Which core atom this substituent is attached to, as :func:`substituents` labels it."""
    for a in frag.GetAtoms():
        if a.GetAtomicNum() == 0:
            return int(a.GetIsotope())
    return -1


def series_labels(mols: Sequence[Chem.Mol], core: Chem.Mol) -> List[int]:
    """The attachment points the series uses, in a fixed order."""
    seen = set()
    for mol in mols:
        if mol is None:
            continue
        for frag in substituents(mol, core):
            seen.add(attachment_label(frag))
    return sorted(seen)


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #

#: default distance between attachment sites, in angstrom.  Each substituent keeps its own
#: topomer frame, and the frames are set so far from one another that no Gaussian of one reaches
#: the other: the overlap integral of two molecules is then the sum of the overlaps of their
#: substituents, attachment point by attachment point, which is what topomer CoMFA gets by
#: giving every position its own block of columns.
SITE_SPACING = 100.0


def topomer_molecule(mol: Chem.Mol, core: Chem.Mol, labels: Sequence[int],
                     spec: TopomerSpec = TopomerSpec(),
                     cache: Optional[Dict[str, Chem.Mol]] = None) -> Optional[Chem.Mol]:
    """Every substituent of `mol` in its own topomer pose, each at its own attachment site.

    The result is one molecule for the field machinery to work on, but it is not a molecule any
    chemist would draw: it is the series' substituents, each built by the rule and placed in a
    region of space reserved for its attachment point.  A position a molecule does not substitute
    simply contributes nothing there.
    """
    return topomer_assembly(substituents(mol, core), labels, spec, cache)


def topomer_assembly(frags: Sequence[Chem.Mol], labels: Sequence[int],
                     spec: TopomerSpec = TopomerSpec(),
                     cache: Optional[Dict[str, Chem.Mol]] = None) -> Optional[Chem.Mol]:
    """The same construction from fragments that were not cut off anything.

    :func:`topomer_molecule` cuts a molecule up and calls this; a design loop builds its
    candidates by replacing one fragment of that list and calls it directly, so that a molecule
    nobody has made is posed by exactly the rule that posed the ones that were.
    """
    cache = {} if cache is None else cache
    pieces = []
    for frag in frags:
        label = attachment_label(frag)
        if label not in labels:
            continue
        key = Chem.MolToSmiles(frag)
        posed = cache.get(key)
        if posed is None:
            posed = topomer_pose(frag, spec)
            cache[key] = posed
        here = Chem.Mol(posed)
        conf = here.GetConformer()
        shift = float(spec.spacing) * float(list(labels).index(label))
        for i in range(here.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, [p.x + shift, p.y, p.z])
        pieces.append(here)
    if not pieces:
        return None
    out = pieces[0]
    for piece in pieces[1:]:
        out = Chem.CombineMols(out, piece)
    return out


def topomer_fields(mols: Sequence[Chem.Mol], names: Sequence[str], fieldset: FieldSet,
                   core: Optional[Chem.Mol] = None, spec: TopomerSpec = TopomerSpec(),
                   labels: Optional[Sequence[int]] = None,
                   progress=None) -> Tuple[List[Optional[MoleculeFields]], List[str]]:
    """Topomer fields of a series: every substituent posed by the rule, at its own site.

    Returns (fields, log).  A molecule whose substituents cannot be built gets ``None``, so the
    caller can drop it and say which ones were lost.  Identical substituents at the same attachment
    point give identical fields, whatever molecule they came from, which is what makes the
    comparison free of any alignment.
    """
    core = core if core is not None else common_core(mols)
    if core is None:
        raise ValueError("no common substructure: a topomer model needs a core to cut from")
    labels = list(labels) if labels is not None else series_labels(mols, core)
    share = cuttable_fraction(mols, core)
    cache: Dict[str, Chem.Mol] = {}
    log: List[str] = []
    if share < 0.9:
        log.append(f"the core comes off cleanly in only {share:.0%} of the molecules: it is "
                   f"fused to the rest or attached by something other than a single bond, and "
                   f"what is left of those molecules is not a substituent. Give a core that is "
                   f"joined to its R-groups by acyclic single bonds.")
    out: List[Optional[MoleculeFields]] = []
    for i, (mol, name) in enumerate(zip(mols, names)):
        if progress is not None and i % 10 == 0:
            progress(f"Topomer fields {i + 1}/{len(mols)}", i / max(len(mols), 1))
        try:
            built = topomer_molecule(mol, core, labels, spec, cache) if mol is not None else None
        except Exception as exc:                 # noqa: BLE001 - one bad fragment, not the set
            built, exc_text = None, str(exc)
            log.append(f"{name}: could not be posed ({exc_text})")
        if built is None:
            out.append(None)
            if not log or not log[-1].startswith(f"{name}:"):
                log.append(f"{name}: nothing outside the core to pose")
            continue
        out.append(build_fields(built, fieldset, name))
    return out, log


def unique_substituents(mols: Sequence[Chem.Mol], core: Optional[Chem.Mol] = None) -> List[str]:
    """The distinct R-groups of a series, as SMILES -- the library a topomer model sees."""
    core = core if core is not None else common_core(mols)
    if core is None:
        return []
    seen = []
    for mol in mols:
        frag = largest_substituent(mol, core) if mol is not None else None
        if frag is None:
            continue
        smi = Chem.MolToSmiles(frag)
        if smi not in seen:
            seen.append(smi)
    return seen
