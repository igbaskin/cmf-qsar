"""
Solvation shells cut from a bulk electrolyte trajectory (plan item BT5).

Item BT4 builds one cation complex and optimises it: a single geometry, the lowest the builder
could find. A real electrolyte is not that. At a given moment one lithium ion sits in a shell of
four carbonate molecules, the next in a shell of three and an anion, and a moment later they have
exchanged. The quantities an electrolyte is judged by -- conductivity, the transference number,
the diffusion coefficient -- are averages over that distribution, so a descriptor that is one
geometry may be describing the wrong object.

This module takes the other route. It reads a bulk simulation through :mod:`cmfqsar.trajectory`,
cuts out every cation with the molecules that touch it, and hands the collection to the ensemble
builder of plan item D4 (:func:`cmfqsar.fields.combine_members`) as members with their own atoms.
The result is one field per electrolyte, the mean field of its solvation shells, which the model
then uses exactly as it uses the field of a molecule. Nothing in the kernel changes: a CMF field
is a sum of atom-centred Gaussians and the kernel is bilinear in the field, so the mean field of
a distribution of shells is itself a field.

What the three steps are, and why each is done this way.

**The cut.** A molecule belongs to the shell when one of its donor atoms -- an oxygen, nitrogen,
fluorine, sulfur or chlorine -- lies within the coordination distance of the ion. The distance is
the one item BT4 measured on relaxed geometries, ``ION_OXYGEN_DISTANCE`` times
:data:`DEFAULT_SHELL_FACTOR`, so a shell here and a coordination number there mean the same thing.
An anion that comes that close is kept like any other molecule, because a contact ion pair is a
real state of the electrolyte and leaving it out would bias the ensemble towards free ions.

**The periodic boundary.** A simulation is written with every atom wrapped into the cell, so a
molecule that straddles a face arrives in two pieces a box length apart, and a shell beside a
cation at the edge of the cell arrives on the other side of the box. Both faults are silent: they
produce a geometry, not an error. Item D7 met them already and :func:`cmfqsar.trajectory.make_whole`
and :func:`cmfqsar.trajectory.reimage` are the repairs; this module calls them for every molecule
of every shell before anything is measured.

**The thinning.** A two-nanosecond box with fifteen cations and two hundred frames gives three
thousand shells, and the kernel costs the square of the atoms kept, so the ensemble has to be
reduced to a few dozen members. The reduction is done in two stages. Shells are grouped by
composition first -- which species, how many of each -- because the composition is what a model
of an electrolyte can act on, and the groups are then thinned to representatives in proportion to
their population. Inside a group the thinning uses a measure that does not depend on how the
molecules happen to be numbered: the sorted distances from the ion to its donors and the sorted
distances between donors. A field is a sum over atoms and so is already blind to that numbering,
which is why no superposition is attempted here.

**The error of the mean.** An ensemble read from a simulation carries sampling error, and
unconverged sampling error looks like chemistry. Item D8 requires that it be reported, so
:func:`block_error` splits the frames into blocks, builds the mean field of each block on its own,
and returns the standard error of the mean over the blocks as a share of the field's own size. A
value near zero says the frames were enough; a large one says the difference between two
electrolytes may be sampling noise.
"""

from dataclasses import dataclass, field as _field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from .fields import AtomTable, FieldSet, MoleculeFields, build_fields, combine_members
from .kernels import kernel_matrices, self_kernels
from .solvation import ION_CHARGE, ION_OXYGEN_DISTANCE, with_frame
from .trajectory import Trajectory, make_whole, reimage

#: elements counted as donors when deciding what touches the ion
DONOR_ELEMENTS: Tuple[str, ...] = ("O", "N", "F", "S", "Cl")

#: the coordination distance of item BT4, times this, is the shell cutoff
DEFAULT_SHELL_FACTOR = 1.3

#: representatives kept by default; the kernel cost is the square of the atoms kept
DEFAULT_MAX_MEMBERS = 16

#: leader-clustering radius inside one composition group, in angstrom
DEFAULT_CLUSTER_RADIUS = 0.75


class ShellError(RuntimeError):
    """A shell could not be cut, or could not be turned into a molecule."""


# ----------------------------------------------------------------------------------------------
# one shell
# ----------------------------------------------------------------------------------------------

@dataclass
class Shell:
    """One cation with the molecules that touch it, in one frame of one trajectory.

    `atoms` indexes the source trajectory and starts with the ion. `coords` holds the same atoms
    in the same order, repaired across the periodic boundary and moved to the image beside the
    ion. `species` names the residue of each molecule, in the order the molecules appear after
    the ion, and `sizes` says how many atoms each of them has.
    """

    frame: int
    ion_atom: int
    ion: str
    atoms: np.ndarray
    coords: np.ndarray
    species: Tuple[str, ...] = ()
    sizes: Tuple[int, ...] = ()
    donor_distances: Tuple[float, ...] = ()
    note: str = ""

    @property
    def n_molecules(self) -> int:
        return len(self.species)

    @property
    def n_atoms(self) -> int:
        return int(len(self.atoms))

    @property
    def composition(self) -> Tuple[Tuple[str, int], ...]:
        """Which species are in the shell and how many of each, in a fixed order."""
        counts: Dict[str, int] = {}
        for s in self.species:
            counts[s] = counts.get(s, 0) + 1
        return tuple(sorted(counts.items()))

    @property
    def coordination(self) -> int:
        """Donor atoms inside the cutoff, which is the coordination number of item BT4."""
        return len(self.donor_distances)

    def molecule_slice(self, i: int) -> slice:
        """Where molecule `i` sits in `atoms` and `coords`; the ion is atom 0."""
        start = 1 + int(sum(self.sizes[:i]))
        return slice(start, start + int(self.sizes[i]))

    def fingerprint(self) -> np.ndarray:
        """A description of the shape that does not depend on how the atoms are numbered.

        The sorted ion-to-donor distances followed by the sorted donor-to-donor distances. Two
        shells of the same composition whose fingerprints are close are the same arrangement,
        whatever order their molecules were written in. A field is a sum over atoms and is blind
        to that order in the same way, which is why this and not a superposition is what the
        thinning uses.
        """
        d = np.sort(np.asarray(self.donor_distances, float))
        if len(d) < 2:
            return d
        donors = self._donor_coords()
        if len(donors) < 2:
            return d
        dd = np.linalg.norm(donors[:, None, :] - donors[None, :, :], axis=-1)
        iu = np.triu_indices(len(donors), k=1)
        return np.concatenate([d, np.sort(dd[iu])])

    def _donor_coords(self) -> np.ndarray:
        """The coordinates of the donor atoms, in the order `donor_distances` has them."""
        keep = getattr(self, "_donors", None)
        if keep is None:
            return np.zeros((0, 3))
        return self.coords[np.asarray(keep, int)]


def ion_atoms(traj: Trajectory, ion: str = "Li") -> np.ndarray:
    """Indices of every atom of element `ion` in the trajectory."""
    want = str(ion).strip().capitalize()
    idx = [i for i, e in enumerate(traj.elements) if str(e).strip().capitalize() == want]
    return np.asarray(idx, int)


def residue_groups(traj: Trajectory) -> List[np.ndarray]:
    """The atoms of each residue, in the order the residues first appear.

    A molecule in a bulk simulation is one residue, which is how every packer and every topology
    writer produces it; the groups are what the shell cut moves and repairs as single pieces.
    """
    order: List[int] = []
    seen: Dict[int, List[int]] = {}
    for i, r in enumerate(np.asarray(traj.resids, int).tolist()):
        if r not in seen:
            seen[r] = []
            order.append(r)
        seen[r].append(i)
    return [np.asarray(seen[r], int) for r in order]


def shell_cutoff(ion: str = "Li", factor: float = DEFAULT_SHELL_FACTOR) -> float:
    """The shell cutoff in angstrom: the coordination distance of item BT4 times `factor`."""
    key = str(ion).strip().capitalize()
    if key not in ION_OXYGEN_DISTANCE:
        raise ShellError(f"unknown ion {ion!r}; this module knows "
                         f"{', '.join(sorted(ION_OXYGEN_DISTANCE))}")
    return float(ION_OXYGEN_DISTANCE[key]) * float(factor)


def shell_at(traj: Trajectory, frame: int, ion_atom: int,
             groups: Optional[Sequence[np.ndarray]] = None, *,
             ion: str = "Li", cutoff: Optional[float] = None,
             donor_elements: Sequence[str] = DONOR_ELEMENTS,
             max_molecules: int = 0) -> Shell:
    """The shell of one ion in one frame, repaired across the periodic boundary.

    `max_molecules` above zero keeps only the molecules whose nearest donor is closest to the
    ion, which bounds the cost when a cutoff happens to catch an extra molecule.
    """
    if groups is None:
        groups = residue_groups(traj)
    k = int(frame)
    if not 0 <= k < traj.n_frames:
        raise ShellError(f"frame {frame} is outside the {traj.n_frames} frames read")
    rad = shell_cutoff(ion, DEFAULT_SHELL_FACTOR) if cutoff is None else float(cutoff)
    donors_wanted = {str(e).strip().capitalize() for e in donor_elements}
    elements = [str(e).strip().capitalize() for e in traj.elements]
    box = None if traj.box is None else np.asarray(traj.box, float)[k:k + 1]
    frame_coords = np.asarray(traj.coords, float)[k:k + 1].copy()

    resids = np.asarray(traj.resids, int)
    ion_resid = int(resids[int(ion_atom)])

    # Which molecules are near the ion is decided first, on every atom at once and with the
    # nearest periodic image of each. Repairing a molecule costs far more than measuring a
    # distance, so only the few molecules that pass are repaired.
    donors = np.asarray([i for i, e in enumerate(elements) if e in donors_wanted], int)
    donors = donors[resids[donors] != ion_resid]
    if len(donors) == 0:
        picked: List[Tuple[float, np.ndarray, List[int]]] = []
    else:
        here = frame_coords[0]
        delta = here[donors] - here[int(ion_atom)]
        if box is not None:
            cell = np.asarray(box[0], float)
            delta = delta - np.round(delta @ np.linalg.inv(cell)) @ cell
        d = np.linalg.norm(delta, axis=1)
        inside = donors[d <= rad]
        by_residue: Dict[int, List[int]] = {}
        for j in inside.tolist():
            by_residue.setdefault(int(resids[j]), []).append(int(j))
        lookup = {int(resids[np.asarray(g, int)[0]]): np.asarray(g, int) for g in groups}
        picked = []
        for resid, local in by_residue.items():
            g = lookup[resid]
            whole = make_whole(frame_coords, g, box)
            whole = reimage(whole, g, [int(ion_atom)], box)
            fixed = whole[0]
            dd = np.linalg.norm(fixed[np.asarray(local, int)] - fixed[int(ion_atom)], axis=1)
            keep = [local[t] for t in range(len(local)) if dd[t] <= rad]
            if not keep:
                continue
            picked.append((float(dd.min()), g, keep))

    picked.sort(key=lambda t: t[0])
    if max_molecules and len(picked) > int(max_molecules):
        picked = picked[:int(max_molecules)]

    atoms: List[int] = [int(ion_atom)]
    coords: List[np.ndarray] = []
    species: List[str] = []
    sizes: List[int] = []
    donor_rows: List[int] = []
    donor_d: List[float] = []
    # rebuild the geometry of the molecules that were kept, in the order they were kept
    base = frame_coords
    ion_xyz = None
    for _, g, donors in picked:
        whole = make_whole(base, g, box)
        whole = reimage(whole, g, [int(ion_atom)], box)
        here = whole[0]
        if ion_xyz is None:
            ion_xyz = here[int(ion_atom)].copy()
        offset = 1 + int(sum(sizes))
        coords.append(here[g])
        atoms.extend(int(j) for j in g)
        species.append(str(traj.residues[g[0]]))
        sizes.append(int(len(g)))
        order = {int(j): t for t, j in enumerate(g.tolist())}
        for j in donors:
            donor_rows.append(offset + order[int(j)])
            donor_d.append(float(np.linalg.norm(here[int(j)] - here[int(ion_atom)])))
    if ion_xyz is None:
        ion_xyz = np.asarray(traj.coords, float)[k][int(ion_atom)].copy()

    xyz = np.vstack([ion_xyz[None, :]] + coords) if coords else ion_xyz[None, :].copy()
    out = Shell(frame=k, ion_atom=int(ion_atom), ion=str(ion).strip().capitalize(),
                atoms=np.asarray(atoms, int), coords=np.asarray(xyz, float),
                species=tuple(species), sizes=tuple(sizes),
                donor_distances=tuple(donor_d))
    setattr(out, "_donors", list(donor_rows))
    return out


def extract_shells(traj: Trajectory, ion: str = "Li", *, cutoff: Optional[float] = None,
                   stride: int = 1, donor_elements: Sequence[str] = DONOR_ELEMENTS,
                   max_molecules: int = 0, min_molecules: int = 1,
                   progress: Optional[Callable[[int, int], None]] = None) -> List[Shell]:
    """Every shell of every ion in every frame taken, as a flat list.

    `stride` takes every n-th frame. A shell with fewer than `min_molecules` molecules is
    dropped, which removes the few frames in which an ion sits in a gap of the liquid.
    """
    ions = ion_atoms(traj, ion)
    if len(ions) == 0:
        raise ShellError(f"the trajectory holds no {ion} atom")
    if traj.box is None:
        # not an error: a constructed, non-periodic case is useful in tests
        pass
    groups = residue_groups(traj)
    frames = list(range(0, traj.n_frames, max(1, int(stride))))
    out: List[Shell] = []
    for t, k in enumerate(frames):
        for a in ions.tolist():
            sh = shell_at(traj, k, int(a), groups, ion=ion, cutoff=cutoff,
                          donor_elements=donor_elements, max_molecules=max_molecules)
            if sh.n_molecules >= int(min_molecules):
                out.append(sh)
        if progress is not None:
            progress(t + 1, len(frames))
    return out


# ----------------------------------------------------------------------------------------------
# a shell as a molecule
# ----------------------------------------------------------------------------------------------

def shell_molecule(shell: Shell, templates: Dict[str, Chem.Mol]) -> Chem.Mol:
    """One shell as an RDKit molecule, with the ion as atom 0.

    A trajectory topology carries bonds but no bond orders, so the species are supplied as
    templates: one molecule per residue name, with its hydrogens explicit and its atoms in the
    order the simulation writes them. That is the order a packed box has, because the box was
    packed from those same files. The template is checked element by element, so a mismatch is an
    error here and not a wrong bond order downstream.
    """
    ion_atom = Chem.Atom(shell.ion)
    ion_atom.SetFormalCharge(int(ION_CHARGE.get(shell.ion, 1)))
    ion_atom.SetNoImplicit(True)
    out = Chem.RWMol()
    out.AddAtom(ion_atom)
    placed: List[np.ndarray] = [shell.coords[0]]
    for i, name in enumerate(shell.species):
        if name not in templates:
            raise ShellError(f"no template for species {name!r}; the templates given are "
                             f"{', '.join(sorted(templates)) or 'none'}")
        template = Chem.Mol(templates[name])
        sl = shell.molecule_slice(i)
        xyz = shell.coords[sl]
        if template.GetNumAtoms() != xyz.shape[0]:
            raise ShellError(f"the template of {name!r} has {template.GetNumAtoms()} atoms and "
                             f"the trajectory's residue has {xyz.shape[0]}")
        out.InsertMol(template)
        placed.append(xyz)
    mol = out.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    at = 0
    for block in placed:
        block = np.atleast_2d(block)
        for xyz in block:
            conf.SetAtomPosition(at, Point3D(*[float(v) for v in xyz]))
            at += 1
    if at != mol.GetNumAtoms():
        raise ShellError(f"{at} coordinates for {mol.GetNumAtoms()} atoms")
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    mol.SetProp("_Name", f"{shell.ion}+ shell, frame {shell.frame}")
    mol.SetProp("shell_composition",
                " ".join(f"{s}:{n}" for s, n in shell.composition))
    mol.SetProp("shell_coordination", str(shell.coordination))
    return mol


def check_templates(traj: Trajectory, templates: Dict[str, Chem.Mol]) -> None:
    """Raise unless every residue of the trajectory matches its template element by element.

    Called once per trajectory rather than once per shell, because the check is about the
    topology and not about a frame.
    """
    groups = residue_groups(traj)
    elements = [str(e).strip().capitalize() for e in traj.elements]
    checked = set()
    for g in groups:
        name = str(traj.residues[g[0]])
        if name in checked or name not in templates:
            continue
        template = templates[name]
        if template.GetNumAtoms() != len(g):
            raise ShellError(f"the template of {name!r} has {template.GetNumAtoms()} atoms and "
                             f"the trajectory's residue has {len(g)}")
        want = [a.GetSymbol().capitalize() for a in template.GetAtoms()]
        got = [elements[int(j)] for j in g]
        if want != got:
            bad = next(t for t in range(len(want)) if want[t] != got[t])
            raise ShellError(f"the template of {name!r} and the trajectory's residue differ at "
                             f"atom {bad}: the template has {want[bad]} and the trajectory "
                             f"{got[bad]}")
        checked.add(name)


# ----------------------------------------------------------------------------------------------
# the thinning
# ----------------------------------------------------------------------------------------------

def composition_table(shells: Sequence[Shell]) -> List[Tuple[Tuple[Tuple[str, int], ...], float]]:
    """Each composition and the share of shells that have it, most common first."""
    counts: Dict[Tuple[Tuple[str, int], ...], int] = {}
    for sh in shells:
        key = sh.composition
        counts[key] = counts.get(key, 0) + 1
    total = float(sum(counts.values())) or 1.0
    return sorted(((k, v / total) for k, v in counts.items()),
                  key=lambda t: (-t[1], t[0]))


def _leader(points: Sequence[np.ndarray], radius: float) -> List[List[int]]:
    """Leader clustering: the first point starts a cluster, each later one joins or starts one."""
    leaders: List[int] = []
    members: List[List[int]] = []
    for i, p in enumerate(points):
        best = -1
        for t, lead in enumerate(leaders):
            q = points[lead]
            if len(q) != len(p):
                continue
            if float(np.linalg.norm(p - q)) <= radius:
                best = t
                break
        if best < 0:
            leaders.append(i)
            members.append([i])
        else:
            members[best].append(i)
    return members


def cluster_shells(shells: Sequence[Shell], *, max_members: int = DEFAULT_MAX_MEMBERS,
                   radius: float = DEFAULT_CLUSTER_RADIUS
                   ) -> Tuple[List[int], np.ndarray]:
    """Representative shells and their weights.

    Shells are grouped by composition, the groups are given representatives in proportion to how
    many shells they hold, and inside a group the representatives are leaders of a clustering on
    :meth:`Shell.fingerprint`. The weights are populations: the share of all shells that the
    representative stands for. They are never Boltzmann factors of a frame energy, for the reason
    item D7 gives -- a trajectory has already sampled the distribution once, and weighting it
    again squares it.

    The first representative returned is the one from the composition with the most distinct
    species, so that the atom map of the ensemble has as few gaps as possible.
    """
    if not shells:
        raise ShellError("no shells to thin")
    by_comp: Dict[Tuple[Tuple[str, int], ...], List[int]] = {}
    for i, sh in enumerate(shells):
        by_comp.setdefault(sh.composition, []).append(i)
    total = float(len(shells))
    order = sorted(by_comp, key=lambda k: (-len(by_comp[k]), k))
    cap = max(1, int(max_members))

    # how many representatives each composition gets: at least one for the largest ones
    shares = np.array([len(by_comp[k]) / total for k in order], float)
    want = np.maximum(1, np.round(shares * cap).astype(int))
    while want.sum() > cap and len(want) > 1:
        # drop the least populated composition rather than shave the largest
        keep = int(np.argmin(shares[: len(want)]))
        if want[keep] > 1:
            want[keep] -= 1
        else:
            order = order[:keep] + order[keep + 1:]
            shares = np.delete(shares, keep)
            want = np.delete(want, keep)
    picked: List[Tuple[int, float]] = []
    for t, key in enumerate(order):
        idx = by_comp[key]
        points = [shells[i].fingerprint() for i in idx]
        clusters = _leader(points, float(radius))
        clusters.sort(key=len, reverse=True)
        n = min(int(want[t]), len(clusters))
        chosen = clusters[:n]
        covered = float(sum(len(c) for c in chosen)) or 1.0
        group_share = len(idx) / total
        for c in chosen:
            picked.append((idx[c[0]], group_share * len(c) / covered))

    # the richest composition first, so that the atom map has as few gaps as possible
    picked.sort(key=lambda t: (-len(shells[t[0]].composition), -t[1]))
    reps = [i for i, _ in picked]
    w = np.array([x for _, x in picked], float)
    return reps, w / w.sum()


def atom_maps(reps: Sequence[Shell], templates: Dict[str, Chem.Mol]) -> List[np.ndarray]:
    """For each representative, the atom of the first representative each atom corresponds to.

    The ion maps to the ion. An atom of a solvent or anion molecule maps to the same atom of the
    *first* molecule of its own species in the first representative, so a per-atom reading of the
    model is per atom of a species, summed over however many molecules of that species the shell
    holds. Where the first representative has no molecule of that species the atoms map to -1,
    which is how :func:`cmfqsar.fields.combine_members` marks an atom with no counterpart.
    """
    if not reps:
        raise ShellError("no representatives to map")
    first = reps[0]
    offset: Dict[str, int] = {}
    for i, name in enumerate(first.species):
        if name not in offset:
            offset[name] = first.molecule_slice(i).start
    out: List[np.ndarray] = []
    for sh in reps:
        amap = np.full(sh.n_atoms, -1, int)
        amap[0] = 0
        for i, name in enumerate(sh.species):
            sl = sh.molecule_slice(i)
            if name in offset:
                amap[sl] = offset[name] + np.arange(int(sh.sizes[i]), dtype=int)
        out.append(amap)
    return out


# ----------------------------------------------------------------------------------------------
# the ensemble
# ----------------------------------------------------------------------------------------------

@dataclass
class ShellEnsemble:
    """The mean field of a set of shells, with what it was built from."""
    fields: MoleculeFields
    members: List[Chem.Mol] = _field(default_factory=list)
    #: the shells the members were made from, in the same order
    chosen: List[Shell] = _field(default_factory=list)
    weights: np.ndarray = _field(default_factory=lambda: np.zeros(0))
    composition: List[Tuple[Tuple[Tuple[str, int], ...], float]] = _field(default_factory=list)
    mean_coordination: float = float("nan")
    n_shells: int = 0
    note: str = ""


def shell_ensemble(shells: Sequence[Shell], templates: Dict[str, Chem.Mol],
                   fieldset: FieldSet, *, max_members: int = DEFAULT_MAX_MEMBERS,
                   radius: float = DEFAULT_CLUSTER_RADIUS, name: str = "",
                   frame: bool = True) -> ShellEnsemble:
    """The mean field of a collection of shells, as one :class:`MoleculeFields`.

    With `frame` set, every member is put in the cation frame of item BT4 first: the origin on
    the ion and the axes on the donor polyhedron. The aligned kernel needs that; the
    alignment-free kernel does not care, and the frame does not change what it reports.
    """
    reps, weights = cluster_shells(shells, max_members=max_members, radius=radius)
    chosen = [shells[i] for i in reps]
    mols = [shell_molecule(sh, templates) for sh in chosen]
    if frame:
        mols = [_in_cation_frame(m) for m in mols]
    members = [build_fields(m, fieldset, name=f"{name or 'shell'}_{t}")
               for t, m in enumerate(mols)]
    maps = atom_maps(chosen, templates)
    fields = combine_members(members, weights, kind="shells",
                             atom_maps=[m.tolist() for m in maps], name=name)
    coord = float(np.mean([sh.coordination for sh in shells])) if shells else float("nan")
    return ShellEnsemble(fields=fields, members=mols, chosen=chosen,
                         weights=np.asarray(weights, float),
                         composition=composition_table(shells), mean_coordination=coord,
                         n_shells=len(shells),
                         note=f"{len(shells)} shells thinned to {len(reps)} members")


def _in_cation_frame(mol: Chem.Mol) -> Chem.Mol:
    """The molecule in the cation frame of item BT4, the ion at the origin and atom 0."""
    return with_frame(mol, index=0, ion=mol.GetAtomWithIdx(0).GetSymbol())


# ------------------------------------------------- an ensemble written to a file and read back

#: the properties a written member carries, so that the ensemble can be rebuilt from a file
MEMBER_PROPERTIES = ("shell_weight", "shell_species", "shell_sizes")


def member_properties(shell: Shell, weight: float) -> Dict[str, str]:
    """What has to be written beside a member so that the ensemble can be rebuilt.

    A trajectory is too large to keep in a repository and the members are not conformations of
    one molecule, so the usual multi-conformer file cannot hold them: each member has its own
    atoms. These three properties are what :func:`ensemble_from_molecules` needs instead.
    """
    return {"shell_weight": f"{float(weight):.8f}",
            "shell_species": " ".join(shell.species),
            "shell_sizes": " ".join(str(int(n)) for n in shell.sizes)}


def ensemble_from_molecules(mols: Sequence[Chem.Mol], fieldset: FieldSet,
                            name: str = "") -> MoleculeFields:
    """The mean field of members read back from a file, in the order the file has them.

    Each molecule carries the three properties of :func:`member_properties`. The first molecule
    is the one every other member's atoms are mapped onto, which is why the writer puts the
    member with the most species first.
    """
    if not mols:
        raise ShellError("an ensemble needs at least one member")
    species: List[List[str]] = []
    sizes: List[List[int]] = []
    weights: List[float] = []
    for i, m in enumerate(mols):
        for prop in MEMBER_PROPERTIES:
            if not m.HasProp(prop):
                raise ShellError(f"member {i} carries no {prop!r}, so the ensemble cannot be "
                                 f"rebuilt from it")
        weights.append(float(m.GetProp("shell_weight")))
        species.append(m.GetProp("shell_species").split())
        sizes.append([int(v) for v in m.GetProp("shell_sizes").split()])
        if 1 + sum(sizes[-1]) != m.GetNumAtoms():
            raise ShellError(f"member {i} says {1 + sum(sizes[-1])} atoms and holds "
                             f"{m.GetNumAtoms()}")
    offset: Dict[str, int] = {}
    at = 1
    for sp, n in zip(species[0], sizes[0]):
        offset.setdefault(sp, at)
        at += int(n)
    maps = []
    for sp, sz, m in zip(species, sizes, mols):
        amap = np.full(m.GetNumAtoms(), -1, int)
        amap[0] = 0
        at = 1
        for s, n in zip(sp, sz):
            if s in offset:
                amap[at:at + int(n)] = offset[s] + np.arange(int(n), dtype=int)
            at += int(n)
        maps.append(amap)
    members = [build_fields(m, fieldset, name=f"{name or 'shell'}_{t}")
               for t, m in enumerate(mols)]
    return combine_members(members, weights, kind="shells",
                           atom_maps=[m.tolist() for m in maps], name=name)


# ----------------------------------------------------------------------------------------------
# the error of the mean
# ----------------------------------------------------------------------------------------------

def _normalised_distance(a: MoleculeFields, b: MoleculeFields, fieldset: FieldSet) -> float:
    """1 - cos(a, b) over the summed channels; 0 when the two fields have the same shape."""
    specs = fieldset.specs
    ta, tb = AtomTable.from_molecules([a]), AtomTable.from_molecules([b])
    kaa = float(np.sum(self_kernels(ta, specs)))
    kbb = float(np.sum(self_kernels(tb, specs)))
    kab = float(np.sum(kernel_matrices(ta, tb, specs)))
    if kaa <= 0 or kbb <= 0:
        return float("nan")
    return float(1.0 - kab / np.sqrt(kaa * kbb))


def _overlap_matrix(members: Sequence[MoleculeFields], fieldset: FieldSet) -> np.ndarray:
    """The overlap of every pair of member fields, channels summed with equal weights.

    This one matrix answers every question about weighted means of those members, because the
    kernel is bilinear: the overlap of two mean fields is the same weighted sum of these
    entries. That is what makes the diagnostics below exact rather than approximations on a
    thinned ensemble.
    """
    table = AtomTable.from_molecules(list(members))
    return np.sum(kernel_matrices(table, table, fieldset.specs, symmetric=True), axis=0)


def _distance_from_matrix(K: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """1 - cos between the mean fields with member weights `u` and `v`."""
    uu = float(u @ K @ u)
    vv = float(v @ K @ v)
    if uu <= 0 or vv <= 0:
        return float("nan")
    return float(1.0 - float(u @ K @ v) / np.sqrt(uu * vv))


def _member_fields(shells: Sequence[Shell], templates: Dict[str, Chem.Mol],
                   fieldset: FieldSet) -> List[MoleculeFields]:
    """One field per shell, each in the cation frame so that overlaps mean something."""
    return [build_fields(_in_cation_frame(shell_molecule(sh, templates)), fieldset,
                         name=f"shell_{t}")
            for t, sh in enumerate(shells)]


def convergence(shells: Sequence[Shell], templates: Dict[str, Chem.Mol], fieldset: FieldSet, *,
                n_blocks: int = 5, sample: int = 200,
                max_members: int = DEFAULT_MAX_MEMBERS,
                radius: float = DEFAULT_CLUSTER_RADIUS) -> Dict[str, float]:
    """How well the mean field is sampled, and what the thinning to members costs.

    Two numbers, both as a share of the field's own size, and both exact rather than measured on
    a thinned ensemble:

    ``block_error``
        The frames are split into `n_blocks` consecutive blocks, the same number of shells is
        taken from each, and the number returned is the standard error of the mean over the
        blocks of the distance from a block's mean field to the mean field of all of them. Item
        D8 requires this beside any accuracy taken from a simulation: an unconverged trajectory
        gives a mean field whose sampling error would otherwise read as chemistry.

    ``thinning_error``
        The distance from the thinned ensemble of `max_members` representatives to the mean field
        of every sampled shell. This is the price of the thinning, which the kernel cost makes
        necessary, and it has to be separated from the sampling error above: thinning more
        finely cannot fix an unconverged trajectory, and a longer trajectory cannot fix a
        thinning that is too coarse.

    `sample` is the total number of shells the overlap matrix is built on. The cost is quadratic
    in it, so it is a setting rather than a constant.
    """
    if n_blocks < 2:
        raise ShellError("a block average needs at least two blocks")
    frames = sorted({int(sh.frame) for sh in shells})
    if len(frames) < n_blocks:
        raise ShellError(f"{len(frames)} frames cannot be split into {n_blocks} blocks")

    # the same number of shells from each block, so that the blocks can be compared
    by_block: List[List[int]] = []
    for block in np.array_split(np.asarray(frames, int), int(n_blocks)):
        keep = {int(f) for f in block.tolist()}
        by_block.append([i for i, sh in enumerate(shells) if int(sh.frame) in keep])
    if any(len(b) == 0 for b in by_block):
        raise ShellError("a block held no shell at all, so the blocks cannot be compared")
    per_block = max(1, min(min(len(b) for b in by_block),
                           int(sample) // int(n_blocks)))
    picked: List[int] = []
    groups: List[List[int]] = []
    for b in by_block:
        step = max(1, len(b) // per_block)
        take = b[::step][:per_block]
        groups.append(list(range(len(picked), len(picked) + len(take))))
        picked.extend(take)

    chosen = [shells[i] for i in picked]
    reps, weights = cluster_shells(shells, max_members=max_members, radius=radius)
    thinned = [shells[i] for i in reps]
    members = _member_fields(chosen + thinned, fieldset=fieldset, templates=templates)
    K = _overlap_matrix(members, fieldset)
    n = len(chosen)

    whole = np.zeros(K.shape[0])
    whole[:n] = 1.0 / n
    d = []
    for g in groups:
        u = np.zeros(K.shape[0])
        u[np.asarray(g, int)] = 1.0 / len(g)
        d.append(_distance_from_matrix(K, u, whole))
    d = np.asarray([v for v in d if np.isfinite(v)], float)
    block = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) >= 2 else float("nan")

    v = np.zeros(K.shape[0])
    v[n:] = np.asarray(weights, float)
    return {"block_error": block,
            "thinning_error": _distance_from_matrix(K, v, whole),
            "sampled": float(n), "members": float(len(reps))}


def block_error(shells: Sequence[Shell], templates: Dict[str, Chem.Mol], fieldset: FieldSet, *,
                n_blocks: int = 5, sample: int = 200,
                max_members: int = DEFAULT_MAX_MEMBERS,
                radius: float = DEFAULT_CLUSTER_RADIUS) -> float:
    """The block-averaged error of the mean field alone; see :func:`convergence`."""
    return convergence(shells, templates, fieldset, n_blocks=n_blocks, sample=sample,
                       max_members=max_members, radius=radius)["block_error"]
