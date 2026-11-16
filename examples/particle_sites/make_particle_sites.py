#!/usr/bin/env python
"""
OH adsorbed on every surface site of Pt and Au particles, computed with the UMA potential (plan
item NT7).

The particles: eight shapes of 38 to 147 atoms (truncated octahedra, cuboctahedra, icosahedra,
decahedra) of each metal at ASE's reference lattice constant, each perfect and with 3 or 8
surface atoms removed at random, which leaves steps, kinks and single vacancies, so that the
sites are many and different. A surface atom has fewer than twelve neighbours, two atoms being
neighbours when they are closer than 1.2 times the sum of their covalent radii (the rule of
``cmfqsar.periodic.coordination_numbers``).

For every surface site an OH group is placed atop it, the O atom 2.0 A out along the local
normal (away from the site's neighbours), and relaxed with FIRE to a largest force of
0.05 eV/A while every metal atom stays fixed (UMA's oc20 head reconstructs free metal surfaces;
see examples/corrosion_fe110/README.md). The O atom moves along the normal only and the H atom
freely, so that the OH stays atop its site and tilts as it likes: this is the atop binding of
the studies of the generalized coordination number. Left free, the O atom slides off a quarter
of the sites to a bridge between two atoms (measured on the first four Pt particles).

Sites with the same environment are computed once: two surface atoms are equivalent when their
sorted distances to every atom of the particle agree to 0.01 A.

The binding energy of a site is

    dE = [E(particle + OH) - E(particle)] - [E(slab + OH) - E(slab)],

the second bracket being OH atop a 3 x 3 x 4 (111) slab of the same metal, relaxed the same way:
the binding relative to the (111) terrace, in eV, negative where the site binds more strongly.

Run in WSL with the UMA environment (not part of the benchmark):

    source ~/uma-env/bin/activate
    python make_particle_sites.py OUTDIR

It writes OUTDIR/particles.extxyz (the bare particles) and OUTDIR/sites.csv (one row per surface
atom), and it resumes where it stopped.
"""
import csv
import os
import sys
import time

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.cluster import Decahedron, Icosahedron, Octahedron
from ase.constraints import FixAtoms, FixedLine
from ase.data import atomic_numbers, covalent_radii, reference_states
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE

METALS = ("Pt", "Au")
#: (name, builder) of the eight shapes; the builder takes the element and the lattice constant
SHAPES = (
    ("octa4", lambda el, a: Octahedron(el, 4, cutoff=1, latticeconstant=a)),
    ("cubo55", lambda el, a: Octahedron(el, 5, cutoff=2, latticeconstant=a)),
    ("octa5", lambda el, a: Octahedron(el, 5, cutoff=1, latticeconstant=a)),
    ("octa6", lambda el, a: Octahedron(el, 6, cutoff=1, latticeconstant=a)),
    ("ico55", lambda el, a: Icosahedron(el, 3, latticeconstant=a)),
    ("ico147", lambda el, a: Icosahedron(el, 4, latticeconstant=a)),
    ("deca55", lambda el, a: Decahedron(el, 3, 3, 0, latticeconstant=a)),
    ("deca147", lambda el, a: Decahedron(el, 4, 4, 0, latticeconstant=a)),
)
#: how many surface atoms each variant removes (0: the perfect particle)
REMOVED = (0, 3, 8)
FMAX = 0.05
STEPS = 400
O_HEIGHT = 2.0
OH_BOND = 0.97
FIELDS = ("particle", "metal", "shape", "removed", "atoms", "site", "coordination", "group",
          "computed_at", "binding_eV", "o_metal_A", "nearest", "steps", "converged")


def neighbours(atoms):
    cut = 1.2 * covalent_radii[atoms.get_atomic_numbers()]
    i, j = neighbor_list("ij", atoms, cut)
    return i, j


def surface_atoms(atoms):
    i, _ = neighbours(atoms)
    cn = np.bincount(i, minlength=len(atoms))
    return np.flatnonzero(cn < 12), cn


def build_particles():
    out = []
    for el in METALS:
        a = float(reference_states[atomic_numbers[el]]["a"])
        for name, build in SHAPES:
            for k, removed in enumerate(REMOVED):
                p = build(el, a)
                if removed:
                    rng = np.random.default_rng(1000 * k + len(out))
                    surf, _ = surface_atoms(p)
                    del p[sorted(rng.choice(surf, removed, replace=False).tolist())]
                p.center(vacuum=10.0)
                p.pbc = False
                p.info.update({"name": f"{el}_{name}_r{removed}", "metal": el, "shape": name,
                               "removed": removed})
                out.append(p)
    return out


def site_groups(atoms, sites):
    """One group label per site: equal sorted distance lists to 0.01 A."""
    D = atoms.get_all_distances()
    keys, labels = {}, []
    for s in sites:
        key = tuple(np.round(np.sort(D[s]), 2))
        labels.append(keys.setdefault(key, len(keys)))
    return labels


def place_oh(atoms, site):
    i, j = neighbours(atoms)
    nb = j[i == site]
    v = atoms.positions[site] - atoms.positions[nb].mean(axis=0) if len(nb) else \
        atoms.positions[site] - atoms.positions.mean(axis=0)
    n = v / max(np.linalg.norm(v), 1e-9)
    tilt = np.cross(n, [0.3, 0.5, 0.8])
    tilt /= max(np.linalg.norm(tilt), 1e-9)
    o = atoms.positions[site] + O_HEIGHT * n
    h = o + OH_BOND * (0.95 * n + 0.31 * tilt)
    p = atoms.copy()
    p += Atoms("OH", positions=[o, h])
    k = len(atoms)
    p.set_constraint([FixAtoms(indices=list(range(k))), FixedLine(k, n)])
    return p


def relax(atoms, calc):
    atoms.calc = calc
    opt = FIRE(atoms, logfile=None)
    converged = bool(opt.run(fmax=FMAX, steps=STEPS))
    return float(atoms.get_potential_energy()), opt.nsteps, converged


def terrace_reference(el, calc):
    a = float(reference_states[atomic_numbers[el]]["a"])
    slab = fcc111(el, (3, 3, 4), a=a, vacuum=10.0)
    slab.calc = calc
    e_slab = float(slab.get_potential_energy())
    top = int(np.argmax(slab.positions[:, 2]))
    s = slab.copy()
    o = s.positions[top] + [0.0, 0.0, O_HEIGHT]
    s += Atoms("OH", positions=[o, o + [0.30, 0.0, 0.92]])
    k = len(slab)
    s.set_constraint([FixAtoms(indices=list(range(k))), FixedLine(k, [0.0, 0.0, 1.0])])
    e, _, ok = relax(s, calc)
    return e - e_slab, ok


def main(outdir):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    os.makedirs(outdir, exist_ok=True)
    unit = pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="cuda",
                                            inference_settings="default")
    calc = FAIRChemCalculator(unit, task_name="oc20")
    ppath = os.path.join(outdir, "particles.extxyz")
    if os.path.isfile(ppath):
        particles = read(ppath, index=":")
    else:
        particles = build_particles()
        write(ppath, particles)
    refs = {}
    rpath = os.path.join(outdir, "terrace.csv")
    if os.path.isfile(rpath):
        with open(rpath) as fh:
            refs = {r["metal"]: float(r["oh_eV"]) for r in csv.DictReader(fh)}
    for el in METALS:
        if el not in refs:
            refs[el], ok = terrace_reference(el, calc)
            print(el, "terrace", refs[el], ok, flush=True)
    with open(rpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metal", "oh_eV"])
        for el in METALS:
            w.writerow([el, refs[el]])
    spath = os.path.join(outdir, "sites.csv")
    done = set()
    if os.path.isfile(spath):
        with open(spath) as fh:
            done = {r["particle"] for r in csv.DictReader(fh)}
    new_file = not os.path.isfile(spath)
    with open(spath, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for p in particles:
            name = p.info["name"]
            if name in done:
                continue
            t0 = time.perf_counter()
            bare = p.copy()
            bare.calc = calc
            e_bare = float(bare.get_potential_energy())
            sites, cn = surface_atoms(p)
            groups = site_groups(p, sites)
            first = {}
            for s, g in zip(sites, groups):
                first.setdefault(g, s)
            result = {}
            for g, s in first.items():
                q = place_oh(p, s)
                e, steps, ok = relax(q, calc)
                d = np.linalg.norm(q.positions[-2] - q.positions[:len(p)], axis=1)
                result[g] = (e - e_bare - refs[p.info["metal"]], float(d.min()),
                             int(np.argmin(d)), s, steps, ok)
            rows = []
            for s, g in zip(sites, groups):
                e, dist, near, at, steps, ok = result[g]
                rows.append({"particle": name, "metal": p.info["metal"],
                             "shape": p.info["shape"], "removed": p.info["removed"],
                             "atoms": len(p), "site": int(s), "coordination": int(cn[s]),
                             "group": g, "computed_at": int(at), "binding_eV": f"{e:.5f}",
                             "o_metal_A": f"{dist:.3f}", "nearest": near, "steps": steps,
                             "converged": int(ok)})
            w.writerows(rows)
            fh.flush()
            print(f"{name}: {len(sites)} surface atoms, {len(first)} computed, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
