# OH binding at every surface site of Pt and Au particles (UMA)

The data set of plan item NT7: the binding energy of an OH group atop every surface site of 48
metal particles, computed with Meta's UMA machine-learned potential (small model `uma-s-1p2p1`,
OC20 task head) by `make_particle_sites.py` in this folder. It is a computed data set, made to
test site-resolved models; it is not a measurement.

## The particles

Eight shapes of each of Pt and Au, at ASE's reference lattice constants (3.92 and 4.08 Å), as
`ase.cluster` builds them:

| Shape | Atoms | What it is |
|---|---:|---|
| `octa4` | 38 | truncated octahedron |
| `cubo55` | 55 | cuboctahedron |
| `octa5` | 79 | truncated octahedron |
| `octa6` | 140 | truncated octahedron |
| `ico55` | 55 | icosahedron |
| `ico147` | 147 | icosahedron |
| `deca55` | 55 | decahedron |
| `deca147` | 147 | decahedron |

Each shape is given perfect (`_r0`) and with 3 (`_r3`) or 8 (`_r8`) surface atoms removed at
random, which leaves steps, kinks and single vacancies. The particles are in `particles.extxyz`
(no cell, periodic in no direction), with `name`, `metal`, `shape` and `removed` in the info of
each.

## The sites and their energies

A surface atom has fewer than twelve neighbours, two atoms being neighbours when they are
closer than 1.2 times the sum of their covalent radii (the rule of
`cmfqsar.periodic.coordination_numbers`). For every surface atom an OH group is put atop it,
the O atom 2.0 Å out along the local normal (the direction away from the mean position of the
atom's neighbours), and relaxed with FIRE to a largest force of 0.05 eV/Å. The metal atoms stay
fixed, because the OC20 head of UMA reconstructs free metal surfaces (see
`examples/corrosion_fe110/README.md`). The O atom moves along the normal only and the H atom
freely, so the OH stays atop its site and tilts as it likes. Left free, the O atom slid off a
quarter of the sites to a bridge between two atoms in a first run on four Pt particles.

The binding energy of a site is

    binding_eV = [E(particle + OH) - E(particle)] - [E(slab + OH) - E(slab)],

with the second bracket OH atop a 3 x 3 x 4 (111) slab of the same metal, relaxed the same way
(`terrace.csv`). So it is the binding relative to the (111) terrace, in eV, negative where the
site binds more strongly.

Surface atoms with the same environment are computed once: two are equivalent when their sorted
distances to every atom of the particle agree to 0.01 Å. `sites.csv` has one row per surface
atom:

| Column | Meaning |
|---|---|
| `particle`, `metal`, `shape`, `removed`, `atoms` | the particle |
| `site` | the index of the surface atom in the particle, counted from 0 |
| `coordination` | its number of neighbours |
| `group` | its equivalence group within the particle |
| `computed_at` | the site of its group that was computed; the row is a copy where this differs from `site` |
| `binding_eV` | the binding energy defined above |
| `o_metal_A` | the shortest distance from the O atom to a metal atom after the relaxation |
| `nearest` | the metal atom nearest to the O atom; at a few defect sites it is a neighbour of the site, because the normal leans toward it |
| `steps`, `converged` | the FIRE steps taken and whether the force criterion was met |

To fit a site model from the command line, take the computed sites as a site table
(`structure,site,value`) and

    cmf build-crystals particles.extxyz --site-file table.csv --fields steric,prop:coordination,prop:generalized_coordination,prop:exposure --cutoff 6 --sigma 0.5 --save sites.cmf
    cmf sites sites.cmf new_particle.extxyz

`benchmarks/run_particle_sites.py` measures the models on this set.

## Running it again

In WSL with the UMA environment of `examples/corrosion_fe110/README.md`:

    source ~/uma-env/bin/activate
    python make_particle_sites.py OUTDIR

It resumes where it stopped. About 4 s per computed site on an RTX 4070 Laptop GPU with
`inference_settings="default"` (the compiled mode recompiles for every new atom count and was
slower here).
