# Structures that are not molecules, from six file formats (plan item NT1)

## What each format keeps

Each set is written in the format and read back as the program reads it ('as read'), then again with the periodic directions given ('directions given'). The kernel difference is the largest difference of the periodic autocorrelation rows (cutoff 8 A, channels every 0.5 A) divided by the largest entry. A displacement with no change of a distance is a rotation or a shift of the whole structure: a CIF file stores the lengths and angles of the cell, so the reader places the cell in the standard orientation. 'Refused' gives the reason ASE gives when it cannot write the structure in that format.

### Crystals: 60 structures, 10 to 36 atoms

| Format | Directions | Read | Right name | Right target | Right periodic directions | Largest atom displacement, A | Largest change of a distance, A | Kernel difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| extended XYZ | as read | 60 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| extended XYZ | given | 60 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE database (SQLite) | as read | 60 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE database (SQLite) | given | 60 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE database (JSON) | as read | 60 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE database (JSON) | given | 60 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE trajectory | as read | 60 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE trajectory | given | 60 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| folder of CIF files | as read | 60 | 100 % | 100 % | 100 % | 3.7e-15 | 6.2e-15 | 4.3e-15 |
| folder of CIF files | given | 60 | 100 % | 100 % | 100 % | 3.7e-15 | 6.2e-15 | 4.3e-15 |
| folder of POSCAR files | as read | 60 | 100 % | 100 % | 100 % | 2.0e-15 | 5.3e-15 | 2.4e-15 |
| folder of POSCAR files | given | 60 | 100 % | 100 % | 100 % | 2.0e-15 | 5.3e-15 | 2.4e-15 |

### Layers: 139 structures, 127 to 127 atoms

| Format | Directions | Read | Right name | Right target | Right periodic directions | Largest atom displacement, A | Largest change of a distance, A | Kernel difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| extended XYZ | as read | 139 | 100 % | 100 % | 100 % | 6.7e-09 | 1.8e-14 | 4.1e-15 |
| extended XYZ | given | 139 | 100 % | 100 % | 100 % | 6.7e-09 | 1.8e-14 | 4.1e-15 |
| ASE database (SQLite) | as read | 139 | 100 % | 100 % | 100 % | 7.9e-15 | 1.4e-14 | 3.1e-15 |
| ASE database (SQLite) | given | 139 | 100 % | 100 % | 100 % | 7.9e-15 | 1.4e-14 | 3.1e-15 |
| ASE database (JSON) | as read | 139 | 100 % | 100 % | 100 % | 7.9e-15 | 1.4e-14 | 3.1e-15 |
| ASE database (JSON) | given | 139 | 100 % | 100 % | 100 % | 7.9e-15 | 1.4e-14 | 3.1e-15 |
| ASE trajectory | as read | 139 | 100 % | 100 % | 100 % | 7.9e-15 | 1.4e-14 | 3.1e-15 |
| ASE trajectory | given | 139 | 100 % | 100 % | 100 % | 7.9e-15 | 1.4e-14 | 3.1e-15 |
| folder of CIF files | as read | 139 | 100 % | 100 % | 0 % | 1.0e-01 | 2.0e-14 | 4.6e+00 |
| folder of CIF files | given | 139 | 100 % | 100 % | 100 % | 1.0e-01 | 2.1e-14 | 8.0e-15 |
| folder of POSCAR files | as read | 139 | 100 % | 100 % | 0 % | 7.3e-15 | 2.0e-14 | 4.6e+00 |
| folder of POSCAR files | given | 139 | 100 % | 100 % | 100 % | 1.5e-14 | 1.8e-14 | 4.1e-15 |

### Chains: 39 structures, 16 to 388 atoms

| Format | Directions | Read | Right name | Right target | Right periodic directions | Largest atom displacement, A | Largest change of a distance, A | Kernel difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| extended XYZ | as read | 39 | 100 % | 100 % | 100 % | 8.4e-09 | 1.5e-08 | 8.7e-09 |
| extended XYZ | given | 39 | 100 % | 100 % | 100 % | 8.4e-09 | 1.5e-08 | 8.7e-09 |
| ASE database (SQLite) | as read | 39 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE database (SQLite) | given | 39 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE database (JSON) | as read | 39 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE database (JSON) | given | 39 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE trajectory | as read | 39 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE trajectory | given | 39 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| folder of CIF files | as read | refused: writing: ValueError: CIF format can only represent systems with 0 or 3 lattice vectors.  Got 1. | | | | | | |
| folder of POSCAR files | as read | refused: writing: RuntimeError: Lattice vectors must be finite and non-parallel. At least one lattice length or angle is zero. | | | | | | |

### Particles: 30 structures, 13 to 309 atoms

| Format | Directions | Read | Right name | Right target | Right periodic directions | Largest atom displacement, A | Largest change of a distance, A | Kernel difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| extended XYZ | as read | 30 | 100 % | 100 % | 100 % | 7.5e-09 | 1.4e-08 | 2.1e-09 |
| extended XYZ | given | 30 | 100 % | 100 % | 100 % | 7.5e-09 | 1.4e-08 | 2.1e-09 |
| ASE database (SQLite) | as read | 30 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE database (SQLite) | given | 30 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE database (JSON) | as read | 30 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE database (JSON) | given | 30 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE trajectory | as read | 30 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| ASE trajectory | given | 30 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| folder of CIF files | as read | 30 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| folder of CIF files | given | 30 | 100 % | 100 % | 100 % | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| folder of POSCAR files | as read | refused: writing: RuntimeError: Lattice vectors must be finite and non-parallel. At least one lattice length or angle is zero. | | | | | | |

With no periodic direction the periodic autocorrelation has no background term. The channels of each cluster with no cell equal those of the same cluster alone in a 60 A periodic box with the background term switched off to 1.1e-15 of the largest entry.

## Reading speed

All 665 cells of the melting-point set (77,820 atoms), the best of three reads, with the target.

| Format | Structures read | With a target | Seconds | Milliseconds per structure | Size on disk, MB |
|---|---:|---:|---:|---:|---:|
| extended XYZ | 665 | 665 | 0.18 | 0.27 | 5.1 |
| ASE database (SQLite) | 665 | 665 | 0.22 | 0.33 | 3.1 |
| ASE database (JSON) | 665 | 665 | 0.08 | 0.12 | 3.3 |
| ASE trajectory | 665 | 665 | 0.07 | 0.11 | 2.7 |
| folder of CIF files | 665 | 665 | 39.09 | 58.79 | 6.7 |
| folder of POSCAR files | 665 | 665 | 0.20 | 0.30 | 5.1 |
