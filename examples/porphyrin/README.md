# Substituted cobalt and iron porphyrins, with computed HER and ORR descriptors

A computed series of metal porphyrins for the hydrogen evolution reaction (HER) and the oxygen
reduction reaction (ORR). The data are made by `examples/make_porphyrin_examples.py` and read by
`benchmarks/run_porphyrin.py`. Nothing in this set is measured. Every number is a DFT result at
one level of theory (see below).

| file | what it holds |
| --- | --- |
| `porphyrins.csv` | one row for each porphyrin and metal (108 rows: 54 porphyrins, Co and Fe) with the descriptors, the spin state used for each species, and notes |
| `structures.extxyz` | 208 geometries (GFN2-xTB), one for each computed species and spin state, with the ORCA energy in the comment line |

**Only the first batch is computed.** The series lists 54 porphyrins. The first batch of 21
porphyrins is finished: 21 cobalt and 19 iron complexes have all descriptors. The other 33
porphyrins have rows with empty descriptor columns and a note. They were not computed, because
the first batch showed a small spread of the descriptors (see below) and the other 33 would need
about 25 hours more.

## Columns of `porphyrins.csv`

| column | meaning |
| --- | --- |
| `name`, `porphyrin`, `metal` | the label, the porphyrin and the metal (Co or Fe) |
| `substituents`, `aza` | the groups on the meso and beta carbons, and the meso carbons replaced by nitrogen |
| `dG_H`, `dG_OH`, `dG_O`, `dG_OOH` | the adsorption free energy of H, OH, O and OOH, in eV |
| `U_L_ORR` | the limiting potential of the ORR, in V |
| `eta_ORR` | the overpotential of the ORR: 1.23 V minus `U_L_ORR` |
| `dG_H_minus_dG_OH` | `dG_H` minus `dG_OH`, in eV |
| `mult_bare`, `mult_H`, `mult_OH`, `mult_O`, `mult_OOH` | the spin multiplicity of the lowest-energy state of each species |
| `notes` | the species without a DFT energy and the geometries that failed a check |

**How the descriptors are defined.** The computational hydrogen electrode model is used.
Each free energy is the DFT energy difference to the bare complex and to the references H2 and
H2O, plus one constant correction for the zero-point energy and entropy of each adsorbate on each
metal (from GFN2-xTB). The limiting potential is

U_L = -max(dG_OOH - 4.92, dG_O - dG_OOH, dG_OH - dG_O, -dG_OH).

## How the data were made

1. RDKit builds the free-base porphyrin. The two N-H hydrogens are removed, and the metal is put
   at the centre of the four inner nitrogen atoms.
2. The adsorbate is put on one face of the metal, along the normal of the ring.
3. GFN2-xTB optimises each geometry. A check refuses a geometry whose metal-nitrogen distance is
   above 2.25 Angstrom, whose ring bonds have changed, or whose metal-adsorbate distance is outside
   a fixed range (1.30 to 1.85 Angstrom for H, 1.55 to 2.25 Angstrom for the others).
4. ORCA 6.1 computes a single point at r2SCAN-3c with tight SCF and the CPCM model of water
   (unrestricted if the multiplicity is above 1).
5. Each species uses its lowest-energy spin state. For the parent porphyrin all states are
   computed. For the substituted members only the states within 0.25 eV of the lowest state of the
   parent are computed.

The cost of the first batch was about 22 hours of computing with 8 to 12 jobs at a time.

## What to keep in mind

- **The spread of the descriptors is small.** For cobalt the standard deviation of `dG_OH` is
  0.17 eV with the phthalocyanine and 0.08 eV without it. For iron the standard deviation of
  `U_L_ORR` is 0.04 V. This is below the usual error of a single DFT functional.
- **The phthalocyanine is an outlier.** Its cobalt `dG_H` is -0.25 eV, against about 0.55 eV for
  the others. The benchmark is run with and without it.
- **The spin order of iron is not certain.** At r2SCAN-3c the states of the iron OH and OOH species
  lie within 0.1 eV of each other.
- **Some cobalt-oxygen doublets are missing.** Seven of them did not converge (the self-consistent
  field stalled), and the quartet is used instead, which is the lowest state of the parent.
- **Two iron complexes have no result.** The bare triplet of Fe-5-NO2 ran out of memory, and the
  bare triplet of Fe-5,15-NH22 reached the time limit of 3 hours with a second, more careful
  setting. Both rows have empty descriptors.
- **Some iron-hydride geometries failed the check** (the iron-hydrogen distance of 1.88 Angstrom is
  too long) and were not computed. For the parent iron hydride only the doublet exists.
- **The sign of the normal of the ring is arbitrary** for substituents that stand out of the plane
  of the ring. This matters only for the overlap kernel in the frame of the core.

## Results

The benchmark (`benchmarks/run_porphyrin.py`, table in `benchmarks/results/porphyrin/table.md`)
compares the field kernels with the count of the substituents and a linear regression on the
Hammett constants. No field kernel is clearly better than the Hammett regression. For cobalt the
Hammett regression is better than the overlap kernel for `dG_H`, `dG_OH` and `dG_O`. For iron the
overlap kernel in the frame of the core is slightly ahead on `U_L_ORR` (0.024 V against 0.030 V
for the Hammett regression, random folds), but the interval of the difference contains zero.

## Source and licence

The structures and the energies are computed here, with ORCA 6.1 and xtb. There is no source
licence to respect. The Hammett constants of the substituents (the sigma-para values) are from
C. Hansch, A. Leo and R. W. Taft, *Chem. Rev.* 91 (1991) 165-195. They are written in
`benchmarks/run_porphyrin.py`.
