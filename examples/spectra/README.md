# Spectroscopic targets: the carbonyl band of substituted benzoic acids

Data for plan item SP2, built by `examples/make_band_examples.py`. Nothing here is downloaded;
the structures are built from SMILES and the bands are computed on this machine.

## `benzoic_acids.sdf` — 189 substituted benzoic acids, force-field geometries

The 95 substituents of the Hammett set of this repository (`examples/hammett/`, after Hansch,
Leo and Taft, Chem. Rev. 1991, 91, 165, Table I), each placed meta and para to the carboxyl group
of benzoic acid; X = H is the same acid in both series and is kept once. Every acid is embedded
with RDKit ETKDG (20 conformers), optimised with MMFF94 (UFF for the six acids of the three
substituents MMFF94 has no parameters for: SF5, SeCH3 and B(OH)2), and the lowest-energy conformer whose
carboxyl group is *syn* (the hydroxyl hydrogen on the side of the carbonyl oxygen) is kept.

UFF distorts the SF5 group strongly (S–F lengths from 1.25 to 2.17 A), and a quantum optimisation
from such a start breaks the group apart. The script therefore sets every SF5 group to a regular
octahedron after the force field: S–F 1.59 A, C–S 1.80 A, one fluorine on the C–S line and four
perpendicular to it, turned by 45 degrees from the ring plane. Every bond length of every acid is
then checked: it must lie between 0.7 and 1.3 times the sum of the covalent radii of its two
atoms, and the script stops if one does not. The values found are 0.79 to 1.07.

Properties: `substituent` (the label of the Hammett table), `position` (`meta`, `para` or `none`
for the unsubstituted acid), `sigma_m`, `sigma_p`, `sigma` (the constant that applies to the
position: `sigma_p` for a para acid, `sigma_m` for a meta acid, 0 for benzoic acid), `pi` where
the Hammett table has it, `smiles`, `ff_energy` and `force_field` (with a note where the SF5
group was rebuilt), `carboxyl` (`syn`).

## `benzoic_acids_co.sdf` — the same acids, optimised, with the carbonyl stretch

Stage 2 of the same script (`--orca`): every acid is optimised and its harmonic frequencies are
computed with ORCA 6.1 at r2SCAN-3c (`cmfqsar.orca.run_band`). The carbonyl stretch is the
normal mode between 1600 and 1900 cm⁻¹ that moves the carbonyl carbon and oxygen apart most, and
it is multiplied by 0.97, the conventional scaling factor of the level
(`cmfqsar.orca.FREQUENCY_SCALE`). The coordinates are the optimised ones. An acid whose
optimisation breaks or makes a bond is reported as failed (`cmfqsar.orca.changed_bonds`: a bond
length outside 0.6 to 1.5 times the sum of the covalent radii); no acid of the series does.

An optimisation can stop on a saddle point, which the frequency run shows as an imaginary
frequency. Such an acid is moved along the mode of its lowest frequency (the atom that moves most
moves 0.2 A) and optimised once more (`cmfqsar.orca.retry_band`); the second result replaces the
first when it has fewer imaginary frequencies, or as many and a lower energy. In this series 13
acids ended the first optimisation with one imaginary frequency. Eleven took the second result:
eight became minima and three keep one imaginary frequency at a lower energy. Two (`para-N=NPh`,
`meta-N=NPh`) kept the first result. So 5 acids have `n_imaginary` 1, with an imaginary frequency
of 8 to 17 cm⁻¹. The second optimisation moved the band by 0.26 cm⁻¹ at the median and by
3.3 cm⁻¹ at most.
The option `--no-retry` leaves this step out.

Terephthalic acid (`para-COOH`) has two carboxyl groups whose carbonyl stretches couple into two
modes less than 2 cm⁻¹ apart. The band given is the mode with the larger share on the target
bond; the error of this choice is below 1 cm⁻¹.

Per-atom property: `atom.dprop.co_stretch`, the scaled band in cm⁻¹ on the carbonyl carbon and
nothing on the other atoms, which is the form `cmf local` and `benchmarks/run_band.py` read.
Molecule properties: `co_stretch` (scaled), `co_stretch_unscaled`, `co_stretch_intensity`
(km/mol), `co_stretch_share` (how much of the mode is the carbonyl bond, between 0 and 1),
`co_stretch_mode`, `co_stretch_atoms` (the carbon and the oxygen, 0-based), `n_imaginary` (0 at a
minimum), `frequency_scale`, `qm_level`, `qm_geometry`, `qm_dir` and `qm_job` (the directory and
the ORCA job the result comes from), `E_total` (Hartree),
`acid_dihedral` (O=C–O–H after the optimisation, in degrees), `carboxyl_after_opt` and, on an
acid that took the result of a second optimisation, `co_stretch_retry` (the scaled band of the
first optimisation).

The ORCA jobs themselves (inputs, outputs, Hessians) are cached under
`benchmarks/results/band/orca_work/`, which is not tracked; an interrupted run resumes from the
jobs that finished. On the serial ORCA build of this machine (one core per job) benzoic acid
alone takes 5 minutes. With ten to twenty acids running at the same time one acid takes 14 to 100
minutes (33 at the median) and about 200 MB of memory. The series took 8.7 hours: 458 minutes for
the first run and 62 minutes for the second, which repeated the two SF5 acids from the repaired
start geometry and optimised the 13 acids with an imaginary frequency once more.

## What the set is for

`benchmarks/run_band.py` (plan item SP2) predicts the band from the field around the carbonyl
carbon — the local invariant kernel of LP2 and the posed overlap kernel of LP3 — against the
atom-centred count of LP2, against the Hammett line and against a line on the C=O bond length,
and reads the co-field at the ring positions. It does so twice: on the optimised geometries of
`benzoic_acids_co.sdf` and on the force-field geometries of `benzoic_acids.sdf`, with the same
bands. The second run is the one that counts, because an optimised geometry holds the C=O bond
length, and that length alone predicts the band. The manual section *A vibrational band as the
local target* has the results.
