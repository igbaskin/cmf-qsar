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

## `benzoic_acids_xtb.sdf` — the force-field geometries with quantum charges on the atoms

Stage 3 of the same script (`--xtb`): one GFN2-xTB calculation per acid on the geometry of
`benzoic_acids.sdf`, without a geometry optimisation (`cmfqsar.xtb`, program xtb 6.7.1,
accuracy 1). GFN2-xTB is a semi-empirical quantum method. The atoms, their order and their
coordinates are the ones of `benzoic_acids.sdf`, so the file holds nothing about the geometry
of the ORCA run that gives the band.

Per-atom properties: `q_xtb` (the partial charge, in e), `alpha_xtb` (the atomic
polarisability), `fukui_plus` and `fukui_minus` (the change of the charge of the atom when one
electron is added to the molecule, and when one electron is removed), `fukui_zero` (the mean of
the two) and `dual` (`fukui_plus` minus `fukui_minus`). The molecule properties of
`benzoic_acids.sdf` are kept, and the calculation adds its own (the total energy, the HOMO and
LUMO energies, their gap and the dipole).

The run takes 13 seconds for the series with 8 acids at a time. The jobs are cached under
`benchmarks/results/band/xtb_work/`, which is not tracked. A second run reads the finished jobs
and writes the same bytes.

One acid group needs care. The charge of the sulfur atom of an SF5 group is +2.05 e. The largest
charge of an atom in all other acids is +1.40 e (`meta-SO2F`). A model that has not seen an SF5
acid predicts the two SF5 acids badly from these charges.

## `benzoic_acids_ir.json` — the infrared stick spectrum of every acid

Stage 4 of the same script (`--ir`). The frequency job of stage 2 gives every normal mode of an
acid with its infrared intensity, and `benzoic_acids_co.sdf` keeps one of them, the carbonyl
stretch. This file keeps all of them. Nothing is computed: the stage reads the ORCA outputs
under `benchmarks/results/band/orca_work/`.

The file is JSON. Its head gives the axis (`wavenumber`, in cm⁻¹), the unit of the intensity
(km/mol), the scaling factor (0.97) and the level of theory. `acids` is a list with one acid on
a line: `name` (the name of the acid in the SD files), `wavenumbers` (the harmonic wavenumbers
of its normal modes times the scaling factor, in rising order) and `intensities` (one per
wavenumber). The modes of translation and rotation are left out, and so is the imaginary mode
of the five acids that keep one. An acid of N atoms therefore has 3N − 6 lines, or 3N − 7; the
series has 39 to 105 lines per acid, and 43 at the median between 400 and 4000 cm⁻¹. The band
of `benzoic_acids_co.sdf` is one of the lines of its acid.

`cmfqsar.spectrum.Spectrum(wavenumbers, intensities)` makes a stick spectrum of an entry, and
`benchmarks/run_spectra_ir.py` (plan item SP1) uses the spectra as samples: it predicts the
Hammett constant of the substituent from the spectrum.

## `benzoic_acids_ir_xtb.json` — the infrared spectrum at a second level of theory

Stage 5 of the same script (`--xtb-ir`). Every acid is optimised with GFN2-xTB, a
semi-empirical quantum method, from the geometry of `benzoic_acids.sdf`, and its harmonic
frequencies and infrared intensities are computed (`cmfqsar.xtb.run_vibrations`, xtb 6.7.1).
The series takes less than a minute. The jobs are cached under
`benchmarks/results/band/xtb_ir_work/`, which is not tracked.

The layout is the one of `benzoic_acids_ir.json`. The wavenumbers are the ones xtb gives (the
scaling factor is 1), and every acid also has `n_imaginary`. When an optimisation ends on a
saddle point, the job is run once more from the geometry moved along the imaginary mode; ten
acids keep one imaginary mode after that and have one line fewer.

The two levels put the lines at different places: the carbonyl band of this file is 29 cm⁻¹
above the one of `benzoic_acids_ir.json`, and the O-H stretch is 182 cm⁻¹ below.
`benchmarks/run_spectra_bridge.py` (plan item SP3) uses the difference: a spectrum of one
level has to find its own acid among the spectra of the other.

## `benzoic_acids_ir_b97-3c.json` — the infrared spectrum at a second density functional

Stage 6 of the same script (`--orca-ir`). Every acid is optimised again with ORCA 6.1 at
B97-3c, from the optimised geometry of `benzoic_acids_co.sdf`, and its harmonic frequencies
are computed. The layout is the one of `benzoic_acids_ir.json`; the wavenumbers are multiplied
by 0.98, the scaling factor of that level (`cmfqsar.orca.FREQUENCY_SCALE`). An acid that ends
with an imaginary frequency is moved along that mode and optimised once more, as in stage 2;
one acid (`meta-SPh`) keeps one imaginary mode.

The series took 4.6 hours with 16 acids at a time on the serial ORCA build (28 minutes per
acid at the median), and no acid failed. The jobs are cached under
`benchmarks/results/band/orca_work_b97-3c/`, which is not tracked.

The lines of this file are near those of `benzoic_acids_ir.json`: 5.0 cm⁻¹ apart at the
median when the lines of an acid are taken in order, and the carbonyl band is 4.8 cm⁻¹ lower.
`benchmarks/run_spectra_bridge.py` (plan item SP3) uses it as the near second level and
`benzoic_acids_ir_xtb.json` as the rough one.

## What the set is for

`benchmarks/run_band.py` (plan item SP2) predicts the band from the field around the carbonyl
carbon — the local invariant kernel of LP2 and the posed overlap kernel of LP3 — against the
atom-centred count of LP2, against a linear regression on the Hammett constant and against a
linear regression on the C=O bond length,
and reads the co-field at the ring positions. It does so twice: on the optimised geometries of
`benzoic_acids_co.sdf` and on the force-field geometries of `benzoic_acids.sdf`, with the same
bands. The second run is the one that counts, because an optimised geometry holds the C=O bond
length, and that length alone predicts the band. Part 7 adds one indicator field per atom
type on the force-field geometries, and folds that hold out both acids of a substituent
together. Part 8 builds the electrostatic field from the quantum charges of
`benzoic_acids_xtb.sdf` in place of Gasteiger charges, and chooses the envelope width and the
ridge inside the folds. The manual section *A vibrational band as the local target* has the
results.

`benchmarks/run_spectra_ir.py` (plan item SP1) turns the question round. The sample is the
infrared spectrum of `benzoic_acids_ir.json`, and the target is the Hammett constant, which is
measured. The manual section *A spectrum as a one-dimensional field* has the results.

`benchmarks/run_spectra_bridge.py` (plan item SP3) ranks the acids by the kernel between a
spectrum of `benzoic_acids_ir_xtb.json` and the spectra of `benzoic_acids_ir.json`, and
analyses which part of the structure moves which part of the spectrum. The manual section
*The spectrum as the bridge* has the results.
