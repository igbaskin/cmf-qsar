# Single-atom sites in nitrogen-doped graphene, with their H, O, OH and OOH binding energies

The data set of He, Wang and Ouyang, *Design principle of carbon-supported single-atom
catalysts: interplay between d-orbital periodicity and local hybridization*, Chem. Mater. 2023,
DOI 10.1021/acs.chemmater.3c02549, as published in the raw-data repository
github.com/Jeff-oakley/SAC_HOER_data (one commit, no licence file; cite the paper).  28 metals
(Ag, Au, Cd, Co, Cr, Cu, Fe, Hf, Ir, La, Mn, Mo, Nb, Ni, Os, Pd, Pt, Re, Rh, Ru, Sc, Ta, Ti, V,
W, Y, Zn, Zr) in five motifs of a graphene divacancy, M@C4, M@C3N, M@C2N2, M@CN3 and M@N4, in an
8x8 sheet of 128 carbon positions (127 atoms with the metal), each relaxed bare and with H, O, OH
and OOH on the metal: VASP, PBE with the D2 dispersion correction, 520 eV, spin-polarised,
Gamma point.  `benchmarks/run_sac.py --prepare` reads the clone and writes the files here.

Files
-----
* `substrates.extxyz`: the 139 relaxed bare sheets (Au@N4 has no relaxed sheet in the source),
  in the order of the table, with the catalyst name in `info`.
* `sac_hoer.csv`: one row per catalyst: the total energies of the bare sheet and of the four
  adsorbed states (`E_sub`, `E_H`, `E_O`, `E_OH`, `E_OOH`, in eV, the last ionic step of each
  OSZICAR), the number of ionic steps recorded, the height of the metal above the sheet, the
  mean distance to its four neighbours, and the shortest distance from the metal to an adsorbate
  atom in each adsorbed state (`d_metal_*`; above 2.6 A the adsorbate ended on the sheet, which
  happens for OOH on Ag and Au and for H on La@C2N2 and Sc@C2N2).  `ldau_sub` = 1 marks the
  40 bare sheets described next.
* `substrates_pbe.csv`: plain-PBE energies of those bare sheets recomputed here (see below), one
  row per finished run.
* `gas_references.json`: the energies of H2 and H2O computed here at the settings of the set
  (PBE+D2, 520 eV, spin-polarised, Gaussian smearing 0.05 eV, a 12 A box), which the binding
  energies of the computational hydrogen electrode need:
  dE_H = E(*H) - E(*) - E(H2)/2, dE_OH = E(*OH) - E(*) - E(H2O) + E(H2)/2,
  dE_O = E(*O) - E(*) - E(H2O) + E(H2), dE_OOH = E(*OOH) - E(*) - 2 E(H2O) + 3 E(H2)/2.

A fault in the source data
--------------------------
The bare sheets of Co, Cr, Fe, Mn, Mo, Ni, V and W (40 of the 140) were relaxed with DFT+U
(LDAU = .TRUE., the Materials Project U values of 3.3 to 6.2 eV on the metal d shell), while
every adsorbed state of the set was relaxed with plain PBE.  Total energies with and without
the +U term differ by about 10 eV, so the binding energies formed from those 40 sheets are
not binding energies (dE_H of -10 to -14 eV).  The differences between adsorbed states of one
catalyst (dE_OH - dE_H, dE_O - dE_OH, dE_OOH - dE_OH) do not involve the bare sheet and are
sound for every catalyst.  `benchmarks/run_sac.py` therefore takes the bare-sheet energy of
those catalysts from `substrates_pbe.csv`, where it was recomputed with plain PBE+D2 at the
settings of the adsorbed runs (the same INCAR without the LDAU tags, the same POTCAR variants,
starting from the +U geometry, up to 40 ionic steps), and gives the rest no binding energy.
W is not recomputed: its runs used the W_pv potential, which this machine's PAW set lacks, and a
different potential would bring the same fault back.
