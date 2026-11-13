# A field-based Hammett equation

One table per run of `benchmarks/run_hammett.py`, with the command that produced it. The tag of the summary file names the run: no tag is `hammett_sym.sdf`, one conformer per substituent with its four symmetry images and Gasteiger charges; `_single` the same conformers without the images; `_ens` the conformer ensemble; `_q_hirshfeld` the Hirshfeld charges of the quantum stage; `_boltzmann` Boltzmann instead of uniform weights. q² and RMSEcv by 10-fold cross-validation, α and λ chosen on it.

## The default data set

Data set `examples/hammett/hammett_sym.sdf`, charges gasteiger, uniform weights over the records of each substituent, 10-fold cross-validation.

`python benchmarks/run_hammett.py`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | electrostatic+steric+hydrophobic+hba+hbd | 95 | 3 | 0.1 | 0.73 | 0.14 | 0.96 | 0.26 |
| sigma_m | electrostatic | 95 | 0.8 | 1 | 0.67 | 0.15 | 0.73 | 0.26 |
| sigma_m | steric | 95 | 1.2 | 0.01 | 0.19 | 0.24 | 0.33 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.27 | 0.05 | 0.26 |
| sigma_m | hba | 95 | 0.8 | 1 | 0.17 | 0.24 | 0.27 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | electrostatic+steric | 95 | 2 | 1 | 0.68 | 0.15 | 0.77 | 0.26 |
| sigma_p | electrostatic+steric+hydrophobic+hba+hbd | 95 | 0.5 | 1 | 0.69 | 0.21 | 0.82 | 0.38 |
| sigma_p | electrostatic | 95 | 0.8 | 1 | 0.68 | 0.22 | 0.75 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.01 | 0.14 | 0.36 | 0.29 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.01 | 0.39 | 0.07 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.24 | 0.33 | 0.40 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | electrostatic+steric | 95 | 1.2 | 1 | 0.69 | 0.22 | 0.76 | 0.38 |
| pi | electrostatic+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.74 | 0.56 | 0.83 | 1.09 |
| pi | electrostatic | 69 | 1.2 | 0.01 | 0.32 | 0.89 | 0.73 | 1.09 |
| pi | steric | 69 | 2 | 0.01 | 0.31 | 0.90 | 0.54 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.77 | 0.52 | 0.82 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.28 | 0.92 | 0.34 | 1.09 |
| pi | hbd | 69 | 0.3 | 1 | 0.28 | 0.93 | 0.33 | 1.09 |
| pi | electrostatic+steric | 69 | 0.3 | 0.1 | 0.42 | 0.83 | 0.61 | 1.09 |

## `_ens`

Data set `examples/hammett/hammett_ens.sdf`, charges gasteiger, uniform weights over the records of each substituent, 10-fold cross-validation.

`python benchmarks/run_hammett.py --sdf examples/hammett/hammett_ens.sdf`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | electrostatic+steric+hydrophobic+hba+hbd | 95 | 3 | 0.1 | 0.71 | 0.14 | 0.96 | 0.26 |
| sigma_m | electrostatic | 95 | 0.8 | 1 | 0.66 | 0.15 | 0.73 | 0.26 |
| sigma_m | steric | 95 | 1.2 | 0.01 | 0.18 | 0.24 | 0.32 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.27 | 0.05 | 0.26 |
| sigma_m | hba | 95 | 0.3 | 1 | 0.17 | 0.24 | 0.24 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | electrostatic+steric | 95 | 2 | 0.1 | 0.70 | 0.14 | 0.86 | 0.26 |
| sigma_p | electrostatic+steric+hydrophobic+hba+hbd | 95 | 0.5 | 1 | 0.69 | 0.21 | 0.82 | 0.38 |
| sigma_p | electrostatic | 95 | 0.8 | 1 | 0.69 | 0.22 | 0.75 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.01 | 0.13 | 0.36 | 0.28 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.02 | 0.39 | 0.06 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.24 | 0.33 | 0.41 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | electrostatic+steric | 95 | 1.2 | 1 | 0.69 | 0.21 | 0.77 | 0.38 |
| pi | electrostatic+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.73 | 0.56 | 0.83 | 1.09 |
| pi | electrostatic | 69 | 0.3 | 0.1 | 0.25 | 0.94 | 0.42 | 1.09 |
| pi | steric | 69 | 1.2 | 0.1 | 0.26 | 0.94 | 0.34 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.77 | 0.52 | 0.82 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.27 | 0.93 | 0.33 | 1.09 |
| pi | hbd | 69 | 0.5 | 1 | 0.28 | 0.92 | 0.34 | 1.09 |
| pi | electrostatic+steric | 69 | 0.3 | 0.1 | 0.39 | 0.85 | 0.56 | 1.09 |

## `_ens_boltzmann`

Data set `examples/hammett/hammett_ens.sdf`, charges gasteiger, boltzmann weights over the records of each substituent, 10-fold cross-validation.

`python benchmarks/run_hammett.py --sdf examples/hammett/hammett_ens.sdf --weighting boltzmann`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | electrostatic+steric+hydrophobic+hba+hbd | 95 | 3 | 0.1 | 0.73 | 0.14 | 0.96 | 0.26 |
| sigma_m | electrostatic | 95 | 0.8 | 1 | 0.66 | 0.15 | 0.73 | 0.26 |
| sigma_m | steric | 95 | 1.2 | 0.001 | 0.20 | 0.23 | 0.40 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.27 | 0.05 | 0.26 |
| sigma_m | hba | 95 | 1.2 | 1 | 0.17 | 0.24 | 0.29 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | electrostatic+steric | 95 | 2 | 0.1 | 0.68 | 0.15 | 0.86 | 0.26 |
| sigma_p | electrostatic+steric+hydrophobic+hba+hbd | 95 | 0.5 | 1 | 0.69 | 0.21 | 0.82 | 0.38 |
| sigma_p | electrostatic | 95 | 0.8 | 1 | 0.68 | 0.22 | 0.75 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.001 | 0.14 | 0.35 | 0.37 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.01 | 0.39 | 0.07 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.24 | 0.33 | 0.40 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | electrostatic+steric | 95 | 2 | 1 | 0.68 | 0.22 | 0.77 | 0.38 |
| pi | electrostatic+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.73 | 0.56 | 0.83 | 1.09 |
| pi | electrostatic | 69 | 0.3 | 0.1 | 0.27 | 0.93 | 0.44 | 1.09 |
| pi | steric | 69 | 1.2 | 0.1 | 0.26 | 0.93 | 0.36 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.77 | 0.52 | 0.82 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.27 | 0.93 | 0.34 | 1.09 |
| pi | hbd | 69 | 0.3 | 1 | 0.28 | 0.92 | 0.33 | 1.09 |
| pi | electrostatic+steric | 69 | 0.3 | 1 | 0.40 | 0.84 | 0.49 | 1.09 |

## `_ens_qm_q_hirshfeld`

Data set `examples/hammett/hammett_ens_qm.sdf`, charges prop:q_hirshfeld, uniform weights over the records of each substituent, 10-fold cross-validation.

`python benchmarks/run_hammett.py --sdf examples/hammett/hammett_ens_qm.sdf --charges-prop q_hirshfeld`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 2 | 0.1 | 0.90 | 0.08 | 0.97 | 0.26 |
| sigma_m | prop:q_hirshfeld | 95 | 0.8 | 0.01 | 0.87 | 0.09 | 0.94 | 0.26 |
| sigma_m | steric | 95 | 1.2 | 0.01 | 0.18 | 0.24 | 0.32 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.27 | 0.05 | 0.26 |
| sigma_m | hba | 95 | 0.3 | 1 | 0.17 | 0.24 | 0.24 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | prop:q_hirshfeld+steric | 95 | 1.2 | 0.1 | 0.86 | 0.10 | 0.93 | 0.26 |
| sigma_p | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 0.3 | 1 | 0.91 | 0.12 | 0.94 | 0.38 |
| sigma_p | prop:q_hirshfeld | 95 | 0.8 | 0.1 | 0.93 | 0.10 | 0.96 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.01 | 0.13 | 0.36 | 0.28 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.02 | 0.39 | 0.06 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.24 | 0.33 | 0.41 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | prop:q_hirshfeld+steric | 95 | 0.8 | 0.1 | 0.93 | 0.10 | 0.96 | 0.38 |
| pi | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.77 | 0.52 | 0.85 | 1.09 |
| pi | prop:q_hirshfeld | 69 | 0.3 | 0.1 | 0.32 | 0.90 | 0.48 | 1.09 |
| pi | steric | 69 | 1.2 | 0.1 | 0.26 | 0.94 | 0.34 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.77 | 0.52 | 0.82 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.27 | 0.93 | 0.33 | 1.09 |
| pi | hbd | 69 | 0.5 | 1 | 0.28 | 0.92 | 0.34 | 1.09 |
| pi | prop:q_hirshfeld+steric | 69 | 3 | 0.1 | 0.50 | 0.77 | 0.82 | 1.09 |

## `_ens_qm_q_hirshfeld_boltzmann`

Data set `examples/hammett/hammett_ens_qm.sdf`, charges prop:q_hirshfeld, boltzmann weights over the records of each substituent, 10-fold cross-validation.

`python benchmarks/run_hammett.py --sdf examples/hammett/hammett_ens_qm.sdf --charges-prop q_hirshfeld --weighting boltzmann`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 2 | 0.1 | 0.89 | 0.09 | 0.97 | 0.26 |
| sigma_m | prop:q_hirshfeld | 95 | 0.3 | 0.001 | 0.87 | 0.09 | 0.95 | 0.26 |
| sigma_m | steric | 95 | 1.2 | 0.001 | 0.20 | 0.23 | 0.40 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.27 | 0.05 | 0.26 |
| sigma_m | hba | 95 | 1.2 | 1 | 0.17 | 0.24 | 0.29 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | prop:q_hirshfeld+steric | 95 | 0.5 | 0.001 | 0.86 | 0.10 | 0.96 | 0.26 |
| sigma_p | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 0.3 | 1 | 0.90 | 0.12 | 0.94 | 0.38 |
| sigma_p | prop:q_hirshfeld | 95 | 1.2 | 0.1 | 0.93 | 0.10 | 0.96 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.001 | 0.14 | 0.35 | 0.37 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.01 | 0.39 | 0.07 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.24 | 0.33 | 0.40 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | prop:q_hirshfeld+steric | 95 | 0.8 | 0.1 | 0.92 | 0.11 | 0.96 | 0.38 |
| pi | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.77 | 0.52 | 0.85 | 1.09 |
| pi | prop:q_hirshfeld | 69 | 0.3 | 0.1 | 0.32 | 0.90 | 0.48 | 1.09 |
| pi | steric | 69 | 1.2 | 0.1 | 0.26 | 0.93 | 0.36 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.77 | 0.52 | 0.82 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.27 | 0.93 | 0.34 | 1.09 |
| pi | hbd | 69 | 0.3 | 1 | 0.28 | 0.92 | 0.33 | 1.09 |
| pi | prop:q_hirshfeld+steric | 69 | 0.8 | 0.001 | 0.52 | 0.76 | 0.90 | 1.09 |

## `_qm_sym_q_hirshfeld`

Data set `examples/hammett/hammett_qm_sym.sdf`, charges prop:q_hirshfeld, uniform weights over the records of each substituent, 10-fold cross-validation.

`python benchmarks/run_hammett.py --sdf examples/hammett/hammett_qm_sym.sdf --charges-prop q_hirshfeld`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 2 | 0.1 | 0.89 | 0.09 | 0.97 | 0.26 |
| sigma_m | prop:q_hirshfeld | 95 | 0.3 | 0.001 | 0.87 | 0.09 | 0.95 | 0.26 |
| sigma_m | steric | 95 | 1.2 | 0.01 | 0.19 | 0.24 | 0.33 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.27 | 0.05 | 0.26 |
| sigma_m | hba | 95 | 0.8 | 1 | 0.17 | 0.24 | 0.27 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | prop:q_hirshfeld+steric | 95 | 0.8 | 0.1 | 0.85 | 0.10 | 0.91 | 0.26 |
| sigma_p | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 0.3 | 1 | 0.90 | 0.12 | 0.94 | 0.38 |
| sigma_p | prop:q_hirshfeld | 95 | 0.8 | 0.1 | 0.92 | 0.11 | 0.95 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.01 | 0.14 | 0.36 | 0.29 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.01 | 0.39 | 0.07 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.24 | 0.33 | 0.40 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | prop:q_hirshfeld+steric | 95 | 0.5 | 0.1 | 0.92 | 0.11 | 0.95 | 0.38 |
| pi | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.78 | 0.51 | 0.85 | 1.09 |
| pi | prop:q_hirshfeld | 69 | 0.3 | 0.1 | 0.30 | 0.91 | 0.49 | 1.09 |
| pi | steric | 69 | 2 | 0.01 | 0.31 | 0.90 | 0.54 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.77 | 0.52 | 0.82 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.28 | 0.92 | 0.34 | 1.09 |
| pi | hbd | 69 | 0.3 | 1 | 0.28 | 0.93 | 0.33 | 1.09 |
| pi | prop:q_hirshfeld+steric | 69 | 0.8 | 0.001 | 0.55 | 0.73 | 0.90 | 1.09 |
