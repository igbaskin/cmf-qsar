# A field-based Hammett equation

One table per run of `benchmarks/run_hammett.py`, named by the tag of its summary file: no tag is `hammett.sdf` with one conformer and Gasteiger charges; `_ens` the conformer ensemble; `_q_hirshfeld` the Hirshfeld charges of the quantum stage; `_sym` the four symmetry images; `_boltzmann` Boltzmann instead of uniform weights. q² and RMSEcv by 10-fold cross-validation, α and λ chosen on it.

## `(single conformer, Gasteiger charges)`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | electrostatic+steric+hydrophobic+hba+hbd | 95 | 0.3 | 1 | 0.70 | 0.14 | 0.80 | 0.26 |
| sigma_m | electrostatic | 95 | 0.8 | 1 | 0.67 | 0.15 | 0.73 | 0.26 |
| sigma_m | steric | 95 | 1.2 | 0.01 | 0.19 | 0.23 | 0.33 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.26 | 0.06 | 0.26 |
| sigma_m | hba | 95 | 0.3 | 1 | 0.18 | 0.24 | 0.24 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | electrostatic+steric | 95 | 2 | 1 | 0.68 | 0.15 | 0.77 | 0.26 |
| sigma_p | electrostatic+steric+hydrophobic+hba+hbd | 95 | 0.5 | 1 | 0.69 | 0.21 | 0.82 | 0.38 |
| sigma_p | electrostatic | 95 | 0.8 | 1 | 0.68 | 0.22 | 0.75 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.01 | 0.14 | 0.36 | 0.30 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.01 | 0.39 | 0.07 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.23 | 0.34 | 0.40 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | electrostatic+steric | 95 | 2 | 1 | 0.68 | 0.22 | 0.77 | 0.38 |
| pi | electrostatic+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.72 | 0.58 | 0.82 | 1.09 |
| pi | electrostatic | 69 | 1.2 | 0.01 | 0.36 | 0.87 | 0.73 | 1.09 |
| pi | steric | 69 | 0.5 | 0.001 | 0.30 | 0.91 | 0.48 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.72 | 0.58 | 0.79 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.27 | 0.93 | 0.34 | 1.09 |
| pi | hbd | 69 | 0.3 | 1 | 0.28 | 0.93 | 0.33 | 1.09 |
| pi | electrostatic+steric | 69 | 0.3 | 0.1 | 0.43 | 0.82 | 0.61 | 1.09 |

## `_ens`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | electrostatic+steric+hydrophobic+hba+hbd | 95 | 0.5 | 1 | 0.70 | 0.14 | 0.82 | 0.26 |
| sigma_m | electrostatic | 95 | 1.2 | 1 | 0.65 | 0.15 | 0.73 | 0.26 |
| sigma_m | steric | 95 | 0.8 | 0.001 | 0.19 | 0.24 | 0.36 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.27 | 0.05 | 0.26 |
| sigma_m | hba | 95 | 0.3 | 1 | 0.18 | 0.24 | 0.24 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | electrostatic+steric | 95 | 2 | 0.1 | 0.68 | 0.15 | 0.85 | 0.26 |
| sigma_p | electrostatic+steric+hydrophobic+hba+hbd | 95 | 0.5 | 1 | 0.69 | 0.21 | 0.82 | 0.38 |
| sigma_p | electrostatic | 95 | 0.8 | 1 | 0.68 | 0.22 | 0.74 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.01 | 0.13 | 0.36 | 0.28 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.01 | 0.39 | 0.07 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.23 | 0.34 | 0.40 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | electrostatic+steric | 95 | 2 | 1 | 0.68 | 0.22 | 0.77 | 0.38 |
| pi | electrostatic+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.71 | 0.59 | 0.82 | 1.09 |
| pi | electrostatic | 69 | 0.3 | 0.1 | 0.27 | 0.93 | 0.43 | 1.09 |
| pi | steric | 69 | 1.2 | 0.1 | 0.26 | 0.94 | 0.35 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.71 | 0.58 | 0.80 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.26 | 0.93 | 0.33 | 1.09 |
| pi | hbd | 69 | 0.5 | 1 | 0.28 | 0.92 | 0.34 | 1.09 |
| pi | electrostatic+steric | 69 | 0.3 | 0.1 | 0.40 | 0.84 | 0.56 | 1.09 |

## `_ens_boltzmann`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | electrostatic+steric+hydrophobic+hba+hbd | 95 | 0.5 | 1 | 0.70 | 0.14 | 0.82 | 0.26 |
| sigma_m | electrostatic | 95 | 1.2 | 1 | 0.66 | 0.15 | 0.73 | 0.26 |
| sigma_m | steric | 95 | 1.2 | 0.001 | 0.20 | 0.23 | 0.41 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.26 | 0.06 | 0.26 |
| sigma_m | hba | 95 | 0.3 | 1 | 0.18 | 0.24 | 0.24 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | electrostatic+steric | 95 | 2 | 1 | 0.68 | 0.15 | 0.77 | 0.26 |
| sigma_p | electrostatic+steric+hydrophobic+hba+hbd | 95 | 0.5 | 1 | 0.69 | 0.22 | 0.82 | 0.38 |
| sigma_p | electrostatic | 95 | 0.8 | 1 | 0.68 | 0.22 | 0.74 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.001 | 0.14 | 0.36 | 0.37 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.01 | 0.39 | 0.07 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.23 | 0.34 | 0.40 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | electrostatic+steric | 95 | 2 | 1 | 0.68 | 0.22 | 0.77 | 0.38 |
| pi | electrostatic+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.70 | 0.59 | 0.82 | 1.09 |
| pi | electrostatic | 69 | 0.3 | 0.1 | 0.28 | 0.92 | 0.45 | 1.09 |
| pi | steric | 69 | 1.2 | 0.1 | 0.27 | 0.93 | 0.37 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.71 | 0.58 | 0.79 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.27 | 0.93 | 0.34 | 1.09 |
| pi | hbd | 69 | 0.3 | 1 | 0.28 | 0.92 | 0.33 | 1.09 |
| pi | electrostatic+steric | 69 | 0.3 | 1 | 0.40 | 0.84 | 0.49 | 1.09 |

## `_ens_qm_q_hirshfeld`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 1.2 | 0.01 | 0.76 | 0.13 | 0.98 | 0.26 |
| sigma_m | prop:q_hirshfeld | 95 | 0.8 | 0.001 | 0.68 | 0.15 | 0.94 | 0.26 |
| sigma_m | steric | 95 | 0.8 | 0.001 | 0.19 | 0.24 | 0.36 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.27 | 0.05 | 0.26 |
| sigma_m | hba | 95 | 0.3 | 1 | 0.18 | 0.24 | 0.24 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | prop:q_hirshfeld+steric | 95 | 1.2 | 0.001 | 0.67 | 0.15 | 0.97 | 0.26 |
| sigma_p | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 2 | 0.01 | 0.80 | 0.17 | 0.99 | 0.38 |
| sigma_p | prop:q_hirshfeld | 95 | 3 | 0.01 | 0.73 | 0.20 | 0.96 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.01 | 0.13 | 0.36 | 0.28 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.01 | 0.39 | 0.07 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.23 | 0.34 | 0.40 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | prop:q_hirshfeld+steric | 95 | 0.8 | 0.1 | 0.70 | 0.21 | 0.91 | 0.38 |
| pi | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.73 | 0.57 | 0.84 | 1.09 |
| pi | prop:q_hirshfeld | 69 | 0.3 | 0.1 | 0.35 | 0.88 | 0.48 | 1.09 |
| pi | steric | 69 | 1.2 | 0.1 | 0.26 | 0.94 | 0.35 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.71 | 0.58 | 0.80 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.26 | 0.93 | 0.33 | 1.09 |
| pi | hbd | 69 | 0.5 | 1 | 0.28 | 0.92 | 0.34 | 1.09 |
| pi | prop:q_hirshfeld+steric | 69 | 3 | 0.1 | 0.54 | 0.74 | 0.82 | 1.09 |

## `_ens_qm_q_hirshfeld_boltzmann`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 0.8 | 0.001 | 0.79 | 0.12 | 0.99 | 0.26 |
| sigma_m | prop:q_hirshfeld | 95 | 0.8 | 0.001 | 0.68 | 0.15 | 0.94 | 0.26 |
| sigma_m | steric | 95 | 1.2 | 0.001 | 0.20 | 0.23 | 0.41 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.26 | 0.06 | 0.26 |
| sigma_m | hba | 95 | 0.3 | 1 | 0.18 | 0.24 | 0.24 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | prop:q_hirshfeld+steric | 95 | 0.8 | 0.001 | 0.63 | 0.16 | 0.96 | 0.26 |
| sigma_p | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 3 | 0.01 | 0.79 | 0.17 | 1.00 | 0.38 |
| sigma_p | prop:q_hirshfeld | 95 | 0.8 | 0.01 | 0.72 | 0.20 | 0.93 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.001 | 0.14 | 0.36 | 0.37 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.01 | 0.39 | 0.07 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.23 | 0.34 | 0.40 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | prop:q_hirshfeld+steric | 95 | 0.8 | 0.1 | 0.70 | 0.21 | 0.91 | 0.38 |
| pi | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.74 | 0.56 | 0.84 | 1.09 |
| pi | prop:q_hirshfeld | 69 | 0.3 | 0.1 | 0.34 | 0.88 | 0.48 | 1.09 |
| pi | steric | 69 | 1.2 | 0.1 | 0.27 | 0.93 | 0.37 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.71 | 0.58 | 0.79 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.27 | 0.93 | 0.34 | 1.09 |
| pi | hbd | 69 | 0.3 | 1 | 0.28 | 0.92 | 0.33 | 1.09 |
| pi | prop:q_hirshfeld+steric | 69 | 0.8 | 0.001 | 0.56 | 0.72 | 0.90 | 1.09 |

## `_qm_sym_q_hirshfeld`

| target | fields | n | α | λ | q² | RMSEcv | r² fit | sd(y) |
|---|---|---|---|---|---|---|---|---|
| sigma_m | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 0.8 | 0.001 | 0.79 | 0.12 | 0.99 | 0.26 |
| sigma_m | prop:q_hirshfeld | 95 | 0.5 | 0.001 | 0.65 | 0.15 | 0.92 | 0.26 |
| sigma_m | steric | 95 | 1.2 | 0.01 | 0.19 | 0.23 | 0.33 | 0.26 |
| sigma_m | hydrophobic | 95 | 3 | 10 | -0.03 | 0.26 | 0.06 | 0.26 |
| sigma_m | hba | 95 | 0.3 | 1 | 0.18 | 0.24 | 0.24 | 0.26 |
| sigma_m | hbd | 95 | 0.3 | 10 | -0.02 | 0.26 | 0.03 | 0.26 |
| sigma_m | prop:q_hirshfeld+steric | 95 | 0.8 | 0.1 | 0.61 | 0.16 | 0.86 | 0.26 |
| sigma_p | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 95 | 3 | 0.01 | 0.78 | 0.18 | 1.00 | 0.38 |
| sigma_p | prop:q_hirshfeld | 95 | 0.8 | 0.01 | 0.71 | 0.21 | 0.93 | 0.38 |
| sigma_p | steric | 95 | 1.2 | 0.01 | 0.14 | 0.36 | 0.30 | 0.38 |
| sigma_p | hydrophobic | 95 | 3 | 10 | -0.01 | 0.39 | 0.07 | 0.38 |
| sigma_p | hba | 95 | 3 | 1 | 0.23 | 0.34 | 0.40 | 0.38 |
| sigma_p | hbd | 95 | 3 | 10 | 0.02 | 0.38 | 0.09 | 0.38 |
| sigma_p | prop:q_hirshfeld+steric | 95 | 0.5 | 0.1 | 0.70 | 0.21 | 0.90 | 0.38 |
| pi | prop:q_hirshfeld+steric+hydrophobic+hba+hbd | 69 | 0.3 | 1 | 0.74 | 0.55 | 0.84 | 1.09 |
| pi | prop:q_hirshfeld | 69 | 0.3 | 0.1 | 0.33 | 0.89 | 0.49 | 1.09 |
| pi | steric | 69 | 0.5 | 0.001 | 0.30 | 0.91 | 0.48 | 1.09 |
| pi | hydrophobic | 69 | 0.3 | 0.1 | 0.72 | 0.58 | 0.79 | 1.09 |
| pi | hba | 69 | 0.5 | 1 | 0.27 | 0.93 | 0.34 | 1.09 |
| pi | hbd | 69 | 0.3 | 1 | 0.28 | 0.93 | 0.33 | 1.09 |
| pi | prop:q_hirshfeld+steric | 69 | 0.8 | 0.001 | 0.60 | 0.68 | 0.90 | 1.09 |
