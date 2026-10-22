# Am/Eu separation: classic, indicator and quantum-chemical fields

46 extractants (Sitnikov et al. 2015), lg SF(Am/Eu) from -0.22 to +1.57 (sd 0.43); the structures carry Sybyl types, Gasteiger charges and no hydrogens, and are already superimposed.

Indicator fields: `Br`, `C.3`, `C.ar`, `N.ar`, `O.3`.
Same alpha and lambda grids and the same 10-fold CV throughout, each model choosing its own (alpha, lambda) on the CV it reports.

| model | q2 | RMSEcv | alpha | lambda |
|---|---:|---:|---:|---:|
| counts | -0.113 | 0.453 | - | 10 |
| classic | +0.803 | 0.191 | 1 | 0.01 |
| classic_rdkit_params | +0.807 | 0.189 | 0.5 | 0.01 |
| indicator | +0.760 | 0.211 | 0.5 | 0.1 |
| classic+indicator | +0.769 | 0.207 | 0.5 | 0.1 |
| classic_with_H | +0.775 | 0.204 | 1 | 0.01 |
| indicator_with_H | +0.760 | 0.211 | 0.5 | 0.1 |
| classic_matched | +0.775 | 0.204 | 1 | 0.01 |
| indicator_matched | +0.760 | 0.211 | 0.5 | 0.1 |
| quantum_charges | +0.803 | 0.191 | 0.3 | 0.01 |
| quantum_fukui | +0.788 | 0.198 | 0.3 | 0.1 |
| quantum_desc | +0.786 | 0.199 | 2 | 0.01 |
| quantum_all | +0.762 | 0.210 | 5 | 0.01 |
| cdft_fukui_plus_grid | +0.645 | 0.256 | - | 0.1 |
| cdft_fukui_plus_fitted | +0.673 | 0.246 | - | 0.1 |
| classic+cdft_fukui_plus_grid | +0.764 | 0.209 | 0.3 | 0.01 |
| classic+cdft_fukui_plus_fitted | +0.783 | 0.200 | 0.1 | 0.01 |
| cdft_dual_grid | +0.670 | 0.247 | - | 0.1 |
| cdft_dual_fitted | +0.675 | 0.245 | - | 0.1 |
| classic+cdft_dual_grid | +0.768 | 0.207 | 0.3 | 0.01 |
| classic+cdft_dual_fitted | +0.769 | 0.207 | 0.3 | 0.01 |

y-randomisation of `classic_rdkit_params`: real q2 +0.807, 200 shuffles mean -0.040, best +0.227, p = 0.0050.
