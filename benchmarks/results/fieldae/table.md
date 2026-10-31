# A field autoencoder with an analytic decoder (item E11)

The relative reconstruction error `||f - f'|| / ||f||` of a decoder of pseudo-atoms, against kernel PCA with the same number of latent dimensions. COMPAS-1D, field set ABCD of `run_compas.py` in the same scaling and the same random0 split; lower is better everywhere in this file.

8678 molecules, 637387 atoms (76 at most in one molecule), fields `steric`, `prop:q_xtb`, `cif:C.H`, `cif:C.2r`, `ring:terminal`, `ring:linear`, `ring:angular`, `ring:branched`, encoder input 1915 columns from 1915 landmark features.

The closed-form self-overlap of `cmfqsar.fieldae` agrees with the diagonal of the stored kernel of `run_compas.py` to 7.2e-08 relative, worst of 8678 molecules.

## Gate (a): the autoencoder against kernel PCA

The code dimension, at 32 pseudo-atoms. 6942 training and 868 held-out molecules.

| latent dimensions | autoencoder, mean | autoencoder, median | kernel PCA, mean | kernel PCA, median | the autoencoder is |
|---:|---:|---:|---:|---:|---|
| 2 | 0.2907 | 0.2798 | 0.3398 | 0.3301 | better by 0.0491 |
| 4 | 0.2166 | 0.2035 | 0.2847 | 0.2761 | better by 0.0681 |
| 8 | 0.1324 | 0.1186 | 0.2158 | 0.2084 | better by 0.0835 |
| 16 | 0.0832 | 0.0738 | 0.1452 | 0.1422 | better by 0.0620 |
| 32 | 0.0697 | 0.0631 | 0.0821 | 0.0794 | better by 0.0123 |

## The number of pseudo-atoms, at code 8

| pseudo-atoms | mean | median | 90th percentile | worst | closest two pseudo-atoms, A | epochs |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.1996 | 0.1934 | 0.2368 | 1.2742 | 2.09 | 150 |
| 16 | 0.1469 | 0.1351 | 0.2116 | 1.5716 | 0.35 | 150 |
| 32 | 0.1324 | 0.1186 | 0.1931 | 1.6798 | 0.17 | 150 |
| 64 | 0.1355 | 0.1251 | 0.1945 | 0.8516 | 0.12 | 133 |

## The floor: the same pseudo-atoms fitted to one molecule directly

No code, no sharing between molecules: the best these pseudo-atoms can do on a field of this set. It says whether a decoder's error is the bottleneck of the code or the limit of the representation.

| pseudo-atoms | mean | median | worst | molecules | atoms per molecule |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.1875 | 0.1878 | 0.2728 | 40 | 56 |
| 16 | 0.0915 | 0.0727 | 0.6033 | 40 | 56 |
| 32 | 0.0434 | 0.0319 | 0.2302 | 40 | 56 |
| 64 | 0.0732 | 0.0205 | 1.0566 | 40 | 56 |

## Gate (b): the reconstruction error as an applicability domain

The model of record predicts the held-out part with R2 0.593 and RMSE 0.263 eV. Every estimate below is scored against the error that model actually made, under the protocol of item E5: rank correlation (rho), risk-coverage efficiency (1 = the ordering the errors themselves give, 0 = no better than ignoring the estimate), and the RMSE of the half it trusts against the half it doubts. The one-class scores are fitted on 6942 training molecules.

| estimate | rho | tau | efficiency | RMSE trusted | RMSE doubted |
|---|---:|---:|---:|---:|---:|
| Distance to the centre of the actives (similarity ensemble) | +0.120 | +0.079 | +0.566 | 0.147 | 0.342 |
| One-class SVM (1-SVM) | +0.052 | +0.035 | +0.531 | 0.156 | 0.338 |
| Distance from the principal subspace (kernel PCA) | +0.001 | +0.001 | +0.519 | 0.157 | 0.337 |
| Distance to the k-th nearest active | +0.061 | +0.042 | +0.507 | 0.153 | 0.339 |
| One-class Gaussian process | +0.050 | +0.033 | +0.497 | 0.159 | 0.336 |
| field autoencoder, relative | -0.059 | -0.039 | +0.467 | 0.167 | 0.332 |
| field autoencoder, absolute | -0.056 | -0.037 | +0.466 | 0.166 | 0.333 |

The ordering the errors themselves give has an area under the risk-coverage curve of 0.063; ignoring the estimate gives 0.263.

