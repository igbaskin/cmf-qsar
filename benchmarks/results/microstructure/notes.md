# A microstructure as the sample: notes on the pilot

`benchmarks/run_microstructure.py`, 7 November 2026. The question: a Continuous Molecular Field
is a sum of Gaussians and its kernel is the overlap of two fields; a microstructure is a field
too. Does the field-overlap kernel, computed on a voxel grid, match or beat the two-point
statistics workflow of the Materials Knowledge Systems (MKS, Kalidindi and co-workers) on an
effective property, and at which training sizes?

## What was built

**The data.** Synthetic two-phase microstructures on a periodic voxel grid, three data sets of
400 samples each: 2D, 51 x 51, contrast 10; 2D, 51 x 51, contrast 100; 3D, 21 x 21 x 21,
contrast 10. The generator is the recipe of `pymks.generate_multiphase`: white noise filtered
with a Gaussian whose width is the correlation length (the grain size), then cut at the quantile
that gives the phase fraction. Four things vary from sample to sample: the phase fraction
(uniform 0.2 to 0.8), the grain size (1.5 to 8 voxels, log-uniform), the elongation of the grains
(aspect ratio 1 to 3) and their direction (random). The elongation is there so that a
direction-dependent property exists.

**The label.** The effective thermal conductivity of the periodic cell, phase 0 at 1 and phase 1
at the contrast. `pymks` 0.4.1 does not install here: its build script calls a `configparser`
class that Python 3.13 no longer has, and its finite-element elastic solver needs `sfepy`. The
script therefore has its own solver, as the task allowed: finite volumes on the voxel grid,
harmonic-mean conductivity on every face between two voxels, a unit mean temperature gradient
applied across the periodic cell, one sparse direct solve per direction, one voxel pinned
because the equation fixes the temperature only up to a constant. It is checked on every run
against the exact results for a uniform cell and for a laminate (the series mean across the
layers, the parallel mean along them), to 1e-12, and the tensor it returns is symmetric to
1e-15. Two labels are fitted: `k_xx`, the component along x, which depends on the direction of
the grains; and the mean of the diagonal, which does not. The data are cached under
`benchmarks/cache/microstructure/` (about 3 MB for the three sets); building them takes 4 s per
2D set and 5 min for the 3D set.

**The descriptions.** All of them computed with numpy FFTs on the grid; the atom-centred
Gaussian engine is not used. Two channels per sample, the indicator of phase 0 and of phase 1,
each optionally smoothed with a periodic Gaussian of width `s` in {0, 1, 2, 3, 5} voxels.

1. *Overlap, generating frame.* The plain CMF kernel: the overlap of the smoothed phase fields
   at zero shift, `K_ij = sum_x sum_c m_c^i(x) m_c^j(x)`.
2. *Overlap, rotation-averaged.* The same, averaged over the point group of the grid (8
   operations in 2D, 48 in 3D).
3. *Autocorrelation, linear.* The translation-averaged route. Each sample's two-point
   autocorrelation `S_ab(r) = (1/n) sum_x m_a(x) m_b(x + r)` (the self-overlap of the field with
   its own translate; the channel pairs 1-1 and 0-1, the pair 0-0 follows from them) and the
   kernel is the overlap of two samples' autocorrelations summed over `r`. By Parseval's theorem
   this is the overlap of their power spectra.
4. *Autocorrelation, normalised.* The package's `autocorrelation_kernel` on the same
   descriptor: one scale per channel block, cosine-normalised.
5. *Autocorrelation, lattice-rotation-averaged.* The autocorrelation averaged over the point
   group of the grid, then the overlap.
6. *Autocorrelation, radial.* The autocorrelation binned by the length of `r` (the average over
   every rotation), weighted so that the linear kernel is the grid sum of the products of the
   rotation-averaged statistics.
7. *Two-point statistics + PCA.* The MKS homogenisation baseline: the autocorrelation and the
   cross-correlation of the raw indicator, principal components fitted on the training part
   (2 to 20), a polynomial of the standardised scores of degree 1 or 2, fitted by ridge
   regression.
8. *Phase fraction.* A cubic polynomial of the fraction of phase 1 alone. This is the
   composition baseline: what any description of the arrangement has to add to.

**The fit.** Kernel ridge regression with a bias term from `cmfqsar.model` (`krr_fit` through
`_Learner`, `cv_predict` for the inner folds, `median_gamma_scale` for the Gaussian layer).
Every kernel is tried linear and under a Gaussian layer with bandwidth 0.5, 1 or 2 times the
median squared distance; the ridge is 1e-6 to 1 times the mean self-kernel. The smoothing
width, the layer and the ridge (for the baseline: the number of components, the degree and the
ridge) are chosen by five-fold cross-validation inside each training set. Training sets of 20,
50, 100 and 200 samples are drawn five times at random; the rest of the 400 are held out. `q2`
is R2 on the held-out samples and RMSE is in units of the conductivity of phase 0; the tables
give the mean over the five draws and, in brackets, the standard deviation over them.

**One identity worth stating.** The two-point statistic of MKS is the translation-averaged
self-overlap of the phase field, so method 3 is the CMF overlap kernel applied to that
statistic. But the plain translation average of the *overlap kernel between two samples*
(the mean over all relative shifts of method 1) is only `n f_i f_j`, the product of their
phase fractions: averaging the kernel over shifts throws the arrangement away, averaging each
field's self-overlap over shifts keeps it. The invariant route of this program is the second.

## Run time

The whole script, from an empty cache, on the author's machine (32 logical processors): 8 min
for the 2D set at contrast 10, 7 min for the one at contrast 100, and 8 min for the 3D set, of
which 5 min is building and solving the cells and 3 min the fit. The kernels take seconds; the
time is the nested cross-validation, which fits about 140 settings per kernel method and draw.
The first full run took 68 min on the 3D fit because another long Python job was using the
processors at the same time; the 3D fit was rerun alone and took 3 min, with the same numbers.
The run time in the 3D block of the tables below is that of the rerun of the fit stage.

## Conclusion

**Does the field-overlap kernel match or beat the two-point-statistics baseline?** Yes, through
the translation-averaged route, and from the smallest training size on. On the
direction-dependent label `k_xx` the overlap of the two-point autocorrelations (method 3) gives,
against the MKS baseline (method 7), q2 of:

| data set | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---:|---:|---:|---:|
| 2D, contrast 10: autocorrelation kernel / baseline | 0.89 / 0.93 | 0.93 / 0.95 | 0.97 / 0.97 | 0.98 / 0.97 |
| 2D, contrast 100: autocorrelation kernel / baseline | 0.81 / 0.81 | 0.83 / 0.79 | 0.89 / 0.88 | 0.90 / 0.88 |
| 3D, contrast 10: autocorrelation kernel / baseline | 0.97 / 0.96 | 0.98 / 0.98 | 0.98 / 0.98 | 0.99 / 0.99 |

So the kernel is within the spread over draws of the baseline at 20 and 50 samples (behind it in
2D at contrast 10, ahead of it at contrast 100 and in 3D), and ahead of it or equal from 100
samples on in every set. The gain is largest where the contrast makes the arrangement matter
most (contrast 100: 0.90 against 0.88 at 200 samples, RMSE 5.6 against 6.1). Both are well above
the phase fraction alone (0.78 to 0.89 at 200 samples), so both read the arrangement; the kernel
reads a little more of it, because it uses the whole statistic where the baseline keeps 10 to
20 principal components of it.

**The zero-shift overlap in the generating frame is poor**, and the reason is the frame, not the
kernel. These samples share a grid and nothing else: their origins are random, so the plain
overlap compares two structures at a random relative shift. At 20 to 50 training samples it is
below the phase fraction alone in 2D; it climbs to 0.90 (2D) and 0.96 (3D) at 200 samples, still
behind the baseline, and it always chooses the widest smoothing, at which the kernel keeps little
beyond the fractions. A microstructure benchmark has no alignment step, and the translation
average is what replaces it. This is the same lesson as the periodic autocorrelation of crystals
in this program: where no frame exists, average the self-overlap over the frame, do not average
the kernel.

**The rotation averages** (methods 2, 5, 6) lose the direction of the grains and with it the
direction-dependent label: on `k_xx` they fall to the level of the phase fraction (0.79 to 0.89).
On the direction-free label they recover everything and are the best or equal-best descriptions
in 3D (0.996 at 200 samples). They are the right invariance for a property that has it and the
wrong one for a property that does not, as a rotation-invariant kernel on molecules is.

**On the direction-free label** every description is within a few thousandths of the phase
fraction alone at contrast 10 (0.98 in 2D, 0.99 in 3D), which is the known situation of the
published homogenisation examples: the composition explains almost all of an effective property
at low contrast. At contrast 100 the autocorrelation kernels gain a little over the fraction
(0.93 against 0.92 at 200 samples) and equal the baseline.

**The smoothing width that worked best.** For the autocorrelation kernels on `k_xx`, zero: the
inner cross-validation chose no smoothing in four or five of the five draws at every size from
50 on, in all three sets. The two-point statistic wants its full resolution; a Gaussian of 1 to 5
voxels on the field is a low-pass filter on the power spectrum, and the arrangement lives in the
part it removes. For the zero-shift overlap, the widest width tried, 5 voxels, which is the size
of the larger grains; there the smoothing is what lets two unaligned structures overlap at all.
On the direction-free label the autocorrelation kernels spread their choice over 0 to 5 voxels
with no clear preference, because the fraction, which every width keeps, is most of the signal.
For a hierarchical (multi-scale) kernel this says: carry the fine scale for the arrangement and a
coarse scale for the composition, as two channels with their own widths, rather than one width
for both. That is the construction this program already uses for two field types.

**Two cautions.** The spread over draws of the kernel methods at 50 training samples is large in
places (0.06 in 2D at contrast 10, 0.33 for the lattice-averaged kernel at contrast 100): the
inner cross-validation chooses among 140 settings per kernel method on 50 samples and sometimes
chooses badly, where the baseline chooses among 40. A smaller grid would make the kernel methods
steadier at the smallest sizes. And the Gaussian layer adds little here: at 200 samples the
chosen layer is the linear kernel or the narrowest Gaussian in every set.

**What this does not show.** The data are synthetic and the property is a scalar conduction
problem on a periodic cell. A real micrograph is one window on a non-periodic medium, so the
autocorrelation would be computed with padding and a mask; the solver would be a finite-element
elasticity or plasticity code; and a hierarchy of scales would need the kernel across scales,
not only the width within one. None of those is tested here. What is tested is the claim that
the method applies: a microstructure is a field, the overlap of fields is its kernel, and on the
smallest fair benchmark that kernel is at least as good as the workflow built for the purpose.

## The tables

What follows is `table.md` of this folder, as `run_microstructure.py --report` writes it.

Synthetic two-phase microstructures on a periodic voxel grid (Gaussian-filtered noise cut at the phase fraction; the phase fraction, the grain size, the aspect ratio and the direction of the grains vary from sample to sample). The label is the effective thermal conductivity of the periodic cell from a finite-volume solver (phase 0 at 1, phase 1 at the contrast): the component along x, which depends on the direction of the grains, and the mean over the directions, which does not. Training sets of each size drawn five times at random, the rest of the set held out; every setting chosen by five-fold cross-validation inside the training part. q2 is R2 on the held-out samples, mean over the five draws with the spread in brackets; RMSE is in units of the conductivity of phase 0.

### 2D, 51 points per side, 400 samples, contrast 10

Labels: k_xx from 1.29 to 7.30 (sd 1.555), mean from 1.51 to 6.83 (sd 1.437). Solver checks: uniform phase 1: largest error 0.0e+00; laminate, layers across x: largest error 1.2e-13; random structure, asymmetry of the tensor 2.2e-16. Settings searched: smoothing width in {0, 1, 2, 3, 5} voxels; Gaussian layer bandwidth in {0.5, 1, 2} times the median squared distance, or linear; ridge in {1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1} times the mean self-kernel; for the baseline {2, 3, 5, 10, 20} components, degree {1, 2}, ridge {1e-08, 0.0001, 0.01, 1}. Run time 8 min.

**Label: the component along x (direction-dependent).**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.867 (0.028) | +0.875 (0.008) | +0.884 (0.008) | +0.889 (0.009) | 0.565 (0.059) | 0.545 (0.015) | 0.531 (0.017) | 0.522 (0.026) |
| two-point statistics + PCA | +0.930 (0.014) | +0.950 (0.009) | +0.965 (0.003) | +0.968 (0.003) | 0.408 (0.042) | 0.343 (0.031) | 0.291 (0.010) | 0.279 (0.012) |
| overlap, generating frame | +0.812 (0.042) | +0.845 (0.021) | +0.879 (0.013) | +0.898 (0.017) | 0.670 (0.070) | 0.607 (0.043) | 0.542 (0.027) | 0.499 (0.046) |
| overlap, rotation-averaged | +0.807 (0.028) | +0.823 (0.030) | +0.862 (0.006) | +0.870 (0.012) | 0.681 (0.054) | 0.648 (0.053) | 0.581 (0.018) | 0.564 (0.031) |
| autocorrelation, linear | +0.890 (0.042) | +0.927 (0.062) | +0.971 (0.003) | +0.979 (0.002) | 0.507 (0.091) | 0.395 (0.153) | 0.265 (0.014) | 0.228 (0.014) |
| autocorrelation, normalised | +0.912 (0.024) | +0.931 (0.055) | +0.972 (0.002) | +0.979 (0.002) | 0.456 (0.058) | 0.387 (0.142) | 0.261 (0.009) | 0.229 (0.012) |
| autocorrelation, lattice-rotation-averaged | +0.868 (0.034) | +0.826 (0.122) | +0.890 (0.009) | +0.893 (0.011) | 0.561 (0.070) | 0.619 (0.192) | 0.517 (0.019) | 0.511 (0.026) |
| autocorrelation, radial | +0.864 (0.043) | +0.836 (0.100) | +0.891 (0.009) | +0.894 (0.011) | 0.567 (0.086) | 0.606 (0.164) | 0.515 (0.018) | 0.511 (0.026) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (five draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (5) | 2 (1), 5 (4) | 2 (1), 5 (4) | 2 (1), 3 (1), 5 (3) |
| overlap, rotation-averaged | 1 (1), 2 (1), 5 (3) | 0 (3), 3 (1), 5 (1) | 1 (1), 5 (4) | 0 (2), 1 (2), 5 (1) |
| autocorrelation, linear | 0 (1), 2 (1), 3 (1), 5 (2) | 0 (5) | 0 (5) | 0 (5) |
| autocorrelation, normalised | 0 (2), 2 (1), 5 (2) | 0 (5) | 0 (5) | 0 (5) |
| autocorrelation, lattice-rotation-averaged | 3 (1), 5 (4) | 2 (1), 5 (4) | 3 (1), 5 (4) | 0 (1), 2 (1), 5 (3) |
| autocorrelation, radial | 0 (1), 3 (1), 5 (3) | 5 (5) | 0 (1), 3 (2), 5 (2) | 0 (1), 1 (1), 5 (3) |

**Label: the mean over the directions (direction-free).**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.982 (0.001) | +0.983 (0.001) | +0.984 (0.001) | +0.985 (0.002) | 0.194 (0.005) | 0.185 (0.006) | 0.183 (0.006) | 0.175 (0.013) |
| two-point statistics + PCA | +0.980 (0.003) | +0.982 (0.002) | +0.984 (0.002) | +0.986 (0.002) | 0.202 (0.014) | 0.192 (0.011) | 0.179 (0.009) | 0.170 (0.009) |
| overlap, generating frame | +0.931 (0.024) | +0.959 (0.006) | +0.965 (0.004) | +0.973 (0.003) | 0.374 (0.066) | 0.288 (0.019) | 0.268 (0.014) | 0.238 (0.012) |
| overlap, rotation-averaged | +0.934 (0.027) | +0.967 (0.009) | +0.974 (0.004) | +0.982 (0.001) | 0.361 (0.078) | 0.258 (0.033) | 0.230 (0.014) | 0.193 (0.002) |
| autocorrelation, linear | +0.980 (0.002) | +0.974 (0.012) | +0.984 (0.002) | +0.987 (0.002) | 0.205 (0.010) | 0.224 (0.048) | 0.183 (0.008) | 0.166 (0.009) |
| autocorrelation, normalised | +0.971 (0.011) | +0.973 (0.010) | +0.984 (0.001) | +0.986 (0.001) | 0.240 (0.042) | 0.232 (0.042) | 0.182 (0.006) | 0.170 (0.008) |
| autocorrelation, lattice-rotation-averaged | +0.976 (0.008) | +0.979 (0.005) | +0.984 (0.002) | +0.986 (0.002) | 0.223 (0.035) | 0.205 (0.023) | 0.181 (0.011) | 0.169 (0.009) |
| autocorrelation, radial | +0.968 (0.017) | +0.976 (0.009) | +0.984 (0.002) | +0.986 (0.002) | 0.249 (0.067) | 0.218 (0.040) | 0.182 (0.011) | 0.169 (0.009) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (five draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (5) | 5 (5) | 2 (3), 5 (2) | 1 (1), 2 (2), 5 (2) |
| overlap, rotation-averaged | 1 (1), 3 (1), 5 (3) | 5 (5) | 3 (1), 5 (4) | 5 (5) |
| autocorrelation, linear | 3 (2), 5 (3) | 0 (2), 2 (1), 5 (2) | 0 (1), 1 (1), 3 (1), 5 (2) | 0 (2), 2 (1), 3 (1), 5 (1) |
| autocorrelation, normalised | 3 (1), 5 (4) | 0 (2), 5 (3) | 3 (1), 5 (4) | 2 (1), 3 (1), 5 (3) |
| autocorrelation, lattice-rotation-averaged | 0 (1), 5 (4) | 3 (2), 5 (3) | 0 (1), 5 (4) | 0 (2), 5 (3) |
| autocorrelation, radial | 2 (1), 3 (2), 5 (2) | 0 (1), 5 (4) | 3 (1), 5 (4) | 0 (2), 3 (1), 5 (2) |

### 2D, 51 points per side, 400 samples, contrast 100

Labels: k_xx from 1.33 to 68.14 (sd 17.604), mean from 1.71 to 61.39 (sd 15.527). Solver checks: uniform phase 1: largest error 0.0e+00; laminate, layers across x: largest error 7.6e-13; random structure, asymmetry of the tensor 1.3e-15. Settings searched: smoothing width in {0, 1, 2, 3, 5} voxels; Gaussian layer bandwidth in {0.5, 1, 2} times the median squared distance, or linear; ridge in {1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1} times the mean self-kernel; for the baseline {2, 3, 5, 10, 20} components, degree {1, 2}, ridge {1e-08, 0.0001, 0.01, 1}. Run time 7 min.

**Label: the component along x (direction-dependent).**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.739 (0.048) | +0.744 (0.023) | +0.766 (0.017) | +0.779 (0.020) | 8.938 (0.832) | 8.822 (0.381) | 8.568 (0.352) | 8.384 (0.464) |
| two-point statistics + PCA | +0.805 (0.039) | +0.788 (0.063) | +0.876 (0.012) | +0.883 (0.009) | 7.701 (0.729) | 7.979 (1.222) | 6.218 (0.347) | 6.104 (0.242) |
| overlap, generating frame | +0.657 (0.042) | +0.694 (0.034) | +0.756 (0.020) | +0.795 (0.034) | 10.250 (0.640) | 9.653 (0.627) | 8.737 (0.366) | 8.059 (0.738) |
| overlap, rotation-averaged | +0.614 (0.069) | +0.635 (0.076) | +0.736 (0.021) | +0.754 (0.025) | 10.852 (1.018) | 10.506 (1.022) | 9.092 (0.423) | 8.840 (0.559) |
| autocorrelation, linear | +0.811 (0.028) | +0.827 (0.028) | +0.893 (0.008) | +0.902 (0.014) | 7.600 (0.582) | 7.245 (0.632) | 5.787 (0.230) | 5.562 (0.498) |
| autocorrelation, normalised | +0.787 (0.056) | +0.821 (0.068) | +0.891 (0.016) | +0.907 (0.006) | 8.031 (1.140) | 7.306 (1.381) | 5.840 (0.491) | 5.450 (0.282) |
| autocorrelation, lattice-rotation-averaged | +0.713 (0.061) | +0.621 (0.325) | +0.785 (0.019) | +0.793 (0.025) | 9.348 (0.991) | 10.164 (3.785) | 8.210 (0.362) | 8.093 (0.492) |
| autocorrelation, radial | +0.677 (0.079) | +0.651 (0.258) | +0.785 (0.018) | +0.788 (0.029) | 9.902 (1.225) | 9.879 (3.169) | 8.200 (0.368) | 8.194 (0.526) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (five draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (5) | 3 (1), 5 (4) | 2 (1), 5 (4) | 3 (1), 5 (4) |
| overlap, rotation-averaged | 2 (3), 5 (2) | 0 (2), 3 (2), 5 (1) | 5 (5) | 0 (1), 2 (1), 5 (3) |
| autocorrelation, linear | 0 (2), 1 (1), 5 (2) | 0 (4), 2 (1) | 0 (4), 1 (1) | 0 (5) |
| autocorrelation, normalised | 0 (2), 1 (1), 3 (1), 5 (1) | 0 (4), 5 (1) | 0 (4), 1 (1) | 0 (5) |
| autocorrelation, lattice-rotation-averaged | 3 (2), 5 (3) | 0 (1), 2 (1), 5 (3) | 3 (1), 5 (4) | 0 (1), 2 (1), 5 (3) |
| autocorrelation, radial | 1 (1), 2 (1), 5 (3) | 3 (1), 5 (4) | 5 (5) | 0 (1), 2 (1), 3 (1), 5 (2) |

**Label: the mean over the directions (direction-free).**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.905 (0.004) | +0.908 (0.012) | +0.914 (0.006) | +0.921 (0.008) | 4.779 (0.083) | 4.636 (0.270) | 4.533 (0.152) | 4.351 (0.196) |
| two-point statistics + PCA | +0.885 (0.024) | +0.899 (0.015) | +0.921 (0.007) | +0.928 (0.006) | 5.239 (0.508) | 4.854 (0.352) | 4.347 (0.177) | 4.158 (0.148) |
| overlap, generating frame | +0.813 (0.051) | +0.865 (0.020) | +0.883 (0.006) | +0.895 (0.008) | 6.669 (0.935) | 5.633 (0.416) | 5.294 (0.192) | 5.020 (0.130) |
| overlap, rotation-averaged | +0.784 (0.080) | +0.864 (0.046) | +0.884 (0.028) | +0.915 (0.009) | 7.120 (1.287) | 5.615 (0.902) | 5.221 (0.547) | 4.529 (0.080) |
| autocorrelation, linear | +0.898 (0.011) | +0.890 (0.033) | +0.901 (0.025) | +0.929 (0.010) | 4.946 (0.276) | 5.036 (0.709) | 4.822 (0.645) | 4.130 (0.170) |
| autocorrelation, normalised | +0.863 (0.034) | +0.880 (0.044) | +0.902 (0.023) | +0.930 (0.010) | 5.728 (0.728) | 5.250 (0.916) | 4.815 (0.578) | 4.111 (0.186) |
| autocorrelation, lattice-rotation-averaged | +0.896 (0.012) | +0.877 (0.045) | +0.910 (0.021) | +0.930 (0.014) | 5.012 (0.300) | 5.323 (0.959) | 4.620 (0.584) | 4.072 (0.262) |
| autocorrelation, radial | +0.885 (0.028) | +0.847 (0.097) | +0.918 (0.011) | +0.931 (0.015) | 5.244 (0.643) | 5.802 (1.756) | 4.422 (0.265) | 4.066 (0.270) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (five draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (5) | 5 (5) | 5 (5) | 2 (1), 5 (4) |
| overlap, rotation-averaged | 1 (1), 3 (1), 5 (3) | 3 (1), 5 (4) | 3 (1), 5 (4) | 5 (5) |
| autocorrelation, linear | 0 (1), 1 (2), 3 (1), 5 (1) | 0 (1), 5 (4) | 2 (3), 5 (2) | 0 (2), 1 (1), 3 (2) |
| autocorrelation, normalised | 3 (3), 5 (2) | 0 (2), 2 (1), 5 (2) | 2 (2), 3 (1), 5 (2) | 1 (1), 3 (3), 5 (1) |
| autocorrelation, lattice-rotation-averaged | 3 (2), 5 (3) | 2 (1), 5 (4) | 2 (1), 3 (1), 5 (3) | 0 (1), 5 (4) |
| autocorrelation, radial | 5 (5) | 0 (1), 5 (4) | 1 (1), 3 (1), 5 (3) | 2 (1), 3 (1), 5 (3) |

### 3D, 21 points per side, 400 samples, contrast 10

Labels: k_xx from 1.34 to 8.03 (sd 1.679), mean from 1.68 to 7.31 (sd 1.524). Solver checks: uniform phase 1: largest error 0.0e+00; laminate, layers across x: largest error 1.5e-14; random structure, asymmetry of the tensor 1.1e-16. Settings searched: smoothing width in {0, 1, 2, 3, 5} voxels; Gaussian layer bandwidth in {0.5, 1, 2} times the median squared distance, or linear; ridge in {1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1} times the mean self-kernel; for the baseline {2, 3, 5, 10, 20} components, degree {1, 2}, ridge {1e-08, 0.0001, 0.01, 1}. Run time 3 min.

**Label: the component along x (direction-dependent).**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.770 (0.043) | +0.805 (0.008) | +0.823 (0.014) | +0.830 (0.010) | 0.803 (0.079) | 0.738 (0.022) | 0.699 (0.019) | 0.695 (0.023) |
| two-point statistics + PCA | +0.959 (0.014) | +0.976 (0.004) | +0.984 (0.001) | +0.987 (0.003) | 0.338 (0.060) | 0.257 (0.021) | 0.209 (0.010) | 0.189 (0.020) |
| overlap, generating frame | +0.746 (0.051) | +0.837 (0.027) | +0.931 (0.013) | +0.959 (0.005) | 0.843 (0.084) | 0.672 (0.052) | 0.434 (0.044) | 0.343 (0.018) |
| overlap, rotation-averaged | +0.721 (0.075) | +0.793 (0.012) | +0.810 (0.011) | +0.817 (0.016) | 0.881 (0.114) | 0.759 (0.028) | 0.723 (0.015) | 0.722 (0.037) |
| autocorrelation, linear | +0.968 (0.010) | +0.979 (0.004) | +0.984 (0.003) | +0.989 (0.002) | 0.299 (0.045) | 0.244 (0.018) | 0.207 (0.014) | 0.179 (0.014) |
| autocorrelation, normalised | +0.948 (0.021) | +0.969 (0.011) | +0.980 (0.006) | +0.989 (0.001) | 0.379 (0.080) | 0.290 (0.054) | 0.231 (0.034) | 0.174 (0.008) |
| autocorrelation, lattice-rotation-averaged | +0.750 (0.088) | +0.783 (0.050) | +0.801 (0.048) | +0.825 (0.011) | 0.830 (0.149) | 0.775 (0.095) | 0.736 (0.074) | 0.706 (0.023) |
| autocorrelation, radial | +0.749 (0.092) | +0.783 (0.050) | +0.819 (0.014) | +0.822 (0.016) | 0.832 (0.154) | 0.774 (0.095) | 0.706 (0.024) | 0.711 (0.032) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (five draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 3 (2), 5 (3) | 5 (5) | 3 (3), 5 (2) | 3 (2), 5 (3) |
| overlap, rotation-averaged | 0 (1), 3 (1), 5 (3) | 5 (5) | 5 (5) | 5 (5) |
| autocorrelation, linear | 0 (4), 3 (1) | 0 (4), 3 (1) | 0 (4), 1 (1) | 0 (4), 1 (1) |
| autocorrelation, normalised | 0 (1), 1 (2), 5 (2) | 0 (3), 2 (1), 3 (1) | 0 (3), 1 (1), 5 (1) | 0 (3), 1 (2) |
| autocorrelation, lattice-rotation-averaged | 2 (1), 3 (2), 5 (2) | 0 (1), 3 (1), 5 (3) | 0 (1), 3 (1), 5 (3) | 0 (1), 2 (1), 5 (3) |
| autocorrelation, radial | 1 (1), 3 (2), 5 (2) | 0 (2), 5 (3) | 0 (1), 2 (1), 5 (3) | 0 (1), 1 (1), 2 (1), 5 (2) |

**Label: the mean over the directions (direction-free).**

| method | q2, n_train 20 | q2, n_train 50 | q2, n_train 100 | q2, n_train 200 | RMSE, n_train 20 | RMSE, n_train 50 | RMSE, n_train 100 | RMSE, n_train 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase fraction | +0.991 (0.001) | +0.991 (0.000) | +0.992 (0.000) | +0.992 (0.000) | 0.147 (0.009) | 0.142 (0.003) | 0.138 (0.003) | 0.135 (0.003) |
| two-point statistics + PCA | +0.992 (0.001) | +0.992 (0.001) | +0.994 (0.000) | +0.995 (0.000) | 0.132 (0.010) | 0.137 (0.009) | 0.115 (0.004) | 0.107 (0.003) |
| overlap, generating frame | +0.974 (0.008) | +0.986 (0.001) | +0.990 (0.001) | +0.991 (0.001) | 0.246 (0.039) | 0.176 (0.006) | 0.151 (0.005) | 0.140 (0.007) |
| overlap, rotation-averaged | +0.969 (0.025) | +0.985 (0.010) | +0.990 (0.002) | +0.992 (0.000) | 0.257 (0.095) | 0.177 (0.051) | 0.149 (0.015) | 0.135 (0.005) |
| autocorrelation, linear | +0.993 (0.002) | +0.993 (0.002) | +0.994 (0.000) | +0.996 (0.001) | 0.127 (0.015) | 0.121 (0.015) | 0.113 (0.003) | 0.101 (0.009) |
| autocorrelation, normalised | +0.992 (0.001) | +0.993 (0.001) | +0.994 (0.000) | +0.995 (0.001) | 0.139 (0.011) | 0.124 (0.008) | 0.115 (0.004) | 0.102 (0.006) |
| autocorrelation, lattice-rotation-averaged | +0.993 (0.002) | +0.993 (0.002) | +0.995 (0.001) | +0.996 (0.000) | 0.127 (0.014) | 0.121 (0.018) | 0.105 (0.009) | 0.093 (0.006) |
| autocorrelation, radial | +0.994 (0.001) | +0.994 (0.001) | +0.994 (0.001) | +0.996 (0.001) | 0.120 (0.014) | 0.112 (0.006) | 0.113 (0.007) | 0.095 (0.007) |

Smoothing width chosen by the inner cross-validation, in voxels, with the number of draws that chose it (five draws per training size):

| method | n_train 20 | n_train 50 | n_train 100 | n_train 200 |
|---|---|---|---|---|
| overlap, generating frame | 5 (5) | 5 (5) | 5 (5) | 5 (5) |
| overlap, rotation-averaged | 5 (5) | 5 (5) | 3 (1), 5 (4) | 5 (5) |
| autocorrelation, linear | 1 (1), 2 (2), 3 (1), 5 (1) | 2 (2), 5 (3) | 0 (2), 1 (1), 2 (1), 5 (1) | 0 (2), 2 (2), 3 (1) |
| autocorrelation, normalised | 0 (1), 2 (1), 3 (1), 5 (2) | 2 (2), 3 (3) | 1 (1), 2 (1), 3 (3) | 3 (5) |
| autocorrelation, lattice-rotation-averaged | 0 (1), 2 (1), 3 (3) | 0 (2), 1 (1), 5 (2) | 0 (4), 3 (1) | 1 (4), 2 (1) |
| autocorrelation, radial | 0 (2), 2 (1), 3 (2) | 0 (3), 2 (1), 5 (1) | 0 (1), 2 (2), 3 (1), 5 (1) | 0 (4), 1 (1) |
