# Data fusion against the one-class descriptions

AUC of the held-out actives against the inactives of the same set, five folds over
the actives, averaged. Nothing tuned. The CMF columns are the attenuation that does
best on that set and the median over the three attenuations tried; a screening run
has no criterion to pick the attenuation with, so the median is what it should
expect.

| method | COX2 | DHFR | BZR | AChE | THR | mean | median over alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| fusion max | 0.732 | 0.860 | 0.523 | 0.606 | 0.757 | **0.696** | 0.686 |
| fusion min | 0.692 | 0.700 | 0.578 | 0.627 | 0.746 | **0.669** | 0.627 |
| fusion sum | 0.746 | 0.774 | 0.438 | 0.530 | 0.772 | **0.652** | 0.643 |
| fusion med | 0.728 | 0.744 | 0.432 | 0.489 | 0.753 | **0.629** | 0.622 |
| fusion euc | 0.741 | 0.776 | 0.443 | 0.532 | 0.771 | **0.653** | 0.643 |
| fusion rrf | 0.786 | 0.783 | 0.588 | 0.535 | 0.769 | **0.692** | 0.679 |
| fusion anz (threshold swept) | 0.746 | 0.835 | 0.561 | 0.726 | 0.776 | **0.729** | 0.666 |
| fusion mnz (threshold swept) | 0.778 | 0.816 | 0.584 | 0.634 | 0.791 | **0.721** | 0.644 |
| oneclass centroid | 0.746 | 0.774 | 0.438 | 0.530 | 0.772 | **0.652** | 0.643 |
| oneclass k-NN | 0.723 | 0.818 | 0.566 | 0.605 | 0.803 | **0.703** | 0.695 |
| oneclass GP | 0.760 | 0.749 | 0.703 | 0.656 | 0.813 | **0.736** | 0.721 |
| oneclass GP/sd | 0.748 | 0.836 | 0.549 | 0.659 | 0.810 | **0.720** | 0.715 |
| oneclass kernel PCA | 0.744 | 0.817 | 0.570 | 0.668 | 0.829 | **0.725** | 0.713 |

## The same rules on ECFP4

The fusion rules are functions of a similarity row, so they run unchanged on a
fingerprint Tanimoto; only the meaning of "similar" changes.

| method | COX2 | DHFR | BZR | AChE | THR | mean |
|---|---:|---:|---:|---:|---:|---:|
| fusion max | 0.643 | 0.879 | 0.742 | 0.793 | 0.636 | **0.738** |
| fusion min | 0.519 | 0.589 | 0.321 | 0.588 | 0.670 | **0.537** |
| fusion sum | 0.693 | 0.743 | 0.675 | 0.691 | 0.626 | **0.686** |
| fusion med | 0.673 | 0.653 | 0.452 | 0.541 | 0.612 | **0.586** |
| fusion euc | 0.684 | 0.772 | 0.734 | 0.733 | 0.624 | **0.709** |
| fusion rrf | 0.758 | 0.733 | 0.536 | 0.597 | 0.645 | **0.654** |
| fusion anz (threshold swept) | 0.693 | 0.860 | 0.757 | 0.809 | 0.706 | **0.765** |
| fusion mnz (threshold swept) | 0.696 | 0.826 | 0.740 | 0.790 | 0.673 | **0.745** |
| oneclass centroid | 0.693 | 0.743 | 0.675 | 0.691 | 0.626 | **0.686** |
| oneclass k-NN | 0.571 | 0.854 | 0.738 | 0.766 | 0.605 | **0.707** |
| oneclass GP | 0.772 | 0.765 | 0.738 | 0.731 | 0.741 | **0.750** |
| oneclass GP/sd | 0.763 | 0.855 | 0.809 | 0.812 | 0.723 | **0.792** |
| oneclass kernel PCA | 0.582 | 0.852 | 0.820 | 0.800 | 0.706 | **0.752** |

## Sets

| set | molecules | actives | inactives | median activity |
|---|---:|---:|---:|---:|
| COX2 | 282 | 139 | 143 | 6.89 |
| DHFR | 361 | 180 | 181 | 6.34 |
| BZR | 147 | 73 | 74 | 7.89 |
| AChE | 111 | 55 | 56 | 6.84 |
| THR | 88 | 44 | 44 | 6.69 |
