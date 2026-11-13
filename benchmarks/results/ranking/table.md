# Does fitting the order beat fitting the values?

8 Sutherland sets, 10 random 70/30 splits each; Spearman's rho of the held-out molecules. The pooled rows give the same molecules to 3 synthetic sources and shift each by an offset of 1.5 standard deviations, then ask for the order of the *true* activity, which no model saw.

| set | n | clean: KRR | clean: rank | pooled: KRR | pooled: rank | pooled: rank by source |
|---|---:|---:|---:|---:|---:|---:|
| ACE | 114 | +0.796 | +0.798 | +0.725 | +0.729 | +0.763 |
| AChE | 111 | +0.772 | +0.707 | +0.731 | +0.719 | +0.761 |
| BZR | 147 | +0.575 | +0.564 | +0.678 | +0.653 | +0.696 |
| COX2 | 282 | +0.752 | +0.742 | +0.749 | +0.753 | +0.776 |
| DHFR | 361 | +0.796 | +0.771 | +0.744 | +0.743 | +0.771 |
| GPB | 66 | +0.621 | +0.558 | +0.716 | +0.696 | +0.703 |
| THER | 76 | +0.674 | +0.670 | +0.696 | +0.692 | +0.723 |
| THR | 88 | +0.721 | +0.727 | +0.744 | +0.748 | +0.768 |
| **mean** |  | **+0.713** | **+0.692** | **+0.723** | **+0.717** | **+0.745** |

## The pooled sets, ordered across sources

The same models and splits, scored by Spearman's rho over *all* held-out pairs. A model that only ever compared molecules inside a source cannot say where one source lies relative to another, so this column shows that limitation.

| set | KRR | rank | rank by source |
|---|---:|---:|---:|
| ACE | +0.612 | +0.602 | +0.731 |
| AChE | +0.562 | +0.535 | +0.632 |
| BZR | +0.432 | +0.406 | +0.575 |
| COX2 | +0.484 | +0.501 | +0.707 |
| DHFR | +0.612 | +0.604 | +0.749 |
| GPB | +0.531 | +0.482 | +0.569 |
| THER | +0.572 | +0.570 | +0.533 |
| THR | +0.596 | +0.597 | +0.687 |
| **mean** | **+0.550** | **+0.537** | **+0.648** |

The clean columns are Spearman's rho over the held-out molecules; the pooled columns are the share of **same-source** held-out pairs ordered correctly, so 0.5 is no better than chance. The two halves are not comparable with each other.
