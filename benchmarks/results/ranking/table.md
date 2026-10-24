# Does fitting the order beat fitting the values?

8 Sutherland sets, 5 random 70/30 splits each; Spearman's rho of the held-out molecules. The pooled rows give the same molecules to 3 synthetic sources and shift each by an offset of 1.5 standard deviations, then ask for the order of the *true* activity, which no model saw.

| set | n | clean: KRR | clean: rank | pooled: KRR | pooled: rank | pooled: rank by source |
|---|---:|---:|---:|---:|---:|---:|
| ACE | 114 | +0.765 | +0.752 | +0.718 | +0.720 | +0.728 |
| AChE | 111 | +0.758 | +0.754 | +0.743 | +0.746 | +0.766 |
| BZR | 147 | +0.600 | +0.607 | +0.682 | +0.659 | +0.708 |
| COX2 | 282 | +0.756 | +0.735 | +0.733 | +0.747 | +0.772 |
| DHFR | 361 | +0.794 | +0.769 | +0.754 | +0.746 | +0.777 |
| GPB | 66 | +0.641 | +0.482 | +0.724 | +0.679 | +0.671 |
| THER | 76 | +0.702 | +0.722 | +0.709 | +0.692 | +0.758 |
| THR | 88 | +0.690 | +0.692 | +0.735 | +0.738 | +0.761 |
| **mean** |  | **+0.713** | **+0.689** | **+0.725** | **+0.716** | **+0.743** |

## The pooled sets, ordered across sources

The same models and splits, scored by Spearman's rho over *all* held-out pairs. A model that only ever compared molecules inside a source has no opinion about where one source sits relative to another, so this column is where its silence shows.

| set | KRR | rank | rank by source |
|---|---:|---:|---:|
| ACE | +0.712 | +0.740 | +0.704 |
| AChE | +0.580 | +0.589 | +0.640 |
| BZR | +0.452 | +0.436 | +0.614 |
| COX2 | +0.480 | +0.521 | +0.705 |
| DHFR | +0.619 | +0.610 | +0.755 |
| GPB | +0.559 | +0.441 | +0.540 |
| THER | +0.592 | +0.600 | +0.681 |
| THR | +0.599 | +0.585 | +0.662 |
| **mean** | **+0.574** | **+0.565** | **+0.663** |

The clean columns are Spearman's rho over the held-out molecules; the pooled columns are the share of **same-source** held-out pairs ordered correctly, so 0.5 is a coin toss. The two halves are not comparable with each other.
