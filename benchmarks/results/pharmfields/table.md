# Six pharmacophore features against the Sybyl indicators

Indicator fields put a Gaussian of weight one on every atom of one kind. Keyed by Sybyl atom type they help above about forty molecules and not below, because a congeneric series has twenty-odd types and each is rare. These six -- donor, acceptor, aromatic, hydrophobe, cation, anion, from RDKit BaseFeatures.fdef -- are each carried by most molecules of a series. Eight Sutherland sets, their own alignment. Same attenuation and ridge grids and the same ten-fold cross-validation as `run_alignment.py`.

| set | n | five | five + pharm | pharm | five + cif | cif | best |
|---|---:|---:|---:|---:|---:|---:|---|
| GPB | 66 | +0.600 | +0.589 | +0.527 | **+0.644** | +0.635 | five + cif |
| THER | 76 | +0.509 | +0.575 | **+0.635** | +0.557 | +0.562 | pharm |
| THR | 88 | **+0.675** | +0.657 | +0.520 | +0.662 | +0.619 | five |
| AChE | 111 | +0.564 | +0.562 | +0.542 | **+0.578** | +0.539 | five + cif |
| ACE | 114 | +0.657 | +0.695 | +0.670 | **+0.700** | +0.690 | five + cif |
| BZR | 147 | +0.388 | +0.384 | +0.297 | +0.408 | **+0.415** | cif |
| COX2 | 282 | +0.543 | +0.542 | +0.459 | **+0.549** | +0.521 | five + cif |
| DHFR | 361 | +0.659 | **+0.687** | +0.655 | +0.680 | +0.655 | five + pharm |
| **mean** | | +0.574 | +0.586 | +0.538 | +0.597 | +0.580 | |

Split at 40 molecules, which is where the Sybyl indicators were found to start helping -- and every set here is above it, which is why the `--mittal` run exists:

| sets | n | five | five + pharm | pharm | five + cif | cif |
|---|---:|---:|---:|---:|---:|---:|
| at or above 40 | 8 | +0.574 | +0.586 | +0.538 | +0.597 | +0.580 |

**Added to the five fields, the six features are worth +0.012 q² on average (3 of 8 sets better) and the Sybyl indicators +0.023 (7 of 8).** Alone, the six reach +0.538 against +0.580 for the Sybyl ones and +0.574 for the five physico-chemical fields.

## Why: what each vocabulary covers

The share of the molecules of a set that carries each kind. This is the whole argument of the item, and it is visible before any model is fitted.

| set | donor | acceptor | aromatic | hydrophobe | cation | anion | Sybyl types, all | median coverage | types kept | median coverage of those |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPB | 100% | 100% | 18% | 44% | 12% | 2% | 19 | 8% | 11 | 56% |
| THER | 100% | 100% | 88% | 100% | 13% | 54% | 13 | 80% | 11 | 88% |
| THR | 100% | 100% | 100% | 100% | 100% | 10% | 13 | 100% | 12 | 100% |
| AChE | 100% | 97% | 100% | 100% | 98% | 0% | 13 | 41% | 9 | 95% |
| ACE | 99% | 100% | 60% | 100% | 42% | 96% | 14 | 52% | 12 | 76% |
| BZR | 64% | 99% | 100% | 100% | 33% | 1% | 17 | 50% | 14 | 69% |
| COX2 | 51% | 100% | 100% | 100% | 32% | 1% | 17 | 31% | 16 | 49% |
| DHFR | 100% | 100% | 100% | 100% | 7% | 4% | 17 | 14% | 15 | 15% |

A pharmacophore feature is carried by 71% of the molecules of a set on average, and the median Sybyl type by 47% -- which is the difference the item was built on. But the Sybyl vocabulary the program actually uses is already pruned: `cif_types_in` keeps only the types in five molecules or more, which on these sets is 12 of 15 types, and the median coverage of what is kept is 67%. **The rarity the item blames was already being filtered out**, so whatever the six features gain is not that.

