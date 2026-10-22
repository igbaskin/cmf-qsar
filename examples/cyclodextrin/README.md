# Cyclodextrin guest sets

Association constants between a cyclodextrin and a small guest, one file per host. The host is
fixed inside a file and the guests are whatever the literature measured, so these are the opposite
of every other series under `examples/`: there is no shared scaffold and **no superposition is
defined**. They are here for that reason, as the data on which an alignment-free kernel can be
asked to earn its place.

## Where the data come from

Tahıl G., Delorme F., Le Berre D., Monflier É., Sayede A., Tilloy S., *Curated dataset of
association constants between a cyclodextrin and a guest for machine learning*, Chemical Data
Collections **45** (2023) 101022. Data deposited at <https://doi.org/10.5281/zenodo.7575539>,
archive `Data.zip`, 488 965 bytes, MD5 `9260a9a8a86db1da967d91a22ece8cb4`.

`upstream/` holds that archive unpacked and unaltered: the authors' raw compilation
(`CDRawData.csv`, 3754 rows as published from the literature), their curated and enriched table
(`CDEnrichedData.csv`, 3459 rows with PubChem CIDs, isomeric SMILES for guest and host,
physicochemical descriptors and their own `iso2vec` embeddings), and the licence.

**Terms.** Open Data Commons Open Database Licence (ODbL 1.0), the copy distributed in the archive
being `upstream/LICENSE.md`. It permits redistribution, which is why the files are kept here rather
than fetched on demand, and it asks three things in return: attribution, that the licence travel
with the data, and that a *publicly used derived database* be offered under the same licence. The
per-host tables in this folder are such a derived database. They are offered under ODbL 1.0, and
anything further derived from them carries the same condition.

## The files

| file | host | guests | log K range |
|---|---|---|---|
| `beta_cyclodextrin.csv` | β-CD | 1162 | 0.00 to 5.37 |
| `alpha_cyclodextrin.csv` | α-CD | 827 | 0.30 to 4.34 |
| `gamma_cyclodextrin.csv` | γ-CD | 292 | 0.30 to 4.81 |
| `hydroxypropyl_beta_cyclodextrin.csv` | HP-β-CD | 135 | 0.00 to 5.05 |
| `sulfobutylether_beta_cyclodextrin.csv` | SBE-β-CD | 135 | 0.95 to 4.34 |
| `methyl_beta_cyclodextrin.csv` | methyl-β-CD | 77 | 0.95 to 5.22 |
| `dimethyl_beta_cyclodextrin.csv` | 2,6-di-O-methyl-β-CD | 43 | 0.90 to 4.00 |

`hosts.csv` lists all fifteen hosts of the database with their counts, so the eight that fall below
the twenty-guest threshold are visible rather than silently absent.

`ph7/` holds the same seven tables with the same activities and the structures in their protonation
state at pH 7, the way `examples/corrosion_hcl/` carries the corrosion series protonated for 1 M
HCl. Dimorphite-DL decides each site from its SMARTS pKa rules; the protonated SMILES are in the
`smiles` column the program reads and the neutral ones in `smiles_neutral`, beside the net
`charge`. Of the 2671 guests, 882 change: 793 become anions, 342 cations, 1536 stay neutral.

Unlike the corrosion export there is no cap on added protons. At pH 0 a cap is needed because the
rules charge every ring nitrogen while the first protonation makes the second site far less basic;
at pH 7 only sites with a rule pKa above 7 are charged, and putrescine, for one, really is doubly
protonated there. A site whose pKa sits on top of the pH is the weak case either way: 4-nitrophenol
is about forty per cent ionised at pH 7 and these rules leave it neutral.

Whether protonating helps is not settled. On a single 75/25 split of the three natural hosts with
the invariant kernel at default settings, held-out prediction improved on all three, from 0.414 to
0.477 on β-CD, 0.259 to 0.341 on α-CD and 0.413 to 0.438 on γ-CD. The fuller measurement,
`benchmarks/run_cyclodextrin.py` over all seven hosts with a grid and ten-fold cross-validation,
is less flattering: the invariant kernel improves on four hosts of seven and loses on the rest, so
the mean barely moves. The two largest sets are among those that improve, which is the reading to
prefer, that the charges carry signal a large enough set can use. See
`benchmarks/results/cyclodextrin/table.md` for the numbers.

For scale: the largest field-based models of this endpoint in the literature are CoMFA with
COSMOsar3D fields on 60 α-CD guests and GRIND on 126 and on 233 β-CD guests.

Each file is an activity table the program reads directly, `activity` being log₁₀ K with K in
M⁻¹. The other columns are `logK`, `n_values` and `spread` for the measurements that were
combined, `K_median`, the PubChem `cid`, the `pH` and `T` of the retained rows, the `host` and its
CID, and the literature `reference`. The structures are SMILES, so the program embeds and
optimises the conformer itself.

## How these were made

```bash
python examples/make_cyclodextrin_examples.py --fetch        # the neutral tables
python examples/make_cyclodextrin_examples.py --protonate     # and ph7/ as well
```

The first downloads the archive, checks its MD5 against the digest the Zenodo record publishes,
unpacks it into `upstream/`, and writes the tables. `--protonate` takes an optional pH, the centre
of the pH window by default, and `--max-protons` restores the corrosion export's cap. The selection is one window of conditions, 298 ± 2 K
and pH 7 ± 0.5, and one row per guest holding the **median** of log₁₀ K over the measurements
inside that window. Widen or move the window with `--temperature`, `--t-window`, `--ph`,
`--ph-window`.

## What to watch before quoting a number

* **The SMILES at the top level are neutral forms**, as PubChem deposits them, so a carboxylic acid
  measured at pH 7 is written protonated there. Use `ph7/` for the structures as they existed in
  the measurement. Both are kept because the neutral set is what the published models of this
  endpoint were built on, so it is the comparable one.
* **Repeated measurements are reduced to their median.** Agreement between references is usually
  excellent, a median spread of 0.00 to 0.01 log units over the repeated guests of the three
  natural cyclodextrins, but one α-CD guest differs by 3.2 log units between two sources. Filter on
  `spread` before trusting an individual value.
* **1:1 stoichiometry is assumed**, which a single constant per row implies but the upstream table
  does not state row by row.
* **The condition window is a convention, not a correction.** Rows between 296 and 300 K are
  pooled as if isothermal.
* **No frame.** Nothing in these files is aligned, and the aligned kernel should not be expected to
  work on them. The intended route is
  `--kernel autocorrelation`, the rotation-invariant field autocorrelation of `cmfqsar/invariant.py`.
  `benchmarks/run_cyclodextrin.py` measures that claim on every host in both variants, against an
  automatic superposition and against atom-type counts, and writes
  `benchmarks/results/cyclodextrin/table.md`.

```bash
python cmf_cli.py build examples/cyclodextrin/beta_cyclodextrin.csv \
    --kernel autocorrelation --lams 1e-2,0.1,1 --test-fraction 0.25
```
