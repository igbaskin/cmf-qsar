# A potency and an anti-target on the same compounds

Medicinal chemistry asks of every series where field may be added for the target that is **not**
also added for the off-target. The two-task model of this program answers that as a difference of
co-fields (plan items SC1 and E1); what plan item MC4 says was missing is a series in the
program's format with both measurements on the same compounds. This folder is that series.

`examples/make_antitarget_examples.py` builds it; the file it writes is kept here so that nothing
has to be downloaded to use it.

## `h3_herg.sdf` — 78 histamine H3 ligands with their hERG activity

* **Source.** One ChEMBL document, which is one paper: Dorota Lazewska and co-workers, *Novel and
  highly potent histamine H3 receptor ligands. Part 1: withdrawing of hERG activity*, Bioorganic &
  Medicinal Chemistry Letters 21 (2011) 5378–5382, <https://doi.org/10.1016/j.bmcl.2011.07.006>,
  ChEMBL document `CHEMBL1821550`. The paper's own subject is this item's question: how to keep
  the H3 affinity of a series of basic amines while taking its hERG activity away.
* **Why one paper.** An anti-target number from another laboratory is a different measurement.
  Both numbers here come from the same work and from one assay each, and the assay descriptions
  travel with every molecule as columns, so a reader can see what was measured:
  * `pKi_H3` — displacement of [125I]iodoproxyfan from the human recombinant H3 receptor
    (ChEMBL assay `CHEMBL1827363`), as −log10 Ki in M;
  * `pIC50_hERG` — displacement of [3H]dofetilide from human recombinant ERG (ChEMBL assay
    `CHEMBL1827362`), as −log10 IC50 in M.
* **Licence.** ChEMBL is distributed under CC BY-SA 3.0, so this copy may be redistributed; the
  attribution above is the condition.
* **Selection.** Of the 224 compounds of the paper that carry both measurements, the 78 that
  belong to one series: their maximum common substructure with the reference molecule covers at
  least half of their heavy atoms and has at least 12 atoms, and the alignment on it succeeded.
  The reference is `CHEMBL1086514`, the molecule most like the rest by the size of its common
  substructure with a sample of them. The 146 that were left out are the paper's other scaffolds;
  a common frame is what a co-field needs, and a set of 126 Murcko scaffolds has none.
* **What was changed.** ChEMBL gives no coordinates, so a conformation was generated here with
  ETKDG and refined with MMFF94, and every molecule was superimposed on the reference by its
  common substructure (`cmfqsar.align.mcs_align`). Where a compound carries several records of
  the same assay their mean is used. Nothing else: the activities are the pChEMBL values as
  deposited.
* **The columns.** `pKi_H3` (also written as `activity`, so the file reads as an ordinary data
  set), `pIC50_hERG`, their difference `selectivity`, the ChEMBL identifiers of the molecule, the
  two assays and the document, the two assay descriptions and standard types, the reference, the
  name of the frame molecule and the size of each molecule's common part with it.

## `h3_candidates.sdf` -- 425 candidates, 16 of them made next

The pool a design loop would choose from, for plan item MC11. Every other histamine H3 ligand in
ChEMBL that fits the same frame as the training set: posed the same way, carrying its measured
H3 affinity (`pKi_H3`), the documents it comes from, and `made_next`, which is 1 for the 16
compounds of **part 2 of the same work** -- Lazewska and co-workers, Bioorg. Med. Chem. Lett. 21
(2011) 5384, ChEMBL document `CHEMBL1821619` -- and 0 for the rest.

Part 2 reports an H3 affinity for its compounds and no hERG number, which is exactly the
position a design loop is in: what was made is measured afterwards on the objective that was
followed up, and not on the other one. No candidate carries a hERG value, so nothing in this
file tells a loop trained on `h3_herg.sdf` anything about its second objective.

    python examples/make_antitarget_examples.py --candidates

rebuilds it; `chembl_h3_pool.json` is the download it is built from.

## What it is used for

`benchmarks/run_antitarget.py` runs the two parts of plan item MC4's gate on `h3_herg.sdf`:
whether tying the anti-target to the potency costs anything on the potency, and where the
difference co-field puts its weight. `benchmarks/run_designloop.py` runs plan item MC11 on both
files: a design loop with two objectives, trained on the first and asked to choose from the
second. `benchmarks/results/antitarget/table.md` and `benchmarks/results/designloop/table.md`
hold the two results. From the command
line the same model is

```bash
cmf build examples/antitarget/h3_herg.sdf --activity pKi_H3 --second-endpoint pIC50_hERG \
    --save h3.cmf
cmf cofield h3.cmf --endpoint selectivity --out-prefix selectivity
```

`chembl_raw.json` is the download the file was built from, kept so that the build can be repeated
exactly (`--offline`) even if ChEMBL changes.
