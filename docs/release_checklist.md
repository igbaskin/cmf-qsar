# The 1.0.0 release, step by step

This is item 0 of `docs/publication_plan.md`: the public repository, the tag, the Zenodo DOI, the
package on PyPI, the manual as supplementary material, and one reproduction run on another
machine. No paper is submitted before it is done, because every paper cites the DOI.

The steps are in the order that works. Two of them only work in this order: Zenodo has to be
connected before the tag is created, and the manifest has to be regenerated before the tag is
created.

## 1. The repository must be clean and the tests green

The continuous-integration workflow runs `pytest tests` on six jobs and
`benchmarks/run_all.py --check` on a seventh, so a fault here is a red badge on the day the
repository becomes public.

* [x] Every benchmark is named in the manual. The `pooled_mixing` benchmark was added without a
  section; it is now described under *Which fields does this series need?* in `docs/manual.md`.
* [x] `docs/CMF_manual.pdf` was rebuilt from the current `docs/manual.md`, and
  `docs/CMF_manual.sha256` with it.
* [x] The four data items of unclear licence status are decided and the reason is written down
  (`docs/public_3dqsar_datasets.md`, *Before the repository is public*).
* [x] **The manifest.** Regenerated on 7 November 2026 from a working tree with nothing
  uncommitted. It records commit `abdcc4c`, the merge that brought in plan item PBH6, 545 data
  files and 1,353 result files, and `dirty: false`. Two earlier states are worth recording,
  because both would have gone into the archive unnoticed: the entry this replaced recorded
  commit `fd43093`, twenty-one commits back, and was missing the eighteen result files of plan
  item M15; the entry before that recorded `85e58f3` with `dirty: true`. Repeat the commands
  below after anything is committed from here on:

  ```bash
  git status --short          # must print nothing
  python benchmarks/run_all.py --manifest
  python benchmarks/run_all.py --check
  ```

  `--manifest` only hashes the files, so it takes about a minute and does not rerun any
  benchmark. One thing to expect when reading the result: the manifest records the commit it was
  built at, which is the parent of the commit that carries the manifest itself. The tag therefore
  goes on the commit that stores the manifest, and that commit's own hash is one ahead of the one
  written inside it. This is not a fault and `--check` does not test for it.

## 2. Make the repository public

**Done 6 November 2026.** `github.com/igbaskin/cmf-qsar` is public, changed with the command
below. The step was brought forward, before the manifest of step 1, for a reason worth recording:
continuous integration had stopped running. Actions minutes are metered for a private repository,
with Linux counted once, Windows twice and macOS ten times, so one push of this seven-job workflow
costs about 40 billed minutes; GitHub refused every job of three runs in a row, each job failing in
two seconds with no step executed. A public repository uses free runners with no quota, so the
refusal ends with this step rather than with a payment.

Three checks were made before the change, and all three passed: no file that looks like a
credential was ever added in the history (`.env`, `*.pem`, `id_rsa`, `netrc` and the rest), none of
the 2,299 tracked files carries a token, key or private-key block, and `git ls-files '*.pdf'`
prints `docs/CMF_manual.pdf` alone.

This step is done in the GitHub web interface (Settings, *Change visibility*), or with an
authenticated `gh`:

```bash
gh repo edit igbaskin/cmf-qsar --visibility public --accept-visibility-change-consequences
```

Check afterwards that the publisher PDFs are still absent: `.gitignore` excludes `*.pdf` and
allows only `docs/CMF_manual.pdf`, and `git ls-files '*.pdf'` must print that one file alone.

## 3. Connect Zenodo before tagging

**Done 7 November 2026.** Zenodo archives a GitHub release only if the connection existed before
the release was created. Sign in at zenodo.org with GitHub, open *GitHub* in the account menu, and
switch the `igbaskin/cmf-qsar` repository on. `.zenodo.json` in the repository root supplies the
title, the author, the licence and the keywords, so the archive does not depend on what GitHub
guesses. Two things that are easy to get wrong on that page: the repository appears in the list
before it is switched on, so seeing it there is not the same as having enabled it; and private
repositories are not supported, which is why step 2 comes first.

## 4. Tag and release

**Done 7 November 2026.** The tag is on commit `65ffa08`, which is the commit that carries the
regenerated manifest of step 1.

```bash
git tag -a v1.0.0 <commit> -m "cmfqsar 1.0.0"
git push origin v1.0.0
gh release create v1.0.0 --title "cmfqsar 1.0.0" --notes-file <notes> docs/CMF_manual.pdf
```

The manual is attached to the release, so the supplementary material of every paper is one file
with a published checksum: 9,209,523 bytes, identical to the copy at the tagged commit. Zenodo
mints the DOI when the release is published, which took under a minute here.

## 5. Write the DOI down

**Done 7 November 2026.** The DOI arrived in the two expected forms. Papers cite the concept DOI,
which always points at the newest version.

* Concept DOI: [10.5281/zenodo.24043660](https://doi.org/10.5281/zenodo.24043660)
* Version 1.0.0: [10.5281/zenodo.24043661](https://doi.org/10.5281/zenodo.24043661)

Written into `CITATION.cff` (the `doi:` field, with `date-released` moved to the day of the
release), `README.md` (the badge under the title and the citation paragraph) and
`docs/publication_plan.md` (item 0).

## 6. The package on PyPI

The name `cmfqsar` is free. Build and check before anything is uploaded:

```bash
python -m build
python -m twine check dist/*
```

Upload to TestPyPI first, install from there into an empty environment, and run `cmf --help` and
the test suite once. Only then upload to PyPI. Both uploads need the account token and are not
reversible: a released version number can be deleted but never reused.

## 7. One reproduction run on another machine

`benchmarks/run_all.py` registers 125 benchmarks and 109.0 hours of computation, so this is not one
continuous-integration job: a GitHub runner stops a job after six hours. Which benchmarks a
reproduction run measures follows from a rule in `benchmarks/run_all.py` rather than from a list
of names: a benchmark joins the run when it registers 30 minutes or less (`REPRODUCE_MINUTES`) and
needs no file or program the machine may not have (`needs`). `REPRODUCE_ALSO` names the one
exception, `hops` at 110 minutes, because the barriers of a lithium hop are the only measurement
this program has compared with published numbers on the same labels. The rule gives **76 of the
125 benchmarks, 19.2 hours in all**, and `reproduce_groups()` splits them into **seven groups of
about 2 hours 45 minutes**, so that a group fits twice over in the 330 minutes its job is allowed.
`python benchmarks/run_all.py --list` prints the groups as they stand.

```bash
gh workflow run reproduce.yml -f group=all        # or group=1 to run one group
```

Each job measures its group again, then rebuilds every table from the new result files and fails
if a table differs from the one the repository carries. That comparison is the reproduction test,
and a red job is a result to read rather than a fault to repair.

### What the first run showed, 7 November 2026

**It measured almost nothing, and the workflow had to be repaired before the step could be
attempted again.** Run 42410923206 was green in all four groups of the time, in 2 hours 15 minutes
of runner time against the 15 hours the scope was supposed to cost. Four faults produced that, and
each of them would have let the release claim a reproduction that had not been made.

* **Nothing was measured again.** Most scripts keep the result files they find and compute only
  what is missing, so in a clone that carries the results they printed `[skip]` and stopped.
  `--fresh` now passes `--force` to every script that has the option; of the 60 benchmarks of the
  time, 41 have it and the other 19 measure everything in any case.
* **The comparison could not fail.** `--compute` rebuilt every table and rewrote the manifest when
  it finished, so the `--check` after it compared the tables with files the same command had just
  written. `--no-rebuild` leaves both as they are, and `--reproduce GROUP` is `--compute` with
  both of these options, which is what each job now runs.
* **A failed script would have looked like a measurement that did not move.** A script that exited
  with an error left the committed results in place and said so in one line of the log.
  `--compute` now fails when a script fails, after it has run the rest of the group.
* **The groups had drifted from the scope they claimed.** They were a list of sixty names in the
  workflow file. Seven benchmarks added to the registry after it was written -- `anchors`,
  `cell_view`, `denmark`, `efield`, `phenazine`, `shells`, `surface_view` -- had never been part
  of a reproduction run, and `hops` was in it although it is nearly four times the stated limit.
  The groups are now computed from the rule, and `tests/test_reproduce_harness.py` fails when a
  benchmark the rule admits is in no group.

**What the run did show**, in the one step that was looking at the numbers: among the benchmarks
that measure everything in any case, several moved. `fluctuation` moved most. On the thrombin set
its q2 went from 0.302 to 0.278, and one row of the same table from 0.017 to 0.106. Its own result
file says where that comes from: the mean number of pseudo-atoms per molecule is 2119.398 here and
2120.648 on the runner, so the conformers the runner generated are not the conformers here.
Whether a difference of that size matters is the question of rule 4, and the repaired check
answers the part a program can: for every table that differs it reports how many of its numbers
moved and the largest difference, so the log says what to read instead of leaving a diff to be
read by eye.

One thing holds for every run: a result file records the absolute path of the checkout it was
written in, so result files always differ between machines and only the tables can be compared.

### What the second run showed, the night of 7 November 2026

Run 42421747146, the repaired workflow on `claude/reproduce-harness`, **measured for real**: no
`[skip]` line appears in any of the seven logs, and the 66 benchmarks took 5 hours 14 minutes of
runner time in all (73, 68, 27, 31, 16, 49 and 50 minutes per group). That is less than the
16.7 hours they register, which are times measured on the author's machine; the runner is faster
for most of these. All seven groups were red, for three different reasons.

**Two groups stopped in the measuring step, for want of a package.** `protonation` needs
Dimorphite-DL and `trajectory` needs mdtraj or MDAnalysis, and `requirements-lock.txt` pinned
none of the three. Both are optional extras of the package (`examples` and `md`), and the
reproduction run is the one place where every benchmark of the scope has to run, so the locked
versions now install all three: `dimorphite-dl==2.1.0`, `mdtraj==1.11.1.post1` and
`MDAnalysis==2.10.0`, the versions the committed results were measured with. `run_trajectory.py`
needs both readers, because it asks whether the two agree. The alternative -- a `needs` entry
that keeps the two out of the run -- would have cost two benchmarks of coverage for two
`pip install` lines.

**Five groups named a table difference**, which is what the run is for: `fe110`, 195 of 751
numbers moved, the largest by 4.8 (88 -> 83.2); `corrosion`, 137 of 508, largest 1;
`corrosion_hcl`, 134 of 509, largest 2; `cyclodextrin`, 44 of 136, largest 1; `topomer`, 10 of
195, largest 0.003; and the README's spliced copy of the multitask table.

**The check saw almost nothing of what moved.** Comparing the tables of the seven result
artifacts with the committed ones, newlines ignored, **30 of the 66 tables moved and the check
named six**. The reason is the fault of 7 November one level down: most scripts write their own
`results/<name>/table.md` while they measure, so the file the check reads is already the new
table and the comparison is the new numbers against themselves. A benchmark that registers no
`report` has no comparison at all. `--check --against-commit` repairs both: every table is
compared with the one the repository carries, and `check_tables()` walks
`benchmarks/results` so that a table no `report` rebuilds is compared as well. Among the 24
movements the earlier comparison passed over: `fluctuation` (on the 31 CBG steroids the q2 of
the mean alone +0.794 -> +0.767), `matbench_phonons` (physchem+mass MAE 117.3 -> 111.2),
`restriction` (33 -> 37 pairs, a grid of 12350 -> 13000 points), `nystrom`, `sites`, `anchors`,
`efield`, `md`, `evidence`, `active`, `batch`, `transfer`, `klr`, `nonlinearity`, `sac`,
`solvent`, `cell_view`, `surface_view`, `design`, `diagnose`, `ranking`, `lowrank`,
`kcca_pocket`, and `speed`, whose table changed shape. The table of `multitask` moved as well
and was caught only through the README's copy of it, which no script rewrites.

One difference repeated itself exactly. `fluctuation` on thrombin gave q2 +0.302 on the author's
machine, +0.278 on the runner of 7 November and +0.278 again on the runner that night, with
2119 pseudo-atoms per variance field here and 2121 there. The conformers a runner generates are
not the conformers generated here, and they are the same on both runners, so this is a
systematic difference of the conformer generation and not noise of one run.

`benchmarks/results/threebody/table.md` was committed with CRLF newlines, which made it show as
rewritten in every job although not one of its numbers had moved. It is committed with LF now,
like every other table, and a test fails if a table is committed with CRLF again. The
comparisons count newlines as equal in any case.

### What the third run showed, 8 November 2026

Run 42454303810 is the first run of the whole scope with a comparison that can fail. It measured
everything: 85 script calls, every one exit 0, no `[skip]` line, 5 hours 23 minutes of runner
time (107, 74, 39, 34, 24, 23 and 22 minutes per group). `protonation` and `trajectory` ran for
the first time (6.6 and 7.1 minutes for the two pH values, 10.7 minutes for the trajectory), so
the three packages added to `requirements-lock.txt` were what they needed.

The check named **29 tables**. Comparing the tables of the seven result artifacts with the
committed ones afterwards, newlines ignored, gives the same 29 with the same figures, so the
comparison in the workflow can now be relied on.

**The reading of those 29, one table at a time.** Rule 4 of the manual applies: a difference
inside the interval of the quantity is not a result.

**Nine tables record time, a date or a file path, so they can never reproduce.** `speed` is
timings and ratios of timings throughout (its table also changed shape, 190 numbers against
180); `klr` reports milliseconds; `nystrom` reports the seconds a kernel took, while its q2
values are identical to four decimal places; `lowrank` said "8 times faster, 0.3 s against
2.3 s" and now says "4 times faster, 0.2 s against 1.0 s"; `trajectory` needed 176 s here and
140 s there; `sac` moved in its run time and nowhere else (1 number of 883); `evidence` moved in
its two timing columns only; `cell_view` and `surface_view` print the command line and the date
the table was written. `fe110` also prints two figure paths, which differ only in the separator
Windows and Linux use.

**Eighteen tables move because the conformers differ, or because a close choice flips.** Every
difference that has a stated interval falls inside it. `fe110` is the clearest: its overlap row
went from +0.089 +/- 0.064 to +0.097 +/- 0.062, and its largest move, -0.163 +/- 0.167 to
-0.106 +/- 0.080, is a third of one interval. `matbench_phonons` has no conformers, and moves
for the other reason: the five folds are fixed (`KFold` with the Matbench seed 18012019), but
the inner selection chose sigma 0.35 instead of 0.75 in one fold, which moved the MAE of
`physchem+mass` from 117.3 +/- 10.1 to 111.2 +/- 5.9 -- two intervals that cover each other.
`md` and `fluctuation` each change the sign of a difference whose interval covers zero in both
runs (`md`: -0.031 [-0.246, +0.433] against +0.202 [+0.035, +0.752]; `fluctuation`: -0.009
[-0.079, +0.062] against +0.004 [-0.056, +0.070]), so neither run makes that difference a
result. `restriction` finds 33 restricted pairs here and 38 there, on a grid of 12350 points
against 13000, both of which follow from the conformers. The remaining tables move in the third
decimal place or less: `topomer` 0.003, `kcca_pocket` 0.008, `transfer` 0.001, `anchors` 0.002,
and `design`, `diagnose`, `nonlinearity`, `multitask`, `corrosion`, `corrosion_hcl`,
`cyclodextrin`, `efield` and `solvent` (one number of 1039) likewise.

**Two tables are stale, and finding them is what the run is for.** Neither difference has
anything to do with the other machine.

`ranking`. The committed result files were measured with five repeats (`"repeats": 5`, repeat
numbers 0 to 4, which is the script's own default), and `benchmarks/run_all.py` registers the
command as `run_ranking.py --repeats 10`. The committed table says "5 random 70/30 splits each"
and the run says 10. So the committed numbers cannot be reproduced by the command the harness
runs. It is measured again with the registered ten repeats, and the manual, which quotes these
numbers, carries the new ones: on clean data Spearman's rho is +0.713 for kernel ridge against
+0.692 for the ranking SVM, ridge ahead on six of the eight sets instead of five, and told which
molecules share a source the ranking SVM beats ridge by 0.098 on seven of the eight sets instead
of 0.089 on six. The reading of the item does not change.

`sites`. The committed table says "measured targets dE_H 104" and the run says 133. The reason
is in the repository, not on the runner: `examples/sac/substrates_pbe.csv` gained 21 bare-sheet
energies on 5 November (commit 08189d9), and `benchmarks/results/sites/table.md` has not been
rebuilt since 1 November (commit 8cdb327). `sac` was measured again after that data arrived and
`sites` was not. It is measured again here. On the larger set the model of the OH binding
energy reaches q2 +0.908 at 0.444 eV on 133 catalysts, against +0.893 at 0.511 eV on 104, and the
reading of the item does not change: the site profile is level with every descriptor built from
fields and clearly better than the counts, now in nine of the twelve paired comparisons instead
of five.

Both repaired tables are byte for byte what the runner produced, so both benchmarks do reproduce
exactly on another machine and the stale tables were the whole of the difference.

**What is left of step 7** is the data statement: which tables another machine reproduced, which
moved and for which of the reasons above, and which record timings and cannot be compared between
machines at all.

### What is covered

**The 49 benchmarks left out** are of four kinds. Forty are long, 76 hours together: `poses`
16 h, `compas` 15 h, `spectra` 3.5 h, `host_pocket` 2.8 h, `gntk_compas` 2.2 h, `litraj` 2.1 h,
`battery`, `2dmd`, `scattering` and `microstructure` 2 h each, `topomer_mittal`, `increments`,
`rowlength`, `threebody` and `fieldae` 1.5 h each, and twenty-five more between 32 and 70
minutes. Four need ORCA, which no runner has (`am_eu`, `anisotropic`, `cdft_kernels`,
`corrosion_qm`). Two need xtb (`shells`, `xtb`). One needs PyBaMM, which
`requirements-lock.txt` does not pin (`cellmodel`). Two need files the repository does not
carry and are started with `--data-root` (`sutherland`, `local_scale`).

What is covered, by paper: the tables of papers 4, 7, 8, 12 and 18 in full, and most of papers 1,
3, 9 and 13. Of what is left out, `host_pocket` belongs to paper 13, `pharmfields` and
`pharmfields_mittal` to paper 3, `tasks` to paper 9, and the two reproduction tables of
`docs/benchmark.md` to paper 1. The seven benchmarks that the repair brought into the run add the
application figures of paper 1 (`cell_view`, `surface_view`), two of the three diagnostics of
paper 3 (`anchors`, `efield`), the catalyst series of paper 20 (`denmark`), the redox series of
paper 22 (`phenazine`) and the hop barriers of paper 17 (`hops`); `shells` is the one of the seven
that stays out, because it needs xtb. The claim the release makes is then exactly what was
tested: these tables were reproduced on a machine that is not the author's, those were not, and
the data statement says which is which.

## What the release promises

The supported interface of nine names, held by `tests/test_stable_surface.py`: within version 1
they are not removed or renamed, and no argument changes meaning. Everything else may change
between minor versions. The manual states this in *The stable surface*, under *Using it as a library*.
