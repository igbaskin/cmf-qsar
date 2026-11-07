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

Zenodo archives a GitHub release only if the connection existed before the release was created.
Sign in at zenodo.org with GitHub, open *GitHub* in the account menu, and switch the
`igbaskin/cmf-qsar` repository on. `.zenodo.json` in the repository root supplies the title, the
author, the licence and the keywords, so the archive does not depend on what GitHub guesses.

## 4. Tag and release

```bash
git tag -a v1.0.0 -m "cmfqsar 1.0.0"
git push origin v1.0.0
```

Then create the GitHub release from that tag and attach `docs/CMF_manual.pdf`, so the
supplementary material of every paper is one file with a published checksum. Zenodo mints the DOI
when the release is published.

## 5. Write the DOI down

The DOI arrives in two forms: one that always points at the newest version (the *concept* DOI)
and one for 1.0.0. Papers cite the concept DOI.

* `CITATION.cff`: add `doi:` and set `date-released` to the day of the release. It still says
  `2026-10-22`.
* `README.md`: the DOI badge and the citation line.
* `docs/publication_plan.md`: item 0 is then done.

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

`benchmarks/run_all.py` registers 89 benchmarks and 81.8 hours of computation, so this is not one
continuous-integration job: a GitHub runner stops a job after six hours. The scope chosen for the
release is **the 60 benchmarks that register 30 minutes or less and need no file the repository
does not carry, 15.2 hours in all**. `.github/workflows/reproduce.yml` runs them on demand in four
groups of about 3.8 hours, then rebuilds every table from the new result files and fails if a
table differs from the one the repository carries. That comparison is the reproduction test.

```bash
gh workflow run reproduce.yml -f group=all        # or group=1 to run one group
```

The 29 benchmarks left out are of three kinds. Twenty-seven are long: `poses` 16 h, `compas` 15 h,
`corrosion_qm` 5 h, `am_eu` 3 h, `host_pocket` 2.8 h, `2dmd`, `scattering`, `xtb`, `threebody`,
`topomer_mittal`, `anisotropic`, `increments` and `fieldae` between 1.5 and 2 h, and fourteen more
between 35 and 70 minutes. Two need ORCA, which no runner has (`cdft_kernels`, and `corrosion_qm`,
which is long as well). Two need files the repository does not carry and are started with
`--data-root` (`sutherland` and `local_scale`).

What is covered, by paper: the tables of papers 4, 7, 8, 12 and 18 in full, and most of papers 1,
3, 9 and 13. Of what is left out, `host_pocket` belongs to paper 13, `pharmfields` and
`pharmfields_mittal` to paper 3, `tasks` to paper 9, and the two reproduction tables of
`docs/benchmark.md` to paper 1. The claim the release makes is then exactly what was tested: these
tables were reproduced on a machine that is not the author's, those were not, and the data
statement says which is which.

## What the release promises

The supported interface of nine names, held by `tests/test_stable_surface.py`: within version 1
they are not removed or renamed, and no argument changes meaning. Everything else may change
between minor versions. The manual states this in *The stable surface*, under *Using it as a library*.
