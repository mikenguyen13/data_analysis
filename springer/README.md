# Springer volume: Practical Issues in Data Analysis and Reporting

Springer project 702652. Production editor Banu Dhayalan, editorial contact Eva Hiripi,
project coordination with Nino Sponagel. Planned initial manuscript delivery
**30 September 2026**.

This folder holds the preparation notes for splitting the Springer volume out of the
online book. It is documentation only. Nothing here is part of any bookdown build.

## Volume contents

Eleven chapters, drawn from Part C (Other Concerns) and Part V (Miscellaneous) of the
online book, plus the Introduction that already exists in
`../data_analysis_practice/index.Rmd`.

| Order | Source file(s) | Chapter |
|-------|----------------|---------|
| 1 | `36-endogeneity.Rmd` | Endogeneity |
| 2 | `37-biases.Rmd` | Biases |
| 3 | `38-dag.Rmd` | Directed Acyclic Graphs |
| 4 | `39-controls.Rmd` | Controls |
| 5 | `40-report.Rmd` | Reporting Your Analysis |
| 6 | `41-EDA.Rmd` | Exploratory Data Analysis |
| 7 | `42-sensitivity-robustness.Rmd`, `42.5-placebo-falsification.Rmd`, `42.6-publication-bias-phacking.Rmd`, `42.7-robustness-conclusion.Rmd` | Sensitivity Analysis and Robustness Checks |
| 8 | `43-rep_synthetic_data.Rmd` | Replication and Synthetic Data |
| 9 | `43.5-differential-privacy.Rmd` | Differential Privacy |
| 10 | `44-hpc.Rmd` | High-Performance Computing |
| 11 | `45-clustered-inference.Rmd` | Clustered and Robust Inference |

Files 42.5 and 42.6 are `##` level sections of chapter 7, not chapters of their own.
They must be copied alongside `42-sensitivity-robustness.Rmd` and must keep their
alphabetical filenames so bookdown globs them in the right order.

## What has been done in the shared source

All of the following now lives in the main repository, so the online book and the
Springer volume stay in sync until the copy is made.

**Abstracts and keywords.** Each of the eleven chapters now opens, immediately after
its title, with a block of the form:

```
::: {.chapter-abstract}
**Abstract.** ...

**Keywords:** ...
:::
```

The fenced div renders as `<div class="chapter-abstract">` in HTML, so the block can be
extracted mechanically for the Springer submission, and passes through pandoc's LaTeX
writer as ordinary paragraphs, so the PDF build is unaffected. Both were verified
against the actual pandoc binary the book builds with. Styling comes from
`_abstract_style.html`, a head include added to the `bs4_book` block of `_output.yml`,
which avoids touching bookdown's own stylesheet handling.

Every chapter has 8 or 9 keywords, inside Springer's 5 to 10 range.

**Cross-references that will not survive the split.** Twenty-one `\@ref()` and table
references in these chapters point at chapters that stay behind in the online book, such as
instrumental variables, matching, and difference in differences. All twenty-one have been
rewritten so the sentence names the topic in words and the reference sits in a trailing
parenthetical. The prose therefore reads correctly whether or not the reference is
present, and the fork strips them with the script described below. Ten sentences needed
real rewriting; the rest already had the right shape.

## Keyword index

| File | Chapter | Count | Keywords |
|------|---------|-------|----------|
| 36-endogeneity.Rmd | Endogeneity | 9 | endogeneity; measurement error; simultaneity; omitted variable bias; Gaussian copula; control function; sample selection; Heckman correction; exclusion restriction |
| 37-biases.Rmd | Biases | 9 | aggregation bias; Simpson's paradox; contamination bias; survivorship bias; attrition bias; recall bias; publication bias; p-hacking; selection models |
| 38-dag.Rmd | Directed Acyclic Graphs | 9 | directed acyclic graphs; d-separation; back-door criterion; front-door criterion; collider bias; M-bias; confounding; causal discovery; structure learning |
| 39-controls.Rmd | Controls | 9 | control variables; bad controls; overcontrol bias; bias amplification; collider bias; neutral controls; adjustment sets; confounding; variance inflation |
| 40-report.Rmd | Reporting Your Analysis | 8 | reproducible reporting; regression tables; cluster-robust standard errors; model comparison; coefficient plots; APA style; descriptive statistics; publication-ready output |
| 41-EDA.Rmd | Exploratory Data Analysis | 8 | exploratory data analysis; data profiling; feature engineering; missing data; outlier detection; summary statistics; automated reporting; interactive visualization |
| 42-sensitivity-robustness.Rmd | Sensitivity Analysis and Robustness Checks | 9 | sensitivity analysis; robustness checks; specification curve; multiverse analysis; coefficient stability; omitted variable bias; robustness value; Rosenbaum bounds; placebo tests |
| 43-rep_synthetic_data.Rmd | Replication and Synthetic Data | 8 | replication; reproducibility; replication standard; data sharing; synthetic data; synthpop; data confidentiality; research transparency |
| 43.5-differential-privacy.Rmd | Differential Privacy | 9 | differential privacy; privacy loss; epsilon; Laplace mechanism; Gaussian mechanism; exponential mechanism; randomized response; composition; statistical disclosure control |
| 44-hpc.Rmd | High-Performance Computing | 9 | high-performance computing; parallel computing; future; foreach; Apache Spark; distributed computing; profiling; scalability; resource estimation |
| 45-clustered-inference.Rmd | Clustered and Robust Inference | 9 | cluster-robust standard errors; within-cluster correlation; few clusters; wild cluster bootstrap; multi-way clustering; spatial correlation; design-based inference; fixest; statistical inference |

## When you make the copy

Copy the fourteen `.Rmd` files listed above, plus `_common.R`, the four `.bib` files,
`preamble.tex`, `logo.png`, and `style.css`, into the standalone folder. Keep the
existing `index.Rmd`, `_bookdown.yml`, and `_output.yml` there, and add
`_abstract_style.html` to the `in_header` list of that folder's `bs4_book` block so the
abstracts pick up their styling.

Two steps then make the volume self-contained.

**1. Run `strip-outward-links.pl`.** Copy it into the standalone folder alongside the
chapters and run it there:

```
perl strip-outward-links.pl            # report only, changes nothing
perl strip-outward-links.pl --write    # apply
```

It does two things. It deletes the 21 outward `\@ref()` references, each of which sits
in a trailing parenthetical after a sentence that already names the topic in words. And
it flattens the 66 outward markdown anchor links of the form `[text](#sec-something)`
down to their own link text, which always names the topic. Anchor links fail silently
rather than breaking the build, so they would otherwise ship as dead links in the
printed and online editions.

This has been tested end to end on a scratch copy of the eleven chapters. Afterwards,
`check-xrefs.R` run inside the standalone folder reports:

```
XREF:  102 references, 342 labels defined, 0 dangling   PASS
LINKS: 34 markdown links, 890 anchors reachable, 0 dead  PASS
```


**2. Use the pruned bibliography.** `references-springer.bib` in this folder is already
built: 202 entries, exactly the ones the eleven chapters cite, down from 1,825 entries and
638 KB across the book's four `.bib` files to 73 KB. Copy it into the standalone folder
and replace the four-file list in `index.Rmd` with a single line:

```yaml
bibliography: references-springer.bib
```

Regenerate it after any citation change by running `perl springer/prune-bib.pl` from the
main repository root. The script re-reads the chapters and the four source bibliographies
and exits non-zero if any cited key has no entry.

### The whole procedure has been dry-run

The steps above were executed end to end against a throwaway copy, not just reasoned
about. A scratch folder was built exactly as described: the chapter files, this
folder's `index.Rmd`, `_bookdown.yml`, `_output.yml`, `_common.R` and assets, and the
pruned bibliography. Then `strip-outward-links.pl --write` ran, `index.Rmd` was pointed
at the single bibliography, and both checks were run on the result:

```
strip:  21 outward references, 64 outward anchor links removed
XREF:   102 references, 345 labels defined, 0 dangling   PASS
LINKS:  36 markdown links, 900 anchors reachable, 0 dead  PASS
citeproc against references-springer.bib alone:   0 missing citations
```

What this does not cover: no chapter was knitted in the scratch folder, so it proves the
volume is internally consistent as a document, not that every code chunk runs there. The
chunks do run, but that was verified in the main repository, where the four chapters
carrying substantive edits were rendered individually and passed.

## Editorial items, all resolved

1. **The Introduction has been rewritten.** `../data_analysis_practice/index.Rmd` now opens
   as a standalone volume rather than as "the final section of this book, and of the
   series as a whole". It covers all eleven chapters in order, explains why clustered
   inference sits last despite arguably belonging first, names the intended reader, and
   states that every result is reproducible from the printed code. It carries its own
   abstract and keywords in the same format as the chapters. The YAML title is now
   "Practical Issues in Data Analysis and Reporting", matching the Springer project, and
   `github-repo` and `description` were corrected. The previous version is backed up in
   this session's scratchpad.

2. **Both dead-in-standalone links now point in-volume.** `39-controls.Rmd` links
   "Coefficient Stability" to `#sec-coefficient-stability`, the fuller Oster treatment in
   chapter 7, instead of the matching methods chapter. Chapter 7's own Rosenbaum section
   was given the explicit anchor `{#sec-rosenbaum-bounds-sensitivity}` and the robustness
   map now points there. Nothing linked to the old auto-generated `#rosenbaum-bounds`
   anchor, so nothing broke.

3. **Spelling is now consistently American.** Eighteen substitutions across four
   chapters: modelled, modelling, summarises, summarised, organises, neighbourhood(s),
   colour, centre, generalise(s), behaviour, analyse, recognises. Only prose was touched.
   Code chunks, inline code, and citation keys were excluded by the scanner, so ggplot's
   `colour` aesthetic and similar are untouched. A rescan reports zero remaining.

4. **The unsourced epigraph is gone.** The Ronald Coase line at the head of the p-Hacking
   section has been replaced with an original line of the author's, which keeps the
   rhetorical function without an attribution nobody can verify. See
   `third-party-permissions.md`.

5. **Three captions no longer depend on color.** Where a figure is readable in grayscale
   because its series are separated by linetype, the caption said "red line" anyway. Fixed
   in `42-sensitivity-robustness.Rmd` (twice) and `37-biases.Rmd` (once). The two
   remaining color words in captions belong to figures that are on the must-print-in-color
   list, so they are consistent with that request.

6. **Missing assets copied into the standalone folder.** `favicon.ico`, `images/cover.jpg`,
   and `css/epub_style.css` were referenced by that folder's config but absent, which
   would have failed the build. They have been copied across, along with
   `_abstract_style.html`, which is now wired into that folder's `bs4_book` block.

7. **The gitbook block in the standalone `_output.yml` is commented out.** That folder
   listed `bookdown::gitbook` first and uncommented, so a bare `render_book()` there built
   gitbook rather than the published bs4_book format. It is now disabled, with a header
   note explaining the first-format-wins behavior, mirroring the main repository. The
   remaining formats parse cleanly and `bs4_book` keeps its `_abstract_style.html`
   include. Note the consequence: `bookdown::pdf_book` is now the first uncommented
   format, so a bare `render_book()` there starts a LaTeX build, exactly as in the main
   repository. Name the format on every render:

   ```r
   bookdown::render_book(input = "index.Rmd", output_format = "bookdown::bs4_book")
   ```

8. **The bibliography is pruned and verified.** See above. Verification was not a key
   count: every chapter file was concatenated and run through pandoc with
   `--citeproc` against the pruned bibliography alone, which reports zero missing
   citations. Because that runs over the raw `.Rmd`, it also catches any citation sitting
   inside a code chunk that a key-extracting regex would miss. The check was confirmed to
   fail on a deliberately bogus key, so a clean result means something.

## What still needs you

1. **The site URL in `../data_analysis_practice/index.Rmd` is wrong.** It still reads
   `url: https\://bookdown.org/mike/data_analysis/`, which is the other book. Left alone
   because guessing the deployment target would put a wrong canonical URL in the
   metadata. Irrelevant to the Springer print manuscript; matters only if you publish this
   volume as its own site.

2. **The cover image is a placeholder.** `images/cover.jpg` is the main book's cover.

3. **The chapter files in that folder are a May 2025 snapshot** and are missing 42.5,
   42.6, 43.5 and 45 entirely. The copy must overwrite them.
