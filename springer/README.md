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

Every chapter has six keywords. See the note on the count below.

## Checked against the published guidelines, 10 September 2026

The points above were taken from Banu Dhayalan's email. Reading the Manuscript
Guidelines on the Springer Nature site turned up four rules the email did not mention.

**Abstracts are capped at 200 words.** All eleven now run between 153 and 192. Two had
to be trimmed after the EDA and HPC chapters were expanded.

**Keyword phrases must each begin with a capital letter.** All eleven chapters were
lowercase throughout and have been corrected.

**The keyword count conflicted, and no longer needs resolving.** The email asks for 5 to
10 per chapter, the published guidelines say "we allow three to six". The chapters
carried 8 or 9, which satisfied the email and exceeded the guidelines. Every chapter now
carries exactly **six**, which satisfies both at once, so the question does not have to
be settled before delivery. Each list keeps the terms most distinctive to its chapter.

**References belong at the end of each chapter, not in the back matter.** The guidelines
are explicit that this is what makes citation linking work on SpringerLink.

Half fixed. `bs4_book` defaults `split_bib` to `FALSE`, which is why the volume was
producing a single consolidated bibliography; `gitbook` defaults it to `TRUE`. The fork's
`_output.yml` now sets `split_bib: true`, so the HTML edition carries references at the
end of every chapter.

The LaTeX deliverable still emits one `\bibliography` at the end, and that is a decision
rather than an omission. `pdf_book` has no `split_bib`. Doing it properly in LaTeX means
switching the manuscript from natbib to biblatex with `refsection=chapter`, which changes
the rendering of every in-text citation in a 519-page book, to solve a problem production
solves again anyway when it converts the manuscript to XML. Destabilizing a working build
three weeks before delivery is the wrong trade.

What production actually needs from that requirement is the mapping: which references
belong to which chapter. `split-bib-by-chapter.pl` writes it out directly, one `.bib` per
chapter in `bib-by-chapter/`, and ships alongside the `.tex`:

```
perl springer/split-bib-by-chapter.pl
```

It emits 211 entries across the eleven chapters, 202 of them unique, the difference being
the nine sources cited by two chapters, which belong in both lists. The 202 is the same
number `prune-bib.pl` arrives at independently, which is a useful cross-check on both.
Chapters 6 and 10 come out empty because they cite nothing.

Two further points from the guidelines, neither of them a problem. There is **no limit
on heading depth**, only a rule against skipping levels, so the four-deep subsections in
the endogeneity chapter are fine. And the accessibility rule, "do not just change the
color, also change shapes and patterns", is what the figure redesign described in
`figures-for-color-print.md` already does.

Figure file formats are an open question: the guidelines ask for EPS, or TIFF at 300 to
1200 dpi depending on the artwork, supplied as separate files. The build emits vector
PDF for the LaTeX output and 300 dpi PNG for the electronic version.

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
| 36-endogeneity.Rmd | Endogeneity | 6 | Endogeneity; Omitted variable bias; Measurement error; Simultaneity; Control function; Gaussian copula |
| 37-biases.Rmd | Biases | 6 | Aggregation bias; Simpson's paradox; Survivorship bias; Attrition bias; Publication bias; P-hacking |
| 38-dag.Rmd | Directed Acyclic Graphs | 6 | Directed acyclic graphs; D-separation; Back-door criterion; Front-door criterion; Collider bias; Causal discovery |
| 39-controls.Rmd | Controls | 6 | Control variables; Bad controls; Overcontrol bias; Bias amplification; Adjustment sets; Confounding |
| 40-report.Rmd | Reporting Your Analysis | 6 | Reproducible reporting; Regression tables; Model comparison; Coefficient plots; APA style; Publication-ready output |
| 41-EDA.Rmd | Exploratory Data Analysis | 6 | Exploratory data analysis; Data profiling; Feature engineering; Missing data; Outlier detection; Researcher degrees of freedom |
| 42-sensitivity-robustness.Rmd | Sensitivity Analysis and Robustness Checks | 6 | Sensitivity analysis; Robustness checks; Specification curve; Multiverse analysis; Robustness value; Rosenbaum bounds |
| 43-rep_synthetic_data.Rmd | Replication and Synthetic Data | 6 | Replication; Reproducibility; Synthetic data; Data sharing; Data confidentiality; Research transparency |
| 43.5-differential-privacy.Rmd | Differential Privacy | 6 | Differential privacy; Privacy loss; Laplace mechanism; Randomized response; Composition; Statistical disclosure control |
| 44-hpc.Rmd | High-Performance Computing | 6 | High-performance computing; Parallel computing; Distributed computing; Apache Spark; Profiling; Reproducibility |
| 45-clustered-inference.Rmd | Clustered and Robust Inference | 6 | Cluster-robust standard errors; Within-cluster correlation; Few clusters; Wild cluster bootstrap; Multi-way clustering; Design-based inference |

## The copy has been made, and it is now scripted

As of 10 September 2026 the standalone folder exists, builds, and is a git repository of
its own. The copy is no longer a one-time manual step: `scripts/sync-from-upstream.R`
in that folder re-derives it from this repository and can be run any number of times.

```powershell
.\scripts\sync.ps1
```

It copies the fourteen chapters, runs `prune-bib.pl` and takes the pruned bibliography,
and copies the build inputs both books share, which are `_common.R`, `preamble.tex`,
`_abstract_style.html` and `css/epub_style.css`. It then localizes the result the same
way `strip-outward-links.pl` does, and verifies that nothing dangles before reporting
`SYNC: PASS`.

It supersedes `strip-outward-links.pl` for that folder, for three reasons. It finds the
outward references by comparing used against defined labels rather than from a fixed
list, so a new cross-reference added upstream is handled without editing anything. It
keeps the twelve sentences whose wording differs between the two books in an explicit
table, and fails loudly if an upstream edit moves the text out from under one of them.
And it is idempotent, so it can be re-run after every upstream change rather than only
at the moment of the split. `strip-outward-links.pl` is kept here as the reference
implementation and as documentation of what the localization does.

`prune-bib.pl` has no counterpart in the new script and is called by it directly.

**Shared prose is edited here, never in the fork.** The fork carries `index.Rmd`,
`_bookdown.yml`, `_output.yml`, `google_analytics.html`, `style.css`, `DESCRIPTION`, its
appendices, and the rewrite table. Everything else arrives by sync.

### The original manual procedure, for reference

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
