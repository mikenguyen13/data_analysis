# A Guide on Data Analysis

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<p align="center">
  <img src="logo.png" alt="A Guide on Data Analysis logo" width="200"/>
</p>

A book on data analysis, causal inference, and econometrics, written for readers with little to no prior background who want to develop both statistical intuition and practical R skills. The print edition is published by Springer Cham as a four-volume series, and this repository hosts the open online edition that the print volumes are drawn from.

**Read it online**: <https://bookdown.org/mike/data_analysis/>

## What the book covers

The online edition runs from descriptive statistics through inference, regression, and causal inference, with a strong emphasis on the methods that have come to define modern empirical practice.

* **Foundations**: descriptive statistics, basic inference, hypothesis testing, sampling, ANOVA.
* **Regression**: OLS, GLS, MLE, penalized, robust, partial least squares.
* **Generalized linear and mixed models**: logistic, count, hierarchical.
* **Modeling workflow**: variable transformation, imputation, model specification, variable selection.
* **Effects and decomposition**: marginal effects, moderation, mediation, prediction vs. estimation.
* **Causal inference**: difference-in-differences with the modern staggered estimators (Goodman-Bacon, Callaway-Sant'Anna, Sun-Abraham, de Chaisemartin-D'Haultfoeuille, Borusyak-Jaravel-Spiess), regression discontinuity (incl. `rdrobust`), interrupted time series, synthetic control, synthetic DiD, changes-in-changes, event studies, instrumental variables (incl. examiner designs and proxies), matching, and DAGs.
* **Sensitivity and robustness**: specification curves, Oster coefficient stability, Cinelli-Hazlett robustness value (`sensemakr`), Rosenbaum bounds.
* **Workflow and reporting**: experimental design, EDA, reproducibility, synthetic data, HPC.

## Table of contents

The book is organized into five parts plus an appendix.

**Part I. Foundations**

1. Prerequisites
2. Descriptive Statistics
3. Basic Statistical Inference

**Part II. Regression**

4. Linear Regression
5. Non-Linear Regression
6. Generalized Linear Models
7. Linear Mixed Models
8. Nonlinear and Generalized Linear Mixed Models
9. Nonparametric Regression

**Part III. Ramifications**

10. Data
11. Variable Transformation
12. Imputation (Missing Data)
13. Model Specification Tests
14. Variable Selection
15. Hypothesis Testing
16. Marginal Effects
17. Moderation
18. Mediation
19. Prediction and Estimation

**Part IV. Causal Inference**

* *A. Experimental Design*

  20. Causal Inference
  21. Experimental Design
  22. Sampling
  23. Analysis of Variance
  24. Multivariate Methods

* *B. Quasi-Experimental Design*

  25. Quasi-Experimental Methods
  26. Regression Discontinuity
  27. Temporal Discontinuity Designs
  28. Synthetic Difference-in-Differences
  29. Difference-in-Differences
  30. Changes-in-Changes
  31. Synthetic Control
  32. Event Studies
  33. Instrumental Variables
  34. Matching Methods

* *C. Other Concerns*

  35. Endogeneity
  36. Other Biases
  37. Directed Acyclic Graphs
  38. Controls

**Part V. Miscellaneous**

39. Reporting Your Analysis
40. Exploratory Data Analysis
41. Sensitivity Analysis and Robustness Checks
42. Replication and Synthetic Data
43. High-Performance Computing

**Appendix**: bookdown cheat sheet and references.

## Print edition (Springer Cham, 2025)

The book is published in four volumes by Springer Cham:

| Volume | Title | Identifier |
|---|---|---|
| 1 | *Foundations of Data Analysis* | [DOI 10.1007/978-3-032-01858-8](https://doi.org/10.1007/978-3-032-01858-8) |
| 2 | *Regression Techniques for Data Analysis* | [ISBN 978-3-032-01834-2](https://link.springer.com/book/9783032018342) |
| 3 | *Advanced Modeling and Data Challenges* | [ISBN 978-3-032-01718-5](https://link.springer.com/book/9783032017185) |
| 4 | *Experimental Design* | [ISBN 978-3-032-01838-0](https://link.springer.com/book/9783032018380) |

## How to cite

> Nguyen, M. (2025). *Foundations of Data Analysis* (Vol. 1). Springer Cham. <https://doi.org/10.1007/978-3-032-01858-8>

```bibtex
@book{Nguyen2025Vol1,
  author    = {Nguyen, Mike},
  title     = {Foundations of Data Analysis},
  volume    = {1},
  year      = {2025},
  publisher = {Springer Cham},
  doi       = {10.1007/978-3-032-01858-8}
}
```

Citation entries for Volumes 2 to 4 are listed in the [preface of the online edition](https://bookdown.org/mike/data_analysis/#how-to-cite-these-books).

## Building locally

The book is built with [bookdown](https://bookdown.org/) and renders to HTML (gitbook), PDF (LuaLaTeX), and EPUB.

```r
install.packages(c("bookdown", "rmarkdown"))
bookdown::render_book(input = "index.Rmd")
```

The PDF route requires a working LaTeX installation with LuaLaTeX. On Windows, [TinyTeX](https://yihui.org/tinytex/) or MiKTeX both work; on macOS, MacTeX; on Linux, TeX Live.

The repo uses [renv](https://rstudio.github.io/renv/) for package version pinning. After cloning, run `renv::restore()` to recreate the package environment used by the book.

## Repository layout

```
.
├── index.Rmd               # title page and preface
├── 02-50*.Rmd              # numbered chapters
├── _bookdown.yml           # book configuration
├── _output.yml             # output formats (gitbook, PDF, EPUB, bs4_book)
├── preamble.tex            # LaTeX preamble for the PDF build
├── *.bib                   # bibliographies (book.bib, references.bib, packages.bib, references1.bib)
├── images/                 # figures and cover art
├── data/                   # datasets used in examples
├── renv/, renv.lock        # pinned package environment
└── _book/                  # rendered output (git-ignored)
```

Files prefixed with `_` (e.g., `_06.01-...`) are draft fragments not included in the live book.

## Errata

Spotted an error or have a suggestion? Please open an [issue](https://github.com/mikenguyen13/data_analysis/issues) on this repository.

## Author

Mike Nguyen, [mikenguyen.netlify.app](https://mikenguyen.netlify.app), GitHub [@mikenguyen13](https://github.com/mikenguyen13).

## License

The online edition is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The print editions of Volumes 1 to 4 published by Springer Cham (2025) are not covered by this license; all rights to the print editions are reserved by the publisher. See [LICENSE](LICENSE) for the full notice.
