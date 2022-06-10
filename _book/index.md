---
title: "A Guide on Data Analysis"
author: "Mike Nguyen"
date: "2022-06-10"
header-includes:
  - \usepackage{titling}
  - \pretitle{\begin{center}
    \includegraphics[width=2in,height=2in]{logo.png}\LARGE\\}
  - \posttitle{\end{center}}
documentclass: book
bibliography:
- book.bib
- packages.bib
- references.bib
biblio-style: apalike
link-citations: true
link-as-notes: true
description: This is a guide on how to conduct data analysis in the field of data science, statistics, or machine learning.
github-repo: mikenguyen13/data_analysis
url: https\://bookdown.org/mike/data_analysis/
favicon: "favicon.ico"
cover-image: "images/cover.jpg"
apple-touch-icon: "logo.png"
apple-touch-icon-size: 120
knit: "bookdown::render_book"
site: bookdown::bookdown_site
---

# Preface {#preface .unnumbered}



<img src="images/cover.jpg" class="cover" width="250" height="328"/> This guide is an attempt to streamline and demystify the data analysis process. By no means this is an ultimate guide, or I am a great source of knowledge, or I claim myself to be a statistician/ econometrician, but I am a strong proponent of learning by teaching, and doing. Hence, this is more like a learning experience for both you and me. This book is completely free. My target audiences are those who have little to no experience in statistics and data science to those that have some interests in these fields and want to dive deeper and have a more holistic method. Even though my substantive domain of interest is marketing, this book can also be used for other disciplines that use scientific methods or data analysis.

<br>

<img src="logo.png" width="25%" style="display: block; margin: auto;" />

More books by the author can be found [here](https://mikenguyen.netlify.app/books/written_books/):

-   [Advanced Data Analysis](https://bookdown.org/mike/advanced_data_analysis/): the second book in the data analysis series, which covers machine learning models (with a focus on prediction)
-   [Marketing Research](https://bookdown.org/mike/marketing_research/)
-   [Communication Theory](https://bookdown.org/mike/comm_theory/)

# Introduction

Since the beginning of the century, we have been bombarded with amazing advancements and inventions, especially in the field of statistics, information technology, computer science, or a new emerging filed - data science. However, I believe the downside of this introduction is that we use **big** and **trendy** words too often (i.e., big data, machine learning, deep learning).

Each substantive field will have a metric subfield:

-   Econometrics in economics

-   Psychometrics in psychology

-   Chemometrics in chemistry

-   Sabermetrics in sports

-   Biostatistics in public health and medicine

But to laymen, these are known as:

-   Data Science

-   Applied Statistics

-   Computational Social Science

It's all fun and exciting when I learned these new tools. But I have to admit that I hardly retain any of these new ideas. However, writing down from the beginning till the end of a data analysis process is the solution that I came up with. Accordingly, let's dive right in.

<img src="images/meme.jpg" width="90%" style="display: block; margin: auto;" />

<br>


**Some general recommendations**:

-   The more you practice/habituate/condition, more line of codes that you write, more function that you memorize, I think the more you will like this journey.

-   Readers can follow this book several ways:

    -   If you are interested in particular methods/tools, you can jump to that section by clicking the section name.
    -   If you want to follow a traditional path of data analysis, read the [Linear Regression] section.
    -   If you want to create your experiment and test your hypothesis, read the [Analysis of Variance (ANOVA)] section.

-   Alternatively, if you rather see the application of models, and disregard any theory or underlying mechanisms, you can skip to summary and application portion of each section.

-   If you don't understand a part, search the title of that part of that part on Google, and read more into that subject. This is just a general guide.

-   If you want to customize your code beyond the ones provided in this book, run in the console `help(code)` or `?code`. For example, I want more information on `hist` function, I'll type in the console `?hist` or `help(hist)`.

-   Another way is that you can search on Google. Different people will use different packages to achieve the same result in R. Accordingly, if you want to create a histogram, search on Google `histogram in R`, then you should be able to find multiple ways to create histogram in R.

Information in this book are from various sources, but most of the content is based on several courses that I have taken formally. I'd like to give professors credit accordingly.

| Course              | Professor         |
|---------------------|-------------------|
| Data Analysis I     | Erin M. Schliep   |
| Data Analysis II    | Christopher Wikle |
| Applied Econometric | Alyssa Carlson    |

**Tools of statistics**

-   Probability Theory
-   Mathematical Analysis
-   Computer Science
-   Numerical Analysis
-   Database Management






<br>

**Code Replication**

This book was built with R version 4.0.4 (2021-02-15) and the following packages:


|package      |version    |source         |
|:------------|:----------|:--------------|
|abind        |1.4-5      |CRAN (R 4.0.3) |
|agridat      |1.20       |CRAN (R 4.0.5) |
|ape          |5.6-1      |CRAN (R 4.0.5) |
|assertthat   |0.2.1      |CRAN (R 4.0.3) |
|backports    |1.4.1      |CRAN (R 4.0.5) |
|bookdown     |0.24       |CRAN (R 4.0.5) |
|boot         |1.3-28     |CRAN (R 4.0.5) |
|broom        |0.7.12     |CRAN (R 4.0.5) |
|bslib        |0.3.1      |CRAN (R 4.0.5) |
|cachem       |1.0.6      |CRAN (R 4.0.5) |
|callr        |3.7.0      |CRAN (R 4.0.5) |
|car          |3.0-12     |CRAN (R 4.0.5) |
|carData      |3.0-5      |CRAN (R 4.0.5) |
|cellranger   |1.1.0      |CRAN (R 4.0.3) |
|cli          |3.2.0      |CRAN (R 4.0.4) |
|coda         |0.19-4     |CRAN (R 4.0.3) |
|colorspace   |2.0-3      |CRAN (R 4.0.4) |
|corpcor      |1.6.10     |CRAN (R 4.0.5) |
|crayon       |1.5.0      |CRAN (R 4.0.5) |
|cubature     |2.0.4.2    |CRAN (R 4.0.5) |
|curl         |4.3.2      |CRAN (R 4.0.5) |
|data.table   |1.14.2     |CRAN (R 4.0.5) |
|DBI          |1.1.2      |CRAN (R 4.0.5) |
|dbplyr       |2.1.1      |CRAN (R 4.0.5) |
|desc         |1.4.0      |CRAN (R 4.0.5) |
|devtools     |2.4.3      |CRAN (R 4.0.5) |
|digest       |0.6.29     |CRAN (R 4.0.5) |
|dplyr        |1.0.8      |CRAN (R 4.0.4) |
|ellipsis     |0.3.2      |CRAN (R 4.0.5) |
|evaluate     |0.15       |CRAN (R 4.0.4) |
|extrafont    |0.17       |CRAN (R 4.0.3) |
|extrafontdb  |1.0        |CRAN (R 4.0.3) |
|fansi        |1.0.2      |CRAN (R 4.0.5) |
|faraway      |1.0.7      |CRAN (R 4.0.3) |
|fastmap      |1.1.0      |CRAN (R 4.0.3) |
|forcats      |0.5.1      |CRAN (R 4.0.3) |
|foreign      |0.8-82     |CRAN (R 4.0.5) |
|fs           |1.5.2      |CRAN (R 4.0.5) |
|generics     |0.1.2      |CRAN (R 4.0.5) |
|ggplot2      |3.3.5      |CRAN (R 4.0.5) |
|glue         |1.6.1      |CRAN (R 4.0.5) |
|gtable       |0.3.0      |CRAN (R 4.0.3) |
|haven        |2.4.3      |CRAN (R 4.0.5) |
|Hmisc        |4.6-0      |CRAN (R 4.0.5) |
|hms          |1.1.1      |CRAN (R 4.0.5) |
|htmltools    |0.5.2      |CRAN (R 4.0.5) |
|htmlwidgets  |1.5.4      |CRAN (R 4.0.5) |
|httr         |1.4.2      |CRAN (R 4.0.3) |
|investr      |1.4.0      |CRAN (R 4.0.3) |
|jpeg         |0.1-9      |CRAN (R 4.0.5) |
|jquerylib    |0.1.4      |CRAN (R 4.0.5) |
|jsonlite     |1.7.3      |CRAN (R 4.0.5) |
|kableExtra   |1.3.4      |CRAN (R 4.0.4) |
|knitr        |1.37       |CRAN (R 4.0.5) |
|lattice      |0.20-45    |CRAN (R 4.0.5) |
|latticeExtra |0.6-29     |CRAN (R 4.0.3) |
|lifecycle    |1.0.1      |CRAN (R 4.0.5) |
|lme4         |1.1-28     |CRAN (R 4.0.4) |
|lmerTest     |3.1-3      |CRAN (R 4.0.3) |
|lsr          |0.5.2      |CRAN (R 4.0.5) |
|ltm          |1.2-0      |CRAN (R 4.0.4) |
|lubridate    |1.8.0      |CRAN (R 4.0.5) |
|magrittr     |2.0.2      |CRAN (R 4.0.5) |
|MASS         |7.3-55     |CRAN (R 4.0.5) |
|matlib       |0.9.5      |CRAN (R 4.0.5) |
|Matrix       |1.4-0      |CRAN (R 4.0.5) |
|MCMCglmm     |2.33       |CRAN (R 4.0.5) |
|memoise      |2.0.1      |CRAN (R 4.0.5) |
|mgcv         |1.8-38     |CRAN (R 4.0.5) |
|minqa        |1.2.4      |CRAN (R 4.0.3) |
|modelr       |0.1.8      |CRAN (R 4.0.3) |
|munsell      |0.5.0      |CRAN (R 4.0.3) |
|nlme         |3.1-155    |CRAN (R 4.0.5) |
|nloptr       |2.0.0      |CRAN (R 4.0.5) |
|nlstools     |2.0-0      |CRAN (R 4.0.5) |
|nnet         |7.3-17     |CRAN (R 4.0.5) |
|numDeriv     |2016.8-1.1 |CRAN (R 4.0.3) |
|openxlsx     |4.2.5      |CRAN (R 4.0.5) |
|pbkrtest     |0.5.1      |CRAN (R 4.0.5) |
|pillar       |1.7.0      |CRAN (R 4.0.5) |
|pkgbuild     |1.3.1      |CRAN (R 4.0.5) |
|pkgconfig    |2.0.3      |CRAN (R 4.0.3) |
|pkgload      |1.2.4      |CRAN (R 4.0.5) |
|png          |0.1-7      |CRAN (R 4.0.3) |
|ppsr         |0.0.2      |CRAN (R 4.0.5) |
|prettyunits  |1.1.1      |CRAN (R 4.0.3) |
|processx     |3.5.2      |CRAN (R 4.0.5) |
|ps           |1.6.0      |CRAN (R 4.0.5) |
|pscl         |1.5.5      |CRAN (R 4.0.3) |
|purrr        |0.3.4      |CRAN (R 4.0.3) |
|R6           |2.5.1      |CRAN (R 4.0.5) |
|RColorBrewer |1.1-2      |CRAN (R 4.0.3) |
|Rcpp         |1.0.8      |CRAN (R 4.0.5) |
|readr        |2.1.2      |CRAN (R 4.0.5) |
|readxl       |1.3.1      |CRAN (R 4.0.3) |
|remotes      |2.4.2      |CRAN (R 4.0.5) |
|reprex       |2.0.1      |CRAN (R 4.0.5) |
|rgl          |0.108.3    |CRAN (R 4.0.5) |
|rio          |0.5.29     |CRAN (R 4.0.5) |
|rlang        |1.0.1      |CRAN (R 4.0.5) |
|RLRsim       |3.1-6      |CRAN (R 4.0.4) |
|rmarkdown    |2.11       |CRAN (R 4.0.5) |
|rprojroot    |2.0.2      |CRAN (R 4.0.3) |
|rstudioapi   |0.13       |CRAN (R 4.0.3) |
|Rttf2pt1     |1.3.10     |CRAN (R 4.0.5) |
|rvest        |1.0.2      |CRAN (R 4.0.5) |
|sass         |0.4.0      |CRAN (R 4.0.5) |
|scales       |1.1.1      |CRAN (R 4.0.3) |
|sessioninfo  |1.2.2      |CRAN (R 4.0.5) |
|stringi      |1.7.6      |CRAN (R 4.0.5) |
|stringr      |1.4.0      |CRAN (R 4.0.3) |
|svglite      |2.1.0      |CRAN (R 4.0.5) |
|systemfonts  |1.0.4      |CRAN (R 4.0.5) |
|tensorA      |0.36.2     |CRAN (R 4.0.3) |
|testthat     |3.1.2      |CRAN (R 4.0.5) |
|tibble       |3.1.6      |CRAN (R 4.0.5) |
|tidyr        |1.2.0      |CRAN (R 4.0.5) |
|tidyselect   |1.1.2      |CRAN (R 4.0.4) |
|tidyverse    |1.3.1      |CRAN (R 4.0.5) |
|tzdb         |0.2.0      |CRAN (R 4.0.5) |
|usethis      |2.1.5      |CRAN (R 4.0.5) |
|utf8         |1.2.2      |CRAN (R 4.0.5) |
|vctrs        |0.3.8      |CRAN (R 4.0.5) |
|viridisLite  |0.4.0      |CRAN (R 4.0.5) |
|webshot      |0.5.2      |CRAN (R 4.0.5) |
|withr        |2.4.3      |CRAN (R 4.0.5) |
|xfun         |0.29       |CRAN (R 4.0.5) |
|xml2         |1.3.3      |CRAN (R 4.0.5) |
|xtable       |1.8-4      |CRAN (R 4.0.3) |
|yaml         |2.3.4      |CRAN (R 4.0.4) |
|zip          |2.2.0      |CRAN (R 4.0.5) |

<br>


```
#> - Session info ---------------------------------------------------------------
#>  setting  value
#>  version  R version 4.0.4 (2021-02-15)
#>  os       Windows 10 x64 (build 19043)
#>  system   x86_64, mingw32
#>  ui       RTerm
#>  language (EN)
#>  collate  English_United States.1252
#>  ctype    English_United States.1252
#>  tz       America/Chicago
#>  date     2022-02-21
#>  pandoc   2.17.1.1 @ C:/Program Files/RStudio/bin/quarto/bin/ (via rmarkdown)
#> 
#> - Packages -------------------------------------------------------------------
#>  package     * version date (UTC) lib source
#>  assertthat    0.2.1   2019-03-21 [2] CRAN (R 4.0.3)
#>  backports     1.4.1   2021-12-13 [1] CRAN (R 4.0.5)
#>  bookdown      0.24    2021-09-02 [2] CRAN (R 4.0.5)
#>  brio          1.1.3   2021-11-30 [1] CRAN (R 4.0.5)
#>  broom         0.7.12  2022-01-28 [1] CRAN (R 4.0.5)
#>  bslib         0.3.1   2021-10-06 [1] CRAN (R 4.0.5)
#>  cachem        1.0.6   2021-08-19 [2] CRAN (R 4.0.5)
#>  callr         3.7.0   2021-04-20 [2] CRAN (R 4.0.5)
#>  cellranger    1.1.0   2016-07-27 [2] CRAN (R 4.0.3)
#>  cli           3.2.0   2022-02-14 [1] CRAN (R 4.0.4)
#>  codetools     0.2-18  2020-11-04 [2] CRAN (R 4.0.4)
#>  colorspace    2.0-3   2022-02-21 [1] CRAN (R 4.0.4)
#>  crayon        1.5.0   2022-02-14 [2] CRAN (R 4.0.5)
#>  DBI           1.1.2   2021-12-20 [1] CRAN (R 4.0.5)
#>  dbplyr        2.1.1   2021-04-06 [2] CRAN (R 4.0.5)
#>  desc          1.4.0   2021-09-28 [1] CRAN (R 4.0.5)
#>  devtools      2.4.3   2021-11-30 [1] CRAN (R 4.0.5)
#>  digest        0.6.29  2021-12-01 [1] CRAN (R 4.0.5)
#>  dplyr       * 1.0.8   2022-02-08 [1] CRAN (R 4.0.4)
#>  ellipsis      0.3.2   2021-04-29 [2] CRAN (R 4.0.5)
#>  evaluate      0.15    2022-02-18 [2] CRAN (R 4.0.4)
#>  fansi         1.0.2   2022-01-14 [1] CRAN (R 4.0.5)
#>  fastmap       1.1.0   2021-01-25 [2] CRAN (R 4.0.3)
#>  forcats     * 0.5.1   2021-01-27 [2] CRAN (R 4.0.3)
#>  fs            1.5.2   2021-12-08 [1] CRAN (R 4.0.5)
#>  generics      0.1.2   2022-01-31 [1] CRAN (R 4.0.5)
#>  ggplot2     * 3.3.5   2021-06-25 [2] CRAN (R 4.0.5)
#>  glue          1.6.1   2022-01-22 [1] CRAN (R 4.0.5)
#>  gtable        0.3.0   2019-03-25 [2] CRAN (R 4.0.3)
#>  haven         2.4.3   2021-08-04 [2] CRAN (R 4.0.5)
#>  highr         0.9     2021-04-16 [2] CRAN (R 4.0.5)
#>  hms           1.1.1   2021-09-26 [1] CRAN (R 4.0.5)
#>  htmltools     0.5.2   2021-08-25 [2] CRAN (R 4.0.5)
#>  httr          1.4.2   2020-07-20 [2] CRAN (R 4.0.3)
#>  jpeg        * 0.1-9   2021-07-24 [2] CRAN (R 4.0.5)
#>  jquerylib     0.1.4   2021-04-26 [2] CRAN (R 4.0.5)
#>  jsonlite      1.7.3   2022-01-17 [1] CRAN (R 4.0.5)
#>  knitr         1.37    2021-12-16 [1] CRAN (R 4.0.5)
#>  lifecycle     1.0.1   2021-09-24 [2] CRAN (R 4.0.5)
#>  lubridate     1.8.0   2021-10-07 [1] CRAN (R 4.0.5)
#>  magrittr      2.0.2   2022-01-26 [1] CRAN (R 4.0.5)
#>  memoise       2.0.1   2021-11-26 [1] CRAN (R 4.0.5)
#>  modelr        0.1.8   2020-05-19 [2] CRAN (R 4.0.3)
#>  munsell       0.5.0   2018-06-12 [2] CRAN (R 4.0.3)
#>  pillar        1.7.0   2022-02-01 [1] CRAN (R 4.0.5)
#>  pkgbuild      1.3.1   2021-12-20 [1] CRAN (R 4.0.5)
#>  pkgconfig     2.0.3   2019-09-22 [2] CRAN (R 4.0.3)
#>  pkgload       1.2.4   2021-11-30 [1] CRAN (R 4.0.5)
#>  prettyunits   1.1.1   2020-01-24 [2] CRAN (R 4.0.3)
#>  processx      3.5.2   2021-04-30 [2] CRAN (R 4.0.5)
#>  ps            1.6.0   2021-02-28 [2] CRAN (R 4.0.5)
#>  purrr       * 0.3.4   2020-04-17 [2] CRAN (R 4.0.3)
#>  R6            2.5.1   2021-08-19 [2] CRAN (R 4.0.5)
#>  Rcpp          1.0.8   2022-01-13 [2] CRAN (R 4.0.5)
#>  readr       * 2.1.2   2022-01-30 [1] CRAN (R 4.0.5)
#>  readxl        1.3.1   2019-03-13 [2] CRAN (R 4.0.3)
#>  remotes       2.4.2   2021-11-30 [1] CRAN (R 4.0.5)
#>  reprex        2.0.1   2021-08-05 [2] CRAN (R 4.0.5)
#>  rlang         1.0.1   2022-02-03 [1] CRAN (R 4.0.5)
#>  rmarkdown     2.11    2021-09-14 [2] CRAN (R 4.0.5)
#>  rprojroot     2.0.2   2020-11-15 [2] CRAN (R 4.0.3)
#>  rstudioapi    0.13    2020-11-12 [2] CRAN (R 4.0.3)
#>  rvest         1.0.2   2021-10-16 [1] CRAN (R 4.0.5)
#>  sass          0.4.0   2021-05-12 [2] CRAN (R 4.0.5)
#>  scales      * 1.1.1   2020-05-11 [2] CRAN (R 4.0.3)
#>  sessioninfo   1.2.2   2021-12-06 [1] CRAN (R 4.0.5)
#>  stringi       1.7.6   2021-11-29 [1] CRAN (R 4.0.5)
#>  stringr     * 1.4.0   2019-02-10 [2] CRAN (R 4.0.3)
#>  testthat      3.1.2   2022-01-20 [1] CRAN (R 4.0.5)
#>  tibble      * 3.1.6   2021-11-07 [1] CRAN (R 4.0.5)
#>  tidyr       * 1.2.0   2022-02-01 [1] CRAN (R 4.0.5)
#>  tidyselect    1.1.2   2022-02-21 [1] CRAN (R 4.0.4)
#>  tidyverse   * 1.3.1   2021-04-15 [2] CRAN (R 4.0.5)
#>  tzdb          0.2.0   2021-10-27 [1] CRAN (R 4.0.5)
#>  usethis       2.1.5   2021-12-09 [1] CRAN (R 4.0.5)
#>  utf8          1.2.2   2021-07-24 [2] CRAN (R 4.0.5)
#>  vctrs         0.3.8   2021-04-29 [2] CRAN (R 4.0.5)
#>  withr         2.4.3   2021-11-30 [1] CRAN (R 4.0.5)
#>  xfun          0.29    2021-12-14 [1] CRAN (R 4.0.5)
#>  xml2          1.3.3   2021-11-30 [1] CRAN (R 4.0.5)
#>  yaml          2.3.4   2022-02-17 [1] CRAN (R 4.0.4)
#> 
#>  [1] C:/Users/tn9k4/OneDrive - University of Missouri/Documents/R/win-library/4.0
#>  [2] C:/Program Files/R/R-4.0.4/library
#> 
#> ------------------------------------------------------------------------------
```
