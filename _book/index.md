---
title: "A Guide on Data Analysis"
author: "Mike Nguyen"
date: "2022-02-01"
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
|agridat      |1.18       |CRAN (R 4.0.3) |
|ape          |5.5        |CRAN (R 4.0.5) |
|assertthat   |0.2.1      |CRAN (R 4.0.3) |
|backports    |1.2.1      |CRAN (R 4.0.3) |
|bookdown     |0.24       |CRAN (R 4.0.5) |
|boot         |1.3-28     |CRAN (R 4.0.5) |
|broom        |0.7.9      |CRAN (R 4.0.5) |
|bslib        |0.3.1      |CRAN (R 4.0.5) |
|cachem       |1.0.6      |CRAN (R 4.0.5) |
|callr        |3.7.0      |CRAN (R 4.0.5) |
|car          |3.0-11     |CRAN (R 4.0.5) |
|carData      |3.0-4      |CRAN (R 4.0.3) |
|cellranger   |1.1.0      |CRAN (R 4.0.3) |
|cli          |3.0.1      |CRAN (R 4.0.5) |
|coda         |0.19-4     |CRAN (R 4.0.3) |
|colorspace   |2.0-2      |CRAN (R 4.0.5) |
|corpcor      |1.6.9      |CRAN (R 4.0.3) |
|crayon       |1.4.2      |CRAN (R 4.0.5) |
|cubature     |2.0.4.2    |CRAN (R 4.0.5) |
|curl         |4.3.2      |CRAN (R 4.0.5) |
|data.table   |1.14.0     |CRAN (R 4.0.4) |
|DBI          |1.1.1      |CRAN (R 4.0.3) |
|dbplyr       |2.1.1      |CRAN (R 4.0.5) |
|desc         |1.3.0      |CRAN (R 4.0.5) |
|devtools     |2.4.2      |CRAN (R 4.0.5) |
|digest       |0.6.27     |CRAN (R 4.0.3) |
|dplyr        |1.0.7      |CRAN (R 4.0.5) |
|ellipsis     |0.3.2      |CRAN (R 4.0.5) |
|evaluate     |0.14       |CRAN (R 4.0.3) |
|extrafont    |0.17       |CRAN (R 4.0.3) |
|extrafontdb  |1.0        |CRAN (R 4.0.3) |
|fansi        |0.5.0      |CRAN (R 4.0.5) |
|faraway      |1.0.7      |CRAN (R 4.0.3) |
|fastmap      |1.1.0      |CRAN (R 4.0.3) |
|forcats      |0.5.1      |CRAN (R 4.0.3) |
|foreign      |0.8-81     |CRAN (R 4.0.4) |
|fs           |1.5.0      |CRAN (R 4.0.3) |
|generics     |0.1.1      |CRAN (R 4.0.5) |
|ggplot2      |3.3.5      |CRAN (R 4.0.5) |
|glue         |1.4.2      |CRAN (R 4.0.3) |
|gtable       |0.3.0      |CRAN (R 4.0.3) |
|haven        |2.4.3      |CRAN (R 4.0.5) |
|hms          |1.1.0      |CRAN (R 4.0.5) |
|htmltools    |0.5.2      |CRAN (R 4.0.5) |
|htmlwidgets  |1.5.4      |CRAN (R 4.0.5) |
|httr         |1.4.2      |CRAN (R 4.0.3) |
|investr      |1.4.0      |CRAN (R 4.0.3) |
|jpeg         |0.1-9      |CRAN (R 4.0.5) |
|jquerylib    |0.1.4      |CRAN (R 4.0.5) |
|jsonlite     |1.7.2      |CRAN (R 4.0.3) |
|kableExtra   |1.3.4      |CRAN (R 4.0.4) |
|knitr        |1.34       |CRAN (R 4.0.5) |
|lattice      |0.20-44    |CRAN (R 4.0.5) |
|latticeExtra |0.6-29     |CRAN (R 4.0.3) |
|lifecycle    |1.0.1      |CRAN (R 4.0.5) |
|lme4         |1.1-27.1   |CRAN (R 4.0.5) |
|lmerTest     |3.1-3      |CRAN (R 4.0.3) |
|lubridate    |1.7.10     |CRAN (R 4.0.5) |
|magrittr     |2.0.1      |CRAN (R 4.0.3) |
|MASS         |7.3-54     |CRAN (R 4.0.5) |
|matlib       |0.9.5      |CRAN (R 4.0.5) |
|Matrix       |1.3-4      |CRAN (R 4.0.5) |
|MCMCglmm     |2.32       |CRAN (R 4.0.5) |
|memoise      |2.0.0      |CRAN (R 4.0.3) |
|mgcv         |1.8-36     |CRAN (R 4.0.5) |
|minqa        |1.2.4      |CRAN (R 4.0.3) |
|modelr       |0.1.8      |CRAN (R 4.0.3) |
|munsell      |0.5.0      |CRAN (R 4.0.3) |
|nlme         |3.1-153    |CRAN (R 4.0.5) |
|nloptr       |1.2.2.2    |CRAN (R 4.0.3) |
|nlstools     |1.0-2      |CRAN (R 4.0.3) |
|nnet         |7.3-16     |CRAN (R 4.0.5) |
|numDeriv     |2016.8-1.1 |CRAN (R 4.0.3) |
|openxlsx     |4.2.4      |CRAN (R 4.0.5) |
|pbkrtest     |0.5.1      |CRAN (R 4.0.5) |
|pillar       |1.6.4      |CRAN (R 4.0.5) |
|pkgbuild     |1.2.0      |CRAN (R 4.0.3) |
|pkgconfig    |2.0.3      |CRAN (R 4.0.3) |
|pkgload      |1.2.2      |CRAN (R 4.0.5) |
|png          |0.1-7      |CRAN (R 4.0.3) |
|prettyunits  |1.1.1      |CRAN (R 4.0.3) |
|processx     |3.5.2      |CRAN (R 4.0.5) |
|ps           |1.6.0      |CRAN (R 4.0.5) |
|pscl         |1.5.5      |CRAN (R 4.0.3) |
|purrr        |0.3.4      |CRAN (R 4.0.3) |
|R6           |2.5.1      |CRAN (R 4.0.5) |
|RColorBrewer |1.1-2      |CRAN (R 4.0.3) |
|Rcpp         |1.0.7      |CRAN (R 4.0.5) |
|readr        |2.0.1      |CRAN (R 4.0.5) |
|readxl       |1.3.1      |CRAN (R 4.0.3) |
|remotes      |2.4.0      |CRAN (R 4.0.5) |
|reprex       |2.0.1      |CRAN (R 4.0.5) |
|rgl          |0.107.14   |CRAN (R 4.0.5) |
|rio          |0.5.27     |CRAN (R 4.0.5) |
|rlang        |0.4.11     |CRAN (R 4.0.5) |
|RLRsim       |3.1-6      |CRAN (R 4.0.4) |
|rmarkdown    |2.11       |CRAN (R 4.0.5) |
|rprojroot    |2.0.2      |CRAN (R 4.0.3) |
|rstudioapi   |0.13       |CRAN (R 4.0.3) |
|Rttf2pt1     |1.3.9      |CRAN (R 4.0.5) |
|rvest        |1.0.1      |CRAN (R 4.0.5) |
|sass         |0.4.0      |CRAN (R 4.0.5) |
|scales       |1.1.1      |CRAN (R 4.0.3) |
|sessioninfo  |1.1.1      |CRAN (R 4.0.3) |
|stringi      |1.7.4      |CRAN (R 4.0.5) |
|stringr      |1.4.0      |CRAN (R 4.0.3) |
|svglite      |2.0.0      |CRAN (R 4.0.4) |
|systemfonts  |1.0.2      |CRAN (R 4.0.5) |
|tensorA      |0.36.2     |CRAN (R 4.0.3) |
|testthat     |3.0.4      |CRAN (R 4.0.5) |
|tibble       |3.1.5      |CRAN (R 4.0.5) |
|tidyr        |1.1.3      |CRAN (R 4.0.4) |
|tidyselect   |1.1.1      |CRAN (R 4.0.5) |
|tidyverse    |1.3.1      |CRAN (R 4.0.5) |
|tzdb         |0.1.2      |CRAN (R 4.0.5) |
|usethis      |2.0.1      |CRAN (R 4.0.3) |
|utf8         |1.2.2      |CRAN (R 4.0.5) |
|vctrs        |0.3.8      |CRAN (R 4.0.5) |
|viridisLite  |0.4.0      |CRAN (R 4.0.5) |
|webshot      |0.5.2      |CRAN (R 4.0.5) |
|withr        |2.4.2      |CRAN (R 4.0.5) |
|xfun         |0.29       |CRAN (R 4.0.5) |
|xml2         |1.3.2      |CRAN (R 4.0.3) |
|xtable       |1.8-4      |CRAN (R 4.0.3) |
|yaml         |2.2.1      |CRAN (R 4.0.3) |
|zip          |2.2.0      |CRAN (R 4.0.5) |

<br>


```
#> - Session info ---------------------------------------------------------------
#>  setting  value                       
#>  version  R version 4.0.4 (2021-02-15)
#>  os       Windows 10 x64              
#>  system   x86_64, mingw32             
#>  ui       RTerm                       
#>  language (EN)                        
#>  collate  English_United States.1252  
#>  ctype    English_United States.1252  
#>  tz       America/Chicago             
#>  date     2022-01-07                  
#> 
#> - Packages -------------------------------------------------------------------
#>  package     * version date       lib source        
#>  assertthat    0.2.1   2019-03-21 [2] CRAN (R 4.0.3)
#>  backports     1.2.1   2020-12-09 [2] CRAN (R 4.0.3)
#>  bookdown      0.24    2021-09-02 [2] CRAN (R 4.0.5)
#>  broom         0.7.9   2021-07-27 [2] CRAN (R 4.0.5)
#>  bslib         0.3.1   2021-10-06 [2] CRAN (R 4.0.5)
#>  cachem        1.0.6   2021-08-19 [2] CRAN (R 4.0.5)
#>  callr         3.7.0   2021-04-20 [2] CRAN (R 4.0.5)
#>  cellranger    1.1.0   2016-07-27 [2] CRAN (R 4.0.3)
#>  cli           3.0.1   2021-07-17 [2] CRAN (R 4.0.5)
#>  colorspace    2.0-2   2021-06-24 [2] CRAN (R 4.0.5)
#>  crayon        1.4.2   2021-10-29 [2] CRAN (R 4.0.5)
#>  DBI           1.1.1   2021-01-15 [2] CRAN (R 4.0.3)
#>  dbplyr        2.1.1   2021-04-06 [2] CRAN (R 4.0.5)
#>  desc          1.3.0   2021-03-05 [2] CRAN (R 4.0.5)
#>  devtools      2.4.2   2021-06-07 [2] CRAN (R 4.0.5)
#>  digest        0.6.27  2020-10-24 [2] CRAN (R 4.0.3)
#>  dplyr       * 1.0.7   2021-06-18 [2] CRAN (R 4.0.5)
#>  ellipsis      0.3.2   2021-04-29 [2] CRAN (R 4.0.5)
#>  evaluate      0.14    2019-05-28 [2] CRAN (R 4.0.3)
#>  fansi         0.5.0   2021-05-25 [2] CRAN (R 4.0.5)
#>  fastmap       1.1.0   2021-01-25 [2] CRAN (R 4.0.3)
#>  forcats     * 0.5.1   2021-01-27 [2] CRAN (R 4.0.3)
#>  fs            1.5.0   2020-07-31 [2] CRAN (R 4.0.3)
#>  generics      0.1.1   2021-10-25 [2] CRAN (R 4.0.5)
#>  ggplot2     * 3.3.5   2021-06-25 [2] CRAN (R 4.0.5)
#>  glue          1.4.2   2020-08-27 [2] CRAN (R 4.0.3)
#>  gtable        0.3.0   2019-03-25 [2] CRAN (R 4.0.3)
#>  haven         2.4.3   2021-08-04 [2] CRAN (R 4.0.5)
#>  hms           1.1.0   2021-05-17 [2] CRAN (R 4.0.5)
#>  htmltools     0.5.2   2021-08-25 [2] CRAN (R 4.0.5)
#>  httr          1.4.2   2020-07-20 [2] CRAN (R 4.0.3)
#>  jpeg        * 0.1-9   2021-07-24 [2] CRAN (R 4.0.5)
#>  jquerylib     0.1.4   2021-04-26 [2] CRAN (R 4.0.5)
#>  jsonlite      1.7.2   2020-12-09 [2] CRAN (R 4.0.3)
#>  knitr         1.34    2021-09-09 [2] CRAN (R 4.0.5)
#>  lifecycle     1.0.1   2021-09-24 [2] CRAN (R 4.0.5)
#>  lubridate     1.7.10  2021-02-26 [2] CRAN (R 4.0.5)
#>  magrittr      2.0.1   2020-11-17 [2] CRAN (R 4.0.3)
#>  memoise       2.0.0   2021-01-26 [2] CRAN (R 4.0.3)
#>  modelr        0.1.8   2020-05-19 [2] CRAN (R 4.0.3)
#>  munsell       0.5.0   2018-06-12 [2] CRAN (R 4.0.3)
#>  pillar        1.6.4   2021-10-18 [2] CRAN (R 4.0.5)
#>  pkgbuild      1.2.0   2020-12-15 [2] CRAN (R 4.0.3)
#>  pkgconfig     2.0.3   2019-09-22 [2] CRAN (R 4.0.3)
#>  pkgload       1.2.2   2021-09-11 [2] CRAN (R 4.0.5)
#>  prettyunits   1.1.1   2020-01-24 [2] CRAN (R 4.0.3)
#>  processx      3.5.2   2021-04-30 [2] CRAN (R 4.0.5)
#>  ps            1.6.0   2021-02-28 [2] CRAN (R 4.0.5)
#>  purrr       * 0.3.4   2020-04-17 [2] CRAN (R 4.0.3)
#>  R6            2.5.1   2021-08-19 [2] CRAN (R 4.0.5)
#>  Rcpp          1.0.7   2021-07-07 [2] CRAN (R 4.0.5)
#>  readr       * 2.0.1   2021-08-10 [2] CRAN (R 4.0.5)
#>  readxl        1.3.1   2019-03-13 [2] CRAN (R 4.0.3)
#>  remotes       2.4.0   2021-06-02 [2] CRAN (R 4.0.5)
#>  reprex        2.0.1   2021-08-05 [2] CRAN (R 4.0.5)
#>  rlang         0.4.11  2021-04-30 [2] CRAN (R 4.0.5)
#>  rmarkdown     2.11    2021-09-14 [2] CRAN (R 4.0.5)
#>  rprojroot     2.0.2   2020-11-15 [2] CRAN (R 4.0.3)
#>  rstudioapi    0.13    2020-11-12 [2] CRAN (R 4.0.3)
#>  rvest         1.0.1   2021-07-26 [2] CRAN (R 4.0.5)
#>  sass          0.4.0   2021-05-12 [2] CRAN (R 4.0.5)
#>  scales      * 1.1.1   2020-05-11 [2] CRAN (R 4.0.3)
#>  sessioninfo   1.1.1   2018-11-05 [2] CRAN (R 4.0.3)
#>  stringi       1.7.4   2021-08-25 [2] CRAN (R 4.0.5)
#>  stringr     * 1.4.0   2019-02-10 [2] CRAN (R 4.0.3)
#>  testthat      3.0.4   2021-07-01 [2] CRAN (R 4.0.5)
#>  tibble      * 3.1.5   2021-09-30 [2] CRAN (R 4.0.5)
#>  tidyr       * 1.1.3   2021-03-03 [2] CRAN (R 4.0.4)
#>  tidyselect    1.1.1   2021-04-30 [2] CRAN (R 4.0.5)
#>  tidyverse   * 1.3.1   2021-04-15 [2] CRAN (R 4.0.5)
#>  tzdb          0.1.2   2021-07-20 [2] CRAN (R 4.0.5)
#>  usethis       2.0.1   2021-02-10 [2] CRAN (R 4.0.3)
#>  utf8          1.2.2   2021-07-24 [2] CRAN (R 4.0.5)
#>  vctrs         0.3.8   2021-04-29 [2] CRAN (R 4.0.5)
#>  withr         2.4.2   2021-04-18 [2] CRAN (R 4.0.5)
#>  xfun          0.29    2021-12-14 [1] CRAN (R 4.0.5)
#>  xml2          1.3.2   2020-04-23 [2] CRAN (R 4.0.3)
#>  yaml          2.2.1   2020-02-01 [2] CRAN (R 4.0.3)
#> 
#> [1] C:/Users/tn9k4/OneDrive - University of Missouri/Documents/R/win-library/4.0
#> [2] C:/Program Files/R/R-4.0.4/library
```
