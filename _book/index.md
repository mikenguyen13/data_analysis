---
title: "A Guide on Data Analysis"
author: "Mike Nguyen"
date: "2021-09-09"
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
link-citations: yes
description: This is a guide on how to conduct data analysis
github-repo: mikenguyen13/data_analysis
favicon: "favicon.ico"
cover-image: "images/cover.jpg"
apple-touch-icon: "logo.png"
apple-touch-icon-size: 120
site: bookdown::bookdown_site
---

# Preface {#preface .unnumbered}



<div style = "text-align: center">

![](images/cover.jpg)

</div>

<img src="logo.png" width="25%" style="display: block; margin: auto;" />

# Introduction

This guide is an attempt to streamline and demystify the data analysis process.

By no means this is an ultimate guide, or I am a great source of knowledge, or I claim myself to be a statistician/ econometrician, but I am a strong proponent of learning by teaching, and doing. Hence, this is more like a learning experience for both you and me.

<br>

Since the beginning of the century, we have been bombarded with amazing advancements and inventions, especially in the field of statistics, information technology, computer science, or a new emerging filed - data science. However, I believe the downside of this introduction is that we use **big** and **trendy** words too often (i.e., big data, machine learning, deep learning).

It's all fun and exciting when I learned these new tools. But I have to admit that I hardly retain any of these new ideas. However, writing down from the beginning till the end of a data analysis process is the solution that I came up with. Accordingly, let's dive right in.


```
## Warning: package 'jpeg' was built under R version 4.0.5
```

<img src="images/meme.jpg" width="600" style="display: block; margin: auto;" />

<br>

**Some general recommendation**:

-   The more you practice/habituate/condition, more line of codes that you write, more function that you memorize, I think the more you will like this journey.

-   Readers can follow this book several ways:

    -   If you are interested in particular methods/tools, you can jump to that section by clicking the section name.
    -   If you want to follow a traditional path of data analysis, read the [Linear Regression] section.
    -   If you want to create your experiment and test your hypothesis, read the [Analysis of Variance (ANOVA)] section.

-   Alternatively, if you rather see the application of models, and disregard any theory or underlying mechanisms, you can skip to summary and application portion of each section.

-   If you don't understand a part, search the title of that part of that part on Google, and read more into that subject. This is just a general guide.

-   If you want to customize your code beyond the ones provided in this book, run in the console `help(code)` or `?code`. For example, I want more information on `hist` function, I'll type in the console `?hist` or `help(hist)`.\

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

**Setup Working Environment**


```r
if (!require("pacman"))
    install.packages("pacman")
if (!require("devtools"))
    install.packages("devtools")
library("pacman")
library("devtools")
```


