# Quasi-experimental {#sec-quasi-experimental}

In most cases, it means that you have pre- and post-intervention data.

Great resources for causal inference include [Causal Inference Mixtape](https://mixtape.scunning.com/introduction.html) and [Recent Advances in Micro](https://christinecai.github.io/PublicGoods/applied_micro_methods.pdf), especially if you like to read about the history of causal inference as a field as well (codes for Stata, R, and Python).

Libraries in R:

-   [Econometrics](https://cran.r-project.org/web/views/Econometrics.html)

-   [Causal Inference](https://cran.r-project.org/web/views/CausalInference.html)

**Identification strategy** for any quasi-experiment (No ways to prove or formal statistical test, but you can provide plausible argument and evidence)

1.  Where the exogenous variation comes from (by argument and institutional knowledge)
2.  Exclusion restriction: Evidence that the variation in the exogenous shock and the outcome is due to no other factors
    1.  The stable unit treatment value assumption (SUTVA) states that the treatment of unit $i$ affect only the outcome of unit $i$ (i.e., no spillover to the control groups)

All quasi-experimental methods involve a tradeoff between power and support for the exogeneity assumption (i.e., discard variation in the data that is not exogenous).

Consequently, we don't usually look at $R^2$ [@ebbes2011sense]. And it can even be misleading to use $R^2$ as the basis for model comparison.

Clustering should be based on the design, not the expectations of correlation [@abadie2023should]. With a **small sample**, you should use the **wild bootstrap procedure** [@cameron2008bootstrap] to correct for the downward bias (see [@cai2022implementation]for additional assumptions).

Typical robustness check: recommended by [@goldfarb2022conducting]

-   Different controls: show models with and without controls. Typically, we want to see the change in the estimate of interest. See [@altonji2005selection] for a formal assessment based on Rosenbaum bounds (i.e., changes in the estimate and threat of Omitted variables on the estimate). For specific applications in marketing, see [@manchanda2015social] [@shin2012fire]

-   Different functional forms

-   Different window of time (in longitudinal setting)

-   Different dependent variables (those that are related) or different measures of the dependent variables

-   Different control group size (matched vs. un-matched samples)

-   Placebo tests: see each placebo test for each setting below.

Showing the mechanism:

-   [Mediation] analysis

-   [Moderation] analysis

    -   Estimate the model separately (for different groups)

    -   Assess whether the three-way interaction between the source of variation (e.g., under DID, cross-sectional and time series) and group membership is significant.

External Validity:

-   Assess how representative your sample is

-   Explain the limitation of the design.

-   Use quasi-experimental results in conjunction with structural models: see [@anderson2015growth; @einav2010beyond; @chung2014bonuses]

Limitation

1.  What is your identifying assumptions or identification strategy
2.  What are threats to the validity of your assumptions?
3.  What you do to address it? And maybe how future research can do to address it.

## Natural Experiments

Reusing the same natural experiments for research, particularly when employing identical methods to determine the treatment effect in a given setting, can pose problems for hypothesis testing.

Simulations show that when $N_{\text{Outcome}} >> N_{\text{True effect}}$, more than 50% of statistically significant findings may be false positives [@heath2023reusing, p.2331].

**Solutions:**

-   Bonferroni correction

-   @romano2005stepwise and @romano2016efficient correction: recommended

-   @benjamini2001control correction

-   Alternatively, refer to the rules of thumb from Table AI [@heath2023reusing, p.2356].

When applying multiple testing corrections, we can either use (but they will give similar results anyway [@heath2023reusing, p.2335]):

1.  **Chronological Sequencing**: Outcomes are ordered by the date they were first reported, with multiple testing corrections applied in this sequence. This method progressively raises the statistical significance threshold as more outcomes are reviewed over time.

2.  **Best Foot Forward Policy**: Outcomes are ordered from most to least likely to be rejected based on experimental data. Used primarily in clinical trials, this approach gives priority to intended treatment effects, which are subjected to less stringent statistical requirements. New outcomes are added to the sequence as they are linked to the primary treatment effect.


```r
# Romano-Wolf correction
library(fixest)
library(wildrwolf)

head(iris)
#>   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
#> 1          5.1         3.5          1.4         0.2  setosa
#> 2          4.9         3.0          1.4         0.2  setosa
#> 3          4.7         3.2          1.3         0.2  setosa
#> 4          4.6         3.1          1.5         0.2  setosa
#> 5          5.0         3.6          1.4         0.2  setosa
#> 6          5.4         3.9          1.7         0.4  setosa

fit1 <- feols(Sepal.Width ~ Sepal.Length , data = iris)
fit2 <- feols(Petal.Length ~ Sepal.Length, data = iris)
fit3 <- feols(Petal.Width ~ Sepal.Length, data = iris)

res <- rwolf(
  models = list(fit1, fit2, fit3), 
  param = "Sepal.Length",  
  B = 500
)
#>   |                                                                              |                                                                      |   0%  |                                                                              |=======================                                               |  33%  |                                                                              |===============================================                       |  67%  |                                                                              |======================================================================| 100%

res
#>   model   Estimate Std. Error   t value     Pr(>|t|) RW Pr(>|t|)
#> 1     1 -0.0618848 0.04296699 -1.440287    0.1518983 0.115768463
#> 2     2   1.858433 0.08585565  21.64602 1.038667e-47 0.001996008
#> 3     3  0.7529176 0.04353017  17.29645 2.325498e-37 0.001996008
```

For all other tests, one can use `multtest::mt.rawp2adjp` which includes:

-   Bonferroni
-   @holm1979simple
-   @vsidak1967rectangular
-   @hochberg1988sharper
-   @benjamini1995controlling
-   @benjamini2001control
-   Adaptive @benjamini2000adaptive
-   Two-stage @benjamini2006adaptive

Permutation adjusted p-values for simple multiple testing procedures


```r
# BiocManager::install("multtest")
library(multtest)

procs <-
    c("Bonferroni",
      "Holm",
      "Hochberg",
      "SidakSS",
      "SidakSD",
      "BH",
      "BY",
      "ABH",
      "TSBH")

mt.rawp2adjp(
    # p-values
    runif(10),
    procs) |> causalverse::nice_tab()
#>    adjp.rawp adjp.Bonferroni adjp.Holm adjp.Hochberg adjp.SidakSS adjp.SidakSD
#> 1       0.12               1         1          0.75         0.72         0.72
#> 2       0.22               1         1          0.75         0.92         0.89
#> 3       0.24               1         1          0.75         0.94         0.89
#> 4       0.29               1         1          0.75         0.97         0.91
#> 5       0.36               1         1          0.75         0.99         0.93
#> 6       0.38               1         1          0.75         0.99         0.93
#> 7       0.44               1         1          0.75         1.00         0.93
#> 8       0.59               1         1          0.75         1.00         0.93
#> 9       0.65               1         1          0.75         1.00         0.93
#> 10      0.75               1         1          0.75         1.00         0.93
#>    adjp.BH adjp.BY adjp.ABH adjp.TSBH_0.05 index h0.ABH h0.TSBH
#> 1     0.63       1     0.63           0.63     2     10      10
#> 2     0.63       1     0.63           0.63     6     10      10
#> 3     0.63       1     0.63           0.63     8     10      10
#> 4     0.63       1     0.63           0.63     3     10      10
#> 5     0.63       1     0.63           0.63    10     10      10
#> 6     0.63       1     0.63           0.63     1     10      10
#> 7     0.63       1     0.63           0.63     7     10      10
#> 8     0.72       1     0.72           0.72     9     10      10
#> 9     0.72       1     0.72           0.72     5     10      10
#> 10    0.75       1     0.75           0.75     4     10      10
```
