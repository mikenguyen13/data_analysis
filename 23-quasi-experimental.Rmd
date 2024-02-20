# Quasi-experimental

In most cases, it means that you have pre- and post-intervention data.

A great resource for causal inference is [Causal Inference Mixtape](https://mixtape.scunning.com/introduction.html), especially if you like to read about the history of causal inference as a field as well (codes for Stata, R, and Python).

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
