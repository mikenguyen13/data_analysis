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

## Assumptions

Assumptions to identify treatment effect in non-randomized studies:

1.  SUTVA
2.  Conditonal Ignorability Asumtpion
3.  Overlap (Positivity) Asumption

### **Stable Unit Treatment Value Assumption**

First, we assume the **Stable Unit Treatment Value Assumption (SUTVA)** holds. SUTVA consists of two key components:

1.  **the treatment levels of** $Z$ (1 and 0) adequately represent all versions of the treatment, often referred to as the **consistency assumption** in the epidemiology literature
2.  a subject's outcomes are not affected by other subjects' exposures. This assumption ensures that potential outcomes are well-defined and independent of external influences, allowing for a **causal interpretation** of treatment effects.

SUTVA is fundamental in **Rubin's Causal Model (RCM)** and provides the foundation for defining potential outcomes. If violated, causal inference methods may lead to **biased estimators** and **incorrect standard errors**. Below, we formally define SUTVA, explore its implications, and discuss methods to handle violations.

------------------------------------------------------------------------

#### **Definition and Mathematical Formulation**

Let $Y_i(Z)$ be the **potential outcome** for unit $i$ under treatment $Z$, where $Z \in \{0,1\}$ represents a binary treatment assignment. SUTVA states that:

$$
Y_i(Z) = Y_i(Z, \mathbf{Z}_{-i})
$$

where $\mathbf{Z}_{-i}$ denotes the treatment assignments of all other units except $i$. If SUTVA holds, then:

$$
Y_i(Z) = Y_i(Z, \mathbf{Z}_{-i}) \quad \forall \mathbf{Z}_{-i}.
$$

This equation ensures that unit $i$'s outcome depends **only** on its own treatment status $Z$ and not on the treatment assignments of other units.

Under SUTVA, the **Average Treatment Effect (ATE)** is well-defined as:

$$
\text{ATE} = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)].
$$

However, if SUTVA is violated due to **interference or treatment inconsistency**, then the standard **potential outcomes framework** must be adjusted to account for these effects.

------------------------------------------------------------------------

#### **No Interference Assumption and Its Mathematical Implications**

##### **Definition of No Interference**

The **no interference** component of SUTVA assumes that one unit's treatment assignment does not affect another unit's outcome. In many real-world scenarios, this assumption is violated due to **spillover effects**, such as:

-   **Epidemiology**: In vaccine studies, an individual's health status may be affected by the vaccination status of their social network.
-   **Marketing Experiments**: In online advertising, one consumer's exposure to an ad campaign may influence their peers' purchasing decisions.

Formally, if interference exists, then unit $i$'s outcome depends on a **neighborhood function** $\mathcal{N}(i)$, where:

$$
Y_i(Z, \mathbf{Z}_{\mathcal{N}(i)}).
$$

If $\mathcal{N}(i) \neq \emptyset$, interference exists, and SUTVA is violated. In such cases, we must redefine treatment effects by considering **direct and indirect effects** using methodologies such as **spatial models** or **network-based causal inference**.

##### **Special Cases of Interference**

-   **Complete Interference**: Every unit's outcome is affected by all other units' treatment assignments.
-   **Partial Interference**: Interference occurs **within** subgroups but not **between** them (e.g., classrooms in an educational experiment).
-   **Network Interference**: Treatment effects propagate through a social or spatial network, requiring models like **graph-based causal inference**.

------------------------------------------------------------------------

#### **No Hidden Variations in Treatment**

The second component of SUTVA ensures that the treatment effect is **uniquely defined**, meaning there are no hidden variations in how the treatment is administered. That is, if multiple **versions** of the treatment exist (e.g., different dosages of a drug), the causal effect may **not be well-defined**.

Mathematically, if there exist multiple versions $v$ of the treatment $Z$, then the potential outcome should be indexed accordingly:

$$
Y_i(Z, v).
$$

If different versions produce different outcomes, then:

$$
Y_i(Z, v_1) \neq Y_i(Z, v_2).
$$

This violates SUTVA and requires **instrumental variables (IV)** or **latent variable models** to adjust for treatment heterogeneity.

------------------------------------------------------------------------

#### **Implications of Violating SUTVA**

If SUTVA is violated, causal inference suffers from **bias, incorrect standard errors, and ambiguous estimands**. Key consequences include:

-   **Bias in Estimators**: If interference is ignored, treatment effects are misestimated.
-   **Incorrect Standard Errors**: Standard errors may be **underestimated** (if spillovers are ignored) or **overestimated** (if hidden treatment variations exist).
-   **Ill-Defined Causal Effects**: If multiple treatment versions exist, it becomes unclear **which causal effect** is being estimated.

In such cases, alternative estimands such as **network-adjusted treatment effects** or **spatial spillover models** are needed.

------------------------------------------------------------------------

#### **Strategies to Address SUTVA Violations**

To mitigate violations of SUTVA, researchers can adopt the following techniques:

1.  **Randomized Saturation Designs**: Introduce varying treatment intensities across clusters to measure spillover effects.
2.  **Network-Based Causal Models**: Use graph theory to model interference.
3.  **Instrumental Variables (IV)**: If multiple versions of treatment exist, use an IV that isolates a single version.
4.  **Stratified Analysis**: If treatment versions differ significantly, analyze each subgroup separately.
5.  **Difference-in-Differences (DiD) with Spatial Controls**: For geographic spillovers, include spatially lagged treatment indicators.

Each of these approaches ensures that causal inferences remain valid despite potential SUTVA violations.

------------------------------------------------------------------------

### **Conditional Ignorability Assumption**

Next, we must assume that **treatment assignment is independent of the potential outcomes conditional on the observed covariates**. This assumption has several equivalent names in the causal inference literature, including **"conditional ignorability," "conditional exchangeability," "no unobserved confounding," and "no omitted variables."** In the language of **causal diagrams**, this assumption ensures that all **backdoor paths** between treatment and outcome are blocked by **observed covariates**.

Formally, we assume that treatment assignment $Z$ is independent of the potential outcomes $Y(Z)$ given a set of **observed covariates** $X$:

$$
Y(1), Y(0) \perp\!\!\!\perp Z \mid X.
$$

This means that **after conditioning on** $X$, the probability of receiving treatment is unrelated to the potential outcomes, ensuring that comparisons between treated and untreated units are **unbiased**. Below, we explore the mathematical implications, violations, and strategies for addressing violations of this assumption.

------------------------------------------------------------------------

#### **Formal Definition and Notation**

In causal inference, **treatment assignment is said to be ignorable** if, conditional on observed covariates $X$, the treatment indicator $Z$ is independent of the potential outcomes:

$$
P(Y(1), Y(0) \mid Z, X) = P(Y(1), Y(0) \mid X).
$$

Equivalently, in terms of conditional probability:

$$
P(Z = 1 \mid Y(1), Y(0), X) = P(Z = 1 \mid X).
$$

This ensures that treatment assignment is **as good as random** once we control for $X$, meaning that the probability of receiving treatment does not depend on unmeasured confounders.

A direct consequence is that we can estimate the **Average Treatment Effect (ATE)** using observational data:

$$
\mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[\mathbb{E}[Y \mid Z=1, X] - \mathbb{E}[Y \mid Z=0, X]].
$$

If **ignorability holds**, standard regression models, matching, or weighting techniques (e.g., propensity score weighting) can provide **unbiased causal estimates**.

------------------------------------------------------------------------

#### **The Role of Causal Diagrams and Backdoor Paths**

In **causal diagrams** (DAGs), confounding arises when a **backdoor path** exists between treatment $Z$ and outcome $Y$. A **backdoor path** is any **non-causal** path that creates **spurious associations** between $Z$ and $Y$. The **conditional ignorability assumption** requires that all such paths be **blocked** by conditioning on a sufficient set of covariates $X$.

##### **Identifying Backdoor Paths**

Consider a simple causal diagram:

X → Z → Y X → Y

Here, $X$ is a **common cause** of both $Z$ and $Y$, creating a **backdoor path** $Z \leftarrow X \rightarrow Y$. If we fail to **control for** $X$, the estimated effect of $Z$ on $Y$ will be **biased**. However, if we **condition on** $X$, we block the backdoor path and obtain an **unbiased estimate** of the treatment effect.

##### **Sufficient Covariate Adjustment**

To satisfy the conditional ignorability assumption, researchers must identify a **sufficient set of confounders** to block all backdoor paths. This is often done using **domain knowledge** and **causal structure learning algorithms**.

-   **Minimal Sufficient Adjustment Set**: The smallest set of covariates $X$ that, when conditioned upon, satisfies ignorability.
-   **Propensity Score Methods**: Instead of adjusting directly for $X$, one can estimate the probability of treatment $P(Z=1 \mid X)$ and use **inverse probability weighting (IPW)** or **matching**.

------------------------------------------------------------------------

#### **Violations of the Ignorability Assumption**

If ignorability does **not** hold, treatment assignment depends on **unobserved confounders**, introducing **omitted variable bias**. Mathematically, if there exists an unmeasured variable $U$ such that:

$$
Y(1), Y(0) \not\perp\!\!\!\perp Z \mid X,
$$

then estimates of the treatment effect will be **biased**.

##### **Consequences of Violations**

-   **Confounded Estimates**: The estimated treatment effect captures both the causal effect and the bias from unobserved confounders.
-   **Selection Bias**: If treatment assignment is related to factors that also influence the outcome, the sample may not be representative.
-   **Overestimation or Underestimation**: Ignoring important confounders can lead to inflated or deflated estimates of treatment effects.

##### **Example of Confounding**

Consider an **observational study on smoking and lung cancer**:

Smoking → Lung Cancer Genetics → Smoking Genetics → Lung Cancer

Here, **genetics** is an unmeasured confounder affecting both **smoking** and **lung cancer**. If we do not control for genetics, the estimated effect of smoking on lung cancer will be **biased**.

------------------------------------------------------------------------

#### **Strategies to Address Violations**

If ignorability is violated due to **unobserved confounding**, several techniques can be used to mitigate bias:

1.  **Instrumental Variables (IV)**:
    -   Use a variable $W$ that affects treatment $Z$ but has **no direct effect** on $Y$, ensuring exogeneity.
    -   Example: Randomized incentives to encourage treatment uptake.
2.  **Difference-in-Differences (DiD)**:
    -   Compare changes in outcomes before and after treatment in a treated vs. control group.
    -   Requires a **parallel trends assumption**.
3.  **Regression Discontinuity (RD)**:
    -   Exploit cutoff-based treatment assignment.
    -   Example: Scholarship eligibility at a certain GPA threshold.
4.  **Propensity Score Methods**:
    -   Estimate the probability of treatment given $X$.
    -   Use **matching, inverse probability weighting (IPW), or stratification** to balance treatment groups.
5.  **Sensitivity Analysis**:
    -   Quantify how much unobserved confounding would be needed to alter conclusions.
    -   Example: **Rosenbaum's sensitivity bounds**.

------------------------------------------------------------------------

#### **Practical Considerations**

##### **How to Select Covariates** $X$?

-   **Domain Knowledge**: Consult experts to identify potential confounders.
-   **Causal Discovery Methods**: Use **Bayesian networks or structure learning** to infer relationships.
-   **Statistical Tests**: Examine balance in pre-treatment characteristics.

##### **Trade-Offs in Covariate Selection**

-   **Too Few Covariates** → Risk of **omitted variable bias**.
-   **Too Many Covariates** → **Overfitting**, loss of **efficiency** in estimation.

------------------------------------------------------------------------

### **Overlap (Positivity) Assumption**

Next, we assume that the probability of receiving treatment is strictly **greater than zero and less than one** over the support of the observed covariates $X_i$. This is known as the **overlap assumption**, also referred to as **common support** or **positivity**. Mathematically, this is written as:

$$
0 < P(Z_i = 1 \mid X_i) < 1, \quad \forall X_i.
$$

This ensures that for every possible value of $X_i$, there is **some** probability of receiving both treatment ($Z_i = 1$) and control ($Z_i = 0$). In other words, there must be **overlap** in the covariate distributions between treated and control units.

When **overlap is limited**, the **Average Treatment Effect (ATE)** may not be identifiable, even if the **Average Treatment effect on the Treated (ATT)** remains identifiable. In extreme cases, even ATT may not be identified due to severe lack of common support. In such situations, an alternative estimand can be used, such as the **Average Treatment Effect for the Overlap Population (ATO)**, which focuses on a **marginal population** where treatment is not deterministic [@li2018balancing].

------------------------------------------------------------------------

#### **Mathematical Formulation of Overlap**

The overlap assumption states that **for every possible value of the covariates** $X_i$, there exists some probability of being in both treatment groups. This prevents situations where treatment assignment is deterministic, which would make causal inference impossible.

Mathematically, overlap requires that:

$$
0 < P(Z_i = 1 \mid X_i) < 1 \quad \forall X_i.
$$

This means that:

1.  **Positivity Condition**: Every unit in the population has a **nonzero probability** of receiving treatment.
2.  **No Deterministic Treatment Assignment**: If $P(Z_i = 1 \mid X_i) = 0$ or $P(Z_i = 1 \mid X_i) = 1$ for some $X_i$, then the causal effect is **not identifiable** for those values of $X_i$.

In practical terms, if some subpopulations **always receive treatment** ($P(Z_i = 1 \mid X_i) = 1$) or **never receive treatment** ($P(Z_i = 1 \mid X_i) = 0$), then there is **no counterfactual** to compare against, making it impossible to estimate causal effects for those groups.

------------------------------------------------------------------------

#### **Implications of Violating the Overlap Assumption**

When the overlap assumption is violated, the identification of causal effects becomes problematic. Some key implications include:

1.  **Limited Generalizability of ATE**: If there is **poor overlap** in covariate distributions between treated and control units, the **Average Treatment Effect (ATE)** may not be identified.
2.  **ATT May Still Be Identifiable**: If **some** overlap exists but is limited, we may still be able to estimate the **Average Treatment Effect on the Treated (ATT)**.
3.  **Extreme Cases - No ATT Identification**: If **no** overlap exists between treated and control groups, even ATT may not be identified.
4.  **Extrapolation Bias**: In extreme cases, estimation relies on extrapolating from regions where overlap is weak, leading to **biased and unstable causal estimates**.

To illustrate, consider an observational study on the effect of an **education intervention** on academic performance. If **only students from high-income families** received the intervention ($P(Z = 1 \mid X) = 1$ for high income), then we cannot compare them to **low-income students** who never received the intervention ($P(Z = 1 \mid X) = 0$). This lack of common support prevents estimation of a valid treatment effect.

------------------------------------------------------------------------

#### **Diagnosing Overlap Violations in Practice**

Before estimating causal effects, it is crucial to assess whether the overlap assumption holds. Some common diagnostic tools include:

##### **Propensity Score Distribution**

A key approach is to **estimate the propensity score** $e(X) = P(Z = 1 \mid X)$ and visualize its distribution across treated and control units. A lack of overlap in propensity scores suggests that **some regions of** $X$ lack common support.

-   **Well-Mixed Propensity Score Distributions** → Good overlap, strong causal identification.
-   **Separated Propensity Score Distributions** → Poor overlap, potential issues with causal estimation.

##### **Standardized Mean Differences (SMD)**

Standardized mean differences compare the covariate distributions between treated and control groups. If large imbalances exist, overlap may be insufficient.

##### **Kernel Density Plots**

Plotting the kernel density of **propensity scores** can reveal whether both groups have sufficient representation across the distribution.

------------------------------------------------------------------------

#### **Strategies to Address Overlap Violations**

When overlap is weak, several strategies can be employed:

1.  **Trimming Non-Overlapping Units**
    -   Exclude observations with extreme propensity scores (e.g., those where $P(Z = 1 \mid X) \approx 0$ or $P(Z = 1 \mid X) \approx 1$).
    -   Improves robustness but reduces sample size.
2.  **Reweighting Approaches**
    -   Use **overlap weights** to emphasize the population where treatment assignment is not deterministic.
    -   The **Average Treatment Effect for the Overlap Population (ATO)** [29] estimates the effect for units where $P(Z = 1 \mid X)$ is close to 0.5.
3.  **Matching on the Propensity Score**
    -   Remove units without suitable matches from the opposite treatment group.
    -   Improves balance at the cost of excluding some observations.
4.  **Covariate Balancing Techniques**
    -   Use **entropy balancing** or **inverse probability weighting (IPW)** to adjust for limited overlap.
    -   Ensures that covariate distributions are similar across groups.
5.  **Sensitivity Analysis**
    -   Quantify how overlap violations affect causal conclusions.
    -   Example: **Rosenbaum's bounds for unmeasured confounding**.

------------------------------------------------------------------------

#### **The Average Treatment Effect for the Overlap Population**

When overlap is limited, one alternative is to estimate the **Average Treatment Effect for the Overlap Population (ATO)**. Unlike ATE, which targets the **entire population**, and ATT, which targets the **treated population**, ATO focuses on the **subset of units where treatment is not deterministic**.

Mathematically, ATO is estimated using **overlap weights**:

$$
W_i = P(Z_i = 1 \mid X_i) (1 - P(Z_i = 1 \mid X_i)).
$$

This **downweights extreme propensity scores** and ensures that inference is focused on a population where treatment was plausibly **assignable in both directions**. ATO is particularly useful when **generalizability is a concern** or when **extrapolation is unreliable**.

------------------------------------------------------------------------

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
