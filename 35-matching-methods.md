# Matching Methods {#sec-matching-methods}

Matching is a strategy that aims to eliminate, or at least minimize, potential sources of bias by constructing treatment and comparison groups with similar observed characteristics. By doing so, any observed differences in outcomes between these groups can be attributed more confidently to the treatment itself rather than to other factors. In observational research, matching is frequently combined with [Difference-in-Differences] (DiD) techniques to address issues of selection bias, particularly when multiple pre-treatment outcomes are available.

> Matching is defined as "any method that aims to equate (or "balance") the distribution of covariates in the treated and control groups." [@stuart2010matching, pp. 1]

Matching is particularly useful when:

-   Outcomes are **not yet observed**, such as in follow-up studies, and you want to construct balanced treatment/control groups.
-   Outcomes **are available**, but you wish to **reduce model dependence** and **improve robustness**.

> Conceptually, matching can also be viewed through the lens of **missing data**, since we never observe both potential outcomes $(Y_i^T, Y_i^C)$ for any unit. Hence, this topic closely relates to [Imputation (Missing Data)].

## Introduction and Motivation

### Why Match?

In many observational studies, researchers do not have the luxury of randomization. Subjects (people, firms, schools, etc.) typically select or are selected into treatment based on certain observed and/or unobserved characteristics. This can introduce systematic differences (selection bias) that confound causal inference. Matching attempts to approximate a randomized experiment by "balancing" these observed characteristics between treated and non-treated (control) units.

-   **Goal:** Reduce model dependence and clarify causal effects by ensuring that treated and control subjects have sufficiently comparable covariates.
-   **Challenge:** Even if matching achieves balance in observed covariates, any **unobserved** confounders remain a threat to identification (i.e., Matching is only a selection observables identification strategy). Matching does not magically fix bias from unobserved variables.

To understand why causal inference is difficult in observational studies, consider:

$$ 
\begin{aligned} E(Y_i^T | T) - E(Y_i^C | C) &= E(Y_i^T - Y_i^C | T) + \underbrace{[E(Y_i^C | T) - E(Y_i^C | C)]}_{\text{Selection Bias}} \\ \end{aligned} $$

-   The term $E(Y_i^T - Y_i^C | T)$ is the **causal effect** (specifically the ATT).
-   The term $E(Y_i^C | T) - E(Y_i^C | C)$ reflects **selection bias** due to systematic differences in the untreated potential outcome across treated and control groups.

Random assignment ensures:

$$ E(Y_i^C | T) = E(Y_i^C | C) $$

which eliminates selection bias. In observational data, however, this equality rarely holds.

Matching aims to mimic randomization by **conditioning on covariates** $X$:

$$ E(Y_i^C | X, T) = E(Y_i^C | X, C) $$

For example, propensity score matching achieves this balance by conditioning on the **propensity score** $P(X)$:

$$ E(Y_i^C | P(X), T) = E(Y_i^C | P(X), C) $$

(See [Propensity Scores](#sec-propensity-scores) for further discussion.)

The [Average Treatment Effect] (ATE) under matching is typically estimated as:

$$ \frac{1}{N_T} \sum_{i=1}^{N_T} \left(Y_i^T - \frac{1}{N_{C_i}} \sum_{j \in \mathcal{C}_i} Y_j^C\right) $$

where $\mathcal{C}_i$ denotes the matched controls for treated unit $i$.

Standard Errors in Matching

-   Matching does not have a closed-form standard error for the ATE or ATT.
-   Therefore, we rely on bootstrapping to estimate uncertainty.

> **Note**: Matching tends to yield larger standard errors than [OLS][Ordinary Least Squares] because it reduces the effective sample size by discarding unmatched observations.

------------------------------------------------------------------------

### Matching as "Pruning"

Matching can be thought of as "**pruning**" (a **preprocessing** step) [@king2017balance]. The goal is to **prune** unmatched or poorly matched units before conducting analysis, reducing model dependence.

Without Matching:

-   Imbalanced data → Model dependence → Researcher discretion → Biased estimates

With Matching:

-   Balanced data → Reduces discretion → More credible causal inference

+-----------------------+------------------------+----------------------+
| Balance of Covariates | Complete Randomization | Fully Exact Matching |
+=======================+========================+======================+
| **Observed**          | On average             | Exact                |
+-----------------------+------------------------+----------------------+
| **Unobserved**        | On average             | On average           |
+-----------------------+------------------------+----------------------+

: Degree of Balance Across Designs

Fully blocked or exactly matched designs are not just *as good as* a complete randomization on these criteria, they dominate it. Imbalance is reduced because covariates are forced into alignment rather than left to chance; model dependence falls because the analyst is no longer using regression to paper over baseline differences; efficiency and statistical power rise because residual variation is no longer eaten by group differences; bias from observable heterogeneity is eliminated by construction rather than averaged away; the resulting estimates are more robust to specification choices; and the up-front design cost typically pays itself back in cleaner inference and a smaller required sample.

------------------------------------------------------------------------

### Matching with DiD

Matching can be fruitfully combined with DiD when multiple pre-treatment periods are available. Such designs can help correct for selection bias under certain assumptions:

-   When selection bias is symmetric around the treatment date, standard DID (implemented symmetrically around the treatment date) remains consistent [@chabe2015analysis].
-   If selection bias is asymmetric, simulations by @chabe2015analysis show that symmetric DID still outperforms matching alone, although having more pre-treatment observations can improve matching performance.

In short, matching is not a universal solution but often provides a helpful preprocessing step before conducting DiD or other causal estimation methods [@smith2005does].

------------------------------------------------------------------------

## Key Assumptions

Matching relies on the standard set of assumptions underpinning [selection on observables](#sec-selection-on-observables), also known as the back-door criterion (see [Assumptions for Identifying Treatment Effects]). When these assumptions hold, matching can yield valid estimates of causal effects by constructing treated and control groups that are comparable on observed covariates.

1.  **Strong** [Conditional Ignorability Assumption] (Unconfoundedness)

Also known as the **no hidden bias** or **ignorability** assumption:

$$
(Y(0), Y(1)) \perp T \big| X
$$

This implies that, conditional on covariates $X$, treatment assignment is independent of the potential outcomes. In other words, there are no unobserved confounders once we adjust for $X$.

-   This assumption is not testable, but it is more plausible when all relevant confounders are observed and included in $X$.
-   It is often satisfied approximately when unobserved covariates are highly correlated with the observed ones.
-   If unobserved variables are unrelated to $X$, you can:
    -   Conduct sensitivity analysis to test the robustness of your estimates.
    -   Apply design sensitivity techniques: If unobserved confounding is suspected, methods such as [@heller2009split]'s design sensitivity approaches or bounding approaches (e.g., the `rbounds` R package) can be used to test how robust findings are to hidden bias.

> This is the cornerstone assumption of matching: without it, causal inference from observational data is generally invalid.

------------------------------------------------------------------------

2.  [Overlap (Positivity) Assumption] **(Common Support)**

$$
0 < P(T=1 \mid X) < 1 \quad \forall X
$$

This condition ensures that, for every value of the covariates $X$, there is a positive probability of receiving both treatment and control.

-   If this assumption fails, there are regions of covariate space where either treatment or control units are absent, making comparison impossible.
-   Matching enforces this assumption by discarding units outside of the region of common support.

> This pruning step is both a strength and a limitation of matching: it improves internal validity at the cost of generalizability.

------------------------------------------------------------------------

3.  [Stable Unit Treatment Value Assumption] (SUTVA)

SUTVA requires that:

-   The potential outcomes for any individual unit do not depend on the treatment assignment of other units.

That is, there are **no interference or spillover effects** between units.

-   Mathematically, $Y_i(T_i)$ depends only on $T_i$, not on $T_j$ for any $j \neq i$.
-   Violations can occur in settings like:
    -   Education (peer effects)
    -   Epidemiology (disease transmission)
    -   Marketing (network influence)

> In cases with known spillover, efforts should be made to reduce interactions or explicitly model interference.

------------------------------------------------------------------------

Summary of Assumptions for Matching

+------------------------------+-------------------------------------------------------------------------------------+----------------------------------------------+
| Assumption                   | Description                                                                         | Notation                                     |
+==============================+=====================================================================================+==============================================+
| **Conditional Ignorability** | No hidden confounding after conditioning on covariates                              | $(Y(0), Y(1)) \perp T \mid X$                |
+------------------------------+-------------------------------------------------------------------------------------+----------------------------------------------+
| **Overlap (Positivity)**     | Each unit has a non-zero probability of treatment and control assignment            | $0 < P(T=1 \mid X) < 1$                      |
+------------------------------+-------------------------------------------------------------------------------------+----------------------------------------------+
| **SUTVA**                    | No interference between units; one unit's outcome unaffected by another's treatment | $Y_i(T_i)$ unaffected by $T_j$ for $j \ne i$ |
+------------------------------+-------------------------------------------------------------------------------------+----------------------------------------------+

These three assumptions form the foundation for valid causal inference using matching methods.

------------------------------------------------------------------------

## Framework for Generalization

With the assumptions in place, the next step is to fix notation that will let us write the matching estimand precisely and reason about which average effect (ATE versus ATT) a particular procedure recovers.

Let:

-   $P_t$, $P_c$: treated and control populations
-   $N_t$, $N_c$: random samples drawn from $P_t$, $P_c$
-   $\mu_i$, $\Sigma_i$: means and covariance matrices of the $p$ covariates in group $i \in \{t, c\}$
-   $X_j$: vector of covariates for individual $j$
-   $T_j \in \{0, 1\}$: treatment indicator (1 = treated, 0 = control)
-   $Y_j$: observed outcome
-   Assume $N_t < N_c$ (i.e., more controls than treated)

The conditional treatment effect is:

$$
\tau(x) = R_1(x) - R_0(x), \quad \text{where } R_1(x) = E[Y(1) \mid X = x], \quad R_0(x) = E[Y(0) \mid X = x]
$$

If we assume **homogeneous treatment effects**, then $\tau(x) = \tau$ for all $x$ (this is a homogeneity assumption on the treatment effect, not the parallel-trends assumption from [DiD](#sec-difference-in-differences), which is a separate restriction on counterfactual *time trends*). If the homogeneity assumption is relaxed, we can still estimate an **average** effect over the distribution of $X$.

### Common Estimands

-   [Average Treatment Effect] (ATE): Average causal effect across all units.
-   [Average Treatment Effect on the Treated] (ATT): Causal effect for treated units only.

------------------------------------------------------------------------

## Steps for Matching

Implementing a matching analysis is a four-step pipeline: pick a way to measure how close two units are, choose an algorithm to pair them up, check that the resulting comparison group is actually balanced, and only then estimate the treatment effect. The three ingredients below, a propensity score, a distance metric, and a covariate set assumed to satisfy ignorability, recur across the steps that follow.

-   Propensity score: summarizes $P(T=1|X)$
-   Distance metric: measures similarity
-   Covariates: assumed to satisfy ignorability

### Step 1: Define "Closeness" (Distance Metrics)

Matching requires a distance metric to define similarity between treated and control units.

#### Variable Selection Guidelines

-   Include as many pre-treatment covariates as possible to support conditional ignorability.
-   Avoid post-treatment variables, which introduce bias.
-   Be cautious with variables (e.g., heavy drug users) highly correlated with the outcome (e.g., heavy drinkers) but not treatment (e.g., mediators).
-   If variables are uncorrelated with both treatment and outcome, the cost of inclusion is small.

#### Distance Measures

+----------------------------+----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Method                     | Formula                                            | Notes                                                                                                                                    |
+============================+====================================================+==========================================================================================================================================+
| **Exact Matching**         | $D_{ij} = 0$ if $X_i = X_j$, else $\infty$         | Only feasible in low dimensions; can be relaxed via [Coarsened Exact Matching]                                                           |
+----------------------------+----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| **Mahalanobis Distance**   | $D_{ij} = (X_i - X_j)' \Sigma^{-1}(X_i - X_j)$     | $\Sigma$ (var-covar matrix of $X$) from control group if ATT is of interest or pooled if ATE is of interest; sensitive to dimensionality |
+----------------------------+----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| **Propensity Score**       | $D_{ij} = |e_i - e_j|$                             | Where $e_k$ is the estimated propensity score $P(T=1 \mid X_k)$ for unit $k$.                                                            |
|                            |                                                    |                                                                                                                                          |
|                            |                                                    | Advanced: **Prognostic scores** [@hansen2008prognostic] require modeling $E[Y(0)|X]$, so they depend on the outcome model.               |
+----------------------------+----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| **Logit Propensity Score** | $D_{ij} = |\text{logit}(e_i) - \text{logit}(e_j)|$ | More stable in tails of distribution                                                                                                     |
+----------------------------+----------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+

> Tip: in high dimensions, exact and Mahalanobis matching perform poorly. Combining Mahalanobis with propensity score calipers can improve robustness [@rubin2000combining].

Advanced methods for longitudinal setting:

-   Marginal Structural Models: for time-varying treatments [@robins2000marginal]
-   Balanced Risk Set Matching: for survival analysis [@li2001balanced]

------------------------------------------------------------------------

### Step 2: Matching Algorithms

The choice of matching algorithm shapes a familiar trade-off. Aggressive algorithms that demand near-exact agreement on every covariate (for instance, exact matching or fine [Coarsened Exact Matching](#sec-cem)) deliver sharp balance but can prune away most of the sample, leaving an estimand that applies to a narrow slice of the population and inflating variance. Permissive algorithms (such as nearest-neighbor matching on a single propensity score) retain more units and yield estimates with smaller standard errors, but residual imbalance can leak into bias. Researchers should think of each algorithm as picking a point on this balance versus sample retention frontier; the [`MatchingFrontier`](#sec-matching-methods) tooling discussed later makes that frontier explicit. The algorithms below differ chiefly in how they navigate that frontier.

1.  Nearest Neighbor Matching

Nearest-neighbor matching is the workhorse default. Each treated unit is paired with the closest available control under a chosen distance metric, producing an intuitive matched sample that mimics a pair-randomized experiment. It is easy to implement and analyze, but its quality depends entirely on the underlying distance metric and the supply of controls. When controls are scarce or covariates are many, "closest" can still be far, and the bias-variance trade-off becomes acute.

-   Greedy matching: fast, but suboptimal under competition for controls.
-   Optimal matching: minimizes global distance across all pairs.
-   Ratio matching (k:1): useful when controls outnumber treated; choose $k$ using trade-off between bias and variance [@rubin1996matching].
-   With vs. without replacement:
    -   With replacement: improves matching quality, but requires frequency weights for analysis.
    -   Without replacement: simpler, but less flexible.

2.  Subclassification, Full Matching, and Weighting

These methods generalize nearest-neighbor approaches by assigning fractional weights. Rather than discarding controls that fail to find a partner, every unit receives a weight that reflects its similarity to units in the other group. The result is a softer form of pruning that often retains the entire sample, trading some interpretability of "matched pairs" for efficiency. Weighting estimators are particularly attractive when the target estimand is the [ATE](#sec-counterfactual-estimators) rather than the [ATT](#sec-counterfactual-estimators), since they use information from both treated and control units symmetrically.

-   Subclassification: partition into strata based on propensity score (e.g., quintiles).
-   Full Matching: each treated unit is matched to a weighted group of controls (and vice versa) to minimize average within-set distance.
-   Weighting: weighting techniques use propensity scores to estimate the ATE. However, if the weights are extreme, the resulting variance may be inflated (not due to the underlying probabilities, but due to the estimation procedure itself). To address this issue, researchers can employ (1) weight trimming or (2) doubly robust methods when using propensity scores for weighting or matching.
    -   Inverse Probability of Treatment Weighting (IPTW): $$
        w_i = \frac{T_i}{\hat{e}_i} + \frac{1 - T_i}{1 - \hat{e}_i}
        $$
    -   Odds weighting: $$
        w_i = T_i + (1 - T_i)\frac{\hat{e}_i}{1 - \hat{e}_i}
        $$
    -   Kernel weighting: smooth average over control group (popular in economics).
    -   Trimming and Doubly-Robust Methods: reduce variance due to extreme weights.

3.  Assessing Common Support

-   Use propensity score histograms to visualize overlap.
-   Units outside the convex hull of $X$ (i.e., unmatched regions) can be discarded.
-   Lack of overlap indicates that some comparisons are extrapolations, not empirical matches.

------------------------------------------------------------------------

### Step 3: Diagnosing Match Quality

#### Balance Diagnostics

Matching aims to balance the covariate distributions between treated and control units. A well-matched sample satisfies:

$$
\tilde{p}(X \mid T=1) \approx \tilde{p}(X \mid T=0)
$$

where $\tilde{p}$ is the empirical distribution.

1.  Numerical Checks

-   Standardized differences in means (most common): should be $< 0.1$
-   Standardized difference of propensity scores: should be $< 0.25$ [@rubin2001using]
-   Variance ratio of propensity scores: between 0.5 and 2.0 [@rubin2001using]
-   Variance of residuals after regression on propensity score (treated vs. control) for each covariate

> Avoid using p-values as diagnostics: they conflate balance with statistical power and are sensitive to sample size.

2.  Graphical Diagnostics

-   Empirical Distribution Plots
-   Quantile-Quantile (QQ) Plots
-   Love Plots: summarize standardized differences before/after matching

------------------------------------------------------------------------

### Step 4: Estimating Treatment Effects

1.  After Matching

-   With k:1 matching with replacement, use weights to adjust for reuse of controls.
-   Use regression adjustment on matched samples to improve precision and adjust for residual imbalance.

2.  After Subclassification or Full Matching

-   ATT: weight subclass-specific estimates by number of treated units.
-   ATE: weight by total units per subclass.

3.  Variance Estimation

Must reflect uncertainty in both:

1.  The matching procedure (sampling and distance calculation) (Step 3)
2.  The outcome model (regression, difference-in-means, etc.) (Step 4)

Often estimated via bootstrapping.

------------------------------------------------------------------------

## Special Considerations

1.  Handling Missing Data

-   Use generalized boosted models or multiple imputation [@qu2009propensity].

2.  Violation of Ignorability

Strategies when unobservables bias treatment:

-   Use pre-treatment measures of outcome
-   Compare multiple control groups
-   Conduct sensitivity analysis:
    -   Quantify correlation between unobserved confounders and both treatment and outcome to nullify the observed effect

------------------------------------------------------------------------

## Choosing a Matching Strategy

No single matching algorithm dominates across applications. The right choice depends on the target estimand (see [counterfactual estimators](#sec-counterfactual-estimators)), the ratio of treated to control units, the dimensionality and type of covariates, and where the analyst wants to land on the balance versus sample retention trade-off. The guidance below sketches sensible defaults and is best read alongside the [balance checks](#sec-balance-checks) used to validate any choice.

### Based on Estimand

-   ATE: use IPTW or full matching
-   ATT:
    -   If many controls ($N_c > 3N_t$): k:1 nearest neighbor without replacement
    -   If few controls: subclassification, full matching, or odds weighting

### Based on Diagnostics

-   If balanced: proceed with regression on matched samples
-   If imbalance on few covariates: Mahalanobis matching on those
-   If imbalance on many covariates: Try k:1 matching with replacement

### Selection Criteria

-   Minimize standardized differences across many covariates
-   Especially prioritize prognostic covariates
-   Minimize number of covariates with large ($>0.25$) imbalance [@diamond2013genetic]

Matching is not one-size-fits-all. Choose methods based on the target estimand, data structure, and diagnostic results.

A short translation of the algorithmic choices into applied guidance: reach for [propensity score](#sec-propensity-scores) methods when covariates are many but the propensity model is plausibly well specified and the target is a population estimand; reach for [Mahalanobis distance](#sec-mahalanobis) when covariates are few, continuous, and well measured, and when emulating a fully blocked design is the priority; reach for [Coarsened Exact Matching](#sec-cem) when the analyst wants transparent, ex ante control over imbalance and is willing to coarsen continuous covariates into substantively meaningful bins; reach for [Genetic Matching](#sec-genetic-matching) when distance-metric specification is itself uncertain and the dataset is large enough to absorb its computational cost; reach for entropy balancing or weighting when treated units are few and discarding any of them would be unacceptable. Sensitivity to unobserved confounding (covered in [Rosenbaum bounds](#sec-rosenbaum-bounds) and the broader discussion of [selection on unobservables](#sec-selection-on-unobservables)) should accompany any matching analysis, since none of these algorithms can rescue a violation of [conditional ignorability](#sec-conditional-ignorability-assumption).

------------------------------------------------------------------------

## Inverse Probability Weighting {#sec-inverse-probability-weighting}

Weighting deserves a closer look than the brief mention it received among the matching algorithms above, because it is the direct bridge between matching and the estimating equations used by modern doubly robust and machine-learning estimators. The central insight is that inverse probability weighting (IPW) is not a causal invention at all. It is the Horvitz-Thompson estimator of survey sampling [@horvitz1952generalization], applied to the missing data problem that treatment assignment creates.

### The Horvitz-Thompson Foundation

In sampling theory, a finite population of units $i = 1, \dots, N$ carries fixed attributes $Y_i$, and each unit is sampled with known inclusion probability $\pi_i = P(S_i = 1)$. The Horvitz-Thompson estimator of the population mean is

$$
\hat{\mu}_{HT} = \frac{1}{N} \sum_{i=1}^N \frac{S_i Y_i}{\pi_i},
$$

which is unbiased because $E[S_i Y_i / \pi_i] = Y_i$. Each observed unit stands in for $1/\pi_i$ population units, so rarely sampled units are upweighted to undo the selection.

Causal inference under the potential outcomes framework poses the same problem. Each unit has two fixed potential outcomes $Y_i(1)$ and $Y_i(0)$, but treatment assignment reveals only one of them. Treatment plays the role of the sampling indicator, and the propensity score $e(X_i) = P(D_i = 1 \mid X_i)$ plays the role of the inclusion probability. Because $E[D / e(X) \mid X] = 1$, we obtain

$$
E[Y(1)] = E\left[\frac{D\, Y}{e(X)}\right], \qquad
E[Y(0)] = E\left[\frac{(1-D)\, Y}{1 - e(X)}\right],
$$

and the sample analog

$$
\widehat{ATE}_{IPW} = \frac{1}{N} \sum_{i=1}^N \left( \frac{D_i Y_i}{\hat{e}(X_i)} - \frac{(1 - D_i) Y_i}{1 - \hat{e}(X_i)} \right)
$$

is algebraically a Horvitz-Thompson estimator applied separately to the treated and control potential-outcome populations. The correspondence is exact:

| Sampling theory | Causal inference |
|---------------------------------|---------------------------------------|
| Population units | Units with potential outcomes |
| Sampling indicator $S_i$ | Treatment indicator $D_i$ |
| Inclusion probability $\pi_i$ | Propensity score $e(X_i)$ |
| Missing by sampling design | Missing counterfactual |
| Positive inclusion probability | Overlap, $0 < e(X) < 1$ |
| Selection on known design | [Selection on observables](#sec-selection-on-observables) |

This mapping clarifies two things at once. First, the identification assumptions of IPW are sampling assumptions in disguise: unconfoundedness is the analog of a known selection mechanism, and overlap is the analog of every unit having a positive chance of being sampled. Horvitz-Thompson fails when $\pi_i = 0$, and IPW fails when $e(X)$ hits 0 or 1. Second, the variance pathologies of IPW are structural rather than incidental. The Horvitz-Thompson estimator is unbiased but can have enormous variance when some inclusion probabilities are tiny, and the same heavy tails in $1/e(X)$ dominate finite samples in causal applications. Trimming and stabilized weights are variance-control devices, not identification tools.

### Weights, Estimands, and the Density-Ratio View

Different target estimands correspond to different weight functions, and the cleanest way to see this is through density ratios. Let $f(x)$ denote the marginal covariate density and $f(x \mid D=1)$, $f(x \mid D=0)$ the group-specific densities. Bayes' rule gives $f(x) = P(D=1)\, f(x \mid D=1) / e(x)$, so reweighting treated units by $1/e(x)$ transforms their covariate distribution into that of the full population. The same logic for other targets yields the standard menu:

| Target estimand | Treated weight | Control weight | Main risk |
|------------------|------------------|------------------|-------------------|
| ATE | $1/e(X)$ | $1/(1-e(X))$ | Extreme weights on both sides |
| ATT | $1$ | $e(X)/(1-e(X))$ | Extreme weights on controls |
| ATO (overlap) | $1-e(X)$ | $e(X)$ | None, but estimand changes |

A useful summary of how much information survives reweighting is the effective sample size,

$$
ESS = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2},
$$

which shrinks as the weights become more dispersed. Even a nominally large control group contributes little if the target population forces a handful of controls with large $e(X)$ to carry most of the weight.

### Overlap Weights

When overlap is limited, plain ATE weights explode near $e(X) = 0$ or $1$. The overlap weights of @li2018balancing replace $1/e(X)$ and $1/(1-e(X))$ with $1-e(X)$ for treated units and $e(X)$ for controls. Three properties make them attractive in practice. They are bounded by construction, so no observation can dominate. They achieve exact mean balance on any covariate included in a logistic propensity model, a small-sample property rather than an asymptotic one. And they minimize the asymptotic variance of the weighted estimator within the class of balancing weights. The price is a change of estimand: overlap weights target the average treatment effect in the overlap population, the units most in equipoise between treatment and control, rather than the full-population ATE. In settings such as clinical decision-making or marginal policy expansion this subpopulation is often the one of substantive interest, which is why overlap weighting has moved quickly from a variance fix to a first-choice design.

### A Small Simulation

The simulation below generates a confounded treatment with true ATE equal to 1 and compares the naive difference in means, regression adjustment, ATE-targeted IPW, and overlap weighting, along with the effective sample size each weighting scheme retains in the control group.


``` r
set.seed(123)
n  <- 5000
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rbinom(n, 1, 0.4)

# Confounded assignment and outcomes, true ATE = 1
e_true <- plogis(-0.5 + 0.8 * x1 - 0.6 * x2 + 0.4 * x3)
d  <- rbinom(n, 1, e_true)
y0 <- 2 + 1.0 * x1 + 0.5 * x2 + 0.3 * x3 + rnorm(n)
y  <- ifelse(d == 1, y0 + 1, y0)
dat <- data.frame(y, d, x1, x2, x3)

ps_fit <- glm(d ~ x1 + x2 + x3, data = dat, family = binomial())
e_hat  <- fitted(ps_fit)

ate_naive <- mean(dat$y[dat$d == 1]) - mean(dat$y[dat$d == 0])
ate_reg   <- coef(lm(y ~ d + x1 + x2 + x3, data = dat))["d"]

# ATE weights (Horvitz-Thompson)
w_ipw <- ifelse(dat$d == 1, 1 / e_hat, 1 / (1 - e_hat))
ate_ipw <- weighted.mean(dat$y[dat$d == 1], w_ipw[dat$d == 1]) -
    weighted.mean(dat$y[dat$d == 0], w_ipw[dat$d == 0])

# Overlap weights
w_ow <- ifelse(dat$d == 1, 1 - e_hat, e_hat)
ate_ow <- weighted.mean(dat$y[dat$d == 1], w_ow[dat$d == 1]) -
    weighted.mean(dat$y[dat$d == 0], w_ow[dat$d == 0])

ess <- function(w) sum(w)^2 / sum(w^2)

data.frame(
    estimator = c("Naive", "Regression", "IPW (ATE)", "Overlap"),
    estimate  = round(c(ate_naive, ate_reg, ate_ipw, ate_ow), 3),
    ess_controls = round(c(NA, NA, ess(w_ipw[dat$d == 0]),
                           ess(w_ow[dat$d == 0])))
)
#>    estimator estimate ess_controls
#> 1      Naive    1.323           NA
#> 2 Regression    0.943           NA
#> 3  IPW (ATE)    0.937         2085
#> 4    Overlap    0.943         2204
```

The naive comparison overstates the effect because treatment is concentrated where baseline outcomes are high, while regression adjustment, IPW, and overlap weighting all recover the truth. In this design overlap is healthy, so IPW and overlap weights perform similarly; as overlap deteriorates, the IPW estimate becomes erratic while the overlap-weighted estimate remains stable, at the cost of answering the question for the overlap population only. In applied work the `WeightIt` package estimates all of these weighting schemes with balance diagnostics, and weighting is usually paired with an outcome model in a doubly robust estimator, a combination developed in [Matching vs. Regression] below and in the machine-learning treatment of Chapter \@ref(sec-ml-causal).

------------------------------------------------------------------------

## Matching vs. Regression

Matching and regression are two core strategies used in observational studies to adjust for differences in covariates $X$ and estimate causal effects. While both aim to remove bias due to confounding, they approach the problem differently, particularly in how they weight observations, handle functional form assumptions, and address covariate balance.

Neither method can resolve the issue of unobserved confounding, but each can be a powerful tool when used with care and supported by appropriate diagnostics.

-   Matching emphasizes covariate balance by pruning the dataset to retain only comparable units. It is nonparametric, focusing on ATT.
-   Regression (typically [OLS][Ordinary Least Squares]) emphasizes functional form and allows for model-based adjustment, enabling the estimation of ATE and continuous or interactive effects of treatment.

Both matching and regression assign implicit or explicit weights to observations during estimation:

-   Matching: weights observations more heavily in strata with more treated units, aligning with the ATT estimand.
-   [OLS][Ordinary Least Squares] regression: places more weight on strata where the variance of treatment assignment is highest, i.e., when groups are approximately balanced between treated and control (near 50/50).

This results in differing estimands and sensitivities:

> Important Caveat: if your [OLS][Ordinary Least Squares] estimate is biased due to unobserved confounding, your matching estimate is likely biased too. Both depend on the [selection on observables](#sec-selection-on-observables) assumption.

We explore the difference in estimands between matching and regression, especially for estimating the ATT.

### Matching Estimand

Suppose we want the treatment effect on the treated:

$$
\delta_{\text{TOT}} = E[Y_{1i} - Y_{0i} \mid D_i = 1]
$$

Using the [Law of Iterated Expectation]:

$$
\delta_{\text{TOT}} = E\left[ E[Y_{1i} \mid X_i, D_i = 1] - E[Y_{0i} \mid X_i, D_i = 1] \mid D_i = 1 \right]
$$

Assuming conditional independence:

$$
E[Y_{0i} \mid X_i, D_i = 0] = E[Y_{0i} \mid X_i, D_i = 1]
$$

Then,

$$
\begin{aligned}
\delta_{TOT} &= E [ E[ Y_{1i} | X_i, D_i = 1] - E[ Y_{0i}|X_i, D_i = 0 ]|D_i = 1 ] \\
&= E\left[ E[Y_i \mid X_i, D_i = 1] - E[Y_i \mid X_i, D_i = 0] \mid D_i = 1 \right] \\
&= E[\delta_X |D_i = 1]
\end{aligned}
$$

where $\delta_X$ is an $X$-specific difference in means at covariate value $X_i$

If $X_i$ is discrete, the matching estimand becomes:

$$
\delta_M = \sum_x \delta_x P(X_i = x \mid D_i = 1)
$$

where $P(X_i = x |D_i = 1)$ is the probability mass function for $X_i$ given $D_i = 1$

By Bayes' rule:

$$
P(X_i = x \mid D_i = 1) = \frac{P(D_i = 1 \mid X_i = x) P(X_i = x)}{P(D_i = 1)}
$$

So,

$$
\begin{aligned}
\delta_M &= \frac{\sum_x \delta_x P (D_i = 1 | X_i = x) P (X_i = x)}{\sum_x P(D_i = 1 |X_i = x)P(X_i = x)} \\
&= \sum_x \delta_x \frac{ P (D_i = 1 | X_i = x) P (X_i = x)}{\sum_x P(D_i = 1 |X_i = x)P(X_i = x)}
\end{aligned}
$$

------------------------------------------------------------------------

### Regression Estimand

In regression:

$$
Y_i = \sum_x d_{ix} \beta_x + \delta_R D_i + \varepsilon_i
$$

-   $d_{ix}$ = indicator that $X_i = x$
-   $\beta_x$ = baseline outcome at $X = x$
-   $\delta_R$ = regression estimand

Then,

$$
\begin{aligned}
\delta_R &= \frac{\sum_x \delta_x [P(D_i = 1 | X_i = x) (1 - P(D_i = 1 | X_i = x))]P(X_i = x)}{\sum_x [P(D_i = 1| X_i = x)(1 - P(D_i = 1 | X_i = x))]P(X_i = x)} \\
&= \sum_x \delta_x \frac{[P(D_i = 1 | X_i = x) (1 - P(D_i = 1 | X_i = x))]P(X_i = x)}{\sum_x [P(D_i = 1| X_i = x)(1 - P(D_i = 1 | X_i = x))]P(X_i = x)}
\end{aligned}
$$

------------------------------------------------------------------------

### Interpretation: Weighting Differences

The distinction between matching and regression comes down to how covariate-specific treatment effects $\delta_x$ are weighted:

+-------------+--------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| Type        | Weighting Function                                                                   | Interpretation                                                                               | Makes Sense Because...                                                                                              |
+=============+======================================================================================+==============================================================================================+=====================================================================================================================+
| Matching    | $P(D_i = 1 \mid X_i = x)$                                                            | Weights more heavily where more treated units exist (ATT-focused)                            | We're interested in the effect on the treated, so more weight is placed where treated units are observed            |
+-------------+--------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| Regression  | $\begin{aligned}P(D_i = 1 \mid X_i = x)\\(1 - P(D_i = 1 \mid X_i = x))\end{aligned}$ | Weights more where treatment assignment has high variance (i.e., near 50/50 treated/control) | These cells provide lowest-variance estimates of $\delta_x$, assuming the treatment effect is homogenous across $X$ |
+-------------+--------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+

### Summary Table: Matching vs. Regression

+----------------------------+----------------------------------------------------------+-----------------------------------------------------+
| Feature                    | Matching                                                 | Regression (OLS)                                    |
+============================+==========================================================+=====================================================+
| **Functional Form**        | Less parametric; no assumption of linearity              | Parametric; usually assumes linearity               |
+----------------------------+----------------------------------------------------------+-----------------------------------------------------+
| **Primary Estimand**       | ATT (effect on the treated)                              | ATE or effects of continuous/interacted treatments  |
+----------------------------+----------------------------------------------------------+-----------------------------------------------------+
| **Balance**                | Enforces balance via matched samples                     | Does not guarantee balance                          |
+----------------------------+----------------------------------------------------------+-----------------------------------------------------+
| **Diagnostics**            | Covariate SMDs, QQ plots, empirical distributions        | Residual plots, R-squared, heteroskedasticity tests |
+----------------------------+----------------------------------------------------------+-----------------------------------------------------+
| **Unobserved Confounding** | Cannot be resolved; assumes ignorability                 | Same limitation                                     |
+----------------------------+----------------------------------------------------------+-----------------------------------------------------+
| **Standard Errors**        | Larger; require bootstrapping                            | Smaller; closed-form under assumptions              |
+----------------------------+----------------------------------------------------------+-----------------------------------------------------+
| **Best Used When**         | High control-to-treated ratio; misspecification concerns | Model is correctly specified; sufficient overlap    |
+----------------------------+----------------------------------------------------------+-----------------------------------------------------+

------------------------------------------------------------------------

### Qualitative Comparisons

+---------------------------------------------------------------------+----------------------------------------------------------------------+
| Matching                                                            | Regression                                                           |
+=====================================================================+======================================================================+
| Not sensitive to the form of covariate-outcome relationship         | Can estimate continuous or interacted treatment effects              |
+---------------------------------------------------------------------+----------------------------------------------------------------------+
| Easier to assess balance and interpret diagnostics                  | Easier to estimate the effects of all covariates, not just treatment |
+---------------------------------------------------------------------+----------------------------------------------------------------------+
| Facilitates clear visual evaluation of overlap and balance          | Less intuitive diagnostics; model diagnostics used                   |
+---------------------------------------------------------------------+----------------------------------------------------------------------+
| Helps when treatment is rare (prunes clearly incomparable controls) | Performs better with balanced treatment assignment                   |
+---------------------------------------------------------------------+----------------------------------------------------------------------+
| Forces explicit enforcement of common support                       | May extrapolate outside the support of covariate distributions       |
+---------------------------------------------------------------------+----------------------------------------------------------------------+

------------------------------------------------------------------------

## Software and Practical Implementation

Many R packages provide functionality for implementing the various matching methods discussed above. Below is an overview of some popular options:

-   MatchIt:\
    Implements a wide range of matching methods (nearest neighbor, optimal, full, subclassification, exact, etc.). It focuses on "preprocessing" data before a final outcome analysis.

-   Matching:\
    Provides multivariate and propensity score matching, including options for *exact* and *nearest neighbor* matching. The package also offers functions to evaluate balance and to conduct sensitivity analyses.

-   cem ([Coarsened Exact Matching](#sec-cem)):\
    Uses a coarsening approach to create strata within which exact matching can be performed. This can reduce imbalance by discarding units that do not overlap in coarsened covariate space.

-   optmatch:\
    Enables *optimal matching* with variable matching ratios and full matching, allowing for flexible group constructions that minimize overall distance.

-   MatchingFrontier [@king2017balance]:\
    Finds the "frontier" of matching solutions by balancing sample size (or other constraints) against covariate balance. Allows analysts to see trade-offs in real time.

-   CBPS (Covariate Balancing Propensity Score):\
    Estimates propensity scores such that covariate balance is directly optimized. This can help avoid iterative re-specification of the propensity score model.

-   PanelMatch [@rauh2025panelmatch]:\
    Tailored to panel (longitudinal) data settings, providing matching methods that exploit repeated observations over time (e.g., for DID-type analyses in a time-series cross-sectional environment). Covered in depth, with a full replication and diagnostics, in Section \@ref(sec-panel-matching) at the end of this chapter.

-   PSAgraphics:\
    Specializes in visual diagnostics for propensity score analyses, offering graphical tools to inspect balance and common support.

-   rbounds:\
    Conducts Rosenbaum bounds sensitivity analysis on matched data. Researchers can examine how a hypothetical unmeasured confounder could undermine their estimated treatment effects.

-   twang:\
    Implements *generalized boosted models (GBM)* to estimate propensity scores. Often used for weighting approaches such as inverse probability weighting (IPW).

In practice, the choice of software and methods hinges on the study design, the nature of the data, and the researcher's theoretical expectations regarding treatment assignment.

------------------------------------------------------------------------

## Selection on Observables {#sec-selection-on-observables}

In observational studies, treatment assignment is typically not randomized. This poses a challenge when estimating causal effects, as treated and control groups may differ systematically. A central assumption that allows us to estimate causal effects from such data is selection on observables, also known as unconfoundedness or conditional independence.

Suppose we observe a binary treatment indicator $T_i \in {0, 1}$ and an outcome $Y_i$. Each unit $i$ has two potential outcomes:

-   $Y_i(1)$: outcome if treated
-   $Y_i(0)$: outcome if untreated

However, only one of these outcomes is observed for each unit. The average treatment effect on the treated (ATT) is:

$$
\text{ATT} = \mathbb{E}[Y(1) - Y(0) \mid T = 1]
$$

To identify this from data, we invoke the conditional independence assumption:

$$
(Y(0), Y(1)) \perp T \mid X
$$

This assumption means that after controlling for covariates $X$, treatment is as good as randomly assigned. A secondary assumption is overlap:

$$
0 < \mathbb{P}(T = 1 \mid X = x) < 1 \quad \text{for all } x
$$

This ensures that for every covariate profile, there is a positive probability of being both treated and untreated.

Matching attempts to approximate the conditions of a randomized experiment by creating a comparison group that is similar to the treated group in terms of observed covariates. Instead of relying solely on model-based adjustment (e.g., regression), matching balances the covariate distribution across treatment groups before estimation.

------------------------------------------------------------------------

### Matching with `MatchIt`

We demonstrate the matching procedure using the `lalonde` dataset, a classic example in the causal inference literature, which investigates the effect of job training on subsequent earnings.


``` r
library(MatchIt)
data("lalonde")
```

We focus on estimating the effect of the treatment (`treat`) on earnings in 1978 (`re78`), conditional on covariates.

------------------------------------------------------------------------

#### Step 1: Planning the Analysis

Before conducting matching, several strategic decisions must be made:

-   Estimand: do you want ATT (effect on treated), ATE (effect in population), or ATC (effect on controls)? Matching typically targets the ATT.

-   Covariate Selection: only include pre-treatment variables that are potential confounders, i.e., that affect both the treatment assignment and the outcome [@austin2011optimal; @vanderweele2019principles].

-   Distance Measure: choose how to quantify similarity between units (e.g., propensity score, Mahalanobis distance).

-   Matching Method: determine the method (e.g., nearest neighbor, full matching, genetic matching).

For our demonstration, we focus on ATT using propensity score matching.

------------------------------------------------------------------------

#### Step 2: Assessing Initial Imbalance

We first assess imbalance between treatment and control groups before matching.


``` r
# Estimate propensity scores with logistic regression
m.out0 <- matchit(
  treat ~ age + educ + race + married + nodegree + re74 + re75,
  data = MatchIt::lalonde,
  method = NULL,     # no matching, only estimates propensity scores
  distance = "glm"
)

# Summary of balance statistics before matching
summary(m.out0)
#> 
#> Call:
#> matchit(formula = treat ~ age + educ + race + married + nodegree + 
#>     re74 + re75, data = MatchIt::lalonde, method = NULL, distance = "glm")
#> 
#> Summary of Balance for All Data:
#>            Means Treated Means Control Std. Mean Diff. Var. Ratio
#> distance          0.5774        0.1822          1.7941     0.9211
#> age              25.8162       28.0303         -0.3094     0.4400
#> educ             10.3459       10.2354          0.0550     0.4959
#> raceblack         0.8432        0.2028          1.7615          .
#> racehispan        0.0595        0.1422         -0.3498          .
#> racewhite         0.0973        0.6550         -1.8819          .
#> married           0.1892        0.5128         -0.8263          .
#> nodegree          0.7081        0.5967          0.2450          .
#> re74           2095.5737     5619.2365         -0.7211     0.5181
#> re75           1532.0553     2466.4844         -0.2903     0.9563
#>            eCDF Mean eCDF Max
#> distance      0.3774   0.6444
#> age           0.0813   0.1577
#> educ          0.0347   0.1114
#> raceblack     0.6404   0.6404
#> racehispan    0.0827   0.0827
#> racewhite     0.5577   0.5577
#> married       0.3236   0.3236
#> nodegree      0.1114   0.1114
#> re74          0.2248   0.4470
#> re75          0.1342   0.2876
#> 
#> Sample Sizes:
#>           Control Treated
#> All           429     185
#> Matched       429     185
#> Unmatched       0       0
#> Discarded       0       0
```

This summary provides:

-   Standardized mean differences

-   Variance ratios

-   Propensity score distributions

These diagnostics help us understand the extent of covariate imbalance.

------------------------------------------------------------------------

#### Step 3: Implementing Matching

1.  Nearest Neighbor Matching (1:1 without Replacement)


``` r
m.out1 <- matchit(
  treat ~ age + educ + race + married + nodegree + re74 + re75,
  data = MatchIt::lalonde,
  method = "nearest",
  distance = "glm"
)

```

Matching is based on estimated propensity scores. Each treated unit is matched to the closest control unit.

Assess Balance After Matching

The post-matching balance diagnostics for the nearest-neighbour 1:1 propensity-score match come in two parts: Figure \@ref(fig:mm-nearest-jitter) shows the propensity-score distribution and Figure \@ref(fig:mm-nearest-qq) the covariate QQ plots.


``` r
summary(m.out1, un = FALSE)  # only show post-matching stats
#> 
#> Call:
#> matchit(formula = treat ~ age + educ + race + married + nodegree + 
#>     re74 + re75, data = MatchIt::lalonde, method = "nearest", 
#>     distance = "glm")
#> 
#> Summary of Balance for Matched Data:
#>            Means Treated Means Control Std. Mean Diff. Var. Ratio
#> distance          0.5774        0.3629          0.9739     0.7566
#> age              25.8162       25.3027          0.0718     0.4568
#> educ             10.3459       10.6054         -0.1290     0.5721
#> raceblack         0.8432        0.4703          1.0259          .
#> racehispan        0.0595        0.2162         -0.6629          .
#> racewhite         0.0973        0.3135         -0.7296          .
#> married           0.1892        0.2108         -0.0552          .
#> nodegree          0.7081        0.6378          0.1546          .
#> re74           2095.5737     2342.1076         -0.0505     1.3289
#> re75           1532.0553     1614.7451         -0.0257     1.4956
#>            eCDF Mean eCDF Max Std. Pair Dist.
#> distance      0.1321   0.4216          0.9740
#> age           0.0847   0.2541          1.3938
#> educ          0.0239   0.0757          1.2474
#> raceblack     0.3730   0.3730          1.0259
#> racehispan    0.1568   0.1568          1.0743
#> racewhite     0.2162   0.2162          0.8390
#> married       0.0216   0.0216          0.8281
#> nodegree      0.0703   0.0703          1.0106
#> re74          0.0469   0.2757          0.7965
#> re75          0.0452   0.2054          0.7381
#> 
#> Sample Sizes:
#>           Control Treated
#> All           429     185
#> Matched       185     185
#> Unmatched     244       0
#> Discarded       0       0
```


``` r
# Visual diagnostic: jitter plot of propensity scores
plot(m.out1, type = "jitter", interactive = FALSE)
```

<div class="figure" style="text-align: center">
<img src="35-matching-methods_files/figure-html/mm-nearest-jitter-1.png" alt="Propensity scores of treated and control units after 1:1 nearest-neighbour matching. Unmatched units at the extremes are the region of the covariate space where no comparison exists, and the matched estimate says nothing about them." width="90%" />
<p class="caption">(\#fig:mm-nearest-jitter)Propensity scores of treated and control units after 1:1 nearest-neighbour matching. Unmatched units at the extremes are the region of the covariate space where no comparison exists, and the matched estimate says nothing about them.</p>
</div>


``` r
# QQ plot for individual covariates
plot(m.out1, type = "qq", which.xs = c("age", "re74"))
```

<div class="figure" style="text-align: center">
<img src="35-matching-methods_files/figure-html/mm-nearest-qq-1.png" alt="Quantile-quantile plots of age and 1974 earnings for matched treated against matched control units. Points on the diagonal mean the two groups share a marginal distribution on that covariate, which is what the match was for." width="90%" />
<p class="caption">(\#fig:mm-nearest-qq)Quantile-quantile plots of age and 1974 earnings for matched treated against matched control units. Points on the diagonal mean the two groups share a marginal distribution on that covariate, which is what the match was for.</p>
</div>

**Interpretation**: Good matches will show overlapping distributions of covariates across groups, and standardized differences should be below 0.1 in absolute value.

2.  Full Matching

Allows many-to-one or one-to-many matches, minimizing overall distance.


``` r
m.out2 <- matchit(
  treat ~ age + educ + race + married + nodegree + re74 + re75,
  data = MatchIt::lalonde,
  method = "full",
  distance = "glm",
  link = "probit"
)

summary(m.out2, un = FALSE)
#> 
#> Call:
#> matchit(formula = treat ~ age + educ + race + married + nodegree + 
#>     re74 + re75, data = MatchIt::lalonde, method = "full", distance = "glm", 
#>     link = "probit")
#> 
#> Summary of Balance for Matched Data:
#>            Means Treated Means Control Std. Mean Diff. Var. Ratio
#> distance          0.5773        0.5764          0.0045     0.9949
#> age              25.8162       25.5347          0.0393     0.4790
#> educ             10.3459       10.5381         -0.0956     0.6192
#> raceblack         0.8432        0.8389          0.0119          .
#> racehispan        0.0595        0.0492          0.0435          .
#> racewhite         0.0973        0.1119         -0.0493          .
#> married           0.1892        0.1633          0.0660          .
#> nodegree          0.7081        0.6577          0.1110          .
#> re74           2095.5737     2100.2150         -0.0009     1.3467
#> re75           1532.0553     1561.4420         -0.0091     1.5906
#>            eCDF Mean eCDF Max Std. Pair Dist.
#> distance      0.0043   0.0486          0.0198
#> age           0.0787   0.2742          1.2843
#> educ          0.0253   0.0730          1.2179
#> raceblack     0.0043   0.0043          0.0162
#> racehispan    0.0103   0.0103          0.4412
#> racewhite     0.0146   0.0146          0.3454
#> married       0.0259   0.0259          0.4473
#> nodegree      0.0504   0.0504          0.9872
#> re74          0.0314   0.1881          0.8387
#> re75          0.0536   0.1984          0.8240
#> 
#> Sample Sizes:
#>               Control Treated
#> All            429.       185
#> Matched (ESS)   50.76     185
#> Matched        429.       185
#> Unmatched        0.         0
#> Discarded        0.         0
```

3.  Exact Matching

Only matches units with exactly the same covariate values (usually categorical):


``` r
m.out3 <- matchit(
  treat ~ race + nodegree,
  data = MatchIt::lalonde,
  method = "exact"
)

```

4.  Optimal Matching

Minimizes the total distance between matched units across the sample.


``` r
m.out4 <- matchit(
  treat ~ age + educ + re74 + re75,
  data = MatchIt::lalonde,
  method = "optimal",
  ratio = 2
)

```

5.  Genetic Matching

Searches over weights assigned to covariates to optimize balance.


``` r
m.out5 <- matchit(
  treat ~ age + educ + re74 + re75,
  data = MatchIt::lalonde,
  method = "genetic"
)

```

------------------------------------------------------------------------

#### Step 4: Estimating Treatment Effects

Once matching is complete, we use the matched data to estimate treatment effects.


``` r
library(lmtest)
library(sandwich)

# Extract matched data
matched_data <- match.data(m.out1)

# Estimate ATT with robust standard errors
model <- lm(re78 ~ treat + age + educ + race + re74 + re75,
            data = matched_data,
            weights = weights)

coeftest(model, vcov. = vcovCL, cluster = ~subclass)
#> 
#> t test of coefficients:
#> 
#>                Estimate  Std. Error t value Pr(>|t|)   
#> (Intercept) -437.664937 1912.759171 -0.2288 0.819143   
#> treat       1398.134870  723.745751  1.9318 0.054164 . 
#> age           -0.343085   39.256789 -0.0087 0.993032   
#> educ         470.767350  147.892765  3.1832 0.001583 **
#> racehispan  1518.303924 1035.083141  1.4668 0.143287   
#> racewhite    557.295853  897.121013  0.6212 0.534856   
#> re74           0.017244    0.166298  0.1037 0.917470   
#> re75           0.226076    0.165722  1.3642 0.173357   
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

The coefficient on `treat` is our estimate of the ATT. The weights and subclass clustering account for the matched design.

### Reporting Standards

To ensure transparency and reproducibility, always report the following:

1.  Matching method (e.g., nearest neighbor, genetic)

2.  Distance metric (e.g., propensity score via logistic regression)

3.  Covariates matched on, and justification for their inclusion

4.  Balance statistics (e.g., standardized mean differences before/after)

5.  Sample sizes: total, matched, unmatched, discarded

6.  Estimation model: whether treatment effects are estimated using regression adjustment, with or without weights

7.  Assumptions: especially unconfoundedness and overlap

------------------------------------------------------------------------

### Optimization-Based Matching via `designmatch`

For more advanced applications, the `designmatch` package provides matching methods based on combinatorial optimization.


``` r
library(designmatch)
```

Notable methods:

-   `distmatch()`: Distance-based matching with custom constraints

-   `bmatch()`: Bipartite matching using linear programming

-   `cardmatch()`: Cardinality matching for maximum matched sample size with balance constraints

-   `profmatch()`: Profile matching for stratified treatment allocation

-   `nmatch()`: Non-bipartite matching (e.g., in interference settings)

------------------------------------------------------------------------

### MatchingFrontier

As mentioned in `MatchIt`, you have to make trade-off (also known as bias-variance trade-off) between balance and sample size. An automated procedure to optimize this trade-off is implemented in `MatchingFrontier` [@king2017balance], which solves this joint optimization problem.

Following `MatchingFrontier` [guide](https://projects.iq.harvard.edu/files/frontier/files/using_matchingfrontier.pdf)


``` r
# library(devtools)
# install_github('ChristopherLucas/MatchingFrontier')
library(MatchingFrontier)
data("lalonde", package = "MatchIt")
# choose var to match on
match.on <-
    colnames(lalonde)[!(colnames(lalonde) %in% c('re78', 'treat'))]
match.on

# Mahanlanobis frontier (default)
mahal.frontier <-
    makeFrontier(
        dataset = lalonde,
        treatment = "treat",
        match.on = match.on
    )
mahal.frontier

# L1 frontier
L1.frontier <-
    makeFrontier(
        dataset = lalonde,
        treatment = 'treat',
        match.on = match.on,
        QOI = 'SATT',
        metric = 'L1',
        ratio = 'fixed'
    )
L1.frontier

# estimate effects along the frontier

# Set base form
my.form <-
    as.formula(re78 ~ treat + age + black + education 
               + hispanic + married + nodegree + re74 + re75)

# Estimate effects for the mahalanobis frontier
mahal.estimates <-
    estimateEffects(
        mahal.frontier,
        're78 ~ treat',
        mod.dependence.formula = my.form,
        continuous.vars = c('age', 'education', 're74', 're75'),
        prop.estimated = .1,
        means.as.cutpoints = TRUE
    )

# Estimate effects for the L1 frontier
L1.estimates <-
    estimateEffects(
        L1.frontier,
        're78 ~ treat',
        mod.dependence.formula = my.form,
        continuous.vars = c('age', 'education', 're74', 're75'),
        prop.estimated = .1,
        means.as.cutpoints = TRUE
    )

# Plot covariates means 
# plotPrunedMeans()


# Plot estimates (deprecated)
# plotEstimates(
#     L1.estimates,
#     ylim = c(-10000, 3000),
#     cex.lab = 1.4,
#     cex.axis = 1.4,
#     panel.first = grid(NULL, NULL, lwd = 2,)
# )

# Plot estimates
plotMeans(L1.frontier)


# parallel plot
parallelPlot(
    L1.frontier,
    N = 400,
    variables = c('age', 're74', 're75', 'black'),
    treated.col = 'blue',
    control.col = 'gray'
)

# export matched dataset
# take 400 units
matched.data <- generateDataset(L1.frontier, N = 400) 
```

------------------------------------------------------------------------

### Propensity Scores {#sec-propensity-scores}

Propensity score methods are widely used for estimating causal effects in observational studies, where random assignment to treatment and control groups is not feasible. Conceptually, the propensity score collapses a high-dimensional vector of covariates into a single scalar (the conditional probability of treatment), and matching, weighting, or stratifying on that scalar is sufficient to remove confounding from the *observed* covariates. The appeal is dimensional: instead of searching for control units that look similar across many variables at once, the analyst searches along a one-dimensional probability axis. The cost is that two units sharing the same propensity score can still differ markedly in covariate space, which is why propensity score methods trade balance on the joint distribution of $X$ for tractability. They are most defensible when the propensity model is well specified, the covariate set is rich enough to make [conditional ignorability](#sec-conditional-ignorability-assumption) credible, and [overlap](#sec-overlap-positivity-assumption) is strong.

The core idea is to mimic a randomized experiment by adjusting for confounding variables that predict treatment assignment. Formally, the propensity score is defined as the probability of assignment to treatment conditional on observed covariates [@rosenbaum1983central; @rosenbaum1985bias]:

$$
e_i(X_i) = P(T_i = 1 \mid X_i)
$$

where:

-   $T_i \in {0, 1}$ is the binary treatment indicator for unit $i$,
-   $X_i$ is a vector of observed pre-treatment covariates for unit $i$.

The key insight from @rosenbaum1983central is that conditioning on the propensity score is sufficient to remove bias due to confounding from observed covariates, under certain assumptions.

------------------------------------------------------------------------

#### Assumptions for Identification

To identify causal effects using propensity scores, the following assumptions must hold:

-   Unconfoundedness / Conditional Independence Assumption (CIA):

$$
(Y_i(0), Y_i(1)) \perp T_i \mid X_i
$$

-   Positivity (Overlap):

$$
0 < P(T_i = 1 \mid X_i) < 1 \quad \text{for all } i
$$

These assumptions ensure that for each unit, we can observe comparable treated and untreated units in the sample. Violations of positivity, especially in high-dimensional covariate spaces, are a critical weakness of propensity score matching.

------------------------------------------------------------------------

#### Why PSM Is Not Recommended Anymore

Despite its intuitive appeal, recent literature has strongly cautioned against using propensity score matching for causal inference [@king2019propensity]. The main criticisms are as follows:

-   Imbalance: PSM often fails to achieve covariate balance better than simpler techniques like covariate adjustment via regression or exact matching. Matching on the propensity score is a *scalar reduction* of a multivariate distribution, and this reduction can distort multivariate relationships.
-   Inefficiency: discarding unmatched units reduces statistical efficiency, especially when better estimators (e.g., inverse probability weighting or doubly robust estimators) can use all data.
-   Model dependence: small changes in the specification of the propensity score model can lead to large changes in matches and estimated treatment effects.
-   Bias: poor matches and irrelevant covariates can introduce additional bias rather than reduce it.

@abadie2016matching show that the asymptotic distribution of treatment effect estimators is sensitive to the estimation of the propensity score itself:

-   The estimated propensity score can improve efficiency over using the true propensity score when estimating the [ATE](#sec-average-treatment-effect). Formally, the adjustment to the asymptotic variance is non-positive.
-   However, for the [ATT](#sec-average-treatment-effect-on-the-treated), the sign of the adjustment is data-dependent. Estimation error in the propensity score can lead to misestimated confidence intervals: they may be too wide or too narrow.

This result suggests that even in large samples, failure to account for the estimation uncertainty in the propensity score can produce misleading inference.

A fundamental flaw in PSM is the asymmetry of match quality:

-   If $X_c = X_t$, then it must be that $e(X_c) = e(X_t)$.
-   However, the converse does not hold:\
    $e(X_c) = e(X_t) \nRightarrow X_c = X_t$

Therefore, two units with identical propensity scores may still differ substantially in covariate space. This undermines the matching goal of achieving similarity across all covariates.

------------------------------------------------------------------------

#### Estimating the Propensity Score

Estimation of the propensity score is typically carried out using:

-   Parametric models:
    -   Logistic regression:\
        $$
        \hat{e}_i = \frac{1}{1 + \exp(-X_i^\top \hat{\beta})}
        $$
-   Nonparametric / machine learning methods:
    -   Generalized Boosted Models (GBM)
    -   Boosted Classification and Regression Trees (CART)
    -   Random forests or Bayesian Additive Regression Trees (BART)

These machine learning approaches often yield better balance due to flexible functional forms, but they also complicate interpretation and inference.

------------------------------------------------------------------------

#### Matching Algorithms

1.  Reduce the high-dimensional vector $X_i$ to the scalar $\hat{e}_i$.
2.  Calculate distances between treated and control units based on $\hat{e}_i$: $$
    d(i, j) = |\hat{e}_i - \hat{e}_j|
    $$
3.  Match each treated unit $i$ to the control unit $j$ with the closest propensity score.
4.  Prune:
    -   Control units not used in any match are discarded.
    -   Matches exceeding a caliper (maximum distance threshold) are also discarded.
5.  No replacement is typically assumed; each control unit is matched at most once.

Let $c > 0$ be a caliper. Then a treated unit $i$ is matched to control unit $j$ only if:

$$
|\hat{e}_i - \hat{e}_j| < c
$$

Caliper matching helps reduce poor matches, but may result in *random pruning*, further reducing balance and efficiency.

------------------------------------------------------------------------

#### Practical Recommendations

-   Do not include irrelevant covariates: including variables that are unrelated to the outcome can increase the variability of the estimated propensity score and reduce matching quality.
-   Avoid instrumental variables [@bhattacharya2007instrumental]: including IVs in the propensity score model can inflate bias by introducing variation in treatment assignment unrelated to potential outcomes.
-   Focus on covariates that are confounders, i.e., those that affect both treatment and outcome.

What remains after pruning is more consequential than the initial covariate set. Matching can discard a large portion of the sample, which distorts representativeness and increases variance.

------------------------------------------------------------------------

#### Diagnostics and Evaluation

After matching, the primary diagnostic tool is covariate balance. Key diagnostics include:

-   **Standardized mean differences (SMD)** between treatment groups before and after matching
-   **Variance ratios**
-   **Visual inspection** via Love plots or density plots

Statistical model fit criteria (e.g., AIC, BIC, c-statistics) are **not valid** for evaluating propensity score models, since the goal is *not prediction*, but achieving balance.

There is **no need to worry about collinearity** in the covariates when estimating propensity scores --- unlike in outcome regression models, where multicollinearity can inflate standard errors.

------------------------------------------------------------------------

#### Applications in Business and Finance

Propensity score methods have been used in empirical finance and marketing, though increasingly replaced by more robust approaches. One illustrative application is found in @hirtle2020impact, which investigates the causal effect of regulatory bank supervision on firm-level outcomes:

-   Treatment: Degree of supervisory attention.
-   Outcomes: Loan risk, profitability, volatility, and firm growth.
-   Method: Propensity score matching to construct treated and control groups of banks with comparable observed characteristics.

Their matched sample analysis reveals that **intensified supervision** leads to:

-   Lower risk (more conservative loan portfolios)
-   Reduced volatility
-   No significant loss in profitability or growth

------------------------------------------------------------------------

#### Conclusion

While the theoretical motivation behind propensity scores remains sound, their application via naive matching methods is no longer considered best practice. The statistical community increasingly favors alternatives such as:

-   **Covariate adjustment via regression**
-   **Inverse probability weighting (IPW)**
-   **Doubly robust estimators**
-   **Targeted Maximum Likelihood Estimation (TMLE)**

These approaches better utilize the full dataset, yield more efficient estimators, and offer more transparent diagnostics. Matching on estimated propensity scores may still be useful for illustration or sensitivity analysis, but should not be the primary method for causal inference in applied research.

------------------------------------------------------------------------

#### Look-Ahead Propensity Score Matching

In observational data, estimating **causal effects** is complicated by **self-selection bias**: individuals who receive a treatment may differ systematically from those who do not. Standard **PSM** methods attempt to control for such biases by matching treated and untreated units on observable characteristics.

However, when **unobservable** or **latent traits** influence both treatment assignment and outcomes, traditional PSM may produce biased estimates. @bapna2018monetizing introduce **Look-Ahead Propensity Score Matching (LA-PSM)**, a novel technique that leverages **future behavior** to correct for **time-invariant unobserved confounding**, particularly useful in rare-event economic decisions.

In the study of @bapna2018monetizing:

-   Treatment: user pays for a premium subscription.
-   Outcome: social engagement (songs listened, playlists created, friends made).
-   Problem: users choosing to subscribe are systematically different (e.g., more engaged).

##### LA-PSM Implementation

-   Treated group: users subscribing between Sept 2011 - June 2012.

-   Control group: users who will subscribe later (June 2012 - Jan 2015).

-   Matching on: observed demographics, past behavior.

-   Analysis: [Difference-in-Differences](#sec-difference-in-differences) estimation.

**Results:** Premium adoption increases:

-   Songs listened: +287.2%
-   Playlists created: +1.92%
-   Forum posts: +2.01%
-   Friends added: +15.77%

------------------------------------------------------------------------

In standard PSM:

-   Match individuals who received treatment with individuals who did not, based on observed covariates.
-   Assume selection into treatment is **strongly ignorable**, conditional on covariates.

Problem:

-   Unobserved factors (e.g., ambition, risk tolerance) may bias both treatment and outcome.

-   Standard PSM cannot correct for this.

##### Solution: Look-Ahead PSM

-   Match treated units with **future-treated** units: those who haven't yet received treatment but will in the future.

-   Future-treated units share unobservable traits with current-treated units.

Thus, LA-PSM controls for **time-invariant unobserved confounders**.

------------------------------------------------------------------------

Let:

-   $i \in \{1, \ldots, N\}$ index individuals.

-   $D_i(t)$ = 1 if individual $i$ is treated at time $t$, 0 otherwise.

-   $X_i$ = observed covariates.

-   $U_i$ = unobserved time-invariant confounders.

-   $Y_i(t)$ = outcome at time $t$.

In standard PSM:

-   Match on $\mathbb{P}(D_i(t) = 1 \mid X_i)$.

In LA-PSM:

-   Match **current treated** individuals with **future treated** individuals based on: $$ \mathbb{P}(D_i(t) = 1 \mid X_i) $$

-   and require: $$ \exists  s > t \quad \text{such that} \quad D_i(s) = 1 $$

Formally, define:

-   Treatment group: $$ T = \{ i \mid D_i(t) = 1 \} $$

-   Control group: $$ C = \{ j \mid D_j(t) = 0 \quad \text{and} \quad \exists  s>t : D_j(s) = 1 \} $$

Thus, both treatment and control groups are "eventual adopters," just at different times.

------------------------------------------------------------------------

**Proposition:**\
If $U_i$ is **time-invariant** and **affects both treatment assignment and outcomes**, then LA-PSM produces unbiased causal estimates under weaker assumptions than standard PSM.

1.  In regular PSM, matching on $X_i$ cannot adjust for $U_i$.
2.  In LA-PSM, by restricting controls to **future adopters**, we select units that share latent traits $U_i$ driving adoption.
3.  Thus, $U_i$ is balanced across treatment and control, eliminating bias from $U_i$.
4.  Remaining bias is due only to differences in $X_i$, which are adjusted for via matching.

------------------------------------------------------------------------

##### Practical Implementation Steps

1.  In **Static Look-Ahead PSM**, we:

-   Fix a single matching window at a given time $t$.
-   Define:
    -   Treatment Group: Individuals who receive treatment at time $t$.
    -   Control Group: Individuals who are not yet treated at $t$ but will receive treatment at a later time $t'>t$.
-   Estimate propensity scores based only on observed covariates $X$.
-   Match treated and (future-treated) control units based on their propensity scores.

This ensures treated and control individuals have similar observed $X$ and similar unobserved $U$, assuming $U$ is time-invariant.

How to implement Static LA-PSM:

1.  Identify individuals **treated now** (treatment group) and **future adopters** (control group).
2.  Estimate propensity scores $\mathbb{P}(D=1 \mid X)$.
3.  Match treated and control units using nearest-neighbor matching.
4.  Estimate treatment effects via a regression on the matched sample.

Important: 

-   Drop individuals who are never treated from the analysis.

-   Only compare current-treated and future-treated!

------------------------------------------------------------------------

2.  In **Dynamic Look-Ahead PSM**, we:

-   Allow time to move forward ($t=1,2,3,\dots$).
-   At each time $t$:
    -   Define treated individuals (those who receive treatment at $t$).
    -   Define control individuals (those untreated at $t$ but who will adopt later).
-   Repeat matching separately for each $t$.
-   Aggregate results across different $t$ periods to estimate the overall effect.

Dynamic LA-PSM is more flexible:

-   It updates control groups over time.

-   Better handles situations where treatment adoption spreads gradually over time.

How to implement Dynamic LA-PSM:

1.  For each time period $t$:
    -   Define treated-now = individuals treated at $t$.
    -   Define control = individuals untreated at $t$ but who adopt later.
2.  Estimate propensity scores $\mathbb{P}(D=1 \mid X)$ within that time window.
3.  Match treated and controls at each $t$ separately.
4.  Pool matched samples from all $t$ together.
5.  Estimate treatment effects using the pooled sample.

Important: 

-   Keep track of time --- matching is done separately each period.

-   Aggregate the treatment effect estimates carefully (either by averaging or pooling matched data).

------------------------------------------------------------------------

+--------------------+------------------------------------+--------------------------------------------+
| Feature            | Static Look-Ahead PSM              | Dynamic Look-Ahead PSM                     |
+:===================+:===================================+:===========================================+
| Matching Window    | Single, fixed window (e.g., $t=5$) | Rolling window at each time $t$            |
+--------------------+------------------------------------+--------------------------------------------+
| Treated Group      | Treated at $t$                     | Treated at $t$                             |
+--------------------+------------------------------------+--------------------------------------------+
| Control Group      | Future treated (after $t$)         | Future treated relative to each $t$        |
+--------------------+------------------------------------+--------------------------------------------+
| Updates over time? | No                                 | Yes                                        |
+--------------------+------------------------------------+--------------------------------------------+
| Suitable for       | Simple adoption settings           | Gradual adoption / time-sensitive settings |
+--------------------+------------------------------------+--------------------------------------------+

> **Challenge:**\
> Simulate your own dataset with hidden confounding and implement both Static and Dynamic Look-Ahead PSM.\
> Compare bias against Standard PSM and Randomized Assignment.

------------------------------------------------------------------------

### Mahalanobis Distance Matching {#sec-mahalanobis}

Where [propensity score matching](#sec-propensity-scores) compresses covariates into a single scalar, Mahalanobis Distance Matching takes the opposite tack: it works in the full multivariate covariate space and asks which control units are jointly close to each treated unit on every covariate at once. The trade-off is straightforward. Mahalanobis matching tends to deliver better joint covariate balance when it succeeds, because two units close in Mahalanobis distance are close on each variable individually; but as the number of covariates grows, "close on every variable at once" becomes scarce, and matches deteriorate. In the bias-variance language of the previous section, Mahalanobis matching pushes hard on bias reduction (assuming the matches succeed) and pays for it in sample retention as dimensionality grows.

Mahalanobis Distance Matching is a method for matching units in observational studies based on the **multivariate similarity** of covariates. Unlike propensity score matching, which reduces the covariate space to a single scalar, Mahalanobis distance operates in the full multivariate space. As a result, Mahalanobis matching can be interpreted as approximating a **fully blocked design**, where each treated unit is paired with a control unit that has nearly identical covariate values.

-   In the ideal case: $X_t = X_c$, implying a perfect match.
-   In practice, Mahalanobis distance allows for "near-exact" matches in high-dimensional space.

Because it preserves the multivariate structure of the covariates, Mahalanobis matching more faithfully emulates randomization within **covariate strata**, making it more robust to specification error than propensity score matching.

This method is particularly appealing when the number of covariates is relatively small and when these covariates are continuous and well-measured.

------------------------------------------------------------------------

Given a set of $p$ covariates $X_i \in \mathbb{R}^p$ for unit $i$, the **Mahalanobis distance** between a treated unit $t$ and a control unit $c$ is defined as:

$$
D_M(X_t, X_c) = \sqrt{(X_t - X_c)^\top S^{-1}(X_t - X_c)}
$$

where:

-   $X_t$ and $X_c$ are the $p$-dimensional vectors of covariates for treated and control units, respectively,
-   $S$ is the sample covariance matrix of the covariates across all units (treated and control),
-   $S^{-1}$ serves to standardize and decorrelate the covariate space, accounting for both scale and correlation among covariates.

------------------------------------------------------------------------

#### Why Not Use Euclidean Distance?

The Euclidean distance:

$$
D_E(X_t, X_c) = \sqrt{(X_t - X_c)^\top (X_t - X_c)}
$$

does **not adjust for different variances** among covariates or for **correlation** between them. Mahalanobis distance corrects for this by incorporating the inverse covariance matrix $S^{-1}$, effectively transforming the data to a space where the covariates are **standardized and orthogonal**.

This makes the Mahalanobis distance **scale-invariant**, i.e., invariant under affine transformations of the data, which is essential when matching on variables of different units (e.g., income in dollars and age in years).

------------------------------------------------------------------------

#### Mahalanobis Matching Algorithm

Let $\mathcal{T}$ be the set of treated units and $\mathcal{C}$ the set of control units. The procedure for Mahalanobis matching can be described as follows:

1.  **Compute the covariance matrix** $S$ of the covariates $X$ across all units.
2.  **For each treated unit** $i \in \mathcal{T}$, compute $D_M(X_i, X_j)$ for all $j \in \mathcal{C}$.
3.  **Match** treated unit $i$ to the control unit $j$ with the smallest Mahalanobis distance.
4.  Prune: 
    -   Control units not used in any match are discarded.
    -   Matches where $D_M > \delta$ (a caliper threshold) are also discarded.

A caliper $\delta$ is used to enforce a maximum allowable distance. That is, a match is made between $i$ and $j$ only if:

$$
D_M(X_i, X_j) < \delta
$$

This prevents poor-quality matches and helps ensure that treated and control units are meaningfully similar. If no match falls within the caliper for a given treated unit, that unit is left unmatched, and potentially discarded from the analysis.

------------------------------------------------------------------------

#### Properties

-   Scale-invariant: Standardizes variables using $S^{-1}$, ensuring that variables with large scales do not dominate the distance metric.
-   Correlation-adjusted: Accounts for linear relationships among covariates, which is critical in multivariate contexts.
-   Non-parametric: No model for treatment assignment is estimated; purely based on observed covariates.

------------------------------------------------------------------------

#### Limitations

-   Sensitive to multicollinearity: If the covariates are highly collinear, the covariance matrix $S$ may be nearly singular, making $S^{-1}$ unstable or non-invertible. Regularization techniques may be required.
-   Not suitable for high-dimensional covariate spaces: As $p$ increases, exact or even near-exact matches become harder to find. Dimensionality reduction techniques (e.g., PCA) may help.
-   Inefficient with categorical variables: Since Mahalanobis distance is based on continuous covariates and assumes multivariate normality (implicitly), it's less effective when most covariates are categorical or binary.

------------------------------------------------------------------------

#### Hybrid Approaches

Mahalanobis matching can be **combined with propensity scores** for improved performance:

-   Within propensity score calipers: Match using Mahalanobis distance only within groups of units that fall within a specified caliper of each other in propensity score space.
-   Stratified matching: Divide the sample into strata based on propensity scores and apply Mahalanobis matching within each stratum.

These hybrid methods aim to preserve the robustness of Mahalanobis matching while mitigating its weaknesses in high dimensions.

------------------------------------------------------------------------

#### Practical Considerations

-   Estimate $S$ from the pooled sample (treated + control) to ensure consistency.
-   Standardize all covariates before applying Mahalanobis distance to ensure comparability and numerical stability.
-   Use diagnostic plots (e.g., QQ plots or multivariate distance histograms) to assess match quality.

------------------------------------------------------------------------

### Coarsened Exact Matching (CEM) {#sec-cem}

Coarsened Exact Matching addresses a different question than [propensity score](#sec-propensity-scores) or [Mahalanobis](#sec-mahalanobis) matching. Rather than letting an algorithm decide what counts as "close," CEM asks the analyst to declare it: how finely should age be binned? Should education be split into four categories or eight? Once the analyst makes those substantive choices, exact matching within the resulting strata is straightforward, and the maximum imbalance is bounded by construction. This makes CEM transparent and tunable, with the trade-off being explicit. Coarser bins keep more of the sample but allow more residual imbalance within each stratum; finer bins guarantee tight balance but discard strata that lack both treated and control units. CEM is often the right choice when the covariate set is moderate, when the analyst has substantive intuition about meaningful cut points, and when a non-parametric procedure that does not lean on a treatment-assignment model is preferred.

Coarsened Exact Matching (CEM) is a **monotonic imbalance bounding** matching method designed to improve covariate balance between treated and control units in observational studies. CEM operates by **coarsening** continuous or high-cardinality covariates into discrete bins and then applying **exact matching** on this coarsened space. The result is a non-parametric method that prioritizes covariate balance and robustness over modeling assumptions.

Introduced and formalized in @iacus2012causal, CEM is well-suited for causal inference when covariates are noisy or when traditional modeling-based approaches (e.g., propensity scores) are unstable or hard to interpret.

------------------------------------------------------------------------

The central idea is to **discretize covariates** into meaningful bins (either automatically or manually), then sort observations into **strata** or subclasses based on the **unique combinations** of these binned covariate values. Exact matching is applied within each stratum.

Let $X_i \in \mathbb{R}^p$ be the $p$-dimensional covariate vector for unit $i$. Define $C(X_i)$ to be a coarsened version of $X_i$, where:

-   Continuous variables are binned into intervals (e.g., age 20--29, 30--39, etc.)
-   Categorical variables may be grouped (e.g., Likert scales aggregated into fewer categories)

Then, matching proceeds as follows:

1.  Temporarily coarsen the covariate space $X \to C(X)$.
2.  Sort observations into strata defined by unique values of $C(X)$.
3.  Prune any stratum that contains only treated or only control units.
4.  Retain all original (uncoarsened) units in matched strata for analysis.

This process produces a matched sample in which each stratum includes at least one treated and one control unit, improving internal validity by design.

As with other matching methods, CEM requires the **ignorability assumption** (also known as unconfoundedness or selection on observables):

$$ (Y_i(0), Y_i(1)) \perp T_i \mid X_i $$

Under this assumption and sufficient overlap in the coarsened strata, CEM yields unbiased estimates of causal effects within the matched sample.

------------------------------------------------------------------------

#### Mathematical Properties

1.  **Monotonic Imbalance Bounding**

CEM is a Monotonic Imbalance Bounding method, meaning that the user can pre-specify the maximum level of imbalance allowed on each covariate. The imbalance measure is guaranteed to be weakly decreasing as coarsening becomes finer. This provides:

-   Transparency: The imbalance tradeoff is determined *ex ante* by the user.
-   Control: Users can make a direct decision about how much imbalance is tolerable.

Formally, let $\mathcal{L}(C(X))$ denote a measure of imbalance under coarsened covariates. Then, for any refinements of $C(X)$:

$$
\text{if } C_1(X) \preceq C_2(X), \text{ then } \mathcal{L}(C_1(X)) \leq \mathcal{L}(C_2(X))
$$

That is, finer coarsenings (more bins) cannot increase imbalance.

2.  **Congruence Principle**

CEM respects the congruence principle, which asserts that analysis should not be more precise than the data allow. By coarsening, CEM protects against overfitting and artificial precision, especially when measurement error is present.

3.  **Robustness**

-   Robust to measurement error: Discretization makes matching less sensitive to noise in covariates.
-   Works with missing data: The `cem` R package supports partial handling of missingness.
-   Multiple imputation: CEM can be applied within imputation routines to preserve matching structure across imputed datasets.
-   Supports multi-valued treatments: Matching can be extended to treatments beyond binary $T \in {0, 1}$.

------------------------------------------------------------------------

1.  Load and Prepare Data


``` r
library(cem)
data(LeLonde)

# Remove missing values
Le <- na.omit(LeLonde)

# Treatment and control indices
tr <- which(Le$treated == 1)
ct <- which(Le$treated == 0)
ntr <- length(tr)
nct <- length(ct)

# Unadjusted bias (naïve difference in means)
mean(Le$re78[tr]) - mean(Le$re78[ct])
#> [1] 759.0479
```

2.  Define Pre-Treatment Covariates


``` r
vars <- c(
    "age", "education", "black", "married", "nodegree",
    "re74", "re75", "hispanic", "u74", "u75", "q1"
)

# Pre-treatment imbalance
imbalance(group = Le$treated, data = Le[vars]) # L1 imbalance = 0.902
#> 
#> Multivariate Imbalance Measure: L1=0.902
#> Percentage of local common support: LCS=5.8%
#> 
#> Univariate Imbalance Measures:
#> 
#>               statistic   type           L1 min 25%      50%       75%
#> age        -0.252373042 (diff) 5.102041e-03   0   0   0.0000   -1.0000
#> education   0.153634710 (diff) 8.463851e-02   1   0   1.0000    1.0000
#> black      -0.010322734 (diff) 1.032273e-02   0   0   0.0000    0.0000
#> married    -0.009551495 (diff) 9.551495e-03   0   0   0.0000    0.0000
#> nodegree   -0.081217371 (diff) 8.121737e-02   0  -1   0.0000    0.0000
#> re74      -18.160446880 (diff) 5.551115e-17   0   0 284.0715  806.3452
#> re75      101.501761679 (diff) 5.551115e-17   0   0 485.6310 1238.4114
#> hispanic   -0.010144756 (diff) 1.014476e-02   0   0   0.0000    0.0000
#> u74        -0.045582186 (diff) 4.558219e-02   0   0   0.0000    0.0000
#> u75        -0.065555292 (diff) 6.555529e-02   0   0   0.0000    0.0000
#> q1          7.494021189 (Chi2) 1.067078e-01  NA  NA       NA        NA
#>                  max
#> age          -6.0000
#> education     1.0000
#> black         0.0000
#> married       0.0000
#> nodegree      0.0000
#> re74      -2139.0195
#> re75        490.3945
#> hispanic      0.0000
#> u74           0.0000
#> u75           0.0000
#> q1                NA
```

3.  Automatically Coarsen and Match


``` r
mat <- cem(
    treatment = "treated",
    data = Le,
    drop = "re78",     # outcome variable
    keep.all = TRUE    # retain unmatched units in output
)
#> 
#> Using 'treated'='1' as baseline group
mat
#>            G0  G1
#> All       392 258
#> Matched    95  84
#> Unmatched 297 174
```

-   `mat$w` contains **weights** for matched units.

-   Summary of matched strata and units retained.

4.  Manual Coarsening (User-Controlled)

Users may coarsen variables explicitly based on theoretical knowledge or data distribution.

Example: Grouped Categorical and Binned Continuous Covariates


``` r
# Inspect levels for grouping
levels(Le$q1)
#> [1] "agree"             "disagree"          "neutral"          
#> [4] "no opinion"        "strongly agree"    "strongly disagree"

# Group Likert responses
q1.grp <- list(
    c("strongly agree", "agree"),
    c("neutral", "no opinion"),
    c("strongly disagree", "disagree")
)

# Custom cutpoints for education
table(Le$education)
#> 
#>   3   4   5   6   7   8   9  10  11  12  13  14  15 
#>   1   5   4   6  12  55 106 146 173 113  19   9   1
educut <- c(0, 6.5, 8.5, 12.5, 17)

# Run CEM with manual coarsening
mat1 <- cem(
    treatment = "treated",
    data = Le,
    drop = "re78",
    cutpoints = list(education = educut),
    grouping = list(q1 = q1.grp)
)
#> 
#> Using 'treated'='1' as baseline group
mat1
#>            G0  G1
#> All       392 258
#> Matched   158 115
#> Unmatched 234 143
```

This allows for **domain-specific discretion** and enhances interpretability.

#### Progressive Coarsening

CEM supports **progressive coarsening**, where matching is attempted with fine coarsening first. If insufficient matches are found, coarsening is gradually relaxed until a suitable number of matched units is retained.

This strategy balances:

-   **Precision** (finer bins)

-   **Sample size retention** (coarser bins)

#### Summary

Strengths

-   Transparent and tunable matching procedure

-   Non-parametric: Does not require estimation of a treatment assignment model

-   Robust to measurement error and model misspecification

-   Supports multi-valued treatments and partial missingness

-   Theoretically grounded via monotonic imbalance bounding

Limitations

-   Loss of information due to coarsening

-   Arbitrary binning may influence results if not theoretically motivated

-   Not designed for high-dimensional settings without careful variable selection

-   Cannot account for unobserved confounding

Coarsened Exact Matching offers a powerful, transparent, and theoretically principled method for preprocessing observational data before estimating causal effects. It overcomes several limitations of traditional matching approaches, particularly propensity score matching, by giving researchers direct control over the quality of matches and by anchoring inference in empirical balance rather than model-dependent estimates.

------------------------------------------------------------------------

### Genetic Matching {#sec-genetic-matching}

Genetic Matching extends the algorithms above by recognizing that the choice of distance metric is itself a researcher degree of freedom. Standard [Mahalanobis distance](#sec-mahalanobis) implicitly assigns weights via the inverse covariance matrix, which is a sensible default but is not optimized for balance on the specific covariates the analyst cares about. Genetic Matching instead searches over the space of weight matrices to find the one that best balances a user-specified set of covariates and transformations, using a stochastic optimizer modeled on natural selection. The method shines when conventional distance metrics leave residual imbalance on substantively important covariates, and when the analyst is willing to pay the computational cost of a population-based search. The trade-offs are familiar: relative to nearest-neighbor matching on a propensity score, Genetic Matching usually delivers better balance and lower bias, at the price of longer run times and an estimand that depends on which covariates the analyst chose to balance.

Genetic Matching (GM) is a generalization of [propensity score](#sec-propensity-scores) and [Mahalanobis distance matching](#sec-mahalanobis), leveraging a search algorithm inspired by evolutionary biology to optimize covariate balance. Unlike classical matching techniques, which rely on pre-specified distance metrics, Genetic Matching *learns* an optimal distance metric through an iterative process aimed at minimizing imbalance between treated and control groups.

Genetic Matching was introduced by @diamond2013genetic and is designed to improve balance on observed covariates by selecting optimal weights for each covariate. The core idea is to define a generalized Mahalanobis distance metric, where the weights used in the distance calculation are chosen by a **genetic search algorithm**. This process adaptively explores the space of possible weighting matrices and evolves toward a set of weights that result in optimal covariate balance across treatment groups.

The method combines two components:

-   [**Propensity Score Matching**](#sec-propensity-scores): Balances on the estimated probability of treatment assignment given covariates.
-   [**Mahalanobis Distance Matching**](#sec-mahalanobis): Accounts for correlations among covariates and ensures geometric proximity in multivariate space.

The Genetic Matching approach treats the selection of covariate weights as an optimization problem, solved using techniques inspired by natural selection: mutation, crossover, and survival of the fittest.

Compared to traditional matching methods such as:

-   Nearest Neighbor Matching: May result in poor balance in high-dimensional or imbalanced datasets.
-   Full Matching: More efficient but not always feasible with severe imbalance.

Genetic Matching is particularly powerful in situations where traditional distance metrics are insufficient to achieve balance. It adaptively searches the weight space to ensure better covariate overlap, which is crucial for unbiased causal inference.

In empirical settings, this often results in superior balance on observed covariates and more credible estimates of treatment effects.

------------------------------------------------------------------------

Let:

-   $T_i \in {0, 1}$ be the binary treatment indicator for unit $i$.

-   $\mathbf{X}_i \in \mathbb{R}^p$ be the covariate vector for unit $i$, where $p$ is the number of observed covariates.

-   $\mathbf{W}$ be a $p \times p$ positive definite weight matrix optimized by the algorithm.

The **generalized Mahalanobis distance** between unit $i$ and $j$ is defined as:

$$
D_{ij} = \sqrt{(\mathbf{X}_i - \mathbf{X}_j)^\top \mathbf{W} (\mathbf{X}_i - \mathbf{X}_j)}
$$

Here, $\mathbf{W}$ is not necessarily the inverse of the covariance matrix (as in standard Mahalanobis distance), but a data-driven weighting matrix found by Genetic Matching to optimize covariate balance.

The genetic algorithm optimizes the weights in $\mathbf{W}$ to minimize a global imbalance metric $\mathcal{B}$ across all covariates. This can involve different balance criteria:

-   Paired $t$-tests for continuous or dichotomous variables
-   Kolmogorov--Smirnov (K--S) test statistics for distributional similarity
-   Standardized mean differences

Let $b_k$ denote the balance metric for the $k$th covariate. Then the total imbalance could be:

$$
\mathcal{B} = \sum_{k=1}^{p} w_k b_k^2
$$

The optimization objective becomes finding $\mathbf{W}$ that minimizes $\mathcal{B}$.

------------------------------------------------------------------------

Genetic Matching is implemented in the `Matching` package in R via the `GenMatch()` function. This function:

-   Accepts a treatment indicator (`Tr`) and a matrix of covariates (`X`).
-   Optionally takes a **Balance Matrix** (`BalanceMatrix`) that specifies which variables should be balanced.
-   Uses a genetic search to find the weight matrix that best achieves balance on the specified variables.

Balance is assessed via:

-   Paired $t$-tests for dichotomous or continuous variables
-   Kolmogorov-Smirnov (K--S) tests for distributional differences

Matching can be performed:

-   **With or without replacement** (with replacement often improves match quality at the cost of increased variance).
-   For different estimands: [Average Treatment Effect], [Average Treatment Effect on the Treated], or [Average Treatment Effect on the Control].

------------------------------------------------------------------------


``` r
library(Matching)
data(lalonde, package = "MatchIt")
attach(lalonde)

# Define the covariates to match on
X <- cbind(age, educ, black, hisp, married, nodegr, u74, u75, re75, re74)

# Define the covariates to balance on
BalanceMat <- cbind(
  age, educ, black, hisp, married, nodegr,
  u74, u75, re75, re74,
  I(re74 * re75) # Include interaction term
)

# Genetic Matching to optimize covariate balance
# Note: pop.size = 16 is too small for real applications
genout <- GenMatch(
  Tr = treat,
  X = X,
  BalanceMatrix = BalanceMat,
  estimand = "ATE",
  M = 1,
  pop.size = 16,
  max.generations = 10,
  wait.generations = 1
)

# Define the outcome variable
Y <- re78 / 1000

# Perform matching with the optimized weights
mout <- Match(
  Y = Y,
  Tr = treat,
  X = X,
  estimand = "ATE",
  Weight.matrix = genout
)

# Summarize the treatment effect estimates
summary(mout)

# Assess post-matching balance
mb <- MatchBalance(
  treat ~ age + educ + black + hisp + married + nodegr +
    u74 + u75 + re75 + re74 + I(re74 * re75),
  match.out = mout,
  nboots = 500
)
```

#### Extensions and Considerations

-   The method can be extended to multinomial treatments (via generalized entropy balancing).

-   For longitudinal or panel data, entropy balancing can be adapted to match across time points.

-   One should avoid balancing on post-treatment variables, which would bias estimates.

Entropy balancing fits naturally into modern workflows for causal inference, and is especially valuable when researchers want to prioritize design over modeling.

------------------------------------------------------------------------

### Entropy Balancing

Entropy balancing departs from the matching paradigm in an instructive way. Rather than picking which control units to keep and which to drop, it keeps every control unit and instead chooses weights that, by construction, force the reweighted control group to match the treated group on a set of pre-specified moments (means, variances, and so on). Because no unit is discarded, the method is attractive when treated units are scarce or when sample size cannot be sacrificed. Because balance is achieved exactly on the chosen moments, no iterative diagnostic loop is needed. The trade-offs run in two directions. Entropy balancing avoids the model dependence of [propensity score](#sec-propensity-scores) estimation but transfers that dependence to the analyst's choice of which moments to balance: if a relevant interaction or non-linear function of covariates is omitted from the constraints, residual imbalance there can still bias the effect estimate. Extreme weights, while less common than under inverse probability weighting, can still inflate variance.

Entropy balancing is a data preprocessing technique for causal inference in observational studies. Introduced by @hainmueller2012entropy, it is particularly useful in settings with binary treatments where covariate imbalance between treated and control groups threatens the validity of causal estimates.

The method applies a **maximum entropy reweighting** scheme to assign **unit weights** to control group observations such that the **reweighted covariate distribution** in the control group matches the **sample moments** (e.g., means, variances) of the treated group.

Entropy balancing directly targets covariate balance and avoids the trial-and-error of iterative propensity score or matching methods. It also reduces reliance on outcome models by ensuring pre-treatment covariates are aligned between groups.

-   Goal: Create a set of weights for the control group such that its covariate distribution matches that of the treated group.
-   Approach: Solve a constrained optimization problem that minimizes information loss (measured via Shannon entropy) subject to **balance constraints** on covariates.

Entropy balancing:

-   Achieves exact balance on the specified covariate moments (e.g., means, variances).

-   Produces unique, data-driven weights without the need for iterative matching.

-   Is compatible with any outcome model in the second stage (e.g., weighted regression, IPW, DID, etc.).

#### Advantages of Entropy Balancing

-   Exact balance: Guarantees moment balance without iterative diagnostics.
-   Flexibility: Can balance on higher moments, interactions, or non-linear transformations.
-   Model independence: Reduces reliance on correct specification of the outcome model.
-   Stability: The optimization procedure is smooth and produces stable, interpretable weights.

Entropy balancing is especially useful in high-dimensional settings or when working with small treated groups, where matching can perform poorly or fail entirely.

------------------------------------------------------------------------

Suppose we have $n_1$ treated units and $n_0$ control units, with covariates $\mathbf{X}_i \in \mathbb{R}^p$ for each unit $i$. Let $T_i \in {0,1}$ be the treatment indicator.

Let the treatment group's sample moment vector be:

$$
\bar{\mathbf{X}}_T = \frac{1}{n_1} \sum_{i: T_i=1} \mathbf{X}_i
$$

We seek a set of weights $\{w_i\}_{i: T_i=0}$ for control units such that:

1.  **Covariate balancing constraints** (e.g., mean balance):

$$
\sum_{i: T_i=0} w_i \mathbf{X}_i = \bar{\mathbf{X}}_T
$$

2.  **Minimum entropy divergence** from uniform weights:

We minimize the Kullback-Leibler divergence between the new weights $w_i$ and uniform base weights $w_i^{(0)} = \frac{1}{n_0}$:

$$
\min_{\{w_i\}} \sum_{i: T_i=0} w_i \log \left( \frac{w_i}{w_i^{(0)}} \right)
$$

subject to:

-   $\sum w_i = 1$ (weights must sum to 1)
-   $\sum w_i \mathbf{X}_i = \bar{\mathbf{X}}_T$
-   (Optionally) higher-order moments, e.g., variances, skewness, etc.

This is a **convex optimization problem**, ensuring a unique global solution.

------------------------------------------------------------------------


``` r
# Load package
library(ebal)

# Simulate data
set.seed(123)
n_treat <- 50
n_control <- 100

# Covariates
X_treat <- matrix(rnorm(n_treat * 3, mean = 1), ncol = 3)
X_control <- matrix(rnorm(n_control * 3, mean = 0), ncol = 3)
X <- rbind(X_control, X_treat)  # Order: control first, then treated

# Treatment vector: 0 = control, 1 = treated
treatment <- c(rep(0, n_control), rep(1, n_treat))

# Apply entropy balancing
eb_out <- ebalance(Treatment = treatment, X = X)

# Simulate outcome variable
Y_control <- rnorm(n_control, mean = 1)
Y_treat <- rnorm(n_treat, mean = 3)
Y <- c(Y_control, Y_treat)

# Construct weights: treated get weight 1, control get weights from ebalance
weights <- c(eb_out$w, rep(1, n_treat))

# Estimate ATE using weighted linear regression
df <- data.frame(Y = Y, treat = treatment, weights = weights)
fit <- lm(Y ~ treat, data = df, weights = weights)
summary(fit)
```

------------------------------------------------------------------------

### Matching for High-Dimensional Data

As the dimensionality of covariates increases, traditional matching methods (such as nearest neighbor matching using propensity scores or Mahalanobis distance) become increasingly unreliable. This issue, often referred to as the **curse of dimensionality**, undermines the ability to find good matches because distances between points become less informative in high-dimensional space.

In high-dimensional settings, where the number of covariates is large (potentially exceeding the number of observations), careful preprocessing is necessary to reduce dimensionality before matching can be effectively applied. The following approaches are commonly used to mitigate these challenges by reducing the feature space while preserving meaningful structure relevant to treatment assignment and outcomes.

#### Dimensionality Reduction Techniques

A variety of dimensionality reduction techniques can be employed prior to matching. The methods below cluster into several conceptual families: supervised approaches (such as Lasso and penalized logistic regression) that select or shrink covariates using their relationship to treatment or outcome; unsupervised linear projections (PCA and random projection) that compress the covariate space without reference to the treatment; locality-aware methods (LPP) that try to preserve neighborhood structure; and nonlinear representation learners (autoencoders and joint causal-representation models) that capture complex covariate geometry at the cost of interpretability. The bias-variance trade-off here mirrors the rest of this chapter. Aggressive compression yields tractable matching but risks discarding information that confounds the treatment-outcome relationship; light compression preserves information but leaves the curse of dimensionality largely intact. As a rule of thumb, supervised sparse methods are sensible defaults when interpretability matters, while nonlinear methods earn their keep only when covariate relationships are genuinely non-linear and the sample is large.

Each method has distinct assumptions and use cases:

-   **Lasso Regression (Least Absolute Shrinkage and Selection Operator)**\
    Lasso imposes an $L_1$ penalty on the regression coefficients, effectively shrinking some coefficients to zero. This property is particularly useful in high-dimensional settings, as it performs both regularization and variable selection.\
    Lasso can be used to identify a smaller set of covariates that are most predictive of treatment assignment or outcome, which can then be used in matching procedures [@gordon2019comparison].

-   **Penalized Logistic Regression**\
    When the treatment is binary, penalized logistic regression (e.g., using an $L_1$ or $L_2$ penalty) can be employed to estimate propensity scores while avoiding overfitting in high-dimensional spaces. These penalized models provide more stable estimates of the propensity score, which is essential for reliable matching [@eckles2021bias].

-   **Principal Component Analysis (PCA)**\
    PCA is an unsupervised linear transformation that projects the original features into a lower-dimensional space by retaining the directions of maximum variance. While PCA does not consider treatment assignment directly, it is effective for denoising and compressing data, particularly when covariates are highly correlated.\
    The principal components can then be used as inputs to standard matching methods.

-   **Locality Preserving Projections (LPP)**\
    LPP is a linear dimensionality reduction technique that, unlike PCA, preserves local neighborhood structures. It constructs a similarity graph and projects data into a lower-dimensional space such that nearby points remain close. This locality-preserving property is beneficial for matching, as it helps maintain the integrity of local relationships within the data [@li2016matching].

-   **Random Projection**\
    Random projection reduces dimensionality by projecting data onto a lower-dimensional subspace using a random matrix. It is computationally efficient and has theoretical guarantees (e.g., Johnson--Lindenstrauss lemma) that distances between points are approximately preserved. This makes it a viable option for extremely high-dimensional data where exact structure preservation is less critical.

-   **Autoencoders**\
    Autoencoders are neural network architectures designed to learn efficient, nonlinear representations (encodings) of data. An autoencoder consists of an encoder that compresses the input and a decoder that attempts to reconstruct the original input from this compressed representation.\
    Autoencoders are particularly effective in capturing complex, nonlinear relationships among features, which may be missed by linear methods like PCA. The latent representation obtained from the encoder can then be used for matching [@ramachandra2018deep].

#### Joint Dimensionality Reduction and Distribution Balancing

Rather than performing dimensionality reduction and matching as separate steps, an emerging strategy is to jointly learn representations that simultaneously:

1.  Reduce dimensionality, and
2.  Balance the covariate distributions between treated and control groups.

This is exemplified by *representation learning for causal inference* approaches. One such method, proposed by @yao2018representation, integrates neural networks with matching objectives to learn latent representations of covariates that are both low-dimensional and balanced. These representations are optimized such that the distributions of treated and control units in the latent space are similar (e.g., using maximum mean discrepancy or other balancing metrics), thereby improving the quality of matches and robustness of treatment effect estimates.

#### Summary of Approaches

+-------------------------------+-----------------+-------------+-------------------------------------------------+--------------------------+
| Method                        | Type            | Supervised? | Key Feature                                     | Reference                |
+===============================+=================+=============+=================================================+==========================+
| Lasso                         | Linear, sparse  | Yes         | Variable selection via $L_1$ penalty            | [@gordon2019comparison]  |
+-------------------------------+-----------------+-------------+-------------------------------------------------+--------------------------+
| Penalized logistic regression | Linear          | Yes         | Regularized propensity score estimation         | [@eckles2021bias]        |
+-------------------------------+-----------------+-------------+-------------------------------------------------+--------------------------+
| PCA                           | Linear          | No          | Projects onto directions of maximal variance    |                          |
+-------------------------------+-----------------+-------------+-------------------------------------------------+--------------------------+
| LPP                           | Linear          | No          | Preserves local neighborhood structure          | [@li2016matching]        |
+-------------------------------+-----------------+-------------+-------------------------------------------------+--------------------------+
| Random projection             | Linear (random) | No          | Fast and preserves pairwise distances           |                          |
+-------------------------------+-----------------+-------------+-------------------------------------------------+--------------------------+
| Autoencoders                  | Nonlinear       | Yes         | Learns nonlinear representations                | [@ramachandra2018deep]   |
+-------------------------------+-----------------+-------------+-------------------------------------------------+--------------------------+
| Joint representation learning | Nonlinear       | Yes         | Learns balanced low-dimensional representations | [@yao2018representation] |
+-------------------------------+-----------------+-------------+-------------------------------------------------+--------------------------+

#### Practical Considerations

-   Model selection and validation: When applying dimensionality reduction prior to matching, one must ensure that the reduced representation still contains sufficient information for confounding adjustment. Cross-validation and balance metrics (e.g., standardized mean differences) should be used to assess the adequacy of the transformation.

-   Interpretability: While methods like PCA or autoencoders can be effective, they may obscure the interpretability of matches since the transformed features may not correspond to original covariates. Sparse methods like Lasso retain interpretability by selecting original covariates.

-   Computational efficiency: Techniques such as random projection or penalized regression are computationally efficient and scalable to large datasets, while autoencoders and joint learning approaches may require more extensive training and hyperparameter tuning.

------------------------------------------------------------------------

### Matching for Multiple Treatments

In many applied settings, researchers face the challenge of estimating causal effects for **more than two treatment levels**. For example, a marketing campaign may have three variants (e.g., control, light exposure, heavy exposure), or a policy evaluation may involve multiple interventions. Standard binary treatment matching methods fall short in these cases, necessitating methodological extensions.

Suppose a dataset includes $T$ distinct treatment groups: $\mathcal{T} = \{0, 1, \ldots, T-1\}$. Here, $T = 0$ typically denotes the control group, and $T \in \{1, \ldots, T-1\}$ are active treatments. The goal is to estimate the [Average Treatment Effect] or **Pairwise Treatment Effects**, such as $\text{ATE}_{j,k} = \mathbb{E}[Y(j) - Y(k)]$ for any $j, k \in \mathcal{T}$.

Key challenges in this setting include:

-   Ensuring common support across multiple groups
-   Adjusting for confounders in a balanced way across all treatment pairs
-   Managing covariate imbalance and dimensionality as the number of treatments increases

------------------------------------------------------------------------

#### Matching Approaches for Multiple Treatments

Several strategies exist for matching in the presence of multiple treatments, and they differ in how they handle the explosion of pairwise comparisons. Generalized propensity score methods extend the binary scalar logic to a vector of treatment probabilities, which is parsimonious but more demanding on the [overlap](#sec-overlap-positivity-assumption) assumption: every treatment level must have a positive probability of being received at every covariate value. Pairwise matching ignores the multinomial structure and runs a sequence of binary matches, which is simple to implement but can yield matched samples that are inconsistent across pairs. Direct multivariate matching using the full GPS vector preserves the joint structure but is harder to balance. The right choice depends on the number of treatment levels, the available sample within each, and whether the estimand of interest is a single contrast or the full set of pairwise effects.

1.  **Generalized Propensity Scores (GPS)**

The generalized propensity score is defined as the conditional probability of receiving each treatment level given covariates:

$$
e_t(X) = \mathbb{P}(T = t \mid X), \quad \text{for } t = 0, 1, ..., T-1
$$

Estimation is typically done using **multinomial logistic regression**. Once GPS scores are estimated, matching can proceed via:

-   One-vs-all: For each treatment level, match treated units to all others combined.
-   Pairwise matching: Conduct separate pairwise comparisons between all treatment levels.
-   Full matching: Attempt to construct a global matched sample covering all treatments, using the GPS vector.

[@mccaffrey2013tutorial] provides a comprehensive overview of GPS-based methods, including their implementation in the `twang` package.

2.  **Covariate Balancing Propensity Scores (CBPS)** for Multiple Treatments

[@lopez2017estimation] extend CBPS to the multinomial case. Rather than merely estimating GPS, CBPS directly optimizes covariate balance across treatment groups while estimating GPS parameters. This dual-objective estimation leads to more robust causal estimates in finite samples.

3.  **Kernel or Distance-Based Matching on Multinomial Scores**

Instead of reducing GPS to scalar scores, matching can be performed using the full vector of propensity scores, using distance metrics such as Euclidean or Mahalanobis distance in the GPS space. This approach aligns with @zhao2021propensity, who also consider **continuous treatments** using **Generalized Propensity Score Density Estimation**.

------------------------------------------------------------------------

#### Matching with Multiple Treatments Using MatchIt and Alternatives

While the `MatchIt` package in R was originally developed with binary treatment settings in mind, it can be adapted to handle multiple treatment groups through pairwise matching and careful design. However, this approach requires manual data preparation and a clear understanding of the causal estimands of interest, such as the ATT, ATC, or ATE.

1.  Pairwise Matching with a Shared Control Group

To estimate the effect of multiple treatments compared to a shared control group, one straightforward method is to perform separate pairwise matchings between the control group and each treatment group:


``` r
# Load required libraries
library(MatchIt)
library(cobalt)  # For balance checking

# Example dataset: Simulated
set.seed(123)
n <- 400
df <- data.frame(
  treat = factor(sample(c("control", "A", "B", "C"), n, replace = TRUE)),
  cov1 = rnorm(n),
  cov2 = runif(n),
  cov3 = rbinom(n, 1, 0.5),
  outcome = rnorm(n)
)

# Define treatment levels
treatment_levels <- c("A", "B", "C")
control_level <- "control"

# Initialize weight column
df$match.weights <- 0

# Perform pairwise matching of each treatment group vs control
for (treat in treatment_levels) {
  # Subset to control and current treatment
  subset_df <- df[df$treat %in% c(treat, control_level), ]
  
  # Create a binary treatment variable: 1 for current treatment, 0 for control
  subset_df$treat_binary <- as.numeric(subset_df$treat == treat)

  # Run matching
  m.out <- matchit(treat_binary ~ cov1 + cov2 + cov3,
                   data = subset_df, method = "nearest")

  # Assign weights back to the original dataset
  matched_units <- names(m.out$weights[m.out$weights > 0])
  df[matched_units, "match.weights"] <- m.out$weights[matched_units]
}

# Check covariate balance
bal.tab(treat ~ cov1 + cov2 + cov3, data = df,
        weights = df$match.weights, method = "matching")
#> Balance summary across all treatment pairs
#>         Type Max.Diff.Adj
#> cov1 Contin.       0.2200
#> cov2 Contin.       0.1416
#> cov3  Binary       0.0412
#> 
#> Sample sizes
#>           A  B  C control
#> All     104 97 85     114
#> Matched 104 97 85     114

# Estimate treatment effects using weighted regression
model <- glm(outcome ~ relevel(treat, ref = "control"),
             data = df[df$match.weights > 0, ],
             weights = match.weights)

summary(model)
#> 
#> Call:
#> glm(formula = outcome ~ relevel(treat, ref = "control"), data = df[df$match.weights > 
#>     0, ], weights = match.weights)
#> 
#> Coefficients:
#>                                  Estimate Std. Error t value Pr(>|t|)
#> (Intercept)                      -0.03758    0.09349  -0.402    0.688
#> relevel(treat, ref = "control")A  0.08584    0.13536   0.634    0.526
#> relevel(treat, ref = "control")B  0.06877    0.13789   0.499    0.618
#> relevel(treat, ref = "control")C  0.03806    0.14305   0.266    0.790
#> 
#> (Dispersion parameter for gaussian family taken to be 0.9964463)
#> 
#>     Null deviance: 395.05  on 399  degrees of freedom
#> Residual deviance: 394.59  on 396  degrees of freedom
#> AIC: 1139.7
#> 
#> Number of Fisher Scoring iterations: 2
```

This approach estimates the **ATT for each treatment group** relative to the control. That is, for each treated group (e.g., A, B, or C), we ask: *what would the outcome have been if those who received treatment had instead received the control?*

2.  Estimating the ATC Using `MatchIt`

In some cases, we may wish to estimate the ATC, i.e., what would have happened to the control group if they had received each of the treatments. To do this, we match control units to each treated group, reversing the focal population. Practically, this means keeping the control group intact and separately matching each treatment group to it.


``` r
# Load required libraries
library(MatchIt)
library(cobalt)

# Simulate example data
set.seed(456)
n <- 400
df <- data.frame(
  treat = factor(sample(c("control", "A", "B", "C"), n, replace = TRUE)),
  cov1 = rnorm(n),
  cov2 = runif(n),
  cov3 = rbinom(n, 1, 0.5),
  outcome = rnorm(n)
)

# Define treatment levels
treatment_levels <- c("A", "B", "C")
control_level <- "control"

# Initialize weight column
df$match.weights <- 0

# Estimate ATC by matching treated units to the control group
for (treat in treatment_levels) {
  # Subset to current treatment and control
  subset_df <- df[df$treat %in% c(treat, control_level), ]
  
  # Binary treatment variable: 0 for treatment group, 1 for control
  # This reverses the focus, targeting the control as the treated group
  subset_df$treat_binary <- as.numeric(subset_df$treat == control_level)

  # Perform matching
  m.out <- matchit(treat_binary ~ cov1 + cov2 + cov3,
                   data = subset_df, method = "nearest")
  
  # Extract matched unit IDs
  matched_ids <- names(m.out$weights[m.out$weights > 0])
  
  # Assign weights back to original dataset
  df[matched_ids, "match.weights"] <- m.out$weights[matched_ids]
}

# Check balance (optional)
bal.tab(treat ~ cov1 + cov2 + cov3, data = df,
        weights = df$match.weights, method = "matching")
#> Balance summary across all treatment pairs
#>         Type Max.Diff.Adj
#> cov1 Contin.       0.1229
#> cov2 Contin.       0.2695
#> cov3  Binary       0.1985
#> 
#> Sample sizes
#>            A  B  C control
#> All       99 90 98     113
#> Matched   99 90 98     111
#> Unmatched  0  0  0       2

# Estimate ATC via weighted regression
model <- glm(outcome ~ relevel(treat, ref = "control"),
             data = df[df$match.weights > 0, ],
             weights = match.weights)

summary(model)
#> 
#> Call:
#> glm(formula = outcome ~ relevel(treat, ref = "control"), data = df[df$match.weights > 
#>     0, ], weights = match.weights)
#> 
#> Coefficients:
#>                                  Estimate Std. Error t value Pr(>|t|)
#> (Intercept)                       0.06826    0.09176   0.744    0.457
#> relevel(treat, ref = "control")A -0.05103    0.13364  -0.382    0.703
#> relevel(treat, ref = "control")B  0.08667    0.13713   0.632    0.528
#> relevel(treat, ref = "control")C  0.09488    0.13400   0.708    0.479
#> 
#> (Dispersion parameter for gaussian family taken to be 0.9346181)
#> 
#>     Null deviance: 369.69  on 397  degrees of freedom
#> Residual deviance: 368.24  on 394  degrees of freedom
#> AIC: 1108.5
#> 
#> Number of Fisher Scoring iterations: 2
```

3.  Regression and Effect Estimation

After constructing the matched dataset, one can use regression on the matched sample to estimate the treatment effects:


``` r
# Subset the data first
matched_data <- df[df$match.weights > 0, ]

# Then fit the weighted regression using the weights from the subset
model <- glm(outcome ~ relevel(treat, ref = "control"),
             data = matched_data,
             weights = matched_data$match.weights)

summary(model)
#> 
#> Call:
#> glm(formula = outcome ~ relevel(treat, ref = "control"), data = matched_data, 
#>     weights = matched_data$match.weights)
#> 
#> Coefficients:
#>                                  Estimate Std. Error t value Pr(>|t|)
#> (Intercept)                       0.06826    0.09176   0.744    0.457
#> relevel(treat, ref = "control")A -0.05103    0.13364  -0.382    0.703
#> relevel(treat, ref = "control")B  0.08667    0.13713   0.632    0.528
#> relevel(treat, ref = "control")C  0.09488    0.13400   0.708    0.479
#> 
#> (Dispersion parameter for gaussian family taken to be 0.9346181)
#> 
#>     Null deviance: 369.69  on 397  degrees of freedom
#> Residual deviance: 368.24  on 394  degrees of freedom
#> AIC: 1108.5
#> 
#> Number of Fisher Scoring iterations: 2
```

Caveats and Limitations

-   Matching one treatment group at a time does not preserve global covariate balance across all treatment groups, only pairwise balance with the control.

-   Overlap (common support) assumptions should be verified individually for each pairwise comparison.

-   Matched samples may vary for each comparison, complicating aggregate inference or joint hypothesis testing.

------------------------------------------------------------------------

#### Alternative: Weighting for Multiple Treatments with `WeightIt`

Matching can be cumbersome in multiple-treatment settings. Weighting approaches offer a more seamless framework, especially when estimating ATE or ATC across all treatment levels simultaneously. The `WeightIt` package extends propensity score weighting to multi-treatment scenarios with strong support for diagnostics and flexible estimation.


``` r
# Load required libraries
library(WeightIt)
library(cobalt)

# Simulate example data
set.seed(789)
n <- 400
df <- data.frame(
  treat = factor(sample(c("control", "A", "B", "C"), n, replace = TRUE)),
  cov1 = rnorm(n),
  cov2 = runif(n),
  cov3 = rbinom(n, 1, 0.5),
  outcome = rnorm(n)
)

# Estimate weights for ATE across all treatment levels using multinomial logistic regression
w.out <- weightit(
  treat ~ cov1 + cov2 + cov3,
  data = df,
  estimand = "ATE",
  method = "glm"  # multinomial model when treat is a factor with >2 levels
)

# Check covariate balance
bal.tab(w.out)
#> Balance summary across all treatment pairs
#>         Type Max.Diff.Adj
#> cov1 Contin.       0.0309
#> cov2 Contin.       0.0341
#> cov3  Binary       0.0109
#> 
#> Effective sample sizes
#>                 A     B      C control
#> Unadjusted 112.   99.   107.     82.  
#> Adjusted   109.59 92.71 100.11   80.73

bal.tab(w.out, which.treat = c("A", "B", "C"))
#> Balance by treatment pair
#> 
#>  - - - A (0) vs. B (1) - - - 
#> Balance Measures
#>         Type Diff.Adj
#> cov1 Contin.  -0.0211
#> cov2 Contin.  -0.0018
#> cov3  Binary   0.0013
#> 
#> Effective sample sizes
#>                 A     B
#> Unadjusted 112.   99.  
#> Adjusted   109.59 92.71
#> 
#>  - - - A (0) vs. C (1) - - - 
#> Balance Measures
#>         Type Diff.Adj
#> cov1 Contin.  -0.0034
#> cov2 Contin.   0.0322
#> cov3  Binary   0.0109
#> 
#> Effective sample sizes
#>                 A      C
#> Unadjusted 112.   107.  
#> Adjusted   109.59 100.11
#> 
#>  - - - B (0) vs. C (1) - - - 
#> Balance Measures
#>         Type Diff.Adj
#> cov1 Contin.   0.0176
#> cov2 Contin.   0.0341
#> cov3  Binary   0.0097
#> 
#> Effective sample sizes
#>                B      C
#> Unadjusted 99.   107.  
#> Adjusted   92.71 100.11
#>  - - - - - - - - - - - - - - - -


# Estimate treatment effects with robust SEs
library(jtools)
model <- glm(outcome ~ relevel(treat, ref = "control"),
             data = df,
             weights = w.out$weights)
summ(model, robust = "HC1")
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Observations </td>
   <td style="text-align:right;"> 400 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Dependent variable </td>
   <td style="text-align:right;"> outcome </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Type </td>
   <td style="text-align:right;"> Linear regression </td>
  </tr>
</tbody>
</table> <table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> χ²(3) </td>
   <td style="text-align:right;"> 33.68 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> p </td>
   <td style="text-align:right;"> 0.03 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Pseudo-R² (Cragg-Uhler) </td>
   <td style="text-align:right;"> 0.02 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Pseudo-R² (McFadden) </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> AIC </td>
   <td style="text-align:right;"> 1143.60 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> BIC </td>
   <td style="text-align:right;"> 1163.56 </td>
  </tr>
</tbody>
</table> <table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;border-bottom: 0;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Est. </th>
   <th style="text-align:right;"> S.E. </th>
   <th style="text-align:right;"> t val. </th>
   <th style="text-align:right;"> p </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> (Intercept) </td>
   <td style="text-align:right;"> -0.21 </td>
   <td style="text-align:right;"> 0.11 </td>
   <td style="text-align:right;"> -1.93 </td>
   <td style="text-align:right;"> 0.05 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> relevel(treat, ref = "control")A </td>
   <td style="text-align:right;"> 0.26 </td>
   <td style="text-align:right;"> 0.15 </td>
   <td style="text-align:right;"> 1.74 </td>
   <td style="text-align:right;"> 0.08 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> relevel(treat, ref = "control")B </td>
   <td style="text-align:right;"> -0.03 </td>
   <td style="text-align:right;"> 0.15 </td>
   <td style="text-align:right;"> -0.21 </td>
   <td style="text-align:right;"> 0.83 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> relevel(treat, ref = "control")C </td>
   <td style="text-align:right;"> 0.29 </td>
   <td style="text-align:right;"> 0.14 </td>
   <td style="text-align:right;"> 2.00 </td>
   <td style="text-align:right;"> 0.05 </td>
  </tr>
</tbody>
<tfoot><tr><td style="padding: 0; " colspan="100%">
<sup></sup> Standard errors: Robust, type = HC1</td></tr></tfoot>
</table>

Additional Notes:

-   To estimate the ATT for a specific treatment (e.g., treatment A), set `focal = "A"` and `estimand = "ATT"` in `weightit()`.

-   Setting `method = "gbm"` in `WeightIt` will use boosted models, equivalent to the `twang` package.

-   Weighting approaches avoid sample size loss from discarding unmatched units and provide smoother covariate balance optimization.

#### Summary: Matching vs. Weighting in Multi-Treatment Settings

+------------+------------------------------------+---------------------+----------------------------------------+---------------------------------------------------------+
| Method     | Approach                           | Estimands Supported | Pros                                   | Cons                                                    |
+============+====================================+=====================+========================================+=========================================================+
| `MatchIt`  | Pairwise matching (manual)         | ATT, ATC            | Intuitive; transparent matched pairs   | Manual; repeated steps; limited to pairwise comparisons |
+------------+------------------------------------+---------------------+----------------------------------------+---------------------------------------------------------+
| `WeightIt` | Simultaneous weighting             | ATT, ATC, ATE       | Flexible; single model; scalable       | Requires model diagnostics                              |
+------------+------------------------------------+---------------------+----------------------------------------+---------------------------------------------------------+
| `twang`    | Boosted weighting for GPS          | ATT, ATE            | Automatic tuning; longitudinal support | More complex interface                                  |
+------------+------------------------------------+---------------------+----------------------------------------+---------------------------------------------------------+
| `CBPS`     | Balance-constrained GPS estimation | ATT, ATE            | Direct balance optimization            | May require customization                               |
+------------+------------------------------------+---------------------+----------------------------------------+---------------------------------------------------------+

In general, weighting is more scalable and coherent for estimating causal effects in multi-treatment contexts, while matching can be useful when interpretability and transparency of individual pairings is important.

------------------------------------------------------------------------

### Matching for Multi-Level Treatments

Some treatments are not just categorical, but **ordinal** (multi-level), such as "low", "medium", and "high" dosage levels. These differ from multinomial treatments in that the levels have a **natural order**. Incorporating this structure can improve both estimation efficiency and interpretability.

#### Propensity Score Estimation

In this context, researchers often use **ordinal logistic regression** to estimate the probability of being in each treatment level:

$$
\text{logit}\left(\mathbb{P}(T \leq t \mid X)\right) = \alpha_t - X^\top \beta
$$

This model imposes the **proportional odds assumption**, where the log-odds are linear in covariates but share the same coefficients across cutoffs.

#### Matching Strategies

@yang2016propensity provide a framework for matching with ordinal treatments, arguing that matching on the **generalized propensity score** derived from an ordinal model helps preserve the order and improves balance. Key considerations include:

-   Matching on estimated ordinal probabilities or cumulative logits
-   Using Mahalanobis distance in the latent score space
-   Implementing stratification or subclassification based on predicted scores

A custom package, [`shuyang1987/multilevelMatching`](https://github.com/shuyang1987/multilevelMatching), provides tools for this kind of analysis. The package includes:

-   Functions to estimate ordinal GPS
-   Matching and subclassification routines
-   Diagnostics to evaluate covariate balance

------------------------------------------------------------------------

### Matching for Continuous Treatments {#sec-gps-continuous}

The natural endpoint of generalizing the propensity score from binary to multinomial to ordinal treatments is the case where the treatment is fully continuous: a dose $D \in \mathcal{D} \subset \mathbb{R}$, not a level of an ordered factor. The substantive question shifts accordingly. The deliverable is no longer a single contrast or a small set of pairwise effects but an entire **dose-response function** $\mu(d) = \mathbb{E}[Y(d)]$ giving the average potential outcome at each dose. Reporting a single number when the data support a curve is a missed opportunity at best and an averaging-away of heterogeneity at worst.

The cross-sectional framework developed below complements the [continuous-treatment difference-in-differences](#sec-continuous-did) machinery in the DiD chapter. The two address the same substantive question (how does the average effect scale with the dose?) but make different identifying bets: the panel-data DiD relies on parallel trends across doses, while the matching framework here relies on a conditional-ignorability extension of the binary-treatment unconfoundedness assumption. In applications where both are feasible, reporting both is informative; in most applications, the data structure dictates which is available.

#### The Generalized Propensity Score for Continuous Treatments

@hirano2004propensity introduced the generalized propensity score for continuous treatments and showed that the central balancing property of the binary propensity score carries over with only a notational change. The GPS is now defined as the conditional density of the treatment given the covariates, evaluated at the treatment actually received:

$$
R = r(D, X), \quad \text{where} \quad r(d, x) = f_{D \mid X}(d \mid x).
$$

The intuition is the same as in the binary case. Units that, conditional on covariates, were equally likely to receive a given dose are comparable at that dose. The GPS reduces all covariate information relevant for the assignment mechanism to a single scalar that is sufficient for balancing.

The framework relies on a **weak unconfoundedness** assumption, which is the natural lift of the binary-treatment ignorability condition:

$$
Y(d) \perp\!\!\!\perp D \mid X, \quad \forall d \in \mathcal{D}.
$$

In words, within strata defined by the covariates, the dose received is independent of the potential outcomes at every dose. The assumption is "weak" only relative to a stronger joint independence; in practice it is no more or less testable than the binary unconfoundedness assumption it generalizes, and it relies on the same substantive defense: that the analyst has measured everything that jointly drives selection into dose and the outcome.

@hirano2004propensity prove the central balancing result: within strata of the GPS $R$, the dose $D$ is independent of the covariates $X$. The implication is the same as in the binary case: conditioning on $R$ in an outcome model is sufficient to remove the confounding bias from the covariates, even though $R$ is a one-dimensional summary of a potentially high-dimensional $X$. The dose-response function is then identified by

$$
\mu(d) = \mathbb{E}\left[\,\mathbb{E}[Y \mid D = d, R = r(d, X)]\,\right],
$$

where the outer expectation is over the marginal distribution of the covariates (equivalently, over the marginal distribution of $r(d, X)$ at the chosen $d$).

@imai2004causal develop a complementary framework that subsumes binary, multinomial, ordinal, and continuous treatments in a single notation by treating the entire treatment-assignment density as the object of interest. The two frameworks coincide in the continuous case and differ mainly in notational emphasis; the @hirano2004propensity formulation is more common in applied work.

#### Estimators in Practice

Three estimation strategies dominate applied work, ordered by how much functional-form discipline they impose.

The first is the **parametric two-step estimator** of @hirano2004propensity. The GPS is modeled with a parametric density (typically a normal or log-normal conditional on covariates, with the parameters estimated by maximum likelihood). The outcome model is then a flexible parametric regression of $Y$ on a polynomial in $(D, R)$, often quadratic in each. The dose-response curve is recovered by averaging the fitted outcome model over the empirical distribution of the covariates, holding $D$ fixed at each value in a grid. This estimator is computationally cheap and has the asymptotic guarantees one expects from a well-specified parametric model; its weakness is that it inherits the standard parametric-model risk of model misspecification, which in the continuous case is harder to diagnose than in the binary case.

The second is the **covariate-balancing GPS** of @lopez2017estimation and related extensions of CBPS to continuous treatments. The conditional-density model is fit not by likelihood but by minimizing an explicit measure of imbalance between the treated and control distributions across the dose range. The resulting weights produce a pseudo-population in which the dose is approximately independent of the covariates, which lets the analyst estimate the dose-response with a simple weighted regression. The advantage is that covariate balance, the object the analyst actually cares about, is the target of optimization rather than a hoped-for consequence of correct density specification.

The third is the **nonparametric doubly robust estimator** of @kennedy2017nonparametric. The dose-response is estimated by a two-step procedure that combines a nonparametric outcome model with a nonparametric density model for the GPS, with cross-fitting to deliver valid inference under flexible (e.g., machine-learning-based) first-stage estimators. The estimator has the doubly-robust property that consistency holds if either the outcome model or the density model is correctly specified, and the nonparametric flexibility addresses the most common failure mode of the parametric two-step. The cost is larger sample-size requirements and computational expense.

The applied literature provides useful templates. @kluve2012evaluating apply the @hirano2004propensity framework to a continuous-dose training programme in Germany, recovering an inverse-U dose-response in which moderate training duration is most effective. @flores2012estimating use the GPS framework to estimate dose-response curves of the Job Corps programme on labor-market outcomes, with the dose measured as the length of exposure to instruction. Both papers serve as detailed methodological worked examples in addition to their substantive contributions.

#### Worked Example: Demonstration First, Production Second

We work through the GPS dose-response estimator in two passes. The first pass implements the @hirano2004propensity two-step procedure from scratch in base R, so that the mechanics of the GPS, the balancing property, and the dose-response averaging are all visible. The second pass calls the published `CausalGPS` package on the same data.

##### Demonstration: Hirano-Imbens GPS from scratch


``` r
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

set.seed(20240515)

# Generate synthetic data: continuous treatment D depends on three
# confounders X1, X2, X3. The true dose-response is quadratic in D
# with a maximum near D = 2 (an inverse-U pattern, common in training
# programmes).
n <- 2000
X1 <- rnorm(n)
X2 <- rnorm(n)
X3 <- rbinom(n, 1, 0.4)

# Treatment-assignment model: D ~ Normal( mu(X), sigma^2 ).
mu_D    <- 1.5 + 0.6 * X1 - 0.4 * X2 + 0.5 * X3
sigma_D <- 1.0
D       <- rnorm(n, mu_D, sigma_D)

# Outcome model: quadratic in D, with confounding via X.
Y <- 0.3 + 1.2 * D - 0.3 * D^2 + 0.5 * X1 + 0.2 * X2 - 0.1 * X3 +
     rnorm(n, sd = 0.5)

dat <- data.frame(Y, D, X1, X2, X3)
head(dat)
#>           Y          D          X1          X2 X3
#> 1  1.230092  2.1599879  0.57737666 -1.15345217  0
#> 2  1.222963  0.7854613  0.54569647 -0.37654112  0
#> 3  1.614434  1.7445367 -0.03917816  0.62338446  1
#> 4  1.798470  1.4843060  1.07327193 -0.04591857  0
#> 5 -1.625341 -0.4028663 -2.06053669  0.56727904  0
#> 6  2.231506  2.2999686  2.45091808  0.56453003  0
```

The first step of the @hirano2004propensity procedure is to estimate the GPS, the conditional density of $D$ given $X$, evaluated at each unit's actual dose. We assume a normal density and fit $\mu(X)$ by OLS:


``` r
# Step 1: fit the conditional mean of D given X.
gps_mean_model <- lm(D ~ X1 + X2 + X3, data = dat)
mu_hat         <- predict(gps_mean_model)
sigma_hat      <- sd(residuals(gps_mean_model))

# GPS = f_{D|X}(D | X), evaluated at the actual treatment.
dat$R <- dnorm(dat$D, mean = mu_hat, sd = sigma_hat)
summary(dat$R)
#>     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#> 0.001138 0.203811 0.314014 0.278125 0.371811 0.393358
```

The `summary(dat$R)` output reports the distribution of the estimated GPS across the sample: it should be a positive, smooth distribution with no point masses, since $D$ given $X$ is modelled as continuous. Extremely small GPS values flag units with implausible dose-covariate combinations (the continuous-treatment analogue of extreme propensity scores in the binary case).

To diagnose the balancing property, we stratify on the estimated GPS and check that, within strata, the covariates are no longer strongly associated with the dose:


``` r
dat$R_quint <- ntile(dat$R, 5)

balance_check <- dat %>%
  group_by(R_quint) %>%
  summarise(
    cor_D_X1 = cor(D, X1),
    cor_D_X2 = cor(D, X2),
    cor_D_X3 = cor(D, X3),
    n        = n()
  )

# For comparison, the raw correlations before adjustment:
raw_cor <- data.frame(
  R_quint  = "RAW (no adjustment)",
  cor_D_X1 = cor(dat$D, dat$X1),
  cor_D_X2 = cor(dat$D, dat$X2),
  cor_D_X3 = cor(dat$D, dat$X3),
  n        = nrow(dat)
)

bind_rows(raw_cor, mutate(balance_check, R_quint = as.character(R_quint)))
#>               R_quint  cor_D_X1   cor_D_X2  cor_D_X3    n
#> 1 RAW (no adjustment) 0.4571030 -0.3452752 0.2170832 2000
#> 2                   1 0.2501806 -0.2663322 0.1406071  400
#> 3                   2 0.4713361 -0.2795251 0.1722035  400
#> 4                   3 0.6182372 -0.4071251 0.3268779  400
#> 5                   4 0.6687856 -0.4836019 0.3080108  400
#> 6                   5 0.7084768 -0.5881923 0.3330472  400
```

Reading the table: the top row shows the unadjusted correlations between $D$ and each covariate (the confounding we are trying to remove). The next five rows show the within-GPS-stratum correlations. If the balancing property holds, the stratum correlations should be much closer to zero than the raw correlations. The pattern is exactly what we expect from theory: conditioning on a one-dimensional summary of the assignment mechanism is sufficient to balance the covariates that drove the assignment.

The second step of the procedure is the outcome model. We fit a flexible regression of $Y$ on a polynomial in $(D, R)$:


``` r
outcome_model <- lm(Y ~ D + I(D^2) + R + I(R^2) + D:R, data = dat)
summary(outcome_model)$coefficients
#>               Estimate  Std. Error     t value      Pr(>|t|)
#> (Intercept)  0.2713847 0.090632543   2.9943403  2.784231e-03
#> D            1.1800287 0.038133501  30.9446717 4.771610e-172
#> I(D^2)      -0.2997616 0.009080299 -33.0123028 4.817516e-191
#> R           -1.4878512 0.775637020  -1.9182313  5.522435e-02
#> I(R^2)       1.0157479 1.508364331   0.6734102  5.007644e-01
#> D:R          0.6888658 0.101171590   6.8088859  1.297181e-11
```

The coefficient table reports the fitted parametric outcome model; individual coefficients are not the substantive object of interest, but the fitted surface is. The dose-response curve is recovered by averaging the fitted outcome over the empirical covariate distribution, holding $D$ fixed at each value in a grid:


``` r
d_grid <- seq(min(dat$D), max(dat$D), length.out = 100)

# For each grid value of D, predict the outcome at every unit holding D
# fixed at the grid value (and recomputing the GPS at the counterfactual
# dose), then average over the empirical covariate distribution. The mean
# of the predictions IS the dose-response value at that dose.
dose_response <- sapply(d_grid, function(d_val) {
  r_at_d <- dnorm(d_val, mean = mu_hat, sd = sigma_hat)
  preds  <- predict(outcome_model,
                    newdata = data.frame(D = d_val, R = r_at_d))
  mean(preds)
})

curve_df <- data.frame(
  D      = d_grid,
  Y_hat  = dose_response,
  Y_true = 0.3 + 1.2 * d_grid - 0.3 * d_grid^2
)

ggplot(curve_df, aes(x = D)) +
  geom_line(aes(y = Y_hat), linewidth = 1) +
  geom_line(aes(y = Y_true), linetype = "dashed", color = "red") +
  labs(x = "Dose D",
       y = "E[Y(d)]: average potential outcome at dose d",
       title = "GPS dose-response (Hirano-Imbens two-step, by hand)",
       subtitle = "Solid: fitted dose-response. Dashed red: true curve.") +
  theme_minimal()
```

<div class="figure" style="text-align: center">
<img src="35-matching-methods_files/figure-html/gps-cont-dose-response-1.png" alt="Estimated dose-response curve from the Hirano-Imbens GPS two-step. Solid line is the fitted average potential outcome at each dose; dashed red is the true quadratic dose-response used in the simulation. The estimator recovers the inverse-U shape and the location of the maximum." width="90%" />
<p class="caption">(\#fig:gps-cont-dose-response)Estimated dose-response curve from the Hirano-Imbens GPS two-step. Solid line is the fitted average potential outcome at each dose; dashed red is the true quadratic dose-response used in the simulation. The estimator recovers the inverse-U shape and the location of the maximum.</p>
</div>

Figure \@ref(fig:gps-cont-dose-response) compares the estimated dose-response to the true curve. The estimator recovers both the shape (inverse-U) and the approximate location of the maximum, demonstrating that the GPS-stratification logic does the work it is meant to do. In a real application, of course, the true curve is unknown; the analyst reports only the solid line, typically with a bootstrap confidence band.

##### Production: `CausalGPS` package

For production work we recommend the `CausalGPS` package on CRAN by Khoshnevis, Wu, and Braun, which implements both matching and weighting versions of the GPS-based dose-response analysis, with support for flexible (machine-learning-based) density estimation and a built-in nonparametric estimator for the exposure-response function. The pipeline has four steps: estimate the GPS, compute matching or weighting counter-weights, generate the pseudo-population, and estimate the dose-response curve.


``` r
# install.packages("CausalGPS")  # CRAN, Khoshnevis, Wu, Braun
library(CausalGPS)

set.seed(20240515)

# Synthetic data: continuous exposure w, six confounders, outcome Y
m_d <- generate_syn_data(sample_size = 1000)
head(m_d)

# Step 1: Estimate the generalized propensity score.
# Here a gradient-boosted (XGBoost) density model is used, with a normal
# density assumption for the conditional distribution of the exposure.
gps_obj <- estimate_gps(
  .data       = m_d,
  .formula    = w ~ I(cf1^2) + cf2 + I(cf3^2) + cf4 + cf5 + cf6,
  sl_lib      = c("m_xgboost"),
  gps_density = "normal"
)

# Step 2: Compute matching weights. The matching is performed in the
# (exposure, GPS) space using L1 distance; delta_n controls the
# exposure-side bandwidth, scale balances exposure vs. GPS proximity.
cw_object <- compute_counter_weight(
  gps_obj      = gps_obj,
  ci_appr      = "matching",
  delta_n      = 0.1,
  dist_measure = "l1",
  scale        = 0.5
)

# Step 3: Build the pseudo-population (each row carries a counter weight
# representing how many units like it would appear in the rebalanced
# population at the dose it actually received).
pseudo_pop <- generate_pseudo_pop(
  .data               = m_d,
  cw_obj              = cw_object,
  covariate_col_names = c("cf1", "cf2", "cf3", "cf4", "cf5", "cf6")
)

# Step 4: Estimate the exposure-response curve at a grid of dose values.
erf_obj <- estimate_erf(
  .data           = pseudo_pop$.data,
  .formula        = Y ~ w,
  weights_col_name = "counter_weight",
  model_type      = "nonparametric",
  w_vals          = seq(2, 20, 0.5),
  kernel_appr     = "kernsmooth"
)

plot(erf_obj)
```

The figure returned by `plot(erf_obj)` is the analogue of the dose-response figure produced in the from-scratch demonstration above (and of `contdid::ggcont_did()` in the panel-data setting): a curve $\hat{\mu}(d) = \hat{\mathbb{E}}[Y(d)]$ at each grid value of the dose, with a confidence band reflecting the sampling uncertainty in both the GPS estimation step and the outcome estimation step. The right way to read it is to look for the dose range over which the band excludes any substantively interesting null (zero, or a policy benchmark) and to report the shape of the curve rather than any single point. The `summary(erf_obj)` companion returns a table with the grid values of the dose, the point estimate, and the band limits, suitable for tabular reporting alongside the figure.

Two further outputs are useful in production. First, `gnomeplot()` (or the package's `check_covar_balance()` function) returns a covariate-balance diagnostic that plots the absolute standardized mean differences of each covariate against the dose, before and after rebalancing; the right interpretation is that every covariate's adjusted curve should hover near zero across the entire dose range, with deviations flagging doses at which the GPS model failed to balance. Second, `pseudo_pop$adjusted_corr_results` reports the absolute weighted correlation between each covariate and the dose in the pseudo-population; values below 0.1 are conventional for asserting that balance has been achieved.

For weighting (rather than matching) the only change is the `ci_appr` argument:


``` r
cw_weight <- compute_counter_weight(
  gps_obj = gps_obj,
  ci_appr = "weighting"
)
```

Weighting tends to be more efficient when the dose distribution is well populated everywhere in the support of interest; matching tends to be more robust when the dose distribution has thin regions, because it makes the loss of effective sample size explicit rather than hiding it in extreme weights.

#### Diagnostics

The diagnostic that matters most in continuous-treatment GPS analysis is **covariate balance as a function of the dose**, not balance averaged over the entire pseudo-population. The right object to plot is the absolute standardized mean difference of each covariate between units near a target dose and units away from it, evaluated at several doses spanning the support. A pseudo-population that is balanced on average but unbalanced at the doses where the dose-response curve has its most interesting features (typically the tails) is not balanced where the analyst needs it to be. `CausalGPS` provides built-in balance diagnostics that report this dose-stratified balance directly.

A second diagnostic is the **effective sample size after rebalancing**, both globally and at each dose. A pseudo-population that achieves balance only by upweighting a tiny subset of the data is fragile; the dose-response confidence intervals will be wide and sensitive to small perturbations.

The third diagnostic is a **common-support check**: identify the range of doses over which units of every covariate profile have positive (or at least non-negligible) probability of being observed. The dose-response is identified only over this range; extrapolating beyond it is a statement about the model, not about the data.

------------------------------------------------------------------------

### Matching for Repeated Treatments (Time-Varying Treatments)

In longitudinal studies, treatments are often administered at multiple time points. Examples include:

-   A patient receiving a drug in multiple doses over weeks
-   A user receiving marketing emails over several days

This setting raises unique challenges, such as:

-   Time-varying confounding: Covariates affected by prior treatment may influence future treatment and outcomes
-   Cumulative dose effects: Treatment assignment is no longer a one-time event

#### Marginal Structural Models

The most popular framework for analyzing repeated treatments is the **Marginal Structural Model (MSMs)**, which estimates causal effects by weighting each observation using **Inverse Probability of Treatment Weights (IPTW)**.

Let $A_t$ be the treatment at time $t$, and $X_t$ be time-varying covariates. The IPTW for a trajectory is:

$$
w_i = \prod_{t=1}^{T} \frac{1}{\mathbb{P}(A_{it} \mid \bar{A}_{i,t-1}, \bar{X}_{i,t})}
$$

Weights are estimated using logistic regression models at each time point. The outcome model then regresses $Y$ on $\bar{A}_T$ using the weights.

The [`twang`](https://cran.r-project.org/web/packages/twang/vignettes/iptw.pdf) package provides tools for:

-   Estimating time-varying propensity scores
-   Computing IPTWs
-   Fitting marginal structural models
-   Checking covariate balance over time

Practical Notes

-   Stabilized weights help reduce variance
-   Trimming or truncating extreme weights is often necessary to maintain robust inference
-   Dynamic treatment regimes may require further generalizations such as structural nested mean models.

------------------------------------------------------------------------

## Selection on Unobservables {#sec-selection-on-unobservables}

While randomized experiments help eliminate bias from unobserved factors, observational data often leave us vulnerable to [selection on unobservables](#sec-selection-on-unobservables), a scenario where omitted variables jointly affect both treatment assignment and outcomes. Unlike [selection on observables](#sec-selection-on-observables), these unobserved confounders cannot be directly controlled by regression or matching techniques.

One popular strategy to mitigate bias from observable variables is [matching methods](#sec-matching-methods). These methods are widely used to estimate causal effects under the assumption of **selection on observables** and offer two primary advantages:

-   They reduce reliance on functional form assumptions (unlike parametric regression).
-   They rely on the assumption that all covariates influencing treatment assignment are observed.

However, this key assumption is often unrealistic. When relevant covariates are unmeasured, **bias from unobservables remains a threat**. To address this concern, researchers turn to **Rosenbaum Bounds**, a sensitivity analysis framework that quantifies how strong hidden bias must be to explain away the observed treatment effect.

Several econometric methods have been developed to test the robustness of causal estimates against such hidden bias. Below are key strategies that researchers commonly employ:

1.  [Rosenbaum Bounds](#sec-rosenbaum-bounds)\
    This approach assesses the sensitivity of treatment effects to potential hidden bias by bounding the treatment effect under varying assumptions about the strength of unobserved confounding. It is particularly useful when treatment assignment is believed to be *conditionally ignorable*, but this assumption may be imperfect.

2.  [Endogenous Sample Selection] (Heckman-style correction)\
    When sample selection is non-random and correlated with unobserved variables influencing the outcome, the Heckman selection model provides a correction. It introduces the inverse Mills ratio $\lambda$ as a control function. A statistically significant $\lambda$ indicates that unobserved factors influencing selection also affect the outcome, pointing to endogenous selection.

    -   Interpretation tip: A significant $\lambda$ suggests selection bias due to unobservables, and the Heckman model adjusts the outcome equation accordingly.
    -   This method is frequently applied in labor economics and marketing analytics (e.g., correcting for self-selection in surveys or consumer choice modeling).

3.  [Relative Correlation Restrictions](#sec-relative-correlation-restrictions)\
    This method imposes assumptions about the correlation between observed and unobserved variables. For instance, by assuming that the selection on unobservables is not stronger than the selection on observables (as in @altonji2005selection), one can identify bounds on causal effects. These approaches often rely on auxiliary assumptions or symmetry restrictions to enable partial identification.

4.  [Coefficient-Stability Bounds](#sec-coefficient-stability-bounds)\
    This technique evaluates how treatment effect estimates change with the inclusion of additional control variables. If the coefficient on the treatment variable remains stable as more covariates are added, it suggests robustness to omitted variable bias.

    -   This idea underlies methods like Oster's $\delta$ method [@oster2019unobservable], which quantifies how much selection on unobservables would be required to nullify the observed effect, assuming proportional selection between observables and unobservables.

In applied business research, such as customer retention, credit scoring, or pricing analytics, selection on unobservables is often unavoidable. These methods do not *eliminate* the problem but provide frameworks to quantify and interpret the risk of hidden bias. The choice among these tools depends on the specific context, data availability, and the plausibility of underlying assumptions.

------------------------------------------------------------------------

### Rosenbaum Bounds {#sec-rosenbaum-bounds}

Rosenbaum Bounds assess the **robustness of treatment effect estimates to hidden bias** by introducing a sensitivity parameter $\Gamma$ (gamma), which captures the potential effect of an unobserved variable on treatment assignment.

-   $\Gamma = 1$: Indicates perfect randomization, where units in a matched pair are equally likely to receive treatment.
-   $\Gamma = 2$: Suggests that one unit in a matched pair could be up to **twice as likely** to receive treatment due to an unmeasured confounder.

Since $\Gamma$ is unknown, sensitivity analysis proceeds by varying $\Gamma$ and examining how the statistical significance and magnitude of the treatment effect respond. This allows us to evaluate how strong an unobserved confounder would need to be to invalidate our findings.

Because the unobserved variable must affect both selection into treatment and outcomes, Rosenbaum Bounds are often described as a **worst-case analysis** [@diprete2004assessing].

------------------------------------------------------------------------

Imagine we match two individuals based on age and education. One receives treatment, the other does not. If there's an unmeasured trait (say, motivation) that makes one of them **twice** as likely to receive treatment, this is equivalent to $\Gamma = 2$.

We then ask: *At what level of* $\Gamma$ does our result lose significance? If that level is very high (e.g., $\Gamma = 3$ or more), we can be more confident in our findings.

------------------------------------------------------------------------

In layman's terms, consider two matched individuals who differ only in an unobserved trait (e.g., motivation). If one of them is twice as likely to receive treatment because of this unobservable, this corresponds to $\Gamma = 2$. We then ask: *Would the treatment effect remain significant under this level of hidden bias?*

If the estimated effect becomes insignificant only at high values of $\Gamma$ (e.g., $\Gamma > 2$), the effect is considered robust. Conversely, if statistical significance vanishes at $\Gamma \approx 1.1$, the effect is highly sensitive to hidden bias.

------------------------------------------------------------------------

#### Technical Summary

-   Rosenbaum Bounds assess sensitivity without requiring knowledge of the **direction** of the unobserved bias.
-   They apply to **matched data**, typically using **Wilcoxon signed-rank tests** or other non-parametric statistics.
-   In the presence of **random treatment assignment**, such non-parametric tests are valid directly.
-   In **observational data**, they are valid under the assumption of **selection on observables**. If that assumption is questionable, Rosenbaum Bounds offer a way to **test its believability** [@rosenbaum2002attributing].

A typical Rosenbaum Bounds analysis presents:

-   The $p$-value for statistical significance at increasing levels of $\Gamma$.
-   The Hodges-Lehmann point estimate (a robust median-based treatment effect estimator).
-   The critical $\Gamma$ value at which the treatment effect becomes insignificant.

These analyses do not provide exact bounds on treatment effect estimates. For that, other approaches such as [Relative Correlation Restrictions](#sec-selection-on-unobservables) are required.

------------------------------------------------------------------------

#### Technical Framework

Let $\pi_i$ denote the probability that unit $i$ receives treatment. The odds of treatment are:

$$
\text{Odds}_i = \frac{\pi_i}{1 - \pi_i}
$$

Ideally, after matching, if there's no hidden bias, $\pi_i = \pi_j$ for unit $i$ and $j$.

Rosenbaum bounds constrain the **odds ratio** between two matched units $i$ and $j$:

$$
\frac{1}{\Gamma} \le \frac{\text{Odds}_i}{\text{Odds}_j} \le \Gamma
$$

This relationship links the hidden bias directly to selection probabilities.

Suppose treatment assignment follows:

$$
\log\left( \frac{\pi_i}{1 - \pi_i} \right) = \kappa x_i + \gamma u_i
$$

where $u_i$ is an unobserved covariate, and $x_i$ is an observed covariate. The greater the $\gamma$, the stronger the influence of the unobservable.

------------------------------------------------------------------------

#### Sensitivity Analysis Procedure

1.  Choose a plausible range of $\Gamma$ (typically from 1 to 2.5).
2.  Assess how the p-value or the magnitude of the treatment effect [@hodges2011estimates] (for more details, see @hollander2013nonparametric) changes with varying $\Gamma$ values.
3.  Employ specific randomization tests based on the type of outcome to establish bounds on inferences.
    -   Report the minimum value of $\Gamma$ at which the treatment effect is nullified (i.e., becomes insignificant). And the literature's rules of thumb is that if $\Gamma > 2$, then we have strong evidence for our treatment effect is robust to large biases [@proserpio2017online].

If treatment is clustered (e.g., by region or school), standard Rosenbaum bounds must be adjusted. See [@hansen2014clustered] for methods that extend sensitivity analysis to clustered settings, analogous to using clustered standard errors in regression.

------------------------------------------------------------------------

#### Empirical Marketing Examples

Table \@ref(tab:rosenbaum-gamma-marketing) reports $\Gamma$ thresholds needed to nullify treatment effects in real-world marketing studies.

Table: (\#tab:rosenbaum-gamma-marketing) Critical Rosenbaum $\Gamma$ values from selected marketing studies.

+---------------------------+-------------------+-----------------------------------------------------------+
| Study                     | Critical $\Gamma$ | Context                                                   |
+===========================+===================+===========================================================+
| [@oestreicher2013content] | 1.5 to 1.8        | User community participation on willingness to pay        |
+---------------------------+-------------------+-----------------------------------------------------------+
| [@sun2013ad]              | 1.5               | Revenue-sharing program and content popularity            |
+---------------------------+-------------------+-----------------------------------------------------------+
| [@manchanda2015social]    | 1.6               | Impact of social referrals on purchase behavior           |
+---------------------------+-------------------+-----------------------------------------------------------+
| [@sudhir2015peter]        | 1.9 to 2.2        | IT adoption effects on productivity                       |
+---------------------------+-------------------+-----------------------------------------------------------+
| [@proserpio2017online]    | 2.0               | Management responses and online reputation                |
+---------------------------+-------------------+-----------------------------------------------------------+
| [@zhang2022makes]         | 1.55              | Verified photos and Airbnb demand                         |
+---------------------------+-------------------+-----------------------------------------------------------+
| [@chae2023paywall]        | 27.0              | Paywall suspensions and future subscriptions (not a typo) |
+---------------------------+-------------------+-----------------------------------------------------------+

Packages

-   `rbounds` [@keele2010overview]

-   `sensitivitymv` [@rosenbaum2015two]

Since we typically assess our estimate sensitivity to unobservables after matching, we first do some matching.


``` r
library(MatchIt)
library(Matching)
data("lalonde", package = "MatchIt")

# Matching
matched <- matchit(treat ~ age + educ, data = MatchIt::lalonde, method = "nearest")
summary(matched)
#> 
#> Call:
#> matchit(formula = treat ~ age + educ, data = MatchIt::lalonde, 
#>     method = "nearest")
#> 
#> Summary of Balance for All Data:
#>          Means Treated Means Control Std. Mean Diff. Var. Ratio
#> distance        0.3080        0.2984          0.2734     0.4606
#> age            25.8162       28.0303         -0.3094     0.4400
#> educ           10.3459       10.2354          0.0550     0.4959
#>          eCDF Mean eCDF Max
#> distance    0.0881   0.1663
#> age         0.0813   0.1577
#> educ        0.0347   0.1114
#> 
#> Summary of Balance for Matched Data:
#>          Means Treated Means Control Std. Mean Diff. Var. Ratio
#> distance        0.3080        0.3077          0.0094     0.9963
#> age            25.8162       25.8649         -0.0068     1.0300
#> educ           10.3459       10.2865          0.0296     0.5886
#>          eCDF Mean eCDF Max Std. Pair Dist.
#> distance    0.0033   0.0432          0.0146
#> age         0.0050   0.0162          0.0597
#> educ        0.0253   0.1189          0.8146
#> 
#> Sample Sizes:
#>           Control Treated
#> All           429     185
#> Matched       185     185
#> Unmatched     244       0
#> Discarded       0       0
matched_data <- match.data(matched)

# Split into groups
treatment_group <- subset(matched_data, treat == 1)
control_group   <- subset(matched_data, treat == 0)

# Load rbounds package
library(rbounds)

# P-value sensitivity
psens_res <- psens(treatment_group$re78,
                   control_group$re78,
                   Gamma = 2,
                   GammaInc = 0.1)
psens_res
#> 
#>  Rosenbaum Sensitivity Test for Wilcoxon Signed Rank P-Value 
#>  
#> Unconfounded estimate ....  0.9651 
#> 
#>  Gamma Lower bound Upper bound
#>    1.0      0.9651      0.9651
#>    1.1      0.8969      0.9910
#>    1.2      0.7778      0.9980
#>    1.3      0.6206      0.9996
#>    1.4      0.4536      0.9999
#>    1.5      0.3047      1.0000
#>    1.6      0.1893      1.0000
#>    1.7      0.1097      1.0000
#>    1.8      0.0597      1.0000
#>    1.9      0.0308      1.0000
#>    2.0      0.0151      1.0000
#> 
#>  Note: Gamma is Odds of Differential Assignment To
#>  Treatment Due to Unobserved Factors 
#> 

# Hodges-Lehmann estimate sensitivity
hlsens_res <- hlsens(treatment_group$re78,
                     control_group$re78,
                     Gamma = 2,
                     GammaInc = 0.1)
hlsens_res
#> 
#>  Rosenbaum Sensitivity Test for Hodges-Lehmann Point Estimate 
#>  
#> Unconfounded estimate ....  1376.248 
#> 
#>  Gamma Lower bound Upper bound
#>    1.0     1376.20      1376.2
#>    1.1      852.45      1682.0
#>    1.2      456.55      2023.1
#>    1.3      137.25      2405.9
#>    1.4     -133.45      2765.2
#>    1.5     -461.65      3063.6
#>    1.6     -751.85      3328.5
#>    1.7     -946.45      3591.6
#>    1.8    -1185.50      3864.8
#>    1.9    -1404.70      4133.1
#>    2.0    -1624.60      4358.4
#> 
#>  Note: Gamma is Odds of Differential Assignment To
#>  Treatment Due to Unobserved Factors 
#> 
```

For matching with more than one control per treated unit:


``` r
library(Matching)
library(MatchIt)

n_ratio <- 2
matched <-
    matchit(treat ~ age + educ,
            data = MatchIt::lalonde,
            method = "nearest",
            ratio = n_ratio)
matched_data <- match.data(matched)

mcontrol_res <- rbounds::mcontrol(
    y          = matched_data$re78,
    grp.id     = matched_data$subclass,
    treat.id   = matched_data$treat,
    group.size = n_ratio + 1,
    Gamma      = 2.5,
    GammaInc   = 0.1
)
mcontrol_res
#> 
#>  Rosenbaum Sensitivity Test for Wilcoxon Strat. Rank P-Value 
#>  
#> Unconfounded estimate ....  0.9977 
#> 
#>  Gamma Lower bound Upper bound
#>    1.0      0.9977      0.9977
#>    1.1      0.9892      0.9996
#>    1.2      0.9651      0.9999
#>    1.3      0.9144      1.0000
#>    1.4      0.8301      1.0000
#>    1.5      0.7149      1.0000
#>    1.6      0.5809      1.0000
#>    1.7      0.4445      1.0000
#>    1.8      0.3209      1.0000
#>    1.9      0.2195      1.0000
#>    2.0      0.1428      1.0000
#>    2.1      0.0888      1.0000
#>    2.2      0.0530      1.0000
#>    2.3      0.0305      1.0000
#>    2.4      0.0170      1.0000
#>    2.5      0.0092      1.0000
#> 
#>  Note: Gamma is Odds of Differential Assignment To
#>  Treatment Due to Unobserved Factors 
#> 
```

Alternative Sensitivity Analysis Tools

`sensitivitymv` is faster than `sensitivitymw`. But `sensitivitymw` can match where matched sets can have differing numbers of controls [@rosenbaum2015two].


``` r
library(sensitivitymv)
data(lead150)
senmv(lead150, gamma = 2, trim = 2)
#> $pval
#> [1] 0.02665519
#> 
#> $deviate
#> [1] 1.932398
#> 
#> $statistic
#> [1] 27.97564
#> 
#> $expectation
#> [1] 18.0064
#> 
#> $variance
#> [1] 26.61524

library(sensitivitymw)
senmw(lead150, gamma = 2, trim = 2)
#> $pval
#> [1] 0.02665519
#> 
#> $deviate
#> [1] 1.932398
#> 
#> $statistic
#> [1] 27.97564
#> 
#> $expectation
#> [1] 18.0064
#> 
#> $variance
#> [1] 26.61524
```

------------------------------------------------------------------------

### Relative Correlation Restrictions {#sec-relative-correlation-restrictions}

In many observational studies, researchers are concerned about the impact of unobserved variables that are not included in the regression model. Even after including a comprehensive set of controls, there is always the possibility that omitted variables still bias the estimated treatment effect. **Relative Correlation Restrictions (RCR)** offer a formal way to assess the **potential magnitude of this bias** by comparing it to the observed selection.

------------------------------------------------------------------------

Let's begin with a linear outcome model:

$$
Y_i = X_i \beta + C_i \gamma + \epsilon_i
$$

Where:

-   $Y_i$ is the outcome variable (e.g., revenue, engagement, churn),
-   $X_i$ is the treatment or variable of interest (e.g., paywall suspension, ad exposure),
-   $C_i$ is a set of observed control covariates,
-   $\epsilon_i$ is the error term, which may include unobserved factors.

In standard OLS regression, we assume:

$$
\text{Cov}(X_i, \epsilon_i) = 0
$$

That is, the treatment $X_i$ is uncorrelated with the omitted variables contained in $\epsilon_i$. This assumption is crucial for the **unbiased estimation of** $\beta$, the treatment effect.

However, in observational settings, this assumption may not hold. There may be unobserved confounders that influence both $X_i$ and $Y_i$, biasing the OLS estimate of $\beta$. The Relative Correlation Restrictions framework provides a way to bound the treatment effect under assumptions about the relative strength of selection on unobservables compared to selection on observables.

------------------------------------------------------------------------

The method was originally proposed by @altonji2005selection and later extended by @krauth2016bounding. The core idea is to assume a proportional relationship between the correlation of the treatment with the error term and its correlation with the observed controls:

$$
\text{Corr}(X_i, \epsilon_i) = \lambda \cdot \text{Corr}(X_i, C_i \gamma)
$$

Here, $\lambda$ represents the relative strength of selection on unobservables compared to observables.

-   If $\lambda = 0$, then we return to the standard OLS assumption: no unobserved selection.
-   If $\lambda = 1$, we assume that selection on unobservables is as strong as selection on observables.
-   If $\lambda > 1$, then unobserved selection is assumed to be stronger than observable selection.
-   In practice, we evaluate treatment effects over a range of plausible values for $\lambda$.

This approach allows us to **bound the treatment effect** based on assumptions about the degree of omitted variable bias.

------------------------------------------------------------------------

The choice of $\lambda$ is crucial and inherently subjective, but some guidelines and empirical precedents help:

-   Small values of $\lambda$ (e.g., 0--1) represent modest levels of bias and suggest that selection on unobservables is not dominant.
-   Large values (e.g., $\lambda > 2$ or higher) imply that any omitted variables must be far more predictive of treatment than the observed controls, a strong and often implausible assumption in well-specified models.

This makes RCR particularly useful for *stress testing* causal estimates: "How bad would unobserved selection have to be to overturn my result?"

------------------------------------------------------------------------

This method has been applied to various marketing and digital strategy settings to assess the robustness of estimated effects. A few notable examples:

+------------------------+---------------------------------------------------------------------------+-------------------------------------+
| Study                  | Context                                                                   | Minimum $\lambda$ to nullify effect |
+========================+===========================================================================+=====================================+
| [@manchanda2015social] | **Social dollar effect**: impact of peer influence on purchasing behavior | 3.23                                |
+------------------------+---------------------------------------------------------------------------+-------------------------------------+
| [@chae2023paywall]     | **Paywall suspension**: effect on future subscription behavior            | 6.69                                |
+------------------------+---------------------------------------------------------------------------+-------------------------------------+
| [@sun2013ad]           | **Ad revenue-sharing**: impact on content popularity                      |                                     |
+------------------------+---------------------------------------------------------------------------+-------------------------------------+

These high $\lambda$ values imply that unobserved selection would need to be 3 to 7 times stronger than observable selection to eliminate the estimated treatment effect. In practical terms, this offers strong evidence that the effects are robust to omitted variable bias.

------------------------------------------------------------------------

We can estimate Relative Correlation Restrictions using the `rcrbounds` package in R, developed by @krauth2016bounding. This package estimates the bounds of the treatment effect across a range of $\lambda$ values and provides inference on whether the effect would remain statistically significant. Figure \@ref(fig:mm-rcrbounds-chickweight) plots the estimated treatment-effect bounds across $\lambda$ for the ChickWeight illustration.


``` r
# Install if necessary:
# remotes::install_github("bvkrauth/rcr/r/rcrbounds")
# Declare the Python backend so reticulate resolves it per session.
reticulate::py_require("rcrbounds")
library(rcrbounds)

# Example using ChickWeight dataset
data("ChickWeight")

# Estimate treatment effect of Time on weight, controlling for Diet
rcr_res <- rcrbounds::rcr(
    formula = weight ~ Time | Diet, 
    data = ChickWeight, 
    # rc_range = c(0, 10) # Test lambda from 0 to 10
    rc_range = c(0, 1)
)

# Print summary
summary(rcr_res)
#> 
#> Call:
#> rcrbounds::rcr(formula = weight ~ Time | Diet, data = ChickWeight, 
#>     rc_range = c(0, 1))
#> 
#> Coefficients:
#>            Estimate  Std. Error    t value      Pr(>|t|)
#> rcInf     34.676505  50.1295005  0.6917385  4.891016e-01
#> effectInf 71.989336 112.5711682  0.6395007  5.224973e-01
#> rc0       34.741955  58.7169195  0.5916856  5.540611e-01
#> effectL    8.624677   0.3337819 25.8392614 3.212708e-147
#> effectH    8.750492   0.2607671 33.5567354 7.180406e-247
#> ---
#> conservative confidence interval:
#>          2.5  %  97.5  %
#> effect 7.970477 9.261586

# Test whether the treatment effect is significantly different from 0
rcrbounds::effect_test(rcr_res, h0 = 0)
#> [1] 5.960464e-08

# Plot the bounds of the treatment effect
plot(rcr_res)
```

<div class="figure" style="text-align: center">
<img src="35-matching-methods_files/figure-html/mm-rcrbounds-chickweight-1.png" alt="Relative-correlation-restriction bounds for the treatment effect of Time on weight in the ChickWeight data, plotted across the sensitivity parameter lambda using the rcrbounds package." width="90%" />
<p class="caption">(\#fig:mm-rcrbounds-chickweight)Relative-correlation-restriction bounds for the treatment effect of Time on weight in the ChickWeight data, plotted across the sensitivity parameter lambda using the rcrbounds package.</p>
</div>

The plot shows how the estimated treatment effect varies as we allow for stronger selection on unobservables (i.e., increasing $\lambda$). If the effect remains consistently different from zero even at high $\lambda$, it provides graphical evidence of robustness.

------------------------------------------------------------------------

### Coefficient-Stability Bounds {#sec-coefficient-stability-bounds}

Robust regression analysis requires methods that account for potential **omitted variable bias**. A contribution to this literature is the coefficient-stability framework developed by @oster2019unobservable, which builds on the proportional selection model of @altonji2005selection. @oster2019unobservable derives conditions under which we can bound the bias due to unobservables using movements in the coefficient of interest and changes in model fit, as measured by $R^2$.

More recently, @masten2022effect have revealed that Oster's commonly used "delta" metric is insufficient to guarantee robustness of the **sign** of the treatment effect. They propose distinguishing between two types of breakdown points: one where the effect becomes zero (explain-away), and another where it changes sign.

Suppose the data‑generating process for outcome $Y$ is

$$
Y
=\beta X+\underbrace{W_1^{\!\top}\gamma_1}_{\text{observed}}
+\underbrace{W_2^{\!\top}\gamma_2}_{\text{unobserved}}
+\varepsilon ,
\quad
\operatorname{E}[\varepsilon|X,W_1,W_2]=0,
\tag{1}
$$

but the econometrician can only regress $Y$ on $X$ and the **observed** covariates $W_1$. By the omitted‑variable‑bias (OVB) formula,

$$
\widehat\beta_{\text{med}}-\beta
= \frac{\operatorname{Cov}(X,W_2^{\!\top}\gamma_2)}{\operatorname{Var}(X)} .
\tag{2}
$$

Because $W_2$ is hidden, nothing inside the sample alone fixes (2). The literature therefore introduces a *sensitivity parameter* that gauges how strongly $X$ is correlated with the missing variables.

@altonji2005selection postulated that the *selection on unobservables* is proportional to the *selection on observables*. @oster2019unobservable formalized the idea:

$$
\frac{\operatorname{Cov}(X,W_2^{\!\top}\gamma_2)}
     {\operatorname{Var}(W_2^{\!\top}\gamma_2)}
=\delta
\frac{\operatorname{Cov}(X,W_1^{\!\top}\gamma_1)}
     {\operatorname{Var}(W_1^{\!\top}\gamma_1)} .
\tag{3}
$$

where

-   $\delta$ is the **selection ratio**.
-   $\delta = 1$ says "unobservables are as correlated with $X$ as observables," yielding the canonical "Altonji‑Elder‑Taber benchmark."

@oster2019unobservable also assumes **coefficient alignment** ($\gamma_1=C\pi_1$) so we can link bias to the *movement* of $\widehat\beta$ and $R^2$ when controls are added. Under (1)--(3) the *long‑run* coefficient (with $W_2$ observed) satisfies

$$
\boxed{
\beta_{\text{long}}
=\beta_{\text{med}}
+(\beta_{\text{med}}-\beta_{\text{short}})
     \frac{R^2_{\text{long}}-R^2_{\text{med}}}
          {R^2_{\text{med}}-R^2_{\text{short}}}
}
\tag{4}
$$ where "short" means no controls, "med" means with $W_1$, and "long" is the unfeasible full model. Equation (4) is the **coefficient‑stability adjustment**. The two sensitivity objects are

-   **Bias‑adjusted coefficient**

    $$
    \beta^*(\delta,R^2_{\max})\equiv \beta_{\text{long}}
    \quad\text{after substituting }R^2_{\text{long}}=R^2_{\max},
    \delta\text{ into (4);} \tag{5}
    $$

-   **Breakdown point**

    $$
    \delta^{\text{EA}}
    =
    \inf\{\delta:\beta^*(\delta,R^2_{\max})=0\}.
    \tag{6}
    $$

When $\delta^{\text{EA}}>1$, unobservables must be *more* predictive of $X$ than observables to wipe out the estimate.

------------------------------------------------------------------------

@masten2022effect prove a stronger result: under Assumptions (1)--(3) the **sign‑change breakdown**

$$
\delta^{\text{SC}}
=\inf\{\delta:\operatorname{sign}\beta^*(\delta,R^2_{\max})
                \neq \operatorname{sign}\beta_{\text{med}}\}
\tag{7}
$$

is *always* ≤ 1 if $R^2_{\max}>R^2_{\text{med}}$ and $\beta_{\text{short}}\neq\beta_{\text{med}}$. Hence the traditional "$\delta = 1$ rule" can never guarantee that the *direction* of the effect is robust; it merely speaks to *magnitude*. Putting both $\delta^{EA}$ and $\delta^{SC}$ on the table is therefore critical.

------------------------------------------------------------------------

@oster2019unobservable recommends $R^2_{\max}=1.3R^2_{\text{med}}$ based on external evidence from randomized experiments, but researchers should vary this choice, especially if $R^2_{\text{med}}$ is modest. @cinelli2020making re‑express the problem in partial‑$R^2$ space and propose the **robustness value**, implemented in the `sensemakr` package. @imbens2003sensitivity supply complementary contour‑plot tools.

------------------------------------------------------------------------


``` r
library(tidyverse)
library(MatchIt)
library(robomit)


data("lalonde", package = "MatchIt")
lalonde <- lalonde %>% mutate(log_re78 = log(re78 + 1))

# create race dummies if needed
if (!("black" %in% names(lalonde))) {
  lalonde <- lalonde %>% mutate(
    black  = as.integer(race == "black"),
    hispan = as.integer(race == "hispanic")
  )
}

# nodegree naming patch
if (!("nodegr" %in% names(lalonde))) {
  if ("nodegree" %in% names(lalonde)) {
    lalonde <- lalonde %>% rename(nodegr = nodegree)
  }
}

# assure 0/1 indicators
lalonde <-
    lalonde %>% mutate(across(c(treat, black, hispan, married, nodegr),
                              as.numeric))

## analysis sample & R^2
vars <- c(
    "log_re78",
    "treat",
    "age",
    "educ",
    "black",
    "hispan",
    "married",
    "nodegr",
    "re74",
    "re75"
)
model_df <- lalonde %>% dplyr::select(all_of(vars)) %>% na.omit()

R2_med <- summary(
    lm(
        log_re78 ~ treat + age + educ + black + hispan +
            married + nodegr + re74 + re75,
        data = model_df
    )
)$r.squared
R2_max <- 1.3 * R2_med # Oster default upper bound

## β* (δ = 1) & δ* (β = 0) 
beta_tbl  <- o_beta(
    y = "log_re78",
    x = "treat",
    con = "age + educ + black + hispan + married + nodegr + re74 + re75",
    delta = 1,
    R2max = R2_max,
    type = "lm",
    data = model_df
)

delta_tbl <- o_delta(
    y = "log_re78",
    x = "treat",
    con = "age + educ + black + hispan + married + nodegr + re74 + re75",
    beta = 0,
    R2max = R2_max,
    type = "lm",
    data = model_df
)

beta_val  <- beta_tbl  %>% filter(Name == "beta*")  %>% pull(Value)
delta_val <- delta_tbl %>% filter(Name == "delta*") %>% pull(Value)

list(
  bias_adjusted_beta = beta_val,
  explain_away_delta = delta_val
)
#> $bias_adjusted_beta
#> [1] 1.378256
#> 
#> $explain_away_delta
#> [1] -1.845051
```


``` r
# 1.  δ* as Rmax varies (explain‑away curve)
o_delta_rsq_viz(
  y   = "log_re78", x = "treat",
  con = "age + educ + black + hispan + married + nodegr + re74 + re75",
  beta = 0,                     # explain‑away target
  type = "lm",
  data = model_df
)
```

<img src="35-matching-methods_files/figure-html/unnamed-chunk-24-1.png" width="90%" style="display: block; margin: auto;" />

``` r

# 2.  bootstrap sampling distribution of δ*
o_delta_boot_viz(
  y   = "log_re78", x = "treat",
  con = "age + educ + black + hispan + married + nodegr + re74 + re75",
  beta   = 0,
  R2max  = R2_max,
  sim    = 100,                 # number of bootstrap draws
  obs    = nrow(model_df),      # draw full‑sample size each time
  rep    = TRUE,                # with replacement
  CI     = c(90, 95, 99),       # show three confidence bands
  type   = "lm",
  norm   = TRUE,                # overlay normal curve
  bin    = 120,                 # histogram bins
  data   = model_df
)
```

<img src="35-matching-methods_files/figure-html/unnamed-chunk-24-2.png" width="90%" style="display: block; margin: auto;" />

``` r

# 3.  bias‑adjusted β* over a grid of Rmax value
o_beta_rsq_viz(
  y   = "log_re78", x = "treat",
  con = "age + educ + black + hispan + married + nodegr + re74 + re75",
  delta = 1,                    # proportional selection benchmark
  type  = "lm",
  data  = model_df
)
```

<img src="35-matching-methods_files/figure-html/unnamed-chunk-24-3.png" width="90%" style="display: block; margin: auto;" />

1.  **Explain‑away curve**: plots $\delta^*$ against a range of $R^2_{\max}$.
2.  **Bootstrap histogram**: sampling distribution of $\delta^*$, with 90/95/99 % bands.
3.  $\beta^*$ vs. $R^2_{\max}$: how the bias‑adjusted coefficient moves as you tighten or loosen the assumed upper bound on $R^2$.
