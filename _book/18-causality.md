# Causal Inference

After all of the mambo jumbo that we have learned so far, I want to now talk about the concept of causality. We usually say that correlation is not causation. Then, what is causation?

This question has been the focus of many disciplines since the 1920s as the field of causal inference garnered interest in econometrics [@imbens2020, p. 1129]. Even though using somewhat similar terminologies, there are two main frameworks to approach causal inference: potential outcome framework, and directed cyclic graph (DAG).

+-------------+---------------------------------------------+------------------------------+
|             | Potential Outcome                           | Directed Acyclic Graph       |
+=============+=============================================+==============================+
| Origin      | Randomized controlled trials (RCT)          | Path analysis (Philip Wright |
|             |                                             |                              |
|             | Demand and supply framework (Jan Tinbergen) |                              |
+-------------+---------------------------------------------+------------------------------+
| Originators | Ronald Fisher                               | Judea pearl                  |
|             |                                             |                              |
|             | Jerzey Nyman                                |                              |
+-------------+---------------------------------------------+------------------------------+
| Field       | Economics, Econometrics                     | Computer Science             |
+-------------+---------------------------------------------+------------------------------+
| Focus       | Estimation, Inference                       | Identification               |
+-------------+---------------------------------------------+------------------------------+

## Intro to DAG Framework

One of my favorite books has explained this concept beautifully [@Pearl_2018]. And I am just going to quickly summarize the gist of it from my understanding. I hope that it can give you an initial grasp on the concept so that later you can continue to read up and develop a deeper understanding.

It's important to have a deep understanding regarding the method research. However, one needs to be aware of its limitation. As mentioned in various sections throughout the book, we see that we need to ask experts for number as our baseline or visit literature to gain insight from past research.

Here, we dive in a more conceptual side statistical analysis as a whole, regardless of particular approach.

You probably heard scientists say correlation doesn't mean causation. There are ridiculous [spurious correlations](http://www.tylervigen.com/spurious-correlations) that give a firm grip on what the previous phrase means. The pioneer who tried to use regression to infer causation in social science was @yule1899 (but it was a fatal attempt where he found relief policy increases poverty). To make a causal inference from statistics, **the equation (function form) must be stable** under intervention (i.e., variables are manipulated). Statistics is used to be a causality-free enterprise in the past.

Not until the development of path analysis by Sewall Wright in the 1920s that the discipline started to pay attention to causation. Then, it remained dormant until the Causal Revolution (quoted Judea Pearl's words). This revolution introduced the calculus of causation which includes (1) causal diagrams), and (2) a symbolic language

The world has been using $P(Y|X)$ (statistics use to derive this), but what we want is to compare the difference between

-   $P(Y|do(X))$: treatment group

-   $P(Y|do(not-X))$: control group

Hence, we can see a clear difference between $P(Y|X) \neq P(Y|do(X))$

The conclusion we want to make from data is counterfactuals: **What would have happened had we not do X?**

To teach a robot to make inference, we need inference engine

![p. 12 [@Pearl_2018]](images/Figure%20I.png "Inference Engine"){style="display: block; margin: 1em auto" width="60%"}

Levels of cognitive ability to be a causal learner:

1.  Seeing
2.  Doing
3.  Imagining

Ladder of causation (associated with levels of cognitive ability as well):

1.  Association: conditional probability, correlation, regression
2.  Intervention
3.  Counterfactuals

+-----------------+-------------+-------------------------------------------+------------------------------------------------------------+
| Level           | Activity    | Questions                                 | Examples                                                   |
+=================+=============+===========================================+============================================================+
| Association     | Seeing      | What is?                                  | What does a symptom tell me about a disease?               |
|                 |             |                                           |                                                            |
| $P(y|x)$        |             | How would seeing X change my belief in Y? |                                                            |
+-----------------+-------------+-------------------------------------------+------------------------------------------------------------+
| Intervention    | Doing       | What if?                                  | What if I spend more time learning, will my result change? |
|                 |             |                                           |                                                            |
| $P(y|do(x),z)$  | Intervening | What if I do X?                           |                                                            |
+-----------------+-------------+-------------------------------------------+------------------------------------------------------------+
| Counterfactuals | Imagining   | Why?\                                     | What if I stopped smoking a year ago?                      |
|                 |             | was it X that caused Y?                   |                                                            |
| $P(y_x|x',y')$  |             |                                           |                                                            |
|                 |             | What if I had acted differently           |                                                            |
+-----------------+-------------+-------------------------------------------+------------------------------------------------------------+

Table by [@pearl2019, p. 2]

You cannot define causation from probability alone

If you say X causes Y if X raises the probability of Y. On the surface, it might sound intuitively right. But when we translate it to probability notation: $P(Y|X) >P(Y)$ , it can't be more wrong. Just because you are seeing X (1st level), it **doesn't mean** the probability of Y increases.

It could be either that (1) X causes Y, or (2) Z affects both X and Y. Hence, people might use **control variables**, which translate: $P(Y|X, Z=z) > P(Y|Z=z)$, then you can be more confident in your probabilistic observation. However, the question is how can you choose $Z$

With the invention of the do-operator, now you can represent X causes Y as

$$
P(Y|do(X)) > P(Y)
$$

and with the help of causal diagram, now you can answer questions at the 2nd level (Intervention)

Note: people under econometrics might still use "Granger causality" and "vector autoregression" to use the probability language to represent causality (but it's not).

<br>

The 7 tools for Structural Causal Model framework [@pearl2019]:

1.  Encoding Causal Assumptions - transparency and testability (with graphical representation)

2.  Do-calculus and the control of confounding: "back-door"

3.  The algorithmization of Counterfactuals

4.  Mediation Analysis and the Assessment of Direct and Indirect Effects

5.  Adaptability, External validity and Sample Selection Bias: are still researched under "domain adaptation", "transfer learning"

6.  Recovering from missing data

7.  Causal Discovery:

    1.  d-separation

    2.  Functional decomposition [@hoyer2008; @shimizu2009; @chen2012a]

    3.  Spontaneous local changes [@pearla]

<br>

Simpson's Paradox:

-   A statistical association seen in an entire population is reversed in sub-population.

Structural Causal Model accompanies graphical causal model to create more efficient language to represent causality

Structural Causal Model is the solution to the curse of dimensionality (i.e., large numbers of variable $p$, and small dataset $n$) thanks to product decomposition. It allows us to solve problems without knowing the function, parameters, or distributions of the error terms.

Suppose you have a causal chain $X \to Y \to Z$:

$$
P(X=x,Y=y, Z=z) = P(X=x)P(Y=y|X=x)P(Z=z|Y=y)
$$

<br>

The three basic paths in DAG are

![simple dags](images/dag.PNG){width="100%"}

+-----------------+--------------------------------+--------------------------------+
| The Chain       | The Fork                       | The Collider                   |
|                 |                                |                                |
| (Mediation)     | (Backdoor)                     |                                |
+=================+================================+================================+
| $A \to C \to B$ | $A \leftarrow C \rightarrow B$ | $A \rightarrow C \leftarrow B$ |
+-----------------+--------------------------------+--------------------------------+
| $A \not\perp B$ | $A \not\perp B$                | $A \perp B$                    |
+-----------------+--------------------------------+--------------------------------+
| $A \perp B | C$ | $A \perp B | C$                | $A \not\perp B | C$            |
+-----------------+--------------------------------+--------------------------------+

Hence, we can see that only in the collider's case that we can open the dependence path between A and B by conditioning on the collider C. More general, a path can be blocked iff:

-   For collider path, the collider (or its descendants) is not conditioned on.

-   For non-collider path, the connector (either mediator or confounder) is conditioned on.

$C$ d-seperates $A$ from $B$ if it blocks all paths from $A \to B$

d-seperation means that $A \perp B | C$

<br>

## Intro to PO Framework

The origin of the potential outcomes framework can be traced back to [@rubin1974]

Let $i = 1, \dots, n$ be a set of i.i.d subjects where we observe

-   $X_i \in R^p$ (feature vector)

$$
T_i = 
\begin{cases}
1 & \text{if unit i is treated} \\
0 & \text{if unit i is untreated}
\end{cases}
$$

Conventionally, people usually use $T$ or $D$ to denote treatment assignment, sometimes even $W$

$Y_i \in R$ (observed outcome) is the variable of interest where we have the fundamental problem of causal inference presents itself. Until we have the time machine to travel back in time or dimensional portal to travel between parallel universes, we will never observe **potential outcome** (outcome that would have happened if the untreated unit was treated or if the treated unit was untreated)

-   Potential outcome that happened is called **factual**.

-   Potential outcome that didn't occur is **counterfactual**.

$$
\text{potential outcome} = 
\begin{cases}
Y_i(1) = Y_{1i} \text{ if } T_i = 1 \\
Y_i(0) = Y_{0i} \text{ if } T_i = 0
\end{cases}
$$

Note: the two notations are equivalent

We only observe is the outcome given treatment assginment

$$
Y_i = Y_i (T_i)= Y_{1i} T_i + Y_{0i} (1 - T_i)
$$

Then, the individual causal treatment effect is

$$
Y_i(1) - Y_i(0)
$$

Our goal is usually to estimate the [Average Treatment Effect]

$$
\tau = E[Y_i(1)- Y_i(0)]
$$

Since we only observe one realization of $Y_i = Y_i(T_i)$ (under we either have a time-machine, or dimensional portal to travel between parallel universes), this is a fundamental problem in causal inference (people also think of this as "missing data" problem).

| Experimental Design                            | Quasi-experimental Design                       |
|------------------------------------------------|-------------------------------------------------|
| Experimentalist                                | Observationalist                                |
| Experimental Data                              | Observational Data                              |
| Random Assignment (reduce treatment imbalance) | Random Sampling (reduce sample selection error) |

Tools in a hierarchical order

1.  [Experimental Design]: Randomized Control Trials (Gold standard): Tier 1

2.  [Quasi-experimental]

    1.  [Regression Discontinuity] Tier 1A

    2.  [Difference-In-Differences] Tier 2

    3.  [Synthetic Control] Tier 2A

    4.  [Event Studies] Tier 2B

    5.  Fixed Effects Estimator \@ref(fixed-effects-estimator): Tier 3

    6.  [Endogenous Treatment]: mostly [Instrumental Variable]: Tier 3A

    7.  [Matching Methods] Tier 4

    8.  [Interrupted Time Series] Tier 4A

    9.  Endogenous Sample Selection \@ref(endogenous-sample-selection): mostly Heckman's correction

    10. [Doubly Robust Estimator]

Internal vs. External Validity

-   Internal Validity: Economists and applied scientists mostly care about

-   External Validity: Localness might affect your external validity

For many economic policies, there is a difference between **treatment** and **intention to treat**.

For example, we might have an effective vaccine (i.e., intention to treat), but it does not mean that everybody will take it (i.e., treatment).

There are four types of subjects that we deal with:

-   **Non-switchers**: we don't care about non-switchers because even if we introduce or don't introduce the intervention, it won't affect them.

    -   **Always takers**

    -   **Never takers**

-   **Switchers**

    -   **Compliers**: defined as those who respect the intervention.

        -   We only care about compliers because when we introduce the intervention, they will do something. When we don't have any interventions, they won't do it.

        -   Tools above are used to identify the causal impact of an intervention on compliers

        -   If we have only **compliers** in our dataset, then **intention to treatment = treatment effect**.

    -   **Defiers**: those who will go to the opposite direction of your treatment.

        -   We typically aren't interested in defiers because they will do the opposite of what we want them to do. And they are typically a small group; hence, we just assume they don't exist.

|               | Treatment Assignment | Control Assignment |
|---------------|----------------------|--------------------|
| Compliers     | Treated              | No Treated         |
| Always-takers | Treated              | Treated            |
| Never-takers  | Not treated          | No treated         |
| Defiers       | Not treated          | Treated            |

<br>

Directional Bias due to selection into treatment comes from 2 general opposite sources

|          | **Mitigation-based**                                                                                                     | **Preference-based**                                                                                                      |
|----------|--------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Reason   | Units select into treatment to combat a problem                                                                          | Units select into treatment because units like that kind of treatment                                                     |
| Notation | $E(Y_0 | T=1) < E(Y_0|T=0)$                                                                                              | $E(Y_0 |T=1) > E(Y_0 |T=0)$                                                                                               |
| In words | Units in the treated group without the treatment is already worse than units in the control group without the treatment. | Units in the treated group without the treatment is already better than units in the control group without the treatment. |

<br>

### Typical Assumptions

#### SUTVA

-   Stable Unit Treatment Values Assumption

1.  The potential outcomes for any unit are not affected by the treatment assignment/exposure of another unit (also known as no interference)
2.  There are no hidden variations of treatment

Violations include

-   Spillover between units in different treatment groups

There are several hypotheses regarding spillovers:

-   Contagion (e.g., vaccine)

-   Displacement (e.g., police intervention displace criminals)

-   Communication (e.g., viral products)

-   Social comparison (e.g., housing assistance can also affect how control group evaluates their homes)

-   Signalling

-   Persistence and memory: individuals can remember, and their treatment effect can be carried over.

#### Independence Assumption

$$
\{Y_i(0)), Y_i(1)\} \perp T_i
$$

Potential outcomes are independent of the treatment status.

Random assignment under [Experimental Design] can introduce Independence between treatment status and potential outcomes

#### Unconfoundedness

$$
[\{Y_i(0)), Y_i(1)\} \perp T_i] | X_i
$$

Potential outcomes given some characteristics are independent of the treatment

This is a weaker form of the [Independence Assumption] where under [Quasi-experimental] design, we typically have this assumption.

#### Bounded propensity score

Also known as overlap: $e(X) \in (0,1)$ (i.e., the propensity scores are bounded for all possible value of $X$).

<br>

### Treatment effect types

Terminology:

-   Quantities of causal interest (i.e., treatment effect types)

-   Estimands: parameters of interest

-   Estimators: procedures to calculate hesitates for the parameters of interest

Sources of bias ([according to prof. Luke Keele](https://www.youtube.com/watch?v=CjZnQ3ToJjg))

$$
\begin{aligned}
&\text{Estimator - True Causal Effect} \\
&= \text{Hidden bias + Misspecification bias + Statistical Noise} \\
&= \text{Due to design + Due to modeling + Due to finite sample}
\end{aligned}
$$

<br>

#### Average Treatment Effects {#average-treatment-effects}

Average treatment effect (ATE) is the difference in means of the treated and control groups

**Randomization** under [Experimental Design] can provide an unbiased estimate of ATE.

Let $Y_i(1)$ denote the outcome of individual $i$ under treatment and

$Y_i(0)$ denote the outcome of individual $i$ under control

Then, the treatment effect for individual $i$ is the difference between her outcome under treatment and control

$$
\tau_i = Y_i(1) - Y_i(0)
$$

Without a time machine or dimension portal, we can only observe one of the two event: either individual $i$ experiences the treatment or she doesn't.

Then, the ATE as a quantity of interest can come in handy since we can observe across all individuals

$$
ATE = \frac{1}{N} \sum_{i=1}^N \tau_i = \frac{\sum_1^N Y_i(1)}{N} - \frac{\sum_i^N Y_i(0)}{N}
$$

With random assignment (i.e., treatment assignment is independent of potential outcome and observables and unobservables), the observed means difference between the two groups is an unbiased estimator of the average treatment effect

$$
E(Y_i (1) |D = 1) = E(Y_i(1)|D=0) = E(Y_i(1)) \\
E(Y_i(0) |D = 1) = E(Y_i(0)|D = 0 ) = E(Y_i(0))
$$

$$
ATE = E(Y_i(1)) - E(Y_i(0))
$$

#### Conditional Average Treatment Effects

Treatment effects can be different for different groups of people. In words, treatment effects can vary across subgroups.

To examine the heterogeneity across groups (e.g., men vs. women), we can estimate the conditional average treatment effects (CATE) for each subgroup

$$
CATE = E[Y(1) - Y(0) |X]
$$

To estimate CATE, we need 2 identification assumptions:

1.  

2.  Bounded propensity score: $0 < e(X) < 1$

Approaches to CATE:

1.  Learner methods (e.g., T-, S-, X-learners)
2.  Transformed outcome [@athey2016]. Causal tree is a regression tree developed to minimize expected MSE of treatment effect.

Why estimating CATE is difficult?

-   Treatment effects can be weak. Hence, estimating its heterogeneity is even harder than [ATE](#average-treatment-effects)

-   Selection on observables $X_i$ which might violate [Unconfoundedness]($(Y(1)%20,%20Y(0))%20\perp%20T%20%7C%20X$)

-   Selection on unobservables $U_i$ which might violate [Unconfoundedness]($(Y(1)%20,%20Y(0))%20\perp%20T%20%7C%20X$)

-   Interference across units (violate [SUTVA])

+------+-----------------------------------------------------------------------------+------------------------+-----------------------------------------------------------------------------+
|      | [T-learners]                                                                | [S-learners]           | [X-learners]                                                                |
+======+=============================================================================+========================+=============================================================================+
| Good | When there are no common trends in the response under control and treatment | When CATE is mostly 0. | CATE is mostly 0                                                            |
|      |                                                                             |                        |                                                                             |
|      | When the treatment effect is very complicated                               |                        | CATE is linear                                                              |
|      |                                                                             |                        |                                                                             |
|      |                                                                             |                        | When the number of units in one group is much larger than that of the other |
+------+-----------------------------------------------------------------------------+------------------------+-----------------------------------------------------------------------------+
| Bad  | When the treatment effect is simple                                         | Can be biased toward 0 | When CATE is 0, it's better than T, but worse than S                        |
|      |                                                                             |                        |                                                                             |
|      |                                                                             |                        | When CATE is complex, it's better than S, and T                             |
+------+-----------------------------------------------------------------------------+------------------------+-----------------------------------------------------------------------------+

Rule of thumb by [@künzel2019] is that unless you know that CATE is mostly 0, use [X-learners] with

-   BART in small data sets

-   RF in big data sets

These meta-learners algorithms were introduced by [@künzel2019] to estimate [Conditional Average Treatment Effects]

Since there is no package in `R` yet, the `causalml` module in `python` is an implementation of these learners' algorithms. [Examples](https://causalml.readthedocs.io/en/latest/examples.html)

##### T-learners

-   This is the most common meta-algorithm approach

-   T here just means two as in two-tree learner under tree-based methods.

Steps:

1.  Estimate the control response function $\mu_0(X) = E(Y(0) |X)$ using control units by either parametric or non-parametric methods
2.  Estimate the treatment response function $\mu_1(X) =E(Y(1)|X)$ using treated units by either parametric or non-parametric methods (can be different from the first step)
3.  Take the difference $\hat{\tau} = \hat{\mu}_1 (X) - \hat{\mu}_0 (X)$

##### S-learners

-   S here means single as single estimator

-   Consider treatment variable $T$ as another covariate in the feature matrix $X$

-   Use a single model to model

Steps:

1.  Estimate one combined response function $\mu(X,W) = E(Y^{obs}|X, W)$ using either parametric or non-parametric methods on the entire dataset
2.  Take the difference between predicted values when a unit takes treatment versus control: $\hat{\tau} = \hat{\mu}(X,1) - \hat{\mu}(X,0)$

##### X-learners

Good with

-   Unbalanced design (e.g., control or treatment group is much larger than the other one)

-   Prior knowledge of CATE structure (e.g., smoothness, sparsity, lots of 0, or approximately linear)

Steps:

1.  Estimate the control response function $\mu_0(X) = E(Y(0) |X)$ using control units by either parametric or non-parametric methods
2.  Estimate the treatment response function $\mu_1(X) =E(Y(1)|X)$ using treated units by either parametric or non-parametric methods (can be different from the first step)
3.  Impute the treatment effects for treated units based on the control outcome estimator $\tilde{D}_i^1 = Y_i^1 - \hat{\mu}_0(X_i^1)$
4.  Impute the treatment effects for control units based on the treatment outcome estimator $\tilde{D}_i^0 = \hat{\mu}_1 (X_i^0)- Y_i^0$
5.  Using the imputed treatment effects as the response variable in the treatment group to obtain $\hat{\tau}_1(X)$
6.  Using the imputed treatment effects as the response variable in the control group to obtain $\hat{\tau}_0(X)$
7.  Estimate CATE by the weighted average of the two estimates in 5 and 6: $\hat{\tau}(X) = g(X) \hat{\tau}_0(X) + (1- g(X)) \hat{\tau}_1(X)$ where $g \in [0,1]$

Now here you have to choice of $g$

1.  Use estimate of the propensity score
2.  Use $g=1$ if the number of treated units is very large compared to the number of control units
3.  Use $g=0$ if the number of treated units is very small compared to the number of control units

##### Other-learner

-   R-learner: [@nie2021quasi] uses the CV out-of-fold estimates of outcomes and propensity scores to minimize the R-loss function.

-   Doubly Robust (DR) learner [@kennedy2020optimal] uses a doubly robust score function to estimate the CATE

-   Doubly Robust Instrumental Variable (DRIV) learner combines the DR-learner with doubly robust score function for LATE to estimate conditional LATE [@chernozhukov2018]

##### Robust Estimation of Heterogeneous Treatment Effect

Assume that we have [SUTVA] and [Unconfoundedness]($(Y(1)%20,%20Y(0))%20\perp%20T%20%7C%20X$)

This section studies how treatment effects vary across units. But knowing the individual treatment effect is impossible. Hence, we move our goal to understand how treatment effect vary across group of units (i.e., units that share the same covariate value)

#### Intent-to-treat Effects

When we encounter non-compliance (either people suppose to receive treatment don't receive it, or people suppose to be in the control group receive the treatment), treatment receipt is not independent of potential outcomes and confounders.

In this case, the difference in observed means between the treatment and control groups is not [Average Treatment Effects](#average-treatment-effects), but [Intent-to-treat Effects] (ITT). In words, ITT is the treatment effect on those who **receive** the treatment

<br>

#### Local Average Treatment Effects

Instead of estimating the treatment effects of those who **receive** the treatment (i.e., [Intent-to-treat Effects]), you want to estimate the treatment effect of those who actually **comply** with the treatment. This is the local average treatment effects (LATE) or complier average causal effects (CACE). I assume we don't use CATE to denote complier average treatment effect because it was reserved for conditional average treatment effects.

-   Using random treatment assignment as an instrument, we can recover the effect of treatment on compliers.

![](images/iv_late.PNG){width="100%"}

-   As the percent of compliers increases, [Intent-to-treat Effects] and [Local Average Treatment Effects] converge

-   Rule of thumb: SE(LATE) = SE(ITT)/(share of compliers)

-   LATE estimate is always greater than the ITT estimate

-   LATE can also be estimated using a pure placebo group [@gerber2010].

-   Partial compliance is hard to study, and IV/2SLS estimator is biased, we have to use Bayesian [@long2010; @jin2009; @jin2008].

##### One-sided noncompliance

-   One-sided noncompliance is when in the sample, we only have compliers and never-takers

-   With the exclusion restriction (i.e., excludability), never-takers have the same results in the treatment or control group (i.e., never treated)

-   With random assignment, we can have the same number of never-takers in the treatment and control groups

-   Hence,

$$
LATE = \frac{ITT}{\text{share of compliers}}
$$

##### Two-sided noncompliance

-   Two-sided noncompliance is when in the sample, we have compliers, never-takers, and always-takers

-   To estimate LATE, beyond excludability like in the [One-sided noncompliance] case, we need to assume that there is no defiers (i.e., monotonicity assumption) (this is excusable in practical studies)

$$
LATE = \frac{ITT}{\text{share of compliers}}
$$

#### Population vs. Sample Average Treatment Effects

See [@imai2008] for when the sample average treatment effect (SATE) diverges from the population average treatment effect (PATE).

To stay consistent, this section uses notations from [@imai2008]'s paper.

In a finite population $N$, we observe $n$ observations ($N>>n$), where half is in the control and half is in the treatment group.

With unknown data generating process, we have

$$
I_i = 
\begin{cases}
1 \text{ if unit i is in the sample} \\
0 \text{ otherwise}
\end{cases}
$$

$$
T_i = 
\begin{cases}
1 \text{ if unit i is in the treatment group} \\
0 \text{ if unit i is in the control group}
\end{cases}
$$

$$
\text{potential outcome} = 
\begin{cases}
Y_i(1) \text{ if } T_i = 1 \\
Y_i(0) \text{ if } T_i = 0
\end{cases}
$$

Observed outcome is

$$
Y_i | I_i = 1= T_i Y_i(1) + (1-T_i)Y_i(0)
$$

Since we can never observed both outcome for the same individual, the treatment effect is always unobserved for unit $i$

$$
TE_i = Y_i(1) - Y_i(0)
$$

Sample average treatment effect is

$$
SATE = \frac{1}{n}\sum_{i \in \{I_i = 1\}} TE_i
$$

Population average treatment effect is

$$
PATE = \frac{1}{N}\sum_{i=1}^N TE_i
$$

Let $X_i$ be observables and $U_i$ be unobservables for unit $i$

The baseline estimator for SATE and PATE is

$$
\begin{aligned}
D &= \frac{1}{n/2} \sum_{i \in (I_i = 1, T_i = 1)} Y_i - \frac{1}{n/2} \sum_{i \in (I_i = 1 , T_i = 0)} Y_i \\
&= \text{observed sample mean of the treatment group} \\
&- \text{observed sample mean of the control group}
\end{aligned}
$$

Let $\Delta$ be the estimation error (deviation from the truth), under an additive model

$$
Y_i(t) = g_t(X_i) + h_t(U_i)
$$

The decomposition of the estimation error is

$$
\begin{aligned}
PATE - D = \Delta &= \Delta_S + \Delta_T \\
&= (PATE - SATE) + (SATE - D)\\
&= \text{sample selection}+ \text{treatment imbalance} \\
&= (\Delta_{S_X} + \Delta_{S_U}) + (\Delta_{T_X} + \Delta_{T_U}) \\
&= \text{(selection on observed + selection on unobserved)} \\
&+ (\text{treatment imbalance in observed + unobserved})
\end{aligned}
$$

##### Estimation Error from Sample Selection

Also known as sample selection error

$$
\Delta_S = PATE - SATE = \frac{N - n}{N}(NATE - SATE)
$$

where NATE is the non-sample average treatment effect (i.e., average treatment effect for those in the population but not in your sample:

$$
NATE = \sum_{i\in (I_i = 0)} \frac{TE_i}{N-n}
$$

From the equation, to have zero sample selection error (i.e., $\Delta_S = 0$), we can either

-   Get $N = n$ by redefining your sample as the population of interest

-   $NATE = SATE$ (e.g., $TE_i$ is constant over $i$ in both your selected sample, and those in the population that you did not select)

Note

-   When you have heterogeneous treatment effects, **random sampling** can only warrant **sample selection bias**, not **sample selection error**.

-   Since we can rarely know the true underlying distributions of the observables ($X$) and unobservables ($U$), we cannot verify whether the empirical distributions of your observables and unobservables for those in your sample is identical to that of your population (to reduce $\Delta_S$). For special case,

    -   Say you have census of your population, you can adjust for the observables $X$ to reduce $\Delta_{S_X}$, but still you cannot adjust your unobservables ($U$)

    -   Say you are willing to assume $TE_i$ is constant over

        -   $X_i$, then $\Delta_{S_X} = 0$

        -   $U_i$, then $\Delta_{U}=0$

##### Estimation Error from Treatment Imbalance

Also known as treatment imbalance error

$$
\Delta_T = SATE - D
$$

$\Delta_T \to 0$ when treatment and control groups are balanced (i.e., identical empirical distributions) for both observables ($X$) and unobservables ($U$)

However, in reality, we can only readjust for observables, not unobservables.

+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                             | [Randomized Block Designs]                                                                                                                                                       | **[Matching Methods]**                                                                                                                                                                                                                                                                                                                                                   |
+=============================================================+==================================================================================================================================================================================+==========================================================================================================================================================================================================================================================================================================================================================================+
| Definition                                                  | Random assignment within strata based on pre-treatment observables                                                                                                               | Dropping, repeating or grouping observations to balance covariates between the treatment and control group [@rubin1973]                                                                                                                                                                                                                                                  |
+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Time                                                        | Before randomization of treatments                                                                                                                                               | After randomization of treatments                                                                                                                                                                                                                                                                                                                                        |
+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| What if the set of covariates used to adjust is irrelevant? | Nothing happens (still caution by [@pashley2021])                                                                                                                                | In the worst case scenario (e.g., these variables are uncorrelated with the treatment assignment, but correlated with the post-treatment variables), matching induces bias that is greater than just using the unadjusted difference in means                                                                                                                            |
|                                                             |                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                          |
| [@imai2008, p. 489, para. 4]                                |                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                          |
+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Benefits                                                    | $\Delta_{T_X}=0$ (no imbalance on observables). But we don't know its effect on unobservables imbalance (might reduce if the unobservables are correlated with the observables). | Reduce model dependence, bias, variance, mean-square error [@ho2007]                                                                                                                                                                                                                                                                                                     |
|                                                             |                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                          |
|                                                             | Increase efficiency because it achieves $\Delta_{T_x}=0$, instead of $E(\Delta_{T_X}) = 0$ if we just use random assignment.                                                     |                                                                                                                                                                                                                                                                                                                                                                          |
+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Dangers                                                     | No hypothesis can check for balance within strata. You can only check for how your randomization was [@imai2008, p. 494, para. 3]                                                | Hypothesis test on covariates balance between treated and control groups should not be trusted because these tests (e.g., t-test K-S test) are based on not only the balance between the two groups, but also on the statistical power (function of your sample size) [@imai2008, p. 495-498]. As you drop observations, your statistical power monotonically decreases. |
|                                                             |                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                          |
|                                                             |                                                                                                                                                                                  | [@imai2008, p. 498] advocate for non-parametric density plots, porepensity score summary statistics [@austin2006] [@rubin], and quantile-quantile plots                                                                                                                                                                                                                  |
|                                                             |                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                          |
|                                                             |                                                                                                                                                                                  | Matching on irrelevant variables is okay (small increase in variance). [@ho2007]                                                                                                                                                                                                                                                                                         |
+-------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
|                                                                                     | Observables Sample selection estimation error ($\Delta_{S_X}$) | Unobservables Sample selection estimation error ($\Delta_{S_U}$) | Observables treatment imbalance estimation error ($\Delta_{T_X}$) | Unobservables Treatment imbalance estimation error ($\Delta_{T_U}$) |
+=====================================================================================+================================================================+==================================================================+===================================================================+=====================================================================+
| Design Feature                                                                      |                                                                |                                                                  |                                                                   |                                                                     |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| Random Sampling                                                                     | $E(\Delta_{S_X}) = 0$                                          | $E(\Delta_{S_U}) =0$                                             |                                                                   |                                                                     |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| SATE as quantity of interest (not PATE)                                             | $\Delta_{S_X}=0$                                               | $\Delta_{S_U}=0$                                                 |                                                                   |                                                                     |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| Weighting for non-random sampling                                                   | $\Delta_{S_X}=0$                                               | $\Delta_{S_u}=?$                                                 |                                                                   |                                                                     |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| $n\to \infty$                                                                       | $E(\Delta_{S_X}) = 0$                                          | $E(\Delta_{S_U}) = 0$                                            | $E(\Delta_{T_X}) = 0$                                             | $E(\Delta_{T_U}) = 0$                                               |
|                                                                                     |                                                                |                                                                  |                                                                   |                                                                     |
|                                                                                     | $\lim_{n \to \infty}(var(\Delta_{S_X}) = 0$                    | $\lim_{n \to \infty}(var(\Delta_{S_U}) = 0$                      | $\lim_{n \to \infty}(var(\Delta_{T_X}) = 0$                       | $\lim_{n \to \infty}(var(\Delta_{T_U}) = 0$                         |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| Random assignment                                                                   |                                                                |                                                                  | $E(\Delta_{T_X}) =0$                                              | $E(\Delta_{T_U})=0$                                                 |
|                                                                                     |                                                                |                                                                  |                                                                   |                                                                     |
| $E(\Delta_{T_X}) = E(\Delta_{T_U}) =0$                                              |                                                                |                                                                  |                                                                   |                                                                     |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| [Randomized Block Designs]                                                          |                                                                |                                                                  | $\Delta_{T_X}=0$                                                  | $\Delta_{T_U}=?$                                                    |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| [Exact Matching][Coarsened Exact Matching] (for studies that estimate PATT or SATT) |                                                                |                                                                  | $\Delta_{T_X}=0$                                                  | $\Delta_{T_U}=?$                                                    |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
|                                                                                     |                                                                |                                                                  |                                                                   |                                                                     |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| Assumption                                                                          |                                                                |                                                                  |                                                                   |                                                                     |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| No Selection Bias                                                                   | $E(\Delta_{S_X}) =0$                                           | $E(\Delta_{S_U})=0$                                              |                                                                   |                                                                     |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| Ignorability                                                                        |                                                                |                                                                  |                                                                   | $E(\Delta_{T_U}) = 0$                                               |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+
| No omitted variables ($X\perp U$ or $U | X \nrightarrow Y$)                         |                                                                |                                                                  |                                                                   | $\Delta_{T_U}=0$                                                    |
+-------------------------------------------------------------------------------------+----------------------------------------------------------------+------------------------------------------------------------------+-------------------------------------------------------------------+---------------------------------------------------------------------+

: Effects of design choices and assumptions on estimation error

This table was modified from Table 1 in [@imai2008, p. 488]

#### Average Treatment Effects on the Treated and Control

Average Effect of treatment on the Treated (ATT) is

$$
\begin{aligned}
ATT &= E(Y_i(1) - Y_i(0)|D_i = 1) \\
&= E(Y_i(1)|D_i = 1) - E(Y_i(0) |D_i = 1)
\end{aligned}
$$

Average Effect of treatment on the Control (ATC) (i.e., the effect **would be** for those weren't treated) is

$$
\begin{aligned}
ATC &= E(Y_i(1) - Y_i (0) |D_i =0) \\
&= E(Y_i(1)|D_i = 0) - E(Y_i(0)|D_i = 0)
\end{aligned}
$$

Under random assignment and full compliance,

$$
ATE = ATT = ATC
$$

**Sample average treatment effect on the treated** is

$$
SATT = \frac{1}{n} \sum_i TE_i
$$

where

-   $TE_i$ is the treatment effect for unit $i$

-   $n$ is the number of treated units in the sample

-   $i$ belongs the subset (i.e., sample) of the population of interest that is treated.

**Population average treatment effect on the treated** is

$$
PATT = \frac{1}{N} \sum_i TE_i
$$

where

-   $TE_i$ is the treatment effect for unit $i$

-   $N$ is the number of treated units in the population

-   $i$ belongs to the population of interest that is treated.

#### Quantile Average Treatment Effects

Instead of the middle point estimate (ATE), we can also understand the changes in the distribution the outcome variable due to the treatment.

Using quantile regression and more assumptions [@abadie2002; @chernozhukov2005], we can have consistent estimate of quantile treatment effects (QTE), with which we can make inference regarding a given quantile.

#### Mediation Effects

With additional assumptions (i.e., sequential ignorability [@Imai_2010_6060; @bullock2010]), we can examine the mechanism of the treatment on the outcome.

Under the causal framework,

-   the indirect effect of treatment via a mediator is called average causal mediation effect (ACME)

-   the direct effect of treatment on outcome is the average direct effect (ADE)

More in the [Mediation] Section \@ref(mediation)

#### Log-odds Treatment Effects

For binary outcome variable, we might be interested in the log-odds of success. See [@freedman2008] on how to estimate a consistent causal effect.

Alternatively, attributable effects [@rosenbaum2002] can also be appropriate for binary outcome.

## Controls under causal inference

Under prediction, adding as many controls as you like is fine. And you can always be better off with some more advanced machine learning methods (which is covered in the second version of this book - [Advanced Data Analysis](https://bookdown.org/mike/advanced_data_analysis/)).

However, under causal inference, there are good controls and there are bad controls.

![green = good; red = bad](images/bad_control.PNG){alt="green = good; red = bad" width="100%"}

In the perfect world, we only need to randomize the treatment and care only about the treatment and control. But since we don't live in such a world, we need to be careful to whether you want to include or exclude certain variables.

From the DAGs literature, controls are determined based on assessing the path between $X \to Y$

-   Good controls: Block all spurious paths between $X \to Y$

-   Bad controls: Do not block any causal paths between $X \to Y$

Technically there are 3 types of control:

-   Good control: ATE bias reduction

-   Bad control: ATE bias increase

-   Neutral control: does not change the asymptotic bias

Code for numerical simulations are taken from [Carlos Cinelli](https://www.kaggle.com/code/carloscinelli/crash-course-in-good-and-bad-controls-linear-r/notebook)'s notebook

### Good Controls

Those that are added to eliminate omitted variable bias and are unaffected by the treatment

Two types of good controls:

1.  Confounders (i.e., those that affect both the treatment and outcome).
2.  Those that can help predict outcome

Good controls help explain variability in the outcome variable (i.e., account for variability in the outcome variable). In statistical terms, they help predict the outcome variable and increase $R^2$ under OLS regression. This way we can think that conditioning on the control variables, it's easier to detect the effect of your causal variable.

Hence, always add control variables that help predict the outcome variables, even if they are not confounders.


```r
# Load packages
library(dagitty)
library(ggdag)
```


```r
# cleans workspace
rm(list = ls())

# DAG

## specify edges
model <- dagitty("dag{x->y; z->x; z->y}")

## coordinates for plotting
coordinates(model) <-  list(
  x = c(x=1, y=3, z=2),
  y = c(x=1, y=1, z=2))

## ggplot
ggdag(model) + theme_dag()
```

<img src="18-causality_files/figure-html/unnamed-chunk-2-1.png" width="90%" style="display: block; margin: auto;" />

### Bad Controls

1.  Mediator 1: those are good predictors of the treatment (not outcome)
    -   If you control for both the confounder and mediator 1, variance of treatment causal estimate will increase (more likely to contain 0). Hence, it will make it harder for your model to detect the effect of the treatment on the outcome.

    -   In this case, regressing the model with only the confounder and the treatment on the outcome is sufficient.
2.  Mediator 2: those that are mediators between the treatment and outcome
    -   Recall from [Intro to DAG Framework] that if we close the path between treatment and outcome by conditioning on this mediator, then treatment and outcome will be independent.
3.  Common effect: those are the joint consequences of both the treatment and outcomes

Mediator 2 and common effect are similar in the sense that they are basic [Mediation] analysis (i.e., $X \to M \to Y$). But in the [Mediation] analysis, we have to readjust the standard error for the treatment $X$. However, if you just include these two variables into your model as controls, your treatment effect's standard error will explode

## Sensitivity Analysis

Recommended by [@athey2017], we should report

-   Simple difference

-   OLS estimator

-   Double selection estimator (DSE)

-   Approximate residual balancing estimator (ARBE)

-   Doubly robust estimator (DRE)

-   Double machine learning estimator (DMLE)

for standard errors:

-   simple bootstrap SE

-   sclaed bootstrap bias
