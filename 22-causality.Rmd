# Causal Inference

After all of the mumbo jumbo that we have learned so far, I want to now talk about the concept of causality. We usually say that correlation is not causation. Then, what is causation?\
One of my favorite books has explained this concept beautifully [@Pearl_2018]. And I am just going to quickly summarize the gist of it from my understanding. I hope that it can give you an initial grasp on the concept so that later you can continue to read up and develop a deeper understanding.

It's important to have a deep understanding regarding the method research. However, one needs to be aware of its limitation. As mentioned in various sections throughout the book, we see that we need to ask experts for number as our baseline or visit literature to gain insight from past research.

Here, we dive in a more conceptual side statistical analysis as a whole, regardless of particular approach.

You probably heard scientists say correlation doesn't mean causation. There are ridiculous [spurious correlations](http://www.tylervigen.com/spurious-correlations) that give a firm grip on what the previous phrase means. Statistics is used to be a causality-free enterprise in the past. Not until the development of path analysis by Sewall Wright in the 1920s that the discipline started to pay attention to causation. Then, it remained dormant until the Causal Revolution (quoted Judea Pearl's words). This revolution introduced the calculus of causation which includes (1) causal diagrams), and (2) a symbolic language

The world has been using $P(Y|X)$ (statistics use to derive this), but what we want is to compare the difference between

-   $P(Y|do(X))$: treatment group

-   $P(Y|do(not-X))$: control group

Hence, we can see a clear difference between $P(Y|X) \neq P(Y|do(X))$

The conclusion we want to make from data is counterfactuals: **What would have happened had we not do X?**

To teach a robot to make inference, we need inference engine

![p. 12 [@Pearl_2018]](images/Figure%20I.png "Inference Engine")

Levels of cognitive ability to be a causal learner:

1.  Seeing
2.  Doing
3.  Imagining

<br>

Simpson's Paradox:

-   A statistical association seen in an entire population is reversed in sub-population.

Structural Causal Model accompanies graphical causal model to create more efficient language to represent causality

<br>

Tools in a hierarchical order

1.  [Experimental Design]: Randomized Control Trials (Gold standard): Tier 1

2.  [Quasi-experimental]

    1.  [Regression Discontinuity] Tier 1A

    2.  [Difference-In-Differences] Tier 2

    3.  [Synthetic Control] Tier 2A

    4.  Fixed Effects Estimator: Tier 3

    5.  [Endogenous Treatment]: mostly [Instrumental Variable]: Tier 3A

    6.  [Matching Methods] Tier 4

    7.  [Interrupted Time Series] Tier 4A

    8.  Endogenous Sample Selection: mostly Heckman's correction

<br>

This section is based on Bernard Koch's [presentaiton](https://www.youtube.com/watch?v=v9uf9rDYEMg&ab_channel=SummerInstituteinComputationalSocialScience) at SICSS - Los Angeles 2021

Identification under Selection on observable/ back-door criterion

Conditions:

-   Strong conditional ignorability

    -   $Y(0),Y(1) \perp T|X$

    -   No hidden confounders

-   Overlap

    -   $\forall x \in X, t \in \{0, 1\}: p (T = t | X = x> 0$

    -   All treatments have non-zero probability of being observed

-   SUTVA/ Consistency

    -   Treatment and outcomes of different subjects are independent

<br>

Example

We have

-   binary treatment $T \in \{ 0,1\}$

-   $Y(1), Y(0)$ are the potential outcomes

-   The average treatment effect is

$$
ATE = E(Y(1) - Y(0)) = E(\tau(x))
$$

-   The conditional average treatment effect is

$$
CATE = \tau(x) = E(Y(1) - Y(0)|X = x)
$$

see <https://github.com/maxwshen/iap-cidl/blob/master/iap-cidl-lecture1_fredrik_potential_outcomes.pdf>
