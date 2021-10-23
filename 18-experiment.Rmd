# Experimental Design

-   Randomized Control Trials (RCT) or Experiments have always been and are likely to continue in the future to be the holy grail of causal inference.
-   RCT means you have two group treatment (or experimental) gorp and control group. Hence, as you introduce the treatment (your exogenous variable) to the treatment group, the only expected difference in the outcomes of the two group should be due to the treatment.
-   Subjects from the same population will be **randomly assigned** to either treatment or control group. This random assignment give us the confidence that changes in the outcome variable will be due only the treatment, not any other source (variable).
-   It can be easier for hard science to have RCT because they can introduce the treatment, and have control environments. But it's hard for social scientists because their subjects are usually human, and some treatment can be hard to introduce, or environments are uncontrollable. Hence, social scientists have to develop different tools ([Quasi-experimental]) to recover causal inference or to recreate the treatment and control group environment.
-   With RCT, you can easily establish internal validity

Selection Problem

Assume we have

-   binary treatment $D_i =(0,1)$

-   outcome of interest $Y_i$ for individual $i$

    -   $Y_{0i}$ are those were **not treated**

    -   $Y_{1i}$ are those were **treated**

$$
\text{Potential Outcome} =
\begin{cases}
Y_{1i} \text{ if } D_i = 1 \\
Y_{0i} \text{ if } D_i = 0
\end{cases}
$$

Then, what we observe in the outcome variable is

$$
Y_i = Y_{0i} + (Y_{1i} - Y_{0i})D_i
$$

It's likely that $Y_{1i}$ and $Y_{0i}$ both have their own distributions (i.e., different treatment effect for different people). Since we can't see both outcomes for the same individual (unless we have a time machine), then we can only make inference regarding the average outcome of those who were treated and those who were not.

$$
\begin{aligned}
E[Y_i | D_i = 1] - E[Y_i | D_i = 0] &= (E[Y_{1i} | D_i = 1] - E[Y_{0i}|D_i = 1] ) + (E[Y_{0i} |D_i = 1] - E[Y_{0i} |D_i = 0]) \\
 &= (E[Y_{1i}-Y_{0i}|D_1 = 1] ) + (E[Y_{0i} |D_i = 1] - E[Y_{0i} |D_i = 0]) \\
\text{Observed difference in treatment} &= \text{Average treatment effect on the treated} + \text{Selection bias}
\end{aligned}
$$

-   **The average treatment effect** is the average between between a person who is treated and the same person (in another parallel universe) who is not treated

-   **The selection bias** is the difference between those who were treated and those who weren't treated

With **random assignment** of treatment ($D_i$) under [Experimental Design], we can have $D_i$ independent of potential outcomes

$$
\begin{aligned}
E[Y_i | D_i = 1] - E[Y_i|D_i = 0] &= E[Y_{1i}|D_i = 1]-E[Y_{0i}|D_i = 0)]\\
&= E[Y_{1i}|D_i = 1]-E[Y_{0i}|D_i = 0)] && D_i \perp Y_i \\
&= E[Y_{1i} - Y_{0i}|D_i = 1] \\
&= E[Y_{1i} - Y_{0i}]
\end{aligned}
$$

<br>

Another representation under regression

Suppose that you know the effect is

$$
Y_{1i} - Y_{0i} = \rho
$$

The observed outcome variable (for an individual) can be rewritten as

$$
\begin{aligned}
Y_i &= E(Y_{0i}) + (Y_{1i}-Y_{0i})D_i + [Y_{0i} - E(Y_{0i})]\\
&= \alpha + \rho D_i + \eta_i
\end{aligned}
$$

where $\eta_i$ = random variation of $Y_{0i}$

Hence, the conditional expectation of an individual outcome on treatment status is

$$
\begin{aligned}
E[Y_i |D_i = 1] &= \alpha + \rho &+ E[\eta_i |D_i = 1] \\
E[Y_i |D_i = 0] &= \alpha &+ E[\eta_i |D_i = 0]
\end{aligned}
$$

Thus,

$$
E[Y_i |D_i = 1] - E[Y_i |D_i = 0] = \rho + E[\eta_i |D_i = 1] -E[\eta_i |D_i = 0]
$$

where $E[\eta_i |D_i = 1] -E[\eta_i |D_i = 0]$ is the selection bias - correlation between the regression error term ($\eta_i$), and the regressor ($D_i$)

Under regression, we have

$$
E[\eta_i |D_i = 1] -E[\eta_i |D_i = 0] = E[Y_{0i} |D_i = 1] -E[Y_{0i}|D_i = 0]
$$

which is the difference in outcomes between **those who weren't treated get treated** and **those who weren't treated stay untreated**

Say you have control variables ($X_i$), that is **uncorrelated with the treatment** ($D_i$), then you can include in your model, and it won't (in principle) affect your estimate of the treatment effect ($\rho$) with an added benefit of reducing the residual variance, which subsequently reduces the standard error of other estimates.

$$
Y_i = \alpha + \rho D_i + X_i'\gamma + \eta_i
$$
