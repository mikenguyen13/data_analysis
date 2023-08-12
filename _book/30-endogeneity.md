# Endogeneity

Refresher

A general model framework

$$
\mathbf{Y = X \beta + \epsilon}
$$

where

-   $\mathbf{Y} = n \times 1$

-   $\mathbf{X} = n \times k$

-   $\beta = k \times 1$

-   $\epsilon = n \times 1$

Then, OLS estimates of coefficients are

$$
\begin{aligned}
\hat{\beta}_{OLS} &= (\mathbf{X}'\mathbf{X})^{-1}(\mathbf{X}'\mathbf{Y}) \\
&= (\mathbf{X}'\mathbf{X})^{-1}(\mathbf{X}'(\mathbf{X \beta + \epsilon})) \\
&= (\mathbf{X}'\mathbf{X})^{-1} (\mathbf{X}'\mathbf{X}) \beta + (\mathbf{X}'\mathbf{X})^{-1} (\mathbf{X}'\mathbf{\epsilon}) \\
\hat{\beta}_{OLS} & \to \beta + (\mathbf{X}'\mathbf{X})^{-1} (\mathbf{X}'\mathbf{\epsilon})
\end{aligned}
$$

To have unbiased estimates, we have to get rid of the second part $(\mathbf{X}'\mathbf{X})^{-1} (\mathbf{X}'\mathbf{\epsilon})$

There are 2 conditions to achieve unbiased estimates:

1.  $E(\epsilon |X) = 0$ (This is easy, putting an intercept can solve this issue)
2.  $Cov(\mathbf{X}, \epsilon) = 0$ (This is the hard part)

We only care about omitted variable

Usually, the problem will stem Omitted Variables Bias, but we only care about omitted variable bias when

1.  Omitted variables correlate with the variables we care about ($X$). If OMV does not correlate with $X$, we don't care, and random assignment makes this correlation goes to 0)
2.  Omitted variables correlates with outcome/ dependent variable

There are more types of endogeneity listed below.

<br>

Types of endogeneity

1.  [Endogenous Treatment]

-   Omitted Variables Bias

    -   Motivation/choiceheck
    -   Ability/talent
    -   Self-selection

-   Feedback Effect ([Simultaneity]): also known as bidirectionality

-   Reverse Causality: Substle difference from [Simultaneity]: Technically, two variables affect each other sequentially, but in a big enough time frame, (e.g., monthly, or yearly), our coefficient will be biased just like simultaneity.

-   [Measurement Error]

2.  [Endogenous Sample Selection]

<br>

To deal with this problem, we have a toolbox (that has been mentioned in previous chapter \@ref(causal-inference) )

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

<br>

Using control variables in regression is a "selection on observables" identification strategy.

In other words, if you believe you have an omitted variable, and you can measure it, including it in the regression model solves your problem. These uninterested variables are called control variables in your model.

However, this is rarely the case (because the problem is we don't have their measurements). Hence, we need more elaborate methods:

-   [Endogenous Treatment]

-   [Endogenous Sample Selection]

Before we get to methods that deal with bias arises from omitted variables, we consider cases where we do have measurements of a variable, but there is measurement error (bias).

<br>

## Measurement Error

-   Data error can stem from

    -   Coding errors

    -   Reporting errors

Two forms of measurement error:

1.  Random (stochastic) (indeterminate error) ([Classical Measurement Errors]): noise or measurement errors do not show up in a consistent or predictable way.
2.  Systematic (determinate error) ([Non-classical Measurement Errors]): When measurement error is consistent and predictable across observations.
    1.  Instrument errors (e.g., faulty scale) -\> calibration or adjustment
    2.  Method errors (e.g., sampling errors) -\> better method development + study design
    3.  Human errors (e.g., judgement)

Usually the systematic measurement error is a bigger issue because it introduces "bias" into our estimates, while random error introduces noise into our estimates

-   Noise -\> regression estimate to 0
-   Bias -\> can pull estimate to upward or downward.

### Classical Measurement Errors

#### Right-hand side

-   Right-hand side measurement error: When the measurement is in the covariates, then we have the endogeneity problem.

Say you know the true model is

$$
Y_i = \beta_0 + \beta_1 X_i + u_i
$$

But you don't observe $X_i$, but you observe

$$
\tilde{X}_i = X_i + e_i
$$

which is known as classical measurement errors where we **assume** $e_i$ is uncorrelated with $X_i$ (i.e., $E(X_i e_i) = 0$)

Then, when you estimate your observed variables, you have (substitute $X_i$ with $\tilde{X}_i - e_i$ ):

$$
\begin{aligned}
Y_i &= \beta_0 + \beta_1 (\tilde{X}_i - e_i)+ u_i \\
&= \beta_0 + \beta_1 \tilde{X}_i + u_i - \beta_1 e_i \\
&= \beta_0 + \beta_1 \tilde{X}_i + v_i
\end{aligned}
$$

In words, the measurement error in $X_i$ is now a part of the error term in the regression equation $v_i$. Hence, we have an endogeneity bias.

Endogeneity arises when

$$
\begin{aligned}
E(\tilde{X}_i v_i) &= E((X_i + e_i )(u_i - \beta_1 e_i)) \\
&= -\beta_1 Var(e_i) \neq 0
\end{aligned}
$$

Since $\tilde{X}_i$ and $e_i$ are positively correlated, then it leads to

-   a negative bias in $\hat{\beta}_1$ if the true $\beta_1$ is positive

-   a positive bias if $\beta_1$ is negative

In other words, measurement errors cause **attenuation bias**, which inter turn pushes the coefficient towards 0

As $Var(e_i)$ increases or $\frac{Var(e_i)}{Var(\tilde{X})} \to 1$ then $e_i$ is a random (noise) and $\beta_1 \to 0$ (random variable $\tilde{X}$ should not have any relation to $Y_i$)

Technical note:

The size of the bias in the OLS-estimator is

$$
\hat{\beta}_{OLS} = \frac{ cov(\tilde{X}, Y)}{var(\tilde{X})} = \frac{cov(X + e, \beta X + u)}{var(X + e)}
$$

then

$$
plim \hat{\beta}_{OLS} = \beta \frac{\sigma^2_X}{\sigma^2_X + \sigma^2_e} = \beta \lambda
$$

where $\lambda$ is **reliability** or signal-to-total variance ratio or attenuation factor

Reliability affect the extent to which measurement error attenuates $\hat{\beta}$. The attenuation bias is

$$
\hat{\beta}_{OLS} - \beta = -(1-\lambda)\beta
$$

Thus, $\hat{\beta}_{OLS} < \beta$ (unless $\lambda = 1$, in which case we don't even have measurement error).

Note:

**Data transformation worsen (magnify) the measurement error**

$$
y= \beta x + \gamma x^2 + \epsilon
$$

then, the attenuation factor for $\hat{\gamma}$ is the square of the attenuation factor for $\hat{\beta}$ (i.e., $\lambda_{\hat{\gamma}} = \lambda_{\hat{\beta}}^2$)

<br>

**Adding covariates increases attenuation bias**

To fix classical measurement error problem, we can

1.  Find estimates of either $\sigma^2_X, \sigma^2_\epsilon$ or $\lambda$ from validation studies, or survey data.
2.  [Endogenous Treatment] Use instrument $Z$ correlated with $X$ but uncorrelated with $\epsilon$
3.  Abandon your project

<br>

#### Left-hand side

When the measurement is in the outcome variable, econometricians or causal scientists do not care because they still have an unbiased estimate of the coefficients (the zero conditional mean assumption is not violated, hence we don't have endogeneity). However, statisticians might care because it might inflate our uncertainty in the coefficient estimates (i.e., higher standard errors).

$$
\tilde{Y} = Y + v
$$

then the model you estimate is

$$
\tilde{Y} = \beta X + u + v
$$

Since $v$ is uncorrelated with $X$, then $\hat{\beta}$ is consistently estimated by OLS

If we have measurement error in $Y_i$, it will pass through $\beta_1$ and go to $u_i$

<br>

### Non-classical Measurement Errors

Relaxing the assumption that $X$ and $\epsilon$ are uncorrelated

Recall the true model we have true estimate is

$$
\hat{\beta} = \frac{cov(X + \epsilon, \beta X + u)}{var(X + \epsilon)}
$$

then without the above assumption, we have

$$
\begin{aligned}
plim \hat{\beta} &= \frac{\beta (\sigma^2_X + \sigma_{X \epsilon})}{\sigma^2_X + \sigma^2_\epsilon + 2 \sigma_{X \epsilon}} \\
&= (1 - \frac{\sigma^2_{\epsilon} + \sigma_{X \epsilon}}{\sigma^2_X + \sigma^2_\epsilon + 2 \sigma_{X \epsilon}}) \beta \\
&= (1 - b_{\epsilon \tilde{X}}) \beta
\end{aligned}
$$

where $b_{\epsilon \tilde{X}}$ is the covariance between $\tilde{X}$ and $\epsilon$ (also the regression coefficient of a regression of $\epsilon$ on $\tilde{X}$)

Hence, the [Classical Measurement Errors] is just a special case of [Non-classical Measurement Errors] where $b_{\epsilon \tilde{X}} = 1 - \lambda$

So when $\sigma_{X \epsilon} = 0$ ([Classical Measurement Errors]), increasing this covariance $b_{\epsilon \tilde{X}}$ increases the covariance increases the attenuation factor if more than half of the variance in $\tilde{X}$ is measurement error, and decreases the attenuation factor otherwise. This is also known as **mean reverting measurement error** [@bound1989]

A general framework for both right-hand side and left-hand side measurement error is [@bound1994]:

consider the true model

$$
\mathbf{Y = X \beta + \epsilon}
$$

then

$$
\begin{aligned}
\hat{\beta} &= \mathbf{(\tilde{X}' \tilde{X})^{-1}\tilde{X} \tilde{Y}} \\
&= \mathbf{(\tilde{X}' \tilde{X})^{-1} \tilde{X}' (\tilde{X} \beta - U \beta + v + \epsilon )} \\
&= \mathbf{\beta + (\tilde{X}' \tilde{X})^{-1} \tilde{X}' (-U \beta + v + \epsilon)} \\
plim \hat{\beta} &= \beta + plim (\tilde{X}' \tilde{X})^{-1} \tilde{X}' ( -U\beta + v) \\
&= \beta + plim (\tilde{X}' \tilde{X})^{-1} \tilde{X}' W 
\left[
\begin{array}
{c}
- \beta \\
1
\end{array}
\right]
\end{aligned}
$$

Since we collect the measurement errors in a matrix $W = [U|v]$, then

$$
( -U\beta + v) = W 
\left[
\begin{array}
{c}
- \beta \\
1
\end{array}
\right]
$$

Hence, in general, biases in the coefficients $\beta$ are regression coefficients from regressing the measurement errors on the mis-measured $\tilde{X}$

Notes:

-   [Instrumental Variable] can help fix this problem

-   There can also be measurement error in dummy variables and you can still use [Instrumental Variable] to fix it.

### Solution to Measurement Errors

#### Correlation

$$
\begin{aligned}
P(\rho | data) &= \frac{P(data|\rho)P(\rho)}{P(data)} \\
\text{Posterior Probability} &\propto \text{Likelihood} \times \text{Prior Probability}
\end{aligned}
$$ where

-   $\rho$ is a correlation coefficient
-   $P(data|\rho)$ is the likelihood function evaluated at $\rho$
-   $P(\rho)$ prior probability
-   $P(data)$ is the normalizing constant

With sample correlation coefficient $r$:

$$
r = \frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}}
$$ Then the posterior density approximation of $\rho$ is [@schisterman2003estimation, pp.3]

$$
P(\rho| x, y)  \propto P(\rho) \frac{(1- \rho^2)^{(n-1)/2}}{(1- \rho \times r)^{n - (3/2)}}
$$

where

-   $\rho = \tanh \xi$ where $\xi \sim N(z, 1/n)$
-   $r = \tanh z$

Then the posterior density follow a normal distribution where

**Mean**

$$
\mu_{posterior} = \sigma^2_{posterior} \times (n_{prior} \times \tanh^{-1} r_{prior}+ n_{likelihood} \times \tanh^{-1} r_{likelihood})
$$

**variance**

$$
\sigma^2_{posterior} = \frac{1}{n_{prior} + n_{Likelihood}}
$$

To simplify the integration process, we choose prior that is

$$
P(\rho) \propto (1 - \rho^2)^c
$$ where

-   $c$ is the weight the prior will have in estimation (i.e., $c = 0$ if no prior info, hence $P(\rho) \propto 1$)

Example:

Current study: $r_{xy} = 0.5, n = 200$

Previous study: $r_{xy} = 0.2765, (n=50205)$

Combining two, we have the posterior following a normal distribution with the **variance** of

$$
\sigma^2_{posterior} =  \frac{1}{n_{prior} + n_{Likelihood}} = \frac{1}{200 + 50205} = 0.0000198393
$$

**Mean**

$$
\begin{aligned}
\mu_{Posterior} &= \sigma^2_{Posterior}  \times (n_{prior} \times \tanh^{-1} r_{prior}+ n_{likelihood} \times \tanh^{-1} r_{likelihood}) \\
&= 0.0000198393 \times (50205 \times \tanh^{-1} 0.2765 + 200 \times \tanh^{-1}0.5 )\\
&= 0.2849415
\end{aligned}
$$

Hence, $Posterior \sim N(0.691, 0.0009)$, which means the correlation coefficient is $\tanh(0.691) = 0.598$ and 95% CI is

$$
\mu_{posterior} \pm 1.96 \times \sqrt{\sigma^2_{Posterior}} = 0.2849415 \pm 1.96 \times (0.0000198393)^{1/2} = (0.2762115, 0.2936714)
$$

Hence, the interval for posterior $\rho$ is (0.2693952, 0.2855105)

If future authors suspect that they have

1.  Large sampling variation
2.  Measurement error in either measures in the correlation, which attenuates the relationship between the two variables

Applying this Bayesian correction can give them a better estimate of the correlation between the two.

To implement this calculation in R, see below


```r
n_new              <- 200
r_new              <- 0.5
alpha              <- 0.05

update_correlation <- function(n_new, r_new, alpha) {
    n_meta             <- 50205
    r_meta             <- 0.2765
    
    # Variance
    var_xi         <- 1 / (n_new + n_meta)
    format(var_xi, scientific = FALSE)
    
    # mean
    mu_xi          <- var_xi * (n_meta * atanh(r_meta) + n_new * (atanh(r_new)))
    format(mu_xi, scientific  = FALSE)
    
    # confidence interval
    upper_xi       <- mu_xi + qnorm(1 - alpha / 2) * sqrt(var_xi)
    lower_xi       <- mu_xi - qnorm(1 - alpha / 2) * sqrt(var_xi)
    
    # rho
    mean_rho       <- tanh(mu_xi)
    upper_rho      <- tanh(upper_xi)
    lower_rho      <- tanh(lower_xi)
    
    # return a list
    return(
        list(
            "mu_xi" = mu_xi,
            "var_xi" = var_xi,
            "upper_xi" = upper_xi,
            "lower_xi" = lower_xi,
            "mean_rho" = mean_rho,
            "upper_rho" = upper_rho,
            "lower_rho" = lower_rho
        )
    )
}




# Old confidence interval
r_new + qnorm(1 - alpha / 2) * sqrt(1/n_new)
#> [1] 0.6385904
r_new - qnorm(1 - alpha / 2) * sqrt(1/n_new)
#> [1] 0.3614096

testing = update_correlation(n_new = n_new, r_new = r_new, alpha = alpha)

# Updated rho
testing$mean_rho
#> [1] 0.2774723

# Updated confidence interval
testing$upper_rho
#> [1] 0.2855105
testing$lower_rho
#> [1] 0.2693952
```

<br>

## Simultaneity

-   When independent variables ($X$'s) are jointly determined with the dependent variable $Y$, typically through an equilibrium mechanism, violates the second condition for causality (i.e., temporal order).

-   Examples: quantity and price by demand and supply, investment and productivity, sales and advertisement

General Simultaneous (Structural) Equations

$$
Y_i = \beta_0 + \beta_1 X_i + u_i \\
X_i = \alpha_0 + \alpha_1 Y_i + v_i
$$

Hence, the solutions are

$$
Y_i = \frac{\beta_0 + \beta_1 \alpha_0}{1 - \alpha_1 \beta_1} + \frac{\beta_1 v_i + u_i}{1 - \alpha_1 \beta_1} \\
X_i = \frac{\alpha_0 + \alpha_1 \beta_0}{1 - \alpha_1 \beta_1} + \frac{v_i + \alpha_1 u_i}{1 - \alpha_1 \beta_1}
$$

If we run only one regression, we will have biased estimators (because of **simultaneity bias**):

$$
\begin{aligned}
Cov(X_i, u_i) &= Cov(\frac{v_i + \alpha_1 u_i}{1 - \alpha_1 \beta_1}, u_i) \\
&= \frac{\alpha_1}{1- \alpha_1 \beta_1} Var(u_i)
\end{aligned}
$$

In an even more general model

$$
\begin{cases}
Y_i = \beta_0 + \beta_1 X_i + \beta_2 T_i + u_i \\
X_i = \alpha_0 + \alpha_1 Y_i + \alpha_2 Z_i + v_i
\end{cases}
$$

where

-   $X_i, Y_i$ are **endogenous** variables determined within the system

-   $T_i, Z_i$ are **exogenous** variables

Then, the reduced form of the model is

$$
\begin{cases}
\begin{aligned}
Y_i &= \frac{\beta_0 + \beta_1 \alpha_0}{1 - \alpha_1 \beta_1} + \frac{\beta_1 \alpha_2}{1 - \alpha_1 \beta_1} Z_i + \frac{\beta_2}{1 - \alpha_1 \beta_1} T_i + \tilde{u}_i \\
&= B_0 + B_1 Z_i + B_2 T_i + \tilde{u}_i
\end{aligned}
\\
\begin{aligned}
X_i &= \frac{\alpha_0 + \alpha_1 \beta_0}{1 - \alpha_1 \beta_1} + \frac{\alpha_2}{1 - \alpha_1 \beta_1} Z_i + \frac{\alpha_1\beta_2}{1 - \alpha_1 \beta_1} T_i + \tilde{v}_i \\
&= A_0 + A_1 Z_i + A_2 T_i + \tilde{v}_i
\end{aligned}
\end{cases}
$$

Then, now we can get consistent estimates of the reduced form parameters

And to get the original parameter estimates

$$
\frac{B_1}{A_1} = \beta_1 \\
B_2 (1 - \frac{B_1 A_2}{A_1B_2}) = \beta_2 \\
\frac{A_2}{B_2} = \alpha_1 \\
A_1 (1 - \frac{B_1 A_2}{A_1 B_2}) = \alpha_2
$$

Rules for Identification

**Order Condition** (necessary but not sufficient)

$$
K - k \ge m - 1
$$

where

-   $M$ = number of endogenous variables in the model

-   K = number of exogenous variables int he model

-   m = number of endogenous variables in a given

-   k = is the number of exogenous variables in a given equation

This is actually the general framework for instrumental variables

<br>

## Endogenous Treatment

Using the OLS estimates as a reference point


```r
library(AER)
library(REndo)
set.seed(421)
data("CASchools")
school <- CASchools
school$stratio <- with(CASchools, students / teachers)
m1.ols <-
  lm(read ~ stratio + english + lunch + grades + income + calworks + county,
     data = school)
summary(m1.ols)$coefficients[1:7, ]
#>                 Estimate Std. Error     t value      Pr(>|t|)
#> (Intercept) 683.45305948 9.56214469  71.4748711 3.011667e-218
#> stratio      -0.30035544 0.25797023  -1.1643027  2.450536e-01
#> english      -0.20550107 0.03765408  -5.4576041  8.871666e-08
#> lunch        -0.38684059 0.03700982 -10.4523759  1.427370e-22
#> gradesKK-08  -1.91291321 1.35865394  -1.4079474  1.599886e-01
#> income        0.71615378 0.09832843   7.2832829  1.986712e-12
#> calworks     -0.05273312 0.06154758  -0.8567863  3.921191e-01
```

### Instrumental Variable

[A3a] requires $\epsilon_i$ to be uncorrelated with $\mathbf{x}_i$

Assume [A1][A1 Linearity] , [A2][A2 Full rank], [A5][A5 Data Generation (random Sampling)]

$$
plim(\hat{\beta}_{OLS}) = \beta + [E(\mathbf{x_i'x_i})]^{-1}E(\mathbf{x_i'}\epsilon_i)
$$

[A3a] is the weakest assumption needed for OLS to be **consistent**

[A3][A3 Exogeneity of Independent Variables] fails when $x_{ik}$ is correlated with $\epsilon_i$

-   Omitted Variables Bias: $\epsilon_i$ includes any other factors that may influence the dependent variable (linearly)
-   [Simultaneity] Demand and prices are simultaneously determined.
-   [Endogenous Sample Selection] we did not have iid sample
-   [Measurement Error]

**Note**

-   Omitted Variable: an omitted variable is a variable, omitted from the model (but is in the $\epsilon_i$) and unobserved has predictive power towards the outcome.
-   Omitted Variable Bias: is the bias (and inconsistency when looking at large sample properties) of the OLS estimator when the omitted variable.
-   We cam have both positive and negative selection bias (it depends on what our story is)

The **structural equation** is used to emphasize that we are interested understanding a **causal relationship**

$$
y_{i1} = \beta_0 + \mathbf{z}_i1 \beta_1 + y_{i2}\beta_2 +  \epsilon_i
$$

where

-   $y_{it}$ is the outcome variable (inherently correlated with $\epsilon_i$)
-   $y_{i2}$ is the endogenous covariate (presumed to be correlated with $\epsilon_i$)
-   $\beta_1$ represents the causal effect of $y_{i2}$ on $y_{i1}$
-   $\mathbf{z}_{i1}$ is exogenous controls (uncorrelated with $\epsilon_i$) ($E(z_{1i}'\epsilon_i) = 0$)

OLS is an inconsistent estimator of the causal effect $\beta_2$

If there was no endogeneity

-   $E(y_{i2}'\epsilon_i) = 0$
-   the exogenous variation in $y_{i2}$ is what identifies the causal effect

If there is endogeneity

-   Any wiggle in $y_{i2}$ will shift simultaneously with $\epsilon_i$

$$
plim(\hat{\beta}_{OLS}) = \beta + [E(\mathbf{x'_ix_i})]^{-1}E(\mathbf{x'_i}\epsilon_i)
$$

where

-   $\beta$ is the causal effect
-   $[E(\mathbf{x'_ix_i})]^{-1}E(\mathbf{x'_i}\epsilon_i)$ is the endogenous effect

Hence $\hat{\beta}_{OLS}$ can be either more positive and negative than the true causal effect.

<br>

Motivation for **Two Stage Least Squares (2SLS)**

$$
y_{i1}=\beta_0 + \mathbf{z}_{i1}\beta_1 + y_{i2}\beta_2 + \epsilon_i
$$

We want to understand how movement in $y_{i2}$ effects movement in $y_{i1}$, but whenever we move $y_{i2}$, $\epsilon_i$ also moves.

**Solution**\
We need a way to move $y_{i2}$ independently of $\epsilon_i$, then we can analyze the response in $y_{i1}$ as a causal effect

-   Find an **instrumental variable(s)** $z_{i2}$

    -   Instrument **Relevance**: when\*\* $z_{i2}$ moves then $y_{i2}$ also moves
    -   Instrument **Exogeneity**: when $z_{i2}$ moves then $\epsilon_i$ does not move.

-   $z_{i2}$ is the **exogenous variation that identifies** the causal effect $\beta_2$

Finding an Instrumental variable:

-   Random Assignment: + Effect of class size on educational outcomes: instrument is initial random
-   Relation's Choice + Effect of Education on Fertility: instrument is parent's educational level
-   Eligibility + Trade-off between IRA and 401K retirement savings: instrument is 401k eligibility

**Example**

Return to College

-   education is correlated with ability - endogenous

-   **Near 4year** as an instrument

    -   Instrument Relevance: when **near** moves then education also moves
    -   Instrument Exogeneity: when **near** moves then $\epsilon_i$ does not move.

-   Other potential instruments; near a 2-year college. Parent's Education. Owning Library Card

$$
y_{i1}=\beta_0 + \mathbf{z}_{i1}\beta_1 + y_{i2}\beta_2 + \epsilon_i
$$

First Stage (Reduced Form) Equation:

$$
y_{i2} = \pi_0 + \mathbf{z_{i1}\pi_1} + \mathbf{z_{i2}\pi_2} + v_i
$$

where

-   $\pi_0 + \mathbf{z_{i1}\pi_1} + \mathbf{z_{i2}\pi_2}$ is exogenous variation $v_i$ is endogenous variation

This is called a **reduced form equation**

-   Not interested in the causal interpretation of $\pi_1$ or $\pi_2$

-   A linear projection of $z_{i1}$ and $z_{i2}$ on $y_{i2}$ (simple correlations)

-   The projections $\pi_1$ and $\pi_2$ guarantee that $E(z_{i1}'v_i)=0$ and $E(z_{i2}'v_i)=0$

Instrumental variable $z_{i2}$

-   **Instrument Relevance**: $\pi_2 \neq 0$
-   **Instrument Exogeneity**: $E(\mathbf{z_{i2}\epsilon_i})=0$

Moving only the exogenous part of $y_i2$ is moving

$$
\tilde{y}_{i2} = \pi_0 + \mathbf{z_{i1}\pi_1 + z_{i2}\pi_2}
$$

**two Stage Least Squares (2SLS)**

$$
y_{i1} = \beta_0 +\mathbf{z_{i1}\beta_1}+ y_{i2}\beta_2 + \epsilon_i
$$

$$
y_{i2} = \pi_0 + \mathbf{z_{i2}\pi_2} + \mathbf{v_i}
$$

Equivalently,

```{=tex}
\begin{equation}
\begin{split}
y_{i1} = \beta_0 + \mathbf{z_{i1}}\beta_1 + \tilde{y}_{i2}\beta_2 + u_i
\end{split}
(\#eq:2SLS)
\end{equation}
```
where

-   $\tilde{y}_{i2} =\pi_0 + \mathbf{z_{i2}\pi_2}$
-   $u_i = v_i \beta_2+ \epsilon_i$

The \@ref(eq:2SLS) holds for [A1][A1 Linearity], [A5][A5 Data Generation (random Sampling)]

-   [A2][A2 Full rank] holds if the instrument is relevant $\pi_2 \neq 0$ + $y_{i1} = \beta_0 + \mathbf{z_{i1}\beta_1 + (\pi_0 + z_{i1}\pi_1 + z_{i2}\pi_2)}\beta_2 + u_i$
-   [A3a] holds if the instrument is exogenous $E(\mathbf{z}_{i2}\epsilon_i)=0$

$$
\begin{aligned}
E(\tilde{y}_{i2}'u_i) &= E((\pi_0 + \mathbf{z_{i1}\pi_1+z_{i2}})(v_i\beta_2 + \epsilon_i)) \\
&= E((\pi_0 + \mathbf{z_{i1}\pi_1+z_{i2}})( \epsilon_i)) \\
&= E(\epsilon_i)\pi_0 + E(\epsilon_iz_{i1})\pi_1 + E(\epsilon_iz_{i2}) \\
&=0 
\end{aligned}
$$

Hence, \@ref(eq:2SLS) is consistent

The 2SLS Estimator\
1. Estimate the first stage using [OLS][Ordinary Least Squares]

$$
y_{i2} = \pi_0 + \mathbf{z_{i2}\pi_2} + \mathbf{v_i}
$$

and obtained estimated value $\hat{y}_{i2}$

2.  Estimate the altered equation using [OLS][Ordinary Least Squares]

$$
y_{i1} = \beta_0 +\mathbf{z_{i1}\beta_1}+ \hat{y}_{i2}\beta_2 + \epsilon_i \\
$$

**Properties of the 2SLS Estimator**

-   Under [A1][A1 Linearity], [A2][A2 Full rank], [A3a] (for $z_{i1}$), [A5][A5 Data Generation (random Sampling)] and if the instrument satisfies the following two conditions, + **Instrument Relevance**: $\pi_2 \neq 0$ + **Instrument Exogeneity**: $E(\mathbf{z}_{i2}'\epsilon_i) = 0$ then the 2SLS estimator is consistent
-   Can handle more than one endogenous variable and more than one instrumental variable

$$
y_{i1} = \beta_0 + z_{i1}\beta_1 + y_{i2}\beta_2 + y_{i3}\beta_3 + \epsilon_i \\
y_{i2} = \pi_0 + z_{i1}\pi_1 + z_{i2}\pi_2 + z_{i3}\pi_3 + z_{i4}\pi_4 + v_{i2} \\
y_{i3} = \gamma_0 + z_{i1}\gamma_1 + z_{i2}\gamma_2 + z_{i3}\gamma_3 + z_{i4}\gamma_4 + v_{i3}
$$

```         
    + **IV estimator**: one endogenous variable with a single instrument 
    + **2SLS estimator**: one endogenous variable with multiple instruments 
    + **GMM estimator**: multiple endogenous variables with multiple instruments
    
```

-   Standard errors produced in the second step are not correct

    -   Because we do not know $\tilde{y}$ perfectly and need to estimate it in the firs step, we are introducing additional variation
    -   We did not have this problem with [FGLS][Feasible Generalized Least Squares] because "the first stage was orthogonal to the second stage." This is generally not true for most multi-step procedure.\
    -   If [A4][A4 Homoskedasticity] does not hold, need to report robust standard errors.

-   2SLS is less efficient than OLS and will always have larger standard errors.\

    -   First, $Var(u_i) = Var(v_i\beta_2 + \epsilon_i) > Var(\epsilon_i)$\
    -   Second, $\hat{y}_{i2}$ is generally highly collinear with $\mathbf{z}_{i1}$\

-   The number of instruments need to be at least as many or more the number of endogenous variables.

**Note**

-   2SLS can be combined with [FGLS][Feasible Generalized Least Squares] to make the estimator more efficient: You have the same first-stage, and in the second-stage, instead of using OLS, you can use FLGS with the weight matrix $\hat{w}$\
-   Generalized Method of Moments can be more efficient than 2SLS.\
-   In the second-stage of 2SLS, you can also use [MLE][Maximum Likelihood], but then you are making assumption on the distribution of the outcome variable, the endogenous variable, and their relationship (joint distribution).

#### Testing Assumption

1.  [Test of Endogeneity]: Is $y_{i2}$ truly endogenous (i.e., can we just use OLS instead of 2SLS)?

2.  [Testing Instrument's assumptions](#testing-instruments-assumptions)

    -   [Exogeneity] (Cannot always test "and when you can it might not be informative")

    -   [Relevancy] (need to avoid "weak instruments")

##### Test of Endogeneity

-   2SLS is generally so inefficient that we may prefer OLS if there is not much endogeneity

-   Biased but inefficient vs efficient but biased

-   Want a sense of "how endogenous" $y_{i2}$ is

    -   if "very" endogeneous - should use 2SLS
    -   if not "very" endogenous - perhaps prefer OLS

**Invalid** Test of Endogeneity: $y_{i2}$ is endogenous if it is correlated with $\epsilon_i$,

$$
\epsilon_i = \gamma_0 + y_{i2}\gamma_1 + error_i
$$

where $\gamma_1 \neq 0$ implies that there is endogeneity

-   $\epsilon_i$ is not observed, but using the residuals

$$
e_i = \gamma_0 + y_{i2}\gamma_1 + error_i
$$

is **NOT** a valid test of endogeneity + The OLS residual, e is mechanically uncorrelated with $y_{i2}$ (by FOC for OLS) + In every situation, $\gamma_1$ will be essentially 0 and you will never be able to reject the null of no endogeneity

<br>

**Valid** test of endogeneity

-   If $y_{i2}$ is not endogenous then $\epsilon_i$ and v are uncorrelated

$$
y_{i1} = \beta_0 + \mathbf{z}_{i1}\beta_1 + y_{i2}\beta_2 + \epsilon_i \\
y_{i2} = \pi_0 + \mathbf{z}_{i1}\pi_1 + z_{i2}\pi_2 + v_i
$$

**variable Addition test**: include the first stage residuals as an additional variable,

$$
y_{i1} = \beta_0 + \mathbf{z}_{i1}\beta_1 + y_{i2}\beta_2 + \hat{v}_i \theta + error_i
$$

Then the usual t-test of significance is a valid test to evaluate the following hypothesis. **note** this test requires your instrument to be valid instrument.

$$
\begin{aligned}
H_0: \theta = 0 && \text{  (not endogenous)} \\
H_1: \theta \neq 0 && \text{  (endogenous)}
\end{aligned}
$$

##### Testing Instrument's assumptions {#testing-instruments-assumptions}

The instrumental variable must satisfy

1.  [Exogeneity] (Cannot always test "and when you can it might not be informative")
2.  [Relevancy] (need to avoid "weak instruments")

###### Exogeneity

Why exogeneity matter?

$$
E(\mathbf{z}_{i2}'\epsilon_i) = 0
$$

-   If [A3a] fails - 2SLS is also inconsistent
-   If instrument is not exogenous, then we need to find a new one.
-   Similar to [Test of Endogeneity], when there is a single instrument

$$
e_i = \gamma_0 + \mathbf{z}_{i2}\gamma_1 + error_i \\
H_0: \gamma_1 = 0
$$

is **NOT** a valid test of endogeneity

-   the OLS residual, e is mechanically uncorrelated with $z_{i2}$: $\hat{\gamma}_1$ will be essentially 0 and you will never be able to determine if the instrument is endogenous.

<br>

**Solution**

Testing Instrumental Exegeneity in an Over-identified Model

-   When there is more than one exogenous instrument (per endogenous variable), we can test for instrument exogeneity.

    -   When we have multiple instruments, the model is said to be over-identified.

    -   Could estimate the same model several ways (i.e., can identify/ estimate $\beta_1$ more than one way)

-   Idea behind the test: if the controls and instruments are truly exogenous then OLS estimation of the following regression,

$$
\epsilon_i = \gamma_0 + \mathbf{z}_{i1}\gamma_1 + \mathbf{z}_{i2}\gamma_2 + error_i
$$

should have a very low $R^2$

-   if the model is **just identified** (one instrument per endogenous variable) then the $R^2 = 0$

Steps:

(1) Estimate the structural equation by 2SLS (using all available instruments) and obtain the residuals e

(2) Regress e on all controls and instruments and obtain the $R^2$

(3) Under the null hypothesis (all IV's are uncorrelated), $nR^2 \sim \chi^2(q)$, where q is the number of instrumental variables minus the number of endogenous variables

    -   if the model is just identified (one instrument per endogenous variable) then q = 0, and the distribution under the null collapses.

low p-value means you reject the null of exogenous instruments. Hence you would like to have high p-value in this test.

<br>

**Pitfalls for the Overid test**

-   the overid test is essentially compiling the following information.\

    -   Conditional on first instrument being exogenous is the other instrument exogenous?\
    -   Conditional on the other instrument being exogenous, is the first instrument exogenous?\

-   If all instruments are endogenous than neither test will be valid

-   really only useful if one instrument is thought to be truly exogenous (randomly assigned). even f you do reject the null, the test does not tell you which instrument is exogenous and which is endogenous.

| Result          | Implication                                                                         |
|----------------------|--------------------------------------------------|
| reject the null | you can be pretty sure there is an endogenous instrument, but don't know which one. |
| fail to reject  | could be either (1) they are both exogenous, (2) they are both endogenous.          |

###### Relevancy

Why Relevance matter?

$$
\pi_2 \neq 0 
$$

\* used to show [A2][A2 Full rank] holds + If $\pi_2 = 0$ (instrument is not relevant) then [A2][A2 Full rank] fails - perfect multicollinearity\
+ If $\pi_2$ is close to 0 (**weak instrument**) then there is near perfect multicollinearity - 2SLS is highly inefficient (Large standard errors).\
\* A weak instrument will exacerbate any inconsistency due to an instrument being (even slightly) endogenous.\
+ In the simple case with no controls and a single endogenous variable and single instrumental variable,

$$
plim(\hat{\beta}_{2_{2SLS}}) = \beta_2 + \frac{E(z_{i2}\epsilon_i)}{E(z_{i2}y_{i2})}
$$

**Testing Weak Instruments**

-   can use t-test (or F-test for over-identified models) in the first stage to determine if there is a weak instrument problem.

-   [@Stock_2005]: a statistical rejection of the null hypothesis in the first stage at the 5% (or even 1%) level is not enough to insure the instrument is not weak\

    -   Rule of Thumb: need a F-stat of at least 10 (or a t-stat of at least 3.2) to reject the null hypothesis that the instrument is weak.

**Summary of the 2SLS Estimator**

$$
y_{i1}=\beta_0 + \mathbf{z}_{i1}\beta_1 + y_{i2}\beta_2 + \epsilon_i \\
y_{i2} = \pi_0 + \mathbf{z_{i1}\pi_1} + \mathbf{z_{i2}\pi_2} + v_i
$$

-   when [A3a] does not hold

$$
E(y_{i2}'\epsilon_i) \neq 0
$$

-   Then the OLS estimator is no longer unbiased or consistent.

-   If we have valid instruments $\mathbf{z}_{i2}$

-   [Relevancy] (need to avoid "weak instruments"): $\pi_2 \neq 0$ Then the 2SLS estimator is consistent under [A1][A1 Linearity], [A2][A2 Full rank], [A5a], and the above two conditions. + If [A4][A4 Homoskedasticity] also holds, then the usual standard errors are valid. + If [A4][A4 Homoskedasticity] does not hold then use the robust standard errors.

$$
y_{i1}=\beta_0 + \mathbf{z}_{i1}\beta_1 + y_{i2}\beta_2 + \epsilon_i \\
y_{i2} = \pi_0 + \mathbf{z_{i1}\pi_1} + \mathbf{z_{i2}\pi_2} + v_i
$$

-   When [A3a] does hold

$$
E(y_{i2}'\epsilon_i) = 0
$$

and we have valid instruments, then both the OLS and 2SLS estimators are consistent.

-   The OLS estimator is always more efficient
-   can use the variable addition test to determine if 2SLS is need (A3a does hold) or if OLS is valid (A3a does not hold)

Sometimes we can test the assumption for instrument to be valid:

-   [Exogeneity] : Only table when there are more instruments than endogenous variables.
-   [Relevancy] (need to avoid "weak instruments"): Always testable, need the F-stat to be greater than 10 to rule out a weak instrument

Application

Expenditure as observed instrument


```r
m2.2sls <-
        ivreg(
                read ~ stratio + english + lunch + grades + income + calworks +
                        county | expenditure + english + lunch + grades + income + calworks +
                        county ,
                data = school
        )
summary(m2.2sls)$coefficients[1:7, ]
#>                 Estimate  Std. Error     t value      Pr(>|t|)
#> (Intercept) 700.47891593 13.58064436  51.5792106 8.950497e-171
#> stratio      -1.13674002  0.53533638  -2.1234126  3.438427e-02
#> english      -0.21396934  0.03847833  -5.5607753  5.162571e-08
#> lunch        -0.39384225  0.03773637 -10.4366757  1.621794e-22
#> gradesKK-08  -1.89227865  1.37791820  -1.3732881  1.704966e-01
#> income        0.62487986  0.11199008   5.5797785  4.668490e-08
#> calworks     -0.04950501  0.06244410  -0.7927892  4.284101e-01
```

<br>

#### Checklist

1.  Regress the dependent variable on the instrument (reduced form). Since under OLS, we have unbiased estimate, the coefficient estimate should be significant (make sure the sign makes sense)
2.  Report F-stat on the excluded instruments. F-stat \< 10 means you have a weak instrument [@stock2002survey].
3.  Present $R^2$ before and after including the instrument [@rossi2014even]
4.  For models with multiple instrument, present firs-t and second-stage result for each instrument separately. Overid test should be conducted (e.g., Sargan-Hansen J)
5.  Hausman test between OLS and 2SLS (don't confuse this test for evidence that endogeneity is irrelevant - under invalid IV, the test is useless)
6.  Compare the 2SLS with the limited information ML. If they are different, you have evidence for weak instruments.

#### Good Instruments

[Exogeneity] and [Relevancy] are necessary but not sufficient for IV to produce consistent estimates.

Without theory or possible explanation, you can always create a new variable that is correlated with $X$ and uncorrelated with $\epsilon$

For example, we want to estimate the effect of price on quantity [@reiss2011, p. 960]

$$
Q = \beta_1 P + \beta_2 X + \epsilon \\
P = \pi_1 X + \eta
$$

where $\epsilon$ and $\eta$ are jointly determined, $X \perp \epsilon, \eta$

Without theory, we can just create a new variable $Z = X + u$ where $E(u) = 0; u \perp X, \epsilon, \eta$

Then, $Z$ satisfied both conditions:

-   Relevancy: $X$ correlates $P$ $\rightarrow$ $Z$ correlates $P$

-   Exogeneity: $u \perp \epsilon$ (random noise)

But obviously, it's not a valid instrument (intuitively). But theoretically, relevance and exogeneity are not sufficient to identify $\beta$ because of unsatisfied rank condition for identification.

Moreover, the functional form of the instrument also plays a role when choosing a good instrument. Hence, we always need to check for the robustness of our instrument.

IV methods even with valid instruments can still have poor sampling properties (finite sample bias, large sampling errors) [@rossi2014even]

When you have a weak instrument, it's important to report it appropriately [@lee2021valid]. This problem will be exacerbated if you have multiple instruments.

##### Lagged dependent variable

In time series data sets, we can use lagged dependent variable as an instrument because it is not influenced by current shocks.

Citations for lagged dependent variable in econ [@chetty2013],

<br>

### Internal instrumental variable

-   (also known as **instrument free methods**). This section is based on Raluca Gui's [guide](https://cran.r-project.org/web/packages/REndo/vignettes/REndo-introduction.pdf)

-   alternative to external instrumental variable approaches

-   All approaches here assume a **continuous dependent variable**

#### Non-hierarchical Data (Cross-classified)

$$
Y_t = \beta_0 + \beta_1 P_t + \beta_2 X_t + \epsilon_t
$$

where

-   $t = 1, .., T$ (indexes either time or cross-sectional units)
-   $Y_t$ is a k x 1 response variable
-   $X_t$ is a k x n exogenous regressor
-   $P_t$ is a k x 1 continuous endogenous regressor
-   $\epsilon_t$ is a structural error term with $\mu_\epsilon =0$ and $E(\epsilon^2) = \sigma^2$
-   $\beta$ are model parameters

The endogeneity problem arises from the correlation of $P_t$ and $\epsilon_t$:

$$
P_t = \gamma Z_t + v_t
$$

where

-   $Z_t$ is a l x 1 vector of internal instrumental variables\
-   $ν_t$ is a random error with $\mu_{v_t}, E(v^2) = \sigma^2_v, E(\epsilon v) = \sigma_{\epsilon v}$\
-   $Z_t$ is assumed to be stochastic with distribution G\
-   $ν_t$ is assumed to have density h(·)

##### Latent Instrumental Variable {#latent-instrumental-variable}

[@Ebbes_2005]

assume $Z_t$ (unobserved) to be uncorrelated with $\epsilon_t$, which is similar to [Instrumental Variable]. Hence, $Z_t$ and $ν_t$ can't be identified without distributional assumptions

The distributions of $Z_t$ and $ν_t$ need to be specified such that:

(1) endogeneity of $P_t$ is corrected\
(2) the distribution of $P_t$ is empirically close to the integral that expresses the amount of overlap of Z as it is shifted over ν (= the convolution between $Z_t$ and $ν_t$).

When the density h(·) = Normal, then G cannot be normal because the parameters would not be identified [@Ebbes_2005] .

Hence,

-   in the [LIV](#latent-instrumental-variable) model the distribution of $Z_t$ is discrete
-   in the [Higher Moments Method] and [Joint Estimation Using Copula] methods, the distribution of $Z_t$ is taken to be skewed.

$Z_t$ are assumed **unobserved, discrete and exogenous**, with

-   an unknown number of groups m
-   $\gamma$ is a vector of group means.

Identification of the parameters relies on the distributional assumptions of

-   $P_t$: a non-Gaussian distribution
-   $Z_t$ discrete with $m \ge 2$

Note:

-   If $Z_t$ is continuous, the model is unidentified
-   If $P_t \sim N$, you have inefficient estimates.


```r
m3.liv <- latentIV(read ~ stratio, data=school)
summary(m3.liv)$coefficients[1:7,]
#>                   Estimate    Std. Error       z-score     Pr(>|z|)
#> (Intercept)   6.996014e+02  2.686186e+02  2.604441e+00 9.529597e-03
#> stratio      -2.272673e+00  1.367757e+01 -1.661605e-01 8.681108e-01
#> pi1          -4.896363e+01  5.526907e-08 -8.859139e+08 0.000000e+00
#> pi2           1.963920e+01  9.225351e-02  2.128830e+02 0.000000e+00
#> theta5       6.939432e-152 3.354672e-160  2.068587e+08 0.000000e+00
#> theta6        3.787512e+02  4.249457e+01  8.912932e+00 1.541524e-17
#> theta7       -1.227543e+00  4.885276e+01 -2.512741e-02 9.799653e-01
```

it will return a coefficient very different from the other methods since there is only one endogenous variable.

##### Joint Estimation Using Copula

assume $Z_t$ (unobserved) to be uncorrelated with $\epsilon_t$, which is similar to [Instrumental Variable]. Hence, $Z_t$ and $ν_t$ can't be identified without distributional assumptions

[@Park_2012] allows joint estimation of the continuous $P_t$ and $\epsilon_t$ using Gaussian copulas, where a copula is a function that maps several conditional distribution functions (CDF) into their joint CDF).

The underlying idea is that using information contained in the observed data, one selects marginal distributions for $P_t$ and $\epsilon_t$. Then, the copula model constructs a flexible multivariate joint distribution that allows a wide range of correlations between the two marginals.

The method allows both continuous and discrete $P_t$.

In the special case of **one continuous** $P_t$, estimation is based on MLE\
Otherwise, based on Gaussian copulas, augmented OLS estimation is used.

**Assumptions**:

-   skewed $P_t$

-   the recovery of the correct parameter estimates

-   $\epsilon_t \sim$ normal marginal distribution. The marginal distribution of $P_t$ is obtained using the **Epanechnikov kernel density estimator**\
    $$
    \hat{h}_p = \frac{1}{T . b} \sum_{t=1}^TK(\frac{p - P_t}{b})
    $$ where

-   $P_t$ = endogenous variables

-   $K(x) = 0.75(1-x^2)I(||x||\le 1)$

-   $b=0.9T^{-1/5}\times min(s, IQR/1.34)$ suggested by [@Silverman_1969]

    -   IQR = interquartile range
    -   s = sample standard deviation
    -   T = n of time periods observed in the data

In augmented OLS and MLE, the inference procedure occurs in two stages:

(1): the empirical distribution of $P_t$ is computed\
(2) used in it constructing the likelihood function)\
Hence, the standard errors would not be correct.

So we use the sampling distributions (from bootstrapping) to get standard errors and the variance-covariance matrix. Since the distribution of the bootstraped parameters is highly skewed, we report the percentile confidence intervals is preferable.


```r
set.seed(110)
m4.cc <-
        copulaCorrection(
                read ~ stratio + english + lunch + calworks +
                        grades + income + county | continuous(stratio),
                data = school,
                optimx.args = list(method = c("Nelder-Mead"), itnmax = 60000),
                num.boots = 2,
                verbose = FALSE
        )
summary(m4.cc)$coefficients[1:7, ]
#>             Point Estimate   Boots SE Lower Boots CI (95%) Upper Boots CI (95%)
#> (Intercept)   683.06900891 2.80554212                   NA                   NA
#> stratio        -0.32434608 0.02075999                   NA                   NA
#> english        -0.21576110 0.01450666                   NA                   NA
#> lunch          -0.37087664 0.01902052                   NA                   NA
#> calworks       -0.05569058 0.02076781                   NA                   NA
#> gradesKK-08    -1.92286128 0.25684614                   NA                   NA
#> income          0.73595353 0.04725700                   NA                   NA
```

we run this model with only one endogenous continuous regressor (`stratio`). Sometimes, the code will not converge, in which case you can use different

-   optimization algorithm
-   starting values
-   maximum number of iterations

##### Higher Moments Method

suggested by [@Lewbel_1997] to identify $\epsilon_t$ caused by **measurement error**.

Identification is achieved by using third moments of the data, with no restrictions on the distribution of $\epsilon_t$\
The following instruments can be used with 2SLS estimation to obtain consistent estimates:

$$
\begin{aligned}
q_{1t} &=  (G_t - \bar{G}) \\
q_{2t} &=  (G_t - \bar{G})(P_t - \bar{P}) \\
q_{3t} &=   (G_t - \bar{G})(Y_t - \bar{Y})\\
q_{4t} &=  (Y_t - \bar{Y})(P_t - \bar{P}) \\
q_{5t} &=  (P_t - \bar{P})^2 \\
q_{6t} &=  (Y_t - \bar{Y})^2 \\
\end{aligned}
$$

where

-   $G_t = G(X_t)$ for any given function G that has finite third own and cross moments
-   X = exogenous variable

$q_{5t}, q_{6t}$ can be used only when the measurement and $\epsilon_t$ are symmetrically distributed. The rest of the instruments does not require any distributional assumptions for $\epsilon_t$.

Since the regressors $G(X) = X$ are included as instruments, $G(X)$ can't be a linear function of X in $q_{1t}$

Since this method has very strong assumptions, [Higher Moments Method] should only be used in case of overidentification


```r
set.seed(111)
m5.hetEr <-
        hetErrorsIV(
                read ~ stratio + english + lunch + calworks + income +
                        grades + county | stratio | IIV(income, english),
                data = school
        )
summary(m5.hetEr)$coefficients[1:7, ]
#>                 Estimate  Std. Error    t value     Pr(>|t|)
#> (Intercept) 662.78791557 27.90173069 23.7543657 2.380436e-76
#> stratio       0.71480686  1.31077325  0.5453322 5.858545e-01
#> english      -0.19522271  0.04057527 -4.8113717 2.188618e-06
#> lunch        -0.37834232  0.03927793 -9.6324402 9.760809e-20
#> calworks     -0.05665126  0.06302095 -0.8989273 3.692776e-01
#> income        0.82693755  0.17236557  4.7975797 2.335271e-06
#> gradesKK-08  -1.93795843  1.38723186 -1.3969968 1.632541e-01
```

recommend using this approach to create additional instruments to use with external ones for better efficiency.

##### Heteroskedastic Error Approach

-   using means of variables that are uncorrelated with the product of heteroskedastic errors to identify structural parameters.
-   This method can be use either when you don't have external instruments or you want to use additional instruments to improve the efficiency of the IV estimator [@lewbel2012using]
-   The instruments are constructed as simple functions of data
-   Model's assumptions:

$$
E(X \epsilon) = 0 \\
E(X v ) = 0 \\
cov(Z, \epsilon v) = 0  \\
cov(Z, v^2) \neq 0 \text{  (for identification)}
$$

Structural parameters are identified by 2SLS regression of Y on X and P, using X and [Z − E(Z)]ν as instruments.

$$
\text{instrument's strength} \propto cov((Z-\bar{Z})v,v)
$$

where $cov((Z-\bar{Z})v,v)$ is the degree of heteroskedasticity of ν with respect to Z [@lewbel2012using], which can be empirically tested.

If it is zero or close to zero (i.e.,the instrument is weak), you might have imprecise estimates, with large standard errors.

-   Under homoskedasticity, the parameters of the model are unidentified.
-   Under heteroskedasticity related to at least some elements of X, the parameters of the model are identified.

#### Hierarchical Data

Multiple independent assumptions involving various random components at different levels mean that any moderate correlation between some predictors and a random component or error term can result in a significant bias of the coefficients and of the variance components. [@kim2007multilevel] proposed a generalized method of moments which uses both, the between and within variations of the exogenous variables, but only assumes the within variation of the variables to be endogenous.

**Assumptions**

-   the errors at each level $\sim iid N$
-   the slope variables are exogenous
-   the level-1 $\epsilon \perp X, P$. If this is not the case, additional, external instruments are necessary

**Hierarchical Model**

$$
\begin{aligned}
Y_{cst} &= Z_{cst}^1 \beta_{cs}^1 + X_{cst}^1 \beta_1 + \epsilon_{cst}^1 \\
\beta^1_{cs} &= Z_{cs}^2 \beta_{c}^2 + X_{cst}^2 \beta_2 + \epsilon_{cst}^2 \\
\beta^2_{c} &= X^3_c \beta_3 + \epsilon_c^3
\end{aligned}
$$

Bias could stem from:

-   errors at the higher two levels ($\epsilon_c^3,\epsilon_{cst}^2$) are correlated with some of the regressors
-   only third level errors ($\epsilon_c^3$) are correlated with some of the regressors

[@kim2007multilevel] proposed

-   When all variables are assumed exogenous, the proposed estimator equals the random effects estimator
-   When all variables are assumed endogenous, it equals the fixed effects estimator
-   also use omitted variable test (based on the Hausman-test [@hausman1978specification] for panel data), which allows the comparison of a robust estimator and an estimator that is efficient under the null hypothesis of no omitted variables or the comparison of two robust estimators at different levels.


```r
set.seed(113)
school$gr08 <- school$grades == "KK-06"
m7.multilevel <-
        multilevelIV(read ~ stratio + english + lunch + income + gr08 +
                             calworks + (1 | county) | endo(stratio),
                     data = school)
summary(m7.multilevel)$coefficients[1:7, ]
#>                Estimate Std. Error     z-score     Pr(>|z|)
#> (Intercept) 675.8228656 5.58008680 121.1133248 0.000000e+00
#> stratio      -0.4956054 0.23922638  -2.0717005 3.829339e-02
#> english      -0.2599777 0.03413530  -7.6160948 2.614656e-14
#> lunch        -0.3692954 0.03560210 -10.3728537 3.295342e-25
#> income        0.6723141 0.08862012   7.5864728 3.287314e-14
#> gr08TRUE      2.1590333 1.28167222   1.6845440 9.207658e-02
#> calworks     -0.0570633 0.05711701  -0.9990596 3.177658e-01
```

Another example using simulated data

-   level-1 regressors: $X_{11}, X_{12}, X_{13}, X_{14}, X_{15}$, where $X_{15}$ is correlated with the level-2 error (i.e., endogenous).\
-   level-2 regressors: $X_{21}, X_{22}, X_{23}, X_{24}$\
-   level-3 regressors: $X_{31}, X_{32}, X_{33}$

We estimate a three-level model with X15 assumed endogenous. Having a three-level hierarchy, `multilevelIV()` returns five estimators, from the most robust to omitted variables (FE_L2), to the most efficient (REF) (i.e. lowest mean squared error).

-   The random effects estimator (REF) is efficient assuming no omitted variables
-   The fixed effects estimator (FE) is unbiased and asymptotically normal even in the presence of omitted variables.
-   Because of the efficiency, the random effects estimator is preferable if you think there is no omitted. variables
-   The robust estimator would be preferable if you think there is omitted variables.


```r
data(dataMultilevelIV)
set.seed(114)
formula1 <-
        y ~ X11 + X12 + X13 + X14 + X15 + X21 + X22 + X23 + X24 +
        X31 + X32 + X33 + (1 | CID) + (1 | SID) | endo(X15)
m8.multilevel <-
        multilevelIV(formula = formula1, data = dataMultilevelIV)
coef(m8.multilevel)
#>                    REF     FE_L2      FE_L3     GMM_L2     GMM_L3
#> (Intercept) 64.3640774  0.000000  0.0000000 64.6642061 64.3644220
#> X11          3.0356390  3.047931  3.0353448  3.0356094  3.0356389
#> X12          9.0005462  8.996679  8.9999438  8.9966073  9.0005417
#> X13         -2.0082559 -2.000106 -2.0090020 -2.0215816 -2.0082712
#> X14          1.9809907  2.001761  1.9803275  1.9849995  1.9809953
#> X15         -0.5739658 -1.036909 -0.5745241 -1.0344864 -0.5744947
#> X21         -2.2423675  0.000000 -2.2319682 -2.2172859 -2.2423387
#> X22         -3.2658889  0.000000 -2.9345899 -3.3146849 -3.2659449
#> X23         -2.8332479  0.000000 -2.8060569 -2.8581647 -2.8332765
#> X24          5.0696401  0.000000  5.0895430  5.0183704  5.0695812
#> X31          2.0770536  0.000000  0.0000000  2.0710383  2.0770467
#> X32          0.4540926  0.000000  0.0000000  0.4571712  0.4540962
#> X33          0.0991915  0.000000  0.0000000  0.0980949  0.0991902
```


```r
summary(m8.multilevel, "REF")
#> 
#> Call:
#> multilevelIV(formula = formula1, data = dataMultilevelIV)
#> 
#> Number of levels: 3
#> Number of observations: 2767
#> Number of groups: L2(CID): 1347  L3(SID): 40
#> 
#> Coefficients for model REF:
#>             Estimate Std. Error z-score Pr(>|z|)    
#> (Intercept) 64.36408    6.45959   9.964   <2e-16 ***
#> X11          3.03564    0.02763 109.863   <2e-16 ***
#> X12          9.00055    0.02608 345.152   <2e-16 ***
#> X13         -2.00826    0.02521 -79.668   <2e-16 ***
#> X14          1.98099    0.02639  75.079   <2e-16 ***
#> X15         -0.57397    0.01980 -28.987   <2e-16 ***
#> X21         -2.24237    0.18661 -12.016   <2e-16 ***
#> X22         -3.26589    0.38703  -8.438   <2e-16 ***
#> X23         -2.83325    0.10330 -27.427   <2e-16 ***
#> X24          5.06964    0.07322  69.240   <2e-16 ***
#> X31          2.07705    0.08935  23.246   <2e-16 ***
#> X32          0.45409    0.19116   2.375   0.0175 *  
#> X33          0.09919    0.04153   2.388   0.0169 *  
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Omitted variable tests for model REF:
#>               df   Chisq  p-value    
#> GMM_L2_vs_REF  7  161.14  < 2e-16 ***
#> GMM_L3_vs_REF 13 -829.75 1.000000    
#> FE_L2_vs_REF  13   40.00 0.000138 ***
#> FE_L3_vs_REF  13   39.99 0.000139 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

True $\beta_{X_{15}} =-1$. We can see that some estimators are bias because $X_{15}$ is correlated with the level-two error, to which only FE_L2 and GMM_L2 are robust

To select the appropriate estimator, we use the omitted variable test.

In a three-level setting, we can have different estimator comparisons:

-   Fixed effects vs. random effects estimators: Test for omitted level-two and level-three omitted effects, simultaneously, one compares FE_L2 to REF. But we will not know at which omitted variables exist.\
-   Fixed effects vs. GMM estimators: Once the existence of omitted effects is established but not sure at which level, we test for level-2 omitted effects by comparing FE_L2 vs GMM_L3. If you reject the null, the omitted variables are at level-2 The same is accomplished by testing FE_L2 vs. GMM_L2, since the latter is consistent only if there are no omitted effects at level-2.\
-   Fixed effects vs. fixed effects estimators: We can test for omitted level-2 effects, while allowing for omitted level-3 effects by comparing FE_L2 vs. FE_L3 since FE_L2 is robust against both level-2 and level-3 omitted effects while FE_L3 is only robust to level-3 omitted variables.

Summary, use the omitted variable test comparing `REF vs. FE_L2` first.

-   If the null hypothesis is rejected, then there are omitted variables either at level-2 or level-3

-   Next, test whether there are level-2 omitted effects, since testing for omitted level three effects relies on the assumption there are no level-two omitted effects. You can use any of these pair of comparisons:

    -   `FE_L2 vs. FE_L3`
    -   `FE_L2 vs. GMM_L2`

-   If no omitted variables at level-2 are found, test for omitted level-3 effects by comparing either

    -   `FE_L3` vs. `GMM_L3`
    -   `GMM_L2` vs. `GMM_L3`


```r
summary(m8.multilevel, "REF")
#> 
#> Call:
#> multilevelIV(formula = formula1, data = dataMultilevelIV)
#> 
#> Number of levels: 3
#> Number of observations: 2767
#> Number of groups: L2(CID): 1347  L3(SID): 40
#> 
#> Coefficients for model REF:
#>             Estimate Std. Error z-score Pr(>|z|)    
#> (Intercept) 64.36408    6.45959   9.964   <2e-16 ***
#> X11          3.03564    0.02763 109.863   <2e-16 ***
#> X12          9.00055    0.02608 345.152   <2e-16 ***
#> X13         -2.00826    0.02521 -79.668   <2e-16 ***
#> X14          1.98099    0.02639  75.079   <2e-16 ***
#> X15         -0.57397    0.01980 -28.987   <2e-16 ***
#> X21         -2.24237    0.18661 -12.016   <2e-16 ***
#> X22         -3.26589    0.38703  -8.438   <2e-16 ***
#> X23         -2.83325    0.10330 -27.427   <2e-16 ***
#> X24          5.06964    0.07322  69.240   <2e-16 ***
#> X31          2.07705    0.08935  23.246   <2e-16 ***
#> X32          0.45409    0.19116   2.375   0.0175 *  
#> X33          0.09919    0.04153   2.388   0.0169 *  
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Omitted variable tests for model REF:
#>               df   Chisq  p-value    
#> GMM_L2_vs_REF  7  161.14  < 2e-16 ***
#> GMM_L3_vs_REF 13 -829.75 1.000000    
#> FE_L2_vs_REF  13   40.00 0.000138 ***
#> FE_L3_vs_REF  13   39.99 0.000139 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# compare REF with all the other estimators. Testing REF (the most efficient estimator) against FE_L2 (the most robust estimator), equivalently we are testing simultaneously for level-2 and level-3 omitted effects. 
```

Since the null hypothesis is rejected (p = 0.000139), there is bias in the random effects estimator.

To test for level-2 omitted effects (regardless of level-3 omitted effects), we compare FE_L2 versus FE_L3


```r
summary(m8.multilevel,"FE_L2")
#> 
#> Call:
#> multilevelIV(formula = formula1, data = dataMultilevelIV)
#> 
#> Number of levels: 3
#> Number of observations: 2767
#> Number of groups: L2(CID): 1347  L3(SID): 40
#> 
#> Coefficients for model FE_L2:
#>               Estimate Std. Error z-score Pr(>|z|)    
#> (Intercept)  0.000e+00  1.373e-18    0.00        1    
#> X11          3.048e+00  3.193e-02   95.47   <2e-16 ***
#> X12          8.997e+00  3.377e-02  266.43   <2e-16 ***
#> X13         -2.000e+00  3.211e-02  -62.29   <2e-16 ***
#> X14          2.002e+00  3.437e-02   58.24   <2e-16 ***
#> X15         -1.037e+00  3.301e-02  -31.41   <2e-16 ***
#> X21          0.000e+00  1.881e-18    0.00        1    
#> X22          0.000e+00  1.060e-18    0.00        1    
#> X23          0.000e+00  1.782e-18    0.00        1    
#> X24          0.000e+00  7.536e-18    0.00        1    
#> X31          0.000e+00  2.747e-17    0.00        1    
#> X32          0.000e+00  2.679e-17    0.00        1    
#> X33          0.000e+00  1.210e-16    0.00        1    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Omitted variable tests for model FE_L2:
#>                 df Chisq  p-value    
#> FE_L2_vs_REF    13 40.00 0.000138 ***
#> FE_L2_vs_FE_L3   9 38.95 1.18e-05 ***
#> FE_L2_vs_GMM_L2 12 40.00 7.20e-05 ***
#> FE_L2_vs_GMM_L3 13 40.00 0.000138 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

The null hypothesis of no omitted level-2 effects is rejected ($p = 3.92e − 05$). Hence, there are omitted effects at level-two. We should use FE_L2 which is consistent with the underlying data that we generated (level-2 error correlated with $X_15$, which leads to biased FE_L3 coefficients.

The omitted variable test between FE_L2 and GMM_L2 should reject the null hypothesis of no omitted level-2 effects (p-value is 0).

If we assume an endogenous variable as exogenous, the RE and GMM estimators will be biased because of the wrong set of internal instrumental variables. To increase our confidence, we should compare the omitted variable tests when the variable is considered endogenous vs. exogenous to get a sense whether the variable is truly endogenous.

### Proxy Variables

-   Can be in place of the omitted variable

-   will not be able to estimate the effect of the omitted variable

-   will be able to reduce some endogeneity caused bye the omitted variable

-   but it can have [Measurement Error]. Hence, you have to be extremely careful when using proxies.

Criteria for a proxy variable:

1.  The proxy is correlated with the omitted variable.
2.  Having the omitted variable in the regression will solve the problem of endogeneity
3.  The variation of the omitted variable unexplained by the proxy is uncorrelated with all independent variables, including the proxy.

IQ test can be a proxy for ability in the regression between wage explained education.

For the third requirement

$$
ability = \gamma_0 + \gamma_1 IQ + \epsilon
$$

where $\epsilon$ is uncorrelated with education and IQ test.

## Endogenous Sample Selection

-   Also known as sample selection or self-selection problem

-   The omitted variable is how people were selected into the sample

Some disciplines consider nonresponse bias and selection bias as sample selection.

-   When unobservable factors that affect who is in the sample are independent of unobservable factors that affect the outcome, the sample selection is not endogenous. Hence, the sample selection is ignorable and estimator that ignores sample selection is still consistent.
-   when the unobservable factors that affect who is included in the sample are correlated with the unobservable factors that affect the outcome, the sample selection is endogenous and not ignorable, because estimators that ignore endogenous sample selection are not consistent (we don't know which part of the observable outcome is related to the causal relationship and which part is due to different people were selected for the treatment and control groups).

Assumptions: - The unobservables that affect the treatment selection and the outcome are jointly distributed as bivariate normal.

Notes:

-   If you don't have strong exclusion restriction, identification is driven by the assumed non linearity in the functional form (through inverse Mills ratio). E.g., the estimate depend on the bivariate normal distribution of the error structure:

    -   With strong exclusion restriction for the covariate in the correction equation, the variation in this variable can help identify the control for selection
    -   With weak exclusion restriction, and the coavriate exists in both steps, it's the assumed error structure that identifies the control for selection.

To combat [Sample Selection], we can

-   Randomization: participants are randomly selected into treatment and control.
-   Instruments that determine the treatment status (i.e., treatment vs. control) but not the outcome ($Y$)
-   Functional form of the selection and outcome processes: originated from [@Heckman_1976], later on generalize by [@amemiya1984tobit]

We have our main model

$$
\mathbf{y^* = xb + \epsilon}
$$

However, the pattern of missingness (i.e., censored) is related to the unobserved (latent) process:

$$
\mathbf{z^* = w \gamma + u}
$$

and

$$
z_i = 
\begin{cases}
1& \text{if } z_i^*>0 \\
0&\text{if } z_i^*\le0\\
\end{cases}
$$

Equivalently, $z_i = 1$ ($y_i$ is observed) when

$$
u_i \ge -w_i \gamma
$$

Hence, the probability of observed $y_i$ is

$$
\begin{aligned}
P(u_i \ge -w_i \gamma) &= 1 - \Phi(-w_i \gamma) \\
&= \Phi(w_i \gamma) & \text{symmetry of the standard normal distribution}
\end{aligned}
$$

We will **assume**

-   the error term of the selection $\mathbf{u \sim N(0,I)}$
-   $Var(u_i) = 1$ for identification purposes

Visually, $P(u_i \ge -w_i \gamma)$ is the shaded area.


```r
x = seq(-3, 3, length = 200)
y = dnorm(x, mean = 0, sd = 1)
plot(x,
     y,
     type = "l",
     main = bquote("Probabibility distribution of" ~ u[i]))
x = seq(0.3, 3, length = 100)
y = dnorm(x, mean = 0, sd = 1)
polygon(c(0.3, x, 3), c(0, y, 0), col = "gray")
text(1, 0.1, bquote(1 - Phi ~ (-w[i] ~ gamma)))
arrows(-0.5, 0.1, 0.3, 0, length = .15)
text(-0.5, 0.12, bquote(-w[i] ~ gamma))
legend(
  "topright",
  "Gray = Prob of Observed",
  pch = 1,
  title = "legend",
  inset = .02
)
```

<img src="30-endogeneity_files/figure-html/unnamed-chunk-12-1.png" width="90%" style="display: block; margin: auto;" />

Hence in our observed model, we see

```{=tex}
\begin{equation}
y_i = x_i\beta + \epsilon_i \text{when $z_i=1$}
\end{equation}
```
and the joint distribution of the selection model ($u_i$), and the observed equation ($\epsilon_i$) as

$$
\left[
\begin{array}
{c}
u \\
\epsilon \\
\end{array}
\right]
\sim^{iid}N
\left(
\left[
\begin{array}
{c}
0 \\
0 \\
\end{array}
\right],
\left[
\begin{array}
{cc}
1 & \rho \\
\rho & \sigma^2_{\epsilon} \\
\end{array}
\right]
\right)
$$

The relation between the observed and selection models:

$$
\begin{aligned}
E(y_i | y_i \text{ observed}) &= E(y_i| z^*>0) \\
&= E(y_i| -w_i \gamma) \\
&= \mathbf{x}_i \beta + E(\epsilon_i | u_i > -w_i \gamma) \\
&= \mathbf{x}_i \beta + \rho \sigma_\epsilon \frac{\phi(w_i \gamma)}{\Phi(w_i \gamma)}
\end{aligned}
$$

where $\frac{\phi(w_i \gamma)}{\Phi(w_i \gamma)}$ is the Inverse Mills Ratio. and $\rho \sigma_\epsilon \frac{\phi(w_i \gamma)}{\Phi(w_i \gamma)} \ge 0$

A property of IMR: Its derivative is: $IMR'(x) = -x IMR(x) - IMR(x)^2$

Great visualization of special cases of correlation patterns amongst data and errors by professor [Rob Hick](https://rlhick.people.wm.edu/stories/econ_407_notes_heckman.html)

Note:

[@Bareinboim_2014] is an excellent summary of cases that we can still do causal inference in case of selection bias. I'll try to summarize their idea here:

Let X be an action, Y be an outcome, and S be a binary indicator of entry into the data pool where (S = 1 = in the sample, S = 0 =out of sample) and Q be the conditional distribution $Q = P(y|x)$.

Usually we want to understand , but because of S, we only have $P(y, x|S = 1)$. Hence, we'd like to recover $P(y|x)$ from $P(y, x|S = 1)$

-   If both X and Y affect S, we can't unbiasedly estimate $P(y|x)$

In the case of Omitted variable bias (U) and sample selection bias (S), you have unblocked extraneous "flow" of information between X and Y, which causes spurious correlation for X and Y. Traditionally, we would recover Q by parametric assumption of

(1) the data generating process (e.g., Heckman 2-step)
(2) type of data-generating model (e..g, treatment-dependent or outcome-dependent)
(3) selection's probability $P(S = 1|P a_s)$ with non-parametrically based causal graphical models, the authors proposed more robust way to model misspecification regardless of the type of data-generating model, and do not require selection's probability. Hence, you can recover Q
    -   without external data
    -   with external data
    -   causal effects with the Selection-backdoor criterion

### Tobit-2

also known as Heckman's standard sample selection model\
Assumption: joint normality of the errors

Data here is taken from [@Mroz_1987]'s paper.

We want to estimate the log(wage) for married women, with education, experience, experience squared, and a dummy variable for living in a big city. But we can only observe the wage for women who are working, which means a lot of married women in 1975 who were out of the labor force are unaccounted for. Hence, an OLS estimate of the wage equation would be bias due to sample selection. Since we have data on non-participants (i.e., those who are not working for pay), we can correct for the selection process.

The Tobit-2 estimates are consistent

#### Example 1


```r
library(sampleSelection)
library(dplyr)
data("Mroz87") #1975 data on married women’s pay and labor-force participation from the Panel Study of Income Dynamics (PSID)
head(Mroz87)
#>   lfp hours kids5 kids618 age educ   wage repwage hushrs husage huseduc huswage
#> 1   1  1610     1       0  32   12 3.3540    2.65   2708     34      12  4.0288
#> 2   1  1656     0       2  30   12 1.3889    2.65   2310     30       9  8.4416
#> 3   1  1980     1       3  35   12 4.5455    4.04   3072     40      12  3.5807
#> 4   1   456     0       3  34   12 1.0965    3.25   1920     53      10  3.5417
#> 5   1  1568     1       2  31   14 4.5918    3.60   2000     32      12 10.0000
#> 6   1  2032     0       0  54   12 4.7421    4.70   1040     57      11  6.7106
#>   faminc    mtr motheduc fatheduc unem city exper  nwifeinc wifecoll huscoll
#> 1  16310 0.7215       12        7  5.0    0    14 10.910060    FALSE   FALSE
#> 2  21800 0.6615        7        7 11.0    1     5 19.499981    FALSE   FALSE
#> 3  21040 0.6915       12        7  5.0    0    15 12.039910    FALSE   FALSE
#> 4   7300 0.7815        7        7  5.0    0     6  6.799996    FALSE   FALSE
#> 5  27300 0.6215       12       14  9.5    1     7 20.100058     TRUE   FALSE
#> 6  19495 0.6915       14        7  7.5    1    33  9.859054    FALSE   FALSE
Mroz87 = Mroz87 %>%
        mutate(kids = kids5+kids618)

library(nnet)
library(ggplot2)
library(reshape2)
```

2-stage Heckman's model:

(1) probit equation estimates the selection process (who is in the labor force?)
(2) the results from 1st stage are used to construct a variable that captures the selection effect in the wage equation. This correction variable is called the **inverse Mills ratio**.


```r
# OLS: log wage regression on LF participants only
ols1 = lm(log(wage) ~ educ + exper + I( exper^2 ) + city, data=subset(Mroz87, lfp==1))
# Heckman's Two-step estimation with LFP selection equation
heck1 = heckit(selection = lfp ~ age + I( age^2 ) + kids + huswage + educ, # the selection process, lfp = 1 if the woman is participating in the labor force 
                 outcome = log(wage) ~ educ + exper + I( exper^2 ) + city, data=Mroz87 )

summary(heck1$probit)
#> --------------------------------------------
#> Probit binary choice model/Maximum Likelihood estimation
#> Newton-Raphson maximisation, 4 iterations
#> Return code 1: gradient close to zero (gradtol)
#> Log-Likelihood: -482.8212 
#> Model: Y == '1' in contrary to '0'
#> 753 observations (325 'negative' and 428 'positive') and 6 free parameters (df = 747)
#> Estimates:
#>                  Estimate  Std. error t value   Pr(> t)    
#> XS(Intercept) -4.18146681  1.40241567 -2.9816  0.002867 ** 
#> XSage          0.18608901  0.06517476  2.8552  0.004301 ** 
#> XSI(age^2)    -0.00241491  0.00075857 -3.1835  0.001455 ** 
#> XSkids        -0.14955977  0.03825079 -3.9100 9.230e-05 ***
#> XShuswage     -0.04303635  0.01220791 -3.5253  0.000423 ***
#> XSeduc         0.12502818  0.02277645  5.4894 4.034e-08 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> Significance test:
#> chi2(5) = 64.10407 (p=1.719042e-12)
#> --------------------------------------------
summary(heck1$lm)
#> 
#> Call:
#> lm(formula = YO ~ -1 + XO + imrData$IMR1, subset = YS == 1, weights = weightsNoNA)
#> 
#> Residuals:
#>      Min       1Q   Median       3Q      Max 
#> -3.09494 -0.30953  0.05341  0.36530  2.34770 
#> 
#> Coefficients:
#>                 Estimate Std. Error t value Pr(>|t|)    
#> XO(Intercept) -0.6143381  0.3768796  -1.630  0.10383    
#> XOeduc         0.1092363  0.0197062   5.543 5.24e-08 ***
#> XOexper        0.0419205  0.0136176   3.078  0.00222 ** 
#> XOI(exper^2)  -0.0008226  0.0004059  -2.026  0.04335 *  
#> XOcity         0.0510492  0.0692414   0.737  0.46137    
#> imrData$IMR1   0.0551177  0.2111916   0.261  0.79423    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 0.6674 on 422 degrees of freedom
#> Multiple R-squared:  0.7734,	Adjusted R-squared:  0.7702 
#> F-statistic:   240 on 6 and 422 DF,  p-value: < 2.2e-16
```

Use only variables that affect the selection process in the selection equation. Technically, the selection equation and the equation of interest could have the same set of regressors. But it is not recommended because we should only use variables (or at least one) in the selection equation that affect the selection process, but not the wage process (i.e., instruments). Here, variable `kids` fulfill that role: women with kids may be more likely to stay home, but working moms with kids would not have their wages change.

Alternatively,


```r
# ML estimation of selection model
ml1 = selection(
    selection = lfp ~ age + I(age ^ 2) + kids + huswage + educ,
    outcome = log(wage) ~ educ + exper + I(exper ^ 2) + city,
    data = Mroz87
) 
summary(ml1)
#> --------------------------------------------
#> Tobit 2 model (sample selection model)
#> Maximum Likelihood estimation
#> Newton-Raphson maximisation, 3 iterations
#> Return code 8: successive function values within relative tolerance limit (reltol)
#> Log-Likelihood: -914.0777 
#> 753 observations (325 censored and 428 observed)
#> 13 free parameters (df = 740)
#> Probit selection equation:
#>               Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) -4.1484037  1.4109302  -2.940 0.003382 ** 
#> age          0.1842132  0.0658041   2.799 0.005253 ** 
#> I(age^2)    -0.0023925  0.0007664  -3.122 0.001868 ** 
#> kids        -0.1488158  0.0384888  -3.866 0.000120 ***
#> huswage     -0.0434253  0.0123229  -3.524 0.000451 ***
#> educ         0.1255639  0.0229229   5.478 5.91e-08 ***
#> Outcome equation:
#>               Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) -0.5814781  0.3052031  -1.905  0.05714 .  
#> educ         0.1078481  0.0172998   6.234 7.63e-10 ***
#> exper        0.0415752  0.0133269   3.120  0.00188 ** 
#> I(exper^2)  -0.0008125  0.0003974  -2.044  0.04129 *  
#> city         0.0522990  0.0682652   0.766  0.44385    
#>    Error terms:
#>       Estimate Std. Error t value Pr(>|t|)    
#> sigma  0.66326    0.02309  28.729   <2e-16 ***
#> rho    0.05048    0.23169   0.218    0.828    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> --------------------------------------------
# summary(ml1$twoStep)
```

Manual


```r
myprob <- probit(lfp ~ age + I( age^2 ) + kids + huswage + educ, 
                 # x = TRUE, 
                 # iterlim = 30, 
                 data = Mroz87)
summary(myprob)
#> --------------------------------------------
#> Probit binary choice model/Maximum Likelihood estimation
#> Newton-Raphson maximisation, 4 iterations
#> Return code 1: gradient close to zero (gradtol)
#> Log-Likelihood: -482.8212 
#> Model: Y == '1' in contrary to '0'
#> 753 observations (325 'negative' and 428 'positive') and 6 free parameters (df = 747)
#> Estimates:
#>                Estimate  Std. error t value   Pr(> t)    
#> (Intercept) -4.18146681  1.40241567 -2.9816  0.002867 ** 
#> age          0.18608901  0.06517476  2.8552  0.004301 ** 
#> I(age^2)    -0.00241491  0.00075857 -3.1835  0.001455 ** 
#> kids        -0.14955977  0.03825079 -3.9100 9.230e-05 ***
#> huswage     -0.04303635  0.01220791 -3.5253  0.000423 ***
#> educ         0.12502818  0.02277645  5.4894 4.034e-08 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> Significance test:
#> chi2(5) = 64.10407 (p=1.719042e-12)
#> --------------------------------------------

imr <- invMillsRatio(myprob)
Mroz87$IMR1 <- imr$IMR1

manually_est <- lm(log(wage) ~ educ + exper + I( exper^2 ) + city + IMR1,
                   data = Mroz87, 
                   subset = (lfp == 1))

summary(manually_est)
#> 
#> Call:
#> lm(formula = log(wage) ~ educ + exper + I(exper^2) + city + IMR1, 
#>     data = Mroz87, subset = (lfp == 1))
#> 
#> Residuals:
#>      Min       1Q   Median       3Q      Max 
#> -3.09494 -0.30953  0.05341  0.36530  2.34770 
#> 
#> Coefficients:
#>               Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) -0.6143381  0.3768796  -1.630  0.10383    
#> educ         0.1092363  0.0197062   5.543 5.24e-08 ***
#> exper        0.0419205  0.0136176   3.078  0.00222 ** 
#> I(exper^2)  -0.0008226  0.0004059  -2.026  0.04335 *  
#> city         0.0510492  0.0692414   0.737  0.46137    
#> IMR1         0.0551177  0.2111916   0.261  0.79423    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 0.6674 on 422 degrees of freedom
#> Multiple R-squared:  0.1582,	Adjusted R-squared:  0.1482 
#> F-statistic: 15.86 on 5 and 422 DF,  p-value: 2.505e-14
```

Similarly,


```r
probit_selection <-
    glm(lfp ~ age + I( age^2 ) + kids + huswage + educ,
        data = Mroz87,
        family = binomial(link = 'probit'))

probit_lp <- -predict(probit_selection)
inv_mills <- dnorm(probit_lp) / (1 - pnorm(probit_lp))
Mroz87$inv_mills <- inv_mills


probit_outcome <-
    glm(
        log(wage) ~ educ + exper + I(exper ^ 2) + city + inv_mills,
        data = Mroz87,
        subset = (lfp == 1)
    )
summary(probit_outcome)
#> 
#> Call:
#> glm(formula = log(wage) ~ educ + exper + I(exper^2) + city + 
#>     inv_mills, data = Mroz87, subset = (lfp == 1))
#> 
#> Deviance Residuals: 
#>      Min        1Q    Median        3Q       Max  
#> -3.09494  -0.30953   0.05341   0.36530   2.34770  
#> 
#> Coefficients:
#>               Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) -0.6143383  0.3768798  -1.630  0.10383    
#> educ         0.1092363  0.0197062   5.543 5.24e-08 ***
#> exper        0.0419205  0.0136176   3.078  0.00222 ** 
#> I(exper^2)  -0.0008226  0.0004059  -2.026  0.04335 *  
#> city         0.0510492  0.0692414   0.737  0.46137    
#> inv_mills    0.0551179  0.2111918   0.261  0.79423    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> (Dispersion parameter for gaussian family taken to be 0.4454809)
#> 
#>     Null deviance: 223.33  on 427  degrees of freedom
#> Residual deviance: 187.99  on 422  degrees of freedom
#> AIC: 876.49
#> 
#> Number of Fisher Scoring iterations: 2
```


```r
library("stargazer")
library("Mediana")
library("plm")
# function to calculate corrected SEs for regression 
cse = function(reg) {
  rob = sqrt(diag(vcovHC(reg, type = "HC1")))
  return(rob)
}

# stargazer table
stargazer(
    # ols1,
    heck1,
    ml1,
    # manually_est,
    
    se = list(cse(ols1), NULL, NULL),
    title = "Married women's wage regressions",
    type = "text",
    df = FALSE,
    digits = 4,
    selection.equation = T
)
#> 
#> Married women's wage regressions
#> ===================================================
#>                           Dependent variable:      
#>                     -------------------------------
#>                                   lfp              
#>                         Heckman        selection   
#>                        selection                   
#>                           (1)             (2)      
#> ---------------------------------------------------
#> age                    0.1861***       0.1842***   
#>                                        (0.0658)    
#>                                                    
#> I(age2)                 -0.0024       -0.0024***   
#>                                        (0.0008)    
#>                                                    
#> kids                  -0.1496***      -0.1488***   
#>                                        (0.0385)    
#>                                                    
#> huswage                 -0.0430       -0.0434***   
#>                                        (0.0123)    
#>                                                    
#> educ                    0.1250         0.1256***   
#>                        (0.0130)        (0.0229)    
#>                                                    
#> Constant              -4.1815***      -4.1484***   
#>                        (0.2032)        (1.4109)    
#>                                                    
#> ---------------------------------------------------
#> Observations              753             753      
#> R2                      0.1582                     
#> Adjusted R2             0.1482                     
#> Log Likelihood                         -914.0777   
#> rho                     0.0830      0.0505 (0.2317)
#> Inverse Mills Ratio 0.0551 (0.2099)                
#> ===================================================
#> Note:                   *p<0.1; **p<0.05; ***p<0.01


stargazer(
    ols1,
    # heck1,
    # ml1,
    manually_est,
    
    se = list(cse(ols1), NULL, NULL),
    title = "Married women's wage regressions",
    type = "text",
    df = FALSE,
    digits = 4,
    selection.equation = T
)
#> 
#> Married women's wage regressions
#> ================================================
#>                         Dependent variable:     
#>                     ----------------------------
#>                              log(wage)          
#>                          (1)            (2)     
#> ------------------------------------------------
#> educ                  0.1057***      0.1092***  
#>                        (0.0130)      (0.0197)   
#>                                                 
#> exper                 0.0411***      0.0419***  
#>                        (0.0154)      (0.0136)   
#>                                                 
#> I(exper2)              -0.0008*      -0.0008**  
#>                        (0.0004)      (0.0004)   
#>                                                 
#> city                    0.0542        0.0510    
#>                        (0.0653)      (0.0692)   
#>                                                 
#> IMR1                                  0.0551    
#>                                      (0.2112)   
#>                                                 
#> Constant              -0.5308***      -0.6143   
#>                        (0.2032)      (0.3769)   
#>                                                 
#> ------------------------------------------------
#> Observations             428            428     
#> R2                      0.1581        0.1582    
#> Adjusted R2             0.1501        0.1482    
#> Residual Std. Error     0.6667        0.6674    
#> F Statistic           19.8561***    15.8635***  
#> ================================================
#> Note:                *p<0.1; **p<0.05; ***p<0.01
```

Rho is an estimate of the correlation of the errors between the selection and wage equations. In the lower panel, the estimated coefficient on the inverse Mills ratio is given for the Heckman model. The fact that it is not statistically different from zero is consistent with the idea that selection bias was not a serious problem in this case.

If the estimated coefficient of the inverse Mills ratio in the Heckman model is not statistically different from zero, then selection bias was not a serious problem.

#### Example 2

This code is from [R package sampleSelection](https://cran.r-project.org/web/packages/sampleSelection/vignettes/selection.pdf)


```r
set.seed(0)
library("sampleSelection")
library("mvtnorm")
eps <- rmvnorm(500, c(0,0), matrix(c(1,-0.7,-0.7,1), 2, 2)) # bivariate normal disturbances
xs <- runif(500)# uniformly distributed explanatory variable (vectors of explanatory variables for the selection )
ys <- xs + eps[,1] > 0 # probit data generating process
xo <- runif(500) # vectors of explanatory variables for outcome equation 
yoX <- xo + eps[,2] # latent outcome
yo <- yoX*(ys > 0) # observable outcome
# true intercepts = 0 and our true slopes = 1
# xs and xo are independent. Hence, exclusion restriction is fulfilled
summary( selection(ys~xs, yo ~xo))
#> --------------------------------------------
#> Tobit 2 model (sample selection model)
#> Maximum Likelihood estimation
#> Newton-Raphson maximisation, 5 iterations
#> Return code 1: gradient close to zero (gradtol)
#> Log-Likelihood: -712.3163 
#> 500 observations (172 censored and 328 observed)
#> 6 free parameters (df = 494)
#> Probit selection equation:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)  -0.2228     0.1081  -2.061   0.0399 *  
#> xs            1.3377     0.2014   6.642 8.18e-11 ***
#> Outcome equation:
#>               Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) -0.0002265  0.1294178  -0.002    0.999    
#> xo           0.7299070  0.1635925   4.462 1.01e-05 ***
#>    Error terms:
#>       Estimate Std. Error t value Pr(>|t|)    
#> sigma   0.9190     0.0574  16.009  < 2e-16 ***
#> rho    -0.5392     0.1521  -3.544 0.000431 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> --------------------------------------------
```

without the exclusion restriction, we generate yo using xs instead of xo.


```r
yoX <- xs + eps[,2]
yo <- yoX*(ys > 0)
summary(selection(ys ~ xs, yo ~ xs))
#> --------------------------------------------
#> Tobit 2 model (sample selection model)
#> Maximum Likelihood estimation
#> Newton-Raphson maximisation, 14 iterations
#> Return code 8: successive function values within relative tolerance limit (reltol)
#> Log-Likelihood: -712.8298 
#> 500 observations (172 censored and 328 observed)
#> 6 free parameters (df = 494)
#> Probit selection equation:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)  -0.1984     0.1114  -1.781   0.0756 .  
#> xs            1.2907     0.2085   6.191 1.25e-09 ***
#> Outcome equation:
#>             Estimate Std. Error t value Pr(>|t|)   
#> (Intercept)  -0.5499     0.5644  -0.974  0.33038   
#> xs            1.3987     0.4482   3.120  0.00191 **
#>    Error terms:
#>       Estimate Std. Error t value Pr(>|t|)    
#> sigma  0.85091    0.05352  15.899   <2e-16 ***
#> rho   -0.13226    0.72684  -0.182    0.856    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> --------------------------------------------
```

We can see that our estimates are still unbiased but standard errors are substantially larger. The exclusion restriction (i.e., independent information about the selection process) has a certain identifying power that we desire. Hence, it's better to have different set of variable for the selection process from the interested equation. Without the exclusion restriction, we solely rely on the functional form identification.

### Tobit-5

Also known as the switching regression model\
Condition: There is at least one variable in X in the selection process not included in the observed process. Used when there are separate models for participants, and non-participants.


```r
set.seed(0)
vc <- diag(3)
vc[lower.tri(vc)] <- c(0.9, 0.5, 0.1)
vc[upper.tri(vc)] <- vc[lower.tri(vc)]
eps <- rmvnorm(500, c(0,0,0), vc) # 3 disturbance vectors by a 3-dimensional normal distribution
xs <- runif(500) # uniformly distributed on [0, 1]
ys <- xs + eps[,1] > 0
xo1 <- runif(500) # uniformly distributed on [0, 1]
yo1 <- xo1 + eps[,2]
xo2 <- runif(500) # uniformly distributed on [0, 1]
yo2 <- xo2 + eps[,3]
```

exclusion restriction is fulfilled when x's are independent.


```r
summary(selection(ys~xs, list(yo1 ~ xo1, yo2 ~ xo2))) # one selection equation and a list of two outcome equations
#> --------------------------------------------
#> Tobit 5 model (switching regression model)
#> Maximum Likelihood estimation
#> Newton-Raphson maximisation, 11 iterations
#> Return code 1: gradient close to zero (gradtol)
#> Log-Likelihood: -895.8201 
#> 500 observations: 172 selection 1 (FALSE) and 328 selection 2 (TRUE)
#> 10 free parameters (df = 490)
#> Probit selection equation:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)  -0.1550     0.1051  -1.474    0.141    
#> xs            1.1408     0.1785   6.390 3.86e-10 ***
#> Outcome equation 1:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)  0.02708    0.16395   0.165    0.869    
#> xo1          0.83959    0.14968   5.609  3.4e-08 ***
#> Outcome equation 2:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)   0.1583     0.1885   0.840    0.401    
#> xo2           0.8375     0.1707   4.908 1.26e-06 ***
#>    Error terms:
#>        Estimate Std. Error t value Pr(>|t|)    
#> sigma1  0.93191    0.09211  10.118   <2e-16 ***
#> sigma2  0.90697    0.04434  20.455   <2e-16 ***
#> rho1    0.88988    0.05353  16.623   <2e-16 ***
#> rho2    0.17695    0.33139   0.534    0.594    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> --------------------------------------------
```

All the estimates are close to the true values.

Example of functional form misspecification


```r
set.seed(5)
eps <- rmvnorm(1000, rep(0, 3), vc)
eps <- eps^2 - 1 # subtract 1 in order to get the mean zero disturbances
xs <- runif(1000, -1, 0) # interval [−1, 0] to get an asymmetric distribution over observed choices
ys <- xs + eps[,1] > 0
xo1 <- runif(1000)
yo1 <- xo1 + eps[,2]
xo2 <- runif(1000)
yo2 <- xo2 + eps[,3]
summary(selection(ys~xs, list(yo1 ~ xo1, yo2 ~ xo2), iterlim=20))
#> --------------------------------------------
#> Tobit 5 model (switching regression model)
#> Maximum Likelihood estimation
#> Newton-Raphson maximisation, 4 iterations
#> Return code 3: Last step could not find a value above the current.
#> Boundary of parameter space?  
#> Consider switching to a more robust optimisation method temporarily.
#> Log-Likelihood: -1665.936 
#> 1000 observations: 760 selection 1 (FALSE) and 240 selection 2 (TRUE)
#> 10 free parameters (df = 990)
#> Probit selection equation:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) -0.53698    0.05808  -9.245  < 2e-16 ***
#> xs           0.31268    0.09395   3.328 0.000906 ***
#> Outcome equation 1:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) -0.70679    0.03573  -19.78   <2e-16 ***
#> xo1          0.91603    0.05626   16.28   <2e-16 ***
#> Outcome equation 2:
#>             Estimate Std. Error t value Pr(>|t|)  
#> (Intercept)   0.1446        NaN     NaN      NaN  
#> xo2           1.1196     0.5014   2.233   0.0258 *
#>    Error terms:
#>        Estimate Std. Error t value Pr(>|t|)    
#> sigma1  0.67770    0.01760   38.50   <2e-16 ***
#> sigma2  2.31432    0.07615   30.39   <2e-16 ***
#> rho1   -0.97137        NaN     NaN      NaN    
#> rho2    0.17039        NaN     NaN      NaN    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> --------------------------------------------
```

Although we still have an exclusion restriction (xo1 and xo2 are independent), we now have problems with the intercepts (i.e., they are statistically significantly different from the true values zero), and convergence problems.

If we don't have the exclusion restriction, we will have a larger variance of xs


```r
set.seed(6)
xs <- runif(1000, -1, 1)
ys <- xs + eps[,1] > 0
yo1 <- xs + eps[,2]
yo2 <- xs + eps[,3]
summary(tmp <- selection(ys~xs, list(yo1 ~ xs, yo2 ~ xs), iterlim=20))
#> --------------------------------------------
#> Tobit 5 model (switching regression model)
#> Maximum Likelihood estimation
#> Newton-Raphson maximisation, 16 iterations
#> Return code 8: successive function values within relative tolerance limit (reltol)
#> Log-Likelihood: -1936.431 
#> 1000 observations: 626 selection 1 (FALSE) and 374 selection 2 (TRUE)
#> 10 free parameters (df = 990)
#> Probit selection equation:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)  -0.3528     0.0424  -8.321 2.86e-16 ***
#> xs            0.8354     0.0756  11.050  < 2e-16 ***
#> Outcome equation 1:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) -0.55448    0.06339  -8.748   <2e-16 ***
#> xs           0.81764    0.06048  13.519   <2e-16 ***
#> Outcome equation 2:
#>             Estimate Std. Error t value Pr(>|t|)
#> (Intercept)   0.6457     0.4994   1.293    0.196
#> xs            0.3520     0.3197   1.101    0.271
#>    Error terms:
#>        Estimate Std. Error t value Pr(>|t|)    
#> sigma1  0.59187    0.01853  31.935   <2e-16 ***
#> sigma2  1.97257    0.07228  27.289   <2e-16 ***
#> rho1    0.15568    0.15914   0.978    0.328    
#> rho2   -0.01541    0.23370  -0.066    0.947    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> --------------------------------------------
```

Usually it will not converge. Even if it does, the results may be seriously biased.

**Note**

The log-likelihood function of the models might not be globally concave. Hence, it might not converge, or converge to a local maximum. To combat this, we can use

-   Different starting value
-   Different maximization methods.
-   refer to [Non-linear Least Squares] for suggestions.

##### Pattern-Mixture Models

-   compared to the Heckman's model where it assumes the value of the missing data is predetermined, pattern-mixture models assume missingness affect the distribution of variable of interest (e.g., Y)
-   To read more, you can check [NCSU](https://www4.stat.ncsu.edu/~davidian/st790/notes/chap6.pdf), [stefvanbuuren](https://stefvanbuuren.name/fimd/sec-nonignorable.html).
