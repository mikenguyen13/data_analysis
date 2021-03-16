# Endogeneity

Types of endogeneity

1.  [Endogenous Treatment]

-   Omitted Variables Bias\

    -   Motivation/choice\
    -   Ability/talent\
    -   Self-selection

-   Feedback Effect (Simultaneity): also known as bidirectionality

-   Measurement Error

2.  [Endogenous Sample Selection]

## Endogenous Treatment

Using the OLS estimates as a reference point


```r
library(AER)
```

```
## Loading required package: car
```

```
## Loading required package: carData
```

```
## Loading required package: lmtest
```

```
## Loading required package: zoo
```

```
## 
## Attaching package: 'zoo'
```

```
## The following objects are masked from 'package:base':
## 
##     as.Date, as.Date.numeric
```

```
## Loading required package: sandwich
```

```
## Loading required package: survival
```

```r
library(REndo)
```

```
## Registered S3 methods overwritten by 'lme4':
##   method                          from
##   cooks.distance.influence.merMod car 
##   influence.merMod                car 
##   dfbeta.influence.merMod         car 
##   dfbetas.influence.merMod        car
```

```r
set.seed(421)
data("CASchools")
school <- CASchools
school$stratio <- with(CASchools, students/teachers)
m1.ols <- lm(read ~ stratio + english + lunch + grades + income + calworks + county,
data=school)
summary(m1.ols)$coefficients[1:7,]
```

```
##                 Estimate Std. Error     t value      Pr(>|t|)
## (Intercept) 683.45305948 9.56214469  71.4748711 3.011667e-218
## stratio      -0.30035544 0.25797023  -1.1643027  2.450536e-01
## english      -0.20550107 0.03765408  -5.4576041  8.871666e-08
## lunch        -0.38684059 0.03700982 -10.4523759  1.427370e-22
## gradesKK-08  -1.91291321 1.35865394  -1.4079474  1.599886e-01
## income        0.71615378 0.09832843   7.2832829  1.986712e-12
## calworks     -0.05273312 0.06154758  -0.8567863  3.921191e-01
```

### Instrumental Variable

[A3a] requires $\epsilon_i$ to be uncorrelated with $\mathbf{x}_i$

Assume [A1][A1 Linearity] , [A2][A2 Full rank], [A5][A5 Data Generation (random Sampling)]

$$
plim(\hat{\beta}_{OLS}) = \beta + [E(\mathbf{x_i'x_i})]^{-1}E(\mathbf{x_i'}\epsilon_i)
$$ [A3a] is the weakest assumption needed for OLS to be **consistent**

[A3] fails when $x_{ik}$ is correlated with $\epsilon_i$

-   [Omitted Variables Bias] $\epsilon_i$ includes any other factors that may influence the dependent variable (linearly)
-   [Feedback Effect (Simultaneity)] Demand and prices are simultaneously determined.
-   [Endogenous sample design (sample selection)] we did not have iid sample
-   [Measurement Error]

**Note**

-   Omitted Variable: an omitted variable is a variable, omitted from the model (but is in the $\epsilon_i$) and unobserved has predictive power towards the outcome.\
-   Omitted Variable Bias: is the bias (and inconsistency when looking at large sample properties) of the OLS estimator when the omitted variable.

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

-   Find an **instrumental variable(s)** $z_{i2}$\

    -   Instrument Relevance**: when** $z_{i2}$ moves then $y_{i2}$ also moves\
    -   Instrument Exogeneity\*\*: when $z_{i2}$ moves then $\epsilon_i$ does not move.\

-   $z_{i2}$ is the **exogenous variation that identifies** the causal effect $\beta_2$

Finding an Instrumental variable:

-   Random Assignment: + Effect of class size on educational outcomes: instrument is initial random
-   Relation's Choice + Effect of Education on Fertility: instrument is parent's educational level
-   Eligibility + Trade-off between IRA and 401K retirement savings: instrument is 401k eligibility

**Example**

Return to College

-   education is correlated with ability - endogenous\

-   **Near 4year** as an instrument\

    -   Instrument Relevance: when **near** moves then education also moves\
    -   Instrument Exogeneity: when **near** moves then $\epsilon_i$ does not move.\

-   Other potential instruments; near a 2-year college. Parent's Education. Owning Library Card

$$
y_{i1}=\beta_0 + \mathbf{z}_{i1}\beta_1 + y_{i2}\beta_2 + \epsilon_i
$$

First Stage (Reduced Form) Equation:

$$
y_{i2} = \pi_0 + \mathbf{z_{i1}\pi_1} + \mathbf{z_{i2}\pi_2} + v_i
$$ where

-   $\pi_0 + \mathbf{z_{i1}\pi_1} + \mathbf{z_{i2}\pi_2}$ is exogenous variation $v_i$ is endogenous variation

This is called a **reduced form equation**\
\* Not interested in the causal interpretation of $\pi_1$ or $\pi_2$ \* A linear projection of $z_{i1}$ and $z_{i2}$ on $y_{i2}$ (simple correlations) \* The projections $\pi_1$ and $\pi_2$ guarantee that $E(z_{i1}'v_i)=0$ and $E(z_{i2}'v_i)=0$

Instrumental variable $z_{i2}$

-   **Instrument Relevance**: $\pi_2 \neq 0$
-   **Instrument Exogeneity**: $E(\mathbf{z_{i2}\epsilon_i})=0$

Moving only the exogenous part of $y_i2$ is moving

$$
\tilde{y}_{i2} = \pi_0 + \mathbf{z_{i1}\pi_1 + z_{i2}\pi_2}
$$

**two Stage Least Squares (2SLS)**

$$
y_{i1} = \beta_0 +\mathbf{z_{i1}\beta_1}+ y_{i2}\beta_2 + \epsilon_i \\
y_{i2} = \pi_0 + \mathbf{z_{i2}\pi_2} + \mathbf{v_i}
$$

Equivalently,

$$
\begin{equation}
y_{i1} = \beta_0 + \mathbf{z_{i1}}\beta_1 + \tilde{y}_{i2}\beta_2 + u_i
(\#eq:2SLS)
\end{equation}
$$ where

-   $\tilde{y}_{i2} =\pi_0 + \mathbf{z_{i2}\pi_2}$
-   $u_i = v_i \beta_2+ \epsilon_i$

The \@ref(eq:2SLS) holds for [A1][A1 Linearity], [A5][A5 Data Generation (random Sampling)]

-   [A2][A2 Full rank] holds if the instrument is relevant $\pi_2 \neq 0$ + $y_{i1} = \beta_0 + \mathbf{z_{i1}\beta_1 + (\pi_0 + z_{i1}\pi_1 + z_{i2}\pi_2)}\beta_2 + u_i$
-   [A3a] holds if the instrument is exogenous $E(\mathbf{z}_{i2}\epsilon_i)=0$

$$
\begin{align}
E(\tilde{y}_{i2}'u_i) &= E((\pi_0 + \mathbf{z_{i1}\pi_1+z_{i2}})(v_i\beta_2 + \epsilon_i)) \\
&= E((\pi_0 + \mathbf{z_{i1}\pi_1+z_{i2}})( \epsilon_i)) \\
&= E(\epsilon_i)\pi_0 + E(\epsilon_iz_{i1})\pi_1 + E(\epsilon_iz_{i2}) \\
&=0 
\end{align}
$$

Hence, \@ref(eq:2SLS) is consistent

The 2SLS Estimator\
1. Estimate the first stage using [OLS][Ordinary Least Squares]

$$
y_{i2} = \pi_0 + \mathbf{z_{i2}\pi_2} + \mathbf{v_i}
$$ and obtained estimated value $\hat{y}_{i2}$

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

        + **IV estimator**: one endogenous variable with a single instrument 
        + **2SLS estimator**: one endogenous variable with multiple instruments 
        + **GMM estimator**: multiple endogenous variables with multiple instruments
        

-   Standard errors produced in the second step are not correct\

    -   Because we do not know $\tilde{y}$ perfectly and need to estimate it in the firs step, we are introducing additional variation\
    -   We did not have this problem with [FGLS][Feasible Generalized Least Squares] because "the first stage was orthogonal to the second stage." This is generally not true for most multi-step procedure.\
    -   If [A4][A4 Homoskedasticity] does not hold, need to report robust standard errors.\

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

2.  [Testing Instrument's assumptions]\

    -   [Exogeneity]: Cannot always test (and when you can it might not be informative)\
    -   

##### Test of Endogeneity

-   2SLS is generally so inefficient that we may prefer OLS if there is not much endogeneity

-   Biased but inefficient vs efficient but biased

-   Want a sense of "how endogenous" $y_{i2}$ is\

    -   if "very" endgeneous - should use 2SLS\
    -   if not "very" endogenous - perhaps prefer OLS

**Invalid** Test of Endogeneity \* $y_{i2}$ is endogenous if it is correlated with $\epsilon_i$,

$$
\epsilon_i = \gamma_0 + y_{i2}\gamma_1 + error_i
$$ where $\gamma_1 \neq 0$ implies that there is endogeneity

-   $\epsilon_i$ is not observed, but using the residuals

$$
e_i = \gamma_0 + y_{i2}\gamma_1 + error_i
$$ is **NOT** a valid test of endogeneity + The OLS residual, e is mechanically uncorrelated with $y_{i2}$ (by FOC for OLS) + In every situation, $\gamma_1$ will be essentially 0 and you will never be able to reject the null of no endogeneity

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
\begin{split}
H_0: \theta = 0 && \text{  (not endogenous)} \\
H_1: \theta \neq 0 && \text{  (endogenous)}
\end{split}
$$

##### Testing Instrument's assumptions

The instrumental variable must satisfy

1.  [Exogeneity]
2.  [Relevancy](need%20to%20avoid "weak instruments")

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
$$ is **NOT** a valid test of endogeneity\
\* the OLS residual, e is mechanically uncorrelated with $z_{i2}$: $\hat{\gamma}_1$ will be essentially 0 and you will never be able to determine if the instrument is endogenous.

**Solution**

Testing Instrumental Exegeneity in an Over-identified Model \* When there is more than one exogenous instrument (per endogenous variable), we can test for instrument exogeneity.\
+ When we have multiple instruments, the model is said to be over-identiifed.\
+ Could estimate the same model several ways (i.e., can identify/ estimate $\beta_1$ more than one way)\
\* Idea behind the test: if the controls and instruments are truly exogenous then OLS estimation of the following regression,

$$
\epsilon_i = \gamma_0 + \mathbf{z}_{i1}\gamma_1 + \mathbf{z}_{i2}\gamma_2 + error_i
$$ should have a very low $R^2$\
\* if the model is **just identified** (one instrument per endogenous variable) then the $R^2 = 0$

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

+-----------------+-------------------------------------------------------------------------------------+
| Result          | Implication                                                                         |
+=================+=====================================================================================+
| reject the null | you can be pretty sure there is an endogenous instrument, but don't know which one. |
+-----------------+-------------------------------------------------------------------------------------+
| fail to reject  | could be either (1) they are both exogenous, (2) they are both endogenous.          |
+-----------------+-------------------------------------------------------------------------------------+

###### Relevancy

Why Relevance matter?

$$
\pi_2 \neq 0 
$$ \* used to show [A2][A2 Full rank] holds + If $\pi_2 = 0$ (instrument is not relevant) then [A2][A2 Full rank] fails - perfect multicollinearity\
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
$$ + Then the OLS estimator is no longer unbiased or consistent.\
\* If we have valid instruments $\mathbf{z}_{i2}$ + [Exogeneity]: $E(\mathbf{z}_{i2}'\epsilon_i) = 0$ + [Relevancy](need%20to%20avoid "weak instruments"): $\pi_2 \neq 0$ Then the 2SLS estimator is consistent under [A1][A1 Linearity], [A2][A2 Full rank], [A5a], and the above two conditions. + If [A4][A4 Homoskedasticity] also holds, then the usual standard errors are valid. + If [A4][A4 Homoskedasticity] does not hold then use the robust standard errors.

$$
y_{i1}=\beta_0 + \mathbf{z}_{i1}\beta_1 + y_{i2}\beta_2 + \epsilon_i \\
y_{i2} = \pi_0 + \mathbf{z_{i1}\pi_1} + \mathbf{z_{i2}\pi_2} + v_i
$$ \* When [A3a] does hold

$$
E(y_{i2}'\epsilon_i) = 0
$$ and we have valid instruments, then both the OLS and 2SLS estimators are consistent.\
+ The OLS estimator is always more efficient + can use the variable addition test to determine if 2SLS is need (A3a does hold) or if OLS is valid (A3a does not hold)

Sometimes we can test the assumption for instrument to be valid:\
+ [Exogeneity]: Only table when there are more instruments than endogenous variables. + [Relevancy](need%20to%20avoid "weak instruments"): Always testable, need the F-stat to be greater than 10 to rule out a weak instrument

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
```

```
##                 Estimate  Std. Error     t value      Pr(>|t|)
## (Intercept) 700.47891593 13.58064436  51.5792106 8.950497e-171
## stratio      -1.13674002  0.53533638  -2.1234126  3.438427e-02
## english      -0.21396934  0.03847833  -5.5607753  5.162571e-08
## lunch        -0.39384225  0.03773637 -10.4366757  1.621794e-22
## gradesKK-08  -1.89227865  1.37791820  -1.3732881  1.704966e-01
## income        0.62487986  0.11199008   5.5797785  4.668490e-08
## calworks     -0.04950501  0.06244410  -0.7927892  4.284101e-01
```

### Internal instrumental variable

(also **instrument free methods**). This section is based on Raluca Gui's [guide](https://cran.r-project.org/web/packages/REndo/vignettes/REndo-introduction.pdf)

alternative to external instrumental variable approaches

All approaches here assume a **continuous dependent variable**

**Application**

#### Non-hierarchical Data (Cross-classified)

$$
Y_t = \beta_0 + \beta_1 P_t + \beta_2 X_t + \epsilon_t
$$

where

-   $t = 1, .., T$ (indexes either time or cross-sectional units)\
-   $Y_t$ is a k x 1 response variable\
-   $X_t$ is a k x n exogenous regressor\
-   $P_t$ is a k x 1 continuous endogenous regressor\
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

-   in the [LIV](#latent-instrumental-variable) model the distribution of $Z_t$ is discrete\
-   in the [Higher Moments Method] and [Joint Estimation Using Copula] methods, the distribution of $Z_t$ is taken to be skewed.

$Z_t$ are assumed **unobserved, discrete and exogenous**, with

-   an unknown number of groups m\
-   $\gamma$ is a vector of group means.

Identification of the parameters relies on the distributional assumptions of

-   $P_t$: a non-Gaussian distribution\
-   $Z_t$ discrete with $m \ge 2$

Note:

-   If $Z_t$ is continuous, the model is unidentified\
-   If $P_t \sim N$, you have inefficient estimates.


```r
m3.liv <- latentIV(read ~ stratio, data=school)
```

```
## No start parameters were given. The linear model read ~ stratio is fitted to derive them.
```

```
## The start parameters c((Intercept)=706.449, stratio=-2.621, pi1=19.64, pi2=21.532, theta5=0.5, theta6=1, theta7=0.5, theta8=1) are used for optimization.
```

```r
summary(m3.liv)$coefficients[1:7,]
```

```
##                   Estimate    Std. Error       z-score     Pr(>|z|)
## (Intercept)   6.996014e+02  2.686186e+02  2.604441e+00 9.529597e-03
## stratio      -2.272673e+00  1.367757e+01 -1.661605e-01 8.681108e-01
## pi1          -4.896363e+01  5.526907e-08 -8.859139e+08 0.000000e+00
## pi2           1.963920e+01  9.225351e-02  2.128830e+02 0.000000e+00
## theta5       6.939432e-152 3.354672e-160  2.068587e+08 0.000000e+00
## theta6        3.787512e+02  4.249457e+01  8.912932e+00 1.541524e-17
## theta7       -1.227543e+00  4.885276e+01 -2.512741e-02 9.799653e-01
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

-   skewed $P_t$\

-   the recovery of the correct parameter estimates\

-   $\epsilon_t \sim$ normal marginal distribution. The marginal distribution of $P_t$ is obtained using the **Epanechnikov kernel density estimator**\
    $$
    \hat{h}_p = \frac{1}{T . b} \sum_{t=1}^TK(\frac{p - P_t}{b})
    $$ where

-   $P_t$ = endogenous variables\

-   $K(x) = 0.75(1-x^2)I(||x||\le 1)$\

-   $b=0.9T^{-1/5}\times min(s, IQR/1.34)$ suggested by [@Silverman_1969]\

    -   IQR = interquartile range\
    -   s = sample standard deviation\
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
```

```
## Warning: It is recommended to run 1000 or more bootstraps.
```

```r
summary(m4.cc)$coefficients[1:7, ]
```

```
##             Point Estimate   Boots SE Lower Boots CI (95%) Upper Boots CI (95%)
## (Intercept)   683.06900891 2.80554212                   NA                   NA
## stratio        -0.32434608 0.02075999                   NA                   NA
## english        -0.21576110 0.01450666                   NA                   NA
## lunch          -0.37087664 0.01902052                   NA                   NA
## calworks       -0.05569058 0.02076781                   NA                   NA
## gradesKK-08    -1.92286128 0.25684614                   NA                   NA
## income          0.73595353 0.04725700                   NA                   NA
```

we run this model with only one endogenous continuous regressor (`stratio`). Sometimes, the code will not converge, in which case you can use different

-   optimization algorithm\
-   starting values\
-   maximum number of iterations

##### Higher Moments Method

suggested by [@Lewbel_1997] to identify $\epsilon_t$ caused by **measurement error**.

Identification is achieved by using third moments of the data, with no restrictions on the distribution of $\epsilon_t$\
The following instruments can be used with 2SLS estimation to obtain consistent estimates:

$$
\begin{align}
q_{1t} &=  (G_t - \bar{G}) \\
q_{2t} &=  (G_t - \bar{G})(P_t - \bar{P}) \\
q_{3t} &=   (G_t - \bar{G})(Y_t - \bar{Y})\\
q_{4t} &=  (Y_t - \bar{Y})(P_t - \bar{P}) \\
q_{5t} &=  (P_t - \bar{P})^2 \\
q_{6t} &=  (Y_t - \bar{Y})^2 \\
\end{align}
$$

where

-   $G_t = G(X_t)$ for any given function G that has finite third own and cross moments\
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
```

```
## Residuals were derived by fitting stratio ~ english + lunch + calworks + income + grades + county.
```

```
## Warning: A studentized Breusch-Pagan test (stratio ~ english) indicates at a 95%
## confidence level that the assumption of heteroscedasticity for the variable is
## not satisfied (p-value: 0.2428). The instrument built from it therefore is weak.
```

```
## The following internal instruments were built: IIV(income), IIV(english).
```

```
## Fitting an instrumental variable regression with model read ~ stratio + english + lunch + calworks + income + grades + |english + lunch + calworks + income + grades + county + IIV(income) + IIV(english)    county|english + lunch + calworks + income + grades + county + IIV(income) + IIV(english).
```

```r
summary(m5.hetEr)$coefficients[1:7, ]
```

```
##                 Estimate  Std. Error    t value     Pr(>|t|)
## (Intercept) 662.78791557 27.90173069 23.7543657 2.380436e-76
## stratio       0.71480686  1.31077325  0.5453322 5.858545e-01
## english      -0.19522271  0.04057527 -4.8113717 2.188618e-06
## lunch        -0.37834232  0.03927793 -9.6324402 9.760809e-20
## calworks     -0.05665126  0.06302095 -0.8989273 3.692776e-01
## income        0.82693755  0.17236557  4.7975797 2.335271e-06
## gradesKK-08  -1.93795843  1.38723186 -1.3969968 1.632541e-01
```

recommend using this approach to create additional instruments to use with external ones for better efficiency.

##### Heteroskedastic Error Approach

-   using means of variables that are uncorrelated with the product of heteroskedastic errors to identify structural parameters.\
-   This method can be use either when you don't have external instruments or you want to use additional instruments to improve the efficiency of the IV estimator [@Lewbel_2012]\
-   The instruments are constructed as simple functions of data\
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
$$ where $cov((Z-\bar{Z})v,v)$ is the degree of heteroskedasticity of ν with respect to Z [@Lewbel_2012], which can be empirically tested.

If it is zero or close to zero (i.e.,the instrument is weak), you might have imprecise estimates, with large standard errors.

-   Under homoskedasticity, the parameters of the model are unidentified.\
-   Under heteroskedasticity related to at least some elements of X, the parameters of the model are identified.

#### Hierarchical Data

Multiple independent assumptions involving various random components at different levels mean that any moderate correlation between some predictors and a random component or error term can result in a significant bias of the coefficients and of the variance components. [@Kim_2007] proposed a generalized method of moments which uses both, the between and within variations of the exogenous variables, but only assumes the within variation of the variables to be endogenous.

**Assumptions**

-   the errors at each level $\sim iid N$\
-   the slope variables are exogenous\
-   the level-1 $\epsilon \perp X, P$. If this is not the case, additional, external instruments are necessary

**Hierarchical Model**

$$
Y_{cst} = Z_{cst}^1 \beta_{cs}^1 + X_{cst}^1 \beta_1 + \epsilon_{cst}^1 \\
\beta^1_{cs} = Z_{cs}^2 \beta_{c}^2 + X_{cst}^2 \beta_2 + \epsilon_{cst}^2 \\
\beta^2_{c} = X^3_c \beta_3 + \epsilon_c^3
$$

Bias could stem from:

-   errors at the higher two levels ($\epsilon_c^3,\epsilon_{cst}^2$) are correlated with some of the regressors\
-   only third level errors ($\epsilon_c^3$) are correlated with some of the regressors

[@Kim_2007] proposed

-   When all variables are assumed exogenous, the proposed estimator equals the random effects estimator\
-   When all variables are assumed endogenous, it equals the fixed effects estimator\
-   also use omitted variable test (based on the Hausman-test [@Hausman_1978] for panel data), which allows the comparison of a robust estimator and an estimator that is efficient under the null hypothesis of no omitted variables or the comparison of two robust estimators at different levels.


```r
set.seed(113)
school$gr08 <- school$grades == "KK-06"
m7.multilevel <-
        multilevelIV(read ~ stratio + english + lunch + income + gr08 +
                             calworks + (1 | county) | endo(stratio),
                     data = school)
```

```
## Fitting linear mixed-effects model read ~ stratio + english + lunch + income + gr08 + calworks +     (1 | county).
```

```
## Detected multilevel model with 2 levels.
```

```
## For county (Level 2), 45 groups were found.
```

```r
summary(m7.multilevel)$coefficients[1:7, ]
```

```
##                Estimate Std. Error     z-score     Pr(>|z|)
## (Intercept) 675.8228656 5.58008680 121.1133248 0.000000e+00
## stratio      -0.4956054 0.23922638  -2.0717005 3.829339e-02
## english      -0.2599777 0.03413530  -7.6160948 2.614656e-14
## lunch        -0.3692954 0.03560210 -10.3728537 3.295342e-25
## income        0.6723141 0.08862012   7.5864728 3.287314e-14
## gr08TRUE      2.1590333 1.28167222   1.6845440 9.207658e-02
## calworks     -0.0570633 0.05711701  -0.9990596 3.177658e-01
```

Another example using simulated data

-   level-1 regressors: $X_{11}, X_{12}, X_{13}, X_{14}, X_{15}$, where $X_{15}$ is correlated with the level-2 error (i.e., endogenous).\
-   level-2 regressors: $X_{21}, X_{22}, X_{23}, X_{24}$\
-   level-3 regressors: $X_{31}, X_{32}, X_{33}$

We estimate a three-level model with X15 assumed endogenous. Having a three-level hierarchy, `multilevelIV()` returns five estimators, from the most robust to omitted variables (FE_L2), to the most efficient (REF) (i.e. lowest mean squared error).

-   The random effects estimator (REF) is efficient assuming no omitted variables
-   The fixed effects estimator (FE) is unbiased and asymptotically normal even in the presence of omitted variables.\
-   Because of the efficiency, the random effects estimator is preferable if you think there is no omitted. variables\
-   The robust estimator would be preferable if you think there is omitted variables.


```r
data(dataMultilevelIV)
set.seed(114)
formula1 <-
        y ~ X11 + X12 + X13 + X14 + X15 + X21 + X22 + X23 + X24 +
        X31 + X32 + X33 + (1 | CID) + (1 | SID) | endo(X15)
m8.multilevel <-
        multilevelIV(formula = formula1, data = dataMultilevelIV)
```

```
## Fitting linear mixed-effects model y ~ X11 + X12 + X13 + X14 + X15 + X21 + X22 + X23 + X24 + X31 +     X32 + X33 + (1 | CID) + (1 | SID).
```

```
## Detected multilevel model with 3 levels.
```

```
## For CID (Level 2), 1368 groups were found.
```

```
## For SID (Level 3), 40 groups were found.
```

```r
coef(m8.multilevel)
```

```
##                    REF      FE_L2      FE_L3     GMM_L2     GMM_L3
## (Intercept) 64.3168856  0.0000000  0.0000000 64.3485944 64.3168868
## X11          3.0213405  3.0459605  3.0214255  3.0146686  3.0213403
## X12          8.9522160  8.9839088  8.9524723  8.9747533  8.9522169
## X13         -2.0194178 -2.0145054 -2.0193321 -2.0021426 -2.0194171
## X14          1.9651420  1.9791437  1.9648317  1.9658681  1.9651421
## X15         -0.5647915 -0.9777361 -0.5647621 -0.9750309 -0.5648070
## X21         -2.3316225  0.0000000 -2.2845297 -2.3052516 -2.3316215
## X22         -3.9564944  0.0000000 -3.9553644 -4.0130975 -3.9564966
## X23         -2.9779887  0.0000000 -2.9756848 -2.9488487 -2.9779876
## X24          4.9078293  0.0000000  4.9084694  4.7933756  4.9078250
## X31          2.1142348  0.0000000  0.0000000  2.1164477  2.1142349
## X32          0.3934770  0.0000000  0.0000000  0.3799626  0.3934764
## X33          0.1082086  0.0000000  0.0000000  0.1108386  0.1082087
```


```r
summary(m8.multilevel, "REF")
```

```
## 
## Call:
## multilevelIV(formula = formula1, data = dataMultilevelIV)
## 
## Number of levels: 3
## Number of observations: 2824
## Number of groups: L2(CID): 1368  L3(SID): 40
## 
## Coefficients for model REF:
##             Estimate Std. Error z-score Pr(>|z|)    
## (Intercept) 64.31689    7.87332   8.169 3.11e-16 ***
## X11          3.02134    0.02576 117.306  < 2e-16 ***
## X12          8.95222    0.02572 348.131  < 2e-16 ***
## X13         -2.01942    0.02409 -83.835  < 2e-16 ***
## X14          1.96514    0.02521  77.937  < 2e-16 ***
## X15         -0.56479    0.01950 -28.962  < 2e-16 ***
## X21         -2.33162    0.16228 -14.368  < 2e-16 ***
## X22         -3.95649    0.13119 -30.160  < 2e-16 ***
## X23         -2.97799    0.06611 -45.044  < 2e-16 ***
## X24          4.90783    0.19796  24.792  < 2e-16 ***
## X31          2.11423    0.10433  20.264  < 2e-16 ***
## X32          0.39348    0.30426   1.293   0.1959    
## X33          0.10821    0.05236   2.067   0.0388 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Omitted variable tests for model REF:
##               df     Chisq  p-value    
## GMM_L2_vs_REF  7     18.74 0.009040 ** 
## GMM_L3_vs_REF 13 -12872.98 1.000000    
## FE_L2_vs_REF  13     39.99 0.000139 ***
## FE_L3_vs_REF  13     39.99 0.000138 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

True $\beta_{X_{15}} =-1$. We can see that some estimators are bias because $X_{15}$ is correlated with the level-two error, to which only FE_L2 and GMM_L2 are robust

To select the appropriate estimator, we use the omitted variable test.

In a three-level setting, we can have different estimator comparisons:

-   Fixed effects vs. random effects estimators: Test for omitted level-two and level-three omitted effects, simultaneously, one compares FE_L2 to REF. But we will not know at which omitted variables exist.\
-   Fixed effects vs. GMM estimators: Once the existence of omitted effects is established but not sure at which level, we test for level-2 omitted effects by comparing FE_L2 vs GMM_L3. If you reject the null, the omitted variables are at level-2 The same is accomplished by testing FE_L2 vs. GMM_L2, since the latter is consistent only if there are no omitted effects at level-2.\
-   Fixed effects vs. fixed effects estimators: We can test for omitted level-2 effects, while allowing for omitted level-3 effects by comparing FE_L2 vs. FE_L3 since FE_L2 is robust against both level-2 and level-3 omitted effects while FE_L3 is only robust to level-3 omitted variables.

Summary, use the omitted variable test comparing `REF vs. FE_L2` first.

-   If the null hypothesis is rejected, then there are omitted variables either at level-2 or level-3

-   Next, test whether there are level-2 omitted effects, since testing for omitted level three effects relies on the assumption there are no level-two omitted effects. You can use any of these pair of comparisons:\

    -   `FE_L2 vs. FE_L3`\
    -   `FE_L2 vs. GMM_L2`\

-   If no omitted variables at level-2 are found, test for omitted level-3 effects by comparing either\

    -   FE_L3 vs. GMM_L3\
    -   GMM_L2 vs. GMM_L3


```r
summary(m8.multilevel, "REF")
```

```
## 
## Call:
## multilevelIV(formula = formula1, data = dataMultilevelIV)
## 
## Number of levels: 3
## Number of observations: 2824
## Number of groups: L2(CID): 1368  L3(SID): 40
## 
## Coefficients for model REF:
##             Estimate Std. Error z-score Pr(>|z|)    
## (Intercept) 64.31689    7.87332   8.169 3.11e-16 ***
## X11          3.02134    0.02576 117.306  < 2e-16 ***
## X12          8.95222    0.02572 348.131  < 2e-16 ***
## X13         -2.01942    0.02409 -83.835  < 2e-16 ***
## X14          1.96514    0.02521  77.937  < 2e-16 ***
## X15         -0.56479    0.01950 -28.962  < 2e-16 ***
## X21         -2.33162    0.16228 -14.368  < 2e-16 ***
## X22         -3.95649    0.13119 -30.160  < 2e-16 ***
## X23         -2.97799    0.06611 -45.044  < 2e-16 ***
## X24          4.90783    0.19796  24.792  < 2e-16 ***
## X31          2.11423    0.10433  20.264  < 2e-16 ***
## X32          0.39348    0.30426   1.293   0.1959    
## X33          0.10821    0.05236   2.067   0.0388 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Omitted variable tests for model REF:
##               df     Chisq  p-value    
## GMM_L2_vs_REF  7     18.74 0.009040 ** 
## GMM_L3_vs_REF 13 -12872.98 1.000000    
## FE_L2_vs_REF  13     39.99 0.000139 ***
## FE_L3_vs_REF  13     39.99 0.000138 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

```r
# compare REF with all the other estimators. Testing REF (the most efficient estimator) against FE_L2 (the most robust estimator), equivalently we are testing simultaneously for level-2 and level-3 omitted effects. 
```

Since the null hypothesis is rejected (p = 0.000139), there is bias in the random effects estimator.

To test for level-2 omitted effects (regardless of level-3 omitted effects), we compare FE_L2 versus FE_L3


```r
summary(m8.multilevel,"FE_L2")
```

```
## 
## Call:
## multilevelIV(formula = formula1, data = dataMultilevelIV)
## 
## Number of levels: 3
## Number of observations: 2824
## Number of groups: L2(CID): 1368  L3(SID): 40
## 
## Coefficients for model FE_L2:
##               Estimate Std. Error z-score Pr(>|z|)    
## (Intercept)  0.000e+00  4.275e-19    0.00        1    
## X11          3.046e+00  2.978e-02  102.30   <2e-16 ***
## X12          8.984e+00  3.360e-02  267.41   <2e-16 ***
## X13         -2.015e+00  3.107e-02  -64.83   <2e-16 ***
## X14          1.979e+00  3.203e-02   61.80   <2e-16 ***
## X15         -9.777e-01  3.364e-02  -29.06   <2e-16 ***
## X21          0.000e+00  1.824e-18    0.00        1    
## X22          0.000e+00  1.303e-18    0.00        1    
## X23          0.000e+00  4.389e-18    0.00        1    
## X24          0.000e+00  1.724e-18    0.00        1    
## X31          0.000e+00  1.468e-17    0.00        1    
## X32          0.000e+00  8.265e-18    0.00        1    
## X33          0.000e+00  2.793e-17    0.00        1    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Omitted variable tests for model FE_L2:
##                 df Chisq  p-value    
## FE_L2_vs_REF    13 39.99 0.000139 ***
## FE_L2_vs_FE_L3   9 36.02 3.92e-05 ***
## FE_L2_vs_GMM_L2 12 39.99 7.21e-05 ***
## FE_L2_vs_GMM_L3 13 39.99 0.000139 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

The null hypothesis of no omitted level-2 effects is rejected ($p = 3.92e − 05$). Hence, there are omitted effects at level-two. We should use FE_L2 which is consistent with the underlying data that we generated (level-2 error correlated with $X_15$, which leads to biased FE_L3 coefficients.

The omitted variable test between FE_L2 and GMM_L2 should reject the null hypothesis of no omitted level-2 effects (p-value is 0).

If we assume an endogenous variable as exogenous, the RE and GMM estimators will be biased because of the wrong set of internal instrumental variables. To increase our confidence, we should compare the omitted variable tests when the variable is considered endogenous vs. exogenous to get a sense whether the variable is truly endogenous.

### Proxy Variables

Can be in place of the omitted variable,\
\* will not be able to estimate the effect of the omitted variable \* will be able to reduce some endogeneity caused bye the omitted variable

Criteria for a proxy variable:

1.  The proxy is correlated with the omitted variable.
2.  Having the omitted variable in the regression will solve the problem of endogeneity 3.The variation of the omitted variable unexplained by the proxy is uncorrelated with all independent variables, including the proxy.

IQ test can be a proxy for ability in the regression between wage explained education.

For the third requirement

$$
ability = \gamma_0 + \gamma_1 IQ + \epsilon
$$

where $\epsilon$ is uncorrelated with education and IQ test.

## Endogenous Sample Selection

sample selection or self-selection problem

the omitted variable is how people were selected into the sample

Some disciplines consider nonresponse bias and selection bias as sample selection.

-   When unobservable factors that affect who is in the sample are independent of unobservable factors that affect the outcome, the sample selection is not endogenous. Hence, the sample selection is ignorable and estimator that ignores sample selection is still consistent.\
-   when the unobservable factors that affect who is included in the sample are correlated with the unobservable factors that affect the outcome, the sample selection is endogenous and not ignorable, because estimators that ignore endogenous sample selection are not consistent (we don't know which part of the observable outcome is related to the causal relationship and which part is due to different people were selected for the treatment and control groups).

To combat Sample selection, we can

-   Randomization: participants are randomly selected into treatment and control.\
-   Instruments that determine the treatment status (i.e., treatment vs. control) but not the outcome (Y)\
-   Functional form of the selection and outcome processes: originated from [@Heckman_1976], later on generalize by [@Amemiya_1984]

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
$$ Equivalently, $z_i = 1$ ($y_i$ is observed) when

$$
u_i \ge -w_i \gamma
$$ Hence, the probability of observed $y_i$ is

$$
\begin{align}
P(u_i \ge -w_i \gamma) &= 1 - \Phi(-w_i \gamma) \\
&= \Phi(w_i \gamma) && \text{symmetry of the standard normal distribution}
\end{align}
$$

We will **assume**

-   the error term of the selection $\mathbf{u \sim N(0,I)}$\
-   $Var(u_i) = 1$ for identification purposes

Visually, $P(u_i \ge -w_i \gamma)$ is the shaded area.


```r
x=seq(-3,3,length=200)
y=dnorm(x,mean=0,sd=1)
plot(x,y,type="l", main = bquote("Probabibility distribution of"~u[i]))
x=seq(0.3,3,length=100)
y=dnorm(x,mean=0,sd=1)
polygon(c(0.3,x,3),c(0,y,0),col="gray")
text(1,0.1,bquote(1 - Phi~(-w[i]~gamma)))
arrows(-0.5,0.1,0.3,0,length=.15)
text(-0.5,0.12,bquote(-w[i]~gamma))
legend("topright", "Gray = Prob of Observed", pch=1, title= "legend",inset = .02)
```

<img src="08-endogeneity_files/figure-html/unnamed-chunk-11-1.png" width="672" />

Hence in our observed model, we see

$$
\begin{equation}
y_i = x_i\beta + \epsilon_i \text{when $z_i=1$}
\end{equation}
$$

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
{c}
1 & \rho \\
\rho & \sigma^2_{\epsilon} \\
\end{array}
\right]
\right)
$$

The relation between the observed and selection models:

$$
\begin{align}
E(y_i | y_i \text{ observed}) &= E(y_i| z^*>0) \\
&= E(y_i| -w_i \gamma) \\
&= \mathbf{x}_i \beta + E(\epsilon_i | u_i > -w_i \gamma) \\
&= \mathbf{x}_i \beta + \rho \sigma_\epsilon \frac{\phi(w_i \gamma)}{\Phi(w_i \gamma)}
\end{align}
$$ where $\frac{\phi(w_i \gamma)}{\Phi(w_i \gamma)}$ is the Inverse Mills Ratio. and $\rho \sigma_\epsilon \frac{\phi(w_i \gamma)}{\Phi(w_i \gamma)} \ge 0$

Great visualization of special cases of correlation patterns amongst data and errors by professor [Rob Hick](https://rlhick.people.wm.edu/stories/econ_407_notes_heckman.html)

Note:

[@Bareinboim_2014] is an excellent summary of cases that we can still do causal inference in case of selection bias. I'll try to summarize their idea here:

Let X be an action, Y be an outcome, and S be a binary indicator of entry into the data pool where (S = 1 = in the sample, S = 0 =out of sample) and Q be the conditional distribution $Q = P(y|x)$.

Usually we want to understand , but because of S, we only have $P(y, x|S = 1)$. Hence, we'd like to recover $P(y|x)$ from $P(y, x|S = 1)$

-   If both X and Y affect S, we can't unbiasedly estimate $P(y|x)$

In the case of Omitted variable bias (U) and sample selection bias (S), you have unblocked extraneous "flow" of information between X and Y, which causes spurious correlation for X and Y. Traditionally, we would recover Q by parametric assumption of

(1) the data generating process (e.g., Heckman 2-step)\
(2) type of data-generating model (e..g, treatment-dependent or outcome-dependent)\
(3) selection's probability $P(S = 1|P a_s)$ with non-parametrically based causal graphical models, the authors proposed more robust way to model misspecification regardless of the type of data-generating model, and do not require selection's probability. Hence, you can recover Q

-   without external data
-   with external data\
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
```

```
## Loading required package: maxLik
```

```
## Loading required package: miscTools
```

```
## 
## Please cite the 'maxLik' package as:
## Henningsen, Arne and Toomet, Ott (2011). maxLik: A package for maximum likelihood estimation in R. Computational Statistics 26(3), 443-458. DOI 10.1007/s00180-010-0217-1.
## 
## If you have questions, suggestions, or comments regarding the 'maxLik' package, please use a forum or 'tracker' at maxLik's R-Forge site:
## https://r-forge.r-project.org/projects/maxlik/
```

```r
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following object is masked from 'package:car':
## 
##     recode
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
data("Mroz87") #1975 data on married women’s pay and labor-force participation from the Panel Study of Income Dynamics (PSID)
head(Mroz87)
```

```
##   lfp hours kids5 kids618 age educ   wage repwage hushrs husage huseduc huswage
## 1   1  1610     1       0  32   12 3.3540    2.65   2708     34      12  4.0288
## 2   1  1656     0       2  30   12 1.3889    2.65   2310     30       9  8.4416
## 3   1  1980     1       3  35   12 4.5455    4.04   3072     40      12  3.5807
## 4   1   456     0       3  34   12 1.0965    3.25   1920     53      10  3.5417
## 5   1  1568     1       2  31   14 4.5918    3.60   2000     32      12 10.0000
## 6   1  2032     0       0  54   12 4.7421    4.70   1040     57      11  6.7106
##   faminc    mtr motheduc fatheduc unem city exper  nwifeinc wifecoll huscoll
## 1  16310 0.7215       12        7  5.0    0    14 10.910060    FALSE   FALSE
## 2  21800 0.6615        7        7 11.0    1     5 19.499981    FALSE   FALSE
## 3  21040 0.6915       12        7  5.0    0    15 12.039910    FALSE   FALSE
## 4   7300 0.7815        7        7  5.0    0     6  6.799996    FALSE   FALSE
## 5  27300 0.6215       12       14  9.5    1     7 20.100058     TRUE   FALSE
## 6  19495 0.6915       14        7  7.5    1    33  9.859054    FALSE   FALSE
```

```r
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
heck1 = heckit( lfp ~ age + I( age^2 ) + kids + huswage + educ, # the selection process, lfp = 1 if the woman is participating in the labor force 
                 log(wage) ~ educ + exper + I( exper^2 ) + city, data=Mroz87 )
```

Use only variables that affect the selection process in the selection equation. Technically, the selection equation and the equation of interest could have the same set of regressors. But it is not recommended because we should only use variables (or at least one) in the selection equation that affect the selection process, but not the wage process (i.e., instruments). Here, variable `kids` fulfill that role: women with kids may be more likely to stay home, but working moms with kids would not have their wages change.

Alternatively,


```r
# ML estimation of selection model
ml1 = selection( lfp ~ age + I( age^2 ) + kids + huswage + educ,
                    log(wage) ~ educ + exper + I( exper^2 ) + city, data=Mroz87 ) 
```


```r
library("stargazer")
```

```
## 
## Please cite as:
```

```
##  Hlavac, Marek (2018). stargazer: Well-Formatted Regression and Summary Statistics Tables.
```

```
##  R package version 5.2.2. https://CRAN.R-project.org/package=stargazer
```

```r
library("Mediana")
library("plm")
```

```
## 
## Attaching package: 'plm'
```

```
## The following objects are masked from 'package:dplyr':
## 
##     between, lag, lead
```

```r
# function to calculate corrected SEs for regression 
cse = function(reg) {
  rob = sqrt(diag(vcovHC(reg, type = "HC1")))
  return(rob)
}

# stargazer table
stargazer(ols1, heck1, ml1,    
          se=list(cse(ols1),NULL,NULL), 
          title="Married women's wage regressions", type="text", 
          df=FALSE, digits=4, selection.equation = T)
```

```
## 
## Married women's wage regressions
## ==============================================================
##                                Dependent variable:            
##                     ------------------------------------------
##                     log(wage)                lfp              
##                        OLS         Heckman        selection   
##                                   selection                   
##                        (1)           (2)             (3)      
## --------------------------------------------------------------
## age                               0.1861***       0.1842***   
##                                   (0.0652)        (0.0658)    
##                                                               
## I(age2)                          -0.0024***      -0.0024***   
##                                   (0.0008)        (0.0008)    
##                                                               
## kids                             -0.1496***      -0.1488***   
##                                   (0.0383)        (0.0385)    
##                                                               
## huswage                          -0.0430***      -0.0434***   
##                                   (0.0122)        (0.0123)    
##                                                               
## educ                0.1057***     0.1250***       0.1256***   
##                      (0.0130)     (0.0228)        (0.0229)    
##                                                               
## exper               0.0411***                                 
##                      (0.0154)                                 
##                                                               
## I(exper2)            -0.0008*                                 
##                      (0.0004)                                 
##                                                               
## city                  0.0542                                  
##                      (0.0653)                                 
##                                                               
## Constant            -0.5308***   -4.1815***      -4.1484***   
##                      (0.2032)     (1.4024)        (1.4109)    
##                                                               
## --------------------------------------------------------------
## Observations           428           753             753      
## R2                    0.1581       0.1582                     
## Adjusted R2           0.1501       0.1482                     
## Log Likelihood                                    -914.0777   
## rho                                0.0830      0.0505 (0.2317)
## Inverse Mills Ratio            0.0551 (0.2099)                
## Residual Std. Error   0.6667                                  
## F Statistic         19.8561***                                
## ==============================================================
## Note:                              *p<0.1; **p<0.05; ***p<0.01
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
```

```
## --------------------------------------------
## Tobit 2 model (sample selection model)
## Maximum Likelihood estimation
## Newton-Raphson maximisation, 5 iterations
## Return code 1: gradient close to zero (gradtol)
## Log-Likelihood: -712.3163 
## 500 observations (172 censored and 328 observed)
## 6 free parameters (df = 494)
## Probit selection equation:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  -0.2228     0.1081  -2.061   0.0399 *  
## xs            1.3377     0.2014   6.642 8.18e-11 ***
## Outcome equation:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -0.0002265  0.1294178  -0.002    0.999    
## xo           0.7299070  0.1635925   4.462 1.01e-05 ***
##    Error terms:
##       Estimate Std. Error t value Pr(>|t|)    
## sigma   0.9190     0.0574  16.009  < 2e-16 ***
## rho    -0.5392     0.1521  -3.544 0.000431 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## --------------------------------------------
```

without the exclusion restriction, we generate yo using xs instead of xo.


```r
yoX <- xs + eps[,2]
yo <- yoX*(ys > 0)
summary(selection(ys ~ xs, yo ~ xs))
```

```
## --------------------------------------------
## Tobit 2 model (sample selection model)
## Maximum Likelihood estimation
## Newton-Raphson maximisation, 14 iterations
## Return code 8: successive function values within relative tolerance limit (reltol)
## Log-Likelihood: -712.8298 
## 500 observations (172 censored and 328 observed)
## 6 free parameters (df = 494)
## Probit selection equation:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  -0.1984     0.1114  -1.781   0.0756 .  
## xs            1.2907     0.2085   6.191 1.25e-09 ***
## Outcome equation:
##             Estimate Std. Error t value Pr(>|t|)   
## (Intercept)  -0.5499     0.5644  -0.974  0.33038   
## xs            1.3987     0.4482   3.120  0.00191 **
##    Error terms:
##       Estimate Std. Error t value Pr(>|t|)    
## sigma  0.85091    0.05352  15.899   <2e-16 ***
## rho   -0.13226    0.72684  -0.182    0.856    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## --------------------------------------------
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
```

```
## --------------------------------------------
## Tobit 5 model (switching regression model)
## Maximum Likelihood estimation
## Newton-Raphson maximisation, 11 iterations
## Return code 1: gradient close to zero (gradtol)
## Log-Likelihood: -895.8201 
## 500 observations: 172 selection 1 (FALSE) and 328 selection 2 (TRUE)
## 10 free parameters (df = 490)
## Probit selection equation:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  -0.1550     0.1051  -1.474    0.141    
## xs            1.1408     0.1785   6.390 3.86e-10 ***
## Outcome equation 1:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  0.02708    0.16395   0.165    0.869    
## xo1          0.83959    0.14968   5.609  3.4e-08 ***
## Outcome equation 2:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   0.1583     0.1885   0.840    0.401    
## xo2           0.8375     0.1707   4.908 1.26e-06 ***
##    Error terms:
##        Estimate Std. Error t value Pr(>|t|)    
## sigma1  0.93191    0.09211  10.118   <2e-16 ***
## sigma2  0.90697    0.04434  20.455   <2e-16 ***
## rho1    0.88988    0.05353  16.623   <2e-16 ***
## rho2    0.17695    0.33139   0.534    0.594    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## --------------------------------------------
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
```

```
## Warning in sqrt(diag(vc)): NaNs produced

## Warning in sqrt(diag(vc)): NaNs produced
```

```
## Warning in sqrt(diag(vcov(object, part = "full"))): NaNs produced
```

```
## --------------------------------------------
## Tobit 5 model (switching regression model)
## Maximum Likelihood estimation
## Newton-Raphson maximisation, 4 iterations
## Return code 3: Last step could not find a value above the current.
## Boundary of parameter space?  
## Consider switching to a more robust optimisation method temporarily.
## Log-Likelihood: -1665.936 
## 1000 observations: 760 selection 1 (FALSE) and 240 selection 2 (TRUE)
## 10 free parameters (df = 990)
## Probit selection equation:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -0.53698    0.05808  -9.245  < 2e-16 ***
## xs           0.31268    0.09395   3.328 0.000906 ***
## Outcome equation 1:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -0.70679    0.03573  -19.78   <2e-16 ***
## xo1          0.91603    0.05626   16.28   <2e-16 ***
## Outcome equation 2:
##             Estimate Std. Error t value Pr(>|t|)  
## (Intercept)   0.1446         NA      NA       NA  
## xo2           1.1196     0.5014   2.233   0.0258 *
##    Error terms:
##        Estimate Std. Error t value Pr(>|t|)    
## sigma1  0.67770    0.01760   38.50   <2e-16 ***
## sigma2  2.31432    0.07615   30.39   <2e-16 ***
## rho1   -0.97137         NA      NA       NA    
## rho2    0.17039         NA      NA       NA    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## --------------------------------------------
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
```

```
## --------------------------------------------
## Tobit 5 model (switching regression model)
## Maximum Likelihood estimation
## Newton-Raphson maximisation, 16 iterations
## Return code 8: successive function values within relative tolerance limit (reltol)
## Log-Likelihood: -1936.431 
## 1000 observations: 626 selection 1 (FALSE) and 374 selection 2 (TRUE)
## 10 free parameters (df = 990)
## Probit selection equation:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  -0.3528     0.0424  -8.321 2.86e-16 ***
## xs            0.8354     0.0756  11.050  < 2e-16 ***
## Outcome equation 1:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -0.55448    0.06339  -8.748   <2e-16 ***
## xs           0.81764    0.06048  13.519   <2e-16 ***
## Outcome equation 2:
##             Estimate Std. Error t value Pr(>|t|)
## (Intercept)   0.6457     0.4994   1.293    0.196
## xs            0.3520     0.3197   1.101    0.271
##    Error terms:
##        Estimate Std. Error t value Pr(>|t|)    
## sigma1  0.59187    0.01853  31.935   <2e-16 ***
## sigma2  1.97257    0.07228  27.289   <2e-16 ***
## rho1    0.15568    0.15914   0.978    0.328    
## rho2   -0.01541    0.23370  -0.066    0.947    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## --------------------------------------------
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
