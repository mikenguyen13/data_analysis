# Data

There are multiple ways to categorize data. For example,

-   Qualitative vs. Quantitative:

| Qualitative                                                                                                        | Quantitative                                                                               |
|----------------------------------------|--------------------------------|
| in-depth interviews, documents, focus groups, case study, ethnography. open-ended questions. observations in words | experiments, observation in words, survey with closed-end questions, structured interviews |
| language, descriptive                                                                                              | quantities, numbers                                                                        |
| Text-based                                                                                                         | Numbers-based                                                                              |
| Subjective                                                                                                         | Objectivity                                                                                |

## Cross-Sectional

## Time Series

$$
y_t = \beta_0 + x_{t1}\beta_1 + x_{t2}\beta_2 + ... + x_{t(k-1)}\beta_{k-1} + \epsilon_t
$$

Examples

-   Static Model

    -   $y_t=\beta_0 + x_1\beta_1 + x_2\beta_2 - x_3\beta_3 - \epsilon_t$

-   Finite Distributed Lag model

    -   $y_t=\beta_0 + pe_t\delta_0 + pe_{t-1}\delta_1 +pe_{t-2}\delta_2 + \epsilon_t$
    -   **Long Run Propensity (LRP)** is $LRP = \delta_0 + \delta_1 + \delta_2$

-   Dynamic Model

    -   $GDP_t = \beta_0 + \beta_1GDP_{t-1} - \epsilon_t$

[Finite Sample Properties] for [Time Series]:

-   A1-A3: OLS is unbiased
-   A1-A4: usual standard errors are consistent and [Gauss-Markov Theorem] holds (OLS is BLUE)
-   A1-A6, A6: Finite Sample [Wald Test] (t-test and F-test) are valid

[A3][A3 Exogeneity of Independent Variables] might not hold under time series setting

-   Spurious Time Trend - solvable
-   [Strict][A3 Exogeneity of Independent Variables] vs Contemporaneous Exogeneity - not solvable

In time series data, there are many processes:

-   Autoregressive model of order p: AR(p)
-   Moving average model of order q: MA(q)
-   Autoregressive model of order p and moving average model of order q: ARMA(p,q)
-   Autoregressive conditional heteroskedasticity model of order p: ARCH(p)
-   Generalized Autoregressive conditional heteroskedasticity of orders p and q; GARCH(p.q)

### Deterministic Time trend

Both the dependent and independent variables are trending over time

**Spurious Time Series Regression**

$$
y_t = \alpha_0 + t\alpha_1 + v_t
$$

and x takes the form

$$
x_t = \lambda_0 + t\lambda_1 + u_t
$$

-   $\alpha_1 \neq 0$ and $\lambda_1 \neq 0$
-   $v_t$ and $u_t$ are independent
-   there is no relationship between $y_t$ and $x_t$

If we estimate the regression,

$$
y_t = \beta_0 + x_t\beta_1 + \epsilon_t
$$

so the true $\beta_1=0$

-   Inconsistent: $plim(\hat{\beta}_1)=\frac{\alpha_1}{\lambda_1}$
-   Invalid Inference: $|t| \to^d \infty$ for $H_0: \beta_1=0$, will always reject the null as $n \to \infty$
-   Uninformative $R^2$: $plim(R^2) = 1$ will be able to perfectly predict as $n \to \infty$

We can rewrite the equation as

$$
\begin{aligned}
y_t &=\beta_0 + \beta_1x_t+\epsilon_t \\
\epsilon_t &= \alpha_1t + v_t
\end{aligned}
$$

where $\beta_0 = \alpha_0$ and $\beta_1=0$. Since $x_t$ is a deterministic function of time, $\epsilon_t$ is correlated with $x_t$ and we have the usual omitted variable bias.\
Even when $y_t$ and $x_t$ are related ($\beta_1 \neq 0$) but they are both trending over time, we still get spurious results with the simple regression on $y_t$ on $x_t$

**Solutions to Spurious Trend**

1.  Include time trend $t$ as an additional control

    -   consistent parameter estimates and valid inference

2.  Detrend both dependent and independent variables and then regress the detrended outcome on detrended independent variables (i.e., regress residuals $\hat{u}_t$ on residuals $\hat{v}_t$)

    -   Detrending is the same as partialing out in the [Frisch-Waugh-Lovell Theorem]

        -   Could allow for non-linear time trends by including $t$ $t^2$, and $\exp(t)$
        -   Allow for seasonality by including indicators for relevant "seasons" (quarters, months, weeks).

[A3][A3 Exogeneity of Independent Variables] does not hold under:

-   [Feedback Effect]

    -   $\epsilon_t$ influences next period's independent variables

-   [Dynamic Specification]

    -   include last time period outcome as an explanatory variable

-   [Dynamically Complete]

    -   For finite distrusted lag model, the number of lags needs to be absolutely correct.

### Feedback Effect

$$
y_t = \beta_0 + x_t\beta_1 + \epsilon_t
$$

[A3][A3 Exogeneity of Independent Variables]

$$
E(\epsilon_t|\mathbf{X})= E(\epsilon_t| x_1,x_2, ...,x_t,x_{t+1},...,x_T)
$$

will not equal 0, because $y_t$ will likely influence $x_{t+1},..,x_T$

-   [A3][A3 Exogeneity of Independent Variables] is violated because we require the error to be uncorrelated with all time observation of the independent regressors (**strict exogeneity**)

### Dynamic Specification

$$
y_t = \beta_0 + y_{t-1}\beta_1 + \epsilon_t
$$

$$
E(\epsilon_t|\mathbf{X})= E(\epsilon_t| y_1,y_2, ...,y_t,y_{t+1},...,y_T)
$$

will not equal 0, because $y_t$ and $\epsilon_t$ are inherently correlated

-   [A3][A3 Exogeneity of Independent Variables] is violated because we require the error to be uncorrelated with all time observation of the independent regressors (**strict exogeneity**)
-   [Dynamic Specification] is not allowed under [A3][A3 Exogeneity of Independent Variables]

### Dynamically Complete

$$
y_t = \beta_0 + x_t\delta_0 + x_{t-1}\delta_1 + \epsilon_t
$$

$$
E(\epsilon_t|\mathbf{X})= E(\epsilon_t| x_1,x_2, ...,x_t,x_{t+1},...,x_T)
$$

will not equal 0, because if we did not include enough lags, $x_{t-2}$ and $\epsilon_t$ are correlated

-   [A3][A3 Exogeneity of Independent Variables] is violated because we require the error to be uncorrelated with all time observation of the independent regressors (strict exogeneity)
-   Can be corrected by including more lags (but when stop? )

Without [A3][A3 Exogeneity of Independent Variables]

-   OLS is biased
-   [Gauss-Markov Theorem]
-   [Finite Sample Properties] are invalid

then, we can

-   Focus on [Large Sample Properties]
-   Can use [A3a] instead of [A3][A3 Exogeneity of Independent Variables]

[A3a] in time series become

$$
A3a: E(\mathbf{x}_t'\epsilon_t)= 0
$$

only the regressors in this time period need to be independent from the error in this time period (**Contemporaneous Exogeneity**)

-   $\epsilon_t$ can be correlated with $...,x_{t-2},x_{t-1},x_{t+1}, x_{t+2},...$
-   can have a dynamic specification $y_t = \beta_0 + y_{t-1}\beta_1 + \epsilon_t$

Deriving [Large Sample Properties] for Time Series

-   Assumptions [A1][A1 Linearity], [A2][A2 Full rank], [A3a]

-   [Weak Law] and [Central Limit Theorem] depend on [A5][A5 Data Generation (random Sampling)]

    -   $x_t$ and $\epsilon_t$ are dependent over t
    -   without [Weak Law] or [Central Limit Theorem] depend on [A5][A5 Data Generation (random Sampling)], we cannot have [Large Sample Properties] for [OLS][Ordinary Least Squares]
    -   Instead of [A5][A5 Data Generation (random Sampling)], we consider [A5a]

-   Derivation of the Asymptotic Variance depends on [A4][A4 Homoskedasticity]

    -   time series setting introduces **Serial Correlation**: $Cov(\epsilon_t, \epsilon_s) \neq 0$

under [A1][A1 Linearity], [A2][A2 Full rank], [A3a], and [A5a], [OLS estimator][Ordinary Least Squares] is **consistent**, and **asymptotically normal**

### Highly Persistent Data

If $y_t, \mathbf{x}_t$ are not weakly dependent stationary process

-   $y_t$ and $y_{t-h}$ are not almost independent for large h

-   [A5a] does not hold and OLS is not **consistent** and does not have a limiting distribution.

-   Example + Random Walk $y_t = y_{t-1} + u_t$ + Random Walk with a drift: $y_t = \alpha+ y_{t-1} + u_t$

**Solution** First difference is a stationary process

$$
y_t - y_{t-1} = u_t
$$

-   If $u_t$ is a weakly dependent process (also called integrated of order 0) then $y_t$ is said to be difference-stationary process (integrated of order 1)
-   For regression, if $\{y_t, \mathbf{x}_t \}$ are random walks (integrated at order 1), can consistently estimate the first difference equation

$$
\begin{aligned}
y_t - y_{t-1} &= (\mathbf{x}_t - \mathbf{x}_{t-1}\beta + \epsilon_t - \epsilon_{t-1}) \\
\Delta y_t &= \Delta \mathbf{x}\beta + \Delta u_t
\end{aligned}
$$

**Unit Root Test**

$$
y_t = \alpha + \alpha y_{t-1} + u_t
$$

tests if $\rho=1$ (integrated of order 1)\

-   Under the null $H_0: \rho = 1$, OLS is not consistent or asymptotically normal.
-   Under the alternative $H_a: \rho < 1$, OLS is consistent and asymptotically normal.
-   usual t-test is not valid, will need to use the transformed equation to produce a valid test.

**Dickey-Fuller Test** $$
\Delta y_t= \alpha + \theta y_{t-1} + v_t
$$ where $\theta = \rho -1$\

-   $H_0: \theta = 0$ and $H_a: \theta < 0$
-   Under the null, $\Delta y_t$ is weakly dependent but $y_{t-1}$ is not.
-   Dickey and Fuller derived the non-normal asymptotic distribution. If you reject the null then $y_t$ is not a random walk.

Concerns with the standard Dickey Fuller Test\
1. Only considers a fairly simplistic dynamic relationship

$$
\Delta y_t = \alpha + \theta y_{t-1} + \gamma_1 \Delta_{t-1} + ..+ \gamma_p \Delta_{t-p} +v_t
$$

-   with one additional lag, under the null $\Delta_{y_t}$ is an AR(1) process and under the alternative $y_t$ is an AR(2) process.
-   Solution: include lags of $\Delta_{y_t}$ as controls.

2.  Does not allow for time trend $$
    \Delta y_t = \alpha + \theta y_{t-1} + \delta t + v_t
    $$

-   allows $y_t$ to have a quadratic relationship with $t$
-   Solution: include time trend (changes the critical values).

**Adjusted Dickey-Fuller Test** $$
\Delta y_t = \alpha + \theta y_{t-1} + \delta t + \gamma_1 \Delta y_{t-1} + ... + \gamma_p \Delta y_{t-p} + v_t
$$ where $\theta = 1 - \rho$\

-   $H_0: \theta_1 = 0$ and $H_a: \theta_1 < 0$
-   Under the null, $\Delta y_t$ is weakly dependent but $y_{t-1}$ is not
-   Critical values are different with the time trend, if you reject the null then $y_t$ is not a random walk.

##### Newey West Standard Errors

If [A4][A4 Homoskedasticity] does not hold, we can use Newey West Standard Errors (HAC - Heteroskedasticity Autocorrelation Consistent)

$$
\hat{B} = T^{-1} \sum_{t=1}^{T} e_t^2 \mathbf{x'_tx_t} + \sum_{h=1}^{g}(1-\frac{h}{g+1})T^{-1}\sum_{t=h+1}^{T} e_t e_{t-h}(\mathbf{x_t'x_{t-h}+ x_{t-h}'x_t})
$$

-   estimates the covariances up to a distance g part

-   downweights to insure $\hat{B}$ is PSD

-   How to choose g:

    -   For yearly data: $g = 1$ or 2 is likely to account for most of the correlation
    -   For quarterly or monthly data: g should be larger (\$g = 4\$ or 8 for quarterly and $g = 12$ or 14 for monthly)
    -   can also take integer part of $4(T/100)^{2/9}$ or integer part of $T^{1/4}$

**Testing for Serial Correlation**

1.  Run OLS regression of $y_t$ on $\mathbf{x_t}$ and obtain residuals $e_t$

2.  Run OLS regression of $e_t$ on $\mathbf{x}_t, e_{t-1}$ and test whether coefficient on $e_{t-1}$ is significant.

3.  Reject the null of no serial correlation if the coefficient is significant at the 5% level.

    -   Test using heteroskedastic robust standard errors
    -   can include $e_{t-2},e_{t-3},..$ in step 2 to test for higher order serial correlation (t-test would now be an F-test of joint significance)

## Repeated Cross Sections

For each time point (day, month, year, etc.), a set of data is sampled. This set of data can be different among different time points.

For example, you can sample different groups of students each time you survey.

Allowing structural change in pooled cross section

$$
y_i = \mathbf{x}_i \beta + \delta_1 y_1 + ... + \delta_T y_T + \epsilon_i
$$

Dummy variables for all but one time period

-   allows different intercept for each time period
-   allows outcome to change on average for each time period

Allowing for structural change in pooled cross section

$$
y_i = \mathbf{x}_i \beta + \mathbf{x}_i y_1 \gamma_1 + ... + \mathbf{x}_i y_T \gamma_T + \delta_1 y_1 + ...+ \delta_T y_T + \epsilon_i
$$

Interact $x_i$ with time period dummy variables

-   allows different slopes for each time period
-   allows effects to change based on time period (**structural break**)
-   Interacting all time period dummies with $x_i$ can produce many variables - use hypothesis testing to determine which structural breaks are needed.

### Pooled Cross Section

$$
y_i=\mathbf{x_i\beta +x_i \times y1\gamma_1 + ...+ x_i \times yT\gamma_T + \delta_1y_1+...+ \delta_Ty_T + \epsilon_i}
$$

Interact $x_i$ with time period dummy variables

-   allows different slopes for each time period

-   allows effect to change based on time period (structural break)

    -   interacting all time period dummies with $x_i$ can produce many variables - use hypothesis testing to determine which structural breaks are needed.

## Panel Data

Detail notes in R can be found [here](https://cran.r-project.org/web/packages/plm/vignettes/plmPackage.html#robust)

Follows an individual over T time periods.

Panel data structure is like having n samples of time series data

**Characteristics**

-   Information both across individuals and over time (cross-sectional and time-series)

-   N individuals and T time periods

-   Data can be either

    -   Balanced: all individuals are observed in all time periods
    -   Unbalanced: all individuals are not observed in all time periods.

-   Assume correlation (clustering) over time for a given individual, with independence over individuals.

**Types**

-   Short panel: many individuals and few time periods.
-   Long panel: many time periods and few individuals
-   Both: many time periods and many individuals

**Time Trends and Time Effects**

-   Nonlinear
-   Seasonality
-   Discontinuous shocks

**Regressors**

-   Time-invariant regressors $x_{it}=x_i$ for all t (e.g., gender, race, education) have zero within variation
-   Individual-invariant regressors $x_{it}=x_{t}$ for all i (e.g., time trend, economy trends) have zero between variation

**Variation for the dependent variable and regressors**

-   Overall variation: variation over time and individuals.
-   Between variation: variation between individuals
-   Within variation: variation within individuals (over time).

| Estimate         | Formula                                                                                                                     |
|--------------|---------------------------------------------------------|
| Individual mean  | $\bar{x_i}= \frac{1}{T} \sum_{t}x_{it}$                                                                                     |
| Overall mean     | $\bar{x}=\frac{1}{NT} \sum_{i} \sum_t x_{it}$                                                                               |
| Overall Variance | $s _O^2 = \frac{1}{NT-1} \sum_i \sum_t (x_{it} - \bar{x})^2$                                                                |
| Between variance | $s_B^2 = \frac{1}{N-1} \sum_i (\bar{x_i} -\bar{x})^2$                                                                       |
| Within variance  | $s_W^2= \frac{1}{NT-1} \sum_i \sum_t (x_{it} - \bar{x_i})^2 = \frac{1}{NT-1} \sum_i \sum_t (x_{it} - \bar{x_i} +\bar{x})^2$ |

**Note**: $s_O^2 \approx s_B^2 + s_W^2$

Since we have n observation for each time period t, we can control for each time effect separately by including time dummies (time effects)

$$
y_{it}=\mathbf{x_{it}\beta} + d_1\delta_1+...+d_{T-1}\delta_{T-1} + \epsilon_{it}
$$

**Note**: we cannot use these many time dummies in time series data because in time series data, our n is 1. Hence, there is no variation, and sometimes not enough data compared to variables to estimate coefficients.

**Unobserved Effects Model** Similar to group clustering, assume that there is a random effect that captures differences across individuals but is constant in time.

$$
y_it=\mathbf{x_{it}\beta} + d_1\delta_1+...+d_{T-1}\delta_{T-1} + c_i + u_{it}
$$

where

-   $c_i + u_{it} = \epsilon_{it}$
-   $c_i$ unobserved individual heterogeneity (effect)
-   $u_{it}$ idiosyncratic shock
-   $\epsilon_{it}$ unobserved error term.

### Pooled OLS Estimator

If $c_i$ is uncorrelated with $x_{it}$

$$
E(\mathbf{x_{it}'}(c_i+u_{it})) = 0
$$

then [A3a] still holds. And we have Pooled OLS consistent.

If [A4][A4 Homoskedasticity] does not hold, OLS is still consistent, but not efficient, and we need cluster robust SE.

Sufficient for [A3a] to hold, we need

-   **Exogeneity** for $u_{it}$ [A3a] (contemporaneous exogeneity): $E(\mathbf{x_{it}'}u_{it})=0$ time varying error
-   **Random Effect Assumption** (time constant error): $E(\mathbf{x_{it}'}c_{i})=0$

Pooled OLS will give you consistent coefficient estimates under [A1][A1 Linearity], [A2][A2 Full rank], [A3a] (for both $u_{it}$ and RE assumption), and [A5][A5 Data Generation (random Sampling)] (randomly sampling across i).

### Individual-specific effects model

-   If we believe that there is unobserved heterogeneity across individual (e.g., unobserved ability of an individual affects $y$), If the individual-specific effects are correlated with the regressors, then we have the [Fixed Effects Estimator]. and if they are not correlated we have the [Random Effects Estimator](#random-effects-estimator).

#### Random Effects Estimator {#random-effects-estimator}

Random Effects estimator is the Feasible GLS estimator that assumes $u_{it}$ is serially uncorrelated and homoskedastic

-   Under [A1][A1 Linearity], [A2][A2 Full rank], [A3a] (for both $u_{it}$ and RE assumption) and [A5][A5 Data Generation (random Sampling)] (randomly sampling across i), RE estimator is consistent.

    -   If [A4][A4 Homoskedasticity] holds for $u_{it}$, RE is the most efficient estimator
    -   If [A4][A4 Homoskedasticity] fails to hold (may be heteroskedasticity across i, and serial correlation over t), then RE is not the most efficient, but still more efficient than pooled OLS.

#### Fixed Effects Estimator

also known as **Within Estimator** uses within variation (over time)

If the **RE assumption** is not hold ($E(\mathbf{x_{it}'}c_i) \neq 0$), then A3a does not hold ($E(\mathbf{x_{it}'}\epsilon_i) \neq 0$).

Hence, the OLS and RE are inconsistent/biased (because of omitted variable bias)

However, FE can only fix bias due to time-invariant factors (both observables and unobservables) correlated with treatment (not time-variant factors that correlated with the treatment).

The traditional FE technique is flawed when lagged dependent variables are included in the model. [@nickell1981biases] [@narayanan2013estimating]

With measurement error in the independent, FE will exacerbate the errors-in-the-variables bias.

##### Demean Approach

To deal with violation in $c_i$, we have

$$
y_{it}= \mathbf{x_{it} \beta} + c_i + u_{it}
$$

$$
\bar{y_i}=\bar{\mathbf{x_i}} \beta + c_i + \bar{u_i}
$$

where the second equation is the time averaged equation

using **within transformation**, we have

$$
y_{it} - \bar{y_i} = \mathbf{(x_{it} - \bar{x_i})}\beta + u_{it} - \bar{u_i}
$$

because $c_i$ is time constant.

The Fixed Effects estimator uses POLS on the transformed equation

$$
y_{it} - \bar{y_i} = \mathbf{(x_{it} - \bar{x_i})} \beta + d_1\delta_1 + ... + d_{T-2}\delta_{T-2} + u_{it} - \bar{u_i}
$$

-   we need [A3][A3 Exogeneity of Independent Variables] (strict exogeneity) ($E((\mathbf{x_{it}-\bar{x_i}})'(u_{it}-\bar{u_i})=0$) to have FE consistent.

-   Variables that are time constant will be absorbed into $c_i$. Hence we cannot make inference on time constant independent variables.

    -   If you are interested in the effects of time-invariant variables, you could consider the OLS or **between estimator**

-   It's recommended that you should still use cluster robust standard errors.

##### Dummy Approach

Equivalent to the within transformation (i.e., mathematically equivalent to [Demean Approach]), we can have the fixed effect estimator be the same with the dummy regression

$$
y_{it} = x_{it}\beta + d_1\delta_1 + ... + d_{T-2}\delta_{T-2} + c_1\gamma_1 + ... + c_{n-1}\gamma_{n-1} + u_{it}
$$

where

$$
c_i
=
\begin{cases}
1 &\text{if observation is i} \\
0 &\text{otherwise} \\
\end{cases}
$$

-   The standard error is incorrectly calculated.
-   the FE within transformation is controlling for any difference across individual which is allowed to correlated with observables.

##### First-difference Approach

Economists typically use this approach

$$
y_{it} - y_{i (t-1)} = (\mathbf{x}_{it} - \mathbf{x}_{i(t-1)}) \beta +  + (u_{it} - u_{i(t-1)})
$$

##### Fixed Effects Summary

-   The three approaches are **almost** equivalent.

    -   [Demean Approach] is mathematically equivalent to [Dummy Approach]

    -   If you have only 1 period, all 3 are the same.

-   Since fixed effect is a within estimator, only **status changes** can contribute to $\beta$ variation.

    -   Hence, with a small number of changes then the standard error for $\beta$ will explode

-   Status changes mean subjects change from (1) control to treatment group or (2) treatment to control group. Those who have status change, we call them **switchers**.

    -   Treatment effect is typically **non-directional**.

    -   You can give a parameter for the direction if needed.

-   Issues:

    -   You could have fundamental difference between switchers and non-switchers. Even though we can't definitive test this, but providing descriptive statistics on switchers and non-switchers can give us confidence in our conclusion.

    -   Because fixed effects focus on bias reduction, you might have larger variance (typically, with fixed effects you will have less df)

-   If the true model is [random effect](#random-effects-estimator), economists typically don't care, especially when $c_i$ is the random effect and $c_i \perp x_{it}$ (because RE assumption is that it is unrelated to $x_{it}$). The reason why economists don't care is because RE wouldn't correct bias, it only improves efficiency over OLS.

-   You can estimate FE for different units (not just individuals).

-   FE removes bias from time invariant factors but not without costs because it uses within variation, which imposes strict exogeneity assumption on $u_{it}$: $E[(x_{it} - \bar{x}_{i})(u_{it} - \bar{u}_{it})]=0$

Recall

$$
Y_{it} = \beta_0 + X_{it}\beta_1 + \alpha_i + u_{it}
$$

where $\epsilon_{it} = \alpha_i + u_{it}$

$$
\hat{\sigma}^2_\epsilon = \frac{SSR_{OLS}}{NT - K}
$$

$$
\hat{\sigma}^2_u = \frac{SSR_{FE}}{NT - (N+K)} = \frac{SSR_{FE}}{N(T-1)-K}
$$

It's ambiguous whether your variance of error changes up or down because SSR can increase while the denominator decreases.

FE can be unbiased, but not consistent (i.e., not converging to the true effect)

##### FE Examples

##### @blau1999

-   Intergenerational mobility

-   If we transfer resources to low income family, can we generate upward mobility (increase ability)?

Mechanisms for intergenerational mobility

1.  Genetic (policy can't affect) (i.e., ability endowment)
2.  Environmental indirect
3.  Environmental direct

$$
\frac{\% \Delta \text{Human capital}}{\% \Delta \text{income}}
$$

4.  Financial transfer

Income measures:

1.  Total household income
2.  Wage income
3.  Non-wage income
4.  Annual versus permanent income

Core control variables:

**Bad controls are those jointly determined with dependent variable**

Control by mother = choice by mother

Uncontrolled by mothers:

-   mother race

-   location of birth

-   education of parents

-   household structure at age 14

$$
Y_{ijt} = X_{jt} \beta_i + I_{jt} \alpha_i + \epsilon_{ijt}
$$

where

-   $i$ = test

-   $j$ = individual (child)

-   $t$ = time

Grandmother's model

Since child is nested within mother and mother nested within grandmother, the fixed effect of child is included in the fixed effect of mother, which is included in the fixed-effect of grandmother

$$
Y_{ijgmt} = X_{it} \beta_{i} + I_{jt} \alpha_i + \gamma_g + u_{ijgmt}
$$

where

-   $i$ = test, $j$ = kid, $m$ = mother, $g$ = grandmother

-   where $\gamma_g$ includes $\gamma_m$ includes $\gamma_j$

Grandma fixed-effect

Pros:

-   control for some genetics + fixed characteristics of how mother are raised

-   can estimate effect of parameter income

Con:

-   Might not be a sufficient control

Common to cluster a the fixed-effect level (common correlated component)

**Fixed effect exaggerates attenuation bias**

Error rate on survey can help you fix this (plug in the number only , but not the uncertainty associated with that number).

##### @babcock2010

$$
T_{ijct} = \alpha_0 + S_{jct} \alpha_1 + X_{ijct} \alpha_2 + u_{ijct}
$$

where

-   $S_{jct}$ is the average class expectation

-   $X_{ijct}\alpha_2$ is the individual characteristics

-   $i$ student

-   $j$ instructor

-   $c$ course

-   $t$ time

$$
T_{ijct} = \beta_0+ S_{jct} \beta_1+ X_{ijct} \beta_2 +\mu_{jc} + \epsilon_{ijct}
$$

where $\mu_{jc}$ is instructor by course fixed effect (unique id), which is different from $(\theta_j + \delta_c)$

1.  Decrease course shopping because conditioned on available information ($\mu_{jc}$) (class grade and instructor's info).
2.  Grade expectation change even though class materials stay the same

Identification strategy is

-   Under (fixed) time-varying factor that could bias my coefficient (simultaneity)

$$
Y_{ijt} = X_{it} \beta_1 + \text{Teacher Experience}_{jt} \beta_2 + \text{Teacher education}_{jt} \beta_3 + \text{Teacher score}_{it}\beta_4 + \dots + \epsilon_{ijt}
$$

Drop teacher characteristics, and include teacher dummy effect

$$
Y_{ijt} = X_{it} \alpha + \Gamma_{it} \theta_j + u_{ijt}
$$

where $\alpha$ is the within teacher (conditional on teacher fixed effect) and $j = 1 \to (J-1)$

Nuisance in the sense that we don't about the interpretation of $\alpha$

The least we can say about $\theta_j$ is the teacher effect conditional on student test score.

$$
Y_{ijt} = X_{it} \gamma + \epsilon_{ijt}
$$

$\gamma$ is between within (unconditional) and $e_{ijt}$ is the prediction error

$$
e_{ijt} = T_{it} \delta_j + \tilde{e}_{ijt}
$$

where $\delta_j$ is the mean for each group

$$
Y_{ijkt} = Y_{ijkt-1} + X_{it} \beta + T_{it} \tau_j + (W_i + P_k + \epsilon_{ijkt})
$$

where

-   $Y_{ijkt-1}$ = lag control

-   $\tau_j$ = teacher fixed time

-   $W_i$ is the student fixed effect

-   $P_k$ is the school fixed effect

-   $u_{ijkt} = W_i + P_k + \epsilon_{ijkt}$

And we worry about selection on class and school

Bias in $\tau$ (for 1 teacher) is

$$
\frac{1}{N_j} \sum_{i = 1}^N (W_i + P_k + \epsilon_{ijkt})
$$

where $N_j$ = the number of student in class with teacher $j$

then we can $P_k + \frac{1}{N_j} \sum_{i = 1}^{N_j} (W_i + \epsilon_{ijkt})$

Shocks from small class can bias $\tau$

$$
\frac{1}{N_j} \sum_{i = 1}^{N_j} \epsilon_{ijkt} \neq 0
$$

which will inflate the teacher fixed effect

Even if we create random teacher fixed effect and put it in the model, it still contains bias mentioned above which can still $\tau$ (but we do not know the way it will affect - whether more positive or negative).

If teachers switch schools, then we can estimate both teacher and school fixed effect (**mobility web** thin vs. thick)

Mobility web refers to the web of switchers (i.e., from one status to another).

$$
Y_{ijkt} = Y_{ijk(t-1)} \alpha + X_{it}\beta + T_{it} \tau + P_k + \epsilon_{ijkt}
$$

If we demean (fixed-effect), $\tau$ (teacher fixed effect) will go away

If you want to examine teacher fixed effect, we have to include teacher fixed effect

Control for school, the article argues that there is no selection bias

For $\frac{1}{N_j} \sum_{i =1}^{N_j} \epsilon_{ijkt}$ (teacher-level average residuals), $var(\tau)$ does not change with $N_j$ (Figure 2 in the paper). In words, the quality of teachers is not a function of the number of students

If $var(\tau) =0$ it means that teacher quality does not matter

Spin-off of [Measurement Error]: Sampling error or estimation error

$$
\hat{\tau}_j = \tau_j + \lambda_j
$$

$$
var(\hat{\tau}) = var(\tau + \lambda)
$$

Assume $cov(\tau_j, \lambda_j)=0$ (reasonable) In words, your randomness in getting children does not correlation with teacher quality.

Hence,

$$
\begin{aligned}
var(\hat{\tau}) &= var(\tau) + var(\lambda) \\
var(\tau) &= var(\hat{\tau}) - var(\lambda) \\
\end{aligned}
$$

We have $var(\hat{\tau})$ and we need to estimate $var(\lambda)$

$$
var(\lambda) = \frac{1}{J} \sum_{j=1}^J \hat{\sigma}^2_j
$$ where $\hat{\sigma}^2_j$ is the squared standard error of the teacher $j$ (a function of $n$)

Hence,

$$
\frac{var(\tau)}{var(\hat{\tau})} = \text{reliability} = \text{true variance signal}
$$ also known as how much noise in $\hat{\tau}$ and

$$
1 - \frac{var(\tau)}{var(\hat{\tau})} = \text{noise}
$$

Even in cases where the true relationship is that $\tau$ is a function of $N_j$, then our recovery method for $\lambda$ is still not affected

To examine our assumption

$$
\hat{\tau}_j = \beta_0 + X_j \beta_1 + \epsilon_j
$$

Regressing teacher fixed-effect on teacher characteristics should give us $R^2$ close to 0, because teacher characteristics cannot predict sampling error ($\hat{\tau}$ contain sampling error)

### Tests for Assumptions

We typically don't test heteroskedasticity because we will use robust covariance matrix estimation anyway.

Dataset


```r
library("plm")
data("EmplUK", package="plm")
data("Produc", package="plm")
data("Grunfeld", package="plm")
data("Wages", package="plm")
```

#### Poolability

also known as an F test of stability (or Chow test) for the coefficients

$H_0$: All individuals have the same coefficients (i.e., equal coefficients for all individuals).

$H_a$ Different individuals have different coefficients.

Notes:

-   Under a within (i.e., fixed) model, different intercepts for each individual are assumed
-   Under random model, same intercept is assumed


```r
library(plm)
plm::pooltest(inv~value+capital, data=Grunfeld, model="within")
#> 
#> 	F statistic
#> 
#> data:  inv ~ value + capital
#> F = 5.7805, df1 = 18, df2 = 170, p-value = 1.219e-10
#> alternative hypothesis: unstability
```

Hence, we reject the null hypothesis that coefficients are stable. Then, we should use the random model.

#### Individual and time effects

use the Lagrange multiplier test to test the presence of individual or time or both (i.e., individual and time).

Types:

-   `honda`: [@honda1985testing] Default
-   `bp`: [@Breusch_1980] for unbalanced panels
-   `kw`: [@King_1997] unbalanced panels, and two-way effects
-   `ghm`: [@gourieroux1982likelihood]: two-way effects


```r
pFtest(inv~value+capital, data=Grunfeld, effect="twoways")
#> 
#> 	F test for twoways effects
#> 
#> data:  inv ~ value + capital
#> F = 17.403, df1 = 28, df2 = 169, p-value < 2.2e-16
#> alternative hypothesis: significant effects
pFtest(inv~value+capital, data=Grunfeld, effect="individual")
#> 
#> 	F test for individual effects
#> 
#> data:  inv ~ value + capital
#> F = 49.177, df1 = 9, df2 = 188, p-value < 2.2e-16
#> alternative hypothesis: significant effects
pFtest(inv~value+capital, data=Grunfeld, effect="time")
#> 
#> 	F test for time effects
#> 
#> data:  inv ~ value + capital
#> F = 0.23451, df1 = 19, df2 = 178, p-value = 0.9997
#> alternative hypothesis: significant effects
```

#### Cross-sectional dependence/contemporaneous correlation

-   Null hypothesis: residuals across entities are not correlated.

##### Global cross-sectional dependence


```r
pcdtest(inv~value+capital, data=Grunfeld, model="within")
#> 
#> 	Pesaran CD test for cross-sectional dependence in panels
#> 
#> data:  inv ~ value + capital
#> z = 4.6612, p-value = 3.144e-06
#> alternative hypothesis: cross-sectional dependence
```

##### Local cross-sectional dependence

use the same command, but supply matrix `w` to the argument.


```r
pcdtest(inv~value+capital, data=Grunfeld, model="within")
#> 
#> 	Pesaran CD test for cross-sectional dependence in panels
#> 
#> data:  inv ~ value + capital
#> z = 4.6612, p-value = 3.144e-06
#> alternative hypothesis: cross-sectional dependence
```

#### Serial Correlation

-   Null hypothesis: there is no serial correlation

-   usually seen in macro panels with long time series (large N and T), not seen in micro panels (small T and large N)

-   Serial correlation can arise from individual effects(i.e., time-invariant error component), or idiosyncratic error terms (e..g, in the case of AR(1) process). But typically, when we refer to serial correlation, we refer to the second one.

-   Can be

    -   **marginal** test: only 1 of the two above dependence (but can be biased towards rejection)

    -   **joint** test: both dependencies (but don't know which one is causing the problem)

    -   **conditional** test: assume you correctly specify one dependence structure, test whether the other departure is present.

##### Unobserved effect test

-   semi-parametric test (the test statistic $W \dot{\sim} N$ regardless of the distribution of the errors) with $H_0: \sigma^2_\mu = 0$ (i.e., no unobserved effects in the residuals), favors pooled OLS.

    -   Under the null, covariance matrix of the residuals = its diagonal (off-diagonal = 0)

-   It is robust against both **unobserved effects** that are constant within every group, and any kind of **serial correlation**.


```r
pwtest(log(gsp) ~ log(pcap) + log(pc) + log(emp) + unemp, data = Produc)
#> 
#> 	Wooldridge's test for unobserved individual effects
#> 
#> data:  formula
#> z = 3.9383, p-value = 8.207e-05
#> alternative hypothesis: unobserved effect
```

Here, we reject the null hypothesis that the no unobserved effects in the residuals. Hence, we will exclude using pooled OLS.

##### Locally robust tests for random effects and serial correlation

-   A joint LM test for **random effects** and **serial correlation** assuming normality and homoskedasticity of the idiosyncratic errors [@baltagi1991joint][@baltagi1995testing]


```r
pbsytest(log(gsp) ~ log(pcap) + log(pc) + log(emp) + unemp,
         data = Produc,
         test = "j")
#> 
#> 	Baltagi and Li AR-RE joint test
#> 
#> data:  formula
#> chisq = 4187.6, df = 2, p-value < 2.2e-16
#> alternative hypothesis: AR(1) errors or random effects
```

Here, we reject the null hypothesis that there is no presence of **serial correlation,** and **random effects**. But we still do not know whether it is because of serial correlation, of random effects or of both

To know the departure from the null assumption, we can use @bera2001tests's test for first-order serial correlation or random effects (both under normality and homoskedasticity assumption of the error).

BSY for serial correlation


```r
pbsytest(log(gsp) ~ log(pcap) + log(pc) + log(emp) + unemp,
         data = Produc)
#> 
#> 	Bera, Sosa-Escudero and Yoon locally robust test
#> 
#> data:  formula
#> chisq = 52.636, df = 1, p-value = 4.015e-13
#> alternative hypothesis: AR(1) errors sub random effects
```

BSY for random effects


```r
pbsytest(log(gsp)~log(pcap)+log(pc)+log(emp)+unemp, 
         data=Produc, 
         test="re")
#> 
#> 	Bera, Sosa-Escudero and Yoon locally robust test (one-sided)
#> 
#> data:  formula
#> z = 57.914, p-value < 2.2e-16
#> alternative hypothesis: random effects sub AR(1) errors
```

Since BSY is only locally robust, if you "know" there is no serial correlation, then this test is based on LM test is more superior:


```r
plmtest(inv ~ value + capital, data = Grunfeld, 
        type = "honda")
#> 
#> 	Lagrange Multiplier Test - (Honda)
#> 
#> data:  inv ~ value + capital
#> normal = 28.252, p-value < 2.2e-16
#> alternative hypothesis: significant effects
```

On the other hand, if you know there is no random effects, to test for serial correlation, use [@breusch1978testing]-[@godfrey1978testing]'s test


```r
lmtest::bgtest()
```

If you "know" there are random effects, use [@baltagi1995testing]'s. to test for serial correlation in both AR(1) and MA(1) processes.

$H_0$: Uncorrelated errors.

Note:

-   one-sided only has power against positive serial correlation.
-   applicable to only balanced panels.


```r
pbltest(
    log(gsp) ~ log(pcap) + log(pc) + log(emp) + unemp,
    data = Produc,
    alternative = "onesided"
)
#> 
#> 	Baltagi and Li one-sided LM test
#> 
#> data:  log(gsp) ~ log(pcap) + log(pc) + log(emp) + unemp
#> z = 21.69, p-value < 2.2e-16
#> alternative hypothesis: AR(1)/MA(1) errors in RE panel model
```

General serial correlation tests

-   applicable to random effects model, OLS, and FE (with large T, also known as long panel).
-   can also test higher-order serial correlation


```r
plm::pbgtest(plm::plm(inv ~ value + capital,
                      data = Grunfeld,
                      model = "within"),
             order = 2)
#> 
#> 	Breusch-Godfrey/Wooldridge test for serial correlation in panel models
#> 
#> data:  inv ~ value + capital
#> chisq = 42.587, df = 2, p-value = 5.655e-10
#> alternative hypothesis: serial correlation in idiosyncratic errors
```

in the case of short panels (small T and large n), we can use


```r
pwartest(log(emp) ~ log(wage) + log(capital), data=EmplUK)
#> 
#> 	Wooldridge's test for serial correlation in FE panels
#> 
#> data:  plm.model
#> F = 312.3, df1 = 1, df2 = 889, p-value < 2.2e-16
#> alternative hypothesis: serial correlation
```

#### Unit roots/stationarity

-   Dickey-Fuller test for stochastic trends.
-   Null hypothesis: the series is non-stationary (unit root)
-   You would want your test to be less than the critical value (p\<.5) so that there is evidence there is not unit roots.

#### Heteroskedasticity

-   Breusch-Pagan test

-   Null hypothesis: the data is homoskedastic

-   If there is evidence for heteroskedasticity, robust covariance matrix is advised.

-   To control for heteroskedasticity: Robust covariance matrix estimation (Sandwich estimator)

    -   "white1" - for general heteroskedasticity but no serial correlation (check serial correlation first). Recommended for random effects.
    -   "white2" - is "white1" restricted to a common variance within groups. Recommended for random effects.
    -   "arellano" - both heteroskedasticity and serial correlation. Recommended for fixed effects

### Model Selection

#### POLS vs. RE

The continuum between RE (used FGLS which more assumption ) and POLS check back on the section of FGLS

**Breusch-Pagan LM** test

-   Test for the random effect model based on the OLS residual
-   Null hypothesis: variances across entities is zero. In another word, no panel effect.
-   If the test is significant, RE is preferable compared to POLS

#### FE vs. RE

-   RE does not require strict exogeneity for consistency (feedback effect between residual and covariates)

| Hypothesis                             | If true                                                                                |
|-----------------------|-------------------------------------------------|
| $H_0: Cov(c_i,\mathbf{x_{it}})=0$      | $\hat{\beta}_{RE}$ is consistent and efficient, while $\hat{\beta}_{FE}$ is consistent |
| $H_0: Cov(c_i,\mathbf{x_{it}}) \neq 0$ | $\hat{\beta}_{RE}$ is inconsistent, while $\hat{\beta}_{FE}$ is consistent             |

**Hausman Test**

For the Hausman test to run, you need to assume that

-   strict exogeneity hold
-   A4 to hold for $u_{it}$

Then,

-   Hausman test statistic: $H=(\hat{\beta}_{RE}-\hat{\beta}_{FE})'(V(\hat{\beta}_{RE})- V(\hat{\beta}_{FE}))(\hat{\beta}_{RE}-\hat{\beta}_{FE}) \sim \chi_{n(X)}^2$ where $n(X)$ is the number of parameters for the time-varying regressors.
-   A low p-value means that we would reject the null hypothesis and prefer FE
-   A high p-value means that we would not reject the null hypothesis and consider RE estimator.


```r
gw <- plm(inv ~ value + capital, data = Grunfeld, model = "within")
gr <- plm(inv ~ value + capital, data = Grunfeld, model = "random")
phtest(gw, gr)
#> 
#> 	Hausman Test
#> 
#> data:  inv ~ value + capital
#> chisq = 2.3304, df = 2, p-value = 0.3119
#> alternative hypothesis: one model is inconsistent
```

-   Violation Estimator
-   Basic Estimator
-   Instrumental variable Estimator
-   Variable Coefficients estimator
-   Generalized Method of Moments estimator
-   General FGLS estimator
-   Means groups estimator
-   CCEMG
-   Estimator for limited dependent variables

### Summary

-   All three estimators (POLS, RE, FE) require [A1][A1 Linearity], [A2][A2 Full rank], [A5][A5 Data Generation (random Sampling)] (for individuals) to be consistent. Additionally,

-   POLS is consistent under A3a(for $u_{it}$): $E(\mathbf{x}_{it}'u_{it})=0$, and RE Assumption $E(\mathbf{x}_{it}'c_{i})=0$

    -   If [A4][A4 Homoskedasticity] does not hold, use cluster robust SE but POLS is not efficient

-   RE is consistent under A3a(for $u_{it}$): $E(\mathbf{x}_{it}'u_{it})=0$, and RE Assumption $E(\mathbf{x}_{it}'c_{i})=0$

    -   If [A4][A4 Homoskedasticity] (for $u_{it}$) holds then usual SE are valid and RE is most efficient
    -   If [A4][A4 Homoskedasticity] (for $u_{it}$) does not hold, use cluster robust SE ,and RE is no longer most efficient (but still more efficient than POLS)

-   FE is consistent under [A3][A3 Exogeneity of Independent Variables] $E((\mathbf{x}_{it}-\bar{\mathbf{x}}_{it})'(u_{it} -\bar{u}_{it}))=0$

    -   Cannot estimate effects of time constant variables
    -   A4 generally does not hold for $u_{it} -\bar{u}_{it}$ so cluster robust SE are needed

**Note**: [A5][A5 Data Generation (random Sampling)] for individual (not for time dimension) implies that you have [A5a] for the entire data set.

| Estimator / True Model | POLS       | RE         | FE           |
|------------------------|------------|------------|--------------|
| POLS                   | Consistent | Consistent | Inconsistent |
| FE                     | Consistent | Consistent | Consistent   |
| RE                     | Consistent | Consistent | Inconsistent |

Based on table provided by [Ani Katchova](https://sites.google.com/site/econometricsacademy/econometrics-models/panel-data-models)

### Application

#### `plm` package

Recommended application of `plm` can be found [here](https://cran.r-project.org/web/packages/plm/vignettes/B_plmFunction.html) and [here](https://cran.r-project.org/web/packages/plm/vignettes/C_plmModelComponents.html) by Yves Croissant


```r
#install.packages("plm")
library("plm")

library(foreign)
Panel <- read.dta("http://dss.princeton.edu/training/Panel101.dta")

attach(Panel)
Y <- cbind(y)
X <- cbind(x1, x2, x3)

# Set data as panel data
pdata <- pdata.frame(Panel, index = c("country", "year"))

# Pooled OLS estimator
pooling <- plm(Y ~ X, data = pdata, model = "pooling")
summary(pooling)

# Between estimator
between <- plm(Y ~ X, data = pdata, model = "between")
summary(between)

# First differences estimator
firstdiff <- plm(Y ~ X, data = pdata, model = "fd")
summary(firstdiff)

# Fixed effects or within estimator
fixed <- plm(Y ~ X, data = pdata, model = "within")
summary(fixed)

# Random effects estimator
random <- plm(Y ~ X, data = pdata, model = "random")
summary(random)

# LM test for random effects versus OLS
# Accept Null, then OLS, Reject Null then RE
plmtest(pooling, effect = "individual", type = c("bp")) 
# other type: "honda", "kw"," "ghm"; other effect : "time" "twoways"


# B-P/LM and Pesaran CD (cross-sectional dependence) test
# Breusch and Pagan's original LM statistic
pcdtest(fixed, test = c("lm")) 
# Pesaran's CD statistic
pcdtest(fixed, test = c("cd")) 

# Serial Correlation
pbgtest(fixed)

# stationary
library("tseries")
adf.test(pdata$y, k = 2)

# LM test for fixed effects versus OLS
pFtest(fixed, pooling)

# Hausman test for fixed versus random effects model
phtest(random, fixed)

# Breusch-Pagan heteroskedasticity
library(lmtest)
bptest(y ~ x1 + factor(country), data = pdata)

# If there is presence of heteroskedasticity
## For RE model
coeftest(random) #orginal coef

# Heteroskedasticity consistent coefficients
coeftest(random, vcovHC) 

t(sapply(c("HC0", "HC1", "HC2", "HC3", "HC4"), function(x)
    sqrt(diag(
        vcovHC(random, type = x)
    )))) #show HC SE of the coef
# HC0 - heteroskedasticity consistent. The default.
# HC1,HC2, HC3 – Recommended for small samples. 
# HC3 gives less weight to influential observations.
# HC4 - small samples with influential observations
# HAC - heteroskedasticity and autocorrelation consistent

## For FE model
coeftest(fixed) # Original coefficients
coeftest(fixed, vcovHC) # Heteroskedasticity consistent coefficients

# Heteroskedasticity consistent coefficients (Arellano)
coeftest(fixed, vcovHC(fixed, method = "arellano")) 

t(sapply(c("HC0", "HC1", "HC2", "HC3", "HC4"), function(x)
    sqrt(diag(
        vcovHC(fixed, type = x)
    )))) #show HC SE of the coef

```

**Advanced**

Other methods to estimate the random model:

-   `"swar"`: *default* [@swamy1972exact]
-   `"walhus"`: [@wallace1969use]
-   `"amemiya"`: [@amemiya1971estimation]
-   `"nerlove"`" [@nerlove1971further]

Other effects:

-   Individual effects: *default*
-   Time effects: `"time"`
-   Individual and time effects: `"twoways"`

**Note**: no random two-ways effect model for `random.method = "nerlove"`


```r
amemiya <-
    plm(
        Y ~ X,
        data = pdata,
        model = "random",
        random.method = "amemiya",
        effect = "twoways"
    )
```

To call the estimation of the variance of the error components


```r
ercomp(Y ~ X,
       data = pdata,
       method = "amemiya",
       effect = "twoways")
```

Check for the unbalancedness. Closer to 1 indicates balanced data [@ahrens1981two]


```r
punbalancedness(random)
```

**Instrumental variable**

-   `"bvk"`: default [@balestra1987full]
-   `"baltagi"`: [@baltagi1981simultaneous]
-   `"am"` [@amemiya1986instrumental]
-   `"bms"`: [@breusch1989efficient]


```r
instr <-
    plm(
        Y ~ X | X_ins,
        data = pdata,
        random.method = "ht",
        model = "random",
        inst.method = "baltagi"
    )
```

##### Other Estimators

###### Variable Coefficients Model


```r
fixed_pvcm  <- pvcm(Y ~ X, data = pdata, model = "within")
random_pvcm <- pvcm(Y ~ X, data = pdata, model = "random")
```

More details can be found [here](https://cran.r-project.org/web/packages/plm/vignettes/plmPackage.html)

###### Generalized Method of Moments Estimator

Typically use in dynamic models. Example is from [plm package](https://cran.r-project.org/web/packages/plm/vignettes/plmPackage.html)


```r
z2 <- pgmm(
    log(emp) ~ lag(log(emp), 1) + lag(log(wage), 0:1) +
        lag(log(capital), 0:1) | lag(log(emp), 2:99) +
        lag(log(wage), 2:99) + lag(log(capital), 2:99),
    data = EmplUK,
    effect = "twoways",
    model = "onestep",
    transformation = "ld"
)
summary(z2, robust = TRUE)
```

###### General Feasible Generalized Least Squares Models

Assume there is no cross-sectional correlation Robust against intragroup heteroskedasticity and serial correlation. Suited when n is much larger than T (long panel) However, inefficient under group-wise heteorskedasticity.


```r
# Random Effects
zz <-
    pggls(log(emp) ~ log(wage) + log(capital),
          data = EmplUK,
          model = "pooling")

# Fixed
zz <-
    pggls(log(emp) ~ log(wage) + log(capital),
          data = EmplUK,
          model = "within")
```

#### `fixest` package

Available functions

-   `feols`: linear models

-   `feglm`: generalized linear models

-   `femlm`: maximum likelihood estimation

-   `feNmlm`: non-linear in RHS parameters

-   `fepois`: Poisson fixed-effect

-   `fenegbin`: negative binomial fixed-effect

Notes

-   can only work for `fixest` object

Examples by the package's [authors](https://cran.r-project.org/web/packages/fixest/vignettes/exporting_tables.html)


```r
library(fixest)
data(airquality)

# Setting a dictionary
setFixest_dict(
    c(
        Ozone   = "Ozone (ppb)",
        Solar.R = "Solar Radiation (Langleys)",
        Wind    = "Wind Speed (mph)",
        Temp    = "Temperature"
    )
)


# On multiple estimations: see the dedicated vignette
est = feols(
    Ozone ~ Solar.R + sw0(Wind + Temp) | csw(Month, Day),
    data = airquality,
    cluster = ~ Day
)

etable(est)
#>                                         est.1              est.2
#> Dependent Var.:                   Ozone (ppb)        Ozone (ppb)
#>                                                                 
#> Solar Radiation (Langleys) 0.1148*** (0.0234)   0.0522* (0.0202)
#> Wind Speed (mph)                              -3.109*** (0.7986)
#> Temperature                                    1.875*** (0.3671)
#> Fixed-Effects:             ------------------ ------------------
#> Month                                     Yes                Yes
#> Day                                        No                 No
#> __________________________ __________________ __________________
#> S.E.: Clustered                       by: Day            by: Day
#> Observations                              111                111
#> R2                                    0.31974            0.63686
#> Within R2                             0.12245            0.53154
#> 
#>                                        est.3              est.4
#> Dependent Var.:                  Ozone (ppb)        Ozone (ppb)
#>                                                                
#> Solar Radiation (Langleys) 0.1078** (0.0329)   0.0509* (0.0236)
#> Wind Speed (mph)                             -3.289*** (0.7777)
#> Temperature                                   2.052*** (0.2415)
#> Fixed-Effects:             ----------------- ------------------
#> Month                                    Yes                Yes
#> Day                                      Yes                Yes
#> __________________________ _________________ __________________
#> S.E.: Clustered                      by: Day            by: Day
#> Observations                             111                111
#> R2                                   0.58018            0.81604
#> Within R2                            0.12074            0.61471
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# in latex
etable(est, tex = T)
#> \begingroup
#> \centering
#> \begin{tabular}{lcccc}
#>    \tabularnewline \midrule \midrule
#>    Dependent Variable: & \multicolumn{4}{c}{Ozone (ppb)}\\
#>    Model:                     & (1)            & (2)            & (3)            & (4)\\  
#>    \midrule
#>    \emph{Variables}\\
#>    Solar Radiation (Langleys) & 0.1148$^{***}$ & 0.0522$^{**}$  & 0.1078$^{***}$ & 0.0509$^{**}$\\   
#>                               & (0.0234)       & (0.0202)       & (0.0329)       & (0.0236)\\   
#>    Wind Speed (mph)           &                & -3.109$^{***}$ &                & -3.289$^{***}$\\   
#>                               &                & (0.7986)       &                & (0.7777)\\   
#>    Temperature                &                & 1.875$^{***}$  &                & 2.052$^{***}$\\   
#>                               &                & (0.3671)       &                & (0.2415)\\   
#>    \midrule
#>    \emph{Fixed-effects}\\
#>    Month                      & Yes            & Yes            & Yes            & Yes\\  
#>    Day                        &                &                & Yes            & Yes\\  
#>    \midrule
#>    \emph{Fit statistics}\\
#>    Observations               & 111            & 111            & 111            & 111\\  
#>    R$^2$                      & 0.31974        & 0.63686        & 0.58018        & 0.81604\\  
#>    Within R$^2$               & 0.12245        & 0.53154        & 0.12074        & 0.61471\\  
#>    \midrule \midrule
#>    \multicolumn{5}{l}{\emph{Clustered (Day) standard-errors in parentheses}}\\
#>    \multicolumn{5}{l}{\emph{Signif. Codes: ***: 0.01, **: 0.05, *: 0.1}}\\
#> \end{tabular}
#> \par\endgroup


# get the fixed-effects coefficients for 1 model
fixedEffects = fixef(est[[1]])
summary(fixedEffects)
#> Fixed_effects coefficients
#> Number of fixed-effects for variable Month is 5.
#> 	Mean = 19.6	Variance = 272
#> 
#> COEFFICIENTS:
#>   Month:     5     6     7     8     9
#>          3.219 8.288 34.26 40.12 12.13

# see the fixed effects for one dimension
fixedEffects$Month
#>         5         6         7         8         9 
#>  3.218876  8.287899 34.260812 40.122257 12.130971

plot(fixedEffects)
```

<img src="13-data_files/figure-html/unnamed-chunk-24-1.png" width="90%" style="display: block; margin: auto;" />

For [multiple estimation](https://cran.r-project.org/web/packages/fixest/vignettes/multiple_estimations.html)


```r
# set up
library(fixest)

# let R know the base dataset (the biggest/ultimate 
# dataset that includes everything in your analysis)
base = iris

# rename variables
names(base) = c("y1", "y2", "x1", "x2", "species")

res_multi = feols(
    c(y1, y2) ~ x1 + csw(x2, x2 ^ 2) |
        sw0(species),
    data = base,
    fsplit = ~ species,
    lean = TRUE,
    vcov = "hc1" # can also clustered at the fixed effect level
)
# it's recommended to use vcov at 
# estimation stage, not summary stage

summary(res_multi, "compact")
#>         sample   fixef lhs               rhs     (Intercept)                x1
#> 1  Full sample 1        y1 x1 + x2           4.19*** (0.104)  0.542*** (0.076)
#> 2  Full sample 1        y1 x1 + x2 + I(x2^2) 4.27*** (0.101)  0.719*** (0.082)
#> 3  Full sample 1        y2 x1 + x2           3.59*** (0.103) -0.257*** (0.066)
#> 4  Full sample 1        y2 x1 + x2 + I(x2^2) 3.68*** (0.097)    -0.030 (0.078)
#> 5  Full sample species  y1 x1 + x2                            0.906*** (0.076)
#> 6  Full sample species  y1 x1 + x2 + I(x2^2)                  0.900*** (0.077)
#> 7  Full sample species  y2 x1 + x2                              0.155* (0.073)
#> 8  Full sample species  y2 x1 + x2 + I(x2^2)                    0.148. (0.075)
#> 9  setosa      1        y1 x1 + x2           4.25*** (0.474)     0.399 (0.325)
#> 10 setosa      1        y1 x1 + x2 + I(x2^2) 4.00*** (0.504)     0.405 (0.325)
#> 11 setosa      1        y2 x1 + x2           2.89*** (0.416)     0.247 (0.305)
#> 12 setosa      1        y2 x1 + x2 + I(x2^2) 2.82*** (0.423)     0.248 (0.304)
#> 13 setosa      species  y1 x1 + x2                               0.399 (0.325)
#> 14 setosa      species  y1 x1 + x2 + I(x2^2)                     0.405 (0.325)
#> 15 setosa      species  y2 x1 + x2                               0.247 (0.305)
#> 16 setosa      species  y2 x1 + x2 + I(x2^2)                     0.248 (0.304)
#> 17 versicolor  1        y1 x1 + x2           2.38*** (0.423)  0.934*** (0.166)
#> 18 versicolor  1        y1 x1 + x2 + I(x2^2)   0.323 (1.44)   0.901*** (0.164)
#> 19 versicolor  1        y2 x1 + x2           1.25*** (0.275)     0.067 (0.095)
#> 20 versicolor  1        y2 x1 + x2 + I(x2^2)   0.097 (1.01)      0.048 (0.099)
#> 21 versicolor  species  y1 x1 + x2                            0.934*** (0.166)
#> 22 versicolor  species  y1 x1 + x2 + I(x2^2)                  0.901*** (0.164)
#> 23 versicolor  species  y2 x1 + x2                               0.067 (0.095)
#> 24 versicolor  species  y2 x1 + x2 + I(x2^2)                     0.048 (0.099)
#> 25 virginica   1        y1 x1 + x2             1.05. (0.539)  0.995*** (0.090)
#> 26 virginica   1        y1 x1 + x2 + I(x2^2)   -2.39 (2.04)   0.994*** (0.088)
#> 27 virginica   1        y2 x1 + x2             1.06. (0.572)     0.149 (0.107)
#> 28 virginica   1        y2 x1 + x2 + I(x2^2)    1.10 (1.76)      0.149 (0.108)
#> 29 virginica   species  y1 x1 + x2                            0.995*** (0.090)
#> 30 virginica   species  y1 x1 + x2 + I(x2^2)                  0.994*** (0.088)
#> 31 virginica   species  y2 x1 + x2                               0.149 (0.107)
#> 32 virginica   species  y2 x1 + x2 + I(x2^2)                     0.149 (0.108)
#>                  x2          I(x2^2)
#> 1   -0.320. (0.170)                 
#> 2  -1.52*** (0.307) 0.348*** (0.075)
#> 3    0.364* (0.142)                 
#> 4  -1.18*** (0.313) 0.446*** (0.074)
#> 5    -0.006 (0.163)                 
#> 6     0.290 (0.408)   -0.088 (0.117)
#> 7  0.623*** (0.114)                 
#> 8    0.951* (0.472)   -0.097 (0.125)
#> 9    0.712. (0.418)                 
#> 10    2.51. (1.47)     -2.91 (2.10) 
#> 11    0.702 (0.560)                 
#> 12     1.27 (2.39)    -0.911 (3.28) 
#> 13   0.712. (0.418)                 
#> 14    2.51. (1.47)     -2.91 (2.10) 
#> 15    0.702 (0.560)                 
#> 16     1.27 (2.39)    -0.911 (3.28) 
#> 17   -0.320 (0.364)                 
#> 18     3.01 (2.31)     -1.24 (0.841)
#> 19 0.929*** (0.244)                 
#> 20    2.80. (1.65)    -0.695 (0.583)
#> 21   -0.320 (0.364)                 
#> 22     3.01 (2.31)     -1.24 (0.841)
#> 23 0.929*** (0.244)                 
#> 24    2.80. (1.65)    -0.695 (0.583)
#> 25    0.007 (0.205)                 
#> 26    3.50. (2.09)    -0.870 (0.519)
#> 27 0.535*** (0.122)                 
#> 28    0.503 (1.56)     0.008 (0.388)
#> 29    0.007 (0.205)                 
#> 30    3.50. (2.09)    -0.870 (0.519)
#> 31 0.535*** (0.122)                 
#> 32    0.503 (1.56)     0.008 (0.388)

# call the first 3 estimated models only
etable(res_multi[1:3],
       
       # customize the headers
       headers = c("mod1", "mod2", "mod3")) 
#>                   res_multi[1:3].1   res_multi[1:3].2    res_multi[1:3].3
#>                               mod1               mod2                mod3
#> Dependent Var.:                 y1                 y1                  y2
#>                                                                          
#> Constant         4.191*** (0.1037)  4.266*** (0.1007)   3.587*** (0.1031)
#> x1              0.5418*** (0.0761) 0.7189*** (0.0815) -0.2571*** (0.0664)
#> x2               -0.3196. (0.1700) -1.522*** (0.3072)    0.3640* (0.1419)
#> x2 square                          0.3479*** (0.0748)                    
#> _______________ __________________ __________________ ___________________
#> S.E. type       Heteroskedas.-rob. Heteroskedas.-rob. Heteroskedast.-rob.
#> Observations                   150                150                 150
#> R2                         0.76626            0.79456             0.21310
#> Adj. R2                    0.76308            0.79034             0.20240
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

##### Multiple estimation (Left-hand side)

-   When you have multiple interested dependent variables


```r
etable(feols(c(y1, y2) ~ x1 + x2, base))
#>                 feols(c(y1, y2)..1 feols(c(y1, y2) ..2
#> Dependent Var.:                 y1                  y2
#>                                                       
#> Constant         4.191*** (0.0970)   3.587*** (0.0937)
#> x1              0.5418*** (0.0693) -0.2571*** (0.0669)
#> x2               -0.3196* (0.1605)    0.3640* (0.1550)
#> _______________ __________________ ___________________
#> S.E. type                      IID                 IID
#> Observations                   150                 150
#> R2                         0.76626             0.21310
#> Adj. R2                    0.76308             0.20240
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

To input a list of dependent variable


```r
depvars <- c("y1", "y2")

res <- lapply(depvars, function(var) {
    res <- feols(xpd(..lhs ~ x1 + x2, ..lhs = var), data = base)
    # summary(res)
})
etable(res)
#>                            model 1             model 2
#> Dependent Var.:                 y1                  y2
#>                                                       
#> Constant         4.191*** (0.0970)   3.587*** (0.0937)
#> x1              0.5418*** (0.0693) -0.2571*** (0.0669)
#> x2               -0.3196* (0.1605)    0.3640* (0.1550)
#> _______________ __________________ ___________________
#> S.E. type                      IID                 IID
#> Observations                   150                 150
#> R2                         0.76626             0.21310
#> Adj. R2                    0.76308             0.20240
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

##### Multiple estimation (Right-hand side)

Options to write the functions

-   `sw` (stepwise): sequentially analyze each elements

    -   `y ~ sw(x1, x2)` will be estimated as `y ~ x1` and `y ~ x2`

-   `sw0` (stepwise 0): similar to `sw` but also estimate a model without the elements in the set first

    -   `y ~ sw(x1, x2)` will be estimated as `y ~ 1` and `y ~ x1` and `y ~ x2`

-   `csw` (cumulative stepwise): sequentially add each element of the set to the formula

    -   `y ~ csw(x1, x2)` will be estimated as `y ~ x1` and `y ~ x1 + x2`

-   `csw0` (cumulative stepwise 0): similar to `csw` but also estimate a model without the elements in the set first

    -   `y ~ csw(x1, x2)` will be estimated as `y~ 1` `y ~ x1` and `y ~ x1 + x2`

-   `mvsw` (multiverse stepwise): all possible combination of the elements in the set (it will get large very quick).

    -   `mvsw(x1, x2, x3)` will be `sw0(x1, x2, x3, x1 + x2, x1 + x3, x2 + x3, x1 + x2 + x3)`

##### Split sample estimation


```r
etable(feols(y1 ~ x1 + x2, fsplit = ~ species, data = base))
#>                  feols(y1 ~ x1 +..1 feols(y1 ~ x1 ..2 feols(y1 ~ x1 +..3
#> Sample (species)        Full sample            setosa         versicolor
#> Dependent Var.:                  y1                y1                 y1
#>                                                                         
#> Constant          4.191*** (0.0970) 4.248*** (0.4114)  2.381*** (0.4493)
#> x1               0.5418*** (0.0693)   0.3990 (0.2958) 0.9342*** (0.1693)
#> x2                -0.3196* (0.1605)   0.7121 (0.4874)   -0.3200 (0.4024)
#> ________________ __________________ _________________ __________________
#> S.E. type                       IID               IID                IID
#> Observations                    150                50                 50
#> R2                          0.76626           0.11173            0.57432
#> Adj. R2                     0.76308           0.07393            0.55620
#> 
#>                  feols(y1 ~ x1 +..4
#> Sample (species)          virginica
#> Dependent Var.:                  y1
#>                                    
#> Constant            1.052* (0.5139)
#> x1               0.9946*** (0.0893)
#> x2                  0.0071 (0.1795)
#> ________________ __________________
#> S.E. type                       IID
#> Observations                     50
#> R2                          0.74689
#> Adj. R2                     0.73612
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

##### Standard Errors

-   `iid`: errors are homoskedastic and independent and identically distributed

-   `hetero`: errors are heteroskedastic using White correction

-   `cluster`: errors are correlated within the cluster groups

-   `newey_west`: [@newey1986simple] use for time series or panel data. Errors are heteroskedastic and serially correlated.

    -   `vcov = newey_west ~ id + period` where `id` is the subject id and `period` is time period of the panel.

    -   to specify lag period to consider `vcov = newey_west(2) ~ id + period` where we're considering 2 lag periods.

-   `driscoll_kraay` [@driscoll1998consistent] use for panel data. Errors are cross-sectionally and serially correlated.

    -   `vcov = discoll_kraay ~ period`

-   `conley`: [@conley1999gmm] for cross-section data. Errors are spatially correlated

    -   `vcov = conley ~ latitude + longitude`

    -   to specify the distance cutoff, `vcov = vcov_conley(lat = "lat", lon = "long", cutoff = 100, distance = "spherical")`, which will use the `conley()` helper function.

-   `hc`: from the `sandwich` package

    -   `vcov = function(x) sandwich::vcovHC(x, type = "HC1"))`

To let R know which SE estimation you want to use, insert `vcov = vcov_type ~ variables`

##### Small sample correction

To specify that R needs to use small sample correction add

`ssc = ssc(adj = T, cluster.adj = T)`
