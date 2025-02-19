# Model Specification Tests

Model specification tests are critical in econometric analysis to verify whether the assumptions underlying a model hold true. These tests help determine if the model is correctly specified, ensuring that the estimators are both reliable and efficient. A mis-specified model can lead to biased, inconsistent, or inefficient estimates, which undermines the validity of inferences drawn from the analysis.

This chapter addresses various model specification tests, including tests for:

-   [Nested Model Tests]
-   [Non-Nested Model Tests]
-   [Heteroskedasticity Tests]
-   [Functional Form Tests]
-   [Autocorrelation Tests]
-   [Multicollinearity Diagnostics]

Understanding these tests allows researchers to evaluate the robustness of their models and make necessary adjustments to improve model performance.

------------------------------------------------------------------------

## Nested Model Tests

Nested models are those where the **restricted model** is a special case of the **unrestricted model**. In other words, the restricted model can be derived from the unrestricted model by imposing constraints on certain parameters, typically setting them equal to zero. This structure allows us to formally test whether the additional variables in the unrestricted model significantly improve the model's explanatory power. The following tests help compare these models:

-   [Wald Test](#sec-wald-test-nested): Assesses the significance of individual coefficients or groups of coefficients.
-   [Likelihood Ratio Test](#sec-likelihood-ratio-test-nested): Compares the goodness-of-fit between restricted and unrestricted models.
-   [F-Test](#sec-f-test-for-linear-regression-nested): Evaluates the joint significance of multiple coefficients.
-   [Chow Test](#sec-chow-test): Evaluates whether the coefficients of a regression model are the same across different groups or time periods.

Consider the following models:

$$
\begin{aligned}
\textbf{Unrestricted Model:} \quad & y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \epsilon \\
\textbf{Restricted Model:} \quad & y = \beta_0 + \beta_1 x_1 + \epsilon
\end{aligned}
$$

-   The **unrestricted model** includes all potential explanatory variables: $x_1$, $x_2$, and $x_3$.
-   The **restricted model** is nested within the unrestricted model, containing a subset of variables (in this case, excluding $x_2$ and $x_3$).

Our goal is to test the null hypothesis that the restrictions are valid:

$$
H_0: \beta_2 = \beta_3 = 0 \quad \text{(restrictions are valid)}
$$

against the alternative hypothesis:

$$
H_1: \text{At least one of } \beta_2, \beta_3 \neq 0 \quad \text{(restrictions are invalid)}
$$

To conduct this test, we use the following methods:

------------------------------------------------------------------------

### Wald Test {#sec-wald-test-nested}

The **Wald Test** assesses whether certain linear restrictions on the parameters of the model are valid. It is commonly used when testing the joint significance of multiple coefficients.

Consider a set of linear restrictions expressed as:

$$
H_0: R\boldsymbol{\beta} = r
$$

where:

-   $R$ is a $q \times k$ restriction matrix,

-   $\boldsymbol{\beta}$ is the $k \times 1$ vector of parameters,

-   $r$ is a $q \times 1$ vector representing the hypothesized values (often zeros),

-   $q$ is the number of restrictions being tested.

For example, if we want to test $H_0: \beta_2 = \beta_3 = 0$, then:

$$
R = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad r = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

The Wald statistic is calculated as:

$$
W = (R\hat{\boldsymbol{\beta}} - r)' \left[ R \, \hat{\text{Var}}(\hat{\boldsymbol{\beta}}) \, R' \right]^{-1} (R\hat{\boldsymbol{\beta}} - r)
$$

Where:

-   $\hat{\boldsymbol{\beta}}$ is the vector of estimated coefficients from the unrestricted model,

-   $\hat{\text{Var}}(\hat{\boldsymbol{\beta}})$ is the estimated covariance matrix of $\hat{\boldsymbol{\beta}}$.

Distribution and Decision Rule

-   Under $H_0$, the Wald statistic follows a $\chi^2$ distribution with $q$ degrees of freedom:

$$
W \sim \chi^2_q
$$

-   **Decision Rule:**
    -   Reject $H_0$ if $W > \chi^2_{q,\alpha}$, where $\alpha$ is the significance level.
    -   A large Wald statistic indicates that the restrictions are invalid.

------------------------------------------------------------------------

### Likelihood Ratio Test {#sec-likelihood-ratio-test-nested}

The **Likelihood Ratio Test** compares the goodness-of-fit between the restricted and unrestricted models. It evaluates whether the additional parameters in the unrestricted model significantly improve the likelihood of observing the data.

Same as before:

$$
H_0: \beta_2 = \beta_3 = 0 \quad \text{vs.} \quad H_1: \text{At least one of } \beta_2, \beta_3 \neq 0
$$

The LR statistic is calculated as:

$$
LR = -2 \left( \ln L_R - \ln L_U \right)
$$

Where:

-   $L_R$ is the maximized likelihood of the restricted model,

-   $L_U$ is the maximized likelihood of the unrestricted model.

Distribution and Decision Rule

-   Under $H_0$, the LR statistic follows a $\chi^2$ distribution with $q$ degrees of freedom:

$$
LR \sim \chi^2_q
$$

-   **Decision Rule:**
    -   Reject $H_0$ if $LR > \chi^2_{q,\alpha}$.
    -   A large LR statistic suggests that the unrestricted model provides a significantly better fit.

**Connection to [OLS](#ordinary-least-squares)**

In the case of linear regression with normally distributed errors, the LR statistic can be expressed in terms of the sum of squared residuals (SSR):

$$
LR = n \ln \left( \frac{SSR_R}{SSR_U} \right)
$$

where $n$ is the sample size.

------------------------------------------------------------------------

### F-Test (for Linear Regression) {#sec-f-test-for-linear-regression-nested}

The **F-Test** is commonly used in linear regression to evaluate the joint significance of multiple coefficients. It compares the fit of the restricted and unrestricted models using their sum of squared residuals.

Again:

$$
H_0: \beta_2 = \beta_3 = 0 \quad \text{vs.} \quad H_1: \text{At least one of } \beta_2, \beta_3 \neq 0
$$

The F-statistic is calculated as:

$$
F = \frac{(SSR_R - SSR_U) / q}{SSR_U / (n - k)}
$$

Where:

-   $SSR_R$ = Sum of Squared Residuals from the restricted model,
-   $SSR_U$ = Sum of Squared Residuals from the unrestricted model,
-   $q$ = Number of restrictions (here, 2),
-   $n$ = Sample size,
-   $k$ = Number of parameters in the unrestricted model.

Distribution and Decision Rule

-   Under $H_0$, the F-statistic follows an $F$-distribution with $(q, n - k)$ degrees of freedom:

$$
F \sim F_{q, n - k}
$$

-   **Decision Rule:**
    -   Reject $H_0$ if $F > F_{q, n - k, \alpha}$.
    -   A large F-statistic indicates that the restricted model fits significantly worse, suggesting the excluded variables are important.

------------------------------------------------------------------------

### Chow Test {#sec-chow-test}

The **Chow Test** evaluates whether the coefficients of a regression model are the same across different groups or time periods. It is often used to detect **structural breaks** in the data.

**Key Question:**\
*Should we run two different regressions for two groups, or can we pool the data and use a single regression?*

**Chow Test Procedure**

1.  Estimate the regression model for the **pooled data** (all observations).
2.  Estimate the model separately for **Group 1** and **Group 2**.
3.  Compare the **sum of squared residuals (SSR)** from these models.

The test statistic follows an F-distribution:

$$
F = \frac{(SSR_P - (SSR_1 + SSR_2)) / k}{(SSR_1 + SSR_2) / (n_1 + n_2 - 2k)}
$$

Where:

-   $SSR_P$ = Sum of Squared Residuals for the pooled model
-   $SSR_1$, $SSR_2$ = SSRs for Group 1 and Group 2 models
-   $k$ = Number of parameters
-   $n_1$, $n_2$ = Number of observations in each group

**Interpretation:**

-   A **significant F-statistic** suggests structural differences between groups, implying separate regressions are more appropriate.

-   A **non-significant F-statistic** indicates no structural break, supporting the use of a pooled model.


```r
# Load necessary libraries
library(car)        # For Wald Test
library(lmtest)     # For Likelihood Ratio Test
library(strucchange)  # For Chow Test

# Simulated dataset
set.seed(123)
n <- 100
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
epsilon <- rnorm(n)
y <- 2 + 1.5 * x1 + 0.5 * x2 - 0.7 * x3 + epsilon

# Creating a group variable (simulating a structural break)
group <- rep(c(0, 1), each = n / 2)  # Group 0 and Group 1

# ----------------------------------------------------------------------
# Wald Test
# ----------------------------------------------------------------------
unrestricted_model <- lm(y ~ x1 + x2 + x3)    # Unrestricted model
restricted_model <- lm(y ~ x1)                # Restricted model

wald_test <- linearHypothesis(unrestricted_model, c("x2 = 0", "x3 = 0"))
print(wald_test)
#> Linear hypothesis test
#> 
#> Hypothesis:
#> x2 = 0
#> x3 = 0
#> 
#> Model 1: restricted model
#> Model 2: y ~ x1 + x2 + x3
#> 
#>   Res.Df    RSS Df Sum of Sq      F    Pr(>F)    
#> 1     98 182.26                                  
#> 2     96 106.14  2    76.117 34.421 5.368e-12 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# ----------------------------------------------------------------------
# Likelihood Ratio Test
# ----------------------------------------------------------------------
lr_test <- lrtest(unrestricted_model, restricted_model)
print(lr_test)
#> Likelihood ratio test
#> 
#> Model 1: y ~ x1 + x2 + x3
#> Model 2: y ~ x1
#>   #Df  LogLik Df  Chisq Pr(>Chisq)    
#> 1   5 -144.88                         
#> 2   3 -171.91 -2 54.064  1.821e-12 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# ----------------------------------------------------------------------
# F-Test (for Linear Regression)
# ----------------------------------------------------------------------
SSR_U <- sum(residuals(unrestricted_model)^2)  # SSR for unrestricted model
SSR_R <- sum(residuals(restricted_model)^2)    # SSR for restricted model
q <- 2                                        # Number of restrictions
n <- length(y)                                # Sample size
k <- length(coef(unrestricted_model))         # Number of parameters in unrestricted model

# F-statistic formula
F_statistic <- ((SSR_R - SSR_U) / q) / (SSR_U / (n - k))
p_value_F <- pf(F_statistic, df1 = q, df2 = n - k, lower.tail = FALSE)

cat("F-statistic:", F_statistic, "\n")
#> F-statistic: 34.42083
cat("P-value:", p_value_F, "\n")
#> P-value: 5.367912e-12

# ----------------------------------------------------------------------
# Chow Test (Proper Use of the Group Variable)
# ----------------------------------------------------------------------
# Pooled model (all data)
pooled_model <- lm(y ~ x1 + x2 + x3)

# Separate models for Group 0 and Group 1
model_group0 <- lm(y[group == 0] ~ x1[group == 0] + x2[group == 0] + x3[group == 0])
model_group1 <- lm(y[group == 1] ~ x1[group == 1] + x2[group == 1] + x3[group == 1])

# Calculating SSRs
SSR_pooled <- sum(residuals(pooled_model)^2)
SSR_group0 <- sum(residuals(model_group0)^2)
SSR_group1 <- sum(residuals(model_group1)^2)

# Chow Test formula
k_chow <- length(coef(pooled_model))  # Number of parameters (including intercept)
n0 <- sum(group == 0)                 # Sample size for Group 0
n1 <- sum(group == 1)                 # Sample size for Group 1

F_chow <- ((SSR_pooled - (SSR_group0 + SSR_group1)) / k_chow) /
          ((SSR_group0 + SSR_group1) / (n0 + n1 - 2 * k_chow))

# Corresponding p-value
p_value_chow <- pf(F_chow, df1 = k_chow, df2 = (n0 + n1 - 2 * k_chow), lower.tail = FALSE)

cat("Chow Test F-statistic:", F_chow, "\n")
#> Chow Test F-statistic: 0.3551197
cat("P-value:", p_value_chow, "\n")
#> P-value: 0.8398657
```

**Interpretation of the Results**

1.  **Wald Test**

    -   **Null Hypothesis (**$H_0$): $\beta_2 = \beta_3 = 0$ (the coefficients for $x_2$ and $x_3$ are zero).

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* \< 0.05: $x_2$ and $x_3$ are jointly significant.
        -   **Fail to reject** $H_0$ if *p-value* ≥ 0.05: $x_2$ and $x_3$ do not significantly contribute to the model.

2.  **Likelihood Ratio Test (LR Test)**

    -   **Null Hypothesis (**$H_0$): The restricted model fits the data as well as the unrestricted model.

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* \< 0.05: The unrestricted model fits better, indicating $x_2$ and $x_3$ improve the model.
        -   **Fail to reject** $H_0$ if *p-value* ≥ 0.05: Adding $x_2$ and $x_3$ doesn't improve the model significantly.

3.  **F-Test (for Linear Regression)**

    -   **Null Hypothesis (**$H_0$): $\beta_2 = \beta_3 = 0$ (similar to the Wald Test).

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* \< 0.05: The excluded variables are significant.
        -   **Fail to reject** $H_0$ if *p-value* ≥ 0.05: The excluded variables are not significant.

4.  **Chow Test (Using the `group` Variable)**

    -   **Null Hypothesis (**$H_0$): No structural break exists; the regression coefficients are the same across **Group 0** and **Group 1**.

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* \< 0.05: A structural break exists---model coefficients differ significantly between the groups.
        -   **Fail to reject** $H_0$ if *p-value* ≥ 0.05: No significant structural break detected; the model coefficients are stable across both groups.

------------------------------------------------------------------------

## Non-Nested Model Tests

Comparing models is essential to identify which specification best explains the data. While nested model comparisons rely on one model being a restricted version of another, **non-nested models** do not share such a hierarchical structure. This situation commonly arises when comparing models with:

-   Different functional forms (e.g., linear vs. logarithmic relationships),
-   Different sets of explanatory variables,
-   Competing theoretical frameworks.

To compare non-nested models, we rely on specialized statistical tests designed to handle these complexities. The two most widely used approaches are:

-   [Vuong Test](#sec-vuong-test) used to compare the fit of two non-nested models to determine which model better explains the data.

-   [Davidson--MacKinnon J-Test](#sec-davidson–mackinnon-j-test) is a regression-based approach for comparing non-nested models. It evaluates which model better fits the data by incorporating the predicted values from the competing model as an additional regressor.

------------------------------------------------------------------------

Consider two competing models:

-   **Model A:**\
    $$
    y = \alpha_0 + \alpha_1 f(X) + \epsilon_A
    $$
-   **Model B:**\
    $$
    y = \beta_0 + \beta_1 g(Z) + \epsilon_B
    $$

Where:

-   $f(X)$ and $g(Z)$ represent different functional forms or different sets of explanatory variables.

-   The models are **non-nested** because one cannot be obtained from the other by restricting parameters.

Our goal is to determine which model better explains the data.

------------------------------------------------------------------------

### Vuong Test {#sec-vuong-test}

The **Vuong Test** is a likelihood-ratio-based approach for comparing non-nested models [@vuong1989likelihood]. It is particularly useful when both models are estimated via [Maximum Likelihood] Estimation.

Hypotheses

-   Null Hypothesis ($H_0$): Both models are equally close to the true data-generating process (i.e., the models have equal predictive power).
-   Alternative Hypothesis ($H_1$):
    -   Positive Test Statistic ($V > 0$): Model A is preferred.
    -   Negative Test Statistic ($V < 0$): Model B is preferred.

------------------------------------------------------------------------

**Vuong Test Statistic**

The Vuong test is based on the difference in the log-likelihood contributions of each observation under the two models. Let:

-   $\ell_{A,i}$ = log-likelihood of observation $i$ under Model A,
-   $\ell_{B,i}$ = log-likelihood of observation $i$ under Model B.

Define the difference in log-likelihoods:

$$
m_i = \ell_{A,i} - \ell_{B,i}
$$

The Vuong test statistic is:

$$
V = \frac{\sqrt{n} \, \bar{m}}{s_m}
$$

Where:

-   $\bar{m} = \frac{1}{n} \sum_{i=1}^n m_i$ is the sample mean of the log-likelihood differences,

-   $s_m = \sqrt{\frac{1}{n} \sum_{i=1}^n (m_i - \bar{m})^2}$ is the sample standard deviation of $m_i$,

-   $n$ is the sample size.

------------------------------------------------------------------------

**Distribution and Decision Rule**

-   Under $H_0$, the Vuong statistic asymptotically follows a **standard normal distribution**:

$$
V \sim N(0, 1)
$$

-   **Decision Rule:**
    -   If $|V| > z_{\alpha/2}$ (critical value from the standard normal distribution), **reject** $H_0$.
        -   If $V > 0$: Prefer **Model A**.
        -   If $V < 0$: Prefer **Model B**.
    -   If $|V| \leq z_{\alpha/2}$: **Fail to reject** $H_0$; no significant difference between models.

------------------------------------------------------------------------

**Corrections for Model Complexity**

When comparing models with different numbers of parameters, a **penalized version** of the Vuong test can be used, similar to adjusting for model complexity in criteria like AIC or BIC. The corrected statistic is:

$$
V_{\text{adjusted}} = V - \frac{(k_A - k_B) \ln(n)}{2 s_m \sqrt{n}}
$$

Where $k_A$ and $k_B$ are the number of parameters in Models A and B, respectively.

------------------------------------------------------------------------

**Limitations of the Vuong Test**

-   Requires models to be estimated via [Maximum Likelihood].
-   Sensitive to **model misspecification** and **heteroskedasticity**.
-   Assumes **independent and identically distributed (i.i.d.)** errors.

------------------------------------------------------------------------

### Davidson--MacKinnon J-Test {#sec-davidson--mackinnon-j-test}

The **Davidson--MacKinnon J-Test** provides a flexible, regression-based approach for comparing non-nested models [@davidson1981several]. It evaluates whether the predictions from one model contain information not captured by the competing model. This test can be thought of as comparing models with transformed independent variables, as opposed to the next section, [Comparing Models with Transformed Dependent Variables].

**Hypotheses**

-   Null Hypothesis ($H_0$): The competing model does not provide additional explanatory power beyond the current model.
-   Alternative Hypothesis ($H_1$): The competing model provides additional explanatory power.

------------------------------------------------------------------------

**Procedure**

Consider two competing models:

-   **Model A:**\
    $$
    y = \alpha_0 + \alpha_1 x + \epsilon_A
    $$
-   **Model B:**\
    $$
    y = \beta_0 + \beta_1 \ln(x) + \epsilon_B
    $$

**Step 1: Testing Model A Against Model B**

1.  **Estimate Model B** and obtain predicted values $\hat{y}_B$.
2.  **Run the auxiliary regression:**

$$
y = \alpha_0 + \alpha_1 x + \gamma \hat{y}_B + u
$$

3.  **Test the null hypothesis:**

$$
H_0: \gamma = 0
$$

-   If $\gamma$ is significant, Model B adds explanatory power beyond Model A.
-   If $\gamma$ is not significant, Model A sufficiently explains the data.

------------------------------------------------------------------------

**Step 2: Testing Model B Against Model A**

1.  **Estimate Model A** and obtain predicted values $\hat{y}_A$.
2.  **Run the auxiliary regression:**

$$
y = \beta_0 + \beta_1 \ln(x) + \gamma \hat{y}_A + u
$$

3.  **Test the null hypothesis:**

$$
H_0: \gamma = 0
$$

------------------------------------------------------------------------

**Decision Rules**

-   **Reject** $H_0$ in Step 1, Fail to Reject in Step 2: Prefer **Model B**.
-   **Fail to Reject** $H_0$ in Step 1, Reject in Step 2: Prefer **Model A**.
-   **Reject** $H_0$ in Both Steps: Neither model is adequate; reconsider the functional form.
-   **Fail to Reject** $H_0$ in Both Steps: No strong evidence to prefer one model; rely on other criteria (e.g., theory, simplicity). Alternatively, $R^2_{adj}$ can also be used to choose between the two.

**Adjusted** $R^2$

-   $R^2$ will always increase with more variables included
-   Adjusted $R^2$ tries to correct by penalizing inclusion of unnecessary variables.

$$ \begin{aligned} {R}^2 &= 1 - \frac{SSR/n}{SST/n} \\ {R}^2_{adj} &= 1 - \frac{SSR/(n-k)}{SST/(n-1)} \\ &= 1 - \frac{(n-1)(1-R^2)}{(n-k)} \end{aligned} $$

-   ${R}^2_{adj}$ increases if and only if the t-statistic on the additional variable is greater than 1 in absolute value.
-   ${R}^2_{adj}$ is valid in models where there is no heteroskedasticity
-   there fore it **should not** be used in determining which variables should be included in the model (the t or F-tests are more appropriate)

### Adjusted $R^2$

The **coefficient of determination** ($R^2$) measures the proportion of the variance in the dependent variable that is explained by the model. However, a key limitation of $R^2$ is that it **always increases** (or at least stays the same) when additional explanatory variables are added to the model, even if those variables are not statistically significant.

To address this issue, the **adjusted** $R^2$ introduces a penalty for including unnecessary variables, making it a more reliable measure when comparing models with different numbers of predictors.

------------------------------------------------------------------------

**Formulas**

**Unadjusted** $R^2$:

$$
R^2 = 1 - \frac{SSR}{SST}
$$

Where:

-   $SSR$ = **Sum of Squared Residuals** (measures unexplained variance),

-   $SST$ = **Total Sum of Squares** (measures total variance in the dependent variable).

**Adjusted** $R^2$:

$$
R^2_{\text{adj}} = 1 - \frac{SSR / (n - k)}{SST / (n - 1)}
$$

Alternatively, it can be expressed in terms of $R^2$ as:

$$
R^2_{\text{adj}} = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - k} \right)
$$

Where:

-   $n$ = Number of observations,

-   $k$ = Number of estimated parameters in the model (including the intercept).

------------------------------------------------------------------------

**Key Insights**

-   **Penalty for Complexity:** Unlike $R^2$, the adjusted $R^2$ **can decrease** when irrelevant variables are added to the model because it adjusts for the number of predictors relative to the sample size.
-   **Interpretation:** It represents the proportion of variance explained **after accounting for model complexity**.
-   **Comparison Across Models:** Adjusted $R^2$ is useful for comparing models with different numbers of predictors, as it discourages overfitting.

------------------------------------------------------------------------

**When Does Adjusted** $R^2$ Increase?

-   The adjusted $R^2$ will **increase** if and only if the inclusion of a new variable **improves the model more than expected by chance**.
-   **Mathematically:** This typically occurs when the **absolute value of the** $t$-statistic for the new variable is **greater than 1** (assuming large samples and standard model assumptions).

------------------------------------------------------------------------

**Limitations of Adjusted** $R^2$

-   **Sensitive to Assumptions:** Adjusted $R^2$ assumes **homoskedasticity** (constant variance of errors) and **no autocorrelation**. In the presence of heteroskedasticity, its interpretation may be misleading.
-   **Not a Substitute for Hypothesis Testing:** It **should not** be the primary criterion for deciding which variables to include in a model.
    -   Use $t$-tests to evaluate the significance of individual coefficients.
    -   Use $F$-tests for assessing the joint significance of multiple variables.

------------------------------------------------------------------------

### Comparing Models with Transformed Dependent Variables

When comparing regression models with different transformations of the dependent variable, such as level and log-linear models, direct comparisons using traditional goodness-of-fit metrics like $R^2$ or adjusted $R^2$ are invalid. This is because the transformation changes the scale of the dependent variable, affecting the calculation of the Total Sum of Squares (SST), which is the denominator in $R^2$ calculations.

------------------------------------------------------------------------

**Model Specifications**

1.  **Level Model (Linear):**

$$
y = \beta_0 + \beta_1 x_1 + \epsilon
$$

2.  **Log-Linear Model:**

$$
\ln(y) = \beta_0 + \beta_1 x_1 + \epsilon
$$

Where:

-   $y$ is the dependent variable,

-   $x_1$ is an independent variable,

-   $\epsilon$ represents the error term.

------------------------------------------------------------------------

**Interpretation of Coefficients**

-   **In the Level Model:**\
    The effect of $x_1$ on $y$ is **constant**, regardless of the magnitude of $y$. Specifically, a **one-unit increase** in $x_1$ results in a change of $\beta_1$ units in $y$. This implies:

    $$
    \Delta y = \beta_1 \cdot \Delta x_1
    $$

-   **In the Log Model:**\
    The effect of $x_1$ on $y$ is **proportional** to the current level of $y$. A **one-unit increase** in $x_1$ leads to a **percentage change** in $y$, approximately equal to $100 \times \beta_1\%$. Specifically:

    $$
    \Delta \ln(y) = \beta_1 \cdot \Delta x_1 \quad \Rightarrow \quad \% \Delta y \approx 100 \times \beta_1
    $$

    -   For small values of $y$, the absolute change is small.
    -   For large values of $y$, the absolute change is larger, reflecting the multiplicative nature of the model.

------------------------------------------------------------------------

Why We Cannot Compare $R^2$ or Adjusted $R^2$ Directly

-   The level model explains variance in the original scale of $y$, while the log model explains variance in the logarithmic scale of $y$.
-   The SST (Total Sum of Squares) differs across the models because the dependent variable is transformed, making direct comparisons of $R^2$ invalid.
-   Adjusted $R^2$ does not resolve this issue because it also depends on the scale of the dependent variable.

------------------------------------------------------------------------

**Approach to Compare Model Fit Across Transformations**

To compare models on the **same scale** as the original dependent variable ($y$), we need to **"un-transform"** the predictions from the log model. Here's the step-by-step procedure:

------------------------------------------------------------------------

**Step-by-Step Procedure**

1.  **Estimate the Log Model**\
    Fit the log-linear model and obtain the predicted values:

    $$
    \widehat{\ln(y)} = \hat{\beta}_0 + \hat{\beta}_1 x_1
    $$

2.  **Un-Transform the Predictions**\
    Convert the predicted values back to the original scale of $y$ using the exponential function:

    $$
    \hat{m} = \exp(\widehat{\ln(y)})
    $$

    -   This transformation assumes that the errors are homoskedastic in the log model.
    -   **Note:** To correct for potential bias due to Jensen's inequality, a **smearing estimator** can be applied, but for simplicity, we use the basic exponential transformation here.

3.  **Fit a Regression Without an Intercept**\
    Regress the actual $y$ on the un-transformed predictions $\hat{m}$ **without an intercept**:

    $$
    y = \alpha \hat{m} + u
    $$

    -   The coefficient $\alpha$ adjusts for any scaling differences between the predicted and actual values.
    -   The residual term $u$ captures the unexplained variance.

4.  **Compute the Scaled** $R^2$\
    Calculate the **squared correlation** between the observed $y$ and the predicted values $\hat{y}$ from the above regression:

    $$
    R^2_{\text{scaled}} = \left( \text{Corr}(y, \hat{y}) \right)^2
    $$

    -   This **scaled** $R^2$ represents how well the log-transformed model predicts the original $y$ on its natural scale.
    -   Now, you can **compare** $R^2_{\text{scaled}}$ from the log model with the **regular** $R^2$ from the level model.

------------------------------------------------------------------------

**Key Insights**

-   **If** $R^2_{\text{scaled}}$ (from the log model) \> $R^2$ (from the level model): The log model fits the data better.
-   **If** $R^2_{\text{scaled}}$ \< $R^2$ (from the level model): The level model provides a better fit.
-   **If both are similar:** Consider other model diagnostics, theoretical justification, or model simplicity.

------------------------------------------------------------------------

**Caveats and Considerations**

-   **Heteroskedasticity:** If heteroskedasticity is present, the un-transformation may introduce bias.

-   **Error Distribution:** Log-transformed models assume multiplicative errors, which may not be appropriate in all contexts.

-   **Smearing Estimator (Advanced Correction):** To adjust for bias in the back-transformation, apply the smearing estimator:

    $$
    \hat{y} = \exp(\widehat{\ln(y)}) \cdot \hat{S}
    $$

    Where $\hat{S}$ is the **mean of the exponentiated residuals** from the log model.


```r
# Install and load necessary libraries
# install.packages("nonnest2")  # Uncomment if not already installed
library(nonnest2)    # For Vuong Test
library(lmtest)      # For J-Test

# Simulated dataset
set.seed(123)
n <- 100
x <- rnorm(n, mean = 50, sd = 10)
z <- rnorm(n, mean = 100, sd = 20)
epsilon <- rnorm(n)

# Competing models (non-nested)
# Model A: Linear relationship with x
y <- 5 + 0.3 * x + epsilon
model_A <- lm(y ~ x)

# Model B: Log-linear relationship with z
model_B <- lm(y ~ log(z))

# ----------------------------------------------------------------------
# Vuong Test (Correct Function)
# ----------------------------------------------------------------------
vuong_test <- vuongtest(model_A, model_B)
print(vuong_test)
#> 
#> Model 1 
#>  Class: lm 
#>  Call: lm(formula = y ~ x)
#> 
#> Model 2 
#>  Class: lm 
#>  Call: lm(formula = y ~ log(z))
#> 
#> Variance test 
#>   H0: Model 1 and Model 2 are indistinguishable 
#>   H1: Model 1 and Model 2 are distinguishable 
#>     w2 = 0.681,   p = 2.35e-08
#> 
#> Non-nested likelihood ratio test 
#>   H0: Model fits are equal for the focal population 
#>   H1A: Model 1 fits better than Model 2 
#>     z = 13.108,   p = <2e-16
#>   H1B: Model 2 fits better than Model 1 
#>     z = 13.108,   p = 1

# ----------------------------------------------------------------------
# Davidson–MacKinnon J-Test
# ----------------------------------------------------------------------

# Step 1: Testing Model A against Model B
# Obtain fitted values from Model B
fitted_B <- fitted(model_B)

# Auxiliary regression: Add fitted_B to Model A
j_test_A_vs_B <- lm(y ~ x + fitted_B)
summary(j_test_A_vs_B)
#> 
#> Call:
#> lm(formula = y ~ x + fitted_B)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -1.8717 -0.6573 -0.1223  0.6154  2.0952 
#> 
#> Coefficients:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) 14.70881   25.98307   0.566    0.573    
#> x            0.28671    0.01048  27.358   <2e-16 ***
#> fitted_B    -0.43702    1.27500  -0.343    0.733    
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 0.951 on 97 degrees of freedom
#> Multiple R-squared:  0.8854,	Adjusted R-squared:  0.883 
#> F-statistic: 374.5 on 2 and 97 DF,  p-value: < 2.2e-16

# Step 2: Testing Model B against Model A
# Obtain fitted values from Model A
fitted_A <- fitted(model_A)

# Auxiliary regression: Add fitted_A to Model B
j_test_B_vs_A <- lm(y ~ log(z) + fitted_A)
summary(j_test_B_vs_A)
#> 
#> Call:
#> lm(formula = y ~ log(z) + fitted_A)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -1.8717 -0.6573 -0.1223  0.6154  2.0952 
#> 
#> Coefficients:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept) -0.77868    2.39275  -0.325    0.746    
#> log(z)       0.16829    0.49097   0.343    0.733    
#> fitted_A     1.00052    0.03657  27.358   <2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 0.951 on 97 degrees of freedom
#> Multiple R-squared:  0.8854,	Adjusted R-squared:  0.883 
#> F-statistic: 374.5 on 2 and 97 DF,  p-value: < 2.2e-16
```

------------------------------------------------------------------------

## Heteroskedasticity Tests

**Heteroskedasticity** occurs when the variance of the error terms ($\epsilon_i$) in a regression model is **not constant** across observations. This violates the Classical [OLS Assumption](#ols-assumptions), specifically the assumption of **homoskedasticity** (Assumption [A4 Homoskedasticity] in the [Gauss-Markov Theorem]), which states:

$$
\text{Var}(\epsilon_i) = \sigma^2 \quad \forall \, i
$$

When heteroskedasticity is present:

-   [Ordinary Least Squares] estimators remain **unbiased** but become **inefficient** (i.e., no longer Best Linear Unbiased Estimators---BLUE).

-   The **standard errors** of the estimates are biased, leading to unreliable **hypothesis tests** (e.g., $t$-tests and $F$-tests).

Detecting heteroskedasticity is crucial for ensuring the validity of regression results. This section covers key tests used to identify heteroskedasticity:

-   [Breusch--Pagan Test](#sec-breusch–pagan-test)
-   [White Test](#sec-white-test-hetero)
-   [Goldfeld--Quandt Test](#sec-goldfeld–quandt-test)
-   [Park Test](#sec-park-test)
-   [Glejser Test](#sec-glejser-test)

------------------------------------------------------------------------

### Breusch--Pagan Test {#sec-breusch--pagan-test}

The **Breusch--Pagan (BP) Test** is one of the most widely used tests for detecting heteroskedasticity [@breusch1979simple]. It examines whether the variance of the residuals depends on the independent variables.

**Hypotheses**

-   Null Hypothesis ($H_0$): Homoskedasticity ($\text{Var}(\epsilon_i) = \sigma^2$ is constant).
-   Alternative Hypothesis ($H_1$): Heteroskedasticity exists; the variance of $\epsilon_i$ depends on the independent variables.

------------------------------------------------------------------------

**Procedure**

1.  **Estimate the original regression model:**

$$
y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \dots + \beta_k x_{ki} + \epsilon_i
$$

Obtain the residuals $\hat{\epsilon}_i$ from this regression.

2.  **Compute the squared residuals:**

$$
\hat{\epsilon}_i^2
$$

3.  **Auxiliary Regression:** Regress the squared residuals on the independent variables:

$$
\hat{\epsilon}_i^2 = \alpha_0 + \alpha_1 x_{1i} + \alpha_2 x_{2i} + \dots + \alpha_k x_{ki} + u_i
$$

4.  **Calculate the Test Statistic:**

The BP test statistic is:

$$
\text{BP} = n \cdot R^2_{\text{aux}}
$$

Where:

-   $n$ is the sample size,

-   $R^2_{\text{aux}}$ is the $R^2$ from the auxiliary regression.

5.  **Decision Rule:**

-   Under $H_0$, the BP statistic follows a **chi-squared distribution** with $k$ degrees of freedom (where $k$ is the number of independent variables):

$$
\text{BP} \sim \chi^2_k
$$

-   **Reject** $H_0$ if the BP statistic exceeds the critical value from the chi-squared distribution.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Simple to implement; directly tests the relationship between residual variance and regressors.
-   **Limitation:** Sensitive to non-normality; less effective when heteroskedasticity is not linearly related to independent variables.

------------------------------------------------------------------------

### White Test {#sec-white-test-hetero}

The **White Test** is a more general heteroskedasticity test that does not require specifying the form of heteroskedasticity [@white1980heteroskedasticity]. It can detect both **linear** and **nonlinear** forms.

**Hypotheses**

-   Null Hypothesis ($H_0$): Homoskedasticity.
-   Alternative Hypothesis ($H_1$): Heteroskedasticity (of any form).

------------------------------------------------------------------------

**Procedure**

1.  **Estimate the original regression model** and obtain residuals $\hat{\epsilon}_i$.

2.  **Auxiliary Regression:** Regress the squared residuals on:

    -   The original independent variables ($x_{1i}, x_{2i}, \dots, x_{ki}$),
    -   Their **squares** ($x_{1i}^2, x_{2i}^2, \dots, x_{ki}^2$),
    -   Their **cross-products** (e.g., $x_{1i} x_{2i}$).

    The auxiliary regression is:

    $$
    \hat{\epsilon}_i^2 = \alpha_0 + \alpha_1 x_{1i} + \alpha_2 x_{2i} + \dots + \alpha_k x_{ki} + \alpha_{k+1} x_{1i}^2 + \dots + \alpha_{2k} x_{ki}^2 + \alpha_{2k+1} (x_{1i} x_{2i}) + u_i
    $$

3.  **Calculate the Test Statistic:**

    $$
    \text{White} = n \cdot R^2_{\text{aux}}
    $$

4.  **Decision Rule:**

-   Under $H_0$, the statistic follows a **chi-squared distribution** with degrees of freedom equal to the number of auxiliary regressors:

    $$
    \text{White} \sim \chi^2_{\text{df}}
    $$

-   **Reject** $H_0$ if the statistic exceeds the critical chi-squared value.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Can detect a wide range of heteroskedasticity patterns.
-   **Limitation:** May suffer from overfitting in small samples due to many auxiliary regressors.

------------------------------------------------------------------------

### Goldfeld--Quandt Test {#sec-goldfeld--quandt-test}

The **Goldfeld--Quandt Test** is a simple test that detects heteroskedasticity by comparing the variance of residuals in two different subsets of the data [@goldfeld1965some].

**Hypotheses**

-   Null Hypothesis ($H_0$): Homoskedasticity.
-   Alternative Hypothesis ($H_1$): Heteroskedasticity; variances differ between groups.

------------------------------------------------------------------------

**Procedure**

1.  **Sort the data** based on an independent variable suspected to cause heteroskedasticity.

2.  **Split the data** into three groups:

    -   **Group 1:** Lower values,
    -   **Group 2:** Middle values (often omitted),
    -   **Group 3:** Higher values.

3.  **Estimate the regression model** separately for Groups 1 and 3. Obtain the residual sum of squares ($SSR_1$ and $SSR_2$).

4.  **Calculate the Test Statistic:**

    $$
    F = \frac{SSR_2 / (n_2 - k)}{SSR_1 / (n_1 - k)}
    $$

    Where:

    -   $n_1$ and $n_2$ are the number of observations in Groups 1 and 3, respectively,
    -   $k$ is the number of estimated parameters.

5.  **Decision Rule:**

-   Under $H_0$, the test statistic follows an $F$-distribution with $(n_2 - k, n_1 - k)$ degrees of freedom:

    $$
    F \sim F_{(n_2 - k, n_1 - k)}
    $$

-   **Reject** $H_0$ if $F$ exceeds the critical value.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Simple to apply when heteroskedasticity is suspected to vary systematically with an independent variable.
-   **Limitation:** Requires arbitrary splitting of data and assumes the error variance changes abruptly between groups.

------------------------------------------------------------------------

### Park Test {#sec-park-test}

The **Park Test** identifies heteroskedasticity by modeling the error variance as a function of an independent variable [@park1966estimation].

**Hypotheses**

-   Null Hypothesis ($H_0$): Homoskedasticity.
-   Alternative Hypothesis ($H_1$): Heteroskedasticity; variance depends on an independent variable.

------------------------------------------------------------------------

**Procedure**

1.  **Estimate the original regression** and obtain residuals $\hat{\epsilon}_i$.

2.  **Transform the residuals:** Take the natural logarithm of the squared residuals:

    $$
    \ln(\hat{\epsilon}_i^2)
    $$

3.  **Auxiliary Regression:** Regress $\ln(\hat{\epsilon}_i^2)$ on the independent variable(s):

    $$
    \ln(\hat{\epsilon}_i^2) = \alpha_0 + \alpha_1 \ln(x_i) + u_i
    $$

4.  **Decision Rule:**

-   Test whether $\alpha_1 = 0$ using a $t$-test.
-   **Reject** $H_0$ if $\alpha_1$ is statistically significant, indicating heteroskedasticity.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Simple to implement; works well when the variance follows a log-linear relationship.
-   **Limitation:** Assumes a specific functional form for the variance, which may not hold in practice.

------------------------------------------------------------------------

### Glejser Test {#sec-glejser-test}

The **Glejser Test** detects heteroskedasticity by regressing the **absolute value of residuals** on the independent variables [@glejser1969new].

**Hypotheses**

-   Null Hypothesis ($H_0$): Homoskedasticity.
-   Alternative Hypothesis ($H_1$): Heteroskedasticity exists.

------------------------------------------------------------------------

**Procedure**

1.  **Estimate the original regression** and obtain residuals $\hat{\epsilon}_i$.

2.  **Auxiliary Regression:** Regress the absolute residuals on the independent variables:

    $$
    |\hat{\epsilon}_i| = \alpha_0 + \alpha_1 x_{1i} + \alpha_2 x_{2i} + \dots + \alpha_k x_{ki} + u_i
    $$

3.  **Decision Rule:**

-   Test the significance of the coefficients ($\alpha_1, \alpha_2, \dots$) using $t$-tests.
-   **Reject** $H_0$ if any coefficient is statistically significant, indicating heteroskedasticity.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Flexible; can detect various forms of heteroskedasticity.
-   **Limitation:** Sensitive to outliers since it relies on absolute residuals.

------------------------------------------------------------------------

### Summary of Heteroskedasticity Tests

| **Test**                                      | **Type**                 | **Assumptions**                     | **Key Statistic** | **When to Use**                          |
|-----------------------------------------------|--------------------------|-------------------------------------|-------------------|------------------------------------------|
| [Breusch--Pagan](#sec-breusch–pagan-test)     | Parametric               | Linear relationship with predictors | $\chi^2$          | General-purpose test                     |
| [White](#sec-white-test-hetero)               | General (non-parametric) | No functional form assumption       | $\chi^2$          | Detects both linear & nonlinear forms    |
| [Goldfeld--Quandt](#sec-goldfeld–quandt-test) | Group comparison         | Assumes known ordering of variance  | $F$-distribution  | When heteroskedasticity varies by groups |
| [Park](#sec-park-test)                        | Parametric (log-linear)  | Assumes log-linear variance         | $t$-test          | When variance depends on predictors      |
| [Glejser](#sec-glejser-test)                  | Parametric               | Based on absolute residuals         | $t$-test          | Simple test for variance dependence      |

Detecting heteroskedasticity is critical for ensuring the reliability of regression models. While each test has strengths and limitations, combining multiple tests can provide robust insights. Once heteroskedasticity is detected, consider using **robust standard errors** or alternative estimation techniques (e.g., [Generalized Least Squares] or [Weighted Least Squares]) to address the issue.


```r
# Install and load necessary libraries
# install.packages("lmtest")      # For Breusch–Pagan Test
# install.packages("car")         # For additional regression diagnostics
# install.packages("sandwich")    # For robust covariance estimation


library(lmtest)
library(car)
library(sandwich)

# Simulated dataset
set.seed(123)
n <- 100
x1 <- rnorm(n, mean = 50, sd = 10)
x2 <- rnorm(n, mean = 30, sd = 5)
epsilon <-
    rnorm(n, sd = x1 * 0.1)  # Heteroskedastic errors increasing with x1
y <- 5 + 0.4 * x1 - 0.3 * x2 + epsilon

# Original regression model
model <- lm(y ~ x1 + x2)

# ----------------------------------------------------------------------
# 1. Breusch–Pagan Test
# ----------------------------------------------------------------------
# Null Hypothesis: Homoskedasticity
bp_test <- bptest(model)
print(bp_test)
#> 
#> 	studentized Breusch-Pagan test
#> 
#> data:  model
#> BP = 7.8141, df = 2, p-value = 0.0201

# ----------------------------------------------------------------------
# 2. White Test (using Breusch–Pagan framework with squares & interactions)
# ----------------------------------------------------------------------
# Create squared and interaction terms
model_white <-
    lm(residuals(model) ^ 2 ~ x1 + x2 + I(x1 ^ 2) + I(x2 ^ 2) + I(x1 * x2))
white_statistic <-
    summary(model_white)$r.squared * n  # White Test Statistic
df_white <-
    length(coef(model_white)) - 1              # Degrees of freedom
p_value_white <- 1 - pchisq(white_statistic, df_white)

# Display White Test result
cat("White Test Statistic:", white_statistic, "\n")
#> White Test Statistic: 11.85132
cat("Degrees of Freedom:", df_white, "\n")
#> Degrees of Freedom: 5
cat("P-value:", p_value_white, "\n")
#> P-value: 0.0368828

# ----------------------------------------------------------------------
# 3. Goldfeld–Quandt Test
# ----------------------------------------------------------------------
# Null Hypothesis: Homoskedasticity
# Sort data by x1 (suspected source of heteroskedasticity)
gq_test <-
    gqtest(model, order.by = ~ x1, fraction = 0.2)  # Omit middle 20% of data
print(gq_test)
#> 
#> 	Goldfeld-Quandt test
#> 
#> data:  model
#> GQ = 1.8352, df1 = 37, df2 = 37, p-value = 0.03434
#> alternative hypothesis: variance increases from segment 1 to 2

# ----------------------------------------------------------------------
# 4. Park Test
# ----------------------------------------------------------------------
# Step 1: Get residuals and square them
residuals_squared <- residuals(model) ^ 2

# Step 2: Log-transform squared residuals
log_residuals_squared <- log(residuals_squared)

# Step 3: Regress log(residuals^2) on log(x1) (assuming variance depends on x1)
park_test <- lm(log_residuals_squared ~ log(x1))
summary(park_test)
#> 
#> Call:
#> lm(formula = log_residuals_squared ~ log(x1))
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -9.3633 -1.3424  0.4218  1.6089  3.0697 
#> 
#> Coefficients:
#>             Estimate Std. Error t value Pr(>|t|)
#> (Intercept)  -1.6319     4.5982  -0.355    0.723
#> log(x1)       0.8903     1.1737   0.759    0.450
#> 
#> Residual standard error: 2.171 on 98 degrees of freedom
#> Multiple R-squared:  0.005837,	Adjusted R-squared:  -0.004308 
#> F-statistic: 0.5754 on 1 and 98 DF,  p-value: 0.4499

# ----------------------------------------------------------------------
# 5. Glejser Test
# ----------------------------------------------------------------------
# Step 1: Absolute value of residuals
abs_residuals <- abs(residuals(model))

# Step 2: Regress absolute residuals on independent variables
glejser_test <- lm(abs_residuals ~ x1 + x2)
summary(glejser_test)
#> 
#> Call:
#> lm(formula = abs_residuals ~ x1 + x2)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -4.3096 -2.2680 -0.4564  1.9554  8.3921 
#> 
#> Coefficients:
#>              Estimate Std. Error t value Pr(>|t|)  
#> (Intercept)  0.755846   2.554842   0.296   0.7680  
#> x1           0.064896   0.032852   1.975   0.0511 .
#> x2          -0.008495   0.062023  -0.137   0.8913  
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 2.98 on 97 degrees of freedom
#> Multiple R-squared:  0.0392,	Adjusted R-squared:  0.01939 
#> F-statistic: 1.979 on 2 and 97 DF,  p-value: 0.1438
```

**Interpretation of the Results**

1.  **Breusch--Pagan Test**

    -   **Null Hypothesis (**$H_0$): Homoskedasticity (constant error variance).

    -   **Alternative Hypothesis (**$H_1$): Heteroskedasticity exists (error variance depends on predictors).

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* $< 0.05$ → Evidence of heteroskedasticity.
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$ → No strong evidence of heteroskedasticity.

2.  **White Test**

    -   **Null Hypothesis (**$H_0$): Homoskedasticity.

    -   **Alternative Hypothesis (**$H_1$): Heteroskedasticity (of any form, linear or nonlinear).

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* $< 0.05$ → Presence of heteroskedasticity.
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$ → Homoskedasticity likely holds.

3.  **Goldfeld--Quandt Test**

    -   **Null Hypothesis (**$H_0$): Homoskedasticity (equal variances across groups).

    -   **Alternative Hypothesis (**$H_1$): Heteroskedasticity (unequal variances between groups).

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* $< 0.05$ → Variances differ between groups, indicating heteroskedasticity.
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$ → No significant evidence of heteroskedasticity.

4.  **Park Test**

    -   **Null Hypothesis (**$H_0$): No relationship between the variance of errors and predictor(s) (homoskedasticity).

    -   **Alternative Hypothesis (**$H_1$): Variance of errors depends on predictor(s).

    -   **Decision Rule:**

        -   **Reject** $H_0$ if the *coefficient of* $\log(x_1)$ *is statistically significant* (*p-value* $< 0.05$).
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$.

5.  **Glejser Test**

    -   **Null Hypothesis (**$H_0$): Homoskedasticity (no relationship between absolute residuals and predictors).

    -   **Alternative Hypothesis (**$H_1$): Heteroskedasticity exists.

    -   **Decision Rule:**

        -   **Reject** $H_0$ if any predictor is statistically significant (*p-value* $< 0.05$).
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$.

------------------------------------------------------------------------

## Functional Form Tests

**Functional form misspecification** occurs when the chosen regression model does not correctly represent the true relationship between the dependent and independent variables. This can happen due to:

-   **Omitted variables** (important predictors not included),
-   **Incorrect transformations** of variables (e.g., missing nonlinear relationships),
-   **Incorrect interaction terms** (missing interaction effects between variables),
-   **Inappropriate linearity assumptions**.

Functional form errors can lead to **biased and inconsistent** estimators, undermining the validity of statistical inferences. To detect such issues, several diagnostic tests are available.

**Key Functional Form Tests:**

1.  [Ramsey RESET Test (Regression Equation Specification Error Test)](#sec-ramsey-reset-test)
2.  [Harvey--Collier Test](#sec-harvey–collier-test)
3.  [Rainbow Test](#sec-rainbow-test)

Each test focuses on identifying different aspects of potential model misspecification.

------------------------------------------------------------------------

### Ramsey RESET Test (Regression Equation Specification Error Test) {#sec-ramsey-reset-test}

The **Ramsey RESET Test** is one of the most widely used tests to detect functional form misspecification [@ramsey1969tests]. It examines whether adding **nonlinear transformations** of the fitted values (or regressors) improves the model fit.

**Hypotheses**

-   Null Hypothesis ($H_0$): The model is correctly specified.
-   Alternative Hypothesis ($H_1$): The model suffers from omitted variables, incorrect functional form, or other specification errors.

------------------------------------------------------------------------

**Procedure**

1.  **Estimate the original regression model:**

    $$
    y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \dots + \beta_k x_{ki} + \epsilon_i
    $$

2.  **Obtain the fitted values:**

    $$
    \hat{y}_i
    $$

3.  **Augment the model with powers of the fitted values (squared, cubed, etc.):**

    $$
    y_i = \beta_0 + \beta_1 x_{1i} + \dots + \beta_k x_{ki} + \gamma_1 \hat{y}_i^2 + \gamma_2 \hat{y}_i^3 + u_i
    $$

4.  **Test the joint significance** of the added terms:

    $$
    H_0: \gamma_1 = \gamma_2 = 0
    $$

5.  **Compute the F-statistic:**

    $$
    F = \frac{(SSR_{\text{restricted}} - SSR_{\text{unrestricted}}) / q}{SSR_{\text{unrestricted}} / (n - k - q - 1)}
    $$

    Where:

    -   $SSR_{\text{restricted}}$ = Sum of Squared Residuals from the original model,
    -   $SSR_{\text{unrestricted}}$ = SSR from the augmented model,
    -   $q$ = Number of additional terms (e.g., 2 if adding $\hat{y}^2$ and $\hat{y}^3$),
    -   $n$ = Sample size,
    -   $k$ = Number of predictors in the original model.

------------------------------------------------------------------------

**Decision Rule**

-   Under $H_0$, the F-statistic follows an $F$-distribution with $(q, n - k - q - 1)$ degrees of freedom.
-   Reject $H_0$ if the F-statistic exceeds the critical value, indicating functional form misspecification.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Simple to implement; detects omitted variables and incorrect functional forms.
-   **Limitation:** Does not identify which variable or functional form is incorrect---only indicates the presence of an issue.

------------------------------------------------------------------------

### Harvey--Collier Test {#sec-harvey--collier-test}

The **Harvey--Collier Test** evaluates whether the model's residuals display systematic patterns, which would indicate functional form misspecification [@harvey1977testing]. It is based on testing for a **non-zero mean** in the residuals after projection onto specific components.

**Hypotheses**

-   Null Hypothesis ($H_0$): The model is correctly specified (residuals are random noise with zero mean).
-   Alternative Hypothesis ($H_1$): The model is misspecified (residuals contain systematic patterns).

------------------------------------------------------------------------

**Procedure**

1.  **Estimate the original regression model** and obtain residuals $\hat{\epsilon}_i$.

2.  **Project the residuals** onto the space spanned by a specially constructed test vector (often derived from the inverse of the design matrix in linear regression).

3.  **Calculate the Harvey--Collier statistic:**

    $$
    t = \frac{\bar{\epsilon}}{\text{SE}(\bar{\epsilon})}
    $$

    Where:

    -   $\bar{\epsilon}$ is the mean of the projected residuals,
    -   $\text{SE}(\bar{\epsilon})$ is the standard error of the mean residual.

**Decision Rule:**

-   The test statistic follows a $t$-distribution under $H_0$.
-   **Reject** $H_0$ if the $t$-statistic is significantly different from zero.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Simple to apply and interpret; good for detecting subtle misspecifications.
-   **Limitation:** Sensitive to outliers; may have reduced power in small samples.

------------------------------------------------------------------------

### Rainbow Test {#sec-rainbow-test}

The **Rainbow Test** is a general-purpose diagnostic tool for functional form misspecification [@utts1982rainbow]. It compares the performance of the model on the **full sample** versus a **central subsample**, where the central subsample contains observations near the median of the independent variables.

**Hypotheses**

-   Null Hypothesis ($H_0$): The model is correctly specified.
-   Alternative Hypothesis ($H_1$): The model is misspecified.

------------------------------------------------------------------------

**Procedure**

1.  **Estimate the regression model** on the full dataset and record the residuals.

2.  **Identify a central subsample** (e.g., observations near the median of key predictors).

3.  **Estimate the model again** on the central subsample.

4.  **Compare the predictive accuracy** between the full sample and subsample using an F-statistic:

    $$
    F = \frac{(SSR_{\text{full}} - SSR_{\text{subsample}}) / q}{SSR_{\text{subsample}} / (n - k - q)}
    $$

    Where $q$ is the number of restrictions implied by using the subsample.

------------------------------------------------------------------------

**Decision Rule**

-   Under $H_0$, the test statistic follows an $F$-distribution.
-   Reject $H_0$ if the F-statistic is significant, indicating model misspecification.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Robust to various forms of misspecification.
-   **Limitation:** Choice of subsample may influence results; less informative about the specific nature of the misspecification.

------------------------------------------------------------------------

### Summary of Functional Form Tests

| **Test**                                         | **Type**             | **Key Statistic** | **Purpose**                               | **When to Use**                           |
|--------------------------------------------------|----------------------|-------------------|-------------------------------------------|-------------------------------------------|
| [Ramsey RESET Test](#sec-ramsey-reset-test)      | Augmented regression | $F$-test          | Detects omitted variables, nonlinearities | General model specification testing       |
| [Harvey--Collier Test](#sec-harvey–collier-test) | Residual-based       | $t$-test          | Detects systematic patterns in residuals  | Subtle misspecifications in linear models |
| [Rainbow Test](#sec-rainbow-test)                | Subsample comparison | $F$-test          | Tests model stability across subsamples   | Comparing central vs. full sample         |

Functional form misspecification can severely distort regression results, leading to biased estimates and invalid inferences. While no single test can detect all types of misspecification, using a combination of tests provides a robust framework for model diagnostics.


```r
# Install and load necessary libraries
# install.packages("lmtest")      # For RESET and Harvey–Collier Test
# install.packages("car")         # For diagnostic tests
# install.packages("strucchange") # For Rainbow Test

library(lmtest)
library(car)
library(strucchange)

# Simulated dataset
set.seed(123)
n <- 100
x1 <- rnorm(n, mean = 50, sd = 10)
x2 <- rnorm(n, mean = 30, sd = 5)
epsilon <- rnorm(n)
y <- 5 + 0.4 * x1 - 0.3 * x2 + epsilon

# Original regression model
model <- lm(y ~ x1 + x2)

# ----------------------------------------------------------------------
# 1. Ramsey RESET Test
# ----------------------------------------------------------------------
# Null Hypothesis: The model is correctly specified
reset_test <-
    resettest(model, power = 2:3, type = "fitted")  # Adds ŷ² and ŷ³
print(reset_test)
#> 
#> 	RESET test
#> 
#> data:  model
#> RESET = 0.1921, df1 = 2, df2 = 95, p-value = 0.8255

# ----------------------------------------------------------------------
# 2. Harvey–Collier Test
# ----------------------------------------------------------------------
# Null Hypothesis: The model is correctly specified (residuals have zero mean)
hc_test <- harvtest(model)
print(hc_test)
#> 
#> 	Harvey-Collier test
#> 
#> data:  model
#> HC = 0.041264, df = 96, p-value = 0.9672

# ----------------------------------------------------------------------
# 3. Rainbow Test
# ----------------------------------------------------------------------
# Null Hypothesis: The model is correctly specified
rainbow_test <- lmtest::raintest (model)
print(rainbow_test)
#> 
#> 	Rainbow test
#> 
#> data:  model
#> Rain = 1.1857, df1 = 50, df2 = 47, p-value = 0.279
```

**Interpretation of the Results**

1.  **Ramsey RESET Test (Regression Equation Specification Error Test)**

    -   **Null Hypothesis (**$H_0$): The model is correctly specified.

    -   **Alternative Hypothesis (**$H_1$): The model suffers from omitted variables, incorrect functional form, or other specification errors.

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* $< 0.05$ → Evidence of model misspecification (e.g., missing nonlinear terms).
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$ → No strong evidence of misspecification.

2.  **Harvey--Collier Test**

    -   **Null Hypothesis (**$H_0$): The model is correctly specified (residuals are random noise with zero mean).

    -   **Alternative Hypothesis (**$H_1$): The model is misspecified (residuals contain systematic patterns).

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* $< 0.05$ → Model misspecification detected (non-random residual patterns).
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$ → No evidence of misspecification.

3.  **Rainbow Test**

    -   **Null Hypothesis (**$H_0$): The model is correctly specified.

    -   **Alternative Hypothesis (**$H_1$): The model is misspecified.

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* $< 0.05$ → Evidence of model misspecification (model performs differently on subsets).
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$ → Model specification appears valid.

------------------------------------------------------------------------

## Autocorrelation Tests

**Autocorrelation** (also known as **serial correlation**) occurs when the error terms ($\epsilon_t$) in a regression model are **correlated across observations**, violating the assumption of **independence** in the classical linear regression model. This issue is particularly common in **time-series data**, where observations are ordered over time.

**Consequences of Autocorrelation:**

-   [OLS](%7B#ordinary-least-squares%7D) estimators remain unbiased but become **inefficient**, meaning they do not have the minimum variance among all linear unbiased estimators.
-   **Standard errors** are biased, leading to unreliable hypothesis tests (e.g., $t$-tests and $F$-tests).
-   Potential **underestimation of standard errors**, increasing the risk of Type I errors (false positives).

------------------------------------------------------------------------

### Durbin--Watson Test {#sec-durbin--watson-test}

The **Durbin--Watson (DW) Test** is the most widely used test for detecting **first-order autocorrelation**, where the current error term is correlated with the previous one:

$$
\epsilon_t = \rho \, \epsilon_{t-1} + u_t
$$

Where:

-   $\rho$ is the autocorrelation coefficient,

-   $u_t$ is a white noise error term.

------------------------------------------------------------------------

**Hypotheses**

-   Null Hypothesis ($H_0$): No first-order autocorrelation ($\rho = 0$).
-   Alternative Hypothesis ($H_1$): First-order autocorrelation exists ($\rho \neq 0$).

------------------------------------------------------------------------

**Durbin--Watson Test Statistic**

The DW statistic is calculated as:

$$
DW = \frac{\sum_{t=2}^{n} (\hat{\epsilon}_t - \hat{\epsilon}_{t-1})^2}{\sum_{t=1}^{n} \hat{\epsilon}_t^2}
$$

Where:

-   $\hat{\epsilon}_t$ are the residuals from the regression,

-   $n$ is the number of observations.

------------------------------------------------------------------------

**Decision Rule**

-   The DW statistic ranges from **0 to 4**:
    -   **DW ≈ 2**: No autocorrelation.
    -   **DW \< 2**: Positive autocorrelation.
    -   **DW \> 2**: Negative autocorrelation.

For more precise interpretation:

-   Use **Durbin--Watson critical values** ($d_L$ and $d_U$) to form decision boundaries.

-   If the test statistic falls into an **inconclusive range**, consider alternative tests like the [Breusch--Godfrey test](#sec-breusch–godfrey-test).

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Simple to compute; specifically designed for detecting first-order autocorrelation.
-   **Limitation:** Inconclusive in some cases; **invalid** when lagged dependent variables are included in the model.

------------------------------------------------------------------------

### Breusch--Godfrey Test {#sec-breusch--godfrey-test}

The **Breusch--Godfrey (BG) Test** is a more general approach that can detect **higher-order autocorrelation** (e.g., second-order, third-order) and is valid even when **lagged dependent variables** are included in the model [@breusch1978testing; @godfrey1978testing].

------------------------------------------------------------------------

**Hypotheses**

-   Null Hypothesis ($H_0$): No autocorrelation of any order (up to lag $p$).
-   Alternative Hypothesis ($H_1$): Autocorrelation exists at some lag(s).

------------------------------------------------------------------------

**Procedure**

1.  **Estimate the original regression model** and obtain residuals $\hat{\epsilon}_t$:

    $$
    y_t = \beta_0 + \beta_1 x_{1t} + \dots + \beta_k x_{kt} + \epsilon_t
    $$

2.  **Run an auxiliary regression** by regressing the residuals on the original regressors plus $p$ lagged residuals:

    $$
    \hat{\epsilon}_t = \alpha_0 + \alpha_1 x_{1t} + \dots + \alpha_k x_{kt} + \rho_1 \hat{\epsilon}_{t-1} + \dots + \rho_p \hat{\epsilon}_{t-p} + u_t
    $$

3.  **Calculate the test statistic:**

    $$
    \text{BG} = n \cdot R^2_{\text{aux}}
    $$

    Where:

    -   $n$ is the sample size,
    -   $R^2_{\text{aux}}$ is the $R^2$ from the auxiliary regression.

**Decision Rule:**

-   Under $H_0$, the BG statistic follows a **chi-squared distribution** with $p$ degrees of freedom:

    $$
    \text{BG} \sim \chi^2_p
    $$

-   **Reject** $H_0$ if the statistic exceeds the critical chi-squared value, indicating the presence of autocorrelation.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Can detect **higher-order autocorrelation**; valid with lagged dependent variables.
-   **Limitation:** More computationally intensive than the [Durbin--Watson test](#sec-durbin–watson-test).

------------------------------------------------------------------------

### Ljung--Box Test (or Box--Pierce Test) {#sec-ljung--box-test}

The **Ljung--Box Test** is a **portmanteau test** designed to detect **autocorrelation at multiple lags** simultaneously [@box1970distribution; @ljung1978measure]. It is commonly used in **time-series analysis** to check residual autocorrelation after model estimation (e.g., in ARIMA models).

------------------------------------------------------------------------

**Hypotheses**

-   Null Hypothesis ($H_0$): No autocorrelation up to lag $h$.
-   Alternative Hypothesis ($H_1$): Autocorrelation exists at one or more lags.

------------------------------------------------------------------------

**Ljung--Box Test Statistic**

The Ljung--Box statistic is calculated as:

$$
Q = n(n + 2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n - k}
$$

Where:

-   $n$ = Sample size,
-   $h$ = Number of lags tested,
-   $\hat{\rho}_k$ = Sample autocorrelation at lag $k$.

------------------------------------------------------------------------

**Decision Rule**

-   Under $H_0$, the $Q$ statistic follows a **chi-squared distribution** with $h$ degrees of freedom:

    $$
    Q \sim \chi^2_h
    $$

-   **Reject** $H_0$ if $Q$ exceeds the critical value, indicating significant autocorrelation.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Detects autocorrelation across multiple lags simultaneously.
-   **Limitation:** Less powerful for detecting specific lag structures; sensitive to model misspecification.

------------------------------------------------------------------------

### Runs Test {#sec-runs-test}

The **Runs Test** is a **non-parametric test** that examines the randomness of residuals. It is based on the number of **runs**---sequences of consecutive residuals with the same sign.

------------------------------------------------------------------------

**Hypotheses**

-   Null Hypothesis ($H_0$): Residuals are randomly distributed (no autocorrelation).
-   Alternative Hypothesis ($H_1$): Residuals exhibit non-random patterns (indicating autocorrelation).

------------------------------------------------------------------------

**Procedure**

1.  **Classify residuals** as positive or negative.

2.  **Count the number of runs:** A **run** is a sequence of consecutive positive or negative residuals.

3.  **Calculate the expected number of runs under randomness:**

    $$
    E(R) = \frac{2 n_+ n_-}{n} + 1
    $$

    Where:

    -   $n_+$ = Number of positive residuals,
    -   $n_-$ = Number of negative residuals,
    -   $n = n_+ + n_-$.

4.  **Compute the test statistic (Z-score):**

    $$
    Z = \frac{R - E(R)}{\sqrt{\text{Var}(R)}}
    $$

    Where $\text{Var}(R)$ is the variance of the number of runs under the null hypothesis.

------------------------------------------------------------------------

**Decision Rule**

-   Under $H_0$, the $Z$-statistic follows a **standard normal distribution**:

    $$
    Z \sim N(0, 1)
    $$

-   **Reject** $H_0$ if $|Z|$ exceeds the critical value from the standard normal distribution.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Non-parametric; does not assume normality or linearity.
-   **Limitation:** Less powerful than parametric tests; primarily useful as a supplementary diagnostic.

------------------------------------------------------------------------

### Summary of Autocorrelation Tests

| **Test**                                      | **Type**                  | **Key Statistic** | **Detects**                      | **When to Use**                                         |
|-----------------------------------------------|---------------------------|-------------------|----------------------------------|---------------------------------------------------------|
| [Durbin--Watson](#sec-durbin–watson-test)     | Parametric                | $DW$              | First-order autocorrelation      | Simple linear models without lagged dependent variables |
| [Breusch--Godfrey](#sec-breusch–godfrey-test) | Parametric (general)      | $\chi^2$          | Higher-order autocorrelation     | Models with lagged dependent variables                  |
| [Ljung--Box](#sec-ljung–box-test)             | Portmanteau (global test) | $\chi^2$          | Autocorrelation at multiple lags | Time-series models (e.g., ARIMA)                        |
| [Runs Test](#sec-runs-test)                   | Non-parametric            | $Z$-statistic     | Non-random patterns in residuals | Supplementary diagnostic for randomness                 |

Detecting autocorrelation is crucial for ensuring the efficiency and reliability of regression models, especially in time-series analysis. While the [Durbin--Watson Test](#sec-durbin–watson-test) is suitable for detecting first-order autocorrelation, the [Breusch--Godfrey Test](#sec-breusch–godfrey-test) and [Ljung--Box Test](#sec-ljung–box-test-or-box–pierce-test) offer more flexibility for higher-order and multi-lag dependencies. Non-parametric tests like the [Runs Test](#sec-runs-test) serve as useful supplementary diagnostics.


```r
# Install and load necessary libraries
# install.packages("lmtest")       # For Durbin–Watson and Breusch–Godfrey Tests
# install.packages("tseries")      # For Runs Test
# install.packages("forecast")     # For Ljung–Box Test

library(lmtest)
library(tseries)
library(forecast)

# Simulated time-series dataset
set.seed(123)
n <- 100
time <- 1:n
x1 <- rnorm(n, mean = 50, sd = 10)
x2 <- rnorm(n, mean = 30, sd = 5)

# Introducing autocorrelation in errors
epsilon <- arima.sim(model = list(ar = 0.6), n = n)  # AR(1) process with ρ = 0.6
y <- 5 + 0.4 * x1 - 0.3 * x2 + epsilon

# Original regression model
model <- lm(y ~ x1 + x2)

# ----------------------------------------------------------------------
# 1. Durbin–Watson Test
# ----------------------------------------------------------------------
# Null Hypothesis: No first-order autocorrelation
dw_test <- dwtest(model)
print(dw_test)
#> 
#> 	Durbin-Watson test
#> 
#> data:  model
#> DW = 0.77291, p-value = 3.323e-10
#> alternative hypothesis: true autocorrelation is greater than 0

# ----------------------------------------------------------------------
# 2. Breusch–Godfrey Test
# ----------------------------------------------------------------------
# Null Hypothesis: No autocorrelation up to lag 2
bg_test <- bgtest(model, order = 2)  # Testing for autocorrelation up to lag 2
print(bg_test)
#> 
#> 	Breusch-Godfrey test for serial correlation of order up to 2
#> 
#> data:  model
#> LM test = 40.314, df = 2, p-value = 1.762e-09

# ----------------------------------------------------------------------
# 3. Ljung–Box Test
# ----------------------------------------------------------------------
# Null Hypothesis: No autocorrelation up to lag 10
ljung_box_test <- Box.test(residuals(model), lag = 10, type = "Ljung-Box")
print(ljung_box_test)
#> 
#> 	Box-Ljung test
#> 
#> data:  residuals(model)
#> X-squared = 50.123, df = 10, p-value = 2.534e-07

# ----------------------------------------------------------------------
# 4. Runs Test (Non-parametric)
# ----------------------------------------------------------------------
# Null Hypothesis: Residuals are randomly distributed
runs_test <- runs.test(as.factor(sign(residuals(model))))
print(runs_test)
#> 
#> 	Runs Test
#> 
#> data:  as.factor(sign(residuals(model)))
#> Standard Normal = -4.2214, p-value = 2.428e-05
#> alternative hypothesis: two.sided
```

**Interpretation of the Results**

1.  **Durbin--Watson Test**

    -   **Null Hypothesis (**$H_0$): No first-order autocorrelation ($\rho = 0$).

    -   **Alternative Hypothesis (**$H_1$): First-order autocorrelation exists ($\rho \neq 0$).

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *DW* $< 1.5$ (positive autocorrelation) or *DW* $> 2.5$ (negative autocorrelation).
        -   **Fail to reject** $H_0$ if *DW* $\approx 2$, suggesting no significant autocorrelation.

2.  **Breusch--Godfrey Test**

    -   **Null Hypothesis (**$H_0$): No autocorrelation up to lag $p$ (here, $p = 2$).

    -   **Alternative Hypothesis (**$H_1$): Autocorrelation exists at one or more lags.

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* $< 0.05$, indicating significant autocorrelation.
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$, suggesting no evidence of autocorrelation.

3.  **Ljung--Box Test**

    -   **Null Hypothesis (**$H_0$): No autocorrelation up to lag $h$ (here, $h = 10$).

    -   **Alternative Hypothesis (**$H_1$): Autocorrelation exists at one or more lags.

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* $< 0.05$, indicating significant autocorrelation.
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$, suggesting no evidence of autocorrelation.

4.  **Runs Test (Non-parametric)**

    -   **Null Hypothesis (**$H_0$): Residuals are randomly distributed (no autocorrelation).

    -   **Alternative Hypothesis (**$H_1$): Residuals exhibit non-random patterns (indicating autocorrelation).

    -   **Decision Rule:**

        -   **Reject** $H_0$ if *p-value* $< 0.05$, indicating non-randomness and potential autocorrelation.
        -   **Fail to reject** $H_0$ if *p-value* $\ge 0.05$, suggesting randomness in residuals.

------------------------------------------------------------------------

## Multicollinearity Diagnostics

**Multicollinearity** occurs when two or more independent variables in a regression model are **highly correlated**, leading to several issues:

-   **Unstable coefficient estimates:** Small changes in the data can cause large fluctuations in parameter estimates.
-   **Inflated standard errors:** Reduces the precision of estimated coefficients, making it difficult to determine the significance of predictors.
-   **Difficulty in assessing variable importance:** It becomes challenging to isolate the effect of individual predictors on the dependent variable.

Multicollinearity does not affect the overall fit of the model (e.g., $R^2$ remains high), but it distorts the reliability of individual coefficient estimates.

**Key Multicollinearity Diagnostics:**

1.  [Variance Inflation Factor](#sec-variance-inflation-factor)
2.  [Tolerance Statistic](#sec-tolerance-statistic)
3.  [Condition Index and Eigenvalue Decomposition](#sec-condition-index-and-eigenvalue-decomposition)
4.  [Pairwise Correlation Matrix](#sec-pairwise-correlation-matrix)
5.  [Determinant of the Correlation Matrix](#sec-determinant-of-the-correlation-matrix)

------------------------------------------------------------------------

### Variance Inflation Factor {#sec-variance-inflation-factor}

The **Variance Inflation Factor (VIF)** is the most commonly used diagnostic for detecting multicollinearity. It measures how much the **variance of an estimated regression coefficient** is inflated due to multicollinearity compared to when the predictors are uncorrelated.

------------------------------------------------------------------------

For each predictor $X_j$, the VIF is defined as:

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

Where:

-   $R_j^2$ is the **coefficient of determination** obtained by regressing $X_j$ on all other independent variables in the model.

------------------------------------------------------------------------

**Interpretation of VIF**

-   **VIF = 1:** No multicollinearity (perfect independence).
-   **1 \< VIF \< 5:** Moderate correlation, typically not problematic.
-   **VIF ≥ 5:** High correlation; consider investigating further.
-   **VIF ≥ 10:** Severe multicollinearity; corrective action is recommended.

------------------------------------------------------------------------

**Procedure**

1.  Regress each independent variable ($X_j$) on the remaining predictors.
2.  Compute $R_j^2$ for each regression.
3.  Calculate $\text{VIF}_j = 1 / (1 - R_j^2)$.
4.  Analyze VIF values to identify problematic predictors.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Easy to compute and interpret.
-   **Limitation:** Detects only **linear** relationships; may not capture complex multicollinearity patterns involving multiple variables simultaneously.

------------------------------------------------------------------------

### Tolerance Statistic {#sec-tolerance-statistic}

The **Tolerance Statistic** is the **reciprocal of the VIF** and measures the proportion of variance in an independent variable **not explained** by the other predictors.

$$
\text{Tolerance}_j = 1 - R_j^2
$$

Where $R_j^2$ is defined as in the VIF calculation.

------------------------------------------------------------------------

**Interpretation of Tolerance**

-   **Tolerance close to 1:** Low multicollinearity.
-   **Tolerance \< 0.2:** Potential multicollinearity problem.
-   **Tolerance \< 0.1:** Severe multicollinearity.

Since **low tolerance** implies **high VIF**, both metrics provide consistent information.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Provides an intuitive measure of how much variance is "free" from multicollinearity.
-   **Limitation:** Similar to VIF, focuses on linear dependencies.

------------------------------------------------------------------------

### Condition Index and Eigenvalue Decomposition {#sec-condition-index-and-eigenvalue-decomposition}

The **Condition Index** is a more advanced diagnostic that detects **multicollinearity involving multiple variables** simultaneously. It is based on the **eigenvalues** of the scaled independent variable matrix.

1.  **Compute the scaled design matrix** $X'X$, where $X$ is the matrix of independent variables.

2.  **Perform eigenvalue decomposition** to obtain the eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_k$.

3.  **Calculate the Condition Index:**

    $$
    \text{CI}_j = \sqrt{\frac{\lambda_{\max}}{\lambda_j}}
    $$

    Where:

    -   $\lambda_{\max}$ is the largest eigenvalue,
    -   $\lambda_j$ is the $j$-th eigenvalue.

------------------------------------------------------------------------

**Interpretation of Condition Index**

-   **CI \< 10:** No serious multicollinearity.
-   **10 ≤ CI \< 30:** Moderate to strong multicollinearity.
-   **CI ≥ 30:** Severe multicollinearity.

A **high condition index** indicates near-linear dependence among variables.

------------------------------------------------------------------------

**Variance Decomposition Proportions**

To identify which variables contribute to multicollinearity:

-   Compute the **Variance Decomposition Proportions (VDP)** for each coefficient across eigenvalues.

-   If **two or more variables** have high VDPs (e.g., \> 0.5) associated with a **high condition index**, this indicates severe multicollinearity.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Detects multicollinearity involving **multiple variables**, which VIF may miss.
-   **Limitation:** Requires matrix algebra knowledge; less intuitive than VIF or tolerance.

------------------------------------------------------------------------

### Pairwise Correlation Matrix {#sec-pairwise-correlation-matrix}

A **Pairwise Correlation Matrix** provides a simple diagnostic by computing the correlation coefficients between each pair of independent variables.

------------------------------------------------------------------------

For variables $X_i$ and $X_j$, the **correlation coefficient** is:

$$
\rho_{ij} = \frac{\text{Cov}(X_i, X_j)}{\sigma_{X_i} \sigma_{X_j}}
$$

Where:

-   $\text{Cov}(X_i, X_j)$ is the covariance,

-   $\sigma_{X_i}$ and $\sigma_{X_j}$ are standard deviations.

------------------------------------------------------------------------

**Interpretation of Correlation Coefficients**

-   $|\rho| < 0.5$: Weak correlation (unlikely to cause multicollinearity).
-   $0.5 \leq |\rho| < 0.8$: Moderate correlation; monitor carefully.
-   $|\rho| ≥ 0.8$: Strong correlation; potential multicollinearity issue.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Quick and easy to compute; useful for initial screening.
-   **Limitation:** Detects **only pairwise relationships**; may miss multicollinearity involving more than two variables.

------------------------------------------------------------------------

### Determinant of the Correlation Matrix {#sec-determinant-of-the-correlation-matrix}

The **Determinant of the Correlation Matrix** provides a global measure of multicollinearity. A **small determinant** indicates high multicollinearity.

1.  Form the **correlation matrix** $R$ of the independent variables.

2.  Compute the **determinant**:

    $$
    \det(R)
    $$

------------------------------------------------------------------------

**Interpretation**

-   $\det(R) \approx 1$: No multicollinearity (perfect independence).
-   $\det(R) \approx 0$: Severe multicollinearity.

A determinant **close to zero** suggests that the correlation matrix is nearly singular, indicating strong multicollinearity.

------------------------------------------------------------------------

**Advantages and Limitations**

-   **Advantage:** Provides a single summary statistic for overall multicollinearity.
-   **Limitation:** Does not indicate **which** variables are causing the problem.

------------------------------------------------------------------------

### Summary of Multicollinearity Diagnostics

| **Diagnostic**                                                                  | **Type**          | **Key Metric**                            | **Threshold for Concern**                       | **When to Use**                                    |
|---------------------------------------------------------------------------------|-------------------|-------------------------------------------|-------------------------------------------------|----------------------------------------------------|
| [Variance Inflation Factor](#sec-variance-inflation-factor)                     | Parametric        | $VIF = \frac{1}{1 - R_j^2}$               | $VIF \geq 5$ (moderate), $VIF \geq 10$ (severe) | General-purpose detection                          |
| [Tolerance Statistic](#sec-tolerance-statistic)                                 | Parametric        | $1 - R_j^2$                               | $< 0.2$ (moderate), $< 0.1$ (severe)            | Reciprocal of VIF for variance interpretation      |
| [Condition Index](#sec-condition-index-and-eigenvalue-decomposition)            | Eigenvalue-based  | $\sqrt{\frac{\lambda_{\max}}{\lambda_j}}$ | $> 10$ (moderate), $> 30$ (severe)              | Detects multicollinearity among multiple variables |
| [Pairwise Correlation Matrix](#sec-pairwise-correlation-matrix)                 | Correlation-based | Pearson correlation ($\rho$)              | $|\rho| \geq 0.8$                               | Initial screening for bivariate correlations       |
| [Determinant of Correlation Matrix](#sec-determinant-of-the-correlation-matrix) | Global diagnostic | $\det(R)$                                 | $\approx 0$ indicates severe multicollinearity  | Overall assessment of multicollinearity            |

------------------------------------------------------------------------

### Addressing Multicollinearity

If multicollinearity is detected, consider the following solutions:

1.  **Remove or combine correlated variables:** Drop one of the correlated predictors or create an index/aggregate.
2.  **Principal Component Analysis:** Reduce dimensionality by transforming correlated variables into uncorrelated components.
3.  [Ridge Regression] **(L2 regularization):** Introduces a penalty term to stabilize coefficient estimates in the presence of multicollinearity.
4.  **Centering variables:** Mean-centering can help reduce multicollinearity, especially in interaction terms.

------------------------------------------------------------------------

Multicollinearity can significantly distort regression estimates, leading to misleading interpretations. While [VIF](#sec-variance-inflation-factor) and [Tolerance](#sec-tolerance-statistic) are commonly used diagnostics, advanced techniques like the [Condition Index](#sec-condition-index-and-eigenvalue-decomposition) and Eigenvalue Decomposition provide deeper insights, especially when dealing with complex datasets.


```r
# Install and load necessary libraries
# install.packages("car")        # For VIF calculation
# install.packages("corpcor")    # For determinant of correlation matrix


library(car)
library(corpcor)

# Simulated dataset with multicollinearity
set.seed(123)
n <- 100
x1 <- rnorm(n, mean = 50, sd = 10)
x2 <- 0.8 * x1 + rnorm(n, sd = 2)   # Highly correlated with x1
x3 <- rnorm(n, mean = 30, sd = 5)
y <- 5 + 0.4 * x1 - 0.3 * x2 + 0.2 * x3 + rnorm(n)

# Original regression model
model <- lm(y ~ x1 + x2 + x3)

# ----------------------------------------------------------------------
# 1. Variance Inflation Factor (VIF)
# ----------------------------------------------------------------------
# Null Hypothesis: No multicollinearity (VIF = 1)
vif_values <- vif(model)
print(vif_values)
#>        x1        x2        x3 
#> 14.969143 14.929013  1.017576

# ----------------------------------------------------------------------
# 2. Tolerance Statistic (Reciprocal of VIF)
# ----------------------------------------------------------------------
tolerance_values <- 1 / vif_values
print(tolerance_values)
#>         x1         x2         x3 
#> 0.06680409 0.06698366 0.98272742

# ----------------------------------------------------------------------
# 3. Condition Index and Eigenvalue Decomposition
# ----------------------------------------------------------------------
# Scaling the independent variables
X <- model.matrix(model)[,-1]  # Removing intercept
eigen_decomp <-
    eigen(cor(X))   # Eigenvalue decomposition of the correlation matrix

# Condition Index
condition_index <-
    sqrt(max(eigen_decomp$values) / eigen_decomp$values)
print(condition_index)
#> [1] 1.000000 1.435255 7.659566

# Variance Decomposition Proportions (VDP)
# Proportions calculated based on the squared coefficients
loadings <- eigen_decomp$vectors
vdp <- apply(loadings ^ 2, 2, function(x)
    x / sum(x))
print(vdp)
#>            [,1]       [,2]         [,3]
#> [1,] 0.48567837 0.01363318 5.006885e-01
#> [2,] 0.48436754 0.01638399 4.992485e-01
#> [3,] 0.02995409 0.96998283 6.307954e-05

# ----------------------------------------------------------------------
# 4. Pairwise Correlation Matrix
# ----------------------------------------------------------------------
correlation_matrix <- cor(X)
print(correlation_matrix)
#>           x1         x2         x3
#> x1  1.000000  0.9659070 -0.1291760
#> x2  0.965907  1.0000000 -0.1185042
#> x3 -0.129176 -0.1185042  1.0000000

# ----------------------------------------------------------------------
# 5. Determinant of the Correlation Matrix
# ----------------------------------------------------------------------
determinant_corr_matrix <- det(correlation_matrix)
cat("Determinant of the Correlation Matrix:",
    determinant_corr_matrix,
    "\n")
#> Determinant of the Correlation Matrix: 0.06586594
```

**Interpretation of the Results**

1.  **Variance Inflation Factor (VIF)**

    -   **Formula:** $\text{VIF}_j = \frac{1}{1 - R_j^2}$

    -   **Decision Rule:**

        -   **VIF** $\approx 1$: No multicollinearity.
        -   $1 < \text{VIF} < 5$: Moderate correlation, usually acceptable.
        -   $\text{VIF} \ge 5$: High correlation; investigate further.
        -   $\text{VIF} \ge 10$: Severe multicollinearity; corrective action recommended.

2.  **Tolerance Statistic**

    -   **Formula:** $\text{Tolerance}_j = 1 - R_j^2 = \frac{1}{\text{VIF}_j}$

    -   **Decision Rule:**

        -   **Tolerance** $> 0.2$: Low risk of multicollinearity.
        -   **Tolerance** $< 0.2$: Possible multicollinearity problem.
        -   **Tolerance** $< 0.1$: Severe multicollinearity.

3.  **Condition Index and Eigenvalue Decomposition**

    -   **Formula:** $\text{CI}_j = \sqrt{\frac{\lambda_{\max}}{\lambda_j}}$

    -   **Decision Rule:**

        -   **CI** $< 10$: No significant multicollinearity.
        -   $10 \le \text{CI} < 30$: Moderate to strong multicollinearity.
        -   **CI** $\ge 30$: Severe multicollinearity.

    -   **Variance Decomposition Proportions (VDP):**

        -   High VDP ($> 0.5$) associated with high CI indicates problematic variables.

4.  **Pairwise Correlation Matrix**

    -   **Decision Rule:**
        -   $|\rho| < 0.5$: Weak correlation.
        -   $0.5 \le |\rho| < 0.8$: Moderate correlation; monitor.
        -   $|\rho| \ge 0.8$: Strong correlation; potential multicollinearity issue.

5.  **Determinant of the Correlation Matrix**

    -   **Decision Rule:**
        -   $\det(R) \approx 1$: No multicollinearity.
        -   $\det(R) \approx 0$: Severe multicollinearity (near-singular matrix).

------------------------------------------------------------------------

Model specification tests are essential for diagnosing and validating econometric models. They ensure that the model assumptions hold true, thereby improving the accuracy and reliability of the estimations. By systematically applying these tests, researchers can identify issues related to nested and non-nested models, heteroskedasticity, functional form, endogeneity, autocorrelation, and multicollinearity, leading to more robust and credible econometric analyses.
