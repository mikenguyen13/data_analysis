# Model Specification

Test whether underlying assumptions hold true  

 * [Nested Model] (A1/A3)
 * [Non-Nested Model] (A1/A3)
 * [Heteroskedasticity] (A4)


## Nested Model

$$
\begin{aligned}
y = \beta_0 + x_1\beta_1 + x_2\beta-2 + x_3\beta_3 + \epsilon && \text{unrestricted model} \\
y = \beta_0 + x_1\beta_1 + \epsilon && \text{restricted model}
\end{aligned}
$$

Unrestricted model is always longer than the restricted model  
The restricted model is "nested" within the unrestricted model  
To determine which variables should be included or exclude, we could use the same [Wald Test]

**Adjusted $R^2$**

 * $R^2$ will always increase with more variables included
 * Adjusted $R^2$ tries to correct by penalizing inclusion of unnecessary variables.

$$
{R}^2 = 1 - \frac{SSR/n}{SST/n} \\
{R}^2_{adj}= 1- \frac{SSR/(n-k)}{SST/(n-1)} = 1 - \frac{(n-1)(1-R^2)}{(n-k)}
$$

 * ${R}^2_{adj}$ increases if and only if the t-statistic on the additional variable is greater than 1 in absolute value. 
 * ${R}^2_{adj}$ is valid in models where there is no heteroskedasticity 
 * there fore it **should not** be used in determining which variables should be included in the model (the t or F-tests are more appropriate)



### Chow test
Should we run two different regressions for two groups?




## Non-Nested Model
compare models with different non-nested specifications


### Davidson-Mackinnon test
#### Independent Variable
should the independent variables be logged?
decide between non-nested alternatives 

$$
\begin{aligned}
y =  \beta_0 + x_1\beta_1 + x_2\beta_2 + \epsilon && \text{(level eq)} \\
y =  \beta_0 + ln(x_1)\beta_1 + x_2\beta_2 + \epsilon && \text{(log eq)}
\end{aligned}
$$

 1. Obtain predict outcome when estimating the model in log equation $\check{y}$ and then estimate the following auxiliary equation, 
 
$$
y = \beta_0 + x_1\beta_1 + x_2\beta_2 + \check{y}\gamma + error
$$

and evaluate the t-statistic for the null hypothesis $H_0: \gamma = 0$

 2. Obtain predict outcome when estimating the model in the level equation $\hat{y}$, then estimate the following auxiliary equation,

$$
y = \beta_0 + ln(x_1)\beta_1 + x_2\beta_2 + \check{y}\gamma + error
$$
and evaluate the t-statistic for the null hypothesis $H_0: \gamma = 0$

 * If you reject the null in the (1) step but fail to reject the null in the second step, then the log equation is preferred.
 * If fail to reject the null in the (1) step but reject the null in the (2) step then, level equation is preferred.
 * If reject in both steps, then you have statistical evidence that neither model should be used and should re-evaluate the functional form of your model. 
 * If fail to reject in both steps, you do not have sufficient evidence to prefer one model over the other. You can compare the $R^2_{adj}$ to choose between the two models.

$$
y = \beta_0 + ln(x)\beta_1 + \epsilon \\
y = \beta_0 + x(\beta_1) + x^2\beta_2 + \epsilon
$$
 * Compare which better fits the data 
 * Compare standard $R^2$ is unfair because the second model is less parsimonious (more parameters to estimate)
 * The $R_{adj}^2$ will penalize the second model for being less parsimonious 
    + Only valid when there is no heteroskedasticity ([A4][A4 Homoskedasticity] holds)
 * Should only compare after a [Davidson-Mackinnon test]

#### Dependent Variable
$$
\begin{aligned}
y = \beta_0 + x_1\beta_1 + \epsilon && \text{level eq} \\
ln(y) = \beta_0 + x_1\beta_1 + \epsilon && \text{log eq} \\
\end{aligned}
$$

 * In the level model, regardless of how big y is, x has a constant effect (i.e., one unit change in $x_1$ results in a $\beta_1$ unit change in y)
 * In the log model, the larger in y is, the effect of x is stronger (i.e., one unit change in $x_1$ could increase y from 1 to $1+\beta_1$ or from 100 to 100+100x$\beta_1$)
 * Cannot compare $R^2$ or $R^2_{adj}$ because the outcomes are complement different, the scaling is different (SST is different)

<br>

We need to "un-transform" the $ln(y)$ back to the same scale as y and then compare,  

 1. Estimate the model in the log equation to obtain the predicted outcome $\hat{ln(y)}$
 2. "Un-transform" the predicted outcome 

$$
\hat{m} = exp(\hat{ln(y)})
$$
 3. Estimate the following model (without an intercept)

$$
y = \alpha\hat{m} + error
$$
and obtain predicted outcome $\hat{y}$

 4. Then take the square of the correlation between $\hat{y}$ and y as a scaled version of the $R^2$ from the log model that can now compare with the usual $R^2$ in the level model.

## Heteroskedasticity
 * Using roust standard errors are always valid 
 * If there is significant evidence of heteroskedasticity implying [A4][A4 Homoskedasticity] does not hold  
    + [Gauss-Markov Theorem] no longer holds, OLS is not BLUE. 
    + Should consider using a better linear unbiased estimator ([Weighted Least Squares] or [Generalized Least Squares])

### Breusch-Pagan test
[A4][A4 Homoskedasticity] implies
$$
E(\epsilon_i^2|\mathbf{x_i})=\sigma^2
$$

$$
\epsilon_i^2 = \gamma_0 + x_{i1}\gamma_1 + ... + x_{ik -1}\gamma_{k-1} + error
$$
and determining whether or not $\mathbf{x}_i$ has any predictive value  

 * if $\mathbf{x}_i$ has predictive value, then the variance changes over the levels of $\mathbf{x}_i$ which is evidence of heteroskedasticity
 * if $\mathbf{x}_i$ does not have predictive value, the variance is constant for all levels of $\mathbf{x}_i$

The [Breusch-Pagan test] for heteroskedasticity would compute the F-test of total significance for the following model

$$
e_i^2 = \gamma_0 + x_{i1}\gamma_1 + ... + x_{ik -1}\gamma_{k-1} + error
$$
A low p-value means we reject the null of homoskedasticity

However, [Breusch-Pagan test] cannot detect heteroskedasticity in non-linear form

### White test
test heteroskedasticity would allow for a non-linear relationship by computing the F-test of total significance for the following model (assume there are three independent random variables) 

$$
e_i^2=\gamma_0 + x_i \gamma_1 + x_{i2}\gamma_2 + x_{i3}\gamma_3 + x_{i1}^2\gamma_4 + x_{i2}^2\gamma_5 + x_{i3}^2\gamma_6 + (x_{i1} \times x_{i2})\gamma_7 + (x_{i1} \times x_{i3})\gamma_8 + (x_{i2} \times x_{i3})\gamma_9 + error
$$
A low p-value means we reject the null of homoskedasticity

Equivalently, we can compute [LM][Lagrange Multiplier (Score)] as $LM = nR^2_{e^2}$
where the $R^2_{e^2}$ come from the regression with the squared residual as the outcome  

 * The [LM][Lagrange Multiplier (Score)] statistic has a [$\chi_k^2$][Chi-squared] distribution 




