# Multivariate Methods

$y_1,...,y_p$ are possibly correlated random variables with means $\mu_1,...,\mu_p$

$$
\mathbf{y} = 
\left(
\begin{array}
{c}
y_1 \\
. \\
y_p \\
\end{array}
\right)
$$

$$
E(\mathbf{y}) = 
\left(
\begin{array}
{c}
\mu_1 \\
. \\
\mu_p \\
\end{array}
\right)
$$

Let $\sigma_{ij} = cov(y_i, y_j)$ for $i,j = 1,…,p$

$$
\mathbf{\Sigma} = (\sigma_{ij}) = 
\left(
\begin{array}
{cccc}
\sigma_{11} & \sigma_{22} & ... &  \sigma_{1p} \\
\sigma_{21} & \sigma_{22} & ... & \sigma_{2p} \\
. & . & . & . \\
\sigma_{p1} & \sigma_{p2} & ... & \sigma_{pp}
\end{array}
\right)
$$

where $\mathbf{\Sigma}$ (symmetric) is the variance-covariance or dispersion matrix

Let $\mathbf{u}_{p \times 1}$ and $\mathbf{v}_{q \times 1}$ be random vectors with means $\mu_u$ and $\mu_v$ . Then

$$
\mathbf{\Sigma}_{uv} = cov(\mathbf{u,v}) = E[(\mathbf{u} - \mu_u)(\mathbf{v} - \mu_v)']
$$

in which $\mathbf{\Sigma}_{uv} \neq \mathbf{\Sigma}_{vu}$ and $\mathbf{\Sigma}_{uv} = \mathbf{\Sigma}_{vu}'$

\
**Properties of Covariance Matrices**

1.  Symmetric $\mathbf{\Sigma}' = \mathbf{\Sigma}$
2.  Non-negative definite $\mathbf{a'\Sigma a} \ge 0$ for any $\mathbf{a} \in R^p$, which is equivalent to eigenvalues of $\mathbf{\Sigma}$, $\lambda_1 \ge \lambda_2 \ge ... \ge \lambda_p \ge 0$
3.  $|\mathbf{\Sigma}| = \lambda_1 \lambda_2 ... \lambda_p \ge 0$ (**generalized variance**) (the bigger this number is, the more variation there is
4.  $trace(\mathbf{\Sigma}) = tr(\mathbf{\Sigma}) = \lambda_1 + ... + \lambda_p = \sigma_{11} + ... + \sigma_{pp} =$ sum of variance (**total variance**)

Note:

-   $\mathbf{\Sigma}$ is typically required to be positive definite, which means all eigenvalues are positive, and $\mathbf{\Sigma}$ has an inverse $\mathbf{\Sigma}^{-1}$ such that $\mathbf{\Sigma}^{-1}\mathbf{\Sigma} = \mathbf{I}_{p \times p} = \mathbf{\Sigma \Sigma}^{-1}$

<br>

**Correlation Matrices**

$$
\rho_{ij} = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii} \sigma_{jj}}}
$$

$$
\mathbf{R} = 
\left(
\begin{array}
{cccc}
\rho_{11} & \rho_{12} & ... & \rho_{1p} \\
\rho_{21} & \rho_{22} & ... & \rho_{2p} \\
. & . & . &. \\
\rho_{p1} & \rho_{p2} & ... & \rho_{pp} \\
\end{array}
\right)
$$

where $\rho_{ij}$ is the correlation, and $\rho_{ii} = 1$ for all i

Alternatively,

$$
\mathbf{R} = [diag(\mathbf{\Sigma})]^{-1/2}\mathbf{\Sigma}[diag(\mathbf{\Sigma})]^{-1/2}
$$

where $diag(\mathbf{\Sigma})$ is the matrix which has the $\sigma_{ii}$'s on the diagonal and 0's elsewhere

and $\mathbf{A}^{1/2}$ (the square root of a symmetric matrix) is a symmetric matrix such as $\mathbf{A} = \mathbf{A}^{1/2}\mathbf{A}^{1/2}$

**Equalities**

Let

-   $\mathbf{x}$ and $\mathbf{y}$ be random vectors with means $\mu_x$ and $\mu_y$ and variance -variance matrices $\mathbf{\Sigma}_x$ and $\mathbf{\Sigma}_y$.

-   $\mathbf{A}$ and $\mathbf{B}$ be matrices of constants and $\mathbf{c}$ and $\mathbf{d}$ be vectors of constants

Then

-   $E(\mathbf{Ay + c} ) = \mathbf{A} \mu_y + c$

-   $var(\mathbf{Ay + c}) = \mathbf{A} var(\mathbf{y})\mathbf{A}' = \mathbf{A \Sigma_y A}'$

-   $cov(\mathbf{Ay + c, By+ d}) = \mathbf{A\Sigma_y B}'$

-   $E(\mathbf{Ay + Bx + c}) = \mathbf{A \mu_y + B \mu_x + c}$

-   $var(\mathbf{Ay + Bx + c}) = \mathbf{A \Sigma_y A' + B \Sigma_x B' + A \Sigma_{yx}B' + B\Sigma'_{yx}A'}$

**Multivariate Normal Distribution**

Let $\mathbf{y}$ be a multivariate normal (MVN) random variable with mean $\mu$ and variance $\mathbf{\Sigma}$. Then the density of $\mathbf{y}$ is

$$
f(\mathbf{y}) = \frac{1}{(2\pi)^{p/2}|\mathbf{\Sigma}|^{1/2}} \exp(-\frac{1}{2} \mathbf{(y-\mu)'\Sigma^{-1}(y-\mu)} )
$$

$\mathbf{y} \sim N_p(\mu, \mathbf{\Sigma})$

### **Properties of MVN**

-   Let $\mathbf{A}_{r \times p}$ be a fixed matrix. Then $\mathbf{Ay} \sim N_r (\mathbf{A \mu, A \Sigma A'})$ . $r \le p$ and all rows of $\mathbf{A}$ must be linearly independent to guarantee that $\mathbf{A \Sigma A}'$ is non-singular.

-   Let $\mathbf{G}$ be a matrix such that $\mathbf{\Sigma}^{-1} = \mathbf{GG}'$. Then $\mathbf{G'y} \sim N_p(\mathbf{G' \mu, I})$ and $\mathbf{G'(y-\mu)} \sim N_p (0,\mathbf{I})$

-   Any fixed linear combination of $y_1,...,y_p$ (say $\mathbf{c'y}$) follows $\mathbf{c'y} \sim N_1 (\mathbf{c' \mu, c' \Sigma c})$

-   Define a partition, $[\mathbf{y}'_1,\mathbf{y}_2']'$ where $\mathbf{y}$\_1 is $p_1 \times 1$ ,$\mathbf{y}_2$ is $p_2 \times 1$, $p_1 + p_2 = p$ and $p_1,p_2 \ge 1$ Then

$$
\left(
\begin{array}
{c}
\mathbf{y}_1 \\
\mathbf{y}_2 \\
\end{array}
\right)
\sim
N
\left(
\left(
\begin{array}
{c}
\mu_1 \\
\mu_2 \\
\end{array}
\right),
\left(
\begin{array}
{cc}
\mathbf{\Sigma}_{11} & \mathbf{\Sigma}_{12} \\
\mathbf{\Sigma}_{21} & \mathbf{\Sigma}_{22}\\
\end{array}
\right)
\right)
$$

-   The marginal distributions of $\mathbf{y}_1$ and $\mathbf{y}_2$ are $\mathbf{y}_1 \sim N_{p1}(\mathbf{\mu_1, \Sigma_{11}})$ and $\mathbf{y}_2 \sim N_{p2}(\mathbf{\mu_2, \Sigma_{22}})$

-   Individual components $y_1,...,y_p$ are all normally distributed $y_i \sim N_1(\mu_i, \sigma_{ii})$

-   The conditional distribution of $\mathbf{y}_1$ and $\mathbf{y}_2$ is normal

    -   $\mathbf{y}_1 | \mathbf{y}_2 \sim N_{p1}(\mathbf{\mu_1 + \Sigma_{12} \Sigma_{22}^{-1}(y_2 - \mu_2),\Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \sigma_{21}})$

        -   In this formula, we see if we know (have info about) $\mathbf{y}_2$, we can re-weight $\mathbf{y}_1$ 's mean, and the variance is reduced because we know more about $\mathbf{y}_1$ because we know $\mathbf{y}_2$

    -   which is analogous to $\mathbf{y}_2 | \mathbf{y}_1$. And $\mathbf{y}_1$ and $\mathbf{y}_2$ are independently distrusted only if $\mathbf{\Sigma}_{12} = 0$

-   If $\mathbf{y} \sim N(\mathbf{\mu, \Sigma})$ and $\mathbf{\Sigma}$ is positive definite, then $\mathbf{(y-\mu)' \Sigma^{-1} (y - \mu)} \sim \chi^2_{(p)}$

-   If $\mathbf{y}_i$ are independent $N_p (\mathbf{\mu}_i , \mathbf{\Sigma}_i)$ random variables, then for fixed matrices $\mathbf{A}_{i(m \times p)}$, $\sum_{i=1}^k \mathbf{A}_i \mathbf{y}_i \sim N_m (\sum_{i=1}^{k} \mathbf{A}_i \mathbf{\mu}_i, \sum_{i=1}^k \mathbf{A}_i \mathbf{\Sigma}_i \mathbf{A}_i)$

**Multiple Regression**

$$
\left(
\begin{array}
{c}
Y \\
\mathbf{x}
\end{array}
\right)
\sim 
N_{p+1}
\left(
\left[
\begin{array}
{c}
\mu_y \\
\mathbf{\mu}_x
\end{array}
\right]
,
\left[
\begin{array}
{cc}
\sigma^2_Y & \mathbf{\Sigma}_{yx} \\
\mathbf{\Sigma}_{yx} & \mathbf{\Sigma}_{xx}
\end{array}
\right]
\right)
$$

The conditional distribution of Y given x follows a univariate normal distribution with

$$
\begin{aligned}
E(Y| \mathbf{x}) &= \mu_y + \mathbf{\Sigma}_{yx} \Sigma_{xx}^{-1} (\mathbf{x}- \mu_x) \\
&= \mu_y - \Sigma_{yx} \Sigma_{xx}^{-1}\mu_x + \Sigma_{yx} \Sigma_{xx}^{-1}\mathbf{x} \\
&= \beta_0 + \mathbf{\beta'x}
\end{aligned} 
$$

where $\beta = (\beta_1,...,\beta_p)' = \mathbf{\Sigma}_{xx}^{-1} \mathbf{\Sigma}_{yx}'$ (e.g., analogous to $\mathbf{(x'x)^{-1}x'y}$ but not the same if we consider $Y_i$ and $\mathbf{x}_i$, $i = 1,..,n$ and use the empirical covariance formula: $var(Y|\mathbf{x}) = \sigma^2_Y - \mathbf{\Sigma_{yx}\Sigma^{-1}_{xx} \Sigma'_{yx}}$)

<br>

**Samples from Multivariate Normal Populations**

A random sample of size n, $\mathbf{y}_1,.., \mathbf{y}_n$ from $N_p (\mathbf{\mu}, \mathbf{\Sigma})$. Then

-   Since $\mathbf{y}_1,..., \mathbf{y}_n$ are iid, their sample mean, $\bar{\mathbf{y}} = \sum_{i=1}^n \mathbf{y}_i/n \sim N_p (\mathbf{\mu}, \mathbf{\Sigma}/n)$. that is, $\bar{\mathbf{y}}$ is an unbiased estimator of $\mathbf{\mu}$

-   The $p \times p$ sample variance-covariance matrix, $\mathbf{S}$ is $\mathbf{S} = \frac{1}{n-1}\sum_{i=1}^n (\mathbf{y}_i - \bar{\mathbf{y}})(\mathbf{y}_i - \bar{\mathbf{y}})' = \frac{1}{n-1} (\sum_{i=1}^n \mathbf{y}_i \mathbf{y}_i' - n \bar{\mathbf{y}}\bar{\mathbf{y}}')$

    -   where $\mathbf{S}$ is symmetric, unbiased estimator of $\mathbf{\Sigma}$ and has $p(p+1)/2$ random variables.

-   $(n-1)\mathbf{S} \sim W_p (n-1, \mathbf{\Sigma})$ is a Wishart distribution with n-1 degrees of freedom and expectation $(n-1) \mathbf{\Sigma}$. The Wishart distribution is a multivariate extension of the Chi-squared distribution.

-   $\bar{\mathbf{y}}$ and $\mathbf{S}$ are independent

-   $\bar{\mathbf{y}}$ and $\mathbf{S}$ are sufficient statistics. (All of the info in the data about $\mathbf{\mu}$ and $\mathbf{\Sigma}$ is contained in $\bar{\mathbf{y}}$ and $\mathbf{S}$ , regardless of sample size).

<br>

**Large Sample Properties**

$\mathbf{y}_1,..., \mathbf{y}_n$ are a random sample from some population with mean $\mathbf{\mu}$ and variance-covariance matrix $\mathbf{\Sigma}$

-   $\bar{\mathbf{y}}$ is a consistent estimator for $\mu$

-   $\mathbf{S}$ is a consistent estimator for $\mathbf{\Sigma}$

-   **Multivariate Central Limit Theorem**: Similar to the univariate case, $\sqrt{n}(\bar{\mathbf{y}} - \mu) \dot{\sim} N_p (\mathbf{0,\Sigma})$ where n is large relative to p ($n \ge 25p$), which is equivalent to $\bar{\mathbf{y}} \dot{\sim} N_p (\mu, \mathbf{\Sigma}/n)$

-   **Wald's Theorem**: $n(\bar{\mathbf{y}} - \mu)' \mathbf{S}^{-1} (\bar{\mathbf{y}} - \mu)$ when n is large relative to p.

<br>

Maximum Likelihood Estimation for MVN

Suppose iid $\mathbf{y}_1 ,... \mathbf{y}_n \sim N_p (\mu, \mathbf{\Sigma})$, the likelihood function for the data is

$$
\begin{aligned}
L(\mu, \mathbf{\Sigma}) &= \prod_{j=1}^n (\frac{1}{(2\pi)^{p/2}|\mathbf{\Sigma}|^{1/2}} \exp(-\frac{1}{2}(\mathbf{y}_j -\mu)'\mathbf{\Sigma}^{-1})(\mathbf{y}_j -\mu)) \\
&= \frac{1}{(2\pi)^{np/2}|\mathbf{\Sigma}|^{n/2}} \exp(-\frac{1}{2} \sum_{j=1}^n(\mathbf{y}_j -\mu)'\mathbf{\Sigma}^{-1})(\mathbf{y}_j -\mu)
\end{aligned}
$$

Then, the MLEs are

$$
\hat{\mu} = \bar{\mathbf{y}}
$$

$$
\hat{\mathbf{\Sigma}} = \frac{n-1}{n} \mathbf{S}
$$

using derivatives of the log of the likelihood function with respect to $\mu$ and $\mathbf{\Sigma}$

<br>

**Properties of MLEs**

-   Invariance: If $\hat{\theta}$ is the MLE of $\theta$, then the MLE of $h(\theta)$ is $h(\hat{\theta})$ for any function h(.)

-   Consistency: MLEs are consistent estimators, but they are usually biased

-   Efficiency: MLEs are efficient estimators (no other estimator has a smaller variance for large samples)

-   Asymptotic normality: Suppose that $\hat{\theta}_n$ is the MLE for $\theta$ based upon n independent observations. Then $\hat{\theta}_n \dot{\sim} N(\theta, \mathbf{H}^{-1})$

    -   $\mathbf{H}$ is the Fisher Information Matrix, which contains the expected values of the second partial derivatives fo the log-likelihood function. the (i,j)th element of $\mathbf{H}$ is $-E(\frac{\partial^2 l(\mathbf{\theta})}{\partial \theta_i \partial \theta_j})$

    -   we can estimate $\mathbf{H}$ by finding the form determined above, and evaluate it at $\theta = \hat{\theta}_n$

-   Likelihood ratio testing: for some null hypothesis, $H_0$ we can form a likelihood ratio test

    -   The statistic is: $\Lambda = \frac{\max_{H_0}l(\mathbf{\mu}, \mathbf{\Sigma|Y})}{\max l(\mu, \mathbf{\Sigma | Y})}$

    -   For large n, $-2 \log \Lambda \sim \chi^2_{(v)}$ where v is the number of parameters in the unrestricted space minus the number of parameters under $H_0$

<br>

**Test of Multivariate Normality**

-   Check univariate normality for each trait (X) separately

    -   Can check [Normality Assessment]

    -   The good thing is that if any of the univariate trait is not normal, then the joint distribution is not normal (see again [Properties of MVN]). If a joint multivariate distribution is normal, then the marginal distribution has to be normal.

    -   However, marginal normality of all traits does not imply joint MVN

    -   Easily rule out multivariate normality, but not easy to prove it

-   Mardia's tests for multivariate normality

    -   Multivariate skewness is$$
        \beta_{1,p} = E[(\mathbf{y}- \mathbf{\mu})' \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})]^3
        $$

    -   where $\mathbf{x}$ and $\mathbf{y}$ are independent, but have the same distribution (note: $\beta$ here is not regression coefficient)

    -   Multivariate kurtosis is defined as

    -   $$
        \beta_{2,p} - E[(\mathbf{y}- \mathbf{\mu})' \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})]^2
        $$

    -   For the MVN distribution, we have $\beta_{1,p} = 0$ and $\beta_{2,p} = p(p+2)$

    -   For a sample of size n, we can estimate

        $$
        \hat{\beta}_{1,p} = \frac{1}{n^2}\sum_{i=1}^n \sum_{j=1}^n g^2_{ij}
        $$

        $$
        \hat{\beta}_{2,p} = \frac{1}{n} \sum_{i=1}^n g^2_{ii}
        $$

        -   where $g_{ij} = (\mathbf{y}_i - \bar{\mathbf{y}})' \mathbf{S}^{-1} (\mathbf{y}_j - \bar{\mathbf{y}})$. Note: $g_{ii} = d^2_i$ where $d^2_i$ is the Mahalanobis distance

    -   [@MARDIA_1970] shows for large n

        $$
        \kappa_1 = \frac{n \hat{\beta}_{1,p}}{6} \dot{\sim} \chi^2_{p(p+1)(p+2)/6}
        $$

        $$
        \kappa_2 = \frac{\hat{\beta}_{2,p} - p(p+2)}{\sqrt{8p(p+2)/n}} \sim N(0,1)
        $$

        -   Hence, we can use $\kappa_1$ and $\kappa_2$ to test the null hypothesis of MVN.

        -   When the data are non-normal, normal theory tests on the mean are sensitive to $\beta_{1,p}$ , while tests on the covariance are sensitive to $\beta_{2,p}$

-   Chi-square Q-Q plot

    -   Let $\mathbf{y}_i, i = 1,...,n$ be a random sample sample from $N_p(\mathbf{\mu}, \mathbf{\Sigma})$

    -   Then $\mathbf{z}_i = \mathbf{\Sigma}^{-1/2}(\mathbf{y}_i - \mathbf{\mu}), i = 1,...,n$ are iid $N_p (\mathbf{0}, \mathbf{I})$. Thus, $d_i^2 = \mathbf{z}_i' \mathbf{z}_i \sim \chi^2_p , i = 1,...,n$

    -   plot the ordered $d_i^2$ values against the qualities of the $\chi^2_p$ distribution. When normality holds, the plot should approximately resemble a straight lien passing through the origin at a 45 degree

    -   it requires large sample size (i.e., sensitive to sample size). Even if we generate data from a MVN, the tail of the Chi-square Q-Q plot can still be out of line.

-   If the data are not normal, we can

    -   ignore it

    -   use nonparametric methods

    -   use models based upon an approximate distirubiton (e.g., GLMM)

    -   try performing a transformation

<br>

### Mean Vector Inference

In the univariate normal distribution, we test $H_0: \mu =\mu_0$ by using

$$
T = \frac{\bar{y}- \mu_0}{s/\sqrt{n}} \sim t_{n-1}
$$

under the null hypothesis. And reject the null if $|T|$ is large relative to $t_{(1-\alpha/2,n-1)}$ because it means that seeing a value as large as what we observed is rare if the null is true

Equivalently,

$$
T^2 = \frac{(\bar{y}- \mu_0)^2}{s^2/n} = n(\bar{y}- \mu_0)(s^2)^{-1}(\bar{y}- \mu_0) \sim f_{(1,n-1)}
$$

**Natural Multivariate Generalization**

$$
H_0: \mathbf{\mu} = \mathbf{\mu}_0 \\
H_a: \mathbf{\mu} \neq \mathbf{\mu}_0
$$

Define **Hotelling's** $T^2$ by

$$
T^2 = n(\bar{\mathbf{y}} - \mathbf{\mu}_0)'\mathbf{S}^{-1}(\bar{\mathbf{y}} - \mathbf{\mu}_0)
$$

which can be viewed as a generalized distance between $\bar{\mathbf{y}}$ and $\mathbf{\mu}_0$

Under the assumption of normality,

$$
F = \frac{n-p}{(n-1)p} T^2 \sim f_{(p,n-p)}
$$

and reject the null hypothesis when $F > f_{(1-\alpha, p, n-p)}$

-   The $T^2$ test is invariant to changes in measurement units.

    -   If $\mathbf{z = Cy + d}$ where $\mathbf{C}$ and $\mathbf{d}$ do not depend on $\mathbf{y}$, then $T^2(\mathbf{z}) - T^2(\mathbf{y})$

-   The $T^2$ test can be derived as a likelihood ratio test of $H_0: \mu = \mu_0$

<br>

Confidence Intervals

Confidence Region

An "exact" $100(1-\alpha)\%$ confidence region for $\mathbf{\mu}$ is the set of all vectors, $\mathbf{v}$, which are "close enough" to the observed mean vector, $\bar{\mathbf{y}}$ to satisfy

$$
n(\bar{\mathbf{y}} - \mathbf{\mu}_0)'\mathbf{S}^{-1}(\bar{\mathbf{y}} - \mathbf{\mu}_0) \le \frac{(n-1)p}{n-p} f_{(1-\alpha, p, n-p)}
$$

## MANOVA

Multivariate Analysis of Variance

## Principal Components

## Factor Analysis

## Discriminant Analysis

## Cluster Analysis
