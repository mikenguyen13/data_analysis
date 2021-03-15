# Prerequisites

This chapter is just a quick review of [Matrix Theory] and [Probability Theory]

If you feel you do not need to brush up on these theories, you can jump right into [Descriptive Statistics]

## Matrix Theory

```{=tex}
\begin{equation}
\begin{split}
A=
\left[\begin{array}{c}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
\end{array}
\right]
\end{split}
\end{equation}
```
```{=tex}
\begin{equation}
\begin{split}
A' =
\left[\begin{array}{c}
a_{11} & a_{21} \\
a_{12} & a_{22} \\
\end{array}
\right]
\end{split}
\end{equation}
```
$$
\mathbf{(ABC)'=C'B'A'} \\
\mathbf{A(B+C)= AB + AC} \\
\mathbf{AB \neq BA} \\
\mathbf{(A')'=A} \\
\mathbf{(A+B)' = A' + B'} \\
\mathbf{(AB)' = B'A'} \\
\mathbf{(AB)^{-1}= B^{-1}A^{-1}} \\
\mathbf{A+B = B +A} \\
\mathbf{AA^{-1} = I }
$$

If A has an inverse, it is called **invertible.** If A is not invertible it is called **singular.**

<br>

```{=tex}
\begin{equation}
\begin{split}
\mathbf{A} &= 
\left(\begin{array}{cccc} 
a_{11} & a_{12} & a_{13} \\ 
a_{21} & a_{22} & a_{23} \\ 
\end{array}\right)
\left(\begin{array}{c}
b_{11} & b_{12} & b_{13} \\
b_{21} & b_{22} & b_{23} \\
b_{31} & b_{32} & b_{33} \\
\end{array}\right) \\
&= 
\left(\begin{array}{c}
a_{11}b_{11}+a_{12}b_{21}+a_{13}b_{31} & \sum_{i=1}^{3}a_{1i}b_{i2} &  \sum_{i=1}^{3}a_{1i}b_{i3} \\
\sum_{i=1}^{3}a_{2i}b_{i1} & \sum_{i=1}^{3}a_{2i}b_{i2} & \sum_{i=1}^{3}a_{2i}b_{i3} \\
\end{array}\right) 
\end{split}
\end{equation}
```
Let $\mathbf{a}$ be a 3 x 1 vector, then the quadratic form is

$$
\mathbf{a'Ba} = \sum_{i=1}^{3}\sum_{i=1}^{3}a_i b_{ij} a_{j}
$$

**Length of a vector**\
Let $\mathbf{a}$ be a vector, $||\mathbf{a}||$ (the 2-norm of the vector) is the length of vector $\mathbf{a}$, is the square root of the inner product of the vector with itself:

$$
||\mathbf{a}|| = \sqrt{\mathbf{a'a}}
$$

### Rank

-   Dimension of space spanned by its columns (or its rows).\
-   Number of linearly indepdent columns/rows

For a n x k matrix **A** and k x k matrix **B**

-   $rank(A)\leq min(n,k)$
-   $rank(A) = rank(A') = rank(A'A)=rank(AA')$\
-   $rank(AB)=min(rank(A),rank(B))$\
-   **B** is invertible if and only if rank(B) = k (non-singular)

<br>

### Inverse

In scalar, a = 0 then 1/a does not exist. In matrix, a matrix is invertible when it's a non-zero matrix.

A non-singular square matrix A is invertible if there exists a non-singular square matrix B such that, $$AB=I$$ Then $A^{-1}=B$. For a 2x2 matrix,

$$
A =
\left(\begin{array}{c}
a & b \\
c & d \\
\end{array}
\right)
$$

$$
A^{-1}=
\frac{1}{ad-bc}
\left(\begin{array}{c}
d & -b \\
-c & a \\
\end{array}
\right)
$$

For the partition matrix,

```{=tex}
\begin{equation}
\begin{split}
\left[\begin{array}{c}
A & B \\
C & D \\
\end{array}
\right]^{-1}
 =
\left[\begin{array}{c}
\mathbf{(A-BD^{-1}C)^{-1}} & \mathbf{-(A-BD^{-1}C)^{-1}BD^-1}\\
\mathbf{-DC(A-BD^{-1}C)^{-1}} & \mathbf{D^{-1}+D^{-1}C(A-BD^{-1}C)^{-1}BD^{-1}}\ \\
\end{array}
\right]
\end{split}
\end{equation}
```
<br>

Properties for a non-singular square matrix

-   $\mathbf{A^{-1}}=A$\
-   for a non-zero scalar b, $\mathbf{(bA)^{-1}=b^{-1}A^{-1}}$\
-   for a matrix B, $\mathbf(BA)^{-1}=B^{-1}A^{-1}$ only if B is non-singular
-   $\mathbf{(A^{-1})'=(A')^{-1}}$\
-   Never notate $\mathbf{1/A}$

<br>

### Definiteness

A symmetric square k x k matrix, $\mathbf{A}$, is Positive Semi-Definite if for any non-zero k x 1 vector $\mathbf{x}$, $$\mathbf{x'Ax \geq 0 }$$

A symmetric square k x k matrix, $\mathbf{A}$, is Negative Semi-Definite if for any non-zero k x 1 vector $\mathbf{x}$ $$\mathbf{x'Ax \leq 0 }$$

$\mathbf{A}$ is indefinite if it is neither positive semi-definite or negative semi-definite.

The identity matrix is positive definite

**Example** Let $\mathbf{x} =(x_1 x_2)'$, then for a 2 x 2 identity matrix,

```{=tex}
\begin{equation}
\begin{split}
\mathbf{x'Ix} 
&= (x_1 x_2) 
\left(\begin{array}{c}
1 & 0 \\
0 & 1 \\
\end{array}
\right)
\left(\begin{array}{c}
x_1 \\
x_2 \\
\end{array}
\right) \\
&=
(x_1 x_2)
\left(\begin{array}{c}
x_1 \\
x_2 \\
\end{array}
\right) \\
&=
x_1^2 + x_2^2 >0
\end{split}
\end{equation}
```
Definiteness gives us the ability to compare matrices $\mathbf{A-B}$ is PSD This property also helps us show efficiency (which variance covariance matrix of one estimator is smaller than another)

Properties

-   any variance matrix is PSD\
-   a matrix $\mathbf{A}$ is PSD if and only if there exists a matrix $\mathbf{B}$ such that $\mathbf{A=B'B}$\
-   if $\mathbf{A}$ is PSD, then $\mathbf{B'AB}$ is PSD\
-   if A and C are non-singular, then A-C is PSD if and only if $\mathbf{C^{-1}-A^{-1}}$\
-   if A is PD (ND) then $A^{-1}$ is PD (ND)

**Note**

-   Indefinite A is neither PSD nor NSD. There is no comparable concept in scalar.
-   If a square matrix is PSD and invertible then it is PD

Example:

1.  Invertible / Indefinite

$$
\left[
\begin{array}{a}
-1 & 0 \\
0 & 10 \\
\end{array}
\right]
$$

2.  Non-invertible/ Indefinite

$$
\left[
\begin{array}{a}
0 & 1 \\
0 & 0 \\
\end{array}
\right]
$$

3.  Invertible / PSD

$$
\left[
\begin{array}{a}
1 & 0 \\
0 & 1 \\
\end{array}
\right]
$$

4.  Non-Invertible / PSD

$$
\left[
\begin{array}{a}
0 & 0 \\
0 & 1 \\
\end{array}
\right]
$$

### Matrix Calculus

$y=f(x_1,x_2,...,x_k)=f(x)$ where x is a 1 x k row vector. The Gradient (first order derivative with respect to a vector) is,

$$
\frac{\partial{f(x)}}{\partial{x}}=
\left(\begin{array}{c}
\frac{\partial{f(x)}}{\partial{x_1}} \\
\frac{\partial{f(x)}}{\partial{x_2}} \\
... \\
\frac{\partial{f(x)}}{\partial{x_k}}
\end{array}
\right)
$$

The **Hessian** (second order derivative with respect to a vector) is,

$$
\frac{\partial^2{f(x)}}{\partial{x}\partial{x'}}=
\left(\begin{array}{c}
\frac{\partial^2{f(x)}}{\partial{x_1}\partial{x_1}} & \frac{\partial^2{f(x)}}{\partial{x_1}\partial{x_2}} & ... & \frac{\partial^2{f(x)}}{\partial{x_1}\partial{x_k}} \\
\frac{\partial^2{f(x)}}{\partial{x_1}\partial{x_2}} & \frac{\partial^2{f(x)}}{\partial{x_2}\partial{x_2}} & ... & \frac{\partial^2{f(x)}}{\partial{x_2}\partial{x_k}} \\
... & ...& & ...\\
\frac{\partial^2{f(x)}}{\partial{x_k}\partial{x_1}} & \frac{\partial^2{f(x)}}{\partial{x_k}\partial{x_2}} & ... & \frac{\partial^2{f(x)}}{\partial{x_k}\partial{x_k}}
\end{array}
\right)
$$

### Optimization

+-----------------------------------+--------------------------------------------------+-----------------------------------------------------------------------------------------------------+
|                                   | **Scalar Optimization**                          | **Vector Optimization**                                                                             |
+===================================+:================================================:+:===================================================================================================:+
| First Order Condition             | $$\frac{\partial{f(x_0)}}{\partial{x}}=0$$       | $$\frac{\partial{f(x_0)}}{\partial{x}}=\left(\begin{array}{c}0 \\ .\\ .\\ .\\ 0\end{array}\right)$$ |
+-----------------------------------+--------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| Second Order Condition            | $$\frac{\partial^2{f(x_0)}}{\partial{x^2}} > 0$$ | $$\frac{\partial^2{f(x_0)}}{\partial{xx'}}>0$$                                                      |
|                                   |                                                  |                                                                                                     |
| **Convex** $\rightarrow$ **Min**  |                                                  |                                                                                                     |
+-----------------------------------+--------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| **Concave** $\rightarrow$ **Max** | $$\frac{\partial^2{f(x_0)}}{\partial{x^2}} < 0$$ | $$\frac{\partial^2{f(x_0)}}{\partial{xx'}}<0$$                                                      |
+-----------------------------------+--------------------------------------------------+-----------------------------------------------------------------------------------------------------+

## Probability Theory

### Axiom and Theorems of Probability

1.  Let S denote a sample space of an experiment P[S]=1
2.  $P[A] \ge 0$ for every event A
3.  Let $A_1,A_2,A_3,...$ be a finite or an infinite collection of mutually exclusive events. Then $P[A_1\cup A_2 \cup A_3 ...]=P[A_1]+P[A_2]+P[A_3]+...$
4.  $P[\emptyset]=0$
5.  $P[A']=1-P[A]$
6.  $P[A_1 \cup A_2] = P[A_1] + P[A_2] - P[A_1 \cap A_2]$

<br>

**Conditional Probability**

$$
P[A|B]=\frac{A \cap B}{P[B]}
$$

<br>

**Independent Events** Two events A and B are independent if and only if:

1.  $P[A\cap B]=P[A]P[B]$
2.  $P[A|B]=P[A]$
3.  $P[B|A]=P[B]$

A finite collection of events $A_1, A_2, ..., A_n$ is independent if and only if any subcollection is independent.

<br>

**Multiplication Rule** $P[A \cap B] = P[A|B]P[B] = P[B|A]P[A]$

<br>

**Bayes' Theorem** Let $A_1, A_2, ..., A_n$ be a collection of mutually exclusive events whose union is S.\
Let b be an event such that $P[B]\neq0$\
Then for any of the events $A_j$, j = 1,2,...,n

$$
P[A_|B]=\frac{P[B|A_j]P[A_j]}{\sum_{i=1}^{n}P[B|A_j]P[A_i]}
$$

**Jensen's Inequality**

-   If g(x) is convex $E(g(X)) \ge g(E(X))$
-   If g(x) is concave $E(g(X)) \le g(E(X))$

#### Law of Iterated Expectations

$E(Y)=E(E(Y|X))$

#### Correlation and Independence

**Independence**

-   $f(x,y)=f_X(x)f_Y(y)$
-   $f_{Y|X}(y|x)=f_Y(y)$ and $f_{X|Y}(x|y)=f_X(x)$
-   $E(g_1(X)g_2(Y))=E(g_1(X))E(g_2(Y))$

**Mean Independence** (implied by independence)

-   Y is mean independent of X if and only if $E(Y|X)=E(Y)$
-   $E(Xg(Y))=E(X)E(g(Y))$

**Uncorrelated** (implied by independence and mean independence)

-   $Cov(X,Y)=0$
-   $Var(X+Y)=Var(X) + Var(Y)$
-   $E(XY)=E(X)E(Y)$

$$
Strongest \\
\downarrow \\
Independence \\
\downarrow \\
Mean Independence \\
\downarrow \\
Uncorrelated \\ 
\downarrow \\
Weakest\\
$$

<br>

### Central Limit Theorem

Let $X_1, X_2,...,X_n$ be a random sample of size n from a distribution (not necessarily normal) X with mean $\mu$ and variance $\sigma^2$. then for large n ($n \ge 25$),

1.  $\bar{X}$ is approximately normal with with mean $\mu_{\bar{X}}=\mu$ and variance $\sigma^2_{\bar{X}} = Var(\bar{X})= \frac{\sigma^2}{n}$\

2.  $\hat{p}$is approximately normal with $\mu_{\hat{p}} = p, \sigma^2_{\hat{p}} = \frac{p(1-p)}{n}$

3.  $\hat{p_1} - \hat{p_2}$ is approximately normal with $\mu_{\hat{p_1} - \hat{p_2}} = p_1 - p_2, \sigma^2_{\hat{p_1} - \hat{p_2}}=\frac{p_1(1-p)}{n_1} + \frac{p_2(1-p)}{n_2}$

4.  $\bar{X_1} - \bar{X_2}$ is approximately normal with $\mu_{\bar{X_1} - \bar{X_2}} = \mu_1 - \mu_2, \sigma^2_{\bar{X_1} - \bar{X_2}} = \frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}$

5.  The following random variables are approximately standard normal:\

    -   $\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}$\
    -   $\frac{\hat{p}-p}{\sqrt{\frac{p(1-p)}{n}}}$\
    -   $\frac{(\hat{p_1}-\hat{p_2})-(p_1-p_2)}{\sqrt{\frac{p_1(1-p_1)}{n_1}-\frac{p_2(1-p_2)}{n_2}}}$\
    -   $\frac{(\bar{X_1}-\bar{X_2})-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma^2_1}{n_1}-\frac{\sigma^2_2}{n_2}}}$

If $\{x_i\}_{i=1}^{n}$ is an iid random sample from a probability distribution with finite mean $\mu$ and finite variance $\sigma^2$ then the sample mean $\bar{x}=n^{-1}\sum_{i=1}^{n}x_i$ scaled by $\sqrt{n}$ has the following limiting distribution

$$
\sqrt{n}(\bar{x}-\mu) \to^d N(0,\sigma^2)
$$

or if we were to standardize the sample mean,

$$
\frac{\sqrt{n}(\bar{x}-\mu)}{\sigma} \to^d N(0,1)
$$

-   holds for most random sample from any distribution (continuous, discrete, unknown).\
-   extends to multivariate case: random sample of a random vector converges to a multivariate normal.\
-   Variance from the limiting distribution is the asymptotic variance (Avar)

$$
Avar(\sqrt{n}(\bar{x}-\mu)) = \sigma^2 \\
\lim_{n \to \infty} Var(\sqrt{n}(\bar{x}-\mu)) = \sigma^2 \\
Avar(.) \neq lim_{n \to \infty} Var(.)
$$

### Random variable

+------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                          | Discrete Variable                                                                                                                | Continuous Variable                                                                                                                                                                             |
+==========================================================================================+==================================================================================================================================+=================================================================================================================================================================================================+
| **Definition**                                                                           | A random variable is discrete if it can assume at most a finite or countably infinite number of possible values                  | A random variable is continuous if it can assume any value in some interval or intervals of real numbers and the probability that it assumes any specific value is 0                            |
+------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Density Function**                                                                     | A function f is called a density for X if: <br> (1) $f(x) \ge 0$ <br> (2) $\sum_{all~x}f(x)=1$ <br> (3) $f(x)=P(X=x)$ for x real | A function f is called a density for X if: <br> (1) $f(x) \ge 0$ for x real <br> (2) $\int_{-\infty}^{\infty} f(x) \; dx=1$ <br> (3) $P[a \le X \le] =\int_{a}^{b} f(x) \; dx$ for a and b real |
+------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Cumulative Distribution Function** <br> for x real                                     | $F(x)=P[X \le x]$                                                                                                                | $F(x)=P[X \le x]=\int_{-\infty}^{\infty}f(t)dt$                                                                                                                                                 |
+------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| $E[H(X)]$                                                                                | $\sum_{all ~x}H(x)f(x)$                                                                                                          | $\int_{-\infty}^{\infty}H(x)f(x)$                                                                                                                                                               |
+------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| $\mu=E[X]$                                                                               | $\sum_{all ~ x}xf(x)$                                                                                                            | $\int_{-\infty}^{\infty}xf(x)$                                                                                                                                                                  |
+------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Ordinary Moments** <br> the kth ordinary moment for variable X is defined as: $E[X^k]$ | $\sum_{all ~ x \in X}(x^kf(x))$                                                                                                  | $\int_{-\infty}^{\infty}(x^kf(x))$                                                                                                                                                              |
+------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Moment generating function (mgf)** <br> $m_X(t)=E[e^{tX}]$                             | $\sum_{all ~ x \in X}(e^{tx}f(x))$                                                                                               | $\int_{-\infty}^{\infty}(e^{tx}f(x)dx)$                                                                                                                                                         |
+------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

<br>

Expected Value Properties:

-   E[c] = c for any constant c
-   E[cX] = cE[X] for any constant c
-   E[X+Y] = E[X] = E[Y]
-   E[XY] = E[X].E[Y] (if X and Y are independent)

Expected Variance Properties:

-   $Var(c) = 0$ for any constant c
-   $Var(cX) = c^2Var(X)$ for any constant c
-   $Var(X) \ge 0$
-   $Var(X) = E(X^2) - (E(X))^2$
-   $Var(X+c)=Var(X)$
-   $Var (X+Y) = Var(X) + Var(Y)$ (if X and Y are independent)

Standard deviation $\sigma=\sqrt(\sigma^2)=\sqrt(Var X)$

Suppose $y_1,...,y_p$ are possibly correlated random variables with means $\mu_1,...,\mu_p$. then

$$
\mathbf{y} = (y_1,...,y_p)'  \\
E(\mathbf{y}) = (\mu_1,...,\mu_p)' = \mathbf{\mu}
$$

Let $\sigma_{ij} = cov(y_i,y_j)$ for $i,j = 1,..,p$.

Define

$$
\mathbf{\Sigma} = (\sigma_{ij}) = 
\left(\begin{array}
{rrrr}
\sigma_{11} & \sigma_{12} & ... & \sigma_{1p} \\
\sigma_{21} & \sigma_{22} & ... & \sigma_{2p} \\
. & . & . & . \\
\sigma_{p1} & \sigma_{p2} & ... & \sigma_{pp}\\
\end{array}\right)
$$

Hence, $\mathbf{\Sigma}$ is the variance-covariance or dispersion matrix. And $\mathbf{\Sigma}$ is symmetric with $(p+1)p/2$ unique parameters.

Alternatively, let $u_{p \times 1}$ and $v_{v \times 1}$ be random vectors with means $\mathbf{\mu_u}$ and $\mathbf{\mu_v}$. then

$$
\mathbf{\Sigma_{uv}} = cov(\mathbf{u,v}) = E[\mathbf{(u-\mu_u)(v-\mu_v)'}]
$$

$\Sigma_{uv} \neq \Sigma_{vu}$ (but $\Sigma_{uv} = \Sigma_{vu}'$)

Properties of Covariance Matrices

1.  Symmetric: $\mathbf{\Sigma' = \Sigma}$\
2.  Eigendecomposition (spectral decomposition,symmetric decomposition): $\mathbf{\Sigma = \Phi \Lambda \Phi}$, where $\mathbf{\Phi}$ is a matrix of eigenvectors such that $\mathbf{\Phi \Phi' = I}$ (orthonormal), and $\mathbf{\Lambda}$ is a diagonal matrix with eigenvalues $(\lambda_1,...,\lambda_p)$ on the diagonal.\
3.  Non-negative definite, $\mathbf{a \Sigma a} \ge 0$ for any $\mathbf{a} \in R^p$. Equivalently, the eigenvalues of $\mathbf{\Sigma}$, $\lambda_1 \ge ... \ge \lambda_p \ge 0$\
4.  $|\mathbf{\Sigma}| = \lambda_1,...\lambda_p \ge 0$ (generalized variance)\
5.  $trace(\mathbf{\Sigma})= tr(\mathbf{\Sigma}) = \lambda_1 +... + \lambda_p = \sigma_{11}+...+ \sigma_{pp}$= sum of variances (total variance)

Note: $\mathbf{\Sigma}$ is usually required to be positive definite. This implies that all eigenvalues are positive, and $\mathbf{\Sigma}$ has an inverse $\mathbf{\Sigma}^{-1}$, such that $\mathbf{\Sigma}^{-1}\mathbf{\Sigma}= \mathbf{I}_{p \times p} = \mathbf{\Sigma}\mathbf{\Sigma}^{-1}$

<br>

**Correlation Matrices**

Define the correlation $\rho_{ij}$ and the correlation matrix by

$$
\rho_{ij} = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii} \sigma_{jj}}}
$$

```{=tex}
\mathbf{R} = \left(
\begin{array}
{rrrr}
\rho_{11} & \rho_{12} & ... & \rho_{1p} \\
\rho_{21} & \rho_{22} & ... & \rho_{2p} \\
. & . & . & . \\
\rho_{p1} & \rho_{p2} & ... & \rho_{pp}\\
\end{array}
\right)
```
where $\rho_{ii}=1$ for all i.

Let x and y be random vectors with means $\mu_x$ and $\mu_y$ and variance-covariance matrices $\Sigma_x$ and $\Sigma_y$. Let $\mathbf{A}$ and $\mathbf{B}$ be matrices of constants and c and d be vectors of constants. Then,

-   $E(\mathbf{Ay+c}) = \mathbf{A \mu_y} +c$
-   $var(\mathbf{Ay +c}) = \mathbf{A}var(\mathbf{y}) \mathbf{A}' = \mathbf{A \Sigma_y A'}$
-   $cov(\mathbf{Ay + c,By+d} = \mathbf{A \Sigma_y B'}$

### Moment generating function

Moment generating function properties:

(a) $\frac{d^k(m_X(t))}{dt^k}|_{t=0}=E[X^k]$\
(b) $\mu=E[X]=m_X'(0)$\
(c) $E[X^2]=m_X''(0)$

**mgf Theorems**

Let $X_1,X_2,...X_n,Y$ be random variables with moment-generating functions $m_{X_1}(t),m_{X_2}(t),...,m_{X_n}(t),m_{Y}(t)$

1.  If $m_{X_1}(t)=m_{X_2}(t)$ for all t in some open interval about 0, then $X_1$ and $X_2$ have the same distribution
2.  If $Y = \alpha + \beta X_1$, then $m_{Y}(t)= e^{\alpha t}m_{X_1}(\beta t)$
3.  If $X_1,X_2,...X_n$ are independent and $Y = \alpha_0 + \alpha_1 X_1 + \alpha_2 X_2 + ... + \alpha_n X_n$ (where $\alpha_0, ... ,\alpha_n$ are real numbers), then $m_{Y}(t)=e^{\alpha_0 t}m_{X_1}(\alpha_1t)m_{X_2}(\alpha_2 t)...m_{X_n}(\alpha_nt)$\
4.  Suppose $X_1,X_2,...X_n$ are independent normal random variables with means $\mu_1,\mu_2,...\mu_n$ and variances $\sigma^2_1,\sigma^2_2,...,\sigma^2_n$. If $Y = \alpha_0 + \alpha_1 X_1 + \alpha_2 X_2 + ... + \alpha_n X_n$ (where $\alpha_0, ... ,\alpha_n$ are real numbers), then Y is normally distributed with mean $\mu_Y = \alpha_0 + \alpha_1 \mu_1 +\alpha_2 \mu_2 + ... + \alpha_n \mu_n$ and variance $\sigma^2_Y = \alpha_1^2 \sigma_1^2 + \alpha_2^2 \sigma_2^2 + ... + \alpha_n^2 \sigma_n^2$

### Moment

| Moment | Uncentered         | Centered                       |
|--------|--------------------|--------------------------------|
| 1st    | $E(X)=\mu=Mean(X)$ |                                |
| 2nd    | $E(X^2)$           | $E((X-\mu)^2)=Var(X)=\sigma^2$ |
| 3rd    | $E(X^3)$           | $E((X-\mu)^3)$                 |
| 4th    | $E(X^4)$           | $E((X-\mu)^4)$                 |

Skewness(X) = $E((X-\mu)^3)/\sigma^3$

Kurtosis(X) = $E((X-\mu)^4)/\sigma^4$

**Conditional Moments**

$$
E(Y|X=x)=
\begin{cases}
\sum_yyf_Y(y|x) & \text{for discrete RV}\\
\int_yyf_Y(y|x)dy & \text{for continous RV}\\
\end{cases}
$$

$$
Var(Y|X=x)=
\begin{cases}
\sum_y(y-E(Y|x))^2f_Y(y|x) & \text{for discrete RV}\\
\int_y(y-E(Y|x))^2f_Y(y|x)dy & \text{for continous RV}\\
\end{cases}
$$

#### Multivariate Moments

$$
\begin{equation}
E=
\left(
\begin{array}{c}
X \\
Y \\
\end{array}
\right)
=
\left(
\begin{array}{c}
E(X) \\
E(Y) \\
\end{array}
\right)
=
\left(
\begin{array}{c}
\mu_X \\
\mu_Y \\
\end{array}
\right)
\end{equation}
$$

$$
\begin{equation}
\begin{split}
Var
\left(
\begin{array}{c}
X \\
Y \\
\end{array}
\right)
&=
\left(
\begin{array}{c}
Var(X) & Cov(X,Y) \\
Cov(X,Y) & Var(Y) \\
\end{array}
\right) \\
&=
\left(
\begin{array}{c}
E((X-\mu_X)^2) & E((X-\mu_X)(Y-\mu_Y)) \\
E((X-\mu_X)(Y-\mu_Y)) & E((Y-\mu_Y)^2) \\
\end{array}
\right)
\end{split}
\end{equation}
$$

<br>

**Properties**

-   $E(aX + bY + c)=aE(X) +bE(Y) + c$
-   $Var(aX + bY + c) = a^2Var(X) + b^2Var(Y) + 2abCov(X,Y)$
-   $Cov(aX + bY, cX + bY) =acVar(X)+bdVar(Y) + (ad+bc)Cov(X,Y)$
-   Correlation: $\rho_{XY} = \frac{Cov(X,Y)}{\sigma_X\sigma_Y}$

### Distributions

**Conditional Distributions** $$
f_{X|Y}(X|Y=y)=\frac{f(X,Y)}{f_Y(y)}
$$ $f_{X|Y}(X|Y=y)=f_X(X)$ if X and Y are independent

#### Discrete

CDF: Cumulative Density Function\
MGF: Moment Generating Function

##### Bernoulli

$Bernoulli(p)$

**PDF**


```r
hist(mc2d::rbern(100000, prob=.5))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-1-1.png" width="672" />

##### Binomial

$B(n,p)$

-   the experiment consists of a fixed number (n) of Bernoulli trials, each of which results in a success (s) or failure (f)
-   The trials are identical and independent, and probability of success (p) and probability of failure (q = 1- p) remains the same for all trials.
-   The random variable X denotes the number of successes obtained in the n trials.

**Density**

$$
f(x)={{n}\choose{x}}p^xq^{n-x}
$$

**CDF**\
You have to use table

**PDF**


```r
# Histogram of 100000 random values from a sample of 100 with probability of 0.5
hist(rbinom(100000, size = 100, prob = 0.5))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-2-1.png" width="672" />

**MGF**

$$
m_X(t) =(q+pe^t)^n
$$

**Mean**

$$
\mu = E(x) = np
$$

**Variance**

$$
\sigma^2 =Var(X) = npq
$$

##### Poisson

$Pois(\lambda)$

-   Arises with Poisson process, which involves observing discrete events in a continuous "interval" of time, length, or space.
-   The random variable X is the number of occurrences of the event within an interval of s units
-   The parameter $\lambda$ is the average number of occurrences of the event in question per measurement unit. For the distribution, we use the parameter $k=\lambda s$

**Density**

$$
f(x) = \frac{e^{-k}k^x}{x!}
$$

,k\> 0, x =0,1,...

**CDF**\
Use table

**PDF**


```r
# Poisson dist with mean of 5 or Poisson(5)
hist(rpois(10000, lambda = 5))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-3-1.png" width="672" />

**MGF**

$$
m_X(t)=e^{k(e^t-1)}
$$

**Mean**

$$
\mu = E(X) = k
$$

**Variance**

$$
\sigma^2 = Var(X) = k
$$

##### Geometric

-   The experiment consists of a series of trails. The outcome of each trial can be classed as being either a "success" (s) or "failure" (f). (This is called a Bernoulli trial).
-   The trials are identical and independent in the sense that the outcome of one trial has no effect on the outcome of any other. The probability of success (p) and probability of failure (q=1-p) remains the same from trial to trial.
-   lack of memory
-   X: the number of trials needed to obtain the first success.

**Density**

$$
f(x)=pq^{x-1}
$$

**CDF**\
$$
F(x) = 1- q^x
$$

**PDF**


```r
# hist of Geometric distribution with probability of success = 0.5
hist(rgeom(n = 10000, prob = 0.5))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-4-1.png" width="672" />

**MGF**

$$
m_X(t) = \frac{pe^t}{1-qe^t}
$$

for $t < -ln(q)$

**Mean**

$$
\mu = \frac{1}{p}
$$

**Variance**

$$
\sigma^2 = Var(X) = \frac{q}{p^2}
$$

##### Hypergeometric

-   The experiment consists of drawing a random sample of size n without replacement and without regard to order from a collection of N objects.
-   Of the N objects, r have a trait of interest; N-r do not have the trait
-   X is the number of objects in the sample with the trait.

**Density**

$$
f(x)=\frac{{{r}\choose{x}}{{N-r}\choose{n-x}}}{{{N}\choose{n}}}
$$

where $max[0,n-(N-r)] \le x \le min(n,r)$

**PDF**


```r
# hist of hypergeometric distribution with the number of white balls = 50, and the number of black balls = 20, and number of balls drawn = 30. 
hist(rhyper(nn = 10000 , m=50, n=20, k=30))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-5-1.png" width="672" />

**Mean**

$$
\mu = E(x)= \frac{nr}{N}
$$

**Variance**

$$
\sigma^2 = var(X) = n (\frac{r}{N})(\frac{N-r}{N})(\frac{N-n}{N-1})
$$

**Note** For large N (if $\frac{n}{N} \le 0.05$), this distribution can be approximated using a Binomial distribution with $p = \frac{r}{N}$

##### 

<br>

#### Continuous

##### Uniform

-   Defined over an interval (a,b) in which the probabilities are "equally likely" for subintervals of equal length.

**Density**

$$
f(x)=\frac{1}{b-a}
$$

for a \< x \< b

**CDF**

```{=tex}
\begin{cases}
0 & \text{if x <a } \\
\frac{x-a}{b-a} & \text{if $a \le x \le b$ }\\
1 & \text{if x >b}\\
\end{cases}
```
**PDF**


```r
hist(runif(100000, min = 0, max = 1))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-6-1.png" width="672" />

**MGF**

```{=tex}
\begin{cases}
\frac{e^{tb} - e^{ta}}{t(b-a)}&\text{ if $t \neq 0$}\\
1&\text{if $ t \neq 0$}\\
\end{cases}
```
**Mean**

$$
\mu = E(X) = \frac{a +b}{2}
$$

**Variance**

$$
\sigma^2 = Var(X) = \frac{(b-a)^2}{12}
$$

##### Gamma

-   is used to define the exponential and chi-squared distributions
-   The gamma function is defined as:

$$
\Gamma(\alpha) = \int_0^{\infty} z^{\alpha-1}e^{-z}dz
$$

where $\alpha > 0$

-   Properties of The Gamma function:\

    -   $\Gamma(1) = 1$ + For $\alpha >1$, $\Gamma(\alpha)=(\alpha-1)\Gamma(\alpha-1)$ + If n is an integer and $n>1$, then $\Gamma(n) = (n-1)!$

**Density**

$$
f(x) = \frac{1}{\Gamma(\alpha)\beta^{\alpha}}x^{\alpha-1}e^{-x/\beta}
$$

**CDF**

$$
F(x,n,\beta) = 1 -\sum_{k=0}^{n-1} \frac{(\frac{x}{\beta})^k e^{-x/\beta}}{k!}
$$

for x\>0, and $\alpha = n$ (a positive integer)

**PDF**


```r
hist(rgamma(n = 10000, shape = 5, rate = 1))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-7-1.png" width="672" />

**MGF**

$$
m_X(t) = (1-\beta t)^{-\alpha}
$$

where $t < \frac{1}{\beta}$

**Mean**

$$
\mu = E(X) = \alpha \beta
$$

**Variance**

$$
\sigma^2 = Var(X) = \alpha \beta^2
$$

##### Normal

$N(\mu,\sigma^2)$

-   is symmetric, bell-shaped curve with parameters $\mu$ and $\sigma^2$
-   also known as Gaussian.

**Density**

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi }}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
$$

for $-\infty < x, \mu< \infty, \sigma > 0$

**CDF**

Use table

**PDF**


```r
hist(rnorm(1000000, mean = 0, sd = 1))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-8-1.png" width="672" />

**MGF**

$$
m_X(t) = e^{\mu t + \frac{\sigma^2 t^2}{2}}
$$

**Mean**

$$
\mu = E(X)
$$

**Variance**

$$
\sigma^2 = Var(X)
$$

**Standard Normal Random Variable**

-   The normal random variable Z with mean $\mu = 0$ and standard deviation $\sigma =1$ is called standard normal
-   Any normal random variable X with mean $\mu$ and standard deviation $\sigma$ can be converted to the standard normal random variable $Z = \frac{X-\mu}{\sigma}$

**Normal Approximation to the Binomial Distribution**

Let X be binomial with parameters n and p. For large n (so that $(A)p \le .5$ and $np > 5$ or (B) $p>.5$ and $nq>5$), X is approximately normally distributed with mean $\mu = np$ and standard deviation $\sigma = \sqrt{npq}$

When using the normal approximation, add or subtract 0.5 as needed for the continuity correction

|  Discrete  | Approximate Normal (corrected) |
|:----------:|:------------------------------:|
|  P(X = c)  |   P(c -0.5 \< Y \< c + 0.5)    |
| P(X \< c)  |        P(Y \< c - 0.5)         |
| P(X \le c) |        P(Y \< c + 0.5)         |
| P(X \> c)  |        P(Y \> c + 0.5)         |
| P(X \ge c) |        P(Y \> c - 0.5)         |

**Normal Probability Rule**

If X is normally distributed with parameters $\mu$ and $\sigma$, then

-   $P(-\sigma < X - \mu < \sigma) \approx .68$ \* $P(-2\sigma < X - \mu < 2\sigma) \approx .95$ \* $P(-3\sigma < X - \mu < 3\sigma) \approx .997$

##### Logistic

$Logistic(\mu,s)$

**PDF**


```r
hist(rlogis(n = 100000, location = 0, scale = 1))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-9-1.png" width="672" />

##### Lognomral

$lognormal(\mu,\sigma^2)$

**PDF**


```r
hist(rlnorm(n = 10000, meanlog = 0, sdlog = 1))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-10-1.png" width="672" />

##### Exponential

$Exp(\lambda)$

-   A special case of the gamma distribution with $\alpha = 1$
-   Lack of memory
-   $\lambda$ = rate Within a Poisson process with parameter $\lambda$, if W is the waiting tine until the occurrence of the first event, then W has an exponential distribution with $\beta = 1/\alpha$

**Density**

$$
f(x) = \frac{1}{\beta} e^{-x/\beta}
$$

for $x,\beta > 0$

**CDF**

```{=tex}
\begin{equation}
F(x) = 
\begin{cases}
0 & \text{ if $x \le 0$}\\
1 - e^{-x/\beta} & \text{if $x > 0$}\\
\end{cases}
\end{equation}
```
**PDF**


```r
hist(rexp(n = 100000, rate = 1))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-11-1.png" width="672" />

**MGF**

$$
m_X(t) = (1-\beta t)^{-1}
$$

for $t < 1/\beta$

**Mean**

$$
\mu = E(X) = \beta
$$

**Variance**

$$
\sigma^2 = Var(X) =\beta^2
$$

##### Chi-squared

$\chi^2=\chi^2(k)$

-   A special case of the gamma distribution with $\beta =2$, and $\alpha = \gamma /2$ for a positive integer $\gamma$
-   The random variable X is denoted $\chi_{\gamma}^2$ and is said to have a chi-squared distribution with $\gamma$ degrees of freedom.

**Density** Use density for Gamma Distribution with $\beta = 2$ and $\alpha = \gamma/2$

**CDF** Use table

**PDF**


```r
hist(rchisq(n = 10000, df=2, ncp = 0))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-12-1.png" width="672" />

**MGF**

$$
m_X(t) = (1-2t)^{-\gamma/2}
$$

**Mean**

$$
\mu = E(X) = \gamma
$$

**Variance**

$$
\sigma^2 = Var(X) = 2\gamma
$$

##### Student T

$T(v)$

-   $T=\frac{Z}{\sqrt{\chi_{\gamma}^2/\gamma}}$, where Z is standard normal follows a student-t distribution with $\gamma$ dof
-   The distribution is symmetric, bell-shaped , with a mean of $\mu=0$


```r
hist(rt(n = 100000, df=2, ncp =1))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-13-1.png" width="672" />

##### F-Distribution

$F(d_1,d_2)$

-   F distribution is strictly positive
-   $F=\frac{\chi_{\gamma_1}^2/\gamma_1}{\chi_{\gamma_2^2}/\gamma_2}$ follows an F distribution with dof $\gamma_1$ and $\gamma_2$, where $\chi_{\gamma_1}^2$ and $\chi_{\gamma_2}^2$ are independent chi-squared random variables.
-   The distribution is asymmetric and never negative.

**PDF**


```r
hist(rf(n = 100000, df1=2, df2=3, ncp=1))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-14-1.png" width="672" />

##### Cauchy

[Central Limit Theorem] and [Weak Law] do not apply to Cauchy because it does not have finite mean and finite variance

**PDF**


```r
hist(rcauchy(n = 100000, location = 0, scale = 1))
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-15-1.png" width="672" />

##### Multivariate Normal Distribution

Let y be a p-dimensional multivariate normal (MVN) rv with mean $\mu$ and variance $\Sigma$. Then, the density of y is

$$
f(\mathbf{y}) = \frac{1}{(2\pi)^{p/2}|\mathbf{\Sigma}|^{1/2}}exp(-\frac{1}{2}\mathbf{(y-\mu)'\Sigma^{-1}(y-\mu)})
$$

We have $\mathbf{y} \sim N_p(\mathbf{\mu,\Sigma})$

Properties:

-   Let $\mathbf{A}_{r \times p}$ be a fixed matrix. then $\mathbf{Ay} \sim N_r(\mathbf{A \mu, A \Sigma A')}$. Note that $r \le p$ and all rows of A must be linearly independent to guarantee that $\mathbf{A\Sigma A'}$ is non-singular.\
-   Let $\mathbf{G}$ be a matrix such that $\mathbf{\Sigma^{-1}= GG'}$. then, $\mathbf{G'y} \sim N_p (\mathbf{G'\mu,I})$ and $\mathbf{G'(y-\mu)} \sim N_p (\mathbf{0,I})$.\
-   Any fixed linear combination of $y_1,...,y_p$ say $\mathbf{c'y}$, follows $\mathbf{c'y} \sim N_1(\mathbf{c'\mu,c'\Sigma c})$

**Large Sample Properties**

Suppose that $y_1,...,y_n$ are a random sample from some population with mean $\mu$ and variance-variance matrix $\Sigma$

$$
\mathbf{Y} \sim MVN(\mathbf{\mu,\Sigma})
$$

Then

-   $\bar{\mathbf{y}} = \frac{1}{n}\sum_{i=1}^n \mathbf{y}_i$ is a consistent estimator for $\mathbf{\mu}$\
-   $\mathbf{S} = \frac{1}{n-1}\sum_{i=1}^n \mathbf{(y_i - \bar{y})(y_i - \bar{y})'}$ is a consistent estimator for $\mathbf{\Sigma}$\
-   Multivariate Central Limit Theorem: Similar to the univariate case, $\sqrt{n}(\mathbf{\bar{y}- \mu}) \sim N_p(\mathbf{0, \Sigma})$ when n is large relative to p (e.g., $n \ge 25p$), which is equivalent to $\bar{y} \sim N_p(\mathbf{\mu,\Sigma/n})$\
-   Wald's Theorem: $n(\mathbf{\bar{y}- \mu)'S^{-1}(\bar{y}- \mu)} \sim \chi^2_{(p)}$ when n is large relative to p.

##### 

## General Math

**Chebyshev's Inequality** Let X be a random variable with mean $\mu$ and standard deviation $\sigma$. Then for any positive number k:

$$
P(|X-\mu| < k\sigma) \ge 1 - \frac{1}{k^2}
$$

Chebyshev's Inequality does not require that X be normally distributed

**Maclaurin series expansion for**

$$
e^z = 1 + z + \frac{z^2}{2!} + \frac{z^3}{3!} + ...
$$

<br>

**Geometric series:**

$$
s_n=\sum_{k=1}^{n}ar^{n-1}=\frac{a(1-r^n)}{1-r}
$$

if \|r\| \< 1

$$
s=\sum_{k=1}^{\infty}ar^{n-1}=\frac{a}{1-r}
$$

### Law of large numbers

Let $X_1,X_2,...$ be an infinite sequence of independent and identically distributed (i.i.d) Then, the sample average is

$$
\bar{X}_n =\frac{1}{n} (X_1 + ... + X_n)
$$

converges to the expected value ($\bar{X}_n \rightarrow \mu$) as $n \rightarrow \infty$

$$
Var(X_i) = Var(\frac{1}{n}(X_1 + ... + X_n)) = \frac{1}{n^2}Var(X_1 + ... + X_n)= \frac{n\sigma^2}{n^2}=\frac{\sigma^2}{n}
$$

The difference between [Weak Law] and [Strong Law] regards the mode of convergence

#### Weak Law

The sample average converges in probability towards the expected value

$$
\bar{X}_n \rightarrow^{p} \mu
$$

when $n \rightarrow \infty$

$$
\lim_{n\to \infty}P(|\bar{X}_n - \mu| > \epsilon) = 0
$$

The sample mean from a iid random sample ($\{ x_i \}_{i=1}^n$) from any population with a finite mean and finite variance $\sigma^2$ is ca consistent estimation for the population mean $\mu$

$$
plim(\bar{x})=plim(n^{-1}\sum_{i=1}^{n}x_i) =\mu
$$

#### Strong Law

The sample average converges almost surely to the expected value

$$
\bar{X}_n \rightarrow^{a.s} \mu 
$$

when $n \rightarrow \infty$

Equivalently,

$$
P(\lim_{n\to \infty}\bar{X}_n =\mu) =1
$$

### Law of Iterated Expectation

Let X, Y be random variables. Then,

$$
E(X) = E(E(X|Y))
$$

means that the expected value of X can be calculated from the probability distribution of X\|Y and Y

### Convergence

#### Convergence in Probability

-   $n \rightarrow \infty$, an estimator (random variable) that is close to the true value.
-   The random variable $\theta_n$ converges in probability to a constant c if

$$
\lim_{n\to \infty}P(|\theta_n - c| \ge \epsilon) = 0
$$

for any positive $\epsilon$

Notation

$$
plim(\theta_n)=c 
$$

Equivalently,

$$
\theta_n \rightarrow^p c
$$

**Properties of Convergence in Probability**

-   Slutsky's Theorem: for a continuous function g(.), if $plim(\theta_n)= \theta$ then $plim(g(\theta_n)) = g(\theta)$

-   if $\gamma_n \rightarrow^p \gamma$ then\

    -   $plim(\theta_n + \gamma_n)=\theta + \gamma$ + $plim(\theta_n \gamma_n) = \theta \gamma$ + $plim(\theta_n/\gamma_n) = \theta/\gamma$ if $\gamma \neq 0$

-   Also hold for random vectors/ matrices

#### Convergence in Distribution

-   As $n \rightarrow \infty$, the distribution of a random variable may converge towards another ("fixed") distribution.
-   The random variable $X_n$ with CDF $F_n(x)$ converges in distribution to a random variable X with CDF $F(X)$ if

$$
\lim_{n\to \infty}|F_n(x) - F(x)| = 0
$$

at all points of continuity of $F(X)$

Notation F(x) is the limiting distribution of $X_n$ or $X_n \rightarrow^d X$

-   E(X) is the limiting mean (asymptotic mean)
-   Var(X) is the limiting variance (asymptotic variance)

**Note**

$$
E(X) \neq \lim_{n\to \infty}E(X_n) \\
Avar(X_n) \neq \lim_{n\to \infty}Var(X_n)
$$

**Properties of Convergence in Distribution**

-   Continuous Mapping Theorem: for a continuous function g(.), if $X_n \to^{d} g(X)$ then $g(X_n) \to^{d} g(X)$

-   if $Y_n\to^{d} c$, then

    -   $X_n + Y_n \to^{d} X + c$

    -   $Y_nX_n \to^{d} cX$

    -   $X_nY_n \to^{d} X/c$ if $c \neq 0$

-   also hold for random vectors/matrices

#### Summary

Properties of Convergence

+---------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| Probability                                                                                                         | Distribution                                                                                                 |
+=====================================================================================================================+==============================================================================================================+
| Slutsky's Theorem: for a continuous function g(.), if $plim(\theta_n)= \theta$ then $plim(g(\theta_n)) = g(\theta)$ | Continuous Mapping Theorem: for a continuous function g(.), if $X_n \to^{d} g(X)$ then $g(X_n) \to^{d} g(X)$ |
+---------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| if $\gamma_n \rightarrow^p \gamma$ then                                                                             | if $Y_n\to^{d} c$, then                                                                                      |
+---------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| $plim(\theta_n + \gamma_n)=\theta + \gamma$                                                                         | $X_n + Y_n \to^{d} X + c$                                                                                    |
+---------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| $plim(\theta_n \gamma_n) = \theta \gamma$                                                                           | $Y_nX_n \to^{d} cX$                                                                                          |
+---------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| $plim(\theta_n/\gamma_n) = \theta/\gamma$ if $\gamma \neq 0$                                                        | $X_nY_n \to^{d} X/c$ if $c \neq 0$                                                                           |
+---------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+

[Convergence in Probability] is stronger than [Convergence in Distribution]. However, [Convergence in Distribution] does not guarantee [Convergence in Probability]

### Sufficient Statistics

**Likelihood**

-   describes the extent to which the sample provides support for any particular parameter value.
-   Higher support corresponds to a higher value for the likelihood
-   The exact value of any likelihood is **meaningless**,
-   The relative value, (i.e., comparing two values of $\theta$), is **informative**.

$$
L(\theta_0; y) = P(Y = y | \theta = \theta_0) = f_Y(y;\theta_0)
$$

**Likelihood Ratio**

$$
\frac{L(\theta_0;y)}{L(\theta_1;y)}
$$

**Likelihood Function**

For a given sample, you can create likelihoods for all possible values of $\theta$, which is called *likelihood function*

$$
L(\theta) = L(\theta; y) = f_Y(y;\theta)
$$

In a sample of size n, the likelihood function takes the form of a product

$$
L(\theta) = \prod_{i=1}^{n}f_i (y_i;\theta)
$$

Equivalently, the log likelihood function

$$
l(\theta) = \sum_{i=1}^{n} logf_i(y_i;\theta)
$$

**Sufficient statistics**

-   A statistic, T(y), is any quantity that can be calculated purely from a sample (independent of $\theta$)
-   A statistic is **sufficient** if it conveys all the available information about the parameter.

$$
L(\theta; y) = c(y)L^*(\theta;T(y))
$$

**Nuisance parameters** If we are interested in a parameter (e.g., mean). Other parameters requiring estimation (e.g., standard deviation) are **nuisance** parameters. We can replace nuisance parameters in likelihood function with their estimates to create a \*\*profile likelihood\*.

### Parameter transformations

log-odds transformation

$$
Log odds = g(\theta)= ln[\frac{\theta}{1-\theta}]
$$

log transformation

## Methods

Trade-off between parametric and non-parametric

## Data Manipulation


```r
# load packages
library(tidyverse)
```

```
## -- Attaching packages --------------------------------------- tidyverse 1.3.0 --
```

```
## v ggplot2 3.3.3     v purrr   0.3.4
## v tibble  3.0.6     v dplyr   1.0.4
## v tidyr   1.1.2     v stringr 1.4.0
## v readr   1.4.0     v forcats 0.5.1
```

```
## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
```

```r
library(lubridate)
```

```
## 
## Attaching package: 'lubridate'
```

```
## The following objects are masked from 'package:base':
## 
##     date, intersect, setdiff, union
```

```r
x <- c(1, 4, 23, 4, 45)
n <- c(1, 3, 5)
g <- c("M", "M", "F")
df <- data.frame(n, g)
df
```

```
##   n g
## 1 1 M
## 2 3 M
## 3 5 F
```

```r
str(df)
```

```
## 'data.frame':	3 obs. of  2 variables:
##  $ n: num  1 3 5
##  $ g: chr  "M" "M" "F"
```

```r
#Similarly
df <- tibble(n, g)
df
```

```
## # A tibble: 3 x 2
##       n g    
##   <dbl> <chr>
## 1     1 M    
## 2     3 M    
## 3     5 F
```

```r
str(df)
```

```
## tibble [3 x 2] (S3: tbl_df/tbl/data.frame)
##  $ n: num [1:3] 1 3 5
##  $ g: chr [1:3] "M" "M" "F"
```

```r
# list form
lst <- list(x, n, g, df)
lst
```

```
## [[1]]
## [1]  1  4 23  4 45
## 
## [[2]]
## [1] 1 3 5
## 
## [[3]]
## [1] "M" "M" "F"
## 
## [[4]]
## # A tibble: 3 x 2
##       n g    
##   <dbl> <chr>
## 1     1 M    
## 2     3 M    
## 3     5 F
```

```r
# Or
lst2 <- list(num = x, size = n, sex = g, data = df)
lst2
```

```
## $num
## [1]  1  4 23  4 45
## 
## $size
## [1] 1 3 5
## 
## $sex
## [1] "M" "M" "F"
## 
## $data
## # A tibble: 3 x 2
##       n g    
##   <dbl> <chr>
## 1     1 M    
## 2     3 M    
## 3     5 F
```

```r
# Or
lst3 <- list(x = c(1, 3, 5, 7),
             y = c(2, 2, 2, 4, 5, 5, 5, 6),
             z = c(22, 3, 3, 3, 5, 10))
lst3
```

```
## $x
## [1] 1 3 5 7
## 
## $y
## [1] 2 2 2 4 5 5 5 6
## 
## $z
## [1] 22  3  3  3  5 10
```

```r
# find the means of x, y, z.

# can do one at a time
mean(lst3$x)
```

```
## [1] 4
```

```r
mean(lst3$y)
```

```
## [1] 3.875
```

```r
mean(lst3$z)
```

```
## [1] 7.666667
```

```r
# list apply
lapply(lst3, mean)
```

```
## $x
## [1] 4
## 
## $y
## [1] 3.875
## 
## $z
## [1] 7.666667
```

```r
# OR
sapply(lst3, mean)
```

```
##        x        y        z 
## 4.000000 3.875000 7.666667
```

```r
# Or, tidyverse function map() 
map(lst3, mean)
```

```
## $x
## [1] 4
## 
## $y
## [1] 3.875
## 
## $z
## [1] 7.666667
```

```r
# The tidyverse requires a modified map function called map_dbl()
map_dbl(lst3, mean)
```

```
##        x        y        z 
## 4.000000 3.875000 7.666667
```

```r
# Binding 
dat01 <- tibble(x = 1:5, y = 5:1)
dat01
```

```
## # A tibble: 5 x 2
##       x     y
##   <int> <int>
## 1     1     5
## 2     2     4
## 3     3     3
## 4     4     2
## 5     5     1
```

```r
dat02 <- tibble(x = 10:16, y = x/2)
dat02
```

```
## # A tibble: 7 x 2
##       x     y
##   <int> <dbl>
## 1    10   5  
## 2    11   5.5
## 3    12   6  
## 4    13   6.5
## 5    14   7  
## 6    15   7.5
## 7    16   8
```

```r
dat03 <- tibble(z = runif(5)) # 5 random numbers from interval (0,1)
dat03
```

```
## # A tibble: 5 x 1
##       z
##   <dbl>
## 1 0.292
## 2 0.716
## 3 0.223
## 4 0.314
## 5 0.365
```

```r
# row binding
bind_rows(dat01, dat02, dat01)
```

```
## # A tibble: 17 x 2
##        x     y
##    <int> <dbl>
##  1     1   5  
##  2     2   4  
##  3     3   3  
##  4     4   2  
##  5     5   1  
##  6    10   5  
##  7    11   5.5
##  8    12   6  
##  9    13   6.5
## 10    14   7  
## 11    15   7.5
## 12    16   8  
## 13     1   5  
## 14     2   4  
## 15     3   3  
## 16     4   2  
## 17     5   1
```

```r
# use ".id" argument to create a new column that contains an identifier for the original data.
bind_rows(dat01, dat02, .id = "id")
```

```
## # A tibble: 12 x 3
##    id        x     y
##    <chr> <int> <dbl>
##  1 1         1   5  
##  2 1         2   4  
##  3 1         3   3  
##  4 1         4   2  
##  5 1         5   1  
##  6 2        10   5  
##  7 2        11   5.5
##  8 2        12   6  
##  9 2        13   6.5
## 10 2        14   7  
## 11 2        15   7.5
## 12 2        16   8
```

```r
# with name
bind_rows("dat01" = dat01, "dat02" = dat02, .id = "id")
```

```
## # A tibble: 12 x 3
##    id        x     y
##    <chr> <int> <dbl>
##  1 dat01     1   5  
##  2 dat01     2   4  
##  3 dat01     3   3  
##  4 dat01     4   2  
##  5 dat01     5   1  
##  6 dat02    10   5  
##  7 dat02    11   5.5
##  8 dat02    12   6  
##  9 dat02    13   6.5
## 10 dat02    14   7  
## 11 dat02    15   7.5
## 12 dat02    16   8
```

```r
# bind_rows() also works on lists of data frames
list01 <- list("dat01" = dat01, "dat02" = dat02)
list01
```

```
## $dat01
## # A tibble: 5 x 2
##       x     y
##   <int> <int>
## 1     1     5
## 2     2     4
## 3     3     3
## 4     4     2
## 5     5     1
## 
## $dat02
## # A tibble: 7 x 2
##       x     y
##   <int> <dbl>
## 1    10   5  
## 2    11   5.5
## 3    12   6  
## 4    13   6.5
## 5    14   7  
## 6    15   7.5
## 7    16   8
```

```r
bind_rows(list01)
```

```
## # A tibble: 12 x 2
##        x     y
##    <int> <dbl>
##  1     1   5  
##  2     2   4  
##  3     3   3  
##  4     4   2  
##  5     5   1  
##  6    10   5  
##  7    11   5.5
##  8    12   6  
##  9    13   6.5
## 10    14   7  
## 11    15   7.5
## 12    16   8
```

```r
bind_rows(list01, .id = "source")
```

```
## # A tibble: 12 x 3
##    source     x     y
##    <chr>  <int> <dbl>
##  1 dat01      1   5  
##  2 dat01      2   4  
##  3 dat01      3   3  
##  4 dat01      4   2  
##  5 dat01      5   1  
##  6 dat02     10   5  
##  7 dat02     11   5.5
##  8 dat02     12   6  
##  9 dat02     13   6.5
## 10 dat02     14   7  
## 11 dat02     15   7.5
## 12 dat02     16   8
```

```r
# The extended example below demonstrates how this can be very handy.

# column binding
bind_cols(dat01, dat03)
```

```
## # A tibble: 5 x 3
##       x     y     z
##   <int> <int> <dbl>
## 1     1     5 0.292
## 2     2     4 0.716
## 3     3     3 0.223
## 4     4     2 0.314
## 5     5     1 0.365
```

```r
# Regular expressions -----------------------------------------------------
names <- c("Ford, MS", "Jones, PhD", "Martin, Phd", "Huck, MA, MLS")

# pattern: first comma and everything after it
str_remove(names, pattern = ", [[:print:]]+")
```

```
## [1] "Ford"   "Jones"  "Martin" "Huck"
```

```r
# [[:print:]]+ = one or more printable characters


# Reshaping ---------------------------------------------------------------

# Example of a wide data frame. Notice each person has multiple test scores
# that span columns.
wide <- data.frame(name=c("Clay","Garrett","Addison"), 
                   test1=c(78, 93, 90), 
                   test2=c(87, 91, 97),
                   test3=c(88, 99, 91))
wide
```

```
##      name test1 test2 test3
## 1    Clay    78    87    88
## 2 Garrett    93    91    99
## 3 Addison    90    97    91
```

```r
# Example of a long data frame. This is the same data as above, but in long
# format. We have one row per person per test.
long <- data.frame(name=rep(c("Clay","Garrett","Addison"),each=3),
                   test=rep(1:3, 3),
                   score=c(78, 87, 88, 93, 91, 99, 90, 97, 91))
long
```

```
##      name test score
## 1    Clay    1    78
## 2    Clay    2    87
## 3    Clay    3    88
## 4 Garrett    1    93
## 5 Garrett    2    91
## 6 Garrett    3    99
## 7 Addison    1    90
## 8 Addison    2    97
## 9 Addison    3    91
```

```r
# mean score per student
aggregate(score ~ name, data = long, mean)
```

```
##      name    score
## 1 Addison 92.66667
## 2    Clay 84.33333
## 3 Garrett 94.33333
```

```r
aggregate(score ~ test, data = long, mean)
```

```
##   test    score
## 1    1 87.00000
## 2    2 91.66667
## 3    3 92.66667
```

```r
# line plot of scores over test, grouped by name
ggplot(long, aes(x = factor(test), y = score, color = name, group = name)) +
  geom_point() +
  geom_line() +
  xlab("Test")
```

<img src="02-prerequisites_files/figure-html/unnamed-chunk-16-1.png" width="672" />

```r
#### reshape wide to long
pivot_longer(wide, test1:test3, names_to = "test", values_to = "score")
```

```
## # A tibble: 9 x 3
##   name    test  score
##   <chr>   <chr> <dbl>
## 1 Clay    test1    78
## 2 Clay    test2    87
## 3 Clay    test3    88
## 4 Garrett test1    93
## 5 Garrett test2    91
## 6 Garrett test3    99
## 7 Addison test1    90
## 8 Addison test2    97
## 9 Addison test3    91
```

```r
# Or
pivot_longer(wide, -name, names_to = "test", values_to = "score")
```

```
## # A tibble: 9 x 3
##   name    test  score
##   <chr>   <chr> <dbl>
## 1 Clay    test1    78
## 2 Clay    test2    87
## 3 Clay    test3    88
## 4 Garrett test1    93
## 5 Garrett test2    91
## 6 Garrett test3    99
## 7 Addison test1    90
## 8 Addison test2    97
## 9 Addison test3    91
```

```r
# drop "test" from the test column with names_prefix argument
pivot_longer(wide, -name, names_to = "test", values_to = "score", 
             names_prefix = "test")
```

```
## # A tibble: 9 x 3
##   name    test  score
##   <chr>   <chr> <dbl>
## 1 Clay    1        78
## 2 Clay    2        87
## 3 Clay    3        88
## 4 Garrett 1        93
## 5 Garrett 2        91
## 6 Garrett 3        99
## 7 Addison 1        90
## 8 Addison 2        97
## 9 Addison 3        91
```

```r
#### reshape long to wide 
pivot_wider(long, name, names_from = test, values_from = score)
```

```
## # A tibble: 3 x 4
##   name      `1`   `2`   `3`
##   <chr>   <dbl> <dbl> <dbl>
## 1 Clay       78    87    88
## 2 Garrett    93    91    99
## 3 Addison    90    97    91
```

```r
# using the names_prefix argument lets us prepend text to the column names.
pivot_wider(long, name, names_from = test, values_from = score,
            names_prefix = "test")
```

```
## # A tibble: 3 x 4
##   name    test1 test2 test3
##   <chr>   <dbl> <dbl> <dbl>
## 1 Clay       78    87    88
## 2 Garrett    93    91    99
## 3 Addison    90    97    91
```

The verbs of data manipulation

-   select: selecting (or not selecting) columns based on their names (eg: select columns Q1 through Q25)
-   slice: selecting (or not selecting) rows based on their position (eg: select rows 1:10)
-   mutate: add or derive new columns (or variables) based on existing columns (eg: create a new column that expresses measurement in cm based on existing measure in inches)
-   rename: rename variables or change column names (eg: change "GraduationRate100" to "grad100")
-   filter: selecting rows based on a condition (eg: all rows where gender = Male)
-   arrange: ordering rows based on variable(s) numeric or alphabetical order (eg: sort in descending order of Income)
-   sample: take random samples of data (eg: sample 80% of data to create a "training" set)
-   summarize: condense or aggregate multiple values into single summary values (eg: calculate median income by age group)
-   group_by: convert a tbl into a grouped tbl so that operations are performed "by group"; allows us to summarize data or apply verbs to data by groups (eg, by gender or treatment)
-   the pipe: %\>%
-   Use Ctrl + Shift + M (Win) or Cmd + Shift + M (Mac) to enter in RStudio
-   The pipe takes the output of a function and "pipes" into the first argument of the next function.
