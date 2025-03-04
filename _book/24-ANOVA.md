# Analysis of Variance {#sec-analysis-of-variance-anova}

Analysis of Variance (ANOVA) shares its underlying mechanism with linear regression. However, ANOVA approaches the analysis from a different perspective, making it particularly useful for studying **qualitative variables** and **designed experiments**.

**Key Terminology**

-   **Factor**: An explanatory or predictor variable studied in an experiment.
-   **Treatment (Factor Level)**: A specific value or category of a factor applied to an experimental unit.
-   **Experimental Unit**: The entity (e.g., person, animal, material) subjected to treatments and providing a response.
-   **Single-Factor Experiment**: An experiment with only one explanatory variable.
-   **Multifactor Experiment**: An experiment involving multiple explanatory variables.
-   **Classification Factor**: A factor that is not controlled by the experimenter (common in observational studies).
-   **Experimental Factor**: A factor that is directly assigned by the experimenter.

A well-designed experiment requires careful planning in the following areas:

-   **Choice of treatments**: Selecting factor levels to be tested.
-   **Selection of experimental units**: Ensuring an appropriate sample.
-   **Treatment assignment**: Avoiding selection bias through proper randomization.
-   **Measurement**: Minimizing measurement bias and considering blinding when necessary.

Advancements in Experimental Design

1.  **Factorial Experiments**:
    -   Investigate multiple factors simultaneously.
    -   Allow for the study of **interactions** between factors.
2.  **Replication**:
    -   Repeating experiments increases statistical power.
    -   Helps estimate **mean squared error**.
3.  **Randomization**:
    -   Introduced formally by R.A. Fisher in the early 1900s.
    -   Ensures that treatment assignment is not systematically biased.
    -   Helps eliminate confounding effects due to time, space, or other lurking variables.
4.  **Local Control (Blocking/Stratification)**:
    -   Reduces experimental error by controlling for known sources of variability.
    -   Increases power by grouping similar experimental units together before randomizing treatments.

Randomization also helps eliminate correlations due to time and space.

------------------------------------------------------------------------

## Completely Randomized Design {#sec-completely-randomized-design}

A **Completely Randomized Design (CRD)** is the simplest type of experimental design, where experimental units are randomly assigned to treatments.

Consider a treatment factor $A$ with $a \geq 2$ treatment levels. Each experimental unit is randomly assigned to one of these levels. The number of units in each group can be:

-   **Balanced**: All groups have equal sample sizes $n$.
-   **Unbalanced**: Groups have different sample sizes $n_i$ (for $i = 1, ..., a$).

The total sample size is given by:

$$
N = \sum_{i=1}^{a} n_i
$$

The number of possible assignments of units to treatments is:

$$
k = \frac{N!}{n_1! n_2! \dots n_a!}
$$

Each assignment has an equal probability of being selected: $1/k$. The response of each experimental unit is denoted as $Y_{ij}$, where:

-   $i$ indexes the treatment group.
-   $j$ indexes the individual unit within treatment $i$.

| Treatment   | 1              | 2              | ... | a              |
|-------------|----------------|----------------|-----|----------------|
|             | $Y_{11}$       | $Y_{21}$       | ... | $Y_{a1}$       |
|             | $Y_{12}$       | ...            | ... | ...            |
|             | ...            | ...            | ... | ...            |
| Sample Mean | $\bar{Y_{1.}}$ | $\bar{Y_{2.}}$ | ... | $\bar{Y_{a.}}$ |
| Sample SD   | $s_1$          | $s_2$          | ... | $s_a$          |

: Treatment Response Table

Where:

$$
\bar{Y_{i.}} = \frac{1}{n_i} \sum_{j=1}^{n_i} Y_{ij}
$$

$$
s_i^2 = \frac{1}{n_i - 1} \sum_{j=1}^{n_i} (Y_{ij} - \bar{Y_{i.}})^2
$$

The **grand mean** is:

$$
\bar{Y_{..}} = \frac{1}{N} \sum_{i} \sum_{j} Y_{ij}
$$

------------------------------------------------------------------------

### Single-Factor Fixed Effects ANOVA {#sec-single-factor-fixed-effects-model}

Also known as **One-Way ANOVA** or **ANOVA Type I Model**.

The total variability in the response variable $Y_{ij}$ can be decomposed as follows:

$$
\begin{aligned}
Y_{ij} - \bar{Y_{..}} &= Y_{ij} - \bar{Y}_{..} + \bar{Y}_{i.} - \bar{Y}_{i.} \\
& = (\bar{Y_{i.}} - \bar{Y_{..}}) + (Y_{ij} - \bar{Y_{i.}})
\end{aligned}
$$

where:

-   The first term represents **between-treatment variability** (deviation of treatment means from the grand mean).
-   The second term represents **within-treatment variability** (deviation of observations from their treatment mean).

Thus, we partition the **total sum of squares (SSTO)** as:

$$
\sum_{i} \sum_{j} (Y_{ij} - \bar{Y_{..}})^2 = \sum_{i} n_i (\bar{Y_{i.}} - \bar{Y_{..}})^2 + \sum_{i} \sum_{j} (Y_{ij} - \bar{Y_{i.}})^2
$$

Or equivalently:

$$
SSTO = SSTR + SSE
$$

Where:

-   **SSTO (Total SS)**: Total variability in the data.
-   **SSTR (Treatment SS)**: Variability due to differences between treatment means.
-   **SSE (Error SS)**: Variability within treatments (unexplained variance).

Degrees of freedom (d.f.):

$$
(N-1) = (a-1) + (N-a)
$$

where we lose a degree of freedom for the total corrected SSTO because of the estimation of the mean ($\sum_i \sum_j (Y_{ij} - \bar{Y}_{..} )= 0$) and for the SSTR ($\sum_i n_i (\bar{Y}_{i.} - \bar{Y}_{..}) = 0$)

Mean squares:

$$
MSTR = \frac{SSTR}{a-1}, \quad MSR = \frac{SSE}{N-a}
$$

+---------------------------+---------------------------------------------+------------+--------------+
| Source of Variation       | SS                                          | df         | MS           |
+===========================+=============================================+============+==============+
| Between Treatments        | $\sum_{i}n_i (\bar{Y_{i.}}-\bar{Y_{..}})^2$ | $a-1$      | $SSTR/(a-1)$ |
+---------------------------+---------------------------------------------+------------+--------------+
| Error (within treatments) | $\sum_{i}\sum_{j}(Y_{ij}-\bar{Y_{i.}})^2$   | $N-a$      | $SSE/(N-a)$  |
+---------------------------+---------------------------------------------+------------+--------------+
| Total (corrected)         | $\sum_{i}n_i (\bar{Y_{i.}}-\bar{Y_{..}})^2$ | $N-1$      |              |
+---------------------------+---------------------------------------------+------------+--------------+

: ANOVA Table

For a linear model interpretation of ANOVA, we have either

1.  Cell Means Model
2.  Treatment Effect (Factor Effects Model)

------------------------------------------------------------------------

#### Cell Means Model {#sec-cell-means-model}

The **cell means model** describes the response as:

$$
Y_{ij} = \mu_i + \epsilon_{ij}
$$

where:

-   $Y_{ij}$: Response for unit $j$ in treatment $i$.
-   $\mu_i$: Fixed population mean for treatment $i$.
-   $\epsilon_{ij} \sim N(0, \sigma^2)$: Independent errors.
-   $E(Y_{ij}) = \mu_i$, $\text{Var}(Y_{ij}) = \sigma^2$.

All observations are assumed to have **equal variance** across treatments.

------------------------------------------------------------------------

Example: ANOVA with $a = 3$ Treatments

Consider a case with **three treatments** ($a = 3$), where each treatment has **two replicates** ($n_1 = n_2 = n_3 = 2$). The response vector can be expressed in matrix form as:

$$
\begin{aligned}
\left(\begin{array}{c} 
Y_{11}\\
Y_{12}\\
Y_{21}\\
Y_{22}\\
Y_{31}\\
Y_{32}\\
\end{array}\right) &= 
\left(\begin{array}{ccc} 
1 & 0 & 0 \\ 
1 & 0 & 0 \\ 
0 & 1 & 0 \\ 
0 & 1 & 0 \\ 
0 & 0 & 1 \\ 
0 & 0 & 1 \\ 
\end{array}\right)
\left(\begin{array}{c}
\mu_1 \\
\mu_2 \\
\mu_3 \\
\end{array}\right) + \left(\begin{array}{c}
\epsilon_{11} \\
\epsilon_{12} \\
\epsilon_{21} \\
\epsilon_{22} \\
\epsilon_{31} \\
\epsilon_{32} \\
\end{array}\right)\\
\mathbf{y} &= \mathbf{X\beta} +\mathbf{\epsilon}
\end{aligned}
$$

where:

-   $X_{k,ij} = 1$ if the $k$-th treatment is applied to unit $(i,j)$.
-   $X_{k,ij} = 0$ otherwise.

**Note:** There is **no intercept** term in this model.

The least squares estimator for $\beta$ is given by:

```{=tex}
\begin{equation}
\begin{aligned}
\mathbf{b}= \left[\begin{array}{c}
\mu_1 \\
\mu_2 \\
\mu_3 \\
\end{array}\right] &= 
(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y} \\
& = 
\left[\begin{array}{ccc}
n_1 & 0 & 0\\
0 & n_2 & 0\\
0 & 0 & n_3 \\
\end{array}\right]^{-1}
\left[\begin{array}{c}
Y_1\\
Y_2\\
Y_3\\
\end{array}\right] \\
& = 
\left[\begin{array}{c}
\bar{Y_1}\\
\bar{Y_2}\\
\bar{Y_3}\\
\end{array}\right] 
\end{aligned} 
(\#eq:betaorigin)
\end{equation}
```
Thus, the estimated treatment means are:

$$
\hat{\mu}_i = \bar{Y_i}, \quad i = 1,2,3
$$

This estimator $\mathbf{b} = [\bar{Y_1}, \bar{Y_2}, \bar{Y_3}]'$ is the **best linear unbiased estimator (BLUE)** for $\beta$ (i.e., $E(\mathbf{b}) = \beta$)

Since $\mathbf{b} \sim N(\beta, \sigma^2 (\mathbf{X'X})^{-1})$, the variance of the estimated treatment means is:

$$
var(\mathbf{b}) = \sigma^2(\mathbf{X'X})^{-1} = \sigma^2
\left[\begin{array}{ccc}
1/n_1 & 0 & 0\\
0 & 1/n_2 & 0\\
0 & 0 & 1/n_3\\
\end{array}\right]
$$

Thus, the variance of each estimated treatment mean is:

$$
var(b_i) = var(\hat{\mu}_i) = \frac{\sigma^2}{n_i}, \quad i = 1,2,3
$$

The **mean squared error (MSE)** is given by:

$$
\begin{aligned}
MSE 
&= \frac{1}{N - a} \sum_{i=1}^a \sum_{j=1}^{n_i} \bigl(Y_{ij} - \overline{Y}_{i\cdot}\bigr)^2
\\[6pt]
&= \frac{1}{N - a} 
   \sum_{i=1}^a 
   \Bigl[ 
     (n_i - 1) \;
     \underbrace{
       \frac{1}{n_i - 1}
       \sum_{j=1}^{n_i} 
         \bigl(Y_{ij} - \overline{Y}_{i\cdot}\bigr)^2
     }_{=\,s_i^2}
   \Bigr]
\\[6pt]
&= \frac{1}{N - a} \sum_{i=1}^a (n_i - 1)\, s_i^2.
\end{aligned}
$$

where $s_i^2$ is the sample variance within the $i$-th treatment group.

Since $E(s_i^2) = \sigma^2$, we get:

$$
E(MSE) = \frac{1}{N-a} \sum_{i} (n_i-1) \sigma^2 = \sigma^2
$$

Thus, **MSE is an unbiased estimator of** $\sigma^2$, regardless of whether the treatment means are equal.

------------------------------------------------------------------------

The expected mean square for treatments (MSTR) is:

$$
E(MSTR) = \sigma^2 + \frac{\sum_{i} n_i (\mu_i - \mu_.)^2}{a-1}
$$

where:

$$
\mu_. = \frac{\sum_{i=1}^{a} n_i \mu_i}{\sum_{i=1}^{a} n_i}
$$

If all treatment means are equal ($\mu_1 = \mu_2 = \dots = \mu_a = \mu_.$), then:

$$
E(MSTR) = \sigma^2
$$

------------------------------------------------------------------------

$F$-Test for Equality of Treatment Means

We test the null hypothesis:

$$
H_0: \mu_1 = \mu_2 = \dots = \mu_a
$$

against the alternative:

$$
H_a: \text{at least one } \mu_i \text{ differs}
$$

The **test statistic** is:

$$
F = \frac{MSTR}{MSE}
$$

-   Large values of $F$ suggest rejecting $H_0$ (since MSTR will be larger than MSE when $H_a$ is true).
-   Values of $F$ near 1 suggest that we fail to reject $H_0$.

Since $MSTR$ and $MSE$ are independent chi-square random variables scaled by their degrees of freedom, under $H_0$:

$$
F \sim F_{(a-1, N-a)}
$$

Decision Rule:

-   If $F \leq F_{(a-1, N-a;1-\alpha)}$, **fail to reject** $H_0$.
-   If $F \geq F_{(a-1, N-a;1-\alpha)}$, **reject** $H_0$.

------------------------------------------------------------------------

If there are only two treatments ($a = 2$), the ANOVA $F$-test **reduces to the two-sample** $t$-test:

$$
F = t^2
$$

where:

$$
t = \frac{\bar{Y_1} - \bar{Y_2}}{\sqrt{MSE \left(\frac{1}{n_1} + \frac{1}{n_2} \right)}}
$$

------------------------------------------------------------------------

#### Treatment Effects (Factor Effects)

Besides [cell means model](#sec-cell-means-model), we have another way to formalize one-way ANOVA:

$$Y_{ij} = \mu + \tau_i + \epsilon_{ij}$$

where:

-   $Y_{ij}$ is the $j$-th response for the $i$-th treatment.
-   $\tau_i$ is the $i$-th treatment effect.
-   $\mu$ is the constant component common to all observations.
-   $\epsilon_{ij}$ are independent random errors, assumed to be normally distributed: $\epsilon_{ij} \sim N(0, \sigma^2)$.

For example, if we have $a = 3$ treatments and $n_1 = n_2 = n_3 = 2$ observations per treatment, the model representation is:

```{=tex}
\begin{equation} 
\begin{aligned}
\left(\begin{array}{c} 
Y_{11}\\
Y_{12}\\
Y_{21}\\
Y_{22}\\
Y_{31}\\
Y_{32}\\
\end{array}\right) &= 
\left(\begin{array}{cccc} 
1 & 1 & 0 & 0 \\ 
1 & 1 & 0 & 0 \\ 
1 & 0 & 1 & 0 \\ 
1 & 0 & 1 & 0 \\ 
1 & 0 & 0 & 1 \\ 
1 & 0 & 0 & 1 \\ 
\end{array}\right)
\left(\begin{array}{c}
\mu \\
\tau_1 \\
\tau_2 \\
\tau_3\\
\end{array}\right) + \left(\begin{array}{c}
\epsilon_{11} \\
\epsilon_{12} \\
\epsilon_{21} \\
\epsilon_{22} \\
\epsilon_{31} \\
\epsilon_{32} \\
\end{array}\right)\\
\mathbf{y} &= \mathbf{X\beta} +\mathbf{\epsilon} 
\end{aligned}
(\#eq:unsolvable)
\end{equation}
```
However, the matrix:

$$
\mathbf{X'X} = 
\left(
\begin{array}
{cccc}
\sum_{i}n_i & n_1 & n_2 & n_3 \\
n_1 & n_1 & 0 & 0 \\
n_2 & 0 & n_2 & 0 \\
n_3 & 0 & 0 & n_3 \\
\end{array}
\right)
$$

is **singular**, meaning $\mathbf{X'X}$ is not invertible. This results in an infinite number of possible solutions for $\mathbf{b}$.

To resolve this, we impose restrictions on the parameters to ensure that $\mathbf{X}$ has full rank. Regardless of the restriction used, the expected value remains:

$$
E(Y_{ij}) = \mu + \tau_i = \mu_i = \text{mean response for the $i$-th treatment}
$$

------------------------------------------------------------------------

##### Restriction on Sum of Treatment Effects {#restriction-on-sum-of-tau}

One common restriction is:

$$
\sum_{i=1}^{a} \tau_i = 0
$$

which implies that:

$$
\mu = \frac{1}{a} \sum_{i=1}^{a} (\mu + \tau_i)
$$

meaning that $\mu$ represents the grand mean (the overall mean response across treatments).

Each treatment effect can then be expressed as:

$$
\begin{aligned}
\tau_i &= \mu_i - \mu \\
&= \text{treatment mean} - \text{grand mean}
\end{aligned}
$$

Since $\sum_{i} \tau_i = 0$, we can solve for $\tau_a$ as:

$$
\tau_a = -(\tau_1 + \tau_2 + \dots + \tau_{a-1})
$$

Thus, the mean for the $a$-th treatment is:

$$
\mu_a = \mu + \tau_a = \mu - (\tau_1 + \tau_2 + \dots + \tau_{a-1})
$$

This reduces the number of parameters from $a + 1$ to just $a$, meaning we estimate:

$$
\mu, \tau_1, \tau_2, ..., \tau_{a-1}
$$

Rewriting Equation \@ref(eq:unsolvable):

```{=tex}
\begin{equation}
\begin{aligned}
\left(\begin{array}{c} 
Y_{11}\\
Y_{12}\\
Y_{21}\\
Y_{22}\\
Y_{31}\\
Y_{32}\\
\end{array}\right) &= 
\left(\begin{array}{ccc} 
1 & 1 & 0 \\ 
1 & 1 & 0 \\ 
1 & 0 & 1 \\ 
1 & 0 & 1 \\ 
1 & -1 & -1 \\ 
1 & -1 & -1 \\ 
\end{array}\right)
\left(\begin{array}{c}
\mu \\
\tau_1 \\
\tau_2 \\
\end{array}\right) + \left(\begin{array}{c}
\epsilon_{11} \\
\epsilon_{12} \\
\epsilon_{21} \\
\epsilon_{22} \\
\epsilon_{31} \\
\epsilon_{32} \\
\end{array}\right)\\
\mathbf{y} &= \mathbf{X\beta} +\mathbf{\epsilon}
\end{aligned}
\end{equation}
```
where $\beta = [\mu, \tau_1, \tau_2]'$.

------------------------------------------------------------------------

##### Restriction on the First $\tau$ {#restriction-on-first-tau}

In **R**, the default parameterization in `lm()` for a one-way ANOVA model sets $\tau_1 = 0$. This effectively chooses the first treatment (or group) as a baseline or reference, making its treatment effect $\tau_1$ equal to zero.

Consider the last example with three treatments, each having two observations, $\,n_1 = n_2 = n_3 = 2$. Under the restriction $\tau_1 = 0$, the treatment means can be expressed as:

$$
\begin{aligned}
\mu_1 &= \mu + \tau_1 \;=\; \mu + 0 \;=\; \mu, \\
\mu_2 &= \mu + \tau_2, \\
\mu_3 &= \mu + \tau_3.
\end{aligned}
$$

Hence, $\mu$ becomes the mean response for the **first** treatment.

We write the observations in vector form:

$$
\begin{aligned}
\mathbf{y} 
&= \begin{pmatrix} 
Y_{11}\\
Y_{12}\\
Y_{21}\\
Y_{22}\\
Y_{31}\\
Y_{32}\\
\end{pmatrix}
= 
\underbrace{
\begin{pmatrix} 
1 & 0 & 0 \\ 
1 & 0 & 0 \\ 
1 & 1 & 0 \\ 
1 & 1 & 0 \\ 
1 & 0 & 1 \\ 
1 & 0 & 1 \\ 
\end{pmatrix}
}_{\mathbf{X}}
\begin{pmatrix}
\mu \\
\tau_2 \\
\tau_3 \\
\end{pmatrix}
+
\begin{pmatrix}
\epsilon_{11} \\
\epsilon_{12} \\
\epsilon_{21} \\
\epsilon_{22} \\
\epsilon_{31} \\
\epsilon_{32} \\
\end{pmatrix} \\
&= \mathbf{X\beta} + \mathbf{\epsilon},
\end{aligned}
$$

where

$$
\beta = 
\begin{pmatrix}
\mu \\ 
\tau_2 \\ 
\tau_3
\end{pmatrix}.
$$

The OLS estimator is:

$$
\mathbf{b}
= 
\begin{pmatrix}
\hat{\mu} \\
\hat{\tau_2} \\
\hat{\tau_3}
\end{pmatrix}
= (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\,\mathbf{y}.
$$

In our specific case with equal sample sizes ($n_1=n_2=n_3=2$), the $(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}$ calculation yields:

$$
\begin{aligned}
\mathbf{b}
 & =     \left[\begin{array}{ccc}          \sum_{i}n_i & n_2 & n_3\\          n_2 & n_2 & 0\\          n_3 & 0 & n_3 \\          \end{array}\right]^{-1}\left[\begin{array}{c}      Y_{..}\\      Y_{2.}\\      Y_{3.}\\      \end{array}\right] \\
&= 
\begin{pmatrix}
\bar{Y}_{1\cdot} \\
\bar{Y}_{2\cdot} - \bar{Y}_{1\cdot} \\
\bar{Y}_{3\cdot} - \bar{Y}_{1\cdot}
\end{pmatrix}
\end{aligned}
$$ where $\bar{Y}_{1\cdot}$, $\bar{Y}_{2\cdot}$, and $\bar{Y}_{3\cdot}$ are the sample means for treatments 1, 2, and 3, respectively.

Taking the expectation of $\mathbf{b}$ confirms:

$$
E(\mathbf{b}) 
= 
\beta
= 
\begin{pmatrix}
\mu \\
\tau_2 \\
\tau_3
\end{pmatrix}
=
\begin{pmatrix}
\mu_1 \\
\mu_2 - \mu_1 \\
\mu_3 - \mu_1
\end{pmatrix}.
$$

Recall that:

$$
\text{Var}(\mathbf{b}) 
= 
\sigma^2\,(\mathbf{X}'\mathbf{X})^{-1}.
$$

Hence,

$$
\begin{aligned}
\text{Var}(\hat{\mu}) 
&= \text{Var}(\bar{Y}_{1\cdot})
= \frac{\sigma^2}{n_1}, \\[6pt]
\text{Var}(\hat{\tau_2})
&= \text{Var}\bigl(\bar{Y}_{2\cdot}-\bar{Y}_{1\cdot}\bigr)
= \frac{\sigma^2}{n_2} + \frac{\sigma^2}{n_1}, \\[6pt]
\text{Var}(\hat{\tau_3})
&= \text{Var}\bigl(\bar{Y}_{3\cdot}-\bar{Y}_{1\cdot}\bigr)
= \frac{\sigma^2}{n_3} + \frac{\sigma^2}{n_1}.
\end{aligned}
$$

------------------------------------------------------------------------

#### Equivalence of Parameterizations

Despite having different ways of writing the model, all three parameterizations yield the **same** ANOVA table:

1.  [Model 1](#sec-cell-means-model): $Y_{ij} = \mu_i + \epsilon_{ij}$.
2.  [Model 2](#restriction-on-sum-of-tau): $Y_{ij} = \mu + \tau_i + \epsilon_{ij}$ where $\sum_i \tau_i = 0$.
3.  [Model 3](#restriction-on-first-tau): $Y_{ij} = \mu + \tau_i + \epsilon_{ij}$ where $\tau_1 = 0$.

All three lead to the same fitted values, because

$$
\mathbf{\hat{Y}} = \mathbf{X}\bigl(\mathbf{X}'\mathbf{X}\bigr)^{-1}\mathbf{X}'\mathbf{Y}
= \mathbf{P\,Y} 
= \mathbf{X\,b}.
$$

------------------------------------------------------------------------

#### ANOVA Table

The generic form of the ANOVA table is:

+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+-------+--------------------+--------------------+
| Source of Variation     | SS                                                                                                                                      | df    | MS                 | F                  |
+=========================+=========================================================================================================================================+=======+====================+====================+
| **Between Treatments**  | $\sum_{i} n_i (\overline{Y}_{i\cdot} - \overline{Y}_{\cdot\cdot})^2 \;=\; \mathbf{Y}'(\mathbf{P} - \mathbf{P}_1)\mathbf{Y}$             | $a-1$ | $\frac{SSTR}{a-1}$ | $\frac{MSTR}{MSE}$ |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+-------+--------------------+--------------------+
| **Error**               | $\sum_{i}\sum_{j}\bigl(Y_{ij} - \overline{Y}_{i\cdot}\bigr)^2 \;=\; \mathbf{e}'\mathbf{e}$                                              | $N-a$ | $\frac{SSE}{N-a}$  |                    |
|                         |                                                                                                                                         |       |                    |                    |
| **(within treatments)** |                                                                                                                                         |       |                    |                    |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+-------+--------------------+--------------------+
| **Total (corrected)**   | $\sum_{i} n_i(\overline{Y}_{i\cdot} - \overline{Y}_{\cdot\cdot})^2 \;=\; \mathbf{Y}'\mathbf{Y} \;-\; \mathbf{Y}'\mathbf{P}_1\mathbf{Y}$ | $N-1$ |                    |                    |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+-------+--------------------+--------------------+

where $\mathbf{P}_1 = \frac{1}{n}\mathbf{J}$, $n = \sum_i n_i$, and $\mathbf{J}$ is the all-ones matrix.

The $F$-statistic has $(a-1, N-a)$ degrees of freedom and the numeric value is unchanged under any of the three parameterizations. The slight difference lies in how we state the null hypothesis:

$$
\begin{aligned}
H_0 &: \mu_1 = \mu_2 = \dots = \mu_a, \\ 
H_0 &: \mu + \tau_1 = \mu + \tau_2 = \dots = \mu + \tau_a, \\ 
H_0 &: \tau_1 = \tau_2 = \dots = \tau_a.
\end{aligned}
$$

The $F$-test here serves as a preliminary analysis, to see if there is any difference at different factors. For more in-depth analysis, we consider different testing of treatment effects.

#### Testing of Treatment Effects

-   [Single Treatment Mean](#sec-single-treatment-mean-anova) $\mu_i$
-   [Differences Between Treatment Means](#sec-differences-between-treatment-means-anova)
-   [Contrast Among Treatment Means](#sec-contrast-among-treatment-means-anova)
-   [Linear Combination of Treatment Means](#sec-linear-combination-of-treatment-means-anova)

##### Single Treatment Mean {#sec-single-treatment-mean-anova}

For a single treatment group, the sample mean serves as an estimate of the population mean:

$$
\hat{\mu_i} = \bar{Y}_{i.} 
$$

where:

-   $E(\bar{Y}_{i.}) = \mu_i$, indicating unbiasedness.
-   $var(\bar{Y}_{i.}) = \frac{\sigma^2}{n_i}$, estimated by $s^2(\bar{Y}_{i.}) = \frac{MSE}{n_i}$.

Since the standardized test statistic

$$
T = \frac{\bar{Y}_{i.} - \mu_i}{s(\bar{Y}_{i.})}
$$

follows a $t$-distribution with $N-a$ degrees of freedom ($t_{N-a}$), a $(1-\alpha)100\%$ confidence interval for $\mu_i$ is:

$$
\bar{Y}_{i.} \pm t_{1-\alpha/2;N-a} s(\bar{Y}_{i.})
$$

To test whether $\mu_i$ is equal to some constant $c$, we set up the hypothesis:

$$
\begin{aligned}
&H_0: \mu_i = c \\
&H_1: \mu_i \neq c
\end{aligned}
$$

The test statistic:

$$
T = \frac{\bar{Y}_{i.} - c}{s(\bar{Y}_{i.})} \sim t_{N-a}
$$

Under $H_0$, we reject $H_0$ at the $\alpha$ level if:

$$
|T| > t_{1-\alpha/2;N-a}
$$

##### Differences Between Treatment Means {#sec-differences-between-treatment-means-anova}

The difference between two treatment means, also called a **pairwise comparison**, is given by:

$$
D = \mu_i - \mu_{i'}
$$

which is estimated by:

$$
\hat{D} = \bar{Y}_{i.} - \bar{Y}_{i'.}
$$

This estimate is unbiased since:

$$
E(\hat{D}) = \mu_i - \mu_{i'}
$$

Since $\bar{Y}_{i.}$ and $\bar{Y}_{i'.}$ are independent, the variance of $\hat{D}$ is:

$$
var(\hat{D}) = var(\bar{Y}_{i.}) + var(\bar{Y}_{i'.}) = \sigma^2 \left(\frac{1}{n_i} + \frac{1}{n_{i'}}\right)
$$

which is estimated by:

$$
s^2(\hat{D}) = MSE \left(\frac{1}{n_i} + \frac{1}{n_{i'}}\right)
$$

Using the same inference structure as the [single treatment mean](#sec-single-treatment-mean-anova):

$$
\frac{\hat{D} - D}{s(\hat{D})} \sim t_{N-a}
$$

A $(1-\alpha)100\%$ confidence interval for $D$ is:

$$
\hat{D} \pm t_{1-\alpha/2;N-a} s(\hat{D})
$$

For hypothesis testing:

$$
\begin{aligned}
&H_0: \mu_i = \mu_{i'} \\
&H_a: \mu_i \neq \mu_{i'}
\end{aligned}
$$

we use the test statistic:

$$
T = \frac{\hat{D}}{s(\hat{D})} \sim t_{N-a}
$$

We reject $H_0$ at the $\alpha$ level if:

$$
|T| > t_{1-\alpha/2;N-a}
$$

##### Contrast Among Treatment Means {#sec-contrast-among-treatment-means-anova}

To generalize the comparison of two means, we introduce **contrasts**.

A **contrast** is a linear combination of treatment means:

$$
L = \sum_{i=1}^{a} c_i \mu_i
$$

where the coefficients $c_i$ are non-random constants that satisfy the constraint:

$$
\sum_{i=1}^{a} c_i = 0
$$

This ensures that contrasts focus on relative comparisons rather than absolute magnitudes.

An unbiased estimator of $L$ is given by:

$$
\hat{L} = \sum_{i=1}^{a} c_i \bar{Y}_{i.}
$$

Since expectation is a linear operator:

$$
E(\hat{L}) = \sum_{i=1}^{a} c_i E(\bar{Y}_{i.}) = \sum_{i=1}^{a} c_i \mu_i = L
$$

Thus, $\hat{L}$ is an unbiased estimator of $L$.

Since the sample means $\bar{Y}_{i.}$ are independent, the variance of $\hat{L}$ is:

$$
\begin{aligned}
var(\hat{L}) &= var\left(\sum_{i=1}^a c_i \bar{Y}_{i.} \right) \\
&= \sum_{i=1}^a c_i^2 var(\bar{Y}_{i.}) \\
&= \sum_{i=1}^a c_i^2 \frac{\sigma^2}{n_i} \\
&= \sigma^2 \sum_{i=1}^{a} \frac{c_i^2}{n_i}
\end{aligned}
$$

Since $\sigma^2$ is unknown, we estimate it using the mean squared error:

$$
s^2(\hat{L}) = MSE \sum_{i=1}^{a} \frac{c_i^2}{n_i}
$$

Since $\hat{L}$ is a linear combination of independent normal random variables, it follows a normal distribution:

$$
\hat{L} \sim N\left(L, \sigma^2 \sum_{i=1}^{a} \frac{c_i^2}{n_i} \right)
$$

Since $SSE/\sigma^2 \sim \chi^2_{N-a}$ and $MSE = SSE/(N-a)$, we use the $t$-distribution:

$$
\frac{\hat{L} - L}{s(\hat{L})} \sim t_{N-a}
$$

Thus, a $(1-\alpha)100\%$ confidence interval for $L$ is:

$$
\hat{L} \pm t_{1-\alpha/2; N-a} s(\hat{L})
$$

To test whether a specific contrast equals zero:

$$
\begin{aligned}
&H_0: L = 0 \quad \text{(no difference in the contrast)} \\
&H_a: L \neq 0 \quad \text{(significant contrast)}
\end{aligned}
$$

We use the test statistic:

$$
T = \frac{\hat{L}}{s(\hat{L})} \sim t_{N-a}
$$

We reject $H_0$ at the $\alpha$ level if:

$$
|T| > t_{1-\alpha/2;N-a}
$$

##### Linear Combination of Treatment Means {#sec-linear-combination-of-treatment-means-anova}

A **linear combination** of treatment means extends the idea of a contrast:

$$
L = \sum_{i=1}^{a} c_i \mu_i
$$

Unlike contrasts, there are **no restrictions** on the coefficients $c_i$ (i.e., they do not need to sum to zero).

Since tests on a [single treatment mean](#sec-single-treatment-mean-anova), [pairwise differences](#sec-differences-between-treatment-means-anova), and [contrasts](#sec-contrast-among-treatment-means-anova) are all special cases of this general form, we can express the hypothesis test as:

$$
\begin{aligned}
&H_0: \sum_{i=1}^{a} c_i \mu_i = c \\
&H_a: \sum_{i=1}^{a} c_i \mu_i \neq c
\end{aligned}
$$

The test statistic follows a $t$-distribution:

$$
T = \frac{\hat{L} - c}{s(\hat{L})} \sim t_{N-a}
$$

Since squaring a $t$-distributed variable results in an $F$-distributed variable,

$$
F = T^2 \sim F_{1,N-a}
$$

This means that all such tests can be viewed as **single-degree-of-freedom** $F$-tests, since the numerator degrees of freedom is always 1.

------------------------------------------------------------------------

**Multiple Contrasts**

When testing $k \geq 2$ contrasts simultaneously, the test statistics $T_1, T_2, ..., T_k$ follow a multivariate $t$-distribution, since they are dependent (as they are based on the same data).

**Limitations of Multiple Comparisons**

1.  **Inflation of Type I Error**:\
    The confidence coefficient $(1-\alpha)$ applies to a single estimate, not a series of estimates. Similarly, the Type I error rate $\alpha$ applies to an individual test, not a collection of tests.

    **Example:** If three $t$-tests are performed at $\alpha = 0.05$, and if they were independent (which they are not), then:

    $$
    (1 - 0.05)^3 = 0.857
    $$

    meaning the overall Type I error rate would be approximately $0.143$, **not** $0.05$.

2.  **Data Snooping Concern**:\
    The significance level $\alpha$ is valid **only if** the test was planned **before** examining the data.

    -   Often, an experiment suggests relationships to investigate.
    -   Exploring effects based on observed data is known as **data snooping**.

To address these issues, we use **Multiple Comparison Procedures**, such as:

-   Tukey -- for all pairwise comparisons of treatment means.
-   Scheffé -- for all possible contrasts.
-   Bonferroni -- for a fixed number of planned comparisons.

------------------------------------------------------------------------

###### Tukey {#sec-tukeys-anova}

Used for all pairwise comparisons of treatment means:

$$
D = \mu_i - \mu_{i'}
$$

Hypothesis test:

$$
\begin{aligned}
&H_0: \mu_i - \mu_{i'} = 0 \\
&H_a: \mu_i - \mu_{i'} \neq 0
\end{aligned}
$$

Properties:

-   When sample sizes are equal ($n_1 = n_2 = ... = n_a$), the family confidence coefficient is exactly $(1-\alpha)$.
-   When sample sizes are unequal, the method is conservative (i.e., the actual significance level is less than $\alpha$).

The Tukey test is based on the **studentized range**:

$$
w = \max(Y_i) - \min(Y_i)
$$

If $Y_1, ..., Y_r$ are observations from a normal distribution with mean $\mu$ and variance $\sigma^2$, then the statistic:

$$
q(r, v) = \frac{w}{s}
$$

follows the studentized range distribution, which requires a special table.

**Notes:**

-   When testing only a subset of pairwise comparisons, the confidence coefficient exceeds $(1-\alpha)$, making the test more conservative.

-   Tukey's method can be used for data snooping, as long as the investigated effects are pairwise comparisons.

------------------------------------------------------------------------

###### Scheffé {#sec-scheffe-anova}

Scheffé's method is used for testing **all possible contrasts**:

$$
L = \sum_{i=1}^{a} c_i \mu_i, \quad \text{where} \quad \sum_{i=1}^{a} c_i = 0
$$

Hypothesis test:

$$
\begin{aligned}
&H_0: L = 0 \\
&H_a: L \neq 0
\end{aligned}
$$

Properties:

-   Valid for any set of contrasts, making it the most general multiple comparison procedure.
-   The family confidence level is exactly $(1-\alpha)$, regardless of sample sizes.

Simultaneous Confidence Intervals:

$$
\hat{L} \pm S s(\hat{L})
$$

where:

-   $\hat{L} = \sum c_i \bar{Y}_{i.}$
-   $s^2(\hat{L}) = MSE \sum \frac{c_i^2}{n_i}$
-   $S^2 = (a-1) f_{1-\alpha; a-1, N-a}$

Test Statistic:

$$
F = \frac{\hat{L}^2}{(a-1) s^2(\hat{L})}
$$

We reject $H_0$ if:

$$
F > f_{1-\alpha; a-1, N-a}
$$

Notes:

-   Finite Family Correction: Since we never test all possible contrasts in practice, the actual confidence coefficient is greater than $(1-\alpha)$. Thus, some researchers use a higher $\alpha$ (e.g., a 90% confidence level instead of 95%).
-   Scheffé is useful for data snooping, since it applies to any contrast.
-   If only pairwise comparisons are needed, Tukey's method gives narrower confidence intervals than Scheffé.

------------------------------------------------------------------------

###### Bonferroni {#sec-bonferroni-anova}

The Bonferroni correction is applicable regardless of whether sample sizes are equal or unequal. It is particularly useful when a small number of planned comparisons are of interest.

A $(1-\alpha)100\%$ simultaneous confidence interval for a set of $g$ comparisons is:

$$
\hat{L} \pm B s(\hat{L})
$$

where:

$$
B = t_{1-\alpha/(2g), N-a}
$$

and $g$ is the **number of comparisons** in the family.

To test:

$$
\begin{aligned}
&H_0: L = 0 \\
&H_a: L \neq 0
\end{aligned}
$$

we use the test statistic:

$$
T = \frac{\hat{L}}{s(\hat{L})}
$$

Reject $H_0$ if:

$$
|T| > t_{1-\alpha/(2g),N-a}
$$

Notes:

-   If all pairwise comparisons are needed, Tukey's method is superior, as it provides narrower confidence intervals.
-   Bonferroni is better than Scheffé when the number of contrasts is similar to or smaller than the number of treatment levels.
-   Practical recommendation: Compute Tukey, Scheffé, and Bonferroni and use the method with the smallest confidence intervals.
-   Bonferroni cannot be used for data snooping, as it assumes the comparisons were planned before examining the data.

------------------------------------------------------------------------

###### Fisher's Least Significant Difference

The Fisher LSD method does not control the family-wise error rate (refer to \@ref(sec-false-discovery-rate)), meaning it does not correct for multiple comparisons. However, it can be useful for exploratory analysis when a preliminary ANOVA is significant.

The hypothesis test for comparing two treatment means:

$$
H_0: \mu_i = \mu_j
$$

uses the $t$-statistic:

$$
t = \frac{\bar{Y}_i - \bar{Y}_j}{\sqrt{MSE \left(\frac{1}{n_i} + \frac{1}{n_j}\right)}}
$$

where:

-   $\bar{Y}_i$ and $\bar{Y}_j$ are the sample means for treatments $i$ and $j$.

-   $MSE$ is the mean squared error from ANOVA.

-   $n_i, n_j$ are the sample sizes for groups $i$ and $j$.

Notes:

-   The LSD method does not adjust for multiple comparisons, which increases the Type I error rate.
-   It is only valid if the overall ANOVA is significant (i.e., the global null hypothesis of no treatment effect is rejected).
-   Tukey and Bonferroni methods are preferred when many comparisons are made.

------------------------------------------------------------------------

###### Newman-Keuls

The Newman-Keuls procedure is a stepwise multiple comparison test similar to Tukey's method but less rigorous.

Key Issues:

-   Unlike Tukey, Newman-Keuls does not control the family-wise error rate.
-   It has less power than ANOVA.
-   It is rarely recommended in modern statistical practice.
-   **Do not recommend using the Newman-Keuls test**.

###### Summary of Multiple Comparison Procedures

+--------------+-------------------------------------+----------------------------------+------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Method       | Type of Comparisons                 | Controls Family-Wise Error Rate? | Best Used For                                                    | Strengths                                                                                         | Weaknesses                                                                         |
+==============+=====================================+==================================+==================================================================+===================================================================================================+====================================================================================+
| **Tukey**    | All pairwise comparisons            | Yes                              | Comparing all treatment means                                    | Exact confidence level when sample sizes are equal; more powerful than Scheffé for pairwise tests | Conservative if sample sizes are unequal                                           |
+--------------+-------------------------------------+----------------------------------+------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Scheffé      | All possible contrasts              | Yes                              | Exploratory analysis, especially when interested in any contrast | Valid for any contrast; can be used for data snooping                                             | Confidence intervals wider than Tukey for pairwise comparisons                     |
+--------------+-------------------------------------+----------------------------------+------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Bonferroni   | Fixed number of planned comparisons | Yes                              | A small number of pre-specified tests                            | Simple and flexible; better than Scheffé for few comparisons                                      | Less powerful than Tukey for many pairwise tests; cannot be used for data snooping |
+--------------+-------------------------------------+----------------------------------+------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Fisher's LSD | Pairwise comparisons                | No                               | Exploratory comparisons after significant ANOVA                  | Most powerful for pairwise comparisons when ANOVA is significant                                  | Inflates Type I error rate; not valid without a significant ANOVA                  |
+--------------+-------------------------------------+----------------------------------+------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Newman-Keuls | Pairwise comparisons                | No                               | \-                                                               | \-                                                                                                | Less power than ANOVA; generally not recommended                                   |
+--------------+-------------------------------------+----------------------------------+------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

###### Dunnett's Test

In some experiments, instead of comparing all treatment groups against each other, we are specifically interested in comparing each treatment to a control. This is common in clinical trials or A/B testing, where one group serves as a baseline.

Dunnett's test is designed for experiments with $a$ groups, where:

-   One group is the control (e.g., placebo or standard treatment).

-   The remaining $a-1$ groups are treatment groups.

Thus, we perform $a-1$ pairwise comparisons:

$$
D_i = \mu_i - \mu_c, \quad i = 1, \dots, a-1
$$

where $\mu_c$ is the mean of the control group.

**Dunnett's Test vs. Other Methods**

-   Unlike Tukey's method (which compares all pairs), Dunnett's method only compares treatments to the control.
-   Dunnett's test controls the family-wise error rate, making it more powerful than Bonferroni for this scenario.
-   If the goal is to compare treatments against each other as well, Tukey's method is preferable.

------------------------------------------------------------------------

### Single Factor Random Effects ANOVA {#sec-single-factor-random-effects-model}

Also known as an **ANOVA Type II model**, the single factor random effects model assumes that treatments are randomly selected from a larger population. Thus, inference extends beyond the observed treatments to the entire population of treatments.

------------------------------------------------------------------------

#### Random Cell Means Model

The model is given by:

$$
Y_{ij} = \mu_i + \epsilon_{ij}
$$

where:

-   $\mu_i \sim N(\mu, \sigma^2_{\mu})$, independent across treatments.
-   $\epsilon_{ij} \sim N(0, \sigma^2)$, independent across observations.
-   $\mu_i$ and $\epsilon_{ij}$ are mutually independent for $i = 1, \dots, a$ and $j = 1, \dots, n$.

When all treatment sample sizes are equal:

$$
\begin{aligned}
E(Y_{ij}) &= E(\mu_i) = \mu \\
var(Y_{ij}) &= var(\mu_i) + var(\epsilon_{ij}) = \sigma^2_{\mu} + \sigma^2
\end{aligned}
$$

------------------------------------------------------------------------

##### Covariance Structure

Since $Y_{ij}$ are not independent, we calculate their covariances:

1.  **Same treatment group (**$i$ fixed, $j \neq j'$):

$$
\begin{aligned}
cov(Y_{ij}, Y_{ij'}) &= E(Y_{ij} Y_{ij'}) - E(Y_{ij}) E(Y_{ij'}) \\
&= E(\mu_i^2 + \mu_i \epsilon_{ij'} + \mu_i \epsilon_{ij} + \epsilon_{ij} \epsilon_{ij'}) - \mu^2 \\
&= \sigma^2_{\mu} + \mu^2 - \mu^2 \\
&= \sigma^2_{\mu}
\end{aligned}
$$

2.  **Different treatment groups (**$i \neq i'$):

$$
\begin{aligned}
cov(Y_{ij}, Y_{i'j'}) &= E(\mu_i \mu_{i'} + \mu_i \epsilon_{i'j'} + \mu_{i'} \epsilon_{ij} + \epsilon_{ij} \epsilon_{i'j'}) - \mu^2 \\
&= \mu^2 - \mu^2 = 0
\end{aligned}
$$

Thus:

-   All observations have the same variance: $var(Y_{ij}) = \sigma^2_{\mu} + \sigma^2$.
-   Observations from the same treatment have covariance: $\sigma^2_{\mu}$.
-   Observations from different treatments are uncorrelated.

The **intraclass correlation** between two responses from the same treatment:

$$
\rho(Y_{ij}, Y_{ij'}) = \frac{\sigma^2_{\mu}}{\sigma^2_{\mu} + \sigma^2}, \quad j \neq j'
$$

------------------------------------------------------------------------

##### Inference for Random Effects Model

The **Intraclass Correlation Coefficient**:

$$
\frac{\sigma^2_{\mu}}{\sigma^2 + \sigma^2_{\mu}}
$$

measures the proportion of total variability in $Y_{ij}$ that is accounted for by treatment differences.

To test whether treatments contribute significantly to variance:

$$
\begin{aligned}
&H_0: \sigma_{\mu}^2 = 0 \quad \text{(No treatment effect, all $\mu_i = \mu$)} \\
&H_a: \sigma_{\mu}^2 \neq 0
\end{aligned}
$$

Under $H_0$, an ANOVA F-test is used:

$$
F = \frac{MSTR}{MSE}
$$

where:

-   $MSTR$ (Mean Square for Treatments) captures variation **between treatments**.
-   $MSE$ (Mean Square Error) captures variation **within treatments**.

If $H_0$ is true, then:

$$
F \sim F_{(a-1, a(n-1))}
$$

Reject $H_0$ if:

$$
F > f_{(1-\alpha; a-1, a(n-1))}
$$

------------------------------------------------------------------------

##### Comparison: Fixed Effects vs. Random Effects Models

Although ANOVA calculations are the same for [fixed](#sec-single-factor-fixed-effects-model) and [random effects](#sec-single-factor-random-effects-model) models, the interpretation of results differs.

+-----------------------------------------+----------------------------------------------------------------+
| **Random Effects Model**                | **Fixed Effects Model**                                        |
+=========================================+================================================================+
| $E(MSE) = \sigma^2$                     | $E(MSE) = \sigma^2$                                            |
+-----------------------------------------+----------------------------------------------------------------+
| $E(MSTR) = \sigma^2 + n \sigma^2_{\mu}$ | $E(MSTR) = \sigma^2 + \frac{ \sum_i n_i (\mu_i - \mu)^2}{a-1}$ |
+-----------------------------------------+----------------------------------------------------------------+

-   If $\sigma^2_{\mu} = 0$, then $E(MSTR) = E(MSE)$, implying no treatment effect.
-   Otherwise, $E(MSTR) > E(MSE)$, suggesting significant treatment variation.

When sample sizes **are not equal**, the $F$-test remains valid, but the degrees of freedom change to:

$$
F \sim F_{(a-1, N-a)}
$$

------------------------------------------------------------------------

##### Estimation of $\mu$

An unbiased estimator of $E(Y_{ij}) = \mu$ is the **grand mean**:

$$
\hat{\mu} = \bar{Y}_{..} = \frac{1}{a n} \sum_{i=1}^{a} \sum_{j=1}^{n} Y_{ij}
$$

The variance of this estimator is:

$$
\begin{aligned}
var(\bar{Y}_{..}) &= var\left(\frac{1}{a} \sum_{i=1}^{a} \bar{Y}_{i.} \right) \\
&= \frac{1}{a^2} \sum_{i=1}^{a} var(\bar{Y}_{i.}) \\
&= \frac{1}{a^2} \sum_{i=1}^{a} \left(\sigma^2_\mu + \frac{\sigma^2}{n} \right) \\
&= \frac{n \sigma^2_{\mu} + \sigma^2}{a n}
\end{aligned}
$$

An unbiased estimator of this variance is:

$$
s^2(\bar{Y}_{..}) = \frac{MSTR}{a n}
$$

Since:

$$
\frac{\bar{Y}_{..} - \mu}{s(\bar{Y}_{..})} \sim t_{a-1}
$$

A $(1-\alpha)100\%$ confidence interval for $\mu$ is:

$$
\bar{Y}_{..} \pm t_{1-\alpha/2; a-1} s(\bar{Y}_{..})
$$

------------------------------------------------------------------------

##### Estimation of Intraclass Correlation Coefficient $\frac{\sigma^2_\mu}{\sigma^2_{\mu}+\sigma^2}$

In both [random](#sec-single-factor-random-effects-model) and [fixed](#sec-single-factor-fixed-effects-model) effects models, $MSTR$ and $MSE$ are **independent**.

When sample sizes are equal ($n_i = n$ for all $i$), the test statistic:

$$
\frac{\frac{MSTR}{n\sigma^2_\mu + \sigma^2}}{\frac{MSE}{\sigma^2}} \sim F_{a-1, a(n-1)}
$$

A $(1-\alpha)100\%$ confidence interval for $\frac{\sigma^2_\mu}{\sigma^2_\mu + \sigma^2}$ follows from:

$$
P\left(f_{\alpha/2; a-1, a(n-1)} \leq \frac{\frac{MSTR}{n\sigma^2_\mu + \sigma^2}}{\frac{MSE}{\sigma^2}} \leq f_{1-\alpha/2; a-1, a(n-1)} \right) = 1 - \alpha
$$

Defining:

$$
\begin{aligned}
L &= \frac{1}{n} \left( \frac{MSTR}{MSE} \times \frac{1}{f_{1-\alpha/2; a-1, a(n-1)}} - 1 \right) \\
U &= \frac{1}{n} \left( \frac{MSTR}{MSE} \times \frac{1}{f_{\alpha/2; a-1, a(n-1)}} - 1 \right)
\end{aligned}
$$

The lower and upper confidence limits for $\frac{\sigma^2_\mu}{\sigma^2_\mu + \sigma^2}$ are:

$$
\begin{aligned}
L^* &= \frac{L}{1+L} \\
U^* &= \frac{U}{1+U}
\end{aligned}
$$

If $L^*$ is negative, we customarily set it to 0.

------------------------------------------------------------------------

##### Estimation of $\sigma^2$

Since:

$$
\frac{a(n-1) MSE}{\sigma^2} \sim \chi^2_{a(n-1)}
$$

A $(1-\alpha)100\%$ confidence interval for $\sigma^2$ is:

$$
\frac{a(n-1) MSE}{\chi^2_{1-\alpha/2; a(n-1)}} \leq \sigma^2 \leq \frac{a(n-1) MSE}{\chi^2_{\alpha/2; a(n-1)}}
$$

If sample sizes are unequal, the same formula applies, but the degrees of freedom change to:

$$
df = N - a
$$

------------------------------------------------------------------------

##### Estimation of $\sigma^2_\mu$

From the expectations:

$$
E(MSE) = \sigma^2, \quad E(MSTR) = \sigma^2 + n\sigma^2_\mu
$$

we solve for $\sigma^2_{\mu}$:

$$
\sigma^2_{\mu} = \frac{E(MSTR) - E(MSE)}{n}
$$

An unbiased estimator of $\sigma^2_\mu$ is:

$$
s^2_\mu = \frac{MSTR - MSE}{n}
$$

If $s^2_\mu < 0$, we set $s^2_\mu = 0$ (since variances cannot be negative).

If sample sizes are unequal, we replace $n$ with an effective sample size $n'$:

$$
s^2_\mu = \frac{MSTR - MSE}{n'}
$$

where:

$$
n' = \frac{1}{a-1} \left(\sum_i n_i - \frac{\sum_i n_i^2}{\sum_i n_i} \right)
$$

------------------------------------------------------------------------

There are no exact confidence intervals for $\sigma^2_\mu$, but we can approximate them using the Satterthwaite procedure.

###### Satterthwaite Approximation {#sec-satterthwaite-approximation}

A linear combination of expected mean squares:

$$
\sigma^2_\mu = \frac{1}{n} E(MSTR) + \left(-\frac{1}{n}\right) E(MSE)
$$

For a general linear combination:

$$
S = d_1 E(MS_1) + \dots + d_h E(MS_h)
$$

where $d_i$ are coefficients, an unbiased estimator of $S$ is:

$$
\hat{S} = d_1 MS_1 + \dots + d_h MS_h
$$

Let $df_i$ be the degrees of freedom associated with each mean square $MS_i$. The Satterthwaite approximation states:

$$
\frac{(df) \hat{S}}{S} \sim \chi^2_{df}
$$

where the degrees of freedom are approximated as:

$$
df = \frac{(d_1 MS_1 + \dots + d_h MS_h)^2}{\sum_{i=1}^{h} \frac{(d_i MS_i)^2}{df_i}}
$$

------------------------------------------------------------------------

Applying the Satterthwaite method to the [single factor random effects model](#sec-single-factor-random-effects-model):

$$
\frac{(df) s^2_\mu}{\chi^2_{1-\alpha/2; df}} \leq \sigma^2_\mu \leq \frac{(df) s^2_\mu}{\chi^2_{\alpha/2; df}}
$$

where the approximate degrees of freedom are:

$$
df = \frac{(s^2_\mu)^2}{\frac{(MSTR)^2}{a-1} + \frac{(MSE)^2}{a(n-1)}}
$$

------------------------------------------------------------------------

#### Random Treatment Effects Model

In a random effects model, treatment levels are considered random samples from a larger population of possible treatments. The model accounts for variability across all potential treatments, not just those observed in the study.

We define the **random treatment effect** as:

$$
\tau_i = \mu_i - E(\mu_i) = \mu_i - \mu
$$

where $\tau_i$ represents the deviation of treatment mean $\mu_i$ from the overall mean $\mu$.

Thus, we rewrite treatment means as:

$$
\mu_i = \mu + \tau_i
$$

Substituting this into the response model:

$$
Y_{ij} = \mu + \tau_i + \epsilon_{ij}
$$

where:

-   $\mu$ = common mean across all observations.
-   $\tau_i \sim N(0, \sigma^2_\tau)$, random treatment effects, assumed independent.
-   $\epsilon_{ij} \sim N(0, \sigma^2)$, random error terms, also independent.
-   $\tau_{i}$ and $\epsilon_{ij}$ are mutually independent for $i = 1, \dots, a$ and $j = 1, \dots, n$.
-   We consider only balanced single-factor ANOVA (equal sample sizes across treatments).

------------------------------------------------------------------------

#### Diagnostic Measures for Model Assumptions

Checking assumptions is crucial for valid inference. Common issues include:

+------------------------------------------------------+---------------------------------------------------------------------------------+
| **Issue**                                            | **Diagnostic Tools**                                                            |
+======================================================+=================================================================================+
| **Non-constant error variance (heteroscedasticity)** | Residual plots, Levene's test, Hartley's test                                   |
+------------------------------------------------------+---------------------------------------------------------------------------------+
| **Non-independence of errors**                       | Residual plots, Durbin-Watson test (for autocorrelation)                        |
+------------------------------------------------------+---------------------------------------------------------------------------------+
| **Outliers**                                         | Boxplots, residual plots, regression influence measures (e.g., Cook's distance) |
+------------------------------------------------------+---------------------------------------------------------------------------------+
| **Non-normality of errors**                          | Histogram, Q-Q plot, Shapiro-Wilk test, Anderson-Darling test                   |
+------------------------------------------------------+---------------------------------------------------------------------------------+
| **Omitted variable bias**                            | Residual plots, checking for unaccounted sources of variation                   |
+------------------------------------------------------+---------------------------------------------------------------------------------+

------------------------------------------------------------------------

#### Remedial Measures

If diagnostic checks indicate violations of assumptions, possible solutions include:

-   [Weighted Least Squares] -- Adjusts for heteroscedasticity.
-   [Variable Transformation] -- Log or Box-Cox transformations may improve normality or stabilize variance.
-   [Non-Parametric Procedures](#sec-nonparametric-anova) -- Kruskal-Wallis test or bootstrapping when normality assumptions fail.

------------------------------------------------------------------------

#### Key Notes on Robustness

-   Fixed effects ANOVA is relatively robust to:
    -   Non-normality, particularly when sample sizes are moderate to large.
    -   Unequal variances when sample sizes are roughly equal.
    -   F-test and multiple comparisons remain valid under mild violations.
-   Random effects ANOVA is sensitive to:
    -   Lack of independence, which severely affects both fixed and random effects models.
    -   Unequal variances, particularly when estimating variance components.

------------------------------------------------------------------------

### Two-Factor Fixed Effects ANOVA {#sec-two-factor-fixed-effects-anova}

A multi-factor experiment offers several advantages:

-   Higher efficiency -- More precise estimates with fewer observations.
-   Increased information -- Allows for testing interactions between factors.
-   Greater validity -- Reduces confounding by controlling additional sources of variation.

**Balanced Two-Factor ANOVA: Assumptions**

-   Equal sample sizes for all treatment combinations.
-   All treatment means are of equal importance (no weighting).
-   Factors are categorical and chosen purposefully.

We assume:

-   Factor A has $a$ levels and Factor B has $b$ levels.
-   All $a \times b$ factor level combinations are included.
-   Each treatment combination has $n$ replications.
-   The total number of observations:\
    $$ N = abn $$

------------------------------------------------------------------------

#### Cell Means Model {#sec-cell-means-model-two-factor-anova}

The response is modeled as:

$$
Y_{ijk} = \mu_{ij} + \epsilon_{ijk}
$$

where:

-   $\mu_{ij}$ are fixed parameters (cell means).
-   $i = 1, \dots, a$ represents levels of Factor A.
-   $j = 1, \dots, b$ represents levels of Factor B.
-   $\epsilon_{ijk} \sim \text{independent } N(0, \sigma^2)$ for all $i, j, k$.

**Expected values and variance**:

$$
\begin{aligned}
E(Y_{ijk}) &= \mu_{ij} \\
var(Y_{ijk}) &= var(\epsilon_{ijk}) = \sigma^2
\end{aligned}
$$

Thus:

$$
Y_{ijk} \sim \text{independent } N(\mu_{ij}, \sigma^2)
$$

This can be expressed in matrix notation:

$$
\mathbf{Y} = \mathbf{X} \beta + \epsilon
$$

where:

$$
\begin{aligned}
E(\mathbf{Y}) &= \mathbf{X} \beta \\
var(\mathbf{Y}) &= \sigma^2 \mathbf{I}
\end{aligned}
$$

------------------------------------------------------------------------

##### Interaction Effects

Interaction measures whether the effect of one factor depends on the level of the other factor. It is defined as:

$$
(\alpha \beta)_{ij} = \mu_{ij} - (\mu_{..} + \alpha_i + \beta_j)
$$

where:

-   **Grand mean**:\
    $$ \mu_{..} = \frac{1}{ab} \sum_i \sum_j \mu_{ij} $$
-   **Main effect for Factor A** (average effect of level $i$):\
    $$ \alpha_i = \mu_{i.} - \mu_{..} $$
-   **Main effect for Factor B** (average effect of level $j$):\
    $$ \beta_j = \mu_{.j} - \mu_{..} $$
-   **Interaction effect**:\
    $$ (\alpha \beta)_{ij} = \mu_{ij} - \mu_{i.} - \mu_{.j} + \mu_{..} $$

------------------------------------------------------------------------

To determine whether interactions exist:

1.  Check if all $\mu_{ij}$ can be written as sums $\mu_{..} + \alpha_i + \beta_j$\
    (i.e., check if interaction terms are zero).
2.  Compare mean differences across levels of Factor B at each level of Factor A.
3.  Compare mean differences across levels of Factor A at each level of Factor B.
4.  Graphical method:
    -   Plot treatment means for each level of Factor B.
    -   If lines are not parallel, an interaction exists.

------------------------------------------------------------------------

The interaction terms satisfy:

For each level of **Factor B**:

$$
\sum_i (\alpha \beta)_{ij} = \sum_i \left(\mu_{ij} - \mu_{..} - \alpha_i - \beta_j \right)
$$

Expanding:

$$
\begin{aligned}
\sum_i (\alpha \beta)_{ij} &= \sum_i \mu_{ij} - a \mu_{..} - \sum_i \alpha_i - a \beta_j \\
&= a \mu_{.j} - a \mu_{..} - \sum_i (\mu_{i.} - \mu_{..}) - a (\mu_{.j} - \mu_{..}) \\
&= a \mu_{.j} - a \mu_{..} - a \mu_{..}+ a \mu_{..} - a (\mu_{.j} - \mu_{..})  \\
&= 0
\end{aligned}
$$

Similarly:

$$
\sum_j (\alpha \beta)_{ij} = 0, \quad i = 1, \dots, a
$$

and:

$$
\sum_i \sum_j (\alpha \beta)_{ij} = 0, \quad \sum_i \alpha_i = 0, \quad \sum_j \beta_j = 0
$$

------------------------------------------------------------------------

#### Factor Effects Model {#sec-factor-effects-model-two-factor-anova}

In the Factor Effects Model, we express the response as:

$$
\begin{aligned}
\mu_{ij} &= \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} \\
Y_{ijk} &= \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ijk}
\end{aligned}
$$

where:

-   $\mu_{..}$ is the grand mean.
-   $\alpha_i$ are main effects for Factor A, subject to:\
    $$ \sum_i \alpha_i = 0 $$
-   $\beta_j$ are main effects for Factor B, subject to:\
    $$ \sum_j \beta_j = 0 $$
-   $(\alpha \beta)_{ij}$ are interaction effects, subject to:\
    $$ \sum_i (\alpha \beta)_{ij} = 0, \quad j = 1, \dots, b $$\
    $$ \sum_j (\alpha \beta)_{ij} = 0, \quad i = 1, \dots, a $$
-   $\epsilon_{ijk} \sim \text{independent } N(0, \sigma^2)$ for $k = 1, \dots, n$.

Thus, we have:

$$
\begin{aligned}
E(Y_{ijk}) &= \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} \\
var(Y_{ijk}) &= \sigma^2 \\
Y_{ijk} &\sim N (\mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij}, \sigma^2)
\end{aligned}
$$

------------------------------------------------------------------------

#### Parameter Counting and Restrictions

The [Cell Means Model](#sec-cell-means-model-two-factor-anova) has $ab$ parameters corresponding to each combination of factor levels.\
In the [Factor Effects Model](#sec-factor-effects-model-two-factor-anova), the imposed constraints reduce the number of estimable parameters:

+---------------------------------------------+-------------------------------------------------+
| Parameter                                   | Count                                           |
+=============================================+=================================================+
| $\mu_{..}$                                  | $1$                                             |
+---------------------------------------------+-------------------------------------------------+
| $\alpha_i$ (Main effects for A)             | $a-1$ (due to constraint $\sum_i \alpha_i = 0$) |
+---------------------------------------------+-------------------------------------------------+
| $\beta_j$ (Main effects for B)              | $b-1$ (due to constraint $\sum_j \beta_j = 0$)  |
+---------------------------------------------+-------------------------------------------------+
| $(\alpha \beta)_{ij}$ (Interaction effects) | $(a-1)(b-1)$ (due to two constraints)           |
+---------------------------------------------+-------------------------------------------------+

Thus, the total number of parameters:

$$
1 + (a-1) + (b-1) + (a-1)(b-1) = ab
$$

which matches the number of parameters in the [Cell Means Model](#sec-cell-means-model-two-factor-anova).

------------------------------------------------------------------------

To uniquely estimate parameters, we apply constraints:

$$
\begin{aligned}
\alpha_a  &= -(\alpha_1 + \alpha_2 + \dots + \alpha_{a-1}) \\
\beta_b &= -(\beta_1 + \beta_2 + \dots + \beta_{b-1}) \\
(\alpha \beta)_{ib} &= -(\alpha \beta)_{i1} - (\alpha \beta)_{i2} - \dots - (\alpha \beta)_{i,b-1}, \quad i = 1, \dots, a \\
(\alpha \beta)_{aj} &= -(\alpha \beta)_{1j} - (\alpha \beta)_{2j} - \dots - (\alpha \beta)_{a-1,j}, \quad j = 1, \dots, b
\end{aligned}
$$

The model can be fitted using least squares or maximum likelihood estimation.

------------------------------------------------------------------------

##### Cell Means Model Estimation

Minimizing:

$$
Q = \sum_i \sum_j \sum_k (Y_{ijk} - \mu_{ij})^2
$$

yields estimators:

$$
\begin{aligned}
\hat{\mu}_{ij} &= \bar{Y}_{ij} \\
\hat{Y}_{ijk} &= \bar{Y}_{ij} \\
e_{ijk} &= Y_{ijk} - \hat{Y}_{ijk} = Y_{ijk} - \bar{Y}_{ij}
\end{aligned}
$$

where $e_{ijk} \sim \text{independent } N(0, \sigma^2)$.

------------------------------------------------------------------------

##### Factor Effects Model Estimation

Minimizing:

$$
Q = \sum_i \sum_j \sum_k (Y_{ijk} - \mu_{..} - \alpha_i - \beta_j - (\alpha \beta)_{ij})^2
$$

subject to the constraints:

$$
\begin{aligned}
\sum_i \alpha_i &= 0 \\
\sum_j \beta_j &= 0 \\
\sum_i (\alpha \beta)_{ij} &= 0, \quad j = 1, \dots, b \\
\sum_j (\alpha \beta)_{ij} &= 0, \quad i = 1, \dots, a
\end{aligned}
$$

yields estimators:

$$
\begin{aligned}
\hat{\mu}_{..} &= \bar{Y}_{...} \\
\hat{\alpha}_i &= \bar{Y}_{i..} - \bar{Y}_{...} \\
\hat{\beta}_j &= \bar{Y}_{.j.} - \bar{Y}_{...} \\
(\hat{\alpha \beta})_{ij} &= \bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...}
\end{aligned}
$$

------------------------------------------------------------------------

The fitted values are:

$$
\hat{Y}_{ijk} = \bar{Y}_{...} + (\bar{Y}_{i..} - \bar{Y}_{...}) + (\bar{Y}_{.j.} - \bar{Y}_{...}) + (\bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...})
$$

which simplifies to:

$$
\hat{Y}_{ijk} = \bar{Y}_{ij.}
$$

The residuals are:

$$
e_{ijk} = Y_{ijk} - \bar{Y}_{ij.}
$$

and follow:

$$
e_{ijk} \sim \text{independent } N(0, \sigma^2)
$$

------------------------------------------------------------------------

The variances of the estimated effects are:

$$
\begin{aligned}
s^2_{\hat{\mu}_{..}} &= \frac{MSE}{nab} \\
s^2_{\hat{\alpha}_i} &= MSE \left(\frac{1}{nb} - \frac{1}{nab} \right) \\
s^2_{\hat{\beta}_j} &= MSE \left(\frac{1}{na} - \frac{1}{nab} \right) \\
s^2_{(\hat{\alpha\beta})_{ij}} &= MSE \left(\frac{1}{n} - \frac{1}{na} - \frac{1}{nb} + \frac{1}{nab} \right)
\end{aligned}
$$

------------------------------------------------------------------------

##### Partitioning the Total Sum of Squares

The total deviation of an observation from the overall mean can be decomposed as:

$$
Y_{ijk} - \bar{Y}_{...} = (\bar{Y}_{ij.} - \bar{Y}_{...}) + (Y_{ijk} - \bar{Y}_{ij.})
$$

where:

-   $Y_{ijk} - \bar{Y}_{...}$: Total deviation of an observation.
-   $\bar{Y}_{ij.} - \bar{Y}_{...}$: Deviation of treatment mean from the overall mean.
-   $Y_{ijk} - \bar{Y}_{ij.}$: Residual deviation of an observation from the treatment mean.

Summing over all observations:

$$
\sum_i \sum_j \sum_k (Y_{ijk} - \bar{Y}_{...})^2 = n \sum_i \sum_j (\bar{Y}_{ij.} - \bar{Y}_{...})^2 + \sum_i \sum_j \sum_k (Y_{ijk} - \bar{Y}_{ij.})^2
$$

Thus:

$$
SSTO = SSTR + SSE
$$

where:

-   $SSTO$ = Total Sum of Squares (Total variation).
-   $SSTR$ = Treatment Sum of Squares (Variation due to factor effects).
-   $SSE$ = Error Sum of Squares (Residual variation).

Since the cross-product terms are 0, the model naturally partitions the variance.

From the [factor effects model](#sec-factor-effects-model-two-factor-anova):

$$
\bar{Y}_{ij.} - \bar{Y}_{...} = (\bar{Y}_{i..} - \bar{Y}_{...}) + (\bar{Y}_{.j.} - \bar{Y}_{...}) + (\bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...})
$$

Squaring and summing:

$$
\begin{aligned}
n\sum_i \sum_j (\bar{Y}_{ij.} - \bar{Y}_{...})^2 &= nb\sum_i (\bar{Y}_{i..} - \bar{Y}_{...})^2 + na\sum_j (\bar{Y}_{.j.} - \bar{Y}_{...})^2 \\
&+ n\sum_i \sum_j (\bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...})^2
\end{aligned}
$$

Thus, treatment sum of squares can be further partitioned as:

$$
SSTR = SSA + SSB + SSAB
$$

where:

-   $SSA$: Sum of Squares for Factor A.
-   $SSB$: Sum of Squares for Factor B.
-   $SSAB$: Sum of Squares for Interaction.

The interaction term can also be expressed as:

$$
SSAB = SSTO - SSE - SSA - SSB
$$

or equivalently:

$$
SSAB = SSTR - SSA - SSB
$$

where:

-   $SSA$ measures the variability of the estimated factor **A** level means ($\bar{Y}_{i..}$). The more variable these means, the larger $SSA$.
-   $SSB$ measures the variability of the estimated factor **B** level means ($\bar{Y}_{.j.}$).
-   $SSAB$ measures the variability in **interaction effects**.

------------------------------------------------------------------------

For [Two-Factor ANOVA](#sec-two-factor-fixed-effects-anova), the degrees of freedom partitioning follows:

| **Sum of Squares**   | **Degrees of Freedom (df)** |
|----------------------|-----------------------------|
| $SSTO$ (Total)       | $N - 1 = abn - 1$           |
| $SSTR$ (Treatments)  | $ab - 1$                    |
| $SSE$ (Error)        | $N - ab = ab(n - 1)$        |
| $SSA$ (Factor A)     | $a - 1$                     |
| $SSB$ (Factor B)     | $b - 1$                     |
| $SSAB$ (Interaction) | $(a-1)(b-1)$                |

Since:

$$
SSTR = SSA + SSB + SSAB
$$

the treatment degrees of freedom also partition as:

$$
ab - 1 = (a - 1) + (b - 1) + (a - 1)(b - 1)
$$

-   $df_{SSA} = a - 1$\
    (One degree of freedom lost due to the constraint $\sum (\bar{Y}_{i..} - \bar{Y}_{...}) = 0$).
-   $df_{SSB} = b - 1$\
    (One degree of freedom lost due to the constraint $\sum (\bar{Y}_{.j.} - \bar{Y}_{...}) = 0$).
-   $df_{SSAB} = (a - 1)(b - 1)$\
    (Due to interaction constraints).

------------------------------------------------------------------------

The Mean Squares are obtained by dividing Sum of Squares by the corresponding degrees of freedom:

$$
\begin{aligned}
MSA &= \frac{SSA}{a - 1} \\
MSB &= \frac{SSB}{b - 1} \\
MSAB &= \frac{SSAB}{(a - 1)(b - 1)}
\end{aligned}
$$

The expectations of the mean squares are:

$$
\begin{aligned}
E(MSE) &= \sigma^2 \\
E(MSA) &= \sigma^2 + nb \frac{\sum \alpha_i^2}{a - 1} = \sigma^2 + nb \frac{\sum (\mu_{i..} - \mu_{..})^2}{a - 1} \\
E(MSB) &= \sigma^2 + na \frac{\sum \beta_j^2}{b - 1} = \sigma^2 + na \frac{\sum (\mu_{.j.} - \mu_{..})^2}{b - 1} \\
E(MSAB) &= \sigma^2 + n \frac{\sum \sum (\alpha \beta)^2_{ij}}{(a-1)(b-1)} = \sigma^2 + n \frac{\sum (\mu_{ij} - \mu_{i..} - \mu_{.j.} + \mu_{..})^2}{(a - 1)(b - 1)}
\end{aligned}
$$

If Factor A has no effect ($\mu_{i..} = \mu_{..}$), then $MSA$ and $MSE$ have the same expectation.\
Similarly, if Factor B has no effect, then $MSB = MSE$.

Thus, MSA \> MSE and MSB \> MSE suggest the presence of factor effects.

------------------------------------------------------------------------

#### Testing for Interaction

Hypotheses:

$$
\begin{aligned}
H_0: \mu_{ij} - \mu_{i..} - \mu_{.j.} + \mu_{..} = 0 &\quad \text{(No interaction)} \\
H_a: \mu_{ij} - \mu_{i..} - \mu_{.j.} + \mu_{..} \neq 0 &\quad \text{(Interaction present)}
\end{aligned}
$$

or equivalently:

$$
\begin{aligned}
&H_0: \text{All } (\alpha \beta)_{ij} = 0 \\
&H_a: \text{Not all } (\alpha \beta)_{ij} = 0
\end{aligned}
$$

The F-statistic is:

$$
F = \frac{MSAB}{MSE}
$$

Under $H_0$, $F \sim F_{(a-1)(b-1), ab(n-1)}$. Reject $H_0$ if:

$$
F > F_{1-\alpha; (a-1)(b-1), ab(n-1)}
$$

------------------------------------------------------------------------

#### Two-Way ANOVA Summary Table

The Two-Way ANOVA table partitions the total variation into its components:

+-------------------------+-------------------------+-----------------------------+----------------------------------+-----------------------------+
| **Source of Variation** | **Sum of Squares (SS)** | **Degrees of Freedom (df)** | **Mean Square (MS)**             | **F-Statistic**             |
+=========================+=========================+=============================+==================================+=============================+
| **Factor A**            | $SSA$                   | $a-1$                       | $MSA = \frac{SSA}{a-1}$          | $F_A = \frac{MSA}{MSE}$     |
+-------------------------+-------------------------+-----------------------------+----------------------------------+-----------------------------+
| **Factor B**            | $SSB$                   | $b-1$                       | $MSB = \frac{SSB}{b-1}$          | $F_B = \frac{MSB}{MSE}$     |
+-------------------------+-------------------------+-----------------------------+----------------------------------+-----------------------------+
| **Interaction (A × B)** | $SSAB$                  | $(a-1)(b-1)$                | $MSAB = \frac{SSAB}{(a-1)(b-1)}$ | $F_{AB} = \frac{MSAB}{MSE}$ |
+-------------------------+-------------------------+-----------------------------+----------------------------------+-----------------------------+
| **Error**               | $SSE$                   | $ab(n-1)$                   | $MSE = \frac{SSE}{ab(n-1)}$      | \-                          |
+-------------------------+-------------------------+-----------------------------+----------------------------------+-----------------------------+
| **Total (corrected)**   | $SSTO$                  | $abn - 1$                   | \-                               | \-                          |
+-------------------------+-------------------------+-----------------------------+----------------------------------+-----------------------------+

------------------------------------------------------------------------

**Interpreting Two-Way ANOVA Results**

When conducting a Two-Way ANOVA, always check interaction effects first:

1.  If the interaction ($A \times B$) is significant:
    -   The effect of one factor depends on the level of the other factor.
    -   Main effects are not interpretable alone because their impact varies across levels of the second factor.
2.  If the interaction is NOT significant:
    -   The factors have independent (additive) effects.
    -   Main effects can be tested individually.

**Post-Hoc Comparisons**

-   If interaction is not significant, proceed with main effect comparisons using:
    -   [Tukey](#sec-tukeys-anova)
    -   [Scheffé](#sec-scheffe-anova)
    -   [Bonferroni](#sec-bonferroni-anova)
-   If interaction is significant, post-hoc tests should examine simple effects (comparisons within each level of a factor).

------------------------------------------------------------------------

##### Contrasts in Two-Way ANOVA

In Two-Way ANOVA, we can define contrasts to test specific hypotheses:

$$
L = \sum c_i \mu_i, \quad \text{where } \sum c_i = 0
$$

An unbiased estimator of $L$:

$$
\hat{L} = \sum c_i \bar{Y}_{i..}
$$

with variance:

$$
\sigma^2(\hat{L}) = \frac{\sigma^2}{bn} \sum c_i^2
$$

and variance estimate:

$$
\frac{MSE}{bn} \sum c_i^2
$$

------------------------------------------------------------------------

###### Orthogonal Contrasts in Two-Way ANOVA

For two contrasts:

$$
\begin{aligned}
L_1 &= \sum c_i \mu_i, \quad \sum c_i = 0 \\
L_2 &= \sum d_i \mu_i, \quad \sum d_i = 0
\end{aligned}
$$

They are orthogonal if:

$$
\sum \frac{c_i d_i}{n_i} = 0
$$

For balanced designs ($n_i = n$):

$$
\sum c_i d_i = 0
$$

This ensures that orthogonal contrasts are uncorrelated:

$$
\begin{aligned}
cov(\hat{L}_1, \hat{L}_2) &= cov\left(\sum_i c_i \bar{Y}_{i..}, \sum_l d_l \bar{Y}_{l..}\right) \\
&= \sum_i \sum_l c_i d_l cov(\bar{Y}_{i..},\bar{Y}_{l..}) \\
&= \sum_i c_i d_i \frac{\sigma^2}{bn} = 0
\end{aligned}
$$

Thus, orthogonal contrasts allow us to partition the sum of squares further.

------------------------------------------------------------------------

###### Orthogonal Polynomial Contrasts

-   Used when factor levels are equally spaced (e.g., dose levels: 0, 15, 30, 45, 60).
-   Requires equal sample sizes across factor levels.

The Sum of Squares (SS) for a given contrast:

$$
SS_L = \frac{\hat{L}^2}{\sum_{i=1}^a \frac{c^2_i}{bn_i}}
$$

The $t$-statistic for testing contrasts:

$$
T = \frac{\hat{L}}{\sqrt{MSE \sum_{i=1}^a \frac{c_i^2}{bn_i}}} \sim t
$$

Since:

$$
t^2_{(1-\alpha/2; df)} = F_{(1-\alpha; 1, df)}
$$

we can equivalently test:

$$
\frac{SS_L}{MSE} \sim F_{(1-\alpha;1,df_{MSE})}
$$

All contrasts have $df = 1$.

------------------------------------------------------------------------

#### Unbalanced Two-Way ANOVA

In many practical situations, sample sizes may be unequal across factor combinations, such as in:

-   Observational studies (e.g., real-world data with missing values).
-   Dropouts in designed studies (e.g., clinical trials with subject attrition).
-   Larger sample sizes for inexpensive treatments.
-   Sample sizes chosen to match population proportions.

We assume the standard Two-Way ANOVA model:

$$
Y_{ijk} = \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ijk}
$$

where sample sizes vary:

$$
\begin{aligned}
n_{i.} &= \sum_j n_{ij} \quad \text{(Total for factor level } i) \\
n_{.j} &= \sum_i n_{ij} \quad \text{(Total for factor level } j) \\
n_T &= \sum_i \sum_j n_{ij} \quad \text{(Total sample size)}
\end{aligned}
$$

However, for unbalanced designs, a major issue arises:

$$
SSTO \neq SSA + SSB + SSAB + SSE
$$

Unlike the balanced case, the design is non-orthogonal, meaning sum-of-squares partitions do not add up cleanly.

------------------------------------------------------------------------

##### Indicator Variables for Factor Levels

To handle unbalanced data, we use indicator (dummy) variables as predictors.

For Factor A ($i = 1, \dots, a-1$):

$$
u_i =
\begin{cases} 
+1 & \text{if observation is from level } i \text{ of Factor A} \\ 
-1 & \text{if observation is from the reference level (level } a \text{)} \\ 
0 & \text{otherwise} 
\end{cases}
$$

For Factor B ($j = 1, \dots, b-1$):

$$
v_j =
\begin{cases} 
+1 & \text{if observation is from level } j \text{ of Factor B} \\ 
-1 & \text{if observation is from the reference level (level } b \text{)} \\ 
0 & \text{otherwise} 
\end{cases}
$$

Rewriting the ANOVA model using indicator variables:

$$
Y = \mu_{..} + \sum_{i=1}^{a-1} \alpha_i u_i + \sum_{j=1}^{b-1} \beta_j v_j + \sum_{i=1}^{a-1} \sum_{j=1}^{b-1}(\alpha \beta)_{ij} u_i v_j + \epsilon
$$

Here, the unknown parameters are:

-   $\mu_{..}$ (grand mean),

-   $\alpha_i$ (main effects for Factor A),

-   $\beta_j$ (main effects for Factor B),

-   $(\alpha \beta)_{ij}$ (interaction effects).

------------------------------------------------------------------------

##### Hypothesis Testing Using Extra Sum of Squares

For unbalanced designs, we use sequential (type I) or adjusted (type III) sum of squares to test hypotheses.

To test for interaction effects, we test:

$$
\begin{aligned}
&H_0: \text{All } (\alpha \beta)_{ij} = 0 \quad \text{(No interaction)} \\
&H_a: \text{Not all } (\alpha \beta)_{ij} = 0 \quad \text{(Interaction present)}
\end{aligned}
$$

To test whether Factor B has an effect:

$$
\begin{aligned}
&H_0: \beta_1 = \beta_2 = \dots = \beta_b = 0 \\
&H_a: \text{At least one } \beta_j \neq 0
\end{aligned}
$$

------------------------------------------------------------------------

##### Factor Mean Analysis and Contrasts

Factor means and contrasts (e.g., pairwise comparisons) work similarly to the balanced case but require adjustments due to unequal sample sizes.

The variance estimate for a contrast:

$$
\sigma^2(\hat{L}) = \frac{\sigma^2}{\sum n_{ij}} \sum c_i^2
$$

is modified to:

$$
\frac{MSE}{\sum n_{ij}} \sum c_i^2
$$

Orthogonal contrasts are harder to define because unequal sample sizes break orthogonality.

------------------------------------------------------------------------

##### Regression Approach to Unbalanced ANOVA

An alternative is to fit the cell means model as a regression model:

$$
Y_{ij} = \mu_{ij} + \epsilon_{ij}
$$

which allows us to analyze each treatment mean separately.

However, if there are empty cells (some factor combinations have no observations), the regression approach fails, and only partial analyses can be conducted.

------------------------------------------------------------------------

### Two-Way Random Effects ANOVA {#sec-two-way-random-effects-anova}

The Two-Way Random Effects ANOVA assumes that both Factor A and Factor B levels are randomly sampled from larger populations.

The model is:

$$
Y_{ijk} = \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ijk}
$$

where:

-   $\mu_{..}$: Overall mean (constant).
-   $\alpha_i \sim N(0, \sigma^2_{\alpha})$ for $i = 1, \dots, a$ (random effects for Factor A, independently distributed).
-   $\beta_j \sim N(0, \sigma^2_{\beta})$ for $j = 1, \dots, b$ (random effects for Factor B, independently distributed).
-   $(\alpha \beta)_{ij} \sim N(0, \sigma^2_{\alpha \beta})$ for $i = 1, \dots, a$, $j = 1, \dots, b$ (random interaction effects, independently distributed).
-   $\epsilon_{ijk} \sim N(0, \sigma^2)$ (random error, independently distributed).

Additionally, all random effects ($\alpha_i, \beta_j, (\alpha \beta)_{ij}$) and error terms ($\epsilon_{ijk}$) are mutually independent.

------------------------------------------------------------------------

#### Expectation

Taking expectations on both sides:

$$
E(Y_{ijk}) = E(\mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ijk})
$$

Since all random effects have mean zero:

$$
E(Y_{ijk}) = \mu_{..}
$$

Thus, the mean response across all factor levels is $\mu_{..}$.

------------------------------------------------------------------------

#### Variance

The total variance of observations is the sum of all variance components:

$$
\begin{aligned}
var(Y_{ijk}) &= var(\alpha_i) + var(\beta_j) + var((\alpha \beta)_{ij}) + var(\epsilon_{ijk}) \\
&= \sigma^2_{\alpha} + \sigma^2_{\beta} + \sigma^2_{\alpha \beta} + \sigma^2
\end{aligned}
$$

Thus:

$$
Y_{ijk} \sim N(\mu_{..}, \sigma^2_{\alpha} + \sigma^2_{\beta} + \sigma^2_{\alpha \beta} + \sigma^2)
$$

------------------------------------------------------------------------

#### Covariance Structure

In random effects models, observations are correlated if they share the same factor levels.

**Case 1: Same factor A, different factor B**

If $i$ is the same but $j \neq j'$, then:

$$
cov(Y_{ijk}, Y_{ij'k'}) = var(\alpha_i) = \sigma^2_{\alpha}
$$

**Case 2: Same factor B, different factor A**

If $j$ is the same but $i \neq i'$, then:

$$
cov(Y_{ijk}, Y_{i'jk'}) = var(\beta_j) = \sigma^2_{\beta}
$$

**Case 3: Same factor A and B, different replication**

If both factor levels are the same ($i, j$ fixed), but different replication ($k \neq k'$):

$$
cov(Y_{ijk}, Y_{ijk'}) = var(\alpha_i) + var(\beta_j) + var((\alpha \beta)_{ij}) = \sigma^2_{\alpha} + \sigma^2_{\beta} + \sigma^2_{\alpha \beta}
$$

**Case 4: Completely different factor levels**

If neither factor A nor B is the same ($i \neq i'$, $j \neq j'$), then:

$$
cov(Y_{ijk}, Y_{i'j'k'}) = 0
$$

since all random effects are independent across different factor levels.

------------------------------------------------------------------------

**Summary of Variance-Covariance Structure**

+------------------------------------------------+---------------------------------+------------------------------------------------------------------+
| **Case**                                       | **Condition**                   | **Covariance**                                                   |
+================================================+=================================+==================================================================+
| **Same factor A, different factor B**          | $i$ same, $j \neq j'$           | $\sigma^2_{\alpha}$                                              |
+------------------------------------------------+---------------------------------+------------------------------------------------------------------+
| **Same factor B, different factor A**          | $j$ same, $i \neq i'$           | $\sigma^2_{\beta}$                                               |
+------------------------------------------------+---------------------------------+------------------------------------------------------------------+
| **Same factor levels, different replications** | $i$ same, $j$ same, $k \neq k'$ | $\sigma^2_{\alpha} + \sigma^2_{\beta} + \sigma^2_{\alpha \beta}$ |
+------------------------------------------------+---------------------------------+------------------------------------------------------------------+
| **Different factor levels**                    | $i \neq i'$, $j \neq j'$        | $0$                                                              |
+------------------------------------------------+---------------------------------+------------------------------------------------------------------+

------------------------------------------------------------------------

### Two-Way Mixed Effects ANOVA {#sec-two-way-mixed-effects-anova}

In a Two-Way Mixed Effects Model, one factor is fixed, while the other is random.\
This is often referred to as a mixed effects model or simply a mixed model.

#### Balanced

For a balanced design, the **restricted** mixed model is:

$$
Y_{ijk} = \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ijk}
$$

where:

-   $\mu_{..}$: Overall mean (constant).
-   $\alpha_i$: Fixed effects for Factor A, subject to the constraint $\sum \alpha_i = 0$.
-   $\beta_j \sim N(0, \sigma^2_\beta)$ (random effects for Factor B).
-   $(\alpha \beta)_{ij} \sim N(0, \frac{a-1}{a} \sigma^2_{\alpha \beta})$\
    (interaction effects, constrained so that $\sum_i (\alpha \beta)_{ij} = 0$ for all $j$). The variance is written as the proportion for convenience, it makes the expected mean squares simpler.
-   $\epsilon_{ijk} \sim N(0, \sigma^2)$ (random error).
-   $\beta_j, (\alpha \beta)_{ij}, \epsilon_{ijk}$ are pairwise independent.

The restriction on interaction variance ($\frac{a-1}{a} \sigma^2_{\alpha \beta}$) simplifies the expected mean squares, though some sources assume $var((\alpha \beta)_{ij}) = \sigma^2_{\alpha \beta}$.

------------------------------------------------------------------------

An **unrestricted** version of the model removes constraints on interaction terms.

Define:

$$
\begin{aligned}
\beta_j &= \beta_j^* + (\bar{\alpha \beta})_{ij}^* \\
(\alpha \beta)_{ij} &= (\alpha \beta)_{ij}^* - (\bar{\alpha \beta})_{ij}^*
\end{aligned}
$$

where $\beta^*$ and $(\alpha \beta)^*_{ij}$ are unrestricted random effects.

Some consider the **restricted model more general**, but we use the **restricted form** for simplicity.

------------------------------------------------------------------------

Taking expectations:

$$
E(Y_{ijk}) = \mu_{..} + \alpha_i
$$

The total variance of responses:

$$
var(Y_{ijk}) = \sigma^2_\beta + \frac{a-1}{a} \sigma^2_{\alpha \beta} + \sigma^2
$$

------------------------------------------------------------------------

**Covariance Structure**

Observations sharing the same **random factor (B)** level are correlated.

**Covariances for Different Cases**

+---------------------------------------------------+------------------------------------------------------------------------------------+
| **Condition**                                     | **Covariance**                                                                     |
+===================================================+====================================================================================+
| Same $i, j$, different replications ($k \neq k'$) | $cov (Y_{ijk}, Y_{ijk'}) = \sigma^2_\beta + \frac{a-1}{a} \sigma^2_{\alpha \beta}$ |
+---------------------------------------------------+------------------------------------------------------------------------------------+
| Same $j$, different $i$ ($i \neq i'$)             | $cov(Y_{ijk}, Y_{i'jk'}) = \sigma^2_\beta - \frac{1}{a} \sigma^2_{\alpha \beta}$   |
+---------------------------------------------------+------------------------------------------------------------------------------------+
| Different $i$ and $j$ ($i \neq i'$, $j \neq j'$)  | $cov(Y_{ijk}, Y_{i'j'k'}) = 0$                                                     |
+---------------------------------------------------+------------------------------------------------------------------------------------+

Thus, observations only become independent when they do not share the same random effect.

An advantage of the **restricted mixed model** is that 2 observations from the same random factor (B) level can be positively or negatively correlated. In the **unrestricted** model, they can only be positively correlated.

------------------------------------------------------------------------

**Comparison of Fixed, Random, and Mixed Effects Models**

+-----------------+--------------------------------------------------------------+----------------------------------------+------------------------------------------------------------------------------------+
| **Mean Square** | **Fixed ANOVA (A, B fixed)**                                 | **Random ANOVA (A, B random)**         | **Mixed ANOVA (A fixed, B random)**                                                |
+=================+==============================================================+========================================+====================================================================================+
| **MSA**         | $\sigma^2 + n b \frac{\sum \alpha_i^2}{a-1}$                 | $\sigma^2 + n b \sigma^2_{\alpha}$     | $\sigma^2 + n b \frac{\sum_{i = 1}^a \alpha_i^2}{a-1} + n \sigma^2_{\alpha \beta}$ |
+-----------------+--------------------------------------------------------------+----------------------------------------+------------------------------------------------------------------------------------+
| **MSB**         | $\sigma^2 + n a \frac{\sum \beta_j^2}{b-1}$                  | $\sigma^2 + n a \sigma^2_{\beta}$      | $\sigma^2 + n a \sigma^2_{\beta} + n \sigma^2_{\alpha \beta}$                      |
+-----------------+--------------------------------------------------------------+----------------------------------------+------------------------------------------------------------------------------------+
| **MSAB**        | $\sigma^2 + n \frac{\sum (\alpha \beta)_{ij}^2}{(a-1)(b-1)}$ | $\sigma^2 + n \sigma^2_{\alpha \beta}$ | $\sigma^2 + n \sigma^2_{\alpha \beta}$                                             |
+-----------------+--------------------------------------------------------------+----------------------------------------+------------------------------------------------------------------------------------+
| **MSE**         | $\sigma^2$                                                   | $\sigma^2$                             | $\sigma^2$                                                                         |
+-----------------+--------------------------------------------------------------+----------------------------------------+------------------------------------------------------------------------------------+

While SS and df are identical across models, the expected mean squares differ, affecting test statistics.

------------------------------------------------------------------------

##### Hypothesis Testing in Mixed ANOVA

In [random ANOVA](#sec-two-way-random-effects-anova), we test:

$$
\begin{aligned}
H_0: \sigma^2 = 0 \quad vs. \quad H_a: \sigma^2 > 0
\end{aligned}
$$

using:

$$
F = \frac{MSA}{MSAB} \sim F_{a-1, (a-1)(b-1)}
$$

For [mixed models](#sec-two-way-mixed-effects-anova), the same test statistic is used for:

$$
H_0: \alpha_i = 0, \quad \forall i
$$

However, for [fixed effects models](#sec-two-factor-fixed-effects-anova), the test statistic differs.

+-------------------------+------------------------------+--------------------------------+-------------------------------------+
| **Test for Effect of**  | **Fixed ANOVA (A, B fixed)** | **Random ANOVA (A, B random)** | **Mixed ANOVA (A fixed, B random)** |
+=========================+==============================+================================+=====================================+
| **Factor A**            | $\frac{MSA}{MSE}$            | $\frac{MSA}{MSAB}$             | $\frac{MSA}{MSAB}$                  |
+-------------------------+------------------------------+--------------------------------+-------------------------------------+
| **Factor B**            | $\frac{MSB}{MSE}$            | $\frac{MSB}{MSAB}$             | $\frac{MSB}{MSE}$                   |
+-------------------------+------------------------------+--------------------------------+-------------------------------------+
| **Interaction (A × B)** | $\frac{MSAB}{MSE}$           | $\frac{MSAB}{MSE}$             | $\frac{MSAB}{MSE}$                  |
+-------------------------+------------------------------+--------------------------------+-------------------------------------+

------------------------------------------------------------------------

##### Variance Component Estimation

In random and mixed effects models, we are interested in estimating variance components.

To estimate $\sigma^2_\beta$:

$$
E(\sigma^2_\beta) = \frac{E(MSB) - E(MSE)}{na} = \frac{\sigma^2 + na \sigma^2_\beta - \sigma^2}{na} = \sigma^2_\beta
$$

which is estimated by:

$$
\hat{\sigma}^2_\beta = \frac{MSB - MSE}{na}
$$

Confidence intervals for variance components can be approximated using:

-   [Satterthwaite procedure](#sec-satterthwaite-approximation).
-   Modified large-sample (MLS) method

------------------------------------------------------------------------

##### Estimating Fixed Effects in Mixed Models

Fixed effects $\alpha_i$ are estimated by:

$$
\begin{aligned}
\hat{\alpha}_i &= \bar{Y}_{i..} - \bar{Y}_{...} \\
\hat{\mu}_{i.} &= \bar{Y}_{...} + (\bar{Y}_{i..} - \bar{Y}_{...}) = \bar{Y}_{i..}
\end{aligned}
$$

Their variances:

$$
\begin{aligned}
\sigma^2(\hat{\alpha}_i) &= \frac{\sigma^2 + n \sigma^2_{\alpha \beta}}{bn} = \frac{E(MSAB)}{bn} \\
s^2(\hat{\alpha}_i) &= \frac{MSAB}{bn}
\end{aligned}
$$

------------------------------------------------------------------------

##### Contrasts on Fixed Effects

For a contrast:

$$
L = \sum c_i \alpha_i, \quad \text{where } \sum c_i = 0
$$

Estimate:

$$
\hat{L} = \sum c_i \hat{\alpha}_i
$$

Variance:

$$
\sigma^2(\hat{L}) = \sum c^2_i \sigma^2(\hat{\alpha}_i), \quad s^2(\hat{L}) = \frac{MSAB}{bn} \sum c^2_i
$$

------------------------------------------------------------------------

#### Unbalanced Two-Way Mixed Effects ANOVA

In an **unbalanced** two-way mixed model (e.g., $a = 2, b = 4$), the model remains:

$$
Y_{ijk} = \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ijk}
$$

where:

-   $\alpha_i$: Fixed effects for Factor A.
-   $\beta_j \sim N(0, \sigma^2_\beta)$: Random effects for Factor B.
-   $(\alpha \beta)_{ij} \sim N(0, \frac{\sigma^2_{\alpha \beta}}{2})$: Interaction effects.
-   $\epsilon_{ijk} \sim N(0, \sigma^2)$: Residual error.

------------------------------------------------------------------------

##### Variance Components

The variance components are:

$$
\begin{aligned}
var(\beta_j) &= \sigma^2_\beta \\
var((\alpha \beta)_{ij}) &= \frac{2-1}{2} \sigma^2_{\alpha \beta} = \frac{\sigma^2_{\alpha \beta}}{2} \\
var(\epsilon_{ijk}) &= \sigma^2
\end{aligned}
$$

##### Expectation and Variance

Taking expectations:

$$
E(Y_{ijk}) = \mu_{..} + \alpha_i
$$

Total variance:

$$
var(Y_{ijk}) = \sigma^2_{\beta} + \frac{\sigma^2_{\alpha \beta}}{2} + \sigma^2
$$

------------------------------------------------------------------------

##### Covariance Structure

Observations sharing **Factor B (random effect)** are correlated.

**Covariances for Different Cases**

+-------------------------------------------------------+-----------------------------------------------------------------------------------+
| **Condition**                                         | **Covariance**                                                                    |
+=======================================================+===================================================================================+
| **Same** $i, j$, different replications ($k \neq k'$) | $cov(Y_{ijk}, Y_{ijk'}) = \sigma^2 + \frac{\sigma^2_{\alpha \beta}}{2}$           |
+-------------------------------------------------------+-----------------------------------------------------------------------------------+
| **Same** $j$, different $i$ ($i \neq i'$)             | $cov (Y_{ijk}, Y_{i'jk'}) = \sigma^2_{\beta} - \frac{\sigma^2_{\alpha \beta}}{2}$ |
+-------------------------------------------------------+-----------------------------------------------------------------------------------+
| **Different** $i$ and $j$ ($i \neq i'$, $j \neq j'$)  | $cov(Y_{ijk}, Y_{i'j'k'}) = 0$                                                    |
+-------------------------------------------------------+-----------------------------------------------------------------------------------+

Thus, only observations within the same random factor level share dependence.

------------------------------------------------------------------------

##### Matrix Representation

Assume:

$$
\mathbf{Y} \sim N(\mathbf{X} \beta, M)
$$

where:

-   $\mathbf{X}$: Fixed effects design matrix.
-   $\beta$: Fixed effect coefficients.
-   $M$: Block diagonal covariance matrix containing variance components.

The density function of $\mathbf{Y}$ is:

$$
f(\mathbf{Y}) = \frac{1}{(2\pi)^{N/2} |M|^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{Y} - \mathbf{X} \beta)' M^{-1} (\mathbf{Y} - \mathbf{X} \beta) \right)
$$

If variance components were known, we could use Generalized Least Squares:

$$
\hat{\beta}_{GLS} = (\mathbf{X}' M^{-1} \mathbf{X})^{-1} \mathbf{X}' M^{-1} \mathbf{Y}
$$

However, since variance components ($\sigma^2, \sigma^2_\beta, \sigma^2_{\alpha \beta}$) are unknown, we estimate them using:

-   [Maximum Likelihood]
-   [Restricted Maximum Likelihood]

Maximizing the likelihood:

$$
\ln L = - \frac{N}{2} \ln (2\pi) - \frac{1}{2} \ln |M| - \frac{1}{2} (\mathbf{Y} - \mathbf{X} \beta)' M^{-1} (\mathbf{Y} - \mathbf{X} \beta)
$$

where:

-   $|M|$: Determinant of the variance-covariance matrix.
-   $(\mathbf{Y} - \mathbf{X} \beta)' M^{-1} (\mathbf{Y} - \mathbf{X} \beta)$: Quadratic form in the likelihood.

------------------------------------------------------------------------

## Nonparametric ANOVA {#sec-nonparametric-anova}

When assumptions of **normality** and **equal variance** are not satisfied, we use nonparametric ANOVA tests, which rank the data instead of using raw values.

------------------------------------------------------------------------

### Kruskal-Wallis Test (One-Way Nonparametric ANOVA)

The Kruskal-Wallis test is a generalization of the Wilcoxon rank-sum test to more than two independent samples. It is an alternative to one-way ANOVA when normality is not assumed.

Setup

-   $a \geq 2$ independent treatments.
-   $n_i$ is the sample size for the $i$-th treatment.
-   $Y_{ij}$ is the $j$-th observation from the $i$-th treatment.
-   No assumption of normality.
-   Assume observations are independent random samples from continuous CDFs $F_1, F_2, \dots, F_a$.

Hypotheses

$$
\begin{aligned}
&H_0: F_1 = F_2 = \dots = F_a \quad \text{(All distributions are identical)} \\
&H_a: F_i < F_j \text{ for some } i \neq j
\end{aligned}
$$ If the data come from a location-scale family, the hypothesis simplifies to:

$$
H_0: \theta_1 = \theta_2 = \dots = \theta_a
$$

------------------------------------------------------------------------

**Procedure**

1.  Rank all $N = \sum_{i=1}^a n_i$ observations in ascending order.\
    Let $r_{ij} = rank(Y_{ij})$\
    The sum of ranks must satisfy:

    $$
    \sum_i \sum_j r_{ij} = \frac{N(N+1)}{2}
    $$

2.  Compute rank sums and averages: $$
    r_{i.} = \sum_{j=1}^{n_i} r_{ij}, \quad \bar{r}_{i.} = \frac{r_{i.}}{n_i}
    $$

3.  Calculate the test statistic:

    $$
    \chi_{KW}^2 = \frac{SSTR}{\frac{SSTO}{N-1}}
    $$

    where:

    -   Treatment Sum of Squares: $$
        SSTR = \sum n_i (\bar{r}_{i.} - \bar{r}_{..})^2
        $$
    -   Total Sum of Squares: $$
        SSTO = \sum_i \sum_j (r_{ij} - \bar{r}_{..})^2
        $$
    -   Overall Mean Rank: $$
        \bar{r}_{..} = \frac{N+1}{2}
        $$

4.  Compare to a chi-square distribution:

    -   For large $n_i$ ($\geq 5$), $\chi^2_{KW} \sim \chi^2_{a-1}$.
    -   Reject $H_0$ if: $$
        \chi^2_{KW} > \chi^2_{(1-\alpha; a-1)}
        $$

5.  Exact Test for Small Samples:

    -   Compute all possible rank assignments:\
        $$
        \frac{N!}{n_1! n_2! \dots n_a!}
        $$
    -   Evaluate each Kruskal-Wallis statistic and determine the empirical p-value.

------------------------------------------------------------------------

### Friedman Test (Nonparametric Two-Way ANOVA)

The Friedman test is a distribution-free alternative to two-way ANOVA when data are measured in a randomized complete block design and normality cannot be assumed.

Setup

-   $Y_{ij}$ represents responses from $n$ blocks and $r$ treatments.
-   Assume no normality or homogeneity of variance.
-   Let $F_{ij}$ be the CDF of $Y_{ij}$, corresponding to observed values.

Hypotheses

$$
\begin{aligned}
&H_0: F_{i1} = F_{i2} = \dots = F_{ir} \quad \forall i \quad \text{(Identical distributions within each block)} \\
&H_a: F_{ij} < F_{ij'} \text{ for some } j \neq j' \quad \forall i
\end{aligned}
$$

For location-scale families, the hypothesis simplifies to:

$$
\begin{aligned}
&H_0: \tau_1 = \tau_2 = \dots = \tau_r \\
&H_a: \tau_j > \tau_{j'} \text{ for some } j \neq j'
\end{aligned}
$$

------------------------------------------------------------------------

**Procedure**

1.  Rank observations within each block separately (ascending order).

    -   If there are ties, assign average ranks.

2.  Compute test statistic:

    $$
    \chi^2_F = \frac{SSTR}{\frac{SSTR + SSE}{n(r-1)}}
    $$

    where:

    -   Treatment Sum of Squares: $$
        SSTR = n \sum (\bar{r}_{.j} - \bar{r}_{..})^2
        $$
    -   Error Sum of Squares: $$
        SSE = \sum_i \sum_j (r_{ij} - \bar{r}_{.j})^2
        $$
    -   Mean Ranks: $$
        \bar{r}_{.j} = \frac{\sum_i r_{ij}}{n}, \quad \bar{r}_{..} = \frac{r+1}{2}
        $$

3.  Alternative Formula for Large Samples (No Ties):

    If no ties, Friedman's statistic simplifies to:

    $$
    \chi^2_F = \left[\frac{12}{nr(n+1)} \sum_j r_{.j}^2\right] - 3n(r+1)
    $$

4.  Compare to a chi-square distribution:

    -   For large $n$, $\chi^2_F \sim \chi^2_{r-1}$.
    -   Reject $H_0$ if: $$
        \chi^2_F > \chi^2_{(1-\alpha; r-1)}
        $$

5.  Exact Test for Small Samples:

    -   Compute all possible ranking permutations: $$
        (r!)^n
        $$
    -   Evaluate each Friedman statistic and determine the empirical p-value.

------------------------------------------------------------------------

## Randomized Block Designs {#sec-randomized-block-designs}

To improve the precision of treatment comparisons, we can **reduce variability** among experimental units by grouping them into **blocks**.\
Each block contains **homogeneous units**, reducing the impact of nuisance variation.

**Key Principles of Blocking**

-   Within each block, treatments are randomly assigned to units.
-   The number of units per block is a multiple of the number of factor combinations.
-   Commonly, each treatment appears once per block.

------------------------------------------------------------------------

**Benefits of Blocking**

-   Reduction in variability of treatment effect estimates

    -   Improved power for t-tests and F-tests.

    -   Narrower confidence intervals.

    -   Smaller mean square error (MSE).

-   Allows comparison of treatments across different conditions (captured by blocks).

**Potential Downsides of Blocking**

-   If blocks are not chosen well, degrees of freedom are wasted on negligible block effects.

-   This reduces df for t-tests and F-tests without reducing MSE, causing a small loss of power.

------------------------------------------------------------------------

#### Random Block Effects with Additive Effects {#sec-random-block-effects-with-additive-effects}

The statistical model for a randomized block design:

$$
Y_{ij} = \mu_{..} + \rho_i + \tau_j + \epsilon_{ij}
$$

where:

-   $i = 1, 2, \dots, n$ (Blocks)

-   $j = 1, 2, \dots, r$ (Treatments)

-   $\mu_{..}$: Overall mean response (averaged across all blocks and treatments).

-   $\rho_i$: Block effect (average difference for the $i$-th block), constrained such that:

    $$
    \sum_i \rho_i = 0
    $$

-   $\tau_j$: Treatment effect (average across blocks), constrained such that:

    $$
    \sum_j \tau_j = 0
    $$

-   $\epsilon_{ij} \sim iid N(0, \sigma^2)$: Random experimental error.

**Interpretation of the Model**

-   Block and treatment effects are additive.

-   The difference in average response between any two treatments is the same within each block:

    $$
    (\mu_{..} + \rho_i + \tau_j) - (\mu_{..} + \rho_i + \tau_j') = \tau_j - \tau_j'
    $$

-   This ensures that blocking only affects variability, not treatment comparisons.

------------------------------------------------------------------------

**Estimators of Model Parameters**

-   **Overall Mean**:

    $$
    \hat{\mu} = \bar{Y}_{..}
    $$

-   **Block Effects**:

    $$
    \hat{\rho}_i = \bar{Y}_{i.} - \bar{Y}_{..}
    $$

-   **Treatment Effects**:

    $$
    \hat{\tau}_j = \bar{Y}_{.j} - \bar{Y}_{..}
    $$

------------------------------------------------------------------------

-   **Fitted Response:** $$
    \hat{Y}_{ij} = \bar{Y}_{..} + (\bar{Y}_{i.} - \bar{Y}_{..}) + (\bar{Y}_{.j} - \bar{Y}_{..})
    $$

    Simplifies to:

    $$
    \hat{Y}_{ij} = \bar{Y}_{i.} + \bar{Y}_{.j} - \bar{Y}_{..}
    $$

-   **Residuals**:

    $$
    e_{ij} = Y_{ij} - \hat{Y}_{ij} = Y_{ij} - \bar{Y}_{i.} - \bar{Y}_{.j} + \bar{Y}_{..}
    $$

------------------------------------------------------------------------

#### ANOVA Table for Randomized Block Design

The ANOVA decomposition partitions total variability into contributions from blocks, treatments, and residual error.

+-------------------------+-------------------------------------------------------------------------+-----------------------------+------------------------------------------+------------------------------------------+
| **Source of Variation** | **Sum of Squares (SS)**                                                 | **Degrees of Freedom (df)** | **Fixed Treatments (E(MS))**             | **Random Treatments (E(MS))**            |
+=========================+=========================================================================+=============================+==========================================+==========================================+
| **Blocks**              | $r \sum_i (\bar{Y}_{i.} - \bar{Y}_{..})^2$                              | $n-1$                       | $\sigma^2 + r \frac{\sum \rho_i^2}{n-1}$ | $\sigma^2 + r \frac{\sum \rho_i^2}{n-1}$ |
+-------------------------+-------------------------------------------------------------------------+-----------------------------+------------------------------------------+------------------------------------------+
| **Treatments**          | $n \sum_j (\bar{Y}_{.j} - \bar{Y}_{..})^2$                              | $r-1$                       | $\sigma^2 + n \frac{\sum \tau_j^2}{r-1}$ | $\sigma^2 + n \sigma^2_{\tau}$           |
+-------------------------+-------------------------------------------------------------------------+-----------------------------+------------------------------------------+------------------------------------------+
| **Error**               | $\sum_i \sum_j (Y_{ij} - \bar{Y}_{i.} - \bar{Y}_{.j} + \bar{Y}_{..})^2$ | $(n-1)(r-1)$                | $\sigma^2$                               | $\sigma^2$                               |
+-------------------------+-------------------------------------------------------------------------+-----------------------------+------------------------------------------+------------------------------------------+
| **Total**               | $SSTO$                                                                  | $nr-1$                      | \-                                       | \-                                       |
+-------------------------+-------------------------------------------------------------------------+-----------------------------+------------------------------------------+------------------------------------------+

#### F-tests in Randomized Block Designs

To test for treatment effects, we use an F-test:

For **fixed treatment effects**:

$$
\begin{aligned}
H_0: \tau_1 = \tau_2 = \dots = \tau_r = 0 \quad \text{(No treatment effect)} \\
H_a: \text{Not all } \tau_j = 0
\end{aligned}
$$

For **random treatment effects**:

$$
\begin{aligned}
H_0: \sigma^2_{\tau} = 0 \quad \text{(No variance in treatment effects)} \\
H_a: \sigma^2_{\tau} \neq 0
\end{aligned}
$$

In both cases, the test statistic is:

$$
F = \frac{MSTR}{MSE}
$$

Reject $H_0$ if:

$$
F > f_{(1-\alpha; r-1, (n-1)(r-1))}
$$

------------------------------------------------------------------------

**Why Not Use an F-Test for Blocks?**

We do not test for block effects because:

1.  Blocks are assumed to be different a priori.
2.  Randomization occurs within each block, ensuring treatments are comparable.

------------------------------------------------------------------------

**Efficiency Gain from Blocking**

To measure the improvement in precision, compare the mean square error (MSE) in a [completely randomized design](#sec-completely-randomized-design) vs. a [randomized block design](#sec-randomized-block-designs).

Estimated variance in a [CRD](#sec-completely-randomized-design):

$$
\hat{\sigma}^2_{CR} = \frac{(n-1)MSBL + n(r-1)MSE}{nr-1}
$$

Estimated variance in an [RBD](#sec-randomized-block-designs):

$$
\hat{\sigma}^2_{RB} = MSE
$$

Relative efficiency:

$$
\frac{\hat{\sigma}^2_{CR}}{\hat{\sigma}^2_{RB}}
$$

-   If greater than 1, blocking reduces experimental error.

-   The percentage reduction in required sample size for an RBD:

$$
\left( \frac{\hat{\sigma}^2_{CR}}{\hat{\sigma}^2_{RB}} - 1 \right) \times 100\%
$$

------------------------------------------------------------------------

**Random Blocks and Mixed Models**

If blocks are randomly selected, they are treated as random effects.\
That is, if the experiment were repeated, a new set of blocks would be selected, with:

$$
\rho_1, \rho_2, \dots, \rho_i \sim N(0, \sigma^2_\rho)
$$

The model remains:

$$
Y_{ij} = \mu_{..} + \rho_i + \tau_j + \epsilon_{ij}
$$

where:

-   $\mu_{..}$ is fixed.
-   $\rho_i \sim iid N(0, \sigma^2_\rho)$ (random block effects).
-   $\tau_j$ is fixed (or random, with $\sum \tau_j = 0$).
-   $\epsilon_{ij} \sim iid N(0, \sigma^2)$.

------------------------------------------------------------------------

#### Variance and Covariance Structure

For **fixed treatment effects**:

$$
\begin{aligned}
E(Y_{ij}) &= \mu_{..} + \tau_j \\
var(Y_{ij}) &= \sigma^2_{\rho} + \sigma^2
\end{aligned}
$$

Observations **within the same block** are **correlated**:

$$
cov(Y_{ij}, Y_{ij'}) = \sigma^2_{\rho}, \quad j \neq j'
$$

Observations **from different blocks** are **independent**:

$$
cov(Y_{ij}, Y_{i'j'}) = 0, \quad i \neq i', j \neq j'
$$

The **intra-block correlation**:

$$
\frac{\sigma^2_{\rho}}{\sigma^2 + \sigma^2_{\rho}}
$$

------------------------------------------------------------------------

**Expected Mean Squares for Fixed Treatments**

| Source         | SS   | E(MS)                                    |
|----------------|------|------------------------------------------|
| **Blocks**     | SSBL | $\sigma^2 + r \sigma^2_\rho$             |
| **Treatments** | SSTR | $\sigma^2 + n \frac{\sum \tau^2_j}{r-1}$ |
| **Error**      | SSE  | $\sigma^2$                               |

------------------------------------------------------------------------

#### Random Block Effects with Interaction {#sec-random-block-effects-with-interaction}

When block-treatment interaction exists, we modify the model:

$$
Y_{ij} = \mu_{..} + \rho_i + \tau_j + (\rho \tau)_{ij} + \epsilon_{ij}
$$

where:

-   $\rho_i \sim iid N(0, \sigma^2_{\rho})$ (random).

-   $\tau_j$ is fixed ($\sum \tau_j = 0$).

-   $(\rho \tau)_{ij} \sim N(0, \frac{r-1}{r} \sigma^2_{\rho \tau})$, constrained such that:

    $$
    \sum_j (\rho \tau)_{ij} = 0, \quad \forall i
    $$

-   Covariance between interaction terms:

    $$
    cov((\rho \tau)_{ij}, (\rho \tau)_{ij'}) = -\frac{1}{r} \sigma^2_{\rho \tau}, \quad j \neq j'
    $$

-   $\epsilon_{ij} \sim iid N(0, \sigma^2)$.

------------------------------------------------------------------------

**Variance and Covariance with Interaction**

-   **Expectation**:

    $$
    E(Y_{ij}) = \mu_{..} + \tau_j
    $$

-   **Total variance**:

    $$
    var(Y_{ij}) = \sigma^2_\rho + \frac{r-1}{r} \sigma^2_{\rho \tau} + \sigma^2
    $$

-   **Within-block covariance**:

    $$
    cov(Y_{ij}, Y_{ij'}) = \sigma^2_\rho - \frac{1}{r} \sigma^2_{\rho \tau}, \quad j \neq j'
    $$

-   **Between-block covariance**:

    $$
    cov(Y_{ij}, Y_{i'j'}) = 0, \quad i \neq i', j \neq j'
    $$

The sum of squares and degrees of freedom for [interaction model](#sec-random-block-effects-with-interaction) are the same as those for the [additive model](#sec-random-block-effects-with-additive-effects). The difference exists only in the expected mean squares.

------------------------------------------------------------------------

#### ANOVA Table with Interaction Effects

+----------------+----------+--------------+-----------------------------------------------------------------+
| Source         | SS       | df           | E(MS)                                                           |
+================+==========+==============+=================================================================+
| **Blocks**     | $SSBL$   | $n-1$        | $\sigma^2 + r \sigma^2_\rho$                                    |
+----------------+----------+--------------+-----------------------------------------------------------------+
| **Treatments** | $SSTR$   | $r-1$        | $\sigma^2 + \sigma^2_{\rho \tau} + n \frac{\sum \tau_j^2}{r-1}$ |
+----------------+----------+--------------+-----------------------------------------------------------------+
| **Error**      | $SSE$    | $(n-1)(r-1)$ | $\sigma^2 + \sigma^2_{\rho \tau}$                               |
+----------------+----------+--------------+-----------------------------------------------------------------+

------------------------------------------------------------------------

-   No exact test is possible for block effects when interaction is present (Not important if blocks are used primarily to reduce experimental error variability)
-   $E(MSE) = \sigma^2 + \sigma^2_{\rho \tau}$ the error term variance and interaction variance $\sigma^2_{\rho \tau}$. We can't estimate these components separately with this model. The two are **confounded**.
-   If more than one observation per treatment block combination, one can consider interaction with fixed block effects, which is called **generalized randomized block designs** (multifactor analysis).

------------------------------------------------------------------------

#### Tukey Test of Additivity

Tukey's 1-degree-of-freedom test for additivity provides a formal test for interaction effects between blocks and treatments in a randomized block design.

This test can also be used in two-way ANOVA when there is only one observation per cell.

------------------------------------------------------------------------

In a [randomized block design](#sec-randomized-block-designs), an additive model assumes:

$$
Y_{ij} = \mu_{..} + \rho_i + \tau_j + \epsilon_{ij}
$$

where:

-   $\mu_{..}$ = overall mean

-   $\rho_i$ = block effect

-   $\tau_j$ = treatment effect

-   $\epsilon_{ij}$ = random error, $iid N(0, \sigma^2)$

To test for interaction, we introduce a less restricted interaction term:

$$
(\rho \tau)_{ij} = D \rho_i \tau_j
$$

where $D$ is a constant measuring interaction strength.

Thus, the interaction model becomes:

$$
Y_{ij} = \mu_{..} + \rho_i + \tau_j + D\rho_i \tau_j + \epsilon_{ij}
$$

The least squares estimate (or MLE) of $D$ is:

$$
\hat{D} = \frac{\sum_i \sum_j \rho_i \tau_j Y_{ij}}{\sum_i \rho_i^2 \sum_j \tau_j^2}
$$

Replacing $\rho_i$ and $\tau_j$ with their estimates:

$$
\hat{D} = \frac{\sum_i \sum_j (\bar{Y}_{i.} - \bar{Y}_{..})(\bar{Y}_{.j} - \bar{Y}_{..}) Y_{ij}}{\sum_i (\bar{Y}_{i.} - \bar{Y}_{..})^2 \sum_j (\bar{Y}_{.j} - \bar{Y}_{..})^2}
$$

The sum of squares for interaction is:

$$
SS_{int} = \sum_i \sum_j \hat{D}^2 (\bar{Y}_{i.} - \bar{Y}_{..})^2 (\bar{Y}_{.j} - \bar{Y}_{..})^2
$$

------------------------------------------------------------------------

**ANOVA Decomposition**

The total sum of squares (SSTO) is decomposed as:

$$
SSTO = SSBL + SSTR + SS_{int} + SS_{Rem}
$$

where:

-   $SSBL$ = Sum of squares due to blocks

-   $SSTR$ = Sum of squares due to treatments

-   $SS_{int}$ = Interaction sum of squares

-   $SS_{Rem}$ = Remainder sum of squares, computed as:

$$
SS_{Rem} = SSTO - SSBL - SSTR - SS_{int}
$$

------------------------------------------------------------------------

We test:

$$
\begin{aligned}
&H_0: D = 0 \quad \text{(No interaction present)} \\
&H_a: D \neq 0 \quad \text{(Interaction of form $D \rho_i \tau_j$ present)}
\end{aligned}
$$

If $D = 0$, then $SS_{int}$ and $SS_{Rem}$ are independent and follow:

$$
SS_{int} \sim \chi^2_1, \quad SS_{Rem} \sim \chi^2_{(rn-r-n)}
$$

Thus, the F-statistic for testing interaction is:

$$
F = \frac{SS_{int} / 1}{SS_{Rem} / (rn - r - n)}
$$

which follows an $F$-distribution:

$$
F \sim F_{(1, nr - r - n)}
$$

We reject $H_0$ if:

$$
F > f_{(1-\alpha; 1, nr - r - n)}
$$

------------------------------------------------------------------------

## Nested Designs

A nested design occurs when one factor is entirely contained within another. This differs from a crossed design, where all levels of one factor are present across all levels of another factor.

-   **Crossed Design**: If Factor B is crossed with Factor A, then each level of Factor B appears at every level of Factor A.
-   **Nested Design**: If Factor B is nested within Factor A, then each level of Factor B is unique to a particular level of Factor A.

Thus, if Factor B is nested within Factor A:

-   Level 1 of B within A = 1 has nothing in common with

-   Level 1 of B within A = 2.

**Types of Factors**

-   **Classification Factors**: Factors that **cannot be manipulated** (e.g., geographical regions, subjects).
-   **Experimental Factors**: Factors that are **randomly assigned** in an experiment.

------------------------------------------------------------------------

### Two-Factor Nested Design

We consider a nested two-factor model where:

-   Factor A has $a$ levels.

-   Factor B is nested within Factor A, with $b$ levels per level of A.

-   Both factors are fixed.

-   All treatment means are equally important.

------------------------------------------------------------------------

The mean response at level $i$ of Factor A:

$$
\mu_{i.} = \frac{1}{b} \sum_j \mu_{ij}
$$

The main effect of Factor A:

$$
\alpha_i = \mu_{i.} - \mu_{..}
$$

where:

$$
\mu_{..} = \frac{1}{ab} \sum_i \sum_j \mu_{ij} = \frac{1}{a} \sum_i \mu_{i.}
$$

with the constraint:

$$
\sum_i \alpha_i = 0
$$

The nested effect of Factor B within A is denoted as $\beta_{j(i)}$, where:

$$
\begin{aligned}
\beta_{j(i)} &= \mu_{ij} - \mu_{i.} \\
&= \mu_{ij} - \alpha_i - \mu_{..}
\end{aligned}
$$

with the restriction:

$$
\sum_j \beta_{j(i)} = 0, \quad \forall i = 1, \dots, a
$$

Since $\beta_{j(i)}$ is the **specific effect** of the $j$-th level of factor $B$ nested within the $i$-th level of factor $A$, the full model can be written as:

$$
\mu_{ij} = \mu_{..} + \alpha_i + \beta_{j(i)}
$$

or equivalently:

$$
\mu_{ij} = \mu_{..} + (\mu_{i.} - \mu_{..}) + (\mu_{ij} - \mu_{i.})
$$

------------------------------------------------------------------------

The statistical model for a two-factor nested design is:

$$
Y_{ijk} = \mu_{..} + \alpha_i + \beta_{j(i)} + \epsilon_{ijk}
$$

where:

-   $Y_{ijk}$ = response for the $k$-th observation when:

    -   **Factor A** is at level $i$.

    -   **Factor B** (nested within A) is at level $j$.

-   $\mu_{..}$ = overall mean.

-   $\alpha_i$ = main effect of **Factor A** (subject to: $\sum_i \alpha_i = 0$).

-   $\beta_{j(i)}$ = nested effect of **Factor B within A** (subject to: $\sum_j \beta_{j(i)} = 0$ for all $i$).

-   $\epsilon_{ijk} \sim iid N(0, \sigma^2)$ = random error.

Thus, the expected value and variance are:

$$
\begin{aligned}
E(Y_{ijk}) &= \mu_{..} + \alpha_i + \beta_{j(i)} \\
var(Y_{ijk}) &= \sigma^2
\end{aligned}
$$

**Note**: There is no interaction term in a nested model, because Factor B levels are unique within each level of A.

------------------------------------------------------------------------

The least squares and maximum likelihood estimates:

| **Parameter**   | **Estimator**                   |
|-----------------|---------------------------------|
| $\mu_{..}$      | $\bar{Y}_{...}$                 |
| $\alpha_i$      | $\bar{Y}_{i..} - \bar{Y}_{...}$ |
| $\beta_{j(i)}$  | $\bar{Y}_{ij.} - \bar{Y}_{i..}$ |
| $\hat{Y}_{ijk}$ | $\bar{Y}_{ij.}$                 |

The residual error:

$$
e_{ijk} = Y_{ijk} - \bar{Y}_{ij.}
$$

------------------------------------------------------------------------

The total sum of squares (SSTO) is partitioned as:

$$
SSTO = SSA + SSB(A) + SSE
$$

where:

$$
\begin{aligned}
\sum_i \sum_j \sum_k (Y_{ijk} - \bar{Y}_{...})^2
&= bn \sum_i (\bar{Y}_{i..} - \bar{Y}_{...})^2
+ n \sum_i \sum_j (\bar{Y}_{ij.} - \bar{Y}_{i..})^2 \\
&+ \sum_i \sum_j \sum_k (Y_{ijk} - \bar{Y}_{ij.})^2
\end{aligned}
$$

------------------------------------------------------------------------

#### ANOVA Table for Nested Designs

+-------------------------+----------+-----------+----------+---------------------------------------------------+
| **Source of Variation** | **SS**   | **df**    | **MS**   | **E(MS)**                                         |
+=========================+==========+===========+==========+===================================================+
| **Factor A**            | $SSA$    | $a-1$     | $MSA$    | $\sigma^2 + bn \frac{\sum \alpha_i^2}{a-1}$       |
+-------------------------+----------+-----------+----------+---------------------------------------------------+
| **Factor B (A)**        | $SSB(A)$ | $a(b-1)$  | $MSB(A)$ | $\sigma^2 + n \frac{\sum \beta_{j(i)}^2}{a(b-1)}$ |
+-------------------------+----------+-----------+----------+---------------------------------------------------+
| **Error**               | $SSE$    | $ab(n-1)$ | $MSE$    | $\sigma^2$                                        |
+-------------------------+----------+-----------+----------+---------------------------------------------------+
| **Total**               | $SSTO$   | $abn -1$  |          |                                                   |
+-------------------------+----------+-----------+----------+---------------------------------------------------+

------------------------------------------------------------------------

#### Tests For Factor Effects

-   Factor A:

    $$
    F = \frac{MSA}{MSB(A)} \sim F_{(a-1, a(b-1))}
    $$

    Reject $H_0$ if $F > f_{(1-\alpha; a-1, a(b-1))}$.

-   Factor B within A:

    $$
    F = \frac{MSB(A)}{MSE} \sim F_{(a(b-1), ab(n-1))}
    $$

    Reject $H_0$ if $F > f_{(1-\alpha; a(b-1), ab(n-1))}$.

------------------------------------------------------------------------

#### Testing Factor Effect Contrasts

A contrast is a linear combination of factor level means:

$$
L = \sum c_i \mu_i, \quad \text{where} \quad \sum c_i = 0
$$

The estimated contrast:

$$
\hat{L} = \sum c_i \bar{Y}_{i..}
$$

The confidence interval for $L$:

$$
\hat{L} \pm t_{(1-\alpha/2; df)} s(\hat{L})
$$

where:

$$
s^2(\hat{L}) = \sum c_i^2 s^2(\bar{Y}_{i..}), \quad \text{where} \quad s^2(\bar{Y}_{i..}) = \frac{MSE}{bn}, \quad df = ab(n-1)
$$

------------------------------------------------------------------------

#### Testing Treatment Means

For treatment means, a similar approach applies:

$$
L = \sum c_i \mu_{.j}, \quad \hat{L} = \sum c_i \bar{Y}_{ij}
$$

The confidence limits for $L$:

$$
\hat{L} \pm t_{(1-\alpha/2; (n-1)ab)} s(\hat{L})
$$

where:

$$
s^2(\hat{L}) = \frac{MSE}{n} \sum c_i^2
$$

------------------------------------------------------------------------

### Unbalanced Nested Two-Factor Designs

When Factor B has different levels for different levels of Factor A, the design is unbalanced.

$$
\begin{aligned}
Y_{ijk} &= \mu_{..} + \alpha_i + \beta_{j(i)} + \epsilon_{ijk} \\
\sum_{i=1}^2 \alpha_i &= 0, \quad
\sum_{j=1}^3 \beta_{j(1)} = 0, \quad
\sum_{j=1}^2 \beta_{j(2)} = 0
\end{aligned}
$$

where:

-   Factor A: $i = 1, 2$.

-   Factor B (nested in A): $j = 1, \dots, b_i$.

-   Observations: $k = 1, \dots, n_{ij}$.

Example case:

-   $b_1 = 3, b_2 = 2$ (Factor B has different levels for A).

-   $n_{11} = n_{13} = 2, n_{12} = 1, n_{21} = n_{22} = 2$.

-   Parameters: $\alpha_1, \beta_{1(1)}, \beta_{2(1)}, \beta_{1(2)}$.

Constraints:

$$
\alpha_2 = -\alpha_1, \quad
\beta_{3(1)} = -\beta_{1(1)} - \beta_{2(1)}, \quad
\beta_{2(2)} = -\beta_{1(2)}
$$

------------------------------------------------------------------------

The unbalanced design can be modeled using **indicator variables**:

1.  **Factor A (School Level):** $$
    X_1 =
    \begin{cases}
    1 & \text{if observation from school 1} \\
    -1 & \text{if observation from school 2}
    \end{cases}
    $$

2.  **Factor B (Instructor within School 1):** $$
    X_2 =
    \begin{cases}
    1 & \text{if observation from instructor 1 in school 1} \\
    -1 & \text{if observation from instructor 3 in school 1} \\
    0 & \text{otherwise}
    \end{cases}
    $$

3.  **Factor B (Instructor within School 1):** $$
    X_3 =
    \begin{cases}
    1 & \text{if observation from instructor 2 in school 1} \\
    -1 & \text{if observation from instructor 3 in school 1} \\
    0 & \text{otherwise}
    \end{cases}
    $$

4.  **Factor B (Instructor within School 1):** $$
    X_4 =
    \begin{cases}
    1 & \text{if observation from instructor 1 in school 1} \\
    -1 & \text{if observation from instructor 2 in school 1} \\
    0 & \text{otherwise}
    \end{cases}
    $$

Using these indicator variables, the **full regression model** is:

$$
Y_{ijk} = \mu_{..} + \alpha_1 X_{ijk1} + \beta_{1(1)} X_{ijk2} + \beta_{2(1)} X_{ijk3} + \beta_{1(2)} X_{ijk4} + \epsilon_{ijk}
$$

where $X_1, X_2, X_3, X_4$ represent different factor effects.

------------------------------------------------------------------------

### Random Factor Effects

If factors are **random**:

$$
\begin{aligned}
\alpha_1 &\sim iid N(0, \sigma^2_\alpha) \\
\beta_{j(i)} &\sim iid N(0, \sigma^2_\beta)
\end{aligned}
$$

------------------------------------------------------------------------

**Expected Mean Squares for Random Effects**

+-----------------+----------------------------------------------------------------+------------------------------------------------------+
| **Mean Square** | **Expected Mean Squares (A Fixed, B Random)**                  | **Expected Mean Squares (A Random, B Random)**       |
+=================+================================================================+======================================================+
| **MSA**         | $\sigma^2 + n \sigma^2_\beta + bn \frac{\sum \alpha_i^2}{a-1}$ | $\sigma^2 + bn \sigma^2_{\alpha} + n \sigma^2_\beta$ |
+-----------------+----------------------------------------------------------------+------------------------------------------------------+
| **MSB(A)**      | $\sigma^2 + n \sigma^2_\beta$                                  | $\sigma^2 + n \sigma^2_\beta$                        |
+-----------------+----------------------------------------------------------------+------------------------------------------------------+
| **MSE**         | $\sigma^2$                                                     | $\sigma^2$                                           |
+-----------------+----------------------------------------------------------------+------------------------------------------------------+

------------------------------------------------------------------------

**F-Tests for Factor Effects**

+-----------------+--------------------------------+---------------------------------+
| **Factor**      | **F-Test (A Fixed, B Random)** | **F-Test (A Random, B Random)** |
+=================+================================+=================================+
| **Factor A**    | $\frac{MSA}{MSB(A)}$           | $\frac{MSA}{MSB(A)}$            |
+-----------------+--------------------------------+---------------------------------+
| **Factor B(A)** | $\frac{MSB(A)}{MSE}$           | $\frac{MSB(A)}{MSE}$            |
+-----------------+--------------------------------+---------------------------------+

------------------------------------------------------------------------

Another way to increase precision in treatment comparisons is by adjusting for covariates using regression models. This is called Analysis of Covariance (ANCOVA).

**Why use ANCOVA?**

-   Reduces variability by accounting for covariate effects.

-   Increases statistical power by removing nuisance variation.

-   Combines ANOVA and regression for more precise comparisons.

------------------------------------------------------------------------

## Sample Size Planning for ANOVA

### Balanced Designs

Choosing an appropriate sample size for an ANOVA study requires ensuring sufficient power while balancing practical constraints.

### Single Factor Studies

#### Fixed Cell Means Model

The probability of rejecting $H_0$ when it is false (power) is given by:

$$
P(F > f_{(1-\alpha; a-1, N-a)} | \phi) = 1 - \beta
$$

where:

-   $\phi$ is the non-centrality parameter (measuring the inequality among treatment means $\mu_i$):

    $$
    \phi = \frac{1}{\sigma} \sqrt{\frac{n}{a} \sum_{i} (\mu_i - \mu_.)^2}, \quad (n_i \equiv n)
    $$

-   $\mu_.$ is the overall mean:

    $$
    \mu_. = \frac{\sum \mu_i}{a}
    $$

To determine power, we use the non-central F distribution.

------------------------------------------------------------------------

**Using Power Tables**

Power tables can be used directly when:

1.  The effects are fixed.

2.  The design is balanced.

3.  The minimum range of factor level means $\Delta$ is known:

    $$
    \Delta = \max(\mu_i) - \min(\mu_i)
    $$

Thus, the required inputs are:

-   Significance level ($\alpha$)
-   Minimum range of means ($\Delta$)
-   Error standard deviation ($\sigma$)
-   Power ($1 - \beta$)

Notes on Sample Size Sensitivity

-   When $\Delta/\sigma$ is small, sample size requirements increase dramatically.
-   Lowering $\alpha$ or $\beta$ increases required sample sizes.
-   Errors in estimating $\sigma$ can significantly impact sample size calculations.

------------------------------------------------------------------------

### Multi-Factor Studies

The same noncentral $F$ tables apply for multi-factor models.

#### Two-Factor Fixed Effects Model

##### Test for Interaction Effects

The non-centrality parameter:

$$
\phi = \frac{1}{\sigma} \sqrt{\frac{n \sum_i \sum_j (\alpha \beta)_{ij}^2}{(a-1)(b-1)+1}}
$$

or equivalently:

$$
\phi = \frac{1}{\sigma} \sqrt{\frac{n \sum_i \sum_j (\mu_{ij} - \mu_{i.} - \mu_{.j} + \mu_{..})^2}{(a-1)(b-1)+1}}
$$

where degrees of freedom are:

$$
\begin{aligned}
\upsilon_1 &= (a-1)(b-1) \\
\upsilon_2 &= ab(n-1)
\end{aligned}
$$

------------------------------------------------------------------------

##### Test for Factor $A$ Main Effects

The non-centrality parameter:

$$
\phi = \frac{1}{\sigma} \sqrt{\frac{nb \sum \alpha_i^2}{a}}
$$

or equivalently:

$$
\phi = \frac{1}{\sigma} \sqrt{\frac{nb \sum (\mu_{i.} - \mu_{..})^2}{a}}
$$

where degrees of freedom are:

$$
\begin{aligned}
\upsilon_1 &= a-1 \\
\upsilon_2 &= ab(n-1)
\end{aligned}
$$

------------------------------------------------------------------------

##### Test for Factor $B$ Main Effects

The non-centrality parameter:

$$
\phi = \frac{1}{\sigma} \sqrt{\frac{na \sum \beta_j^2}{b}}
$$

or equivalently:

$$
\phi = \frac{1}{\sigma} \sqrt{\frac{na \sum (\mu_{.j} - \mu_{..})^2}{b}}
$$

where degrees of freedom are:

$$
\begin{aligned}
\upsilon_1 &= b-1 \\
\upsilon_2 &= ab(n-1)
\end{aligned}
$$

------------------------------------------------------------------------

### Procedure for Sample Size Selection

1.  Specify the minimum range of Factor $A$ means.
2.  Obtain sample size from power tables using $r = a$.
    -   The resulting sample size is $bn$, from which $n$ can be derived.
3.  Repeat steps 1-2 for Factor $B$.
4.  Choose the larger sample size from the calculations for Factors $A$ and $B$.

------------------------------------------------------------------------

### Randomized Block Experiments

Analogous to completely randomized designs . The power of the F-test for treatment effects for randomized block design uses the same non-centrality parameter as completely randomized design:

$$
\phi = \frac{1}{\sigma} \sqrt{\frac{n}{r} \sum (\mu_i - \mu_.)^2}
$$

However, the power level is different from the randomized block design because

-   error variance $\sigma^2$ is different
-   df(MSE) is different.

------------------------------------------------------------------------

## Single Factor Covariance Model

The **single-factor covariance model** (Analysis of Covariance, ANCOVA) accounts for both treatment effects and a continuous covariate:

$$
Y_{ij} = \mu_{.} + \tau_i + \gamma(X_{ij} - \bar{X}_{..}) + \epsilon_{ij} 
$$

for $i = 1, \dots, r$ (treatments) and $j = 1, \dots, n_i$ (observations per treatment).

-   $\mu_{.}$: Overall mean response.
-   $\tau_i$: Fixed treatment effects ($\sum \tau_i = 0$).
-   $\gamma$: Fixed regression coefficient (relationship between covariate $X$ and response $Y$).
-   $X_{ij}$: Observed covariate (fixed, not random).
-   $\epsilon_{ij} \sim iid N(0, \sigma^2)$: Independent random errors.

If we use $\gamma X_{ij}$ directly (without centering), then $\mu_{.}$ is no longer the overall mean. **Thus, centering the covariate** is necessary to maintain interpretability.

**Expectation and Variance**

$$
\begin{aligned}
E(Y_{ij}) &= \mu_. + \tau_i + \gamma(X_{ij}-\bar{X}_{..}) \\
var(Y_{ij}) &= \sigma^2
\end{aligned}
$$

Since $Y_{ij} \sim N(\mu_{ij},\sigma^2)$, we express:

$$
\mu_{ij} = \mu_. + \tau_i + \gamma(X_{ij} - \bar{X}_{..})
$$

where $\sum \tau_i = 0$. The mean response $\mu_{ij}$ is a regression line with intercept $\mu_. + \tau_i$ and slope $\gamma$ for each treatment $i$.

------------------------------------------------------------------------

**Key Assumptions**

1.  All treatments share the same slope ($\gamma$).
2.  No interaction between treatment and covariate (parallel regression lines).
3.  If slopes differ, ANCOVA is not appropriate → use separate regressions per treatment.

A more general model allows multiple covariates:

$$
Y_{ij} = \mu_. + \tau_i + \gamma_1(X_{ij1}-\bar{X}_{..1}) + \gamma_2(X_{ij2}-\bar{X}_{..2}) + \epsilon_{ij}
$$

------------------------------------------------------------------------

Using indicator variables for treatments:

For treatment $i = 1$: $$
l_1 =
\begin{cases}
1 & \text{if case belongs to treatment 1} \\
-1 & \text{if case belongs to treatment $r$} \\
0 & \text{otherwise}
\end{cases}
$$

For treatment $i = r-1$: $$
l_{r-1} =
\begin{cases}
1 & \text{if case belongs to treatment $r-1$} \\
-1 & \text{if case belongs to treatment $r$} \\
0 & \text{otherwise}
\end{cases}
$$

Defining $x_{ij} = X_{ij}- \bar{X}_{..}$, the regression model is:

$$
Y_{ij} = \mu_. + \tau_1 l_{ij,1} + \dots + \tau_{r-1} l_{ij,r-1} + \gamma x_{ij} + \epsilon_{ij}
$$

where $I_{ij,1}$ is the indicator variable $l_1$ for the $j$-th case in treatment $i$.

The treatment effects ($\tau_i$) are simply regression coefficients for the indicator variables.

------------------------------------------------------------------------

### Statistical Inference for Treatment Effects

To test treatment effects:

$$
\begin{aligned}
&H_0: \tau_1 = \tau_2 = \dots = 0 \\
&H_a: \text{Not all } \tau_i = 0
\end{aligned}
$$

1.  **Full Model (with treatment effects):** $$
    Y_{ij} = \mu_. + \tau_i + \gamma X_{ij} + \epsilon_{ij}
    $$

2.  **Reduced Model (without treatment effects):** $$
    Y_{ij} = \mu_. + \gamma X_{ij} + \epsilon_{ij}
    $$

**F-Test for Treatment Effects**

The test statistic is:

$$
F = \frac{SSE(R) - SSE(F)}{(N-2)-(N-(r+1))} \Big/ \frac{SSE(F)}{N-(r+1)}
$$

where:

-   $SSE(R)$: Sum of squared errors for the **reduced model**.

-   $SSE(F)$: Sum of squared errors for the **full model**.

-   $N$: Total number of observations.

-   $r$: Number of treatment groups.

Under $H_0$, the statistic follows an $F$-distribution:

$$
F \sim F_{(r-1, N-(r+1))}
$$

------------------------------------------------------------------------

**Comparisons of Treatment Effects**

For $r = 3$, we estimate:

+-------------------+---------------------------------+-----------------------------------------------------------------------------+
| **Comparison**    | **Estimate**                    | **Variance of Estimator**                                                   |
+===================+=================================+=============================================================================+
| $\tau_1 - \tau_2$ | $\hat{\tau}_1 - \hat{\tau}_2$   | $var(\hat{\tau}_1) + var(\hat{\tau}_2) - 2cov(\hat{\tau}_1, \hat{\tau}_2)$  |
+-------------------+---------------------------------+-----------------------------------------------------------------------------+
| $\tau_1 - \tau_3$ | $2 \hat{\tau}_1 + \hat{\tau}_2$ | $4var(\hat{\tau}_1) + var(\hat{\tau}_2) - 4cov(\hat{\tau}_1, \hat{\tau}_2)$ |
+-------------------+---------------------------------+-----------------------------------------------------------------------------+
| $\tau_2 - \tau_3$ | $\hat{\tau}_1 + 2 \hat{\tau}_2$ | $var(\hat{\tau}_1) + 4var(\hat{\tau}_2) - 4cov(\hat{\tau}_1, \hat{\tau}_2)$ |
+-------------------+---------------------------------+-----------------------------------------------------------------------------+

------------------------------------------------------------------------

### Testing for Parallel Slopes

To check if **slopes differ across treatments**, we use the model:

$$
Y_{ij} = \mu_{.} + \tau_1 I_{ij,1} + \tau_2 I_{ij,2} + \gamma X_{ij} + \beta_1 I_{ij,1}X_{ij} + \beta_2 I_{ij,2}X_{ij} + \epsilon_{ij}
$$

where:

-   $\beta_1, \beta_2$: Interaction coefficients (slope differences across treatments).

**Hypothesis Test**

$$
\begin{aligned}
&H_0: \beta_1 = \beta_2 = 0 \quad (\text{Slopes are equal}) \\
&H_a: \text{At least one } \beta \neq 0 \quad (\text{Slopes differ})
\end{aligned}
$$

If the $F$-test fails to reject $H_0$, then **we assume parallel slopes**.

------------------------------------------------------------------------

### Adjusted Means

The **adjusted treatment means** account for covariate effects:

$$
Y_{i.}(\text{adj}) = \bar{Y}_{i.} - \hat{\gamma}(\bar{X}_{i.} - \bar{X}_{..})
$$

where:

-   $\bar{Y}_{i.}$: Observed mean response for treatment $i$.

-   $\hat{\gamma}$: Estimated regression coefficient.

-   $\bar{X}_{i.}$: Mean covariate value for treatment $i$.

-   $\bar{X}_{..}$: Overall mean covariate value.

This provides estimated treatment means after controlling for covariate effects.

------------------------------------------------------------------------
