# Analysis of Variance (ANOVA) {#sec-analysis-of-variance-anova}

ANOVA is using the same underlying mechanism as linear regression. However, the angle that ANOVA chooses to look at is slightly different from the traditional linear regression. It can be more useful in the case with **qualitative variables** and **designed experiments**.

Experimental Design

-   **Factor**: explanatory or predictor variable to be studied in an investigation
-   **Treatment** (or Factor Level): "value" of a factor applied to the experimental unit
-   **Experimental Unit**: person, animal, piece of material, etc. that is subjected to treatment(s) and provides a response
-   **Single Factor Experiment**: one explanatory variable considered
-   **Multifactor Experiment**: more than one explanatory variable
-   **Classification Factor**: A factor that is not under the control of the experimenter (observational data)
-   **Experimental Factor**: assigned by the experimenter

Basics of experimental design:

-   Choices that a statistician has to make:

    -   set of treatments
    -   set of experimental units
    -   treatment assignment (selection bias)
    -   measurement (measurement bias, blind experiments)

-   Advancements in experimental design:

    1.  **Factorial Experiments**:\
        consider multiple factors at the same time (interaction)

    2.  **Replication**: repetition of experiment

        -   assess mean squared error
        -   control over precision of experiment (power)

    3.  **Randomization**

        -   Before R.A. Fisher (1900s), treatments were assigned systematically or subjectively
        -   randomization: assign treatments to experimental units at random, which averages out systematic effects that cannot be control by the investigator

    4.  **Local control**: Blocking or Stratification

        -   Reduce experimental errors and increase power by placing restrictions on the randomization of treatments to experimental units.

Randomization may also eliminate correlations due to time and space.

## Completely Randomized Design (CRD)

Treatment factor A with $a\ge2$ treatments levels. Experimental units are randomly assigned to each treatment. The number of experimental units in each group can be

-   equal (balanced): n
-   unequal (unbalanced): $n_i$ for the i-th group (i = 1,...,a).

The total sample size is $N=\sum_{i=1}^{a}n_i$

Possible assignments of units to treatments are $k=\frac{N!}{n_1!n_2!...n_a!}$

Each has probability 1/k of being selected. Each experimental unit is measured with a response $Y_{ij}$, in which j denotes unit and i denotes treatment.

Treatment

+-------------+----------------+----------------+------------+----------------+
|             | 1              | 2              | ...        | a              |
+:============+:==============:+:==============:+:==========:+:==============:+
|             | $Y_{11}$       | $Y_{21}$       | ...        | $Y_{a1}$       |
+-------------+----------------+----------------+------------+----------------+
|             | $Y_{12}$       | ...            | ...        | ...            |
+-------------+----------------+----------------+------------+----------------+
|             | ...            | ...            | ...        | ...            |
+-------------+----------------+----------------+------------+----------------+
| Sample Mean | $\bar{Y_{1.}}$ | $\bar{Y_{2.}}$ | ...        | $\bar{Y_{a.}}$ |
+-------------+----------------+----------------+------------+----------------+
| Sample SD   | $s_1$          | $s_2$          | ...        | $s_a$          |
+-------------+----------------+----------------+------------+----------------+

where $\bar{Y_{i.}}=\frac{1}{n_i}\sum_{j=1}^{n_i}Y_{ij}$

$s_i^2=\frac{1}{n_i-1}\sum_{j=1}^{n_i}(Y_{ij}-\bar{Y_i})^2$

And the grand mean is $\bar{Y_{..}}=\frac{1}{N}\sum_{i}\sum_{j}Y_{ij}$

### Single Factor Fixed Effects Model

also known as Single Factor (One-Way) ANOVA or ANOVA Type I model.

Partitioning the Variance

The total variability of the $Y_{ij}$ observation can be measured as the deviation of $Y_{ij}$ around the overall mean $\bar{Y_{..}}$: $Y_{ij} - \bar{Y_{..}}$

This can be rewritten as: 

$$
\begin{aligned}
Y_{ij} - \bar{Y_{..}}&=Y_{ij} - \bar{Y_{..}} + \bar{Y_{i.}} - \bar{Y_{i.}} \\
&= (\bar{Y_{i.}}-\bar{Y_{..}})+(Y_{ij}-\bar{Y_{i.}})
\end{aligned}
$$ 

where

-   the first term is the *between* treatment differences (i.e., the deviation of the treatment mean from the overall mean)
-   the second term is *within* treatment differences (i.e., the deviation of the observation around its treatment mean)

$$
\begin{aligned}
\sum_{i}\sum_{j}(Y_{ij} - \bar{Y_{..}})^2 &=  \sum_{i}n_i(\bar{Y_{i.}}-\bar{Y_{..}})^2+\sum_{i}\sum_{j}(Y_{ij}-\bar{Y_{i.}})^2 \\
SSTO &= SSTR + SSE \\
total~SS &= treatment~SS + error~SS \\
(N-1)~d.f. &= (a-1)~d.f. + (N - a) ~ d.f.
\end{aligned}
$$

we lose a d.f. for the total corrected SSTO because of the estimation of the mean ($\sum_{i}\sum_{j}(Y_{ij} - \bar{Y_{..}})=0$)\
And, for the SSTR $\sum_{i}n_i(\bar{Y_{i.}}-\bar{Y_{..}})=0$

Accordingly, $MSTR= \frac{SST}{a-1}$ and $MSR=\frac{SSE}{N-a}$

**ANOVA Table**

+---------------------------+---------------------------------------------+------------+------------+
| Source of Variation       | SS                                          | df         | MS         |
+===========================+:===========================================:+:==========:+:==========:+
| Between Treatments        | $\sum_{i}n_i (\bar{Y_{i.}}-\bar{Y_{..}})^2$ | a-1        | SSTR/(a-1) |
+---------------------------+---------------------------------------------+------------+------------+
| Error (within treatments) | $\sum_{i}\sum_{j}(Y_{ij}-\bar{Y_{i.}})^2$   | N-a        | SSE/(N-a)  |
+---------------------------+---------------------------------------------+------------+------------+
| Total (corrected)         | $\sum_{i}n_i (\bar{Y_{i.}}-\bar{Y_{..}})^2$ | N-1        |            |
+---------------------------+---------------------------------------------+------------+------------+

Linear Model Explanation of ANOVA

#### Cell means model

$$
Y_{ij}=\mu_i+\epsilon\_{ij}
$$

where

-   $Y_{ij}$ response variable in $j$-th subject for the $i$-th treatment

-   $\mu_i$: parameters (fixed) representing the unknown population mean for the i-th treatment

-   $\epsilon_{ij}$ independent $N(0,\sigma^2)$ errors

-   $E(Y_{ij})=\mu_i$ $var(Y_{ij})=var(\epsilon_{ij})=\sigma^2$

-   All observations have the same variance

Example:

$a = 3$ (3 treatments) $n_1=n_2=n_3=2$

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

$X_{k,ij}=1$ if the $k$-th treatment is used

$X_{k,ij}=0$ Otherwise

**Note**: no intercept term.


\begin{equation}
\begin{aligned}
\mathbf{b}= \left[\begin{array}{c}
\mu_1 \\
\mu_2 \\
\mu_3 \\
\end{array}\right] &= 
(\mathbf{x}'\mathbf{x})^{-1}\mathbf{x}'\mathbf{y} \\
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


is the BLUE (best linear unbiased estimator) for $\beta=[\mu_1 \mu_2\mu_3]'$

$$
E(\mathbf{b})=\beta
$$

$$
var(\mathbf{b})=\sigma^2(\mathbf{X'X})^{-1}=\sigma^2
\left[\begin{array}{ccc}
1/n_1 & 0 & 0\\
0 & 1/n_2 & 0\\
0 & 0 & 1/n_3\\
\end{array}\right]
$$

$var(b_i)=var(\hat{\mu_i})=\sigma^2/n_i$ where $\mathbf{b} \sim N(\beta,\sigma^2(\mathbf{X'X})^{-1})$

$$
\begin{aligned}
MSE &= \frac{1}{N-a} \sum_{i}\sum_{j}(Y_{ij}-\bar{Y_{i.}})^2 \\
    &= \frac{1}{N-a} \sum_{i}[(n_i-1)\frac{\sum_{i}(Y_{ij}-\bar{Y_{i.}})^2}{n_i-1}] \\
    &= \frac{1}{N-a} \sum_{i}(n_i-1)s_1^2
\end{aligned}
$$

We have $E(s_i^2)=\sigma^2$

$E(MSE)=\frac{1}{N-a}\sum_{i}(n_i-1)\sigma^2=\sigma^2$

Hence, MSE is an unbiased estimator of $\sigma^2$, regardless of whether the treatment means are equal or not.

$E(MSTR)=\sigma^2+\frac{\sum_{i}n_i(\mu_i-\mu_.)^2}{a-1}$\
where $\mu_.=\frac{\sum_{i=1}^{a}n_i\mu_i}{\sum_{i=1}^{a}n_i}$\
If all treatment means are equals (=$\mu_.$), $E(MSTR)=\sigma^2$.

Then we can use an $F$-test for the equality of all treatment means:

$$H_0:\mu_1=\mu_2=..=\mu_a$$

$$H_a: not~al l~ \mu_i ~ are ~ equal $$

$F=\frac{MSTR}{MSE}$\
where large values of F support $H_a$ (since MSTR will tend to exceed MSE when $H_a$ holds)\
and F near 1 support $H_0$ (upper tail test)

**Equivalently**, when $H_0$ is true, $F \sim f_{(a-1,N-a)}$

-   If $F \leq f_{(a-1,N-a;1-\alpha)}$, we cannot reject $H_0$
-   If $F \geq f_{(a-1,N-a;1-\alpha)}$, we reject $H_0$

Note: If $a = 2$ (2 treatments), $F$-test = two sample $t$-test

#### Treatment Effects (Factor Effects)

Besides Cell means model, we have another way to formalize one-way ANOVA: $$Y_{ij} = \mu + \tau_i + \epsilon_{ij}$$ where

-   $Y_{ij}$ is the $j$-th response for the $i$-th treatment
-   $\tau_i$ is $i$-th treatment effect
-   $\mu$ constant component, common to all observations
-   $\epsilon_{ij}$ independent random errors \~ $N(0,\sigma^2)$

For example, $a = 3$, $n_1=n_2=n_3=2$


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


However,

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

is **singular** thus does not exist, $\mathbf{b}$ is insolvable (infinite solutions)

Hence, we have to impose restrictions on the parameters to a model matrix $\mathbf{X}$ of full rank.

Whatever restriction we use, we still have:

$E(Y_{ij})=\mu + \tau_i = \mu_i = mean ~ response ~ for ~ i-th ~ treatment$

##### Restriction on sum of tau {#restriction-on-sum-of-tau}

$\sum_{i=1}^{a}\tau_i=0$

implies

$$
\mu= \mu +\frac{1}{a}\sum_{i=1}^{a}(\mu+\tau_i)
$$

is the average of the treatment mean (grand mean) (overall mean)

$$
\begin{aligned}
\tau_i  &=(\mu+\tau_i) -\mu = \mu_i-\mu \\
        &= \text{treatment  mean} - \text{grand~mean} \\
        &= \text{treatment  effect}
\end{aligned}
$$

$$
\tau_a=-\tau_1-\tau_2-...-\tau_{a-1}
$$

Hence, the mean for the a-th treatment is

$$
\mu_a=\mu+\tau_a=\mu-\tau_1-\tau_2-...-\tau_{a-1}
$$

Hence, the model need only "a" parameters:

$$
\mu,\tau_1,\tau_2,..,\tau_{a-1}
$$

Equation \@ref(eq:unsolvable) becomes


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


where $\beta\equiv[\mu,\tau_1,\tau_2]'$

Equation \@ref(eq:betaorigin) with $\sum_{i}\tau_i=0$ becomes

$$
\begin{aligned}
\mathbf{b}= \left[\begin{array}{c}
\hat{\mu} \\
\hat{\tau_1} \\
\hat{\tau_2} \\
\end{array}\right] &= 
(\mathbf{x}'\mathbf{x})^{-1}\mathbf{x}'\mathbf{y} \\
& = 
\left[\begin{array}{ccc}
\sum_{i}n_i & n_1-n_3 & n_2-n_3\\
n_1-n_3 & n_1+n_3 & n_3\\
n_2-n_3 & n_3 & n_2-n_3 \\
\end{array}\right]^{-1}
\left[\begin{array}{c}
Y_{..}\\
Y_{1.}-Y_{3.}\\
Y_{2.}-Y_{3.}\\
\end{array}\right] \\
& =
\left[\begin{array}{c}
\frac{1}{3}\sum_{i=1}^{3}\bar{Y_{i.}}\\
\bar{Y_{1.}}-\frac{1}{3}\sum_{i=1}^{3}\bar{Y_{i.}}\\
\bar{Y_{2.}}-\frac{1}{3}\sum_{i=1}^{3}\bar{Y_{i.}}\\
\end{array}\right]\\
& = 
\left[\begin{array}{c}
\hat{\mu}\\
\hat{\tau_1}\\
\hat{\tau_2}\\
\end{array}\right]
\end{aligned}
$$

and $\hat{\tau_3}=-\hat{\tau_1}-\hat{\tau_2}=\bar{Y_3}-\frac{1}{3} \sum_{i}\bar{Y_{i.}}$

##### Restriction on first tau {#restriction-on-first-tau}

In R, `lm()` uses the restriction $\tau_1=0$

For the previous example, for $n_1=n_2=n_3=2$, and $\tau_1=0$.

Then the treatment means can be written as:

$$
\begin{aligned}
\mu_1 &= \mu + \tau_1 = \mu + 0 = \mu  \\
\mu_2 &= \mu + \tau_2 \\
\mu_3 &= \mu + \tau_3
\end{aligned}
$$

Hence, $\mu$ is the mean response for the first treatment

In the matrix form,

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
1 & 1 & 0 \\ 
1 & 1 & 0 \\ 
1 & 0 & 1 \\ 
1 & 0 & 1 \\ 
\end{array}\right)
\left(\begin{array}{c}
\mu \\
\tau_2 \\
\tau_3 \\
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

$\beta = [\mu,\tau_2,\tau_3]'$

$$
\begin{aligned}
\mathbf{b}= \left[\begin{array}{c}
\hat{\mu} \\
\hat{\tau_2} \\
\hat{\tau_3} \\
\end{array}\right] &= 
(\mathbf{x}'\mathbf{x})^{-1}\mathbf{x}'\mathbf{y} \\
& = 
\left[\begin{array}{ccc}
\sum_{i}n_i & n_2 & n_3\\
n_2 & n_2 & 0\\
n_3 & 0 & n_3 \\
\end{array}\right]^{-1}
\left[\begin{array}{c}
Y_{..}\\
Y_{2.}\\
Y_{3.}\\
\end{array}\right] \\
& = 
\left[
\begin{array}{c}
\bar{Y_{1.}} \\
\bar{Y_{2.}} - \bar{Y_{1.}} \\
\bar{Y_{3.}} - \bar{Y_{1.}}\\
\end{array}\right]
\end{aligned}
$$

$$
E(\mathbf{b})= \beta = 
\left[\begin{array}{c}
{\mu}\\
{\tau_2}\\
{\tau_3}\\
\end{array}\right]
=
\left[\begin{array}{c}
\mu_1\\
\mu_2-\mu_1\\
\mu_3-\mu_1\\
\end{array}\right]
$$

$$
\begin{aligned}
var(\mathbf{b}) &= \sigma^2(\mathbf{X'X})^{-1} \\
var(\hat{\mu}) &= var(\bar{Y_{1.}})=\sigma^2/n_1 \\
var(\hat{\tau_2}) &= var(\bar{Y_{2.}}-\bar{Y_{1.}}) = \sigma^2/n_2 + \sigma^2/n_1 \\
var(\hat{\tau_3}) &= var(\bar{Y_{3.}}-\bar{Y_{1.}}) = \sigma^2/n_3 + \sigma^2/n_1
\end{aligned}
$$

**Note** For all three parameterization, the ANOVA table is the same

-   [Model 1](#cell-means-model-1): $Y_{ij} = \mu_i + \epsilon_{ij}$
-   [Model 2](#restriction-on-sum-of-tau): $Y_{ij} = \mu + \tau_i + \epsilon_{ij}$ where $\sum_{i} \tau_i=0$
-   [Model 3](#restriction-on-first-tau): $Y_{ij}= \mu + \tau_i + \epsilon_{ij}$ where $\tau_1=0$

All models have the same calculation for $\hat{Y}$ as

$$
\mathbf{\hat{Y} = X(X'X)^{-1}X'Y=PY = Xb}
$$

**ANOVA Table**

+---------------------+------------------------------------------------------------------------------+---------+--------------------+--------------------+
| Source of Variation | SS                                                                           | df      | MS                 | F                  |
+=====================+:============================================================================:+:=======:+:==================:+:==================:+
| Between Treatments  | $\sum_{i} n _ i (\bar { Y_ {i .} } -\bar{Y_{..}})^2 = \mathbf{Y ' (P-P_1)Y}$ | a-1     | $\frac{SSTR}{a-1}$ | $\frac{MSTR}{MSE}$ |
+---------------------+------------------------------------------------------------------------------+---------+--------------------+--------------------+
| Error               | $\sum_{i}\sum_{j}(Y_{ij} -\bar{Y_{i.}})^2=\mathbf{e'e}$                      | N-a     | $\frac{SSE}{N-a}$  |                    |
|                     |                                                                              |         |                    |                    |
| (within treatments) |                                                                              |         |                    |                    |
+---------------------+------------------------------------------------------------------------------+---------+--------------------+--------------------+
| Total (corrected)   | $\sum_{i } n_i(\bar{Y_{i.}}-\bar{Y_{..}})^2=\mathbf{Y'Y - Y'P_1Y}$           | N-1     |                    |                    |
+---------------------+------------------------------------------------------------------------------+---------+--------------------+--------------------+

where $\mathbf{P_1} = \frac{1}{n}\mathbf{J}$

The $F$-statistic here has $(a-1,N-a)$ degrees of freedom, which gives the same value for all three parameterization, but the hypothesis test is written a bit different:

$$
\begin{aligned}
&H_0 : \mu_1 = \mu_2 = ... = \mu_a \\
&H_0 : \mu + \tau_1 = \mu + \tau_2 = ... = \mu + \tau_a \\
&H_0 : \tau_1 = \tau_2 = ...= \tau_a 
\end{aligned}
$$

The $F$-test here serves as a preliminary analysis, to see if there is any difference at different factors. For more in-depth analysis, we consider different testing of treatment effects.

#### Testing of Treatment Effects

-   A [Single Treatment Mean] $\mu_i$
-   A [Differences Between Treatment Means]
-   A [Contrast Among Treatment Means]
-   A [Linear Combination of Treatment Means]

##### Single Treatment Mean

We have $\hat{\mu_i}=\bar{Y_{i.}}$ where

-   $E(\bar{Y_{i.}})=\mu_i$
-   $var(\bar{Y_{i}})=\sigma^2/n_i$ estimated by $s^2(\bar{Y_{i.}})=MSE / n_i$

Since $\frac{\bar{Y_{i.}}-\mu_i}{s(\bar{Y_{i.}})} \sim t_{N-a}$ and the confidence interval for $\mu_i$ is $\bar{Y_{i.}} \pm t_{1-\alpha/2;N-a}s(\bar{Y_{i.}})$,\
then we can do a t-test for the means difference with some constant $c$

$$
\begin{aligned}
&H_0: \mu_i = c \\
&H_1: \mu_i \neq c
\end{aligned}
$$

where

$$
T =\frac{\bar{Y_{i.}}-c}{s(\bar{Y_{i.}})}
$$

follows $t_{N-a}$ when $H_0$ is true.\
If $|T| > t_{1-\alpha/2;N-a}$, we can reject $H_0$

##### Differences Between Treatment Means

Let $D=\mu_i - \mu_i'$, also known as **pairwise comparison**\
$D$ can be estimated by $\hat{D}=\bar{Y_{i}}-\bar{Y_{i}}'$ is unbiased ($E(\hat{D})=\mu_i-\mu_i'$)

Since $\bar{Y_{i}}$ and $\bar{Y_{i}}'$ are independent, then

$$
var(\hat{D})=var(\bar{Y_{i}}) + var(\bar{Y_{i'}}) = \sigma^2(1/n_i + 1/n_i')
$$

can be estimated with

$$
s^2(\hat{D}) = MSE(1/n_i + 1/n_i')
$$

With the single treatment inference,

$$
\frac{\hat{D}-D}{s(\hat{D})} \sim t_{N-a}
$$

hence,

$$
\hat{D} \pm t_{(1-\alpha/2;N-a)}s(\hat{D})
$$

Hypothesis tests:

$$
\begin{aligned}
&H_0: \mu_i = \mu_i' \\
&H_a: \mu_i \neq \mu_i'
\end{aligned}
$$

can be tested by the following statistic

$$
T = \frac{\hat{D}}{s(\hat{D})} \sim t_{1-\alpha/2;N-a}
$$

reject $H_0$ if $|T| > t_{1-\alpha/2;N-a}$

##### Contrast Among Treatment Means

generalize the comparison of two means, we have **contrasts**

A contrast is a linear combination of treatment means:

$$
L = \sum_{i=1}^{a}c_i \mu_i
$$

where each $c_i$ is non-random constant and sum to 0:

$$
\sum_{i=1}^{a} c_i = 0
$$

An unbiased estimator of a contrast L is

$$
\hat{L} = \sum_{i=1}^{a}c_i \bar{Y}_{i.}
$$

and $E(\hat{L}) = L$. Since the $\bar{Y}_{i.}$, i = 1,..., a are independent.

$$
\begin{aligned}
var(\hat{L}) &= var(\sum_{i=1}^a c_i \bar{Y}_{i.}) = \sum_{i=1}^a var(c_i \bar{Y}_i)  \\
&= \sum_{i=1}^a c_i^2 var(\bar{Y}_i) = \sum_{i=1}^a c_i^2 \sigma^2 /n_i \\
&= \sigma^2 \sum_{i=1}^{a} c_i^2 /n_i
\end{aligned}
$$

Estimation of the variance:

$$
s^2(\hat{L}) = MSE \sum_{i=1}^{a} \frac{c_i^2}{n_i}
$$

$\hat{L}$ is normally distributed (since it is a linear combination of independent normal random variables).

Then, since $SSE/\sigma^2$ is $\chi_{N-a}^2$

$$
\frac{\hat{L}-L}{s(\hat{L})} \sim t_{N-a}
$$

A $1-\alpha$ confidence limits are given by

$$
\hat{L} \pm t_{1-\alpha/2; N-a}s(\hat{L})
$$

Hypothesis testing

$$
\begin{aligned}
&H_0: L = 0 \\
&H_a: L \neq 0
\end{aligned}
$$

with

$$
T = \frac{\hat{L}}{s(\hat{L})}
$$

reject $H_0$ if $|T| > t_{1-\alpha/2;N-a}$

##### Linear Combination of Treatment Means

just like contrast $L = \sum_{i=1}^a c_i \mu_i$ but no restrictions on the $c_i$ coefficients.

Tests on a single treatment mean, two treatment means, and contrasts can all be considered form the same perspective.

$$
\begin{aligned}
&H_0: \sum c_i \mu_i = c \\
&H_a: \sum c_i \mu_i \neq c 
\end{aligned}
$$

The test statistics ( $t$-stat) can be considered equivalently as $F$-tests; $F = (T)^2$ where $F \sim F_{1,N-a}$. Since the numerator degrees of freedom is always 1 in these cases, we refer to them as single-degree-of-freedom tests.

**Multiple Contrasts**

To test simultaneously $k \ge 2$ contrasts, let $T_1,...,T_k$ be the t-stat. The joint distribution of these random variables is a multivariate t-distribution (the tests are dependent since they re based on the same data).

Limitations for comparing multiple contrasts:

1.  The confidence coefficient $1-\alpha$ only applies to a particular estimate, not a series of estimates; similarly, the Type I error rate, $\alpha$, applies to a particular test, not a series of tests. Example: 3 $t$-tests at $\alpha = 0.05$, if tests are independent (which they are not), $0.95^3 = 0.857$ (thus $\alpha - 0.143$ not 0.05)\

2.  The confidence coefficient $1-\alpha$ and significance level $\alpha$ are appropriate only if the test was not suggest by the data.

    -   often, the results of an experiment suggest important (i.e.,..g, potential significant) relationships.
    -   the process of studying effects suggests by the data is called **data snooping**

Multiple Comparison Procedures:

-   [Tukey]
-   [Scheffe]
-   [Bonferroni]

###### Tukey

All pairwise comparisons of factor level means. All pairs $D = \mu_i - \mu_i'$ or all tests of the form:

$$
\begin{aligned}
&H_0: \mu_i -\mu_i' = 0 \\
&H_a: \mu_i - \mu_i' \neq 0
\end{aligned}
$$

-   When all sample sizes are equal ($n_1 = n_2 = ... = n_a$) then the Tukey method family confidence coefficient is exactly $1-\alpha$ and the significance level is exactly $\alpha$\
-   When the sample sizes are not equal, the family confidence coefficient is greater than $1-\alpha$ (i.e., the significance level is less than $\alpha$) so the test **conservative**\
-   Tukey considers the **studentized range distribution**. If we have $Y_1,..,Y_r$, observations from a normal distribution with mean $\alpha$ and variance $\sigma^2$. Define: $$
    w = max(Y_i) - min(Y_i)
    $$ as the range of the observations. Let $s^2$ be an estimate of $\sigma^2$ with v degrees of freedom. Then, $$
    q(r,v) = \frac{w}{s}
    $$ is called the studentized range. The distribution of q uses a special table.

**Notes**

-   when we are not interested in testing all pairwise comparison,s the confidence coefficient for the family of comparisons under consideration will be greater than $1-\alpha$ (with the significance level less than $\alpha$)
-   Tukey can be used for "data snooping" as long as the effects to be studied on the basis of preliminary data analysis are pairwise comparisons.

###### Scheffe

This method applies when the family of interest is the set of possible contrasts among the treatment means:

$$
L = \sum_{i=1}^a c_i \mu_i
$$

where $\sum_{i=1}^a c_i =0$

That is, the family of all possible contrasts $L$ or

$$
\begin{aligned}
&H_0: L = 0 \\
&H_a: L \neq 0
\end{aligned}
$$

The family confidence level for the Scheffe procedure is exactly $1-\alpha$ (i.e., significance level = $\alpha$) whether the sample sizes are equal or not.

For simultaneous confidence intervals,

$$
\hat{L} \pm Ss(\hat{L})
$$

where $\hat{L}=\sum c_i \bar{Y}_{i.},s^2(\hat{L}) = MSE \sum c_i^2/n_i$ and $S^2 = (a-1)f_{1-\alpha;a-1,N-a}$

The Scheffe procedure considers

$$
F = \frac{\hat{L}^2}{(a-1)s^2(\hat{L})}
$$

where we reject $H_0$ at the family significance level $\alpha$ if $F > f_{(1-\alpha;a-1,N-a)}$

**Note**

-   Since applications of the Scheffe never involve all conceivable contrasts, the **finite family** confidence coefficient will be larger than $1-\alpha$, so $1-\alpha$ is a lower bound. Thus, people often consider a larger $\alpha$ (e.g., 90% confidence interval)
-   Scheffe can be used for "data scooping" since the family of statements contains all possible contrasts.
-   If only pairwise comparisons are to be considered, The Tukey procedure gives narrower confidence limits.

###### Bonferroni

Applicable whether the sample sizes are equal or unequal.

For the confidence intervals,

$$
\hat{L} \pm B s(\hat{L})
$$

where $B= t_{(1-\alpha/(2g);N-a)}$ and g is the number of comparisons in the family.

Hypothesis testing

$$
\begin{aligned}
&H_0: L = 0 \\
&H_a: L \neq 0 
\end{aligned}
$$

Let $T= \frac{\hat{L}}{s(\hat{L})}$ and reject $H_0$ if $|T|>t_{1-\alpha/(2g),N-a}$

**Notes**

-   If all pairwise comparisons are of interest, the Tukey procedure is superior (narrower confidence intervals). If not, Bonferroni may be better.
-   Bonferroni is better than Scheffe when the number of contrasts is about the same as the treatment levels (or less).
-   Recommendation: compute all threes and pick the smallest.
-   Bonferroni can't be used for **data snooping**

###### Fisher's LSD

does not control for family error rate

use $t$-stat for testing

$$
H_0: \mu_i = \mu_j
$$

t-stat

$$
t = \frac{\bar{y}_i - \bar{y}_j}{\sqrt{MSE(\frac{1}{n_i}+ \frac{1}{n_j})}}
$$

###### Newman-Keuls

Do not recommend using this test since it has less power than ANOVA.

##### Multiple comparisons with a control

###### Dunnett

We have $a$ groups where the last group is the control group, and the $a-1$ treatment groups.

Then, we compare treatment groups to the control group. Hence, we have $a-1$ contrasts (i.e., $a-1$ pairwise comparisons)

##### Summary

When choosing a multiple contrast method:

-   Pairwise

    -   Equal groups sizes: [Tukey]
    -   Unequal groups sizes: [Tukey], [Scheffe]

-   Not pairwise

    -   with control: [Dunnett]
    -   general: [Bonferroni], [Scheffe]

### Single Factor Random Effects Model

Also known as ANOVA Type II models.

Treatments are chosen at from from larger population. We extend inference to all treatments in the population and not restrict our inference to those treatments that happened to be selected for the study.

#### Random Cell Means

$$
Y_{ij} = \mu_i + \epsilon_{ij}
$$

where

-   $\mu_i \sim N(\mu, \sigma^2_{\mu})$ and independent
-   $\epsilon_{ij} \sim N(0,\sigma^2)$ and independent

$\mu_i$ and $\epsilon_{ij}$ are mutually independent for $i =1,...,a; j = 1,...,n$

With all treatment sample sizes are equal

$$
\begin{aligned}
E(Y_{ij}) &= E(\mu_i) = \mu \\
var(Y_{ij}) &= var(\mu_i) + var(\epsilon_i) = \sigma^2_{\mu} + \sigma^2
\end{aligned}
$$

Since $Y_{ij}$ are not independent

$$
\begin{aligned}
cov(Y_{ij},Y_{ij'}) &= E(Y_{ij}Y_{ij'}) - E(Y_{ij})E(Y_{ij'})  \\
&= E(\mu_i^2 + \mu_i \epsilon_{ij'} + \mu_i \epsilon_{ij} + \epsilon_{ij}\epsilon_{ij'}) - \mu^2 \\
&= \sigma^2_{\mu} + \mu^2 - \mu^2 & \text{if} j \neq j' \\
&= \sigma^2_{\mu} & \text{if} j \neq j' 
\end{aligned}
$$

$$
\begin{aligned}
cov(Y_{ij},Y_{i'j'}) &= E(\mu_i \mu_{i'} + \mu_i \epsilon_{i'j'}+ \mu_{i'}\epsilon_{ij}+ \epsilon_{ij}\epsilon_{i'j'}) - \mu^2 \\
&= \mu^2 - \mu^2 & \text{if } i \neq i' \\
&= 0 \\
\end{aligned}
$$

Hence,

-   all observations have the same variance
-   any two observations from the same treatment have covariance $\sigma^2_{\mu}$
-   The correlation between any two responses from the same treatment:\
    $$
    \begin{aligned}
    \rho(Y_{ij},Y_{ij'}) &= \frac{\sigma^2_{\mu}}{\sigma^2_{\mu}+ \sigma^2} && \text{$j \neq j'$}
    \end{aligned}
    $$

**Inference**

**Intraclass Correlation Coefficient**

$$
\frac{\sigma^2_{\mu}}{\sigma^2 + \sigma^2_{\mu}}
$$

which measures the proportion of total variability of $Y_{ij}$ accounted for by the variance of $\mu_i$

$$
\begin{aligned}
&H_0: \sigma_{\mu}^2 = 0 \\
&H_a: \sigma_{\mu}^2 \neq 0
\end{aligned}
$$

$H_0$ implies $\mu_i = \mu$ for all i, which can be tested by the F-test in ANOVA.

The understandings of the [Single Factor Fixed Effects Model] and the [Single Factor Random Effects Model] are different, the ANOVA is same for the one factor model. The difference is in the expected mean squares

+----------------------------------------+----------------------------------------------------------------+
| **Random Effects** Model               | **Fixed Effects** Model                                        |
+========================================+================================================================+
| $E(MSE) = \sigma^2$                    | $E(MSE) = \sigma^2$                                            |
+----------------------------------------+----------------------------------------------------------------+
| $E(M STR) = \sigma^2 - n \sigma^2_\mu$ | $E(MSTR) = \sigma^2 + \frac{ \sum_i n_i (\mu_i - \mu)^2}{a-1}$ |
+----------------------------------------+----------------------------------------------------------------+

If $\sigma^2_\mu$, then MSE and MSTR have the same expectation ($\sigma^2$). Otherwise, $E(MSTR) >E(MSE)$. Large values of the statistic

$$
F = \frac{MSTR}{MSE}
$$

suggest we reject $H_0$.

Since $F \sim F_{(a-1,a(n-1))}$ when $H_0$ holds. If $F > f_{(1-\alpha;a-1,a(n-1))}$ we reject $H_0$.

If sample sizes are not equal, $F$-test can still be used, but the df are $a-1$ and $N-a$.

##### Estimation of $\mu$

An unbiased estimator of $E(Y_{ij})=\mu$ is the grand mean: $\hat{\mu} = \hat{Y}_{..}$

The variance of this estimator is

$$
\begin{aligned}
var(\bar{Y}_{..}) &= var(\sum_i \bar{Y}_{i.}/a) \\
&= \frac{1}{a^2}\sum_ivar(\bar{Y}_{i.}) \\
&= \frac{1}{a^2}\sum_i(\sigma^2_\mu+\sigma^2/n) \\
&= \frac{1}{a^2}(\sigma^2_{\mu}+\sigma^2/n) \\
&= \frac{n\sigma^2_{\mu}+ \sigma^2}{an}
\end{aligned}
$$

An unbiased estimator of this variance is $s^2(\bar{Y})=\frac{MSTR}{an}$. Thus $\frac{\bar{Y}_{..}-\mu}{s(\bar{Y}_{..})} \sim t_{a-1}$

A $1-\alpha$ confidence interval is $\bar{Y}_{..} \pm t_{(1-\alpha/2;a-1)}s(\bar{Y}_{..})$

##### Estimation of $\sigma^2_\mu/(\sigma^2_{\mu}+\sigma^2)$

In the random and fixed effects model, MSTR and MSE are independent. When the sample sizes are equal ($n_i = n$ for all i),

$$
\frac{\frac{MSTR}{n\sigma^2_\mu+ \sigma^2}}{\frac{MSE}{\sigma^2}} \sim f_{(a-1,a(n-1))}
$$

$$
P(f_{(\alpha/2;a-1,a(n-1))}\le \frac{\frac{MSTR}{n\sigma^2_\mu+ \sigma^2}}{\frac{MSE}{\sigma^2}} \le f_{(1-\alpha/2;a-1,a(n-1))}) = 1-\alpha
$$

$$
\begin{aligned}
L &= \frac{1}{n}(\frac{MSTR}{MSE}(\frac{1}{f_{(1-\alpha/2;a-1,a(n-1))}})-1) \\
U &= \frac{1}{n}(\frac{MSTR}{MSE}(\frac{1}{f_{(\alpha/2;a-1,a(n-1))}})-1)
\end{aligned}
$$

The lower and upper $(L^*,U^*)$ confidence limits for $\frac{\sigma^2_\mu}{\sigma^2_\mu + \sigma^2}$

$$
\begin{aligned}
L^* &= \frac{L}{1+L} \\
U^* &= \frac{U}{1+U}
\end{aligned}
$$

If the lower limit for $\frac{\sigma^2_\mu}{\sigma^2}$ is negative, it is customary to set $L = 0$.

##### Estimation of $\sigma^2$

$a(n-1)MSE/\sigma^2 \sim \chi^2_{a(n-1)}$, the $(1-\alpha)$ confidence interval for $\sigma^2$:

$$
\frac{a(n-1)MSE}{\chi^2_{1-\alpha/2;a(n-1)}} \le \sigma^2 \le \frac{a(n-1)MSE}{\chi^2_{\alpha/2;a(n-1)}}
$$

can also be used in case sample sizes are not equal - then df is N-a.

##### Estimation of $\sigma^2_\mu$

$E(MSE) = \sigma^2$ $E(MSTR) = \sigma^2 + n\sigma^2_\mu$. Hence,

$$
\sigma^2_{\mu} = \frac{E(MSTR)- E(MSE)}{n}
$$

An unbiased estimator of $\sigma^2_\mu$ is given by

$$
s^2_\mu =\frac{MSTR-MSE}{n}
$$

if $s^2_\mu < 0$, set $s^2_\mu = 0$

If sample sizes are not equal,

$$
s^2_\mu = \frac{MSTR - MSE}{n'}
$$

where $n' = \frac{1}{a-1}(\sum_i n_i- \frac{\sum_i n^2_i}{\sum_i n_i})$

no exact confidence intervals for $\sigma^2_\mu$, but we can approximate intervals.

**Satterthewaite Procedure** can be used to construct approximate confidence intervals for linear combination of expected mean squares\
A linear combination:

$$
\sigma^2_\mu = \frac{1}{n} E(MSTR) + (-\frac{1}{n}) E(MSE)
$$

$$
S = d_1 E(MS_1) + ..+ d_h E(MS_h)
$$

where $d_i$ are coefficients.

An unbiased estimator of S is

$$
\hat{S} = d_1 MS_1 + ...+ d_h  MS_h 
$$

Let $df_i$ be the degrees of freedom associated with the mean square $MS_i$. The **Satterthwaite** approximation:

$$
\frac{(df)\hat{S}}{S} \sim \chi^2_{df}
$$

where

$$
df = \frac{(d_1MS_1+...+d_hMS_h)^2}{(d_1MS_1)^2/df_1 + ...+ (d_hMS_h)^2/df_h}
$$

An approximate $1-\alpha$ confidence interval for S:

$$
\frac{(df)\hat{S}}{\chi^2_{1-\alpha/2;df}} \le S \le \frac{(df)\hat{S}}{\chi^2_{\alpha/2;df}}
$$

For the single factor random effects model

$$
\frac{(df)s^2_\mu}{\chi^2_{1-\alpha/2;df}} \le \sigma^2_\mu \le \frac{(df)s^2_\mu}{\chi^2_{\alpha/2;df}}
$$

where

$$
df = \frac{(sn^2_\mu)^2}{\frac{(MSTR)^2}{a-1}+ \frac{(MSE)^2}{a(n-1)}}
$$

#### Random Treatment Effects Model

$$
\tau_i = \mu_i - E(\mu_i) = \mu_i - \mu
$$

we have $\mu_i = \mu + \tau_i$ and

$$
Y_{ij} = \mu + \tau_i + \epsilon_{ij}
$$

where

-   $\mu$ = constant, common to all observations
-   $\tau_i \sim N(0,\sigma^2_\tau)$ independent (random variables)
-   $\epsilon_{ij} \sim N(0,\sigma^2)$ independent.
-   $\tau_{i}, \epsilon_{ij}$ are independent (i=1,...,a; j =1,..,n)
-   our model is concerned with only balanced single factor ANOVA.

**Diagnostics Measures**

-   Non-constant error variance (plots, Levene test, Hartley test).
-   Non-independence of errors (plots, Durban-Watson test).
-   Outliers (plots, regression methods).
-   Non-normality of error terms (plots, Shapiro-Wilk, Anderson-Darling).
-   Omitted Variable Bias (plots)

**Remedial**

-   [Weighted Least Squares]
-   [Transformations]
-   Non-parametric Procedures.

**Note**

-   Fixed effect ANOVA is relatively robust to

    -   non-normality
    -   unequal variances when sample sizes are approximately equal; at least the F-test and multiple comparisons. However, single comparisons of treatment means are sensitive to unequal variances.

-   Lack of independence can seriously affect both fixed and random effect ANVOA.

### Two Factor Fixed Effect ANOVA

The multi-factor experiment is

-   more efficient
-   provides more info
-   gives more validity to the findings.

#### Balanced

Assumption:

-   All treatment sample sizes are equal
-   All treatment means are of equal importance

Assume:

-   Factor $A$ has `a` levels and Factor $B$ has `b` levels. All $a \times b$ factor levels are considered.
-   The number of treatments for each level is n. $N = abn$ observations in the study.

##### Cell Means Model {#cell-means-model-1}

$$
Y_{ijk} = \mu_{ij} + \epsilon_{ijk}
$$

where

-   $\mu_{ij}$ are fixed parameters (cell means)
-   $i = 1,...,a$ = the levels of Factor A
-   $j = 1,...,b$ = the levels of Factor B.
-   $\epsilon_{ijk} \sim \text{indep } N(0,\sigma^2)$ for $i = 1,...,a$, $j = 1,..,b$ and $k = 1,..,n$

And

$$
\begin{aligned}
E(Y_{ijk}) &= \mu_{ij} \\
var(Y_{ijk}) &= var(\epsilon_{ijk}) = \sigma^2
\end{aligned}
$$

Hence,

$$
Y_{ijk} \sim \text{indep } N(\mu_{ij},\sigma^2)
$$

And the model is\

$$
\mathbf{Y} = \mathbf{X} \beta + \epsilon
$$

Thus,

$$
\begin{aligned}
E(\mathbf{Y}) &= \mathbf{X}\beta \\
var(\mathbf{Y}) &= \sigma^2 \mathbf{I}
\end{aligned}
$$

**Interaction**

$$
(\alpha \beta)_{ij} = \mu_{ij} - (\mu_{..}+ \alpha_i + \beta_j)
$$

where

-   $\mu_{..} = \sum_i \sum_j \mu_{ij}/ab$ is the grand mean
-   $\alpha_i = \mu_{i.}-\mu_{..}$ is the main effect for factor $A$ at the $i$-th level
-   $\beta_j = \mu_{.j} - \mu_{..}$ is the main effect for factor $B$ at the $j$-th level
-   $(\alpha \beta)_{ij}$ is the interaction effect when factor $A$ is at the $i$-th level and factor $B$ is at the $j$-th level.
-   $(\alpha \beta)_{ij} = \mu_{ij} - \mu_{i.}-\mu_{.j}+ \mu_{..}$

Examine interactions:

-   Examine whether all $\mu_{ij}$ can be expressed as the sums $\mu_{..} + \alpha_i + \beta_j$
-   Examine whether the difference between the mean responses for any two levels of factor $B$ is the same for all levels of factor $A$.
-   Examine whether the difference between the mean response for any two levels of factor $A$ is the same for all levels of factor $B$
-   Examine whether the treatment mean curves for the different factor levels in a treatment plot are parallel.

For $j = 1,...,b$

$$
\begin{aligned}
\sum_i(\alpha \beta)_{ij} &= \sum_i (\mu_{ij} - \mu_{..} - \alpha_i - \beta_j) \\
&= \sum_i \mu_{ij} - a \mu_{..} - \sum_i \alpha_i - a \beta_j \\
&= a \mu_{.j} - a \mu_{..}- \sum_i (\mu_{i.} - \mu_{..}) - a(\mu_{.j}-\mu_{..}) \\
&= a \mu_{.j} - a \mu_{..} - a \mu_{..}+ a \mu_{..} - a (\mu_{.j} - \mu_{..}) \\
&= 0
\end{aligned}
$$

Similarly, $\sum_j (\alpha \beta) = 0, i = 1,...,a$ and $\sum_i \sum_j (\alpha \beta)_{ij} =0$, $\sum_i \alpha_i = 0$, $\sum_j \beta_j = 0$

##### Factor Effects Model

$$
\begin{aligned}
\mu_{ij} &= \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} \\
Y_{ijk} &= \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ijk}
\end{aligned}
$$

where

-   $\mu_{..}$ is a constant
-   $\alpha_i$ are constants subject to the restriction $\sum_i \alpha_i=0$
-   $\beta_j$ are constants subject to the restriction $\sum_j \beta_j = 0$
-   $(\alpha \beta)_{ij}$ are constants subject to the restriction $\sum_i(\alpha \beta)_{ij} = 0$ for $j=1,...,b$ and $\sum_j(\alpha \beta)_{ij} = 0$ for $i = 1,...,a$
-   $\epsilon_{ijk} \sim \text{indep } N(0,\sigma^2)$ for $k = 1,..,n$

We have

$$
\begin{aligned}
E(Y_{ijk}) &= \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij}\\
var(Y_{ijk}) &= \sigma^2 \\
Y_{ijk} &\sim N (\mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij}, \sigma^2)
\end{aligned}
$$

We have $1+a+b+ab$ parameters. But there are $ab$ parameters in the [Cell Means Model](#cell-means-model-1). In the [Factor Effects Model], the restrictions limit the number of parameters that can be estimated:

$$
\begin{aligned}
1 &\text{ for } \mu_{..} \\
(a-1) &\text{ for } \alpha_i \\
(b-1) &\text{ for } \beta_j \\
(a-1)(b-1) &\text{ for } (\alpha \beta)_{ij}
\end{aligned}
$$

Hence, there are

$$
1 + a - 1 + b - 1 + ab - a- b + 1 = ab
$$

parameters in the model.

We can have several restrictions when considering the model in the form $\mathbf{Y} = \mathbf{X} \beta + \epsilon$

One way:

$$
\begin{aligned}
\alpha_a  &= \alpha_1 - \alpha_2 - ... - \alpha_{a-1} \\
\beta_b &= -\beta_1 - \beta_2 - ... - \beta_{b-1} \\
(\alpha \beta)_{ib} &= -(\alpha \beta)_{i1} -(\alpha \beta)_{i2} -...-(\alpha \beta)_{i,b-1} ; i = 1,..,a \\
(\alpha \beta)_{aj}& = -(\alpha \beta)_{1j}-(\alpha \beta)_{2j} - ... -(\alpha \beta)_{a-1,j}; j = 1,..,b
\end{aligned}
$$

We can fit the model by least squares or maximum likelihood

**Cell Means Model**\
minimize\

$$
Q = \sum_i \sum_j \sum_k (Y_{ijk}-\mu_{ij})^2
$$

estimators

$$
\begin{aligned}
\hat{\mu}_{ij} &= \bar{Y}_{ij} \\
\hat{Y}_{ijk} &= \bar{Y}_{ij} \\
e_{ijk} = Y_{ijk} - \hat{Y}_{ijk} &= Y_{ijk} - \bar{Y}_{ij}
\end{aligned}
$$

**Factor Effects Model**

$$
Q = \sum_i \sum_j \sum_k (Y_{ijk} - \mu_{..}-\alpha_i = \beta_j - (\alpha \beta)_{ij})^2
$$

subject to the restrictions

$$
\begin{aligned}
\sum_i \alpha_i &= 0 \\
\sum_j \beta_j &= 0 \\
\sum_i (\alpha \beta)_{ij} &= 0 \\
\sum_j (\alpha \beta)_{ij} &= 0
\end{aligned}
$$

estimators

$$
\begin{aligned}
\hat{\mu}_{..} &= \bar{Y}_{...} \\
\hat{\alpha}_i &= \bar{Y}_{i..} - \bar{Y}_{...} \\
\hat{\beta}_j &= \bar{Y}_{.j.}-\bar{Y}_{...} \\
(\hat{\alpha \beta})_{ij} &= \bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.}+ \bar{Y}_{...}
\end{aligned}
$$

The fitted values

$$
\hat{Y}_{ijk} = \bar{Y}_{...}+ (\bar{Y}_{i..}- \bar{Y}_{...})+ (\bar{Y}_{.j.}- \bar{Y}_{...}) + (\bar{Y}_{ij.} - \bar{Y}_{i..}-\bar{Y}_{.j.}+\bar{Y}_{...}) = \bar{Y}_{ij.}
$$

where

$$
\begin{aligned}
e_{ijk} &= Y_{ijk} - \bar{Y}_{ij.} \\
e_{ijk} &\sim \text{ indep } (0,\sigma^2)
\end{aligned}
$$

and

$$
\begin{aligned}
s^2_{\hat{\mu}..} &= \frac{MSE}{nab} \\
s^2_{\hat{\alpha}_i} &= MSE(\frac{1}{nb} - \frac{1}{nab}) \\
s^2_{\hat{\beta}_j} &= MSE(\frac{1}{na} - \frac{1}{nab}) \\
s^2_{(\hat{\alpha\beta})_{ij}} &= MSE (\frac{1}{n} - \frac{1}{na}- \frac{1}{nb} + \frac{1}{nab})
\end{aligned}
$$

###### Partitioning the Total Sum of Squares

$$
Y_{ijk} - \bar{Y}_{...} = \bar{Y}_{ij.} - \bar{Y}_{...} + Y_{ijk} - \bar{Y}_{ij.}
$$

$Y_{ijk} - \bar{Y}_{...}$: Total deviation\
$\bar{Y}_{ij.} - \bar{Y}_{...}$: Deviation of treatment mean from overall mean\
$Y_{ijk} - \bar{Y}_{ij.}$: Deviation of observation around treatment mean (residual).

$$
\begin{aligned}
\sum_i \sum_j \sum_k (Y_{ijk} - \bar{Y}_{...})^2 &= n \sum_i \sum_j (\bar{Y}_{ij.}- \bar{Y}_{...})^2+ \sum_i \sum_j sum_k (Y_{ijk} - \bar{ij.})^2 \\
SSTO &= SSTR + SSE
\end{aligned}
$$

(cross product terms are 0)

$$
\bar{Y}_{ij.}- \bar{Y}_{...} = \bar{Y}_{i..}-\bar{Y}_{...} + \bar{Y}_{.j.}-\bar{Y}_{...} + \bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...}
$$

squaring and summing:\

$$
\begin{aligned}
n\sum_i \sum_j (\bar{Y}_{ij.}-\bar{Y}_{...})^2 &= nb\sum_i (\bar{Y}_{i..}-\bar{Y}_{...})^2 + na \sum_j (\bar{Y}_{.j.}-\bar{Y}_{...})^2 \\
&+ n \sum_i \sum_j (\bar{Y}_{ij.}-\bar{Y}_{i..}- \bar{Y}_{.j.}+ \bar{Y}_{...})^2 \\
SSTR &= SSA + SSB + SSAB
\end{aligned}
$$

The interaction term from

$$
\begin{aligned}
SSAB &= SSTO - SSE - SSA - SSB \\
SSAB &= SSTR - SSA - SSB 
\end{aligned}
$$

where

-   $SSA$ is the factor $A$ sum of squares (measures the variability of the estimated factor $A$ level means $\bar{Y}_{i..}$)- the more variable, the larger $SSA$
-   $SSB$ is the factor $B$ sum of squares
-   $SSAB$ is the interaction sum of squares, measuring the variability of the estimated interactions.

###### Partitioning the df

$N = abn$ cases and $ab$ treatments.

For one-way ANOVA and regression, the partition has df:

$$
SS: SSTO = SSTR + SSE
$$

$$
df: N-1 = (ab-1) + (N-ab) 
$$

we must further partition the $ab-1$ df with SSTR

$$
SSTR = SSA + SSB + SSAB
$$

$$
ab-1 = (a-1) + (b-1) + (a-1)(b-1) 
$$

-   $df_{SSA} = a-1$: a treatment deviations but 1 df is lost due to the restriction $\sum (\bar{Y}_{i..}- \bar{Y}_{...})=0$\
-   $df_{SSB} = b-1$: b treatment deviations but 1 df is lost due to the restriction $\sum (\bar{Y}_{.j.}- \bar{Y}_{...})=0$\
-   $df_{SSAB} = (a-1)(b-1)= (ab-1)-(a-1)-(b-1)$: ab interactions, there are (a+b-1) restrictions, so df = ab-a-(b-1)= (a-1)(b-1)

###### Mean Squares

$$
\begin{aligned}
MSA &= \frac{SSA}{a-1}\\
MSB &= \frac{SSB}{b-1}\\
MSAB &= \frac{SSAB}{(a-1)(b-1)}
\end{aligned}
$$

The expected mean squares are

$$
\begin{aligned}
E(MSE) &= \sigma^2 \\
E(MSA) &= \sigma^2 + nb \frac{\sum \alpha_i^2}{a-1} = \sigma^2 + nb \frac{\sum(\sum_{i.}-\mu_{..})^2}{a-1}  \\
E(MSB) &= \sigma^2 + na \frac{\sum \beta_i^2}{b-1} = \sigma^2 + na \frac{\sum(\sum_{.j}-\mu_{..})^2}{b-1} \\
E(MSAB) &= \sigma^2 + n \frac{\sum \sum (\alpha \beta)_{ij}^2}{(a-1)(b-1)} = \sigma^2 + n \frac{\sum (\mu_{ij}- \mu_{i.}- \mu_{.j}+ \mu_{..} )^2}{(a-1)(b-1)}
\end{aligned}
$$

If there are no factor A main effects (all $\mu_{i.} = 0$ or $\alpha_i = 0$) the MSA and MSE have the same expectation; otherwise MSA \> MSE. Same for factor B, and interaction effects. which case we can examine F-statistics.

**Interaction**

$$
\begin{aligned}
H_0: \mu_{ij}- \mu_{i.} - \mu_{.j} + \mu_{..} = 0 && \text{for all i,j} \\
H_a: \mu_{ij}- \mu_{i.} - \mu_{.j} + \mu_{..} \neq 0 && \text{for some i,j}
\end{aligned}
$$

or

$$
\begin{aligned}
&H_0: \text{All}(\alpha \beta)_{ij} = 0 \\
&H_a: \text{Not all} (\alpha \beta) = 0
\end{aligned}
$$

Let $F = \frac{MSAB}{MSE}$. When $H_0$ is true $F \sim f_{((a-1)(b-1),ab(n-1))}$. So reject $H_0$ when $F > f_{((a-1)(b-1),ab(n-1))}$

Factor A main effects:\

$$
\begin{aligned}
&H_0: \mu_{1.} = \mu_{2.} = ... = \mu_{a.} \\
&H_a: \text{Not all $\mu_{i.}$ are equal}
\end{aligned}
$$

or

$$
\begin{aligned}
&H_0: \alpha_1 = ... = \alpha_a = 0 \\
&H_a: \text{Not all $\alpha_i$ are equal to 0}
\end{aligned}
$$

$F= \frac{MSA}{MSE}$ and reject $H_0$ if $F>f_{(1-\alpha;a-1,ab(n-1))}$

###### Two-way ANOVA

+---------------------+------------+--------------+---------------------+------------+
| Source of Variation | SS         | df           | MS                  | F          |
+=====================+============+==============+=====================+============+
| Factor A            | $SSA$      | $a-1$        | $MSA = SSA/(a-1)$   | $MSA/MSE$  |
+---------------------+------------+--------------+---------------------+------------+
| Factor B            | $SSB$      | $b-1$        | $MSB = SSB/(b-1)$   | $MSB/MSE$  |
+---------------------+------------+--------------+---------------------+------------+
| AB interactions     | $SSAB$     | $(a-1)(b-1)$ | $MSAB = SSAB /MSE$  |            |
+---------------------+------------+--------------+---------------------+------------+
| Error               | $SSE$      | $ab(n-1)$    | $MSE = SSE/ab(n-1)$ |            |
+---------------------+------------+--------------+---------------------+------------+
| Total (corrected)   | $SSTO$     | $abn - 1$    |                     |            |
+---------------------+------------+--------------+---------------------+------------+

Doing 2-way ANOVA means you always check interaction first, because if there are significant interactions, checking the significance of the main effects becomes moot.

The main effects concern the mean responses for levels of one factor averaged over the levels of the other factor. When interaction is present, we can't conclude that a given factor has no effect, even if these averages are the same. It means that the effect of the factor depends on the level of the other factor.

On the other hand, if you can establish that there is no interaction, then you can consider inference on the factor main effects, which are then said to be **additive**.\
And we can also compare factor means like the [Single Factor Fixed Effects Model] using [Tukey], [Scheffe], [Bonferroni].

We can also consider contrasts in the 2-way model

$$
L = \sum c_i \mu_i
$$

where $\sum c_i =0$\
which is estimated by

$$
\hat{L} = \sum c_i \bar{Y}_{i..}
$$

with variance

$$
\sigma^2(\hat{L}) = \frac{\sigma^2}{bn} \sum c_i^2
$$

and variance estimate

$$
\frac{MSE}{bn} \sum c_i^2
$$

**Orthogonal Contrasts**

$$
\begin{aligned}
L_1 &= \sum c_i \mu_i, \sum c_i = 0 \\
L_2 &= \sum d_i \mu_i , \sum d_i = 0
\end{aligned}
$$

these contrasts are said to be **orthogonal** if

$$
\sum \frac{c_i d_i}{n_i} = 0
$$

in balanced case $\sum c_i d_i =0$

$$
\begin{aligned}
cov(\hat{L}_1, \hat{L}_2) &= cov(\sum_i c_i \bar{Y}_{i..}, \sum_l d_l \bar{Y}_{l..}) \\
&= \sum_i \sum_l c_i d_l cov(\bar{Y}_{i..},\bar{Y}_{l..}) \\
&= \sum_i c_i d_i \frac{\sigma^2}{bn} = 0
\end{aligned}
$$

Orthogonal contrasts can be used to further partition the model sum of squares. There are many sets of orthogonal contrasts and thus, many ways to partition the sum of squares.

A special set of orthogonal contrasts that are used when the levels of a factor can be assigned values on a metric scale are called **orthogonal polynomials**

Coefficients can be found for the special case of

-   equal spaced levels (e.g., (0 15 30 45 60))\
-   equal sample sizes ($n_1 = n_2 = ... = n_{ab}$)

We can define the SS for a given contrast:

$$
SS_L = \frac{\hat{L}^2}{\sum_{i=1}^a (c^2_i/bn_i)}
$$

$$
T = \frac{\hat{L}}{\sqrt{MSE\sum_{i=1}^a(c_i^2/bn_i)}} \sim t
$$

Moreover,

$$
t^2_{(1-\alpha/2;df)}=F_{(1-\alpha;1,df)}
$$

So,

$$
\frac{SS_L}{MSE} \sim F_{(1-\alpha;1,df_{MSE})}
$$

all contrasts have d.f = 1

#### Unbalanced

We could have unequal numbers of replications for all treatment combinations:

-   Observational studies
-   Dropouts in designed studies
-   Larger sample sizes for inexpensive treatments
-   Sample sizes to match population makeup.

Assume that each factor combination has at least 1 observation (no empty cells)

Consider the same model as:

$$
Y_{ijk} = \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ijk}
$$

where sample sizes are: $n_{ij}$:

$$
\begin{aligned}
n_{i.} &= \sum_j n_{ij} \\
n_{.j} &= \sum_i n_{ij} \\
n_T &= \sum_i \sum_j n_{ij}
\end{aligned}
$$

Problem here is that

$$
SSTO \neq SSA + SSB + SSAB + SSE
$$

(the design is **non-orthogonal**)

-   For $i = 1,...,a-1,$

$$
u_i = \begin{cases} +1 & \text{if the obs is from the i-th level of Factor 1} \\ -1 & \text{if the obs is from the a-th level of Factor 1} \\ 0 & \text{otherwise} \\ \end{cases}
$$

-   For $j=1,...,b-1$

$$
v_i = 
\begin{cases} +1 & \text{if the obs is from the j-th level of Factor 1} \\ -1 & \text{if the obs is from the b-th level of Factor 1} \\ 0 & \text{otherwise} \\ 
\end{cases}
$$

We can use these indicator variables as predictor variables and $\mu_{..}, \alpha_i ,\beta_j, (\alpha \beta)_{ij}$ as unknown parameters.

$$
Y = \mu_{..} + \sum_{i=1}^{a-1} \alpha_i u_i + \sum_{j=1}^{b-1} \beta_j v_j + \sum_{i=1}^{a-1} \sum_{j=1}^{b-1}(\alpha \beta)_{ij} u_i v_j + \epsilon
$$

To test hypotheses, we use the extra sum of squares idea.

For interaction effects

$$
\begin{aligned}
&H_0: all (\alpha \beta)_{ij} = 0 \\
&H_a: \text{not all }(\alpha \beta)_{ij} =0
\end{aligned}
$$

Or to test

$$
\begin{aligned}
&H_0: \beta_1 = \beta_2 = \beta_3 = 0 \\
&H_a: \text{not all } \beta_j = 0
\end{aligned}
$$

**Analysis of Factor Means**

(e.g., contrasts) is analogous to the balanced case, with modifications in the formulas for means and standard errors to account for unequal sample sizes.

Or , we can fit the cell means model and consider it from a regression perspective

If you have empty cells (i.e., some factor combinations have no observation), then the equivalent regression approach can't be used. But you can still do partial analyses

### Two-Way Random Effects ANOVA

$$
Y_{ijk} = \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ij}
$$

where

-   $\mu_{..}$: constant
-   $\alpha_i \sim N(0,\sigma^2_{\alpha}), i = 1,..,a$ (independent)
-   $\beta_j \sim N(0,\sigma^2_{\beta}), j = 1,..,b$ (independent)
-   $(\alpha \beta)_{ij} \sim N(0,\sigma^2_{\alpha \beta}),i=1,...,a,j=1,..,b$ (independent)
-   $\epsilon_{ijk} \sim N(0,\sigma^2)$ (independent)

All $\alpha_i, \beta_j, (\alpha \beta)_{ij}$ are pairwise independent

Theoretical means, variances, and covariances are

$$
\begin{aligned}
E(Y_{ijk}) &= \mu_{..} \\
var(Y_{ijk}) &= \sigma^2_Y= \sigma^2_\alpha + \sigma^2_\beta +  \sigma^2_{\alpha \beta} + \sigma^2 
\end{aligned}
$$

So

$Y_{ijk} \sim N(\mu_{..},\sigma^2_\alpha + \sigma^2_\beta + \sigma^2_{\alpha \beta} + \sigma^2)$

$$
\begin{aligned}
cov(Y_{ijk},Y_{ij'k'}) &= \sigma^2_{\alpha}, j \neq j' \\
cov(Y_{ijk},Y_{i'jk'}) &= \sigma^2_{\beta}, i \neq i'\\
cov(Y_{ijk},Y_{ijk'}) &= \sigma^2_\alpha + \sigma^2_{\beta} + \sigma^2_{\alpha \beta}, k \neq k' \\
cov(Y_{ijk},Y_{i'j'k'}) &= , i \neq i', j \neq j'
\end{aligned}
$$

### Two-Way Mixed Effects ANOVA

#### Balanced

One fixed factor, while other is random treatment levels, we have a **mixed effects model** or a **mixed model**

**Restricted mixed model** for 2-way ANOVA:

$$
Y_{ijk} = \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ijk}
$$

where

-   $\mu_{..}$: constant
-   $\alpha_i$: fixed effects with constraints subject to restriction $\sum \alpha_i = 0$
-   $\beta_j \sim indep N(0,\sigma^2_\beta)$
-   $(\alpha \beta)_{ij} \sim N(0,\frac{a-1}{a}\sigma^2_{\alpha \beta})$ subject to restriction $\sum_i (\alpha \beta)_{ij} = 0$ for all j, the variance here is written as the proportion for convenience; it makes the expected mean squares simpler (other assumed $var((\alpha \beta)_{ij}= \sigma^2_{\alpha \beta}$)
-   $cov((\alpha \beta)_{ij},(\alpha \beta)_{i'j'}) = - \frac{1}{a} \sigma^2_{\alpha \beta}, i \neq i'$
-   $\epsilon_{ijk}\sim indepN(0,\sigma^2)$
-   $\beta_j, (\alpha \beta)_{ij}, \epsilon_{ijk}$ are pairwise independent

Two-way mixed models are written in an "unrestricted" form, with no restrictions on the interaction effects $(\alpha \beta)_{ij}$, they are pairwise independent.

Let $\beta^*, (\alpha \beta)^*_{ij}$ be the unrestricted random effects, and $(\bar{\alpha \beta})_{ij}^*$ the means averaged over the fixed factor for each level of random factor B.

$$
\begin{aligned}
\beta_j &= \beta_j^* + (\bar{\alpha \beta})_{ij}^* \\
(\alpha \beta)_{ij} &= (\alpha \beta)_{ij}^* - (\bar{\alpha \beta})_{ij}^*
\end{aligned}
$$

Some consider the restricted model to be more general. but here we consider the restricted form.

$$
\begin{aligned}
E(Y_{ijk}) &= \mu_{..} + \alpha_i \\
var(Y_{ijk}) &= \sigma^2_\beta + \frac{a-1}{a} \sigma^2_{\alpha \beta} + \sigma^2
\end{aligned}
$$

Responses from the same random factor $(B)$ level are correlated

$$
\begin{aligned}
cov(Y_{ijk},Y_{ijk'}) &= E(Y_{ijk}Y_{ijk'}) - E(Y_{ijk})E(Y_{ijk'}) \\
&= \sigma^2_\beta + \frac{a-1}{a} \sigma^2_{\alpha \beta} , k \neq k'
\end{aligned}
$$

Similarly,

$$
\begin{aligned}
cov(Y_{ijk},Y_{i'jk'}) &= \sigma^2_\beta - \frac{1}{a} \sigma^2_{\alpha\ \beta}, i \neq i' \\
cov(Y_{ijk},Y_{i'j'k'}) &= 0,  j \neq j'
\end{aligned}
$$

Hence, you can see that the only way you don't have dependence in the $Y$ is when they don't share the same random effect.

An advantage of the **restricted mixed model** is that 2 observations from the same random factor b level can be positively or negatively correlated. In the **unrestricted model**, they can only be positively correlated.

+-------------+--------------+-----------------------------------------------------------------------+--------------------------------------------------------------+
| Mean Square | Fixed ANOVA  | Random ANOVA                                                          | Mixed ANVOA                                                  |
|             |              |                                                                       |                                                              |
|             | (A, B Fixed) | (A,B random)                                                          | (A fixed, B random)                                          |
+=============+==============+=======================================================================+==============================================================+
| MSA         | a - 1        | $\sigma ^2+ n b \frac{\sum\alpha_i^2}{a-1}$                           | $\sigma^2 + nb\sigma^ 2_ \alpha +n \sigma^ 2_{\alpha \beta}$ |
+-------------+--------------+-----------------------------------------------------------------------+--------------------------------------------------------------+
| MSB         | b-1          | $\sigma^2 + n a \frac{\sum\beta ^2_j}{b-1}$                           | $\sigma^ 2 + na\sigma^2_ \beta +n \sigma^ 2_{\alpha \beta}$  |
+-------------+--------------+-----------------------------------------------------------------------+--------------------------------------------------------------+
| MSAB        | ( a-1)(b-1)  | $\sigma^2 + n \frac{\sum \sum(\alpha \beta )^2_ {ij}} { ( a-1)(b-1)}$ | $\sigma^2+n \sigma^2_{\alpha \beta}$                         |
+-------------+--------------+-----------------------------------------------------------------------+--------------------------------------------------------------+
| MSE         | (n-1)ab      | $\sigma^2$                                                            | $\sigma^2$                                                   |
+-------------+--------------+-----------------------------------------------------------------------+--------------------------------------------------------------+

For fixed, random, and mixed models (balanced), the ANOVA table sums of squares calculations are identical. (also true for df and mean squares). The only difference is with the expected mean squares, thus the test statistics.

In Random ANOVA, we test

$$
\begin{aligned}
&H_0: \sigma^2 = 0 \\
&H_a: \sigma^2 > 0 
\end{aligned}
$$

by considering $F= \frac{MSA}{MSAB} \sim F_{a-1;(a-1)(b-1)}$

The same test statistic is used for mixed models, but in that case we are testing null hypothesis that all of the $\alpha_i = 0$

The test statistic different for the same null hypothesis under the fixed effects model.

+---------------------+--------------------+--------------------+---------------------+
| Test for effects of | Fixed ANOVA        | Random ANOVA       | Mixed ANOVA         |
|                     |                    |                    |                     |
|                     | (A&B fixed)        | (A&B random)       | (A fixed, B random) |
+=====================+====================+====================+=====================+
| Factor A            | $\frac{MSA}{MSE}$  | $\frac{MSA}{MSAB}$ | $\frac{MSA}{MSAB}$  |
+---------------------+--------------------+--------------------+---------------------+
| Factor B            | $\frac{MSB}{MSE}$  | $\frac{MSB}{MSAB}$ | $\frac{MSB}{MSE}$   |
+---------------------+--------------------+--------------------+---------------------+
| AB interactions     | $\frac{MSAB}{MSE}$ | $\frac{MSAB}{MSE}$ | $\frac{MSAB}{MSE}$  |
+---------------------+--------------------+--------------------+---------------------+

**Estimation Of Variance Components**

In random and mixed effects models, we are interested in estimating the **variance components**\
Variance component $\sigma^2_\beta$ in the mixed ANOVA.

$$
E(\sigma^2_\beta) = \frac{E(MSB)-E(MSE)}{na} = \frac{\sigma^2 + na \sigma^2_\beta - \sigma^2}{na} = \sigma^2_\beta
$$

which can be estimated with

$$
\hat{\sigma}^2_\beta = \frac{MSB - MSE}{na}
$$

Confidence intervals for variance components can be constructed (approximately) by using the **Satterthwaite** procedure or the MLS procedure (like the 1-way random effects)

**Estimation of Fixed Effects in Mixed Models**

$$
\begin{aligned}
\hat{\alpha}_i &= \bar{Y}_{i..} - \bar{Y}_{...} \\
\hat{\mu}_{i.} &= \bar{Y}_{...} + (\bar{Y}_{i..}- \bar{Y}_{...}) = \bar{Y}_{i..}  \\
\sigma^2(\hat{\alpha}_i) &= \frac{\sigma^2 + n \sigma^2_{\alpha \beta}}{bn} = \frac{E(MSAB)}{bn} \\
s^2(\hat{\alpha}_i) &= \frac{MSAB}{bn}
\end{aligned}
$$

Contrasts on the **Fixed Effects**

$$
\begin{aligned}
L &= \sum c_i \alpha_i \\
\sum c_i &= 0 \\
\hat{L} &= \sum c_i \hat{\alpha}_i \\
\sigma^2(\hat{L}) &= \sum c^2_i \sigma^2 (\hat{\alpha}_i) \\
s^2(\hat{L}) &= \frac{MSAB}{bn} \sum c^2_i
\end{aligned}
$$

Confidence intervals and tests can be constructed as usual

#### Unbalanced

For a mixed model with a = 2, b = 4

$$
\begin{aligned}
Y_{ijk} &= \mu_{..} + \alpha_i + \beta_j + (\alpha \beta)_{ij} + \epsilon_{ijk} \\
var(\beta_j)&= \sigma^2_\beta \\
var((\alpha \beta)_{ij})&= \frac{2-1}{2}\sigma^2_{\alpha \beta} = \frac{\sigma^2_{\alpha \beta}}{2} \\
var(\epsilon_{ijk}) &= \sigma^2 \\
E(Y_{ijk}) &= \mu_{..} + \alpha_i \\
var(Y_{ijk}) &= \sigma^2_{\beta} + \frac{\sigma^2_{\alpha \beta}}{2} + \sigma^2 \\
cov(Y_{ijk},Y_{ijk'}) &= \sigma^2 + \frac{\sigma^2_{\alpha \beta}}{2}, k \neq k' \\
cov(Y_{ijk},Y_{i'jk'}) &= \sigma^2_{\beta} - \frac{\sigma^2_{\alpha \beta}}{2}, i \neq i' \\
cov(Y_{ijk},Y_{i'j'k'}) &= 0, j \neq j' 
\end{aligned}
$$

assume

$$
\mathbf{Y} \sim N(\mathbf{X}\beta, M)
$$

where $M$ is block diagonal

density function

$$
f(\mathbf{Y}) = \frac{1}{(2\pi)^{N/2}|M|^{1/2}}exp(-\frac{1}{2}\mathbf{(Y - X \beta)' M^{-1}(Y-X\beta)})
$$

if we knew the variance components, we could use GLS:

$$
\hat{\beta}_{GLS} = \mathbf{(X'M^{-1}X)^{-1}X'M^{-1}Y}
$$

but we usually don't know the variance components $\sigma^2, \sigma^2_\beta, \sigma^2_{\alpha \beta}$ that make up $M$\
Another way to get estimates is by **Maximum likelihood estimation**

we try to maximize its log

$$
\ln L = - \frac{N}{2} \ln (2\pi) - \frac{1}{2}\ln|M| - \frac{1}{2} \mathbf{(Y-X \beta)'\Sigma^{-1}(Y-X\beta)}
$$

## Nonparametric ANOVA

### Kruskal-Wallis

Generalization of independent samples Wilcoxon Rank sum test for 2 independent samples (like F-test of one-way ANOVA is a generalization to several independent samples of the two sample t-test)

Consider the one-way case:

We have

-   $a\ge2$ treatments
-   $n_i$ is the sample size for the $i$-th treatment
-   $Y_{ij}$ is the $j$-th observation from the $i$-th treatment.
-   we make **no** assumption of normality
-   We only assume that observations on the $i$-th treatment are a random sample from the continuous CDF $F_i$, i = 1,..,n, and are mutually independent.

$$
\begin{aligned}
&H_0: F_1 = F_2 = ... = F_a \\
&H_a: F_i < F_j \text{ for some } i \neq j
\end{aligned}
$$

or if distribution is from the location-scale family, $H_0: \theta_1 = \theta_2 = ... = \theta_a$)

**Procedure**

-   Rank all $N = \sum_{i=1}^a n_i$ observations in ascending order. Let $r_{ij} = rank(Y_{ij})$, note $\sum_i \sum_j r_{ij} = 1 + 2 .. + N = \frac{N(N+1)}{2}$\
-   Calculate the rank sums and averages:\
    $$
    r_{i.} = \sum_{j=1}^{n_i} r_{ij}
    $$ and $$
    \bar{r}_{i.} = \frac{r_{i.}}{n_i}, i = 1,..,a
    $$
-   Calculate the test statistic on the ranks: $$
    \chi_{KW}^2 = \frac{SSTR}{\frac{SSTO}{N-1}}
    $$ where $SSTR = \sum n_i (\bar{r}_{i.}- \bar{r}_{..})^2$ and $SSTO = \sum \sum (\bar{r}_{ij}- \bar{r}_{..})^2$
-   For large $n_i$ ($\ge 5$ observations) the Kruskal-Wallis statistic is approximated by a $\chi^2_{a-1}$ distribution when all the treatment means are equal. Hence, reject $H_0$ if $\chi^2_{KW} > \chi^2_{(1-\alpha;a-1)}$.\
-   If sample sizes are small, one can exhaustively work out all possible distinct ways of assigning N ranks to the observations from a treatments and calculate the value of the KW statistic in each case ($\frac{N!}{n_1!..n_a!}$ possible combinations). Under $H_0$ all of these assignments are equally likely.

### Friedman Test

When the responses $Y_{ij} = 1,..,n, j = 1,..,r$ in a randomized complete block design are not normally distributed (or do not have constant variance), a nonparametric test is more helpful.

A distribution-free rank-based test for comparing the treatments in this setting is the Friedman test. Let $F_{ij}$ be the CDF of random $Y_{ij}$, corresponding to the observed value $y_{ij}$

Under the null hypothesis, $F_{ij}$ are identical for all treatments j separately for each block i.

$$
\begin{aligned}
&H_0: F_{i1} = F_{i2} = ... = F_{ir}  \text{ for all i} \\
&H_a: F_{ij} < F_{ij'} \text{ for some } j \neq j' \text{ for all } i
\end{aligned}
$$

For location parameter distributions, treatment effects can be tested:

$$
\begin{aligned}
&H_0: \tau_1 = \tau_2 = ... = \tau_r \\
&H_a: \tau_j > \tau_{j'} \text{ for some } j \neq j'
\end{aligned}
$$

**Procedure**

-   Rank observations from the r treatments separately within each block (in ascending order; if ties, each tied observation is given the mean of ranks involved). Let the ranks be called $r_{ij}$\
-   Calculate the Friedman test statistic\
    $$
    \chi^2_F = \frac{SSTR}{\frac{SSTR + SSE}{n(r-1)}}
    $$ where $$
    \begin{aligned}
    SSTR &= n \sum (\bar{r}_{.j}-\bar{r}_{..})^2 \\
    SSE &= \sum \sum (r_{ij} - \bar{r}_{.j})^2 \\
    \bar{r}_{.j} &= \frac{\sum_i r_{ij}}{n}\\
    \bar{r}_{..} &= \frac{r+1}{2}
    \end{aligned}
    $$

If there is no ties, it can be rewritten as

$$
\chi^2_{F} = [\frac{12}{nr(n+1)}\sum_j r_{.j}^2] - 3n(r+1)
$$

with large number of blocks, $\chi^2_F$ is approximately $\chi^2_{r-1}$ under $H_0$. Hence, we reject $H_0$ if $\chi^2_F > \chi^2_{(1-\alpha;r-1)}$\
The exact null distribution for $\chi^2_F$ can be derived since there are r! possible ways of assigning ranks 1,2,...,r to the r observations within each block. There are n blocks and thus $(r!)^n$ possible assignments to the ranks, which are equally likely when $H_0$ is true.

## Sample Size Planning for ANOVA

### Balanced Designs

#### Single Factor Studies

##### Fixed cell means

$$
P(F>f_{(1-\alpha;a-1,N-a)}|\phi) = 1 - \beta
$$

where $\phi$ is the non-centrality **parameter** (measures how unequal the treatment means $\mu_i$ are)

$$
\phi = \frac{1}{\sigma}\sqrt{\frac{n}{a}\sum_i (\mu_i - \mu_.)^2} , (n_i \equiv n)
$$

and

$$
\mu_. = \frac{\sum \mu_i}{a}
$$

To decide on the power probabilities we use the non-central F distribution.

We could use the power table directly when effects are fixed and design is balanced by using **minimum range** of factor level means for your desired differences

$$
\Delta = \max(\mu_i) - \min(\mu_i)
$$

Hence, we need

-   $\alpha$ level
-   $\Delta$
-   $\sigma$
-   $\beta$

Notes:

-   When $\Delta/\sigma$ is small greatly affects sample size, but if $\Delta/\sigma$ is large.
-   Reducing $\alpha$ or $\beta$ increases the required sample sizes.
-   Error in estimating $\sigma$ can make a large difference.

#### Multi-factor Studies

The same noncentral $F$ tables can be used here

For two-factor fixed effect model

Test for interactions:

$$
\begin{aligned}
\phi &= \frac{1}{\sigma} \sqrt{\frac{n \sum \sum (\alpha \beta_{ij})^2}{(a-1)(b-1)+1}} = \frac{1}{\sigma} \sqrt{\frac{n \sum \sum (\mu_{ij}- \mu_{i.} - \mu_{.j} + \mu_{..})^2}{(a-1)(b-1)+1}} \\
\upsilon_1 &= (a-1)(b-1) \\
\upsilon_2 &= ab(n-1)
\end{aligned}
$$

Test for Factor $A$ main effects:

$$
\begin{aligned}
\phi &= \frac{1}{\sigma} \sqrt{\frac{nb \sum \alpha_i^2}{a}} = \frac{1}{\sigma}\sqrt{\frac{nb \sum (\mu_{i.}- \mu_{..})^2}{a}} \\
\upsilon_1 &= a-1 \\
\upsilon_2 &= ab(n-1)
\end{aligned}
$$

Test for Factor $B$ main effects:

$$
\begin{aligned}
\phi &= \frac{1}{\sigma} \sqrt{\frac{na \sum \beta_j^2}{b}} = \frac{1}{\sigma}\sqrt{\frac{na \sum (\mu_{.j}- \mu_{..})^2}{b}} \\
\upsilon_1 &= b-1 \\
\upsilon_2 &= ab(n-1)
\end{aligned}
$$

Procedure:

1.  Specify the minimum range of Factor $A$ means
2.  Obtain sample sizes with $r = a$. The resulting sample size is $bn$, from which $n$ can be obtained.
3.  Repeat the first 2 steps for Factor $B$ minimum range.
4.  Choose the greater number of sample size between $A$ and $B$.

### Randomized Block Experiments

Analogous to completely randomized designs . The power of the F-test for treatment effects for randomized block design uses the same non-centrality parameter as completely randomized design:

$$
\phi = \frac{1}{\sigma} \sqrt{\frac{n}{r} \sum (\mu_i - \mu_.)^2}
$$

However, the power level is different from the randomized block design because

-   error variance $\sigma^2$ is different
-   df(MSE) is different.

## Randomized Block Designs

To improve the precision of treatment comparisons, we can reduce variability among the experimental units. We can group experimental units into **blocks** so that each block contains relatively homogeneous units.

-   Within each block, random assignment treatments to units (separate random assignment for each block)
-   The number of units per block is a multiple of the number of factor combinations.
-   Commonly, use each treatment once in each block.

Benefits of **Blocking**

-   Reduction in variability of estimators for treatment means

    -   Improved power for t-tests and F-tests
    -   Narrower confidence intervals
    -   Smaller MSE

-   Compare treatments under different conditions (related to different blocks).

Loss from **Blocking** (little to lose)

-   If you don't do blocking well, you waste df on negligible block effects that could have been used to estimate $\sigma^2$
-   Hence, the df for $t$-tests and denominator df for $F$-tests will be reduced without reducing MSE and small loss of power for both tests.

Consider

$$
Y_{ij} = \mu_{..} + \rho_i + \tau_j + \epsilon_{ij}
$$

where

-   $i = 1, 2, \dots, n$
-   $j = 1, 2, \dots, r$
-   $\mu_{..}$: overall mean response, averaging across all blocks and treatments
-   $\rho_i$: block effect, average difference in response for i-th block ($\sum \rho_i =0$)
-   $\tau_j$ treatment effect, average across blocks ($\sum \tau_j = 0$)
-   $\epsilon_{ij} \sim iid N(0,\sigma^2)$: random experimental error.

Here, we assume that the block and treatment effects are additive. The difference in average response for any pair of treatments i the same **within** each block

$$
(\mu_{..} +  \rho_i + \tau_j) - (\mu_{..} + \rho_i + \tau_j') = \tau_j - \tau_j'
$$

for all $i=1,..,n$ blocks

$$
\begin{aligned}
\hat{\mu} &= \bar{Y}_{..} \\
\hat{\rho}_i &= \bar{Y}_{i.} - \bar{Y}_{..} \\
\hat{\tau}_j &= \bar{Y}_{.j} - \bar{Y}_{..}
\end{aligned}
$$

Hence,

$$
\begin{aligned}
\hat{Y}_{ij} &= \bar{Y}_{..} + (\bar{Y}_{i.} - \bar{Y}_{..}) + (\bar{Y}_{.j}- \bar{Y}_{..}) = \bar{Y}_{i.} + \bar{Y}_{.j} - \bar{Y}_{..} \\
e_{ij} &= Y_{ij} - \hat{Y}_{ij} = Y_{ij}- \bar{Y}_{i.} - \bar{Y}_{.j} + \bar{Y}_{..}
\end{aligned}
$$

**ANOVA table**

+---------------------+--------------------------------------------------------------------------------+--------------+------------------------------------------+------------------------------------------+
| Source of Variation | SS                                                                             | df           | Fixed Treatments                         | Random Treatments                        |
|                     |                                                                                |              |                                          |                                          |
|                     |                                                                                |              | E(MS)                                    | E(MS)                                    |
+=====================+================================================================================+==============+==========================================+==========================================+
| Blocks              | $r \sum_i(\bar{Y}_{i.}-\bar{Y}_{..})^2$                                        | $n - 1$      | $\sigma^2 +r \frac{\sum \rho^2_i}{n-1}$  | $\sigma^2 + r \frac{\sum \rho^2_i}{n-1}$ |
+---------------------+--------------------------------------------------------------------------------+--------------+------------------------------------------+------------------------------------------+
| Treatments          | $n\sum_ j (\bar{Y} _ {.j}-\bar{ Y}_{..})^2$                                    | $r - 1$      | $\sigma^2 + n \frac{\sum \tau^2_j}{r-1}$ | $\sigma^2 + n \sigma^2_\tau$             |
+---------------------+--------------------------------------------------------------------------------+--------------+------------------------------------------+------------------------------------------+
| Error               | $\sum_i \sum _j ( Y_{ ij } - \bar { Y}_{i.} - \bar{Y}_{.j} + \bar{ Y}_{..})^2$ | $(n-1)(r-1)$ | $\sigma^2$                               | $\sigma^2$                               |
+---------------------+--------------------------------------------------------------------------------+--------------+------------------------------------------+------------------------------------------+
| Total               | $SSTO$                                                                         | $nr-1$       |                                          |                                          |
+---------------------+--------------------------------------------------------------------------------+--------------+------------------------------------------+------------------------------------------+

**F-tests**

$$
\begin{aligned}
H_0: \tau_1 = \tau_2 = ... = \tau_r = 0 && \text{Fixed Treatment Effects} \\
H_a: \text{not all } \tau_j = 0 \\
\\
H_0: \sigma^2_{\tau} = 0 && \text{Random Treatment Effects} \\
H_a: \sigma^2_{\tau} \neq 0 
\end{aligned}
$$

In both cases $F = \frac{MSTR}{MSE}$, reject $H_0$ if $F > f_{(1-\alpha; r-1,(n-1)(r-1))}$

we don't use F-test to compare blocks, because

-   We have a priori that blocs are different\
-   Randomization is done "within" block.

To estimate the efficiency that was gained by blocking (relative to completely randomized design).

$$
\begin{aligned}
\hat{\sigma}^2_{CR} &= \frac{(n-1)MSBL + n(r-1)MSE}{nr-1} \\
\hat{\sigma}^2_{RB} &= MSE \\
\frac{\hat{\sigma}^2_{CR}}{\hat{\sigma}^2_{RB}} &= \text{above 1} \\
\end{aligned}
$$

then a completely randomized experiment would

$$
(\frac{\hat{\sigma}^2_{CR}}{\hat{\sigma}^2_{RB}}-1)\%%
$$

more observations than the randomized block design to get the same MSE

If batches are randomly selected then they are random effects. That is , if the experiment was repeated, a new sample of i batches would be selected,d yielding new values for $\rho_1, \rho_2,...,\rho_i$ then.

$$
\rho_1, \rho_2,...,\rho_j \sim N(0,\sigma^2_\rho)
$$

Then,

$$
Y_{ij} = \mu_{..} + \rho_i + \tau_j + \epsilon_{ij}
$$

where

-   $\mu_{..}$ fixed
-   $\rho_i$: random iid $N(0,\sigma^2_p)$
-   $\tau_j$ fixed (or random) $\sum \tau_j = 0$
-   $\epsilon_{ij} \sim iid N(0,\sigma^2)$

**Fixed Treatment**

$$
\begin{aligned}
E(Y_{ij}) &= \mu_{..} + \tau_j \\
var(Y_{ij}) &= \sigma^2_{\rho} + \sigma^2
\end{aligned}
$$

$$
\begin{aligned}
cov(Y_{ij},Y_{ij'}) &= \sigma^2 , j \neq j' \text{ treatments within same block are correlated} \\
cov(Y_{ij},Y_{i'j'}) &= 0 , i \neq i' , j \neq j'
\end{aligned}
$$

Correlation between 2 observations in the same block

$$
\frac{\sigma^2_{\rho}}{\sigma^2 + \sigma^2_{\rho}}
$$

The expected MS for the additive fixed treatment effect, random block effect is

| Source    | SS   | E(MS)                                    |
|-----------|------|------------------------------------------|
| Blocks    | SSBL | $\sigma^2 + r \sigma^2_\rho$             |
| Treatment | SSTR | $\sigma^2 + n \frac{\sum \tau^2_j}{r-1}$ |
| Error     | SSE  | $\sigma^2$                               |

**Interactions and Blocks**\
without replications within each block for each treatment, we can't consider interaction between block and treatment when the block effect is fixed. Hence, only in the random block effect, we have

$$
Y_{ij} = \mu_{..} + \rho_i + \tau_j + (\rho \tau)_{ij} + \epsilon_{ij}
$$

where

-   $\mu_{..}$ constant
-   $\rho_i \sim idd N(0,\sigma^2_{\rho})$ random
-   $\tau_j$ fixed ($\sum \tau_j = 0$)
-   $(\rho \tau)_{ij} \sim N(0,\frac{r-1}{r}\sigma^2_{\rho \tau})$ with $\sum_j (\rho \tau)_{ij}=0$ for all i
-   $cov((\rho \tau)_{ij},(\rho \tau)_{ij'})= -\frac{1}{r} \sigma^2_{\rho \tau}$ for $j \neq j'$
-   $\epsilon_{ij} \sim iid N(0,\sigma^2)$ random

Note: a special case of mixed 2-factor model with 1 observation per "cell"

$$
\begin{aligned}
E(Y_{ij}) &= \mu_{..} + \tau_j \\
var(Y_{ij}) &= \sigma^2_\rho + \frac{r-1}{r} \sigma^2_{\rho \tau} + \sigma^2
\end{aligned}
$$

$$
\begin{aligned}
cov(Y_{ij},Y_{ij'}) &= \sigma^2_\rho - \frac{1}{r} \sigma^2_{\rho \tau}, j \neq j' \text{ obs from the same block are correlated} \\
cov(Y_{ij},Y_{i'j'}) &= 0, i \neq i', j \neq j' \text{ obs from different blocks are independent}
\end{aligned}
$$

The sum of squares and degrees of freedom for interaction model are the same as those for the additive model. The difference exists only with the expected mean squares

+-----------+---------+--------------+------------------------------------------------------------------+
| Source    | SS      | df           | E(MS)                                                            |
+===========+=========+==============+==================================================================+
| Blocks    | $SSBL$  | $n-1$        | $\sigma^2 + r \sigma^2_\rho$                                     |
+-----------+---------+--------------+------------------------------------------------------------------+
| Treatment | $SSTR$  | $r -1$       | $\sigma^2 + \sigma ^2_{\rho \tau} + n \frac{\sum \tau_j^2}{r-1}$ |
+-----------+---------+--------------+------------------------------------------------------------------+
| Error     | $SSE$   | $(n-1)(r-1)$ | $\sigma^2 + \sigma ^2_{\rho \tau}$                               |
+-----------+---------+--------------+------------------------------------------------------------------+

-   No exact test is possible for block effects when interaction is present (Not important if blocks are used primarily to reduce experimental error variability)\
-   $E(MSE) = \sigma^2 + \sigma^2_{\rho \tau}$ the error term variance and interaction variance $\sigma^2_{\rho \tau}$. We can't estimate these components separately with this model. The two are **confounded**.\
-   If more than 1 observation per treatment block combination, one can consider interaction with fixed block effects, which is called **generalized randomized block designs** (multifactor analysis).

### Tukey Test of Additivity

(Tukey's 1 df test for additivity)

formal test of interaction effects between blocks and treatments for a randomized block design. can also considered for testing additivity in 2-way analyses when there is only one observation per cell.

we consider a less restricted interaction term

$$
(\rho \tau)_{ij} = D\rho_i \tau_j \text{(D: Constant)}
$$

So,

$$
Y_{ij} = \mu_{..} + \rho_i + \tau_j + D\rho_i \tau_j + \epsilon_{ij}
$$

the least square estimate or MLE for D

$$
\hat{D} = \frac{\sum_i \sum_j \rho_i \tau_j Y_{ij}}{\sum_i \rho_i^2 \sum_j \tau^2_j}
$$

replacing the parameters by their estimates

$$
\hat{D} = \frac{\sum_i \sum_j (\bar{Y}_{i.}- \bar{Y}_{..})(\bar{Y}_{.j}- \bar{Y}_{..})Y_{ij}}{\sum_i (\bar{Y}_{i.}- \bar{Y}_{..})^2 \sum_j(\bar{Y}_{.j}- \bar{Y}_{..})^2}
$$

Thus, the interaction sum of squares

$$
SSint = \sum_i \sum_j \hat{D}^2(\bar{Y}_{i.}- \bar{Y}_{..})^2(\bar{Y}_{.j}- \bar{Y}_{..})^2
$$

The ANOVA decomposition

$$
SSTO = SSBL + SSTR + SSint + SSRem
$$

where $SSRem$: remainder sum of squares

$$
SSRem = SSTO - SSBL - SSTR - SSint
$$

if $D = 0$ (i.e., no interactions of the type $D \rho_i \tau_j$). $SSint$ and $SSRem$ are independent $\chi^2_{1,rn-r-n}$.

If $D = 0$,

$$
F = \frac{SSint/1}{SSRem/(rn-r-n)} \sim f_{(1-\alpha;rn-r-n)}
$$

if

$$
\begin{aligned}
&H_0: D = 0 \text{ no interaction present} \\
&H_a: D \neq 0 \text{ interaction of form $D \rho_i \tau_j$ present}
\end{aligned}
$$

we reject $H_0$ if $F > f_{(1-\alpha;1,nr-r-n)}$

## Nested Designs

Let $\mu_{ij}$ be the mean response when factor A is at the i-th level and factor B is at the j-th level.\
If the factors are crossed, the $j$-th level of B is the same for all levels of A.\
If factor B is nested within A, the j-th level of B when A is at level 1 has nothing in common with the j-th level of B when A is at level 2.

Factors that can't be manipulated are designated as **classification factors**, as opposed to **experimental factors** (i.e., you assign to the experimental units).

### Two-Factor Nested Designs

-   Consider B is nested within A.
-   both factors are fixed
-   All treatment means are equally important.

**Mean responses**

$$
\mu_{i.} = \sum_j \mu_{ij}/b
$$

Main effect factor A

$$
\alpha_i = \mu_{i.} - \mu_{..}
$$

where $\mu_{..} = \frac{\mu_{ij}}{ab} = \frac{\sum_i \mu_{i.}}{a}$ and $\sum_i \alpha_i = 0$

Individual effects of $B$ is denoted as $\beta_{j(i)}$ where $j(i)$ indicates the $j$-th level of factor $B$ is nested within the it-h level of factor A

$$
\begin{aligned}
\beta_{j(i)} &= \mu_{ij} - \mu_{i.} \\
&= \mu_{ij} - \alpha_i - \mu_{..} \\
\sum_j \beta_{j(i)}&=0 , i = 1,...,a
\end{aligned}
$$

$\beta_{j(i)}$ is the **specific effect** of the $j$-th level of factor $B$ nested within the $i$-th level of factor $A$. Hence,

$$
\mu_{ij} \equiv \mu_{..} + \alpha_i + \beta_{j(i)} \equiv \mu_{..} + (\mu_{i.} - \mu_{..}) + (\mu_{ij} - \mu_{i.})
$$

**Model**

$$
Y_{ijk} = \mu_{..} + \alpha_i + \beta_{j(i)} + \epsilon_{ijk}
$$

where

-   $Y_{ijk}$ response for the $k$-th treatment when factor $A$ is at the $i$-th level and factor $B$ is at the $j$-th level $(i = 1,..,a; j = 1,..,b; k = 1,..n)$
-   $\mu_{..}$ constant
-   $\alpha_i$ constants subject to restriction $\sum_i \alpha_i = 0$
-   $\beta_{j(i)}$ constants subject to restriction $\sum_j \beta_{j(i)} = 0$ for all $i$
-   $\epsilon_{ijk} \sim iid N(0,\sigma^2)$

$$
\begin{aligned}
E(Y_{ijk}) &= \mu_{..} + \alpha_i + \beta_{j(i)} \\
var(Y_{ijk}) &= \sigma^2
\end{aligned}
$$

there is no interaction term in a nested model

**ANOVA for Two-Factor Nested Designs**

Least Squares and MLE estimates

+----------------------+---------------------------------+
| Parameter            | Estimator                       |
+======================+=================================+
| $\mu_{..}$           | $\bar{Y}_{...}$                 |
+----------------------+---------------------------------+
| $\alpha_i$           | $\bar{Y}_{i..} - \bar{Y}_{...}$ |
+----------------------+---------------------------------+
| $\beta_{j(i)}$       | $\bar{Y}_{ij.} - \bar{Y}_{i..}$ |
+----------------------+---------------------------------+
| $\hat{Y}_{ijk}$      | $\bar{Y}_{ij.}$                 |
+----------------------+---------------------------------+

residual $e_{ijk} = Y_{ijk} - \bar{Y}_{ijk}$

$$
\begin{aligned}
SSTO &= SSA + SSB(A) + SSE \\
\sum_i \sum_j \sum_k (Y_{ijk}- \bar{Y}_{...})^2 &= bn \sum_i (\bar{Y}_{i..}- \bar{Y}_{...})^2 + n \sum_i \sum_j (\bar{Y}_{ij.}- \bar{Y}_{i..})^2  \\
&+ \sum_i \sum_j \sum_k (Y_{ijk} -\bar{Y}_{ij.})^2
\end{aligned}
$$

ANOVA Table

+---------------------+----------+-----------+----------+---------------------------------------------------------------+
| Source of Variation | SS       | df        | MS       | E(MS)                                                         |
+=====================+==========+===========+==========+===============================================================+
| Factor A            | $SSA$    | $a-1$     | $MSA$    | $\sigma^2 + bn \frac{\sum \alpha_i^2}{a-1}$                   |
+---------------------+----------+-----------+----------+---------------------------------------------------------------+
| Factor B            | $SSB(A)$ | $a(b-1)$  | $MSB(A)$ | $\sigma^2 + n \frac{\  | | | | um \sum e ta_{i)}^ 2}{a(b-1)}$ |
+---------------------+----------+-----------+----------+---------------------------------------------------------------+
| Error               | $SSE$    | $ab(n-1)$ | $MSE$    | $\sigma^2$                                                    |
+---------------------+----------+-----------+----------+---------------------------------------------------------------+
| Total               | $SSTO$   | $abn -1$  |          |                                                               |
+---------------------+----------+-----------+----------+---------------------------------------------------------------+

**Tests For Factor Effects**

$$
\begin{aligned}
&H_0: \text{ All } \alpha_i =0 \\
&H_a: \text{ not all } \alpha_i = 0
\end{aligned}
$$

$F = \frac{MSA}{MSE} \sim f_{(1-\alpha;a-1,(n-1)ab)}$ reject if $F > f$

$$
\begin{aligned}
&H_0: \text{ All } \beta_{j(i)} =0 \\
&H_a: \text{ not all } \beta_{j(i)} = 0
\end{aligned}
$$

$F = \frac{MSB(A)}{MSE} \sim f_{(1-\alpha;a(b-1),(n-1)ab)}$ reject $F>f$

**Testing Factor Effect Contrasts**

$L = \sum c_i \mu_i$ where $\sum c_i =0$

$$
\begin{aligned}
\hat{L} &= \sum c_i \bar{Y}_{i..} \\
\hat{L} &\pm t_{(1-\alpha/2;df)}s(\hat{L})
\end{aligned}
$$

where $s^2(\hat{L}) = \sum c_i^2 s^2(\bar{Y}_{i..})$, where $s^2(\bar{Y}_{i..}) = \frac{MSE}{bn}, df = ab(n-1)$

**Testing Treatment Means**

$L = \sum c_i \mu_{.j}$ estimated by $\hat{L} = \sum c_i \bar{Y}_{ij}$ with confidence limits:

$$
\hat{L} \pm t_{(1-\alpha/2;(n-1)ab)}s(\hat{L})
$$

where

$$
s^2(\hat{L}) = \frac{MSE}{n}\sum c^2_i
$$

**Unbalanced Nested Two-Factor Designs**

If there are different number of levels of factor $B$ for different levels of factor $A$, then the design is called **unbalanced**

The model

$$
\begin{aligned}
Y_{ijk} &= \mu_{..} + \alpha_i + \beta_{j(i)} + \epsilon_{ijk} \\
\sum_{i=1}^2 \alpha_i &=0 \\
\sum_{j=1}^3 \beta_{j(1)} &= 0 \\
\sum_{j=1}^2 \beta_{j(2)}&=0
\end{aligned}
$$

where

-   $i = 1,2;j =1,..,b_i;k=1,..,n_{ij}$

-   $b_1 = 3, b_2= 2, n_{11} = n_{13} =2, n_{12}=1,n_{21} = n_{22} = 2$

-   $\alpha_1,\beta_{1(1)}, \beta_{2(1)}, \beta_{1(2)}$ are parameters.

And constraints: $\alpha_2 = - \alpha_1, \beta_{3(1)}= - \beta_{1(1)}-\beta_{2(1)}, \beta_{2(2)}=-\beta_{1(2)}$

4 indicator variables


\begin{equation}
X_1 = 
\begin{cases}
1&\text{if obs from school 1}\\
-1&\text{if obs from school 2}\\
\end{cases}
\end{equation}


\begin{equation}
X_2 = 
\begin{cases}
1&\text{if obs from instructor 1 in school 1}\\
-1&\text{if obs from instructor 3 in school 1}\\
0&\text{otherwise}\\
\end{cases}
\end{equation}


\begin{equation}
X_3 = 
\begin{cases}
1&\text{if obs from instructor 2 in school 1}\\
-1&\text{if obs from instructor 3 in school 1}\\
0&\text{otherwise}\\
\end{cases}
\end{equation}


\begin{equation}
X_4 = 
\begin{cases}
1&\text{if obs from instructor 1 in school 1}\\
-1&\text{if obs from instructor 2 in school 1}\\
0&\text{otherwise}\\
\end{cases}
\end{equation}


Regression Full Model

$$
Y_{ijk} = \mu_{..} + \alpha_1 X_{ijk1} + \beta_{1(1)}X_{ijk2} + \beta_{2(1)}X_{ijk3} + \beta_{1(2)}X_{ijk4} + \epsilon_{ijk}
$$

**Random Factor Effects**

If

$$
\begin{aligned}
\alpha_1 &\sim iid N(0,\sigma^2_\alpha) \\
\beta_{j(i)} &\sim iid N(0,\sigma^2_\beta)
\end{aligned}
$$

+-------------+-----------------------------------------------------------------+------------------------------------------------------+
| Mean Square | Expected Mean Squares                                           | Expected Mean Squares                                |
|             |                                                                 |                                                      |
|             | A fixed, B random                                               | A random, B random                                   |
+=============+=================================================================+======================================================+
| MSA         | $\sigma^ 2 + n \sigma^2_\beta + bn \frac{\sum \alpha_i^2}{a-1}$ | $\sigma^2 + bn \sigma^2_{\alpha} + n \sigma^2_\beta$ |
+-------------+-----------------------------------------------------------------+------------------------------------------------------+
| MSB(A)      | $\sigma^2 + n \sigma^2_\beta$                                   | $\sigma^2 + n \sigma^2_\beta$                        |
+-------------+-----------------------------------------------------------------+------------------------------------------------------+
| MSE         | $\sigma^2$                                                      | $\sigma^2$                                           |
+-------------+-----------------------------------------------------------------+------------------------------------------------------+

Test Statistics

| Factor A    | $\frac{MSA}{MSB(A)}$ | $\frac{MSA}{MSB(A)}$ |
|-------------|----------------------|----------------------|
| Factor B(A) | $\frac{MSB(A)}{MSE}$ | $\frac{MSB(A)}{MSE}$ |

Another way to increase the precision of treatment comparisons by reducing variability is to use regression models to adjust for differences among experimental units (also known as **analysis of covariance**).

## Single Factor Covariance Model

$$
Y_{ij} = \mu_{.} + \tau_i + \gamma(X_{ij} - \bar{X}_{..}) + \epsilon_{ij} 
$$

for $i = 1,...,r;j=1,..,n_i$

where

-   $\mu_.$ overall mean
-   $\tau_i$: fixed treatment effects ($\sum \tau_i =0$)
-   $\gamma$: fixed regression coefficient effect between X and Y
-   $X_{ij}$ covariate (not random)
-   $\epsilon_{ij} \sim iid N(0,\sigma^2)$: random errors

If we just use $\gamma X_{ij}$ as the regression term (rather than $\gamma(X_{ij}-\bar{X}_{..})$), then $\mu_.$ is no longer the overall mean; thus we need to centered mean.

$$
\begin{aligned}
E(Y_{ij}) &= \mu_. + \tau_i + \gamma(X_{ij}-\bar{X}_{..}) \\
var(Y_{ij}) &= \sigma^2
\end{aligned}
$$

$Y_{ij} \sim N(\mu_{ij},\sigma^2)$,

where

$$
\begin{aligned}
\mu_{ij} &= \mu_. + \tau_i + \gamma(X_{ij} - \bar{X}_{..}) \\
\sum \tau_i &=0 
\end{aligned}
$$

Thus, the mean response ($\mu_{ij}$) is a regression line with intercept $\mu_. + \tau_i$ and slope $\gamma$ for each treatment \$\$i.

**Assumption**:

-   All treatment regression lines have the same slope\
-   when treatment interact with covariate $X$ (non-parallel slopes), covariance analysis is **not** appropriate. in which case we should use separate regression lines.

More complicated regression features (e.g., quadratic, cubic) or additional covariates e.g.,

$$
Y_{ij} = \mu_. + \tau_i + \gamma_1(X_{ij1}-\bar{X}_{..2}) + \gamma_2(X_{ij2}-\bar{X}_{..2}) + \epsilon_{ij}
$$

**Regression Formulation**

We can use indicator variables for treatments

$$
l_1 =
\begin{cases}
1 & \text{if case is from treatment 1}\\
-1 & \text{if case is from treatment r}\\
0 &\text{otherwise}\\
\end{cases}
$$

$$
.
$$

$$
.
$$

$$
l_{r-1} =
\begin{cases}
1 & \text{if case is from treatment r-1}\\
-1 & \text{if case is from treatment r}\\
0 &\text{otherwise}\\
\end{cases}
$$

Let $x_{ij} = X_{ij}- \bar{X}_{..}$. the regression model is

$$
Y_{ij} = \mu_. + \tau_1l_{ij,1} + .. + \tau_{r-1}l_{ij,r-1} + \gamma x_{ij}+\epsilon_{ij}
$$

where $I_{ij,1}$ is the indicator variable $l_1$ for the j-th case from treatment i. The treatment effect $\tau_1,..\tau_{r-1}$ are just regression coefficients for the indicator variables.

We could use the same diagnostic tools for this case.

**Inference**

Treatment effects

$$
\begin{aligned}
&H_0: \tau_1 = \tau_2 = ...= 0 \\
&H_a: \text{not all } \tau_i =0
\end{aligned}
$$

$$
\begin{aligned}
&\text{Full Model}: Y_{ij} = \mu_. + \tau_i + \gamma X_{ij} +\epsilon_{ij}  \\
&\text{Reduced Model}: Y_{ij} = \mu_. + \gamma X_{ij} + \epsilon_{ij}
\end{aligned}
$$

$$
F = \frac{SSE(R) - SSE(F)}{(N-2)-(N-(r+1))} / \frac{SSE(F)}{N-(r+1)} \sim F_{(r-1,N-(r+1))}
$$

If we are interested in comparisons of treatment effects.\
For example, r - 3. We estimate $\tau_1,\tau_2, \tau_3 = -\tau_1 - \tau_2$

+-------------------+---------------------------------+-----------------------------------------------------------------------------+
| Comparison        | Estimate                        | Variance of Estimator                                                       |
+===================+=================================+=============================================================================+
| $\tau_1 - \tau_2$ | $\hat{\tau}_1 - \hat{\tau}_2$   | $var(\hat {\tau}_1) + var(\hat{\tau}_2) - 2cov(\hat{ \tau}_1\hat{\tau}_2)$  |
+-------------------+---------------------------------+-----------------------------------------------------------------------------+
| $\tau_1 - \tau_3$ | $2 \hat{\tau}_1 + \hat{\tau}_2$ | $4var(\hat {\tau}_1) + var(\hat{\tau}_2) - 4cov(\hat{ \tau}_1\hat{\tau}_2)$ |
+-------------------+---------------------------------+-----------------------------------------------------------------------------+
| $\tau_2 - \tau_3$ | $\hat{\tau}_1 + 2 \hat{\tau}_2$ | $var(\hat{\tau}_1) + 4var(\hat{\tau}_2) - 4cov(\hat{\tau}_1\hat{\tau}_2)$   |
+-------------------+---------------------------------+-----------------------------------------------------------------------------+

Testing for Parallel Slopes

Example:

r = 3

$$
Y_{ij} = \mu_{.} + \tau_1 I_{ij,1} + \tau_2 I_{ij,2} + \gamma X_{ij} + \beta_1 I_{ij,1}X_{ij} + \beta_2 I_{ij,2}X_{ij} + \epsilon_{ij}
$$

where $\beta_1,\beta_2$: interaction coefficients.

$$
\begin{aligned}
&H_0: \beta_1 = \beta_2 = 0 \\
&H_a: \text{at least one} \beta \neq 0 
\end{aligned}
$$

If we can't reject $H_0$ using F-test then we have evidence that the slopes are parallel

**Adjusted Means**

The means in response after adjusting for the covariate effect

$$
Y_{i.}(adj) = \bar{Y}_{i.} - \hat{\gamma}(\bar{X}_{i.} - \bar{X}_{..})
$$
