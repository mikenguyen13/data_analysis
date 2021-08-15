# Generalized Method of Moments

# Minimum Distance

# Spline Regression

This chapter is based on [CMU stat](https://www.stat.cmu.edu/~ryantibs/advmethods/notes/smoothspline.pdf)

Definition: a k-th order spline is a piecewise polynomial function of degree k, that is continuous and has continuous derivatives of orders 1,..., k -1, at its knot points

Equivalently, a function f is a k-th order spline with knot points at $t_1 < ...< t_m$ if

-   f is a polynomial of degree k on each of the intervals $(-\infty, t_1], [t_1,t_2],...,[t_m, \infty)$\
-   $f^{(j)}$, the j-th derivative of f, is continuous at $t_1,...,t_m$ for each j = 0,1,...,k-1

A common case is when k = 3, called cubic splines. (piecewise cubic functions are continuous, and also continuous at its first and second derivatives)

To parameterize the set of splines, we could use **truncated power basis**, defined as

$$
g_1(x) = 1 \\
g_2(x) = x \\
... \\
g_{k+1}(x) = x^k \\
g_{k+1+j}(x) = (x-t_j)^k_+
$$

where j = 1,...,m and $x_+$ = max{x,0}

However, now software typically use B-spline basis.

## Regression Splines

To estimate the regression function $r(X) = E(Y|X =x)$, we can fit a k-th order spline with knots at some prespecified locations $t_1,...,t_m$

Regression splines are functions of

$$
\sum_{j=1}^{m+k+1} \beta_jg_j
$$

where

$\beta_1,..\beta_{m+k+1}$ are coefficients $g_1,...,g_{m+k+1}$ are the truncated power basis functions for k-th order splines over the knots $t_1,...,t_m$

To estimate the coefficients

$$
\sum_{i=1}^{n} (y_i - \sum_{j=1}^{m} \beta_j g_j (x_i))^2
$$

then regression spline is

$$
\hat{r}(x) = \sum_{j=1}^{m+k+1} \hat{\beta}_j g_j (x)
$$

If we define the basis matrix $G \in R^{n \times (m+k+1)}$ by $$
G_{ij} = g_j(x_i) 
$$ where $i = 1,..,n$ , $j = 1,..,m+k+1$

Then,

$$
\sum_{i=1}^{n} (y_i - \sum_{j=1}^{m} \beta_j g_j (x_i))^2 = ||y - G \beta||_2^2
$$

and the regression spline estimate at x is

$$
\hat{r} (x) = g(x)^T \hat{\beta}= g(x)^T(G^TG)^{-1}G^Ty
$$

## Natural splines

A natural spline of order k, with knots at $t_1 <...< t_m$, is a piecewise polynomial function f such that

-   f is polynomial of degree k on each of $[t_1,t_2],...,[t_{m-1},t_m]$
-   f is a polynomial of degree $(k-1)/2$ on $(-\infty,t_1]$ and $[t_m,\infty)$
-   f is continuous and has continuous derivatives of orders 1,.,,, k -1 at its knots $t_1,..,t_m$

**Note**

natural splines are only defined for odd orders k.

## Smoothing splines

These estimators use a regularized regression over the natural spline basis: placing knots at all points $x_1,...x_n$

For the case of cubic splines, the coefficients are the minimization of

$$
||y - G\beta||^2_2 + \lambda \beta^T \Omega \beta
$$

where $\Omega \in R^{n \times n}$ is the penalty matrix

$$
\Omega_{ij} = \int g''_i(t) g''_j(t) dt,
$$

and i,j = 1,..,n

and $\lambda \beta^T \Omega \beta$ is the **regularization term** used to shrink the components of $\hat{\beta}$ towards 0. $\lambda > 0$ is the tuning parameter (or smoothing parameter). Higher value of $\lambda$, faster shrinkage (shrinking away basis functions)

**Note**\
smoothing splines have similar fits as kernel regression.

|                  | Smoothing splines             | kernel regression |
|------------------|-------------------------------|-------------------|
| tuning parameter | smoothing parameter $\lambda$ | bandwidth h       |

## Application


```r
library(tidyverse)
```

```
## Warning: package 'tidyverse' was built under R version 4.0.5
```

```
## -- Attaching packages --------------------------------------- tidyverse 1.3.1 --
```

```
## v ggplot2 3.3.3     v purrr   0.3.4
## v tibble  3.1.2     v dplyr   1.0.6
## v tidyr   1.1.3     v stringr 1.4.0
## v readr   1.4.0     v forcats 0.5.1
```

```
## Warning: package 'dplyr' was built under R version 4.0.5
```

```
## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
```

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## 
## Attaching package: 'caret'
```

```
## The following object is masked from 'package:purrr':
## 
##     lift
```

```r
theme_set(theme_classic())

# Load the data
data("Boston", package = "MASS")
# Split the data into training and test set
set.seed(123)
training.samples <- Boston$medv %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- Boston[training.samples, ]
test.data <- Boston[-training.samples, ]

knots <- quantile(train.data$lstat, p = c(0.25, 0.5, 0.75)) # we use 3 knots at 25,50,and 75 quantile.

library(splines)
# Build the model
knots <- quantile(train.data$lstat, p = c(0.25, 0.5, 0.75))
model <- lm (medv ~ bs(lstat, knots = knots), data = train.data)
# Make predictions
predictions <- model %>% predict(test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test.data$medv),
  R2 = R2(predictions, test.data$medv)
)
```

```
##       RMSE        R2
## 1 5.317372 0.6786367
```

```r
ggplot(train.data, aes(lstat, medv) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ splines::bs(x, df = 3))
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-1-1.png)<!-- -->

```r
attach(train.data)
#fitting smoothing splines using smooth.spline(X,Y,df=...)

fit1<-smooth.spline(train.data$lstat,train.data$medv,df=3 ) # 3 degrees of freedom
#Plotting both cubic and Smoothing Splines 
plot(train.data$lstat,train.data$medv,col="grey")
lstat.grid = seq(from = range(lstat)[1], to = range(lstat)[2])
points(lstat.grid,predict(model,newdata = list(lstat=lstat.grid)),col="darkgreen",lwd=2,type="l")
#adding cutpoints
abline(v=c(10,20,30),lty=2,col="darkgreen")
lines(fit1,col="red",lwd=2)

legend("topright",c("Smoothing Spline with 3 df","Cubic Spline"),col=c("red","darkgreen"),lwd=2)
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-1-2.png)<!-- -->

# Generalized Additive Models

To overcome [Spline Regression]'s requirements for specifying the knots, we can use [Generalized Additive Models] or GAM.


```r
library(mgcv)
```

```
## Loading required package: nlme
```

```
## 
## Attaching package: 'nlme'
```

```
## The following object is masked from 'package:dplyr':
## 
##     collapse
```

```
## This is mgcv 1.8-34. For overview type 'help("mgcv-package")'.
```

```r
# Build the model
model <- gam(medv ~ s(lstat), data = train.data)
plot(model)
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-2-1.png)<!-- -->

```r
# Make predictions
# predictions <- model %>% predict(test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test.data$medv),
  R2 = R2(predictions, test.data$medv)
)
```

```
##       RMSE        R2
## 1 5.317372 0.6786367
```

```r
ggplot(train.data, aes(lstat, medv) ) +
  geom_point() +
  stat_smooth(method = gam, formula = y ~ s(x))
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-2-2.png)<!-- -->

```r
detach(train.data)
```

# Quantile Regression

For academic review on quantile regression, check [@Yu_2003]

[Linear Regression] is based on the conditional mean function $E(y|x)$

In Quantile regression, we can view each points in the conditional distribution of y. Quantile regression estimates the conditional median or any other quantile of Y.

In the case that we're interested in the 50th percentile, quantile regression is median regression, also known as least-absolute-deviations (LAD) regression, minimizes $\sum_{i}|e_i|$

Properties of estimators $\beta$

-   Asymptotically normally distributed

Advantages

-   More robust to outliers compared to [OLS][Ordinary Least Squares]
-   In the case the dependent variable has a bimodal or multimodal (multiple humps with multiple modes) distribution, quantile regression can be extremely useful.
-   Avoids parametric distribution assumption of the error process. In another word, no assumptions regarding the distribution of the error term.
-   Better characterization of the data (not just its conditional mean)
-   is invariant to monotonic transformations (such as log) while OLS is not. In another word, $E(g(y))=g(E(y))$

Disadvantages

-   The dependent variable needs to be continuous with no zeroes or too many repeated values.

$$
y_i = x_i'\beta_q + e_i
$$

Let $e(x) = y -\hat{y}(x)$, then $L(e(x)) = L(y -\hat{y}(x))$ is the loss function of the error term.

If $L(e) = |e|$ (called absolute-error loss function) then $\hat{\beta}$ can be estimated by minimizing $\sum_{i}|y_i-x_i'\beta|$

More specifically, the objective function is 


$$
Q(\beta_q)=\sum_{i:y_i \ge x_i'\beta}^{N} q|y_i - x_i'\beta_q| + \sum_{i:y_i < x_i'\beta}^{N} (1-q)|y_i-x_i'\beta_q
$$ 

where $0<q<1$

The sum penalizes $q|e_i|$ for under-prediction and $(1-q)|e_i|$ for over-prediction

We use simplex method to minimize this function (cannot use analytical solution since it's non-differentiable). Standard errors can be estimated by bootstrap.

The absolute-error loss function is symmetric.

**Interpretation** For the jth regressor ($x_j$), the marginal effect is the coefficient for the qth quantile

$$
\frac{\partial Q_q(y|x)}{\partial x_j} = \beta_{qj}
$$ 

At the quantile q of the dependent variable y, $\beta_q$ represents a one unit change in the independent variable $x_j$ on the dependent variable y.

In other words, at the qth percentile, a one unit change in x results in $\beta_q$ unit change in y.

## Application


```r
# generate data with non-constant variance

x <- seq(0,100,length.out = 100)        # independent variable
sig <- 0.1 + 0.05*x                     # non-constant variance
b_0 <- 3                                # true intercept
b_1 <- 0.05                             # true slope
set.seed(1)                             # reproducibility
e <- rnorm(100,mean = 0, sd = sig)      # normal random error with non-constant variance
y <- b_0 + b_1*x + e                    # dependent variable
dat <- data.frame(x,y)
hist(y)
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-3-1.png)<!-- -->

```r
library(ggplot2)
ggplot(dat, aes(x,y)) + geom_point()
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-3-2.png)<!-- -->

```r
ggplot(dat, aes(x,y)) + geom_point() + geom_smooth(method="lm")
```

```
## `geom_smooth()` using formula 'y ~ x'
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-3-3.png)<!-- -->

We follow [@Roger_1996] to estimate quantile regression


```r
library(quantreg)
```

```
## Warning: package 'quantreg' was built under R version 4.0.5
```

```
## Loading required package: SparseM
```

```
## 
## Attaching package: 'SparseM'
```

```
## The following object is masked from 'package:base':
## 
##     backsolve
```

```
## Warning in .recacheSubclasses(def@className, def, env): undefined subclass
## "numericVector" of class "Mnumeric"; definition not updated
```

```r
qr <- rq(y ~ x, data=dat, tau = 0.5) # tau: quantile of interest. Here we have it at 50th percentile.
summary(qr)
```

```
## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique
```

```
## 
## Call: rq(formula = y ~ x, tau = 0.5, data = dat)
## 
## tau: [1] 0.5
## 
## Coefficients:
##             coefficients lower bd upper bd
## (Intercept) 3.02410      2.80975  3.29408 
## x           0.05351      0.03838  0.06690
```

adding the regression line


```r
ggplot(dat, aes(x,y)) + geom_point() + 
  geom_abline(intercept=coef(qr)[1], slope=coef(qr)[2])
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-5-1.png)<!-- -->

To have R estimate multiple quantile at once


```r
qs <- 1:9/10
qr1 <- rq(y ~ x, data=dat, tau = qs)
#check for its coefficients
coef(qr1)
```

```
##                tau= 0.1   tau= 0.2   tau= 0.3   tau= 0.4   tau= 0.5   tau= 0.6
## (Intercept)  2.95735740 2.93735462 3.19112214 3.08146314 3.02409828 3.16840820
## x           -0.01203696 0.01942669 0.02394535 0.04208019 0.05350556 0.06507385
##               tau= 0.7   tau= 0.8 tau= 0.9
## (Intercept) 3.09507770 3.10539343 3.041681
## x           0.07783556 0.08782548 0.111254
```

```r
# plot
ggplot(dat, aes(x,y)) + geom_point() + geom_quantile(quantiles = qs)
```

```
## Smoothing formula not specified. Using: y ~ x
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-6-1.png)<!-- -->

To examine if the quantile regression is appropriate, we can see its plot compared to least squares regression


```r
plot(summary(qr1), parm="x")
```

```
## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-7-1.png)<!-- -->

where red line is the least squares estimates, and its confidence interval. x-axis is the quantile y-axis is the value of the quantile regression coefficients at different quantile

If the error term is normally distributed, the quantile regression line will fall inside the coefficient interval of least squares regression.


```r
# generate data with constant variance

x <- seq(0, 100, length.out = 100)    # independent variable
b_0 <- 3                              # true intercept
b_1 <- 0.05                           # true slope
set.seed(1)                           # reproducibility
e <- rnorm(100, mean = 0, sd = 1)     # normal random error with constant variance
y <- b_0 + b_1 * x + e                # dependent variable
dat2 <- data.frame(x, y)
qr2 = rq(y ~ x, data = dat2, tau = qs)
plot(summary(qr2), parm = "x")
```

```
## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique

## Warning in rq.fit.br(x, y, tau = tau, ci = TRUE, ...): Solution may be nonunique
```

![](06-2-nonlinear_regression2_files/figure-epub3/unnamed-chunk-8-1.png)<!-- -->
