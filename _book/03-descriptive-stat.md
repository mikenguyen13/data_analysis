# Descriptive Statistics {#descriptive-stat}

When you have an area of interest that you want to research, a problem that you want to solve, a relationship that you want to investigate, theoretical and empirical processes will help you.

Estimand is defined as "a quantity of scientific interest that can be calculated in the population and does not change its value depending on the data collection design used to measure it (i.e., it does not vary with sample size and survey design, or the number of nonrespondents, or follow-up efforts)." [@Rubin_1996]

Estimands include:

-   population means
-   Population variances
-   correlations
-   factor loading
-   regression coefficients

## Numerical Measures

There are differences between a population and a sample

+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| Measures of      | Category                                                                  | Population                                                                            | Sample                                                                                                               |
+==================+===========================================================================+=======================================================================================+======================================================================================================================+
| \-               | What is it?                                                               | Reality                                                                               | A small fraction of reality (inference)                                                                              |
+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| \-               | Characteristics described by                                              | Parameters                                                                            | Statistics                                                                                                           |
+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| Central Tendency | Mean                                                                      | $\mu = E(Y)$                                                                          | $\hat{\mu} = \overline{y}$                                                                                           |
+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| Central Tendency | Median                                                                    | 50-th percentile                                                                      | $y_{(\frac{n+1}{2})}$                                                                                                |
+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| Dispersion       | Variance                                                                  | $\sigma^2=var(Y)$  $=E(Y-\mu)^2$                                                      | $s^2=\frac{1}{n-1} \sum_{i = 1}^{n} (y_i-\overline{y})^2$  $=\frac{1}{n-1} \sum_{i = 1}^{n} (y_i^2-n\overline{y}^2)$ |
+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| Dispersion       | Coefficient of Variation                                                  | $\frac{\sigma}{\mu}$                                                                  | $\frac{s}{\overline{y}}$                                                                                             |
+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| Dispersion       | Interquartile Range                                                       | difference between 25th and 75th percentiles. Robust to outliers                      |                                                                                                                      |
+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| Shape            | Skewness  Standardized 3rd central moment (unitless)                      | $g_1=\frac{\mu_3}{\mu_2^{3/2}}$                                                       | $\hat{g_1}=\frac{m_3}{m_2sqrt(m_2)}$                                                                                 |
+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| Shape            | Central moments                                                           | $\mu=E(Y)$  $\mu_2 = \sigma^2=E(Y-\mu)^2$  $\mu_3 = E(Y-\mu)^3$  $\mu_4 = E(Y-\mu)^4$ | $m_2=\sum_{i=1}^{n}(y_1-\overline{y})^2/n$   $m_3=\sum_{i=1}^{n}(y_1-\overline{y})^3/n$                              |
+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| Shape            | Kurtosis (peakedness and tail thickness)  Standardized 4th central moment | $g_2^*=\frac{E(Y-\mu)^4}{\sigma^4}$                                                   | $\hat{g_2}=\frac{m_4}{m_2^2}-3$                                                                                      |
+------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+

Note:

-   Order Statistics: $y_{(1)},y_{(2)},...,y_{(n)}$ where $y_{(1)}<y_{(2)}<...<y_{(n)}$

-   Coefficient of variation: standard deviation over mean. This metric is stable, dimensionless statistic for comparison.

-   Symmetric: mean = median, skewness = 0

-   Skewed right: mean > median, skewness > 0

-   Skewed left: mean \< median, skewness \< 0

-   Central moments: $\mu=E(Y)$ , $\mu_2 = \sigma^2=E(Y-\mu)^2$ , $\mu_3 = E(Y-\mu)^3$, $\mu_4 = E(Y-\mu)^4$

-   For normal distributions, $\mu_3=0$, so $g_1=0$

-   $\hat{g_1}$ is distributed approximately as N(0,6/n) if sample is from a normal population. (valid when n > 150)

    -   For large samples, inference on skewness can be based on normal tables with 95% confidence interval for $g_1$ as $\hat{g_1}\pm1.96\sqrt{6/n}$
    -   For small samples, special tables from Snedecor and Cochran 1989, Table A 19(i) or Monte Carlo test

+-----------------------------+--------------+---------------------------------------------------------------------------------+
| Kurtosis \> 0 (leptokurtic) | heavier tail | compared to a normal distribution with the same $\sigma$ (e.g., t-distribution) |
+-----------------------------+--------------+---------------------------------------------------------------------------------+
| Kurtosis \< 0 (platykurtic) | lighter tail | compared to a normal distribution with the same $\sigma$                        |
+-----------------------------+--------------+---------------------------------------------------------------------------------+

-   For a normal distribution, $g_2^*=3$. Kurtosis is often redefined as: $g_2=\frac{E(Y-\mu)^4}{\sigma^4}-3$ where the 4th central moment is estimated by $m_4=\sum_{i=1}^{n}(y_i-\overline{y})^4/n$

    -   the asymptotic sampling distribution for $\hat{g_2}$ is approximately N(0,24/n) (with n > 1000)
    -   large sample on kurtosis uses standard normal tables
    -   small sample uses tables by Snedecor and Cochran, 1989, Table A 19(ii) or Geary 1936


```r
data = rnorm(100)
library(e1071)
skewness(data,type = 1)
#> [1] 0.2054197
kurtosis(data, type = 1)
#> [1] -0.4962623
```

## Graphical Measures

### Shape

It's a good habit to label your graph, so others can easily follow.


```r
data = rnorm(100)

# Histogram
hist(data,labels = T,col="grey",breaks = 12) 
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-2-1.png" width="90%" style="display: block; margin: auto;" />

```r
# Interactive histogram  
pacman::p_load("highcharter")
hchart(data) 
```

```{=html}
<div id="htmlwidget-4977aa583d485554e5f9" style="width:100%;height:500px;" class="highchart html-widget"></div>
<script type="application/json" data-for="htmlwidget-4977aa583d485554e5f9">{"x":{"hc_opts":{"chart":{"reflow":true,"zoomType":"x"},"title":{"text":null},"yAxis":{"title":{"text":null}},"credits":{"enabled":false},"exporting":{"enabled":false},"boost":{"enabled":false},"plotOptions":{"series":{"label":{"enabled":false},"turboThreshold":0},"treemap":{"layoutAlgorithm":"squarified"}},"tooltip":{"formatter":"function() { return  this.point.name + '<br/>' + this.y; }"},"series":[{"data":[{"x":-2.75,"y":1,"name":"(-3, -2.5]"},{"x":-2.25,"y":1,"name":"(-2.5, -2]"},{"x":-1.75,"y":6,"name":"(-2, -1.5]"},{"x":-1.25,"y":11,"name":"(-1.5, -1]"},{"x":-0.75,"y":21,"name":"(-1, -0.5]"},{"x":-0.25,"y":16,"name":"(-0.5, 0]"},{"x":0.25,"y":21,"name":"(0, 0.5]"},{"x":0.75,"y":12,"name":"(0.5, 1]"},{"x":1.25,"y":7,"name":"(1, 1.5]"},{"x":1.75,"y":2,"name":"(1.5, 2]"},{"x":2.25,"y":2,"name":"(2, 2.5]"}],"type":"column","pointRange":0.5,"groupPadding":0,"pointPadding":0,"borderWidth":0}]},"theme":{"chart":{"backgroundColor":"transparent"},"colors":["#7cb5ec","#434348","#90ed7d","#f7a35c","#8085e9","#f15c80","#e4d354","#2b908f","#f45b5b","#91e8e1"]},"conf_opts":{"global":{"Date":null,"VMLRadialGradientURL":"http =//code.highcharts.com/list(version)/gfx/vml-radial-gradient.png","canvasToolsURL":"http =//code.highcharts.com/list(version)/modules/canvas-tools.js","getTimezoneOffset":null,"timezoneOffset":0,"useUTC":true},"lang":{"contextButtonTitle":"Chart context menu","decimalPoint":".","downloadJPEG":"Download JPEG image","downloadPDF":"Download PDF document","downloadPNG":"Download PNG image","downloadSVG":"Download SVG vector image","drillUpText":"Back to {series.name}","invalidDate":null,"loading":"Loading...","months":["January","February","March","April","May","June","July","August","September","October","November","December"],"noData":"No data to display","numericSymbols":["k","M","G","T","P","E"],"printChart":"Print chart","resetZoom":"Reset zoom","resetZoomTitle":"Reset zoom level 1:1","shortMonths":["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],"thousandsSep":" ","weekdays":["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]}},"type":"chart","fonts":[],"debug":false},"evals":["hc_opts.tooltip.formatter"],"jsHooks":[]}</script>
# Box-and-Whisker plot
boxplot(count ~ spray, data = InsectSprays,col = "lightgray",main="boxplot")
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-2-3.png" width="90%" style="display: block; margin: auto;" />

```r
# Notched Boxplot
boxplot(len~supp*dose, data=ToothGrowth, notch=TRUE,
  col=(c("gold","darkgreen")),
  main="Tooth Growth", xlab="Suppliment and Dose")
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-2-4.png" width="90%" style="display: block; margin: auto;" />

```r
# If notches differ -> medians differ

# Stem-and-Leaf Plots
stem(data)
#> 
#>   The decimal point is at the |
#> 
#>   -2 | 72
#>   -1 | 998765544332221000000
#>   -0 | 998888887777777655554322222111100
#>    0 | 000011122222233333344666677788889
#>    1 | 012224567
#>    2 | 14
# Bagplot - A 2D Boxplot Extension
pacman::p_load(aplpack)
attach(mtcars)
bagplot(wt,mpg, xlab="Car Weight", ylab="Miles Per Gallon",
  main="Bagplot Example")
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-2-5.png" width="90%" style="display: block; margin: auto;" />

Others more advanced plots


```r
# boxplot.matrix()  #library("sfsmisc")
# boxplot.n()       #library("gplots")
# vioplot()         #library("vioplot")
```

### Scatterplot


```r
# pairs(mtcars)
```

## Normality Assessment

Since Normal (Gaussian) distribution has many applications, we typically want/ wish our data or our variable is normal. Hence, we have to assess the normality based on not only [Numerical Measures] but also [Graphical Measures]

### Graphical Assessment


```r
pacman::p_load("car")
qqnorm(precip, ylab = "Precipitation [in/yr] for 70 US cities")
qqline(precip)
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-5-1.png" width="90%" style="display: block; margin: auto;" />

The straight line represents the theoretical line for normally distributed data. The dots represent real empirical data that we are checking. If all the dots fall on the straight line, we can be confident that our data follow a normal distribution. If our data wiggle and deviate from the line, we should be concerned with the normality assumption.

### Summary Statistics

Sometimes it's hard to tell whether your data follow the normal distribution by just looking at the graph. Hence, we often have to conduct statistical test to aid our decision. Common tests are

-   [Methods based on normal probability plot]

    -   [Correlation Coefficient with Normal Probability Plots]
    -   [Shapiro-Wilk Test]

-   [Methods based on empirical cumulative distribution function]

    -   [Anderson-Darling Test]
    -   [Kolmogorov-Smirnov Test]
    -   [Cramer-von Mises Test]
    -   [Jarque--Bera Test](#jarquebera-test)

#### Methods based on normal probability plot

##### Correlation Coefficient with Normal Probability Plots

[@Looney_1985] [@Shapiro_1972] The correlation coefficient between $y_{(i)}$ and $m_i^*$ as given on the normal probability plot:

$$W^*=\frac{\sum_{i=1}^{n}(y_{(i)}-\bar{y})(m_i^*-0)}{(\sum_{i=1}^{n}(y_{(i)}-\bar{y})^2\sum_{i=1}^{n}(m_i^*-0)^2)^.5}$$

where $\bar{m^*}=0$

Pearson product moment formula for correlation:

$$\hat{p}=\frac{\sum_{i-1}^{n}(y_i-\bar{y})(x_i-\bar{x})}{(\sum_{i=1}^{n}(y_{i}-\bar{y})^2\sum_{i=1}^{n}(x_i-\bar{x})^2)^.5}$$

-   When the correlation is 1, the plot is exactly linear and normality is assumed.
-   The closer the correlation is to zero, the more confident we are to reject normality
-   Inference on W\* needs to be based on special tables [@Looney_1985]


```r
library("EnvStats")
gofTest(data,test="ppcc")$p.value #Probability Plot Correlation Coefficient 
#> [1] 0.9435573
```

##### Shapiro-Wilk Test

[@Shapiro_1965]

$$W=(\frac{\sum_{i=1}^{n}a_i(y_{(i)}-\bar{y})(m_i^*-0)}{(\sum_{i=1}^{n}a_i^2(y_{(i)}-\bar{y})^2\sum_{i=1}^{n}(m_i^*-0)^2)^.5})^2$$

where $a_1,..,a_n$ are weights computed from the covariance matrix for the order statistics.

-   Researchers typically use this test to assess normality. (n \< 2000) Under normality, W is close to 1, just like $W^*$. Notice that the only difference between W and W\* is the "weights".


```r
gofTest(data,test="sw")$p.value #Shapiro-Wilk is the default.
#> [1] 0.9877556
```

#### Methods based on empirical cumulative distribution function

The formula for the empirical cumulative distribution function (CDF) is:

$F_n(t)$ = estimate of probability that an observation $\le$ t = (number of observation $\le$ t)/n

This method requires large sample sizes. However, it can apply to distributions other than the normal (Gaussian) one.


```r
# Empirical CDF hand-code
plot.ecdf(data,verticals = T, do.points=F)
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-8-1.png" width="90%" style="display: block; margin: auto;" />

##### Anderson-Darling Test

[@Anderson_1952]

The Anderson-Darling statistic:

$$A^2=\int_{-\infty}^{\infty}(F_n(t)=F(t))^2\frac{dF(t)}{F(t)(1-F(t))}$$

-   a weight average of squared deviations (it weights small and large values of t more)

For the normal distribution,

$A^2 = - (\sum_{i=1}^{n}(2i-1)(ln(p_i) +ln(1-p_{n+1-i}))/n-n$

where $p_i=\Phi(\frac{y_{(i)}-\bar{y}}{s})$, the probability that a standard normal variable is less than $\frac{y_{(i)}-\bar{y}}{s}$

-   Reject normal assumption when $A^2$ is too large

-   Evaluate the null hypothesis that the observations are randomly selected from a normal population based on the critical value provided by [@Marsaglia_2004] and [@Stephens_1974]

-   This test can be applied to other distributions:

    -   Exponential
    -   Logistic
    -   Gumbel
    -   Extreme-value
    -   Weibull: log(Weibull) = Gumbel
    -   Gamma
    -   Logistic
    -   Cauchy
    -   von Mises
    -   Log-normal (two-parameter)

Consult [@Stephens_1974] for more detailed transformation and critical values.


```r
gofTest(data,test="ad")$p.value #Anderson-Darling
#> [1] 0.852593
```

##### Kolmogorov-Smirnov Test

-   Based on the largest absolute difference between empirical and expected cumulative distribution
-   Another deviation of K-S test is Kuiper's test


```r
gofTest(data,test="ks")$p.value #Komogorov-Smirnov 
#> [1] 0.7647543
```

##### Cramer-von Mises Test

-   Based on the average squared discrepancy between the empirical distribution and a given theoretical distribution. Each discrepancy is weighted equally (unlike Anderson-Darling test weights end points more heavily)


```r
gofTest(data,test="cvm")$p.value #Cramer-von Mises
#> [1] 0.6645547
```

##### Jarque--Bera Test {#jarquebera-test}

[@Bera_1981]

Based on the skewness and kurtosis to test normality.

$JB = \frac{n}{6}(S^2+(K-3)^2/4)$ where S is the sample skewness and K is the sample kurtosis

$S=\frac{\hat{\mu_3}}{\hat{\sigma}^3}=\frac{\sum_{i=1}^{n}(x_i-\bar{x})^3/n}{(\sum_{i=1}^{n}(x_i-\bar{x})^2/n)^\frac{3}{2}}$

$K=\frac{\hat{\mu_4}}{\hat{\sigma}^4}=\frac{\sum_{i=1}^{n}(x_i-\bar{x})^4/n}{(\sum_{i=1}^{n}(x_i-\bar{x})^2/n)^2}$

recall $\hat{\sigma^2}$ is the estimate of the second central moment (variance) $\hat{\mu_3}$ and $\hat{\mu_4}$ are the estimates of third and fourth central moments.

If the data comes from a normal distribution, the JB statistic asymptotically has a chi-squared distribution with two degrees of freedom.

The null hypothesis is a joint hypothesis of the skewness being zero and the excess kurtosis being zero.

## Bivariate Statistics

Correlation between

-   [Two Continuous] variables
-   [Two Discrete] variables
-   [Categorical and Continuous]

+----------------+------------------------------+------------------------+
|                | Categorical                  | Continuous             |
+================+==============================+========================+
| Categorical    | [Phi coefficient]            |                        |
|                |                              |                        |
|                | [Cramer's V]                 |                        |
|                |                              |                        |
|                | [Tschuprow's T]              |                        |
|                |                              |                        |
|                | [Freeman's Theta]            |                        |
|                |                              |                        |
|                | [Epsilon-squared]            |                        |
|                |                              |                        |
|                | [Goodman Kruskal's Lambda]   |                        |
|                |                              |                        |
|                | [Somers' D]                  |                        |
|                |                              |                        |
|                | [Kendall's Tau-b]            |                        |
|                |                              |                        |
|                | [Yule's Q and Y]             |                        |
|                |                              |                        |
|                | [Tetrachoric Correlation]    |                        |
|                |                              |                        |
|                | [Polychoric Correlation]     |                        |
+----------------+------------------------------+------------------------+
| Continuous     | [Point-Biserial Correlation] | [Pearson Correlation]  |
|                |                              |                        |
|                | [Logistic Regression]        | [Spearman Correlation] |
+----------------+------------------------------+------------------------+

Questions to keep in mind:

1.  Is the relationship linear or non-linear?
2.  If the variable is continuous, is it normal and homoskadastic?
3.  How big is your dataset?

<br>

## Two Continuous


```r
n = 100 # (sample size)

data = data.frame(A = sample(1:20, replace = TRUE, size = n),
                  B = sample(1:30, replace = TRUE, size = n))
```

### Pearson Correlation

-   Good with linear relationship


```r
library(Hmisc)
rcorr(data$A, data$B, type="pearson") 
#>       x     y
#> x  1.00 -0.04
#> y -0.04  1.00
#> 
#> n= 100 
#> 
#> 
#> P
#>   x      y     
#> x        0.7249
#> y 0.7249
```

### Spearman Correlation


```r
library(Hmisc)
rcorr(data$A, data$B, type="spearman") 
#>       x     y
#> x  1.00 -0.03
#> y -0.03  1.00
#> 
#> n= 100 
#> 
#> 
#> P
#>   x      y     
#> x        0.7363
#> y 0.7363
```

## Categorical and Continuous

### Point-Biserial Correlation

Similar to the Pearson correlation coefficient, the point-biserial correlation coefficient is between -1 and 1 where:

-   -1 means a perfectly negative correlation between two variables

-   0 means no correlation between two variables

-   1 means a perfectly positive correlation between two variables


```r
x <- c(0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0)
y <- c(12, 14, 17, 17, 11, 22, 23, 11, 19, 8, 12)

#calculate point-biserial correlation
cor.test(x, y)
#> 
#> 	Pearson's product-moment correlation
#> 
#> data:  x and y
#> t = 0.67064, df = 9, p-value = 0.5193
#> alternative hypothesis: true correlation is not equal to 0
#> 95 percent confidence interval:
#>  -0.4391885  0.7233704
#> sample estimates:
#>       cor 
#> 0.2181635
```

Alternatively


```r
ltm::biserial.cor(y,x, use = c("all.obs"), level = 2)
#> [1] 0.2181635
```

### Logistic Regression

See \@ref(logistic-regression)

## Two Discrete

### Distance Metrics

Some consider distance is not a correlation metric because it isn't unit independent (i.e., if you scale the distance, the metrics will change), but it's still a useful proxy. Distance metrics are more likely to be used for similarity measure.

-   Euclidean Distance

-   Manhattan Distance

-   Chessboard Distance

-   Minkowski Distance

-   Canberra Distance

-   Hamming Distance

-   Cosine Distance

-   Sum of Absolute Distance

-   Sum of Squared Distance

-   Mean-Absolute Error

### Statistical Metrics

#### Chi-squared test

##### Phi coefficient

-   2 binary


```r
dt = matrix(c(1,4,3,5), nrow = 2)
dt
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    4    5
psych::phi(dt)
#> [1] -0.18
```

##### Cramer's V

-   between nominal categorical variables (no natural order)

$$
\text{Cramer's V} = \sqrt{\frac{\chi^2/n}{\min(c-1,r-1)}}
$$

where

-   $\chi^2$ = Chi-square statistic

-   $n$ = sample size

-   $r$ = \# of rows

-   $c$ = \# of columns


```r
library('lsr')
n = 100 # (sample size)
set.seed(1)
data = data.frame(A = sample(1:5, replace = TRUE, size = n),
                  B = sample(1:6, replace = TRUE, size = n))


cramersV(data$A, data$B)
#> [1] 0.1944616
```

Alternatively,

-   `ncchisq` noncentral Chi-square

-   `nchisqadj` Adjusted noncentral Chi-square

-   `fisher` Fisher Z transformation

-   `fisheradj` bias correction Fisher z transformation


```r
DescTools::CramerV(data, conf.level = 0.95,method = "ncchisqadj")
#>  Cramer V    lwr.ci    upr.ci 
#> 0.3472325 0.3929964 0.4033053
```

##### Tschuprow's T

-   2 nominal variables


```r
DescTools::TschuprowT(data)
#> [1] 0.1100808
```

### Ordinal Association (Rank correlation)

-   Good with non-linear relationship

#### Ordinal and Nominal


```r
n = 100 # (sample size)
set.seed(1)
dt = table(data.frame(
    A = sample(1:4, replace = TRUE, size = n), # ordinal
    B = sample(1:3, replace = TRUE, size = n)  # nominal
)) 
dt
#>    B
#> A    1  2  3
#>   1  7 11  9
#>   2 11  6 14
#>   3  7 11  4
#>   4  6  4 10
```

##### Freeman's Theta

-   Ordinal and nominal


```r
# this package is not available for R >= 4.0.0
rcompanion::freemanTheta(dt,group = "column" ) # because column is the grouping variable (i.e., nominal)
```

##### Epsilon-squared

-   Ordinal and nominal


```r
# this package is not available for R >= 4.0.0
rcompanion::epsilonSquared(dt,group = "column" ) # because column is the grouping variable (i.e., nominal)
```

#### Two Ordinal


```r
n = 100 # (sample size)
set.seed(1)
dt = table(data.frame(
    A = sample(1:4, replace = TRUE, size = n), # ordinal
    B = sample(1:3, replace = TRUE, size = n)  # ordinal
)) 
dt
#>    B
#> A    1  2  3
#>   1  7 11  9
#>   2 11  6 14
#>   3  7 11  4
#>   4  6  4 10
```

##### Goodman Kruskal's Gamma

-   2 ordinal variables


```r
DescTools::GoodmanKruskalGamma(dt, conf.level = 0.95)
#>        gamma       lwr.ci       upr.ci 
#>  0.006781013 -0.229032069  0.242594095
```

##### Somers' D

-   or Somers' Delta

-   2 ordinal variables


```r
DescTools::SomersDelta(dt, conf.level = 0.95)
#>       somers       lwr.ci       upr.ci 
#>  0.005115859 -0.172800185  0.183031903
```

##### Kendall's Tau-b

-   2 ordinal variables


```r
DescTools::KendallTauB(dt, conf.level = 0.95)
#>        tau_b       lwr.ci       upr.ci 
#>  0.004839732 -0.163472443  0.173151906
```

##### Yule's Q and Y

-   2 ordinal variables

Special version (2 x 2) of the [Goodman Kruskal's Gamma] coefficient.

|            | Variable 1 |     |
|------------|------------|-----|
| Variable 2 | a          | b   |
|            | c          | d   |

$$
\text{Yule's Q} = \frac{ad - bc}{ad + bc}
$$

We typically use Yule's Q in practice while Yule's Y has the following relationship with Q.

$$
\text{Yule's Y} = \frac{\sqrt{ad} - \sqrt{bc}}{\sqrt{ad} + \sqrt{bc}}
$$

$$
Q = \frac{2Y}{1 + Y^2}
$$

$$
Y = \frac{1 = \sqrt{1-Q^2}}{Q}
$$


```r
n = 100 # (sample size)
set.seed(1)
dt = table(data.frame(A = sample(c(0, 1), replace = TRUE, size = n),
                  B = sample(c(0, 1), replace = TRUE, size = n)))
dt
#>    B
#> A    0  1
#>   0 25 24
#>   1 28 23
DescTools::YuleQ(dt)
#> [1] -0.07778669
```

##### Tetrachoric Correlation

-   is a special case of [Polychoric Correlation] when both variables are binary


```r
library(psych)

n = 100 # (sample size)

data = data.frame(A = sample(c(0, 1), replace = TRUE, size = n),
                  B = sample(c(0, 1), replace = TRUE, size = n))

#view table
head(data)
#>   A B
#> 1 1 0
#> 2 1 0
#> 3 0 0
#> 4 1 0
#> 5 1 0
#> 6 1 0
table(data)
#>    B
#> A    0  1
#>   0 21 23
#>   1 34 22
#calculate tetrachoric correlation
tetrachoric(data)
#> Call: tetrachoric(x = data)
#> tetrachoric correlation 
#>   A    B   
#> A  1.0     
#> B -0.2  1.0
#> 
#>  with tau of 
#>     A     B 
#> -0.15  0.13
```

##### Polychoric Correlation

-   between ordinal categorical variables (natural order).
-   Assumption: Ordinal variable is a discrete representation of a latent normally distributed continuous variable. (Income = low, normal, high).


```r
library(polycor)

n = 100 # (sample size)

data = data.frame(A = sample(1:4, replace = TRUE, size = n),
                  B = sample(1:6, replace = TRUE, size = n))

head(data)
#>   A B
#> 1 1 3
#> 2 1 1
#> 3 3 5
#> 4 2 3
#> 5 3 5
#> 6 4 4
#calculate polychoric correlation between ratings
polychor(data$A, data$B)
#> [1] 0.01607982
```

### Summary


```r
library(tidyverse)

data("mtcars")
df = mtcars 

df_factor = df %>% 
    mutate(cyl = factor(cyl), 
           vs = factor(vs), 
           am = factor(am), 
           gear = factor(gear), 
           carb = factor(carb))
# summary(df)
str(df)
#> 'data.frame':	32 obs. of  11 variables:
#>  $ mpg : num  21 21 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2 ...
#>  $ cyl : num  6 6 4 6 8 6 8 4 4 6 ...
#>  $ disp: num  160 160 108 258 360 ...
#>  $ hp  : num  110 110 93 110 175 105 245 62 95 123 ...
#>  $ drat: num  3.9 3.9 3.85 3.08 3.15 2.76 3.21 3.69 3.92 3.92 ...
#>  $ wt  : num  2.62 2.88 2.32 3.21 3.44 ...
#>  $ qsec: num  16.5 17 18.6 19.4 17 ...
#>  $ vs  : num  0 0 1 1 0 1 0 1 1 1 ...
#>  $ am  : num  1 1 1 0 0 0 0 0 0 0 ...
#>  $ gear: num  4 4 4 3 3 3 3 4 4 4 ...
#>  $ carb: num  4 4 1 1 2 1 4 2 2 4 ...
str(df_factor)
#> 'data.frame':	32 obs. of  11 variables:
#>  $ mpg : num  21 21 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2 ...
#>  $ cyl : Factor w/ 3 levels "4","6","8": 2 2 1 2 3 2 3 1 1 2 ...
#>  $ disp: num  160 160 108 258 360 ...
#>  $ hp  : num  110 110 93 110 175 105 245 62 95 123 ...
#>  $ drat: num  3.9 3.9 3.85 3.08 3.15 2.76 3.21 3.69 3.92 3.92 ...
#>  $ wt  : num  2.62 2.88 2.32 3.21 3.44 ...
#>  $ qsec: num  16.5 17 18.6 19.4 17 ...
#>  $ vs  : Factor w/ 2 levels "0","1": 1 1 2 2 1 2 1 2 2 2 ...
#>  $ am  : Factor w/ 2 levels "0","1": 2 2 2 1 1 1 1 1 1 1 ...
#>  $ gear: Factor w/ 3 levels "3","4","5": 2 2 2 1 1 1 1 2 2 2 ...
#>  $ carb: Factor w/ 6 levels "1","2","3","4",..: 4 4 1 1 2 1 4 2 2 4 ...
```

Get the correlation table for continuous variables only


```r
cor(df)
#>             mpg        cyl       disp         hp        drat         wt
#> mpg   1.0000000 -0.8521620 -0.8475514 -0.7761684  0.68117191 -0.8676594
#> cyl  -0.8521620  1.0000000  0.9020329  0.8324475 -0.69993811  0.7824958
#> disp -0.8475514  0.9020329  1.0000000  0.7909486 -0.71021393  0.8879799
#> hp   -0.7761684  0.8324475  0.7909486  1.0000000 -0.44875912  0.6587479
#> drat  0.6811719 -0.6999381 -0.7102139 -0.4487591  1.00000000 -0.7124406
#> wt   -0.8676594  0.7824958  0.8879799  0.6587479 -0.71244065  1.0000000
#> qsec  0.4186840 -0.5912421 -0.4336979 -0.7082234  0.09120476 -0.1747159
#> vs    0.6640389 -0.8108118 -0.7104159 -0.7230967  0.44027846 -0.5549157
#> am    0.5998324 -0.5226070 -0.5912270 -0.2432043  0.71271113 -0.6924953
#> gear  0.4802848 -0.4926866 -0.5555692 -0.1257043  0.69961013 -0.5832870
#> carb -0.5509251  0.5269883  0.3949769  0.7498125 -0.09078980  0.4276059
#>             qsec         vs          am       gear        carb
#> mpg   0.41868403  0.6640389  0.59983243  0.4802848 -0.55092507
#> cyl  -0.59124207 -0.8108118 -0.52260705 -0.4926866  0.52698829
#> disp -0.43369788 -0.7104159 -0.59122704 -0.5555692  0.39497686
#> hp   -0.70822339 -0.7230967 -0.24320426 -0.1257043  0.74981247
#> drat  0.09120476  0.4402785  0.71271113  0.6996101 -0.09078980
#> wt   -0.17471588 -0.5549157 -0.69249526 -0.5832870  0.42760594
#> qsec  1.00000000  0.7445354 -0.22986086 -0.2126822 -0.65624923
#> vs    0.74453544  1.0000000  0.16834512  0.2060233 -0.56960714
#> am   -0.22986086  0.1683451  1.00000000  0.7940588  0.05753435
#> gear -0.21268223  0.2060233  0.79405876  1.0000000  0.27407284
#> carb -0.65624923 -0.5696071  0.05753435  0.2740728  1.00000000
# only complete obs
# cor(df, use = "complete.obs")
```

Alternatively, you can also have the


```r
Hmisc::rcorr(as.matrix(df), type = "pearson")
#>        mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
#> mpg   1.00 -0.85 -0.85 -0.78  0.68 -0.87  0.42  0.66  0.60  0.48 -0.55
#> cyl  -0.85  1.00  0.90  0.83 -0.70  0.78 -0.59 -0.81 -0.52 -0.49  0.53
#> disp -0.85  0.90  1.00  0.79 -0.71  0.89 -0.43 -0.71 -0.59 -0.56  0.39
#> hp   -0.78  0.83  0.79  1.00 -0.45  0.66 -0.71 -0.72 -0.24 -0.13  0.75
#> drat  0.68 -0.70 -0.71 -0.45  1.00 -0.71  0.09  0.44  0.71  0.70 -0.09
#> wt   -0.87  0.78  0.89  0.66 -0.71  1.00 -0.17 -0.55 -0.69 -0.58  0.43
#> qsec  0.42 -0.59 -0.43 -0.71  0.09 -0.17  1.00  0.74 -0.23 -0.21 -0.66
#> vs    0.66 -0.81 -0.71 -0.72  0.44 -0.55  0.74  1.00  0.17  0.21 -0.57
#> am    0.60 -0.52 -0.59 -0.24  0.71 -0.69 -0.23  0.17  1.00  0.79  0.06
#> gear  0.48 -0.49 -0.56 -0.13  0.70 -0.58 -0.21  0.21  0.79  1.00  0.27
#> carb -0.55  0.53  0.39  0.75 -0.09  0.43 -0.66 -0.57  0.06  0.27  1.00
#> 
#> n= 32 
#> 
#> 
#> P
#>      mpg    cyl    disp   hp     drat   wt     qsec   vs     am     gear  
#> mpg         0.0000 0.0000 0.0000 0.0000 0.0000 0.0171 0.0000 0.0003 0.0054
#> cyl  0.0000        0.0000 0.0000 0.0000 0.0000 0.0004 0.0000 0.0022 0.0042
#> disp 0.0000 0.0000        0.0000 0.0000 0.0000 0.0131 0.0000 0.0004 0.0010
#> hp   0.0000 0.0000 0.0000        0.0100 0.0000 0.0000 0.0000 0.1798 0.4930
#> drat 0.0000 0.0000 0.0000 0.0100        0.0000 0.6196 0.0117 0.0000 0.0000
#> wt   0.0000 0.0000 0.0000 0.0000 0.0000        0.3389 0.0010 0.0000 0.0005
#> qsec 0.0171 0.0004 0.0131 0.0000 0.6196 0.3389        0.0000 0.2057 0.2425
#> vs   0.0000 0.0000 0.0000 0.0000 0.0117 0.0010 0.0000        0.3570 0.2579
#> am   0.0003 0.0022 0.0004 0.1798 0.0000 0.0000 0.2057 0.3570        0.0000
#> gear 0.0054 0.0042 0.0010 0.4930 0.0000 0.0005 0.2425 0.2579 0.0000       
#> carb 0.0011 0.0019 0.0253 0.0000 0.6212 0.0146 0.0000 0.0007 0.7545 0.1290
#>      carb  
#> mpg  0.0011
#> cyl  0.0019
#> disp 0.0253
#> hp   0.0000
#> drat 0.6212
#> wt   0.0146
#> qsec 0.0000
#> vs   0.0007
#> am   0.7545
#> gear 0.1290
#> carb
```


```r
modelsummary::datasummary_correlation(df)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> mpg </th>
   <th style="text-align:right;"> cyl </th>
   <th style="text-align:right;"> disp </th>
   <th style="text-align:right;"> hp </th>
   <th style="text-align:right;"> drat </th>
   <th style="text-align:right;"> wt </th>
   <th style="text-align:right;"> qsec </th>
   <th style="text-align:right;"> vs </th>
   <th style="text-align:right;"> am </th>
   <th style="text-align:right;"> gear </th>
   <th style="text-align:right;"> carb </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> mpg </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
  </tr>
  <tr>
   <td style="text-align:left;"> cyl </td>
   <td style="text-align:right;"> -.85 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
  </tr>
  <tr>
   <td style="text-align:left;"> disp </td>
   <td style="text-align:right;"> -.85 </td>
   <td style="text-align:right;"> .90 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
  </tr>
  <tr>
   <td style="text-align:left;"> hp </td>
   <td style="text-align:right;"> -.78 </td>
   <td style="text-align:right;"> .83 </td>
   <td style="text-align:right;"> .79 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
  </tr>
  <tr>
   <td style="text-align:left;"> drat </td>
   <td style="text-align:right;"> .68 </td>
   <td style="text-align:right;"> -.70 </td>
   <td style="text-align:right;"> -.71 </td>
   <td style="text-align:right;"> -.45 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
  </tr>
  <tr>
   <td style="text-align:left;"> wt </td>
   <td style="text-align:right;"> -.87 </td>
   <td style="text-align:right;"> .78 </td>
   <td style="text-align:right;"> .89 </td>
   <td style="text-align:right;"> .66 </td>
   <td style="text-align:right;"> -.71 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
  </tr>
  <tr>
   <td style="text-align:left;"> qsec </td>
   <td style="text-align:right;"> .42 </td>
   <td style="text-align:right;"> -.59 </td>
   <td style="text-align:right;"> -.43 </td>
   <td style="text-align:right;"> -.71 </td>
   <td style="text-align:right;"> .09 </td>
   <td style="text-align:right;"> -.17 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
  </tr>
  <tr>
   <td style="text-align:left;"> vs </td>
   <td style="text-align:right;"> .66 </td>
   <td style="text-align:right;"> -.81 </td>
   <td style="text-align:right;"> -.71 </td>
   <td style="text-align:right;"> -.72 </td>
   <td style="text-align:right;"> .44 </td>
   <td style="text-align:right;"> -.55 </td>
   <td style="text-align:right;"> .74 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
  </tr>
  <tr>
   <td style="text-align:left;"> am </td>
   <td style="text-align:right;"> .60 </td>
   <td style="text-align:right;"> -.52 </td>
   <td style="text-align:right;"> -.59 </td>
   <td style="text-align:right;"> -.24 </td>
   <td style="text-align:right;"> .71 </td>
   <td style="text-align:right;"> -.69 </td>
   <td style="text-align:right;"> -.23 </td>
   <td style="text-align:right;"> .17 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> . </td>
   <td style="text-align:right;"> . </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gear </td>
   <td style="text-align:right;"> .48 </td>
   <td style="text-align:right;"> -.49 </td>
   <td style="text-align:right;"> -.56 </td>
   <td style="text-align:right;"> -.13 </td>
   <td style="text-align:right;"> .70 </td>
   <td style="text-align:right;"> -.58 </td>
   <td style="text-align:right;"> -.21 </td>
   <td style="text-align:right;"> .21 </td>
   <td style="text-align:right;"> .79 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> . </td>
  </tr>
  <tr>
   <td style="text-align:left;"> carb </td>
   <td style="text-align:right;"> -.55 </td>
   <td style="text-align:right;"> .53 </td>
   <td style="text-align:right;"> .39 </td>
   <td style="text-align:right;"> .75 </td>
   <td style="text-align:right;"> -.09 </td>
   <td style="text-align:right;"> .43 </td>
   <td style="text-align:right;"> -.66 </td>
   <td style="text-align:right;"> -.57 </td>
   <td style="text-align:right;"> .06 </td>
   <td style="text-align:right;"> .27 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
</tbody>
</table>


```r
ggcorrplot::ggcorrplot(cor(df))
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-35-1.png" width="90%" style="display: block; margin: auto;" />

<br>

Different comparison between different correlation between different types of variables (i.e., continuous vs. categorical) can be problematic. Moreover, the problem of detecting non-linear vs. linear relationship/correlatiton is another one. Hence, a solution is that using mutual information from information theory (i.e., knowing one variable can reduce uncertainty about the other).

To implement mutual information, we have the following approximations

$$
\downarrow \text{prediction error} \approx \downarrow \text{uncertainty} \approx \downarrow \text{association strength}
$$

More specificlly, following the [X2Y metric](https://rviews.rstudio.com/2021/04/15/an-alternative-to-the-correlation-coefficient-that-works-for-numeric-and-categorical-variables/), we have the following steps:

1.  Predict $y$ without $x$ (i.e., baseline model)

    1.  Averge of $y$ when $y$ is continuous

    2.  Most frequent value when $y$ is categorical

2.  Predict $y$ with $x$ (e.g., linear, random forest, etc.)

3.  Calculate the prediction error difference between 1 and 2

To have a comprehensive table that could handle

-   continuous vs. continuous

-   categorical vs. continuous

-   continuous vs. categorical

-   categorical vs. categorical

the suggested model would be Classification and Regression Trees (CART). But we can certainly use other models as well.

The downfall of this method is that you might suffer

1.  Symmetry: $(x,y) \neq (y,x)$
2.  Comparability : Different pair of comparison might use different metrics (e.g., misclassification error vs. MAE)


```r
library(ppsr)
# ppsr::score_df(iris) # if you want a dataframe
ppsr::score_matrix(iris, do_parallel = TRUE, n_cores = parallel::detectCores()/2 ) # if you want a similar correlation matrix
#>              Sepal.Length Sepal.Width Petal.Length Petal.Width   Species
#> Sepal.Length   1.00000000  0.04632352    0.5491398   0.4127668 0.4075487
#> Sepal.Width    0.06790301  1.00000000    0.2376991   0.2174659 0.2012876
#> Petal.Length   0.61608360  0.24263851    1.0000000   0.7917512 0.7904907
#> Petal.Width    0.48735314  0.20124105    0.7437845   1.0000000 0.7561113
#> Species        0.55918638  0.31344008    0.9167580   0.9398532 1.0000000
ppsr::score_matrix(df, do_parallel = TRUE, n_cores = parallel::detectCores()/2 )
#>            mpg        cyl       disp          hp       drat         wt
#> mpg  1.0000000 0.32362397 0.25436628 0.210509478 0.20883649 0.24609235
#> cyl  0.3861810 1.00000000 0.57917897 0.537257954 0.33458867 0.38789293
#> disp 0.3141056 0.54883158 1.00000000 0.485122916 0.35317905 0.23669306
#> hp   0.2311418 0.37853515 0.35542647 1.000000000 0.24544714 0.12721154
#> drat 0.1646116 0.19540490 0.38730966 0.165537791 1.00000000 0.35076928
#> wt   0.2075760 0.11113261 0.20447239 0.155585827 0.12978458 1.00000000
#> qsec 0.1521642 0.10498746 0.07192679 0.134441221 0.08171630 0.05880165
#> vs   0.2000000 0.02514286 0.02514286 0.025142857 0.06862112 0.02514286
#> am   0.0615873 0.12825397 0.24409373 0.004444444 0.30608113 0.17742706
#> gear 0.1785968 0.43293014 0.43554416 0.154438566 0.54542788 0.31526071
#> carb 0.3472565 0.30798148 0.21228704 0.151523221 0.01103355 0.15957673
#>            qsec         vs         am      gear       carb
#> mpg  0.11030342 0.17957228 0.13297202 0.1752449 0.25426760
#> cyl  0.32753721 0.39827893 0.13263224 0.2877488 0.20925329
#> disp 0.31714642 0.35324790 0.23897094 0.4231630 0.15461337
#> hp   0.33941571 0.37794795 0.03821570 0.2159412 0.24105326
#> drat 0.16134068 0.17783324 0.30379298 0.4475122 0.03137800
#> wt   0.09367580 0.12214824 0.24118900 0.1590473 0.14181111
#> qsec 1.00000000 0.24973489 0.02334953 0.0000000 0.07539415
#> vs   0.40000000 1.00000000 0.10000000 0.1251429 0.20000000
#> am   0.15936508 0.11250000 1.00000000 0.3972789 0.00000000
#> gear 0.04791667 0.08012053 0.30341155 1.0000000 0.01486068
#> carb 0.21944241 0.25373093 0.00000000 0.0000000 1.00000000
```

### Visualization


```r
corrplot::corrplot(cor(df))
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-37-1.png" width="90%" style="display: block; margin: auto;" />

Alternatively,


```r
PerformanceAnalytics::chart.Correlation(df, histogram = T, pch =19)
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-38-1.png" width="90%" style="display: block; margin: auto;" />


```r
heatmap(as.matrix(df))
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-39-1.png" width="90%" style="display: block; margin: auto;" />

More general form,


```r
ppsr::visualize_pps(df = iris, do_parallel = TRUE, n_cores = parallel::detectCores()/2 )
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-40-1.png" width="90%" style="display: block; margin: auto;" />


```r
ppsr::visualize_correlations(
    df = iris
)
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-41-1.png" width="90%" style="display: block; margin: auto;" />

Both heatmap and correlation at the same time


```r
ppsr::visualize_both(
    df = iris,
    do_parallel = TRUE,
    n_cores = parallel::detectCores() / 2
)
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-42-1.png" width="90%" style="display: block; margin: auto;" />

More elaboration with `ggplot2`


```r
ppsr::visualize_pps(df = iris,
                    color_value_high = 'red', 
                    color_value_low = 'yellow',
                    color_text = 'black') +
  ggplot2::theme_classic() +
  ggplot2::theme(plot.background = ggplot2::element_rect(fill = "lightgrey")) +
  ggplot2::theme(title = ggplot2::element_text(size = 15)) +
  ggplot2::labs(title = 'Correlation aand Heatmap', 
                subtitle = 'Subtitle',
                caption = 'Caption',
                x = 'More info')
```

<img src="03-descriptive-stat_files/figure-html/unnamed-chunk-43-1.png" width="90%" style="display: block; margin: auto;" />
