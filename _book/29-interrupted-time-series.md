# Interrupted Time Series

-   Regression Discontinuity in Time

-   Control for

    -   Seasonable trends

    -   Concurrent events

-   Pros [@penfold2013use]

    -   control for long-term trends

-   Cons

    -   Min of 8 data points before and 8 after an intervention

    -   Multiple events hard to distinguish

Notes:

-   For subgroup analysis (heterogeneity in effect size), see [@harper2017did]
-   To interpret with control variables, see [@bottomley2019analysing]

Interrupted time series should be used when

1.  longitudinal data (outcome over time - observations before and after the intervention)
2.  full population was affected at one specific point in time (or can be stacked based on intervention)

In each ITS framework, there can be 4 possible scenarios of outcome after an intervention

-   No effects

-   Immediate effect

-   Sustained (long-term) effect (smooth)

-   Both immediate and sustained effect

$$
Y = \beta_0 + \beta_1 T + \beta_2 D + \beta_3 P + \epsilon
$$

where

-   $Y$ is the outcome variable

    -   $\beta_0$ is the baseline level of the outcome

-   $T$ is the time variable (e.g., days, weeks, etc.) passed from the start of the observation period

    -   $\beta_1$ is the slope of the line before the intervention

-   $D$ is the treatment variable where $1$ is after the intervention and $0$ is before the intervention.

    -   $\beta_2$ is the **immediate effect** after the intervention

-   $P$ is the time variable indicating time passed since the intervention (before the intervention, the value is set to 0) (to examine the sustained effect).

    -   $\beta_3$ is the **sustained effect** = difference between the slope of the line prior to the intervention and the slope of the line subsequent to the intervention

**Example**

Create a fictitious dataset where we know the true data generating process

$$
Outcome = 10 * time + 20 * treatment + 25 * timesincetreatment + noise
$$


```r
# number of days
n = 365

# intervention at day 
interven = 200

# time index from 1 to 365
time = c(1:n)

# treatment variable: before internvation = day 1 to 200, after intervention = day 201 to 365
treatment = c(rep(0, interven), rep(1, n - interven))

# time since treatment
timesincetreat = c(rep(0, interven), c(1:(n - interven)))

# outcome 
outcome = 10 + 15 * time + 20 * treatment + 25 * timesincetreat + rnorm(n, mean = 0, sd = 1)

df = data.frame(outcome, time, treatment, timesincetreat)

head(df, 10)
#>      outcome time treatment timesincetreat
#> 1   25.15403    1         0              0
#> 2   39.54239    2         0              0
#> 3   55.02046    3         0              0
#> 4   70.21851    4         0              0
#> 5   84.69178    5         0              0
#> 6   98.63307    6         0              0
#> 7  114.52445    7         0              0
#> 8  129.47930    8         0              0
#> 9  143.88703    9         0              0
#> 10 160.59054   10         0              0
```

Visualize


```r
plot(df$time, df$outcome)

# intervention date
abline(v = interven, col = "blue")

# regression line
ts <- lm(outcome ~ time + treatment + timesincetreat, data = df)
lines(df$time, ts$fitted.values, col = "red")
```

<img src="29-interrupted-time-series_files/figure-html/unnamed-chunk-2-1.png" width="90%" style="display: block; margin: auto;" />


```r
summary(ts)
#> 
#> Call:
#> lm(formula = outcome ~ time + treatment + timesincetreat, data = df)
#> 
#> Residuals:
#>      Min       1Q   Median       3Q      Max 
#> -2.83371 -0.58288 -0.02394  0.58471  2.58488 
#> 
#> Coefficients:
#>                 Estimate Std. Error  t value Pr(>|t|)    
#> (Intercept)     9.842786   0.140878    69.87   <2e-16 ***
#> time           15.001545   0.001215 12342.02   <2e-16 ***
#> treatment      20.034928   0.208917    95.90   <2e-16 ***
#> timesincetreat 24.997934   0.002027 12332.76   <2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 0.9924 on 361 degrees of freedom
#> Multiple R-squared:      1,	Adjusted R-squared:      1 
#> F-statistic: 9.681e+08 on 3 and 361 DF,  p-value: < 2.2e-16
```

Interpretation

-   Time coefficient shows before-intervention outcome trend. Positive and significant, indicating a rising trend. Every day adds 15 points.

-   The treatment coefficient shows the **immediate** increase in outcome. **Immediate effect** is positive and significant, increasing outcome by 20 points.

-   The time since treatment coefficient reflects a change in trend subsequent to the intervention. The **sustained effect** is positive and statistically significant, showing that the outcome increases by 25 points per day after the intervention.

See @lee2014graphical for suggestions

Plot of counterfactual


```r
# treatment prediction
pred <- predict(ts, df)

# counterfactual dataset
new_df <-
    as.data.frame(cbind(
        time = time,
        # treatment = 0 means counterfactual
        treatment = rep(0, n),
        # time since treatment = 0 means counterfactual
        timesincetreat = rep(0)
    ))

# counterfactual predictions
pred_cf <- predict(ts, new_df)

# plot
plot(
    outcome,
    col = gray(0.2, 0.2),
    pch = 19,
    xlim  = c(1,365),
    ylim = c(0, 10000),
    xlab = "xlab",
    ylab = "ylab"
)

# regression line before treatment
lines(rep(1:interven), pred[1:interven], col = "blue", lwd = 3)

# regression line after treatment
lines(rep((interven+1):n), pred[(interven + 1):n], col = "blue", lwd = 3)

# regression line after treatment (counterfactual)
lines(rep(interven:n), pred_cf[(interven): n], col = "yellow", lwd = 3, lty = 5)

abline(v = interven, col = "red", lty = 2)
```

<img src="29-interrupted-time-series_files/figure-html/unnamed-chunk-4-1.png" width="90%" style="display: block; margin: auto;" />

Possible threats to the validity of interrupted time series analysis [@baicker2019testing]

-   Delayed effects [@rodgers2005did] (may have to make assess some time after the intervention - do not assess the immediate dates).

-   Other confounding events [@linden2016using, @linden2017comprehensive]

-   Intervention is introduced but later withdrawn [@linden2015conducting]

-   [Autocorrelation] (for every time series data): might cause underestimation in the standard errors (i.e., overestimating the statistical significance of the treatment effect)

-   Regression to the mean: after a the short-term shock to the outcome, individuals can revert back to their initial states.

-   Selection bias: only certain individuals are affected by the treatment (could use a [Multiple Groups]).

## Autocorrelation

Assess autocorrelation from residual


```r
# simple regression on time 
simple_ts <- lm(outcome ~ time, data = df)

plot(resid(simple_ts))
```

<img src="29-interrupted-time-series_files/figure-html/unnamed-chunk-5-1.png" width="90%" style="display: block; margin: auto;" />

```r

# alternatively
acf(resid(simple_ts))
```

<img src="29-interrupted-time-series_files/figure-html/unnamed-chunk-5-2.png" width="90%" style="display: block; margin: auto;" />

This is not the best example since I created this dataset. But when residuals do have autocorrelation, you should not see any patterns (i.e., points should be randomly distributed on the plot)

To formally test for autocorrelation, we can use the Durbin-Watson test


```r
lmtest::dwtest(df$outcome ~ df$time)
#> 
#> 	Durbin-Watson test
#> 
#> data:  df$outcome ~ df$time
#> DW = 0.00037521, p-value < 2.2e-16
#> alternative hypothesis: true autocorrelation is greater than 0
```

From the p-value, we know that there is autocorrelation in the time series

A solution to this problem is to use more advanced time series analysis (e.g., ARIMA - coming up in the book) to adjust for seasonality and other dependency.


```r
forecast::auto.arima(df$outcome, xreg = as.matrix(df[, -1]))
#> Series: df$outcome 
#> Regression with ARIMA(0,0,0) errors 
#> 
#> Coefficients:
#>       intercept     time  treatment  timesincetreat
#>          9.8428  15.0015    20.0349         24.9979
#> s.e.     0.1401   0.0012     0.2078          0.0020
#> 
#> sigma^2 = 0.9849:  log likelihood = -513.13
#> AIC=1036.25   AICc=1036.42   BIC=1055.75
```

## Multiple Groups

When you suspect that you might have confounding events or selection bias, you can add a control group that did not experience the treatment (very much similar to [Difference-in-differences])

The model then becomes

$$
Y = \beta_0 + \beta_1 time+ \beta_2 treatment +\beta_3 * timesincetreat + \\
\beta_4 group + \beta_5 group * time + \beta_6 group * treatment + \beta_7 group * timesincetreat
$$

where

-   Group = 1 when the observation is under treatment and 0 under control

-   $\beta_4$ = baseline difference between the treatment and control group

-   $\beta_5$ = slope difference between the treatment and control group before treatment

-   $\beta_6$ = baseline difference between the treatment and control group associated with the treatment.

-   $\beta_7$ = difference between the sustained effect of the treatment and control group after the treatment.
