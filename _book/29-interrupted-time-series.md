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
Outcome = 10 \times time + 20 \times treatment + 25 \times timesincetreatment + noise
$$


```r
# number of days
n = 365


# intervention at day
interven = 200

# time index from 1 to 365
time = c(1:n)

# treatment variable: before internvation = day 1 to 200, 
# after intervention = day 201 to 365
treatment = c(rep(0, interven), rep(1, n - interven))

# time since treatment
timesincetreat = c(rep(0, interven), c(1:(n - interven)))

# outcome
outcome = 10 + 15 * time + 20 * treatment + 
    25 * timesincetreat + rnorm(n, mean = 0, sd = 1)

df = data.frame(outcome, time, treatment, timesincetreat)

head(df, 10)
#>      outcome time treatment timesincetreat
#> 1   25.27547    1         0              0
#> 2   38.20899    2         0              0
#> 3   54.99056    3         0              0
#> 4   70.24955    4         0              0
#> 5   85.03771    5         0              0
#> 6   99.96599    6         0              0
#> 7  114.12969    7         0              0
#> 8  130.88334    8         0              0
#> 9  145.49623    9         0              0
#> 10 160.08668   10         0              0
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
#> -2.58259 -0.67261 -0.06189  0.64334  2.87399 
#> 
#> Coefficients:
#>                 Estimate Std. Error  t value Pr(>|t|)    
#> (Intercept)    10.103786   0.140627    71.85   <2e-16 ***
#> time           14.998537   0.001213 12361.64   <2e-16 ***
#> treatment      20.268569   0.208543    97.19   <2e-16 ***
#> timesincetreat 25.001552   0.002023 12356.64   <2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 0.9907 on 361 degrees of freedom
#> Multiple R-squared:      1,	Adjusted R-squared:      1 
#> F-statistic: 9.715e+08 on 3 and 361 DF,  p-value: < 2.2e-16
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
lines(rep((interven + 1):n), pred[(interven + 1):n], 
      col = "blue", lwd = 3)

# regression line after treatment (counterfactual)
lines(
    rep(interven:n),
    pred_cf[(interven):n],
    col = "yellow",
    lwd = 3,
    lty = 5
)

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
#> DW = 0.00037603, p-value < 2.2e-16
#> alternative hypothesis: true autocorrelation is greater than 0
```

From the p-value, we know that there is autocorrelation in the time series

A solution to this problem is to use more advanced time series analysis (e.g., ARIMA - coming up in the book) to adjust for seasonality and other dependency.


```r
forecast::auto.arima(df$outcome, xreg = as.matrix(df[,-1]))
#> Series: df$outcome 
#> Regression with ARIMA(2,0,0) errors 
#> 
#> Coefficients:
#>          ar1      ar2  intercept     time  treatment  timesincetreat
#>       0.0791  -0.1493    10.1103  14.9985    20.2737         25.0016
#> s.e.  0.0517   0.0522     0.1293   0.0011     0.1921          0.0019
#> 
#> sigma^2 = 0.9605:  log likelihood = -507.56
#> AIC=1029.12   AICc=1029.44   BIC=1056.42
```

## Multiple Groups

When you suspect that you might have confounding events or selection bias, you can add a control group that did not experience the treatment (very much similar to [Difference-in-differences])

The model then becomes

$$
\begin{aligned}
Y = \beta_0 &+ \beta_1 time+ \beta_2 treatment +\beta_3 \times timesincetreat \\
&+\beta_4 group + \beta_5 group \times time + \beta_6 group \times treatment \\
&+ \beta_7 group \times timesincetreat
\end{aligned}
$$

where

-   Group = 1 when the observation is under treatment and 0 under control

-   $\beta_4$ = baseline difference between the treatment and control group

-   $\beta_5$ = slope difference between the treatment and control group before treatment

-   $\beta_6$ = baseline difference between the treatment and control group associated with the treatment.

-   $\beta_7$ = difference between the sustained effect of the treatment and control group after the treatment.
