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
#> 1   25.41695    1         0              0
#> 2   38.97229    2         0              0
#> 3   57.51737    3         0              0
#> 4   70.46006    4         0              0
#> 5   84.99633    5         0              0
#> 6   98.73584    6         0              0
#> 7  115.66507    7         0              0
#> 8  129.42565    8         0              0
#> 9  144.81652    9         0              0
#> 10 159.49916   10         0              0
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



\begin{center}\includegraphics[width=0.9\linewidth]{31-interrupted-time-series_files/figure-latex/unnamed-chunk-2-1} \end{center}


```r
summary(ts)
#> 
#> Call:
#> lm(formula = outcome ~ time + treatment + timesincetreat, data = df)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -3.8159 -0.6665  0.0125  0.6887  2.7439 
#> 
#> Coefficients:
#>                 Estimate Std. Error  t value Pr(>|t|)    
#> (Intercept)     9.765354   0.146870    66.49   <2e-16 ***
#> time           15.002711   0.001267 11839.43   <2e-16 ***
#> treatment      19.649970   0.217803    90.22   <2e-16 ***
#> timesincetreat 24.997691   0.002113 11829.51   <2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 1.035 on 361 degrees of freedom
#> Multiple R-squared:      1,	Adjusted R-squared:      1 
#> F-statistic: 8.907e+08 on 3 and 361 DF,  p-value: < 2.2e-16
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



\begin{center}\includegraphics[width=0.9\linewidth]{31-interrupted-time-series_files/figure-latex/unnamed-chunk-4-1} \end{center}

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



\begin{center}\includegraphics[width=0.9\linewidth]{31-interrupted-time-series_files/figure-latex/unnamed-chunk-5-1} \end{center}

```r

# alternatively
acf(resid(simple_ts))
```



\begin{center}\includegraphics[width=0.9\linewidth]{31-interrupted-time-series_files/figure-latex/unnamed-chunk-5-2} \end{center}

This is not the best example since I created this dataset. But when residuals do have autocorrelation, you should not see any patterns (i.e., points should be randomly distributed on the plot)

To formally test for autocorrelation, we can use the Durbin-Watson test


```r
lmtest::dwtest(df$outcome ~ df$time)
#> 
#> 	Durbin-Watson test
#> 
#> data:  df$outcome ~ df$time
#> DW = 0.00037693, p-value < 2.2e-16
#> alternative hypothesis: true autocorrelation is greater than 0
```

From the p-value, we know that there is autocorrelation in the time series

A solution to this problem is to use more advanced time series analysis (e.g., ARIMA - coming up in the book) to adjust for seasonality and other dependency.


```r
forecast::auto.arima(df$outcome, xreg = as.matrix(df[,-1]))
#> Series: df$outcome 
#> Regression with ARIMA(0,0,0) errors 
#> 
#> Coefficients:
#>       intercept     time  treatment  timesincetreat
#>          9.7654  15.0027    19.6500         24.9977
#> s.e.     0.1461   0.0013     0.2166          0.0021
#> 
#> sigma^2 = 1.07:  log likelihood = -528.33
#> AIC=1066.66   AICc=1066.83   BIC=1086.16
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

## Recurrent Events

@zhang2023causal
