# Interrupted Time Series

-   Regression Discontinuity in Time

-   Control for

    -   Seasonable trends

    -   Concurrent events

-   Pros [@Penfold_2013]

    -   control for long-term trends

-   Cons

    -   Min of 8 data points before and 8 after an intervention

    -   Multiple events hard to distinguish

Notes:

-   For subgroup analysis (heterogeneity in effect size), see [@harper2017]

Example by [Leihua Ye](https://towardsdatascience.com/what-is-the-strongest-quasi-experimental-method-interrupted-time-series-period-f59fe5b00b31)


```r
# data preparation
set.seed(1)
CaseID = rep(1:100, 6)

# intervention
Intervention = c(rep(0, 300), rep(1, 300))
Outcome_Variable = c(rnorm(300), abs(rnorm(300) * 4))

mydata = cbind(CaseID, Intervention, Outcome_Variable)

mydata = as.data.frame(mydata)

#construct a simple OLS model
model = lm(Outcome_Variable ~ Intervention, data = mydata)
summary(model) # there is a signficant effect 
#> 
#> Call:
#> lm(formula = Outcome_Variable ~ Intervention, data = mydata)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -3.3050 -1.2315 -0.1734  0.8691 11.9185 
#> 
#> Coefficients:
#>              Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)   0.03358    0.11021   0.305    0.761    
#> Intervention  3.28903    0.15586  21.103   <2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 1.909 on 598 degrees of freedom
#> Multiple R-squared:  0.4268,	Adjusted R-squared:  0.4259 
#> F-statistic: 445.3 on 1 and 598 DF,  p-value: < 2.2e-16
```
