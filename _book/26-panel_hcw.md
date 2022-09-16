# Panel Data Methods

-   Proposed by [@hsiao2011], using dependence among cross-sectional units to create counterfactual for the treated unit.

-    If the pre-treatment period observations are less than the number of control units, we can't use this method because of not enough degrees of freedom


```r
# install.packages("pampe")
library('pampe')

# dataset form the original paper 
data(growth)
head(growth)
#>        HongKong  Australia      Austria     Canada      Denmark      Finland
#> 1993Q1    0.062 0.04048913 -0.013083510 0.01006395 -0.012291821 -0.028357059
#> 1993Q2    0.059 0.03785692 -0.007580798 0.02126387 -0.003092842 -0.023396894
#> 1993Q3    0.058 0.02250948  0.000542671 0.01891943 -0.007764421 -0.006017587
#> 1993Q4    0.062 0.02874655  0.001180751 0.02531683 -0.004048589 -0.004773978
#> 1994Q1    0.079 0.03399039  0.025510849 0.04356715  0.031094401  0.012885600
#> 1994Q2    0.068 0.03791937  0.019941313 0.05022538  0.064280003  0.035090496
#>              France     Germany        Italy        Japan      Korea
#> 1993Q1 -0.015176804 -0.01967995 -0.023382736  0.012683435 0.05858640
#> 1993Q2 -0.014549011 -0.01544115 -0.018115838 -0.005570701 0.06952104
#> 1993Q3 -0.016703892 -0.01270084 -0.016874733 -0.017558422 0.08164631
#> 1993Q4 -0.007475638 -0.01166675 -0.004963401 -0.010101253 0.08553336
#> 1994Q1  0.003748037  0.02295041 -0.002249426 -0.022502562 0.08592229
#> 1994Q2  0.016164869  0.02107002  0.011634742 -0.005157348 0.08841456
#>             Mexico  Netherlands NewZealand      Norway  Switzerland
#> 1993Q1 0.043746305 -0.008439514 0.04724391 0.004946546 -0.032864896
#> 1993Q2 0.012290427  0.006628538 0.03875869 0.020743349 -0.019817572
#> 1993Q3 0.004462324  0.008677404 0.08991753 0.038871277 -0.004587253
#> 1993Q4 0.015492945  0.004312360 0.06975085 0.053402924  0.013651317
#> 1994Q1 0.034447585  0.016938685 0.06019911 0.031001539  0.026643555
#> 1994Q2 0.063911468  0.018987630 0.06255518 0.051312977  0.010767571
#>        UnitedKingdom UnitedStates  Singapore  Philippines  Indonesia   Malaysia
#> 1993Q1    0.01512431   0.02295922 0.08714474 -0.004380854 0.06402444 0.08593838
#> 1993Q2    0.01479464   0.01893592 0.11807464  0.016635614 0.06606820 0.13118861
#> 1993Q3    0.02914905   0.01798999 0.11112953  0.031504281 0.05795939 0.10966639
#> 1993Q4    0.03658115   0.02068346 0.12532370  0.034007481 0.06236489 0.07580059
#> 1994Q1    0.03007825   0.02991833 0.13070900  0.049344055 0.04974312 0.04914714
#> 1994Q2    0.04035924   0.03784032 0.10098695  0.059129677 0.07198844 0.06117326
#>          Thailand     Taiwan China
#> 1993Q1 0.08000000 0.06490222 0.143
#> 1993Q2 0.08000000 0.06512348 0.141
#> 1993Q3 0.08000000 0.06737918 0.135
#> 1993Q4 0.08000000 0.06916444 0.135
#> 1994Q1 0.11250878 0.06945150 0.125
#> 1994Q2 0.09261321 0.07013538 0.120
colnames(growth)
#>  [1] "HongKong"      "Australia"     "Austria"       "Canada"       
#>  [5] "Denmark"       "Finland"       "France"        "Germany"      
#>  [9] "Italy"         "Japan"         "Korea"         "Mexico"       
#> [13] "Netherlands"   "NewZealand"    "Norway"        "Switzerland"  
#> [17] "UnitedKingdom" "UnitedStates"  "Singapore"     "Philippines"  
#> [21] "Indonesia"     "Malaysia"      "Thailand"      "Taiwan"       
#> [25] "China"
rownames(growth)
#>  [1] "1993Q1" "1993Q2" "1993Q3" "1993Q4" "1994Q1" "1994Q2" "1994Q3" "1994Q4"
#>  [9] "1995Q1" "1995Q2" "1995Q3" "1995Q4" "1996Q1" "1996Q2" "1996Q3" "1996Q4"
#> [17] "1997Q1" "1997Q2" "1997Q3" "1997Q4" "1998Q1" "1998Q2" "1998Q3" "1998Q4"
#> [25] "1999Q1" "1999Q2" "1999Q3" "1999Q4" "2000Q1" "2000Q2" "2000Q3" "2000Q4"
#> [33] "2001Q1" "2001Q2" "2001Q3" "2001Q4" "2002Q1" "2002Q2" "2002Q3" "2002Q4"
#> [41] "2003Q1" "2003Q2" "2003Q3" "2003Q4" "2004Q1" "2004Q2" "2004Q3" "2004Q4"
#> [49] "2005Q1" "2005Q2" "2005Q3" "2005Q4" "2006Q1" "2006Q2" "2006Q3" "2006Q4"
#> [57] "2007Q1" "2007Q2" "2007Q3" "2007Q4" "2008Q1"
```

Hong Kong is the treated unit, while the other 24 donors are control units


```r
hcw <-
    pampe(
        time.pretr = 1:18,
        time.tr = 19:44,
        treated = "HongKong",
        controls = c(
            'China',
            'Indonesia',
            'Japan',
            'Korea',
            'Malaysia',
            'Philippines',
            'Singapore',
            'Taiwan',
            'UnitedStates',
            'Thailand'
        ),
        data = growth
    )

summary(hcw)
#> Selected controls:
#>  Japan, Korea, UnitedStates, and Taiwan.
#> 
#>  
#> Time-average estimated treatment effect:
#>  -0.0396291
#> 
#> Optimal model estimation results:
#> 
#>               Estimate Std. Error t value  Pr(>|t|)    
#> (Intercept)   0.026300   0.017048  1.5427   0.14689    
#> Japan        -0.675964   0.111688 -6.0522 4.084e-05 ***
#> Korea        -0.432298   0.063377 -6.8211 1.223e-05 ***
#> UnitedStates  0.486032   0.219521  2.2141   0.04531 *  
#> Taiwan        0.792593   0.309892  2.5576   0.02385 *  
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 0.0058 on 13 degrees of freedom
#> Multiple R-squared: 0.931,     Adjusted R-squared: 0.931
#> F-statistic: 44.15 on 4 and 13 DF, p-value: 1.919427e-07

plot(hcw)
```

<img src="26-panel_hcw_files/figure-html/unnamed-chunk-2-1.png" width="90%" style="display: block; margin: auto;" />

```r

# robustness(hcw)
```
