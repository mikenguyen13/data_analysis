# Quasi-experimental

In most cases, it means that you have pre- and post-intervention data.

## Difference-In-Differences

-   $D_i = 1$ treatment group

-   $D_i = 0$ control group

-   $T= 1$ After the treatment

-   $T =0$ Before the treatment

|                   | After (T = 1)          | Before (T = 0)       |
|-------------------|------------------------|----------------------|
| Treated $D_i =1$  | $E[Y_{1i}(1)|D_i = 1]$ | $E[Y_{0i}(0)|D)i=1]$ |
| Control $D_i = 0$ | $E[Y_{0i}(1) |D_i =0]$ | $E[Y_{0i}(0)|D_i=0]$ |

missing $E[Y_{0i}(1)|D=1]$

The Average Treatment Effect on Treated

$$
E[Y_1(1) - Y_0(1)|D=1] \\
= \{E[Y(1)|D=1] - E[Y(1)|D=0] \} - \{E[Y(0)|D=1] - E[Y(0)|D=0] \}
$$

**Assumption**:

-   Parallel Trends: Difference between the treatment and control groups remain constant if there were no treatment.

should be used in cases where

-   you observe before and after an event

-   you have treatment and control groups

not in cases where

-   treatment is not random

-   confounders.

Example from [Princeton](https://www.princeton.edu/~otorres/DID101R.pdf)


```r
library(foreign)
mydata = read.dta("http://dss.princeton.edu/training/Panel101.dta")
```

create a dummy variable to indicate the time when the treatment started


```r
mydata$time = ifelse(mydata$year >= 1994, 1, 0)
```

create a dummy variable to identify the treatment group


```r
mydata$treated = ifelse(mydata$country == "E" |
                            mydata$country == "F" | mydata$country == "G" ,
                        1,
                        0)
```

create an interaction between time and treated


```r
mydata$did = mydata$time * mydata$treated
```

estimate the DID estimator


```r
didreg = lm(y ~ treated + time + did, data = mydata)
summary(didreg)
```

```
## 
## Call:
## lm(formula = y ~ treated + time + did, data = mydata)
## 
## Residuals:
##        Min         1Q     Median         3Q        Max 
## -9.768e+09 -1.623e+09  1.167e+08  1.393e+09  6.807e+09 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)  
## (Intercept)  3.581e+08  7.382e+08   0.485   0.6292  
## treated      1.776e+09  1.128e+09   1.575   0.1200  
## time         2.289e+09  9.530e+08   2.402   0.0191 *
## did         -2.520e+09  1.456e+09  -1.731   0.0882 .
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 2.953e+09 on 66 degrees of freedom
## Multiple R-squared:  0.08273,	Adjusted R-squared:  0.04104 
## F-statistic: 1.984 on 3 and 66 DF,  p-value: 0.1249
```

The `did` coefficient is the differences-in-differences estimator. Treat has a negative effect

<br>

Example by [Philipp Leppert](https://rpubs.com/phle/r_tutorial_difference_in_differences) replicating [Card and Krueger (1994)](https://davidcard.berkeley.edu/data_sets.html)

Example by [Anthony Schmidt](https://bookdown.org/aschmi11/causal_inf/difference-in-differences.html)

## Synthetic Control

Synthetic control method (SCM) is a generalization of the dif-in-dif model

Advantages over dif-in-dif:

1.  Maximization of the observable similarity between control and treatment (maybe also unobservables)
2.  Can also be used in cases where no untreated case with similar on matching dimensions with treated cases
3.  Objective selection of controls.

a data driven procedure to construct more comparable control groups.

To do causal inference with control and treatment group using [Matching Methods], you typically have to have similar covariates in the control and the treated groups. However, if you don't methods like [Propensity Scores] and DID can perform rather poorly (i.e., large bias).

SCM is recommended when

1.  Social events to evaluate large-scale program or policy
2.  Only one treated case with several control candidates.

Advantages:

1.  From the selection criteria, researchers can understand the relative importance of each candidate
2.  Post-intervention outcomes are not used in synthetic. Hence, you can't retro-fit.
3.  Observable similarity between control and treatment cases is maximized

`Synth` provides an algorithm that finds weighted combination of the comparison units where the weights are chosen such that it best resembles the values of predictors of the outcome variable for the affected units before the intervention.

### Example 1

by [Danilo Freire](https://rpubs.com/danilofreire/synth)


```r
# install.packages("Synth")
# install.packages("gsynth")
library("Synth")
```

```
## Warning: package 'Synth' was built under R version 4.0.5
```

```r
library("gsynth")
```

```
## Warning: package 'gsynth' was built under R version 4.0.5
```

simulate data for 10 states and 30 years. State A receives the treatment `T = 20` after year 15.


```r
set.seed(1)
year <- rep(1:30, 10) 
state <- rep(LETTERS[1:10], each = 30)
X1 <- round(rnorm(300, mean = 2, sd = 1), 2)
X2 <- round(rbinom(300, 1, 0.5) + rnorm(300), 2)
Y <- round(1 + 2*X1 + rnorm(300), 2)
df <- as.data.frame(cbind(Y, X1, X2, state, year))
df$Y <- as.numeric(as.character(df$Y))
df$X1 <- as.numeric(as.character(df$X1))
df$X2 <- as.numeric(as.character(df$X2))
df$year <- as.numeric(as.character(df$year))
df$state.num <- rep(1:10, each = 30)
df$state <- as.character(df$state)
df$`T` <- ifelse(df$state == "A" & df$year >= 15, 1, 0)
df$Y <- ifelse(df$state == "A" & df$year >= 15, df$Y + 20, df$Y)
```


```r
str(df)
```

```
## 'data.frame':	300 obs. of  7 variables:
##  $ Y        : num  2.29 4.51 2.07 8.87 4.37 1.32 8 7.49 6.98 3.72 ...
##  $ X1       : num  1.37 2.18 1.16 3.6 2.33 1.18 2.49 2.74 2.58 1.69 ...
##  $ X2       : num  1.96 0.4 -0.75 -0.56 -0.45 1.06 0.51 -2.1 0 0.54 ...
##  $ state    : chr  "A" "A" "A" "A" ...
##  $ year     : num  1 2 3 4 5 6 7 8 9 10 ...
##  $ state.num: int  1 1 1 1 1 1 1 1 1 1 ...
##  $ T        : num  0 0 0 0 0 0 0 0 0 0 ...
```


```r
dataprep.out <-
    dataprep(
        df,
        predictors = c("X1", "X2"),
        dependent     = "Y",
        unit.variable = "state.num",
        time.variable = "year",
        unit.names.variable = "state",
        treatment.identifier  = 1,
        controls.identifier   = c(2:10),
        time.predictors.prior = c(1:14),
        time.optimize.ssr     = c(1:14),
        time.plot             = c(1:30)
    )


synth.out <- synth(dataprep.out)
```

```
## 
## X1, X0, Z1, Z0 all come directly from dataprep object.
## 
## 
## **************** 
##  searching for synthetic control unit  
##  
## 
## **************** 
## **************** 
## **************** 
## 
## MSPE (LOSS V): 9.831789 
## 
## solution.v:
##  0.3888387 0.6111613 
## 
## solution.w:
##  0.1115941 0.1832781 0.1027237 0.312091 0.06096758 0.03509706 0.05893735 0.05746256 0.07784853
```


```r
print(synth.tables   <- synth.tab(
        dataprep.res = dataprep.out,
        synth.res    = synth.out)
      )
```

```
## $tab.pred
##    Treated Synthetic Sample Mean
## X1   2.028     2.028       2.017
## X2   0.513     0.513       0.394
## 
## $tab.v
##    v.weights
## X1 0.389    
## X2 0.611    
## 
## $tab.w
##    w.weights unit.names unit.numbers
## 2      0.112          B            2
## 3      0.183          C            3
## 4      0.103          D            4
## 5      0.312          E            5
## 6      0.061          F            6
## 7      0.035          G            7
## 8      0.059          H            8
## 9      0.057          I            9
## 10     0.078          J           10
## 
## $tab.loss
##            Loss W   Loss V
## [1,] 9.761708e-12 9.831789
```


```r
path.plot(synth.res    = synth.out,
          dataprep.res = dataprep.out,
          Ylab         = c("Y"),
          Xlab         = c("Year"),
          Legend       = c("State A","Synthetic State A"),
          Legend.position = c("topleft")
)

abline(v   = 15,
       lty = 2)
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-11-1.png)<!-- -->

Gaps plot:


```r
gaps.plot(synth.res    = synth.out,
          dataprep.res = dataprep.out,
          Ylab         = c("Gap"),
          Xlab         = c("Year"),
          Ylim         = c(-30, 30),
          Main         = ""
)

abline(v   = 15,
       lty = 2)
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-12-1.png)<!-- -->

Alternatively, `gsynth` provides options to estimate iterative fixed effects, and handle multiple treated units at tat time.

Here, we use two=way fixed effects and bootstrapped standard errors


```r
gsynth.out <- gsynth(
  Y ~ `T` + X1 + X2,
  data = df,
  index = c("state", "year"),
  force = "two-way",
  CV = TRUE,
  r = c(0, 5),
  se = TRUE,
  inference = "parametric",
  nboots = 1000,
  parallel = F # TRUE
)
```

```
## Cross-validating ... 
##  r = 0; sigma2 = 0.97435; IC = 0.01548; MSPE = 1.65502
##  r = 1; sigma2 = 0.81720; IC = 0.62752; MSPE = 1.33375
##  r = 2; sigma2 = 0.67509; IC = 1.18295; MSPE = 1.27341*
##  r = 3; sigma2 = 0.57336; IC = 1.72459; MSPE = 1.79319
##  r = 4; sigma2 = 0.48099; IC = 2.21245; MSPE = 2.02301
##  r = 5; sigma2 = 0.39641; IC = 2.64109; MSPE = 2.79596
## 
##  r* = 2
## 
## Simulating errors .............Bootstrapping ...
## ..........
```


```r
plot(gsynth.out)
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-14-1.png)<!-- -->


```r
plot(gsynth.out, type = "counterfactual")
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-15-1.png)<!-- -->


```r
plot(gsynth.out, type = "counterfactual", raw = "all") # shows estimations for the control cases
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-16-1.png)<!-- -->

### Example 2

by [Leihua Ye](https://towardsdatascience.com/causal-inference-using-synthetic-control-the-ultimate-guide-a622ad5cf827)


```r
library(Synth)
data("basque")
dim(basque) #774*17
```

```
## [1] 774  17
```

```r
head(basque)
```

```
##   regionno     regionname year   gdpcap sec.agriculture sec.energy sec.industry
## 1        1 Spain (Espana) 1955 2.354542              NA         NA           NA
## 2        1 Spain (Espana) 1956 2.480149              NA         NA           NA
## 3        1 Spain (Espana) 1957 2.603613              NA         NA           NA
## 4        1 Spain (Espana) 1958 2.637104              NA         NA           NA
## 5        1 Spain (Espana) 1959 2.669880              NA         NA           NA
## 6        1 Spain (Espana) 1960 2.869966              NA         NA           NA
##   sec.construction sec.services.venta sec.services.nonventa school.illit
## 1               NA                 NA                    NA           NA
## 2               NA                 NA                    NA           NA
## 3               NA                 NA                    NA           NA
## 4               NA                 NA                    NA           NA
## 5               NA                 NA                    NA           NA
## 6               NA                 NA                    NA           NA
##   school.prim school.med school.high school.post.high popdens invest
## 1          NA         NA          NA               NA      NA     NA
## 2          NA         NA          NA               NA      NA     NA
## 3          NA         NA          NA               NA      NA     NA
## 4          NA         NA          NA               NA      NA     NA
## 5          NA         NA          NA               NA      NA     NA
## 6          NA         NA          NA               NA      NA     NA
```

transform data to be used in `synth()`


```r
dataprep.out <- dataprep(
    foo = basque,
    predictors = c(
        "school.illit",
        "school.prim",
        "school.med",
        "school.high",
        "school.post.high",
        "invest"
    ),
    predictors.op =  "mean",
    # the operator
    time.predictors.prior = 1964:1969,
    #the entire time frame from the #beginning to the end
    special.predictors = list(
        list("gdpcap", 1960:1969,  "mean"),
        list("sec.agriculture", seq(1961, 1969, 2), "mean"),
        list("sec.energy", seq(1961, 1969, 2), "mean"),
        list("sec.industry", seq(1961, 1969, 2), "mean"),
        list("sec.construction", seq(1961, 1969, 2), "mean"),
        list("sec.services.venta", seq(1961, 1969, 2), "mean"),
        list("sec.services.nonventa", seq(1961, 1969, 2), "mean"),
        list("popdens", 1969,  "mean")
    ),
    dependent =  "gdpcap",
    # dv
    unit.variable =  "regionno",
    #identifying unit numbers
    unit.names.variable =  "regionname",
    #identifying unit names
    time.variable =  "year",
    #time-periods
    treatment.identifier = 17,
    #the treated case
    controls.identifier = c(2:16, 18),
    #the control cases; all others #except number 17
    time.optimize.ssr = 1960:1969,
    #the time-period over which to optimize
    time.plot = 1955:1997
)#the entire time period before/after the treatment
```

where

-   X1 = the control case before the treatment

-   X0 = the control cases after the treatment

-   Z1: the treatment case before the treatment

-   Z0: the treatment case after the treatment


```r
synth.out = synth(data.prep.obj = dataprep.out, method = "BFGS")
```

```
## 
## X1, X0, Z1, Z0 all come directly from dataprep object.
## 
## 
## **************** 
##  searching for synthetic control unit  
##  
## 
## **************** 
## **************** 
## **************** 
## 
## MSPE (LOSS V): 0.008864606 
## 
## solution.v:
##  0.02773094 1.194e-07 1.60609e-05 0.0007163836 1.486e-07 0.002423908 0.0587055 0.2651997 0.02851006 0.291276 0.007994382 0.004053188 0.009398579 0.303975 
## 
## solution.w:
##  2.53e-08 4.63e-08 6.44e-08 2.81e-08 3.37e-08 4.844e-07 4.2e-08 4.69e-08 0.8508145 9.75e-08 3.2e-08 5.54e-08 0.1491843 4.86e-08 9.89e-08 1.162e-07
```

Calculate the difference between the real basque region and the synthetic control


```r
gaps = dataprep.out$Y1plot - (dataprep.out$Y0plot 
                                     %*% synth.out$solution.w)
gaps[1:3,1]
```

```
##       1955       1956       1957 
## 0.15023473 0.09168035 0.03716475
```


```r
synth.tables = synth.tab(dataprep.res = dataprep.out,
                         synth.res = synth.out)
names(synth.tables)
```

```
## [1] "tab.pred" "tab.v"    "tab.w"    "tab.loss"
```

```r
synth.tables$tab.pred[1:13,]
```

```
##                                          Treated Synthetic Sample Mean
## school.illit                              39.888   256.337     170.786
## school.prim                             1031.742  2730.104    1127.186
## school.med                                90.359   223.340      76.260
## school.high                               25.728    63.437      24.235
## school.post.high                          13.480    36.153      13.478
## invest                                    24.647    21.583      21.424
## special.gdpcap.1960.1969                   5.285     5.271       3.581
## special.sec.agriculture.1961.1969          6.844     6.179      21.353
## special.sec.energy.1961.1969               4.106     2.760       5.310
## special.sec.industry.1961.1969            45.082    37.636      22.425
## special.sec.construction.1961.1969         6.150     6.952       7.276
## special.sec.services.venta.1961.1969      33.754    41.104      36.528
## special.sec.services.nonventa.1961.1969    4.072     5.371       7.111
```

Relative importance of each unit


```r
synth.tables$tab.w[8:14, ]
```

```
##    w.weights            unit.names unit.numbers
## 9      0.000    Castilla-La Mancha            9
## 10     0.851              Cataluna           10
## 11     0.000  Comunidad Valenciana           11
## 12     0.000           Extremadura           12
## 13     0.000               Galicia           13
## 14     0.149 Madrid (Comunidad De)           14
## 15     0.000    Murcia (Region de)           15
```


```r
# plot the changes before and after the treatment 
path.plot(
    synth.res = synth.out,
    dataprep.res = dataprep.out,
    Ylab = "real per-capita gdp (1986 USD, thousand)",
    Xlab = "year",
    Ylim = c(0, 12),
    Legend = c("Basque country",
               "synthetic Basque country"),
    Legend.position = "bottomright"
)
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-23-1.png)<!-- -->


```r
gaps.plot(
    synth.res = synth.out,
    dataprep.res = dataprep.out,
    Ylab =  "gap in real per - capita GDP (1986 USD, thousand)",
    Xlab =  "year",
    Ylim = c(-1.5, 1.5),
    Main = NA
)
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-24-1.png)<!-- -->

Doubly Robust Difference-in-Differences

Example from `DRDID` package


```r
library(DRDID)
```

```
## Warning: package 'DRDID' was built under R version 4.0.5
```

```r
data(nsw_long)
# Form the Lalonde sample with CPS comparison group
eval_lalonde_cps <- subset(nsw_long, nsw_long$treated == 0 | nsw_long$sample == 2)
```

Estimate Average Treatment Effect on Treated using Improved Locally Efficient Doubly Robust DID estimator


```r
out <-
    drdid(
        yname = "re",
        tname = "year",
        idname = "id",
        dname = "experimental",
        xformla = ~ age + educ + black + married + nodegree + hisp + re74,
        data = eval_lalonde_cps,
        panel = TRUE
    )
summary(out)
```

```
##  Call:
## drdid(yname = "re", tname = "year", idname = "id", dname = "experimental", 
##     xformla = ~age + educ + black + married + nodegree + hisp + 
##         re74, data = eval_lalonde_cps, panel = TRUE)
## ------------------------------------------------------------------
##  Further improved locally efficient DR DID estimator for the ATT:
##  
##    ATT     Std. Error  t value    Pr(>|t|)  [95% Conf. Interval] 
## -901.2703   393.6247   -2.2897     0.022    -1672.7747  -129.766 
## ------------------------------------------------------------------
##  Estimator based on panel data.
##  Outcome regression est. method: weighted least squares.
##  Propensity score est. method: inverse prob. tilting.
##  Analytical standard error.
## ------------------------------------------------------------------
##  See Sant'Anna and Zhao (2020) for details.
```

### Example 3

by `Synth` package's authors


```r
library(Synth)
data("basque")
```

`synth()` requires

-   $X_1$ vector of treatment predictors

-   $X_0$ matrix of same variables for control group

-   $Z_1$ vector of outcome variable for treatment group

-   $Z_0$ matrix of outcome variable for control group

use `dataprep()` to prepare data in the format that can be used throughout the `Synth` package


```r
dataprep.out <- dataprep(
    foo = basque,
    predictors = c(
        "school.illit",
        "school.prim",
        "school.med",
        "school.high",
        "school.post.high",
        "invest"
    ),
    predictors.op = "mean",
    time.predictors.prior = 1964:1969,
    special.predictors = list(
        list("gdpcap", 1960:1969 , "mean"),
        list("sec.agriculture", seq(1961, 1969, 2), "mean"),
        list("sec.energy", seq(1961, 1969, 2), "mean"),
        list("sec.industry", seq(1961, 1969, 2), "mean"),
        list("sec.construction", seq(1961, 1969, 2), "mean"),
        list("sec.services.venta", seq(1961, 1969, 2), "mean"),
        list("sec.services.nonventa", seq(1961, 1969, 2), "mean"),
        list("popdens", 1969, "mean")
    ),
    dependent = "gdpcap",
    unit.variable = "regionno",
    unit.names.variable = "regionname",
    time.variable = "year",
    treatment.identifier = 17,
    controls.identifier = c(2:16, 18),
    time.optimize.ssr = 1960:1969,
    time.plot = 1955:1997
)
```

find optimal weights that identifies the synthetic control for the treatment group


```r
synth.out <- synth(data.prep.obj = dataprep.out, method = "BFGS")
```

```
## 
## X1, X0, Z1, Z0 all come directly from dataprep object.
## 
## 
## **************** 
##  searching for synthetic control unit  
##  
## 
## **************** 
## **************** 
## **************** 
## 
## MSPE (LOSS V): 0.008864606 
## 
## solution.v:
##  0.02773094 1.194e-07 1.60609e-05 0.0007163836 1.486e-07 0.002423908 0.0587055 0.2651997 0.02851006 0.291276 0.007994382 0.004053188 0.009398579 0.303975 
## 
## solution.w:
##  2.53e-08 4.63e-08 6.44e-08 2.81e-08 3.37e-08 4.844e-07 4.2e-08 4.69e-08 0.8508145 9.75e-08 3.2e-08 5.54e-08 0.1491843 4.86e-08 9.89e-08 1.162e-07
```


```r
gaps <- dataprep.out$Y1plot - (dataprep.out$Y0plot %*% synth.out$solution.w)
gaps[1:3, 1]
```

```
##       1955       1956       1957 
## 0.15023473 0.09168035 0.03716475
```


```r
synth.tables <-
    synth.tab(dataprep.res = dataprep.out, synth.res = synth.out)
names(synth.tables) # you can pick tables to see 
```

```
## [1] "tab.pred" "tab.v"    "tab.w"    "tab.loss"
```


```r
path.plot(
    synth.res = synth.out,
    dataprep.res = dataprep.out,
    Ylab = "real per-capita GDP (1986 USD, thousand)",
    Xlab = "year",
    Ylim = c(0, 12),
    Legend = c("Basque country",
               "synthetic Basque country"),
    Legend.position = "bottomright"
)
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-32-1.png)<!-- -->


```r
gaps.plot(
    synth.res = synth.out,
    dataprep.res = dataprep.out,
    Ylab = "gap in real per-capita GDP (1986 USD, thousand)",
    Xlab = "year",
    Ylim = c(-1.5, 1.5),
    Main = NA
)
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-33-1.png)<!-- -->

You could also run placebo tests

<br>

### Example 4

by [Michael Robbins and Steven Davenport](https://cran.r-project.org/web/packages/microsynth/vignettes/introduction.html) who are authors of `MicroSynth` with the following improvements:

-   Standardization `use.survey = TRUE` and permutation ( `perm = 250` and `jack = TRUE` ) for placebo tests

-   Omnibus statistic (set to `omnibus.var` ) for multiple outcome variables

-   incorporate multiple follow-up periods `end.post`

Notes:

-   Both predictors and outcome will be used to match units before intervention

    -   Outcome variable has to be **time-variant**

    -   Predictors are **time-invariant**

-   


```r
library(microsynth)
```

```
## Warning: package 'microsynth' was built under R version 4.0.5
```

```r
data("seattledmi")

cov.var <- c("TotalPop", "BLACK", "HISPANIC", "Males_1521", "HOUSEHOLDS", 
             "FAMILYHOUS", "FEMALE_HOU", "RENTER_HOU", "VACANT_HOU")
match.out <- c("i_felony", "i_misdemea", "i_drugs", "any_crime")
```


```r
sea1 <- microsynth(
    seattledmi,
    idvar = "ID",
    timevar = "time",
    intvar = "Intervention",
    start.pre = 1,
    end.pre = 12,
    end.post = 16,
    match.out = match.out, # outcome variable will be matched on exactly
    match.covar = cov.var, # specify covariates will be matched on exactly
    result.var = match.out, # used to report results
    omnibus.var = match.out, # feature in the omnibus p-value
    test = "lower",
    n.cores = min(parallel::detectCores(), 2)
)
```

```
## Calculating weights...
```

```
## Created main weights for synthetic control: Time = 1.74
```

```
## Matching summary for main weights:
```

```
##               Targets Weighted.Control All.scaled
## Intercept          39          39.0002    39.0000
## TotalPop         2994        2994.0519  2384.7477
## BLACK             173         173.0010   190.5224
## HISPANIC          149         149.0026   159.2682
## Males_1521         49          49.0000    97.3746
## HOUSEHOLDS       1968        1968.0340  1113.5588
## FAMILYHOUS        519         519.0108   475.1876
## FEMALE_HOU        101         101.0010    81.1549
## RENTER_HOU       1868        1868.0203   581.9340
## VACANT_HOU        160         160.0115    98.4222
## i_felony.12        14          14.0000     4.9023
## i_felony.11        11          11.0002     4.6313
## i_felony.10         9           9.0000     3.0741
## i_felony.9          5           5.0000     3.2642
## i_felony.8         20          20.0000     4.4331
## i_felony.7          8           8.0000     3.7617
## i_felony.6         13          13.0000     3.0012
## i_felony.5         20          20.0007     3.1549
## i_felony.4         10          10.0000     4.0246
## i_felony.3          7           7.0000     3.3693
## i_felony.2         13          13.0002     3.2803
## i_felony.1         12          12.0000     3.4381
## i_misdemea.12      15          15.0002     4.2470
## i_misdemea.11      12          12.0000     4.6070
## i_misdemea.10      12          12.0000     4.0772
## i_misdemea.9       14          14.0000     3.7414
## i_misdemea.8       12          12.0000     3.9680
## i_misdemea.7       20          20.0000     4.2551
## i_misdemea.6       16          16.0005     3.5594
## i_misdemea.5       24          24.0000     3.5635
## i_misdemea.4       21          21.0002     4.3360
## i_misdemea.3       21          21.0000     4.3846
## i_misdemea.2       14          14.0000     3.5352
## i_misdemea.1       16          16.0000     4.1540
## i_drugs.12         13          13.0000     1.6543
## i_drugs.11          8           8.0000     1.5128
## i_drugs.10          3           3.0000     1.3227
## i_drugs.9           4           4.0000     0.9788
## i_drugs.8           4           4.0000     1.1123
## i_drugs.7          10          10.0000     1.0516
## i_drugs.6           4           4.0000     1.2377
## i_drugs.5           2           2.0000     1.2296
## i_drugs.4           1           1.0000     1.1245
## i_drugs.3           5           5.0000     1.3550
## i_drugs.2          12          12.0000     1.1366
## i_drugs.1           8           8.0002     1.3591
## any_crime.12      272         272.0012    65.3398
## any_crime.11      227         227.0017    64.2396
## any_crime.10      183         183.0010    55.6929
## any_crime.9       176         176.0005    53.2377
## any_crime.8       228         228.0005    55.8143
## any_crime.7       246         246.0024    55.8062
## any_crime.6       200         200.0010    52.8292
## any_crime.5       270         270.0014    50.6531
## any_crime.4       250         250.0010    57.2946
## any_crime.3       236         236.0010    58.8681
## any_crime.2       250         250.0012    51.5429
## any_crime.1       242         242.0010    55.1145
## 
## Calculation of weights complete: Total time = 2.41
## 
## Calculating basic statistics for end.post = 16...
## Completed calculation of basic statistics for end.post = 16.  Time = 3.07
## 
## Calculating survey statistics for end.post = 16...
## Completed survey statistics for main weights: Time = 6.19
## Completed calculation of survey statistics for end.post = 16.  Time = 6.19
## 
## microsynth complete: Overall time = 14.92
```

```r
sea1
```

```
## 	microsynth object
## 
## Scope:
## 	Units:			Total: 9642	Treated: 39	Untreated: 9603
## 	Study Period(s):	Pre-period: 1 - 12	Post-period: 13 - 16
## 	Constraints:		Exact Match: 58		Minimized Distance: 0
## Time-variant outcomes:
## 	Exact Match: i_felony, i_misdemea, i_drugs, any_crime (4)
## 	Minimized Distance: (0)
## Time-invariant covariates:
## 	Exact Match: TotalPop, BLACK, HISPANIC, Males_1521, HOUSEHOLDS, FAMILYHOUS, FEMALE_HOU, RENTER_HOU, VACANT_HOU (9)
## 	Minimized Distance: (0)
## 
## Results:
## end.post = 16
##            Trt    Con Pct.Chng Linear.pVal Linear.Lower Linear.Upper
## i_felony    46  68.22   -32.6%      0.0109       -50.3%        -8.4%
## i_misdemea  45  71.80   -37.3%      0.0019       -52.8%       -16.7%
## i_drugs     20  23.76   -15.8%      0.2559       -46.4%        32.1%
## any_crime  788 986.44   -20.1%      0.0146       -32.9%        -4.9%
## Omnibus     --     --       --      0.0006           --           --
```

```r
summary(sea1)
```

```
## Weight Balance Table: 
## 
##               Targets Weighted.Control   All.scaled
## Intercept          39        39.000239   39.0000000
## TotalPop         2994      2994.051921 2384.7476665
## BLACK             173       173.000957  190.5224020
## HISPANIC          149       149.002632  159.2682016
## Males_1521         49        49.000000   97.3746111
## HOUSEHOLDS       1968      1968.033976 1113.5588052
## FAMILYHOUS        519       519.010767  475.1876167
## FEMALE_HOU        101       101.000957   81.1549471
## RENTER_HOU       1868      1868.020338  581.9340386
## VACANT_HOU        160       160.011485   98.4222153
## i_felony.12        14        14.000000    4.9023024
## i_felony.11        11        11.000239    4.6313006
## i_felony.10         9         9.000000    3.0740510
## i_felony.9          5         5.000000    3.2641568
## i_felony.8         20        20.000000    4.4331052
## i_felony.7          8         8.000000    3.7616677
## i_felony.6         13        13.000000    3.0012446
## i_felony.5         20        20.000718    3.1549471
## i_felony.4         10        10.000000    4.0245800
## i_felony.3          7         7.000000    3.3693217
## i_felony.2         13        13.000239    3.2803360
## i_felony.1         12        12.000000    3.4380834
## i_misdemea.12      15        15.000239    4.2470442
## i_misdemea.11      12        12.000000    4.6070317
## i_misdemea.10      12        12.000000    4.0771624
## i_misdemea.9       14        14.000000    3.7414437
## i_misdemea.8       12        12.000000    3.9679527
## i_misdemea.7       20        20.000000    4.2551338
## i_misdemea.6       16        16.000479    3.5594275
## i_misdemea.5       24        24.000000    3.5634723
## i_misdemea.4       21        21.000239    4.3360299
## i_misdemea.3       21        21.000000    4.3845675
## i_misdemea.2       14        14.000000    3.5351587
## i_misdemea.1       16        16.000000    4.1540137
## i_drugs.12         13        13.000000    1.6543248
## i_drugs.11          8         8.000000    1.5127567
## i_drugs.10          3         3.000000    1.3226509
## i_drugs.9           4         4.000000    0.9788426
## i_drugs.8           4         4.000000    1.1123211
## i_drugs.7          10        10.000000    1.0516490
## i_drugs.6           4         4.000000    1.2377100
## i_drugs.5           2         2.000000    1.2296204
## i_drugs.4           1         1.000000    1.1244555
## i_drugs.3           5         5.000000    1.3550093
## i_drugs.2          12        12.000000    1.1365899
## i_drugs.1           8         8.000239    1.3590541
## any_crime.12      272       272.001196   65.3397635
## any_crime.11      227       227.001675   64.2395769
## any_crime.10      183       183.000957   55.6929060
## any_crime.9       176       176.000479   53.2377100
## any_crime.8       228       228.000479   55.8142502
## any_crime.7       246       246.002393   55.8061605
## any_crime.6       200       200.000957   52.8291848
## any_crime.5       270       270.001436   50.6530803
## any_crime.4       250       250.000957   57.2946484
## any_crime.3       236       236.000957   58.8680772
## any_crime.2       250       250.001196   51.5429371
## any_crime.1       242       242.000957   55.1144991
## 
## Results: 
## 
## end.post = 16
##            Trt    Con Pct.Chng Linear.pVal Linear.Lower Linear.Upper
## i_felony    46  68.22   -32.6%      0.0109       -50.3%        -8.4%
## i_misdemea  45  71.80   -37.3%      0.0019       -52.8%       -16.7%
## i_drugs     20  23.76   -15.8%      0.2559       -46.4%        32.1%
## any_crime  788 986.44   -20.1%      0.0146       -32.9%        -4.9%
## Omnibus     --     --       --      0.0006           --           --
```


```r
plot_microsynth(sea1)
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-36-1.png)<!-- -->![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-36-2.png)<!-- -->![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-36-3.png)<!-- -->![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-36-4.png)<!-- -->


```r
sea2 <- microsynth(seattledmi, 
                   idvar="ID", timevar="time", intvar="Intervention", 
                   start.pre=1, end.pre=12, end.post=c(14, 16),
                   match.out=match.out, match.covar=cov.var, 
                   result.var=match.out, omnibus.var=match.out, 
                   test="lower", 
                   perm=250, jack=TRUE,
                   n.cores = min(parallel::detectCores(), 2))
```

## Interrupted Time Series

-   Control for

    -   seasonable trends

    -   Concurrent events

-   Pros [@Penfold_2013]

    -   control for long-term trends

-   Cons

    -   Min of 8 data points before and 8 after an intervention

    -   Multiple events hard to distinguish

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
summary(model)
```

```
## 
## Call:
## lm(formula = Outcome_Variable ~ Intervention, data = mydata)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -3.3050 -1.2315 -0.1734  0.8691 11.9185 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   0.03358    0.11021   0.305    0.761    
## Intervention  3.28903    0.15586  21.103   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.909 on 598 degrees of freedom
## Multiple R-squared:  0.4268,	Adjusted R-squared:  0.4259 
## F-statistic: 445.3 on 1 and 598 DF,  p-value: < 2.2e-16
```

## Regression Discontinuity Design

$$
D_i = 1_{X_i > c}
$$

$$
D_i = 
\begin{cases}
D_i = 1 \text{ if } X_i > C \\
D_i = 0 \text{ if } X_i < C
\end{cases}
$$

where

-   $D_i$ = treatment effect

-   $X_i$ = score variable (continuous)

-   $c$ = cutoff point

$$
\begin{aligned}
\alpha_{SRDD} &= E[Y_{1i} - Y_{0i} | X_i = c] \\
&= E[Y_{1i}|X_i = c] - E[Y_{0i}|X_i = c]\\
&= \lim_{x \to c^+} E[Y_{1i}|X_i = c] - \lim_{x \to c^=} E[Y_{0i}|X_i = c]
\end{aligned}
$$

RDD estimates the local average treatment effect (LATE), at the cutoff point which is not at the individual or population levels.

Assumptions:

-   Independent assignment

-   Continuity of conditional regression functions

    -   $E[Y(0)|X=x]$ and $E[Y(1)|X=x]$ are continuous in x.

Example by [Leihua Ye](https://towardsdatascience.com/the-crown-jewel-of-causal-inference-regression-discontinuity-design-rdd-bad37a68e786)

$$
Y_i = \beta_0 + \beta_1 X_i + \beta_2 W_i + u_i
$$

$$
X_i = 
\begin{cases}
1, W_i \ge c \\
0, W_i < c
\end{cases}
$$


```r
#cutoff point = 3.5
GPA <- runif(1000, 0, 4)
future_success <- 10 + 2 * GPA + 10 * (GPA >= 3.5) + rnorm(1000)
#install and load the package ‘rddtools’
#install.packages(“rddtools”)
library(rddtools)
```

```
## Warning: package 'rddtools' was built under R version 4.0.5
```

```
## Warning: package 'zoo' was built under R version 4.0.5
```

```
## Warning: package 'survival' was built under R version 4.0.5
```

```
## Warning: package 'np' was built under R version 4.0.5
```

```
## Warning in .recacheSubclasses(def@className, def, env): undefined subclass
## "numericVector" of class "Mnumeric"; definition not updated
```

```r
data <- rdd_data(future_success, GPA, cutpoint = 3.5)
# plot the dataset
plot(
    data,
    col =  "red",
    cex = 0.1,
    xlab =  "GPA",
    ylab =  "future_success"
)
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-39-1.png)<!-- -->


```r
# estimate the sharp RDD model
rdd_mod <- rdd_reg_lm(rdd_object = data, slope =  "same")
summary(rdd_mod)
```

```
## 
## Call:
## lm(formula = y ~ ., data = dat_step1, weights = weights)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -3.2171 -0.6724 -0.0341  0.7433  3.6647 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept) 16.97723    0.06973  243.48   <2e-16 ***
## D            9.99393    0.12152   82.24   <2e-16 ***
## x            2.00382    0.03348   59.85   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.044 on 997 degrees of freedom
## Multiple R-squared:  0.9592,	Adjusted R-squared:  0.9591 
## F-statistic: 1.171e+04 on 2 and 997 DF,  p-value: < 2.2e-16
```


```r
# plot the RDD model along with binned observations
plot(
    rdd_mod,
    cex = 0.1,
    col =  "red",
    xlab =  "GPA",
    ylab =  "future_success"
)
```

![](26-dif-in-dif_files/figure-epub3/unnamed-chunk-41-1.png)<!-- -->
