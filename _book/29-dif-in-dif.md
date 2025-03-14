# Difference-in-differences {#sec-difference-in-differences}

[List of packages](https://github.com/lnsongxf/DiD-1)

Examples in marketing

-   [@liaukonyte2015television]: TV ad on online shopping
-   [@wang2018border]: political ad source and message tone on vote shares and turnout using discontinuities in the level of political ads at the borders
-   [@datta2018changing]: streaming service on total music consumption using timing of users adoption of a music streaming service
-   [@janakiraman2018effect]: data breach announcement affect customer spending using timing of data breach and variation whether customer info was breached in that event
-   [@israeli2018online]: digital monitoring and enforcement on violations using enforcement of min ad price policies
-   [@ramani2019effects]: firms respond to foreign direct investment liberalization using India's reform in 1991.
-   [@pattabhiramaiah2019paywalls]: paywall affects readership
-   [@akca2020value]: aggregators for airlines business effect
-   [@lim2020competitive]: nutritional labels on nutritional quality for other brands in a category using variation in timing of adoption of nutritional labels across categories
-   [@guo2020let]: payment disclosure laws effect on physician prescription behavior using Timing of the Massachusetts open payment law as the exogenous shock
-   [@he2022market]: using Amazon policy change to examine the causal impact of fake reviews on sales, average ratings.
-   [@peukert2022regulatory]: using European General data protection Regulation, examine the impact of policy change on website usage.

Examples in econ:

-   [@rosenzweig2000natural]

-   [@angrist2001instrumental]

-   [@fuchs2016natural]: macro

Show the mechanism via

-   [Mediation Under DiD] analysis: see [@habel2021variable]

-   Moderation analysis: see [@goldfarb2011online]

Steps to trust DID:

1.  Visualize the treatment rollout (e.g., `panelView`).

2.  Document the number of treated units in each cohort (e.g., control and treated).

3.  Visualize the trajectory of average outcomes across cohorts (if you have multiple periods).

4.  [Parallel Trends](#prior-parallel-trends-test) Conduct an event-study analysis with and without covariates.

5.  For the case with covariates, check for overlap in covariates between treated and control groups to ensure control group validity (e.g., if the control is relatively small than the treated group, you might not have overlap, and you have to make extrapolation).

6.  Conduct sensitivity analysis for parallel trend violations (e.g., `honestDiD`).

## Visualization


```r
library(panelView)
library(fixest)
library(tidyverse)
base_stagg <- fixest::base_stagg |>
    # treatment status
    dplyr::mutate(treat_stat = dplyr::if_else(time_to_treatment < 0, 0, 1)) |> 
    select(id, year, treat_stat, y)

head(base_stagg)
#>   id year treat_stat           y
#> 2 90    1          0  0.01722971
#> 3 89    1          0 -4.58084528
#> 4 88    1          0  2.73817174
#> 5 87    1          0 -0.65103066
#> 6 86    1          0 -5.33381664
#> 7 85    1          0  0.49562631

panelView::panelview(
    y ~ treat_stat,
    data = base_stagg,
    index = c("id", "year"),
    xlab = "Year",
    ylab = "Unit",
    display.all = F,
    gridOff = T,
    by.timing = T
)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-1-1.png" width="90%" style="display: block; margin: auto;" />

```r

# alternatively specification
panelView::panelview(
    Y = "y",
    D = "treat_stat",
    data = base_stagg,
    index = c("id", "year"),
    xlab = "Year",
    ylab = "Unit",
    display.all = F,
    gridOff = T,
    by.timing = T
)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-1-2.png" width="90%" style="display: block; margin: auto;" />

```r

# Average outcomes for each cohort
panelView::panelview(
    data = base_stagg, 
    Y = "y",
    D = "treat_stat",
    index = c("id", "year"),
    by.timing = T,
    display.all = F,
    type = "outcome", 
    by.cohort = T
)
#> Number of unique treatment histories: 10
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-1-3.png" width="90%" style="display: block; margin: auto;" />

## Simple Dif-n-dif

-   A tool developed intuitively to study "natural experiment", but its uses are much broader.

-   [Fixed Effects Estimator] is the foundation for DID

-   Why is dif-in-dif attractive? Identification strategy: Inter-temporal variation between groups

    -   **Cross-sectional estimator** helps avoid omitted (unobserved) **common trends**

    -   **Time-series estimator** helps overcome omitted (unobserved) **cross-sectional differences**

Consider

-   $D_i = 1$ treatment group

-   $D_i = 0$ control group

-   $T= 1$ After the treatment

-   $T =0$ Before the treatment

|                   | After (T = 1)          | Before (T = 0)       |
|-------------------|------------------------|----------------------|
| Treated $D_i =1$  | $E[Y_{1i}(1)|D_i = 1]$ | $E[Y_{0i}(0)|D)i=1]$ |
| Control $D_i = 0$ | $E[Y_{0i}(1) |D_i =0]$ | $E[Y_{0i}(0)|D_i=0]$ |

missing $E[Y_{0i}(1)|D=1]$

**The Average Treatment Effect on Treated (ATT)**

$$
\begin{aligned}
E[Y_1(1) - Y_0(1)|D=1] &= \{E[Y(1)|D=1] - E[Y(1)|D=0] \} \\
&- \{E[Y(0)|D=1] - E[Y(0)|D=0] \}
\end{aligned}
$$

More elaboration:

-   For the treatment group, we isolate the difference between being treated and not being treated. If the untreated group would have been affected in a different way, the DiD design and estimate would tell us nothing.
-   Alternatively, because we can't observe treatment variation in the control group, we can't say anything about the treatment effect on this group.

**Extension**

1.  **More than 2 groups** (multiple treatments and multiple controls), and more than 2 period (pre and post)

$$
Y_{igt} = \alpha_g + \gamma_t + \beta I_{gt} + \delta X_{igt} + \epsilon_{igt}
$$

where

-   $\alpha_g$ is the group-specific fixed effect

-   $\gamma_t$ = time specific fixed effect

-   $\beta$ = dif-in-dif effect

-   $I_{gt}$ = interaction terms (n treatment indicators x n post-treatment dummies) (capture effect heterogeneity over time)

This specification is the "two-way fixed effects DiD" - **TWFE** (i.e., 2 sets of fixed effects: group + time).

-   However, if you have [Staggered Dif-n-dif] (i.e., treatment is applied at different times to different groups). TWFE is really bad.

2.  **Long-term Effects**

To examine the dynamic treatment effects (that are not under rollout/staggered design), we can create a centered time variable,

+------------------------+------------------------------------------------+
| Centered Time Variable | Period                                         |
+========================+================================================+
| ...                    |                                                |
+------------------------+------------------------------------------------+
| $t = -1$               | 2 periods before treatment period              |
+------------------------+------------------------------------------------+
| $t = 0$                | Last period right before treatment period      |
|                        |                                                |
|                        | Remember to use this period as reference group |
+------------------------+------------------------------------------------+
| $t = 1$                | Treatment period                               |
+------------------------+------------------------------------------------+
| ...                    |                                                |
+------------------------+------------------------------------------------+

By interacting this factor variable, we can examine the dynamic effect of treatment (i.e., whether it's fading or intensifying)

$$
\begin{aligned}
Y &= \alpha_0 + \alpha_1 Group + \alpha_2 Time  \\
&+ \beta_{-T_1} Treatment+  \beta_{-(T_1 -1)} Treatment + \dots +  \beta_{-1} Treatment \\
&+ \beta_1 + \dots + \beta_{T_2} Treatment
\end{aligned}
$$

where

-   $\beta_0$ is used as the reference group (i.e., drop from the model)

-   $T_1$ is the pre-treatment period

-   $T_2$ is the post-treatment period

With more variables (i.e., interaction terms), coefficients estimates can be less precise (i.e., higher SE).

3.  DiD on the relationship, not levels. Technically, we can apply DiD research design not only on variables, but also on coefficients estimates of some other regression models with before and after a policy is implemented.

Goal:

1.  Pre-treatment coefficients should be non-significant $\beta_{-T_1}, \dots, \beta_{-1} = 0$ (similar to the [Placebo Test])
2.  Post-treatment coefficients are expected to be significant $\beta_1, \dots, \beta_{T_2} \neq0$
    -   You can now examine the trend in post-treatment coefficients (i.e., increasing or decreasing)


```r
library(tidyverse)
library(fixest)

od <- causaldata::organ_donations %>%
    
    # Treatment variable
    dplyr::mutate(California = State == 'California') %>%
    # centered time variable
    dplyr::mutate(center_time = as.factor(Quarter_Num - 3))  
# where 3 is the reference period precedes the treatment period

class(od$California)
#> [1] "logical"
class(od$State)
#> [1] "character"

cali <- feols(Rate ~ i(center_time, California, ref = 0) |
                  State + center_time,
              data = od)

etable(cali)
#>                                              cali
#> Dependent Var.:                              Rate
#>                                                  
#> California x center_time = -2    -0.0029 (0.0051)
#> California x center_time = -1   0.0063** (0.0023)
#> California x center_time = 1  -0.0216*** (0.0050)
#> California x center_time = 2  -0.0203*** (0.0045)
#> California x center_time = 3    -0.0222* (0.0100)
#> Fixed-Effects:                -------------------
#> State                                         Yes
#> center_time                                   Yes
#> _____________________________ ___________________
#> S.E.: Clustered                         by: State
#> Observations                                  162
#> R2                                        0.97934
#> Within R2                                 0.00979
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

iplot(cali, pt.join = T)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-2-1.png" width="90%" style="display: block; margin: auto;" />

```r
coefplot(cali)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-2-2.png" width="90%" style="display: block; margin: auto;" />

## Notes

-   [Matching Methods]

    -   Match treatment and control based on pre-treatment observables

    -   Modify SEs appropriately [@heckman1997matching]. It's might be easier to just use the [Doubly Robust DiD] [@sant2020doubly] where you just need either matching or regression to work in order to identify your treatment effect

    -   Whereas the group fixed effects control for the group time-invariant effects, it does not control for selection bias (i.e., certain groups are more likely to be treated than others). Hence, with these backdoor open (i.e., selection bias) between (1) propensity to be treated and (2) dynamics evolution of the outcome post-treatment, matching can potential close these backdoor.

    -   Be careful when matching time-varying covariates because you might encounter "regression to the mean" problem, where pre-treatment periods can have an unusually bad or good time (that is out of the ordinary), then the post-treatment period outcome can just be an artifact of the regression to the mean [@daw2018matching]. This problem is not of concern to time-invariant variables.

    -   Matching and DiD can use pre-treatment outcomes to correct for selection bias. From real world data and simulation, [@chabe2015analysis] found that matching generally underestimates the average causal effect and gets closer to the true effect with more number of pre-treatment outcomes. When selection bias is symmetric around the treatment date, DID is still consistent when implemented symmetrically (i.e., the same number of period before and after treatment). In cases where selection bias is asymmetric, the MC simulations show that Symmetric DiD still performs better than Matching.

    -   Forward DID is a simple algo that helps choose control units [@li2024frontiers].

-   It's always good to show results with and without controls because

    -   If the controls are fixed within group or within time, then those should be absorbed under those fixed effects

    -   If the controls are dynamic across group and across, then your parallel trends assumption is not plausible.

-   Under causal inference, $R^2$ is not so important.

For count data, one can use the fixed-effects Poisson pseudo-maximum likelihood estimator (PPML) [@athey2006identification, @puhani2012treatment] (For applied papers, see @burtch2018can in management and @he2021end in marketing). This also allows for robust standard errors under over-dispersion [@wooldridge1999quasi].

-   This estimator outperforms a log OLS when data have many 0s[@silva2011further], since log-OLS can produce biased estimates [@o2010not] under heteroskedascity [@silva2006log].

-   For those thinking of negative binomial with fixed effects, there isn't an estimator right now [@allison20027].

For [Zero-valued Outcomes], we have to distinguish the treatment effect on the intensive (outcome: 10 to 11) vs. extensive margins (outcome: 0 to 1), and we can't readily interpret the treatment coefficient of log-transformed outcome regression as percentage change [@chen2023logs]. Alternatively, we can either focus on

-   **Proportional treatment effects**: $\theta_{ATT\%} = \frac{E(Y_{it}(1) | D_i = 1, Post_t = 1) - E(Y_{it}(0) |D_i = 1, Post_t = 1)}{E(Y_{it}(0) | D_i = 1 , Post_t = 1}$ (i.e., percentage change in treated group's average post-treatment outcome). Instead of relying on the parallel trends assumption in levels, we could also rely on parallel trends assumption in ratio [@wooldridge2023simple].

    -   We can use Poisson QMLE to estimate the treatment effect: $Y_{it} = \exp(\beta_0 + D_i \times \beta_1 Post_t + \beta_2 D_i + \beta_3 Post_t + X_{it}) \epsilon_{it}$ and $\hat{\theta}_{ATT \%} = \exp(\hat{\beta}_1-1)$.

    -   To examine the parallel trends assumption in ratio holds, we can also estimate a dynamic version of the Poisson QMLE: $Y_{it} = \exp(\lambda_t + \beta_2 D_i + \sum_{r \neq -1} \beta_r D_i \times (RelativeTime_t = r)$, we would expect $\exp(\hat{\beta_r}) - 1 = 0$ for $r < 0$.

    -   Even if we see the plot of these coefficients are 0, we still should run sensitivity analysis [@rambachan2023more] to examine violation of this assumption (see [Prior Parallel Trends Test](#prior-parallel-trends-test)).

-   **Log Effects with Calibrated Extensive-margin value**: due to problem with the mean value interpretation of the proportional treatment effects with outcomes that are heavy-tailed, we might be interested in the extensive margin effect. Then, we can explicit model how much weight we put on the intensive vs. extensive margin [@chen2023logs, p. 39].

1.  **Proportional treatment effects**


```r
set.seed(123) # For reproducibility

n <- 500 # Number of observations per group (treated and control)
# Generating IDs for a panel setup
ID <- rep(1:n, times = 2)

# Defining groups and periods
Group <- rep(c("Control", "Treated"), each = n)
Time <- rep(c("Before", "After"), times = n)
Treatment <- ifelse(Group == "Treated", 1, 0)
Post <- ifelse(Time == "After", 1, 0)

# Step 1: Generate baseline outcomes with a zero-inflated model
lambda <- 20 # Average rate of occurrence
zero_inflation <- 0.5 # Proportion of zeros
Y_baseline <-
    ifelse(runif(2 * n) < zero_inflation, 0, rpois(2 * n, lambda))

# Step 2: Apply DiD treatment effect on the treated group in the post-treatment period
Treatment_Effect <- Treatment * Post
Y_treatment <-
    ifelse(Treatment_Effect == 1, rpois(n, lambda = 2), 0)

# Incorporating a simple time trend, ensuring outcomes are non-negative
Time_Trend <- ifelse(Time == "After", rpois(2 * n, lambda = 1), 0)

# Step 3: Combine to get the observed outcomes
Y_observed <- Y_baseline + Y_treatment + Time_Trend

# Ensure no negative outcomes after the time trend
Y_observed <- ifelse(Y_observed < 0, 0, Y_observed)

# Create the final dataset
data <-
    data.frame(
        ID = ID,
        Treatment = Treatment,
        Period = Post,
        Outcome = Y_observed
    )

# Viewing the first few rows of the dataset
head(data)
#>   ID Treatment Period Outcome
#> 1  1         0      0       0
#> 2  2         0      1      25
#> 3  3         0      0       0
#> 4  4         0      1      20
#> 5  5         0      0      19
#> 6  6         0      1       0
```


```r
library(fixest)
res_pois <-
    fepois(Outcome ~ Treatment + Period + Treatment * Period,
           data = data,
           vcov = "hetero")
etable(res_pois)
#>                             res_pois
#> Dependent Var.:              Outcome
#>                                     
#> Constant           2.249*** (0.0717)
#> Treatment           0.1743. (0.0932)
#> Period               0.0662 (0.0960)
#> Treatment x Period   0.0314 (0.1249)
#> __________________ _________________
#> S.E. type          Heteroskeda.-rob.
#> Observations                   1,000
#> Squared Cor.                 0.01148
#> Pseudo R2                    0.00746
#> BIC                         15,636.8
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Average percentage change
exp(coefficients(res_pois)["Treatment:Period"]) - 1
#> Treatment:Period 
#>       0.03191643

# SE using delta method
exp(coefficients(res_pois)["Treatment:Period"]) *
    sqrt(res_pois$cov.scaled["Treatment:Period", "Treatment:Period"])
#> Treatment:Period 
#>        0.1288596
```

In this example, the DID coefficient is not significant. However, say that it's significant, we can interpret the coefficient as 3 percent increase in posttreatment period due to the treatment.


```r
library(fixest)

base_did_log0 <- base_did |> 
    mutate(y = if_else(y > 0, y, 0))

res_pois_es <-
    fepois(y ~ x1 + i(period, treat, 5) | id + period,
           data = base_did_log0,
           vcov = "hetero")

etable(res_pois_es)
#>                            res_pois_es
#> Dependent Var.:                      y
#>                                       
#> x1                  0.1895*** (0.0108)
#> treat x period = 1    -0.2769 (0.3545)
#> treat x period = 2    -0.2699 (0.3533)
#> treat x period = 3     0.1737 (0.3520)
#> treat x period = 4    -0.2381 (0.3249)
#> treat x period = 6     0.3724 (0.3086)
#> treat x period = 7    0.7739* (0.3117)
#> treat x period = 8    0.5028. (0.2962)
#> treat x period = 9   0.9746** (0.3092)
#> treat x period = 10  1.310*** (0.3193)
#> Fixed-Effects:      ------------------
#> id                                 Yes
#> period                             Yes
#> ___________________ __________________
#> S.E. type           Heteroskedas.-rob.
#> Observations                     1,080
#> Squared Cor.                   0.51131
#> Pseudo R2                      0.34836
#> BIC                            5,868.8
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
iplot(res_pois_es)
```

<img src="29-dif-in-dif_files/figure-html/estiamte teh proprortion treatment effects for event-study form-1.png" width="90%" style="display: block; margin: auto;" />

This parallel trend is the "ratio" version as in @wooldridge2023simple :

$$
\frac{E(Y_{it}(0) |D_i = 1, Post_t = 1)}{E(Y_{it}(0) |D_i = 1, Post_t = 0)} = \frac{E(Y_{it}(0) |D_i = 0, Post_t = 1)}{E(Y_{it}(0) |D_i =0, Post_t = 0)}
$$

which means without treatment, the average percentage change in the mean outcome for treated group is identical to that of the control group.

2.  **Log Effects with Calibrated Extensive-margin value**

If we want to study the treatment effect on a concave transformation of the outcome that is less influenced by those in the distribution's tail, then we can perform this analysis.

Steps:

1.  Normalize the outcomes such that 1 represents the minimum non-zero and positve value (i.e., divide the outcome by its minimum non-zero and positive value).
2.  Estimate the treatment effects for the new outcome

$$
m(y) =
\begin{cases}
\log(y) & \text{for } y >0 \\
-x & \text{for } y = 0
\end{cases}
$$

The choice of $x$ depends on what the researcher is interested in:

+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Value of $x$ | Interest                                                                                                                                                                        |
+==============+=================================================================================================================================================================================+
| $x = 0$      | The treatment effect in logs where all zero-valued outcomes are set to equal the minimum non-zero value (i.e., we exclude the extensive-margin change between 0 and $y_{min}$ ) |
+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| $x>0$        | Setting the change between 0 and $y_{min}$ to be valued as the equivalent of a $x$ log point change along the intensive margin.                                                 |
+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


```r
library(fixest)
base_did_log0_cali <- base_did_log0 |> 
    # get min 
    mutate(min_y = min(y[y > 0])) |> 
    
    # normalized the outcome 
    mutate(y_norm = y / min_y)

my_regression <-
    function(x) {
        base_did_log0_cali <-
            base_did_log0_cali %>% mutate(my = ifelse(y_norm == 0,-x,
                                                      log(y_norm)))
        my_reg <-
            feols(
                fml = my ~ x1 + i(period, treat, 5) | id + period,
                data = base_did_log0_cali,
                vcov = "hetero"
            )
        
        return(my_reg)
    }

xvec <- c(0, .1, .5, 1, 3)
reg_list <- purrr::map(.x = xvec, .f = my_regression)


iplot(reg_list, 
      pt.col =  1:length(xvec),
      pt.pch = 1:length(xvec))
legend("topleft", 
       col = 1:length(xvec),
       pch = 1:length(xvec),
       legend = as.character(xvec))
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-3-1.png" width="90%" style="display: block; margin: auto;" />

```r


etable(
    reg_list,
    headers = list("Extensive-margin value (x)" = as.character(xvec)),
    digits = 2,
    digits.stats = 2
)
#>                                   model 1        model 2        model 3
#> Extensive-margin value (x)              0            0.1            0.5
#> Dependent Var.:                        my             my             my
#>                                                                        
#> x1                         0.43*** (0.02) 0.44*** (0.02) 0.46*** (0.03)
#> treat x period = 1           -0.92 (0.67)   -0.94 (0.69)    -1.0 (0.73)
#> treat x period = 2           -0.41 (0.66)   -0.42 (0.67)   -0.43 (0.71)
#> treat x period = 3           -0.34 (0.67)   -0.35 (0.68)   -0.38 (0.73)
#> treat x period = 4            -1.0 (0.67)    -1.0 (0.68)    -1.1 (0.73)
#> treat x period = 6            0.44 (0.66)    0.44 (0.67)    0.45 (0.72)
#> treat x period = 7            1.1. (0.64)    1.1. (0.65)    1.2. (0.70)
#> treat x period = 8            1.1. (0.64)    1.1. (0.65)     1.1 (0.69)
#> treat x period = 9           1.7** (0.65)   1.7** (0.66)    1.8* (0.70)
#> treat x period = 10         2.4*** (0.62)  2.4*** (0.63)  2.5*** (0.68)
#> Fixed-Effects:             -------------- -------------- --------------
#> id                                    Yes            Yes            Yes
#> period                                Yes            Yes            Yes
#> __________________________ ______________ ______________ ______________
#> S.E. type                  Heterosk.-rob. Heterosk.-rob. Heterosk.-rob.
#> Observations                        1,080          1,080          1,080
#> R2                                   0.43           0.43           0.43
#> Within R2                            0.26           0.26           0.25
#> 
#>                                   model 4        model 5
#> Extensive-margin value (x)              1              3
#> Dependent Var.:                        my             my
#>                                                         
#> x1                         0.49*** (0.03) 0.62*** (0.04)
#> treat x period = 1            -1.1 (0.79)     -1.5 (1.0)
#> treat x period = 2           -0.44 (0.77)   -0.51 (0.99)
#> treat x period = 3           -0.43 (0.78)    -0.60 (1.0)
#> treat x period = 4            -1.2 (0.78)     -1.5 (1.0)
#> treat x period = 6            0.45 (0.77)     0.46 (1.0)
#> treat x period = 7             1.2 (0.75)     1.3 (0.97)
#> treat x period = 8             1.2 (0.74)     1.3 (0.96)
#> treat x period = 9            1.8* (0.75)    2.1* (0.97)
#> treat x period = 10         2.7*** (0.73)  3.2*** (0.94)
#> Fixed-Effects:             -------------- --------------
#> id                                    Yes            Yes
#> period                                Yes            Yes
#> __________________________ ______________ ______________
#> S.E. type                  Heterosk.-rob. Heterosk.-rob.
#> Observations                        1,080          1,080
#> R2                                   0.42           0.41
#> Within R2                            0.25           0.24
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

We have the dynamic treatment effects for different hypothesized extensive-margin value of $x \in (0, .1, .5, 1, 3, 5)$

The first column is when the zero-valued outcome equal to $y_{min, y>0}$ (i.e., there is no different between the minimum outcome and zero outcome - $x = 0$)

For this particular example, as the extensive margin increases, we see an increase in the effect magnitude. The second column is when we assume an extensive-margin change from 0 to $y_{min, y >0}$ is equivalent to a 10 (i.e., $0.1 \times 100$) log point change along the intensive margin.

## Standard Errors

Serial correlation is a big problem in DiD because [@bertrand2004much]

1.  DiD often uses long time series
2.  Outcomes are often highly positively serially correlated
3.  Minimal variation in the treatment variable over time within a group (e.g., state).

To overcome this problem:

-   Using parametric correction (standard AR correction) is not good.
-   Using nonparametric (e.g., **block bootstrap**- keep all obs from the same group such as state together) is good when number of groups is large.
-   Remove time series dimension (i.e., aggregate data into 2 periods: pre and post). This still works with small number of groups (See [@donald2007inference] for more notes on small-sample aggregation).
-   Empirical and arbitrary variance-covariance matrix corrections work only in large samples.

## Examples

Example by [Philipp Leppert](https://rpubs.com/phle/r_tutorial_difference_in_differences) replicating [Card and Krueger (1994)](https://davidcard.berkeley.edu/data_sets.html)

Example by [Anthony Schmidt](https://bookdown.org/aschmi11/causal_inf/difference-in-differences.html)

### Example by @doleac2020unintended

-   The purpose of banning a checking box for ex-criminal was banned because we thought that it gives more access to felons

-   Even if we ban the box, employers wouldn't just change their behaviors. But then the unintended consequence is that employers statistically discriminate based on race

3 types of ban the box

1.  Public employer only
2.  Private employer with government contract
3.  All employers

Main identification strategy

-   If any county in the Metropolitan Statistical Area (MSA) adopts ban the box, it means the whole MSA is treated. Or if the state adopts "ban the ban," every county is treated

Under [Simple Dif-n-dif]

$$ Y_{it} = \beta_0 + \beta_1 Post_t + \beta_2 treat_i + \beta_2 (Post_t \times Treat_i) + \epsilon_{it} $$

But if there is no common post time, then we should use [Staggered Dif-n-dif]

$$ \begin{aligned} E_{imrt} &= \alpha + \beta_1 BTB_{imt} W_{imt} + \beta_2 BTB_{mt} + \beta_3 BTB_{mt} H_{imt}\\  &+ \delta_m + D_{imt} \beta_5 + \lambda_{rt} + \delta_m\times f(t) \beta_7 + e_{imrt} \end{aligned} $$

where

-   $i$ = person; $m$ = MSA; $r$ = region (US regions e.g., Midwest) ; $r$ = region; $t$ = year

-   $W$ = White; $B$ = Black; $H$ = Hispanic

-   $\beta_1 BTB_{imt} W_{imt} + \beta_2 BTB_{mt} + \beta_3 BTB_{mt} H_{imt}$ are the 3 dif-n-dif variables ($BTB$ = "ban the box")

-   $\delta_m$ = dummy for MSI

-   $D_{imt}$ = control for people

-   $\lambda_{rt}$ = region by time fixed effect

-   $\delta_m \times f(t)$ = linear time trend within MSA (but we should not need this if we have good pre-trend)

If we put $\lambda_r - \lambda_t$ (separately) we will more broad fixed effect, while $\lambda_{rt}$ will give us deeper and narrower fixed effect.

Before running this model, we have to drop all other races. And $\beta_1, \beta_2, \beta_3$ are not collinear because there are all interaction terms with $BTB_{mt}$

If we just want to estimate the model for black men, we will modify it to be

$$ E_{imrt} = \alpha + BTB_{mt} \beta_1 + \delta_m + D_{imt} \beta_5 + \lambda_{rt} + (\delta_m \times f(t)) \beta_7 + e_{imrt} $$

$$ \begin{aligned} E_{imrt} &= \alpha + BTB_{m (t - 3t)} \theta_1 + BTB_{m(t-2)} \theta_2 + BTB_{mt} \theta_4 \\ &+ BTB_{m(t+1)}\theta_5 + BTB_{m(t+2)}\theta_6 + BTB_{m(t+3t)}\theta_7 \\ &+ [\delta_m + D_{imt}\beta_5 + \lambda_r + (\delta_m \times (f(t))\beta_7 + e_{imrt}] \end{aligned} $$

We have to leave $BTB_{m(t-1)}\theta_3$ out for the category would not be perfect collinearity

So the year before BTB ($\theta_1, \theta_2, \theta_3$) should be similar to each other (i.e., same pre-trend). Remember, we only run for places with BTB.

If $\theta_2$ is statistically different from $\theta_3$ (baseline), then there could be a problem, but it could also make sense if we have pre-trend announcement.

### Example from [Princeton](https://www.princeton.edu/~otorres/DID101R.pdf)


```r
library(foreign)
mydata = read.dta("http://dss.princeton.edu/training/Panel101.dta") %>%
    # create a dummy variable to indicate the time when the treatment started
    dplyr::mutate(time = ifelse(year >= 1994, 1, 0)) %>%
    # create a dummy variable to identify the treatment group
    dplyr::mutate(treated = ifelse(country == "E" |
                                country == "F" | country == "G" ,
                            1,
                            0)) %>%
    # create an interaction between time and treated
    dplyr::mutate(did = time * treated)
```

estimate the DID estimator


```r
didreg = lm(y ~ treated + time + did, data = mydata)
summary(didreg)
#> 
#> Call:
#> lm(formula = y ~ treated + time + did, data = mydata)
#> 
#> Residuals:
#>        Min         1Q     Median         3Q        Max 
#> -9.768e+09 -1.623e+09  1.167e+08  1.393e+09  6.807e+09 
#> 
#> Coefficients:
#>               Estimate Std. Error t value Pr(>|t|)  
#> (Intercept)  3.581e+08  7.382e+08   0.485   0.6292  
#> treated      1.776e+09  1.128e+09   1.575   0.1200  
#> time         2.289e+09  9.530e+08   2.402   0.0191 *
#> did         -2.520e+09  1.456e+09  -1.731   0.0882 .
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 2.953e+09 on 66 degrees of freedom
#> Multiple R-squared:  0.08273,	Adjusted R-squared:  0.04104 
#> F-statistic: 1.984 on 3 and 66 DF,  p-value: 0.1249
```

The `did` coefficient is the differences-in-differences estimator. Treat has a negative effect

### Example by @card1993minimum

found that increase in minimum wage increases employment

Experimental Setting:

-   New Jersey (treatment) increased minimum wage

-   Penn (control) did not increase minimum wage

|           |     | After | Before |                   |
|-----------|-----|-------|--------|-------------------|
| Treatment | NJ  | A     | B      | A - B             |
| Control   | PA  | C     | D      | C - D             |
|           |     | A - C | B - D  | (A - B) - (C - D) |

where

-   A - B = treatment effect + effect of time (additive)

-   C - D = effect of time

-   (A - B) - (C - D) = dif-n-dif

**The identifying assumptions**:

-   Can't have **switchers**

-   PA is the control group

    -   is a good counter factual

    -   is what NJ would look like if they hadn't had the treatment

$$
Y_{jt} = \beta_0 + NJ_j \beta_1 + POST_t \beta_2 + (NJ_j \times POST_t)\beta_3+ X_{jt}\beta_4 + \epsilon_{jt}
$$

where

-   $j$ = restaurant

-   $NJ$ = dummy where $1 = NJ$, and $0 = PA$

-   $POST$ = dummy where $1 = post$, and $0 = pre$

Notes:

-   We don't need $\beta_4$ in our model to have unbiased $\beta_3$, but including it would give our coefficients efficiency

-   If we use $\Delta Y_{jt}$ as the dependent variable, we don't need $POST_t \beta_2$ anymore

-   Alternative model specification is that the authors use NJ high wage restaurant as control group (still choose those that are close to the border)

-   The reason why they can't control for everything (PA + NJ high wage) is because it's hard to interpret the causal treatment

-   Dif-n-dif utilizes similarity in pretrend of the dependent variables. However, this is neither a necessary nor sufficient for the identifying assumption.

    -   It's not sufficient because they can have multiple treatments (technically, you could include more control, but your treatment can't interact)

    -   It's not necessary because trends can be parallel after treatment

-   However, we can't never be certain; we just try to find evidence consistent with our theory so that dif-n-dif can work.

-   Notice that we don't need before treatment the **levels of the dependent variable** to be the same (e.g., same wage average in both NJ and PA), dif-n-dif only needs **pre-trend (i.e., slope)** to be the same for the two groups.

### Example by @butcher2014effects

Theory:

-   Highest achieving students are usually in hard science. Why?

    -   Hard to give students students the benefit of doubt for hard science

    -   How unpleasant and how easy to get a job. Degrees with lower market value typically want to make you feel more pleasant

Under OLS

$$
E_{ij} = \beta_0 + X_i \beta_1 + G_j \beta_2 + \epsilon_{ij}
$$

where

-   $X_i$ = student attributes

-   $\beta_2$ = causal estimate (from grade change)

-   $E_{ij}$ = Did you choose to enroll in major $j$

-   $G_j$ = grade given in major $j$

Examine $\hat{\beta}_2$

-   Negative bias: Endogenous response because department with lower enrollment rate will give better grade

-   Positive bias: hard science is already having best students (i.e., ability), so if they don't their grades can be even lower

Under dif-n-dif

$$
Y_{idt} = \beta_0 + POST_t \beta_1 + Treat_d \beta_2 + (POST_t \times Treat_d)\beta_3 + X_{idt} + \epsilon_{idt}
$$

where

-   $Y_{idt}$ = grade average

+--------------+-----------------------------------+----------+----------+-------------+
|              | Intercept                         | Treat    | Post     | Treat\*Post |
+==============+===================================+==========+==========+=============+
| Treat Pre    | 1                                 | 1        | 0        | 0           |
+--------------+-----------------------------------+----------+----------+-------------+
| Treat Post   | 1                                 | 1        | 1        | 1           |
+--------------+-----------------------------------+----------+----------+-------------+
| Control Pre  | 1                                 | 0        | 0        | 0           |
+--------------+-----------------------------------+----------+----------+-------------+
| Control Post | 1                                 | 0        | 1        | 0           |
+--------------+-----------------------------------+----------+----------+-------------+
|              | Average for pre-control $\beta_0$ |          |          |             |
+--------------+-----------------------------------+----------+----------+-------------+

A more general specification of the dif-n-dif is that

$$
Y_{idt} = \alpha_0 + (POST_t \times Treat_d) \alpha_1 + \theta_d + \delta_t + X_{idt} + u_{idt}
$$

where

-   $(\theta_d + \delta_t)$ richer , more df than $Treat_d \beta_2 + Post_t \beta_1$ (because fixed effects subsume Post and treat)

-   $\alpha_1$ should be equivalent to $\beta_3$ (if your model assumptions are correct)

## One Difference

The regression formula is as follows [@liaukonyte2023frontiers]:

$$
y_{ut} = \beta \text{Post}_t + \gamma_u + \gamma_w(t) + \gamma_l + \gamma_g(u)p(t) + \epsilon_{ut}
$$

where

-   $y_{ut}$: Outcome of interest for unit u in time t.
-   $\text{Post}_t$: Dummy variable representing a specific post-event period.
-   $\beta$: Coefficient measuring the average change in the outcome after the event relative to the pre-period.
-   $\gamma_u$: Fixed effects for each unit.
-   $\gamma_w(t)$: Time-specific fixed effects to account for periodic variations.
-   $\gamma_l$: Dummy variable for a specific significant period (e.g., a major event change).
-   $\gamma_g(u)p(t)$: Group x period fixed effects for flexible trends that may vary across different categories (e.g., geographical regions) and periods.
-   $\epsilon_{ut}$: Error term.

This model can be used to analyze the impact of an event on the outcome of interest while controlling for various fixed effects and time-specific variations, but using units themselves pre-treatment as controls.

## Two-way Fixed-effects

A generalization of the dif-n-dif model is the two-way fixed-effects models where you have multiple groups and time effects. But this is not a designed-based, non-parametric causal estimator [@imai2021use]

When applying TWFE to multiple groups and multiple periods, the supposedly causal coefficient is the weighted average of all two-group/two-period DiD estimators in the data where some of the weights can be negative. More specifically, the weights are proportional to group sizes and treatment indicator's variation in each pair, where units in the middle of the panel have the highest weight.

The canonical/standard TWFE only works when

-   Effects are homogeneous across units and across time periods (i.e., no dynamic changes in the effects of treatment). See [@goodman2021difference; @de2020two; @sun2021estimating; @borusyak2021revisiting] for details. Similarly, it relies on the assumption of **linear additive effects** [@imai2021use]

    -   Have to argue why treatment heterogeneity is not a problem (e.g., plot treatment timing and decompose treatment coefficient using [Goodman-Bacon Decomposition]) know the percentage of observation are never treated (because as the never-treated group increases, the bias of TWFE decreases, with 80% sample to be never-treated, bias is negligible). The problem is worsen when you have long-run effects.

    -   Need to manually drop two relative time periods if everyone is eventually treated (to avoid multicollinearity). Programs might do this randomly and if it chooses to drop a post-treatment period, it will create biases. The choice usually -1, and -2 periods.

    -   Treatment heterogeneity can come in because (1) it might take some time for a treatment to have measurable changes in outcomes or (2) for each period after treatment, the effect can be different (phase in or increasing effects).

-   2 time periods.

Within this setting, TWFE works because, using the baseline (e.g., control units where their treatment status is unchanged across time periods), the comparison can be

-   Good for

    -   Newly treated units vs. control

    -   Newly treated units vs not-yet treated

-   Bad for

    -   Newly treated vs. already treated (because already treated cannot serve as the potential outcome for the newly treated).
    -   Strict exogeneity (i.e., time-varying confounders, feedback from past outcome to treatment) [@imai2019should]
    -   Specific functional forms (i.e., treatment effect homogeneity and no carryover effects or anticipation effects) [@imai2019should]

Note: Notation for this section is consistent with [@arkhangelsky2021double]

$$
Y_{it} = \alpha_i + \lambda_t + \tau W_{it} + \beta X_{it} + \epsilon_{it}
$$

where

-   $Y_{it}$ is the outcome

-   $\alpha_i$ is the unit FE

-   $\lambda_t$ is the time FE

-   $\tau$ is the causal effect of treatment

-   $W_{it}$ is the treatment indicator

-   $X_{it}$ are covariates

When $T = 2$, the TWFE is the traditional DiD model

Under the following assumption, $\hat{\tau}_{OLS}$ is unbiased:

1.  homogeneous treatment effect
2.  parallel trends assumptions
3.  linear additive effects [@imai2021use]

**Remedies for TWFE's shortcomings**

-   [@goodman2021difference]: diagnostic robustness tests of the TWFE DiD and identify influential observations to the DiD estimate ([Goodman-Bacon Decomposition])

-   [@callaway2021difference]: 2-step estimation with a bootstrap procedure that can account for autocorrelation and clustering,

    -   the parameters of interest are the group-time average treatment effects, where each group is defined by when it was first treated ([Multiple periods and variation in treatment timing])

    -   Comparing post-treatment outcomes fo groups treated in a period against a similar group that is never treated (using matching).

    -   Treatment status cannot switch (once treated, stay treated for the rest of the panel)

    -   Package: `did`

-   [@sun2021estimating]: a specialization of [@callaway2021difference] in the event-study context.

    -   They include lags and leads in their design

    -   have cohort-specific estimates (similar to group-time estimates in [@callaway2021difference]

    -   They propose the "interaction-weighted" estimator.

    -   Package: `fixest`

-   [@imai2021use]

    -   Different from [@callaway2021difference] because they allow units to switch in and out of treatment.

    -   Based on matching methods, to have weighted TWFE

    -   Package: `wfe` and `PanelMatch`

-   [@gardner2022two]: two-stage DiD

    -   `did2s`

-   In cases with an unaffected unit (i.e., never-treated), using the exposure-adjusted difference-in-differences estimators can recover the average treatment effect [@de2020two]. However, if you want to see the treatment effect heterogeneity (in cases where the true heterogeneous treatment effects vary by the exposure rate), exposure-adjusted did still fails [@sun2022linear].

-   [@arkhangelsky2021double]: see below

To be robust against

1.  time- and unit-varying effects

We can use the reshaped inverse probability weighting (RIPW)- TWFE estimator

With the following assumptions:

-   SUTVA

-   Binary treatment: $\mathbf{W}_i = (W_{i1}, \dots, W_{it})$ where $\mathbf{W}_i \sim \mathbf{\pi}_i$ generalized propensity score (i.e., each person treatment likelihood follow $\pi$ regardless of the period)

Then, the unit-time specific effect is $\tau_{it} = Y_{it}(1) - Y_{it}(0)$

Then the Doubly Average Treatment Effect (DATE) is

$$
\tau(\xi) = \sum_{T=1}^T \xi_t \left(\frac{1}{n} \sum_{i = 1}^n \tau_{it} \right)
$$

where

-   $\frac{1}{n} \sum_{i = 1}^n \tau_{it}$ is the unweighted effect of treatment across units (i.e., time-specific ATE).

-   $\xi = (\xi_1, \dots, \xi_t)$ are user-specific weights for each time period.

-   This estimand is called DATE because it's weighted (averaged) across both time and units.

A special case of DATE is when both time and unit-weights are equal

$$
\tau_{eq} = \frac{1}{nT} \sum_{t=1}^T \sum_{i = 1}^n \tau_{it} 
$$

Borrowing the idea of inverse propensity-weighted least squares estimator in the cross-sectional case that we reweight the objective function via the treatment assignment mechanism:

$$
\hat{\tau} \triangleq \arg \min_{\tau} \sum_{i = 1}^n (Y_i -\mu - W_i \tau)^2 \frac{1}{\pi_i (W_i)}
$$

where

-   the first term is the least squares objective

-   the second term is the propensity score

In the panel data case, the IPW estimator will be

$$
\hat{\tau}_{IPW} \triangleq \arg \min_{\tau} \sum_{i = 1}^n \sum_{t =1}^T (Y_{i t}-\alpha_i - \lambda_t - W_{it} \tau)^2 \frac{1}{\pi_i (W_i)}
$$

Then, to have DATE that users can specify the structure of time weight, we use reshaped IPW estimator [@arkhangelsky2021double]

$$
\hat{\tau}_{RIPW} (\Pi) \triangleq \arg \min_{\tau} \sum_{i = 1}^n \sum_{t =1}^T (Y_{i t}-\alpha_i - \lambda_t - W_{it} \tau)^2 \frac{\Pi(W_i)}{\pi_i (W_i)}
$$

where it's a function of a data-independent distribution $\Pi$ that depends on the support of the treatment path $\mathbb{S} = \cup_i Supp(W_i)$

This generalization can transform to

-   IPW-TWFE estimator when $\Pi \sim Unif(\mathbb{S})$

-   randomized experiment when $\Pi = \pi_i$

To choose $\Pi$, we don't need to data, we just need possible assignments in your setting.

-   For most practical problems (DiD, staggered, transient), we have closed form solutions

-   For generic solver, we can use nonlinear programming (e..g, BFGS algorithm)

As argued in [@imai2021use] that TWFE is not a non-parametric approach, it can be subjected to incorrect model assumption (i.e., model dependence).

-   Hence, they advocate for matching methods for time-series cross-sectional data [@imai2021use]

-   Use `wfe` and `PanelMatch` to apply their paper.

This package is based on [@somaini2016algorithm]


```r
# dataset
library(bacondecomp)
df <- bacondecomp::castle
```


```r
# devtools::install_github("paulosomaini/xtreg2way")

library(xtreg2way)
# output <- xtreg2way(y,
#                     data.frame(x1, x2),
#                     iid,
#                     tid,
#                     w,
#                     noise = "1",
#                     se = "1")

# equilvalently
output <- xtreg2way(l_homicide ~ post,
                    df,
                    iid = df$state, # group id
                    tid = df$year, # time id
                    # w, # vector of weight
                    se = "1")
output$betaHat
#>                  [,1]
#> l_homicide 0.08181162
output$aVarHat
#>             [,1]
#> [1,] 0.003396724

# to save time, you can use your structure in the 
# last output for a new set of variables
# output2 <- xtreg2way(y, x1, struc=output$struc)
```

Standard errors estimation options

+----------------------+---------------------------------------------------------------------------------------------+
| Set                  | Estimation                                                                                  |
+======================+=============================================================================================+
| `se = "0"`           | Assume homoskedasticity and no within group correlation or serial correlation               |
+----------------------+---------------------------------------------------------------------------------------------+
| `se = "1"` (default) | robust to heteroskadasticity and serial correlation [@arellano1987computing]                |
+----------------------+---------------------------------------------------------------------------------------------+
| `se = "2"`           | robust to heteroskedasticity, but assumes no correlation within group or serial correlation |
+----------------------+---------------------------------------------------------------------------------------------+
| `se = "11"`          | Aerllano SE with df correction performed by Stata xtreg [@somaini2021twfem]                 |
+----------------------+---------------------------------------------------------------------------------------------+

Alternatively, you can also do it manually or with the `plm` package, but you have to be careful with how the SEs are estimated


```r
library(multiwayvcov) # get vcov matrix 
library(lmtest) # robust SEs estimation

# manual
output3 <- lm(l_homicide ~ post + factor(state) + factor(year),
              data = df)

# get variance-covariance matrix
vcov_tw <- multiwayvcov::cluster.vcov(output3,
                        cbind(df$state, df$year),
                        use_white = F,
                        df_correction = F)

# get coefficients
coeftest(output3, vcov_tw)[2,] 
#>   Estimate Std. Error    t value   Pr(>|t|) 
#> 0.08181162 0.05671410 1.44252696 0.14979397
```


```r
# using the plm package
library(plm)

output4 <- plm(l_homicide ~ post, 
               data = df, 
               index = c("state", "year"), 
               model = "within", 
               effect = "twoways")

# get coefficients
coeftest(output4, vcov = vcovHC, type = "HC1")
#> 
#> t test of coefficients:
#> 
#>      Estimate Std. Error t value Pr(>|t|)
#> post 0.081812   0.057748  1.4167   0.1572
```

As you can see, differences stem from SE estimation, not the coefficient estimate.

## Multiple periods and variation in treatment timing

This is an extension of the DiD framework to settings where you have

-   more than 2 time periods

-   different treatment timing

When treatment effects are heterogeneous across time or units, the standard [Two-way Fixed-effects] is inappropriate.

Notation is consistent with `did` [package](https://cran.r-project.org/web/packages/did/vignettes/multi-period-did.html) [@callaway2021difference]

-   $Y_{it}(0)$ is the potential outcome for unit $i$

-   $Y_{it}(g)$ is the potential outcome for unit $i$ in time period $t$ if it's treated in period $g$

-   $Y_{it}$ is the observed outcome for unit $i$ in time period $t$

$$
Y_{it} = 
\begin{cases}
Y_{it} = Y_{it}(0) & \forall i \in \text{never-treated group} \\
Y_{it} = 1\{G_i > t\} Y_{it}(0) +  1\{G_i \le t \}Y_{it}(G_i) & \forall i \in \text{other groups}
\end{cases}
$$

-   $G_i$ is the time period when $i$ is treated

-   $C_i$ is a dummy when $i$ belongs to the **never-treated** group

-   $D_{it}$ is a dummy for whether $i$ is treated in period $t$

**Assumptions**:

-   Staggered treatment adoption: once treated, a unit cannot be untreated (revert)

-   Parallel trends assumptions (conditional on covariates):

    -   Based on never-treated units: $E[Y_t(0)- Y_{t-1}(0)|G= g] = E[Y_t(0) - Y_{t-1}(0)|C=1]$

        -   Without treatment, the average potential outcomes for group $g$ equals the average potential outcomes for the never-treated group (i.e., control group), which means that we have (1) enough data on the never-treated group (2) the control group is similar to the eventually treated group.

    -   Based on not-yet treated units: $E[Y_t(0) - Y_{t-1}(0)|G = g] = E[Y_t(0) - Y_{t-1}(0)|D_s = 0, G \neq g]$

        -   Not-yet treated units by time $s$ ( $s \ge t$) can be used as comparison groups to calculate the average treatment effects for the group first treated in time $g$

        -   Additional assumption: pre-treatment trends across groups [@marcus2021role]

-   Random sampling

-   Irreversibility of treatment (once treated, cannot be untreated)

-   Overlap (the treatment propensity $e \in [0,1]$)

Group-Time ATE

-   This is the equivalent of the average treatment effect in the standard case (2 groups, 2 periods) under multiple time periods.

$$
ATT(g,t) = E[Y_t(g) - Y_t(0) |G = g]
$$

which is the average treatment effect for group $g$ in period $t$

-   Identification: When the parallel trends assumption based on

    -   Never-treated units: $ATT(g,t) = E[Y_t - Y_{g-1} |G = g] - E[Y_t - Y_{g-1}|C=1] \forall t \ge g$

    -   Not-yet-treated units: $ATT(g,t) = E[Y_t - Y_{g-1}|G= g] - E[Y_t - Y_{g-1}|D_t = 0, G \neq g] \forall t \ge g$

-   Identification: when the parallel trends assumption only holds conditional on covariates and based on

    -   Never-treated units: $ATT(g,t) = E[Y_t - Y_{g-1} |X, G = g] - E[Y_t - Y_{g-1}|X, C=1] \forall t \ge g$

    -   Not-yet-treated units: $ATT(g,t) = E[Y_t - Y_{g-1}|X, G= g] - E[Y_t - Y_{g-1}|X, D_t = 0, G \neq g] \forall t \ge g$

    -   This is plausible when you have suspected selection bias that can be corrected by using covariates (i.e., very much similar to matching methods to have plausible parallel trends).

Possible parameters of interest are:

1.  Average treatment effect per group

$$
\theta_S(g) = \frac{1}{\tau - g + 1} \sum_{t = 2}^\tau \mathbb{1} \{ \le t \} ATT(g,t)
$$

2.  Average treatment effect across groups (that were treated) (similar to average treatment effect on the treated in the canonical case)

$$
\theta_S^O := \sum_{g=2}^\tau \theta_S(g) P(G=g)
$$

3.  Average treatment effect dynamics (i.e., average treatment effect for groups that have been exposed to the treatment for $e$ time periods):

$$
\theta_D(e) := \sum_{g=2}^\tau \mathbb{1} \{g + e \le \tau \}ATT(g,g + e) P(G = g|G + e \le \tau)
$$

4.  Average treatment effect in period $t$ for all groups that have treated by period $t$)

$$
\theta_C(t) = \sum_{g=2}^\tau \mathbb{1}\{g \le t\} ATT(g,t) P(G = g|g \le t)
$$

5.  Average treatment effect by calendar time

$$
\theta_C = \frac{1}{\tau-1}\sum_{t=2}^\tau \theta_C(t)
$$

## Staggered Dif-n-dif

See @wing2024designing checklist.

Recommendations by @baker2022much

-   TWFE DiD regressions are suitable for single treatment periods or when treatment effects are homogeneous, provided there's a solid rationale for effect homogeneity.

-   For TWFE staggered DiD, researchers should evaluate bias risks, plot treatment timings to check for variations, and use decompositions like @goodman2021difference when possible. If decompositions aren't feasible (e.g., unbalanced panel), the percentage of never-treated units can indicate bias severity. Expected treatment effect variability should also be discussed.

-   In TWFE staggered DiD event studies, avoid binning time periods without evidence of uniform effects. Use full relative-time indicators, justify reference periods, and be wary of multicollinearity causing bias.

-   To address treatment timing and bias concerns, use alternative estimators like stacked regressions, @sun2021estimating, @callaway2021difference, or separate regressions for each event with "clean" controls.

-   Justify the selection of comparison groups (not-yet treated, last treated, never treated) and ensure the parallel-trends assumption holds, especially when anticipating no effects for certain groups.

Notes:

-   When subjects are treated at different point in time (variation in treatment timing across units), we have to use staggered DiD (also known as DiD event study or dynamic DiD).
-   For design where a treatment is applied and units are exposed to this treatment at all time afterward, see [@athey2022design]

For example, basic design [@stevenson2006bargaining]

$$
\begin{aligned}
Y_{it} &= \sum_k \beta_k Treatment_{it}^k + \sum_i \eta_i  State_i \\
&+ \sum_t \lambda_t Year_t + Controls_{it} + \epsilon_{it}
\end{aligned}
$$

where

-   $Treatment_{it}^k$ is a series of dummy variables equal to 1 if state $i$ is treated $k$ years ago in period $t$

-   SE is usually clustered at the group level (occasionally time level).

-   To avoid collinearity, the period right before treatment is usually chosen to drop.

The more general form of TWFE [@sun2021estimating]:

First, define the relative period bin indicator as

$$
D_{it}^l = \mathbf{1}(t - E_i = l)
$$

where it's an indicator function of unit $i$ being $l$ periods from its first treatment at time $t$

1.  **Static** specification

$$
Y_{it} = \alpha_i + \lambda_t + \mu_g \sum_{l \ge0} D_{it}^l + \epsilon_{it}
$$

where

-   $\alpha_i$ is the the unit FE

-   $\lambda_t$ is the time FE

-   $\mu_g$ is the coefficient of interest $g = [0,T)$

-   we exclude all periods before first adoption.

2.  **Dynamic** specification

$$
Y_{it} = \alpha_i + \lambda_t + \sum_{\substack{l = -K \\ l \neq -1}}^{L} \mu_l D_{it}^l + \epsilon_{it}
$$

where we have to exclude some relative periods to avoid multicollinearity problem (e.g., either period right before treatment, or the treatment period).

In this setting, we try to show that the treatment and control groups are not statistically different (i.e., the coefficient estimates before treatment are not different from 0) to show pre-treatment parallel trends.

However, this two-way fixed effects design has been criticized by @sun2021estimating; @callaway2021difference; @goodman2021difference. When researchers include leads and lags of the treatment to see the long-term effects of the treatment, these leads and lags can be biased by effects from other periods, and pre-trends can falsely arise due to treatment effects heterogeneity.

Applying the new proposed method, finance and accounting researchers find that in many cases, the causal estimates turn out to be null [@baker2022much].

**Assumptions of Staggered DID**

-   **Rollout Exogeneity** (i.e., exogeneity of treatment adoption): if the treatment is randomly implemented over time (i.e., unrelated to variables that could also affect our dependent variables)

    -   Evidence: Regress adoption on pre-treatment variables. And if you find evidence of correlation, include linear trends interacted with pre-treatment variables [@hoynes2009consumption]
    -   Evidence: [@deshpande2019screened, p. 223]
        -   Treatment is random: Regress treatment status at the unit level to all pre-treatment observables. If you have some that are predictive of treatment status, you might have to argue why it's not a worry. At best, you want this.
        -   Treatment timing is random: Conditional on treatment, regress timing of the treatment on pre-treatment observables. At least, you want this.

-   **No confounding events**

-   **Exclusion restrictions**

    -   ***No-anticipation assumption***: future treatment time do not affect current outcomes

    -   ***Invariance-to-history assumption***: the time a unit under treatment does not affect the outcome (i.e., the time exposed does not matter, just whether exposed or not). This presents causal effect of early or late adoption on the outcome.

-   And all the assumptions in listed in the [Multiple periods and variation in treatment timing]

-   Auxiliary assumptions:

    -   Constant treatment effects across units

    -   Constant treatment effect over time

    -   Random sampling

    -   Effect Additivity

Remedies for staggered DiD [@baker2022much]:

-   Each treated cohort is compared to appropriate controls (not-yet-treated, never-treated)

    -   [@goodman2021difference]

    -   [@callaway2021difference] consistent for average ATT. more complicated but also more flexible than [@sun2021estimating]

        -   [@sun2021estimating] (a special case of [@callaway2021difference])

    -   [@de2020two]

    -   [@borusyak2021revisiting]

-   [Stacked DID] (biased but simple):

    -   [@gormley2011growing]

    -   [@cengiz2019effect]

    -   [@deshpande2019screened]

### Stacked DID

Notations following [these slides](https://scholarworks.iu.edu/dspace/bitstream/handle/2022/26875/2021-10-22_wim_wing_did_slides.pdf?sequence=1&isAllowed=y)

$$
Y_{it} = \beta_{FE} D_{it} + A_i + B_t + \epsilon_{it}
$$

where

-   $A_i$ is the group fixed effects

-   $B_t$ is the period fixed effects

Steps

1.  Choose Event Window
2.  Enumerate Sub-experiments
3.  Define Inclusion Criteria
4.  Stack Data
5.  Specify Estimating Equation

**Event Window**

Let

-   $\kappa_a$ be the length of the pre-event window

-   $\kappa_b$ be the length of the post-event window

By setting a common event window for the analysis, we essentially exclude all those events that do not meet this criteria.

**Sub-experiments**

Let $T_1$ be the earliest period in the dataset

$T_T$ be the last period in the dataset

Then, the collection of all policy adoption periods that are under our event window is

$$
\Omega_A = \{ A_i |T_1 + \kappa_a \le A_i \le T_T - \kappa_b\}
$$

where these events exist

-   at least $\kappa_a$ periods after the earliest period

-   at least $\kappa_b$ periods before the last period

Let $d = 1, \dots, D$ be the index column of the sub-experiments in $\Omega_A$

and $\omega_d$ be the event date of the d-th sub-experiment (e.g., $\omega_1$ = adoption date of the 1st experiment)

**Inclusion Criteria**

1.  Valid treated Units
    -   Within sub-experiment $d$, all treated units have the same adoption date

    -   This makes sure a unit can only serve as a treated unit in only 1 sub-experiment
2.  Clean controls
    -   Only units satisfying $A_i >\omega_d + \kappa_b$ are included as controls in sub-experiment d

    -   This ensures controls are only

        -   never treated units

        -   units that are treated in far future

    -   But a unit can be control unit in multiple sub-experiments (need to correct SE)
3.  Valid Time Periods
    -   All observations within sub-experiment d are from time periods within the sub-experiment's event window

    -   This ensures in sub-experiment d, only observations satisfying $\omega_d - \kappa_a \le t \le \omega_d + \kappa_b$ are included


```r
library(did)
library(tidyverse)
library(fixest)

data(base_stagg)

# first make the stacked datasets
# get the treatment cohorts
cohorts <- base_stagg %>%
    select(year_treated) %>%
    # exclude never-treated group
    filter(year_treated != 10000) %>%
    unique() %>%
    pull()

# make formula to create the sub-datasets
getdata <- function(j, window) {
    #keep what we need
    base_stagg %>%
        # keep treated units and all units not treated within -5 to 5
        # keep treated units and all units not treated within -window to window
        filter(year_treated == j | year_treated > j + window) %>%
        # keep just year -window to window
        filter(year >= j - window & year <= j + window) %>%
        # create an indicator for the dataset
        mutate(df = j)
}

# get data stacked
stacked_data <- map_df(cohorts, ~ getdata(., window = 5)) %>%
    mutate(rel_year = if_else(df == year_treated, time_to_treatment, NA_real_)) %>%
    fastDummies::dummy_cols("rel_year", ignore_na = TRUE) %>%
    mutate(across(starts_with("rel_year_"), ~ replace_na(., 0)))

# get stacked value
stacked <-
    feols(
        y ~ `rel_year_-5` + `rel_year_-4` + `rel_year_-3` +
            `rel_year_-2` + rel_year_0 + rel_year_1 + rel_year_2 + rel_year_3 +
            rel_year_4 + rel_year_5 |
            id ^ df + year ^ df,
        data = stacked_data
    )$coefficients

stacked_se = feols(
    y ~ `rel_year_-5` + `rel_year_-4` + `rel_year_-3` +
        `rel_year_-2` + rel_year_0 + rel_year_1 + rel_year_2 + rel_year_3 +
        rel_year_4 + rel_year_5 |
        id ^ df + year ^ df,
    data = stacked_data
)$se

# add in 0 for omitted -1
stacked <- c(stacked[1:4], 0, stacked[5:10])
stacked_se <- c(stacked_se[1:4], 0, stacked_se[5:10])


cs_out <- att_gt(
    yname = "y",
    data = base_stagg,
    gname = "year_treated",
    idname = "id",
    # xformla = "~x1",
    tname = "year"
)
cs <-
    aggte(
        cs_out,
        type = "dynamic",
        min_e = -5,
        max_e = 5,
        bstrap = FALSE,
        cband = FALSE
    )



res_sa20 = feols(y ~ sunab(year_treated, year) |
                     id + year, base_stagg)
sa = tidy(res_sa20)[5:14, ] %>% pull(estimate)
sa = c(sa[1:4], 0, sa[5:10])

sa_se = tidy(res_sa20)[6:15, ] %>% pull(std.error)
sa_se = c(sa_se[1:4], 0, sa_se[5:10])

compare_df_est = data.frame(
    period = -5:5,
    cs = cs$att.egt,
    sa = sa,
    stacked = stacked
)

compare_df_se = data.frame(
    period = -5:5,
    cs = cs$se.egt,
    sa = sa_se,
    stacked = stacked_se
)

compare_df_longer <- compare_df_est %>%
    pivot_longer(!period, names_to = "estimator", values_to = "est") %>%
    
    full_join(compare_df_se %>% 
                  pivot_longer(!period, names_to = "estimator", values_to = "se")) %>%
    
    mutate(upper = est +  1.96 * se,
           lower = est - 1.96 * se)


ggplot(compare_df_longer) +
    geom_ribbon(aes(
        x = period,
        ymin = lower,
        ymax = upper,
        group = estimator
    )) +
    geom_line(aes(
        x = period,
        y = est,
        group = estimator,
        col = estimator
    ),
    linewidth = 1) + 
    causalverse::ama_theme()
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-10-1.png" width="90%" style="display: block; margin: auto;" />

**Stack Data**

Estimating Equation

$$
Y_{itd} = \beta_0 + \beta_1 T_{id} + \beta_2 P_{td} + \beta_3 (T_{id} \times P_{td}) + \epsilon_{itd}
$$

where

-   $T_{id}$ = 1 if unit $i$ is treated in sub-experiment $d$, 0 if control

-   $P_{td}$ = 1 if it's the period after the treatment in sub-experiment $d$

Equivalently,

$$
Y_{itd} = \beta_3 (T_{id} \times P_{td}) + \theta_{id} + \gamma_{td} + \epsilon_{itd}
$$

$\beta_3$ averages all the time-varying effects into a single number (can't see the time-varying effects)

**Stacked Event Study**

Let $YSE_{td} = t - \omega_d$ be the "time since event" variable in sub-experiment $d$

Then, $YSE_{td} = -\kappa_a, \dots, 0, \dots, \kappa_b$ in every sub-experiment

In each sub-experiment, we can fit

$$
Y_{it}^d = \sum_{j = -\kappa_a}^{\kappa_b} \beta_j^d \times 1(TSE_{td} = j) + \sum_{m = -\kappa_a}^{\kappa_b} \delta_j^d (T_{id} \times 1 (TSE_{td} = j)) + \theta_i^d + \epsilon_{it}^d
$$

-   Different set of event study coefficients in each sub-experiment

$$
Y_{itd} = \sum_{j = -\kappa_a}^{\kappa_b} \beta_j \times 1(TSE_{td} = j) + \sum_{m = -\kappa_a}^{\kappa_b} \delta_j (T_{id} \times 1 (TSE_{td} = j)) + \theta_{id} + \epsilon_{itd}
$$

**Clustering**

-   Clustered at the unit x sub-experiment level [@cengiz2019effect]

-   Clustered at the unit level [@deshpande2019screened]

### Goodman-Bacon Decomposition

Paper: [@goodman2021difference]

For an excellent explanation slides by the author, [see](https://www.stata.com/meeting/chicago19/slides/chicago19_Goodman-Bacon.pdf)

Takeaways:

-   A pairwise DID ($\tau$) gets more weight if the change is close to the middle of the study window

-   A pairwise DID ($\tau$) gets more weight if it includes more observations.

Code from `bacondecomp` vignette


```r
library(bacondecomp)
library(tidyverse)
data("castle")
castle <- bacondecomp::castle %>% 
    dplyr::select("l_homicide", "post", "state", "year")
head(castle)
#>   l_homicide post   state year
#> 1   2.027356    0 Alabama 2000
#> 2   2.164867    0 Alabama 2001
#> 3   1.936334    0 Alabama 2002
#> 4   1.919567    0 Alabama 2003
#> 5   1.749841    0 Alabama 2004
#> 6   2.130440    0 Alabama 2005


df_bacon <- bacon(
    l_homicide ~ post,
    data = castle,
    id_var = "state",
    time_var = "year"
)
#>                       type  weight  avg_est
#> 1 Earlier vs Later Treated 0.05976 -0.00554
#> 2 Later vs Earlier Treated 0.03190  0.07032
#> 3     Treated vs Untreated 0.90834  0.08796

# weighted average of the decomposition
sum(df_bacon$estimate * df_bacon$weight)
#> [1] 0.08181162
```

Two-way Fixed effect estimate


```r
library(broom)
fit_tw <- lm(l_homicide ~ post + factor(state) + factor(year),
             data = bacondecomp::castle)
head(tidy(fit_tw))
#> # A tibble: 6 × 5
#>   term                    estimate std.error statistic   p.value
#>   <chr>                      <dbl>     <dbl>     <dbl>     <dbl>
#> 1 (Intercept)               1.95      0.0624    31.2   2.84e-118
#> 2 post                      0.0818    0.0317     2.58  1.02e-  2
#> 3 factor(state)Alaska      -0.373     0.0797    -4.68  3.77e-  6
#> 4 factor(state)Arizona      0.0158    0.0797     0.198 8.43e-  1
#> 5 factor(state)Arkansas    -0.118     0.0810    -1.46  1.44e-  1
#> 6 factor(state)California  -0.108     0.0810    -1.34  1.82e-  1
```

Hence, naive TWFE fixed effect equals the weighted average of the Bacon decomposition (= 0.08).


```r
library(ggplot2)

ggplot(df_bacon) +
    aes(
        x = weight,
        y = estimate,
        # shape = factor(type),
        color = type
    ) +
    labs(x = "Weight", y = "Estimate", shape = "Type") +
    geom_point() +
    causalverse::ama_theme()
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-13-1.png" width="90%" style="display: block; margin: auto;" />

With time-varying controls that can identify variation within-treatment timing group, the"early vs. late" and "late vs. early" estimates collapse to just one estimate (i.e., both treated).

### DID with in and out treatment condition

#### Panel Match

@imai2021use

This case generalizes the staggered adoption setting, allowing units to vary in treatment over time. For $N$ units across $T$ time periods (with potentially unbalanced panels), let $X_{it}$ represent treatment and $Y_{it}$ the outcome for unit $i$ at time $t$. We use the two-way linear fixed effects model:

$$
Y_{it} = \alpha_i + \gamma_t + \beta X_{it} + \epsilon_{it}
$$

for $i = 1, \dots, N$ and $t = 1, \dots, T$. Here, $\alpha_i$ and $\gamma_t$ are unit and time fixed effects. They capture time-invariant unit-specific and unit-invariant time-specific unobserved confounders, respectively. We can express these as $\alpha_i = h(\mathbf{U}_i)$ and $\gamma_t = f(\mathbf{V}_t)$, with $\mathbf{U}_i$ and $\mathbf{V}_t$ being the confounders. The model doesn't assume a specific form for $h(.)$ and $f(.)$, but that they're additive and separable given binary treatment.

The least squares estimate of $\beta$ leverages the covariance in outcome and treatment [@imai2021use, p. 406]. Specifically, it uses the within-unit and within-time variations. Many researchers prefer the two fixed effects (2FE) estimator because it adjusts for both types of unobserved confounders without specific functional-form assumptions, but this is wrong [@imai2019should]. We do need functional-form assumption (i.e., linearity assumption) for the 2FE to work [@imai2021use, p. 406]

-   **Two-Way Matching Estimator**:

    -   It can lead to mismatches; units with the same treatment status get matched when estimating counterfactual outcomes.

    -   Observations need to be matched with opposite treatment status for correct causal effects estimation.

    -   Mismatches can cause attenuation bias.

    -   The 2FE estimator adjusts for this bias using the factor $K$, which represents the net proportion of proper matches between observations with opposite treatment status.

-   **Weighting in 2FE**:

    -   Observation $(i,t)$ is weighted based on how often it acts as a control unit.

    -   The weighted 2FE estimator still has mismatches, but fewer than the standard 2FE estimator.

    -   Adjustments are made based on observations that neither belong to the same unit nor the same time period as the matched observation.

    -   This means there are challenges in adjusting for unit-specific and time-specific unobserved confounders under the two-way fixed effect framework.

-   **Equivalence & Assumptions**:

    -   Equivalence between the 2FE estimator and the DID estimator is dependent on the linearity assumption.

    -   The multi-period DiD estimator is described as an average of two-time-period, two-group DiD estimators applied during changes from control to treatment.

-   **Comparison with DiD**:

    -   In simple settings (two time periods, treatment given to one group in the second period), the standard nonparametric DiD estimator equals the 2FE estimator.

    -   This doesn't hold in multi-period DiD designs where units change treatment status multiple times at different intervals.

    -   Contrary to popular belief, the unweighted 2FE estimator isn't generally equivalent to the multi-period DiD estimator.

    -   While the multi-period DiD can be equivalent to the weighted 2FE, some control observations may have negative regression weights.

-   **Conclusion**:

    -   Justifying the 2FE estimator as the DID estimator isn't warranted without imposing the linearity assumption.

**Application [@imai2021matching]**

-   **Matching Methods**:

    -   Enhance the validity of causal inference.

    -   Reduce model dependence and provide intuitive diagnostics [@ho2007matching]

    -   Rarely utilized in analyzing time series cross-sectional data.

    -   The proposed matching estimators are more robust than the standard two-way fixed effects estimator, which can be biased if mis-specified

    -   Better than synthetic controls (e.g., [@xu2017generalized]) because it needs less data to achieve good performance and and adapt the the context of unit switching treatment status multiple times.

-   Notes:

    -   Potential carryover effects (treatment may have a long-term effect), leading to post-treatment bias.

-   **Proposed Approach**:

    1.  Treated observations are matched with control observations from other units in the same time period with the same treatment history up to a specified number of lags.

    2.  Standard matching and weighting techniques are employed to further refine the matched set.

    3.  Apply a DiD estimator to adjust for time trend.

    4.  The goal is to have treated and matched control observations with similar covariate values.

-   **Assessment**:

    -   The quality of matches is evaluated through covariate balancing.

-   **Estimation**:

    -   Both short-term and long-term average treatment effects on the treated (ATT) are estimated.


```r
library(PanelMatch)
```

**Treatment Variation plot**

-   Visualize the variation of the treatment across space and time

-   Aids in discerning whether the treatment fluctuates adequately over time and units or if the variation is primarily clustered in a subset of data.


```r
DisplayTreatment(
    unit.id = "wbcode2",
    time.id = "year",
    legend.position = "none",
    xlab = "year",
    ylab = "Country Code",
    treatment = "dem",
    
    hide.x.tick.label = TRUE, hide.y.tick.label = TRUE, 
    # dense.plot = TRUE,
    data = dem
)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-15-1.png" width="90%" style="display: block; margin: auto;" />

1.  Select $F$ (i.e., the number of leads - time periods after treatment). Driven by what authors are interested in estimating:

-   $F = 0$ is the contemporaneous effect (short-term effect)

-   $F = n$ is the the treatment effect on the outcome two time periods after the treatment. (cumulative or long-term effect)

2.  Select $L$ (number of lags to adjust).

-   Driven by the identification assumption.

-   Balances bias-variance tradeoff.

-   Higher $L$ values increase credibility but reduce efficiency by limiting potential matches.

**Model assumption**:

-   No spillover effect assumed.

-   Carryover effect allowed up to $L$ periods.

-   Potential outcome for a unit depends neither on others' treatment status nor on its past treatment after $L$ periods.

After defining causal quantity with parameters $L$ and $F$.

-   Focus on the average treatment effect of treatment status change.
-   $\delta(F,L)$ is the average causal effect of treatment change (ATT), $F$ periods post-treatment, considering treatment history up to $L$ periods.
-   Causal quantity considers potential future treatment reversals, meaning treatment could revert to control before outcome measurement.

Also possible to estimate the average treatment effect of treatment reversal on the reversed (ART).

Choose $L,F$ based on specific needs.

-   A large $L$ value:

    -   Increases the credibility of the limited carryover effect assumption.

    -   Allows more past treatments (up to $t−L$) to influence the outcome $Y_{i,t+F}$.

    -   Might reduce the number of matches and lead to less precise estimates.

-   Selecting an appropriate number of lags

    -   Researchers should base this choice on substantive knowledge.

    -   Sensitivity of empirical results to this choice should be examined.

-   The choice of $F$ should be:

    -   Substantively motivated.

    -   Decides whether the interest lies in short-term or long-term causal effects.

    -   A large $F$ value can complicate causal effect interpretation, especially if many units switch treatment status during the $F$ lead time period.

**Identification Assumption**

-   Parallel trend assumption conditioned on treatment, outcome (excluding immediate lag), and covariate histories.

-   Doesn't require strong unconfoundedness assumption.

-   Cannot account for unobserved time-varying confounders.

-   Essential to examine outcome time trends.

    -   Check if they're parallel between treated and matched control units using pre-treatment data

-   **Constructing the Matched Sets**:

    -   For each treated observation, create matched control units with identical treatment history from $t−L$ to $t−1$.

    -   Matching based on treatment history helps control for carryover effects.

    -   Past treatments often act as major confounders, but this method can correct for it.

    -   Exact matching on time period adjusts for time-specific unobserved confounders.

    -   Unlike staggered adoption methods, units can change treatment status multiple times.

    -   Matched set allows treatment switching in and out of treatment

-   **Refining the Matched Sets**:

    -   Initially, matched sets adjust only for treatment history.

    -   Parallel trend assumption requires adjustments for other confounders like past outcomes and covariates.

    -   Matching methods:

        -   Match each treated observation with up to $J$ control units.

        -   Distance measures like Mahalanobis distance or propensity score can be used.

        -   Match based on estimated propensity score, considering pretreatment covariates.

        -   Refined matched set selects most similar control units based on observed confounders.

    -   Weighting methods:

        -   Assign weight to each control unit in a matched set.

        -   Weights prioritize more similar units.

        -   Inverse propensity score weighting method can be applied.

        -   Weighting is a more generalized method than matching.

**The Difference-in-Differences Estimator**:

-   Using refined matched sets, the ATT (Average Treatment Effect on the Treated) of policy change is estimated.

-   For each treated observation, estimate the counterfactual outcome using the weighted average of control units in the refined set.

-   The DiD estimate of the ATT is computed for each treated observation, then averaged across all such observations.

-   For noncontemporaneous treatment effects where $F > 0$:

    -   The ATT doesn't specify future treatment sequence.

    -   Matched control units might have units receiving treatment between time $t$ and $t + F$.

    -   Some treated units could return to control conditions between these times.

**Checking Covariate Balance**:

-   The proposed methodology offers the advantage of checking covariate balance between treated and matched control observations.

-   This check helps to see if treated and matched control observations are comparable with respect to observed confounders.

-   Once matched sets are refined, covariate balance examination becomes straightforward.

-   Examine the mean difference of each covariate between a treated observation and its matched controls for each pretreatment time period.

-   Standardize this difference using the standard deviation of each covariate across all treated observations in the dataset.

-   Aggregate this covariate balance measure across all treated observations for each covariate and pretreatment time period.

-   Examine balance for lagged outcome variables over multiple pretreatment periods and time-varying covariates.

    -   This helps evaluate the validity of the parallel trend assumption underlying the proposed DiD estimator.

**Relations with Linear Fixed Effects Regression Estimators**:

-   The standard DiD estimator is equivalent to the linear two-way fixed effects regression estimator when:

    -   Only two time periods exist.

    -   Treatment is given to some units exclusively in the second period.

-   This equivalence doesn't extend to multiperiod DiD designs, where:

    -   More than two time periods are considered.

    -   Units might receive treatment multiple times.

-   Despite this, many researchers relate the use of the two-way fixed effects estimator to the DiD design.

**Standard Error Calculation**:

-   Approach:

    -   Condition on the weights implied by the matching process.

    -   These weights denote how often an observation is utilized in matching [@imbens2015causal]

-   Context:

    -   Analogous to the conditional variance seen in regression models.

    -   Resulting standard errors don't factor in uncertainties around the matching procedure.

    -   They can be viewed as a measure of uncertainty conditional upon the matching process [@ho2007matching].

**Key Findings**:

-   Even in conditions favoring OLS, the proposed matching estimator displayed higher robustness to omitted relevant lags than the linear regression model with fixed effects.

-   The robustness offered by matching came at a cost - reduced statistical power.

-   This emphasizes the classic statistical tradeoff between bias (where matching has an advantage) and variance (where regression models might be more efficient).

**Data Requirements**

-   The treatment variable is binary:

    -   0 signifies "assignment" to control.

    -   1 signifies assignment to treatment.

-   Variables identifying units in the data must be: Numeric or integer.

-   Variables identifying time periods should be: Consecutive numeric/integer data.

-   Data format requirement: Must be provided as a standard `data.frame` object.

Basic functions:

1.  Utilize treatment histories to create matching sets of treated and control units.

2.  Refine these matched sets by determining weights for each control unit in the set.

    -   Units with higher weights have a larger influence during estimations.

**Matching on Treatment History**:

-   Goal is to match units transitioning from untreated to treated status with control units that have similar past treatment histories.

-   Setting the Quantity of Interest (`qoi =`)

    -   `att` average treatment effect on treated units

    -   `atc` average treatment effect of treatment on the control units

    -   `art` average effect of treatment reversal for units that experience treatment reversal

    -   `ate` average treatment effect


```r
library(PanelMatch)
# All examples follow the package's vignette
# Create the matched sets
PM.results.none <-
    PanelMatch(
        lag = 4,
        time.id = "year",
        unit.id = "wbcode2",
        treatment = "dem",
        refinement.method = "none",
        data = dem,
        match.missing = TRUE,
        size.match = 5,
        qoi = "att",
        outcome.var = "y",
        lead = 0:4,
        forbid.treatment.reversal = FALSE,
        use.diagonal.variance.matrix = TRUE
    )

# visualize the treated unit and matched controls
DisplayTreatment(
    unit.id = "wbcode2",
    time.id = "year",
    legend.position = "none",
    xlab = "year",
    ylab = "Country Code",
    treatment = "dem",
    data = dem,
    matched.set = PM.results.none$att[1],
    # highlight the particular set
    show.set.only = TRUE
)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-16-1.png" width="90%" style="display: block; margin: auto;" />

Control units and the treated unit have identical treatment histories over the lag window (1988-1991)


```r
DisplayTreatment(
    unit.id = "wbcode2",
    time.id = "year",
    legend.position = "none",
    xlab = "year",
    ylab = "Country Code",
    treatment = "dem",
    data = dem,
    matched.set = PM.results.none$att[2],
    # highlight the particular set
    show.set.only = TRUE
)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-17-1.png" width="90%" style="display: block; margin: auto;" />

This set is more limited than the first one, but we can still see that we have exact past histories.

-   **Refining Matched Sets**

    -   Refinement involves assigning weights to control units.

    -   Users must:

        1.  Specify a method for calculating unit similarity/distance.

        2.  Choose variables for similarity/distance calculations.

-   **Select a Refinement Method**

    -   Users determine the refinement method via the **`refinement.method`** argument.

    -   Options include:

        -   `mahalanobis`

        -   `ps.match`

        -   `CBPS.match`

        -   `ps.weight`

        -   `CBPS.weight`

        -   `ps.msm.weight`

        -   `CBPS.msm.weight`

        -   `none`

    -   Methods with "match" in the name and Mahalanobis will assign equal weights to similar control units.

    -   "Weighting" methods give higher weights to control units more similar to treated units.

-   **Variable Selection**

    -   Users need to define which covariates will be used through the **`covs.formula`** argument, a one-sided formula object.

    -   Variables on the right side of the formula are used for calculations.

    -   "Lagged" versions of variables can be included using the format: **`I(lag(name.of.var, 0:n))`**.

-   **Understanding `PanelMatch` and `matched.set` objects**

    -   The **`PanelMatch` function** returns a **`PanelMatch` object**.

    -   The most crucial element within the `PanelMatch` object is the **matched.set object**.

    -   Within the `PanelMatch` object, the matched.set object will have names like att, art, or atc.

    -   If **`qoi = ate`**, there will be two matched.set objects: att and atc.

-   **Matched.set Object Details**

    -   matched.set is a named list with added attributes.

    -   Attributes include:

        -   Lag

        -   Names of treatment

        -   Unit and time variables

    -   Each list entry represents a matched set of treated and control units.

    -   Naming follows a structure: **`[id variable].[time variable]`**.

    -   Each list element is a vector of control unit ids that match the treated unit mentioned in the element name.

    -   Since it's a matching method, weights are only given to the **`size.match`** most similar control units based on distance calculations.


```r
# PanelMatch without any refinement
PM.results.none <-
    PanelMatch(
        lag = 4,
        time.id = "year",
        unit.id = "wbcode2",
        treatment = "dem",
        refinement.method = "none",
        data = dem,
        match.missing = TRUE,
        size.match = 5,
        qoi = "att",
        outcome.var = "y",
        lead = 0:4,
        forbid.treatment.reversal = FALSE,
        use.diagonal.variance.matrix = TRUE
    )

# Extract the matched.set object
msets.none <- PM.results.none$att

# PanelMatch with refinement
PM.results.maha <-
    PanelMatch(
        lag = 4,
        time.id = "year",
        unit.id = "wbcode2",
        treatment = "dem",
        refinement.method = "mahalanobis", # use Mahalanobis distance
        data = dem,
        match.missing = TRUE,
        covs.formula = ~ tradewb,
        size.match = 5,
        qoi = "att" ,
        outcome.var = "y",
        lead = 0:4,
        forbid.treatment.reversal = FALSE,
        use.diagonal.variance.matrix = TRUE
    )
msets.maha <- PM.results.maha$att
```


```r
# these 2 should be identical because weights are not shown
msets.none |> head()
#>   wbcode2 year matched.set.size
#> 1       4 1992               74
#> 2       4 1997                2
#> 3       6 1973               63
#> 4       6 1983               73
#> 5       7 1991               81
#> 6       7 1998                1
msets.maha |> head()
#>   wbcode2 year matched.set.size
#> 1       4 1992               74
#> 2       4 1997                2
#> 3       6 1973               63
#> 4       6 1983               73
#> 5       7 1991               81
#> 6       7 1998                1
# summary(msets.none)
# summary(msets.maha)
```

**Visualizing Matched Sets with the plot method**

-   Users can visualize the distribution of the matched set sizes.

-   A red line, by default, indicates the count of matched sets where treated units had no matching control units (i.e., empty matched sets).

-   Plot adjustments can be made using **`graphics::plot`**.


```r
plot(msets.none)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-20-1.png" width="90%" style="display: block; margin: auto;" />

**Comparing Methods of Refinement**

-   Users are encouraged to:

    -   Use substantive knowledge for experimentation and evaluation.

    -   Consider the following when configuring `PanelMatch`:

        1.  The number of matched sets.

        2.  The number of controls matched to each treated unit.

        3.  Achieving covariate balance.

    -   **Note**: Large numbers of small matched sets can lead to larger standard errors during the estimation stage.

    -   Covariates that aren't well balanced can lead to undesirable comparisons between treated and control units.

    -   Aspects to consider include:

        -   Refinement method.

        -   Variables for weight calculation.

        -   Size of the lag window.

        -   Procedures for addressing missing data (refer to **`match.missing`** and **`listwise.delete`** arguments).

        -   Maximum size of matched sets (for matching methods).

-   **Supportive Features:**

    -   **`print`**, **`plot`**, and **`summary`** methods assist in understanding matched sets and their sizes.

    -   **`get_covariate_balance`** helps evaluate covariate balance:

        -   Lower values in the covariate balance calculations are preferred.


```r
PM.results.none <-
    PanelMatch(
        lag = 4,
        time.id = "year",
        unit.id = "wbcode2",
        treatment = "dem",
        refinement.method = "none",
        data = dem,
        match.missing = TRUE,
        size.match = 5,
        qoi = "att",
        outcome.var = "y",
        lead = 0:4,
        forbid.treatment.reversal = FALSE,
        use.diagonal.variance.matrix = TRUE
    )
PM.results.maha <-
    PanelMatch(
        lag = 4,
        time.id = "year",
        unit.id = "wbcode2",
        treatment = "dem",
        refinement.method = "mahalanobis",
        data = dem,
        match.missing = TRUE,
        covs.formula = ~ I(lag(tradewb, 1:4)) + I(lag(y, 1:4)),
        size.match = 5,
        qoi = "att",
        outcome.var = "y",
        lead = 0:4,
        forbid.treatment.reversal = FALSE,
        use.diagonal.variance.matrix = TRUE
    )

# listwise deletion used for missing data
PM.results.listwise <-
    PanelMatch(
        lag = 4,
        time.id = "year",
        unit.id = "wbcode2",
        treatment = "dem",
        refinement.method = "mahalanobis",
        data = dem,
        match.missing = FALSE,
        listwise.delete = TRUE,
        covs.formula = ~ I(lag(tradewb, 1:4)) + I(lag(y, 1:4)),
        size.match = 5,
        qoi = "att",
        outcome.var = "y",
        lead = 0:4,
        forbid.treatment.reversal = FALSE,
        use.diagonal.variance.matrix = TRUE
    )

# propensity score based weighting method
PM.results.ps.weight <-
    PanelMatch(
        lag = 4,
        time.id = "year",
        unit.id = "wbcode2",
        treatment = "dem",
        refinement.method = "ps.weight",
        data = dem,
        match.missing = FALSE,
        listwise.delete = TRUE,
        covs.formula = ~ I(lag(tradewb, 1:4)) + I(lag(y, 1:4)),
        size.match = 5,
        qoi = "att",
        outcome.var = "y",
        lead = 0:4,
        forbid.treatment.reversal = FALSE
    )

get_covariate_balance(
    PM.results.none$att,
    data = dem,
    covariates = c("tradewb", "y"),
    plot = FALSE
)
#>         tradewb            y
#> t_4 -0.07245466  0.291871990
#> t_3 -0.20930129  0.208654876
#> t_2 -0.24425207  0.107736647
#> t_1 -0.10806125 -0.004950238

get_covariate_balance(
    PM.results.maha$att,
    data = dem,
    covariates = c("tradewb", "y"),
    plot = FALSE
)
#>         tradewb          y
#> t_4  0.04558637 0.09701606
#> t_3 -0.03312750 0.10844046
#> t_2 -0.01396793 0.08890753
#> t_1  0.10474894 0.06618865


get_covariate_balance(
    PM.results.listwise$att,
    data = dem,
    covariates = c("tradewb", "y"),
    plot = FALSE
)
#>         tradewb          y
#> t_4  0.05634922 0.05223623
#> t_3 -0.01104797 0.05217896
#> t_2  0.01411473 0.03094133
#> t_1  0.06850180 0.02092209

get_covariate_balance(
    PM.results.ps.weight$att,
    data = dem,
    covariates = c("tradewb", "y"),
    plot = FALSE
)
#>         tradewb          y
#> t_4 0.014362590 0.04035905
#> t_3 0.005529734 0.04188731
#> t_2 0.009410044 0.04195008
#> t_1 0.027907540 0.03975173
```

**get_covariate_balance Function Options:**

-   Allows for the generation of plots displaying covariate balance using **`plot = TRUE`**.

-   Plots can be customized using arguments typically used with the base R **`plot`** method.

-   Option to set **`use.equal.weights = TRUE`** for:

    -   Obtaining the balance of unrefined sets.

    -   Facilitating understanding of the refinement's impact.


```r
# Use equal weights
get_covariate_balance(
    PM.results.ps.weight$att,
    data = dem,
    use.equal.weights = TRUE,
    covariates = c("tradewb", "y"),
    plot = TRUE,
    # visualize by setting plot to TRUE
    ylim = c(-1, 1)
)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-22-1.png" width="90%" style="display: block; margin: auto;" />

```r

# Compare covariate balance to refined sets
# See large improvement in balance
get_covariate_balance(
    PM.results.ps.weight$att,
    data = dem,
    covariates = c("tradewb", "y"),
    plot = TRUE,
    # visualize by setting plot to TRUE
    ylim = c(-1, 1)
)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-22-2.png" width="90%" style="display: block; margin: auto;" />

```r


balance_scatter(
    matched_set_list = list(PM.results.maha$att,
                            PM.results.ps.weight$att),
    data = dem,
    covariates = c("y", "tradewb")
)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-22-3.png" width="90%" style="display: block; margin: auto;" />

**`PanelEstimate`**

-   **Standard Error Calculation Methods**

    -   There are different methods available:

        -   **Bootstrap** (default method with 1000 iterations).

        -   **Conditional**: Assumes independence across units, but not time.

        -   **Unconditional**: Doesn't make assumptions of independence across units or time.

    -   For **`qoi`** values set to `att`, `art`, or `atc` [@imai2021matching]:

        -   You can use analytical methods for calculating standard errors, which include both "conditional" and "unconditional" methods.


```r
PE.results <- PanelEstimate(
    sets              = PM.results.ps.weight,
    data              = dem,
    se.method         = "bootstrap",
    number.iterations = 1000,
    confidence.level  = .95
)

# point estimates
PE.results[["estimates"]]
#>       t+0       t+1       t+2       t+3       t+4 
#> 0.2609565 0.9630847 1.2851017 1.7370930 1.4871846

# standard errors
PE.results[["standard.error"]]
#>       t+0       t+1       t+2       t+3       t+4 
#> 0.6399349 1.0304938 1.3825265 1.7625951 2.1672629


# use conditional method
PE.results <- PanelEstimate(
    sets             = PM.results.ps.weight,
    data             = dem,
    se.method        = "conditional",
    confidence.level = .95
)

# point estimates
PE.results[["estimates"]]
#>       t+0       t+1       t+2       t+3       t+4 
#> 0.2609565 0.9630847 1.2851017 1.7370930 1.4871846

# standard errors
PE.results[["standard.error"]]
#>       t+0       t+1       t+2       t+3       t+4 
#> 0.4844805 0.8170604 1.1171942 1.4116879 1.7172143

summary(PE.results)
#> Weighted Difference-in-Differences with Propensity Score
#> Matches created with 4 lags
#> 
#> Standard errors computed with conditional  method
#> 
#> Estimate of Average Treatment Effect on the Treated (ATT) by Period:
#> $summary
#>      estimate std.error       2.5%    97.5%
#> t+0 0.2609565 0.4844805 -0.6886078 1.210521
#> t+1 0.9630847 0.8170604 -0.6383243 2.564494
#> t+2 1.2851017 1.1171942 -0.9045586 3.474762
#> t+3 1.7370930 1.4116879 -1.0297644 4.503950
#> t+4 1.4871846 1.7172143 -1.8784937 4.852863
#> 
#> $lag
#> [1] 4
#> 
#> $qoi
#> [1] "att"

plot(PE.results)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-23-1.png" width="90%" style="display: block; margin: auto;" />

**Moderating Variables**


```r
# moderating variable
dem$moderator <- 0
dem$moderator <- ifelse(dem$wbcode2 > 100, 1, 2)

PM.results <-
    PanelMatch(
        lag                          = 4,
        time.id                      = "year",
        unit.id                      = "wbcode2",
        treatment                    = "dem",
        refinement.method            = "mahalanobis",
        data                         = dem,
        match.missing                = TRUE,
        covs.formula                 = ~ I(lag(tradewb, 1:4)) + I(lag(y, 1:4)),
        size.match                   = 5,
        qoi                          = "att",
        outcome.var                  = "y",
        lead                         = 0:4,
        forbid.treatment.reversal    = FALSE,
        use.diagonal.variance.matrix = TRUE
    )
PE.results <-
    PanelEstimate(sets      = PM.results,
                  data      = dem,
                  moderator = "moderator")

# Each element in the list corresponds to a level in the moderator
plot(PE.results[[1]])
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-24-1.png" width="90%" style="display: block; margin: auto;" />

```r

plot(PE.results[[2]])
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-24-2.png" width="90%" style="display: block; margin: auto;" />

To write up for journal submission, you can follow the following report:

In this study, closely aligned with the research by [@acemoglu2019democracy], two key effects of democracy on economic growth are estimated: the impact of democratization and that of authoritarian reversal. The treatment variable, $X_{it}$, is defined to be one if country $i$ is democratic in year $t$, and zero otherwise.

The Average Treatment Effect for the Treated (ATT) under democratization is formulated as follows:

$$
\begin{aligned}
\delta(F, L) &= \mathbb{E} \left\{ Y_{i, t + F} (X_{it} = 1, X_{i, t - 1} = 0, \{X_{i,t-l}\}_{l=2}^L) \right. \\
&\left. - Y_{i, t + F} (X_{it} = 0, X_{i, t - 1} = 0, \{X_{i,t-l}\}_{l=2}^L) | X_{it} = 1, X_{i, t - 1} = 0 \right\}
\end{aligned}
$$

In this framework, the treated observations are countries that transition from an authoritarian regime $X_{it-1} = 0$ to a democratic one $X_{it} = 1$. The variable $F$ represents the number of leads, denoting the time periods following the treatment, and $L$ signifies the number of lags, indicating the time periods preceding the treatment.

The ATT under authoritarian reversal is given by:

$$
\begin{aligned}
&\mathbb{E} \left[ Y_{i, t + F} (X_{it} = 0, X_{i, t - 1} = 1, \{ X_{i, t - l}\}_{l=2}^L ) \right. \\
&\left. - Y_{i, t + F} (X_{it} = 1, X_{it-1} = 1, \{X_{i, t - l} \}_{l=2}^L ) | X_{it} = 0, X_{i, t - 1} = 1 \right]
\end{aligned}
$$

The ATT is calculated conditioning on 4 years of lags ($L = 4$) and up to 4 years following the policy change $F = 1, 2, 3, 4$. Matched sets for each treated observation are constructed based on its treatment history, with the number of matched control units generally decreasing when considering a 4-year treatment history as compared to a 1-year history.

To enhance the quality of matched sets, methods such as Mahalanobis distance matching, propensity score matching, and propensity score weighting are utilized. These approaches enable us to evaluate the effectiveness of each refinement method. In the process of matching, we employ both up-to-five and up-to-ten matching to investigate how sensitive our empirical results are to the maximum number of allowed matches. For more information on the refinement process, please see the Web Appendix

> The Mahalanobis distance is expressed through a specific formula. We aim to pair each treated unit with a maximum of $J$ control units, permitting replacement, denoted as $| \mathcal{M}_{it} \le J|$. The average Mahalanobis distance between a treated and each control unit over time is computed as:
>
> $$ S_{it} (i') = \frac{1}{L} \sum_{l = 1}^L \sqrt{(\mathbf{V}_{i, t - l} - \mathbf{V}_{i', t -l})^T \mathbf{\Sigma}_{i, t - l}^{-1} (\mathbf{V}_{i, t - l} - \mathbf{V}_{i', t -l})} $$
>
> For a matched control unit $i' \in \mathcal{M}_{it}$, $\mathbf{V}_{it'}$ represents the time-varying covariates to adjust for, and $\mathbf{\Sigma}_{it'}$ is the sample covariance matrix for $\mathbf{V}_{it'}$. Essentially, we calculate a standardized distance using time-varying covariates and average this across different time intervals.
>
> In the context of propensity score matching, we employ a logistic regression model with balanced covariates to derive the propensity score. Defined as the conditional likelihood of treatment given pre-treatment covariates [@rosenbaum1983central], the propensity score is estimated by first creating a data subset comprised of all treated and their matched control units from the same year. This logistic regression model is then fitted as follows:
>
> $$ \begin{aligned} & e_{it} (\{\mathbf{U}_{i, t - l} \}^L_{l = 1}) \\ &= Pr(X_{it} = 1| \mathbf{U}_{i, t -1}, \ldots, \mathbf{U}_{i, t - L}) \\ &= \frac{1}{1 = \exp(- \sum_{l = 1}^L \beta_l^T \mathbf{U}_{i, t - l})} \end{aligned} $$
>
> where $\mathbf{U}_{it'} = (X_{it'}, \mathbf{V}_{it'}^T)^T$. Given this model, the estimated propensity score for all treated and matched control units is then computed. This enables the adjustment for lagged covariates via matching on the calculated propensity score, resulting in the following distance measure:
>
> $$ S_{it} (i') = | \text{logit} \{ \hat{e}_{it} (\{ \mathbf{U}_{i, t - l}\}^L_{l = 1})\} - \text{logit} \{ \hat{e}_{i't}( \{ \mathbf{U}_{i', t - l} \}^L_{l = 1})\} | $$
>
> Here, $\hat{e}_{i't} (\{ \mathbf{U}_{i, t - l}\}^L_{l = 1})$ represents the estimated propensity score for each matched control unit $i' \in \mathcal{M}_{it}$.
>
> Once the distance measure $S_{it} (i')$ has been determined for all control units in the original matched set, we fine-tune this set by selecting up to $J$ closest control units, which meet a researcher-defined caliper constraint $C$. All other control units receive zero weight. This results in a refined matched set for each treated unit $(i, t)$:
>
> $$ \mathcal{M}_{it}^* = \{i' : i' \in \mathcal{M}_{it}, S_{it} (i') < C, S_{it} \le S_{it}^{(J)}\} $$
>
> $S_{it}^{(J)}$ is the $J$th smallest distance among the control units in the original set $\mathcal{M}_{it}$.
>
> For further refinement using weighting, a weight is assigned to each control unit $i'$ in a matched set corresponding to a treated unit $(i, t)$, with greater weight accorded to more similar units. We utilize inverse propensity score weighting, based on the propensity score model mentioned earlier:
>
> $$ w_{it}^{i'} \propto \frac{\hat{e}_{i't} (\{ \mathbf{U}_{i, t-l} \}^L_{l = 1} )}{1 - \hat{e}_{i't} (\{ \mathbf{U}_{i, t-l} \}^L_{l = 1} )} $$
>
> In this model, $\sum_{i' \in \mathcal{M}_{it}} w_{it}^{i'} = 1$ and $w_{it}^{i'} = 0$ for $i' \notin \mathcal{M}_{it}$. The model is fitted to the complete sample of treated and matched control units.

> Checking Covariate Balance A distinct advantage of the proposed methodology over regression methods is the ability it offers researchers to inspect the covariate balance between treated and matched control observations. This facilitates the evaluation of whether treated and matched control observations are comparable regarding observed confounders. To investigate the mean difference of each covariate (e.g., $V_{it'j}$, representing the $j$-th variable in $\mathbf{V}_{it'}$) between the treated observation and its matched control observation at each pre-treatment time period (i.e., $t' < t$), we further standardize this difference. For any given pretreatment time period, we adjust by the standard deviation of each covariate across all treated observations in the dataset. Thus, the mean difference is quantified in terms of standard deviation units. Formally, for each treated observation $(i,t)$ where $D_{it} = 1$, we define the covariate balance for variable $j$ at the pretreatment time period $t - l$ as: \begin{equation}
> B_{it}(j, l) = \frac{V_{i, t- l,j}- \sum_{i' \in \mathcal{M}_{it}}w_{it}^{i'}V_{i', t-l,j}}{\sqrt{\frac{1}{N_1 - 1} \sum_{i'=1}^N \sum_{t' = L+1}^{T-F}D_{i't'}(V_{i', t'-l, j} - \bar{V}_{t' - l, j})^2}}
> \label{eq:covbalance}
> \end{equation} where $N_1 = \sum_{i'= 1}^N \sum_{t' = L+1}^{T-F} D_{i't'}$ denotes the total number of treated observations and $\bar{V}_{t-l,j} = \sum_{i=1}^N D_{i,t-l,j}/N$. We then aggregate this covariate balance measure across all treated observations for each covariate and pre-treatment time period: \begin{equation}
> \bar{B}(j, l) = \frac{1}{N_1} \sum_{i=1}^N \sum_{t = L+ 1}^{T-F}D_{it} B_{it}(j,l)
> \label{eq:aggbalance}
> \end{equation} Lastly, we evaluate the balance of lagged outcome variables over several pre-treatment periods and that of time-varying covariates. This examination aids in assessing the validity of the parallel trend assumption integral to the DiD estimator justification.

In Figure \@ref(fig:balancescatter), we demonstrate the enhancement of covariate balance thank to the refinement of matched sets. Each scatter plot contrasts the absolute standardized mean difference, as detailed in Equation \@ref(eq:aggbalance), before (horizontal axis) and after (vertical axis) this refinement. Points below the 45-degree line indicate an improved standardized mean balance for certain time-varying covariates post-refinement. The majority of variables benefit from this refinement process. Notably, the propensity score weighting (bottom panel) shows the most significant improvement, whereas Mahalanobis matching (top panel) yields a more modest improvement.


```r
library(PanelMatch)
library(causalverse)

runPanelMatch <- function(method, lag, size.match=NULL, qoi="att") {
    
    # Default parameters for PanelMatch
    common.args <- list(
        lag = lag,
        time.id = "year",
        unit.id = "wbcode2",
        treatment = "dem",
        data = dem,
        covs.formula = ~ I(lag(tradewb, 1:4)) + I(lag(y, 1:4)),
        qoi = qoi,
        outcome.var = "y",
        lead = 0:4,
        forbid.treatment.reversal = FALSE,
        size.match = size.match  # setting size.match here for all methods
    )
    
    if(method == "mahalanobis") {
        common.args$refinement.method <- "mahalanobis"
        common.args$match.missing <- TRUE
        common.args$use.diagonal.variance.matrix <- TRUE
    } else if(method == "ps.match") {
        common.args$refinement.method <- "ps.match"
        common.args$match.missing <- FALSE
        common.args$listwise.delete <- TRUE
    } else if(method == "ps.weight") {
        common.args$refinement.method <- "ps.weight"
        common.args$match.missing <- FALSE
        common.args$listwise.delete <- TRUE
    }
    
    return(do.call(PanelMatch, common.args))
}

methods <- c("mahalanobis", "ps.match", "ps.weight")
lags <- c(1, 4)
sizes <- c(5, 10)
```

You can either do it sequentailly


```r
res_pm <- list()

for(method in methods) {
    for(lag in lags) {
        for(size in sizes) {
            name <- paste0(method, ".", lag, "lag.", size, "m")
            res_pm[[name]] <- runPanelMatch(method, lag, size)
        }
    }
}

# Now, you can access res_pm using res_pm[["mahalanobis.1lag.5m"]] etc.

# for treatment reversal
res_pm_rev <- list()

for(method in methods) {
    for(lag in lags) {
        for(size in sizes) {
            name <- paste0(method, ".", lag, "lag.", size, "m")
            res_pm_rev[[name]] <- runPanelMatch(method, lag, size, qoi = "art")
        }
    }
}
```

or in parallel


```r
library(foreach)
library(doParallel)
registerDoParallel(cores = 4)
# Initialize an empty list to store results
res_pm <- list()

# Replace nested for-loops with foreach
results <-
  foreach(
    method = methods,
    .combine = 'c',
    .multicombine = TRUE,
    .packages = c("PanelMatch", "causalverse")
  ) %dopar% {
    tmp <- list()
    for (lag in lags) {
      for (size in sizes) {
        name <- paste0(method, ".", lag, "lag.", size, "m")
        tmp[[name]] <- runPanelMatch(method, lag, size)
      }
    }
    tmp
  }

# Collate results
for (name in names(results)) {
  res_pm[[name]] <- results[[name]]
}

# Treatment reversal
# Initialize an empty list to store results
res_pm_rev <- list()

# Replace nested for-loops with foreach
results_rev <-
  foreach(
    method = methods,
    .combine = 'c',
    .multicombine = TRUE,
    .packages = c("PanelMatch", "causalverse")
  ) %dopar% {
    tmp <- list()
    for (lag in lags) {
      for (size in sizes) {
        name <- paste0(method, ".", lag, "lag.", size, "m")
        tmp[[name]] <-
          runPanelMatch(method, lag, size, qoi = "art")
      }
    }
    tmp
  }

# Collate results
for (name in names(results_rev)) {
  res_pm_rev[[name]] <- results_rev[[name]]
}


stopImplicitCluster()
```


```r
library(gridExtra)

# Updated plotting function
create_balance_plot <- function(method, lag, sizes, res_pm, dem) {
    matched_set_lists <- lapply(sizes, function(size) {
        res_pm[[paste0(method, ".", lag, "lag.", size, "m")]]$att
    })
    
    return(
        balance_scatter_custom(
            matched_set_list = matched_set_lists,
            legend.title = "Possible Matches",
            set.names = as.character(sizes),
            legend.position = c(0.2, 0.8),
            
            # for compiled plot, you don't need x,y, or main labs
            x.axis.label = "",
            y.axis.label = "",
            main = "",
            data = dem,
            dot.size = 5,
            # show.legend = F,
            them_use = causalverse::ama_theme(base_size = 32),
            covariates = c("y", "tradewb")
        )
    )
}

plots <- list()

for (method in methods) {
    for (lag in lags) {
        plots[[paste0(method, ".", lag, "lag")]] <-
            create_balance_plot(method, lag, sizes, res_pm, dem)
    }
}

# # Arranging plots in a 3x2 grid
# grid.arrange(plots[["mahalanobis.1lag"]],
#              plots[["mahalanobis.4lag"]],
#              plots[["ps.match.1lag"]],
#              plots[["ps.match.4lag"]],
#              plots[["ps.weight.1lag"]],
#              plots[["ps.weight.4lag"]],
#              ncol=2, nrow=3)


# Standardized Mean Difference of Covariates
library(gridExtra)
library(grid)

# Create column and row labels using textGrob
col_labels <- c("1-year Lag", "4-year Lag")
row_labels <- c("Maha Matching", "PS Matching", "PS Weigthing")

major.axes.fontsize = 40
minor.axes.fontsize = 30

png(
    file.path(getwd(), "images", "did_balance_scatter.png"),
    width = 1200,
    height = 1000
)

# Create a list-of-lists, where each inner list represents a row
grid_list <- list(
    list(
        nullGrob(),
        textGrob(col_labels[1], gp = gpar(fontsize = minor.axes.fontsize)),
        textGrob(col_labels[2], gp = gpar(fontsize = minor.axes.fontsize))
    ),
    
    list(textGrob(
        row_labels[1],
        gp = gpar(fontsize = minor.axes.fontsize),
        rot = 90
    ), plots[["mahalanobis.1lag"]], plots[["mahalanobis.4lag"]]),
    
    list(textGrob(
        row_labels[2],
        gp = gpar(fontsize = minor.axes.fontsize),
        rot = 90
    ), plots[["ps.match.1lag"]], plots[["ps.match.4lag"]]),
    
    list(textGrob(
        row_labels[3],
        gp = gpar(fontsize = minor.axes.fontsize),
        rot = 90
    ), plots[["ps.weight.1lag"]], plots[["ps.weight.4lag"]])
)

# "Flatten" the list-of-lists into a single list of grobs
grobs <- do.call(c, grid_list)

grid.arrange(
    grobs = grobs,
    ncol = 3,
    nrow = 4,
    widths = c(0.15, 0.42, 0.42),
    heights = c(0.15, 0.28, 0.28, 0.28)
)

grid.text(
    "Before Refinement",
    x = 0.5,
    y = 0.03,
    gp = gpar(fontsize = major.axes.fontsize)
)
grid.text(
    "After Refinement",
    x = 0.03,
    y = 0.5,
    rot = 90,
    gp = gpar(fontsize = major.axes.fontsize)
)
dev.off()
#> png 
#>   2
```



Note: Scatter plots display the standardized mean difference of each covariate $j$ and lag year $l$ as defined in Equation \@ref(eq:aggbalance) before (x-axis) and after (y-axis) matched set refinement. Each plot includes varying numbers of possible matches for each matching method. Rows represent different matching/weighting methods, while columns indicate adjustments for various lag lengths.


```r
# Step 1: Define configurations
configurations <- list(
    list(refinement.method = "none", qoi = "att"),
    list(refinement.method = "none", qoi = "art"),
    list(refinement.method = "mahalanobis", qoi = "att"),
    list(refinement.method = "mahalanobis", qoi = "art"),
    list(refinement.method = "ps.match", qoi = "att"),
    list(refinement.method = "ps.match", qoi = "art"),
    list(refinement.method = "ps.weight", qoi = "att"),
    list(refinement.method = "ps.weight", qoi = "art")
)

# Step 2: Use lapply or loop to generate results
results <- lapply(configurations, function(config) {
    PanelMatch(
        lag                       = 4,
        time.id                   = "year",
        unit.id                   = "wbcode2",
        treatment                 = "dem",
        data                      = dem,
        match.missing             = FALSE,
        listwise.delete           = TRUE,
        size.match                = 5,
        outcome.var               = "y",
        lead                      = 0:4,
        forbid.treatment.reversal = FALSE,
        refinement.method         = config$refinement.method,
        covs.formula              = ~ I(lag(tradewb, 1:4)) + I(lag(y, 1:4)),
        qoi                       = config$qoi
    )
})

# Step 3: Get covariate balance and plot
plots <- mapply(function(result, config) {
    df <- get_covariate_balance(
        if (config$qoi == "att")
            result$att
        else
            result$art,
        data = dem,
        covariates = c("tradewb", "y"),
        plot = F
    )
    causalverse::plot_covariate_balance_pretrend(df, main = "", show_legend = F)
}, results, configurations, SIMPLIFY = FALSE)

# Set names for plots
names(plots) <- sapply(configurations, function(config) {
    paste(config$qoi, config$refinement.method, sep = ".")
})
```

To export


```r
library(gridExtra)
library(grid)

# Column and row labels
col_labels <-
    c("None",
      "Mahalanobis",
      "Propensity Score Matching",
      "Propensity Score Weighting")
row_labels <- c("ATT", "ART")

# Specify your desired fontsize for labels
minor.axes.fontsize <- 16
major.axes.fontsize <- 20

png(file.path(getwd(), "images", "p_covariate_balance.png"), width=1200, height=1000)

# Create a list-of-lists, where each inner list represents a row
grid_list <- list(
    list(
        nullGrob(),
        textGrob(col_labels[1], gp = gpar(fontsize = minor.axes.fontsize)),
        textGrob(col_labels[2], gp = gpar(fontsize = minor.axes.fontsize)),
        textGrob(col_labels[3], gp = gpar(fontsize = minor.axes.fontsize)),
        textGrob(col_labels[4], gp = gpar(fontsize = minor.axes.fontsize))
    ),
    
    list(
        textGrob(
            row_labels[1],
            gp = gpar(fontsize = minor.axes.fontsize),
            rot = 90
        ),
        plots$att.none,
        plots$att.mahalanobis,
        plots$att.ps.match,
        plots$att.ps.weight
    ),
    
    list(
        textGrob(
            row_labels[2],
            gp = gpar(fontsize = minor.axes.fontsize),
            rot = 90
        ),
        plots$art.none,
        plots$art.mahalanobis,
        plots$art.ps.match,
        plots$art.ps.weight
    )
)

# "Flatten" the list-of-lists into a single list of grobs
grobs <- do.call(c, grid_list)

# Arrange your plots with text labels
grid.arrange(
    grobs   = grobs,
    ncol    = 5,
    nrow    = 3,
    widths  = c(0.1, 0.225, 0.225, 0.225, 0.225),
    heights = c(0.1, 0.45, 0.45)
)

# Add main x and y axis titles
grid.text(
    "Refinement Methods",
    x  = 0.5,
    y  = 0.01,
    gp = gpar(fontsize = major.axes.fontsize)
)
grid.text(
    "Quantities of Interest",
    x   = 0.02,
    y   = 0.5,
    rot = 90,
    gp  = gpar(fontsize = major.axes.fontsize)
)

dev.off()
```


```r
library(knitr)
include_graphics(file.path(getwd(), "images", "p_covariate_balance.png"))
```

<img src="images/p_covariate_balance.png" width="90%" style="display: block; margin: auto;" />

Note: Each graph displays the standardized mean difference, as outlined in Equation \@ref(eq:aggbalance), plotted on the vertical axis across a pre-treatment duration of four years represented on the horizontal axis. The leftmost column illustrates the balance prior to refinement, while the subsequent three columns depict the covariate balance post the application of distinct refinement techniques. Each individual line signifies the balance of a specific variable during the pre-treatment phase.The red line is tradewb and blue line is the lagged outcome variable.

In Figure \@ref(fig:balancepretreat), we observe a marked improvement in covariate balance due to the implemented matching procedures during the pre-treatment period. Our analysis prioritizes methods that adjust for time-varying covariates over a span of four years preceding the treatment initiation. The two rows delineate the standardized mean balance for both treatment modalities, with individual lines representing the balance for each covariate.

Across all scenarios, the refinement attributed to matched sets significantly enhances balance. Notably, using propensity score weighting considerably mitigates imbalances in confounders. While some degree of imbalance remains evident in the Mahalanobis distance and propensity score matching techniques, the standardized mean difference for the lagged outcome remains stable throughout the pre-treatment phase. This consistency lends credence to the validity of the proposed DiD estimator.

**Estimation Results**

We now detail the estimated ATTs derived from the matching techniques. Figure below offers visual representations of the impacts of treatment initiation (upper panel) and treatment reversal (lower panel) on the outcome variable for a duration of 5 years post-transition, specifically, (F = 0, 1, ..., 4). Across the five methods (columns), it becomes evident that the point estimates of effects associated with treatment initiation consistently approximate zero over the 5-year window. In contrast, the estimated outcomes of treatment reversal are notably negative and maintain statistical significance through all refinement techniques during the initial year of transition and the 1 to 4 years that follow, provided treatment reversal is permissible. These effects are notably pronounced, pointing to an estimated reduction of roughly X% in the outcome variable.

Collectively, these findings indicate that the transition into the treated state from its absence doesn't invariably lead to a heightened outcome. Instead, the transition from the treated state back to its absence exerts a considerable negative effect on the outcome variable in both the short and intermediate terms. Hence, the positive effect of the treatment (if we were to use traditional DiD) is actually driven by the negative effect of treatment reversal.


```r
# sequential
# Step 1: Apply PanelEstimate function

# Initialize an empty list to store results
res_est <- vector("list", length(res_pm))

# Iterate over each element in res_pm
for (i in 1:length(res_pm)) {
  res_est[[i]] <- PanelEstimate(
    res_pm[[i]],
    data = dem,
    se.method = "bootstrap",
    number.iterations = 1000,
    confidence.level = .95
  )
  # Transfer the name of the current element to the res_est list
  names(res_est)[i] <- names(res_pm)[i]
}

# Step 2: Apply plot_PanelEstimate function

# Initialize an empty list to store plot results
res_est_plot <- vector("list", length(res_est))

# Iterate over each element in res_est
for (i in 1:length(res_est)) {
    res_est_plot[[i]] <-
        plot_PanelEstimate(res_est[[i]],
                           main = "",
                           theme_use = causalverse::ama_theme(base_size = 14))
    # Transfer the name of the current element to the res_est_plot list
    names(res_est_plot)[i] <- names(res_est)[i]
}

# check results
# res_est_plot$mahalanobis.1lag.5m


# Step 1: Apply PanelEstimate function for res_pm_rev

# Initialize an empty list to store results
res_est_rev <- vector("list", length(res_pm_rev))

# Iterate over each element in res_pm_rev
for (i in 1:length(res_pm_rev)) {
  res_est_rev[[i]] <- PanelEstimate(
    res_pm_rev[[i]],
    data = dem,
    se.method = "bootstrap",
    number.iterations = 1000,
    confidence.level = .95
  )
  # Transfer the name of the current element to the res_est_rev list
  names(res_est_rev)[i] <- names(res_pm_rev)[i]
}

# Step 2: Apply plot_PanelEstimate function for res_est_rev

# Initialize an empty list to store plot results
res_est_plot_rev <- vector("list", length(res_est_rev))

# Iterate over each element in res_est_rev
for (i in 1:length(res_est_rev)) {
    res_est_plot_rev[[i]] <-
        plot_PanelEstimate(res_est_rev[[i]],
                           main = "",
                           theme_use = causalverse::ama_theme(base_size = 14))
  # Transfer the name of the current element to the res_est_plot_rev list
  names(res_est_plot_rev)[i] <- names(res_est_rev)[i]
}
```


```r
# parallel
library(doParallel)
library(foreach)

# Detect the number of cores to use for parallel processing
num_cores <- 4

# Register the parallel backend
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Step 1: Apply PanelEstimate function in parallel
res_est <-
    foreach(i = 1:length(res_pm), .packages = "PanelMatch") %dopar% {
        PanelEstimate(
            res_pm[[i]],
            data = dem,
            se.method = "bootstrap",
            number.iterations = 1000,
            confidence.level = .95
        )
    }

# Transfer names from res_pm to res_est
names(res_est) <- names(res_pm)

# Step 2: Apply plot_PanelEstimate function in parallel
res_est_plot <-
    foreach(
        i = 1:length(res_est),
        .packages = c("PanelMatch", "causalverse", "ggplot2")
    ) %dopar% {
        plot_PanelEstimate(res_est[[i]],
                           main = "",
                           theme_use = causalverse::ama_theme(base_size = 10))
    }

# Transfer names from res_est to res_est_plot
names(res_est_plot) <- names(res_est)



# Step 1: Apply PanelEstimate function for res_pm_rev in parallel
res_est_rev <-
    foreach(i = 1:length(res_pm_rev), .packages = "PanelMatch") %dopar% {
        PanelEstimate(
            res_pm_rev[[i]],
            data = dem,
            se.method = "bootstrap",
            number.iterations = 1000,
            confidence.level = .95
        )
    }

# Transfer names from res_pm_rev to res_est_rev
names(res_est_rev) <- names(res_pm_rev)

# Step 2: Apply plot_PanelEstimate function for res_est_rev in parallel
res_est_plot_rev <-
    foreach(
        i = 1:length(res_est_rev),
        .packages = c("PanelMatch", "causalverse", "ggplot2")
    ) %dopar% {
        plot_PanelEstimate(res_est_rev[[i]],
                           main = "",
                           theme_use = causalverse::ama_theme(base_size = 10))
    }

# Transfer names from res_est_rev to res_est_plot_rev
names(res_est_plot_rev) <- names(res_est_rev)

# Stop the cluster
stopCluster(cl)
```

To export


```r
library(gridExtra)
library(grid)

# Column and row labels
col_labels <- c("Mahalanobis 5m", 
                "Mahalanobis 10m", 
                "PS Matching 5m", 
                "PS Matching 10m", 
                "PS Weighting 5m")

row_labels <- c("ATT", "ART")

# Specify your desired fontsize for labels
minor.axes.fontsize <- 16
major.axes.fontsize <- 20

png(file.path(getwd(), "images", "p_did_est_in_n_out.png"), width=1200, height=1000)

# Create a list-of-lists, where each inner list represents a row
grid_list <- list(
  list(
    nullGrob(),
    textGrob(col_labels[1], gp = gpar(fontsize = minor.axes.fontsize)),
    textGrob(col_labels[2], gp = gpar(fontsize = minor.axes.fontsize)),
    textGrob(col_labels[3], gp = gpar(fontsize = minor.axes.fontsize)),
    textGrob(col_labels[4], gp = gpar(fontsize = minor.axes.fontsize)),
    textGrob(col_labels[5], gp = gpar(fontsize = minor.axes.fontsize))
  ),
  
  list(
    textGrob(row_labels[1], gp = gpar(fontsize = minor.axes.fontsize), rot = 90),
    res_est_plot$mahalanobis.1lag.5m,
    res_est_plot$mahalanobis.1lag.10m,
    res_est_plot$ps.match.1lag.5m,
    res_est_plot$ps.match.1lag.10m,
    res_est_plot$ps.weight.1lag.5m
  ),
  
  list(
    textGrob(row_labels[2], gp = gpar(fontsize = minor.axes.fontsize), rot = 90),
    res_est_plot_rev$mahalanobis.1lag.5m,
    res_est_plot_rev$mahalanobis.1lag.10m,
    res_est_plot_rev$ps.match.1lag.5m,
    res_est_plot_rev$ps.match.1lag.10m,
    res_est_plot_rev$ps.weight.1lag.5m
  )
)

# "Flatten" the list-of-lists into a single list of grobs
grobs <- do.call(c, grid_list)

# Arrange your plots with text labels
grid.arrange(
  grobs   = grobs,
  ncol    = 6,
  nrow    = 3,
  widths  = c(0.1, 0.18, 0.18, 0.18, 0.18, 0.18),
  heights = c(0.1, 0.45, 0.45)
)

# Add main x and y axis titles
grid.text(
  "Methods",
  x  = 0.5,
  y  = 0.02,
  gp = gpar(fontsize = major.axes.fontsize)
)
grid.text(
  "",
  x   = 0.02,
  y   = 0.5,
  rot = 90,
  gp  = gpar(fontsize = major.axes.fontsize)
)

dev.off()
```


```r
library(knitr)
include_graphics(file.path(getwd(), "images", "p_did_est_in_n_out.png"))
```

#### Counterfactual Estimators

-   Also known as **imputation approach** [@liu2022practical]
-   This class of estimator consider observation treatment as missing data. Models are built using data from the control units to impute conterfactuals for the treated observations.
-   It's called counterfactual estimators because they predict outcomes as if the treated observations had not received the treatment.
-   Advantages:
    -   Avoids negative weights and biases by not using treated observations for modeling and applying uniform weights.
    -   Supports various models, including those that may relax strict exogeneity assumptions.
-   Methods including
    -   Fixed-effects conterfactual estimator (FEct) (DiD is a special case):
        -   Based on the [Two-way Fixed-effects], where assumes linear additive functional form of unobservables based on unit and time FEs. But FEct fixes the improper weighting of TWFE by comparing within each matched pair (where each pair is the treated observation and its predicted counterfactual that is the weighted sum of all untreated observations).
    -   Interactive Fixed Effects conterfactual estimator (IFEct) [@gobillon2016regional, @xu2017generalized]:
        -   When we suspect unobserved time-varying confounder, FEct fails. Instead, IFEct uses the factor-augmented models to relax the strict exogeneity assumption where the effects of unobservables can be decomposed to unit FE + time FE + unit x time FE.
        -   Generalized Synthetic Controls are a subset of IFEct when treatments don't revert.
    -   [Matrix completion] (MC) [@athey2021matrix]:
        -   Generalization of factor-augmented models. Different from IFEct which uses hard impute, MC uses soft impute to regularize the singular values when decomposing the residual matrix.
        -   Only when latent factors (of unobservables) are strong and sparse, IFEct outperforms MC.
    -   [Synthetic Controls] (case studies)

**Identifying Assumptions**:

1.  **Function Form**: Additive separability of observables, unobservables, and idiosyncratic error term.
    -   Hence, these models are scale dependent [@athey2006identification] (e.g., log-transform outcome can invadiate this assumption).
2.  **Strict Exogeneity**: Conditional on observables and unobservables, potential outcomes are independent of treatment assignment (i.e., baseline quasi-randomization)
    -   In DiD, where unobservables = unit + time FEs, this assumption is the parallel trends assumption
3.  **Low-dimensional Decomposition (Feasibility Assumption)**: Unobservable effects can be decomposed in low-dimension.
    -   For the case that $U_{it} = f_t \times \lambda_i$ where $f_t$ = common time trend (time FE), and $\lambda_i$ = unit heterogeneity (unit FE). If $U_{it} = f_t \times \lambda_i$ , DiD can satisfy this assumption. But this assumption is weaker than that of DID, and allows us to control for unobservables based on data.

**Estimation Procedure**:

1.  Using all control observations, estimate the functions of both observable and unobservable variables (relying on Assumptions 1 and 3).
2.  Predict the counterfactual outcomes for each treated unit using the obtained functions.
3.  Calculate the difference in treatment effect for each treated individual.
4.  By averaging over all treated individuals, you can obtain the Average Treatment Effect on the Treated (ATT).

Notes:

-   Use jackknife when number of treated units is small [@liu2022practical, p.166].

##### Imputation Method

@liu2022practical can also account for treatment reversals and heterogeneous treatment effects.

Other imputation estimators include

-   [\@gardner2022two and \@borusyak2021revisiting]

-   @RePEc:arx:papers:2301.11358


```r
library(fect)

PanelMatch::dem

model.fect <-
    fect(
        Y = "y",
        D = "dem",
        X = "tradewb",
        data = na.omit(PanelMatch::dem),
        method = "fe",
        index = c("wbcode2", "year"),
        se = TRUE,
        parallel = TRUE,
        seed = 1234,
        # twfe
        force = "two-way"
    )
print(model.fect$est.avg)

plot(model.fect)

plot(model.fect, stats = "F.p")
```

F-test $H_0$: residual averages in the pre-treatment periods = 0

To see treatment reversal effects


```r
plot(model.fect, stats = "F.p", type = 'exit')
```

##### Placebo Test

By selecting a part of the data and excluding observations within a specified range to improve the model fitting, we then evaluate whether the estimated Average Treatment Effect (ATT) within this range significantly differs from zero. This approach helps us analyze the periods before treatment.

If this test fails, either the functional form or strict exogeneity assumption is problematic.


```r
out.fect.p <-
    fect(
        Y = "y",
        D = "dem",
        X = "tradewb",
        data = na.omit(PanelMatch::dem),
        method = "fe",
        index = c("wbcode2", "year"),
        se = TRUE,
        placeboTest = TRUE,
        # using 3 periods
        placebo.period = c(-2, 0)
    )
plot(out.fect.p, proportion = 0.1, stats = "placebo.p")
```

##### (No) Carryover Effects Test

The placebo test can be adapted to assess carryover effects by masking several post-treatment periods instead of pre-treatment ones. If no carryover effects are present, the average prediction error should approximate zero. For the carryover test, set `carryoverTest = TRUE`. Specify a post-treatment period range in carryover.period to exclude observations for model fitting, then evaluate if the estimated ATT significantly deviates from zero.

Even if we have carryover effects, in most cases of the staggered adoption setting, researchers are interested in the cumulative effects, or aggregated treatment effects, so it's okay.


```r
out.fect.c <-
    fect(
        Y = "y",
        D = "dem",
        X = "tradewb",
        data = na.omit(PanelMatch::dem),
        method = "fe",
        index = c("wbcode2", "year"),
        se = TRUE,
        carryoverTest = TRUE,
        # how many periods of carryover
        carryover.period = c(1, 3)
    )
plot(out.fect.c,  stats = "carryover.p")
```

We have evidence of carryover effects.

#### Matrix Completion

Applications in marketing:

-   @bronnenberg2020consumer

To estimate average causal effects in panel data with units exposed to treatment intermittently, two literatures are pivotal:

-   **Unconfoundedness** [@imbens2015causal]: Imputes missing potential control outcomes for treated units using observed outcomes from similar control units in previous periods.

-   **Synthetic Control** [@abadie2010synthetic]: Imputes missing control outcomes for treated units using weighted averages from control units, matching lagged outcomes between treated and control units.

Both exploit missing potential outcomes under different assumptions:

-   Unconfoundedness assumes time patterns are stable across units.

-   Synthetic control assumes unit patterns are stable over time.

Once regularization is applied, both approaches are applicable in similar settings [@athey2021matrix].

**Matrix Completion** method, nesting both, is based on matrix factorization, focusing on imputing missing matrix elements assuming:

1.  Complete matrix = low-rank matrix + noise.
2.  Missingness is completely at random.

It's distinguished by not imposing factorization restrictions but utilizing regularization to define the estimator, particularly effective with the nuclear norm as a regularizer for complex missing patterns [@athey2021matrix].

Contributions of @athey2021matrix matrix completion include:

1.  Recognizing structured missing patterns allowing time correlation, enabling staggered adoption.
2.  Modifying estimators for unregularized unit and time fixed effects.
3.  Performing well across various $T$ and $N$ sizes, unlike unconfoundedness and synthetic control, which falter when $T >> N$ or $N >> T$, respectively.

Identifying Assumptions:

1.  SUTVA: Potential outcomes indexed only by the unit's contemporaneous treatment.
2.  No dynamic effects (it's okay under staggered adoption, it gives a different interpretation of estimand).

Setup:

-   $Y_{it}(0)$ and $Y_{it}(1)$ represent potential outcomes of $Y_{it}$.
-   $W_{it}$ is a binary treatment indicator.

Aim to estimate the average effect for the treated:

$$
\tau = \frac{\sum_{(i,t): W_{it} = 1}[Y_{it}(1) - Y_{it}(0)]}{\sum_{i,t}W_{it}}
$$

We observe all relevant values for $Y_{it}(1)$

We want to impute missing entries in the $Y(0)$ matrix for treated units with $W_{it} = 1$.

Define $\mathcal{M}$ as the set of pairs of indices $(i,t)$, where $i \in N$ and $t \in T$, corresponding to missing entries with $W_{it} = 1$; $\mathcal{O}$ as the set of pairs of indices corresponding to observed entries in $Y(0)$ with $W_{it} = 0$.

Data is conceptualized as two $N \times T$ matrices, one incomplete and one complete:

$$
Y = \begin{pmatrix}
Y_{11} & Y_{12} & ? & \cdots & Y_{1T} \\
? & ? & Y_{23} & \cdots & ? \\
Y_{31} & ? & Y_{33} & \cdots & ? \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
Y_{N1} & ? & Y_{N3} & \cdots & ?
\end{pmatrix},
$$

and

$$
W = \begin{pmatrix}
0 & 0 & 1 & \cdots & 0 \\
1 & 1 & 0 & \cdots & 1 \\
0 & 1 & 0 & \cdots & 1 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 1 & 0 & \cdots & 1
\end{pmatrix},
$$

where

$$
W_{it} =
\begin{cases}
1 & \text{if } (i,t) \in \mathcal{M}, \\
0 & \text{if } (i,t) \in \mathcal{O},
\end{cases}
$$

is an indicator for the event that the corresponding component of $Y$, that is $Y_{it}$, is missing.

Patterns of missing data in $\mathbf{Y}$:

-   Block (treatment) structure with 2 special cases

    -   Single-treated-period block structure [@imbens2015causal]

    -   Single-treated-unit block structure [@abadie2010synthetic]

-   Staggered Adoption

Shape of matrix $\mathbf{Y}$:

-   Thin ($N >> T$)

-   Fat ($T >> N$)

-   Square ($N \approx T$)

Combinations of patterns of missingness and shape create different literatures:

-   Horizontal Regression = Thin matrix + single-treated-period block (focusing on cross-section correlation patterns)

-   Vertical Regression = Fat matrix + single-treated-unit block (focusing on time-series correlation patterns)

-   TWFE = Square matrix

To combine, we can exploit both stable patterns over time, and across units (e.g., TWFE, interactive FEs or matrix completion).

For the same factor model

$$
\mathbf{Y = UV}^T + \mathbf{\epsilon}
$$

where $\mathbf{U}$ is $N \times R$ and $\mathbf{V}$ is $T\times R$

The interactive FE literature focuses on a fixed number of factors $R$ in $\mathbf{U, V}$, while matrix completion focuses on impute $\mathbf{Y}$ using some forms regularization (e.g., nuclear norm).

-   We can also estimate the number of factors $R$ [@bai2002determining, @moon2015linear]

To use the nuclear norm minimization estimator, we must add a penalty term to regularize the objective function. However, before doing so, we need to explicitly estimate the time ($\lambda_t$) and unit ($\mu_i$) fixed effects implicitly embedded in the missing data matrix to reduce the bias of the regularization term.

[Specifically](https://bookdown.org/stanfordgsbsilab/ml-ci-tutorial/matrix-completion-methods.html),

$$
Y_{it}  =L_{it} + \sum_{p = 1}^P \sum_{q= 1}^Q X_{ip} H_{pq}Z_{qt} + \mu_i + \lambda_t + V_{it} \beta + \epsilon_{it}
$$

where

-   $X_{ip}$ is a matrix of $p$ variables for unit $i$

-   $Z_{qt}$ is a matrix of $q$ variables for time $t$

-   $V_{it}$ is a matrix of time-varying variables.

Lasso-type $l_1$ norm ($||H|| = \sum_{p = 1}^p \sum_{q = 1}^Q |H_{pq}|$) is used to shrink $H \to 0$

There are several options to regularize $L$:

1.  Frobenius (i.e., Ridge): not informative since it imputes missing values as 0.
2.  Nuclear Norm (i.e., Lasso): computationally feasible (using SOFT-IMPUTE algorithm [@Mazumder2010SpectralRA]).
3.  Rank (i.e., Subset selection): not computationally feasible

This method allows to

-   use more covariates

-   leverage data from treated units (can be used when treatment effect is constant and pattern of missing is not complex).

-   have autocorrelated errors

-   have weighted loss function (i.e., take into account the probability of outcomes for a unit being missing)

### @gardner2022two and @borusyak2021revisiting

-   Estimate the time and unit fixed effects separately

-   Known as the imputation method [@borusyak2021revisiting] or two-stage DiD [@gardner2022two]


```r
# remotes::install_github("kylebutts/did2s")
library(did2s)
library(ggplot2)
library(fixest)
library(tidyverse)
data(base_stagg)


est <- did2s(
    data = base_stagg |> mutate(treat = if_else(time_to_treatment >= 0, 1, 0)),
    yname = "y",
    first_stage = ~ x1 | id + year,
    second_stage = ~ i(time_to_treatment, ref = c(-1,-1000)),
    treatment = "treat" ,
    cluster_var = "id"
)

fixest::esttable(est)
#>                                       est
#> Dependent Var.:                         y
#>                                          
#> time_to_treatment = -9  0.3518** (0.1332)
#> time_to_treatment = -8  -0.3130* (0.1213)
#> time_to_treatment = -7    0.0894 (0.2367)
#> time_to_treatment = -6    0.0312 (0.2176)
#> time_to_treatment = -5   -0.2079 (0.1519)
#> time_to_treatment = -4   -0.1152 (0.1438)
#> time_to_treatment = -3   -0.0127 (0.1483)
#> time_to_treatment = -2    0.1503 (0.1440)
#> time_to_treatment = 0  -5.139*** (0.3680)
#> time_to_treatment = 1  -3.480*** (0.3784)
#> time_to_treatment = 2  -2.021*** (0.3055)
#> time_to_treatment = 3   -0.6965. (0.3947)
#> time_to_treatment = 4    1.070** (0.3501)
#> time_to_treatment = 5   2.173*** (0.4456)
#> time_to_treatment = 6   4.449*** (0.3680)
#> time_to_treatment = 7   4.864*** (0.3698)
#> time_to_treatment = 8   6.187*** (0.2702)
#> ______________________ __________________
#> S.E. type                          Custom
#> Observations                          950
#> R2                                0.62486
#> Adj. R2                           0.61843
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

fixest::iplot(
    est,
    main = "Event study",
    xlab = "Time to treatment",
    ref.line = -1
)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-39-1.png" width="90%" style="display: block; margin: auto;" />

```r

coefplot(est)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-39-2.png" width="90%" style="display: block; margin: auto;" />


```r
mult_est <- did2s::event_study(
    data = fixest::base_stagg |>
        dplyr::mutate(year_treated = dplyr::if_else(year_treated == 10000, 0, year_treated)),
    gname = "year_treated",
    idname = "id",
    tname = "year",
    yname = "y",
    estimator = "all"
)
#> Error in purrr::map(., function(y) { : ℹ In index: 1.
#> ℹ With name: y.
#> Caused by error in `.subset2()`:
#> ! no such index at level 1
did2s::plot_event_study(mult_est)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-40-1.png" width="90%" style="display: block; margin: auto;" />

@borusyak2021revisiting `didimputation`

This version is currently not working


```r
library(didimputation)
library(fixest)
data("base_stagg")

did_imputation(
    data = base_stagg,
    yname = "y",
    gname = "year_treated",
    tname = "year",
    idname = "id"
)
```

### @de2020two

use `twowayfeweights` from [GitHub](https://github.com/shuo-zhang-ucsb/twowayfeweights) [@de2020two]

-   Average instant treatment effect of changes in the treatment

    -   This relaxes the no-carryover-effect assumption.

-   Drawbacks:

    -   Cannot observe treatment effects that manifest over time.

There still isn't a good package for this estimator.


```r
# remotes::install_github("shuo-zhang-ucsb/did_multiplegt") 
library(DIDmultiplegt)
library(fixest)
library(tidyverse)

data("base_stagg")

res <-
    did_multiplegt(
        df = base_stagg |>
            dplyr::mutate(treatment = dplyr::if_else(time_to_treatment < 0, 0, 1)),
        Y        = "y",
        G        = "year_treated",
        T        = "year",
        D        = "treatment",
        controls = "x1",
        # brep     = 20, # getting SE will take forever
        placebo  = 5,
        dynamic  = 5, 
        average_effect = "simple"
    )

head(res)
#> $effect
#> treatment 
#> -5.214207 
#> 
#> $N_effect
#> [1] 675
#> 
#> $N_switchers_effect
#> [1] 45
#> 
#> $dynamic_1
#> [1] -3.63556
#> 
#> $N_dynamic_1
#> [1] 580
#> 
#> $N_switchers_effect_1
#> [1] 40
```

I don't recommend the `TwoWayFEWeights` since it only gives the aggregated average treatment effect over all post-treatment periods, but not for each period.


```r
library(TwoWayFEWeights)

res <- twowayfeweights(
    data = base_stagg |> dplyr::mutate(treatment = dplyr::if_else(time_to_treatment < 0, 0, 1)),
    Y = "y",
    G = "year_treated",
    T = "year",
    D = "treatment", 
    summary_measures = T
)

print(res)
#> Under the common trends assumption, beta estimates a weighted sum of 45 ATTs.
#> 41 ATTs receive a positive weight, and 4 receive a negative weight.
#> 
#> ────────────────────────────────────────── 
#> Treat. var: treatment    ATTs    Σ weights 
#> ────────────────────────────────────────── 
#> Positive weights           41       1.0238 
#> Negative weights            4      -0.0238 
#> ────────────────────────────────────────── 
#> Total                      45            1 
#> ──────────────────────────────────────────
#> 
#> Summary Measures:
#>   TWFE Coefficient (β_fe): -3.4676
#>   min σ(Δ) compatible with β_fe and Δ_TR = 0: 4.8357
#>   min σ(Δ) compatible with β_fe and Δ_TR of a different sign: 36.1549
#>   Reference: Corollary 1, de Chaisemartin, C and D'Haultfoeuille, X (2020a)
#> 
#> The development of this package was funded by the European Union (ERC, REALLYCREDIBLE,GA N. 101043899).
```

### @callaway2021difference {#callaway2021difference}

-   `staggered` [package](https://github.com/jonathandroth/staggered)

-   Group-time average treatment effect


```r
library(staggered) 
library(fixest)
data("base_stagg")

# simple weighted average
staggered(
    df = base_stagg,
    i = "id",
    t = "year",
    g = "year_treated",
    y = "y",
    estimand = "simple"
)
#>     estimate        se se_neyman
#> 1 -0.7110941 0.2211943 0.2214245

# cohort weighted average
staggered(
    df = base_stagg,
    i = "id",
    t = "year",
    g = "year_treated",
    y = "y",
    estimand = "cohort"
)
#>    estimate        se se_neyman
#> 1 -2.724242 0.2701093 0.2701745

# calendar weighted average
staggered(
    df = base_stagg,
    i = "id",
    t = "year",
    g = "year_treated",
    y = "y",
    estimand = "calendar"
)
#>     estimate        se se_neyman
#> 1 -0.5861831 0.1768297 0.1770729

res <- staggered(
    df = base_stagg,
    i = "id",
    t = "year",
    g = "year_treated",
    y = "y",
    estimand = "eventstudy", 
    eventTime = -9:8
)
head(res)
#>      estimate        se se_neyman eventTime
#> 1  0.20418779 0.1045821 0.1045821        -9
#> 2 -0.06215104 0.1669703 0.1670886        -8
#> 3  0.02744671 0.1413273 0.1420377        -7
#> 4 -0.02131747 0.2203695 0.2206338        -6
#> 5 -0.30690897 0.2015697 0.2036412        -5
#> 6  0.05594029 0.1908101 0.1921745        -4


ggplot(
    res |> mutate(
        ymin_ptwise = estimate + 1.96 * se,
        ymax_ptwise = estimate - 1.96 * se
    ),
    aes(x = eventTime, y = estimate)
) +
    geom_pointrange(aes(ymin = ymin_ptwise, ymax = ymax_ptwise)) +
    geom_hline(yintercept = 0) +
    xlab("Event Time") +
    ylab("Estimate") +
    causalverse::ama_theme()
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-44-1.png" width="90%" style="display: block; margin: auto;" />


```r
# Callaway and Sant'Anna estimator for the simple weighted average
staggered_cs(
    df = base_stagg,
    i = "id",
    t = "year",
    g = "year_treated",
    y = "y",
    estimand = "simple"
)
#>     estimate        se se_neyman
#> 1 -0.7994889 0.4484987 0.4486122

# Sun and Abraham estimator for the simple weighted average
staggered_sa(
    df = base_stagg,
    i = "id",
    t = "year",
    g = "year_treated",
    y = "y",
    estimand = "simple"
)
#>     estimate        se se_neyman
#> 1 -0.7551901 0.4407818 0.4409525
```

Fisher's Randomization Test (i.e., permutation test)

$H_0$: $TE = 0$


```r
staggered(
    df = base_stagg,
    i = "id",
    t = "year",
    g = "year_treated",
    y = "y",
    estimand = "simple",
    compute_fisher = T,
    num_fisher_permutations = 100
)
#>     estimate        se se_neyman fisher_pval fisher_pval_se_neyman
#> 1 -0.7110941 0.2211943 0.2214245           0                     0
#>   num_fisher_permutations
#> 1                     100
```

### @sun2021estimating

This paper utilizes the Cohort Average Treatment Effects on the Treated (CATT), which measures the cohort-specific average difference in outcomes relative to those never treated, offering a more detailed analysis than @goodman2021difference. In scenarios lacking a never-treated group, this method designates the last cohort to be treated as the control group.

Parameter of interest is the cohort-specific ATT $l$ periods from int ital treatment period $e$

$$
CATT = E[Y_{i, e + I} - Y_{i, e + I}^\infty|E_i = e]
$$

This paper uses an **interaction-weighted estimator** in a panel data setting, where the original paper @gibbons2018broken used the same idea in a cross-sectional setting.

-   @callaway2021difference explores group-time average treatment effects, employing cohorts that have not yet been treated as controls, and permits conditioning on time-varying covariates.

-   @athey2022design examines the treatment effect in relation to the counterfactual outcome of the always-treated group, diverging from the conventional focus on the never-treated.

-   @borusyak2021revisiting presumes a uniform treatment effect across cohorts, effectively simplifying CATT to ATT.

Identifying Assumptions for dynamic TWFE:

1.  **Parallel Trends**: Baseline outcomes follow parallel trends across cohorts before treatment.

    -   This gives us all CATT (including own, included bins, and excluded bins)

2.  **No Anticipatory Behavior**: There is no effect of the treatment during pre-treatment periods, indicating that outcomes are not influenced by the anticipation of treatment.

3.  **Treatment Effect Homogeneity**: The treatment effect is consistent across cohorts for each relative period. Each adoption cohort should have the same path of treatment effects. In other words, the trajectory of each treatment cohort is similar. Compare to other designs:

    1.  @athey2022design assume heterogeneity of treatment effects vary over adoption cohorts, but not over time.

    2.  @borusyak2021revisiting assume heterogeneity of treatment effects vary over time, but not over adoption cohorts.

    3.  @callaway2021difference assume heterogeneity of treatment effects vary over time and across cohorts.

    4.  @de2023two assume heterogeneity of treatment effects vary across groups and over time.

    5.  @goodman2021difference assume heterogeneity either "vary across units but not over time" or "vary over time but not across units".

    6.  @sun2021estimating allows for treatment effect heterogeneity across units and time.

Sources of Heterogeneous Treatment Effects

-   Adoption cohorts can differ based on certain covariates. Similarly, composition of units within each adoption cohort is different.

-   The response to treatment varies among cohorts if units self-select their initial treatment timing based on anticipated treatment effects. However, this self-selection is still compatible with the parallel trends assumption. This is true if units choose based on an evaluation of baseline outcomes - that is, if baseline outcomes are similar (following parallel trends), then we might not see selection into treatment based on the evaluation of the baseline outcome.

-   Treatment effects can vary across cohorts due to calendar time-varying effects, such as changes in economic conditions.

Notes:

-   If you do TWFE, you actually have to drop 2 terms to avoid multicollinearity:

    -   Period right before treatment (this one was known before this paper)

    -   Drop or bin or trim a distant lag period (this one was clarified by the paper). The reason is before of the multicollinearity in the linear relationship between TWFE and the relative period indicators.

-   Contamination of the treatment effect estimates from excluded periods is a type of "normalization". To avoid this, we have to assume that all pre-treatment periods have the same CATT.

    -   @sun2021estimating estimation method gives reasonable weights to CATT (i..e, weights that sum to 1, and are non negative). They estimate the weighted average of CATT where the weights are shares of cohorts that experience at least $l$ periods after to treatment, normalized by the size of total periods $g$.

-   Aggregation of CATT is similar to that of @callaway2021difference

**Application**

can use `fixest` in r with `sunab` function


```r
library(fixest)
data("base_stagg")
res_sa20 = feols(y ~ x1 + sunab(year_treated, year) | id + year, base_stagg)
iplot(res_sa20)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-47-1.png" width="90%" style="display: block; margin: auto;" />

```r

summary(res_sa20, agg = "att")
#> OLS estimation, Dep. Var.: y
#> Observations: 950
#> Fixed-effects: id: 95,  year: 10
#> Standard-errors: Clustered (id) 
#>      Estimate Std. Error  t value  Pr(>|t|)    
#> x1   0.994678   0.018378 54.12293 < 2.2e-16 ***
#> ATT -1.133749   0.205070 -5.52858 2.882e-07 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> RMSE: 0.921817     Adj. R2: 0.887984
#>                  Within R2: 0.876406


summary(res_sa20, agg = c("att" = "year::[^-]")) 
#> OLS estimation, Dep. Var.: y
#> Observations: 950
#> Fixed-effects: id: 95,  year: 10
#> Standard-errors: Clustered (id) 
#>                      Estimate Std. Error   t value   Pr(>|t|)    
#> x1                   0.994678   0.018378 54.122928  < 2.2e-16 ***
#> year::-9:cohort::10  0.351766   0.359073  0.979649 3.2977e-01    
#> year::-8:cohort::9   0.033914   0.471437  0.071937 9.4281e-01    
#> year::-8:cohort::10 -0.191932   0.352896 -0.543876 5.8781e-01    
#> year::-7:cohort::8  -0.589387   0.736910 -0.799809 4.2584e-01    
#> year::-7:cohort::9   0.872995   0.493427  1.769249 8.0096e-02 .  
#> year::-7:cohort::10  0.019512   0.603411  0.032336 9.7427e-01    
#> year::-6:cohort::7  -0.042147   0.865736 -0.048683 9.6127e-01    
#> year::-6:cohort::8  -0.657571   0.573257 -1.147078 2.5426e-01    
#> year::-6:cohort::9   0.877743   0.533331  1.645775 1.0315e-01    
#> year::-6:cohort::10 -0.403635   0.347412 -1.161832 2.4825e-01    
#> year::-5:cohort::6  -0.658034   0.913407 -0.720418 4.7306e-01    
#> year::-5:cohort::7  -0.316974   0.697939 -0.454158 6.5076e-01    
#> year::-5:cohort::8  -0.238213   0.469744 -0.507113 6.1326e-01    
#> year::-5:cohort::9   0.301477   0.604201  0.498968 6.1897e-01    
#> year::-5:cohort::10 -0.564801   0.463214 -1.219308 2.2578e-01    
#> year::-4:cohort::5  -0.983453   0.634492 -1.549984 1.2451e-01    
#> year::-4:cohort::6   0.360407   0.858316  0.419900 6.7552e-01    
#> year::-4:cohort::7  -0.430610   0.661356 -0.651102 5.1657e-01    
#> year::-4:cohort::8  -0.895195   0.374901 -2.387816 1.8949e-02 *  
#> year::-4:cohort::9  -0.392478   0.439547 -0.892914 3.7418e-01    
#> year::-4:cohort::10  0.519001   0.597880  0.868069 3.8757e-01    
#> year::-3:cohort::4   0.591288   0.680169  0.869324 3.8688e-01    
#> year::-3:cohort::5  -1.000650   0.971741 -1.029749 3.0577e-01    
#> year::-3:cohort::6   0.072188   0.652641  0.110609 9.1216e-01    
#> year::-3:cohort::7  -0.836820   0.804275 -1.040465 3.0079e-01    
#> year::-3:cohort::8  -0.783148   0.701312 -1.116691 2.6697e-01    
#> year::-3:cohort::9   0.811285   0.564470  1.437251 1.5397e-01    
#> year::-3:cohort::10  0.527203   0.320051  1.647250 1.0285e-01    
#> year::-2:cohort::3   0.036941   0.673771  0.054828 9.5639e-01    
#> year::-2:cohort::4   0.832250   0.859544  0.968246 3.3541e-01    
#> year::-2:cohort::5  -1.574086   0.525563 -2.995051 3.5076e-03 ** 
#> year::-2:cohort::6   0.311758   0.832095  0.374666 7.0875e-01    
#> year::-2:cohort::7  -0.558631   0.871993 -0.640638 5.2332e-01    
#> year::-2:cohort::8   0.429591   0.305270  1.407250 1.6265e-01    
#> year::-2:cohort::9   1.201899   0.819186  1.467188 1.4566e-01    
#> year::-2:cohort::10 -0.002429   0.682087 -0.003562 9.9717e-01    
#> att                 -1.133749   0.205070 -5.528584 2.8820e-07 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> RMSE: 0.921817     Adj. R2: 0.887984
#>                  Within R2: 0.876406

# alternatively
summary(res_sa20, agg = c("att" = "year::[012345678]")) |> 
    etable(digits = 2)
#>                         summary(res_..
#> Dependent Var.:                      y
#>                                       
#> x1                      0.99*** (0.02)
#> year = -9 x cohort = 10    0.35 (0.36)
#> year = -8 x cohort = 9     0.03 (0.47)
#> year = -8 x cohort = 10   -0.19 (0.35)
#> year = -7 x cohort = 8    -0.59 (0.74)
#> year = -7 x cohort = 9    0.87. (0.49)
#> year = -7 x cohort = 10    0.02 (0.60)
#> year = -6 x cohort = 7    -0.04 (0.87)
#> year = -6 x cohort = 8    -0.66 (0.57)
#> year = -6 x cohort = 9     0.88 (0.53)
#> year = -6 x cohort = 10   -0.40 (0.35)
#> year = -5 x cohort = 6    -0.66 (0.91)
#> year = -5 x cohort = 7    -0.32 (0.70)
#> year = -5 x cohort = 8    -0.24 (0.47)
#> year = -5 x cohort = 9     0.30 (0.60)
#> year = -5 x cohort = 10   -0.56 (0.46)
#> year = -4 x cohort = 5    -0.98 (0.63)
#> year = -4 x cohort = 6     0.36 (0.86)
#> year = -4 x cohort = 7    -0.43 (0.66)
#> year = -4 x cohort = 8   -0.90* (0.37)
#> year = -4 x cohort = 9    -0.39 (0.44)
#> year = -4 x cohort = 10    0.52 (0.60)
#> year = -3 x cohort = 4     0.59 (0.68)
#> year = -3 x cohort = 5     -1.0 (0.97)
#> year = -3 x cohort = 6     0.07 (0.65)
#> year = -3 x cohort = 7    -0.84 (0.80)
#> year = -3 x cohort = 8    -0.78 (0.70)
#> year = -3 x cohort = 9     0.81 (0.56)
#> year = -3 x cohort = 10    0.53 (0.32)
#> year = -2 x cohort = 3     0.04 (0.67)
#> year = -2 x cohort = 4     0.83 (0.86)
#> year = -2 x cohort = 5   -1.6** (0.53)
#> year = -2 x cohort = 6     0.31 (0.83)
#> year = -2 x cohort = 7    -0.56 (0.87)
#> year = -2 x cohort = 8     0.43 (0.31)
#> year = -2 x cohort = 9      1.2 (0.82)
#> year = -2 x cohort = 10  -0.002 (0.68)
#> att                     -1.1*** (0.21)
#> Fixed-Effects:          --------------
#> id                                 Yes
#> year                               Yes
#> _______________________ ______________
#> S.E.: Clustered                 by: id
#> Observations                       950
#> R2                             0.90982
#> Within R2                      0.87641
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Using the same syntax as `fixest`


```r
# devtools::install_github("kylebutts/fwlplot")
library(fwlplot)
fwl_plot(y ~ x1, data = base_stagg)
```

<img src="29-dif-in-dif_files/figure-html/plot residuals-1.png" width="90%" style="display: block; margin: auto;" />

```r

fwl_plot(y ~ x1 | id + year, data = base_stagg, n_sample = 100)
```

<img src="29-dif-in-dif_files/figure-html/plot residuals-2.png" width="90%" style="display: block; margin: auto;" />

```r

fwl_plot(y ~ x1 | id + year, data = base_stagg, n_sample = 100, fsplit = ~ treated)
```

<img src="29-dif-in-dif_files/figure-html/plot residuals-3.png" width="90%" style="display: block; margin: auto;" />

### @wooldridge2022simple

use [etwfe](https://grantmcdermott.com/etwfe/)(Extended two-way Fixed Effects) [@wooldridge2022simple]

### Doubly Robust DiD

Also known as the locally efficient doubly robust DiD [@sant2020doubly]

[Code example by the authors](https://psantanna.com/DRDID/index.html)

The package (not method) is rather limited application:

-   Use OLS (cannot handle `glm`)

-   Canonical DiD only (cannot handle DDD).


```r
library(DRDID)
data("nsw_long")
eval_lalonde_cps <-
    subset(nsw_long, nsw_long$treated == 0 | nsw_long$sample == 2)
head(eval_lalonde_cps)
#>   id year treated age educ black married nodegree dwincl      re74 hisp
#> 1  1 1975      NA  42   16     0       1        0     NA     0.000    0
#> 2  1 1978      NA  42   16     0       1        0     NA     0.000    0
#> 3  2 1975      NA  20   13     0       0        0     NA  2366.794    0
#> 4  2 1978      NA  20   13     0       0        0     NA  2366.794    0
#> 5  3 1975      NA  37   12     0       1        0     NA 25862.322    0
#> 6  3 1978      NA  37   12     0       1        0     NA 25862.322    0
#>   early_ra sample experimental         re
#> 1       NA      2            0     0.0000
#> 2       NA      2            0   100.4854
#> 3       NA      2            0  3317.4678
#> 4       NA      2            0  4793.7451
#> 5       NA      2            0 22781.8555
#> 6       NA      2            0 25564.6699


# locally efficient doubly robust DiD Estimators for the ATT
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
#>  Call:
#> drdid(yname = "re", tname = "year", idname = "id", dname = "experimental", 
#>     xformla = ~age + educ + black + married + nodegree + hisp + 
#>         re74, data = eval_lalonde_cps, panel = TRUE)
#> ------------------------------------------------------------------
#>  Further improved locally efficient DR DID estimator for the ATT:
#>  
#>    ATT     Std. Error  t value    Pr(>|t|)  [95% Conf. Interval] 
#> -901.2703   393.6247   -2.2897     0.022    -1672.7747  -129.766 
#> ------------------------------------------------------------------
#>  Estimator based on panel data.
#>  Outcome regression est. method: weighted least squares.
#>  Propensity score est. method: inverse prob. tilting.
#>  Analytical standard error.
#> ------------------------------------------------------------------
#>  See Sant'Anna and Zhao (2020) for details.



# Improved locally efficient doubly robust DiD estimator 
# for the ATT, with panel data
# drdid_imp_panel()

# Locally efficient doubly robust DiD estimator for the ATT, 
# with panel data
# drdid_panel()

# Locally efficient doubly robust DiD estimator for the ATT, 
# with repeated cross-section data
# drdid_rc()

# Improved locally efficient doubly robust DiD estimator for the ATT, 
# with repeated cross-section data
# drdid_imp_rc()
```

### Augmented/Forward DID

-   DID Methods for Limited Pre-Treatment Periods:

+--------------------+---------------------------------------------------------+-----------------------------------------------------------------------------------------+
| **Method**         | **Scenario**                                            | **Approach**                                                                            |
+====================+=========================================================+=========================================================================================+
| **Augmented DID**  | Treatment outcome is outside the range of control units | Constructs the treatment counterfactual using a scaled average of control units         |
|                    |                                                         |                                                                                         |
| [@li2023augmented] |                                                         |                                                                                         |
+--------------------+---------------------------------------------------------+-----------------------------------------------------------------------------------------+
| **Forward DID**    | Treatment outcome is within the range of control units  | Uses a forward selection algorithm to choose relevant control units before applying DID |
|                    |                                                         |                                                                                         |
| [@li2024frontiers] |                                                         |                                                                                         |
+--------------------+---------------------------------------------------------+-----------------------------------------------------------------------------------------+

## Multiple Treatments

When you have 2 treatments in a setting, you should always try to model both of them under one regression to see whether they are significantly different.

-   Never use one treated groups as control for the other, and run separate regression.
-   Could check this [answer](https://stats.stackexchange.com/questions/474533/difference-in-difference-with-two-treatment-groups-and-one-control-group-classi)

$$
\begin{aligned}
Y_{it} &= \alpha + \gamma_1 Treat1_{i} + \gamma_2 Treat2_{i} + \lambda Post_t  \\
&+ \delta_1(Treat1_i \times Post_t) + \delta_2(Treat2_i \times Post_t) + \epsilon_{it}
\end{aligned}
$$

[@fricke2017identification]

[@de2023two] [video](https://www.youtube.com/watch?v=UHeJoc27qEM&ab_channel=TaylorWright) [code](https://drive.google.com/file/d/156Fu73avBvvV_H64wePm7eW04V0jEG3K/view)

## Mediation Under DiD

Check this [post](https://stats.stackexchange.com/questions/261218/difference-in-difference-model-with-mediators-estimating-the-effect-of-differen)

## Assumptions

-   **Parallel Trends**: Difference between the treatment and control groups remain constant if there were no treatment.

    -   should be used in cases where

        -   you observe before and after an event

        -   you have treatment and control groups

    -   not in cases where

        -   treatment is not random

        -   confounders.

    -   To support we use

        -   [Placebo test]

        -   [Prior Parallel Trends Test](#prior-parallel-trends-test)

-   **Linear additive effects** (of group/unit specific and time-specific):

    -   If they are not additively interact, we have to use the weighted 2FE estimator [@imai2021use]

    -   Typically seen in the [Staggered Dif-n-dif]

-   No anticipation: There is no causal effect of the treatment before its implementation.

**Possible issues**

-   Estimate dependent on functional form:

    -   When the size of the response depends (nonlinearly) on the size of the intervention, we might want to look at the the difference in the group with high intensity vs. low.

-   Selection on (time--varying) unobservables

    -   Can use the overall sensitivity of coefficient estimates to hidden bias using [Rosenbaum Bounds]

-   Long-term effects

    -   Parallel trends are more likely to be observed over shorter period (window of observation)

-   Heterogeneous effects

    -   Different intensity (e.g., doses) for different groups.

-   Ashenfelter dip [@ashenfelter1985] (job training program participant are more likely to experience an earning drop prior enrolling in these programs)

    -   Participants are systemically different from nonparticipants before the treatment, leading to the question of permanent or transitory changes.
    -   A fix to this transient endogeneity is to calculate long-run differences (exclude a number of periods symmetrically around the adoption/ implementation date). If we see a sustained impact, then we have strong evidence for the causal impact of a policy. [@proserpio2017] [@heckman1999c] [@jepsen2014] [@li2011]

-   Response to event might not be immediate (can't be observed right away in the dependent variable)

    -   Using lagged dependent variable $Y_{it-1}$ might be more appropriate [@blundell1998initial]

-   Other factors that affect the difference in trends between the two groups (i.e., treatment and control) will bias your estimation.

-   Correlated observations within a group or time

-   Incidental parameters problems [@lancaster2000incidental]: it's always better to use individual and time fixed effect.

-   When examining the effects of variation in treatment timing, we have to be careful because negative weights (per group) can be negative if there is a heterogeneity in the treatment effects over time. Example: [@athey2022design][@borusyak2021revisiting][@goodman2021difference]. In this case you should use new estimands proposed by [\@callaway2021difference](#callaway2021difference)[@de2020two], in the `did` package. If you expect lags and leads, see [@sun2021estimating]

-   [@gibbons2018broken] caution when we suspect the treatment effect and treatment variance vary across groups

### Prior Parallel Trends Test {#prior-parallel-trends-test}

1.  Plot the average outcomes over time for both treatment and control group before and after the treatment in time.
2.  Statistical test for difference in trends (**using data from before the treatment period**)

$$
Y = \alpha_g + \beta_1 T + \beta_2 T\times G + \epsilon
$$

where

-   $Y$ = the outcome variable

-   $\alpha_g$ = group fixed effects

-   $T$ = time (e.g., specific year, or month)

-   $\beta_2$ = different time trends for each group

Hence, if $\beta_2 =0$ provides evidence that there are no differences in the trend for the two groups prior the time treatment.

You can also use different functional forms (e..g, polynomial or nonlinear).

If $\beta_2 \neq 0$ statistically, possible reasons can be:

-   Statistical significance can be driven by large sample

-   Or the trends are so consistent, and just one period deviation can throw off the trends. Hence, statistical statistical significance.

Technically, we can still salvage the research by including time fixed effects, instead of just the before-and-after time fixed effect (actually, most researchers do this mechanically anyway nowadays). However, a side effect can be that the time fixed effects can also absorb some part your treatment effect as well, especially in cases where the treatment effects vary with time (i.e., stronger or weaker over time) [@wolfers2003business].

Debate:

-   [@kahn2020promise] argue that DiD will be more plausible when the treatment and control groups are similar not only in **trends**, but also in **levels**. Because when we observe dissimilar in levels prior to the treatment, why is it okay to think that this will not affect future trends?

    -   Show a plot of the dependent variable's time series for treated and control groups and also a similar plot with matched sample. [@ryan2019now] show evidence of matched DiD did well in the setting of non-parallel trends (at least in health care setting).

    -   In the case that we don't have similar levels ex ante between treatment and control groups, functional form assumptions matter and we need justification for our choice.

-   Pre-trend statistical tests: [@roth2022pretest] provides evidence that these test are usually under powered.

    -   See [PretrendsPower](https://github.com/jonathandroth/PretrendsPower) and [pretrends](https://github.com/jonathandroth/pretrends) packages for correcting this.

-   Parallel trends assumption is specific to both the transformation and units of the outcome [@roth2023parallel]

    -   See falsification test ($H_0$: parallel trends is insensitive to functional form).


```r
library(tidyverse)
library(fixest)
od <- causaldata::organ_donations %>%
    # Use only pre-treatment data
    filter(Quarter_Num <= 3) %>% 
    # Treatment variable
    dplyr::mutate(California = State == 'California')

# use my package
causalverse::plot_par_trends(
    data = od,
    metrics_and_names = list("Rate" = "Rate"),
    treatment_status_var = "California",
    time_var = list(Quarter_Num = "Time"),
    display_CI = F
)
#> [[1]]
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-49-1.png" width="90%" style="display: block; margin: auto;" />

```r

# do it manually
# always good but plot the dependent out
od |>
    # group by treatment status and time
    dplyr::group_by(California, Quarter) |>
    dplyr::summarize_all(mean) |>
    dplyr::ungroup() |>
    # view()
    
    ggplot2::ggplot(aes(x = Quarter_Num, y = Rate, color = California)) +
    ggplot2::geom_line() +
    causalverse::ama_theme()
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-49-2.png" width="90%" style="display: block; margin: auto;" />

```r


# but it's also important to use statistical test
prior_trend <- fixest::feols(Rate ~ i(Quarter_Num, California) | State + Quarter,
               data = od)

fixest::coefplot(prior_trend, grid = F)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-49-3.png" width="90%" style="display: block; margin: auto;" />

```r
fixest::iplot(prior_trend, grid = F)
```

<img src="29-dif-in-dif_files/figure-html/unnamed-chunk-49-4.png" width="90%" style="display: block; margin: auto;" />

This is alarming since one of the periods is significantly different from 0, which means that our parallel trends assumption is not plausible.

In cases where you might have violations of parallel trends assumption, check [@rambachan2023more]

-   Impose restrictions on how different the post-treatment violations of parallel trends can be from the pre-trends.

-   Partial identification of causal parameter

-   Sensitivity analysis


```r
# https://github.com/asheshrambachan/HonestDiD
# remotes::install_github("asheshrambachan/HonestDiD")
# library(HonestDiD)
```

Alternatively, @ban2022generalized propose a method that with an information set (i.e., pre-treatment covariates), and an assumption on the selection bias in the post-treatment period (i.e., lies within the convex hull of all selection biases), they can still identify a set of ATT, and with stricter assumption on selection bias from the policymakers perspective, they can also have a point estimate.

Alternatively, we can use the `pretrends` package to examine our assumptions [@roth2022pretest]

### Placebo Test

Procedure:

1.  Sample data only in the period before the treatment in time.
2.  Consider different fake cutoff in time, either
    1.  Try the whole sequence in time

    2.  Generate random treatment period, and use **randomization inference** to account for sampling distribution of the fake effect.
3.  Estimate the DiD model but with the post-time = 1 with the fake cutoff
4.  A significant DiD coefficient means that you violate the parallel trends! You have a big problem.

Alternatively,

-   When data have multiple control groups, drop the treated group, and assign another control group as a "fake" treated group. But even if it fails (i.e., you find a significant DiD effect) among the control groups, it can still be fine. However, this method is used under [Synthetic Control]

[Code by theeffectbook.net](https://theeffectbook.net/ch-DifferenceinDifference.html)


```r
library(tidyverse)
library(fixest)

od <- causaldata::organ_donations %>%
    # Use only pre-treatment data
    dplyr::filter(Quarter_Num <= 3) %>%
    
    # Create fake treatment variables
    dplyr::mutate(
        FakeTreat1 = State == 'California' &
            Quarter %in% c('Q12011', 'Q22011'),
        FakeTreat2 = State == 'California' &
            Quarter == 'Q22011'
    )


clfe1 <- fixest::feols(Rate ~ FakeTreat1 | State + Quarter,
               data = od)
clfe2 <- fixest::feols(Rate ~ FakeTreat2 | State + Quarter,
               data = od)

fixest::etable(clfe1,clfe2)
#>                           clfe1            clfe2
#> Dependent Var.:            Rate             Rate
#>                                                 
#> FakeTreat1TRUE  0.0061 (0.0051)                 
#> FakeTreat2TRUE                  -0.0017 (0.0028)
#> Fixed-Effects:  --------------- ----------------
#> State                       Yes              Yes
#> Quarter                     Yes              Yes
#> _______________ _______________ ________________
#> S.E.: Clustered       by: State        by: State
#> Observations                 81               81
#> R2                      0.99377          0.99376
#> Within R2               0.00192          0.00015
#> ---
#> Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

We would like the "supposed" DiD to be insignificant.

### Assumption Violations

1.  Endogenous Timing

If the timing of units can be influenced by strategic decisions in a DID analysis, an instrumental variable approach with a control function can be used to control for endogeneity in timing.

2.  Questionable Counterfactuals

In situations where the control units may not serve as a reliable counterfactual for the treated units, matching methods such as propensity score matching or generalized random forest can be utilized. Additional methods can be found in [Matching Methods].

### Robustness Checks

-   Placebo DiD (if the DiD estimate $\neq 0$, parallel trend is violated, and original DiD is biased):

    -   Group: Use fake treatment groups: A population that was **not** affect by the treatment

    -   Time: Redo the DiD analysis for period before the treatment (expected treatment effect is 0) (e.g., for previous year or period).

-   Possible alternative control group: Expected results should be similar

-   Try different windows (further away from the treatment point, other factors can creep in and nullify your effect).

-   Treatment Reversal (what if we don't see the treatment event)

-   Higher-order polynomial time trend (to relax linearity assumption)

-   Test whether other dependent variables that should not be affected by the event are indeed unaffected.

    -   Use the same control and treatment period (DiD $\neq0$, there is a problem)

-   The **triple-difference strategy** involves examining the interaction between the **treatment variable** and **the probability of being affected by the program**, and the group-level participation rate. The identification assumption is that there are no differential trends between high and low participation groups in early versus late implementing countries.
