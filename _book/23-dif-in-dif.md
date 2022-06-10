# Difference-in-differences

Examples in marketing

-   [@liaukonyte2015]: TV ad on online shopping
-   [@akca2020]: aggregators for airlines business effect
-   [@pattabhiramaiah2018]: paywall affects readership
-   [@wang2018]: political ad source and message tone on vote shares and turnout using discontinuities in the level of political ads at the borders
-   [@datta2018]: streaming service on total music consumption using timing of users adoption of a music streaming service
-   [@janakiraman2018]: data breach announcement affect customer spending using timing of data breach and variation whether customer info was breached in that event
-   [@lim2020]: nutritional labels on nutritional quality for other brands in a category using variation in timing of adoption of nutritional labels across categories
-   [@guo2020]: payment disclosure laws effect on physician prescription behavior using Timing of the Massachusetts open payment law as the exogenous shock
-   [@israeli2018]: digital monitoring and enforcement on violations using enforcement of min ad price policies
-   [@ramani2019]: firms respond to foreign direct investment liberalization using India's reform in 1991.
-   [@he2022]: using Aamzon policy change to examine the causal impact of fake reviews on sales, average ratings.
-   [@peukert2022]: using European General data proection Regulation, examien the impact of policy change on website usage.

Show the mechanism via

-   Mediation analysis: see [@habel2021]

-   Moderation analysis: see [@goldfarb2011]

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
E[Y_1(1) - Y_0(1)|D=1] \\
= \{E[Y(1)|D=1] - E[Y(1)|D=0] \} - \{E[Y(0)|D=1] - E[Y(0)|D=0] \}
$$

More elaboration:

-   For the treatment group, we isolate the difference between being treated and not being treated. If the untreated group would have been affected in a different way, the DiD design and estimate would tell us nothing.

<br>

**Extension**

More than 2 groups (multiple treatments and multiple controls), and more than 2 period (pre and post)

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

<br>

[Matching Methods]

-   Match treatment and control based on pre-treatment observables

-   Modify SEs appropriately [@heckman1997]

<br>

### Assumptions

-   **Parallel Trends**: Difference between the treatment and control groups remain constant if there were no treatment.

    -   should be used in cases where

        -   you observe before and after an event

        -   you have treatment and control groups

    -   not in cases where

        -   treatment is not random

        -   confounders.

    -   To support we use

        -   [Placebo test]

        -   [Prior trends test]

-   **Linear additive effects** (of group/unit specific and time-specific):

    -   If they are not additively interact, we have to use the weighted 2FE estimator [@imai2020]

    -   Typically seen in the [Staggered Dif-n-dif]

-   No anticipation: There is no causal effect of the treatment before its implementation.

#### Prior Trends Test

1.  Plot the average outcomes over time for both treatment and control group before and after the treatment in time.
2.  Statistical test for difference in trends (using data from before the treatment period)

$$
Y = \alpha_g + \beta_1 T + \beta_2 T\times G + \epsilon
$$

where

-   $Y$ = the outcome variable

-   $\alpha_g$ = group fixed effects

-   $T$ = time (e.g., specific year, or month)

-   $\beta_2$ = different time trend for each group

Hence, if $\beta_2 =0$ provides evidence that there are no difference in the trend for the two groups prior the time treatment.

You can also use different functional forms (e..g, polynomial or nonlinear).

If $\beta_2 \neq 0$ statistically, possible reasons can be:

-   Statistical significance can be driven by large sample

-   Or the trends are so consistent, and just one period deviation can throw off the trends. Hence, statistical statistical significance.

Technically, we can still salvage the research by including time fixed effects, instead of just the before-and-after time fixed effect (actually, most researchers do this mechanically anyways nowadays). However, a side effect can be that the time fixed effects can also absorb some part your treatment effect as well, especially in cases where the treatment effects vary with time (i.e., stronger or weaker over time) [@wolfers2003].

#### Placebo Test

Procedure:

1.  Sample data only in the period before the treatment in time.
2.  Consider different fake cutoff in time, either
    1.  Try the whole sequence in time

    2.  Generate random treatment period, and use **randomization inference** to account for sampling distribution of the fake effect.
3.  Estimate the DiD model but with the post-time = 1 with the fake cutoff
4.  A significant DiD coefficient means that you violate the parallel trends! You have a big problem.

Alternatively,

-   When data have multiple control groups, drop the treated group, and assign another control group as a "fake" treated group. But even if it fails (i.e., you find a significant DiD effect) among the control groups, it can still be fine. However, this method is used under [Synthetic Control]

**Possible issues**

-   Estimate dependent on functional form:

    -   When the size of the response depends (nonlinearly) on the size of the intervention, we might want to look at the the difference in the group with high intensity vs. low.

-   Long-term effects

    -   Parallel trends are more likely to be observed over shorter period (window of observation)

-   Heterogeneous effects

    -   Different intensity (e.g., doses) for different groups.

-   Ashenfelter dip

    -   Participants are systemically different from nonparticipants before the treatment, leading to the question of permanent or transitory changes.

-   Response to event might not be immediate (can't be observed right away in the dependent variable)

    -   Using lagged dependent variable $Y_{it-1}$ might be more appropriate [@blundell1998]

-   Other factors that affect the difference in trends between the two groups (i.e., treatment and control) will bias your estimation.

-   Correlated observations within a group or time.

-   Incidental parameters problems [@lancaster2000]: it's always better to use individual and time fixed effect.

-   When examining the effects of variation in treatment timing, we have to be careful because negative weights (per group) can be negative if there is a heterogeneity in the treatment effects over time. Example: [@athey2022][@borusyak2021][@goodman-bacon2021]. In this case you should use new estimands proposed by [@callaway2021][@dechaisemartin2020], in the `did` package. If you expect lags and leads, see [@sun2021]

-   [@gibbons2018] caution when we suspect the treatment effect and treatment variance vary across groups

<br>

**Robustness Check**

-   Placebo DiD (if the DiD estimate $\neq 0$, parallel trend is violated, and original DiD is biased):

    -   Group: Use fake treatment groups: A population that was **not** affect by the treatment

    -   Time: Redo the DiD analysis for period before the treatment (expected treatment effect is 0) (e..g, for previous year or period).

-   Possible alternative control group: Expected results should be similar

-   Try different windows (further away from the treatment point, other factors can creep in and nullify your effect).

-   Treatment Reversal (what if we don't see the treatment event)

-   Higher-order polynomial time trend (to relax linearity assumption)

-   Test whether other dependent variables that should be affected by the event are indeed unaffected.

    -   Use the same control and treatment period (DiD $\neq0$, there is a problem)

<br>

### Examples

#### Example from [Princeton](https://www.princeton.edu/~otorres/DID101R.pdf)


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

<br>

#### Example by @card1993

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

-   $NJ$ = dummy where 1 = NJ, and 0 = PA

-   $POST$ = dummy where 1 = post, and 0 = pre

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

<br>

#### Example by @butcher2014

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

|              | Intercept                         | Treat | Post | Treat\*Post |
|--------------|-----------------------------------|-------|------|-------------|
| Treat Pre    | 1                                 | 1     | 0    | 0           |
| Treat Post   | 1                                 | 1     | 1    | 1           |
| Control Pre  | 1                                 | 0     | 0    | 0           |
| Control Post | 1                                 | 0     | 1    | 0           |
|              | Average for pre-control $\beta_0$ |       |      |             |

A more general specification of the dif-n-dif is that

$$
Y_{idt} = \alpha_0 + (POST_t \times Treat_d) \alpha_1 + \theta_d + \delta_t + X_{idt} + u_{idt}
$$

where

-   $(\theta_d + \delta_t)$ richer , more df than $Treat_d \beta_2 + Post_t \beta_1$ (because fixed effects subsume Post and treat)

-   $\alpha_1$ should be equivalent to $\beta_3$ (if your model assumptions are correct)

Under causal inference, $R^2$ is not so important.

## Multiple periods and variation in treatment timing

This is an extension of the DiD framework to settings where you have

-   more than 2 time periods

-   different treatment timing

Wheen treatment effects are heterogeneous across time or units,

<br>

## Staggered Dif-n-dif

-   When subjects are treated at different point in time, we have to use staggered DiD
-   For design where a treatment is applied and units are exposed to this treatment at all time afterward, see [@athey2022]

Basic design [@stevenson2006]

$$
Y_{it} = \alpha_i + \delta_t + \sum_k \gamma_k D_{ik} + \epsilon_{it}
$$

where

-   $\alpha_i$ are the country dummies

-   $\delta_t$ are the time dummies

-   $D_{ik} = 1$ for country $i$ with treatment in all $k \ge t$ periods

**Assumptions**

-   **Rollout Exogeneity**: if the treatment is randomly implemented over time (i.e., unrelated to variables that could also affect our dependent variables

-   **No counfounding events**

-   **Exclusion restrictions**

    -   ***No-anticipation assumption***: future treatment time do not affect current outcomes

    -   ***Invariance-to-history assumption***: the time a unit under treatment does not affect the outcome (i.e., the time exposed does not matter, just whether exposed or not). This presents causal effect of early or late adoption on the outcome.

-   Auxiliary assumptions:

    -   Contant treatment effects across units

    -   Constnat treatment effect over time

    -   Random sampling

    -   Effect Addiviity

### Example by @doleac2020

-   The purpose of banning a checking box for ex-criminal was banned because we thought that it gives more access to felons

-   Even if we ban the box, employers wouldn't just change their behaviors. But then the unintended consequence is that employers statistically discriminate based on race

3 types of ban the box

1.  Public employer only
2.  Private employer with government contract
3.  All employers

Main identification strategy

-   If any county in the Metropolitan Statistical Area (MSA) adopts ban the box, it means the whole MSA is treated. Or if the state adopts "ban the ban," every county is treated

Under [Simple Dif-n-dif]

$$
Y_{it} = \beta_0 + \beta_1 Post_t + \beta_2 treat_i + \beta_2 (Post_t \times Treat_i) + \epsilon_{it}
$$

But if there is no common post time, then we should use [Staggered Dif-n-dif]

$$
E_{imrt} = \alpha + \beta_1 BTB_{imt} W_{imt} + \beta_2 BTB_{mt} + \beta_3 BTB_{mt} H_{imt}+\\ \delta_m + D_{imt} \beta_5 + \lambda_{rt} + \delta_m\times f(t) \beta_7 + e_{imrt}
$$

where

-   $i$ = person; $m$ = MSA; $r$ = region (US regions e.g., midwest) ; $r$ = region; $t$ = year

-   $W$ = White; $B$ = Black; $H$ = Hispanic

-   $\beta_1 BTB_{imt} W_{imt} + \beta_2 BTB_{mt} + \beta_3 BTB_{mt} H_{imt}$ are the 3 dif-n-dif variables ($BTB$ = "ban the box")

-   $\delta_m$ = dummy for MSI

-   $D_{imt}$ = control for people

-   $\lambda_{rt}$ = region by time fixed effect

-   $\delta_m \times f(t)$ = linear time trend within MSA (but we should not need this if we have good pre-trend)

If we put $\lambda_r - \lambda_t$ (separately) we will more broad fixed effect, while $\lambda_{rt}$ will give us deeper and narrower fixed effect.

Before running this model, we have to drop all other races. And $\beta_1, \beta_2, \beta_3$ are not collinear because there are all interaction terms with $BTB_{mt}$

If we just want to estimate the model for black men, we will modify it to be

$$
E_{imrt} = \alpha + BTB_{mt} \beta_1 + \delta_m + D_{imt} \beta_5 + \lambda_{rt} + (\delta_m \times f(t)) \beta_7 + e_{imrt}
$$

$$
E_{imrt} = \alpha + BTB_{m (t - 3t)} \theta_1 + BTB_{m(t-2)} \theta_2 + BTB_{mt} \theta_4 \\
+ BTB_{m(t+1)}\theta_5 + BTB_{m(t+2)}\theta_6 + BTB_{m(t+3t)}\theta_7 \\
+ [\delta_m + D_{imt}\beta_5 + \lambda_r + (\delta_m \times (f(t))\beta_7 + e_{imrt}]
$$

We have to leave $BTB_{m(t-1)}\theta_3$ out for the category would not be perfect collinearity

So the year before BTB ($\theta_1, \theta_2, \theta_3$) should be similar to each other (i.e., same pre-trend). Remember, we only run for places with BTB.

If $\theta_2$ is statistically different from $\theta_3$ (baseline), then there could be a problem, but it could also make sense if we have pre-trend announcement.

<br>

Example by [Philipp Leppert](https://rpubs.com/phle/r_tutorial_difference_in_differences) replicating [Card and Krueger (1994)](https://davidcard.berkeley.edu/data_sets.html)

Example by [Anthony Schmidt](https://bookdown.org/aschmi11/causal_inf/difference-in-differences.html)

## Two-way Fixed-effects

A generalization of the dif-n-dif model is the two-way fixed-effects models where you have multiple groups and time effects.

This package is based on [@somaini2016]


```r
# dataset
library(bacondecomp)
df <- bacondecomp::castle

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
#> [1,] 0.003751709

# to save time, you can use your structure in the last output for a new set of variables
# output2 <- xtreg2way(y, x1, struc=output$struc)
```

Standard errors estimation options

| Set                  | Estimation                                                                                  |
|----------------------|---------------------------------------------------------------------------------------------|
| `se = "0"`           | Assume homoskedasticity and no within group correlation or serial correlation               |
| `se = "1"` (default) | robust to heteroskadasticity and serial correlation [@arellano1987computing]                |
| `se = "2"`           | robust to heteroskedasticity, but assumes no correlation within group or serial correlation |
| `se = "11"`          | Aerllano SE with df correction performed by Stata xtreg [@somaini2021twfem]                 |

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
