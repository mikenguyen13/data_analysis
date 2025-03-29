# Difference-in-Differences {#sec-difference-in-differences}

[Difference-in-Differences](#sec-difference-in-differences) (DID) is a widely used causal inference method for estimating the effect of **policy interventions** or **exogenous shocks** when randomized experiments are not feasible. The key idea behind DID is to compare changes in outcomes over time between **treated** and **control** groups, under the assumption that---absent treatment---both groups would have followed parallel trends.

[List of packages](https://github.com/lnsongxf/DiD-1)

DID analysis can go beyond simple treatment effects by exploring causal mechanisms using mediation and moderation analyses:

-   [Mediation Under DiD]: Examines how intermediate variables (e.g., consumer sentiment, brand perception) mediate the treatment effect [@habel2021variable].
-   [Moderation] Analysis: Studies how treatment effects vary across different groups (e.g., high vs. low brand loyalty) [@goldfarb2011online].

## Empirical Studies

### Applications of DID in Marketing

DID has been extensively applied in marketing and business research to measure the impact of policy changes, advertising campaigns, and competitive actions. Below are several notable examples:

-   **TV Advertising & Online Shopping** [@liaukonyte2015television]: Examines how TV ads influence consumer behavior in online shopping.
-   **Political Advertising & Voting Behavior** [@wang2018border]: Uses geographic discontinuities at state borders to analyze how ad sources and tone affect voter turnout.
-   **Music Streaming & Consumption** [@datta2018changing]: Investigates how adopting a music streaming service affects total music consumption.
-   **Data Breaches & Customer Spending** [@janakiraman2018effect]: Analyzes how customer spending changes after a firm announces a data breach.
-   **Price Monitoring & Policy Enforcement** [@israeli2018online]: Studies the effect of digital monitoring on minimum advertised price policy enforcement.
-   **Foreign Direct Investment & Firm Responses** [@ramani2019effects]: Examines how firms in India responded to FDI liberalization reforms in 1991.
-   **Paywalls & Readership** [@pattabhiramaiah2019paywalls]: Investigates how implementing paywalls affects online news consumption.
-   **Aggregators & Airline Business** [@akca2020value]: Evaluates how online aggregators impact airline ticket sales.
-   **Nutritional Labels & Competitive Response** [@lim2020competitive]: Analyzes whether nutrition labels affect the nutritional quality of competing brands.
-   **Payment Disclosure & Physician Behavior** [@guo2020let]: Studies how payment disclosure laws impact prescription behavior.
-   **Fake Reviews & Sales** [@he2022market]: Uses an Amazon policy change to measure the effect of fake reviews on sales and ratings.
-   **Data Protection Regulations & Website Usage** [@peukert2022regulatory]: Assesses the impact of GDPR regulations on website usage and online business models.

### Applications of DID in Economics

DID has also been extensively applied in **economics**, particularly in policy evaluation, labor economics, and macroeconomics:

-   **Natural Experiments in Development Economics** [@rosenzweig2000natural]
-   **Instrumental Variables & Natural Experiments** [@angrist2001instrumental]
-   **DID in Macroeconomic Policy Analysis** [@fuchs2016natural]

------------------------------------------------------------------------

## Visualization


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-1-1.png" width="90%" style="display: block; margin: auto;" />

``` r

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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-1-2.png" width="90%" style="display: block; margin: auto;" />

``` r

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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-1-3.png" width="90%" style="display: block; margin: auto;" />

## Simple Difference-in-Differences {#sec-simple-difference-in-differences}

Difference-in-Differences originated as a tool to analyze [natural experiments](#sec-natural-experiments), but its applications extend far beyond that. DID is built on the [Fixed Effects Estimator], making it a fundamental approach for policy evaluation and causal inference in observational studies.

DID leverages inter-temporal variation between groups:

-   **Cross-sectional comparison**: Helps avoid omitted variable bias due to common trends.
-   **Time-series comparison**: Helps mitigate omitted variable bias due to cross-sectional heterogeneity.

------------------------------------------------------------------------

### Basic Setup of DID

Consider a simple setting with:

-   **Treatment Group** ($D_i = 1$)
-   **Control Group** ($D_i = 0$)
-   **Pre-Treatment Period** ($T = 0$)
-   **Post-Treatment Period** ($T = 1$)

|                     | **After Treatment (**$T = 1$**)** | **Before Treatment (**$T = 0$**)** |
|---------------------|-----------------------------------|------------------------------------|
| Treated ($D_i = 1$) | $E[Y_{1i}(1)|D_i = 1]$            | $E[Y_{0i}(0)|D_i = 1]$             |
| Control ($D_i = 0$) | $E[Y_{0i}(1)|D_i = 0]$            | $E[Y_{0i}(0)|D_i = 0]$             |

The **fundamental challenge**: We cannot observe $E[Y_{0i}(1)|D_i = 1]$---i.e., the **counterfactual outcome** for the treated group had they not received treatment.

------------------------------------------------------------------------

DID estimates the [Average Treatment Effect on the Treated] using the following formula:

$$
\begin{aligned}
E[Y_1(1) - Y_0(1) | D = 1] &= \{E[Y(1)|D = 1] - E[Y(1)|D = 0] \} \\
&- \{E[Y(0)|D = 1] - E[Y(0)|D = 0] \}
\end{aligned}
$$

This formulation differences out time-invariant unobserved factors, assuming the parallel trends assumption holds.

-   For the treated group, we isolate the difference between being treated and not being treated.
-   If the control group would have experienced a different trajectory, the DID estimate may be biased.
-   Since we cannot observe treatment variation in the control group, we cannot infer the treatment effect for this group.

------------------------------------------------------------------------

### Extensions of DID

#### DID with More Than Two Groups or Time Periods

DID can be extended to **multiple treatments, multiple controls**, and more than two periods:

$$
Y_{igt} = \alpha_g + \gamma_t + \beta I_{gt} + \delta X_{igt} + \epsilon_{igt}
$$

where:

-   $\alpha_g$ = Group-Specific Fixed Effects (e.g., firm, region).

-   $\gamma_t$ = Time-Specific Fixed Effects (e.g., year, quarter).

-   $\beta$ = DID Effect.

-   $I_{gt}$ = Interaction Terms (Treatment × Post-Treatment).

-   $\delta X_{igt}$ = Additional Covariates.

This is known as the Two-Way Fixed Effects DID model. However, TWFE performs poorly under staggered treatment adoption, where different groups receive treatment at different times.

------------------------------------------------------------------------

#### Examining Long-Term Effects (Dynamic DID)

To examine the dynamic treatment effects (that are not under rollout/staggered design), we can create a centered time variable.

+------------------------+---------------------------------------------------------+
| Centered Time Variable | Interpretation                                          |
+========================+=========================================================+
| $t = -2$               | Two periods before treatment                            |
+------------------------+---------------------------------------------------------+
| $t = -1$               | One period before treatment                             |
+------------------------+---------------------------------------------------------+
| $t = 0$                | Last pre-treatment period right before treatment period |
|                        |                                                         |
|                        | (Baseline/Reference Group)                              |
+------------------------+---------------------------------------------------------+
| $t = 1$                | Treatment period                                        |
+------------------------+---------------------------------------------------------+
| $t = 2$                | One period after treatment                              |
+------------------------+---------------------------------------------------------+

**Dynamic Treatment Model Specification**

By interacting this factor variable, we can examine the dynamic effect of treatment (i.e., whether it's fading or intensifying):

$$
\begin{aligned}
Y &= \alpha_0 + \alpha_1 Group + \alpha_2 Time  \\
&+ \beta_{-T_1} Treatment + \beta_{-(T_1 -1)} Treatment + \dots + \beta_{-1} Treatment \\
&+ \beta_1 + \dots + \beta_{T_2} Treatment
\end{aligned}
$$

where:

-   $\beta_0$ (Baseline Period) is the reference group (i.e., drop from the model).

-   $T_1$ = Pre-Treatment Period.

-   $T_2$ = Post-Treatment Period.

-   Treatment coefficients ($\beta_t$) measure the effect over time.

Key Observations:

-   Pre-treatment coefficients should be close to zero ($\beta_{-T_1}, \dots, \beta_{-1} \approx 0$), ensuring no pre-trend bias.

-   Post-treatment coefficients should be significantly different from zero ($\beta_1, \dots, \beta_{T_2} \neq 0$), measuring the treatment effect over time.

-   Higher standard errors (SEs) with more interactions: Including too many lags can reduce precision.

------------------------------------------------------------------------

#### DID on Relationships, Not Just Levels

DID can also be applied to relationships between variables rather than just outcome levels.

For example, DID can be used to estimate treatment effects on regression coefficients by comparing relationships before and after a policy change.

### Goals of DID

1.  **Pre-Treatment Coefficients Should Be Insignificant**
    -   Ensure that $\beta_{-T_1}, \dots, \beta_{-1} = 0$ (similar to a [Placebo Test](Ensure%20no%20pre-treatment%20effects.)).
2.  **Post-Treatment Coefficients Should Be Significant**
    -   Verify that $\beta_1, \dots, \beta_{T_2} \neq 0$.
    -   Examine whether the trend in post-treatment coefficients is increasing or decreasing over time.

------------------------------------------------------------------------


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-2-1.png" width="90%" style="display: block; margin: auto;" />

``` r
coefplot(cali)
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-2-2.png" width="90%" style="display: block; margin: auto;" />

## Empirical Studies

### Example: The Unintended Consequences of "Ban the Box" Policies

@doleac2020unintended examine the unintended effects of "Ban the Box" (BTB) policies, which prevent employers from asking about criminal records during the hiring process. The intended goal of BTB was to increase job access for individuals with criminal records. However, the study found that employers, unable to observe criminal history, resorted to statistical discrimination based on race, leading to unintended negative consequences.

Three Types of "Ban the Box" Policies:

1.  Public employers only
2.  Private employers with government contracts
3.  All employers

Identification Strategy

-   If any county within a Metropolitan Statistical Area (MSA) adopts BTB, the entire MSA is considered treated.
-   If a state passes a law banning BTB ("ban the ban"), then all counties in that state are treated.

------------------------------------------------------------------------

The [basic DiD model](#sec-simple-difference-in-differences) is:

$$
Y_{it} = \beta_0 + \beta_1 \text{Post}_t + \beta_2 \text{Treat}_i + \beta_3 (\text{Post}_t \times \text{Treat}_i) + \epsilon_{it}
$$

where:

-   $Y_{it}$ = employment outcome for individual $i$ at time $t$
-   $\text{Post}_t$ = indicator for post-treatment period
-   $\text{Treat}_i$ = indicator for treated MSAs
-   $\beta_3$ = the DiD coefficient, capturing the effect of BTB on employment
-   $\epsilon_{it}$ = error term

**Limitations**: If different locations adopt BTB at different times, this model is not valid due to staggered treatment timing.

------------------------------------------------------------------------

For settings where different MSAs adopt BTB at different times, we use a **staggered DiD** approach:

$$
\begin{aligned} 
E_{imrt} &= \alpha + \beta_1 BTB_{imt} W_{imt} + \beta_2 BTB_{mt} + \beta_3 BTB_{mt} H_{imt} \\
&+ \delta_m + D_{imt} \beta_5 + \lambda_{rt} + \delta_m \times f(t) \beta_7 + e_{imrt} 
\end{aligned}
$$

where:

-   $i$ = individual, $m$ = MSA, $r$ = region (e.g., Midwest, South), $t$ = year
-   $W$ = White; $B$ = Black; $H$ = Hispanic
-   $BTB_{imt}$ = Ban the Box policy indicator
-   $\delta_m$ = MSA fixed effect
-   $D_{imt}$ = individual-level controls
-   $\lambda_{rt}$ = region-by-time fixed effect
-   $\delta_m \times f(t)$ = linear time trend within MSA

**Fixed Effects Considerations**:

-   Including $\lambda_r$ and $\lambda_t$ separately gives broader fixed effects.

-   Using $\lambda_{rt}$ provides more granular controls for regional time trends.

------------------------------------------------------------------------

To estimate the effects for Black men specifically, the model simplifies to:

$$
E_{imrt} = \alpha + BTB_{mt} \beta_1 + \delta_m + D_{imt} \beta_5 + \lambda_{rt} + (\delta_m \times f(t)) \beta_7 + e_{imrt}
$$

------------------------------------------------------------------------

To check for pre-trends and dynamic effects, we estimate:

$$
\begin{aligned} 
E_{imrt} &= \alpha + BTB_{m (t - 3)} \theta_1 + BTB_{m (t - 2)} \theta_2 + BTB_{m (t - 1)} \theta_3 \\
&+ BTB_{mt} \theta_4 + BTB_{m (t + 1)} \theta_5 + BTB_{m (t + 2)} \theta_6 + BTB_{m (t + 3)} \theta_7 \\
&+ \delta_m + D_{imt} \beta_5 + \lambda_{r} + (\delta_m \times f(t)) \beta_7 + e_{imrt}
\end{aligned}
$$

Key points:

-   Leave out $BTB_{m (t - 1)} \theta_3$ as the reference category (to avoid perfect collinearity).
-   If $\theta_2$ is significantly different from $\theta_3$, it suggests pre-trend issues, which could indicate anticipatory effects before BTB implementation.

------------------------------------------------------------------------

### Example: Minimum Wage and Employment

@card1993minimum famously studied the effect of an increase in the minimum wage on employment, challenging the traditional economic view that higher wages reduce employment.

-   [Philipp Leppert](https://rpubs.com/phle/r_tutorial_difference_in_differences) provides an R-based replication.
-   Original datasets are available at [David Card's website](https://davidcard.berkeley.edu/data_sets.html).

Setting

-   **Treatment group**: New Jersey (NJ), which increased its minimum wage.
-   **Control group**: Pennsylvania (PA), which did not change its minimum wage.
-   **Outcome variable**: Employment levels in fast-food restaurants.

The study used a Difference-in-Differences approach to estimate the impact:

|           | State | After (Post) | Before (Pre) | Difference        |
|-----------|-------|--------------|--------------|-------------------|
| Treatment | NJ    | A            | B            | A - B             |
| Control   | PA    | C            | D            | C - D             |
|           |       | A - C        | B - D        | (A - B) - (C - D) |

where:

-   $A - B$ captures the treatment effect plus general time trends.
-   $C - D$ captures only the general time trends.
-   $(A - B) - (C - D)$ isolates the causal effect of the minimum wage increase.

For the DiD estimator to be valid, the following conditions must hold:

1.  **Parallel Trends Assumption**
    -   The employment trends in NJ and PA would have been the same in the absence of the policy change.
    -   Pre-treatment employment trends should be similar between the two states.
2.  **No "Switchers"**
    -   The policy must not induce restaurants to switch locations between NJ and PA (e.g., a restaurant relocating across the border).
3.  **PA as a Valid Counterfactual**
    -   PA represents what NJ would have looked like had it not changed the minimum wage.
    -   The study focuses on bordering counties to increase comparability.

------------------------------------------------------------------------

The main regression specification is:

$$
Y_{jt} = \beta_0 + NJ_j \beta_1 + POST_t \beta_2 + (NJ_j \times POST_t)\beta_3+ X_{jt}\beta_4 + \epsilon_{jt}
$$

where:

-   $Y_{jt}$ = Employment in restaurant $j$ at time $t$
-   $NJ_j$ = 1 if restaurant is in NJ, 0 if in PA
-   $POST_t$ = 1 if post-policy period, 0 if pre-policy
-   $(NJ_j \times POST_t)$ = **DiD interaction term**, capturing the causal effect of NJ's minimum wage increase
-   $X_{jt}$ = Additional controls (optional)
-   $\epsilon_{jt}$ = Error term

Notes on Model Specification

-   $\beta_3$ (DiD coefficient) is the key parameter of interest, representing the causal impact of the policy.

-   $\beta_4$ (controls $X_{jt}$) is not necessary for unbiasedness but improves efficiency.

-   If we difference out the pre-period ($\Delta Y_{jt} = Y_{j,Post} - Y_{j,Pre}$), we can simplify the model:

    $$
    \Delta Y_{jt} = \alpha + NJ_j \beta_1 + \epsilon_{jt}
    $$

    Here, we no longer need $\beta_2$ for the post-treatment period.

------------------------------------------------------------------------

An alternative specification uses high-wage NJ restaurants as a control group, arguing that they were not affected by the minimum wage increase. However:

-   This approach eliminates cross-state differences, but
-   It may be harder to interpret causality, as the control group is not entirely untreated.

------------------------------------------------------------------------

A common misconception in DiD is that treatment and control groups must have the same baseline levels of the dependent variable (e.g., employment levels). However:

-   DiD only requires parallel trends, meaning the slopes of employment changes should be the same pre-treatment.
-   If pre-treatment trends diverge, this threatens validity.
-   If post-treatment trends converge, it may suggest policy effects rather than pre-trend violations.

Is Parallel Trends a Necessary or Sufficient Condition?

-   Not sufficient: Even if pre-trends are parallel, other confounders could affect results.
-   Not necessary: Parallel trends may emerge only after treatment, depending on behavioral responses.

Thus, we cannot prove DiD is valid---we can only present evidence that supports the assumptions.

------------------------------------------------------------------------

### Example: The Effects of Grade Policies on Major Choice

@butcher2014effects investigate how grading policies influence students' major choices. The central theory is that grading standards vary by discipline, which affects students' decisions.

Why do the highest-achieving students often major in hard sciences?

1.  **Grading Practices Differ Across Majors**
    -   In STEM fields, grading is often stricter, meaning professors are less likely to give students the benefit of the doubt.
    -   In contrast, softer disciplines (e.g., humanities) may have more lenient grading, making students' experiences more pleasant.
2.  **Labor Market Incentives**
    -   Degrees with lower market value (e.g., humanities) might compensate by offering a more pleasant academic experience.
    -   STEM degrees tend to be more rigorous but provide higher job market returns.

------------------------------------------------------------------------

To examine how grades influence major selection, the study first estimates an OLS model:

$$
E_{ij} = \beta_0 + X_i \beta_1 + G_j \beta_2 + \epsilon_{ij}
$$

where:

-   $E_{ij}$ = Indicator for whether student $i$ chooses major $j$.
-   $X_i$ = Student-level attributes (e.g., SAT scores, demographics).
-   $G_j$ = Average grade in major $j$.
-   $\beta_2$ = Key coefficient, capturing how grading standards influence major choice.

Potential Biases in $\hat{\beta}_2$:

-   **Negative Bias**:
    -   Departments with lower enrollment rates may offer higher grades to attract students.
    -   This endogenous response leads to a downward bias in the OLS estimate.
-   **Positive Bias**:
    -   STEM majors attract the best students, so their grades would naturally be higher if ability were controlled.
    -   If ability is not fully accounted for, $\hat{\beta}_2$ may be upward biased.

------------------------------------------------------------------------

To address potential endogeneity in OLS, the study uses a difference-in-differences approach:

$$
Y_{idt} = \beta_0 + POST_t \beta_1 + Treat_d \beta_2 + (POST_t \times Treat_d)\beta_3 + X_{idt} + \epsilon_{idt}
$$

where:

-   $Y_{idt}$ = Average grade in department $d$ at time $t$ for student $i$.
-   $POST_t$ = 1 if post-policy period, 0 otherwise.
-   $Treat_d$ = 1 if department is treated (i.e., grade policy change), 0 otherwise.
-   $(POST_t \times Treat_d)$ = **DiD interaction term**, capturing the causal effect of grade policy changes on major choice.
-   $X_{idt}$ = Additional student controls.

------------------------------------------------------------------------

| Group             | Intercept ($\beta_0$) | Treatment ($\beta_2$) | Post ($\beta_1$) | Interaction ($\beta_3$) |
|-------------------|-----------------------|-----------------------|------------------|-------------------------|
| **Treated, Pre**  | 1                     | 1                     | 0                | 0                       |
| **Treated, Post** | 1                     | 1                     | 1                | 1                       |
| **Control, Pre**  | 1                     | 0                     | 0                | 0                       |
| **Control, Post** | 1                     | 0                     | 1                | 0                       |

: Difference-in-Differences Table

-   The average pre-period outcome for the control group is given by $\beta_0$.
-   The key coefficient of interest is $\beta_3$, which captures the difference in the post-treatment effect between treated and control groups.

------------------------------------------------------------------------

A more flexible specification includes fixed effects:

$$
Y_{idt} = \alpha_0 + (POST_t \times Treat_d) \alpha_1 + \theta_d + \delta_t + X_{idt} + u_{idt}
$$

where:

-   $\theta_d$ = Department fixed effects (absorbing $Treat_d$).
-   $\delta_t$ = Time fixed effects (absorbing $POST_t$).
-   $\alpha_1$ = Effect of policy change (equivalent to $\beta_3$ in the simpler model).

Why Use Fixed Effects?

-   **More flexible specification**:
    -   Instead of assuming a uniform treatment effect across groups, this model allows for department-specific differences ($\theta_d$) and time-specific shocks ($\delta_t$).
-   **Higher degrees of freedom**:
    -   Fixed effects absorb variation that would otherwise be attributed to $POST_t$ and $Treat_d$, making the estimation more efficient.

Interpretation of Results

-   If $\alpha_1 > 0$, then the policy **increased** grades in treated departments.
-   If $\alpha_1 < 0$, then the policy **decreased** grades in treated departments.

------------------------------------------------------------------------

## One Difference

The regression formula is as follows @liaukonyte2023frontiers:

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

------------------------------------------------------------------------

## Two-Way Fixed Effects {#sec-two-way-fixed-effects}

A generalization of the [difference-in-differences](#sec-difference-in-differences) model is the t**wo-way fixed effects (TWFE) model**, which accounts for **multiple groups** and **multiple time periods** by including both unit and time fixed effects. However, TWFE is not a design-based, non-parametric causal estimator [@imai2021use].

When applying TWFE to datasets with multiple treatment groups and periods, the estimated causal coefficient is a **weighted average** of all possible two-group, two-period DiD comparisons. Importantly, some weights can be **negative**, leading to potential biases. The weighting scheme is driven by:

-   **Group sizes**
-   **Variation in treatment timing**
-   **Placement in the middle of the panel (units closer to the middle tend to receive the highest weight)**

The canonical TWFE model is valid only when:

-   **Treatment effects are homogeneous** across both units and time periods, meaning:
    -   No dynamic treatment effects (i.e., treatment effects do not evolve over time).
    -   The treatment effect is constant across units [@goodman2021difference; @de2020two; @sun2021estimating; @borusyak2021revisiting].
    -   The model assumes linear additive effects [@imai2021use].
-   In reality, we almost always see treatment heterogeneity. Hence, to use the TWFE, we actually have to argue why the effects are homogeneous to justify TWFE use:
    -   **Assess treatment heterogeneity**: If heterogeneity exists, TWFE may produce biased estimates. Researchers should:
        -   Plot treatment timing across units.
        -   Decompose the treatment effect using the [Goodman-Bacon decomposition](#sec-goodman-bacon-decomposition) to identify negative weights.
        -   Check the proportion of never-treated observations: When 80% or more of the sample is never treated, TWFE bias is negligible.
        -   Beware of bias worsening with long-run effects.
    -   **Dropping relative time periods**:
        -   If all units eventually receive treatment, two relative time periods must be dropped to avoid multicollinearity.
        -   Some software packages drop periods randomly; if a post-treatment period is dropped, bias may result.
        -   The standard approach is to drop periods -1 and -2.
    -   **Sources of treatment heterogeneity**:
        -   Delayed treatment effects: The impact of treatment may take time to manifest.
        -   Evolving effects: Treatment effects can increase or change over time (e.g., phase-in effects).
-   When there are only two time periods, TWFE simplifies to the **traditional DiD model**.

TWFE compares different types of treatment/control groups:

-   **Valid comparisons**:
    -   Newly treated units vs. control units
    -   Newly treated units vs. not-yet treated units
-   **Problematic comparisons**:
    -   Newly treated units vs. already treated units (since already treated units do not represent the correct counterfactual).
    -   **Strict exogeneity violations**:
        -   Presence of time-varying confounders
        -   Feedback from past outcomes to treatment [@imai2019should]
    -   **Functional form restrictions**:
        -   Assumes treatment effect homogeneity.
        -   No carryover effects or anticipation effects [@imai2019should].

------------------------------------------------------------------------

Notation follows @arkhangelsky2024design. The TWFE model is:

$$
Y_{it} = \alpha_i + \lambda_t + \tau W_{it} + \beta X_{it} + \epsilon_{it}
$$

where:

-   $Y_{it}$ = Outcome for unit $i$ at time $t$
-   $\alpha_i$ = Unit fixed effect
-   $\lambda_t$ = Time fixed effect
-   $\tau$ = Causal effect of treatment
-   $W_{it}$ = Treatment indicator ($1$ if treated, $0$ otherwise)
-   $X_{it}$ = Covariates
-   $\epsilon_{it}$ = Error term

When there are only two time periods ($T=2$), TWFE reduces to the standard DiD model.

Under the following assumptions, $\hat{\tau}_{OLS}$ is unbiased:

1.  Homogeneous treatment effect across units and periods.
2.  Parallel trends assumption holds.
3.  Linear additive effects are valid [@imai2021use].

------------------------------------------------------------------------

### Remedies for TWFE's Shortcomings

1.  **Diagnosing and Understanding Bias in TWFE**

A key issue with TWFE is that it averages treatment effects across groups and time, sometimes assigning negative weights to comparisons. Several approaches help diagnose and correct for these biases:

-   [Goodman-Bacon Decomposition](#sec-goodman-bacon-decomposition) [@goodman2021difference]:
    -   Decomposes the TWFE DiD estimate into two-group, two-period comparisons.
    -   Identifies which comparisons receive negative weights, which can lead to biased estimates.
    -   Helps determine the influence of specific groups on the overall estimate.

------------------------------------------------------------------------

2.  **Alternative Estimators to Address Heterogeneous Treatment Effects**

When treatment effects vary over time or across groups, TWFE produces biased estimates. The following methods provide alternatives:

**(a) Group-Time Average Treatment Effects**

[@callaway2021difference] propose a **two-step estimation** method:

-   **Step 1:** Estimate group-time treatment effects, where each group is defined by the period when it first received treatment.
-   **Step 2:** Use a bootstrap procedure to account for autocorrelation and clustering.

Key features:

-   Allows for heterogeneous treatment effects across time and groups.

-   Compares post-treatment outcomes of groups treated in a given period against a never-treated comparison group (using matching).

-   Treatment status must be monotonic (i.e., once treated, always treated).

**R Package:** `did`

------------------------------------------------------------------------

**(b) Event-Study Design with Cohort-Specific Estimates**

@sun2021estimating extend @callaway2021difference to an event-study setting:

-   Includes lags and leads in the model to capture dynamic treatment effects.
-   Uses cohort-specific estimates, similar to the group-time average treatment effects in @callaway2021difference.
-   Proposes the interaction-weighted estimator, which adjusts for treatment timing differences.

**R Package:** `fixest`

------------------------------------------------------------------------

**(c) Weighted TWFE with Time-Varying Treatment Status**

Unlike @callaway2021difference, @imai2021use allow units to switch in and out of treatment. Their approach:

-   Uses matching methods to estimate weighted TWFE.
-   Reduces biases from treatment effect heterogeneity.

**R Packages:** `wfe`, `PanelMatch`

------------------------------------------------------------------------

**(d) Two-Stage Difference-in-Differences (DiD2S)**

[@gardner2022two] propose a **two-stage DiD estimator (DiD2S)**, which:

-   Provides an alternative approach to account for heterogeneous treatment effects.
-   Works particularly well when there are never-treated units in the sample.

**R Package:** `did2s`

------------------------------------------------------------------------

**(e) Exposure-Adjusted DiD for Never-Treated Units**

-   If a study contains **never-treated units**, [@de2020two] suggest using an **exposure-adjusted DiD estimator** to recover the **average treatment effect**.
-   However, **exposure-adjusted DiD still fails** to detect treatment effect heterogeneity when treatment effects vary based on exposure rate [@sun2022linear].

------------------------------------------------------------------------

**(f) [Reshaped Inverse Probability Weighting - TWFE](#sec-reshaped-inverse-probability-weighting-twfe-estimator)**

[@arkhangelsky2024design]: Further refinements in **design-based approaches** to DiD estimation.

------------------------------------------------------------------------

### Reshaped Inverse Probability Weighting - TWFE Estimator {#sec-reshaped-inverse-probability-weighting-twfe-estimator}

The **Reshaped Inverse Probability Weighting (RIPW) estimator** extends the classic **TWFE** regression framework to account for arbitrary, time- and unit-varying treatment assignment mechanisms. This approach leverages an explicit model for treatment assignment to achieve **design robustness**, maintaining consistency even when traditional fixed-effects outcome models are misspecified.

The RIPW-TWFE framework is particularly relevant in panel data settings with **general treatment patterns**

-   **staggered adoption**

-   **transient treatments**.

------------------------------------------------------------------------

Setting and Notation

-   Panel data with $n$ units observed over $T$ time periods.

-   **Potential outcomes**: For each unit $i \in \{1, \dots, n\}$ and time $t \in \{1, \dots, T\}$:

    $$
    Y_{it}(1), \quad Y_{it}(0)
    $$

-   **Observed outcomes**:

    $$
    Y_{it} = W_{it} Y_{it}(1) + (1 - W_{it}) Y_{it}(0)
    $$

-   **Treatment assignment path** for unit $i$:

    $$
    \mathbf{W}_i = (W_{i1}, \dots, W_{iT}) \in \{0,1\}^T
    $$

-   **Generalized Propensity Score (GPS)**: For unit $i$, the probability distribution over treatment paths:

    $$
    \mathbf{W}_i \sim \pi_i(\cdot)
    $$

    where $\pi_i(w)$ is known or estimated.

------------------------------------------------------------------------

**Assumptions**

1.  **Binary Treatment**: $W_{it} \in \{0,1\}$ for all $i$ and $t$.

2.  **No Dynamic Effects**: Current outcomes depend only on current treatment, not past treatments.

3.  **Overlap Condition** (Assumption 2.2 from @arkhangelsky2024design):

    There exists a subset $S^* \subseteq \{0,1\}^T$, with $|S^*| \ge 2$ and $S^* \not\subseteq \{0_T, 1_T\}$, such that:

    $$
    \pi_i(w) > c > 0, \quad \forall w \in S^*, \forall i \in \{1, \dots, n\}
    $$

4.  **Maximal Correlation Decay** (Assumption 2.1): Dependence between units decays at rate $n^{-q}$ for some $q \in (0,1]$.

5.  **Bounded Second Moments** (Assumption 2.3): $\sup_{i,t,w} \mathbb{E}[Y_{it}^2(w)] < M < \infty$.

------------------------------------------------------------------------

Key Quantities of Interest

-   **Unit-Time Specific Treatment Effect**:

    $$
    \tau_{it} = Y_{it}(1) - Y_{it}(0)
    $$

-   **Time-Specific** [Average Treatment Effect]:

    $$
    \tau_t = \frac{1}{n} \sum_{i=1}^n \tau_{it}
    $$

-   **Doubly Averaged Treatment Effect (DATE)**:

    $$
    \tau(\xi) = \sum_{t=1}^T \xi_t \tau_t = \sum_{t=1}^T \xi_t \left( \frac{1}{n} \sum_{i=1}^n \tau_{it} \right)
    $$

    where $\xi = (\xi_1, \dots, \xi_T)$ is a vector of non-negative weights such that $\sum_{t=1}^T \xi_t = 1$.

-   **Special Case**: Equally weighted DATE:

    $$
    \tau_{\text{eq}} = \frac{1}{nT} \sum_{t=1}^T \sum_{i=1}^n \tau_{it}
    $$

------------------------------------------------------------------------

Inverse Probability Weighting (IPW) methods are widely used to correct for **selection bias** in treatment assignment by reweighting observations according to their probability of receiving a given treatment. In panel data settings with TWFE regression, the IPW approach can be incorporated to address non-random treatment assignments over time and across units.

We begin with the **classic TWFE regression objective**, then show how IPW modifies it, and finally generalize to the **Reshaped IPW (RIPW)** estimator.

------------------------------------------------------------------------

The **unweighted** TWFE estimator minimizes the following objective function:

$$
\min_{\tau, \mu, \{\alpha_i\}, \{\lambda_t\}} \sum_{i=1}^{n} \sum_{t=1}^{T} \left( Y_{it} - \mu - \alpha_i - \lambda_t - W_{it} \tau \right)^2
$$

where

-   $n$: Total number of units (e.g., individuals, firms, regions).
-   $T$: Total number of time periods.
-   $Y_{it}$: Observed outcome for unit $i$ at time $t$.
-   $W_{it}$: Binary treatment indicator for unit $i$ at time $t$.
    -   $W_{it} = 1$ if unit $i$ is treated at time $t$; $0$ otherwise.
-   $\tau$: Parameter of interest, representing the [Average Treatment Effect] under the TWFE model.
-   $\mu$: Common intercept, capturing the overall average outcome level across all units and times.
-   $\alpha_i$: Unit-specific fixed effects, controlling for time-invariant heterogeneity across units.
-   $\lambda_t$: Time-specific fixed effects, controlling for shocks or common trends that affect all units in time period $t$.

This standard TWFE regression assumes [parallel trends](#prior-parallel-trends-test) across units in the absence of treatment and **ignores** the treatment assignment mechanism.

------------------------------------------------------------------------

The **IPW-TWFE estimator** modifies the classic TWFE regression by **reweighting** the contribution of each observation according to the **inverse probability of the entire treatment path** for unit $i$.

The weighted objective function is:

$$
\min_{\tau, \mu, \{\alpha_i\}, \{\lambda_t\}} \sum_{i=1}^{n} \sum_{t=1}^{T} \left( Y_{it} - \mu - \alpha_i - \lambda_t - W_{it} \tau \right)^2 \cdot \frac{1}{\pi_i(\mathbf{W}_i)}
$$

where

-   $\pi_i(\mathbf{W}_i)$: The **generalized propensity score (GPS)** for unit $i$.
    -   This is the joint probability that unit $i$ follows the entire treatment assignment path $\mathbf{W}_i = (W_{i1}, W_{i2}, \dots, W_{iT})$.
    -   It represents the assignment mechanism, which may be known (in experimental designs) or estimated (in observational studies).

By weighting the squared residual for each unit-time observation by $\frac{1}{\pi_i(\mathbf{W}_i)}$, the IPW-TWFE estimator **adjusts for non-random treatment assignment**, similar to the role of IPW in cross-sectional data.

------------------------------------------------------------------------

The **Reshaped IPW (RIPW)** estimator further generalizes the IPW approach by introducing a **user-specified reshaped design distribution**, denoted by $\Pi$, over the space of treatment assignment paths.

The RIPW-TWFE estimator minimizes the following weighted objective:

$$
\hat{\tau}_{RIPW}(\Pi) = \arg \min_{\tau, \mu, \{\alpha_i\}, \{\lambda_t\}} \sum_{i=1}^{n} \sum_{t=1}^{T} \left( Y_{it} - \mu - \alpha_i - \lambda_t - W_{it} \tau \right)^2 \cdot \frac{\Pi(\mathbf{W}_i)}{\pi_i(\mathbf{W}_i)}
$$

where

-   $\Pi(\mathbf{W}_i)$: A **user-specified reshaped distribution** over the treatment assignment paths $\mathbf{W}_i$.
    -   It describes an alternative "design" the researcher wants to emulate, possibly reflecting hypothetical or target assignment mechanisms.
-   The weight $\frac{\Pi(\mathbf{W}_i)}{\pi_i(\mathbf{W}_i)}$ can be interpreted as a likelihood ratio:
    -   If $\pi_i(\cdot)$ is the true assignment distribution, reweighting by $\Pi(\cdot)$ effectively shifts the sampling design from $\pi_i$ to $\Pi$.
-   The ratio $\frac{\Pi(\mathbf{W}_i)}{\pi_i(\mathbf{W}_i)}$ adjusts for differences between the observed assignment mechanism and the target design.

------------------------------------------------------------------------

Support of $\mathbf{W}_i$

The support of the treatment assignment paths is defined as:

$$
\mathbb{S} = \bigcup_{i=1}^{n} \text{Supp}(\mathbf{W}_i)
$$

-   $\text{Supp}(\mathbf{W}_i)$: The support of the random variable $\mathbf{W}_i$, i.e., the set of all treatment paths with positive probability under $\pi_i(\cdot)$.
-   $\mathbb{S}$ represents the combined support across all units $i = 1, \dots, n$.
-   $\Pi(\cdot)$ should have support contained within $\mathbb{S}$, to ensure valid reweighting.

------------------------------------------------------------------------

**Special Cases of the RIPW Estimator**

The choice of $\Pi(\cdot)$ determines the behavior and interpretation of the RIPW estimator. Several special cases are noteworthy:

-   **Uniform Reshaped Design**:

    $$
    \Pi(\cdot) \sim \text{Uniform}(\mathbb{S})
    $$

    -   Here, $\Pi$ places equal probability mass on each possible treatment path in $\mathbb{S}$.

    -   The weight becomes:

        $$
        \frac{\Pi(\mathbf{W}_i)}{\pi_i(\mathbf{W}_i)} = \frac{1 / |\mathbb{S}|}{\pi_i(\mathbf{W}_i)}
        $$

    -   This reduces RIPW to the standard **IPW-TWFE estimator**, in which the target is a uniform treatment assignment design.

-   **Reshaped Design Equals True Assignment**:

    $$
    \Pi(\cdot) = \pi_i(\cdot)
    $$

    -   The weight simplifies to:

        $$
        \frac{\Pi(\mathbf{W}_i)}{\pi_i(\mathbf{W}_i)} = 1
        $$

    -   The RIPW estimator reduces to the **unweighted TWFE regression**, consistent with an experiment where the assignment mechanism $\pi_i$ is known and correctly specified.

------------------------------------------------------------------------

To ensure that $\hat{\tau}_{RIPW}(\Pi)$ consistently estimates the DATE $\tau(\xi)$, we solve the **DATE Equation**:

$$
\mathbb{E}_{\mathbf{W} \sim \Pi} \left[ \left( \text{diag}(\mathbf{W}) - \xi \mathbf{W}^\top \right) J \left( \mathbf{W} - \mathbb{E}_{\Pi}[\mathbf{W}] \right) \right] = 0
$$

-   $J = I_T - \frac{1}{T} \mathbf{1}_T \mathbf{1}_T^\top$ is a projection matrix removing the mean.
-   Solving this equation ensures consistency of the RIPW estimator for $\tau(\xi)$.

------------------------------------------------------------------------

Choosing the Reshaped Distribution $\Pi$

-   If the support $\mathbb{S}$ and $\pi_i(\cdot)$ are known, $\Pi$ can be specified directly.
-   Closed-form solutions for $\Pi$ are available in settings such as staggered adoption designs.
-   When closed-form solutions are unavailable, optimization algorithms (e.g., BFGS) can be employed to solve the DATE equation numerically.

------------------------------------------------------------------------

**Properties**

-   The RIPW estimator provides design-robustness:
    -   It can correct for misspecified outcome models by properly reweighting according to the assignment mechanism.
    -   It accommodates complex treatment assignment processes, such as staggered adoption and non-random assignment.
-   The flexibility to choose $\Pi(\cdot)$ allows researchers to target estimands that represent specific policy interventions or hypothetical designs.

The RIPW estimator has a **double robustness** property:

-   $\hat{\tau}_{RIPW}(\Pi)$ is consistent if **either**:

    -   The assignment model $\pi_i(\cdot)$ is correctly specified **or**

    -   The outcome regression (TWFE) model is correctly specified.

This feature is particularly valuable in [quasi-experimental designs](#sec-quasi-experimental) where the parallel trends assumption may not hold globally.

-   **Design-Robustness**: RIPW corrects for negative weighting issues identified in the TWFE literature (e.g., @goodman2021difference; @de2023two).
-   Unlike conventional TWFE regressions, which can yield biased estimands under heterogeneity, RIPW explicitly targets user-specified weighted averages (DATE).
-   In randomized experiments, RIPW ensures the **effective estimand** is interpretable as a population-level average, determined by the design $\Pi$.

------------------------------------------------------------------------

## Multiple Periods and Variation in Treatment Timing

This framework extends [Difference-in-Differences](#sec-difference-in-differences) to settings where:

-   Treatment is adopted **at different times** across units (staggered adoption).
-   There are **more than two time periods**, and outcomes are measured repeatedly over time.

Such situations are increasingly common in applied economics, public policy evaluations, and longitudinal studies.

------------------------------------------------------------------------

### Group-Time Average Treatment Effects [@callaway2021difference]

**Notation Recap**

-   $Y_{it}(0)$: Potential outcome for unit $i$ at time $t$ in the absence of treatment.

-   $Y_{it}(g)$: Potential outcome for unit $i$ at time $t$ if first treated in period $g$.

-   $Y_{it}$: Observed outcome for unit $i$ at time $t$.

    $$
    Y_{it} =
    \begin{cases}
    Y_{it}(0), & \text{if unit } i \text{ never treated ( } C_i = 1 \text{)} \\
    1\{G_i > t\} Y_{it}(0) + 1\{G_i \le t\} Y_{it}(G_i), & \text{otherwise}
    \end{cases}
    $$

-   $G_i$: Group assignment, i.e., the time period when unit $i$ first receives treatment.

-   $C_i = 1$: Indicator that unit $i$ never receives treatment (the never-treated group).

-   $D_{it} = 1\{G_i \le t\}$: Indicator that unit $i$ has been treated by time $t$.

------------------------------------------------------------------------

**Assumptions**

The following assumptions are typically imposed to identify treatment effects in staggered adoption settings.

1.  **Staggered Treatment Adoption**\
    Once treated, a unit remains treated in all subsequent periods.\
    Formally, $D_{it}$ is non-decreasing in $t$.

2.  **Parallel Trends Assumptions** (Conditional or Unconditional on Covariates)

    Two common variants:

    -   **Parallel trends based on never-treated units**: $$
        \mathbb{E}[Y_t(0) - Y_{t-1}(0) | G_i = g] = \mathbb{E}[Y_t(0) - Y_{t-1}(0) | C_i = 1]
        $$ Interpretation:
        -   The average potential outcome trends of the treated group ($G_i = g$) are the same as the **never-treated** group, absent treatment.
    -   **Parallel trends based on not-yet-treated units**: $$
        \mathbb{E}[Y_t(0) - Y_{t-1}(0) | G_i = g] = \mathbb{E}[Y_t(0) - Y_{t-1}(0) | D_{is} = 0, G_i \ne g]
        $$ Interpretation:
        -   Units **not yet treated** by time $s$ ($D_{is} = 0$) can serve as controls for units first treated at $g$.

    These assumptions can also be **conditional on covariates** $X$, as:

    $$
    \mathbb{E}[Y_t(0) - Y_{t-1}(0) | X_i, G_i = g] = \mathbb{E}[Y_t(0) - Y_{t-1}(0) | X_i, C_i = 1]
    $$

3.  **Random Sampling**\
    Units are sampled independently and identically from the population.

4.  **Irreversibility of Treatment**\
    Once treated, units do not revert to untreated status.

5.  **Overlap (Positivity)**\
    For each group $g$, the propensity of receiving treatment at $g$ lies strictly within $(0, 1)$: $$
    0 < \mathbb{P}(G_i = g | X_i) < 1
    $$

------------------------------------------------------------------------

The Group-Time ATT, $ATT(g, t)$, measures the average treatment effect for units first treated in period $g$, evaluated at time $t$.

$$
ATT(g, t) = \mathbb{E}[Y_t(g) - Y_t(0) | G_i = g]
$$

Interpretation:

-   $g$ indexes when the group first receives treatment.

-   $t$ is the time period when the effect is evaluated.

-   $ATT(g, t)$ captures how treatment effects evolve over time, following adoption at time $g$.

------------------------------------------------------------------------

Identification of $ATT(g, t)$

1.  **Using Never-Treated Units as Controls**: $$
    ATT(g, t) = \mathbb{E}[Y_t - Y_{g-1} | G_i = g] - \mathbb{E}[Y_t - Y_{g-1} | C_i = 1], \quad \forall t \ge g
    $$

2.  **Using Not-Yet-Treated Units as Controls**: $$
    ATT(g, t) = \mathbb{E}[Y_t - Y_{g-1} | G_i = g] - \mathbb{E}[Y_t - Y_{g-1} | D_{it} = 0, G_i \ne g], \quad \forall t \ge g
    $$

3.  **Conditional Parallel Trends (with Covariates)**:\
    If treatment assignment depends on covariates $X_i$, adjust the parallel trends assumption:

    -   **Never-treated controls**: $$
        ATT(g, t) = \mathbb{E}[Y_t - Y_{g-1} | X_i, G_i = g] - \mathbb{E}[Y_t - Y_{g-1} | X_i, C_i = 1], \quad \forall t \ge g
        $$
    -   **Not-yet-treated controls**: $$
        ATT(g, t) = \mathbb{E}[Y_t - Y_{g-1} | X_i, G_i = g] - \mathbb{E}[Y_t - Y_{g-1} | X_i, D_{it} = 0, G_i \ne g], \quad \forall t \ge g
        $$

------------------------------------------------------------------------

Aggregating $ATT(g, t)$: Common Parameters of Interest

1.  [Average Treatment Effect] per Group ($\theta_S(g)$):\
    Average effect over all periods after treatment for group $g$: $$
    \theta_S(g) = \frac{1}{\tau - g + 1} \sum_{t = g}^{\tau} ATT(g, t)
    $$

    -   $\tau$: Last time period in the panel.

2.  Overall [Average Treatment Effect on the Treated] (ATT) ($\theta_S^O$):\
    Weighted average of $\theta_S(g)$ across groups $g$, weighted by their group size: $$
    \theta_S^O = \sum_{g=2}^{\tau} \theta_S(g) \cdot \mathbb{P}(G_i = g)
    $$

3.  **Dynamic Treatment Effects** ($\theta_D(e)$):\
    Average effect after $e$ periods of treatment exposure: $$
    \theta_D(e) = \sum_{g=2}^{\tau} \mathbb{1}\{g + e \le \tau\} \cdot ATT(g, g + e) \cdot \mathbb{P}(G_i = g | g + e \le \tau)
    $$

4.  **Calendar Time Treatment Effects** ($\theta_C(t)$):\
    Average treatment effect at time $t$ across all groups treated by $t$: $$
    \theta_C(t) = \sum_{g=2}^{\tau} \mathbb{1}\{g \le t\} \cdot ATT(g, t) \cdot \mathbb{P}(G_i = g | g \le t)
    $$

5.  **Average Calendar Time Treatment Effect** ($\theta_C$):\
    Average of $\theta_C(t)$ across all post-treatment periods: $$
    \theta_C = \frac{1}{\tau - 1} \sum_{t=2}^{\tau} \theta_C(t)
    $$

------------------------------------------------------------------------

## Staggered Difference-in-Differences

In settings where treatment is introduced at different times across units---known as **staggered treatment adoption**---researchers often use staggered DiD designs (sometimes called event-study DiD or dynamic DiD). However, standard TWFE regression methods can lead to biased estimates in these contexts due to treatment effect heterogeneity and variation in treatment timing.

For applied guidance, see [@wing2024designing] and recommendations in [@baker2022much].

------------------------------------------------------------------------

Best Practices and Recommendations

1.  **When is TWFE Appropriate?**

-   **Single Treatment Period**\
    TWFE DiD regressions are suitable when there is **only one treatment period** (i.e., no variation in treatment timing).

-   **Homogeneous Treatment Effects**\
    If there is a strong theoretical rationale to assume that treatment effects are homogeneous across cohorts and over time, TWFE may still be appropriate.

2.  **Diagnosing and Addressing Bias in TWFE with Staggered Adoption**

-   **Plot Treatment Timing**\
    Examine the distribution of **treatment timing across units**. Heterogeneous timing suggests **high risk of bias** in standard TWFE regressions.

-   **Decomposition Methods**\
    Use decomposition approaches, such as [@goodman2021difference], to understand the weights TWFE places on treatment effects across cohorts and time.

    -   If decomposition is infeasible (e.g., unbalanced panels), the share of never-treated units can indicate potential bias severity.

-   **Discuss Treatment Effect Heterogeneity**\
    Researchers should explicitly discuss the likelihood and nature of treatment effect heterogeneity.

3.  **Event-Study Specifications with TWFE**

-   **Avoid Arbitrary Binning**\
    Do not collapse or bin time periods unless you can **justify homogeneity** in effects.

-   **Full Relative-Time Indicators**\
    Include **fully flexible** relative time indicators, and **justify the reference period** (usually $l = -1$ or the period just prior to treatment).

-   **Multicollinearity and Bias**\
    Be cautious about multicollinearity when including leads and lags, which can bias estimates and lead to false detection of pre-trends.

4.  **Alternative Estimators to Address Bias**

-   **Stacked Regressions**

-   [@callaway2021difference]: Group-Time Average Treatment Effects

-   [@sun2021estimating]: Event-Study DiD with clean controls

    Alternative methods provide **unbiased estimates** of ATT even with staggered adoption and heterogeneous effects.

------------------------------------------------------------------------

Example of a Basic TWFE Event-Study Specification

Adapted from [@stevenson2006bargaining]:

$$
\begin{aligned}
Y_{it} &= \sum_k \beta_k \cdot Treatment_{it}^k + \eta_i + \lambda_t + Controls_{it} + \epsilon_{it}
\end{aligned}
$$

-   $Treatment_{it}^k$: Dummy variables equal to 1 if unit $i$ is treated $k$ years before period $t$.
-   $\eta_i$: Unit fixed effects controlling for time-invariant heterogeneity.
-   $\lambda_t$: Time fixed effects controlling for common shocks.
-   **Standard Errors (SE)**: Typically clustered at the **group level** (e.g., state or cohort).

*Common practice*: Drop the period **immediately before treatment** to avoid perfect multicollinearity.

------------------------------------------------------------------------

### General TWFE Event-Study Model [@sun2021estimating]

Relative Period Bin Indicator

$$
D_{it}^l = \mathbb{1}(t - E_i = l)
$$

-   $E_i$: The time period when unit $i$ first receives treatment.
-   $l$: The **relative time period**---how many periods have passed since treatment began.

------------------------------------------------------------------------

1.  **Static Specification**

$$
Y_{it} = \alpha_i + \lambda_t + \mu_g \sum_{l \ge 0} D_{it}^l + \epsilon_{it}
$$

-   $\alpha_i$: Unit fixed effects.
-   $\lambda_t$: Time fixed effects.
-   $\mu_g$: Effect for group $g$.
-   Excludes periods **prior to treatment**.

------------------------------------------------------------------------

2.  **Dynamic Specification**

$$
Y_{it} = \alpha_i + \lambda_t + \sum_{\substack{l = -K \\ l \neq -1}}^{L} \mu_l D_{it}^l + \epsilon_{it}
$$

-   Includes leads and lags of treatment indicators $D_{it}^l$.
-   Excludes one period (typically $l = -1$) to avoid perfect collinearity.
-   Tests for pre-treatment parallel trends rely on the leads ($l < 0$).

------------------------------------------------------------------------

## Critiques of TWFE in Staggered DiD

Authors like [@sun2021estimating], [@callaway2021difference], and [@goodman2021difference] have raised concerns that TWFE DiD regressions:

-   Mix treatment effects across cohorts, leading to negative weights and biased estimates.
-   Pre-treatment leads may appear non-zero due to contamination by post-treatment effects from earlier-treated groups.
-   Long-term treatment effects (lags) may be biased due to heterogeneous adoption timing.

Recent evidence in finance and accounting (e.g., [@baker2022much]) shows that using newer estimators often leads to **null or much smaller causal estimates**.

------------------------------------------------------------------------

## Key Assumptions of Staggered DiD Designs

1.  **Rollout Exogeneity**\
    Treatment assignment and timing should be uncorrelated with potential outcomes.

    -   Evidence: Regress adoption on pre-treatment variables. And if you find evidence of correlation, include linear trends interacted with pre-treatment variables [@hoynes2009consumption]
    -   Evidence: [@deshpande2019screened, p. 223]
        -   Treatment is random: Regress treatment status at the unit level to all pre-treatment observables. If you have some that are predictive of treatment status, you might have to argue why it's not a worry. At best, you want this.
        -   Treatment timing is random: Conditional on treatment, regress timing of the treatment on pre-treatment observables. At least, you want this.

2.  **No Confounding Events**\
    Ensure no other policies or shocks coincide with the staggered treatment rollout.

3.  **Exclusion Restrictions**

    -   **No Anticipation**: Treatment timing should not affect outcomes prior to treatment.
    -   **Invariance to History**: Treatment duration shouldn't matter; only the treated status matters (often violated).

4.  **Standard DID Assumptions**

    -   **Parallel Trends** (Conditional or Unconditional)
    -   **Random Sampling**
    -   **Overlap** (Common Support)
    -   **Effect Additivity**

------------------------------------------------------------------------

## Remedies for Staggered DiD Biases

[@baker2022much]

1.  **Cohort-Specific Comparisons**

-   Compare **each treated cohort** to appropriate controls:
    -   **Not-yet-treated** units.
    -   **Never-treated** units.

2.  **Alternative Estimators**

-   [@goodman2021difference]: Decomposition methods for TWFE DiD.
-   [@callaway2021difference]: ATT estimators, allowing for heterogeneity and flexible design.
-   [@sun2021estimating]: Event-study estimation with "clean" controls (a special case of @callaway2021difference)
-   [@de2020two]: Two-stage DiD approaches.
-   [@borusyak2021revisiting]: Simple DiD estimators with automatic inference.

3.  [Stacked DID](#sec-stacked-difference-in-differences) (Simpler but Biased)

-   Construct stacked datasets for each treatment cohort.
-   Run separate regressions for each event.
-   Examples:
    -   [@gormley2011growing]
    -   [@cengiz2019effect]
    -   [@deshpande2019screened]

------------------------------------------------------------------------

### Stacked Difference-in-Differences {#sec-stacked-difference-in-differences}

The **Stacked DiD** approach addresses key limitations of standard TWFE models in **staggered adoption designs**, particularly **treatment effect heterogeneity** and **timing variations**. By constructing **sub-experiments** around each treatment event, researchers can isolate cleaner comparisons and reduce contamination from improperly specified control groups.

Basic TWFE Specification

$$
Y_{it} = \beta_{FE} D_{it} + A_i + B_t + \epsilon_{it}
$$

-   $Y_{it}$: Outcome for unit $i$ at time $t$.
-   $D_{it}$: Treatment indicator (1 if treated, 0 otherwise).
-   $A_i$: Unit (group) fixed effects.
-   $B_t$: Time period fixed effects.
-   $\epsilon_{it}$: Idiosyncratic error term.

------------------------------------------------------------------------

Steps in the Stacked DiD Procedure

#### Choose an Event Window

Define:

-   $\kappa_a$: Number of **pre-treatment** periods to include in the event window (lead periods).
-   $\kappa_b$: Number of **post-treatment** periods to include in the event window (lag periods).

**Implication**:\
Only events where sufficient **pre- and post-treatment periods** exist will be included (i.e., excluding those events that do not meet this criteria).

------------------------------------------------------------------------

#### Enumerate Sub-Experiments

Define:

-   $T_1$: First period in the panel.
-   $T_T$: Last period in the panel.
-   $\Omega_A$: The set of **treatment adoption periods** that fit within the event window:

$$
\Omega_A = \left\{ A_i \;\middle|\; T_1 + \kappa_a \le A_i \le T_T - \kappa_b \right\}
$$

-   Each $A_i$ represents an **adoption period** for unit $i$ that has enough time on both sides of the event.

Let $d = 1, \dots, D$ index the **sub-experiments** in $\Omega_A$.

-   $\omega_d$: The event (adoption) date of the $d$-th sub-experiment.

------------------------------------------------------------------------

#### Define Inclusion Criteria

**Valid Treated Units**

-   In sub-experiment $d$, treated units have adoption date exactly equal to $\omega_d$.
-   A unit may only be treated in one sub-experiment to avoid duplication.

**Clean Control Units**

-   Controls are units where $A_i > \omega_d + \kappa_b$, i.e.,
    -   They are **never treated**, or
    -   They are **treated in the far future** (beyond the post-event window).
-   A control unit can appear in multiple sub-experiments, but this requires correcting standard errors (see below).

**Valid Time Periods**

-   Only observations where\
    $$
    \omega_d - \kappa_a \le t \le \omega_d + \kappa_b
    $$\
    are included.
-   This ensures the analysis is centered on the event window.

#### Specify Estimating Equation

Basic DiD Specification in the Stacked Dataset

$$
Y_{itd} = \beta_0 + \beta_1 T_{id}  + \beta_2 P_{td} + \beta_3 (T_{id} \times P_{td}) + \epsilon_{itd}
$$

Where:

-   $i$: Unit index

-   $t$: Time index

-   $d$: Sub-experiment index

-   $T_{id}$: Indicator for **treated units** in sub-experiment $d$

-   $P_{td}$: Indicator for **post-treatment periods** in sub-experiment $d$

-   $\beta_3$: Captures the **DiD estimate** of the treatment effect.

Equivalent Form with Fixed Effects

$$
Y_{itd} = \beta_3 (T_{id} \times P_{td}) + \theta_{id} + \gamma_{td} + \epsilon_{itd}
$$

where

-   $\theta_{id}$: Unit-by-sub-experiment fixed effect.

-   $\gamma_{td}$: Time-by-sub-experiment fixed effect.

Note:

-   $\beta_3$ summarizes the average treatment effect across all sub-experiments but does not allow for dynamic effects by time since treatment.

#### Stacked Event Study Specification

Define Time Since Event ($YSE_{td}$):

$$
YSE_{td} = t- \omega_d
$$

where

-   Measures time since the event (relative time) in sub-experiment $d$.

-   $YSE_{td} \in [-\kappa_a, \dots, 0, \dots, \kappa_b]$ in every sub-experiment.

**Event-Study Regression (Sub-Experiment Level)**

$$
Y_{it}^d = \sum_{j = -\kappa_a}^{\kappa_b} \beta_j^d . 1 (YSE_{td} = j) + \sum_{j = -\kappa_a}^{\kappa_b} \delta_j^d (T_{id} . 1 (YSE_{td} = j)) + \theta_i^d + \epsilon_{it}^d
$$

where

-   Separate coefficients for each sub-experiment $d$.

-   $\delta_j^d$: Captures treatment effects at relative time $j$ within sub-experiment $d$.

**Pooled Stacked Event-Study Regression**

$$
Y_{itd} = \sum_{j = -\kappa_a}^{\kappa_b} \beta_j \cdot \mathbb{1}(YSE_{td} = j) + \sum_{j = -\kappa_a}^{\kappa_b} \delta_j \left( T_{id} \cdot \mathbb{1}(YSE_{td} = j) \right) + \theta_{id} + \epsilon_{itd}
$$

-   Pooled coefficients $\delta_j$ reflect average treatment effects by event time $j$ across sub-experiments.

#### Clustering in Stacked DID

-   **Cluster at Unit × Sub-Experiment Level** [@cengiz2019effect]: Accounts for units appearing multiple times across sub-experiments.

-   **Cluster at Unit Level** [@deshpande2019screened]: Appropriate when units are uniquely identified and do not appear in multiple sub-experiments.

------------------------------------------------------------------------


``` r
library(did)
library(tidyverse)
library(fixest)

# Load example data
data(base_stagg)

# Get treated cohorts (exclude never-treated units coded as 10000)
cohorts <- base_stagg %>%
    filter(year_treated != 10000) %>%
    distinct(year_treated) %>%
    pull()

# Function to generate data for each sub-experiment
getdata <- function(j, window) {
    base_stagg %>%
        filter(
            year_treated == j |               # treated units in cohort j
            year_treated > j + window         # controls not treated soon after
        ) %>%
        filter(
            year >= j - window &
            year <= j + window                # event window bounds
        ) %>%
        mutate(df = j)                        # sub-experiment indicator
}

# Generate the stacked dataset
stacked_data <- map_df(cohorts, ~ getdata(., window = 5)) %>%
    mutate(
        rel_year = if_else(df == year_treated, time_to_treatment, NA_real_)
    ) %>%
    fastDummies::dummy_cols("rel_year", ignore_na = TRUE) %>%
    mutate(across(starts_with("rel_year_"), ~ replace_na(., 0)))

# Estimate fixed effects regression on the stacked data
stacked_result <- feols(
    y ~ `rel_year_-5` + `rel_year_-4` + `rel_year_-3` + `rel_year_-2` +
        rel_year_0 + rel_year_1 + rel_year_2 + rel_year_3 +
        rel_year_4 + rel_year_5 |
        id ^ df + year ^ df,
    data = stacked_data
)

# Extract coefficients and standard errors
stacked_coeffs <- stacked_result$coefficients
stacked_se <- stacked_result$se

# Insert zero for the omitted period (usually -1)
stacked_coeffs <- c(stacked_coeffs[1:4], 0, stacked_coeffs[5:10])
stacked_se <- c(stacked_se[1:4], 0, stacked_se[5:10])
```


``` r
# Plotting estimates from three methods: Callaway & Sant'Anna, Sun & Abraham, and Stacked DiD

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
    stacked = stacked_coeffs
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
    ), alpha = 0.2) +
    geom_line(aes(
        x = period,
        y = est,
        group = estimator,
        color = estimator
    ),
    linewidth = 1.2) +
    
    labs(
        title = "Comparison of Dynamic Treatment Effects",
        x = "Event Time (Periods since Treatment)",
        y = "Estimated ATT",
        color = "Estimator"
    ) + 
    causalverse::ama_theme()
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-4-1.png" width="90%" style="display: block; margin: auto;" />

------------------------------------------------------------------------

### Goodman-Bacon Decomposition {#sec-goodman-bacon-decomposition}

Paper: [@goodman2021difference]

For an excellent explanation slides by the author, [see](https://www.stata.com/meeting/chicago19/slides/chicago19_Goodman-Bacon.pdf)

Takeaways:

-   A pairwise DID ($\tau$) gets more weight if the change is close to the middle of the study window

-   A pairwise DID ($\tau$) gets more weight if it includes more observations.

Code from `bacondecomp` vignette


``` r
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


``` r
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


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-7-1.png" width="90%" style="display: block; margin: auto;" />

With time-varying controls that can identify variation within-treatment timing group, the"early vs. late" and "late vs. early" estimates collapse to just one estimate (i.e., both treated).

### DID with in and out treatment condition

#### Panel Match

As noted in [@imai2021use], the TWFE estimator is not a fully nonparametric approach and is sensitive to incorrect model specifications (i.e., model dependence).

-   To mitigate model dependence, they propose matching methods for panel data.
-   Implementations are available via the `wfe` (Weighted Fixed Effects) and `PanelMatch` R packages.

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


``` r
library(PanelMatch)
```

**Treatment Variation plot**

-   Visualize the variation of the treatment across space and time

-   Aids in discerning whether the treatment fluctuates adequately over time and units or if the variation is primarily clustered in a subset of data.


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-9-1.png" width="90%" style="display: block; margin: auto;" />

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


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-10-1.png" width="90%" style="display: block; margin: auto;" />

Control units and the treated unit have identical treatment histories over the lag window (1988-1991)


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-11-1.png" width="90%" style="display: block; margin: auto;" />

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


``` r
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


``` r
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


``` r
plot(msets.none)
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-14-1.png" width="90%" style="display: block; margin: auto;" />

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


``` r
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
#> t_0 -0.09493854 -0.015198483

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
#> t_0  0.15885415 0.05691437


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
#> t_0  0.05044958 0.01943728

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
#> t_0 0.040272235 0.04167921
```

**get_covariate_balance Function Options:**

-   Allows for the generation of plots displaying covariate balance using **`plot = TRUE`**.

-   Plots can be customized using arguments typically used with the base R **`plot`** method.

-   Option to set **`use.equal.weights = TRUE`** for:

    -   Obtaining the balance of unrefined sets.

    -   Facilitating understanding of the refinement's impact.


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-16-1.png" width="90%" style="display: block; margin: auto;" />

``` r

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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-16-2.png" width="90%" style="display: block; margin: auto;" />

``` r


balance_scatter(
    matched_set_list = list(PM.results.maha$att,
                            PM.results.ps.weight$att),
    data = dem,
    covariates = c("y", "tradewb")
)
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-16-3.png" width="90%" style="display: block; margin: auto;" />

**`PanelEstimate`**

-   **Standard Error Calculation Methods**

    -   There are different methods available:

        -   **Bootstrap** (default method with 1000 iterations).

        -   **Conditional**: Assumes independence across units, but not time.

        -   **Unconditional**: Doesn't make assumptions of independence across units or time.

    -   For **`qoi`** values set to `att`, `art`, or `atc` [@imai2021matching]:

        -   You can use analytical methods for calculating standard errors, which include both "conditional" and "unconditional" methods.


``` r
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
#> 0.6387986 1.0548882 1.4282656 1.7870905 2.2085197


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
#> Standard errors computed with conditional method
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-17-1.png" width="90%" style="display: block; margin: auto;" />

**Moderating Variables**


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-18-1.png" width="90%" style="display: block; margin: auto;" />

``` r

plot(PE.results[[2]])
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-18-2.png" width="90%" style="display: block; margin: auto;" />

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
> \end{equation} where $N_1 = \sum_{i'= 1}^N \sum_{t' = L+1}^{T-F} D_{i't'}$ denotes the total number of treated observations and $\bar{V}_{t-l,j} = \sum_{i=1}^N D_{i,t-l,j}/N$. We then aggregate this covariate balance measure across all treated observations for each covariate and pre-treatment time period:

```{=tex}
\begin{equation}
\bar{B}(j, l) = \frac{1}{N_1} \sum_{i=1}^N \sum_{t = L+ 1}^{T-F}D_{it} B_{it}(j,l)
\label{eq:aggbalance}
\end{equation}
```
> Lastly, we evaluate the balance of lagged outcome variables over several pre-treatment periods and that of time-varying covariates. This examination aids in assessing the validity of the parallel trend assumption integral to the DiD estimator justification.

In Figure for balance scatter, we demonstrate the enhancement of covariate balance thank to the refinement of matched sets. Each scatter plot contrasts the absolute standardized mean difference, as detailed in Equation \@ref(eq:aggbalance), before (horizontal axis) and after (vertical axis) this refinement. Points below the 45-degree line indicate an improved standardized mean balance for certain time-varying covariates post-refinement. The majority of variables benefit from this refinement process. Notably, the propensity score weighting (bottom panel) shows the most significant improvement, whereas Mahalanobis matching (top panel) yields a more modest improvement.


``` r
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


``` r
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


``` r
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


``` r
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


``` r
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


``` r
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


``` r
library(knitr)
include_graphics(file.path(getwd(), "images", "p_covariate_balance.png"))
```

Note: Each graph displays the standardized mean difference, as outlined in Equation \@ref(eq:aggbalance), plotted on the vertical axis across a pre-treatment duration of four years represented on the horizontal axis. The leftmost column illustrates the balance prior to refinement, while the subsequent three columns depict the covariate balance post the application of distinct refinement techniques. Each individual line signifies the balance of a specific variable during the pre-treatment phase.The red line is tradewb and blue line is the lagged outcome variable.

In Figure \@ref(fig:balancepretreat), we observe a marked improvement in covariate balance due to the implemented matching procedures during the pre-treatment period. Our analysis prioritizes methods that adjust for time-varying covariates over a span of four years preceding the treatment initiation. The two rows delineate the standardized mean balance for both treatment modalities, with individual lines representing the balance for each covariate.

Across all scenarios, the refinement attributed to matched sets significantly enhances balance. Notably, using propensity score weighting considerably mitigates imbalances in confounders. While some degree of imbalance remains evident in the Mahalanobis distance and propensity score matching techniques, the standardized mean difference for the lagged outcome remains stable throughout the pre-treatment phase. This consistency lends credence to the validity of the proposed DiD estimator.

**Estimation Results**

We now detail the estimated ATTs derived from the matching techniques. Figure below offers visual representations of the impacts of treatment initiation (upper panel) and treatment reversal (lower panel) on the outcome variable for a duration of 5 years post-transition, specifically, ($F = 0, 1, …, 4$). Across the five methods (columns), it becomes evident that the point estimates of effects associated with treatment initiation consistently approximate zero over the 5-year window. In contrast, the estimated outcomes of treatment reversal are notably negative and maintain statistical significance through all refinement techniques during the initial year of transition and the 1 to 4 years that follow, provided treatment reversal is permissible. These effects are notably pronounced, pointing to an estimated reduction of roughly X% in the outcome variable.

Collectively, these findings indicate that the transition into the treated state from its absence doesn't invariably lead to a heightened outcome. Instead, the transition from the treated state back to its absence exerts a considerable negative effect on the outcome variable in both the short and intermediate terms. Hence, the positive effect of the treatment (if we were to use traditional DiD) is actually driven by the negative effect of treatment reversal.


``` r
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


``` r
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


``` r
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


``` r
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

-   [@gardner2022two; @borusyak2021revisiting]

-   [@brown2023simple]


``` r
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


``` r
plot(model.fect, stats = "F.p", type = 'exit')
```

##### Placebo Test

By selecting a part of the data and excluding observations within a specified range to improve the model fitting, we then evaluate whether the estimated Average Treatment Effect (ATT) within this range significantly differs from zero. This approach helps us analyze the periods before treatment.

If this test fails, either the functional form or strict exogeneity assumption is problematic.


``` r
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


``` r
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
2.  Nuclear Norm (i.e., Lasso): computationally feasible (using SOFT-IMPUTE algorithm [@mazumder2010spectral]).
3.  Rank (i.e., Subset selection): not computationally feasible

This method allows to

-   use more covariates

-   leverage data from treated units (can be used when treatment effect is constant and pattern of missing is not complex).

-   have autocorrelated errors

-   have weighted loss function (i.e., take into account the probability of outcomes for a unit being missing)

### @gardner2022two and @borusyak2024revisiting

-   Estimate the time and unit fixed effects separately

-   Known as the imputation method [@borusyak2024revisiting] or two-stage DiD [@gardner2022two]


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-33-1.png" width="90%" style="display: block; margin: auto;" />

``` r

coefplot(est)
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-33-2.png" width="90%" style="display: block; margin: auto;" />


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-34-1.png" width="90%" style="display: block; margin: auto;" />

@borusyak2024revisiting `didimputation`

This version is currently not working


``` r
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


``` r
# remotes::install_github("shuo-zhang-ucsb/did_multiplegt") 
library(DIDmultiplegt)
library(fixest)
library(tidyverse)

data("base_stagg")

res <-
    did_multiplegt(
        mode = "dyn",
        df = base_stagg |>
            dplyr::mutate(treatment = dplyr::if_else(time_to_treatment < 0, 0, 1)),
        outcome = "y",
        group = "year_treated",
        time = "year",
        treatment = "treatment",
        effects = 5,
        controls = "x1",
        placebo = 2
    )
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-36-1.png" width="90%" style="display: block; margin: auto;" />

``` r

head(res)
#> $args
#> $args$df
#> ..1
#> 
#> $args$outcome
#> [1] "y"
#> 
#> $args$group
#> [1] "year_treated"
#> 
#> $args$time
#> [1] "year"
#> 
#> $args$treatment
#> [1] "treatment"
#> 
#> $args$effects
#> [1] 5
#> 
#> $args$normalized
#> [1] FALSE
#> 
#> $args$normalized_weights
#> [1] FALSE
#> 
#> $args$effects_equal
#> [1] FALSE
#> 
#> $args$placebo
#> [1] 2
#> 
#> $args$controls
#> [1] "x1"
#> 
#> $args$trends_lin
#> [1] FALSE
#> 
#> $args$same_switchers
#> [1] FALSE
#> 
#> $args$same_switchers_pl
#> [1] FALSE
#> 
#> $args$switchers
#> [1] ""
#> 
#> $args$only_never_switchers
#> [1] FALSE
#> 
#> $args$ci_level
#> [1] 95
#> 
#> $args$graph_off
#> [1] FALSE
#> 
#> $args$save_sample
#> [1] FALSE
#> 
#> $args$less_conservative_se
#> [1] FALSE
#> 
#> $args$dont_drop_larger_lower
#> [1] FALSE
#> 
#> $args$drop_if_d_miss_before_first_switch
#> [1] FALSE
#> 
#> 
#> $results
#> $results$N_Effects
#> [1] 5
#> 
#> $results$N_Placebos
#> [1] 2
#> 
#> $results$Effects
#>              Estimate      SE    LB CI    UB CI  N Switchers N.w Switchers.w
#> Effect_1     -5.21421 0.27175 -5.74682 -4.68160 54         9 675          45
#> Effect_2     -3.59669 0.47006 -4.51800 -2.67538 44         8 580          40
#> Effect_3     -2.18743 0.03341 -2.25292 -2.12194 35         7 490          35
#> Effect_4     -0.90231 0.77905 -2.42922  0.62459 27         6 405          30
#> Effect_5      0.98492 0.23620  0.52198  1.44786 20         5 325          25
#> 
#> $results$ATE
#>              Estimate      SE LB CI    UB CI  N Switchers N.w Switchers.w
#> Av_tot_eff   -2.61436 0.34472 -3.29 -1.93872 80        35 805         175
#> 
#> $results$delta_D_avg_total
#> [1] 2.714286
#> 
#> $results$max_pl
#> [1] 8
#> 
#> $results$max_pl_gap
#> [1] 7
#> 
#> $results$p_jointeffects
#> [1] 0
#> 
#> $results$Placebos
#>              Estimate      SE    LB CI   UB CI  N Switchers N.w Switchers.w
#> Placebo_1     0.08247 0.25941 -0.42597 0.59091 44         8 580          40
#> Placebo_2    -0.12395 0.54014 -1.18260 0.93469 27         6 405          30
#> 
#> $results$p_jointplacebo
#> [1] 0.9043051
#> 
#> 
#> $coef
#> $coef$b
#> Effect_1     Effect_2     Effect_3     Effect_4     Effect_5     Placebo_1    
#>     -5.21421     -3.59669     -2.18743     -0.90231      0.98492      0.08247 
#> Placebo_2    
#>     -0.12395 
#> 
#> $coef$vcov
#>              Effect_1   Effect_2     Effect_3   Effect_4    Effect_5
#> Effect_1   0.07384592 -0.1474035 -0.037481125 -0.3403796 -0.06481787
#> Effect_2  -0.14740348  0.2209610 -0.111038691 -0.4139371 -0.13837544
#> Effect_3  -0.03748112 -0.1110387  0.001116335 -0.3040148 -0.02845308
#> Effect_4  -0.34037958 -0.4139371 -0.304014789  0.6069132 -0.33135154
#> Effect_5  -0.06481787 -0.1383754 -0.028453082 -0.3313515  0.05578983
#> Placebo_1 -0.07056996 -0.1441275 -0.034205168 -0.3371036 -0.06154192
#> Placebo_2 -0.18279619 -0.2563538 -0.146431403 -0.4493299 -0.17376815
#>             Placebo_1  Placebo_2
#> Effect_1  -0.07056996 -0.1827962
#> Effect_2  -0.14412752 -0.2563538
#> Effect_3  -0.03420517 -0.1464314
#> Effect_4  -0.33710362 -0.4493299
#> Effect_5  -0.06154192 -0.1737681
#> Placebo_1  0.06729400 -0.1795202
#> Placebo_2 -0.17952024  0.2917465
#> 
#> 
#> $plot
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-36-2.png" width="90%" style="display: block; margin: auto;" />

I don't recommend the `TwoWayFEWeights` since it only gives the aggregated average treatment effect over all post-treatment periods, but not for each period.


``` r
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
#> 
#> Under the common trends assumption,
#> the TWFE coefficient beta, equal to -3.4676, estimates a weighted sum of 45 ATTs.
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
#>   min σ(Δ) compatible with treatment effect of opposite sign than β_fe in all (g,t) cells: 36.1549
#>   Reference: Corollary 1, de Chaisemartin, C and D'Haultfoeuille, X (2020a)
#> 
#> The development of this package was funded by the European Union (ERC, REALLYCREDIBLE,GA N. 101043899).
```

### @callaway2021difference {#callaway2021difference}

-   `staggered` [package](https://github.com/jonathandroth/staggered)

-   Group-time average treatment effect


``` r
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
#>       estimate        se se_neyman eventTime
#> 1  0.211213653 0.1132266 0.1136513        -9
#> 2 -0.104342678 0.1792772 0.1800337        -8
#> 3 -0.006515977 0.1518244 0.1536703        -7
#> 4 -0.108274019 0.2375445 0.2381854        -6
#> 5 -0.353660876 0.2090569 0.2102740        -5
#> 6  0.066907666 0.2000045 0.2154505        -4


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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-38-1.png" width="90%" style="display: block; margin: auto;" />


``` r
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


``` r
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

-   @borusyak2024revisiting presumes a uniform treatment effect across cohorts, effectively simplifying CATT to ATT.

Identifying Assumptions for dynamic TWFE:

1.  **Parallel Trends**: Baseline outcomes follow parallel trends across cohorts before treatment.

    -   This gives us all CATT (including own, included bins, and excluded bins)

2.  **No Anticipatory Behavior**: There is no effect of the treatment during pre-treatment periods, indicating that outcomes are not influenced by the anticipation of treatment.

3.  **Treatment Effect Homogeneity**: The treatment effect is consistent across cohorts for each relative period. Each adoption cohort should have the same path of treatment effects. In other words, the trajectory of each treatment cohort is similar. Compare to other designs:

    1.  @athey2022design assume heterogeneity of treatment effects vary over adoption cohorts, but not over time.

    2.  @borusyak2024revisiting assume heterogeneity of treatment effects vary over time, but not over adoption cohorts.

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


``` r
library(fixest)
data("base_stagg")
res_sa20 = feols(y ~ x1 + sunab(year_treated, year) | id + year, base_stagg)
iplot(res_sa20)
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-41-1.png" width="90%" style="display: block; margin: auto;" />

``` r

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


``` r
# devtools::install_github("kylebutts/fwlplot")
library(fwlplot)
fwl_plot(y ~ x1, data = base_stagg)
```

<img src="30-dif-in-dif_files/figure-html/plot residuals-1.png" width="90%" style="display: block; margin: auto;" />

``` r

fwl_plot(y ~ x1 | id + year, data = base_stagg, n_sample = 100)
```

<img src="30-dif-in-dif_files/figure-html/plot residuals-2.png" width="90%" style="display: block; margin: auto;" />

``` r

fwl_plot(y ~ x1 | id + year, data = base_stagg, n_sample = 100, fsplit = ~ treated)
```

<img src="30-dif-in-dif_files/figure-html/plot residuals-3.png" width="90%" style="display: block; margin: auto;" />

### @wooldridge2023simple

use [etwfe](https://grantmcdermott.com/etwfe/)(Extended two-way Fixed Effects) [@wooldridge2023simple]

### Doubly Robust DiD

Also known as the locally efficient doubly robust DiD [@sant2020doubly]

[Code example by the authors](https://psantanna.com/DRDID/index.html)

The package (not method) is rather limited application:

-   Use OLS (cannot handle `glm`)

-   Canonical DiD only (cannot handle DDD).


``` r
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

------------------------------------------------------------------------

## Multiple Treatments

In some settings, researchers encounter **two (or more) treatments** rather than a single treatment and control group. This complicates standard DiD estimation, but a properly structured model ensures accurate identification.

**Additional References**

-   [@fricke2017identification]: Discusses identification challenges in multiple treatment settings.
-   [@de2023two]: Provides a **video tutorial** ([YouTube](https://www.youtube.com/watch?v=UHeJoc27qEM&ab_channel=TaylorWright)) and **code** ([Google Drive](https://drive.google.com/file/d/156Fu73avBvvV_H64wePm7eW04V0jEG3K/view)) for implementing multiple-treatment DiD models.

**Key Principles When Dealing with Multiple Treatments**

1.  **Always include all treatment groups in a single regression model.**

    -   This ensures proper identification of treatment-specific effects while maintaining a clear comparison against the control group.

2.  **Never use one treated group as a control for the other.**

    -   Running separate regressions for each treatment group can lead to biased estimates because each treatment group may differ systematically from the control group in ways that a separate model cannot fully capture.

3.  **Compare the significance of treatment effects** ($\delta_1$ vs. $\delta_2$).

    -   Instead of assuming equal effects, we should formally test whether the effects of the two treatments are statistically different using an F-test or Wald test:

    $$
    H_0: \delta_1 = \delta_2
    $$

    -   If we reject $H_0$, we conclude that the two treatments have significantly different effects.

------------------------------------------------------------------------

### Multiple Treatment Groups: Model Specification

A properly specified DiD regression model with two treatments takes the following form:

$$
\begin{aligned}
Y_{it} &= \alpha + \gamma_1 Treat1_{i} + \gamma_2 Treat2_{i} + \lambda Post_t  \\
&+ \delta_1(Treat1_i \times Post_t) + \delta_2(Treat2_i \times Post_t) + \epsilon_{it}
\end{aligned}
$$

where:

-   $Y_{it}$ = Outcome variable for individual $i$ at time $t$.
-   $Treat1_i$ = 1 if individual $i$ is in **Treatment Group 1**, 0 otherwise.
-   $Treat2_i$ = 1 if individual $i$ is in **Treatment Group 2**, 0 otherwise.
-   $Post_t$ = 1 for post-treatment period, 0 otherwise.
-   **DiD coefficients**:
    -   $\delta_1$ = Effect of Treatment 1.
    -   $\delta_2$ = Effect of Treatment 2.
-   $\epsilon_{it}$ = Error term.

------------------------------------------------------------------------

### Understanding the Control Group in Multiple Treatment DiD

One common concern in multiple-treatment DiD models is how to properly define the control group. A well-specified model ensures that:

-   The control group consists only of untreated individuals, not individuals from another treatment group.
-   The reference category in the regression represents the control group (i.e., individuals with $Treat1_i = 0$ and $Treat2_i = 0$).
-   If $Treat1_i = 1$, then $Treat2_i = 0$ and vice versa.

Failing to correctly specify the control group could lead to incorrect estimates of treatment effects. For example, omitting one of the treatment indicators could unintentionally redefine the control group as a mix of treated and untreated individuals.

------------------------------------------------------------------------

### Alternative Approaches: Separate Regressions vs. One Model

A common question is whether to run one large regression including all treatment groups or to run separate DiD models on subsets of the data. Each approach has implications:

1.  **One Model Approach (Preferred)**

-   Running one comprehensive regression allows for direct comparison between treatment effects in a statistically valid way.
-   The interaction terms ($\delta_1, \delta_2$) ensure that each treatment effect is estimated relative to a common control group.
-   The F-test (or Wald test) enables a formal test of whether the two treatments have significantly different effects.

2.  **Separate Regressions Approach**

-   Running separate DiD models for each treatment group can still be valid, but:
    -   The estimated treatment effects are less efficient because they come from separate samples.
    -   Comparisons become less straightforward, as they rely on confidence interval overlap rather than direct hypothesis testing.
    -   If homoscedasticity holds (i.e., equal error variances across groups), the separate regressions approach is unnecessary. The combined model is more efficient.

Thus, unless there is strong justification for separate regressions (e.g., significant heterogeneity in error variance), the one-model approach is preferred.

------------------------------------------------------------------------

### Handling Treatment Intensity

In some cases, treatments differ **not just in type, but also in intensity** (e.g., low vs. high treatment exposure). If we observe different levels of treatment intensity, we can model it using a **single categorical variable** rather than multiple treatment dummies:

Rather than coding separate dummies for each treatment group, we define a **multi-valued treatment variable**:

$$
Y_{it} = \alpha + \sum_{j=1}^{J} \beta_j (Treatment_j \times Post_t) + \lambda Post_t + \epsilon_{it}
$$

where:

-   $Treatment_j$ is a categorical variable indicating whether an individual belongs to the control group, low-intensity treatment, or high-intensity treatment.
-   This approach allows for cleaner implementation and avoids excessive interaction terms.

This approach has the advantage of:

-   Automatically setting the control group as the reference category.

-   Ensuring correct interpretation of coefficients for different treatment levels.

### Considerations When Individuals Can Move Between Treatment Groups

One potential complication in multiple-treatment DiD settings is when individuals can switch treatment groups over time (e.g., moving from low-intensity to high-intensity treatment after policy implementation).

-   If movement is rare, it may not significantly affect estimates.

-   If movement is frequent, it creates a challenge in causal identification because treatment effects might be confounded by self-selection.

A possible solution is to use an intention-to-treat (ITT) approach, where treatment assignment is based on the initially assigned group, regardless of whether individuals later switch.

### Parallel Trends Assumption in Multiple-Treatment DiD

-   Just as in standard DiD, a key assumption in multiple-treatment DiD models is that the treatment and control groups would have followed parallel trends in the absence of treatment.

-   With multiple treatments, we must check pre-trends separately for each treated group against the control group.

-   If pre-treatment trends are not parallel, we may need to adopt alternative methods such as synthetic control models or event study analyses.

------------------------------------------------------------------------

## Mediation Under DiD

Mediation analysis helps determine whether a treatment affects the outcome directly or through an intermediate variable (mediator). In a DiD framework, this allows us to separate:

1.  Direct effects: The effect of the treatment on the outcome independent of the mediator.
2.  Indirect (mediated) effects: The effect of the treatment that operates through the mediator.

This is useful when a treatment consists of multiple components or when we want to understand mechanisms behind an observed effect.

------------------------------------------------------------------------

### Mediation Model in DiD

To incorporate mediation, we estimate two equations:

**Step 1: Effect of Treatment on the Mediator**

$$
M_{it} = \alpha + \gamma Treat_i + \lambda Post_t + \delta (Treat_i \times Post_t) + \epsilon_{it}
$$ where:

-   $M_{it}$ = Mediator variable (e.g., job search intensity, firm investment, police presence).
-   $\delta$ = Effect of the treatment on the mediator (capturing how the treatment changes $M$).

**Step 2: Effect of Treatment and Mediator on the Outcome**

$$
Y_{it} = \alpha' + \gamma' Treat_i + \lambda' Post_t + \delta' (Treat_i \times Post_t) + \theta M_{it} + \epsilon'_{it}
$$ where:

-   $Y_{it}$ = Outcome variable (e.g., employment, crime rate, firm performance).
-   $\theta$ = Effect of the mediator on the outcome.
-   $\delta'$ = **Direct effect** of the treatment (controlling for the mediator).

------------------------------------------------------------------------

### Interpreting the Results

-   If $\theta$ is statistically significant, it suggests that mediation is occurring---that is, the treatment affects the outcome partly through the mediator.
-   If $\delta'$ is smaller than $\delta$, this indicates that part of the treatment effect is explained by the mediator. The remaining portion of $\delta'$ represents the direct effect.

Thus, we can decompose the total treatment effect as:

$$
\text{Total Effect} = \delta' + (\theta \times \delta)
$$

where:

-   $\delta'$ = Direct effect (holding the mediator constant).

-   $\theta \times \delta$ = Indirect (mediated) effect.

------------------------------------------------------------------------

### Challenges in Mediation Analysis for DiD

Mediation in a DiD setting introduces several challenges that require careful consideration:

1.  **Potential Confounding of the Mediator**

-   A key assumption is that no unmeasured confounders affect both the mediator and the outcome.
-   If such confounders exist, estimates of $\theta$ may be biased.

2.  **Mediator-Outcome Endogeneity**

-   If the mediator is itself influenced by unobserved factors correlated with the outcome, it introduces endogeneity, making direct OLS estimates of $\theta$ problematic.
-   For example, in a crime policy evaluation:
    -   The number of police officers (mediator) may be influenced by crime rates (outcome), leading to reverse causality.

3.  **Interaction Between Multiple Mediators**

-   If there are multiple mediators (e.g., a policy that increases both police presence and surveillance cameras), they may interact with each other.
-   A useful test is to regress each mediator on treatment and other mediators. If a mediator predicts another, their effects are not independent, complicating interpretation.

------------------------------------------------------------------------

### Alternative Approach: Instrumental Variables for Mediation

One way to address mediator endogeneity is to use **instrumental variables**, where treatment serves as an instrument for the mediator:

**Two-Stage Estimation:**

1.  **First Stage: Predict the Mediator Using the Treatment** $$
    M_{it} = \alpha + \pi Treat_i + \lambda Post_t + \delta (Treat_i \times Post_t) + \nu_{it}
    $$
2.  **Second Stage: Predict the Outcome Using the Instrumented Mediator** $$
    Y_{it} = \alpha' + \gamma' Treat_i + \lambda' Post_t + \phi \hat{M}_{it} + \epsilon'_{it}
    $$

-   Here, $\hat{M}_{it}$ (predicted values from the first stage) replaces $M_{it}$, eliminating endogeneity concerns if the exclusion restriction holds (i.e., treatment only affects $Y$ through $M$).

**Key Limitation of IV Approach**

-   The IV strategy assumes that treatment affects the outcome only through the mediator, which may be too strong of an assumption in complex policy settings.

------------------------------------------------------------------------

## Assumptions

1.  **Parallel Trends Assumption**

-   The treatment and control groups must follow parallel trends in the absence of treatment.
-   Mathematically, this means the expected difference in potential outcomes remains constant over time:\
    $$E[Y_{it}(0) | D_i = 1] - E[Y_{it}(0) | D_i = 0] \text{ is constant over time.}$$\
-   This assumption is crucial, as violations lead to biased estimates.
-   Use DiD when:
    -   You have pre- and post-treatment data.
    -   You have clear treatment and control groups.
-   Avoid DiD when:
    -   Treatment assignment is not random.
    -   There are confounders affecting trends differently.
-   Testing Parallel Trends: [Prior Parallel Trends Test](#prior-parallel-trends-test).

2.  **No Anticipation Effect (Pre-Treatment Exogeneity)**

-   Individuals or groups should not change their behavior before the treatment is implemented in expectation of the treatment.

-   If units anticipate the treatment and adjust their behavior beforehand, it can introduce bias in the estimates.

3.  **Exogenous Treatment Assignment**

-   Treatment should not be assigned based on potential outcomes.
-   Ideally, assignment should be as good as random, conditional on observables.

4.  **Stable Composition of Groups (No Attrition or Spillover)**

-   Treatment and control groups should remain stable over time.
-   There should be no selective attrition (where individuals enter/leave due to treatment).
-   No spillover effects: Control units should not be indirectly affected by treatment.

5.  **No Simultaneous Confounding Events (Exogeneity of Shocks)**

-   There should be no other major shocks that affect treatment/control groups differently at the same time as treatment implementation.

------------------------------------------------------------------------

**Limitations and Common Issues**

1.  **Functional Form Dependence**

-   If the response to treatment is nonlinear, compare high- vs. low-intensity groups.

2.  **Selection on (Time-Varying) Unobservables**

-   Use [Rosenbaum Bounds] to check the sensitivity of estimates to unobserved confounders.

3.  **Long-Term Effects**

-   Parallel trends are more reliable in short time windows.
-   Over long periods, other confounding factors may emerge.

4.  **Heterogeneous Effects**

-   Treatment intensity (e.g., different doses) may vary across groups, leading to different effects.

5.  **Ashenfelter's Dip** [@ashenfelter1978estimating]

-   Participants in job training programs often experience earnings drops before enrolling, making them systematically different from nonparticipants.
-   **Fix**: Compute long-run differences, excluding periods around treatment, to test for sustained impact [@proserpio2017online; @heckman1999economics; @jepsen2014labor].

6.  **Lagged Treatment Effects**

-   If effects are not immediate, using a lagged dependent variable $Y_{it-1}$ may be more appropriate [@blundell1998initial].

7.  **Bias from Unobserved Factors Affecting Trends**

-   If external shocks influence treatment and control groups differently, this biases DiD estimates.

8.  **Correlated Observations**

-   Standard errors should be clustered appropriately.

9.  **Incidental Parameters Problem [@lancaster2000incidental]**

-   Always prefer individual and time fixed effects to reduce bias.

10. **Treatment Timing and Negative Weights**

-   If treatment timing varies across units, negative weights can arise in standard DiD estimators when treatment effects are heterogeneous [@athey2022design; @borusyak2024revisiting; @goodman2021difference].
-   **Fix:** Use estimators from @callaway2021difference and @de2020two (`did` package).
-   If expecting lags and leads, see @sun2021estimating.

11. **Treatment Effect Heterogeneity Across Groups**

-   If treatment effects vary across groups and interact with treatment variance, standard estimators may be invalid [@gibbons2018broken].

12. Endogenous Timing

If the timing of units can be influenced by strategic decisions in a DID analysis, an instrumental variable approach with a control function can be used to control for endogeneity in timing.

13. Questionable Counterfactuals

In situations where the control units may not serve as a reliable counterfactual for the treated units, matching methods such as propensity score matching or generalized random forest can be utilized. Additional methods can be found in [Matching Methods].

------------------------------------------------------------------------

### Prior Parallel Trends Test {#prior-parallel-trends-test}

The parallel trends assumption ensures that, absent treatment, the treated and control groups would have followed similar outcome trajectories. Testing this assumption involves visualization and statistical analysis.

@marcus2021role discuss pre-trend testing in staggered DiD.

1.  **Visual Inspection: Outcome Trends and Treatment Rollout**

-   Plot raw outcome trends for both groups before and after treatment.
-   Use **event-study plots** to check for pre-trend violations and anticipation effects.
-   Visualization tools like `ggplot2` or `panelView` help illustrate treatment timing and trends.

2.  **Event-Study Regressions**

A formal test for pre-trends uses the event-study model:

$$ Y_{it} = \alpha + \sum_{k=-K}^{K} \beta_k 1(T = k) + X_{it} \gamma + \lambda_i + \delta_t + \epsilon_{it} $$

where:

-   $1(T = k)$ are time dummies for periods before and after treatment.

-   $\beta_k$ captures deviations in outcomes before treatment; these should be **statistically indistinguishable from zero** if parallel trends hold.

-   $\lambda_i$ and $\delta_t$ are unit and time fixed effects.

-   $X_{it}$ are optional covariates.

**Violation of parallel trends** occurs if pre-treatment coefficients ($\beta_k$ for $k < 0$) are statistically significant.

3.  **Statistical Test for Pre-Treatment Trend Differences**

Using only pre-treatment data, estimate:

$$
Y = \alpha_g + \beta_1 T + \beta_2 (T \times G) + \epsilon
$$

where:

-   $\beta_2$ measures differences in time trends between groups.

-   If $\beta_2 = 0$, trends are parallel before treatment.

**Considerations:**

-   Alternative functional forms (e.g., polynomials or nonlinear trends) can be tested.
-   If $\beta_2 \neq 0$, potential explanations include:
    -   Large sample size driving statistical significance.
    -   Small deviations in one period disrupting an otherwise stable trend.

While time fixed effects can partially address violations of parallel trends (and are commonly used in modern research), they may also absorb part of the treatment effect, especially when treatment effects vary over time [@wolfers2003business].

------------------------------------------------------------------------

**Debate on Parallel Trends**

-   **Levels vs. Trends**: @kahn2020promise argue that **similarity in levels** is also crucial. If treatment and control groups start at **different levels**, why assume their trends will be the same?
    -   **Solution**:
        -   Plot time series for the treated and control groups.
        -   Use matched samples to improve comparability [@ryan2019now] (useful when parallel trends assumption is questionable).
    -   If levels differ significantly, functional form assumptions become more critical and must be justified.
-   **Power of Pre-Trend Tests**:
    -   Pre-trend tests often lack statistical power, making false negatives common [@roth2022pretest].
    -   See: [PretrendsPower](https://github.com/jonathandroth/PretrendsPower) and [pretrends](https://github.com/jonathandroth/pretrends) (for adjustments).
-   **Outcome Transformations Matter**:
    -   The parallel trends assumption is specific to both the transformation and units of the outcome variable [@roth2023parallel].
    -   Conduct falsification tests to check whether the assumption holds under different functional forms.

------------------------------------------------------------------------


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-43-1.png" width="90%" style="display: block; margin: auto;" />

``` r

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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-43-2.png" width="90%" style="display: block; margin: auto;" />

``` r


# but it's also important to use statistical test
prior_trend <- fixest::feols(Rate ~ i(Quarter_Num, California) |
                                 State + Quarter,
                             data = od)

fixest::coefplot(prior_trend, grid = F)
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-43-3.png" width="90%" style="display: block; margin: auto;" />

``` r
fixest::iplot(prior_trend, grid = F)
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-43-4.png" width="90%" style="display: block; margin: auto;" />

This is alarming since one of the periods is significantly different from 0, which means that our parallel trends assumption is not plausible.

In cases where the parallel trends assumption is questionable, researchers should consider methods for assessing and addressing potential violations. Some key approaches are discussed in @rambachan2023more:

-   **Imposing Restrictions**: Constrain how different the post-treatment violations of parallel trends can be relative to pre-treatment deviations.

-   **Partial Identification**: Rather than assuming a single causal effect, derive bounds on the ATT.

-   **Sensitivity Analysis**: Evaluate how sensitive the results are to potential deviations from parallel trends.

To implement these approaches, the `HonestDiD` package by @rambachan2023more provides robust statistical tools:


``` r
# https://github.com/asheshrambachan/HonestDiD
# remotes::install_github("asheshrambachan/HonestDiD")
# library(HonestDiD)
```

Alternatively, @ban2022generalized propose a method that incorporates pre-treatment covariates as an information set and makes an assumption about the selection bias in the post-treatment period. Specifically, they assume that the selection bias lies within the convex hull of all pre-treatment selection biases. Under this assumption:

-   They identify a set of possible ATT values.

-   With a stronger assumption on selection bias---grounded in policymakers' perspectives---they can estimate a point estimate of ATT.

Another useful tool for assessing parallel trends is the `pretrends` package by @roth2022pretest, which provides formal pre-trend tests:


``` r
# Install and load the pretrends package
# install.packages("pretrends")
# library(pretrends)

```

### Placebo Test {#sec-placebo-test-did}

A placebo test is a diagnostic tool used in Difference-in-Differences analysis to assess whether the estimated treatment effect is driven by pre-existing trends rather than the treatment itself. The idea is to estimate a treatment effect in a scenario where no actual treatment occurred. If a significant effect is found, it suggests that the parallel trends assumption may not hold, casting doubt on the validity of the causal inference.

**Types of Placebo DiD Tests**

1.  **Group-Based Placebo Test**

-   Assign treatment to a group that was never actually treated and rerun the DiD model.
-   If the estimated treatment effect is statistically significant, this suggests that differences between groups---not the treatment---are driving results.
-   This test helps rule out the possibility that the estimated effect is an artifact of unobserved systematic differences.

A valid treatment effect should be consistent across different reasonable control groups. To assess this:

-   Rerun the DiD model using an alternative but comparable control group.

-   Compare the estimated treatment effects across multiple control groups.

-   If results vary significantly, this suggests that the choice of control group may be influencing the estimated effect, indicating potential selection bias or unobserved confounding.

2.  **Time-Based Placebo Test**

-   Conduct DiD using only pre-treatment data, pretending that treatment occurred at an earlier period.
-   A significant estimated treatment effect implies that differences in pre-existing trends---not treatment---are responsible for observed post-treatment effects.
-   This test is particularly useful when concerns exist about unobserved shocks or anticipatory effects.

**Random Reassignment of Treatment**

-   Keep the same treatment and control periods but randomly assign treatment to units that were not actually treated.
-   If a significant DiD effect still emerges, it suggests the presence of biases, unobserved confounding, or systematic differences between groups that violate the parallel trends assumption.

------------------------------------------------------------------------

**Procedure for a Placebo Test**

1.  **Using Pre-Treatment Data Only**

A robust placebo test often involves analyzing only pre-treatment periods to check whether spurious treatment effects appear. The procedure includes:

-   Restricting the sample to pre-treatment periods only.

-   Assigning a fake treatment period before the actual intervention.

-   Testing a sequence of placebo cutoffs over time to examine whether different assumed treatment timings yield significant effects.

-   Generating random treatment periods and using randomization inference to assess the sampling distribution of the placebo effect.

-   Estimating the DiD model using the fake post-treatment period (`post_time = 1`).

-   **Interpretation**: If the estimated treatment effect is statistically significant, this indicates that pre-existing trends (not treatment) might be influencing results, violating the parallel trends assumption.

2.  **Using Control Groups for a Placebo Test**

If multiple control groups are available, a placebo test can also be conducted by:

-   Dropping the actual treated group from the analysis.

-   Assigning one of the control groups as a fake treated group.

-   Estimating the DiD model and checking whether a significant effect is detected.

-   **Interpretation**:

    -   If a placebo effect appears (i.e., the estimated treatment effect is significant), it suggests that even among control groups, systematic differences exist over time.

    -   However, this result is not necessarily disqualifying. Some methods, such as [Synthetic Control], explicitly model such differences while maintaining credibility.

------------------------------------------------------------------------


``` r
# Load necessary libraries
library(tidyverse)
library(fixest)
library(ggplot2)
library(causaldata)

# Load the dataset
od <- causaldata::organ_donations %>%
    # Use only pre-treatment data
    dplyr::filter(Quarter_Num <= 3) %>%
    
    # Create fake (placebo) treatment variables
    dplyr::mutate(
        FakeTreat1 = as.integer(State == 'California' &
                                    Quarter %in% c('Q12011', 'Q22011')),
        FakeTreat2 = as.integer(State == 'California' &
                                    Quarter == 'Q22011')
    )

# Estimate the placebo effects using fixed effects regression
clfe1 <- fixest::feols(Rate ~ FakeTreat1 | State + Quarter, data = od)
clfe2 <- fixest::feols(Rate ~ FakeTreat2 | State + Quarter, data = od)

# Display the regression results
fixest::etable(clfe1, clfe2)
#>                           clfe1            clfe2
#> Dependent Var.:            Rate             Rate
#>                                                 
#> FakeTreat1      0.0061 (0.0051)                 
#> FakeTreat2                      -0.0017 (0.0028)
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

# Extract coefficients and confidence intervals
coef_df <- tibble(
    Model = c("FakeTreat1", "FakeTreat2"),
    Estimate = c(coef(clfe1)["FakeTreat1"], coef(clfe2)["FakeTreat2"]),
    SE = c(summary(clfe1)$coeftable["FakeTreat1", "Std. Error"], 
           summary(clfe2)$coeftable["FakeTreat2", "Std. Error"]),
    Lower = Estimate - 1.96 * SE,
    Upper = Estimate + 1.96 * SE
)

# Plot the placebo effects
ggplot(coef_df, aes(x = Model, y = Estimate)) +
    geom_point(size = 3, color = "blue") +
    geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2, color = "blue") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    theme_minimal() +
    labs(
        title = "Placebo Treatment Effects",
        y = "Estimated Effect on Organ Donation Rate",
        x = "Placebo Treatment"
    )
```

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-46-1.png" width="90%" style="display: block; margin: auto;" />

We would like the "supposed" DiD to be insignificant.

## Robustness Checks

A well-executed Difference-in-Differences analysis requires **robustness checks** to verify the validity of estimated treatment effects and **best practices** to ensure methodological rigor.

------------------------------------------------------------------------

### Robustness Checks to Strengthen Causal Interpretation

Once the parallel trends assumption is assessed, additional **robustness tests** ensure that treatment effects are not driven by confounding factors or modeling choices.

1.  **Varying the Time Window**

-   Shorter time windows reduce exposure to long-term confounders but risk losing statistical power.
-   Longer time windows capture persistent effects but may introduce unrelated policy changes.
-   Solution: Estimate the DiD model across different time horizons and check if results are stable.

2.  **Higher-Order Polynomial Time Trends**

-   Standard DiD models assume a linear time trend.
-   If trends are nonlinear, this assumption may be too restrictive.
-   Solution: Introduce quadratic or cubic time trends and verify whether results hold.

3.  **Testing Alternative Dependent Variables**

-   The treatment should only affect the expected dependent variable.
-   A robustness check involves running the DiD on unrelated dependent variables.
-   If treatment effects appear where they should not, this signals a possible identification problem.

4.  **Triple-Difference (DDD) Strategy**

A Triple-Difference (DDD) model adds an additional comparison group to address remaining biases:

$$
\begin{aligned}
Y_{ijt} &= \alpha + \gamma Treat_{i} + \lambda Post_t + \theta Group_j + \delta_1 (Treat_i \times Post_t) \\
&+ \delta_2 (Treat_i \times Group_j) + \delta_3 (Post_t \times Group_j) \\
&+ \delta_4 (Treat_i \times Post_t \times Group_j) + \epsilon_{ijt}
\end{aligned}
$$

where:

-   $Group_j$ represents a subgroup within treatment/control (e.g., high- vs. low-intensity exposure).

-   $\delta_4$ captures the DDD effect, which removes residual biases present in the standard DiD model.

------------------------------------------------------------------------

### Best Practices for Reliable DiD Implementation

To improve the credibility and transparency of DiD estimates, researchers should adhere to the following best practices:

1.  **Documenting Treatment Cohorts**

-   Clearly report the number of treated and control units over time.
-   If treatment is staggered, adjust for different exposure durations.

2.  **Checking Covariate Balance & Overlap**

-   Verify whether the distribution of covariates is similar across treatment and control groups.
-   If treatment and control groups differ significantly, consider using matching methods.

3.  **Conducting Sensitivity Analyses for Parallel Trends**

-   Apply alternative weighting schemes (e.g., entropy balancing) to reduce dependence on model assumptions.
-   Use `honestDiD` to test robustness under different parallel trends violations.

## Concerns in DID

### Matching Methods in DID

Matching methods are often used in **causal inference** to balance treated and control units based on **pre-treatment observables**. In the context of Difference-in-Differences, matching helps:

-   Reduce selection bias by ensuring that treated and control units are comparable before treatment.
-   Improve parallel trends validity by selecting control units with similar pre-treatment trajectories.
-   Enhance robustness when treatment assignment is non-random across groups.

**Key Considerations in Matching**

-   **Standard Errors Need Adjustment**
    -   Standard errors should account for the fact that matching reduces variance [@heckman1997matching].
    -   A more robust alternative is Doubly Robust DID [@sant2020doubly], where either matching or regression suffices for unbiased treatment effect identification.
-   **Group Fixed Effects Alone Do Not Eliminate Selection Bias**
    -   Fixed effects absorb time-invariant heterogeneity, but do not correct for selection into treatment.
    -   Matching helps close the "backdoor path" between:
        1.  Propensity to be treated
        2.  Dynamics of outcome evolution post-treatment
-   **Matching on Time-Varying Covariates**
    -   Beware of regression to the mean: extreme pre-treatment outcomes may artificially bias post-treatment estimates [@daw2018matching].
    -   This issue is less concerning for time-invariant covariates.
-   **Comparing Matching vs. DID Performance**
    -   Matching and DID both use pre-treatment outcomes to mitigate selection bias.
    -   Simulations [@chabe2015analysis] show that:
        -   Matching tends to underestimate the true treatment effect but improves with more pre-treatment periods.
        -   When selection bias is symmetric, Symmetric DID (equal pre- and post-treatment periods) performs well.
        -   When selection bias is asymmetric, DID generally outperforms Matching.
-   **Forward DID as a Control Unit Selection Algorithm**
    -   An efficient way to select control units is Forward DID [@li2024frontiers].

------------------------------------------------------------------------

### Control Variables in DID

-   Always report results with and without controls:
    -   If controls are fixed within groups or time periods, they should be absorbed in fixed effects.
    -   If controls vary across both groups and time, this suggests the parallel trends assumption is questionable.
-   $R^2$ is not crucial in causal inference:
    -   Unlike predictive models, causal models do not prioritize explanatory power ($R^2$), but rather unbiased identification of treatment effects.

------------------------------------------------------------------------

### DID for Count Data: Fixed-Effects Poisson Model

For count data, one can use the fixed-effects Poisson pseudo-maximum likelihood estimator (PPML) [@athey2006identification; @puhani2012treatment]. Applications of this method can be found in management [@burtch2018can] and marketing [@he2021end].

This approach offers robust standard errors under over-dispersion [@wooldridge1999quasi] and is particularly useful when dealing with excess zeros in the data.

Key advantages of PPML:

-   Handles zero-inflated data better than log-OLS: A log-OLS regression may produce biased estimates [@o2010not] when heteroskedasticity is present [@silva2006log], especially in datasets with many zeros [@silva2011further].
-   Avoids the limitations of negative binomial fixed effects: Unlike Poisson, there is no widely accepted fixed-effects estimator for the negative binomial model [@allison20027].

------------------------------------------------------------------------

### Handling Zero-Valued Outcomes in DID

When dealing with **zero-valued outcomes**, it is crucial to separate the **intensive margin effect** (e.g., outcome changes from 10 to 11) from the **extensive margin effect** (e.g., outcome changes from 0 to 1).

A common issue is that the treatment coefficient from a **log-transformed regression** cannot be directly interpreted as a percentage change when zeros are present [@chen2023logs]. To address this, we can consider two alternative approaches:

1.  **Proportional Treatment Effects**

We define the percentage change in the treated group's post-treatment outcome as:

$$
\theta_{ATT\%} = \frac{E[Y_{it}(1) \mid D_i = 1, Post_t = 1] - E[Y_{it}(0) \mid D_i = 1, Post_t = 1]}{E[Y_{it}(0) \mid D_i = 1, Post_t = 1]}
$$

Instead of assuming parallel trends in levels, we can rely on a parallel trends assumption in ratios [@wooldridge2023simple].

The Poisson QMLE model is:

$$
Y_{it} = \exp(\beta_0 + \beta_1 D_i \times Post_t + \beta_2 D_i + \beta_3 Post_t + X_{it}) \epsilon_{it}
$$

The treatment effect is estimated as:

$$
\hat{\theta}_{ATT\%} = \exp(\hat{\beta}_1) - 1
$$

To validate the parallel trends in ratios assumption, we can estimate a dynamic Poisson QMLE model:

$$
Y_{it} = \exp(\lambda_t + \beta_2 D_i + \sum_{r \neq -1} \beta_r D_i \times (RelativeTime_t = r))
$$

If the assumption holds, we expect:

$$
\exp(\hat{\beta}_r) - 1 = 0 \quad \text{for} \quad r < 0.
$$

Even if the pre-treatment estimates appear close to zero, we should still conduct a sensitivity analysis [@rambachan2023more] to assess robustness (see [Prior Parallel Trends Test](#prior-parallel-trends-test)).


``` r
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


``` r
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

In this example, the DID coefficient is not significant. However, say that it's significant, we can interpret the coefficient as 3 percent increase in post-treatment period due to the treatment.


``` r
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

<img src="30-dif-in-dif_files/figure-html/estiamte teh proprortion treatment effects for event-study form-1.png" width="90%" style="display: block; margin: auto;" />

This parallel trend is the "ratio" version as in @wooldridge2023simple :

$$
\frac{E(Y_{it}(0) |D_i = 1, Post_t = 1)}{E(Y_{it}(0) |D_i = 1, Post_t = 0)} = \frac{E(Y_{it}(0) |D_i = 0, Post_t = 1)}{E(Y_{it}(0) |D_i =0, Post_t = 0)}
$$

which means without treatment, the average percentage change in the mean outcome for treated group is identical to that of the control group.

2.  **Log Effects with Calibrated Extensive-Margin Value**

A potential limitation of proportional treatment effects is that they may not be well-suited for heavy-tailed outcomes. In such cases, we may prefer to explicitly model the extensive margin effect.

Following [@chen2023logs, p. 39], we can calibrate the weight placed on the intensive vs. extensive margin to ensure meaningful interpretation of the treatment effect.

If we want to study the treatment effect on a concave transformation of the outcome that is less influenced by those in the distribution's tail, then we can perform this analysis.

Steps:

1.  Normalize the outcomes such that 1 represents the minimum non-zero and positive value (i.e., divide the outcome by its minimum non-zero and positive value).
2.  Estimate the treatment effects for the new outcome

$$
m(y) =
\begin{cases}
\log(y) & \text{for } y >0 \\
-x & \text{for } y = 0
\end{cases}
$$

The choice of $x$ depends on what the researcher is interested in:

| Value of $x$ | Interest                                                                                                                                                                        |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $x = 0$      | The treatment effect in logs where all zero-valued outcomes are set to equal the minimum non-zero value (i.e., we exclude the extensive-margin change between 0 and $y_{min}$ ) |
| $x>0$        | Setting the change between 0 and $y_{min}$ to be valued as the equivalent of a $x$ log point change along the intensive margin.                                                 |


``` r
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

<img src="30-dif-in-dif_files/figure-html/unnamed-chunk-47-1.png" width="90%" style="display: block; margin: auto;" />

``` r


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

------------------------------------------------------------------------

### Standard Errors

One of the major statistical challenges in DiD estimation is serial correlation in the error terms. This issue is particularly problematic because it can lead to underestimated standard errors, inflating the likelihood of Type I errors (false positives). As discussed in @bertrand2004much, serial correlation arises in DiD settings due to several factors:

1.  **Long time series**: Many DiD studies use multiple time periods, increasing the risk of correlated errors.
2.  **Highly positively serially correlated outcomes**: Many economic and business variables (e.g., GDP, sales, employment rates) exhibit strong persistence over time.
3.  **Minimal within-group variation in treatment timing**: For example, in a state-level policy change, all individuals in a state receive treatment at the same time, leading to correlation within state-time clusters.

To correct for serial correlation, various methods can be employed. However, some approaches work better than others:

-   **Avoid standard parametric corrections**: A common approach is to model the error term using an autoregressive (AR) process. However, @bertrand2004much show that this often fails in DiD settings because it does not fully account for within-group correlation.
-   **Nonparametric solutions (preferred when the number of groups is large)**:
    -   **Block bootstrap**: Resampling entire groups (e.g., states) rather than individual observations maintains the correlation structure and provides robust standard errors.
-   **Collapsing data into two periods (Pre vs. Post)**:
    -   Aggregating the data into a single pre-treatment and single post-treatment period can mitigate serial correlation issues. This approach is particularly useful when the number of groups is small [@donald2007inference].
    -   Note: While this reduces the power of the analysis by discarding variation across time, it ensures that standard errors are not artificially deflated.
-   **Variance-covariance matrix corrections**:
    -   Empirical corrections (e.g., cluster-robust standard errors) and arbitrary variance-covariance matrix adjustments (e.g., Newey-West) can work well, but they are reliable only in large samples.

Overall, selecting the appropriate correction method depends on the sample size and structure of the data. When possible, block bootstrapping and collapsing data into pre/post periods are among the most effective approaches.
