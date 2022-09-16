# Event Studies

The earliest paper that used event study was [@dolley1933characteristics]

[@campbell1997] introduced this method, which based on the efficient markets theory by [@fama1970]

Previous marketing studies:

-   [@horsky1987]: name change

-   [@chaney1991] new product announcements

-   [@agrawal1995]: celebrity endorsement

-   [@lane1995]: brand extensions

-   [@johnson2000a]: joint venture

-   [@geyskens2002]: Internet channel (for newspapers)

-   [@wiles2010]: Regulatory Reports of Deceptive Advertising

-   [@sood2009]: innovation payoff

Potential avenues:

-   Ad campaigns

-   Market entry

-   product failure/recalls

-   Patents

Pros:

-   Better than accounting based measures (e.g., profits) because managers can manipulate profits [@benston1985validity]

-   Easy to do

Events can be

-   Internal (e.g., stock repurchase)

-   External (e.g., macroeconomic variables)

**Assumptions**:

1.  Efficient market theory
2.  Shareholders are the most important group among stakeholders
3.  The event sharply affects share price
4.  Expected return is calculated appropriately

**Steps**:

1.  Event Identification: (e.g., dividends, M&A, stock buyback, laws or regulation, privatization vs. nationalization, celebrity endorsements, name changes, or brand extensions etc.)
    1.  Estimation window: Normal return expected return ($T_0 \to T_1$) (sometimes include days before to capture leakages).

        -   Recommendation by [@moorman2004assessing, p. 13] is to use 250 days before the event (and 45-day between the estimation window and the event window).

        -   Similarly, [@mcwilliams1997a] and [@fornell2006a] 255 days ending 46 days before the event date

    2.  Event window: contain the event date ($T_1 \to T_2$) (have to argue for the event window and can't do it empirically)

    3.  Post Event window: $T_2 \to T_3$
2.  Normal vs. Abnormal returns

$$
\epsilon_{it}^* = R_{it} - E(R_{it}|X_t)
$$

where

-   $\epsilon_{it}^*$ = abnormal return

-   $R_{it}$ = realized (actual) return

-   $E(R_{it}|X_t)$ normal expected return

There are several model to calculate the expected return

A. Statistical Models: assumes jointly multivariate normal and iid over time (need distributional assumptions for valid finite-sample estimation) rather robust (hence, recommended)

1.  Constant Mean Return Model
2.  Market Model
3.  Adjusted Market Return Model
4.  Factor Model

B. Economic Models (strong assumption regarding investor behavior)

1.  Capital Asset Pricing Model
2.  Arbitrage pricing theory

## Other Issues

### Economic significance

Total wealth gain (loss) from the event

$$
\Delta W_t = CAR_t \times MKTVAL_0
$$

where

-   $\Delta W_t$ = gain (loss)

-   $CAR_t$ = cumulative residuals to date $t$

-   $MKTVAL_0$ market value of the firm before the event window

### Statistical Power

increases with

-   more firms

-   less days in the event window (avoiding potential contamination from confounds)

### Testing

#### Parametric Test

Applying CLT

$$
t_{CAR} = \frac{\bar{CAR_{it}}}{\sigma (CAR_{it})/\sqrt{n}} \\
t_{BHAR} = \frac{\bar{BHAR_{it}}}{\sigma (BHAR_{it})/\sqrt{n}}
$$

#### Non-parametric Test

-   No assumptions about return distribution

-   Sign Test (assumes symmetry in returns)

-   Rank Test (allows for non-symmetry in returns)

### Confounders

According to [@fornell2006a], need to control:

-   one-day event period = day when Wall Street Journal publish ACSI announcement.

-   5 days before and after event to rule out other news (PR Newswires, Dow Jones, Business Wires)

    -   M&A, Spin-offs, stock splits

    -   CEO or CFO changes,

    -   Layoffs, restructurings, earnings announcements, lawsuits

    -   Capital IQ - Key Developments: covers almost all important events so you don't have to search on news.

### Biases

-   Different closing time obscure estimation of the abnormal returns, check [@campbell1998econometrics]

-   Upward bias in aggregating CAR + transaction prices (bid and ask)

-   Cross-sectional dependence in the returns bias the standard deviation estimates downward, which inflates the test statistics when events share common dates [@mackinlay1997event]. Hence, [@jaffe1974] [Portfolio method] should be used to correct for this bias.

### Long-run event studies

-   12 - 60 months event window: [@loughran1995] [@brav1997] [@desai1999]

-   Exemplar: [@dutta2018]


```r
library(crseEventStudy)

# example by the package's author
data(demo_returns)
SAR <-
    sar(event = demo_returns$EON,
        control = demo_returns$RWE,
        logret = FALSE)
mean(SAR)
#> [1] 0.006870196
```

#### Buy and Hold Abnormal Returns (BHAR)

-   Classic references: [@loughran1995] [@barber1997] [@lyon1999]

$$
AR_{it} = R_{it} - E(R_{it}|X_t)
$$

$$
BHAR_{t \to t + K}^i = \Pi_k(1 + AR_{i (t+k)})
$$

Where as CAR is the arithmetic sum, BHAR is the geometric sum

#### Portfolio method {data-link="Portfolio method"}

This section follows strictly the procedure in [@wiles2010]

A portfolio for every day in calendar time (including all securities which experience an event that time).

For each portfolio, the securities and their returns are equally weighted

1.  For all portfolios, the average abnormal return are calculated as

$$
AAR_{Pt} = \frac{\sum_{i=1}^S AR_i}{S}
$$

where

-   $S$ is the number of securities in portfolio $P$
-   $AR_i$ is the abnormal return for the stock $i$ in the portfolio

2.  For every portfolio $P$, a time series estimate of $\sigma(AAR_{Pt})$ is calculated for the preceding $k$ days, assuming that the $AAR_{Pt}$ are independent over time.
3.  Each portfolio's average abnormal return is standardized

$$
SAAR_{Pt} = \frac{AAR_{Pt}}{SD(AAR_{Pt})}
$$

4.  Average standardized residual across all portfolio's in calendar time

$$
ASAAR = \frac{1}{n}\sum_{i=1}^{255} SAAR_{Pt} \times D_t
$$

where

-   $D_t = 1$ when there is at least one security in portfolio $t$

-   $D_t = 0$ when there are no security in portfolio $t$

-   $n$ is the number of days in which the portfolio have at least one security $n = \sum_{i = 1}^{255}D_t$

5.  The cumulative average standardized average abnormal returns is

$$
CASSAR_{S_1, S_2} = \sum_{i=S_1}^{S_2} ASAAR
$$

If the ASAAR are independent over time, then standard deviation for the above estimate is $\sqrt{S_2 - S_1 + 1}$

then, the test statistics is

$$
t = \frac{CASAAR_{S_1,S_2}}{\sqrt{S_2 - S_1 + 1}}
$$

## Aggregation

### Over Time

We calculate the cumulative abnormal (CAR) for the event windows

$H_0$: Standardized cumulative abnormal return for stock $i$ is 0 (no effect of events on stock performance)

$H_1$: SCAR is not 0 (there is an effect of events on stock performance)

### Across Firms + Over Time

Additional assumptions: Abnormal returns of different socks are uncorrelated (rather strong), but it's very valid if event windows for different stocks do not overlap. If the windows for different overlap, follow [@bernard1987] and [@schipper1983]

$H_0$: The mean of the abnormal returns across all firms is 0 (no effect)

$H_1$: The mean of the abnormal returns across all firms is different form 0 (there is an effect)

Parametric (empirically either one works fine) (assume abnormal returns is normally distributed) :

1.  Aggregate the CAR of all stocks (Use this if the true abnormal variance is greater for stocks with higher variance)
2.  Aggregate the SCAR of all stocks (Use this if the true abnormal return is constant across all stocks)

Non-parametric (no parametric assumptions):

1.  Sign test:
    -   Assume both the abnormal returns and CAR to be independent across stocks

    -   Assume 50% with positive abnormal returns and 50% with negative abnormal return

    -   The null will be that there is a positive abnormal return correlated with the event (if you want the alternative to be there is a negative relationship)

    -   With skewed distribution (likely in daily stock data), the size test is not trustworthy. Hence, rank test might be better
2.  Rank test
    -   Null: there is no abnormal return during the event window

## Heterogeneity in the event effect

$$
y = X \theta + \eta
$$

where

-   $y$ = CAR

-   $X$ = Characteristics that lead to heterogeneity in the event effect (i.e., abnormal returns) (e.g., firm or event specific)

-   $\eta$ = error term

Note:

-   In cases with selection bias (firm characteristics and investor anticipation of the event: larger firms might enjoy great positive effect of an event, and investors endogenously anticipate this effect and overvalue the stock), we have to use the White's $t$-statistics to have the lower bounds of the true significance of the estimates.

## Expected Return Calculation

### Statistical Models

-   based on statistical assumptions about the behavior of returns (e..g, multivariate normality)

-   we only need to assume stable distributions [@owen1983]

#### Constant Mean Return Model

The expected normal return is the mean of the real returns

$$
Ra_{it} = R_{it} - \bar{R}_i
$$

Assumption:

-   returns revert to its mean (very questionable)

The basic mean returns model generally delivers similar findings to more complex models since the variance of abnormal returns is not decreased considerably [@brown1985]

#### Market Model

$$
R_{it} = \alpha_i + \beta R_{mt} + \epsilon_{it}
$$

where

-   $R_{it}$ = stock return $i$ in period $t$

-   $R_{mt}$ = market return

-   $\epsilon_{it}$ = zero mean ($E(e_{it}) = 0$) error term with its own variance $\sigma^2$

Notes:

-   People typically use S&P 500, CRSP value-weighed or equal-weighted index as the market portfolio.

-   When $\beta =0$, the [Market Model] is the [Constant Mean Return Model]

-   better fit of the market-model, the less variance in abnormal return, and the more easy to detect the event's effect

-   recommend generalized method of moments to be robust against auto-correlation and heteroskedasticity

#### FF3

[@fama1993]

$$
E(R_{it}|X_t) - r_{ft} = \alpha_i + \beta_{1i} (E(R_{mt}|X_t )- r_{ft}) \\
+ b_{2i} SML_t + b_{3i} HML_t
$$

where

-   $r_{ft}$ risk-free rate (e.g., 3-month Treasury bill)

-   $R_{mt}$ is the market-rate (e.g., S&P 500)

-   SML: returns on small (size) portfolio minus returns on big portfolio

-   HML: returns on high (B/M) portfolio minus returns on low portfolio.

### Economic Model

The only difference difference between CAPM and APT is that APT has multiple factors (including factors beyond the focal company)

Economic models put limits on a statistical model that come from assumed behavior that is derived from theory.

#### Capital Asset Pricing Model (CAPM)

$$
E(R_i) = R_f + \beta_i (E(R_m) - R_f)
$$

where

-   $E(R_i)$ = expected firm return

-   $R_f$ = risk free rate

-   $E(R_m - R_f)$ = market risk premium

-   $\beta_i$ = firm sensitivity

#### Arbitrage Pricing Theory (APT)

$$
R = R_f + \Lambda f + \epsilon
$$

where

-   $\epsilon \sim N(0, \Psi)$

-   $\Lambda$ = factor loadigns

-   $f \sim N(\mu, \Omega)$ = general factor model

    -   $\mu$ = expected risk premium vector

    -   $\Omega$ = factor covariance matrix

## Application

Packages:

-   `eventstudies`

-   `erer`

-   `EventStudy`

-   `AbnormalReturns`

-   [Event Study Tools](https://www.eventstudytools.com/)

-   [estudy2](https://irudnyts.github.io/estudy2/)

-   `PerformanceAnalytics`

In practice, people usually sort portfolio because they are not sure whether the FF model is specified correctly.

Steps:

1.  Sort all returns in CRSP into 10 deciles based on size.
2.  In each decile, sort returns into 10 decides based on BM
3.  Get the average return of the 100 portfolios for each period (i.e., expected returns of stocks given decile - characteristics)
4.  For each stock in the event study: Compare the return of the stock to the corresponding portfolio based on size and BM.

Notes:

-   Sorting produces outcomes that are often more conservative (e.g., FF abnormal returns can be greater than those that used sorting).

-   If the results change when we do B/M first then size or vice versa, then the results are not robust (this extends to more than just two characteristics - e.g., momentum).

-   

Examples:

Forestry:

-   [@mei2008] M&A on financial performance (forest product)

-   [@sun2011] litigation on firm values


```r
library(erer)

# example by the package's author
data(daEsa)
hh <- evReturn(
    y = daEsa,       # dataset
    firm = "wpp",    # firm name
    y.date = "date", # date in y 
    index = "sp500", # index
    est.win = 250,   # estimation window wedith in days
    digits = 3, 
    event.date = 19990505, # firm event dates 
    event.win = 5          # one-side event window wdith in days (default = 3, where 3 before + 1 event date + 3 days after = 7 days)
)
hh; plot(hh)
#> 
#> === Regression coefficients by firm =========
#>   N firm event.date alpha.c alpha.e alpha.t alpha.p alpha.s beta.c beta.e
#> 1 1  wpp   19990505  -0.135   0.170  -0.795   0.428          0.665  0.123
#>   beta.t beta.p beta.s
#> 1  5.419  0.000    ***
#> 
#> === Abnormal returns by date ================
#>    day Ait.wpp    HNt
#> 1   -5   4.564  4.564
#> 2   -4   0.534  5.098
#> 3   -3  -1.707  3.391
#> 4   -2   2.582  5.973
#> 5   -1  -0.942  5.031
#> 6    0  -3.247  1.784
#> 7    1  -0.646  1.138
#> 8    2  -2.071 -0.933
#> 9    3   0.368 -0.565
#> 10   4   4.141  3.576
#> 11   5   0.861  4.437
#> 
#> === Average abnormal returns across firms ===
#>      name estimate error t.value p.value sig
#> 1 CiT.wpp    4.437 8.888   0.499   0.618    
#> 2     GNT    4.437 8.888   0.499   0.618
```

<img src="28-event-study_files/figure-html/unnamed-chunk-2-1.png" width="90%" style="display: block; margin: auto;" />

Example by [Ana Julia Akaishi Padula, Pedro Albuquerque (posted on LAMFO)](<https://lamfo-unb.github.io/2017/08/17/Teste-de-Eventos-en/#>:\~:text=The%20abnormal%20return%20(Ra,regression%20in%20the%20estimation%20window.)


Example in `AbnormalReturns` package


