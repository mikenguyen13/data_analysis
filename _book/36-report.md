# Report

Structure

-   Exploratory analysis

    -   plots
    -   preliminary results
    -   interesting structure/features in the data
    -   outliers

-   Model

    -   Assumptions
    -   Why this model/ How is this model the best one?
    -   Consideration: interactions, collinearity, dependence

-   Model Fit

    -   How well does it fit?

    -   Are the model assumptions met?

        -   Residual analysis

-   Inference/ Prediction

    -   Are there different way to support your inference?

-   Conclusion

    -   Recommendation

    -   Limitation of the analysis

    -   How to correct those in the future

This chapter is based on the `jtools` package. More information can be found [here.](https://www.rdocumentation.org/packages/jtools/versions/2.1.0)

## One summary table

Packages for reporting:

Summary Statistics Table:

-   [qwraps2](https://cran.r-project.org/web/packages/qwraps2/vignettes/summary-statistics.html)
-   [vtable](https://cran.r-project.org/web/packages/vtable/vignettes/sumtable.html)
-   [gtsummary](http://www.danieldsjoberg.com/gtsummary/)
-   [apaTables](https://cran.r-project.org/web/packages/apaTables/apaTables.pdf)
-   [stargazer](https://cran.r-project.org/web/packages/stargazer/vignettes/stargazer.pdf)

Regression Table

-   [gtsummary](http://www.danieldsjoberg.com/gtsummary/)
-   [sjPlot,sjmisc, sjlabelled](https://cran.r-project.org/web/packages/sjPlot/vignettes/tab_model_estimates.html)
-   [stargazer](https://cran.r-project.org/web/packages/stargazer/vignettes/stargazer.pdf): recommended ([Example](https://www.jakeruss.com/cheatsheets/stargazer/))
-   [modelsummary](https://github.com/vincentarelbundock/modelsummary#a-simple-example)


```r
library(jtools)
data(movies)
fit <- lm(metascore ~ budget + us_gross + year, data = movies)
summ(fit)
```

\begin{table}[!h]
\centering
\begin{tabular}{lr}
\toprule
\cellcolor{gray!6}{Observations} & \cellcolor{gray!6}{831 (10 missing obs. deleted)}\\
Dependent variable & metascore\\
\cellcolor{gray!6}{Type} & \cellcolor{gray!6}{OLS linear regression}\\
\bottomrule
\end{tabular}
\end{table} \begin{table}[!h]
\centering
\begin{tabular}{lr}
\toprule
\cellcolor{gray!6}{F(3,827)} & \cellcolor{gray!6}{26.23}\\
R² & 0.09\\
\cellcolor{gray!6}{Adj. R²} & \cellcolor{gray!6}{0.08}\\
\bottomrule
\end{tabular}
\end{table} \begin{table}[!h]
\centering
\begin{threeparttable}
\begin{tabular}{lrrrr}
\toprule
  & Est. & S.E. & t val. & p\\
\midrule
\cellcolor{gray!6}{(Intercept)} & \cellcolor{gray!6}{52.06} & \cellcolor{gray!6}{139.67} & \cellcolor{gray!6}{0.37} & \cellcolor{gray!6}{0.71}\\
budget & -0.00 & 0.00 & -5.89 & 0.00\\
\cellcolor{gray!6}{us\_gross} & \cellcolor{gray!6}{0.00} & \cellcolor{gray!6}{0.00} & \cellcolor{gray!6}{7.61} & \cellcolor{gray!6}{0.00}\\
year & 0.01 & 0.07 & 0.08 & 0.94\\
\bottomrule
\end{tabular}
\begin{tablenotes}
\item Standard errors: OLS
\end{tablenotes}
\end{threeparttable}
\end{table}

```r
summ(
    fit,
    scale = TRUE,
    vifs = TRUE,
    part.corr = TRUE,
    confint = TRUE,
    pvals = FALSE
) # notice that scale here is TRUE
```

\begin{table}[!h]
\centering
\begin{tabular}{lr}
\toprule
\cellcolor{gray!6}{Observations} & \cellcolor{gray!6}{831 (10 missing obs. deleted)}\\
Dependent variable & metascore\\
\cellcolor{gray!6}{Type} & \cellcolor{gray!6}{OLS linear regression}\\
\bottomrule
\end{tabular}
\end{table} \begin{table}[!h]
\centering
\begin{tabular}{lr}
\toprule
\cellcolor{gray!6}{F(3,827)} & \cellcolor{gray!6}{26.23}\\
R² & 0.09\\
\cellcolor{gray!6}{Adj. R²} & \cellcolor{gray!6}{0.08}\\
\bottomrule
\end{tabular}
\end{table} \begin{table}[!h]
\centering
\begin{threeparttable}
\begin{tabular}{lrrrrrrr}
\toprule
  & Est. & 2.5\% & 97.5\% & t val. & VIF & partial.r & part.r\\
\midrule
\cellcolor{gray!6}{(Intercept)} & \cellcolor{gray!6}{63.01} & \cellcolor{gray!6}{61.91} & \cellcolor{gray!6}{64.11} & \cellcolor{gray!6}{112.23} & \cellcolor{gray!6}{NA} & \cellcolor{gray!6}{NA} & \cellcolor{gray!6}{NA}\\
budget & -3.78 & -5.05 & -2.52 & -5.89 & 1.31 & -0.20 & -0.20\\
\cellcolor{gray!6}{us\_gross} & \cellcolor{gray!6}{5.28} & \cellcolor{gray!6}{3.92} & \cellcolor{gray!6}{6.64} & \cellcolor{gray!6}{7.61} & \cellcolor{gray!6}{1.52} & \cellcolor{gray!6}{0.26} & \cellcolor{gray!6}{0.25}\\
year & 0.05 & -1.18 & 1.28 & 0.08 & 1.24 & 0.00 & 0.00\\
\bottomrule
\end{tabular}
\begin{tablenotes}
\item Standard errors: OLS; Continuous predictors are mean-centered and scaled by 1 s.d. The outcome variable remains in its original units.
\end{tablenotes}
\end{threeparttable}
\end{table}

```r

#obtain clsuter-robust SE
data("PetersenCL", package = "sandwich")
fit2 <- lm(y ~ x, data = PetersenCL)
summ(fit2, robust = "HC3", cluster = "firm") 
```

\begin{table}[!h]
\centering
\begin{tabular}{lr}
\toprule
\cellcolor{gray!6}{Observations} & \cellcolor{gray!6}{5000}\\
Dependent variable & y\\
\cellcolor{gray!6}{Type} & \cellcolor{gray!6}{OLS linear regression}\\
\bottomrule
\end{tabular}
\end{table} \begin{table}[!h]
\centering
\begin{tabular}{lr}
\toprule
\cellcolor{gray!6}{F(1,4998)} & \cellcolor{gray!6}{1310.74}\\
R² & 0.21\\
\cellcolor{gray!6}{Adj. R²} & \cellcolor{gray!6}{0.21}\\
\bottomrule
\end{tabular}
\end{table} \begin{table}[!h]
\centering
\begin{threeparttable}
\begin{tabular}{lrrrr}
\toprule
  & Est. & S.E. & t val. & p\\
\midrule
\cellcolor{gray!6}{(Intercept)} & \cellcolor{gray!6}{0.03} & \cellcolor{gray!6}{0.07} & \cellcolor{gray!6}{0.44} & \cellcolor{gray!6}{0.66}\\
x & 1.03 & 0.05 & 20.36 & 0.00\\
\bottomrule
\end{tabular}
\begin{tablenotes}
\item Standard errors: Cluster-robust, type = HC3
\end{tablenotes}
\end{threeparttable}
\end{table}

Model to Equation


```r
# install.packages("equatiomatic") # not available for R 4.2
fit <- lm(metascore ~ budget + us_gross + year, data = movies)
# show the theoretical model
equatiomatic::extract_eq(fit)
# display the actual coefficients
equatiomatic::extract_eq(fit, use_coefs = TRUE)
```

## Model Comparison


```r
fit <- lm(metascore ~ log(budget), data = movies)
fit_b <- lm(metascore ~ log(budget) + log(us_gross), data = movies)
fit_c <- lm(metascore ~ log(budget) + log(us_gross) + runtime, data = movies)
coef_names <- c("Budget" = "log(budget)", "US Gross" = "log(us_gross)",
                "Runtime (Hours)" = "runtime", "Constant" = "(Intercept)")
export_summs(fit, fit_b, fit_c, robust = "HC3", coefs = coef_names)
```



```{=latex}
 
  \providecommand{\huxb}[2]{\arrayrulecolor[RGB]{#1}\global\arrayrulewidth=#2pt}
  \providecommand{\huxvb}[2]{\color[RGB]{#1}\vrule width #2pt}
  \providecommand{\huxtpad}[1]{\rule{0pt}{#1}}
  \providecommand{\huxbpad}[1]{\rule[-#1]{0pt}{#1}}

\begin{table}[ht]
\begin{centerbox}
\begin{threeparttable}
\captionsetup{justification=centering,singlelinecheck=off}
\caption{(\#tab:unnamed-chunk-3) }
 \setlength{\tabcolsep}{0pt}
\begin{tabular}{l l l l}


\hhline{>{\huxb{0, 0, 0}{0.8}}->{\huxb{0, 0, 0}{0.8}}->{\huxb{0, 0, 0}{0.8}}->{\huxb{0, 0, 0}{0.8}}-}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}c!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\centering \hspace{6pt}  \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{c!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\centering \hspace{6pt} Model 1 \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{c!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\centering \hspace{6pt} Model 2 \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{c!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\centering \hspace{6pt} Model 3 \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{>{\huxb{255, 255, 255}{0.4}}->{\huxb{0, 0, 0}{0.4}}->{\huxb{0, 0, 0}{0.4}}->{\huxb{0, 0, 0}{0.4}}-}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt} Budget \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} -2.43 *** \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} -5.16 *** \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} -6.70 *** \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt}  \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} (0.44)\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} (0.62)\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} (0.67)\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt} US Gross \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} \hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 3.96 *** \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 3.85 *** \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt}  \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} \hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} (0.51)\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} (0.48)\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt} Runtime (Hours) \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} \hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} \hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 14.29 *** \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt}  \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} \hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} \hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} (1.63)\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt} Constant \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 105.29 *** \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 81.84 *** \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 83.35 *** \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt}  \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} (7.65)\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} (8.66)\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} (8.82)\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{>{\huxb{255, 255, 255}{0.4}}->{\huxb{0, 0, 0}{0.4}}->{\huxb{0, 0, 0}{0.4}}->{\huxb{0, 0, 0}{0.4}}-}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt} N \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 831\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 831\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 831\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{}
\arrayrulecolor{black}

\multicolumn{1}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt} R2 \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 0.03\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 0.09\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} &
\multicolumn{1}{r!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedleft \hspace{6pt} 0.17\hphantom{0}\hphantom{0}\hphantom{0}\hphantom{0} \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{>{\huxb{0, 0, 0}{0.8}}->{\huxb{0, 0, 0}{0.8}}->{\huxb{0, 0, 0}{0.8}}->{\huxb{0, 0, 0}{0.8}}-}
\arrayrulecolor{black}

\multicolumn{4}{!{\huxvb{0, 0, 0}{0}}l!{\huxvb{0, 0, 0}{0}}}{\huxtpad{6pt + 1em}\raggedright \hspace{6pt} Standard errors are heteroskedasticity robust.  *** p $<$ 0.001;  ** p $<$ 0.01;  * p $<$ 0.05. \hspace{6pt}\huxbpad{6pt}} \tabularnewline[-0.5pt]


\hhline{}
\arrayrulecolor{black}
\end{tabular}
\end{threeparttable}\par\end{centerbox}

\end{table}
 
```



Another package is `modelsummary`


```r
library(modelsummary)
lm_mod <- lm(mpg ~ wt + hp + cyl, mtcars)
msummary(lm_mod, vcov = c("iid","robust","HC4"))
```

\begin{table}
\centering
\begin{tabular}[t]{lccc}
\toprule
  & (1) & (2) & (3)\\
\midrule
(Intercept) & \num{38.752} & \num{38.752} & \num{38.752}\\
 & (\num{1.787}) & (\num{2.286}) & (\num{2.177})\\
wt & \num{-3.167} & \num{-3.167} & \num{-3.167}\\
 & (\num{0.741}) & (\num{0.833}) & (\num{0.819})\\
hp & \num{-0.018} & \num{-0.018} & \num{-0.018}\\
 & (\num{0.012}) & (\num{0.010}) & (\num{0.013})\\
cyl & \num{-0.942} & \num{-0.942} & \num{-0.942}\\
 & (\num{0.551}) & (\num{0.573}) & (\num{0.572})\\
\midrule
Num.Obs. & \num{32} & \num{32} & \num{32}\\
R2 & \num{0.843} & \num{0.843} & \num{0.843}\\
R2 Adj. & \num{0.826} & \num{0.826} & \num{0.826}\\
AIC & \num{155.5} & \num{155.5} & \num{155.5}\\
BIC & \num{162.8} & \num{162.8} & \num{162.8}\\
Log.Lik. & \num{-72.738} & \num{-72.738} & \num{-72.738}\\
F & \num{50.171} & \num{31.065} & \num{32.623}\\
RMSE & \num{2.35} & \num{2.35} & \num{2.35}\\
Std.Errors & IID & HC3 & HC4\\
\bottomrule
\end{tabular}
\end{table}



```r
modelplot(lm_mod, vcov = c("iid","robust","HC4"))
```



\begin{center}\includegraphics[width=0.9\linewidth]{36-report_files/figure-latex/unnamed-chunk-4-1} \end{center}

Another package is `stargazer`


```r
library("stargazer")
stargazer(attitude)
#> 
#> % Table created by stargazer v.5.2.3 by Marek Hlavac, Social Policy Institute. E-mail: marek.hlavac at gmail.com
#> % Date and time: Fri, Jan 12, 2024 - 5:16:27 PM
#> \begin{table}[!htbp] \centering 
#>   \caption{} 
#>   \label{} 
#> \begin{tabular}{@{\extracolsep{5pt}}lccccc} 
#> \\[-1.8ex]\hline 
#> \hline \\[-1.8ex] 
#> Statistic & \multicolumn{1}{c}{N} & \multicolumn{1}{c}{Mean} & \multicolumn{1}{c}{St. Dev.} & \multicolumn{1}{c}{Min} & \multicolumn{1}{c}{Max} \\ 
#> \hline \\[-1.8ex] 
#> rating & 30 & 64.633 & 12.173 & 40 & 85 \\ 
#> complaints & 30 & 66.600 & 13.315 & 37 & 90 \\ 
#> privileges & 30 & 53.133 & 12.235 & 30 & 83 \\ 
#> learning & 30 & 56.367 & 11.737 & 34 & 75 \\ 
#> raises & 30 & 64.633 & 10.397 & 43 & 88 \\ 
#> critical & 30 & 74.767 & 9.895 & 49 & 92 \\ 
#> advance & 30 & 42.933 & 10.289 & 25 & 72 \\ 
#> \hline \\[-1.8ex] 
#> \end{tabular} 
#> \end{table}
## 2 OLS models
linear.1 <-
    lm(rating ~ complaints + privileges + learning + raises + critical,
       data = attitude)
linear.2 <-
    lm(rating ~ complaints + privileges + learning, data = attitude)
## create an indicator dependent variable, and run a probit model
attitude$high.rating <- (attitude$rating > 70)

probit.model <-
    glm(
        high.rating ~ learning + critical + advance,
        data = attitude,
        family = binomial(link = "probit")
    )
stargazer(linear.1,
          linear.2,
          probit.model,
          title = "Results",
          align = TRUE)
#> 
#> % Table created by stargazer v.5.2.3 by Marek Hlavac, Social Policy Institute. E-mail: marek.hlavac at gmail.com
#> % Date and time: Fri, Jan 12, 2024 - 5:16:27 PM
#> % Requires LaTeX packages: dcolumn 
#> \begin{table}[!htbp] \centering 
#>   \caption{Results} 
#>   \label{} 
#> \begin{tabular}{@{\extracolsep{5pt}}lD{.}{.}{-3} D{.}{.}{-3} D{.}{.}{-3} } 
#> \\[-1.8ex]\hline 
#> \hline \\[-1.8ex] 
#>  & \multicolumn{3}{c}{\textit{Dependent variable:}} \\ 
#> \cline{2-4} 
#> \\[-1.8ex] & \multicolumn{2}{c}{rating} & \multicolumn{1}{c}{high.rating} \\ 
#> \\[-1.8ex] & \multicolumn{2}{c}{\textit{OLS}} & \multicolumn{1}{c}{\textit{probit}} \\ 
#> \\[-1.8ex] & \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)}\\ 
#> \hline \\[-1.8ex] 
#>  complaints & 0.692^{***} & 0.682^{***} &  \\ 
#>   & (0.149) & (0.129) &  \\ 
#>   & & & \\ 
#>  privileges & -0.104 & -0.103 &  \\ 
#>   & (0.135) & (0.129) &  \\ 
#>   & & & \\ 
#>  learning & 0.249 & 0.238^{*} & 0.164^{***} \\ 
#>   & (0.160) & (0.139) & (0.053) \\ 
#>   & & & \\ 
#>  raises & -0.033 &  &  \\ 
#>   & (0.202) &  &  \\ 
#>   & & & \\ 
#>  critical & 0.015 &  & -0.001 \\ 
#>   & (0.147) &  & (0.044) \\ 
#>   & & & \\ 
#>  advance &  &  & -0.062 \\ 
#>   &  &  & (0.042) \\ 
#>   & & & \\ 
#>  Constant & 11.011 & 11.258 & -7.476^{**} \\ 
#>   & (11.704) & (7.318) & (3.570) \\ 
#>   & & & \\ 
#> \hline \\[-1.8ex] 
#> Observations & \multicolumn{1}{c}{30} & \multicolumn{1}{c}{30} & \multicolumn{1}{c}{30} \\ 
#> R$^{2}$ & \multicolumn{1}{c}{0.715} & \multicolumn{1}{c}{0.715} &  \\ 
#> Adjusted R$^{2}$ & \multicolumn{1}{c}{0.656} & \multicolumn{1}{c}{0.682} &  \\ 
#> Log Likelihood &  &  & \multicolumn{1}{c}{-9.087} \\ 
#> Akaike Inf. Crit. &  &  & \multicolumn{1}{c}{26.175} \\ 
#> Residual Std. Error & \multicolumn{1}{c}{7.139 (df = 24)} & \multicolumn{1}{c}{6.863 (df = 26)} &  \\ 
#> F Statistic & \multicolumn{1}{c}{12.063$^{***}$ (df = 5; 24)} & \multicolumn{1}{c}{21.743$^{***}$ (df = 3; 26)} &  \\ 
#> \hline 
#> \hline \\[-1.8ex] 
#> \textit{Note:}  & \multicolumn{3}{r}{$^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01} \\ 
#> \end{tabular} 
#> \end{table}
```


```r
# Latex
stargazer(
    linear.1,
    linear.2,
    probit.model,
    title = "Regression Results",
    align = TRUE,
    dep.var.labels = c("Overall Rating", "High Rating"),
    covariate.labels = c(
        "Handling of Complaints",
        "No Special Privileges",
        "Opportunity to Learn",
        "Performance-Based Raises",
        "Too Critical",
        "Advancement"
    ),
    omit.stat = c("LL", "ser", "f"),
    no.space = TRUE
)
```


```r
# ASCII text output
stargazer(
    linear.1,
    linear.2,
    type = "text",
    title = "Regression Results",
    dep.var.labels = c("Overall Rating", "High Rating"),
    covariate.labels = c(
        "Handling of Complaints",
        "No Special Privileges",
        "Opportunity to Learn",
        "Performance-Based Raises",
        "Too Critical",
        "Advancement"
    ),
    omit.stat = c("LL", "ser", "f"),
    ci = TRUE,
    ci.level = 0.90,
    single.row = TRUE
)
#> 
#> Regression Results
#> ========================================================================
#>                                        Dependent variable:              
#>                          -----------------------------------------------
#>                                          Overall Rating                 
#>                                    (1)                     (2)          
#> ------------------------------------------------------------------------
#> Handling of Complaints   0.692*** (0.447, 0.937) 0.682*** (0.470, 0.894)
#> No Special Privileges    -0.104 (-0.325, 0.118)  -0.103 (-0.316, 0.109) 
#> Opportunity to Learn      0.249 (-0.013, 0.512)   0.238* (0.009, 0.467) 
#> Performance-Based Raises -0.033 (-0.366, 0.299)                         
#> Too Critical              0.015 (-0.227, 0.258)                         
#> Advancement              11.011 (-8.240, 30.262) 11.258 (-0.779, 23.296)
#> ------------------------------------------------------------------------
#> Observations                       30                      30           
#> R2                                0.715                   0.715         
#> Adjusted R2                       0.656                   0.682         
#> ========================================================================
#> Note:                                        *p<0.1; **p<0.05; ***p<0.01
```


```r
stargazer(
    linear.1,
    linear.2,
    probit.model,
    title = "Regression Results",
    align = TRUE,
    dep.var.labels = c("Overall Rating", "High Rating"),
    covariate.labels = c(
        "Handling of Complaints",
        "No Special Privileges",
        "Opportunity to Learn",
        "Performance-Based Raises",
        "Too Critical",
        "Advancement"
    ),
    omit.stat = c("LL", "ser", "f"),
    no.space = TRUE
)
```

Correlation Table


```r
correlation.matrix <-
    cor(attitude[, c("rating", "complaints", "privileges")])
stargazer(correlation.matrix, title = "Correlation Matrix")
```

## Changes in an estimate


```r
coef_names <- coef_names[1:3] # Dropping intercept for plots
plot_summs(fit, fit_b, fit_c, robust = "HC3", coefs = coef_names)
```



\begin{center}\includegraphics[width=0.9\linewidth]{36-report_files/figure-latex/unnamed-chunk-10-1} \end{center}

```r
plot_summs(
    fit_c,
    robust = "HC3",
    coefs = coef_names,
    plot.distributions = TRUE
)
```



\begin{center}\includegraphics[width=0.9\linewidth]{36-report_files/figure-latex/unnamed-chunk-10-2} \end{center}

## Standard Errors

`sandwich` [vignette](cran.r-project.org/web/packages/sandwich/vignettes/sandwich-CL.pdf)

+------------+------------+------------------------------------------------------------------------------------------+--------------------------+
| Type       | Applicable | Usage                                                                                    | Reference                |
+============+============+==========================================================================================+==========================+
| `const`    |            | Assume constant variances                                                                |                          |
+------------+------------+------------------------------------------------------------------------------------------+--------------------------+
| `HC` `HC0` | `vcovCL`   | Heterogeneity                                                                            | [@white1980]             |
|            |            |                                                                                          |                          |
|            |            | White's estimator                                                                        |                          |
|            |            |                                                                                          |                          |
|            |            | All other heterogeneity SE methods are derivatives of this.                              |                          |
|            |            |                                                                                          |                          |
|            |            | No small sample bias adjustment                                                          |                          |
+------------+------------+------------------------------------------------------------------------------------------+--------------------------+
| `HC1`      | `vcovCL`   | Uses a degrees of freedom-based correction                                               | [@mackinnon1985some]     |
|            |            |                                                                                          |                          |
|            |            | When the number of clusters is small, `HC2` and `HC3` are better [@cameron2008bootstrap] |                          |
+------------+------------+------------------------------------------------------------------------------------------+--------------------------+
| `HC2`      | `vcovCL`   | Better with the linear model, but still applicable for [Generalized Linear Models]       |                          |
|            |            |                                                                                          |                          |
|            |            | Needs a hat (weighted) matrix                                                            |                          |
+------------+------------+------------------------------------------------------------------------------------------+--------------------------+
| `HC3`      | `vcovCL`   | Better with the linear model, but still applicable for [Generalized Linear Models]       |                          |
|            |            |                                                                                          |                          |
|            |            | Needs a hat (weighted) matrix                                                            |                          |
+------------+------------+------------------------------------------------------------------------------------------+--------------------------+
| `HC4`      | `vcovHC`   |                                                                                          | [@cribari2004asymptotic] |
+------------+------------+------------------------------------------------------------------------------------------+--------------------------+
| `HC4m`     | `vcovHC`   |                                                                                          | [@cribari2007inference]  |
+------------+------------+------------------------------------------------------------------------------------------+--------------------------+
| `HC5`      | `vcovHC`   |                                                                                          | [@cribari2011new]        |
+------------+------------+------------------------------------------------------------------------------------------+--------------------------+


```r
data(cars)
model <- lm(speed ~ dist, data = cars)
summary(model)
#> 
#> Call:
#> lm(formula = speed ~ dist, data = cars)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -7.5293 -2.1550  0.3615  2.4377  6.4179 
#> 
#> Coefficients:
#>             Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)  8.28391    0.87438   9.474 1.44e-12 ***
#> dist         0.16557    0.01749   9.464 1.49e-12 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 3.156 on 48 degrees of freedom
#> Multiple R-squared:  0.6511,	Adjusted R-squared:  0.6438 
#> F-statistic: 89.57 on 1 and 48 DF,  p-value: 1.49e-12
lmtest::coeftest(model, vcov. = sandwich::vcovHC(model, type = "HC1"))
#> 
#> t test of coefficients:
#> 
#>             Estimate Std. Error t value  Pr(>|t|)    
#> (Intercept) 8.283906   0.891860  9.2883 2.682e-12 ***
#> dist        0.165568   0.019402  8.5335 3.482e-11 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Coefficient Uncertainty and Distribution

The `ggdist` allows us to visualize uncertainty under both frequentist and Bayesian frameworks


```r
library(ggdist)
```

## Descriptive Tables

Export APA theme


```r
data("mtcars")

library(flextable)
theme_apa(flextable(mtcars[1:5,1:5]))
```

Export to Latex


```r
print(xtable::xtable(mtcars, type = "latex"),
      file = file.path(getwd(), "output", "mtcars_xtable.tex"))

# American Economic Review style
stargazer::stargazer(
    mtcars,
    title = "Testing",
    style = "aer",
    out = file.path(getwd(), "output", "mtcars_stargazer.tex")
)

# other styles include
# Administrative Science Quarterly
# Quarterly Journal of Economics
```

However, the above codes do not play well with notes. Hence, I create my own custom code that follows the AMA guidelines


```r
ama_tbl <- function(data, caption, label, note, output_path) {
  library(tidyverse)
  library(xtable)
  # Function to determine column alignment
  get_column_alignment <- function(data) {
    # Start with the alignment for the header row
    alignment <- c("l", "l")
    
    # Check each column
    for (col in seq_len(ncol(data))[-1]) {
      if (is.numeric(data[[col]])) {
        alignment <- c(alignment, "r")  # Right alignment for numbers
      } else {
        alignment <- c(alignment, "c")  # Center alignment for other data
      }
    }
    
    return(alignment)
  }
  
  data %>%
    # bold + left align first column 
    rename_with(~paste("\\multicolumn{1}{l}{\\textbf{", ., "}}"), 1) %>% 
    # bold + center align all other columns
    `colnames<-`(ifelse(colnames(.) != colnames(.)[1],
                        paste("\\multicolumn{1}{c}{\\textbf{", colnames(.), "}}"),
                        colnames(.))) %>% 
    
    xtable(caption = caption,
           label = label,
           align = get_column_alignment(data),
           auto = TRUE) %>%
    print(
      include.rownames = FALSE,
      caption.placement = "top",
      
      hline.after=c(-1, 0),
      
       # p{0.9\linewidth} sets the width of the column to 90% of the line width, and the @{} removes any extra padding around the cell.
      
      add.to.row = list(pos = list(nrow(data)), # Add at the bottom of the table
                        command = c(paste0("\\hline \n \\multicolumn{",ncol(data), "}{l} {", "\n \\begin{tabular}{@{}p{0.9\\linewidth}@{}} \n","Note: ", note, "\n \\end{tabular}  } \n"))), # Add your note here
      
      # make sure your heading is untouched (because you manually change it above)
      sanitize.colnames.function = identity,
      
      # place a the top of the page
      table.placement = "h",
      
      file = output_path
    )
}
```


```r
ama_tbl(
    mtcars,
    caption     = "This is caption",
    label       = "tab:this_is_label",
    note        = "this is note",
    output_path = file.path(getwd(), "output", "mtcars_custom_ama.tex")
)
```

## Visualizations and Plots

You can customize your plots based on your preferred journals. Here, I am creating a custom setting for the American Marketing Association.

American-Marketing-Association-ready theme for plots


```r
library(ggplot2)

# check available fonts
# windowsFonts()

# for Times New Roman
# names(windowsFonts()[windowsFonts()=="TT Times New Roman"])
```


```r
# Making a theme
amatheme = theme_bw(base_size = 14, base_family = "serif") + # This is Time New Roman
    
    theme(
        # remove major gridlines
        panel.grid.major   = element_blank(),

        # remove minor gridlines
        panel.grid.minor   = element_blank(),

        # remove panel border
        panel.border       = element_blank(),

        line               = element_line(),

        # change font
        text               = element_text(),

        # if you want to remove legend title
        # legend.title     = element_blank(),

        legend.title       = element_text(size = rel(0.6), face = "bold"),

        # change font size of legend
        legend.text        = element_text(size = rel(0.6)),
        
        legend.background  = element_rect(color = "black"),
        
        # legend.margin    = margin(t = 5, l = 5, r = 5, b = 5),
        # legend.key       = element_rect(color = NA, fill = NA),

        # change font size of main title
        plot.title         = element_text(
            size           = rel(1.2),
            face           = "bold",
            hjust          = 0.5,
            margin         = margin(b = 15)
        ),
        
        plot.margin        = unit(c(1, 1, 1, 1), "cm"),

        # add black line along axes
        axis.line          = element_line(colour = "black", linewidth = .8),
        
        axis.ticks         = element_line(),
        

        # axis title
        axis.title.x       = element_text(size = rel(1.2), face = "bold"),
        axis.title.y       = element_text(size = rel(1.2), face = "bold"),

        # axis text size
        axis.text.y        = element_text(size = rel(1)),
        axis.text.x        = element_text(size = rel(1))
    )
```

Example


```r
library(tidyverse)
library(ggsci)
data("mtcars")
yourplot <- mtcars %>%
    select(mpg, cyl, gear) %>%
    ggplot(., aes(x = mpg, y = cyl, fill = gear)) + 
    geom_point() +
    labs(title="Some Plot") 

yourplot + 
    amatheme + 
    # choose different color theme
    scale_color_npg() 
```



\begin{center}\includegraphics[width=0.9\linewidth]{36-report_files/figure-latex/unnamed-chunk-19-1} \end{center}

```r

yourplot + 
    amatheme + 
    scale_color_continuous()
```



\begin{center}\includegraphics[width=0.9\linewidth]{36-report_files/figure-latex/unnamed-chunk-19-2} \end{center}

Other pre-specified themes


```r
library(ggthemes)


# Stata theme
yourplot +
    theme_stata()
```



\begin{center}\includegraphics[width=0.9\linewidth]{36-report_files/figure-latex/unnamed-chunk-20-1} \end{center}

```r

# The economist theme
yourplot + 
    theme_economist()
```



\begin{center}\includegraphics[width=0.9\linewidth]{36-report_files/figure-latex/unnamed-chunk-20-2} \end{center}

```r

yourplot + 
    theme_economist_white()
```



\begin{center}\includegraphics[width=0.9\linewidth]{36-report_files/figure-latex/unnamed-chunk-20-3} \end{center}

```r

# Wall street journal theme
yourplot + 
    theme_wsj()
```



\begin{center}\includegraphics[width=0.9\linewidth]{36-report_files/figure-latex/unnamed-chunk-20-4} \end{center}

```r

# APA theme
yourplot +
    jtools::theme_apa(
        legend.font.size = 24,
        x.font.size = 20,
        y.font.size = 20
    )
```



\begin{center}\includegraphics[width=0.9\linewidth]{36-report_files/figure-latex/unnamed-chunk-20-5} \end{center}
