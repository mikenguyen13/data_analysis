# Moderation

-   Spotlight Analysis: Compare the mean of the dependent of the two groups (treatment and control) at every value ([Simple Slopes Analysis])
-   Floodlight Analysis: is spotlight analysis on the whole range of the moderator ([Johnson-Neyman intervals])

Other Resources:

-   `BANOVAL` : floodlight analysis on Bayesian ANOVA models

-   `cSEM` : `doFloodlightAnalysis` in SEM model

-   [@spiller2013]

Terminology:

-   Main effects (slopes): coefficients that do no involve interaction terms

-   Simple slope: when a continuous independent variable interact with a moderating variable, its slope at a particular level of the moderating variable

-   Simple effect: when a categorical independent variable interacts with a moderating variable, its effect at a particular level of the moderating variable.

Example:

$$
Y = \beta_0 + \beta_1 X + \beta_2 M + \beta_3 X \times M
$$

where

-   $\beta_0$ = intercept

-   $\beta_1$ = simple effect (slope) of $X$ (independent variable)

-   $\beta_2$ = simple effect (slope) of $M$ (moderating variable)

-   $\beta_3$ = interaction of $X$ and $M$

Three types of interactions:

1.  [Continuous by continuous]
2.  [Continuous by categorical]
3.  [Categorical by categorical]

## emmeans package


```r
install.packages("emmeans")
```


```r
library(emmeans)
```

Data set is from [UCLA seminar](https://stats.oarc.ucla.edu/r/seminars/interactions-r/) where `gender` and `prog` are categorical


```r
dat <- readRDS("data/exercise.rds") %>%
    mutate(prog = factor(prog, labels = c("jog", "swim", "read"))) %>%
    mutate(gender = factor(gender, labels = c("male", "female")))
```

### Continuous by continuous


```r
contcont <- lm(loss~hours*effort,data=dat)
summary(contcont)
#> 
#> Call:
#> lm(formula = loss ~ hours * effort, data = dat)
#> 
#> Residuals:
#>    Min     1Q Median     3Q    Max 
#> -29.52 -10.60  -1.78  11.13  34.51 
#> 
#> Coefficients:
#>              Estimate Std. Error t value Pr(>|t|)  
#> (Intercept)   7.79864   11.60362   0.672   0.5017  
#> hours        -9.37568    5.66392  -1.655   0.0982 .
#> effort       -0.08028    0.38465  -0.209   0.8347  
#> hours:effort  0.39335    0.18750   2.098   0.0362 *
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 13.56 on 896 degrees of freedom
#> Multiple R-squared:  0.07818,	Adjusted R-squared:  0.07509 
#> F-statistic: 25.33 on 3 and 896 DF,  p-value: 9.826e-16
```

Simple slopes for a continuous by continuous model

Spotlight analysis [@aiken2005interaction]: usually pick 3 values of moderating variable:

-   Mean Moderating Variable + $\sigma \times$ (Moderating variable)

-   Mean Moderating Variable

-   Mean Moderating Variable - $\sigma \times$ (Moderating variable)


```r
effar <- round(mean(dat$effort) + sd(dat$effort), 1)
effr  <- round(mean(dat$effort), 1)
effbr <- round(mean(dat$effort) - sd(dat$effort), 1)
```


```r
# specify list of points
mylist <- list(effort = c(effbr, effr, effar))

# get the estimates
emtrends(contcont, ~ effort, var = "hours", at = mylist)
#>  effort hours.trend    SE  df lower.CL upper.CL
#>    24.5       0.261 1.352 896   -2.392     2.91
#>    29.7       2.307 0.915 896    0.511     4.10
#>    34.8       4.313 1.308 896    1.745     6.88
#> 
#> Confidence level used: 0.95

# plot
mylist <- list(hours = seq(0, 4, by = 0.4),
               effort = c(effbr, effr, effar))
emmip(contcont, effort ~ hours, at = mylist, CIs = TRUE)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-6-1.png" width="90%" style="display: block; margin: auto;" />

```r

# statistical test for slope difference
emtrends(
    contcont,
    pairwise ~ effort,
    var = "hours",
    at = mylist,
    adjust = "none"
)
#> $emtrends
#>  effort hours.trend    SE  df lower.CL upper.CL
#>    24.5       0.261 1.352 896   -2.392     2.91
#>    29.7       2.307 0.915 896    0.511     4.10
#>    34.8       4.313 1.308 896    1.745     6.88
#> 
#> Results are averaged over the levels of: hours 
#> Confidence level used: 0.95 
#> 
#> $contrasts
#>  contrast                estimate    SE  df t.ratio p.value
#>  effort24.5 - effort29.7    -2.05 0.975 896  -2.098  0.0362
#>  effort24.5 - effort34.8    -4.05 1.931 896  -2.098  0.0362
#>  effort29.7 - effort34.8    -2.01 0.956 896  -2.098  0.0362
#> 
#> Results are averaged over the levels of: hours
```

The 3 p-values are the same as the interaction term.

For publication, we use


```r
library(ggplot2)

# data
mylist <- list(hours = seq(0, 4, by = 0.4),
               effort = c(effbr, effr, effar))
contcontdat <-
    emmip(contcont,
          effort ~ hours,
          at = mylist,
          CIs = TRUE,
          plotit = FALSE)
contcontdat$feffort <- factor(contcontdat$effort)
levels(contcontdat$feffort) <- c("low", "med", "high")

# plot
p  <-
    ggplot(data = contcontdat, 
           aes(x = hours, y = yvar, color = feffort)) +  
    geom_line()
p1 <-
    p + 
    geom_ribbon(aes(ymax = UCL, ymin = LCL, fill = feffort), 
                    alpha = 0.4)
p1  + labs(x = "Hours",
           y = "Weight Loss",
           color = "Effort",
           fill = "Effort")
```

<img src="17-moderation_files/figure-html/unnamed-chunk-7-1.png" width="90%" style="display: block; margin: auto;" />

### Continuous by categorical


```r
# use Female as basline
dat$gender <- relevel(dat$gender, ref = "female")

contcat <- lm(loss ~ hours * gender, data = dat)
summary(contcat)
#> 
#> Call:
#> lm(formula = loss ~ hours * gender, data = dat)
#> 
#> Residuals:
#>     Min      1Q  Median      3Q     Max 
#> -27.118 -11.350  -1.963  10.001  42.376 
#> 
#> Coefficients:
#>                  Estimate Std. Error t value Pr(>|t|)  
#> (Intercept)         3.335      2.731   1.221    0.222  
#> hours               3.315      1.332   2.489    0.013 *
#> gendermale          3.571      3.915   0.912    0.362  
#> hours:gendermale   -1.724      1.898  -0.908    0.364  
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 14.06 on 896 degrees of freedom
#> Multiple R-squared:  0.008433,	Adjusted R-squared:  0.005113 
#> F-statistic:  2.54 on 3 and 896 DF,  p-value: 0.05523
```

Get simple slopes by each level of the categorical moderator


```r
emtrends(contcat, ~ gender, var = "hours")
#>  gender hours.trend   SE  df lower.CL upper.CL
#>  female        3.32 1.33 896    0.702     5.93
#>  male          1.59 1.35 896   -1.063     4.25
#> 
#> Confidence level used: 0.95

# test difference in slopes
emtrends(contcat, pairwise ~ gender, var = "hours")
#> $emtrends
#>  gender hours.trend   SE  df lower.CL upper.CL
#>  female        3.32 1.33 896    0.702     5.93
#>  male          1.59 1.35 896   -1.063     4.25
#> 
#> Confidence level used: 0.95 
#> 
#> $contrasts
#>  contrast      estimate  SE  df t.ratio p.value
#>  female - male     1.72 1.9 896   0.908  0.3639
# which is the same as the interaction term
```


```r
# plot
(mylist <- list(
    hours = seq(0, 4, by = 0.4),
    gender = c("female", "male")
))
#> $hours
#>  [1] 0.0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.6 4.0
#> 
#> $gender
#> [1] "female" "male"
emmip(contcat, gender ~ hours, at = mylist, CIs = TRUE)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-10-1.png" width="90%" style="display: block; margin: auto;" />

### Categorical by categorical


```r
# relevel baseline
dat$prog   <- relevel(dat$prog, ref = "read")
dat$gender <- relevel(dat$gender, ref = "female")
```


```r
catcat <- lm(loss ~ gender * prog, data = dat)
summary(catcat)
#> 
#> Call:
#> lm(formula = loss ~ gender * prog, data = dat)
#> 
#> Residuals:
#>      Min       1Q   Median       3Q      Max 
#> -19.1723  -4.1894  -0.0994   3.7506  27.6939 
#> 
#> Coefficients:
#>                     Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)          -3.6201     0.5322  -6.802 1.89e-11 ***
#> gendermale           -0.3355     0.7527  -0.446    0.656    
#> progjog               7.9088     0.7527  10.507  < 2e-16 ***
#> progswim             32.7378     0.7527  43.494  < 2e-16 ***
#> gendermale:progjog    7.8188     1.0645   7.345 4.63e-13 ***
#> gendermale:progswim  -6.2599     1.0645  -5.881 5.77e-09 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 6.519 on 894 degrees of freedom
#> Multiple R-squared:  0.7875,	Adjusted R-squared:  0.7863 
#> F-statistic: 662.5 on 5 and 894 DF,  p-value: < 2.2e-16
```

Simple effects


```r
emcatcat <- emmeans(catcat, ~ gender*prog)

# differences in predicted values
contrast(emcatcat, 
         "revpairwise", 
         by = "prog", 
         adjust = "bonferroni")
#> prog = read:
#>  contrast      estimate    SE  df t.ratio p.value
#>  male - female   -0.335 0.753 894  -0.446  0.6559
#> 
#> prog = jog:
#>  contrast      estimate    SE  df t.ratio p.value
#>  male - female    7.483 0.753 894   9.942  <.0001
#> 
#> prog = swim:
#>  contrast      estimate    SE  df t.ratio p.value
#>  male - female   -6.595 0.753 894  -8.762  <.0001
```

Plot


```r
emmip(catcat, prog ~ gender,CIs=TRUE)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-14-1.png" width="90%" style="display: block; margin: auto;" />

Bar graph


```r
catcatdat <- emmip(catcat,
                   gender ~ prog,
                   CIs = TRUE,
                   plotit = FALSE)
p <-
    ggplot(data = catcatdat,
           aes(x = prog, y = yvar, fill = gender)) +
    geom_bar(stat = "identity", position = "dodge")

p1 <-
    p + geom_errorbar(
        position = position_dodge(.9),
        width = .25,
        aes(ymax = UCL, ymin = LCL),
        alpha = 0.3
    )
p1  + labs(x = "Program", y = "Weight Loss", fill = "Gender")
```

<img src="17-moderation_files/figure-html/unnamed-chunk-15-1.png" width="90%" style="display: block; margin: auto;" />

## probmod package

-   Not recommend: package has serious problem with subscript.


```r
install.packages("probemod")
```


```r
library(probemod)

myModel <-
    lm(loss ~ hours * gender, data = dat %>% 
           select(loss, hours, gender))
jnresults <- jn(myModel,
                dv = 'loss',
                iv = 'hours',
                mod = 'gender')


pickapoint(
    myModel,
    dv = 'loss',
    iv = 'hours',
    mod = 'gender',
    alpha = .01
)

plot(jnresults)
```

## interactions package

-   Recommend


```r
install.packages("interactions")
```

### Continuous interaction

-   (at least one of the two variables is continuous)


```r
library(interactions)
library(jtools) # for summ()
states <- as.data.frame(state.x77)
fiti <- lm(Income ~ Illiteracy * Murder + `HS Grad`, data = states)
summ(fiti)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Observations </td>
   <td style="text-align:right;"> 50 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Dependent variable </td>
   <td style="text-align:right;"> Income </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Type </td>
   <td style="text-align:right;"> OLS linear regression </td>
  </tr>
</tbody>
</table> <table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> F(4,45) </td>
   <td style="text-align:right;"> 10.65 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> R² </td>
   <td style="text-align:right;"> 0.49 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Adj. R² </td>
   <td style="text-align:right;"> 0.44 </td>
  </tr>
</tbody>
</table> <table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;border-bottom: 0;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Est. </th>
   <th style="text-align:right;"> S.E. </th>
   <th style="text-align:right;"> t val. </th>
   <th style="text-align:right;"> p </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> (Intercept) </td>
   <td style="text-align:right;"> 1414.46 </td>
   <td style="text-align:right;"> 737.84 </td>
   <td style="text-align:right;"> 1.92 </td>
   <td style="text-align:right;"> 0.06 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Illiteracy </td>
   <td style="text-align:right;"> 753.07 </td>
   <td style="text-align:right;"> 385.90 </td>
   <td style="text-align:right;"> 1.95 </td>
   <td style="text-align:right;"> 0.06 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Murder </td>
   <td style="text-align:right;"> 130.60 </td>
   <td style="text-align:right;"> 44.67 </td>
   <td style="text-align:right;"> 2.92 </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> `HS Grad` </td>
   <td style="text-align:right;"> 40.76 </td>
   <td style="text-align:right;"> 10.92 </td>
   <td style="text-align:right;"> 3.73 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Illiteracy:Murder </td>
   <td style="text-align:right;"> -97.04 </td>
   <td style="text-align:right;"> 35.86 </td>
   <td style="text-align:right;"> -2.71 </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
</tbody>
<tfoot><tr><td style="padding: 0; " colspan="100%">
<sup></sup> Standard errors: OLS</td></tr></tfoot>
</table>

For continuous moderator, the three values chosen are:

-   -1 SD above the mean

-   The mean

-   -1 SD below the mean


```r
interact_plot(fiti,
              pred = Illiteracy,
              modx = Murder,
              
              # if you don't want the plot to mean-center
              # centered = "none", 
              
              # exclude the mean value of the moderator
              # modx.values = "plus-minus", 
              
              # split moderator's distribution into 3 groups
              # modx.values = "terciles" 
              
              plot.points = T, # overlay data
              
              
              # different shape for differennt levels of the moderator
              point.shape = T, 
              
              # if two data points are on top one another, 
              # this moves them apart by little
              jitter = 0.1, 
              
              # other appearance option
              x.label = "X label", 
              y.label = "Y label",
              main.title = "Title",
              legend.main = "Legend Title",
              colors = "blue",
              
              # include confidence band
              interval = TRUE, 
              int.width = 0.9, 
              robust = TRUE # use robust SE
              ) 
```

<img src="17-moderation_files/figure-html/unnamed-chunk-20-1.png" width="90%" style="display: block; margin: auto;" />

To include weights from the regression inn the plot


```r
fiti <- lm(Income ~ Illiteracy * Murder,
           data = states,
           weights = Population)

interact_plot(fiti,
              pred = Illiteracy,
              modx = Murder,
              plot.points = TRUE)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-21-1.png" width="90%" style="display: block; margin: auto;" />

Partial Effect Plot


```r
library(ggplot2)
data(cars)
fitc <- lm(cty ~ year + cyl * displ + class + fl + drv, 
           data = mpg)
summ(fitc)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Observations </td>
   <td style="text-align:right;"> 234 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Dependent variable </td>
   <td style="text-align:right;"> cty </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Type </td>
   <td style="text-align:right;"> OLS linear regression </td>
  </tr>
</tbody>
</table> <table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> F(16,217) </td>
   <td style="text-align:right;"> 99.73 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> R² </td>
   <td style="text-align:right;"> 0.88 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Adj. R² </td>
   <td style="text-align:right;"> 0.87 </td>
  </tr>
</tbody>
</table> <table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;border-bottom: 0;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Est. </th>
   <th style="text-align:right;"> S.E. </th>
   <th style="text-align:right;"> t val. </th>
   <th style="text-align:right;"> p </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> (Intercept) </td>
   <td style="text-align:right;"> -200.98 </td>
   <td style="text-align:right;"> 47.01 </td>
   <td style="text-align:right;"> -4.28 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> year </td>
   <td style="text-align:right;"> 0.12 </td>
   <td style="text-align:right;"> 0.02 </td>
   <td style="text-align:right;"> 5.03 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> cyl </td>
   <td style="text-align:right;"> -1.86 </td>
   <td style="text-align:right;"> 0.28 </td>
   <td style="text-align:right;"> -6.69 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> displ </td>
   <td style="text-align:right;"> -3.56 </td>
   <td style="text-align:right;"> 0.66 </td>
   <td style="text-align:right;"> -5.41 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> classcompact </td>
   <td style="text-align:right;"> -2.60 </td>
   <td style="text-align:right;"> 0.93 </td>
   <td style="text-align:right;"> -2.80 </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> classmidsize </td>
   <td style="text-align:right;"> -2.63 </td>
   <td style="text-align:right;"> 0.93 </td>
   <td style="text-align:right;"> -2.82 </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> classminivan </td>
   <td style="text-align:right;"> -4.41 </td>
   <td style="text-align:right;"> 1.04 </td>
   <td style="text-align:right;"> -4.24 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> classpickup </td>
   <td style="text-align:right;"> -4.37 </td>
   <td style="text-align:right;"> 0.93 </td>
   <td style="text-align:right;"> -4.68 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> classsubcompact </td>
   <td style="text-align:right;"> -2.38 </td>
   <td style="text-align:right;"> 0.93 </td>
   <td style="text-align:right;"> -2.56 </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> classsuv </td>
   <td style="text-align:right;"> -4.27 </td>
   <td style="text-align:right;"> 0.87 </td>
   <td style="text-align:right;"> -4.92 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> fld </td>
   <td style="text-align:right;"> 6.34 </td>
   <td style="text-align:right;"> 1.69 </td>
   <td style="text-align:right;"> 3.74 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> fle </td>
   <td style="text-align:right;"> -4.57 </td>
   <td style="text-align:right;"> 1.66 </td>
   <td style="text-align:right;"> -2.75 </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> flp </td>
   <td style="text-align:right;"> -1.92 </td>
   <td style="text-align:right;"> 1.59 </td>
   <td style="text-align:right;"> -1.21 </td>
   <td style="text-align:right;"> 0.23 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> flr </td>
   <td style="text-align:right;"> -0.79 </td>
   <td style="text-align:right;"> 1.57 </td>
   <td style="text-align:right;"> -0.50 </td>
   <td style="text-align:right;"> 0.61 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> drvf </td>
   <td style="text-align:right;"> 1.40 </td>
   <td style="text-align:right;"> 0.40 </td>
   <td style="text-align:right;"> 3.52 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> drvr </td>
   <td style="text-align:right;"> 0.49 </td>
   <td style="text-align:right;"> 0.46 </td>
   <td style="text-align:right;"> 1.06 </td>
   <td style="text-align:right;"> 0.29 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> cyl:displ </td>
   <td style="text-align:right;"> 0.36 </td>
   <td style="text-align:right;"> 0.08 </td>
   <td style="text-align:right;"> 4.56 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
</tbody>
<tfoot><tr><td style="padding: 0; " colspan="100%">
<sup></sup> Standard errors: OLS</td></tr></tfoot>
</table>

```r

interact_plot(
    fitc,
    pred = displ,
    modx = cyl,
    # the observed data is based on displ, cyl, and model error
    partial.residuals = TRUE, 
    modx.values = c(4, 5, 6, 8)
)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-22-1.png" width="90%" style="display: block; margin: auto;" />

Check linearity assumption in the model

Plot the lines based on the subsample (red line), and whole sample (black line)


```r
x_2 <- runif(n = 200, min = -3, max = 3)
w   <- rbinom(n = 200, size = 1, prob = 0.5)
err <- rnorm(n = 200, mean = 0, sd = 4)
y_2 <- 2.5 - x_2 ^ 2 - 5 * w + 2 * w * (x_2 ^ 2) + err

data_2 <- as.data.frame(cbind(x_2, y_2, w))

model_2 <- lm(y_2 ~ x_2 * w, data = data_2)
summ(model_2)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Observations </td>
   <td style="text-align:right;"> 200 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Dependent variable </td>
   <td style="text-align:right;"> y_2 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Type </td>
   <td style="text-align:right;"> OLS linear regression </td>
  </tr>
</tbody>
</table> <table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> F(3,196) </td>
   <td style="text-align:right;"> 1.57 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> R² </td>
   <td style="text-align:right;"> 0.02 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Adj. R² </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
</tbody>
</table> <table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;border-bottom: 0;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Est. </th>
   <th style="text-align:right;"> S.E. </th>
   <th style="text-align:right;"> t val. </th>
   <th style="text-align:right;"> p </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> (Intercept) </td>
   <td style="text-align:right;"> -1.12 </td>
   <td style="text-align:right;"> 0.50 </td>
   <td style="text-align:right;"> -2.27 </td>
   <td style="text-align:right;"> 0.02 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> x_2 </td>
   <td style="text-align:right;"> 0.28 </td>
   <td style="text-align:right;"> 0.27 </td>
   <td style="text-align:right;"> 1.04 </td>
   <td style="text-align:right;"> 0.30 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> w </td>
   <td style="text-align:right;"> 1.42 </td>
   <td style="text-align:right;"> 0.71 </td>
   <td style="text-align:right;"> 2.00 </td>
   <td style="text-align:right;"> 0.05 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> x_2:w </td>
   <td style="text-align:right;"> -0.23 </td>
   <td style="text-align:right;"> 0.40 </td>
   <td style="text-align:right;"> -0.58 </td>
   <td style="text-align:right;"> 0.56 </td>
  </tr>
</tbody>
<tfoot><tr><td style="padding: 0; " colspan="100%">
<sup></sup> Standard errors: OLS</td></tr></tfoot>
</table>

```r
interact_plot(
    model_2,
    pred = x_2,
    modx = w,
    linearity.check = TRUE,
    plot.points = TRUE
)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-23-1.png" width="90%" style="display: block; margin: auto;" />

#### Simple Slopes Analysis

-   continuous by continuous variable interaction (still work for binary)

-   conditional slope of the variable of interest (i.e., the slope of $X$ when we hold $M$ constant at a value)

Using `sim_slopes` it will

-   mean-center all variables except the variable of interest

-   For moderator that is

    -   Continuous, it will pick mean, and plus/minus 1 SD

    -   Categorical, it will use all factor

`sim_slopes` requires

-   A regression model with an interaction term)

-   Variable of interest (`pred =`)

-   Moderator: (`modx =`)


```r
sim_slopes(fiti,
           pred = Illiteracy,
           modx = Murder,
           johnson_neyman = FALSE)
#> SIMPLE SLOPES ANALYSIS 
#> 
#> Slope of Illiteracy when Murder =  5.420973 (- 1 SD): 
#> 
#>     Est.     S.E.   t val.      p
#> -------- -------- -------- ------
#>   -71.59   268.65    -0.27   0.79
#> 
#> Slope of Illiteracy when Murder =  8.685043 (Mean): 
#> 
#>      Est.     S.E.   t val.      p
#> --------- -------- -------- ------
#>   -437.12   175.82    -2.49   0.02
#> 
#> Slope of Illiteracy when Murder = 11.949113 (+ 1 SD): 
#> 
#>      Est.     S.E.   t val.      p
#> --------- -------- -------- ------
#>   -802.66   145.72    -5.51   0.00

# plot the coefficients
ss <- sim_slopes(fiti,
                 pred = Illiteracy,
                 modx = Murder,
                 modx.values = c(0, 5, 10))
plot(ss)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-24-1.png" width="90%" style="display: block; margin: auto;" />

```r

# table 
ss <- sim_slopes(fiti,
                 pred = Illiteracy,
                 modx = Murder,
                 modx.values = c(0, 5, 10))
library(huxtable)
as_huxtable(ss)
```


```{=html}
<table class="huxtable" style="border-collapse: collapse; border: 0px; margin-bottom: 2em; margin-top: 2em; ; margin-left: auto; margin-right: auto;  " id="tab:unnamed-chunk-24">
<caption style="caption-side: top; text-align: center;">(#tab:unnamed-chunk-24) </caption><col><col><tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">Value of Murder</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">Slope of Illiteracy</td></tr>
<tr>
<th style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">Value of Murder</th><th style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">slope</th></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">0.00</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">535.50 (458.77)</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">5.00</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">-24.44 (282.48)</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">10.00</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">-584.38 (152.37)***</td></tr>
</table>

```


#### Johnson-Neyman intervals

To know all the values of the moderator for which the slope of the variable of interest will be statistically significant, we can use the Johnson-Neyman interval [@johnson1936tests]

Even though we kind of know that the alpha level when implementing the Johnson-Neyman interval is not correct [@bauer2005probing], not until recently that there is a correction for the type I and II errors [@esarey2018marginal].

Since Johnson-Neyman inflates the type I error (comparisons across all values of the moderator)


```r
sim_slopes(
    fiti,
    pred = Illiteracy,
    modx = Murder,
    johnson_neyman = TRUE,
    control.fdr = TRUE,
    # correction for type I and II
    
    # include conditional intecepts
    # cond.int = TRUE, 
    
    robust = "HC3",
    # rubust SE
    
    # don't mean-centered non-focal variables
    # centered = "none",
    jnalpha = 0.05
)
#> JOHNSON-NEYMAN INTERVAL 
#> 
#> When Murder is OUTSIDE the interval [-11.70, 8.75], the slope of Illiteracy
#> is p < .05.
#> 
#> Note: The range of observed values of Murder is [1.40, 15.10]
#> 
#> Interval calculated using false discovery rate adjusted t = 2.33 
#> 
#> SIMPLE SLOPES ANALYSIS 
#> 
#> Slope of Illiteracy when Murder =  5.420973 (- 1 SD): 
#> 
#>     Est.     S.E.   t val.      p
#> -------- -------- -------- ------
#>   -71.59   256.60    -0.28   0.78
#> 
#> Slope of Illiteracy when Murder =  8.685043 (Mean): 
#> 
#>      Est.     S.E.   t val.      p
#> --------- -------- -------- ------
#>   -437.12   191.07    -2.29   0.03
#> 
#> Slope of Illiteracy when Murder = 11.949113 (+ 1 SD): 
#> 
#>      Est.     S.E.   t val.      p
#> --------- -------- -------- ------
#>   -802.66   178.75    -4.49   0.00
```

For plotting, we can use `johnson_neyman`


```r
johnson_neyman(fiti,
               pred = Illiteracy,
               modx = Murder,
               
               # correction for type I and II
               control.fdr = TRUE, 
               alpha = .05)
#> JOHNSON-NEYMAN INTERVAL 
#> 
#> When Murder is OUTSIDE the interval [-22.57, 8.52], the slope of Illiteracy
#> is p < .05.
#> 
#> Note: The range of observed values of Murder is [1.40, 15.10]
#> 
#> Interval calculated using false discovery rate adjusted t = 2.33
```

<img src="17-moderation_files/figure-html/unnamed-chunk-26-1.png" width="90%" style="display: block; margin: auto;" />

Note:

-   y-axis is the **conditional slope** of the variable of interest

#### 3-way interaction


```r
# fita3 <-
#     lm(rating ~ privileges * critical * learning, 
#        data = attitude)
# 
# probe_interaction(
#     fita3,
#     pred = critical,
#     modx = learning,
#     mod2 = privileges,
#     alpha = .1
# )


mtcars$cyl <- factor(mtcars$cyl,
                     labels = c("4 cylinder", "6 cylinder", "8 cylinder"))
fitc3 <- lm(mpg ~ hp * wt * cyl, data = mtcars)
interact_plot(fitc3,
              pred = hp,
              modx = wt,
              mod2 = cyl) +
    theme_apa(legend.pos = "bottomright")
```

<img src="17-moderation_files/figure-html/unnamed-chunk-27-1.png" width="90%" style="display: block; margin: auto;" />

Johnson-Neyman 3-way interaction


```r
library(survey)
data(api)

dstrat <- svydesign(
    id = ~ 1,
    strata = ~ stype,
    weights = ~ pw,
    data = apistrat,
    fpc = ~ fpc
)

regmodel3 <-
    survey::svyglm(api00 ~ avg.ed * growth * enroll, design = dstrat)

sim_slopes(
    regmodel3,
    pred = growth,
    modx = avg.ed,
    mod2 = enroll,
    jnplot = TRUE
)
#> ███████████████ While enroll (2nd moderator) =  153.0518 (- 1 SD) ██████████████ 
#> 
#> JOHNSON-NEYMAN INTERVAL 
#> 
#> When avg.ed is OUTSIDE the interval [2.75, 3.82], the slope of growth is p
#> < .05.
#> 
#> Note: The range of observed values of avg.ed is [1.38, 4.44]
#> 
#> SIMPLE SLOPES ANALYSIS 
#> 
#> Slope of growth when avg.ed = 2.085002 (- 1 SD): 
#> 
#>   Est.   S.E.   t val.      p
#> ------ ------ -------- ------
#>   1.25   0.32     3.86   0.00
#> 
#> Slope of growth when avg.ed = 2.787381 (Mean): 
#> 
#>   Est.   S.E.   t val.      p
#> ------ ------ -------- ------
#>   0.39   0.22     1.75   0.08
#> 
#> Slope of growth when avg.ed = 3.489761 (+ 1 SD): 
#> 
#>    Est.   S.E.   t val.      p
#> ------- ------ -------- ------
#>   -0.48   0.35    -1.37   0.17
#> 
#> ████████████████ While enroll (2nd moderator) =  595.2821 (Mean) ███████████████ 
#> 
#> JOHNSON-NEYMAN INTERVAL 
#> 
#> When avg.ed is OUTSIDE the interval [2.84, 7.83], the slope of growth is p
#> < .05.
#> 
#> Note: The range of observed values of avg.ed is [1.38, 4.44]
#> 
#> SIMPLE SLOPES ANALYSIS 
#> 
#> Slope of growth when avg.ed = 2.085002 (- 1 SD): 
#> 
#>   Est.   S.E.   t val.      p
#> ------ ------ -------- ------
#>   0.72   0.22     3.29   0.00
#> 
#> Slope of growth when avg.ed = 2.787381 (Mean): 
#> 
#>   Est.   S.E.   t val.      p
#> ------ ------ -------- ------
#>   0.34   0.16     2.16   0.03
#> 
#> Slope of growth when avg.ed = 3.489761 (+ 1 SD): 
#> 
#>    Est.   S.E.   t val.      p
#> ------- ------ -------- ------
#>   -0.04   0.24    -0.16   0.87
#> 
#> ███████████████ While enroll (2nd moderator) = 1037.5125 (+ 1 SD) ██████████████ 
#> 
#> JOHNSON-NEYMAN INTERVAL 
#> 
#> The Johnson-Neyman interval could not be found. Is the p value for your
#> interaction term below the specified alpha?
#> 
#> SIMPLE SLOPES ANALYSIS 
#> 
#> Slope of growth when avg.ed = 2.085002 (- 1 SD): 
#> 
#>   Est.   S.E.   t val.      p
#> ------ ------ -------- ------
#>   0.18   0.31     0.58   0.56
#> 
#> Slope of growth when avg.ed = 2.787381 (Mean): 
#> 
#>   Est.   S.E.   t val.      p
#> ------ ------ -------- ------
#>   0.29   0.20     1.49   0.14
#> 
#> Slope of growth when avg.ed = 3.489761 (+ 1 SD): 
#> 
#>   Est.   S.E.   t val.      p
#> ------ ------ -------- ------
#>   0.40   0.27     1.49   0.14
```

<img src="17-moderation_files/figure-html/unnamed-chunk-28-1.png" width="90%" style="display: block; margin: auto;" />

Report


```r
ss3 <-
    sim_slopes(regmodel3,
               pred = growth,
               modx = avg.ed,
               mod2 = enroll)
plot(ss3)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-29-1.png" width="90%" style="display: block; margin: auto;" />

```r
as_huxtable(ss3)
```


```{=html}
<table class="huxtable" style="border-collapse: collapse; border: 0px; margin-bottom: 2em; margin-top: 2em; ; margin-left: auto; margin-right: auto;  " id="tab:unnamed-chunk-29">
<caption style="caption-side: top; text-align: center;">(#tab:unnamed-chunk-29) </caption><col><col><tr>
<td colspan="2" style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal; font-style: italic;">enroll = 153</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">Value of avg.ed</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">Slope of growth</td></tr>
<tr>
<th style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">Value of avg.ed</th><th style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">slope</th></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">2.09</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">1.25 (0.32)***</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">2.79</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">0.39 (0.22)#</td></tr>
<tr>
<td colspan="2" style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal; font-style: italic;">enroll = 595.28</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">Value of avg.ed</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">Slope of growth</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">3.49</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">-0.48 (0.35)</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">2.09</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">0.72 (0.22)**</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">2.79</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">0.34 (0.16)*</td></tr>
<tr>
<td colspan="2" style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal; font-style: italic;">enroll = 1037.51</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">Value of avg.ed</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 1pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">Slope of growth</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">3.49</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 1pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">-0.04 (0.24)</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">2.09</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">0.18 (0.31)</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">2.79</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">0.29 (0.20)</td></tr>
<tr>
<td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">3.49</td><td style="vertical-align: top; text-align: left; white-space: normal; border-style: solid solid solid solid; border-width: 0pt 0pt 0pt 0pt;    padding: 6pt 6pt 6pt 6pt; font-weight: normal;">0.40 (0.27)</td></tr>
</table>

```


### Categorical interaction


```r
library(ggplot2)
mpg2 <- mpg %>% 
    mutate(cyl = factor(cyl))

mpg2["auto"] <- "auto"
mpg2$auto[mpg2$trans %in% c("manual(m5)", "manual(m6)")] <- "manual"
mpg2$auto <- factor(mpg2$auto)
mpg2["fwd"] <- "2wd"
mpg2$fwd[mpg2$drv == "4"] <- "4wd"
mpg2$fwd <- factor(mpg2$fwd)
## Drop the two cars with 5 cylinders (rest are 4, 6, or 8)
mpg2 <- mpg2[mpg2$cyl != "5", ]
## Fit the model
fit3 <- lm(cty ~ cyl * fwd * auto, data = mpg2)

library(jtools) # for summ()
summ(fit3)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Observations </td>
   <td style="text-align:right;"> 230 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Dependent variable </td>
   <td style="text-align:right;"> cty </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Type </td>
   <td style="text-align:right;"> OLS linear regression </td>
  </tr>
</tbody>
</table> <table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> F(11,218) </td>
   <td style="text-align:right;"> 61.37 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> R² </td>
   <td style="text-align:right;"> 0.76 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> Adj. R² </td>
   <td style="text-align:right;"> 0.74 </td>
  </tr>
</tbody>
</table> <table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;border-bottom: 0;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Est. </th>
   <th style="text-align:right;"> S.E. </th>
   <th style="text-align:right;"> t val. </th>
   <th style="text-align:right;"> p </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;"> (Intercept) </td>
   <td style="text-align:right;"> 21.37 </td>
   <td style="text-align:right;"> 0.39 </td>
   <td style="text-align:right;"> 54.19 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> cyl6 </td>
   <td style="text-align:right;"> -4.37 </td>
   <td style="text-align:right;"> 0.54 </td>
   <td style="text-align:right;"> -8.07 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> cyl8 </td>
   <td style="text-align:right;"> -8.37 </td>
   <td style="text-align:right;"> 0.67 </td>
   <td style="text-align:right;"> -12.51 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> fwd4wd </td>
   <td style="text-align:right;"> -2.91 </td>
   <td style="text-align:right;"> 0.76 </td>
   <td style="text-align:right;"> -3.83 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> automanual </td>
   <td style="text-align:right;"> 1.45 </td>
   <td style="text-align:right;"> 0.57 </td>
   <td style="text-align:right;"> 2.56 </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> cyl6:fwd4wd </td>
   <td style="text-align:right;"> 0.59 </td>
   <td style="text-align:right;"> 0.96 </td>
   <td style="text-align:right;"> 0.62 </td>
   <td style="text-align:right;"> 0.54 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> cyl8:fwd4wd </td>
   <td style="text-align:right;"> 2.13 </td>
   <td style="text-align:right;"> 0.99 </td>
   <td style="text-align:right;"> 2.15 </td>
   <td style="text-align:right;"> 0.03 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> cyl6:automanual </td>
   <td style="text-align:right;"> -0.76 </td>
   <td style="text-align:right;"> 0.90 </td>
   <td style="text-align:right;"> -0.84 </td>
   <td style="text-align:right;"> 0.40 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> cyl8:automanual </td>
   <td style="text-align:right;"> 0.71 </td>
   <td style="text-align:right;"> 1.18 </td>
   <td style="text-align:right;"> 0.60 </td>
   <td style="text-align:right;"> 0.55 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> fwd4wd:automanual </td>
   <td style="text-align:right;"> -1.66 </td>
   <td style="text-align:right;"> 1.07 </td>
   <td style="text-align:right;"> -1.56 </td>
   <td style="text-align:right;"> 0.12 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> cyl6:fwd4wd:automanual </td>
   <td style="text-align:right;"> 1.29 </td>
   <td style="text-align:right;"> 1.52 </td>
   <td style="text-align:right;"> 0.85 </td>
   <td style="text-align:right;"> 0.40 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;"> cyl8:fwd4wd:automanual </td>
   <td style="text-align:right;"> -1.39 </td>
   <td style="text-align:right;"> 1.76 </td>
   <td style="text-align:right;"> -0.79 </td>
   <td style="text-align:right;"> 0.43 </td>
  </tr>
</tbody>
<tfoot><tr><td style="padding: 0; " colspan="100%">
<sup></sup> Standard errors: OLS</td></tr></tfoot>
</table>


```r
cat_plot(fit3,
         pred = cyl,
         modx = fwd,
         plot.points = T)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-31-1.png" width="90%" style="display: block; margin: auto;" />

```r
#line plots
cat_plot(
    fit3,
    pred = cyl,
    modx = fwd,
    geom = "line",
    point.shape = TRUE,
    # colors = "Set2", # choose color
    vary.lty = TRUE
)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-31-2.png" width="90%" style="display: block; margin: auto;" />

```r


# bar plot
cat_plot(
    fit3,
    pred = cyl,
    modx = fwd,
    geom = "bar",
    interval = T,
    plot.points = TRUE
)
```

<img src="17-moderation_files/figure-html/unnamed-chunk-31-3.png" width="90%" style="display: block; margin: auto;" />

## interactionR package

-   For publication purposes
-   Following
    -   [@knol2012recommendations] for presentation

    -   [@hosmer1992confidence] for confidence intervals based on the delta method

    -   [@zou2008estimation] for variance recovery "mover" method

    -   [@assmann1996confidence] for bootstrapping


```r
install.packages("interactionR")
```

## sjPlot package

-   For publication purposes (recommend, but more advanced)

-   [link](https://strengejacke.github.io/sjPlot/articles/plot_interactions.html)
