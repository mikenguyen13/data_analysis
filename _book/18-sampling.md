# Sampling

## Simple Sampling

Simple (random) Sampling


```r
library(dplyr)
iris_df <- iris
set.seed(1)
sample_n(iris_df, 10)
#>    Sepal.Length Sepal.Width Petal.Length Petal.Width    Species
#> 1           5.8         2.7          4.1         1.0 versicolor
#> 2           6.4         2.8          5.6         2.1  virginica
#> 3           4.4         3.2          1.3         0.2     setosa
#> 4           4.3         3.0          1.1         0.1     setosa
#> 5           7.0         3.2          4.7         1.4 versicolor
#> 6           5.4         3.0          4.5         1.5 versicolor
#> 7           5.4         3.4          1.7         0.2     setosa
#> 8           7.6         3.0          6.6         2.1  virginica
#> 9           6.1         2.8          4.7         1.2 versicolor
#> 10          4.6         3.4          1.4         0.3     setosa
```


```r
library(sampling)
# set unique id number for each row 
iris_df$id = 1:nrow(iris_df)

# Simple random sampling with replacement
srswor(10, length(iris_df$id))
#>   [1] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1
#>  [38] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
#>  [75] 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0
#> [112] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#> [149] 0 0

# Simple random sampling without replacement (sequential method)
srswor1(10, length(iris_df$id))
#>   [1] 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#>  [38] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#>  [75] 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#> [112] 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0
#> [149] 0 0

# Simple random sampling with replacement
srswr(10, length(iris_df$id))
#>   [1] 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 1 0 0 0 0
#>  [38] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#>  [75] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
#> [112] 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
#> [149] 0 0
```


```r
library(survey)
data("api")
srs_design <- svydesign(data = apistrat,
                        weights = ~pw, 
                        fpc = ~fpc, 
                        id = ~1)
```


```r
library(sampler)
rsamp(albania,
      n = 260,
      over = 0.1, # desired oversampling proportion
      rep = F)
```

Identify missing points between sample and collected data


```r
alsample <- rsamp(df = albania, 544)
alreceived <- rsamp(df = alsample, 390)
rmissing(sampdf = alsample,
         colldf = alreceived,
         col_name = qvKod)
```

## Stratified Sampling

A stratum is a subset of the population that has at least one common characteristic.

Steps:

1.  Identify relevant stratums and their representation in the population.
2.  Randomly sample to select a sufficient number of subjects from each stratum.

Stratified sampling reduces sampling error.


```r
library(dplyr)
# by number of rows
sample_iris <- iris %>%
    group_by(Species) %>%
    sample_n(5)
sample_iris
#> # A tibble: 15 × 5
#> # Groups:   Species [3]
#>    Sepal.Length Sepal.Width Petal.Length Petal.Width Species   
#>           <dbl>       <dbl>        <dbl>       <dbl> <fct>     
#>  1          4.4         3            1.3         0.2 setosa    
#>  2          5.2         3.5          1.5         0.2 setosa    
#>  3          5.1         3.8          1.5         0.3 setosa    
#>  4          5.2         3.4          1.4         0.2 setosa    
#>  5          4.5         2.3          1.3         0.3 setosa    
#>  6          5.5         2.5          4           1.3 versicolor
#>  7          7           3.2          4.7         1.4 versicolor
#>  8          6.7         3            5           1.7 versicolor
#>  9          6.1         2.9          4.7         1.4 versicolor
#> 10          5.5         2.4          3.8         1.1 versicolor
#> 11          6.4         2.7          5.3         1.9 virginica 
#> 12          6.4         2.8          5.6         2.1 virginica 
#> 13          6.4         3.2          5.3         2.3 virginica 
#> 14          6.8         3.2          5.9         2.3 virginica 
#> 15          7.2         3.6          6.1         2.5 virginica

# by fraction
sample_iris <- iris %>%
    group_by(Species) %>%
    sample_frac(size = .15)
sample_iris
#> # A tibble: 24 × 5
#> # Groups:   Species [3]
#>    Sepal.Length Sepal.Width Petal.Length Petal.Width Species   
#>           <dbl>       <dbl>        <dbl>       <dbl> <fct>     
#>  1          5.5         4.2          1.4         0.2 setosa    
#>  2          5           3            1.6         0.2 setosa    
#>  3          5.2         4.1          1.5         0.1 setosa    
#>  4          4.6         3.1          1.5         0.2 setosa    
#>  5          5.1         3.7          1.5         0.4 setosa    
#>  6          4.8         3.4          1.9         0.2 setosa    
#>  7          5.1         3.3          1.7         0.5 setosa    
#>  8          5.5         3.5          1.3         0.2 setosa    
#>  9          5           2.3          3.3         1   versicolor
#> 10          5.6         2.9          3.6         1.3 versicolor
#> # ℹ 14 more rows
```


```r
library(sampler)
# Stratified sample using proportional allocation without replacement
ssamp(df=albania, n=360, strata=qarku, over=0.1)
#> # A tibble: 395 × 45
#>    qarku  Q_ID bashkia   BAS_ID zaz   njesiaAdministrative COM_ID qvKod zgjedhes
#>    <fct> <int> <fct>      <int> <fct> <fct>                 <int> <fct>    <int>
#>  1 Berat     1 Berat         11 ZAZ … "Berat "               1101 "\"3…      558
#>  2 Berat     1 Berat         11 ZAZ … "Berat "               1101 "\"3…      815
#>  3 Berat     1 Berat         11 ZAZ … "Sinje"                1108 "\"3…      419
#>  4 Berat     1 Kucove        13 ZAZ … "Lumas"                1104 "\"3…      237
#>  5 Berat     1 Kucove        13 ZAZ … "Kucove"               1201 "\"3…      562
#>  6 Berat     1 Skrapar       17 ZAZ … "Corovode"             1303 "\"3…      829
#>  7 Berat     1 Berat         11 ZAZ … "Roshnik"              1107 "\"3…      410
#>  8 Berat     1 Ura Vajg…     19 ZAZ … "Ura Vajgurore"        1110 "\"3…      708
#>  9 Berat     1 Kucove        13 ZAZ … "Perondi"              1203 "\"3…      835
#> 10 Berat     1 Kucove        13 ZAZ … "Kucove"               1201 "\"3…      907
#> # ℹ 385 more rows
#> # ℹ 36 more variables: meshkuj <int>, femra <int>, totalSeats <int>,
#> #   vendndodhja <fct>, ambienti <fct>, totalVoters <int>, femVoters <int>,
#> #   maleVoters <int>, unusedBallots <int>, damagedBallots <int>,
#> #   ballotsCast <int>, invalidVotes <int>, validVotes <int>, lsi <int>,
#> #   ps <int>, pkd <int>, sfida <int>, pr <int>, pd <int>, pbdksh <int>,
#> #   adk <int>, psd <int>, ad <int>, frd <int>, pds <int>, pdiu <int>, …
```

Identify number of missing points by strata between sample and collected data


```r
alsample <- rsamp(df = albania, 544)
alreceived <- rsamp(df = alsample, 390)
smissing(
    sampdf = alsample,
    colldf = alreceived,
    strata = qarku,
    col_name = qvKod
)
```

## Unequal Probability Sampling


```r
UPbrewer()
UPmaxentropy()
UPmidzuno()
UPmidzunopi2()
UPmultinomial()
UPpivotal()
UPrandompivotal()
UPpoisson()
UPsampford()
UPsystematic()
UPrandomsystematic()
UPsystematicpi2()
UPtille()
UPtillepi2()
```

## Balanced Sampling

-   Purpose: to get the same means in the population and the sample for all the auxiliary variables

-   Balanced sampling is different from purposive selection

Balancing equations

$$
\sum_{k \in S} \frac{\mathbf{x}_k}{\pi_k} = \sum_{k \in U} \mathbf{x}_k
$$

where $\mathbf{x}_k$ is a vector of auxiliary variables

### Cube

-   flight phase

-   landing phase


```r
samplecube()
fastflightcube()
landingcube()

```

### Stratification

-   Try to replicate the population based on the original multivariate histogram


```r
library(survey)
data("api")
srs_design <- svydesign(data = apistrat,
                        weights = ~pw, 
                        fpc = ~fpc, 
                        strata = ~stype,
                        id = ~1)
```


```r
balancedstratification()

```

### Cluster


```r
library(survey)
data("api")
srs_design <- svydesign(data = apiclus1,
                        weights = ~pw, 
                        fpc = ~fpc, 
                        id = ~dnum)
```


```r
balancedcluster()
```

### Two-stage


```r
library(survey)
data("api")
srs_design <- svydesign(data = apiclus2, 
                        fpc = ~fpc1 + fpc2, 
                        id = ~ dnum + snum)
```


```r
balancedtwostage()
```
