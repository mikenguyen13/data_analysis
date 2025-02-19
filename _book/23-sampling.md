# Sampling

Sampling allows us to draw conclusions about a population without analyzing every individual in it. In business applications---such as marketing research, and financial forecasting---sampling enables efficient decision-making while reducing costs and effort.

------------------------------------------------------------------------

## Population and Sample

This is a refresher on terminology regarding sampling.

-   **Population** ($N$): The complete set of all elements under study.
-   **Sample** ($n$): A subset of the population selected for analysis.
-   **Parameter**: A numerical measure that describes a characteristic of a population (e.g., population mean $\mu$, population variance $\sigma^2$).
-   **Statistic**: A numerical measure computed from a sample, used to estimate a population parameter (e.g., sample mean $\bar{x}$, sample variance $s^2$).

A well-chosen sample ensures that results generalize to the population, reducing **sampling bias**.

------------------------------------------------------------------------

## Sampling Techniques

### Probability Sampling

Probability sampling methods ensure that every element in the population has a known, nonzero probability of being selected. These methods are preferred in inferential statistics since they allow for the estimation of sampling error.

### Simple Random Sampling {#sec-simple-random-sampling}

Simple Random Sampling (SRS) ensures that every element in the population has an **equal chance** of being selected. This can be done **with replacement** or **without replacement**, impacting whether an element can be chosen more than once.

Below is an example of drawing a simple random sample without replacement from a population of 100 elements:


```r
set.seed(123)
population <- 1:100  # A population of 100 elements
sample_srs <- sample(population, size = 10, replace = FALSE)
sample_srs
#>  [1] 31 79 51 14 67 42 50 43 97 25
```

**Advantages:**

-   Simple and easy to implement

-   Ensures unbiased selection

**Disadvantages:**

-   May not represent subgroups well, especially in heterogeneous populations

-   Requires access to a complete list of the population

#### Using `dplyr`

The `sample_n()` function in `dplyr` allows for simple random sampling from a dataset:


```r
library(dplyr)
iris_df <- iris
set.seed(1)
sample_n(iris_df, 5)  # Randomly selects 5 rows from the iris dataset
#>   Sepal.Length Sepal.Width Petal.Length Petal.Width    Species
#> 1          5.8         2.7          4.1         1.0 versicolor
#> 2          6.4         2.8          5.6         2.1  virginica
#> 3          4.4         3.2          1.3         0.2     setosa
#> 4          4.3         3.0          1.1         0.1     setosa
#> 5          7.0         3.2          4.7         1.4 versicolor
```

#### Using the `sampling` Package

The `sampling` package provides functions for random sampling **with** and **without** replacement.


```r
library(sampling)
# Assign a unique ID to each row in the dataset
iris_df$id <- 1:nrow(iris_df)

# Simple random sampling without replacement
srs_sample <- srswor(10, length(iris_df$id))  
# srs_sample

# Simple random sampling with replacement
srs_sample_wr <- srswr(10, length(iris_df$id))
# srs_sample_wr
```

#### Using the `sampler` Package

The `sampler` package provides additional functionality, such as **oversampling** to account for non-response.


```r
library(sampler)
rsamp(albania, n = 260, over = 0.1, rep = FALSE)
```

#### Handling Missing Data in Sample Collection

To compare a sample with received (collected) data and identify missing elements:


```r
alsample <- rsamp(df = albania, 544)  # Initial sample
alreceived <- rsamp(df = alsample, 390)  # Collected data
rmissing(sampdf = alsample, colldf = alreceived, col_name = qvKod)
```

### Stratified Sampling {#sec-stratified-sampling}

Stratified sampling involves dividing the population into distinct **strata** based on a characteristic (e.g., age, income level, region). A **random sample** is then drawn from each stratum, often in proportion to its size within the population. This method ensures that all subgroups are adequately represented, improving the precision of estimates.

------------------------------------------------------------------------

The following example demonstrates stratified sampling where individuals belong to three different groups **(A, B, C)**, and a random sample is drawn from each.


```r
library(dplyr)

set.seed(123)
data <- data.frame(
  ID = 1:100,
  Group = sample(c("A", "B", "C"), 100, replace = TRUE)
)

# Stratified random sampling: selecting 10 elements per group
stratified_sample <- data %>%
  group_by(Group) %>%
  sample_n(size = 10)

# stratified_sample
```

**Advantages:**

-   Ensures representation of all subgroups

-   More precise estimates compared to [Simple Random Sampling](#sec-simple-random-sampling)

-   Reduces sampling error by accounting for population variability

**Disadvantages:**

-   Requires prior knowledge of population strata

-   More complex to implement than [SRS](#sec-simple-random-sampling)

#### Using `dplyr` for Stratified Sampling

**Sampling by Fixed Number of Rows**

Here, we extract **5 random observations** from each species in the `iris` dataset.


```r
library(dplyr)

set.seed(123)
sample_iris <- iris %>%
  group_by(Species) %>%
  sample_n(5)  # Selects 5 samples per species

# sample_iris
```

**Sampling by Fraction of Each Stratum**

Instead of selecting a fixed number, we can sample **15% of each species**:


```r
set.seed(123)
sample_iris <- iris %>%
  group_by(Species) %>%
  sample_frac(size = 0.15)  # Selects 15% of each species

# sample_iris
```

#### Using the `sampler` Package

The `sampler` package allows stratified sampling with proportional allocation:


```r
library(sampler)

# Stratified sample using proportional allocation without replacement
ssamp(df = albania, n = 360, strata = qarku, over = 0.1)
```

#### Handling Missing Data in Stratified Sampling

To identify the number of missing values **by stratum** between the initial sample and the collected data:


```r
alsample <- rsamp(df = albania, 544)  # Initial sample
alreceived <- rsamp(df = alsample, 390)  # Collected data

smissing(
  sampdf = alsample,
  colldf = alreceived,
  strata = qarku,   # Strata column
  col_name = qvKod  # Column for checking missing values
)
```

### Systematic Sampling

Selects every $k$th element after a random starting point.


```r
k <- 10  # Select every 10th element
start <- sample(1:k, 1)  # Random start point
sample_systematic <- population[seq(start, length(population), by = k)]
```

**Advantages:**

-   Simple to implement

-   Ensures even coverage

**Disadvantages:**

-   If data follows a pattern, bias may be introduced

### Cluster Sampling

Instead of selecting individuals, entire clusters (e.g., cities, schools) are randomly chosen, and all members of selected clusters are included.


```r
data$Cluster <- sample(1:10, 100, replace = TRUE)  # Assign 10 clusters
chosen_clusters <- sample(1:10, size = 3)  # Select 3 clusters
cluster_sample <- filter(data, Cluster %in% chosen_clusters)
```

**Advantages:**

-   Cost-effective when the population is large

-   Useful when the population is naturally divided into groups

**Disadvantages:**

-   Higher variability

-   Risk of unrepresentative clusters

### Non-Probability Sampling

These methods do not give all elements a known probability of selection. They are used in exploratory research but are not suitable for making formal statistical inferences.

### Convenience Sampling

Selecting individuals who are easiest to reach (e.g., mall surveys).

**Pros:** Quick and inexpensive\
**Cons:** High risk of bias, not generalizable

### Quota Sampling

Similar to stratified sampling but non-random.

**Pros:** Ensures subgroup representation\
**Cons:** Subject to selection bias

### Snowball Sampling

Used for hard-to-reach populations (e.g., networking through referrals).

**Pros:** Useful when the population is unknown\
**Cons:** High bias, dependency on initial subjects

## Unequal Probability Sampling {#sec-unequal-probability-sampling}

Unequal probability sampling assigns different selection probabilities to elements in the population. This approach is often used when certain units are more important, have higher variability, or require higher precision in estimation.

Common methods for unequal probability sampling include:

-   **Probability Proportional to Size (PPS)**: Selection probability is proportional to a given auxiliary variable (e.g., revenue, population size).

-   **Poisson Sampling**: Independent selection of each unit with a given probability.

-   **Systematic Sampling with Unequal Probabilities**: Uses a systematic approach while ensuring different probabilities.

The following functions from the `sampling` package implement various unequal probability sampling methods:


```r
library(sampling)

# Different methods for unequal probability sampling
UPbrewer()         # Brewer's method
UPmaxentropy()     # Maximum entropy method
UPmidzuno()        # Midzuno’s method
UPmidzunopi2()     # Midzuno’s method with second-order inclusion probabilities
UPmultinomial()    # Multinomial method
UPpivotal()        # Pivotal method
UPrandompivotal()  # Randomized pivotal method
UPpoisson()        # Poisson sampling
UPsampford()       # Sampford’s method
UPsystematic()     # Systematic sampling
UPrandomsystematic() # Randomized systematic sampling
UPsystematicpi2()  # Systematic sampling with second-order probabilities
UPtille()          # Tillé’s method
UPtillepi2()       # Tillé’s method with second-order inclusion probabilities
```

Each of these methods has specific use cases and theoretical justifications. For example:

-   **Poisson sampling** allows flexible control over sample size but may lead to variable sample sizes.

-   **Systematic sampling** is useful when population elements are arranged in a meaningful order.

-   **Tillé's method** ensures better control over the sample's second-order inclusion probabilities.

## Balanced Sampling {#sec-balanced-sampling}

Balanced sampling ensures that the **means of auxiliary variables** in the sample match those in the population. This method improves estimation efficiency and reduces variability without introducing bias.

Balanced sampling **differs from purposive selection** because it still involves **randomization**, ensuring statistical validity.

The **balancing equation** is given by: $$
\sum_{k \in S} \frac{\mathbf{x}_k}{\pi_k} = \sum_{k \in U} \mathbf{x}_k
$$ where:

-   $\mathbf{x}_k$ is a vector of auxiliary variables (e.g., income, age, household size).

-   $\pi_k$ is the inclusion probability of unit $k$.

-   $S$ is the sample, and $U$ is the population.

This ensures that the total weighted sum of auxiliary variables in the sample matches the total sum in the population.

### Cube Method for Balanced Sampling

The **Cube Method** is a widely used approach for balanced sampling, consisting of two phases:

1.  **Flight Phase**: Ensures initial balance on auxiliary variables.

2.  **Landing Phase**: Adjusts the sample to meet constraints while keeping randomness.


```r
library(sampling)

# Cube method functions
samplecube()       # Standard cube method
fastflightcube()   # Optimized flight phase
landingcube()      # Landing phase method
```

### Balanced Sampling with Stratification

Stratification attempts to **replicate the population structure** in the sample by preserving the original **multivariate histogram**.


```r
library(survey)
data("api")

# Stratified design with proportional allocation
srs_design <- svydesign(data = apistrat,
                        weights = ~pw, 
                        fpc = ~fpc, 
                        strata = ~stype,
                        id = ~1)
```

An additional method for balanced stratification is:


```r
balancedstratification()
```

This method ensures that within each stratum, the sample retains the original proportions of auxiliary variables.

### Balanced Sampling in Cluster Sampling

Cluster sampling involves selecting **entire groups (clusters)** instead of individual units. A balanced approach ensures that the sampled clusters **preserve the overall distribution of auxiliary variables**.


```r
library(survey)
data("api")

# Cluster sampling design
srs_design <- svydesign(data = apiclus1,
                        weights = ~pw, 
                        fpc = ~fpc, 
                        id = ~dnum)
```

For explicitly balanced cluster sampling:


```r
balancedcluster()

```

This method ensures that the **cluster-level characteristics** of the sample match those of the population.

### Balanced Sampling in Two-Stage Sampling

Two-stage sampling first selects primary units (e.g., schools, cities) and then samples within them. A balanced approach ensures **representative selection at both stages**.


```r
library(survey)
data("api")

# Two-stage sampling design
srs_design <- svydesign(data = apiclus2, 
                        fpc = ~fpc1 + fpc2, 
                        id = ~dnum + snum)
```

For explicitly balanced two-stage sampling:


```r
balancedtwostage()
```

This method ensures that **auxiliary variables remain balanced across both selection stages**, reducing variability while maintaining randomness.

## Sample Size Determination

The appropriate sample size depends on the **margin of error**, **confidence level**, and **population variability**. A commonly used formula for estimating the required sample size for a proportion is:

$$
n = \frac{Z^2 p (1 - p)}{E^2}
$$ where:

-   $Z$ is the Z-score corresponding to the confidence level

-   $p$ is the estimated proportion

-   $E$ is the margin of error


```r
z <- qnorm(0.975)  # 95% confidence level
p <- 0.5  # Estimated proportion
E <- 0.05  # 5% margin of error
n <- (z^2 * p * (1 - p)) / (E^2)
ceiling(n)  # Round up to nearest integer
#> [1] 385
```
