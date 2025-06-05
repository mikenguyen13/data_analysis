### Cluster Randomization

Standard randomized controlled trials (RCTs) rely on the **Stable Unit Treatment Value Assumption (SUTVA)**, which has two components:

1.  **No interference**: The potential outcome of one unit is unaffected by the treatment assignment of other units. In other words, a unit's potential outcomes depend only on its own treatment assignment. The potential outcome $Y_i(Z)$ depends only on $Z_i$, not on $Z_j$ for $j \ne i$.
2.  **No hidden versions of treatment**: Each treatment condition is well-defined and consistent across units.

Violations of SUTVA---particularly **interference**---are common in **online marketplaces** (e.g., Airbnb, eBay, Uber, Etsy), where agents interact strategically and share common resources like customer attention and platform visibility. More specifically, interference occurs when $Y_i(Z)$ depends on the treatment status of others, i.e., $Y_i(Z_1, ..., Z_n)$. In such settings, **interference bias** distorts causal estimates, often misleading decision-makers.

In interconnected digital environments, agents (sellers, listings, users) affect each other through:

-   Algorithmic rankings
-   Pricing competition
-   Substitution behavior among buyers

> **Example**: If Airbnb treats a host by recommending a price increase, nearby untreated hosts may adjust their own pricing in response. Buyers also respond based on all available listings. Hence, observed outcomes reflect **systemic responses** rather than isolated treatment effects.

This interconnectedness violates the **no interference** condition, causing the estimated **Total Average Treatment Effect (TATE)** to be biased:

$$
\text{TATE} = \mathbb{E}[Y_i(1) - Y_i(0)]
$$

Estimating TATE via **individual-level randomization** assumes SUTVA. However, when interference exists, this approach can overestimate or underestimate true effects depending on the nature and direction of spillovers.

Interference in marketplaces arises from several mechanisms:

-   **Demand-Side Spillovers**: Buyers shift demand across multiple listings. For example, improving one listing's appeal diverts attention from others.
-   **Supply-Side Spillovers**: Sellers observe each other's strategies (e.g., price changes) and respond competitively.
-   **Platform-Level Feedback**: Treatment effects propagate via platform algorithms, such as dynamic re-ranking or recommendation engines.

These interactions violate unit independence and affect not only outcomes but also treatment uptake and exposure.

@blake2014marketplace (eBay) formalized this dynamic as **test--control interference**: treated sellers attract buyers away from control sellers, thus **inflating** estimated gains. @fradkin2015airbnb simulated these dynamics in Airbnb and showed that **overstatements of treatment effects can exceed 100%** under realistic assumptions.

To mitigate interference bias, researchers increasingly use **cluster randomization**, where treatment is assigned at the level of **clusters** (e.g., neighborhoods, product categories, or seller types) rather than individual units. This reduces spillovers between treated and control groups by aligning treatment boundaries with natural interaction clusters.

-   In **social networks**, graph-cluster randomization (GCR) partitions the network to minimize cross-group links.
-   In **marketplaces**, meta-experiments (experiments over experiments) can assess the extent of interference and evaluate cluster-based designs.

Cluster randomization aims to preserve causal identification by reducing contamination, especially when direct or indirect interactions are likely.

------------------------------------------------------------------------

#### Modeling Linear Interference

Following @eckles2021bias, consider the potential outcome for seller $i$ as:

$$
Y_i(Z) = \alpha_i + \beta Z_i + \gamma \rho_i + \varepsilon_i
$$

-   $\beta$ is the direct treatment effect.
-   $\gamma$ is the spillover (indirect) effect.
-   $\rho_i = \frac{1}{|N_i|} \sum_{j \in N_i} Z_j$ is the fraction of treated neighbors.

This model implies:

-   When $\gamma \ne 0$, there is interference.
-   If $\beta$ and $\gamma$ have opposite signs, interference can *inflate or deflate* TATE estimates.

In matrix form, define an **interference matrix** $B$ such that:

$$
\mathbb{E}[Y_i(Z)] = \alpha_i + \sum_{j} B_{ij} Z_j
$$

-   $B_{ij}$ quantifies the extent to which treatment on unit $j$ affects unit $i$.
-   Diagonal elements capture direct effects.
-   Off-diagonal elements capture interference.

This formulation allows for structural modeling of interference.

#### Cluster Randomization: Design and Theory

**Cluster Randomization** assigns entire clusters (groups) of units to treatment or control, rather than randomizing at the individual level.

Rationale:

-   Units within clusters likely interfere with each other.

-   By treating clusters homogeneously, we contain interference within groups.

This reduces bias in estimating average treatment effects.

@holtz2025reducing extend @eckles2021bias 's result and demonstrate:

> For a linear model with uniform-sign $B_{ii}$ and $B_{ij}$ (diagonal and off-diagonal entries), cluster randomization **always** reduces interference bias relative to individual-level randomization.

Formally, let $\hat{\tau}_{\text{ind}}$ and $\hat{\tau}_{\text{cr}}$ denote TATE estimates under individual and cluster randomization, respectively. Then:

$$ |\mathbb{E}[\hat{\tau}_{\text{cr}} - \tau]| \leq |\mathbb{E}[\hat{\tau}_{\text{ind}} - \tau]| $$

This result holds regardless of whether spillovers are positive or negative.

------------------------------------------------------------------------

Let $C(i)$ denote the cluster of unit $i$. Then **cluster quality** is defined as:

$$ Q_C(B) = \sum_{i,j} B_{ij} \cdot 1\{C(i) \ne C(j)\} $$

-   Lower $Q_C(B)$ implies less across-cluster interference, and hence, better clustering.
-   In practice, $B$ is unobservable, so a proxy matrix $P$ (e.g., co-search or co-view behavior) is used.

#### Graph-Cluster Randomization

**GCR** partitions a social or economic network into clusters and randomizes treatment at the cluster level. This reduces mixed-treatment neighborhoods, a primary source of interference.

> **Key Result**: @ugander2013graph showed that GCR yields unbiased Horvitz--Thompson estimators of treatment effects under certain exposure models: $$
> \hat{\tau}_{HT} = \sum_{i \in T} \frac{Y_i}{\pi_i} - \sum_{i \in C} \frac{Y_i}{1 - \pi_i}
> $$ where $\pi_i$ is the probability that unit $i$ is treated, accounting for the randomization design.

@eckles2021bias derived analytic expressions for **variance inflation** due to clustering: $$
\text{Var}(\hat{\tau}_{CR}) \propto \rho \cdot \frac{\sigma^2}{n_c}
$$

where $\rho$ is the intra-cluster correlation and $n_c$ is the number of clusters. Pre-stratification, regression adjustment, or increasing cluster count can partially mitigate this variance growth.

------------------------------------------------------------------------

#### Case Study: The Airbnb Meta-Experiment

We now turn to a case study: a large-scale **meta-experiment** conducted by @holtz2025reducing on Airbnb to empirically measure the extent of interference bias and evaluate the effectiveness of cluster randomization in online marketplaces.

Airbnb is a global two-sided platform with over 5 million listings, where guests search for accommodations through a **ranked interface** based on location, price, and historical booking data. Importantly, guests often substitute between similar listings, and hosts dynamically adjust their prices and availability based on market conditions. These interactions create a fertile ground for interference, violating the SUTVA assumptions that underpin most experimental designs.

To mitigate this, Airbnb engineers constructed high-quality clusters based on guest browsing behavior. Each listing was first embedded into a 16-dimensional vector space using co-click data from guest sessions. A **recursive partitioning tree** was then applied to cluster these vectors, ensuring that at least 90% of listings within a node shared common neighbors. The resulting \~15,000 clusters had a median size of 150 listings, and the within-cluster substitution rate exceeded 95%, indicating effective isolation. Hold-out data confirmed that **cross-cluster substitution** was minimal, validating the clustering procedure.

Using this clustering, Airbnb conducted two major platform-wide experiments. In both cases, 75% of clusters were randomized at the **cluster level**, while 25% retained standard **Bernoulli (individual-level) randomization**.

The experimental sample included over **2 million listings worldwide**. For the guest-fee intervention, the estimated TATE was −0.207 bookings per listing under individual-level randomization, but only −0.139 under cluster randomization. This implies that interference inflated the naive estimate by **32.6%**.

To formally quantify this distortion, @holtz2025reducing defined:

-   $\hat{\beta}$: TATE under cluster randomization
-   $\hat{\nu}$: Difference between individual-level and cluster-level estimates

The **interference bias ratio** was calculated as:

$$
\text{Bias Ratio} = \frac{\hat{\nu}}{\hat{\beta} + \hat{\nu}} \approx 19.76\%
$$

Bias levels varied by geography and market dynamics:

-   **Demand-constrained areas** (e.g., New York in peak season): Bias reached 28.65%
-   **Supply-constrained areas**: Bias was lower, around 12.05%
-   **Higher cluster quality** (i.e., better separation in embedding space) was associated with greater bias reduction

For the second intervention---pricing recommendations via a machine learning algorithm---the cluster-level estimate was not statistically significant (−0.051 bookings per listing), compared to −0.106 from Bernoulli randomization. This null result was largely attributed to limited statistical power.

> A **post-hoc power analysis** indicated that to detect the same level of bias with 80% power, Airbnb would have needed **3.5 times more clusters**, underscoring the high cost of identifying and correcting for interference in real-world platforms.

This case study illustrates both the **prevalence of interference bias** in online marketplaces and the **potential of cluster randomization** to mitigate it---provided clusters are intelligently constructed and contextually grounded.

------------------------------------------------------------------------

## 5 Methodological Blueprint for Applied Researchers

### 5.1 Diagnosing Interference

-   **Step 1**: Theorize likely spillover channels (e.g., ranking algorithms, price comparisons).
-   **Step 2**: Conduct a meta-experiment with dual randomization and compare TATEs.
-   **Step 3**: Quantify exposure probabilities using network or item similarity matrices.

### 5.2 Cluster Construction Algorithm

``` text
Input: substitution matrix S (from clicks/bookings)

Steps:
1. Symmetrize: W = (S + Sᵀ)/2
2. Compute normalized graph Laplacian L
3. Embed nodes via top-k eigenvectors of L
4. Apply k-means clustering (k via gap statistic)
5. Merge tiny clusters (<m listings) if needed
```

Evaluate using **neighbor purity**: proportion of nearest neighbors within the same cluster. Airbnb threshold: ≥0.9 → \<5% cross-cluster exposure.

### 5.3 Analysis Plan

-   Use **cluster-robust standard errors** or **Horvitz--Thompson weights**.
-   Adjust for **intra-cluster correlation** using methods from Eckles et al.
-   Apply **OLS with pre-stratification** to improve power.

### 5.4 Practical Considerations

| Challenge              | Recommended Approach                                         |
|------------------------|--------------------------------------------------------------|
| Power loss             | Increase experiment duration; pre-stratify clusters          |
| Uneven traffic         | Weight treatment assignment by expected demand               |
| Two-sided interactions | Analyze both market sides separately and track cross-effects |

------------------------------------------------------------------------

## 6 Extensions and Research Opportunities

1.  **Two-Stage Designs**: Separate between-cluster assignment (e.g., policy) and within-cluster nudges (e.g., messages).
2.  **Dynamic Interference Models**: Incorporate time-varying exposure using synthetic controls.
3.  **Adaptive Clustering**: Continuously update embeddings as listing inventory changes.
4.  **General Equilibrium Simulation**: Embed results in a structural model to forecast long-run impacts.

------------------------------------------------------------------------

## 7 Pre-Experiment Checklist

1.  Map out unit interactions and plausible spillover networks.
2.  Collect high-resolution substitution logs or co-click data.
3.  Generate clusters and validate using exposure metrics.
4.  Design meta-experiment with ≥25% traffic allocated to Bernoulli randomization.
5.  Write and register a pre-analysis plan (estimators, heterogeneity checks, etc.).
6.  Monitor compliance; re-balance if significant imbalance arises.
7.  Report bias estimates and effect sizes from both designs.
8.  Discuss implications for external validity and platform-wide rollout.

------------------------------------------------------------------------

# Chapter: Cluster Randomization and Interference Bias

## 6. Implementation: Practical Guidelines

### 6.1 Constructing Clusters

Use one or more of the following:

-   Geographic proximity.
-   Embedding techniques (e.g., Word2Vec on search sequences).
-   Co-view or co-search frequency.
-   Pricing similarity or estimated cross-price elasticities.

### 6.2 Evaluating Cluster Quality

Use proxy interference matrix \$P\$ (e.g., PDP view co-occurrence), then:


``` r
Q <- function(B, clusters) {
  sum(sapply(1:nrow(B), function(i) {
    sum(B[i, clusters != clusters[i]])
  }))
}
```

Lower $Q$ indicates better clustering.

### 6.3 Trade-offs

| Factor                  | Individual Randomization | Cluster Randomization       |
|-------------------------|--------------------------|-----------------------------|
| Bias (Interference)     | High                     | Low                         |
| Power                   | High                     | Lower                       |
| Complexity              | Low                      | Moderate                    |
| Implementation overhead | Minimal                  | Requires new infrastructure |


``` r
set.seed(123)
n <- 1000
clusters <- rep(1:100, each=10)
Z <- rbinom(n, 1, 0.5)
B <- matrix(runif(n^2, 0, 0.1), n, n)
diag(B) <- runif(n, 0.5, 1)

# Simulate outcome
Y <- sapply(1:n, function(i) sum(B[i,] * Z)) + rnorm(n)

# Estimate TATE under individual randomization
t_ind <- mean(Y[Z==1]) - mean(Y[Z==0])

# Estimate TATE under cluster randomization
cluster_Z <- tapply(Z, clusters, mean)
cluster_Y <- tapply(Y, clusters, mean)
t_cr <- mean(cluster_Y[cluster_Z > 0.5]) - mean(cluster_Y[cluster_Z <= 0.5])

t_ind
#> [1] 0.7931955
t_cr
#> [1] 0.3104478
```

Observe that $t_{\text{ind}}$ is inflated relative to $t_{\text{cr}}$.

## 8. Limitations and Open Questions

-   **Loss of precision**: Especially problematic in low-interference settings.
-   **Complexity of clustering**: No universal optimal method exists.
-   **Temporal dynamics**: Cluster randomization doesn't inherently address time-based spillovers.
-   **Heterogeneous effects**: Bias correction may vary by treatment strength or context.
