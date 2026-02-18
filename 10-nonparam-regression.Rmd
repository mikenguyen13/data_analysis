# Nonparametric Regression {#sec-nonparametric-regression}

This chapter surveys regression techniques that relax functional-form assumptions. Beginning with kernel and local-polynomial estimators, we derive bias-variance trade-offs and bandwidth-selection criteria. We then explore splines, generalized additive models, regression trees, random forests, and wavelet regression, emphasizing their interpretability and robustness. Multivariate nonparametrics are introduced through radial-basis functions. Confidence-interval construction via asymptotics and bootstrap methods is detailed, and a forward-looking conclusion discusses how nonparametric ideas underpin modern machine-learning algorithms, reinforcing the evolving landscape of regression analysis.

**Nonparametric regression** refers to a class of regression techniques that do not assume a specific functional form (e.g., linear, polynomial of fixed degree) for the relationship between a predictor $x \in \mathbb{R}$ (or $\mathbf{x} \in \mathbb{R}^p$) and a response variable $y \in \mathbb{R}$. Instead, nonparametric methods aim to estimate this relationship directly from the data, allowing the data to "speak for themselves."

In a standard regression framework, we have a response variable $Y$ and one or more predictors $\mathbf{X} = (X_1, X_2, \ldots, X_p)$. Let us start with a univariate setting for simplicity. We assume the following model:

$$
Y = m(x) + \varepsilon,
$$

where:

-   $m(x) = \mathbb{E}[Y \mid X = x]$ is the **regression function** we aim to estimate,
-   $\varepsilon$ is a random **error term** (noise) with $\mathbb{E}[\varepsilon \mid X = x] = 0$ and constant variance $\operatorname{Var}(\varepsilon) = \sigma^2$.

In **parametric regression** (e.g., [Linear Regression]), we might assume $m(x)$ has a specific form, such as:

$$
m(x) = \beta_0 + \beta_1 x + \cdots + \beta_d x^d,
$$

where $\beta_0, \beta_1, \ldots, \beta_d$ are parameters to be estimated. In contrast, **nonparametric regression** relaxes this assumption and employs methods that can adapt to potentially complex shapes in $m(x)$ without pre-specifying its structure.



> **This chapter is fully available in the published Springer volumes.**\
> The online preview is limited per publisher guidelines.

To access the complete content, purchase the book on Springer:

| Vol. | Title                                     | Link                                       |
|---------------|---------------------------|--------------------|
| 1    | *Foundations of Data Analysis*            | [Buy on Springer](https://tidd.ly/4oL3N2X) |
| 2    | *Regression Techniques for Data Analysis* | [Buy on Springer](https://tidd.ly/47PJ7kB) |
| 3    | *Advanced Modeling and Data Challenges*   | [Buy on Springer](https://tidd.ly/3JrB3xm) |
| 4    | *Experimental Design*                     | [Buy on Springer](https://tidd.ly/4oFridQ) |
