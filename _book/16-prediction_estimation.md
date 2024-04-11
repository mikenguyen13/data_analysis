# Prediction and Estimation

Prediction and Estimation (or Causal Inference) serve distinct roles in understanding and modeling data.

## Prediction

-   **Definition**: Prediction, denoted as $\hat{y}$, is about creating an algorithm for predicting the outcome variable $y$ from predictors $x$.

-   **Goal**: The primary goal is loss minimization, aiming for model accuracy on unseen data:

    $$
    \hat{f} \approx \min E_{(y,x)} L(f(x), y)
    $$

-   **Applications in Economics**:

    -   Measure variables.
    -   Embed prediction tasks within parameter estimation or treatment effects.
    -   Control for observed confounders.

## Parameter Estimation

-   **Definition**: Parameter estimation, represented by $\hat{\beta}$, focuses on estimating the relationship between $y$ and $x$.

-   **Goal**: The aim is consistency, ensuring that models perform well on the training data:

    $$
    E[\hat{f}] = f
    $$

-   **Challenges**:

    -   High-dimensional spaces can lead to covariance among variables and multicollinearity.
    -   This leads to the bias-variance tradeoff [@hastie2009elements].

## Causation versus Prediction

Understanding the relationship between causation and prediction is crucial in statistical modeling.

Let $Y$ be an outcome variable dependent on $X$, and our aim is to manipulate $X$ to maximize a payoff function $\pi(X, Y)$ [@kleinberg2015prediction]. The decision on $X$ hinges on:

$$ 
\begin{aligned}
\frac{d\pi(X, Y)}{d X} &= \frac{\partial \pi}{\partial X} (Y) + \frac{\partial \pi}{\partial Y} \frac{\partial Y}{\partial X} \\
&= \frac{\partial \pi}{\partial X} \text{(Prediction)} + \frac{\partial \pi}{\partial Y} \text{(Causation)} 
\end{aligned}
$$

Empirical work is essential for estimating the derivatives in this equation:

-   $\frac{\partial Y}{\partial X}$ is required for causal inference to determine $X$'s effect on $Y$,

-   $\frac{\partial \pi}{\partial X}$ is required for prediction of $Y$.

![(SICSS 2018 - Sendhil Mullainathan's presentation slide)](images/prediction_causation.PNG){style="display: block; margin: 1em auto" width="600" height="350"}
