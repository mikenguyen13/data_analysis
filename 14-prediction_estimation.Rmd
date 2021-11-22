# Prediction and Estimation

Prediction and Estimation (Inference) have been the two fundamental pillars in statistics.

-   You cannot have both. You can either have high prediction or high estimation.

    -   In prediction, you minimize the loss function.

    -   In estimation, you try to best fit the data. Because the goal of estimation is to best fit the data, you always run the risk of not predicting well.

In high dimension, you always have weak to strong collinearity. Hence, your estimation can be undesirable. And you can't pick which one variable to stay in the model, but all these troubles would not affect your prediction. In Plateau problem

-   If two functions are similar in output space, you can still do prediction, but you can't do estimation because of exploded standard errors.

![](images/prediction_causation.PNG){style="display: block; margin: 1em auto" width="60%"}

(SICSS 2018 - Sendhil Mullainathan's presentation slide)

Selective Labels Problem ([The Selective Labels Problem: Evaluating Algorithmic Predictions in the Presence of Unobservables](https://cs.stanford.edu/~jure/pubs/contraction-kdd17.pdf))

<br>

Recall Linear Regression \@ref(linear-regression) OLS estimates

$$
\begin{aligned}
\hat{\beta}_{OLS} &= (\mathbf{X}'\mathbf{X})^{-1}(\mathbf{X}'\mathbf{Y}) \\
&= (\mathbf{X}'\mathbf{X})^{-1}(\mathbf{X}'(\mathbf{X \beta}+ \epsilon)) \\
&= (\mathbf{X}'\mathbf{X})^{-1} (\mathbf{X}'\mathbf{X}) \beta + (\mathbf{X}'\mathbf{X})^{-1}(\mathbf{X}'\epsilon) \\
\hat{\beta}_{OLS} & \to \beta + (\mathbf{X}'\mathbf{X})^{-1}(\mathbf{X}'\epsilon)
\end{aligned}
$$

Hence, OLS estimates will be unbiased (i.e., get rid of $(\mathbf{X}'\mathbf{X})^{-1}(\mathbf{X}'\epsilon)$) if we have the following 2 conditions:

1.  $E(\epsilon|\mathbf{X}) = 0$ With an intercept, we can usually solve this problem
2.  $Cov(\mathbf{X}, \epsilon) = 0$

Problem with estimation usually stems from the second condition.

Tools to combat this problem can be found in causal inference \@ref(causal-inference)
