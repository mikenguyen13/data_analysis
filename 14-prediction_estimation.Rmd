# Prediction and Estimation

You cannot have both. You can either have high prediction or high estimation. In prediction, you minimize the loss function. In estimation, you try to best fit the data. Because the goal of estimation is to best fit the data, you always run the risk of not predicting well.

In high dimension, you always have weak to strong collinearity. Hence, your estimation can be undesirable. And you can't pick which one variable to stay in the model, but all these troubles would not affect your prediction. In Plateau problem

-   If two functions are similar in output space, you can still do prediction, but you can't do estimation because of exploded standard errors.

![](images/prediction_causation.PNG)

(SICSS 2018 - Sendhil Mullainathan's presentation slide)

Selective Labels Problem ([The Selective Labels Problem: Evaluating Algorithmic Predictions in the Presence of Unobservables](https://cs.stanford.edu/~jure/pubs/contraction-kdd17.pdf))
