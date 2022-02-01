---
---
---

# Identification 

$$
\Theta = g(\mathcal{F}(.))
$$

$\Theta$ is said to be identified if it **exist** and **unique** if $F \in \mathcal{F}$

-   $E(Y|X= x)$ is identified if $E(Y) < \infty$

-   Linear Projection Coefficient is identified: $L(Y|1, X)$ is identified as long as $E(X'X)$ is non-singular.

Most of the time: $\Theta$ = some function of observed moment random variables

`=` represents both existence and uniqueness

Examples:

Example 1: Linear model

$$
Y= \beta_0 + X \beta_1 + U
$$

Suppose identification assumption is $E(u |X) = 0$, which means $E(Y|X) = \beta + X \beta_1$

then we cann identifiy

$$
\beta_0 = E(Y|X =0) 
$$

and

$$
\begin{aligned}
Cov(Y,X) &= Cov( \beta_0, X) + Cov(X \beta_1, X) + Cov( u, X) \\
&= 0 + Var(X) \beta_1 + 0 \\
\beta_1&= \frac{Cov(Y,X)}{Var(X)} && \text{additional assumption } Var(X) \neq 0
\end{aligned}
$$

Example 2: Random Coefficient Model

$$
Y = U_0 + U_1 X
$$

Example 3:
