# Causality

After all of the mumbo jumbo that we have learned so far, I want to now talk about the concept of causality.\
We usually say that correlation is not causation. Then, what is causation?\
One of my favorite books has explained this concept beautifully [@Pearl_2018]. And I am just going to quickly summarize the gist of it from my understanding. I hope that it can give you an initial grasp on the concept so that later you can continue to read up and develop a deeper understanding.

It's important to have a deep understanding regarding the method research. However, one needs to be aware of its limitation and compliment with conceptual understanding. The aspect of concepts is typically referred in statistics when as expert knowledge. As mentioned in various sections throughout the book, we see that we need to ask experts for number as our baseline or visit literature to gain insight from past research.

Here, we dive in a more conceptual side statistical analysis as a whole, regardless of particular approach.

You probably heard scientists say correlation doesn't mean causation. There are ridiculous [spurious correlations](http://www.tylervigen.com/spurious-correlations) that give a firm grip on what the previous phrase means

<br>

This section is based on Bernard Koch's [presentaiton](https://www.youtube.com/watch?v=v9uf9rDYEMg&ab_channel=SummerInstituteinComputationalSocialScience) at SICSS - Los Angeles 2021

Identification under Selection on observable/ back-door criterion

Conditions:

-   Strong conditional ignorability

    -   $Y(0),Y(1) \perp T|X$

    -   No hidden confounders

-   Overlap

    -   $\forall x \in X, t \in \{0, 1\}: p (T = t | X = x> 0$

    -   All treatments have non-zero probability of being observed

-   SUTVA/ Consistency

    -   Treatment and outcomes of different subjects are independent

<br>

Example

We have

-   binary treatment $T \in \{ 0,1\}$

-   $Y(1), Y(0)$ are the potential outcomes

-   The average treatment effect is

$$
ATE = E(Y(1) - Y(0)) = E(\tau(x))
$$

-   The conditional average treatment effect is

$$
CATE = \tau(x) = E(Y(1) - Y(0)|X = x)
$$

see <https://github.com/maxwshen/iap-cidl/blob/master/iap-cidl-lecture1_fredrik_potential_outcomes.pdf>
