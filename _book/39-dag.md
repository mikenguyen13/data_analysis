# Directed Acyclic Graph

Native R:

-   `dagitty`

-   `ggdag`

-   `dagR`

-   `r-causal`: by [Center for Causal Discovery](https://www.ccd.pitt.edu/data-science/). Also available in Python

Publication-ready (with `R` and `Latex`): [shinyDAG](https://www.gerkelab.com/project/shinydag/)

Standalone program: [DAG program](https://hsz.dife.de/dag/) by Sven Knuppel

## Basic Notations

Basic building blocks of DAG

-   Mediators (chains): $X \to Z \to Y$

    -   controlling for Z blocks (closes) the causal impact of $X \to Y$

-   Common causes (forks): $X \leftarrow Z \to Y$

    -   Z (i.e., confounder) is a common cause in which it induces a non-causal association between $X$ and $Y$.

    -   Controlling for $Z$ should close this association.

    -   $Z$ d-separates $X$ from $Y$ when it blocks (closes) all paths from $X$ to $Y$ (i.e., $X \perp Y |Z$). This applies to both common causes and mediators.

-   Common effects (colliders): $X \to Z \leftarrow Y$

    -   Not controlling for $Z$ does not induce an association between $X$ and $Y$

    -   Controlling for $Z$ induces a non-causal association between $X$ and $Y$

Notes:

-   A descendant of a variable behavior similarly to that variable (e.g., a descendant of $Z$ can behave like $Z$ and partially control for $Z$)

-   Rule of thumb for multiple [Controls]: o have [Causal inference] $X \to Y$, we must

    -   Close all backdoor path between $X$ and $Y$ (to eliminate spurious correlation)

    -   Do not close any causal path between $X$ and $Y$ (any mediators).

## Causal Discovery

@eberhardt2024discovering
