# Coffee certification experiments (BRIDGE)

`coffee_certification.csv` is an analysis-ready extract of the two coffee
certification experiments reported in:

> Mukherjee, Anirban, Hannah H. Chang, and Sachin Gupta (2026), "Behavioral
> Research Through Interpretable, Dimensionality-reduced Generative AI
> Embeddings (BRIDGE): A Method to Incorporate Real-World Stimuli in Consumer
> Experiments," *Journal of Marketing Research*.
> https://doi.org/10.1177/00222437261484068

Source: the authors' reproducibility bundle,
https://github.com/dranirbanmukherjee/bridge-coffee-certification
(files `precomputed/ft_data_with_nuisance.csv` and
`precomputed/org_data_with_nuisance.csv`), released under
Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
This extract is a derivative and carries the same licence.

## What was done to build the extract

Filters, exactly as in the authors' `code/05_coffee_certification_estimate.R`:

1. keep `Q_TerminateFlag == "Complete"`;
2. recode the seven-point preference item to a -3..+3 scale, undoing the
   left/right randomization: `pref = DVPref_Rating_1 - 4` when the base
   description was shown on the left, `8 - DVPref_Rating_1 - 4` otherwise,
   so a positive value means the certified description was preferred;
3. drop rows with a missing preference or condition;
4. keep `nuisance_matched == TRUE`.

`wc_diff` is the word count of the comparison description minus the word count
of the base description. `INTN1` and `INTN2` are the first two BRIDGE nuisance
controls, taken unchanged from the bundle. The free-text stimuli and all
participant demographics are not redistributed here.

Resulting cell sizes match the paper: fair trade 125 / 118 / 110 and organic
101 / 117 / 134 for the matched, shorter, and longer conditions.
