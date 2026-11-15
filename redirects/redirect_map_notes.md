# data_analysis: old bookdown.org URL -> new Posit URL redirect map

Built 2026-11-15. 560 old filenames (Wayback index of bookdown.org/mike/data_analysis/*) that 404 on mike-data-analysis.share.connect.posit.cloud.
Each gets a stub page (meta refresh 0 + canonical + JS location.replace). Stubs: data_analysis_redirect_stubs/. Machine map: data_analysis_redirect_map.json.

GA views = page views Oct 2025-Apr 2026 (GA4 Data Analysis Book, top 250 pages, summed across hosts). "stub" = target chapter is a Springer paywall preview stub.

| old file | GA views | new target | match | stub? |
|---|---:|---|---|---|
| sec-two-way-fixed-effects.html | 6585 | sec-difference-in-differences.html#sec-two-way-fixed-effects | id |  |
| sec-interrupted-time-series.html | 4783 | sec-temporal-discontinuity-designs.html#sec-interrupted-time-series | id |  |
| sec-regression-discontinuity-in-time.html | 3996 | sec-temporal-discontinuity-designs.html#sec-regression-discontinuity-in-time | id |  |
| sec-negative-binomial-regression.html | 2977 | generalized-linear-models.html | chapter | stub |
| sec-two-one-sided-tests-equivalence-testing.html | 2626 | hypothesis-testing.html | chapter | stub |
| sec-quasi-poisson-regression.html | 2039 | generalized-linear-models.html | chapter | stub |
| sec-panel-data.html | 2017 | data.html#sec-panel-data | id |  |
| sec-repeated-cross-sectional-data.html | 1819 | data.html#sec-repeated-cross-sectional-data | id |  |
| sec-fuzzy-regression-discontinuity-design.html | 1606 | sec-regression-discontinuity.html#sec-fuzzy-regression-discontinuity-design | id |  |
| types-of-treatment-effects.html | 1392 | sec-causal-inference.html#types-of-treatment-effects | id |  |
| categorical-data-analysis.html | 1323 | basic-statistical-inference.html#categorical-data-analysis | id |  |
| heteroskedasticity-tests.html | 1178 | model-specification-tests.html | chapter | stub |
| sec-generalized-additive-models.html | 1153 | sec-nonparametric-regression.html | chapter | stub |
| divergence-metrics-and-tests-for-comparing-distributions.html | 1039 | basic-statistical-inference.html#divergence-metrics-and-tests-for-comparing-distributions | id |  |
| repeated-measures-in-mixed-models.html | 1025 | sec-linear-mixed-models.html | chapter | stub |
| autocorrelation-tests.html | 1022 | model-specification-tests.html | chapter | stub |
| nested-model-tests.html | 990 | model-specification-tests.html | chapter | stub |
| steps-for-conducting-an-event-study.html | 984 | sec-event-studies.html#steps-for-conducting-an-event-study | id |  |
| sec-false-discovery-rate.html | 914 | hypothesis-testing.html | chapter | stub |
| information-criteria-for-model-selection.html | 904 | sec-linear-mixed-models.html | chapter | stub |
| generalized-synthetic-control.html | 875 | sec-synthetic-control.html#sec-generalized-synthetic-control | id |  |
| sec-local-polynomial-regression.html | 815 | sec-nonparametric-regression.html | chapter | stub |
| methods-for-handling-missing-data.html | 813 | imputation-missing-data.html#methods-for-handling-missing-data | id |  |
| non-nested-model-tests.html | 792 | model-specification-tests.html | chapter | stub |
| ordinary-least-squares.html | 742 | linear-regression.html#ordinary-least-squares | id | stub |
| sec-kernel-regression.html | 718 | sec-nonparametric-regression.html | chapter | stub |
| bivariate-statistics.html | 588 | descriptive-statistics.html | chapter | stub |
| testing-assumptions.html | 570 | sec-instrumental-variables.html#testing-assumptions | id |  |
| packages-for-marginal-effects.html | 546 | sec-marginal-effects.html | chapter | stub |
| types-of-marginal-effect.html | 542 | sec-marginal-effects.html | chapter | stub |
| bayesian-synthetic-control.html | 530 | sec-synthetic-control.html#bayesian-synthetic-control | id |  |
| split-plot-designs.html | 529 | sec-linear-mixed-models.html | chapter | stub |
| sec-inference-iv.html | 519 | sec-instrumental-variables.html#sec-inference-iv | id |  |
| long-run-event-studies.html | 474 | sec-event-studies.html#long-run-event-studies | id |  |
| robustness-checks-3.html | 472 | sec-difference-in-differences.html#robustness-checks-5 | title |  |
| sec-logistic-regression.html | 471 | generalized-linear-models.html | chapter | stub |
| sec-delta-method.html | 468 | sec-marginal-effects.html | chapter | stub |
| sec-smoothing-splines.html | 467 | sec-nonparametric-regression.html | chapter | stub |
| diagnosing-the-missing-data-mechanism.html | 450 | imputation-missing-data.html#diagnosing-the-missing-data-mechanism | id |  |
| assumptions-3.html | 436 | sec-difference-in-differences.html#assumptions-3 | id |  |
| functional-form-tests.html | 420 | model-specification-tests.html | chapter | stub |
| types-of-validity-in-research.html | 404 | sec-causal-inference.html#types-of-validity-in-research | id |  |
| sec-modern-estimators-for-staggered-adoption.html | 402 | sec-difference-in-differences.html#sec-modern-estimators-for-staggered-adoption | id |  |
| inference-in-linear-mixed-models.html | 401 | sec-linear-mixed-models.html | chapter | stub |
| sec-poisson-regression.html | 401 | generalized-linear-models.html | chapter | stub |
| estimation-in-linear-mixed-models.html | 372 | sec-linear-mixed-models.html | chapter | stub |
| coefficient-stability.html | 370 | sensitivity-analysis-and-robustness-checks.html#sec-coefficient-stability | id |  |
| sec-endogenous-sample-selection.html | 358 | sec-endogeneity.html#endogenous-sample-selection | title |  |
| assumptions-for-identifying-treatment-effects.html | 332 | sec-quasi-experimental.html#assumptions-for-identifying-treatment-effects | id |  |
| sec-sharp-regression-discontinuity-design.html | 330 | sec-regression-discontinuity.html#sec-sharp-regression-discontinuity-design | id |  |
| synthetic-control.html | 327 | sec-synthetic-control.html | chapter |  |
| sec-aggregation-bias.html | 315 | sec-biases.html#sec-aggregation-bias | id |  |
| other-biases.html | 299 | sec-biases.html | chapter |  |
| sec-wavelet-regression.html | 293 | sec-nonparametric-regression.html | chapter | stub |
| application-mixed-models-in-practice.html | 291 | sec-linear-mixed-models.html | chapter | stub |
| multicollinearity-diagnostics.html | 284 | model-specification-tests.html | chapter | stub |
| simpsons-paradox.html | 278 | sec-causal-inference.html#simpsons-paradox | id |  |
| specification-checks.html | 262 | sec-regression-discontinuity.html#specification-checks | id |  |
| sec-multiple-periods-and-variation-in-treatment-timing.html | 255 | sec-difference-in-differences.html#sec-multiple-periods-and-variation-in-treatment-timing | id |  |
| unbalanced-or-unequally-spaced-data.html | 247 | sec-linear-mixed-models.html | chapter | stub |
| multiple-treatments.html | 246 | sec-difference-in-differences.html#multiple-treatments | id |  |
| case-crossover-study-design.html | 244 | sec-temporal-discontinuity-designs.html#case-crossover-study-design | id |  |
| sec-multinomial-logistic-regression.html | 235 | generalized-linear-models.html | chapter | stub |
| combining-both-rdit-and-its.html | 229 | sec-temporal-discontinuity-designs.html#combining-both-rdit-and-its | id |  |
| applications-1.html | 226 | sec-synthetic-control.html#applications-1 | id |  |
| normality-assessment.html | 198 | descriptive-statistics.html | chapter | stub |
| application-of-event-study.html | 186 | sec-event-studies.html#application-of-event-study | id |  |
| dependent-data.html | 171 | sec-linear-mixed-models.html | chapter | stub |
| confounders-in-event-studies.html | 155 | sec-event-studies.html#confounders-in-event-studies | id |  |
| sec-probit-regression.html | 154 | generalized-linear-models.html | chapter | stub |
| sec-augmented-synthetic-control.html | 144 | sec-synthetic-control.html#sec-augmented-synthetic-control | id |  |
| sec-simple-difference-in-differences.html | 131 | sec-difference-in-differences.html#sec-simple-difference-in-differences | id |  |
| sec-generalization-of-generalized-linear-models.html | 130 | generalized-linear-models.html | chapter | stub |
| bad-controls.html | 118 | sec-controls.html#bad-controls | id |  |
| comparison-delta-method-vs.-alternative-approaches.html | 110 | sec-marginal-effects.html | chapter | stub |
| mediation-under-did.html | 104 | sec-difference-in-differences.html#mediation-under-did | id |  |
| sec-multi-cutoff-regression-discontinuity-design.html | 103 | sec-regression-discontinuity.html#sec-multi-cutoff-regression-discontinuity-design | id |  |
| sec-nested-designs.html | 101 | sec-analysis-of-variance-anova.html#sec-nested-designs | id |  |
| sec-selection-on-unobservables.html | 101 | sec-matching-methods.html#sec-selection-on-unobservables | id |  |
| choosing-the-right-type-of-data.html | 96 | data.html#choosing-the-right-type-of-data | id |  |
| sensitivity-analysis-robustness-check.html | 93 | sensitivity-analysis-and-robustness-checks.html | chapter |  |
| evaluation-of-imputation-methods.html | 91 | imputation-missing-data.html#evaluation-of-imputation-methods | id |  |
| hypothesis-testing-framework.html | 91 | basic-statistical-inference.html#hypothesis-testing-framework | id |  |
| synthetic-control-with-staggered-adoption.html | 89 | sec-synthetic-control.html#synthetic-control-with-staggered-adoption | id |  |
| sec-regression-kink-design.html | 88 | sec-regression-discontinuity.html#sec-regression-kink-design | id |  |
| sec-estimation.html | 86 | sec-instrumental-variables.html#sec-estimation | id |  |
| sec-balanced-sampling.html | 83 | sampling.html | chapter | stub |
| sec-cross-sectional-data.html | 82 | data.html#sec-cross-sectional-data | id |  |
| experimental-vs.-quasi-experimental-designs.html | 79 | sec-causal-inference.html#experimental-vs.-quasi-experimental-designs | id |  |
| sec-completely-randomized-design.html | 79 | sec-analysis-of-variance-anova.html#sec-completely-randomized-design | id |  |
| non-linear-least-squares-estimation.html | 74 | non-linear-regression.html#non-linear-least-squares-estimation | id | stub |
| theoretical-foundations-of-missing-data.html | 74 | imputation-missing-data.html#theoretical-foundations-of-missing-data | id |  |
| additional-resources-1.html |  | sec-causal-inference.html | chapter |  |
| additional-resources.html |  | sec-moderation.html | chapter | stub |
| advanced-experimental-designs.html |  | sec-experimental-design.html#advanced-experimental-designs | id |  |
| advanced-topics-in-sensitivity-analysis.html |  | sensitivity-analysis-and-robustness-checks.html#advanced-topics-in-sensitivity-analysis | id(no-old-toc) |  |
| advantages-of-scm.html |  | sec-synthetic-control.html | chapter |  |
| aggregation-bias-1.html |  | sec-biases.html#sec-aggregation-bias | title |  |
| aggregation-bias.html |  | sec-biases.html#sec-aggregation-bias | id |  |
| aggregation.html |  | sec-event-studies.html#aggregation | id |  |
| analysis-of-variance-anova.html |  | sec-analysis-of-variance-anova.html | chapter |  |
| another-perspective.html |  | imputation-missing-data.html | chapter |  |
| appendix-reproducible-snippets.html |  | reporting-your-analysis.html#appendix-reproducible-snippets | id(no-old-toc) |  |
| application-10.html |  | imputation-missing-data.html | chapter |  |
| application-11.html |  | sec-synthetic-difference-in-differences.html#application-1 | title |  |
| application-12.html |  | sec-changes-in-changes.html#application-2 | title |  |
| application-13.html |  | sec-event-studies.html | chapter |  |
| application-14.html |  | replication-and-synthetic-data.html#application-3 | title |  |
| application-15.html |  | sec-event-studies.html | chapter |  |
| application-16.html |  | replication-and-synthetic-data.html#application-3 | title |  |
| application-2.html |  | sec-moderation.html | chapter | stub |
| application-3.html |  | sec-synthetic-difference-in-differences.html#application-1 | title |  |
| application-4.html |  | sec-changes-in-changes.html#application-2 | title |  |
| application-5.html |  | replication-and-synthetic-data.html#application-3 | title |  |
| application-6.html |  | sec-changes-in-changes.html#application-2 | title |  |
| application-7.html |  | sec-event-studies.html | chapter |  |
| application-8.html |  | replication-and-synthetic-data.html#application-3 | title |  |
| application-9.html |  | sec-quantile-distributional.html | chapter |  |
| application-nonlinear-and-generalized-linear-mixed-models.html |  | sec-nonlinear-and-generalized-linear-mixed-models.html | chapter | stub |
| application-of-imputation-in-r.html |  | imputation-missing-data.html | chapter |  |
| application-of-imputation.html |  | imputation-missing-data.html#application-of-imputation | id |  |
| application.html |  | non-linear-regression.html | chapter | stub |
| applications-of-rd-designs.html |  | sec-regression-discontinuity.html#applications-of-rd-designs | id |  |
| applications.html |  | sec-synthetic-control.html#applications-1 | title |  |
| assumption-violation.html |  | sec-difference-in-differences.html | chapter |  |
| assumptions-1.html |  | sec-difference-in-differences.html#assumptions-2 | title |  |
| assumptions-2.html |  | sec-difference-in-differences.html#assumptions-2 | id |  |
| assumptions-4.html |  | sec-synthetic-control.html#assumptions-5 | title |  |
| assumptions-5.html |  | sec-synthetic-control.html#assumptions-5 | id(no-old-toc) |  |
| asymptotic-properties-of-the-iv-estimator.html |  | sec-instrumental-variables.html#asymptotic-properties-of-the-iv-estimator | id |  |
| augmented-synthetic-control-method.html |  | sec-synthetic-control.html#sec-augmented-synthetic-control | title |  |
| autocorrelation-1.html |  | sec-temporal-discontinuity-designs.html#sec-interrupted-time-series | chapter |  |
| autocorrelation.html |  | sec-temporal-discontinuity-designs.html#sec-interrupted-time-series | chapter |  |
| average-marginal-effect-algorithm.html |  | sec-marginal-effects.html | chapter | stub |
| bad-controls-1.html |  | sec-controls.html#bad-controls | title |  |
| balanced-sampling.html |  | sampling.html | chapter | stub |
| basic-concepts-in-nonparametric-estimation.html |  | sec-nonparametric-regression.html | chapter | stub |
| basic-notation-and-graph-structures.html |  | sec-directed-acyclic-graphs.html#basic-notation-and-graph-structures | id |  |
| basic-notations-1.html |  | sec-directed-acyclic-graphs.html | chapter |  |
| basic-notations.html |  | sec-directed-acyclic-graphs.html | chapter |  |
| basic-understanding.html |  | sec-multivariate-methods.html#basic-understanding | id |  |
| best-practices-for-hpc-in-data-analysis.html |  | high-performance-computing.html#best-practices-for-hpc-in-data-analysis | id |  |
| biases-in-event-studies.html |  | sec-event-studies.html#biases-in-event-studies | id |  |
| biases.html |  | sec-event-studies.html | chapter |  |
| binomial-regression.html |  | generalized-linear-models.html | chapter | stub |
| categorical-variables.html |  | variable-transformation.html | chapter | stub |
| causal-discovery.html |  | sec-directed-acyclic-graphs.html#causal-discovery | id |  |
| causal-inference-approach-1.html |  | sec-mediation.html | chapter |  |
| causal-inference-approach-to-mediation.html |  | sec-mediation.html#causal-inference-approach-to-mediation | id |  |
| causal-inference-approach.html |  | sec-mediation.html | chapter |  |
| causal-inference.html |  | sec-causal-inference.html | chapter |  |
| causality.html |  | sec-causal-inference.html | chapter |  |
| causation-versus-prediction.html |  | prediction-and-estimation.html | chapter | stub |
| cautions-in-iv.html |  | sec-instrumental-variables.html | chapter |  |
| challenges-and-ethical-considerations.html |  | imputation-missing-data.html#challenges-and-ethical-considerations | id |  |
| challenges-with-instrumental-variables.html |  | sec-instrumental-variables.html#challenges-with-instrumental-variables | id |  |
| changes-in-an-estimate-1.html |  | reporting-your-analysis.html#changes-in-an-estimate | title |  |
| changes-in-an-estimate.html |  | reporting-your-analysis.html#changes-in-an-estimate | id |  |
| changes-in-changes.html |  | sec-changes-in-changes.html | chapter |  |
| chapter-cluster-randomization-and-interference-bias.html |  | sec-interference-spillover.html | chapter |  |
| choosing-a-matching-strategy.html |  | sec-matching-methods.html#choosing-a-matching-strategy | id |  |
| choosing-controls-1.html |  | sec-controls.html#choosing-controls | title |  |
| choosing-controls.html |  | sec-controls.html#choosing-controls | id |  |
| citation.html |  | appendix.html#citation | id |  |
| classical-experimental-designs.html |  | sec-experimental-design.html#classical-experimental-designs | id |  |
| cluster-analysis.html |  | sec-multivariate-methods.html | chapter |  |
| coarsened-exact-matching.html |  | sec-matching-methods.html | chapter |  |
| coefficient-stability-1.html |  | sensitivity-analysis-and-robustness-checks.html#sec-coefficient-stability | title |  |
| coefficient-uncertainty-and-distribution-1.html |  | reporting-your-analysis.html#coefficient-uncertainty-and-distribution | title |  |
| coefficient-uncertainty-and-distribution.html |  | reporting-your-analysis.html#coefficient-uncertainty-and-distribution | id |  |
| comparison-of-testing-frameworks.html |  | hypothesis-testing.html | chapter | stub |
| completely-randomized-design-crd.html |  | sec-analysis-of-variance-anova.html#sec-completely-randomized-design | manual |  |
| conceptual-framework.html |  | sec-regression-discontinuity.html#conceptual-framework | id |  |
| conceptual-framing.html |  | prediction-and-estimation.html | chapter | stub |
| concerns-in-did.html |  | sec-difference-in-differences.html#concerns-in-did | id |  |
| conclusion-and-recommendations.html |  | sensitivity-analysis-and-robustness-checks.html#conclusion-and-recommendations | id(no-old-toc) |  |
| conclusion-the-evolving-landscape-of-regression-analysis.html |  | sec-nonparametric-regression.html | chapter | stub |
| conclusion.html |  | prediction-and-estimation.html | chapter | stub |
| confidence-intervals-in-nonparametric-regression.html |  | sec-nonparametric-regression.html | chapter | stub |
| contamination-bias.html |  | sec-biases.html#sec-contamination-bias | id |  |
| continuous-variables.html |  | variable-transformation.html | chapter | stub |
| control-function.html |  | sec-instrumental-variables.html | chapter |  |
| controls-1.html |  | sec-controls.html | chapter |  |
| controls.html |  | sec-controls.html | chapter |  |
| criteria-for-choosing-an-effective-approach.html |  | imputation-missing-data.html#criteria-for-choosing-an-effective-approach | id |  |
| critiques-of-twfe-in-staggered-did.html |  | sec-difference-in-differences.html#sec-two-way-fixed-effects | manual |  |
| cross-sectional.html |  | data.html#sec-cross-sectional-data | manual |  |
| customize-your-dailyautomatic-report.html |  | exploratory-data-analysis.html#customize-your-dailyautomatic-report | id(no-old-toc) |  |
| data-importexport.html |  | prerequisites.html | chapter |  |
| data-manipulation.html |  | prerequisites.html | chapter |  |
| data-quality-and-ethical-considerations.html |  | data.html#data-quality-and-ethical-considerations | id |  |
| data-report.html |  | exploratory-data-analysis.html#data-report | id(no-old-toc) |  |
| data-types.html |  | data.html#data-types | id |  |
| definition-of-marginal-effects.html |  | sec-marginal-effects.html | chapter | stub |
| delta-method.html |  | sec-marginal-effects.html | chapter | stub |
| demonstration.html |  | high-performance-computing.html#demonstration | id |  |
| descriptive-stat.html |  | descriptive-statistics.html | chapter | stub |
| descriptive-tables-1.html |  | reporting-your-analysis.html#descriptive-tables | title |  |
| descriptive-tables.html |  | reporting-your-analysis.html#descriptive-tables | id |  |
| design-vs.-model-based-approaches.html |  | sec-quasi-experimental.html#design-vs.-model-based-approaches | id |  |
| designmatch.html |  | sec-matching-methods.html | chapter |  |
| diagnosing-the-mechanism.html |  | imputation-missing-data.html | chapter |  |
| difference-in-differences.html |  | sec-difference-in-differences.html | chapter |  |
| directed-acyclic-graph-1.html |  | sec-directed-acyclic-graphs.html | chapter |  |
| directed-acyclic-graph.html |  | sec-directed-acyclic-graphs.html | chapter |  |
| disadvantages-of-scm.html |  | sec-synthetic-control.html | chapter |  |
| discriminant-analysis.html |  | sec-multivariate-methods.html | chapter |  |
| divergence-metrics-and-test-for-comparing-distributions.html |  | basic-statistical-inference.html#divergence-metrics-and-tests-for-comparing-distributions | manual |  |
| economic-significance.html |  | sec-event-studies.html#economic-significance | id |  |
| embedded-methods-integrated-into-model-training.html |  | variable-selection.html | chapter | stub |
| emerging-research-1.html |  | sec-instrumental-variables.html | chapter |  |
| emerging-research.html |  | sec-experimental-design.html#emerging-research | id |  |
| emerging-trends-in-missing-data-handling.html |  | imputation-missing-data.html#emerging-trends-in-missing-data-handling | id |  |
| emmeans-package.html |  | sec-moderation.html | chapter | stub |
| empirical-research-walkthrough.html |  | sec-difference-in-differences.html#empirical-research-walkthrough | id |  |
| empirical-studies-1.html |  | sec-difference-in-differences.html | chapter |  |
| empirical-studies.html |  | sec-difference-in-differences.html | chapter |  |
| endogeneity-1.html |  | sec-endogeneity.html | chapter |  |
| endogeneity.html |  | sec-endogeneity.html | chapter |  |
| endogenous-sample-selection-1.html |  | sec-endogeneity.html#endogenous-sample-selection | title |  |
| endogenous-sample-selection.html |  | sec-endogeneity.html#endogenous-sample-selection | id |  |
| endogenous-treatment-1.html |  | sec-endogeneity.html#endogenous-treatment | title |  |
| endogenous-treatment.html |  | sec-endogeneity.html#endogenous-treatment | id |  |
| entropy-balancing.html |  | sec-matching-methods.html#entropy-balancing | id |  |
| error-identification.html |  | exploratory-data-analysis.html#error-identification | id(no-old-toc) |  |
| establishing-mechanisms.html |  | sec-quasi-experimental.html#establishing-mechanisms | id |  |
| estimating-qtt-with-cic.html |  | sec-changes-in-changes.html#estimating-qtt-with-cic | id |  |
| estimation-1.html |  | sec-instrumental-variables.html#sec-estimation | title |  |
| estimation-2.html |  | sec-synthetic-control.html#estimation-3 | title |  |
| estimation-3.html |  | sec-synthetic-control.html#estimation-3 | id |  |
| estimation-4.html |  | sec-instrumental-variables.html#sec-estimation | title |  |
| estimation-and-inference.html |  | sec-regression-discontinuity.html#estimation-and-inference | id |  |
| estimation-in-nonlinear-and-generalized-linear-mixed-models.html |  | sec-nonlinear-and-generalized-linear-mixed-models.html | chapter | stub |
| evaluation-of-a-regression-discontinuity-design.html |  | sec-regression-discontinuity.html#evaluation-of-a-regression-discontinuity-design | id |  |
| evaluation-of-an-rd.html |  | sec-regression-discontinuity.html | chapter |  |
| event-studies-in-marketing.html |  | sec-event-studies.html#event-studies-in-marketing | id |  |
| event-studies.html |  | sec-event-studies.html | chapter |  |
| example-dag.html |  | sec-directed-acyclic-graphs.html | manual |  |
| example-workflow-in-r.html |  | high-performance-computing.html#example-workflow-in-r | id |  |
| examples.html |  | sec-difference-in-differences.html | chapter |  |
| expected-return-calculation.html |  | sec-event-studies.html#sec-expected-return-calculation | id |  |
| exploratory-analysis.html |  | reporting-your-analysis.html#exploratory-analysis | id(no-old-toc) |  |
| exploratory-data-analysis-1.html |  | exploratory-data-analysis.html | chapter |  |
| extended-mathematical-points.html |  | prediction-and-estimation.html | chapter | stub |
| extensions-and-research-opportunities.html |  | bookdown-cheat-sheet.html | chapter |  |
| factor-analysis.html |  | sec-multivariate-methods.html | chapter |  |
| feasiable-prais-winsten.html |  | linear-regression.html | chapter | stub |
| feasible-generalized-least-squares.html |  | linear-regression.html | chapter | stub |
| feasible-group-level-random-effects.html |  | linear-regression.html | chapter | stub |
| feature-engineering.html |  | exploratory-data-analysis.html#feature-engineering | id(no-old-toc) |  |
| formal-definition.html |  | sec-regression-discontinuity.html#formal-definition | id |  |
| framework-for-generalization.html |  | sec-matching-methods.html#framework-for-generalization | id |  |
| framework-for-instrumental-variables.html |  | sec-instrumental-variables.html#sec-iv-framework | title |  |
| framework.html |  | sec-instrumental-variables.html | chapter |  |
| function-short-cut.html |  | appendix.html#function-short-cut | id |  |
| fuzzy-rd-design.html |  | sec-regression-discontinuity.html#sec-fuzzy-regression-discontinuity-design | manual |  |
| general-math.html |  | prerequisites.html | chapter |  |
| general-recommendations.html |  | introduction.html#general-recommendations | id |  |
| generalization.html |  | generalized-linear-models.html | chapter | stub |
| generalized-additive-models.html |  | sec-nonparametric-regression.html | chapter | stub |
| generalized-least-squares.html |  | linear-regression.html | chapter | stub |
| generalized-method-of-moments.html |  | sec-structural-estimation.html | chapter |  |
| genetic-matching.html |  | sec-matching-methods.html#sec-genetic-matching | id |  |
| git.html |  | appendix.html#git | id |  |
| good-controls-1.html |  | sec-controls.html#good-controls | title |  |
| good-controls.html |  | sec-controls.html#good-controls | id |  |
| graphical-measures.html |  | descriptive-statistics.html | chapter | stub |
| heterogeneity-in-the-event-effect.html |  | sec-event-studies.html#heterogeneity-in-the-event-effect | id |  |
| heteroskedasticity-1.html |  | model-specification-tests.html | chapter | stub |
| heteroskedasticity.html |  | model-specification-tests.html | chapter | stub |
| hierarchical-ordering-of-causal-tools.html |  | sec-causal-inference.html#hierarchical-ordering-of-causal-tools | id |  |
| how-to-cite-these-books.html |  | index.html | home |  |
| how-to-cite-this-book.html |  | index.html | home |  |
| identification-strategy-in-quasi-experiments.html |  | sec-quasi-experimental.html#identification-strategy-in-quasi-experiments | id |  |
| illustrative-equations-and-mathematical-contrasts.html |  | prediction-and-estimation.html | chapter | stub |
| implementation-practical-guidelines.html |  | sec-interference-spillover.html | chapter |  |
| implications-for-causal-inference.html |  | sec-endogeneity.html | chapter |  |
| imputation_miss-ing_data.html |  | imputation-missing-data.html | manual |  |
| imputation_missing_data.html |  | imputation-missing-data.html | manual |  |
| inference-1.html |  | non-linear-regression.html#inference | title | stub |
| inference-2.html |  | sec-instrumental-variables.html | chapter |  |
| inference-3.html |  | sec-instrumental-variables.html | chapter |  |
| inference-4.html |  | sec-instrumental-variables.html | chapter |  |
| inference-in-scm.html |  | sec-synthetic-control.html#inference-in-scm | id |  |
| inference-prediction.html |  | reporting-your-analysis.html#inference-prediction | id(no-old-toc) |  |
| inference.html |  | non-linear-regression.html#inference | id | stub |
| information-criteria-1.html |  | sec-linear-mixed-models.html | chapter | stub |
| information-criteria.html |  | sec-linear-mixed-models.html | chapter | stub |
| install-all-necessary-packages-on-your-local-machine.html |  | appendix.html#install-all-necessary-packages-on-your-local-machine | id |  |
| install-all-necessary-packageslibaries-on-your-local-machine.html |  | appendix.html#install-all-necessary-packages-on-your-local-machine | manual |  |
| instrumental-variables.html |  | sec-instrumental-variables.html | chapter |  |
| interaction-debate-binning-estimators-vs.-generalized-additive-models.html |  | sec-moderation.html | chapter | stub |
| interactionr-package.html |  | sec-moderation.html | chapter | stub |
| interactions-package.html |  | sec-moderation.html | chapter | stub |
| interrupted-time-series-1.html |  | sec-temporal-discontinuity-designs.html#sec-interrupted-time-series | chapter |  |
| interrupted-time-series.html |  | sec-temporal-discontinuity-designs.html#sec-interrupted-time-series | chapter |  |
| introduction-and-motivation.html |  | sec-matching-methods.html#introduction-and-motivation | id |  |
| introduction-to-missing-data.html |  | imputation-missing-data.html#introduction-to-missing-data | id |  |
| key-assumptions-3.html |  | sec-event-studies.html#sec-key-assumptions | title |  |
| key-assumptions-4.html |  | sec-matching-methods.html#key-assumptions-5 | title |  |
| key-assumptions-of-staggered-did-designs.html |  | sec-difference-in-differences.html#sec-multiple-periods-and-variation-in-treatment-timing | manual |  |
| key-concepts-and-definitions.html |  | basic-statistical-inference.html#key-concepts-and-definitions | id |  |
| key-concepts.html |  | sec-changes-in-changes.html#key-concepts | id |  |
| key-features-of-scm.html |  | sec-synthetic-control.html | chapter |  |
| key-terminology.html |  | sec-moderation.html | chapter | stub |
| lagrange-multiplier-score.html |  | hypothesis-testing.html | chapter | stub |
| limitations-and-open-questions.html |  | sec-interference-spillover.html | chapter |  |
| limitations-of-quasi-experiments.html |  | sec-quasi-experimental.html#sec-limitations-of-quasi-experiments | id |  |
| linear-mixed-models.html |  | sec-linear-mixed-models.html | chapter | stub |
| logistic-regression-1.html |  | generalized-linear-models.html | chapter | stub |
| logistic-regression.html |  | generalized-linear-models.html | chapter | stub |
| mahalanobis-distance.html |  | sec-matching-methods.html | chapter |  |
| manova.html |  | sec-multivariate-methods.html | chapter |  |
| marginal-effects-in-different-contexts.html |  | sec-marginal-effects.html | chapter | stub |
| marginal-effects-interpretation.html |  | sec-marginal-effects.html | chapter | stub |
| marginal-effects.html |  | sec-marginal-effects.html | chapter | stub |
| marginal-properties-of-glmms.html |  | sec-nonlinear-and-generalized-linear-mixed-models.html | chapter | stub |
| marketing-applications.html |  | sec-synthetic-control.html | chapter |  |
| matching-for-multi-level-treatments.html |  | sec-matching-methods.html#matching-for-multi-level-treatments | id |  |
| matching-for-multiple-treatments.html |  | sec-matching-methods.html#matching-for-multiple-treatments | id |  |
| matching-for-repeated-treatments.html |  | sec-matching-methods.html | chapter |  |
| matching-for-time-series-cross-section-data.html |  | sec-matching-methods.html | chapter |  |
| matching-methods-1.html |  | sec-matching-methods.html | chapter |  |
| matching-methods.html |  | sec-matching-methods.html | chapter |  |
| matching-vs.-regression.html |  | sec-matching-methods.html#matching-vs.-regression | id |  |
| matchingfrontier.html |  | sec-matching-methods.html#matchingfrontier | id |  |
| matchit.html |  | sec-matching-methods.html | chapter |  |
| math-expression-syntax.html |  | bookdown-cheat-sheet.html#math-expression-syntax | id |  |
| math-expresssion-syntax.html |  | bookdown-cheat-sheet.html#math-expression-syntax | manual |  |
| mathematical-setup.html |  | prediction-and-estimation.html | chapter | stub |
| matrix-theory.html |  | prerequisites.html#matrix-theory | id |  |
| maximum-likelihood-estimator.html |  | linear-regression.html | chapter | stub |
| maximum-likelihood-regression.html |  | linear-regression.html | chapter | stub |
| mediation-1.html |  | sec-mediation.html | chapter |  |
| methodological-blueprint-for-applied-researchers.html |  | bookdown-cheat-sheet.html | chapter |  |
| methods.html |  | prerequisites.html | chapter |  |
| minimum-distance.html |  | sec-structural-estimation.html | chapter |  |
| missing-data.html |  | exploratory-data-analysis.html#missing-data | id(no-old-toc) |  |
| model-based-causal-mediation-analysis-1.html |  | sec-mediation.html | chapter |  |
| model-based-causal-mediation-analysis.html |  | sec-mediation.html | chapter |  |
| model-comparison-1.html |  | reporting-your-analysis.html#model-comparison | title |  |
| model-comparison.html |  | reporting-your-analysis.html#model-comparison | id |  |
| model-estimation-strategies.html |  | sec-regression-discontinuity.html#model-estimation-strategies | id |  |
| model-specification.html |  | model-specification-tests.html | chapter | stub |
| model.html |  | sec-mediation.html#model | id(no-old-toc) |  |
| moderation-model.html |  | sec-moderation.html | chapter | stub |
| more-books.html |  | index.html | home |  |
| multi-cutoff.html |  | sec-regression-discontinuity.html#sec-multi-cutoff-regression-discontinuity-design | manual |  |
| multi-score.html |  | sec-regression-discontinuity.html#sec-multi-score-regression-discontinuity-design | manual |  |
| multinomial.html |  | generalized-linear-models.html | chapter | stub |
| multiple-groups-1.html |  | sec-temporal-discontinuity-designs.html#sec-interrupted-time-series | chapter |  |
| multiple-groups.html |  | sec-temporal-discontinuity-designs.html#sec-interrupted-time-series | chapter |  |
| multiple-periods-and-variation-in-treatment-timing.html |  | sec-difference-in-differences.html#sec-multiple-periods-and-variation-in-treatment-timing | id |  |
| multiple-treatment.html |  | sec-difference-in-differences.html | chapter |  |
| multivariate-methods.html |  | sec-multivariate-methods.html | chapter |  |
| multivariate-nonparametric-regression.html |  | sec-nonparametric-regression.html | chapter | stub |
| natural-experiments.html |  | sec-quasi-experimental.html#sec-natural-experiments | id |  |
| natural-splines.html |  | sec-nonparametric-regression.html | chapter | stub |
| negative-binomial-regression.html |  | generalized-linear-models.html | chapter | stub |
| negative-r2-in-iv.html |  | sec-instrumental-variables.html#negative-r2-in-iv | id |  |
| negative-r2.html |  | sec-instrumental-variables.html | chapter |  |
| nested-designs.html |  | sec-analysis-of-variance-anova.html#sec-nested-designs | id |  |
| nested-model.html |  | model-specification-tests.html | chapter | stub |
| neutral-controls-1.html |  | sec-controls.html#neutral-controls | title |  |
| neutral-controls.html |  | sec-controls.html#neutral-controls | id |  |
| new-advances.html |  | sec-instrumental-variables.html | chapter |  |
| non-linear-least-squares.html |  | non-linear-regression.html | chapter | stub |
| non-nested-model.html |  | model-specification-tests.html | chapter | stub |
| non-probability-sampling.html |  | sampling.html | chapter | stub |
| nonlinear-and-generalized-linear-mixed-models.html |  | sec-nonlinear-and-generalized-linear-mixed-models.html | chapter | stub |
| nonparametric-anova.html |  | sec-analysis-of-variance-anova.html#sec-nonparametric-anova | id |  |
| nonparametric-regression.html |  | sec-nonparametric-regression.html | chapter | stub |
| not-so-code-y-process.html |  | exploratory-data-analysis.html#not-so-code-y-process | id(no-old-toc) |  |
| notes-1.html |  | sec-difference-in-differences.html | chapter |  |
| notes.html |  | sec-experimental-design.html | chapter |  |
| numerical-measures.html |  | descriptive-statistics.html | chapter | stub |
| omitted-variable-bias-quantification.html |  | sensitivity-analysis-and-robustness-checks.html#quantifying-omitted-variable-bias | manual |  |
| one-difference.html |  | sec-difference-in-differences.html#one-difference | id |  |
| one-sample-inference.html |  | basic-statistical-inference.html#one-sample-inference | id |  |
| one-summary-table-1.html |  | reporting-your-analysis.html#one-summary-table-packages | manual |  |
| one-summary-table-packages.html |  | reporting-your-analysis.html#one-summary-table-packages | id(no-old-toc) |  |
| one-summary-table.html |  | reporting-your-analysis.html#one-summary-table-packages | manual |  |
| one-table-summary.html |  | reporting-your-analysis.html#one-table-summary | id(no-old-toc) |  |
| operation.html |  | bookdown-cheat-sheet.html#operation | id |  |
| other-advances.html |  | sec-synthetic-control.html | chapter |  |
| other-biases-1.html |  | sec-biases.html | chapter |  |
| other-issues.html |  | sec-event-studies.html | chapter |  |
| p-hacking.html |  | sec-biases.html#p-hacking | id(no-old-toc) |  |
| packages.html |  | sec-marginal-effects.html | chapter | stub |
| panel-data.html |  | data.html#sec-panel-data | id |  |
| parameter-estimation-and-causal-inference.html |  | prediction-and-estimation.html | chapter | stub |
| parameter-estimation.html |  | prediction-and-estimation.html | chapter | stub |
| partial-least-squares-estimator.html |  | linear-regression.html | chapter | stub |
| partial-least-squares.html |  | linear-regression.html | chapter | stub |
| penalized-regularized-estimators.html |  | linear-regression.html | chapter | stub |
| poisson-regression.html |  | generalized-linear-models.html | chapter | stub |
| population-and-sample.html |  | sampling.html | chapter | stub |
| practical-considerations-2.html |  | sec-instrumental-variables.html#practical-considerations-2 | id |  |
| pre-experiment-checklist.html |  | bookdown-cheat-sheet.html | chapter |  |
| prediction-1.html |  | prediction-and-estimation.html | chapter | stub |
| prediction-2.html |  | prediction-and-estimation.html | chapter | stub |
| prediction-in-detail.html |  | prediction-and-estimation.html | chapter | stub |
| principal-component-regression.html |  | linear-regression.html | chapter | stub |
| principal-components.html |  | sec-multivariate-methods.html | chapter |  |
| principles-of-experimental-design.html |  | sec-experimental-design.html#principles-of-experimental-design | id |  |
| probability-sampling.html |  | sampling.html | chapter | stub |
| probability-theory.html |  | prerequisites.html#probability-theory | id |  |
| probit-regression.html |  | generalized-linear-models.html | chapter | stub |
| probmod-package.html |  | sec-moderation.html | chapter | stub |
| propensity-scores.html |  | sec-matching-methods.html#sec-propensity-scores | id |  |
| publication-bias-1.html |  | sec-biases.html#sec-publication-bias | title |  |
| publication-bias.html |  | sec-biases.html#sec-publication-bias | id |  |
| putting-it-all-together-comparing-objectives.html |  | prediction-and-estimation.html | chapter | stub |
| quantifying-omitted-variable-bias.html |  | sensitivity-analysis-and-robustness-checks.html#quantifying-omitted-variable-bias | id(no-old-toc) |  |
| quantile-regression.html |  | sec-quantile-distributional.html | chapter |  |
| quasi-experimental.html |  | sec-quasi-experimental.html | chapter |  |
| randomized-block-designs.html |  | sec-analysis-of-variance-anova.html#sec-randomized-block-designs | id |  |
| recommendations.html |  | high-performance-computing.html#recommendations-1 | title |  |
| recommended-structure.html |  | reporting-your-analysis.html#recommended-structure | id(no-old-toc) |  |
| recurrent-events.html |  | sec-temporal-discontinuity-designs.html#sec-interrupted-time-series | chapter |  |
| references-1.html |  | references.html | manual |  |
| regression-discontinuity.html |  | sec-regression-discontinuity.html | chapter |  |
| regression-kink-design.html |  | sec-regression-discontinuity.html#sec-regression-kink-design | id |  |
| regression-splines.html |  | sec-nonparametric-regression.html | chapter | stub |
| regression-trees-and-random-forests.html |  | sec-nonparametric-regression.html | chapter | stub |
| relationship-between-nlmms-and-glmms.html |  | sec-nonlinear-and-generalized-linear-mixed-models.html | chapter | stub |
| remedies-for-staggered-did-biases.html |  | sec-difference-in-differences.html#sec-modern-estimators-for-staggered-adoption | manual |  |
| repeated-cross-sections.html |  | data.html#sec-repeated-cross-sectional-data | manual |  |
| replication-and-synthetic-data-1.html |  | replication-and-synthetic-data.html | chapter |  |
| report-1.html |  | reporting-your-analysis.html | chapter |  |
| report.html |  | reporting-your-analysis.html | chapter |  |
| reporting-sensitivity-analysis-results.html |  | sensitivity-analysis-and-robustness-checks.html#reporting-sensitivity-analysis-results | id(no-old-toc) |  |
| rerandomization.html |  | sec-experimental-design.html#sec-re-randomization | title |  |
| review-of-event-studies-across-disciplines.html |  | sec-event-studies.html | chapter |  |
| ridge-regression.html |  | linear-regression.html | chapter | stub |
| robust-estimators.html |  | linear-regression.html | chapter | stub |
| robust-regression.html |  | linear-regression.html | chapter | stub |
| robustness-checks-4.html |  | sec-difference-in-differences.html#robustness-checks-5 | title |  |
| robustness-checks.html |  | sec-quasi-experimental.html#sec-robustness-checks-quasi-exp | title |  |
| rule-of-thumb-for-causal-inference.html |  | sec-directed-acyclic-graphs.html | chapter |  |
| sample-in-event-studies.html |  | sec-event-studies.html#sample-in-event-studies | id |  |
| sample-size-determination.html |  | basic-statistical-inference.html#sample-size-determination | id |  |
| sample-size-planning-for-anova.html |  | sec-analysis-of-variance-anova.html#sec-sample-size-planning-for-anova | id |  |
| sample.html |  | sec-event-studies.html | chapter |  |
| sec-binomial-regression.html |  | generalized-linear-models.html | chapter | stub |
| sec-coefficient-stability.html |  | sensitivity-analysis-and-robustness-checks.html#sec-coefficient-stability | id(no-old-toc) |  |
| sec-contamination-bias.html |  | sec-biases.html#sec-contamination-bias | id |  |
| sec-endogenous-treatment.html |  | sec-endogeneity.html#sec-endogenous-treatment | id |  |
| sec-filter-methods.html |  | variable-selection.html | chapter | stub |
| sec-generalized-least-squares.html |  | linear-regression.html | chapter | stub |
| sec-generalized-linear-mixed-models.html |  | sec-nonlinear-and-generalized-linear-mixed-models.html | chapter | stub |
| sec-limitations-of-quasi-experiments.html |  | sec-quasi-experimental.html#sec-limitations-of-quasi-experiments | id |  |
| sec-multi-score-regression-discontinuity-design.html |  | sec-regression-discontinuity.html#sec-multi-score-regression-discontinuity-design | id |  |
| sec-multivariate-analysis-of-variance.html |  | sec-multivariate-methods.html#sec-multivariate-analysis-of-variance | id |  |
| sec-natural-experiments.html |  | sec-quasi-experimental.html#sec-natural-experiments | id |  |
| sec-nonlinear-mixed-models.html |  | sec-nonlinear-and-generalized-linear-mixed-models.html | chapter | stub |
| sec-nonparametric-anova.html |  | sec-analysis-of-variance-anova.html#sec-nonparametric-anova | id |  |
| sec-null-hypothesis-significance-testing.html |  | hypothesis-testing.html | chapter | stub |
| sec-partial-least-squares.html |  | linear-regression.html | chapter | stub |
| sec-publication-bias.html |  | sec-biases.html#sec-publication-bias | id |  |
| sec-randomized-block-designs.html |  | sec-analysis-of-variance-anova.html#sec-randomized-block-designs | id |  |
| sec-robustness-checks-quasi-exp.html |  | sec-quasi-experimental.html#sec-robustness-checks-quasi-exp | id |  |
| sec-sample-size-planning-for-anova.html |  | sec-analysis-of-variance-anova.html#sec-sample-size-planning-for-anova | id |  |
| sec-selection-on-observables.html |  | sec-matching-methods.html#sec-selection-on-observables | id |  |
| sec-single-factor-covariance-model.html |  | sec-analysis-of-variance-anova.html#sec-single-factor-covariance-model | id |  |
| sec-specification-curve-analysis.html |  | sensitivity-analysis-and-robustness-checks.html#sec-specification-curve-analysis | id(no-old-toc) |  |
| sec-survivorship-bias.html |  | sec-biases.html#sec-survivorship-bias | id |  |
| sec-the-gold-standard-randomized-controlled-trials.html |  | sec-experimental-design.html#sec-the-gold-standard-randomized-controlled-trials | id |  |
| sec-the-ladder-of-causation.html |  | sec-causal-inference.html | chapter |  |
| sec-time-series-data.html |  | data.html#sec-time-series-data | id |  |
| sec-unequal-probability-sampling.html |  | sampling.html | chapter | stub |
| sec-visualization-did.html |  | sec-difference-in-differences.html#sec-visualization-did | id |  |
| section.html |  | sec-directed-acyclic-graphs.html | chapter |  |
| selection-on-observables-1.html |  | sec-matching-methods.html#sec-selection-on-observables | title |  |
| selection-on-observables.html |  | sec-matching-methods.html#sec-selection-on-observables | id |  |
| selection-on-unobservables-1.html |  | sec-matching-methods.html#sec-selection-on-unobservables | title |  |
| selection-on-unobservables.html |  | sec-matching-methods.html#sec-selection-on-unobservables | id |  |
| selection-problem.html |  | sec-experimental-design.html#selection-problem | id |  |
| semi-random-experiment.html |  | sec-experimental-design.html | chapter |  |
| sensitivity-analysis-robustness-check-1.html |  | sensitivity-analysis-and-robustness-checks.html | chapter |  |
| shiny-app-based-tableau-style.html |  | exploratory-data-analysis.html#shiny-app-based-tableau-style | id(no-old-toc) |  |
| short-cut.html |  | appendix.html#short-cut | id |  |
| simple-dif-n-dif.html |  | sec-difference-in-differences.html#sec-simple-difference-in-differences | manual |  |
| simple-sampling.html |  | sampling.html | chapter | stub |
| single-factor-covariance-model.html |  | sec-analysis-of-variance-anova.html#sec-single-factor-covariance-model | id |  |
| sjplot-package.html |  | sec-moderation.html | chapter | stub |
| smoothing-splines.html |  | sec-nonparametric-regression.html | chapter | stub |
| software-and-practical-implementation.html |  | sec-matching-methods.html#software-and-practical-implementation | id |  |
| solutions-to-missing-data.html |  | imputation-missing-data.html | chapter |  |
| special-considerations-for-zero-valued-outcomes.html |  | sec-instrumental-variables.html | chapter |  |
| special-considerations.html |  | sec-matching-methods.html#special-considerations | id |  |
| specification-curve-1.html |  | sensitivity-analysis-and-robustness-checks.html#sec-specification-curve-analysis | manual |  |
| specification-curve-analysis.html |  | sensitivity-analysis-and-robustness-checks.html#sec-specification-curve-analysis | manual |  |
| specification-curve.html |  | sensitivity-analysis-and-robustness-checks.html#sec-specification-curve-analysis | manual |  |
| spline-regression.html |  | sec-nonparametric-regression.html | chapter | stub |
| staggered-dif-n-dif.html |  | sec-difference-in-differences.html#sec-multiple-periods-and-variation-in-treatment-timing | manual |  |
| staggered-difference-in-differences.html |  | sec-difference-in-differences.html#sec-staggered-difference-in-differences | id |  |
| standard-errors-2.html |  | reporting-your-analysis.html | chapter |  |
| standard-errors-3.html |  | reporting-your-analysis.html | chapter |  |
| standard-errors-4.html |  | reporting-your-analysis.html | chapter |  |
| statistical-test-selection-for-comparing-means.html |  | sec-multivariate-methods.html#statistical-test-selection-for-comparing-means | id |  |
| steps-for-fuzzy-rd.html |  | sec-regression-discontinuity.html#steps-for-fuzzy-rd | id |  |
| steps-for-matching.html |  | sec-matching-methods.html#steps-for-matching | id |  |
| steps-for-rdit-regression-discontinuity-in-time.html |  | sec-temporal-discontinuity-designs.html#sec-regression-discontinuity-in-time | manual |  |
| steps-for-sharp-rd.html |  | sec-regression-discontinuity.html#steps-for-sharp-rd | id |  |
| stratified-sampling.html |  | sampling.html | chapter | stub |
| summary-1.html |  | sec-nonlinear-and-generalized-linear-mixed-models.html | chapter | stub |
| summary-2.html |  | sec-nonlinear-and-generalized-linear-mixed-models.html | chapter | stub |
| summary-statistics-1.html |  | exploratory-data-analysis.html#summary-statistics | manual |  |
| summary-table-1.html |  | variable-selection.html | chapter | stub |
| summary.html |  | basic-statistical-inference.html#summary | id(no-old-toc) |  |
| survivorship-bias-1.html |  | sec-biases.html#sec-survivorship-bias | title |  |
| survivorship-bias.html |  | sec-biases.html#sec-survivorship-bias | id |  |
| synthetic-controls-with-staggered-adoption.html |  | sec-synthetic-control.html#synthetic-control-with-staggered-adoption | manual |  |
| synthetic-data-an-overview-1.html |  | replication-and-synthetic-data.html | chapter |  |
| synthetic-data-an-overview.html |  | replication-and-synthetic-data.html | chapter |  |
| synthetic-data.html |  | replication-and-synthetic-data.html#synthetic-data | id |  |
| synthetic-difference-in-differences-1.html |  | sec-synthetic-difference-in-differences.html | title |  |
| synthetic-difference-in-differences.html |  | sec-synthetic-difference-in-differences.html | chapter |  |
| table.html |  | bookdown-cheat-sheet.html#table | id |  |
| temporal-discontinuity-designs.html |  | sec-temporal-discontinuity-designs.html | chapter |  |
| testing-in-event-studies.html |  | sec-event-studies.html#sec-testing-in-event-studies | id |  |
| testing.html |  | sec-event-studies.html | chapter |  |
| the-7-tools-of-structural-causal-models.html |  | sec-causal-inference.html | chapter |  |
| the-formal-notation-of-causality.html |  | sec-causal-inference.html#the-formal-notation-of-causality | id |  |
| the-likelihood-ratio-test.html |  | hypothesis-testing.html | chapter | stub |
| the-philosophy-of-robustness.html |  | sensitivity-analysis-and-robustness-checks.html#the-philosophy-of-robustness | id(no-old-toc) |  |
| the-replication-standard-1.html |  | replication-and-synthetic-data.html#the-replication-standard | title |  |
| the-replication-standard.html |  | replication-and-synthetic-data.html#the-replication-standard | id |  |
| theoretical-considerations.html |  | sec-synthetic-control.html#theoretical-considerations | id |  |
| three-way-interactions.html |  | sec-moderation.html | chapter | stub |
| time-series-1.html |  | data.html#sec-time-series-data | manual |  |
| time-series.html |  | data.html#sec-time-series-data | manual |  |
| traditional-approach-1.html |  | sec-mediation.html#traditional-approach | title |  |
| traditional-approach.html |  | sec-mediation.html#traditional-approach | id |  |
| traditional.html |  | sec-mediation.html | chapter |  |
| treatment-effect-types.html |  | sec-causal-inference.html | chapter |  |
| treatment-intensity.html |  | sec-instrumental-variables.html#treatment-intensity | id |  |
| two-one-sided-tests-tost-equivalence-testing.html |  | hypothesis-testing.html | chapter | stub |
| two-sample-inference.html |  | basic-statistical-inference.html#two-sample-inference | id |  |
| two-stage-randomized-experiments-with-interference-and-noncompliance.html |  | sec-experimental-design.html#two-stage-randomized-experiments-with-interference-and-noncompliance | id |  |
| two-way-fixed-effects.html |  | sec-difference-in-differences.html#sec-two-way-fixed-effects | id |  |
| types-of-hypothesis-testing.html |  | hypothesis-testing.html | chapter | stub |
| types-of-interactions.html |  | sec-moderation.html | chapter | stub |
| types-of-iv.html |  | sec-instrumental-variables.html | chapter |  |
| types-of-moderation-analyses.html |  | sec-moderation.html | chapter | stub |
| types-of-regression-discontinuity-designs.html |  | sec-regression-discontinuity.html#types-of-regression-discontinuity-designs | id |  |
| types-of-subjects-in-a-treatment-setting.html |  | sec-causal-inference.html#types-of-subjects-in-a-treatment-setting | id |  |
| understanding.html |  | sec-synthetic-difference-in-differences.html#understanding | id |  |
| unequal-probability-sampling.html |  | sampling.html | chapter | stub |
| using-multiple-outcomes-to-improve-the-synthetic-control-method.html |  | sec-synthetic-control.html#using-multiple-outcomes-to-improve-the-synthetic-control-method | id |  |
| visualization-1.html |  | sec-difference-in-differences.html#sec-visualization-did | title |  |
| visualization.html |  | sec-difference-in-differences.html#sec-visualization-did | title |  |
| visualizations-and-plots-1.html |  | reporting-your-analysis.html#visualizations-plots | manual |  |
| visualizations-and-plots.html |  | reporting-your-analysis.html#visualizations-plots | manual |  |
| visualizations-plots.html |  | reporting-your-analysis.html#visualizations-plots | id(no-old-toc) |  |
| wald-test.html |  | hypothesis-testing.html | chapter | stub |
| weighted-least-squares.html |  | linear-regression.html | chapter | stub |
| why-nonparametric.html |  | sec-nonparametric-regression.html | chapter | stub |
| wrapper-methods-model-based-subset-evaluation.html |  | variable-selection.html | chapter | stub |
