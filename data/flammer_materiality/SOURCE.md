# Environmental CSR events and Eventus results (Flammer 2013 replication)

This folder is an analysis-ready extract of the replication package for:

> Kofford, Steven D., Ben W. Lewis, and Myles Melancon (2026), "Materiality and
> investor reaction to environmental corporate social responsibility: A
> replication and extension of Flammer (2013)," *Strategic Management Journal*.
> https://doi.org/10.1002/smj.70135

Source: Kofford, Lewis, and Melancon, "Replication Data for: Materiality and
Investor Reaction to Environmental CSR: A Replication and Extension of Flammer
(2013)," Harvard Dataverse, version 1.0, https://doi.org/10.7910/DVN/CAWEJW,
released under CC0 1.0 (public domain dedication). The original study is
Flammer (2013), *Academy of Management Journal* 56(3), 758-781,
https://doi.org/10.5465/amj.2011.0744.

The package does not contain CRSP, Compustat, IBES, or Eventus data, and
neither does this extract. Per-event abnormal returns are therefore not here.

## Files

`events.csv`, 512 rows, one per event. From the package's hand-coded files:
`Flammer 2013 - Sample.dta` (company, issue, event date, valence, calendar
year, replication and extension flags), `Flammer 2013 - SASB Materiality.dta`
(SASB sector, industry, category, and the 0/1 materiality flag), and
`Flammer 2013 - Confounding Events.dta` (the 0/1 confound flag for the
(-1, 0) window). Joined to these are the market-model parameters that Eventus
printed for each event in the two Table 4 listings
(`documentation/eventus_output/Flammer 2013 - Table 4 - Eco-friendly (-1,0).txt`
and `... Eco-harmful (-1,0).txt`): mean total return, share of raw returns
above zero, alpha, beta, share of market-model residuals above zero, total
return variance, residual standard deviation, and first-order residual
autocorrelation, all estimated on the 200 trading days ending 41 days before
the event against the CRSP equally weighted index.

`eventus_cells.csv`, one row per published cell and window, parsed from the
49 result listings in `documentation/eventus_output/`: count, mean CAR,
precision-weighted CAAR, positive and negative counts, Patell Z (or the
standardized cross-sectional Z for the four-factor job), generalized sign Z,
and the portfolio time-series (CDA) t where the listing's layout prints it.
`phat_listing` is the mean share of positive estimation-window residuals over
the events in that listing. Mean CARs are printed by Eventus to two decimals
of a percent and are stored at that precision.

## What was changed

Three SASB category labels in the deposit are spelled inconsistently and are
normalized: "Water & Wastewater Managementt", "Employee Health and Safety",
and "GHG emissions". Nothing else is altered. The materiality flag is taken as
deposited.

## Checks

`build.R` stops unless the sample has 512 events and every event matches a
parameter row. The book chapter (section `sec-csr-materiality-replication` in
`33-event-study.Rmd`) further verifies that the hand coding reproduces the
event count of every listing it can define and that the generalized sign Z
recomputed from these files matches every printed value to rounding.

Rebuild with

    Rscript data/flammer_materiality/build.R "<path>/Replication Package" data/flammer_materiality
