################################################################################################
###########################  The Value of Descriptive Analytics:  ############################## 
###########################    Evidence from Online Retailers     ##############################
###########################    by Ron Berman and Ayelet Israei    ##############################
###########################        Marketing Science, 2022        ##############################
################################################################################################
# This R file runs the analysis for the main results in the SynthDiD section of the  paper.    #
# The file uses a limited data file including randomly selected 5% of the retailer-channels,   #
# so results differ from those reported in the paper.                                          #
###############################################################################################
rm(list=ls())
library(data.table)
# source("SynthDiD-fun.R")
source(
    file.path(
        getwd(),
        "replication",
        "sdid_mksc.2022.1352",
        "data and replication files",
        "SynthDiD-fun.R"
    )
)

data =
    # fread("data_file_for_replication_small.csv")
    data.table::fread(file.path(
        getwd(),
        "replication",
        "sdid_mksc.2022.1352",
        "data and replication files",
        "data_file_for_replication_small.csv"
    ))

set.seed(1)
lags=6
leads=5
target_leads=5

cohorts = 7:18 ### Jan 2016 to June 2017
cohort = 7

timevar = "month_id"
idvar = "company_source"
joinvar = "month_join_id"
treatvar="m_after"
yvar="m_log_rev_usd"
# Main analysis, as in Columns (3) and (4) of Table 5 in the paper
estNplot(
    data = data,
    placebo = F,
    pooled = F,
    months = cohorts,
    lags = lags,
    leads = leads,
    treatvar = treatvar,
    timevar = timevar,
    idvar = idvar,
    yvar = yvar,
    joinvar = joinvar,
    prefix = "replication_"
)
estSC(data = data, months = cohorts, lags = lags, leads = leads, timevar = timevar, idvar = idvar, joinvar = joinvar, treatvar = treatvar, yvar = yvar, placebo = T)

library(tidyverse)
data <- fixest::base_stagg |>
    dplyr::mutate(treatvar = if_else(time_to_treatment >= 0, 1, 0)) |>
    dplyr::mutate(treatvar = as.integer(if_else(year_treated > (5 + 2), 0, treatvar)))

set.seed(1)
est = estSC(
    data = data,
    months = 5:7,
    lags = 2,
    leads = 2,
    timevar = "year",
    idvar = "id",
    joinvar = "year_treated",
    treatvar = "treatvar",
    pooled = F,
    placebo = F,
    yvar = "y"
)


plotATEs <- function(x,y,l,u,...) {
    plot(x=x, y=y,xaxt='null',...)
    polygon(c(x,rev(x)),c(l,rev(u)),col = "lightblue", border = FALSE)
    points(x=x, y=y, pch=16)
    lines(x=x, y=y)
    abline(h=0, col="red")
    lines(x, u, col="lightblue4")
    lines(x, l, col="lightblue4")
    axis(1,x,font=1)
}

ylim = c(min(est$TE.l.mean.w), max(est$TE.u.mean.w))
plotATEs(
    est$time,
    est$TE.mean.w[1:length(est$time)],
    est$TE.l.mean.w[1:length(est$time)],
    est$TE.u.mean.w[1:length(est$time)],
    ylim = ylim,
    main = paste(yvar, "ATEs with jackknife SEs"),
    xlab = "time",
    ylab = "ATE"
)



get_synthdid(
    data = data,
    cohort = "7",
    lags = lags,
    leads = leads,
    timevar = timevar,
    idvar = idvar,
    joinvar = joinvar,
    treatvar = "m_after",
    yvar = yvar
)


