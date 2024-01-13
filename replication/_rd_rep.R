################################################################################
## RD Designs (Parts I and II)
## NBER Summer Institute Methods Lectures, July 2021
## Author: Matias D. Cattaneo and Rocio Titiunik
## Last update: 12-FEB-2022
# https://github.com/rdpackages-replication/CT_2021_NBER

################################################################################
## Website: https://rdpackages.github.io/
################################################################################
## RDROBUST: install.packages('rdrobust')
## RDDENSITY: install.packages('rddensity',dependencies=TRUE)
## RDLOCRAND: install.packages('rdlocrand')
################################################################################
rm(list=ls(all=TRUE))
library(ggplot2)
library(rdrobust)
library(rdlocrand)
library(rddensity)


################################################################################
## Head Start Data
################################################################################
data <- read.csv("https://raw.githubusercontent.com/rdpackages-replication/CT_2021_NBER/main/headstart.csv")
# data <- read.csv("headstart.csv")
attach(data)

Y <- mort_age59_related_postHS
X <- povrate60
Z <- cbind(census1960_pop, census1960_pctsch1417, census1960_pctsch534,
           census1960_pctsch25plus, census1960_pop1417, census1960_pop534,
           census1960_pop25plus, census1960_pcturban, census1960_pctblack)
C <- 59.1984

R <- X - C
T <- (X>C)

################################################################################
## RDPLOTS
################################################################################
rdplot(Y, X, C, binselect="esmv")
rdplot(Y, X, C, p=1)
rdplot(Y, X, C, nbins=1000)

tempdata = as.data.frame(R); colnames(tempdata) = c("v1");
plot2 = ggplot(data=tempdata, aes(tempdata$v1)) + theme_bw(base_size = 17) +
    geom_histogram(data = tempdata, aes(x = v1, y= ..count..), 
                   breaks = seq(min(R,na.rm = TRUE), 1, 1), 
                   fill = "blue", col = "black", alpha = 1) +
    geom_histogram(data = tempdata, aes(x = v1, y= ..count..), 
                   breaks = seq(0, max(R,na.rm = TRUE), 1), 
                   fill = "red", col = "black", alpha = 1) +
    labs(x = "Score", y = "Number of Observations") + geom_vline(xintercept = 0, color = "black")
plot2


################################################################################
## Replicating Ludwig and Miller (2007, QJE)
################################################################################
## Note: HC0 is not available in Stata, but was used by Ludwig and Miller
out <- lm(Y ~ T + R + R*T, subset=(-9<=R & R<=9)); summary(out)

out <- rdrobust(Y, R, h=9, kernel="uni", vce="hc0"); summary(out)
out <- rdrobust(Y, X, C, h=9, kernel="uni", vce="hc0"); summary(out)

rdrandinf(Y, R, 0, wl=-9, wr=9, p=1, reps=10)
rdrandinf(Y, X, C, wl=50.1984, wr=68.1984, p=1, reps=10)

################################################################################
## Local Polynomial Methods
################################################################################
out <- rdrobust(Y, X, C); summary(out)
out <- rdrobust(Y, X, C, h=9, kernel="uni", vce="hc0"); summary(out)
out <- rdrobust(Y, X, C, h=9, kernel="tri", vce="hc0"); summary(out)
out <- rdrobust(Y, X, C, h=9, kernel="tri"); summary(out)
out <- rdrobust(Y, X, C); summary(out)

out <- rdbwselect(Y, X, C, kernel="uni"); summary(out)
out <- rdbwselect(Y, X, C, kernel="uni", all=TRUE); summary(out)
out <- rdbwselect(Y, X, C, all=TRUE); summary(out)

out <- rdrobust(Y, X, C, bwselect="msetwo"); summary(out)

## PLOTS -- LOCAL
out <- rdrobust(Y, X, C, h=9, kernel="uni", vce="hc0"); summary(out)

out <- rdrobust(Y, R, h=9, kernel="uni", vce="hc0"); summary(out)

out <- rdrobust(Y, X, C); summary(out)
rdplot(Y, R, subset=-out$bws[1,1]<= R & R <= out$bws[1,2],
       binselect="esmv", kernel="triangular",
       h=c(out$bws[1,1], out$bws[1,2]), p=1)

## RDROBUST with covariates
out <- rdrobust(Y, X, C)
len1 <- out$ci[3,2] - out$ci[3,1]
out <- rdrobust(Y, X, C, covs=Z)
len2 <- out$ci[3,2] - out$ci[3,1]
paste("CI length change: ", round((len2/len1-1)*100,2), "%")


################################################################################
## Local Randomization Methods
################################################################################
rdrandinf(Y, X, C, wl=57, wr=61, reps=1000)

rdrandinf(Y, X, C, wl=57, wr=61, reps=1000, stat="all")

rdwinselect(X, cbind(census1960_pop,census1960_pcturban,census1960_pctblack),
            cutoff=59.1984, obsmin=10, wobs=5, nwindows=20, seed=43)

id = (57<=X & X<=61); rdplot(Y[id], X[id], C, nbins=1000, p=0)


################################################################################
## Falsification/Validation Methods
################################################################################

## Density Tests: Binomial and Continuity-Based
tempdata = as.data.frame(R); colnames(tempdata) = c("v1");
plot2 = ggplot(data=tempdata, aes(tempdata$v1)) + theme_bw(base_size = 17) +
    geom_histogram(data = tempdata, aes(x = v1, y= ..count..), 
                   breaks = seq(min(R,na.rm = TRUE), 1, 1), 
                   fill = "blue", col = "black", alpha = 1) +
    geom_histogram(data = tempdata, aes(x = v1, y= ..count..), 
                   breaks = seq(0, max(R,na.rm = TRUE), 1), 
                   fill = "red", col = "black", alpha = 1) +
    labs(x = "Score", y = "Number of Observations") + geom_vline(xintercept = 0, color = "black")
plot2

out <- rddensity(R); summary(out);
rdplotdensity(out, R)

rdwinselect(X, cutoff=C)

## Pre-intervention covariates and placebo outcomes
rdplot(Y, X, C)
rdplot(census1960_pop, X, C)
rdplot(census1960_pctsch1417, X, C)

summary(rdrobust(Y, X, C, h=9, kernel="uni"))
summary(rdrobust(Y, X, C))

summary(rdrobust(census1960_pop, X, C, h=9, kernel="uni"))
summary(rdrobust(census1960_pop, X, C))

summary(rdrobust(census1960_pctsch1417, X, C, h=9, kernel="uni"))
summary(rdrobust(census1960_pctsch1417, X, C))

## Placebo cutoff
rdplot(Y, R, p=2, binselect="esmv")
summary(rdrobust(Y[R>0], R[R>0], c=3))

## Recall RD Effect
out <- rdrobust(Y, X, C); summary(out)
rdplot(Y, R, subset=-out$bws[1,1]<= R & R <= out$bws[1,2],
       binselect="esmv", kernel="triangular",
       h=c(out$bws[1,1], out$bws[1,2]), p=1)

## Different bandwidths
summary(rdbwselect(Y, R, all=TRUE))
summary(rdrobust(Y, R))
summary(rdrobust(Y, R, h=9))
summary(rdrobust(Y, R, h=4))

## Donut hole
summary(rdrobust(Y, R))
summary(rdrobust(Y[R<=-.1 | .1 <= R], R[R<=-.1 | .1 <= R]))


