################################################################################################
###########################  The Value of Descriptive Analytics:  ############################## 
###########################    Evidence from Online Retailers     ##############################
###########################    by Ron Berman and Ayelet Israei    ##############################
###########################        Marketing Science, 2022        ##############################
################################################################################################
# This R file contains the functions used by SynthDiD.R                                        #
################################################################################################

library(plm) # Used to make sure panels are complete and balanced
if (!require(devtools)) install.packages("devtools") # Needed to install packages from GitHub
if (!require(synthdid)) devtools::install_github("synth-inference/synthdid") # Install SynthDiD package from Github
library(synthdid) # The synthetic difference in differences library
library(stargazer) # Used to generate regression tables

### These functions are a modification of functions
# From the SynthDiD package to allow us to handle 
# Staggered adoption and creating event study plot
###

# Placebo Standard Errors: Algorithm 4 of Arkhangelsky et al.
# Used when there is one treated unit
placebo_se = function(estimate, replications=500) {
  setup = attr(estimate, 'setup')
  opts = attr(estimate, 'opts')
  weights = attr(estimate, 'weights')
  N1 = nrow(setup$Y) - setup$N0
  if (setup$N0 <= N1) { stop('must have more controls than treated units to use the placebo se') }
  theta = function(ind) {
    N0 = length(ind)-N1
    weights.boot = weights
    weights.boot$omega = sum_normalize(weights$omega[ind[1:N0]])
    # Original synthdid line:   
    # do.call(synthdid_estimate, c(list(Y=setup$Y[ind,], N0=N0,  T0=setup$T0,  X=setup$X[ind, ,], weights=weights.boot), opts))
    # Replace with our get_estimate function:
    est.pl=do.call(get_estimate, list(Y=setup$Y[ind,], N0=N0,  T0=setup$T0, weights=weights.boot))
    est.pl$est
  }
  sqrt((replications-1)/replications) * apply(replicate(replications, theta(sample(1:setup$N0))),1,sd)
}

# The fixed-weights jackknife estimate of variance: Algorithm 3 of Arkhangelsky et al.
# if weights = NULL is passed explicitly, calculates the usual jackknife estimate of variance.
# returns NA if there is one treated unit or, for the fixed-weights jackknife, one control with nonzero weight
jackknife_se = function(estimate, weights = attr(estimate, 'weights')) {
  setup = attr(estimate, 'setup')
  opts = attr(estimate, 'opts')
  
  if (!is.null(weights)) {
    opts$update.omega = opts$update.lambda = FALSE
  }
  
  if (setup$N0 == nrow(setup$Y) - 1) {
    return(placebo_se(estimate))
  }
  
  if (!is.null(weights) && sum(weights$omega != 0) == 1) { return(NA) }
  
  theta = function(ind) {
    weights.jk = weights
    if (!is.null(weights)) { weights.jk$omega = sum_normalize(weights$omega[ind[ind <= setup$N0]]) }
    # Original synthdid line:
    #    estimate.jk = do.call(synthdid_estimate,
    #                          c(list(Y=setup$Y[ind, ], N0=sum(ind <= setup$N0), T0=setup$T0, X = setup$X[ind, , ], weights = weights.jk), opts))
    # Replace with our get_estimate function:
    est.jk = do.call(get_estimate,
                     list(Y=setup$Y[ind, ], N0=sum(ind <= setup$N0), T0=setup$T0, weights = weights.jk))
    est.jk$est
  }
  jackknife(1:nrow(setup$Y), theta)
}

# Normalize weights to equal to one.
sum_normalize = function(x) {
  if(sum(x) != 0) { x / sum(x) }
  else { rep(1/length(x), length(x)) }
  # if given a vector of zeros, return uniform weights
  # this fine when used in bootstrap and placebo standard errors, where it is used only for initialization
  # for jackknife standard errors, where it isn't, we handle the case of a vector of zeros without calling this function.
}

#' Jackknife standard error of function `theta` at samples `x`.
#' @param x vector of samples
#' @param theta a function which returns a scalar estimate
#' @importFrom stats var
#' @keywords internal
jackknife = function(x, theta) {
  n = length(x)
  u = NULL
  # Original synthdid code assumed only the TE is output and worked on a vector.
  # We compute the variance for each column in a matrix.
  # Each column in a time period or aggregate TE
  for (i in 1:n) {
    u = rbind(u,theta(x[-i]))
  }
  jack.se = sqrt(apply(u,2,var)*((n - 1) / n) * (n - 1))
  
  jack.se
}

###
# Given the estimated synthdid object/output, output estimates of TEs per period.
# Y - data
# N0 - Number of control units
# T0 - No. pre-treamtment periods.
# weights - output from synthdid
#
# Return list of:
# est - TE per time period and cumulative ATEs for post-treatment periods
# y_obs - observed outcomes for treated units
# y_pred - predicted outcomes for treated units
# lambda.synth - synthetic control lambda weights
# Ntr - number of treated units
# Nco - number of control units
####
get_estimate <- function(Y, N0, T0, weights)
{
  N1 = nrow(Y) - N0 # Number of treated units
  T1 = ncol(Y) - T0 # Number of post-treat periods
  
  # lambda and omega weights from synthdid per time period or unit
  lambda.synth = c(weights$lambda, rep(0, T1))
  lambda.target = c(rep(0, T0), rep(1 / T1, T1))
  lambda = lambda.synth+lambda.target
  omega.synth = c(weights$omega, rep(0, N1))
  omega.target = c(rep(0, N0), rep(1 / N1, N1))
  
  # compute the intercept offset between treated and control
  over=1
  intercept.offset = over * c((omega.target - omega.synth) %*% Y %*% lambda.synth)
  
  obs.trajectory = as.numeric(omega.target %*% Y)   # observed trajectory 
  syn.trajectory = as.numeric(omega.synth %*% Y) + intercept.offset # synthetic trajectory
  
  # Per unit observed and predicted trajectories
  # Useful for placebo, aggregate and other robustness tests
  y_obs = matrix(t(t(Y[-(1:N0),])*lambda), nrow=N1)
  y_pred = t(apply(Y[-(1:N0),] %*% lambda.synth, 1, function(x) x+((omega.synth %*% Y)-c(omega.synth %*% Y %*% lambda.synth)))*lambda)
  if (T1>1) {
    if (N1>1 ) {
      y_obs = cbind(y_obs, t(apply(y_obs[,-(1:T0)],1,cumsum)))
      y_pred = cbind(y_pred, t(apply(y_pred[,-(1:T0)],1,cumsum)))
      
    }
    if (N1==1 ) {
      y_obs = matrix(c(y_obs, cumsum(y_obs[,-(1:T0)])),nrow=N1)
      y_pred = matrix(c(y_pred, cumsum(y_pred[,-(1:T0)])),nrow=N1)
      
    }
  }
  
  # Compute the TEs
  effects = (obs.trajectory-syn.trajectory)*(lambda.synth+lambda.target*T1)
  # Add an average TE for post-treatment periods.
  effects = c(effects, cumsum(effects[-(1:T0)])/seq_along(effects[-(1:T0)]))
  
  return(list(est=effects, y_obs=y_obs, y_pred=y_pred, lambda.synth=lambda.synth, Ntr=N1, Nco=N0))
}

###
# Estimate syntehtic dif-in-diff using the synthdid package and modifications
# Inputs:
# data - data frame to analyze
# cohort - which cohort in data to use as treated
# subgroup - list of IDs to use as treated subgroup
# lags - number of lags to use pre-treatment
# leads - number of post-treatment periods (0 means only the treatment period)
# time_var - calendar time column name
# idvar - unit id column name
# joinvar - treatment time period column name
# treatvar - treatment indicator column name
# yvar - outcome variable column name
#
# Return list with items:
# est - TE per time period and cumulative ATEs for post-treatment periods
# se - standard errors of each time period
# y_obs - observed outcomes for treated units
# y_pred - predicted outcomes for treated units
# lambda.synth - synthetic control lambda weights
# Ntr - number of treated units
# Nco - number of control units
###
get_synthdid <- function (data, cohort, subgroup=NULL, lags, leads, timevar="month_id", idvar="company_source", joinvar="month_join_id", treatvar="m_after",yvar) {
  # extract columns from data, remove missing values, change treatment to logical.
  data2=data
  data2[,treatvar] = as.logical(data2[,treatvar])
  data2 = data2[,c(treatvar,timevar,idvar,joinvar,yvar)]
  if(sum(is.na(data2))>0) {
    data2=remove_missing(data2,idvar, yvar)
  }
  
  # Setup data structure for synthdid package
  setup=panel.matrices(panel=data2, unit=idvar, time=timevar, outcome=yvar, treatment=treatvar)
  # Run synthdid estimation
  sdid = synthdid_estimate(setup$Y, setup$N0, setup$T0)
  
  # If doing subgroup analysis, change the setup matrix accordingly
  # so the treated are only the subgroup of intrest
  if (!is.null(subgroup)) {
    treat_ids = rownames(setup$Y[-(1:setup$N0),]) # get all treated unit IDs
    keep = c(rep(TRUE,setup$N0), treat_ids %in% subgroup) # keep only those in subgroup
    sdid.setup = attr(sdid,'setup') # change the setup matrix to contain only that data.
    sdid.setup$Y=sdid.setup$Y[keep,]
    sdid.setup$X=sdid.setup$X[keep,,]
    attr(sdid,'setup')<-sdid.setup
  }
  
  # Extract outcomes from synthdid estimates
  setup = attr(sdid, 'setup')
  weights = attr(sdid, 'weights')
  Y = setup$Y - synthdid:::contract3(setup$X, weights$beta)
  N0 = setup$N0; 
  T0 = setup$T0; 
  
  # Pass to our own function that computes TEs by time period (and not only for the target ATT)
  est=get_estimate(Y, N0, T0, weights)
  # Compute SEs using the Jackknife
  se=jackknife_se(sdid)
  
  return(list(est=est$est, se=se,y_pred=est$y_pred, y_obs=est$y_obs, lambda.synth=est$lambda.synth, Ntr=est$Ntr, Nco=est$Nco))
}


# Will remove units with missing data in a time period
# Input:
# data: data 
# timevar: column with time
# idvar: column of unit IDs
# yvar: column of outcomes
#
# Return: outcome data in wide format with columns with missing values removed
remove_missing <- function(data, idvar, yvar) {
  dt = as.data.table(data)
  dt[, nas := min(get(yvar)), by=idvar]
  dt=na.omit(dt)
  dt$nas=NULL
  return(as.data.frame(dt))
}

# Return weighted average SE wegithed by weights
sum_SEs<-function(se, weights=rep(1, sum(!is.na(se)))) {
  sqrt(sum(se^2*weights^2, na.rm=T)/sum(weights)^2)
}


# Estimate the full SC ATEs and Standard Errors
#
# Input:
# data: long format of data to analyze
# months: which months in data to use for join times
# lags, leads: how many lags and leads of join time to analyze
# timevar: column name of time variables
# idvar: ID column of companies
# joinvar: column with join time of each unit
# yvar: column of outcome to analyze
# placebo: run placebo analysis
# pooled: run pooled analysis of all treated units
# subgroup: IDs for subgroup analysis
#
# Return: A list with:
#
#time: time periods used in estimation from -lags to leads
#TE.mean: ATT in each time period
#SE.mean: Standard error of ATT each time period
#TE.l.mean: Lower C.I. for ATT per period
#TE.u.mean: Upper C.I. for ATT per period
#TE.mean.w, SE.mean.w, TE.l.mean.w, TE.u.mean.w: Same as above but weighted by number
#                                                of treated units in each time period
#Ntr: No. treated units
#Nco: No. control units
#TE: Treatment effect for each cohort in each time period
#SE: Standard error of TE of each cohort in each time period
#y_obs: observed outcomes of teated units
#y_pred: predicted outcomes of treated units
# time: time from treatment
# cnames: column names for TE and SE matrices (times and ATTs)
estSC <- function(data, months, lags, leads,timevar="month_id", idvar="company_source", joinvar="month_join_id", treatvar="m_after",yvar, placebo=F, pooled=F, subgroup=NULL) {
  data = data.frame(data)
  # Return values
  TE=NULL
  SE=NULL
  y_pred=NULL
  y_obs=NULL
  
  Ntr = NULL
  Nco = NULL
  out_months=NULL
  
  for (cohort in months) {
    cat("Cohort:",cohort,"\n" )
    
    # Create balanced panel with lags and leads for all units treated in cohort
    # as treated and all units treated outside the panel after leads as control
    data2=get_balanced_panel(data=data, cohort=cohort, lags=lags, leads=leads, timevar=timevar, idvar=idvar,joinvar=joinvar)
    # Compute number of treated units
    n_treat = length(unique(data2[data2[,joinvar]==cohort, idvar]))
    
    # If Placebo analysis, select a random group of control untis
    # to serve as treated and readjust the data
    if (placebo) { 
      cat("Running Placebo analysis\n")
      ctrl_ids = unique(data2[data2[,joinvar]>cohort,idvar])
      placebo_treat_ids = sample(ctrl_ids, n_treat, replace=F)
      data2 = data2[data2[,joinvar]>cohort,]
      data2[data2[,idvar] %in% placebo_treat_ids, joinvar] = cohort
      data2[data2[,timevar]>=data2[,joinvar],treatvar] = 1
    }
    
    # If pooled analysis, replace treated unit data with average outcome of all treated units
    if (pooled) {
      cat("Running Pooled analysis\n")
      treat_data = data2[data2[,joinvar]==cohort,c(timevar, treatvar,joinvar,yvar, idvar)]
      data2 = data2[data2[,joinvar]>cohort,c(timevar, treatvar,joinvar,yvar, idvar)]
      
      for (time in seq(cohort-lags,cohort-1)) {
        new_data = c(time, 0,cohort,mean(treat_data[treat_data[,joinvar]==cohort,yvar]),0)
        data2=rbind(data2,new_data)             
      }
      
      for (time in seq(cohort+0,cohort+leads)) {
        new_data = c(time, 1,cohort,mean(data2[data2[,joinvar]==cohort,yvar]),0)
        data2=rbind(data2,new_data)             
      }
    }
    
    # Make sure all units treated outside cohort panel are marked as not treated
    data2[data2[,joinvar] > cohort, treatvar]=0
    # data2[data2[,joinvar] > (cohort + lags), treatvar]=0
    
    # recompute number of treated and control if we changed the data
    n_control = length(unique(data2[data2[,joinvar]>cohort,idvar]))
    n_treat = length(unique(data2[data2[,joinvar]==cohort, idvar]))
    if (!is.null(subgroup)) {
      n_treat = sum(unique(data2[data2[,joinvar]==cohort, idvar]) %in% subgroup)
    }
    
    cat("Treated:", n_treat,"Control:", n_control ,"\n")
    
    # If there are treated units to analyze, run synthetic diff-in-diff 
    if (n_treat>0) {
      Ntr = c(Ntr, n_treat)
      Nco = c(Nco, n_control)
      
      out_months = c(out_months, cohort) # keep cohort analyzed names
      sdid=get_synthdid(data=data2, cohort=cohort, subgroup=subgroup, lags=lags, leads=leads, timevar=timevar, idvar=idvar, joinvar=joinvar, treatvar=treatvar,yvar=yvar)
      
      # Save output of analysis in matrices as new row
      TE = rbind(TE, sdid$est)
      SE = rbind(SE, sdid$se)
      y_pred=rbind(y_pred, sdid$y_pred)
      y_obs=rbind(y_obs, sdid$y_obs)
    }
  }
  
  # Aggregate the cohort level ATTs, SEs etc into an overall value
  time = seq(-lags, leads)
  cnames = c(time, paste("c.", 0:leads,sep=""))
  TE=data.frame(TE)
  colnames(TE)<-cnames
  rownames(TE)<-out_months
  
  TE.mean.w=colSums(TE*Ntr/sum(Ntr))
  TE.mean=colMeans(TE)
  
  SE=data.frame(SE)
  colnames(SE)<-cnames
  rownames(SE)<-out_months
  SE.mean.w=apply(SE, 2, sum_SEs,Ntr)
  SE.mean=apply(SE, 2, sum_SEs)
  
  TE.l.mean = TE.mean-1.96*SE.mean
  TE.u.mean = TE.mean+1.96*SE.mean
  
  TE.l.mean.w = TE.mean.w-1.96*SE.mean.w
  TE.u.mean.w = TE.mean.w+1.96*SE.mean.w
  
  colnames(y_obs) <- cnames
  colnames(y_pred) <- cnames
  
  return(list(TE.mean=TE.mean,SE.mean = SE.mean, TE.l.mean=TE.l.mean,TE.u.mean=TE.u.mean, TE.mean.w=TE.mean.w,SE.mean.w = SE.mean.w,TE.l.mean.w=TE.l.mean.w,TE.u.mean.w=TE.u.mean.w, Ntr=Ntr, Nco=Nco,TE=TE, SE=SE, y_obs=y_obs, y_pred=y_pred,time=time, cnames=cnames))
}

# Plot ATEs
#
#Input:
# x: time periods
#y: ATE per time period
# l: lower C.I. for each time period
# u: upper C.I. for each time period
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


# Save outcome from estSC to csv files.
# input:
# est: outcome of estSC
# base: base file name to attach to output csv files.
savecsv <- function(est, base) {
  write.csv(cbind(time=names(est$TE.mean), TE=est$TE.mean, SE=est$SE.mean, lower=est$TE.l.mean, upper=est$TE.u.mean,TE.w=est$TE.mean.w, SE.w=est$SE.mean.w, lower.w=est$TE.l.mean.w, upper.w=est$TE.u.mean.w), file=paste(base,".mean.csv", sep=""))
  write.csv(est$TE, file=paste(base,".TE.csv", sep=""))
  write.csv(est$SE, file=paste(base,".SE.csv", sep=""))
  write.csv(est$y_obs, file=paste(base,".y_obs.csv", sep=""))
  write.csv(est$y_pred, file=paste(base,".y_pred.csv", sep=""))
}

# Run above functions for depvar, plot, and save file with special prefix prefix, also save plots
#
# Input:
# prefix: string prefix to use for filenames
# pooled: run pooled analysis
# placebo: run placebo analysis
# subgroup: IDs for subgroup analysis
# data: long format of data to analyze
# months: which months in data to use for join times
# lags, leads: how many lags and leads of join time to analyze
# timevar: column name of time variables
# idvar: ID column of companies
# joinvar: column with join time of each unit
# yvar: column of outcome to analyze
#
estNplot <- function(prefix="",pooled=F, placebo=F, subgroup=NULL, data, months, lags, leads, treatvar, timevar, idvar, yvar, joinvar) {
  
  # Run complete synthetic diff-in-diff analysis by cohort and aggregate
  start_all = Sys.time()
  
  est = estSC(data, placebo=placebo, pooled=pooled,subgroup=subgroup, months=months, lags=lags, leads=leads, treatvar=treatvar, timevar=timevar, idvar=idvar, yvar=yvar, joinvar=joinvar)
  
  end_all = Sys.time()
  print(end_all-start_all)
  
  # Generate charts
  ylim=c(min(est$TE.l.mean.w),max(est$TE.u.mean.w))
  jpeg(paste(prefix,yvar,".JK.jpg", sep=""),width=600, height=240)
  plotATEs(est$time, est$TE.mean.w[1:length(est$time)], est$TE.l.mean.w[1:length(est$time)], est$TE.u.mean.w[1:length(est$time)],ylim=ylim, main=paste(yvar, "ATEs with jackknife SEs"), xlab="time", ylab="ATE")
  dev.off()
  plotATEs(est$time, est$TE.mean.w[1:length(est$time)], est$TE.l.mean.w[1:length(est$time)], est$TE.u.mean.w[1:length(est$time)],ylim=ylim, main=paste(yvar, "ATEs with jackknife SEs"), xlab="time", ylab="ATE")

  # Create regression like tables
  d <- as.data.frame(matrix(rnorm(10 * (length(est$cnames)+1)), nc = length(est$cnames)+1))
  cnames = c(paste("l",seq(lags,1,-1),sep="."),paste("f",seq(0,leads),sep="."),paste("c",seq(0,leads),sep="."))
  colnames(d) <- c(yvar, cnames)
  fm =paste(yvar, "~ 0 +", paste(cnames, collapse = "+"))
  f <-as.formula(fm)
  p <- lm(f,d)
  
  coef = list(est$TE.mean.w)
  se = list(est$SE.mean.w)
  t = list(unlist(coef)/unlist(se))
  names(coef[[1]])=names(p$coefficients)
  names(se[[1]])=names(p$coefficients)
  names(t[[1]])=names(p$coefficients)
  stargazer(p, type = "text", coef = coef, se = se, t = t, omit.stat = "all",out=paste(prefix,yvar,".weighted.txt",sep=""), star.char = c("+", "*", "**"),star.cutoffs = c(.1, .05, .01))
}

# Extract balaned panel from data with specific cohort
# It drops units that have missing observations in the panel
# Keep units that are either treated in a specific cohort as treated, or first treated
# outside the panel after number of leads from the cohort as control
# Input:
# data: data to use
# cohort: which treatment cohort to take as treated
# lags, leads: number of months before and after treatment for panel length
# timevar: name of time variable
# idvar: name of ID variable
# joinvar: name of time treatment variable
get_balanced_panel <-function(data, cohort, lags, leads, timevar="month_id", idvar="company_source", joinvar="month_join_id") {
  data=as.data.frame(data)
  data2=data[(data[,timevar] >= (cohort-lags)) & (data[,timevar] <= (cohort+leads)) & (data[,joinvar]==cohort | (data[,joinvar]>(cohort+leads))),]
  data2=make.pbalanced(data2, "shared.individuals", index=c(idvar, timevar))
}

