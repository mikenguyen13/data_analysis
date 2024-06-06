#NOTE: code is modified from rdbounds.R v1.01 on 3/11/2020 by LG.
#Specifically, the function rdbounds_export is modified to contain
#only those outputs in the final Quantitative Economics publication.

#--------------------------------------------------------------------------
#VERSION 1.01 (JUNE 12, 2019)
#DISCLAIMER: THIS CODE SHOULD BE CONSIDERED PRELIMINARY AND IS OFFERED WITHOUT WARRANTY.
#WE APPRECIATE YOUR COMMENTS AND ANY ISSUES NOTICED, AT LEONARD.GOFF AT COLUMBIA DOT EDU
#Copyright (C) 2018 Francois Gerard, Leonard Goff, Miikka Rokkanen, & Christoph Rothe.

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

#--------------------------------------------------------------------------
#install.packages("data.table")
#library(data.table)
#install.packages("formattable")
#install.packages("parallel")

#' Manipulation Robust Regression Discontinuity Bounds Estimation
#'
#' This function implements the estimation procedure in Gerard, Rokkanen, and Rothe (2018) to estimate bounds on treatment effects under potential manipulation of the running varible. Returns an \code{rdbounds} object, which can then be passed to \code{\link{rdbounds_summary}} and \code{\link{rdbounds_export}}.
#' Note on refinements: "Refinement A" calculates bounds based on the assumption that always-assigned units are at least as likely to receive treatment than potentially-assigned units (Theorem 3 in paper). "Refinement B" calculates bounds based on the assumption that always-assigned units \eqn{always} receive treatment (Theorem 4 in paper).
#' @param y specifies the outcome/dependent variable. Required.
#' @param x specifies the running variable that determines eligibilty for treatment. Required.
#' @param covs specifies covariates to implement the covariate-based refinement. Expected as a single factor variable.
#' @param treatment specifies the treatment status variable if implementing a Fuzzy RDD. Defaults to computation of Sharp RDD results only.
#' @param c specifies the threshold for assignment to treatment (assigned iff \code{x >= c}). Defaults to 0.
#' @param discrete_x Boolean. If \code{TRUE}, treat each value of x as a mass-point for density estimation. Defaults to FALSE.
#' @param discrete_y Boolean. If \code{TRUE}, treat each value of y as a mass-point for density estimation. Defaults to FALSE.
#' @param bwsx is a vector of bandwidths in x, respectively for 1) estimation of the discontinuity in the density of x at the cutoff; and 2)local polynomial estimation of conditional means. Expects either a single bandwidth to be used for both or a vector of two. Required.
#' @param bwy is a bandwidth for density estimation of y, implemented if \code{discrete_y=FALSE}. Required if \code{discrete_y=FALSE}.
#' @param kernel specificies a kernel function to be used throughout estimation for x. Choices are \code{triangular}, \code{rectangular}, \code{gaussian} and \code{epanechnikov}. Defaults to \code{triangular}.
#' @param orders specifies the order of polynomial regression, for: 1) estimation of the discontinuity in the density at the cutoff (\eqn{\tau} in paper), and 2) local polynomial regressions. Expects either a single integer to be used for both or a vector of two values. Defaults to 1 (local linear regression) for all. Estimation of \eqn{\tau} can only be implemented up to quadratic order if \code{discrete_X=FALSE}.
#' @param evaluation_ys an explicit vector of y-values to evaluate CDF's at (and PDF's if \code{discrete_y = FALSE}). If \code{evaluation_ys} is not set, the set of unique values of y in the sample will be used. Caution is required if \code{discrete_y=TRUE}, because computation will assume a probability mass function can be estimated from differences in estimated CDF's at subsequent values of \code{evaluation_ys}. This can bias FRD estimates if \code{evaluation_ys} does not contain all values in the support of y.
#' @param ymin left/lower bound on y at which to implement a boundary kernel correction if \code{discrete_y=FALSE} and \code{y} is a variable with bounded support (e.g. after censoring). Defaults to \code{NULL}, meaning no boundary kernel correction is implemented on the left side of the support of \code{y}.
#' @param ymax right/upper bound on y at which to implement a boundary kernel correction if \code{discrete_y=FALSE} and \code{y} is a variable with bounded support (e.g. after censoring). Defaults to \code{NULL}, meaning no boundary kernel correction is implemented on the left side of the support of \code{y}.
#' @param type "ate" for average treatment effects (default) or "qte" for quantile treatment effects at the percentiles given by parameter \code{percentiles}. Defaults to \code{ate}.
#' @param percentiles vector of percentiles at which to asses quantile treatment effects. Defaults to median (.5). User may add -1 as a percentile, in order to estimate average treatment effects along with QTE's. For example, \code{percentiles=c(-1,.3,.5)} will compute ATEs as well as the 30 percent and 50 percent QTEs
#' @param num_tau_pairs integer number of points to search over in the set of possible values for \eqn{(\tau_0, \tau_1)} in notation of paper, for fuzzy RD estimation. Defaults to 50. If set to 1, the single tau is set to the "rightmost" (t=1) extreme of the set \eqn{T}, such that user can enforce the assumption that always-assigned units always receive treatment (see below), if this is consistent with data.
#' @param refinement_A Boolean. If \code{TRUE}, additionally calculate refined bounds with the restriction that always assigned units are at least as likely to be treated as potentially assigned units (i.e. \eqn{\tau_1\ge \tau}; see Corollary 1 in paper) Defaults to FALSE.
#' @param refinement_B Boolean. If \code{TRUE}, additionally calculate refined bounds for \code{right_effects} with the restriction that always assigned units on the right side of the cutoff are always treated (i.e. \eqn{\tau_0=0}; see Corollary 2 in paper) Defaults to FALSE.
#' @param right_effects boolean. If set to \code{TRUE}, additionally estimate causal effects for units just to the right of the cutoff. Defaults to \code{FALSE}.
#' @param yextremes extreme values \eqn{Y_L} and \eqn{Y_U} to assume if \code{right_effects=TRUE}, e.g. \code{yextremes=c(0,100)}. Defaults to the sample range of \code{y}.
#' @param num_lambdas integer number of points to search over for the causal effect of units just to the right of the cutoff (lambda in paper). Defaults to 50.
#' @param num_bootstraps A vector of the number of bootstrap resamples desired, where the first component is the number of bootstrap samples for estimating confidence intervals, and the second is the number of samples for diagnostic testing of the estimated discontinuity in the density at the cutoff. If a scalar is given, the same number is used for both. Defaults to \code{num_bootstraps = c(100,20)}. To avoid bootstrap testing altogether, set \code{num_bootstraps=NULL} or \code{num_bootstraps=c(0,0)}.
#' @param Kn a hardcoded constant for \eqn{\kappa_n} (see Section 5.2 on inference in paper). Defaults to \eqn{log(n)^{1/2}}, where n is the number of observations.
#' @param alpha sets the level for confidence intervals. Defaults to alpha=.05 for 95 percent confidence intervals.
#' @param potential_taus vector of different values of \eqn{\tau} to use for the confidence intervals estimating the potential impact of manipulation, e.g. \code{potential_taus=c(.025, .05, .1, .2)}.
#' @param parallelize indicates whether to parallelize bootstrap computations across the available number of cores on machine, minus one. Defaults to \code{TRUE}.
#' @param progressFile a file to output progress to (useful if \code{parallelize=TRUE} and the individual cores can't write to screen). File will be appended to.
#' @param warningsFile a file to output full warning messages to from bootstrap estimation if \code{parallelize=TRUE}). File will be appended to.
#' @param kernel_y allows a separate kernel for density estimation of \code{y}. Same choices as kernel for \code{x}. Defaults to kernel specified for use with \code{x}.
#' @param bwsxcov an optional separate \code{bwsx} to use for quantities that are computed on a subsample conditioned on a value of \code{covs} (e.g. covariate-conditional CDFs).
#' @param bwycov an optional separate \code{bwy} to use for quantities that are computed on a subsample conditioned on a value of \code{covs} (e.g. covariate-conditional CDFs).
#' @param CDFinputs optional, the \code{rdbounds$CDFinputs} object from a previous run of \code{\link{rdbounds}} on the same dataset. This can be used to speed up processing by allowing CDF and PDF estimation to be skipped on a second run.

#' @keywords regression discontinuity, RDD, manipulation, manipulation robust regression discontinuity
#' @export
#' @references Francois Gerard, Miikka Rokkanen, and Christoph Rothe (2016)."Bounds on Treatment Effects in Regression Discontinuity Designs under Manipulation of the Running Variable, with an Application to Unemployment Insurance in Brazil". NBER Working Paper 22892.
#' @examples \donttest{df<-rdbounds_sampledata(50000, covs=TRUE)
#' rdbounds_est<-rdbounds(y=df$y,x=df$x, covs=as.factor(df$cov), treatment=df$treatment, c=0,
#'                        discrete_x=FALSE, discrete_y=FALSE,
#'                        bwsx=c(.2,.5), bwy = .1, kernel="epanechnikov", orders=1,
#'                        evaluation_ys = seq(from = 0, to=23, by=.2),
#'                        refinement_A=TRUE, refinement_B=TRUE,
#'                        right_effects=TRUE, yextremes = c(0,23),
#'                        num_bootstraps=0)
#' rdbounds_summary(rdbounds_est, title_prefix="Sample Data Results")}

#' @import data.table

#------------------------------------------------------------------
#------------------------------------------------------------------
#rdbounds FUNCTION: ESTIMATION OF MANIPULATION-ROBUST BOUNDS ON RDD TREATMENT EFFECT
#------------------------------------------------------------------
#------------------------------------------------------------------

rdbounds <- function(y, x, covs = NULL, treatment = NULL, c = 0,
                     discrete_x=FALSE, discrete_y=FALSE, bwsx, bwy=NULL,
                     kernel = "triangular", orders = array(1,dim=c(2)),
                     evaluation_ys=NULL, ymin=NULL, ymax=NULL,
                     type="ate", percentiles=NULL,
                     num_tau_pairs=50, refinement_A=FALSE, refinement_B=FALSE, right_effects=FALSE, yextremes=NULL, num_lambdas=50,
                     num_bootstraps = c(100,20), Kn=NULL, alpha=0.05, potential_taus=NULL, parallelize=TRUE, progressFile=NULL, warningsFile=NULL,
                     kernel_y=NULL, bwsxcov=NULL, bwycov=NULL, CDFinputs=list(original=NULL, CIsetup=NULL)){

  #INTERPRET AND CHECK PARAMETERS
  #------------------------------------------------------------------
  options(warn=1, warnings.length=8170)
  start_time <- Sys.time()

  #Check if user has package data.table installed
  if(!requireNamespace("data.table", quietly = TRUE)) {
    stop("package 'data.table' needed to run rdbounds estimation. Please install it.",
         call. = FALSE)
  }

  #Check if user passed scalar for bwsx, orders, or bootstraps
  if (length(orders)==1) {orders<-array(orders,dim=c(2))}
  if (length(num_bootstraps)==1) {num_bootstraps<-array(num_bootstraps,dim=c(2))}

  if (is.null(kernel_y)){kernel_y=kernel}
  if (is.null(num_bootstraps)){num_bootstraps<-c(0,0)}

  if (length(bwsx)==1) {bwsx<-array(bwsx,dim=c(2))}
  if (length(bwsxcov)==1) {bwsxcov<-array(bwsxcov,dim=c(2))}

  if (is.null(bwy) & discrete_y==FALSE){
    stop("discrete_y = FALSE but no bandwidth specified for y estimation by the bwy parameter.")
  }

  if (is.null(bwsxcov)){bwsxcov<-bwsx}
  if (is.null(bwycov)){bwycov<-bwy}

  #Check that x, y, and treatment (if set) have the same number of observations
  if((length(x)!=length(y))){stop("Error: x and y vectors are not of the same length")}
  if(!is.null(treatment)){
    if((length(x)!=length(treatment))){stop("Error: x and treatment vectors are not of the same length")}
  }

  #Define distance from cutoff variable
  if(is.null(c)){
    c<-0
    warning("c not set, assuming to be zero")
  }
  dist_cut <- x-c

  #Define treatment variable and set fuzzy to boolean indicator of whether to compute FRD
  if (is.null(treatment)) {
    treatment<-(x>=c)
    fuzzy<-FALSE
  }
  else{
    if(length(treatment)==1){
      if(treatment==FALSE){treatment<-(x>=c); fuzzy<-FALSE}
    }
    fuzzy<-TRUE
  }

  #If fuzzy set, check that treatment adds information beyond x
  if(fuzzy & sum(treatment - (dist_cut>=0))==0){
    warning("Treatment variable passed by argument treatment appears to be identical to the condition (x>=c), computing Sharp RDD estimators only")
    fuzzy<-FALSE
  }

  #Check for missing values
  missingVal<- (is.na(y) | is.na(x) | is.na(treatment))
  if(sum(missingVal)>0){
    stop(paste0(sum(missingVal)," observations have missing values. Please re-run with a set of complete observations (no missing values of y, x, or treatment variable (if implementing a fuzzy RDD)."))
  }

  #Check for covariates
  gotCovs<-FALSE

  if(!is.null(covs) & !is.factor(covs)) {stop("covs must be a factor object")}
  if(is.factor(covs)) {
    if((length(x)!=length(covs))){
      stop("Error: factor variable covs does not have the same number of observations as x")
    } else {
      gotCovs<-TRUE
    }

    missingVal<- (is.na(covs))
    if(sum(missingVal)>0){
      stop(paste0(sum(missingVal)," observations of covs have missing values. Please re-run with a set of complete observations."))
    }
  }

  #Warn user if right_effects=TRUE but yextremes not set
  if(right_effects&is.null(yextremes)){
    yextremes<-c(min(y),max(y))
    warning(paste0("yextremes not specified while right_effects set to TRUE, setting as range of y values in sample (",yextremes[1]," to ",yextremes[2], ")"))
  }
  if(!is.null(yextremes)){
    if(yextremes[2]<yextremes[1]){stop("lower bound in yextremes cannot be larger than upper bound in yextremes")}
    if(yextremes[1]>min(y)){stop("there exist values of y smaller than the lower bound in yextremes")}
    if(yextremes[2]<max(y)){stop("there exist values of y larger than the upper bound in yextremes")}
    if(yextremes[1]<min(y)){warning("the lower bound in yextremes is smaller than the smallest y value in sample")}
    if(yextremes[2]>max(y)){warning("the upper bound in yextremes is larger than the largest y value in sample")}
  }

  yboundaries<-list(min=ymin, max=ymax)
  if(!is.null(yboundaries$min) && !is.null(yboundaries$max) && yboundaries$max<yboundaries$min){stop("ymin cannot be larger than ymax")}
  if(!is.null(yboundaries$min) && yboundaries$min>min(y)){stop("there exist values of y smaller than ymin")}
  if(!is.null(yboundaries$max) && yboundaries$max<max(y)){stop("there exist values of y larger than ymax")}

  #Check if user has package 'parallel', if parallelize=TRUE
  if (parallelize==TRUE){
    if(!requireNamespace("parallel", quietly = TRUE)) {
      stop("package 'parallel' needed if parallelize=TRUE. Please install it.",
           call. = FALSE)
    }
  }

  #collect behavioral refinement choices into single vector (easter egg, can set refinement_C to TRUE to get bounds for units on the right side of cutoff, assuming tau0=lambda=0)
  refinements <- list(refinement_A = refinement_A, refinement_B = refinement_B, refinement_C = FALSE)

  #Check vector of percentiles
  if(!is.null(percentiles)){
    lapply(percentiles, function(p){
      if (!(p==-1) & !(p>0 & p<1)) {
        stop("Each value in vector percentiles must be in (0,1) or be equal to -1")
      }
    })
  }

  #Check vector of potential_taus
  if(!is.null(potential_taus)){
    lapply(potential_taus, function(t){
      if (!(t>=0 & t<1)) {
        stop("Each value in vector potential_taus must be in [0,1)")
      }
    })
  }

  #GET SETUP FOR ESTIMATION
  #------------------------------------------------------------------
  #Generate kernel weights for conditional mean regressions
  W<-Kh(h<-bwsx[2],kernel,dist_cut)
  if(gotCovs){Wcov<-Kh(h<-bwsxcov[2],kernel,dist_cut)} else{Wcov<-NULL}

  #Generate matrix of powers of dist_cut for regressions
  polynomial <- generate_polynomial_matrix(dist_cut,orders[2])

  #Prepare vector of y -values for CDFs
  if (is.null(evaluation_ys)) {
    little_ys<-sort(unique(y))
  }
  else{
    little_ys<-evaluation_ys
  }

  #if yextremes are set, make sure they are included in evaluation ys (in particular if yextremes[2] isn't included righteffects CDFs may not make it all the way to one)
  if(!is.null(yextremes) & !is.null(evaluation_ys)){
    if(yextremes[1]<min(evaluation_ys)){
      warning("the lower bound in yextremes is smaller than the smallest y value in little_ys, so is being added to it")
      little_ys = c(yextremes[1],little_ys)
    }
    if(yextremes[2]>max(evaluation_ys)){
      warning("the upper bound in yextremes is larger than the largest y value in little_ys, so is being added to it")
      little_ys = c(little_ys,yextremes[2])
    }
  }

  #make sure little_ys are strictly increasing:
  little_ys<-sort(unique(little_ys))

  #Warnings for user-provided evaluation_ys
  if(discrete_y & !all(y %in% little_ys)){warning("When discrete_y is set to TRUE, a PMF function for y is be estimated by differencing the estimated CDF of y for consecutive values of evaluation_ys. However, the vector evaluation_ys does not contain all values of y detected in the sample, so this PMF will be innacurate. Re-running with evaluation_ys set to NULL recommended.")}
  if(little_ys[1]>min(y)){warning("Smallest value of evaluation_ys is greater than smallest value of y. CDF's will assume there is no mass to the left of this value. Suggested to re-run with evaluation_ys containing a lower bound for y.")}


  #COMPUTE DISCONTINUITY IN DENSITY AT CUTOFF (TAU_HAT)
  #------------------------------------------------------------------
  tau_tilde <- compute_tau_tilde(kernel=kernel, order=orders[1], x=dist_cut, bwsx=bwsx, discrete_x=discrete_x)
  tau_hat <- max(0,tau_tilde)

  print(paste0("The proportion of always-assigned units just to the right of the cutoff is estimated to be ", round(tau_hat, digits=5)))
  if(tau_hat==0){warning("No evidence of manipulation found, thus bounds based on estimated tau will not differ from estimation that assumes no manipulation")}

  #Calculate covariate conditional tau_hats if user has passed covariates
  if(gotCovs){
    tau_tilde_w<-sapply(levels(covs), function(w) compute_tau_tilde(kernel=kernel, order=orders[1], x=dist_cut[covs==w], bwsx=bwsxcov, discrete_x=discrete_x), USE.NAMES=TRUE)
    tau_w<-unlist(sapply(levels(covs), function(w) max(0,tau_tilde_w[w]), USE.NAMES=TRUE))

    sapply(levels(covs), function(w){
      if(tau_tilde_w[[w]]<=0){warning(paste0("No evidence of manipulation found for covariate subsample: ", w))}
    })

  }
  else{
    tau_w = NULL
  }

  #CHECK ALL VALUES OF TAU AGAINST CONSTRAINTS OF T-SET BINDING (FOOTNOTE 15 IN PAPER)
  #------------------------------------------------------------------
  if(num_bootstraps[2]>1){
    check_taus(num_bootstraps=num_bootstraps[2], kernel=kernel, orders=orders, dist_cut=dist_cut, bwsx=bwsx, discrete_x=discrete_x, treatment=treatment, W=W)
    if(!is.null(potential_taus) & num_bootstraps[1]>1){
      lapply(potential_taus, function(tau) check_taus(fixed_tau=tau, num_bootstraps=num_bootstraps[2], kernel=kernel, orders=orders, dist_cut=dist_cut, bwsx=bwsx, discrete_x=discrete_x, treatment=treatment, W=W))
    }
    if(gotCovs){
      lapply(levels(covs), function(w) check_taus(iscov=w, num_bootstraps=num_bootstraps[2], kernel=kernel, orders=orders, dist_cut=dist_cut[covs==w], bwsx=bwsxcov, discrete_x=discrete_x, treatment=treatment[covs==w], W=Wcov[covs==w]))
    }
  }

  #DO ALL PREP WORK (E.G. UPPER AND LOWER BOUND CDF ESTIMATION) NECESSARY TO ESTIMATE TREATMENT EFFECTS
  #------------------------------------------------------------------
  #If CDFinputs$original is set, then the user has passed pre-computed CDFinputs (from a previous run on the same dataset) that they wish to recycle to save time
  if(is.null(CDFinputs$original)){
    allCDFs<-get_allCDFs(tau_hat=tau_hat, y=y, dist_cut=dist_cut, fuzzy = fuzzy, treatment=treatment, polynomial=polynomial, W=W, Wcov=Wcov, num_tau_pairs=num_tau_pairs, little_ys=little_ys, discrete_y=discrete_y, discrete_x=discrete_x, gotCovs=gotCovs, covs=covs, tau_w=tau_w, right_effects=right_effects, yextremes=yextremes, num_lambdas=num_lambdas, bwy=bwy, bwycov=bwycov, kernel_y=kernel_y, yboundaries=yboundaries, orders=orders, bwsx=bwsx, bwsxcov=bwsxcov, kernel=kernel, progressFile=progressFile, isBs=0, refinements=refinements)
    #The underlying CDF's stored in CDFinputs are independent of tau_hat, etc and can be used for further computations on the same sample, e.g. allCDFs_star. So we save them.
    CDFinputs$original<-allCDFs$CDFinputs
  } else{
    allCDFs<-get_allCDFs(CDFinputs=CDFinputs$original, tau_hat=tau_hat, y=y, dist_cut=dist_cut, fuzzy = fuzzy, treatment=treatment, polynomial=polynomial, W=W, Wcov=Wcov, num_tau_pairs=num_tau_pairs, little_ys=little_ys, discrete_y=discrete_y, discrete_x=discrete_x, gotCovs=gotCovs, covs=covs, tau_w=tau_w, right_effects=right_effects, yextremes=yextremes, num_lambdas=num_lambdas, bwy=bwy, bwycov=bwycov, kernel_y=kernel_y, yboundaries=yboundaries, orders=orders, bwsx=bwsx, bwsxcov=bwsxcov, kernel=kernel, progressFile=progressFile, isBs=0, refinements=refinements)
  }
  takeup_increase<-allCDFs$prelims$takeup_increase; tau1s<-allCDFs$prelims$tau1s; tau0s<-allCDFs$prelims$tau0s

  #GET POINT ESTIMATES FOR TREATMENT EFFECTS, BASED ON CDF BOUNDS COMPUTED ABOVE
  #------------------------------------------------------------------
  #Encode ATE estimation as u=-1, so that we can send all treatment effects as one vector (percentiles)
  just_ate<-(type=="ate")
  if(just_ate){percentiles<-c(-1)}

  point_estimates<-get_estimates_by_TE(allCDFs=allCDFs, fuzzy=fuzzy, gotCovs=gotCovs, covs=covs, right_effects=right_effects, yextremes=yextremes, num_lambdas=num_lambdas, percentiles=percentiles, isBs=0, refinements=refinements)

  #GENERATE CONFIDENCE INTERVALS BY NONPARAMETERIC BOOTSTRAP
  #------------------------------------------------------------------
  if(num_bootstraps[1]>1){
    CIs<-CIs_estimate(CDFinputs=CDFinputs, parallelize=parallelize, fuzzy = fuzzy, gotCovs=gotCovs, covs=covs, right_effects=right_effects, yextremes=yextremes, discrete_y=discrete_y, num_tau_pairs=num_tau_pairs, num_lambdas=num_lambdas, percentiles=percentiles, tau_tilde=tau_tilde, y=y, dist_cut=dist_cut, treatment=treatment, polynomial=polynomial, W=W, Wcov=Wcov, num_bootstraps=num_bootstraps, alpha=alpha, Kn=Kn, discrete_x=discrete_x, orders=orders, bwsx=bwsx, bwsxcov=bwsxcov, kernel=kernel, little_ys=little_ys, bwy=bwy, bwycov=bwycov, kernel_y=kernel_y, yboundaries=yboundaries, potential_taus=potential_taus, takeup_increase=takeup_increase, progressFile=progressFile, warningsFile=warningsFile, refinements=refinements)
    bs_estimates<-CIs$estimates
    CDFinputs=CIs$CDFinputs #Update CDFinputs to include those computed on the bootstrap samples
  } else{
    CIs_by_TE <-sapply(
      1:length(percentiles),
      function(i)
      {
        list(tau_hat_CI=c(NA, NA), takeup_increase_CI=c(NA, NA), TE_SRD_naive_CI=c(NA, NA), TE_SRD_CI=c(NA, NA), TE_SRD_covs_CI=c(NA, NA), TE_FRD_naive_CI=c(NA, NA), TE_FRD_CI=c(NA, NA), TE_FRD_refinementA_CI=c(NA, NA), TE_FRD_refinementB_CI=c(NA, NA), TE_SRD_CIs_manipulation=c(NA, NA), TE_FRD_CIs_manipulation=c(NA, NA), TE_FRD_covs_CI=c(NA, NA), TE_SRD_right_CI=c(NA, NA), TE_FRD_right_CI=c(NA, NA))#, TE_FRD_right_CI_refinementC=c(NA, NA))
      }, USE.NAMES=TRUE)
    CIs<-list(tau_hat_CI=c(NA,NA), takeup_increase_CI=c(NA,NA), CIs_by_TE=CIs_by_TE)
    bs_estimates<-NA
  }

  #ORGANIZE OUTPUT (This produces a vector of lists, each list containing the complete set of point estimates for a given type of treatment effect (ATE or a QTE))
  #------------------------------------------------------------------
  output_by_TE <-sapply(
    1:length(percentiles),
    function(u)
    {list( tau_hat=tau_hat,tau_hat_CI=CIs$tau_hat_CI,takeup_increase=takeup_increase,takeup_increase_CI=CIs$takeup_increase_CI,
           TE_SRD_naive=point_estimates[['estimates_naive',u]]$TE_SRD_naive,TE_SRD_naive_CI=CIs$CIs_by_TE[['TE_SRD_naive_CI',u]],
           TE_SRD_bounds = c(point_estimates[['estimates_main',u]]$TE_lower_SRD, point_estimates[['estimates_main',u]]$TE_upper_SRD),TE_SRD_CI=CIs$CIs_by_TE[['TE_SRD_CI',u]],
           TE_SRD_covs_bounds=point_estimates[['estimates_covs',u]]$TE_SRD_covs_bounds,TE_SRD_covs_CI=CIs$CIs_by_TE[['TE_SRD_covs_CI',u]],
           TE_FRD_naive=point_estimates[['estimates_naive',u]]$TE_FRD_naive,TE_FRD_naive_CI=CIs$CIs_by_TE[['TE_FRD_naive_CI',u]],
           TE_FRD_bounds= c(point_estimates[['estimates_main',u]]$TE_lower_FRD, point_estimates[['estimates_main',u]]$TE_upper_FRD),TE_FRD_CI=CIs$CIs_by_TE[['TE_FRD_CI',u]],
           TE_FRD_bounds_refinementA = c(point_estimates[['estimates_main',u]]$TE_lower_FRD_refinementA, point_estimates[['estimates_main',u]]$TE_upper_FRD_refinementA),TE_FRD_refinementA_CI=CIs$CIs_by_TE[['TE_FRD_refinementA_CI',u]],
           TE_FRD_bounds_refinementB = c(point_estimates[['estimates_main',u]]$TE_lower_FRD_refinementB, point_estimates[['estimates_main',u]]$TE_upper_FRD_refinementB),TE_FRD_refinementB_CI=CIs$CIs_by_TE[['TE_FRD_refinementB_CI',u]],
           TE_FRD_covs_bounds=point_estimates[['estimates_covs',u]]$TE_FRD_covs_bounds,TE_FRD_covs_CI=CIs$CIs_by_TE[['TE_FRD_covs_CI',u]],
           TE_SRD_CIs_manipulation=CIs$CIs_by_TE[['TE_SRD_CIs_manipulation',u]],TE_FRD_CIs_manipulation=CIs$CIs_by_TE[['TE_FRD_CIs_manipulation',u]],
           TE_SRD_right_bounds=point_estimates[['estimates_rightside',u]]$TE_SRD_right_bounds,TE_SRD_right_CI=CIs$CIs_by_TE[['TE_SRD_right_CI',u]],
           TE_FRD_right_bounds=point_estimates[['estimates_rightside',u]]$TE_FRD_right_bounds,TE_FRD_right_CI=CIs$CIs_by_TE[['TE_FRD_right_CI',u]]
           #TE_FRD_right_bounds_refinementC=point_estimates[['estimates_rightside',u]]$TE_FRD_right_bounds_refinementC,TE_FRD_right_CI_refinementC=CIs$CIs_by_TE[['TE_FRD_right_CI_refinementC',u]]
    )}, USE.NAMES=TRUE)

  #FINISH UP
  #------------------------------------------------------------------
  end_time <- Sys.time()
  time_taken<-difftime(end_time,start_time,units="mins")
  print_message(paste0("Time taken:", round(time_taken, digits=2), " minutes"), progressFile=progressFile)

  return(list(CDFinputs=CDFinputs, time_taken=time_taken, TEs=percentiles, sample_size=length(dist_cut), estimates=output_by_TE, estimates_raw=point_estimates, allCDFs_estimates=allCDFs, bs_estimates=bs_estimates, potential_taus=potential_taus))
}

#------------------------------------------------------------------
#------------------------------------------------------------------
#INTERMEDIATE FUNCTIONS
#------------------------------------------------------------------
#------------------------------------------------------------------

#ESTIMATE NAIVE TREATMENT EFFECTS
#------------------------------------------------------------------
naive_estimate <- function(type="ate", CDFs, u=NULL){

  #No need to pass fuzzy boolean to this function, as the below computes without error Y0_FRD=NA from CDFs$naive_Fs$F_Y0_FRD, for example

  #Treatment effects
  if(type=="ate"){
    Y0_SRD<-cdf2expectation(little_ys=CDFs$naive_Fs$little_y, cdfy=CDFs$naive_Fs$F_Y0_SRD)
    Y1_SRD<-cdf2expectation(little_ys=CDFs$naive_Fs$little_y, cdfy=CDFs$naive_Fs$F_Y1_SRD)
    Y0_FRD<-cdf2expectation(little_ys=CDFs$naive_Fs$little_y, cdfy=CDFs$naive_Fs$F_Y0_FRD)
    Y1_FRD<-cdf2expectation(little_ys=CDFs$naive_Fs$little_y, cdfy=CDFs$naive_Fs$F_Y1_FRD)
  }
  if(type=="qte"){
    Y1_SRD<-min(CDFs$naive_Fs$little_y[CDFs$naive_Fs$F_Y1_SRD>=u])
    Y0_SRD<-min(CDFs$naive_Fs$little_y[CDFs$naive_Fs$F_Y0_SRD>=u])
    Y1_FRD<-min(CDFs$naive_Fs$little_y[CDFs$naive_Fs$F_Y1_FRD>=u])
    Y0_FRD<-min(CDFs$naive_Fs$little_y[CDFs$naive_Fs$F_Y0_FRD>=u])
  }

  TE_SRD_naive<-Y1_SRD-Y0_SRD
  TE_FRD_naive<-Y1_FRD-Y0_FRD

  return(list(TE_SRD_naive=TE_SRD_naive, TE_FRD_naive=TE_FRD_naive, TE_outcomes=list(Y1_SRD=Y1_SRD, Y0_SRD=Y0_SRD, Y1_FRD=Y1_FRD, Y0_FRD=Y0_FRD)))
}

#ESTIMATION OF TREATMENT EFFECT BOUNDS FOR GIVEN LEVEL OF MANIPULATION AT THE CUTOFF (TAU_HAT)
#------------------------------------------------------------------
#' @export
bounds_estimate <- function(fuzzy=TRUE, type="ate", tau_hat=NULL, num_tau_pairs, tau1s, tau0s, CDFs, u=NULL, isCovEstimate=FALSE, isRightEstimate=FALSE, warningfile=NULL, isBs=1, refinements=NULL){

  if(is.null(refinements)){
    refinements <- list(refinement_A = FALSE, refinement_B=FALSE, refinement_C=FALSE)
  }

  #COMPUTE BOUNDS FOR SHARP DESIGN
  #------------------------------------------------------------------
  if(type=="ate"){
    Y0_lower<-cdf2expectation(little_ys=CDFs$F_Y0_SRD$little_y, cdfy=CDFs$F_Y0_SRD$lower)
    Y0_upper<-cdf2expectation(little_ys=CDFs$F_Y0_SRD$little_y, cdfy=CDFs$F_Y0_SRD$upper)
    Y1_lower<-cdf2expectation(little_ys=CDFs$F_Y1_SRD$little_y, cdfy=CDFs$F_Y1_SRD$lower)
    Y1_upper<-cdf2expectation(little_ys=CDFs$F_Y1_SRD$little_y, cdfy=CDFs$F_Y1_SRD$upper)
  }
  if(type=="qte"){
    Y0_lower<-min(CDFs$F_Y0_SRD$little_y[CDFs$F_Y0_SRD$lower>=u])
    Y0_upper<-min(CDFs$F_Y0_SRD$little_y[CDFs$F_Y0_SRD$upper>=u])
    Y1_lower<-min(CDFs$F_Y1_SRD$little_y[CDFs$F_Y1_SRD$lower>=u])
    Y1_upper<-min(CDFs$F_Y1_SRD$little_y[CDFs$F_Y1_SRD$upper>=u])
  }
  TE_upper_SRD<-Y1_upper-Y0_lower
  TE_lower_SRD<-Y1_lower-Y0_upper

  #COMPUTE BOUNDS FOR FUZZY DESIGN
  #------------------------------------------------------------------

  if(fuzzy){
    #Calculate number of elements in set that fuzzy will take inf and sup over (this will be equal to num_tau_pairs normally, unless isCovEstimate or isRightEstimate is TRUE)
    num_inTset=length(tau1s)

    if(type=="ate"){
      Y0_tau1s_lower<-unlist(lapply(1:num_inTset, function(t) cdf2expectation(little_ys=CDFs$F_Y0_FRD_t[[t]]$little_y, cdfy=CDFs$F_Y0_FRD_t[[t]]$lower)))
      Y0_tau1s_upper<-unlist(lapply(1:num_inTset, function(t) cdf2expectation(little_ys=CDFs$F_Y0_FRD_t[[t]]$little_y, cdfy=CDFs$F_Y0_FRD_t[[t]]$upper)))
      Y1_tau1s_lower<-unlist(lapply(1:num_inTset, function(t) cdf2expectation(little_ys=CDFs$F_Y1_FRD_t[[t]]$little_y, cdfy=CDFs$F_Y1_FRD_t[[t]]$lower)))
      Y1_tau1s_upper<-unlist(lapply(1:num_inTset, function(t) cdf2expectation(little_ys=CDFs$F_Y1_FRD_t[[t]]$little_y, cdfy=CDFs$F_Y1_FRD_t[[t]]$upper)))
    }
    if(type=="qte"){
      Y0_tau1s_lower<-unlist(lapply(1:num_inTset, function(t) min(CDFs$F_Y0_FRD_t[[t]]$little_y[CDFs$F_Y0_FRD_t[[t]]$lower>=u])))
      Y0_tau1s_upper<-unlist(lapply(1:num_inTset, function(t) min(CDFs$F_Y0_FRD_t[[t]]$little_y[CDFs$F_Y0_FRD_t[[t]]$upper>=u])))
      Y1_tau1s_lower<-unlist(lapply(1:num_inTset, function(t) min(CDFs$F_Y1_FRD_t[[t]]$little_y[CDFs$F_Y1_FRD_t[[t]]$lower>=u])))
      Y1_tau1s_upper<-unlist(lapply(1:num_inTset, function(t) min(CDFs$F_Y1_FRD_t[[t]]$little_y[CDFs$F_Y1_FRD_t[[t]]$upper>=u])))
    }

    #Compute treatment effects as a function of tau1 (or more generally, "t")
    TE_tau1s_upper<-Y1_tau1s_upper-Y0_tau1s_lower
    TE_tau1s_lower<-Y1_tau1s_lower-Y0_tau1s_upper

    #Take max and min over tau1 for bounds (theoretically there could be ties in which case we take the first)
    upperBoundIndex<-which.max(TE_tau1s_upper)
    lowerBoundIndex<-which.min(TE_tau1s_lower)

    #If treatment effects for different values t are all NA, assign upper and/or lower bound index to 1 so that we can return an NA in TE_outcomes below
    if(all(is.na(TE_tau1s_upper))){upperBoundIndex=1}
    if(all(is.na(TE_tau1s_lower))){lowerBoundIndex=1}

    TE_upper_FRD<-TE_tau1s_upper[upperBoundIndex]
    TE_lower_FRD<-TE_tau1s_lower[lowerBoundIndex]
  }
  else{
    Y0_tau1s_lower<-NA; Y0_tau1s_upper<-NA; Y1_tau1s_lower<-NA; Y1_tau1s_upper<-NA; TE_tau1s_upper<-NA; TE_tau1s_lower<-NA; TE_lower_FRD<-NA; TE_upper_FRD<-NA; upperBoundIndex<-NA; lowerBoundIndex<-NA
  }

  #REFINEMENTS OF THE FUZZY RD BOUNDS
  #------------------------------------------------------------------
  #Refinement A: assume that always-assigned units are weakly more likely to receive treatment, conditional on being eligible
  if(fuzzy & !isCovEstimate & !isRightEstimate & refinements$refinement_A & !all(is.na(TE_tau1s_upper)) & !all(is.na(TE_tau1s_lower))){
    refinementA<-(tau1s >= tau_hat)
    upperBoundIndex_refinementA<-which.max(TE_tau1s_upper[refinementA])
    lowerBoundIndex_refinementA<-which.min(TE_tau1s_lower[refinementA])
    TE_upper_FRD_refinementA<-(TE_tau1s_upper[refinementA])[upperBoundIndex_refinementA]
    TE_lower_FRD_refinementA<-(TE_tau1s_lower[refinementA])[lowerBoundIndex_refinementA]
    Y1_upper_refinementA<-(Y1_tau1s_upper[refinementA])[upperBoundIndex_refinementA]; Y0_lower_refinementA<-(Y0_tau1s_lower[refinementA])[upperBoundIndex_refinementA]
    Y1_lower_refinementA<-(Y1_tau1s_lower[refinementA])[lowerBoundIndex_refinementA]; Y0_upper_refinementA<-(Y0_tau1s_upper[refinementA])[lowerBoundIndex_refinementA]
  } else{
    refinementA<-NA; TE_upper_FRD_refinementA<-NA; TE_lower_FRD_refinementA<-NA; Y0_lower_refinementA<-NA;Y0_upper_refinementA<-NA;Y1_lower_refinementA<-NA;Y1_upper_refinementA<-NA
  }

  #Refinement B: assume that always-assigned units just to the right of cutoff are certain to receive treatment
  if(fuzzy & !isCovEstimate & !isRightEstimate & refinements$refinement_B & !all(is.na(TE_tau1s_upper)) & !all(is.na(TE_tau1s_lower))){
    tau0U<-tau_hat-(1-tau_hat)*(CDFs$prelims$treatRight-CDFs$prelims$treatLeft)/(1-CDFs$prelims$treatRight)
    if(tau0s[num_tau_pairs]>0 & tau0U>0){
      TE_upper_FRD_refinementB<-NA; TE_lower_FRD_refinementB<-NA; Y0_lower_refinementB<-NA;Y0_upper_refinementB<-NA;Y1_lower_refinementB<-NA;Y1_upper_refinementB<-NA
    } else{
      TE_upper_FRD_refinementB<-TE_tau1s_upper[num_tau_pairs]
      TE_lower_FRD_refinementB<-TE_tau1s_lower[num_tau_pairs]
      Y1_upper_refinementB<-Y1_tau1s_upper[num_tau_pairs]; Y0_lower_refinementB<-Y0_tau1s_lower[num_tau_pairs]
      Y1_lower_refinementB<-Y1_tau1s_lower[num_tau_pairs]; Y0_upper_refinementB<-Y0_tau1s_upper[num_tau_pairs]
    }
  }
  else{
    TE_upper_FRD_refinementB<-NA; TE_lower_FRD_refinementB<-NA; Y0_lower_refinementB<-NA;Y0_upper_refinementB<-NA;Y1_lower_refinementB<-NA;Y1_upper_refinementB<-NA
  }

  #Refinement C: assume that always-assigned units just to the right of cutoff are certain to receive treatment (tau0=0), but they would not receive treatment were they not assigned (lambda=0)
  if(fuzzy & isRightEstimate & refinements$refinement_C & !all(is.na(TE_tau1s_upper)) & !all(is.na(TE_tau1s_lower))){
    if(tau0s[[num_tau_pairs]]$tau0>0.00000001 | tau0s[[num_tau_pairs]]$lambda>0){
      if(isBs==0){warning("The data are inconsistent with tau0=0 & lambda=0 so refinement C will not be computed")};
      TE_upper_FRD_refinementC<-NA; TE_lower_FRD_refinementC<-NA; Y0_lower_refinementC<-NA;Y0_upper_refinementC<-NA;Y1_lower_refinementC<-NA;Y1_upper_refinementC<-NA
    } else{
      #Since "tau1s" is ordered by increasing lambda, and increasing tau1 (decreasing tau0) within each value of lambda, the [[num_tau_pairs]] element of the gammaplus lists indexed by t contains the values for tau0=0, lambda=0, if tau0=0 is a possibility
      TE_upper_FRD_refinementC<-TE_tau1s_upper[[num_tau_pairs]]
      TE_lower_FRD_refinementC<-TE_tau1s_lower[[num_tau_pairs]]
      Y1_upper_refinementC<-Y1_tau1s_upper[[num_tau_pairs]]; Y0_lower_refinementC<-Y0_tau1s_lower[[num_tau_pairs]]
      Y1_lower_refinementC<-Y1_tau1s_lower[[num_tau_pairs]]; Y0_upper_refinementC<-Y0_tau1s_upper[[num_tau_pairs]]
    }
    TE_components<-cbind(tau1s, tau0s, Y0_tau1s_lower, Y0_tau1s_upper, Y1_tau1s_lower, Y1_tau1s_upper, TE_tau1s_lower,TE_tau1s_upper, refinementC=((1:num_inTset)==num_tau_pairs))
  }
  else{
    TE_upper_FRD_refinementC<-NA; TE_lower_FRD_refinementC<-NA; Y0_lower_refinementC<-NA;Y0_upper_refinementC<-NA;Y1_lower_refinementC<-NA;Y1_upper_refinementC<-NA
    TE_components<-cbind(tau1s, tau0s, Y0_tau1s_lower, Y0_tau1s_upper, Y1_tau1s_lower, Y1_tau1s_upper, TE_tau1s_lower,TE_tau1s_upper, refinementA,refinementB=((1:num_tau_pairs)==num_tau_pairs))
  }

  #Store Y0 and Y1 values for bounds in case user is interested in them
  TE_outcomes<-list(sharp=list(Y1_lower=Y1_lower,Y1_upper=Y1_upper,Y0_lower=Y0_lower,Y0_upper=Y0_upper), fuzzy=list(Y1_lower=Y1_tau1s_lower[lowerBoundIndex],Y0_upper=Y0_tau1s_upper[lowerBoundIndex],Y1_upper=Y1_tau1s_upper[upperBoundIndex],Y0_lower=Y0_tau1s_lower[upperBoundIndex]), refinementA=list(Y0_lower=Y0_lower_refinementA, Y0_upper=Y0_upper_refinementA, Y1_lower=Y1_lower_refinementA,Y1_upper=Y1_upper_refinementA), refinementB=list(Y0_lower=Y0_lower_refinementB, Y0_upper=Y0_upper_refinementB, Y1_lower=Y1_lower_refinementB,Y1_upper=Y1_upper_refinementB), refinementC=list(Y0_lower=Y0_lower_refinementC, Y0_upper=Y0_upper_refinementC, Y1_lower=Y1_lower_refinementC,Y1_upper=Y1_upper_refinementC))

  return(list(TE_lower_SRD=TE_lower_SRD, TE_upper_SRD=TE_upper_SRD, TE_lower_FRD=TE_lower_FRD, TE_upper_FRD=TE_upper_FRD, TE_lower_FRD_refinementA=TE_lower_FRD_refinementA, TE_upper_FRD_refinementA=TE_upper_FRD_refinementA, TE_lower_FRD_refinementB=TE_lower_FRD_refinementB, TE_lower_FRD_refinementC=TE_lower_FRD_refinementC, TE_upper_FRD_refinementC=TE_upper_FRD_refinementC, TE_upper_FRD_refinementB=TE_upper_FRD_refinementB, TE_outcomes=TE_outcomes, TE_components = TE_components))
  #return(list(TE_lower_SRD=TE_lower_SRD, TE_upper_SRD=TE_upper_SRD, TE_lower_FRD=TE_lower_FRD, TE_upper_FRD=TE_upper_FRD, , TE_outcomes=TE_outcomes, TE_components = TE_components))
}

#LOOP OVER THE VARIOUS TREATMENT EFFECTS BEING COMPUTED AND GATHER ESTIMATES
#------------------------------------------------------------------
get_estimates_by_TE <- function(allCDFs, fuzzy, gotCovs, covs, right_effects, yextremes, num_lambdas, percentiles, warningfile=NULL, isBs=1, refinements){

  tau_hat<-allCDFs$prelims$tau_hat
  tau1s<-allCDFs$prelims$tau1s
  tau0s<-allCDFs$prelims$tau0s
  num_tau_pairs<-length(tau1s)

  results_by_TE <-sapply(
    percentiles,
    function(u)
    {

      if(u==-1){type="ate"} else{type="qte"}

      #ESTIMATE NAIVE AND MANIPULATION-ROBUST BOUNDS ON TREATMENT EFFECTS
      #------------------------------------------------------------------
      estimates_naive<-naive_estimate(type=type, CDFs=allCDFs$CDFs, u=u)
      estimates_main<-bounds_estimate(fuzzy=fuzzy, type=type, tau_hat=tau_hat, num_tau_pairs=num_tau_pairs, tau1s=tau1s, tau0s=tau0s, CDFs=allCDFs$CDFs, u=u, warningfile=warningfile, isBs=isBs, refinements=refinements)

      #OPTIONAL ESTIMATION 1: ESTIMATES FOR REFINEMENT BASED ON COVARIATES
      #------------------------------------------------------------------
      estimates_covs<-list(TE_SRD_covs_bounds=c(NA,NA), TE_FRD_covs_bounds=c(NA,NA), TE_components_covariates=NA, TE_outcomes_covariates=list(sharp=list(Y1_lower=NA, Y1_upper=NA, Y0_lower=NA, Y0_upper=NA), fuzzy=list(Y1_lower=NA, Y1_upper=NA, Y0_lower=NA, Y0_upper=NA)))
      if(gotCovs){
        cov_estimates<-bounds_estimate(fuzzy=fuzzy, type=type, num_tau_pairs=num_tau_pairs, tau1s=allCDFs$covCDFs$tau1s, tau0s=allCDFs$covCDFs$tau0s, CDFs=allCDFs$covCDFs$CDFs, u=u, isCovEstimate=TRUE, isRightEstimate=FALSE, warningfile=warningfile)
        estimates_covs<-list(TE_SRD_covs_bounds=c(cov_estimates$TE_lower_SRD,cov_estimates$TE_upper_SRD), TE_FRD_covs_bounds=c(cov_estimates$TE_lower_FRD,cov_estimates$TE_upper_FRD), TE_components_covariates=cov_estimates$TE_components, TE_outcomes_covariates=cov_estimates$TE_outcomes)
      }

      #OPTIONAL ESTIMATION 2: ESTIMATES FOR UNITS ON THE RIGHT OF THE CUTOFF
      #------------------------------------------------------------------
      estimates_rightside = list(TE_SRD_right_bounds=c(NA,NA), TE_FRD_right_bounds=c(NA,NA),TE_FRD_right_bounds_refinementC=c(NA,NA), TE_components_rightside=NA, TE_outcomes_rightside=list(sharp=list(Y1_lower=NA,Y1_upper=NA,Y0_lower=NA,Y0_upper=NA), fuzzy=list(Y1_lower=NA,Y0_upper=NA,Y1_upper=NA,Y0_lower=NA), refinementA=NA, refinementB=NA, refinementC=list(Y0_lower=NA, Y0_upper=NA, Y1_lower=NA,Y1_upper=NA)))
      if(right_effects){
        right_estimates<-bounds_estimate(fuzzy=fuzzy, type=type, num_tau_pairs=num_tau_pairs, tau1s=allCDFs$rightCDFs$tau1s, tau0s=allCDFs$rightCDFs$tau0s, CDFs=allCDFs$rightCDFs$CDFs, u=u, isCovEstimate=FALSE, isRightEstimate=TRUE, warningfile=warningfile)

        if(fuzzy){
          #Reformat TE_components to be readable
          TE_components_rightside<-data.frame(right_estimates$TE_components)

          lambdas<-unlist(lapply(1:length(TE_components_rightside$tau0s), function(t) TE_components_rightside$tau0s[[t]]$lambda))
          TE_components_rightside$tau1s<-unlist(lapply(1:length(TE_components_rightside$tau0s), function(t) TE_components_rightside$tau1s[[t]]$tau1))
          TE_components_rightside$tau0s<-unlist(lapply(1:length(TE_components_rightside$tau0s), function(t) TE_components_rightside$tau0s[[t]]$tau0))
          TE_components_rightside<-as.data.frame(cbind(lambdas, TE_components_rightside))
        }
        else{
          TE_components_rightside<-NA
        }

        estimates_rightside = list(TE_SRD_right_bounds = c(right_estimates$TE_lower_SRD,right_estimates$TE_upper_SRD), TE_FRD_right_bounds = c(right_estimates$TE_lower_FRD,right_estimates$TE_upper_FRD),TE_FRD_right_bounds_refinementC=c(right_estimates$TE_lower_FRD_refinementC,right_estimates$TE_upper_FRD_refinementC), TE_components_rightside=TE_components_rightside, TE_outcomes_rightside=right_estimates$TE_outcomes)
      }

      list(prelims=allCDFs$prelims, estimates_naive=estimates_naive, estimates_main=estimates_main, estimates_covs=estimates_covs, estimates_rightside=estimates_rightside)

    }, USE.NAMES=TRUE)

  return(results_by_TE)
}

#COMPUTATATION OF NECESSARY INPUTS TO CDFS AS A FUNCTION OF Y (THESE CAN BE RECYCLED FOR DIFFERENT VALUES OF TAU ON THE SAME SAMPLE)
#------------------------------------------------------------------
compute_CDFinputs <- function(y, dist_cut, fuzzy, treatment, polynomial, W, little_ys, discrete_y, bwy, kernel_y, yboundaries){

  #Estimate CDFs by local polynomial regression
  inputs<-as.data.frame(rbindlist(lapply(
    little_ys,
    function(little_y)
    {
      below_y<-(y<=little_y)
      F_right<-as.numeric(lm(below_y~polynomial, weights=W, subset=(dist_cut>=0))$coefficients["(Intercept)"])
      F_left<-as.numeric(lm(below_y~polynomial, weights=W, subset=(dist_cut<0))$coefficients["(Intercept)"])

      if(fuzzy){
        F_right_treated<-as.numeric(lm(below_y~polynomial, weights=W, subset=((dist_cut>=0)&(treatment==1)))$coefficients["(Intercept)"])
        F_left_untreated<-as.numeric(lm(below_y~polynomial, weights=W, subset=((dist_cut<0)&(treatment==0)))$coefficients["(Intercept)"])
      }
      else{
        F_right_treated<-NA
        F_left_untreated<-NA
      }

      if(fuzzy & sum((dist_cut>=0)&(treatment==0)&(W>0))>0){
        F_right_untreated<-as.numeric(lm(below_y~polynomial, weights=W, subset=((dist_cut>=0)&(treatment==0)))$coefficients["(Intercept)"])
      } else{F_right_untreated<-NA}

      if(fuzzy & sum((dist_cut<0)&(treatment==1)&(W>0))>0){
        F_left_treated<-as.numeric(lm(below_y~polynomial, weights=W, subset=((dist_cut<0)&(treatment==1)))$coefficients["(Intercept)"])
      } else{F_left_treated<-NA}

      list(little_y=little_y,F_right=F_right,F_left=F_left,F_right_treated=F_right_treated,F_right_untreated=F_right_untreated,F_left_treated=F_left_treated,F_left_untreated=F_left_untreated)
    }
  )))

  #Censor CDFs to unit interval
  inputs$F_right<-pmin(pmax(inputs$F_right,0),1)
  inputs$F_left<-pmin(pmax(inputs$F_left,0),1)
  if(fuzzy){
    inputs$F_right_treated<-pmin(pmax(inputs$F_right_treated,0),1)
    inputs$F_left_treated<-pmin(pmax(inputs$F_left_treated,0),1)
    inputs$F_right_untreated<-pmin(pmax(inputs$F_right_untreated,0),1)
    inputs$F_left_untreated<-pmin(pmax(inputs$F_left_untreated,0),1)
  }

  #Normalize CDFs
  inputs$F_right<-inputs$F_right/max(inputs$F_right)
  inputs$F_left<-inputs$F_left/max(inputs$F_left)
  if(fuzzy){
    inputs$F_right_treated<-inputs$F_right_treated/max(inputs$F_right_treated)
    inputs$F_left_treated<-inputs$F_left_treated/max(inputs$F_left_treated)
    inputs$F_right_untreated<-inputs$F_right_untreated/max(inputs$F_right_untreated)
    inputs$F_left_untreated<-inputs$F_left_untreated/max(inputs$F_left_untreated)
  }

  #Monotonize all columns of inputs
  inputs<-data.frame(apply(inputs,2,sort,na.last=TRUE))

  #Separately estimate untreated densities on either side of the cutoff, unless discrete_y=TRUE (in which case PMF's will be estimated by differencing CDFs)
  if(!discrete_y & fuzzy){
    densities<-as.data.frame(rbindlist(lapply(
      little_ys,
      function(little_y)
      {
        if(!is.null(yboundaries$min)){
          kernel_left<-1/2-integrate_kernel(kernel=kernel_y, x=((yboundaries$min-little_y)/bwy))
        } else{
          kernel_left<-1/2
        }
        if(!is.null(yboundaries$max)){
          kernel_right<-integrate_kernel(kernel=kernel_y, x=((yboundaries$max-little_y)/bwy))-1/2
        } else{
          kernel_right<-1/2
        }
        kernel_coverage<-kernel_left+kernel_right
        Khy<-Kh(h=bwy, kernel=kernel_y, x=(y-little_y))/kernel_coverage

        dens_left_untreated<-as.numeric(lm(Khy~polynomial, weights=W, subset=((dist_cut<0)&(treatment==0)))$coefficients["(Intercept)"])

        if(fuzzy & sum((dist_cut>=0)&(treatment==0)&(W>0))>0){
          dens_right_untreated<-as.numeric(lm(Khy~polynomial, weights=W, subset=((dist_cut>=0)&(treatment==0)))$coefficients["(Intercept)"])
        } else{dens_right_untreated<-NA}

        list(little_y=little_y,dens_left_untreated=dens_left_untreated, dens_right_untreated=dens_right_untreated, kernel_coverage=kernel_coverage)
      }
    )))

    inputs$d_left_untreated<-densities$dens_left_untreated
    inputs$d_right_untreated<-densities$dens_right_untreated

    #censor PDF estimates at zero
    inputs$d_left_untreated<-pmax(inputs$d_left_untreated,0)
    inputs$d_right_untreated<-pmax(inputs$d_right_untreated,0)


    #normalize PDF estimates
    inputs$d_left_untreated<-inputs$d_left_untreated/pdf2cdf(little_ys=little_ys, pdfy = inputs$d_left_untreated, integral_only=TRUE)
    inputs$d_right_untreated<-inputs$d_right_untreated/pdf2cdf(little_ys=little_ys, pdfy = inputs$d_right_untreated, integral_only=TRUE)
  }

  return(inputs)
}

#COMPUTE CDFS
#------------------------------------------------------------------
estimateCDFs <- function(inputs=NULL, tau_hat, treatLeft, treatRight, y, dist_cut, fuzzy, treatment, polynomial, W, num_tau_pairs, tau1s, tau0s, little_ys, discrete_y, bwy, kernel_y, yboundaries, isBs=0, isTau="estimated", isCov=NULL, warningfile=NULL, refinements=NULL){

  if(is.null(isTau)){isTau<-"estimated"}
  if(is.null(isCov)){isCov<-""}

  if(is.null(inputs)){inputs<-compute_CDFinputs(y=y, dist_cut=dist_cut, fuzzy = fuzzy, treatment=treatment, polynomial=polynomial, W=W, little_ys=little_ys, discrete_y=discrete_y, bwy=bwy, kernel_y=kernel_y, yboundaries=yboundaries)}

  #Get CDF bounds for SRD
  #----------------------------------------------------------------
  F_Y0_SRD<-data.frame(little_y=little_ys,upper=inputs$F_left, lower=inputs$F_left)
  F_Y1_SRD<-data.frame(little_y=little_ys,upper=trimLeft(cdf=inputs$F_right,toTrim=tau_hat),lower=trimRight(cdf=inputs$F_right,toTrim=tau_hat))

  #Get naive extremes of identified set for tau0, tau1
  #------------------------------------------------------------------
  #Limits of treatment probability at cutoff
  treatLeft<-as.numeric(lm(treatment~polynomial, weights=W, subset=(dist_cut<0))$coefficients["(Intercept)"])
  treatRight<-as.numeric(lm(treatment~polynomial, weights=W, subset=(dist_cut>=0))$coefficients["(Intercept)"])
  if(fuzzy){
    takeup_increase<-treatRight-treatLeft
  } else{
    takeup_increase<-NA
  }

  #Get CDF bounds for FRD
  #----------------------------------------------------------------
  if(fuzzy){

    #First calculate T set boundary without the potential role of s(y)
    tau1L<-max(0, 1-(1-tau_hat)/treatRight)
    tau1U<-min(1-treatLeft/treatRight*(1-tau_hat),tau_hat/treatRight)
    tau0L<-min(1,tau_hat/(1-treatRight))
    tau0U<-max(0,tau_hat-(1-tau_hat)*(treatRight-treatLeft)/(1-treatRight))

    #Bounds for Y0
    if(sum((dist_cut>=0)&(treatment==0)&(W>0))==0){
      #If there are no untreated units on the right, then kappa0 = 0. There are no never-takers and the distribution of Y0 is point identified by untreated units on left
      F_Y0_FRD_t<-lapply(1:length(tau0s), function(t) list(little_y=little_ys, upper=inputs$F_left_untreated, lower=inputs$F_left_untreated))
    } else{
      if(discrete_y){
        #Set first value of pmf to be same as first value of cdf
        dens_left_untreated<-c(inputs$F_left_untreated[1], diff(inputs$F_left_untreated))
        dens_right_untreated<-c(inputs$F_right_untreated[1], diff(inputs$F_right_untreated))
      } else{
        dens_left_untreated<-inputs$d_left_untreated
        dens_right_untreated<-inputs$d_right_untreated
      }

      kappa0<-1/(1-tau_hat)*(1-treatRight)/(1-treatLeft)
      s_notau<-pmin(dens_left_untreated/kappa0, dens_right_untreated)
      inputs$s_notau<-s_notau
      if(!discrete_y){s_notau_mass<-pdf2cdf(little_ys=little_ys, pdfy = s_notau, integral_only=TRUE)} else{s_notau_mass<-pmf2cdf(pmfy=s_notau, integral_only=TRUE)}

      #Warnings/checks on s_notau
      if(isBs==0 & (s_notau_mass<1-tau0L) & tau_hat>0){
        stop(paste0("Error for tau=", isTau, isCov, ": the estimated identified set for tau0 is null. This occurs when s(y)*(1-tau0) (see paper for definition) integrates to less than 1-tau0L. In this case it integrates to ", round(s_notau_mass, digits=3), " and tau0L is ", round(tau0L, digits=3), ". If this is a covariate subsample or tau != estimated, you will need to re-run without covs or the potential_taus option, respectively. If bootstrap=-1 (meaning the starred estimands in notation of the paper--see section on inference), you may wish to rerun."))
      }

      if(isBs==-1 & (s_notau_mass<1-tau0L) & tau_hat>0){
        stop(paste0("Error for tau=", isTau, isCov, ", for starred estimands in notation of the paper (see the use of bootstrap to nudge tau away from zero in section on inference): the estimated identified set for tau0 is null. This occurs when s(y)*(1-tau0) (see paper for definition) integrates to less than 1-tau0L. In this case it integrates to ", round(s_notau_mass, digits=3), " and tau0L is ", round(tau0L, digits=3), ". If this is a covariate subsample or tau != estimated, you will need to re-run without covs or the potential_taus option, respectively. Otherwise, you may wish to re-run to generate new bootstrap resamples."))
      }

      if(isBs > 0 & (s_notau_mass<1-tau0L) & tau_hat>0){
        warning_bs(paste0("Warning from bootstrap: ", isBs, ", tau=", isTau, isCov, ": the estimated identified set for tau0 is null. This occurs when s(y)*(1-tau0) (see paper for definition) integrates to less than 1-tau0L. This bootstrap sample will be ignored for fuzzy bounds."), warningfile=warningfile)
        #Note that this bootstrap sample should be dropped for fuzzy estimand confidence intervals
        keepfuzzy<-FALSE
      }
      else{
        keepfuzzy<-TRUE
      }

      if(isBs==0 & (s_notau_mass<.99*(1-tau0U)) & tau_hat>0){warning(paste0("For tau=", isTau, isCov, ": the integral of stilde(y)=s(y)*(1-tau0) is small enough to shrink the identified set of (tau0, tau1) pairs (see paper for definitions)."))}

      if(isBs==0 & (s_notau_mass>1.01*(1-tau0U)) & tau_hat>0){warning(paste0("For tau=", isTau, isCov, ": the integral of stilde(y)=s(y)*(1-tau0) is larger than one minus the smallest possible value of tau0. This could be evidence against the model."))}


      if(isCov=="" & isTau=="estimated" & refinements$refinement_B & keepfuzzy){
        if(tau0U>0){
          if(isBs==0){warning(paste0("For tau=", isTau, isCov, ": refinement B will not be computed because tau/(1-tau) > (treatRight-treatLeft))/(1-treatRight), which implies that tau0 > 0."))}
        } else{
          if(s_notau_mass < 1 & isBs==0){warning(paste0("For tau=", isTau, isCov, ": bounds for refinement B impose tau0=0 even though the integral/sum of stilde(y)=s(y)*(1-tau0) is less than unity, suggesting that tau0>0."))}
        }
      }

      #End warnings/checks

      if(discrete_y){
        F<-pmf2cdf(pmfy=s_notau)
      } else{
        F<-pdf2cdf(little_ys=little_ys, pdfy=s_notau)
      }

      #Get grid of tau0, tau1 pairs
      #-----------------------------------------------------
      if(s_notau_mass<1-tau0U & tau_hat>0){
        tau0U = 1-s_notau_mass
        tau1U = (tau_hat - tau0U*(1-treatRight))/treatRight
      }

      #Define sequence of num_tau_pairs values between extremes
      if(num_tau_pairs>1){
        tau1s<-(0:(num_tau_pairs-1))/(num_tau_pairs-1)*(tau1U-tau1L)+tau1L
        tau0s<-(0:(num_tau_pairs-1))/(num_tau_pairs-1)*(tau0U-tau0L)+tau0L
      }
      else if(num_tau_pairs==1){
        #if num_tau_pairs = 1, make the single tau the "rightmost" extreme of the t-set, such that user can enforce the always treated only assumption if it is consistent with data
        tau1s<-c(tau1U)
        tau0s<-c(tau0U)
      }
      else{stop("num_tau_pairs must be greater than or equal to one")}

      #define a vector "N0_weights" that is kappa0 times 1-tau0
      N0_weights<-(1-tau0s)*kappa0

      F_Y0_FRD_t<-lapply(
        1:length(tau0s),
        function(t)
        {
          if(!keepfuzzy){
            F_Y0_FRD_L<-NA
            F_Y0_FRD_U<-NA
          } else {
            if(tau0s[t]==1){
              F_Y0_FRD_L<-inputs$F_left_untreated
              F_Y0_FRD_U<-inputs$F_left_untreated
            }
            else {
              if(tau0s[t]==0){
                F_N0_L<-inputs$F_right_untreated
                F_N0_U<-inputs$F_right_untreated
              }
              else{

                #Fs will have a mass of s_notau_mass/1-tau0 ge 1
                d<-data.frame(y=1:length(F), Fs=F/(1-tau0s[t])) #note: here y=1:length(cdf) is only being used as a strictly increasing index of y; we're not assuming that y takes values on the integers

                #For F_N0_L, want to find the point QL at which it integrates to one to the left of QL, the 1 "quantile" of Fs.
                quantile=min(d$y[d$Fs>=1])
                F_N0_L<-(d$y<quantile)*d$Fs+(d$y>=quantile)*1

                #For F_N0_U, want to find the point QU at which it integrates to one to the right of QU, the s_notau_mass/(1-tau0)-1 "quantile" of Fs
                toTrim=s_notau_mass/(1-tau0s[t])-1
                quantile=min(d$y[d$Fs>=toTrim])
                F_N0_U<-(d$y>=quantile)*(d$Fs-toTrim)
              }

              F_Y0_FRD_L<-(inputs$F_left_untreated-N0_weights[t]*F_N0_U)/(1-N0_weights[t])
              F_Y0_FRD_U<-(inputs$F_left_untreated-N0_weights[t]*F_N0_L)/(1-N0_weights[t])
            }
          }

          list(little_y=little_ys, upper=F_Y0_FRD_U, lower=F_Y0_FRD_L)
        }
      )
    }

    if(keepfuzzy){
      #Bounds for Y1
      kappa1<-(1-tau_hat)*treatLeft/treatRight

      if(sum((dist_cut<0)&(treatment==1)&(W>0))==0){
        #If there are no treated units on the left, then kappa1 = 0 and the CDF G is just equal to F_right_treated
        inputs$G<-inputs$F_right_treated
      } else{
        inputs$G<-(inputs$F_right_treated-kappa1*inputs$F_left_treated)/(1-kappa1)
        #Correct G()
        if(mean(inputs$G!=pmin(pmax(inputs$G,0),1))>.02){warning_bs(paste0("warning from bootstrap: ", isBs, ", tau=", isTau, isCov, ": the function G(y) (see paper for definition) should be a CDF, but it contains more than 2% of values outside of the unit interval. Values have been censored to the unit interval."), warningfile=warningfile)}
        inputs$G<-pmin(pmax(inputs$G,0),1)
        if(mean(inputs$G != sort(inputs$G))>.02){warning_bs(paste0("Warning from bootstrap: ", isBs, ", tau=", isTau, isCov, ": the function G(y) (see paper for definition) should be a CDF, but is not completely monotonic (at least 2% would need to be reordered). Values have been monotonized, but this could be evidence against the model."), warningfile=warningfile)}
        inputs$G<-sort(inputs$G)
      }

      F_Y1_FRD_t<-lapply(
        1:length(tau0s),
        function(t)
        {
          if(tau1s[t]/(1-kappa1)>=1){
            #if max(tau1) = 1-kappa1, we can't for this maximal value of tau1 trim the quantity tau1/1-kappa1, since that's one
            #the statement below "This possible value of tau1 will be ignored" is true because bounds_estimate uses which.min rather than min, so it will simply ignore the NA that will result from trimming 1 below
            warning_bs(paste0("Warning from bootstrap: ", isBs, ", tau=", isTau, isCov, ": the admissible value tau1=",tau1s[t]," results in zero potentially assigned compliers on the right of the cutoff. This possible value of tau1 will be ignored."), warningfile=warningfile)
          }

          upper<-trimLeft(cdf=inputs$G, t= (tau1s[t]/(1-kappa1)))
          lower<-trimRight(cdf=inputs$G, t= (tau1s[t]/(1-kappa1)))
          list(little_y=little_ys, upper=upper, lower=lower)
        }
      )
    }
    else{
      F_Y1_FRD_t<-lapply(1:length(tau0s),function(t) list(little_y=little_ys, upper=NA, lower=NA))
    }

    if(!keepfuzzy){
      F_Y0_FRD_t<-lapply(1:length(tau0s),function(t) list(little_y=little_ys, upper=NA, lower=NA))
    }

  } else {
    F_Y0_FRD_t<-NA
    F_Y1_FRD_t<-NA
  }

  #CDFs under no manipulation (for estimation of naive effects)
  #----------------------------------------------------------------
  #Sharp
  F_Y0_SRD_n<-inputs$F_left
  F_Y1_SRD_n<-inputs$F_right
  #Fuzzy
  if(fuzzy){
    kappa0_n<-(1-treatRight)/(1-treatLeft)
    if(sum((dist_cut>=0)&(treatment==0)&(W>0))==0){
      F_Y0_FRD_n<-inputs$F_left_untreated
    } else{
      F_Y0_FRD_n<-(inputs$F_left_untreated-kappa0_n*inputs$F_right_untreated)/(1-kappa0_n)
    }

    kappa1_n<-treatLeft/treatRight
    if(sum((dist_cut<0)&(treatment==1)&(W>0))==0){
      F_Y1_FRD_n<-inputs$F_right_treated
    } else{
      F_Y1_FRD_n<-(inputs$F_right_treated-kappa1_n*inputs$F_left_treated)/(1-kappa1_n)
    }
  }
  else{
    F_Y1_FRD_n<-NA
    F_Y0_FRD_n<-NA
  }
  naive_Fs<-list(little_y=little_ys,F_Y0_SRD=F_Y0_SRD_n, F_Y1_SRD=F_Y1_SRD_n,F_Y0_FRD=F_Y0_FRD_n, F_Y1_FRD=F_Y1_FRD_n)

  if(fuzzy){
    prelims<-list(tau_hat=tau_hat, tau1s=tau1s, tau0s=tau0s, treatLeft=treatLeft, treatRight=treatRight, takeup_increase=takeup_increase)
  } else {
    prelims<-list(tau_hat=tau_hat, tau1s=NA, tau0s=NA, treatLeft=treatLeft, treatRight=treatRight, takeup_increase=takeup_increase)
  }

  return(list(F_Y0_SRD=F_Y0_SRD,F_Y1_SRD=F_Y1_SRD,F_Y0_FRD_t=F_Y0_FRD_t,F_Y1_FRD_t=F_Y1_FRD_t, naive_Fs=naive_Fs, inputs=inputs, prelims=prelims))
}

#ESTIMATION OF CDF BOUNDS WITH COVARIATE REFINEMENT
#------------------------------------------------------------------
get_covCDFs <- function(covinputs=NULL, type, covs, treatLefts, treatRights, tau_w, y, dist_cut, fuzzy, treatment, polynomial, order, W, Wcov, discrete_y, bwy, bwycov, kernel_y, yboundaries, num_tau_pairs, CDFs, little_ys, u, progressFile, kernel, warningfile=NULL, isBs=0, refinements=NULL){

  num_levels<-length(levels(covs))

  #Get probabailities for each cell of covs
  pw<-unlist(lapply(
    levels(covs),
    function(w)
    {
      isvaluew<-(covs==w)
      probLimit<-as.numeric(lm(isvaluew~polynomial, weights=W, subset=(dist_cut<0))$coefficients["(Intercept)"])
      if(probLimit<(-0.00000001)){
        warning_bs(paste0("Warning: estimated left limit of prob(covs=w|x) at the cutoff for w=",w, " is ", round(probLimit,digits=8), " < 0 , replacing with zero"), warningfile=warningfile)
        probLimit<-0
      }
      if(probLimit>1.00000001){
        warning_bs(paste0("Warning: estimated left limit of prob(covs=w|x) at the cutoff for w=",w," is ", round(probLimit,digits=8), " > 1, replacing with one"), warningfile=warningfile)
        probLimit<-1
      }
      probLimit
    }
  ))
  #Normalize sum of probabilities to one
  pw<-pw/sum(pw)

  #Estimate CDFs for each value of w
  if(is.null(covinputs)){covinputs<-lapply(1:num_levels, function(w) NULL);saveinputs<-TRUE} else{saveinputs<-FALSE}
  CDFs_w<-lapply(
    1:num_levels,
    function(w)
    {
      wval=levels(covs)[w]
      if(isBs<1){
        print_message(paste0(".....Estimating covariate-conditional CDF for w=",wval), progressFile=NULL)
      } else{
        print_message(paste0(".....Estimating covariate-conditional CDF for w=",wval,", bootstrap sample: ",isBs), progressFile=NULL)
      }
      estimateCDFs(inputs=covinputs[[w]], tau_hat=tau_w[w], treatLeft=treatLefts[w], treatRight=treatRights[w], y=y[covs==wval], dist_cut=dist_cut[covs==wval], fuzzy=fuzzy, treatment=treatment[covs==wval], polynomial=generate_polynomial_matrix(dist_cut[covs==wval], order), W=Wcov[covs==wval], num_tau_pairs=num_tau_pairs, tau1s=tau1s[[w]], tau0s=tau0s[[w]], little_ys=little_ys, discrete_y=discrete_y, bwy=bwycov, kernel_y=kernel_y, yboundaries=yboundaries, isCov=paste0(", cov = ", wval), warningfile=warningfile, isBs=isBs, refinements=refinements)
    }
  )
  if(saveinputs){covinputs<-lapply(1:num_levels, function(w) CDFs_w[[w]]$inputs)}
  treatLefts<-unlist(lapply(1:num_levels, function(w) CDFs_w[[w]]$prelims$treatLeft)); treatRights<-unlist(lapply(1:num_levels, function(w) CDFs_w[[w]]$prelims$treatRight))
  tau1s<-lapply(1:num_levels, function(w) CDFs_w[[w]]$prelims$tau1s); tau0s<-lapply(1:num_levels, function(w) CDFs_w[[w]]$prelims$tau0s);

  #Integrate over values of covs to get bounds for CDF

  #Sharp
  #--------------------------------
  #could also use the following for Y0 (by law of total prob), but its natural to expect F_left to perform better in finite sample
  #F_Y0_SRD<-data.frame(little_y=little_ys, lower=matrix(unlist(lapply(1:num_levels, function(w) CDFs_w[[w]]$F_Y0_SRD$lower)), ncol=num_levels)%*%pw, upper=matrix(unlist(lapply(1:num_levels, function(w) CDFs_w[[w]]$F_Y0_SRD$upper)), ncol=num_levels)%*%pw)
  F_Y0_SRD<-data.frame(little_y=little_ys, lower=CDFs$inputs$F_left, upper=CDFs$inputs$F_left)
  F_Y1_SRD<-data.frame(little_y=little_ys, lower=matrix(unlist(lapply(1:num_levels, function(w) CDFs_w[[w]]$F_Y1_SRD$lower)), ncol=num_levels)%*%pw, upper=matrix(unlist(lapply(1:num_levels, function(w) CDFs_w[[w]]$F_Y1_SRD$upper)), ncol=num_levels)%*%pw)

  #Fuzzy
  #--------------------------------

  if(fuzzy){
    #Estimate Pi_minus (input to omega)
    Pi_minus_t_w<-lapply(1:num_tau_pairs,
       function(i){
         Pi_minus_w<-unlist(lapply(1:num_levels,
             function(w)
             {
               Pi_minus<-as.numeric((1-tau1s[[w]][i])/(1-tau_w[w])*treatRights[w]-treatLefts[w])
               if(Pi_minus<0){
                 warning_bs(paste0("Warning: estimated proportion of potentially-assigned compliers at the cutoff conditional on w=",w," is negative, replacing with zero"), warningfile=warningfile)
                 Pi_minus<-0
               }
               if(Pi_minus>1){
                 warning_bs(paste0("Warning: estimated proportion of potentially-assigned compliers at the cutoff conditional on w=",w," is greater than unity, replacing with one"), warningfile=warningfile)
                 Pi_minus<-1
               }
               Pi_minus
             }
         ))
         Pi_minus_w
             }
    )
    #Get all combinations of t-values across the values of w (the cartesian product of the vector (1:num_tau_pairs), num_levels times)
    tcombos<-expand.grid(data.frame(matrix(unlist(lapply(1:num_levels,function(w) 1:num_tau_pairs)), ncol=num_levels)))
    num_combos<-dim(tcombos)[1]

    #For each combination, get the vector of omegas corresponding to the t(w), across values w
    omega_numerators<-lapply(1:num_combos, function(i) unlist(lapply(1:num_levels, function(w) Pi_minus_t_w[[tcombos[i,w]]][w])))
    omegas<-lapply(1:num_combos, function(i) omega_numerators[[i]]/rep(omega_numerators[[i]]%*%pw, length(omega_numerators[[i]]))) #normalize

    #Get tau1s and tau0s as well to pass to bounds_estimate
    tau1s<-lapply(1:num_combos, function(i) unlist(lapply(1:num_levels, function(w) tau1s[[w]][tcombos[i,w]])))
    tau0s<-lapply(1:num_combos, function(i) unlist(lapply(1:num_levels, function(w) tau0s[[w]][tcombos[i,w]])))

    F_Y0_FRD_W_t_lower<-lapply(1:num_combos, function(i) matrix(unlist(lapply(1:num_levels, function(w) CDFs_w[[w]]$F_Y0_FRD_t[[tcombos[i,w]]]$lower)), ncol=num_levels)%*%diag(omegas[[i]])%*%pw)
    F_Y0_FRD_W_t_upper<-lapply(1:num_combos, function(i) matrix(unlist(lapply(1:num_levels, function(w) CDFs_w[[w]]$F_Y0_FRD_t[[tcombos[i,w]]]$upper)), ncol=num_levels)%*%diag(omegas[[i]])%*%pw)
    F_Y1_FRD_W_t_lower<-lapply(1:num_combos, function(i) matrix(unlist(lapply(1:num_levels, function(w) CDFs_w[[w]]$F_Y1_FRD_t[[tcombos[i,w]]]$lower)), ncol=num_levels)%*%diag(omegas[[i]])%*%pw)
    F_Y1_FRD_W_t_upper<-lapply(1:num_combos, function(i) matrix(unlist(lapply(1:num_levels, function(w) CDFs_w[[w]]$F_Y1_FRD_t[[tcombos[i,w]]]$upper)), ncol=num_levels)%*%diag(omegas[[i]])%*%pw)

    F_Y0_FRD_t<-lapply(1:num_combos, function(i) {
      data.frame(little_y=little_ys, lower=F_Y0_FRD_W_t_lower[[i]], upper=F_Y0_FRD_W_t_upper[[i]])
    })

    F_Y1_FRD_t<-lapply(1:num_combos, function(i) {
      data.frame(little_y=little_ys, lower=F_Y1_FRD_W_t_lower[[i]], upper=F_Y1_FRD_W_t_upper[[i]])
    })

  } else{
    F_Y0_FRD_t<-NA
    F_Y1_FRD_t<-NA
  }
  covCDFs<-list(F_Y0_SRD=F_Y0_SRD,F_Y1_SRD=F_Y1_SRD,F_Y0_FRD_t=F_Y0_FRD_t,F_Y1_FRD_t=F_Y1_FRD_t)
  return(list(tau1s=tau1s, tau0s=tau0s, CDFs=covCDFs, covinputs=covinputs))
}

#ESTIMATION OF CDFS FOR UNITS JUST TO THE RIGHT OF THE CUTOFF
#------------------------------------------------------------------
get_gammaPlusCDFs <- function(mainCDFs, yextremes, num_lambdas, num_tau_pairs, tau_hat, tau1s, tau0s, treatLeft, treatRight, little_ys, progressFile, fuzzy, isBs=0){

  if(isBs < 1){
    print_message(paste0(".....Estimating CDFs for units just to the right of the cutoff"), progressFile=progressFile)
  } else{
    print_message(paste0(".....Estimating CDFs for units just to the right of the cutoff, bootstrap sample: ",isBs), progressFile=NULL)
  }

  inputs<-mainCDFs$inputs

  #Sharp
  #--------------------------------
  F_Y0_SRD<-data.frame(little_y=little_ys, lower=(1-tau_hat)*inputs$F_left+tau_hat*(little_ys>=yextremes[1]), upper=(1-tau_hat)*inputs$F_left+tau_hat*(little_ys>=yextremes[2]))
  F_Y1_SRD<-data.frame(little_y=little_ys, lower=inputs$F_right, upper=inputs$F_right)

  #Fuzzy
  #--------------------------------
  if(fuzzy){
    #Get all combinations of t-values and lambda-values
    tlambdaCombos<-expand.grid(1:num_tau_pairs, 1:num_lambdas)
    num_combos<-dim(tlambdaCombos)[1]

    lambdas<-(0:(num_lambdas-1))/(num_lambdas-1)
    tau0_combos<-lapply(1:num_combos, function(i) list(tau0=tau0s[tlambdaCombos[i,1]], lambda=lambdas[tlambdaCombos[i,2]]))
    tau1_combos<-lapply(1:num_combos, function(i) list(tau1=tau1s[tlambdaCombos[i,1]], lambda=lambdas[tlambdaCombos[i,2]]))

    kappa1<-(1-tau_hat)*treatLeft/treatRight

    F_Y1_FRD_t<-lapply(
      1:num_combos,
      function(i)
      {
        toTrim<-tau1s[tlambdaCombos[i,1]]*lambdas[tlambdaCombos[i,2]]/(1-kappa1)
        upper<-trimLeft(cdf=inputs$G, t=toTrim)
        lower<-trimRight(cdf=inputs$G, t=toTrim)
        list(little_y=little_ys, upper=upper, lower=lower)
      }
    )

    staulambda<-((1-tau1s[tlambdaCombos[,1]])*treatRight-(1-tau_hat)*treatLeft)/((1-tau1s[tlambdaCombos[,1]]*lambdas[tlambdaCombos[,2]])*treatRight-(1-tau_hat)*treatLeft)
    F_Y0_FRD_t<-lapply(
      1:num_combos,
      function(i)
      {
        F_Y0_FRD_L<-staulambda[i]*mainCDFs$F_Y0_FRD_t[[tlambdaCombos[i,1]]]$lower+(1-staulambda[i])*(little_ys >= yextremes[1])
        F_Y0_FRD_U<-staulambda[i]*mainCDFs$F_Y0_FRD_t[[tlambdaCombos[i,1]]]$upper+(1-staulambda[i])*(little_ys >= yextremes[2])
        list(little_y=little_ys, upper=F_Y0_FRD_U, lower=F_Y0_FRD_L)
      }
    )
  }
  else{
    F_Y1_FRD_t<-NA
    F_Y0_FRD_t<-NA
    tau1_combos<-NA
    tau0_combos<-NA
  }
  return(list(CDFs=list(F_Y0_SRD=F_Y0_SRD,F_Y1_SRD=F_Y1_SRD,F_Y0_FRD_t=F_Y0_FRD_t,F_Y1_FRD_t=F_Y1_FRD_t), tau1s=tau1_combos, tau0s=tau0_combos))

}

#WrAPPER FUNCTION FOR ALL PREP WORK BEFORE ESTIMATION
#------------------------------------------------------------------
get_allCDFs <- function(CDFinputs=NULL, tau_hat, y, dist_cut, fuzzy, treatment, polynomial, W, Wcov, num_tau_pairs, little_ys, discrete_y, discrete_x, gotCovs, covs, tau_w, right_effects, yextremes, bwy, bwycov, kernel_y, yboundaries, num_lambdas, isBs=0, isTau=NULL, isStar=FALSE, orders, bwsx, bwsxcov, kernel, progressFile, warningfile=NULL, refinements){

  if(is.null(CDFinputs)){CDFinputs<-list(inputs=NULL, covinputs=NULL); saveinputs<-TRUE} else{saveinputs<-FALSE}

  #GET CDFS
  #------------------------------------------------------------------
  #The following messages won't be written to progressFile, because they look confusing if things are run in parallel and they come out of order
  if(isStar){
    if(is.null(isTau)){print_message(paste0("Estimating CDFs with nudged tau (tau_star)"), progressFile=NULL)}
    if(!is.null(isTau)){print_message(paste0("Estimating CDFs with fixed tau value of: ",isTau), progressFile=NULL)}
    isBs<-(-1)
  } else if(isBs==0){
    if(is.null(isTau)){print_message(paste0("Estimating CDFs for point estimates"), progressFile=NULL)}
    if(!is.null(isTau)){print_message(paste0("Estimating CDFs for point estimates, tau value of: ",isTau), progressFile=NULL)}
  } else{
    if(is.null(isTau)){print_message(paste0("Estimating CDFs for bootstrap sample: ",isBs), progressFile=NULL)}
    if(!is.null(isTau)){print_message(paste0("Estimating CDFs for bootstrap sample: ",isBs, ", tau value of: ",isTau), progressFile=NULL)}
  }

  #Main CDFs
  CDFs<-estimateCDFs(inputs=CDFinputs$inputs, tau_hat=tau_hat, y=y, dist_cut=dist_cut, fuzzy=fuzzy, treatment=treatment, polynomial=polynomial, W=W, num_tau_pairs=num_tau_pairs, little_ys=little_ys, discrete_y=discrete_y, bwy=bwy, kernel_y=kernel_y, yboundaries=yboundaries, isBs=isBs, isTau=isTau, warningfile=warningfile, refinements=refinements)
  if(saveinputs){CDFinputs$inputs=CDFs$inputs}
  prelims = CDFs$prelims
  treatLeft<-prelims$treatLeft; treatRight<-prelims$treatRight; takeup_increase<-prelims$takeup_increase; tau1s<-prelims$tau1s; tau0s<-prelims$tau0s

  #CDFs for covariate refinement
  covCDFs<-NA
  if(gotCovs){
    covCDFs<-get_covCDFs(CDFinputs$covinputs, type=type, covs=covs, tau_w=tau_w, y=y, dist_cut=dist_cut, fuzzy=fuzzy, treatment=treatment, polynomial=polynomial, order=orders[2], W=W, Wcov=Wcov, discrete_y=discrete_y, bwy=bwy, bwycov=bwycov, kernel_y=kernel_y, yboundaries=yboundaries, num_tau_pairs=num_tau_pairs, CDFs=CDFs, little_ys=little_ys, progressFile=progressFile, kernel=kernel, warningfile=warningfile, isBs=isBs, refinements=refinements)
    if(saveinputs){CDFinputs$covinputs=covCDFs$covinputs}
  }

  #CDFs for "right side" cDFs to estimate Gamma_plus
  rightCDFs<-NA
  if(right_effects){rightCDFs<-get_gammaPlusCDFs(mainCDFs=CDFs, yextremes=yextremes, num_lambdas=num_lambdas, num_tau_pairs=num_tau_pairs, tau_hat=tau_hat, tau1s=tau1s, tau0s=tau0s, treatLeft=treatLeft, treatRight=treatRight, little_ys=little_ys, progressFile=progressFile, fuzzy=fuzzy, isBs=isBs)}

  #If CDFinputs was not just computed, don't save it again, which would be redundant and waste RAM
  if(!saveinputs){
    CDFinputs<-NA
  }

  return(list(CDFs=CDFs, covCDFs=covCDFs, rightCDFs=rightCDFs, prelims=prelims, CDFinputs=CDFinputs))
}

#GET SETUP FOR BOOSTRAP (INCLUDING COMPUTING CDFS FOR BOOTSTRAP SAMPLES)
#------------------------------------------------------------------
CIs_setup<-function(tau_tilde, dist_cut, num_bootstraps, alpha, Kn, discrete_x, orders, bwsx, bwsxcov, kernel, gotCovs, covs){

  #Setup
  #-----------------------------------------------------------------------------------
  tau_hat<-max(tau_tilde,0)

  #Generate bootstrap samples
  B<-num_bootstraps[1]
  n<-length(dist_cut)

  resample_indices <- lapply(1:B, function(b) sample(1:n, replace = TRUE))

  #Compute bootstrap tau_hats
  #-----------------------------------------------------------------------------------
  tau_tilde_stars<-unlist(lapply(1:B, function(b) compute_tau_tilde(kernel=kernel, order=orders[1], x=dist_cut[unlist(resample_indices[b])], bwsx=bwsx, discrete_x=discrete_x)))
  sd_tau<-sd(tau_tilde_stars)
  if(is.null(Kn)){
    Kn<-sqrt(log(n))
  }

  #Nudge tau away from zero, if necessary
  tau_star<-max(tau_tilde,Kn*sd_tau)
  tau_tildes<-tau_tilde_stars-tau_tilde+max(tau_hat,Kn*sd_tau)
  tau_hats_b<-unlist(lapply(1:B, function(b) max(0,tau_tildes[b])))
  if(max(tau_hats_b) >= 1){
    warning("Warning: the bootstrap nudge (see step 3 of inference section of paper) resulted in tau values greater than one, replacing with one")
    tau_hats_b<-pmin(tau_hats_b,1)
  }

  if(gotCovs){
    tau_tilde_stars_w<-sapply(levels(covs), function(w) unlist(lapply(1:B, function(b) compute_tau_tilde(kernel=kernel, order=orders[1], x=dist_cut[unlist(resample_indices[b])][covs[unlist(resample_indices[b])]==w], bwsx=bwsxcov, discrete_x=discrete_x))), USE.NAMES=TRUE)
    #Nudge tau_w's away from zero, if necessary
    sd_tau_w<-sapply(levels(covs), function(w) sd(tau_tilde_stars_w[,w]), USE.NAMES=TRUE)
    tau_tilde_w<-sapply(levels(covs), function(w) compute_tau_tilde(kernel=kernel, order=orders[1], x=dist_cut[covs==w], bwsx=bwsxcov, discrete_x=discrete_x), USE.NAMES=TRUE)
    tau_star_w<-sapply(levels(covs), function(w) max(tau_tilde_w[w],Kn*sd_tau_w[w]), USE.NAMES=TRUE)
    tau_tildes_w<-sapply(levels(covs), function(w) tau_tilde_stars_w[,w]-tau_tilde_w[w]+max(tau_tilde_w[w],Kn*sd_tau_w[w]), USE.NAMES=TRUE)
    tau_hats_w<-sapply(levels(covs), function(w) unlist(lapply(1:B, function(b) max(0,tau_tildes_w[b,w]))), USE.NAMES=TRUE)
    if(max(tau_hats_w) >= 1){
      warning("Warning: the bootstrap nudge (see step 3 of inference section of paper) for one or more of the covariate values resulted in tau values greater than one, replacing with one")
      print("estimated tau_tildes (before nudge) by w and b:")
      print(tau_tildes_w)
      tau_hats_w<-pmin(tau_hats_w,1)
    }
  }
  else{
    tau_hats_w<-NA
    tau_star_w<-NA
  }

  return(list(B=B, resample_indices=resample_indices, tau_star=tau_star,tau_hats_b=tau_hats_b, tau_hats_w=tau_hats_w, tau_star_w=tau_star_w))
}

#GENERATE CONFIDENCE INTERVALS BY NONPARAMETERIC BOOTSTRAP
#------------------------------------------------------------------
CIs_estimate <- function(CDFinputs=NULL, parallelize=FALSE, fuzzy, gotCovs, covs, right_effects, yextremes, discrete_y, num_tau_pairs, num_lambdas, percentiles, tau_tilde, y, dist_cut, treatment, polynomial, W, Wcov, num_bootstraps, alpha, Kn, discrete_x, orders, bwsx, bwsxcov, kernel, little_ys, bwy, bwycov, kernel_y, yboundaries, potential_taus, takeup_increase, progressFile, warningsFile, refinements=refinements){

  #Generate random bootstrap samples (if necessary)
  if(is.null(CDFinputs$CIsetup)){
    CIsetup<-CIs_setup(tau_tilde=tau_tilde, dist_cut=dist_cut, num_bootstraps=num_bootstraps, alpha=alpha, Kn=Kn, discrete_x=discrete_x, orders=orders, bwsx=bwsx, bwsxcov=bwsxcov, kernel=kernel, gotCovs=gotCovs, covs=covs)
    CDFinputs$CIsetup<-CIsetup
  } else{
    CIsetup<-CDFinputs$CIsetup
  }
  B<-CIsetup$B; resample_indices<-CIsetup$resample_indices; tau_star<-CIsetup$tau_star; tau_hats_b<-CIsetup$tau_hats_b; tau_hats_w<-CIsetup$tau_hats_w; tau_star_w=CIsetup$tau_star_w

  #Generate "star" estimates with modified tau_hat
  allCDFs_star<-get_allCDFs(CDFinputs=CDFinputs$original, isStar=TRUE, tau_hat=tau_star, tau_w=tau_star_w, y=y, dist_cut=dist_cut, fuzzy = fuzzy, treatment=treatment, polynomial=polynomial, W=W, Wcov=Wcov, num_tau_pairs=num_tau_pairs, little_ys=little_ys, discrete_y=discrete_y, discrete_x=discrete_x, gotCovs=gotCovs, covs=covs, right_effects=right_effects, yextremes=yextremes, num_lambdas=num_lambdas, bwy=bwy, bwycov=bwycov, kernel_y=kernel_y, yboundaries=yboundaries, orders=orders, bwsx=bwsx, bwsxcov=bwsxcov, kernel=kernel, progressFile=progressFile, refinements=refinements)
  estimates_star_u<-get_estimates_by_TE(allCDFs=allCDFs_star, fuzzy=fuzzy, gotCovs=gotCovs, covs=covs, right_effects=right_effects, yextremes=yextremes, num_lambdas=num_lambdas, percentiles=percentiles, refinements=refinements)

  #Generate estimates for bootstrap samples
  if(is.null(CDFinputs$bootstraps)){CDFinputs$bootstraps<-lapply(1:B, function(b) NULL);saveinputs<-TRUE} else{saveinputs<-FALSE}
  if(parallelize){
    if(is.null(warningsFile)){
      warningfile<-tempfile("warnings")
    }
    else{
      warningfile<-warningsFile
    }
    clus <- parallel::makeCluster(parallel::detectCores()-1,outfile="")
    parallel::clusterExport(clus, envir=environment(), varlist=list("K", "Kh", "CDFinputs", "tau_hats_w", "tau_hats_b", "resample_indices", "y", "dist_cut", "orders", "W", "num_tau_pairs", "little_ys", "discrete_y", "discrete_x", "gotCovs", "covs", "right_effects", "yextremes", "bwy", "kernel_y", "num_lambdas", "bwsx", "kernel", "get_allCDFs", "generate_polynomial_matrix", "estimateCDFs", "compute_CDFinputs", "rbindlist", "trimLeft", "trimRight", "pdf2cdf", "pmf2cdf", "cdf2expectation", "get_covCDFs", "get_gammaPlusCDFs", "get_estimates_by_TE",  "naive_estimate", "bounds_estimate", "print_message", "fuzzy", "yboundaries", "integrate_kernel", "Wcov", "bwsxcov", "bwycov", "warningfile", "warning_bs", "refinements"))
    print_message(paste0("Beginning parallelized output by bootstrap.."), progressFile=progressFile)
    #Generate point estimates for each bootstrap sample
    computations_by_b<-parallel::parLapply(clus, 1:B, function(b){
      this_b_allCDFs<-get_allCDFs(CDFinputs=CDFinputs$bootstraps[[b]], tau_hat=tau_hats_b[b], tau_w=tau_hats_w[b,], y=y[unlist(resample_indices[b])], dist_cut=dist_cut[unlist(resample_indices[b])], fuzzy = fuzzy, treatment=treatment[unlist(resample_indices[b])], polynomial=generate_polynomial_matrix(dist_cut[unlist(resample_indices[b])],orders[2]), W=W[unlist(resample_indices[b])],  Wcov=Wcov[unlist(resample_indices[b])], num_tau_pairs=num_tau_pairs, little_ys=little_ys, discrete_y=discrete_y, discrete_x=discrete_x, gotCovs=gotCovs, covs=covs[unlist(resample_indices[b])], right_effects=right_effects, yextremes=yextremes, num_lambdas=num_lambdas, bwy=bwy, bwycov=bwycov, kernel_y=kernel_y, yboundaries=yboundaries, isBs=b, orders=orders, bwsx=bwsx, bwsxcov=bwsxcov, kernel=kernel, progressFile=progressFile, warningfile=warningfile, refinements=refinements)
      this_b_estimates<-get_estimates_by_TE(allCDFs=this_b_allCDFs, fuzzy=fuzzy, gotCovs=gotCovs, covs=covs, right_effects=right_effects, yextremes=yextremes, num_lambdas=num_lambdas, percentiles=percentiles, isBs=b, refinements=refinements, warningfile=warningfile)
      list(takeup_increase_b=this_b_allCDFs$prelims$takeup_increase, estimates_by_TE_b=this_b_estimates, CDFinputs_b=this_b_allCDFs$CDFinputs)
    })
    parallel::stopCluster(clus)
  } else{
    computations_by_b<-lapply(1:B, function(b){
      this_b_allCDFs<-get_allCDFs(CDFinputs=CDFinputs$bootstraps[[b]], tau_hat=tau_hats_b[b], tau_w=tau_hats_w[b,], y=y[unlist(resample_indices[b])], dist_cut=dist_cut[unlist(resample_indices[b])], fuzzy = fuzzy, treatment=treatment[unlist(resample_indices[b])], polynomial=generate_polynomial_matrix(dist_cut[unlist(resample_indices[b])],orders[2]), W=W[unlist(resample_indices[b])], Wcov=Wcov[unlist(resample_indices[b])], num_tau_pairs=num_tau_pairs, little_ys=little_ys, discrete_y=discrete_y, discrete_x=discrete_x, gotCovs=gotCovs, covs=covs[unlist(resample_indices[b])], right_effects=right_effects, yextremes=yextremes, num_lambdas=num_lambdas, bwy=bwy, bwycov=bwycov, kernel_y=kernel_y, yboundaries=yboundaries, isBs=b, orders=orders, bwsx=bwsx, bwsxcov=bwsxcov, kernel=kernel, progressFile=progressFile, refinements=refinements)
      this_b_estimates<-get_estimates_by_TE(allCDFs=this_b_allCDFs, fuzzy=fuzzy, gotCovs=gotCovs, covs=covs, right_effects=right_effects, yextremes=yextremes, num_lambdas=num_lambdas, percentiles=percentiles, warningfile=warningfile, isBs=b, refinements=refinements)
      list(takeup_increase_b=this_b_allCDFs$prelims$takeup_increase, estimates_by_TE_b=this_b_estimates, CDFinputs_b=this_b_allCDFs$CDFinputs)
    })
  }

  estimates_by_TE_b<-lapply(1:B, function(b) computations_by_b[[b]]$estimates_by_TE_b)
  takeup_increases_b<-unlist(lapply(1:B, function(b) computations_by_b[[b]]$takeup_increase_b))
  if(saveinputs){CDFinputs$bootstraps<-lapply(1:B, function(b) computations_by_b[[b]]$CDFinputs_b)}
  computations_by_b<-NULL #Free up some memory

  #Generate estimates for user-specified fixed tau in both original and bootstrap samples
  if(!is.null(potential_taus)){
    allCDFs_tau_star<-lapply(1:length(potential_taus), function(tau) get_allCDFs(CDFinputs=CDFinputs$original, isStar=TRUE, tau_hat=potential_taus[tau], y=y, dist_cut=dist_cut, fuzzy = fuzzy, treatment=treatment, polynomial=polynomial, W=W, num_tau_pairs=num_tau_pairs, little_ys=little_ys, discrete_y=discrete_y, discrete_x=discrete_x, gotCovs=FALSE, right_effects=FALSE, yextremes=yextremes, num_lambdas=num_lambdas, bwy=bwy, kernel_y=kernel_y, yboundaries=yboundaries, orders=orders, bwsx=bwsx, kernel=kernel, isTau=potential_taus[tau], progressFile=progressFile, refinements=refinements))
    estimates_tau_star<-lapply(1:length(potential_taus), function(tau) get_estimates_by_TE(allCDFs=allCDFs_tau_star[[tau]], fuzzy=fuzzy, gotCovs=FALSE, right_effects=FALSE, yextremes=yextremes, num_lambdas=num_lambdas, percentiles=percentiles, refinements=refinements))

    if(parallelize){
      if(is.null(warningsFile)){
        warningfile<-tempfile("warnings")
      }
      else{
        warningfile<-warningsFile
      }
      clus <- parallel::makeCluster(parallel::detectCores()-1,outfile="")
      tauBcombos<-expand.grid(1:length(potential_taus), 1:B)
      num_tauBcombos<-dim(tauBcombos)[1]
      parallel::clusterExport(clus, envir=environment(), varlist=list("K", "Kh", "tauBcombos", "potential_taus", "CDFinputs", "tau_hats_w", "tau_hats_b", "resample_indices", "y", "dist_cut", "orders", "W", "num_tau_pairs", "little_ys", "discrete_y", "discrete_x", "gotCovs", "covs", "right_effects", "yextremes", "bwy", "kernel_y", "num_lambdas", "bwsx", "kernel", "get_allCDFs", "generate_polynomial_matrix", "estimateCDFs", "compute_CDFinputs", "rbindlist", "trimLeft", "trimRight", "pdf2cdf", "pmf2cdf", "cdf2expectation", "get_covCDFs", "get_gammaPlusCDFs", "get_estimates_by_TE", "naive_estimate", "bounds_estimate", "print_message", "fuzzy", "yboundaries", "integrate_kernel", "Wcov", "bwsxcov", "bwycov", "warningfile", "warning_bs", "refinements"))
      print_message(paste0("Beginning parallelized output by bootstrap x fixed tau.."), progressFile=progressFile)
      estimates_tauBcombos<-parallel::parLapply(clus, 1:num_tauBcombos, function(i){
        tau<-tauBcombos[i,1]
        b<-tauBcombos[i,2]
        this_allCDFs<-get_allCDFs(CDFinputs=CDFinputs$bootstraps[[b]], tau_hat=potential_taus[tau], y=y[unlist(resample_indices[b])], dist_cut=dist_cut[unlist(resample_indices[b])], fuzzy = fuzzy, treatment=treatment[unlist(resample_indices[b])], polynomial=generate_polynomial_matrix(dist_cut[unlist(resample_indices[b])],orders[2]), W=W[unlist(resample_indices[b])], Wcov=Wcov[unlist(resample_indices[b])], num_tau_pairs=num_tau_pairs, little_ys=little_ys, discrete_y=discrete_y, discrete_x=discrete_x, gotCovs=FALSE, right_effects=FALSE, yextremes=yextremes, num_lambdas=num_lambdas, bwy=bwy, bwycov=bwycov, kernel_y=kernel_y, yboundaries=yboundaries, isBs=b, isTau=potential_taus[tau], orders=orders, bwsx=bwsx, bwsxcov=bwsxcov, kernel=kernel, progressFile=progressFile, refinements=refinements, warningfile=warningfile)
        get_estimates_by_TE(allCDFs=this_allCDFs, fuzzy=fuzzy, gotCovs=FALSE, right_effects=FALSE, yextremes=yextremes, num_lambdas=num_lambdas, percentiles=percentiles, isBs=b, refinements=refinements, warningfile=warningfile)
      })
      parallel::stopCluster(clus)
      estimates_tau_b<-lapply(1:length(potential_taus), function(tau) lapply(1:B, function(b) estimates_tauBcombos[[(b-1)*length(potential_taus)+tau]]))

      if(!is.na(file.info(warningfile)$size)){
        if(file.info(warningfile)$size>0){
          warning(paste0("There were supressed warnings from parellized bootstrap runs (or warningFile already had contents before rdbounds was run). Here they are: ", readChar(warningfile, file.info(warningfile)$size)))
        }
      }

      if(!is.null(warningsFile)){
        unlist(warningfile)
      } 

    } else{
      estimates_tau_b<-lapply(1:length(potential_taus), function(tau) lapply(1:B, function(b){
        this_allCDFs<-get_allCDFs(CDFinputs=CDFinputs$bootstraps[[b]], tau_hat=potential_taus[tau], y=y[unlist(resample_indices[b])], dist_cut=dist_cut[unlist(resample_indices[b])], fuzzy = fuzzy, treatment=treatment[unlist(resample_indices[b])], polynomial=generate_polynomial_matrix(dist_cut[unlist(resample_indices[b])],orders[2]), W=W[unlist(resample_indices[b])], Wcov=Wcov[unlist(resample_indices[b])], num_tau_pairs=num_tau_pairs, little_ys=little_ys, discrete_y=discrete_y, discrete_x=discrete_x, gotCovs=FALSE, right_effects=FALSE, yextremes=yextremes, num_lambdas=num_lambdas, bwy=bwy, bwycov=bwycov, kernel_y=kernel_y, yboundaries=yboundaries, isBs=b, isTau=potential_taus[tau], orders=orders, bwsx=bwsx, bwsxcov=bwsxcov, kernel=kernel, progressFile=progressFile, refinements=refinements)
        get_estimates_by_TE(allCDFs=this_allCDFs, fuzzy=fuzzy, gotCovs=FALSE, right_effects=FALSE, yextremes=yextremes, num_lambdas=num_lambdas, percentiles=percentiles, isBs=b, refinements=refinements)
      }))
    }
  }
  else{
    estimates_tau_star<-NA
    estimates_tau_b<-NA
  }

  #CALCULATE CIs
  #-----------------------------------------------------------------------------------
  #-----------------------------------------------------------------------------------
  print_message(paste0("Computing Confidence Intervals"), progressFile=progressFile)

  #Define function to search for r_alpha by numerical optimization
  rfn<- function(r,GammaU,GammaL,sU,sL) {abs(pnorm(r+(GammaU-GammaL)/max(sU,sL))-pnorm(-r)-(1-alpha))}

  minimize_check <-function(rfn, interval,GammaU, GammaL,sU,sL)
  {
    if(is.na(sU)|is.na(sL)){
      if(!already_warned){warning("a value of sU or sL is NA for some reason, possibly due to tau0=1-kappa1, or because all but one bootstrap sample was dropped for some estimand (note: you may need to parellize=FALSE, or set warningsFile to see warnings from bootstrap iterations). Using qnorm(1-alpha/2) here.")}
      already_warned <<- TRUE
      return(qnorm(1-alpha/2))
    }
    if(max(sU,sL)==0){
      if(!already_warned){warning("max of SU and sL is zero, using qnorm(1-alpha/2)")}
      already_warned <<- TRUE
      return(qnorm(1-alpha/2))
    } else{
      return(optimize(rfn,interval=interval, GammaU=GammaU, GammaL=GammaL, sU=sU, sL=sL)$minimum)
    }
  }
  already_warned <<- FALSE

  if(alpha>=0.5){warning("For confidence intervals r_alpha calculated assuming that alpha<0.5. Since alpha>=0.5, true r_alpha may be very far from value used.")}

  #CI for tau_hat
  tau_hat_CI<-c(tau_star-qnorm(1-alpha/2)*sd(unlist(tau_hats_b)),tau_star+qnorm(1-alpha/2)*sd(unlist(tau_hats_b)))

  #CI for takeup increase
  takeup_increase_CI<-c(takeup_increase-qnorm(1-alpha/2)*sd(unlist(takeup_increases_b)),takeup_increase+qnorm(1-alpha/2)*sd(unlist(takeup_increases_b)))

  CIs_by_TE <-sapply(
    1:length(percentiles),
    function(i)
    {

      #CI's for naive estimates
      TE_SRD_naives<-unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_naive',i]]$TE_SRD_naive))
      TE_FRD_naives<-unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_naive',i]]$TE_FRD_naive))
      TE_SRD_naive_CI = c(estimates_star_u[['estimates_naive',i]]$TE_SRD_naive-qnorm(1-alpha/2)*sd(TE_SRD_naives),estimates_star_u[['estimates_naive',i]]$TE_SRD_naive+qnorm(1-alpha/2)*sd(TE_SRD_naives, na.rm=TRUE))
      TE_FRD_naive_CI = c(estimates_star_u[['estimates_naive',i]]$TE_FRD_naive-qnorm(1-alpha/2)*sd(TE_FRD_naives),estimates_star_u[['estimates_naive',i]]$TE_FRD_naive+qnorm(1-alpha/2)*sd(TE_FRD_naives, na.rm=TRUE))

      estimates_star<-estimates_star_u[['estimates_main',i]]

      #CI's for manipulation robust estimates
      #------------------------------------------------------------------
      #CI for SRD under manipulation
      sL<-sd(unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_main',i]]$TE_lower_SRD)), na.rm=TRUE)
      sU<-sd(unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_main',i]]$TE_upper_SRD)), na.rm=TRUE)

      r<-minimize_check(rfn, interval=c(qnorm(1-alpha), qnorm(1-alpha/2)),GammaU=estimates_star$TE_upper_SRD,GammaL=estimates_star$TE_lower_SRD,sU=sU,sL=sL)
      TE_SRD_CI<-c(estimates_star$TE_lower_SRD-r*sL,estimates_star$TE_upper_SRD+r*sU)

      #CI for FRD under manipulation
      if(fuzzy){
        sL_t<-unlist(lapply(1:num_tau_pairs, function(t) sd(unlist(lapply(1:B, function(b) data.frame(estimates_by_TE_b[[b]][['estimates_main',i]]$TE_components)$TE_tau1s_lower[t])), na.rm=TRUE)))
        sU_t<-unlist(lapply(1:num_tau_pairs, function(t) sd(unlist(lapply(1:B, function(b) data.frame(estimates_by_TE_b[[b]][['estimates_main',i]]$TE_components)$TE_tau1s_upper[t])), na.rm=TRUE)))

        TE_lower_star_t<-data.frame(estimates_star$TE_components)$TE_tau1s_lower
        TE_upper_star_t<-data.frame(estimates_star$TE_components)$TE_tau1s_upper
        r_t<-unlist(lapply(1:num_tau_pairs, function(t) minimize_check(rfn, interval=c(qnorm(1-alpha), qnorm(1-alpha/2)),GammaU=TE_upper_star_t[t],GammaL=TE_lower_star_t[t],sU=sU_t[t],sL=sL_t[t])))

        lowerbounds_t<- data.frame(estimates_star$TE_components)$TE_tau1s_lower-r_t*sL_t
        upperbounds_t<-data.frame(estimates_star$TE_components)$TE_tau1s_upper+r_t*sU_t
        #note: using which,min rather than min to be robust to NAs, ignoring them
        TE_FRD_CI<-c(lowerbounds_t[which.min(lowerbounds_t)], upperbounds_t[which.max(upperbounds_t)])

        if(refinements$refinement_A){
          #CI for SRD under manipulation, refinement A
          refA<-which(data.frame(estimates_star$TE_components)$refinementA==1)
          sL_t<-sL_t[refA]
          sU_t<-sU_t[refA]
          TE_lower_star_t<-data.frame(estimates_star$TE_components)$TE_tau1s_lower[refA]
          TE_upper_star_t<-data.frame(estimates_star$TE_components)$TE_tau1s_upper[refA]
          r_t<-unlist(lapply(1:length(refA), function(t) minimize_check(rfn, interval=c(qnorm(1-alpha), qnorm(1-alpha/2)),GammaU=TE_upper_star_t[t],GammaL=TE_lower_star_t[t],sU=sU_t[t],sL=sL_t[t])))
          lowerbounds_t<-TE_lower_star_t-r_t*sL_t
          upperbounds_t<-TE_upper_star_t+r_t*sU_t
          TE_FRD_refinementA_CI<-c(lowerbounds_t[which.min(lowerbounds_t)], upperbounds_t[which.max(upperbounds_t)])
        } else {
          TE_FRD_refinementA_CI<-c(NA, NA)
        }

        if(refinements$refinement_B){
          #CI for SRD under manipulation, refinement B
          sL_B<-sd(unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_main',i]]$TE_lower_FRD_refinementB)), na.rm=TRUE)
          sU_B<-sd(unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_main',i]]$TE_upper_FRD_refinementB)), na.rm=TRUE)

          #It could still be the case that refinement B was not computed, if it was deemed incompatible with the data, so catch this case:
          if(!is.na(sL_B)){
            r_B<-minimize_check(rfn, interval=c(qnorm(1-alpha), qnorm(1-alpha/2)),GammaU=estimates_star$TE_upper_FRD_refinementB,GammaL=estimates_star$TE_lower_FRD_refinementB,sU=sU_B,sL=sL_B)
            TE_FRD_refinementB_CI<-c(estimates_star$TE_lower_FRD_refinementB-r_B*sL_B, estimates_star$TE_upper_FRD_refinementB+r_B*sU_B)
          } else{
            TE_FRD_refinementB_CI<-c(NA,NA)
          }
        } else {
          TE_FRD_refinementB_CI<-c(NA, NA)
        }

      }
      else{
        TE_FRD_CI<-c(NA,NA)
        TE_FRD_refinementA_CI<-c(NA,NA)
        TE_FRD_refinementB_CI<-c(NA,NA)
      }

      # CI's for effects on units just to the right of the cutoff
      #------------------------------------------------------------------
      if(right_effects){

        estimates_star_rightside<-estimates_star_u[['estimates_rightside',i]]

        #CI for SRD under manipulation
        sL<-sd(unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_rightside',i]]$TE_SRD_right_bounds[1])), na.rm=TRUE)
        sU<-sd(unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_rightside',i]]$TE_SRD_right_bounds[2])), na.rm=TRUE)
        r<-minimize_check(rfn, interval=c(qnorm(1-alpha), qnorm(1-alpha/2)),GammaU=estimates_star_rightside$TE_SRD_right_bounds[2],GammaL=estimates_star_rightside$TE_SRD_right_bounds[1],sU=sU,sL=sL)
        TE_SRD_right_CI<-c(estimates_star_rightside$TE_SRD_right_bounds[1]-r*sL,estimates_star_rightside$TE_SRD_right_bounds[2]+r*sU)

        #CI for FRD under manipulation
        if(fuzzy){
          num_tau_pairs_rightside<-length(unlist(data.frame(estimates_by_TE_b[[1]][['estimates_rightside',i]]$TE_components_rightside)$TE_tau1s_upper))
          sL_t<-unlist(lapply(1:num_tau_pairs_rightside, function(t) sd(unlist(lapply(1:B, function(b) unlist(data.frame(estimates_by_TE_b[[b]][['estimates_rightside',i]]$TE_components_rightside)$TE_tau1s_lower)[t])), na.rm=TRUE)))
          sU_t<-unlist(lapply(1:num_tau_pairs_rightside, function(t) sd(unlist(lapply(1:B, function(b) unlist(data.frame(estimates_by_TE_b[[b]][['estimates_rightside',i]]$TE_components_rightside)$TE_tau1s_upper)[t])), na.rm=TRUE)))
          TE_lower_star_t<-unlist(data.frame(estimates_star_u[['estimates_rightside',i]]$TE_components_rightside)$TE_tau1s_lower)
          TE_upper_star_t<-unlist(data.frame(estimates_star_u[['estimates_rightside',i]]$TE_components_rightside)$TE_tau1s_upper)
          r_t<-unlist(lapply(1:num_tau_pairs_rightside, function(t) minimize_check(rfn, interval=c(qnorm(1-alpha), qnorm(1-alpha/2)),GammaU=TE_upper_star_t[t],GammaL=TE_lower_star_t[t],sU=sU_t[t],sL=sL_t[t])))
          lowerbounds_t<-TE_lower_star_t-r_t*sL_t
          upperbounds_t<-TE_upper_star_t+r_t*sU_t
          TE_FRD_right_CI<-c(lowerbounds_t[which.min(lowerbounds_t)], upperbounds_t[which.max(upperbounds_t)])

          if(refinements$refinement_C){
            #CI for SRD under manipulation, refinement C
            sL_C<-sd(unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_rightside',i]]$TE_FRD_right_bounds_refinementC[1])), na.rm=TRUE)
            sU_C<-sd(unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_rightside',i]]$TE_FRD_right_bounds_refinementC[2])), na.rm=TRUE)

            #It could be the case that refinement C was not computed, if it was deemed incompatible with the data, so catch this case:
            if(!is.na(sL_C)){
              r_C<-minimize_check(rfn, interval=c(qnorm(1-alpha), qnorm(1-alpha/2)),GammaU=estimates_star_rightside$TE_FRD_right_bounds_refinementC[2],GammaL=estimates_star_rightside$TE_FRD_right_bounds_refinementC[1],sU=sU_C,sL=sL_C)
              TE_FRD_right_CI_refinementC<-c(estimates_star_rightside$TE_FRD_right_bounds_refinementC[1]-r_C*sL_C, estimates_star_rightside$TE_FRD_right_bounds_refinementC[2]+r_C*sU_C)
            } else{
              TE_FRD_right_CI_refinementC<-c(NA,NA)
            }
          } else{
            TE_FRD_right_CI_refinementC<-c(NA,NA)
          }
        }
        else{
          TE_FRD_right_CI<-c(NA,NA)
          TE_FRD_right_CI_refinementC<-c(NA,NA)
        }
      }
      else{
        TE_SRD_right_CI<-c(NA,NA)
        TE_FRD_right_CI<-c(NA,NA)
        TE_FRD_right_CI_refinementC<-c(NA,NA)
      }

      #CI's for refinement based on covariates
      #------------------------------------------------------------------
      if(gotCovs){
        #CI for SRD under manipulation
        sL<-sd(unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_covs',i]]$TE_SRD_covs_bounds[1])), na.rm=TRUE)
        sU<-sd(unlist(lapply(1:B, function(b) estimates_by_TE_b[[b]][['estimates_covs',i]]$TE_SRD_covs_bounds[2])), na.rm=TRUE)
        GammaL<-estimates_star_u[['estimates_covs',i]]$TE_SRD_covs_bounds[1]
        GammaU<-estimates_star_u[['estimates_covs',i]]$TE_SRD_covs_bounds[2]
        r<-minimize_check(rfn, interval=c(qnorm(1-alpha), qnorm(1-alpha/2)),GammaU=GammaU,GammaL=GammaL,sU=sU,sL=sL)
        TE_SRD_covs_CI<-c(GammaL-r*sL,GammaU+r*sU)

        if(fuzzy){
          #CI for FRD under manipulation
          num_tau_pairs_covs<-length(unlist(data.frame(estimates_by_TE_b[[1]][['estimates_covs',i]]$TE_components_covariates)$TE_tau1s_upper))
          sL_t<-unlist(lapply(1:num_tau_pairs_covs, function(t) sd(unlist(lapply(1:B, function(b) unlist(data.frame(estimates_by_TE_b[[b]][['estimates_covs',i]]$TE_components_covariates)$TE_tau1s_lower[[t]]))), na.rm=TRUE)))
          sU_t<-unlist(lapply(1:num_tau_pairs_covs, function(t) sd(unlist(lapply(1:B, function(b) unlist(data.frame(estimates_by_TE_b[[b]][['estimates_covs',i]]$TE_components_covariates)$TE_tau1s_upper[[t]]))), na.rm=TRUE)))
          TE_lower_star_t<-unlist(data.frame(estimates_star_u[['estimates_covs',i]]$TE_components_covariates)$TE_tau1s_lower)
          TE_upper_star_t<-unlist(data.frame(estimates_star_u[['estimates_covs',i]]$TE_components_covariates)$TE_tau1s_upper)
          r_t<-unlist(lapply(1:num_tau_pairs_covs, function(t) minimize_check(rfn, interval=c(qnorm(1-alpha), qnorm(1-alpha/2)),GammaU=TE_upper_star_t[t],GammaL=TE_lower_star_t[t],sU=sU_t[t],sL=sL_t[t])))
          lowerbounds_t<- TE_lower_star_t-r_t*sL_t
          upperbounds_t<- TE_upper_star_t+r_t*sU_t
          TE_FRD_covs_CI<-c(lowerbounds_t[which.min(lowerbounds_t)], upperbounds_t[which.max(upperbounds_t)])
        }
        else{
          TE_FRD_covs_CI<-c(NA,NA)
        }

      }
      else{
        TE_SRD_covs_CI<-c(NA,NA)
        TE_FRD_covs_CI<-c(NA,NA)
      }

      #CI's DEMONSTRATING POTENTIAL IMPACT OF MANIPULATION
      #------------------------------------------------------------------
      if(!is.null(potential_taus)){

        #Sharp
        sL_tau<-unlist(lapply(1:length(potential_taus), function(tau) sd(unlist(lapply(1:B, function(b) estimates_tau_b[[tau]][[b]][['estimates_main',i]]$TE_lower_SRD)), na.rm=TRUE)))
        sU_tau<-unlist(lapply(1:length(potential_taus), function(tau) sd(unlist(lapply(1:B, function(b) estimates_tau_b[[tau]][[b]][['estimates_main',i]]$TE_upper_SRD)), na.rm=TRUE)))
        TE_lower_star_tau<-unlist(lapply(1:length(potential_taus), function(tau) estimates_tau_star[[tau]][['estimates_main',i]]$TE_lower_SRD))
        TE_upper_star_tau<-unlist(lapply(1:length(potential_taus), function(tau) estimates_tau_star[[tau]][['estimates_main',i]]$TE_upper_SRD))
        r_tau<-unlist(lapply(1:length(potential_taus), function(tau) minimize_check(rfn, interval=c(qnorm(1-alpha), qnorm(1-alpha/2)),GammaU=TE_upper_star_tau[tau],GammaL=TE_lower_star_tau[tau],sU=sU_tau[tau],sL=sU_tau[tau])))
        TE_SRD_CIs_manipulation_lower<-TE_lower_star_tau-r_tau*sL_tau
        TE_SRD_CIs_manipulation_upper<-TE_upper_star_tau+r_tau*sU_tau
        TE_SRD_CIs_manipulation<-cbind(potential_taus,TE_lower=TE_lower_star_tau, TE_upper=TE_upper_star_tau, TE_SRD_CIs_manipulation_lower,TE_SRD_CIs_manipulation_upper)

        #Fuzzy
        if(fuzzy){
          sL_tau_t<-lapply(1:length(potential_taus), function(tau) unlist(lapply(1:num_tau_pairs, function(t) sd(unlist(lapply(1:B, function(b) data.frame(estimates_tau_b[[tau]][[b]][['estimates_main',i]]$TE_components)$TE_tau1s_lower[t])), na.rm=TRUE))))
          sU_tau_t<-lapply(1:length(potential_taus), function(tau) unlist(lapply(1:num_tau_pairs, function(t) sd(unlist(lapply(1:B, function(b) data.frame(estimates_tau_b[[tau]][[b]][['estimates_main',i]]$TE_components)$TE_tau1s_upper[t])), na.rm=TRUE))))
          TE_lower_star_tau_t <- lapply(1:length(potential_taus), function(tau) data.frame(estimates_tau_star[[tau]][['estimates_main',i]]$TE_components)$TE_tau1s_lower)
          TE_upper_star_tau_t <-lapply(1:length(potential_taus), function(tau) data.frame(estimates_tau_star[[tau]][['estimates_main',i]]$TE_components)$TE_tau1s_upper)
          r_tau_t <- lapply(1:length(potential_taus), function(tau) unlist(lapply(1:num_tau_pairs, function(t) minimize_check(rfn, interval=c(qnorm(1-alpha),qnorm(1-alpha/2)),GammaU=TE_upper_star_tau_t[[tau]][t],GammaL=TE_lower_star_tau_t[[tau]][t],sU=sU_tau_t[[tau]][t],sL=sL_tau_t[[tau]][t]))))
          lowerbounds_tau_t<- lapply(1:length(potential_taus), function(tau) unlist(lapply(1:num_tau_pairs, function(t) TE_lower_star_tau_t[[tau]][t]-r_tau_t[[tau]][t]*sL_tau_t[[tau]][t])))
          upperbounds_tau_t<-lapply(1:length(potential_taus), function(tau) unlist(lapply(1:num_tau_pairs, function(t) TE_upper_star_tau_t[[tau]][t]+r_tau_t[[tau]][t]*sU_tau_t[[tau]][t])))
          TE_FRD_CIs_manipulation_lower<-unlist(lapply(1:length(potential_taus), function(tau) min(lowerbounds_tau_t[[tau]])))
          TE_FRD_CIs_manipulation_upper<-unlist(lapply(1:length(potential_taus), function(tau) max(upperbounds_tau_t[[tau]])))
          TE_lower_star_tau<-unlist(lapply(1:length(potential_taus), function(tau) estimates_tau_star[[tau]][['estimates_main',i]]$TE_lower_FRD))
          TE_upper_star_tau<-unlist(lapply(1:length(potential_taus), function(tau) estimates_tau_star[[tau]][['estimates_main',i]]$TE_upper_FRD))
          TE_FRD_CIs_manipulation<-cbind(potential_taus,TE_lower=TE_lower_star_tau, TE_upper=TE_upper_star_tau, TE_FRD_CIs_manipulation_lower, TE_FRD_CIs_manipulation_upper)
        }
        else{
          TE_FRD_CIs_manipulation<-c(NA,NA)
        }
      }
      else{
        TE_SRD_CIs_manipulation<-c(NA,NA)
        TE_FRD_CIs_manipulation<-c(NA,NA)
      }

      list(tau_hat_CI=tau_hat_CI, takeup_increase_CI=takeup_increase_CI, TE_SRD_naive_CI=TE_SRD_naive_CI, TE_SRD_CI=TE_SRD_CI, TE_SRD_covs_CI=TE_SRD_covs_CI, TE_FRD_naive_CI=TE_FRD_naive_CI, TE_FRD_CI=TE_FRD_CI, TE_FRD_refinementA_CI=TE_FRD_refinementA_CI, TE_FRD_refinementB_CI=TE_FRD_refinementB_CI, TE_SRD_CIs_manipulation=TE_SRD_CIs_manipulation, TE_FRD_CIs_manipulation=TE_FRD_CIs_manipulation, TE_FRD_covs_CI=TE_FRD_covs_CI, TE_SRD_right_CI=TE_SRD_right_CI, TE_FRD_right_CI=TE_FRD_right_CI, TE_FRD_right_CI_refinementC=TE_FRD_right_CI_refinementC)
    }, USE.NAMES=TRUE)

  #Don't send back CDFinputs$bootstraps if we didn't just estimate (this is to save memory)
  if(!saveinputs){CDFinputs<-NA}

  return(list(tau_hat_CI=tau_hat_CI,takeup_increase_CI=takeup_increase_CI, CIs_by_TE=CIs_by_TE, CDFinputs=CDFinputs, estimates=list(estimates_by_TE_b=estimates_by_TE_b, estimates_tau_b=estimates_tau_b, estimates_star_u=estimates_star_u, estimates_tau_star=estimates_tau_star)))
}

#GENERATE MATRIX OF POWERS OF RUNNING VARIABLE FOR LOCAL POLYNOMIAL REGRESSIONS
#------------------------------------------------------------------
generate_polynomial_matrix <- function(x, p){
  if(p>0){
    polynomial<-cbind(1,poly(x, degree=p, raw=TRUE, simple=TRUE))
  }
  else{
    polynomial = array(1,dim=c(length(x),1))
  }
  return(polynomial)
}

#INTEGRATE KERNEL FUNCTION FROM LEFT TO POINT X
#------------------------------------------------------------------
integrate_kernel <- function(kernel,x){
  if(kernel=="triangular"){
    if(x<=-1){return(0)}
    if(x>-1&x<1){return(x-1/2*x^2*sign(x)+0.5)}
    if(x>=1){return(1)}
  }
  else if(kernel=="rectangular"){
    if(x<=-1){return(0)}
    if(x>-1&x<1){return(1/2*x+0.5)}
    if(x>=1){return(1)}
  }
  else if(kernel=="gaussian"){
    return(pnorm(x))
  }
  else if(kernel=="epanechnikov"){
    if(x<=-1){return(0)}
    if(x>-1&x<1){return((-3*(x^3/3-x))/4+0.5)}
    if(x>=1){return(1)}
  }
  else{
    stop("Valid kernel not specified for integration.")
  }
}

#TAU-HAT FUNCTION FOR DISCONTINUITY IN DENSITY AT CUTOFF
#------------------------------------------------------------------
#' @export
compute_tau_tilde <- function(kernel,order,x,bwsx, discrete_x=FALSE){

  if(discrete_x){
    freqs<-data.frame(table(x))
    #Rather than rejoining freqs to the vector of observations (inefficient), we'll just weigh each x according to it's frequency in regressions (gives identical result quicker)
    fhat<-freqs$Freq
    little_xs<-as.numeric(as.vector(freqs$x))
    x<-little_xs
    Wtau<-Kh(bwsx[1],kernel,x)*fhat
    polynomial <- generate_polynomial_matrix(x,order)
    Fminus<-as.numeric(lm(fhat~polynomial, weights=Wtau, subset=(x<0))$coefficients["(Intercept)"])
    Fplus<-as.numeric(lm(fhat~polynomial, weights=Wtau, subset=(x>=0))$coefficients["(Intercept)"])
  }
  else{
    if(kernel=="triangular"){
      a0<-1/2;            b0<-1/2
      a1<-1/6;            b1<-(-1/6)
      a2<-1/12;           b2<-1/12
      a3<-1/20;           b3<-(-1/20)
      a4<-1/30;           b4<-1/30
    }
    else if(kernel=="rectangular"){
      a0<-1/2;            b0<-1/2
      a1<-1/4;            b1<-(-1/4)
      a2<-1/6;            b2<-1/6
      a3<-1/8;            b3<-(-1/8)
      a4<-1/10;           b4<-1/10
    }
    else if(kernel=="gaussian"){
      a0<-1/2;            b0<-1/2
      a1<-sqrt(1/(2*pi)); b1<-(-sqrt(1/(2*pi)))
      a2<-1/2;            b2<-1/2
      a3<-sqrt(2/pi);     b3<-(-sqrt(2/pi))
      a4<-3/2;            b4<-3/2
    }
    else if(kernel=="epanechnikov"){
      a0<-1/2;            b0<-1/2
      a1<-3/16;           b1<-(-3/16)
      a2<-1/10;           b2<-1/10
      a3<-1/16;           b3<-(-1/16)
      a4<-3/70;           b4<-3/70
    }
    else{
      stop("Valid kernel not specified.")
    }

    h<-bwsx[1]
    if(order==0){
      # user has requested constant
      Fplus<-mean((x>=0)/a0*Kh(h,kernel,x))
      Fminus<-mean((x<0)/b0*Kh(h,kernel,x))
    }
    else if(order==1){
      # user has requested linear
      det<-(a0*a2-a1^2)
      Fplus<-mean((x>=0)*1/det*c(a2,-a1)%*%t(cbind(1,(x/h)))*Kh(h,kernel,x))
      det<-(b0*b2-b1^2)
      Fminus<-mean((x<0)*1/det*c(b2,-b1)%*%t(cbind(1,(x/h)))*Kh(h,kernel,x))
    }
    else if(order==2){
      # user has requested quadratic
      det<-a0*(a2*a4-a3^2)-a1*(a1*a4-a2*a3)+a2*(a1*a3-a2^2)
      Fplus<-mean((x>=0)*1/det*c(a2*a4-a3^2,a2*a3-a1*a4,a1*a3-a2*a2)%*%t(cbind(1,(x/h),(x/h)^2))*Kh(h,kernel,x))
      det<-b0*(b2*b4-b3^2)-b1*(b1*b4-b2*b3)+b2*(b1*b3-b2^2)
      Fminus<-mean((x<0)*1/det*c(b2*b4-b3^2,b2*b3-b1*b4,b1*b3-b2*b2)%*%t(cbind(1,(x/h),(x/h)^2))*Kh(h,kernel,x))
    }
    else{
      stop("For estimation of discontinuity in density at cutoff (tau_hat) must be of quadratic order or lower.")
    }

  }

  tau_tilde<-1-Fminus/Fplus

  return(tau_tilde)
}

#KERNEL FUNCTIONS
#------------------------------------------------------------------
Kh <- function(h,kernel,x)
{
  return(1/h*K(kernel,x/h))
}

K <- function(kernel,x)
{
  if(kernel=="triangular"){
    return((1-abs(x))*(abs(x)<=1))
  }
  else if(kernel=="rectangular"){
    return(1/2*(abs(x)<=1))
  }
  else if(kernel=="gaussian"){
    return(1/sqrt(2*pi)*exp(-x^2/2))
  }
  else if(kernel=="epanechnikov"){
    return((1-abs(x)^2)*(abs(x)<=1))
  }
}

#GET CDF FROM PDF
#------------------------------------------------------------------
pdf2cdf<-function(little_ys, pdfy, integral_only=FALSE){
  n<-length(pdfy)
  #Setting the "width" of the first interval to be zero, so that the CDF evaluated at little_ys[1], the smallest value of little_ys, is zero
  #This effectively assumes that there is no mass to the left of the little_ys[1]
  interval_widths<-c(0,diff(little_ys))
  summand<-pdfy*interval_widths
  if(integral_only){
    return(sum(summand))
  }
  else{
    return(cumsum(summand))
  }
}

#GET CDF FROM PMF
#------------------------------------------------------------------
pmf2cdf<-function(pmfy, integral_only=FALSE){
  if(integral_only){
    return(sum(pmfy))
  }
  else{
    return(cumsum(pmfy))
  }
}

#EvALUATE EXPECTATION VALUE FROM A CDF
#------------------------------------------------------------------
cdf2expectation<-function(little_ys, cdfy){
  #Taking right Reimann sum to compute expectation, which is exact if y is discretely distributed. Treating cdfy[1] as the total mass at or to the left of little_ys[1].
  mass_in_interval<-c(cdfy[1],diff(cdfy))
  summand<-little_ys*mass_in_interval
  return(sum(summand))
}

#TRIM A CDF, ACCOUNTING FOR DISCRETENESS
#-----------------------------------------------------------------
trimLeft<-function(cdf, toTrim){
  if(toTrim<0|toTrim>=1){
    warning(paste0("Warning: trimming from the left of ", toTrim, " requested, returning NA"))
    return(array(NA,dim=length(cdf)))
  }
  else {
    d<-data.frame(y=1:length(cdf), F=cdf)
    #note: here y=1:length(cdf) is only being used as a strictly increasing index of y; we're not assuming that y takes values on the integers
    quantile=min(d$y[d$F>=toTrim])
    truncated_cdf<-(d$y>quantile)*(d$F-toTrim)+(d$y==quantile)*(d$F[d$y==quantile]-toTrim)
    trimmed_cdf<-truncated_cdf/(1-toTrim)
    return(trimmed_cdf)
  }
}

trimRight<-function(cdf, toTrim){
  if(toTrim<0|toTrim>=1){
    warning(paste0("Warning: trimming from the left of ", toTrim, " requested, returning NA"))
    return(array(NA,dim=length(cdf)))
  }
  else {
    d<-data.frame(y=1:length(cdf), F=cdf)
    quantile=min(d$y[d$F>=(1-toTrim)])
    truncated_cdf<-(d$y<quantile)*d$F+(d$y>=quantile)*(1-toTrim)
    trimmed_cdf<-truncated_cdf/(1-toTrim)
    return(trimmed_cdf)
  }
}

#TEST CONDITION ON TAU FOR FINITENESS OF BOUNDS
#------------------------------------------------------------------
check_taus<-function(num_bootstraps, kernel, orders, dist_cut, bwsx, discrete_x, treatment, W, iscov = NULL, fixed_tau=NULL){

  B<-num_bootstraps
  n<-length(dist_cut)
  resample_indices <- lapply(1:B, function(b) sample(1:n, replace = TRUE))

  #compute tau_hat and treatRight in full sample
  polynomial<-generate_polynomial_matrix(dist_cut,orders[2])
  treatRight<-as.numeric(lm(treatment~polynomial, weights=W, subset=(dist_cut>=0))$coefficients["(Intercept)"])
  if(is.null(fixed_tau)){
    tau_hat <- fixed_tau
  }
  else{
    tau_hat <- max(0,compute_tau_tilde(kernel=kernel, order=orders[1], x=dist_cut, bwsx=bwsx, discrete_x=discrete_x))
  }
  theta_hat<-tau_hat-(1-treatRight)

  #compute tau_hat and treatRight in bootstrap samples
  theta_b<-unlist(lapply(1:B, function(b){
    thisdist_cut<-dist_cut[unlist(resample_indices[b])]
    thistreatment<-treatment[unlist(resample_indices[b])]
    thisW<-W[unlist(resample_indices[b])]
    thispolynomial<-generate_polynomial_matrix(dist_cut[unlist(resample_indices[b])],orders[2])
    thistreatRight<-as.numeric(lm(thistreatment~thispolynomial, weights=thisW, subset=(thisdist_cut>=0))$coefficients["(Intercept)"])
    if(is.null(fixed_tau)){
      thistau_hat = max(0,compute_tau_tilde(kernel=kernel, order=orders[1], x=dist_cut[unlist(resample_indices[b])], bwsx=bwsx, discrete_x=discrete_x))
    }
    else{
      thistau_hat = fixed_tau
    }
    thistau_hat-(1-thistreatRight)
  }))

  #output warning if we can't reject theta_hat = 0 at 5% level using bootstrap variance of theta
  if(abs(treatRight)<1.96*sd(unlist(theta_b))){
    if(is.null(fixed_tau) & is.null(iscov)){
      warning("Note: the constraint that tau != P(D=0|c+) is close to binding for estimated tau. CI's may not have correct coverage.")
    } else if (!is.null(fixed_tau)){
      warning(paste0("Note: the constraint that tau != P(D=0|c+) is close to binding for fixed tau = ", fixed_tau,". CI's may not have correct coverage."))
    } else {
      warning(paste0("Note: the constraint that tau != P(D=0|c+) is close to binding for estimated tau when covs = ", iscov,". CI's may not have correct coverage."))
    }
  }
}

#PRINT A MESSAGE TO SCREEN AND/OR FILE
#------------------------------------------------------------------
print_message <- function(message, progressFile){
  print(paste0(Sys.time(), " ", message))
  if(!is.null(progressFile)){cat(strftime(Sys.time(),"%Y-%m-%d %H:%M:%S"), " ", message, "\n", file = progressFile, append = TRUE)}
}

#DISPLAY WARNING MESSAGE OR PRINT TO FILE IF RUNNING PARALELLIZED BOOTSTRAP (AND HENCE CANT WRITE TO SCREEN)
#------------------------------------------------------------------
warning_bs <- function(message, warningfile=NULL){
  warning(message)
  if(!is.null(warningfile)){cat(strftime(Sys.time(),"%Y-%m-%d %H:%M:%S"), " ", message, "\n", file = warningfile, append = TRUE)}
}

#------------------------------------------------------------------
#------------------------------------------------------------------
# HIGH LEVEL FUNCTIONS TO DISPLAY/EXPORT RESULTS AND GENERATE DATA
#------------------------------------------------------------------
#------------------------------------------------------------------

#' Generate a simulated dataset for testing estimation
#'
#' This function generates a simulated dataset with which to test \code{\link{rdbounds}}.
#' The x-values of potentially-assigned units (95% of sample) are normally distributed (variance of 5, censored at -10 and 10) around zero (the cutoff) and always-assigned units (5% of sample) follow a triangular distribution from the cutoff (x=0) to x=5. A random 5% of all units are never-takers, another 25% are always-takers, and the remaining units are compliers. Outcome values are then generated with a treatment effect of 2 for potentially-assigned units, 5 for always-assigned units, and following a linear trend with some normally distributed noise.
#'  Specifically: \deqn{y=(x+10)/2*treatment*(alwaysassigned=0)+5*treatment*(alwaysassigned=1)+normal(0,1))} and y is censored at 0 and 23.
#' @param sample_size Sample size for the dataset.
#' @param covs If set to \code{TRUE}, generates a sample in which half of the units have one of two covariate values, where the proportion of always-assigned units is slightly different for each.
#' @examples \donttest{df<-rdbounds_sampledata(50000);}
#' @export

#GENERATE SAMPLE DATA
#------------------------------------------------------------------
rdbounds_sampledata<- function(sample_size=50000, covs=FALSE){

  #If user asked for covs, generate covariate=0 and covariate=1 subsamples separately
  if(covs==TRUE){
    ss0 = floor(sample_size/2)
    ss1 = sample_size-ss0
    sample0<-generate_subsample(sample_size=ss0, covs="group 1")
    sample1<-generate_subsample(sample_size=ss1, covs="group 2")
    df<-as.data.frame(rbind(sample0$df, sample1$df))
    df$cov<-as.factor(df$cov)
    {print("---------------------------"); print("Full sample"); print("---------------------------")}
    n0 = sample0$n0+sample1$n0
  } else{
    sample<-generate_subsample(sample_size=sample_size, covs=FALSE)
    n0 = sample$n0
    df<-sample$df
  }

  n1 = sample_size-n0
  tau<-(.05/.95*2/5)/(dnorm(0, mean = 0, sd = sqrt(5), log = FALSE))
  print(paste0("True tau: ",tau))
  print("True treatment effect on potentially-assigned: 2")
  print(paste0("True treatment effect on right side of cutoff: ",(1-tau)*2+tau*5))

  return(df)

}

generate_subsample<- function(sample_size, covs){

  #x values for potentially-assigned, ~ N(0,5) censored at -10 and 10
  if(covs==FALSE){n0<-floor(sample_size*.95)}
  if(covs=="group 1"){n0<-floor(sample_size*.975)}
  if(covs=="group 2"){n0<-floor(sample_size*.925)}
  x0<-rnorm(n0)*sqrt(5)
  x0[x0<(-10)]<-(-10)
  x0[x0>10]<-10

  #x values for always-assigned ~ triangular decline from 0 to 5. f(x) = 2/5*(1-x/5); F(x) = 2/5x-x^2/25; F^{-1}(u) = 5*(1-sqrt(1-u))
  n1<-sample_size-n0
  u<-runif(n1)
  x1<-5*(1-sqrt(1-u))

  #combine
  x=c(x0,x1)
  n<-n0+n1
  if(covs=="group 1" || covs=="group 2"){cov<-array(covs,dim=n)}
  always_assigned<-c(array(0,dim=n0), array(1,dim=n1))

  #introduce taker groups
  temp_rand<-runif(n)
  always_taker<-(temp_rand<0.05)
  never_taker<-(temp_rand >= 0.05) & (temp_rand<.3)
  complier = 1-always_taker-never_taker
  treatment = (always_taker+(x>=0)*(complier))*(1-never_taker)

  #outcome variable. Treatment effect is 2 for potentially assigned and 5 for always assigned units
  y = pmin(23,pmax(0,(x+10)/2+2*treatment*(always_assigned==0)+5*treatment*(always_assigned==1)+rnorm(n)))

  if(covs=="group 1" | covs=="group 2"){
    if(covs=="group 1"){tau<-(.025/.975*2/5)/(dnorm(0, mean = 0, sd = sqrt(5), log = FALSE))}
    if(covs=="group 2"){tau<-(.075/.925*2/5)/(dnorm(0, mean = 0, sd = sqrt(5), log = FALSE))}

    if(covs=="group 1"){print("---------------------------"); print("Subsample: cov=group 1"); print("---------------------------")}
    if(covs=="group 2"){print("---------------------------"); print("Subsample: cov=group 2"); print("---------------------------")}
    print(paste0("True tau: ",tau))
    print("True treatment effect on potentially-assigned: 2")
    print(paste0("True treatment effect on right side of cutoff: ",(1-tau)*2+tau*5))

    return(list(n0=n0, df=data.frame(x,y,treatment,cov)))
  } else{
    return(list(n0=n0, df=data.frame(x,y,treatment)))
  }

}

#' Summarize Results from Manipulation Robust RD Estimation
#'
#' This function reports main estimands from \code{\link{rdbounds}}, as a formatted table, and optionally as text output.
#' @param rdbounds an rdbounds object resulting from the function rdbounds(). Required.
#' @param text if set to \code{TRUE}, display results as text as well as formatted table. Defaults to \code{TRUE}.
#' @param title_prefix Optional prefix before "Average Treatment EFfects" or "Quantile Treatment Effects" in table.
#' @examples \donttest{df<-rdbounds_sampledata(50000, covs=TRUE)
#' rdbounds_est<-rdbounds(y=df$y,x=df$x, covs=as.factor(df$cov), treatment=df$treatment, c=0,
#'                        discrete_x=FALSE, discrete_y=FALSE,
#'                        bwsx=c(.2,.5), bwy = .1, kernel="epanechnikov", orders=1,
#'                        evaluation_ys = seq(from = 0, to=23, by=.2),
#'                        refinement_A=TRUE, refinement_B=TRUE,
#'                        right_effects=TRUE, yextremes = c(0,23),
#'                        num_bootstraps=0)
#' rdbounds_summary(rdbounds_est, title_prefix="Sample Data Results")}
#' @export

#SUMMARY - DISPLAY SUMMARY OF RESULTS
#------------------------------------------------------------------
rdbounds_summary <- function(rdbounds, title_prefix="", text=TRUE) {

  if (!requireNamespace("formattable", quietly = TRUE)) {
    stop("package 'formattable' needed for rdbounds_summary to work. Please install it.",
         call. = FALSE)
  }

  print(paste0("Time taken: ", round(rdbounds$time_taken, digits=2), " minutes"))
  print(paste0("Sample size: ", rdbounds$sample_size))

  results_u<-sapply(1:length(rdbounds$TEs),
                    function(i)
                    {
                      u<-rdbounds$TEs[i]
                      if(u==-1){type_prefix<-"LATE"; TE_name<-"Average Treatment Effects"} else{type_prefix<-"QTE"; TE_name<-paste0((rdbounds$TEs[i]*100),"% Quantile Treatment Effects")}
                      e<-rdbounds$estimates[,i]
                      tabledf<-data.frame(rbind(
                        list(name="A. Basic Inputs",estimate=" ",CI=" "),
                        list("Share of always-assigned units", e$tau_hat, e$tau_hat_CI),
                        list("Increase in treatment take-up at the cutoff", e$takeup_increase, e$takeup_increase_CI),
                        list("B. ITT/SRD estimates"," "," "),
                        list("ITT-Ignoring Manipulation", e$TE_SRD_naive,e$TE_SRD_naive_CI),
                        list("ITT-Bounds on potentially-assigned", e$TE_SRD_bounds,e$TE_SRD_CI),
                        list("ITT-Bounds on potentially-assigned covariate refinement", e$TE_SRD_covs_bounds,e$TE_SRD_covs_CI),
                        list("ITT-Bounds on units just to right of cutoff", e$TE_SRD_right_bounds,e$TE_SRD_right_CI),
                        list("C. TE/FRD estimates"," "," "),
                        list(paste0(type_prefix,"-Ignoring Manipulation"), e$TE_FRD_naive,e$TE_FRD_naive_CI),
                        list(paste0(type_prefix,"-Bounds on potentially-assigned"), e$TE_FRD_bounds,e$TE_FRD_CI),
                        list(paste0(type_prefix,"-Bounds on potentially-assigned, refinement A"), e$TE_FRD_bounds_refinementA,e$TE_FRD_refinementA_CI),
                        list(paste0(type_prefix,"-Bounds on potentially-assigned, refinement B"), e$TE_FRD_bounds_refinementB,e$TE_FRD_refinementB_CI),
                        list(paste0(type_prefix,"-Bounds on potentially-assigned covariate refinement"), e$TE_FRD_covs_bounds,e$TE_FRD_covs_CI),
                        list(paste0(type_prefix,"-Bounds on units just to right of cutoff"), e$TE_FRD_right_bounds,e$TE_FRD_right_CI)
                        #list(paste0(type_prefix,"-Bounds just to right of cutoff, refinement C"), e$TE_FRD_right_bounds_refinementC, e$TE_FRD_right_CI_refinementC)
                      ))
                      if(nchar(title_prefix)>0){
                        colnames(tabledf)<-c(paste0("<i><div align='center' style='font-weight:normal'>",paste(title_prefix, TE_name, sep=": "),"</div></i>"), "Point Estimate", "CI")
                      } else{
                        colnames(tabledf)<-c(paste0("<i><div align='center' style='font-weight:normal'>",TE_name,"</div></i>"), "Point Estimate", "CI")
                      }
                      show(formattable(tabledf, list(area(row = c(1,4,9))~formatter("span", style=style("font-weight" = "bold", "text-decoration" = "underline")))))
                    })

  if(text){
    #The one for loop I'll allow! Prevents extraneous output that would come from lapply
    for(i in 1:length(rdbounds$TEs)){
      if(rdbounds$TEs[i]==-1){
        print("Local Average Treatment Effect:")
      }
      else(
        print(paste0((rdbounds$TEs[i]*100),"% Quantile Treatment Effect"))
      )
      print(rdbounds$estimates[,i])
    }
  }
}

#' Export Results from Manipulation Robust RD Estimation
#'
#' This function exports tables from manipulation robust RD estimation
#' @param rdbounds An rdbounds object resulting from \code{\link{rdbounds}}. Required.
#' @param file_name base filename to output tables to. Expects a string of the form "path/filename", where filename has no extension and will be the root filename for a series of different files containing different tables. If omitted no files will be produced.
#' @param view_it Boolean. View main results table in Rstudio viewer. Defaults to \code{FALSE}.
#' @examples \donttest{df<-rdbounds_sampledata(50000, covs=TRUE)
#' rdbounds_est<-rdbounds(y=df$y,x=df$x, covs=as.factor(df$cov), treatment=df$treatment, c=0,
#'                        discrete_x=FALSE, discrete_y=FALSE,
#'                        bwsx=c(.2,.5), bwy = .1, kernel="epanechnikov", orders=1,
#'                        evaluation_ys = seq(from = 0, to=23, by=.2),
#'                        refinement_A=TRUE, refinement_B=TRUE,
#'                        right_effects=TRUE, yextremes = c(0,23),
#'                        num_bootstraps=0)
#' rdbounds_summary(rdbounds_est, title_prefix="Sample Data Results")}
#' @export

#EXPORT - EXPORT TABLES OF RESULTS AND INTERMEDIATE QUANTITIES
#------------------------------------------------------------------
rdbounds_export<- function(rdbounds, file_name=NULL, view_it=FALSE) {

  estimates<-rdbounds$estimates_raw
  bs_estimates<-rdbounds$bs_estimates

  if(length(rdbounds$bs_estimates)==1 && is.na(rdbounds$bs_estimates)){   #using && here so the is.na is not evaluated if length() > 0 (which is this case if bootstrap is run), to avoid warning message
    B<-0
  } else{
    B<-length(rdbounds$bs_estimates$estimates_b)
  }

  length_taus<-length(rdbounds$potential_taus)

  for(u in 1:length(rdbounds$TEs)){
    main<-data.frame(
      b=0,
      tau_fixed=0,
      tau_hat=estimates[['prelims',u]]$tau_hat, treatLeft=estimates[['prelims',u]]$treatLeft, treatRight=estimates[['prelims',u]]$treatRight,
      naive_SRD_Y1=estimates[['estimates_naive',u]]$TE_outcomes$Y1_SRD, naive_SRD_Y0=estimates[['estimates_naive',u]]$TE_outcomes$Y0_SRD,
      naive_FRD_Y1=estimates[['estimates_naive',u]]$TE_outcomes$Y1_FRD, naive_FRD_Y0=estimates[['estimates_naive',u]]$TE_outcomes$Y0_FRD,
      SRD_Y1_lower=estimates[['estimates_main',u]]$TE_outcomes$sharp$Y1_lower, SRD_Y1_upper=estimates[['estimates_main',u]]$TE_outcomes$sharp$Y1_upper,
      SRD_Y0_lower=estimates[['estimates_main',u]]$TE_outcomes$sharp$Y0_lower, SRD_Y0_upper=estimates[['estimates_main',u]]$TE_outcomes$sharp$Y0_upper,
      FRD_Y1_lower=estimates[['estimates_main',u]]$TE_outcomes$fuzzy$Y1_lower, FRD_Y1_upper=estimates[['estimates_main',u]]$TE_outcomes$fuzzy$Y1_upper,
      FRD_Y0_lower=estimates[['estimates_main',u]]$TE_outcomes$fuzzy$Y0_lower, FRD_Y0_upper=estimates[['estimates_main',u]]$TE_outcomes$fuzzy$Y0_upper,
      FRD_refA_Y1_lower=estimates[['estimates_main',u]]$TE_outcomes$refinementA$Y1_lower, FRD_refA_Y1_upper=estimates[['estimates_main',u]]$TE_outcomes$refinementA$Y1_upper,
      FRD_refA_Y0_lower=estimates[['estimates_main',u]]$TE_outcomes$refinementA$Y0_lower, FRD_refA_Y0_upper=estimates[['estimates_main',u]]$TE_outcomes$refinementA$Y0_upper,
      FRD_refB_Y1_lower=estimates[['estimates_main',u]]$TE_outcomes$refinementB$Y1_lower, FRD_refB_Y1_upper=estimates[['estimates_main',u]]$TE_outcomes$refinementB$Y1_upper,
      FRD_refB_Y0_lower=estimates[['estimates_main',u]]$TE_outcomes$refinementB$Y0_lower, FRD_refB_Y0_upper=estimates[['estimates_main',u]]$TE_outcomes$refinementB$Y0_upper,
      SRD_plus_Y1_lower=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$sharp$Y1_lower, SRD_plus_Y1_upper=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$sharp$Y1_upper,
      SRD_plus_Y0_lower=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$sharp$Y0_lower, SRD_plus_Y0_upper=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$sharp$Y0_upper,
      FRD_plus_Y1_lower=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$fuzzy$Y1_lower, FRD_plus_Y1_upper=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$fuzzy$Y1_upper,
      FRD_plus_Y0_lower=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$fuzzy$Y0_lower, FRD_plus_Y0_upper=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$fuzzy$Y0_upper,
      #FRD_plus_refC_Y1_lower=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$refinementC$Y1_lower, FRD_plus_refC_Y1_upper=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$refinementC$Y1_upper,
      #FRD_plus_refC_Y0_lower=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$refinementC$Y0_lower, FRD_plus_refC_Y0_upper=estimates[['estimates_rightside', u]]$TE_outcomes_rightside$refinementC$Y0_upper,
      SRD_covs_Y1_lower=estimates[['estimates_covs', u]]$TE_outcomes_covariates$sharp$Y1_lower, SRD_covs_Y1_upper=estimates[['estimates_covs', u]]$TE_outcomes_covariates$sharp$Y1_upper,
      SRD_covs_Y0_lower=estimates[['estimates_covs', u]]$TE_outcomes_covariates$sharp$Y0_lower, SRD_covs_Y0_upper=estimates[['estimates_covs', u]]$TE_outcomes_covariates$sharp$Y0_upper,
      FRD_covs_Y1_lower=estimates[['estimates_covs', u]]$TE_outcomes_covariates$fuzzy$Y1_lower, FRD_covs_Y1_upper=estimates[['estimates_covs', u]]$TE_outcomes_covariates$fuzzy$Y1_upper,
      FRD_covs_Y0_lower=estimates[['estimates_covs', u]]$TE_outcomes_covariates$fuzzy$Y0_lower, FRD_covs_Y0_upper=estimates[['estimates_covs', u]]$TE_outcomes_covariates$fuzzy$Y0_upper
    )

    if(B>0){
      main<-rbind(main, data.frame(
        rbindlist(
          lapply(
            1:B,
            function(b)
            {
              list(
                b=b,
                tau_fixed=0,
                tau_hat=bs_estimates$estimates_b[[b]][['prelims',u]]$tau_hat, treatLeft=bs_estimates$estimates_b[[b]][['prelims',u]]$treatLeft, treatRight=bs_estimates$estimates_b[[b]][['prelims',u]]$treatRight,
                naive_SRD_Y1=bs_estimates$estimates_b[[b]][['estimates_naive',u]]$TE_outcomes$Y1_SRD, naive_SRD_Y0=bs_estimates$estimates_b[[b]][['estimates_naive',u]]$TE_outcomes$Y0_SRD,
                naive_FRD_Y1=bs_estimates$estimates_b[[b]][['estimates_naive',u]]$TE_outcomes$Y1_FRD, naive_FRD_Y0=bs_estimates$estimates_b[[b]][['estimates_naive',u]]$TE_outcomes$Y0_FRD,
                SRD_Y1_lower=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$sharp$Y1_lower, SRD_Y1_upper=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$sharp$Y1_upper,
                SRD_Y0_lower=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$sharp$Y0_lower, SRD_Y0_upper=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$sharp$Y0_upper,
                FRD_Y1_lower=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$fuzzy$Y1_lower, FRD_Y1_upper=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$fuzzy$Y1_upper,
                FRD_Y0_lower=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$fuzzy$Y0_lower, FRD_Y0_upper=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$fuzzy$Y0_upper,
                FRD_refA_Y1_lower=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$refinementA$Y1_lower, FRD_refA_Y1_upper=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$refinementA$Y1_upper,
                FRD_refA_Y0_lower=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$refinementA$Y0_lower, FRD_refA_Y0_upper=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$refinementA$Y0_upper,
                FRD_refB_Y1_lower=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$refinementB$Y1_lower, FRD_refB_Y1_upper=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$refinementB$Y1_upper,
                FRD_refB_Y0_lower=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$refinementB$Y0_lower, FRD_refB_Y0_upper=bs_estimates$estimates_b[[b]][['estimates_main',u]]$TE_outcomes$refinementB$Y0_upper,
                SRD_plus_Y1_lower=bs_estimates$estimates_b[[b]][['estimates_rightside', u]]$TE_outcomes_rightside$sharp$Y1_lower, SRD_plus_Y1_upper=bs_estimates$estimates_b[[b]][['estimates_rightside', u]]$TE_outcomes_rightside$sharp$Y1_upper,
                SRD_plus_Y0_lower=bs_estimates$estimates_b[[b]][['estimates_rightside', u]]$TE_outcomes_rightside$sharp$Y0_lower, SRD_plus_Y0_upper=bs_estimates$estimates_b[[b]][['estimates_rightside', u]]$TE_outcomes_rightside$sharp$Y0_upper,
                FRD_plus_Y1_lower=bs_estimates$estimates_b[[b]][['estimates_rightside', u]]$TE_outcomes_rightside$fuzzy$Y1_lower, FRD_plus_Y1_upper=bs_estimates$estimates_b[[b]][['estimates_rightside', u]]$TE_outcomes_rightside$fuzzy$Y1_upper,
                FRD_plus_Y0_lower=bs_estimates$estimates_b[[b]][['estimates_rightside', u]]$TE_outcomes_rightside$fuzzy$Y0_lower, FRD_plus_Y0_upper=bs_estimates$estimates_b[[b]][['estimates_rightside', u]]$TE_outcomes_rightside$fuzzy$Y0_upper,
                #FRD_plus_refC_Y1_lower=bs_estimates$estimates_b[[b]][['estimates_rightside',u]]$TE_outcomes_rightside$refinementC$Y1_lower, FRD_plus_refC_Y1_upper=bs_estimates$estimates_b[[b]][['estimates_rightside',u]]$TE_outcomes_rightside$refinementC$Y1_upper,
                #FRD_plus_refC_Y0_lower=bs_estimates$estimates_b[[b]][['estimates_rightside',u]]$TE_outcomes_rightside$refinementC$Y0_lower, FRD_plus_refC_Y0_upper=bs_estimates$estimates_b[[b]][['estimates_rightside',u]]$TE_outcomes_rightside$refinementC$Y0_upper,
                SRD_covs_Y1_lower=bs_estimates$estimates_b[[b]][['estimates_covs', u]]$TE_outcomes_covariates$sharp$Y1_lower, SRD_covs_Y1_upper=bs_estimates$estimates_b[[b]][['estimates_covs', u]]$TE_outcomes_covariates$sharp$Y1_upper,
                SRD_covs_Y0_lower=bs_estimates$estimates_b[[b]][['estimates_covs', u]]$TE_outcomes_covariates$sharp$Y0_lower, SRD_covs_Y0_upper=bs_estimates$estimates_b[[b]][['estimates_covs', u]]$TE_outcomes_covariates$sharp$Y0_upper,
                FRD_covs_Y1_lower=bs_estimates$estimates_b[[b]][['estimates_covs', u]]$TE_outcomes_covariates$fuzzy$Y1_lower, FRD_covs_Y1_upper=bs_estimates$estimates_b[[b]][['estimates_covs', u]]$TE_outcomes_covariates$fuzzy$Y1_upper,
                FRD_covs_Y0_lower=bs_estimates$estimates_b[[b]][['estimates_covs', u]]$TE_outcomes_covariates$fuzzy$Y0_lower, FRD_covs_Y0_upper=bs_estimates$estimates_b[[b]][['estimates_covs', u]]$TE_outcomes_covariates$fuzzy$Y0_upper
              )
            }
          ))))

      if(length_taus>0){

        main<-rbind(main, data.frame(
          rbindlist(
            lapply(
              1:length_taus,
              function(tau)
              {
                rbind(list(
                  b=0,
                  tau_fixed=1,
                  tau_hat=bs_estimates$estimates_tau_star[[tau]][['prelims',u]]$tau_hat, treatLeft=bs_estimates$estimates_tau_star[[tau]][['prelims',u]]$treatLeft, treatRight=bs_estimates$estimates_tau_star[[tau]][['prelims',u]]$treatRight,
                  naive_SRD_Y1=bs_estimates$estimates_tau_star[[tau]][['estimates_naive',u]]$TE_outcomes$Y1_SRD, naive_SRD_Y0=bs_estimates$estimates_tau_star[[tau]][['estimates_naive',u]]$TE_outcomes$Y0_SRD,
                  naive_FRD_Y1=bs_estimates$estimates_tau_star[[tau]][['estimates_naive',u]]$TE_outcomes$Y1_FRD, naive_FRD_Y0=bs_estimates$estimates_tau_star[[tau]][['estimates_naive',u]]$TE_outcomes$Y0_FRD,
                  SRD_Y1_lower=bs_estimates$estimates_tau_star[[tau]][['estimates_main',u]]$TE_outcomes$sharp$Y1_lower, SRD_Y1_upper=bs_estimates$estimates_tau_star[[tau]][['estimates_main',u]]$TE_outcomes$sharp$Y1_upper,
                  SRD_Y0_lower=bs_estimates$estimates_tau_star[[tau]][['estimates_main',u]]$TE_outcomes$sharp$Y0_lower, SRD_Y0_upper=bs_estimates$estimates_tau_star[[tau]][['estimates_main',u]]$TE_outcomes$sharp$Y0_upper,
                  FRD_Y1_lower=bs_estimates$estimates_tau_star[[tau]][['estimates_main',u]]$TE_outcomes$fuzzy$Y1_lower, FRD_Y1_upper=bs_estimates$estimates_tau_star[[tau]][['estimates_main',u]]$TE_outcomes$fuzzy$Y1_upper,
                  FRD_Y0_lower=bs_estimates$estimates_tau_star[[tau]][['estimates_main',u]]$TE_outcomes$fuzzy$Y0_lower, FRD_Y0_upper=bs_estimates$estimates_tau_star[[tau]][['estimates_main',u]]$TE_outcomes$fuzzy$Y0_upper,
                  FRD_refA_Y1_lower=NA, FRD_refA_Y1_upper=NA,
                  FRD_refA_Y0_lower=NA, FRD_refA_Y0_upper=NA,
                  FRD_refB_Y1_lower=NA, FRD_refB_Y1_upper=NA,
                  FRD_refB_Y0_lower=NA, FRD_refB_Y0_upper=NA,
                  SRD_plus_Y1_lower=NA, SRD_plus_Y1_upper=NA,
                  SRD_plus_Y0_lower=NA, SRD_plus_Y0_upper=NA,
                  FRD_plus_Y1_lower=NA, FRD_plus_Y1_upper=NA,
                  FRD_plus_Y0_lower=NA, FRD_plus_Y0_upper=NA,
                  #FRD_plus_refC_Y1_lower=NA, FRD_plus_refC_Y1_upper=NA,
                  #FRD_plus_refC_Y0_lower=NA, FRD_plus_refC_Y0_upper=NA,
                  SRD_covs_Y1_lower=NA, SRD_covs_Y1_upper=NA,
                  SRD_covs_Y0_lower=NA, SRD_covs_Y0_upper=NA,
                  FRD_covs_Y1_lower=NA, FRD_covs_Y1_upper=NA,
                  FRD_covs_Y0_lower=NA, FRD_covs_Y0_upper=NA
                ),
                rbindlist(
                  lapply(1:B, function(b) {
                    list(
                      b=b,
                      tau_fixed=1,
                      tau_hat=bs_estimates$estimates_tau_b[[tau]][[b]][['prelims',u]]$tau_hat, treatLeft=bs_estimates$estimates_tau_b[[tau]][[b]][['prelims',u]]$treatLeft, treatRight=bs_estimates$estimates_tau_b[[tau]][[b]][['prelims',u]]$treatRight,
                      naive_SRD_Y1=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_naive',u]]$TE_outcomes$Y1_SRD, naive_SRD_Y0=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_naive',u]]$TE_outcomes$Y0_SRD,
                      naive_FRD_Y1=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_naive',u]]$TE_outcomes$Y1_FRD, naive_FRD_Y0=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_naive',u]]$TE_outcomes$Y0_FRD,
                      SRD_Y1_lower=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_main',u]]$TE_outcomes$sharp$Y1_lower, SRD_Y1_upper=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_main',u]]$TE_outcomes$sharp$Y1_upper,
                      SRD_Y0_lower=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_main',u]]$TE_outcomes$sharp$Y0_lower, SRD_Y0_upper=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_main',u]]$TE_outcomes$sharp$Y0_upper,
                      FRD_Y1_lower=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_main',u]]$TE_outcomes$fuzzy$Y1_lower, FRD_Y1_upper=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_main',u]]$TE_outcomes$fuzzy$Y1_upper,
                      FRD_Y0_lower=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_main',u]]$TE_outcomes$fuzzy$Y0_lower, FRD_Y0_upper=bs_estimates$estimates_tau_b[[tau]][[b]][['estimates_main',u]]$TE_outcomes$fuzzy$Y0_upper,
                      FRD_refA_Y1_lower=NA, FRD_refA_Y1_upper=NA,
                      FRD_refA_Y0_lower=NA, FRD_refA_Y0_upper=NA,
                      FRD_refB_Y1_lower=NA, FRD_refB_Y1_upper=NA,
                      FRD_refB_Y0_lower=NA, FRD_refB_Y0_upper=NA,
                      SRD_plus_Y1_lower=NA, SRD_plus_Y1_upper=NA,
                      SRD_plus_Y0_lower=NA, SRD_plus_Y0_upper=NA,
                      FRD_plus_Y1_lower=NA, FRD_plus_Y1_upper=NA,
                      FRD_plus_Y0_lower=NA, FRD_plus_Y0_upper=NA,
                      #FRD_plus_refC_Y1_lower=NA, FRD_plus_refC_Y1_upper=NA,
                      #FRD_plus_refC_Y0_lower=NA, FRD_plus_refC_Y0_upper=NA,
                      SRD_covs_Y1_lower=NA, SRD_covs_Y1_upper=NA,
                      SRD_covs_Y0_lower=NA, SRD_covs_Y0_upper=NA,
                      FRD_covs_Y1_lower=NA, FRD_covs_Y1_upper=NA,
                      FRD_covs_Y0_lower=NA, FRD_covs_Y0_upper=NA
                    )
                  }
                  )))
              }
            ))))
      }
    }

    if(!is.null(file_name)){
      
      if(rdbounds$TEs[[u]]==-1){TE_name<-"ATE"} else{TE_name<-paste0("QTE",as.integer(rdbounds$TEs[[u]]*100))}

      e<-rdbounds$estimates[,u]
      resultsdf<-data.frame(rbind(
        list(name="A. Basic Inputs",estimate_lower=" ",estimate_upper=" ",CI_lower=" ",CI_upper=" "),
        list("Share of always-assigned units", e$tau_hat, " ", e$tau_hat_CI[1], e$tau_hat_CI[2]),
        list("Increase in treatment take-up at the cutoff", e$takeup_increase, " ", e$takeup_increase_CI[1], e$takeup_increase_CI[2]),
        list("B. ITT/SRD estimates"," "," "," "," "),
        list("ITT-Ignoring Manipulation", e$TE_SRD_naive," ",e$TE_SRD_naive_CI[1],e$TE_SRD_naive_CI[2]),
        list("ITT-Bounds on potentially-assigned", e$TE_SRD_bounds[1], e$TE_SRD_bounds[2],e$TE_SRD_CI[1],e$TE_SRD_CI[2]),
        #list("ITT-Bounds on potentially-assigned covariate refinement", e$TE_SRD_covs_bounds[1],e$TE_SRD_covs_bounds[2],e$TE_SRD_covs_CI[1],e$TE_SRD_covs_CI[2]),
        #list("ITT-Bounds on units just to right of cutoff", e$TE_SRD_right_bounds[1],e$TE_SRD_right_bounds[2],e$TE_SRD_right_CI[1],e$TE_SRD_right_CI[2]),
        list("C. TE/FRD estimates"," "," ", " ", " "),
        list(paste0(TE_name,"-Ignoring Manipulation"), e$TE_FRD_naive," ",e$TE_FRD_naive_CI[1],e$TE_FRD_naive_CI[2]),
        list(paste0(TE_name,"-Bounds on potentially-assigned"), e$TE_FRD_bounds[1],e$TE_FRD_bounds[2],e$TE_FRD_CI[1],e$TE_FRD_CI[2])#,
        #list(paste0(TE_name,"-Bounds on potentially-assigned refinement A"), e$TE_FRD_bounds_refinementA[1], e$TE_FRD_bounds_refinementA[2],e$TE_FRD_refinementA_CI[1], e$TE_FRD_refinementA_CI[2]),
        #list(paste0(TE_name,"-Bounds on potentially-assigned refinement B"), e$TE_FRD_bounds_refinementB[1], e$TE_FRD_bounds_refinementB[2],e$TE_FRD_refinementB_CI[1],e$TE_FRD_refinementB_CI[2]),
        #list(paste0(TE_name,"-Bounds on potentially-assigned covariate refinement"), e$TE_FRD_covs_bounds[1], e$TE_FRD_covs_bounds[2],e$TE_FRD_covs_CI[1], e$TE_FRD_covs_CI[2]),
        #list(paste0(TE_name,"-Bounds on units just to right of cutoff"), e$TE_FRD_right_bounds[1], e$TE_FRD_right_bounds[2],e$TE_FRD_right_CI[1],e$TE_FRD_right_CI[2])
        #list(paste0(TE_name,"-Bounds just to right of cutoff refinement C"), e$TE_FRD_right_bounds_refinementC[1], e$TE_FRD_right_bounds_refinementC[2], e$TE_FRD_right_CI_refinementC[1], e$TE_FRD_right_CI_refinementC[2])
      ))

      write.csv(as.matrix(resultsdf), quote=FALSE, row.names=F, paste0(file_name, "_results_",TE_name,".csv"))

      if(length_taus>0 & B>0){
        #write.csv(as.matrix(e$TE_SRD_CIs_manipulation), quote=FALSE, row.names=F, paste0(file_name, "_fixedtau_sharp_",TE_name,".csv"))
        write.csv(as.matrix(e$TE_FRD_CIs_manipulation), quote=FALSE, row.names=F, paste0(file_name, "_fixedtau_fuzzy_",TE_name,".csv"))
      }

    }

  }

  if(view_it){View(resultsdf)}
}

