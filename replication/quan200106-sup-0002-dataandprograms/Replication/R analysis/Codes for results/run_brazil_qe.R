library(data.table)
library(formattable)

set.seed(519)

rootPath<-"C:/Users/Len/Dropbox/Research/__Existing/RDManipulation/Final results for paper/feb 2020"

pathData<-paste0(rootPath,'/data_grr_QE_reg.csv')
pathOutput<-paste0(rootPath,"/results/")

#Load the rdbounds function
debugSource(paste0(rootPath,'/rdboundsv1.01_qe.R'), echo=TRUE)

#Create sink log file
sink(paste0(pathOutput,"rdbounds_r_output.txt"), append=FALSE, split=TRUE)

Sys.getenv("R_HISTSIZE")
Sys.setenv(R_HISTSIZE = 10000)

#DATA PREP
#-----------------------------------------------------------------
  dat<-read.csv(pathData)
  dat$days_without_formal_job24<-dat$days_without_formal_job
  dat$days_without_formal_job24[dat$days_without_formal_job24>=731]<-731
  dat$days_without_formal_job6<-dat$days_without_formal_job24
  dat$days_without_formal_job6[dat$days_without_formal_job24>=(365.5/2)]<-(365.5/2)
  
  #Code missing Y observations as censoring point
  dat$days_without_formal_job24[is.na(dat$days_without_formal_job24)]=731
  dat$days_without_formal_job6[is.na(dat$days_without_formal_job6)]=365.5/2

  #Now no missing observations to worry about
  print(sum(is.na(dat$days_without_formal_job6))+sum(is.na(dat$days_without_formal_job24))+sum(is.na(dat$dist_cut))+sum(is.na(dat$treatment)))

#TREATMENT EFFECT ESTIMATION
#-----------------------------------------------------------------
  discrete_y<-TRUE
  bwy<-50
  parallelize<-TRUE
  num_bootstraps<-500
  num_tau_pairs<-50
  
  potential_taus<-c(.025,.05, 0.06369131,.1,.2)
  
  progressFile<-paste0(pathOutput, "progress_file.txt")
  cat(strftime(Sys.time(),"%Y-%m-%d %H:%M:%S"), "  Starting estimation \n", file=progressFile, append=FALSE)
  
  print("Outcome censored at 6 months:")
  print("------------------------------------------------")
  rdbounds_est6<-rdbounds(y=dat$days_without_formal_job6,x=dat$dist_cut, c=0, treatment=dat$treatment, bwsx=30, discrete_y=discrete_y, bwy=bwy,kernel="triangular",discrete_x=TRUE, num_tau_pairs=num_tau_pairs, orders=c(1,1), num_bootstraps=c(num_bootstraps), type="ate", potential_taus=potential_taus, right_effects=FALSE, yextremes=c(0,182.75), parallelize=parallelize, progressFile=progressFile)
  rdbounds_export(rdbounds_est6,paste0(pathOutput,"rdbounds6"), view_it=FALSE)
  
  print("Outcome censored at 24 months:")
  print("------------------------------------------------")
  rdbounds_est24<-rdbounds(y=dat$days_without_formal_job24,x=dat$dist_cut, c=0, treatment=dat$treatment, bwsx=30, discrete_y=discrete_y, bwy=bwy,kernel="triangular",discrete_x=TRUE, num_tau_pairs=num_tau_pairs, orders=c(1,1), num_bootstraps=c(num_bootstraps), type="qte", percentiles=c(-1,.5), potential_taus=potential_taus, right_effects=FALSE, yextremes=c(0,731), parallelize=parallelize, progressFile=progressFile)
  rdbounds_export(rdbounds_est24,paste0(pathOutput,"rdbounds24"), view_it=FALSE)

#CHARACTERISTICS OF POTENTIALLY AND ALWAYS ASSIGNED UNITS
#-----------------------------------------------------------------
  
W<-Kh(30,"triangular",dist_cut)  
dist_cut=dat$dist_cut
  
get_characteristics<-function(characteristics, kernel, order, h, discrete_x, dist_cut, dat){
  tau_hat<-compute_tau_tilde(x=dist_cut, kernel, order, h, discrete_x)
  return(as.data.frame(rbindlist(lapply(characteristics, function(var){
    w<-dat[var][,1]
    if(is.factor(w)){
      stop("Function output_characteristics Cannot accept factor variables")
    }


    Eleft<-as.numeric(lm(w~dist_cut, weights=W, subset=(dist_cut<0))$coefficients["(Intercept)"])
    Eright<-as.numeric(lm(w~dist_cut, weights=W, subset=(dist_cut>=0))$coefficients["(Intercept)"])
    diff<-Eright-Eleft
    potentially_mean<-Eleft
    always_mean<-1/tau_hat*(diff)+Eleft
    list(variable=var, diff=diff, potentially_assigned=potentially_mean, always_assigned=always_mean)

  }))))

}
  
dat$high_school<-dat$educ>=11
dat$prime_age<-dat$age>=36
characteristics<-c("male","age","educ","tenure","logwage","rrate","commerce","construction","industry","services","microfirm")
char_means_point<-get_characteristics(characteristics=characteristics,kernel="triangular", order=1, h=30, discrete_x=TRUE, dist_cut=dat$dist_cut, dat=dat)

B<-num_bootstraps
alpha<-.05
r<-qnorm(1-alpha/2)
n<-length(dist_cut)
resample_indices <- lapply(1:B, function(b) sample(1:n, replace = T))

char_means_bs<-lapply(1:B, function(b) get_characteristics(characteristics=characteristics,kernel="triangular", order=1, h=30, discrete_x=TRUE, dist_cut=dat$dist_cut[resample_indices[[b]]], dat=dat[resample_indices[[b]],]))
sd_diff_w<-unlist(lapply(characteristics, function(var){sd(unlist(lapply(1:B, function(b) char_means_bs[[b]]$diff[char_means_bs[[b]]$variable==var])))}))
sd_potentially_w<-unlist(lapply(characteristics, function(var){sd(unlist(lapply(1:B, function(b) char_means_bs[[b]]$potentially_assigned[char_means_bs[[b]]$variable==var])))}))
sd_always_w<-unlist(lapply(characteristics, function(var){sd(unlist(lapply(1:B, function(b) char_means_bs[[b]]$always_assigned[char_means_bs[[b]]$variable==var])))}))

char_means<-char_means_point
char_means$diff_lowerCI<-char_means$diff-r*sd_diff_w
char_means$diff_upperCI<-char_means$diff+r*sd_diff_w
char_means$potentially_lowerCI<-char_means$potentially_assigned-r*sd_potentially_w
char_means$potentially_upperCI<-char_means$potentially_assigned+r*sd_potentially_w
char_means$always_lowerCI<-char_means$always_assigned-r*sd_always_w
char_means$always_upperCI<-char_means$always_assigned+r*sd_always_w

write.csv(as.matrix(char_means), quote=FALSE, row.names=F, paste0(pathOutput,"mean_characteristics.csv"))

sink()

#Make sure all connections are closed:
sink.reset <- function(){
  for(i in seq_len(sink.number())){
    sink(NULL)
  }
}
sink.reset()

