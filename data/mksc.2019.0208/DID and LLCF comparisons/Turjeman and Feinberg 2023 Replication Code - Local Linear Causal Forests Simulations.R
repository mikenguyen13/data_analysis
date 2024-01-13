# simulation code - based on DID and testing with causal forests
# October 17, 2018, updated on June 28, 2023


library(foreach)
library(grf)
library(dplyr)
library(tidyverse)
library(lfe)
library(ggplot2)
set.seed(123)
library(here)
setwd(here())
getwd()
library(foreach)
combine_null = function(a, b) {}


N = 1000
TT = 8
RERUN_MODELS = FALSE
if(RERUN_MODELS){
  causal_forests_w_counterfactual = function(X, Y, W, t = 0){
    forest.W = grf::ll_regression_forest(X = X, Y = W, #compute.oob.predictions = TRUE, 
                                         tune.parameters = "all")
    predictions_w = predict(forest.W)$predictions
    forest.Y = grf::ll_regression_forest(X = X, Y = Y, #compute.oob.predictions = TRUE, 
                                         tune.parameters = "all")
    predictions_y = predict(forest.Y)$predictions
    cf = grf::causal_forest(X = X, Y = Y, W = W, tune.parameters = "all",
                            W.hat = predictions_w,
                            Y.hat = predictions_y, compute.oob.predictions = TRUE,
                            honesty = TRUE)
    predictions_cf = cf$predictions
    mu.hat.1 = predictions_y + (1 - predictions_w) * predictions_cf
    mu.hat.0 = predictions_y - predictions_w * predictions_cf
    results = data.frame(predictions_cf, cf$debiased.error, predictions_y, 
                         mu.hat.0, mu.hat.1, predictions_w)
    colnames  = c(paste(rep("tau_hat", 2), c("predictions", "debiased_error"),  rep(paste0("week_", t), 2), sep = "_"),
                  paste0("cf_y_hat_predictions_week_", t),
                  paste0("mu_0_cf_week_", t), paste0("mu_1_cf_week_", t), 
                  paste0("cf_w_hat_predictions_week_", t))
    names(results) = colnames
    results[W==1, ]
  }
  
  did_w_counterfactual = function(t = 0){
    a = simulated_data %>% mutate(log_act = log_act + heterogeneous_treatment)  
    fe.model = felm(formula = log_act ~
                      interaction + after_treatment +
                      membership_age + membership_age_square |
                      user_id,
                    data = a)
    lm.model = lm(data=demeanlist(a[, c("log_act", 
                                        "interaction", "after_treatment", 
                                        "membership_age","membership_age_square")], 
                                  list(a$user_id)), 
                  formula = log_act ~ 
                    interaction + after_treatment +
                    membership_age + membership_age_square)
    fe <- getfe(fe.model)
    newdata = a[which(a$treated == 1 & a$after_treatment == 1), ]
    #newdata = newdata[order(newdata$user_id),]
    counter = newdata
    counter$interaction =0
    fe_effect = fe[fe$idx %in% newdata$user_id, c("effect")]
    mu.hat.1 = predict.lm(object = lm.model, newdata = newdata, se.fit = TRUE, level = 0.95, interval = "confidence")
    mu.hat.1$fit = mu.hat.1$fit + fe_effect
    mu.hat.0 = predict.lm(object = lm.model, newdata = counter, se.fit = TRUE, level = 0.95, interval = "confidence")
    mu.hat.0$fit = mu.hat.0$fit + fe_effect
    results = data.frame(mu.hat.0$fit[,1], mu.hat.0$se.fit, mu.hat.1$fit[,1], mu.hat.1$se.fit)
    colnames  = c(paste0("mu_0_did_week_", t), paste0("mu_0_did_week_", t, "_se"), 
                  paste0("mu_1_did_week_", t), paste0("mu_1_did_week_", t, "_se"))
    names(results) = colnames
    results
  }
  
  
  simulated_data = cbind.data.frame(user_id = as.factor(rep(x = 1:N, each = TT)),
                                    membership_age = rep(x = 7:(6+TT), times = N),
                                    after_treatment = rep(c(rep(x = 0, times = TT-1), 1), times = N), #0000001, 0000001, ....
                                    treated = as.integer(rep(x = rbinom(n = N, size = 1, p = 0.3), each = TT)), # random assignment to treatment 111111, 0000000, ...
                                    ind_effect = rep(x = rnorm(n= N, mean = 3, sd = 5), each = TT))#random fixed effect
  
  simulated_data = cbind(simulated_data, interaction = simulated_data$treated*simulated_data$after_treatment, 
                         membership_age_square = simulated_data$membership_age^2)
  
  
  coefficients = data.frame(matrix(data = c(0.35, 0.0018, -0.122, 0.003), 
                                   dimnames = list(c("interaction", "after_treatment","membership_age","membership_age_square"), "log_act")))
  
  
  foreach(LINEAR_MODEL = c(TRUE, FALSE), .combine = combine_null) %do% {
    RMSE = data.frame()
    
    foreach(sigma_epsilon = c(0.01, 0.1, 0.5, 1, 2, 3.5, 5), .combine = combine_null) %do% {
      epsilon = rnorm(n = N*T, mean = 0, sd = sigma_epsilon) # noise
      
      simulated_data$log_act = (coefficients["interaction",] * simulated_data$interaction +
                                  coefficients["after_treatment",] * simulated_data$after_treatment + 
                                  coefficients["membership_age", ] * simulated_data$membership_age + 
                                  coefficients["membership_age_square", ] * simulated_data$membership_age_square + 
                                  simulated_data$ind_effect + 
                                  epsilon)
      #adding random noise to treatment effect for those treated
      foreach(sigma_effect = c(0.01, 0.1, 0.5, 1, 2, 3.5, 5), .combine = combine_null) %do% {
        
        simulated_data$heterogeneous_treatment = 0
        # This is where the non-linear vs linear model diverge:
        #Non-linear (individual treatment effect depends on the individual fixed effect)
        if(LINEAR_MODEL == FALSE) {
          for(i in which(simulated_data$interaction == 1)){
            simulated_data[i, "heterogeneous_treatment"] = 
              rnorm(n=1, mean = simulated_data$ind_effect[i], sd = sigma_effect)
          }
        }
        #Linear (individual treatment effect is simply with added, random noise):
        if(LINEAR_MODEL == TRUE) {
          simulated_data[which(simulated_data$interaction == 1), "heterogeneous_treatment"] = 
            rnorm(n=sum(simulated_data$interaction), mean = 0, sd = sigma_effect)
        }
        
        #Real average treatment effect:
        real_mean = mean(simulated_data[which(simulated_data$interaction == 1), "heterogeneous_treatment"] + coefficients["interaction",])
        real_sd   = sd(simulated_data[which(simulated_data$interaction == 1), "heterogeneous_treatment"] + coefficients["interaction",])
        
        #Changing data structure (not content) to fit Local Linear Causal Forests
        wide_df <- simulated_data[, c("user_id", "treated", "membership_age", "log_act", "heterogeneous_treatment")] %>% 
          mutate(sumrow = log_act + heterogeneous_treatment)  %>% #adding heterogeneity
          select(user_id, membership_age, treated, sumrow) %>% 
          spread(membership_age, sumrow) #making data in form X1, X2, ..., XT (column for each week)
        
        cf_X = wide_df[, -c(1,2,(TT+2))] # removing "treated", "membesrhip_age", and the last week which will turn into y
        cf_Y = wide_df[, (TT+2)] #taking only the last column (week) as the response
        cf_W = wide_df$treated
        #Running Causal Forests:
        a = simulated_data[which(simulated_data$interaction == 1), c("user_id", "log_act", "heterogeneous_treatment")]
        a = cbind(a, mu_0 = a$log_act-coefficients["interaction",], mu_1 = a$log_act + a$heterogeneous_treatment)
        estimates = cbind(a, 
                          causal_forests_w_counterfactual(X = cf_X, Y= cf_Y, W = cf_W, t = 0), 
                          did_w_counterfactual(t = 0))
        
        error = estimates %>% 
          mutate(error_cf = ((mu_1-mu_0) - (mu_1_cf_week_0-mu_0_cf_week_0))^2, 
                 error_did = ((mu_1-mu_0) - (mu_1_did_week_0-mu_0_did_week_0))^2) %>%
          select(error_cf, 
                 error_did)
        RMSE = rbind.data.frame(RMSE, cbind(sigma_epsilon = sigma_epsilon, sigma_effect = sigma_effect,
                                            RMSE_cf = sqrt(mean(error$error_cf)),
                                            RMSE_did = sqrt(mean(error$error_did))))
        print(RMSE)
      }
    }
    write_csv(x = RMSE, 
              file = paste0("RMSE_did_cf_", ifelse(LINEAR_MODEL, yes = "linear_model_", no = "non-linear_model_"), N, ".csv"))
  }
} else { #don't run models again
  foreach(LINEAR_MODEL = c(TRUE, FALSE), .combine = combine_null) %do% {
    RMSE = read_csv(file = paste0("./RMSE_did_cf_", ifelse(LINEAR_MODEL, yes = "linear_model_", no = "non-linear_model_"), N, ".csv"))
    RMSE_plot = RMSE %>% select(sigma_epsilon, sigma_effect, "RMSE_Causal Forests" = RMSE_cf, "RMSE_DiD" = RMSE_did) %>% 
      gather(key = "method", value = "RMSE", -sigma_epsilon, -sigma_effect) %>% 
      mutate(method = as.factor(str_replace(method, "RMSE_", "")))
    library(car)
    aspect3d(1,1,1)
    a = scatter3d(formula = RMSE ~ sigma_epsilon + sigma_effect | method, 
                  xlab = "Error Variance", zlab = "Treatment Heterogeneity", 
                  surface.col = c("#385999", "#A69F00"), axis.ticks = TRUE,
                  axis.col = c("black", "black", "black"),
                  fit = "smooth", grid = TRUE, parallel = FALSE, data = RMSE_plot, model.summary = TRUE)
    a
    library(rgl)
    
    moviename = paste0("./did_llcf_sim/RMSE_simulation_did_llcf_", if(LINEAR_MODEL) {"linear"} else {"non_linear"}, N)
    movie3d(spin3d(axis = c(0,1,0)), duration = 15, convert = FALSE, fps = 1, movie = moviename, dir = getwd())
  }
}
