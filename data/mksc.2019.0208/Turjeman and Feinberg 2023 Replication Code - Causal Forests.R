library(here)
library(foreach)
setwd(here::here())
getwd()
seed = 123
set.seed(seed)
library(grf)
library(plyr)
library(dplyr)

dv_names = c("deleted_photos", 
             "searches",
             "sent_messages")
dv_names_strings = c("Deleted Photos",
                     "Searches",
                     "Sent Messages")

#note that "married" actually indicates "single" - which is pretty confusing, I know...
num_covariates = 11
added_covariates = c("age","married", "has_public_photo") #put the desired covariates here

combine_null = function(a, b){NULL}
#' Once control and treatment groups are constructed, run this function. 
#' This function estimates the nuisance params y_hat and w_hat which are the predicted outcome and probability of being treated, respectively, using Local Linear Regression Forests.
#' Then, it uses them and the full timeline along with selected covariates, to predict the treatment effect.
#' If in stage 1, clustering will be based on whether the users had ANY activity after the treatment.
#' 
#' @param X timeline AND (if adding covariates) also covariates of all users. 
#' Each user in a different row and each timeline should be aligned, unless using non local linear correction and then can have varying lengths of timelines.
#' Matrix of size N*(number_of_dvs*T+number of added covariates). Number of DVs is 3
#' @param W Indicator vector - whether this user was treated or not. Vector of length N = N_control + N_treatment. 
#' @param Y Response variable - reaction after the treatment. Vector of length N = N_control + N_treatment. 
#' If multiple DVs or covariates at a time, a matrix of size 3*(N_control + N_treatment)
#' @param variance an indicator of whether to compute variance around the treatment effect. Default to FALSE
#' @param stage_2 binary indicator of whether this is stage 2 of the 2-stage model. Obsolete
#' @param dr_scores an indicator of whether to compute doubly robust scores. Default to FALSE
#' 
#' @return a data frame of estimation of all params and if applicable, errors, of all treated users.
#'
#' @examples

causal_forests_ll_nuisance <- function(treated_group_data, X, W, Y, file_name, variance = FALSE, stage_2 = FALSE,
                                       dr_scores = FALSE){
  #see below about stage 2 which is now obsolete
  if(stage_2){
    clusters = X[,(ncol(X)-1)] # This is the tau_hat for stage 1
  } else {
    clusters = NULL
  }
  # At the time of analysis, causal forests with local linear correction did not support varying lengths of timelines, so we used the regular causal forests instead. llcf was used as a robustness.
  #forest.W = grf::ll_regression_forest(X = X, Y = W, tune.parameters = "all", 
  #                                     seed = seed, clusters = clusters)
  #W_predictions = predict(object = forest.W)
  all_results = treated_group_data
  for(dv_number in 1:3){
    dv_name = dv_names[dv_number]
    for(t in 1:3){
      print(paste0("In week ", t, ". Sys_time is ", Sys.time()))
      print(paste(dv_name, t))
      current_Y = Y[, (3*(t-1)+dv_number)] #This should grab the appropriate dv's column
      #forest.Y = grf::ll_regression_forest(X = X, Y = current_Y, tune.parameters = "all", 
      #                                     seed = seed, clusters = clusters)
      #Y_predictions = predict(object = forest.Y)
      cf = causal_forest(X = X, Y = current_Y, W = W, tune.parameters = "all",
                         #W.hat = W_predictions$predictions,
                         #Y.hat = Y_predictions$predictions, 
                         compute.oob.predictions = TRUE, num.trees = 4000,
                         seed = seed, honesty = TRUE, 
                         ci.group.size = 10, clusters = clusters)
      results = data.frame(cf$predictions, cf$debiased.error, cf$excess.error#, 
                           
                           #Y_predictions$predictions, #Y_predictions$debiased.error,
                           #W_predictions$predictions#, W_predictions$debiased.error
      )
      
      
      colnames  = c(paste(rep("tau_hat", 3), c("predictions", "error", "excess_error"), sep = "_"))#,
      #"y_hat_predictions",
      #"w_hat_predictions")
      if(variance){
        cf_predictions = predict(cf, estimate.variance = TRUE)
        results = cbind.data.frame(results, cf_predictions$variance.estimates)
        colnames = c(colnames, "tau_hat_variance")
      }
      if(dr_scores){
        cf_scores = grf::get_scores(forest = cf, num.trees.for.weights = 1000)
        results = cbind.data.frame(results, cf_scores)
        colnames = c(colnames, "cf_scores")
      }
      names(results) = paste(dv_name, colnames, "week", t, sep = "_")
      all_results = cbind.data.frame(all_results, results[W == 1,])
      
      # This will save the forest but will take lots of time and memory
      #saveRDS(cf, file = paste("./full_forest", dv_name, 
      #                         "week", t, ".Rds", sep = "_"))
      write.csv(x = all_results, file = file_name)
    }
  }
  all_results
}

#' Title
#'
#' @param i the treatment cohort
#' @param restricted_weeks which weeks prior to treatment to include. Can be anything from 1 to 27. If don't want to restrict - put any non positive number. 
#' @param restricted_cohorts how many control cohorts to include. Can be anything from 1 to number of cohorts. If don't want to restrict - put either negative number, or i or more.
#' @param placebo If want to run a placebo test, put any number larger than 3 (number of weeks post treatment), which will indicate the time, before the data breach, that the placebo treatment "took place".
#' @param add_covariates If desired, add covariates in addition to the timeline. In the paper we had False
#'
#' @return a DF of all treated users along with their estimated parameters for all weeks after the treatment. 
#'
#' @examples
#' 
restricted_cohorts = 5 # The number of cohorts to act as control. For the first cohorts, there are less cohorts to use as control. 
placebo = 0 #0 indicates not a placebo test.  Placebo greater than 3 indicates the number of weeks prior to the treatment to have a placebo treatment in.

add_covariates = FALSE
compute_variance = TRUE
RUN_STAGE_2 = FALSE # in earlier version of the analysis, we also did a not-only binarized version of the analysis. Eventually, we realized the data are too noisy for this, and removed this analysis. It is left here if anyone is interested, but is not tested for final versions.

get_DR_scores = FALSE #should we get the doubly robust scores? In the paper we realized this did not work properly, results were not reasonable (out of range completely). This can happen. 


file_name = paste0("./all_cohorts",
                   (if(RUN_STAGE_2) "_2stages_"),
                   (if(placebo >0) (paste0("_placebo_", placebo))),
                   (if(add_covariates) "_add_cov"),
                   (if(compute_variance) "_variance"),
                   (if(get_DR_scores) "dr_scores_1000"),
                   ".csv")

# reading the data
all_cohorts = 
  foreach::foreach(dv_name = dv_names, .inorder = TRUE, .combine = left_join) %do% {
    read.csv(file = paste("./", dv_name,"_deidentified_sample.csv", sep = "")) %>% 
      rename_at(.vars = vars(starts_with("X")), .funs = funs(paste0(dv_name, "_", .)))
  }


#Taking all cohorts or (if debugging) just a random sample of the data
rand_users = all_cohorts #sample_n(tbl = all_cohorts, size = 5000, replace = FALSE)

#Initial cohort
min_cohort = 3
max_cohort = max(rand_users$join_week)-placebo-3
#temporary to ease on debugging. comment these two lines if you want to run on all cohorts.
min_cohort = 10
max_cohort = 13


X_stage_1 = data.frame()
Y_stage_1 = data.frame()
treated_indicator = c()
treated_group_data = data.frame()

#Creating the data of all cohorts between min_cohort and max_cohort
for(i in min_cohort:max_cohort){
  #check input
  if(restricted_cohorts <= 0){
    print("Negative number of restricted_cohorts - changing to all cohorts prior to the i")
    restricted_cohorts = i # cannot put anything less than 1.
  }
  #if(restricted_cohorts )
  if(placebo > 0 & placebo <= 3){
    print("Placebo should be greater than 3 (number of observed weeks after treatment) if wanted a Placebo test. 
          Still, continue, but expect that some weeks will have an effect")
  }
  join_week = rand_users$join_week
  num_weeks = 27-i-placebo #before the treatment
  if(num_weeks < 3){
    print("check your input: 27-i-placebo is smaller than 3, needs to be at least 3")
    stop()
  }
  
  x = data.frame()
  y = data.frame()
  
  
  # Shifting the data of all control cohorts to align with that of the treated cohort.
  # j runs on at most restricted_cohorts minus 3 (i-3). Then j is i and gets the treatment group's data directly
  for(j in c(max((i-restricted_cohorts), 0):(i-3),i)){
    users = which(join_week == j)
    a_df = rand_users[users, ]
    
    #Creates two data frames - one before the treatment, and one after the treatment
    before_treatment_df = a_df %>% select(ends_with(paste0(".", c((i-j+num_weeks):(i-j+1+placebo)))))#[, c(start:treatment)]
    if(i == j){
      treated_group_data = rbind(treated_group_data, a_df)
      if(placebo == 0){
        after_treatment_df =  a_df %>% select(ends_with(paste0("X", 0:2)))
      } else {
        after_treatment_df = a_df %>% select(ends_with(paste0("X.", c((i-j+placebo):(i-j-2+placebo)))))
      }
    } else {
      #this takes the weeks that align with after the treatment weeks, for the control group. 
      after_treatment_df = a_df %>% select(ends_with(paste0("X.",c((i-j+placebo):(i-j-2+placebo))))) #[, c((treatment+1):end)]a_df %>% 
    }
    
    #This is the only difference between "all_activities_in_covariates" and "all_cohorts_single_forest"
    #Where, instead of changing the variances name to be week1...week_<number of weeks for this cohort>, it is week(*i+*1)...week<*i + *number of weeks for this cohort>
    colnames(before_treatment_df) = paste0(dv_names, "_week", rep((i+1):(i+num_weeks-placebo), each = 3))
    colnames(after_treatment_df) = paste0(dv_names, "_week", rep(1:3, each = 3))#rep((num_weeks+1):(num_weeks+3), each = 3))
    
    if(add_covariates) {
      before_treatment_df = cbind(a_df[,added_covariates], before_treatment_df)
    } else {
      added_covariates = c()
    }
    x = rbind(x, cbind(join_week = j, before_treatment_df))
    y = rbind(y, after_treatment_df)
  }
  
  sub_join_week = x$join_week
  
  treated_indicator_cur = (sub_join_week == i)
  X = x[, -1] #removing the column indicating the join_week
  # stage one - changing timeline to binary
  if(add_covariates){
    x_stage_1_cur = cbind.data.frame(X[,c(1:length(added_covariates))], 
                                     apply(X = X[, -c(1:length(added_covariates))], 
                                           MARGIN = 2, 
                                           FUN = function(temp) {as.integer(as.integer(temp) > 0)}))
  } else {
    x_stage_1_cur = as.data.frame(apply(X = X, 
                                        MARGIN = 2, 
                                        FUN = function(temp) {as.integer(as.integer(temp) > 0)}))
  }
  y_stage_1_cur = as.data.frame(apply(y, MARGIN = 2, FUN = function(a){as.integer(as.integer(a) > 0) }))
  treated_indicator = c(treated_indicator, treated_indicator_cur)
  Y_stage_1 = rbind.fill(Y_stage_1, y_stage_1_cur)
  X_stage_1 = rbind.fill(X_stage_1, x_stage_1_cur)
}

rm("x", "y", "treated_indicator_cur", "a_df", "after_treatment_df", "before_treatment_df", "x_stage_1_cur", "y_stage_1_cur", "join_week", "rand_users")
results = data.frame()
possibleError <- tryCatch(expr = {
  
  results = causal_forests_ll_nuisance(treated_group_data = treated_group_data, 
                                       X = X_stage_1, 
                                       Y = Y_stage_1, 
                                       W = treated_indicator, 
                                       file_name = file_name,
                                       variance = compute_variance,
                                       dr_scores = get_DR_scores)    
  
  print("finished stage 1")
  
}, 
error = function(e) {
  print(paste("error in running forest: error message = ", geterrmessage(), sep= " "))
}
)
if(!is.null(possibleError)) { 
  #next
}

write.csv(x = results, file = file_name)

summary(results %>% select(contains("tau_hat_predictions")))

#all_cohorts_single_forest_data = as.data.frame(read_csv(file_name))
all_cohorts_single_forest_data = results
demogs = as.data.frame(as.matrix(all_cohorts_single_forest_data %>% 
                                   select(age, married, has_public_photo, join_week) %>% 
                                   mutate(private = 1-has_public_photo, married = 1-married) %>%  #yes, married was mis-coded
                                   select(-has_public_photo) %>% 
                                   mutate(Age = scale(x = age, center = TRUE, scale = FALSE),
                                          AgeSq = scale(x = age^2, center = TRUE, scale = FALSE),
                                          Married = scale(x = married, center = TRUE, scale = FALSE),
                                          Private = scale(x = private, center = TRUE, scale = FALSE),
                                          Cohort = scale(x = join_week, center = TRUE, scale = FALSE),
                                          CohortSq = scale(x = join_week^2, center = TRUE, scale = FALSE)) %>% 
                                   mutate(Age_Cohort = scale(x = Age * Cohort, center = TRUE, scale = FALSE),
                                          Age_Married = scale(x = Age * Married, center = TRUE, scale = FALSE), 
                                          Age_Private = scale(x = Age * Private, center = TRUE, scale = FALSE),
                                          Cohort_Married = scale(x = Cohort * Married, center = TRUE, scale = FALSE),
                                          Cohort_Private = scale(x = Cohort * Private, center = TRUE, scale = FALSE),
                                          Married_Private = scale(x = Married * Private, center = TRUE, scale = FALSE)) %>%
                                   select(-age, -married, -private, -join_week)))
quantiles_age = quantile(all_cohorts_single_forest_data$age)
quantiles_cohort = quantile(all_cohorts_single_forest_data$join_week)
summaries = data.frame()

for(dv_name in dv_names){
  for(t in 1:3){
    print(paste0("In DV ", dv_name, ", week ", t))#, ". Sys_time is ", Sys.time()))
    cur_df = all_cohorts_single_forest_data %>% 
      select(user_id, contains("predictions"), contains("variance")) %>% 
      select(user_id, starts_with(dv_name)) %>% 
      select(user_id, ends_with(paste0("week_" ,t)))
    colnames(cur_df) = c("user_id", "Mean", "Sigma")
    
    mean_effect = mean(cur_df$Mean)
    var_effect = (sum(cur_df$Sigma^2 + (cur_df$Mean - mean_effect)^2)) / (nrow(cur_df)^2)
    summary = cbind.data.frame(DV = dv_name, 
                               Week = t, 
                               Mean = mean_effect, 
                               SD = sqrt(var_effect))
    summaries = rbind.data.frame(summaries, summary)
    #}
  }
  #for(pos in which(search() == "df")){detach(pos = pos)}
}

summaries_2 = (left_join(x = summaries, 
                         y = cbind.data.frame(DV = dv_names, DV_names = dv_names_strings)))
summaries_2 = summaries_2 %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate(DV = DV_names) %>% select(-DV_names)

# this should be table 1
library(tidyr)
summaries_table = data.frame(summaries_2 %>% 
                               mutate(Mean = 100*Mean, SD = 100*SD) %>% #This is to make it percentages
                               pivot_longer(cols = c(Mean, SD)) %>% 
                               pivot_wider(id_cols = c(DV, name), names_from = Week, names_prefix = "Week "))

stargazer::stargazer(summaries_table, type = "html", out = "./main_results.html",
                     summary = FALSE, rownames = FALSE, digits = 12, digits.extra = 12)

library(viridis)
library(ggridges)
library(ggplot2)

all_changes = data.frame()

for(dv_name in dv_names){
  for(t in 1:3){
    print(paste0("In DV ", dv_name, ", week ", t))#, ". Sys_time is ", Sys.time()))
    cur_df = all_cohorts_single_forest_data %>% 
      select(user_id, contains("predictions"), contains("variance")) %>% 
      select(user_id, starts_with(dv_name)) %>% 
      select(user_id, ends_with(paste0("week_" ,t)))
    colnames(cur_df) = c("user_id", "Mean", "Sigma")
    
    changes = cbind.data.frame(DV = dv_name, 
                               Week = t, 
                               Mean = cur_df$Mean)
    all_changes = rbind.data.frame(all_changes, changes)
    #}
  }
  #for(pos in which(search() == "df")){detach(pos = pos)}
}

all_changes_2 = (left_join(x = all_changes, 
                           y = cbind.data.frame(DV = dv_names, DV_names = dv_names_strings)))
all_changes_2 = all_changes_2 %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate(DV = DV_names) %>% select(-DV_names)

min_y = min(all_changes_2 %>% 
              group_by(Week, DV) %>% 
              summarize(min = quantile(Mean, probs = 0.025)) %>% 
              ungroup() %>% 
              select(min))

max_y = max(all_changes_2 %>% 
              group_by(Week, DV) %>% 
              summarize(max = quantile(Mean, probs = 0.975)) %>% 
              ungroup() %>% 
              select(max))

plot_ridges = ggplot(data = all_changes_2, 
                     mapping = aes(x = Mean, 
                                   y = as.factor(Week), 
                                   fill= (0.5 - abs(0.5-after_stat(ecdf)))))  + 
  stat_density_ridges(geom = "density_ridges_gradient",  
                      calc_ecdf = TRUE,
                      rel_min_height = 0.000000005, 
                      bandwidth = 0.003) + 
  ggtitle(label = "Distribution of Change in Probability to be Active", 
          subtitle = "Density of average individual treatment effect") +
  scale_fill_viridis(name = "Tail probability", direction = -1) +
  facet_wrap(~ DV, ncol = 3) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 0.01), 
                     limits = c(min_y, max_y))+
  ylab(label = "Week After Announcement") + 
  xlab(label = "Percent Change") + 
  coord_flip()

plot(plot_ridges)

for_star = list() 
for(dv_name in dv_names){
  for(t in 1:3){
    cur_df_2 = cbind.data.frame(all_cohorts_single_forest_data %>% 
                                  select(dv = starts_with(paste0(dv_name, "_tau_hat_predictions_week_", t))), 
                                demogs)
    cur_df_3 = cur_df_2 %>% mutate(dv = 100*dv)# just to have it in percentages
    stargazer_lm = lm(dv ~ . , data = cur_df_3)
 
    ####### CHANGE MODEL ABOVE THIS LINE ##########
    
    for_star[[dv_name]][[paste("Week", t)]][["LM"]] = stargazer_lm
    
    
  }
}

library(broom)

for_plot = data.frame()
for(dv_name in dv_names){
  for(t in 1:3){
    for_plot = rbind.data.frame(for_plot, cbind(DV = dv_name, Week = t, tidy(for_star[[dv_name]][[paste("Week", t)]][["LM"]])))
  }
}


#Setting the data to be with explanatory names
colnames(for_plot) = c("DV","Week","Coefficient","Estimate","stderr","t_value","Pr_t")

regression_results_2 = (dplyr::left_join(x = for_plot,
                                         y = cbind.data.frame(DV = dv_names, DV_names = dv_names_strings)))
regression_results_2 = regression_results_2 %>%
  mutate_if(is.character, as.factor) %>%
  mutate(DV = DV_names) %>% 
  select(-DV_names)
levels(regression_results_2$Coefficient)[1] = "Intercept"


#colors by DV
# Note that if you chose to analyze just several cohorts this may make the results appear much different due to the Cohort interquantile range being much smaller.
age_interuqantile = as.numeric(quantiles_age[4]-quantiles_age[2])
cohort_interquantile = as.numeric(quantiles_cohort[4]-quantiles_cohort[2])

regression_results_with_age_cohort_interquantile = regression_results_2 %>% 
  filter(Coefficient %in% c("Age", "Married", "Private", "Cohort")) %>% 
  select(-t_value, -Pr_t) %>% 
  mutate(Estimate = Estimate/100, stderr = stderr/100) %>% 
  mutate(Estimate = ifelse(test = (Coefficient == "Age"), 
                           yes = Estimate * age_interuqantile,
                           no = ifelse(test = (Coefficient == "Cohort"), 
                                       yes = Estimate * cohort_interquantile, 
                                       no = Estimate)), 
         stderr = ifelse(test = (Coefficient == "Age"), 
                         yes = stderr * age_interuqantile,
                         no = ifelse(test = (Coefficient == "Cohort"), 
                                     yes = stderr * cohort_interquantile, 
                                     no = stderr)))
library("ggthemes")

# Note: I removed "Cohort" from the plot, as the sample of cohorts may bias the results. If you chose all cohorts you may put this back.
plot_colors_by_DV =
  (ggplot(regression_results_with_age_cohort_interquantile %>% filter(Coefficient != "Cohort"),
          aes(x = as.factor(Week), y = Estimate, group = DV, fill = DV)) +
     geom_bar(stat="identity", position=position_dodge(width=0.9)) +
     geom_errorbar(mapping= aes(ymin = Estimate - 1.96 * stderr,
                                ymax = Estimate + 1.96 * stderr),
                   width = 0.2, position = position_dodge(width = 0.9)) +
     facet_grid(. ~ Coefficient, scale = "free") +
     labs(x = "Week After Breach", y = "Coefficient Estimate",
          title = "Predictors of Percent Change in Weekly Activities", 
          subtitle = "Age and Cohort were rescaled to fit the 75%-25% interuqantiles") +
     geom_vline(xintercept = c(1.5, 2.5), colour = "grey") +
     scale_y_continuous(labels = scales::percent_format(accuracy = 0.01)) #+
   
  ) + scale_fill_colorblind()

plot_colors_by_DV 
