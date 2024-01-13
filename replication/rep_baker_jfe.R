library(did)
library(tidyverse)
library(fixest)

data(base_stagg)

# first make the stacked datasets
# get the treatment cohorts
cohorts <- base_stagg %>%
    select(year_treated) %>%
    # exclude never-treated group
    filter(year_treated != 10000) %>%
    unique() %>%
    pull()

# make formula to create the sub-datasets
getdata <- function(j, window) {
    #keep what we need
    base_stagg %>%
        # keep treated units and all units not treated within -5 to 5
        # keep treated units and all units not treated within -window to window
        filter(year_treated == j | year_treated > j + window) %>%
        # keep just year -window to window
        filter(year >= j - window & year <= j + window) %>%
        # create an indicator for the dataset
        mutate(df = j)
}

# get data stacked
stacked_data <- map_df(cohorts, ~getdata(., window = 5)) %>%
    mutate(rel_year = if_else(df == year_treated, time_to_treatment, NA_real_)) %>%
    fastDummies::dummy_cols("rel_year", ignore_na = TRUE) %>%
    mutate(across(starts_with("rel_year_"), ~ replace_na(., 0)))

# get stacked value
stacked <-
    feols(
        y ~ `rel_year_-5` + `rel_year_-4` + `rel_year_-3` +
            `rel_year_-2` + rel_year_0 + rel_year_1 + rel_year_2 + rel_year_3 +
            rel_year_4 + rel_year_5 |
            id ^ df + year ^ df,
        data = stacked_data
    )$coefficients

stacked_se = feols(
    y ~ `rel_year_-5` + `rel_year_-4` + `rel_year_-3` +
        `rel_year_-2` + rel_year_0 + rel_year_1 + rel_year_2 + rel_year_3 +
        rel_year_4 + rel_year_5 |
        id ^ df + year ^ df,
    data = stacked_data
)$se

# add in 0 for omitted -1
stacked <- c(stacked[1:4], 0, stacked[5:10])
stacked_se <- c(stacked_se[1:4], 0, stacked_se[5:10])


cs_out <- att_gt(yname = "y",
                 data = base_stagg,
                 gname = "year_treated",
                 idname = "id",
                 # xformla = "~x1",
                 tname = "year"
)
cs <- aggte(cs_out, type = "dynamic", min_e = -5, max_e = 5, bstrap = FALSE, cband = FALSE)



res_sa20 = feols(y ~ sunab(year_treated, year) | id + year, base_stagg)
sa = tidy(res_sa20)[5:14, ] %>% pull(estimate)
sa = c(sa[1:4], 0, sa[5:10])

sa_se = tidy(res_sa20)[6:15, ] %>% pull(std.error)
sa_se = c(sa_se[1:4], 0, sa_se[5:10])

compare_df_est = data.frame(
    period = -5:5,
    cs = cs$att.egt,
    sa = sa,
    stacked = stacked
)

compare_df_se = data.frame(
    period = -5:5,
    cs = cs$se.egt,
    sa = sa_se,
    stacked = stacked_se
)

compare_df_longer <- compare_df_est %>% 
    pivot_longer(!period, names_to = "estimator", values_to = "est") %>% 
    
    full_join(compare_df_se %>% pivot_longer(!period, names_to = "estimator", values_to = "se") ) %>% 
    
    mutate(upper = est +  1.96 * se,
           lower = est - 1.96 * se)


ggplot(compare_df_longer) +
    geom_ribbon(aes(
        x = period,
        ymin = lower,
        ymax = upper,
        group = estimator
    )) +
    geom_line(aes(
        x = period,
        y = est,
        group = estimator,
        col = estimator
    ), linewidth = 1)



#### Fail attempt

```{r}
library(fixest)
data("base_stagg")
head(base_stagg)

# get D index the collection of sub-experiments in Omega_A
omega <-
    base_stagg %>% select(year_treated) %>% 
    
    # exclude control group value
    filter(year_treated != 10000) %>% 
    
    unique() %>% pull()

# get the full dataset
df <- base_stagg
final_data <- data.frame()
sub_exp <- data.frame()
# event window
kappa <- 2  # in year

for (d in omega) {
    sub_exp <- df %>%
        
        # make a dummy where it's 1 if the group is treated in the sub-experiment d
        mutate(dummy_unit_treated = ifelse(year_treated == d, 1, 0)) %>%
        
        # make a dummy where it's 1 if the group is treated in the sub-experiment d
        # this should include our strong control (i.e., never-treated) because it's set to 10000
        mutate(dummy_unit_control = ifelse(year_treated > (d + kappa), 1, 0)) %>%
        
        # make a dummy where it's 1 if the calendar date t belongs in sub-experiment d
        mutate(dummy_time =
                   ifelse(year >= d - kappa & year <= d + kappa, 1, 0)) %>%
        
        # make a dummy for inclusion where it's 1 if observation (i,t) in sub-experiment d
        mutate(dummy_inclusion =
                   dummy_time * (dummy_unit_treated + dummy_unit_control))
    
    sub_experiment_d <- sub_exp[sub_exp$dummy_inclusion == 1,]
    
    final_data <- rbind(final_data, sub_experiment_d)
}
```

function to get stacked data

```{r}
get_stacked_data <- function(df, window, unit_status, time_status) {
    stacked_data <- data.frame(stringsAsFactors = FALSE)
    sub_exp <- data.frame(stringsAsFactors = FALSE)
    unit_status <- as.character(unit_status)
    time_status <- as.character(time_status)
    
    omega <-
        base_stagg %>% select(all_of(unit_status)) %>%
        
        # exclude control group value
        filter(unit_status != 10000) %>%
        
        unique() %>% pull()
    
    for (d in omega) {
        sub_exp <- df %>%
            
            # make a dummy where it's 1 if the group is treated in the sub-experiment d
            mutate(dummy_unit_treated = ifelse(df[[unit_status]] == d, 1, 0)) %>%
            
            # make a dummy where it's 1 if the group is treated in the sub-experiment d
            # this should include our strong control (i.e., never-treated) because it's set to 10000
            mutate(dummy_unit_control = ifelse(df[[unit_status]] > (d + window), 1, 0)) %>%
            
            # make a dummy where it's 1 if the calendar date t belongs in sub-experiment d
            mutate(dummy_time =
                       ifelse(df[[as.character(time_status)]] >= (d - window) & df[[as.character(time_status)]] <= (d + window), 1, 0)) %>%
            
            # make a dummy for inclusion where it's 1 if observation (i,t) in sub-experiment d
            mutate(dummy_inclusion =
                       dummy_time * (dummy_unit_treated + dummy_unit_control)) %>% 
            
            mutate(sub_exp = d)
        
        sub_experiment_d <- sub_exp[sub_exp$dummy_inclusion == 1,]
        
        stacked_data <- rbind(stacked_data, sub_experiment_d)
    }
    # create indicator that period t is in the post-period in sub-experiment d
    stacked_data <- stacked_data %>% 
        group_by(sub_exp) %>% 
        mutate(post_sub_exp = if_else(year >= sub_exp, 1, 0)) %>% 
        ungroup()
    rm(sub_exp)
    
    return(stacked_data)
}
```

test function

```{r}
final_data <- get_stacked_data(base_stagg, window = 5, unit_status = "year_treated", time_status = "year" )
# final_data %>% 
#     select(year, sub_exp, post_sub_exp) %>% 
#     view()
```

```{r}
# first make the stacked datasets
# get the treatment cohorts
cohorts <- base_stagg %>%
    select(year_treated) %>%
    # exclude never-treated group
    filter(year_treated != 10000) %>%
    unique() %>%
    pull()

# make formula to create the sub-datasets
getdata <- function(j, window) {
    #keep what we need
    base_stagg %>%
        # keep treated units and all units not treated within -5 to 5
        # keep treated units and all units not treated within -window to window
        filter(year_treated == j | year_treated > j + window) %>%
        # keep just year -window to window
        filter(year >= j - window & year <= j + window) %>%
        # create an indicator for the dataset
        mutate(df = j)
}

# get data stacked
stacked_data <- map_df(cohorts, ~getdata(., window = 5)) %>%
    mutate(rel_year = if_else(df == year_treated, time_to_treatment, NA_real_)) %>%
    fastDummies::dummy_cols("rel_year", ignore_na = TRUE) %>%
    mutate(across(starts_with("rel_year_"), ~ replace_na(., 0)))

# get stacked value
stacked <-
    feols(
        y ~ `rel_year_-5` + `rel_year_-4` + `rel_year_-3` +
            `rel_year_-2` + rel_year_0 + rel_year_1 + rel_year_2 + rel_year_3 +
            rel_year_4 + rel_year_5 |
            id ^ df + year ^ df,
        data = stacked_data
    )$coefficients

stacked_se = feols(
    y ~ `rel_year_-5` + `rel_year_-4` + `rel_year_-3` +
        `rel_year_-2` + rel_year_0 + rel_year_1 + rel_year_2 + rel_year_3 +
        rel_year_4 + rel_year_5 |
        id ^ df + year ^ df,
    data = stacked_data
)$se

# add in 0 for omitted -1
stacked <- c(stacked[1:4], 0, stacked[5:10])
stacked_se <- c(stacked_se[1:4], 0, stacked_se[5:10])
```
