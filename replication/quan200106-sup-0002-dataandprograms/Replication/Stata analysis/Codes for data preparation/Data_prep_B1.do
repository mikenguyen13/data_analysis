**********************************************************************
**********************************************************************
****CODE CREATING THE SPECIFIC DATASET AND VARIABLES FOR PROJECT USING 16-MONTH RULE
****This version: February, 2020

****Input : 
*sel_sample_all`i'_prematch_102016.dta (one file per month `i' of layoff)
*DDduration_all`i'.dta (one file per month `i' of layoff)
*sel_sample_all`i'_onlyID_match_this.dta (one file per month `i' of layoff)

****Output: data_grr_QE_reg.dta
**********************************************************************


********************************
**** A) Setup
********************************

clear
set more off
set mem 10000m

********************************
**** B) Load dataset of analysis for first and new layoffs (16-month rule)
********************************

*We use data from 2002 onward because we have precise hiring and separation days
*We stop in december 2008 because we want at least 2 years post-layoff
*j=85 is january 2002; j=168 is december 2008

forvalues j=85(5)165 {

***create empty dataset to store the dataset for analysis
clear
gen var=.
save "data_manipulation`j'.dta", replace

***put together the main datasets
*loop over separation month =`j' to `j'+5
* we are concerned about the size of the resulting datasets so we do it 5 months at the time
local t=`j'+4
forvalues i=`j'(1)`t' {
use "sel_sample_all`i'_prematch_102016.dta", clear 
drop tworker lastprevious_whenreq lastprevious_firstpay lastprevious_npay
sort id_worker
merge id_worker using "DDduration_all`i'.dta", _merge(mergeduration)
sort id_worker
merge id_worker using "sel_sample_all`i'_onlyID_match_this.dta", _merge(mergeUI)
sort id_worker


********************************
**** C) Sample selection 
********************************

*drop workers who were too young or too old at time of first layoff
drop if age<=18 | age>=55

*Look at prevalence of UI extensions
cap drop x
gen x=0
forvalues i=1(1)7 {
tab  tipo`i'
replace x=1 if tipo`i'==3
}
tab month_hat x
egen X=sum(x), by(month_hat)
egen XX=count(x), by(month_hat)
gen ratioX=X/XX
egen tagmonth=tag(month_hat)
sum ratioX if tagmonth==1, detail

*Drop workers in periods with a lot of extensions (more than 2.5% of sample)
drop if ratioX>.025

*Drop remaining workers collecting "extensions"
drop if x==1
drop x X XX ratioX tagmonth


********************************
**** D) Create variables related to first layoff
********************************

*construct day of first layoff as recorded in employment data and as recorded by UI agency
*employment data
gen day_of_layoff=mdy(sep_month, sep_day, year)
*UI agency
gen year_dem=int((date_dem-1)/12)+1995
gen month_dem=date_dem-(year_dem-1995)*12
gen day_of_layoffUI=mdy(month_dem,day_dem,year_dem)
*difference (not everybody go collect UI so does not have to be the same date for those - date for UI agency related to other layoff)
gen diff_day_of_layoff=day_of_layoff-day_of_layoffUI

*construct application date and whether the worker applied for UI within 120 days of first layoff
gen year_req=int((date_req-1)/12)+1995
gen month_req=date_req-(year_req-1995)*12
gen day_of_reqUI=mdy(month_req,day_req,year_req)
gen diff_b_req_demUI=day_of_reqUI-day_of_layoffUI
gen diff_b_req_dem=day_of_reqUI-day_of_layoff
*we identify a worker as having applied within 120 days of first layoff if less than 120 days but also if layoff
*dates in employement and UI data are close "enough"
gen applied_within_120=diff_b_req_dem!=. & diff_b_req_dem>=0 & diff_b_req_dem<=120 & diff_day_of_layoff>=-10 & diff_day_of_layoff<=10
gen applied_within_120UI=diff_b_req_demUI!=. & diff_b_req_demUI>=0 & diff_b_req_demUI<=120 & diff_day_of_layoff>=-10 & diff_day_of_layoff<=10

*construct benefit date of first layoff
gen year_ben=int((date_ben-1)/12)+1995
gen month_ben=date_ben-(year_ben-1995)*12
gen day_of_benUI=mdy(month_ben,day_ben,year_ben)
gen diff_b_ben_demUI=day_of_benUI-day_of_layoffUI
gen diff_b_ben_dem=day_of_benUI-day_of_layoff

*construct reemployment date for first layoff
gen day_of_reempl=mdy(reempl_mhire,reempl_dhire,reempl_yhire)
gen diff_b_reempl_demUI=day_of_reempl-day_of_layoffUI
gen diff_b_reempl_dem=day_of_reempl-day_of_layoff
drop if diff_b_reempl_dem!=. & diff_b_reempl_dem<0
drop if duration<0 & duration!=.

***construct UI benefits outcomes
*date of benefit emission (for valid benefits)
forvalues i=1(1)5 {
gen year_emiss`i'=int((date_emiss`i'-1)/12)+1995 if valor_pago`i'!=0 &  valor_pago`i'!=.
gen month_emiss`i'=date_emiss`i'-(year_emiss`i'-1995)*12 if valor_pago`i'!=0 &  valor_pago`i'!=.
gen day_of_emissUI`i'=mdy(month_emiss`i',day_emiss`i',year_emiss`i') if valor_pago`i'!=0 &  valor_pago`i'!=.
} 

*date of benefit collection
forvalues i=1(1)5 {
gen year_ben`i'=int((date_pag`i'-1)/12)+1995 if valor_pago`i'!=0 &  valor_pago`i'!=.
gen month_ben`i'=date_pag`i'-(year_ben`i'-1995)*12 if valor_pago`i'!=0 &  valor_pago`i'!=.
gen day_of_benUI`i'=mdy(month_ben`i',day_pag`i',year_ben`i') if valor_pago`i'!=0 &  valor_pago`i'!=.
} 

*We need to know who took up benefits after first layoff
gen takeup=valor_pago1!=. & valor_pago1!=0

*We need to know who drew the maximum number of benefits after first layoff
*benefit payment
gen npay_cont2002=0 if takeup==1
forvalues i=1(1)5 {
replace npay_cont2002=npay_cont2002+1 if valor_pago`i'!=0 &  valor_pago`i'!=. & takeup==1
}
replace npay_cont2002=. if npay_cont2002==0
 

********************************
**** E) Create variables related to new layoff
********************************

***construct empty variables to store new layoff date according to employment data and tenure then
*new layoff date according to employment data
gen date_new_layoff=.
gen year_new_layoff=.
gen month_new_layoff=.
gen day_new_layoff=.

*tenure at new layoff date
gen tenure_new_layoff=.

***fill empty variables that we just created
*note: the variables fired_`i'months_after are created in RAIS_UI_prematch_Oct2016
forvalues i=1(1)24 {
*new layoff date according to employment data
gen date_layoff`i'=sep_month_hat+`i' if fired_`i'months_after!=.
replace date_new_layoff=date_layoff`i' if fired_`i'months_after!=. & date_new_layoff==.
replace year_new_layoff=int((fired_`i'months_after-1)/12)+1995 if fired_`i'months_after!=. & year_new_layoff==.
replace month_new_layoff=fired_`i'months_after-(year_new_layoff-1995)*12 if fired_`i'months_after!=. & month_new_layoff==.
replace day_new_layoff=sepday_`i'months_after if fired_`i'months_after!=. & day_new_layoff==.
drop date_layoff`i'
*tenure at new layoff date
replace tenure_new_layoff=tenure_`i'months_after if tenure_`i'months_after!=. & tenure_new_layoff==.
}
gen day_of_new_layoff=mdy(month_new_layoff,day_new_layoff,year_new_layoff)
replace tenure_new_layoff=round(tenure_new_layoff,.1)


********************************
**** F) Keep sample and variables of interest for later 
********************************

*create variables related to difference between first and new layoff dates
*we use the date according to UI agency for the first layoff as it is the date that matters for eligibility around the new layoff date
gen days_b_layoff=day_of_new_layoff-day_of_layoffUI

*keep only individuals with days_b_layoff not too far from 16 months threshold (483 days)
keep if days_b_layoff>=300 & days_b_layoff<=700 & days_b_layoff!=.

*keep variables created here
keep id_worker year date_dem day_of_layoff day_of_layoffUI diff_day_of_layoff day_of_reqUI diff_b_req_demUI diff_b_req_dem applied_within_120 applied_within_120UI day_of_benUI diff_b_ben_demUI diff_b_ben_dem day_of_reempl date_new_layoff tenure_new_layoff  day_of_new_layoff   days_b_layoff  takeup npay_cont2002  


*change variable names such that no confusion later
foreach var in date_dem day_of_layoff day_of_layoffUI diff_day_of_layoff day_of_reqUI diff_b_req_demUI diff_b_req_dem applied_within_120 applied_within_120UI day_of_benUI diff_b_ben_demUI diff_b_ben_dem day_of_reempl date_new_layoff tenure_new_layoff  day_of_new_layoff   days_b_layoff takeup npay_cont2002    {
rename `var' `var'1
}

*the variable date of new layoff will be important to match to outcome variables later on
gen sep_month_hat=date_new_layoff1
gen tenure=tenure_new_layoff1

*save and continue loop
append using "data_manipulation`j'.dta"
save "data_manipulation`j'.dta", replace
}
}


********************************
**** G) Prepare dataset for outcomes of new layoffs: keep information identifying new layoffs 
********************************

clear
gen var=.
save "data_manipulation_ID.dta", replace

forvalues j=85(5)165 {
use "data_manipulation`j'.dta", clear
keep id_worker sep_month_hat 
append using "data_manipulation_ID.dta"
sort id_worker sep_month_hat 
save "data_manipulation_ID.dta", replace
}
duplicates drop
drop var
sort id_worker sep_month_hat 
save "data_manipulation_ID.dta", replace


********************************
**** H) Prepare dataset for outcomes of new layoffs: load main datasets for outcomes, keeping observations selected above
********************************

forvalues j=90(5)180 {

***create empty dataset to store the dataset for analysis (outcomes)
clear
gen var=.
save "data_manipulation_outcome`j'.dta", replace

***loop over separation month =`j' to `j'+5
* we are concerned about the size of the resulting datasets so we prefer to do it 5 months at the time
local t=`j'+4
forvalues i=`j'(1)`t' {
use "sel_sample_all`i'_prematch_102016.dta", clear 
drop tworker lastprevious_whenreq lastprevious_firstpay lastprevious_npay
sort id_worker
merge id_worker using "DDduration_all`i'.dta", _merge(mergeduration)
sort id_worker
merge id_worker using "sel_sample_all`i'_onlyID_match_this.dta", _merge(mergeUI)
duplicates drop

***keep only observations selected above
cap drop _merge
sort id_worker sep_month_hat 
merge 1:1 id_worker sep_month_hat  using "data_manipulation_ID.dta",  keep(match)


********************************
**** I) Prepare dataset for outcomes of new layoffs: sample selection
********************************

*drop workers who were too young or too old at time of new layoff
drop if age<=18 | age>=55

*Look at prevalence of UI extensions
cap drop x
gen x=0
forvalues i=1(1)7 {
tab  tipo`i'
replace x=1 if tipo`i'==3
}
tab month_hat x
egen X=sum(x), by(month_hat)
egen XX=count(x), by(month_hat)
gen ratioX=X/XX
egen tagmonth=tag(month_hat)
sum ratioX if tagmonth==1, detail

*Drop workers in periods with a lot of extensions (more than 2.5% of sample)
drop if ratioX>.025

*Drop remaining workers collecting "extensions"
drop if x==1
drop x X XX ratioX tagmonth


********************************
**** J) Create variables related to new layoff
********************************

*construct day of new layoff as recorded in employment data and as recorded by UI agency
*employment data
gen day_of_layoff=mdy(sep_month, sep_day, year)
*UI agency
gen year_dem=int((date_dem-1)/12)+1995
gen month_dem=date_dem-(year_dem-1995)*12
gen day_of_layoffUI=mdy(month_dem,day_dem,year_dem)
*difference (not everybody go collect UI so does not have to be the same date for those - date for UI agency related to other layoff)
gen diff_day_of_layoff=day_of_layoff-day_of_layoffUI

*construct application date and whether the worker applied for UI within 120 days of new layoff
gen year_req=int((date_req-1)/12)+1995
gen month_req=date_req-(year_req-1995)*12
gen day_of_reqUI=mdy(month_req,day_req,year_req)
gen diff_b_req_demUI=day_of_reqUI-day_of_layoffUI
gen diff_b_req_dem=day_of_reqUI-day_of_layoff
*we identify a worker as having applied within 120 days of new layoff if less than 120 days but also if layoff
*dates in employement and UI data are close "enough"
gen applied_within_120=diff_b_req_dem!=. & diff_b_req_dem>=0 & diff_b_req_dem<=120 & diff_day_of_layoff>=-10 & diff_day_of_layoff<=10
gen applied_within_120UI=diff_b_req_demUI!=. & diff_b_req_demUI>=0 & diff_b_req_demUI<=120 & diff_day_of_layoff>=-10 & diff_day_of_layoff<=10

*construct benefit date of new layoff
gen year_ben=int((date_ben-1)/12)+1995
gen month_ben=date_ben-(year_ben-1995)*12
gen day_of_benUI=mdy(month_ben,day_ben,year_ben)
gen diff_b_ben_demUI=day_of_benUI-day_of_layoffUI
gen diff_b_ben_dem=day_of_benUI-day_of_layoff

*construct reemployment date for new layoff
gen day_of_reempl=mdy(reempl_mhire,reempl_dhire,reempl_yhire)
gen diff_b_reempl_demUI=day_of_reempl-day_of_layoffUI
gen diff_b_reempl_dem=day_of_reempl-day_of_layoff
drop if diff_b_reempl_dem!=. & diff_b_reempl_dem<0
drop if duration<0 & duration!=.

***construct UI benefits outcomes
*date of benefit emission (for valid benefits)
forvalues i=1(1)5 {
gen year_emiss`i'=int((date_emiss`i'-1)/12)+1995 if valor_pago`i'!=0 &  valor_pago`i'!=.
gen month_emiss`i'=date_emiss`i'-(year_emiss`i'-1995)*12 if valor_pago`i'!=0 &  valor_pago`i'!=.
gen day_of_emissUI`i'=mdy(month_emiss`i',day_emiss`i',year_emiss`i') if valor_pago`i'!=0 &  valor_pago`i'!=.
} 
*date of benefit collection
forvalues i=1(1)5 {
gen year_ben`i'=int((date_pag`i'-1)/12)+1995 if valor_pago`i'!=0 &  valor_pago`i'!=.
gen month_ben`i'=date_pag`i'-(year_ben`i'-1995)*12 if valor_pago`i'!=0 &  valor_pago`i'!=.
gen day_of_benUI`i'=mdy(month_ben`i',day_pag`i',year_ben`i') if valor_pago`i'!=0 &  valor_pago`i'!=.
} 


********************************
**** K) Save dataset with outcomes
********************************

append using "data_manipulation_outcome`j'.dta"
save "data_manipulation_outcome`j'.dta", replace
}
}


********************************
**** L) Reorganize the different datasets to get a final dataset of analysis
********************************
forvalues j=90(5)180 {

***reorganize the dataset of first and new layoffs
clear
gen var=.
save "data_manipulation_neworg`j'.dta", replace

forvalues k=85(5)165 {
use "data_manipulation`k'.dta", clear
keep if sep_month_hat>=`j' & sep_month_hat<=`j'+4
append using "data_manipulation_neworg`j'.dta"
save "data_manipulation_neworg`j'.dta", replace
}

*drop if duplicates
bysort id_worker sep_month_hat: gen N=_N
drop if N!=1
drop N
sort id_worker sep_month_hat
save "data_manipulation_neworg`j'.dta", replace

***merge with the dataset of outcome variables
use "data_manipulation_outcome`j'.dta", clear
cap drop _merge
sort id_worker sep_month_hat 
merge 1:1 id_worker sep_month_hat  using "data_manipulation_neworg`j'.dta", keep(match)
sort id_worker sep_month_hat 
save "data_manipulation_outcome_short`j'.dta", replace
}

***put the different datasets together
*we start at 100 (more or less 85+16) and we stop at 179 (such that we can track everybody for at least one year)
use "data_manipulation_outcome_short100.dta", clear
forvalues j=105(5)175 {
append using "data_manipulation_outcome_short`j'.dta"
}
save "data_manipulation_outcome_short.dta", replace


********************************
**** M) Final sample selection and construction of variables
********************************

set more off
use "data_manipulation_outcome_short.dta", replace

*ONLY KEEP OBSERVATIONS WHERE SOME BENEFITS WERE COLLECTED LAST TIME
keep if takeup1==1

*ONLY KEEP OBSERVATIONS WHERE FIVE MONTHLY PAYMENTS WERE COLLECTED LAST TIME
keep if npay_cont20021==5

*ONLY KEEP OBSERVATIONS WHERE LAYOFF DAY WAS AFTER 29th OF THE MONTH
*(if laid off on Feb 5, 2002; eligible again on June 5, 2003 / 16 months)
*drop if previous laid off after the 28th of a month because then the 16 month rule may create
*a discontinuity even in the absence of manipulation
*for instance, all the people laid off between October 29th and 31st of 2006 would be eligible again
*on February 28 of 2008
gen year_layoffUI1=year(day_of_layoffUI1)
gen month_layoffUI1=month(day_of_layoffUI1)
gen day_layoffUI1=day(day_of_layoffUI1)
drop if day_layoffUI1>=29

*ONLY KEEP OBSERVATIONS WITH MORE THAN 6 MONTHS OF TENURE (OTHER ELIGIBILITY CRITERIA)
keep if tenure>6

*ONLY KEEP OBSERVATIONS WITHIN 120 DAYS OF THRESHOLD

gen ind_cutoff=(year_layoffUI1-1995)*12+month_layoffUI1
replace ind_cutoff=ind_cutoff+16
gen year_cutoff=int((ind_cutoff-1)/12)+1995
gen month_cutoff=ind_cutoff-(year_cutoff-1995)*12
gen day_cutoff=day_layoffUI1
gen day_of_cutoff=mdy(month_cutoff,day_cutoff,year_cutoff)
drop day_cutoff month_cutoff year_cutoff ind_cutoff day_layoffUI1 month_layoffUI1 year_layoffUI1

*Defining cutoff related variables
*cutoff
gen cutoff=0
gen dist_cut=day_of_layoff-day_of_cutoff
gen above_cut=dist_cut>=0
gen dist_cut_right=dist_cut*above_cut

*Restrict sample to 120 days around cutoff
keep if dist_cut>=-120 & dist_cut<=120

*before 2005, we don't have the day of benefit emission and payment within a month so we impute it following Gerard and Gonzaga (2016)
set matsize 1600
forvalues i=1(1)5 {
xi: reg day_emiss`i' i.month_emiss`i'*i.sep_day if year>=2005
predict day_emiss_alt`i'
drop _Imon* _Isep*
xi: reg day_pag`i' i.month_ben`i'*i.sep_day if year>=2005
predict day_pag_alt`i'
drop _Imon* _Isep*
}
forvalues i=1(1)5 {
gen day_of_emissUI_alt`i'=mdy(month_emiss`i',day_emiss_alt`i',year_emiss`i') if valor_pago`i'!=0 &  valor_pago`i'!=.
gen day_of_benUI_alt`i'=mdy(month_ben`i',day_pag_alt`i',year_ben`i') if valor_pago`i'!=0 &  valor_pago`i'!=.
} 

*create takeup variable that is defined for the whole sample (as in Appendix in Gerard and Gonzaga, 2016)
gen takeup_alt=applied_within_120UI==1 & day_of_benUI_alt1!=. & valor_pago1!=. & valor_pago1!=0 & day_of_benUI_alt1>=day_of_layoff & (day_of_benUI_alt1<day_of_reempl | day_of_reempl==.) 

*create paid UI duration that is defined for the whole sample (as in Appendix in Gerard and Gonzaga, 2016)
gen npay_alt=0 if takeup_alt==1
forvalues i=1(1)1 {
replace npay_alt=`i' if npay_alt==`i'-1 & valor_pago`i'!=0 &  valor_pago`i'!=. &  date_pag`i'!=. & date_pag`i'>=sep_month_hat & (day_of_emissUI_alt`i'<=day_of_reempl | day_of_reempl==.) & takeup_alt==1
}
forvalues i=2(1)5 {
local min=`i'-1
replace npay_alt=`i' if npay_alt==`i'-1 & valor_pago`i'!=0 &  valor_pago`i'!=. &  date_pag`i'!=. & date_pag`i'>=sep_month_hat & (day_of_emissUI_alt`i'<=day_of_reempl | day_of_reempl==.) & takeup_alt==1
}
replace npay_alt=. if npay_alt==0

*create days without formal job
gen days_without_formal_job=day_of_reempl-day_of_layoff
drop if days_without_formal_job<0 & days_without_formal_job!=. 

*focus on layoffs between 2004 and 2008
keep if sep_month_hat>=109 & sep_month_hat<=168
drop fired_1months_after- hours_4_new minwage1- minwage12 hir_month- hire_type_prior mergeduration- mergeUI 

*censoring the duration without a formal job at 2 years after layoff
gen censoring_point=2*365.5
gen duration_censored=days_without_formal_job
replace duration_censored=censoring_point if (days_without_formal_job>=censoring_point & days_without_formal_job!=.)|days_without_formal_job==.
gen censored=((days_without_formal_job>=censoring_point & days_without_formal_job!=.)|days_without_formal_job==.)

*UI outcomes
gen application=applied_within_120UI
gen treatment=takeup_alt
gen benefit_duration=npay_alt
replace benefit_duration=0 if benefit_duration==.

*covariates: education
cap gen haseducinfo=(schooling!=99)
cap gen educ=0*haseducinfo if schooling==1 | schooling==99
replace educ=2*haseducinfo if schooling==2
replace educ=4*haseducinfo if schooling==3
replace educ=6*haseducinfo if schooling==4
replace educ=8*haseducinfo if schooling==5
replace educ=9*haseducinfo if schooling==6
replace educ=11*haseducinfo if schooling==7
replace educ=13*haseducinfo if schooling==8
replace educ=15*haseducinfo if schooling==9

*covariates: wage, statutory UI benefit, and statutory UI replacement rate
*drop if earned less than minimum wage
drop if avg_w_mw<1
gen logwage=log(real_avg_w)
gen UIbenefit=1 if avg_w_mw<1.25
gen rrate=1/avg_w_mw if avg_w_mw<1.25
replace UIbenefit=0.8*avg_w_mw if avg_w_mw>=1.25 & avg_w_mw<1.65
replace rrate=0.8 if avg_w_mw>=1.25 & avg_w_mw<1.65
replace UIbenefit=(0.8*1.65+0.5*(avg_w_mw-1.65)) if avg_w_mw>=1.65 & avg_w_mw<2.75
replace rrate=(0.8*1.65+0.5*(avg_w_mw-1.65))/avg_w_mw if avg_w_mw>=1.65 & avg_w_mw<2.75
replace UIbenefit=1.87 if avg_w_mw>=2.75
replace rrate=1.87/avg_w_mw if avg_w_mw>=2.75

*covariates: sectors of activity and firm size
gen commerce=(agg_sector==2202|agg_sector==2203)
gen construction=(agg_sector==3304)
gen industry=(agg_sector>=4000 & agg_sector<=4999)
gen services=(agg_sector>=5000)
gen largefirm=firm_size>=6
gen microfirm=firm_size<=2
gen mediumfirm=microfirm==0 & largefirm==0

*50-day bandwidth around the cutoff for graphs
gen h=50
gen in_bandw=1 if abs(dist_cut)<=h & dist_cut<0
replace in_bandw=1 if abs(dist_cut)<=h-1 & dist_cut>=0
keep if in_bandw==1

*keep only variables necessary for empirical analysis
keep id_worker avg_w_mw days_without_formal_job duration_censored dist_cut application treatment benefit_duration logwage male age educ tenure rrate UIbenefit commerce construction industry services microfirm largefirm mediumfirm 

*save dataset of analysis
sort id_worker dist_cut days_without_formal_job
save "Data\data_grr_QE_reg.dta", replace
outsheet using "Data\data_grr_QE_reg.csv", replace comma

