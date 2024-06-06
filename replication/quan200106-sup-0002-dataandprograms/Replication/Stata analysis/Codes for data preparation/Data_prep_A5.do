**********************************************************************
**********************************************************************
****LAST ROUND OF GENERAL DATA PREPARATION FOR BOTH UI PROJECTS
****This version: February, 2020

****Input: 
*sel_sample_all`i'_onlyID_match_this_short.dta (one file per month `i' of layoff)
*sel_sample_all`i'.dta (one file per month `i' of layoff)

****Output: sel_sample_all`i'_prematch_102016.dta (one file per month `i' of layoff)
**********************************************************************

********************************
**** A) Setup
********************************

clear
set more off
set mem 10000m


********************************
**** B) Keep months of layoffs for which we have 3 full years of pre-layoff data 
******** and recalculate variables key for institutional details of both UI projects
********************************

*** focus on layoffs starting in 1997 to have 3 full years of pre-layoff data
forvalues i=25(1)192 {

*keep minimal info for matching later on
use "sel_sample_all`i'_onlyID_match_this_short.dta", clear
keep id_worker lastprevious_whenreq lastprevious_firstpay lastprevious_npay
sort id_worker
save "temp.dta", replace

***recalculate variables key for institutional details of both UI projects starting with the detailed data on displaced workers 
use "sel_sample_all`i'.dta", clear
duplicates drop

**1) re-do the basic sample selection and consistency checks to get where we can recalculate key variables for relevant sample (similar to Data_prep_A3)

*identify observation related to layoff of interest
gen a=(sep_month_hat==`i' & reas_sep==11)

*drop unnecessary variables
cap drop causafast1_fonte causafast3_fonte causafast2_fonte ceivinc indceivinc_fonte indpat_fonte muntrab_fonte nacionalidad_fonte portdefic_fonte qtdiasafas radiccnpj sbclas20_fonte tpdefic_fonte dtadmissao

*create continuous month of "hiring" and "separation" for observations with no separation or hiring in the year itself (indicate already employed at start of the year and still employed at end of the year)
gen month_hat=sep_month_hat
replace month_hat=12*(year-1995)+12.5 if sep_month==99 & reas_sep==0
gen month_hat2=hire_month_hat
replace month_hat2=12*(year-1995)+0.5 if hir_month==0 & hire_type==8

*drop if worker ID number inconsistent
drop if pisOK!=1 & year!=1994

*drop unnecessary variables
drop pisOK var1 _merge sampled

*we will not keep layoff with less than 1 month of tenure (never the ones we will study)
replace a=0 if month_hat==hire_month_hat & a==1

*keep layoffs with single job in month of layoff
sum year if a==1
local yyy=r(mean)
gen flag=(month_hat2<`i' & month_hat>=`i') if year==`yyy'
egen FLAG=sum(flag), by(id_worker)
egen a2=sum(a), by(id_worker)
keep if a2==1 & FLAG==1
drop a2 FLAG flag tworker

*drop if establishment ID number inconsistent
replace a=0 if cnpjOK!=1

*keep history of layoffs of interest given above restrictions
egen a2=sum(a), by(id_worker)
keep if a2==1
drop a2

*now match with UI data (useful for RD with manipulation project, i.e., 16-month rule)
sort id_worker
merge m:1 id_worker using "temp.dta", keep(master match)
drop _merge

*check consistency of work history and tenure calculation in 3 years before layoff
cap drop n
sort id_worker month_hat month_hat2 hire_month_hat
bysort id_worker: gen n=_n
gen consistency=. if a==1 & tenure<36
replace consistency=0 if a==1 & tenure<36 & hire_type!=8 & tenure<=sep_month & (month_hat[_n-1]<=month_hat2|id_worker[_n-1]!=id_worker)
replace consistency=1 if a==1 & tenure<36 & id_worker==id_worker[_n-1] & id_firm==id_firm[_n-1] & id_estb==id_estb[_n-1] & hire_type==8 & sep_month[_n-1]==99 & tenure[_n-1]+sep_month+1>=tenure & tenure[_n-1]+sep_month-1<=tenure & hir_month[_n-1]!=0 & year==year[_n-1]+1 & (month_hat[_n-2]<=month_hat2[_n-1]|id_worker[_n-2]!=id_worker[_n-1])
replace consistency=2 if a==1 & tenure<36 & id_worker==id_worker[_n-1] & id_firm==id_firm[_n-1] & id_estb==id_estb[_n-1] & hire_type==8 & sep_month[_n-1]==99 & tenure[_n-1]+sep_month+1>=tenure & tenure[_n-1]+sep_month-1<=tenure & hir_month[_n-1]==0 & year==year[_n-1]+1 & id_worker==id_worker[_n-2] & id_firm==id_firm[_n-2] & id_estb==id_estb[_n-2] & sep_month[_n-2]==99 & tenure[_n-2]+12+sep_month+1>=tenure & tenure[_n-2]+12+sep_month-1<=tenure & hir_month[_n-2]!=0 & year==year[_n-2]+2 & (month_hat[_n-3]<=month_hat2[_n-2]|id_worker[_n-3]!=id_worker[_n-2])
replace consistency=3 if a==1 & tenure<36 & id_worker==id_worker[_n-1] & id_firm==id_firm[_n-1] & id_estb==id_estb[_n-1] & hire_type==8 & sep_month[_n-1]==99 & tenure[_n-1]+sep_month+1>=tenure & tenure[_n-1]+sep_month-1<=tenure & hir_month[_n-1]==0 & year==year[_n-1]+1 & id_worker==id_worker[_n-2] & id_firm==id_firm[_n-2] & id_estb==id_estb[_n-2] & sep_month[_n-2]==99 & tenure[_n-2]+12+sep_month+1>=tenure & tenure[_n-2]+12+sep_month-1<=tenure & hir_month[_n-2]==0 & year==year[_n-2]+2 & id_worker==id_worker[_n-3] & id_firm==id_firm[_n-3] & id_estb==id_estb[_n-3] & sep_month[_n-3]==99 & tenure[_n-3]+24+sep_month+1>=tenure & tenure[_n-3]+24+sep_month-1<=tenure & hir_month[_n-3]!=0 & year==year[_n-3]+3 & (month_hat[_n-4]<=month_hat2[_n-3]|id_worker[_n-4]!=id_worker[_n-3])
egen Consistency=max(consistency), by(id_worker)
gen consistency2=. if a==1 & tenure>=36
replace consistency2=3 if a==1 & tenure>=36 & id_worker==id_worker[_n-1] & id_firm==id_firm[_n-1] & id_estb==id_estb[_n-1] & hire_type==8 & sep_month[_n-1]==99 & tenure[_n-1]+sep_month+1>=tenure & tenure[_n-1]+sep_month-1<=tenure & hir_month[_n-1]==0 & year==year[_n-1]+1 & id_worker==id_worker[_n-2] & id_firm==id_firm[_n-2] & id_estb==id_estb[_n-2] & sep_month[_n-2]==99 & tenure[_n-2]+12+sep_month+1>=tenure & tenure[_n-2]+12+sep_month-1<=tenure & hir_month[_n-2]==0 & year==year[_n-2]+2 & id_worker==id_worker[_n-3] & id_firm==id_firm[_n-3] & id_estb==id_estb[_n-3] & sep_month[_n-3]==99 & tenure[_n-3]+24+sep_month+1>=tenure & tenure[_n-3]+24+sep_month-1<=tenure & year==year[_n-3]+3 & (month_hat[_n-4]<=month_hat2[_n-3]|id_worker[_n-4]!=id_worker[_n-3])
egen Consistency2=max(consistency2), by(id_worker)

gen llimit=0 if consistency==0
replace llimit= sep_month-1 if consistency==1
replace llimit= 12+sep_month-1 if consistency==2
replace llimit= 24+sep_month-1 if consistency==3
gen ulimit=sep_month+1 if consistency==0
replace ulimit= 12+sep_month+1 if consistency==1
replace ulimit=24+sep_month+1 if consistency==2
replace ulimit=36+sep_month+1 if consistency==3
gen mistake=(tenure<llimit | tenure>ulimit) if a==1 
replace mistake=0 if consistency2!=.|consistency==11|consistency==22
replace mistake=. if consistency2==. & consistency==.

gen this=hire_type if consistency==0 & mistake==0
replace this =hire_type[_n-1] if consistency==1 & mistake==0
replace this =hire_type[_n-2] if consistency==2 & mistake==0
replace this =hire_type[_n-3] if consistency==3 & mistake==0

gen xx=0 if a==1 & mistake==0 & (consistency==0|consistency==1|consistency==2|consistency==3|consistency==11|consistency==22|consistency2==3|consistency2==11|consistency2==22)
replace xx=1 if a[_n+1]==1 & mistake[_n+1]==0 & (consistency[_n+1]==1|consistency[_n+1]==2|consistency[_n+1]==3|consistency[_n+1]==11|consistency[_n+1]==22|consistency2[_n+1]==3|consistency2[_n+1]==11|consistency2[_n+1]==22) 
replace xx=2 if a[_n+2]==1 & mistake[_n+2]==0 & (consistency[_n+2]==2|consistency[_n+2]==3|consistency[_n+2]==22|consistency2[_n+2]==3|consistency2[_n+2]==22) 
replace xx=3 if a[_n+3]==1 & mistake[_n+3]==0 & (consistency[_n+3]==3 | consistency2[_n+3]==3)
gen thisstart=month_hat2 if xx==3 & (consistency[_n+3]==3|consistency2[_n+3]==3)
replace thisstart=month_hat2 if xx==2 & (consistency[_n+2]==2|consistency[_n+2]==22|consistency2[_n+2]==22)
replace thisstart=month_hat2 if xx==1 & (consistency[_n+1]==1|consistency[_n+1]==11|consistency2[_n+1]==11)
replace thisstart=month_hat2 if xx==0 & consistency==0
egen THISstart=max(thisstart), by(id_worker)
replace THISstart=. if consistency==. & consistency2==.
gen thistenure=tenure if xx==3 & (consistency[_n+3]==3|consistency2[_n+3]==3)
replace thistenure = tenure if xx==2 & (consistency[_n+2]==2|consistency[_n+2]==22|consistency2[_n+2]==22)
replace thistenure = tenure if xx==1 & (consistency[_n+1]==1|consistency[_n+1]==11|consistency2[_n+1]==11)
replace thistenure = tenure if xx==0 & consistency==0
egen THIStenure=max(thistenure), by(id_worker)
egen THIShiretype=max(this), by(id_worker)

*calculate total tenure in last 36 months and other jobs in last 36 months
gen otherjob=1 if (month_hat>=`i'-36 & xx==. & Consistency!=. & month_hat2<`i')
egen OTHERjob=max(otherjob), by(id_worker)
gen yy=n if thisstart!=. 
egen firstjob=max(yy), by(id_worker)
sum n if a==1
local maxmax=r(max)
gen ppp=0 if otherjob==1
replace ppp=1 if thisstart[_n+1]!=. & sep_month==99 & otherjob==1 & id_worker==id_worker[_n+1]
forvalues h=1(1)`maxmax' {
replace ppp=1 if thisstart[_n+`h'+1]!=. & id_worker==id_worker[_n+`h'+1] & ppp[_n+1]==0 & otherjob==1 & ((sep_month!=99 & hir_month[_n+1]==0)|(sep_month==99 & hir_month[_n+1]!=0))
}
egen inconsistentotherPRE=max(ppp), by(id_worker)
drop ppp

**2) now calculate accumulated tenure in last 36 months (useful for project on RD at 24 months of tenure) 

*identify multiple jobs in the past (small group, cannot calculate accumulated eligibility for this group)
gen FFLAG=0
local t=`i'-36
forvalues j=`i'(-1)`t' {
gen flagg=(month_hat2<`j' & month_hat>=`j')
egen FLAGG=sum(flagg), by(id_worker)
replace FFLAG=1 if FLAG>1
drop flagg FLAGG
}

*number of other jobs in last 36 months
gen endotherjob=1 if otherjob==1 & sep_month!=99
bysort id_worker: gen Rankotherjob=sum(endotherjob)
gen rankotherjob=Rankotherjob if endotherjob==1
egen Njobs=max(rankotherjob), by(id_worker)
drop Rankotherjob

*sample of interest
gen INSS=1 if a==1 & (Consistency!=.|Consistency2!=.) & ((OTHERjob==1 & inconsistentotherPRE==0)|OTHERjob!=1) & FFLAG==0
egen inss=max(INSS), by(id_worker)

*number of other jobs
sum Njobs if inss==1 & a==1
local maxx=r(max)

**2a) calculate lower bound on accumulated tenure
gen testten_min=tenure if inss==1 & a==1
replace testten_min=36 if testten_min>36 & inss==1 & a==1
* previous jobs
gen ppp=0 if otherjob==1 & inss==1 
forvalues h=1(1)`maxx' {
replace ppp=`h' if otherjob==1 & rankotherjob==Njob+(1-`h') & inss!=.
*start and end in same year (or censored in one year)
replace testten_min=tenure if ppp==`h' & hire_type!=8 & inss!=. & month_hat2>`i'-36 & tenure<=(sep_month_hat-hire_month_hat+1)
replace testten_min=sep_month_hat-(`i'-36+1) if ppp==`h' & hire_type!=8 & inss!=. & month_hat2<=`i'-36
replace testten_min=sep_month_hat-(`i'-36+1) if ppp==`h' & hire_type==8 & id_worker!=id_worker[_n-1] & inss!=. & month_hat2<=`i'-36
*start in previous year
replace testten_min=tenure if ppp==`h' & hire_type==8 & hire_type[_n-1]!=8 & id_worker==id_worker[_n-1] & inss!=. & month_hat2[_n-1]>`i'-36 & tenure<=(sep_month_hat-hire_month_hat[_n-1]+1)
replace testten_min=sep_month_hat-(`i'-36+1) if ppp==`h' & hire_type==8 & hire_type[_n-1]!=8 & id_worker==id_worker[_n-1] & inss!=. & month_hat2[_n-1]<=`i'-36
replace testten_min=sep_month_hat-(`i'-36+1) if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & id_worker==id_worker[_n-1] & id_worker!=id_worker[_n-2] & inss!=. & month_hat2[_n-1]<=`i'-36
*start in previous previous year
replace testten_min=tenure if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]!=8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & inss!=. & month_hat2[_n-2]>`i'-36 & tenure<=(sep_month_hat-hire_month_hat[_n-2]+1)
replace testten_min=sep_month_hat-(`i'-36+1) if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]!=8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & inss!=. & month_hat2[_n-2]<=`i'-36
replace testten_min=sep_month_hat-(`i'-36+1) if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]==8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & id_worker!=id_worker[_n-3] & inss!=. & month_hat2[_n-2]<=`i'-36
*start in previous previous previous year
replace testten_min=tenure if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]==8 & hire_type[_n-3]!=8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & id_worker==id_worker[_n-3] & inss!=. & month_hat2[_n-3]>`i'-36 & tenure<=(sep_month_hat-hire_month_hat[_n-3]+1)
replace testten_min=sep_month_hat-(`i'-36+1) if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]==8 & hire_type[_n-3]!=8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & id_worker==id_worker[_n-3] & inss!=. & month_hat2[_n-3]<=`i'-36
replace testten_min=sep_month_hat-(`i'-36+1) if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]==8 & hire_type[_n-3]==8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & id_worker==id_worker[_n-3] & id_worker!=id_worker[_n-4] & inss!=. & month_hat2[_n-3]<=`i'-36
*check
count if ppp==`h' & testten_min!=.
count if ppp==`h' & testten_min==.
}
replace testten_min=0 if sep_month_hat==`i'-36
gen testten_miss=1 if ppp!=0 & otherjob==1 & inss==1 & testten_min==.
egen testten_minmiss=max(testten_miss), by(id_worker)
egen ten36min=sum(testten_min), by(id_worker)
sum ten36min if inss==1 & a==1 & testten_minmiss!=1, detail

**2ab) calculate uper bound on accumulated tenure
cap drop ppp testten_miss
gen testten_max=tenure if inss==1 & a==1
replace testten_max=36 if testten_max>36 & inss==1 & a==1
gen ppp=0 if otherjob==1 & inss==1 
sum Njobs if inss==1 & a==1
local maxx=r(max)
forvalues h=1(1)`maxx' {
replace ppp=`h' if otherjob==1 & rankotherjob==Njob+(1-`h') & inss!=.
*start and end in same year (or censored in one year)
replace testten_max=tenure if ppp==`h' & hire_type!=8 & inss!=. & month_hat2>`i'-36-1 & tenure<=(sep_month_hat-hire_month_hat+1)
replace testten_max=sep_month_hat-(`i'-36-1) if ppp==`h' & hire_type!=8 & inss!=. & month_hat2<=`i'-36-1
replace testten_max=sep_month_hat-(`i'-36-1) if ppp==`h' & hire_type==8 & id_worker!=id_worker[_n-1] & inss!=. & month_hat2<=`i'-36-1
*start in previous year
replace testten_max=tenure if ppp==`h' & hire_type==8 & hire_type[_n-1]!=8 & id_worker==id_worker[_n-1] & inss!=. & month_hat2[_n-1]>`i'-36-1 & tenure<=(sep_month_hat-hire_month_hat[_n-1]+1)
replace testten_max=sep_month_hat-(`i'-36-1) if ppp==`h' & hire_type==8 & hire_type[_n-1]!=8 & id_worker==id_worker[_n-1] & inss!=. & month_hat2[_n-1]<=`i'-36-1
replace testten_max=sep_month_hat-(`i'-36-1) if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & id_worker==id_worker[_n-1] & id_worker!=id_worker[_n-2] & inss!=. & month_hat2[_n-1]<=`i'-36-1
*start in previous previous year
replace testten_max=tenure if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]!=8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & inss!=. & month_hat2[_n-2]>`i'-36-1 & tenure<=(sep_month_hat-hire_month_hat[_n-2]+1)
replace testten_max=sep_month_hat-(`i'-36-1) if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]!=8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & inss!=. & month_hat2[_n-2]<=`i'-36-1
replace testten_max=sep_month_hat-(`i'-36-1) if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]==8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & id_worker!=id_worker[_n-3] & inss!=. & month_hat2[_n-2]<=`i'-36-1
*start in previous previous previous year
replace testten_max=tenure if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]==8 & hire_type[_n-3]!=8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & id_worker==id_worker[_n-3] & inss!=. & month_hat2[_n-3]>`i'-36-1 & tenure<=(sep_month_hat-hire_month_hat[_n-3]+1)
replace testten_max=sep_month_hat-(`i'-36-1) if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]==8 & hire_type[_n-3]!=8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & id_worker==id_worker[_n-3] & inss!=. & month_hat2[_n-3]<=`i'-36-1
replace testten_max=sep_month_hat-(`i'-36-1) if ppp==`h' & hire_type==8 & hire_type[_n-1]==8 & hire_type[_n-2]==8 & hire_type[_n-3]==8 & id_worker==id_worker[_n-1] & id_worker==id_worker[_n-2] & id_worker==id_worker[_n-3] & id_worker!=id_worker[_n-4] & inss!=. & month_hat2[_n-3]<=`i'-36-1
*check
count if ppp==`h' & testten_max!=.
count if ppp==`h' & testten_max ==.
}
cap drop testten_miss
replace testten_max=0 if sep_month_hat==`i'-36
gen testten_miss=1 if ppp!=0 & otherjob==1 & inss==1 & testten_max ==.
egen testten_maxmiss=max(testten_miss), by(id_worker)
egen ten36max=sum(testten_max), by(id_worker)
sum ten36max if inss==1 & a==1 & testten_maxmiss!=1, detail

*identify people with accumulated tenure not equal to tenure in lost job
cap drop www
gen www=tenure if inss==1 & a==1
replace www=36 if testten_min>36 & inss==1 & a==1
gen otheracc=(www!=ten36min) if inss==1 & a==1
egen OTHERacc=max(otheracc), by(id_worker)

**3) recalculate time since last layoff associated with UI payment (useful for project using 16-month rule)

*identify possible layoff associated with past UI collection
gen ttt=sep_month_hat if sep_month_hat<`i' & reas_sep==11
egen previouslayoff=max(ttt), by(id_worker)
replace previouslayoff=. if (previouslayoff==0|previouslayoff==.) 
drop ttt
gen ttt=sep_month_hat if sep_month_hat<=lastprevious_whenreq & lastprevious_whenreq!=. &  lastprevious_firstpay!=.
egen previouslayoffUI=max(ttt), by(id_worker)
replace previouslayoffUI=. if (previouslayoffUI==0|previouslayoffUI==.) 
drop ttt

*time between last UI collection and eligible layoff
gen lastprevious_time=lastprevious_whenreq-previouslayoffUI if lastprevious_whenreq!=. &  lastprevious_firstpay!=.
gen ttt=tenure if sep_month_hat==previouslayoffUI
egen previouslayoff_tenure=min(ttt), by(id_worker)
drop ttt

*identify possible other job between last UI collection and this layoff
gen ttt=1 if sep_month_hat>lastprevious_whenreq & sep_month_hat<173 & lastprevious_whenreq!=. &  lastprevious_firstpay!=. & sep_month_hat!=0
replace ttt=. if ttt==0
egen betweenjobs=sum(ttt), by(id_worker)
drop ttt

**save info (only the observation of the layoff, relevant info on that line now)
keep if (a==1)
sort id_worker
save "sel_sample_all`i'_prematch_112014.dta", replace
}


********************************
**** C) Keep months of layoffs for which we have 3 full years of pre-layoff data and at least one year of post-layoff data
******** and create additional variable recording future layoffs around 16 months after this layoff (useful for project on RD using 16-month rule)
********************************

forvalues i=25(1)180 {

use "sel_sample_all`i'.dta", clear
duplicates drop

**1) re-do the basic sample selection and consistency checks to get where we can calculate key post-layoff variables for relevant sample (similar to Data_prep_A3)

*identify observation related to layoff of interest
gen a=(sep_month_hat==`i' & reas_sep==11)

*drop unnecessary variables
cap drop causafast1_fonte causafast3_fonte causafast2_fonte ceivinc indceivinc_fonte indpat_fonte muntrab_fonte nacionalidad_fonte portdefic_fonte qtdiasafas radiccnpj sbclas20_fonte tpdefic_fonte dtadmissao

*create continuous month of "hiring" and "separation" for observations with no separation or hiring in the year itself (indicate already employed at start of the year and still employed at end of the year)
gen month_hat=sep_month_hat
replace month_hat=12*(year-1995)+12.5 if sep_month==99 & reas_sep==0
gen month_hat2=hire_month_hat
replace month_hat2=12*(year-1995)+0.5 if hir_month==0 & hire_type==8

*drop if worker ID number inconsistent
drop if pisOK!=1 & year!=1994

*drop unnecessary variables
drop pisOK var1 _merge sampled

*we will not keep layoff with less than 1 month of tenure (never the ones we will study)
replace a=0 if month_hat==hire_month_hat & a==1

*keep layoffs with single job in month of layoff
sum year if a==1
local yyy=r(mean)
gen flag=(month_hat2<`i' & month_hat>=`i') if year==`yyy'
egen FLAG=sum(flag), by(id_worker)
egen a2=sum(a), by(id_worker)
keep if a2==1 & FLAG==1
drop a2 FLAG flag

*drop if establishment ID number inconsistent
replace a=0 if cnpjOK!=1

*keep history of layoffs of interest given above restrictions
egen a2=sum(a), by(id_worker)
keep if a2==1
drop a2

**2) keep only relevant post-layoff data for constructing new variable)
keep if (a==1)|((month_hat>=`i') & month_hat2>=`i')
cap drop n

*order by worker such that first you have the separation event and then each rehiring event
sort id_worker month_hat2 month_hat hire_month_hat

*index observations per worker
bysort id_worker: gen n=_n

*keep only jobs in next 5 years (should not matter anymore)
gen yearbase=year if n==1
egen yearb=max(yearbase), by(id_worker)
drop yearbase
gen yeardiff=year-yearb
keep if yeardiff==0|yeardiff==1|yeardiff==2 |yeardiff==3|yeardiff==4|yeardiff==5
sort id_worker month_hat2 month_hat hire_month_hat

*loop over 24 months after layoff (so we cover both below and above 16-month rule)
forvalues j=1(1)24 {
gen xxx=month_hat if month_hat==`i'+`j' & (reas_sep==11) & a!=1
gen yyy=tenure if xxx==month_hat
cap gen zzz=sep_day if xxx==month_hat 
egen fired_`j'months_after=max(xxx), by(id_worker)
egen tenure_`j'months_after=max(yyy), by(id_worker)
cap egen sepday_`j'months_after=max(zzz), by(id_worker)
drop xxx yyy zzz
}

*save info (only the observation of the layoff, relevant info on that line now)
keep if (a==1)
keep id_worker sep_month_hat fired_* tenure_* sepday_*

*merge onto dataset created above
sort id_worker
merge id_worker using  "sel_sample_all`i'_prematch_102016.dta", replace update _merge(mergetest)
save "sel_sample_all`i'_prematch_102016.dta", replace
}



