**********************************************************************
**********************************************************************
****CODE CREATING THE GRAPHS FOR THE EMPIRICAL APPLICATION
****This version: February, 2020

****Input : data_grr_QE.dta

**********************************************************************


********************************
**** A) setup
********************************

clear
set more off
version 12
timer on 1

cap log close
log using "Stata analysis/Codes for results/Create_graphs", replace

********************************
**** B) Create graphs for Figure 2 and Figure 6
********************************

insheet using "data_grr_QE_reg.csv", clear comma names

***aggregate variables by day
egen sample_size=count(dist_cut)
egen N_dist_cut=count(dist_cut), by(dist_cut)
replace N_dist_cut=N_dist_cut/sample_size
egen application_dist_cut=mean(application), by(dist_cut)
egen takeup_dist_cut=mean(treatment), by(dist_cut)
egen benefit_duration_dist_cut=mean(benefit_duration), by(dist_cut)
egen duration_censored_dist_cut=mean(duration_censored), by(dist_cut)
egen tag_dist_cut=tag(dist_cut)
foreach covariate in male age educ tenure logwage rrate commerce construction industry services largefirm mediumfirm microfirm {
	egen `covariate'_dist_cut=mean(`covariate'), by(dist_cut)
}


***create graphs for Figure 2
twoway(scatter N_dist_cut dist_cut, mcolor(gs7))(lpoly N_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly N_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black)) graphregion(color(white)) title("A. Density", color(black)) ytitle(Share of observations, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/density", replace
graph export "Stata analysis/Graphs/density.png", as(png) replace

twoway(scatter rrate_dist_cut dist_cut, mcolor(gs7))(lpoly rrate_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly rrate_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("B. Statutory UI replacement rate", color(black)) ytitle(Statutory UI replacement rate, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/statutoryUIrrate", replace
graph export "Stata analysis/Graphs/statutoryUIrrate.png", as(png) replace

twoway(scatter application_dist_cut dist_cut, mcolor(gs7))(lpoly application_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly application_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("C. Share applying for UI", color(black)) ytitle(Share applying for UI, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/application", replace
graph export "Stata analysis/Graphs/application.png", as(png) replace

twoway(scatter takeup_dist_cut dist_cut, mcolor(gs7))(lpoly takeup_dist_cut dist_cut if dist_cut<0,  lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly takeup_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))   graphregion(color(white)) title("D. Share drawing some UI benefits (takeup)", color(black)) ytitle(Share drawing some UI benefits, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/takeup", replace
graph export "Stata analysis/Graphs/takeup.png", as(png) replace
sum takeup_dist_cut if dist_cut==0
 
twoway(scatter benefit_duration_dist_cut dist_cut, mcolor(gs7))(lpoly benefit_duration_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly benefit_duration_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))   graphregion(color(white)) title("E. Average benefit duration", color(black)) ytitle(Average benefit duration (in months), size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/benefit_duration", replace
graph export "Stata analysis/Graphs/benefit_duration.png", as(png) replace
sum benefit_duration_dist_cut if dist_cut==0

twoway(scatter duration_censored_dist_cut dist_cut, mcolor(gs7))(lpoly duration_censored_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly duration_censored_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("F. Duration without a formal job (censored at two years)", color(black)) ytitle(Duration without a formal job (in days), size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/duration_censored", replace
graph export "Stata analysis/Graphs/duration_censored.png", as(png) replace
sum benefit_duration_dist_cut if dist_cut==-1
sum benefit_duration_dist_cut if dist_cut==0


***create graphs for Figure 6
twoway(scatter male_dist_cut dist_cut, mcolor(gs7))(lpoly male_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly male_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Share male", color(black)) ytitle(Share male, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/male", replace
graph export "Stata analysis/Graphs/male.png", as(png) replace

twoway(scatter age_dist_cut dist_cut, mcolor(gs7))(lpoly age_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly age_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Age", color(black)) ytitle(Age (in years), size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/age", replace
graph export "Stata analysis/Graphs/age.png", as(png) replace

twoway(scatter educ_dist_cut dist_cut, mcolor(gs7))(lpoly educ_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly educ_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Years of education", color(black)) ytitle(Years of education, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/educ", replace
graph export "Stata analysis/Graphs/educ.png", as(png) replace

twoway(scatter tenure_dist_cut dist_cut, mcolor(gs7))(lpoly tenure_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly tenure_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Tenure in the lost job", color(black)) ytitle(Tenure in the lost job (in months), size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/tenure", replace
graph export "Stata analysis/Graphs/tenure.png", as(png) replace

twoway(scatter logwage_dist_cut dist_cut, mcolor(gs7))(lpoly logwage_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly logwage_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Log wage in the lost job", color(black)) ytitle(Log wage in the lost job (reais), size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/logwage", replace
graph export "Stata analysis/Graphs/logwage.png", as(png) replace

twoway(scatter rrate_dist_cut dist_cut, mcolor(gs7))(lpoly rrate_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly rrate_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Statutory UI replacement rate", color(black)) ytitle(Statutory UI replacement rate, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/rrate", replace
graph export "Stata analysis/Graphs/rrate.png", as(png) replace

twoway(scatter commerce_dist_cut dist_cut, mcolor(gs7))(lpoly commerce_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly commerce_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Share from commercial sector", color(black)) ytitle(Share from commercial sector, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/commerce", replace
graph export "Stata analysis/Graphs/commerce.png", as(png) replace

twoway(scatter construction_dist_cut dist_cut, mcolor(gs7))(lpoly construction_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly construction_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Share from construction sector", color(black)) ytitle(Share from construction sector, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/construction", replace
graph export "Stata analysis/Graphs/construction.png", as(png) replace

twoway(scatter industry_dist_cut dist_cut, mcolor(gs7))(lpoly industry_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly industry_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Share from industrial sector", color(black)) ytitle(Share from industrial sector, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/industry", replace
graph export "Stata analysis/Graphs/industry.png", as(png) replace

twoway(scatter services_dist_cut dist_cut, mcolor(gs7))(lpoly services_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly services_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Share from service sector", color(black)) ytitle(Share from service sector, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/services", replace
graph export "Stata analysis/Graphs/services.png", as(png) replace

twoway(scatter largefirm_dist_cut dist_cut, mcolor(gs7))(lpoly largefirm_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly largefirm_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Share from large firms", color(black)) ytitle(Share from large firms, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/largefirm", replace
graph export "Stata analysis/Graphs/largefirm.png", as(png) replace

twoway(scatter mediumfirm_dist_cut dist_cut, mcolor(gs7))(lpoly mediumfirm_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly mediumfirm_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Share from medium-size firms", color(black)) ytitle(Share from medium-size firms, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/mediumfirm", replace
graph export "Stata analysis/Graphs/mediumfirm.png", as(png) replace

twoway(scatter microfirm_dist_cut dist_cut, mcolor(gs7))(lpoly microfirm_dist_cut dist_cut if dist_cut<0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1))(lpoly microfirm_dist_cut dist_cut if dist_cut>=0, lcolor(black) lw(thick) kernel(triangle) bwidth(30) degree(1)) if tag_dist_cut==1 & dist_cut>=-50  & dist_cut<=50, xline(0, lcolor(black))  graphregion(color(white)) title("Share from small firms", color(black)) ytitle(Share from small firms, size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Difference in days between layoff and eligibility dates, size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/microfirm", replace
graph export "Stata analysis/Graphs/microfirm.png", as(png) replace


********************************
**** C) Create graphs for Figure 4
********************************
count if dist_cut>=-30 & dist_cut<30
histogram duration_censored if dist_cut>=0 & dist_cut<30, color(black) fintensity(inten60) percent  graphregion(color(white)) title("Right of the eligibility cutoff", color(black)) ytitle(Percent of observations, size(medlarge)) xscale(range(0 750)) xlabel(0(50)750, labsize(medlarge)) ylabel(0(5)30, labsize(medlarge)) xtitle("Duration without a formal job (censored at 24 months)", size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/duration_censored_hist_right", replace
graph export "Stata analysis/Graphs/duration_censored_hist_right.png", as(png) replace

histogram duration_censored if dist_cut<0 & dist_cut>=-30, color(black) fintensity(inten60) percent  graphregion(color(white)) title("Left of the eligibility cutoff", color(black)) ytitle(Percent of observations, size(medlarge)) xscale(range(0 750)) xlabel(0(50)750, labsize(medlarge)) ylabel(0(5)30, labsize(medlarge)) xtitle("Duration without a formal job (censored at 24 months)", size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/duration_censored_hist_left", replace
graph export "Stata analysis/Graphs/duration_censored_hist_left.png", as(png) replace


********************************
**** D) Create graphs for Figure 5
********************************
sort avg_w_mw
twoway(line uibenefit avg_w_mw, lcolor(black) lw(thick)) if dist_cut>=-50  & dist_cut<=50 & avg_w_mw>=1 & avg_w_mw<=6,  xline(1.25 1.65 2.75, lcolor(black) lpattern(longdash)) graphregion(color(white)) ytitle(Benefit amount (in minimum wage), size(medlarge)) xlabel(, labsize(medlarge)) ylabel(, labsize(medlarge)) xtitle(Average wage in the 3 months prior to layoff (in minimum wage), size(medlarge)) legend(off)
graph save "Stata analysis/Graphs/UIbenefit_schedule", replace
graph export "Stata analysis/Graphs/UIbenefit_schedule.png", as(png) replace
