**********************************************************************
**********************************************************************
****CODE CREATING THE GRAPHS FOR FIGURE 3 BASED ON RESULTS FROM R ANALYSIS
****This version: February, 2020

****Input : 
*rdbounds24_fixedtau_fuzzy_QTE50.csv
*rdbounds24_fixedtau_fuzzy_ATE.csv
**********************************************************************

********************************
**** A) setup
********************************

clear
set more off
version 12
timer on 1

cap log close
log using "R analysis/Codes for results/Create_figure3", replace

********************************
**** B) Create left panel for Figure 3
********************************

insheet using "R analysis/Results/rdbounds24_fixedtau_fuzzy_ATE.csv", clear comma names

*add results ignoring manipulation (potential_taus=0)
count 
local NNN=r(N)+1
set obs `NNN'
replace potential_taus=0 if _n==6
replace te_lower=87.7 if _n==6
replace te_upper=87.7 if _n==6
replace te_frd_cis_manipulation_lower=79.1 if _n==6
replace te_frd_cis_manipulation_upper=96.2 if _n==6

sort potential_taus
twoway(connected te_upper potential_taus, lcolor(black) mcolor(black))(connected te_lower potential_taus, lcolor(black) mcolor(black)) (line te_frd_cis_manipulation_lower potential_taus, lpattern(dash) lcolor(black) )(line te_frd_cis_manipulation_upper potential_taus, lpattern(dash) lcolor(black) ), xline(0.064, lcolor(black)) xline(0.038 0.089, lcolor(black) lpattern(dash))   graphregion(color(white)) title("A. Average Treatment Effect (FRD)", color(black)) subtitle("(Fixed-manipulation inference; outcome censored at 24 months)", color(black)) ytitle("Average treatment effect", size(medlarge)) xlabel(0 .025 .05 .064 .1 .2, labsize(medlarge)) ylabel(-100 -50 0 50 100 150 200, labsize(medlarge)) xtitle("Hypothetical share of always-assigned units", size(medlarge)) legend(off)
graph save "R analysis/Results/rdbounds24_fixedtau_fuzzy_ATE", replace
graph export "R analysis/Results/rdbounds24_fixedtau_fuzzy_ATE.png", as(png) replace


********************************
**** C) Create right panel for Figure 3
********************************

insheet using "R analysis/Results/rdbounds24_fixedtau_fuzzy_QTE50.csv", clear comma names

*add results ignoring manipulation (potential_taus=0)
count 
local NNN=r(N)+1
set obs `NNN'
replace potential_taus=0 if _n==6
replace te_lower=99 if _n==6
replace te_upper=99 if _n==6
replace te_frd_cis_manipulation_lower=91.0 if _n==6
replace te_frd_cis_manipulation_upper=107.0 if _n==6

sort potential_taus
twoway(connected te_upper potential_taus, lcolor(black) mcolor(black))(connected te_lower potential_taus, lcolor(black) mcolor(black)) (line te_frd_cis_manipulation_lower potential_taus, lpattern(dash) lcolor(black) )(line te_frd_cis_manipulation_upper potential_taus, lpattern(dash) lcolor(black) ), xline(0.064, lcolor(black)) xline(0.038 0.089, lcolor(black) lpattern(dash))   graphregion(color(white)) title("B. Median Treatment Effect (FRD)", color(black)) subtitle("(Fixed-manipulation inference; outcome censored at 24 months)", color(black)) ytitle("Median treatment effect", size(medlarge)) xlabel(0 .025 .05 .064 .1 .2, labsize(medlarge)) ylabel(-100 -50 0 50 100 150 200, labsize(medlarge)) xtitle("Hypothetical share of always-assigned units", size(medlarge)) legend(off)
graph save "R analysis/Results/rdbounds24_fixedtau_fuzzy_QTE50", replace
graph export "R analysis/Results/rdbounds24_fixedtau_fuzzy_QTE50.png", as(png) replace

