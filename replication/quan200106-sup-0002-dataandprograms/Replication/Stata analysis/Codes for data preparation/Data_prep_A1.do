**********************************************************************
**********************************************************************
****CODE SELECTING DISPLACED FORMAL WORKERS IN MATCHED EMPLOYEE-EMPLOYER DATA
****This version: February, 2020

****Input: `year'/`year'_`i'done.dta (multiple files `i' per year `year')

****Output: `year'sample.dta (one file per year `year' of layoff)
**********************************************************************


********************************
**** A) Setup
********************************

clear
set more off
set mem 5000m


********************************
**** B) Load raw data from 1995 to 2010 and keep displaced formal workers of interest (one year at a time)
********************************

***1995
forvalues year=1995(1)1995 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)11 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***1996
forvalues year=1996(1)1996 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)12 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***1997
forvalues year=1997(1)1997 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)11 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***1998
forvalues year=1998(1)1998 {
clear
gen var=1
save "year'sample.dta", replace

forvalues i=1(1)11 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***1999
forvalues year=1999(1)1999 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)8 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***2000
forvalues year=2000(1)2000 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)8 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***2001
forvalues year=2001(1)2001 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)13 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***2002
forvalues year=2002(1)2002 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)13 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***2003
forvalues year=2003(1)2003 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)13 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***2004
forvalues year=2004(1)2004 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)14 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***2005
forvalues year=2005(1)2005 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)15 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***2006
forvalues year=2006(1)2006 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)19 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}



***2007
forvalues year=2007(1)2007 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)20 {
use "`year'/`year'_`i'done.dta", clear

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***2008
forvalues year=2008(1)2008 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)5 {
use "`year'/`year'a_`i'done.dta", clear // files indexed by "a" are for spells that ended during the year

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***2009
forvalues year=2009(1)2009 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)4 {
use "`year'/`year'a_`i'done.dta", clear // files indexed by "a" are for spells that ended during the year

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}


***2010
forvalues year=2010(1)2010 {
clear
gen var=1
save "`year'sample.dta", replace

forvalues i=1(1)5 {
use "`year'/`year'a_`i'done.dta", clear // files indexed by "a" are for spells that ended during the year

*create municipality code
gen muni_code=state*10000+munic

*rearrange separation month and hiring month variables within a year
replace sep_month=99 if reas_sep==0
gen hir_month=hire_month
replace hir_month=0 if hire_type==8

*create continuous hiring month and separation month variables
gen hire_month_hat = 0
replace hire_month_hat = 12*(year-1995) + hir_month if hir_month~=0
gen sep_month_hat = 0
replace sep_month_hat = 12*(year-1995) + sep_month if sep_month~=99

*drop public sector, state-owned firms, mixed-ownership firms, and Itaipu
drop if nat_estb <= 2038

*drop international organizations and unincorporated businesses
drop if nat_estb >= 4000

*drop cases with missing information on nature of firm
drop if nat_estb == . 

*keep standard undetermined-lenght contracts (CLT)
keep if contr_type == 10

*keep working-age workers
drop if age<=18 | age>=55

*drop firms in agriculture, public utilities, public administration, or unknown
drop if agg_sector==1101| agg_sector==4618| agg_sector==5719|agg_sector==9999

*keep workers with standard full-time contracts
drop if hours<30 | hours>50

*keep displaced formal workers
keep if reas_sep==11

*keep selected variables for merging in future codes
gen sampled=1
keep id_worker muni_code sampled sep_month_hat hire_month_hat tenure
rename sep_month_hat this_sep_month_hat 
rename hire_month_hat this_hire_month_hat
rename muni_code this_muni_code
rename tenure this_tenure

*append to yearly file
compress
append using "`year'sample.dta"
sort id_worker this_hire_month_hat this_sep_month_hat this_tenure
save  "`year'sample.dta", replace
}

*clean yearly file
drop var
compress
save  "`year'sample.dta", replace
}
