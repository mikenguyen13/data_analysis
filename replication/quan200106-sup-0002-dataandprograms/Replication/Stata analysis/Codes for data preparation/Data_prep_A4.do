**********************************************************************
**********************************************************************
****CODE MATCHING UI DATA TO SAMPLE OF DISPLACED WORKERS AND PREPARING UI DATA FOR ANALYSIS
****This version: February, 2020

****Input: 
*`year'/ui_req_`year'_cp.dta (one file per year `year' of UI application)
*`year'/ui_pag_`year'_cp.dta (one file per year `year' of UI payment)
*sel_sample_all`i'_onlyID.dta (one file per month `i' of layoff)

****Output: 
*sel_sample_all`i'_onlyID_match_req.dta (one file per month `i' of layoff)
*sel_sample_all`i'_onlyID_match_pag.dta (one file per month `i' of layoff)
*sel_sample_all`i'_onlyID_match_all.dta (one file per month `i' of layoff)
*sel_sample_all`i'_onlyID_match_this.dta (one file per month `i' of layoff)
*sel_sample_all`i'_onlyID_match_this_short.dta (one file per month `i' of layoff)
**********************************************************************

********************************
**** A) Setup
********************************

clear
set more off
set mem 10000m


********************************
**** B) Match UI data for 3 years before and 5 years after layoff with sample of displaced formal workers
********************************

*create empty datasets to host UI data matched to displaced formal workers
gen var=1
forvalues i=1(1)192 {
save "sel_sample_all`i'_onlyID_match_req.dta", replace
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}

***match the data for year of layoff and 5 years after layoff for 1995
forvalues i=1(1)12 {
forvalues year=1995(1)2000 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 1 year before layoff and 5 years after layoff for 1996
forvalues i=13(1)24 {
forvalues year=1995(1)2001 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 2 years before layoff and 5 years after layoff for 1997
forvalues i=25(1)36 {
forvalues year=1995(1)2002 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 5 years after layoff for 1998
forvalues i=37(1)48 {
forvalues year=1995(1)2003 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 5 years after layoff for 1999
forvalues i=49(1)60 {
forvalues year=1996(1)2004 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 5 years after layoff for 2000
forvalues i=61(1)72 {
forvalues year=1997(1)2005 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 5 years after layoff for 2001
forvalues i=73(1)84 {
forvalues year=1998(1)2006 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 5 years after layoff for 2002
forvalues i=85(1)96 {
forvalues year=1999(1)2007 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 5 years after layoff for 2003
forvalues i=97(1)108 {
forvalues year=2000(1)2008 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 5 years after layoff for 2004
forvalues i=109(1)120 {
forvalues year=2001(1)2010 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 5 years after layoff for 2005
forvalues i=121(1)132 {
forvalues year=2002(1)2010 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 4 years after layoff for 2006
forvalues i=133(1)144 {
forvalues year=2003(1)2010 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 3 years after layoff for 2007
forvalues i=145(1)156 {
forvalues year=2004(1)2010 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 2 years after layoff for 2008
forvalues i=157(1)168 {
forvalues year=2005(1)2010 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and 1 year after layoff for 2009
forvalues i=169(1)180 {
forvalues year=2006(1)2010 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}

***match the data for 3 years before layoff and year of layoff for 2009
forvalues i=181(1)192 {
forvalues year=2007(1)2010 {
*application
use  "`year'/ui_req_`year'_cp.dta", clear
drop cpf genero schooling ocupacao setor
rename  n_req_req num_req_req
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_req.dta"
save "sel_sample_all`i'_onlyID_match_req.dta", replace
*payment
use  "`year'/ui_pag_`year'_cp.dta", clear
rename  situacao status_payment
label copy situacao status_payment
label drop situacao
label values status_payment status_payment
sort id_worker
merge m:1 id_worker using "sel_sample_all`i'_onlyID.dta", keep(match)
append using "sel_sample_all`i'_onlyID_match_pag.dta"
save "sel_sample_all`i'_onlyID_match_pag.dta", replace
}
}


********************************
**** B) Clean matched UI application data
********************************

forvalues i=1(1)192 {

***sort payment data for merging later on with application data
use "sel_sample_all`i'_onlyID_match_pag.dta", clear
rename num_req_pag num_req
cap drop _merge
sort id_worker num_req
save "sel_sample_all`i'_onlyID_match_pag.dta", replace

***clean application data before matching to payment data
use "sel_sample_all`i'_onlyID_match_req.dta", clear
cap drop _merge
duplicates drop 

*variables keeping track of relevant dates in UI application data
foreach var in adm dem req seg ben {
gen date_`var'=(ano_`var'-1995)*12+mes_`var'
gen day_`var'=dia_`var'
}

*check for multiple observations with same application number
sort id_worker num_req_req 
bysort id_worker num_req_req: gen N=_N
tab N

*keep if observations from later in process (segurado or beneficiario date)
gen x=(date_seg>0)+(date_ben>0)
egen X=max(x), by(id_worker num_req_req)
keep if x==X
drop N x X
bysort id_worker num_req_req: gen N=_N
tab N

*keep if observations from later in process (habilitado vs notificado)
egen X=max(status), by(id_worker num_req_req)
keep if status==X
drop X

*keep if observations from later in process (benef vs segurado vs requer)
egen X=min(situacao), by(id_worker num_req_req)
keep if situacao==X
drop X

*if everything else the same, pick earlier application
egen X=min(date_req), by(id_worker num_req_req)
keep if date_req==X
drop X N

*if everything else the same, pick one of the remaining observation
bysort id_worker num_req_req : gen N=_N
bysort id_worker num_req_req : gen n=_n

*keep key variables
keep id_worker num_req_req uf_demissao uf_atend situacao status sal_ultimo sal_antepe sal_penult date_adm day_adm date_dem day_dem date_req day_req date_seg day_seg date_ben day_ben issue_req

*merge with payment data
rename num_req_req num_req
sort id_worker num_req
merge 1:m id_worker num_req using "sel_sample_all`i'_onlyID_match_pag.dta", keep(match)
save "sel_sample_all`i'_onlyID_match_all.dta", replace
}


********************************
**** C) Clean matched UI application and payment data
********************************

forvalues i=1(1)192 {

use "sel_sample_all`i'_onlyID_match_all.dta", clear

*variables keeping track of relevant dates in UI payment data
foreach var in emiss pag {
gen date_`var'=(ano_`var'-1995)*12+mes_`var'
gen day_`var'=dia_`var'
}

*** organize data such that one line per application (so prepare for reshape)
*sequencial cannot have missing value for reshape
replace sequencial=0 if sequencial==.

*no variable can be named _merge for reshape
rename _merge merge_reg_pag

*sequencial must be unique for reshape
duplicates drop
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X

*sometimes when additional benefit, time difference between application/emission and payment
*then one line for application and another one when payment actually made
bysort id_worker num_req sequencial: gen N=_N
bysort id_worker num_req sequencial: gen n=_n
gen xxx=1 if tipo==3 & status_payment==2 & N!=1
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if tipo==3 & status_payment==1 & N!=1
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen delay_add=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if delay_add==1 & tipo==3 & status_payment==2 & N!=1

*sometimes, time difference between application/emission and payment for regular payment
*then one line for application and another one when payment actually made
drop x X n N
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X
bysort id_worker num_req sequencial: gen N=_N
bysort id_worker num_req sequencial: gen n=_n
gen xxx=1 if tipo==2 & status_payment==2 & N!=1
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if tipo==2 & status_payment==1 & N!=1
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen delay_norm=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if delay_norm==1 & tipo==2 & status_payment==2 & N!=1

*sometimes other issue with a payment not made
drop x X n N
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X
bysort id_worker num_req sequencial: gen N=_N
bysort id_worker num_req sequencial: gen n=_n
gen xxx=1 if status_payment!=1 & N!=1
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if status_payment==1 & N!=1
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen issue_notpaid=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if issue_notpaid==1 & status_payment!=1 & N!=1

*sometimes one line with a payment not made that seems to belong to other application (later, other amount)
drop x X n N
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X
bysort id_worker num_req sequencial: gen N=_N
bysort id_worker num_req sequencial: gen n=_n
egen yyy=median(valor_emitido), by(id_worker num_req)
replace yyy=int(yyy)
gen xxx=1 if status_payment!=1 & N!=1 & int(valor_emitido)!=yyy
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if status_payment!=1 & N!=1 & int(valor_emitido)==yyy
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen issue_notbelong=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if issue_notbelong==1 & status_payment!=1 & N!=1 & int(valor_emitido)!=yyy
drop yyy

*sometimes several lines of payments that seem to belong to another application (later, other amount): keep one closer to application date
drop x X n N
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X
bysort id_worker num_req sequencial: gen N=_N
bysort id_worker num_req sequencial: gen n=_n
gen timediff=date_emiss-date_req
egen mindiff=min(timediff), by(id_worker num_req sequencial)
gen xxx=1 if N!=1 & timediff!=mindiff
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if N!=1 & timediff==mindiff
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen issue_mult=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if issue_mult==1 & N!=1 & timediff!=mindiff
drop timediff mindiff

*sometimes different lines for a parcela at different stages. Keep the most advanced stage
drop x X n N
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X
bysort id_worker num_req sequencial: gen N=_N
bysort id_worker num_req sequencial: gen n=_n
egen mindate_emiss=min(date_emiss), by(id_worker num_req sequencial)
egen minstat=min(status_payment), by(id_worker num_req sequencial)
gen xxx=1 if N!=1 & mindate_emiss==date_emiss & minstat!=status_payment
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if N!=1 & mindate_emiss==date_emiss & minstat==status_payment
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen issue_stages=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if issue_stages==1 & N!=1 & mindate_emiss==date_emiss & minstat!=status_payment
drop mindate_emiss minstat

*sometimes additional payment recorded with same sequential as last normal payment if in same month
*add one level of sequencial 
drop x X n N
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X
bysort id_worker num_req sequencial: gen N=_N
bysort id_worker num_req sequencial: gen n=_n
egen maxseq=max(sequencial), by(id_worker num_req)
egen mindate_emiss=min(date_emiss), by(id_worker num_req sequencial)
egen minday_emiss=min(day_emiss), by(id_worker num_req sequencial)
gen xxx=1 if N!=1 & sequencial==maxseq & mindate_emiss==date_emiss & day_emiss>minday_emiss
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if N!=1 & sequencial==maxseq & mindate_emiss==date_emiss & day_emiss==minday_emiss
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen issue_seqwrong=issue_add==1 & OK==1 & N!=1
drop issue_add OK
replace sequencial=sequencial+1 if issue_seqwrong==1 & N!=1 & sequencial==maxseq & mindate_emiss==date_emiss & day_emiss>minday_emiss
drop maxseq mindate_emiss minday_emiss

*data not matched to application: sometimes several lines of payments that seem to belong to another application (later, other amount)
*keep earlier date of emission
drop x X n N
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X
bysort id_worker num_req sequencial: gen N=_N
bysort id_worker num_req sequencial: gen n=_n
egen minemiss=min(date_emiss), by(id_worker num_req sequencial)
gen xxx=1 if N!=1 & date_emiss!=minemiss
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if N!=1 & date_emiss==minemiss
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen issue_mult_notmatched=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if issue_mult_notmatched==1 & N!=1 & date_emiss!=minemiss
drop minemiss 

*sometimes different lines for a parcela seem to be a repetition even if different day within month (not for paid benefits)
*keep earlier one
drop x X n N
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X
bysort id_worker num_req sequencial: gen N=_N
bysort id_worker num_req sequencial: gen n=_n
egen mindate_emiss=min(date_emiss), by(id_worker num_req sequencial)
egen minday_emiss=min(day_emiss), by(id_worker num_req sequencial)
egen minstat=min(status_payment), by(id_worker num_req sequencial)
egen mintipo=min(tipo), by(id_worker num_req sequencial)
gen xxx=1 if N!=1 & mindate_emiss==date_emiss & minstat==status_payment & minstat!=1 & mintipo==tipo & day_emiss>minday_emiss
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if N!=1 & mindate_emiss==date_emiss & minstat==status_payment
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen issue_rep_notpaid=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if issue_rep_notpaid==1 & N!=1 & mindate_emiss==date_emiss & minstat==status_payment & minstat!=1 & mintipo==tipo & day_emiss>minday_emiss
drop  mindate_emiss minday_emiss minstat mintipo

*sometimes issues with addicional payments with same sequential than other
*keep either closer value to other payments or normal payment if same sequencial and date
drop x X n N
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X
bysort id_worker num_req sequencial: gen N=_N
bysort id_worker num_req sequencial: gen n=_n
egen mindate_emiss=min(date_emiss), by(id_worker num_req sequencial)
egen minday_emiss=min(day_emiss), by(id_worker num_req sequencial)
egen minstat=min(status_payment), by(id_worker num_req sequencial)
egen mintipo=min(tipo), by(id_worker num_req sequencial)
egen yyy=median(valor_emitido), by(id_worker num_req)
replace yyy=int(yyy)
gen diffpay=int(valor_emitido)-yyy
egen mindiffpay=min(diffpay), by(id_worker num_req sequencial)
gen xxx=1 if N!=1 & mindate_emiss==date_emiss & day_emiss==minday_emiss & minstat==status_payment & mintipo==tipo & diffpay!=mindiffpay
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if N!=1 & mindate_emiss==date_emiss & day_emiss==minday_emiss & minstat==status_payment & mintipo==tipo & diffpay==mindiffpay
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen issue_otheradd=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if issue_otheradd==1 & N!=1 & mindate_emiss==date_emiss & day_emiss==minday_emiss & minstat==status_payment & mintipo==tipo & diffpay!=mindiffpay
*
gen xxx=1 if N!=1 & mindate_emiss==date_emiss & minstat==status_payment & int(valor_emitido)==yyy & mintipo!=tipo
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if N!=1 & mindate_emiss==date_emiss & minstat==status_payment & int(valor_emitido)==yyy & mintipo==tipo
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
replace issue_otheradd=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if N!=1 & mindate_emiss==date_emiss & minstat==status_payment & int(valor_emitido)==yyy & mintipo!=tipo
*
gen xxx=1 if N!=1 & mindate_emiss==date_emiss & minstat==status_payment & int(valor_emitido)!=yyy & mintipo!=tipo
egen issue_add=max(xxx), by(id_worker num_req sequencial)
drop xxx
gen xxx=1 if N!=1 & mindate_emiss==date_emiss & minstat==status_payment & int(valor_emitido)==yyy & mintipo==tipo
egen OK=max(xxx), by(id_worker num_req sequencial)
drop xxx
replace issue_otheradd=issue_add==1 & OK==1 & N!=1
drop issue_add OK
drop if N!=1 & mindate_emiss==date_emiss & minstat==status_payment & int(valor_emitido)!=yyy & mintipo!=tipo
drop mindate_emiss minday_emiss minstat mintipo yyy diffpay mindiffpay


*check if still issues
drop x X n N
sort id_worker num_req sequencial
gen x=1  if id_worker==id_worker[_n-1] & num_req==num_req[_n-1] & sequencial==sequencial[_n-1]
egen X=max(x), by(id_worker num_req )
tab X
drop x X

*drop unnecessary variables
drop  dia_emissao mes_emissao ano_emissao dia_pagam mes_pagam ano_pagam year var

*keep record of few data issues
gen issue_pag=1 if delay_add==1
replace issue_pag=2 if delay_norm==1
replace issue_pag=3 if issue_notpaid==1
replace issue_pag=4 if issue_notbelong==1
replace issue_pag=5 if issue_mult==1
replace issue_pag=6 if issue_stages==1
replace issue_pag=7 if issue_seqwrong==1
replace issue_pag=8 if issue_mult_notmatched==1
replace issue_pag=9 if issue_rep_notpaid==1
replace issue_pag=10 if issue_otheradd==1

*drop unnecessary variables
drop delay_add delay_norm issue_notpaid issue_notbelong issue_mult issue_mult_notmatched issue_seqwrong issue_stages issue_rep_notpaid issue_otheradd

*reshare data such that one line per UI application
reshape wide status_payment valor_emitido valor_pago tipo date_emiss day_emiss date_pag day_pag issue_pag, i(id_worker num_req) j(sequencial)

save "sel_sample_all`i'_onlyID_match_all.dta", replace
}


********************************
**** D) Create variables of interest for data analysis
********************************

forvalues i=1(1)192 {
use  "sel_sample_all`i'_onlyID_match_all.dta", clear

*drop unnecessary variables
drop status_payment0- issue_pag0

*merge with duration outcomes from matched employee-employed data
sort id_worker date_req date_seg date_ben date_dem 
merge id_worker using  "DDduration_all`i'.dta"
tab _merge
drop if _merge!=3
drop _merge

*identify different applications with same date_req date_seg date_ben date_dem and keep only one (with payment if any) 
bysort id_worker date_req date_seg date_ben date_dem: gen XXX=_N
gen thiskeep=1 if status_payment1!=. & status_payment1[_n-1]==. & XXX!=1 & id_worker==id_worker[_n-1] & date_req==date_req[_n-1] & date_seg==date_seg[_n-1] & date_ben==date_ben[_n-1] & date_dem==date_dem[_n-1]
replace thiskeep=1 if status_payment1!=. & status_payment1[_n+1]==. & XXX!=1 & id_worker==id_worker[_n+1] & date_req==date_req[_n+1] & date_seg==date_seg[_n+1] & date_ben==date_ben[_n+1] & date_dem==date_dem[_n+1]
gen nokeep=1 if status_payment1==. & status_payment1[_n-1]==. & XXX!=1 & id_worker==id_worker[_n-1] & date_req==date_req[_n-1] & date_seg==date_seg[_n-1] & date_ben==date_ben[_n-1] & date_dem==date_dem[_n-1]
egen minthiskeep=min(_n) if thiskeep==1, by(id_worker date_req date_seg date_ben date_dem)
egen minnokeep=min(_n) if nokeep==1, by(id_worker date_req date_seg date_ben date_dem)
replace thiskeep=. if thiskeep==1 & _n!=minthiskeep
replace nokeep=. if nokeep==1 & _n!=minnokeep
drop if XXX!=1 & thiskeep==. & nokeep==.
gen same_application=1 if XXX!=1
drop XXX thiskeep nokeep minnokeep minthiskeep
bysort id_worker date_req date_seg date_ben date_dem: gen XXX=_N
tab XXX
drop XXX

*identify cases where issue in consecutive sequencial
gen issue_seq_pag=.
forvalues j=2(1)7 {
local low=`j'-1
replace issue_seq_pag=1 if status_payment`j'!=. & status_payment`low'==. & merge==3
}
tab issue_seq_pag if merge==3, missing

*record last time you collected prior to layoff month
sort id_worker date_req date_seg date_ben date_dem 
egen last=max(_n) if date_req<`i' & date_req!=., by(id_worker)
gen xxx=date_req if _n==last
egen lastprevious_whenreq=max(xxx), by(id_worker)
drop xxx
gen xxx=date_dem if _n==last
egen lastprevious_whendem=max(xxx), by(id_worker)
drop xxx
gen first_pay=.
gen last_pay=.
gen npay=0 if last==_n
forvalues j=1(1)10 {
cap replace first_pay=date_pag`j' if valor_pago`j'!=0 & valor_pago`j'!=. & first_pay==. & last==_n
cap replace last_pay=date_pag`j' if valor_pago`j'!=0 & valor_pago`j'!=. & last==_n
cap replace npay=npay+1 if valor_pago`j'!=0 & valor_pago`j'!=. & last==_n
}
egen lastprevious_firstpay=max(first_pay), by(id_worker)
egen lastprevious_lastpay=max(last_pay), by(id_worker)
egen lastprevious_npay=max(npay), by(id_worker)
drop first_pay last_pay npay

*record number of times you applied prior to layoff month in last 3 years
gen xxx=1 if date_req<`i' & date_req>=`i'-36 & date_req!=.
egen last_napplied=count(xxx), by(id_worker)
drop xxx
gen xxx=.
forvalues j=1(1)10 {
cap replace xxx=1 if date_pag`j'<`i'  & date_pag`j'>=`i'-36 & date_pag`j'!=. & valor_pago`j'!=0 & valor_pago`j'!=. & xxx==.
} 
egen last_nben=count(xxx), by(id_worker)
drop xxx
gen xxx=0
forvalues j=1(1)10 {
cap replace xxx=xxx+1 if date_pag`j'<`i'  & date_pag`j'>=`i'-36 & date_pag`j'!=. & valor_pago`j'!=0 & valor_pago`j'!=.
} 
egen last_ntotalben=sum(xxx), by(id_worker)
drop xxx

*record most recent time you collected after layoff month
sort id_worker date_req date_seg date_ben date_dem 
egen this=min(_n) if date_req>=`i'  & date_req!=., by(id_worker)

*record next most recent time you collected after layoff month
sort id_worker date_req date_seg date_ben date_dem 
egen next=min(_n) if date_req>=`i'  & date_req!=. & _n!=this, by(id_worker)
gen xxx=date_req if _n==next
egen firstnext_whenreq=max(xxx), by(id_worker)
drop xxx
gen xxx=date_dem if _n==next
egen firstnext_whendem=max(xxx), by(id_worker)
drop xxx
gen first_pay=.
gen last_pay=.
gen npay=0 if next==_n
forvalues j=1(1)10 {
cap replace first_pay=date_pag`j' if valor_pago`j'!=0 & valor_pago`j'!=. & first_pay==. & next==_n
cap replace last_pay=date_pag`j' if valor_pago`j'!=0 & valor_pago`j'!=. & next==_n
cap replace npay=npay+1 if valor_pago`j'!=0 & valor_pago`j'!=. & next==_n
}
egen firstnext_firstpay=max(first_pay), by(id_worker)
egen firstnext_lastpay=max(last_pay), by(id_worker)
egen firstnext_npay=max(npay), by(id_worker)
drop first_pay last_pay npay

*same but must be after return to formal sector
sort id_worker date_req date_seg date_ben date_dem 
egen next_after=min(_n) if date_req>=`i'  & date_req!=. & _n!=this & date_req>`i'+duration, by(id_worker)
gen xxx=date_req if _n==next_after
egen firstnext_after_whenreq=max(xxx), by(id_worker)
drop xxx
gen xxx=date_dem if _n==next_after
egen firstnext_after_whendem=max(xxx), by(id_worker)
drop xxx
gen first_pay=.
gen last_pay=.
gen npay=0 if next==_n
forvalues j=1(1)10 {
cap replace first_pay=date_pag`j' if valor_pago`j'!=0 & valor_pago`j'!=. & first_pay==. & next_after==_n
cap replace last_pay=date_pag`j' if valor_pago`j'!=0 & valor_pago`j'!=. & next_after==_n
cap replace npay=npay+1 if valor_pago`j'!=0 & valor_pago`j'!=. & next_after==_n
}
egen firstnext_after_firstpay=max(first_pay), by(id_worker)
egen firstnext_after_lastpay=max(last_pay), by(id_worker)
egen firstnext_after_npay=max(npay), by(id_worker)
drop first_pay last_pay npay

*record number of times you applied after layoff month in next 2 years
gen xxx=1 if date_req>=`i' & date_req<=`i'+24 & date_req!=.
egen next_napplied_24=count(xxx), by(id_worker)
drop xxx
gen xxx=.
forvalues j=1(1)10 {
cap replace xxx=1 if date_pag`j'>=`i'  & date_pag`j'<=`i'+24 & date_pag`j'!=. & valor_pago`j'!=0 & valor_pago`j'!=. & xxx==.
} 
egen next_nben_24=count(xxx), by(id_worker)
drop xxx
gen xxx=0
forvalues j=1(1)10 {
cap replace xxx=xxx+1 if date_pag`j'>=`i'  & date_pag`j'<=`i'+24 & date_pag`j'!=. & valor_pago`j'!=0 & valor_pago`j'!=.
} 
egen next_ntotalben_24=sum(xxx), by(id_worker)
drop xxx

*record number of times you applied after layoff month in next 3 years
gen xxx=1 if date_req>=`i' & date_req<=`i'+36 & date_req!=.
egen next_napplied_36=count(xxx), by(id_worker)
drop xxx
gen xxx=.
forvalues j=1(1)10 {
cap replace xxx=1 if date_pag`j'>=`i'  & date_pag`j'<=`i'+36 & date_pag`j'!=. & valor_pago`j'!=0 & valor_pago`j'!=. & xxx==.
} 
egen next_nben_36=count(xxx), by(id_worker)
drop xxx
gen xxx=0
forvalues j=1(1)10 {
cap replace xxx=xxx+1 if date_pag`j'>=`i'  & date_pag`j'<=`i'+36 & date_pag`j'!=. & valor_pago`j'!=0 & valor_pago`j'!=.
} 
egen next_ntotalben_36=sum(xxx), by(id_worker)
drop xxx

*record number of times you applied after layoff month in next 5 years
gen xxx=1 if date_req>=`i' & date_req<=`i'+60 & date_req!=.
egen next_napplied_60=count(xxx), by(id_worker)
drop xxx
gen xxx=.
forvalues j=1(1)10 {
cap replace xxx=1 if date_pag`j'>=`i'  & date_pag`j'<=`i'+60 & date_pag`j'!=. & valor_pago`j'!=0 & valor_pago`j'!=. & xxx==.
} 
egen next_nben_60=count(xxx), by(id_worker)
drop xxx
gen xxx=0
forvalues j=1(1)10 {
cap replace xxx=xxx+1 if date_pag`j'>=`i'  & date_pag`j'<=`i'+60 & date_pag`j'!=. & valor_pago`j'!=0 & valor_pago`j'!=.
} 
egen next_ntotalben_60=sum(xxx), by(id_worker)
drop xxx

*investigate if collected benefit around 16 months after previous layoff
forvalues j=8(1)24 {
gen xxx=day_dem if date_dem==`i'+`j' & date_req!=.
gen yyy=.
forvalues h=1(1)7 {
replace yyy=day_dem if date_dem==`i'+`j' & date_req!=. & date_pag`h'>=`i'+`j'  & date_pag`h'!=. & valor_pago`h'!=0 & valor_pago`h'!=. & yyy==.
}
egen req_`j'months_after=max(xxx), by(id_worker)
egen ben_`j'months_after=max(yyy), by(id_worker)
drop xxx yyy
}

*keep application closest to layoff month
gen x=1 if this==_n
egen X=max(x), by(id_worker)
replace x=1 if last==_n & X!=1
drop X
keep if x==1

*save matched data
drop this last next
sort id_worker
save "sel_sample_all`i'_onlyID_match_this.dta", replace

*keep shorter version of data with fewer variables
gen first_pay=.
gen last_pay=.
gen npay=0 
forvalues j=1(1)10 {
cap replace first_pay=date_pag`j' if valor_pago`j'!=0 & valor_pago`j'!=. & first_pay==. 
cap replace last_pay=date_pag`j' if valor_pago`j'!=0 & valor_pago`j'!=. 
cap replace npay=npay+1 if valor_pago`j'!=0 & valor_pago`j'!=. 
}
drop status_payment* valor_emitido* valor_pago* tipo* date_emiss* day_emiss* date_pag* day_pag* issue_pag*
drop  cnpj_cei-hours_1
drop x
sort id_worker
save "sel_sample_all`i'_onlyID_match_this_short.dta", replace
}










