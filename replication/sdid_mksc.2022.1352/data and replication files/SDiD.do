************************************************************************************************
***************************  The Value of Descriptive Analytics:  ****************************** 
***************************    Evidence from Online Retailers     ******************************
***************************    by Ron Berman and Ayelet Israei    ******************************
***************************        Marketing Science, 2022        ******************************
************************************************************************************************
* This Stata do file runs the analyses for the main results in the SDiD section of the  paper. *
* The file uses a limited data file including randomly selected 5\% of the retailer-channels,  *
* so results differ from those reported in the paper.                                          *
************************************************************************************************

// Import  data sample
import delimited "data_file_for_replication_small.csv", clear
/**
Variable descriptions:
		
name        	type	format		label
------------------------------------------------------------------------------------------------
company_id      long	%12.0g		retailer identifier
company_source  int		%8.0g		retailer-channel identifier
month_id        byte	%8.0g		month identifier
m_after         byte	%8.0g		indicator whether the observation is after adoption
month_join_id   byte	%8.0g		the month_id in which the retailer adopted analytics
join_lag        byte	%8.0g		lag/lead relative to joining, defined as month_id-month_join_id
m_log_rev_usd   float	%9.0g		log average weekly revenue in a specific month
**/				

//Data includes observations from 7/2015-6/2017. Treated units are retailers that adopted from 1/2016 to 6/2017.
gen cohort = month_join_id // cohort will indicate which cohort the company belongs to. 
replace cohort = 999 if (month_join_id>24) // Those not treated will be assigned to cohort 999
tabulate month_id, gen(month) // Create time indicators (1 to above 24, the first adoption of 2016 is January which is month_id==7).
tabulate cohort, gen(cohort) // Create cohort indicators 
tabulate join_lag, gen(lag) //creates lag indicators.

/* Eq (3) of the paper: Pre-trends, with cohort time trends, only pre-tretmentunit data, including company/time clustering*/
reghdfe m_log_rev_usd c.cohort1-cohort18 ///
c.cohort1#c.month_id c.cohort2#c.month_id c.cohort3#c.month_id c.cohort4#c.month_id c.cohort5#c.month_id c.cohort6#c.month_id ///
c.cohort7#c.month_id c.cohort8#c.month_id c.cohort9#c.month_id c.cohort10#c.month_id c.cohort11#c.month_id c.cohort12#c.month_id ///
c.cohort13#c.month_id c.cohort14#c.month_id c.cohort15#c.month_id c.cohort16#c.month_id c.cohort17#c.month_id c.cohort18#c.month_id ///
c.lag1-lag17 c.lag19-lag23 ///
if m_after==0, absorb(month_id) vce(cluster company_id month_id) nocons

testparm lag19-lag23

/* Eq (2) of the paper, following Wooldrige 2021, including all time periods, company/time clustering, with cohort trends*/
reghdfe m_log_rev_usd c.cohort1-cohort18 ///
c.cohort1#c.month7-month24 ///
c.cohort2#c.month8-month24 ///
c.cohort3#c.month9-month24 ///
c.cohort4#c.month10-month24 ///
c.cohort5#c.month11-month24 ///
c.cohort6#c.month12-month24 ///
c.cohort7#c.month13-month24 ///
c.cohort8#c.month14-month24 ///
c.cohort9#c.month15-month24 ///
c.cohort10#c.month16-month24 ///
c.cohort11#c.month17-month24 ///
c.cohort12#c.month18-month24 ///
c.cohort13#c.month19-month24 ///
c.cohort14#c.month20-month24 ///
c.cohort15#c.month21-month24 ///
c.cohort16#c.month22-month24 ///
c.cohort17#c.month23-month24 ///
c.cohort18#c.month24 ///
c.cohort1#c.month_id c.cohort2#c.month_id c.cohort3#c.month_id c.cohort4#c.month_id c.cohort5#c.month_id c.cohort6#c.month_id ///
c.cohort7#c.month_id c.cohort8#c.month_id c.cohort9#c.month_id c.cohort10#c.month_id c.cohort11#c.month_id c.cohort12#c.month_id ///
c.cohort13#c.month_id c.cohort14#c.month_id c.cohort15#c.month_id c.cohort16#c.month_id c.cohort17#c.month_id c.cohort18#c.month_id ///
, absorb(month_id) vce(cluster company_id month_id) nocons


// ATT for 3 months (Table 5 column 1)
lincom (c.cohort1#c.month7+ ///
c.cohort2#c.month8+ ///
c.cohort3#c.month9+ ///
c.cohort4#c.month10+ ///
c.cohort5#c.month11+ ///
c.cohort6#c.month12+ ///
c.cohort7#c.month13+ ///
c.cohort8#c.month14+ ///
c.cohort9#c.month15+ ///
c.cohort10#c.month16+ ///
c.cohort11#c.month17+ ///
c.cohort12#c.month18+ ///
c.cohort13#c.month19+ ///
c.cohort14#c.month20+ ///
c.cohort15#c.month21+ ///
c.cohort16#c.month22+ ///
c.cohort17#c.month23+ ///
c.cohort18#c.month24+ ///
c.cohort1#c.month8+ ///
c.cohort2#c.month9+ ///
c.cohort3#c.month10+ ///
c.cohort4#c.month11+ ///
c.cohort5#c.month12+ ///
c.cohort6#c.month13+ ///
c.cohort7#c.month14+ ///
c.cohort8#c.month15+ ///
c.cohort9#c.month16+ ///
c.cohort10#c.month17+ ///
c.cohort11#c.month18+ ///
c.cohort12#c.month19+ ///
c.cohort13#c.month20+ ///
c.cohort14#c.month21+ ///
c.cohort15#c.month22+ ///
c.cohort16#c.month23+ ///
c.cohort17#c.month24+ ///
c.cohort1#c.month9+ ///
c.cohort2#c.month10+ ///
c.cohort3#c.month11+ ///
c.cohort4#c.month12+ ///
c.cohort5#c.month13+ ///
c.cohort6#c.month14+ ///
c.cohort7#c.month15+ ///
c.cohort8#c.month16+ ///
c.cohort9#c.month17+ ///
c.cohort10#c.month18+ ///
c.cohort11#c.month19+ ///
c.cohort12#c.month20+ ///
c.cohort13#c.month21+ ///
c.cohort14#c.month22+ ///
c.cohort15#c.month23+ ///
c.cohort16#c.month24 ///
)/(16+17+18)


// ATT for 6 months (Table 5 column 2)
lincom (c.cohort1#c.month7+ ///
c.cohort2#c.month8+ ///
c.cohort3#c.month9+ ///
c.cohort4#c.month10+ ///
c.cohort5#c.month11+ ///
c.cohort6#c.month12+ ///
c.cohort7#c.month13+ ///
c.cohort8#c.month14+ ///
c.cohort9#c.month15+ ///
c.cohort10#c.month16+ ///
c.cohort11#c.month17+ ///
c.cohort12#c.month18+ ///
c.cohort13#c.month19+ ///
c.cohort14#c.month20+ ///
c.cohort15#c.month21+ ///
c.cohort16#c.month22+ ///
c.cohort17#c.month23+ ///
c.cohort18#c.month24+ ///
c.cohort1#c.month8+ ///
c.cohort2#c.month9+ ///
c.cohort3#c.month10+ ///
c.cohort4#c.month11+ ///
c.cohort5#c.month12+ ///
c.cohort6#c.month13+ ///
c.cohort7#c.month14+ ///
c.cohort8#c.month15+ ///
c.cohort9#c.month16+ ///
c.cohort10#c.month17+ ///
c.cohort11#c.month18+ ///
c.cohort12#c.month19+ ///
c.cohort13#c.month20+ ///
c.cohort14#c.month21+ ///
c.cohort15#c.month22+ ///
c.cohort16#c.month23+ ///
c.cohort17#c.month24+ ///
c.cohort1#c.month9+ ///
c.cohort2#c.month10+ ///
c.cohort3#c.month11+ ///
c.cohort4#c.month12+ ///
c.cohort5#c.month13+ ///
c.cohort6#c.month14+ ///
c.cohort7#c.month15+ ///
c.cohort8#c.month16+ ///
c.cohort9#c.month17+ ///
c.cohort10#c.month18+ ///
c.cohort11#c.month19+ ///
c.cohort12#c.month20+ ///
c.cohort13#c.month21+ ///
c.cohort14#c.month22+ ///
c.cohort15#c.month23+ ///
c.cohort16#c.month24+ ///
c.cohort1#c.month10+ ///
c.cohort2#c.month11+ ///
c.cohort3#c.month12+ ///
c.cohort4#c.month13+ ///
c.cohort5#c.month14+ ///
c.cohort6#c.month15+ ///
c.cohort7#c.month16+ ///
c.cohort8#c.month17+ ///
c.cohort9#c.month18+ ///
c.cohort10#c.month19+ ///
c.cohort11#c.month20+ ///
c.cohort12#c.month21+ ///
c.cohort13#c.month22+ ///
c.cohort14#c.month23+ ///
c.cohort15#c.month24+ ///
c.cohort1#c.month11+ ///
c.cohort2#c.month12+ ///
c.cohort3#c.month13+ ///
c.cohort4#c.month14+ ///
c.cohort5#c.month15+ ///
c.cohort6#c.month16+ ///
c.cohort7#c.month17+ ///
c.cohort8#c.month18+ ///
c.cohort9#c.month19+ ///
c.cohort10#c.month20+ ///
c.cohort11#c.month21+ ///
c.cohort12#c.month22+ ///
c.cohort13#c.month23+ ///
c.cohort14#c.month24+ ///
c.cohort1#c.month12+ ///
c.cohort2#c.month13+ ///
c.cohort3#c.month14+ ///
c.cohort4#c.month15+ ///
c.cohort5#c.month16+ ///
c.cohort6#c.month17+ ///
c.cohort7#c.month18+ ///
c.cohort8#c.month19+ ///
c.cohort9#c.month20+ ///
c.cohort10#c.month21+ ///
c.cohort11#c.month22+ ///
c.cohort12#c.month23+ ///
c.cohort13#c.month24 ///
)/(13+14+15+16+17+18)
