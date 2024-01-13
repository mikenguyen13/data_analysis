This folder is intended to help in simulating some of the results of Turjeman and Feinberg (2023).

Note, that due to confidentiality agreement (NDA) the data cannot be shared, and some of the code was reducted. 
However, sample of de-identified and simulated data, as well as most of the code (after cleaning and removing off-the-shelf analyses), are provided for those interested in exploring the analyses offered.
This means that some of the results in the paper may not be directly reproduced with given data and code, but will be close to those in the paper.


Please contact the corresponding author if you have any questions.


Where to begin: 
	1. Turjeman and Feinberg 2023 Replication Code - Temporal Causal Inference.Rmd

This file includes code of the creation of the sample data, as well as code for illustration of Temporal Causal Inference, and the associated figures.


DID and LLCF comparisons folder includes this file:
	2. Turjeman and Feinberg 2023 Replication Code - Local Linear Causal Forests Simulations.R 
which is a file intended for simulating data (unrelated to the data breach) to compare Diff in Diff and Causal Forests. 
The results are provided in the Appendix of the paper.
Changing the N to be 1000 will make things much faster and will still be close enough to the real results (which were ran with N = 10,000).

For Causal Forests, run this file:
	3. Turjeman and Feinberg 2023 Replication Code - Causal Forests.R
	
This file will run Temporal Causal Forests and will create some of the associated figures.

Other files in the folder:

- summary_timeline.csv (can be re-created in file #1)
- all_cohorts_variance.csv - Causal forests analysis of cohorts 10-13 for ease of reproduction. This can be re-created using file #3 (you may also remove the limit on the cohorts, to not just run cohorts 10-13).
