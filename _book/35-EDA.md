# Exploratory Data Analysis


```r
# load to get txhousing data
library(ggplot2)
```

Data Report

Feature Engineering

Missing Data


```r
# install.packages("DataExplorer")
library(DataExplorer)

# creat a html file that contain all reports
create_report(txhousing)

introduce() # see basic info


dummify() # create binary columns from discrete variables
split_columns() # split data into discrete and continuous parts



plot_correlation() # heatmap for discrete var
plot_intro() 

plot_missing() # plot missing value
profile_missing() # profile missing values


plot_prcomp() # plot PCA
```

Error Identification


```r
# install.packages("dataReporter")
library(dataReporter)
makeDataReport() # detailed report like DataExplorer
```

Summary statistics


```r
library(skimr)
skim() # give only few quick summary stat, not as detailed as the other two packages
```

Not so code-y process

Quick and dirty way to look at your data


```r
# install.packages("rpivotTable")
library(rpivotTable)
# give set up just like Excel table 
data %>% 
    rpivotTable::rpivotTable()
```

Code generation and wrangling

Shiny-app based Tableu style


```r
# install.packages("esquisse")
library(esquisse)
esquisse::esquisser()
```

Customized your daily/automatic report


```r
# install.packages("chronicle")
library(chronicle)
```


```r
# install.packages("dlookr")
# install.packages("descriptr")
```
