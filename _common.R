knitr::opts_chunk$set(
    comment = "#>",
    collapse = TRUE,
    cache = TRUE,
    warning = FALSE,
    message = FALSE,
    echo = TRUE,
    dpi = 300,
    cache.lazy = FALSE,
    tidy = "styler",
    out.width = "90%",
    fig.align = "center",
    fig.width = 5,
    fig.height = 7
)

options(crayon.enabled = FALSE)

# suppressPackageStartupMessages(library(tidyverse))
options(warn = -1)
suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
    library(tidyverse)
    library(data.table)
    library(lubridate)
})

theme_set(theme_light())

library(scales)
library(methods)