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

# Avoid duplicate `fig:unnamed-chunk-N` cross-chapter clashes by prefixing
# auto-generated chunk labels with the chapter file basename.
options(knitr.duplicate.label = "allow")
local({
    ci <- tryCatch(knitr::current_input(), error = function(e) NULL)
    if (!is.null(ci) && nzchar(ci)) {
        prefix <- tools::file_path_sans_ext(basename(ci))
        prefix <- gsub("[^A-Za-z0-9]+", "-", prefix)
        prefix <- gsub("^-+|-+$", "", prefix)
        if (nzchar(prefix)) knitr::opts_knit$set(unnamed.chunk.label = prefix)
    }
})

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