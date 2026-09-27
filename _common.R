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

# PDF figures go through cairo_pdf, set as `dev` of pdf_book in _output.yml, which
# embeds its fonts where the default pdf() device embeds none. It is set there
# because new_session: true sources this file before rmarkdown::render() starts,
# so the output format is not yet known here and the format's own `dev` would
# override one set now. Greek letters in plotmath also need a Unicode symbol font:
# with the default Symbol mapping on Windows, cairo draws Phi as "q". Keyed by
# device, so the PNG device of the HTML build does not receive the argument.
knitr::opts_chunk$set(
    dev.args = list(cairo_pdf = list(
        symbolfamily = grDevices::cairoSymbolFont("Arial", usePUA = FALSE)
    ))
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

# jtools::summ() prints its own footnote and joins the two halves with a
# semicolon: "Standard errors: OLS; Continuous predictors are mean-centered...".
# It comes from inside the package, so there is no argument to change and no
# huxtable object to edit on the way past. A document hook is the one place the
# string can still be reached, and it runs after cached chunks are restored, so
# fixing it here does not force an expensive chunk to run again. Deliberately
# narrow: it matches that one sentence and nothing else.
local({
    prev <- knitr::knit_hooks$get("document")
    knitr::knit_hooks$set(document = function(x) {
        # Chain rather than replace: knitr and rmarkdown install document hooks
        # of their own, and dropping one would lose whatever it does.
        if (is.function(prev)) x <- prev(x)
        gsub("(Standard errors: [^;\n]{1,40}); (Continuous predictors)", "\\1. \\2", x)
    })
})

# A code line fits about 73 characters at \small in the PDF text block, so
# print console output narrower than that. preamble.tex turns on fvextra
# breaklines to catch whatever still runs long (wide tables, long strings),
# but wrapping at a word boundary here reads better than a forced break.
options(width = 72)

# Attach a caption and a cross-reference label to a huxtable, which is what
# jtools::export_summs returns. Without this the table renders as an anonymous
# block: no number, no caption, and nothing for the text to refer to. Springer
# requires every table to be captioned and cited from the running text.
#
# The label is prefixed with "tab:" so that \@ref(tab:<label>) resolves.
cap_tbl <- function(h, caption, label) {
    if (requireNamespace("huxtable", quietly = TRUE)) {
        huxtable::caption(h) <- caption
        huxtable::label(h)   <- paste0("tab:", label)
        # huxtable centres a caption by default, which does not match the
        # left-aligned captions pandoc produces for every other table in the
        # volume. "topleft" switches it to raggedright.
        huxtable::caption_pos(h) <- "topleft"

        # huxreg's significance note separates the cutoffs with semicolons.
        # It sits in the first cell of the last row. Replace it so the printed
        # page follows the volume's punctuation.
        nr <- nrow(h)
        if (nr > 0) {
            last <- as.character(h[[1]][nr])
            if (length(last) == 1 && !is.na(last) && grepl("p <", last, fixed = TRUE)) {
                h[nr, 1] <- "* p < 0.05, ** p < 0.01, *** p < 0.001"
            }
        }

        # export_summs(scale = TRUE) appends its own footnote, and joins the two
        # halves with a semicolon: "Standard errors: OLS; Continuous predictors
        # are mean-centered...". Same rule, and the note is long enough that it
        # also has to be allowed to wrap, or it runs into the margin.
        for (j in seq_len(ncol(h))) {
            cells <- as.character(h[[j]])
            hit <- which(!is.na(cells) & grepl("Standard errors:", cells, fixed = TRUE))
            for (i in hit) h[i, j] <- gsub("; ", ". ", cells[i], fixed = TRUE)
        }

        # Long footnotes and long coefficient names are the two things that push
        # one of these tables past the text block. wrap() lets a cell break
        # across lines instead.
        huxtable::wrap(h) <- TRUE
    }
    h
}

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