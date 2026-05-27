#!/usr/bin/env Rscript
# Regenerates the table-of-contents block in README.md from the chapter Rmd files.
#
# Run from the repo root:   Rscript tools/build-toc.R
# CI runs the same command; see .github/workflows/update-toc.yml.

suppressWarnings(Sys.setlocale("LC_COLLATE", "C"))

repo_root <- getwd()
if (!file.exists(file.path(repo_root, "README.md"))) {
  stop("Run from the repo root: Rscript tools/build-toc.R")
}

readme_path <- file.path(repo_root, "README.md")
start_tag   <- "<!-- toc:start -->"
end_tag     <- "<!-- toc:end -->"

# Files to ignore: build artifacts, front matter, and underscore-prefixed drafts.
skip_files <- c("index.Rmd", "data_analysis.Rmd")

rmd_files <- list.files(repo_root, pattern = "\\.Rmd$", full.names = FALSE)
rmd_files <- rmd_files[!startsWith(rmd_files, "_")]
rmd_files <- rmd_files[!rmd_files %in% skip_files]
rmd_files <- sort(rmd_files)

strip_braces <- function(s) sub("\\s*\\{[^}]*\\}\\s*$", "", s)

classify <- function(file) {
  con   <- file(file.path(repo_root, file), open = "r", encoding = "UTF-8")
  first <- readLines(con, n = 1, warn = FALSE)
  close(con)
  if (length(first) == 0) return(NULL)
  line <- trimws(first)

  # Top-of-file part dividers:  # (PART\*) Title {-}   or   # (PART) Title {-}
  m <- regmatches(line, regexec("^#\\s*\\(PART\\\\?\\*?\\)\\s*(.+?)\\s*\\{-\\}\\s*$", line))[[1]]
  if (length(m) == 2) {
    title <- strip_braces(m[2])
    # Roman-numeral prefix => top-level part; single capital letter => sub-part.
    level <- if (grepl("^[IVX]+\\.", title)) "part" else "subpart"
    return(list(kind = level, title = title))
  }

  # Appendix divider
  if (grepl("^#\\s*\\(APPENDIX\\)", line)) {
    return(list(kind = "appendix_marker", title = NA_character_))
  }

  # Real chapter heading: a single '#', then space, then text (not '##').
  m <- regmatches(line, regexec("^#\\s+([^#].*)$", line))[[1]]
  if (length(m) == 2) {
    return(list(kind = "chapter", title = strip_braces(m[2])))
  }

  NULL  # Sub-section file (## / ###), code, YAML, etc. — skip.
}

entries <- Filter(Negate(is.null), lapply(rmd_files, classify))

# If chapters appear before the first PART marker, group them under "Part I. Foundations".
# (The real Part I divider lives in an underscore-prefixed draft file in this repo.)
first_part_idx <- which(vapply(entries, function(e) e$kind == "part", logical(1)))[1]
if (length(first_part_idx) && !is.na(first_part_idx) && first_part_idx > 1) {
  entries <- c(
    list(list(kind = "part", title = "I. FOUNDATIONS")),
    entries
  )
}

# Render to markdown.
out <- c(start_tag)
chapter_n <- 0
appendix_letter_idx <- 0
in_appendix <- FALSE

# Title-case "WORDS LIKE THIS" -> "Words Like This", preserving hyphenated parts.
title_case_words <- function(s) {
  parts <- strsplit(s, " ", fixed = TRUE)[[1]]
  cap_token <- function(tok) {
    sub_parts <- strsplit(tok, "-", fixed = TRUE)[[1]]
    sub_parts <- vapply(sub_parts, function(w) {
      if (nchar(w) == 0) return(w)
      paste0(toupper(substring(w, 1, 1)), tolower(substring(w, 2)))
    }, character(1))
    paste(sub_parts, collapse = "-")
  }
  paste(vapply(parts, cap_token, character(1)), collapse = " ")
}

# "II. REGRESSION" -> "Part II. Regression"   (roman numerals stay upper)
format_part_title <- function(s) {
  m <- regmatches(s, regexec("^([IVX]+)\\.\\s*(.+)$", s))[[1]]
  if (length(m) == 3) paste0("Part ", m[2], ". ", title_case_words(m[3])) else s
}
# "A. EXPERIMENTAL DESIGN" -> "A. Experimental Design"   (label stays upper)
format_subpart_title <- function(s) {
  m <- regmatches(s, regexec("^([A-Z])\\.\\s*(.+)$", s))[[1]]
  if (length(m) == 3) paste0(m[2], ". ", title_case_words(m[3])) else s
}

for (e in entries) {
  if (e$kind == "part") {
    out <- c(out, "", paste0("**", format_part_title(e$title), "**"), "")
  } else if (e$kind == "subpart") {
    out <- c(out, "", paste0("- *", format_subpart_title(e$title), "*"), "")
  } else if (e$kind == "appendix_marker") {
    in_appendix <- TRUE
    out <- c(out, "", "**Appendix**", "")
  } else if (e$kind == "chapter") {
    if (in_appendix) {
      appendix_letter_idx <- appendix_letter_idx + 1
      label <- LETTERS[appendix_letter_idx]
    } else {
      chapter_n <- chapter_n + 1
      label <- as.character(chapter_n)
    }
    out <- c(out, paste0(label, ". ", e$title))
  }
}

out <- c(out, "", end_tag)

# Splice into README.md
readme <- readLines(readme_path, warn = FALSE, encoding = "UTF-8")
start_i <- which(readme == start_tag)
end_i   <- which(readme == end_tag)

if (length(start_i) != 1 || length(end_i) != 1 || end_i <= start_i) {
  stop("README.md must contain exactly one '", start_tag, "' and one '", end_tag,
       "' line, with start before end.")
}

new_readme <- c(readme[seq_len(start_i - 1)], out, readme[(end_i + 1):length(readme)])

if (!identical(readme, new_readme)) {
  writeLines(new_readme, readme_path, useBytes = TRUE)
  message("README.md updated.")
} else {
  message("README.md already up to date.")
}
