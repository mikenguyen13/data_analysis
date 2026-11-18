# Build the analysis-ready extract in data/flammer_materiality/ from the
# Kofford, Lewis & Melancon (2026) replication package,
# https://doi.org/10.7910/DVN/CAWEJW (CC0 1.0).
#
# Usage: Rscript build.R <path to unzipped "Replication Package"> <output dir>
#
# Download the package zip from the DOI above, unzip it to a short path
# (some listing file names are long), and point the first argument at the
# "Replication Package" folder. See SOURCE.md for what each output holds.

args <- commandArgs(TRUE)
pkg <- args[1]
out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(haven); library(dplyr); library(stringr); library(purrr); library(tibble)
})

hc <- file.path(pkg, "data", "hand_coded")

# ---- 1. Event sample with SASB materiality and confound flags ----------------
smp  <- read_dta(file.path(hc, "Flammer 2013 - Sample.dta")) |> zap_labels() |> zap_formats()
mat  <- read_dta(file.path(hc, "Flammer 2013 - SASB Materiality.dta")) |> zap_labels() |> zap_formats()
conf <- read_dta(file.path(hc, "Flammer 2013 - Confounding Events.dta")) |> zap_labels() |> zap_formats()

key <- c("permno", "evt_date", "type", "comnam")
stopifnot(!anyDuplicated(smp[key]), !anyDuplicated(mat[key]), !anyDuplicated(conf[key]))

events <- smp |>
  select(permno, comnam, evt_date, type, issue, cyear, replication, extension) |>
  left_join(select(mat, all_of(key), sasb_sector = sasb_sct, sasb_industry = sasb_ind,
                   sasb_category = sasb_cat, sasb_material = sasb_mat), by = key) |>
  left_join(select(conf, all_of(key), confound), by = key) |>
  # three SASB category labels in the deposit are spelled inconsistently
  mutate(sasb_category = recode(sasb_category,
                                "Water & Wastewater Managementt" = "Water & Wastewater Management",
                                "Employee Health and Safety"     = "Employee Health & Safety",
                                "GHG emissions"                  = "GHG Emissions")) |>
  mutate(permno = as.integer(permno), cyear = as.integer(cyear),
         replication = as.integer(replication), extension = as.integer(extension),
         sasb_material = as.integer(sasb_material), confound = as.integer(confound)) |>
  arrange(evt_date, permno, type)
stopifnot(nrow(events) == 512, !anyNA(events$sasb_material), !anyNA(events$confound))

# ---- 2. Parse an Eventus listing ---------------------------------------------
num <- function(x) as.numeric(str_remove_all(x, "[%(),<>$*]"))

parse_params <- function(lines) {
  rx <- "^\\s*(\\d+)\\s+(\\d{2}[A-Z]{3}\\d{4})\\s+(-?[0-9.]+)\\s+([0-9.]+)%\\s+(-?[0-9.]+)\\s+(-?[0-9.]+)\\s+([0-9.]+)%\\s+([0-9.]+)\\s+([0-9.]+)\\s+(-?[0-9.]+)\\s*$"
  m <- str_match(lines, rx)
  m <- m[!is.na(m[, 1]), , drop = FALSE]
  tibble(permno = as.integer(m[, 2]),
         evt_date = as.Date(m[, 3], format = "%d%b%Y"),
         mean_ret = as.numeric(m[, 4]), pct_raw_pos = as.numeric(m[, 5]) / 100,
         alpha = as.numeric(m[, 6]), beta = as.numeric(m[, 7]),
         pct_resid_pos = as.numeric(m[, 8]) / 100, total_var = as.numeric(m[, 9]),
         resid_sd = as.numeric(m[, 10]), autocorr = as.numeric(m[, 11]))
}

parse_windows <- function(lines) {
  # last "Cumulative" table in the file holds the window results
  hdr <- max(grep("Cumulative", lines))
  hdr_txt <- paste(lines[hdr:(hdr + 3)], collapse = " ")
  new_layout <- grepl("Time-Series", hdr_txt)
  stat_name <- if (grepl("StdCsect|Csect", hdr_txt)) "stdcsect_z" else "patell_z"
  body <- lines[(hdr + 4):length(lines)]
  wrx <- "^\\s*\\(\\+?-?\\d+,\\+?-?\\d+\\)"
  idx <- grep(wrx, body)
  # a narrow listing wraps the last statistic onto the next line
  rows <- vapply(idx, function(i) {
    l <- body[i]
    if (i < length(body) && grepl("^\\s*-?[0-9.]+[*$()<>]*\\s*$", body[i + 1])) l <- paste(l, body[i + 1])
    l
  }, character(1))
  map_dfr(rows, function(l) {
    tok <- str_split(str_squish(l), " ")[[1]]
    tok[1] <- str_remove_all(tok[1], "\\+")
    pn <- str_match(tok[5], "^(\\d+):(\\d+)")
    vals <- num(tok[6:length(tok)])
    r <- tibble(window = tok[1], n = as.integer(tok[2]),
                mean_car = num(tok[3]) / 100, pw_caar = num(tok[4]) / 100,
                n_pos = as.integer(pn[2]), n_neg = as.integer(pn[3]),
                cda_t = NA_real_, patell_z = NA_real_, stdcsect_z = NA_real_,
                gsign_z = NA_real_, layout = if (new_layout) "current" else "older")
    if (new_layout) {
      r$cda_t <- vals[1]; r[[stat_name]] <- vals[2]; r$gsign_z <- vals[3]
    } else {
      r[[stat_name]] <- vals[1]; r$gsign_z <- vals[3]
    }
    r
  })
}

lst_dir <- file.path(pkg, "documentation", "eventus_output")
files <- list.files(lst_dir, pattern = "\\.txt$", full.names = TRUE)

cells <- map_dfr(files, function(f) {
  lines <- readLines(f, warn = FALSE)
  w <- parse_windows(lines)
  pr <- parse_params(lines)
  w$listing <- str_remove(basename(f), "^Flammer 2013 - ") |> str_remove("\\.txt$")
  w$n_param_rows <- nrow(pr)
  w$phat_listing <- mean(pr$pct_resid_pos)
  w
}) |>
  relocate(listing, window)

# ---- 3. Per-event market-model parameters (Table 4 listings: all 512) --------
params <- bind_rows(
  parse_params(readLines(file.path(lst_dir, "Flammer 2013 - Table 4 - Eco-friendly (-1,0).txt"))) |>
    mutate(type = "eco-friendly"),
  parse_params(readLines(file.path(lst_dir, "Flammer 2013 - Table 4 - Eco-harmful (-1,0).txt"))) |>
    mutate(type = "eco-harmful"))
stopifnot(nrow(params) == 512, !anyDuplicated(params[c("permno", "evt_date", "type")]))

ev_params <- events |>
  left_join(params, by = c("permno", "evt_date", "type"))
stopifnot(!anyNA(ev_params$resid_sd))

write.csv(ev_params, file.path(out, "events.csv"), row.names = FALSE, na = "")
write.csv(cells, file.path(out, "eventus_cells.csv"), row.names = FALSE, na = "")
cat("events:", nrow(ev_params), " cells:", nrow(cells), "\n")
print(as.data.frame(cells[cells$window == "(-1,0)", c("listing", "n", "mean_car", "n_pos", "n_neg", "patell_z", "gsign_z", "cda_t", "layout")]), digits = 4)
print(as.data.frame(cells[cells$window != "(-1,0)", c("listing", "window", "n", "mean_car", "patell_z", "stdcsect_z", "gsign_z")]), digits = 4)
