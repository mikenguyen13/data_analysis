# Point every page of a rendered book at one canonical host.
#
# Google indexes almost nothing on share.connect.posit.cloud (Search Console,
# Nov 2026: 1 of 161 pages), so the book is also published on GitHub Pages and
# every page names the GitHub Pages copy as canonical. The SAME output is
# deployed to both hosts: on GitHub Pages each page is self-canonical, on Posit
# each page tells Google the original lives on GitHub Pages. Readers on Posit
# are not redirected.
#
# Usage (from the repo root, after render + gen_sitemap_posit.R):
#   Rscript redirects/set_canonical_host.R [_book] [base_url]
#
# What it does to <dir>:
#   * content pages: adds/replaces <link rel="canonical" href="<base><page>">
#     (index.html -> <base>), rewrites og:url and the JSON-LD Book @id/url,
#     and fixes the "//images/cover.jpg" double slash;
#   * every page and stub: rewrites absolute links on the old hosts to <base>;
#   * redirect stubs: their canonical moves to <base><target page>;
#   * writes sitemap.xml (content pages only) and robots.txt for <base>;
#   * writes .nojekyll so GitHub Pages serves *_files/ and libs/ as-is.

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) >= 1) args[[1]] else "_book"
base <- if (length(args) >= 2) args[[2]] else "https://mikenguyen13.github.io/data_analysis/"
if (!grepl("/$", base)) base <- paste0(base, "/")

old_hosts <- c("https://mike-data-analysis.share.connect.posit.cloud/",
               "https://bookdown.org/mike/data_analysis/")

read_text <- function(p) readChar(p, file.size(p), useBytes = TRUE)
write_text <- function(x, p) {
  con <- file(p, open = "wb"); on.exit(close(con))
  writeChar(x, con, eos = NULL, useBytes = TRUE)
}
is_stub <- function(x) grepl('http-equiv="refresh"', x, fixed = TRUE) && nchar(x, type = "bytes") < 4000

files <- list.files(dir, pattern = "[.]html$")
n_pages <- n_stubs <- 0L
pages <- character()
for (f in files) {
  p <- file.path(dir, f)
  x <- read_text(p)
  stub <- is_stub(x)
  # live links on the old Posit host -> base (bookdown.org citation text in
  # comments is left alone; only the Posit host and JSON-LD are rewritten)
  x <- gsub("https://mike-data-analysis.share.connect.posit.cloud//", base, x, fixed = TRUE, useBytes = TRUE)
  x <- gsub("https://mike-data-analysis.share.connect.posit.cloud/", base, x, fixed = TRUE, useBytes = TRUE)
  if (stub) {
    n_stubs <- n_stubs + 1L
  } else if (f != "404.html") {
    canon <- paste0(base, if (f == "index.html") "" else f)
    x <- gsub('<link rel="canonical"[^>]*>\\s*', "", x, useBytes = TRUE)
    x <- sub("</head>", paste0('<link rel="canonical" href="', canon, '">\n</head>'), x, fixed = TRUE, useBytes = TRUE)
    x <- gsub('(<meta property="og:url" content=")[^"]*(")', paste0("\\1", canon, "\\2"), x, useBytes = TRUE)
    x <- gsub('"@id": "https://bookdown.org/mike/data_analysis/#book"', paste0('"@id": "', base, '#book"'), x, fixed = TRUE, useBytes = TRUE)
    x <- gsub('"url": "https://bookdown.org/mike/data_analysis/"', paste0('"url": "', base, '"'), x, fixed = TRUE, useBytes = TRUE)
    n_pages <- n_pages + 1L
    pages <- c(pages, f)
  }
  write_text(x, p)
}

mt <- format(file.mtime(file.path(dir, pages)), "%Y-%m-%d")
loc <- paste0(base, ifelse(pages == "index.html", "", pages))
writeLines(c('<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
             paste0("  <url><loc>", loc, "</loc><lastmod>", mt, "</lastmod></url>"),
             "</urlset>"), file.path(dir, "sitemap.xml"))
writeLines(c("User-agent: *", "Allow: /", paste0("Sitemap: ", base, "sitemap.xml")),
           file.path(dir, "robots.txt"))
file.create(file.path(dir, ".nojekyll"))
cat("canonical base:", base, "\n", n_pages, "content pages,", n_stubs, "stubs rewritten; sitemap.xml has", length(pages), "urls\n")
