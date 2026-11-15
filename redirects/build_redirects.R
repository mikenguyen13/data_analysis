# Redirect stubs for the book's old bookdown.org URLs.
#
# The book moved from bookdown.org/mike/data_analysis/ to Posit Connect Cloud
# in May 2026 and many chapter/section filenames changed, so old URLs that
# Google still ranks returned 404. Each old filename now gets a tiny HTML page
# (meta refresh + canonical + JS location.replace) that forwards to its new home.
#
#   redirects/redirect_map.json   old filename -> {to, how, old_title, old_chapter}
#   redirects/stubs/*.html        generated stub pages (committed)
#
# Usage (from the repo root):
#   Rscript redirects/build_redirects.R stubs           # regenerate redirects/stubs/ from the map
#   Rscript redirects/build_redirects.R install [_book] # copy the stubs into a rendered book
#
# gen_sitemap_posit.R sources this file and runs install_redirects("_book")
# as part of the pre-deploy step, so a normal render + deploy ships the stubs.

redirect_base_url <- "https://mike-data-analysis.share.connect.posit.cloud/"
redirect_dir <- if (dir.exists("redirects")) "redirects" else "."

read_redirect_map <- function(path = file.path(redirect_dir, "redirect_map.json")) {
  jsonlite::read_json(path)
}

html_escape <- function(x) {
  x <- gsub("&", "&amp;", x, fixed = TRUE)
  x <- gsub("<", "&lt;", x, fixed = TRUE)
  x <- gsub(">", "&gt;", x, fixed = TRUE)
  gsub("\"", "&quot;", x, fixed = TRUE)
}

stub_html <- function(old, entry) {
  to <- entry$to
  title <- entry$old_title
  if (is.null(title) || !nzchar(title)) {
    title <- gsub("-", " ", sub("[.]html$", "", old))
  }
  to_attr <- html_escape(to)
  c(
    "<!DOCTYPE html>",
    "<html lang=\"en\">",
    "<head>",
    "<meta charset=\"utf-8\">",
    paste0("<title>Moved: ", html_escape(title), " | A Guide on Data Analysis</title>"),
    # canonical names the page, not the section, so it drops any #fragment;
    # index.html is canonical as the site root, matching sitemap.xml
    paste0("<link rel=\"canonical\" href=\"", redirect_base_url,
           html_escape(sub("^index[.]html$", "", sub("#.*$", "", to))), "\">"),
    paste0("<meta http-equiv=\"refresh\" content=\"0; url=", to_attr, "\">"),
    "<script>",
    "(function () {",
    paste0("  var to = ", jsonlite::toJSON(to, auto_unbox = TRUE), ";"),
    "  if (to.indexOf(\"#\") < 0 && location.hash) to += location.hash;",
    "  location.replace(to);",
    "})();",
    "</script>",
    "</head>",
    "<body>",
    paste0("<p>This page has moved to <a href=\"", to_attr, "\">", to_attr, "</a>.</p>"),
    "</body>",
    "</html>"
  )
}

write_stubs <- function(map = read_redirect_map(), out_dir = file.path(redirect_dir, "stubs")) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  old <- list.files(out_dir, pattern = "[.]html$")
  unlink(file.path(out_dir, setdiff(old, names(map))))
  for (name in names(map)) {
    con <- file(file.path(out_dir, name), open = "wb")
    writeLines(stub_html(name, map[[name]]), con, sep = "\n", useBytes = TRUE)
    close(con)
  }
  cat("wrote", length(map), "stubs to", out_dir, "\n")
  invisible(names(map))
}

# Copy stubs into a rendered book. A stub never replaces a page the render
# produced: if a current chapter reuses an old filename, the real page wins.
# Also warns when a stub points at a page the render did not produce.
install_redirects <- function(book_dir = "_book",
                              stub_dir = file.path(redirect_dir, "stubs"),
                              map = read_redirect_map()) {
  stopifnot(dir.exists(book_dir), dir.exists(stub_dir))
  stubs <- list.files(stub_dir, pattern = "[.]html$")
  dest <- file.path(book_dir, stubs)
  real <- stubs[file.exists(dest) & !vapply(dest, is_redirect_stub, logical(1))]
  if (length(real)) {
    warning("skipped ", length(real), " stub(s) whose filename is a real page in ", book_dir,
            ": ", paste(real, collapse = ", "), call. = FALSE)
  }
  todo <- setdiff(stubs, real)
  ok <- file.copy(file.path(stub_dir, todo), book_dir, overwrite = TRUE)
  targets <- unique(sub("#.*$", "", vapply(map, `[[`, "", "to")))
  missing <- targets[!file.exists(file.path(book_dir, targets))]
  if (length(missing)) {
    warning("redirect targets not found in ", book_dir, ": ",
            paste(missing, collapse = ", "), call. = FALSE)
  }
  cat("copied", sum(ok), "redirect stubs into", book_dir, "\n")
  invisible(todo[ok])
}

# TRUE if the file is one of our stubs (not a rendered page), so the sitemap
# can leave it out and install_redirects() may overwrite it.
is_redirect_stub <- function(path) {
  file.exists(path) &&
    any(startsWith(readLines(path, n = 6, warn = FALSE), "<title>Moved: "))
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  cmd <- if (length(args)) args[[1]] else "stubs"
  if (cmd == "stubs") {
    write_stubs()
  } else if (cmd == "install") {
    install_redirects(if (length(args) > 1) args[[2]] else "_book")
  } else {
    stop("usage: Rscript redirects/build_redirects.R [stubs | install [book_dir]]")
  }
}
