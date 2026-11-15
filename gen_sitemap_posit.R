# Pre-deploy step for Posit Connect Cloud. Run after the bs4_book render and
# before rsconnect::deployApp(appDir = "_book", ...):
#   Rscript gen_sitemap_posit.R
# It writes sitemap.xml, copies the root-level site files into _book, and adds
# the redirect stubs for old bookdown.org URLs (redirects/).
source("redirects/build_redirects.R")

base <- "https://mike-data-analysis.share.connect.posit.cloud/"
pages <- setdiff(list.files("_book", pattern = "[.]html$"), "404.html")
# Redirect stubs (from an earlier run) are not real pages; keep them out.
pages <- pages[!vapply(file.path("_book", pages), is_redirect_stub, logical(1))]
mt <- format(file.mtime(file.path("_book", pages)), "%Y-%m-%d")
loc <- paste0(base, ifelse(pages == "index.html", "", pages))
xml <- c('<?xml version="1.0" encoding="UTF-8"?>',
         '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
         paste0("  <url><loc>", loc, "</loc><lastmod>", mt, "</lastmod></url>"),
         "</urlset>")
writeLines(xml, "sitemap.xml")
cat("wrote sitemap.xml with", length(pages), "urls\n")

# bs4_book ignores resource_files, so copy the root-level site files into _book
# before deploying; otherwise llms.txt is not served.
site_files <- c("sitemap.xml", "robots.txt", "llms.txt")
file.copy(site_files, "_book", overwrite = TRUE)
cat("copied", paste(site_files, collapse = ", "), "into _book\n")

# Old bookdown.org filenames -> redirect stubs. Without these, every old URL
# that Google still ranks returns 404 after a redeploy.
install_redirects("_book")
