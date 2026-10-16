base <- "https://mike-data-analysis.share.connect.posit.cloud/"
pages <- setdiff(list.files("_book", pattern = "[.]html$"), "404.html")
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
