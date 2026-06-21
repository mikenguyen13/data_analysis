use strict; use warnings;
# Patch bs4_book pages where the split left an empty left TOC (#book-toc).
# Copy the populated chapter list from a known-good page into every empty page.
my $good;
for my $cand ('_book/data.html','_book/index.html') {
  open(my $fh,'<',$cand) or next; local $/; my $h=<$fh>; close($fh);
  if ($h =~ /(<ul class="book-toc list-unstyled">.*?<\/ul>)/s) { $good=$1; last; }
}
die "could not extract a populated TOC\n" unless $good;
$good =~ s/ class="active"/ class=""/g;   # avoid a wrong active highlight

my $patched=0; my $already=0;
for my $f (glob('_book/*.html')) {
  open(my $fh,'<',$f) or next; local $/; my $c=<$fh>; close($fh);
  if ($c =~ /<div id="book-toc"><\/div>/) {
    $c =~ s/<div id="book-toc"><\/div>/$good/;
    open($fh,'>',$f) or next; print $fh $c; close($fh);
    $patched++;
  } elsif ($c =~ /<ul class="book-toc list-unstyled">\s*<li/) {
    $already++;
  }
}
print "TOC patch: filled $patched empty pages, $already already had it\n";
