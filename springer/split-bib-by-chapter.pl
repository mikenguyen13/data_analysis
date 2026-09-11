#!/usr/bin/env perl
#
# split-bib-by-chapter.pl
#
# Write one .bib per chapter of the Springer volume, holding exactly the entries
# that chapter cites. Run from the MAIN repository root:
#
#   perl springer/split-bib-by-chapter.pl
#
# Writes springer/bib-by-chapter/ch01-....bib through ch11-....bib.
#
# Why this exists. Springer's Manuscript Guidelines require references at the end
# of each chapter rather than in the back matter, so that citation linking works
# on SpringerLink. The HTML edition does that already (bs4_book's split_bib). The
# LaTeX deliverable does not: pdf_book has no split_bib, and doing it in LaTeX
# would mean switching the whole manuscript from natbib to biblatex with
# refsection=chapter, which changes the rendering of every in-text citation in a
# 519-page book to solve a problem production re-solves anyway when it converts
# to XML.
#
# So instead of restructuring the manuscript, this hands production the mapping
# they actually need: which references belong to which chapter, machine-readable,
# alongside the single consolidated bibliography the .tex already uses.
#
# Parsing is shared with prune-bib.pl: skip fenced and inline code, ignore
# bookdown's \@ref() which looks like a citation to a naive regex, and walk the
# .bib files by brace counting so multi-line and nested entries survive.

use strict;
use warnings;

my $AT = chr(64);

# Chapter number => [title slug, source files]. Chapter 7 is four files, and
# 42.5/42.6/42.7 are sections of it rather than chapters of their own.
my @CHAPTERS = (
  [ 1, 'endogeneity',          ['36-endogeneity.Rmd'] ],
  [ 2, 'biases',               ['37-biases.Rmd'] ],
  [ 3, 'dag',                  ['38-dag.Rmd'] ],
  [ 4, 'controls',             ['39-controls.Rmd'] ],
  [ 5, 'reporting',            ['40-report.Rmd'] ],
  [ 6, 'eda',                  ['41-EDA.Rmd'] ],
  [ 7, 'sensitivity',          ['42-sensitivity-robustness.Rmd',
                                '42.5-placebo-falsification.Rmd',
                                '42.6-publication-bias-phacking.Rmd',
                                '42.7-robustness-conclusion.Rmd'] ],
  [ 8, 'replication',          ['43-rep_synthetic_data.Rmd'] ],
  [ 9, 'differential-privacy', ['43.5-differential-privacy.Rmd'] ],
  [10, 'hpc',                  ['44-hpc.Rmd'] ],
  [11, 'clustered-inference',  ['45-clustered-inference.Rmd'] ],
);
my @BIBS = qw(book.bib packages.bib references.bib references1.bib);
my $DIR  = 'springer/bib-by-chapter';

for my $b (@BIBS) {
    die "cannot find $b. Run this from the main repository root.\n" unless -e $b;
}

sub keys_in {
    my @files = @_;
    my %k;
    for my $f (@files) {
        die "cannot find $f\n" unless -e $f;
        open my $fh, '<', $f or die "open $f: $!";
        my $inchunk = 0;
        while (my $l = <$fh>) {
            if ($l =~ /^```/) { $inchunk = !$inchunk; next }
            next if $inchunk;
            $l =~ s/`[^`]*`//g;
            while ($l =~ /(?<![\\w])\Q$AT\E(-?)([A-Za-z0-9_][A-Za-z0-9_:.+\/#\$%&?<>~-]*)/g) {
                my $key = $2;
                $key =~ s/[.,;:]+$//;
                next if $key eq 'ref';
                $k{$key} = 1;
            }
        }
        close $fh;
    }
    return \%k;
}

# ---- parse the source bibliographies ------------------------------------------
my (%entry, @order);
for my $bib (@BIBS) {
    open my $fh, '<', $bib or die "open $bib: $!";
    local $/;
    my $t = <$fh>;
    close $fh;
    while ($t =~ /\Q$AT\E([A-Za-z]+)\s*\{\s*([^,\s}]+)\s*,/g) {
        my ($type, $key) = ($1, $2);
        next if lc($type) =~ /^(comment|string|preamble)$/;
        my $start = $-[0];
        my $i     = index($t, '{', $start);
        my ($depth, $j) = (0, $i);
        while ($j < length $t) {
            my $c = substr($t, $j, 1);
            $depth++ if $c eq '{';
            if ($c eq '}') { $depth--; last if $depth == 0 }
            $j++;
        }
        next if exists $entry{$key};
        $entry{$key} = substr($t, $start, $j - $start + 1);
        push @order, $key;
    }
}

# ---- write one file per chapter -----------------------------------------------
mkdir $DIR unless -d $DIR;
my (%seen_overall, @missing, $total);
for my $ch (@CHAPTERS) {
    my ($num, $slug, $files) = @$ch;
    my $want = keys_in(@$files);

    # crossref targets belong with the entry that needs them
    for my $k (keys %$want) {
        next unless exists $entry{$k};
        while ($entry{$k} =~ /crossref\s*=\s*[{"]([^}"]+)[}"]/gi) { $want->{$1} = 1 }
    }
    push @missing, grep { !exists $entry{$_} } sort keys %$want;

    my @found = grep { $want->{$_} && exists $entry{$_} } @order;
    $seen_overall{$_} = 1 for @found;
    $total += @found;

    my $path = sprintf '%s/ch%02d-%s.bib', $DIR, $num, $slug;
    open my $out, '>', $path or die "write $path: $!";
    printf $out "%% Chapter %d references, Practical Issues in Data Analysis and Reporting.\n", $num;
    printf $out "%% Source: %s\n", join(', ', @$files);
    print  $out "% Generated by springer/split-bib-by-chapter.pl. Do not edit by hand.\n\n";
    print  $out $entry{$_}, "\n\n" for @found;
    close $out;

    printf "  ch%02d %-22s %3d entries  (%s)\n", $num, $slug, scalar @found,
           scalar(@$files) == 1 ? $files->[0] : scalar(@$files) . ' files';
}

printf "\n%d entries written across %d chapters, %d unique\n",
       $total, scalar @CHAPTERS, scalar keys %seen_overall;
print  "A source cited by two chapters appears in both files, which is what a\n";
print  "per-chapter reference list means.\n";
if (@missing) {
    my %u; @u{@missing} = ();
    print "MISSING (cited but in no .bib):\n";
    print "  $_\n" for sort keys %u;
    exit 1;
}
print "all cited keys resolved\n";
