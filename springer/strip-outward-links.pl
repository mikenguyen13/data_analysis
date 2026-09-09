#!/usr/bin/env perl
#
# strip-outward-links.pl
#
# Run this ONCE, inside the standalone Springer folder, after copying the eleven
# chapters across. It makes the volume self-contained by removing every reference
# to a chapter that stays behind in the online book.
#
#   perl strip-outward-links.pl            # report only, changes nothing
#   perl strip-outward-links.pl --write    # apply the changes
#
# Two passes:
#   1. Section and Chapter \@ref() references. Every one of these sits in a trailing
#      parenthetical, and the sentence names the topic in words, so deleting the
#      parenthetical leaves correct prose. Verified in the source repository.
#   2. Markdown anchor links [text](#sec-something). The link text always names the
#      topic, so the link collapses to its own text.
#
# Anything pointing INSIDE the volume is left alone.

use strict;
use warnings;

my $write = grep { $_ eq '--write' } @ARGV;

# Built from character codes so that no shell, on any platform, can mangle them.
my $BS = chr(92);   # backslash
my $AT = chr(64);   # at sign

my @outward = qw(
  sec-bigger-data-tinier-results
  sec-causal-inference
  sec-coefficient-stability-bounds
  sec-conditional-ignorability-assumption
  sec-control-function-approach
  sec-difference-in-differences
  sec-event-studies
  sec-instrumental-variables
  sec-matching-methods
  sec-mediation-analysis-explaining-the-causal-pathway
  sec-moderation-analysis-for-whom-or-under-what-conditions
  sec-modern-estimators-for-staggered-adoption
  sec-notation-quasi-experimental
  sec-overlap-positivity-assumption
  sec-partial-identification-did
  sec-propensity-scores
  sec-proxy-variables
  sec-quasi-experimental
  sec-regression-discontinuity
  sec-robustness-checks-quasi-exp
  sec-rosenbaum-bounds
  sec-selection-on-observables
  sec-selection-on-unobservables
  sec-shift-share-iv
  sec-sutva
  sec-synthetic-control
  sec-the-gold-standard-randomized-controlled-trials
  sec-two-stage-least-squares-estimation
);
my $ids = join '|', @outward;

my $ref_re    = qr/ \((?:see )?(?:Section|Chapter) \Q$BS$AT\Eref\((?:$ids)\)\)/;
my $anchor_re = qr/\[([^\]]+)\]\(#(?:$ids)\)/;

# One table reference also points outside the volume: the Rosenbaum Gamma table
# that lives in the matching methods chapter. Its sentence names the chapter, so
# the trailing parenthetical comes out the same way the section references do.
my $tab_re    = qr/ \(Table \Q$BS$AT\Eref\(tab:rosenbaum-gamma-marketing\)\)/;

my @files = glob '*.Rmd';
die "no .Rmd files here; run this inside the standalone book folder\n" unless @files;

my ($tot_ref, $tot_anchor) = (0, 0);
for my $f (@files) {
    open my $in, '<', $f or die "open $f: $!";
    local $/;
    my $t = <$in>;
    close $in;

    my $n_ref    = ($t =~ s/$ref_re//g)       || 0;
    $n_ref      += ($t =~ s/$tab_re//g)       || 0;
    my $n_anchor = ($t =~ s/$anchor_re/$1/g)  || 0;
    next unless $n_ref || $n_anchor;

    $tot_ref += $n_ref;
    $tot_anchor += $n_anchor;
    printf "%-38s %3d refs  %3d anchors\n", $f, $n_ref, $n_anchor;

    if ($write) {
        open my $out, '>', $f or die "write $f: $!";
        print $out $t;
        close $out;
    }
}
printf "\n%s: %d outward references, %d outward anchor links\n",
    ($write ? 'REWRITTEN' : 'WOULD REWRITE (pass --write to apply)'),
    $tot_ref, $tot_anchor;
