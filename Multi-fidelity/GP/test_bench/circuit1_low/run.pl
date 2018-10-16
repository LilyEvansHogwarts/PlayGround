#!/usr/bin/perl
use strict;
use warnings;
use 5.010;

# script for MO-WEIBO
# minimize(-1*eff, thd, -1*pout)

`cp ./param ./origin_circuit/ && cd ./origin_circuit && perl run.pl`;

open my $fh, "<", "./origin_circuit/result.po";
chomp(my $line = <$fh>);
my($eff, $thd, $pout) = split /\s+/, $line;
close $fh;

my @objs = ($eff, $thd, $pout);
open my $ofh, ">", "result.po";
say $ofh "@objs";
close $ofh;

`cat param result.po >> backup`;
