#!/usr/bin/perl
use strict;
use warnings;
use 5.010;

open my $paramFh, "<", "param" or die "fuck param";
open my $part2FH, ">", "part2" or die "fuck part2";
while(my $line = <$paramFh>)
{
    chomp($line);
    if($line =~ /\.param\s+(\w+)\s*=\s*(.*)/)
    {
        my $k = $1;
        my $v = $2;
        my $lb = $v > 0 ? $v * 0.5 : $v * 1.5;
        my $ub = $v > 0 ? $v * 1.5 : $v * 0.5;
        say $part2FH "\tparam $k = ($lb ~ $ub)[ic:$v];";
    }
    else
    {
        die "invalid line: $line\n";
    }
}
close $part2FH;
close $paramFh;
run_cmd("cat part1 part2 part3 > tmp.spec");
run_cmd("./sim.sh");

open my $retFh, "<", "./bak/solver.0/trace.log" or die "fuck trace.log\n";
my $vio_part;
my $fom_part;
while(my $line = <$retFh>)
{
    chomp($line);
    if($line =~ /c : (.*)/)
    {
        $vio_part = $1;
    }
    if($line =~ /y : (.*)/)
    {
        $fom_part = $1;
    }
}
close $retFh;

my $values = "$fom_part $vio_part";
open my $ofh, ">", "result.po" or die "fuck result.po";
#say $ofh "fom vio_diff_upup vio_diff_updn vio_diff_loup vio_diff_lodn vio_deviation";
say $ofh $values;
close $ofh;

sub run_cmd
{
    my $cmd = shift;
    my $ret = system($cmd);
    if($ret != 0)
    {
        die "error in run_cmd:$cmd\n";
    }
}
