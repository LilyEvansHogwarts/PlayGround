#!/usr/bin/perl
use strict;
use warnings;
use 5.010;

# chomp(my $dir = `pwd`);
# say "pwd: $dir";

# Read the param file, and write to ocan desvar file
open my $para_fh, "<", "param" or die "Can't read param:$!\n";
open my $desvar_fh, ">", "./PA/part2" or die "Can't create part2: $!\n";
while(my $line = <$para_fh>)
{
    chomp($line);
    if($line =~ /\.param\s+(\w+)\s*=\s*(.*?)$/)
    {
        my $name = $1;
        my $val  = $2 + 0.0;
        say $desvar_fh "desVar( \"$name\" $val )";
    }
}
close $desvar_fh;
close $para_fh;
run_cmd("cat param ./PA/part2");
run_cmd("cd ./PA && cat part1 part2 part3 > oceanScript.ocn");
run_cmd("cd ./PA && rm -f sim.out ocean.log* err");
run_cmd("cd ./PA && ./run.sh");

open my $sim_fh, "<", "./PA/sim.out" or die "Can't open sim.out:$!\n";
open my $ofh, ">", "./result.po";
while(my $line = <$sim_fh>)
{
    chomp($line);
    if($line)
    {
        say " maximize eff";
        say " s.t.     thd  < 13.65";
        say "          pout > 23 dBm";
        my ($pout, $thd, $pdc, $eff) = split /\s+/, $line;
        # my $obj      = -1 * $eff;
        # my $vio_thd  = $thd - 13.65;
        # my $vio_pout = 23 - $pout;
        # say $ofh $obj $vio_thd $vio_pout";
        say $ofh "$eff $thd $pout";
        last;
    }
}
close $ofh;
close $sim_fh;

sub run_cmd
{
    my $cmd = shift;
    my $ret = system($cmd);
    if($ret != 0)
    {
        die "Fail to run cmd: $cmd\n";
    }
}
