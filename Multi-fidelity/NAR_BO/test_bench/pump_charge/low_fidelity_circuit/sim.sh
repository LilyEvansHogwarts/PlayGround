#!/bin/bash
source ~/.bashrc
rm -f opt.spec
dir=`pwd`
echo folder::prj_dir is \"$dir\"\; > opt.spec
cat tmp.spec >> opt.spec
mpiexec -n 12 msp.multiprocess ~/mysoft/msp/lic.dat opt.spec
