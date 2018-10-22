#!/bin/bash
dir=`pwd`
echo "workdir $dir/" > conf
cat tmp.conf >> conf
# nohup weibo conf > run.log 2> err.log &
nohup mpiexec -n 2 -f mach apaweibo conf > run.log 2> err.log &
# nohup runCircuitOpt.sh ./conf > run.log 2>&1 &
