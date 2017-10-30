#!/usr/bin/bash

#declare -a dsets=("iris" "breast-cancer-wisconsin" "column3C" "ecoli" "ionosphere" "indian-liver" "spect-singleproton" "userknowledge" "column-2C" "pop-failures" "spect-test")

declare -a dsets=("iris")

for dset in "${dsets[@]}" 
do
    for sigma in `seq 0.02 0.005 0.08 | sed 's/,/\./'`
    do
        python sensitivity_analysis.py real -sigma $sigma -u $dset UCI
    done
    cd imgs
    bash rename.sh $dset 
done

