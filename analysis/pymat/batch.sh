#!/usr/bin/bash

#declare -a dsets=("iris" "breast-cancer-wisconsin" "column3C" "ecoli" "ionosphere" "indian-liver" "spect-singleproton" "userknowledge" "column-2C" "pop-failures" "spect-test")

declare -a dsets=("iris")

for dset in "${dsets[@]}" 
do
    for sigma in `seq 0 0.5 10 | sed 's/,/\./'`
    do
        python final_metrix.py $sigma
    done
done

