#!/bin/bash

width=$1
height=$2

for i in `seq 1 $width`;
do
    for i in `seq 1 $height`;
    do
        echo -n $((RANDOM % 2))
    done
    echo
done
