#!/bin/bash
mpiexec -n 5 python run_cuda.py \
    --lr 0.4 \
    --bs 16 \
    --epoch 3 \
    --LLDSGD \
    --budget 0.5 \
    -n LLDSGD \
    --model res \
    -p \
    --description experiment \
    --graphid 7 \
    --dataset cifar10 \
    --datasetRoot ~/data/ \
    --savePath MATCHA \
    --randomSeed 1234 \
    --isNonIID False \
    --iteration 5
