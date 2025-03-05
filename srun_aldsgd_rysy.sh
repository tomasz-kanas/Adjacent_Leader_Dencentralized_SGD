#!/bin/bash
srun --job-name=MATCHA --nodes=5 --tasks-per-node=1 --cpus-per-task=1 --time=00:30:00 --gres=gpu:1 ~/venv_run.sh python run_cuda.py \
    --lr 0.4 \
    --bs 16 \
    --epoch 10 \
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
