#!/usr/bin/env bash
#
#SBATCH --job-name=ALDSGD
#SBATCH --time=30
#SBATCH --output=output.txt
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

echo "NODELIST="${SLURM_NODELIST}

source ~/venv/bin/activate

srun python3 run_cuda.py \
    --lr 0.4 \
    --bs 16 \
    --epoch 5 \
    --LLDSGD \
    --budget 0.5 \
    -n LLDSGD \
    --model res \
    -p \
    --description experiment \
    --graphid 7 \
    --dataset cifar10 \
    --datasetRoot ~/data/ \
    --savePath LLDSGD \
    --c1 0.3 \
    --c2 0.1 \
    --p1 0.2 \
    --p2 0.2 \
    --randomSeed 1234 \
    --isNonIID False \
    --iteration 1
