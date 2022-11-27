# ALDSGD: Adjacent Leader Decentralized SGD

This repository is for paper ALDSGD: Adjacent Leader Decentralized SGD. Authors: Haoze He, Anna Choromanska. The manuscript will be submitted to International Conference on Machine Learning (ICML-23).

## Environment and Package

The code run on a environment which has `PyTorch` with `CUDA `aware` MPI`, it is compiled with `OpenMPI: 4.41, CUDA: 11.62, cuDNN: 8.14.1`. You need to install `mpi4py ` to run the code. 

- Please note that `torch.distributed` do not support complex asynchronous communication, that's why we use `mpi4py` instead.
- Please not that `PyTorch` must build on `CUDA` aware `MPI`, otherwise it can not support `mpi4py` package.



## How to use this general library? 

This code can be use as a general framework to implement any centralized/ decentralized, synchronous/ asynchronous distributed SGD algorithms. It includes **ring all reduce**, **D-PSGD**, **MATCHA**, **ALSGD**,  and **centralized SGD** with parameter server. If you want to extend the framework to any other algorithm, please :

- Go to `communicator.py` to define your communication scheme. 
- If it's decentralized SGD with topology network structure, you also need to go to `MACHA_util.py` to define your topology graph. 



## Settings for Experiments

We run the experiments using the following settings:

#### Dataset and models

he performance of all algorithms is evaluated in multiple deep learning tasks including image classification on `CIFAR-10` and `CIFAR-100`. All training datasets are evenly partitioned over a network of workers. `ResNet50`, `VGG`, `ResNet 50`, and `Wide ResNet` are provided in `models.py`.

#### Compared algorithms

We implement the proposed **LLDSGD** on the state-of-the-art algorithms **D-PSGD** and **MATCHA** with a communication budget `c_b = 0.5`. In **MATCHA**, each worker can communicate less frequently by setting a communication budget.  To run the baseline, use the following command:

```
MATCHA:
srun --job-name=MATCHA --nodes=8 --tasks-per-node=1 --cpus-per-task=1 --time=05:00:00 --mem=10GB --gres=gpu:rtx8000:1 ~/pyenv/run-pytorch-mpi.bash  python /home/hh2537/LLDSGD/run_cuda.py \
--lr 0.4 \
--bs 16 \
--epoch 200 \
--matcha \
--budget 0.5 \
-n MATCHA \
--model res \
-p \
--description experiment \
--graphid 0 \
--dataset cifar10 \
--datasetRoot ./data/ \
--savePath ./MATCHA_random_lr0.8 \
--randomSeed 1234 \
--sleep 'no' \
--isNonIID False \
--iteration 5

D-PSGD:
srun --job-name=DPSGD --nodes=8 --tasks-per-node=1 --cpus-per-task=1 --time=05:00:00 --mem=10GB --gres=gpu:rtx8000:1 ~/pyenv/run-pytorch-mpi.bash  python /home/hh2537/LLDSGD/run_cuda.py \
--lr 0.4 \
--bs 16 \
--epoch 200 \
--budget 0.5 \
-n DPSGD \
--model res \
-p \
--description experiment \
--graphid 0 \
--dataset cifar10 \
--datasetRoot ./data/ \
--savePath ./DPSGD_random_lr0.4_BS16 \
--randomSeed 1234 \
--sleep 'no' \
--isNonIID False \
--iteration 5
```

#### Machines/Clusters

All the implementations are compiled with `PyTorch` and `OpenMPI` within `mpi4py`. We conduct experiments on `NYU HPC` cluster with `100Gbit/s ` network. In all of our experiments, we use `RTX8000 GPU` as workers. 

#### Implementations

All algorithms are trained for a sufficiently long time until convergence or over-fitting. The learning rate is fine-tuned for the **D-PSGD** baseline and then used for all other algorithms. Learning rate decay and adjust according to the number of finished mini-batches in the program. The batch size of baseline and the mini-batch size of Non-blocking algorithm are the same.



## Run AL-SGD

To run the AL-DSGD, use the following commands:

```
AL-DSGD:
srun --job-name=LSGD_DPSGD --nodes=8 --tasks-per-node=1 --cpus-per-task=1 --time=05:00:00 --mem=10GB --gres=gpu:rtx8000:1 ~/pyenv/run-pytorch-mpi.bash  python /home/hh2537/LLDSGD/run_cuda.py \
--lr 0.4 \
--bs 16 \
--epoch 200 \
--budget 0.5 \
-n LLDSGD_DPSGD \
--model res \
--LLDSGD \
-p \
--description experiment \
--graphid 0 \
--dataset cifar10 \
--datasetRoot ./data/ \
--savePath ./LLDSGD_DPSGD_iter1 \
--c1 0.3 \
--c2 0.1 \
--p1 0.2 \
--p2 0.2 \
--randomSeed 1234 \
--isNonIID False \
--iteration 1
```
