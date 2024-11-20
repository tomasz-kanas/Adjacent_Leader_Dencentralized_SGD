import os
import shutil
import json
import numpy as np
import time
import argparse

#from mpi4py import MPI
from math import ceil
from random import Random
import networkx as nx
from torch.optim import SGD
import copy

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models

from models import *



class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID = False):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        if (isNonIID == True):
            # self.partitions = np.loadtxt("./final_partition.txt").tolist()
            # for i in range(len(self.partitions)):
            #     self.partitions[i] = [int(x) for x in self.partitions[i]]
            self.partitions = self.__getNonIIDdata__(self, data, sizes, seed)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed):
        labelList = data.train_labels
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label, [])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum
        partitions = [list() for i in range(len(sizes))]
        eachPartitionLen = int(len(labelList) / len(sizes))
        majorLabelNumPerPartition = ceil(labelNum / len(partitions))
        basicLabelRatio = 0.4

        interval = 1
        labelPointer = 0

        # basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio * len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start + idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        # random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]
        return partitions



def partition_dataset(rank, size, args):

    print('==> load train data')
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.datasetRoot,
                                                train=True,
                                                download=True,
                                                transform=transform_train)

        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition,
                                                   batch_size=args.bs,
                                                   shuffle=True,
                                                   pin_memory=True)

        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root=args.datasetRoot,
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  num_workers=size)

    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])


        trainset = torchvision.datasets.CIFAR100(root=args.datasetRoot,
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)

        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition,
                                                   batch_size=args.bs,
                                                   shuffle=True,
                                                   pin_memory=True)

        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        testset = torchvision.datasets.CIFAR100(root=args.datasetRoot,
                                                train=False,
                                                download=True,
                                                transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  num_workers=size)

        print("The number of training data is:",len(trainset))
        print("The number of test data is:",len(testset))

    elif args.dataset == 'imagenet':
        datadir = args.datasetRoot
        traindir = os.path.join(datadir, 'CLS-LOC/train/')
        # valdir = os.path.join(datadir, 'CLS-LOC/')
        # testdir = os.path.join(datadir, 'CLS-LOC/')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)

        train_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.bs, shuffle=True,
            pin_memory=True)
        '''
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.bs, shuffle=False,
            pin_memory=True)
        val_loader = None
        '''
        test_loader = None

    if args.dataset == 'emnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.EMNIST(root=args.datasetRoot,
                                                    split='balanced',
                                                    train=True,
                                                    download=True,
                                                    transform=transform_train)
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)

        train_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.bs, shuffle=True,
            pin_memory=True)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.EMNIST(root=args.datasetRoot,
                                              split='balanced',
                                              train=False,
                                              download=True,
                                              transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  num_workers=size)

    return train_loader, test_loader


def select_model(num_class, args):
    if args.model == 'VGG':
        model = vggnet.VGG(16, num_class)
    elif args.model == 'res':
        if args.dataset == 'cifar10':
            # model = large_resnet.ResNet18()
            model = resnet.ResNet(50, num_class)
        elif args.dataset == 'imagenet':
            model = models.resnet18()
    elif args.model == 'wrn':
        model = wrn.Wide_ResNet(28, 10, 0, num_class)
    elif args.model == 'mlp':
        if args.dataset == 'emnist':
            model = MLP.MNIST_MLP(47)
    return model


def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder(object):
    def __init__(self, args, rank):
        #self.record_accuracy = list()
        #self.record_timing = list()
        #self.record_comp_timing = list()
        #self.record_comm_timing = list()
        #self.record_losses = list()
        #self.record_trainacc = list()
        #self.total_record_timing = list()
        self.args = args
        self.rank = rank
        self.saveFolderName = f"results/{args.savePath}"
        if rank == 0 and self.args.save:
            if os.path.exists(self.saveFolderName):
                shutil.rmtree(self.saveFolderName)
            os.mkdir(self.saveFolderName)
        self.training_stats = {
            "batch_time": [],
            "communication_time": [],
            "computing_time": [],
            "train_loss": [],
            #"test_loss": [],
            "test_acc": []
        }


    #def add_new(self, record_time, comp_time, comm_time, epoch_time, top1, losses, test_acc):
    #    self.total_record_timing.append(np.array(record_time))
    #    self.record_timing.append(np.array(epoch_time))
    #    self.record_comp_timing.append(np.array(comp_time))
    #    self.record_comm_timing.append(np.array(comm_time))
    #    self.record_trainacc.append(np.array(top1.cpu()))
    #    self.record_losses.append(np.array(losses))
    #    self.record_accuracy.append(np.array(test_acc.cpu()))
    #    print("test accuracy is:", test_acc)

    def add_new(self, record_time, comp_time, comm_time, epoch_time, losses, test_acc):
        self.training_stats["batch_time"].append(record_time)
        self.training_stats["computing_time"].append(comp_time)
        self.training_stats["communication_time"].append(comm_time)
        self.training_stats["train_loss"].append(losses)
        self.training_stats["test_acc"].append(test_acc.cpu())
        print("test accuracy is:", test_acc)

    def add_total_time(self):
        self.training_stats["total_time"] = sum(self.training_stats["batch_time"])

    #def save_to_file(self):
    #    np.savetxt(
    #        self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
    #            self.rank) + '-recordtime.log', self.total_record_timing, delimiter=',')
    #    np.savetxt(
    #        self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
    #            self.rank) + '-time.log', self.record_timing, delimiter=',')
    #    np.savetxt(
    #        self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
    #            self.rank) + '-comptime.log', self.record_comp_timing, delimiter=',')
    #    np.savetxt(
    #        self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
    #            self.rank) + '-commtime.log', self.record_comm_timing, delimiter=',')
    #    np.savetxt(
    #        self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
    #            self.rank) + '-acc.log', self.record_accuracy, delimiter=',')
    #    np.savetxt(
    #        self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
    #            self.rank) + '-losses.log', self.record_losses, delimiter=',')
    #    np.savetxt(
    #        self.saveFolderName + '/dsgd-lr' + str(self.args.lr) + '-budget' + str(self.args.budget) + '-r' + str(
    #            self.rank) + '-tacc.log', self.record_trainacc, delimiter=',')
    #    with open(self.saveFolderName + '/ExpDescription', 'w') as f:
    #        f.write(str(self.args) + '\n')
    #        f.write(self.args.description + '\n')

    def save_to_file(self):
        with open(f"{self.saveFolderName}/{self.rank}.json", 'w') as f:
            json.dump(self.training_stats, f)



def test(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    # correct = 0
    # total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        outputs = model(inputs)
        acc1 = comp_accuracy(outputs, targets)
        top1.update(acc1[0], inputs.size(0))
    return top1.avg


def collectGradient(optimizer):
    gradient = list()
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue
            # get the gradient
            d_p = p.grad.data
            # operate the gradient according to weight_decay and momentum
            if weight_decay != 0:
                d_p.add_(p.data, alpha=weight_decay)
            if momentum != 0:
                param_state = optimizer.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            gradient.append(d_p)
            # p.data.add_(-group['lr'], d_p)
    return gradient



class newSGD(SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, maximize=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        super(SGD, self).__init__(params, defaults)

    def step(self, gradient, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            for i, param in enumerate(params_with_grad):
                d_p = gradient[i]
                alpha = lr if maximize else -lr
                param.data.add_(d_p, alpha=alpha)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
        return loss
