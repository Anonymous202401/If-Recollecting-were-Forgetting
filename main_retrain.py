#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from torchvision.datasets import ImageFolder
import numpy as np
from torchvision import datasets, transforms
import torch
import os
from utils.options import args_parser
from models.Update_retrain import  train
from models.Nets import MLP, CNNMnist, CNNCifar,LeNet,Logistic,resnet18
from utils.subset import reduce_dataset_size
from models.test import test_img,test_per_img
import utils.loading_data as dataset
from torch.utils.data import Subset
import shutil

if __name__ == '__main__':

    ########### Setup
    pycache_folder = "__pycache__"
    if os.path.exists(pycache_folder):
        shutil.rmtree(pycache_folder)
    # parse args
    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)


    path2="./data"
    if not os.path.exists(path2):
        os.makedirs(path2)


    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    # load dataset and split users
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
    elif args.dataset == 'cifar':

        transform = transforms.Compose([transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])     
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
    elif args.dataset == 'celeba':
        args.num_classe = 2
        args.bs = 64
        custom_transform =transforms.Compose([
                                                transforms.Resize((128, 128)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])


        dataset_train = dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-train.csv',
                                      img_dir='./data/celeba/img_align_celeba/',
                                      transform=custom_transform)
        # valid_celeba= dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-valid.csv',
        #                               img_dir='./data/celeba/img_align_celeba/',
        #                               transform=custom_transform)
        dataset_test = dataset.CelebaDataset(csv_path='./data/celeba/celeba-gender-test.csv',
                                     img_dir='./data/celeba/img_align_celeba/',
                                     transform=custom_transform)
    else:
        exit('Error: unrecognized dataset')

    
    dataset_train = reduce_dataset_size(dataset_train, args.num_dataset,random_seed=args.seed)
    testsize = math.floor(args.num_dataset * args.test_train_rate)
    dataset_test = reduce_dataset_size(dataset_test,testsize,random_seed=args.seed)
    img_size = dataset_train[0][0].shape

    net = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net = CNNMnist(args=args).to(args.device)
    elif args.model == 'lenet' and args.dataset == 'fashion-mnist':
        net = LeNet().to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'celeba':
        net = resnet18(num_classes=2).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'logistic':
        len_in = 1
        for x in img_size:
            len_in *= x
        net = Logistic(dim_in=len_in,  dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net)
    net.train()
    # copy weights
    w = net.state_dict()
    total = sum(p.numel() for p in net.parameters())
    print("The total number of model parameters: %.2d \t" % (total )," The memory footprint for a float32 model:  %.2f M" % (total * 4 / 1e6))

    # training
    acc_test = []
    loss_test = []
    lr = args.lr
    info=[]
    step=0
    all_indices = list(range(len(dataset_train)))
    indices_to_unlearn = random.sample(all_indices, k=args.num_forget)


    for iter in range(args.epochs):
        t_start = time.time()
        w, loss,lr,step = train(step,args=args, net=copy.deepcopy(net).to(args.device), dataset=dataset_train,  learning_rate=lr, indices_to_unlearn=indices_to_unlearn)
        t_end = time.time()   

        # copy weight to net
        net.load_state_dict(w)
        # print accuracy
        net.eval()
        acc_t, loss_t = test_img(net, dataset_test, args)
         
        print(" Epoch {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.2f}s \n".format(iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())
        loss_test.append(loss_t)
    print(" Retrained {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.2f}s \n".format(iter, acc_t, t_end - t_start))


    
    # save model
    rootpath1 = './log/Retrain/Model/'
    if not os.path.exists(rootpath1):
        os.makedirs(rootpath1)
    torch.save(net.state_dict(),  rootpath1+ 'Retrain_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))

    # save acc
    rootpath3 = './log/Retrain/ACC/'
    if not os.path.exists(rootpath3):
        os.makedirs(rootpath3)
    accfile = open(rootpath3 + 'Retrain_accfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.
                   format(args.model,args.dataset, args.num_forget,args.epochs,args.seed), "w")
    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()
    # plot acc curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath3 + 'Retrain_plot_model_{}_data_{}_remove_{}_epoch_{}_seed{}.png'.format(
        args.model,args.dataset, args.num_forget,args.epochs,args.seed))

    # save Loss
    rootpath4 = './log/Retrain/losstest/'
    if not os.path.exists(rootpath4):
        os.makedirs(rootpath4)
    lossfile = open(rootpath4 + 'Retrain_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.
                   format(args.model,args.dataset, args.num_forget,args.epochs,args.seed), "w")

    for loss in loss_test:
        sloss = str(loss)
        lossfile.write(sloss)
        lossfile.write('\n')
    lossfile.close()
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_test)), loss_test)
    plt.ylabel('test loss')
    plt.savefig(rootpath4 + 'Retrain_plot_model_{}_data_{}_remove_{}_epoch_{}_seed{}.png'.format(
        args.model,args.dataset, args.num_forget,args.epochs,args.seed))


    # Compute loss to Evaluate_Spearman
    _, test_loss_list = test_per_img(net, dataset_train, args,indices_to_test=indices_to_unlearn)
    rootpath = './log/Retrain/lossforget/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)    
    lossfile = open(rootpath + 'Retrain_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    for loss in test_loss_list:
        sloss = str(loss)
        lossfile.write(sloss)
        lossfile.write('\n')
    lossfile.close()

    # # Compute loss to Evaluate_Spearman
    # all_indices = list(range(len(dataset_test)))
    # indices_to_test = random.sample(all_indices, k=100)
    # _, test_loss_list = test_per_img(net, dataset_test, args,indices_to_test=indices_to_test)
    # rootpath = './log/Retrain/lossforget/'
    # if not os.path.exists(rootpath):
    #     os.makedirs(rootpath)    
    # lossfile = open(rootpath + 'Retrain_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    # args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    # for loss in test_loss_list:
    #     sloss = str(loss)
    #     lossfile.write(sloss)
    #     lossfile.write('\n')
    # lossfile.close()
 


    # # Test
    # batch_idx_list_per_batch0=info[0]
    # batch_idx_list_per_batch1=info[1]
    # batch_idx_list_per_batch2=info[2]
    # batch_idx_list_per_batch3=info[3]
    # print("Forget: ",indices_to_unlearn)
    # b1=batch_idx_list_per_batch0["batch_idx_list"]
    # b2=batch_idx_list_per_batch1["batch_idx_list"]
    # b3=batch_idx_list_per_batch2["batch_idx_list"]
    # b4=batch_idx_list_per_batch3["batch_idx_list"]
    # print(b1)
    # print(b2)
    # print(b3)
    # print(b4)
    # # exist same number?
    # intersection = set(m) & set(l)
    # if intersection:
    #     print("same")
    #     print("same lenth:", len(intersection))
    # else:
    #     print("not same")









