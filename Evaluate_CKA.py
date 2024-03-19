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
import numpy as np
from torchvision import datasets, transforms
import torch
import os
from models.Nets import MLP, CNNMnist, CNNCifar,LeNet,Logistic,resnet18
from utils.subset import reduce_dataset_size
import utils.loading_data as dataset
from torch.utils.data import DataLoader
import shutil
from utils.cka import CKACalculator
from torchvision.datasets import ImageFolder


def Evaluate_CKA(args):
    ########### Setup
    pycache_folder = "__pycache__"
    if os.path.exists(pycache_folder):
        shutil.rmtree(pycache_folder)
    # parse args
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    path2="./data"
    if not os.path.exists(path2):
        os.makedirs(path2)

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
    total = sum(p.numel() for p in net.parameters())
    print("The total number of model parameters: %.2d \t" % (total )," The memory footprint for a float32 model:  %.2f M" % (total * 4 / 1e6))


    # copy weights
    net_0=copy.deepcopy(net).to(args.device)
    net_1=copy.deepcopy(net).to(args.device)
    net_2=copy.deepcopy(net).to(args.device)
    net_3=copy.deepcopy(net).to(args.device)
    # net_0.eval()
    # net_1.eval()
    # net_2.eval()
    # net_3.eval()


    model_0 = torch.load( './log/Retrain/Model/Retrain_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'
                .format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))
    net_0.state_dict(model_0)
    net_0.to(args.device)

    model_1 = torch.load( './log/Proposed/Model/Proposed_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'
                .format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))
    net_1.state_dict(model_1)
    net_1.to(args.device)

    model_2 = torch.load( './log/IJ/Model/IJ_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'
                .format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))
    net_2.state_dict(model_2) 
    net_2.to(args.device)

    model_3 = torch.load( './log/NU/Model/NU_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'
                .format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))
    net_3.state_dict(model_3)   
    net_3.to(args.device)

    dataloader = DataLoader(dataset_train, batch_size=256, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    calculator1 = CKACalculator(model1=net_0, model2=net_1, dataloader=dataloader)
    
    cka_output1 = calculator1.calculate_cka_matrix()
    print(f"CKA output size: {cka_output1.size()}")
    plt.rcParams['figure.figsize'] = (7, 7)
    xticks_labels = ['1', '2', '3', '4']
    yticks_labels = ['1', '2', '3', '4']
    plt.imshow(cka_output1.cpu().numpy(), cmap='Greens')
    plt.xticks(range(len(xticks_labels)), xticks_labels,fontsize=18, fontweight='bold')
    plt.yticks(range(len(yticks_labels)), yticks_labels, fontsize=18, fontweight='bold')
    plt.xlabel('Layer', fontsize=18, fontweight='bold')
    plt.ylabel('Layer', fontsize=18, fontweight='bold')
    plt.colorbar()
    rootpath = './results/cka/Proposed/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath) 
    plt.savefig(rootpath+'CKA_Proposed_plot_model_{}_data_{}_remove_{}_{}_seed{}.png'.format(args.model, args.dataset, args.num_forget, args.epochs, args.seed))
    plt.close()

    calculator2 = CKACalculator(model1=net_0, model2=net_2, dataloader=dataloader)
    cka_output2 = calculator2.calculate_cka_matrix()
    plt.rcParams['figure.figsize'] = (7, 7)
    plt.imshow(cka_output2.cpu().numpy(), cmap='Greens')
    xticks_labels = ['1', '2', '3', '4']
    yticks_labels = ['1', '2', '3', '4']
    plt.xticks(range(len(xticks_labels)), xticks_labels,fontsize=18, fontweight='bold')
    plt.yticks(range(len(yticks_labels)), yticks_labels, fontsize=18, fontweight='bold')
    plt.xlabel('Layer', fontsize=18, fontweight='bold')
    plt.ylabel('Layer', fontsize=18, fontweight='bold')
    plt.colorbar()
    rootpath = './results/cka/IJ/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath) 
    plt.savefig(rootpath+'CKA_IJ_plot_model_{}_data_{}_remove_{}_{}_seed{}.png'.format(args.model, args.dataset, args.num_forget, args.epochs, args.seed))
    plt.close()

    calculator3 = CKACalculator(model1=net_0, model2=net_3, dataloader=dataloader)
    cka_output3 = calculator3.calculate_cka_matrix()
    plt.rcParams['figure.figsize'] = (7, 7)
    plt.imshow(cka_output3.cpu().numpy(), cmap='Greens')
    xticks_labels = ['1', '2', '3', '4']
    yticks_labels = ['1', '2', '3', '4']
    plt.xticks(range(len(xticks_labels)), xticks_labels,fontsize=18, fontweight='bold')
    plt.yticks(range(len(yticks_labels)), yticks_labels, fontsize=18, fontweight='bold')
    plt.xlabel('Layer', fontsize=18, fontweight='bold')
    plt.ylabel('Layer', fontsize=18, fontweight='bold')
    plt.colorbar()
    rootpath = './results/cka/NU/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath) 
    plt.savefig(rootpath+'CKA_NU_plot_model_{}_data_{}_remove_{}_{}_seed{}.png'.format(args.model, args.dataset, args.num_forget, args.epochs, args.seed))
    plt.close()






def Evaluate_CKATest(args):
    ########### Setup
    pycache_folder = "__pycache__"
    if os.path.exists(pycache_folder):
        shutil.rmtree(pycache_folder)
    # parse args
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    path2="./data"
    if not os.path.exists(path2):
        os.makedirs(path2)

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
    
    elif args.dataset == 'ck':
        data_dir = './data/ckplus/'
        normalize = transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            )
        # define transforms
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.2),
            normalize,
        ])
        dataset_train  = datasets.ImageFolder(root=data_dir+'train',transform=transform)
        dataset_test = datasets.ImageFolder(root=data_dir+'test',transform=transform)
    elif args.dataset == 'utk':
        transform_train = transforms.Compose([
            transforms.Resize(size=128),
            transforms.RandomCrop(104),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize( [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size=128),
            transforms.CenterCrop(size=104),
            transforms.ToTensor(),
            transforms.Normalize( [0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
        dataset_train = ImageFolder('data/utk/utk_races/train/', transform=transform_train)
        dataset_test = ImageFolder('data/utk/utk_races/test/', transform=transform_test)
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
    total = sum(p.numel() for p in net.parameters())
    print("The total number of model parameters: %.2d \t" % (total )," The memory footprint for a float32 model:  %.2f M" % (total * 4 / 1e6))


    # copy weights
    net_0=copy.deepcopy(net).to(args.device)
    net_1=copy.deepcopy(net).to(args.device)

    # net_0.eval()
    # net_1.eval()
    # net_2.eval()
    # net_3.eval()


    model_0 = torch.load( './log/Retrain/Model/Retrain_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'
                .format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))
    net_0.state_dict(model_0)

    model_1 = torch.load( './log/Proposed/Model/Proposed_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'
                .format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))
    net_1.state_dict(model_1)




    dataloader = DataLoader(dataset_train, batch_size=256, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    calculator1 = CKACalculator(model1=net_0, model2=net_1, dataloader=dataloader)
    
    cka_output1 = calculator1.calculate_cka_matrix()
    print(f"CKA output size: {cka_output1.size()}")
    plt.rcParams['figure.figsize'] = (7, 7)
    xticks_labels = ['1', '2', '3', '4']
    yticks_labels = ['1', '2', '3', '4']
    plt.imshow(cka_output1.cpu().numpy(), cmap='Greens')
    plt.xticks(range(len(xticks_labels)), xticks_labels,fontsize=18, fontweight='bold')
    plt.yticks(range(len(yticks_labels)), yticks_labels, fontsize=18, fontweight='bold')
    plt.xlabel('Layer', fontsize=18, fontweight='bold')
    plt.ylabel('Layer', fontsize=18, fontweight='bold')
    plt.colorbar()
    rootpath = './results/cka/Proposed/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath) 
    plt.savefig(rootpath+'CKA_Proposed_plot_model_{}_data_{}_remove_{}_{}_seed{}.png'.format(args.model, args.dataset, args.num_forget, args.epochs, args.seed))
    plt.close()
