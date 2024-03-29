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
from torchvision.datasets import ImageFolder
import os
from utils.Approximator import getapproximator
from utils.Approximator_for_celeba import  getapproximator_celeba
from utils.options import args_parser
from models.Update import  train
from models.Nets import MLP, CNNMnist, CNNCifar,Logistic,LeNet,FashionCNN4
# resnet18
from utils.subset import reduce_dataset_size
from torch.utils.data import Subset
from models.test import test_img, test_per_img
import utils.loading_data as dataset
import shutil
import joblib
from torchvision.models import resnet18

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
    elif args.model == 'cnn4' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net = FashionCNN4().to(args.device)
    elif args.model == 'lenet' and args.dataset == 'fashion-mnist':
        net = LeNet().to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'celeba':
        net = resnet18(num_classes=2).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net = resnet18(pretrained=True).to(args.device)
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


    ########### Model training
    # training
    acc_test = []
    loss_test = []
    lr = args.lr
    step=0

    for iter in range(args.epochs):
        info=[]
        t_start = time.time()
        w, loss,lr,Dataset2recollect,step,info = train(step,args=args, net=copy.deepcopy(net).to(args.device), dataset=dataset_train,learning_rate=lr,info=info)
        t_end = time.time()   
        # copy weight to net
        net.load_state_dict(w)
        # print accuracy
        net.eval()
        acc_t, loss_t = test_img(net, dataset_test, args)
        print(" Epoch {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.2f}s \n".format(iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())
        loss_test.append(loss_t)
        # load file
        path1 = "./Checkpoint/Resnet/model_{}_checkpoints". format(args.model)
        # path1 = "../../../data/sda2/qiaoxinbao/Resnet/model_{}_checkpoints". format(args.model)
        file_name = "check_{}_remove_{}_{}_seed{}_iter_{}.dat". format(args.dataset, args.num_forget,args.epochs,args.seed,iter)
        file_path = os.path.join(path1, file_name)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        info = joblib.dump(info,file_path)
        del info 
         

    ########### Save
    all_indices_train = list(range(len(dataset_train)))
    indices_to_unlearn = random.sample(all_indices_train, k=args.num_forget)
    remaining_indices = list(set(all_indices_train) - set(indices_to_unlearn))
    forget_dataset = Subset(dataset_train, indices_to_unlearn)
    remain_dataset = Subset(dataset_train, remaining_indices)
    
    ##### Compute original loss/acc on forget
    # loss
    forget_acc_list, forget_loss_list = test_img(net, forget_dataset, args)
    rootpath = './log/Original/lossforget/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    lossfile = open(rootpath + 'Proposed_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    lossfile.write(str(forget_loss_list))
    lossfile.close()
    # acc 
    rootpath = './log/Original/accforget/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    accfile = open(rootpath + 'Proposed_accfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    accfile.write(str(forget_acc_list))
    accfile.close()

    ##### Compute original loss/acc on remain
    # loss
    remain_acc_list, remain_loss_list = test_img(net, remain_dataset , args)
    rootpath = './log/Original/lossremain/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    lossfile = open(rootpath + 'Proposed_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    lossfile.write(str(remain_loss_list))
    lossfile.close()
    # acc
    rootpath = './log/Original/accremain/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    accfile = open(rootpath + 'Proposed_accfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    accfile.write(str(remain_acc_list))
    accfile.close()

    # ########### Compute unlearning statistics
    # print("Forget: ",indices_to_unlearn)
    save_path = './log/Proposed/statistics/Approximators_all_{}_{}_{}_{}.pth'.format(args.model,args.dataset,args.epochs,args.seed)

    ##############  Case 1: Large scale forgetting data
    Approximator_proposed = {j: torch.zeros_like(param) for j, param in enumerate(net.parameters())}
    if args.dataset in ['celeba', 'cifar']:
        for i in range(0, len(indices_to_unlearn), 25):
            indices_to_unlearn_i = indices_to_unlearn[i:i+25]
            Approximators=getapproximator_celeba(args,img_size,Dataset2recollect=Dataset2recollect,indices_to_unlearn=indices_to_unlearn_i)
            for idx in indices_to_unlearn_i:
                for j, param in enumerate(net.parameters()):
                    Approximator_proposed[j] += Approximators[idx][j]
            del Approximators
    else:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if not os.path.exists(save_path):
            print("Calculate unlearning statistics")
            Approximators=getapproximator(args,img_size,Dataset2recollect=Dataset2recollect)
            torch.save(Approximators, save_path)
        else:
            print("Load approximator")
            Approximators = torch.load(save_path)

    ########### Unlearning
    print("(Proposed) Begin unlearning")
    unlearn_t_start = time.time()
    model_params = net.state_dict()
    for j, param in enumerate(net.parameters()):
        param.data += Approximator_proposed[j]

    ##############  Case 2: Small scale forgetting data  ##############
    # if args.dataset in ['celeba', 'cifar']:
    #     Approximators=getapproximator_celeba(args,img_size,Dataset2recollect=Dataset2recollect,indices_to_unlearn=indices_to_unlearn)
    # else:
    #     if not os.path.exists(os.path.dirname(save_path)):
    #         os.makedirs(os.path.dirname(save_path))
    #     if not os.path.exists(save_path):
    #         print("Calculate unlearning statistics")
    #         Approximators=getapproximator(args,img_size,Dataset2recollect=Dataset2recollect)
    #         torch.save(Approximators, save_path)
    #     else:
    #         print("Load approximator")
    #         Approximators = torch.load(save_path)

    # ########### Unlearning
    # print("(Proposed) Begin unlearning")
    # unlearn_t_start = time.time()
    # model_params = net.state_dict()
    # # print(model_params)
    # Approximator_proposed = {j: torch.zeros_like(param) for j, param in enumerate(net.parameters())}
    # for idx in indices_to_unlearn:
    #     for j, param in enumerate(net.parameters()):
    #         Approximator_proposed[j] += Approximators[idx][j]
    # for j, param in enumerate(net.parameters()):
    #     param.data += Approximator_proposed[j]

    unlearn_t_end = time.time()

    acc_t, loss_t = test_img(net, dataset_test, args)
    acc_test.append(acc_t.item())
    loss_test.append(loss_t)

    print("(Proposed) Unlearned {:3d},Testing accuracy: {:.2f},Time Elapsed:  {:.2f}s \n".format(iter, acc_t, unlearn_t_end - unlearn_t_start))

    ########### Save Test
    # save unlearned model
    rootpath1 = './log/Proposed/Model/'
    if not os.path.exists(rootpath1):
        os.makedirs(rootpath1)   
    torch.save(net.state_dict(),  rootpath1+ 'Proposed_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'
               .format(args.model,args.dataset, args.num_forget,args.epochs,args.seed))

    # save approximator
    rootpath2 = './log/Proposed/Approximator/'
    if not os.path.exists(rootpath2):
        os.makedirs(rootpath2)    
    torch.save(Approximator_proposed,  rootpath2+ 'Proposed_Approximator_model_{}_data_{}_remove_{}_epoch_{}_seed{}.pth'.format(
                        args.model,args.dataset, args.num_forget,args.epochs,args.seed))

    # save ACC test
    rootpath3 = './log/Proposed/ACC/'
    if not os.path.exists(rootpath3):
        os.makedirs(rootpath3)
    accfile = open(rootpath3 + 'Proposed_accfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.
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
    plt.savefig(rootpath3 + 'Proposed_plot_model_{}_data_{}_remove_{}_epoch_{}_seed{}.png'.format(
        args.model,args.dataset, args.num_forget,args.epochs,args.seed))

    # save Loss on test
    rootpath4 = './log/Proposed/losstest/'
    if not os.path.exists(rootpath4):
        os.makedirs(rootpath4)
    lossfile = open(rootpath4 + 'Proposed_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.
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
    plt.savefig(rootpath4 + 'Proposed_plot_model_{}_data_{}_remove_{}_epoch_{}_seed{}.png'.format(
        args.model,args.dataset, args.num_forget,args.epochs,args.seed))
    

 
    ##### Compute unlearned loss/acc on forget
    # loss
    forget_acc_list, forget_loss_list = test_img(net, forget_dataset, args)
    rootpath = './log/Proposed/lossforget/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    lossfile = open(rootpath + 'Proposed_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    lossfile.write(str(forget_loss_list))
    lossfile.close()
    # acc 
    rootpath = './log/Proposed/accforget/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    accfile = open(rootpath + 'Proposed_accfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    accfile.write(str(forget_acc_list))
    accfile.close()

    ##### Compute unlearned loss/acc on remain
    # loss
    remain_acc_list, remain_loss_list = test_img(net, remain_dataset , args)
    rootpath = './log/Proposed/lossremain/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    lossfile = open(rootpath + 'Proposed_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    lossfile.write(str(remain_loss_list))
    lossfile.close()
    # acc
    rootpath = './log/Proposed/accremain/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)  
    accfile = open(rootpath + 'Proposed_accfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat'.format(
    args.model, args.dataset, args.num_forget, args.epochs, args.seed), 'w')
    accfile.write(str(remain_acc_list))
    accfile.close()
