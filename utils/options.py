#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--lr', type=float, default=0.5, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")
    parser.add_argument('--seed', type=int, default=42, help='random seed')  
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping')
    parser.add_argument('--damping_factor', type=float, default=1e-2, help="to make Hessian invertible.") 
    parser.add_argument('--test_train_rate', type=float, default=0.4, help="Ratio of test set to training set Translation")
    parser.add_argument('--regularization', type=float, default=1e-6, help="l2 regularization")



    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_forget', type=int, default=10, help="num of forgetting dataset")
    parser.add_argument('--batch_size', type=int, default=50, help="batch size")
    parser.add_argument('--num_dataset', type=int, default=100, help="number of train dataset")

    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--bs', type=int, default=2048, help="test batch size")


    args = parser.parse_args()


    return args
