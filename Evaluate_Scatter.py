import torch
from scipy.stats import pearsonr  # Import pearsonr instead of spearmanr
import os
from utils.options import args_parser
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import random
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_loss_list(file_path):
    with open(file_path, 'r') as file:
        loss_list = [float(line.strip()) for line in file]
    return loss_list

def Evaluate_Scatter(args):  # Modify function name to Evaluate_Scatter


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    # File paths
    file_paths = [
        './log/Retrain/lossforget/Retrain_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat',
        './log/Proposed/lossforget/Proposed_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat',
        './log/IJ/lossforget/IJ_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat',
        './log/NU/lossforget/NU_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat',
        './log/Original/lossforget/Original_lossfile_model_{}_data_{}_epoch_{}_seed{}.dat'
    ]

    loss_lists = [load_loss_list(file_path.format(args.model, args.dataset, args.num_forget, args.epochs, args.seed)) for file_path in file_paths]

    # Calculate differences
    lower_bound = -0.675
    upper_bound = None
    diff_Proposed = pd.Series(loss_lists[1]) - pd.Series(loss_lists[4])
    diff_Proposed = np.clip(diff_Proposed, lower_bound, upper_bound)

    diff_IJ = pd.Series(loss_lists[2]) - pd.Series(loss_lists[4])
    diff_IJ = np.clip(diff_IJ, lower_bound, upper_bound)

    diff_NU = pd.Series(loss_lists[3]) - pd.Series(loss_lists[4])
    diff_NU = np.clip(diff_NU, lower_bound, upper_bound)

    diff_True = pd.Series(loss_lists[0]) - pd.Series(loss_lists[4])
    diff_True = np.clip(diff_True, lower_bound, upper_bound)

    plt.rcParams['font.sans-serif'] = ['Arial']  
    plt.rcParams['axes.unicode_minus'] = False  

    plt.figure(figsize=(8, 6))

    plt.scatter(diff_True, diff_Proposed, label='Proposed', marker='o', color='red')
    plt.scatter(diff_True, diff_IJ, label='Infinitesimal Jacknife', marker='s', color='green')
    plt.scatter(diff_True, diff_NU, label='Newton Update', marker='^', color='blue')


    min_val = min(min(diff_True), min(diff_Proposed), min(diff_IJ), min(diff_NU))
    max_val = max(max(diff_True), max(diff_Proposed), max(diff_IJ), max(diff_NU))
    diagonal_line = [min_val, max_val]
    plt.plot(diagonal_line, diagonal_line, color='gray', linestyle='--')

    plt.xlabel('Norm of Exact Paremeter Change', fontsize=18, fontweight='bold')
    plt.ylabel('Norm of Approximate Paremeter Change', fontsize=18, fontweight='bold')
    plt.title('Approximate Loss Change vs. Actual Loss Change', fontsize=18, fontweight='bold')
    plt.legend()
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15, fontweight='bold')  

    rootpath = './results/Scatter/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath) 
    plt.savefig(rootpath+'loss_comparison_plot_model_{}_data_{}_remove_{}_{}_seed{}.png'.format(args.model, args.dataset, args.num_forget, args.epochs, args.seed))
    plt.close()

def Evaluate_ScatterTEST(args):  # Modify function name to Evaluate_Scatter

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    # File paths
    file_paths = [
        './log/Retrain/lossforget/Retrain_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat',
        './log/Proposed/lossforget/Proposed_lossfile_model_{}_data_{}_remove_{}_epoch_{}_seed{}.dat',
        './log/Original/lossforget/Original_lossfile_model_{}_data_{}_epoch_{}_seed{}.dat'
    ]

    loss_lists = [load_loss_list(file_path.format(args.model, args.dataset, args.num_forget, args.epochs, args.seed)) for file_path in file_paths]

    # Calculate differences
    diff_Proposed = pd.Series(loss_lists[1]) - pd.Series(loss_lists[2])
    diff_True =  pd.Series(loss_lists[0]) - pd.Series(loss_lists[2])

    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  

    plt.figure(figsize=(8, 6))

    plt.scatter(diff_True, diff_Proposed, label='Proposed', marker='o', color='red')


    diagonal_line = [min(diff_True), max(diff_True)]
    plt.plot(diagonal_line, diagonal_line, color='gray', linestyle='--')

    plt.xlabel('Actual Loss Change', fontsize=18, fontweight='bold')
    plt.ylabel('Approximate Loss Change', fontsize=18, fontweight='bold')
    plt.title('Approximate Loss Change vs. Actual Loss Change', fontsize=18, fontweight='bold')
    plt.legend()
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15, fontweight='bold')  
    
    plt.savefig('./loss_comparison_plot.png')
    plt.close()




