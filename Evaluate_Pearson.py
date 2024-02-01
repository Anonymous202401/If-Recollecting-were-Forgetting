import torch
from scipy.stats import pearsonr  # Import pearsonr instead of Pearsonr
from scipy.stats import spearmanr
import os
from utils.options import args_parser
import random
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import pandas as pd
def load_loss_list(file_path):
    with open(file_path, 'r') as file:
        loss_list = [float(line.strip()) for line in file]
    return loss_list



def Evaluate_Pearson(args):  # Modify function name to Evaluate_Pearson

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
    diff_Proposed = pd.Series(loss_lists[1]) - pd.Series(loss_lists[4])
    diff_IJ = pd.Series(loss_lists[2]) - pd.Series(loss_lists[4])
    diff_NU = pd.Series(loss_lists[3]) - pd.Series(loss_lists[4])
    diff_True =  pd.Series(loss_lists[0]) - pd.Series(loss_lists[4])

    # Calculate Pearson correlation coefficients
    correlation_coefficient_Proposed = pearsonr(diff_Proposed, diff_True)
    correlation_coefficient_IJ = pearsonr(diff_IJ, diff_True)
    correlation_coefficient_NU = pearsonr(diff_NU, diff_True)

    # Save results to a file
    rootpath = './results/Pearson'
    filename = 'Evaluate_Pearson_model_{}_data_{}_remove_{}_epoch_{}_seed{}.txt'.format(args.model, args.dataset, args.num_forget, args.epochs, args.seed)
    output_file_path = os.path.expanduser(os.path.join(rootpath, filename))
    os.makedirs(rootpath, exist_ok=True)

    # Open a file and write the results
    with open(output_file_path, 'w') as file:
        file.write("Pearson Correlation Coefficient (Proposed vs. True): {}\n".format(correlation_coefficient_Proposed))
        file.write("Pearson Correlation Coefficient (IJ vs. True): {}\n".format(correlation_coefficient_IJ))
        file.write("Pearson Correlation Coefficient (NU vs. True): {}\n".format(correlation_coefficient_NU))

    print("Results saved to:", output_file_path)

    print("Pearson Correlation Coefficient (Proposed vs. True):", correlation_coefficient_Proposed)
    print("Pearson Correlation Coefficient (IJ vs. True):", correlation_coefficient_IJ)
    print("Pearson Correlation Coefficient (NU vs. True):", correlation_coefficient_NU)


    # Calculate Spearman correlation coefficients
    correlation_coefficient_Proposed = spearmanr(diff_Proposed, diff_True)
    correlation_coefficient_IJ = spearmanr(diff_IJ, diff_True)
    correlation_coefficient_NU = spearmanr(diff_NU, diff_True)

    # Save results to a file
    rootpath = './results/Spearman'
    filename = 'Evaluate_Spearman_model_{}_data_{}_remove_{}_epoch_{}_seed{}.txt'.format(args.model, args.dataset, args.num_forget, args.epochs, args.seed)
    output_file_path = os.path.expanduser(os.path.join(rootpath, filename))
    os.makedirs(rootpath, exist_ok=True)

    # Open a file and write the results
    with open(output_file_path, 'w') as file:
        file.write("Spearman Correlation Coefficient (Proposed vs. True): {}\n".format(correlation_coefficient_Proposed))
        file.write("Spearman Correlation Coefficient (IJ vs. True): {}\n".format(correlation_coefficient_IJ))
        file.write("Spearman Correlation Coefficient (NU vs. True): {}\n".format(correlation_coefficient_NU))

    print("Results saved to:", output_file_path)

    print("Spearman Correlation Coefficient (Proposed vs. True):", correlation_coefficient_Proposed)
    print("Spearman Correlation Coefficient (IJ vs. True):", correlation_coefficient_IJ)
    print("Spearman Correlation Coefficient (NU vs. True):", correlation_coefficient_NU)

def Evaluate_PearsonTEST(args):  # Modify function name to Evaluate_Pearson

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

    # Calculate Pearson correlation coefficients
    # correlation_coefficient_Proposed = pearsonr(diff_Proposed, diff_True)[0]

    correlation_coefficient_Proposed = spearmanr(diff_Proposed, diff_True)


    print("Pearson Correlation Coefficient (Proposed vs. True):", correlation_coefficient_Proposed)