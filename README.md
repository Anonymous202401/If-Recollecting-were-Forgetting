This repository is the supplementary code of the paper "**Efficient and Generalizable Second-Order Unlearning: A Hessian-free Online Model Updates Approach**".

# Quick Start Guide
### 1. Model Learning and Unlearning
We first obtain the learned model and the unlearned models, which can be done using the following commands:

**Proposed Hessian-Free method**:

        python3 -u main_proposed.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget 50 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        
**Retraining method**:

        python3 -u main_retrain.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget 50 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42

**Newton Step method (Unofficial implementation)**:

        python3 -u main_NU.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget 50 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42

**Infinitesimal Jackknife method (Unofficial implementation)**:

        python3 -u main_IJ.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget 50 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42

All results are included in ./log.

### 2. Evaluate Distance
We then evaluate the $L_2$ norm between unleaned model and retrained model:

        python3 -u main_eva.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget 50 --lr 0.05 --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42

All results are included in ./results.


### 3. Membership Inference Attack
Finally, we conduct the MIA (Membership Inference Attack) evaluation:

        python3 -u main_MIAU.py --model logistic --dataset mnist --epochs 15 --num_dataset 1000 --batch_size 32 --num_forget 50 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0 --seed 42
        python3 -u main_MIAL.py --model logistic --dataset mnist --epochs 15 --num_dataset 1000 --batch_size 32 --num_forget 50 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0 --seed 42

All results are included in ./results.

# Experiments Codebase
This section contains the bash scripts to run all the experiments for the paper.

    bash MNIST.sh
    bash FMNIST.sh
    bash Cifar.sh
    bash CelebA.sh
    bash LFW.sh
    bash MIAU.sh
    bash MIAL.sh
    bash Ablation.sh
