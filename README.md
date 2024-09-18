This repository is the supplementary code of the paper "Efficient and Generalizable Certified Unlearning: A Hessian-free Recollection Approach"
.
├── Ablation
│   ├── log
│   └── results
├── Ablation.sh
├── CelebA.sh
├── Checkpoint
│   └── model_logistic_checkpoints
├── Cifar.sh
├── data
│   ├── celeba
│   ├── cifar
│   ├── fashion-mnist
│   ├── lfw
│   ├── mnist
│   ├── MNIST
│   └── svhn
├── FMNIST.sh
├── LFW.sh
├── log
│   ├── Finetune
│   ├── IJ
│   ├── NegGrad
│   ├── NU
│   ├── Original
│   ├── Proposed
│   └── Retrain
├── main_eva.py
├── main_finetune.py
├── main_IJ.py
├── main_MIAL.py
├── main_MIAU.py
├── main_neggrad.py
├── main_NU.py
├── main_proposed.py
├── main_proposedresnet.py
├── main_retrain.py
├── MIAL.sh
├── MIAU.sh
├── MNIST.sh
├── models
│   ├── __init__.py
│   ├── Nets.py
│   ├── __pycache__
│   ├── ResNet18.py
│   ├── test.py
│   ├── Update_neggrad.py
│   ├── Update_NU.py
│   ├── Update.py
│   └── Update_retrain.py
├── output.md
├── results
│   ├── Euclidean
│   ├── MIAL
│   └── MIAU
└── utils
    ├── Approximator.py
    ├── Approximator_resnet.py
    ├── dataset.py
    ├── Evaluate_Euclidean.py
    ├── IJ.py
    ├── __init__.py
    ├── language_utils.py
    ├── loading_data.py
    ├── metrics.py
    ├── NU.py
    ├── options.py
    ├── perturbation.py
    ├── power_iteration.py
    ├── __pycache__
    └── subset.py

29 directories, 41 files



## Experiments Codebase
This section contains the bash scripts to run all the experiments for the paper.

    bash MNIST.sh
    bash FMNIST.sh
    bash Cifar.sh
    bash CelebA.sh
    bash LFW.sh
    bash MIAU.sh
    bash MIAL.sh
    bash Ablation.sh
