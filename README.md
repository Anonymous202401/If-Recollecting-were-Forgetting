This repository is the supplementary code of the paper "Efficient and Generalizable Certified Unlearning: A Hessian-free Recollection Approach"

.
├── Checkpoint
├── data
│   ├── celeba
│   ├── cifar
│   ├── fashion-mnist
│   ├── lfw
│   ├── mnist
│   ├── MNIST
│   └── svhn
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
├── models
│   ├── __init__.py
│   ├── Nets.py
│   ├── ResNet18.py
│   ├── test.py
│   ├── Update_neggrad.py
│   ├── Update_NU.py
│   ├── Update.py
│   └── Update_retrain.py
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
    └── subset.py


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
