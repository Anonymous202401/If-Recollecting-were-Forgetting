This repository is the supplementary code of the paper "Efficient and Generalizable Certified Unlearning: A Hessian-free Recollection Approach"


When implementing the NS and IJ methods, please note that storing the Hessian for CNN and LeNet requires 1.78GB and 14.18GB of space respectively. Ensure that you have sufficient space to save the corresponding results.

We recommend prioritizing the execution of the NS method over IJ because IJ can utilize the results of NS for calculation, thereby significantly saving computing time of experiments.

## Proposed Method
    python3 -u main_proposed.py --model logistic --dataset mnist  --epochs 50 --num_dataset 1000 --batch_size 1000 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_proposed.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_proposed.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget 800 --lr 0.5 --clip 0.5 --gpu 7  --seed 42
    python3 -u main_proposed.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256  --num_forget 800 --lr 0.5 --clip 0.5  --gpu 7  --seed 42
    
    python3 -u main_proposedresnet.py --model resnet18 --dataset celeba --epochs 5 --num_dataset 10000 --batch_size 64 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.01 --clip 10 --gpu 7 --seed 42
    python3 -u main_proposedresnet.py --model resnet18 --dataset lfw --epochs 49 --num_dataset 984 --batch_size 41 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.004 --clip 5.5 --gpu 7 --seed 42
    python3 -u main_proposedresnet.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --regularization 1e-2 --lr_decay 0.99995 --clip 10  --gpu 7  --seed 930


## NS Method (Unofficial implementation)
    python3 -u main_NU.py --model logistic --dataset mnist  --epochs 50 --num_dataset 1000 --batch_size 1000 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_NU.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_NU.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget 800 --lr 0.5 --clip 0.5 --gpu 7  --seed 42
    python3 -u main_NU.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256  --num_forget 800 --lr 0.5 --clip 0.5  --gpu 7  --seed 42

## IJ Method (Unofficial implementation)
    python3 -u main_IJ.py --model logistic --dataset mnist  --epochs 50 --num_dataset 1000 --batch_size 1000 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_IJ.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_IJ.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget 800 --lr 0.5 --clip 0.5 --gpu 7  --seed 42
    python3 -u main_IJ.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256  --num_forget 800 --lr 0.5 --clip 0.5  --gpu 7  --seed 42

## Retrain Method
    python3 -u main_retrain.py --model logistic --dataset mnist  --epochs 50 --num_dataset 1000 --batch_size 1000 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_retrain.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_retrain.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget 800 --lr 0.5 --clip 0.5 --gpu 7  --seed 42
    python3 -u main_retrain.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256  --num_forget 800 --lr 0.5 --clip 0.5  --gpu 7  --seed 42

    python3 -u main_retrain.py --model resnet18 --dataset celeba --epochs 5 --num_dataset 10000 --batch_size 64 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.01 --clip 10 --gpu 7 --seed 42
    python3 -u main_retrain.py --model resnet18 --dataset lfw --epochs 49 --num_dataset 984 --batch_size 41 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.004 --clip 5.5 --gpu 7 --seed 42
    python3 -u main_retrain.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --regularization 1e-2 --lr_decay 0.99995 --clip 10  --gpu 7  --seed 930

## Additional Baseline Experiments

    python3 -u main_finetune.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --regularization 1e-2 --lr_decay 0.99995 --clip 10  --gpu 7  --seed 930
    python3 -u main_neggrad.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --regularization 1e-2 --lr_decay 0.99995 --clip 10  --gpu 7  --seed 930

    python3 -u main_finetune.py --model resnet18 --dataset lfw --epochs 49 --num_dataset 984 --batch_size 41 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.004 --clip 5.5 --gpu 7 --seed 42
    python3 -u main_neggrad.py --model resnet18 --dataset lfw --epochs 49 --num_dataset 984 --batch_size 41 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.004 --clip 5.5 --gpu 7 --seed 42

    python3 -u main_finetune.py --model resnet18 --dataset celeba --epochs 5 --num_dataset 10000 --batch_size 64 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.01 --clip 10 --gpu 7 --seed 42
    python3 -u main_neggrad.py --model resnet18 --dataset celeba --epochs 5 --num_dataset 10000 --batch_size 64 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.01 --clip 10 --gpu 7 --seed 42

It's important to emphasize the challenges in evaluating prior studies NS and IJ on larger models, mainly due to their high complexity requirements and more restrictive assumptions. Therefore, we opt to use the following baseline methods instead:
1. ***FineTune:** In case of finetuning, the original learned model is finetuned on the remaining dataset.* 
2. ***NegGrad**: In case of gradient ascent, the learned model is finetuned using negative of the models gradients on the forgetting dataset.*

## Experiments Codebase
This section contains the bash scripts to run all the experiments for the paper.

    bash MNIST.sh
    bash FMNIST.sh
    bash Cifar.sh
    bash CelebA.sh
    bash LFW.sh
    
## Evaluation
Please be aware that the evaluation results can only be saved in "./result" once all processes for NS, IJ, and our method have completed data removal. Otherwise, only the results of our method will be printed.

    python3 -u main_eva.py --model logistic --dataset mnist  --epochs 50 --num_dataset 1000 --batch_size 1000 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_eva.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_eva.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget 800 --lr 0.5 --clip 0.5 --gpu 7  --seed 42
    python3 -u main_eva.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256  --num_forget 800 --lr 0.5 --clip 0.5  --gpu 7  --seed 42
    
    python3 -u main_eva.py --model resnet18 --dataset celeba --epochs 5 --num_dataset 10000 --batch_size 64 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.01 --clip 10 --gpu 7 --seed 42
    python3 -u main_eva.py --model resnet18 --dataset lfw --epochs 49 --num_dataset 984 --batch_size 41 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.004 --clip 5.5 --gpu 7 --seed 42
    python3 -u main_eva.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --regularization 1e-2 --lr_decay 0.99995 --clip 10  --gpu 7  --seed 930



