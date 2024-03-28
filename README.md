This repository is the supplementary code of the paper "Efficient Online Unlearning via Hessian-Free Recollection of Individual Data Statistics"


Note that when executing the methods of NS and IJ, storing the Hessian for CNN and LeNet requires 1.78GB and 14.18GB of space, please make sure you have enough space to save the corresponding results.

We recommend executing the NS method in preference to IJ because IJ can use the results of NS to calculate and thus significantly reduce the computing time.

## Additional Experiments (Rebuttal)

It is important to highlight the challenges associated with evaluating the prior studies NS and IJ within the rebuttal period, primarily due to their high complexity requirements and the more restrictive assumptions made. 
Hence, we instead use the following baseline methods:

    python3 -u main_proposedresnet.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --clip 10  --gpu 1  --seed 930
    python3 -u main_retrain.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --clip 10  --gpu 2  --seed 930
    python3 -u main_finetune.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --clip 10  --gpu 3  --seed 930
    python3 -u main_neggrad.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --clip 10  --gpu 4  --seed 930
    python3 -u main_eva.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --clip 10  --gpu 5  --seed 930

    python3 -u main_proposedresnet.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  5000 --lr 0.001 --clip 10  --gpu 1  --seed 930
    python3 -u main_retrain.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  5000 --lr 0.001 --clip 10  --gpu 2  --seed 930
    python3 -u main_finetune.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  5000 --lr 0.001 --clip 10  --gpu 3  --seed 930
    python3 -u main_neggrad.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  5000 --lr 0.001 --clip 10  --gpu 4  --seed 930
    python3 -u main_eva.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  5000 --lr 0.001 --clip 10  --gpu 5  --seed 930

## Proposed Method
    python3 -u main_proposed.py --model logistic --dataset mnist  --epochs 50 --num_dataset 1000 --batch_size 1000 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_proposed.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_proposed.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget 800 --lr 0.5 --clip 0.5 --gpu 7  --seed 42
    python3 -u main_proposed.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256  --num_forget 400 --lr 0.5 --clip 0.5  --gpu 7  --seed 42
    python3 -u main_proposedresnet.py --model resnet18 --dataset celeba --epochs 10 --num_dataset 1000 --batch_size 32 --num_forget 200  --lr 0.05 --clip 10 --gpu 7  --seed 42

## NS Method (Unofficial implementation)
    python3 -u main_NU.py --model logistic --dataset mnist  --epochs 50 --num_dataset 1000 --batch_size 1000 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_NU.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_NU.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget 800 --lr 0.5 --clip 0.5 --gpu 7  --seed 42
    python3 -u main_NU.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256  --num_forget 400 --lr 0.5 --clip 0.5  --gpu 7  --seed 42

## IJ Method (Unofficial implementation)
    python3 -u main_IJ.py --model logistic --dataset mnist  --epochs 50 --num_dataset 1000 --batch_size 1000 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_IJ.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_IJ.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget 800 --lr 0.5 --clip 0.5 --gpu 7  --seed 42
    python3 -u main_IJ.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256  --num_forget 400 --lr 0.5 --clip 0.5  --gpu 7  --seed 42

## Retrain Method
    python3 -u main_retrain.py --model logistic --dataset mnist  --epochs 50 --num_dataset 1000 --batch_size 1000 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_retrain.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_retrain.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget 800 --lr 0.5 --clip 0.5 --gpu 7  --seed 42
    python3 -u main_retrain.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256  --num_forget 400 --lr 0.5 --clip 0.5  --gpu 7  --seed 42

## Evaluation
Please note that the evaluation results can only be saved in "./result" when all the NS, IJ, and ours have completed data removal. Otherwise, only the results of our method will be printed.

    python3 -u main_eva.py --model logistic --dataset mnist  --epochs 50 --num_dataset 1000 --batch_size 1000 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_eva.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 7  --seed 42
    python3 -u main_eva.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget 800 --lr 0.5 --clip 0.5 --gpu 7  --seed 42
    python3 -u main_eva.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256  --num_forget 400 --lr 0.5 --clip 0.5  --gpu 7  --seed 42
    python3 -u main_eva.py --model resnet18 --dataset celeba --epochs 10 --num_dataset 1000 --batch_size 32 --num_forget 200  --lr 0.05 --clip 10 --gpu 7  --seed 42

