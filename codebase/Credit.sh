for algorithm in main_proposed.py main_retrain.py main_NU.py main_IJ.py main_eva.py; do
for seed in 42 124 3407 114514 5 1; do
    python3 -u $algorithm  --model mlp --dataset creditcard --epochs 10  --num_dataset 20000 --batch_size 5000 --num_forget 500 --lr 0.01 --regularization 0.5 --lr_decay 0.995 --clip 10 --gpu 0  --seed $seed
done
done