for algorithm in main_proposed.py main_retrain.py main_NU.py main_IJ.py main_eva.py; do
for seed in 42 124 3407 114514 5 1; do
    python3 -u $algorithm  --model mlp --dataset cancer --epochs 35  --num_dataset 569 --batch_size 64 --num_forget 28 --lr 0.05 --regularization 0.1 --lr_decay 0.9995 --clip 10 --gpu 0  --seed $seed

    # python3 -u $algorithm  --model mlp --dataset cancer --epochs 190 --num_dataset 569 --batch_size 64 --num_forget 10 --lr 0.01 --regularization 0.1 --lr_decay 0.9995 --clip 10 --gpu 0  --seed $seed
done
done