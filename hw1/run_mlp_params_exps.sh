# Эксперимент на vanilla policy gradient, чтобы подобрать параметры для MLP

for HIDDEN_DIM in 32 64 128 256; do
    echo "  -> Testing hidden_dim=$HIDDEN_DIM"
    python3 train.py \
        --n_epochs 500 \
        --gamma 0.99 \
        --lr 1e-3 \
        --hidden_dim $HIDDEN_DIM \
        --loss_type "vanilla" \
        --seed 33 \
        --batch_size 8 \
        --save_prefix "tuning_mlp_dim_${HIDDEN_DIM}_vanilla"
done