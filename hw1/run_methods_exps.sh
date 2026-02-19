# Эксперимент на разные loss + регуляризацию энтропии с параметром

for LOSS in "vanilla" "pg_baseline_mean" "pg_baseline_vf" "pg_rloo"; do
    for ENTROPY_REG_COEF in 0 0.01 0.05 0.1; do
        echo "  -> Testing entropy_reg_coef=$ENTROPY_REG_COEF, loss=$LOSS"
        python3 train.py \
            --n_epochs 500 \
            --gamma 0.99 \
            --lr 1e-3 \
            --hidden_dim 64 \
            --loss_type $LOSS \
            --seed 33 \
            --batch_size 8 \
            --entropy_reg \
            --entropy_reg_coef $ENTROPY_REG_COEF \
            --save_prefix "tuning_methods_entropy_reg_coef_${ENTROPY_REG_COEF}_${LOSS}"
    done
done