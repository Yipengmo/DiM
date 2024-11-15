export CUDA_VISIBLE_DEVICES=0

model_name=DiM

for pred_len in 96 192 336 720; do
      python -u run.py \
        --is_training 1 \
        --root_path ./data/ETT/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_${pred_len} \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --e_layers 1 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 256 \
        --n_heads 4 \
        --batch_size 128 \
        --learning_rate 0.001 \
        --itr 1
done
wait