export CUDA_VISIBLE_DEVICES=0

model_name=DiM

for pred_len in 96 192 336 720; do
        python -u run.py \
          --is_training 1 \
          --root_path ./data/ETT/ \
          --data_path ETTm2.csv \
          --model_id ETTm2_96_${pred_len} \
          --model $model_name \
          --data ETTm2 \
          --features M \
          --seq_len 96 \
          --pred_len $pred_len \
          --e_layers 1 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --d_model 512 \
          --d_ff 2048 \
          --n_heads 8 \
          --batch_size 32 \
          --learning_rate 0.0002 \
          --itr 1
done
wait