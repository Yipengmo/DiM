export CUDA_VISIBLE_DEVICES=0

model_name=DiM

for pred_len in 96 192 336 720; do
        python -u run.py \
          --is_training 1 \
          --root_path ./data/weather/ \
          --data_path weather.csv \
          --model_id weather_96_${pred_len} \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --pred_len $pred_len \
          --e_layers 1 \
          --enc_in 21 \
          --dec_in 21 \
          --c_out 21 \
          --des 'Exp' \
          --d_model 256 \
          --d_ff 256 \
          --n_heads 8 \
          --batch_size 128 \
          --learning_rate 0.001 \
          --itr 1
done
wait