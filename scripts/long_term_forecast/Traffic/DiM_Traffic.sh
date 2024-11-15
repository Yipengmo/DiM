export CUDA_VISIBLE_DEVICES=0

model_name=DiM

for pred_len in 96 192 336 720; do
        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/traffic/ \
          --data_path traffic.csv \
          --model_id traffic_96_${pred_len} \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --pred_len $pred_len \
          --e_layers 6 \
          --enc_in 862 \
          --dec_in 862 \
          --c_out 862 \
          --des 'Exp' \
          --d_model 512\
          --d_ff 2048 \
          --n_heads 8 \
          --batch_size 32 \
          --learning_rate 0.0002 \
          --train_epochs 100 \
          --patience 10 \
          --lradj type4 \
          --affine False \
          --itr 1
done
wait