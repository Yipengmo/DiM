export CUDA_VISIBLE_DEVICES=0

model_name=DiM

for pred_len in 96 192 336 720; do
        python -u run.py \
          --is_training 1 \
          --root_path ./data/electricity/ \
          --data_path electricity.csv \
          --model_id ECL_96_${pred_len} \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --pred_len $pred_len \
          --e_layers 2 \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --d_model 512 \
          --d_ff 2048 \
          --n_heads 8 \
          --batch_size 64 \
          --learning_rate 0.001 \
          --train_epochs 60 \
          --patience 8 \
          --lradj type4 \
          --itr 1
done
wait
