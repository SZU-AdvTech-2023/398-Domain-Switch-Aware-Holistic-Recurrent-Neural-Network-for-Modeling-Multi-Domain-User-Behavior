#!/bin/bash
for behavior_regularizer_weight in 0.1 0.01 0.001
do
  for supplement_loss_weight in 0.1 0.01 0.001
  do
    CUDA_VISIBLE_DEVICES=0,1,2, python main.py --train_dir=default --maxlen=300 --batch_size=128 --num_epochs=500 --device=cuda --hidden_units=50 --interval=20 --load_processed_data=true --l2_emb=0.001 --behavior_regularizer_weight=${behavior_regularizer_weight} --supplement_loss_weight=${supplement_loss_weight}
  done
done