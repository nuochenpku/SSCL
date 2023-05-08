#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# about how to use PyTorch's distributed data parallel.

python train.py \
    --model_name_or_path bert-base-uncased\
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/bert-base_avg_neg2_trans \
    --do_neg \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64\
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --eval_transfer \
    --evaluation_strategy steps \
    --metric_for_best_model eval_avg_transfer \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type avg \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
#     --eval_transfer \
