#!/bin/bash

# Bash script to run the Python training script with specified arguments

python scripts/train.py --lr 0.5 \
                        --batch_size 32 \
                        --epoch 50 \
                        --max_length 512 \
                        --model_name_or_path t5-base \
                        --tokenizer_name_or_path t5-base \
                        --warmup_ratio 0.06 \
                        --method prefix-tuning \
                        --dataset_name rte_superglue \
                        --num_virtual_tokens 20 \
                        --num_virtual_tokens_full 80 \
                        --perturb_router True \
                        --topk 1 \
                        --seed 11