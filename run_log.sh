#!/bin/bash

# Function to run the Python training script with specified arguments
run_training() {
    local dataset_name=$1
    local num_virtual_tokens=$2
    local num_shared_virtual_tokens=$3
    local num_virtual_tokens_full=$4
    local log_file="4MultiMask_${dataset_name}_${num_virtual_tokens}_${num_virtual_tokens_full}_${num_shared_virtual_tokens}.log"

    nohup python scripts/train.py --lr 0.01 \
                        --subsetlr 0.01 \
                        --routerlr 0.01 \
                        --sharedEmbeddinglr 0.05 \
                        --batch_size 32 \
                        --epoch 50 \
                        --max_length 512 \
                        --model_name_or_path t5-base \
                        --tokenizer_name_or_path t5-base \
                        --warmup_ratio 0.06 \
                        --method prefix-routing \
                        --dataset_name $1 \
                        --num_virtual_tokens 1 \
                        --num_shared_virtual_tokens $3 \
                        --num_virtual_tokens_full 4 \
                        --perturb_router True \
                        --topk 1 \
                        --shareType Only \
                        --apply_adaptive_mask True\
                        --apply_adaptive_subset_mask True > "${log_file}" 2>&1 &
}

# Call the function with the specified arguments
run_training "cb" 1 10 4
# Call the function with the specified arguments
run_training "cb" 1 10 2
# Call the function with the specified arguments
run_training "cb" 1 10 8
