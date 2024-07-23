#!/bin/bash

# Function to run the Python training script with specified arguments
run_training() {
    local dataset_name=$1
    local num_virtual_tokens=$2
    local num_shared_virtual_tokens=$3
    local num_virtual_tokens_full=$4
    local seed=$5
    mkdir -p "test_log/${dataset_name}"
    local log_file="test_log/${dataset_name}/4MultiMask_${num_shared_virtual_tokens}_seed${seed}.log"
    nohup python scripts/train.py --lr 0.01 \
                        --subsetlr 0.01 \
                        --routerlr 0.01 \
                        --sharedEmbeddinglr 0.05 \
                        --batch_size 32 \
                        --epoch 15 \
                        --max_length 512 \
                        --model_name_or_path t5-base \
                        --tokenizer_name_or_path t5-base \
                        --warmup_ratio 0.06 \
                        --method prefix-routing \
                        --dataset_name $dataset_name \
                        --num_virtual_tokens $num_virtual_tokens \
                        --num_shared_virtual_tokens $num_shared_virtual_tokens \
                        --num_virtual_tokens_full $num_virtual_tokens_full \
                        --perturb_router True \
                        --topk 1 \
                        --shareType Only \
                        --apply_adaptive_mask True \
                        --apply_adaptive_subset_mask True \
                        --seed $seed > "${log_file}" 2>&1 &
}

run_training "boolq" 1 10 4 11
wait

run_training "boolq" 1 10 4 12
wait

run_training "boolq" 1 10 4 13
wait

run_training "boolq" 1 10 4 14
wait

# Call the function with the specified arguments
# run_training "cb" 1 10 4 10
# # Call the function with the specified arguments
# run_training "cb" 1 10 2 11
# # Call the function with the specified arguments
# run_training "cb" 1 10 8 12
