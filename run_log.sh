#!/bin/bash

# Function to run the Python training script with specified arguments
run_training() {
    local dataset_name=$1
    local num_virtual_tokens=$2
    local num_shared_virtual_tokens=$3
    local num_virtual_tokens_full=$4
    local log_file="FixSubsetAdaptive_${dataset_name}_${num_virtual_tokens}_${num_virtual_tokens_full}_${num_shared_virtual_tokens}.log"

    nohup python scripts/train.py --lr 0.005 \
                                  --subsetlr 0.05 \
                                  --routerlr 0.0005 \
                                  --sharedEmbeddinglr 0.05 \
                                  --batch_size 32 \
                                  --epoch 50 \
                                  --max_length 256 \
                                  --model_name_or_path t5-base \
                                  --tokenizer_name_or_path t5-base \
                                  --warmup_ratio 0.06 \
                                  --method prefix-routing \
                                  --dataset_name "${dataset_name}" \
                                  --num_virtual_tokens "${num_virtual_tokens}" \
                                  --num_shared_virtual_tokens "${num_shared_virtual_tokens}" \
                                  --num_virtual_tokens_full "${num_virtual_tokens_full}" \
                                  --perturb_router True \
                                  --topk 1 \
                                  --shareType None \
                                  --apply_adaptive_mask True > "${log_file}" 2>&1 &
}

# Call the function with the specified arguments
run_training "wic" 10 0 40
# Call the function with the specified arguments
run_training "wic" 5 0 20
# Call the function with the specified arguments
run_training "wic" 2 0 8
