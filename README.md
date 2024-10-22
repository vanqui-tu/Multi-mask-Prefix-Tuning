# Multi-mask Prefix Tuning

This repository contains the code for the **PACLIC 338 (2024)** paper titled **"Multi-mask Prefix Tuning: Applying Multiple Adaptive Masks on Deep Prompt Tuning"**, authored by Qui Tu, Trung Nguyen, Long Nguyen, and Dien Dinh.

## Overview

The Multi-mask Prefix Tuning method leverages multiple adaptive masks to enhance deep prompt tuning. The model is applied to the SuperGLUE datasets and is based on the HuggingFace PEFT library.

## Setup Instructions

### Prerequisites

To get started, you'll need to install the necessary libraries and download the SuperGLUE datasets. You can do this by running the following command:

```
bash start.sh
```

### Training

Training can be initiated by executing the train.py script with specified arguments. Here is how to set up the training configuration:

```
python scripts/train.py --lr 0.005 \
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
                        --dataset_name rte_superglue \
                        --num_virtual_tokens 1 \
                        --num_shared_virtual_tokens 10 \
                        --num_virtual_tokens_full 4 \
                        --perturb_router True \
                        --topk 1 \
                        --shareType Only \
                        --apply_adaptive_mask True \
                        --apply_adaptive_subset_mask True
```
Alternatively, you can run the example script:

```
bash run.sh
```
## Acknowledgments
This implementation is largely based on the HuggingFace PEFT library (https://github.com/huggingface/peft) and SMoP (https://github.com/jyjohnchoi/SMoP).


