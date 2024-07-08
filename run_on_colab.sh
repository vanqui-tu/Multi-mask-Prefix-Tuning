%%shell
MODEL=t5-base
MAX_LENGTH=256
MAX_STEPS=10000
PREFIX_LENGTH=20
for TASK_NAME in rte; do
    for lr in 3e-1; do
        CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dept \
            --peft_type prompt-routing \
            --learning_rate ${lr} \
            --num_virtual_tokens 20 \
            --num_virtual_tokens_full 80 \
            --prefix_length ${PREFIX_LENGTH} \
            --task_name ${TASK_NAME} \
            --dataset_config_name en \
            --model_name_or_path ${MODEL} \
            --do_train \
            --do_eval \
            --do_predict \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --max_seq_length ${MAX_LENGTH} \
            --save_strategy steps \
            --evaluation_strategy steps \
            --max_steps ${MAX_STEPS} \
            --eval_steps 1000 \
            --save_steps 1000 \
            --warmup_steps 500 \
            --weight_decay 1e-5 \
            --load_best_model_at_end \
            --save_total_limit 1 \
            --perturb_router True \
            --topk 1 \
            --seed 11 \
            --output_dir saved_${MODEL}/${TASK_NAME}_lr${lr}_pl${PREFIX_LENGTH}_r${R}_st${MAX_STEPS};
    done;
done