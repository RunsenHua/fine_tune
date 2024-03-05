PRE_SEQ_LEN=8
LR=1e-2

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train \
    --train_file AdvertiseGen_Simple/train.json \
    --validation_file AdvertiseGen_Simple/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path chatglm-6b \
    --output_dir output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --logging_steps 10 \
    --save_steps 6 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    --num_train_epochs 1
