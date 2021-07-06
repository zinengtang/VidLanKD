# The name of experiment
GPUS=$1

NAME=$2

# Create dirs and make backup
output=snap/vlm/$NAME
mkdir -p $output/src
cp -r vlm $output/src/
cp scripts/run_glue_at_epoch.bash $output/run_glue_at_epoch.bash 
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$1 python3 vteacher/run_vlm_distributed.py \
    --output_dir=$output \
	--overwrite_output_dir \
	--config_name=vlm/configs/bert-6L-512H.json \
	--tokenizer_name=bert-base-uncased \
    --model_type=bert \
	--block_size=126 \
	--per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
	--gradient_accumulation_steps=1 \
	--num_train_epochs=200 \
	--learning_rate=5e-5 \
	--weight_decay=0.01 \
	--warmup_steps=0 \
    --mlm_probability 0.15 \
    --mlm_ratio 1.0 \
    --do_train \
    --do_eval \
    --col_data \
    --split_sent \
    --voken_labels all \
    --dim 64 \
    --voken_hinge_loss \
    --fp16 \
    --fp16_opt_level O2 \
    --mlm ${@:3} | tee $output/log.log \


