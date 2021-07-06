# The name of experiment
GPUS=$1
TEACHER_NAME=$2
NAME=$3

# Create dirs and make backup
teacher_checkpoint=snap/vlm/$TEACHER_NAME
output=snap/vlm/$NAME
mkdir -p $output
mkdir -p $output/src
cp -r vlm $output/src/
cp scripts/run_glue_at_epoch.bash $output/run_glue_at_epoch.bash 
cp $0 $output/run.bash

export TRAIN_FILE=data/wiki103-cased/wiki.train.raw
export TEST_FILE=data/wiki103-cased/wiki.valid.raw

# Pre-training
CUDA_VISIBLE_DEVICES=$1 python vlm/run_vlm_distributed.py \
    --output_dir=$output \
	--overwrite_output_dir \
	--config_name=vlm/configs/bert-6L-512H.json \
	--tokenizer_name=bert-base-uncased \
    --model_type=bert \
	--block_size=126 \
	--per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
	--gradient_accumulation_steps=2 \
	--num_train_epochs=40 \
	--learning_rate=2e-4 \
	--weight_decay=0.01 \
	--warmup_steps=10000 \
    --mlm_probability 0.15 \
    --voken_ratio 1.0 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --col_data \
    --split_sent \
    --do_kd1_objective \
    --fp16 \
    --fp16_opt_level O2 \
    --teacher_dir $teacher_checkpoint \
    --mlm | tee $output/log.log
    
# --do_kd1_objective mmd KD loss
# --do_kd2_objective CRD KD loss