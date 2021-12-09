#!/bin/bash

TASK=83
MODEL=ctrl_lxmert
MODEL_CONFIG=ctrl_lxmert
TASKS_CONFIG=ctrl_trainval_tasks
PRETRAINED=checkpoints/conceptual_captions/${MODEL}/${MODEL_CONFIG}/pytorch_model_9.bin
OUTPUT_DIR=checkpoints/talk2car/${MODEL}_region_all_layer_fusion_self_attn_30epoch
LOGGING_DIR=logs/talk2car

source activate volta

cd ../../..
python train_task.py \
        --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
        --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
        --output_dir ${OUTPUT_DIR} \
        --logdir ${LOGGING_DIR} \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 30 \
  --seed 0

conda deactivate