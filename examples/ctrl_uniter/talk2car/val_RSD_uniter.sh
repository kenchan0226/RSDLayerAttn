#!/bin/bash

TASK=68
MODEL=ctrl_uniter_region_all_layer_fusion_self_attn
MODEL_CONFIG=ctrl_uniter_base
TASKS_CONFIG=ctrl_test_tasks
PRETRAINED=checkpoints/talk2car/${MODEL}/talk2car_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=results/talk2car/${MODEL}

conda activate volta

cd ../../..
python3 eval_task.py \
        --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
        --output_dir ${OUTPUT_DIR} --split val

conda deactivate