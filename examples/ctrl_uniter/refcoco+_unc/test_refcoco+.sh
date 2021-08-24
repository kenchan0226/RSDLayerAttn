#!/bin/bash
#SBATCH --job-name=g4_volta
#SBATCH --output=/home/iotsc_g4/gmx/volta/examples/ctrl_uniter/refcoco+_unc/test_output.%j
#SBATCH --gres=gpu:1
#SBATCH --mem                   120G
#SBATCH --cpus-per-task     4
TASK=10
MODEL=ctrl_uniter
MODEL_CONFIG=ctrl_uniter_base
TASKS_CONFIG=ctrl_test_tasks
PRETRAINED=checkpoints/refcoco+_unc/${MODEL}/refcoco+_${MODEL_CONFIG}/pytorch_model_18.bin
OUTPUT_DIR=results/refcoco+_unc/${MODEL}

source activate volta

cd /home/iotsc_g4/gmx/volta
python eval_task.py \
	--config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

conda deactivate
