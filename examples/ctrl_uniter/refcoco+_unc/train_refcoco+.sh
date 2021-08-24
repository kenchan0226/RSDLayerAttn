#!/bin/bash
#SBATCH --job-name=g4_volta
#SBATCH --output=/home/iotsc_g4/gmx/volta/examples/ctrl_uniter/refcoco+_unc/output.%j
#SBATCH --gres=gpu:1
#SBATCH --mem                   120G
#SBATCH --cpus-per-task     4




TASK=10
MODEL=ctrl_uniter
MODEL_CONFIG=ctrl_uniter_base
TASKS_CONFIG=ctrl_trainval_tasks
PRETRAINED=checkpoints/conceptual_captions/${MODEL}/${MODEL_CONFIG}/pytorch_model_9.bin
OUTPUT_DIR=checkpoints/refcoco+_unc/${MODEL}
LOGGING_DIR=logs/refcoco+_unc
source activate volta
python --version

cd /home/iotsc_g4/gmx/volta

python train_task.py \
	--config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
	--output_dir ${OUTPUT_DIR} \
	--logdir ${LOGGING_DIR} \
#	--resume_file ${OUTPUT_DIR}/refcoco+_${MODEL_CONFIG}/pytorch_ckpt_latest.tar
conda deactivate
