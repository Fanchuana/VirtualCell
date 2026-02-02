#!/bin/bash

PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
EXPNAME=ba_loss_01_09
CONFIG_NAME=all_3_loss
CELL=jurkat

# 定义日志路径
INFO_LOG="${PROJECT_PATH}/experiment/${EXPNAME}/scripts/output/${CONFIG_NAME}/info.log"
ERR_LOG="${PROJECT_PATH}/experiment/${EXPNAME}/scripts/output/${CONFIG_NAME}/error.log"

# 1. 重定向标准输出 (stdout) 到 info.log，同时在屏幕显示
exec > >(tee -a "${INFO_LOG}")
# 2. 重定向标准错误 (stderr) 到 error.log，同时在屏幕显示
exec 2> >(tee -a "${ERR_LOG}")

echo "Running experiment EXPNAME=${EXPNAME}, CELL=${CELL}"

source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh

cd ${PROJECT_PATH}/model
conda activate state_bk

# 禁用 IB 驱动程序, 如果四卡可训练但单卡爆炸, 可开启该环境变量设置
# export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=src

python -m state tx train \
    data.kwargs.toml_config_path=${PROJECT_PATH}/config/${CELL}.toml \
    data.kwargs.num_workers=16 \
    data.kwargs.output_space="gene" \
    data.kwargs.batch_col="gem_group" \
    data.kwargs.pert_col="gene" \
    data.kwargs.cell_type_key="cell_line" \
    data.kwargs.control_pert="non-targeting" \
    training.devices=1 \
    training.max_steps=80000 \
    training.ckpt_every_n_steps=2000 \
    training.batch_size=64 \
    training.lr=1e-3 \
    model.kwargs.sinkhorn_weight: 1 \
    model.kwargs.mean_weight: 1 \
    model.kwargs.cov_weight: 1 \
    model.kwargs.cell_set_len=64 \
    model.kwargs.hidden_dim=128 \
    model.kwargs.batch_encoder=True \
    model=our_state \
    wandb.tags="[replogle_run_${CELL}]" \
    use_wandb=False \
    output_dir="${PROJECT_PATH}/experiment/${EXPNAME}/model_output/${CONFIG_NAME}" \
    name="${CELL}"
