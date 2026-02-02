#!/bin/bash
EXPNAME=sd_loss
CELL=k562
# EXPNAME=sd_loss_12_30
# CELL=hepg2
# CELL=jurkat

echo "Running experiment EXPNAME=${EXPNAME}, CELL=${CELL}"

source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh

cd /work/home/cryoem666/czx/project/state/model
conda activate state
conda env list

# 禁用 IB 驱动程序, 如果四卡可训练但单卡爆炸, 可开启该环境变量设置
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=2

export PYTHONPATH=src

python -m state tx predict \
    --output-dir "$HOME/czx/project/state/experiment/${EXPNAME}/model_output/${CELL}" \
    --checkpoint "step=44000.ckpt"