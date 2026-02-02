#!/bin/bash

# 接收参数
CELL=${1:-"jurkat"} # 测试集
STAGE=${2:-"train"} # train or test
CONFIG_NAME=${3:-"vaeloss1_freeze1"}
GPU_ID=${4:-0}
DE_LOSS_TYPE=${5:-ASYM}
DIR_LOSS_TYPE=${6:-ASYM}
CONS_LOSS=${7:-True}

PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
EXP_NAME=DE_01_30

# 创建日志目录
LOG_DIR="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/output/${CELL}/${STAGE}/${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"

# 获取当前时间
TIMESTAMP=$(date +%m_%d_%H_%M_%S)
INFO_LOG="${LOG_DIR}/info_${TIMESTAMP}.log"
ERR_LOG="${LOG_DIR}/error_${TIMESTAMP}.log"

# 重定向输出
exec > >(tee -a "${INFO_LOG}")
exec 2> >(tee -a "${ERR_LOG}")

echo "========== Experiment Configuration =========="
echo "Config Name:              $CONFIG_NAME"
echo "Stage:                    $STAGE"
echo "GPU ID:                   $GPU_ID"
echo "DE_LOSS_TYPE:             $DE_LOSS_TYPE"
echo "DIR_LOSS_TYPE:            $DIR_LOSS_TYPE"
echo "CONS_LOSS:                $CONS_LOSS"
echo "Test dataset: CELL=$CELL"
echo "Stdout path:  INFO_LOG=$INFO_LOG"
echo "Stderr path:  ERR_LOG=$ERR_LOG"
echo "=============================================="

source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
cd ${PROJECT_PATH}/model
conda activate state_bk

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH=src

python -m state tx train \
    data.kwargs.toml_config_path=${PROJECT_PATH}/config/${CELL}.toml \
    data.kwargs.num_workers=16 \
    data.kwargs.output_space="gene" \
    data.kwargs.batch_col="gem_group" \
    data.kwargs.pert_col="gene" \
    data.kwargs.cell_type_key="cell_line" \
    data.kwargs.control_pert="non-targeting" \
    data.kwargs.embed_key="X_hvg" \
    training.devices=1 \
    training.max_steps=80000 \
    training.ckpt_every_n_steps=2000 \
    training.batch_size=64 \
    training.lr=1e-3 \
    model=vae_transition \
    model.kwargs.cell_set_len=64 \
    model.kwargs.hidden_dim=128 \
    model.kwargs.batch_encoder=True \
    model.kwargs.loss_kwargs.main_loss_kwargs.use_main_loss=False \
    model.kwargs.loss_kwargs.vae_loss_kwargs.use_vae_loss=False \
    model.kwargs.loss_kwargs.DE_loss_kwargs.type=${DE_LOSS_TYPE} \
    model.kwargs.loss_kwargs.direction_loss_kwargs.type=${DIR_LOSS_TYPE} \
    model.kwargs.loss_kwargs.cons_loss_kwargs.use_cons_loss=${CONS_LOSS} \
    wandb.tags="[replogle_run_${CELL}]" \
    use_wandb=False \
    output_dir="${PROJECT_PATH}/experiment/${EXP_NAME}/model_output/${CELL}" \
    name="${CONFIG_NAME}"