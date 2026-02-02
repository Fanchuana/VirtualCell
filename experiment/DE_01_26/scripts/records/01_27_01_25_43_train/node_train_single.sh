#!/bin/bash
# 路径：${PROJECT_PATH}/run_single.sh

# 接收参数
CELL=${1:-"jurkat"} # 测试集
STAGE=${2:-"train"} # train or test
CONFIG_NAME=${3:-"vaeloss1_freeze1"}
GPU_ID=${4:-0}
DE_DECODER=${5:-True}
DE_LOSS_WEIGHT=${6:-1}
DIR_DECODER=${7:-True}
DIR_LOSS_WEIGHT=${8:-1}
P_THRESHOLD=${9:-0.5}

PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
EXPNAME=DE_01_26

# 创建日志目录
LOG_DIR="${PROJECT_PATH}/experiment/${EXPNAME}/scripts/output/${CELL}/${STAGE}/${CONFIG_NAME}"
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
echo "DE_DECODER:               $DE_DECODER"
echo "DE_LOSS_WEIGHT:           $DE_LOSS_WEIGHT"
echo "DIR_DECODER:              $DIR_DECODER"
echo "DIR_LOSS_WEIGHT:          $DIR_LOSS_WEIGHT"
echo "P_THRESHOLD:              $P_THRESHOLD"
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
    model.kwargs.DE_kwargs.predict_DE=${DE_DECODER} \
    model.kwargs.DE_kwargs.predict_direction=${DIR_DECODER} \
    model.kwargs.DE_kwargs.p_val_threshold=${P_THRESHOLD} \
    model.kwargs.loss_kwargs.DE_loss_weight=${DE_LOSS_WEIGHT} \
    model.kwargs.loss_kwargs.direction_loss_weight=${DIR_LOSS_WEIGHT} \
    wandb.tags="[replogle_run_${CELL}]" \
    use_wandb=False \
    output_dir="${PROJECT_PATH}/experiment/${EXPNAME}/model_output/${CELL}" \
    name="${CONFIG_NAME}"