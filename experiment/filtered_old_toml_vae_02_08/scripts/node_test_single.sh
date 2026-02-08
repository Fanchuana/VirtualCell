#!/bin/bash

# --- 1. 接收命令行参数 ---
CELL=${1:-"jurkat"}
STAGE=${2:-"train"}
CONFIG_NAME=${3:-"filtered_data_vae_ourloss"}
GPU_ID=${4:-0}
LOSS_TYPE=${5:-ba}

# --- 2. 变量继承与初始化 ---
# 如果父脚本 export 了这些变量，这里会直接引用；否则使用冒号后的默认值
PROJECT_PATH=${PROJECT_PATH:-"/work/home/cryoem666/czx/project/state_training_debug"}
EXP_NAME=${EXP_NAME:-"re_vae_02_02"}
TIMESTAMP=${TIMESTAMP:-$(date +%m_%d_%H_%M_%S)}

# --- 3. 日志目录配置 ---
LOG_DIR="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/output/${STAGE}_${TIMESTAMP}_${CONFIG_NAME}/${CELL}"
mkdir -p "${LOG_DIR}"

INFO_LOG="${LOG_DIR}/info.log"
ERR_LOG="${LOG_DIR}/error.log"

# --- 4. 重定向输出 ---
exec > >(tee -a "${INFO_LOG}")
exec 2> >(tee -a "${ERR_LOG}")

echo "========== Experiment Configuration =========="
echo "Config Name:              $CONFIG_NAME"
echo "Stage:                    $STAGE"
echo "GPU ID:                   $GPU_ID"
echo "Test dataset: CELL=$CELL"
echo "Loss type:    LOSS_TYPE=${LOSS_TYPE}"
echo "Stdout path:  INFO_LOG=$INFO_LOG"
echo "Stderr path:  ERR_LOG=$ERR_LOG"
echo "=============================================="

source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
cd ${PROJECT_PATH}/model
conda activate state

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH=src

python -m state tx predict \
    --output-dir="${PROJECT_PATH}/experiment/${EXP_NAME}/model_output/${CONFIG_NAME}/${CELL}" \
    --checkpoint="best.ckpt" \
    