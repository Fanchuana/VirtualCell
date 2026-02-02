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

python -m state tx predict \
    --output-dir="${PROJECT_PATH}/experiment/${EXP_NAME}/model_output/${CELL}/${CONFIG_NAME}" \
    --checkpoint "best.ckpt"
    