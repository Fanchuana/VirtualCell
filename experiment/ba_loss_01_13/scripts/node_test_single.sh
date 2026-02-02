#!/bin/bash
# 路径：${PROJECT_PATH}/run_single.sh

# 接收参数
CONFIG_NAME=${1:-"0.7_0.3"}
GPU_ID=${2:-0}
S_W=${3:-1}  # sinkhorn_weight
M_W=${4:-1}  # mean_weight
C_W=${5:-1}  # cov_weight
CELL=${6:-"jurkat"} # 测试集
STAGE=${7:-"train"} # train or test

PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
EXPNAME=ba_loss_01_13

# 自动创建日志目录
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
echo "Config Name:  $CONFIG_NAME"
echo "Stage:        $STAGE"
echo "GPU ID:       $GPU_ID"
echo "Weights:      S=$S_W, M=$M_W, C=$C_W"
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
    --output-dir="${PROJECT_PATH}/experiment/${EXPNAME}/model_output/${CELL}/${CONFIG_NAME}" \
    --checkpoint "best.ckpt"
    