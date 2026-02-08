#!/bin/bash

# --- 定义并导出全局变量 (子脚本将自动继承) ---
export PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
export EXP_NAME=newconfig_vae_02_06
# export STAGE=train
export STAGE=test
export CONFIG_NAME="vae_MMD_MMD_lr1e-3"
export TIMESTAMP=$(date +%m_%d_%H_%M_%S)

TASK="fewshot"
# TASK="zeroshot"

# --- 脚本备份 ---
RECORD_DIR="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/output/${STAGE}_${TIMESTAMP}_${CONFIG_NAME}"

BASH_PATH="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh"

# task | test_dataset | train/test | config_name | gpu_id | main+vae_loss_type
bash ${BASH_PATH} ${TASK} jurkat ${STAGE} ${CONFIG_NAME} 0 energy 0.001 &

bash ${BASH_PATH} ${TASK} hepg2 ${STAGE} ${CONFIG_NAME} 1 energy 0.001 &

bash ${BASH_PATH} ${TASK} k562 ${STAGE} ${CONFIG_NAME} 2 energy 0.001 &

bash ${BASH_PATH} ${TASK} rpe1 ${STAGE} ${CONFIG_NAME} 3 energy 0.001 &

# 备份当前执行的这两个脚本
mkdir -p "${RECORD_DIR}"
cp "$0" "${RECORD_DIR}/backup_$(basename "$0")"
cp "${BASH_PATH}" "${RECORD_DIR}/backup_$(basename "${BASH_PATH}")"
echo "Scripts backed up to: ${RECORD_DIR}"

echo "TIMESTAMP: ${TIMESTAMP}"
echo "All experiments submitted. Check info.log in respective folders."
wait
