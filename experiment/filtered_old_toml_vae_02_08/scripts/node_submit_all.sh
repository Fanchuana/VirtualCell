#!/bin/bash

# --- 定义并导出全局变量 (子脚本将自动继承) ---
export PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
export EXP_NAME=filtered_old_toml_vae_02_08
export STAGE=train
# export STAGE=test
export CONFIG_NAME="filtered_data_vae_MMD"
export TIMESTAMP=$(date +%m_%d_%H_%M_%S)

# --- 脚本备份 ---
RECORD_DIR="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/output/${STAGE}_${TIMESTAMP}_${CONFIG_NAME}"

BASH_PATH="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh"

# test_dataset | train/test | config_name | gpu_id | main+vae_loss_type
bash ${BASH_PATH} jurkat ${STAGE} ${CONFIG_NAME} 0 energy &

bash ${BASH_PATH} hepg2 ${STAGE} ${CONFIG_NAME} 1 energy &

bash ${BASH_PATH} k562 ${STAGE} ${CONFIG_NAME} 2 energy &

bash ${BASH_PATH} rpe1 ${STAGE} ${CONFIG_NAME} 3 energy &

# 备份当前执行的这两个脚本
mkdir -p "${RECORD_DIR}"
cp "$0" "${RECORD_DIR}/backup_$(basename "$0")"
cp "${BASH_PATH}" "${RECORD_DIR}/backup_$(basename "${BASH_PATH}")"
echo "Scripts backed up to: ${RECORD_DIR}"

echo "TIMESTAMP: ${TIMESTAMP}"
echo "All experiments submitted. Check info.log in respective folders."
wait
