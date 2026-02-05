#!/bin/bash

# --- 定义并导出全局变量 (子脚本将自动继承) ---
export PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
export EXP_NAME=STATE_02_02
export STAGE=test
export CONFIG_NAME="ST-HVG-Replogle"
export TIMESTAMP=$(date +%m_%d_%H_%M_%S)

TASK="fewshot"

# --- 脚本备份 ---
RECORD_DIR="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/output/${STAGE}_${TIMESTAMP}_${CONFIG_NAME}"

BASH_PATH="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh"

# task | test_dataset | train/test | config_name | gpu_id | DE_loss_type | direction_loss_type | cons_loss
bash ${BASH_PATH} ${TASK} jurkat ${STAGE} ${CONFIG_NAME} 0 &

bash ${BASH_PATH} ${TASK} hepg2 ${STAGE} ${CONFIG_NAME} 1 &

bash ${BASH_PATH} ${TASK} k562 ${STAGE} ${CONFIG_NAME} 2 &

bash ${BASH_PATH} ${TASK} rpe1 ${STAGE} ${CONFIG_NAME} 3 &

# bash ${BASH_PATH} zeroshot jurkat ${STAGE} ${CONFIG_NAME} 0 &

# bash ${BASH_PATH} zeroshot hepg2 ${STAGE} ${CONFIG_NAME} 1 &

# bash ${BASH_PATH} zeroshot k562 ${STAGE} ${CONFIG_NAME} 2 &

# bash ${BASH_PATH} zeroshot rpe1 ${STAGE} ${CONFIG_NAME} 3 &


mkdir -p "${RECORD_DIR}"
cp "$0" "${RECORD_DIR}/backup_$(basename "$0")"
cp "${BASH_PATH}" "${RECORD_DIR}/backup_$(basename "${BASH_PATH}")"
echo "Scripts backed up to: ${RECORD_DIR}"

echo "TIMESTAMP: ${TIMESTAMP}"
echo "All experiments submitted. Check info.log in respective folders."
wait
