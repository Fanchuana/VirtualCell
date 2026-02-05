#!/bin/bash

# --- 定义并导出全局变量 (子脚本将自动继承) ---
export PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
export EXP_NAME=re_vae_02_02
# export STAGE=train
export STAGE=test
export CONFIG_NAME="vae_freeze_ourloss"
export TIMESTAMP=$(date +%m_%d_%H_%M_%S)

# --- 脚本备份 ---
RECORD_DIR="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/output/${STAGE}_${TIMESTAMP}_${CONFIG_NAME}"
mkdir -p "${RECORD_DIR}"

BASH_PATH="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh"

# task | test_dataset | train/test | config_name | gpu_id
bash ${BASH_PATH} fewshot jurkat ${STAGE} ${CONFIG_NAME} 0 &

bash ${BASH_PATH} fewshot hepg2 ${STAGE} ${CONFIG_NAME} 1 &

bash ${BASH_PATH} fewshot k562 ${STAGE} ${CONFIG_NAME} 2 &

bash ${BASH_PATH} fewshot rpe1 ${STAGE} ${CONFIG_NAME} 3 &

bash ${BASH_PATH} zeroshot jurkat ${STAGE} ${CONFIG_NAME} 4 &

bash ${BASH_PATH} zeroshot hepg2 ${STAGE} ${CONFIG_NAME} 5 &

bash ${BASH_PATH} zeroshot k562 ${STAGE} ${CONFIG_NAME} 6 &

bash ${BASH_PATH} zeroshot rpe1 ${STAGE} ${CONFIG_NAME} 7 &

# 备份当前执行的这两个脚本
if [[ "${STAGE}" == "train" ]]; then
    mkdir -p "${RECORD_DIR}"
    cp "$0" "${RECORD_DIR}/backup_$(basename "$0")"
    cp "${BASH_PATH}" "${RECORD_DIR}/backup_$(basename "${BASH_PATH}")"
    echo "Scripts backed up to: ${RECORD_DIR}"
fi
echo "TIMESTAMP: ${TIMESTAMP}"
echo "All experiments submitted. Check info.log in respective folders."
wait
