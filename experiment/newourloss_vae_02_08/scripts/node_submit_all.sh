#!/bin/bash

# --- 定义并导出全局变量 (子脚本将自动继承) ---
export PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
export EXP_NAME=newourloss_vae_02_08
export STAGE=train
# export STAGE=test
export CONFIG_NAME="vae_genemean_sd0_cov0_cos1_var0"
export TIMESTAMP=$(date +%m_%d_%H_%M_%S)

TASK0="fewshot"
TASK1="zeroshot"

# --- 脚本备份 ---
RECORD_DIR="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/output/${STAGE}_${TIMESTAMP}_${CONFIG_NAME}"

BASH_PATH="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh"

# task | test_dataset | train/test | config_name | gpu_id | gene_mean | sd_weight | cov_weight | cos_weight | var_weight
bash ${BASH_PATH} ${TASK0} jurkat ${STAGE} ${CONFIG_NAME} 0 True 0 0 1 0 &

bash ${BASH_PATH} ${TASK0} k562 ${STAGE} ${CONFIG_NAME} 1 True 0 0 1 0 &

bash ${BASH_PATH} ${TASK1} jurkat ${STAGE} ${CONFIG_NAME} 2 True 0 0 1 0 &

bash ${BASH_PATH} ${TASK1} k562 ${STAGE} ${CONFIG_NAME} 3 True 0 0 1 0 &

# 备份当前执行的这两个脚本
mkdir -p "${RECORD_DIR}"
cp "$0" "${RECORD_DIR}/backup_$(basename "$0")"
cp "${BASH_PATH}" "${RECORD_DIR}/backup_$(basename "${BASH_PATH}")"
echo "Scripts backed up to: ${RECORD_DIR}"

echo "TIMESTAMP: ${TIMESTAMP}"
echo "All experiments submitted. Check info.log in respective folders."
wait
