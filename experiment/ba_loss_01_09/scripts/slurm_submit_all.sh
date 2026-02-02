#!/bin/bash
STAGE=train
PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
EXPNAME=ba_loss_01_09
CELL=jurkat
BASHPATH=/work/home/cryoem666/czx/project/state_training_debug/experiment/ba_loss_01_09/scripts/slurm_${STAGE}_single.sh

# 定义提交函数，减少重复代码
submit_job() {
    local config_name=$1
    local s_w=$2
    local m_w=$3
    local c_w=$4

    # 自动创建该实验的日志目录
    local log_dir="${PROJECT_PATH}/experiment/${EXPNAME}/scripts/output/${config_name}"
    mkdir -p "${log_dir}"
    
    # 时间戳用于区分多次运行
    local ts=$(date +%m_%d_%H_%M)
    local out_log="${log_dir}/${ts}_slurm.out"
    local err_log="${log_dir}/${ts}_slurm.err"

    echo "Submitting: ${config_name}..."

    # 使用 sbatch 提交，并通过 --export 传递变量，通过 --output 指定 Slurm 日志
    sbatch \
        --job-name="state_${config_name}" \
        --output="${out_log}" \
        --error="${err_log}" \
        --export=ALL,PROJECT_PATH="${PROJECT_PATH}",EXPNAME="${EXPNAME}",CELL="${CELL}",CONFIG_NAME="${config_name}",S_W="${s_w}",M_W="${m_w}",C_W="${c_w}" \
        ${BASHPATH}
}

# --- 消融实验列表 ---

# 实验 1: All 3 loss
# submit_job "all_3_loss" 1 1 1

# 实验 2: SD 1 loss
submit_job "sd_1_loss" 1 0 0

# 实验 3: Mean 2 loss
submit_job "mean_2_loss" 1 1 0

# 实验 4: Cov 2 loss
submit_job "cov_2_loss" 1 0 1

echo "All jobs submitted! Use 'squeue -u $(whoami)' to check status."