#!/bin/bash
STAGE=train
PROJECT_PATH=/work/home/cryoem666/czx/project/state_training_debug
EXPNAME=ba_loss_01_13
CELL=jurkat
BASHPATH=/work/home/cryoem666/czx/project/state_training_debug/experiment/${EXPNAME}/scripts/slurm_${STAGE}_single.sh

# 提交函数
submit_job() {
    local config_name=$1
    local s_w=$2
    local m_w=$3
    local c_w=$4

    # 自动创建该实验的日志目录
    local log_dir="${PROJECT_PATH}/experiment/${EXPNAME}/scripts/output/${CELL}/${STAGE}/${config_name}"
    mkdir -p "${log_dir}"
    
    # 时间戳用于区分多次运行
    local ts=$(date +%m_%d_%H_%M)
    local out_log="${log_dir}/${ts}_slurm.out"
    local err_log="${log_dir}/${ts}_slurm.err"

    echo "Submitting: ${config_name}..."

    # 使用 sbatch 提交，并通过 --export 传递变量，通过 --output 指定 Slurm 日志
    sbatch \
        --job-name="state_${config_name}" \
        --partition="8gpu" \
        --output="${out_log}" \
        --error="${err_log}" \
        --export=ALL,PROJECT_PATH="${PROJECT_PATH}",EXPNAME="${EXPNAME}",CELL="${CELL}",CONFIG_NAME="${config_name}",S_W="${s_w}",M_W="${m_w}",C_W="${c_w}" \
        ${BASHPATH}
}

# --- 消融实验列表 ---

# 实验 1: All 3 loss
submit_job "0.7_0.3" 1 0.7 0.3

# 实验 2: SD 1 loss
submit_job "0.6_0.4" 1 0.6 0.4

# 实验 3: Mean 2 loss
submit_job "0.5_0.5" 1 0.5 0.5

# 实验 2: SD 1 loss
submit_job "0.4_0.6" 1 0.4 0.6

# 实验 3: Mean 2 loss
submit_job "0.3_0.7" 1 0.3 0.7

echo "All jobs submitted! Use 'squeue -u $(whoami)' to check status."
