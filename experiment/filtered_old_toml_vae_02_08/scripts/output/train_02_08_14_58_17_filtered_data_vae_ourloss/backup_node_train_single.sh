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

python -m state tx train \
    "data.kwargs.toml_config_path='${PROJECT_PATH}/config/filtered_old_dataset_old_toml/${CELL}.toml'" \
    "data.kwargs.num_workers=16" \
    "data.kwargs.output_space=gene" \
    "data.kwargs.batch_col=gem_group" \
    "data.kwargs.pert_col=gene" \
    "data.kwargs.cell_type_key=cell_line" \
    "data.kwargs.control_pert=non-targeting" \
    "data.kwargs.embed_key=X_hvg" \
    "training.devices=1" \
    "training.max_steps=80000" \
    "training.ckpt_every_n_steps=2000" \
    "training.batch_size=64" \
    "model=vae_transition" \
    "model.kwargs.cell_set_len=64" \
    "model.kwargs.hidden_dim=128" \
    "model.kwargs.batch_encoder=True" \
    "model.kwargs.loss_kwargs.main_loss_kwargs.type=${LOSS_TYPE}" \
    "model.kwargs.loss_kwargs.vae_loss_kwargs.type=${LOSS_TYPE}" \
    "model.kwargs.vae_kwargs.model_path='${PROJECT_PATH}/model_weight/old_dataset_old_toml/model_seed=0_step=199999.pt'" \
    "wandb.tags=['replogle_run_${CELL}']" \
    "use_wandb=False" \
    "output_dir='${PROJECT_PATH}/experiment/${EXP_NAME}/model_output/${CONFIG_NAME}'" \
    "name='${CELL}'"
