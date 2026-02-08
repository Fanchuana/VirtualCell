#!/bin/bash

# --- 1. 接收命令行参数 ---
TASK=${1:-"fewshot"}
CELL=${2:-"jurkat"}
STAGE=${3:-"train"}
CONFIG_NAME=${4:-"vae_ourloss_lr1e-3"}
GPU_ID=${5:-0}
LOSS_TYPE=${6:-ba}
LR=${7:-0.001}

# --- 2. 变量继承与初始化 ---
# 如果父脚本 export 了这些变量，这里会直接引用；否则使用冒号后的默认值
PROJECT_PATH=${PROJECT_PATH:-"/work/home/cryoem666/czx/project/state_training_debug"}
EXP_NAME=${EXP_NAME:-"newconfig_vae_02_06"}
TIMESTAMP=${TIMESTAMP:-$(date +%m_%d_%H_%M_%S)}

# --- 3. 日志目录配置 ---
LOG_DIR="${PROJECT_PATH}/experiment/${EXP_NAME}/scripts/output/${STAGE}_${TIMESTAMP}_${CONFIG_NAME}/${TASK}_${CELL}"
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
echo "TASK:         TASK=$TASK"
echo "Test dataset: CELL=$CELL"
echo "Loss type:    LOSS_TYPE=${LOSS_TYPE}"
echo "LR:           LR=${LR}"
echo "Stdout path:  INFO_LOG=$INFO_LOG"
echo "Stderr path:  ERR_LOG=$ERR_LOG"
echo "=============================================="

source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
cd ${PROJECT_PATH}/model
conda activate state

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH=src

python -m state tx train \
    "data.kwargs.toml_config_path='${PROJECT_PATH}/config/new_dataset/${TASK}_${CELL}.toml'" \
    "data.kwargs.num_workers=16" \
    "data.kwargs.output_space=gene" \
    "data.kwargs.batch_col=gem_group" \
    "data.kwargs.pert_col=gene" \
    "data.kwargs.cell_type_key=cell_line" \
    "data.kwargs.control_pert=non-targeting" \
    "data.kwargs.embed_key=X_hvg" \
    "training.devices=1" \
    "training.max_steps=100000" \
    "training.ckpt_every_n_steps=4000" \
    "training.batch_size=64" \
    "training.lr=${LR}" \
    "model=vae_transition" \
    "model.kwargs.cell_set_len=64" \
    "model.kwargs.hidden_dim=328" \
    "model.kwargs.batch_encoder=True" \
    "model.kwargs.transformer_backbone_kwargs.bidirectional_attention=true" \
    "model.kwargs.transformer_backbone_kwargs.intermediate_size=3072" \
    "model.kwargs.transformer_backbone_kwargs.head_dim=64" \
    "model.kwargs.loss_kwargs.main_loss_kwargs.type=${LOSS_TYPE}" \
    "model.kwargs.loss_kwargs.vae_loss_kwargs.type=${LOSS_TYPE}" \
    "model.kwargs.vae_kwargs.model_path='${PROJECT_PATH}/model_weight/new_dataset/state_${TASK}_VAE_${CELL}/model_seed=0_step=199999.pt'" \
    "wandb.tags=['replogle_run_${TASK}_${CELL}']" \
    "use_wandb=False" \
    "output_dir='${PROJECT_PATH}/experiment/${EXP_NAME}/model_output/${CONFIG_NAME}'" \
    "name='${TASK}_${CELL}'"
