#!/bin/bash
#SBATCH --job-name=state_ba_loss       # 基础任务名
#SBATCH --ntasks=1                      # 总任务数量
#SBATCH --cpus-per-task=16              # CPU核数
#SBATCH --gres=gpu:1                    # 申请1张卡
#SBATCH --mem=256G                      # 内存
#SBATCH --time=10-00:00:00               # 运行10天


# 加载环境
source /work/home/cryoem666/miniconda3/etc/profile.d/conda.sh
cd ${PROJECT_PATH}/model
conda activate state_bk

export PYTHONPATH=src

# 执行训练
# 这里的变量（CONFIG_NAME, S_W 等）由 sbatch --export 传入
echo "Job started on $(date)"
echo "Running Config: ${CONFIG_NAME} | Weights: S=${S_W}, M=${M_W}, C=${C_W}"

python -m state tx train \
    data.kwargs.toml_config_path=${PROJECT_PATH}/config/${CELL}.toml \
    data.kwargs.num_workers=8 \
    data.kwargs.output_space="gene" \
    data.kwargs.batch_col="gem_group" \
    data.kwargs.pert_col="gene" \
    data.kwargs.cell_type_key="cell_line" \
    data.kwargs.control_pert="non-targeting" \
    training.devices=1 \
    training.max_steps=80000 \
    training.ckpt_every_n_steps=2000 \
    training.batch_size=64 \
    training.lr=1e-3 \
    model.kwargs.sinkhorn_weight=${S_W} \
    model.kwargs.mean_weight=${M_W} \
    model.kwargs.cov_weight=${C_W} \
    model.kwargs.cell_set_len=64 \
    model.kwargs.hidden_dim=128 \
    model.kwargs.batch_encoder=True \
    model=our_state \
    wandb.tags="[replogle_run_${CELL}]" \
    use_wandb=False \
    output_dir="${PROJECT_PATH}/experiment/${EXPNAME}/model_output/${CELL}" \
    name="${CONFIG_NAME}"

scontrol show job $SLURM_JOB_ID
echo "Job finished on $(date)"
