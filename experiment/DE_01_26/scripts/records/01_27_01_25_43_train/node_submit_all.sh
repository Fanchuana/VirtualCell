#!/bin/bash
# STAGE=train
STAGE=test
EXP_NAME=DE_01_26
BASH_PATH=/work/home/cryoem666/czx/project/state_training_debug/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh
TEST_DATASET=jurkat
# jurkat
# hepg2
# k562

# 实验 1-4: 启用DE_decoder_损失系数1_启用direction_decoder_损失系数1_数据集的显著性阈值0.5
# test_dataset | train/test | config_name | gpu_id | DE_decoder | DE_loss_weight | direction_decoder | direction_loss_weight | p_val_threshold
# vae_latent_loss | vae_latent_loss_weight | vae_freeze |  
bash ${BASH_PATH} jurkat ${STAGE} DE_1_dir_1_p_5 0 True 1 True 1 0.5 &

bash ${BASH_PATH} hepg2 ${STAGE} DE_1_dir_1_p_5 1 True 1 True 1 0.5 &

bash ${BASH_PATH} k562 ${STAGE} DE_1_dir_1_p_5 2 True 1 True 1 0.5 &

bash ${BASH_PATH} rpe1 ${STAGE} DE_1_dir_1_p_5 3 True 1 True 1 0.5 &

echo "All experiments submitted. Check info.log in respective folders."
wait
