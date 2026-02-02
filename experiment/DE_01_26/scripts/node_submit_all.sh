#!/bin/bash
STAGE=train
# STAGE=test
EXP_NAME=DE_01_26
BASH_PATH=/work/home/cryoem666/czx/project/state_training_debug/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh
# TEST_DATASET=jurkat
# jurkat
# hepg2
# k562

# 实验 1-4: 启用DE_decoder_损失系数1_启用direction_decoder_损失系数1_数据集的显著性阈值0.05
# test_dataset | train/test | config_name | gpu_id | DE_decoder | DE_loss_weight | direction_decoder | direction_loss_weight | direction_loss_type | p_val_threshold 
bash ${BASH_PATH} jurkat ${STAGE} DE_1_dir_1_p_05 0 True 1 True 1 MSE 0.05 &

bash ${BASH_PATH} hepg2 ${STAGE} DE_1_dir_1_p_05 1 True 1 True 1 MSE 0.05 &

bash ${BASH_PATH} k562 ${STAGE} DE_1_dir_1_p_05 2 True 1 True 1 MSE 0.05 &

bash ${BASH_PATH} rpe1 ${STAGE} DE_1_dir_1_p_05 3 True 1 True 1 MSE 0.05 &

echo "All experiments submitted. Check info.log in respective folders."
wait
