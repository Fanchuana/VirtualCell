#!/bin/bash
STAGE=train
# STAGE=test
EXP_NAME=DE_01_30
BASH_PATH=/work/home/cryoem666/czx/project/state_training_debug/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh
# TEST_DATASET=jurkat
# jurkat
# hepg2
# k562

# 实验 1-4: 启用DE_decoder_损失系数1_启用direction_decoder_损失系数1_数据集的显著性阈值0.05
# test_dataset | train/test | config_name | gpu_id | DE_loss_type | direction_loss_type | cons_loss
# bash ${BASH_PATH} jurkat ${STAGE} DE_ASYM_cons_true 0 ASYM smooth_l1 True &

# bash ${BASH_PATH} hepg2 ${STAGE} DE_ASYM_cons_true 1 ASYM smooth_l1 True &

# bash ${BASH_PATH} k562 ${STAGE} DE_ASYM_cons_true 2 ASYM smooth_l1 True &

# bash ${BASH_PATH} rpe1 ${STAGE} DE_ASYM_cons_true 3 ASYM smooth_l1 True &

bash ${BASH_PATH} jurkat ${STAGE} DE_ASYM_cons_false 0 ASYM smooth_l1 False &

bash ${BASH_PATH} hepg2 ${STAGE} DE_ASYM_cons_false 1 ASYM smooth_l1 False &

bash ${BASH_PATH} k562 ${STAGE} DE_ASYM_cons_false 2 ASYM smooth_l1 False &

bash ${BASH_PATH} rpe1 ${STAGE} DE_ASYM_cons_false 3 ASYM smooth_l1 False &

echo "All experiments submitted. Check info.log in respective folders."
wait
