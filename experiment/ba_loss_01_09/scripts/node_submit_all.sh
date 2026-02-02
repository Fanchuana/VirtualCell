#!/bin/bash
# STAGE=train
STAGE=test
BASH_PATH=/work/home/cryoem666/czx/project/state_training_debug/experiment/ba_loss_01_09/scripts/node_${STAGE}_single.sh
TEST_DATASET=rpe1
# jurkat
# hepg2
# k562

# 实验 1: all_3_loss -> GPU 0
# name | gpu_id | sd_weight | mean_weight | cov_weight | test_dataset
bash ${BASH_PATH} all_3_loss 0 1 1 1 ${TEST_DATASET} ${STAGE} &

# 实验 2: sd_1_loss -> GPU 1
bash ${BASH_PATH} sd_1_loss 1 1 0 0 ${TEST_DATASET} ${STAGE} &

# 实验 3: mean_2_loss -> GPU 2
bash ${BASH_PATH} mean_2_loss 2 1 1 0 ${TEST_DATASET} ${STAGE} &

# 实验 4: mean_2_loss -> GPU 3
bash ${BASH_PATH} cov_2_loss 3 1 0 1 ${TEST_DATASET} ${STAGE} &

echo "All experiments submitted. Check info.log in respective folders."
wait
