#!/bin/bash
STAGE=train
# STAGE=test
EXP_NAME=ba_loss_01_13
BASH_PATH=/work/home/cryoem666/czx/project/state_training_debug/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh
TEST_DATASET=jurkat
# jurkat
# hepg2
# k562

# 实验 1: -> GPU 0
# name | gpu_id | sd_weight | mean_weight | cov_weight | test_dataset
bash ${BASH_PATH} 0.7_0.3 0 1 0.7 0.3 ${TEST_DATASET} ${STAGE} &

# 实验 2: -> GPU 1
bash ${BASH_PATH} 0.6_0.4 1 1 0.6 0.4 ${TEST_DATASET} ${STAGE} &

# 实验 3: -> GPU 2
bash ${BASH_PATH} 0.5_0.5 2 1 0.5 0.5 ${TEST_DATASET} ${STAGE} &

# 实验 4: -> GPU 3
bash ${BASH_PATH} 0.4_0.6 3 1 0.4 0.6 ${TEST_DATASET} ${STAGE} &

# 实验 5: -> GPU 3
# bash ${BASH_PATH} 0.3_0.7 3 1 0.3 0.7 ${TEST_DATASET} ${STAGE} &

echo "All experiments submitted. Check info.log in respective folders."
wait
