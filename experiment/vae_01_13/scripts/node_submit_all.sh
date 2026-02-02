#!/bin/bash
# STAGE=train
STAGE=test
EXP_NAME=vae_01_13
BASH_PATH=/work/home/cryoem666/czx/project/state_training_debug/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh
TEST_DATASET=jurkat
# jurkat
# hepg2
# k562

# 实验 1-4: 冻结vae_启用vaeloss
# test_dataset | train/test | config_name | gpu_id | main_loss | vae_freeze | use_vae_latent_loss 
bash ${BASH_PATH} jurkat ${STAGE} freeze0_vaeloss1 0 energy False True  &

bash ${BASH_PATH} hepg2 ${STAGE} freeze0_vaeloss1 1 energy False True  &

bash ${BASH_PATH} k562 ${STAGE} freeze0_vaeloss1 2 energy False True  &

bash ${BASH_PATH} rpe1 ${STAGE} freeze0_vaeloss1 3 energy False True  &

# 实验 5-8: 冻结vae_关闭vaeloss
bash ${BASH_PATH} jurkat ${STAGE} freeze0_vaeloss0 4 energy False False  &

bash ${BASH_PATH} hepg2 ${STAGE} freeze0_vaeloss0 5 energy False False  &

bash ${BASH_PATH} k562 ${STAGE} freeze0_vaeloss0 6 energy False False  &

bash ${BASH_PATH} rpe1 ${STAGE} freeze0_vaeloss0 7 energy False False  &

echo "All experiments submitted. Check info.log in respective folders."
wait
