#!/bin/bash
# STAGE=train
STAGE=test
EXP_NAME=vae_01_19
BASH_PATH=/work/home/cryoem666/czx/project/state_training_debug/experiment/${EXP_NAME}/scripts/node_${STAGE}_single.sh
TEST_DATASET=jurkat
# jurkat
# hepg2
# k562

# 实验 1-4: 不冻结vae_启用次要损失_主损失ourloss_次要损失MMD_次要损失系数1
# test_dataset | train/test | config_name | gpu_id | main_loss | use_vae_latent_loss | 
# vae_latent_loss | vae_latent_loss_weight | vae_freeze |  
bash ${BASH_PATH} jurkat ${STAGE} our_MMD_1 0 ba True energy 1 False &

bash ${BASH_PATH} hepg2 ${STAGE} our_MMD_1 1 ba True energy 1 False &

bash ${BASH_PATH} k562 ${STAGE} our_MMD_1 2 ba True energy 1 False &

bash ${BASH_PATH} rpe1 ${STAGE} our_MMD_1 3 ba True energy 1 False &

# 实验 5-8: 不冻结vae_启用次要损失_主损失ourloss_次要损失ourloss_次要损失系数1
bash ${BASH_PATH} jurkat ${STAGE} our_our_1 4 ba True ba 1 False &

bash ${BASH_PATH} hepg2 ${STAGE} our_our_1 5 ba True ba 1 False &

bash ${BASH_PATH} k562 ${STAGE} our_our_1 6 ba True ba 1 False &

bash ${BASH_PATH} rpe1 ${STAGE} our_our_1 7 ba True ba 1 False &

# # 实验 9-12: 不冻结vae_启用次要损失_主损失MMD_次要损失MMD_次要损失系数13
# bash ${BASH_PATH} jurkat ${STAGE} MMD_MMD_13 0 energy True energy 13 False &

# bash ${BASH_PATH} hepg2 ${STAGE} MMD_MMD_13 1 energy True energy 13 False &

# bash ${BASH_PATH} k562 ${STAGE} MMD_MMD_13 2 energy True energy 13 False &

# bash ${BASH_PATH} rpe1 ${STAGE} MMD_MMD_13 3 energy True energy 13 False &

echo "All experiments submitted. Check info.log in respective folders."
wait
