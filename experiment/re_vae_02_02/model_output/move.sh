# !/bin/bash

mkdir -p vae_freeze_ourloss/{zeroshot,fewshot}_{hepg2,jurkat,k562,rpe1} && \
for prefix in zeroshot fewshot; do
    for cell in hepg2 jurkat k562 rpe1; do
        if [ -d "${prefix}_${cell}/vae_freeze_ourloss" ]; then
            mv "${prefix}_${cell}/vae_freeze_ourloss"/* "vae_freeze_ourloss/${prefix}_${cell}/" 2>/dev/null
        fi
    done
done