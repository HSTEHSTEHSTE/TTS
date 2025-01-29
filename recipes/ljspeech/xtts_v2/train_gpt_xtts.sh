#!/usr/bin/env bash

source /home/hltcoe/xli/.bashrc
source /home/hltcoe/xli/anaconda3/etc/profile.d/conda.sh

# export CUDA_VISIBLE_DEVICES=$(free-gpu)
echo $CUDA_VISIBLE_DEVICES

conda activate xtts
cd /home/hltcoe/xli/ARTS/TTS

python /home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/train_gpt_xtts.py