#!/usr/bin/env bash

source /home/hltcoe/xli/.bashrc
source /home/hltcoe/xli/anaconda3/etc/profile.d/conda.sh

echo $CUDA_VISIBLE_DEVICES

conda activate xtts
cd /home/hltcoe/xli/ARTS/TTS

python /home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/train_gpt_xtts.py \
    --run_name GPT_XTTS_v2.0_CV_FT_1e-5_mixed_full_noaug_langemb \
    --meta_file_train "/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_filelist/cv-train-unfiltered.csv" \
    --language_cond_type 'embedding' \
    --language_cond_path "/exp/xli/ARTS/geolocation/cv_embs" \
    --language_cond_emb_dim 1024 \
    --augmentation_type 'None'