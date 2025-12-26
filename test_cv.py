from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch, torchaudio
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

config = XttsConfig()
config.load_json("/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
# checkpoint_dir = '/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/GPT_XTTS_v2.0_CV_FT_1e-5_base_mono-May-23-2025_07+22PM-e355c13'
checkpoint_dir = '/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/XTTS_v2.0_original_model_files'
print(checkpoint_dir)
model.load_checkpoint(config, checkpoint_dir = checkpoint_dir, eval = True)

model.cuda()
data_root_path = '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/commonvoice/en/clips'
out_path = '/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/samples/cv_inference/base_mono'
print(out_path)
Path(out_path).mkdir(parents = True, exist_ok = True)

gpt_cond_len = 3
language = "en"

test_accents = ['England', 'US', 'India', 'Germany', 'Southern Africa', 'Canada', 'Australia', 'Philippines', 'Scotland', 'Ireland', 'Malaysia', 'Wales']
test_langs = ['en', 'en', 'hi', 'de', 'ar', 'en', 'en', 'hi', 'en', 'en', 'hi', 'en']
test_file = '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_filelist/cv-test-full.csv'
test_file = pd.read_csv(test_file)

for index, line in tqdm(test_file.iterrows(), total = test_file.shape[0]):
    if line['accents'] in test_accents:
        out_wav_path = os.path.join(out_path, line['path'])
        if not os.path.isfile(out_wav_path):
            # outputs = model.synthesize(
            #     line['sentence'],
            #     config,
            #     speaker_wav = os.path.join(data_root_path, line['ref_wav']),
            #     # speaker_wav = '/home/hltcoe/xli/ARTS/TTS/corpora/temp/spks/7.mp3',
            #     gpt_cond_len = gpt_cond_len,
            #     language = 'en',
            #     accents = line['accents']
            # )
            breakpoint()
            outputs = model.synthesize(
                line['sentence'],
                config,
                speaker_wav = os.path.join(data_root_path, line['ref_wav']),
                gpt_cond_len = gpt_cond_len,
                # language = test_langs[test_accents.index(line['accents'])],
                language = 'en',
            )
            torchaudio.save(out_wav_path, torch.tensor(outputs['wav']).unsqueeze(0), 24000)


