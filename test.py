from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch, torchaudio
import os
from pathlib import Path
from tqdm import tqdm

config = XttsConfig()
config.load_json("/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2", eval=True)
# model.load_checkpoint(config, checkpoint_dir="/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/GPT_XTTS_v2.0_CV_FT_1e-5_fl-February-08-2025_10+57AM-0ebc1b9", eval=True)
# model.load_checkpoint(config, checkpoint_dir="/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/GPT_XTTS_v2.0_CV_FT_1e-5_l-February-08-2025_10+49AM-0ebc1b9", eval=True)
model.load_checkpoint(config, checkpoint_dir="/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/GPT_XTTS_v2.0_CV_FT_1e-5_b-February-07-2025_11+08AM-0ebc1b9", eval=True)

model.cuda()
out_path = '/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/samples/xtts_b'

speaker_wav = "/home/hltcoe/xli/ARTS/Recording.wav"
gpt_cond_len = 3,
language = "en",

test_sentences = [
    "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
    "This cake is great. It's so delicious and moist.",
    "What is the most resilient parasite?",
    "Two peanuts were walking down the road. One was assaulted. Peanut.",
    "What is the airspeed velocity of an unladen swallow?",
]

test_accents = ['England', 'US', 'India', 'Germany', 'Southern Africa']

for test_accent in test_accents:
    target_path = os.path.join(out_path, test_accent)
    Path(target_path).mkdir(parents = True, exist_ok = True)
    for sentence_index, sentence in tqdm(enumerate(test_sentences), total = len(test_sentences)):
        out_wav_path = os.path.join(target_path, str(sentence_index) + '.wav')
        outputs = model.synthesize(
            sentence,
            config,
            speaker_wav = speaker_wav,
            gpt_cond_len = gpt_cond_len,
            language = 'en',
            accents = test_accent
        )
        torchaudio.save(out_wav_path, torch.tensor(outputs['wav']).unsqueeze(0), 24000)
