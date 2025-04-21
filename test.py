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
model.load_checkpoint(config, checkpoint_dir="/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/ul/", eval=True)

model.cuda()
out_path = '/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/samples/xtts_ul'

speaker_wavs = [
    "/home/hltcoe/xli/ARTS/Recording.wav",
    "/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/dev-clean/84/121123/84-121123-0002.flac",
    "/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id09185/_cDhsiYaV3c/00111.wav",
    "/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/dev-clean/251/118436/251-118436-0002.flac"
]
gpt_cond_len = 3
language = "en"

test_sentences = [
    "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
    "This cake is great. It's so delicious and moist.",
    "What is the most resilient parasite? Bacteria? Virus? Intestinal Worm?",
    "Two peanuts were walking down the road. One was assaulted. Peanut.",
    "What is the airspeed velocity of an unladen swallow?",
    "No human knows what everyone thinks, but such is life. Was that philosophical enough?",
    "Cease, cows, life is short.",
    "Do you want to take a leap of faith, or become an old man, filled with regret, waiting to die alone?",
    "I've spent my life avoiding love and romance, they are distractions.",
    "I say the whole world must learn of our peaceful ways, by force!",
    "I don't believe it is sign of strength to keep moving forward no matter what.",
    "What makes a species turn neutral: lust for gold? Power? Or were they just born with hearts full of neutrality?",
    "Gravity is desire, time is sight. What was, will be; what will be, was.",
]

test_accents = ['England', 'US', 'India', 'Germany', 'Southern Africa']

for test_accent in test_accents:
    target_path = os.path.join(out_path, test_accent)
    Path(target_path).mkdir(parents = True, exist_ok = True)
    for sentence_index, sentence in tqdm(enumerate(test_sentences), total = len(test_sentences)):
        for target_speaker_index, speaker_wav in enumerate(speaker_wavs):
            out_wav_path = os.path.join(target_path, 'sentence_' + str(sentence_index) + '_spk_' + str(target_speaker_index) + '.wav')
            outputs = model.synthesize(
                sentence,
                config,
                speaker_wav = speaker_wav,
                gpt_cond_len = gpt_cond_len,
                language = 'en',
                accents = test_accent
            )
            torchaudio.save(out_wav_path, torch.tensor(outputs['wav']).unsqueeze(0), 24000)
