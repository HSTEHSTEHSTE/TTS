from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch, torchaudio

config = XttsConfig()
config.load_json("/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2", eval=True)
model.load_checkpoint(config, checkpoint_dir="/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/XTTS_orig_new", eval=True)
model.cuda()
out_path = '/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/samples/test.wav'

outputs = model.synthesize(
    "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
    config,
    speaker_wav="/home/hltcoe/xli/ARTS/Recording.wav",
    gpt_cond_len=3,
    language="en",
    accents="India"
)

torchaudio.save(out_path, torch.tensor(outputs['wav']).unsqueeze(0), 24000)
