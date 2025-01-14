from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch, torchaudio

config = XttsConfig()
config.load_json("/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2", eval=True)
model.cuda()
out_path = '/home/hltcoe/xli/ARTS/TTS/tts_models/multilingual/multi-dataset/XTTS-v2/samples/test.wav'

outputs = model.synthesize(
    "This limits the functions that could be executed during unpickling.",
    config,
    speaker_wav="/home/hltcoe/xli/ARTS/Recording.wav",
    gpt_cond_len=3,
    language="es",
)

torchaudio.save(out_path, torch.tensor(outputs['wav']).unsqueeze(0), 24000)
