from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch, torchaudio
import os, argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description="XTTS-v2 inference arguments.")
parser.add_argument("--config", required=True, default="tts_models/xtts_release/multi-dataset/XTTS-v2/config.json")
parser.add_argument("--checkpoint_dir", required=True, default="tts_models/xtts_release/multi-dataset/XTTS-v2", help="Path to folder containing model.pth, dvae.pth, mel_stats.pth, and vocab.json.")
parser.add_argument("--out_path", required=True, default="tts_models/xtts_release/multi-dataset/XTTS-v2/samples/cv_inference/eval")
parser.add_argument("--sentences_file", required=True, default="tts_models/xtts_release/multi-dataset/XTTS-v2/eval/sentences.txt")
parser.add_argument("--speakers_file", required=True, default="tts_models/xtts_release/multi-dataset/XTTS-v2/eval/speakers.txt")

args = parser.parse_args()

config = XttsConfig()
config.load_json(args.config)
model = Xtts.init_from_config(config)
checkpoint_dir = args.checkpoint_dir
model.load_checkpoint(config, checkpoint_dir = checkpoint_dir, eval = True)
print('Checkpoint dir: ', checkpoint_dir)

model.cuda()
out_path = args.out_path
print('Out path: ', out_path)

speaker_wavs = []
with open(args.speakers_file, 'r') as speakers_file:
    for line in speakers_file:
        speaker_wavs.append(line.strip())
gpt_cond_len = 3
language = "en"


test_sentences = []
with open(args.sentences_file, 'r') as sentences_file:
    for line in sentences_file:
        test_sentences.append(line.strip())


test_accents = ['England', 'US', 'India', 'Germany', 'Southern Africa', 'Canada', 'Australia', 'Philippines', 'Scotland', 'Ireland', 'Malaysia', 'Wales']

print(test_accents)
for test_accent_index, test_accent in enumerate(test_accents):
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
