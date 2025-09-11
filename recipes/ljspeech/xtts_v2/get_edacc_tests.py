import os, random, json, torchaudio

edacc_path = '/home/hltcoe/xli/ARTS/TTS/corpora/edacc'

accent_spks = {
    'English': ['EDACC-C06-A', 'EDACC-C06-B', 'EDACC-C11-B'],
    'African': ['EDACC-C19-B', 'EDACC-C29-A', 'EDACC-C35-A'],
    'American': ['EDACC-C63-B', 'EDACC-C14-A', 'EDACC-C64-A'],
    'Indian': ['EDACC-C09-A', 'EDACC-C10-A', 'EDACC-C34-B']
}

spks = []
for accent_spk in accent_spks:
    spks += accent_spks[accent_spk]

utt2spk_files = ['/home/hltcoe/xli/ARTS/TTS/corpora/edacc/dev/utt2spk', '/home/hltcoe/xli/ARTS/TTS/corpora/edacc/test/utt2spk']
text_files = ['/home/hltcoe/xli/ARTS/TTS/corpora/edacc/dev/text', '/home/hltcoe/xli/ARTS/TTS/corpora/edacc/test/text']
segment_files = ['/home/hltcoe/xli/ARTS/TTS/corpora/edacc/dev/segments', '/home/hltcoe/xli/ARTS/TTS/corpora/edacc/test/segments']

spk2utt = {}

for utt2spk_file in utt2spk_files:
    with open(utt2spk_file, 'r') as utt2spk_file_lines:
        for line in utt2spk_file_lines:
            line_elements = line.strip().split()
            utt_id = line_elements[0]
            spk = line_elements[1]
            if spk in spks:
                if spk not in spk2utt:
                    spk2utt[spk] = [utt_id]
                else:
                    spk2utt[spk].append(utt_id)

random.seed(24)
sample_dict = {}
utt2spk = {}
for spk in spk2utt:
    # spk_samples = random.sample(spk2utt[spk], k = 10)
    spk_samples = spk2utt[spk]
    sample_dict[spk] = {}
    for spk_sample in spk_samples:
        utt2spk[spk_sample] = spk
        sample_dict[spk][spk_sample] = {}

for text_file in text_files:
    with open(text_file, 'r') as text_file_lines:
        for line in text_file_lines:
            line_elements = line.strip().split()
            utt_id = line_elements[0]
            if utt_id in utt2spk:
                transcript = ' '.join(line_elements[1:])
                sample_dict[utt2spk[utt_id]][utt_id]['transcript'] = transcript

for segment_file in segment_files:
    with open(segment_file, 'r') as segment_file_lines:
        for line in segment_file_lines:
            line_elements = line.strip().split()
            utt_id = line_elements[0]
            if utt_id in utt2spk:
                sample_dict[utt2spk[utt_id]][utt_id]['file_name'] = line_elements[1]
                sample_dict[utt2spk[utt_id]][utt_id]['start_time'] = float(line_elements[2])
                sample_dict[utt2spk[utt_id]][utt_id]['end_time'] = float(line_elements[3])

wav_base_path = '/home/hltcoe/xli/ARTS/TTS/corpora/edacc/data'
out_path = '/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/data/edacc_segments'
with open(os.path.join(out_path, 'meta.json'), 'w') as out_file:
    json.dump(sample_dict, out_file)
wavs = {}
for spk in sample_dict:
    for utt_id in sample_dict[spk]:
        file_name = sample_dict[spk][utt_id]['file_name']
        if file_name not in wavs:
            wav, sr = torchaudio.load(os.path.join(wav_base_path, file_name + '.wav'))
            wavs[file_name] = wav
        start_time = int(sample_dict[spk][utt_id]['start_time'] * sr)
        end_time = int(sample_dict[spk][utt_id]['end_time'] * sr)
        out_wav = wavs[file_name][:, start_time:end_time]
        torchaudio.save(os.path.join(out_path, utt_id + '.wav'), out_wav, sr)
