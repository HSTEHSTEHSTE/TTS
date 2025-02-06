import torch

state_dict_path = '/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/XTTS_orig_new/model.pth'
model_dict = torch.load(state_dict_path)

en_index = 259
tgt_indices = [262, 284, 285, 294, 286, 260]

en = model_dict['model']['gpt.text_embedding.weight'][en_index, :]
for tgt_index in tgt_indices:
    model_dict['model']['gpt.text_embedding.weight'][tgt_index, :] = en

torch.save(model_dict, state_dict_path)
