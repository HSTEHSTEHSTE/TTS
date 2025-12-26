import torch
import torchaudio
import numpy as np
import os
import argparse
import random
import torch.nn.functional as F
from pathlib import Path
from typing import Union
from tqdm import tqdm
import time
import math
import json
import faiss
import pandas as pd
from multiprocessing import Pool

profile_path = '/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/profiles'
base_wav_path = '/home/hltcoe/xli/ARTS/base.wav'
target_path = '/exp/xli/ARTS/librispeech_centroids'
SPEAKER_INFORMATION_LAYER = 6

# This function makes the FAISS index. It implements the NN-search
def make_index(data,
    num_coarse_clusters=100,
    m=8,
    bits_per_code=8,
    dim=1024,
    num_points=1500000,
    device=None,
):
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(
        quantizer, dim, num_coarse_clusters, m, bits_per_code
    )
    
    if data is None:
        data = np.random.random((1500000, dim)).astype('float32') - 0.5
    
    index.train(data)
    index.add(data)

    index.make_direct_map()
    return index

def fast_cosine_dist(
    source_feats: torch.Tensor,
    matching_pool: torch.Tensor,
    device: str = 'cpu'
) -> torch.Tensor:
    """Computes cosine distance efficiently."""
    #source_feats = source_feats.unsqueeze(0)  # Ensure correct shape
    source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)

    dotprod = (
        -torch.cdist(source_feats.to(device), matching_pool.unsqueeze(0), p=2)[0] ** 2
        + source_norms[:, None] ** 2
        + matching_norms[None] ** 2
    )
    dotprod /= 2

    dists = 1 - (dotprod / (source_norms[:, None] * matching_norms[None]))
    return dists


class KNN_VC_Wrapper:
    """Wrapper for KNN-VC model."""

    def __init__(self, knn_vc):
        self.knn_vc_model = knn_vc
        self.device = knn_vc.device
        self.hop_length = 320
        self.weighting = knn_vc.weighting

        self.h = knn_vc.h
        self.sr = knn_vc.h.sampling_rate

        self.nprobe = 2

    def get_features(self, audio, weights=None, vad_trigger_level=0):
        # load audio
        if weights == None: weights = self.weighting
        if type(audio) in [str, Path]:
            x, sr = torchaudio.load(audio, normalize=True)
        else:
            x: Tensor = audio
            sr = self.sr
            if x.dim() == 1: x = x[None]
                
        if not sr == self.sr :
            # print(f"resample {sr} to {self.sr} in {audio}")
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.sr)
            sr = self.sr
        
        x = x[:, :20 * 16000]

        audio_len = x.shape[1]
            
        # trim silence from front and back
        if vad_trigger_level > 1e-3:
            transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            # original way, disabled because it lacks windows support
            #waveform_reversed, sr = apply_effects_tensor(x_front_trim, sr, [["reverse"]])
            waveform_reversed = torch.flip(x_front_trim, (-1,))
            waveform_reversed_front_trim = transform(waveform_reversed)
            waveform_end_trim = torch.flip(waveform_reversed_front_trim, (-1,))
            #waveform_end_trim, sr = apply_effects_tensor(
            #    waveform_reversed_front_trim, sr, [["reverse"]]
            #)
            x = waveform_end_trim

        # extract the representation of each layer
        wav_input_16khz = x.to(self.device)
        if torch.allclose(weights, self.weighting):
            # use fastpath
            features = self.knn_vc_model.wavlm.extract_features(wav_input_16khz, output_layer=SPEAKER_INFORMATION_LAYER, ret_layer_results=False)[0]
            features = features.squeeze(0)
        else:
            # use slower weighted
            rep, layer_results = self.wavlm.extract_features(wav_input_16khz, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
            features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)
            # save full sequence
            features = ( features*weights[:, None] ).sum(dim=0) # (seq_len, dim)
        
        return features, audio_len

    def get_matching_set(self, wavs: list[Union[Path, torch.Tensor]], weights=None, vad_trigger_level=0) -> torch.Tensor:
        feats = []
        total_length = 0
        for p in wavs:
            feat, audio_len = self.get_features(
                p, 
                weights = self.weighting if weights is None else weights, 
                vad_trigger_level = vad_trigger_level
            )
            feats.append(feat)
            total_length += audio_len
            if audio_len > 15 * 60 * 16000:
                break
        
        feats = torch.concat(feats, dim=0).cpu()

        index = make_index(
            feats.detach().numpy(),
            num_coarse_clusters = 1024,
            device = str(self.device),
        )
        return feats, index

    def match_list(
        self, 
        query_seq: torch.Tensor, 
        matching_sets: list[torch.Tensor], 
        indices,
        recon_indices,
        topk: int = 4,
        target_duration: float | None = None, 
        weights: torch.Tensor | None = None
    ):
        """Finds the best-matching features."""
        
        device = self.device
        matching_sets = [matching_set for matching_set in matching_sets]
        query_seq = query_seq[0].cpu().detach()

        if target_duration is not None:
            target_samples = int(target_duration * 16000)
            scale_factor = (target_samples / self.hop_length) / query_seq.shape[0]
            query_seq = F.interpolate(query_seq.T[None], scale_factor=scale_factor, mode='linear')[0].T

        out_feats = []
        query_seq = query_seq.numpy()

        for matching_set_index, matching_set in enumerate(matching_sets):
            recon_index = recon_indices[matching_set_index]

            # gpu search and reconstruct
            index = indices[matching_set_index]
            distances, vectors = index.search(
                query_seq, topk
            )
            rows = []
            
            for timestep in range(vectors.shape[0]):
                row = []
                for k in range(vectors.shape[1]):
                    if int(vectors[timestep, k]) > -1:
                        row.append(recon_index.reconstruct(int(vectors[timestep, k])))
                    else:
                        row.append(recon_index.reconstruct(0))
                rows.append(row)
            rows = torch.tensor(np.array(rows)).mean(dim = 1)
            out_feats.append(rows)


        
        return out_feats

    def vocode(self, c: torch.Tensor) -> torch.Tensor:
        """Generates waveform from features."""
        return self.knn_vc_model.vocode(c.to(self.device))

# Device Selection
if torch.cuda.is_available():
    device = 'cuda'
    res = faiss.StandardGpuResources()
else:
    device = 'cpu'

# Device Selection

# Load KNN-VC Model
knn_vc = torch.hub.load(
    'bshall/knn-vc',
    'knn_vc',
    prematched=True,
    trust_repo=True,
    pretrained=True,
)
knn_vc.to(device)
    
# Wrap the model
knn_vc = KNN_VC_Wrapper(knn_vc)

profiles = list(Path(profile_path).rglob('*.pt'))
# Feature Extraction
query_seq = knn_vc.get_features(base_wav_path)

for profile in tqdm(profiles):
    index_path = str(profile)[:-3] + '.index'
    profile_name = str(profile).split('/')[-1][:-3]

    if not os.path.isfile(os.path.join(target_path, 'reduced', profile_name + '.pt')):

        recon_index = faiss.read_index(index_path)
        index = [faiss.index_cpu_to_gpu(res, 0, recon_index)]
        recon_index = [recon_index]
        matching_set = [torch.load(profile)]

        with torch.no_grad():
            # Match & Vocode
            out_feats = knn_vc.match_list(
                query_seq, 
                matching_set, 
                index,
                recon_index, 
                topk = 4, 
                weights = None
            )[0]
            
            out_feats_reduced = torch.mean(out_feats, dim = 0)
            torch.save(out_feats_reduced, os.path.join(target_path, 'reduced', profile_name + '.pt'))