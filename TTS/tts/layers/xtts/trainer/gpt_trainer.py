from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from coqpit import Coqpit
from torch.nn import functional as F
from torch.utils.data import DataLoader
from trainer.torch import DistributedSampler
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.utils.io import load_fsspec

import numpy as np
import os
import random
from pathlib import Path
from typing import Union
import math
import faiss
from multiprocessing import Pool


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
            features = self.knn_vc_model.wavlm.extract_features(wav_input_16khz, output_layer=6, ret_layer_results=False)[0]
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

@dataclass
class GPTTrainerConfig(XttsConfig):
    lr: float = 5e-06
    training_seed: int = 1
    optimizer_wd_only_on_weights: bool = False
    weighted_loss_attrs: dict = field(default_factory=lambda: {})
    weighted_loss_multipliers: dict = field(default_factory=lambda: {})
    test_sentences: List[dict] = field(default_factory=lambda: [])
    augmentation_type: str = 'None' # None, pitchshift, kNN-VC
    language_cond_type: str = 'token' # 'token', 'embedding', 'code'
    language_cond_emb_dim: int = -1


@dataclass
class XttsAudioConfig(XttsAudioConfig):
    dvae_sample_rate: int = 22050


@dataclass
class GPTArgs(XttsArgs):
    min_conditioning_length: int = 66150
    max_conditioning_length: int = 132300
    gpt_loss_text_ce_weight: float = 0.01
    gpt_loss_mel_ce_weight: float = 1.0
    gpt_num_audio_tokens: int = 8194
    debug_loading_failures: bool = False
    max_wav_length: int = 255995  # ~11.6 seconds
    max_text_length: int = 200
    tokenizer_file: str = ""
    mel_norm_file: str = "https://coqui.gateway.scarf.sh/v0.14.0_models/mel_norms.pth"
    dvae_checkpoint: str = ""
    xtts_checkpoint: str = ""
    gpt_checkpoint: str = ""  # if defined it will replace the gpt weights on xtts model
    vocoder: str = ""  # overide vocoder key on the config to avoid json write issues
    language_cond_type = 'token' # 'token', 'embedding', 'code'
    language_cond_path = ''


def callback_clearml_load_save(operation_type, model_info):
    # return None means skip the file upload/log, returning model_info will continue with the log/upload
    # you can also change the upload destination file name model_info.upload_filename or check the local file size with Path(model_info.local_model_path).stat().st_size
    assert operation_type in ("load", "save")
    # print(operation_type, model_info.__dict__)

    if "similarities.pth" in model_info.__dict__["local_model_path"]:
        return None

    return model_info


class GPTTrainer(BaseTTS):
    def __init__(self, config: Coqpit):
        """
        Tortoise GPT training class
        """
        super().__init__(config, ap=None, tokenizer=None)
        self.config = config
        # init XTTS model
        self.xtts = Xtts(self.config)
        # create the tokenizer with the target vocabulary
        self.xtts.tokenizer = VoiceBpeTokenizer(self.args.tokenizer_file)
        # init gpt encoder and hifigan decoder
        self.xtts.init_models()

        if self.args.xtts_checkpoint:
            self.load_checkpoint(self.config, self.args.xtts_checkpoint, eval=False, strict=False)

        # set mel stats
        if self.args.mel_norm_file:
            self.xtts.mel_stats = load_fsspec(self.args.mel_norm_file)

        # load GPT if available
        if self.args.gpt_checkpoint:
            gpt_checkpoint = torch.load(self.args.gpt_checkpoint, map_location=torch.device("cpu"))
            # deal with coqui Trainer exported model
            if "model" in gpt_checkpoint.keys() and "config" in gpt_checkpoint.keys():
                print("Coqui Trainer checkpoint detected! Converting it!")
                gpt_checkpoint = gpt_checkpoint["model"]
                states_keys = list(gpt_checkpoint.keys())
                for key in states_keys:
                    if "gpt." in key:
                        new_key = key.replace("gpt.", "")
                        gpt_checkpoint[new_key] = gpt_checkpoint[key]
                        del gpt_checkpoint[key]
                    else:
                        del gpt_checkpoint[key]

            # edit checkpoint if the number of tokens is changed to ensures the better transfer learning possible
            if (
                "text_embedding.weight" in gpt_checkpoint
                and gpt_checkpoint["text_embedding.weight"].shape != self.xtts.gpt.text_embedding.weight.shape
            ):
                num_new_tokens = (
                    self.xtts.gpt.text_embedding.weight.shape[0] - gpt_checkpoint["text_embedding.weight"].shape[0]
                )
                print(f" > Loading checkpoint with {num_new_tokens} additional tokens.")

                # add new tokens to a linear layer (text_head)
                emb_g = gpt_checkpoint["text_embedding.weight"]
                new_row = torch.randn(num_new_tokens, emb_g.shape[1])
                start_token_row = emb_g[-1, :]
                emb_g = torch.cat([emb_g, new_row], axis=0)
                emb_g[-1, :] = start_token_row
                gpt_checkpoint["text_embedding.weight"] = emb_g

                # add new weights to the linear layer (text_head)
                text_head_weight = gpt_checkpoint["text_head.weight"]
                start_token_row = text_head_weight[-1, :]
                new_entry = torch.randn(num_new_tokens, self.xtts.gpt.text_head.weight.shape[1])
                text_head_weight = torch.cat([text_head_weight, new_entry], axis=0)
                text_head_weight[-1, :] = start_token_row
                gpt_checkpoint["text_head.weight"] = text_head_weight

                # add new biases to the linear layer (text_head)
                text_head_bias = gpt_checkpoint["text_head.bias"]
                start_token_row = text_head_bias[-1]
                new_bias_entry = torch.zeros(num_new_tokens)
                text_head_bias = torch.cat([text_head_bias, new_bias_entry], axis=0)
                text_head_bias[-1] = start_token_row
                gpt_checkpoint["text_head.bias"] = text_head_bias

            self.xtts.gpt.load_state_dict(gpt_checkpoint, strict=True)
            print(">> GPT weights restored from:", self.args.gpt_checkpoint)

        # Mel spectrogram extractor for conditioning
        if self.args.gpt_use_perceiver_resampler:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=2048,
                hop_length=256,
                win_length=1024,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )
        else:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )

        # Load DVAE
        self.dvae = DiscreteVAE(
            channels=80,
            normalization=None,
            positional_dims=1,
            num_tokens=self.args.gpt_num_audio_tokens - 2,
            codebook_dim=512,
            hidden_dim=512,
            num_resnet_blocks=3,
            kernel_size=3,
            num_layers=2,
            use_transposed_convs=False,
        )

        self.dvae.eval()
        if self.args.dvae_checkpoint:
            dvae_checkpoint = torch.load(self.args.dvae_checkpoint, map_location=torch.device("cpu"))
            self.dvae.load_state_dict(dvae_checkpoint, strict=False)
            print(">> DVAE weights restored from:", self.args.dvae_checkpoint)
        else:
            raise RuntimeError(
                "You need to specify config.model_args.dvae_checkpoint path to be able to train the GPT decoder!!"
            )

        # Mel spectrogram extractor for DVAE
        self.torch_mel_spectrogram_dvae = TorchMelSpectrogram(
            mel_norm_file=self.args.mel_norm_file, sampling_rate=config.audio.dvae_sample_rate
        )

        self.augmentation_type = self.config.augmentation_type
        if self.augmentation_type == 'pitchshift':
            self.pitch_augment_range = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
            self.pitch_augments = {}
            for pitch_augment_n_step in self.pitch_augment_range:
                if pitch_augment_n_step != 0:
                    self.pitch_augments[pitch_augment_n_step] = torchaudio.transforms.PitchShift(self.config.audio.sample_rate, pitch_augment_n_step).to('cuda')
        elif self.augmentation_type == 'kNN-VC':
            # Load KNN-VC Model
            knn_vc = torch.hub.load(
                'bshall/knn-vc',
                'knn_vc',
                prematched=True,
                trust_repo=True,
                pretrained=True,
            )
            knn_vc.to('cuda')
            
            # Wrap the model
            self.knn_vc = KNN_VC_Wrapper(knn_vc)

            # Load Speaker Data
            data_dir = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech')
            speaker_file = data_dir / "SPEAKERS.TXT"

            self.profile_dir = '/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/profiles'

            # Get the speakers
            self.speakers = []
            with open(speaker_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) > 2 and parts[1] != 'SEX' and 'train' in parts[2]:
                        spk_id, spk_set = parts[0].strip(), parts[2].strip()
                        self.speakers.append(spk_id)

            self.sampler = torchaudio.transforms.Resample(self.config.audio.sample_rate, 16000)
            self.resampler = torchaudio.transforms.Resample(16000, self.config.audio.sample_rate)
            self.res = faiss.StandardGpuResources()
        

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens, language_cond):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, num_samples, 80,t_m)
        cond_idxs: cond start and end indexs, (b, 2)
        cond_lens: long tensor, (b,)
        """
        losses = self.xtts.gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
            language_cond=language_cond,
        )
        return losses

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:  # pylint: disable=W0613
        test_audios = {}
        if self.config.test_sentences:
            # init gpt for inference mode
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.xtts.gpt.eval()
            print(" | > Synthesizing test sentences.")
            for idx, s_info in enumerate(self.config.test_sentences):
                wav = self.xtts.synthesize(
                    s_info["text"],
                    self.config,
                    s_info["speaker_wav"],
                    s_info["language"],
                    accents=s_info["accents"],
                    gpt_cond_len=3,
                    language_cond_emb=s_info["language_cond_emb"],
                )["wav"]
                test_audios["{}-audio".format(idx)] = wav

            # delete inference layers
            del self.xtts.gpt.gpt_inference
            del self.xtts.gpt.gpt.wte
        return {"audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.args.output_sample_rate)

    def format_batch(self, batch: Dict) -> Dict:
        return batch

    @torch.no_grad()  # torch no grad to avoid gradients from the pre-processing and DVAE codes extraction
    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        batch["text_lengths"] = batch["text_lengths"]
        batch["wav_lengths"] = batch["wav_lengths"]
        batch["text_inputs"] = batch["padded_text"]
        batch["cond_idxs"] = batch["cond_idxs"]
        
        if self.augmentation_type == 'pitchshift':
            # pitchshift augmentation
            pitch_augment_n_step = random.randint(self.pitch_augment_range[0], self.pitch_augment_range[-1])
            if pitch_augment_n_step != 0:
                wavs = []
                for source_wav_index in range(batch['wav'].shape[0]):
                    source_wav = batch['wav'][source_wav_index]
                    out_wav = self.pitch_augments[pitch_augment_n_step](source_wav)
                    wavs.append(out_wav)

                conds = []
                for source_wav_index in range(batch['conditioning'].shape[0]):
                    source_wav = batch['conditioning'][source_wav_index]
                    out_wav = self.pitch_augments[pitch_augment_n_step](source_wav)
                    conds.append(out_wav)
                batch['wav'] = torch.stack(wavs, dim = 0)
                batch['conditioning'] = torch.stack(conds, dim = 0)
        elif self.augmentation_type == 'kNN-VC':
            # kNN-VC augmentation
            augment = random.randint(0, 1) # aug
            # augment = 0 # noaug

            if augment == 1:
                # perform knn-vc data augmentation
                spk = random.choice(self.speakers)
                recon_index = faiss.read_index(os.path.join(self.profile_dir, spk + '.index'))
                index = [faiss.index_cpu_to_gpu(self.res, 0, recon_index)]
                recon_index = [recon_index]
                matching_set = [torch.load(os.path.join(self.profile_dir, spk + '.pt'))]
                wavs = []
                for source_wav_index in range(batch['wav'].shape[0]):
                    source_wav = batch['wav'][source_wav_index]
                    source_wav = self.sampler(source_wav)
                    # Feature Extraction
                    query_seq = self.knn_vc.get_features(source_wav)
                    # Match & Vocode
                    out_feats = self.knn_vc.match_list(
                        query_seq, 
                        matching_set, 
                        index,
                        recon_index, 
                        topk = 4, 
                        weights = None
                    )[0]
                    out_wav = self.resampler(self.knn_vc.vocode(out_feats.unsqueeze(0)))
                    wavs.append(out_wav)
                conds = []
                for source_wav_index in range(batch['conditioning'].shape[0]):
                    source_wav = batch['conditioning'][source_wav_index].squeeze(0)
                    # Feature Extraction
                    query_seq = self.knn_vc.get_features(source_wav)
                    # Match & Vocode
                    out_feats = self.knn_vc.match_list(
                        query_seq, 
                        matching_set, 
                        index,
                        recon_index, 
                        topk = 4, 
                        weights = None
                    )[0]
                    out_wav = self.resampler(self.knn_vc.vocode(out_feats.unsqueeze(0))).unsqueeze(0)
                    conds.append(out_wav)
                
                batch['wav'] = torch.stack(wavs, dim = 0)
                batch['conditioning'] = torch.stack(conds, dim = 0)

        # compute conditioning mel specs
        # transform waves from torch.Size([B, num_cond_samples, 1, T] to torch.Size([B * num_cond_samples, 1, T] because if is faster than iterate the tensor
        B, num_cond_samples, C, T = batch["conditioning"].size()
        conditioning_reshaped = batch["conditioning"].view(B * num_cond_samples, C, T)
        paired_conditioning_mel = self.torch_mel_spectrogram_style_encoder(conditioning_reshaped)
        # transform torch.Size([B * num_cond_samples, n_mel, T_mel]) in torch.Size([B, num_cond_samples, n_mel, T_mel])
        n_mel = self.torch_mel_spectrogram_style_encoder.n_mel_channels  # paired_conditioning_mel.size(1)
        T_mel = paired_conditioning_mel.size(2)
        paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond_samples, n_mel, T_mel)
        # get the conditioning embeddings
        batch["cond_mels"] = paired_conditioning_mel
        # compute codes using DVAE
        if self.config.audio.sample_rate != self.config.audio.dvae_sample_rate:
            dvae_wav = torchaudio.functional.resample(
                batch["wav"],
                orig_freq=self.config.audio.sample_rate,
                new_freq=self.config.audio.dvae_sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        else:
            dvae_wav = batch["wav"]
        dvae_mel_spec = self.torch_mel_spectrogram_dvae(dvae_wav)
        codes = self.dvae.get_codebook_indices(dvae_mel_spec)

        batch["audio_codes"] = codes
        # delete useless batch tensors
        del batch["padded_text"]
        del batch["wav"]
        del batch["conditioning"]
        return batch

    def train_step(self, batch, criterion):
        loss_dict = {}
        cond_mels = batch["cond_mels"]
        text_inputs = batch["text_inputs"]
        text_lengths = batch["text_lengths"]
        audio_codes = batch["audio_codes"]
        wav_lengths = batch["wav_lengths"]
        cond_idxs = batch["cond_idxs"]
        cond_lens = batch["cond_lens"]
        language_cond = batch["language_cond"]

        loss_text, loss_mel, _ = self.forward(
            text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens, language_cond
        )
        loss_dict["loss_text_ce"] = loss_text * self.args.gpt_loss_text_ce_weight
        loss_dict["loss_mel_ce"] = loss_mel * self.args.gpt_loss_mel_ce_weight
        loss_dict["loss"] = loss_dict["loss_text_ce"] + loss_dict["loss_mel_ce"]
        return {"model_outputs": None}, loss_dict

    def eval_step(self, batch, criterion):
        # ignore masking for more consistent evaluation
        batch["cond_idxs"] = None
        return self.train_step(batch, criterion)

    def on_train_epoch_start(self, trainer):
        trainer.model.eval()  # the whole model to eval
        # put gpt model in training mode
        if hasattr(trainer.model, "module") and hasattr(trainer.model.module, "xtts"):
            trainer.model.module.xtts.gpt.train()
        else:
            trainer.model.xtts.gpt.train()

    def on_init_end(self, trainer):  # pylint: disable=W0613
        # ignore similarities.pth on clearml save/upload
        if self.config.dashboard_logger.lower() == "clearml":
            from clearml.binding.frameworks import WeightsFileHandler

            WeightsFileHandler.add_pre_callback(callback_clearml_load_save)

    @torch.no_grad()
    def inference(
        self,
        x,
        aux_input=None,
    ):  # pylint: disable=dangerous-default-value
        return None

    @staticmethod
    def get_criterion():
        return None

    def get_sampler(self, dataset: TTSDataset, num_gpus=1):
        # sampler for DDP
        batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        return batch_sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":  # pylint: disable=W0613
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = XTTSDataset(self.config, samples, self.xtts.tokenizer, config.audio.sample_rate, is_eval)

            # wait all the DDP process to be ready
            if num_gpus > 1:
                torch.distributed.barrier()

            # sort input sequences from short to long
            # dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(dataset, num_gpus)

            # ignore sampler when is eval because if we changed the sampler parameter we will not be able to compare previous runs
            if sampler is None or is_eval:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size = config.eval_batch_size if is_eval else config.batch_size,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
        return loader

    def get_optimizer(self) -> List:
        """Initiate and return the optimizer based on the config parameters."""
        # ToDo: deal with multi GPU training
        if self.config.optimizer_wd_only_on_weights:
            # parameters to only GPT model
            net = self.xtts.gpt

            # normalizations
            norm_modules = (
                nn.BatchNorm2d,
                nn.InstanceNorm2d,
                nn.BatchNorm1d,
                nn.InstanceNorm1d,
                nn.BatchNorm3d,
                nn.InstanceNorm3d,
                nn.GroupNorm,
                nn.LayerNorm,
            )
            # nn.Embedding
            emb_modules = (nn.Embedding, nn.EmbeddingBag)

            param_names_notweights = set()
            all_param_names = set()
            param_map = {}
            for mn, m in net.named_modules():
                for k, v in m.named_parameters():
                    v.is_bias = k.endswith(".bias")
                    v.is_weight = k.endswith(".weight")
                    v.is_norm = isinstance(m, norm_modules)
                    v.is_emb = isinstance(m, emb_modules)

                    fpn = "%s.%s" % (mn, k) if mn else k  # full param name
                    all_param_names.add(fpn)
                    param_map[fpn] = v
                    if v.is_bias or v.is_norm or v.is_emb:
                        param_names_notweights.add(fpn)

            params_names_notweights = sorted(list(param_names_notweights))
            params_notweights = [param_map[k] for k in params_names_notweights]
            params_names_weights = sorted(list(all_param_names ^ param_names_notweights))
            params_weights = [param_map[k] for k in params_names_weights]

            groups = [
                {"params": params_weights, "weight_decay": self.config.optimizer_params["weight_decay"]},
                {"params": params_notweights, "weight_decay": 0},
            ]
            # torch.optim.AdamW
            opt = get_optimizer(
                self.config.optimizer,
                self.config.optimizer_params,
                self.config.lr,
                parameters=groups,
            )
            opt._group_names = [params_names_weights, params_names_notweights]
            return opt

        return get_optimizer(
            self.config.optimizer,
            self.config.optimizer_params,
            self.config.lr,
            # optimize only for the GPT model
            parameters=self.xtts.gpt.parameters(),
        )

    def get_scheduler(self, optimizer) -> List:
        """Set the scheduler for the optimizer.

        Args:
            optimizer: `torch.optim.Optimizer`.
        """
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        eval=False,
        strict=True,
        cache_storage="/tmp/tts_cache",
        target_protocol="s3",
        target_options={"anon": True},
    ):  # pylint: disable=unused-argument, disable=W0201, disable=W0102, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""

        state = self.xtts.get_compatible_checkpoint_state_dict(checkpoint_path)

        # load the model weights
        self.xtts.load_state_dict(state, strict=strict)

        if eval:
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: "GPTTrainerConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (GPTTrainerConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        return GPTTrainer(config)
