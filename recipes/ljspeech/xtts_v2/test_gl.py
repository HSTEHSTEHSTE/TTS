import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import torch, torchaudio, librosa

# Logging parameters
RUN_NAME = "GPT_XTTS_v2.0_CV_FT_1e-5"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "wandb"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp")

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 3  # set here the batch size
GRAD_ACUMM_STEPS = 84  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

# Define here the dataset that you want to use for the fine-tuning on.
config_dataset = BaseDatasetConfig(
    formatter = "commonvoice_accents",
    dataset_name = "commonvoice",
    path = '/home/hltcoe/xli/ARTS/TTS/corpora/commonvoice',
    meta_file_train="/home/hltcoe/xli/ARTS/TTS/corpora/accent_filelist/cv-train.csv",
    language="en",
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [config_dataset]

# Define the path where XTTS v2.0.1 files will be downloaded
# CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
CHECKPOINTS_OUT_PATH = '/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/GPT_XTTS_v2.0_CV_FT_1e-5-January-26-2025_08+22PM-744fa48'
MODEL_DOWNLOAD_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(MODEL_DOWNLOAD_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(MODEL_DOWNLOAD_PATH, os.path.basename(MEL_NORM_LINK))


# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(MODEL_DOWNLOAD_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
# XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, 'checkpoint_140000.pth')


# Training sentences generations
SPEAKER_REFERENCE = [
    "/home/hltcoe/xli/ARTS/Recording.wav"  # speaker reference to be used in training test sentences
]
LANGUAGE = config_dataset.language


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = XttsConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        run_eval_steps=2000,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=1e-5,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
                "accents": 'England'
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
                "accents": 'India'
            },
            {
                "text": "What is the most resilient parasite?",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
                "accents": 'US'
            },
            {
                "text": "Two peanuts were walking down the road. One was assaulted. Peanut.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
                "accents": 'Germany'
            },
            {
                "text": "What is the airspeed velocity of an unladen swallow?",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
                "accents": 'Southern Africa'
            },
        ],
    )

    # init the model from config
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/home/hltcoe/xli/ARTS/TTS/tts_models/accent_finetune/5e-5", eval=True)
    model.cuda()
    dvae = DiscreteVAE(
            channels=80,
            normalization=None,
            positional_dims=1,
            num_tokens=1024,
            codebook_dim=512,
            hidden_dim=512,
            num_resnet_blocks=3,
            kernel_size=3,
            num_layers=2,
            use_transposed_convs=False,
        )
    dvae.eval()
    dvae_checkpoint = torch.load('/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/XTTS_v2.0_original_model_files/dvae.pth')
    dvae.load_state_dict(dvae_checkpoint, strict=False)
    dvae.cuda()
    output = model.synthesize(text = "This cake is great. It's so delicious and moist.",
                speaker_wav = SPEAKER_REFERENCE,
                language = LANGUAGE,
                accents = 'Germany',
                config = config)
    tokens = output['gpt_tokens']
    mel = dvae.decode(tokens[:, :-1])[0].cpu().detach()
    mel_norm = torch.load(MEL_NORM_FILE)
    inverse_mel_recon = mel * mel_norm.unsqueeze(0).unsqueeze(-1) # mel_norm is given for acoustic feature extraction, you will have it from the xtts repo
    inverse_mel_recon = torch.exp(inverse_mel_recon)
    n_stft = int((1024//2) + 1)
    inverse_mel_recon = torchaudio.transforms.InverseMelScale(n_stft=n_stft, n_mels=80, sample_rate=22050, f_min=0, f_max=8000, norm="slaney")(inverse_mel_recon) # 1, 1024, T
    inverse_mel_recon = inverse_mel_recon.detach().cpu()
    gl = torchaudio.transforms.GriffinLim(n_fft = 1024, n_iter = 64, hop_length = 256, win_length = 1024)
    wav = gl(inverse_mel_recon)
    torchaudio.save('/home/hltcoe/xli/ARTS/temp/test.wav', wav, 22050)

if __name__ == "__main__":
    main()