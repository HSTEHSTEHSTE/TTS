import os, argparse

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# Args
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_name', required=True)
parser.add_argument('-f', '--meta_file_train', required=True)
parser.add_argument('-l', '--language_cond_type', required=True, default='token', help='token, embedding, code')
parser.add_argument('-ld', '--language_cond_path', default='')
parser.add_argument('--language_cond_emb_dim', default=-1)
parser.add_argument('-a', '--augmentation_type', default='None', help='None, pitchshift, kNN-VC')

class CustomTrainer(Trainer):
    def __init__(
        self,
        args,
        config,
        output_path,
        model,
        train_samples,
        eval_samples,
    ):
        super().__init__(
            args, 
            config,
            output_path,
            model = model,
            train_samples = train_samples,
            eval_samples = eval_samples
        )

    def eval_epoch(self):
        super().test_run()
        super().eval_epoch()


def main():
    args = parser.parse_args()
    RUN_NAME = args.run_name
    meta_file_train = args.meta_file_train

    # Logging parameters
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
        formatter="commonvoice_accents",
        dataset_name="commonvoice",
        path='/home/hltcoe/xli/ARTS/TTS/corpora/commonvoice',
        meta_file_train=meta_file_train,
        language="en",
    )
    print('Training file: ', meta_file_train)

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    # CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_orig_new/")
    # CHECKPOINTS_OUT_PATH = '/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/GPT_XTTS_v2.0_CV_FT_1e-5_ul-March-01-2025_04+09PM-0ebc1b9'
    MODEL_DOWNLOAD_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(MODEL_DOWNLOAD_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(MODEL_DOWNLOAD_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(MODEL_DOWNLOAD_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
    # XTTS_CHECKPOINT = os.path.join('/home/hltcoe/xli/ARTS/TTS/recipes/ljspeech/xtts_v2/exp/GPT_XTTS_v2.0_CV_FT_1e-5_mixed_full-May-04-2025_03+11PM-e355c13/checkpoint_150000.pth') # load checkpoint

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 files!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )


    # Training sentences generations
    SPEAKER_REFERENCE = [
        "/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/dev-clean/84/121123/84-121123-0002.flac",
        "/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id09185/_cDhsiYaV3c/00111.wav",
        "/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/dev-clean/251/118436/251-118436-0002.flac"
    ]
    LANGUAGE = config_dataset.language

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
    model_args.language_cond_type=args.language_cond_type
    model_args.language_cond_path=args.language_cond_path

    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
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
        run_eval_steps=10000,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=50000,
        save_n_checkpoints=100,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=1e-5,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {#0
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE[0],
                "language": LANGUAGE,
                "accents": 'England',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/England.pt',
            },
            {#1
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE[1],
                "language": LANGUAGE,
                "accents": 'US',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/US.pt',
            },
            {#2
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE[1],
                "language": LANGUAGE,
                "accents": 'Canada',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/Canada.pt',
            },
            {#3
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE[2],
                "language": LANGUAGE,
                "accents": 'India',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/India.pt',
            },
            {#4
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE[2],
                "language": LANGUAGE,
                "accents": 'Scotland',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/Scotland.pt',
            },
            {#5
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE[2],
                "language": LANGUAGE,
                "accents": 'Philippines',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/Philippines.pt',
            },
            {#6
                "text": "What is the most resilient parasite?",
                "speaker_wav": SPEAKER_REFERENCE[0],
                "language": LANGUAGE,
                "accents": 'Ireland',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/Ireland.pt',
            },
            {#7
                "text": "What is the most resilient parasite?",
                "speaker_wav": SPEAKER_REFERENCE[1],
                "language": LANGUAGE,
                "accents": 'US',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/US.pt',
            },
            {#8
                "text": "What is the most resilient parasite?",
                "speaker_wav": SPEAKER_REFERENCE[2],
                "language": LANGUAGE,
                "accents": 'Malaysia',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/Malaysia.pt',
            },
            {#9
                "text": "Two peanuts were walking down the road. One was assaulted. Peanut.",
                "speaker_wav": SPEAKER_REFERENCE[0],
                "language": LANGUAGE,
                "accents": 'India',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/India.pt',
            },
            {#10
                "text": "Two peanuts were walking down the road. One was assaulted. Peanut.",
                "speaker_wav": SPEAKER_REFERENCE[1],
                "language": LANGUAGE,
                "accents": 'Germany',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/Germany.pt',
            },
            {#11
                "text": "Two peanuts were walking down the road. One was assaulted. Peanut.",
                "speaker_wav": SPEAKER_REFERENCE[1],
                "language": LANGUAGE,
                "accents": 'Wales',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/Wales.pt',
            },
            {#12
                "text": "What is the airspeed velocity of an unladen swallow?",
                "speaker_wav": SPEAKER_REFERENCE[2],
                "language": LANGUAGE,
                "accents": 'Southern Africa',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/Southern Africa.pt',
            },
            {#13
                "text": "What is the airspeed velocity of an unladen swallow?",
                "speaker_wav": SPEAKER_REFERENCE[2],
                "language": LANGUAGE,
                "accents": 'Australia',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/Australia.pt',
            },
            {#14
                "text": "What is the airspeed velocity of an unladen swallow?",
                "speaker_wav": SPEAKER_REFERENCE[2],
                "language": LANGUAGE,
                "accents": 'Scotland',
                "language_cond_emb": '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/accent_embs/Scotland.pt',
            },
        ],
    )
    config.augmentation_type = args.augmentation_type
    config.language_cond_type = args.language_cond_type
    config.language_cond_emb_dim = args.language_cond_emb_dim

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split = True,
        eval_split_max_size = config.eval_split_max_size,
        eval_split_size = config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = CustomTrainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
