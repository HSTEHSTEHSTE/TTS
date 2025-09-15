
# Official Release for Scalable Controllabe Accented TTS
[Paper link](https://arxiv.org/abs/2508.07426d)

## Environment
pip install -r requirements.txt

## Inference
```
python test.py  --config path_to_config.yaml\
                --checkpoint_dir path_to_checkpoint_dir\
                --out_path path_to_output_dir\
                --sentences_file path_to_test_sentences.txt\
                --speakers_file path_to_speaker_wav_list.txt
```

Note that checkpoint_dir is a directory that should contain model.pth, dvae.pth, mel_stats.pth, and vocab.json.

An inference example, as well as example test sentences and speaker wavs, may be download from https://huggingface.co/HSTE/XTTS-v2-accent-finetune and extracted to path_to_TTS_repo/tts_models/xtts_release. You may then run python test.py and use the default arguments. 


## Checkpoints
For details on the specifications of each of the following models, please refer to our manuscript.
- [Filtered](https://huggingface.co/HSTE/XTTS-v2-accent-finetune-filtered_full)
- [Unfiltered](https://huggingface.co/HSTE/XTTS-v2-accent-finetune-unfiltered_full)
- [Unlabeled](https://huggingface.co/HSTE/XTTS-v2-accent-finetune-unlabeled_full)

## Contact
For any questions please contact [Henry Li Xinyuan](https://hstehstehste.github.io/contact.html).