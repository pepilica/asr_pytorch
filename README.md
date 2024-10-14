# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains an ASR pipeline using PyTorch and DeepSpeech2 as main model.

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## Training

The model was trained in 4 steps. To reproduce, train model using the following commands:

1. Train 30 epochs without augmentations on clean-100 dataset

   ```bash
   python train.py -cn=deepspeech2_pretrain
   ```

2. Train 50 epochs wigh augmentations on other-500 dataset

   ```bash
   python train.py -cn=deepspeech2_finetune_other
   ```

3. Train 20 epochs wigh augmentations on clean-360 dataset

   ```bash
   python train.py -cn=deepspeech2_finetune_clean_2
   ```

4. Train 100k steps (or until Kaggle won't shut your process) wigh augmentations on clean-360 dataset

   ```bash
   python train.py -cn=deepspeech2_finetune_clean_3
   ```

### Inference

   1. If you want only to decode audio to text, your directory with audio should has the following format:
      ```
      NameOfTheDirectoryWithUtterances
      └── audio
         ├── UtteranceID1.wav # may be flac or mp3
         ├── UtteranceID2.wav
         .
         .
         .
         └── UtteranceIDn.wav
      ```
      Run the following command:
      ```bash
      python inference.py datasets=inference_custom inferencer.save_path=SAVE_PATH datasets.test.audio_dir=TEST_DATA/audio
      ```
      where `SAVE_PATH` is a path to save predicted text and `TEST_DATA` is directory with audio.
   2. If you have ground truth text and want to evaluate model, make sure that directory with audio and ground truth text has the following format:
      ```
      NameOfTheDirectoryWithUtterances
      ├── audio
      │   ├── UtteranceID1.wav # may be flac or mp3
      │   ├── UtteranceID2.wav
      │   .
      │   .
      │   .
      │   └── UtteranceIDn.wav
      └── transcriptions
         ├── UtteranceID1.txt
         ├── UtteranceID2.txt
         .
         .
         .
         └── UtteranceIDn.txt
      ```
      Then run the following command:
      ```bash
      python inference.py datasets=inference_custom inferencer.save_path=SAVE_PATH datasets.test.audio_dir=TEST_DATA/audio datasets.test.transcription_dir=TEST_DATA/transcriptions
      ```
   3. If you only have predicted and ground truth texts and only want to evaluate model, make sure that directory with ones has the following format:
      ```
      NameOfTheDirectoryWithUtterances
      ├── ID1.json # may be flac or mp3
      .
      .
      .
      └── IDn.json

      ID1 = {"pred_text": "ye are newcomers", "text": "YE ARE NEWCOMERS"}
      ```
      Then run the following command:
      ```bash
      python calculate_wer_cer.py --dir_path=DIR
      ```
   4. Finally, if you want to reproduce results, run the following code:
      ```bash
      python inference.py
      ```
      Feel free to choose what kind of metrics you want to evaluate (see [this config](src/configs/metrics/inference.yaml)).

## Results

This results were obtained using beam search w/ langauge model:

```angular2html
                WER     CER
test-clean     28.12     8.6
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
