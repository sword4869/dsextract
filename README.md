# Routines for DeepSpeech features processing
Several routines for [DeepSpeech](https://github.com/mozilla/DeepSpeech) features processing, like speech features generation for [VOCA](https://github.com/TimoBolkart/voca) model.

## Installation

```bash
conda create -n py38 python=3.8 -y
pip install -r requirements.txt
```

https://github.com/osmr/deepspeech_features/releases/download/v0.0.1/deepspeech-0_1_0-b90017e8.pb.zip

## Usage

Generate wav files:
```bash
python3 extract_wav.py --in-video=<you_data_dir>
```

Generate files with DeepSpeech features:
```bash
# one wav file
python extract_ds_features.py --input /home/sword/Videos/语音/result/ans_wav/001.wav --output ~/Desktop/001.ds.npy --deepspeech ~/Downloads/deepspeech-0_1_0-b90017e8.pb

# a directory includes wav files
python extract_ds_features.py --input /home/sword/Videos/语音/result/ans_wav --deepspeech ~/Downloads/deepspeech-0_1_0-b90017e8.pb
```
