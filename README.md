# Routines for DeepSpeech features processing
Several routines for [DeepSpeech](https://github.com/mozilla/DeepSpeech) features processing, like speech features generation for [VOCA](https://github.com/TimoBolkart/voca) model.

## Installation


```bash
conda install cudatoolkit=11.7 -c nvidia

pip install -r requirements.txt
```

https://github.com/osmr/deepspeech_features/releases/download/v0.0.1/deepspeech-0_1_0-b90017e8.pb.zip

## Differences
1. **using the new version of tensorflow 2.12.0** , rather than original version of 1.15
2. add the feature: `video-fps=50`
## Usage

extract wav audio(ar=16000, ac=1) files from video:
```bash
# 单个视频
# 输出的音频同视频名.wav
python3 extract_wav.py --in-video=/home/sword/Music/Obama.mp4
# 指定输出的音频名，但还是wav格式
python3 extract_wav.py --in-video=/home/sword/Music/Obama.mp4 --out-audio=/home/sword/Music/1.wav

# 包含多个视频的文件夹
# 跳过非文件的子文件夹，只支持mp4,mkv,avi格式，输出的音频同各视频名
python3 extract_wav.py --in-video=/home/sword/Music
```

extract DeepSpeech features from wav audio:
```bash
# one wav file
# 默认输出的fps=25, 下载deepspeech权重文件，输出同路径下
python extract_ds_features.py --input /home/sword/Music/Obama.wav 
# 指定输出路径文件名，指定已下载的deepspeech权重文件，指定video-fps=50
python extract_ds_features.py --input /home/sword/Music/Obama.wav --output ~/Desktop/001.ds.npy --deepspeech ~/Downloads/deepspeech-0_1_0-b90017e8.pb --`video-fps=50`

# a directory includes some wav files
python extract_ds_features.py --input /home/sword/Music
python extract_ds_features.py --input /home/sword/Music --deepspeech ~/Downloads/deepspeech-0_1_0-b90017e8.pb --video-fps=50
```
