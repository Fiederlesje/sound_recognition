import torch
import torchaudio

import io
import os
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset

import torchaudio.functional as F
import torchaudio.transforms as T

HARD_ZACHT = 'hard'

if HARD_ZACHT == 'zacht':
    AUDIO_FILE = os.path.join('/Users/fiederlesje/git/sound_recognition/resources/audio_files/individu/fold1/gigantisch_001.wav')
    AUDIO_LABEL = 'zachte g'
else:
    AUDIO_FILE = os.path.join('/Users/fiederlesje/git/sound_recognition/resources/audio_files/individu/fold2/gigantisch_002.wav')
    AUDIO_LABEL = 'harde g'

metadata = torchaudio.info(AUDIO_FILE)
print(metadata)

waveform, sample_rate = torchaudio.load(AUDIO_FILE)

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform - " + AUDIO_LABEL)

def plot_specgram(waveform, sample_rate, title="Spectrogram - " + AUDIO_LABEL):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)


plot_waveform(waveform, sample_rate)
plot_specgram(waveform, sample_rate)




