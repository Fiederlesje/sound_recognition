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

HARD_G_FILE = os.path.join('/Users/fiederlesje/git/sound_recognition/resources', 'harde_g', 'AUDIO-2025-02-11-19-23-31.wav')
SOFT_G_FILE = os.path.join('/Users/fiederlesje/git/sound_recognition/resources', 'zachte_g', 'AUDIO-2025-02-11-19-23-39.wav')

metadata = torchaudio.info(HARD_G_FILE)
print(metadata)
metadata = torchaudio.info(SOFT_G_FILE)
print(metadata)


waveform, sample_rate = torchaudio.load(HARD_G_FILE)

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
    figure.suptitle("waveform")

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
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




