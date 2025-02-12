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


HARD_WAV = download_asset("/Users/fiederlesje/git/sound_recognition/resources/harde_g/AUDIO-2025-02-11-19-23-31.wav")
ZACHT_WAV = download_asset("/Users/fiederlesje/git/sound_recognition/resources/zachte_g/AUDIO-2025-02-11-19-23-39.wav")


# Define transform
spectrogram = T.Spectrogram(n_fft=512)

# Perform transform
spec = spectrogram(HARD_WAV)

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=n_mels,
    mel_scale="htk",
)

melspec = mel_spectrogram(HARD_WAV)
plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")