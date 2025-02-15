import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

HARD_ZACHT = 'zacht'

if HARD_ZACHT == 'zacht':
    AUDIO_FILE = os.path.join('/Users/fiederlesje/git/sound_recognition/resources/audio_files/individu/fold1/gigantisch_001.wav')
    AUDIO_LABEL = 'zachte g'
else:
    AUDIO_FILE = os.path.join('/Users/fiederlesje/git/sound_recognition/resources/audio_files/individu/fold2/gigantisch_002.wav')
    AUDIO_LABEL = 'harde g'

#waveform, sample rate
#sr = 48000
#y, sr = librosa.load(SOFT_G_FILE, sr=48000)
y, sr = librosa.load(AUDIO_FILE)

print(y)
print(sr)

D = librosa.stft(y)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

#spectrogram
fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
ax.set(title=AUDIO_LABEL)
fig.colorbar(img, ax=ax, format="%+2.f dB")

#mel spectrogram
fig, ax = plt.subplots()
M = librosa.feature.melspectrogram(y=y, sr=sr)
M_db = librosa.power_to_db(M, ref=np.max)
img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax)
ax.set(title='Mel spectrogram display - ' + AUDIO_LABEL)
fig.colorbar(img, ax=ax, format="%+2.f dB")

#Chromagram
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
fig, ax = plt.subplots()
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
ax.set(title='Chromagram demonstration - ' + AUDIO_LABEL)
fig.colorbar(img, ax=ax)