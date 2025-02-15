import os
import torch 
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class Zachte_G_Dataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation,
                 target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        # [1,1,1] -> [1,1,1,0,0]
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # load audiofiles in dirs to dataset
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        # use gpu if possible
        signal = signal.to(self.device)
        #signal -> (num channels, samples) -> (1, 48000 * 1,35 sec = 64800 samples) 
        # audio to uniform sample rate
        signal = self._resample_if_necessary(signal, sr)
        # audio terugbrengen tot mono
        signal = self._mix_down_if_necessary(signal)
        # audio files vergelijkbare lengte maken voor vergelijking
        signal = self._cut_if_necessary(signal)
        # als na de cut minder samples zijn dan opgegeven, het geluidsfragment is korter dan 1,35 sec in ons geval, 
        # dan wordt deze aangevuld
        signal = self._right_pad_if_necessary(signal)
        # tensor waveform transformeren, in dit geval tot mel spectrogram
        signal = self.transformation(signal)

        return signal, label
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 3]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 4]



if __name__ == "__main__":
    AUDIO_DIR = '/Users/fiederlesje/git/sound_recognition/resources/audio_files/individu'
    ANNOTATIONS_FILE = '/Users/fiederlesje/git/sound_recognition/resources/annotations/individu/annotations_sound_recognition.csv'
    # langste sample = 1,35 sec, kortste sample is 0.98, sample rate = 48000, right padding tot 1,35 sec
    # 48000 * 1,35 s = 64800
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 64800

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        # n_fft, specifies number of bins for frequency, hoger nummer -> meer bins
        #2 x zoveel als voorbeeld, want sample rate is 2 x zo veel
        #!!!!
        n_fft=2048,
        hop_length=1024,
        n_mels=128
    )

    #ms = mel_spectrogram(signal)

    zgd = Zachte_G_Dataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    print(f"There are {len(zgd)} samples in the dataset.")
    signal, label = zgd[0]
    print(signal)
    print(label)

