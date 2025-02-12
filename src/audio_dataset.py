import os

from torch.utils.data import Dataset
import pandas as pd
import torchaudio

# audio files inlezen in de dataset
class Zachte_G_Dataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 3]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 4]


if __name__ == "__main__":
    AUDIO_DIR = '/Users/fiederlesje/git/sound_recognition/resources/audio_files'
    ANNOTATIONS_FILE = '/Users/fiederlesje/git/sound_recognition/resources/annotations/annotations_sound_recognition.csv'
    
    usd = Zachte_G_Dataset(ANNOTATIONS_FILE, AUDIO_DIR)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
    print(signal)
    print(label)

