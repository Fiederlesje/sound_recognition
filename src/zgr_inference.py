import torch
import torchaudio

import zgr_subject_settings as zss
from zgr_cnn import CNNNetwork
from zgr_dataset import Zgr_Dataset
from zgr_train_model import SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "zachte_g",
    "harde_g"
]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


SUBJECT, MODEL_DIR = zss.model_location()
MODEL_FILE = f'{MODEL_DIR}cnnnet_{SUBJECT}.pth'

if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load(MODEL_FILE)
    cnn.load_state_dict(state_dict)

    # load zachte g dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    AUDIO_DIR, ANNOTATIONS_FILE = zss.file_location('test')
    SAMPLE_RATE, NUM_SAMPLES = zss.model_settings('test')

    zgt = Zgr_Dataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")

    tensor_pos = 0
    label_pos = 1
  
    for audio_data in zgt:
        # get a sample from the urban sound dataset for inference
        input, target = audio_data[tensor_pos], audio_data[label_pos] # [batch size, num_channels, fr, time]
        input.unsqueeze_(0)

        #print(input)
        #print(target)
        # make an inference
        predicted, expected = predict(cnn, input, target,
                                    class_mapping)
        print(f"Predicted: '{predicted}', expected: '{expected}'")
