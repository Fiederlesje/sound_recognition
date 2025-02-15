import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

import zgr_subject_settings as zss
from zgr_dataset import Zgr_Dataset
from zgr_cnn import CNNNetwork


BATCH_SIZE = 3
EPOCHS = 10
LEARNING_RATE = 0.001

AUDIO_DIR, ANNOTATIONS_FILE = zss.file_location('train')
SAMPLE_RATE, NUM_SAMPLES = zss.model_settings('train')


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        # n_fft, specifies number of bins for frequency, hoger nummer -> meer bins
        #2 x zoveel als voorbeeld, want sample rate is 2 x zo veel
        #!!!!
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    zgd = Zgr_Dataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    
    train_dataloader = create_data_loader(zgd, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    SUBJECT, MODEL_DIR = zss.model_location()
    MODEL_FILE = f'{MODEL_DIR}cnnnet_{SUBJECT}.pth'
    torch.save(cnn.state_dict(), MODEL_FILE)
    print(f'Trained feed forward net saved at {MODEL_FILE}')