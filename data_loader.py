import torch
import torch.utils.data as data
import librosa
import numpy as np
from preprocess import mel_spectrogram, resample_audio_file


class AudioDataset(data.Dataset):
    """Custom Dataset for the TF Speech Recognition data compatible with torch.utils.data.DataLoader."""

    def __init__(self, dataframe):
        """Set the dataframe where to find audio file paths and labels.
        """
        self.dataframe = dataframe

    def __getitem__(self, index):
        """Returns one data pair (spect and label)."""
        audio_file_name = self.dataframe["path"][index]
        spect = torch.from_numpy(mel_spectrogram(audio_file_name)).float()
        label = self.dataframe["label"][index]
        return spect, label

    def __len__(self):
        return len(self.dataframe)


class PaseDataset(data.Dataset):
    """Custom Dataset for the TF Speech Recognition data compatible with torch.utils.data.DataLoader."""

    def __init__(self, dataframe):
        """Set the dataframe where to find audio file paths and labels.
        """
        self.dataframe = dataframe

    def __getitem__(self, index):
        """Returns one data pair (spect and label)."""
        audio_file_name = self.dataframe["path"][index]
        wav, sr = librosa.load(audio_file_name)
        if sr != 16000:
            wav, sr = resample_audio_file(wav, sr)
        assert sr == 16000
        padded_wav = np.zeros(16000)
        if len(wav) > 16000:
            padded_wav = wav[:16000]
        else:
            padded_wav[:len(wav)] = wav
        label = self.dataframe["label"][index]
        return padded_wav, label

    def __len__(self):
        return len(self.dataframe)
