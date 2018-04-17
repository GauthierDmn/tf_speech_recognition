import torch
import torch.utils.data as data
from preprocess import mel_spectrogram


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