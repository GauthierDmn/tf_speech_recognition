import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import librosa
from scipy.io import wavfile
from scipy import signal
from augmentation import data_transformer
from config import train_audio_path, train_labels_path

list_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']


def create_labels_csv(train_audio_path, train_labels_path):
    #List the sub-folders
    sub_folders = []
    for file in os.listdir(train_audio_path):
        if os.path.isdir(os.path.join(train_audio_path, file)):
            sub_folders.append(file)
    print(sub_folders)

    columns = ['audio', 'label-str','path']
    df_pred = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)

    for i, label in enumerate(sub_folders):
        # get all the wave files
        all_files = [y for y in os.listdir(os.path.join(train_audio_path, label)) if '.wav' in y]
        for file in all_files:
            path = os.path.join(train_audio_path, label, file)
            if label in list_labels:
                df_pred = df_pred.append({'audio':file, 'label-str':label,'path':path}, ignore_index=True)
            else:
                df_pred = df_pred.append({'audio': file, 'label-str': 'unknown', 'path': path}, ignore_index=True)


    #Encode the categorical labels as numeric data
    df_pred['label'] = LabelEncoder().fit_transform(df_pred['label-str'])

    #Save dataframe as .csv file
    df_pred.to_csv(os.path.join(train_labels_path, "train_labels.csv"), index=None)


def resample_audio_file(samples, sample_rate, new_sample_rate=16000):
    if sample_rate != 16000:
        samples = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
    return samples, new_sample_rate


def mel_spectrogram(audio_filename, resampled=True, max_len=1, normalize=True, augmentation=True):
    sample_rate, samples = wavfile.read(audio_filename)

    if augmentation:
        samples = data_transformer(samples)

    if samples.shape[0] < sample_rate * max_len:
        reshaped_samples = np.zeros((sample_rate * max_len,))
        reshaped_samples[:samples.shape[0]] = samples
    else:
        reshaped_samples = samples[:(max_len * sample_rate)]

    if resampled:
        reshaped_samples, sample_rate = resample_audio_file(reshaped_samples, sample_rate)

    S = librosa.feature.melspectrogram(reshaped_samples, sr=sample_rate, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    # z-score normalization
    if normalize:
        mean = log_S.mean()
        std = log_S.std()
        if std != 0:
            log_S -= mean
            log_S /= std

    return log_S.T


if __name__ == "__main__":
    create_labels_csv(train_audio_path, train_labels_path)