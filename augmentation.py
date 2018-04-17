import cv2
import numpy as np
from scipy.io import wavfile


def random_speed(wav, u=0.5):
    if np.random.random() < u:
        speed_rate = np.random.uniform(0.7, 1.3)
        wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
    else:
        wav_speed_tune = wav
    return wav_speed_tune


def random_noise(wav, u=0.5):
    # https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/46839
    # print('noiza', np.random.random())
    if np.random.random() < u:
        if len(wav) != 16000:
            new_wav = np.zeros(16000)
            new_wav[:len(wav)] = wav.copy()
            wav = new_wav

        i_n = np.random.random_integers(0, len(config.noise_files) - 1)

        samplerate, bg_wav = wavfile.read(config.noise_files[i_n])

        start_ = np.random.random_integers(bg_wav.shape[0] - 16000)
        bg_slice = np.array(bg_wav[start_: start_ + 16000]).astype(float)
        wav = wav.astype(float)

        # get energy
        noise_energy = float(np.sqrt(bg_slice.dot(bg_slice) / float(bg_slice.size)))
        data_energy = float(np.sqrt(wav.dot(wav) / float(wav.size)))
        coef = (data_energy / noise_energy)
        wav_with_bg = wav * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.4) * coef
        res = wav_with_bg
    return res


def random_shift(wav, u=0.5):
    if np.random.random() < u:
        start_ = int(np.random.uniform(-4800, 4800))
        if start_ >= 0:
            wav_time_shift = np.r_[wav[start_:], np.random.uniform(-0.001, 0.001, start_)]
        else:
            wav_time_shift = np.r_[np.random.uniform(-0.001, 0.001, -start_), wav[:start_]]
    else:
        wav_time_shift = wav

    return wav_time_shift


def data_transformer(wav, u=0.5):
    wav = random_shift(wav, u=u)
    wav = random_speed(wav, u=u)

    return wav