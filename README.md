# tf_speech_recognition

This is my code for the Kaggle Challenge TensorFlow Speech Recognition.

# Dataset

The data provided by TensorFlow consists in 65,000 one-second long utterances of 30 short words, by thousands of different people.

In this competition, we wer challenged to use the Speech Commands Dataset to build an algorithm that understands simple spoken commands. 

These are the labels to detect: yes, no, up, down, left, right, on, off, stop, go, unknown, silence.

# Preprocessing

After downloading the training data at https://www.kaggle.com/c/7634/download/train.7z, running the command `python preprocess.py` will create a CSV file containing the name of each audio file, together with its associated lable (not only as a string but also encoded as an integer).

Then, I decided to use mel spectrograms as a representation of the WAV files. They are often used in Speech Processing tasks, outperforming algorithms applied directly on the wave form.

Finally, some data augmentation was applied to make the algorithm more robust to noise and silence.

![alt text](https://github.com/GauthierDmn/tf_speech_recognition/blob/master/wav_mel_spec.png)

# Algorithm

I used a VGG-like algorithm, which demonstrated great results in image processing.

I also tried other methods such as LSTM, CNN+LSTM, but it did not improved the results on this specific task.

Once the data was downloaded and the CSV file with labels created, you can run the command `python train.py` to train the algorithm.

If you possess a GPU and CUDA, you should allow CUDA in the training script to accelerate the training time.
