import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from net import VGG
from config import train_labels_path
from data_loader import AudioDataset
import os
import logging

logger_module_name = 'train'
logger = logging.getLogger('step_plus.' + logger_module_name)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.propagate = False


# Hyper Parameters
num_epochs = 5
batch_size = 128
valid_size = 0.1
learning_rate = 0.0001
cuda = False

# Open labels dataframe
dataframe = pd.read_csv(os.path.join(train_labels_path, "train_labels.csv"))

print("Set of labels is:", set(dataframe["label"].tolist()))

print("Creating dataset...")
audio_dataset = AudioDataset(dataframe)
print("Dataset sucessfully loaded!")

#Define a split for train/valid
num_train = len(dataframe)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

print("Loading dataloader...")
train_dataloader = DataLoader(audio_dataset,
                        shuffle=False,
                        batch_size=batch_size,
                        sampler=train_sampler)

valid_dataloader = DataLoader(audio_dataset,
                        shuffle=False,
                        batch_size=batch_size,
                        sampler=valid_sampler)
print("Dataloader sucessfully loaded!")

print("Length of training data loader is:", len(train_dataloader))

print("Loading model...")
model = VGG('VGG11')

if cuda:
    model = model.cuda()
print("Model successfully loaded!")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)

# train
losses = []

# Train the Model
for epoch in range(num_epochs):
    model.train()
    print("##### epoch {:2d}".format(epoch))
    for i, batch in enumerate(train_dataloader):
        spect = autograd.Variable(batch[0]).unsqueeze(1)
        score = autograd.Variable(batch[1]).long()
        if cuda:
            spect, score = spect.cuda(), score.cuda()
        optimizer.zero_grad()
        pred = model(spect)
        loss = criterion(pred, score)
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataloader), loss.data[0]))

    model.eval()
    valid_losses = 0
    acc = 0
    n_samples = 0
    for i, batch in enumerate(valid_dataloader):
        batch_dim = len(batch[1])
        spect = autograd.Variable(batch[0], volatile=True).unsqueeze(1)
        score = autograd.Variable(batch[1], volatile=True).long()
        if cuda:
            spect, score = spect.cuda(), score.cuda()
        pred = model(spect)
        loss = criterion(pred.view(batch_dim,3), score)
        valid_losses += loss.data[0]
        preds = np.argmax(pred.data.numpy(), axis=1)
        acc += sum([1 if p == y else 0 for p, y in zip(preds, score.data.numpy())])
        n_samples += len(preds)

    print('Loss on validation is:', np.round(valid_losses / len(valid_dataloader), 2))
    print('Accuracy on validation is:', np.round(100 *acc / n_samples, 2), "%")

# Save the Trained Model
torch.save(model.state_dict(), 'tf_model.pkl')