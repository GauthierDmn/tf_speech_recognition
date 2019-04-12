import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import logging

from net import VGG, PASE
import config
from data_loader import AudioDataset, PaseDataset
from preprocess import save_checkpoint

logger_module_name = 'train'
logger = logging.getLogger('step_plus.' + logger_module_name)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.propagate = False


# Hyper Parameters
num_epochs = 5
batch_size = 64
valid_size = 0.1
learning_rate = 0.0001
cuda = True
is_pase = True

device = torch.device("cuda") if cuda else torch.device("cpu")

# Define a path to save experiment logs
experiment_path = "output/{}".format(config.exp)
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

# Open labels dataframe
dataframe = pd.read_csv(os.path.join(config.train_labels_path, "train_labels.csv"))

print("Set of labels is:", set(dataframe["label"].tolist()))

print("Creating dataset...")
if is_pase:
    audio_dataset = PaseDataset(dataframe)
else:
    audio_dataset = AudioDataset(dataframe)
print("Dataset sucessfully loaded!")

# Define a split for train/valid
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
print("Length of valid data loader is:", len(train_dataloader))

print("Loading model...")
if is_pase:
    model = PASE()
else:
    model = VGG('VGG11')
model.to(device)
print("Model successfully loaded!")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)

# Train the Model
losses = []
best_valid_loss = 100
for epoch in range(num_epochs):
    model.train()
    print("##### epoch {:2d}".format(epoch))
    for i, batch in enumerate(train_dataloader):
        x = batch[0].unsqueeze(1).float().to(device)
        score = batch[1].long().to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, score)
        loss.backward()
        optimizer.step()

        #if (i + 1) % 1 == 0:
        #    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
        #          % (epoch + 1, num_epochs, i + 1, len(train_dataloader), loss.item()))

    model.eval()
    valid_losses = 0
    acc = 0
    n_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            batch_dim = len(batch[1])
            x = batch[0].unsqueeze(1).float().to(device)
            score = batch[1].long().to(device)
            pred = model(x)
            loss = criterion(pred, score)
            valid_losses += loss.item()
            if cuda:
                preds = np.argmax(pred.cpu().numpy(), axis=1)
                acc += sum([1 if p == y else 0 for p, y in zip(preds, score.cpu().numpy())])
            else:
                preds = np.argmax(pred.numpy(), axis=1)
                acc += sum([1 if p == y else 0 for p, y in zip(preds, score.numpy())])
            n_samples += len(preds)

        print('Loss on validation is:', np.round(valid_losses / len(valid_dataloader), 2))
        print('Accuracy on validation is:', np.round(100 * acc / n_samples, 2), "%")

        # save last model weights
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_valid_loss": np.round(valid_losses / len(valid_dataloader), 2)
        }, True, os.path.join(experiment_path, "model_last_checkpoint.pkl"))

        # save model with best validation error
        is_best = bool(np.round(valid_losses / len(valid_dataloader), 2) < best_valid_loss)
        best_valid_loss = min(np.round(valid_losses / len(valid_dataloader), 2), best_valid_loss)
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_valid_loss": best_valid_loss
        }, is_best, os.path.join(experiment_path, "model.pkl"))
