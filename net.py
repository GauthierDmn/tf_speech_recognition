import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from pase.pase.models.frontend import wf_builder

device = torch.device("cuda") if config.cuda else torch.device("cpu")

pase = wf_builder("pase/cfg/PASE.cfg").to(device)
pase.eval()
pase.load_pretrained("pase/PASE.ckpt", load_last=True)

if config.freeze:
    for param in pase.parameters():
        param.requires_grad = False


def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 11)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x):
        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class PaseDNN(nn.Module):
    def __init__(self):
        super(PaseDNN, self).__init__()
        self.fc = nn.Linear(10000, 11)

    def forward(self, x):
        batch_zise = x.size(0)
        enc = pase(x)
        enc = enc.view(batch_zise, -1)
        out = self.fc(enc)

        return out


class PaseLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(PaseLSTM, self).__init__()
        self.enc = RNNEncoder(input_size=100,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=0.)
        self.fc = nn.Linear(10000, 11)

    def forward(self, x):
        batch_size = x.size(0)
        x = pase(x)
        x = self.enc(x)
        x = x.reshape(batch_size, -1)
        out = self.fc(x)

        return out
