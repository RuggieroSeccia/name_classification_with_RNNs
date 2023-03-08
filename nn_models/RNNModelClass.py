from time import time

import torch
from torch import nn


class RNNModelClass(nn.Module):
    """Interface for RNN models"""
    ALLOWED_TYPES = ['HomeMade', 'RNN', 'LSTM', 'GRU']
    def __init__(self):
        super(RNNModelClass, self).__init__()
        pass

    def forward(self, input_tensor):
        pass

    def train_one_epoch(self, dataloader, tic, limit_time, criterion, optimizer,counter):
        pass

    def predict(self, sequences):
        with torch.no_grad():
            output = self(sequences)

            _, predictions = torch.max(output, 1)

            return predictions