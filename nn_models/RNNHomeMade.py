from time import time

import torch
from torch import nn

from nn_models.RNNModelClass import RNNModelClass
from utils import update_plot_info


class RNNHomeMade(RNNModelClass):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size: int, hidden_size: int, output_size: int = None, num_layers: int = 1, rnn_type: str = None, device: torch.device = 'cpu'):
        super(RNNHomeMade, self).__init__()

        if rnn_type != None:
            raise ValueError(f"{rnn_type=} but no value should be passed in input")
        self.hidden_size = hidden_size
        hidden_layers = [nn.Linear(input_size + hidden_size, hidden_size)]
        for l in range(num_layers - 1):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.i2h = nn.Sequential(*hidden_layers)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.to(device)
        self.device = device

    def forward(self, input_tensor):
        combined = None
        hidden_tensor = torch.zeros(input_tensor.size()[0], self.hidden_size).to(self.device)
        for i in range(input_tensor.size()[1]):
            combined = torch.cat((input_tensor[:, i, :], hidden_tensor), 1)
            hidden_tensor = self.i2h(combined)
        output = self.i2o(combined)
        # since we use CrossEntropy we don't need softmax activation function
        return output

    def train_one_epoch(self, dataloader, tic, limit_time, criterion, optimizer, plot_info):
        for (seq_batch, labels) in dataloader:
            if time() - tic > limit_time:
                break
            output = self(seq_batch)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_plot_info(plot_info, loss)
