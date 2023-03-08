from time import time

import torch
from torch import nn

from nn_models.RNNModelClass import RNNModelClass
from utils import update_plot_info


class GeneralRNNPytorch(RNNModelClass):
    pytorch_rnn_fn = {'RNN': nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, rnn_type: str, device: torch.device):
        """
        via rnn_type you can specify if you want an RNN, LSTM or GRU network
        """
        super(GeneralRNNPytorch, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.rnn = self.pytorch_rnn_fn[rnn_type](input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device
        self.to(device)

    def forward(self, input_tensor):
        h0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size, device=self.device)
        if self.rnn_type == 'LSTM':
            # for lstm we need to initialize the cell state as well
            c0 = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size)
            h0 = (h0, c0)

        out, _ = self.rnn(input_tensor.float(), h0)
        out = self.fc(out[:, -1, :])
        # since we use CrossEntropy we don't need softmax activation function
        return out

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
