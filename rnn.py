from time import time

import torch
from torch import nn

from utils import line_to_tensor


class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # since we use CrossEntropy we don't need softmax activation function
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def train_one_epoch(self, dataloader, tic, limit_time, criterion, optimizer):
        for _, (seq_batch, labels) in enumerate(dataloader):
            if time() - tic > limit_time:
                break
            hidden = self.init_hidden(seq_batch.size()[0])
            for i in range(seq_batch.size()[1]):
                output, hidden = self(seq_batch[:, i, :], hidden)

            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, sequences, labels, all_categories):
        with torch.no_grad():
            hidden = self.init_hidden(sequences.size()[0])
            for i in range(sequences.size()[1]):
                output, hidden = self(sequences[:, i, :], hidden)

            _, predictions = torch.max(output, 1)

            print(predictions, labels)
            for p, l in zip(predictions, labels):
                print(all_categories[p], all_categories[l])
            print(f"Percentage correct guesses: {(predictions == labels).float().mean():.3f}")
        return predictions
