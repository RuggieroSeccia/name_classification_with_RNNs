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
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def train_rnn(self,
                  line_tensor,
                  category_tensor,
                  criterion,
                  optimizer):
        """ train on one sample"""
        hidden = self.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return output, loss.item()

    def predict(self, input_line, all_categories):
        print(f"\n> {input_line}")
        with torch.no_grad():
            line_tensor = line_to_tensor(input_line)

            hidden = self.init_hidden()

            for i in range(line_tensor.size()[0]):
                output, hidden = self(line_tensor[i], hidden)

            guess = category_from_output(output, all_categories)
            print(guess)

def category_from_output(output, all_categories):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]


