from time import time

import torch
from torch import nn
import numpy as np

from data_loader import letters_dataset
from rnn import RNN
from utils import load_data, N_LETTERS

nations, person_names = load_data()
all_categories = np.unique(nations)
train_index = np.random.choice(len(person_names), int(len(person_names)*0.75), False)
test_index = [k for k in range(len(person_names)) if k not in train_index]
db_train = letters_dataset(nations[train_index], person_names[train_index])
db_test = letters_dataset(nations[test_index], person_names[test_index])
batch_size = 32
num_epochs = 100
limit_time = 5

dataloader_train = db_train.get_dataloader(batch_size=batch_size, shuffle=True)
dataloader_test = db_train.get_dataloader(batch_size=batch_size, shuffle=True)

n_hidden = 32

rnn = RNN(N_LETTERS, n_hidden, len(all_categories))

criterion = nn.CrossEntropyLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

tic = time()
loss = None
output = None
for k in range(num_epochs):
    print(f"epoch {k}: loss {loss}")
    if time() - tic > limit_time:
        break
    loss = rnn.train_one_epoch(dataloader_train, tic, limit_time, criterion, optimizer)

print(f"Training time: {time() - tic}")
res = []
for _, (sequences, labels) in enumerate(dataloader_test):
    p = rnn.predict(sequences, labels, all_categories)
    res.append(p == labels)
print(torch.cat(res).float().mean())
