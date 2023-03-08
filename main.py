from time import time

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from data_loader import LettersDataset
from nn_models.GeneralRNNPytorch import GeneralRNNPytorch
from nn_models.RNNHomeMade import RNNHomeMade
from nn_models.RNNModelClass import RNNModelClass
from utils import load_data, N_LETTERS
from utils import line_to_tensor

# fix the seed
torch.manual_seed(2)
np.random.seed(1)

# specify the device where we want to run computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parameters
rnn_type = 'HomeMade'  # type of RNN used. RNNModelClass.ALLOWED_TYPES are supported
pad_names = False  # set to True if you want to pad all names to have the same length (e.g. when using a batch_size>1, it might help in learning)
limit_time = 60
plot_steps = 20

# learning params
batch_size = 32
num_epochs = 5
criterion = nn.CrossEntropyLoss()
learning_rate = 0.005
# NN size
n_hidden = 32
num_layers = 1

# set up the data
nations, person_names = load_data()
if pad_names:
    max_seq = max([len(p) for p in person_names])
    person_names = np.array([p.rjust(max_seq, '_') for p in person_names])

# define train and test dataset
all_categories = np.unique(nations)
train_index = np.random.choice(len(person_names), int(len(person_names) * 0.95), False)
test_index = [k for k in range(len(person_names)) if k not in train_index]
db_train = LettersDataset(nations[train_index], person_names[train_index], names_are_padded=pad_names)
db_test = LettersDataset(nations[test_index], person_names[test_index], names_are_padded=pad_names)

dataloader_train = db_train.get_dataloader(batch_size=batch_size, shuffle=True, device=device)
dataloader_test = db_test.get_dataloader(batch_size=batch_size, shuffle=True, device=device)

# define the NN and the optimization problem
if rnn_type not in RNNModelClass.ALLOWED_TYPES:
    raise ValueError(f"{rnn_type=} not supported. Supported values are {RNNModelClass.ALLOWED_TYPES}")

if rnn_type == 'HomeMade':
    rnn = RNNHomeMade(N_LETTERS, n_hidden, len(all_categories), device=device)
else:
    rnn = GeneralRNNPytorch(N_LETTERS, n_hidden, num_layers, len(all_categories), rnn_type=rnn_type, device=device)

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# Train
tic = time()

# dictionary with all the info needed for plotting the loss over time while training
plot_info = {
    "current_loss": 0,
    "all_losses": [],
    "counter": 0,
    "plot_steps": plot_steps,
    "tic": tic
}
for k in range(num_epochs):
    print(f"Epoch {k}")
    if time() - tic > limit_time:
        break
    rnn.train_one_epoch(dataloader_train, tic, limit_time, criterion, optimizer, plot_info)

print(f"Training time: {time() - tic}")
plt.figure()
plt.plot(plot_info['all_losses'])
plt.show()

res = []
for _, (sequences, labels) in enumerate(dataloader_train):
    p = rnn.predict(sequences)
    res.append(p == labels)
print(f"Accuracy on training set: {torch.cat(res).float().mean():.3f}")

res = []
for _, (sequences, labels) in enumerate(dataloader_test):
    p = rnn.predict(sequences)
    res.append(p == labels)
print(f"Accuracy on test set: {torch.cat(res).float().mean():.3f}")

while True:
    sentence = input("Input:")
    if pad_names:
        sentence = sentence.rjust(max_seq, '_')
    line_tensor = line_to_tensor(sentence).to(device)
    if sentence == "quit":
        break

    output = rnn(line_tensor.view(-1, line_tensor.size(0), line_tensor.size(1)))
    _, pred = torch.max(output, 1)

    print(all_categories[pred])
