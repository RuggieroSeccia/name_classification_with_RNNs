import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from nn_utils import RNN, category_from_output
from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example

category_lines, all_categories = load_data()
n_categories = len(all_categories)

n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)

    output, loss = rnn.train_rnn(line_tensor, category_tensor, criterion, optimizer)
    current_loss += loss

    if (i + 1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (i + 1) % print_steps == 0:
        guess = category_from_output(output, all_categories)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i + 1} {(i + 1) / n_iters * 100} {loss:.4f} {line} / {guess} {correct}")

plt.figure()
plt.plot(all_losses)
plt.show()

while True:
    sentence = input("Input:")
    if sentence == "quit":
        break

    rnn.predict(sentence, all_categories)
