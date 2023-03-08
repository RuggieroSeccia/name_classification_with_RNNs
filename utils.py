# data: https://download.pytorch.org/tutorial/data.zip
import io
import os
from time import time

import numpy as np
import unicodedata
import string
import glob

import torch
import pandas as pd

# alphabet small + capital letters + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


def load_data():
    # Build the category_lines dictionary, a list of names per language
    def unicode_to_ascii(s):
        # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
        return ''.join(
            c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)

    def find_files(path):
        return glob.glob(path)

    # Read a file and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]

    category_lines = {}
    all_categories = []
    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)

        lines = read_lines(filename)
        category_lines[category] = lines

    df = pd.DataFrame([category_lines]).T
    df = df.explode(0)
    df = df.reset_index(drop=False).rename(columns={'index': 'nationality', 0: 'person_name'})
    df = df.sample(frac=1)
    nations, person_names = df.nationality.values, df.person_name.values
    return nations, person_names


def line_to_tensor(line):
    def letter_to_index(letter):
        # Find letter index from all_letters, e.g. "a" = 0
        return ALL_LETTERS.find(letter)

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    tensor = torch.zeros(len(line), N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][letter_to_index(letter)] = 1
    return tensor


def update_plot_info(plot_info, loss):
    plot_info['current_loss'] += loss
    plot_info['counter'] += 1
    if (plot_info['counter']) % plot_info['plot_steps'] == 0:
        plot_info['all_losses'].append(plot_info['current_loss'] / plot_info['plot_steps'])
        print(f"Running time: {time() - plot_info['tic']:.2f}, "
              f"Current loss {plot_info['current_loss'] / plot_info['plot_steps']:.3f}")
        plot_info['current_loss'] = 0
