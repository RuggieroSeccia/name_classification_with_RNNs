import string
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils import line_to_tensor


class LettersDataset(Dataset):

    def __init__(self, nations: List[str], person_names: List[str], names_are_padded: bool):
        """

        :param nations: list with all the nations
        :param person_names: list with all the names
        :param names_are_padded: set to True if the input data are already padded
        """
        self.all_categories = np.unique(nations)
        self.nations, self.person_names = nations, person_names
        self.num_samples = len(person_names)
        self.names_are_padded = names_are_padded

    def __getitem__(self, index: int):
        nation_i, person_name_i = self.nations[index], self.person_names[index]
        label = torch.tensor([np.argmax(self.all_categories == nation_i)], dtype=torch.long)
        line_tensor = line_to_tensor(person_name_i)

        return line_tensor, label

    def __len__(self):
        return self.num_samples

    def get_dataloader(self, batch_size, shuffle, device):
        def collate_fn(data):
            '''
            We should build a custom collate_fn rather than using default collate_fn,
            as the size of every sentence is different and merging sequences (including padding)
            is not supported in default.
            Args:
                data: list of tuple (sequence, label)
            Return:
                padded_seq - Padded Sequence, tensor of shape (batch_size, padded_length)
                length - Original length of each sequence(without padding), tensor of shape(batch_size)
                label - tensor of shape (batch_size)
            '''

            sequences, label = zip(*data)
            if self.names_are_padded or len(sequences) == 1:
                return torch.stack(sequences).to(device), torch.tensor(label, device=device)
            # if batches are not padded, or we have more than one sample, we pad the sequences
            length = torch.tensor([len(seq) for seq in sequences])
            labels = torch.tensor(label, device=device)
            padded_seq = torch.zeros(len(sequences),
                                     max(length),
                                     len(sequences[0][1])).long()
            for i, seq in enumerate(sequences):
                end = length[i]
                padded_seq[i, :end] = seq
            return padded_seq.to(device), labels

        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
