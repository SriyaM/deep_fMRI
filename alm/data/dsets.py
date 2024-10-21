from typing import List
from torch.utils.data import Dataset
import torch
from os.path import join
import joblib
import os
import numpy as np

# get path to current file
path_to_file = os.path.dirname(os.path.abspath(__file__))
TEXT_MINI = '''In the quiet village of Willow Creek, nestled between rolling hills and whispering woods, lived an old gardener named Eli. With hands weathered by time and soil, Eli tended to his garden with a love so deep it made the flowers blush and the trees stand a bit taller. One spring morning, as the first rays of sunshine pierced the dewy air, Eli discovered a rare blue rose blooming among the sea of greens and colors—a rose he had heard of in stories but never believed to exist. This miraculous find became the talk of the village, drawing curious eyes and eager hearts to Eli's garden. But in his wisdom, Eli knew the rose wasn't meant for fame or fortune; it was a reminder from the earth, a symbol of the beauty and mystery that lies in the simplest moments of life. He cared for the blue rose, letting it thrive in its natural home, while continuing his daily rituals, teaching all who visited that the truest treasures are often hidden in plain sight, nurtured by patience and love.'''


class NextWordDataset(Dataset):
    '''This class is used to create a dataset for the next word prediction task.
    It returns tensors of token indexes (numbers) that can be decoded and inspected with the tokenizer.
    '''

    def __init__(self, tokens_file: str = None, raw_tokens: List = None, max_n_tokens=32):
        self.max_n_tokens = max_n_tokens
        if tokens_file is None and raw_tokens is None:
            raise ValueError(
                'Either tokens_file or raw_tokens must be provided')
        if raw_tokens is not None:
            self.tokens_ = raw_tokens
        else:
            self.tokens_ = joblib.load(tokens_file)

    def __len__(self):
        return len(self.tokens_) - self.max_n_tokens

    def __getitem__(self, idx):
        return torch.tensor(self.tokens_[idx: idx + self.max_n_tokens]), torch.tensor(self.tokens_[idx + self.max_n_tokens: idx + self.max_n_tokens + 1])
        # return torch.from_numpy(self.tokens_[idx: idx + self.max_n_tokens].astype(np.int64)), torch.from_numpy(self.tokens_[idx + self.max_n_tokens:idx + self.max_n_tokens+1].astype(np.int64))


class ALMDataset(Dataset):
    def __init__(self, tokens_file, max_n_tokens=128):
        if tokens_file is None:
            raise ValueError(
                'Either tokens_file or raw_tokens must be provided')
        else:
            self.tokens_ = joblib.load(tokens_file)
        self.max_n_tokens = max_n_tokens
        # self.device_type = device_type
    
    def __len__(self):
        return len(self.tokens_) - self.max_n_tokens

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens_[idx: idx + self.max_n_tokens])
        y = torch.tensor(self.tokens_[idx + 1 : idx + 1 + self.max_n_tokens])
        
        # if self.device_type == 'cuda':
        #     # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        #     x, y = x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)
        # else:
        #     x, y = x.to('cpu'), y.to('cpu')
        return x, y