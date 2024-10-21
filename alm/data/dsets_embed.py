from typing import List
from torch.utils.data import Dataset
import torch
from os.path import join
import joblib
import os
import numpy as np
from datasets import load_dataset
from numpy.lib.stride_tricks import sliding_window_view

class SentenceDataset(Dataset):
    '''This class is used to create a dataset for the next word prediction task.
    It returns tensors of token indexes (numbers) that can be decoded and inspected with the tokenizer.
    '''

    def __init__(self, data_dir: str, max_n_tokens: int, min_n_tokens: int, split: str = 'train', pad_token_id: int = 50256):
        self.max_n_tokens = max_n_tokens
        self.split = split
        self.pad_token_id = pad_token_id
        self.tokens_ = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        if os.path.exists(os.path.join(data_dir, f'{split}_selected_indices_{min_n_tokens}-{max_n_tokens}.npy')):
            selected_indices = np.load(os.path.join(data_dir, f'{split}_selected_indices_{min_n_tokens}-{max_n_tokens}.npy'))
        else:
            pad_index = (self.tokens_ == self.pad_token_id)
            window = sliding_window_view(pad_index, self.max_n_tokens)
            # if split == 'all':
            #     non_exist_pad = ~window.any(1)
            #     next_pad_indices = np.concatenate([np.array([True]), window[:-1, 0]])
            #     selected_indices = np.stack([non_exist_pad, next_pad_indices], axis=1).any(1)
            # else:
            selected_indices = ~window[:, :min_n_tokens].any(1)
            np.save(os.path.join(data_dir, f'{split}_selected_indices_{min_n_tokens}-{max_n_tokens}.npy'), selected_indices)
        self.idx2newidx = np.arange(len(selected_indices))[selected_indices]

    def __len__(self):
        # if self.split == 'train':
        #     return len(self.tokens_) - self.max_n_tokens
        return len(self.idx2newidx)

    def __getitem__(self, idx):
        # if self.split == 'train':
        #     tokens = torch.tensor(self.tokens_[idx: idx + self.max_n_tokens]).long()
        new_idx = self.idx2newidx[idx]
        tokens = torch.tensor(self.tokens_[new_idx: new_idx + self.max_n_tokens]).long()
        if tokens.max() == self.pad_token_id:
            tokens[tokens.argmax():] = self.pad_token_id
        return new_idx, tokens

class SentenceSlidingWindowDataset(Dataset):
    '''This class is used to create a dataset for the next word prediction task.
    It returns tensors of token indexes (numbers) that can be decoded and inspected with the tokenizer.
    '''

    def __init__(self, data_dir: str, max_n_tokens: int, min_n_tokens: int, window_n_tokens: int, split: str = 'train', pad_token_id: int = 50256):
        self.max_n_tokens = max_n_tokens
        self.window_n_tokens = window_n_tokens
        self.split = split
        self.pad_token_id = pad_token_id
        self.tokens_ = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        if os.path.exists(os.path.join(data_dir, f'{split}_selected_indices_{min_n_tokens+1}-{max_n_tokens+1}.npy')):
            selected_indices = np.load(os.path.join(data_dir, f'{split}_selected_indices_{min_n_tokens+1}-{max_n_tokens+1}.npy'))
        else:
            pad_index = (self.tokens_ == self.pad_token_id)
            window = sliding_window_view(pad_index, self.max_n_tokens+1)
            # if split == 'all':
            #     non_exist_pad = ~window.any(1)
            #     next_pad_indices = np.concatenate([np.array([True]), window[:-1, 0]])
            #     selected_indices = np.stack([non_exist_pad, next_pad_indices], axis=1).any(1)
            # else:
            selected_indices = ~window[:, :min_n_tokens].any(1)
            np.save(os.path.join(data_dir, f'{split}_selected_indices_{min_n_tokens+1}-{max_n_tokens+1}.npy'), selected_indices)
        self.idx2newidx = np.arange(len(selected_indices))[selected_indices]

    def __len__(self):
        return len(self.idx2newidx)

    def __getitem__(self, idx):
        new_idx = self.idx2newidx[idx]
        tokens = torch.tensor(self.tokens_[new_idx: new_idx + self.max_n_tokens + 1]).long()
        next_token = tokens[-1]
        tokens = tokens[:-1]
        if tokens.max() == self.pad_token_id:
            tokens[tokens.argmax():] = self.pad_token_id
        all_tokens_window = torch.Tensor(sliding_window_view(tokens.numpy(), self.window_n_tokens)).long()
        return tokens, all_tokens_window, next_token

