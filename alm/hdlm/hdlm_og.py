from collections import defaultdict
import torchhd
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from os.path import join
import joblib
import numpy as np
from typing import List, Dict
import numpy as np


class HDLM:
    def __init__(
        self,
        tokenizer_checkpoint='gpt2',
        emb_size=10000,
        learning_rate=0.1,
        context_length=5,
        similarity_function='cosine',
        device='cuda',
        random_state=42,
    ):

        # set parameters
        self.learning_rate = learning_rate
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.emb_size = emb_size
        self.context_length = context_length
        self.similarity_function = similarity_function
        self.device = device
        self.random_state = random_state

        # initialize model parameters
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.tokenizer_ = AutoTokenizer.from_pretrained(
            self.tokenizer_checkpoint)
        self.vocab_size_ = len(self.tokenizer_)
        self.vocab_ = torch.rand((self.vocab_size_, self.emb_size)).to(
            self.device)  # uniform
        # self.vocab_ = torch.load("/home/t-ziyangwu/alt-lm-gpt/gpt2xl_embed.pt",map_location=torch.device('cpu')).to(self.device)
        # self._normalize_vocab()
        # self.vocab[self.vocab < 0.1] = 0  # binarize with threshold
        # self.vocab[self.vocab >= 0.1] = 1
        self.positional_vectors_ = torch.Tensor(torchhd.level(
            self.context_length, self.emb_size)).to(self.device)  # context_length x emb_size

    def fit_and_calc_perplexity(
        self, dset, fit=False, n_examples=None,
            seed=None, eval_perfect_match=False) -> Dict[str, float]:
        '''Calculates perplexity over the dataset and fits the model if fit=True.
        Perplexity is calculated as the exponential of the mean negative log-probabilities of the next token.
        Lower is better.

        Params
        ------
        dset: pytorch Dataset
            Dataset to calculate perplexity over
        fit: bool
            Whether to fit the model
        n_examples: int
            Number of examples to calculate perplexity over
        seed: int
            Random seed for sampling examples
        eval_perfect_match: bool
            Whether to evaluate the model on perfect match (i.e. does the model predict the next token perfectly?)
        '''
        # subselect the dataset
        if n_examples is None:
            n_examples = len(dset)
        if seed is None:
            example_nums = np.arange(n_examples)
        else:
            rng = np.random.default_rng(seed)
            example_nums = rng.choice(
                np.arange(len(dset)), size=n_examples, replace=False)

        # initialize perplexity calculation
        ans_dict = defaultdict(list)
        for ex_num in example_nums:
            # get a data example
            token_ids, next_token_id = dset[ex_num]

            # pad inputs
            token_ids = self._pad_inputs(token_ids.squeeze())

            # retrieve embeddings for each input by id
            token_embs = self._retrieve_embs(token_ids)

            # predict next token embedding
            next_token_emb = self._predict_next_emb_from_embs(token_embs)

            # calculate and store the next-token probabilities using the embedding
            next_token_probs = self._emb_to_token_probs(next_token_emb)
            ans_dict['next_token_probs'].append(
                next_token_probs[next_token_id].item())
            if eval_perfect_match:
                # check the rank of the correct next token (1 is best)
                ans_dict['next_token_rank'].append(
                    torch.argsort(next_token_probs, descending=True).tolist().index(next_token_id) + 1)

            # update the vocab embedding
            if fit:
                self._update_vocab_emb(next_token_emb, next_token_id)

        if fit:
            # normalize vocab
            self.vocab_ /= torch.norm(self.vocab_, dim=1).unsqueeze(1)

        # process outputs
        for k in ['next_token_probs', 'next_token_rank']:
            if k in ans_dict:
                ans_dict[k] = np.array(ans_dict[k])

        return ans_dict
    
    def fit_and_calc_perplexity_fixed(
        self, token_ids_list, next_token_id_list, fit=False, n_examples=None,
            seed=None, eval_perfect_match=False) -> Dict[str, float]:
        '''Calculates perplexity over the dataset and fits the model if fit=True.
        Perplexity is calculated as the exponential of the mean negative log-probabilities of the next token.
        Lower is better.

        Params
        ------
        dset: pytorch Dataset
            Dataset to calculate perplexity over
        fit: bool
            Whether to fit the model
        n_examples: int
            Number of examples to calculate perplexity over
        seed: int
            Random seed for sampling examples
        eval_perfect_match: bool
            Whether to evaluate the model on perfect match (i.e. does the model predict the next token perfectly?)
        '''
        # initialize perplexity calculation
        ans_dict = defaultdict(list)
        example_nums = np.arange(token_ids_list.shape[0])
        for ex_num in example_nums:
            # get a data example
            token_ids, next_token_id = token_ids_list[ex_num], next_token_id_list[ex_num]

            # pad inputs
            token_ids = self._pad_inputs(token_ids.squeeze())

            # retrieve embeddings for each input by id
            token_embs = self._retrieve_embs(token_ids)
            
            # predict next token embedding
            next_token_emb = self._predict_next_emb_from_embs(token_embs)
            
            # calculate and store the next-token probabilities using the embedding
            next_token_probs = self._emb_to_token_probs(next_token_emb)
            ans_dict['next_token_probs'].append(
                next_token_probs[next_token_id].item())
            if eval_perfect_match:
                # check the rank of the correct next token (1 is best)
                ans_dict['next_token_rank'].append(
                    torch.argsort(next_token_probs, descending=True).tolist().index(next_token_id) + 1)

            # update the vocab embedding
            if fit:
                self._update_vocab_emb(next_token_emb, next_token_id)

        if fit:
            # normalize vocab
            self.vocab_ /= torch.norm(self.vocab_, dim=1).unsqueeze(1)

        # process outputs
        for k in ['next_token_probs', 'next_token_rank']:
            if k in ans_dict:
                ans_dict[k] = np.array(ans_dict[k])

        return ans_dict

    def _pad_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        '''Ensure input size matches context_length
        '''
        input_length = input_ids.numel()
        if input_length == self.context_length:
            return input_ids

        # inputs too long (left truncate)
        elif input_length > self.context_length:
            return input_ids[-self.context_length:]

        # inputs too short (left pad)
        elif input_length < self.context_length:
            return torch.cat((
                torch.zeros(self.context_length -
                            input_ids.numel(), dtype=torch.int),
                input_ids
            )).unsqueeze(0)

    def _retrieve_embs(self, input_ids):
        '''Returns array of shape (len(input_ids), emb_size)
        '''
        if isinstance(input_ids, int) or isinstance(input_ids, float):
            input_ids = [input_ids]
        return torch.vstack([self.vocab_[i] for i in input_ids])

    def _predict_next_emb_from_embs(self, token_embs: torch.Tensor) -> torch.Tensor:
        '''All the inductive bias comes from this function
        Returns next emb (emb_size)
        '''
        # TODO: might want to keep track of what was padded/masked by previous function to zero-out padded embeddings?

        # multiply with positional vectors (context_length, emb_size) then take mean
        token_embs = token_embs * self.positional_vectors_  # elementwise multiplication

        # apply fixed nonlinearity?
        # token_embs = torch.relu(token_embs)

        # aggregate embs
        next_emb = torch.mean(token_embs, dim=0).squeeze()
        # next_emb = torch.max(token_embs, dim=0).squeeze()

        return next_emb

    def _emb_to_token_probs(self, token_emb):
        '''Returns token probabilities
        '''
        if self.similarity_function == 'cosine':
            return torch.softmax(torch.matmul(self.vocab_, token_emb), dim=0)
        elif self.similarity_function == 'euclidean':
            return torch.softmax(-torch.norm(self.vocab_ - token_emb, dim=1), dim=0)

    def _update_vocab_emb(self, predicted_emb, next_token_correct_id):
        '''This is where all the parameter updating actually happens
        '''
        emb = self.vocab_[next_token_correct_id]
        self.vocab_[next_token_correct_id] = (
            1 - self.learning_rate) * emb + self.learning_rate * predicted_emb
