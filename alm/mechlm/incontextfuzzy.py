import os
from collections import defaultdict
import torchhd
from transformers import AutoTokenizer, AutoModelForCausalLM
from infini_gram.engine import InfiniGramEngine
from tqdm import tqdm
import torch
from os.path import join
import joblib
import numpy as np
from typing import List, Dict
import requests
import math
from scipy.sparse import coo_array
import itertools
import torch.nn.functional as F

import alm.config
from numpy.lib.stride_tricks import sliding_window_view

from .build_infinigram import InfiniGramModel
from .mini_gpt import GPTConfig, GPT


def get_sparse_array_from_result(result_by_token_id, vocab_size, key='cont_cnt'):
    indices, values = [], []
    for k, v in result_by_token_id.items():
        indices.append(k)
        values.append(v[key])
    return coo_array((values, (indices, [0] * len(indices))), shape=(vocab_size, 1))


class IncontextFuzzyLM:
    '''Class that fits mechanistically interpretable LM
    '''

    def __init__(
        self,
        tokenizer,
        fuzzy_tokenizer=None,
        fuzzy_lm_name=None,
        context_length=64,
        device='cuda',
        random_state=42,
    ):
        
        # set parameters
        self.tokenizer_ = tokenizer
        self.context_length = context_length
        self.device = device
        self.random_state = random_state

        self.fuzzy_tokenizer = fuzzy_tokenizer
        self._load_fuzzy_lm(fuzzy_lm_name)

        # initialize model parameters
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.vocab_size_ = len(self.tokenizer_)

    def _load_fuzzy_lm(self, fuzzy_lm_name):
        if not fuzzy_lm_name.endswith('.pt'):
            self.fuzzy_llm = AutoModelForCausalLM.from_pretrained(fuzzy_lm_name, token=alm.config.TOKEN_HF, load_in_8bit=True, device_map="auto").eval()
            self.use_llm_as_fuzzy = True
        else:
            checkpoint = torch.load(fuzzy_lm_name, map_location=self.device)
            # create the model
            gptconf = GPTConfig(**checkpoint['model_kwargs'])
            fuzzy_llm = GPT(gptconf)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = 'module.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            fuzzy_llm.load_state_dict(state_dict)
            fuzzy_llm.to(self.device)
            self.fuzzy_llm = fuzzy_llm
            print(f"Loaded model from {fuzzy_lm_name} (saved at iter {checkpoint['iter_num']})")

            self.temperature = 0.1
            if 'temp' in fuzzy_lm_name:
                temperature = float(fuzzy_lm_name[fuzzy_lm_name.index('temp')+4:].split('_')[0].split('/')[0])
                print(f"\tTemperature: {temperature}")
                self.temperature = temperature
            self.use_llm_as_fuzzy = False

    def _fuzzy_matching_in_context(self, batch_input_ids, topk=-1):
        B = len(batch_input_ids)
        if self.context_length > batch_input_ids.shape[-1]:
            input_ids_tensor = torch.Tensor(batch_input_ids).unsqueeze(1).long()
        else:
            input_ids_tensor = torch.Tensor(np.stack([sliding_window_view(np.array(input_ids), self.context_length) for input_ids in batch_input_ids], axis=0)).long()
        
        if self.fuzzy_tokenizer.__class__ != self.tokenizer_.__class__:
            x_str = []
            for ids in input_ids_tensor:
                for i, id in enumerate(ids):
                    x_str += [self.tokenizer_.decode(id[:l+1]) for l in range(len(id))] if i == 0 else [self.tokenizer_.decode(id)]
            token_ids_fuzzy = self.fuzzy_tokenizer(x_str, return_tensors="pt", padding=True).to(self.device)
        
        if self.use_llm_as_fuzzy:
            if self.fuzzy_tokenizer.__class__ != self.tokenizer_.__class__:
                all_logits = []
                for _batch_input_ids, _batch_attention in zip(token_ids_fuzzy['input_ids'].split(16), token_ids_fuzzy['attention_mask'].split(16)):
                    with torch.no_grad():
                        batch_logits = self.fuzzy_llm(_batch_input_ids[:, :_batch_attention.sum(dim=1).max()], _batch_attention[:, :_batch_attention.sum(dim=1).max()]).logits
                    all_logits.append(torch.stack([batch_logits[i, j] for i, j in enumerate(_batch_attention.sum(dim=1) - 1)]))
                logits = torch.cat(all_logits, dim=0).view(B, len(x_str) // B, -1)
            else:
                all_logits = []
                with torch.no_grad():
                    for _batch_input_ids in input_ids_tensor.view(-1, input_ids_tensor.shape[-1]).split(16):
                        logits = self.fuzzy_llm(_batch_input_ids.to(self.device)).logits
                        all_logits.append(logits)
                    logits = torch.cat(all_logits, dim=0)
                logits = logits.view(B, input_ids_tensor.shape[1], *logits.shape[1:])
                logits = torch.concat([logits[:, 0], logits[:, 1:, -1]], dim=1) if logits.shape[1] > 1 else logits[:, 0]
            log_probs = logits.log_softmax(dim=-1)
            distance = (log_probs[:, :-1].exp() * (log_probs[:, :-1] - log_probs[:, -1:])).sum(dim=-1).cpu().detach() # KL divergence
            distance += (log_probs[:, -1:].exp() * (log_probs[:, -1:] - log_probs[:, :-1])).sum(dim=-1).cpu().detach()
            distance = distance / 2
        else:
            if self.fuzzy_tokenizer.__class__ != self.tokenizer_.__class__:
                indices = token_ids_fuzzy['attention_mask'].view(batch_input_ids.shape + (-1,))
                indices[..., :-1] = indices[..., :-1] - indices[..., 1:]
                indices1, indices2 = indices.bool(), indices.bool()
                if indices1.shape[1] > 1:
                    indices1[:, -1:] = False
                    indices2[:, :-1] = False
                input_ids_tensor = token_ids_fuzzy['input_ids']
            else:
                indices1 = torch.zeros_like(input_ids_tensor).bool()
                indices2 = torch.zeros_like(input_ids_tensor).bool()
                indices1[:, 0, :-1] = True
                indices1[:, :-1, -1] = True
                indices2[:, -1, -1] = True
                input_ids_tensor = input_ids_tensor.view(-1, input_ids_tensor.shape[-1])
            distance, _ = self.fuzzy_llm.get_distance(input_ids_tensor.to(self.device), indices1, indices2, temperature=self.temperature)
            distance = distance.cpu().detach()[..., 0] # len(input_ids) - 1
        if topk > 0:
            distance[:, distance.sort().indices[topk:]] = torch.inf
        weight = (-distance).exp().numpy().astype(np.float64)
        cnt = coo_array((weight.reshape(-1), (np.repeat(np.arange(B).reshape(-1, 1), weight.shape[-1], axis=1).reshape(-1), np.array(batch_input_ids[:, -weight.shape[-1]:]).reshape(-1))), shape=(B, self.vocab_size_)).toarray()
        return cnt

    def predict_prob(self, batch_input_ids, incontext_mode=None, topk=-1, return_others=False):
        batch = True
        if batch_input_ids.ndim == 1:
            batch_input_ids = batch_input_ids.unsqueeze(0)
            batch = False
        if incontext_mode == 'infinigram': # do not work as batch
            all_incontext_cnt, others = [], []
            for input_ids in batch_input_ids:
                input_ids = input_ids.tolist() if not isinstance(input_ids, list) else input_ids
                incontext_lm = InfiniGramModel.from_data(documents_tkn=input_ids[:-1], tokenizer=self.tokenizer_)
                prob_next_distr = incontext_lm.predict_prob(np.array(input_ids))
                incontext_cnt = prob_next_distr.count
                suffix_len = prob_next_distr.effective_n
                incontext_sparse = (len(incontext_cnt.nonzero()[0]) == 1)
                all_incontext_cnt.append(incontext_cnt)
                others.append({'suffix_len': suffix_len, 'prompt_cnt': incontext_cnt, 'sparse': incontext_sparse})
            all_incontext_cnt = np.stack(all_incontext_cnt, axis=0)
        elif incontext_mode == 'fuzzy': # work as batch
            all_incontext_cnt = self._fuzzy_matching_in_context(batch_input_ids, topk=topk)
            others = [{'suffix_len': 0, 'prompt_cnt': incontext_cnt, 'sparse': (len(incontext_cnt.nonzero()[0]) == 1)} for incontext_cnt in all_incontext_cnt]
        else:
            raise ValueError(f"Unknown incontext_mode: {incontext_mode}")

        all_incntext_probs = all_incontext_cnt / all_incontext_cnt.sum()

        if batch:
            if return_others:
                return all_incntext_probs, others
            return all_incntext_probs
        else:
            if return_others:
                return all_incntext_probs[0], others[0]
            return all_incntext_probs[0]