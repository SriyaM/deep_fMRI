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
import numpy as np
import requests
from scipy.sparse import coo_array

from .build_infinigram import InfiniGramModel

def get_sparse_array_from_result(result_by_token_id, vocab_size):
    indices, values = [], []
    for k, v in result_by_token_id.items():
        indices.append(k)
        values.append(v['cont_cnt'])
    return coo_array((values, (indices, [0] * len(indices))), shape=(vocab_size, 1))


class InfiniGram:
    '''Class that fits Infini-gram
    '''

    def __init__(
        self,
        tokenizer,
        infinigram_checkpoint='v4_pileval_gpt2',
        context_length=5,
        device='cuda',
        random_state=42,
        load_to_ram=False,
    ):

        # set parameters
        self.tokenizer_ = tokenizer
        self.infinigram_checkpoint = infinigram_checkpoint
        self.context_length = context_length
        self.device = device
        self.random_state = random_state

        # initialize model parameters
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.vocab_size_ = len(self.tokenizer_)
        try:
            self.infinigram_engine = InfiniGramEngine(load_to_ram=load_to_ram, index_dir=infinigram_checkpoint, eos_token_id=self.tokenizer_.eos_token_id)
        except:
            print("Fail to load local indexes, so use API endpoint")
        

    def predict_prob(self, input_ids, smoothing_type=None, return_others=False, use_incontext=False):
        input_ids = input_ids.tolist() if not isinstance(input_ids, list) else input_ids
        cut_input_ids = input_ids[-self.context_length:] if len(input_ids) > self.context_length else input_ids
        if hasattr(self, 'infinigram_engine'):
            results = self.infinigram_engine.infgram_ntd(prompt_ids=cut_input_ids)
        else:
            assert len(input_ids) <= 500
            payload = {
                'index': self.infinigram_checkpoint.split('/')[-1],
                'query_type': 'infgram_ntd',
                'query_ids': input_ids,
            }
            results = requests.post('https://api.infini-gram.io/', json=payload).json()
        result_by_token_id, suffix_len, prompt_cnt = results['result_by_token_id'], results['suffix_len'], results['prompt_cnt']

        if use_incontext:
            incontext_lm = InfiniGramModel.from_data(documents_tkn=input_ids[:-1], tokenizer=self.tokenizer_)
            incontext_results = incontext_lm.predict_prob(np.array(input_ids), smoothing_type=smoothing_type)
            incontext_suffix_len, incontext_cnt = incontext_results.effective_n, incontext_results.count
            incontext_sparse = (len(incontext_cnt.nonzero()[0]) == 1)

        infinigram_cnt = get_sparse_array_from_result(result_by_token_id, self.vocab_size_)
        sparse = (len(infinigram_cnt.nonzero()[0]) == 1)
        infinigram_cnt = infinigram_cnt.toarray()[:, 0]
        if smoothing_type == 'add-one':
            infinigram_cnt = infinigram_cnt + 1
        infinigram_probs = infinigram_cnt / infinigram_cnt.sum()
        if use_incontext:
            if incontext_results.effective_n > 0:
                infinigram_probs = incontext_results.distr
                sparse = incontext_sparse

        if return_others:
            if use_incontext:
                return infinigram_probs, {'suffix_len': suffix_len, 'prompt_cnt': prompt_cnt, 'sparse': sparse,
                                          'incontext_suffix_len': incontext_suffix_len, 'incontext_prompt_cnt': int(incontext_cnt.sum().item()), 'incontext_sparse': incontext_sparse}
            return infinigram_probs, {'suffix_len': suffix_len, 'prompt_cnt': prompt_cnt, 'sparse': sparse}
        return infinigram_probs
