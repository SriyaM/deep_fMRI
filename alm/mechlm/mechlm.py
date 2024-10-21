from collections import defaultdict
import torchhd
from transformers import AutoTokenizer
from infini_gram.engine import InfiniGramEngine
from tqdm import tqdm
import torch
from os.path import join
import joblib
import numpy as np
from typing import List, Dict
import numpy as np
import requests


class MechLM:
    '''Class that fits mechanistically interpretable LM
    '''

    def __init__(
        self,
        tokenizer,
        infinigram_checkpoint='v4_pileval_gpt2',
        learning_rate=0.1,
        context_length=5,
        device='cuda',
        random_state=42,
    ):

        # set parameters
        self.learning_rate = learning_rate
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
            self.infinigram_engine = InfiniGramEngine(index_dir=infinigram_checkpoint, eos_token_id=self.tokenizer_.eos_token_id)
        except:
            print("Fail to load local indexes, so use API endpoint")
        # self.lm = AutoModelForCausalLM.from_pretrained(self.tokenizer_checkpoint).to(self.device)

    def fit(self, dset):
        '''
        Calculates perplexity over the dataset and fits the model if fit=True.
        Perplexity is calculated as the exponential of the mean negative log-probabilities of the next token.
        Lower is better.

        Params
        ------
        dset: pytorch Dataset
            Dataset to calculate perplexity over
        '''
        pass
    
    def predict_prob(self, input_ids):
        if hasattr(self, 'infinigram_engine'):
            results = self.infinigram_engine.infgram_ntd(prompt_ids=input_ids.tolist())['result_by_token_id']
        else:
            payload = {
                'index': self.infinigram_checkpoint.split('/')[-1],
                'query_type': 'infgram_ntd',
                'query_ids': input_ids.tolist(),
            }
            results = requests.post('https://api.infini-gram.io/', json=payload).json()['result_by_token_id']
        infinigram_probs = torch.zeros(self.vocab_size_)
        for k, v in results.items():
            infinigram_probs[int(k)] = v['prob']
        return infinigram_probs

