from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from infini_gram.engine import InfiniGramEngine
from tqdm import tqdm
from os.path import join
import joblib
import logging
import numpy as np
import random
import torch
import os
import argparse
import alm.hdlm.hdlm
import alm.config
import alm.data.dsets
from alm.eval.scalar_metrics import perplexity_from_token_probs
from alm.mechlm import InfiniGram, MechLM, InfiniGramModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="if debug")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(alm.config.SAVE_DIR_ROOT, "mechlm", "eval"),
        help="directory for saving",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="infini-gram",
        choices=['infini-gram', 'infini-gram-w-incontext', 'incontext-infini-gram', 'mechlm', 'llm'],
        help="type of model to evaluate",
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        type=str,
        default="gpt2",
        help="checkpoint for tokenizer",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="v4_pileval_gpt2",
        help="checkpoint for tokenizer",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=3,
        help="context length of the model",
    )

    # data args
    parser.add_argument(
        '--dataset', type=str, default='babylm', help='dataset to use'
    )
    parser.add_argument(
        '--num_examples_test', type=int, default=10000, help='number of examples to test on'
    )

    return parser.parse_args()


if __name__ == '__main__':
    # hyperparams ######################
    args = get_args()

    model_kwargs = dict(
        device='cuda',
        infinigram_checkpoint=join(alm.config.INFINIGRAM_INDEX_PATH, args.checkpoint),
        context_length=args.context_length,
        random_state=args.seed,
    )

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set up saving
    r = defaultdict(list)
    r.update(vars(args))
    if args.debug:
        args.save_dir = './debug'
    _str_model = f'{args.model_type}_{args.tokenizer_checkpoint}' if args.model_type == 'llm' else f'{args.model_type}_{args.checkpoint.split("/")[-1]}'
    args.save_dir = join(args.save_dir, f'context{args.context_length}', _str_model)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(join(args.save_dir, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(filename=join(args.save_dir, 'eval.log'), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


    # set up data ######################
    # dset = NextWordDataset(raw_tokens=tokenizer(data.TEXT_MINI)[
    # 'input_ids'], max_n_tokens=max_n_tokens)
    if args.dataset == 'babylm':
        # dset = alm.data.dsets.NextWordDataset(tokens_file=join(
        # alm.config.BABYLM_DATA_DIR, f'babylm_dev', 'full.joblib'), max_n_tokens=args.context_length)
        if args.tokenizer_checkpoint == 'gpt2':
            data_dir_name = 'babylm_test'
        elif args.tokenizer_checkpoint == 'meta-llama/Llama-2-7b-hf':
            data_dir_name = 'babylm_test_llama2'
        dset_test = alm.data.dsets.NextWordDataset(tokens_file=join(
            alm.config.BABYLM_DATA_DIR, data_dir_name, 'full.joblib'), max_n_tokens=args.context_length)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')

    # evaluate infini-gram perplexity on dset_test
    use_incontext = False
    if args.model_type in ['infini-gram', 'infini-gram-w-incontext']:
        if args.model_type == 'infini-gram-w-incontext':
            use_incontext=True
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint, add_bos_token=False, add_eos_token=False, token=alm.config.TOKEN_HF)
        if args.checkpoint in ['v4_pileval_gpt2', 'v4_piletrain_llama']:
            lm = InfiniGram(
                tokenizer=tokenizer,
                **model_kwargs
            )
        else:
            lm = InfiniGramModel.from_pretrained(join(alm.config.INFINIGRAM_INDEX_PATH, args.checkpoint))
    elif args.model_type == 'incontext-infini-gram':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint, add_bos_token=False, add_eos_token=False, token=alm.config.TOKEN_HF)
        lm = None
    elif args.model_type == 'mechlm':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint, add_bos_token=False, add_eos_token=False, token=alm.config.TOKEN_HF)
        lm = MechLM(
            tokenizer=tokenizer,
            **model_kwargs
        )
    elif args.model_type == 'llm':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint, token=alm.config.TOKEN_HF)
        lm = AutoModelForCausalLM.from_pretrained(args.tokenizer_checkpoint, token=alm.config.TOKEN_HF).eval().to('cuda')

    token_probs = []

    # should batch this to make it faster...
    recall_1 = []
    recall_5 = []
    recall_1_equal = []
    for i in tqdm(range(args.num_examples_test)):
        token_ids, next_token_id = dset_test[i]
        if token_ids[0].item() == tokenizer.bos_token_id:
            token_ids = token_ids[1:]
            logging.info("First token is deleted because it is bos_token.")
        if args.model_type == 'incontext-infini-gram':
            lm = InfiniGramModel.from_data(documents_tkn=token_ids.tolist()[:-1], tokenizer=tokenizer)
        if isinstance(lm, (InfiniGramModel)):
            prob_next_distr = lm.prob_next_distr(token_ids.numpy())
            all_count = prob_next_distr.count
            # add-one estimation
            all_count = all_count + 1
            next_token_probs = torch.Tensor(all_count / all_count.sum())
        elif args.model_type == 'llm':
            token_ids = token_ids.to('cuda')
            next_token_probs = lm(token_ids).logits[-1, :]
            next_token_probs = torch.softmax(next_token_probs, dim=-1).cpu()
        else:
            next_token_probs = lm.predict_prob(token_ids, estimation='add-one', use_incontext=use_incontext)
        _next_token_probs = next_token_probs[next_token_id].item()
        token_probs.append(_next_token_probs)
        recall_1.append(
            next_token_probs.argmax().item() == next_token_id.item())
        recall_5.append(
            next_token_id.item() in next_token_probs.topk(5).indices.tolist())
        recall_1_equal.append(
            (next_token_probs.max().item() == _next_token_probs) and (next_token_probs.min().item() != _next_token_probs))

    # calculate perplexity
    perplexity = perplexity_from_token_probs(token_probs)
    logging.info(f'test perplexity: \t{perplexity}')
    logging.info(f'test perfect recall@1: \t{np.mean(recall_1)}')
    logging.info(f'test perfect recall@5: \t{np.mean(recall_5)}')
    logging.info(f'test perfect recall@1 (equal): \t{np.mean(recall_1_equal)}')
