"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os, glob
from os.path import join
import time
import math
import logging
from contextlib import nullcontext
from collections import defaultdict, deque
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datasets import load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from alm.data.dsets_embed import SentenceDataset
from alm.data.dsets import NextWordDataset
from alm.mechlm.mini_gpt import GPTConfig, GPT
import alm.config
import joblib
from torch.utils.tensorboard import SummaryWriter
import wandb
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="if debug")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=alm.config.DATA_DIR_ROOT,
        help="directory for saving",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8192, help="# of layers"
    )
    parser.add_argument(
        "--slm_model_path",
        type=str,
        default=join(alm.config.BLOB_DIR, "Exp/alt-lm-gpt/mechlm/train_simlm_sing0826/llama3-8B/exp_layer2_ModifiedTransformerBlock_head8_outembd128_spe_temp0.1/lr0.0001_embed-update0.0001_kl1.00_batch32_sampling128-256_length32_gpu2_maxiter20000/checkpoint/ckpt-best-00.pt"),
        help="directory for saving",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="tokenizer",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=32,
        help="context length of the model",
    )
    parser.add_argument(
        '--dataset', type=str, default='openwebtext', help='dataset to use for training'
    )
    parser.add_argument(
        '--part', type=int, default=0
    )
    parser.add_argument(
        '--n_part', type=int, default=1
    )
    return parser.parse_args()

if __name__ == '__main__':
    # hyperparams ######################
    args = get_args()

    # various inits, derived attributes, I/O setup
    backend = 'nccl' # 'nccl', 'gloo', etc.
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        ddp_rank = seed_offset = 0
        ddp_world_size = 1

    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

    save_dir = os.path.join('/'.join(args.slm_model_path.split('/')[:-2]), 'embeds')
    if master_process:
        os.makedirs(save_dir, exist_ok=True)
            
        # set up logging
        logger = logging.getLogger()
        logging.basicConfig(filename=join(save_dir, 'get_embed.log'), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

    if master_process:
        logger.info(f'Save Dir: {save_dir}')

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

    if args.tokenizer == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)
    elif args.tokenizer == 'llama':
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=os.environ.get('HF_TOKEN'), use_fast=False, add_bos_token=False, add_eos_token=False) # The fast tokenizer seems unbearably slow ...
    eos_token_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    # load the data
    if master_process:
        logger.info(f"Loading data from {args.data_dir}")
    if args.dataset == 'openwebtext':
        data_all = SentenceDataset(data_dir=join(alm.config.DATA_DIR_ROOT, 'openwebtext'), max_n_tokens=args.context_length, min_n_tokens=args.context_length//2, split='all', pad_token_id=eos_token_id)
        ddp_local_len_data = len(data_all) // (ddp_world_size * args.n_part)
        if ddp_rank == ddp_world_size - 1 and args.part == args.n_part - 1:
            data_all.idx2newidx = data_all.idx2newidx[ddp_local_len_data * (args.part * ddp_world_size + ddp_rank):]
        else:
            data_all.idx2newidx = data_all.idx2newidx[ddp_local_len_data * (args.part * ddp_world_size + ddp_rank):ddp_local_len_data * (args.part * ddp_world_size + ddp_rank + 1)]

    # model init
    ckpt_path = join(alm.config.BLOB_DIR, args.slm_model_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_kwargs = checkpoint['model_kwargs']
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
    fuzzy_llm.to(device)
    if master_process:
        logger.info(f"Loaded model from {ckpt_path} (saved at iter {checkpoint['iter_num']})")

    # training loop
    t0 = time.time()
    
    # set up saving
    r = defaultdict(list)
    r.update(vars(args))
    os.makedirs(save_dir, exist_ok=True)

    # setup the dataloader
    data_loader = DataLoader(
        data_all,
        shuffle=False,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=8,
    )

    if master_process:
        logger.info("Starting inference...")
    all_indices, all_x, all_logits = [], [], []
    save_num = 0
    for iter_num, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        idx, x = batch
        x = x.to(device)
        assert x.max() < vocab_size, f"out of bounds: {x.max()}"
        
        with torch.no_grad():
            logits, _, _ = fuzzy_llm(x)
        
        all_indices.append(idx)
        all_x.append(x.detach())
        all_logits.append(logits[:, args.context_length//2-1:].detach().cpu())

        if len(all_indices) > 256 or iter_num == len(data_loader) - 1:
            all_indices = torch.cat(all_indices, dim=0)
            all_x = torch.cat(all_x, dim=0)
            all_logits = torch.cat(all_logits, dim=0)
            all_attention_mask = (all_x != eos_token_id).float()
            for l in range(args.context_length // 2, args.context_length + 1):
                mask = (all_attention_mask[:, :l].sum(1) == l).cpu()
                l_all_x, l_all_logits, l_all_indices = all_x[mask], all_logits[mask], all_indices[mask]
                unique, inverse = torch.unique(l_all_x[:, :l].flip(1), sorted=True, return_inverse=True, dim=0)
                unique = unique.flip(1)
                inverse = inverse.cpu()
                perm = torch.arange(inverse.size(0), dtype=inverse.dtype)
                inverse, perm = inverse.flip([0]), perm.flip([0])
                perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
                unique_logits = l_all_logits[perm, l-1 - args.context_length//2-1]
                unique_indices = l_all_indices[perm]
                unique_counts = inverse.bincount()
                assert len(unique_counts) == len(unique_indices)

                np.save(join(save_dir, f'all_logits_l{l:02d}_{ddp_rank}_iter{save_num:05d}.npy'), unique_logits.numpy())
                file_index = open(join(save_dir, f'all_index_l{l:02d}_p{args.part * ddp_world_size + ddp_rank}.{ddp_world_size * args.n_part}_iter{save_num:05d}'), 'wb')
                file_count = open(join(save_dir, f'all_counts_l{l:02d}_{ddp_rank}_iter{save_num:05d}'), 'wb')
                index_bytes = np.array(unique_indices, dtype=np.uint16).reshape(-1).view(np.uint8).tobytes()
                file_index.write(index_bytes)
                count_bytes = np.array(unique_counts, dtype=np.uint64).view(np.uint8).tobytes()
                file_count.write(count_bytes)
                file_index.close()
                file_count.close()
                if master_process:
                    logger.info(f"Save iter{save_num:05d} for length {l}")
            save_num += 1
            del all_indices
            del all_x
            del all_logits
            with torch.no_grad():
                torch.cuda.empty_cache()
            all_indices, all_x, all_logits = [], [], []