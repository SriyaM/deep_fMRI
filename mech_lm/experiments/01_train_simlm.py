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
from scipy.sparse import coo_array

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from alm.data.dsets_embed import SentenceDataset, SentenceSlidingWindowDataset
from alm.data.dsets import NextWordDataset
from alm.mechlm.mini_gpt import GPTConfig, GPT
import alm.config
import joblib
import wandb
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="if debug")
    parser.add_argument("--save_data", type=bool, default=False, help="if debug")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=alm.config.DATA_DIR_ROOT,
        help="directory for saving",
    )
    parser.add_argument(
        "--prep_data_dir",
        type=str,
        default=join(alm.config.BLOB_DIR, "Exp/alt-lm-gpt/mechlm/train_simlm_sing0823/llama3-8B/data/batch32_sampling128-512_length32"),
        help="directory for saving",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(alm.config.SAVE_DIR_ROOT, "mechlm", "train_simlm"),
        help="directory for saving",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default='exp',
        help="directory for saving",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1
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
        default=1024,
        help="context length of the model",
    )
    parser.add_argument(
        "--sampling_size",
        type=int,
        default=128,
        help="context length of the model",
    )
    parser.add_argument(
        "--sampling_size2",
        type=int,
        default=None,
        help="context length of the model",
    )
    parser.add_argument(
        '--init_from', type=str, default='gpt2', help='scratch or resume or gpt2*', choices=['scratch', 'resume', 'gpt2']
    )
    parser.add_argument(
        "--emb_size", type=int, default=5000, help="embedding size"
    )
    parser.add_argument(
        "--n_layer", type=int, default=1, help="# of layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=12, help="# of heads"
    )
    parser.add_argument(
        "--n_embd", type=int, default=768, help="embedding dimension"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="embedding dimension"
    )
    parser.add_argument(
        "--n_out_embd", type=int, default=768, help="embedding dimension"
    )
    parser.add_argument(
        "--use_trained_pe", type=bool, default=False, help="embedding dimension"
    )
    parser.add_argument(
        "--attention_block", type=str, default='AttentionOnlyBlock', choices=['AttentionOnlyBlock', 'ModifiedAttentionOnlyBlock', 'TransformerBlock', 'ModifiedTransformerBlock']
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="# of layers"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--lr_embed", type=float, default=0.0, help="learning rate"
    )
    parser.add_argument(
        '--similarity_function', type=str, default='cosine', help='similarity function', choices=['cosine', 'jsd', 'l2']
    )
    parser.add_argument(
        '--num_epochs', type=int, default=2, help='number of epochs'
    )
    parser.add_argument(
        '--teacher_llm', type=str, default='gpt2', help='gpt2', choices=['gpt2', 'llama2-7B', 'llama3-8B']
    )
    parser.add_argument(
        '--lamb_ce', type=float, default=0., help='gpt2'
    )
    parser.add_argument(
        '--lamb_kl', type=float, default=0., help='gpt2'
    )
    parser.add_argument(
        '--lamb_mse', type=float, default=0., help='gpt2'
    )

    # data args
    parser.add_argument(
        '--train_dataset', type=str, default='openwebtext', help='dataset to use for training'
    )
    parser.add_argument(
        '--test_dataset', type=str, default='babylm', help='dataset to use for testing'
    )
    parser.add_argument(
        '--num_examples_test', type=int, default=2000, help='number of examples to test on'
    )
    parser.add_argument("--local-rank", type=int)

    return parser.parse_args()

if __name__ == '__main__':
    # hyperparams ######################
    args = get_args()
    
    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    eval_interval = 1000
    log_interval = 100
    if args.debug:
        eval_interval = 10
        log_interval = 10
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    # wandb logging
    wandb_log = False # disabled by default
    wandb_project = 'owt'
    wandb_run_name = 'gpt2' # 'run' + str(time.time())
    # data
    gradient_accumulation_steps = 8 # used to simulate larger batch sizesbat
    block_size = 1024
    batch_size = args.batch_size
    
    num_workers = 4
    # model
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    temperature = args.temperature
    allow_embed_update = args.lr_embed > 0
    # adamw optimizer
    max_iters = 30000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_norm_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 1000 # how many steps to warm up for
    # lr_decay_iters = 10000 # should be ~= max_iters per Chinchilla
    lr_decay_iters = max_iters
    min_lr = args.learning_rate * 0.1 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open(os.path.join(alm.config.HDLM_EXP_DIR, 'configurator.py')).read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

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
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        ddp_rank = seed_offset = 0
        ddp_world_size = 1

    # Set directory to save
    if args.sampling_size2 is None:
        args.sampling_size2 = args.sampling_size

    if args.exp_name is None:
        args.exp_name = 'exp'
    args.exp_name += f"_layer{args.n_layer}_{args.attention_block}_head{args.n_head}_outembd{args.n_out_embd}"
    if not args.use_trained_pe:
        args.exp_name += "_spe"
    if args.similarity_function == 'cosine':
        args.exp_name += f"_temp{args.temperature}"
    else:
        args.exp_name += f"_sim-{args.similarity_function}"
    if allow_embed_update:
        setting_name = f"lr{args.learning_rate}_embed-update{args.lr_embed}"
    else:
        setting_name = f"lr{args.learning_rate}"
    if args.lamb_ce > 0:
        setting_name += f"_ce{args.lamb_ce:0.2f}"
    if args.lamb_mse > 0:
        setting_name += f"_mse{args.lamb_mse:0.2f}"
    if args.lamb_kl > 0:
        setting_name += f"_kl{args.lamb_kl:0.2f}"
    if args.sampling_size != args.sampling_size2:
        setting_name += f"_batch{args.batch_size}_sampling{args.sampling_size}-{args.sampling_size2}_length{args.context_length}"
    else:
        setting_name += f"_batch{args.batch_size}_sampling{args.sampling_size}_length{args.context_length}"
    if args.n_gpus > 1:
        setting_name += f"_gpu{args.n_gpus}"
    if max_iters != 60000:
        setting_name += f"_maxiter{max_iters}"

    prep_data_dir = ''
    if args.sampling_size != args.sampling_size2:
        prep_data_dir += f"batch{args.batch_size}_sampling{args.sampling_size}-{args.sampling_size2}_length{args.context_length}"
    else:
        prep_data_dir += f"batch{args.batch_size}_sampling{args.sampling_size}_length{args.context_length}"
    if args.prep_data_dir is not None:
        if prep_data_dir != args.prep_data_dir.split('/')[-1]:
            prep_data_dir = join(alm.config.BLOB_DIR, args.prep_data_dir)
            setting_name += '_' + args.prep_data_dir.split('/')[-1].split('_')[0]
        else:
            prep_data_dir = join(alm.config.BLOB_DIR, 'Exp/alt-lm-gpt/mechlm/train_simlm_sing0823', args.teacher_llm, 'data', prep_data_dir)
    else:
        prep_data_dir = join(alm.config.BLOB_DIR, 'Exp/al`t-lm-gpt/mechlm/train_simlm_sing0823', args.teacher_llm, 'data', prep_data_dir)
    prep_data_dir_val = join(alm.config.BLOB_DIR, 'Exp/alt-lm-gpt/mechlm/train_simlm_sing0823', args.teacher_llm, 'data/batch32_sampling128-256_length32')
    save_dir = join(args.save_dir, args.teacher_llm, args.exp_name, setting_name)
    last_ckpt_index = -1
    existing_ckpt_list = glob.glob(join(save_dir, 'checkpoint/ckpt-last-*.pt'))
    if len(existing_ckpt_list) > 0:
        args.init_from = 'resume'
        last_ckpt_index = max([int(f.split('-')[-1].split('.')[0]) for f in existing_ckpt_list])
        ckpt_prelast_path = join(save_dir, f'checkpoint/ckpt-last-{last_ckpt_index:02d}.pt')
    ckpt_best_path = join(save_dir, f'checkpoint/ckpt-best-{last_ckpt_index+1:02d}.pt')
    ckpt_last_path = join(save_dir, f'checkpoint/ckpt-last-{last_ckpt_index+1:02d}.pt')

    if master_process:
        os.makedirs(join(save_dir, 'checkpoint'), exist_ok=True)

        with open(join(save_dir, 'args.txt'), 'w') as f:
            for k, v in vars(args).items():
                f.write(f'{k}: {v}\n')
            for k, v in config.items():
                f.write(f'{k}: {v}\n')
        
        log_tool = ['wandb']
        if args.debug:
            log_tool = []
        if 'wandb' in log_tool:
            wandb.login(key=alm.config.WANDB_API_KEY)
            wandb_id, wandb_resume = None, False
            wandb_runfile = glob.glob(join(save_dir, 'wandb/run-*'))
            if args.init_from == 'resume' or len(wandb_runfile) > 0:
                wandb_id = wandb_runfile[0].split('-')[-1]
                wandb_resume = True
            config.update(vars(args))
            wandb_run = wandb.init(project='alt-lm-gpt',
                                    name=join(args.teacher_llm, args.exp_name, setting_name),
                                    config=config, dir=save_dir,
                                    resume=wandb_resume, id=wandb_id)
        
        # set up logging
        logger = logging.getLogger()
        logging.basicConfig(filename=join(save_dir, 'train.log'), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        logger.info(f'Save Dir: {save_dir}')
        logger.info(f'Preprocessed Data Dir: {prep_data_dir}')

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if args.tokenizer == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)
    elif args.tokenizer == 'llama':
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=os.environ.get('HF_TOKEN'), use_fast=False, add_bos_token=False, add_eos_token=False) # The fast tokenizer seems unbearably slow ...
    eos_token_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    # load the data
    if master_process:
        logger.info(f"Loading data from {args.data_dir}")
    if args.train_dataset == 'openwebtext':
        data_train = SentenceDataset(data_dir=join(alm.config.DATA_DIR_ROOT, 'openwebtext'), max_n_tokens=args.context_length, min_n_tokens=args.context_length//2, split='train', pad_token_id=eos_token_id)
        data_val = SentenceDataset(data_dir=join(alm.config.DATA_DIR_ROOT, 'openwebtext'), max_n_tokens=args.context_length, min_n_tokens=args.context_length//2, split='val', pad_token_id=eos_token_id)
        data_test = SentenceSlidingWindowDataset(data_dir=join(alm.config.DATA_DIR_ROOT, 'openwebtext'), max_n_tokens=block_size, min_n_tokens=block_size, window_n_tokens=args.context_length, split='val', pad_token_id=eos_token_id)
    elif args.train_dataset == 'babylm':
        data_dir = os.path.join(alm.config.DATA_DIR_ROOT, args.train_dataset)
        data_train = joblib.load(os.path.join(data_dir, f'babylm_dev', 'full.joblib'))
    if args.test_dataset == 'babylm':
        dset_test = NextWordDataset(tokens_file=join(
            alm.config.BABYLM_DATA_DIR, 'babylm_test', 'full.joblib'), max_n_tokens=args.context_length)
        example_nums = np.arange(args.num_examples_test)
        sub_dset_test = Subset(dset_test, example_nums)
        test_loader = DataLoader(sub_dset_test, batch_size=10, num_workers=4, drop_last=False)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num, iter_num_restart = 0, 0
    best_val_loss = 1e9

    # model init
    model_kwargs = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=block_size, n_out_embd=args.n_out_embd,
                        bias=bias, vocab_size=vocab_size, dropout=dropout,
                        use_trained_pe=args.use_trained_pe,
                        similarity_function=args.similarity_function,
                        attention_block=args.attention_block) # start with model_kwargs from command line
    if args.init_from == 'scratch':
        # init a new model from scratch
        if master_process:
            logger.info("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_kwargs)
        model = GPT(gptconf)
    elif args.init_from == 'resume':
        if master_process:
            logger.info(f"Resuming training from {save_dir}")
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_prelast_path, map_location=device)
        checkpoint_model_kwargs = checkpoint['model_kwargs']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k, v in checkpoint_model_kwargs.items():
            model_kwargs[k] = v
        # create the model
        gptconf = GPTConfig(**model_kwargs)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        iter_num_restart = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        args.init_from = True
    elif args.init_from.startswith('gpt2'):
        if master_process:
            logger.info(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(n_layer=args.n_layer, n_head=args.n_head, block_size=block_size, n_out_embd=args.n_out_embd,
                             bias=bias, dropout=dropout, use_trained_pe=args.use_trained_pe, 
                             similarity_function=args.similarity_function, attention_block=args.attention_block)
        model = GPT.from_pretrained(args.init_from, override_args, allow_embed_update=allow_embed_update)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['vocab_size', 'n_embd']:
            model_kwargs[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_kwargs['block_size'] = block_size # so that the checkpoint will have the right value
    
    # wrap model into DDP container
    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    load_in_8bit = False
    if args.teacher_llm == 'llama2-7B':
        args.teacher_llm = 'meta-llama/Llama-2-7b-hf'
        load_in_8bit = True
    if args.teacher_llm == 'llama3-8B':
        args.teacher_llm = 'meta-llama/Meta-Llama-3-8B'
        load_in_8bit = True
    tokenizer_teacher = AutoTokenizer.from_pretrained(args.teacher_llm, token=alm.config.TOKEN_HF)
    tokenizer_teacher.pad_token = tokenizer_teacher.eos_token
    teacher_llm = AutoModelForCausalLM.from_pretrained(args.teacher_llm, token=alm.config.TOKEN_HF, load_in_8bit=load_in_8bit,
                                                       device_map=device)
    if args.teacher_llm == 'gpt2':
        teacher_llm = teacher_llm.eval().to(device)

    # optimizer
    optimizer = model.configure_optimizers(learning_rate=args.learning_rate, learning_rate_embed=args.lr_embed, weight_decay=config['weight_decay'],
                                           betas=(config['beta1'], config['beta2']), device_type=config['device'])
    if args.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return args.learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (args.learning_rate - min_lr)
    
    # training loop
    t0 = time.time()
    
    # set up saving
    r = defaultdict(list)
    r.update(vars(args))

    # setup the dataloader
    train_loader = DataLoader(
        data_train,
        sampler=torch.utils.data.RandomSampler(data_train, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    val_loader = DataLoader(
        data_val,
        shuffle=False,
        pin_memory=True,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    test_loader = DataLoader(
        data_test,
        shuffle=False,
        pin_memory=True,
        batch_size=config['batch_size'] // 2,
        num_workers=config['num_workers'],
    )

    def forward_teacher_llm(x, attention_mask, indices1, indices2):
        with torch.no_grad():
            if args.teacher_llm != args.tokenizer:
                indices = torch.cat([indices1, indices2])
                _new_x, _new_attention_mask, new_indices = [], [], []
                max_len = 0
                for i, j in zip(indices // args.context_length, indices % args.context_length):
                    str = tokenizer_teacher(tokenizer.decode(x[i, :j+1].cpu()), return_tensors="pt")
                    _new_x.append(str['input_ids'])
                    _new_attention_mask.append(str['attention_mask'])
                    new_indices.append(str['attention_mask'].sum() - 1)
                    max_len = max(max_len, str['attention_mask'].sum())
                new_indices = torch.stack(new_indices).to(indices.device)
                new_indices += indices // args.context_length * max_len
                unique_new_indices, inverse_indices = new_indices.unique(return_inverse=True)
                unique_indices = torch.zeros(len(unique_new_indices), dtype=indices.dtype)
                new_x = torch.zeros(len(unique_new_indices), max_len, dtype=_new_x[0].dtype)
                new_attention_mask = torch.zeros(len(unique_new_indices), max_len, dtype=_new_attention_mask[0].dtype)
                for in_idx, idx, _id, _attn in zip(inverse_indices, indices, _new_x, _new_attention_mask):
                    unique_indices[in_idx] = idx
                    new_x[in_idx, :len(_id[0])] = _id[0]
                    new_attention_mask[in_idx, :len(_attn[0])] = _attn[0]
                new_x[new_attention_mask == 0] = tokenizer_teacher.eos_token_id
                logit_log_probs = teacher_llm(new_x.to(device), attention_mask=new_attention_mask.to(device), return_dict=True).logits
                logit_log_probs_reshape = logit_log_probs[torch.arange(len(unique_new_indices)), unique_new_indices % max_len]
                logit_log_probs_reshape1, logit_log_probs_reshape2 = [], []
                unique_indices1, unique_indices2 = [], []
                for idx, logprob in zip(unique_indices, logit_log_probs_reshape):
                    if idx in indices1:
                        logit_log_probs_reshape1.append(logprob)
                        unique_indices1.append(idx)
                    else:
                        logit_log_probs_reshape2.append(logprob)
                        unique_indices2.append(idx)
                logit_log_probs_reshape1, logit_log_probs_reshape2 = torch.stack(logit_log_probs_reshape1, dim=0), torch.stack(logit_log_probs_reshape2, dim=0)
                unique_indices1, unique_indices2 = torch.Tensor(unique_indices1).to(indices1.dtype), torch.Tensor(unique_indices2).to(indices2.dtype)

            else:
                new_x, new_attention_mask, unique_indices1, unique_indices2 = x, attention_mask, indices1, indices2
                logit_log_probs = teacher_llm(new_x, attention_mask=new_attention_mask, return_dict=True).logits
                logit_log_probs_reshape = logit_log_probs.view(-1, logit_log_probs.size(-1))
                logit_log_probs_reshape1 = logit_log_probs_reshape[unique_indices1]
                logit_log_probs_reshape2 = logit_log_probs_reshape[unique_indices2]
            if logit_log_probs_reshape.shape[-1] > 50000:
                _values, _indices = logit_log_probs_reshape.softmax(dim=-1).sort(descending=True, dim=1)
                selected = _values.cumsum(dim=1) < 0.95
                selected = torch.cat([selected[..., -1:], selected[..., :-1]], dim=-1)
                selected[:, 0] = True
                selected = torch.concat([_indices[i, j] for i, j in enumerate(selected)]).unique()
                # if len(selected) > 50000:
                #     selected = _values.cumsum(dim=1) < 0.9
                #     selected[:, 0] = True
                #     selected = torch.concat([_indices[i, j] for i, j in enumerate(selected)]).unique()
                logit_log_probs_reshape1 = logit_log_probs_reshape1[:, selected]
                logit_log_probs_reshape2 = logit_log_probs_reshape2[:, selected]
            logit_log_probs_reshape1 = logit_log_probs_reshape1.log_softmax(dim=-1)
            logit_log_probs_reshape2 = logit_log_probs_reshape2.log_softmax(dim=-1)
            distance = (logit_log_probs_reshape2[None, :].exp() * (logit_log_probs_reshape2[None, :] - logit_log_probs_reshape1[:, None])).sum(dim=-1)
            distance += (logit_log_probs_reshape1[:, None].exp() * (logit_log_probs_reshape1[:, None] - logit_log_probs_reshape2[None, :])).sum(dim=-1)
            distance = distance / 2
            labels = distance.argsort(dim=-1)
        return logit_log_probs, labels[:, 0], labels, unique_indices1, unique_indices2, distance

    raw_model = model.module if ddp else model # unwrap DDP container if needed
    if master_process:
        logger.info("Starting training...")
    model.train()
    iter_num, local_num = 0, 0
    iter_time = time.time()
    data_iter = iter(train_loader)
    loss_hist = deque(maxlen=log_interval * gradient_accumulation_steps)
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter_num < iter_num_restart:
            iter_num += 1
            local_num += gradient_accumulation_steps
            continue

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            # fetch the next batch (x, y) and re-init iterator if needed
            with ctx:
                try:
                    _, batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    _, batch = next(data_iter)
                if os.path.exists(join(prep_data_dir, f'train_iter{local_num:08d}_rank{ddp_rank}.pt')):
                    try:
                        preprocessed_data_iter = torch.load(join(prep_data_dir, f'train_iter{local_num:08d}_rank{ddp_rank}.pt'))
                        labels, labels_rank, indices1, indices2, distance, x = preprocessed_data_iter['labels'], preprocessed_data_iter['labels_rank'], preprocessed_data_iter['indices1'], preprocessed_data_iter['indices2'], preprocessed_data_iter['distance'], preprocessed_data_iter['x']
                        x = x.to(device)
                    except:
                        raise FileExistsError(f"Error in loading {join(prep_data_dir, f'train_iter{local_num:08d}_rank{ddp_rank}.pt')}")
                else:
                    logger.info(f"Process iter {local_num}")
                    x = batch.to(device)
                    attention_mask = (x != eos_token_id).float()
                    assert x.max() < vocab_size, f"out of bounds: {x.max()}"

                    indices = torch.arange(x.shape[0] * x.shape[1], device=attention_mask.device).view_as(x)[attention_mask > 0]
                    indices_perm = indices[torch.randperm(len(indices))]
                    if len(indices_perm) <= args.sampling_size + args.sampling_size2:
                        indices1, indices2 = indices_perm[:len(indices_perm) // 2], indices_perm[len(indices_perm) // 2:]
                    else:
                        indices1, indices2 = indices_perm[:args.sampling_size], indices_perm[args.sampling_size:args.sampling_size+args.sampling_size2]

                    _, labels, labels_rank, indices1, indices2, distance = forward_teacher_llm(x, attention_mask, indices1, indices2)

                    if args.save_data:
                        save_data_iter = {
                            'labels': labels.detach().cpu(),
                            'labels_rank': labels_rank.detach().cpu(),
                            'indices1': indices1.detach().cpu(),
                            'indices2': indices2.detach().cpu(),
                            'distance': distance.detach().cpu(),
                            'x': x.detach().cpu()
                        }
                        torch.save(save_data_iter, join(prep_data_dir, f'train_iter{local_num:08d}_rank{ddp_rank}.pt'))

                labels, labels_rank, indices1, indices2, distance = labels.to(device), labels_rank.to(device), indices1.to(device), indices2.to(device), distance.to(device)
                # forward the model
                logits, distance, loss = model(x.to(device), indices=indices1, indices2=indices2, targets=labels.detach(), distance_targets=distance.detach(), temperature=temperature,
                                            lamb_ce=args.lamb_ce, lamb_mse=args.lamb_mse, lamb_kl=args.lamb_kl)
                
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            scaler.scale(loss).backward()
            local_num += 1
        # clip the gradient
        if config['grad_norm_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1
        tnow = time.time()

        loss_hist.append(loss.item() * gradient_accumulation_steps)

        # Log the training metrics
        if iter_num % log_interval == 0 and master_process:
            loss = np.array(loss_hist).mean()
            with torch.no_grad():
                pred_rank = distance.argsort(dim=-1)
                recall1 = (pred_rank[:, 0] == labels_rank[:, 0]).float().mean().item()
                recall5 = (pred_rank[:, :5] == labels_rank[:, :1]).float().sum(1).mean().item()
            logging.info(f"iter {iter_num}, train loss={loss:.4f}, train recall1={recall1:.4f}, train recall5={recall5:.4f}, time={tnow - iter_time:.2f}s")
            
            lr_dict = {f'training/lr_{g_idx}': p_group['lr'] for g_idx, p_group in enumerate(optimizer.param_groups)}
            if 'wandb' in log_tool:
                wandb.log({'loss/train': loss, 'metrics/train_recall1': recall1, 'metrics/train_recall5': recall5}, step=iter_num)
                wandb.log(lr_dict, step=iter_num)
        
        # Log the evaluation metrics
        if iter_num % eval_interval == 0 and master_process:
            model.eval()
            loss_hist_val = []
            target_rank_all, pred_rank_all, mrr_rank_all = [], [], []
            with torch.no_grad():
                for iter_val, (_, x) in enumerate(val_loader):
                    if os.path.exists(join(prep_data_dir_val, f'val_iter{iter_val:08d}_rank{ddp_rank}.pt')):
                        preprocessed_data_iter = torch.load(join(prep_data_dir_val, f'val_iter{iter_val:08d}_rank{ddp_rank}.pt'))
                        labels, labels_rank, indices1, indices2, distance, x = preprocessed_data_iter['labels'], preprocessed_data_iter['labels_rank'], preprocessed_data_iter['indices1'], preprocessed_data_iter['indices2'], preprocessed_data_iter['distance'], preprocessed_data_iter['x']
                        x = x.to(device)
                    else:
                        x = x.to(device)
                        attention_mask = (x != eos_token_id).float()
                        assert x.max() < vocab_size, f"out of bounds: {x.max()}"

                        indices = torch.arange(x.shape[0] * x.shape[1], device=attention_mask.device).view_as(x)[attention_mask > 0]
                        indices_perm = indices[torch.randperm(len(indices))]
                        if len(indices_perm) <= args.sampling_size + args.sampling_size2:
                            indices1, indices2 = indices_perm[:len(indices_perm) // 2], indices_perm[len(indices_perm) // 2:]
                        else:
                            indices1, indices2 = indices_perm[:args.sampling_size], indices_perm[args.sampling_size:args.sampling_size+args.sampling_size2]

                        _, labels, labels_rank, indices1, indices2, distance = forward_teacher_llm(x, attention_mask, indices1, indices2)
                        if args.save_data:
                            save_data_iter = {
                                'labels': labels.detach().cpu(),
                                'labels_rank': labels_rank.detach().cpu(),
                                'indices1': indices1.detach().cpu(),
                                'indices2': indices2.detach().cpu(),
                                'distance': distance.detach().cpu(),
                                'x': x.detach().cpu()
                            }
                            torch.save(save_data_iter, join(prep_data_dir_val, f'val_iter{iter_num:08d}_rank{ddp_rank}.pt'))
                    
                    labels, labels_rank, indices1, indices2, distance = labels.to(device), labels_rank.to(device), indices1.to(device), indices2.to(device), distance.to(device)

                    target_rank_all.append(labels_rank[:, :5].detach().cpu())
                    # forward the model
                    logits, distance, loss = model(x, indices=indices1, indices2=indices2, targets=labels.detach(), distance_targets=distance.detach(), temperature=temperature,
                                                   lamb_ce=args.lamb_ce, lamb_mse=args.lamb_mse, lamb_kl=args.lamb_kl)
                    pred_rank = distance.argsort(dim=-1).detach().cpu()
                    pred_rank_all.append(pred_rank[:, :5])
                    mrr_rank_all.append(1 / ((pred_rank == labels.unsqueeze(1).detach().cpu()).int().argmax(1) + 1))
                    loss_hist_val.append(loss.item())

                    if len(loss_hist_val) > args.num_examples_test:
                        break
            
            loss = np.array(loss_hist_val).mean()
            pred_rank_all = torch.cat(pred_rank_all, dim=0)
            target_rank_all = torch.cat(target_rank_all, dim=0)
            recall1 = (pred_rank_all[:, 0] == target_rank_all[:, 0]).float().mean()
            recall5 = (pred_rank_all[:, :5] == target_rank_all[:, :1]).float().sum(1).mean()
            mrr = torch.cat(mrr_rank_all).mean()
            logging.info(f"iter {iter_num}, val loss={loss:.4f}, val recall1={recall1:.4f}, val recall5={recall5:.4f}, val mrr={mrr:.4f}, time={time.time() - iter_time:.2f}s")
            if 'wandb' in log_tool:
                wandb.log({'loss/val': loss, 'metrics/val_recall1': recall1, 'metrics/val_recall5': recall5, 'metrics/val_mrr': mrr}, step=iter_num)
            if best_val_loss > loss:
                best_val_loss = loss
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_kwargs': model_kwargs,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                logging.info(f"Saving checkpoint to {ckpt_best_path}")
                torch.save(checkpoint, ckpt_best_path)
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_kwargs': model_kwargs,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            logging.info(f"Saving checkpoint to {ckpt_last_path}")
            torch.save(checkpoint, ckpt_last_path)

            with torch.no_grad():
                recall_1, recall_5, recall_1_eq = [], [], []
                recall_1_topk, recall_5_topk, recall_1_eq_topk = [], [], []
                for token_ids, token_ids_window, next_tokens in test_loader:
                    token_ids_window = token_ids_window.to(device)

                    indices1 = torch.stack([torch.cat([torch.ones_like(id[:1]), torch.cat([torch.zeros_like(id[1:-1, :-1]), torch.ones_like(id[1:-1, -1:])], dim=-1), torch.zeros_like(id[-1:])], dim=0) for id in token_ids_window], dim=0).bool()
                    indices2 = torch.stack([torch.cat([torch.zeros_like(id[:-1]), torch.cat([torch.zeros_like(id[-1:, :-1]), torch.ones_like(id[-1:, -1:])], dim=-1)], dim=0) for id in token_ids_window], dim=0).bool()
                    if ddp:
                        distance = model.module.get_distance(token_ids_window.view(-1, token_ids_window.shape[-1]), indices1, indices2, temperature=temperature)
                    else:
                         distance = model.get_distance(token_ids_window.view(-1, token_ids_window.shape[-1]), indices1, indices2, temperature=temperature)
                    distance = distance.cpu().detach()[..., 0]

                    for i_batch in range(len(token_ids)):
                        dis = distance[i_batch]
                        all_count = coo_array(((-dis).exp().numpy(), (token_ids[i_batch][-len(dis):].cpu().numpy(), [0] * len(dis))), shape=(vocab_size, 1)).toarray()[:, 0]
                        
                        recall_1.append(all_count.argmax().item() == next_tokens[i_batch].item())
                        recall_5.append(next_tokens[i_batch].item() in torch.Tensor(all_count).topk(5).indices.tolist())
                        recall_1_eq.append((all_count.max().item() == all_count[next_tokens[i_batch]]) and (all_count.min().item() != all_count[next_tokens[i_batch]]))

                        dis[dis.sort().indices[10:]] = torch.inf
                        all_count = coo_array(((-dis).exp().numpy(), (token_ids[i_batch][-len(dis):].cpu().numpy(), [0] * len(dis))), shape=(vocab_size, 1)).toarray()[:, 0]
                        
                        recall_1_topk.append(all_count.argmax().item() == next_tokens[i_batch].item())
                        recall_5_topk.append(next_tokens[i_batch].item() in torch.Tensor(all_count).topk(5).indices.tolist())
                        recall_1_eq_topk.append((all_count.max().item() == all_count[next_tokens[i_batch]]) and (all_count.min().item() != all_count[next_tokens[i_batch]]))
                    
                    if len(recall_1) > args.num_examples_test:
                        break
            recall1, recall5, recall1_eq = np.mean(recall_1), np.mean(recall_5), np.mean(recall_1_eq)
            recall1_topk, recall5_topk, recall1_eq_topk = np.mean(recall_1_topk), np.mean(recall_5_topk), np.mean(recall_1_eq_topk)
            logging.info(f"iter {iter_num}, test recall1={recall1:.4f}, test recall5={recall5:.4f}, test recall1_eq={recall1_eq:.4f}, time={time.time() - iter_time:.2f}s")
            logging.info(f"iter {iter_num}, test recall1_top10={recall1_topk:.4f}, test recall5_top10={recall5_topk:.4f}, test recall1_eq_top10={recall1_eq_topk:.4f}, time={time.time() - iter_time:.2f}s")
            if 'wandb' in log_tool:
                wandb.log({'metrics/test_recall1': recall1, 'metrics/test_recall5': recall5, 'metrics/test_recall1eq': recall1_eq}, step=iter_num)
                wandb.log({'metrics/test_recall1_top10': recall1_topk, 'metrics/test_recall5_top10': recall5_topk, 'metrics/test_recall1eq_top10': recall1_eq_topk}, step=iter_num)

            model.train()

        # termination conditions
        if max_iters is not None and iter_num >= max_iters:
            break
    
    if ddp:
        destroy_process_group()