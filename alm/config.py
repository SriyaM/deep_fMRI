from os.path import join, expanduser, dirname
import os
path_to_repo = dirname(dirname(os.path.abspath(__file__)))
if 'ALM_DIR' in os.environ:
    ALM_DIR = os.environ['ALM_DIR']
else:
    ALM_DIR = path_to_repo
DATA_DIR_ROOT = join(ALM_DIR, 'data')
SAVE_DIR_ROOT = join(ALM_DIR, 'results')

# individual datasets...
BABYLM_DATA_DIR = join(DATA_DIR_ROOT, 'babylm')

HDLM_EXP_DIR = join(ALM_DIR, 'hd_lm/experiments')

# Infini-gram for MechLM
INFINIGRAM_INDEX_PATH = join(ALM_DIR, 'infini-gram-index')

# hugging face token
TOKEN_HF = None