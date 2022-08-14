import os

import torch

from utils import file_utils
from utils.shell_args import SHELL_ARGS

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
INPUT_DIR = os.path.join(STATIC_DIR, 'input')

OUTPUT_DIR = os.path.join(STATIC_DIR, 'output', SHELL_ARGS.out_dir)
file_utils.ensure_path_exist(OUTPUT_DIR)

MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
file_utils.ensure_path_exist(MODELS_DIR)
LOGGING_FILENAME = os.path.join(OUTPUT_DIR, 'logs.txt')


GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if GPU_AVAILABLE else 'cpu')

# For preprocess.
WORD_FREQUENCY_MIN = 5
DEPS_NUM_MIN = 10
SPLIT_TEST_RATE = 0.2

# For seq2seq.
WORD_START = '<START>'
WORD_END = '<END>'
WORD_PADDING = '<PADDING>'
