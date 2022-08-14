import os

import numpy as np

import constants
from utils import file_utils


def random_char():
    return np.random.randint(0, 26) + 3


def random_seq(seq_len_min, seq_len_max):
    seq_len = np.random.randint(seq_len_min, seq_len_max)
    return [random_char() for _ in range(seq_len)]


def gen_simulate_dataset(items_num):
    source_seqs = []
    target_seqs = []
    for _ in range(items_num):
        source_seq = random_seq(5, 10)
        target_seq = list(reversed(source_seq))
        source_seqs.append(source_seq)
        target_seqs.append(target_seq)

    vocab = {
        constants.WORD_PADDING: 0,
        constants.WORD_START: 1,
        constants.WORD_END: 2,
    }
    for i in range(26):
        vocab[chr(ord('a') + i)] = len(vocab)

    file_utils.ensure_path_exist(os.path.join(constants.INPUT_DIR, 'simulate'))
    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'simulate', 'vocab.json'), vocab)
    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'simulate', 'train_source_seqs.json'), source_seqs)
    file_utils.dump_json(os.path.join(constants.INPUT_DIR, 'simulate', 'train_target_seqs.json'), target_seqs)


if __name__ == '__main__':
    gen_simulate_dataset(10000)
