import torch
from torch.utils.data import Dataset
import numpy as np

import constants
from utils import file_utils


class JsonDataset(Dataset):
    def __init__(self, source_path, vocab_path, tag_path, tag_vocab_path, target_path=None):
        """
        :param source_path:
        :param vocab_path:
        :param tag_path:
        :param target_path: 如果为 ''，则 collate_fn 中随机取一半作为 target.
        """
        self.source_seqs = file_utils.load_json(source_path)
        self.vocab = file_utils.load_json(vocab_path)
        self.id2word = {idx: word for word, idx in self.vocab.items()}
        self.dep_id2tag_ids = file_utils.load_json(tag_path)
        self.tag_vocab = file_utils.load_json(tag_vocab_path)
        if target_path:
            self.target_seqs = file_utils.load_json(target_path)
        else:
            self.target_seqs = None

    def collate_fn(self, batch_data):
        batch_size = len(batch_data)
        vocab_size = self.vocab_size()
        tag_vocab_size = self.tag_vocab_size()

        max_source_len = 0
        max_target_len = 0
        if self.target_seqs is None:
            for i, (source_seq, target_seq) in enumerate(batch_data):
                half = len(source_seq) // 2
                np.random.shuffle(source_seq)
                batch_data[i] = (source_seq[: half], source_seq[-half:])
                max_source_len = max(max_source_len, len(batch_data[i][0]))
                max_target_len = max(max_target_len, len(batch_data[i][1]))
        else:
            for source_seq, target_seq in batch_data:
                max_source_len = max(max_source_len, len(source_seq))
                max_target_len = max(max_target_len, len(target_seq))

        source_vec = torch.zeros(batch_size, max_source_len, dtype=torch.long, device=constants.DEVICE)
        target_vec = torch.zeros(batch_size, vocab_size, dtype=torch.float, device=constants.DEVICE)
        source_tags_vec = torch.zeros(batch_size, max_source_len, tag_vocab_size, dtype=torch.float,
                                      device=constants.DEVICE)

        for source_vec_row, target_vec_row, source_tags_vec_row, (source_seq, target_seq) in zip(
                source_vec, target_vec, source_tags_vec, batch_data):
            for i, source_word in enumerate(source_seq):
                source_vec_row[i] = source_word

            for target_word in target_seq:
                target_vec_row[target_word] = 1.

            for i, source_word in enumerate(source_seq):
                tag_ids = self.dep_id2tag_ids[str(source_word)]
                for tag_id in tag_ids:
                    source_tags_vec_row[i][tag_id] = 1.

        return source_vec, target_vec, source_tags_vec

    def vocab_size(self):
        return len(self.vocab)

    def tag_vocab_size(self):
        return len(self.tag_vocab)

    def word2id(self, word):
        return self.vocab[word]

    def tag_word2id(self, word):
        return self.tag_vocab[word]

    def __len__(self):
        return len(self.source_seqs)

    def __getitem__(self, index):
        if self.target_seqs is None:
            return self.source_seqs[index], None
        return self.source_seqs[index], self.target_seqs[index]
