import os

import torch
from torch.utils.data import DataLoader

import constants
from hyper_params import HyperParams
from module.seq2seq import Deps2deps
from my_dataset.json_dataset import JsonDataset
from utils import file_utils
from utils.evaluation_utils import recall_rate_score, recall_score, precision_score
from utils.log_utils import LOGGER
from utils.shell_args import SHELL_ARGS


def _get_latest_model_file(root_dir):
    files = os.listdir(root_dir)
    max_global_step = 0
    max_file_path = None
    for file in files:
        file_path = os.path.join(root_dir, file)
        if os.path.isfile(file_path) and 'model' in file_path:
            val = file.replace('model-', '').replace('.pth', '')
            val = int(val)
            if val > max_global_step:
                max_global_step = val
                max_file_path = file_path
    return max_file_path


class PredictHelper(object):
    def __init__(self, source_path, target_path):
        self.opts = HyperParams.from_disk(os.path.join(constants.OUTPUT_DIR, 'hyper_params.json'))

        self.dataset = JsonDataset(source_path=source_path,
                                   target_path=target_path,
                                   vocab_path=self.opts.vocab_path,
                                   tag_path=self.opts.tag_path,
                                   tag_vocab_path=self.opts.tag_vocab_path)
        self.data_loader = DataLoader(self.dataset, batch_size=1, collate_fn=self.dataset.collate_fn)

        self.deps2deps: Deps2deps = Deps2deps(vocab_size=self.dataset.vocab_size(),
                                              embed_size=self.opts.embed_size,
                                              tag_vocab_size=self.dataset.tag_vocab_size(),
                                              tags_linear_sizes=self.opts.tags_linear_sizes,
                                              dropout_p=self.opts.dropout_p,
                                              padding_idx=self.dataset.word2id(constants.WORD_PADDING),
                                              attn_hidden_size=self.opts.hidden_size,
                                              hidden_linear_sizes=self.opts.hidden_linear_sizes,
                                              use_tags=self.opts.use_tags)
        self.deps2deps = self.deps2deps.to(constants.DEVICE)

        if self.opts.load_model_path:
            self.load_model(self.opts.load_model_path)
        else:
            load_model_path = _get_latest_model_file(os.path.join(constants.MODELS_DIR))
            self.load_model(load_model_path)

    def load_model(self, load_model_path):
        LOGGER.info('Loading model from {}...'.format(load_model_path))

        checkpoint = torch.load(load_model_path, map_location=constants.DEVICE)

        self.deps2deps.load_state_dict(checkpoint['model_state_dict'])
        self.deps2deps.eval()

    def predict(self):
        out_list = []
        for source_vec, target_vec, source_tags_vec in self.data_loader:
            with torch.no_grad():
                out = self.deps2deps(source_vec, source_tags_vec)[0]
            idx_seq = out.topk(20)[1].tolist()
            out_list.append(idx_seq)
        return out_list


def main():
    helper = PredictHelper(SHELL_ARGS.test_source_path, SHELL_ARGS.test_target_path)
    out_list = helper.predict()
    file_utils.dump_json(os.path.join(constants.OUTPUT_DIR, 'predict.json'), out_list)

    source_list = file_utils.load_json(SHELL_ARGS.test_source_path)
    target_list = file_utils.load_json(SHELL_ARGS.test_target_path)

    k_list = [1, 5, 10]
    recall_rate_sums = [0] * len(k_list)
    recall_sums = [0] * len(k_list)
    precision_sums = [0] * len(k_list)
    for source, target, out, in zip(source_list, target_list, out_list):
        # Remove item in source.
        out = [item for item in out if item not in source]

        recall_10 = recall_score(target, out, 10)
        if recall_10 >= 0.4:
            LOGGER.info(
                'Select Source: {}\nTarget: {}\nOut: {}\n{}\n'.format([helper.dataset.id2word[idx] for idx in source],
                                                                      [helper.dataset.id2word[idx] for idx in target],
                                                                      [helper.dataset.id2word[idx] for idx in out],
                                                                      recall_10))

        for i, k in enumerate(k_list):
            recall_rate_sums[i] += recall_rate_score(target, out, k)
            recall_sums[i] += recall_score(target, out, k)
            precision_sums[i] += precision_score(target, out, k)

    for k, recall_rate_sum in zip(k_list, recall_rate_sums):
        LOGGER.info('RecallRate@{}: {}'.format(k, recall_rate_sum / len(target_list)))

    for k, recall_sum in zip(k_list, recall_sums):
        LOGGER.info('Recall@{}: {}'.format(k, recall_sum / len(target_list)))

    for k, precision_sum in zip(k_list, precision_sums):
        LOGGER.info('Precision@{}: {}'.format(k, precision_sum / len(target_list)))


if __name__ == '__main__':
    main()
