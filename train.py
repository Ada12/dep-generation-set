import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import SubsetRandomSampler, DataLoader

import constants
from hyper_params import HyperParams
from module.seq2seq import Deps2deps
from my_dataset.json_dataset import JsonDataset
from utils.log_utils import LOGGER
from utils.net_helper import loss_func
from utils.shell_args import SHELL_ARGS


class TrainHelper(object):
    def __init__(self, shell_args):
        self.shell_args = shell_args
        self.opts = HyperParams.from_shell_args(self.shell_args)
        self.opts.dump(os.path.join(constants.OUTPUT_DIR, 'hyper_params.json'))

        self.dataset = JsonDataset(source_path=self.opts.train_source_path,
                                   target_path=self.opts.train_target_path,
                                   vocab_path=self.opts.vocab_path,
                                   tag_path=self.opts.tag_path,
                                   tag_vocab_path=self.opts.tag_vocab_path)
        self.train_loader, self.valid_loader = self._get_data_loader(self.dataset)

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

        self.criterion = loss_func(self.opts)
        self.optimizer = optim.Adam(self.deps2deps.parameters(), lr=self.opts.lr)

        self.global_step = 0
        self.start_epoch = 0
        self.epoch = 0

        self.writer = SummaryWriter(os.path.join(constants.OUTPUT_DIR, 'summary'))
        self._saved_models = []

        if self.opts.load_model_path:
            self.load_model(self.opts.load_model_path)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def _get_data_loader(self, dataset):
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        if not self.opts.use_valid:
            # Use all data instances to train.
            train_sampler = SubsetRandomSampler(indices)
            train_loader = DataLoader(self.dataset, batch_size=self.opts.batch_size, sampler=train_sampler,
                                      collate_fn=self.dataset.collate_fn)
            test_loader = None
        else:
            split = int(dataset_size // self.opts.batch_size * 0.2) * self.opts.batch_size
            train_indices, test_indices = indices[split:], indices[: split]
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(self.dataset, batch_size=self.opts.batch_size, sampler=train_sampler,
                                      collate_fn=self.dataset.collate_fn)
            test_loader = DataLoader(self.dataset, batch_size=self.opts.batch_size, sampler=test_sampler,
                                     collate_fn=self.dataset.collate_fn)

        return train_loader, test_loader

    def save_model(self):
        model_path = os.path.join(constants.MODELS_DIR, 'model-{}.pth'.format(self.global_step))
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.deps2deps.state_dict(),
        }, model_path)

        if len(self._saved_models) >= 5:
            os.remove(self._saved_models[0])
            self._saved_models = self._saved_models[1:]
        self._saved_models.append(model_path)

    def load_model(self, load_model_path):
        checkpoint = torch.load(load_model_path)
        self.global_step = checkpoint['global_step']
        self.start_epoch = checkpoint['epoch'] + 1
        self.epoch = self.start_epoch

        self.deps2deps.load_state_dict(checkpoint['model_state_dict'])
        self.deps2deps.train()

    def train_batch(self, source_vec, source_tags_vec, target_vec):
        self.deps2deps.train()
        self.optimizer.zero_grad()

        source_vec = source_vec.to(constants.DEVICE)
        target_vec = target_vec.to(constants.DEVICE)
        source_tags_vec = source_tags_vec.to(constants.DEVICE)

        outputs = self.deps2deps(source_vec, source_tags_vec)
        loss = self.criterion(outputs, target_vec)

        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        self.writer.add_scalars('loss', {'training': loss}, self.global_step)

        return float(loss)

    def valid_batch(self, source_vec, source_tags_vec, target_vec):
        self.deps2deps.eval()
        self.optimizer.zero_grad()

        source_vec = source_vec.to(constants.DEVICE)
        target_vec = target_vec.to(constants.DEVICE)
        source_tags_vec = source_tags_vec.to(constants.DEVICE)

        outputs = self.deps2deps(source_vec, source_tags_vec)
        loss = self.criterion(outputs, target_vec)

        self.writer.add_scalars('loss', {'valid': loss}, self.global_step)

        return float(loss), outputs

    def train(self):
        LOGGER.info('shell_args: {}'.format(self.shell_args))
        LOGGER.info('HyperParams: {}'.format(self.opts))
        LOGGER.info('Dataset total: {}'.format(len(self.dataset)))
        LOGGER.info('Model: {}'.format(self.deps2deps))
        LOGGER.info('Train Start:')

        best_loss = 1e12
        bad_times = 0
        for self.epoch in range(self.start_epoch, self.opts.epoch_max):
            # Train.
            train_loss_sum = 0
            train_batches_num = 0
            for batch_i, (source_vec, target_vec, source_tags_vec) in enumerate(self.train_loader):
                loss = self.train_batch(source_vec, source_tags_vec, target_vec)
                train_loss_sum += loss
                train_batches_num += 1

                if batch_i % 100 == 0:
                    LOGGER.info('Epoch#{}, Batch#{}, training loss: {}'.format(self.epoch, batch_i, loss))
            train_loss_avg = train_loss_sum / train_batches_num
            self.writer.add_scalars('avg-loss', {'training': train_loss_avg}, self.epoch)

            if not self.opts.use_valid:
                # Use train loss to early stop if use_valid is False.
                valid_loss_avg = train_loss_avg
            else:
                # Valid.
                valid_loss_sum = 0
                valid_batches_num = 0
                for batch_i, (source_vec, target_vec, source_tags_vec) in enumerate(self.valid_loader):
                    loss, outputs = self.valid_batch(source_vec, source_tags_vec, target_vec)
                    valid_loss_sum += loss
                    valid_batches_num += 1

                    if batch_i % 100 == 0:
                        LOGGER.info('Epoch#{}, Batch#{}, valid loss: {}'.format(self.epoch, batch_i, loss))
                        LOGGER.info('Outputs: {}'.format(outputs[:5].topk(5, dim=1)))
                        LOGGER.info('Targets: {}'.format(target_vec[:5].topk(5, dim=1)[1]))
                valid_loss_avg = valid_loss_sum / valid_batches_num
                self.writer.add_scalars('avg-loss', {'valid': valid_loss_avg}, self.epoch)

            LOGGER.info('Epoch#{} finished, Avg train loss: {}, Avg valid loss: {}'.format(
                self.epoch, train_loss_avg, valid_loss_avg))
            if valid_loss_avg - best_loss > 1e-5:
                bad_times += 1
                LOGGER.warning('Not the best valid loss: {}, best_loss: {}, bad_times: {}'.format(
                    valid_loss_avg, best_loss, bad_times))
                if bad_times >= 10:
                    break
            else:
                self.save_model()
                best_loss = valid_loss_avg
                bad_times = 0


def main():
    helper = TrainHelper(SHELL_ARGS)
    helper.train()


if __name__ == '__main__':
    main()
