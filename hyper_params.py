import json

from utils import file_utils


class HyperParams(object):
    def __init__(self,
                 embed_size=None,
                 hidden_size=None,
                 tags_linear_sizes=None,
                 dropout_p=None,
                 hidden_linear_sizes=None,
                 loss_func=None,
                 # For train.
                 batch_size=None,
                 epoch_max=None,
                 lr=None,
                 use_valid=None,
                 use_tags=None,
                 # For file path.
                 train_source_path=None,
                 train_target_path=None,
                 test_source_path=None,
                 test_target_path=None,
                 vocab_path=None,
                 tag_path=None,
                 tag_vocab_path=None,
                 load_model_path=None):
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.tags_linear_sizes = tags_linear_sizes
        self.dropout_p = dropout_p
        self.hidden_linear_sizes = hidden_linear_sizes
        self.loss_func = loss_func

        self.batch_size = batch_size
        self.epoch_max = epoch_max
        self.lr = lr
        self.use_valid = use_valid
        self.use_tags = use_tags

        self.train_source_path = train_source_path
        self.train_target_path = train_target_path
        self.test_source_path = test_source_path
        self.test_target_path = test_target_path
        self.vocab_path = vocab_path
        self.tag_path = tag_path
        self.tag_vocab_path = tag_vocab_path
        self.load_model_path = load_model_path

    def __str__(self):
        return json.dumps(self.get_dict(), indent=4)

    def get_dict(self):
        return {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}

    def dump(self, out_path):
        obj = self.get_dict()
        file_utils.dump_json(out_path, obj)

    @staticmethod
    def get_var_names():
        opts = HyperParams()
        obj = opts.get_dict()
        return tuple(obj.keys())

    @staticmethod
    def from_disk(in_path):
        obj = file_utils.load_json(in_path)
        return HyperParams(**obj)

    @staticmethod
    def from_shell_args(shell_args):
        obj = {var_name: getattr(shell_args, var_name) for var_name in HyperParams.get_var_names()}
        return HyperParams(**obj)


