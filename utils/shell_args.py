import argparse
import ast


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "{}" is not a list'.format(s))
    return v


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--out_dir', type=str, default='test')

    parser.add_argument('--train_source_path', type=str, default='static/input/train_seqs.json')
    parser.add_argument('--train_target_path', type=str, default='')
    parser.add_argument('--test_source_path', type=str, default='static/input/test_source_seqs.json')
    parser.add_argument('--test_target_path', type=str, default='static/input/test_target_seqs.json')
    parser.add_argument('--vocab_path', type=str, default='static/input/dep2id.json')
    parser.add_argument('--tag_path', type=str, default='static/input/dep_id2tag_ids.json')
    parser.add_argument('--tag_vocab_path', type=str, default='static/input/tag2id.json')
    parser.add_argument('--load_model_path', type=str, default='')

    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--tags_linear_sizes', type=arg_as_list, default=[128])
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--hidden_linear_sizes', type=arg_as_list, default=[128, 512])
    parser.add_argument('--loss_func', type=str, choices=['ce', 'mce'], default='ce')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch_max', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--use_valid', type=bool, default=False)
    parser.add_argument('--use_tags', type=bool, default=False)


_parser = argparse.ArgumentParser()
add_arguments(_parser)
SHELL_ARGS, _ = _parser.parse_known_args()
