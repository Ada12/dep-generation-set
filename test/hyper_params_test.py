from hyper_params import HyperParams
from utils.shell_args import SHELL_ARGS


def test_from_shell_args():
    opts = HyperParams.from_shell_args(SHELL_ARGS)
    print(opts.get_dict())

