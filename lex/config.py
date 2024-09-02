from os.path import abspath, dirname

import torch.cuda

ROOT = dirname(abspath(__file__))


def get_device(rank=None):
    if torch.cuda.is_available():
        if rank is None:
            return "cuda"
        return f"cuda:{rank}"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
