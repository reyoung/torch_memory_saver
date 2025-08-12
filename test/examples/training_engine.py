import logging
import os
import sys
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory


def run(hook_mode: str):
    assert os.environ["TMS_INIT_ENABLE"] == "1"
    assert hook_mode == "preload"

    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    TODO


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
