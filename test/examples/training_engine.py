import logging
import os
import sys
from functools import reduce
from typing import List

import torch
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory


def run(hook_mode: str):
    assert os.environ["TMS_INIT_ENABLE"] == "1"
    assert hook_mode == "preload"

    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    initial_tensor = torch.full((1_000_000,), 42, dtype=torch.uint8, device='cuda')
    mem_initial = get_and_print_gpu_memory("Init")

    model_weights = [
        torch.full((size,), 42, dtype=torch.uint8, device="cuda")
        for size in [1024 ** 3, 1024 ** 2, 1024 ** 1, 42]
    ]

    _execute_forward_pass_and_assert(model_weights)

    TODO


def _execute_forward_pass_and_assert(model_weights: List[torch.Tensor]):
    ones = torch.ones((1024 ** 3,), dtype=torch.float32, device="cuda")
    sum_avg_weights = reduce(lambda a, b: a + b, [w.float().mean() for w in model_weights])
    outs = ones * sum_avg_weights
    out = outs.mean().item()
    assert out == 42, f"{out=}"


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
