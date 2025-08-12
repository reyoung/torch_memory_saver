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

    TODO_global_enable_cpu_backup

    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    initial_tensor = torch.full((1_000_000,), 42, dtype=torch.uint8, device='cuda')
    mem_initial = get_and_print_gpu_memory("Init")

    model_weights = [
        torch.full((size,), 42, dtype=torch.uint8, device="cuda")
        for size in [1024 ** 3, 1024 ** 2, 1024 ** 1, 42]
    ]

    _execute_forward_pass_and_assert(model_weights)
    mem_after_forward_pass = get_and_print_gpu_memory("After forward pass")
    assert mem_after_forward_pass > mem_initial + 5 * 1024 ** 3

    torch_memory_saver.pause()
    mem_after_pause = get_and_print_gpu_memory("After pause")
    assert mem_after_pause < mem_initial - 1_000_000

    with torch_memory_saver.disable():
        mem_after_disable = get_and_print_gpu_memory("After disable")
        assert mem_after_disable == mem_after_pause

        # Can still execute code in disabled region
        tensor_in_disabled_region = torch.full((1024 ** 3,), 53, dtype=torch.uint8, device='cuda')
        out = tensor_in_disabled_region.float().mean().item()
        assert out == 53, f"{out=}"

        mem_after_exec_in_disable = get_and_print_gpu_memory("After exec in disable")
        assert mem_after_exec_in_disable > mem_after_disable + 4 * 1024 ** 3

    # should do cleanup
    mem_after_exit_disable = get_and_print_gpu_memory("After existing disable")
    assert mem_after_exit_disable <= mem_after_pause + 1024 ** 2

    TODO

    del initial_tensor


def _execute_forward_pass_and_assert(weights: List[torch.Tensor]):
    # simulate large activations during forward pass
    ones = torch.ones((1024 ** 3,), dtype=torch.float32, device="cuda")
    sum_avg_weights = reduce(lambda a, b: a + b, [w.float().mean() for w in weights])
    outs = ones * sum_avg_weights
    out = outs.mean().item()
    assert out == 42, f"{out=}"


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
