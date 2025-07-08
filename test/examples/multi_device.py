import logging
import sys

import torch

from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    with torch_memory_saver.region(enable_cpu_backup=True):
        dev0_a = torch.full((100_000_000,), 10, dtype=torch.uint8, device='cuda')
        dev1_a = torch.full((100_000_000,), 10, dtype=torch.uint8, device='cuda:1')

        torch.cuda.set_device(1)
        dev0_b = torch.full((100_000_000,), 10, dtype=torch.uint8, device='cuda:0')
        dev1_b = torch.full((100_000_000,), 10, dtype=torch.uint8, device='cuda')

    torch_memory_saver.pause()
    torch_memory_saver.resume()


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
