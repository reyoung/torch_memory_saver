import logging
import sys
import time
import os

import torch

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from torch_memory_saver import torch_memory_saver
from examples.util import print_gpu_memory_gb

normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

with torch_memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

original_address = pauseable_tensor.data_ptr()
print(f"Pauseable tensor virtual address: 0x{original_address:x}")

print_gpu_memory_gb("Before pause")

print(f'{normal_tensor=} {pauseable_tensor=}')

print('sleep...')
time.sleep(3)

torch_memory_saver.pause()
print_gpu_memory_gb("After pause")

print('sleep...')
time.sleep(3)

torch_memory_saver.resume()
print_gpu_memory_gb("After resume")

new_address = pauseable_tensor.data_ptr()
print(f"Pauseable tensor virtual address: 0x{new_address:x}")

assert original_address == new_address, 'Tensor virtual address should be the same'

print('sleep...')
time.sleep(3)

print(f'{normal_tensor=} {pauseable_tensor=}')

# exit this process gracefully, bypassing CUDA cleanup
# Checkout for more details: https://github.com/fzyzcjy/torch_memory_saver/pull/18 
os._exit(0)
