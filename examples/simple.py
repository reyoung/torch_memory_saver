import logging
import sys
import time

import torch

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from torch_memory_saver import TorchMemorySaver

import os

memory_saver = TorchMemorySaver()

normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

with memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# Get the virtual address
original_address = pauseable_tensor.data_ptr()
print(f"Tensor virtual address: 0x{original_address:x}")
os.system("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")

print(f'{normal_tensor=} {pauseable_tensor=}')

print('sleep...')
time.sleep(3)

memory_saver.pause()
os.system("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")

print('sleep...')
time.sleep(3)

memory_saver.resume()
os.system("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")

new_address = pauseable_tensor.data_ptr()
print(f"Tensor virtual address: 0x{new_address:x}")

assert original_address == new_address, 'Tensor virtual address should be the same'

print('sleep...')
time.sleep(3)

print(f'{normal_tensor=} {pauseable_tensor=}')
