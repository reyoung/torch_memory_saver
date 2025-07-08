import logging
import sys
import os

import torch

from torch_memory_saver import torch_memory_saver

with torch_memory_saver.region(enable_cpu_backup=True):
    tensor_with_backup = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

with torch_memory_saver.region(enable_cpu_backup=False):
    tensor_without_backup = torch.full((1_000_000,), 200, dtype=torch.uint8, device='cuda')

assert tensor_with_backup[:3].tolist() == [100, 100, 100]
assert tensor_without_backup[:3].tolist() == [200, 200, 200]

torch_memory_saver.pause()
torch_memory_saver.resume()

assert tensor_with_backup[:3].tolist() == [100, 100, 100]
assert tensor_without_backup[:3].tolist() != [200, 200, 200]

# exit this process gracefully, bypassing CUDA cleanup
# Checkout for more details: https://github.com/fzyzcjy/torch_memory_saver/pull/18 
os._exit(0)
