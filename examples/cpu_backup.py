import logging
import sys
import os

import torch

from torch_memory_saver import torch_memory_saver

with torch_memory_saver.region(enable_cpu_backup=True):
    tensor_with_backup = torch.full((1_000_000,), 10, dtype=torch.uint8, device='cuda')

with torch_memory_saver.region(enable_cpu_backup=False):
    tensor_without_backup = torch.full((1_000_000,), 20, dtype=torch.uint8, device='cuda')

print(f"{tensor_with_backup[:3]=} {tensor_without_backup[:3]=}")
assert tensor_with_backup[:3].tolist() == [10, 10, 10]
assert tensor_without_backup[:3].tolist() == [20, 20, 20]

torch_memory_saver.pause()

# occupy some space
tensor_unrelated = torch.full((2_000_000,), 30, dtype=torch.uint8, device='cuda')

torch_memory_saver.resume()

print(f"{tensor_with_backup[:3]=} {tensor_without_backup[:3]=}")
assert tensor_with_backup[:3].tolist() == [10, 10, 10]
assert tensor_without_backup[:3].tolist() != [20, 20, 20]

# exit this process gracefully, bypassing CUDA cleanup
# Checkout for more details: https://github.com/fzyzcjy/torch_memory_saver/pull/18 
os._exit(0)
