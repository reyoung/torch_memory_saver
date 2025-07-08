import logging
import sys
import time
import os
from typing import Callable

import torch
from torch_memory_saver import torch_memory_saver

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

dummy_tensor_size = (5, 100_000_000,)


def _ptr(x):
    assert isinstance(x, torch.Tensor)
    return hex(x.data_ptr())


class KVCache:
    def __init__(self):
        self.create_buffers(1)

    def create_buffers(self, value):
        with torch_memory_saver.region(tag="kv_cache"):
            # or model weights, etc
            self.kv_buffer = torch.full(dummy_tensor_size, value, dtype=torch.float32, device='cuda')
        print(f'create_buffers {_ptr(self.kv_buffer)=}')

    def clear_buffers(self):
        del self.kv_buffer

    def execute(self, arg: torch.Tensor) -> torch.Tensor:
        # print(f'KVCache.execute {arg=} {self.kv_buffer=}')
        return (arg + self.kv_buffer.mean(dim=1)).mean()


# https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
def create_cuda_graph(fn: Callable):
    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        print('with torch.cuda.stream(s) execute fn')
        fn()
    torch.cuda.current_stream().wait_stream(s)

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        print('with torch.cuda.graph(g) execute fn')
        fn()

    return g


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    cache = KVCache()
    static_input = torch.zeros((5,), dtype=torch.float32, device='cuda')
    static_output = torch.zeros((5,), dtype=torch.float32, device='cuda')
    print(f'{_ptr(static_input)=} {_ptr(static_output)=}')

    def fn():
        nonlocal static_output
        static_output = cache.execute(static_input)

    g = create_cuda_graph(fn)

    print('replay #1')
    static_input[...] = 100
    g.replay()
    print(f'{static_output=}')
    assert static_output == 101, f'{static_output=}'

    print('torch.cuda.empty_cache()')
    torch.cuda.empty_cache()

    print('sleep...')
    time.sleep(1)

    print('call memory_saver.pause("kv_cache")')
    torch_memory_saver.pause("kv_cache")

    print('sleep...')
    time.sleep(1)

    print('when kv cache is released, we can allocate *other* big tensors')
    other_big_tensor = torch.zeros((2500_000_000,), dtype=torch.uint8, device='cuda')
    print('sleep...')
    time.sleep(1)
    print(f'{other_big_tensor=}')
    del other_big_tensor
    torch.cuda.empty_cache()
    print('sleep...')
    time.sleep(1)

    print('call memory_saver.resume("kv_cache")')
    torch_memory_saver.resume("kv_cache")

    dummy = torch.zeros((3,), device='cuda')
    print(f'{_ptr(dummy)=}')

    cache.kv_buffer[...] = 2

    print('replay #2')
    static_input[...] = 200
    g.replay()
    print(f'{static_output=}')
    assert static_output == 202, f'{static_output=}'

    print('sleep...')
    time.sleep(1)

    print(f'{dummy=}')


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
