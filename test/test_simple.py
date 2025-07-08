import time
import os

from utils import run_in_subprocess, get_and_print_gpu_memory


def _test_simple_inner():
    import torch

    from torch_memory_saver import torch_memory_saver

    normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

    with torch_memory_saver.region():
        pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

    original_address = pauseable_tensor.data_ptr()
    print(f"Pauseable tensor virtual address: 0x{original_address:x}")
    print(f'{normal_tensor=} {pauseable_tensor=}')

    mem_before_pause = get_and_print_gpu_memory("Before pause")

    print('sleep...')
    time.sleep(3)

    torch_memory_saver.pause()
    mem_after_pause = get_and_print_gpu_memory("After pause")

    assert mem_before_pause - mem_after_pause > 0.9 * 1024 ** 3

    print('sleep...')
    time.sleep(3)

    torch_memory_saver.resume()
    get_and_print_gpu_memory("After resume")

    new_address = pauseable_tensor.data_ptr()
    print(f"Pauseable tensor virtual address: 0x{new_address:x}")

    assert original_address == new_address, 'Tensor virtual address should be the same'

    print('sleep...')
    time.sleep(3)

    print(f'{normal_tensor=} {pauseable_tensor=}')

    # TODO
    # # exit this process gracefully, bypassing CUDA cleanup
    # # Checkout for more details: https://github.com/fzyzcjy/torch_memory_saver/pull/18
    # os._exit(0)


def test_simple():
    run_in_subprocess(_test_simple_inner)
