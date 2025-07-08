from contextlib import nullcontext

import torch
import logging
import multiprocessing
import sys
import traceback
import torch_memory_saver


def get_and_print_gpu_memory(message, gpu_id=0):
    """Print GPU memory usage with optional message"""
    mem = torch.cuda.device_memory_used(gpu_id)
    print(f"GPU {gpu_id} memory: {mem / 1024 ** 3:.2f} GB ({message})")
    return mem


def configure_tms_and_run_in_subprocess(fn, hook_mode):
    ctx = torch_memory_saver.configure_subprocess() if hook_mode == "preload" else nullcontext()
    with ctx:
        run_in_subprocess(_configure_tms_and_run_in_subprocess_fn_wrapper, fn_kwargs=dict(fn=fn, hook_mode=hook_mode))


def _configure_tms_and_run_in_subprocess_fn_wrapper(fn, hook_mode):
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    torch_memory_saver.torch_memory_saver.hook_mode = hook_mode
    fn()


def run_in_subprocess(fn, fn_kwargs):
    ctx = multiprocessing.get_context('spawn')
    output_queue = ctx.Queue()
    proc = ctx.Process(target=_subprocess_fn_wrapper, args=(fn, fn_kwargs, output_queue))
    proc.start()
    proc.join()
    success = output_queue.get()
    assert success


def _subprocess_fn_wrapper(fn, fn_kwargs, output_queue):
    try:
        print(f"Subprocess execution start")
        fn(**fn_kwargs)
        print(f"Subprocess execution end (may see error messages when CUDA exit which is normal)", flush=True)
        output_queue.put(True)
    except Exception as e:
        print(f"Subprocess has error: {e}", flush=True)
        traceback.print_exc()
        output_queue.put(False)
        raise
