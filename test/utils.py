import os
import torch
import logging
import multiprocessing
import sys
import traceback


def get_and_print_gpu_memory(message, gpu_id=0):
    """Print GPU memory usage with optional message"""
    mem = torch.cuda.device_memory_used(gpu_id)
    print(f"GPU {gpu_id} memory: {mem / 1024 ** 3:.2f} GB ({message})")
    return mem


def run_in_subprocess(fn, exit_without_cleanup: bool):
    ctx = multiprocessing.get_context('spawn')
    output_queue = ctx.Queue()
    proc = ctx.Process(
        target=_subprocess_fn_wrapper,
        args=(fn, output_queue,),
        kwargs=dict(exit_without_cleanup=exit_without_cleanup),
    )
    proc.start()
    proc.join()
    success = output_queue.get()
    assert success


def _subprocess_fn_wrapper(fn, output_queue, exit_without_cleanup: bool):
    try:
        print(f"Subprocess execution start")
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

        fn()

        print(f"Subprocess execution end", flush=True)
        output_queue.put(True)

        if exit_without_cleanup:
            # Exit this process bypassing CUDA cleanup
            # Checkout for more details: https://github.com/fzyzcjy/torch_memory_saver/pull/18
            os._exit(0)
    except Exception as e:
        print(f"Subprocess has error: {e}", flush=True)
        traceback.print_exc()
        output_queue.put(False)
        raise
