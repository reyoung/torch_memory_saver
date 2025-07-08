import logging
import multiprocessing
import sys
import traceback
import subprocess


def print_gpu_memory(message, gpu_id=0):
    """Print GPU memory usage with optional message"""
    cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id={gpu_id}"
    result = subprocess.check_output(cmd, shell=True, text=True).strip()
    print(f"GPU {gpu_id} memory: {int(result) / 1024:.2f} GB ({message})")


def run_in_subprocess(fn):
    ctx = multiprocessing.get_context('spawn')
    output_queue = ctx.Queue()
    proc = ctx.Process(target=_subprocess_fn_wrapper, args=(fn, output_queue,))
    proc.start()
    proc.join()
    success = output_queue.get()
    assert success


def _subprocess_fn_wrapper(fn, output_queue):
    try:
        print(f"Subprocess execution start")
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
        fn()
        print(f"Subprocess execution end", flush=True)
        output_queue.put(True)
    except Exception as e:
        print(f"Subprocess has error: {e}", flush=True)
        traceback.print_exc()
        output_queue.put(False)
        raise
