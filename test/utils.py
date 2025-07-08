import multiprocessing
import traceback


def run_in_subprocess(fn):
    output_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_subprocess_fn_wrapper, args=(fn, output_queue,))
    proc.start()
    proc.join()
    success = output_queue.get()
    assert success


def _subprocess_fn_wrapper(fn, output_queue):
    try:
        fn()
        output_queue.put(True)
    except Exception as e:
        print(f"subprocess has error: {e}", flush=True)
        traceback.print_exc()
        output_queue.put(False)
        raise
