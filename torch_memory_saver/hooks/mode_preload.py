import logging
import os
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def configure_subprocess():
    """Configure environment variables for subprocesses. Only needed for hook_mode=preload."""
    with _change_env("LD_PRELOAD", str(_get_binary_path_from_package())):
        yield


# TODO move?
def _get_binary_path_from_package():
    dir_package = Path(__file__).parent
    candidates = [
        p
        for d in [dir_package, dir_package.parent]
        for p in d.glob("torch_memory_saver_hook_mode_preload.*.so")
    ]
    assert (
            len(candidates) == 1
    ), f"Expected exactly one torch_memory_saver_cpp library, found: {candidates}"
    return candidates[0]


def get_binary_path_from_env():
    env_ld_preload = os.environ.get("LD_PRELOAD", "")
    assert "torch_memory_saver" in env_ld_preload, (
        f"TorchMemorySaver observes invalid LD_PRELOAD. "
        f"You can use configure_subprocess() utility, "
        f"or directly specify `LD_PRELOAD=/path/to/torch_memory_saver_cpp.some-postfix.so python your_script.py. "
        f'(LD_PRELOAD="{env_ld_preload}" process_id={os.getpid()})'
    )
    return env_ld_preload


@contextmanager
def _change_env(key: str, value: str):
    old_value = os.environ.get(key, "")
    os.environ[key] = value
    logger.debug(f"change_env set key={key} value={value}")
    try:
        yield
    finally:
        assert os.environ[key] == value
        os.environ[key] = old_value
        logger.debug(f"change_env restore key={key} value={old_value}")
