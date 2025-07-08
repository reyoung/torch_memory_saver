import logging
import os
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

@contextmanager
def configure_subprocess():
    with _change_env("LD_PRELOAD", str(_get_binary_path())):
        yield


def _get_binary_path():
    dir_package = Path(__file__).parent
    TODO_handle_so_postfix
    candidates = [
        p
        for d in [dir_package, dir_package.parent]
        for p in d.glob("torch_memory_saver_cpp.*.so")
    ]
    assert (
            len(candidates) == 1
    ), f"Expected exactly one torch_memory_saver_cpp library, found: {candidates}"
    return candidates[0]


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
