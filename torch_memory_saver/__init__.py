import ctypes
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class TorchMemorySaver:
    def __init__(self):
        self._mem_pool = torch.cuda.MemPool()
        self._id = _global_info.next_id()
        assert self._id == 1, 'Only support one single instance yet (multi-instance will be implemented later)'

    @contextmanager
    def region(self):
        with torch.cuda.use_mem_pool(self._mem_pool):
            _global_info.cdll.tms_region_enter()
            try:
                yield
            finally:
                _global_info.cdll.tms_region_leave()

    def pause(self):
        _global_info.cdll.tms_pause()

    def resume(self):
        _global_info.cdll.tms_resume()


class _GlobalInfo:
    def __init__(self):
        self._cdll: Optional[ctypes.CDLL] = None
        self._last_id = 0

    @property
    def cdll(self):
        if self._cdll is None:
            self._cdll = _compute_cdll()
            logger.debug(f'Use cdll={self._cdll}')
        return self._cdll

    def next_id(self):
        self._last_id += 1
        return self._last_id


_global_info = _GlobalInfo()


def _compute_cdll():
    env_ld_preload = os.environ.get('LD_PRELOAD', '')
    assert 'torch_memory_saver' in env_ld_preload, f'Please specify correct LD_PRELOAD (currently: {env_ld_preload})'
    return ctypes.CDLL(env_ld_preload)


def get_binary_path():
    dir_package = Path(__file__).parent
    candidates = [
        p
        for d in [dir_package, dir_package.parent]
        for p in d.glob('torch_memory_saver_cpp.*.so')
    ]
    assert len(candidates) == 1, f'{candidates=}'
    return candidates[0]


@contextmanager
def configure_subprocess():
    with change_env('LD_PRELOAD', str(get_binary_path())):
        yield


@contextmanager
def change_env(key: str, value: str):
    old_value = os.environ.get(key, '')
    os.environ[key] = value
    logger.debug(f'change_env set key={key} value={value}')
    try:
        yield
    finally:
        assert os.environ[key] == value
        os.environ[key] = old_value
        logger.debug(f'change_env restore key={key} value={old_value}')
