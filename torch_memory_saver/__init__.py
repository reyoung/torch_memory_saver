import ctypes
import logging
import os
from abc import ABC
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class TorchMemorySaver:
    def __init__(self):
        self._impl: _TorchMemorySaverImplBase = \
            _TorchMemorySaverImplNormal() if _global_info.enabled else _TorchMemorySaverImplNoop()

    @contextmanager
    def region(self):
        with self._impl.region():
            yield

    def pause(self):
        self._impl.pause()

    def resume(self):
        self._impl.resume()


class _TorchMemorySaverImplBase(ABC):
    def region(self):
        raise NotImplementedError

    def pause(self):
        raise NotImplementedError

    def resume(self):
        raise NotImplementedError


class _TorchMemorySaverImplNormal(_TorchMemorySaverImplBase):
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


class _TorchMemorySaverImplNoop(_TorchMemorySaverImplBase):
    @contextmanager
    def region(self):
        yield

    def pause(self):
        pass

    def resume(self):
        pass


def _compute_cdll():
    env_ld_preload = os.environ.get('LD_PRELOAD', '')
    if 'torch_memory_saver' in env_ld_preload:
        return ctypes.CDLL(env_ld_preload)
    else:
        logger.warning(
            f'TorchMemorySaver is disabled for the current process because invalid LD_PRELOAD="{env_ld_preload}" (process_id={os.getpid()})')
        return None


class _GlobalInfo:
    def __init__(self):
        self.cdll: Optional[ctypes.CDLL] = _compute_cdll()
        self._last_id = 0
        logger.debug(f'Use cdll={self.cdll}')

    @property
    def enabled(self):
        return self.cdll is not None

    def next_id(self):
        self._last_id += 1
        return self._last_id


_global_info = _GlobalInfo()


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
