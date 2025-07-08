import ctypes
import logging
import os
from contextlib import contextmanager
from typing import Optional
import torch

from .binary_wrapper import BinaryWrapper

logger = logging.getLogger(__name__)

_TAG_DEFAULT = "default"


class TorchMemorySaver:
    def __init__(self):
        self._impl: Optional[_TorchMemorySaverImpl] = None

    @contextmanager
    def region(self, tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False):
        """Context manager for memory saving with optional tag"""
        self._ensure_initialized()
        with self._impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup):
            yield

    def pause(self, tag: Optional[str] = None):
        """Pause memory for specific tag or all memory if tag is None"""
        self._impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None"""
        self._impl.resume(tag=tag)

    # for compatibility
    @property
    def enabled(self):
        return True

    def _ensure_initialized(self):
        if self._impl is None:
            self._impl = _TorchMemorySaverImpl()


class _TorchMemorySaverImpl:
    def __init__(self):
        self._binary_wrapper = BinaryWrapper.compute()
        self._mem_pool = torch.cuda.MemPool()
        _sanity_checks()

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool):
        with torch.cuda.use_mem_pool(self._mem_pool):
            self._binary_wrapper.set_config(tag=tag, interesting_region=True, enable_cpu_backup=enable_cpu_backup)
            try:
                yield
            finally:
                self._binary_wrapper.set_config(tag=_TAG_DEFAULT, interesting_region=False, enable_cpu_backup=False)

    def pause(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_pause(tag_bytes)

    def resume(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_resume(tag_bytes)


def _sanity_checks():
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError(
            "TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet."
        )
