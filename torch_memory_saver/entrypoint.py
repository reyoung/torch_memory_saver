import ctypes
import logging
import os
from contextlib import contextmanager
from typing import Optional
import torch

from .binary_wrapper import BinaryWrapper

logger = logging.getLogger(__name__)


class TorchMemorySaver:
    def __init__(self):
        self._impl: Optional[_TorchMemorySaverImpl] = None

    @contextmanager
    def region(self, tag: str = "default", enable_cpu_backup: bool = False):
        """Context manager for memory saving with optional tag"""
        self._ensure_initialized()
        self._impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup)

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
            self._binary_wrapper.cdll.tms_set_current_tag(tag.encode("utf-8"))
            self._binary_wrapper.cdll.tms_set_interesting_region(True)
            self._binary_wrapper.cdll.tms_set_enable_cpu_backup(enable_cpu_backup)
            try:
                yield
            finally:
                self._binary_wrapper.cdll.tms_set_current_tag(b"default")
                self._binary_wrapper.cdll.tms_set_interesting_region(False)
                self._binary_wrapper.cdll.tms_set_enable_cpu_backup(False)

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
