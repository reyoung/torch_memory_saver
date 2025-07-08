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
        self._mem_pool = None
        self._binary_wrapper: Optional[BinaryWrapper] = None

        TODO_move_check_to_real_init
        if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            raise RuntimeError(
                "TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet (please create an issue if you need it)"
            )

    @contextmanager
    def region(self, tag: str = "default", enable_cpu_backup: bool = False):
        """Context manager for memory saving with optional tag"""
        if not self._binary_wrapper.enabled:
            yield
            return

        self._ensure_mem_pool()
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

    def pause(self, tag: Optional[str] = None):
        """Pause memory for specific tag or all memory if tag is None"""
        if not self._binary_wrapper.enabled:
            return

        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_pause(tag_bytes)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None"""
        if not self._binary_wrapper.enabled:
            return

        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_resume(tag_bytes)

    @property
    def enabled(self):
        return self._binary_wrapper.enabled

    def _ensure_mem_pool(self):
        if self._mem_pool is None:
            self._mem_pool = torch.cuda.MemPool()
