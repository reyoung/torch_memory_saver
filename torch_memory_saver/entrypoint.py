import ctypes
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class TorchMemorySaver:
    def __init__(self):
        self._mem_pool = None

        TODO_move_check_to_real_init
        if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            raise RuntimeError(
                "TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet (please create an issue if you need it)"
            )

    @contextmanager
    def region(self, tag: str = "default", enable_cpu_backup: bool = False):
        """Context manager for memory saving with optional tag"""
        if _global_info.binary_info.enabled:
            self._ensure_mem_pool()
            with torch.cuda.use_mem_pool(self._mem_pool):
                _global_info.binary_info.cdll.tms_set_current_tag(tag.encode("utf-8"))
                _global_info.binary_info.cdll.tms_set_interesting_region(True)
                _global_info.binary_info.cdll.tms_set_enable_cpu_backup(enable_cpu_backup)
                try:
                    yield
                finally:
                    _global_info.binary_info.cdll.tms_set_current_tag(b"default")
                    _global_info.binary_info.cdll.tms_set_interesting_region(False)
                    _global_info.binary_info.cdll.tms_set_enable_cpu_backup(False)
        else:
            yield

    def pause(self, tag: Optional[str] = None):
        """Pause memory for specific tag or all memory if tag is None"""
        if _global_info.binary_info.enabled:
            tag_bytes = tag.encode("utf-8") if tag else None
            _global_info.binary_info.cdll.tms_pause(tag_bytes)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None"""
        if _global_info.binary_info.enabled:
            tag_bytes = tag.encode("utf-8") if tag else None
            _global_info.binary_info.cdll.tms_resume(tag_bytes)

    @property
    def enabled(self):
        return _global_info.binary_info.enabled

    def _ensure_mem_pool(self):
        if self._mem_pool is None:
            self._mem_pool = torch.cuda.MemPool()

