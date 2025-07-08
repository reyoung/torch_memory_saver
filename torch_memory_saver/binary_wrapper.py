import ctypes
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class BinaryWrapper:
    cdll: Optional[ctypes.CDLL]

    @property
    def enabled(self):
        return self.cdll is not None

    @staticmethod
    def compute():
        TODO_refactor_this
        TODO_throw_when_fail
        env_ld_preload = os.environ.get("LD_PRELOAD", "")
        if "torch_memory_saver" in env_ld_preload:
            try:
                cdll = ctypes.CDLL(env_ld_preload)
                _setup_function_signatures(cdll)
                return BinaryWrapper(cdll=cdll)
            except OSError as e:
                logger.error(f"Failed to load CDLL from {env_ld_preload}: {e}")
                return BinaryWrapper(cdll=None)
        else:
            print(
                f"TorchMemorySaver is disabled for the current process because invalid LD_PRELOAD. "
                f"You can use configure_subprocess() utility, "
                f"or directly specify `LD_PRELOAD=/path/to/torch_memory_saver_cpp.some-postfix.so python your_script.py. "
                f'(LD_PRELOAD="{env_ld_preload}" process_id={os.getpid()})'
            )
            return BinaryWrapper(cdll=None)

    def set_config(self, *, tag: str, interesting_region: bool, enable_cpu_backup: bool):
        self.cdll.tms_set_current_tag(tag.encode("utf-8"))
        self.cdll.tms_set_interesting_region(interesting_region)
        self.cdll.tms_set_enable_cpu_backup(enable_cpu_backup)


def _setup_function_signatures(cdll):
    """Define function signatures for the C library"""
    cdll.tms_set_current_tag.argtypes = [ctypes.c_char_p]
    cdll.tms_set_interesting_region.argtypes = [ctypes.c_bool]
    cdll.tms_set_enable_cpu_backup.argtypes = [ctypes.c_bool]
    cdll.tms_pause.argtypes = [ctypes.c_char_p]
    cdll.tms_resume.argtypes = [ctypes.c_char_p]
