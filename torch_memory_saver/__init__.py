import ctypes
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

@dataclass
class _BinaryInfo:
    cdll: Optional[ctypes.CDLL]

    @property
    def enabled(self):
        return self.cdll is not None

    @staticmethod
    def _setup_function_signatures(cdll):
        """Define function signatures for the C library"""
        cdll.tms_set_current_tag.argtypes = [ctypes.c_char_p]
        cdll.tms_set_interesting_region.argtypes = [ctypes.c_bool]
        cdll.tms_set_enable_cpu_backup.argtypes = [ctypes.c_bool]
        cdll.tms_pause.argtypes = [ctypes.c_char_p]
        cdll.tms_resume.argtypes = [ctypes.c_char_p]

    @staticmethod
    def compute():
        env_ld_preload = os.environ.get("LD_PRELOAD", "")
        if "torch_memory_saver" in env_ld_preload:
            try:
                cdll = ctypes.CDLL(env_ld_preload)
                _BinaryInfo._setup_function_signatures(cdll)
                return _BinaryInfo(cdll=cdll)
            except OSError as e:
                logger.error(f"Failed to load CDLL from {env_ld_preload}: {e}")
                return _BinaryInfo(cdll=None)
        else:
            print(
                f"TorchMemorySaver is disabled for the current process because invalid LD_PRELOAD. "
                f"You can use configure_subprocess() utility, "
                f"or directly specify `LD_PRELOAD=/path/to/torch_memory_saver_cpp.some-postfix.so python your_script.py. "
                f'(LD_PRELOAD="{env_ld_preload}" process_id={os.getpid()})'
            )
            return _BinaryInfo(cdll=None)


class _GlobalInfo:
    def __init__(self):
        self._binary_info: Optional[_BinaryInfo] = None

    @property
    def binary_info(self):
        if self._binary_info is None:
            self._binary_info = _BinaryInfo.compute()
        return self._binary_info


_global_info = _GlobalInfo()

# Global singleton instance
torch_memory_saver = TorchMemorySaver()

