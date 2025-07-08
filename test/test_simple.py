import pytest

from test.utils import configure_tms_and_run_in_subprocess
from .examples import simple

_HOOK_MODES = ["preload", "torch"]


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_simple(hook_mode):
    configure_tms_and_run_in_subprocess(simple.run, hook_mode=hook_mode)
