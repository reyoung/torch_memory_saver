from torch_memory_saver.hooks.base import HookUtilBase
from torch_memory_saver.utils import get_binary_path_from_package


class HookUtilModeTorch(HookUtilBase):
    def get_path_binary(self):
        return str(get_binary_path_from_package("torch_memory_saver_hook_mode_torch"))

    def create_allocator(self):
        return TODO
