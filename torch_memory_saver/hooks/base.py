from abc import ABC


class HookUtilBase(ABC):
    def get_path_binary(self):
        raise NotImplementedError

    def create_allocator(self):
        return None
