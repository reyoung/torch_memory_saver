from setuptools import setup
from torch.utils import cpp_extension

ext_module = cpp_extension.CppExtension(
    'torch_memory_saver_cpp',
    ['csrc/torch_memory_saver.cpp'],
    extra_compile_args=['-I/usr/local/cuda/include'],
    extra_link_args=['-lcuda'],
)

setup(
    name='torch_memory_saver',
    version='0.0.1',
    # https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-c-extension
    ext_modules=[ext_module],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    python_requires=">=3.9",
    packages=['torch_memory_saver'],
)
