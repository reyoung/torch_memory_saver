import logging

import setuptools
from setuptools import setup

logger = logging.getLogger(__name__)

# try:
#     from torch.utils import cpp_extension
#
#     has_torch = True
# except ImportError:
#     logger.warning('setup.py fail to import torch, thus do not build C++ modules')
#     has_torch = False

# if has_torch:
# https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-c-extension
ext_modules = [setuptools.Extension(
    'torch_memory_saver_cpp',
    ['csrc/torch_memory_saver.cpp'],
    extra_compile_args=['-I/usr/local/cuda/include'],
    extra_link_args=['-lcuda'],
)]
# cmdclass = {'build_ext': cpp_extension.BuildExtension}
cmdclass = {}  # TODO
# else:
#     ext_modules = []
#     cmdclass = {}

setup(
    name='torch_memory_saver',
    version='0.0.1',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    packages=['torch_memory_saver'],
)
