import logging

import setuptools
from setuptools import setup

logger = logging.getLogger(__name__)

setup(
    name='torch_memory_saver',
    version='0.0.2',
    ext_modules=[setuptools.Extension(
        'torch_memory_saver_cpp',
        ['csrc/torch_memory_saver.cpp'],
        extra_compile_args=['-I/usr/local/cuda/include'],
        extra_link_args=['-lcuda'],
    )],
    python_requires=">=3.9",
    packages=['torch_memory_saver'],
)
