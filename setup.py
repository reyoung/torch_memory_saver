
import logging
import os
import shutil
from pathlib import Path
import setuptools
from setuptools import setup
from setuptools.command.build_ext import build_ext

logger = logging.getLogger(__name__)


# copy & modify from torch/utils/cpp_extension.py
def _find_cuda_home():
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            cuda_home = '/usr/local/cuda'
    return cuda_home


def _find_rocm_home():
    """Find the ROCm/HIP install path."""
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        hipcc_path = shutil.which("hipcc")
        if hipcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(hipcc_path))
        else:
            rocm_home = '/opt/rocm'
    return rocm_home


def _detect_platform():
    """Detect whether to use CUDA or HIP based on available tools."""
    # Check for HIP first (since it might be preferred on AMD systems)
    if shutil.which("hipcc") is not None:
        return "hip"
    elif shutil.which("nvcc") is not None:
        return "cuda"
    else:
        # Default to CUDA if neither is found
        return "cuda"


class HipExtension(setuptools.Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)


class CudaExtension(setuptools.Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)


class build_hip_ext(build_ext):
    def build_extensions(self):
        # Set hipcc as the compiler
        self.compiler.set_executable("compiler_so", "hipcc")
        self.compiler.set_executable("compiler_cxx", "hipcc")
        self.compiler.set_executable("linker_so", "hipcc --shared")
        
        # Add extra compiler and linker flags
        for ext in self.extensions:
            ext.extra_compile_args = ['-fPIC']
            ext.extra_link_args = ['-shared']
        
        build_ext.build_extensions(self)


class build_cuda_ext(build_ext):
    def build_extensions(self):
        # Use default compiler for CUDA
        build_ext.build_extensions(self)


# Detect platform and set up accordingly
platform = _detect_platform()
print(f"Detected platform: {platform}")

if platform == "hip":
    # HIP/ROCm configuration
    rocm_home = Path(_find_rocm_home())
    include_dirs = [
        str(rocm_home.resolve() / 'include'),
    ]
    library_dirs = [
        str(rocm_home.resolve() / 'lib'),
    ]

    ext_modules=[
        setuptools.Extension(
            name,
            [
                'csrc/api_forwarder.cpp',
                'csrc/core.cpp',
                'csrc/entrypoint.cpp',
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=['amdhip64', 'dl'],
            define_macros=[
                ('Py_LIMITED_API', '0x03090000'),('USE_HIP', '1'),
                *extra_macros,
            ],
            py_limited_api=True,
        )
        for name, extra_macros in [
            ('torch_memory_saver_hook_mode_preload', [('TMS_HOOK_MODE_PRELOAD', '1')]),
            ('torch_memory_saver_hook_mode_torch', [('TMS_HOOK_MODE_TORCH', '1')]),
        ]
    ]
    cmdclass = {'build_ext': build_hip_ext}
    
else:
    # CUDA configuration
    cuda_home = Path(_find_cuda_home())
    include_dirs = [
        str((cuda_home / 'include').resolve()),
    ]

    library_dirs = [
        str((cuda_home / 'lib64').resolve()),
        str((cuda_home / 'lib64/stubs').resolve()),
    ]

    ext_modules=[
        setuptools.Extension(
            name,
            [
                'csrc/api_forwarder.cpp',
                'csrc/core.cpp',
                'csrc/entrypoint.cpp',
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=['cuda'],
            define_macros=[
                ('Py_LIMITED_API', '0x03090000'),('USE_CUDA', '1'),
                *extra_macros,
            ],
            py_limited_api=True,
        )
        for name, extra_macros in [
            ('torch_memory_saver_hook_mode_preload', [('TMS_HOOK_MODE_PRELOAD', '1')]),
            ('torch_memory_saver_hook_mode_torch', [('TMS_HOOK_MODE_TORCH', '1')]),
        ]
    ]
    cmdclass = {'build_ext': build_cuda_ext}


setup(
    name='torch_memory_saver',
    version='0.0.8',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    packages=setuptools.find_packages(include=["torch_memory_saver", "torch_memory_saver.*"]),
)
