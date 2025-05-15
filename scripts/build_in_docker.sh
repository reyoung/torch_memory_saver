#!/usr/bin/env bash
set -euxo pipefail

# NOTE MODIFIED FROM https://github.com/sgl-project/sglang/blob/main/sgl-kernel/build.sh

${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.}
${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core
export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0+PTX'
export CUDA_VERSION=${CUDA_VERSION}
mkdir -p /usr/lib/${ARCH}-linux-gnu/
ln -s /usr/local/cuda-${CUDA_VERSION}/targets/${LIBCUDA_ARCH}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so

cd /app
ls -la ${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages/wheel/
PYTHONPATH=${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages ${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation
/app/scripts/rename_wheels.sh
