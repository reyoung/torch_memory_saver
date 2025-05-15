#!/bin/bash
set -euxo pipefail

# NOTE MODIFIED FROM https://github.com/sgl-project/sglang/blob/main/sgl-kernel/build.sh

PYTHON_VERSION=$1
CUDA_VERSION=$2
PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}

if [ -z "$3" ]; then
   ARCH=$(uname -i)
else
   ARCH=$3
fi

echo "ARCH:  $ARCH"
if [ ${ARCH} = "aarch64" ]; then
   LIBCUDA_ARCH="sbsa"
   BUILDER_NAME="pytorch/manylinuxaarch64-builder"
else
   LIBCUDA_ARCH=${ARCH}
   if [ ${CUDA_VERSION} = "12.8" ]; then
      BUILDER_NAME="pytorch/manylinux2_28-builder"
   else
      BUILDER_NAME="pytorch/manylinux-builder"
   fi
fi

if [ ${CUDA_VERSION} = "12.8" ]; then
   DOCKER_IMAGE="${BUILDER_NAME}:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION//.}"
else
   DOCKER_IMAGE="${BUILDER_NAME}:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.}"
fi

docker run --rm \
   -v $(pwd):/sgl-kernel \
   ${DOCKER_IMAGE} \
   bash -c "
   ${PYTHON_ROOT_PATH}/bin/${TORCH_INSTALL} && \
   ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core && \
   export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0+PTX' && \
   export CUDA_VERSION=${CUDA_VERSION} && \
   mkdir -p /usr/lib/${ARCH}-linux-gnu/ && \
   ln -s /usr/local/cuda-${CUDA_VERSION}/targets/${LIBCUDA_ARCH}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so && \

   cd /sgl-kernel && \
   ls -la ${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages/wheel/ && \
   PYTHONPATH=${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages ${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation && \
   ./rename_wheels.sh
   "
