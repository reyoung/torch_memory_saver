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

DOCKER_IMAGE="${BUILDER_NAME}:cuda${CUDA_VERSION}"

docker run --rm \
   -v $(pwd):/app \
   ${DOCKER_IMAGE} \
   bash -c "
   PYTHON_ROOT_PATH=${PYTHON_ROOT_PATH} \
   PYTHON_VERSION=${PYTHON_VERSION} \
   CUDA_VERSION=${CUDA_VERSION} \
   ARCH=${ARCH} \
   LIBCUDA_ARCH=${LIBCUDA_ARCH} \
   ./build_in_docker.sh
   "
