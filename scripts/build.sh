#!/bin/bash
set -euxo pipefail

# NOTE MODIFIED FROM https://github.com/sgl-project/sglang/blob/main/sgl-kernel/build.sh

echo "python=${PYTHON_VERSION} cuda=${CUDA_VERSION}" # args

PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}
ARCH=$(uname -m)
TEGRA="tegra"
if [ -z "${SYSTEM_ARCH}" ]; then
  echo "ARCH:  $ARCH"
  if [ "$ARCH" = "aarch64" ]; then
    echo "### ARM64 architecture detected"
    if uname -a | grep -qi "$TEGRA"; then
      echo "### Jetson Detected"
      LIBCUDA_ARCH="aarch64"
      BUILDER_NAME="pytorch/manylinuxaarch64-builder"
    else
      echo "### SBSA Detected"
      LIBCUDA_ARCH="sbsa"
      BUILDER_NAME="pytorch/manylinuxaarch64-builder"
    fi
  else
    echo "### x86 Detected"
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
   bash /app/scripts/build_in_docker.sh
   "
