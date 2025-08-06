#pragma once

// Define platform macros and include appropriate headers
#if defined(USE_ROCM)
    // Lookup the table to define the macros: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Driver_API_functions_supported_by_HIP.html
    // Lookup the table to define the macros: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html?utm_source=chatgpt.com
    #include <hip/hip_runtime_api.h>
    #include <hip/hip_runtime.h>

    #include <sstream>
    #include <cstdlib>
    // Define a general alias
    #define CUresult hipError_t
    #define cudaError_t hipError_t
    #define CUDA_SUCCESS hipSuccess
    #define cudaSuccess hipSuccess
    #define cuGetErrorString hipDrvGetErrorString
    #define cuMemGetAllocationGranularity hipMemGetAllocationGranularity
    #define CUdevice hipDevice_t

    #define cudaStream_t hipStream_t

    #define cudaMallocHost hipHostMalloc
    #define cudaMemcpy hipMemcpy
    #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

    #define cuGetErrorString hipDrvGetErrorString
    #define cudaGetErrorString hipGetErrorString   

    #define cuMemUnmap hipMemUnmap
    #define cuMemRelease hipMemRelease

    // #define cudaMalloc hipMalloc
    // #define cudaFree hipFree
    // #define CUdevice hipDevice_t
    // #define CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t

    #define MEMCREATE_CHUNK_SIZE (2 * 1024 * 1024)
    #define MIN(a, b) (a < b ? a : b)
#elif defined(USE_CUDA)
    #include <cuda_runtime_api.h>
    #include <cuda.h>
    // // Define a general alias
    // #define CUresult CUresult
    // #define cudaError_t cudaError_t
    // #define CUDA_SUCCESS CUDA_SUCCESS

    // #define cudaMalloc cudaMalloc
    // #define cudaFree cudaFree
    // #define CUdevice CUdevice
    // #define CUmemGenericAllocationHandle CUmemGenericAllocationHandle
#else
    #error "USE_PLATFORM is not set"
#endif
