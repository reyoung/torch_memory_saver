#pragma once
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

//#define TMS_DEBUG_LOG

// Cannot use pytorch (libc10.so) since LD_PRELOAD happens earlier than `import torch`
// Thus copy from torch Macros.h
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

#define SIMPLE_CHECK(COND, MSG) \
  do { \
    if (!(COND)) { \
        std::cerr << "[torch_memory_saver.cpp] " << MSG \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)

#define CURESULT_CHECK(EXPR) \
  do { \
    CUresult __result = (EXPR); \
    if (__result != CUDA_SUCCESS) { \
        const char* err_str = nullptr; \
        cuGetErrorString(__result, &err_str); \
        std::cerr << "[torch_memory_saver.cpp] CUresult error: " \
                  << __result << " (" << (err_str ? err_str : "Unknown error") << ") " \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)

#define CUDA_ERROR_CHECK(EXPR) \
  do { \
    cudaError_t __result = (EXPR); \
    if (__result != cudaSuccess) { \
        const char* err_str = cudaGetErrorString(__result); \
        std::cerr << "[torch_memory_saver.cpp] cudaError error: " \
                  << __result << " (" << (err_str ? err_str : "Unknown error") << ") " \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)

namespace CUDAUtils {
    static void cu_mem_create(CUmemGenericAllocationHandle *alloc_handle, size_t size, CUdevice device) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        CURESULT_CHECK(cuMemCreate(alloc_handle, size, &prop, 0));
    }

    static void cu_mem_set_access(void *ptr, size_t size, CUdevice device) {
        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CURESULT_CHECK(cuMemSetAccess((CUdeviceptr) ptr, size, &access_desc, 1));
    }

    static CUdevice cu_ctx_get_device() {
        CUdevice ans;
        CURESULT_CHECK(cuCtxGetDevice(&ans));
        return ans;
    }

    static CUdevice cu_device_get(int device_ordinal) {
        CUdevice ans;
        CURESULT_CHECK(cuDeviceGet(&ans, device_ordinal));
        return ans;
    }
}

bool get_bool_env_var(const char* name) {
    const char* env_cstr = std::getenv(name);
    if (env_cstr == nullptr) {
        return false;
    }

    std::string env_str(env_cstr);
    if (env_str == "1" || env_str == "true" || env_str == "TRUE" || env_str == "yes" || env_str == "YES") {
        return true;
    }
    if (env_str == "0" || env_str == "false" || env_str == "FALSE" || env_str == "no" || env_str == "NO") {
        return false;
    }

    std::cerr << "[torch_memory_saver.cpp] Unsupported environment varialbe value "
              << " name=" << name << " value=" << env_str
              << std::endl;
    exit(1);
}
