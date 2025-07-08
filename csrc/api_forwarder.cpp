#include "api_forwarder.h"

namespace APIForwarder {
    static void *check_dlsym(void *value) {
        if (nullptr == value) {
            std::cerr << "[torch_memory_saver.cpp] dlsym failed dlerror=" << dlerror() << std::endl;
            exit(1);
        }
        return value;
    }

    using CudaMallocFunc = cudaError_t (*)(void**, size_t);
    using CudaFreeFunc   = cudaError_t (*)(void*);

    static CudaMallocFunc real_cudaMalloc = NULL;
    static CudaFreeFunc real_cudaFree = NULL;

    static cudaError_t call_real_cuda_malloc(void **ptr, size_t size) {
        if (C10_UNLIKELY(nullptr == real_cudaMalloc)) {
            real_cudaMalloc = (CudaMallocFunc) check_dlsym(dlsym(RTLD_NEXT, "cudaMalloc"));
        }

        cudaError_t ret = real_cudaMalloc(ptr, size);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] cudaMalloc [MODE NORMAL]"
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }

    static cudaError_t call_real_cuda_free(void *ptr) {
        if (C10_UNLIKELY(nullptr == real_cudaFree)) {
            real_cudaFree = (CudaFreeFunc) check_dlsym(dlsym(RTLD_NEXT, "cudaFree"));
        }

        cudaError_t ret = real_cudaFree(ptr);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] cudaFree [MODE NORMAL]"
                  << " ptr=" << ptr << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }
}
