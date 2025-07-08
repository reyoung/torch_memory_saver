#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <dlfcn.h>
#include <unordered_map>
#include <mutex>
#include <string>

// #define TMS_DEBUG_LOG

// ----------------------------------------------- copied code --------------------------------------------------

// Cannot use pytorch (libc10.so) since LD_PRELOAD happens earlier than `import torch`
// #include <ATen/cuda/Exceptions.h>

// torch Macros.h
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

// ----------------------------------------------- utils --------------------------------------------------

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

namespace APIForwarder {
    static void *check_dlsym(void *value) {
        if (nullptr == value) {
            std::cerr << "[torch_memory_saver.cpp] dlsym failed dlerror=" << dlerror() << std::endl;
            exit(1);
        }
        return value;
    }

    typedef cudaError_t (*CudaMallocFunc)(void **, size_t);

    typedef cudaError_t (*CudaFreeFunc)(void *);

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

namespace CUDAUtils {
    static void cu_mem_create(CUmemGenericAllocationHandle *allocHandle, size_t size, CUdevice device) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        CURESULT_CHECK(cuMemCreate(allocHandle, size, &prop, 0));
    }

    static void cu_mem_set_access(void *ptr, size_t size, CUdevice device) {
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = device;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CURESULT_CHECK(cuMemSetAccess((CUdeviceptr) ptr, size, &accessDesc, 1));
    }
}

// ----------------------------------------------- primary class --------------------------------------------------

// TODO unify variable cases etc
struct _AllocationMetadata {
    size_t size;
    CUdevice device;
    CUmemGenericAllocationHandle allocHandle;
    std::string tag;
    bool enableCpuBackup;
    void* cpuBackup;
};

enum CopyDirection {
    DEVICE_TO_HOST,
    HOST_TO_DEVICE,
};

class TorchMemorySaver {
public:
    TorchMemorySaver() {}

    cudaError_t malloc(void **ptr, size_t size, const std::string& tag, const bool enable_cpu_backup) {
        CUdevice device;
        CURESULT_CHECK(cuCtxGetDevice(&device));

        CUmemGenericAllocationHandle allocHandle;
        CUDAUtils::cu_mem_create(&allocHandle, size, device);
        CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
        CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
        CUDAUtils::cu_mem_set_access(*ptr, size, device);

        {
            const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
            allocation_metadata_.emplace(*ptr, _AllocationMetadata{size, device, allocHandle, tag, enable_cpu_backup, nullptr});
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
                  << " allocHandle=" << allocHandle << " tag=" << tag
                  << std::endl;
#endif

        return cudaSuccess;
    }

    cudaError_t free(void *ptr) {
        _AllocationMetadata metadata;
        {
            const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
            SIMPLE_CHECK(allocation_metadata_.count(ptr), "Trying to free a pointer not allocated here");
            metadata = allocation_metadata_[ptr];
            allocation_metadata_.erase(ptr);
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
        CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_free "
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
                  << std::endl;
#endif

        return cudaSuccess;
    }

    void pause(const std::string& tag) {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata metadata = it->second;

            if (!tag.empty() && metadata.tag != tag) {
                continue;
            }

            if (metadata.enableCpuBackup) {
                if (nullptr == metadata.cpuBackup) {
                    CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpuBackup, metadata.size));
                }
                // TODO may use cudaMemcpyAsync if needed
                CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpuBackup, ptr, metadata.size, cudaMemcpyDeviceToHost));
            }

            CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
            CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                      << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                      << std::endl;
#endif
        }
    }

    void resume(const std::string& tag) {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata &metadata = it->second;

            if (!tag.empty() && metadata.tag != tag) {
                continue;
            }

            CUmemGenericAllocationHandle newAllocHandle;
            CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);

            CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

            CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

            if (metadata.enableCpuBackup) {
                SIMPLE_CHECK(metadata.cpuBackup != nullptr, "cpuBackup should not be nullptr");
                // TODO may use cudaMemcpyAsync if needed
                CUDA_ERROR_CHECK(cudaMemcpy(ptr, metadata.cpuBackup, metadata.size, cudaMemcpyHostToDevice));
                // maybe we can free host memory if needed (currently keep it there to reduce re-alloc time)
            }

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size << " (old)metadata.allocHandle="
                      << metadata.allocHandle
                      << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                      << std::endl;
#endif

            metadata.allocHandle = newAllocHandle;
        }
    }

    static TorchMemorySaver &instance() {
        static TorchMemorySaver instance;
        return instance;
    }


private:
    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void *, _AllocationMetadata> allocation_metadata_;
};


// ----------------------------------------------- threadlocal configs --------------------------------------------------

struct _ThreadLocalConfig {
    bool is_interesting_region_ = false;
    std::string current_tag_ = "default";
    bool enable_cpu_backup_ = false;
};
static thread_local _ThreadLocalConfig thread_local_config;

// ------------------------------------------------- entrypoints ------------------------------------------------

cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region_) {
        return TorchMemorySaver::instance().malloc(ptr, size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup_);
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

cudaError_t cudaFree(void *ptr) {
    if (thread_local_config.is_interesting_region_) {
        return TorchMemorySaver::instance().free(ptr);
    } else {
        return APIForwarder::call_real_cuda_free(ptr);
    }
}

extern "C" {
void tms_set_interesting_region(bool is_interesting_region) {
    thread_local_config.is_interesting_region_ = is_interesting_region;
}

void tms_set_current_tag(const char* tag) {
    SIMPLE_CHECK(tag != nullptr, "tag should not be null");
    thread_local_config.current_tag_ = tag;
}

void tms_set_enable_cpu_backup(bool enable_cpu_backup) {
    thread_local_config.enable_cpu_backup_ = enable_cpu_backup;
}

void tms_pause(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().pause(tag_str);
}

void tms_resume(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().resume(tag_str);
}
}