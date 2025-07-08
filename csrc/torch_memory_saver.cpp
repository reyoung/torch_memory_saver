#include <sys/types.h>
#include <iostream>
#include <stdio.h>
#include <unordered_map>
#include <mutex>
#include <string>
#include "utils.h"

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

    cudaError_t malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup) {
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
            _AllocationMetadata& metadata = it->second;

            if (!tag.empty() && metadata.tag != tag) {
                continue;
            }

            if (metadata.enableCpuBackup) {
                if (nullptr == metadata.cpuBackup) {
                    CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpuBackup, metadata.size));
                }
                SIMPLE_CHECK(metadata.cpuBackup != nullptr, "cpuBackup should not be nullptr");
                // TODO may use cudaMemcpyAsync if needed
                CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpuBackup, ptr, metadata.size, cudaMemcpyDeviceToHost));
            }

            CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
            CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                      << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                      << " metadata.enableCpuBackup=" << metadata.enableCpuBackup
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
                      << " metadata.enableCpuBackup=" << metadata.enableCpuBackup
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

// ------------------------------------------------- entrypoints :: hook ------------------------------------------------

#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region_) {
        return TorchMemorySaver::instance().malloc(
            ptr, CUDAUtils::cu_ctx_get_device(), size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup_);
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
#endif

#ifdef TMS_HOOK_MODE_TORCH
extern "C" {
void *tms_torch_malloc(ssize_t size, int device, cudaStream_t stream) {
    SIMPLE_CHECK(thread_local_config.is_interesting_region_, "only support interesting region");
    void *ptr;
    TorchMemorySaver::instance().malloc(
        &ptr, CUDAUtils::cu_device_get(), size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup_);
    return ptr;
}

void tms_torch_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
    SIMPLE_CHECK(thread_local_config.is_interesting_region_, "only support interesting region");
    TorchMemorySaver::instance().free(ptr);
}
}
#endif

// ------------------------------------------------- entrypoints :: others ------------------------------------------------

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