#include <iostream>
#include "core.h"
#include "utils.h"

TorchMemorySaver::TorchMemorySaver() {}

static TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup) {
    CUmemGenericAllocationHandle allocHandle;
    CUDAUtils::cu_mem_create(&allocHandle, size, device);
    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(*ptr, AllocationMetadata{size, device, allocHandle, tag, enable_cpu_backup, nullptr});
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
    AllocationMetadata metadata;
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

void TorchMemorySaver::pause(const std::string& tag) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

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

void TorchMemorySaver::resume(const std::string& tag) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

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
