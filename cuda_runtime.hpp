
#ifndef __CUDA_RUNTIME_HPP__
#define __CUDA_RUNTIME_HPP__

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda.h>
#include "utils.hpp"
#include "buffer_pool.hpp"
#include "buffer.hpp"


namespace tensorrtInference {

typedef enum {
    CudaRuntimeMemcpyHostToDevice = 1,
    CudaRuntimeMemcpyDeviceToHost = 2,
    CudaRuntimeMemcpyDeviceToDevice = 3,
} CudaRuntimeMemcpyKind_t;

class CUDARuntime {
public:
    CUDARuntime(int device_id);
    ~CUDARuntime();
    CUDARuntime(const CUDARuntime &) = delete;
    CUDARuntime &operator=(const CUDARuntime &) = delete;

    bool isSupportedFP16();
    bool isSupportedDotInt8();
    bool isSupportedDotAccInt8();
    bool isCreateError();
    int device_id();
    size_t mem_alignment_in_bytes();
    void activate();

    cudaStream_t stream();
    cublasHandle_t cublas_handle();
    void initialize_cusolver();
    cusolverDnHandle_t cusolver_handle();

    int threads_num() { return mProp.maxThreadsPerBlock; }
    int major_sm() { return mProp.major; }
    int blocks_num(const int total_threads) {
        return std::min(((total_threads - 1) / mProp.maxThreadsPerBlock) + 1, mProp.multiProcessorCount);
    }
    bool onAcquireBuffer(Buffer *buffer, StorageType storageType);
    bool onReleaseBuffer(Buffer *buffer, StorageType storageType);
    bool onClearBuffer();
    bool onWaitFinish();
    void onCopyBuffer(Buffer *srcBuffer, Buffer *dstBuffer);
    void copyFromDevice(Buffer* srcBuffer, Buffer* dstBuffer);
    void copyToDevice(Buffer* srcBuffer, Buffer* dstBuffer);
private:
    void memcpy(void *dst, const void *src, size_t size_in_bytes, CudaRuntimeMemcpyKind_t kind, bool sync = false);
    void memset(void *dst, int value, size_t size_in_bytes);
    void synchronize();

    std::shared_ptr<BufferPool> mBufferPool;
    std::shared_ptr<BufferPool> mStaticBufferPool;
    cudaDeviceProp mProp;
    int mDeviceId;
    cudaStream_t mStream = nullptr;
    cublasHandle_t mCublasHandle = nullptr;
    cusolverDnHandle_t mCusolverHandle = nullptr;
    std::once_flag mCusolverInitialized;
    bool mIsSupportedFP16 = false;
    bool mSupportDotInt8 = false;
    bool mSupportDotAccInt8 = false;
    bool mIsCreateError{false};
};

} // namespace tensorrtInference
#endif  /* __CUDA_RUNTIME_HPP__ */
