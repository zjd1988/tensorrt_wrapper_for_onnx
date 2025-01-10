/********************************************
 * Filename: cuda_runtime.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
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
#include "common/non_copyable.hpp"
#include "common/utils.hpp"
#include "common/buffer_pool.hpp"
#include "common/buffer.hpp"

namespace TENSORRT_WRAPPER
{

    typedef enum CudaRuntimeMemcpyKind_t
    {
        CudaRuntimeMemcpyHostToDevice = 1,
        CudaRuntimeMemcpyDeviceToHost = 2,
        CudaRuntimeMemcpyDeviceToDevice = 3,
    } CudaRuntimeMemcpyKind_t;

    class CUDARuntime : public NonCopyable
    {
    public:
        CUDARuntime(int device_id);
        ~CUDARuntime();

        bool isSupportedFP16();
        bool isSupportedDotInt8();
        bool isSupportedDotAccInt8();
        bool isCreateError();
        int device_id();
        size_t memAlignmentInBytes();
        void activate();

        cudaStream_t stream();
        cublasHandle_t cublasHandle();
        void initializeCusolver();
        cusolverDnHandle_t cusolverHandle();

        int threadsNum() { return mProp.maxThreadsPerBlock; }
        int majorSm() { return mProp.major; }
        int blocksNum(const int total_threads)
        {
            return std::min(((total_threads - 1) / mProp.maxThreadsPerBlock) + 1, mProp.multiProcessorCount);
        }
        bool onAcquireBuffer(Buffer *buffer, StorageType storageType);
        bool onReleaseBuffer(Buffer *buffer, StorageType storageType);
        bool onClearBuffer();
        bool onWaitFinish();
        // void onCopyBuffer(Buffer *srcBuffer, Buffer *dstBuffer);
        void copyFromDevice(Buffer* srcBuffer, Buffer* dstBuffer, bool sync = true);
        void copyToDevice(Buffer* srcBuffer, Buffer* dstBuffer, bool sync = false);
        void copyFromDeviceToDevice(Buffer* srcBuffer, Buffer* dstBuffer, bool sync = false);
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

} // namespace TENSORRT_WRAPPER
