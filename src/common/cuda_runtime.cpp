/********************************************
 * Filename: cuda_runtime.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "common/cuda_runtime.hpp"
#include "common/buffer.hpp"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define CUDNN_VERSION_STR STR(CUDNN_MAJOR) "." STR(CUDNN_MINOR) "." STR(CUDNN_PATCHLEVEL)

#pragma message "compile with cuda " STR(CUDART_VERSION) " "
#pragma message "compile with cuDNN " CUDNN_VERSION_STR " "

#undef STR
#undef STR_HELPER

namespace TENSORRT_WRAPPER
{

    bool CUDARuntime::isCreateError() {
        return mIsCreateError;
    }

    CUDARuntime::CUDARuntime(int device_id) {
        int version;
        CUDA_CHECK(cudaRuntimeGetVersion(&version));
        if(version != CUDART_VERSION)
        {
            CHECK_ASSERT(version == CUDART_VERSION, "compiled with cuda %d, get %d at runtime\n", CUDART_VERSION, version);
        }
        int id = device_id;
        if (id < 0) {
            CUDA_CHECK(cudaGetDevice(&id));
        }
        mDeviceId = id;
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaGetDeviceProperties(&mProp, id));
        CUDA_CHECK(cudaStreamCreate(&mStream));

        CUBLAS_CHECK(cublasCreate(&mCublasHandle));
        // Set stream for cuDNN and cublas handles.
        CUBLAS_CHECK(cublasSetStream(mCublasHandle, mStream));

        // Note that all cublas scalars (alpha, beta) and scalar results such as dot
        // output resides at device side.
        CUBLAS_CHECK(cublasSetPointerMode(mCublasHandle, CUBLAS_POINTER_MODE_DEVICE));

        mBufferPool.reset(new BufferPool());
        mStaticBufferPool.reset(new BufferPool());
    }

    CUDARuntime::~CUDARuntime() {
        if (mStream) {
            CUDA_CHECK(cudaStreamDestroy(mStream));
        }
        CUBLAS_CHECK(cublasDestroy(mCublasHandle));
        if (mCusolverHandle) {
            CUSOLVER_CHECK(cusolverDnDestroy(mCusolverHandle));
        }
    }

    bool CUDARuntime::isSupportedFP16() {
        return mIsSupportedFP16;
    }

    bool CUDARuntime::isSupportedDotInt8() {
        return mSupportDotInt8;
    }

    bool CUDARuntime::isSupportedDotAccInt8() {
        return mSupportDotAccInt8;
    }

    size_t CUDARuntime::mem_alignment_in_bytes() {
        return std::max(mProp.textureAlignment, mProp.texturePitchAlignment);
    }

    int CUDARuntime::device_id() {
        return mDeviceId;
    }

    void CUDARuntime::activate()
    {
        int id = device_id();
        if (id >= 0) {
            CUDA_CHECK(cudaSetDevice(id));
        }
    }

    cudaStream_t CUDARuntime::stream() {
        return mStream;
    }

    void CUDARuntime::memcpy(void *dst, const void *src, size_t size_in_bytes, CudaRuntimeMemcpyKind_t kind, bool sync)
    {
        cudaMemcpyKind cuda_kind;
        switch (kind) {
            case CudaRuntimeMemcpyDeviceToHost:
                cuda_kind = cudaMemcpyDeviceToHost;
                break;
            case CudaRuntimeMemcpyHostToDevice:
                cuda_kind = cudaMemcpyHostToDevice;
                break;
            case CudaRuntimeMemcpyDeviceToDevice:
                cuda_kind = cudaMemcpyDeviceToDevice;
                break;
            default:
                CHECK_ASSERT(false, "bad cuda memcpy kind\n");
        }
        if(sync == false)
            CUDA_CHECK(cudaMemcpyAsync(dst, src, size_in_bytes, cuda_kind, mStream));
        else
            CUDA_CHECK(cudaMemcpy(dst, src, size_in_bytes, cuda_kind));
    }

    void CUDARuntime::memset(void *dst, int value, size_t size_in_bytes)
    {
        CUDA_CHECK(cudaMemsetAsync(dst, value, size_in_bytes, mStream));
    }

    void CUDARuntime::synchronize()
    {
        CUDA_CHECK(cudaStreamSynchronize(mStream));
    }

    cublasHandle_t CUDARuntime::cublas_handle() {
        return mCublasHandle;
    }

    void CUDARuntime::initialize_cusolver() {
        CUSOLVER_CHECK(cusolverDnCreate(&mCusolverHandle));
        CUSOLVER_CHECK(cusolverDnSetStream(mCusolverHandle, mStream));
    }
    cusolverDnHandle_t CUDARuntime::cusolver_handle() {
        std::call_once(mCusolverInitialized,
                        [this] { initialize_cusolver(); });
        return mCusolverHandle;
    }

    bool CUDARuntime::onAcquireBuffer(Buffer* buffer, StorageType storageType) {
        int mallocSize = buffer->getSize();
        if (storageType == DYNAMIC) {
            auto bufferPtr = mBufferPool->alloc(mallocSize, false);
            buffer->setDevice(bufferPtr);
            return true;
        }
        CHECK_ASSERT(storageType == STATIC, "storageType must be one of (DYNAMIC/STATIC)\n");
        auto bufferPtr = mStaticBufferPool->alloc(mallocSize, false);
        buffer->setDevice(bufferPtr);
        return true;
    }

    bool CUDARuntime::onReleaseBuffer(Buffer* buffer, StorageType storageType) {
        auto bufferPtr = buffer->device<void>();
        if (storageType == DYNAMIC) {
            mBufferPool->recycle((void*)bufferPtr);
        }
        else if (storageType == STATIC) {
            mStaticBufferPool->recycle((void*)bufferPtr, true);
        }
        else{
            LOG("unknown storage type!\n");
            return false;
        }
        return true;
    }

    bool CUDARuntime::onClearBuffer() {
        mBufferPool->clear();
        mStaticBufferPool->clear();
        return true;
    }

    bool CUDARuntime::onWaitFinish() {
        synchronize();
        return true;
    }

    void CUDARuntime::copyFromDevice(Buffer* srcBuffer, Buffer* dstBuffer, bool sync) {
        auto dstSize = dstBuffer->getSize();
        auto srcSize = srcBuffer->getSize();
        CHECK_ASSERT(srcSize <= dstSize, "src buffer size must less than dst buffer size!\n");
        auto hostPtr = dstBuffer->host<void>();
        auto devicePtr = srcBuffer->device<void>();
        memcpy(hostPtr, devicePtr, srcSize, CudaRuntimeMemcpyDeviceToHost, sync);
    }

    void CUDARuntime::copyToDevice(Buffer* srcBuffer, Buffer* dstBuffer, bool sync) {
        auto dstSize = dstBuffer->getSize();
        auto srcSize = srcBuffer->getSize();
        CHECK_ASSERT(srcSize == dstSize, "src buffer size must be equal to dst buffer size!\n");
        auto hostPtr = srcBuffer->host<void>();
        auto devicePtr = dstBuffer->device<void>();
        memcpy(devicePtr, hostPtr, dstSize, CudaRuntimeMemcpyHostToDevice, sync);
    }

    void CUDARuntime::copyFromDeviceToDevice(Buffer* srcBuffer, Buffer* dstBuffer, bool sync)
    {
        auto dstSize = dstBuffer->getSize();
        auto srcSize = srcBuffer->getSize();
        CHECK_ASSERT(srcSize <= dstSize, "src buffer size must less than dst buffer size!\n");
        auto srcPtr = srcBuffer->device<void>();
        auto dstPtr = dstBuffer->device<void>();
        memcpy(dstPtr, srcPtr, srcSize, CudaRuntimeMemcpyDeviceToDevice, sync);
    }

    // void CUDARuntime::onCopyBuffer(Buffer* srcBuffer, Buffer* dstBuffer) {
    //     if (srcBuffer->device<void>() == nullptr && dstBuffer->device<void>() != nullptr) {
    //         copyToDevice(srcBuffer, dstBuffer);
    //     }else if(srcBuffer->device<void>() != nullptr && dstBuffer->device<void>() == nullptr){
    //         copyFromDevice(srcBuffer, dstBuffer);
    //     }else{
    //         LOG("onCopyBuffer float error !!! \n");
    //     }
    // }

} // namespace TENSORRT_WRAPPER
