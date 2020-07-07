#include "format_convert_execution.hpp"

namespace tensorrtInference
{
    template <typename T>
    __global__ void FormatConvertExecutionKernel(const T* src, T* dst, const int size, const int channel)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size){
            int mod = channel - 1;
            int new_pos = index / channel * channel + mod - index % channel;
            dst[new_pos] = src[index];
        }
    }

    void FormatConvertExecution::callFormatConvertExecutionKernel(Buffer* src, Buffer* dst, std::string &convertType, CUDARuntime *runtime)
    {
        void* srcPtr = src->device<void>();
        void* dstPtr = dst->device<void>();
        int size = src->getElementCount();
        int blockSize = 256;
        int gridSize = size / blockSize + 1;
        auto stream = runtime->stream();
        if(convertType.compare("RGB2BGR") == 0 || convertType.compare("BGR2RGB") == 0)
        {
            FormatConvertExecutionKernel<unsigned char><<<gridSize, blockSize, 0, stream>>>((unsigned char *)srcPtr,
                (unsigned char *)dstPtr, size, 3);
        }
        else
        {
            CHECK_ASSERT(false, "current not support (%s) \n", convertType.c_str());
        }
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "run format convert kernel(%s) fail: %s\n", convertType.c_str(),
            cudaGetErrorString(cudastatus));
    }

    FormatConvertExecution::FormatConvertExecution(CUDARuntime *runtime, std::string executionType) : Execution(runtime, executionType)
    {
        setExecutionType("FormatConvert");
        setSubExecutionType(executionType);
    }
    
    FormatConvertExecution::~FormatConvertExecution()
    {
    }

    bool FormatConvertExecution::init(std::vector<Buffer*> inputBuffers)
    {
        CHECK_ASSERT(inputBuffers.size() == 1, "input buffer size must be 1\n");
        auto shape = inputBuffers[0]->getShape();
        auto dataType = inputBuffers[0]->getDataType();
        auto runtime = getCudaRuntime();
        auto subType = getSubExecutionType();
        Buffer *outBuffer = nullptr;
        outBuffer = new Buffer(shape, dataType);
        CHECK_ASSERT(outBuffer != nullptr, "new Buffer fail\n");
        addOutput(outBuffer);
        addInput(inputBuffers[0]);
        if(inputBuffers[0]->device<void>() == nullptr)
        {
            runtime->onAcquireBuffer(inputBuffers[0], StorageType::STATIC);
            needMemCpy = true;
        }
        runtime->onAcquireBuffer(outBuffer, StorageType::DYNAMIC);
        recycleBuffers();
        return true;
    }

    void FormatConvertExecution::run(bool sync)
    {
        auto inputBuffers = getInputs();
        auto outputBuffers = getOutputs();
        auto runtime = getCudaRuntime();
        auto subType = getSubExecutionType();
        if(needMemCpy)
            runtime->copyToDevice(inputBuffers[0], inputBuffers[0]);

        callFormatConvertExecutionKernel(inputBuffers[0], outputBuffers[0], subType, runtime);
        if(sync)
            runtime->onWaitFinish();
        return;
    }
}