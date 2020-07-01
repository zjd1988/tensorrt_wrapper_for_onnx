#include "convert_execution.hpp"

namespace tensorrtInference
{
    __global__ void ConvertExecutionKernel(unsigned char* src, float* dst, const int size)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size){
            dst[index] = (float)src[index];
        }
    }

    ConvertExecution::ConvertExecution(CUDARuntime *runtime, std::string executionType) : Execution(runtime, executionType)
    {
        setExecutionType("Convert");
        setSubExecutionType(executionType);
    }
    
    ConvertExecution::~ConvertExecution()
    {
    }

    bool ConvertExecution::init(std::vector<Buffer*> inputBuffers)
    {
        CHECK_ASSERT(inputBuffers.size() == 1, "input buffer vector size must be 1\n");
        auto shape = inputBuffers[0]->getShape();
        auto outBuffer = new Buffer(shape, OnnxDataType::FLOAT);
        CHECK_ASSERT(outBuffer != nullptr, "new Buffer fail\n");
        addOutput(outBuffer);
        addInput(inputBuffers[0]);
        auto runtime = getCudaRuntime();
        if(inputBuffers[0]->device() == nullptr)
        {
            runtime->onAcquireBuffer(inputBuffers[0], StorageType::STATIC);
            runtime->onAcquireBuffer(outBuffer, StorageType::STATIC);
            needMemCpy = true;
        }
        else
            runtime->onAcquireBuffer(outBuffer, StorageType::DYNAMIC);
        
        return true;
    }

    void ConvertExecution::run(bool sync)
    {
        auto inputBuffers = getInputs();
        auto outputBuffers = getOutputs();
        auto runtime = getCudaRuntime();
        int size = inputBuffers[0]->getSize();
        if(needMemCpy)
            runtime->copyToDevice(inputBuffers[0], inputBuffers[0]);

        int blockSize = 256;
        int gridSize = size / blockSize + 1;
        auto stream = runtime->stream();
        ConvertExecutionKernel<<<gridSize, blockSize, 0, stream>>>((unsigned char *)(inputBuffers[0]->device()),
                 (float*)(outputBuffers[0]->device()), size);
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "run convert kernel(%s) fail: %s\n", getSubExecutionType().c_str(),
            cudaGetErrorString(cudastatus));
        if(sync)
            runtime->onWaitFinish();
        return;
    }
}