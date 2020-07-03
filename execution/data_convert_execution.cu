#include "data_convert_execution.hpp"

namespace tensorrtInference
{
    __global__ void DataConvertExecutionKernel(const unsigned char* src, float* dst, const int size)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size){
            dst[index] = (float)src[index];
        }
    }

    DataConvertExecution::DataConvertExecution(CUDARuntime *runtime, std::string executionType) : Execution(runtime, executionType)
    {
        setExecutionType("DataConvert");
        setSubExecutionType(executionType);
    }
    
    DataConvertExecution::~DataConvertExecution()
    {
    }

    bool DataConvertExecution::init(std::vector<Buffer*> inputBuffers)
    {
        CHECK_ASSERT(inputBuffers.size() == 1, "input buffer vector size must be 1\n");
        auto shape = inputBuffers[0]->getShape();
        auto runtime = getCudaRuntime();
        auto subType = getSubExecutionType();
        Buffer *outBuffer = nullptr;
        if(subType.compare("ConvertUint8ToFloat32") == 0)
            outBuffer = new Buffer(shape, OnnxDataType::FLOAT);
        else
            LOG("current not support %s\n", subType.c_str());
        CHECK_ASSERT(outBuffer != nullptr, "new Buffer fail\n");
        addOutput(outBuffer);
        addInput(inputBuffers[0]);
        
        if(inputBuffers[0]->device<void>() == nullptr)
        {
            runtime->onAcquireBuffer(inputBuffers[0], StorageType::STATIC);
            runtime->onAcquireBuffer(outBuffer, StorageType::STATIC);
            needMemCpy = true;
        }
        else
            runtime->onAcquireBuffer(outBuffer, StorageType::DYNAMIC);
        
        return true;
    }

    void DataConvertExecution::run(bool sync)
    {
        auto inputBuffers = getInputs();
        auto outputBuffers = getOutputs();
        auto runtime = getCudaRuntime();
        auto stream = runtime->stream();
        int size = inputBuffers[0]->getElementCount();
        auto subType = getSubExecutionType();
        if(needMemCpy)
            runtime->copyToDevice(inputBuffers[0], inputBuffers[0]);

        int blockSize = 256;
        int gridSize = size / blockSize + 1;
        if(subType.compare("ConvertUint8ToFloat32") == 0)
        {
            DataConvertExecutionKernel<<<gridSize, blockSize, 0, stream>>>(inputBuffers[0]->device<unsigned char>(),
                outputBuffers[0]->device<float>(), size);
        }
        else
        {
            CHECK_ASSERT(false, "current not support (%s) \n", getSubExecutionType().c_str());
        }
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "run data convert kernel(%s) fail: %s\n", getSubExecutionType().c_str(),
            cudaGetErrorString(cudastatus));
        if(sync)
            runtime->onWaitFinish();
        return;
    }
}