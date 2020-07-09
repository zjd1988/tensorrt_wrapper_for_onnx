#include "data_convert_execution.hpp"

namespace tensorrtInference
{
    template <typename SRC_T, typename DST_T>
    __global__ void DataConvertExecutionKernel(const SRC_T* src, DST_T* dst, const int size)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size){
            dst[index] = (DST_T)src[index];
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
        else if(subType.compare("ConvertUint8ToFloat16") == 0)
            outBuffer = new Buffer(shape, OnnxDataType::FLOAT16);
        else
            LOG("current not support %s\n", subType.c_str());
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

    void DataConvertExecution::run(bool sync)
    {
        auto inputBuffers = getInputs();
        auto outputBuffers = getOutputs();
        auto runtime = getCudaRuntime();
        auto stream = runtime->stream();
        const int size = inputBuffers[0]->getElementCount();
        auto subType = getSubExecutionType();
        if(needMemCpy)
            runtime->copyToDevice(inputBuffers[0], inputBuffers[0]);

        int blockSize = runtime->threads_num();
        int gridSize = size / blockSize + 1;
        if(subType.compare("ConvertUint8ToFloat32") == 0)
        {
            DataConvertExecutionKernel<<<gridSize, blockSize, 0, stream>>>(inputBuffers[0]->device<const unsigned char>(),
                outputBuffers[0]->device<float>(), size);
        }
        else if(subType.compare("ConvertUint8ToFloat16") == 0)
        {
            DataConvertExecutionKernel<<<gridSize, blockSize, 0, stream>>>(inputBuffers[0]->device<const unsigned char>(),
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