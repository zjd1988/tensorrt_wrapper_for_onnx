#include "datamovement_execution.hpp"

namespace tensorrtInference
{
    DataMovementExecution::DataMovementExecution(CUDARuntime *runtime, std::string executionType) : Execution(runtime, executionType)
    {
        setExecutionType("DataMovement");
        setSubExecutionType(executionType);
    }
    
    DataMovementExecution::~DataMovementExecution()
    {
    }

    bool DataMovementExecution::init(std::vector<Buffer*> inputBuffers)
    {
        auto type = getSubExecutionType();
        auto runtime = getCudaRuntime();
        for(int i = 0; i < inputBuffers.size(); i++)
        {
            Buffer *outBuffer = nullptr;
            if(type.compare("CopyFromDevice") == 0)
            {
                auto hostPtr = inputBuffers[i]->host<void>();
                auto devicePtr = inputBuffers[i]->device<void>();
                CHECK_ASSERT(hostPtr == nullptr && devicePtr != nullptr, "device ptr should not be null!\n");
                auto shape = inputBuffers[i]->getShape();
                OnnxDataType dataType = inputBuffers[i]->getDataType();
                outBuffer = new Buffer(shape, dataType, true);
                CHECK_ASSERT(outBuffer != nullptr, "new Buffer fail\n");
            }
            else if(type.compare("CopyToDevice") == 0)
            {
                auto hostPtr = inputBuffers[i]->host<void>();
                auto devicePtr = inputBuffers[i]->device<void>();
                CHECK_ASSERT(hostPtr != nullptr && devicePtr == nullptr, "host ptr should not be null!\n");
                auto shape = inputBuffers[i]->getShape();
                OnnxDataType dataType = inputBuffers[i]->getDataType();
                outBuffer = new Buffer(shape, dataType);
                CHECK_ASSERT(outBuffer != nullptr, "new Buffer fail\n");
                runtime->onAcquireBuffer(outBuffer, StorageType::DYNAMIC);
            }
            else
                CHECK_ASSERT(false, "not support %s !\n");
            addOutput(outBuffer);
            addInput(inputBuffers[i]);
        }
        return true;
    }

    void DataMovementExecution::run(bool sync)
    {
        auto inputBuffers = getInputs();
        auto outputBuffers = getOutputs();
        auto runtime = getCudaRuntime();
        auto type = getSubExecutionType();
        for(int i = 0; i < inputBuffers.size(); i++)
        {
            if(type.compare("CopyFromDevice") == 0)
                runtime->copyFromDevice(inputBuffers[i], outputBuffers[i]);
            else if(type.compare("CopyToDevice") == 0)
                runtime->copyToDevice(inputBuffers[i], outputBuffers[i]);
            else
                CHECK_ASSERT(false, "not support %s !\n");
        }
        if(sync)
            runtime->onWaitFinish();
        return;
    }
}