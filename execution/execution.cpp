#include <iostream>
#include <map>
#include "execution.hpp"
#include "utils.hpp"
#include "data_convert_execution.hpp"
#include "format_convert_execution.hpp"
#include "datamovement_execution.hpp"
#include "normalization_execution.hpp"
#include "transpose_execution.hpp"
#include "yolo_nms_execution.hpp"
using namespace std;
#define DEBUG_BUFFER_SIZE 4096000

#define CONSTUCT_EXECUTION_FUNC_DEF(type)                                          \
Execution* construct##type##Execution(CUDARuntime *runtime, std::string subType)   \
{                                                                                  \
    Execution* execution = new type##Execution(runtime, subType);                  \
    return execution;                                                              \
}

#define CONSTUCT_EXECUTION_FUNC(type) construct##type##Execution

namespace tensorrtInference {

    Execution::Execution(CUDARuntime *runtime, std::string executionType)
    {
        inputs.clear();
        outputs.clear();
        cudaRuntime = runtime;
        debugBuffer = new Buffer(DEBUG_BUFFER_SIZE, OnnxDataType::UINT8, true);
    }
    Execution::~Execution()
    {
        for(int i = 0; i < outputs.size(); i++){
            delete outputs[i];
            outputs[i] = nullptr;
        }
        delete debugBuffer;
        debugBuffer = nullptr;
        inputs.clear();
        outputs.clear();
        cudaRuntime = nullptr;
        executionType = "";
    }
    std::vector<Buffer*> Execution::getInputs() {return inputs;}
    std::vector<Buffer*> Execution::getOutputs() {return outputs;}
    std::string Execution::getExecutionType() {return executionType;}
    std::string Execution::getSubExecutionType() {return subExecutionType;}
    CUDARuntime* Execution::getCudaRuntime() {return cudaRuntime;}
    void Execution::setExecutionType(std::string type) {executionType = type;}
    void Execution::setSubExecutionType(std::string subType) {subExecutionType = subType;}
    void Execution::addInput(Buffer* buffer) {inputs.push_back(buffer);}
    void Execution::addOutput(Buffer* buffer) {outputs.push_back(buffer);}
    void Execution::printExecutionInfo() {
        LOG("################### Execution INFO ######################\n");
        LOG("currend execution type is %s\n", executionType.c_str());
        auto input = getInputs();
        LOG("Input tensor size is %d\n", input.size());
        auto output = getOutputs();
        LOG("Output tensor size is %d\n", output.size());
    }
    void Execution::recycleBuffers()
    {
        auto runtime = getCudaRuntime();
        for(int i = 0; i < inputs.size(); i++)
        {
            if(inputs[i]->getStorageType() == StorageType::DYNAMIC 
                && inputs[i]->device<void>() != nullptr)
            {
                runtime->onReleaseBuffer(inputs[i], StorageType::DYNAMIC);
            }
        }
    }
    Buffer* Execution::mallocBuffer(int size, OnnxDataType dataType, bool mallocHost, 
        bool mallocDevice, StorageType type)
    {
        auto runtime = getCudaRuntime();
        Buffer* buffer = nullptr;
        buffer = new Buffer(size, dataType, mallocHost);
        CHECK_ASSERT(buffer != nullptr, "new Buffer fail\n");
        if(mallocDevice)
            runtime->onAcquireBuffer(buffer, type);
        return buffer;
    }
    Buffer* Execution::mallocBuffer(std::vector<int> shape, OnnxDataType dataType, bool mallocHost, 
        bool mallocDevice, StorageType type)
    {
        auto runtime = getCudaRuntime();
        Buffer* buffer = nullptr;
        buffer = new Buffer(shape, dataType, mallocHost);
        CHECK_ASSERT(buffer != nullptr, "new Buffer fail\n");
        if(mallocDevice)
            runtime->onAcquireBuffer(buffer, type);
        return buffer;
    }        

    CONSTUCT_EXECUTION_FUNC_DEF(DataConvert)
    CONSTUCT_EXECUTION_FUNC_DEF(FormatConvert)
    CONSTUCT_EXECUTION_FUNC_DEF(DataMovement)
    CONSTUCT_EXECUTION_FUNC_DEF(Normalization)
    CONSTUCT_EXECUTION_FUNC_DEF(Transpose)
    CONSTUCT_EXECUTION_FUNC_DEF(YoloNMS)

    constructExecutionFunc ConstructExecution::getConstructExecutionFunc(std::string executionType)
    {
        if(constructExecutionFuncMap.size() == 0)
            registerConstructExecutionFunc();
        if(constructExecutionFuncMap.count(executionType) != 0)
            return constructExecutionFuncMap[executionType];
        else
            return (constructExecutionFunc)nullptr;
    }

    void ConstructExecution::registerConstructExecutionFunc()
    {
        constructExecutionFuncMap["ConvertUint8ToFloat32"]            = CONSTUCT_EXECUTION_FUNC(DataConvert);
        constructExecutionFuncMap["Scale_0_1"]                        = CONSTUCT_EXECUTION_FUNC(Normalization);
        constructExecutionFuncMap["CopyFromDevice"]                   = CONSTUCT_EXECUTION_FUNC(DataMovement);
        constructExecutionFuncMap["CopyToDevice"]                     = CONSTUCT_EXECUTION_FUNC(DataMovement);
        constructExecutionFuncMap["NHWC2NCHW"]                        = CONSTUCT_EXECUTION_FUNC(Transpose);
        constructExecutionFuncMap["BGR2RGB"]                          = CONSTUCT_EXECUTION_FUNC(FormatConvert);
        constructExecutionFuncMap["RGB2BGR"]                          = CONSTUCT_EXECUTION_FUNC(FormatConvert);
        constructExecutionFuncMap["YOLO_NMS"]                         = CONSTUCT_EXECUTION_FUNC(YoloNMS);
    }

    ConstructExecution* ConstructExecution::instance = new ConstructExecution;

    constructExecutionFunc getConstructExecutionFuncMap(std::string executionType)
    {
        auto instance = ConstructExecution::getInstance();
        if(instance != nullptr)
            return instance->getConstructExecutionFunc(executionType);
        else
            return (constructExecutionFunc)nullptr;
    }
}