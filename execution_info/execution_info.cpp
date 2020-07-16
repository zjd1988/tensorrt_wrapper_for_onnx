#include <iostream>
#include <map>
#include "execution_info.hpp"
#include "utils.hpp"
#include "datatype_convert_execution_info.hpp"
#include "dataformat_convert_execution_info.hpp"
#include "reshape_execution_info.hpp"
#include "transpose_execution_info.hpp"
#include "normalization_execution_info.hpp"
#include "onnx_model_execution_info.hpp"
#include "yolo_nms_execution_info.hpp"
using namespace std;

#define CONSTUCT_EXECUTIONINFO_FUNC_DEF(type)                                                                  \
ExecutionInfo* construct##type##ExecutionInfo(CUDARuntime *runtime,                                            \
        std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root)                        \
{                                                                                                              \
    ExecutionInfo* executionInfo = new type##ExecutionInfo(runtime, tensorsInfo, root);                        \
    return executionInfo;                                                                                      \
}

#define CONSTUCT_EXECUTIONINFO_FUNC(type) construct##type##ExecutionInfo

namespace tensorrtInference {

    ExecutionInfo::ExecutionInfo(CUDARuntime *runtime, std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root)
    {
        inputTensorNames.clear();
        outputTensorNames.clear();
        cudaRuntime = runtime;
        executionInfoType = root["type"].asString();
        // init input tensor names
        {
            int size = root["inputs"].size();
            for(int i = 0; i < size; i++)
            {
                std::string tensorName;
                tensorName = root["inputs"][i].asString();
                inputTensorNames.push_back(tensorName);
            }
        }
        // init output tensor names
        {
            int size = root["outputs"].size();
            for(int i = 0; i < size; i++)
            {
                std::string tensorName;
                tensorName = root["outputs"][i].asString();
                outputTensorNames.push_back(tensorName);
            }
        }
        // init tensor info(malloc buffer)
        initTensorInfo(tensorsInfo, root["tensor_info"]);
        // recycle dynamic buffer
        // recycleBuffers();
    }

    ExecutionInfo::~ExecutionInfo()
    {
        tensors.clear();
        inputTensorNames.clear();
        outputTensorNames.clear();
        cudaRuntime = nullptr;
        executionInfoType = "";
    }

    void ExecutionInfo::initTensorInfo(std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root)
    {
        for (auto elem : root.getMemberNames()) {
            if(tensorsInfo.count(elem) == 0) {
                std::vector<int> shape;
                int size = root[elem]["shape"].size();
                for(int i = 0; i < size; i++) {
                    shape.push_back(root[elem]["shape"][i].asInt());
                }
                auto dataType = (OnnxDataType)(root[elem]["data_type"].asInt());
                auto mallocHost = root[elem]["malloc_host"].asBool();
                auto mallocType = root[elem]["malloc_type"].asString();
                Buffer* buffer;
                if(mallocType.compare("STATIC") == 0)
                    buffer = mallocBuffer(shape, dataType, mallocHost, true, StorageType::STATIC);
                else
                    buffer = mallocBuffer(shape, dataType, mallocHost, true, StorageType::DYNAMIC);
                tensors[elem] = buffer;
                tensorsInfo[elem].reset(buffer);
                if(root[elem]["memcpy_dir"].isString()) {
                    memcpyDir[elem] = root[elem]["memcpy_dir"].asString();
                }
            }
            else
            {
                tensors[elem] = tensorsInfo[elem].get();
            }
        }
    }
    void ExecutionInfo::printExecutionInfo() {
        LOG("################### Execution Info ######################\n");
        LOG("current execution type is %s\n", executionInfoType.c_str());
        LOG("Input tensor is :\n");
        for(int i = 0; i < inputTensorNames.size(); i++)
        {
            LOG("%s \n", inputTensorNames[i].c_str());
        }
        LOG("Output tensor is :\n");
        for(int i = 0; i < outputTensorNames.size(); i++)
        {
            LOG("%s \n", outputTensorNames[i].c_str());
        }        
    }
    void ExecutionInfo::recycleBuffers()
    {
        auto runtime = getCudaRuntime();
        for(int i = 0; i < inputTensorNames.size(); i++)
        {
            auto tensor_name = inputTensorNames[i];
            if(tensors[tensor_name]->getStorageType() == StorageType::DYNAMIC &&
                 tensors[tensor_name]->device<void>() != nullptr)
            {
                runtime->onReleaseBuffer(tensors[tensor_name], StorageType::DYNAMIC);
            }
        }
    }
    Buffer* ExecutionInfo::mallocBuffer(int size, OnnxDataType dataType, bool mallocHost, 
        bool mallocDevice, StorageType type)
    {
        Buffer* buffer = nullptr;
        buffer = new Buffer(size, dataType, mallocHost);
        CHECK_ASSERT(buffer != nullptr, "new Buffer fail\n");
        if(mallocDevice)
        {
            buffer->setStorageType(type);
            cudaRuntime->onAcquireBuffer(buffer, type);
        }
        return buffer;
    }
    Buffer* ExecutionInfo::mallocBuffer(std::vector<int> shape, OnnxDataType dataType, bool mallocHost, 
        bool mallocDevice, StorageType type)
    {
        Buffer* buffer = nullptr;
        buffer = new Buffer(shape, dataType, mallocHost);
        CHECK_ASSERT(buffer != nullptr, "new Buffer fail\n");
        if(mallocDevice)
        {
            buffer->setStorageType(type);
            cudaRuntime->onAcquireBuffer(buffer, type);
        }
        return buffer;
    }

    void ExecutionInfo::beforeRun()
    {
        for(auto item : memcpyDir)
        {
            if(item.second.compare("host_to_device") == 0)
            {
                auto buffer = tensors[item.first];
                cudaRuntime->copyToDevice(buffer, buffer);
            }
        }
    }

    void ExecutionInfo::afterRun()
    {
        for(auto item : memcpyDir)
        {
            if(item.second.compare("device_to_host") == 0)
            {
                auto buffer = tensors[item.first];
                cudaRuntime->copyFromDevice(buffer, buffer);
            }
        }        
    }

    CONSTUCT_EXECUTIONINFO_FUNC_DEF(YoloNMS)
    CONSTUCT_EXECUTIONINFO_FUNC_DEF(Normalization)
    CONSTUCT_EXECUTIONINFO_FUNC_DEF(Transpose)
    CONSTUCT_EXECUTIONINFO_FUNC_DEF(Reshape)
    CONSTUCT_EXECUTIONINFO_FUNC_DEF(DataFormatConvert)
    CONSTUCT_EXECUTIONINFO_FUNC_DEF(DataTypeConvert)
    CONSTUCT_EXECUTIONINFO_FUNC_DEF(OnnxModel)

    constructExecutionInfoFunc ConstructExecutionInfo::getConstructExecutionInfoFunc(std::string executionType)
    {
        if(constructExecutionInfoFuncMap.size() == 0)
            registerConstructExecutionInfoFunc();
        if(constructExecutionInfoFuncMap.count(executionType) != 0)
            return constructExecutionInfoFuncMap[executionType];
        else
            return (constructExecutionInfoFunc)nullptr;
    }

    void ConstructExecutionInfo::registerConstructExecutionInfoFunc()
    {
        constructExecutionInfoFuncMap["YoloNMS"]                    = CONSTUCT_EXECUTIONINFO_FUNC(YoloNMS);
        constructExecutionInfoFuncMap["Normalization"]              = CONSTUCT_EXECUTIONINFO_FUNC(Normalization);
        constructExecutionInfoFuncMap["Transpose"]                  = CONSTUCT_EXECUTIONINFO_FUNC(Transpose);
        constructExecutionInfoFuncMap["Reshape"]                    = CONSTUCT_EXECUTIONINFO_FUNC(Reshape);
        constructExecutionInfoFuncMap["DataFormatConvert"]          = CONSTUCT_EXECUTIONINFO_FUNC(DataFormatConvert);
        constructExecutionInfoFuncMap["DataTypeConvert"]            = CONSTUCT_EXECUTIONINFO_FUNC(DataTypeConvert);
        constructExecutionInfoFuncMap["OnnxModel"]                  = CONSTUCT_EXECUTIONINFO_FUNC(OnnxModel);
    }

    ConstructExecutionInfo* ConstructExecutionInfo::instance = new ConstructExecutionInfo;

    constructExecutionInfoFunc getConstructExecutionInfoFuncMap(std::string executionType)
    {
        auto instance = ConstructExecutionInfo::getInstance();
        if(instance != nullptr)
            return instance->getConstructExecutionInfoFunc(executionType);
        else
            return (constructExecutionInfoFunc)nullptr;
    }
}