/********************************************
 * Filename: base_execution.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <iostream>
#include <map>
#include "common/logger.hpp"
#include "execution/base_execution.hpp"
using namespace std;

namespace TENSORRT_WRAPPER
{

    BaseExecution::BaseExecution(CUDARuntime *runtime, Json::Value& root)
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

    void BaseExecution::initTensorInfo(std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root)
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
    void BaseExecution::printExecutionInfo()
    {
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

    void BaseExecution::recycleBuffers()
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
        return;
    }

    Buffer* BaseExecution::mallocBuffer(int size, OnnxDataType dataType, bool mallocHost, 
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

    Buffer* BaseExecution::mallocBuffer(std::vector<int> shape, OnnxDataType dataType, bool mallocHost, 
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

    void BaseExecution::beforeRun()
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

    void BaseExecution::afterRun()
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

    static std::map<ExecutionType, const ExecutionCreator*>& getExecutionCreatorMap()
    {
        static std::once_flag gInitFlag;
        static std::map<ExecutionType, const ExecutionCreator*>* gExecutionCreatorMap;
        std::call_once(gInitFlag, [&]() {
            gExecutionCreatorMap = new std::map<ExecutionType, const ExecutionCreator*>;
        });
        return *gExecutionCreatorMap;
    }

    extern void registerExecutionCreator();
    const ExecutionCreator* getExecutionCreator(const ExecutionType type)
    {
        registerExecutionCreator();
        auto& creator_map = getExecutionCreatorMap();
        auto iter = creator_map.find(type);
        if (iter == creator_map.end())
        {
            return nullptr;
        }
        if (iter->second)
        {
            return iter->second;
        }
        return nullptr;
    }

    bool insertExecutionCreator(const ExecutionType type, const ExecutionCreator* creator)
    {
        auto& creator_map = getExecutionCreatorMap();
        if (gExecutionTypeToStr.end() == gExecutionTypeToStr.find(type))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "invalid execution creator:{}", int(type));
            return false;
        }
        std::string type_str = gExecutionTypeToString[type];
        if (creator_map.find(type) != creator_map.end())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "insert duplicate {} execution creator", type_str);
            return false;
        }
        creator_map.insert(std::make_pair(type, creator));
        return true;
    }

    void logRegisteredExecutionCreator()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "registered execution creator as follows:");
        auto& creator_map = getExecutionCreatorMap();
        for (const auto& it : creator_map)
        {
            std::string execution_type = gExecutionTypeToString[it.first];
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "{}:{} execution creator", int(it.first), execution_type);
        }
        return;
    }

} // namespace TENSORRT_WRAPPER