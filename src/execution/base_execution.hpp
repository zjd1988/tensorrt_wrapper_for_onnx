/********************************************
 * Filename: base_execution.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <vector>
#include "cuda_runtime.hpp"
#include "json/json.h"
#include "common/buffer.hpp"
#include "execution/execution_common.hpp"
#define DIVUP(m,n) (((m)+(n)-1) / (n))

namespace TENSORRT_WRAPPER
{

    class BaseExecution
    {
    public:
        BaseExecution(CUDARuntime *runtime, Json::Value& root);
        virtual ~BaseExecution() = default;
        virtual bool init(Json::Value& root) = 0;
        virtual void run() = 0;
        void printExecutionInfo();
        const std::string getExecutionInfoType() { return executionInfoType; }
        const std::vector<std::string>& getInputTensorNames() {return inputTensorNames;}
        const std::vector<std::string>& getOutputTensorNames() {return outputTensorNames;}
        CUDARuntime* getCudaRuntime() { return cudaRuntime; }
        void initTensorInfo(std::map<std::string, std::shared_ptr<Buffer>>& tensorsInfo, Json::Value& root);
        const std::map<std::string, Buffer*>& getTensorsInfo() { return tensors; }
        Buffer* mallocBuffer(int size, OnnxDataType dataType, bool mallocHost, bool mallocDevice, 
            StorageType type = StorageType::DYNAMIC);
        Buffer* mallocBuffer(std::vector<int> shape, OnnxDataType dataType, bool mallocHost, 
            bool mallocDevice, StorageType type = StorageType::DYNAMIC);
        void recycleBuffers();
        void beforeRun();
        void afterRun();
        template<typename T>
        void printBuffer(Buffer* buffer, int start, int end)
        {
            
            CHECK_ASSERT(buffer != nullptr, "buffer must not be none!\n");
            CHECK_ASSERT(start >= 0, "start index must greater than 1!\n");
            auto shape = buffer->getShape();
            auto dataType = buffer->getDataType();
            std::shared_ptr<Buffer> debugBuffer(mallocBuffer(shape, dataType, true, false));
            cudaRuntime->copyFromDevice(buffer, debugBuffer.get());
            cudaError_t cudastatus = cudaGetLastError();
            CHECK_ASSERT(cudastatus == cudaSuccess, "launch memcpy kernel fail: %s\n", cudaGetErrorString(cudastatus));
            
            auto debugData = debugBuffer->host<T>();
            int count = debugBuffer->getElementCount();
            int printStart = (start > count) ? 0 : start;
            int printEnd   = ((end - start) > count) ? (start + count) : end;
            std::cout << "buffer data is :" << std::endl;
            if(onnxDataTypeEleCount[buffer->getDataType()] != 1)
            {
                for(int i = printStart; i < printEnd; i++)
                {
                    std::cout << debugData[i] << " " << std::endl;
                }
            }
            else
            {
                for(int i = printStart; i < printEnd; i++)
                {
                    printf("%d \n", debugData[i]);
                }
            }
        }

    private:
        CUDARuntime* cudaRuntime;
        std::string executionInfoType;
        std::map<std::string, Buffer*> tensors;
        std::vector<std::string> inputTensorNames;
        std::vector<std::string> outputTensorNames;
        std::map<std::string, std::string> memcpyDir;
    };

    /** abstract execution register */
    class ExecutionCreator
    {
    public:
        virtual ~ExecutionCreator() = default;
        virtual BaseExecution* onCreate() const = 0;

    protected:
        ExecutionCreator() = default;
    };

    const ExecutionCreator* getExecutionCreator(const ExecutionType type);
    bool insertExecutionCreator(const ExecutionType type, const ExecutionCreator* creator);
    void logRegisteredExecutionCreator();

} // namespace TENSORRT_WRAPPER