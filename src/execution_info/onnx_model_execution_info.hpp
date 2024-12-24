#ifndef __ONNX_MODEL_EXECUTION_INFO_HPP__
#define __ONNX_MODEL_EXECUTION_INFO_HPP__
#include <iostream>
#include <fstream>
#include "cuda_runtime.hpp"
#include "execution_info.hpp"

namespace TENSORRT_WRAPPER
{
    class OnnxModelExecutionInfo : public ExecutionInfo
    {
    public:
        OnnxModelExecutionInfo(CUDARuntime *runtime, 
            std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root);
        ~OnnxModelExecutionInfo();
        bool init(Json::Value& root) override;
        void run() override;
    private:
        Logger mLogger;
        nvinfer1::IRuntime* inferRuntime;
        nvinfer1::ICudaEngine* cudaEngine;
        nvinfer1::IExecutionContext* executionContext;
        std::vector<void*> engineBufferArray;
        int batchSize = 1;
    };
} // namespace TENSORRT_WRAPPER

#endif