/********************************************
 * Filename: trt_engine_execution.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <vector>
#include "cuda_runtime.hpp"
#include "execution/trt_engine_execution.hpp"

namespace TENSORRT_WRAPPER
{

    class TrtEngineExecution : public BaseExecution
    {
    public:
        TrtEngineExecution(CUDARuntime* runtime, Json::Value& root);
        ~TrtEngineExecution();
        bool init(Json::Value& root) override;
        void run() override;

    private:
        Logger                           m_logger;
        nvinfer1::IRuntime*              m_infer_runtime = nullptr;
        nvinfer1::ICudaEngine*           m_cuda_engine = nullptr;
        nvinfer1::IExecutionContext*     m_execution_context = nullptr;
        std::vector<void*>               engineBufferArray;
        int                              batchSize = 1;
    };

} // namespace TENSORRT_WRAPPER