/********************************************
 * Filename: trt_engine_execution.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <iostream>
#include <fstream>
#include "execution/trt_engine_execution.hpp"

namespace TENSORRT_WRAPPER
{

    TrtEngineExecution::TrtEngineExecution(CUDARuntime* runtime, const Json::Value& root) : BaseExecution(runtime, root)
    {
    }

    TrtEngineExecution::~TrtEngineExecution()
    {
        if(m_execution_context != nullptr)
            m_execution_context->destroy();
        if(m_cuda_engine != nullptr)
            m_cuda_engine->destroy();
        if(m_infer_runtime != nullptr)
            m_infer_runtime->destroy();
        engineBufferArray.clear();
    }

    bool TrtEngineExecution::init(Json::Value& root)
    {
        auto engine_file = root["attr"]["engine_file"].asString();
        char* engine_buffer = nullptr;
        size_t file_size = 0;
        std::ifstream file(engine_file.c_str(), std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            file_size = file.tellg();
            file.seekg(0, file.beg);
            engine_buffer = new char[file_size];
            CHECK_ASSERT(engine_buffer, "malloc fail !\n");
            file.read(engine_buffer, file_size);
            file.close();
        }
        m_infer_runtime = nvinfer1::createInferRuntime(m_logger);
        CHECK_ASSERT(m_infer_runtime != nullptr, "create runtime fail!\n");
        m_cuda_engine = m_infer_runtime->deserializeCudaEngine(engine_buffer, file_size, nullptr);
        CHECK_ASSERT(m_cuda_engine != nullptr, "create engine fail!\n");
        m_execution_context = m_cuda_engine->createExecutionContext();
        CHECK_ASSERT(m_execution_context != nullptr, "create context fail!\n");
        delete[] engine_buffer;

        auto tensors = getTensorsInfo();
        const nvinfer1::ICudaEngine& engine = m_execution_context->getEngine();
        int nbBinds = engine.getNbBindings();
        for(int i = 0; i < nbBinds; i++)
        {
            auto name = engine.getBindingName(i);
            auto buffer = tensors[name]->device<void>();
            engineBufferArray.push_back(buffer);
        }
        recycleBuffers();
        return true;
    }


    void TrtEngineExecution::run()
    {
        beforeRun();
        CUDARuntime *runtime = getCudaRuntime();
        auto engineStream = runtime->stream();
        m_execution_context->enqueue(batchSize, &engineBufferArray[0], engineStream, nullptr);
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch onnx tensorrt engine fail: %s\n", cudaGetErrorString(cudastatus));
        afterRun();
        return;
    }

    class TrtEngineExecutionCreator : public ExecutionCreator
    {
    public:
        virtual BaseNode* onCreate(const NodeBaseConfig* config) const override 
        {
            return new TrtEngineExecution(config);
        }
    };

    void registerTrtEngineExecutionCreator()
    {
        insertExecutionCreator(TRT_ENGINE_EXECUTION_TYPE, new TrtEngineExecutionCreator);
    }

} // namespace TENSORRT_WRAPPER