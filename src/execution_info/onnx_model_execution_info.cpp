#include "onnx_model_execution_info.hpp"

namespace TENSORRT_WRAPPER
{
    OnnxModelExecutionInfo::OnnxModelExecutionInfo(CUDARuntime *runtime,
        std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root) : ExecutionInfo(runtime, tensorsInfo, root)
    {
    }
    
    OnnxModelExecutionInfo::~OnnxModelExecutionInfo()
    {
        if(executionContext != nullptr)
            executionContext->destroy();
        if(cudaEngine != nullptr)
            cudaEngine->destroy();
        if(inferRuntime != nullptr)
            inferRuntime->destroy();
        engineBufferArray.clear();
    }

    bool OnnxModelExecutionInfo::init(Json::Value& root)
    {
        auto engineFile = root["attr"]["onnx_file"].asString();
        char *trtModelStream = nullptr;
        size_t size = 0;
        std::ifstream file(engineFile.c_str(), std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            CHECK_ASSERT(trtModelStream, "malloc fail !\n");
            file.read(trtModelStream, size);
            file.close();
        }
        inferRuntime = nvinfer1::createInferRuntime(mLogger);
        CHECK_ASSERT(inferRuntime != nullptr, "create runtime fail!\n");
        cudaEngine = inferRuntime->deserializeCudaEngine(trtModelStream, size, nullptr);
        CHECK_ASSERT(cudaEngine != nullptr, "create engine fail!\n");
        executionContext = cudaEngine->createExecutionContext();
        CHECK_ASSERT(executionContext != nullptr, "create context fail!\n");
        delete[] trtModelStream;

        auto tensors = getTensorsInfo();
        const nvinfer1::ICudaEngine& engine = executionContext->getEngine();
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


    void OnnxModelExecutionInfo::run()
    {
        beforeRun();
        CUDARuntime *runtime = getCudaRuntime();
        auto engineStream = runtime->stream();
        executionContext->enqueue(batchSize, &engineBufferArray[0], engineStream, nullptr);
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch onnx tensorrt engine fail: %s\n", cudaGetErrorString(cudastatus));
        // {
        //     auto tensorsInfo = getTensorsInfo();
        //     printBuffer<float>(tensorsInfo["prefix/pred/global_head/l2_normalize:0"], 0, 10);
        //     printBuffer<float>(tensorsInfo["prefix/image:0"], 0, 10);
        // }
        afterRun();
        return;
    }
}