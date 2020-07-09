#ifndef __TENSORRT_ENGINE_HPP__
#define __TENSORRT_ENGINE_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "utils.hpp"
#include "weights_graph_parse.hpp"
#include "cuda_runtime.hpp"
#include "execution.hpp"

using namespace nvinfer1;
using namespace std;

namespace tensorrtInference
{
    class tensorrtEngine
    {
    public:
        tensorrtEngine(std::string jsonFile, std::string weightsFile, bool fp16Flag = false);
        tensorrtEngine(std::string engineFile);
        ~tensorrtEngine();
        bool saveEnginePlanFile(std::string saveFile);
        void doInference(bool syncFlag);
        void createEngine(unsigned int maxBatchSize, bool fp16Flag);
        std::map<std::string, void*> getBindingNamesHostMemMap();
        std::map<std::string, int> getBindingNamesIndexMap();
        void prepareData(std::map<int, unsigned char*> dataMap);
        void prepareData(std::map<int, unsigned char*> dataMap, std::map<int, std::vector<int>> dataShape, 
            std::vector<std::string> preExecution, std::vector<std::string> postExecution);
        std::map<std::string, void*> getInferenceResult();
        std::vector<Buffer*> getPreProcessResult();
        std::vector<Buffer*> getPostProcessResult();
    private:
        void initConstTensors(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void setNetInput(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        std::vector<int> getBindingByteCount();
        bool mallocEngineMem();

        std::vector<void*> getEngineBufferArray();

        Logger mLogger;
        std::shared_ptr<tensorrtInference::weightsAndGraphParse> weightsAndGraph;
        nvinfer1::IRuntime* runtime;
        nvinfer1::IBuilder* builder;
        nvinfer1::ICudaEngine* cudaEngine;
        nvinfer1::IExecutionContext* context;
        //malloc before inference
        std::map<int, void*> hostMemMap;
        std::map<int, void*> deviceMemMap;
        std::map<int, void*> deviceFp16MemMap;
        //cuda stream
        bool inferenceFlag = false;
        cudaStream_t engineStream;

        //gpu runtime
        std::shared_ptr<CUDARuntime> cudaRuntime;
        std::map<int, shared_ptr<Buffer>> hostNetworkInputBuffers;
        std::map<int, shared_ptr<Buffer>> deviceNetWorkOutputBuffers;
        std::vector<std::shared_ptr<Execution>> preProcessExecution;
        std::vector<std::shared_ptr<Execution>> postProcessExecution;
    };
}

#endif